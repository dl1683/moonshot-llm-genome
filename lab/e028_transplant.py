"""E028 — Organ transplants across anatomies: within- vs cross-anatomy swap damage.

Implements scratch/e028_transplant_design.md exactly; answers T006 PL3 and the
registered prediction P3 ("cross-anatomy swaps cost >= 2x same-anatomy swap
damage").

Arms (host <- donor), sites L0/L2/L3/L5, kinds attn/mlp:
  C0 self-transplant  B<-B, R<-R at {attn,mlp}-L0: must be EXACTLY zero change
     (bitwise-identical state_dict + per-batch CE identity); abort otherwise.
  C1 within-anatomy   B <- B43 (fresh baseline, seed 43, trained in-run;
                     ckpt runs/checkpoints/e028_b43.pt, 252 s cap, resumable).
  X  cross-anatomy    B <- R (bare) and R <- B (register_renorm hooks at eval;
                     R's ckpt was saved WITHOUT hooks — every R eval re-pins).
  C2 lottery          R <- seed-99 fresh-init organ at {attn,mlp} x {L0,L5}.
  layer-pair          attn+mlp together at L0/L5, cross, both directions.
  whole-prefix        blocks 0..k INCLUDING block LNs (wte/wpe/ln_f/lm_head
                     always host-native), k in {1,3}, cross, both directions.

Eval: 30 fixed deterministic val batches (per-batch losses kept; 2000-resample
bootstrap). R = dCE_transplant / dCE_ablate(host,site,kind) with ablation
references reused from runs/e001 + runs/e014b (2 spot-recomputed in-run).
R bands: <0.5 strong compatibility; 0.5-0.9 partial; 0.9-1.1 inert; >1.1
active interference. On the B host also R/R_rand vs E011b's random rung.

P3 (B host): rho = dCE_cross/dCE_within per cell. CONFIRMED iff median rho >= 2
AND rho >= 2 in >= 6/8 cells AND bootstrap CI of dCE_cross - 2*dCE_within
excludes 0 in >= 6/8 cells (noise-floor cells with dCE_within < 0.05 excluded
from the median and listed separately with the additive statement
dCE_cross >= dCE_within + 0.30). REFUTED iff median rho <= 1.2 AND
dCE_cross <= dCE_within in >= 4/8 cells. Partial otherwise.

Conservative choices where the memo left slack (noted per task instructions):
- B43 is a DONOR only (memo arms table: C1 = B <- B43, 8 evals). Symmetric
  B43-as-host cells do not fit the 15-min budget and were not run.
- Layer-pair / prefix R denominators = SUM of the host's per-organ ablation
  damages (joint block ablation was never measured in e001/e014b).
- Memo table counts 11 keys per prefix block; its own pseudocode block_keys()
  gives 10 (ln1 2 + attn 2 + ln2 2 + mlp 4). Pseudocode followed.
- R_rand (for R/R_rand on the B host) reused from runs/e011b (20-batch rung).
- S2 needs single-block cross damages at L0..L3; measured only if slack
  remains (pair extension after all memo blocks), else a proxy sum over the
  measured organs of blocks 0/2/3 is reported and flagged as a proxy.
- B43 "divergence" fallback retrain (memo §6) triggered at val > 1.75.
- Memo O2 (LN co-transplant variant) optional — skipped (budget).

Run: python lab/e028_transplant.py
"""
from __future__ import annotations

import json
import random
import statistics
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, cfg_dict, lesion_loss,
                    now_iso, run_dir, save_json, set_seed, train_model)

E001_CKPT = REPO / "runs" / "checkpoints" / "e001.pt"
R_CKPT = REPO / "runs" / "checkpoints" / "e014b.pt"
B43_CKPT = REPO / "runs" / "checkpoints" / "e028_b43.pt"
B43_TRAIN = REPO / "runs" / "checkpoints" / "e028_b43.train.pt"
C_RENORM = 5.6
SITES = (0, 2, 3, 5)
KINDS = ("attn", "mlp")
N_EVAL = 30
N_BOOT = 2000
B43_VAL_GATE = 1.7224          # e001 val 1.6224 + 0.10 (memo §2)
B43_DIVERGED = 1.75            # divergence threshold for the lr-6e-4 fallback
T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():6.1f}s] {msg}", flush=True)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


# ------------------------------------------------ surgery units (memo §7)
def mlp_keys(i: int) -> list[str]:
    return [f"h.{i}.mlp.0.weight", f"h.{i}.mlp.0.bias",
            f"h.{i}.mlp.2.weight", f"h.{i}.mlp.2.bias"]


def attn_keys(i: int) -> list[str]:
    return [f"h.{i}.attn.c_attn.weight", f"h.{i}.attn.c_proj.weight"]


def block_keys(i: int) -> list[str]:
    return ([f"h.{i}.ln1.weight", f"h.{i}.ln1.bias"] + attn_keys(i)
            + [f"h.{i}.ln2.weight", f"h.{i}.ln2.bias"] + mlp_keys(i))


def organ_keys(site: int, kind: str) -> list[str]:
    """The SURGERY unit — LNs stay with the host (memo LN decision)."""
    return mlp_keys(site) if kind == "mlp" else attn_keys(site)


# ------------------------------------------------ renorm hooks (copied from lab/e014b_stream_renorm.py)
def register_renorm(model, c=C_RENORM):
    hooks = []
    for block in model.h:
        def pre(m, args, _c=c):
            x = args[0]
            n = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            return (x * (_c / n),)
        hooks.append(block.register_forward_pre_hook(pre))
    return hooks


# ------------------------------------------------ snapshot / transplant
def snapshot(model) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def restore(model, snap: dict) -> None:
    model.load_state_dict(snap, strict=True)


def assert_identical(model, snap: dict, name: str) -> None:
    now = model.state_dict()
    bad = [k for k in snap if not torch.equal(now[k], snap[k])]
    if bad:
        raise SystemExit(f"HOST {name} NOT BITWISE-RESTORED after a cell: {bad}")


def transplant(model, donor_sd: dict, keys: list[str]) -> None:
    """Graft donor organ(s) into host IN PLACE. One-way copy; donor untouched."""
    sd = model.state_dict()
    for k in keys:
        assert sd[k].shape == donor_sd[k].shape, k
        sd[k] = donor_sd[k].detach().clone().to(sd[k].dtype)
    model.load_state_dict(sd, strict=True)


# ------------------------------------------------ deterministic 30-batch eval
@torch.no_grad()
def per_batch_losses(model, corpus, n_batches: int = N_EVAL) -> list[float]:
    """Exact replica of common.estimate_loss(model, corpus, 'val', n_batches)
    that returns per-batch losses (same generator discipline => same batches)."""
    model.eval()
    cfg = model.cfg
    gen = torch.Generator().manual_seed(corpus.seed)
    src = corpus.val
    losses = []
    for _ in range(n_batches):
        ix = torch.randint(len(src) - cfg.block_size - 1, (16,), generator=gen)
        x = torch.stack([src[i: i + cfg.block_size] for i in ix]).to(DEVICE)
        y = torch.stack([src[i + 1: i + 1 + cfg.block_size] for i in ix]).to(DEVICE)
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return losses


def eval_cell(host, corpus, donor_sd, keys: list[str], renorm: bool,
              base_losses: list[float], self_transplant: bool = False) -> list[float]:
    """Graft -> eval on the fixed 30 batches -> restore host bitwise."""
    snap = snapshot(host)
    donor = snap if self_transplant else donor_sd          # C0: own params through surgery
    transplant(host, donor, keys)
    if self_transplant:
        now = host.state_dict()
        bad = [k for k in snap if not torch.equal(now[k], snap[k])]
        if bad:
            raise SystemExit(f"C0 GATE FAILURE: self-transplant not bitwise ({len(bad)} keys, "
                             f"e.g. {bad[:3]}). Surgery helper broken — do not interpret anything.")
    hs = register_renorm(host) if renorm else []
    losses = per_batch_losses(host, corpus)
    for h in hs:
        h.remove()
    restore(host, snap)
    return losses


# ------------------------------------------------ bootstrap (E011c discipline)
def bootstrap_ci(diffs: list[float], n: int = N_BOOT, seed: int = 2028):
    rng = random.Random(seed)
    nb = len(diffs)
    vals = []
    for _ in range(n):
        idx = [rng.randrange(nb) for _ in range(nb)]
        vals.append(sum(diffs[i] for i in idx) / nb)
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def bootstrap_contrast(d_cross: list[float], d_within: list[float], n: int = N_BOOT,
                       seed: int = 2029):
    """Paired CI of mean(d_cross) - 2*mean(d_within)."""
    rng = random.Random(seed)
    nb = len(d_cross)
    vals = []
    for _ in range(n):
        idx = [rng.randrange(nb) for _ in range(nb)]
        mc = sum(d_cross[i] for i in idx) / nb
        mw = sum(d_within[i] for i in idx) / nb
        vals.append(mc - 2 * mw)
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def bootstrap_ratio(d_cross: list[float], d_within: list[float], n: int = N_BOOT,
                    seed: int = 2030):
    """Paired CI of mean(d_cross)/mean(d_within) (invalid resamples dropped)."""
    rng = random.Random(seed)
    nb = len(d_cross)
    vals = []
    for _ in range(n):
        idx = [rng.randrange(nb) for _ in range(nb)]
        mw = sum(d_within[i] for i in idx) / nb
        if mw > 1e-9:
            vals.append(sum(d_cross[i] for i in idx) / nb / mw)
    if not vals:
        return None, None
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


# ------------------------------------------------ renorm liveness assert (memo §3)
@torch.no_grad()
def renorm_liveness_assert(model, corpus) -> list[float]:
    hs = register_renorm(model)
    probes, handles = [], []
    for block in model.h:
        def mk():
            def pre(m, args):
                probes.append(float(args[0].norm(dim=-1).mean()))
                return None
            return pre
        handles.append(block.register_forward_pre_hook(mk()))  # AFTER renorm hooks
    x, _ = corpus.get_batch("val", model.cfg.block_size, 16,
                             gen=torch.Generator().manual_seed(7))
    model(x)
    for h in handles:
        h.remove()
    for h in hs:
        h.remove()
    if len(probes) != model.cfg.n_layer or not all(abs(n - C_RENORM) <= 1e-3 for n in probes):
        raise SystemExit(f"RENORM LIVENESS ASSERT FAILED: block-input norms {probes} "
                         f"!= {C_RENORM} +- 1e-3. R-host evals would be off-manifold.")
    return probes


# ---------------------------------------------------------------- main
def main():
    rd = run_dir("e028")
    log("E028 organ transplants — loading corpus + checkpoints + references")
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)

    e1 = json.loads((REPO / "runs" / "e001" / "metrics.json").read_text(encoding="utf-8"))
    e14 = json.loads((REPO / "runs" / "e014b" / "metrics.json").read_text(encoding="utf-8"))
    e11 = json.loads((REPO / "runs" / "e011b" / "metrics.json").read_text(encoding="utf-8"))
    abl = {"B": {"attn": e1["attn_block_damage"], "mlp": e1["mlp_block_damage"]},
           "R": {"attn": e14["renorm_damage"]["attn"], "mlp": e14["renorm_damage"]["mlp"]}}
    r_rand = {"attn": e11["orthogonal_vs_zero"]["same_norm_random"]["attn"],
              "mlp": e11["orthogonal_vs_zero"]["same_norm_random"]["mlp"]}

    B = TinyGPT(cfg).to(DEVICE)
    B.load_state_dict(torch.load(E001_CKPT, map_location=DEVICE, weights_only=True))
    R = TinyGPT(cfg).to(DEVICE)
    R.load_state_dict(torch.load(R_CKPT, map_location=DEVICE, weights_only=True))

    # ---- base CEs (recomputed in-run) + renorm liveness ----
    base_B = per_batch_losses(B, corpus)
    log(f"B base CE {mean(base_B):.6f} (expect 1.622391)")
    if abs(mean(base_B) - 1.622391394774119) > 0.02:
        raise SystemExit(f"B base CE {mean(base_B):.4f} far from e001 reference — wrong checkpoint?")
    live = renorm_liveness_assert(R, corpus)
    hs = register_renorm(R)
    base_R = per_batch_losses(R, corpus)
    for h in hs:
        h.remove()
    log(f"R base CE (hooks) {mean(base_R):.6f} (expect 1.609990); liveness {live}")

    # ---- C0 self-transplant gate (before any other cell; memo §8) ----
    c0 = {}
    for host, name, base, renorm in ((B, "B", base_B, False), (R, "R", base_R, True)):
        for kind in KINDS:
            losses = eval_cell(host, corpus, None, organ_keys(0, kind), renorm, base,
                               self_transplant=True)
            dce = mean(losses) - mean(base)
            exact = losses == base                       # per-batch float identity
            c0[f"{name}_{kind}_L0"] = {"dce": dce, "ce_exactly_zero": bool(exact),
                                       "bitwise_state_identical": True}
            if not exact or dce != 0.0:
                raise SystemExit(f"C0 GATE FAILURE: {name} {kind}-L0 self-transplant "
                                 f"changed CE by {dce!r} (exact per-batch match: {exact}). "
                                 "Surgery path broken — aborting, do not interpret anything.")
            log(f"C0 {name} {kind}-L0: bitwise-identical state, dCE exactly 0.0")

    # ---- spot-check 2 ablation references (memo §3) ----
    spot_b = lesion_loss(B, corpus, "mlp", 0, n_batches=N_EVAL) - mean(base_B)
    hs = register_renorm(R)
    spot_r = lesion_loss(R, corpus, "attn", 2, n_batches=N_EVAL) - mean(base_R)
    for h in hs:
        h.remove()
    spot_pass = (abs(spot_b - abl["B"]["mlp"][0]) < 0.05
                 and abs(spot_r - abl["R"]["attn"][2]) < 0.05)
    log(f"spot-check: B mlp-L0 {spot_b:+.4f} vs ref {abl['B']['mlp'][0]:+.4f} | "
        f"R attn-L2 {spot_r:+.4f} vs ref {abl['R']['attn'][2]:+.4f} "
        f"({'PASS' if spot_pass else 'MISMATCH'})")

    # ---- B43: fresh baseline seed 43 (C1 donor; resumable, 252 s cap) ----
    b43_retrained = False
    if B43_CKPT.exists():
        b43 = TinyGPT(cfg).to(DEVICE)
        b43.load_state_dict(torch.load(B43_CKPT, map_location=DEVICE, weights_only=True))
        b43_steps = None
        log(f"B43 found at {B43_CKPT.name} — skipping training")
    else:
        set_seed(43)
        b43 = TinyGPT(cfg).to(DEVICE)
        log("training B43 (seed 43 baseline, steps 4000, lr 1e-3, bs 64, cap 252 s)")
        hist = train_model(b43, corpus, steps=4000, lr=1e-3, batch_size=64,
                           max_seconds=252.0, ckpt=B43_TRAIN)
        torch.save(b43.state_dict(), B43_CKPT)
        b43_steps = hist[-1]["step"] if hist else 0
    base_B43 = per_batch_losses(b43, corpus)
    b43_val = mean(base_B43)
    b43_gate = b43_val <= B43_VAL_GATE
    log(f"B43 val {b43_val:.4f} (gate <= {B43_VAL_GATE}: {'PASS' if b43_gate else 'FLAG — still usable'})")
    if b43_val > B43_DIVERGED and elapsed() < 480 and B43_CKPT.exists():
        log("B43 diverged -> one retrain at lr 6e-4 (memo §6 fallback ladder)")
        B43_TRAIN.unlink(missing_ok=True)
        set_seed(43)
        b43 = TinyGPT(cfg).to(DEVICE)
        hist = train_model(b43, corpus, steps=4000, lr=6e-4, batch_size=64,
                           max_seconds=252.0, ckpt=B43_TRAIN)
        torch.save(b43.state_dict(), B43_CKPT)
        b43_steps = hist[-1]["step"] if hist else 0
        base_B43 = per_batch_losses(b43, corpus)
        b43_val = mean(base_B43)
        b43_gate = b43_val <= B43_VAL_GATE
        b43_retrained = True
        log(f"B43(retrain) val {b43_val:.4f} (gate: {'PASS' if b43_gate else 'FLAG'})")

    B_sd, R_sd, b43_sd = snapshot(B), snapshot(R), snapshot(b43)
    set_seed(99)
    lottery_sd = snapshot(TinyGPT(cfg).to(DEVICE))     # random tissue, std-0.02 init

    CELLS: dict[str, dict] = {}

    def run_cell(label: str, host, base: list[float], donor_sd: dict, keys: list[str],
                 renorm: bool, abl_val: float | None = None, r_rand_val: float | None = None):
        losses = eval_cell(host, corpus, donor_sd, keys, renorm, base)
        diffs = [a - b for a, b in zip(losses, base)]
        dce = mean(diffs)
        lo, hi = bootstrap_ci(diffs)
        rec = {"dce": dce, "ci95": [lo, hi], "per_batch_dce": diffs}
        if abl_val:
            rec["ablate_ref"] = abl_val
            rec["r"] = dce / abl_val
        if r_rand_val:
            rec["r_rand_ref"] = r_rand_val
            rec["r_over_rand"] = dce / r_rand_val
        CELLS[label] = rec
        log(f"  {label:24s} dCE {dce:+.4f} (CI {lo:+.3f},{hi:+.3f})"
            + (f"  R {rec['r']:+.2f}" if "r" in rec else ""))
        return rec

    # ---- P3 core: B host, within (B<-B43) + cross (B<-R) ----
    log("P3 core on B host: 8 within (B<-B43) + 8 cross (B<-R) cells")
    first_B_cell = True
    for site in SITES:
        for kind in KINDS:
            run_cell(f"within|L{site}|{kind}", B, base_B, b43_sd,
                     organ_keys(site, kind), False, abl_val=abl["B"][kind][site])
            run_cell(f"cross_B|L{site}|{kind}", B, base_B, R_sd,
                     organ_keys(site, kind), False, abl_val=abl["B"][kind][site],
                     r_rand_val=r_rand[kind][site])
            if first_B_cell:
                assert_identical(B, B_sd, "B")   # memo note 1: assert once
                first_B_cell = False

    # ---- X cross on R host (hooks active) ----
    log("X cross on R host (B<-donor -> R host with renorm hooks)")
    x_r_sites = []
    first_R_cell = True
    for site in (0, 5, 2, 3):                     # L0/L5 first (stop-rule §5)
        if site in (2, 3) and elapsed() > 900:
            log(f"stop-rule: dropping X_R L{site} (wall clock > 15 min)")
            continue
        x_r_sites.append(site)
        for kind in KINDS:
            run_cell(f"cross_R|L{site}|{kind}", R, base_R, B_sd,
                     organ_keys(site, kind), True, abl_val=abl["R"][kind][site])
            if first_R_cell:
                assert_identical(R, R_sd, "R")
                first_R_cell = False

    # ---- optional blocks (drop order: prefix, layer-pair, C2; memo §5) ----
    run_lottery = elapsed() <= 720
    if run_lottery:
        log("C2 lottery on R host (R <- seed-99 fresh init) at attn/mlp x L0/L5")
        for site in (0, 5):
            for kind in KINDS:
                run_cell(f"lottery|L{site}|{kind}", R, base_R, lottery_sd,
                         organ_keys(site, kind), True, abl_val=abl["R"][kind][site])
    else:
        log(f"stop-rule: C2 lottery dropped (elapsed {elapsed():.0f}s > 720 s)")

    run_pair = elapsed() <= 720
    if run_pair:
        log("layer-pair (attn+mlp together) cross, L0/L5, both directions")
        for s in (0, 5):
            keys = attn_keys(s) + mlp_keys(s)
            run_cell(f"pair_cross_B|L{s}", B, base_B, R_sd, keys, False,
                     abl_val=abl["B"]["attn"][s] + abl["B"]["mlp"][s])
            run_cell(f"pair_cross_R|L{s}", R, base_R, B_sd, keys, True,
                     abl_val=abl["R"]["attn"][s] + abl["R"]["mlp"][s])
    else:
        log(f"stop-rule: layer-pair dropped (elapsed {elapsed():.0f}s > 720 s)")

    run_prefix = elapsed() <= 720
    if run_prefix:
        log("whole-prefix 0..k (incl. block LNs) cross, k in {1,3}, both directions")
        for k in (1, 3):
            keys = [key for j in range(k + 1) for key in block_keys(j)]
            ablB = sum(abl["B"]["attn"][j] + abl["B"]["mlp"][j] for j in range(k + 1))
            ablR = sum(abl["R"]["attn"][j] + abl["R"]["mlp"][j] for j in range(k + 1))
            run_cell(f"prefix{k}_cross_B", B, base_B, R_sd, keys, False, abl_val=ablB)
            run_cell(f"prefix{k}_cross_R", R, base_R, B_sd, keys, True, abl_val=ablR)
    else:
        log(f"stop-rule: prefix dropped (elapsed {elapsed():.0f}s > 720 s)")

    # S2 enabler: single-block cross damages at L1-L3 (beyond-memo slack use)
    if elapsed() <= 780 and run_pair:
        log("slack: pair extension L1-L3 (enables registered S2)")
        for s in (1, 2, 3):
            if elapsed() > 810:
                break
            keys = attn_keys(s) + mlp_keys(s)
            run_cell(f"pair_cross_B|L{s}", B, base_B, R_sd, keys, False,
                     abl_val=abl["B"]["attn"][s] + abl["B"]["mlp"][s])
            run_cell(f"pair_cross_R|L{s}", R, base_R, B_sd, keys, True,
                     abl_val=abl["R"]["attn"][s] + abl["R"]["mlp"][s])

    # ------------------------------------------------ P3 verdict (memo §4)
    p3_cells = []
    for site in SITES:
        for kind in KINDS:
            w, c = CELLS[f"within|L{site}|{kind}"], CELLS[f"cross_B|L{site}|{kind}"]
            dw, dc = w["dce"], c["dce"]
            lo, hi = bootstrap_contrast(c["per_batch_dce"], w["per_batch_dce"])
            ci_excl0 = bool(lo > 0 or hi < 0)
            rlo, rhi = bootstrap_ratio(c["per_batch_dce"], w["per_batch_dce"])
            noise_floor = dw < 0.05
            rho = dc / dw if dw > 1e-12 else None
            p3_cells.append({"site": site, "kind": kind, "dce_within": dw, "dce_cross": dc,
                             "rho": rho, "rho_ci95": [rlo, rhi],
                             "contrast_ci95": [lo, hi], "ci_excludes_0": ci_excl0,
                             "noise_floor": bool(noise_floor),
                             "additive_ge_030": bool(dc >= dw + 0.30)})
    eligible = [c for c in p3_cells if not c["noise_floor"]]
    excluded = [c for c in p3_cells if c["noise_floor"]]
    median_rho = statistics.median([c["rho"] for c in eligible]) if eligible else None
    n_rho_ge2 = sum(1 for c in eligible if c["rho"] is not None and c["rho"] >= 2)
    n_ci0 = sum(1 for c in eligible if c["ci_excludes_0"])
    n_le_all8 = sum(1 for c in p3_cells if c["dce_cross"] <= c["dce_within"])
    p3_confirmed = bool(median_rho is not None and median_rho >= 2
                        and n_rho_ge2 >= 6 and n_ci0 >= 6)
    p3_refuted = bool(median_rho is not None and median_rho <= 1.2 and n_le_all8 >= 4)
    verdict = "CONFIRMED" if p3_confirmed else ("REFUTED" if p3_refuted else "PARTIAL")
    p3_median_only = bool(median_rho is not None and median_rho >= 2)

    # ------------------------------------------------ S1/S2/S3 (memo §4)
    s1 = {
        "B_mlp0_to_R": {"cell": "cross_R|L0|mlp", "r": CELLS["cross_R|L0|mlp"]["r"],
                        "dce": CELLS["cross_R|L0|mlp"]["dce"],
                        "pred_r_ge_1": bool(CELLS["cross_R|L0|mlp"]["r"] >= 1)},
        "R_mlp0_to_B": {"cell": "cross_B|L0|mlp", "r": CELLS["cross_B|L0|mlp"]["r"],
                        "dce": CELLS["cross_B|L0|mlp"]["dce"],
                        "pred_r_near_1": bool(0.9 <= CELLS["cross_B|L0|mlp"]["r"] <= 1.1),
                        "pred_dce_near_4.08": bool(abs(CELLS["cross_B|L0|mlp"]["dce"] - 4.079) <= 0.5)},
        "note": "both R~1 => transplants fail by silence, not violence",
    }
    s1["both_r_near_1"] = bool(0.9 <= s1["B_mlp0_to_R"]["r"] <= 1.1
                               and 0.9 <= s1["R_mlp0_to_B"]["r"] <= 1.1)

    s2 = {}
    for hn in ("B", "R"):
        pref = CELLS.get(f"prefix3_cross_{hn}")
        if pref is None:
            continue
        pairs = [CELLS.get(f"pair_cross_{hn}|L{s}") for s in range(4)]
        if all(p is not None for p in pairs):
            rhs = sum(p["dce"] for p in pairs)
            s2[hn] = {"prefix_dce": pref["dce"], "sum_single_block_cross": rhs,
                      "mode": "registered (pairs L0-L3)", "superadditive": bool(pref["dce"] > rhs)}
        else:
            rhs = sum(CELLS[f"cross_{hn}|L{s}|{k}"]["dce"] for s in (0, 2, 3) for k in KINDS)
            s2[hn] = {"prefix_dce": pref["dce"], "sum_organ_cross_proxy": rhs,
                      "mode": "PROXY (block-1 organs unmeasured; RHS underestimated)",
                      "superadditive_proxy": bool(pref["dce"] > rhs)}

    s3_cells = [(s, k) for s in (0, 5) for k in KINDS
                if f"lottery|L{s}|{k}" in CELLS and f"cross_R|L{s}|{k}" in CELLS]
    s3_hits = sum(1 for s, k in s3_cells
                  if CELLS[f"lottery|L{s}|{k}"]["r"] >= CELLS[f"cross_R|L{s}|{k}"]["r"])
    s3 = {"n_cells": len(s3_cells), "hits_lottery_ge_cross": s3_hits,
          "pass_ge_3_of_4": bool(len(s3_cells) == 4 and s3_hits >= 3)}

    # ------------------------------------------------ outputs
    log("writing metrics + figures")
    metrics = {
        "experiment": "e028_transplant",
        "date": now_iso(),
        "device": DEVICE,
        "config": cfg_dict(cfg),
        "sites": list(SITES), "kinds": list(KINDS),
        "eval_protocol": {"n_batches": N_EVAL, "batch_size": 16, "split": "val",
                          "deterministic": "generator seeded from corpus.seed (1337); "
                                           "identical batches for every arm"},
        "base_ce": {"B": mean(base_B), "R_hooks": mean(base_R), "B43": b43_val},
        "b43": {"val": b43_val, "steps": b43_steps, "val_gate": B43_VAL_GATE,
                "val_gate_pass": bool(b43_gate), "retrained_lr6e4": b43_retrained,
                "ckpt": str(B43_CKPT.name)},
        "ablation_refs": {"B": abl["B"], "R": abl["R"],
                          "r_rand_e011b": r_rand,
                          "source": ["runs/e001/metrics.json", "runs/e014b/metrics.json",
                                     "runs/e011b/metrics.json"]},
        "spot_checks": {"B_mlp_L0": {"value": spot_b, "ref": abl["B"]["mlp"][0]},
                        "R_attn_L2": {"value": spot_r, "ref": abl["R"]["attn"][2]},
                        "pass": bool(spot_pass)},
        "renorm_liveness": {"block_input_norms": live, "pass": True},
        "c0_self_transplant": c0 | {"gate_pass": True,
                                    "note": "bitwise state + exactly-zero dCE on all 4 cells"},
        "cells": CELLS,
        "p3": {"rule": "CONFIRMED iff median rho>=2 AND rho>=2 in >=6/8 AND contrast CI "
                       "(dCE_cross - 2*dCE_within) excludes 0 in >=6/8; noise-floor cells "
                       "(dCE_within<0.05) excluded from median",
               "per_cell": p3_cells, "median_rho": median_rho,
               "n_eligible": len(eligible), "n_rho_ge_2": n_rho_ge2,
               "n_ci_excludes_0": n_ci0, "n_cross_le_within_all8": n_le_all8,
               "noise_floor_cells": [{"site": c["site"], "kind": c["kind"],
                                      "dce_within": c["dce_within"],
                                      "dce_cross": c["dce_cross"],
                                      "additive_ge_030": c["additive_ge_030"]} for c in excluded],
               "p3_confirmed": p3_confirmed, "p3_refuted": p3_refuted, "verdict": verdict,
               "verdict_median_rule_only": p3_median_only},
        "s1_keystone_asymmetry": s1,
        "s2_prefix_superadditivity": s2,
        "s3_lottery_placement": s3,
        "blocks_run": {"x_r_sites": x_r_sites, "lottery": run_lottery,
                       "layer_pair": run_pair, "prefix": run_prefix,
                       "pair_extension_L1_L3": bool(
                           all(f"pair_cross_B|L{s}" in CELLS for s in (1, 2, 3)))},
        "timing_s": {"total": round(elapsed(), 1)},
    }
    save_json(rd / "metrics.json", metrics)

    # figure 1: R values grouped by site/kind, within vs cross (+ rho + R host)
    labels = [f"L{s}|{k}" for s in SITES for k in KINDS]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    xs = np.arange(len(labels))
    w_r = [CELLS[f"within|{l}"].get("r", np.nan) for l in labels]
    c_r = [CELLS[f"cross_B|{l}"].get("r", np.nan) for l in labels]
    axes[0].bar(xs - 0.2, w_r, 0.4, label="within B<-B43", color="#8a8a8a")
    axes[0].bar(xs + 0.2, c_r, 0.4, label="cross B<-R", color="#c0392b")
    axes[0].axhline(1.0, color="k", ls="--", lw=1)
    for x, v in zip(xs - 0.2, w_r):
        axes[0].text(x, v + 0.05 * np.sign(v or 1), f"{v:.2f}", ha="center", fontsize=6)
    for x, v in zip(xs + 0.2, c_r):
        axes[0].text(x, v + 0.05 * np.sign(v or 1), f"{v:.2f}", ha="center", fontsize=6)
    axes[0].set_xticks(xs, labels, fontsize=8)
    axes[0].set_ylabel("R = dCE_transplant / dCE_ablate")
    axes[0].set_title("B host: within vs cross-anatomy R (dashed = R 1, ablation-equivalent)")
    axes[0].legend(fontsize=8)

    rhos = [c["rho"] if c["rho"] is not None else np.nan for c in p3_cells]
    colors = ["#c0392b" if (c["rho"] is not None and c["rho"] >= 2) else "#7f8c8d"
              for c in p3_cells]
    bars = axes[1].bar(xs, rhos, color=colors)
    for c, b in zip(p3_cells, bars):
        if c["noise_floor"]:
            b.set_hatch("//")
    axes[1].axhline(2.0, color="k", ls="--", lw=1.2, label="P3 threshold (2x)")
    axes[1].axhline(1.0, color="k", ls=":", lw=1)
    for x, v in zip(xs, rhos):
        axes[1].text(x, v + 0.08, f"{v:.2f}", ha="center", fontsize=7)
    axes[1].set_xticks(xs, labels, fontsize=8)
    axes[1].set_ylabel("rho = dCE_cross / dCE_within")
    axes[1].set_title(f"P3 (B host): median rho {median_rho:.2f} -> {verdict} "
                      "(hatched = noise-floor cell)")
    axes[1].legend(fontsize=8)

    r_labels = [f"L{s}|{k}" for s in (0, 5, 2, 3) for k in KINDS]
    cr = [CELLS[f"cross_R|{l}"]["r"] if f"cross_R|{l}" in CELLS else np.nan
          for l in r_labels]
    lr_ = [CELLS[f"lottery|{l}"]["r"] if f"lottery|{l}" in CELLS else np.nan
           for l in r_labels]
    xr = np.arange(len(r_labels))
    axes[2].bar(xr - 0.2, cr, 0.4, label="cross R<-B", color="#2c6fbb")
    axes[2].bar(xr + 0.2, lr_, 0.4, label="lottery R<-init99", color="#b8860b")
    axes[2].axhline(1.0, color="k", ls="--", lw=1)
    axes[2].set_xticks(xr, r_labels, fontsize=8)
    axes[2].set_ylabel("R (vs R-host ablation ref)")
    axes[2].set_title("R host (renorm hooks active): direction-2 cross + lottery")
    axes[2].legend(fontsize=8)
    fig.suptitle("E028 — organ transplants across anatomies (T006 PL3 / P3); "
                 f"verdict: {verdict}, median rho {median_rho:.2f}")
    fig.tight_layout()
    fig.savefig(rd / "transplant.png", dpi=140)
    plt.close(fig)

    # figure 2: R heatmaps per host/arm with the R=1 contour (memo §8)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
    panels = [("within  B<-B43", lambda s, k: CELLS.get(f"within|L{s}|{k}", {}).get("r")),
              ("cross  B<-R", lambda s, k: CELLS.get(f"cross_B|L{s}|{k}", {}).get("r")),
              ("cross  R<-B (hooks)", lambda s, k: CELLS.get(f"cross_R|L{s}|{k}", {}).get("r"))]
    grids = [np.array([[get(s, k) for s in SITES] for k in KINDS], dtype=float)
             for _, get in panels]
    finite = [v for g in grids for v in g.flat if np.isfinite(v)]
    vmin, vmax = (min(-0.5, min(finite)), max(1.5, max(finite)))
    for ax, (title, get), grid in zip(axes, panels, grids):
        im = ax.imshow(np.ma.masked_invalid(grid), cmap="coolwarm", aspect="auto",
                       vmin=vmin, vmax=vmax)
        for i in range(len(KINDS)):
            for j in range(len(SITES)):
                v = grid[i, j]
                ax.text(j, i, "n/a" if not np.isfinite(v) else f"{v:.2f}",
                        ha="center", va="center", fontsize=8)
        if np.nanmin(grid) < 1.0 < np.nanmax(grid):
            gx, gy = np.meshgrid(np.arange(len(SITES)), np.arange(len(KINDS)))
            ax.contour(gx, gy, grid, levels=[1.0], colors="k", linewidths=2)
        ax.set_xticks(range(len(SITES)), [f"L{s}" for s in SITES])
        ax.set_yticks(range(len(KINDS)), list(KINDS))
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, label="R")
    fig.suptitle("E028 R heatmap per host/arm (contour = R 1, ablation-equivalent)")
    fig.tight_layout()
    fig.savefig(rd / "r_heatmap.png", dpi=140)
    plt.close(fig)

    # ------------------------------------------------ headline prints
    log("=" * 78)
    log(f"P3 VERDICT: {verdict} — median rho {median_rho:.3f} over {len(eligible)} eligible "
        f"cells; rho>=2 in {n_rho_ge2}/{len(eligible)}; contrast-CI excludes 0 in "
        f"{n_ci0}/{len(eligible)}; cross<=within in {n_le_all8}/8 overall")
    if excluded:
        for c in excluded:
            log(f"  noise-floor cell L{c['site']} {c['kind']}: dCE_within "
                f"{c['dce_within']:.4f} < 0.05 (excluded); additive "
                f"dCE_cross >= dCE_within + 0.30: {c['additive_ge_030']}")
    log(f"S1 keystone: B.mlp0->R R={s1['B_mlp0_to_R']['r']:.2f} (pred >=1: "
        f"{s1['B_mlp0_to_R']['pred_r_ge_1']}) | R.mlp0->B R={s1['R_mlp0_to_B']['r']:.2f} "
        f"dCE={s1['R_mlp0_to_B']['dce']:.3f} (R~1: {s1['R_mlp0_to_B']['pred_r_near_1']})")
    for hn, v in s2.items():
        log(f"S2 prefix superadditivity ({hn}): {v}")
    log(f"S3 lottery: {s3}")
    log(f"outputs: {rd}")
    return metrics


if __name__ == "__main__":
    main()
