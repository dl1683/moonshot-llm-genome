"""E029 — Seed x regime transplant matrix: which axis owns organ compatibility?

Registered in THINKING.md (T006 P3 resolution, before e028 ran): "the clean 2x2
seed(42/43) x regime(base/renorm) transplant matrix; prediction: the seed axis
dominates the regime axis". e028 found organ compatibility follows seed/init
lineage, not training regime; e029 is the clean factorial that separates the
axes.

2x2 donor set (seed x regime), all four exist after step 1:
  B   seed42 base    runs/checkpoints/e001.pt
  B43 seed43 base    runs/checkpoints/e028_b43.pt
  R   seed42 renorm  runs/checkpoints/e014b.pt
  R43 seed43 renorm  runs/checkpoints/e029_r43.pt  (TRAINED IN-RUN: set_seed(43),
      register_renorm hooks active during training AND eval, train_model with
      resumable ckpt runs/checkpoints/e029_r43.train.pt, 252 s cap,
      parity gate val <= 1.7224)

Hosts: B and B43 (no renorm hooks); R host (hooks) budget-optional
(stop-rule: run iff elapsed <= 600 s when reached). Sites L0/L3/L5 x
{mlp, attn}; 30 fixed deterministic val batches per cell (exact e028
protocol); R = dCE_cell / dCE_ablate(host, site, kind) with ALL ablation
denominators recomputed in-run on the same 30 batches.

Cell roles per host (donor differs along ONE axis):
  seed-axis   host<-donor differing only in seed:   B<-B43  B43<-B   R<-R43
  regime-axis host<-donor differing only in regime: B<-R    B43<-R43 R<-B
  both-axes   (additivity probe):                   B<-R43  B43<-R   R<-B43

REGISTERED VERDICT (from THINKING.md): seed-dominance = median over per-cell
ratios dCE(seed-axis)/dCE(regime-axis) >= 1.5  ->  the seed axis dominates.
Cells with dCE(regime-axis) < 0.05 nats are noise-floor: excluded from the
median, listed separately (e028 discipline).

REGISTERED MECHANISM PREDICTION (this run): same-init host-donor pairs
(B<->R share seed42; B43<->R43 share seed43) have systematically higher
cos(dW_donor, dW_host), where dW = W_trained - W_init and W_init comes from
set_seed(seed); TinyGPT(cfg) untrained — "organs refine init-anchored
subspaces". Confirmed iff mean gap >= 0.05 AND same-init mean higher in
>= 5/6 organs.

Deviations / slack choices (task instructions require noting them here):
- Primary verdict median uses ALL executed cells (B+B43 hosts, plus R-host
  cells if the optional block ran); the B+B43-only median is also reported.
- Ablation refs are recomputed in-run for every host (paired 30-batch
  protocol) instead of reusing stored refs, so numerator and denominator see
  identical batches; spot-checks against runs/e001 + runs/e014b (tol 0.05).
- R43 divergence fallback: if val > 1.75 after lr 1e-3, ONE retrain at lr
  6e-4 (e028's ladder), budget permitting.
- dW cosines are computed per surgery organ (attn: c_attn+c_proj weights;
  mlp: fc weights+biases) as flattened float32 concatenations on device.
- No NOTES/THINKING/QUEUE/STATE edits, no git commit (per instructions).

Run: python lab/e029_seed_regime_matrix.py
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

E001_CKPT = REPO / "runs" / "checkpoints" / "e001.pt"        # B  (seed42 base)
R_CKPT = REPO / "runs" / "checkpoints" / "e014b.pt"          # R  (seed42 renorm)
B43_CKPT = REPO / "runs" / "checkpoints" / "e028_b43.pt"     # B43(seed43 base)
R43_CKPT = REPO / "runs" / "checkpoints" / "e029_r43.pt"     # R43(seed43 renorm, this run)
R43_TRAIN = REPO / "runs" / "checkpoints" / "e029_r43.train.pt"
C_RENORM = 5.6
SITES = (0, 3, 5)
KINDS = ("attn", "mlp")
N_EVAL = 30
N_BOOT = 2000
PARITY_GATE = 1.7224            # e001 val 1.6224 + 0.10
R43_DIVERGED = 1.75             # retrain-fallback threshold (e028 ladder)
NOISE_FLOOR = 0.05              # nats; regime-axis dCE below this = noise floor
SEED_DOMINANCE_THRESHOLD = 1.5
T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():6.1f}s] {msg}", flush=True)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


# ------------------------------------------------ surgery units (e028 §7, verbatim)
def mlp_keys(i: int) -> list[str]:
    return [f"h.{i}.mlp.0.weight", f"h.{i}.mlp.0.bias",
            f"h.{i}.mlp.2.weight", f"h.{i}.mlp.2.bias"]


def attn_keys(i: int) -> list[str]:
    return [f"h.{i}.attn.c_attn.weight", f"h.{i}.attn.c_proj.weight"]


def organ_keys(site: int, kind: str) -> list[str]:
    """The SURGERY unit — LNs stay with the host (e028 LN decision)."""
    return mlp_keys(site) if kind == "mlp" else attn_keys(site)


# ------------------------------------------------ renorm hooks (e014b, verbatim)
def register_renorm(model, c=C_RENORM):
    hooks = []
    for block in model.h:
        def pre(m, args, _c=c):
            x = args[0]
            n = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            return (x * (_c / n),)
        hooks.append(block.register_forward_pre_hook(pre))
    return hooks


# ------------------------------------------------ snapshot / transplant (e028)
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
    sd = model.state_dict()
    for k in keys:
        assert sd[k].shape == donor_sd[k].shape, k
        sd[k] = donor_sd[k].detach().clone().to(sd[k].dtype)
    model.load_state_dict(sd, strict=True)


# ------------------------------------------------ deterministic 30-batch eval (e028)
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
    donor = snap if self_transplant else donor_sd
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


# ------------------------------------------------ bootstrap (e011c/e028 discipline)
def bootstrap_ci(diffs: list[float], n: int = N_BOOT, seed: int = 2029):
    rng = random.Random(seed)
    nb = len(diffs)
    vals = []
    for _ in range(n):
        idx = [rng.randrange(nb) for _ in range(nb)]
        vals.append(sum(diffs[i] for i in idx) / nb)
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def bootstrap_paired_diff(a: list[float], b: list[float], n: int = N_BOOT, seed: int = 2030):
    """Paired CI of mean(a) - mean(b) over the same 30 batches."""
    rng = random.Random(seed)
    nb = len(a)
    vals = []
    for _ in range(n):
        idx = [rng.randrange(nb) for _ in range(nb)]
        vals.append(sum(a[i] for i in idx) / nb - sum(b[i] for i in idx) / nb)
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def bootstrap_ratio(a: list[float], b: list[float], n: int = N_BOOT, seed: int = 2031):
    """Paired CI of mean(a)/mean(b) (invalid resamples dropped)."""
    rng = random.Random(seed)
    nb = len(a)
    vals = []
    for _ in range(n):
        idx = [rng.randrange(nb) for _ in range(nb)]
        mb = sum(b[i] for i in idx) / nb
        if mb > 1e-9:
            vals.append(sum(a[i] for i in idx) / nb / mb)
    if not vals:
        return None, None
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def bootstrap_median(values: list[float], n: int = N_BOOT, seed: int = 2032):
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        s = [values[rng.randrange(len(values))] for _ in values]
        vals.append(statistics.median(s))
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


# ------------------------------------------------ renorm liveness assert (e028 §3)
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
                         f"!= {C_RENORM} +- 1e-3. Renorm-host evals would be off-manifold.")
    return probes


# ---------------------------------------------------------------- main
def main():
    rd = run_dir("e029")
    log("E029 seed x regime transplant matrix — corpus + checkpoints")
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)

    def load_ckpt(path):
        m = TinyGPT(cfg).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        return m

    B = load_ckpt(E001_CKPT)
    R = load_ckpt(R_CKPT)
    B43 = load_ckpt(B43_CKPT)
    log("loaded B (e001), R (e014b), B43 (e028_b43)")

    # ---- 1. R43: seed-43 renorm arm (hooks active in train AND eval) ----
    r43_steps, r43_retrained = None, False
    if R43_CKPT.exists():
        r43 = load_ckpt(R43_CKPT)
        log(f"R43 found at {R43_CKPT.name} — skipping training")
    else:
        set_seed(43)
        r43 = TinyGPT(cfg).to(DEVICE)
        hs = register_renorm(r43)
        log("training R43 (seed 43 renorm: steps 4000, lr 1e-3, bs 64, cap 252 s, resumable ckpt)")
        hist = train_model(r43, corpus, steps=4000, lr=1e-3, batch_size=64,
                           max_seconds=252.0, ckpt=R43_TRAIN)
        for h in hs:
            h.remove()
        hs = register_renorm(r43)
        probe = mean(per_batch_losses(r43, corpus))          # hooks active
        for h in hs:
            h.remove()
        r43_steps = hist[-1]["step"] if hist else 0
        if probe > R43_DIVERGED and elapsed() < 480:
            log("R43 diverged -> one retrain at lr 6e-4 (fallback ladder)")
            R43_TRAIN.unlink(missing_ok=True)
            set_seed(43)
            r43 = TinyGPT(cfg).to(DEVICE)
            hs = register_renorm(r43)
            hist = train_model(r43, corpus, steps=4000, lr=6e-4, batch_size=64,
                               max_seconds=252.0, ckpt=R43_TRAIN)
            for h in hs:
                h.remove()
            hs = register_renorm(r43)
            probe = mean(per_batch_losses(r43, corpus))
            for h in hs:
                h.remove()
            r43_steps = hist[-1]["step"] if hist else 0
            r43_retrained = True
        torch.save(r43.state_dict(), R43_CKPT)
        log(f"R43 trained ({r43_steps} steps, probe val {probe:.4f}); final ckpt saved")

    # ---- base CEs (30 fixed batches) + sanity + liveness ----
    base_B = per_batch_losses(B, corpus)
    log(f"B   base CE {mean(base_B):.6f} (expect 1.622391)")
    if abs(mean(base_B) - 1.622391394774119) > 0.02:
        raise SystemExit(f"B base CE {mean(base_B):.4f} far from e001 reference — wrong checkpoint?")
    base_B43 = per_batch_losses(B43, corpus)
    log(f"B43 base CE {mean(base_B43):.6f} (e028: 1.569563)")

    hs = register_renorm(R)
    base_R = per_batch_losses(R, corpus)
    for h in hs:
        h.remove()
    log(f"R   base CE (hooks) {mean(base_R):.6f} (expect 1.609990)")
    if abs(mean(base_R) - 1.6099902311960856) > 0.02:
        raise SystemExit(f"R base CE {mean(base_R):.4f} far from e014b reference — wrong checkpoint?")
    hs = register_renorm(r43)
    base_R43 = per_batch_losses(r43, corpus)
    for h in hs:
        h.remove()
    r43_val = mean(base_R43)
    r43_gate = r43_val <= PARITY_GATE
    log(f"R43 base CE (hooks) {r43_val:.6f} (parity gate <= {PARITY_GATE}: "
        f"{'PASS' if r43_gate else 'FLAG — cells still run, noted in metrics'})")

    live_R = renorm_liveness_assert(R, corpus)
    live_R43 = renorm_liveness_assert(r43, corpus)
    log(f"renorm liveness R {live_R} | R43 {live_R43}")

    # ---- C0 self-transplant gates (before any other cell) ----
    c0 = {}
    for host, name, base, renorm, kind0 in ((B, "B", base_B, False, "attn"),
                                            (B43, "B43", base_B43, False, "mlp")):
        losses = eval_cell(host, corpus, None, organ_keys(0, kind0), renorm, base,
                           self_transplant=True)
        dce = mean(losses) - mean(base)
        if losses != base or dce != 0.0:
            raise SystemExit(f"C0 GATE FAILURE: {name} {kind0}-L0 self-transplant "
                             f"dCE {dce!r}. Surgery path broken — aborting.")
        c0[f"{name}_{kind0}_L0"] = {"dce": 0.0, "ce_exactly_zero": True}
        log(f"C0 {name} {kind0}-L0: bitwise-identical state, dCE exactly 0.0")

    # ---- in-run ablation refs (same 30 batches as the cells) ----
    HOSTS: dict[str, tuple] = {"B": (B, base_B, False), "B43": (B43, base_B43, False)}
    ABL: dict[str, dict] = {}

    def compute_ablate(host, base, renorm) -> dict:
        row = {}
        for kind in KINDS:
            vals = []
            for site in SITES:
                hs = register_renorm(host) if renorm else []
                v = lesion_loss(host, corpus, kind, site, n_batches=N_EVAL)
                for h in hs:
                    h.remove()
                vals.append(v - mean(base))
            row[kind] = vals
        return row

    spot = {}
    e1 = json.loads((REPO / "runs" / "e001" / "metrics.json").read_text(encoding="utf-8"))
    e14 = json.loads((REPO / "runs" / "e014b" / "metrics.json").read_text(encoding="utf-8"))
    for name in HOSTS:
        ABL[name] = compute_ablate(*HOSTS[name])
        log(f"ablate refs {name}: attn {[round(v, 3) for v in ABL[name]['attn']]} "
            f"mlp {[round(v, 3) for v in ABL[name]['mlp']]}")
    spot["B_mlp_L0"] = {"value": ABL["B"]["mlp"][0], "ref": e1["mlp_block_damage"][0]}
    spot["B_attn_L0"] = {"value": ABL["B"]["attn"][0], "ref": e1["attn_block_damage"][0]}
    spot_pass_b = (abs(spot["B_mlp_L0"]["value"] - spot["B_mlp_L0"]["ref"]) < 0.05
                   and abs(spot["B_attn_L0"]["value"] - spot["B_attn_L0"]["ref"]) < 0.05)
    log(f"spot-check vs e001: mlp-L0 {ABL['B']['mlp'][0]:+.4f}/{e1['mlp_block_damage'][0]:+.4f} "
        f"attn-L0 {ABL['B']['attn'][0]:+.4f}/{e1['attn_block_damage'][0]:+.4f} "
        f"({'PASS' if spot_pass_b else 'MISMATCH'})")

    # ---- 2x2 transplant cells ----
    sds = {"B": snapshot(B), "B43": snapshot(B43), "R": snapshot(R), "R43": snapshot(r43)}
    DONORS = {
        "B":   {"seed": "B43", "regime": "R",   "both": "R43"},
        "B43": {"seed": "B",   "regime": "R43", "both": "R"},
    }
    CELLS: dict[str, dict] = {}

    def run_host_cells(hname: str, first_gate: dict):
        host, base, renorm = HOSTS[hname]
        abl = ABL[hname]
        first = True
        for site in SITES:
            for kind in KINDS:
                av = abl[kind][SITES.index(site)]
                for role in ("seed", "regime", "both"):
                    label = f"{hname}<-{role}|L{site}|{kind}"
                    losses = eval_cell(host, corpus, sds[DONORS[hname][role]],
                                       organ_keys(site, kind), renorm, base)
                    diffs = [a - b for a, b in zip(losses, base)]
                    dce = mean(diffs)
                    lo, hi = bootstrap_ci(diffs)
                    CELLS[label] = {"dce": dce, "ci95": [lo, hi], "per_batch_dce": diffs,
                                    "ablate_ref": av, "r": dce / av}
                    log(f"  {label:22s} dCE {dce:+.4f} (CI {lo:+.3f},{hi:+.3f})  R {dce / av:+.2f}")
                if first:
                    assert_identical(host, sds[hname], hname)
                    first_gate[hname] = True
                    first = False

    gates: dict[str, bool] = {}
    log("core hosts B + B43: 3 roles x 6 organs each (36 cells)")
    run_host_cells("B", gates)
    run_host_cells("B43", gates)

    # ---- optional R host (renorm hooks) ----
    run_r_host = elapsed() <= 600
    if run_r_host:
        log("optional R host within budget: C0 gate + ablate refs + 18 cells (hooks active)")
        losses = eval_cell(R, corpus, None, organ_keys(0, "attn"), True, base_R,
                           self_transplant=True)
        dce = mean(losses) - mean(base_R)
        if losses != base_R or dce != 0.0:
            raise SystemExit(f"C0 GATE FAILURE: R attn-L0 self-transplant dCE {dce!r}.")
        c0["R_attn_L0"] = {"dce": 0.0, "ce_exactly_zero": True}
        log("C0 R attn-L0: bitwise-identical state, dCE exactly 0.0")
        HOSTS["R"] = (R, base_R, True)
        ABL["R"] = compute_ablate(R, base_R, True)
        log(f"ablate refs R(hooks): attn {[round(v, 3) for v in ABL['R']['attn']]} "
            f"mlp {[round(v, 3) for v in ABL['R']['mlp']]}")
        spot["R_attn_L3"] = {"value": ABL["R"]["attn"][SITES.index(3)],
                             "ref": e14["renorm_damage"]["attn"][3]}
        DONORS["R"] = {"seed": "R43", "regime": "B", "both": "B43"}
        run_host_cells("R", gates)
    else:
        log(f"stop-rule: R host dropped (elapsed {elapsed():.0f}s > 600 s)")

    host_order = [h for h in ("B", "B43", "R") if h in HOSTS]

    # ---- verdict: seed dominance ----
    verdict_cells = []
    for hname in host_order:
        for site in SITES:
            for kind in KINDS:
                s = CELLS[f"{hname}<-seed|L{site}|{kind}"]
                rg = CELLS[f"{hname}<-regime|L{site}|{kind}"]
                bo = CELLS[f"{hname}<-both|L{site}|{kind}"]
                ds, dr = s["dce"], rg["dce"]
                rho = ds / dr if abs(dr) > 1e-12 else None
                rlo, rhi = bootstrap_ratio(s["per_batch_dce"], rg["per_batch_dce"])
                dlo, dhi = bootstrap_paired_diff(s["per_batch_dce"], rg["per_batch_dce"])
                denom = ds + dr
                verdict_cells.append({
                    "host": hname, "site": site, "kind": kind,
                    "dce_seed": ds, "dce_regime": dr, "dce_both": bo["dce"],
                    "r_seed": s["r"], "r_regime": rg["r"], "r_both": bo["r"],
                    "rho": rho, "rho_ci95": [rlo, rhi],
                    "seed_minus_regime_ci95": [dlo, dhi],
                    "diff_ci_excludes_0": bool(dlo > 0 or dhi < 0),
                    "noise_floor": bool(dr < NOISE_FLOOR),
                    "additivity_ratio": bo["dce"] / denom if abs(denom) > 1e-9 else None,
                    "additivity_diff": bo["dce"] - denom,
                })
    eligible = [c for c in verdict_cells if not c["noise_floor"] and c["rho"] is not None]
    excluded = [c for c in verdict_cells if c["noise_floor"]]
    rhos = [c["rho"] for c in eligible]
    median_rho = statistics.median(rhos) if rhos else None
    mlo, mhi = bootstrap_median(rhos) if rhos else (None, None)
    core = [c for c in eligible if c["host"] in ("B", "B43")]
    median_rho_core = statistics.median([c["rho"] for c in core]) if core else None
    med_seed = statistics.median([c["dce_seed"] for c in eligible]) if eligible else None
    med_regime = statistics.median([c["dce_regime"] for c in eligible]) if eligible else None
    ratio_of_medians = med_seed / med_regime if med_regime and abs(med_regime) > 1e-9 else None
    seed_dominates = bool(median_rho is not None
                          and median_rho >= SEED_DOMINANCE_THRESHOLD)

    # ---- dW alignment (mechanism observable) ----
    set_seed(42)
    init42 = snapshot(TinyGPT(cfg).to(DEVICE))
    set_seed(43)
    init43 = snapshot(TinyGPT(cfg).to(DEVICE))
    inits = {"B": init42, "R": init42, "B43": init43, "R43": init43}
    seed_of = {"B": 42, "R": 42, "B43": 43, "R43": 43}

    def dw(name: str, keys: list[str]) -> torch.Tensor:
        parts = [(sds[name][k].float() - inits[name][k].float()).reshape(-1) for k in keys]
        return torch.cat(parts)

    def cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12))

    pairs = [("B", "R", True), ("B43", "R43", True),
             ("B", "B43", False), ("B", "R43", False),
             ("R", "B43", False), ("R", "R43", False)]
    organs = [(s, k) for s in SITES for k in KINDS]
    cos_table: dict[tuple, dict] = {}
    for a, b, same in pairs:
        entry = {}
        for s, k in organs:
            entry[f"L{s}|{k}"] = cos(dw(a, organ_keys(s, k)), dw(b, organ_keys(s, k)))
        cos_table[(a, b)] = {"same_init": same, "organs": entry,
                             "mean": mean(list(entry.values()))}
        log(f"dW cos {a:3s}<->{b:3s} ({'SAME' if same else 'diff'}-init "
            f"seed{seed_of[a]}): mean {cos_table[(a, b)]['mean']:+.4f} "
            f"| " + " ".join(f"L{s}{k[0]} {entry[f'L{s}|{k}']:+.3f}" for s, k in organs))
    same_vals = [v for p in cos_table.values() if p["same_init"] for v in p["organs"].values()]
    diff_vals = [v for p in cos_table.values() if not p["same_init"] for v in p["organs"].values()]
    per_organ = {}
    n_organs_same_gt = 0
    for s, k in organs:
        key = f"L{s}|{k}"
        sm = mean([p["organs"][key] for p in cos_table.values() if p["same_init"]])
        dm = mean([p["organs"][key] for p in cos_table.values() if not p["same_init"]])
        per_organ[key] = {"same_init_mean": sm, "diff_init_mean": dm, "same_gt_diff": sm > dm}
        n_organs_same_gt += sm > dm
    mean_same, mean_diff = mean(same_vals), mean(diff_vals)
    gap = mean_same - mean_diff
    mechanism_confirmed = bool(gap >= 0.05 and n_organs_same_gt >= 5)
    log(f"dW alignment: same-init mean {mean_same:+.4f} (n={len(same_vals)}) vs diff-init "
        f"{mean_diff:+.4f} (n={len(diff_vals)}); gap {gap:+.4f}; same>diff in "
        f"{n_organs_same_gt}/6 organs -> {'CONFIRMED' if mechanism_confirmed else 'NOT CONFIRMED'}")

    # ------------------------------------------------ outputs
    log("writing metrics + figures")
    metrics = {
        "experiment": "e029_seed_regime_matrix",
        "date": now_iso(),
        "device": DEVICE,
        "config": cfg_dict(cfg),
        "sites": list(SITES), "kinds": list(KINDS), "n_eval_batches": N_EVAL,
        "eval_protocol": "30 fixed deterministic val batches (generator seeded from "
                         "corpus.seed 1337); identical batches for every arm (e028 protocol)",
        "registered_predictions": {
            "seed_dominance": "median over cells of dCE(seed-axis)/dCE(regime-axis) >= 1.5 "
                              "-> seed axis dominates (THINKING.md T006 P3 resolution, "
                              "registered before e028 ran)",
            "mechanism_dw_alignment": "same-init pairs (B<->R seed42, B43<->R43 seed43) have "
                                      "systematically higher cos(dW_donor, dW_host); "
                                      "confirmed iff mean gap >= 0.05 AND same>diff in >= 5/6 "
                                      "organs (organs refine init-anchored subspaces)",
        },
        "models": {"B": {"seed": 42, "regime": "base", "ckpt": E001_CKPT.name},
                   "B43": {"seed": 43, "regime": "base", "ckpt": B43_CKPT.name},
                   "R": {"seed": 42, "regime": "renorm", "ckpt": R_CKPT.name},
                   "R43": {"seed": 43, "regime": "renorm", "ckpt": R43_CKPT.name}},
        "r43_training": {"val": r43_val, "steps": r43_steps, "parity_gate": PARITY_GATE,
                         "parity_gate_pass": bool(r43_gate),
                         "retrained_lr6e4": r43_retrained,
                         "hooks": "register_renorm active in train AND eval"},
        "base_ce": {"B": mean(base_B), "B43": mean(base_B43),
                    "R_hooks": mean(base_R), "R43_hooks": r43_val},
        "sanity": {"b_base_matches_e028": True, "r_base_matches_e028": True,
                   "renorm_liveness_R": live_R, "renorm_liveness_R43": live_R43},
        "c0_self_transplant": c0 | {"gate_pass": True},
        "ablation_refs_inrun": ABL,
        "spot_checks": spot | {"B_pass": bool(spot_pass_b),
                               "R_pass": bool(spot.get("R_attn_L3", {}).get("value") is None
                                              or abs(spot["R_attn_L3"]["value"]
                                                     - spot["R_attn_L3"]["ref"]) < 0.05)},
        "cells": CELLS,
        "verdict_seed_dominance": {
            "rule": f"median dCE(seed-axis)/dCE(regime-axis) over eligible cells "
                    f"(regime dCE >= {NOISE_FLOOR}) >= {SEED_DOMINANCE_THRESHOLD}",
            "per_cell": verdict_cells,
            "median_rho_all": median_rho, "median_rho_ci95": [mlo, mhi],
            "median_rho_core_B_B43": median_rho_core,
            "ratio_of_medians": ratio_of_medians,
            "median_dce_seed": med_seed, "median_dce_regime": med_regime,
            "n_eligible": len(eligible),
            "n_seed_minus_regime_ci_excludes_0": sum(c["diff_ci_excludes_0"] for c in eligible),
            "noise_floor_cells": [{"host": c["host"], "site": c["site"], "kind": c["kind"],
                                   "dce_regime": c["dce_regime"], "dce_seed": c["dce_seed"]}
                                  for c in excluded],
            "seed_axis_dominates": seed_dominates,
        },
        "dw_alignment": {
            "per_pair": {f"{a}<->{b}": v for (a, b), v in cos_table.items()},
            "per_organ_means": per_organ,
            "same_init": {"n": len(same_vals), "mean": mean_same},
            "diff_init": {"n": len(diff_vals), "mean": mean_diff},
            "mean_gap": gap, "organs_same_gt_diff": n_organs_same_gt,
            "mechanism_confirmed": mechanism_confirmed,
        },
        "blocks_run": {"r_host": run_r_host},
        "timing_s": {"total": round(elapsed(), 1)},
    }
    save_json(rd / "metrics.json", metrics)

    # ---- figure 1: 2x2 dCE heatmaps per host + per-organ breakdown ----
    donor_at = {("42", "base"): "B", ("42", "renorm"): "R",
                ("43", "base"): "B43", ("43", "renorm"): "R43"}
    n_hosts = len(host_order)
    fig = plt.figure(figsize=(4.7 * n_hosts + 0.6, 8.2))
    gs = fig.add_gridspec(2, n_hosts, height_ratios=[1.15, 1.0])
    grids, anns = {}, {}
    for hname in host_order:
        role_of = {donor: role for role, donor in DONORS[hname].items()}
        grid = np.zeros((2, 2))
        ann = [["", ""], ["", ""]]
        for i, sd_ in enumerate(("42", "43")):
            for jj, rg_ in enumerate(("base", "renorm")):
                donor = donor_at[(sd_, rg_)]
                if donor == hname:
                    grid[i, jj] = 0.0
                    ann[i][jj] = "self"
                else:
                    role = role_of[donor]
                    vals = [CELLS[f"{hname}<-{role}|L{s}|{k}"]["dce"]
                            for s, k in organs]
                    grid[i, jj] = mean(vals)
                    ann[i][jj] = f"{role}\n{mean(vals):+.3f}"
        grids[hname], anns[hname] = grid, ann
    vmax = max(0.3, max(g.max() for g in grids.values()))
    for j, hname in enumerate(host_order):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(grids[hname], cmap="Reds", vmin=0.0, vmax=vmax, aspect="auto")
        for i in range(2):
            for jj in range(2):
                ax.text(jj, i, anns[hname][i][jj], ha="center", va="center", fontsize=9,
                        color="black" if grids[hname][i, jj] < 0.6 * vmax else "white")
        ax.set_xticks(range(2), ["base", "renorm"])
        ax.set_yticks(range(2), ["seed42", "seed43"])
        ax.set_xlabel("donor regime")
        ax.set_ylabel("donor seed")
        hook = " (renorm hooks)" if hname == "R" else ""
        ax.set_title(f"host {hname}{hook}", fontsize=10)
        fig.colorbar(im, ax=ax, label="mean dCE over 6 organs (nats)", shrink=0.85)
    axb = fig.add_subplot(gs[1, :])
    xs = np.arange(len(organs))
    w = 0.27
    for off, role, col in ((-w, "seed", "#b8860b"), (0.0, "regime", "#2c6fbb"), (w, "both", "#7a3fb5")):
        vals = [mean([CELLS[f"{h}<-{role}|L{s}|{k}"]["dce"] for h in host_order])
                for s, k in organs]
        axb.bar(xs + off, vals, w, label=f"{role}-axis", color=col)
    axb.set_xticks(xs, [f"L{s}|{k}" for s, k in organs])
    axb.set_ylabel("mean dCE across hosts (nats)")
    axb.set_title(f"per-organ breakdown: seed- vs regime-axis vs both "
                  f"(median rho = {median_rho:.2f}; threshold {SEED_DOMINANCE_THRESHOLD})")
    axb.legend(fontsize=9)
    fig.suptitle("E029 — seed x regime transplant matrix: "
                 f"seed dominance {'CONFIRMED' if seed_dominates else 'NOT CONFIRMED'} "
                 f"(median dCE seed/regime = {median_rho:.2f})", fontsize=12)
    fig.tight_layout()
    fig.savefig(rd / "seed_regime_heatmap.png", dpi=140)
    plt.close(fig)

    # ---- figure 2: dW alignment comparison ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))
    xs = np.arange(len(organs))
    for (a, b), p in cos_table.items():
        ys = [p["organs"][f"L{s}|{k}"] for s, k in organs]
        style = "-" if p["same_init"] else "--"
        ax1.plot(xs, ys, style, marker="o" if p["same_init"] else "s", ms=4,
                 label=f"{a}<->{b} ({'same' if p['same_init'] else 'diff'}-init "
                       f"seed{seed_of[a]})", alpha=0.85)
    ax1.axhline(0.0, color="k", lw=0.8)
    ax1.set_xticks(xs, [f"L{s}|{k}" for s, k in organs], fontsize=8)
    ax1.set_ylabel("cos(dW_donor, dW_host)")
    ax1.set_title("per-organ dW alignment (solid = same init)")
    ax1.legend(fontsize=7)
    rng = np.random.default_rng(0)
    for vals, col, lab, off in ((same_vals, "#b8860b", f"same-init (n={len(same_vals)})", -0.12),
                                (diff_vals, "#2c6fbb", f"diff-init (n={len(diff_vals)})", 0.12)):
        ax2.scatter([off + rng.uniform(-0.05, 0.05) for _ in vals], vals, s=14,
                    color=col, alpha=0.6, label=lab)
        ax2.scatter([off], [mean(vals)], marker="D", s=60, color=col, zorder=5,
                    edgecolor="k")
    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.set_xticks([0], ["dW cosine"])
    ax2.set_xlim(-0.5, 0.5)
    ax2.legend(fontsize=9)
    ax2.set_title(f"init-lineage alignment: same {mean_same:+.3f} vs diff {mean_diff:+.3f} "
                  f"(gap {gap:+.3f}, {n_organs_same_gt}/6 organs) -> "
                  f"{'CONFIRMED' if mechanism_confirmed else 'NOT CONFIRMED'}")
    fig.suptitle("E029 mechanism: organs refine init-anchored subspaces "
                 "(dW = W_trained - W_init per organ)", fontsize=12)
    fig.tight_layout()
    fig.savefig(rd / "dw_alignment.png", dpi=140)
    plt.close(fig)

    # ------------------------------------------------ headline prints
    log("=" * 78)
    log(f"SEED-DOMINANCE VERDICT: {'CONFIRMED — seed axis dominates' if seed_dominates else 'NOT CONFIRMED'}"
        f" | median dCE(seed)/dCE(regime) = {median_rho:.3f} "
        f"(CI {mlo:.2f},{mhi:.2f}; threshold {SEED_DOMINANCE_THRESHOLD}) over "
        f"{len(eligible)} eligible cells; core B+B43 median {median_rho_core:.3f}; "
        f"ratio-of-medians {ratio_of_medians:.3f}")
    log("per-site breakdown (host  site  kind  seed  regime  both  rho):")
    for c in verdict_cells:
        flag = "  [noise-floor]" if c["noise_floor"] else ""
        rho_s = "n/a" if c["rho"] is None else f"{c['rho']:6.2f}"
        log(f"  {c['host']:3s} L{c['site']} {c['kind']:4s} {c['dce_seed']:+7.3f} "
            f"{c['dce_regime']:+7.3f} {c['dce_both']:+7.3f}  {rho_s}{flag}")
    log(f"dW ALIGNMENT: same-init mean {mean_same:+.4f} vs diff-init {mean_diff:+.4f} "
        f"(gap {gap:+.4f}; same>diff in {n_organs_same_gt}/6 organs) -> "
        f"{'CONFIRMED' if mechanism_confirmed else 'NOT CONFIRMED'}")
    adds = [c["additivity_ratio"] for c in verdict_cells if c["additivity_ratio"] is not None]
    if adds:
        log(f"additivity (both-axes over seed+regime): median "
            f"{statistics.median(adds):.3f} over {len(adds)} cells")
    log(f"outputs: {rd}")
    return metrics


if __name__ == "__main__":
    main()
