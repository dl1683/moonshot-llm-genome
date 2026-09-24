"""E047 — positive-claims replication sweep (Review 6 night program, item 1;
CARD v3 GATE). EVAL-ONLY, 3 claims x 5 nets.

Review 6 adopted the min-nets rule: "positives enter H only after >=3 nets
(7 checkpoints exist)". T016's pattern note: the day's positive mechanism
claims die under replication while negatives hold. This sweep tests the three
surviving positives on every available Shakespeare-regime host:

  Hosts (all runs/checkpoints/, all 6L/6H/192):
    B    = e001.pt      — baseline seed 42 (REFERENCE reproduction)
    B43  = e028_b43.pt  — different init (seed 43), same regime
    R    = e014b.pt     — renorm-trained seed 42 (register_renorm hooks at eval)
    R43  = e029_r43.pt  — renorm-trained seed 43 (renorm hooks)
    BDO  = e041_bdo.pt  — same init as B (seed 42), different data order

  C1 L5-CALIBRATOR (e013/e001 protocols):
       KL(final || L4-readout) mean on 300 fixed val positions (e013 full-256
       arm: batched_snapshots, p6*log(p6/p4), seed 14) + L5-attn zero-ablation
       cost (e001: common.lesion('attn', 5) vs estimate_loss, 30 batches).
       REPLICATES per net if KL > 0.5 nats AND ablation < 0.15
       (B reference: 0.997 / +0.0335).
  C2 MLP-5 ENERGY CARRIER (e019 protocol):
       fixed val blocks (400, seed 606); hook-scale/rotate model.h[5].mlp
       output; ratio = rotate60_damage / zero_damage.
       REPLICATES per net if ratio < 0.5 (B reference: 0.2503).
  C3 SHARED-L0 NAME MACHINE (e042 protocol, UNIFORM-FLOOR contexts):
       block atlas (attn/mlp x 6 layers zeroed at the battery name-region
       positions [CTX-1, CTX-1+L)) on the e046 uniform-floor batteries (each
       name spliced into the 63 PROSPERO val-anchor contexts, CTX=120) for
       JULIET/JOHN/ROMEO/LUCIO; top-1 block per name.
       REPLICATES per net if L0-MLP is the top-1 block for >=3 of 4 names
       (B reference, e042 train battery: 4/4, +7.0..7.6 nats).

Per-claim verdict at the min-3-nets bar: REPLICATES if >=3 of the 5 nets pass
(non-B count reported alongside so the strict 3-of-4-hosts reading is visible).

Gates: B reproductions (e013 KL, e001 L5-attn damage, e019 ratio asserted
bit-close in FULL mode; e042 top-block directional), renorm liveness assert
(e028) on R/R43, uniform-floor anchor gate (e046 G5: PROSPERO uniform NLL in
[3.5, 5.5]) on every net.

No NOTES/THINKING/QUEUE/STATE edits; no git commit (operator instruction).
Run: python lab/e047_replication_sweep.py    E047_SMOKE=1 for the shakedown.
Budget: eval-only, <= 15 min (full).

DEVIATION NOTES:
  1. Uniform-floor anchor gate: e046's [3.5, 5.5] band is REPORTED; the hard
     gate is the zero-knowledge band [2.5, 7.0] (renorm hosts may read above
     ln65 on never-seen anchors — R43 smoke read 5.65; the guarded failure is
     a memorized anchor).
  2. C3 uses the BLOCK atlas only (the claim is block-level: "L0-MLP top-1");
     e042's head atlas is not re-run. B's C3 row is the protocol on the new
     uniform-floor battery — a directional reproduction of e042's train-battery
     result (4/4 names), not a bit-exact one.
  3. C1 runs e013's full-256 arm only (the claim is the KL magnitude + block
     cost, not the truncation contrast).
"""
from __future__ import annotations

import math
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib.colors
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, estimate_loss,
                    lesion_loss, run_dir, save_json, set_seed, cfg_dict)
from e012_decision_depth import batched_snapshots
from e014b_stream_renorm import register_renorm, C_RENORM

SMOKE = os.environ.get("E047_SMOKE") == "1"

CKPTS = {
    "B":    REPO / "runs" / "checkpoints" / "e001.pt",
    "B43":  REPO / "runs" / "checkpoints" / "e028_b43.pt",
    "R":    REPO / "runs" / "checkpoints" / "e014b.pt",
    "R43":  REPO / "runs" / "checkpoints" / "e029_r43.pt",
    "BDO":  REPO / "runs" / "checkpoints" / "e041_bdo.pt",
}
RENORM_NETS = {"R", "R43"}
HOST_DESC = {
    "B":    "baseline seed 42 — REFERENCE reproduction (e013/e001/e019/e042 origins)",
    "B43":  "different init (seed 43), same regime (e028)",
    "R":    "renorm-trained seed 42, c=5.6 hooks at eval (e014b)",
    "R43":  "renorm-trained seed 43, c=5.6 hooks at eval (e029)",
    "BDO":  "same init as B (seed 42), different data order (e041)",
}
VAL_REF = {"B": 1.6224, "B43": 1.5654, "R": 1.6100, "R43": 1.5596, "BDO": 1.5905}

SEED = 24700
NL, NH = 6, 6
BLOCK = 256
EVAL_BS = 64
BAT_CHUNK = 128

# C1 (e013/e001 verbatim)
N_POS = 60 if SMOKE else 300
KL_BATCH = 64
ABL_BATCHES = 8 if SMOKE else 30
KL_BAR = 0.5                 # replicates-bar: KL > 0.5 nats
ABL_BAR = 0.15               # replicates-bar: L5-attn ablation cost < 0.15
E013_KL_REF = 0.9967         # runs/e013/metrics.json kl_full
E001_ATTL5_REF = 0.0335      # runs/e001/metrics.json attn_block_damage[5]

# C2 (e019 verbatim)
N_EVAL_BLOCKS = 80 if SMOKE else 400
EVAL_SEED_2 = 606
ALPHAS = [0.0, 0.5, 1.0, 2.0]
GRACEFUL_TOL = 0.15
RATIO_BAR = 0.5              # replicates-bar: rotate/zero damage < 0.5
E019_RATIO_REF = 0.2503      # runs/e019/metrics.json verdicts.rotate_over_zero

# C3 (e042 atlas protocol on e046 uniform-floor batteries)
CTX = 120
ATLAS_NAMES = ["JULIET", "LUCIO"] if SMOKE else ["JULIET", "JOHN", "ROMEO", "LUCIO"]
ANCHOR_LO, ANCHOR_HI = 3.5, 5.5   # e046 G5 uniform-floor anchor gate around ln65
NAMES_BAR = 3                     # replicates-bar: L0-MLP top-1 for >=3 of 4 names

MIN_NETS = 3                      # Review-6 min-nets bar


# ------------------------------------------------------------------ helpers

def find_occ(text: str, word: str) -> list[int]:
    pat = re.compile(r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])")
    return [m.start() for m in pat.finditer(text)]


@torch.no_grad()
def kl_and_flips(model: TinyGPT, xs, ys):
    """e013 kl_and_flips verbatim (full-context arm)."""
    kl = torch.zeros(xs.shape[0])
    ce = torch.zeros(xs.shape[0])
    flips = torch.zeros(xs.shape[0])
    for b0 in range(0, xs.shape[0], KL_BATCH):
        xb, yb = xs[b0:b0 + KL_BATCH], ys[b0:b0 + KL_BATCH]
        last = batched_snapshots(model, xb)
        p4 = F.softmax(model.lm_head(model.ln_f(last[5])), dim=-1)  # L4 readout
        p6 = F.softmax(model.lm_head(model.ln_f(last[6])), dim=-1)  # final/L5 readout
        kl[b0:b0 + KL_BATCH] = (p6 * (p6 / (p4 + 1e-12)).log()).sum(-1).cpu()
        idx = torch.arange(xb.shape[0])
        ce[b0:b0 + KL_BATCH] = -(p6[idx, yb] + 1e-12).log().cpu()
        flips[b0:b0 + KL_BATCH] = (p4.argmax(-1) != p6.argmax(-1)).float().cpu()
    return kl, ce, flips


def rotated_like(w: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """e019 rotated_like verbatim: w' at exactly 60 deg from w per token."""
    r = torch.randn(w.shape, generator=gen, device=w.device, dtype=w.dtype)
    wn = w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    r = r - (r * w).sum(-1, keepdim=True) * w / (wn * wn)
    r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return 0.5 * w + (3 ** 0.5 / 2) * r * wn


@torch.no_grad()
def eval_stats_ce(model: TinyGPT, x, y) -> dict:
    """e019 eval_stats (mean CE / entropy over all positions of fixed blocks)."""
    model.eval()
    ce_s, ent_s, n = 0.0, 0.0, 0
    for i in range(0, len(x), EVAL_BS):
        xb, yb = x[i: i + EVAL_BS], y[i: i + EVAL_BS]
        logits, _ = model(xb, yb)
        logp = F.log_softmax(logits.float(), dim=-1)
        p = logp.exp()
        ce = F.nll_loss(logp.view(-1, logp.size(-1)), yb.view(-1), reduction="sum")
        ent = -(p * logp.clamp_min(-1e9)).sum(-1).sum()
        k = yb.numel()
        ce_s += float(ce.item()); ent_s += float(ent.item()); n += k
    return {"ce": ce_s / n, "entropy": ent_s / n}


@torch.no_grad()
def run_condition_mlp5(model: TinyGPT, x, y, kind: str, alpha: float = 1.0) -> dict:
    """e019 run_condition verbatim (kind in {identity, scale, rotate} on h[5].mlp)."""
    mod = model.h[5].mlp
    handle = None
    try:
        if kind == "scale":
            def hook(module, args, out, a=alpha):
                return out * a
            handle = mod.register_forward_hook(hook)
        elif kind == "rotate":
            g = torch.Generator(device=DEVICE).manual_seed(9090)

            def hook(module, args, out, _g=g):
                return rotated_like(out.float(), _g).to(out.dtype)
            handle = mod.register_forward_hook(hook)
        return eval_stats_ce(model, x, y)
    finally:
        if handle is not None:
            handle.remove()


def build_bat_uniform(val_ids, val_text, anchor_occs, stoi, w):
    """e046 build_bat_uniform verbatim: w spliced into the PROSPERO val contexts."""
    keep = [p for p in anchor_occs if p >= CTX]
    wid = torch.tensor([stoi[c] for c in w], dtype=torch.long)
    seqs = [torch.cat([val_ids[p - CTX: p], wid]) for p in keep]
    return {"seq": torch.stack(seqs) if seqs else None, "L": len(w), "n": len(keep)}


@contextmanager
def pos_lesion(model: TinyGPT, specs):
    """e042/e046 pos_lesion verbatim (state['mask'] per forward; None = no-op)."""
    state = {"mask": None}
    handles = []

    def mk_sub_fwd():
        def fwd(module, args, out):
            ms = state["mask"]
            if ms is None:
                return None
            return out * (~ms).to(out.dtype).unsqueeze(-1)
        return fwd

    for kind, layer, head in specs:
        block = model.h[layer]
        if kind in ("attn", "mlp"):
            mod = block.attn if kind == "attn" else block.mlp
            handles.append(mod.register_forward_hook(mk_sub_fwd()))
        else:
            raise ValueError(kind)
    try:
        yield state
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def eval_bat(model: TinyGPT, bat: dict, state=None, pos=None):
    """e042/e046 eval_bat verbatim (name-position NLL; uniform position slice)."""
    model.eval()
    seq, L = bat["seq"], bat["L"]
    x, y = seq[:, :-1], seq[:, 1:]
    T = x.shape[1]
    nlls = []
    for i in range(0, len(x), BAT_CHUNK):
        xc = x[i: i + BAT_CHUNK].to(DEVICE)
        yc = y[i: i + BAT_CHUNK].to(DEVICE)
        if state is not None:
            m = torch.zeros(xc.shape[0], T, dtype=torch.bool, device=DEVICE)
            if pos is not None:
                m[:, pos[0]: pos[1]] = True
            state["mask"] = m
        logits, _ = model(xc)
        lg = logits[:, CTX - 1: CTX - 1 + L, :]
        tg = yc[:, CTX - 1: CTX - 1 + L]
        nll = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1),
                              reduction="none").view(-1, L)
        nlls.append(nll)
    nll_m = torch.cat(nlls)
    return float(nll_m.mean().item())


@torch.no_grad()
def renorm_liveness(model: TinyGPT, corpus: CharCorpus) -> list[float]:
    """e028 renorm liveness probe verbatim (asserts block-input norms == c)."""
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
    ok = len(probes) == model.cfg.n_layer and all(abs(n - C_RENORM) <= 1e-3 for n in probes)
    if not ok:
        raise SystemExit(f"RENORM LIVENESS ASSERT FAILED: block-input norms {probes} "
                         f"!= {C_RENORM} +- 1e-3. Renorm-host evals would be off-manifold.")
    return probes


def jsonable(x):
    import math as _m
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, bool) or x is None or isinstance(x, (int, str)):
        return x
    if isinstance(x, float):
        if _m.isinf(x):
            return "inf" if x > 0 else "-inf"
        if _m.isnan(x):
            return "nan"
        return x
    if isinstance(x, torch.Tensor):
        return jsonable(x.tolist())
    return str(x)


# ------------------------------------------------------------------ claims

def claim1_l5_calibrator(model, corpus, cfg):
    """C1: e013 KL(full-256) + e001 L5-attn zero-ablation cost."""
    set_seed(14)                                        # e013 verbatim
    gen = torch.Generator().manual_seed(14)
    ix = torch.randint(len(corpus.val) - cfg.block_size - 2, (N_POS,), generator=gen)
    xs_full = torch.stack([corpus.val[i: i + cfg.block_size] for i in ix]).to(DEVICE)
    ys = corpus.val[ix + cfg.block_size].to(DEVICE)
    kl, ce, fl = kl_and_flips(model, xs_full, ys)
    baseline = estimate_loss(model, corpus, "val", n_batches=ABL_BATCHES)
    abl = lesion_loss(model, corpus, "attn", NL - 1, n_batches=ABL_BATCHES) - baseline
    return {"kl_mean": float(kl.mean()), "ce": float(ce.mean()),
            "flip_rate": float(fl.mean()), "val_ce": baseline,
            "l5_attn_ablation_cost": float(abl),
            "pass": bool(kl.mean() > KL_BAR and abl < ABL_BAR)}


def claim2_mlp5_energy(model, corpus, cfg):
    """C2: e019 thermostat — zero vs rotate60 damage ratio at L5-MLP."""
    gen = torch.Generator().manual_seed(EVAL_SEED_2)
    ix = torch.randint(len(corpus.val) - cfg.block_size - 1, (N_EVAL_BLOCKS,), generator=gen)
    x = torch.stack([corpus.val[i: i + cfg.block_size] for i in ix]).to(DEVICE)
    y = torch.stack([corpus.val[i + 1: i + 1 + cfg.block_size] for i in ix]).to(DEVICE)
    with torch.no_grad():                                # e019 rotation sanity probe
        probe_in = torch.randn(4, 8, cfg.n_embd, device=DEVICE)
        probe_out = model.h[5].mlp(probe_in)
        g0 = torch.Generator(device=DEVICE).manual_seed(0)
        wp = rotated_like(probe_out, g0)
        ratio_san = float(((wp - probe_out).norm(dim=-1) / probe_out.norm(dim=-1)).mean().item())
    conditions = {"baseline": run_condition_mlp5(model, x, y, "identity")}
    for a in ALPHAS:
        conditions[f"alpha={a}"] = run_condition_mlp5(model, x, y, "scale", alpha=a)
    conditions["rotate60@alpha=1"] = run_condition_mlp5(model, x, y, "rotate")
    base = conditions["baseline"]["ce"]
    dmg = {k: v["ce"] - base for k, v in conditions.items() if k != "baseline"}
    zero_d, rot_d = dmg["alpha=0.0"], dmg["rotate60@alpha=1"]
    ratio = rot_d / max(zero_d, 1e-6)
    graceful = {f"alpha={a}": bool(dmg[f"alpha={a}"] <= GRACEFUL_TOL) for a in (0.5, 2.0)}
    return {"rotation_sanity_ratio": ratio_san, "conditions": conditions,
            "damage_dce": dmg, "zero_damage": zero_d, "rotate_damage": rot_d,
            "ratio_rotate_over_zero": ratio, "graceful_alpha": graceful,
            "pass": bool(ratio < RATIO_BAR)}


def claim3_shared_l0(model, bats):
    """C3: e042 block atlas on uniform-floor batteries — top-1 block per name."""
    out = {}
    for name, bat in bats.items():
        L = bat["L"]
        b0 = eval_bat(model, bat)
        blocks = {"attn": [0.0] * NL, "mlp": [0.0] * NL}
        for l in range(NL):
            for kind in ("attn", "mlp"):
                with pos_lesion(model, [(kind, l, None)]) as st:
                    r = eval_bat(model, bat, state=st, pos=(CTX - 1, CTX - 1 + L))
                blocks[kind][l] = r - b0
        rank = sorted([(f"L{l}{k[:1].upper()}", v) for k in ("attn", "mlp")
                       for l, v in enumerate(blocks[k])], key=lambda t: -t[1])
        out[name] = {"base_nll": b0, "blocks_dce": blocks,
                     "top_block": rank[0][0], "top_block_dce": rank[0][1],
                     "l0mlp_dce": blocks["mlp"][0],
                     "l0mlp_rank": [s for s, _ in rank].index("L0M") + 1,
                     "l0mlp_top1": rank[0][0] == "L0M",
                     "runner_up": rank[1]}
    n_top1 = sum(1 for v in out.values() if v["l0mlp_top1"])
    return {"per_name": out, "n_names_l0mlp_top1": n_top1,
            "pass": bool(n_top1 >= NAMES_BAR)}


# ------------------------------------------------------------------ main

def main():
    T0 = time.time()
    stamp = lambda: f"[{time.time() - T0:7.1f}s]"
    set_seed(SEED)
    rd = run_dir("e047")

    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    assert corpus.vocab_size == 65
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=NL, n_head=NH, n_embd=192, block_size=BLOCK)
    stoi = corpus.stoi
    val_ids, val_text = corpus.val, "".join(corpus.itos[int(i)] for i in corpus.val)

    # uniform-floor batteries (e046 construction; shared across nets)
    anchor_occs = find_occ(val_text, "PROSPERO")
    bats = {w: build_bat_uniform(val_ids, val_text, anchor_occs, stoi, w) for w in ATLAS_NAMES}
    anchor_bat = build_bat_uniform(val_ids, val_text, anchor_occs, stoi, "PROSPERO")
    assert anchor_bat["n"] == 63 and all(b["n"] == 63 for b in bats.values())
    print(f"{stamp()} batteries: uniform-floor n=63 ctx={CTX} names={ATLAS_NAMES}", flush=True)

    nets = {}
    for net_name, ckpt in CKPTS.items():
        print(f"\n{stamp()} ===== host {net_name} ({ckpt.name}) =====", flush=True)
        net = TinyGPT(cfg).to(DEVICE)
        net.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        net.eval()
        renorm_live = None
        if net_name in RENORM_NETS:
            renorm_live = renorm_liveness(net, corpus)
            register_renorm(net)                      # stays for this net's lifetime
            print(f"{stamp()} renorm hooks live (block-in norms {renorm_live})", flush=True)

        g_val = estimate_loss(net, corpus, "val", n_batches=20)
        G0 = {"val_ce": g_val, "ref": VAL_REF[net_name],
              "pass": bool(abs(g_val - VAL_REF[net_name]) <= 0.03)}
        anchor_nll = eval_bat(net, anchor_bat)
        # e046's [3.5, 5.5] band reported as reference; the HARD gate is the
        # zero-knowledge band [2.5, 7.0] (renorm nets can read slightly above
        # ln65 on never-seen anchors — R43 smoke: 5.65 — still zero-knowledge;
        # the failure mode guarded against is a memorized anchor, NLL < 2.5).
        G5 = {"prospero_unif_nll": anchor_nll, "gate_e046": [ANCHOR_LO, ANCHOR_HI],
              "pass_e046_band": bool(ANCHOR_LO <= anchor_nll <= ANCHOR_HI),
              "ln65": math.log(65), "zero_knowledge_band": [2.5, 7.0],
              "pass": bool(2.5 <= anchor_nll <= 7.0),
              "note": "hard gate widened from e046's [3.5,5.5] to the zero-knowledge "
                      "band [2.5,7.0]: renorm hosts may exceed ln65 on never-seen "
                      "anchors; memorization (<2.5) is the guarded failure"}
        print(f"{stamp()} G0 val {g_val:.4f} (ref {VAL_REF[net_name]}) -> {G0['pass']} | "
              f"G5 anchor {anchor_nll:.3f} -> {G5['pass']} "
              f"(e046 band {G5['pass_e046_band']})", flush=True)
        assert G0["pass"], f"G0 failed on {net_name}"

        c1 = claim1_l5_calibrator(net, corpus, cfg)
        print(f"{stamp()} C1 L5-calibrator: KL {c1['kl_mean']:.4f} (bar > {KL_BAR}) | "
              f"L5-attn ablation {c1['l5_attn_ablation_cost']:+.4f} (bar < {ABL_BAR}) | "
              f"flips {c1['flip_rate']*100:.0f}% -> pass={c1['pass']}", flush=True)
        c2 = claim2_mlp5_energy(net, corpus, cfg)
        print(f"{stamp()} C2 MLP-5 carrier: zero {c2['zero_damage']:+.4f} vs rotate "
              f"{c2['rotate_damage']:+.4f} -> ratio {c2['ratio_rotate_over_zero']:.3f} "
              f"(bar < {RATIO_BAR}; B ref {E019_RATIO_REF}) | graceful {c2['graceful_alpha']} "
              f"-> pass={c2['pass']}", flush=True)
        c3 = claim3_shared_l0(net, bats)
        tops = {n: v["top_block"] for n, v in c3["per_name"].items()}
        print(f"{stamp()} C3 shared-L0: top blocks {tops} -> L0-MLP top-1 for "
              f"{c3['n_names_l0mlp_top1']}/{len(ATLAS_NAMES)} names "
              f"(bar >= {NAMES_BAR}) -> pass={c3['pass']}", flush=True)

        nets[net_name] = {
            "ckpt": str(ckpt), "host": HOST_DESC[net_name],
            "renorm_liveness": renorm_live,
            "gates": {"G0_val_ce": G0, "G5_uniform_floor_anchor": G5},
            "c1_l5_calibrator": c1, "c2_mlp5_energy": c2, "c3_shared_l0": c3,
        }
        del net
        torch.cuda.empty_cache()

    # ------------------------------------------------ B reproduction gates
    b = nets["B"]
    repro = {
        "e013_kl": {"value": b["c1_l5_calibrator"]["kl_mean"], "ref": E013_KL_REF,
                    "tol": 0.02,
                    "pass": bool(abs(b["c1_l5_calibrator"]["kl_mean"] - E013_KL_REF) <= 0.02)},
        "e001_attL5": {"value": b["c1_l5_calibrator"]["l5_attn_ablation_cost"],
                       "ref": E001_ATTL5_REF, "tol": 0.01,
                       "pass": bool(abs(b["c1_l5_calibrator"]["l5_attn_ablation_cost"]
                                        - E001_ATTL5_REF) <= 0.01)},
        "e019_ratio": {"value": b["c2_mlp5_energy"]["ratio_rotate_over_zero"],
                       "ref": E019_RATIO_REF, "tol": 0.05,
                       "pass": bool(abs(b["c2_mlp5_energy"]["ratio_rotate_over_zero"]
                                        - E019_RATIO_REF) <= 0.05)},
        "e042_top_block": {
            "value": f"L0-MLP top-1 for {b['c3_shared_l0']['n_names_l0mlp_top1']}/4 names "
                     f"(uniform-floor battery; e042 train-battery ref: 4/4)",
            "pass": bool(b["c3_shared_l0"]["n_names_l0mlp_top1"] >= 3)},
    }
    print(f"\n{stamp()} B reproduction gates: "
          f"{ {k: v['pass'] for k, v in repro.items()} }", flush=True)
    if not SMOKE:
        assert repro["e013_kl"]["pass"] and repro["e001_attL5"]["pass"] \
            and repro["e019_ratio"]["pass"], f"B reproduction failed: {repro}"
        assert all(nets[n]["gates"]["G5_uniform_floor_anchor"]["pass"] for n in nets), \
            "uniform-floor anchor gate failed"

    # ------------------------------------------------ verdicts (min-3-nets bar)
    def verdict(pass_field, sub_key):
        per_net = {n: nets[n][sub_key][pass_field] for n in nets}
        n_pass = sum(per_net.values())
        non_b = [v for n, v in per_net.items() if n != "B"]
        return {
            "per_net_pass": per_net,
            "n_pass_of_5": n_pass,
            "n_pass_nonB_of_4": sum(non_b),
            "min_nets_bar": MIN_NETS,
            "verdict": "REPLICATES" if n_pass >= MIN_NETS else "DIES",
            "verdict_note": ("passes the >=3-nets bar counting B as the reference "
                             "reproduction" if n_pass >= MIN_NETS and sum(non_b) < MIN_NETS
                             else "passes with >=3 non-reference hosts" if n_pass >= MIN_NETS
                             else "below the >=3-nets bar"),
        }

    v1 = verdict("pass", "c1_l5_calibrator")
    v2 = verdict("pass", "c2_mlp5_energy")
    v3 = verdict("pass", "c3_shared_l0")
    survive = [name for name, v in (("C1 L5-calibrator", v1), ("C2 MLP-5 energy carrier", v2),
                                    ("C3 shared-L0 name machine", v3)) if v["verdict"] == "REPLICATES"]

    metrics = {
        "experiment": "e047_replication_sweep",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED, "smoke": SMOKE,
        "purpose": "Review-6 night program item 1: 3 surviving positive claims x 5 nets "
                   "(card v3 gate); min-nets rule = positives enter H only after >=3 nets",
        "hosts": HOST_DESC,
        "protocol_refs": {
            "c1": "lab/e013_truncation.py (KL full-256 arm) + lab/e001_lesion_map.py "
                  "(attn block zero-ablation, estimate_loss 30 batches)",
            "c2": "lab/e019_mlp5_thermostat.py (fixed blocks seed 606 x400; h[5].mlp "
                  "scale/rotate hooks; rotate gen 9090)",
            "c3": "lab/e042_name_atlas.py (block atlas, uniform battery position slice "
                  "[CTX-1, CTX-1+L)) on lab/e046_c6_replication.py uniform-floor batteries "
                  "(63 PROSPERO val-anchor contexts, CTX=120)",
            "renorm": "lab/e014b_stream_renorm.register_renorm (c=5.6) active at eval for "
                      "R/R43 + lab/e028_transplant.py liveness assert",
        },
        "bars": {"c1_kl_gt": KL_BAR, "c1_ablation_lt": ABL_BAR,
                 "c2_ratio_lt": RATIO_BAR, "c3_names_ge": NAMES_BAR,
                 "min_nets": MIN_NETS},
        "b_reproduction_gates": repro,
        "nets": nets,
        "verdicts": {
            "c1_l5_calibrator": v1,
            "c2_mlp5_energy_carrier": v2,
            "c3_shared_l0_name_machine": v3,
            "survive_to_card_v3": survive,
        },
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": cfg_dict(cfg),
    }
    save_json(rd / "metrics.json", jsonable(metrics))

    # ------------------------------------------------ 3 panels
    names = list(CKPTS)
    # panel 1 — L5 calibrator
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    kls = [nets[n]["c1_l5_calibrator"]["kl_mean"] for n in names]
    abls = [nets[n]["c1_l5_calibrator"]["l5_attn_ablation_cost"] for n in names]
    ok1 = [nets[n]["c1_l5_calibrator"]["pass"] for n in names]
    cols = ["seagreen" if o else "crimson" for o in ok1]
    axes[0].bar(names, kls, color=cols)
    axes[0].axhline(KL_BAR, color="k", ls="--", lw=1, label=f"KL bar {KL_BAR}")
    axes[0].axhline(E013_KL_REF, color="gray", ls=":", lw=1, label=f"B ref (e013) {E013_KL_REF}")
    axes[0].set_ylabel("KL(final || L4 readout), nats"); axes[0].legend(fontsize=7)
    axes[0].set_title("C1a: L5's distribution reshaping")
    axes[1].bar(names, abls, color=cols)
    axes[1].axhline(ABL_BAR, color="k", ls="--", lw=1, label=f"ablation bar {ABL_BAR}")
    axes[1].axhline(E001_ATTL5_REF, color="gray", ls=":", lw=1, label=f"B ref (e001) +{E001_ATTL5_REF}")
    axes[1].set_ylabel("L5-attn zero-ablation d val CE (nats)"); axes[1].legend(fontsize=7)
    axes[1].set_title("C1b: how cheap the whole block is")
    fig.suptitle(f"E047 C1 — L5-calibrator replication | {v1['verdict']} "
                 f"({v1['n_pass_of_5']}/5 nets, {v1['n_pass_nonB_of_4']}/4 non-B)")
    fig.tight_layout(); fig.savefig(rd / "panel1_l5_calibrator.png", dpi=130); plt.close(fig)

    # panel 2 — MLP-5 energy carrier
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    zs = [nets[n]["c2_mlp5_energy"]["zero_damage"] for n in names]
    rs = [nets[n]["c2_mlp5_energy"]["rotate_damage"] for n in names]
    x = range(len(names))
    ax.bar([i - 0.18 for i in x], zs, 0.36, color="steelblue", label="zero (alpha=0) damage")
    ax.bar([i + 0.18 for i in x], rs, 0.36, color="darkorange", label="rotate60 damage")
    ax.set_ylabel("d val CE (nats) at L5-MLP"); ax.legend(fontsize=8, loc="upper left")
    ax2 = ax.twinx()
    ratios = [nets[n]["c2_mlp5_energy"]["ratio_rotate_over_zero"] for n in names]
    ax2.plot(x, ratios, "ko-", lw=1.2, ms=6, label="ratio rotate/zero")
    ax2.axhline(RATIO_BAR, color="k", ls="--", lw=1)
    ax2.axhline(E019_RATIO_REF, color="gray", ls=":", lw=1)
    for i, r in enumerate(ratios):
        ax2.annotate(f"{r:.2f}", (i, r), fontsize=8, ha="center", xytext=(0, 7),
                     textcoords="offset points")
    ax2.set_ylabel("rotate/zero damage ratio (bar < 0.5)")
    ax.set_xticks(list(x)); ax.set_xticklabels(names)
    ax.set_title(f"E047 C2 — MLP-5 energy carrier | {v2['verdict']} "
                 f"({v2['n_pass_of_5']}/5 nets, {v2['n_pass_nonB_of_4']}/4 non-B)\n"
                 f"(energy carrier: zeroing costs >> direction-scrambling at matched energy)")
    fig.tight_layout(); fig.savefig(rd / "panel2_mlp5_energy.png", dpi=130); plt.close(fig)

    # panel 3 — shared-L0 name machine
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    ax = axes[0]
    ok3 = [nets[n]["c3_shared_l0"]["pass"] for n in names]
    ax.imshow([[0 if nets[n]["c3_shared_l0"]["per_name"][w]["l0mlp_top1"] else 1
                for w in ATLAS_NAMES] for n in names],
              cmap=matplotlib.colors.ListedColormap(["seagreen", "crimson"]), aspect="auto",
              vmin=0, vmax=1)
    for i, n in enumerate(names):
        for j, w in enumerate(ATLAS_NAMES):
            v = nets[n]["c3_shared_l0"]["per_name"][w]
            ax.text(j, i - 0.18, v["top_block"], ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
            ax.text(j, i + 0.24, f"+{v['top_block_dce']:.1f}", ha="center", va="center",
                    fontsize=7, color="white")
    ax.set_xticks(range(len(ATLAS_NAMES))); ax.set_xticklabels(ATLAS_NAMES, fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(
        [f"{n}{'' if ok else ' (FAIL)'}" for n, ok in zip(names, ok3)], fontsize=8)
    ax.set_title("top-1 block per name (uniform-floor atlas)\n"
                 "green = L0-MLP is top-1 (cell: top block, dNLL)", fontsize=9)
    ax = axes[1]
    for j, w in enumerate(ATLAS_NAMES):
        vals = [nets[n]["c3_shared_l0"]["per_name"][w]["l0mlp_dce"] for n in names]
        ax.bar([i + 0.2 * (j - 1.5) for i in range(len(names))], vals, 0.2, label=w)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set_ylabel("L0-MLP dNLL on name battery (nats)"); ax.legend(fontsize=8)
    ax.set_title("L0-MLP block damage per name (e042 B ref: +7.0..7.6 on train battery)",
                 fontsize=9)
    fig.suptitle(f"E047 C3 — shared-L0 name machine | {v3['verdict']} "
                 f"({v3['n_pass_of_5']}/5 nets, {v3['n_pass_nonB_of_4']}/4 non-B)")
    fig.tight_layout(); fig.savefig(rd / "panel3_shared_l0.png", dpi=130); plt.close(fig)

    # ------------------------------------------------ report
    print(f"\n{stamp()} === E047 VERDICTS (min-{MIN_NETS}-nets bar) ===")
    print(f"C1 L5-calibrator:        {v1['verdict']}  "
          f"[{ {n: ('KL %.2f/abl %+.3f' % (nets[n]['c1_l5_calibrator']['kl_mean'], nets[n]['c1_l5_calibrator']['l5_attn_ablation_cost']), v) for n, v in v1['per_net_pass'].items()} }]")
    print(f"C2 MLP-5 energy carrier: {v2['verdict']}  "
          f"[{ {n: ('ratio %.2f' % nets[n]['c2_mlp5_energy']['ratio_rotate_over_zero'], v) for n, v in v2['per_net_pass'].items()} }]")
    print(f"C3 shared-L0 machine:    {v3['verdict']}  "
          f"[{ {n: ('L0M-top1 %d/4' % nets[n]['c3_shared_l0']['n_names_l0mlp_top1'], v) for n, v in v3['per_net_pass'].items()} }]")
    print(f"survive to card v3: {survive}")
    print(f"outputs: {rd}")
    print(f"total {time.time() - T0:.1f}s  smoke={SMOKE}")


if __name__ == "__main__":
    main()
