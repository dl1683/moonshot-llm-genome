"""E041 — dW ceiling null: same-init / different-DATA-ORDER replicate.

Pays T010 C3's top debt: "dW ceiling null (same-init/diff-data-order) unrun"
— the missing rung that scales the +0.152 same-init dW alignment (e029).
Question: how aligned are two runs that share the EXACT init (seed 42) and
differ ONLY in the order/composition of training batches?

BDO (Batch-Data-Order arm): set_seed(42) -> TinyGPT(cfg) (bitwise-identical
init to B/e001; asserted against the reconstructed init42 snapshot), but the
training corpus is CharCorpus(seed=7777). In this harness (common.py
train_model), the batch generator is `torch.Generator().manual_seed(
corpus.seed)`, so corpus seed alone changes every training batch while the
init is untouched. Config otherwise identical to e001 (steps 4000, lr 1e-3,
bs 64) at the 252 s cap with resumable ckpt (e029 R43 discipline).

Observable (e029 protocol, verbatim organ surgery units): per organ at
L0/L3/L5 x {attn, mlp}, dW = W_trained - W_init (init42), flattened float32
concatenation on device; cos(dW_B, dW_BDO) per organ, mean over 6 organs =
the CEILING of same-init dW alignment.

Ladder (cos dW, mean over the same 6 organs):
  same-init SAME order     1.0 trivially (B vs B, identical tensors)
  same-init DIFF order     CEILING (this run: B vs BDO)
  same-init DIFF regime    0.1412 B<->R / 0.1522 pooled same-init (e029)
  diff-init                -0.0024 ~ 0 (e029, 4 diff-init pairs)

REGISTERED VERDICT (task instruction): ceiling >= 0.5 -> the B<->R 0.15
alignment is a small fraction of what same-init allows (anchoring PARTIAL);
ceiling ~ 0.2 -> B<->R is essentially AT the ceiling (batch-order noise is
the only room left; anchoring TOTAL). Reported as fraction
same_init_diff_regime / ceiling.

Deviations / slack choices (lab discipline):
- 252 s cap (task instruction / e029 r43 protocol); e001 originally ran
  240 s — its exact stopping step was not retained, so step parity is
  approximated by the identical time budget; val-CE parity gate
  (<= 1.7224, e029's gate) is the acceptance check instead.
- Parity CE is measured on the FIXED 30 val batches of the seed-1337 corpus
  (e029 per_batch_losses protocol), identical batches for B (reference
  1.622391 from e029) and BDO.
- e029 rungs are read from runs/e029/metrics.json; hardcoded fallback
  constants if that file is missing.
- No NOTES/THINKING/QUEUE/STATE edits, no git commit (per instructions).

Run: python lab/e041_ceiling_null.py
"""
from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, cfg_dict, now_iso,
                    run_dir, save_json, set_seed, train_model)

CORPUS = REPO / "data" / "input.txt"
E001_CKPT = REPO / "runs" / "checkpoints" / "e001.pt"      # B  (seed42 base)
BDO_CKPT = REPO / "runs" / "checkpoints" / "e041_bdo.pt"   # this run, final
BDO_TRAIN = REPO / "runs" / "checkpoints" / "e041_bdo.train.pt"  # resumable
E029_METRICS = REPO / "runs" / "e029" / "metrics.json"

INIT_SEED = 42          # B's init lineage (e001; e029 convention)
B_BATCH_SEED = 1337     # e001's corpus seed => its batch order
BDO_BATCH_SEED = 7777   # ONLY change vs e001: batch order
SITES = (0, 3, 5)
KINDS = ("attn", "mlp")
N_EVAL = 30
B_CE_REF = 1.622391394774119     # e029 30-batch protocol, B on seed-1337
PARITY_GATE = 1.7224            # e001 val + 0.10 (e029 gate)
E029_FALLBACK = {"same_init_regime_B_R": 0.141176, "same_init_regime_pooled": 0.152189,
                 "diff_init_pooled": -0.002434,
                 "same_init_organs": [0.207, 0.249, 0.048, 0.090, 0.070, 0.183],
                 "diff_init_organs": [0.001, 0.001, 0.003, -0.013, 0.001, -0.001]}
T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():6.1f}s] {msg}", flush=True)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


# ------------------------------------------------ organ surgery units (e029, verbatim)
def mlp_keys(i: int) -> list[str]:
    return [f"h.{i}.mlp.0.weight", f"h.{i}.mlp.0.bias",
            f"h.{i}.mlp.2.weight", f"h.{i}.mlp.2.bias"]


def attn_keys(i: int) -> list[str]:
    return [f"h.{i}.attn.c_attn.weight", f"h.{i}.attn.c_proj.weight"]


def organ_keys(site: int, kind: str) -> list[str]:
    return mlp_keys(site) if kind == "mlp" else attn_keys(site)


def snapshot(model) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


# ------------------------------------------------ fixed 30-batch eval (e029, verbatim)
@torch.no_grad()
def per_batch_losses(model, corpus, n_batches: int = N_EVAL) -> list[float]:
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


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12))


def dw_vec(sd: dict, init: dict, keys: list[str]) -> torch.Tensor:
    parts = [(sd[k].float() - init[k].float()).reshape(-1) for k in keys]
    return torch.cat(parts)


# ---------------------------------------------------------------- main
def main():
    rd = run_dir("e041")
    log("E041 dW ceiling null — same init (seed42), different data order (corpus seed 7777)")
    corpus_bdo = CharCorpus(CORPUS, seed=BDO_BATCH_SEED)   # train batches: 7777
    corpus_eval = CharCorpus(CORPUS, seed=B_BATCH_SEED)    # parity eval: 1337 (e029 protocol)
    cfg = Cfg(vocab=corpus_bdo.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)

    # ---- init42 (authoritative, e029 convention) ----
    set_seed(INIT_SEED)
    init = snapshot(TinyGPT(cfg).to(DEVICE))
    log("init42 snapshot built (e029 convention; B and BDO share it bitwise)")

    # ---- B (e001) ----
    B = TinyGPT(cfg).to(DEVICE)
    B.load_state_dict(torch.load(E001_CKPT, map_location=DEVICE, weights_only=True))
    base_B = per_batch_losses(B, corpus_eval)
    ce_B = mean(base_B)
    log(f"B base CE {ce_B:.6f} (e029 reference {B_CE_REF:.6f})")
    if abs(ce_B - B_CE_REF) > 0.02:
        raise SystemExit(f"B base CE {ce_B:.4f} far from e029 reference — wrong checkpoint?")

    # ---- BDO: train or load ----
    bdo_steps, bdo_hist = None, []
    if BDO_CKPT.exists():
        bdo = TinyGPT(cfg).to(DEVICE)
        bdo.load_state_dict(torch.load(BDO_CKPT, map_location=DEVICE, weights_only=True))
        log(f"BDO found at {BDO_CKPT.name} — skipping training")
    else:
        set_seed(INIT_SEED)
        bdo = TinyGPT(cfg).to(DEVICE)
        now = bdo.state_dict()
        bad = [k for k in init if not torch.equal(now[k], init[k])]
        if bad:
            raise SystemExit(f"BDO INIT NOT BITWISE-IDENTICAL to init42: {bad[:5]} — "
                             "the ceiling reading would be invalid.")
        log("BDO init gate: bitwise == init42 (same init as B) PASS")
        log("training BDO (steps 4000, lr 1e-3, bs 64, cap 252 s, corpus seed 7777, resumable ckpt)")
        bdo_hist = train_model(bdo, corpus_bdo, steps=4000, lr=1e-3, batch_size=64,
                               max_seconds=252.0, ckpt=BDO_TRAIN)
        bdo_steps = bdo_hist[-1]["step"] if bdo_hist else 0
        torch.save(bdo.state_dict(), BDO_CKPT)
        log(f"BDO trained ({bdo_steps} steps); final ckpt saved")

    base_BDO = per_batch_losses(bdo, corpus_eval)
    ce_BDO = mean(base_BDO)
    parity_pass = ce_BDO <= PARITY_GATE
    log(f"BDO base CE {ce_BDO:.6f} on the SAME 30 batches (parity gate <= {PARITY_GATE}: "
        f"{'PASS' if parity_pass else 'FLAG'}; BDO {ce_BDO:.4f} vs B {ce_B:.4f}, "
        f"delta {ce_BDO - ce_B:+.4f})")

    # ---- dW ceiling (e029 protocol organs) ----
    sd_B, sd_BDO = snapshot(B), snapshot(bdo)
    organs = [(s, k) for s in SITES for k in KINDS]
    per_organ = {}
    for s, k in organs:
        key = f"L{s}|{k}"
        d_B = dw_vec(sd_B, init, organ_keys(s, k))
        d_BDO = dw_vec(sd_BDO, init, organ_keys(s, k))
        per_organ[key] = {
            "cos_dW_B_vs_BDO": cos(d_B, d_BDO),
            "norm_dW_B": float(d_B.norm()),
            "norm_dW_BDO": float(d_BDO.norm()),
            "norm_ratio_BDO_over_B": float(d_BDO.norm() / d_B.norm()),
        }
        log(f"  {key:8s} cos {per_organ[key]['cos_dW_B_vs_BDO']:+.4f} | |dW_B| "
            f"{per_organ[key]['norm_dW_B']:7.3f} | |dW_BDO| "
            f"{per_organ[key]['norm_dW_BDO']:7.3f} "
            f"(x{per_organ[key]['norm_ratio_BDO_over_B']:.2f})")
    ceiling = mean([v["cos_dW_B_vs_BDO"] for v in per_organ.values()])
    log(f"CEILING cos(dW_B, dW_BDO) mean over 6 organs: {ceiling:+.4f}")

    # identity rung (trivial but computed, not asserted)
    ident = [cos(dw_vec(sd_B, init, organ_keys(s, k)), dw_vec(sd_B, init, organ_keys(s, k)))
             for s, k in organs]
    log(f"identity rung cos(dW_B, dW_B) max dev from 1.0: {abs(max(ident) - 1.0):.2e}")

    # ---- e029 rungs ----
    if E029_METRICS.exists():
        e029 = json.loads(E029_METRICS.read_text(encoding="utf-8"))
        br = e029["dw_alignment"]["per_pair"]["B<->R"]
        ladder_regime_br = br["mean"]
        ladder_regime_organs = list(br["organs"].values())
        ladder_regime_pooled = e029["dw_alignment"]["same_init"]["mean"]
        di = e029["dw_alignment"]["diff_init"]
        ladder_diffinit = di["mean"]
        di_pair = e029["dw_alignment"]["per_pair"]["B<->B43"]
        ladder_diffinit_organs = list(di_pair["organs"].values())
        e029_source = "runs/e029/metrics.json"
    else:
        ladder_regime_br = E029_FALLBACK["same_init_regime_B_R"]
        ladder_regime_pooled = E029_FALLBACK["same_init_regime_pooled"]
        ladder_diffinit = E029_FALLBACK["diff_init_pooled"]
        ladder_regime_organs = E029_FALLBACK["same_init_organs"]
        ladder_diffinit_organs = E029_FALLBACK["diff_init_organs"]
        e029_source = "hardcoded fallback constants"

    ladder = {
        "same_init_same_order": {"value": 1.0, "basis": "trivial: B vs B, identical tensors "
                                                      f"(computed identity check {mean(ident):.6f})"},
        "same_init_diff_order": {"value": ceiling, "basis": "B vs BDO (this run)"},
        "same_init_diff_regime": {"value": ladder_regime_br, "basis": "B<->R (e029)",
                                  "pooled_same_init": ladder_regime_pooled},
        "diff_init": {"value": ladder_diffinit, "basis": "pooled 4 diff-init pairs (e029)"},
    }
    fraction = ladder_regime_br / ceiling if abs(ceiling) > 1e-9 else None
    if ceiling >= 0.5:
        verdict = ("PARTIAL ANCHORING — ceiling >= 0.5: the B<->R 0.15 alignment is a small "
                   f"fraction ({fraction:.2f}) of the same-init alignment the data allows")
    elif fraction is not None and fraction >= 0.6:
        verdict = ("TOTAL ANCHORING — ceiling ~ same magnitude as B<->R: same-init trajectories "
                   f"are as similar as batch-order noise allows (B<->R is {fraction:.0%} of "
                   "ceiling)")
    else:
        verdict = (f"INTERMEDIATE — ceiling {ceiling:.3f}; B<->R is {fraction:.2f} of ceiling "
                   "(neither rung dominates)")
    log("=" * 78)
    log("LADDER cos(dW): same-init-same-order 1.0 | same-init-DIFF-ORDER (ceiling) "
        f"{ceiling:+.4f} | same-init-diff-regime (B<->R) {ladder_regime_br:+.4f} "
        f"| diff-init {ladder_diffinit:+.4f}")
    log(f"VERDICT: {verdict}")

    # ------------------------------------------------ outputs
    metrics = {
        "experiment": "e041_ceiling_null",
        "date": now_iso(),
        "device": DEVICE,
        "task": "T010 C3 debt: dW ceiling null (same-init / diff-data-order replicate)",
        "config": cfg_dict(cfg),
        "arms": {
            "B": {"init_seed": 42, "batch_seed": B_BATCH_SEED, "ckpt": E001_CKPT.name,
                  "ce_30batch": ce_B, "ref": B_CE_REF},
            "BDO": {"init_seed": 42, "batch_seed": BDO_BATCH_SEED, "ckpt": BDO_CKPT.name,
                    "steps": bdo_steps, "ce_30batch": ce_BDO,
                    "parity_gate": PARITY_GATE, "parity_gate_pass": bool(parity_pass),
                    "ce_delta_vs_B": ce_BDO - ce_B,
                    "train_config": "steps 4000, lr 1e-3, bs 64, cap 252 s, "
                                    "corpus CharCorpus(seed=7777) => ONLY batch order differs"},
        },
        "init_gate": "BDO init bitwise == init42 (set_seed(42); TinyGPT(cfg)); "
                     "batch order changed via corpus.seed=7777 (train_model's torch.Generator "
                     "is seeded from corpus.seed)",
        "eval_protocol": "30 fixed deterministic val batches, generator seeded from corpus "
                         "seed 1337 (e029 per_batch_losses); identical batches for B and BDO",
        "dw_ceiling": {
            "rule": "cos((W_B - W_init42), (W_BDO - W_init42)) per organ, flattened float32; "
                    "organs = e029 surgery units (attn: c_attn+c_proj; mlp: fc weights+biases) "
                    "at L0/L3/L5",
            "per_organ": per_organ,
            "mean": ceiling,
        },
        "ladder": ladder,
        "ladder_source": e029_source,
        "fraction_regime_over_ceiling": fraction,
        "verdict": verdict,
        "verdict_rule": "ceiling >= 0.5 -> anchoring PARTIAL (B<->R is a small fraction of "
                        "possible same-init alignment); ceiling ~ 0.2 with B<->R 0.15-0.25 -> "
                        "anchoring TOTAL (B<->R essentially at ceiling)",
        "timing_s": {"total": round(elapsed(), 1)},
    }
    save_json(rd / "metrics.json", metrics)

    # ---- one-panel ladder bar figure ----
    rungs = [("same-init\nsame order\n(B vs B)", ladder["same_init_same_order"]["value"], "#4c9a2a"),
             ("same-init\nDIFF data order\n(B vs BDO) = CEILING", ceiling, "#b8860b"),
             ("same-init\ndiff regime\n(B vs R, e029)", ladder_regime_br, "#2c6fbb"),
             ("diff init\n(B vs B43, e029)", ladder_diffinit, "#888888")]
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    xs = np.arange(len(rungs))
    for i, (lab, val, col) in enumerate(rungs):
        ax.bar(i, val, 0.62, color=col, alpha=0.88, zorder=2)
        ax.text(i, val + (0.03 if val >= 0 else -0.06), f"{val:+.3f}", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=11, fontweight="bold")
    # per-organ scatter for the three computed/measured rungs
    for i, vals in ((1, [per_organ[f"L{s}|{k}"]["cos_dW_B_vs_BDO"] for s, k in organs]),
                    (2, ladder_regime_organs), (3, ladder_diffinit_organs)):
        rng = np.random.default_rng(41 + i)
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(vals)), vals, s=22, color="k",
                   alpha=0.55, zorder=3, label="per-organ values" if i == 1 else None)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(ceiling, color="#b8860b", lw=1.2, ls="--", alpha=0.8)
    ax.text(3.42, ceiling, f"ceiling {ceiling:+.3f}", fontsize=9, color="#b8860b",
            va="center", ha="left")
    ax.axhline(ladder_regime_br, color="#2c6fbb", lw=1.0, ls=":", alpha=0.8)
    ax.text(3.42, ladder_regime_br - 0.05, f"B<->R {ladder_regime_br:+.3f}", fontsize=9,
            color="#2c6fbb", va="center", ha="left")
    ax.set_xticks(xs, [r[0] for r in rungs], fontsize=9)
    ax.set_ylabel("cos(dW_a, dW_b)  —  mean over 6 organs (L0/L3/L5 x attn/mlp)")
    frac_s = "n/a" if fraction is None else f"{fraction:.0%}"
    ax.set_title("E041 dW-alignment ladder: how much room does batch order leave?\n"
                 f"ceiling {ceiling:+.3f} | B<->R is {frac_s} of ceiling | "
                 f"verdict: {verdict.split(' — ')[0]}", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(rd / "ceiling_ladder.png", dpi=140)
    plt.close(fig)

    log(f"BDO steps {bdo_steps} | val CE B {ce_B:.4f} vs BDO {ce_BDO:.4f} "
        f"(delta {ce_BDO - ce_B:+.4f}) | |dW| norms per organ in metrics.json")
    log(f"outputs: {rd}")
    return metrics


if __name__ == "__main__":
    main()
