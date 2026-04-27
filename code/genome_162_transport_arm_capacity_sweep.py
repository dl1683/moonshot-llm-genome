"""
genome_162_transport_arm_capacity_sweep.py

Path A activation post-g158c PASS_canonical (per Codex direction consult
2026-04-27 in `codex_outputs/heartbeats/post_g158c_design_*.md`).

Tests whether the architecture-prior advantage from removing MLP layers is
monotone in transport-arm capacity (number of transport-only layers k).

Pre-reg DRAFT: research/prereg/genome_162_transport_arm_capacity_sweep_2026-04-27.md
Trigger:      g158c PASS_canonical (LOCKED at canonical 3-seed scale 2026-04-27)
Decision tree: research/programs/post_g158c_decision_tree.md Path A

Setup:
  Arms:     baseline_6L+MLP (hidden=384, ffn=1024, ~30M params)
            noMLP at k ∈ {3, 4, 5, 7} (hidden=384, ZeroMLP)
  Contexts: L ∈ {32, 256} (drop intermediate per spec)
  Seeds:    [42] (single seed, parallel to g158 PILOT scope)
  FLOPs:    per-cell matched at 193.27 TFLOP (same as g158c)

Locked PASS / FAIL criteria (Codex direction consult):
  PASS:
    - At L=256: rho(k, Delta_256) >= +0.8 (rank correlation, n=4 transport depths)
    - At L=256: Delta_256(7L) - Delta_256(3L) >= +0.8pp
    - At L=32: Delta_32(7L) <= Delta_32(3L) + 0.1pp
  FAIL:
    - rho(k, Delta_256) < +0.3 OR Delta_256(7L) - Delta_256(3L) < +0.3pp

Results: results/genome_162_transport_arm_capacity_sweep.json
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

# Reuse g158c helpers — no duplication
from genome_158c_3seed_canonical import (  # noqa: E402
    BATCH_SIZE, LR_GRID, LR_WARMUP_STEPS,
    ZeroMLP, make_llama, eval_perplexity_top1,
    warmup_lr, train_arm, tokenize_at_L,
    select_lr_per_arm,
)
from stimulus_banks import c4_clean_v1  # noqa: E402

ROOT = _THIS_DIR.parent

# g162-specific config
SEEDS = [42]  # single-seed pilot per Codex spec
CONTEXT_LENGTHS = [32, 256]  # endpoints only
N_C4_EVAL = 200
N_OOD_EVAL = 200
N_TRAIN_256 = 32768
TARGET_FLOPS_PER_CELL = 193.274e12  # match g158c FLOP budget

# Arms: 1 baseline (6L MLP) + 4 noMLP variants at k ∈ {3, 4, 5, 7}
NOMLP_DEPTHS = [3, 4, 5, 7]


def estimate_flops_per_step(params: int, seq_len: int) -> float:
    """Crude 6 * params * seq_len FLOPs per token, batch_size tokens per step."""
    return 6.0 * params * seq_len * BATCH_SIZE


def main():
    print("genome_162: transport-arm capacity sweep (Path A activation post-g158c PASS)")
    print(f"  noMLP depths: {NOMLP_DEPTHS}; baseline: 6L+MLP")
    print(f"  L: {CONTEXT_LENGTHS}; seeds: {SEEDS}")

    t_start = time.time()
    from transformers import AutoTokenizer
    TOK_ID = "Qwen/Qwen3-0.6B"  # tokenizer only; model is custom Llama
    tok = AutoTokenizer.from_pretrained(TOK_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    vocab = tok.vocab_size

    train_texts_all = c4_clean_v1(seed=77, n=20000, min_len=20)
    val_lr_texts = c4_clean_v1(seed=78, n=2000, min_len=20)
    c4_eval_texts = c4_clean_v1(seed=99, n=N_C4_EVAL, min_len=20)
    ood_eval_texts = c4_clean_v1(seed=999, n=N_OOD_EVAL, min_len=20)

    # Build arm specifications
    def make_arm(name, layers, ffn, no_mlp):
        return {
            "name": name,
            "layers": layers,
            "ffn": ffn,
            "no_mlp": no_mlp,
            "k": layers if no_mlp else None,
        }

    arms = [make_arm("baseline_6L+MLP", layers=6, ffn=1024, no_mlp=False)]
    for k in NOMLP_DEPTHS:
        arms.append(make_arm(f"minimal_{k}L_noMLP", layers=k, ffn=1024, no_mlp=True))

    # Compute n_steps per arm per L for matched FLOPs
    print(f"\n  target total FLOPs/cell: {TARGET_FLOPS_PER_CELL/1e12:.3f} TFLOP")
    arm_steps = {}  # (arm_name, L) -> n_steps
    for arm in arms:
        # Build temp model just to count params
        tmp = make_llama(vocab, hidden=384, layers=arm["layers"], heads=6,
                         ffn=arm["ffn"], no_mlp=arm["no_mlp"], max_pos=320, seed=42)
        n_params = sum(p.numel() for p in tmp.parameters())
        del tmp
        torch.cuda.empty_cache()
        for L in CONTEXT_LENGTHS:
            flops_per_step = estimate_flops_per_step(n_params, L)
            n_steps = int(TARGET_FLOPS_PER_CELL / flops_per_step)
            arm_steps[(arm["name"], L)] = n_steps
            actual_flops = n_steps * flops_per_step
            print(f"  {arm['name']:24s} L={L:3d} n_params={n_params/1e6:.2f}M n_steps={n_steps:6d} actual={actual_flops/1e12:.3f} TFLOP")

    # LR selection per arm: at L=32 and L=256 (then take min)
    print("\n=== LR SELECTION at L=32 and L=256 (min per arm) ===")
    train_ids_32, train_mask_32 = tokenize_at_L(tok, train_texts_all, 32)
    val_ids_32, val_mask_32 = tokenize_at_L(tok, val_lr_texts, 32)
    train_ids_256, train_mask_256 = tokenize_at_L(tok, train_texts_all, 256)
    val_ids_256, val_mask_256 = tokenize_at_L(tok, val_lr_texts, 256)

    arm_lr = {}
    for arm in arms:
        kw = {"hidden": 384, "layers": arm["layers"], "heads": 6, "ffn": arm["ffn"],
              "no_mlp": arm["no_mlp"]}
        lr_at_32 = select_lr_per_arm(
            arm["name"] + "_at_L32", kw, max_pos=32 + 64, vocab=vocab,
            train_ids=train_ids_32, train_mask=train_mask_32,
            val_ids=val_ids_32, val_mask=val_mask_32, n_steps_select=2000,
        )
        lr_at_256 = select_lr_per_arm(
            arm["name"] + "_at_L256", kw, max_pos=256 + 64, vocab=vocab,
            train_ids=train_ids_256, train_mask=train_mask_256,
            val_ids=val_ids_256, val_mask=val_mask_256, n_steps_select=1000,
        )
        chosen = min(lr_at_32, lr_at_256)
        print(f"  {arm['name']}: lr_at_L32={lr_at_32}, lr_at_L256={lr_at_256}, CHOSEN={chosen}")
        arm_lr[arm["name"]] = chosen

    # Main run: per L, per arm, per seed
    results = {str(L): {} for L in CONTEXT_LENGTHS}
    c4_eval_ids = {L: tokenize_at_L(tok, c4_eval_texts, L) for L in CONTEXT_LENGTHS}
    ood_eval_ids = {L: tokenize_at_L(tok, ood_eval_texts, L) for L in CONTEXT_LENGTHS}

    for L in CONTEXT_LENGTHS:
        print(f"\n=== L={L} ===")
        train_ids, train_mask = tokenize_at_L(tok, train_texts_all, L)
        for arm in arms:
            results[str(L)][arm["name"]] = {}
            for seed in SEEDS:
                n_steps = arm_steps[(arm["name"], L)]
                print(f"\n  -- {arm['name']} L={L} seed={seed} n_steps={n_steps} --")
                model = make_llama(vocab, hidden=384, layers=arm["layers"], heads=6,
                                   ffn=arm["ffn"], no_mlp=arm["no_mlp"],
                                   max_pos=L + 64, seed=seed).to("cuda")
                lr = arm_lr[arm["name"]]
                n_total, t_arm, nan_seen = train_arm(arm["name"], lr, model, train_ids, train_mask, n_steps, seed)
                c4_metrics = eval_perplexity_top1(model, *c4_eval_ids[L])
                ood_metrics = eval_perplexity_top1(model, *ood_eval_ids[L])
                results[str(L)][arm["name"]][str(seed)] = {
                    "c4": c4_metrics,
                    "ood": ood_metrics,
                    "wallclock_s": t_arm,
                    "nan_seen": nan_seen,
                    "n_steps": n_steps,
                }
                print(f"    c4 top1={100*c4_metrics['top1_acc']:.2f}%  ood top1={100*ood_metrics['top1_acc']:.2f}%")
                del model
                torch.cuda.empty_cache()

    # Analysis: per L, compute Delta(k, L) = top1(noMLP_k, L) - top1(baseline, L)
    print("\n=== ANALYSIS ===")
    deltas = {}  # L -> {k: delta_pp_c4}
    for L in CONTEXT_LENGTHS:
        deltas[L] = {}
        baseline_top1 = results[str(L)]["baseline_6L+MLP"]["42"]["c4"]["top1_acc"]
        for arm in arms:
            if not arm["no_mlp"]:
                continue
            k = arm["k"]
            arm_top1 = results[str(L)][arm["name"]]["42"]["c4"]["top1_acc"]
            d_pp = (arm_top1 - baseline_top1) * 100
            deltas[L][k] = d_pp
            print(f"  L={L:3d} k={k}: Delta_c4 = {d_pp:+.2f}pp")

    # Compute rho(k, Delta_256)
    ks_256 = sorted(deltas[256].keys())
    rho_256, _ = spearmanr(ks_256, [deltas[256][k] for k in ks_256])
    delta_7L_256 = deltas[256][7]
    delta_3L_256 = deltas[256][3]
    delta_7L_32 = deltas[32][7]
    delta_3L_32 = deltas[32][3]

    margin_256 = delta_7L_256 - delta_3L_256
    margin_32 = delta_7L_32 - delta_3L_32

    print(f"\n  rho(k, Delta_256_c4) = {rho_256:+.3f}")
    print(f"  Delta_256(7L) - Delta_256(3L) = {margin_256:+.2f}pp")
    print(f"  Delta_32(7L)  - Delta_32(3L)  = {margin_32:+.2f}pp")

    # Verdict (Codex-locked thresholds)
    if rho_256 >= 0.8 and margin_256 >= 0.8 and margin_32 <= 0.1:
        verdict = (f"PASS: rho={rho_256:+.2f}, margin_256={margin_256:+.2f}pp, "
                   f"margin_32={margin_32:+.2f}pp. Dose-response architecture intervention "
                   "validated. Theory's transport-budget law holds.")
    elif rho_256 < 0.3 or margin_256 < 0.3:
        verdict = (f"FAIL: rho={rho_256:+.2f} or margin_256={margin_256:+.2f}pp insufficient. "
                   "Transport story does not extend to dose-response. 3L-vs-6L was a brittle binary.")
    else:
        verdict = (f"PARTIAL: rho={rho_256:+.2f}, margin_256={margin_256:+.2f}pp. "
                   "Direction consistent but not all thresholds clear; consider expanding seeds.")

    print(f"\n  VERDICT: {verdict}")

    out = {
        "genome": 162,
        "name": "transport_arm_capacity_sweep",
        "config": {
            "seeds": SEEDS,
            "context_lengths": CONTEXT_LENGTHS,
            "nomlp_depths": NOMLP_DEPTHS,
            "lr_grid": LR_GRID,
            "lr_select_L_actual_policy": "min(lr_at_L32, lr_at_L256)",
            "arm_lr": arm_lr,
            "n_train_256": N_TRAIN_256,
            "warmup_steps": LR_WARMUP_STEPS,
        },
        "results": results,
        "deltas_c4": {str(L): deltas[L] for L in CONTEXT_LENGTHS},
        "rho_256_c4": rho_256,
        "margin_256_pp": margin_256,
        "margin_32_pp": margin_32,
        "verdict": verdict,
        "elapsed_s": time.time() - t_start,
    }

    out_path = ROOT / "results" / "genome_162_transport_arm_capacity_sweep.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}  ({time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
