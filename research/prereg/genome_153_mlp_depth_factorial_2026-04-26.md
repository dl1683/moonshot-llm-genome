# Pre-registration: genome_153 MLP × depth factorial

**Date:** 2026-04-26
**Status:** LOCKED at first commit adding this file.

## Hypothesis (Codex mechanism conjecture)

The minimal_7L architecture's HP-fragility (g149/g150 collapse at lr=1e-3) is driven by **active residual-branch density**, not by MLP-absence per se. At fixed global LR, fewer residual branches means a larger effective per-branch update, which narrows the stable-LR window. Predicted stability ordering:

`14L+MLP (28 branches) > {14L noMLP, 7L+MLP} (14 branches each) > 7L noMLP (7 branches)`

If this holds, the architecture-prior win is a property of the *attention substrate at any branch density*, and increasing depth-without-MLP recovers HP-stability without losing the win. If branch density does NOT explain it (no-MLP arms collapse at any depth), then MLP is doing something special the attention substrate alone cannot.

## End-goal alignment

This is a mechanism test, not a capability test. Knowing *why* MLP-free wins is a prerequisite for the first-principles derivation we owe per CLAUDE.md §0.1 (the move big labs cannot publish). It also informs the g155 production-distillation student design: if branch density is the relevant axis, we may build a 14L noMLP student to widen the stable-LR window.

## System

- Llama-3 backbone, 200M-class (hidden=1024, ffn=2304, 16 heads)
- Tokenizer: Pythia GPT-NeoX (matches g141..g151 line)
- Stimulus bank: `c4_clean_v1`, N_TRAIN=32768, SEQ_LEN=256
- Eval: N_C4_EVAL=200 c4 sequences (in-distribution)
- Seed: 42 (single seed × 24 cells; multi-seed deferred)

## Arms (4 architectures × 6 LRs = 24 cells)

Architectures:
- **A. 14L+MLP** — full Llama-3 baseline (~209M params, 28 residual branches)
- **B. 14L noMLP** — drop MLP, keep depth (~88M params, 14 branches)
- **C. 7L+MLP** — half depth, keep MLP (~110M params, 14 branches → branch-matches B)
- **D. 7L noMLP** — minimal as in g141..g151 line (~81M params, 7 branches)

LR grid: `{2e-4, 3e-4, 4e-4, 6e-4, 8e-4, 1e-3}`. Linear warmup 200 steps on all cells.

Note: these arms differ in parameter count (we are matching steps, not FLOPs, not params). This is an optimization-geometry test, not an efficiency comparison. The capability comparison was already locked at matched-FLOPs in g141..g151.

## Metrics

- **Primary:** critical-LR per arm = highest LR at which final-1000-step mean loss is non-increasing (`warmup_ok` flag)
- **Secondary:** final C4 top-1 at the per-arm best-LR cell
- **Tertiary:** loss curve shape over the 4000 steps (collapse-pattern signature)

## Pre-stated criteria

- **PASS_BRANCH (mechanism = branch density):** Critical-LR ordering is `A ≥ {B ≈ C} > D` with `|critLR(B) - critLR(C)| < 0.5 * |critLR(C) - critLR(D)|`. Arm C ("7L+MLP", branch-matched to B) recovers stability; depth alone is not the story.

- **PASS_MLP (mechanism = MLP-as-special):** Both no-MLP arms (B and D) collapse at the same critical-LR while both MLP arms (A and C) hold stable to a higher LR. Branch density does not explain it; MLP carries something the attention substrate cannot replicate at any depth.

- **AMBIGUOUS:** Critical-LR ordering does not cleanly separate the two hypotheses. Re-design needed.

## Universality level claimed

**None.** Mechanism diagnostic, not a universal claim. Result feeds into MEASUREMENT_PRIMITIVES.md as a diagnostic ("residual-branch density predicts critical-LR") only after replication on at least one second model class.

## What a null result means

If AMBIGUOUS, the next step is either (a) finer-grained branch-density variation (e.g. 14L with MLP on every other layer = 21 branches), or (b) replacing MLP with a parameter-matched attention block to test whether *attention vs MLP* matters at fixed branch count.

## Compute envelope (COMPUTE.md §9 compliance)

- VRAM: 200M-class BF16 + activations at batch=8 ≈ 11 GB peak. ✓
- RAM: tokenized C4 pool < 4 GB. ✓
- Wall-clock: 24 cells × ~5 min ≈ 2 hr. ✓
- Quantization: BF16 throughout. ✓
- Disk: result JSON ~50 KB.
- Checkpointing: not required at per-cell 5-min runtime.
- Single-seed risk: acknowledged. If a cell shows borderline result, follow-up with seed 7 + 13 on that cell only.

## Artifacts

- `code/genome_153_mlp_depth_factorial.py` (already committed 53283a9)
- `results/genome_153_mlp_depth_factorial.json`
- `results/genome_153_run.log`

## Locked at commit

LOCKED upon commit. Modifying hypothesis/arms/criteria invalidates the pre-registration.
