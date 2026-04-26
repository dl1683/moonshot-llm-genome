# Pre-registration: genome_156 prefix-destruction at 200M

**Date:** 2026-04-26
**Status:** LOCKED at first commit adding this file.
**Author:** Devansh / Neural Genome
**Theory ref:** `research/derivations/prefix_information_transport.md`
**Codex source:** `codex_outputs/first_principles_derivation.md`

## 0. End-goal alignment (CLAUDE.md §0)

This is the breakthrough-axis experiment per CLAUDE.md §0.1. It tests the first-principles derivation Codex identified (Prefix-Information Transport) by running its prescribed killer experiment. If the architecture-prior win survives prefix destruction, the theory is wrong. If it collapses as predicted, we have a derivation route a big lab cannot publish without contradicting their "bigger MLP = better" product story.

## 1. Hypothesis

The architecture-prior win observed in g147/g151 (minimal_7L noMLP beats baseline_14L+MLP at matched-budget on 200M scale) is caused by superior allocation of parameters to prefix-information transport. Specifically: when ordered prefix information is destroyed by per-sequence token permutation, the minimal arm should LOSE its advantage.

## 2. System

- Same 200M family as g147/g151
- **baseline:** `14L + MLP`, hidden=1024, ffn=2304, ~209M params, 4000 steps
- **minimal:** `7L noMLP`, hidden=1024, no ffn (ZeroMLP), ~81M params, 8000 steps
- Arm-specific best LRs from g151: baseline=2e-4, minimal=3e-4
- Linear LR warmup 200 steps
- SEQ_LEN=256, N_TRAIN=32768, BATCH=8
- Seeds: {42, 7, 13}
- Tokenizer: Pythia GPT-NeoX (matches g141..g151 line)

## 3. Stimulus conditions

1. **`natural`:** standard `c4_clean_v1` token sequences (same as g147)
2. **`token_shuffled`:** for each sequence, tokenize once, then apply ONE fixed random permutation per sequence. Token multiset and length preserved; prefix order destroyed. The shuffle seed is locked at 42 and used identically across all training seeds and arms (same shuffled corpus seen by every model).

Both conditions train and evaluate on parallel datasets — natural-trained eval on natural, shuffled-trained eval on shuffled. We do NOT cross-test (no train-on-natural, eval-on-shuffled), because the question is whether the ARCHITECTURE-PRIOR effect depends on data structure, not whether models trained on shuffled generalize to natural.

## 4. Metrics

Per arm × condition × seed: final C4 top-1 (or shuffled-top-1) and final eval NLL after the arm's full step budget.

- `Δ_nat = top1_minimal_natural − top1_baseline_natural` (mean across 3 seeds)
- `Δ_shuf = top1_minimal_shuffled − top1_baseline_shuffled` (mean across 3 seeds)
- **Support statistic:** `C := Δ_nat − Δ_shuf`

## 5. Pre-stated criteria

- **PASS_TRANSPORT** (theory supported): `Δ_nat ≥ +0.5pp` AND `Δ_shuf ≤ +0.1pp` AND `C ≥ +0.4pp`. The architecture-prior win exists in the natural regime and collapses (or reverses) under prefix destruction.

- **PARTIAL_TRANSPORT:** `Δ_nat ≥ +0.3pp` AND `Δ_shuf ≤ +0.2pp` AND `C ≥ +0.2pp`. Directionally consistent with theory but signal weaker than expected; revise probe scale.

- **KILL_TRANSPORT:** `|Δ_nat − Δ_shuf| ≤ 0.2pp`, i.e. the win persists (or fails to manifest) similarly in both conditions. The transport theory is badly damaged. The architecture-prior win then must come from something other than ordered-context transport.

## 6. Universality level claimed

**None.** This is a mechanism test for a single-architecture-family claim (Llama-3 derivatives). A PASS would promote the *theory* to "Codex-validated derivation skeleton" but not yet to a Level-1 atlas claim. Cross-architecture extension (Mamba, RWKV) is future work documented in the derivation doc §future.

## 7. Compute envelope (COMPUTE.md §9)

- VRAM: same as g147 (200M BF16 + activations at batch=8) ≈ 11 GB peak. ✓
- RAM: 2× tokenized c4 pool (natural + shuffled) ≈ 8 GB. ✓
- Wall-clock: 2 conditions × 2 arms × 3 seeds = 12 runs. baseline=4000 steps, minimal=8000 steps. Average ~5 min per run on RTX 5090 → ~1 hour total. ✓
- Disk: result JSON + tokenized cache ≈ 4 GB. ✓
- Quantization: BF16 throughout. ✓
- Checkpointing: not required at per-run 5-min runtime.

## 8. Implementation notes

- Reuse the `make_minimal()` and `make_baseline()` factories from `code/genome_151_arm_specific_lr.py`.
- Tokenize c4_clean_v1 once with Pythia tokenizer, then for `token_shuffled` apply `np.random.RandomState(42).permutation(SEQ_LEN)` to each row independently (one shuffle seed per row — but same shuffle pattern across training seeds for fairness). NOT random per epoch; the shuffled corpus is FROZEN at file-creation time.
- Save shuffled corpus to disk so the experiment is reproducible across reruns.
- Eval split is drawn from a separately-tokenized 200-sequence held-out slice; same shuffle protocol applied for the shuffled eval.
- All 12 runs sequential on single GPU; no parallel runs.

## 9. What a null result means

- **KILL_TRANSPORT** would mean the architecture-prior win is NOT primarily a prefix-information-transport effect. Candidates that survive in that case: rate-distortion (Candidate 3, partially independent of order), operator-energy local term (Candidate 1's local part). Theoretical work pivots to those.
- It would NOT kill the empirical architecture-prior thesis (C10-C13 stand independently). It would only kill the proposed mechanistic explanation.

## 10. Artifacts to save

- `code/genome_156_prefix_destruction_200m.py`
- `results/genome_156_prefix_destruction_200m.json` with per-(arm × condition × seed) metrics
- `results/genome_156_run.log` (PYTHONUNBUFFERED stdout)
- Tokenized + shuffled corpus saved as `cache/c4_shuffled_seed42_pythia_n32768.pt`
- Ledger entry per CLAUDE.md §4.6

## 11. Locking

LOCKED upon commit. Modifying hypothesis, conditions, criteria, or thresholds invalidates the pre-registration.
