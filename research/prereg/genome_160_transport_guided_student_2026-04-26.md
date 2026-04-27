# Pre-registration: genome_160 transport-guided student comparison

**Date:** 2026-04-26
**Status:** LOCKED at first commit. CONDITIONAL: launches only after g156, g157, g158, g159 PASS (or are skipped per their conditional rules).
**Author:** Devansh / Neural Genome
**Theory ref:** `research/derivations/prefix_information_transport.md`
**Program ref:** `research/programs/post_g156_pass_program.md` §g160

## 0. End-goal alignment

g160 is the cash-out experiment. The transport theory must do more than explain phenomena — it must SELECT a better matched-cost design. If a transport-heavy student beats a local-heavy student under matched inference FLOPs and matched distillation budget, the theory becomes a model-selection rule for the manifesto end-goal: capability transfer + electricity-grade efficiency.

## 1. Hypothesis

Under matched student inference FLOPs (±2%) and matched distillation budget, a transport-heavy student (deeper, attention-only, hidden-wide) beats a local-heavy student (shallower, MLP-equipped) on:
- C3_macro = mean(HellaSwag, PIQA, Winogrande) accuracy (full validation sets)
- CtQ_90: compute-to-90%-of-final-quality (training-efficiency proxy)

## 2. System

- **Teacher:** Qwen3-0.6B (canonical registry; matches g154 smoke-test teacher)
- **Two students at matched inference FLOPs within ±2%:**
  - **transport-heavy:** `6L_noMLP` Llama, hidden=512, ~50-70M params (no MLP, more depth/width)
  - **local-heavy:** `4L_MLP` Llama, hidden=384, ffn=1024, ~50-70M params (MLP-equipped, shallower)
- Tokenizer: Qwen3 (matches teacher; required for KD)
- Distillation: top-k=64 KD with γ=0.5, T=2.0 (matches g154 protocol; if g154 returns KILL, fall back to γ=1.0 full-CE-skip and rerun this prereg)
- Training data: 8192 C4 train windows (deduped against eval splits)
- Seeds: {42, 7, 13}
- Training steps: matched within ±2% total FLOPs across the two students

## 3. Evaluation protocol

- **C3_macro:** full validation sets of HellaSwag, PIQA, Winogrande — multiple-choice log-likelihood, 0-shot, deterministic, no chat template, no CoT
- **CtQ_90:** compute (FLOPs) needed to reach 90% of own-final C3_macro
- Optional **TEI/kJ:** if wall-power meter is available, also compute TEI/kJ ratio

## 4. Pre-stated criteria (per locked program §g160)

- **PASS:** C3_macro_transport − C3_macro_local ≥ +1.0pp AND CtQ_90_transport ≤ 0.80 × CtQ_90_local in ≥2/3 seeds. If wall-power available: also TEI/kJ_transport / TEI/kJ_local ≥ 1.25.
- **PARTIAL:** C3 gain ≥ +0.5pp OR only the CtQ_90 criterion lands.
- **KILL:** local-heavy ties or wins on C3_macro AND on convergence speed.

## 5. Universality level claimed

**None.** Engineering claim about model selection, not a representational-geometry claim.

## 6. Compute envelope (COMPUTE.md §9)

- VRAM: teacher BF16 (~1.2 GB) + student (~150 MB) + KD top-k cache. Peak < 8 GB. ✓
- RAM: top-k cache for 8192 × 256 × 64 ≈ 270 MB. ✓
- Wall-clock: teacher logit cache ~10 min + 2 students × ~30 min × 3 seeds = ~3 hr. ✓
- Disk: top-k cache + checkpoints < 5 GB.
- Quantization: BF16.

## 7. Conditional launch

Launches when:
- g154 PASS_DISTILL (validates pipeline mechanics)
- g156 PASS_TRANSPORT (theory direction confirmed)
- g157 PASS_G157 or PARTIAL (transport budget criterion observed)
- g158 PASS_INVERSION or PARTIAL (theory predicts inversion)
- g159 PASS or PARTIAL (cross-class generalization observed)

If any of g157/g158/g159 KILL, the theory weakens; g160 is still informative as a direct engineering test, but is no longer "the cash-out for a validated theory" — it becomes "a Llama-only design hypothesis test." Run anyway; framing changes.

## 8. Audit-hard protocol

- All eval datasets full validation splits
- Same harness commit, same prompt templates, same eval order across both students
- FLOP counter logged + matched within ±2% (frozen analytic counter)
- Dedup audit (13-token rolling hash) of C4 train against C3 eval sets

## 9. What a null result means

KILL = transport theory does not select a better matched-cost design. The theory may explain pretraining behavior but does NOT translate to capability-transfer / efficiency design (which IS the manifesto end-goal). Theory becomes "interesting but not load-bearing for the moonshot."

## 10. Artifacts

- `code/genome_160_transport_guided_student.py`
- `results/genome_160_transport_guided_student.json`
- `results/genome_160_run.log`
- `cache/g160_teacher_topk_qwen3-0.6b_n8192.pt`
- Student final checkpoints (gitignored)
- Ledger entry per CLAUDE.md §4.6

## 11. Locking

LOCKED upon commit.
