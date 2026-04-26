# Pre-registration: g155 edge benchmark — C3 teacher-equivalent items per kilojoule

**Date:** 2026-04-26
**Status:** LOCKED at commit where this file is first added.
**Author:** Devansh / Neural Genome
**Source:** Codex Competitive-Analyst + Edge-Deployment + Research-Integrity consult, output in `codex_outputs/edge_benchmark_spec.md`.

## 0. End-goal alignment (CLAUDE.md §0)

This benchmark operationalizes the manifesto-aligned outcome of the architecture-prior thread (g138-g154): take a small MLP-free student, distill it from a strong teacher, and demonstrate quality-per-joule advantage on consumer hardware. If H1 holds, this IS the "electricity-grade efficiency on one real task" move identified in CLAUDE.md §0.1 as the kind of result big labs cannot publish for product-conflict reasons.

## 1. Research question

Can a distilled MLP-free student in the 0.3B-1.0B range retain most of an 8B teacher's zero-shot commonsense quality while delivering materially better whole-system energy efficiency on a single RTX 5090 laptop?

## 2. Primary hypothesis

Let:

- `C3_macro(model) = mean(acc_hellaswag, acc_piqa, acc_winogrande)`
- `TEI/kJ(model) = (C3_macro(model) / C3_macro(Qwen3-8B)) * (N_items_total / kJ_wall(model))`

where:
- `acc_hellaswag` is accuracy on full `allenai/hellaswag` validation
- `acc_piqa` is accuracy on full `ybisk/piqa` validation
- `acc_winogrande` is accuracy on full `allenai/winogrande` `winogrande_debiased` validation
- scoring is exact multiple-choice log-likelihood, 0-shot, deterministic
- `kJ_wall(model)` is whole-system AC energy from the first benchmark item to the last benchmark item

**H1:** The g155 distilled student will achieve:
- `C3_macro(student) / C3_macro(Qwen3-8B) >= 0.90`
- `TEI/kJ(student) / TEI/kJ(Qwen3-8B) >= 4.0`
- `TEI/kJ(student) / max(TEI/kJ(best_non_distilled_sub_2B_baseline)) >= 1.25`

## 3. Systems

### Student
- `g155_student` = final distilled MLP-free student artifact produced by genome_155
- Architecture constraint: Llama-family attention-only student, 0.3B-1.0B params
- Tokenizer constraint: must use the teacher tokenizer

### Baselines
- `Qwen3-8B` -- teacher anchor
- `Qwen3-1.7B`
- `Gemma-4-E2B`
- `LFM2.5-1.2B-Base`
- `Mamba2-780M`

### Optional sanity row
- `Llama-3.1-8B`

**Registry note:** Any baseline not currently in `Projects/models/MODEL_DIRECTORY.md` must be added (with quantization tier + VRAM metadata) in a dedicated commit BEFORE this benchmark is run. No silent additions.

## 4. Evaluation protocol

- All evaluations are 0-shot.
- No chain-of-thought prompting.
- No self-consistency.
- No chat template.
- Deterministic scoring only.
- Multiple-choice answers are scored by total continuation log-likelihood.
- Full validation splits are used; no subsets.
- Prompt order is fixed and identical across models.
- Harness commit is pinned at lock time and cannot change afterward.

## 5. Batch and runtime policy

- All models are evaluated through one pinned runtime family and one pinned harness commit.
- Batch size is selected by a fixed policy: largest batch size that passes a 32-item smoke test without OOM and preserves score within 0.1pp of batch size 1 on a 100-item check.
- Quantization is logged per model in the ledger.

## 6. Power measurement hierarchy

### Primary
- External AC wall-power meter with logging.
- Headline claim is only valid if wall-power data exists.

### Secondary
- `nvidia-smi` average and instantaneous GPU board power
- Intel Power Gadget CPU package power

### Reporting rule
- Wall energy is the official metric.
- GPU-only energy may be reported as decomposition, never as the sole headline.

## 7. Outcome categories

### BREAKTHROUGH
All are true:
- `C3_ratio >= 0.90`
- `TEI/kJ_ratio_vs_Qwen3-8B >= 4.0`
- `TEI/kJ_ratio_vs_best_sub_2B_baseline >= 1.25`
- No single dataset gap vs `Qwen3-8B` exceeds 5pp

### PARTIAL
All are true:
- `0.85 <= C3_ratio < 0.90`
- `TEI/kJ_ratio_vs_Qwen3-8B >= 2.5`
- No single dataset gap exceeds 8pp

### FAIL_QUALITY
Any is true:
- `C3_ratio < 0.85`
- Any single dataset gap exceeds 8pp

### FAIL_EFFICIENCY
Any is true:
- `TEI/kJ_ratio_vs_Qwen3-8B < 2.5`
- Student does not beat the best non-distilled sub-2B baseline on `TEI/kJ`

## 8. Explicit kill conditions

The flagship claim is disallowed if any of the following occur:
- `Qwen3-8B` is not at least 5pp ahead of `Qwen3-1.7B` on `C3_macro` in this setup
- Wall-power data is missing
- The student does not use the teacher tokenizer
- Post-lock prompt/template/harness changes are made
- Only macro average holds while one dataset collapses beyond the single-dataset floor

## 9. What a null result means

A null means the architecture-prior result did not translate into a market-grade edge product under honest whole-system energy accounting. It does not rescue the claim to point at params, FLOPs, or GPU-only power if wall-power `TEI/kJ` does not land.

## 10. Artifacts to save

- Raw per-item predictions for all models
- Per-dataset accuracies
- Wall-power log
- GPU-power log
- CPU-package-power log
- Harness commit hash
- Exact prompt templates
- Ledger entry with quantization and batch size

## 11. Compute envelope checklist (COMPUTE.md §9)

- [ ] Max concurrent VRAM usage <= 22 GB (Qwen3-8B at Q4 ≈ 5 GB, 8B at Q5 ≈ 6 GB; eval is single-model at-a-time, max_seq=2048)
- [ ] Max system RAM usage <= 56 GB (HellaSwag full = ~10k items, PIQA = ~2k, Winogrande = ~1.3k; raw data + cache < 8 GB)
- [ ] Wall-clock <= 4 h (estimate: 7 models × ~15 min each = ~2 h end-to-end)
- [ ] Disk footprint logged (raw predictions ~500 MB across all models)
- [ ] Quantization per model logged (Q4_K_M for 7-30B, Q6+ for sub-2B)
- [ ] Save/resume path verified on a smoke test (10-item dry run on every model first)

## 12. Hardware acquisition note

Before running this benchmark, an external AC wall-power meter must be acquired (gold: Yokogawa WT310E; practical: any logging smart plug like Tasmota-flashed Sonoff Pow R3 or Shelly Plug S Plus). Without wall-power, this pre-reg cannot be executed honestly. Acquisition is a prerequisite, not a footnote.

## 13. Locking

This file is LOCKED at the first commit adding it. Any change after lock invalidates the preregistration and requires a new dated file. Reference: `codex_outputs/edge_benchmark_spec.md` (Codex consult of 2026-04-26).
