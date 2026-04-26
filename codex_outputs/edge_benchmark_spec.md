Do not headline `tokens/sec/joule`. Across different tokenizers that number is gameable and an integrity audit will rip it apart. Headline `benchmark items per kJ` or `requests per kJ`, with wall-power as primary and GPU-board power only as secondary.

**A. Top 3 benchmark specs**

1. `C3-TEI/kJ`: Commonsense-3 teacher-equivalent items per kilojoule. This is the one you can lock tomorrow.
- Datasets and metrics: full `allenai/hellaswag` validation, full `ybisk/piqa` validation, full `allenai/winogrande` `winogrande_debiased` validation. Score each by exact multiple-choice log-likelihood, 0-shot, deterministic, no CoT. Primary quality metric: `C3_macro = mean(acc_hellaswag, acc_piqa, acc_winogrande)`. Headline efficiency metric: `TEI/kJ = (C3_macro(model) / C3_macro(Qwen3-8B)) * (N_items_total / kJ_wall)`.
- Baselines: `Qwen3-8B` as teacher anchor, `Qwen3-1.7B`, `Gemma-4-E2B`, `LFM2.5-1.2B-Base`, `Mamba2-780M`. Optional large sanity row: `Llama-3.1-8B`.
- Hardware and power: RTX 5090 laptop on AC, fixed Windows performance mode, no battery discharge. Primary energy = external AC wall meter. Gold standard if you can get it: SPEC/PTDaemon-class meter like Yokogawa WT310E. Practical fallback for internal runs: metering smart plug with logging. Secondary telemetry: `nvidia-smi` average/instant board power and Intel Power Gadget CPU package power. Headline uses wall energy only.
- Target headline if it works: `>=90%` of `Qwen3-8B` `C3_macro`, with `>=4x` `TEI/kJ` versus `Qwen3-8B`, and `>=25%` better `TEI/kJ` than the best non-distilled sub-2B baseline.
- Exact failure mode: the student is just “small-model efficiency,” not capability transfer. Claim dies if `Qwen3-1.7B` or `Gemma-4-E2B` matches the student on both `C3_macro` and `TEI/kJ`, or if the student drops >5pp on any one dataset while the macro average hides it.
- Pre-register against it: full validation splits only, fixed harness commit, fixed prompt templates, fixed evaluation order, no answer extraction, no chat template, same teacher tokenizer for the distilled student, and require per-dataset floors in addition to macro average.

2. `IFEval-TER/kJ`: teacher-equivalent requests per kilojoule on an OpenAI-compatible server. This has the best PR ceiling, but you cannot honestly lock it today without changing the registry and likely the training objective.
- Datasets and metrics: full `google/IFEval`. Primary quality metric: prompt-level strict accuracy. Secondary: instance-level strict accuracy. Primary efficiency metric: `TER/kJ = (prompt_acc(model) / prompt_acc(teacher)) * (completed_requests / kJ_wall)` at the highest concurrency that still satisfies the latency SLO.
- Baselines: `LFM2.5-1.2B-Instruct`, `Gemma-3-4B-IT`, plus a 7B-8B instruct teacher that must be added to `MODEL_DIRECTORY.md` before lock. Without that registry change, this prereg is malformed.
- Hardware and power: same wall-power hierarchy as above. Serve through one OpenAI-compatible codepath for every model. Measure `requests/sec`, `p95 TTFT`, `p95 request latency`, `J/request`.
- Target headline if it works: `>=92%` of teacher prompt-level IFEval at `>=4x` teacher-equivalent requests/kJ, with `p95 TTFT <= 350 ms` at concurrency 4.
- Exact failure mode: the number is really a chat-template or SFT artifact, not distillation. Claim dies if changing template/system prompt materially changes ranking, or if the student only wins by emitting shorter/truncated outputs.
- Pre-register against it: zero hidden system prompt, fixed template family, fixed `temperature=0`, fixed `max_tokens`, truncation rate cap `<1%`, same tokenizer as teacher, and raw IFEval outputs archived.

3. `GSM8K-TEA/kJ`: teacher-equivalent exact answers per kilojoule. Stronger reasoning signal than C3, but much more prompt-fragile.
- Datasets and metrics: full `openai/gsm8k` test set. Primary metric: exact match on final numeric answer with a locked parser. Headline efficiency metric: `TEA/kJ = (EM(model) / EM(Qwen3-8B)) * (N_questions / kJ_wall)`.
- Baselines: `Qwen3-8B`, `Qwen3-1.7B`, `Gemma-4-E2B`, `Phi-4`, `LFM2.5-1.2B-Base`.
- Hardware and power: same wall-power hierarchy. `temperature=0`, answer-only prompt, fixed `max_tokens` to stop CoT length gaming.
- Target headline if it works: `>=85%` of `Qwen3-8B` exact-match at `>=4x` teacher-equivalent answers/kJ.
- Exact failure mode: prompt engineering drives the result more than the model. Claim dies if a standard alternate prompt changes the student/teacher gap by >2pp.
- Pre-register against it: one prompt only, one parser only, no self-consistency, no majority voting, no hidden reasoning channel, and report all parse failures.

**B. Which one to commit to**

Commit to `C3-TEI/kJ`.

Reason: it is the strongest benchmark you can lock tomorrow without lying to yourself. It matches the current repo state, matches the current g154 base-LM KD direction, uses model IDs already in the registry, avoids chat-template and CoT theatrics, and still yields a practitioner-readable claim: “near-8B quality at far better energy efficiency on one laptop.”

`IFEval-TER/kJ` is the better demo once you decide to build an instruct-distilled product and register an instruct teacher. Today it is not clean. Pre-registering it now would be aspiration cosplay.

`GSM8K-TEA/kJ` is good as a secondary figure, not the flagship. Too easy to lose the room in prompt-parser arguments.

**C. Pre-registration template**

File: `research/prereg/genome_155_edge_benchmark_c3_energy_2026-04-26.md`

```md
# Pre-registration: g155 edge benchmark -- C3 teacher-equivalent items per kilojoule

**Date:** 2026-04-26
**Status:** LOCKED at commit where this file is first added.
**Author:** Dev / Neural Genome

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

## 11. Compute envelope checklist

- [ ] Max concurrent VRAM usage <= 22 GB
- [ ] Max system RAM usage <= 56 GB
- [ ] Wall-clock <= 4 h
- [ ] Disk footprint logged
- [ ] Quantization per model logged
- [ ] Save/resume path verified on a smoke test

## 12. Locking

This file is LOCKED at the first commit adding it. Any change after lock invalidates the preregistration and requires a new dated file.
```

Sources: Hugging Face leaderboard task spec for IFEval `https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about`, HellaSwag `https://huggingface.co/datasets/allenai/hellaswag`, WinoGrande `https://huggingface.co/datasets/allenai/winogrande`, PIQA `https://huggingface.co/datasets/ybisk/piqa`, GSM8K `https://huggingface.co/datasets/openai/gsm8k`, NVIDIA GenAI-Perf `https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_benchmark/genai-perf-README.html`, MLPerf power measurement `https://docs.mlcommons.org/inference/power/`, NVIDIA `nvidia-smi` power docs `https://docs.nvidia.com/deploy/nvidia-smi/`.