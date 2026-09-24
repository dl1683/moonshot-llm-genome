# Transfer Width Probe: MC005 / MC003 / MC004

Date: 2026-07-01

Status: preregistered transfer-width packet; no transfer result claimed.

Machine-readable artifact:

> `data/transfer_width_probe_mc005_mc003_mc004.json`

Builder:

> `code/transfer_width_probe_mc005_mc003_mc004.py`

Commands:

```powershell
python code\transfer_width_probe_mc005_mc003_mc004.py --write
python code\transfer_width_probe_mc005_mc003_mc004.py
python code\validate_control_surface_atlas.py
```

## Target Gap

- work order: `run_width_transfer_probe`;
- urgency: `immediate`;
- track: `widening_probe`;
- target gaps: `["transfer_ready_mechanism_absent", "model_family_coverage_qwen_dominant", "full_reliability_absent", "mc005_singleton_internal_causal_reference", "axis_rules_sparse_or_singleton_heavy"]`;
- iteration budget: One bounded/reference row plus one diagnostic row before more deepening..

## Purpose

This packet makes transfer failure or success measurable instead of
assumed. It deliberately pairs one bounded reference specimen with two
diagnostic rows:

- `mc005_associative_lookup`: `bounded_reference_surface`; transfer class `bounded_transfer_fragile_reference`; question: Does source-value lookup mediation transfer with null locality, side rows, source-disjoint holdouts, and prompt robustness, or does the primary effect appear before reliability?
- `mc003_delayed_copy`: `output_shadow_diagnostic_baseline`; transfer class `output_shadow_widening_baseline`; question: Does a delayed-copy lead-time-looking signal remain an output shadow under the same transfer harness?
- `mc004_in_context_binding`: `predecision_monitor_diagnostic_baseline`; transfer class `conditional_widening_requires_new_controls`; question: Does an in-context-binding predecision monitor transfer as a monitor-only surface without becoming a lever?

## Target Models

- primary non-Qwen targets: `["google/gemma-2-2b-it", "google/gemma-2-2b"]`;
- rationale: The transfer matrix already records Qwen-dominant evidence; the first width probe must test a materially comparable non-Qwen model family before any small-model generalization claim.

## Probe Panels

| Panel | Row | Type | Required Measurements | Success Rule | Failure Exports |
| --- | --- | --- | --- | --- | --- |
| `mc005_gemma_primary_lookup_effect` | `mc005_associative_lookup` | `primary_effect` | `["answer-present lookup accuracy on source-disjoint holdout", "source-value source-mask or homologous local intervention effect", "target versus distractor source-row specificity", "late-band coordinate mapping stated by fractional depth, not copied layer numbers"]` | Pass only if the non-Qwen target preserves high answer-present lookup accuracy and the local intervention moves behavior in the MC005-predicted direction without relying on Qwen layer numbers. | `["TRANSFER_PRIMARY_FAILED"]` |
| `mc005_gemma_answer_absent_null_locality` | `mc005_associative_lookup` | `null_locality` | `["answer-absent null rows stratified by target-model margin", "off-target null flips under intervention", "low-margin and high-margin null strata reported separately", "comparison against no-intervention, source-deletion, and neutral-rewrite controls"]` | Pass only if answer-absent null flips stay below the predeclared trivial threshold across strata, holdouts, and intervention variants. | `["TRANSFER_PRIMARY_BEFORE_RELIABILITY", "NULL_ROW_LOW_MARGIN_FLIP_REPLICATES_UNDER_WIDTH"]` |
| `mc005_gemma_side_rows_and_prompt_robustness` | `mc005_associative_lookup` | `side_effects` | `["layout, lexicon, pair-count, and long-context prompt variants", "query-only path comparison", "neutral rewrite comparison", "side-row value and distractor-row effects", "fluency or parseability side effects"]` | Pass only if primary effect, side rows, prompt variants, and fluency remain within predefined tolerances together. | `["TRANSFER_PRIMARY_BEFORE_RELIABILITY", "TRANSFER_SIDE_EFFECT_BOUNDARY"]` |
| `mc005_gemma_prompt_robustness` | `mc005_associative_lookup` | `prompt_robustness` | `["same grammar as source Qwen panel", "Response-marker variant", "neutral marker variant", "source-disjoint prompt-family holdout"]` | Pass only if the effect survives the preregistered prompt-family holdouts without changing the claim boundary. | `["PROMPT_CONTRACT_TRANSFER_FRAGILITY"]` |
| `mc003_delayed_copy_output_shadow_baseline` | `mc003_delayed_copy` | `output_shadow_baseline` | `["behavior table balance", "same-stage output margin baseline", "final output or candidate margin baseline", "shuffled-label selection null", "source-disjoint holdout"]` | This panel is not expected to promote. It passes as a diagnostic baseline if output/candidate or shuffle controls still explain the signal. | `["OUTPUT_SHADOW_REPLICATES_UNDER_WIDTH"]` |
| `mc004_binding_monitor_only_baseline` | `mc004_in_context_binding` | `monitor_only_baseline` | `["pre-update or pre-answer coordinate selected before final answer", "subgroup robustness", "source-disjoint holdout", "shuffled-label selection null", "same-stage and final-stage output/candidate controls"]` | This panel only opens a future intervention route if the monitor beats subgroup, shuffle, same-stage, and final-stage controls. | `["PREDECISION_MONITOR_NO_LEVER_REPLICATES"]` |

## Decision Rules

- promotion rule: Promote a transfer claim only if primary effect, null locality, side effects, prompt-contract robustness, and holdouts all transfer.
- bound rule: Bound if primary effects reproduce but null locality, side rows, or prompt robustness fail.
- kill rule: Kill the transfer route for a surface if two model-family or prompt-family tests reproduce the same reliability failure.
- containment rule: A primary-effect replication is not a transferred mechanism unless reliability and null fields move beyond bounded.
- export rule: Export TRANSFER_PRIMARY_BEFORE_RELIABILITY or TRANSFER_PRIMARY_FAILED as the widening diagnostic.

## Claim Boundary

This packet makes transfer measurable by requiring primary effect, null locality, side rows, prompt robustness, and diagnostic baselines before any transfer field may improve.

This packet does not prove MC005 transfers, does not license MC003 or MC004 interventions, and does not reduce the current count of zero transfer-ready mechanisms.

## What This Proves

It proves that the first transfer-width test is now a branch
contract, not a vague widening wish. A future result cannot count
as transfer unless the primary effect, null locality, side rows,
prompt robustness, and diagnostic baselines are reported together.

## What It Does Not Prove

It does not prove any new model result. It does not move MC005 beyond
bounded transfer-fragile status, and it does not make MC003 or MC004
intervention-ready.
