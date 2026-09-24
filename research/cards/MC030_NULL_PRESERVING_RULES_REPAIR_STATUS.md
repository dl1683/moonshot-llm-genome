# MC030 Null-Preserving Rules Repair Status

Status: null_preserving_rules_branch_null_tradeoff_persists.

Observed pattern: `branch_preserved_null_not_repaired`.

Date: 2026-07-01

## Artifact

- runner:
  `code/mc030_null_preserving_rules_repair.py`
- result:
  `results/cards/MC030/mc030_null_preserving_rules_repair_behavior_20260701T175942.json`

## Verdict

MC030 tests whether MC029's rules-only branch gain can keep null reliability.
It does not establish a hidden signature, intervention, or mechanism card.

## Gate Criteria

| Criterion | Value |
| --- | --- |
| `smoke_mode` | `false` |
| `structural_passed` | `true` |
| `full_source_count_is_40` | `true` |
| `source_disjoint_holdout` | `true` |
| `prompts_have_no_status_lexemes` | `true` |
| `target_atomic_hidden_in_conflicts` | `true` |
| `answer_absent_omits_query_local_row` | `true` |
| `variant_gate_by_template` | `{"decision_order_guard_after_query": false, "decision_order_guard_query_last": false, "no_memory_for_absent_guard": false, "row_absence_guard_before_rules": false, "rules_only_baseline": false}` |
| `any_variant_behavior_gate_passed` | `false` |
| `baseline_operation_atomic_rate` | `0.875` |
| `baseline_other_number_rate` | `0.100` |
| `baseline_answer_absent_unknown_rate` | `0.806` |
| `selected_template` | `rules_only_baseline` |
| `selected_template_operation_atomic_rate` | `0.875` |
| `selected_template_other_number_rate` | `0.100` |
| `selected_template_answer_absent_unknown_rate` | `0.806` |
| `max_operation_atomic_template` | `rules_only_baseline` |
| `max_operation_atomic_rate` | `0.875` |
| `max_operation_atomic_answer_absent_unknown_rate` | `0.806` |
| `max_answer_absent_unknown_template` | `rules_only_baseline` |
| `max_answer_absent_unknown_rate` | `0.806` |
| `best_null_template_operation_atomic_rate` | `0.875` |
| `min_other_number_template` | `decision_order_guard_query_last` |
| `min_other_number_rate` | `0.056` |
| `min_other_number_template_operation_atomic_rate` | `0.487` |
| `candidate_and_output_margins_reported` | `false` |

## Variant Comparison

| Variant | Atomic Control | Local Branch | Atomic Branch | Other Number | Rule Null | Answer Null | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `rules_only_baseline` | 1.000 | 0.994 | 0.875 | 0.100 | 1.000 | 0.806 | `false` |
| `row_absence_guard_before_rules` | 1.000 | 0.900 | 0.863 | 0.081 | 0.275 | 0.231 | `false` |
| `no_memory_for_absent_guard` | 1.000 | 1.000 | 0.787 | 0.087 | 0.013 | 0.431 | `false` |
| `decision_order_guard_after_query` | 1.000 | 0.981 | 0.869 | 0.100 | 0.975 | 0.425 | `false` |
| `decision_order_guard_query_last` | 1.000 | 0.906 | 0.487 | 0.056 | 1.000 | 0.425 | `false` |

## Claim Boundary

MC030 can only say whether explicit absence guards repair the MC029
rules-only branch/null tradeoff. Hidden-state work remains forbidden
unless a full-source variant passes behavior gates and then survives
candidate/output margin baselines.
