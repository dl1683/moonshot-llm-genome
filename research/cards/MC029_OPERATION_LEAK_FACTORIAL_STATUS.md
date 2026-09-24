# MC029 Operation Leak Factorial Status

Status: operation_leak_factorial_branch_null_tradeoff.

Observed pattern: `rules_only_improves_atomic_branch_but_nulls_fail`.

Date: 2026-07-01

## Artifact

- runner:
  `code/mc029_operation_leak_factorial.py`
- result:
  `results/cards/MC029/mc029_operation_leak_factorial_behavior_20260701T170417.json`

## Verdict

MC029 is a factorized diagnostic for MC028's full-source other-number leak.
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
| `variant_gate_by_template` | `{"baseline_numeric_examples": false, "label_examples_no_numbers": false, "query_before_examples": false, "query_row_last": false, "rules_only": false}` |
| `any_variant_behavior_gate_passed` | `false` |
| `baseline_operation_atomic_rate` | `0.713` |
| `baseline_other_number_rate` | `0.244` |
| `selected_template_operation_atomic_rate` | `0.713` |
| `selected_template_other_number_rate` | `0.244` |
| `max_operation_atomic_template` | `rules_only` |
| `max_operation_atomic_rate` | `0.875` |
| `min_other_number_template` | `query_row_last` |
| `min_other_number_rate` | `0.081` |
| `rules_only_operation_atomic_rate` | `0.875` |
| `rules_only_other_number_rate` | `0.100` |
| `rules_only_answer_absent_unknown_rate` | `0.806` |
| `baseline_worked_example_other_count` | `7` |
| `label_examples_worked_example_other_count` | `0` |
| `query_before_worked_example_other_count` | `54` |
| `candidate_and_output_margins_reported` | `false` |

## Variant Comparison

| Variant | Atomic Control | Local Branch | Atomic Branch | Other Number | Rule Null | Answer Null | Worked-Example Other | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline_numeric_examples` | 1.000 | 1.000 | 0.713 | 0.244 | 1.000 | 0.988 | 7 | `false` |
| `label_examples_no_numbers` | 1.000 | 0.950 | 0.800 | 0.175 | 1.000 | 0.725 | 0 | `false` |
| `rules_only` | 1.000 | 0.994 | 0.875 | 0.100 | 1.000 | 0.806 | 0 | `false` |
| `query_before_examples` | 1.000 | 0.981 | 0.256 | 0.450 | 1.000 | 1.000 | 54 | `false` |
| `query_row_last` | 1.000 | 0.925 | 0.613 | 0.081 | 1.000 | 0.988 | 6 | `false` |

## Claim Boundary

MC029 can only say which prompt-surface factors move the MC028 leak.
It cannot license hidden-state work until a behavior-passing variant
also reports candidate/output margin baselines.
