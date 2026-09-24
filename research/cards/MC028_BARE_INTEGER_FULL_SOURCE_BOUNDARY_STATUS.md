# MC028 Bare-Integer Full-Source Boundary Status

Status: bare_integer_full_source_atomic_other_number_leak.

Observed pattern: `full_source_atomic_other_number_leak`.

Date: 2026-07-01

## Artifact

- runner:
  `code/mc028_bare_integer_full_source_boundary.py`
- result:
  `results/cards/MC028/mc028_bare_integer_full_source_boundary_behavior_20260701T160314.json`

## Verdict

MC028 is a full-source boundary test for MC027's winning bare-integer
interface. It does not establish a hidden signature, intervention, or
mechanism card. If behavior passes, candidate/output margin baselines are
still required before hidden-state work.

## Gate Criteria

| Criterion | Value |
| --- | --- |
| `smoke_mode` | `false` |
| `structural_passed` | `true` |
| `full_source_count_is_40` | `true` |
| `source_disjoint_holdout` | `true` |
| `selected_prompt_audit_passed` | `true` |
| `prompts_have_no_status_lexemes` | `true` |
| `familiar_lookup_at_least_90p` | `true` |
| `atomic_control_at_least_85p` | `true` |
| `answer_absent_unknown_at_least_90p` | `true` |
| `operation_rule_absent_unknown_at_least_90p` | `true` |
| `operation_local_conflict_at_least_85p` | `true` |
| `operation_atomic_conflict_at_least_85p` | `false` |
| `operation_atomic_other_number_leak_below_10p` | `false` |
| `all_panels_parseable_at_least_95p` | `true` |
| `candidate_and_output_margins_reported` | `false` |
| `all_controls_passed` | `true` |

## Panels

| Panel | Rows | Parseable | Local Rate | Atomic Rate | Unknown Rate | Other Rate | Expected Correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `familiar_interface_lookup` | 160 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| `atomic_interface_control` | 160 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| `operation_local_interface_conflict` | 160 | 1.000 | 0.994 | 0.006 | 0.000 | 0.000 | 0.994 |
| `operation_atomic_interface_conflict` | 160 | 1.000 | 0.006 | 0.819 | 0.000 | 0.175 | 0.819 |
| `operation_rule_absent_interface_null` | 160 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| `answer_absent_interface_null` | 160 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 |

## Claim Boundary

This result can only decide whether the bare-integer answer interface
survives full-source behavior gates. It does not license a probe or
intervention until output/candidate baselines are added.
