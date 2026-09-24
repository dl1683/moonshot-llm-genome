# KSQ004 Bridge Answer-Interface Minimal Pairs First-Run Status

Date: 2026-07-01

Runner:

> `code/ksq004_bridge_answer_interface_minimal_pairs_first_run.py`

Result:

> `results/cards/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS/ksq004_bridge_answer_interface_minimal_pairs_full_behavior.json`

Status: bridge_minimal_pair_contrast_absent.

## Verdict

The behavior gate failed. Hidden-state work remains forbidden.

## Structural Criteria

| Criterion | Passed |
| --- | --- |
| `expected_row_count` | `true` |
| `all_panels_present` | `true` |
| `all_templates_present` | `true` |
| `all_subtypes_present` | `true` |
| `source_split_disjoint` | `true` |
| `holdout_sources_present` | `true` |
| `calibration_sources_present` | `true` |
| `primary_expected_labels_balanced` | `true` |
| `matched_expected_labels_balanced` | `true` |
| `primary_atomic_answer_hidden` | `true` |
| `answer_absent_omits_query_local_number` | `true` |
| `side_panels_have_single_side_number` | `true` |
| `candidate_answers_parseable` | `true` |
| `no_candidate_collisions` | `true` |
| `single_answer_suffix` | `true` |
| `no_multiple_choice_interface_words` | `true` |

## Counts

- records: `720`
- sources: `40`
- panels: `{"local_learned_direct_controls": 160, "matched_minimal_pairs": 160, "minimal_pair_conflict": 160, "null_and_holdout": 80, "side_answer_leakage": 160}`
- templates: `{"compact_form": 360, "question_form": 360}`
- subtypes: `{"answer_absent_null": 80, "atomic_branch_conflict": 80, "atomic_direct": 80, "atomic_side_leakage": 80, "local_branch_conflict": 80, "local_direct": 80, "local_side_leakage": 80, "matched_atomic_contract": 80, "matched_local_contract": 80}`
- split source counts: `{"calibration": 8, "discovery": 24, "holdout": 8}`

## Behavior Result

- result: `results/cards/KSQ004_BRIDGE_ANSWER_INTERFACE_MINIMAL_PAIRS/ksq004_bridge_answer_interface_minimal_pairs_full_behavior.json`
- diagnostic class: `bridge_minimal_pair_contrast_absent`
- behavior ready: `false`
- behavior candidate: `false`
- selected template: `compact_form`
- candidate/output margins reported: `true`

| Panel | Label counts | Expected-correct rate |
| --- | --- | --- |
| `matched_minimal_pairs` | `{"atomic_number": 40, "local_number": 40}` | `1.000` |
| `local_learned_direct_controls` | `{"atomic_number": 40, "local_number": 40}` | `1.000` |
| `minimal_pair_conflict` | `{"atomic_number": 9, "local_number": 70, "other_number": 1}` | `0.613` |
| `side_answer_leakage` | `{"atomic_number": 8, "local_number": 72}` | `0.600` |
| `null_and_holdout` | `{"unknown": 40}` | `1.000` |

Template contrast:

| Template | Conflict expected-correct | Atomic-branch atomic rate | Holdout expected-correct |
| --- | --- | --- | --- |
| `question_form` | `0.850` | `0.700` | `0.875` |
| `compact_form` | `0.613` | `0.225` | `0.625` |

Subtype counts:

| Subtype | Label counts |
| --- | --- |
| `matched_local_contract` | `{"local_number": 40}` |
| `matched_atomic_contract` | `{"atomic_number": 40}` |
| `local_direct` | `{"local_number": 40}` |
| `atomic_direct` | `{"atomic_number": 40}` |
| `local_branch_conflict` | `{"local_number": 40}` |
| `atomic_branch_conflict` | `{"atomic_number": 9, "local_number": 30, "other_number": 1}` |
| `local_side_leakage` | `{"local_number": 40}` |
| `atomic_side_leakage` | `{"atomic_number": 8, "local_number": 32}` |
| `answer_absent_null` | `{"unknown": 40}` |

Interpretation: KSQ004 is a shortcut detector. A pass would only
license a later signature-screen decision. A fail identifies whether
the bridge disappears under matched answer interface, leaks through
side answers, fails null/holdout, or is already output-geometry visible.

## Forbidden Claims

- KSQ004 is a mechanism card.
- KSQ004 supports intervention.
- KSQ004 found an internal bridge, knowledge, or answer-interface control surface.
- Any hidden-state or causal claim follows from this first run alone.
