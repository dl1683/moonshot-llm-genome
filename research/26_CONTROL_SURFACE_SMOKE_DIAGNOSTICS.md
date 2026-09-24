# Control-Surface Smoke Diagnostics

Date: 2026-07-01

Status: generated smoke-diagnostic ledger implemented and validated.

Machine-readable artifact:

> `data/control_surface_smoke_diagnostics.json`

Builder:

> `code/control_surface_smoke_diagnostics.py`

Commands:

```powershell
python code\control_surface_smoke_diagnostics.py --write
python code\control_surface_smoke_diagnostics.py
python code\validate_control_surface_atlas.py
```

## Purpose

MC017-MC033 are not promoted atlas rows. They are bridge
diagnostics that test whether the bridge from prompt-local numeric
tables to learned atomic facts has a behavior substrate worth probing.

This ledger exists so typed failures remain auditable data. It keeps the
project from treating these runs as anecdotes or accidentally upgrading
them into mechanism claims.

## Current Generated Facts

- smoke cards: 17;
- smoke-only cards: 13;
- structural-passed cards: 17;
- behavior-ready cards: 0;
- signature-ready cards: 0;
- hidden-state-allowed cards: 0.

Failure-axis counts:

- `absence_guards_worsen_answer_absent_nulls`: 1;
- `answer_absent_null_failed`: 2;
- `answer_interface_dispersion`: 1;
- `answer_option_order_bias`: 1;
- `answer_token_collapse`: 1;
- `atomic_or_lure_conflict_selection_absent`: 1;
- `atomic_selector_control_failed`: 1;
- `bare_integer_full_source_atomic_branch_failed`: 1;
- `bare_integer_smoke_only_candidate`: 1;
- `candidate_output_baselines_reported`: 2;
- `checksum_valid_local_branch_clean`: 1;
- `choice_interface_answer_absent_null_failed`: 1;
- `choice_interface_atomic_branch_failed`: 1;
- `choice_interface_atomic_control_failed`: 1;
- `conditional_route_atomic_failed`: 1;
- `cross_table_match_branch_below_gate`: 1;
- `definition_order_bias`: 1;
- `direct_atomic_recall_available`: 1;
- `direct_controls_and_nulls_clean`: 3;
- `direct_controls_clean_conflict_failed`: 1;
- `expected_atomic_route_weak`: 1;
- `fact_claim_match_branch_below_gate`: 1;
- `fact_claim_mismatch_local_and_claim_leak`: 1;
- `factorial_branch_null_tradeoff`: 1;
- `fewshot_operation_atomic_branch_failed`: 1;
- `full_source_boundary_inherited_from_mc024`: 1;
- `full_source_controls_clean`: 1;
- `full_source_other_number_leak`: 1;
- `generic_visible_route_instability`: 1;
- `guarded_query_last_reduces_other_numbers_but_branch_collapses`: 1;
- `holdout_conflict_balance_failed`: 1;
- `learned_branch_worse_than_visible`: 1;
- `learned_memory_branch_collapse`: 1;
- `local_selector_prior`: 1;
- `local_source_salience`: 2;
- `neutral_label_rule_following_failed`: 1;
- `numeric_list_marker_structural_confounds_avoided`: 1;
- `numeric_option_atomic_branch_failed`: 1;
- `numeric_option_atomic_control_abstention`: 1;
- `numeric_options_nulls_clean`: 1;
- `operation_atomic_branch_below_gate`: 1;
- `operation_atomic_other_number_leak`: 1;
- `operation_local_branch_below_gate`: 1;
- `post_checksum_crosstable_mismatch_local_collapse`: 1;
- `post_mc032_bridge_route_closed`: 1;
- `prompt_local_branch_dominance`: 1;
- `prompt_visible_choice_options_not_substrate`: 1;
- `prompt_visible_numeric_options_not_substrate`: 1;
- `query_before_examples_amplifies_example_leak`: 1;
- `query_operation_substrate_failed`: 1;
- `query_row_last_reduces_other_numbers_but_local_collapses`: 1;
- `route_code_substrate_failed`: 1;
- `row_code_rule_following_failed`: 1;
- `rule_order_sensitivity`: 1;
- `rules_only_answer_absent_null_fails`: 1;
- `rules_only_atomic_branch_improves`: 1;
- `rules_only_branch_null_tradeoff_persists`: 1;
- `semantic_source_label_substrate_failed`: 1;
- `side_number_leak_absent`: 1;
- `smoke_only_no_candidate_output_baselines`: 1;
- `smoke_to_full_source_boundary`: 1;
- `statusless_checksum_invalid_branch_collapses_to_local`: 1;
- `structured_interfaces_break_behavior`: 1;
- `worked_example_output_leak`: 1;
- `worked_examples_controls_clean`: 1;
- `worked_examples_local_branch_passed`: 1;

## Ledger

| Card | Selected Template | Pattern | Headline Metrics | Exported Diagnostics |
| --- | --- | --- | --- | --- |
| `MC017` | `selector_rule` | `smoke_only` | atomic selector control atomic rate: `0.000`<br>answer-absent UNKNOWN rate: `0.100`<br>primary expected-correct rate: `0.500` | `ANSWER_TOKEN_LOCAL_COLLAPSE`<br>`ATOMIC_SELECTOR_CONTROL_FAILED`<br>`NULL_SELECTOR_LOCAL_COLLAPSE` |
| `MC018` | `explicit_source_labels` | `mixed_or_unresolved_selector_failure` | primary expected-correct rate: `0.525`<br>primary first-listed-choice rate: `0.762`<br>primary local-selector rate: `0.637`<br>answer-absent UNKNOWN rate: `0.325` | `NEUTRAL_SELECTOR_RULE_FAILED`<br>`FIRST_LISTED_CHOICE_BIAS`<br>`LOCAL_SOURCE_SALIENCE`<br>`ANSWER_ABSENT_SELECTOR_NULL_FAILED` |
| `MC019` | `compact_route_column` | `mixed_or_unresolved_row_code_failure` | primary expected-correct rate: `0.525`<br>expected-atomic atomic rate: `0.400`<br>route-rule-absent UNKNOWN rate: `1.000`<br>answer-absent UNKNOWN rate: `1.000` | `ROW_CODE_RULE_FAILED`<br>`EXPECTED_ATOMIC_ROUTE_WEAK`<br>`ROW_CODE_CONTROLS_CLEAN_CONFLICT_FAILED` |
| `MC020` | `plain_table` | `route_rule_expected_atomic_failed` | query-row atomic control atomic rate: `0.900`<br>repeated-query atomic control atomic rate: `0.900`<br>route-local local rate: `0.850`<br>route-atomic atomic rate: `0.100` | `DIRECT_ATOMIC_RECALL_SURVIVES_TABLE_PRESSURE`<br>`CONDITIONAL_ROUTE_ATOMIC_COLLAPSE`<br>`PROMPT_LOCAL_BRANCH_DOMINANCE` |
| `MC021` | `plain_branch_table` | `generic_visible_route_failed` | visible conflict expected-correct rate: `0.725`<br>learned conflict expected-correct rate: `0.500`<br>learned atomic-branch atomic rate: `0.300` | `GENERIC_VISIBLE_ROUTE_FAILED`<br>`LEARNED_BRANCH_ARBITRATION_WEAKER`<br>`ROUTE_CODE_SUBSTRATE_FAILED` |
| `MC022` | `explicit_source_column` | `explicit_visible_source_routing_failed` | visible conflict expected-correct rate: `0.750`<br>learned conflict expected-correct rate: `0.600`<br>ATOMIC-source atomic rate: `0.250`<br>nonlocal-first visible expected-correct rate: `1.000` | `SEMANTIC_SOURCE_LABEL_SUBSTRATE_FAILED`<br>`RULE_ORDER_SENSITIVITY`<br>`LOCAL_SOURCE_SALIENCE`<br>`LEARNED_MEMORY_BRANCH_COLLAPSE` |
| `MC023` | `query_first_operation` | `operation_local_branch_failed` | synthetic local control rate: `1.000`<br>atomic control atomic rate: `1.000`<br>answer-absent UNKNOWN rate: `1.000`<br>operation-local local rate: `0.775`<br>operation-atomic atomic rate: `0.700` | `QUERY_OPERATION_SUBSTRATE_FAILED`<br>`OPERATION_LOCAL_BRANCH_BELOW_GATE`<br>`OPERATION_ATOMIC_BRANCH_BELOW_GATE`<br>`QUERY_OPERATION_CONTROLS_CLEAN_CONFLICT_FAILED` |
| `MC024` | `compact_worked_examples` | `operation_atomic_branch_failed` | synthetic local control rate: `1.000`<br>atomic control atomic rate: `1.000`<br>answer-absent UNKNOWN rate: `1.000`<br>operation-local local rate: `0.994`<br>operation-atomic atomic rate: `0.812`<br>operation-atomic other-number rate: `0.169` | `FEWSHOT_OPERATION_ATOMIC_BRANCH_FAILED`<br>`WORKED_EXAMPLES_CONTROLS_CLEAN`<br>`WORKED_EXAMPLES_LOCAL_BRANCH_PASSED`<br>`OPERATION_ATOMIC_OTHER_NUMBER_LEAK` |
| `MC025` | `compact_choice_examples` | `choice_interface_smoke_failed` | synthetic choice local rate: `1.000`<br>atomic choice control atomic rate: `0.367`<br>answer-absent choice UNKNOWN rate: `0.692`<br>operation-local choice local rate: `0.908`<br>operation-atomic choice atomic rate: `0.150` | `CHOICE_INTERFACE_ATOMIC_CONTROL_FAILED`<br>`CHOICE_INTERFACE_ANSWER_ABSENT_NULL_FAILED`<br>`CHOICE_INTERFACE_ATOMIC_BRANCH_FAILED`<br>`PROMPT_VISIBLE_CHOICE_OPTIONS_NOT_SUBSTRATE` |
| `MC026` | `compact_numeric_options` | `numeric_option_smoke_failed` | synthetic option local rate: `1.000`<br>atomic option control atomic rate: `0.000`<br>atomic option control UNKNOWN rate: `0.967`<br>answer-absent option UNKNOWN rate: `1.000`<br>operation-local option local rate: `0.925`<br>operation-atomic option atomic rate: `0.158` | `NUMERIC_OPTION_ATOMIC_CONTROL_ABSTENTION`<br>`NUMERIC_OPTION_ATOMIC_BRANCH_FAILED`<br>`NUMERIC_OPTIONS_NULLS_CLEAN`<br>`PROMPT_VISIBLE_NUMERIC_OPTIONS_NOT_SUBSTRATE` |
| `MC027` | `bare_integer` | `answer_interface_substrate_candidate` | selected interface: `240`<br>bare integer atomic control rate: `1.000`<br>bare integer operation-atomic rate: `0.950`<br>answer-prefix operation-atomic rate: `0.275`<br>JSON answer-absent UNKNOWN rate: `0.075`<br>letter-choice operation-atomic rate: `0.158`<br>numeric-option atomic control rate: `0.000`<br>numeric-option operation-atomic rate: `0.175` | `ANSWER_INTERFACE_DISPERSION`<br>`BARE_INTEGER_SMOKE_ONLY_CANDIDATE`<br>`STRUCTURED_INTERFACES_BREAK_BEHAVIOR`<br>`FULL_SOURCE_BOUNDARY_INHERITED_FROM_MC024` |
| `MC028` | `bare_integer` | `full_source_atomic_other_number_leak` | familiar lookup local rate: `1.000`<br>atomic control atomic rate: `1.000`<br>answer-absent UNKNOWN rate: `1.000`<br>operation-local local rate: `0.994`<br>operation-atomic atomic rate: `0.819`<br>operation-atomic other-number rate: `0.175` | `BARE_INTEGER_FULL_SOURCE_ATOMIC_BRANCH_FAILED`<br>`FULL_SOURCE_OTHER_NUMBER_LEAK`<br>`FULL_SOURCE_CONTROLS_CLEAN`<br>`SMOKE_TO_FULL_SOURCE_BOUNDARY` |
| `MC029` | `baseline_numeric_examples` | `rules_only_improves_atomic_branch_but_nulls_fail` | baseline operation-atomic rate: `0.713`<br>baseline other-number rate: `0.244`<br>rules-only operation-atomic rate: `0.875`<br>rules-only answer-absent UNKNOWN rate: `0.806`<br>query-before operation-atomic rate: `0.256`<br>query-before other-number rate: `0.450`<br>query-row-last other-number rate: `0.081` | `WORKED_EXAMPLE_OUTPUT_LEAK`<br>`RULES_ONLY_ATOMIC_BRANCH_IMPROVES`<br>`RULES_ONLY_ANSWER_ABSENT_NULL_FAILS`<br>`QUERY_BEFORE_EXAMPLES_AMPLIFIES_EXAMPLE_LEAK`<br>`QUERY_ROW_LAST_REDUCES_OTHER_NUMBERS_BUT_LOCAL_COLLAPSES`<br>`FACTORIAL_BRANCH_NULL_TRADEOFF` |
| `MC030` | `rules_only_baseline` | `branch_preserved_null_not_repaired` | baseline operation-atomic rate: `0.875`<br>baseline answer-absent UNKNOWN rate: `0.806`<br>row-guard operation-atomic rate: `0.863`<br>row-guard answer-absent UNKNOWN rate: `0.231`<br>decision-order answer-absent UNKNOWN rate: `0.425`<br>query-last operation-atomic rate: `0.487`<br>query-last other-number rate: `0.056` | `RULES_ONLY_BRANCH_NULL_TRADEOFF_PERSISTS`<br>`ABSENCE_GUARDS_WORSEN_ANSWER_ABSENT_NULLS`<br>`GUARDED_QUERY_LAST_REDUCES_OTHER_NUMBERS_BUT_BRANCH_COLLAPSES`<br>`NUMERIC_LIST_MARKERS_CAN_LEAK_ATOMIC_TARGETS` |
| `MC031` | `arithmetic_checksum` | `smoke_only` | synthetic lookup local rate: `1.000`<br>real atomic control rate: `1.000`<br>answer-absent UNKNOWN rate: `1.000`<br>valid-checksum local rate: `1.000`<br>invalid-checksum local rate: `1.000`<br>invalid-checksum atomic/lure rate: `0.000` | `STATUSLESS_CHECKSUM_INVALID_BRANCH_LOCAL_COLLAPSE`<br>`CHECKSUM_VALID_LOCAL_BRANCH_CLEAN`<br>`DIRECT_CONTROLS_AND_NULLS_CLEAN`<br>`ATOMIC_OR_LURE_CONFLICT_SELECTION_ABSENT`<br>`SMOKE_ONLY_NO_CANDIDATE_OUTPUT_BASELINES` |
| `MC032` | `mirror_registry` | `smoke_only` | synthetic lookup local rate: `1.000`<br>real atomic control rate: `1.000`<br>answer-absent UNKNOWN rate: `1.000`<br>match-conflict local rate: `0.800`<br>mismatch-conflict local rate: `0.700`<br>mismatch-conflict atomic/lure rate: `0.000`<br>mismatch side-number rate: `0.000` | `POST_CHECKSUM_CROSSTABLE_LOCAL_COLLAPSE`<br>`CROSSTABLE_MATCH_BRANCH_BELOW_GATE`<br>`DIRECT_CONTROLS_AND_NULLS_CLEAN`<br>`CROSSTABLE_SIDE_NUMBER_LEAK_ABSENT`<br>`CANDIDATE_OUTPUT_BASELINES_REPORTED` |
| `MC033` | `memory_comparison` | `smoke_only` | synthetic lookup local rate: `1.000`<br>real atomic control rate: `1.000`<br>answer-absent UNKNOWN rate: `1.000`<br>match-conflict local rate: `0.400`<br>match-conflict atomic rate: `0.500`<br>mismatch-conflict atomic rate: `0.100`<br>mismatch-conflict local rate: `0.500`<br>mismatch claimed-number/lure rate: `0.400` | `FACT_CLAIM_MATCH_BRANCH_FAILED`<br>`FACT_CLAIM_MISMATCH_LOCAL_AND_CLAIM_LEAK`<br>`DIRECT_CONTROLS_AND_NULLS_CLEAN`<br>`CANDIDATE_OUTPUT_BASELINES_REPORTED`<br>`POST_MC032_BRIDGE_ROUTE_CLOSED` |

## Validation Checks

| Check | Card | Passed | Actual |
| --- | --- | --- | --- |
| `all_cards_smoke_blocked` | `ALL` | `true` | `{"behavior_ready_count": 0, "signature_ready_count": 0}` |
| `mc017_atomic_selector_collapsed_to_local` | `MC017` | `true` | `0.000` |
| `mc018_first_listed_choice_bias` | `MC018` | `true` | `0.762` |
| `mc018_selector_rule_not_followed` | `MC018` | `true` | `0.525` |
| `mc019_expected_atomic_route_weak` | `MC019` | `true` | `0.400` |
| `mc019_nulls_clean_despite_conflict_failure` | `MC019` | `true` | `1.000` |
| `mc020_direct_atomic_recall_survives_query_row` | `MC020` | `true` | `0.900` |
| `mc020_route_atomic_branch_failed` | `MC020` | `true` | `0.100` |
| `mc021_visible_route_unstable` | `MC021` | `true` | `0.725` |
| `mc021_learned_route_unstable` | `MC021` | `true` | `0.500` |
| `mc022_semantic_visible_route_unstable` | `MC022` | `true` | `0.750` |
| `mc022_atomic_source_branch_collapsed` | `MC022` | `true` | `0.250` |
| `mc022_nonlocal_first_visible_rescue` | `MC022` | `true` | `1.000` |
| `mc022_local_first_visible_collapse` | `MC022` | `true` | `0.500` |
| `mc023_controls_clean_before_conflict_failure` | `MC023` | `true` | `True` |
| `mc023_operation_local_branch_below_gate` | `MC023` | `true` | `0.775` |
| `mc023_operation_atomic_branch_below_gate` | `MC023` | `true` | `0.700` |
| `mc024_controls_clean_before_atomic_branch_failure` | `MC024` | `true` | `True` |
| `mc024_fewshot_local_branch_passed` | `MC024` | `true` | `0.994` |
| `mc024_fewshot_atomic_branch_below_gate` | `MC024` | `true` | `0.812` |
| `mc024_atomic_branch_other_number_leak` | `MC024` | `true` | `0.169` |
| `mc025_choice_controls_failed` | `MC025` | `true` | `False` |
| `mc025_choice_atomic_control_failed` | `MC025` | `true` | `0.367` |
| `mc025_choice_answer_absent_null_failed` | `MC025` | `true` | `0.692` |
| `mc025_choice_local_branch_passed` | `MC025` | `true` | `0.908` |
| `mc025_choice_atomic_branch_failed` | `MC025` | `true` | `0.150` |
| `mc026_numeric_option_controls_failed` | `MC026` | `true` | `False` |
| `mc026_numeric_option_atomic_control_zero` | `MC026` | `true` | `0.000` |
| `mc026_numeric_option_atomic_control_abstains` | `MC026` | `true` | `0.967` |
| `mc026_numeric_option_nulls_clean` | `MC026` | `true` | `1.000` |
| `mc026_numeric_option_local_branch_passed` | `MC026` | `true` | `0.925` |
| `mc026_numeric_option_atomic_branch_failed` | `MC026` | `true` | `0.158` |
| `mc027_selected_bare_integer_atomic_control_passed` | `MC027` | `true` | `1.000` |
| `mc027_selected_bare_integer_atomic_branch_passed_smoke` | `MC027` | `true` | `0.950` |
| `mc027_numeric_options_zero_atomic_control` | `MC027` | `true` | `0.000` |
| `mc027_json_null_collapsed` | `MC027` | `true` | `0.075` |
| `mc027_letter_choice_atomic_branch_failed` | `MC027` | `true` | `0.158` |
| `mc027_answer_prefix_atomic_branch_failed` | `MC027` | `true` | `0.275` |
| `mc028_full_source_controls_clean` | `MC028` | `true` | `True` |
| `mc028_full_source_local_branch_passed` | `MC028` | `true` | `0.994` |
| `mc028_full_source_atomic_branch_failed` | `MC028` | `true` | `0.819` |
| `mc028_full_source_other_number_leak` | `MC028` | `true` | `0.175` |
| `mc029_rules_only_atomic_branch_improves` | `MC029` | `true` | `0.875` |
| `mc029_rules_only_nulls_fail` | `MC029` | `true` | `0.806` |
| `mc029_query_before_examples_atomic_branch_collapses` | `MC029` | `true` | `0.256` |
| `mc029_query_before_examples_amplifies_other_numbers` | `MC029` | `true` | `0.450` |
| `mc029_query_row_last_reduces_other_numbers` | `MC029` | `true` | `0.081` |
| `mc029_query_row_last_still_not_behavior_gate` | `MC029` | `true` | `0.613` |
| `mc030_baseline_preserves_rules_only_branch` | `MC030` | `true` | `0.875` |
| `mc030_baseline_null_still_fails` | `MC030` | `true` | `0.806` |
| `mc030_row_absence_guard_worsens_nulls` | `MC030` | `true` | `0.231` |
| `mc030_decision_guard_worsens_nulls` | `MC030` | `true` | `0.425` |
| `mc030_query_last_reduces_other_numbers` | `MC030` | `true` | `0.056` |
| `mc030_query_last_collapses_atomic_branch` | `MC030` | `true` | `0.487` |
| `mc031_synthetic_lookup_control_clean` | `MC031` | `true` | `1.000` |
| `mc031_real_atomic_control_clean` | `MC031` | `true` | `1.000` |
| `mc031_answer_absent_null_clean` | `MC031` | `true` | `1.000` |
| `mc031_valid_checksum_local_branch_clean` | `MC031` | `true` | `1.000` |
| `mc031_invalid_checksum_collapses_to_local` | `MC031` | `true` | `1.000` |
| `mc031_invalid_checksum_atomic_lure_absent` | `MC031` | `true` | `0.000` |
| `mc032_synthetic_lookup_control_clean` | `MC032` | `true` | `1.000` |
| `mc032_real_atomic_control_clean` | `MC032` | `true` | `1.000` |
| `mc032_answer_absent_null_clean` | `MC032` | `true` | `1.000` |
| `mc032_mismatch_atomic_lure_absent` | `MC032` | `true` | `0.000` |
| `mc032_mismatch_collapses_toward_local` | `MC032` | `true` | `0.700` |
| `mc032_side_number_leak_absent` | `MC032` | `true` | `0.000` |
| `mc033_synthetic_lookup_control_clean` | `MC033` | `true` | `1.000` |
| `mc033_real_atomic_control_clean` | `MC033` | `true` | `1.000` |
| `mc033_answer_absent_null_clean` | `MC033` | `true` | `1.000` |
| `mc033_match_branch_below_gate` | `MC033` | `true` | `0.400` |
| `mc033_mismatch_atomic_branch_below_gate` | `MC033` | `true` | `0.100` |
| `mc033_mismatch_claimed_number_leak` | `MC033` | `true` | `0.400` |

## Interpretation

The current smoke chain says the bridge failure is layered:

1. `LOCAL`/`ATOMIC` answer tokens can create a local-selector collapse.
2. Neutral source labels expose first-listed-choice and local-source salience.
3. Row-local route codes can clean up controls and nulls while still failing expected-atomic routing.
4. Direct atomic recall survives local-table pressure, so recall availability is not the bottleneck.
5. Opaque route codes fail even between prompt-visible branches.
6. Semantic answer-source labels remain sensitive to rule order and still fail learned-memory branch selection.
7. Query-level operation handles preserve direct controls and nulls, but both conflict branches remain below gate quality.
8. Balanced worked examples repair prompt-local operation routing, but the learned atomic branch remains below full-source gate quality.
9. Prompt-visible A/B/C choices are not a neutral answer-interface fix; they break atomic controls and answer-absent nulls.
10. Prompt-visible numeric options preserve nulls but turn direct atomic recall into UNKNOWN and leave the learned branch weak.
11. Sweeping answer interfaces shows that bare integer is the only 10-source smoke survivor; structured schemas and option interfaces break controls, nulls, or learned routing.
12. The bare-integer smoke survivor fails full-source promotion: controls and nulls stay clean, but operation-atomic rows leak other numbers below gate.
13. Factorizing the leak and then adding absence guards moves the error axes but does not repair the rules-only branch/null tradeoff.

The next bridge cannot just rename labels, repeat rows, or optimize direct
atomic recall. It also cannot rely on query-level operation handles unless
both local and learned branches exceed gate thresholds, and it cannot treat
candidate choices, numeric option lists, JSON schemas, or answer-prefix
schemas as free output repairs. It must create a behavior substrate where the same rule
selects prompt-local, prompt-visible nonlocal, and learned-memory branches
under balanced rule-order, local-salience, full-source, and
other-number side-effect controls.

## Claim Boundary

This ledger does not establish a signature, intervention, or mechanism card.
It is a generated diagnostic layer for behavior-substrate failures that
should constrain future mechanism-card attempts.
