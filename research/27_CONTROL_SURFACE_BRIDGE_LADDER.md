# Control-Surface Bridge Ladder

Date: 2026-07-01

Status: generated bridge ladder implemented and validated.

Machine-readable artifact:

> `data/control_surface_bridge_ladder.json`

Builder:

> `code/control_surface_bridge_ladder.py`

Commands:

```powershell
python code\control_surface_bridge_ladder.py --write
python code\control_surface_bridge_ladder.py
python code\validate_control_surface_atlas.py
```

## Purpose

The bridge ladder turns MC010-MC033 into one cumulative object. It asks
what each repair changed, which behavior gate failed, and what the next
bridge must beat before hidden-state work is allowed.

## Generated Facts

- rungs: 24;
- atlas rungs: 7;
- smoke rungs: 17;
- behavior-ready rungs: 1;
- signature-ready rungs: 0;
- hidden-state-allowed rungs: 0;
- clean unconfounded bridge rungs: 0;
- prompt-visible positive controls: 1.

## Ladder

| Card | Axis | Outcome | Mixture | Headline Metrics | Boundary |
| --- | --- | --- | --- | --- | --- |
| `MC010` | `source_path` | `failed_controls_and_conflict` | `absent` | synthetic task-code rate: `0.575`<br>real memory symbol rate: `0.300`<br>conflict task-code rate: `0.954`<br>conflict learned-symbol rate: `0.000` | Two-hop indirection did not create a behavior substrate. |
| `MC011` | `answer_interface` | `controls_clean_conflict_local_collapse` | `absent` | synthetic local rate: `1.000`<br>real atomic control rate: `1.000`<br>null UNKNOWN rate: `1.000`<br>conflict local rate: `1.000`<br>conflict atomic rate: `0.000` | Answer-format repair is insufficient under prompt-local table authority. |
| `MC012` | `source_contract` | `behavior_ready_but_prompt_visible` | `clean_prompt_visible` | trusted local rate: `1.000`<br>untrusted atomic rate: `0.975`<br>primary local rate: `0.500`<br>primary atomic rate: `0.487`<br>prompt-channel locality gate: `false` | Positive control only; source-status text carries the answer rule. |
| `MC013` | `prompt_channel_locality` | `positive_control_reproduced_ablation_collapsed` | `statused_only` | statused trusted local rate: `1.000`<br>statused untrusted atomic rate: `0.975`<br>ablation local rate: `1.000`<br>ablation atomic rate: `0.000`<br>ablation prompts identical: `true` | The contrast does not survive status-channel removal. |
| `MC014` | `inferred_source_validity` | `calibration_inference_collapsed` | `absent` | status lexemes absent: `true`<br>consistent local rate: `1.000`<br>inconsistent atomic rate: `0.000`<br>primary local rate: `1.000` | Calibration evidence did not overcome local table authority. |
| `MC015` | `learned_fact_gate` | `mixed_outputs_wrong_rule` | `mixed_wrong_rule` | expected labels balanced: `true`<br>primary expected-correct rate: `0.487`<br>expected-local local rate: `0.675`<br>expected-atomic atomic rate: `0.300` | Mixed outputs are not enough when expected labels are not followed. |
| `MC016` | `visible_nonstatus_gate` | `visible_gate_local_collapse` | `absent` | expected labels balanced: `true`<br>primary local rate: `1.000`<br>expected-local local rate: `1.000`<br>expected-atomic atomic rate: `0.000` | A visible non-status gate still collapses expected-atomic rows. |
| `MC017` | `source_answer_interface` | `answer_token_local_collapse` | `absent` | atomic selector atomic rate: `0.000`<br>answer-absent UNKNOWN rate: `0.100`<br>primary expected-correct rate: `0.500` | Source-token answers are themselves a confounded behavior surface. |
| `MC018` | `neutral_source_answer_interface` | `neutral_selector_rule_failed` | `mixed_wrong_rule` | primary expected-correct rate: `0.525`<br>first-listed-choice rate: `0.762`<br>local-selector rate: `0.637` | Neutral labels expose first-listed-choice and local-source salience. |
| `MC019` | `row_local_route_code` | `row_code_conflict_failed` | `mixed_wrong_rule` | primary expected-correct rate: `0.525`<br>expected-atomic atomic rate: `0.400`<br>answer-absent UNKNOWN rate: `1.000` | Clean controls and nulls do not imply expected-atomic routing. |
| `MC020` | `atomic_recall_vs_routing` | `direct_recall_passes_route_atomic_fails` | `asymmetric_route_failure` | query-row atomic control rate: `0.900`<br>route-local local rate: `0.850`<br>route-atomic atomic rate: `0.100` | The bottleneck is conditional arbitration, not basic atomic recall. |
| `MC021` | `visible_vs_learned_branch_routing` | `route_code_unstable_even_visible` | `asymmetric_route_failure` | visible conflict expected-correct rate: `0.725`<br>learned conflict expected-correct rate: `0.500`<br>learned atomic-branch atomic rate: `0.300` | Opaque route codes are not a clean arbitration substrate. |
| `MC022` | `semantic_branch_label_routing` | `semantic_labels_rule_order_sensitive` | `asymmetric_route_failure` | visible conflict expected-correct rate: `0.750`<br>learned conflict expected-correct rate: `0.600`<br>ATOMIC-source atomic rate: `0.250`<br>nonlocal-first visible expected-correct rate: `1.000` | Semantic labels do not solve learned-memory branch arbitration. |
| `MC023` | `query_operation_routing` | `query_operation_conflict_below_gate` | `mixed_below_gate` | synthetic local control rate: `1.000`<br>atomic control atomic rate: `1.000`<br>answer-absent UNKNOWN rate: `1.000`<br>operation-local local rate: `0.775`<br>operation-atomic atomic rate: `0.700` | Query-level operation handles preserve controls and nulls but do not clear both conflict branches. |
| `MC024` | `fewshot_query_operation_routing` | `fewshot_operation_atomic_branch_failed` | `asymmetric_route_failure` | synthetic local control rate: `1.000`<br>atomic control atomic rate: `1.000`<br>answer-absent UNKNOWN rate: `1.000`<br>operation-local local rate: `0.994`<br>operation-atomic atomic rate: `0.812`<br>operation-atomic other-number rate: `0.169` | Worked examples repair local routing but not full-source learned atomic routing. |
| `MC025` | `choice_answer_interface` | `choice_interface_controls_failed` | `choice_interface_confounded` | synthetic choice local rate: `1.000`<br>atomic choice control atomic rate: `0.367`<br>answer-absent choice UNKNOWN rate: `0.692`<br>operation-local choice local rate: `0.908`<br>operation-atomic choice atomic rate: `0.150` | Prompt-visible candidate choices break atomic controls and nulls while leaving learned conflict routing weak. |
| `MC026` | `numeric_option_answer_interface` | `numeric_option_atomic_abstention_failed` | `numeric_option_confounded` | synthetic option local rate: `1.000`<br>atomic option control atomic rate: `0.000`<br>atomic option control UNKNOWN rate: `0.967`<br>answer-absent option UNKNOWN rate: `1.000`<br>operation-local option local rate: `0.925`<br>operation-atomic option atomic rate: `0.158` | Numeric options preserve nulls but turn direct atomic control into UNKNOWN and leave learned conflict routing weak. |
| `MC027` | `answer_interface_law` | `bare_integer_smoke_only_structured_interfaces_failed` | `interface_dependent_smoke_candidate` | bare integer atomic control rate: `1.000`<br>bare integer operation-atomic rate: `0.950`<br>answer-prefix operation-atomic rate: `0.275`<br>JSON answer-absent UNKNOWN rate: `0.075`<br>letter-choice operation-atomic rate: `0.158`<br>numeric-option atomic control rate: `0.000`<br>numeric-option operation-atomic rate: `0.175` | Bare integer clears the 10-source smoke, but structured interfaces distort controls, nulls, or learned routing; MC024 remains the full-source boundary. |
| `MC028` | `full_source_boundary` | `bare_integer_full_source_atomic_other_number_leak` | `full_source_boundary_failed` | familiar lookup local rate: `1.000`<br>atomic control atomic rate: `1.000`<br>answer-absent UNKNOWN rate: `1.000`<br>operation-local local rate: `0.994`<br>operation-atomic atomic rate: `0.819`<br>operation-atomic other-number rate: `0.175` | The bare-integer smoke survivor fails full-source promotion: controls and nulls stay clean, but operation-atomic rows fall below gate with other-number leakage. |
| `MC029` | `factorized_leak_boundary` | `rules_only_improves_atomic_branch_but_nulls_fail` | `factorial_branch_null_tradeoff` | baseline operation-atomic rate: `0.713`<br>baseline other-number rate: `0.244`<br>rules-only operation-atomic rate: `0.875`<br>rules-only answer-absent UNKNOWN rate: `0.806`<br>query-before operation-atomic rate: `0.256`<br>query-before other-number rate: `0.450`<br>query-row-last other-number rate: `0.081` | No variant passes the bridge gate. Rules-only improves operation-atomic routing to 0.875 and removes worked-example outputs, but answer-absent nulls fall to 0.806; query-before-examples amplifies other-number leakage. |
| `MC030` | `null_preserving_guard_repair` | `absence_guards_worsen_nulls` | `guarded_branch_null_tradeoff` | baseline operation-atomic rate: `0.875`<br>baseline answer-absent UNKNOWN rate: `0.806`<br>row-guard operation-atomic rate: `0.863`<br>row-guard answer-absent UNKNOWN rate: `0.231`<br>decision-order answer-absent UNKNOWN rate: `0.425`<br>query-last operation-atomic rate: `0.487`<br>query-last other-number rate: `0.056` | No guard variant passes. The unguarded rules-only baseline remains the best branch/null compromise; guards either worsen nulls or collapse learned atomic routing. |
| `MC031` | `statusless_checksum_reliability` | `statusless_checksum_invalid_branch_local_collapse` | `statusless_reliability_local_collapse` | synthetic lookup local rate: `1.000`<br>real atomic control rate: `1.000`<br>answer-absent UNKNOWN rate: `1.000`<br>valid-checksum local rate: `1.000`<br>invalid-checksum local rate: `1.000`<br>invalid-checksum atomic/lure rate: `0.000` | Direct controls, valid-checksum local rows, and answer-absent nulls are clean in smoke, but invalid-checksum rows select local numbers on every selected conflict. |
| `MC032` | `statusless_cross_table_consistency` | `cross_table_mismatch_local_collapse` | `statusless_reliability_local_collapse` | synthetic lookup local rate: `1.000`<br>real atomic control rate: `1.000`<br>answer-absent UNKNOWN rate: `1.000`<br>match-conflict local rate: `0.800`<br>mismatch-conflict local rate: `0.700`<br>mismatch-conflict atomic/lure rate: `0.000`<br>mismatch side-number rate: `0.000` | Direct controls, answer-absent nulls, and side-number locality are clean in smoke, but mismatch rows still avoid the learned atomic branch. |
| `MC033` | `learned_fact_claim_validity` | `fact_claim_match_and_mismatch_failed` | `statusless_reliability_local_and_claim_leak` | synthetic lookup local rate: `1.000`<br>real atomic control rate: `1.000`<br>answer-absent UNKNOWN rate: `1.000`<br>match-conflict local rate: `0.400`<br>match-conflict atomic rate: `0.500`<br>mismatch-conflict atomic rate: `0.100`<br>mismatch-conflict local rate: `0.500`<br>mismatch claimed-number/lure rate: `0.400` | Direct controls and answer-absent nulls are clean in smoke, but the match branch returns atomic answers too often and mismatch rows split between local and the wrong claimed number. |

## Validation Checks

| Check | Card | Passed | Actual |
| --- | --- | --- | --- |
| `no_bridge_rung_allows_hidden_state` | `ALL` | `true` | `0` |
| `exactly_one_prompt_visible_positive_control` | `ALL` | `true` | `1` |
| `mc011_numeric_controls_but_conflict_local` | `MC011` | `true` | `{"conflict_atomic": 0.0, "conflict_local": 1.0, "real_atomic_control": 1.0}` |
| `mc012_prompt_visible_contrast_only` | `MC012` | `true` | `{"prompt_channel_locality_gate": false, "trusted_local": 1.0, "untrusted_atomic": 0.975}` |
| `mc013_status_ablation_kills_contrast` | `MC013` | `true` | `{"ablation_atomic": 0.0, "ablation_local": 1.0, "statused_untrusted_atomic": 0.975}` |
| `mc016_visible_nonstatus_gate_collapses_atomic` | `MC016` | `true` | `{"expected_atomic_atomic": 0.0, "primary_local": 1.0}` |
| `mc020_atomic_recall_not_bottleneck` | `MC020` | `true` | `{"query_row_atomic": 0.9, "route_atomic": 0.1}` |
| `mc022_semantic_labels_not_enough` | `MC022` | `true` | `{"atomic_source_atomic": 0.25, "nonlocal_first_visible": 1.0}` |
| `mc023_query_operation_controls_clean_but_conflict_below_gate` | `MC023` | `true` | `{"answer_absent_unknown": 1.0, "atomic_control": 1.0, "operation_atomic": 0.7, "operation_local": 0.775, "synthetic_local": 1.0}` |
| `mc024_fewshot_operations_repair_local_not_atomic` | `MC024` | `true` | `{"answer_absent_unknown": 1.0, "atomic_control": 1.0, "operation_atomic": 0.8125, "operation_atomic_other": 0.16875, "operation_local": 0.99375, "synthetic_local": 1.0}` |
| `mc025_choice_interface_breaks_controls_and_atomic_branch` | `MC025` | `true` | `{"answer_absent_unknown": 0.6916666666666667, "atomic_control": 0.36666666666666664, "operation_atomic": 0.15, "operation_local": 0.9083333333333333, "synthetic_local": 1.0}` |
| `mc026_numeric_options_preserve_nulls_but_break_atomic_control` | `MC026` | `true` | `{"answer_absent_unknown": 1.0, "atomic_control": 0.0, "atomic_control_unknown": 0.9666666666666667, "operation_atomic": 0.15833333333333333, "operation_local": 0.925, "synthetic_local": 1.0}` |
| `mc027_answer_interface_sweep_bare_integer_only_smoke_survivor` | `MC027` | `true` | `{"bare_atomic_control": 1.0, "bare_operation_atomic": 0.95, "choice_operation_atomic": 0.15833333333333333, "json_answer_absent_unknown": 0.075, "numeric_atomic_control": 0.0, "numeric_operation_atomic": 0.175, "prefix_operation_atomic": 0.275}` |
| `mc028_bare_integer_full_source_boundary_failed` | `MC028` | `true` | `{"answer_absent_unknown": 1.0, "atomic_control": 1.0, "familiar_lookup": 1.0, "operation_atomic": 0.81875, "operation_atomic_other": 0.175, "operation_local": 0.99375}` |
| `mc029_factorial_branch_null_tradeoff` | `MC029` | `true` | `{"baseline_operation_atomic": 0.7125, "baseline_other": 0.24375, "query_before_operation_atomic": 0.25625, "query_before_other": 0.45, "query_row_last_other": 0.08125, "rules_only_answer_absent_unknown": 0.80625, "rules_only_operation_atomic": 0.875}` |
| `mc030_absence_guard_repair_fails` | `MC030` | `true` | `{"baseline_answer_absent_unknown": 0.80625, "baseline_operation_atomic": 0.875, "decision_order_answer_absent_unknown": 0.425, "query_last_operation_atomic": 0.4875, "query_last_other": 0.05625, "row_guard_answer_absent_unknown": 0.23125, "row_guard_operation_atomic": 0.8625}` |
| `mc031_statusless_checksum_invalid_branch_collapses` | `MC031` | `true` | `{"answer_absent_unknown": 1.0, "atomic_control": 1.0, "invalid_checksum_atomic_lure": 0.0, "invalid_checksum_local": 1.0, "synthetic_lookup": 1.0, "valid_checksum_local": 1.0}` |
| `mc032_crosstable_mismatch_collapses_toward_local` | `MC032` | `true` | `{"answer_absent_unknown": 1.0, "atomic_control": 1.0, "mismatch_atomic_lure": 0.0, "mismatch_local": 0.7, "mismatch_side": 0.0, "synthetic_lookup": 1.0}` |
| `mc033_fact_claim_route_fails_both_branches` | `MC033` | `true` | `{"answer_absent_unknown": 1.0, "atomic_control": 1.0, "match_atomic": 0.5, "match_local": 0.4, "mismatch_atomic": 0.1, "mismatch_local": 0.5, "mismatch_lure": 0.4, "synthetic_lookup": 1.0}` |

## Interpretation

The ladder's central finding is narrower and more useful than another
single failed prompt: only MC012 makes the desired local-versus-learned
contrast cleanly, and MC012 does it through visible trusted/untrusted
source text. Removing that text, replacing it with calibration, using
learned parity, using visible non-status features, changing answer
interfaces, adding row codes, or using semantic source labels has not
created an unconfounded behavior substrate. Query-level operation handles
preserve direct controls and nulls in MC023, but still leave both local
and atomic operation-conflict branches below gate quality. Balanced
worked examples in MC024 repair the local operation branch but leave
the learned atomic branch below gate on the full source-disjoint run.
Constrained choices in MC025 fail even earlier: direct atomic control
and answer-absent nulls break while the atomic branch stays weak.
Numeric options in MC026 restore nulls but turn direct atomic control
into UNKNOWN and still leave the atomic branch weak.
MC027 turns the interface problem into a direct sweep: the bare-integer
format is the only 10-source smoke survivor, while answer-prefix, JSON,
A/B/C choice, and numeric-option interfaces each break controls, nulls,
or learned routing.
MC028 then tests the survivor at full-source scale and closes it before
hidden-state work: controls and nulls remain clean, but the learned
atomic operation branch falls below gate with other-number leakage.

So the current bridge law is:

> Prompt-local table authority is easy to make clean; learned-memory
> branch selection is easy only when the prompt visibly names the source
> status. Non-status repairs so far either collapse to local outputs,
> become order-sensitive, or produce mixed outputs that do not follow
> the intended rule. Query-level operations improve controls but do not
> yet cross the behavior-gate threshold because the learned branch remains
> the brittle edge under full-source evaluation. Prompt-visible choice
> constraints are not a neutral fix; they introduce their own control
> and null failures. Numeric option lists are not neutral either; they
> can convert learned recall into abstention. More generally, answer
> schemas are behavior surfaces, not passive wrappers around the same
> internal computation. A smoke-passing answer interface is still not a
> substrate until it survives full-source side-effect checks.

## Claim Boundary

Across MC010-MC033, the only clean local-versus-learned bridge contrast is MC012's prompt-visible status-label positive control; every attempted non-status or ablated repair remains blocked before hidden-state work. MC024 shows that balanced worked examples can repair the prompt-local branch while the learned atomic branch remains the limiting failure. MC025 shows that prompt-visible choice constraints add control and null failures rather than repairing that branch. MC026 shows that numeric option lists preserve nulls but turn direct atomic recall into UNKNOWN. MC027 shows that answer-interface format is itself a strong behavior surface: bare integer answers survive 10-source smoke, while prefix, JSON, choice, and numeric-option interfaces fail different gates. MC028 closes that survivor at full-source scale: controls and nulls stay clean, but the learned atomic branch falls below gate with other-number leakage. MC029 shows that removing numeric examples can improve the learned branch, but the resulting rules-only variant breaks answer-absent nulls; factor movement is not bridge repair. MC030 shows that explicit absence guards do not repair that tradeoff and can make answer-absent rows substantially worse. MC031 tests a materially different statusless checksum cue: direct controls and nulls stay clean, but invalid-checksum conflict rows still collapse to local answers. MC032 then replaces checksum validity with cross-table consistency and finds the same broader boundary: mismatch rows still avoid the learned atomic branch while side-number leakage stays absent. MC033 then tests row-local fact claims against learned atomic memory and closes the post-MC032 repair route: direct controls and nulls stay clean, but the match and mismatch branches do not implement stable routing.

This ladder does not establish a hidden signature, causal intervention, or deployable knowledge-control mechanism.
