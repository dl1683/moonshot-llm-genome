# MC008 Symbolic Fact-Code Arbitration Behavior Status

Status: symbolic_conflict_contrast_absent.

Date: 2026-07-01

## Artifact

- runner:
  `code/mc008_symbolic_fact_code_arbitration.py`
- result:
  `results/cards/MC008/mc008_qwen3_1p7b_symbolic_fact_code_arbitration_v2_null_authority_repair_20260701T062249.json`
- SHA256:
  `63A864747233F472A8A1C6C909EAE05B675157D6DBB1F28206DA83D8C8D5D5A0`

## Verdict

The behavior gate did not pass. The result is a behavior diagnostic
only, and hidden-state work remains forbidden for this route.

V2 is the one materially different null/authority repair allowed after V1.

It succeeded on the null boundary:

- answer-absent null rows were 40/40 `UNKNOWN`;
- null parseability was 40/40;
- synthetic lookup stayed clean at 39/40 artificial-code rows;
- real-world symbol recall improved to 40/40 real-symbol rows.

It failed the bridge boundary:

- primary conflict rows stayed table-dominant: 220 artificial-code rows versus
  only 2 real-symbol rows and 0 lure-symbol rows;
- non-holdout real/lure conflict balance failed;
- holdout real/lure conflict balance failed;
- even authority-0 rows produced only 1/40 real-symbol answers.

So V2 proves that MC008's V1 null failure was repairable, but the deeper
symbolic bridge failure is authority conflict: once a task table contains a
matching artificial code, Qwen3-1.7B overwhelmingly returns that code under
this symbolic interface, even when the prompt says ordinary chemistry should
control.

## Selected Template

- selected template: `membership_authority_split`
- selection key: `[0.975, 0.9625, 222.0, 2, 0, 0]`

## Gate Criteria

| Criterion | Passed |
| --- | --- |
| `smoke_mode` | `false` |
| `structural_passed` | `true` |
| `full_source_count_is_40` | `true` |
| `source_disjoint_holdout` | `true` |
| `selected_prompt_audit_passed` | `true` |
| `synthetic_panel_a_artificial_at_least_90p` | `true` |
| `real_world_panel_c_real_or_lure_at_least_85p` | `true` |
| `real_world_panel_c_parseable_at_least_95p` | `true` |
| `answer_absent_panel_f_parseable_at_least_95p` | `true` |
| `answer_absent_panel_f_unknown_at_least_90p` | `true` |
| `primary_conflict_binary_rows_at_least_40` | `true` |
| `non_holdout_conflict_artificial_at_least_10` | `true` |
| `non_holdout_conflict_real_or_lure_at_least_10` | `false` |
| `holdout_conflict_artificial_at_least_4` | `true` |
| `holdout_conflict_real_or_lure_at_least_4` | `false` |
| `primary_conflict_parseability_at_least_90p` | `true` |
| `candidate_and_output_margins_reported` | `false` |
| `non_holdout_conflict_label_balance_passed` | `false` |
| `holdout_conflict_label_balance_passed` | `false` |

## Selected Controls

| Panel | Rows | Parseable | Key Label Rate |
| --- | ---: | ---: | ---: |
| `synthetic_code_lookup` | 40 | 1.000 | 0.975 |
| `real_world_memory_control` | 40 | 1.000 | 1.000 |
| `answer_absent_null` | 40 | 1.000 | 1.000 |

## Primary Conflict

- rows: 240
- parseable rate: 0.963
- artificial-code rows: 220
- real/lure-symbol rows: 2
- binary conflict rows: 222

## Allowed Claims

- This run documents the exact MC008 behavior boundary reached by
  the selected prompt contract.
- The V2 membership/null repair fixes answer-absent symbolic-code null
  behavior without damaging synthetic lookup or real-symbol recall controls.
- The V2 authority repair fails to create artificial-versus-real contrast:
  low-authority rows still mostly emit task-table codes.
- The first MC008 symbolic route is not hidden-state-ready and should close as
  a diagnostic bridge unless a new behavior family is introduced.

## Forbidden Claims

- MC008 is a mechanism card.
- MC008 supports intervention.
- MC008 found a truth vector or general factual-recall mechanism.
- Any hidden-state or causal claim follows from this behavior run alone.
- V2 repaired the synthetic-to-factual bridge.
- The authority dial is a valid causal control surface.

## Next Decision

Close the first MC008 symbolic route as a diagnostic bridge.

Do not continue prompt-only repairs inside this route. The next bridge attempt
must change the behavior family, the answer representation, or the conflict
construction more materially than another instruction/template rewrite.
