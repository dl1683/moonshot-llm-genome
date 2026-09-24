# MC008 Symbolic Fact-Code Arbitration Behavior Status

Status: symbolic_null_control_failed.

Date: 2026-07-01

## Artifact

- runner:
  `code/mc008_symbolic_fact_code_arbitration.py`
- result:
  `results/cards/MC008/mc008_qwen3_1p7b_symbolic_fact_code_arbitration_behavior_20260701T061006.json`
- SHA256:
  `9A586E0265529D4F47958DFAD09E9EF3DE2389A7A33594EB3E8B73631D8FA65A`

## Verdict

The behavior gate did not pass. The result is a behavior diagnostic
only, and hidden-state work remains forbidden for this route.

The useful finding is not that symbolic codes failed generically. The selected
`symbol_field` template fixed two problems that killed earlier bridge work:

- synthetic prompt-local code lookup was clean enough at 39/40 artificial-code
  rows;
- real-world chemical-symbol recall was clean enough at 39/40 real-symbol
  rows.

That means compact symbolic answers can repair direct generated-answer controls
relative to open city generation.

The route still failed because the controls that matter for mechanism search
did not compose:

- answer-absent null rows were parseable but only 30/40 `UNKNOWN`;
- primary conflict rows were only 80.4 percent parseable;
- authority pressure remained table-dominant, with 176 artificial-code rows
  versus only 8 real-symbol rows and 0 lure-symbol rows;
- holdout real/lure balance was absent.

So MC008 V1 moves the bridge frontier, but it does not create a
hidden-state-ready bridge substrate.

## Selected Template

- selected template: `symbol_field`
- selection key: `[0.75, 0.8041666666666667, 184.0, 8, 0, -1]`

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
| `answer_absent_panel_f_unknown_at_least_90p` | `false` |
| `primary_conflict_binary_rows_at_least_40` | `true` |
| `non_holdout_conflict_artificial_at_least_10` | `true` |
| `non_holdout_conflict_real_or_lure_at_least_10` | `false` |
| `holdout_conflict_artificial_at_least_4` | `true` |
| `holdout_conflict_real_or_lure_at_least_4` | `false` |
| `primary_conflict_parseability_at_least_90p` | `false` |
| `candidate_and_output_margins_reported` | `false` |
| `non_holdout_conflict_label_balance_passed` | `false` |
| `holdout_conflict_label_balance_passed` | `false` |

## Selected Controls

| Panel | Rows | Parseable | Key Label Rate |
| --- | ---: | ---: | ---: |
| `synthetic_code_lookup` | 40 | 1.000 | 0.975 |
| `real_world_memory_control` | 40 | 0.975 | 0.975 |
| `answer_absent_null` | 40 | 1.000 | 0.750 |

## Primary Conflict

- rows: 240
- parseable rate: 0.804
- artificial-code rows: 176
- real/lure-symbol rows: 8
- binary conflict rows: 184

## Allowed Claims

- This run documents the exact MC008 behavior boundary reached by
  the selected prompt contract.
- Compact symbolic generated answers can make direct synthetic lookup and
  real-symbol recall controls clean on Qwen3-1.7B.
- Null reliability and conflict balance remain binding gates even when the
  output class is short and parseable.

## Forbidden Claims

- MC008 is a mechanism card.
- MC008 supports intervention.
- MC008 found a truth vector or general factual-recall mechanism.
- Any hidden-state or causal claim follows from this behavior run alone.

## Next Decision

Do not probe MC008 V1 hidden states.

One materially different null/authority repair is justified only if it keeps:

- synthetic lookup at or above 90 percent artificial-code adherence;
- real-world symbol recall at or above 85 percent real/lure and 95 percent
  parseability;
- no true-symbol prompt leaks;
- source-disjoint holdout.

It must repair:

- answer-absent null `UNKNOWN` to at least 90 percent;
- primary conflict parseability to at least 90 percent;
- non-holdout and holdout artificial-versus-real/lure balance.

If that repair fails, close the first symbolic bridge route as diagnostic
rather than continue prompt polishing.
