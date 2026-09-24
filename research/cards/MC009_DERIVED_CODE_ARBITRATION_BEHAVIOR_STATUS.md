# MC009 Derived-Code Arbitration Behavior Status

Status: first derived-code route closed as diagnostic; hidden-state work forbidden.

Date: 2026-07-01

## Artifacts

- runner:
  `code/mc009_derived_code_arbitration.py`
- primary 5-source smoke:
  `results/cards/MC009/mc009_qwen3_1p7b_derived_code_arbitration_behavior_20260701T064442.json`
- primary 5-source SHA256:
  `51E266522C8FE82BB99F4AD6203CC703A5D54F4623087A5EBB6BE73B9763D858`
- failed `format_repair_v2` smoke:
  `results/cards/MC009/mc009_qwen3_1p7b_derived_code_arbitration_behavior_20260701T065157.json`
- failed `format_repair_v2` SHA256:
  `75B273D9253C05BC7EFD537C9057580B04FEBF5E56748B0150B24E3E93496F80`
- failed `code_slot_repair` smoke:
  `results/cards/MC009/mc009_qwen3_1p7b_derived_code_arbitration_behavior_20260701T065345.json`
- failed `code_slot_repair` SHA256:
  `9CFE7E879028EEF0C62838CABA2EC58BBE041DB701357C2B2B589AA7B9188118`
- 10-source membership smoke:
  `results/cards/MC009/mc009_qwen3_1p7b_derived_code_arbitration_behavior_20260701T065442.json`
- 10-source membership SHA256:
  `9A18EEFFAB1719D969B2FDE8CA772575B6045E47EBFF7C8B3477ABA09D7E6040`
- 10-source typed-slot V2 smoke:
  `results/cards/MC009/mc009_qwen3_1p7b_derived_code_arbitration_behavior_20260701T070111.json`
- 10-source typed-slot V2 SHA256:
  `ACC396B877D332F8B72E0877CCF4583A58785623729776E87BC447E0B1D4F5B8`

## Verdict

MC009 now has a runnable behavior harness and structurally valid prompt bank.
The first derived-code route is closed as a diagnostic bridge, not a
mechanism-promotion route.

The important result is the 10-source `membership_authority_split` smoke. It
shows that removing the direct `entity -> artificial code` row repairs the
direct controls:

- synthetic ordinal lookup: 10/10 derived-code rows;
- real-world memory control: 10/10 real-symbol rows;
- answer-absent null: 10/10 `UNKNOWN` rows;
- structural and prompt-leak audits pass.

But the bridge is still not clean:

- primary conflict rows are 40 derived-code, 7 real-symbol, 0 lure-symbol, and
  13 unparsed;
- primary conflict parseability is 47/60, or 78.3 percent, below the
  90 percent gate;
- non-holdout real/lure conflict balance fails;
- holdout real/lure conflict balance fails;
- no candidate/output margins were reported because this is still a smoke
  screen, not the full behavior gate.

So the membership route preserves controls but does not produce a clean bridge.

## Failed Repairs

Two bounded format repairs were tested and rejected.

`format_repair_v2` added explicit step-by-step row-code instructions and a
harder output contract. It made the model start explanations such as
"To determine the row code..." and collapsed synthetic ordinal lookup to 0/5
derived-code rows.

`code_slot_repair` used a terse `Code:` slot. It raised primary-conflict
parseability to 30/30 on five sources, but broke answer-absent nulls:
null rows produced 0/5 `UNKNOWN` and mostly row codes. It also produced many
wrong derived codes.

Both repairs are diagnostic only. They are kept selectable in the runner for
reproducibility, but are not part of the default template bank.

## Typed-Slot V2

After the ordinary format repairs failed, one materially different conflict
construction was tested: `typed_slot_v2`.

This template uses source-typed answer slots such as `Row code:` and
`Chemical symbol:`. It finally created source-disjoint conflict balance on the
10-source smoke:

- primary conflict parseability: 54/60, or 90.0 percent;
- primary conflict rows: 23 derived-code, 19 real-symbol, 0 lure-symbol,
  12 wrong-derived-code, and 6 unparsed;
- non-holdout conflict balance passed;
- holdout conflict balance passed.

But it broke direct controls:

- synthetic ordinal lookup was 0/10 derived-code rows because the model wrote
  row numbers such as `1`, `2`, `3`, and `4`;
- answer-absent null was 0/10 `UNKNOWN` because the model wrote explanatory
  absent-query text;
- the source channel is prompt-visible by construction.

So typed slots show the bridge can be forced at the behavior surface, but only
by changing the output interface into a prompt-visible source selector and by
breaking controls. This is not a hidden-state substrate.

## Current Best Template

- selected template: `membership_authority_split`
- 10-source selection key: `[1.0, 0.7833333333333333, 47.0, 6, 1, 0]`

## 10-Source Gate Criteria

| Criterion | Passed |
| --- | --- |
| `smoke_mode` | `true` |
| `structural_passed` | `true` |
| `full_source_count_is_40` | `false` |
| `source_disjoint_holdout` | `true` |
| `selected_prompt_audit_passed` | `true` |
| `synthetic_panel_a_derived_at_least_90p` | `true` |
| `real_world_panel_c_real_or_lure_at_least_85p` | `true` |
| `real_world_panel_c_parseable_at_least_95p` | `true` |
| `answer_absent_panel_f_parseable_at_least_95p` | `true` |
| `answer_absent_panel_f_unknown_at_least_90p` | `true` |
| `primary_conflict_binary_rows_at_least_40` | `true` |
| `non_holdout_conflict_derived_at_least_10` | `true` |
| `non_holdout_conflict_real_or_lure_at_least_10` | `false` |
| `holdout_conflict_derived_at_least_4` | `true` |
| `holdout_conflict_real_or_lure_at_least_4` | `false` |
| `primary_conflict_parseability_at_least_90p` | `false` |
| `candidate_and_output_margins_reported` | `false` |
| `non_holdout_conflict_label_balance_passed` | `false` |
| `holdout_conflict_label_balance_passed` | `false` |

## Allowed Claims

- MC009 has a live runner with structural, prompt-leak, direct-mapping,
  row-position, null, split, parseability, and template-selection audits.
- Removing direct `entity -> artificial code` source-value rows preserves
  synthetic lookup, real-symbol recall, and answer-absent null controls under
  the current 10-source smoke.
- The current derived-code bridge still fails before hidden-state work because
  primary conflict parseability and real/lure label balance are below gate.
- Step-style and terse code-slot repairs are not viable next defaults.
- Typed answer slots can create apparent conflict balance, but only as
  prompt-visible behavior control and with broken direct controls.
- The first MC009 route is closed as a diagnostic control/conflict tradeoff.

## Forbidden Claims

- MC009 is a mechanism card.
- MC009 supports hidden-state probing.
- MC009 supports intervention.
- MC009 found a truth vector or general factual-recall mechanism.
- Any hidden-state or causal claim follows from these smoke runs.
- The five- or ten-source smoke proves the full MC009 behavior substrate.
- Typed slots are evidence for an internal control surface.

## Next Decision

Do not run hidden-state work.

Do not continue ordinary prompt-only repairs inside the first MC009 route.

The next bridge attempt must change the conflict family more materially than
row-code derivation or typed answer slots, or the project should move to a
different bridge family.
