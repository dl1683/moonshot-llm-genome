# MC006 Parametric Fact Override V18 Margin-Matched Lead-Time Status

Status: margin matching was blocked by globally separated candidate-score and
final-output margins; the V16 lead-time signal remains monitor-only.

Date: 2026-07-01

## Artifact

- design source:
  `research/21_CONTROL_SURFACE_LAWS_AND_GAPS.md`
- runner:
  `code/mc006_parametric_fact_override_v18_margin_matched_leadtime.py`
- source V14 behavior artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v14_parser_normalized_20260701T000538.json`
- source V14 SHA256:
  `7596e8c6869cfcb7413e9bbe8d9b6324b2ed3115021bbb1bf7091059cbac54c5`
- reference V16 signature artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v16_pre_output_position_signature_20260701T002113.json`
- reference V16 SHA256:
  `6c99826028de85d9b03b22cd8c467ccc42483e1b80ca4f1bbcab6fc67074eb31`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v18_margin_matched_leadtime_20260701T022343.json`
- result SHA256:
  `b53667cef823234498d88472c76314de3022d8bcf8fb694548aa2f415d448fd6`

## Verdict

V18 does not promote MC006 to an intervention-ready knowledge signature.

The diagnostic class is:

```text
global_margin_separation_blocks_matching
```

The important result is not that the early hidden signal disappeared. It
reproduced. The important result is that the V14 generated-answer table is so
separated by final candidate-score and final next-token output margins on
holdout that margin-matched hidden-state auditing cannot be fairly performed
on this row set.

## What V18 Tested

V18 reran feature extraction for the same 30 V14 binary rows and reproduced the
V16 selected early position:

- position: `after_mapping_line`;
- layer: 4;
- discovery AUC: 1.000;
- holdout AUC: 1.000;
- direction norm: 22.6191.

It then added three controls that V16 did not store row scores for:

- row-level selected hidden scores;
- residualized hidden scores after candidate-score and final-output margins;
- margin-overlap and matched-pair audits on discovery and holdout splits.

## Baselines

Holdout AUCs:

| Signal | Holdout AUC |
| --- | ---: |
| selected hidden `after_mapping_line/layer_4` | 1.000 |
| same-position next-token output margin | 0.917 |
| candidate-score margin | 1.000 |
| final next-token output margin | 1.000 |
| prompt length | 0.583 |
| generated first-token id | 0.833 |

Residualizing the selected hidden score against global margins erased the
holdout signal:

| Residualized Hidden Score | Holdout AUC |
| --- | ---: |
| after candidate-score margin | 0.333 |
| after final next-token output margin | 0.333 |
| after both global margins | 0.333 |

## Margin-Matching Audit

Candidate-score margin on holdout:

- true rows: 6;
- override rows: 2;
- true z-range: -0.1305 to 0.6321;
- override z-range: -0.8505 to -0.7281;
- overlap exists: false;
- separation gap: 0.5976;
- matched true/override holdout pairs at z <= 0.5: 0.

Final next-token output margin on holdout:

- true rows: 6;
- override rows: 2;
- true z-range: -0.2658 to 0.5824;
- override z-range: -1.0917 to -0.8685;
- overlap exists: false;
- separation gap: 0.6027;
- matched true/override holdout pairs at z <= 0.5: 0.

This means the V14 holdout table does not contain same-margin true/override
comparisons for either global margin. A margin-matched hidden-signature claim
cannot be made from this split.

## Criteria

| Criterion | Result |
| --- | --- |
| source artifact and structural checks pass | pass |
| selected V16 pre-output signal reproduced | pass |
| candidate-score margin has holdout overlap | fail |
| final-output margin has holdout overlap | fail |
| matched holdout pairs exist at z <= 0.5 | fail |
| hidden score beats 0.85 AUC after candidate residualization | fail |
| hidden score beats 0.85 AUC after final-output residualization | fail |
| hidden score beats 0.85 AUC after both residualizations | fail |

## Interpretation

V18 upgrades the MC006 failure from "globally output-confounded" to a sharper
table diagnosis:

- V16 showed lead-time before same-position output geometry fully caught up.
- V17 showed that the early direction is not a simple additive control vector.
- V18 shows that the same V14 row set cannot support margin-matched promotion
  because true and override holdout rows occupy separated global-margin ranges.

Allowed claim:

> MC006 V14/V16 contains a real early monitoring signal, but the current row
> table is globally margin-separated and therefore not a valid substrate for a
> margin-matched mechanism signature or intervention.

Forbidden claims:

- MC006 V18 rescues the V16 signature;
- MC006 V18 supports intervention;
- residualized hidden signal survives final-output and candidate-score
  controls;
- the current V14 row set can adjudicate margin-matched MC006 causality.

## Next Step

The next MC006 attempt should not search another final-token or same-table
classifier. It should either:

1. construct a new generated-answer table with overlapping candidate-score and
   final-output margins across true/override labels; or
2. move to source-token or path-specific lead-time audits where the claim is
   explicitly about upstream source flow rather than a scalar hidden separator.

Until one of those exists, MC006 remains:

- behavior-supported by V14;
- lead-time-monitor-supported by V16;
- additive-steering-failed by V17;
- margin-matching-blocked by V18;
- not intervention-ready.
