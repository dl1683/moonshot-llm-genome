# MC006 Parametric Fact Override V12 Real-After-Fiction Signature Status

Status: matched-surface hidden signal present, but promotion failed; holdout
balance, candidate-score/output baselines, and shuffled-label controls block
intervention.

Date: 2026-07-01

## Artifact

- preregistration:
  `research/prereg/MC006_PARAMETRIC_FACT_OVERRIDE_V12_REAL_AFTER_FICTION_SIGNATURE.md`
- runner:
  `code/mc006_parametric_fact_override_v12_real_after_fiction_signature.py`
- source behavior artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v2_repair_20260630T222815.json`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v12_real_after_fiction_signature_20260630T234703.json`
- result SHA256:
  `5594836b401c1d8e68f5b04dea7102135dcb029f99357fe3d15cc95c6b5eac56`

## Verdict

MC006 V12 does not pass the matched-surface hidden-signature gate. Do not start
intervention from this signature.

V12 tested the V2 `real_after_fiction` prompt surface, where every row says the
fictional codebook is not real-world geography and asks for the real-world
capital. This removes the explicit requested-mode confound from V11, but it
does not produce a promotable mechanism signature.

The diagnostic class is:

```text
holdout_balance_failed
```

The stricter interpretation is broader: V12 is also candidate-score confounded,
next-token output-margin confounded, and shuffled-selection confounded.

## Structural Checks

Structural checks passed for the diagnostic table:

- 30 binary `real_after_fiction` rows;
- 10 excluded lure side rows;
- 9 `true_answer` rows;
- 21 `override_answer` rows;
- discovery: 6 true, 12 override;
- calibration: 2 true, 4 override;
- holdout: 1 true, 5 override;
- no duplicate record ids.

The holdout had both labels, but only one true-answer row. That fails the
preregistered class-balance floor for promotion.

## Signature Results

| Criterion | Result |
| --- | --- |
| structural checks pass | pass |
| holdout has at least 2 rows per label | fail: 1 true, 5 override |
| selected hidden holdout AUC at least 0.85 | pass: 1.000 |
| selected hidden beats candidate-score margin by 0.02 | fail: both 1.000 |
| selected hidden beats next-token output margin by 0.02 | fail: both 1.000 |
| selected hidden beats prompt length by 0.02 | pass: 1.000 vs 0.800 |
| selected hidden beats shuffled-selection p95 by 0.05 | fail: both 1.000 |

Selected hidden candidate:

- `layer_7` at final prompt token;
- discovery AUC: 1.000;
- holdout AUC: 1.000.

Baselines:

- candidate-score margin holdout AUC: 1.000;
- next-token output-margin holdout AUC: 1.000;
- prompt-length holdout AUC: 0.800;
- override-token-count holdout AUC: 0.800;
- final-token-id holdout AUC: 0.500;
- shuffled-label selection p95: 1.000.

## Interpretation

V12 improves on V11 in one important way: the critical label no longer equals an
explicit requested-mode string. The prompt surface is matched across rows.

But the result still cannot support intervention. The source table has only one
truth-following holdout row, and the selected hidden direction is matched by
candidate scoring, next-token output margin, and shuffled-label selection.

MC006 should remain bounded as:

- V10: chat-rendered generated-answer behavior substrate passed;
- V11: chat-mode hidden signal was requested-mode/output/null-confounded;
- V12: matched-surface hidden signal was weak-holdout/output/null-confounded.

The next MC006 repair should build a larger matched-surface table with at least
two, preferably many, truth-following and fictional-code-following holdout rows
before another signature or intervention attempt.
