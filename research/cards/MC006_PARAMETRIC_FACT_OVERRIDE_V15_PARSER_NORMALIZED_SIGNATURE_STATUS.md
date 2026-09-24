# MC006 Parametric Fact Override V15 Parser-Normalized Signature Status

Status: hidden signal present, but signature promotion failed; candidate-score
and next-token output baselines match the selected hidden direction.

Date: 2026-07-01

## Artifact

- preregistration:
  `research/prereg/MC006_PARAMETRIC_FACT_OVERRIDE_V15_PARSER_NORMALIZED_SIGNATURE.md`
- runner:
  `code/mc006_parametric_fact_override_v15_parser_normalized_signature.py`
- source behavior artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v14_parser_normalized_20260701T000538.json`
- source artifact SHA256:
  `7596e8c6869cfcb7413e9bbe8d9b6324b2ed3115021bbb1bf7091059cbac54c5`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v15_parser_normalized_signature_20260701T000911.json`
- result SHA256:
  `25a68fadb36b5f60ec909f9e000c89adfb7779ba2ffeefea2b46bdfcb4e0b57c`

## Verdict

MC006 V15 does not pass the hidden-signature gate. Do not start intervention
from this signature.

The diagnostic class is:

```text
candidate_score_confounded
```

The stricter interpretation is broader: V15 is candidate-score confounded and
next-token output-margin confounded.

## Structural Checks

Structural checks passed for the selected V14 table:

- 30 binary rows;
- 10 side rows;
- 21 true-answer rows;
- 9 override-answer rows;
- discovery: 11 true, 4 override;
- calibration: 4 true, 3 override;
- holdout: 6 true, 2 override;
- unique source ids;
- no duplicate record ids.

Side rows were 3 lure answers and 7 unparsed rows.

## Signature Results

Selected hidden candidate:

- `layer_16` at final prompt token;
- discovery AUC: 1.000;
- holdout AUC: 1.000.

Baselines:

| Baseline | Discovery AUC | Holdout AUC |
| --- | ---: | ---: |
| candidate-score margin | 0.990 | 1.000 |
| next-token output margin | 1.000 | 1.000 |
| prompt length | 0.643 | 0.583 |
| final prompt token id | 0.500 | 0.500 |
| true candidate token count | 0.667 | 0.375 |
| override candidate token count | 0.643 | 0.583 |
| generated first-token id | 0.838 | 0.833 |
| parser-normalization delta | 0.533 | 0.500 |

Shuffled-label selection null:

- iterations: 100;
- holdout AUC p95: 0.833;
- holdout AUC max: 1.000;
- holdout AUC mean: 0.4475.

## Criteria

| Criterion | Result |
| --- | --- |
| source artifact and structural checks pass | pass |
| holdout has at least 2 rows per label | pass |
| selected hidden holdout AUC at least 0.85 | pass: 1.000 |
| selected hidden beats candidate-score margin by 0.02 | fail: both 1.000 |
| selected hidden beats next-token output margin by 0.02 | fail: both 1.000 |
| selected hidden beats prompt length by 0.02 | pass |
| selected hidden beats final-token id by 0.02 | pass |
| selected hidden beats shuffled-selection p95 by 0.05 | pass: 1.000 vs 0.833 |

## Interpretation

V15 is the cleanest MC006 hidden-signal diagnostic so far because it uses a
matched generated-answer table with source-disjoint holdout label balance. The
hidden separation is real, but it is still not mechanism-grade. The same labels
are perfectly visible to candidate scoring and next-token output margin.

MC006 should now be bounded as:

- V14: matched generated behavior substrate passed under a narrow normalized
  parser;
- V15: hidden signal present but candidate-score and output-margin confounded.

The next MC006 step should not be intervention. It should either construct a
table where labels are not already perfectly visible in candidate/output
margins, or run a more demanding pre-output/earlier-position diagnostic that
must beat those same baselines before any causal test is allowed.
