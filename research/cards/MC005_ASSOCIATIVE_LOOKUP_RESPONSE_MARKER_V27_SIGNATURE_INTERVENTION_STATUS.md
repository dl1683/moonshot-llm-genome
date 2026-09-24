# MC005 Associative Lookup Response-Marker V27 Signature Intervention Status

Status: additive residual intervention on the V26 internal signature failed.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_RESPONSE_MARKER_V27_SIGNATURE_INTERVENTION.md`
- runner:
  `code/mc005_associative_lookup_response_marker_v27_signature_intervention.py`
- V25 source artifact:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v25_factorial_row_heterogeneity_20260630T205454.json`
- V25 source SHA256:
  `efdab6fad30de1762b8e92f7e46c7f4e1e160bcc988ad093c7d0bf9a5617b9cf`
- V26 source artifact:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v26_internal_row_signature_20260630T210810.json`
- V26 source SHA256:
  `ba8b890de3ca0ed518aa6e3a4d963ee4838285798a7a6a4cbf9dbcf335bcaf03`
- result:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v27_signature_intervention_20260630T212004.json`
- result SHA256:
  `830914646f67105d5ee9532f0319adf6a8026ed0d17d758d7a425335c56bb605`

## Verdict

V27 does not pass the causal row-signature intervention gate.

The V26 signature remains real as a predictor: the recomputed
`l20_target_value` direction had discovery AUC 0.9840 and holdout AUC 0.7933.
But adding a +4.0-score residual delta at the target source-value token did not
make the held-out rows more all-three-like. The primary `plus_target` arm had
43/128 all-three rows, below the no-intervention baseline of 44/128, and its
median best-pair share moved in the wrong direction: 0.6700 versus 0.6614.

The diagnostic class is:

```text
plus_target_no_row_effect
```

This means V26 is a predictive internal signature, not a demonstrated causal
control surface under this additive residual intervention.

## Direction

| Field | Value |
| --- | ---: |
| signature layer | 20 |
| signature position | `target_value` |
| hidden index convention | 21 |
| discovery seed | 233 |
| holdout seed | 239 |
| discovery AUC | 0.9840 |
| holdout AUC | 0.7933 |
| score shift | +4.0 / -4.0 |
| raw delta norm | 38.2839 |

## Row-Structure Result

All rows below are V25 seed-239 parent-effect holdout rows.

| Condition | All-three rows | All-three fraction | Median best-pair share | Median singles residual | Median parent delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `none` | 44/128 | 0.3438 | 0.6614 | -3.9688 | -6.3438 |
| `plus_target` | 43/128 | 0.3359 | 0.6700 | -3.9062 | -6.2812 |
| `minus_target` | 48/128 | 0.3750 | 0.6661 | -3.9375 | -6.2812 |
| `random_target` | 46/128 | 0.3594 | 0.6641 | -4.0000 | -6.2500 |
| `plus_final_colon` | 46/128 | 0.3594 | 0.6569 | -4.0000 | -6.2812 |
| `plus_distractor` | 46/128 | 0.3594 | 0.6684 | -4.0625 | -6.3750 |

The primary intervention did not beat baseline or controls. The negative arm
also did not oppose the positive arm in the preregistered direction.

## Parent Source Controls Under `plus_target`

The parent source controls remained target-specific, so the failure is not a
source-control failure.

| Source arm | Target-win loss | Mean margin delta | Label |
| --- | ---: | ---: | --- |
| `target_value` | 11 | -6.1689 | `side_effect` |
| `distractor_value` | 0 | +0.8354 | `weak` |
| `random_value` | 0 | +0.0317 | `clean` |

## Residual Nulls

The answer-absent residual nulls were clean on both seeds.

| Seed | Baseline target wins | Arm | Target-win loss | Mean delta | Clean |
| --- | ---: | --- | ---: | ---: | --- |
| 251 | 95/96 | `plus_source_value` | 0 | -0.0111 | yes |
| 251 | 95/96 | `plus_final_colon` | 0 | +0.0319 | yes |
| 251 | 95/96 | `random_source_value` | 0 | +0.0072 | yes |
| 257 | 96/96 | `plus_source_value` | 0 | +0.0234 | yes |
| 257 | 96/96 | `plus_final_colon` | 0 | +0.0365 | yes |
| 257 | 96/96 | `random_source_value` | 0 | +0.0033 | yes |

## Criteria

| Criterion | Result |
| --- | --- |
| source signature valid | pass |
| base holdout parent valid | pass |
| `plus_target` all-three fraction gain at least +0.10 | fail |
| `plus_target` best-pair share drop at least 0.05 | fail |
| `minus_target` opposes `plus_target` | fail |
| controls smaller than `plus_target` | pass |
| `plus_target` parent source controls pass | pass |
| residual nulls clean | pass |

## Interpretation

What V27 supports:

- the V26 signature is reproducible and predictive on the held-out split;
- the additive layer-20 target-value residual delta is null-clean on the
  answer-absent residual holdouts;
- the parent source-control separation still holds under the primary residual
  intervention.

What V27 rules out under this design:

- do not claim that adding the V26 `l20_target_value` signature direction
  causally increases all-three row structure;
- do not treat the V26 signature as a mechanism card;
- do not treat simple additive residual steering of the selected signature as a
  reliable row-control surface.

## Next Step

The next MC005 row-mechanism attempt should not repeat this additive residual
test. If the row signature is still worth pursuing, it needs a new
preregistered intervention family, such as donor activation patching,
score-matched replacement, attention/write-path intervention, or a nonlinear
feature probe, while carrying V27 as a required negative control.
