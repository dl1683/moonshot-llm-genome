# MC005 Associative Lookup Response-Marker V28 Donor Replacement Status

Status: donor activation replacement failed; it disrupted the parent surface
and answer-absent nulls.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_RESPONSE_MARKER_V28_DONOR_REPLACEMENT.md`
- runner:
  `code/mc005_associative_lookup_response_marker_v28_donor_replacement.py`
- V25 source artifact:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v25_factorial_row_heterogeneity_20260630T205454.json`
- V25 source SHA256:
  `efdab6fad30de1762b8e92f7e46c7f4e1e160bcc988ad093c7d0bf9a5617b9cf`
- V26 source artifact:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v26_internal_row_signature_20260630T210810.json`
- V26 source SHA256:
  `ba8b890de3ca0ed518aa6e3a4d963ee4838285798a7a6a4cbf9dbcf335bcaf03`
- V27 negative-control artifact:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v27_signature_intervention_20260630T212004.json`
- V27 negative-control SHA256:
  `830914646f67105d5ee9532f0319adf6a8026ed0d17d758d7a425335c56bb605`
- result:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v28_donor_replacement_20260630T213704.json`
- result SHA256:
  `ab0b92f89c3cfcb384d8014bf15c3e2a9d269dfe4569f640d13c1bd183668c06`

## Verdict

V28 does not support causal row-signature control.

The source signature reproduced: `l20_target_value` had discovery AUC 0.9840
and holdout AUC 0.7933. The donor pools were also valid: 29 positive discovery
donors, 99 negative discovery donors, and 128 discovery rows for the random
pool.

But positive target donor replacement did not preserve the behavior surface it
was supposed to modulate. It reduced baseline target wins from 128/128 to
106/128 and collapsed parent-effect rows from 128 to 3. The apparent all-three
fraction increase, from 0.3438 to 0.6667, is therefore not a valid row-control
success; it is a denominator-collapse artifact.

The run failed with diagnostic class:

```text
source_control_failed
```

The answer-absent nulls also failed, so the stronger interpretation is:
matched donor replacement is a disruptive intervention, not a reliable control
surface.

## Row-Structure Result

| Condition | Baseline target wins | Parent-effect rows | All-three rows | All-three fraction | Median best-pair share | Median parent delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `none` | 128/128 | 128 | 44 | 0.3438 | 0.6614 | -6.3438 |
| `positive_target` | 106/128 | 3 | 2 | 0.6667 | 0.5135 | -6.9375 |
| `negative_target` | 106/128 | 6 | 0 | 0.0000 | 0.7510 | -3.2188 |
| `random_target` | 103/128 | 3 | 1 | 0.3333 | 0.8750 | -2.3750 |
| `positive_final_colon` | 104/128 | 91 | 12 | 0.1319 | 0.7722 | -4.6875 |
| `positive_distractor` | 128/128 | 128 | 47 | 0.3672 | 0.6642 | -6.0000 |

The row-fraction metrics for `positive_target`, `negative_target`, and
`random_target` are not interpretable as mechanism support because those arms
destroyed most parent-effect rows.

## Parent Source Controls Under `positive_target`

| Source arm | Target-win loss | Mean margin delta | Label |
| --- | ---: | ---: | --- |
| `target_value` | 2 | -0.1729 | `weak` |
| `distractor_value` | -14 | +1.1172 | `side_effect` |
| `random_value` | 2 | +0.0115 | `weak` |

The target source-mask effect was no longer large after donor replacement, and
the distractor source control changed 14 target-win rows in the opposite
direction. This fails target-specific source-control reliability.

## Replacement Nulls

| Seed | Null label | Arm | Target-win loss | Mean delta | Clean |
| --- | --- | --- | ---: | ---: | --- |
| 251 | `side_effect` | `positive_source_value` | 0 | -0.0716 | yes |
| 251 | `side_effect` | `positive_final_colon` | 11 | -3.4505 | no |
| 251 | `side_effect` | `random_source_value` | 0 | -0.0566 | yes |
| 257 | `side_effect` | `positive_source_value` | 0 | +0.0104 | yes |
| 257 | `side_effect` | `positive_final_colon` | 21 | -4.1947 | no |
| 257 | `side_effect` | `random_source_value` | 0 | -0.0111 | yes |

The final-colon donor replacement is a large answer-absent side effect on both
null seeds.

## Criteria

| Criterion | Result |
| --- | --- |
| V25/V26/V27 source artifacts valid | pass |
| source signature valid | pass |
| base holdout parent valid | pass |
| nominal `positive_target` all-three fraction gain at least +0.10 | pass, but denominator collapsed |
| nominal `positive_target` best-pair share drop at least 0.05 | pass, but denominator collapsed |
| `negative_target` opposes `positive_target` | pass |
| controls smaller than `positive_target` | pass |
| `positive_target` parent source controls pass | fail |
| replacement nulls clean | fail |

## Interpretation

What V28 supports:

- V26's internal row signature remains reproducible.
- The failure of V27 was not merely because the implementation could not touch
  layer-20 activations; V28 touched them strongly.
- Matched donor replacement is too disruptive for the current MC005 row-control
  claim.

What V28 rules out under this design:

- do not claim that donor replacement of `l20_target_value` controls the
  all-three row structure;
- do not interpret all-three fraction gains when the parent-effect denominator
  collapses;
- do not use final-colon donor replacement as a clean control surface.

## Next Step

The current row-signature branch now has two negative intervention families:
additive residual steering and donor activation replacement. The next MC005
mechanism attempt should either move to a write-path/attention intervention
that targets the layers-24-26 parent computation directly, or pause the row
signature branch and start a fresh behavior family with a cleaner intervention
surface.
