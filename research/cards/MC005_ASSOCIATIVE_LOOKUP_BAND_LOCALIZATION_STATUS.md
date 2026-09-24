# MC005 Associative Lookup Band-Localization Status

Status: late-band source-value effect found; preregistered band-localization
gate failed because `all_layers` was selected.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_BAND_LOCALIZATION.md`
- runner:
  `code/mc005_associative_lookup_band_audit.py`
- result:
  `results/cards/MC005/mc005_qwen3_1p7b_associative_lookup_band_localization_20260630T170539.json`
- result SHA256:
  `fef087f0e42172d2787790b48297d4f70b940ebd055a233562ff34e346ca468b`

## Verdict

V2 strengthened the MC005 source-path result but did not promote it.

The preregistered selector chose `all_layers`, which makes the result full-path
control by definition, not band-localized mechanism support.

The important diagnostic is that the late layer bands are highly active while
early and mid bands are weak.

## Fixed Split

The V2 run reproduced the MC005 clean split:

| Metric | Value |
| --- | ---: |
| rows | 48 |
| clean rows | 41 |
| discovery clean rows | 28 |
| holdout clean rows | 13 |

## Discovery Selection

Discovery target source-value masking:

| Candidate band | Layers | Mean delta | Target-win loss |
| --- | --- | ---: | ---: |
| `early_0_6` | 0-6 | -0.054 | 1/28 |
| `mid_7_13` | 7-13 | -0.114 | 1/28 |
| `signature_14_18` | 14-18 | -0.143 | 2/28 |
| `late_19_23` | 19-23 | -0.125 | 1/28 |
| `late_24_27` | 24-27 | -2.759 | 13/28 |
| `late_20_26` | 20-26 | -3.388 | 19/28 |
| `all_layers` | 0-27 | -5.071 | 25/28 |

Selected band:

```text
all_layers
```

This selection automatically blocks the band-localization pass condition.

## Holdout Results

Holdout target source-value masking:

| Candidate band | Mean delta | Target-win loss |
| --- | ---: | ---: |
| `early_0_6` | 0.000 | 0/13 |
| `mid_7_13` | -0.091 | 2/13 |
| `signature_14_18` | -0.192 | 2/13 |
| `late_19_23` | -0.495 | 1/13 |
| `late_24_27` | -2.370 | 8/13 |
| `late_20_26` | -3.976 | 8/13 |
| `all_layers` | -5.154 | 11/13 |

For the strongest non-all band, `late_20_26`:

| Arm | Mean delta | Target-win loss |
| --- | ---: | ---: |
| target source-value | -3.976 | 8/13 |
| distractor source-value | +1.615 | 0/13 |
| random value | -0.005 | 1/13 |

This is strong evidence for a late source-value path, but the preregistered
selector did not choose it over `all_layers`.

## Criteria

| Criterion | Result |
| --- | --- |
| fixed clean rows match MC005 | pass |
| selected band is not `all_layers` | fail |
| selected-band target mask reduces margin by 1.0 | pass |
| selected-band target mask flips at least 3 holdout rows | pass |
| selected-band target beats distractor by 0.50 | pass |
| selected-band target beats random by 0.50 | pass |
| selected-band target beats earlier bands by 0.50 | pass |

Overall result:

```text
band-localization gate: fail
late-band diagnostic: strong
mechanism-card promotion: fail
```

## Interpretation

V2 changes the MC005 map:

- the source-value control surface is not uniformly distributed across the
  whole network;
- layers 20-27 carry most of the causal effect;
- layers 0-18 are weak controls on this behavior;
- the full path remains stronger than any preregistered band.

This supports a next V3 audit that excludes `all_layers` from selection and
asks whether `late_20_26` or `late_24_27` survives a larger row bank, prompt
layout holdout, and head-group controls. It does not yet support claiming a
localized mechanism.
