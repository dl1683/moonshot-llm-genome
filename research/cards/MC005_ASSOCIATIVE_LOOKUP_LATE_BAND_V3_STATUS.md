# MC005 Associative Lookup Late-Band V3 Status

Status: late-band source-value control surface passed within the synthetic
associative lookup scope.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_LATE_BAND_V3.md`
- runner:
  `code/mc005_associative_lookup_late_band_v3.py`
- result:
  `results/cards/MC005/mc005_qwen3_1p7b_associative_lookup_late_band_v3_20260630T171331.json`
- result SHA256:
  `6f2e0937a5dfb590c1fd291344fd4159fc4457d3f56f63714279b4eb638e002b`

## Verdict

V3 passes as a narrow late-band source-value control surface.

Allowed claim:

> In Qwen3-1.7B synthetic key/value associative lookup prompts, layers 20-26
> form a late-band source-value control surface: masking the target value source
> path in that band reliably damages retrieval, while masking distractor or
> random source values does not.

Disallowed claims:

- this is a complete mechanism card;
- this is a single-head or single-layer mechanism;
- this generalizes beyond the tested synthetic lookup layouts;
- this proves the model stores facts in layers 20-26.

## Behavior

The V3 row bank used 96 synthetic key/value lookup rows with two layouts.

| Metric | Value |
| --- | ---: |
| rows | 96 |
| clean rows | 81 |
| discovery clean rows | 40 |
| same-layout holdout clean rows | 18 |
| arrow-layout holdout clean rows | 23 |

The clean-row floors passed.

## Discovery Selection

Discovery used only dash-colon discovery rows and target source-value masking.
`all_layers` was excluded before selection.

| Candidate band | Layers | Mean delta | Target-win loss |
| --- | --- | ---: | ---: |
| `early_0_6` | 0-6 | -0.100 | 0/40 |
| `mid_7_13` | 7-13 | -0.072 | 0/40 |
| `signature_14_18` | 14-18 | -0.044 | 1/40 |
| `late_19_23` | 19-23 | -1.069 | 5/40 |
| `late_24_27` | 24-27 | -2.594 | 20/40 |
| `late_20_26` | 20-26 | -4.345 | 29/40 |

Selected band:

```text
late_20_26
```

## Same-Layout Holdout

Dash-colon holdout, selected `late_20_26` band:

| Arm | Mean delta | Target-win loss |
| --- | ---: | ---: |
| target source-value | -4.497 | 10/18 |
| distractor source-value | +1.733 | 0/18 |
| random value | -0.118 | 0/18 |

Earlier target-band controls:

| Band | Mean delta | Target-win loss |
| --- | ---: | ---: |
| `early_0_6` | -0.069 | 0/18 |
| `mid_7_13` | -0.302 | 0/18 |
| `signature_14_18` | -0.271 | 0/18 |

## Layout Holdout

Arrow-layout holdout, selected `late_20_26` band:

| Arm | Mean delta | Target-win loss |
| --- | ---: | ---: |
| target source-value | -5.375 | 17/23 |
| distractor source-value | +3.196 | 0/23 |
| random value | -0.071 | 0/23 |

Earlier target-band controls:

| Band | Mean delta | Target-win loss |
| --- | ---: | ---: |
| `early_0_6` | +0.082 | 0/23 |
| `mid_7_13` | -0.120 | 1/23 |
| `signature_14_18` | +0.030 | 0/23 |

The effect survived the held-out prompt layout.

## Head-Group Controls

Selected `late_20_26` target source-value masking by head group:

| Split | Head group | Mean delta | Target-win loss |
| --- | --- | ---: | ---: |
| same-layout holdout | all heads | -4.497 | 10/18 |
| same-layout holdout | lower half | -0.149 | 1/18 |
| same-layout holdout | upper half | -2.201 | 3/18 |
| same-layout holdout | even heads | -1.753 | 2/18 |
| same-layout holdout | odd heads | -0.990 | 2/18 |
| layout holdout | all heads | -5.375 | 17/23 |
| layout holdout | lower half | +0.125 | 1/23 |
| layout holdout | upper half | -2.796 | 8/23 |
| layout holdout | even heads | -1.633 | 4/23 |
| layout holdout | odd heads | -0.791 | 3/23 |

The effect is not a single small head group in this audit. The upper half of
heads carries more of the effect than the lower half, but all-head masking is
substantially stronger.

## Criteria

| Criterion | Result |
| --- | --- |
| clean discovery rows at least 36 | pass |
| clean same-layout holdout rows at least 16 | pass |
| clean layout-holdout rows at least 16 | pass |
| selected band is late | pass |
| same-layout target mask reduces margin by 1.0 | pass |
| same-layout target mask flips at least 3 rows | pass |
| same-layout target mask beats distractor by 0.50 | pass |
| same-layout target mask beats random by 0.50 | pass |
| same-layout target mask beats earlier bands by 0.50 | pass |
| layout target mask reduces margin by 1.0 | pass |
| layout target mask flips at least 3 rows | pass |
| layout target mask beats distractor by 0.50 | pass |
| layout target mask beats random by 0.50 | pass |
| layout target mask beats earlier bands by 0.50 | pass |

Overall result:

```text
late-band V3 gate: pass
mechanism-card promotion: not yet
```

## Interpretation

MC005 now has a real internal control surface inside a narrow scope:

1. Source-value attention identifies the relevant value token on MC005 V1.
2. Full-path target source-value masking changes behavior strongly on MC005 V1.
3. V3 localizes most of that causal dependence to late layers 20-26 and shows
   the effect survives an arrow-layout prompt holdout.

This is not enough for the full mechanism-card contract because the reliability
atlas is still thin. Missing surfaces include:

- larger and disjoint lexicon banks;
- longer contexts and more pair counts;
- off-target side-effect rows;
- model-family replication;
- generation rather than only next-token forced-choice margin;
- finer head/path decomposition inside layers 20-26.

The next step should build MC005 V4 as a reliability atlas rather than another
single-run localization claim.
