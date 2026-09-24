# MC005 Associative Lookup Response-Marker V25 Factorial Row-Heterogeneity Status

Status: factorial row-heterogeneity diagnostic failed; parent/control/null
validity passed.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_RESPONSE_MARKER_V25_FACTORIAL_ROW_HETEROGENEITY.md`
- runner:
  `code/mc005_associative_lookup_response_marker_v25_factorial_row_heterogeneity.py`
- result:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v25_factorial_row_heterogeneity_20260630T205454.json`
- result SHA256:
  `efdab6fad30de1762b8e92f7e46c7f4e1e160bcc988ad093c7d0bf9a5617b9cf`

## Verdict

V25 does not pass the preregistered factorial row-heterogeneity gate. The
all-head layers-24-26 parent effect replicated, every factorial cell had valid
baselines and parent effects, source controls stayed separated, and both
answer-absent null holdouts were clean. But the designed factors did not
explain the row split.

The four target-position by distractor-relation cells were nearly flat:

- best cell: `mid_late_near`, 20/64 all-three rows, fraction 0.3125;
- worst cell: `early_far`, 17/64 all-three rows, fraction 0.2656;
- best-minus-worst range: 0.0469, below the preregistered 0.25 threshold.

Distractor relation barely modulated the source-position effect. The
late-minus-early effect was +0.0469 for near distractors and +0.0313 for far
distractors, an absolute modulation of 0.0156 against the 0.15 threshold.
Baseline margin remained directional but weak: the largest high-minus-low
within-position contrast was 0.0864, below the 0.20 threshold.

Summary:

```text
model: Qwen/Qwen3-1.7B
marker: Response
pair count: 16
lookup seeds: 233, 239
base families per seed: 32
lookup rows: 256
null seeds: 251, 257
passed: false
diagnostic class: factorial_cell_contrast_failed
```

## Path Results

| Path | Mean target delta | Target-win loss |
| --- | ---: | ---: |
| `slice_l24_26_all` | -6.1912 | 27 |
| `slice_l24_25_all` | -3.3423 | 2 |
| `slice_l24_26_all_pair` | -1.9377 | 1 |
| `slice_l25_26_all` | -2.4739 | 3 |
| `single_l24_all` | -0.4663 | 0 |
| `single_l25_all` | -1.0266 | 0 |
| `single_l26_all` | -0.8425 | 0 |

## Factorial Cells

| Cell | Target slot | Distractor slot | Parent-effect rows | All-three rows | All-three fraction | All-three families |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `early_near` | 0 | 1 | 64 | 17 | 0.2656 | 17 |
| `early_far` | 0 | 13 | 64 | 17 | 0.2656 | 17 |
| `mid_late_near` | 13 | 14 | 64 | 20 | 0.3125 | 20 |
| `mid_late_far` | 13 | 0 | 64 | 19 | 0.2969 | 19 |

All cells spanned both lookup seeds. The best cell was not a single-family
artifact, but the effect size was far too small to count as an explanatory
factorial split.

## Margin Strata

| Target group | Low-margin fraction | High-margin fraction | High-minus-low | Low rows | High rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| early | 0.2418 | 0.3243 | +0.0826 | 91 | 37 |
| mid/late | 0.2432 | 0.3297 | +0.0864 | 37 | 91 |

Baseline margin moved in the expected direction, but did not explain enough of
the row-level all-three structure under the preregistered threshold.

## Null Holdout

`slice_l24_26_all` preserved both pair16 answer-absent null seeds:

| Seed | Label | Source value | Control value | Earlier colon | Final label | Final colon |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 251 | clean_null | -0.0085 | +0.1406 | -0.0020 | -0.0358 | +0.0586 |
| 257 | clean_null | +0.0267 | +0.1302 | +0.0065 | -0.0065 | +0.0553 |

## Criteria

| Criterion | Result |
| --- | --- |
| overall lookup baseline valid | pass |
| all factorial-cell baselines valid | pass |
| all factorial-cell parent effects pass | pass |
| all factorial-cell source controls pass | pass |
| parent null holdouts clean | pass |
| every cell has at least 24 parent-effect rows | pass |
| four-cell all-three fraction range at least 0.25 | fail |
| distractor relation modulates source-position effect by at least 0.15 | fail |
| high margin beats low margin within a position group by at least 0.20 | fail |
| best cell spans at least 8 families and both seeds | pass |

## Interpretation

What V25 supports:

- the all-head layers-24-26 parent surface remains strong under a stricter
  matched factorial layout;
- source controls and answer-absent nulls remain clean;
- source position and baseline margin are directionally relevant but weak in
  this controlled layout.

What V25 blocks:

- do not claim that target source position, distractor position, and baseline
  margin explain the V22/V23 row split;
- do not promote a broad row-level all-three mechanism from these factors;
- do not treat value identity blocking by family as sufficient to explain row
  heterogeneity.

## Next Step

The next MC005 move should stop trying to explain row-level heterogeneity with
simple prompt-layout factors alone. The stronger unresolved target is whether
all-three rows are selected by an internal state feature: for example a
pre-intervention activation/logit-margin signature that predicts pair
resistance inside the already reliable parent surface.

Update: V26 tested this internal-signature target and passed the signature
gate with a layer-20 target-source residual direction. See
`research/cards/MC005_ASSOCIATIVE_LOOKUP_RESPONSE_MARKER_V26_INTERNAL_ROW_SIGNATURE_STATUS.md`.
