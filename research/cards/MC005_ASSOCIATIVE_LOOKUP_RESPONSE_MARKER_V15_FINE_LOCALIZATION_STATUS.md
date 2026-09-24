# MC005 Associative Lookup Response-Marker V15 Fine Localization Status

Status: compact localization failed; full late band still required.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_RESPONSE_MARKER_V15_FINE_LOCALIZATION.md`
- runner:
  `code/mc005_associative_lookup_response_marker_v15_fine_localization.py`
- result:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v15_fine_localization_20260630T190737.json`
- result SHA256:
  `83734ff4f8fc46ad7c43aad3f6f416d23991eaefa822df51a713d24cea78f1b0`

## Verdict

V15 does not support compact localization. The selected compact candidate was
`full_l20_26_upper_heads`, which is layers 20-26 with heads 8-15. It moved the
lookup holdout in the predicted direction and preserved the answer-absent null
holdout, but it recovered only 38.9 percent of the full-band target-mask effect.

Summary:

```text
model: Qwen/Qwen3-1.7B
marker: Response
pair count: 16
discovery seeds: 17, 23
lookup holdout seed: 31
answer-absent null holdout seeds: 37, 41
selected compact candidate: full_l20_26_upper_heads
selected effect share of full holdout: 0.3894
passed: false
diagnostic class: full_band_required
```

## Discovery Screen

| Candidate | Discovery target mean delta | Target-win loss |
| --- | ---: | ---: |
| `full_l20_26_all` | -8.8896 | 35 |
| `single_l20_all` | -0.3049 | 0 |
| `single_l21_all` | -1.0725 | 0 |
| `single_l22_all` | +0.3328 | -1 |
| `single_l23_all` | -0.1716 | 0 |
| `single_l24_all` | -0.5039 | 1 |
| `single_l25_all` | -0.9829 | 1 |
| `single_l26_all` | -0.8760 | 0 |
| `slice_l20_22_all` | -1.3376 | 0 |
| `slice_l23_24_all` | -0.4727 | 0 |
| `slice_l25_26_all` | -2.3374 | 2 |
| `full_l20_26_lower_heads` | -2.2878 | 2 |
| `full_l20_26_upper_heads` | -3.5093 | 5 |
| `full_l20_26_even_heads` | -2.0432 | 1 |
| `full_l20_26_odd_heads` | -3.3560 | 4 |

## Holdout Controls

Lookup holdout baseline:

- rows: 64;
- target wins: 63/64;
- mean margin: 10.1973;
- greedy target outputs: 49/64.

| Path | Arm | Mean delta | Target-win loss | Label |
| --- | --- | ---: | ---: | --- |
| selected upper heads | target source value | -3.3213 | 6 | target effect |
| selected upper heads | distractor source value | +0.5752 | -1 | control |
| selected upper heads | random source value | +0.0322 | 0 | control |
| full late band | target source value | -8.5283 | 21 | benchmark |
| full late band | distractor source value | +0.9863 | -1 | control |
| full late band | random source value | +0.0381 | 0 | control |

The selected upper-head path passes the direction and source-control criteria,
but its target effect is only 0.3894 of the full late-band effect. The
preregistered compact-localization threshold was 0.60.

## Null Holdout

The selected upper-head path kept the V14 pair16 answer-absent null clean:

| Seed | Label | Source value | Control value | Earlier colon | Final label | Final colon |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 37 | clean_null | +0.0225 | +0.0703 | -0.0059 | -0.1543 | +0.1953 |
| 41 | clean_null | -0.0098 | +0.0674 | -0.0117 | -0.1338 | +0.2822 |

## Interpretation

What V15 supports:

- the upper-half heads inside layers 20-26 carry a real partial source-value
  intervention effect;
- that partial path has clean target/distractor/random source controls on the
  lookup holdout;
- that partial path keeps the pair16 answer-absent null clean on two holdout
  seeds.

What V15 blocks:

- MC005 cannot yet claim a compact layer slice, single layer, single head group,
  or single-head mechanism;
- the full layers-20-26 all-head path remains the supported intervention
  surface;
- the current mechanism remains broad and distributed inside the late band.

## Next Step

The next MC005 localization pass should either:

- test finer head subsets inside the upper-half and odd-head partial paths; or
- switch to an attribution/decomposition method that can explain why the full
  late-band effect is much larger than any preregistered compact candidate.

Any follow-up must preserve V13/V14 lookup and answer-absent controls and keep
V10-V12 as negative cross-size reliability evidence.
