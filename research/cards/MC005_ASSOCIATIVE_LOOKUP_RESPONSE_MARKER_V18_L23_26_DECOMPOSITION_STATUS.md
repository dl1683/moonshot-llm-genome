# MC005 Associative Lookup Response-Marker V18 L23-26 Decomposition Status

Status: layers 24-26 decomposition passed under split discovery, lookup holdout, and null controls.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_RESPONSE_MARKER_V18_L23_26_DECOMPOSITION.md`
- runner:
  `code/mc005_associative_lookup_response_marker_v18_l23_26_decomposition.py`
- result:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v18_l23_26_decomposition_20260630T193525.json`
- result SHA256:
  `66dcb5bd38c35517a10f06633ba9cb179ee4ffd1e9a477c95ec35898d50c3dec`

## Verdict

V18 passes the preregistered layers-23-26 decomposition gate. It narrows the
supported Qwen3-1.7B intervention surface from layers 23-26 to the all-head
layers-24-26 block under the current synthetic associative lookup contract.

Summary:

```text
model: Qwen/Qwen3-1.7B
marker: Response
pair count: 16
discovery seeds: 61, 67
lookup holdout seeds: 71, 73
answer-absent null seeds: 79, 83
discovery rows: 128
lookup holdout rows: 192
parent candidate: slice_l23_26_all
selected candidate: slice_l24_26_all
selected effect share of parent: 0.9242
passed: true
diagnostic class: compact_l23_26_decomposition_supported
```

## Discovery

Selection used only discovery target source-value masks. The selected candidate
was `slice_l24_26_all`.

| Path | Discovery mean delta | Target-win loss |
| --- | ---: | ---: |
| `full_l20_26_all` | -8.7739 | 39 |
| `slice_l23_26_all` | -6.3159 | 19 |
| `slice_l24_26_all` | -5.7490 | 17 |
| `slice_l23_25_all` | -2.9453 | 5 |
| `slice_l24_25_all` | -2.8003 | 5 |
| `slice_l25_26_all` | -2.4502 | 7 |
| `l23_26_upper_heads` | -2.2202 | 4 |
| `l23_26_lower_heads` | -2.1426 | 6 |

Single layers and head quartiles were weaker than the selected three-layer
block on the discovery split.

## Lookup Holdout

| Path | Holdout mean delta | Target-win loss |
| --- | ---: | ---: |
| `full_l20_26_all` | -8.9644 | 59 |
| `slice_l23_26_all` | -6.4159 | 33 |
| `slice_l24_26_all` | -5.9295 | 23 |
| `slice_l23_25_all` | -3.1203 | 7 |
| `slice_l24_25_all` | -3.0239 | 7 |
| `slice_l25_26_all` | -2.3296 | 4 |
| `l23_26_lower_heads` | -2.3182 | 5 |
| `l23_26_upper_heads` | -2.2192 | 2 |
| `l23_26_heads_12_15` | -2.0059 | 2 |
| `l23_26_heads_4_7` | -1.8055 | 5 |

`slice_l24_26_all` recovered 92.42 percent of the parent layers-23-26
target-mask mean-delta magnitude on disjoint holdout rows. It was also the
strongest selectable holdout path.

Source controls:

| Path | Arm | Mean delta | Target-win loss |
| --- | --- | ---: | ---: |
| `slice_l23_26_all` | target source value | -6.4159 | 33 |
| `slice_l23_26_all` | distractor source value | +1.0049 | -3 |
| `slice_l23_26_all` | random source value | +0.0072 | 0 |
| `slice_l24_26_all` | target source value | -5.9295 | 23 |
| `slice_l24_26_all` | distractor source value | +0.9631 | -2 |
| `slice_l24_26_all` | random source value | +0.0081 | 0 |

## Null Holdout

`slice_l24_26_all` preserved the pair16 answer-absent null:

| Seed | Label | Source value | Control value | Earlier colon | Final label | Final colon |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 79 | clean_null | -0.0007 | +0.0579 | -0.0039 | -0.0065 | +0.0417 |
| 83 | clean_null | +0.0072 | +0.0612 | +0.0059 | -0.0443 | +0.0592 |

## Criteria

All preregistered criteria passed:

- parent holdout effect present;
- parent source controls passed;
- selected candidate holdout effect present;
- selected candidate source controls passed;
- selected candidate recovered at least 60 percent of parent effect;
- selected candidate remained strongest among selectable holdout paths;
- selected candidate null holdouts were clean.

## Interpretation

What V18 supports:

- MC005's Qwen3-1.7B `Response:` lookup intervention can be narrowed from
  layers 23-26 to all-head layers 24-26;
- layer 23 is not required for the tested pair16 source-value masking effect;
- coarse head partitions inside layers 23-26 did not recover the parent effect;
- the selected layers-24-26 block preserved source specificity and
  answer-absent null reliability on fresh seeds.

What V18 does not support:

- it does not localize the mechanism to a single layer, head partition, or head;
- it does not prove model-family generality;
- it does not repair the Qwen3-0.6B strict answer-absent boundary;
- it does not prove reliability outside the synthetic associative lookup
  contract.

## Next Step

The next MC005 localization pass should decompose layers 24-26:

- compare layers 24-25, 25-26, single layers 24/25/26, and layer-head
  intersections on fresh split holdouts;
- carry forward target/distractor/random source controls and the pair16
  answer-absent null;
- retain the V18 discovery/holdout split so selected subpaths are not promoted
  from the same rows that measure them;
- keep V10-V12 as negative cross-size reliability evidence.
