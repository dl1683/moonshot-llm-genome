# MC005 Associative Lookup Response-Marker V19 L24-26 Decomposition Status

Status: smaller layers-24-26 decomposition failed the effect-share gate.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_RESPONSE_MARKER_V19_L24_26_DECOMPOSITION.md`
- runner:
  `code/mc005_associative_lookup_response_marker_v19_l24_26_decomposition.py`
- result:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v19_l24_26_decomposition_20260630T194614.json`
- result SHA256:
  `e4906dc205105423edd7c6b07c751dda5e0458d0eca92f2843c959e7064fa070`

## Verdict

V19 fails the preregistered layers-24-26 decomposition gate. The selected
smaller path, `slice_l24_25_all`, was directional, source-specific,
rank-stable, and answer-absent-null-clean, but it recovered only 49.88 percent
of the parent `slice_l24_26_all` holdout effect. The V18 all-head layers-24-26
block remains the current supported MC005 localization surface.

Summary:

```text
model: Qwen/Qwen3-1.7B
marker: Response
pair count: 16
discovery seeds: 89, 97
lookup holdout seeds: 101, 103
answer-absent null seeds: 107, 109
discovery rows: 128
lookup holdout rows: 192
parent candidate: slice_l24_26_all
selected candidate: slice_l24_25_all
selected effect share of parent: 0.4988
passed: false
diagnostic class: l24_26_block_still_required
```

## Discovery

Selection used only discovery target source-value masks. The selected candidate
was `slice_l24_25_all`.

| Path | Discovery mean delta | Target-win loss |
| --- | ---: | ---: |
| `full_l20_26_all` | -8.8325 | 44 |
| `slice_l23_26_all` | -6.2246 | 20 |
| `slice_l24_26_all` | -5.8120 | 14 |
| `slice_l24_25_all` | -2.8882 | 3 |
| `l24_26_upper_heads` | -2.2769 | 3 |
| `slice_l25_26_all` | -2.2549 | 7 |
| `l24_26_heads_12_15` | -1.9097 | 2 |
| `l24_26_lower_heads` | -1.8477 | 3 |

No tested single layer or layer-head intersection came close to the parent
layers-24-26 all-head block on discovery rows.

## Lookup Holdout

| Path | Holdout mean delta | Target-win loss |
| --- | ---: | ---: |
| `full_l20_26_all` | -9.0951 | 57 |
| `slice_l23_26_all` | -6.5716 | 30 |
| `slice_l24_26_all` | -5.9502 | 27 |
| `slice_l24_25_all` | -2.9681 | 9 |
| `slice_l25_26_all` | -2.2786 | 5 |
| `l24_26_upper_heads` | -2.2383 | 5 |
| `l24_26_lower_heads` | -1.9733 | 5 |
| `l24_26_heads_12_15` | -1.9339 | 5 |
| `l24_26_odd_heads` | -1.8955 | 5 |
| `l24_26_even_heads` | -1.5518 | 4 |

`slice_l24_25_all` remained the strongest selectable holdout path, but it
recovered only 49.88 percent of the parent layers-24-26 target-mask mean-delta
magnitude. The preregistered threshold was 60 percent.

Source controls:

| Path | Arm | Mean delta | Target-win loss |
| --- | --- | ---: | ---: |
| `slice_l24_26_all` | target source value | -5.9502 | 27 |
| `slice_l24_26_all` | distractor source value | +0.9972 | 0 |
| `slice_l24_26_all` | random source value | -0.0003 | 1 |
| `slice_l24_25_all` | target source value | -2.9681 | 9 |
| `slice_l24_25_all` | distractor source value | +0.4928 | 0 |
| `slice_l24_25_all` | random source value | -0.0023 | 1 |

## Null Holdout

`slice_l24_25_all` preserved the pair16 answer-absent null:

| Seed | Label | Source value | Control value | Earlier colon | Final label | Final colon |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 107 | clean_null | +0.0114 | +0.0482 | +0.0111 | -0.0231 | -0.0225 |
| 109 | clean_null | -0.0072 | +0.0312 | -0.0072 | -0.0137 | -0.0579 |

## Criteria

Passed:

- parent holdout effect present;
- parent source controls passed;
- selected candidate holdout effect present;
- selected candidate source controls passed;
- selected candidate remained strongest among selectable holdout paths;
- selected candidate null holdouts were clean.

Failed:

- selected candidate recovered only 49.88 percent of the parent effect, below
  the preregistered 60 percent threshold.

## Interpretation

What V19 supports:

- `slice_l24_25_all` is a stable partial path inside the layers-24-26 surface;
- the selected partial path is source-specific and null-clean under V19
  controls;
- tested single layers, lower/upper layer-head intersections, and coarse
  parent-level head partitions do not replace the all-head layers-24-26 block.

What V19 does not support:

- it does not narrow the supported MC005 surface below all-head layers 24-26;
- it does not localize the mechanism to a single layer, head partition, or head;
- it does not prove model-family generality;
- it does not repair the Qwen3-0.6B strict answer-absent boundary;
- it does not prove reliability outside the synthetic associative lookup
  contract.

## Next Step

The next MC005 localization pass should treat layers 24-26 as the current
supported block and diagnose the interaction that makes layer 26 necessary:

- run an additivity or omission diagnostic inside layers 24-26;
- compare `24+25`, `25+26`, `24+26`, and parent `24+25+26` effects on the
  same rows;
- preserve the split discovery/holdout/null discipline before promoting any
  smaller path;
- keep V10-V12 as negative cross-size reliability evidence.
