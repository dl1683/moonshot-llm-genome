# MC005 Associative Lookup Response-Marker V16 Path Additivity Status

Status: full-band effect is mixed-superadditive; layers 23-26 are the main new localization lead.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_RESPONSE_MARKER_V16_PATH_ADDITIVITY.md`
- runner:
  `code/mc005_associative_lookup_response_marker_v16_path_additivity.py`
- result:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v16_path_additivity_20260630T191623.json`
- result SHA256:
  `cd08c112636f183041b2fe0084bd34029f264e5138b2a903a16f034200571be2`

## Verdict

V16 explains why V15's compact candidates were too weak: the full late-band
effect is not the simple sum of independent small layer/head pieces. It is
superadditive across head partitions and most layer partitions.

Summary:

```text
model: Qwen/Qwen3-1.7B
marker: Response
pair count: 16
seeds: 17, 23, 31
rows per seed: 128
rows: 384
baseline target wins: 381/384
full layers 20-26 all-head target mean delta: -8.6110
full layers 20-26 all-head target-win loss: 112
diagnostic class: mixed_superadditive
```

## Path Effects

| Path | Target mean delta | Target-win loss |
| --- | ---: | ---: |
| `full_l20_26_all` | -8.6110 | 112 |
| `full_l20_26_lower_heads` | -2.0614 | 15 |
| `full_l20_26_upper_heads` | -3.4473 | 20 |
| `full_l20_26_even_heads` | -2.0111 | 11 |
| `full_l20_26_odd_heads` | -3.2293 | 21 |
| `slice_l20_22_all` | -1.2790 | 1 |
| `slice_l23_24_all` | -0.4369 | 3 |
| `slice_l25_26_all` | -2.3381 | 17 |
| `slice_l20_24_all` | -1.6781 | 9 |
| `slice_l23_26_all` | -6.2678 | 51 |
| `slice_l20_22_l25_26_all` | -4.1667 | 33 |

## Additivity Tests

| Family | Component sum | Full delta | Residual | Residual fraction | Label |
| --- | ---: | ---: | ---: | ---: | --- |
| lower + upper heads | -5.5086 | -8.6110 | -3.1024 | -0.3603 | superadditive_full |
| even + odd heads | -5.2404 | -8.6110 | -3.3706 | -0.3914 | superadditive_full |
| layers 20-22 + 23-24 + 25-26 | -4.0540 | -8.6110 | -4.5570 | -0.5292 | superadditive_full |
| layers 20-24 + 25-26 | -4.0161 | -8.6110 | -4.5949 | -0.5336 | superadditive_full |
| layers 20-22 + 23-26 | -7.5468 | -8.6110 | -1.0642 | -0.1236 | additive |
| layers 20-22 + 25-26, omitting 23-24 | -3.6170 | -8.6110 | -4.9940 | -0.5800 | superadditive_full |

The near-additive split is `slice_l20_22_all` plus `slice_l23_26_all`. That
makes layers 23-26 the main current localization lead. It is not a complete
mechanism claim: V16 did not carry null holdouts for `slice_l23_26_all`, and it
did not preregister `slice_l23_26_all` as a promotion candidate.

## Controls

The target effect remained source-specific at the full path and at both head
halves:

| Path | Target delta | Distractor delta | Random delta |
| --- | ---: | ---: | ---: |
| full layers 20-26 all heads | -8.6110 | +1.0082 | +0.0238 |
| lower heads | -2.0614 | +0.4034 | -0.0028 |
| upper heads | -3.4473 | +0.5637 | +0.0202 |

## Interpretation

What V16 supports:

- the full late-band intervention is larger than the sum of lower/upper or
  even/odd head partitions;
- the full late-band intervention is larger than the sum of the small layer
  slices tested in V15;
- layers 23-26 form a stronger block than the previously tested `slice_l23_24`
  and `slice_l25_26` components separately;
- the next locality target should be `slice_l23_26_all`, not a single V15
  compact candidate.

What V16 blocks:

- the V15 compact-localization failure cannot be explained as merely choosing
  the wrong half-head partition;
- MC005 still cannot claim a single-head, single-layer, or small-slice
  mechanism;
- V16 alone does not prove `slice_l23_26_all` is reliable, because null
  holdouts were not part of this additivity diagnostic.

## Next Step

The next MC005 pass should preregister `slice_l23_26_all` as a new localization
candidate and test it on disjoint lookup and answer-absent null holdouts against
the full layers-20-26 benchmark. It should preserve V13/V14 controls and keep
V10-V12 as negative cross-size evidence.
