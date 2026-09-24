# MC005 Associative Lookup Response-Marker V23 Row Heterogeneity Status

Status: row heterogeneity diagnostic completed.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_RESPONSE_MARKER_V23_ROW_HETEROGENEITY.md`
- runner:
  `code/mc005_associative_lookup_response_marker_v23_row_heterogeneity.py`
- source V22 result:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v22_row_interaction_20260630T201924.json`
- source V22 SHA256:
  `0b10c833d27a36666691973fe2dbde05de1a1ed74ef16220a1c43aa0299b627d`
- result:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v23_row_heterogeneity_20260630T202756.json`
- result SHA256:
  `4a65a4f373e22266d6ba7a660c71d7ae8ebce660e349aef3fc720f14ba5c14a5`

## Verdict

V23 explains a meaningful part of the V22 row split. The diagnostic label is
`best_pair_position_heterogeneity`, with `target_source_position` as the
dominant family under the preregistered contrast rule.

The strongest row-level contrast is positional: early target source-value
positions have little or no all-three support, while several mid/late positions
reach 43.75 percent all-three rows. Baseline margin also has a strong contrast,
but it is secondary under the preregistered priority order.

Summary:

```text
model: Qwen/Qwen3-1.7B
source diagnostic: V22 mean_only_interaction
source parent-effect rows: 253
source all-three rows: 67
source all-three fraction: 0.2648
diagnostic label: best_pair_position_heterogeneity
dominant family: target_source_position
```

## Source Position Contrast

Eligible source-position groups had a max-minus-min all-three fraction contrast
of 0.4375.

| Target source position | Parent-effect rows | All-three rows | All-three fraction | Median best-pair share | Median single-layer residual |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 16 | 0 | 0.0000 | 0.7649 | -2.5313 |
| 11 | 15 | 0 | 0.0000 | 0.7857 | -1.6250 |
| 21 | 15 | 2 | 0.1333 | 0.6930 | -1.7500 |
| 26 | 16 | 1 | 0.0625 | 0.7749 | -2.0000 |
| 36 | 16 | 7 | 0.4375 | 0.6267 | -4.5625 |
| 56 | 16 | 7 | 0.4375 | 0.6411 | -4.5938 |
| 71 | 16 | 7 | 0.4375 | 0.6052 | -3.6563 |
| 76 | 16 | 7 | 0.4375 | 0.6243 | -3.9063 |

The same pattern appears by query-pair index because the prompt layout maps
query index to source-value position. Query index 0 had 0/16 all-three rows;
query indices 6, 10, 13, and 14 each had 7/16 all-three rows.

## Best-Pair Contrast

Best-pair winner was informative but did not reach the preregistered strong
contrast threshold.

| Best pair winner | Parent-effect rows | All-three rows | All-three fraction | Parent-flip pair-resistant rows | Median best-pair share |
| --- | ---: | ---: | ---: | ---: | ---: |
| `slice_l24_25_all` | 127 | 42 | 0.3307 | 15 | 0.6715 |
| `slice_l25_26_all` | 73 | 18 | 0.2466 | 1 | 0.7195 |
| `slice_l24_26_all_pair` | 53 | 7 | 0.1321 | 4 | 0.7639 |

The most pair-recovered parent-effect rows are those where the best
leave-one-layer-out pair is layers 24+26, omitting layer 25.

## Baseline Margin Contrast

Baseline margin was also a strong contrast: high-margin rows were more likely
to meet the all-three margin definition, while low-margin rows concentrated
the parent-flip pair-resistant cases.

| Baseline margin band | Parent-effect rows | All-three rows | All-three fraction | Parent-flip pair-resistant rows | Median best-pair share |
| --- | ---: | ---: | ---: | ---: | ---: |
| `q1_low` | 64 | 9 | 0.1406 | 18 | 0.7625 |
| `q2_mid_low` | 63 | 11 | 0.1746 | 1 | 0.7238 |
| `q3_mid_high` | 63 | 19 | 0.3016 | 1 | 0.6835 |
| `q4_high` | 63 | 28 | 0.4444 | 0 | 0.6438 |

This matters for future row-level claims: parent flips are concentrated in
low-margin rows, but all-three margin structure is most common in high-margin
rows.

## Failure Reasons

There were 186 parent-effect rows that were not all-three rows.

| Failure reason | Count |
| --- | ---: |
| `pair_share_recovered` | 186 |
| `pair_plus_single_residual_weak` | 46 |
| `single_residual_weak` | 23 |

Failure combinations:

| Failure combination | Count |
| --- | ---: |
| `pair_share_recovered` | 139 |
| `pair_share_recovered + pair_plus_single_residual_weak` | 24 |
| `pair_share_recovered + single_residual_weak` | 1 |
| `pair_share_recovered + single_residual_weak + pair_plus_single_residual_weak` | 22 |

The dominant blocker is not weak single-layer residual. Every non-all-three
parent-effect row failed because at least one leave-one-layer-out pair recovered
60 percent or more of the parent negative-effect magnitude.

## Interpretation

What V23 supports:

- V22 row heterogeneity is not random noise: it has a strong source-position
  and query-index structure;
- early source positions are mostly pair-recovered, while several mid/late
  positions are much more likely to show all-three margin structure;
- low baseline-margin rows are where parent-only behavior flips concentrate,
  while high baseline-margin rows are where all-three margin structure is most
  frequent.

What V23 does not support:

- it does not rescue V22's failed broad row-level all-three mechanism claim;
- it does not identify a smaller intervention path;
- it does not prove that source position is causal rather than correlated with
  prompt geometry, margin, or token identity.

## Next Step

The next MC005 pass should test whether source position is causal. A clean
follow-up should hold the same key/value identities and query index fixed while
moving the target pair to early versus mid/late source positions, then rerun the
parent, leave-one-layer-out pairs, and answer-absent nulls.
