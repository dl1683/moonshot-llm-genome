# MC005 Associative Lookup Response-Marker V16 Path Additivity Preregistration

Date: 2026-06-30

## Question

MC005 V15 found that the best preregistered compact candidate inside the
Qwen3-1.7B late-band surface was `full_l20_26_upper_heads`. It was directional
and null-clean, but it recovered only 38.9 percent of the full layers-20-26
all-head holdout effect.

V16 asks:

> Is the full late-band effect larger because it is distributed additively over
> many heads/layers, because components interact superadditively, or because
> the compact candidate screen missed a simple partition?

## Fixed Surface

V16 keeps the same positive Qwen3-1.7B surface:

- model: `Qwen/Qwen3-1.7B`;
- task: synthetic associative lookup;
- pair count: 16;
- final answer marker: `Response:`;
- intervention: eager-attention source mask at the final answer position.

V16 uses lookup rows only. V14 already established the pair16 answer-absent
null for this surface, and V15 carried that null holdout for the selected
upper-head candidate.

## Rows

Lookup seeds:

- 17;
- 23;
- 31.

Each seed contributes 128 rows from `lookup_pair16_response`. The row bank is
not split for selection; V16 does not select a winner. It estimates
decomposition on a fixed diagnostic bank.

## Paths

All paths use target source-value masking unless otherwise stated.

Head partitions:

- `full_l20_26_all`: layers 20-26, all heads;
- `full_l20_26_lower_heads`: layers 20-26, heads 0-7;
- `full_l20_26_upper_heads`: layers 20-26, heads 8-15;
- `full_l20_26_even_heads`: layers 20-26, even heads;
- `full_l20_26_odd_heads`: layers 20-26, odd heads.

Layer partitions:

- `slice_l20_22_all`: layers 20-22, all heads;
- `slice_l23_24_all`: layers 23-24, all heads;
- `slice_l25_26_all`: layers 25-26, all heads;
- `slice_l20_24_all`: layers 20-24, all heads;
- `slice_l23_26_all`: layers 23-26, all heads;
- `slice_l20_22_l25_26_all`: layers 20-22 and 25-26, all heads.

Controls:

- full-band distractor source-value mask;
- full-band random source-value mask;
- upper-head distractor and random source-value masks;
- lower-head distractor and random source-value masks.

## Metrics

For every path:

- baseline target-minus-distractor margin;
- arm mean delta;
- target-win loss;
- row-level delta list.

For each partition family:

- sum of component mean deltas;
- full mean delta;
- residual: `full_mean_delta - sum(component_mean_deltas)`;
- residual fraction: `residual / abs(full_mean_delta)`.

Partition families:

- lower + upper heads;
- even + odd heads;
- layer slices 20-22 + 23-24 + 25-26;
- layer split 20-24 + 25-26;
- layer split 20-22 + 23-26;
- layer split 20-22 + 25-26 versus full, with middle slice omitted as a
  non-contiguous sanity check.

## Diagnostic Classes

For each partition family:

- `additive`: absolute residual fraction at most 0.20;
- `superadditive_full`: residual fraction less than -0.20, meaning the full
  mask is more negative than the sum of components;
- `subadditive_or_overlap`: residual fraction greater than 0.20, meaning
  component effects overlap or the sum exceeds the full effect magnitude.

V16 suite class:

- `distributed_additive`: both head partitions and layer slices are additive;
- `head_superadditive`: at least one head partition is superadditive while
  layer partitions are not;
- `layer_superadditive`: at least one layer partition is superadditive while
  head partitions are not;
- `mixed_superadditive`: both head and layer partitions show superadditivity;
- `subadditive_or_overlap`: no superadditive family appears and at least one
  family is subadditive/overlapping.

## Pass/Fail Boundary

V16 is explanatory, not a promotion gate. It cannot by itself support a compact
mechanism card. A mechanism-card localization claim still requires a future
intervention that matches the full-band effect while preserving V13/V14/V15
controls.
