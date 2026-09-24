# MC005 Associative Lookup Response-Marker V20 L24-26 Interaction Status

Status: layers 24-26 interaction diagnostic passed.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_RESPONSE_MARKER_V20_L24_26_INTERACTION.md`
- runner:
  `code/mc005_associative_lookup_response_marker_v20_l24_26_interaction.py`
- result:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v20_l24_26_interaction_20260630T195500.json`
- result SHA256:
  `bc42173d1ac55c829df29d755baa3be4ca9640301ca1c8934ebfa1b7444c1860`

## Verdict

V20 passes the preregistered layers-24-26 interaction diagnostic. The parent
`slice_l24_26_all` effect replicated on fresh lookup seeds, every
leave-one-layer-out pair stayed below 60 percent of the parent effect, the
single-layer sum was strongly superadditive relative to the parent, and parent
answer-absent nulls stayed clean.

Summary:

```text
model: Qwen/Qwen3-1.7B
marker: Response
pair count: 16
lookup seeds: 113, 127
answer-absent null seeds: 131, 137
lookup rows: 256
parent candidate: slice_l24_26_all
passed: true
diagnostic class: l24_26_three_layer_interaction_supported
```

## Lookup Diagnostic

| Path | Target mean delta | Target-win loss | Parent share |
| --- | ---: | ---: | ---: |
| `full_l20_26_all` | -8.8129 | 83 | n/a |
| `slice_l23_26_all` | -6.4435 | 41 | n/a |
| `slice_l24_26_all` | -5.9028 | 32 | 1.0000 |
| `slice_l24_25_all` | -3.0656 | 7 | 0.5193 |
| `slice_l24_26_all_pair` | -2.2140 | 11 | 0.3751 |
| `slice_l25_26_all` | -2.3329 | 8 | 0.3952 |
| `single_l24_all` | -0.5890 | 0 | 0.0998 |
| `single_l25_all` | -0.9200 | 1 | 0.1559 |
| `single_l26_all` | -0.9379 | 5 | 0.1589 |

Every leave-one-layer-out pair was below the preregistered 60 percent
parent-effect threshold.

Source controls for the parent:

| Path | Arm | Mean delta | Target-win loss |
| --- | --- | ---: | ---: |
| `slice_l24_26_all` | target source value | -5.9028 | 32 |
| `slice_l24_26_all` | distractor source value | +1.0309 | -4 |
| `slice_l24_26_all` | random source value | -0.0060 | 0 |

The distractor and random source controls did not match the target source-value
effect.

## Additivity

| Family | Component sum | Residual | Residual fraction | Label |
| --- | ---: | ---: | ---: | --- |
| `single_l24 + single_l25 + single_l26` | -2.4469 | -3.4559 | -0.5855 | superadditive_parent |
| `slice_l24_25 + single_l26` | -4.0034 | -1.8994 | -0.3218 | superadditive_parent |
| `slice_l24_26_pair + single_l25` | -3.1340 | -2.7688 | -0.4691 | superadditive_parent |
| `slice_l25_26 + single_l24` | -2.9219 | -2.9810 | -0.5050 | superadditive_parent |

The parent effect is not explained by independent single-layer contributions.
It also remains superadditive relative to every pair-plus-omitted-layer
decomposition.

## Null Holdout

`slice_l24_26_all` preserved the pair16 answer-absent null:

| Seed | Label | Source value | Control value | Earlier colon | Final label | Final colon |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 131 | clean_null | -0.0033 | +0.0866 | -0.0059 | +0.0000 | +0.0124 |
| 137 | clean_null | +0.0156 | +0.0983 | +0.0026 | +0.0033 | +0.0326 |

## Criteria

All preregistered criteria passed:

- parent effect present;
- parent source controls passed;
- all leave-one-layer-out pairs recovered less than 60 percent of parent
  effect;
- single-layer sum had a superadditive parent residual;
- parent null holdouts were clean.

## Interpretation

What V20 supports:

- the current MC005 Qwen3-1.7B surface should be treated as a layers-24-26
  interaction block;
- no tested leave-one-layer-out pair recovers enough of the parent effect to
  replace the all-head layers-24-26 surface;
- layer 26 is necessary in the specific sense that the layers-24-25 path stays
  below the 60 percent parent-effect threshold;
- the parent block preserves source specificity and answer-absent null
  reliability on fresh seeds.

What V20 does not support:

- it does not localize the mechanism to a single layer, head partition, or head;
- it does not prove model-family generality;
- it does not repair the Qwen3-0.6B strict answer-absent boundary;
- it does not prove reliability outside the synthetic associative lookup
  contract.

## Next Step

The next MC005 pass should keep layers 24-26 as the current interaction block
and test whether the interaction result survives harder reliability axes:

- layout and lexicon holdouts at pair count 16;
- pair-count stress beyond 16 or mixed pair-count banks;
- row-level interaction diagnostics for which examples require all three
  layers;
- continued answer-absent null and source-control checks.
