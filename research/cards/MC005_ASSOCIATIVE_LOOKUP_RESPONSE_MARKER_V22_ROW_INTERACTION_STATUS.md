# MC005 Associative Lookup Response-Marker V22 Row Interaction Status

Status: row-level interaction diagnostic failed; parent/control/null replication
passed.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_RESPONSE_MARKER_V22_ROW_INTERACTION.md`
- runner:
  `code/mc005_associative_lookup_response_marker_v22_row_interaction.py`
- result:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v22_row_interaction_20260630T201924.json`
- result SHA256:
  `0b10c833d27a36666691973fe2dbde05de1a1ed74ef16220a1c43aa0299b627d`

## Verdict

V22 does not pass the preregistered row-level interaction gate. The parent
`slice_l24_26_all` target source-value effect replicated on fresh pair16 rows,
source controls passed, and answer-absent nulls stayed clean. But only 67 of
253 parent-effect rows were all-three margin rows, or 26.48 percent, below the
preregistered 40 percent fraction threshold.

This means V20/V21 remain evidence for a useful aggregate layers-24-26 control
surface, but the stronger claim that the interaction is broadly row-level
all-three structure is not supported by V22.

Summary:

```text
model: Qwen/Qwen3-1.7B
marker: Response
pair count: 16
lookup seeds: 191, 193
answer-absent null seeds: 197, 199
lookup rows: 256
parent candidate: slice_l24_26_all
passed: false
diagnostic class: mean_only_interaction
```

## Aggregate Lookup

| Path | Target mean delta | Target-win loss |
| --- | ---: | ---: |
| `slice_l24_26_all` | -5.9562 | 33 |
| `slice_l24_25_all` | -3.0098 | 11 |
| `slice_l24_26_all_pair` | -2.1232 | 4 |
| `slice_l25_26_all` | -2.4711 | 5 |
| `single_l24_all` | -0.4637 | 1 |
| `single_l25_all` | -0.9781 | 3 |
| `single_l26_all` | -0.9690 | 2 |

Parent source controls:

| Arm | Mean delta | Target-win loss |
| --- | ---: | ---: |
| target source value | -5.9562 | 33 |
| distractor source value | +0.8573 | -2 |
| random source value | +0.0040 | 0 |

The parent source controls passed: distractor and random source-value masks did
not match the target source-value effect.

## Row-Level Interaction

| Metric | Value |
| --- | ---: |
| baseline target wins | 253/256 |
| parent-effect rows | 253 |
| all-three margin rows | 67 |
| all-three fraction of parent-effect rows | 0.2648 |
| parent-flip rows | 33 |
| parent-flip pair-resistant rows | 20 |
| median single-layer-sum residual on parent-effect rows | -3.1875 |
| median best leave-one-layer-out pair share on parent-effect rows | 0.7143 |
| p90 best leave-one-layer-out pair share on parent-effect rows | 0.9033 |

V22 found real row-level all-three evidence in a minority of examples: 67 rows
met the all-three margin criterion, and 20 rows flipped under the parent while
all leave-one-layer-out pairs preserved the target win. But this support was
not broad enough for the preregistered row-level mechanism claim because the
all-three fraction was below 40 percent of parent-effect rows.

## Null Holdout

`slice_l24_26_all` preserved both pair16 answer-absent null seeds:

| Seed | Label | Source value | Control value | Earlier colon | Final label | Final colon |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 197 | clean_null | +0.0130 | +0.1445 | +0.0007 | -0.0313 | +0.0482 |
| 199 | clean_null | +0.0156 | +0.1406 | +0.0072 | -0.0130 | +0.0449 |

## Criteria

| Criterion | Result |
| --- | --- |
| lookup baseline valid | pass |
| parent effect replicated | pass |
| parent source controls passed | pass |
| at least 24 all-three margin rows | pass |
| at least 40 percent all-three rows among parent-effect rows | fail |
| at least 8 parent-flip pair-resistant rows | pass |
| median single-layer residual at most -1.0 | pass |
| parent null holdouts clean | pass |

## Interpretation

What V22 supports:

- the Qwen3-1.7B MC005 parent path remains a strong source-specific control
  surface on fresh pair16 rows;
- the V20/V21 aggregate interaction result is not explained by a parent failure,
  source-control failure, or fresh answer-absent null failure;
- a substantial minority of rows show all-three margin structure, and some
  behavior flips require the parent rather than any leave-one-layer-out pair.

What V22 blocks:

- do not claim the layers-24-26 interaction is broadly row-level all-three
  structure;
- do not treat `slice_l24_26_all` as a compact row-local circuit;
- do not promote a smaller path on the basis of V22, because the parent remains
  stronger in aggregate and source-specific.

## Next Step

The next MC005 pass should explain the row heterogeneity: why about one quarter
of parent-effect rows look all-three, while most parent-effect rows still allow
a leave-one-layer-out pair to recover at least 60 percent of the parent
negative-effect magnitude. A useful follow-up would stratify rows by query
index, source position, margin band, token identity, and best-pair winner before
trying another intervention.
