# MC005 Associative Lookup Response-Marker V24 Source Position Causal Status

Status: source-position causal diagnostic failed; directional position effect
observed.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_RESPONSE_MARKER_V24_SOURCE_POSITION_CAUSAL.md`
- runner:
  `code/mc005_associative_lookup_response_marker_v24_source_position_causal.py`
- result:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v24_source_position_causal_20260630T203903.json`
- result SHA256:
  `78925ef6909a0b3c2008b7740553b429e4c107b2c1ae4d5fc5aba52ad49a2876`

## Verdict

V24 does not pass the preregistered source-position causality gate. Moving the
same target key/value pair from early slots to mid/late slots changed the row
statistics in the predicted direction, but the movement was not large enough:

- all-three fraction rose from 0.1575 in early rows to 0.3175 in mid/late rows,
  a +0.1600 shift below the +0.20 threshold;
- paired family net gain was +7 mid/late-only all-three families, below the
  required +12;
- median best-pair share shifted downward from 0.7381 to 0.6769, passing that
  directional criterion.

Parent effect, source controls, and answer-absent nulls all passed. The failure
is therefore not a parent/control/null failure; it is a failure of source
position to explain enough of the V23 row split under fixed key/value identity.

Summary:

```text
model: Qwen/Qwen3-1.7B
marker: Response
pair count: 16
lookup seeds: 211, 223
base families per seed: 32
lookup rows: 256
null seeds: 227, 229
passed: false
diagnostic class: position_fraction_not_causal
```

## Position Groups

| Group | Baseline target wins | Target delta | Target-win loss | Distractor delta | Random delta | All-three rows | All-three fraction | Median best-pair share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| early | 127/128 | -5.6353 | 26 | +1.0122 | +0.0088 | 20/127 | 0.1575 | 0.7381 |
| mid/late | 126/128 | -6.3171 | 14 | +0.9832 | -0.0083 | 40/126 | 0.3175 | 0.6769 |

The parent source-value effect remained strong in both groups, and distractor
and random source controls did not match it.

## Position Slots

| Variant | Target slot | Median source token position | All-three rows | All-three fraction | Median best-pair share | Median single-layer residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `early_slot0` | 0 | 6 | 13/64 | 0.2031 | 0.6962 | -3.3125 |
| `early_slot1` | 1 | 11 | 7/63 | 0.1111 | 0.7778 | -1.9375 |
| `mid_slot10` | 10 | 56 | 20/63 | 0.3175 | 0.6752 | -4.0000 |
| `late_slot13` | 13 | 71 | 20/63 | 0.3175 | 0.6786 | -4.0000 |

The V23 pattern partially reproduced under controlled identity: the mid/late
slots were more all-three-like than the early slots. However, the strongest
early slot, slot 0, already had 13/64 all-three rows, so moving position alone
did not create a large enough causal contrast.

## Paired Families

| Paired outcome | Count |
| --- | ---: |
| mid/late-only all-three | 11 |
| early-only all-three | 4 |
| both early and mid/late all-three | 12 |
| neither early nor mid/late all-three | 37 |
| net mid/late minus early | 7 |

Median best-pair-share change from early to mid/late was -0.0468. This is
directionally consistent with source position reducing pair recovery, but the
preregistered group-level criterion used a -0.05 threshold and passed only
because the group medians moved from 0.7381 to 0.6769.

## Null Holdout

`slice_l24_26_all` preserved both pair16 answer-absent null seeds:

| Seed | Label | Source value | Control value | Earlier colon | Final label | Final colon |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 227 | clean_null | +0.0059 | +0.1777 | +0.0091 | +0.0098 | +0.0143 |
| 229 | clean_null | -0.0007 | +0.0850 | +0.0000 | -0.0544 | +0.0260 |

## Criteria

| Criterion | Result |
| --- | --- |
| all position-group baselines valid | pass |
| all position-group parent effects pass | pass |
| all position-group source controls pass | pass |
| mid/late all-three fraction beats early by at least 0.20 | fail |
| paired mid/late net gain at least 12 families | fail |
| mid/late best-pair share lower by at least 0.05 | pass |
| parent null holdouts clean | pass |

## Interpretation

What V24 supports:

- source position contributes directionally to row-level all-three structure;
- V23's source-position contrast was not pure noise;
- the parent path remains source-specific and null-clean under controlled
  position movement.

What V24 blocks:

- do not claim source position is the causal explanation for the V22 row split;
- do not treat mid/late positioning alone as enough to promote a broad
  row-level all-three mechanism;
- do not promote a smaller path from this result.

## Next Step

The next MC005 pass should test a stronger factorial account rather than a
single-factor position account. The obvious factors are source position,
baseline margin, and distractor position/value identity, because V24 preserved
the directional source-position effect but showed that position alone is too
weak to explain V22's row split.

Update: V25 ran this factorial test and failed the simple prompt-layout
explanation while preserving parent/control/null validity. See
`research/cards/MC005_ASSOCIATIVE_LOOKUP_RESPONSE_MARKER_V25_FACTORIAL_ROW_HETEROGENEITY_STATUS.md`.
