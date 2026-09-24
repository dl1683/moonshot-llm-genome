# MC005 Associative Lookup Response-Marker V30 Write Null Sweep Status

Status: fresh answer-absent write-replacement null boundary reproduced.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_RESPONSE_MARKER_V30_WRITE_NULL_SWEEP.md`
- runner:
  `code/mc005_associative_lookup_response_marker_v30_write_null_sweep.py`
- V25 source artifact:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v25_factorial_row_heterogeneity_20260630T205454.json`
- V25 source SHA256:
  `efdab6fad30de1762b8e92f7e46c7f4e1e160bcc988ad093c7d0bf9a5617b9cf`
- V29 source artifact:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v29_attention_write_replacement_20260630T214645.json`
- V29 source SHA256:
  `c90bbd9a65fdd3866ef4e19e787d7c93b7faa693a89ddd0f843e58829097804b`
- result:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v30_write_null_sweep_20260630T215937.json`
- result SHA256:
  `fab8f176ff5fd8a618529315bc547070555073ea9ab0c55e6db4c18683c13543`

## Verdict

V30 confirms that the V29 answer-absent write-replacement null failure is not
only an original-sample artifact.

The replay panel reproduced V29 exactly enough for the preregistered sentinel:
seed 251 `non_source_control_value` had `target_win_loss = -1` with mean delta
`+0.1406`; seed 257 was clean.

The fresh 8-seed, 1,024-row answer-absent sweep then found 3 strict row-change
failures:

- seed 263, `non_source_control_value`: one target-win gain;
- seed 283, `source_value`: one target-win loss;
- seed 283, `non_source_control_value`: one target-win loss.

Every fresh arm stayed inside the absolute mean-delta tolerance of 0.25, so the
failure is not broad mean-margin drift. The failure is the exact row-change
criterion: final-query write replacement can flip low-margin answer-absent
rows even when aggregate mean deltas remain small.

The diagnostic class is:

```text
fresh_write_null_failed
```

## Panel Summary

| Panel | Seeds | Rows | Clean seeds | Changed rows | Diagnostic |
| --- | ---: | ---: | ---: | ---: | --- |
| `v25_replay_96` | 2 | 192 | 1 | 1 | V29 failure reproduced |
| `fresh_128` | 8 | 1024 | 6 | 3 | fresh strict null failure |

## Failing Arms

| Panel | Seed | Arm | Target-win loss | Mean delta | Row effect |
| --- | ---: | --- | ---: | ---: | --- |
| `v25_replay_96` | 251 | `non_source_control_value` | -1 | +0.1406 | one target-win gain |
| `fresh_128` | 263 | `non_source_control_value` | -1 | +0.1255 | one target-win gain |
| `fresh_128` | 283 | `source_value` | 1 | +0.0078 | one target-win loss |
| `fresh_128` | 283 | `non_source_control_value` | 1 | +0.1333 | one target-win loss |

## Row Audit

| Panel | Seed | Row | Arm | Baseline margin | Arm margin | Delta | Target | Distractor | Focus/control |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `v25_replay_96` | 251 | `mc005_v25_seed251_answer_absent_pair16_response_null_041` | `non_source_control_value` | -0.7500 | +0.6250 | +1.3750 | `temple` | `velvet` | `puzzle` / `violet` |
| `fresh_128` | 263 | `mc005_v25_seed263_answer_absent_pair16_response_null_121` | `non_source_control_value` | -0.2500 | +0.2500 | +0.5000 | `magnet` | `basket` | `rocket` / `river` |
| `fresh_128` | 283 | `mc005_v25_seed283_answer_absent_pair16_response_null_080` | `source_value` | +0.1875 | +0.0000 | -0.1875 | `sailor` | `anchor` | `hammer` / `window` |
| `fresh_128` | 283 | `mc005_v25_seed283_answer_absent_pair16_response_null_080` | `non_source_control_value` | +0.1875 | +0.0000 | -0.1875 | `sailor` | `anchor` | `hammer` / `window` |

All changed rows had greedy label `other` both before and after replacement.
The row flips are target-versus-distractor margin threshold crossings, not
greedy answer switches.

## Criteria

| Criterion | Result |
| --- | --- |
| source artifact valid | pass |
| V29 artifact valid | pass |
| V29 seed-251 failure reproduced | pass |
| fresh seeds clean | fail |
| fresh no large mean delta | pass |
| fresh no target-win change | fail |

## Interpretation

V29 remains a real bounded mediation result for lookup rows: layers-24-26
final-query attention-write replacement exactly recovers the direct target
source-mask effect on the tested holdout. V30 does not erase that result.

V30 does block mechanism-card promotion. The same write-replacement operation
has a measurable answer-absent locality boundary: it can change the
target-versus-distractor margin sign on low-margin null rows, including fresh
rows outside the V29 source sample.

The current claim should therefore stay:

> MC005 has exact lookup mediation by final-query attention writes in layers
> 24-26 on the tested holdout, but the write surface is not reliable under
> strict answer-absent null locality.

## Next Step

Do not treat the V29 null failure as sample-fragile. The next useful MC005 work
is either:

1. refine the write intervention to preserve answer-absent low-margin null rows
   while retaining lookup mediation; or
2. characterize the low-margin null-row boundary as an explicit failure mode
   and move to a different behavior family for the first full mechanism card.

Follow-up completed:

- V31 tested a preregistered absolute baseline-margin cutoff of 0.5. The new
  fresh null flip was at baseline margin 0.0 and lookup target losses were all
  high-margin, but the imported V30 replay flip had absolute margin 0.75. The
  tight 0.5 explanation failed. See
  `research/cards/MC005_ASSOCIATIVE_LOOKUP_RESPONSE_MARKER_V31_MARGIN_BOUNDARY_STATUS.md`.
