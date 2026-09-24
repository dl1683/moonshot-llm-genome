# MC005 Associative Lookup Response-Marker V31 Margin Boundary Status

Status: tight low-margin boundary failed; lookup/null margin separation mapped.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_RESPONSE_MARKER_V31_MARGIN_BOUNDARY.md`
- runner:
  `code/mc005_associative_lookup_response_marker_v31_margin_boundary.py`
- V25 source artifact:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v25_factorial_row_heterogeneity_20260630T205454.json`
- V25 source SHA256:
  `efdab6fad30de1762b8e92f7e46c7f4e1e160bcc988ad093c7d0bf9a5617b9cf`
- V29 source artifact:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v29_attention_write_replacement_20260630T214645.json`
- V29 source SHA256:
  `c90bbd9a65fdd3866ef4e19e787d7c93b7faa693a89ddd0f843e58829097804b`
- V30 source artifact:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v30_write_null_sweep_20260630T215937.json`
- V30 source SHA256:
  `fab8f176ff5fd8a618529315bc547070555073ea9ab0c55e6db4c18683c13543`
- result:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v31_margin_boundary_20260630T220916.json`
- result SHA256:
  `bd5e0e04bc6abd760256a7100e7c618b3fe8fe4fdfc4405f7808347f193947ff`

## Verdict

V31 failed the preregistered tight margin-boundary criterion.

The lookup target write effect reproduced exactly:

- baseline: 128/128 target wins;
- target write replacement: mean delta -6.1978;
- target-win loss: 11;
- all 11 target-win-loss rows had baseline margin greater than 2.0;
- the minimum baseline margin among lookup loss rows was 5.25.

The new fresh null panel found one additional strict row flip:

- seed 313;
- `non_source_control_value`;
- baseline margin 0.0000;
- arm margin +0.8750;
- one target-win gain.

All V31 fresh null arms stayed within the absolute mean-delta tolerance of
0.25. The new fresh flip supports the low-margin part of the hypothesis.

The preregistered diagnostic still failed because V31 also imported V30's
changed rows, and the original V25 replay row from seed 251 had absolute
baseline margin 0.75. That exceeds the preregistered 0.5 threshold.

The diagnostic class is:

```text
null_boundary_broad
```

This means broader than the strict 0.5 baseline-margin explanation. It does not
mean broad aggregate margin drift: every fresh null arm remained mean-delta
clean, and all combined null flips had absolute baseline margin at most 0.75.

## Margin Separation

| Row set | Count | `<=0.25` | `0.25-0.5` | `0.5-1` | `1-2` | `>2` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| combined null flips | 5 | 4 | 0 | 1 | 0 | 0 |
| lookup target-win losses | 11 | 0 | 0 | 0 | 0 | 11 |

The lookup effect and null failures are margin-separated in practice, but not
under the preregistered tight 0.5 null-boundary threshold.

## Null Flip Rows

| Source | Seed | Row | Arm | Baseline margin | Arm margin | Delta | Target | Distractor |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- |
| V30 replay | 251 | `mc005_v25_seed251_answer_absent_pair16_response_null_041` | `non_source_control_value` | -0.7500 | +0.6250 | +1.3750 | `temple` | `velvet` |
| V30 fresh | 263 | `mc005_v25_seed263_answer_absent_pair16_response_null_121` | `non_source_control_value` | -0.2500 | +0.2500 | +0.5000 | `magnet` | `basket` |
| V30 fresh | 283 | `mc005_v25_seed283_answer_absent_pair16_response_null_080` | `source_value` | +0.1875 | +0.0000 | -0.1875 | `sailor` | `anchor` |
| V30 fresh | 283 | `mc005_v25_seed283_answer_absent_pair16_response_null_080` | `non_source_control_value` | +0.1875 | +0.0000 | -0.1875 | `sailor` | `anchor` |
| V31 fresh | 313 | `mc005_v25_seed313_answer_absent_pair16_response_null_007` | `non_source_control_value` | +0.0000 | +0.8750 | +0.8750 | `temple` | `velvet` |

All null flip rows had greedy label `other` before and after replacement.

## Criteria

| Criterion | Result |
| --- | --- |
| source artifacts valid | pass |
| V30 imported flips abs margin <= 0.5 | fail |
| V31 fresh null flips abs margin <= 0.5 | pass |
| V31 fresh null mean deltas within 0.25 | pass |
| V31 lookup target effect reproduced | pass |
| lookup loss rows baseline margin >= 2.0 | pass |

## Interpretation

V31 strengthens the map while failing the preregistered explanation.

What it supports:

- the V29 lookup write effect is stable and large;
- lookup target-win losses are not near-threshold artifacts;
- fresh answer-absent null flips continue to be rare and close to the
  target/distractor threshold;
- null failures remain aggregate-mean clean.

What it blocks:

- no claim that the answer-absent write-replacement null problem is fully
  explained by an absolute baseline-margin cutoff of 0.5;
- no mechanism-card promotion for the V29/V30/V31 write surface;
- no post-hoc threshold relaxation from 0.5 to 1.0 as a success claim.

The current best description is:

> Layers-24-26 final-query attention-write replacement is a real lookup
> mediation surface with high-margin target effects, but it has a rare
> answer-absent locality boundary on low-to-moderate margin null rows.

## Next Step

Further MC005 work should not attempt to promote this exact write-replacement
surface without changing the intervention. A useful refinement would need to
preserve the high-margin lookup effect while suppressing low-to-moderate-margin
answer-absent row flips under a preregistered rule. Otherwise, this is a mapped
failure mode and the project should move to a new behavior family for the first
full mechanism card.
