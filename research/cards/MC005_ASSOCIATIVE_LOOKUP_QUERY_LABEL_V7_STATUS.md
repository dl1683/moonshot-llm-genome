# MC005 Associative Lookup Query-Label V7 Status

Status: source/control arms stable; final-position and prompt-surface baselines
still block promotion.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_QUERY_LABEL_V7.md`
- runner:
  `code/mc005_associative_lookup_query_label_v7.py`
- result:
  `results/cards/MC005/mc005_qwen3_1p7b_associative_lookup_query_label_v7_20260630T175032.json`
- result SHA256:
  `65dfc8c156ca62877ab1b65dcf54ecc969e47022db51c686aa5d2b95b97e45e9`

## Verdict

V7 confirms that the remaining boundary is not source-line masking. It also
shows the query-label/position surface is not stable enough for promotion.

Summary:

```text
scenario labels:
  invalid_baseline: 6
  query_label_position_boundary: 3
  clean_null: 2
  label_semantic_boundary: 1

arm labels:
  clean: 49
  weak: 7
  side_effect: 1

source/control non-clean arms: 0
final-position non-clean arms: 7
label-semantic non-clean arms: 1
passed: false
```

All source/control arms were clean across the full sweep. The source value,
source key, source colon, and non-source control-value masks did not reproduce
the V4/V5 side effect.

The final-position controls were not clean. The strongest valid-baseline
finding is `generic_answer_label`: all three seeds had 32/32 baseline clean
rows, the `Answer` token mask was clean, but the final colon mask was weak with
mean deltas from -0.584 to -0.641.

The random-label surfaces also exposed a substrate problem: six of nine random
label scenarios were `invalid_baseline`, mostly because arbitrary final labels
do not reliably cue the required answer.

## Scenario Table

| Seed | Surface | Label | Clean rows | Final label | Final colon | Source value | Control value |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 17 | `original_random_label` | invalid_baseline | 17/32 | +0.178 | +0.076 | -0.004 | +0.033 |
| 17 | `repeated_random_label` | invalid_baseline | 20/32 | +0.012 | +0.135 | -0.043 | -0.090 |
| 17 | `generic_answer_label` | query_label_position_boundary | 32/32 | +0.053 | -0.641 | -0.002 | +0.180 |
| 17 | `irrelevant_random_label` | invalid_baseline | 17/32 | +0.137 | +0.074 | +0.029 | +0.062 |
| 23 | `original_random_label` | clean_null | 25/32 | +0.164 | +0.064 | -0.027 | +0.170 |
| 23 | `repeated_random_label` | label_semantic_boundary | 27/32 | -0.014 | +0.002 | -0.088 | +0.082 |
| 23 | `generic_answer_label` | query_label_position_boundary | 32/32 | +0.014 | -0.584 | -0.027 | +0.197 |
| 23 | `irrelevant_random_label` | invalid_baseline | 23/32 | +0.174 | +0.027 | -0.051 | +0.061 |
| 31 | `original_random_label` | invalid_baseline | 21/32 | +0.238 | +0.053 | +0.033 | +0.016 |
| 31 | `repeated_random_label` | clean_null | 24/32 | +0.160 | +0.080 | +0.006 | -0.037 |
| 31 | `generic_answer_label` | query_label_position_boundary | 32/32 | -0.026 | -0.627 | +0.035 | -0.090 |
| 31 | `irrelevant_random_label` | invalid_baseline | 18/32 | +0.164 | +0.055 | +0.064 | -0.021 |

## Interpretation

What V7 supports:

- the late-band source-line mask is stable under this 3-seed sweep;
- the source-value side-effect interpretation from V4/V5 is no longer the best
  explanation;
- the generic `Answer:` surface provides a clean baseline substrate for further
  query-position testing.

What V7 blocks:

- arbitrary random final labels are not reliable answer surfaces;
- final-position punctuation remains a weak intervention boundary on the
  generic `Answer:` surface;
- one repeated-label scenario had an earlier-label weak arm;
- MC005 still has not passed a complete reliability atlas.

## Next Step

The next MC005 pass should narrow final-position punctuation sensitivity:

- keep the `generic_answer_label` surface because it has 32/32 clean baselines
  on all V7 seeds;
- compare final colon, final label, final newline-adjacent token, and earlier
  matched colon masks;
- test alternate final answer markers such as `Response:`, `Output:`, and
  `Result:`;
- retain source/control arms as regression checks.
