# MC005 Associative Lookup Final-Marker V8 Status

Status: marker-specific clean nulls found; all-marker suite failed.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_FINAL_MARKER_V8.md`
- runner:
  `code/mc005_associative_lookup_final_marker_v8.py`
- result:
  `results/cards/MC005/mc005_qwen3_1p7b_associative_lookup_final_marker_v8_20260630T175821.json`
- result SHA256:
  `1b1dfc6ce3010dfcf2ec60fddfdb626a52adaadfd225d71647066f8aed32cf99`

## Verdict

V8 resolves the V7 final-position question in a useful but non-promoting way:
the boundary is marker-specific.

Summary:

```text
scenario labels:
  clean_null: 9
  final_label_boundary: 2
  final_punctuation_boundary: 1

arm labels:
  clean: 57
  weak: 3

structural non-clean arms: 0
final-label non-clean arms: 2
final-punctuation non-clean arms: 1
passed: false
```

`Response:` and `Output:` were clean on all three seeds. `Answer:` was clean on
two seeds but had a final-colon weak arm on seed 31. `Result:` was clean on one
seed but had final-label weak arms on seeds 17 and 23.

All structural controls stayed clean: source value, non-source control value,
and earlier neutral colon masks had no non-clean arms. That strengthens the
conclusion that the remaining problem is final-marker surface choice, not broad
source-line or punctuation masking.

## Marker Summary

| Marker | Clean seeds | Non-clean seeds | Boundary |
| --- | ---: | ---: | --- |
| `Answer:` | 2/3 | 1/3 | seed 31 final colon weak, delta -0.553 |
| `Response:` | 3/3 | 0/3 | clean marker |
| `Output:` | 3/3 | 0/3 | clean marker |
| `Result:` | 1/3 | 2/3 | seed 17/23 final label weak, deltas -0.547 and -0.576 |

## Scenario Table

| Seed | Marker | Label | Clean rows | Source value | Control value | Earlier colon | Final label | Final colon |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | `Answer:` | clean_null | 32/32 | -0.010 | +0.088 | -0.035 | +0.098 | -0.486 |
| 17 | `Response:` | clean_null | 32/32 | -0.039 | +0.096 | +0.029 | -0.316 | -0.004 |
| 17 | `Output:` | clean_null | 32/32 | -0.019 | +0.258 | -0.015 | +0.121 | -0.125 |
| 17 | `Result:` | final_label_boundary | 32/32 | -0.037 | +0.096 | +0.014 | -0.547 | +0.021 |
| 23 | `Answer:` | clean_null | 32/32 | -0.002 | +0.189 | -0.016 | +0.248 | -0.256 |
| 23 | `Response:` | clean_null | 32/32 | -0.008 | +0.162 | +0.004 | -0.105 | -0.031 |
| 23 | `Output:` | clean_null | 32/32 | -0.011 | +0.238 | +0.008 | +0.246 | -0.054 |
| 23 | `Result:` | final_label_boundary | 32/32 | +0.004 | +0.182 | -0.006 | -0.576 | +0.133 |
| 31 | `Answer:` | final_punctuation_boundary | 32/32 | +0.014 | -0.023 | +0.031 | +0.363 | -0.553 |
| 31 | `Response:` | clean_null | 32/32 | -0.014 | -0.084 | -0.025 | -0.281 | -0.231 |
| 31 | `Output:` | clean_null | 31/32 | -0.008 | +0.170 | -0.004 | +0.232 | -0.073 |
| 31 | `Result:` | clean_null | 32/32 | -0.012 | +0.006 | -0.006 | -0.311 | +0.006 |

## Interpretation

What V8 supports:

- `Response:` and `Output:` are clean final-marker surfaces for the current
  answer-absent same-grammar null;
- the earlier neutral colon control is clean, so colon masking is not generally
  destructive;
- source/control arms remain clean across another 12 scenarios.

What V8 blocks:

- not all answer markers are equivalent;
- `Answer:` should not be the default marker for promotion because it has a
  seed-31 final-colon weak arm;
- `Result:` should not be the default marker because its final label is weak on
  two seeds;
- MC005 still needs reliability work before model-family replication.

## Next Step

The next MC005 pass should use `Response:` or `Output:` as the clean final
marker and rerun a compact reliability atlas:

- include source-line nulls, off-target nulls, and lookup scenarios under the
  selected clean marker;
- preserve `Answer:` and `Result:` as known marker-specific negative controls;
- only after that consider model-family replication.
