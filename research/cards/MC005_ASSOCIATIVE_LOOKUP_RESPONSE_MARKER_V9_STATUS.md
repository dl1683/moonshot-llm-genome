# MC005 Associative Lookup Response-Marker V9 Status

Status: compact `Response:` atlas passed.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_RESPONSE_MARKER_V9.md`
- runner:
  `code/mc005_associative_lookup_response_marker_v9.py`
- result:
  `results/cards/MC005/mc005_qwen3_1p7b_associative_lookup_response_marker_v9_20260630T180904.json`
- result SHA256:
  `00765451567217004133ceaf7a26e9281922d9618d12effdab75c3df535fc428`

## Verdict

V9 passes the compact `Response:` marker atlas for the fixed
`late_20_26` source-value mask on `Qwen/Qwen3-1.7B`.

Summary:

```text
selected band: late_20_26
selected layers: 20, 21, 22, 23, 24, 25, 26
marker: Response
seeds: 17, 23, 31
scenario count: 12
lookup scenarios: 6
null scenarios: 6
label counts:
  works: 6
  clean_null: 6
passed: true
```

Both lookup scenarios worked on all three seeds. The target source-value mask
produced large negative target-vs-distractor margin deltas, while distractor
and random source-value masks did not match that effect.

Both null scenarios were clean on all three seeds. The off-target same-grammar
lookup null stayed clean for irrelevant value, irrelevant key, and random value
masks. The answer-absent null stayed clean for source value, non-source control
value, earlier neutral colon, final label, and final colon masks.

## Scenario Table

| Seed | Scenario | Label | Clean rows | Key arm means |
| ---: | --- | --- | ---: | --- |
| 17 | `lookup_pair5_response` | works | 32/32 | target -10.066, distractor +2.215, random -0.090 |
| 17 | `lookup_pair8_response` | works | 32/32 | target -9.746, distractor +1.459, random +0.018 |
| 17 | `offtarget_pair5_response` | clean_null | 32/32 | irrelevant value -0.031, irrelevant key +0.021, random -0.064 |
| 17 | `answer_absent_response_null` | clean_null | 31/32 | source +0.000, control +0.098, earlier colon -0.014, final label -0.404, final colon +0.000 |
| 23 | `lookup_pair5_response` | works | 32/32 | target -10.746, distractor +2.541, random -0.031 |
| 23 | `lookup_pair8_response` | works | 32/32 | target -9.729, distractor +1.785, random +0.027 |
| 23 | `offtarget_pair5_response` | clean_null | 32/32 | irrelevant value -0.062, irrelevant key +0.006, random -0.006 |
| 23 | `answer_absent_response_null` | clean_null | 32/32 | source +0.025, control +0.141, earlier colon +0.008, final label -0.258, final colon +0.078 |
| 31 | `lookup_pair5_response` | works | 32/32 | target -10.695, distractor +2.171, random -0.008 |
| 31 | `lookup_pair8_response` | works | 32/32 | target -10.252, distractor +1.564, random +0.006 |
| 31 | `offtarget_pair5_response` | clean_null | 32/32 | irrelevant value -0.094, irrelevant key -0.076, random -0.096 |
| 31 | `answer_absent_response_null` | clean_null | 32/32 | source -0.025, control +0.057, earlier colon -0.008, final label -0.217, final colon -0.027 |

## Interpretation

What V9 supports:

- the fixed `Response:` marker removes the V7/V8 final-position reliability
  blocker for this compact atlas;
- `late_20_26` target source-value masking remains a large, directionally
  specific intervention under pair-count 5 and pair-count 8 lookup prompts;
- off-target same-grammar source-line masks are clean under the selected
  marker;
- answer-absent source/control/final-marker masks are clean under the selected
  marker;
- key/value disjoint row sampling avoids a local lexical shortcut where a
  value-side intervention is also a key-token intervention.

What V9 does not support by itself:

- model-family or model-size generalization;
- longer-context associative lookup beyond pair count 8;
- a single-head or single-layer circuit inside `late_20_26`;
- deployable control outside synthetic associative lookup.

## Next Step

V9 upgrades MC005 from "marker-specific reliability boundary open" to "compact
`Response:` reliability atlas passed" for Qwen3-1.7B. The next promotion gate
should not retest `Answer:` as a positive surface; it should preserve
`Answer:` and `Result:` as negative controls and move to:

- model-size or model-family replication under `Response:`;
- longer pair-count/context holdouts;
- finer path/head decomposition inside layers 20-26.
