# MC005 Associative Lookup Response-Marker V12 0.6B Marker-Specificity Status

Status: no clean Qwen3-0.6B answer-absent marker found.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_RESPONSE_MARKER_V12_0P6B_MARKER_SPECIFICITY.md`
- runner:
  `code/mc005_associative_lookup_response_marker_v12_0p6b_marker_specificity.py`
- result:
  `results/cards/MC005/mc005_qwen3_0p6b_response_marker_v12_marker_specificity_20260630T183729.json`
- result SHA256:
  `0243591ef35c6ed7b0aaaad86a84613d4fcf7e676ea7227aba8daf317ee99400`

## Verdict

V12 does not find a clean Qwen3-0.6B answer-absent marker surface.

Summary:

```text
model: Qwen/Qwen3-0.6B
selected band: late_20_26
selected layers: 20, 21, 22, 23, 24, 25, 26
markers: response, output, answer, result
seeds: 17, 23, 31, 37, 41
rows per marker/seed: 128
full128 labels:
  clean_null: 4
  weak_null: 10
  side_effect: 6
first32 labels:
  clean_null: 15
  weak_null: 5
all-seed clean markers: none
diagnostic class: no_clean_marker
```

The V11 `Response:` parity check reproduced: seed 23 first-32 remained
`weak_null`, with two low-margin non-source-control flips. That confirms V12
is testing the same smaller-model boundary, not a changed row bank.

`Output:` improved the small first-32 slice: all five `Output:` first-32
scenarios were `clean_null`. But the 128-row diagnostic still failed:
`Output:` had only 2/5 clean full seeds, with weak nulls on seeds 17 and 23 and
a side effect on seed 41.

## Marker Table

| Marker | Full 128-row labels | First-32 labels | Clean full seeds | Main full-seed blockers |
| --- | --- | --- | ---: | --- |
| `Response:` | 1 clean, 2 weak, 2 side effect | 3 clean, 2 weak | 1/5 | final label, final colon, seed-41 control arms |
| `Output:` | 2 clean, 2 weak, 1 side effect | 5 clean | 2/5 | seed-17 earlier colon, seed-23 control/final colon, seed-41 non-source control |
| `Answer:` | 0 clean, 3 weak, 2 side effect | 4 clean, 1 weak | 0/5 | final/control/source arms |
| `Result:` | 1 clean, 3 weak, 1 side effect | 3 clean, 2 weak | 1/5 | source, final label, earlier colon, final colon, control arms |

## Interpretation

What V12 supports:

- the V11 `Response:` boundary is reproducible under the marker-comparison
  runner;
- `Output:` is locally better than `Response:` on the first-32 slice;
- the smaller-model null failure is not fixed by simply switching to `Output:`;
- no V8 marker candidate is clean across five 128-row Qwen3-0.6B seeds.

What V12 blocks:

- Qwen3-0.6B strict size replication under any tested marker;
- treating the 0.6B answer-absent issue as a `Response:`-only marker artifact;
- returning to a full Qwen3-0.6B size-replication atlas without changing the
  smaller-model null design or explicitly scoping out 0.6B.

## Next Step

MC005 should stop spending runs trying to rescue the current Qwen3-0.6B
answer-absent marker surface. The live positive claim should remain:

- Qwen3-1.7B has a passed compact `Response:` atlas;
- Qwen3-0.6B replicates the lookup intervention but fails strict answer-absent
  null reliability.

The next useful MC005 work is either:

- longer-context and finer-path work on the Qwen3-1.7B positive surface; or
- a new smaller-model null design with different answer-absent wording,
  preregistered as a new repair attempt rather than as marker selection.
