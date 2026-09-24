# MC005 Associative Lookup Response-Marker V11 Weak-Null Diagnostic Status

Status: persistent Qwen3-0.6B answer-absent null boundary.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_RESPONSE_MARKER_V11_WEAK_NULL_DIAGNOSTIC.md`
- runner:
  `code/mc005_associative_lookup_response_marker_v11_weak_null_diagnostic.py`
- result:
  `results/cards/MC005/mc005_qwen3_0p6b_response_marker_v11_weak_null_20260630T182625.json`
- result SHA256:
  `196243ad12781b3f5057d92cafe966c01965053a12f4bd38cef0cda18bea3c92`

## Verdict

V11 diagnoses the V10 Qwen3-0.6B weak null as a persistent boundary, not a
small-row-count repair.

Summary:

```text
model: Qwen/Qwen3-0.6B
selected band: late_20_26
selected layers: 20, 21, 22, 23, 24, 25, 26
marker: Response
seeds: 17, 23, 31, 37, 41
rows per seed: 128
first32 labels:
  clean_null: 3
  weak_null: 2
full128 labels:
  clean_null: 1
  weak_null: 2
  side_effect: 2
diagnostic class: persistent_boundary
```

The V10 seed-23 first-32 pattern reproduced. The non-source-control-value arm
again produced a weak-null target-win increase, with two low-margin
distractor-to-target flips:

| Row | Baseline margin | Arm margin | Delta | Target | Distractor | Control |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `seed23_answer_absent_response_null_002` | +0.000 | +0.625 | +0.625 | winter | engine | flame |
| `seed23_answer_absent_response_null_019` | -1.000 | +0.688 | +1.688 | sailor | island | puzzle |

That first-32 failure is low-margin. But the larger diagnostic did not become
clean: only seed 17 was `clean_null` at 128 rows. Seeds 23 and 37 were
`weak_null`; seeds 31 and 41 were `side_effect`.

## Full 128-Row Table

| Seed | Label | Clean rows | Main non-clean arms |
| ---: | --- | ---: | --- |
| 17 | clean_null | 124/128 | none |
| 23 | weak_null | 124/128 | final label target-win loss 2 |
| 31 | side_effect | 121/128 | final label target-win loss 3 |
| 37 | weak_null | 125/128 | final colon target-win loss 2 |
| 41 | side_effect | 123/128 | final colon target-win change 4; final label target-win loss 2; non-source control and earlier colon target-win changes 2 |

The dominant larger-bank issue is no longer the exact V10
non-source-control-value arm. The broader issue is that Qwen3-0.6B has
seed-sensitive final-label/final-colon target-win changes under the
answer-absent `Response:` null.

## Interpretation

What V11 supports:

- the exact V10 seed-23 first-32 weak-null pattern reproduced;
- the two reproduced seed-23 non-source-control flips were low-margin rows;
- the V10 strict failure was not a logging or runner artifact.

What V11 blocks:

- the Qwen3-0.6B answer-absent null cannot be treated as repaired by adding
  rows;
- the Qwen3-0.6B `Response:` surface is not clean enough for strict
  size-replication reliability;
- MC005 still cannot claim cross-size strict reliability.

## Next Step

The next MC005 diagnostic should stop treating the Qwen3-0.6B issue as only a
non-source-control-value problem. The better question is marker/control
specificity on the smaller model:

- compare `Response:` against `Output:` for Qwen3-0.6B answer-absent nulls;
- keep final-label and final-colon arms as primary controls;
- preserve the first-32 parity check against V10/V11;
- only return to a full size-replication atlas if a smaller-model marker
  surface is clean.
