# MC005 Associative Lookup Response-Marker V13 Long-Context Status

Status: longer-context lookup passed; strict pair16 answer-absent null failed.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_RESPONSE_MARKER_V13_LONG_CONTEXT.md`
- runner:
  `code/mc005_associative_lookup_response_marker_v13_long_context.py`
- result:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v13_long_context_20260630T184515.json`
- result SHA256:
  `8690de803a45351beff3220d198fd961be2719e8aeb0e7b14b29a782989a3f53`

## Verdict

V13 does not pass the preregistered longer-context reliability gate, but it
does extend the positive lookup and off-target evidence through pair count 16.

Summary:

```text
model: Qwen/Qwen3-1.7B
selected band: late_20_26
selected layers: 20, 21, 22, 23, 24, 25, 26
marker: Response
seeds: 17, 23, 31
pair counts: 12, 16
scenario count: 18
lookup scenarios: 6
null scenarios: 12
label counts:
  works: 6
  clean_null: 11
  weak_null: 1
passed: false
```

All lookup scenarios worked at pair counts 12 and 16. Target source-value masks
reduced target-vs-distractor margin by -8.316 to -9.814 and caused target-win
losses of 9 to 16 rows, while distractor masks moved margins upward and random
masks stayed near zero.

All off-target null scenarios were clean at pair counts 12 and 16. The
pair-count 12 answer-absent null was also clean on all three seeds.

The only failure was `answer_absent_pair16_response_null` on seed 23. It was
`weak_null` because the final-label arm had mean delta -0.5039, just beyond the
strict clean bound of 0.50. It changed zero target-win rows.

## Scenario Table

| Seed | Scenario | Label | Clean rows | Key arm means |
| ---: | --- | --- | ---: | --- |
| 17 | `lookup_pair12_response` | works | 31/32 | target -8.861, distractor +1.383, random +0.023 |
| 17 | `lookup_pair16_response` | works | 32/32 | target -8.509, distractor +0.824, random +0.046 |
| 17 | `offtarget_pair12_response` | clean_null | 32/32 | irrelevant value +0.037, irrelevant key -0.020, random -0.018 |
| 17 | `offtarget_pair16_response` | clean_null | 31/32 | irrelevant value +0.014, irrelevant key -0.014, random -0.016 |
| 17 | `answer_absent_pair12_response_null` | clean_null | 32/32 | source +0.000, control -0.045, earlier colon +0.000, final label -0.367, final colon -0.025 |
| 17 | `answer_absent_pair16_response_null` | clean_null | 32/32 | source +0.045, control +0.053, earlier colon +0.006, final label -0.348, final colon +0.186 |
| 23 | `lookup_pair12_response` | works | 32/32 | target -9.583, distractor +1.393, random -0.015 |
| 23 | `lookup_pair16_response` | works | 32/32 | target -9.814, distractor +1.087, random +0.082 |
| 23 | `offtarget_pair12_response` | clean_null | 32/32 | irrelevant value +0.006, irrelevant key +0.051, random +0.025 |
| 23 | `offtarget_pair16_response` | clean_null | 30/32 | irrelevant value +0.016, irrelevant key +0.040, random -0.009 |
| 23 | `answer_absent_pair12_response_null` | clean_null | 32/32 | source -0.008, control +0.209, earlier colon -0.014, final label -0.084, final colon +0.061 |
| 23 | `answer_absent_pair16_response_null` | weak_null | 32/32 | source -0.027, control +0.135, earlier colon +0.002, final label -0.504, final colon +0.215 |
| 31 | `lookup_pair12_response` | works | 31/32 | target -8.895, distractor +1.352, random -0.025 |
| 31 | `lookup_pair16_response` | works | 31/32 | target -8.316, distractor +0.744, random +0.062 |
| 31 | `offtarget_pair12_response` | clean_null | 31/32 | irrelevant value -0.006, irrelevant key -0.022, random +0.002 |
| 31 | `offtarget_pair16_response` | clean_null | 32/32 | irrelevant value +0.037, irrelevant key +0.008, random +0.004 |
| 31 | `answer_absent_pair12_response_null` | clean_null | 32/32 | source -0.023, control +0.031, earlier colon -0.033, final label -0.361, final colon -0.018 |
| 31 | `answer_absent_pair16_response_null` | clean_null | 32/32 | source +0.012, control +0.172, earlier colon -0.012, final label -0.357, final colon +0.252 |

## Interpretation

What V13 supports:

- the Qwen3-1.7B lookup intervention remains strong through pair count 16;
- off-target same-grammar nulls remain clean through pair count 16;
- answer-absent nulls remain clean through pair count 12;
- the pair16 answer-absent failure is a strict margin-bound failure, not a
  target-win flip failure.

What V13 blocks:

- MC005 cannot claim complete Qwen3-1.7B longer-context reliability through
  pair count 16;
- the final-label answer-absent control becomes tight at pair count 16;
- mechanism-card promotion still needs this context-boundary resolved or
  explicitly scoped.

## Next Step

The next MC005 pass should diagnose the pair16 answer-absent final-label
boundary on Qwen3-1.7B:

- more seeds and rows for pair16 answer-absent `Response:`;
- row-level margins for the final-label arm;
- compare whether the effect is specific to pair16 or appears gradually between
  pair counts 12 and 16;
- do not rerun the full lookup atlas until the null boundary is understood.
