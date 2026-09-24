# MC005 Associative Lookup Response-Marker V10 Size-Replication Status

Status: size-replication gate failed; lookup replicated, one strict null did not.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_RESPONSE_MARKER_V10_SIZE_REPLICATION.md`
- runner:
  `code/mc005_associative_lookup_response_marker_v10_size_replication.py`
- shared atlas implementation:
  `code/mc005_associative_lookup_response_marker_v9.py`
- result:
  `results/cards/MC005/mc005_qwen3_0p6b_associative_lookup_response_marker_v10_20260630T181650.json`
- result SHA256:
  `1ef00ee5534a03f190aca394d6e52031a3464920b2f9fba314d0584fb01dfd1b`

## Verdict

V10 did not pass the preregistered size-replication gate on
`Qwen/Qwen3-0.6B`.

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
  clean_null: 5
  weak_null: 1
passed: false
```

The positive lookup effect replicated across all six lookup scenarios. The
target source-value mask reduced the target-vs-distractor margin by -3.820 to
-4.919, with target-win losses from 14 to 21 rows. Distractor source-value
masks moved margins upward, and random source-value masks stayed near zero.

The strict suite failed because `answer_absent_response_null` on seed 23 was
`weak_null`, not `clean_null`. Its mean deltas were small, but the
non-source-control-value arm changed two target-win rows, exceeding the strict
clean-null limit of one row. The same scenario was clean on seeds 17 and 31,
and all three off-target same-grammar nulls were clean.

## Scenario Table

| Seed | Scenario | Label | Clean rows | Key arm means |
| ---: | --- | --- | ---: | --- |
| 17 | `lookup_pair5_response` | works | 29/32 | target -4.128, distractor +1.831, random +0.020 |
| 17 | `lookup_pair8_response` | works | 31/32 | target -4.645, distractor +1.366, random +0.001 |
| 17 | `offtarget_pair5_response` | clean_null | 27/32 | irrelevant value -0.033, irrelevant key +0.197, random +0.024 |
| 17 | `answer_absent_response_null` | clean_null | 30/32 | source +0.004, control -0.101, earlier colon +0.042, final label -0.208, final colon -0.267 |
| 23 | `lookup_pair5_response` | works | 28/32 | target -3.820, distractor +1.758, random +0.021 |
| 23 | `lookup_pair8_response` | works | 31/32 | target -4.520, distractor +1.762, random -0.019 |
| 23 | `offtarget_pair5_response` | clean_null | 28/32 | irrelevant value -0.025, irrelevant key +0.126, random +0.007 |
| 23 | `answer_absent_response_null` | weak_null | 30/32 | source -0.002, control +0.039, earlier colon +0.100, final label -0.195, final colon -0.102 |
| 31 | `lookup_pair5_response` | works | 27/32 | target -4.396, distractor +1.956, random +0.104 |
| 31 | `lookup_pair8_response` | works | 29/32 | target -4.919, distractor +1.577, random +0.048 |
| 31 | `offtarget_pair5_response` | clean_null | 30/32 | irrelevant value +0.055, irrelevant key +0.003, random +0.023 |
| 31 | `answer_absent_response_null` | clean_null | 31/32 | source +0.021, control +0.004, earlier colon +0.018, final label -0.273, final colon +0.037 |

## Interpretation

What V10 supports:

- the late-band target source-value intervention is not unique to Qwen3-1.7B;
  it also works on Qwen3-0.6B for pair-count 5 and 8 lookup prompts;
- off-target same-grammar irrelevant source masks were clean on Qwen3-0.6B;
- the strict failure is not a broad source-line collapse or a failed lookup
  substrate.

What V10 blocks:

- MC005 cannot claim size-replicated strict reliability yet;
- the Qwen3-0.6B answer-absent null has a seed-sensitive target-win boundary;
- V10 does not justify model-family or deployable claims.

## Next Step

The next MC005 pass should diagnose the Qwen3-0.6B answer-absent weak null
before treating size replication as solved. Useful follow-ups:

- rerun the answer-absent null with more rows and seeds;
- inspect the two target-win flips in seed 23;
- test whether the weak null is specific to the non-source control value arm,
  the `Response:` marker, or the smaller model's lower baseline margins.
