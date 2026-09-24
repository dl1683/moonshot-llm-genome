# MC005 Associative Lookup Off-Target Null V5 Status

Status: repaired primary off-target nulls passed; strict suite failed.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_OFFTARGET_NULL_V5.md`
- runner:
  `code/mc005_associative_lookup_offtarget_null_v5.py`
- result:
  `results/cards/MC005/mc005_qwen3_1p7b_associative_lookup_offtarget_null_v5_20260630T173401.json`
- result SHA256:
  `0405610a166edfb053066a71dbc364492297be3185eb1f8cf29908e34a3826b0`

## Verdict

V5 narrowed the V4 off-target failure but did not fully clear it.

Summary:

```text
primary repaired nulls: 4/4 clean
same-grammar diagnostic: weak_null
strict null suite pass: false
primary repaired-null pass: true
```

The out-of-grammar repaired nulls were clean: explicit-answer prompts with
reference pairs, sentence-style reference notes, no-reference background notes,
and non-lookup marker prompts all stayed within the preregistered mean-delta and
target-win-change bounds.

The same-grammar lookup diagnostic stayed outside the strict clean-null bound:
its three source arms had mean deltas +0.519, +0.533, and +0.619. That is much
weaker than the V4 side effect, but it still means same-grammar lookup prompts
are a boundary.

## Scenario Table

| Scenario | Role | Label | Clean rows | Primary delta | Secondary delta | Random delta | Max abs delta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `same_grammar_query_other_pair` | diagnostic | weak_null | 29/32 | +0.519 | +0.533 | +0.619 | 0.619 |
| `reference_explicit_answer` | primary | clean_null | 32/32 | +0.118 | +0.056 | +0.116 | 0.118 |
| `sentence_reference_explicit_answer` | primary | clean_null | 32/32 | +0.057 | +0.090 | +0.098 | 0.098 |
| `no_reference_note_explicit_answer` | primary | clean_null | 32/32 | +0.264 | +0.131 | +0.260 | 0.264 |
| `nonlookup_marker_answer` | primary | clean_null | 32/32 | +0.295 | +0.266 | +0.264 | 0.295 |

## Target-Win Changes

| Scenario | Primary loss | Secondary loss | Random loss |
| --- | ---: | ---: | ---: |
| `same_grammar_query_other_pair` | -1 | 0 | 0 |
| `reference_explicit_answer` | 0 | 0 | 0 |
| `sentence_reference_explicit_answer` | 0 | 0 | 0 |
| `no_reference_note_explicit_answer` | 0 | 0 | 0 |
| `nonlookup_marker_answer` | 0 | 0 | 0 |

## Interpretation

What V5 supports:

- the late-band source mask is not a broad prompt-destroying perturbation;
- no-reference and non-lookup repaired nulls are clean under the tested prompts;
- the V4 failure was at least partly tied to same-grammar lookup contrast, not
  every irrelevant source token.

What V5 blocks:

- MC005 still has not passed a strict off-target-null suite;
- same-grammar lookup prompts remain sensitive to irrelevant source masking;
- the result does not justify model-family replication as a promotion step yet.

## Next Step

The next MC005 pass should isolate why same-grammar irrelevant source masking
still raises target margins:

- add same-grammar controls with target/distractor answer words absent from all
  source values;
- add query-position and source-position matched no-op masks;
- compare source-value masking against masking key tokens, punctuation tokens,
  and matched non-source value tokens;
- only then decide whether to run Qwen3-0.6B or Gemma replication.
