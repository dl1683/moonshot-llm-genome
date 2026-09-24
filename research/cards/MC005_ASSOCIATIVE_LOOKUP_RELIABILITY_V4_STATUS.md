# MC005 Associative Lookup Reliability V4 Status

Status: lookup reliability strengthened; off-target null failed.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_RELIABILITY_V4.md`
- runner:
  `code/mc005_associative_lookup_reliability_v4.py`
- result:
  `results/cards/MC005/mc005_qwen3_1p7b_associative_lookup_reliability_v4_20260630T172154.json`
- result SHA256:
  `bc9c98738c1a5e86e3c1f61fc3f43eb4f1f6680e5ecda82b1809aeffa1c9155c`

## Verdict

V4 strengthens the MC005 reliability atlas but does not promote MC005 to a full
mechanism card.

Summary:

```text
lookup scenarios: 8/8 work
off-target scenarios: 0/1 clean nulls
atlas pass: false
```

The late-band layers-20-26 source-value intervention is robust across the
tested lookup layouts, pair counts, and shifted lexicon. The off-target null
failed because masking an unrelated source value still moved the target-vs-
distractor margin by +0.713, above the preregistered 0.50 null bound.

## Scenario Table

| Scenario | Mode | Label | Clean rows | Target delta | Target-win loss | Distractor delta | Random delta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `pair3_dash_base` | lookup | works | 23/32 | -3.803 | 15 | +1.455 | +1.404 |
| `pair5_dash_base` | lookup | works | 23/32 | -3.268 | 15 | +0.850 | +0.420 |
| `pair8_dash_base` | lookup | works | 28/32 | -3.412 | 17 | +0.227 | +0.389 |
| `pair5_arrow_base` | lookup | works | 29/32 | -6.010 | 26 | +0.537 | +0.479 |
| `pair8_arrow_base` | lookup | works | 30/32 | -5.793 | 22 | +0.371 | +0.545 |
| `pair5_sentence_base` | lookup | works | 29/32 | -7.525 | 23 | +0.697 | +0.500 |
| `pair5_arrow_holdout_lexicon` | lookup | works | 28/32 | -6.383 | 26 | +1.107 | +0.668 |
| `pair8_dash_holdout_lexicon` | lookup | works | 24/32 | -4.088 | 14 | +0.457 | +0.246 |
| `offtarget_pair5_dash` | off-target | side_effect | 28/32 | +0.713 | -1 | +0.471 | +0.994 |

## Generation-Mode Readout

V4 also tracked greedy next-token labels, not just forced-choice margins.

| Scenario | Baseline greedy target | Target-mask greedy target |
| --- | ---: | ---: |
| `pair3_dash_base` | 9 | 0 |
| `pair5_dash_base` | 12 | 0 |
| `pair8_dash_base` | 16 | 2 |
| `pair5_arrow_base` | 22 | 0 |
| `pair8_arrow_base` | 28 | 3 |
| `pair5_sentence_base` | 28 | 0 |
| `pair5_arrow_holdout_lexicon` | 24 | 0 |
| `pair8_dash_holdout_lexicon` | 12 | 1 |
| `offtarget_pair5_dash` | 16 | 15 |

The target source-value mask usually removes greedy target outputs in lookup
scenarios. The off-target scenario again behaves differently: target-mask
greedy target count stays nearly unchanged.

## Interpretation

What V4 supports:

- the late-band intervention is not a one-layout artifact;
- it survives pair counts 3, 5, and 8 in this synthetic setup;
- it survives a shifted lexicon bank;
- it changes greedy next-token behavior, not only forced-choice margin.

What V4 blocks:

- MC005 is not yet deployable or broad;
- the off-target null is not clean under the current construction;
- distractor and random source masks often move the margin upward, so the path
  is source-specific but not side-effect-free;
- no cross-model replication has been run in V4.

## Next Step

The next MC005 run should repair off-target controls before expanding the claim:

- construct off-target prompts where the scored output does not share the same
  prompt grammar as lookup rows;
- add no-reference and non-lookup language prompts;
- test whether random-value positive deltas are a benign contrast effect or a
  broad source-mask side effect;
- then run Qwen3-0.6B or Gemma 2B replication only after the off-target null is
  clean.
