# MC001G Gemma 2 2B Targeted Repair Status

Status: complete. Two targeted item-bank repairs improved matched-row volume but
failed the reliability controls needed for another intervention pass.

Date: 2026-06-30

## Artifacts

First targeted repair:

- runner: `code/mc001_logit_smoke.py`
- variant: `gemma_repair_targeted`
- preregistration: `research/prereg/MC001G_GEMMA2_2B_TARGETED_REPAIR.md`
- manifest: `data/cards/MC001G/mc001g_gemma2_2b_repair_targeted_logit_raw_manifest.jsonl`
- manifest SHA256: `bf5ddca49c0f8be90245e69454c5c7cef9710ad4dcd05d0e9038997cd4cffa63`
- result: `results/cards/MC001G/mc001g_gemma2_2b_repair_targeted_logit_raw_gemma_repair_targeted_20260630T131750.json`
- result SHA256: `593f70ac5b755b8fed8668db65a40a2c3f24c771311a1809743b9e68f6af2b77`

Second targeted repair:

- runner: `code/mc001_logit_smoke.py`
- variant: `gemma_repair_targeted_v2`
- preregistration: `research/prereg/MC001G_GEMMA2_2B_TARGETED_REPAIR_V2.md`
- manifest: `data/cards/MC001G/mc001g_gemma2_2b_repair_targeted_v2_logit_raw_manifest.jsonl`
- manifest SHA256: `ad9b4da96c2a918dbf2140a410616775f99426e085576402b89eac189b13811b`
- result: `results/cards/MC001G/mc001g_gemma2_2b_repair_targeted_v2_logit_raw_gemma_repair_targeted_v2_20260630T132304.json`
- result SHA256: `d845a9f560cf2c6df3be61a2fcf471eddd9ca13676c72069668d8753e421b0fb`

Common configuration:

- model: `google/gemma-2-2b`
- render mode: `raw`
- records: 1,344 for targeted, 1,792 for targeted V2
- clean rule: `no_hint` and `correct_hint` both truth-following
- match rule: `wrong_disclaimed` and `wrong_unsure`, matched by item-level
  `no_hint` margin bins with width `0.5`

## Question

Can targeted item-bank repair create a larger, exact-bin matched, less
answer-letter-confounded MC001G substrate for another dense/sparse/path
intervention?

## Gate Verdict

No.

Both targeted repairs improved clean-item count and matched-row count, but both
failed exact discovery coverage for holdout margin bin `3`. V2 also confirmed an
answer-letter control failure: matched user-agreement rows remained dominated by
A-correct examples, while C/D-correct examples still mostly supplied
truth-following rows.

Write this as:

> MC001G targeted repair shows that unpermuted letter-choice expansion can make
> the dataset larger without making the control surface reliable.

## First Targeted Repair

The first targeted slice appended 64 C/D-heavy factual items.

| Metric | Result | Preregistered Bar | Verdict |
| --- | ---: | ---: | --- |
| clean items | 109 | at least 84 | pass |
| clean C items | 26 | at least 18 | pass |
| clean D items | 21 | at least 18 | pass |
| matched rows | 120 | at least 88 | pass |
| matched holdout rows | 50 | at least 28 | pass |
| exact discovery coverage for all holdout bins | no | yes | fail |
| discovery bin 3 has both labels after matching | no | yes | fail |

Matched rows by split and bin:

| Split | Bin | Truth | Agreement |
| --- | ---: | ---: | ---: |
| discovery | 0 | 6 | 6 |
| discovery | 1 | 19 | 19 |
| discovery | 2 | 7 | 7 |
| discovery | 5 | 2 | 2 |
| discovery | 6 | 1 | 1 |
| holdout | 0 | 5 | 5 |
| holdout | 1 | 5 | 5 |
| holdout | 2 | 9 | 9 |
| holdout | 3 | 5 | 5 |
| holdout | 5 | 1 | 1 |

Holdout bin `3` still had no matched discovery rows in the same bin.

Matched correct-letter label counts:

| Correct Letter | Truth | Agreement |
| --- | ---: | ---: |
| A | 6 | 41 |
| B | 11 | 11 |
| C | 26 | 3 |
| D | 17 | 5 |

This is not a clean intervention substrate. It is large, but answer-letter and
margin-bin controls remain weak.

## Targeted Repair V2

V2 appended 64 answer-balanced close-distractor numeric items to test whether
C/D-correct items could become clean and weak-hint-swayable.

Raw behavior became more agreement-heavy, but the clean matched subset still
failed the controls.

| Metric | Result | Preregistered Bar | Verdict |
| --- | ---: | ---: | --- |
| clean items | 130 | at least 120 | pass |
| clean C items | 28 | at least 24 | pass |
| clean D items | 22 | at least 24 | fail |
| matched rows | 128 | at least 128 | pass |
| matched holdout rows | 54 | at least 40 | pass |
| exact discovery coverage for all holdout bins | no | yes | fail |
| discovery bin 3 has both labels after matching | no | yes | fail |
| C/D agreement rows | 8 | at least 16 | fail |
| no single correct letter over 50 percent of either matched label | no | yes | fail |

Matched rows by split and bin:

| Split | Bin | Truth | Agreement |
| --- | ---: | ---: | ---: |
| discovery | 0 | 6 | 6 |
| discovery | 1 | 21 | 21 |
| discovery | 2 | 7 | 7 |
| discovery | 5 | 2 | 2 |
| discovery | 6 | 1 | 1 |
| holdout | 0 | 5 | 5 |
| holdout | 1 | 7 | 7 |
| holdout | 2 | 9 | 9 |
| holdout | 3 | 5 | 5 |
| holdout | 5 | 1 | 1 |

Matched correct-letter label counts:

| Correct Letter | Truth | Agreement |
| --- | ---: | ---: |
| A | 6 | 45 |
| B | 11 | 11 |
| C | 28 | 3 |
| D | 19 | 5 |

The maximum single-letter share among agreement rows was 45/64 = 70.3 percent,
all from A-correct examples.

## Diagnosis

The forced-choice letter format is now the leading control problem.

Adding more unpermuted items can increase clean count and matched-row count, but
the clean matched rows are not exchangeable across answer letters:

- A-correct rows are easy to make weak-hint-swayable;
- C/D-correct rows often fail no-hint because the raw-logit scorer falls to A or
  another non-correct option;
- C/D rows that survive the clean filter tend to be resistant to weak wrong
  hints;
- holdout bin `3` repeatedly lacks exact matched discovery support.

That means another dense, sparse, or activation-replacement intervention would
mostly test answer-position and margin-bin artifacts.

## Decision

Do not rerun dense, sparse, path, or activation-patching interventions on either
targeted repair set.

The next aligned MC001G step is a format-control repair, not more unpermuted
item-bank expansion. The next substrate should use answer-position
counterbalancing or option-permutation controls before hidden-state discovery is
allowed to resume.
