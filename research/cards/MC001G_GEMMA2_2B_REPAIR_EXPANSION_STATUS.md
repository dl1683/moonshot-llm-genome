# MC001G Gemma 2 2B Repair Expansion Status

Status: complete. Expanded repair gate partially improved the substrate but did
not pass the preregistered matching bar.

Date: 2026-06-30

## Artifacts

- runner: `code/mc001_logit_smoke.py`
- variant: `gemma_repair_expanded`
- preregistration: `research/prereg/MC001G_GEMMA2_2B_REPAIR_EXPANSION.md`
- manifest: `data/cards/MC001G/mc001g_gemma2_2b_repair_expanded_logit_raw_manifest.jsonl`
- manifest SHA256: `8f69580cf8fd193f09da9b525e264aae87195b4ed398200a05c0decc80a37004`
- result: `results/cards/MC001G/mc001g_gemma2_2b_repair_expanded_logit_raw_gemma_repair_expanded_20260630T130645.json`
- result SHA256: `9edde8dfa7ff817e64aaece7ba0dd2b4dceeaf161ce3ae1eeea5cdaf7203671c`
- model: `google/gemma-2-2b`
- render mode: `raw`
- items: 128
- records: 896

## Question

Does adding 64 stable objective multiple-choice items create a larger repaired
and pre-hint-margin matched substrate for future sparse/path intervention work?

## Gate Verdict

No, not by the preregistered bar.

The expansion improved the substrate, but it did not produce enough matched
wrong-hint rows or exact discovery/holdout bin coverage to justify another
intervention pass.

Write this as:

> MC001G expanded repair increased clean items and matched holdout size, but it
> still does not provide a reliable intervention substrate.

## Behavior Summary

| Condition | Truth | Agreement | Other | Notes |
| --- | ---: | ---: | ---: | --- |
| no_hint | 78/128 | 21/128 | 29/128 | noisy baseline |
| correct_hint | 125/128 | 1/128 | 2/128 | strong hint repair |
| wrong_disclaimed | 45/128 | 81/128 | 2/128 | mixed pressure |
| wrong_unsure | 37/128 | 91/128 | 0/128 | mixed pressure |
| wrong_direct | 5/128 | 123/128 | 0/128 | near saturated agreement |
| wrong_high | 0/128 | 128/128 | 0/128 | saturated agreement |
| anti_wrong | 90/128 | 0/128 | 38/128 | protects against agreement but produces other errors |

## Clean Items

Clean means `no_hint` and `correct_hint` are both truth-following.

| Metric | Result | Preregistered Bar | Verdict |
| --- | ---: | ---: | --- |
| clean items | 76 | at least 72 | pass |
| clean A items | 29 | at least 12 | pass |
| clean B items | 21 | at least 12 | pass |
| clean C items | 13 | at least 12 | pass |
| clean D items | 13 | at least 12 | pass |

The expanded bank increased clean items from 42 to 76, but C/D remain the weak
letters.

## Matched Rows

Rows were matched on item-level `no_hint` margin with bin width `0.5`, using the
same `wrong_disclaimed` and `wrong_unsure` conditions as the prior matched
gates.

| Metric | Result | Preregistered Bar | Verdict |
| --- | ---: | ---: | --- |
| primary wrong-hint rows | 151 | not fixed | informative |
| primary truth rows | 71 | not fixed | informative |
| primary agreement rows | 80 | not fixed | informative |
| matched rows | 74 | at least 80 | fail |
| matched holdout rows | 32 | at least 24 | pass |

Matched row count improved from 48 to 74 and matched holdout improved from 14
to 32, but the total matched set missed the preregistered floor by 6 rows.

## Bin Coverage

Discovery matched bins:

| No-Hint Margin Bin | Truth | Agreement |
| --- | ---: | ---: |
| 0 | 2 | 2 |
| 1 | 10 | 10 |
| 2 | 6 | 6 |
| 5 | 2 | 2 |
| 6 | 1 | 1 |

Holdout matched bins:

| No-Hint Margin Bin | Truth | Agreement |
| --- | ---: | ---: |
| 0 | 3 | 3 |
| 1 | 2 | 2 |
| 2 | 6 | 6 |
| 3 | 5 | 5 |

Holdout bin `3` still lacks discovery-bin coverage. Future donor-based
activation patching would still require nearest-bin fallback for those rows.

## Decision

Do not rerun sparse or path interventions on this expanded set yet.

The expansion is a useful substrate improvement, not a pass. The next aligned
step is a targeted item-bank repair focused on:

1. more clean C/D items;
2. more discovery rows in no-hint-margin bin `3`;
3. enough matched rows to exceed the 80-row floor with exact donor-bin coverage.
