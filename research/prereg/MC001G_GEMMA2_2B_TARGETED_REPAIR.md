# MC001G Gemma 2 2B Targeted Repair Preregistration

Date: 2026-06-30

## Motivation

The broad `gemma_repair_expanded` bank improved MC001G but did not pass the
matching gate. It produced 76 clean items and 74 matched wrong-hint rows, but
discovery split bin `3` had only truth-following rows while holdout bin `3`
needed exact discovery donor coverage.

This run tests a narrower repair: append a candidate slice designed to add
moderate-confidence, wrong-hint-swayable examples without changing the scoring
or matching rules.

## Frozen Inputs

- model: `google/gemma-2-2b`
- runner: `code/mc001_logit_smoke.py`
- variant: `gemma_repair_targeted`
- render mode: `raw`
- conditions: unchanged `CONDITIONS_GEMMA_REPAIR`
- matching conditions: `wrong_disclaimed`, `wrong_unsure`
- bin width: `0.5`
- clean item rule: `no_hint` and `correct_hint` must both be truth-following
- split rule: deterministic per answer letter, sorted item ids, every third
  clean item held out

## Candidate Slice

`gemma_repair_targeted` equals:

1. the original 64 `gemma_repair` items;
2. the 64 broad expansion items;
3. 64 appended `gemma_repair_z*` targeted items.

The targeted slice is C/D-heavy, but not C/D-only:

| Correct Letter | Targeted Items |
| --- | ---: |
| A | 8 |
| B | 8 |
| C | 24 |
| D | 24 |

The `z*` ids intentionally sort after the existing `x*` ids, so existing clean
items keep their prior deterministic split assignments.

## Primary Success Criteria

This is a substrate repair, not a mechanism claim. It passes only if all of the
following are true:

- at least 84 clean items;
- at least 18 clean C items and at least 18 clean D items;
- pre-hint-margin matching produces at least 88 matched wrong-hint rows;
- matched holdout contains at least 28 rows;
- every matched holdout no-hint-margin bin has at least one matched discovery
  row in the same bin;
- discovery bin `3` has both truth-following and user-agreement rows after
  matching.

## Failure Criteria

The run fails as a substrate repair if any of the primary success criteria fail.
It also fails if added rows pass only by creating a new obvious confound, such as
matched labels being dominated by one answer letter or one condition.

## Downstream Decision

If the targeted repair passes, rerun pre-hint-margin dense discovery on the
targeted result before any sparse/path/intervention work. If it fails, document
the failure and do not rerun intervention.
