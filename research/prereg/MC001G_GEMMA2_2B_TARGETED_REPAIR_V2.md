# MC001G Gemma 2 2B Targeted Repair V2 Preregistration

Date: 2026-06-30

## Motivation

The first targeted repair produced 109 clean items and 120 matched rows, but it
still failed exact same-bin discovery coverage for holdout bin `3`. It also
introduced a matched-label skew: A-correct rows dominated user-agreement errors,
while C/D-correct rows mostly supplied truth-following rows.

V2 tests whether close numeric distractors can add cleaner, less
answer-letter-confounded weak-hint examples.

## Frozen Inputs

- model: `google/gemma-2-2b`
- runner: `code/mc001_logit_smoke.py`
- variant: `gemma_repair_targeted_v2`
- render mode: `raw`
- conditions: unchanged `CONDITIONS_GEMMA_REPAIR`
- matching conditions: `wrong_disclaimed`, `wrong_unsure`
- bin width: `0.5`
- clean item rule: `no_hint` and `correct_hint` must both be truth-following
- split rule: deterministic per answer letter, sorted item ids, every third
  clean item held out

## Candidate Slice

`gemma_repair_targeted_v2` equals:

1. the original 64 `gemma_repair` items;
2. the 64 broad expansion items;
3. the 64 first targeted `z*` items;
4. 64 appended `zz*` close-distractor numeric items.

The V2 slice is answer-balanced:

| Correct Letter | V2 Items |
| --- | ---: |
| A | 16 |
| B | 16 |
| C | 16 |
| D | 16 |

The `zz*` ids sort after the earlier `z*` ids, preserving prior split
assignments for all previously clean items.

## Primary Success Criteria

This is still only a substrate repair. It passes only if all of the following
are true:

- at least 120 clean items;
- at least 24 clean C items and at least 24 clean D items;
- pre-hint-margin matching produces at least 128 matched wrong-hint rows;
- matched holdout contains at least 40 rows;
- every matched holdout no-hint-margin bin has at least one matched discovery
  row in the same bin;
- discovery bin `3` has both truth-following and user-agreement rows after
  matching;
- C/D-correct rows contribute at least 16 matched user-agreement rows;
- no single correct answer letter accounts for more than 50% of either matched
  binary label.

## Failure Criteria

The run fails as a substrate repair if any primary success criterion fails. If
it increases matched-row count but preserves the bin-3 or answer-letter
confound, it is logged as a useful diagnostic control, not an intervention
substrate.

## Downstream Decision

If V2 passes, rerun pre-hint-margin dense discovery on the V2 result before any
sparse/path/intervention work. If V2 fails, document the failure and inspect
whether the task format itself is creating an answer-letter control problem.
