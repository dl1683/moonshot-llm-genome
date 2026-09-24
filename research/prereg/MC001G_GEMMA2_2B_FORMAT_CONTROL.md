# MC001G Gemma 2 2B Format-Control Preregistration

Date: 2026-06-30

## Motivation

The expanded and targeted MC001G repair banks made the dataset larger, but not
cleaner. The targeted V2 bank reached 128 matched wrong-hint rows while still
failing exact margin-bin coverage and showing a strong answer-letter confound:
A-correct rows supplied 45/64 matched user-agreement rows, while C/D-correct
rows supplied only 8/64.

This run tests whether the forced-choice letter format is the problem. It uses
the original repaired base-Gemma questions but counterbalances answer positions
by construction.

## Frozen Inputs

- model: `google/gemma-2-2b`
- runner: `code/mc001_logit_smoke.py`
- audit: `code/mc001_gemma_format_control_audit.py`
- variant: `gemma_repair_permuted`
- render mode: `raw`
- conditions: unchanged `CONDITIONS_GEMMA_REPAIR`
- source items: the 64 original `gemma_repair_*` items
- generated items: 256, four option-position permutations per source item
- generated records: 1,792
- clean item rule: `no_hint` and `correct_hint` must both be truth-following
- matching conditions: `wrong_disclaimed`, `wrong_unsure`
- margin bin width: `0.5`
- split rule for this gate: source-group split, so all permutations of a source
  question are assigned to discovery or holdout together

## Structural Expectations

The manifest must be balanced by construction:

- 64 items with correct answer A;
- 64 items with correct answer B;
- 64 items with correct answer C;
- 64 items with correct answer D;
- 64 items with wrong hint A;
- 64 items with wrong hint B;
- 64 items with wrong hint C;
- 64 items with wrong hint D;
- no source group appears in both discovery and holdout under the audit split.

## Primary Success Criteria

This is a behavior-substrate gate, not a mechanism claim. It passes only if all
of the following are true:

- at least 96 clean items;
- at least 18 clean items for each correct answer letter;
- at least 16 source groups have all four answer positions clean;
- primary weak-wrong-hint rows include at least 60 truth-following rows and at
  least 60 user-agreement rows;
- bin-only matching produces at least 96 matched rows;
- strict matching by both no-hint-margin bin and correct answer letter produces
  at least 64 matched rows;
- strict matched holdout contains at least 24 rows;
- no single correct answer letter supplies more than 50 percent of either strict
  matched binary label;
- every strict matched holdout bin-letter key has a matched discovery key.

## Failure Criteria

The run fails as a substrate repair if any primary success criterion fails. If
the bank passes size bars but fails strict answer-letter matching, the result
should be written as evidence that letter-position effects still dominate the
forced-choice substrate.

## Downstream Decision

If this gate passes, rerun dense discovery with the source-group split and
strict matched rows before any intervention. If it fails, do not continue Gemma
hidden-state work on MC001G until the task format is changed more deeply, such
as by using non-letter answers, pairwise choices, or generated-answer grading.
