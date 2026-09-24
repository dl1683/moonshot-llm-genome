# MC001G Gemma 2 2B Format-Control V2 Preregistration

Date: 2026-06-30

## Motivation

The first option-position counterbalanced gate fixed part of the prior control
failure: strict matching by no-hint-margin bin and correct answer letter reduced
the A-correct agreement dominance. It still failed as a substrate because only
12 source groups were clean in all four answer positions and some strict
holdout bin-letter keys lacked discovery support.

V2 tests the same format control on a larger source bank. This is the right
repair because the failure was sparse exact coverage under source-group splitting,
not a reason to return to unpermuted item expansion.

## Frozen Inputs

- model: `google/gemma-2-2b`
- runner: `code/mc001_logit_smoke.py`
- audit: `code/mc001_gemma_format_control_audit.py`
- variant: `gemma_repair_permuted_expanded`
- render mode: `raw`
- conditions: unchanged `CONDITIONS_GEMMA_REPAIR`
- source items: the 64 original `gemma_repair_*` items plus the 64 broad
  `gemma_repair_x*` expansion items
- generated items: 512, four option-position permutations per source item
- generated records: 3,584
- clean item rule: `no_hint` and `correct_hint` must both be truth-following
- matching conditions: `wrong_disclaimed`, `wrong_unsure`
- margin bin width: `0.5`
- split rule: source-group split, so all permutations of a source question are
  assigned to discovery or holdout together

## Structural Expectations

The manifest must be balanced by construction:

- 128 source groups;
- 512 items;
- 128 items with each correct answer letter;
- 128 items with each wrong hint letter;
- no source group appears in both discovery and holdout under the audit split.

## Primary Success Criteria

This is a behavior-substrate gate, not a mechanism claim. It passes only if all
of the following are true:

- at least 220 clean items;
- at least 32 clean items for each correct answer letter;
- at least 24 source groups have all four answer positions clean;
- primary weak-wrong-hint rows include at least 100 truth-following rows and at
  least 100 user-agreement rows;
- bin-only matching produces at least 160 matched rows;
- strict matching by both no-hint-margin bin and correct answer letter produces
  at least 128 matched rows;
- strict matched holdout contains at least 40 rows;
- every correct answer letter contributes at least 8 strict matched rows to each
  binary label;
- no single correct answer letter supplies more than 50 percent of either strict
  matched binary label;
- every strict matched holdout bin-letter key has a matched discovery key.

## Failure Criteria

The run fails as a substrate repair if any primary success criterion fails. If
the gate passes only the bin-only matching but fails strict answer-letter
matching, it remains a format-control failure and hidden-state work must not
resume.

## Downstream Decision

If V2 passes, rerun dense discovery on the strict matched rows with source-group
splits before any intervention. If V2 fails, close the current letter-choice
Gemma MC001G route and redesign the behavior task format more deeply.
