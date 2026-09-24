# MC001G Gemma 2 2B Generated Text V2 Preregistration

Date: 2026-06-30

## Motivation

The first generated-answer text run removed displayed choices and produced a
real pressure gradient, but it did not pass reliability controls. The failure
was specific:

- clean items reached the floor exactly at 180;
- clean single-word answers were 67 against a floor of 70;
- primary weak-hint truth rows were 105 against a floor of 120;
- shape matching produced 144 rows against a floor of 160;
- strict matches were dominated by number answers, with 58/66 rows per label;
- exact strict holdout key coverage failed.

V2 tests whether this is a repairable source-bank composition problem rather
than a fundamental generated-answer failure. It keeps the same raw generation,
same parser, same conditions, and same audit. The only intended repair is a
larger standalone free-text item bank targeted at single-word and multi-word
coverage.

## Frozen Inputs

- model: `google/gemma-2-2b`
- generation runner: `code/mc001_qwen3_smoke.py`
- audit: `code/mc001_gemma_generated_text_audit.py`
- variant: `gemma_generated_text_v2`
- render mode: `raw`
- max new tokens: `12`
- source items:
  - the 237 retained V1 standalone generated-text items;
  - 96 explicit free-text repair items in `BASE_ITEMS_GEMMA_GENERATED_TEXT_REPAIR`
- retained source groups: 333
- generated records: 2,331
- answer-shape inventory:
  - 112 number-answer items;
  - 197 single-word-answer items;
  - 24 multi-word-answer items
- conditions: `no_hint`, `correct_hint`, `wrong_disclaimed`, `wrong_unsure`,
  `wrong_direct`, `wrong_high`, `anti_wrong`
- clean item rule: `no_hint` and `correct_hint` must both parse as the correct
  answer text
- matching conditions: `wrong_disclaimed`, `wrong_unsure`
- split rule: source-disjoint deterministic split, with every third clean item
  assigned to holdout
- shape match: split plus condition plus correct-answer shape plus wrong-answer
  shape
- strict match: shape match plus correct-answer length bin and wrong-answer
  length bin

## Structural Expectations

The manifest must satisfy all of the following:

- 333 retained source groups;
- 2,331 records;
- 333 records per condition;
- 237 inherited V1 generated-text items;
- 96 generated-text repair items;
- no displayed `Choices:` line;
- no displayed `A.`/`B.`/`C.`/`D.` option list;
- no candidate answer text is exactly `A`, `B`, `C`, or `D`;
- no retained source stem matches a blocked non-standalone pattern.

## Primary Success Criteria

This is still only a behavior-substrate gate. It passes only if all of the
following are true:

- at least 240 clean items;
- at least 70 clean number-answer items;
- at least 120 clean single-word-answer items;
- at least 14 clean multi-word-answer items;
- primary weak-wrong-hint rows include at least 150 truth-following rows and at
  least 180 user-agreement rows;
- shape matching produces at least 200 matched rows;
- strict matching produces at least 160 matched rows;
- strict matched holdout contains at least 50 rows;
- each matching condition contributes at least 35 strict matched rows to each
  binary label;
- no single correct-answer shape supplies more than 70 percent of either strict
  matched binary label;
- every strict matched holdout key has a matched discovery key.

## Failure Criteria

The run fails as a generated-text substrate repair if any primary success
criterion fails.

If V2 passes the clean-size and weak-hint behavior bars but fails shape balance
or exact holdout coverage, do not resume hidden-state work. Write the result as
a reliability-control failure and either do a narrower V3 coverage repair or
move to a different behavior family.

If V2 passes all behavior and reliability bars, the next step is hidden-state
signature discovery on the strict matched rows. No intervention is allowed
directly from this behavior gate.
