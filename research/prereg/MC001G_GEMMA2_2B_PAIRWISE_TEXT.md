# MC001G Gemma 2 2B Pairwise Text Preregistration

Date: 2026-06-30

## Motivation

The MC001G letter-choice branch is closed as a mechanism substrate. Option
position counterbalancing improved the controls, but the expanded
format-control run still failed exact source-disjoint holdout coverage and left
a small answer-position cell underfilled.

This run changes the behavior interface. Instead of asking the model to choose
`A`, `B`, `C`, or `D`, each source question is reduced to a pairwise choice
between the correct answer text and the weak wrong-hint answer text. The scorer
compares sequence logprob for the two answer texts directly.

## Frozen Inputs

- model: `google/gemma-2-2b`
- runner: `code/mc001_logit_smoke.py`
- audit: `code/mc001_gemma_pairwise_text_audit.py`
- variant: `gemma_pairwise_text`
- render mode: `raw`
- source items: the 64 original `gemma_repair_*` items plus the 64 broad
  `gemma_repair_x*` expansion items
- excluded source rule: drop source items where either the correct or wrong
  answer text is exactly `A`, `B`, `C`, or `D`
- retained source groups: 125
- generated items: 250, two pair orders per retained source
- generated records: 1,750
- pair orders:
  - `cw`: correct answer text appears first in the pair;
  - `wc`: wrong answer text appears first in the pair
- conditions: unchanged weak-to-strong hint ladder translated from option
  letters to answer text
- clean item rule: `no_hint` and `correct_hint` must both be truth-following
- matching conditions: `wrong_disclaimed`, `wrong_unsure`
- margin bin width: `0.5`
- split rule: source-group split, so both pair orders for a source question are
  assigned to discovery or holdout together
- strict match: item-level `no_hint` margin bin plus pair order

## Structural Expectations

The manifest must satisfy all of the following:

- 125 retained source groups;
- 250 pairwise items;
- 1,750 scored records;
- 125 `cw` items and 125 `wc` items;
- no source group appears in both discovery and holdout under the audit split;
- no candidate answer text is exactly `A`, `B`, `C`, or `D`.

## Primary Success Criteria

This is a behavior-substrate gate, not a mechanism claim. It passes only if all
of the following are true:

- at least 150 clean items;
- at least 60 clean `cw` items and at least 60 clean `wc` items;
- at least 60 source groups have both pair orders clean;
- primary weak-wrong-hint rows include at least 80 truth-following rows and at
  least 80 user-agreement rows;
- bin-only matching produces at least 120 matched rows;
- strict matching by both no-hint-margin bin and pair order produces at least
  96 matched rows;
- strict matched holdout contains at least 30 rows;
- each pair order contributes at least 20 strict matched rows to each binary
  label;
- no single pair order supplies more than 65 percent of either strict matched
  binary label;
- every strict matched holdout bin-order key has a matched discovery key.

## Failure Criteria

The run fails as a substrate repair if any primary success criterion fails. If
the behavior gate passes size bars but fails pair-order matching or exact
holdout coverage, do not resume hidden-state work; write the result as another
interface-control failure.

If the gate passes, the next step is dense discovery on the strict matched rows
with source-group splitting and text-choice sequence-logprob recomputation in
the hidden-state forward pass. No intervention is allowed directly from this
behavior gate.
