# MC001G Gemma 2 2B Pairwise Text V2 Preregistration

Date: 2026-06-30

## Motivation

The first pairwise answer-text gate changed the behavior interface away from
letters and passed the main clean-size and exact holdout coverage checks, but
it did not pass the full reliability bar. The strict matched set was still
pair-order skewed: `cw` contributed only 18 rows per binary label against a
floor of 20, while `wc` supplied 44/62 rows for each strict matched label.

V2 tests whether that remaining order skew is a sparse-coverage problem rather
than a fundamental interface failure. It keeps the same answer-text scoring and
source-group split, but adds the targeted source banks that were not used in
the first pairwise text run.

## Frozen Inputs

- model: `google/gemma-2-2b`
- runner: `code/mc001_logit_smoke.py`
- audit: `code/mc001_gemma_pairwise_text_audit.py`
- variant: `gemma_pairwise_text_v2`
- render mode: `raw`
- source items: the 64 original `gemma_repair_*` items, the 64 broad
  `gemma_repair_x*` expansion items, the 64 `gemma_repair_z*` targeted items,
  and the 64 `gemma_repair_zz*` targeted V2 items
- excluded source rule: drop source items where either the correct or wrong
  answer text is exactly `A`, `B`, `C`, or `D`
- retained source groups: 253
- generated items: 506, two pair orders per retained source
- generated records: 3,542
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

- 253 retained source groups;
- 506 pairwise items;
- 3,542 scored records;
- 253 `cw` items and 253 `wc` items;
- no source group appears in both discovery and holdout under the audit split;
- no candidate answer text is exactly `A`, `B`, `C`, or `D`.

## Primary Success Criteria

This is a behavior-substrate gate, not a mechanism claim. It passes only if all
of the following are true:

- at least 320 clean items;
- at least 140 clean `cw` items and at least 140 clean `wc` items;
- at least 140 source groups have both pair orders clean;
- primary weak-wrong-hint rows include at least 160 truth-following rows and at
  least 160 user-agreement rows;
- bin-only matching produces at least 240 matched rows;
- strict matching by both no-hint-margin bin and pair order produces at least
  192 matched rows;
- strict matched holdout contains at least 60 rows;
- each pair order contributes at least 40 strict matched rows to each binary
  label;
- no single pair order supplies more than 65 percent of either strict matched
  binary label;
- every strict matched holdout bin-order key has a matched discovery key.

## Failure Criteria

The run fails as a substrate repair if any primary success criterion fails. If
V2 still fails pair-order balance or exact holdout coverage, close pairwise
answer-text MC001G as an interface-control failure and move to a generated
answer grading interface rather than hidden-state work.

If V2 passes, the next step is dense discovery on the strict matched rows with
source-group splitting and text-choice sequence-logprob recomputation in the
hidden-state forward pass. No intervention is allowed directly from this
behavior gate.
