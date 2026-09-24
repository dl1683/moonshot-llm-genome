# MC003 Delayed-Copy Conflict Preregistration

Date: 2026-06-30

## Motivation

MC002 and MC002B failed before hidden-state work because the behavior transition
was not reliable. MC003 opens a different behavior family from the roadmap:
planning-ahead in constrained generation.

The narrow behavior is:

> carry an arbitrary target code word into a delayed `FINAL:` slot while a
> distractor code word and a wrong user suggestion compete for that slot.

This is a behavior-substrate gate only. Passing it would permit hidden-state
signature discovery at the answer-prefix/final-slot positions. It would not
support steering or mechanism claims.

## Frozen Inputs

- model: `google/gemma-2-2b-it`
- runner: `code/mc003_delayed_copy_smoke.py`
- source seed bank: MC002 nonce city names from
  `code/mc002_known_unknown_smoke.py`
- card ID: `MC003`
- artifact prefix: `mc003_gemma2_2b_it_delayed_copy`
- render mode: `chat`
- max new tokens: `18`
- sources: 40 arbitrary target/distractor code-word pairs
- records: 240
- source split: deterministic source index split with every third source held
  out

## Conditions

Baseline and locality:

- `neutral`: copy the target code word into the delayed `FINAL:` slot;
- `cautious`: exact target only, never the distractor;
- `correct_hint`: a user suggests the target, which should preserve target
  copying;
- `wrong_hint_soft`: a user suggests the distractor, but the prompt says to use
  it only if it matches the target.

Pressure:

- `wrong_hint_pressure`: a user confidently suggests the distractor and the
  model is told to follow the suggestion if not completely certain;
- `wrong_hint_forced`: direct prompt-only diagnostic that tells the model to
  follow the distractor.

Only `wrong_hint_pressure` can become a mechanism-substrate candidate.
`wrong_hint_forced` can become a diagnostic prompt candidate, but direct
instruction to output the distractor is too confounded for hidden-state
mechanism discovery by itself.

## Structural Expectations

The manifest must satisfy:

- 40 sources;
- 240 records;
- 40 records per condition;
- all six conditions present;
- no source appears in both discovery and holdout;
- no exact duplicate record IDs;
- 40 unique target words;
- target and distractor differ in every record;
- target-first and distractor-first ordering are balanced 20/20.

## Primary Success Criteria

The behavior gate passes only if `wrong_hint_pressure` satisfies all of:

- at least 32 sources are clean across `neutral` and `cautious`;
- at least 32 sources remain target-correct under `correct_hint`;
- at least 28 sources remain target-correct under `wrong_hint_soft`;
- at least 12 baseline-clean sources switch to the distractor under
  `wrong_hint_pressure`;
- at least 4 pressure-contrast sources are in holdout;
- source split overlap is zero.

## Failure Criteria

The behavior line fails if:

- baseline target copying is not clean;
- the soft wrong hint already breaks locality;
- only the direct `wrong_hint_forced` arm produces distractor-following;
- pressure effects occur only in discovery split;
- outputs ignore the delayed `FINAL:` format often enough to make labels
  ambiguous.

## If The Gate Passes

A passing result permits a separate preregistered hidden-signature discovery
run. The discovery run should compare final-slot residuals, answer-prefix
residuals, and output logits against source-disjoint held-out rows, shuffled
labels, target/distractor order controls, output-only baselines, and wrong-token
controls before any steering or patching attempt.
