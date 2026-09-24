# MC003 Delayed-Copy Conflict V2 Preregistration

Date: 2026-06-30

## Motivation

MC003 V1 found a strong behavior transition but failed the correct-hint
locality guard. Baseline target copying was 40/40, `wrong_hint_pressure`
created 35 same-source target-to-distractor contrasts with 12 holdout
contrasts, and `wrong_hint_soft` preserved 39/40 target outputs. The failure
was that `correct_hint` preserved only 28/40 target outputs, mostly on
distractor-first rows.

V2 keeps the source bank, split, parser, output format, model, and success
criteria fixed. The only intended change is explicit hint-locality wording:

> when the user's suggestion matches the target, say directly that FINAL must
> be the target and must not be the distractor.

## Frozen Inputs

- model: `google/gemma-2-2b-it`
- runner: `code/mc003_delayed_copy_v2_smoke.py`
- source seed bank: identical to MC003 V1
- card ID: `MC003`
- artifact prefix: `mc003_gemma2_2b_it_delayed_copy_v2`
- prompt variant: `v2_explicit_hint_locality`
- render mode: `chat`
- max new tokens: `18`
- sources: 40 arbitrary target/distractor code-word pairs
- records: 240
- source split: identical to MC003 V1

## Structural Expectations

The manifest must satisfy the V1 structural expectations plus:

- every record has `prompt_variant = v2_explicit_hint_locality`.

## Primary Success Criteria

The V2 behavior gate uses the same thresholds as V1. It passes only if
`wrong_hint_pressure` satisfies all of:

- at least 32 sources are clean across `neutral` and `cautious`;
- at least 32 sources remain target-correct under `correct_hint`;
- at least 28 sources remain target-correct under `wrong_hint_soft`;
- at least 12 baseline-clean sources switch to the distractor under
  `wrong_hint_pressure`;
- at least 4 pressure-contrast sources are in holdout;
- source split overlap is zero.

## Failure Criteria

The repair fails if:

- explicit correct-hint wording still fails locality;
- pressure disappears after the locality wording is clarified;
- the pressure effect is only a target-order artifact;
- only the direct `wrong_hint_forced` diagnostic crosses the contrast floor.

## If The Gate Passes

Passing permits a separate hidden-signature discovery preregistration. It does
not permit steering, patching, sparse-feature work, or a mechanism claim.
