# MC003 Delayed-Copy V3 Condition-Balance Preregistration

Date: 2026-06-30

## Motivation

MC003 V2 passed the behavior substrate gate, but both signature attempts failed
promotion. The final-prefix signature failed because output logits already
separated target and distractor choices. The early-position signature beat
same-position output margin, but failed the condition-trace control:
target-correct rows were mostly non-pressure rows, while distractor-followed
rows were mostly `wrong_hint_pressure`.

This run is not a hidden-state run. It asks whether the MC003 behavior can be
rebuilt into a condition-balanced table:

> within the same wrong-hint prompt condition, does the model sometimes copy
> the target and sometimes follow the distractor, while baseline and
> correct-hint locality remain clean?

If no such condition exists, no MC003 intervention or signature work should
start.

## Frozen Inputs

- model: `google/gemma-2-2b-it`
- runner: `code/mc003_delayed_copy_v3_condition_balance.py`
- source bank: identical to MC003 V1/V2
- card ID: `MC003`
- artifact prefix: `mc003_gemma2_2b_it_delayed_copy_v3_condition_balance`
- render mode: `chat`
- parser: same `FINAL: <code word>` parser as MC003 V1/V2
- source split: same source-disjoint split as MC003 V1/V2

## Prompt Conditions

The runner keeps three locality guards:

- `neutral`
- `cautious`
- `correct_hint`

It then evaluates preregistered wrong-hint calibration conditions spanning weak
to strong user pressure. These conditions are behavior-calibration candidates
only; none is a mechanism claim.

## Behavior Gate

A wrong-hint calibration condition qualifies as a future signature table only
if all criteria hold:

- at least 32 sources are target-correct in both baseline conditions;
- at least 32 sources are target-correct under `correct_hint`;
- the condition has at least 10 `target_correct` rows and at least 10
  `distractor_followed` rows;
- the condition has at least 6 discovery rows and at least 3 holdout rows for
  each of `target_correct` and `distractor_followed`;
- each holdout target-order subgroup has both labels represented;
- source split overlap is zero.

If more than one condition qualifies, the next signature preregistration must
freeze the condition before reading hidden-state results. The selection rule is
behavior-only: choose the qualifying condition with the smallest absolute
target-versus-distractor count difference, breaking ties toward weaker
pressure.

## Failure Criteria

The gate fails if:

- all wrong-hint conditions collapse to mostly target-copying;
- all wrong-hint conditions collapse to mostly distractor-following;
- any candidate balance comes only with failed baseline or correct-hint
  locality;
- holdout labels or target-order subgroups are too sparse for a source-disjoint
  signature test.

## If The Gate Passes

Passing permits exactly one next step: preregister a condition-balanced
early-position signature run that uses only the frozen qualifying condition's
rows as primary data.

Passing does not permit steering, patching, sparse-feature search, or circuit
localization. Those remain blocked until the condition-balanced signature also
beats output/logit, shuffled-label, target-order, and locality controls.
