# MC004 In-Context Binding Preregistration

Date: 2026-06-30

## Motivation

MC003 showed that arbitrary delayed-copy prompts can produce a real behavior
transition, but the hidden signature failed once condition confounding was
removed. MC004 moves to a different behavior family:

> bind a nonce entity to an in-context attribute value and preserve that
> binding under wrong-hint pressure.

This is still controlled and artificial, but it is closer to a knowledge-use
primitive than copying one private code word into a delayed final slot.

## Frozen Inputs

- model: `google/gemma-2-2b-it`
- runner: `code/mc004_in_context_binding_smoke.py`
- source bank: 40 nonce country names and city-like code words from MC002
- card ID: `MC004`
- artifact prefix: `mc004_gemma2_2b_it_in_context_binding`
- render mode: `chat`
- source split: source-disjoint, same modulo-3 split pattern as MC003
- parser: exact answer containment for target versus distractor code values

## Task

Each source contains two in-context reference notes:

- target nonce entity -> target code value;
- distractor nonce entity -> distractor code value.

The prompt asks for the target entity's access code. The model must output only
the code value.

Target/distractor note order is counterbalanced across sources.

## Conditions

Locality guards:

- `neutral`
- `cautious`
- `correct_hint`

Wrong-hint calibration conditions:

- `wrong_hint_guarded`
- `wrong_hint_default`
- `wrong_hint_balanced`
- `wrong_hint_pressure`
- `wrong_hint_authority`

The wrong-hint conditions span weak to strong user pressure. They are behavior
calibration candidates only; none is a mechanism claim.

## Behavior Gate

MC004 qualifies for a future signature attempt only if all locality guards pass
and at least one wrong-hint condition has within-condition target/distractor
variation:

- at least 32 sources are target-correct in both baseline conditions;
- at least 32 sources are target-correct under `correct_hint`;
- a candidate wrong-hint condition has at least 10 `target_correct` rows and at
  least 10 `distractor_followed` rows;
- that condition has at least 6 discovery rows and at least 3 holdout rows for
  each of `target_correct` and `distractor_followed`;
- each holdout target-order subgroup has both labels represented;
- source split overlap is zero.

If more than one condition qualifies, freeze the qualifying condition with the
smallest absolute target-versus-distractor count difference, breaking ties
toward weaker pressure.

## Failure Criteria

The gate fails if:

- the model cannot cleanly read the in-context reference notes;
- correct hints or cautious instructions damage locality;
- all wrong-hint conditions collapse to target-copying;
- all wrong-hint conditions collapse to distractor-following;
- the only balanced condition lacks source-disjoint or target-order holdout
  support.

## If The Gate Passes

Passing permits a separate condition-balanced signature preregistration. It
does not permit steering, patching, sparse-feature search, circuit
localization, or a mechanism claim.
