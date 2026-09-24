# MC004 In-Context Binding V2 Preregistration

Date: 2026-06-30

## Motivation

MC004 V1 cleanly tested nonce entity-to-code binding, but wrong user hints were
too weak. The model stayed with the reference notes in nearly every row, so no
same-condition target/distractor table existed for signature discovery.

V2 keeps the same source bank and parser but changes the pressure mechanism:

> the model sees original reference notes and then a possible later update for
> the queried entity.

The behavior question becomes whether the model follows the original binding
or the conflicting update.

## Frozen Inputs

- model: `google/gemma-2-2b-it`
- runner: `code/mc004_in_context_binding_v2_smoke.py`
- source bank: identical to MC004 V1
- card ID: `MC004`
- artifact prefix: `mc004_gemma2_2b_it_in_context_binding_v2`
- render mode: `chat`
- source split: identical to MC004 V1
- parser: exact answer containment for original target code versus update code

## Conditions

Locality guards:

- `neutral`
- `cautious`
- `correct_hint`

Conflicting-update calibration conditions:

- `update_unverified`
- `update_ambiguous`
- `update_prefer_latest`
- `update_authoritative`
- `update_forced`

These conditions are behavior calibration candidates only. Direct update
following does not itself count as a mechanism; it only helps find a
same-condition contrast if the model mixes original and update choices.

## Behavior Gate

MC004 V2 qualifies for a future signature attempt only if all locality guards
pass and at least one update condition has within-condition original/update
variation:

- at least 32 sources are target-correct in both baseline conditions;
- at least 32 sources are target-correct under `correct_hint`;
- a candidate update condition has at least 10 `target_correct` rows and at
  least 10 `distractor_followed` rows;
- that condition has at least 6 discovery rows and at least 3 holdout rows for
  each of `target_correct` and `distractor_followed`;
- each holdout target-order subgroup has both labels represented;
- source split overlap is zero.

If more than one condition qualifies, freeze the qualifying condition with the
smallest absolute target-versus-distractor count difference, breaking ties
toward weaker update pressure.

## Failure Criteria

The gate fails if:

- the model cannot cleanly read the original reference notes;
- correct hints or cautious instructions damage locality;
- all update conditions collapse to original-note copying;
- all update conditions collapse to update-following;
- the only balanced condition lacks source-disjoint or target-order holdout
  support.

## If The Gate Passes

Passing permits a separate condition-balanced signature preregistration. It
does not permit steering, patching, sparse-feature search, circuit
localization, or a mechanism claim.
