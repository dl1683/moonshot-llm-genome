# MC011 Atomic-Number Code Arbitration Preregistration

Date: 2026-07-01

## Objective

MC011 tests whether the MC008-MC010 bridge failures were partly caused by answer
type mismatch. The task makes both sides of the conflict short integers:

- prompt-local local lab numbers such as `101`;
- learned real-world atomic numbers such as `1`.

If the model can answer synthetic lookup rows, familiar-entity lookup rows,
real atomic-number controls, and answer-absent null rows, then a conflict table
can test whether learned numeric facts ever compete with prompt-local numeric
authority before hidden-state work begins.

## Behavior Contract

The runner is:

> `code/mc011_atomic_number_code_arbitration.py`

The full behavior table uses 40 chemical elements across source-disjoint
discovery, calibration, and holdout splits. It creates six panels:

- `synthetic_numeric_lookup`;
- `familiar_entity_numeric_lookup`;
- `real_world_atomic_number_control`;
- `authority_dial_conflict`;
- `unlabeled_conflict`;
- `answer_absent_null`.

Candidate labels are:

- `local_number`;
- `atomic_number`;
- `lure_atomic_number`;
- `unknown`.

The prompt must not leak the true atomic number in conflict rows except through
the model's learned memory. Authority dials are represented by words rather than
numeric scores so the prompt does not accidentally reveal atomic numbers such as
`30` or `50`.

## Promotion Rule

MC011 is hidden-state-ready only if a single selected template satisfies all of
these in the full 40-source behavior run:

- synthetic numeric lookup is at least 90 percent local-number answers;
- familiar entity numeric lookup is at least 90 percent local-number answers;
- real-world atomic-number control is at least 85 percent atomic-number answers;
- answer-absent null is at least 90 percent `UNKNOWN`;
- primary conflict parseability is at least 90 percent;
- non-holdout conflict rows contain at least 10 local-number and 10
  atomic/lure-number rows;
- holdout conflict rows contain at least 4 local-number and 4
  atomic/lure-number rows;
- candidate and output margins are reported.

## Death Rule

The route is closed before hidden-state work if direct controls pass but conflict
rows collapse to one side. This is a diagnostic rather than a near miss:
same-format numeric answers repaired the answer interface, so absent conflict
mixture points to prompt-local authority dominance, not a parser artifact.

## Containment Rule

If the route fails, the allowed claim is limited to a behavior diagnostic:
numeric answer-format repair can clean direct controls, but it does not by
itself create a learned-fact-versus-local-table control surface.

## Exported Diagnostics

- `NUMERIC_DIRECT_CONTROLS_PASSED`;
- `NUMERIC_CONFLICT_CONTRAST_ABSENT`;
- `MC011_BEHAVIOR_GATE_FAILED_DIAGNOSTIC_BRIDGE`.

## Forbidden Claims

- MC011 is a mechanism card.
- MC011 supports intervention or steering.
- MC011 found a knowledge-control surface.
- A hidden-state signature should be searched on the MC011 conflict table if
  source-disjoint conflict balance fails.
