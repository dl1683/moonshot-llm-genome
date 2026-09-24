# MC031 Statusless Reliability Bridge

Date: 2026-07-01

Status: preregistered bridge-substrate attempt.

Runner:

> `code/mc031_statusless_reliability_bridge.py`

## Purpose

MC031 targets the closure-plan work order
`build_statusless_bridge_substrate`.

The bridge program has one prompt-visible positive control, MC012, where
trusted/untrusted status text cleanly creates local-versus-learned numeric
arbitration. It also has multiple failures after removing or replacing that
channel:

- MC013: text-identical status-channel ablation collapsed;
- MC014: calibration-inferred reliability collapsed to prompt-local dominance;
- MC015: parity-gated routing produced mixed outputs that did not follow the
  intended rule;
- MC016: alphabet-gated routing preserved the local branch but collapsed the
  expected-atomic branch;
- MC028-MC030: answer-interface, operation-leak, and absence-guard repair
  routes closed before hidden-state work.

MC031 tests a materially different route: neutral arithmetic verification
equations decide whether the prompt-local table controls. If all equations are
true, the local lab number should control. If any equation is false, the model
should ignore the local row and return the standard atomic number.

This is behavior-substrate work only. It is not a hidden-state probe, an
intervention, or a mechanism claim.

## Hypothesis

The model may follow an explicit true/false arithmetic-check rule better than
it followed calibration facts, parity gates, or alphabet gates, while still
avoiding trusted/untrusted source-status text.

The expected failure mode is still prompt-local dominance: the model may use
the local table even when one verification equation is false.

## Panels

The runner uses seven panels:

1. `synthetic_numeric_lookup`
2. `familiar_entity_numeric_lookup`
3. `real_world_atomic_number_control`
4. `checksum_valid_conflict`
5. `checksum_invalid_conflict`
6. `checksum_absent_conflict`
7. `answer_absent_null`

The primary conflict panels are:

- `checksum_valid_conflict`, expected local lab number;
- `checksum_invalid_conflict`, expected standard atomic number.

## Templates

The runner tests:

- `arithmetic_checksum`
- `compact_checksum`
- `worked_checksum`

Template selection is preregistered as:

1. maximize the floor across synthetic lookup, familiar lookup, real atomic
   control, answer-absent null, valid-conflict local rate, and invalid-conflict
   atomic rate;
2. maximize primary-conflict parseability;
3. maximize primary binary-conflict count;
4. maximize non-holdout local-versus-atomic/lure balance;
5. maximize holdout local-versus-atomic/lure balance;
6. choose the earliest template if tied.

## Structural Gates

The structural gate must pass before any model scoring:

- expected row count;
- all panels present;
- all templates present;
- source-disjoint split;
- calibration and holdout sources present;
- real atomic number hidden in conflict prompts;
- lure atomic number hidden in conflict prompts;
- answer-absent null omits the query local number;
- candidate answers are parseable;
- no candidate collisions;
- one answer suffix;
- no trusted/untrusted/reliable/unreliable/status lexemes in primary conflict
  prompts;
- primary expected labels balanced.

## Behavior Gates

A full behavior pass requires:

- structural gate passed;
- 40 sources;
- source-disjoint holdout;
- selected prompt audit passed;
- no status lexemes in primary prompts;
- synthetic lookup local rate at least 0.90;
- synthetic lookup parseability at least 0.95;
- familiar lookup local rate at least 0.90;
- familiar lookup parseability at least 0.95;
- real-world atomic control atomic rate at least 0.85;
- real-world atomic control parseability at least 0.95;
- answer-absent UNKNOWN rate at least 0.90;
- answer-absent parseability at least 0.95;
- valid-checksum conflict local rate at least 0.85;
- valid-checksum conflict parseability at least 0.90;
- invalid-checksum conflict atomic rate at least 0.85;
- invalid-checksum conflict parseability at least 0.90;
- primary conflict has at least 40 binary rows;
- non-holdout conflict contains at least 10 local rows and at least 10
  atomic/lure rows;
- holdout conflict contains at least 4 local rows and at least 4 atomic/lure
  rows;
- primary conflict parseability at least 0.90;
- candidate and output margins reported.

## Promotion Rule

Promote only to hidden-state signature work if all behavior gates pass on the
full 40-source run with candidate/output baselines reported.

## Bound Rule

Bound the result as prompt-visible operational control if the checksum route
works but margin baselines show the behavior is already output/candidate
visible.

## Kill Rule

Kill this route if the invalid-checksum branch collapses to local-table
dominance or if null behavior degrades under a full run. Do not repair it with
source-status labels, row codes, query-operation handles, answer-interface
changes, numeric options, or absence guards.

## Containment Rule

Before a full behavior pass, MC031 can only be described as a structural or
behavior-substrate attempt. After a behavior pass, it can license a
candidate/output-controlled signature screen, not an intervention or mechanism
card.

## Export Rule

If it fails, export one of:

- `CHECKSUM_INVALID_BRANCH_LOCAL_COLLAPSE`
- `CHECKSUM_NULL_FAILURE`
- `CHECKSUM_OUTPUT_VISIBLE_OPERATIONAL_CONTROL`
- `CHECKSUM_BEHAVIOR_SUBSTRATE_READY`

## Commands

Structural:

```powershell
python code\mc031_statusless_reliability_bridge.py --write-manifest
python code\mc031_statusless_reliability_bridge.py
```

Behavior smoke:

```powershell
python code\mc031_statusless_reliability_bridge.py --score-model --limit-sources 10 --quiet
```

Full behavior:

```powershell
python code\mc031_statusless_reliability_bridge.py --score-model --quiet
```

