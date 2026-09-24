# MC032 Post-Checksum Bridge

Date: 2026-07-01

Status: preregistered structural bridge-substrate attempt.

Runner:

> `code/mc032_post_checksum_bridge.py`

## Purpose

MC032 implements the post-MC031 bridge work order:
`build_post_checksum_bridge_substrate`.

The bridge line has already closed source labels, status ablations,
calibration-inferred reliability, parity gates, alphabet gates, source-token
answers, row codes, query-operation handles, worked examples, answer-interface
sweeps, numeric options, absence guards, and arithmetic checksum validity cues.
MC031 was especially useful: it kept direct controls and nulls clean, but the
invalid-checksum branch still collapsed to local answers.

MC032 tests a materially different source-validity cue: cross-table
consistency. A primary local table and a second neutral table both list local
lab numbers. If the query element has the same number in both tables, the local
number should control. If the query element has different numbers across the
two tables, both local numbers should be ignored and the model should return
the standard atomic number from learned memory.

This is behavior-substrate work only. It is not a hidden-state probe, an
intervention, or a mechanism claim.

## Hypothesis

The model may handle two-table agreement better than checksum validity because
the decision is lexical and table-local rather than arithmetic.

The expected failure mode is still prompt-local dominance: mismatch rows may
select the primary local number, the second-table side number, or another
visible number instead of the learned atomic number.

## Panels

The runner uses seven panels:

1. `synthetic_numeric_lookup`
2. `familiar_entity_numeric_lookup`
3. `real_world_atomic_number_control`
4. `crosscheck_match_conflict`
5. `crosscheck_mismatch_conflict`
6. `crosscheck_absent_conflict`
7. `answer_absent_null`

Primary conflict panels:

- `crosscheck_match_conflict`, expected local lab number;
- `crosscheck_mismatch_conflict`, expected standard atomic number.

## Templates

The runner tests:

- `paired_tables`
- `mirror_registry`
- `dual_notebook`

These templates must avoid trusted/untrusted/reliable/status source labels and
must not reopen arithmetic checksum, equation, valid, or invalid wording.

## Structural Gates

The structural gate must pass before model scoring:

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
- no checksum/equation/arithmetic/valid/invalid route lexemes in primary
  conflict prompts;
- mismatch side number visible exactly once in mismatch conflict prompts;
- side number does not collide with local, atomic, lure, or UNKNOWN candidates;
- primary expected labels balanced.

## Behavior Gates

A future model-scored behavior pass must require:

- structural gate passed;
- 40 sources;
- source-disjoint holdout;
- selected prompt audit passed;
- no status lexemes or checksum-route lexemes in primary prompts;
- synthetic lookup local rate at least 0.90;
- familiar lookup local rate at least 0.90;
- real-world atomic control atomic rate at least 0.85;
- answer-absent UNKNOWN rate at least 0.90;
- match-conflict local rate at least 0.85;
- mismatch-conflict atomic rate at least 0.85;
- mismatch side-number rate below the predeclared side-effect threshold of 0.10;
- primary conflict parseability at least 0.90;
- source-disjoint non-holdout and holdout local/atomic branch coverage;
- candidate and output margins reported before hidden-state work.

## Promotion Rule

Promote only to hidden-state signature work if the full behavior run passes all
behavior gates with candidate/output baselines reported.

## Bound Rule

Bound if cross-table consistency works only as a prompt-visible operational
contract or if output/candidate margins explain the behavior.

## Kill Rule

Kill this route if mismatch rows collapse to the primary local number, the
second-table side number, or UNKNOWN while controls/nulls are otherwise clean.
Do not repair with source-status labels, row codes, operation handles, worked
examples, answer-interface changes, numeric options, absence guards, or
arithmetic checksum cues.

## Containment Rule

Before a full behavior pass, MC032 can only be described as a structural or
behavior-substrate attempt. After a behavior pass, it can license a
candidate/output-controlled signature screen, not an intervention or mechanism
card.

## Export Rule

If it fails, export one of:

- `POST_CHECKSUM_CROSSTABLE_LOCAL_COLLAPSE`
- `POST_CHECKSUM_CROSSTABLE_SIDE_NUMBER_LEAK`
- `POST_CHECKSUM_CROSSTABLE_NULL_FAILURE`
- `POST_CHECKSUM_CROSSTABLE_OUTPUT_VISIBLE_CONTROL`
- `POST_CHECKSUM_CROSSTABLE_BEHAVIOR_READY`

## Commands

Structural:

```powershell
python code\mc032_post_checksum_bridge.py
```

The first structural result must be written before any model smoke:

```powershell
python code\mc032_post_checksum_bridge.py --limit-sources 10
python code\mc032_post_checksum_bridge.py
```
