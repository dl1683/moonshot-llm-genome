# MC033 Fact-Claim Bridge Closeout

Date: 2026-07-01

Status: preregistered post-MC032 repair-or-close attempt.

Runner:

> `code/mc033_fact_claim_bridge_closeout.py`

## Purpose

MC033 implements the single repair pass allowed by the post-MC032 closure plan.

MC031 showed that arithmetic checksum validity preserved direct controls and
nulls but did not route invalid rows to learned atomic answers. MC032 removed
the checksum-specific objection by replacing arithmetic validity with
cross-table consistency; mismatch rows still avoided learned atomic answers and
mostly selected the primary local row, with no side-number copying.

MC033 tests a different source-validity cue: a local row may carry one
standard-number claim. The model must compare that claim against learned
atomic-number memory.

- If the query claim is the element's standard atomic number, return the local
  lab number.
- If the query claim is not the element's standard atomic number, ignore the
  local lab number and return the standard atomic number.

This is still behavior-substrate work. It is not a hidden-state probe, an
intervention, or a mechanism claim.

## What This Must Not Reopen

MC033 must not reopen:

- source-status labels;
- row codes;
- query-operation handles;
- worked examples;
- constrained choices;
- numeric options;
- answer-interface sweeps;
- absence guards;
- arithmetic checksum cues;
- simple cross-table consistency cues.

The only new cue is a row-local standard-number claim checked against learned
atomic memory.

## Panels

The runner uses seven panels:

1. `synthetic_numeric_lookup`
2. `familiar_entity_numeric_lookup`
3. `real_world_atomic_number_control`
4. `fact_claim_match_conflict`
5. `fact_claim_mismatch_conflict`
6. `fact_claim_absent_conflict`
7. `answer_absent_null`

Primary conflict panels:

- `fact_claim_match_conflict`, expected local lab number;
- `fact_claim_mismatch_conflict`, expected standard atomic number.

## Templates

The runner tests:

- `claim_column`
- `field_note`
- `memory_comparison`

These templates must avoid trusted/untrusted/reliable/status source labels,
checksum/equation/arithmetic/valid/invalid wording, and cross-table wording.

## Structural Gates

The structural gate must pass before model scoring:

- expected row count;
- all panels present;
- all templates present;
- source-disjoint split;
- calibration and holdout sources present;
- mismatch rows hide the real atomic target;
- mismatch rows show the wrong standard-number claim exactly once;
- answer-absent null omits the query local number;
- candidate answers are parseable;
- no candidate collisions;
- one answer suffix;
- no trusted/untrusted/reliable/unreliable/status lexemes in primary prompts;
- no checksum/equation/arithmetic/valid/invalid/cross-table route lexemes in
  primary prompts;
- primary expected labels balanced.

## Behavior Gates

A full model-scored behavior pass must require:

- structural gate passed;
- 40 sources;
- source-disjoint holdout;
- selected prompt audit passed;
- no status lexemes or closed-route lexemes in primary prompts;
- synthetic lookup local rate at least 0.90;
- familiar lookup local rate at least 0.90;
- real-world atomic control atomic rate at least 0.85;
- answer-absent UNKNOWN rate at least 0.90;
- fact-claim match local rate at least 0.85;
- fact-claim mismatch atomic rate at least 0.85;
- mismatch claimed-number/lure rate below 0.10;
- primary conflict parseability at least 0.90;
- source-disjoint non-holdout and holdout local/atomic branch coverage;
- candidate and output margins reported before hidden-state work.

## Promotion Rule

Promote only to hidden-state signature work if the full behavior run passes all
behavior gates with candidate/output baselines reported.

## Bound Rule

Bound if behavior works only as a prompt-visible fact-checking contract or if
output/candidate margins explain the behavior.

## Kill Rule

Kill the post-MC032 bridge route if mismatch rows collapse to the local lab
number, copy the claimed wrong number, become UNKNOWN, or trade learned-branch
accuracy against null cleanliness.

## Containment Rule

Before a full behavior pass, MC033 can only be described as a structural or
behavior-substrate attempt. After a behavior pass, it can license a
candidate/output-controlled signature screen, not an intervention or mechanism
card.

## Export Rule

If it fails, export one of:

- `FACT_CLAIM_MISMATCH_LOCAL_COLLAPSE`
- `FACT_CLAIM_CLAIMED_NUMBER_LEAK`
- `FACT_CLAIM_NULL_FAILURE`
- `FACT_CLAIM_OUTPUT_VISIBLE_CONTROL`
- `POST_MC032_BRIDGE_ROUTE_CLOSED`

## Commands

Structural:

```powershell
python code\mc033_fact_claim_bridge_closeout.py --limit-sources 10 --artifact-prefix mc033_fact_claim_bridge_structural_smoke
python code\mc033_fact_claim_bridge_closeout.py
```

Model smoke:

```powershell
python code\mc033_fact_claim_bridge_closeout.py --score-model --limit-sources 10 --quiet --artifact-prefix mc033_fact_claim_bridge_behavior_smoke
```
