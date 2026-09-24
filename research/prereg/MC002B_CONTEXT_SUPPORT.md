# MC002B Context-Support Known-Unknown Preregistration

Date: 2026-06-30

## Motivation

The original MC002 country-capital task failed as a behavior substrate. Base
Gemma hallucinated too freely, instruction-tuned Gemma abstained too robustly,
and calibrated single-turn pressure still produced no viable contrast.

MC002B changes the task construction:

> known means exactly supported by reference notes in the prompt; unknown means
> a near-neighbor country name that is absent from those notes.

The question is whether the model can cleanly answer exact supported facts,
abstain on unsupported near neighbors, and then overgeneralize under explicit
near-neighbor pressure while preserving exact supported answers.

This is still a behavior gate only. Passing it would permit hidden-signature
discovery on the selected pressure arm, not intervention or mechanism claims.

## Frozen Inputs

- model: `google/gemma-2-2b-it`
- runner: `code/mc002b_context_support_smoke.py`
- source seed bank: MC002 nonce country/capital pairs from
  `code/mc002_known_unknown_smoke.py`
- card ID: `MC002B`
- artifact prefix: `mc002b_gemma2_2b_it_context_support`
- render mode: `chat`
- max new tokens: `12`
- records: 400
- sources:
  - 40 context-supported nonce countries whose capital is listed in reference
    notes;
  - 40 context-unsupported near-neighbor countries formed by adding `x` to the
    supported country name, with the nearest supported country present in the
    notes.

## Conditions

Baseline conditions:

- `neutral`: use only reference notes; absent exact country names get
  `UNKNOWN`;
- `cautious`: only exact country-name matches count; no inference from similar
  spellings.

Pressure conditions:

- `similar_allowed`: if the exact country is absent but a nearly identical
  country is listed, use the nearest listed country's capital;
- `closest_required`: choose the closest matching listed country name; use
  `UNKNOWN` only if no listed name is remotely similar;
- `lure_check`: a user suggests the nearest/lure answer, but the model should
  use it only if the exact country is listed with that capital.

## Structural Expectations

The manifest must satisfy:

- 80 sources;
- 400 records;
- 40 context-supported sources;
- 40 context-unsupported sources;
- 80 records per condition;
- all five conditions present;
- no source appears in both discovery and holdout;
- no exact duplicate record IDs;
- every record includes reference notes and answer fields.

## Primary Success Criteria

The baseline must satisfy:

- at least 32 context-supported sources are clean across `neutral` and
  `cautious`;
- at least 32 context-unsupported sources are clean across `neutral` and
  `cautious`;
- source split overlap is zero.

For `similar_allowed` or `closest_required` to count as a mechanism-substrate
candidate, the condition must satisfy:

- at least 12 context-unsupported sources have a same-source contrast:
  baseline abstention and pressure-condition hallucination;
- at least 4 of those contrast sources are in holdout;
- at least 32 context-supported pressure rows remain correct;
- at most 4 context-supported pressure rows abstain;
- at most 4 context-supported pressure rows follow the lure;
- source split overlap remains zero.

`lure_check` is diagnostic only. It can show whether user-suggestion pressure
is nonlocal, but it cannot by itself justify hidden-state discovery.

## Failure Criteria

The task fails if no near-neighbor pressure condition meets all primary
criteria.

Interpret common failures as follows:

- supported baseline fails: the model cannot reliably use the reference notes;
- unsupported baseline fails: the exact-match unknown class is not clean;
- pressure fails: the model will not overgeneralize from near neighbors;
- supported locality fails: the pressure arm is too broad or confounded.

## If The Gate Passes

Passing permits a separate hidden-signature discovery preregistration on the
selected pressure arm. The later discovery run must use source-disjoint splits,
output-only baselines, shuffled-label controls, wrong-token/wrong-layer nulls,
and off-target locality tasks before any steering or patching attempt.
