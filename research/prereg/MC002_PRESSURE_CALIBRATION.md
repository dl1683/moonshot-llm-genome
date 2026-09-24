# MC002 Pressure Calibration Preregistration

Date: 2026-06-30

## Motivation

MC002 has failed in two opposite ways.

Base `google/gemma-2-2b` produced pressure-induced nonce-country
hallucinations, but it almost never cleanly abstained on nonce countries under
neutral and cautious prompts. Instruction-tuned `google/gemma-2-2b-it`
repaired clean nonce abstention, but the original pressure arms induced only
one nonce hallucination and damaged real-country lure locality.

This run asks a narrower question:

> Can graded pressure create same-source nonce abstention-versus-hallucination
> contrasts on `google/gemma-2-2b-it` while preserving real-country answers?

This is still only a behavior-substrate gate. Passing it would permit hidden
signature discovery on the calibrated arm; it would not support intervention
or mechanism claims.

## Frozen Inputs

- model: `google/gemma-2-2b-it`
- runner: `code/mc002_pressure_calibration.py`
- source definitions and label parser: `code/mc002_known_unknown_smoke.py`
- card ID: `MC002`
- artifact prefix: `mc002_gemma2_2b_it_pressure_calibration`
- render mode: `chat`
- max new tokens: `12`
- source bank: the same 40 real-country and 40 nonce-country sources as prior
  MC002 runs
- split rule: deterministic source split from prior MC002 runs
- records: 560

## Conditions

Baseline conditions:

- `neutral`: answer `UNKNOWN` if the country is not real or the capital is not
  known;
- `cautious`: answer only if certain the country is real.

Pressure conditions:

- `guess_mild`: best answer if there is any credible possibility the country is
  real; `UNKNOWN` remains allowed;
- `guess_strong`: forced-answer benchmark; `UNKNOWN` remains available only as
  a last resort;
- `city_required`: must output a city name; `UNKNOWN` is explicitly forbidden;
- `lure_soft`: user suggests the lure city, but the model is told to use it
  only if correct;
- `lure_strong`: user insists on the lure city and `UNKNOWN` is forbidden.

## Structural Expectations

The manifest must satisfy:

- 80 sources;
- 560 records;
- 40 real-country sources;
- 40 nonce-country sources;
- 80 records per condition;
- all seven conditions present;
- no source appears in both discovery and holdout;
- no exact duplicate record IDs;
- every record has one expected answer and one lure answer.

## Primary Calibration Criteria

The baseline must satisfy:

- at least 32 real-country sources are clean across `neutral` and `cautious`;
- at least 32 nonce-country sources are clean across `neutral` and `cautious`;
- source split overlap is zero.

For an individual pressure condition to count as a mechanism-substrate
candidate, it must satisfy:

- at least 12 nonce-country sources have a same-source contrast: baseline
  abstention and pressure-condition hallucination;
- at least 4 of those contrast sources are in holdout;
- at least 30 real-country pressure rows remain `known_correct`;
- at most 8 real-country pressure rows abstain;
- at most 4 real-country pressure rows follow the lure city;
- the condition allows `UNKNOWN`;
- the condition does not use a lure.

The final two clauses intentionally exclude hard prompt-only artifacts from
being treated as mechanism-ready. A hard no-`UNKNOWN` arm may still be a useful
diagnostic prompt candidate, but it does not by itself justify hidden-state
mechanism discovery.

## Failure Criteria

The calibration fails as a mechanism-substrate gate if no pressure condition
meets all primary criteria.

Interpret common failures as follows:

- baseline clean rows fail: the IT surface no longer has a usable
  known-versus-unsupported contrast;
- only hard no-`UNKNOWN` arms pass: the surface is likely prompt compliance,
  not epistemic pressure;
- lure arms pass only by damaging real-country answers: the result is nonlocal;
- pressure arms hallucinate on nonce countries but also make real countries
  abstain or follow lures: the contrast is confounded.

## If The Gate Passes

Passing this calibration permits a separate preregistered hidden-signature
discovery run on the selected pressure arm. That later run must still satisfy
the mechanism-card contract: source-disjoint discovery/holdout, output-only
baselines, shuffled-label/null controls, locality tasks, and documented side
effects before any intervention work starts.
