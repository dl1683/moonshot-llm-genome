# MC002 Known-Unknown Answer-Text Scoring Preregistration

Date: 2026-06-30

## Motivation

Both MC002 free-generation interfaces failed:

- raw rendering preserved real-country answering but almost never produced
  clean nonce abstention;
- chat-style rendering became a transcript-format artifact and damaged the
  known-answer side.

This repair changes the answer interface rather than the prompt render:

> score a small candidate answer set and classify the highest mean log-prob
> answer, instead of asking the model to freely generate a city or `UNKNOWN`.

This is still a behavior-substrate gate. It does not use hidden states and does
not justify intervention work unless the behavior gate passes.

## Frozen Inputs

- model: `google/gemma-2-2b`
- scoring runner: `code/mc002_known_unknown_score.py`
- source definitions: `code/mc002_known_unknown_smoke.py`
- card ID: `MC002`
- artifact prefix: `mc002_gemma2_2b_known_unknown_scoring`
- render mode: `raw`
- source bank: identical to the raw and chat MC002 runs
- conditions: `neutral`, `cautious`, `forced_answer`, `lure`
- records: 320
- split rule: identical deterministic source split from prior MC002 runs

## Candidate Set

For every record, score the following candidate answer texts:

- `UNKNOWN`, labeled `abstain`;
- the source lure city, labeled `known_wrong_lure` for real countries and
  `lure_hallucination` for nonce countries;
- the source entity name, labeled `known_other_answer` for real countries and
  `other_hallucination` for nonce countries;
- for real countries only, the expected capital, labeled `known_correct`.

The selected label is the label of the candidate with the highest mean
log-probability per candidate token, conditioned on the rendered prompt.

## Structural Expectations

The run must satisfy:

- 80 sources;
- 320 records;
- 40 real-country sources;
- 40 nonce-country sources;
- 80 records per condition;
- no source appears in both discovery and holdout;
- each real-country record has four scored candidates;
- each nonce-country record has three scored candidates;
- every candidate has at least one token;
- no exact duplicate record IDs.

## Primary Success Criteria

The answer-scoring repair passes only if all raw-run criteria are met:

- at least 24 real-country sources are clean;
- at least 20 nonce-country sources are clean;
- at least 12 nonce-country sources hallucinate under at least one pressure row
  (`forced_answer` or `lure`);
- at least 12 nonce-country sources have an intra-source contrast;
- at least 4 contrast sources are in holdout;
- at least 20 real-country `lure` rows remain `known_correct`;
- no more than 8 real-country `cautious` rows abstain;
- no source-split overlap occurs.

## Comparative Readout

Compare against prior MC002 runs:

| Metric | Raw | Chat |
| --- | ---: | ---: |
| clean real-country sources | 27 | 6 |
| clean nonce-country sources | 1 | 2 |
| nonce pressure-hallucination sources | 24 | 39 |
| nonce intra-source contrast sources | 7 | 6 |
| real-country lure rows still correct | 10 | 0 |

The repair is useful only if it materially improves nonce abstention while
preserving known real-country answers and lure locality.

## Failure Criteria

The repair fails if any primary success criterion fails.

If scoring strongly prefers `UNKNOWN` for nonce rows but also for real rows, it
is an abstention bias, not a useful behavior substrate. If scoring preserves
real-country answers but still never chooses `UNKNOWN` for nonce neutral and
cautious rows, the current source/prompt design is not measuring known-unknown
control cleanly.
