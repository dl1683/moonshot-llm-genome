# MC002 Known-Unknown Gemma 2 2B IT Chat Preregistration

Date: 2026-06-30

## Motivation

Base `google/gemma-2-2b` failed MC002 under three interfaces:

- raw generation;
- chat-style rendering on the base model;
- answer-text scoring.

The best base-model interface, answer scoring, preserved real-country knowledge
but still preferred nonce entity names or lure cities over `UNKNOWN` on most
unsupported entities. The next question is whether this is a base-model
instruction-following failure rather than a fundamental task failure.

This comparison changes one major variable:

> use instruction-tuned `google/gemma-2-2b-it` with its chat template, while
> keeping the MC002 sources, labels, source split, and success criteria fixed.

## Frozen Inputs

- model: `google/gemma-2-2b-it`
- generation runner: `code/mc002_known_unknown_smoke.py`
- card ID: `MC002`
- artifact prefix: `mc002_gemma2_2b_it_known_unknown_chat`
- render mode: `chat`
- max new tokens: `12`
- source bank: identical to prior MC002 runs
- conditions: `neutral`, `cautious`, `forced_answer`, `lure`
- records: 320
- split rule: identical deterministic source split from prior MC002 runs
- label rules: identical to `code/mc002_known_unknown_smoke.py`

## Structural Expectations

The manifest must satisfy:

- 80 sources;
- 320 records;
- 40 real-country sources;
- 40 nonce-country sources;
- 80 records per condition;
- no source appears in both discovery and holdout;
- no exact duplicate record IDs;
- every record has one expected answer and one lure answer.

## Primary Success Criteria

The instruction-tuned comparison passes only if the original MC002 behavior
criteria are met:

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

Compare against the three base-Gemma interfaces:

| Metric | Raw | Base Chat | Base Scoring |
| --- | ---: | ---: | ---: |
| clean real-country sources | 27 | 6 | 26 |
| clean nonce-country sources | 1 | 2 | 1 |
| nonce pressure-hallucination sources | 24 | 39 | 40 |
| nonce intra-source contrast sources | 7 | 6 | 9 |
| real-country lure rows still correct | 10 | 0 | 15 |

The comparison is useful only if instruction tuning materially improves nonce
abstention while preserving real-country answering and real-country lure
locality.

## Failure Criteria

The comparison fails if any primary success criterion fails.

If it fails by making all nonce rows abstain and pressure rows never
hallucinate, the behavior lacks an intervention target. If it fails by
preserving hallucination under neutral/cautious rows, the task still lacks a
clean negative class. If it fails by damaging real-country lure locality, the
pressure arm remains too confounded for hidden-state intervention work.

## If The Gate Passes

Passing this behavior gate permits hidden-state signature discovery on
instruction-tuned Gemma. It still does not permit steering, patching, sparse
feature suppression, or any mechanism claim.
