# KSQ009 Schema-Specific Value Lookup

Status: KSQ008 diagnostic follow-up.

Runner:

> `code/ksq009_schema_specific_value_lookup.py`

Artifacts:

> `results/cards/KSQ009_SCHEMA_SPECIFIC_VALUE_LOOKUP/ksq009_schema_specific_value_lookup_first_run.json`

> `results/cards/KSQ009_SCHEMA_SPECIFIC_VALUE_LOOKUP/ksq009_schema_specific_value_lookup_smoke_limit10.json`

> `results/cards/KSQ009_SCHEMA_SPECIFIC_VALUE_LOOKUP/ksq009_schema_specific_value_lookup_full_behavior.json`

## Claim Under Test

KSQ008 failed because neutral positive evidence was weak and counted
old-schema answer_for rows still reproduced values. This packet tests
whether a stricter ALLOW-row value schema can separate positive lookup
from answer_for schema specificity and allowed-row locality.

## Promotion Rule

Promote only to behavior-substrate candidate status if exact ALLOW rows
answer, ALLOW rows beat an uncounted answer_for alternate, every
abstention/control panel passes, selected prompt audit passes, source-
disjoint holdout passes, and margins are reported. Hidden-state claims
remain forbidden.

## Kill / Boundary Rule

If exact ALLOW rows do not answer, export a positive-parseability
failure. If answer_for rows reproduce values, export schema-specificity
failure. If uncounted or quoted ALLOW rows reproduce values, export
allowed-row locality failure. Treat the typed failure as the datum.

## Forbidden Claims

- This is not a mechanism card.
- This is not real uncertainty, refusal, or factual correction.
- Hidden-state probing remains forbidden unless a later full behavior
  run and margin report admit only a signature screen.
