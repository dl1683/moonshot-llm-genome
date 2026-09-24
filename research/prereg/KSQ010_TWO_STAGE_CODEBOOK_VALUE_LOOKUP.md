# KSQ010 Two-Stage Codebook Value Lookup

Status: KSQ009 diagnostic follow-up.

Runner:

> `code/ksq010_two_stage_codebook_value_lookup.py`

Artifacts:

> `results/cards/KSQ010_TWO_STAGE_CODEBOOK_VALUE_LOOKUP/ksq010_two_stage_codebook_value_lookup_first_run.json`

> `results/cards/KSQ010_TWO_STAGE_CODEBOOK_VALUE_LOOKUP/ksq010_two_stage_codebook_value_lookup_smoke_limit10.json`

> `results/cards/KSQ010_TWO_STAGE_CODEBOOK_VALUE_LOOKUP/ksq010_two_stage_codebook_value_lookup_full_behavior.json`

## Claim Under Test

KSQ009 failed because schema-specific ALLOW rows remained weaker than
answer-bearing answer_for syntax. This packet separates the task into
entity-code extraction and code-value lookup so the value is not directly
attached to the entity in counted rows.

## Promotion Rule

Promote only to behavior-substrate candidate status if exact two-stage
bridges answer, two-stage bridges beat an uncounted answer_for alternate,
every abstention/control panel passes, selected prompt audit passes,
source-disjoint holdout passes, and margins are reported. Hidden-state
claims remain forbidden.

## Kill / Boundary Rule

If exact two-stage bridges do not answer, export a positive-parseability
or multi-step-following failure. If answer_for rows reproduce values or
override the bridge, export answer_for competition. If uncounted or quoted
codebook rows reproduce values, export locality failure. Treat the typed
failure as the datum.

## Forbidden Claims

- This is not a mechanism card.
- This is not real uncertainty, refusal, or factual correction.
- Hidden-state probing remains forbidden unless a later full behavior
  run and margin report admit only a signature screen.
