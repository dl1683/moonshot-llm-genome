# KSQ001 Familiar Entity Prior Counterbalance First Run

Status: behavior-substrate first-run preregistration; no hidden-state work.

Runner:

> `code/ksq001_familiar_entity_prior_counterbalance_first_run.py`

Default result:

> `results/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE/ksq001_familiar_entity_prior_counterbalance_first_run.json`

Default 10-source smoke result:

> `results/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE/ksq001_familiar_entity_prior_counterbalance_smoke_limit10.json`

Full behavior result:

> `results/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE/ksq001_familiar_entity_prior_counterbalance_full_behavior.json`

Status card:

> `research/cards/KSQ001_FAMILIAR_ENTITY_PRIOR_COUNTERBALANCE_FIRST_RUN_STATUS.md`

## Purpose

Test whether familiar country names produce measurable competition
between prompt-local artificial city values and learned capital priors
without explicit status labels or source-authority wording.

## Panels

- `source_local_artificial_lookup`
- `semantic_prior_direct_control`
- `semantic_prior_lure`
- `answer_absent_irrelevant_nulls`

## Decision Boundary

Promote only to behavior-substrate admission if local artificial
lookup, direct semantic-prior recall, answer-absent nulls, familiar
entity conflict mixture, source-disjoint holdout, prompt audit, and
candidate/output baseline reporting pass together.

Death rule: kill if behavior reduces to local copy, semantic prior
recall, authority wording, parse/answer-shape effects, or null
leakage.

Forbidden claims:

- KSQ001 is a mechanism card.
- KSQ001 licenses hidden-state search before the behavior gate passes.
- KSQ001 proves a learned-memory control surface.
