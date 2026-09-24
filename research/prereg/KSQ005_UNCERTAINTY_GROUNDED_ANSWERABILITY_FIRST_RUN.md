# KSQ005 Grounded Answerability First Run

Status: behavior-substrate first-run preregistration; no hidden-state work.

Runner:

> `code/ksq005_uncertainty_grounded_answerability_first_run.py`

Default result:

> `results/cards/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY/ksq005_uncertainty_grounded_answerability_first_run.json`

Default 10-source smoke result:

> `results/cards/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY/ksq005_uncertainty_grounded_answerability_smoke_limit10.json`

Full behavior result:

> `results/cards/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY/ksq005_uncertainty_grounded_answerability_full_behavior.json`

Status card:

> `research/cards/KSQ005_UNCERTAINTY_GROUNDED_ANSWERABILITY_FIRST_RUN_STATUS.md`

## Purpose

Build a grounded answerability substrate before testing factual
uncertainty, refusal, or correction signatures. The run separates
real answerable facts, nonce unknowns, unsupported context rows, and
contradicted context rows under the same answer schema.

## Panels

- `known_factual_direct`
- `unknown_nonce_rows`
- `unsupported_context_rows`
- `contradicted_context_rows`

## Decision Boundary

Promote only to behavior-substrate admission if known factual
answers, unknown nonce abstention, unsupported-context abstention,
contradicted-context correction-or-abstention, source-disjoint
holdout, prompt audit, shared requested-mode suffix, and
candidate/output baseline reporting pass together.

Death rule: kill if abstention follows answer schema, unsupported
context cities, caution wording, entity familiarity, or output
margin instead of grounded answerability.

Forbidden claims:

- KSQ005 is a mechanism card.
- KSQ005 licenses hidden-state search before the behavior gate passes.
- KSQ005 proves a factual uncertainty, refusal, or correction control surface.
