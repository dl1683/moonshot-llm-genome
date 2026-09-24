# KSQ006 Context-Support Counterfactuals First Run

Status: behavior-substrate first-run preregistration; no hidden-state work.

Runner:

> `code/ksq006_uncertainty_context_support_counterfactuals_first_run.py`

Default result:

> `results/cards/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS/ksq006_uncertainty_context_support_counterfactuals_first_run.json`

Default 10-source smoke result:

> `results/cards/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS/ksq006_uncertainty_context_support_counterfactuals_smoke_limit10.json`

Full behavior result:

> `results/cards/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS/ksq006_uncertainty_context_support_counterfactuals_full_behavior.json`

Status card:

> `research/cards/KSQ006_UNCERTAINTY_CONTEXT_SUPPORT_COUNTERFACTUALS_FIRST_RUN_STATUS.md`

## Purpose

Test whether context support can be made counterfactual before
testing factual uncertainty, refusal, or correction signatures. The
run separates supported, irrelevant, contradicting, insufficient,
claim-only, and context-only branches under a shared answer schema.

## Panels

- `supported_context_rows`
- `irrelevant_context_rows`
- `contradicting_context_rows`
- `insufficient_context_rows`
- `claim_only_and_context_only_controls`

## Decision Boundary

Promote only to behavior-substrate admission if supported rows
answer, irrelevant rows abstain, contradicting rows detect
disagreement, insufficient rows abstain without support-word
leakage, claim-only/context-only controls do not reproduce
supported behavior, source-disjoint holdout passes, and
candidate/output baseline reporting is present.

Death rule: kill if support language, answer shape, claim-only
text, context-only city mention, caution wording, or output
margin explains the behavior.

Forbidden claims:

- KSQ006 is a mechanism card.
- KSQ006 licenses hidden-state search before the behavior gate passes.
- KSQ006 proves a factual uncertainty, refusal, context-support, or correction control surface.
