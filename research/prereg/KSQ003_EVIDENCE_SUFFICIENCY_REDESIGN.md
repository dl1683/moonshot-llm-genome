# KSQ003 Evidence Sufficiency Redesign

Status: material redesign preregistration; no hidden-state work.

Runner:

> `code/ksq003_evidence_sufficiency_redesign.py`

Default artifacts:

> `results/cards/KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN/ksq003_evidence_sufficiency_redesign_first_run.json`

> `results/cards/KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN/ksq003_evidence_sufficiency_redesign_smoke_limit10.json`

> `results/cards/KSQ003_EVIDENCE_SUFFICIENCY_REDESIGN/ksq003_evidence_sufficiency_redesign_full_behavior.json`

## Claim Under Test

A statusless evidence-sufficiency contract can prevent local-table
dominance without trusted/untrusted labels: complete identity evidence
routes to the learned atomic-number branch, contradictory or incomplete
evidence routes to UNKNOWN, and direct local lookup remains clean.

## Promotion Rule

Promote only to behavior substrate if direct local lookup, direct learned
atomic recall, complete-identity conflict, contradiction nulls,
single-feature ablations, answer-absent nulls, source-disjoint holdout,
and candidate/output baseline reporting all pass.

## Kill Rule

Kill same-family statusless evidence aggregation if complete conflict,
contradiction, or single-feature ablation rows again collapse to local
lab-number answers in the full run.

## Forbidden Claims

- This preregistration does not license hidden-state probing.
- This redesign is not a mechanism card.
- A passing behavior substrate would still require signature and intervention gates.
