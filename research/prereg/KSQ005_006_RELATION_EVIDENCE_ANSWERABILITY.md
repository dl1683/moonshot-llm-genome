# KSQ005/KSQ006 Relation-Evidence Answerability Redesign

Status: material second-wave redesign; no hidden-state work.

Runner:

> `code/ksq005_006_relation_evidence_answerability_redesign.py`

Artifacts:

> `results/cards/KSQ005_006_RELATION_EVIDENCE_ANSWERABILITY/ksq005_006_relation_evidence_answerability_first_run.json`

> `results/cards/KSQ005_006_RELATION_EVIDENCE_ANSWERABILITY/ksq005_006_relation_evidence_answerability_smoke_limit10.json`

> `results/cards/KSQ005_006_RELATION_EVIDENCE_ANSWERABILITY/ksq005_006_relation_evidence_answerability_full_behavior.json`

## Claim Under Test

Real answerability may require a visible relation-evidence grammar before
unknown, unsupported, contradiction, claim-only, and mention-only controls
pass together. The tested contract is exact `capital_of(entity)=city`
relations. This is a behavior contract, not an internal uncertainty claim.

## Promotion Rule

Admit only if known factual direct answers, relation-supported answers,
unknown nonce abstention, unsupported relation abstention, contradictory
relation abstention, claim-only controls, mention-only controls, source-
disjoint holdout, prompt audit, parser, and candidate/output baselines all
pass together.

## Kill Rule

Kill the current uncertainty route if the redesign fixes supported or
correction-looking rows while still failing unknown, unsupported,
claim-only, or mention-only controls.

## Forbidden Claims

- This redesign is not a mechanism card.
- This redesign does not prove uncertainty, refusal, or correction control.
- Hidden-state probing remains forbidden until behavior admission passes.
