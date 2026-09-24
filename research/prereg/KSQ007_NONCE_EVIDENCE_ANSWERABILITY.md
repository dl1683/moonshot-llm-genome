# KSQ007 Nonce-Evidence Answerability Calibrator

Status: post-KSQ005/006 behavior-only calibration work order.

Runner:

> `code/ksq007_nonce_evidence_answerability_calibrator.py`

Artifacts:

> `results/cards/KSQ007_NONCE_EVIDENCE_ANSWERABILITY/ksq007_nonce_evidence_answerability_first_run.json`

> `results/cards/KSQ007_NONCE_EVIDENCE_ANSWERABILITY/ksq007_nonce_evidence_answerability_smoke_limit10.json`

> `results/cards/KSQ007_NONCE_EVIDENCE_ANSWERABILITY/ksq007_nonce_evidence_answerability_full_behavior.json`

## Claim Under Test

The KSQ005/006 relation-evidence failure may be caused by familiar
capital priors and city-value salience rather than by answerability
grammar itself. KSQ007 removes real-world priors with nonce entities
and nonce slot values while preserving exact evidence, absent evidence,
unrelated entity, contradiction, claim-only, mention-only, query-only,
holdout, prompt-audit, and candidate/output controls.

## Promotion Rule

Promote only to calibrator status if exact evidence rows answer, all
absent/unrelated/conflicting/control rows abstain, source-disjoint
holdout passes, prompt audit passes, and candidate/output margins are
reported. A pass still does not license hidden-state work.

## Kill Rule

Kill this nonce-evidence route if it cannot jointly preserve exact
evidence answering and abstention on unrelated, contradiction, claim,
mention, and query-only controls.

## Forbidden Claims

- This is not a mechanism card.
- This is not real uncertainty, refusal, or factual correction.
- Hidden-state probing remains forbidden from this result alone.
