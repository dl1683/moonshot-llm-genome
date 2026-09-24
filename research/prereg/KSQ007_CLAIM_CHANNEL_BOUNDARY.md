# KSQ007B Claim-Channel Boundary Audit

Status: KSQ007 diagnostic follow-up.

Runner:

> `code/ksq007_claim_channel_boundary_audit.py`

Artifacts:

> `results/cards/KSQ007_CLAIM_CHANNEL_BOUNDARY/ksq007_claim_channel_boundary_first_run.json`

> `results/cards/KSQ007_CLAIM_CHANNEL_BOUNDARY/ksq007_claim_channel_boundary_smoke_limit10.json`

> `results/cards/KSQ007_CLAIM_CHANNEL_BOUNDARY/ksq007_claim_channel_boundary_full_behavior.json`

## Claim Under Test

KSQ007 failed because claim-only rows reproduced nonce values. This
audit tests whether that failure is localized to exact claim syntax,
bare answer_for syntax, NOT_EVIDENCE or quoted EVIDENCE strings,
evidence-looking rows placed outside the counted block, prose claims,
wrong-predicate claims, or mere mentions.

## Promotion Rule

Promote only to claim-boundary calibrator status if exact EVIDENCE rows
answer, every control panel abstains, prompt audit passes, source-
disjoint holdout passes, and candidate/output margins are reported.
A pass still licenses no hidden-state or intervention work.

## Kill / Boundary Rule

If exact EVIDENCE rows answer but controls fail, export the narrowest
typed failure: section-boundary evidence-prefix leakage, answer_for
syntax leakage, semantic claim leakage, or baseline answerability
failure. Treat the typed failure as the datum.

## Forbidden Claims

- This is not a mechanism card.
- This is not real uncertainty, refusal, or factual correction.
- Hidden-state probing remains forbidden from this result alone.
