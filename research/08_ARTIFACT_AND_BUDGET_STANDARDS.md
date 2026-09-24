# Artifact And Budget Standards

This file defines future experiment hygiene. It is not code.

## Artifact Principle

Every future run should leave enough evidence for a skeptical reviewer to reconstruct:

- what was claimed;
- what was tested;
- what was held out;
- what controls were run;
- what failed;
- why the verdict follows.

## Required Future Artifact Tree

Recommended future shape:

- `research/prereg/` for preregistrations;
- `research/cards/` for final mechanism cards;
- `research/reviews/` for adversarial reviews;
- `data/cards/<card_id>/` for prompt manifests and splits;
- `results/cards/<card_id>/` for generated results;
- `artifacts/cards/<card_id>/` for plots and human-readable summaries.

The exact tree can change when implementation starts. The rule cannot: prereg, raw results, summaries, and review must be separable.

## Required Metadata

Every future result should record:

- date;
- card ID;
- git commit;
- model ID;
- tokenizer ID;
- interpretability artifact IDs;
- prompt manifest hash;
- generation settings;
- intervention settings;
- random seeds;
- hardware if relevant;
- runtime and cost if relevant.

## Budget Gates

Gate 0: no-code doctrine.

- Cost: none.
- Exit: selected target and preregistration.

Gate 1: tiny smoke.

- Purpose: verify prompt labels, parsing, and activation access.
- No scientific claim allowed.
- Exit: data pipeline and instrumentation are not obviously broken.

Gate 2: discovery run.

- Purpose: identify candidate signatures on discovery split.
- Claim allowed: candidate signatures only.
- Exit: at least one signature beats baselines on calibration data.

Gate 3: intervention calibration.

- Purpose: tune intervention strength without touching holdout.
- Claim allowed: preregistered intervention is viable enough to test.
- Exit: nulls and side-effect metrics are ready.

Gate 4: holdout card run.

- Purpose: final verdict.
- Claim allowed: mechanism card verdict.
- Exit: card is written and adversarially reviewed.

## Spend Discipline

Do not spend major compute on:

- a target without a preregistration;
- a signature that does not beat simple baselines;
- an intervention that has not passed nulls;
- a method that only improves examples selected after looking at results;
- broad cross-model generalization before one model works.

## Result Labels

Allowed labels:

- `success`
- `failed_controls`
- `diagnostic_only`
- `control_without_explanation`
- `artifact`
- `inconclusive`
- `retired`

Do not label a result "promising" without saying which gate it passed.

## Review Requirements

Every card gets two reviews:

1. correctness review: did the experiment do what it said?
2. direction review: should the project spend more on this line?

The adversarial review should be allowed to kill the line.
