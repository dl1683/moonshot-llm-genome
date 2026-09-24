# MC001G Gemma 2 2B Residualized Dense Discovery Preregistration

Date: 2026-06-30

Status: preregistered for residualized signature discovery.

## Header

- Card ID: `MC001G`
- Parent target: `MC-001` truth-versus-user-agreement conflict
- Model: `google/gemma-2-2b`
- Stage: residualized signature discovery, no intervention claim allowed
- Runner: `code/mc001_gemma_residual_discovery.py`
- Prior dense status: `research/cards/MC001G_GEMMA2_2B_DISCOVERY_STATUS.md`
- Repair result: `results/cards/MC001G/mc001g_gemma2_2b_repair_logit_raw_gemma_repair_20260630T121025.json`
- Result directory: `results/cards/MC001G/`

## Purpose

The first dense discovery pass found real hidden-state signal, but the
correct-minus-wrong option-margin baseline reached holdout AUC 1.000. This pass
asks whether any dense final-token signal survives after obvious nuisance
variables are regressed out.

## Primary Rows

Use the same primary rows as dense discovery:

- clean repaired base-Gemma items only;
- `wrong_disclaimed` and `wrong_unsure` conditions only;
- truth-following versus user-agreement rows only;
- deterministic item split inherited from the dense discovery pass.

## Nuisance Variables

Fit nuisance regression on discovery rows only, then apply it to discovery and
holdout hidden states.

Nuisance features:

- intercept;
- standardized correct-minus-wrong option logit margin;
- prompt condition;
- correct answer letter;
- wrong answer letter;
- baseline answer letter.

The residualized hidden states are the original final-token residual vectors
minus the discovery-fit linear nuisance prediction.

## Promotion Rule

Promote to intervention design only if:

- residualized hidden direction or residualized hidden logistic probe has strong holdout AUC;
- the result is meaningfully above nuisance-feature baselines;
- the best residualized layer is not just a high-dimensional overfit artifact;
- the status card reports both raw and residualized metrics.

If residualization collapses the signature or leaves only weak/noisy signal, do
not intervene. Move to margin-matched rows or sparse-feature discovery with the
same controls.
