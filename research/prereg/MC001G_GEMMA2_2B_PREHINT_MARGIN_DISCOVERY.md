# MC001G Gemma 2 2B Pre-Hint Margin Matched Discovery Preregistration

Date: 2026-06-30

Status: preregistered for non-tautological margin-matched discovery.

## Header

- Card ID: `MC001G`
- Parent target: `MC-001` truth-versus-user-agreement conflict
- Model: `google/gemma-2-2b`
- Stage: pre-hint margin matched signature discovery, no intervention claim allowed
- Runner: `code/mc001_gemma_prehint_margin_discovery.py`
- Prior residualized status: `research/cards/MC001G_GEMMA2_2B_RESIDUALIZED_DISCOVERY_STATUS.md`
- Repair result: `results/cards/MC001G/mc001g_gemma2_2b_repair_logit_raw_gemma_repair_20260630T121025.json`
- Result directory: `results/cards/MC001G/`

## Correction To The Prior Next Step

Same-row correct-minus-wrong option margin is tautologically tied to forced-choice
labels. If the argmax is the correct option, the correct-minus-wrong margin is
nonnegative. If the argmax is the hinted wrong option, the margin is nonpositive.

Therefore, a large same-row option-margin overlap set cannot be produced under
the current forced-choice label definition except through exact ties or artificial
wide bins.

This pass uses item-level pre-hint margins instead:

- no-hint correct-minus-wrong margin;
- correct-hint correct-minus-wrong margin.

These margins are measured before the wrong-hint condition and are not
tautologically determined by the later wrong-hint label.

## Design

Use the repaired base-Gemma result:

- clean items only;
- `wrong_disclaimed` and `wrong_unsure` rows only;
- truth-following versus user-agreement rows only;
- existing item split from the dense discovery pass;
- match truth and agreement rows within each split and no-hint-margin bin;
- default no-hint-margin bin width: `0.5`.

The run must report:

- matched row counts by split and bin;
- same-row margin AUC as a tautological control;
- no-hint margin AUC;
- correct-hint margin AUC;
- prompt and answer-letter baselines;
- dense final-token direction/probe metrics.

## Promotion Rule

Promote only if:

- discovery and holdout both contain matched truth/agreement rows in shared pre-hint-margin bins;
- dense hidden signatures generalize on the matched holdout set;
- pre-hint margin and answer-letter baselines do not explain the signal;
- the status explicitly reports same-row option margin as a tautological control.

Do not promote if the same-row margin is the only strong signal, or if hidden
signals fail on matched holdout.
