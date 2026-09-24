# MC001G Gemma 2 2B Matched Dense Intervention Preregistration

Date: 2026-06-30

Status: preregistered for first intervention gate.

## Header

- Card ID: `MC001G`
- Parent target: `MC-001` truth-versus-user-agreement conflict
- Model: `google/gemma-2-2b`
- Stage: intervention gate
- Runner: `code/mc001_gemma_intervention.py`
- Prior matched signature status: `research/cards/MC001G_GEMMA2_2B_PREHINT_MARGIN_DISCOVERY_STATUS.md`
- Repair result: `results/cards/MC001G/mc001g_gemma2_2b_repair_logit_raw_gemma_repair_20260630T121025.json`
- Result directory: `results/cards/MC001G/`

## Purpose

The pre-hint-margin matched discovery pass found a simple layer 14
truth-minus-agreement direction with holdout AUC 0.755. This pass asks whether
adding that direction changes forced-choice behavior in the predicted direction.

## Design

Direction:

- train only on matched discovery rows;
- layer: `14`;
- direction: truth-following mean minus user-agreement mean;
- evaluate on matched holdout rows and locality rows.

Evaluation rows:

- matched holdout `wrong_disclaimed` and `wrong_unsure` rows;
- `no_hint` rows for the same holdout items;
- `correct_hint` rows for the same holdout items.

Arms:

- baseline;
- layer 14 positive direction at alpha `0.25`, `0.5`, and `1.0`;
- layer 14 sign-flip at alpha `1.0`;
- layer 14 random matched-norm control at alpha `1.0`;
- wrong-layer control at layer `13`, alpha `1.0`.

## Promotion Rule

Promote only if:

- positive steering improves truth-following on matched holdout wrong-hint rows;
- the sign-flip moves in the opposite direction or at least does not copy the positive effect;
- random and wrong-layer controls are weaker than the positive effect;
- no-hint and correct-hint locality rows do not materially degrade;
- answer distribution does not collapse to one option.

If the intervention fails or controls match it, record the result as a
signature-only diagnostic and do not claim a mechanism.
