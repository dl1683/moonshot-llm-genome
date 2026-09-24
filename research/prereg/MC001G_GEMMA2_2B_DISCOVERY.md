# MC001G Gemma 2 2B Dense Signature Discovery Preregistration

Date: 2026-06-30

Status: preregistered for Gate 1 signature discovery.

## Header

- Card ID: `MC001G`
- Parent target: `MC-001` truth-versus-user-agreement conflict
- Model: `google/gemma-2-2b`
- Stage: signature discovery, no intervention claim allowed
- Runner: `code/mc001_gemma_repair_discovery.py`
- Prior status: `research/cards/MC001G_GEMMA2_2B_REPAIR_STATUS.md`
- Repair result: `results/cards/MC001G/mc001g_gemma2_2b_repair_logit_raw_gemma_repair_20260630T121025.json`
- Result directory: `results/cards/MC001G/`

## Purpose

The repair gate produced 42 clean base-Gemma items. This pass asks whether
dense final-token hidden states contain a measurable signature of
truth-following versus user-agreement on the repaired substrate.

This pass does not test steering, patching, or sparse features.

## Primary Rows

Use only clean items:

- clean means truth-following under both `no_hint` and `correct_hint`;
- primary rows are `wrong_disclaimed` and `wrong_unsure`;
- direct/high wrong-hint rows are excluded because they are nearly saturated;
- rows labeled `other_error` are excluded from the binary signature test.

This should leave the signature test focused on intermediate-pressure rows where
both truth-following and user-agreement appear.

## Split

Use a deterministic item split stratified by correct answer letter:

- discovery split: two out of every three clean items per letter;
- holdout split: one out of every three clean items per letter.

No row from a holdout item may enter discovery fitting.

## Signals And Baselines

Signals:

- final-token residual vector at each model layer;
- truth-minus-agreement mean direction per layer;
- logistic probe per layer.

Baselines:

- correct-minus-wrong option logit margin;
- prompt condition;
- correct answer letter;
- wrong answer letter;
- condition plus correct/wrong answer letters;
- baseline answer letter.

The baseline-answer feature is expected to be tautological for binary labels and
must not be treated as a mechanism. It is included only to prevent overclaiming.

## Promotion Rule

Promote to intervention design only if:

- a hidden-state direction or probe has nontrivial holdout AUC;
- the result is not weaker than the output-margin baseline;
- condition and answer-letter baselines do not explain the signal;
- the best layer is stable enough to justify a targeted intervention;
- the status card explicitly reports failure modes and all baseline AUCs.

If the margin or prompt/letter baselines dominate, record this as a signature
failure and do not intervene.
