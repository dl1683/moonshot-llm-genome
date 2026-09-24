# MC001B Qwen3-1.7B Residualized Preregistration

Date: 2026-06-30

Status: preregistered for Gate 2b residualized control-surface audit.

## Header

- Card ID: `MC001B`
- Parent target: `MC-001` truth-versus-user-agreement conflict
- Model: `Qwen/Qwen3-1.7B`
- Prior status: `research/cards/MC001B_QWEN3_1P7B_CONTROLLED_STATUS.md`
- Runner: `code/mc001_qwen3_controlled_v3.py`
- Manifest: `data/cards/MC001B/mc001b_qwen3_1p7b_controlled_v3_manifest.jsonl`
- Result directory: `results/cards/MC001B/`

## Purpose

The first Qwen3-1.7B controlled pass found a useful `h21 alpha=0.5` intervention, but it did not promote to a mechanism card because the next-token correct-minus-wrong logit margin reached AUC 1.000 on calibration, holdout, and paraphrase holdout.

This pass asks the next narrow question:

> Is there any remaining causal control surface after output margin, prompt condition, correct answer letter, wrong answer letter, and baseline next-token answer are treated as nuisance variables?

## Design

Use the balanced v2/v3 factual-ladder manifest from the Qwen3-0.6B residualization sequence, relabeled as `MC001B`.

Hidden index:

- `21`, chosen because the previous Qwen3-1.7B controlled run selected `h21 alpha=0.5`.

Directions:

- raw truth-minus-agreement direction on discovery rows;
- nuisance-residualized truth-minus-agreement direction on discovery rows.

Nuisance features for residualization:

- intercept;
- standardized baseline correct-minus-wrong logit margin;
- prompt condition;
- correct answer letter;
- wrong answer letter;
- baseline next-token answer.

Evaluation:

- calibration, holdout, and paraphrase holdout;
- raw and residualized direction scalar probes;
- margin-only and margin-plus-direction probes;
- logit sweep over `0`, `0.1`, `0.25`, `0.5`;
- generation arms for raw, residualized, random, nearby-layer, wrong-token, prompt-guard, and baseline controls.

## Promotion Rule

Promote only if all of the following hold:

- the residualized direction has nontrivial predictive value on holdout and paraphrase holdout beyond the margin baseline;
- residualized generation improves wrong-hint truth-following on holdout and paraphrase holdout;
- no/correct rows remain near baseline;
- answer distributions do not collapse into a single letter;
- random, nearby-layer, wrong-token, and prompt-guard controls do not explain the effect.

If the raw direction works but the residualized direction loses the effect, record a control-surface result and keep mechanism promotion closed.

If both raw and residualized directions fail, record Qwen3-1.7B residualized failure and move MC-001 to Gemma or to a different intervention family.
