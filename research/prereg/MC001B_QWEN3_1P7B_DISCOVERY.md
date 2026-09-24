# MC001B Qwen3-1.7B Discovery Preregistration

Date: 2026-06-30

Status: preregistered for Gate 2 discovery and calibration.

## Header

- Card ID: `MC001B`
- Parent target: `MC-001` truth-versus-user-agreement conflict
- Model: `Qwen/Qwen3-1.7B`
- Prior gate: `research/cards/MC001B_QWEN3_1P7B_SMOKE_STATUS.md`
- Runner: `code/mc001_qwen3_controlled.py`
- Manifest: `data/cards/MC001B/mc001b_qwen3_1p7b_controlled_manifest.jsonl`
- Result directory: `results/cards/MC001B/`

## Purpose

Test whether Qwen3-1.7B has a hidden-state signature for wrong-hint agreement that beats prompt-condition and output/logit baselines.

This pass may produce:

- a discovery candidate;
- a failed-controls result;
- a reason to move to Gemma or revise the prompt family.

It may not produce a supported mechanism card by itself.

## Design

Use the controlled factual-ladder manifest from the Qwen3-0.6B series, but relabeled as `MC001B`.

Splits:

- discovery;
- calibration;
- holdout;
- paraphrase holdout.

Hidden indices:

- `7`, `14`, `21`, chosen as early/mid/late checkpoints in the 28-layer Qwen3 stack.

Baselines:

- prompt-condition baseline;
- correct-minus-wrong next-token logit margin baseline;
- label-permutation hidden-probe baseline.

Intervention calibration:

- train truth-minus-agreement directions only on discovery rows;
- choose dose only on calibration rows;
- keep random, wrong-layer, wrong-token, and sign-flip controls for any promoted intervention.

## Promotion Rule

Promote beyond discovery only if:

- at least one hidden probe beats the output/logit margin baseline on calibration;
- the same hidden probe is not explained by label permutation;
- holdout and paraphrase holdout do not collapse;
- the candidate is not concentrated in one item family or one prompt condition.

If output/logit margin matches or beats hidden probes again, record `diagnostic_only` or `failed_controls` and do not run a larger intervention sweep on Qwen3-1.7B without a new reason.

## Difference From Qwen3-0.6B

Do not reuse h14 as a mechanism prior.

Qwen3-0.6B found behavior control but failed mechanism promotion because output-margin, residualization, nearby-layer transport, and source-rewrite controls explained or demoted the apparent mechanism. This pass starts over at the signature gate for the larger model.
