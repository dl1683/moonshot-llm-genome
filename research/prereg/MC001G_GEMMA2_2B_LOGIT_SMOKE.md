# MC001G Gemma 2 2B Logit Smoke Preregistration

Date: 2026-06-30

Status: preregistered for forced-choice Gate 1 behavior smoke.

## Header

- Card ID: `MC001G`
- Parent target: `MC-001` truth-versus-user-agreement conflict
- Model: `google/gemma-2-2b`
- Stage: Gate 1 forced-choice behavior smoke, no mechanism claim allowed
- Runner: `code/mc001_logit_smoke.py`
- Prior generation smoke: `research/prereg/MC001G_GEMMA2_2B_SMOKE.md`
- Manifest: `data/cards/MC001G/mc001g_gemma2_2b_logit_factual_ladder_manifest.jsonl`
- Result directory: `results/cards/MC001G/`

## Purpose

The generation smoke failed for base Gemma 2 2B because the pretrained model did
not reliably follow the answer-only interface. That does not by itself falsify a
base-model substrate, because Gemma Scope sparse artifacts target pretrained
Gemma models.

This pass therefore scores the next-token log probability of the option letters
`A`, `B`, `C`, and `D` and treats the highest-scoring option as the forced-choice
behavior.

## Design

Use the same factual-ladder manifest shape:

- 20 objective multiple-choice items;
- 8 conditions per item;
- 160 total records;
- deterministic single-forward scoring;
- no free-generation parser;
- no intervention or hidden-state claim.

The runner scores single-token variants of each option letter with and without a
leading space, then classifies the argmax option as truth-following,
user-agreement error, or other error.

Render modes:

- first pass: existing model formatter (`chat`) for continuity with the generation runner;
- control pass if the first pass is dominated by fixed option priors: raw prompt ending in `Answer:`.

The raw-render control must be interpreted only as a substrate/interface audit,
not as a mechanism result.

## Acceptance Criteria

Promote base Gemma 2 2B to hidden-state or sparse-feature discovery if:

- no-hint plus correct-hint truth-following is high enough to make wrong-hint errors meaningful;
- wrong-hint user-agreement is between 10 percent and 90 percent overall;
- at least two wrong-hint conditions show nontrivial user-agreement;
- the effect is not just a fixed option-letter prior;
- item-level clean subsets are large enough for discovery and holdout splits.

Do not promote if:

- no-hint or correct-hint truth-following is too low;
- wrong-hint agreement is absent or saturated;
- option-letter priors explain most choices;
- the clean subset is too small for a reliable signature/intervention split.

## Next Step If Passed

Install the minimal Gemma Scope loader stack and run sparse-feature discovery
against output-margin and prompt-condition baselines before any intervention.

## Next Step If Failed

Treat base Gemma 2 2B as unavailable for the current MC-001 sparse mechanism
route. The instruction-tuned fallback may remain useful for behavior and dense
diagnostics, but it should not be represented as a direct Gemma Scope feature
substrate without separate transfer evidence.
