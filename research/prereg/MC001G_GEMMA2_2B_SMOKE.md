# MC001G Gemma 2 2B Smoke Preregistration

Date: 2026-06-30

Status: preregistered for Gate 1 behavior smoke.

## Header

- Card ID: `MC001G`
- Parent target: `MC-001` truth-versus-user-agreement conflict
- Model: `google/gemma-2-2b`
- Stage: Gate 1 smoke, no mechanism claim allowed
- Runner: `code/mc001_qwen3_smoke.py`
- Stack decision: `research/19_GEMMA_STACK_DECISION.md`
- Manifest: `data/cards/MC001G/mc001g_gemma2_2b_smoke_factual_ladder_manifest.jsonl`
- Result directory: `results/cards/MC001G/`

## Question

Does base Gemma 2 2B provide a viable MC-001 behavior substrate that can justify
later sparse-feature or path-level work?

This smoke answers only:

- Are answers parseable?
- Is no-hint/correct-hint truth-following high enough?
- Is wrong-hint agreement measurable but not saturated?
- Do pressure conditions produce useful behavioral variation?
- Does the base pretrained model obey the answer-only interface well enough?

## Design

Use the existing factual-ladder prompt family:

- 20 objective multiple-choice items;
- 8 conditions per item;
- 160 total records;
- answer format constrained to `A`, `B`, `C`, or `D`;
- deterministic greedy generation;
- no chain-of-thought requirement;
- parser inherited from the MC-001 smoke runner.

Conditions:

- `no_hint`;
- `correct_hint`;
- `wrong_marked_false`;
- `wrong_untrusted`;
- `wrong_unsure`;
- `wrong_direct`;
- `wrong_high`;
- `anti_wrong`.

## Acceptance Criteria

Promote to sparse-feature or path discovery if:

- total parseability is at least 95 percent;
- no-hint plus correct-hint truth-following is high enough to make wrong-hint errors meaningful;
- wrong-hint user-agreement is between 10 percent and 90 percent overall;
- at least two wrong-hint conditions show nontrivial user-agreement;
- prompt artifacts do not dominate the result;
- the base model does not collapse into a repeated boilerplate completion or option-letter prior.

Do not promote if:

- answers are often unparseable;
- no-hint accuracy is too low;
- wrong-hint agreement is nearly absent or saturated;
- the model ignores the answer-only instruction;
- behavior is explainable as a manifest, template, or parser failure.

## Tooling Boundary

Sparse mechanism work is deliberately out of scope for this smoke. `sae_lens`,
`transformer_lens`, and `nnsight` are not installed in the current environment.
The known Gemma Scope 2B pretrained artifacts are a reason to choose this
substrate, not evidence that a feature-level mechanism exists.

## Next Step If Passed

Install or vendor the minimal sparse-feature stack and run a model-specific
discovery pass that:

- checks output/logit margin first;
- measures hidden signatures at answer-token, hint-token, and final-token sites;
- tests sparse features against dense and prompt-condition baselines;
- proceeds to intervention only if the sparse signature adds evidence beyond output margin.

## Next Step If Failed

If the failure is instruction-following or parseability specific, run one bounded
fallback smoke on `google/gemma-2-2b-it`.

If the behavior itself is absent or saturated, do not force Gemma into the
mechanism pipeline. Return to a narrow Qwen3-1.7B causal-path test conditioned on
prompt guard and agreement-favored output-margin bins.
