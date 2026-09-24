# MC-001 Qwen3-0.6B Smoke Status

Status: smoke complete; controlled follow-up complete.

This is not a final mechanism-card result. It is evidence that `Qwen/Qwen3-0.6B` is worth using for fast MC-001 iteration. The controlled follow-up is now recorded in [MC-001 Qwen3-0.6B Controlled Status](MC001_QWEN3_0P6B_CONTROLLED_STATUS.md).

## Purpose

Test whether `Qwen/Qwen3-0.6B` has measurable wrong-hint compliance, recoverable hidden-state signatures, and any non-null activation steering surface.

## Scripts

- behavior runner: `code/mc001_qwen3_smoke.py`
- hidden-state probe: `code/mc001_qwen3_probe.py`
- answer-option logit steering sweep: `code/mc001_qwen3_logit_steer.py`
- generation steering validation: `code/mc001_qwen3_steer.py`

Implementation notes:

- the viable prompt variant is `factual_ladder`;
- hidden-state index `h` maps to transformer block hook `h - 1`;
- MC parsing now accepts direct `A-D` letters and option-text completions.

## Behavior Smoke

Main result:

- result: `results/cards/MC001/mc001_qwen3_0p6b_smoke_factual_ladder_20260629T193212.json`
- manifest: `artifacts/cards/MC001/qwen3_0p6b_smoke_factual_ladder_manifest.jsonl`
- prompts: 20 factual multiple-choice items x 8 conditions = 160
- parseability: 160/160

Condition summary:

- `no_hint`: truth 14/20, user-agreement error 5/20, other 1/20
- `correct_hint`: truth 16/20, user-agreement error 3/20, other 1/20
- `wrong_marked_false`: truth 11/20, user-agreement error 6/20, other 3/20
- `wrong_untrusted`: truth 11/20, user-agreement error 6/20, other 3/20
- `wrong_unsure`: truth 1/20, user-agreement error 19/20
- `wrong_direct`: truth 1/20, user-agreement error 19/20
- `wrong_high`: truth 0/20, user-agreement error 20/20
- `anti_wrong`: truth 8/20, user-agreement error 10/20, other 2/20

Interpretation:

- the broad mixed direct and MC-balanced manifests are too weak or saturated for Qwen3-0.6B;
- the factual-ladder manifest gives a usable pressure gradient;
- direct/high-confidence wrong hints saturate, while marked-false/untrusted conditions remain measurable.

## Hidden Probe

Result:

- result: `results/cards/MC001/mc001_qwen3_0p6b_probe_factual_ladder_20260629T193342.json`
- source behavior result: `results/cards/MC001/mc001_qwen3_0p6b_smoke_factual_ladder_20260629T193212.json`
- probe rows: 112 suggested-answer records after excluding `no_hint`, `correct_hint`, and `other_error`
- positive label: `user_agreement_error`

Scores:

- condition-only baseline grouped-CV AUC: 0.880
- best hidden-state grouped-CV AUC: 0.957
- best hidden-state index: 7
- best hidden-state grouped-CV accuracy: 0.803

Interpretation:

- hidden states predict agreement-vs-truth above the condition-only baseline;
- this is diagnostic, not a mechanism claim, because condition and answer-token confounds are still present.

## Steering

Logit sweep:

- result: `results/cards/MC001/mc001_qwen3_0p6b_logit_steer_factual_ladder_20260629T194016.json`
- evaluated hidden indices: 5, 7, 14
- strongest useful next-token surface: hidden index 14, positive alpha
- baseline alpha 0 at hidden index 14: truth 13/52, user-agreement error 39/52
- alpha 1 at hidden index 14: truth 19/52, user-agreement error 23/52, other 10/52
- alpha 3 at hidden index 14: truth 21/52, user-agreement error 18/52, other 13/52

Generation validation:

- result: `results/cards/MC001/mc001_qwen3_0p6b_steer_factual_ladder_20260629T194456.json`
- hidden index: 14
- alphas: 0, 1, 3
- max new tokens: 16

Validation summary:

- alpha 0: parseable 52/52, truth 13/52, user-agreement error 39/52
- alpha 1: parseable 48/52, truth 33/52, user-agreement error 5/52, other 10/52
- alpha 3: parseable 0/52

Interpretation:

- hidden index 14, alpha 1 is a real positive steering result on this split;
- it is not clean enough for a final card because it introduces format drift, semantic completions, and third-option errors;
- alpha 3 is a failure mode, not an improvement, because it destroys parseability.

## Verdict

Continue with Qwen3-0.6B for rapid MC-001 iteration, but only on the factual-ladder substrate.

The controlled follow-up has now run. It found a real h14 activation-direction intervention surface but did not pass the final mechanism-card standard because the output/logit baseline predicts behavior perfectly and the intervention is not robust enough under paraphrase and side-effect checks.

Use the controlled status file as the current source of truth for the next iteration.
