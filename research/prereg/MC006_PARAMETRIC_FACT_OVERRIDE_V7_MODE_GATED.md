# MC006 Parametric Fact Override V7 Mode-Gated Preregistration

Date: 2026-07-01

## Purpose

MC006 V6 fixed the V5 prompt-family confound structurally, but failed behavior
reliability: the model selected the real-world capital on only 17/22
`balanced_real` rows and clean contrasts reached only 15/22.

V7 repairs the behavior table differently. Instead of placing both the real
capital and fictional city in the prompt as two source lines, V7 gives only the
fictional city code and asks for one of two modes:

- `REAL_WORLD_CAPITAL`: answer from real-world geography;
- `FICTIONAL_CITY_CODE`: answer the task-local fictional city code.

This keeps both labels inside one prompt family while avoiding a pure
source-line copy task for the real-world side. V7 is a behavior gate only. It
does not claim a hidden signature, intervention, or mechanism.

## Model

- `Qwen/Qwen3-1.7B`

## Source Bank

V7 uses the same 22 clean MC006 V4 source-level contrast sources used by V6,
excluding Japan, the single non-clean V4 source.

The bank preserves the original source splits:

- 5 holdout sources;
- 2 calibration sources;
- 15 discovery sources.

## Prompt Family

Every row uses the same prompt family:

```text
Use the requested answer mode for this country.
Country: <country>
Fictional city code for this task: <fictional-city>
Mode definitions:
- REAL_WORLD_CAPITAL: answer the actual capital from real-world geography.
- FICTIONAL_CITY_CODE: answer the fictional city code listed above.
Requested answer mode: <mode>
Question: Which city should be returned for <country>?
Answer with only the city name.
Answer:
```

The true capital is not listed as a source value in the prompt. A real-mode
success therefore requires the model to prefer its parametric fact over the
task-local fictional code.

## Conditions

For every source, score two rows:

1. `mode_real`: requested mode is `REAL_WORLD_CAPITAL`; expected label is
   `true_answer`.
2. `mode_fictional`: requested mode is `FICTIONAL_CITY_CODE`; expected label is
   `override_answer`.

## Scoring

V7 uses candidate-answer mean log-probability, matching V1-V6.

Candidates per row:

- true capital, labeled `true_answer`;
- fictional override city, labeled `override_answer`;
- unrelated lure city, labeled `lure_answer`;
- `UNKNOWN`, labeled `unknown`.

The selected label is the candidate with the highest mean log-probability per
candidate token conditioned on the rendered prompt.

## Structural Criteria

The run is structurally valid only if:

- there are exactly 22 sources;
- exactly 5 selected sources are original holdout sources;
- every source has exactly two rows;
- there are exactly 44 rows;
- every source appears in exactly one split;
- every condition has exactly 22 rows;
- both requested modes have exactly 22 rows;
- no row has duplicate candidate answers;
- every scored candidate has at least one token;
- the true capital does not appear as a word-bounded source value in the prompt.

## Behavior Success Criteria

The V7 behavior substrate passes only if all criteria pass:

1. At least 20/22 `mode_real` rows select the true answer.
2. At least 20/22 `mode_fictional` rows select the override answer.
3. At least 18/22 sources have a clean source-level contrast:
   `mode_real=true_answer` and `mode_fictional=override_answer`.
4. At least 4/5 original holdout sources have a clean source-level contrast.
5. No prompt contains the true capital as a word-bounded prompt value.

## Diagnostic Labels

- `mode_gated_v7_substrate_passed`: all behavior criteria pass.
- `structural_invalid`: structural criteria fail.
- `true_answer_prompt_leak`: the true capital is present in the prompt.
- `real_mode_failed`: real-world mode rows are too weak.
- `fictional_mode_failed`: fictional-code mode rows are too weak.
- `source_contrast_failed`: source-level contrasts are too sparse or fail
  holdout.
- `mixed_behavior_failure`: any other mixed failure.

## Allowed Interpretation

If V7 passes:

> MC006 has a same-prompt-family behavior substrate where real-world capital
> recall and task-local fictional-code following can both be elicited without
> listing the true capital in the prompt. Hidden-state signature discovery may
> proceed only on this V7 table, with requested-mode, output-margin,
> prompt-length, source-split holdout, and shuffled-label controls.

If V7 fails:

> MC006 remains behavior-supported only under prompt-bounded V4. The
> same-prompt-family mode repair is not yet strong enough for hidden-state
> discovery.
