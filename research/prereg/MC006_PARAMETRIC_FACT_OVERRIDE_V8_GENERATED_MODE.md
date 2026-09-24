# MC006 Parametric Fact Override V8 Generated-Mode Preregistration

Date: 2026-07-01

## Purpose

MC006 V7 removed true-capital prompt leakage but still used candidate scoring.
The exposed fictional city line dominated the scored interface: `mode_real`
selected the true capital on only 8/22 rows, while `mode_fictional` selected the
fictional city on 22/22 rows.

V8 changes the output interface. It keeps the V7 same-prompt-family mode
contract, but scores raw greedy generated answers instead of candidate
mean-logprob choices. This tests whether the candidate-scoring surface itself is
amplifying the fictional city line.

This is a behavior gate only. It does not claim a hidden signature,
intervention, or mechanism.

## Model

- `Qwen/Qwen3-1.7B`

## Source Bank

V8 uses the same 22 clean MC006 V4 source-level contrast sources used by V6 and
V7, excluding Japan, the single non-clean V4 source.

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

The true capital is not listed as a source value in the prompt.

## Conditions

For every source, generate two rows:

1. `mode_real`: requested mode is `REAL_WORLD_CAPITAL`; expected label is
   `true_answer`.
2. `mode_fictional`: requested mode is `FICTIONAL_CITY_CODE`; expected label is
   `override_answer`.

## Generation

Use deterministic greedy decoding:

- `do_sample=False`;
- `max_new_tokens=8`;
- model/tokenizer loaded from the same local Qwen3-1.7B stack as V1-V7.

## Parsing

The generated suffix is parsed strictly:

1. Strip leading whitespace and common quote/backtick characters.
2. Take the first generated line.
3. Match the start of that first line against the true capital, fictional city,
   lure city, or `UNKNOWN`, case-insensitively and with a word/punctuation/end
   boundary.
4. If exactly one candidate matches, assign its label.
5. Otherwise label the row `unparsed`.

This parser intentionally does not search later in a sentence. A generated
answer like `The city is Paris` is not a strict pass because the prompt asked
for only the city name.

## Structural Criteria

The run is structurally valid only if:

- there are exactly 22 sources;
- exactly 5 selected sources are original holdout sources;
- every source has exactly two rows;
- there are exactly 44 rows;
- every source appears in exactly one split;
- every condition has exactly 22 rows;
- both requested modes have exactly 22 rows;
- no row has duplicate candidate strings;
- the true capital does not appear as a word-bounded source value in the prompt.

## Behavior Success Criteria

The V8 behavior substrate passes only if all criteria pass:

1. At least 40/44 rows parse as a candidate answer.
2. At least 20/22 `mode_real` rows select the true answer.
3. At least 20/22 `mode_fictional` rows select the override answer.
4. At least 18/22 sources have a clean source-level contrast:
   `mode_real=true_answer` and `mode_fictional=override_answer`.
5. At least 4/5 original holdout sources have a clean source-level contrast.
6. No prompt contains the true capital as a word-bounded prompt value.

## Diagnostic Labels

- `generated_mode_v8_substrate_passed`: all behavior criteria pass.
- `structural_invalid`: structural criteria fail.
- `true_answer_prompt_leak`: the true capital is present in the prompt.
- `parseability_failed`: fewer than 40/44 generated rows parse.
- `real_mode_failed`: real-world mode rows are too weak.
- `fictional_mode_failed`: fictional-code mode rows are too weak.
- `source_contrast_failed`: source-level contrasts are too sparse or fail
  holdout.
- `mixed_behavior_failure`: any other mixed failure.

## Allowed Interpretation

If V8 passes:

> MC006 has a same-prompt-family generated-answer behavior substrate where
> real-world capital recall and task-local fictional-code following can both be
> elicited without listing the true capital or scoring hidden answer choices.
> Hidden-state signature discovery may proceed only on this V8 table, with
> requested-mode, output-text, prompt-length, source-split holdout, and
> shuffled-label controls.

If V8 fails:

> MC006 remains behavior-supported only under prompt-bounded V4. The generated
> answer interface does not yet recover a same-prompt-family knowledge/control
> substrate.
