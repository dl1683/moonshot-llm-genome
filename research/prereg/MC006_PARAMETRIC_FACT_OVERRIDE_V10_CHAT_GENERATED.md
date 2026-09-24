# MC006 Parametric Fact Override V10 Chat-Generated Preregistration

Date: 2026-07-01

## Purpose

MC006 V8 showed that raw generated answers under the mode-gated prompt did not
form a reliable behavior substrate. MC006 V9 showed that lenient parsing rescued
some prose rows but still left real-world mode at only 11/22 true answers.

V10 changes the rendering contract rather than the parser. It uses the local
Qwen3 chat template with a system-level instruction to reply with exactly one
city name and no explanation. It also renders with `enable_thinking=False` so
the Qwen3 template inserts an empty thinking block before generation.

This is a behavior gate only. It does not claim a hidden signature,
intervention, or mechanism.

## Model

- `Qwen/Qwen3-1.7B`

## Source Bank

V10 uses the same 22 clean MC006 V4 source-level contrast sources used by
V6-V9, excluding Japan, the single non-clean V4 source.

The bank preserves the original source splits:

- 5 holdout sources;
- 2 calibration sources;
- 15 discovery sources.

## Chat Prompt

Every row is rendered through `tokenizer.apply_chat_template` with:

- `add_generation_prompt=True`;
- `enable_thinking=False`;
- a system message:

```text
You are a strict answer engine. Reply with exactly one city name and no other
text. Do not explain. Do not write a sentence.
```

The user message is:

```text
Use the requested answer mode for this country.
Country: <country>
Fictional city code for this task: <fictional-city>
Mode definitions:
- REAL_WORLD_CAPITAL: answer the actual capital from real-world geography.
- FICTIONAL_CITY_CODE: answer the fictional city code listed above.
Requested answer mode: <mode>
Question: Which city should be returned for <country>?
Return only the city name.
```

The true capital is not listed as a source value in either message.

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
- model/tokenizer loaded from the same local Qwen3-1.7B stack as V1-V9.

## Primary Parser

Use the V8 strict parser:

1. Strip leading whitespace and common quote/backtick characters.
2. Take the first generated line.
3. Match the start of that first line against the true capital, fictional city,
   lure city, or `UNKNOWN`, case-insensitively and with a word/punctuation/end
   boundary.
4. If exactly one candidate matches, assign its label.
5. Otherwise label the row `unparsed`.

The V9 lenient parser is reported as a diagnostic only. It is not the primary
pass criterion.

## Structural Criteria

The run is structurally valid only if:

- the tokenizer has a chat template;
- rendering with `enable_thinking=False` succeeds;
- there are exactly 22 sources;
- exactly 5 selected sources are original holdout sources;
- every source has exactly two rows;
- there are exactly 44 rows;
- every source appears in exactly one split;
- every condition has exactly 22 rows;
- both requested modes have exactly 22 rows;
- no row has duplicate candidate strings;
- the true capital does not appear as a word-bounded source value in the system
  message, user message, or rendered chat prompt.

## Behavior Success Criteria

The V10 behavior substrate passes only if all criteria pass:

1. At least 40/44 rows parse under the strict parser.
2. At least 20/22 `mode_real` rows select the true answer.
3. At least 20/22 `mode_fictional` rows select the override answer.
4. At least 18/22 sources have a clean source-level contrast:
   `mode_real=true_answer` and `mode_fictional=override_answer`.
5. At least 4/5 original holdout sources have a clean source-level contrast.
6. No prompt contains the true capital as a word-bounded prompt value.

## Diagnostic Labels

- `chat_generated_v10_substrate_passed`: all behavior criteria pass.
- `structural_invalid`: structural criteria fail.
- `true_answer_prompt_leak`: the true capital is present in the prompt.
- `parseability_failed`: fewer than 40/44 generated rows parse strictly.
- `real_mode_failed`: real-world mode rows are too weak.
- `fictional_mode_failed`: fictional-code mode rows are too weak.
- `source_contrast_failed`: source-level contrasts are too sparse or fail
  holdout.
- `mixed_behavior_failure`: any other mixed failure.

## Allowed Interpretation

If V10 passes:

> MC006 has a chat-rendered same-prompt-family generated-answer behavior
> substrate where real-world capital recall and task-local fictional-code
> following can both be elicited without listing the true capital or scoring
> hidden answer choices. Hidden-state signature discovery may proceed only on
> this V10 table, with requested-mode, output-text, prompt-length, source-split
> holdout, and shuffled-label controls.

If V10 fails:

> MC006 remains behavior-supported only under prompt-bounded V4. Chat rendering
> did not recover a same-prompt-family knowledge/control substrate.
