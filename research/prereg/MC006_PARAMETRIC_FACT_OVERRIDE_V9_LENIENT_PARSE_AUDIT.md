# MC006 Parametric Fact Override V9 Lenient Parse Audit Preregistration

Date: 2026-07-01

## Purpose

MC006 V8 replaced hidden candidate scoring with greedy generated answers, but
failed the strict first-line city parser: only 35/44 rows parsed. Some unparsed
rows nevertheless contained a true factual answer in prose, for example
`The actual capital of Germany is Berlin.`

V9 is an offline parser-rescue audit on the frozen V8 artifact. It asks:

> Did V8 fail mainly because the parser was too strict, or does the generated
> behavior still fail real-world mode and source contrasts after a preregistered
> lenient parser?

V9 does not run the model. It does not claim a hidden signature, intervention,
or mechanism.

## Source Artifact

- `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v8_generated_mode_20260630T230955.json`

## Lenient Parser

For each generated row:

1. Normalize whitespace and case in the full generated suffix.
2. Search for the full true capital, fictional city, lure city, and `UNKNOWN`
   strings anywhere in the generated suffix, using word-boundary matching.
3. If exactly one candidate string is found, assign that candidate label.
4. If multiple candidate strings are found, assign the label whose first match
   starts earliest, but only if that earliest start position is unique.
5. If no candidate string is found or the earliest start position is tied,
   assign `unparsed`.

The parser does not accept partial candidate strings. A truncated generation
such as `Reyk` does not count as `Reykjavik`.

## Structural Criteria

The audit is structurally valid only if:

- the source artifact exists;
- the source artifact run type is `parametric_fact_override_v8_generated_mode`;
- there are exactly 44 records;
- there are exactly 22 sources;
- exactly 5 sources are original holdout sources;
- each source has `mode_real` and `mode_fictional` rows;
- no row has duplicate candidate strings;
- V8 structural criteria include no true-capital prompt leaks.

## Behavior Success Criteria

The lenient parser rescue passes only if all criteria pass:

1. At least 40/44 rows parse under the lenient parser.
2. At least 20/22 `mode_real` rows select the true answer.
3. At least 20/22 `mode_fictional` rows select the override answer.
4. At least 18/22 sources have a clean source-level contrast:
   `mode_real=true_answer` and `mode_fictional=override_answer`.
5. At least 4/5 original holdout sources have a clean source-level contrast.

## Diagnostic Labels

- `lenient_parse_v9_rescue_passed`: all behavior criteria pass.
- `source_artifact_invalid`: source artifact or structural criteria fail.
- `parseability_failed`: fewer than 40/44 rows parse.
- `real_mode_failed`: real-world mode rows are too weak.
- `fictional_mode_failed`: fictional-code mode rows are too weak.
- `source_contrast_failed`: source-level contrasts are too sparse or fail
  holdout.
- `mixed_behavior_failure`: any other mixed failure.

## Allowed Interpretation

If V9 passes:

> The V8 generated behavior was mostly a strict-parser failure. MC006 may
> proceed only after freezing this lenient parser and rerunning, or after
> enforcing compact generated answers directly.

If V9 fails:

> MC006 remains behavior-supported only under prompt-bounded V4. The V8
> generated outputs do not contain enough recoverable same-prompt-family
> real-world behavior to start hidden-state discovery.
