# MC006 Parametric Fact Override V13 Real-After-Fiction Generated Preregistration

Date: 2026-07-01

## Purpose

MC006 V12 tested the matched `real_after_fiction` prompt surface from V2 and
found a perfect hidden signal, but that table was candidate-scored. Because the
binary label was selected by candidate score, candidate-score margin was
tautologically perfect. V13 repairs that diagnostic flaw by using generated
answers and strict parsing instead of candidate scoring.

V13 is a behavior-table expansion only. It does not run hidden-state signature
discovery and does not perform intervention.

The goal is to find whether a matched prompt surface can produce enough
truth-following and fictional-code-following generated rows, including
source-disjoint holdout rows, to justify a later signature audit.

## Model

- `Qwen/Qwen3-1.7B`

## Source Bank

V13 uses the 40-country MC006 V2 source bank and V2 source split:

- 24 discovery sources;
- 8 calibration sources;
- 8 holdout sources.

Each source has:

- country;
- true capital;
- fictional-code city;
- lure city.

## Prompt Templates

V13 evaluates six matched prompt templates. Every template contains a fictional
mapping and asks for the real-world capital. No template asks for a requested
mode, and no template lists the true capital.

The templates are:

1. `v2_original_ignore`
2. `terse_not_geography`
3. `fake_mapping_warning`
4. `separate_task_then_real`
5. `misleading_note`
6. `memory_check`

All templates are source-matched: the same template is used for every source.
The template identity is therefore not itself the label inside a selected
template.

## Generation

Use deterministic greedy decoding:

- `do_sample=False`;
- `max_new_tokens=8`;
- local `Qwen/Qwen3-1.7B` model/tokenizer.

## Parser

Use the same strict first-line prefix parser family as V8/V10:

1. Strip leading whitespace and quote/backtick characters.
2. Take the first generated line.
3. Match the start of that line against true capital, fictional-code city, lure
   city, or `UNKNOWN`, with case-insensitive word/punctuation/end boundary.
4. If exactly one candidate matches, assign its label.
5. Otherwise label the row `unparsed`.

## Template Selection

Selection is discovery/calibration only. Holdout is never used to select the
primary template.

For each template, compute non-holdout counts:

- `true_answer`;
- `override_answer`;
- side labels: `lure_answer`, `unknown`, `unparsed`.

Select the template with the lexicographically largest tuple:

1. `min(non_holdout_true, non_holdout_override)`;
2. `non_holdout_true + non_holdout_override`;
3. negative side-row count;
4. negative template index.

This selects the most balanced binary matched surface before seeing holdout
success.

## Structural Criteria

The run is structurally valid only if:

- exactly 40 sources;
- exactly 6 templates;
- exactly 240 rows;
- exactly 40 rows per template;
- every template has exactly 8 holdout rows;
- every source appears once per template;
- no duplicate record ids;
- no duplicate candidate answers in any row;
- no prompt contains the true capital as a word-bounded prompt value.

## Behavior Success Criteria

The V13 generated matched-surface behavior table passes only if all criteria
pass for the discovery-selected template:

1. At least 30/40 rows are binary rows:
   `true_answer` or `override_answer`.
2. Non-holdout rows include at least 6 `true_answer` rows.
3. Non-holdout rows include at least 6 `override_answer` rows.
4. Holdout rows include at least 2 `true_answer` rows.
5. Holdout rows include at least 2 `override_answer` rows.
6. Holdout side rows are at most 4/8.
7. No prompt contains the true capital.

This is intentionally a behavior gate, not a mechanism gate. If it passes,
V14 may run a hidden-signature audit on the selected generated table.

## Diagnostic Labels

- `real_after_fiction_generated_substrate_passed`: all criteria pass.
- `structural_invalid`: structural criteria fail.
- `true_answer_prompt_leak`: any true capital appears in a prompt.
- `binary_volume_failed`: selected template has fewer than 30 binary rows.
- `non_holdout_balance_failed`: selected template lacks at least 6 true and 6
  override rows in non-holdout.
- `holdout_balance_failed`: selected template lacks at least 2 true and 2
  override rows in holdout.
- `holdout_side_rows_failed`: selected template has too many holdout side rows.
- `mixed_behavior_failure`: any other mixed failure.

## Allowed Interpretation

If V13 passes:

> MC006 has a generated-answer matched `real_after_fiction` behavior table where
> truth-following and fictional-code-following both occur under one
> discovery-selected prompt template, with source-disjoint holdout support.
> Hidden-state signature discovery may proceed only on the selected V13 table
> and must compare against output-margin, prompt-length, template/source
> baselines, and shuffled-label controls.

If V13 fails:

> MC006 still lacks a generated-answer matched-surface behavior table for
> learned capital facts versus fictional-code contamination. The next repair
> should enlarge the source bank or change the matched prompt family before
> hidden-state work.
