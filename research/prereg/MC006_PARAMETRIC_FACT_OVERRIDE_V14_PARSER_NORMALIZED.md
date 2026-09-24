# MC006 Parametric Fact Override V14 Parser-Normalized Preregistration

Date: 2026-07-01

## Purpose

MC006 V13 nearly passed the generated matched `real_after_fiction` behavior
table. The selected template had enough non-holdout and holdout label balance,
but failed the binary-volume floor at `29/40` binary rows against a `30/40`
threshold.

V14 is a frozen-artifact parser-normalization audit. It does not rerun the
model, does not inspect hidden states, and does not perform intervention.

The narrow question is whether V13's one-row binary-volume failure is caused by
orthographic normalization rather than behavior. The only allowed parser repair
is Unicode diacritic stripping before the same strict first-line prefix parser.

## Source Artifact

V14 uses the frozen V13 generated output artifact:

`results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v13_real_after_fiction_generated_20260630T235712.json`

The source artifact must have:

- `run_type == parametric_fact_override_v13_real_after_fiction_generated`;
- 40 sources;
- 6 templates;
- 240 generated rows;
- V13 structural checks passed.

## Parser

For each generated row:

1. Strip leading whitespace and quote/backtick characters.
2. Take the first generated line.
3. Normalize both the first line and candidate answers with Unicode NFKD
   decomposition, removal of combining marks, lowercase, and whitespace
   collapse.
4. Match the normalized start of that first line against true capital,
   fictional-code city, lure city, or `UNKNOWN`, with the same
   case-insensitive word/punctuation/end boundary family as V13.
5. If exactly one candidate matches, assign its label.
6. Otherwise label the row `unparsed`.

No substring search is allowed. No explanatory-prefix rescue is allowed. No
candidate alias list is allowed beyond diacritic stripping.

## Template Selection

Recompute template selection after normalized parsing. Selection is still
discovery/calibration only:

1. maximize `min(non_holdout_true, non_holdout_override)`;
2. maximize `non_holdout_true + non_holdout_override`;
3. minimize non-holdout side rows;
4. choose the earliest template.

Holdout is never used for selection.

## Structural Criteria

The run is structurally valid only if:

- the source artifact has the expected V13 run type;
- the source artifact SHA256 is recorded;
- V13 structural checks passed;
- exactly 40 sources;
- exactly 6 templates;
- exactly 240 rows;
- exactly 40 rows per template;
- exactly 8 holdout rows per template;
- every source appears once per template;
- no duplicate record ids;
- no duplicate candidate answers;
- no prompt contains the true capital as a word-bounded prompt value.

## Parser-Delta Criteria

Normalization must be narrow:

- changed rows must be rows that V13 labeled `unparsed`;
- no row may change from one candidate label to another candidate label;
- at most five rows may change label under normalization;
- every changed row must match by `strict_first_line_prefix_nfkd`.

These criteria prevent V14 from becoming a broad parser rescue.

## Behavior Success Criteria

The V14 normalized generated table passes only if all criteria pass for the
discovery/calibration-selected template:

1. At least 30/40 rows are binary rows:
   `true_answer` or `override_answer`.
2. Non-holdout rows include at least 6 `true_answer` rows.
3. Non-holdout rows include at least 6 `override_answer` rows.
4. Holdout rows include at least 2 `true_answer` rows.
5. Holdout rows include at least 2 `override_answer` rows.
6. Holdout side rows are at most 4/8.
7. No prompt contains the true capital.
8. Parser-delta criteria pass.

## Diagnostic Labels

- `parser_normalized_generated_substrate_passed`: all criteria pass.
- `source_artifact_invalid`: source artifact or structural checks fail.
- `true_answer_prompt_leak`: any true capital appears in a prompt.
- `normalization_delta_too_broad`: parser-delta criteria fail.
- `binary_volume_failed`: selected template has fewer than 30 binary rows.
- `non_holdout_balance_failed`: selected template lacks at least 6 true and 6
  override rows in non-holdout.
- `holdout_balance_failed`: selected template lacks at least 2 true and 2
  override rows in holdout.
- `holdout_side_rows_failed`: selected template has too many holdout side rows.
- `mixed_behavior_failure`: any other mixed failure.

## Allowed Interpretation

If V14 passes:

> MC006 has a generated-answer matched `real_after_fiction` behavior table under
> a narrow accent-normalized strict parser. The selected table may proceed to a
> hidden-signature diagnostic, but that diagnostic must report parser-normalized
> labels, candidate-score margin, next-token output margin, prompt/source
> baselines, shuffled-label controls, and side rows.

If V14 fails:

> MC006 still lacks a generated-answer matched-surface behavior table for
> learned capital facts versus fictional-code contamination. The next repair
> should improve the generation contract or enlarge the source/template bank
> before hidden-state work.
