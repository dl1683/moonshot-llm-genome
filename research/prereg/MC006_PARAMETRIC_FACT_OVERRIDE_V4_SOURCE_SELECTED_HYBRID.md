# MC006 Parametric Fact Override V4 Source-Selected Hybrid Preregistration

Date: 2026-06-30

## Purpose

MC006 V3 tested the V2 source-selected three-way lead under prompt paraphrase.
The real-world side passed:

- `direct_real_paraphrase`: 23/23 true;
- `true_fact_paraphrase`: 23/23 true;
- `false_claim_check`: 22/23 true and 0/23 override.

The failure was the weakened fictional-code paraphrase:

- `fictional_code_lookup`: 18/23 override, below the 20/23 floor;
- clean source contrasts: 17/23, below the 18-source floor.

V4 tests a one-change repair: keep the V3 source bank and V3 real-world
paraphrases, but restore the stronger V2 fictional-codebook wording.

This is a source-selected behavior smoke. It is not a broad country-capital
population claim.

## Model

- `Qwen/Qwen3-1.7B`

## Calibration Source

The selected source bank is the same 23-source bank used by V3, derived from:

`results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v2_repair_20260630T222815.json`

V4 also depends on the V3 failure artifact for the single-change rationale:

`results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v3_source_selected_20260630T223516.json`

## Conditions

For every selected source, score four conditions:

1. `direct_real_paraphrase`: same as V3.
2. `true_fact_paraphrase`: same as V3.
3. `false_claim_check`: same as V3.
4. `fictional_code_lookup`: same condition label as V3, but rendered with the
   stronger V2 fictional-codebook wording.

Expected labels:

- `true_answer` for `direct_real_paraphrase`, `true_fact_paraphrase`, and
  `false_claim_check`;
- `override_answer` for `fictional_code_lookup`.

## Scoring

V4 uses candidate-answer mean log-probability, matching V1, V2, and V3.

Candidates per row:

- true capital, labeled `true_answer`;
- fictional override city, labeled `override_answer`;
- unrelated lure city, labeled `lure_answer`;
- `UNKNOWN`, labeled `unknown`.

The selected label is the candidate with the highest mean log-probability per
candidate token conditioned on the rendered prompt.

## Structural Criteria

The run is structurally valid only if:

- there are exactly 23 sources;
- exactly 6 selected sources are original holdout sources;
- every source has exactly four rows;
- there are exactly 92 rows;
- every source appears in exactly one split;
- every condition has exactly 23 rows;
- no row has duplicate candidate answers;
- every scored candidate has at least one token.

## Behavior Success Criteria

The source-selected substrate passes only if all criteria pass:

1. At least 20/23 `direct_real_paraphrase` rows select the true answer.
2. At least 22/23 `true_fact_paraphrase` rows select the true answer.
3. At least 20/23 `false_claim_check` rows select the true answer.
4. No more than 3/23 `false_claim_check` rows select the override answer.
5. At least 20/23 `fictional_code_lookup` rows select the override answer.
6. At least 18 sources have a clean source-level contrast:
   `direct_real_paraphrase=true_answer`,
   `false_claim_check=true_answer`, and
   `fictional_code_lookup=override_answer`.
7. At least 5 of the 6 original holdout sources have a clean source-level
   contrast.

## Diagnostic Labels

- `source_selected_v4_substrate_passed`: all behavior criteria pass.
- `structural_invalid`: structural criteria fail.
- `parametric_fact_failed`: direct or true-fact capital answering is too weak.
- `false_claim_check_failed`: false draft answers still pull the model away
  from the true capital.
- `fictional_code_pressure_failed`: fictional lookup-table pressure is too
  weak.
- `holdout_contrast_failed`: source-level contrasts do not survive the original
  holdout split.
- `mixed_behavior_failure`: any other mixed failure.

## Allowed Interpretation

If V4 passes:

> In a V2-calibrated source-selected country-capital bank, Qwen3-1.7B separates
> real-world capital preference from a strong fictional codebook lookup prompt.
> Hidden-state discovery may proceed only on this exact source-selected and
> prompt-bounded substrate. V2's `real_after_fiction` failure and V3's weaker
> fictional-code prompt failure remain documented reliability boundaries.

If V4 fails:

> The source-selected MC006 lead is too fragile for hidden-state discovery.
> Switch task family or rebuild the behavior substrate before further mechanism
> work.
