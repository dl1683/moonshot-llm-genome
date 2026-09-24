# MC006 Parametric Fact Override V3 Source-Selected Preregistration

Date: 2026-06-30

## Purpose

MC006 V1 failed because false reference lines dominated. MC006 V2 failed the
full prompt suite, mainly because direct capital scoring remained weak and
`real_after_fiction` showed broad contamination from a fictional codebook rule.

V2 also exposed a narrower lead: 23/40 sources were clean on a three-way
contrast:

- direct real-world capital answer selected the true capital;
- false-claim audit selected the true capital;
- fictional codebook lookup selected the override city.

V3 tests that lead under prompt paraphrase. It is a source-selected behavior
substrate smoke, not a broad country-capital population claim.

## Model

- `Qwen/Qwen3-1.7B`

## Calibration Source

The selected source bank is derived from:

`results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v2_repair_20260630T222815.json`

V3 includes only sources that passed:

`direct_real=true_answer`, `false_claim_audit=true_answer`, and
`fictional_override=override_answer`.

This yields 23 sources, including 6 original holdout sources.

## Conditions

For every selected source, score four paraphrased prompt conditions:

1. `direct_real_paraphrase`: ask for the official national capital using a
   different direct-real wording from V2.
2. `true_fact_paraphrase`: provide a verified table row with the true capital.
3. `false_claim_check`: present a false draft answer and ask for the corrected
   real-world capital.
4. `fictional_code_lookup`: present a fictional lookup-table label and ask for
   the fictional label.

Expected labels:

- `true_answer` for `direct_real_paraphrase`, `true_fact_paraphrase`, and
  `false_claim_check`;
- `override_answer` for `fictional_code_lookup`.

## Scoring

V3 uses candidate-answer mean log-probability, matching V1 and V2.

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

- `source_selected_v3_substrate_passed`: all behavior criteria pass.
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

If V3 passes:

> In a V2-calibrated source-selected country-capital bank, Qwen3-1.7B separates
> real-world capital preference from fictional codebook lookup under paraphrased
> prompts. Hidden-state discovery may proceed only on this source-selected
> substrate, and V2's `real_after_fiction` failure remains a documented locality
> boundary.

If V3 fails:

> The V2 three-way lead does not survive prompt paraphrase. MC006 should either
> switch task family or treat capital-fact/context-override as behavior-gated.
