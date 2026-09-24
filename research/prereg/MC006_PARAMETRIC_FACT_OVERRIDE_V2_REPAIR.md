# MC006 Parametric Fact Override V2 Repair Preregistration

Date: 2026-06-30

## Purpose

MC006 V1 failed before hidden-state work. The prompt/scoring interface let a
false reference line dominate even when the prompt warned that the line might
be wrong.

V2 repairs the behavior substrate without changing the model family:

> Can Qwen3-1.7B separate real-world capital recall from a fictional task-local
> city-code override when the false city is not presented as an authoritative
> reference fact?

This is still a behavior smoke only. It does not claim a hidden signature,
intervention, or mechanism.

## Model

- `Qwen/Qwen3-1.7B`

## Source Bank

V2 uses 40 stable country-capital facts. The source bank keeps the V1
country/split structure, but replaces several task-local override cities with
less salient non-capital cities. This is intentional: V1 showed that major-city
overwrites such as Sydney, Toronto, Istanbul, and Casablanca can beat true
capitals in direct candidate scoring.

Each source has:

- country;
- true current national capital;
- fictional task-local city-code override;
- lure city;
- deterministic split.

The source bank is embedded in the runner and written into the result artifact.

## Conditions

For every source, score five prompt conditions:

1. `direct_real`: ask for the current real-world national capital with no
   context.
2. `true_fact`: provide a verified true fact, then ask for the current
   national capital.
3. `false_claim_audit`: present a false capital claim as a claim to check, then
   ask for the correct real-world city.
4. `fictional_override`: present a fictional codebook rule and ask for the
   task-local city code.
5. `real_after_fiction`: mention the same fictional codebook rule, explicitly
   say to ignore it for geography, then ask for the real-world national
   capital.

Expected labels:

- `true_answer` for `direct_real`, `true_fact`, `false_claim_audit`, and
  `real_after_fiction`;
- `override_answer` for `fictional_override`.

## Scoring

V2 uses candidate-answer mean log-probability, matching V1.

Candidates per row:

- true capital, labeled `true_answer`;
- fictional override city, labeled `override_answer`;
- unrelated lure city, labeled `lure_answer`;
- `UNKNOWN`, labeled `unknown`.

The selected label is the candidate with the highest mean log-probability per
candidate token conditioned on the rendered prompt.

## Structural Criteria

The run is structurally valid only if:

- there are exactly 40 sources;
- every source has exactly five rows;
- there are exactly 200 rows;
- every source appears in exactly one split;
- every condition has exactly 40 rows;
- no row has duplicate candidate answers;
- every scored candidate has at least one token.

## Behavior Success Criteria

The behavior substrate passes only if all criteria pass:

1. At least 32/40 `direct_real` rows select the true answer.
2. At least 36/40 `true_fact` rows select the true answer.
3. At least 30/40 `false_claim_audit` rows select the true answer.
4. No more than 8/40 `false_claim_audit` rows select the override answer.
5. At least 24/40 `fictional_override` rows select the override answer.
6. At least 30/40 `real_after_fiction` rows select the true answer.
7. No more than 8/40 `real_after_fiction` rows select the override answer.
8. At least 16 sources have a clean source-level contrast:
   `direct_real=true_answer`, `false_claim_audit=true_answer`,
   `fictional_override=override_answer`, and
   `real_after_fiction=true_answer`.
9. At least 6 clean source-level contrasts are in holdout.

## Diagnostic Labels

- `parametric_override_v2_substrate_passed`: all behavior criteria pass.
- `structural_invalid`: structural criteria fail.
- `parametric_fact_failed`: direct or true-fact capital answering is too weak.
- `false_claim_locality_failed`: the false claim still pulls the model to the
  override city.
- `override_pressure_failed`: the fictional codebook does not pull the model
  to the override city.
- `real_after_fiction_failed`: the fictional rule contaminates real-world
  answering even when explicitly ignored.
- `holdout_contrast_failed`: contrasts do not survive the split requirement.
- `mixed_behavior_failure`: any other mixed failure.

## Allowed Interpretation

If V2 passes:

> Qwen3-1.7B has a repaired behavior substrate for separating real-world
> capital preference from fictional task-local city-code override. Hidden-state
> discovery may proceed only on source-level contrast rows.

If V2 fails:

> MC006 still lacks a clean parametric-fact/context-override behavior substrate.
> Hidden-state discovery should remain blocked for this behavior family until
> the interface or task family is replaced.
