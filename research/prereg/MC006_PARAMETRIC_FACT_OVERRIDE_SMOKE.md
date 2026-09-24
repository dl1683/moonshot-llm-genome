# MC006 Parametric Fact Override Smoke Preregistration

Date: 2026-06-30

## Purpose

MC005 mapped a strong synthetic associative-lookup surface but did not produce
a full mechanism card. The next behavior family should move closer to learned
knowledge while staying cheap enough for strict controls.

MC006 tests a behavior substrate for parametric knowledge versus task-local
context overwrite:

> Can a small LLM prefer the real stored fact when asked for the real-world
> answer, but prefer an explicit task-local overwrite when the prompt tells it
> to use an updated mapping for this task?

This is a behavior smoke only. It does not claim a hidden signature or a
mechanism.

## Model

- `Qwen/Qwen3-1.7B`

The model is already used by MC001B and MC005, so this keeps tooling and model
family fixed while changing the behavior family.

## Source Bank

V1 uses 40 stable country-capital facts. Each source has:

- country;
- true capital;
- task-local override capital;
- lure capital;
- deterministic split.

The source bank is embedded in the runner and written into the result artifact.

## Conditions

For every source, score five prompt conditions:

1. `no_context`: ask the real-world fact with no supporting context.
2. `true_context`: provide the true fact as context, then ask the fact.
3. `irrelevant_context`: provide an unrelated true fact, then ask the fact.
4. `task_override`: tell the model to use task-local updated mappings even if
   they conflict with common knowledge, provide a false mapping for the queried
   country, then ask for the answer in this task.
5. `mistake_context`: provide the same false mapping but state that the context
   may contain a mistake and ask for the real-world answer.

Expected labels:

- `true_answer` for `no_context`, `true_context`, `irrelevant_context`, and
  `mistake_context`;
- `override_answer` for `task_override`.

## Scoring

MC006 uses candidate-answer mean log-probability, not free generation.

Candidates per row:

- true capital, labeled `true_answer`;
- task-local override capital, labeled `override_answer`;
- unrelated lure capital, labeled `lure_answer`;
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
- every candidate has at least one token.

## Behavior Success Criteria

The behavior substrate passes only if all criteria pass:

1. At least 30/40 `no_context` rows select the true answer.
2. At least 30/40 `true_context` rows select the true answer.
3. At least 30/40 `irrelevant_context` rows select the true answer.
4. At least 30/40 `mistake_context` rows select the true answer.
5. At least 16/40 `task_override` rows select the override answer.
6. No more than 8/40 `mistake_context` rows select the override answer.
7. At least 16 sources have a clean source-level contrast:
   `no_context=true_answer`, `mistake_context=true_answer`, and
   `task_override=override_answer`.
8. At least 6 clean source-level contrasts are in holdout.

## Diagnostic Labels

- `parametric_override_substrate_passed`: all behavior criteria pass.
- `parametric_fact_failed`: true-answer behavior is too weak in no-context or
  true-context rows.
- `context_locality_failed`: irrelevant or mistake context pulls the model away
  from the true answer.
- `override_pressure_failed`: task-local overwrite pressure is too weak.
- `holdout_contrast_failed`: contrasts do not survive the split requirement.
- `structural_invalid`: structural criteria fail.
- `mixed_behavior_failure`: any other mixed failure.

## Allowed Interpretation

If MC006 passes:

> Qwen3-1.7B has a behavior substrate for separating parametric fact preference
> from task-local context overwrite. Hidden-state discovery may proceed on
> source-level contrast rows.

If MC006 fails:

> The current prompt/scoring interface does not cleanly separate stored fact
> preference from context overwrite. Hidden-state discovery should not start
> until the behavior substrate is repaired or replaced.
