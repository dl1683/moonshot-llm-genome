# MC006 Parametric Fact Override V6 Condition-Balanced Preregistration

Date: 2026-07-01

## Purpose

MC006 V5 found a perfect hidden-state separation on the V4 behavior substrate,
but output margin, prompt length, prompt format, and shuffled-label selection
controls matched it. The failure was not lack of signal; it was that the task
made the label too easy to read from prompt family.

V6 repairs the behavior substrate by putting both labels inside one prompt
family:

> Given a two-source answer card, can Qwen3-1.7B return either the real-world
> capital or the fictional city code depending only on a query source tag?

This is a behavior gate. It does not claim a hidden signature, intervention, or
mechanism.

## Model

- `Qwen/Qwen3-1.7B`

## Source Bank

V6 uses the 22 clean sources from the MC006 V4 source-level contrast set,
excluding Japan, the single V4 non-clean source.

The bank preserves the original source splits:

- 5 holdout sources;
- 2 calibration sources;
- 15 discovery sources.

## Prompt Family

Every row uses the same prompt family:

```text
Use this two-source answer card.
Country: <country>
Source A: <source-type> = <city>
Source B: <source-type> = <city>
Query source: <A-or-B>
Question: Which city should be returned for <country>?
Answer with only the city name.
Answer:
```

For each source, one source tag points to the real-world capital and the other
points to the fictional city code. The mapping is counterbalanced across
sources:

- half-ish of the sources use `A=real`, `B=fictional`;
- the rest use `A=fictional`, `B=real`.

This prevents a global `A` or `B` label shortcut.

## Conditions

For every source, score two rows:

1. `balanced_real`: query whichever source tag contains the real-world capital.
2. `balanced_fictional`: query whichever source tag contains the fictional city
   code.

Expected labels:

- `true_answer` for `balanced_real`;
- `override_answer` for `balanced_fictional`.

## Scoring

V6 uses candidate-answer mean log-probability, matching V1-V5.

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
- both query tags appear with both expected labels;
- no row has duplicate candidate answers;
- every scored candidate has at least one token.

## Behavior Success Criteria

The condition-balanced behavior substrate passes only if all criteria pass:

1. At least 20/22 `balanced_real` rows select the true answer.
2. At least 20/22 `balanced_fictional` rows select the override answer.
3. At least 18/22 sources have a clean source-level contrast:
   `balanced_real=true_answer` and `balanced_fictional=override_answer`.
4. At least 4/5 original holdout sources have a clean source-level contrast.
5. Query-tag/expected-label balance is valid: both query tags appear under both
   expected labels.

## Diagnostic Labels

- `condition_balanced_v6_substrate_passed`: all behavior criteria pass.
- `structural_invalid`: structural criteria fail.
- `real_query_failed`: real-world query rows are too weak.
- `fictional_query_failed`: fictional-code query rows are too weak.
- `source_contrast_failed`: source-level contrasts are too sparse or fail
  holdout.
- `tag_label_balance_failed`: query tags remain label-confounded.
- `mixed_behavior_failure`: any other mixed failure.

## Allowed Interpretation

If V6 passes:

> MC006 has a condition-balanced source-selected behavior substrate where both
> labels occur inside one prompt family. Hidden-state signature discovery may
> proceed only on this V6 table, with query-tag, output-margin, prompt-length,
> and shuffled-label controls.

If V6 fails:

> MC006 remains behavior-supported only under prompt-bounded V4. The
> condition-balanced repair is not yet strong enough for hidden-state discovery.
