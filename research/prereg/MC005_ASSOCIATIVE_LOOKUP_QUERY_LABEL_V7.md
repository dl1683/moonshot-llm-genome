# MC005 Associative Lookup Query-Label V7 Preregistration

Date: 2026-06-30

## Question

MC005 V6 made same-grammar source-line masks look clean, but one seed-17
answer-absent control failed because masking the final query label changed
three target-win rows despite a small mean-margin delta. V7 asks:

> Is the remaining same-grammar boundary a source-line issue, a final-position
> label issue, or a row-flip threshold artifact?

## Fixed Intervention

V7 does not reselect a layer band.

Fixed model:

- `Qwen/Qwen3-1.7B`

Fixed intervention:

- layers 20-26;
- all attention heads;
- eager-attention mask at the final answer position.

## Seed Sweep

Run seeds:

- 17;
- 23;
- 31.

Each seed builds the same scenario surfaces with 32 rows per surface. Target
and distractor answer words are absent from all reference-pair values in every
surface.

## Surfaces

| Surface | Final label form | Arms | Purpose |
| --- | --- | --- | --- |
| `original_random_label` | `- <random label>:` | source value, source key, source colon, control value, final label, final colon | repeat the V6 answer-absent shape |
| `repeated_random_label` | label appears earlier and again at the final line | source value, control value, earlier label, final label, final colon | separate label semantics from final-position masking |
| `generic_answer_label` | `Answer:` | source value, control value, final label, final colon | test a conventional answer label |
| `irrelevant_random_label` | `- <random label>:` with instruction to ignore the label | source value, control value, final label, final colon | test a semantically irrelevant final label |

## Metrics

For each seed and surface:

- baseline target-minus-distractor margin;
- baseline clean rows;
- greedy next-token target/distractor/other counts;
- arm mean delta;
- arm min and max delta;
- target-win loss;
- absolute target-win change.

## Arm Labels

Arm label:

- `clean`: absolute mean delta at most 0.50 and absolute target-win change at
  most one row.
- `weak`: absolute mean delta at most 1.00 and absolute target-win change at
  most two rows.
- `side_effect`: exceeds the `weak` limits.

## Scenario Labels

Scenario label:

- `clean_null`: baseline clean rows at least 24/32 and every arm is `clean`.
- `query_label_position_boundary`: baseline clean rows at least 24/32,
  source/control arms are `clean`, and at least one final-label or final-colon
  arm is not `clean`.
- `label_semantic_boundary`: baseline clean rows at least 24/32 and an
  earlier-label arm is not `clean`.
- `source_control_regression`: baseline clean rows at least 24/32 and any
  source value, source key, source colon, or non-source control-value arm is not
  `clean`.
- `mixed_boundary`: more than one boundary class appears in the same scenario.
- `invalid_baseline`: fewer than 24/32 baseline rows prefer the target answer.

## Pass Boundary

V7 repairs query-label/position robustness only if every seed-surface scenario
is `clean_null`.

Useful but non-promoting outcomes:

- all source/control arms clean but final-label or final-colon arms not clean:
  the blocker is final-position label robustness, not source-line masking.
- mean deltas clean but target-win row flips fail: the blocker is discrete
  row-flip sensitivity near the decision boundary.
- source/control regressions appear: the V6 source-line repair was not stable.

V7 cannot by itself promote MC005 to a complete mechanism card. It can only
decide whether query-label/position controls are stable enough to justify
model-family replication.
