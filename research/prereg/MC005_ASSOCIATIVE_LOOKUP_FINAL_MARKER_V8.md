# MC005 Associative Lookup Final-Marker V8 Preregistration

Date: 2026-06-30

## Question

MC005 V7 kept source/control masks clean but found a stable weak boundary on the
final colon of the valid `Answer:` surface. V8 asks:

> Is the remaining boundary specific to final-position punctuation, or does it
> disappear when the final answer marker changes?

## Fixed Intervention

V8 does not reselect a layer band.

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

Each seed builds the same valid answer-word substrate with 32 rows per marker.
Target and distractor answer words are absent from all reference-pair values.

## Final Markers

Planned final answer markers:

| Marker | Final line | Arms |
| --- | --- | --- |
| `answer` | `Answer:` | source value, control value, earlier neutral colon, final label, final colon |
| `response` | `Response:` | source value, control value, earlier neutral colon, final label, final colon |
| `output` | `Output:` | source value, control value, earlier neutral colon, final label, final colon |
| `result` | `Result:` | source value, control value, earlier neutral colon, final label, final colon |

The earlier neutral colon comes from a non-answer line that should not affect
the target-vs-distractor answer margin. It tests whether punctuation masking is
only a final-position effect.

## Metrics

For each seed and marker:

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
- `final_punctuation_boundary`: baseline clean rows at least 24/32,
  source/control/earlier-colon arms are `clean`, and the final-colon arm is not
  `clean`.
- `final_label_boundary`: baseline clean rows at least 24/32,
  source/control/earlier-colon arms are `clean`, and the final-label arm is not
  `clean`.
- `structural_boundary`: baseline clean rows at least 24/32 and a source,
  control, or earlier-colon arm is not `clean`.
- `mixed_boundary`: more than one boundary class appears in the same scenario.
- `invalid_baseline`: fewer than 24/32 baseline rows prefer the target answer.

## Pass Boundary

V8 repairs final-marker robustness only if every seed-marker scenario is
`clean_null`.

Useful but non-promoting outcomes:

- all source/control/earlier-colon arms clean while final-colon arms fail:
  final-position punctuation remains the blocker;
- some markers clean and others fail: choose a clean marker for the next atlas
  and document marker-specific reliability;
- invalid baselines: the marker surface is not a valid null substrate.

V8 cannot by itself promote MC005 to a complete mechanism card. It can only
decide whether final-marker punctuation is stable enough to revisit broader
replication and head/path decomposition.
