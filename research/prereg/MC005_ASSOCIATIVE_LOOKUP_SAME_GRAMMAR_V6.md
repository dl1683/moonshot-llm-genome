# MC005 Associative Lookup Same-Grammar V6 Preregistration

Date: 2026-06-30

## Question

MC005 V5 showed that repaired out-of-grammar nulls are clean, but the
same-grammar lookup diagnostic is still only a weak null. V6 asks:

> Is the same-grammar residual effect tied to masking irrelevant source values,
> to source-line structure more generally, or to same-position/no-op masking?

## Fixed Intervention

V6 does not reselect a layer band.

Fixed model:

- `Qwen/Qwen3-1.7B`

Fixed intervention:

- layers 20-26;
- all attention heads;
- eager-attention source mask at the final answer position;
- candidate masks inside same-grammar prompts:
  - irrelevant source value;
  - key token paired with that irrelevant value;
  - colon/punctuation token on that same source line;
  - non-source control value;
  - final query-label token in the answer-absent control.

## Scenarios

Planned scenarios:

| Scenario | Target/distractor location | Arms | Purpose |
| --- | --- | --- | --- |
| `query_other_pair_original` | target and distractor are reference values | value, key, colon | repeat V5 same-grammar failure with structural controls |
| `query_other_pair_control_word` | target and distractor are reference values | value, key, colon, non-source control value | compare source-value masking with non-source value controls |
| `answer_absent_reference_control` | target and distractor are absent from reference values | value, key, colon, non-source control value, final query label | test same-grammar prompt shape when answer words are not source values |

`query_other_pair_*` scenarios preserve the original V4/V5 same-grammar lookup
surface: the prompt has reference pairs and queries another pair. The masked
source line belongs to a different pair than the queried target.

`answer_absent_reference_control` keeps the reference-pair grammar and final
query-style line, but the scored target and distractor answer words are
specified outside the reference values. Source-value masking should therefore be
clean if the same-grammar effect only comes from actual answer-value contrast.

## Metrics

For each scenario and arm:

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
- `value_specific_boundary`: baseline clean rows at least 24/32, the irrelevant
  source-value arm is not `clean`, and every non-value structural/control arm is
  `clean`.
- `structural_boundary`: baseline clean rows at least 24/32 and any key,
  punctuation, non-source, or query-label control arm is not `clean`.
- `invalid_baseline`: fewer than 24/32 baseline rows prefer the target answer.

## Pass Boundary

V6 does not promote MC005 to a full mechanism card by itself.

Useful outcomes:

- `clean_null` on all scenarios would repair the same-grammar off-target
  boundary and permit model-family replication.
- `value_specific_boundary` would show that same-grammar sensitivity is tied to
  irrelevant source values rather than line structure.
- `structural_boundary` would show that the late-band mask is still too broad
  over same-grammar prompt structure for mechanism-card promotion.
