# MC005 Associative Lookup Reliability V4 Preregistration

Date: 2026-06-30

## Question

MC005 V3 established a narrow late-band source-value control surface in
Qwen3-1.7B synthetic associative lookup. V4 starts the reliability atlas:

> Where does the `late_20_26` source-value intervention work, weaken, or stop
> being the right explanation?

## Fixed Claim Under Test

Fixed model:

- `Qwen/Qwen3-1.7B`

Fixed intervention:

- layers 20-26;
- all attention heads;
- eager-attention source mask at the final query token;
- source arms: target source value, distractor source value, random same-prompt
  value.

V4 does not reselect a layer band. It tests the V3 band across scope axes.

## Atlas Scenarios

Planned scenarios:

| Scenario | Pair count | Layout | Lexicon | Mode |
| --- | ---: | --- | --- | --- |
| `pair3_dash_base` | 3 | dash-colon | base | lookup |
| `pair5_dash_base` | 5 | dash-colon | base | lookup |
| `pair8_dash_base` | 8 | dash-colon | base | lookup |
| `pair5_arrow_base` | 5 | arrow | base | lookup |
| `pair8_arrow_base` | 8 | arrow | base | lookup |
| `pair5_sentence_base` | 5 | sentence | base | lookup |
| `pair5_arrow_holdout_lexicon` | 5 | arrow | shifted | lookup |
| `pair8_dash_holdout_lexicon` | 8 | dash-colon | shifted | lookup |
| `offtarget_pair5_dash` | 5 | dash-colon | base | off-target |

Lookup scenarios query the value paired with the masked target source value.
Off-target scenarios query a different pair while the masked target source value
belongs to an unrelated pair. Off-target target-source masking should therefore
have a small effect.

## Metrics

For each scenario:

- baseline target-minus-distractor margin;
- baseline clean rows;
- greedy next-token target/distractor/other counts;
- late-band target source-value mask effect;
- late-band distractor source-value mask effect;
- late-band random value mask effect;
- target-win loss under each arm.

## Scenario Labels

Lookup scenario label:

- `works`: clean rows at least 16, target mask mean delta at most -1.0, target
  mask flips at least three rows, target mask beats distractor and random masks
  by at least 0.50 mean delta.
- `weak`: target mask moves in the predicted direction but misses one or more
  work criteria.
- `breaks`: clean behavior fails or target mask does not move in the predicted
  direction.

Off-target scenario label:

- `clean_null`: target source mask absolute mean delta at most 0.50 and target
  win loss at most one row.
- `side_effect`: target source mask mean delta exceeds 0.50 in magnitude or
  flips more than one row.

## Promotion Boundary

V4 can strengthen the MC005 reliability atlas, but it does not by itself promote
MC005 to a complete mechanism card.

A full mechanism card would still need:

- model-family or size replication;
- generation beyond first-token greedy readout;
- longer contexts beyond eight pairs;
- a larger disjoint lexicon bank;
- off-target tasks outside synthetic lookup;
- finer head/path decomposition inside layers 20-26.
