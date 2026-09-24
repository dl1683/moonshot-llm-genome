# MC005 Associative Lookup Response-Marker V24 Source Position Causal Preregistration

Date: 2026-06-30

## Question

MC005 V23 found that V22 row heterogeneity is structured by target source
position and query index. Early target source positions were mostly
pair-recovered, while several mid/late source positions had much higher
all-three row fractions. V23 was offline and correlational.

V24 asks:

> Does moving the same target key/value pair from an early source position to a
> mid/late source position causally increase row-level all-three structure?

## Fixed Surface

Fixed model:

- `Qwen/Qwen3-1.7B`

Fixed behavior:

- synthetic associative lookup;
- pair count 16;
- final answer marker: `Response:`.

Fixed intervention:

- eager-attention source mask at the final answer position.

Parent path:

- `slice_l24_26_all`: layers 24-26, all heads.

Component paths:

- leave-one-layer-out pairs:
  - `slice_l24_25_all`;
  - `slice_l24_26_all_pair`;
  - `slice_l25_26_all`;
- single-layer components:
  - `single_l24_all`;
  - `single_l25_all`;
  - `single_l26_all`.

## Position Manipulation

V24 builds matched row families. Each family fixes:

- the full set of 16 key/value pairs;
- the queried key;
- the target value;
- the distractor value;
- the random source-control value.

Only the source-list slot of the target pair changes.

Target source slots:

- early: slots 0 and 1;
- mid/late: slots 10 and 13.

Each base family produces four rows:

- `early_slot0`;
- `early_slot1`;
- `mid_slot10`;
- `late_slot13`.

The query line remains `Query key: {same_key}` in all four rows. The source
pair order changes only enough to place the same target pair in the requested
slot.

## Rows

Lookup diagnostic:

- base-family seeds: 211 and 223;
- base families per seed: 32;
- variants per family: 4;
- total lookup rows: 256.

Answer-absent null holdout:

- scenario: `answer_absent_pair16_response_null`;
- seeds: 227 and 229;
- 96 rows per seed.

These seeds are disjoint from V15-V23 lookup and null seeds.

## Metrics

Aggregate metrics by position group and slot:

- baseline target-minus-distractor margin;
- baseline target-win count;
- parent target, distractor, and random source-value mask mean deltas;
- pair and single-layer target source-value mask mean deltas;
- target-win losses;
- all-three row count and fraction among parent-effect rows;
- median best-pair share among parent-effect rows;
- median single-layer-sum residual among parent-effect rows.

Paired metrics by base family:

- whether any early variant is all-three;
- whether any mid/late variant is all-three;
- paired all-three gain: mid/late any minus early any;
- paired best-pair-share change;
- paired parent-effect presence.

Null metrics:

- parent source value, non-source control value, earlier neutral colon, final
  label, and final colon mean deltas and target-win changes.

## Row Definitions

Use the V22 row definitions:

- parent-effect row: baseline target wins and parent target source-value mask
  delta is at most -1.0;
- all-three margin row:
  - every leave-one-layer-out pair recovers less than 60 percent of the parent
    negative-effect magnitude;
  - the single-layer sum residual is at most -1.0;
  - every pair-plus-omitted-layer residual is at most -0.5.

## Pass Boundary

V24 supports source-position causality only if all criteria pass:

1. every position group has baseline target wins at least 75 percent of rows;
2. every position group has parent target source-value mean delta at most -1.0
   and at least three target-win losses;
3. in every position group, parent target source-value masking beats distractor
   and random source controls by at least 0.50 mean delta;
4. the mid/late all-three fraction exceeds the early all-three fraction by at
   least 0.20;
5. at least 12 more base families have mid/late-only all-three support than
   early-only all-three support;
6. median best-pair share on parent-effect rows is lower in mid/late rows than
   early rows by at least 0.05;
7. both answer-absent null holdout seeds are `clean_null` under the parent path.

## Diagnostic Classes

- `source_position_causal_supported`: all pass-boundary criteria pass.
- `position_parent_failed`: parent effect, source controls, or baselines fail
  in at least one position group.
- `position_fraction_not_causal`: parent/control evidence passes, but mid/late
  all-three fraction does not beat early by the required margin.
- `paired_position_not_causal`: fraction moves, but the paired family-level
  contrast is too weak.
- `best_pair_share_not_shifted`: all-three fractions move, but best-pair share
  does not shift in the predicted direction.
- `null_failed`: answer-absent parent null holdout fails.
- `mixed_failure`: multiple criteria fail without a more specific label.

V24 is still not a smaller-path promotion. A pass says source position is a
causal contributor to row-level all-three structure under the pair16
`Response:` lookup contract. A fail says V23's source-position contrast was
correlational or insufficient under controlled key/value identity.
