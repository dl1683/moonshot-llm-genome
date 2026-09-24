# MC005 Associative Lookup Band-Localization Preregistration

Date: 2026-06-30

## Question

MC005 found a real source-value path intervention for Qwen3-1.7B associative
lookup, but the selected single-layer claim failed because a wrong-layer target
source mask was stronger on holdout.

The V2 question is:

> Is the source-value path localizable to a late layer band, even if not to a
> single layer/head?

## Fixed Inputs

Reuse the MC005 prompt generator and split:

- model: `Qwen/Qwen3-1.7B`
- row count: 48
- pair count: 5
- seed: 5
- clean-row definition: baseline target-minus-distractor margin > 0

The expected clean split from MC005 is:

- 41 clean rows;
- 28 discovery rows;
- 13 holdout rows.

If that split changes, do not compare the V2 result to MC005 without explaining
why.

## Candidate Bands

Discovery selection may choose among:

- `early_0_6`: layers 0-6;
- `mid_7_13`: layers 7-13;
- `signature_14_18`: layers 14-18, around the selected attention head layer;
- `late_19_23`: layers 19-23;
- `late_24_27`: layers 24-27, around the selected causal layer;
- `late_20_26`: layers 20-26, the broad late-path hypothesis;
- `all_layers`: layers 0-27.

Selection uses only discovery target source-value mask effect:

```text
masked target-minus-distractor margin - baseline margin
```

The most negative mean delta is selected.

## Holdout Controls

For the selected band, measure:

- target source-value mask;
- distractor source-value mask;
- random same-prompt value mask.

Also measure target source-value masking for every non-selected candidate band.

## Pass Rule

Band localization passes only if all are true on holdout:

1. selected-band target mask reduces mean margin by at least 1.0;
2. selected-band target mask flips at least 3 of 13 holdout target wins;
3. selected-band target mask beats selected-band distractor mask by at least
   0.50 mean margin delta;
4. selected-band target mask beats selected-band random-value mask by at least
   0.50 mean margin delta;
5. selected-band target mask beats every non-overlapping earlier-band target
   mask by at least 0.50 mean margin delta.

If `all_layers` is selected, the result is full-path control only and cannot
pass band localization.

If a late band passes but a nearby late band matches it, the result is
late-band control, not a precise layer-band mechanism.
