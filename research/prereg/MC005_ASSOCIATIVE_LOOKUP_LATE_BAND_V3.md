# MC005 Associative Lookup Late-Band V3 Preregistration

Date: 2026-06-30

## Question

MC005 found a real source-value path intervention. V2 showed that late bands,
especially `late_20_26`, carry most of the non-all-layer effect, but the
preregistered selector chose `all_layers`, so no band-localized claim was
licensed.

V3 asks:

> If `all_layers` is excluded before selection, does a late source-value band
> survive same-layout holdout, prompt-layout holdout, and source/head-group
> controls?

## Fixed Model

- `Qwen/Qwen3-1.7B`

## Row Bank

Generate 96 synthetic associative lookup rows:

- pair count: 5;
- seed: 7;
- split schedule:
  - `discovery`: two rows per four-row block, dash-colon layout;
  - `holdout`: one row per four-row block, dash-colon layout;
  - `layout_holdout`: one row per four-row block, arrow layout.

Layouts:

```text
Reference pairs:
- river: velvet
...
- river:
```

```text
Lookup table:
river -> velvet
...
river ->
```

Clean rows are rows where the baseline target-minus-distractor next-token
margin is positive.

## Candidate Bands

`all_layers` is not a candidate.

- `early_0_6`: layers 0-6;
- `mid_7_13`: layers 7-13;
- `signature_14_18`: layers 14-18;
- `late_19_23`: layers 19-23;
- `late_24_27`: layers 24-27;
- `late_20_26`: layers 20-26.

Selection uses only discovery rows and only target source-value masking. The
selected band is the candidate with the most negative discovery mean margin
delta, tie-broken by larger target-win loss.

## Holdout Arms

For every candidate band on both holdout splits:

- target source-value mask;
- distractor source-value mask;
- random same-prompt value mask.

For the selected band, additionally test head-group masks:

- all heads;
- lower half heads;
- upper half heads;
- even heads;
- odd heads.

Head-group tests are controls. They can show whether the band effect is broad
over heads or concentrated, but V3 does not promote a single-head mechanism.

## Pass Rule

V3 passes as a late-band control surface only if:

1. clean discovery rows are at least 36;
2. clean same-layout holdout rows are at least 16;
3. clean layout-holdout rows are at least 16;
4. selected band is one of `late_19_23`, `late_24_27`, or `late_20_26`;
5. selected-band target source masking reduces mean margin by at least 1.0 on
   same-layout holdout;
6. selected-band target source masking reduces mean margin by at least 1.0 on
   layout holdout;
7. selected-band target mask flips at least three target wins on each holdout;
8. selected-band target mask beats selected-band distractor mask by at least
   0.50 mean delta on each holdout;
9. selected-band target mask beats selected-band random-value mask by at least
   0.50 mean delta on each holdout;
10. selected-band target mask beats every earlier-band target mask by at least
    0.50 mean delta on each holdout.

If all pass, the allowed claim is still narrow:

> Qwen3-1.7B associative lookup has a late-band source-value control surface
> under this synthetic key/value setup.

It is not yet a full mechanism card until reliability axes such as larger
lexicon, longer context, paraphrase/layout variation, and off-target side
effects are mapped.
