# MC006 Parametric Fact Override V5 Hidden Signature Status

Status: hidden signal present, but signature promotion failed; output, prompt-length, prompt-format, and shuffle-selection controls match it.

Date: 2026-07-01

## Artifact

- preregistration:
  `research/prereg/MC006_PARAMETRIC_FACT_OVERRIDE_V5_SIGNATURE.md`
- runner:
  `code/mc006_parametric_fact_override_v5_signature.py`
- source behavior artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v4_source_selected_hybrid_20260630T223734.json`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v5_signature_20260630T224601.json`
- result SHA256:
  `97d322a974dc5101deac60795333edb3bcf14cf9866b08518db4eae966b5954a`

## Verdict

MC006 V5 does not pass the hidden-signature gate. Do not start intervention from
this signature.

The run found a perfect hidden classifier:

- primary rows: V4 clean source-level contrasts only;
- primary shape: 22 clean sources, 5 original holdout sources, 88 rows;
- selected hidden candidate: `layer_0` at final prompt token;
- selected hidden holdout AUC: 1.000;
- holdout source real-vs-fictional ordering: 5/5 sources passed.

But easy controls matched it:

- output-margin holdout AUC: 1.000;
- prompt-length holdout AUC: 1.000;
- prompt-format holdout AUC: 1.000;
- shuffled-label selection p95: 1.000.

The diagnostic class is:

```text
output_margin_confounded
```

The stricter interpretation is broader: V5 is output-margin confounded,
prompt-format confounded, and shuffle-selection confounded.

## Criteria

| Criterion | Result |
| --- | --- |
| structural checks pass | pass |
| selected hidden holdout AUC at least 0.85 | pass |
| selected hidden beats output margin by 0.02 | fail |
| selected hidden beats prompt format by 0.02 | fail |
| selected hidden beats shuffled-selection p95 by 0.05 | fail |
| every holdout source has real-world mean score greater than fictional score | pass |

## Interpretation

V5 shows that the V4 behavior table has a strong internal separation between
real-world prompts and fictional-code prompts, but that separation is already
available at the output logits and even from prompt shape. The hidden signal is
therefore not mechanism-grade evidence.

The next MC006 repair should make a condition-balanced signature task where
both labels occur inside the same prompt family, or use paired prompt rewriting
that controls prompt length and final prompt token before hidden-state
selection.
