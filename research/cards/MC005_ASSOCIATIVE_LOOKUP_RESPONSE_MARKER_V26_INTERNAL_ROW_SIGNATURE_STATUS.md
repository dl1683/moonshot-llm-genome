# MC005 Associative Lookup Response-Marker V26 Internal Row Signature Status

Status: internal row-signature diagnostic passed; V27 additive intervention
failed to turn it into causal row control.

Date: 2026-06-30

## Artifact

- preregistration:
  `research/prereg/MC005_ASSOCIATIVE_LOOKUP_RESPONSE_MARKER_V26_INTERNAL_ROW_SIGNATURE.md`
- runner:
  `code/mc005_associative_lookup_response_marker_v26_internal_row_signature.py`
- source artifact:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v25_factorial_row_heterogeneity_20260630T205454.json`
- result:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v26_internal_row_signature_20260630T210810.json`
- result SHA256:
  `ba8b890de3ca0ed518aa6e3a4d963ee4838285798a7a6a4cbf9dbcf335bcaf03`

## Verdict

V26 passes the preregistered internal row-signature gate. A baseline
pre-intervention residual-stream direction at layer 20, target-value token
position, predicts which V25 parent-effect rows are all-three/pair-resistant
rows.

The selected internal candidate was `l20_target_value`:

- discovery AUC: 0.9840 on seed 233;
- holdout AUC: 0.7933 on seed 239;
- best non-internal baseline: `baseline_margin`, holdout AUC 0.5586;
- shuffled-selection null p95: 0.6510;
- all eligible holdout subgroups stayed above 0.60 AUC.

This is the first positive internal signature for the MC005 row-level
heterogeneity problem after source-position-only and simple prompt-layout
factorial explanations failed. It is not yet a mechanism card, because V26 did
not intervene on the selected signature.

Summary:

```text
model: Qwen/Qwen3-1.7B
source artifact: V25 factorial row heterogeneity
discovery seed: 233
holdout seed: 239
rows: 256 parent-effect rows
label: all_three_margin_row
selected internal candidate: l20_target_value
passed: true
diagnostic class: internal_row_signature_supported
```

## Label Balance

| Split | Rows | All-three positives | Non-all-three negatives | Valid |
| --- | ---: | ---: | ---: | --- |
| discovery seed 233 | 128 | 29 | 99 | yes |
| holdout seed 239 | 128 | 44 | 84 | yes |

## Top Internal Candidates

| Candidate | Discovery AUC | Holdout AUC |
| --- | ---: | ---: |
| `l20_target_value` | 0.9840 | 0.7933 |
| `l23_target_value` | 0.9840 | 0.7608 |
| `l24_target_value` | 0.9836 | 0.7627 |
| `l21_target_value` | 0.9829 | 0.7746 |
| `l26_target_value` | 0.9819 | 0.7560 |
| `l26_final_colon` | 0.9801 | 0.6742 |
| `l22_target_value` | 0.9798 | 0.7673 |
| `l25_target_value` | 0.9798 | 0.7459 |

The strongest and most stable candidates concentrate at the target source
token, not at the final marker.

## Baselines

| Baseline | Discovery AUC | Holdout AUC |
| --- | ---: | ---: |
| `baseline_margin` | 0.5221 | 0.5586 |
| `target_source_position` | 0.5111 | 0.5346 |
| `target_slot` | 0.5111 | 0.5346 |
| `layout_margin_linear` | 0.5186 | 0.5318 |
| `target_distractor_token_distance` | 0.5167 | 0.5130 |
| `abs_target_distractor_token_distance` | 0.5111 | 0.4827 |
| `distractor_source_position` | 0.5167 | 0.4740 |
| `distractor_slot` | 0.5167 | 0.4740 |

The selected internal holdout AUC beats the best baseline by +0.2347.

## Shuffled-Selection Null

| Null statistic | Value |
| --- | ---: |
| iterations | 100 |
| holdout AUC mean | 0.5032 |
| holdout AUC p95 | 0.6510 |
| holdout AUC max | 0.7135 |

The selected internal holdout AUC beats the shuffled p95 by +0.1423.

## Holdout Subgroups

| Subgroup | Rows | Positives | Negatives | AUC |
| --- | ---: | ---: | ---: | ---: |
| target position early | 64 | 20 | 44 | 0.7909 |
| target position mid/late | 64 | 24 | 40 | 0.7938 |
| distractor relation far | 64 | 21 | 43 | 0.7973 |
| distractor relation near | 64 | 23 | 41 | 0.7890 |

## Criteria

| Criterion | Result |
| --- | --- |
| discovery and holdout label balance valid | pass |
| selected internal holdout AUC at least 0.70 | pass |
| selected internal beats best baseline by at least 0.05 | pass |
| selected internal beats shuffled-selection p95 by at least 0.03 | pass |
| eligible holdout subgroups at least 0.60 AUC | pass |

## Interpretation

What V26 supports:

- a real internal signature exists for row-level all-three structure inside the
  reliable MC005 parent surface;
- the signature is not explained by baseline output margin or simple
  source/distractor layout features;
- target-source residual state around layer 20 is the next mechanistic lead.

What V26 does not support:

- no causal claim for the signature yet;
- no deployment or arbitrary-context claim;
- no single-head or single-layer intervention claim;
- no claim that row-level all-three structure is fully explained.

## Next Step

V27 tested a simple additive residual intervention on the `l20_target_value`
row signature and failed: `plus_target` had 43/128 all-three rows versus 44/128
for the no-intervention baseline. The next MC005 pass should treat V26 as a
predictive internal signature and V27 as a negative additive-intervention
control, not as a solved mechanism.
