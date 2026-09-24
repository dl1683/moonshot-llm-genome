# MC006 Parametric Fact Override V22 Source/Path Lead-Time Curve Status

Status: source/path lead-time map supported as a diagnostic, but final
candidate-score and output margins still dominate; no MC006 signature or
intervention route is promoted.

Date: 2026-07-01

## Artifact

- design source:
  `research/21_CONTROL_SURFACE_LAWS_AND_GAPS.md`
- runner:
  `code/mc006_parametric_fact_override_v22_source_path_leadtime_curve.py`
- source V19 artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v19_overlapping_margin_table_20260701T023921.json`
- source V19 SHA256:
  `aea4913e580da82f52f301af9360749b8b048d95bcfcb20db59c7411dd5da29f`
- source V20 artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v20_strict_overlap_selection_audit_20260701T025043.json`
- source V20 SHA256:
  `d6f677aab7fe38f681124fc939cd18d8a322c35d694d81750fcf6f76c2598708`
- source V21 artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v21_pair_matched_leadtime_20260701T030507.json`
- source V21 SHA256:
  `2a609d24b1bb73df447be6dcdb4aac00adee604e5a7278d682fbf2550035baf3`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v22_source_path_leadtime_curve_20260701T032420.json`
- result SHA256:
  `0b59ed20477ff79dc58c65d7a29edb3fc7b0024e4d49e27b9c8a1eb86b1183aa`

## Verdict

V22 does not make MC006 signature-ready.

The diagnostic class is:

```text
source_path_final_margin_shadow
```

The supported diagnostic is narrower than promotion:

- source artifacts valid: pass;
- structural table passed: pass;
- position mapping complete: pass;
- holdout pair count >= 20: pass;
- curve has hidden holdout support: pass;
- curve has at least one hidden point that beats same-position output by 0.05:
  pass;
- curve has a hidden point that beats candidate-score and final-output margins
  by 0.05: fail;
- selected source/path candidates beat same-position output by 0.05: fail;
- signature-ready: false;
- intervention-ready: false.

This is a useful typed failure. It shows that the MC006 source/path frontier has
real monitor structure, but that structure remains downstream-shadowed by
global candidate and final next-token geometry.

## Source Check

V22 validated all source artifacts:

- V19 expected diagnostic: `non_holdout_candidate_margin_overlap_failed`;
- V20 expected diagnostic: `strict_final_margin_overlap_absent`;
- V21 expected diagnostic:
  `approximate_pair_matching_failed_margin_baselines`;
- all source artifacts were not signature-ready.

The V22 row table reused the V19 pooled binary rows under the V21 approximate
pair contract:

| Split | True rows | Override rows |
| --- | ---: | ---: |
| discovery | 108 | 43 |
| calibration | 30 | 19 |
| holdout | 42 | 15 |

Pooling templates remains a diagnostic upper bound, not a prompt-controlled
promotion substrate.

## Position Mapping

V22 mapped all 257 rows at eight positions:

| Position | Role |
| --- | --- |
| `mapping_country_token` | country mention inside the fictional mapping line |
| `mapping_value_token` | fictional value token inside the mapping line |
| `question_country_token` | queried country token |
| `after_mapping_line` | line boundary after the mapping |
| `after_instruction_line` | line boundary after the instruction |
| `after_question_line` | line boundary after the question |
| `after_return_line` | line boundary after `Return only the city name.` |
| `final_prompt_token` | final prompt token before answer generation |

Mapping failures: 0/257.

## Pair Boundary

V22 used the same approximate joint margin-pair boundary as V21:

| Split | Joint z <= 0.5 pairs | Participants | Mean max absolute z delta |
| --- | ---: | ---: | ---: |
| non-holdout | 451 | 63 | 0.350 |
| holdout | 28 | 14 | 0.344 |

This is enough for a bounded diagnostic curve. It is still not a strict
overlap table.

## Selected Source/Path Signals

Selection was restricted to the three source/path positions:

- `mapping_country_token`;
- `mapping_value_token`;
- `question_country_token`.

The best non-holdout pair-accuracy candidate was:

| Field | Value |
| --- | --- |
| selected candidate | `question_country_token/layer_20` |
| non-holdout AUC | 0.927 |
| non-holdout pair accuracy | 0.798 |
| holdout AUC | 0.800 |
| holdout pair accuracy | 0.607 |
| same-position output holdout pair accuracy | 0.679 |

The best non-holdout AUC candidate was:

| Field | Value |
| --- | --- |
| selected candidate | `question_country_token/layer_22` |
| non-holdout AUC | 0.932 |
| non-holdout pair accuracy | 0.783 |
| holdout AUC | 0.832 |
| holdout pair accuracy | 0.714 |
| same-position output holdout pair accuracy | 0.679 |

So the source/path selected signal is a monitor, but not a promotion-ready
signature. The AUC-selected candidate clears the weak support floor and beats
same-position output in pair accuracy by 0.036, not by the preregistered 0.05
gap.

## Full Lead-Time Curve

The full curve shows why this is still a real atlas datum.

| Position | Pair-selected hidden | Holdout pair acc | Holdout AUC | AUC-selected hidden | Holdout AUC | AUC-selected pair acc | Same-position output pair acc |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| `mapping_country_token` | `layer_19` | 0.500 | 0.643 | `layer_19` | 0.643 | 0.500 | 0.232 |
| `mapping_value_token` | `layer_11` | 0.393 | 0.644 | `layer_15` | 0.654 | 0.357 | 0.607 |
| `question_country_token` | `layer_20` | 0.607 | 0.800 | `layer_22` | 0.832 | 0.714 | 0.679 |
| `after_mapping_line` | `layer_11` | 0.393 | 0.644 | `layer_15` | 0.654 | 0.357 | 0.607 |
| `after_instruction_line` | `layer_27` | 0.750 | 0.852 | `layer_27` | 0.852 | 0.750 | 0.232 |
| `after_question_line` | `layer_18` | 0.821 | 0.878 | `layer_22` | 0.849 | 0.893 | 0.714 |
| `after_return_line` | `layer_11` | 0.786 | 0.776 | `layer_26` | 0.797 | 0.750 | 0.714 |
| `final_prompt_token` | `layer_25` | 0.679 | 0.884 | `layer_23` | 0.881 | 0.679 | 1.000 |

The strongest pre-output line-boundary monitors appear after the instruction,
question, and return lines. They can beat same-position output on approximate
holdout pairs. That is lead-time structure.

But they do not beat global margins.

| Baseline | Holdout AUC | Holdout pair accuracy |
| --- | ---: | ---: |
| candidate-score margin | 1.000 | 1.000 |
| V19 final next-token margin | 1.000 | 1.000 |
| V22 final-prompt next-token margin | 1.000 | 1.000 |
| prompt token count | 0.500 | 0.589 |

## Interpretation

V22 converts the post-V21 source/path option into a map rather than a rescue.

Allowed claim:

> The current MC006 V19/V20/V21 bank contains source/path and pre-output
> lead-time monitors. The best source/path monitor is at the queried country
> token, while stronger later pre-output line-boundary monitors appear after
> the instruction/question/return lines. These monitors do not support
> intervention, and final candidate/output margins still perfectly order the
> holdout pairs.

Forbidden claims:

- V22 supports an MC006 mechanism signature;
- V22 supports MC006 intervention;
- the `question_country_token` directions are knowledge-control vectors;
- source/path probing neutralizes candidate-score or final-output confounds;
- the V19/V20/V21/V22 row bank is now mechanism-ready;
- line-boundary monitor strength is evidence of causal locality.

## Next Decision

Do not run another scalar probe inside the existing V19/V20/V21/V22 row bank
unless the claim is explicitly a diagnostic map. The source/path audit is now
done enough to support the following branch decision:

1. Generate new MC006 rows targeted at the final-margin boundary, with strict
   candidate and final-output overlap as a behavior-table gate; or
2. Change the intervention family materially and target the later pre-output
   line-boundary monitors as a known-confounded causal stress test, not a
   mechanism-card promotion route.

Until then, MC006 remains:

- behavior-supported;
- lead-time-monitor-supported;
- final-margin-shadowed;
- V17 additive-steering-failed;
- V18/V19/V20/V21 table-rescue-blocked;
- V22 source/path-curve-mapped;
- not signature-ready;
- not intervention-ready.
