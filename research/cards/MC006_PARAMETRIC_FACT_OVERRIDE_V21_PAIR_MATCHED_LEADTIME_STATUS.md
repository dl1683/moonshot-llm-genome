# MC006 Parametric Fact Override V21 Pair-Matched Lead-Time Status

Status: approximate V19/V20 pair matching failed because margin baselines still
perfectly ordered the holdout pairs; no MC006 signature or intervention route
is promoted.

Date: 2026-07-01

## Artifact

- design source:
  `research/21_CONTROL_SURFACE_LAWS_AND_GAPS.md`
- runner:
  `code/mc006_parametric_fact_override_v21_pair_matched_leadtime.py`
- source V19 artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v19_overlapping_margin_table_20260701T023921.json`
- source V19 SHA256:
  `aea4913e580da82f52f301af9360749b8b048d95bcfcb20db59c7411dd5da29f`
- source V20 artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v20_strict_overlap_selection_audit_20260701T025043.json`
- source V20 SHA256:
  `d6f677aab7fe38f681124fc939cd18d8a322c35d694d81750fcf6f76c2598708`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v21_pair_matched_leadtime_20260701T030507.json`
- result SHA256:
  `2a609d24b1bb73df447be6dcdb4aac00adee604e5a7278d682fbf2550035baf3`

## Verdict

V21 does not make MC006 signature-ready.

The diagnostic class is:

```text
approximate_pair_matching_failed_margin_baselines
```

The result is a useful claim-killing diagnostic. V20 allowed only one bounded
use of the existing V19 row bank: approximate pair-matched lead-time auditing,
explicitly not mechanism promotion. V21 ran that audit and found that the
approximate match did not actually neutralize the easy baselines.

## Source Check

V21 validated both source artifacts:

- V19 expected run type: pass;
- V19 expected diagnostic:
  `non_holdout_candidate_margin_overlap_failed`;
- V19 not signature-ready: pass;
- V19 has 400 generated rows: pass;
- V20 expected run type: pass;
- V20 expected diagnostic: `strict_final_margin_overlap_absent`;
- V20 pair-matched diagnostic ready: pass;
- V20 not signature-ready: pass.

The V21 row table contains 257 pooled binary true/override rows from V19:

| Split | True | Override |
| --- | ---: | ---: |
| discovery | 108 | 43 |
| calibration | 30 | 19 |
| holdout | 42 | 15 |

Pooling templates is intentionally treated as a diagnostic upper bound, not a
prompt-controlled promotion substrate.

## Pair-Matched Boundary

V21 used the V20-approved joint match rule:

> true/override row pairs with both candidate-score-margin and final-margin
> z-deltas <= 0.5.

Pair counts:

| Split | Pairs | Participants | Mean max z delta |
| --- | ---: | ---: | ---: |
| non-holdout | 451 | 63 | 0.3496 |
| holdout | 28 | 14 | 0.3436 |

The holdout pair participants were 8 true rows and 6 override rows across 8
prompt templates.

That is enough volume for a bounded diagnostic. It is not enough for a
mechanism claim, because strict final-margin overlap is absent and prompt
template is pooled.

## Hidden Result

The selected pre-output candidate was:

```text
after_mapping_line/layer_14
```

It looked strong on non-holdout matched pairs but failed on holdout:

| Metric | Non-holdout | Holdout |
| --- | ---: | ---: |
| selected hidden pair accuracy | 0.9180 | 0.2857 |
| selected hidden AUC | 0.8069 | 0.5603 |

This is not a stable lead-time surface. The non-holdout pair score overfit the
available pair geometry and reversed on the holdout pairs.

The selected final-reference hidden candidate was:

```text
final_prompt_token/layer_20
```

It also failed to provide a robust pair result:

| Metric | Non-holdout | Holdout |
| --- | ---: | ---: |
| final hidden pair accuracy | 0.8514 | 0.5000 |
| final hidden AUC | 0.7108 | 0.5873 |

## Baseline Result

The easy baselines did not die under approximate pair matching:

| Baseline | Non-holdout pair accuracy | Holdout pair accuracy |
| --- | ---: | ---: |
| candidate-score margin | 0.9313 | 1.0000 |
| V19 final next-token margin | 1.0000 | 1.0000 |
| final-prompt next-token margin | 1.0000 | 1.0000 |
| same-position next-token margin | 0.6364 | 0.6071 |
| prompt token count | 0.6330 | 0.5893 |
| same-position prefix token count | 0.6452 | 0.6786 |

This is the central finding. Even though the pair rows are close in z-space,
the candidate-score and final-output margins still rank every holdout pair in
the behavior-label direction.

Approximate 0.5z matching is therefore too weak a control for MC006 mechanism
promotion.

## Criteria

| Criterion | Result |
| --- | --- |
| source artifacts valid | pass |
| structural table passed | pass |
| strict final-margin overlap absent | pass |
| pair-matched diagnostic ready | pass |
| non-holdout pair count >= 20 | pass |
| holdout pair count >= 20 | pass |
| candidate/final margin holdout pair accuracy <= 0.65 | fail |
| selected hidden holdout pair accuracy >= 0.65 | fail |
| selected hidden beats same-position output by 0.05 | fail |
| selected hidden beats candidate and final margins by 0.05 | fail |
| selected hidden beats prompt-shape baselines by 0.05 | fail |

## Interpretation

V21 converts the V20 "maybe bounded pair diagnostic" branch into a negative
result.

Allowed claim:

> The existing V19 row bank supports approximate pair construction, but those
> pairs remain perfectly ordered by candidate-score and final-output margins on
> holdout. V21 therefore kills approximate pair matching as an MC006 promotion
> route.

Forbidden claims:

- V21 supports an MC006 hidden signature;
- V21 supports MC006 intervention;
- the V21 `after_mapping_line/layer_14` direction is a knowledge surface;
- V19/V20 approximate 0.5z matching neutralizes output/candidate baselines;
- pooled-template pair matching is a prompt-controlled substrate;
- the existing V19 bank can be rescued by another hidden probe without new
  behavior rows or a different causal route.

## Next Step

The current V19/V20/V21 row-bank route is closed for promotion.

The remaining MC006 options are now narrower:

1. Generate new rows targeted directly at the final-margin boundary, with
   strict candidate and final-output overlap as a behavior-table gate.
2. Stop scalar table rescue and audit source/path-specific internals, with
   output/candidate controls reported as the expected downstream shadow.
3. Treat MC006 as a lead-time-monitor-only diagnostic until a new behavior
   table or intervention family changes the evidence.

Until then, MC006 remains:

- behavior-supported by V14;
- lead-time-monitor-supported by V16;
- additive-steering-failed by V17;
- V14 margin-matching-blocked by V18;
- V19 strict-overlap-table-failed;
- V20 strict-overlap-selection-blocked;
- V21 approximate-pair-matching-blocked;
- not signature-ready;
- not intervention-ready.
