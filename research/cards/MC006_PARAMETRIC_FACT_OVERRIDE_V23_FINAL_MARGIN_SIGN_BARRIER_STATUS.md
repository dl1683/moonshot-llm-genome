# MC006 Parametric Fact Override V23 Final-Margin Sign-Barrier Status

Status: final next-token margin overlap is structurally blocked under the
current greedy binary generated-answer interface; no MC006 signature or
intervention route is promoted.

Date: 2026-07-01

## Artifact

- design source:
  `research/21_CONTROL_SURFACE_LAWS_AND_GAPS.md`
- runner:
  `code/mc006_parametric_fact_override_v23_final_margin_sign_barrier.py`
- source V14 artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v14_parser_normalized_20260701T000538.json`
- source V18 artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v18_margin_matched_leadtime_20260701T022343.json`
- source V19 artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v19_overlapping_margin_table_20260701T023921.json`
- source V20 artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v20_strict_overlap_selection_audit_20260701T025043.json`
- source V21 artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v21_pair_matched_leadtime_20260701T030507.json`
- source V22 artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v22_source_path_leadtime_curve_20260701T032420.json`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v23_final_margin_sign_barrier_20260701T033829.json`
- result SHA256:
  `f10a727294edc245c30e8677485186282672f447b164a3e39e9455d79607142a`

## Verdict

V23 does not make MC006 signature-ready.

The diagnostic class is:

```text
greedy_final_margin_sign_barrier
```

The result is a claim-killing diagnostic for one previously proposed rescue
route: "just generate rows until strict final-output margin overlap appears."
Under the current greedy generated-answer setup, final next-token margin is not
a neutral confound axis. It is almost the decision boundary for the parsed
binary answer.

## Source Check

V23 validated the current MC006 chain:

| Source | Expected diagnostic | Valid |
| --- | --- | --- |
| V14 | `parser_normalized_generated_substrate_passed` | yes |
| V18 | `global_margin_separation_blocks_matching` | yes |
| V19 | `non_holdout_candidate_margin_overlap_failed` | yes |
| V20 | `strict_final_margin_overlap_absent` | yes |
| V21 | `approximate_pair_matching_failed_margin_baselines` | yes |
| V22 | `source_path_final_margin_shadow` | yes |

## Main Result

V23 audited final true-minus-override next-token margins on two generated
binary row sets:

| Row set | Binary rows | True rows | Override rows | Final sign accuracy | Raw overlap | Raw gap |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| V18 / V14 selected table | 30 | 21 | 9 | 1.000 | no | 1.500 |
| V19 all binary rows | 257 | 180 | 77 | 1.000 | no | 0.375 |
| V19 selected template | 35 | 15 | 20 | 1.000 | no | 0.375 |

For V19 all binary rows:

- true rows with positive final margin: 180/180;
- true rows with non-positive final margin: 0/180;
- override rows with non-positive final margin: 77/77;
- override rows with positive final margin: 0/77.

For V18 / V14 selected rows:

- true rows with positive final margin: 21/21;
- true rows with non-positive final margin: 0/21;
- override rows with non-positive final margin: 9/9;
- override rows with positive final margin: 0/9.

That is the sign barrier.

## Candidate-Score Contrast

Candidate-score margin did not show the same hard barrier.

For V19 all binary rows:

- candidate-score sign accuracy: 0.953;
- candidate-score raw overlap: yes;
- true rows with non-positive candidate-score margin: 12/180;
- override rows with positive candidate-score margin: 0/77.

This matters because it separates two baselines:

- candidate-score margin can overlap and is a useful control axis;
- final next-token margin is coupled to the greedy binary generated label.

The current final-margin problem is therefore not just "candidate-score
confounding." It is an output-interface gate problem.

## Token Alignment

The parsed selected answer usually starts with the same first token that V15
uses for final-margin scoring:

| Row set | Aligned rows | Checked rows | Accuracy |
| --- | ---: | ---: | ---: |
| V14 selected template | 30 | 30 | 1.000 |
| V19 selected template | 35 | 35 | 1.000 |
| V19 all binary rows | 255 | 257 | 0.992 |

The two V19 all-row exceptions were both `Copenhagen`, where generation began
with token `" C"` while the answer-token helper used `" Copenhagen"`. Even
with those tokenization exceptions, final margin sign still predicted the
parsed binary label on 257/257 rows.

## Interpretation

V23 changes the MC006 doctrine.

Before V23, the project treated strict final-output overlap as a hard row-table
repair target. V23 shows why that target is probably malformed for this
specific generated-answer interface. If a binary row is produced by greedy
generation and parsed from the first answer token, requiring strict overlap in
the first-token true-minus-override margin asks the row table to violate the
same output geometry that produced the label.

Allowed claim:

> In the current MC006 greedy generated-answer contracts, final next-token
> margin behaves as a sign boundary for parsed true/override labels. Strict
> final-margin overlap is therefore not a good default promotion gate for this
> interface. Future MC006 work should change the answer interface or use a
> different control design rather than broadening prompt search under the same
> greedy first-token gate.

Forbidden claims:

- V23 supports an MC006 hidden signature;
- V23 supports MC006 intervention;
- V23 proves final-output controls can be ignored;
- V23 proves knowledge retrieval is non-causal or purely output-level;
- V23 proves strict final-margin overlap is impossible for all models or all
  generated-answer interfaces;
- more prompt variants under the same greedy binary first-token interface are
  likely to rescue strict final-margin overlap.

## Next Decision

Stop treating "strict final next-token margin overlap" as the primary MC006 row
generation target under the current greedy generated-answer interface.

The next MC006 experiment should choose one of two materially different
routes:

1. Change the answer interface so the label is not determined by the same
   first-token true-minus-override margin being used as the control. Examples:
   forced-choice hidden scoring without greedy first-token labels,
   delayed-answer formats, multi-token scoring, or non-greedy sampling with
   explicit label adjudication.
2. Keep greedy generation, but replace strict final-margin overlap with a
   preregistered boundary-band diagnostic: near-zero final margins, candidate
   overlap, source/path lead-time, and causal stress all reported as bounded
   diagnostics rather than mechanism promotion.

Until one of those routes is implemented, MC006 remains:

- behavior-supported;
- lead-time-monitor-supported;
- final-margin-sign-barrier-diagnosed;
- not signature-ready;
- not intervention-ready.
