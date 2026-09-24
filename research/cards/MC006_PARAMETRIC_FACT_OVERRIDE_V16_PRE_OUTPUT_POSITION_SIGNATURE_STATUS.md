# MC006 Parametric Fact Override V16 Pre-Output Position Signature Status

Status: pre-output lead-time signal found, but mechanism promotion failed
because global output-interface controls still match the selected hidden
direction.

Date: 2026-07-01

## Artifact

- preregistration:
  `research/prereg/MC006_PARAMETRIC_FACT_OVERRIDE_V16_PRE_OUTPUT_POSITION_SIGNATURE.md`
- runner:
  `code/mc006_parametric_fact_override_v16_pre_output_position_signature.py`
- source behavior artifact:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v14_parser_normalized_20260701T000538.json`
- source artifact SHA256:
  `7596e8c6869cfcb7413e9bbe8d9b6324b2ed3115021bbb1bf7091059cbac54c5`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v16_pre_output_position_signature_20260701T002113.json`
- result SHA256:
  `6c99826028de85d9b03b22cd8c467ccc42483e1b80ca4f1bbcab6fc67074eb31`

## Verdict

MC006 V16 supports a narrow pre-output lead-time diagnostic signal, but it does
not support intervention.

The diagnostic class is:

```text
leadtime_signal_supported_but_output_global_confounded
```

The selected hidden direction appears before the final `Answer:` interface and
beats the same-position next-token output margin, prompt/token controls, and
the shuffled-selection null. However, candidate-score margin and final
next-token output margin still reach perfect holdout AUC, so the signal is not
mechanism-grade.

## Structural Checks

Structural checks passed for the selected V14 table:

- 30 binary rows;
- 10 side rows;
- 21 true-answer rows;
- 9 override-answer rows;
- discovery: 11 true, 4 override;
- calibration: 4 true, 3 override;
- holdout: 6 true, 2 override;
- unique source ids;
- no duplicate record ids.

Side rows were 3 lure answers and 7 unparsed rows.

## Position Contract

V16 scored hidden states at five prompt positions:

- `after_mapping_line`;
- `after_instruction_line`;
- `after_question_line`;
- `after_return_line`;
- `final_prompt_token`.

Only the first four were selectable. `final_prompt_token` was a reference
position because V15 had already shown that the final answer interface was
candidate-score and output-margin confounded.

Positions were mapped with the fast-tokenizer full-prompt offset map and no
special tokens. The full prompt was run once per row without padding.

## Signature Results

Selected pre-output hidden candidate:

- `after_mapping_line/layer_4`;
- discovery AUC: 1.000;
- holdout AUC: 1.000.

Final-position reference candidate:

- `final_prompt_token/layer_16`;
- discovery AUC: 1.000;
- holdout AUC: 1.000.

Selected-position baselines:

| Baseline | Discovery AUC | Holdout AUC |
| --- | ---: | ---: |
| same-position next-token output margin | 0.652 | 0.917 |
| prefix token count | 0.643 | 0.583 |
| position token id | 0.610 | 0.333 |

Global output-interface baselines:

| Baseline | Discovery AUC | Holdout AUC |
| --- | ---: | ---: |
| candidate-score margin | 0.990 | 1.000 |
| final next-token output margin | 1.000 | 1.000 |
| prompt length | 0.643 | 0.583 |
| final prompt token id | 0.500 | 0.500 |
| true candidate token count | 0.667 | 0.375 |
| override candidate token count | 0.643 | 0.583 |
| generated first-token id | 0.838 | 0.833 |
| parser-normalization delta | 0.533 | 0.500 |

Shuffled-label selection null:

- iterations: 100;
- holdout AUC p95: 0.917;
- holdout AUC max: 1.000;
- holdout AUC mean: 0.509;
- top selected shuffled candidate: `after_mapping_line/layer_0` in 100/100
  shuffles.

## Criteria

### Lead-Time Criteria

| Criterion | Result |
| --- | --- |
| source artifact and structural checks pass | pass |
| holdout has at least 2 rows per label | pass |
| selected pre-output hidden holdout AUC at least 0.85 | pass: 1.000 |
| selected hidden beats same-position output by 0.02 | pass: 1.000 vs 0.917 |
| selected hidden beats prefix token count by 0.02 | pass: 1.000 vs 0.583 |
| selected hidden beats position token id by 0.02 | pass: 1.000 vs 0.333 |
| selected hidden beats shuffled-selection p95 by 0.05 | pass: 1.000 vs 0.917 |

### Mechanism Criteria

| Criterion | Result |
| --- | --- |
| all lead-time criteria pass | pass |
| selected hidden beats candidate-score margin by 0.02 | fail: both 1.000 |
| selected hidden beats final next-token output by 0.02 | fail: both 1.000 |

## Interpretation

V16 is the first MC006 result that finds a hidden signal before the final answer
interface and shows that it is stronger than the same-position output-logit
view at that earlier point. That matters: the model has an internal
true-versus-override trace at the fake-mapping-line boundary, before the
explicit instruction, question, return-format line, or final `Answer:` marker.

But the stricter mechanism claim remains blocked. The completed prompt's
candidate-score margin and final next-token output margin still separate the
labels perfectly. V16 therefore supports a lead-time monitoring claim, not a
causal control-surface claim.

MC006 should now be bounded as:

- V14: matched generated behavior substrate passed under a narrow normalized
  parser;
- V15: final-prompt-token hidden signal present but final output-interface
  baselines match it;
- V16: earlier hidden signal present and stronger than same-position output
  margin, but still globally output-confounded.

The next MC006 step should not be intervention unless the project accepts an
explicit monitoring-only target. For mechanism work, the next repair should
either construct labels that are not perfectly visible to final candidate-score
and output margins, or use V16 as a donor for an intervention preregistration
that treats global output confounding as a known failure mode rather than as a
passed signature gate.
