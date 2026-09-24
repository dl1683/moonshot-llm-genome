# MC006 Parametric Fact Override V26 Locked-Coordinate Transfer Status

## Status

Diagnostic note. V26 tested the V25 candidate-decoupled hidden monitor under a
stricter selection rule: no free layer search, no free position search, and no
new hidden candidate selection.

The coordinate was locked from V25:

- position/layer: `after_mapping_line/layer_10`;
- source template: `untrusted_note_real`;
- transfer template: `separate_task_weak`.

The locked coordinate reproduced on the source template, but failed to transfer.
It reached only `0.500` holdout AUC on the transfer template, while a
position-local next-token city-margin control reached `0.833` holdout AUC and a
fixed-coordinate train-label shuffle null had p95 `0.833`.

V26 therefore kills the simple repair route "use the V25 coordinate without
search." No MC006 signature or intervention is justified.

## Artifacts

- runner:
  `code/mc006_parametric_fact_override_v26_locked_coordinate_transfer.py`
- V24 source result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v24_delayed_city_interface_20260701T040731.json`
- V25 source result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v25_candidate_decoupled_template_20260701T041843.json`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v26_locked_coordinate_transfer_20260701T043334.json`
- SHA256:
  `0846AF73DCC9DF412FB6131F99FCCF785D6F991D9DC78D94FB69E17F39F98702`

## Source Check

V26 validates both source artifacts before running:

- V24 run type:
  `parametric_fact_override_v24_delayed_city_interface`;
- V24 diagnostic:
  `delayed_city_interface_decouples_first_token_margin`;
- V25 run type:
  `parametric_fact_override_v25_candidate_decoupled_template`;
- V25 diagnostic:
  `candidate_decoupled_hidden_shuffle_overfit`;
- V25 signature ready: false;
- V25 intervention ready: false.

The source coordinate is not selected by V26. It is imported from the V25 result
artifact.

## Source Template Reproduction

Source template: `untrusted_note_real`.

Training split: non-holdout rows.

| Field | Value |
| --- | ---: |
| train rows | 30 |
| train labels | 20 true / 10 override |
| train AUC | 1.000 |
| source holdout rows | 7 |
| source holdout labels | 4 true / 3 override |
| source holdout AUC | 0.750 |
| direction norm | 18.8720 |

The locked coordinate barely clears the source holdout reproduction threshold.
This is weaker than V25 because V26 uses a fixed coordinate and a non-holdout
fit rather than a free selected-search procedure.

## Transfer Test

Transfer template: `separate_task_weak`.

The transfer template is behavior-ready and candidate-score-decoupled under the
V25 criteria:

- binary rows: 30;
- transfer holdout labels: 2 true / 3 override;
- JSON-completion candidate-score holdout AUC: 0.667.

Transfer results:

| Measure | Holdout AUC |
| --- | ---: |
| locked hidden coordinate | 0.500 |
| final next-token city margin | 0.500 |
| city candidate-score margin | 0.333 |
| JSON candidate-score margin | 0.667 |
| `after_mapping_line` next-token city margin | 0.833 |
| fixed-coordinate train-label shuffle p95 | 0.833 |

Transfer gates:

| Gate | Passed |
| --- | --- |
| hidden holdout AUC >= 0.75 | false |
| hidden beats best control holdout | false |
| hidden beats train-label shuffle p95 | false |

## Interpretation

V25 showed that MC006 can produce a candidate-score-decoupled delayed-city table,
but its hidden monitor failed selected-search nulls. V26 asks whether the
obvious repair works: freeze the V25 coordinate and test it on another
candidate-decoupled delayed-city template.

It does not work. The V25 coordinate is at best source-template-local. The
transfer template is not hidden-signal-ready under this coordinate, and the best
position-local output control beats the locked hidden monitor.

This is a stronger negative than V25. The failure is no longer just "too much
hidden search." The discovered coordinate itself does not carry across the only
other candidate-decoupled delayed-city template in the V24 bank.

## Allowed Claims

- The V25 `after_mapping_line/layer_10` coordinate reproduces weakly on its
  source template when fit without free layer/position search.
- The V24 row bank contains at least two candidate-decoupled delayed-city
  templates: `untrusted_note_real` and `separate_task_weak`.
- The locked V25 coordinate does not transfer to `separate_task_weak`.
- V26 exports the diagnostic class `LOCKED_COORDINATE_TRANSFER_FAILED`.

## Forbidden Claims

- V26 is a mechanism card.
- V26 supports any MC006 intervention.
- The V25 coordinate is a reusable MC006 knowledge surface.
- Candidate-score decoupling plus coordinate locking is enough to promote the
  current MC006 hidden monitor.

## Next Decision

Do not steer from V25 or V26. The next MC006 route must either:

1. build a larger candidate-decoupled delayed-city bank with enough independent
   transfer/holdout rows to support a preregistered hidden coordinate test; or
2. stop repairing this coordinate and move to a different behavior family or a
   materially different MC006 intervention stress.

