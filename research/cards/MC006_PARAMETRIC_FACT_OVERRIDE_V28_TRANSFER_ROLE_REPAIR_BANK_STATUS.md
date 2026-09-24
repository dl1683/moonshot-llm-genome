# MC006 Parametric Fact Override V28 Transfer-Role Repair Bank Status

## Status

Diagnostic note. V28 targeted the V27 bottleneck directly: transfer-role prompt
construction for delayed JSON city answers. It did not search hidden states.

The result is another clean negative. V28 found one transfer-role
candidate-decoupled template, but the predeclared gate required at least two.
Combining V28 with V27's source-ready side therefore still does not produce a
hidden-state-ready source/transfer behavior bank.

Diagnostic class:

`transfer_role_repair_bank_insufficient`

Canonical atlas type:

`TRANSFER_ROLE_REPAIR_BANK_INSUFFICIENT`

## Artifacts

- runner:
  `code/mc006_parametric_fact_override_v28_transfer_role_repair_bank.py`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v28_transfer_role_repair_bank_20260701T051821.json`
- SHA256:
  `5E1751880B051EA3A6B84C99E1FBF182E339FC2DC66EAE537112F48D914524FB`

## Source Check

V28 validates V24, V25, V26, and V27 before generation:

- V24 diagnostic:
  `delayed_city_interface_decouples_first_token_margin`;
- V25 diagnostic:
  `candidate_decoupled_hidden_shuffle_overfit`;
- V26 diagnostic:
  `locked_coordinate_transfer_failed`;
- V27 diagnostic:
  `expanded_candidate_decoupled_bank_insufficient`;
- all source artifacts are not signature-ready;
- all source artifacts are not intervention-ready;
- V27 source side has at least two source-ready templates;
- V27 transfer side remains insufficient.

## Design

V28 generated a transfer-only repair grid:

- 40 sources;
- 16 predeclared transfer-role prompt templates;
- 640 generated rows;
- 8 source-disjoint holdout rows per template;
- delayed JSON answer format retained;
- no true answer printed in the prompt.

A transfer template counted as ready only if it passed the same behavior and
candidate-decoupling gate used by V27:

- at least 30 binary true/override rows;
- at least 6 non-holdout true rows;
- at least 6 non-holdout override rows;
- at least 2 holdout true rows;
- at least 2 holdout override rows;
- selected-city first-token rate at most 0.1;
- final city margin not a sign barrier;
- final city margin overlap exists;
- JSON candidate-score holdout AUC is below 1.000.

The repair-bank promotion rule required:

- at least 2 transfer-role candidate-decoupled templates in V28;
- at least 2 source-role ready templates carried from V27;
- at least 5 combined ready templates;
- at least 160 combined pooled binary rows;
- at least 12 combined pooled holdout true rows;
- at least 12 combined pooled holdout override rows.

## Result

V28 failed the transfer-repair gate:

| Criterion | Result |
| --- | --- |
| source artifacts valid | true |
| structural checks passed | true |
| any transfer candidate-decoupled template | true |
| transfer-ready templates >= 2 | false |
| V27 source-ready templates >= 2 | true |
| combined ready templates >= 5 | false |
| combined pooled binary rows >= 160 | false |
| combined pooled holdout true >= 12 | true |
| combined pooled holdout override >= 12 | true |
| transfer repair bank ready | false |

Only one transfer template passed:

| Template | Binary Rows | Holdout Labels | JSON Holdout AUC | Final-City Holdout AUC | City-Candidate Holdout AUC |
| --- | ---: | --- | ---: | ---: | ---: |
| `transfer_sandbox_mapping_then_geo` | 35 | 4 true / 4 override | 0.1875 | 0.3125 | 0.4375 |

Combined with V27's source-ready templates:

| Field | Value |
| --- | ---: |
| source-ready templates | 3 |
| transfer-ready templates | 1 |
| combined ready templates | 4 |
| combined pooled binary rows | 143 |
| combined non-holdout true rows | 61 |
| combined non-holdout override rows | 51 |
| combined holdout true rows | 16 |
| combined holdout override rows | 15 |

## Near Misses

V28 isolates several transfer-role failure modes:

| Template | Failure |
| --- | --- |
| `transfer_previous_answer_key` | behavior-ready, but JSON candidate-score holdout AUC was 1.000 |
| `transfer_prior_exercise_weak` | binary volume high, but holdout was override-only |
| `transfer_old_lookup_row` | binary volume high, but holdout was true-only |
| `transfer_context_switch_short` | binary volume high, but holdout was true-heavy and JSON holdout AUC was 1.000 |
| `transfer_dataset_line_review` | binary volume high, but holdout override count was only 1 |
| `transfer_weak_interference_question` | binary volume high, but holdout override count was only 1 |
| `transfer_training_example_memory` | candidate-score decoupled, but binary volume and holdout balance failed |

## Interpretation

V28 strengthens the MC006 map without promoting a mechanism claim.

It shows that transfer-role candidate decoupling is possible: one predeclared
transfer template passed all behavior, holdout, first-token, final-margin, and
JSON-candidate-score gates. But it also shows that the transfer role is still
too brittle to support hidden-state search. The current prompt family can find
individual transfer candidates, but not yet a transfer-ready bank.

The sharper lesson is that MC006 transfer prompts live in a narrow ambiguity
band. If the prompt is too clear about real-world geography, candidate scoring
or final margins explain the generated labels. If it leans too hard into task
locality or caution, the table becomes parse-fragile or label-imbalanced.

## Allowed Claims

- V28 generated a structurally valid 640-row transfer-only delayed-city bank.
- V28 found one transfer-role candidate-decoupled template:
  `transfer_sandbox_mapping_then_geo`.
- V28 did not produce enough transfer-role templates to repair the MC006
  source/transfer behavior bank.
- Combined V27+V28 ready rows had balanced holdout labels, but insufficient
  ready-template count and pooled binary volume.
- V28 exports the diagnostic class
  `TRANSFER_ROLE_REPAIR_BANK_INSUFFICIENT`.

## Forbidden Claims

- V28 is a mechanism card.
- V28 supports hidden-state probing or intervention.
- V28 repairs the MC006 candidate-decoupled transfer substrate.
- One transfer-ready template is enough for coordinate selection, transfer
  testing, or steering.

## Next Decision

Do not probe V28 hidden states. The current delayed-city MC006 branch has now
failed flexible hidden selection, locked-coordinate transfer, broad expanded
bank construction, and direct transfer-role repair. The next MC006 attempt must
either:

1. preregister a materially different causal stress test that treats the
   monitor as known-confounded; or
2. close the delayed-city route as monitor-only and move to a different
   behavior family or bridge task.
