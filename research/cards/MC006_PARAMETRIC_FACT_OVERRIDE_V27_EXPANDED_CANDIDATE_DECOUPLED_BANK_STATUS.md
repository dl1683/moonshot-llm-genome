# MC006 Parametric Fact Override V27 Expanded Candidate-Decoupled Bank Status

## Status

Diagnostic note. V27 tested whether the current delayed-city MC006 prompt
family can build a larger candidate-decoupled source/transfer bank after V26
showed that the V25 coordinate does not transfer.

The answer is partial. V27 found four candidate-decoupled delayed-city
templates and enough pooled binary/holdout volume, but only one of the four was
a predeclared transfer-role template. The expanded bank is therefore not ready
for hidden-state probing or intervention.

Diagnostic class:

`expanded_candidate_decoupled_bank_insufficient`

Canonical atlas type:

`EXPANDED_CANDIDATE_DECOUPLED_BANK_INSUFFICIENT`

## Artifacts

- runner:
  `code/mc006_parametric_fact_override_v27_expanded_candidate_decoupled_bank.py`
- result:
  `results/cards/MC006/mc006_qwen3_1p7b_parametric_fact_override_v27_expanded_candidate_decoupled_bank_20260701T045409.json`
- SHA256:
  `2800010D546D992796154DF58ED616EBF6ABCD7EB30B61B33ED7538E908540D1`

## Source Check

V27 validates the V24, V25, and V26 source artifacts before generation:

- V24 diagnostic:
  `delayed_city_interface_decouples_first_token_margin`;
- V25 diagnostic:
  `candidate_decoupled_hidden_shuffle_overfit`;
- V26 diagnostic:
  `locked_coordinate_transfer_failed`;
- all three source artifacts are not signature-ready;
- all three source artifacts are not intervention-ready.

## Design

V27 generated a 640-row delayed-city bank:

- 40 sources;
- 16 prompt templates;
- 8 predeclared source-role templates;
- 8 predeclared transfer-role templates;
- 8 source-disjoint holdout rows per template;
- JSON answer format retained, so the first generated token is not the city.

A template counted as candidate-decoupled only if it passed all behavior and
evaluation criteria:

- at least 30 binary true/override rows;
- at least 6 non-holdout true rows;
- at least 6 non-holdout override rows;
- at least 2 holdout true rows;
- at least 2 holdout override rows;
- selected-city first-token rate at most 0.1;
- final city margin not a sign barrier;
- final city margin overlap exists;
- JSON candidate-score holdout AUC is below 1.000.

The expanded bank promotion rule required:

- at least 4 candidate-decoupled templates;
- at least 2 source-role candidate-decoupled templates;
- at least 2 transfer-role candidate-decoupled templates;
- at least 120 pooled binary rows;
- at least 8 pooled holdout true rows;
- at least 8 pooled holdout override rows.

## Result

V27 passed most bank-size criteria:

| Criterion | Result |
| --- | --- |
| source artifacts valid | true |
| structural checks passed | true |
| any candidate-decoupled template | true |
| candidate-decoupled templates >= 4 | true |
| source-role ready templates >= 2 | true |
| transfer-role ready templates >= 2 | false |
| pooled binary rows >= 120 | true |
| pooled holdout true >= 8 | true |
| pooled holdout override >= 8 | true |
| expanded bank ready | false |

Ready templates:

| Template | Role | Binary Rows | Holdout Labels | JSON Holdout AUC |
| --- | --- | ---: | --- | ---: |
| `source_untrusted_note_real_v24` | source | 37 | 4 true / 3 override | 0.333 |
| `source_unverified_note_check` | source | 37 | 3 true / 5 override | 0.533 |
| `source_mistaken_source` | source | 34 | 5 true / 3 override | 0.600 |
| `transfer_separate_task_weak_v24` | transfer | 30 | 2 true / 3 override | 0.667 |

Pooled ready bank:

| Field | Value |
| --- | ---: |
| ready templates | 4 |
| source-ready templates | 3 |
| transfer-ready templates | 1 |
| pooled binary rows | 138 |
| pooled non-holdout true rows | 59 |
| pooled non-holdout override rows | 51 |
| pooled holdout true rows | 14 |
| pooled holdout override rows | 14 |

## Failed Transfer Templates

The transfer-role side failed for two different reasons:

- several transfer templates were behavior-ready but still JSON-candidate-score
  visible on holdout;
- several transfer templates were not behavior-ready because they were
  true-heavy, override-heavy, or parse-fragile.

Examples:

| Template | Failure |
| --- | --- |
| `transfer_game_code_real_json` | behavior-ready, but JSON holdout AUC 1.000 |
| `transfer_puzzle_lookup_then_geo` | behavior-ready, but JSON holdout AUC 1.000 |
| `transfer_memory_interference_light` | holdout coverage present, but JSON holdout AUC 1.000 |
| `transfer_old_flashcard` | true-heavy holdout |
| `transfer_data_line_maybe_wrong` | true-heavy holdout |
| `transfer_ambiguous_note_json` | override-heavy holdout |

## Interpretation

V27 improves the MC006 map in a specific way. V26 made clear that the small V25
candidate-decoupled bank could not support locked-coordinate transfer. V27 then
asked whether the same broad prompt family can build a larger bank.

It can build the source side. It cannot yet build the transfer side.

That means the next MC006 repair should not search hidden states and should not
try another coordinate on this bank. The current bottleneck is transfer-role
prompt construction: the model either makes transfer prompts candidate-visible
or collapses their holdout labels into one side.

## Allowed Claims

- A 640-row delayed-city MC006 prompt bank can produce four candidate-decoupled
  templates.
- The expanded bank has enough pooled binary and holdout true/override volume.
- The source-role side is repairable under this prompt family: three source-role
  templates passed.
- The transfer-role side is not yet repairable under this V27 prompt family:
  only one transfer-role template passed.
- V27 exports the diagnostic class
  `EXPANDED_CANDIDATE_DECOUPLED_BANK_INSUFFICIENT`.

## Forbidden Claims

- V27 is a mechanism card.
- V27 supports hidden-state probing or intervention.
- V27 repairs the MC006 candidate-decoupled transfer substrate.
- The current V27 bank is ready for coordinate selection, transfer testing, or
  steering.

## Next Decision

Do not probe V27 hidden states. The next MC006 route must target transfer-role
prompt construction directly: generate transfer-role templates that preserve
true/override holdout balance while keeping JSON candidate-score holdout AUC
below 1.000. If that cannot be done, close the current MC006 delayed-city route
as a monitor-only diagnostic branch.

