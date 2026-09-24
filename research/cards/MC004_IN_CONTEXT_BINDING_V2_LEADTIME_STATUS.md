# MC004 In-Context Binding V2 Lead-Time Status

Status: pre-answer diagnostic signal found; lead-time signature gate failed
against shuffled-label and target-order controls.

Date: 2026-06-30

## Artifacts

- runner: `code/mc004_in_context_binding_v2_leadtime.py`
- preregistration: `research/prereg/MC004_IN_CONTEXT_BINDING_V2_LEADTIME.md`
- behavior status: `research/cards/MC004_IN_CONTEXT_BINDING_V2_STATUS.md`
- signature status: `research/cards/MC004_IN_CONTEXT_BINDING_V2_SIGNATURE_STATUS.md`
- behavior result:
  `results/cards/MC004/mc004_gemma2_2b_it_in_context_binding_v2_smoke_chat_20260630T163531.json`
- lead-time result:
  `results/cards/MC004/mc004_gemma2_2b_it_in_context_binding_v2_leadtime_chat_20260630T164607.json`
- lead-time result SHA256:
  `fa05648194dc1d63f9bc7fe03124d89e9b4bc5919df3c72a441eed3f60ff242a`

Configuration:

- model: `google/gemma-2-2b-it`
- render mode: `chat`
- selected condition: `update_prefer_latest`
- primary rows: 39 target/update rows, excluding one `other` row
- discovery primary rows: 26
- holdout primary rows: 13
- stages:
  - `after_notes`
  - `after_question`
  - `after_update`
  - `after_answer_instruction`

## Question

Before the final answer instruction, does a hidden residual signal predict
whether the model will use the original reference note or the later update, and
does it beat same-stage output margin and null controls?

## Gate Verdict

No.

The selected signal is interesting but not reliable enough. Global discovery
selection chose `after_question` layer 23, before the later-update instruction,
with 1.000 discovery AUC and 0.905 source-disjoint holdout AUC. Same-stage
output margin was weak at 0.262 holdout AUC, so the hidden signal did beat the
output baseline at that stage.

The result still failed preregistered reliability. The selected shuffled-label
p95 was 0.929, above the selected hidden holdout AUC, and the target-first
holdout subgroup reached only 0.667 AUC.

Write this as:

> MC004 V2 has a pre-answer diagnostic signal before the update instruction,
> but the current 39-row table cannot support a lead-time mechanism claim. The
> signal may reflect source/order regularities rather than a robust internal
> update-choice state.

## Audit Against Preregistration

| Metric | Result | Bar | Verdict |
| --- | ---: | ---: | --- |
| selected stage | `after_question` | before answer instruction | pass |
| selected layer | 23 | selected by discovery AUC | pass |
| selected hidden discovery AUC | 1.000 | report | pass |
| selected hidden holdout AUC | 0.905 | at least 0.85 | pass |
| selected shuffled-label p95 | 0.929 | hidden at least p95 + 0.05 | fail |
| target-first holdout AUC | 0.667 | at least 0.75 | fail |
| distractor-first holdout AUC | 0.800 | at least 0.75 | pass |
| same-stage output-margin holdout AUC | 0.262 | hidden must beat by 0.02 | pass |
| single-condition primary rows | true | true | pass |

Stage sweep:

| Stage | Selected layer | Hidden discovery AUC | Hidden holdout AUC | Output-margin holdout AUC |
| --- | ---: | ---: | ---: | ---: |
| `after_notes` | 24 | 0.970 | 0.690 | 0.143 |
| `after_question` | 23 | 1.000 | 0.905 | 0.262 |
| `after_update` | 22 | 1.000 | 0.619 | 0.952 |
| `after_answer_instruction` | 20 | 1.000 | 0.738 | 1.000 |

## Interpretation

The lead-time audit separates two failure modes:

- after the update instruction, output margin is already strong;
- before the update instruction, hidden-state separation exists but is not
  robust to shuffled-label and target-order controls.

That pattern argues against intervention on this signal. A mechanism-ready
lead-time claim would need a larger source bank, balanced target-order support,
and a null that cannot match the selected layer.

## Decision

Do not start intervention work from the MC004 V2 lead-time direction.

The result should be retained as a diagnostic clue for future larger-bank
update-conflict experiments, not as a mechanism-card-ready signature.
