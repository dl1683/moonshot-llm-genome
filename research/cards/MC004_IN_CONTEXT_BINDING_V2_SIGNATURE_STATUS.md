# MC004 In-Context Binding V2 Signature Status

Status: condition-balanced signature gate failed; output margin already
separated the held-out original/update choices.

Date: 2026-06-30

## Artifacts

- runner: `code/mc004_in_context_binding_v2_signature.py`
- preregistration: `research/prereg/MC004_IN_CONTEXT_BINDING_V2_SIGNATURE.md`
- behavior status: `research/cards/MC004_IN_CONTEXT_BINDING_V2_STATUS.md`
- behavior result:
  `results/cards/MC004/mc004_gemma2_2b_it_in_context_binding_v2_smoke_chat_20260630T163531.json`
- signature result:
  `results/cards/MC004/mc004_gemma2_2b_it_in_context_binding_v2_signature_chat_20260630T163803.json`
- signature result SHA256:
  `d0551f3318e96888a202138a4859d1deb175f3ef0ec685299712d677e24c7ecc`

Configuration:

- model: `google/gemma-2-2b-it`
- render mode: `chat`
- selected condition: `update_prefer_latest`
- primary rows: 39 target/update rows, excluding one `other` row
- discovery primary rows: 26
- holdout primary rows: 13
- labels:
  - `target_correct` = original reference note;
  - `distractor_followed` = later update.

## Question

Inside the frozen `update_prefer_latest` condition, does an early hidden
residual direction predict whether the model follows the original note or the
later update better than output margin and null controls?

## Gate Verdict

No.

The selected hidden direction had perfect discovery AUC but did not satisfy
the holdout or control bars. Global discovery selection chose `prompt_end`
layer 20 with 1.000 discovery AUC and 0.738 holdout AUC. The same-position
output-margin baseline reached 1.000 holdout AUC.

Write this as:

> MC004 V2 produced a clean condition-balanced behavior table, but the
> original-versus-update choice was already visible in output logits and the
> selected hidden direction did not pass source-disjoint reliability controls.

## Audit Against Preregistration

| Metric | Result | Bar | Verdict |
| --- | ---: | ---: | --- |
| selected condition | `update_prefer_latest` | frozen condition | pass |
| primary rows | 39 | report | pass |
| discovery primary rows | 26 | report | pass |
| holdout primary rows | 13 | report | pass |
| selected position | `prompt_end` | selected by discovery AUC | pass |
| selected layer | 20 | selected by discovery AUC | pass |
| selected hidden discovery AUC | 1.000 | report | pass |
| selected hidden holdout AUC | 0.738 | at least 0.85 | fail |
| selected shuffled-label p95 | 0.905 | hidden at least p95 + 0.05 | fail |
| target-first holdout AUC | 0.500 | at least 0.75 | fail |
| distractor-first holdout AUC | 0.800 | at least 0.75 | pass |
| same-position output-margin holdout AUC | 1.000 | hidden must beat by 0.02 | fail |
| single-condition primary rows | true | true | pass |

Position sweep:

| Position | Selected layer | Hidden discovery AUC | Hidden holdout AUC | Output-margin holdout AUC |
| --- | ---: | ---: | ---: | ---: |
| `prompt_end` | 20 | 1.000 | 0.738 | 1.000 |
| `after_answer_prefix` | 24 | 0.994 | 0.381 | 0.952 |
| `after_final_prefix` | 18 | 1.000 | 0.738 | 1.000 |

## Interpretation

MC004 is a useful contrast to MC003. MC003 failed because condition-balanced
hidden directions did not generalize. MC004 V2 reached a clean behavior
balance and some hidden layers had nontrivial holdout signal, but the
preselected hidden direction did not clear the preregistered gate and output
logits were already perfect at the prompt end.

This blocks an internal mechanism claim and an intervention preregistration.
The table may still be useful for future output-margin residualization or a
larger source bank, but not as a mechanism-card-ready signature.

## Decision

Do not start steering, patching, sparse-feature search, or circuit localization
from MC004 V2.

The current MC004 evidence supports:

- clean in-context nonce entity/code binding;
- a condition-balanced original-note versus later-update behavior table;
- failed hidden-signature promotion against holdout, null, target-order, and
  output-margin controls.
