# MC003 Delayed-Copy V3 Condition-Balanced Signature Status

Status: condition-balanced signature gate failed; discovery signal did not
generalize to source-disjoint holdout.

Date: 2026-06-30

## Artifacts

- runner: `code/mc003_delayed_copy_v3_signature.py`
- preregistration: `research/prereg/MC003_DELAYED_COPY_V3_SIGNATURE.md`
- behavior status:
  `research/cards/MC003_DELAYED_COPY_V3_CONDITION_BALANCE_STATUS.md`
- behavior result:
  `results/cards/MC003/mc003_gemma2_2b_it_delayed_copy_v3_condition_balance_smoke_chat_20260630T161743.json`
- signature result:
  `results/cards/MC003/mc003_gemma2_2b_it_delayed_copy_v3_condition_balanced_signature_chat_20260630T162002.json`
- signature result SHA256:
  `03d184b85e24948dc633fc58715a8fe0db53a1f30b658efd870d3236a9bdf9e8`

Configuration:

- model: `google/gemma-2-2b-it`
- render mode: `chat`
- selected condition: `wrong_hint_balanced`
- primary rows: 40
- discovery primary rows: 27
- holdout primary rows: 13
- labels: `target_correct` versus `distractor_followed`
- candidate positions:
  - `prompt_end`
  - `after_wait`
  - `after_wait_newline`
- direction: discovery mean difference,
  `mean(target_correct) - mean(distractor_followed)`

## Question

Inside the frozen condition-balanced `wrong_hint_balanced` table, does an early
hidden residual direction predict source-disjoint target-versus-distractor
outcomes better than output margin and null controls?

## Gate Verdict

No.

The discovery split produced high apparent hidden separability, but it did not
hold out. Global discovery selection chose `after_wait` layer 25 with 0.994
discovery AUC and only 0.238 holdout AUC. The same-position output-margin
baseline reached 0.714 holdout AUC, the shuffled-label p95 was 0.690, and both
target-order subgroup AUCs collapsed.

Write this as:

> MC003 V3 removed the condition-trace confound and the hidden signature
> disappeared on holdout. This closes the current delayed-copy branch as a
> behavior-supported but signature-failed result.

## Audit Against Preregistration

| Metric | Result | Bar | Verdict |
| --- | ---: | ---: | --- |
| selected condition | `wrong_hint_balanced` | frozen condition | pass |
| primary rows | 40 | report | pass |
| discovery primary rows | 27 | report | pass |
| holdout primary rows | 13 | report | pass |
| selected position | `after_wait` | selected by discovery AUC | pass |
| selected layer | 25 | selected by discovery AUC | pass |
| selected hidden discovery AUC | 0.994 | report | pass |
| selected hidden holdout AUC | 0.238 | at least 0.85 | fail |
| selected shuffled-label p95 | 0.690 | hidden at least p95 + 0.05 | fail |
| target-first holdout AUC | 0.500 | at least 0.75 | fail |
| distractor-first holdout AUC | 0.250 | at least 0.75 | fail |
| same-position output-margin holdout AUC | 0.714 | hidden must beat by 0.02 | fail |
| single-condition primary rows | true | true | pass |

Position sweep:

| Position | Selected layer | Hidden discovery AUC | Hidden holdout AUC | Output-margin holdout AUC |
| --- | ---: | ---: | ---: | ---: |
| `prompt_end` | 6 | 0.972 | 0.357 | 0.667 |
| `after_wait` | 25 | 0.994 | 0.238 | 0.714 |
| `after_wait_newline` | 24 | 0.978 | 0.381 | 0.690 |

## Interpretation

This is the cleanest MC003 signature test so far because the primary rows come
from one prompt condition. The result is therefore more damaging to the
delayed-copy branch than the earlier failures.

V2 showed large hidden AUCs, but the labels were aligned with condition state.
V3 removed that confound. Once target-correct and distractor-followed examples
coexist inside the same `wrong_hint_balanced` condition, the discovery-selected
hidden directions fail to generalize. The output-margin baseline is not good
enough for a mechanism claim either, but it is still stronger than the selected
hidden direction on holdout.

## Decision

Do not start steering, patching, sparse-feature search, or circuit localization
from MC003 delayed-copy.

The current MC003 evidence supports:

- a real prompt-level behavior transition;
- a condition-balanced behavior table;
- failed hidden-signature promotion once condition confounding is removed.

The next mechanism-card attempt should move to a new behavior family or a
larger condition-balanced source bank. Repeating MC003 hidden-state sweeps on
the current 40-source delayed-copy table is not justified.
