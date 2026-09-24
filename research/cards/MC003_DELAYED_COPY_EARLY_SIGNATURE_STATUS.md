# MC003 Delayed-Copy Early Signature Status

Status: early hidden signal found; mechanism-signature gate failed against the
shuffled-label null and condition-trace control.

Date: 2026-06-30

## Artifacts

- runner: `code/mc003_delayed_copy_early_signature.py`
- preregistration: `research/prereg/MC003_DELAYED_COPY_EARLY_SIGNATURE.md`
- behavior substrate: `research/cards/MC003_DELAYED_COPY_STATUS.md`
- prior final-prefix signature status:
  `research/cards/MC003_DELAYED_COPY_SIGNATURE_STATUS.md`
- behavior result:
  `results/cards/MC003/mc003_gemma2_2b_it_delayed_copy_v2_smoke_chat_20260630T155437.json`
- early signature result:
  `results/cards/MC003/mc003_gemma2_2b_it_delayed_copy_v2_early_signature_chat_20260630T160612.json`
- early signature result SHA256:
  `3d324597bfa2346e25085d182df45cd1bc727ba2d2ff4e0eddf851e37799224d`

Configuration:

- model: `google/gemma-2-2b-it`
- render mode: `chat`
- primary rows: 200 MC003 V2 rows, excluding direct `wrong_hint_forced`
- discovery primary rows: 135
- holdout primary rows: 65
- labels: `target_correct` versus `distractor_followed`
- candidate positions:
  - `prompt_end`: final rendered prompt token before `WAIT`
  - `after_wait`: after teacher-forced `WAIT`
  - `after_wait_newline`: after teacher-forced `WAIT\n`, before `FINAL`
- direction: discovery mean difference,
  `mean(target_correct) - mean(distractor_followed)`

## Question

Does an earlier hidden residual direction predict the final delayed-copy choice
before output logits or prompt-condition traces already explain the behavior?

## Gate Verdict

No.

The selected early hidden direction is strong in raw AUC terms. The global
discovery selection chose `after_wait_newline` layer 14, with 0.996 discovery
AUC and 0.997 source-disjoint holdout AUC. It also beat the same-position
first-token output-margin control, which reached 0.913 holdout AUC.

That is still not a mechanism-signature pass. The selected position/layer
shuffled-label null reached 0.998 p95 on holdout, slightly above the selected
hidden direction, and the simple condition-trace baseline reached 0.991 holdout
AUC. The hidden direction did not clear either preregistered reliability
control.

Write this as:

> MC003 has early residual signals that track delayed-copy outcomes, but the
> current behavior table is too condition-confounded to support an internal
> signature claim. Do not start intervention work from this direction.

## Audit Against Preregistration

| Metric | Result | Bar | Verdict |
| --- | ---: | ---: | --- |
| selected position | `after_wait_newline` | selected by discovery AUC | pass |
| selected layer | 14 | selected by discovery AUC | pass |
| selected hidden discovery AUC | 0.996 | report | pass |
| selected hidden holdout AUC | 0.997 | at least 0.85 | pass |
| selected shuffled-label p95 | 0.998 | hidden at least p95 + 0.05 | fail |
| target-first holdout AUC | 1.000 | at least 0.75 | pass |
| distractor-first holdout AUC | 1.000 | at least 0.75 | pass |
| same-position output-margin holdout AUC | 0.913 | hidden must beat by 0.02 | pass |
| condition-trace holdout AUC | 0.991 | hidden must beat by 0.02 | fail |

Position sweep:

| Position | Selected layer | Hidden discovery AUC | Hidden holdout AUC | Output-margin holdout AUC |
| --- | ---: | ---: | ---: | ---: |
| `prompt_end` | 0 | 0.993 | 0.981 | 0.775 |
| `after_wait` | 7 | 0.991 | 0.991 | 0.903 |
| `after_wait_newline` | 14 | 0.996 | 0.997 | 0.913 |

## Interpretation

This run fixed one problem from the final-prefix signature attempt but exposed
another.

The output-margin control is no longer the blocker. At earlier positions,
first-token logits do not fully explain the hidden signal.

The condition structure is the blocker. Primary positive rows are mostly
non-pressure rows, while primary negative rows are mostly `wrong_hint_pressure`
rows. A simple trace score that knows only whether the row is pressure or
non-pressure reaches 0.991 holdout AUC. The selected residual direction may
therefore be detecting prompt condition or pressure state, not the delayed
target-versus-distractor computation.

The shuffled-label null strengthens that objection. At the selected
position/layer, random discovery labels can produce holdout AUC as high as
0.998 p95. With this row count and table structure, high AUC alone is not
specific enough to promote a hidden signature.

## Decision

Do not start steering, patching, sparse-feature search, or circuit localization
from the MC003 V2 early residual direction.

The next MC003 attempt must rebuild the signature table before intervention:

- create target-correct and distractor-followed variation within the same
  prompt condition, especially within `wrong_hint_pressure`;
- balance pressure/non-pressure traces across labels or residualize them before
  selection;
- keep source-disjoint holdouts and target-order subgroup checks;
- require the selected hidden signature to beat both the shuffled-label null
  and condition-trace control before any intervention gate.

Until that condition-balanced table exists, MC003 remains a behavior-supported
and failed-signature artifact, not an intervention-ready mechanism.
