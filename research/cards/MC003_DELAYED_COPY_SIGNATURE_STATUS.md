# MC003 Delayed-Copy Signature Status

Status: diagnostic signature found; mechanism-signature gate failed against
the output-margin control.

Date: 2026-06-30

## Artifacts

- runner: `code/mc003_delayed_copy_signature.py`
- preregistration: `research/prereg/MC003_DELAYED_COPY_SIGNATURE.md`
- behavior substrate: `research/cards/MC003_DELAYED_COPY_STATUS.md`
- behavior result: `results/cards/MC003/mc003_gemma2_2b_it_delayed_copy_v2_smoke_chat_20260630T155437.json`
- signature result: `results/cards/MC003/mc003_gemma2_2b_it_delayed_copy_v2_signature_chat_20260630T155831.json`
- signature result SHA256: `f2d082333f950941ea22519ca734245d267cf0994d3ce3f7f6e273426acd4283`

Configuration:

- model: `google/gemma-2-2b-it`
- render mode: `chat`
- answer prefix: `WAIT\nFINAL:`
- primary rows: 200 V2 rows, excluding direct `wrong_hint_forced`
- discovery primary rows: 135
- holdout primary rows: 65
- label: `target_correct` versus `distractor_followed`
- hidden feature: residual-stream state at the final answer-prefix token
- direction: discovery mean difference,
  `mean(target_correct) - mean(distractor_followed)`

## Question

Does a hidden residual direction at the delayed final-answer prefix predict the
target-versus-distractor output choice on source-disjoint holdout rows, beyond
simple output logits?

## Gate Verdict

No, not as a mechanism-signature gate.

The hidden direction is real as a diagnostic correlate. The selected layer 22
direction reached 0.994 holdout AUC, beat the selected-layer shuffled-label
p95 by more than 0.05, and held in both target-order subgroups. But the
first-token output-margin control reached 1.000 holdout AUC, so the hidden
signature did not beat the output-only baseline.

Write this as:

> MC003 has a strong residual correlate of delayed-copy behavior, but the
> behavior is already visible in output logits at the teacher-forced final
> prefix. Do not promote this to an internal mechanism signature.

## Audit Against Preregistration

| Metric | Result | Bar | Verdict |
| --- | ---: | ---: | --- |
| selected layer | 22 | selected by discovery AUC | pass |
| selected-layer discovery AUC | 0.998 | report | pass |
| selected-layer holdout AUC | 0.994 | at least 0.85 | pass |
| shuffled-label selected-layer p95 | 0.918 | hidden at least p95 + 0.05 | pass |
| target-first holdout AUC | 0.983 | at least 0.75 | pass |
| distractor-first holdout AUC | 1.000 | at least 0.75 | pass |
| first-token output-margin holdout AUC | 1.000 | hidden must beat by 0.02 | fail |

## Interpretation

The hidden direction passes the basic diagnostic tests. It is source-disjoint,
not explained by the shuffled-label null, and not confined to one target-order
subgroup. That makes it a useful instrument reading.

The output-margin control blocks promotion. At the same `WAIT\nFINAL:` prefix,
the target-minus-distractor first-token logit margin separates the held-out
target-correct and distractor-followed rows perfectly. A downstream residual
direction that does not beat this baseline cannot support a mechanism-card
claim.

## Decision

Do not start steering, patching, sparse-feature search, or circuit localization
from the MC003 V2 residual direction.

The next MC003 attempt should move the signature earlier if this behavior line
continues:

- probe before the `WAIT` token instead of after the teacher-forced `FINAL:`
  prefix;
- test whether any hidden signature predicts the final choice before output
  logits expose it;
- include output-margin controls at every candidate position;
- keep target-order subgroup controls.

Until an earlier hidden signature beats the output/trace baseline, MC003 is a
behavior-supported and diagnostic-signature-only artifact.
