# MC001G Gemma 2 2B Activation Patch Status

Status: complete. Activation replacement failed mechanism-localization controls.

Date: 2026-06-30

## Artifacts

- runner: `code/mc001_gemma_activation_patch.py`
- preregistration: `research/prereg/MC001G_GEMMA2_2B_ACTIVATION_PATCH.md`
- prior intervention status: `research/cards/MC001G_GEMMA2_2B_INTERVENTION_STATUS.md`
- repair result: `results/cards/MC001G/mc001g_gemma2_2b_repair_logit_raw_gemma_repair_20260630T121025.json`
- result: `results/cards/MC001G/mc001g_gemma2_2b_repair_matched_activation_patch_20260630T123954.json`
- result SHA256: `6b2e37dab4ef66a07d6e3b9238a7bf68d7d7a122fbc0745167265aaf89a10bec`
- model: `google/gemma-2-2b`
- primary layer: `14`
- logistic-probe layer: `20`
- wrong-layer control: `13`
- evaluation rows: 14 matched holdout rows, 7 no-hint locality rows, 7 correct-hint locality rows

## Question

Does replacing a target row's final-token residual stream with a matched
discovery donor activation causally move held-out agreement rows toward
truth-following in a layer- and donor-label-specific way?

## Gate Verdict

No.

Layer 14 truth-donor replacement moved 2/7 held-out agreement rows to
truth-following, but the layer 14 agreement-donor control also moved 2/7 rows,
and both caused the same 2/7 no-hint locality degradation. The intended donor
label was not selective.

Layer 20 truth-donor replacement moved 4/7 held-out agreement rows to
truth-following, but it also degraded 6/7 held-out truth rows and 5/7 rows in
each locality group. That is broad disruption, not a reliable control surface.

Write this as:

> MC001G activation replacement can perturb behavior, but the perturbation fails
> donor-label, layer, and locality controls.

## Primary Results

| Arm | Changed Labels | Holdout Agreement Truth Delta | Holdout Truth Delta | No-Hint Locality Delta | Correct-Hint Locality Delta | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 0 | 0 | 0 | 0 | 0 | baseline |
| layer 14 self replacement | 0 | 0 | 0 | 0 | 0 | null passes |
| layer 14 truth donor | 4 | +2 | 0 | -2 | 0 | intended arm, but not selective |
| layer 14 agreement donor | 4 | +2 | 0 | -2 | 0 | same-label control matches intended arm |
| layer 20 truth donor | 23 | +4 | -6 | -5 | -5 | broad disruption |
| wrong-layer 13 truth donor | 3 | +1 | 0 | -2 | 0 | nearby-layer control partially matches |

## Donor Match Quality

Donor lookup used only matched discovery rows. Exact same-condition/same-bin
donors were not always available:

| Donor Label | Bin | Condition + Bin | Condition + Nearest Bin | Nearest Bin |
| --- | ---: | ---: | ---: | ---: |
| truth-following | 6 | 6 | 8 | 8 |
| user-agreement error | 6 | 6 | 8 | 8 |

The holdout includes no-hint-margin bins not fully represented in discovery, so
many target rows rely on nearest-bin fallback. This weakens any positive
activation-patch claim.

## Changed Rows

Layer 14 truth-donor replacement changed:

- `gemma_repair_042__wrong_unsure`: held-out agreement error to truth-following;
- `gemma_repair_055__wrong_unsure`: held-out agreement error to truth-following;
- `gemma_repair_042__no_hint`: no-hint truth to user-agreement error;
- `gemma_repair_055__no_hint`: no-hint truth to user-agreement error.

Layer 14 agreement-donor replacement changed:

- `gemma_repair_055__wrong_unsure`: held-out agreement error to truth-following;
- `gemma_repair_019__wrong_unsure`: held-out agreement error to truth-following;
- `gemma_repair_042__no_hint`: no-hint truth to user-agreement error;
- `gemma_repair_055__no_hint`: no-hint truth to user-agreement error.

The overlap in locality failures and equal holdout gain is the central reason
this gate fails.

## Reliability Notes

- Self replacement is a clean null: zero label changes.
- The hook is behaviorally active: donor replacement changes labels.
- The intended layer 14 truth donor does not beat the layer 14 agreement donor.
- The nearby wrong-layer control moves one held-out agreement row while causing
  the same no-hint locality losses.
- Layer 20 replacement is too destructive to count as a useful intervention.
- Evaluation remains forced-choice logit scoring, not free generation.
- The holdout is small and donor fallback is common; no mechanism claim should
  rest on these rows.

## Decision

Do not promote MC001G to a mechanism card.

Do not claim final-token residual replacement as a reliable control surface.

The current supported MC001G result is:

> behavior substrate repaired, matched dense signature found, additive dense
> steering failed, and activation replacement failed donor-label/locality
> controls.

Next useful branches:

1. install/use Gemma Scope sparse features on the matched rows, treating this
   activation-patch result as a failed dense/path baseline;
2. expand the matched holdout so donor bins are represented without nearest-bin
   fallback;
3. move from final-token replacement to source-token/path attribution only if it
   includes matched donor-label and locality controls from the start.
