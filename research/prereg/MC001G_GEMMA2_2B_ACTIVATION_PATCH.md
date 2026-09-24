# MC001G Gemma 2 2B Activation Patch Preregistration

Date: 2026-06-30

Status: preregistered for path/source localization gate.

## Scope

- Card ID: `MC001G`
- Model: `google/gemma-2-2b`
- Behavior substrate: repaired base-Gemma raw-logit MC001G gate
- Stage: causal activation replacement after failed dense steering
- Runner: `code/mc001_gemma_activation_patch.py`
- Prior intervention status: `research/cards/MC001G_GEMMA2_2B_INTERVENTION_STATUS.md`
- Repair result: `results/cards/MC001G/mc001g_gemma2_2b_repair_logit_raw_gemma_repair_20260630T121025.json`
- Result directory: `results/cards/MC001G/`

## Rationale

The first matched dense intervention failed: the layer 14 truth-minus-agreement
direction changed zero matched-holdout labels, while sign-flip and wrong-layer
controls each improved one row. That rules out a simple additive dense
direction as the current mechanism candidate.

This gate tests a stronger causal operation on the same matched substrate:
replace the target row's final-token residual stream at selected layers with a
discovery-row donor activation. If the matched signature reflects a local
causal state, truth-donor replacement should move held-out agreement rows
toward truth-following more than self-patch, same-label donor, and wrong-layer
controls.

## Fixed Data

Use the repaired MC001G result and the same no-hint-margin matching rule as
`MC001G_GEMMA2_2B_PREHINT_MARGIN_DISCOVERY`:

- conditions: `wrong_disclaimed`, `wrong_unsure`;
- match field: item-level `no_hint` correct-minus-wrong log-probability;
- bin width: `0.5`;
- train/donor split: matched discovery rows only;
- primary eval split: matched holdout rows only;
- locality rows: `no_hint` and `correct_hint` rows for the same holdout items.

Donors are selected deterministically from discovery rows with this priority:

1. same condition, same no-hint-margin bin, same correct and wrong letters;
2. same condition and same bin;
3. same bin;
4. same condition and nearest bin;
5. nearest bin.

## Arms

- baseline forced-choice logit scoring;
- layer 14 self replacement;
- layer 14 truth-donor replacement;
- layer 14 agreement-donor replacement;
- layer 20 truth-donor replacement, because the prior matched logistic probe
  peaked at layer 20;
- layer 13 truth-donor replacement as a nearby wrong-layer control.

All patches replace only the final-token layer output for the evaluated prompt.

## Success Criteria

A mechanism-like result requires:

- layer 14 truth-donor replacement increases truth-following on matched holdout
  rows whose baseline/source label is `user_agreement_error`;
- matched holdout rows whose baseline/source label is `truth_following` are not
  materially degraded;
- `no_hint` and `correct_hint` locality rows are not materially degraded;
- self replacement is a null;
- layer 14 agreement-donor replacement does not produce the same truth gain;
- the nearby wrong-layer control does not match or exceed the layer 14
  truth-donor effect.

## Failure Criteria

Record a failed causal-localization result if:

- truth-donor replacement does not improve held-out agreement rows;
- self, same-label donor, or wrong-layer controls match or exceed the intended
  effect;
- locality rows degrade enough that the patch is better described as broad
  answer disruption;
- the result depends on donor fallback rules rather than matched donors.

This gate cannot by itself produce a mechanism card. A pass would justify a
more specific source/path localization or sparse-feature search; a fail keeps
MC001G as a repaired-substrate / matched-signature / failed-intervention
artifact.
