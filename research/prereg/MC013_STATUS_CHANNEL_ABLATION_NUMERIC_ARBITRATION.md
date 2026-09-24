# MC013 Status-Channel Ablation Numeric Arbitration Preregistration

Date: 2026-07-01

## Objective

MC013 tests the immediate post-MC012 bridge question:

> Does the MC012 local-versus-learned numeric contrast survive when the
> visible trusted/untrusted source-status channel is removed?

MC012 produced the first clean mixed bridge behavior table in this branch, but
the contrast was prompt-channel visible by construction. MC013 keeps the same
integer answer interface and chemical-element source bank, then adds a matched
ablation control where the two primary conflict prompts are text-identical.

## Behavior Contract

The runner is:

> `code/mc013_status_channel_ablation_numeric_arbitration.py`

The full table uses 40 chemical elements across source-disjoint discovery,
calibration, and holdout splits. It creates eight panels:

- `synthetic_numeric_lookup`;
- `familiar_entity_numeric_lookup`;
- `real_world_atomic_number_control`;
- `statused_trusted_conflict`;
- `statused_untrusted_conflict`;
- `matched_ablation_trusted_conflict`;
- `matched_ablation_untrusted_conflict`;
- `answer_absent_null`.

The statused conflict panels are positive controls. They should reproduce the
MC012 behavior shape:

- trusted source status -> local lab number;
- untrusted source status -> learned atomic number.

The matched ablation panels are the primary locality test. For each
source/template pair, the trusted-ablation and untrusted-ablation prompts must
be exactly identical. Their expected labels remain inherited from the original
trusted/untrusted condition, but the model receives no visible status cue.

## Promotion Rule

MC013 can be behavior-ready for a later hidden-signature screen only if one
selected template satisfies all of:

- synthetic numeric lookup at least 90 percent local-number answers;
- familiar entity numeric lookup at least 90 percent local-number answers;
- real-world atomic-number control at least 85 percent atomic-number answers;
- answer-absent null at least 90 percent `UNKNOWN`;
- statused trusted conflict at least 85 percent local-number answers;
- statused untrusted conflict at least 85 percent atomic-number answers;
- ablated trusted conflict at least 85 percent local-number answers;
- ablated untrusted conflict at least 85 percent atomic-number answers;
- ablated primary conflict parseability at least 90 percent;
- non-holdout ablated conflict has at least 10 local and 10 atomic/lure rows;
- holdout ablated conflict has at least 4 local and 4 atomic/lure rows;
- candidate and output margins are reported;
- matched ablation prompt pairs are exactly identical.

## Death Rule

The route is not signature-ready if the statused positive controls pass but the
matched ablation collapses to one side. In that case the verdict is:

> status-channel ablation collapsed contrast; hidden-state work blocked.

## Containment Rule

If the ablation fails, the allowed claim is only that the MC012 contrast depends
on the visible source-status channel under this prompt contract. The forbidden
claim is that MC012 exposed an internal knowledge-control surface.

## Exported Diagnostics

- `STATUS_CHANNEL_POSITIVE_CONTROL_REPRODUCED`;
- `STATUS_CHANNEL_ABLATION_COLLAPSED_CONTRAST`;
- `MC013_STATUS_ABLATION_FAILED_DIAGNOSTIC_BRIDGE`.

## Forbidden Claims

- MC013 is a mechanism card.
- MC013 supports intervention or steering.
- MC013 found an internal knowledge-control surface.
- A hidden-state signature should be searched after status-channel ablation
  collapses the behavior contrast.
