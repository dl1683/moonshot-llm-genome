# MC014 Inferred-Reliability Numeric Arbitration Preregistration

Date: 2026-07-01

Status: behavior-gate bridge attempt, not a mechanism claim.

## Objective

Test whether the MC012 local-versus-learned numeric contrast can be recreated
without explicit trusted/untrusted source-status text.

The intended bridge is:

- prompt-local local lab numbers are available in a local table;
- learned atomic numbers are available from standard chemistry;
- source control is inferred from calibration rows;
- calibration-consistent sources should select the local lab number;
- calibration-inconsistent sources should select the learned atomic number.

## Promotion Rule

The behavior substrate can only become signature-ready if the full run passes:

- structural prompt audit;
- 40-source full table;
- source-disjoint discovery/calibration/holdout splits;
- synthetic numeric lookup control;
- familiar entity numeric lookup control;
- real atomic-number control;
- answer-absent UNKNOWN null;
- no explicit trusted/untrusted/reliable/unreliable/status lexemes in primary
  conflict prompts;
- calibration-consistent conflict local-number rate at least 85 percent;
- calibration-inconsistent conflict atomic-number rate at least 85 percent;
- non-holdout and holdout local-versus-atomic/lure balance;
- candidate/output margin reporting.

Passing this gate permits only a preregistered hidden-signature screen. It does
not permit intervention or mechanism-card promotion.

## Death Rule

Close this route as a diagnostic if either conflict side collapses before
hidden-state work, especially if calibration-inconsistent rows still select the
prompt-local local number.

## Containment Rule

If direct controls and nulls pass but the calibration-inconsistent side
collapses, the allowed claim is only:

> Removing explicit source-status labels is not sufficient; under this prompt
> contract, prompt-local table dominance survives calibration evidence.

## Exported Diagnostics

- `CALIBRATION_STATUS_LABEL_ABSENT`
- `CALIBRATION_INFERENCE_CONFLICT_COLLAPSED`
- `MC014_BEHAVIOR_GATE_FAILED_DIAGNOSTIC_BRIDGE`

## Forbidden Claims

- MC014 is a mechanism card.
- MC014 is ready for hidden-state probing if the behavior gate fails.
- Calibration inference produced a clean internal knowledge-control surface.
- Candidate/output margin reporting alone upgrades the behavior table.
