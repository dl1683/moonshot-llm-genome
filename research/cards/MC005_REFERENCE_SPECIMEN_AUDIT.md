# MC005 Reference Specimen Audit

Date: 2026-07-01

Status: generated bounded-reference audit; no full promotion.

Machine-readable artifact:

> `data/mc005_reference_specimen_audit.json`

Builder:

> `code/mc005_reference_specimen_audit.py`

Commands:

```powershell
python code\mc005_reference_specimen_audit.py --write
python code\mc005_reference_specimen_audit.py
python code\validate_control_surface_atlas.py
```

## Purpose

Freeze MC005 as the calibrated bounded internal-causal reference specimen: exact high-margin lookup mediation exists, but null locality and transfer keep it below full mechanism-card promotion.

## Generated Facts

- row id: `mc005_associative_lookup`;
- verdict: `bounded_mechanism_card`;
- route status: `bounded_frozen_not_promoted`;
- terminal stage: `reliability_null_boundary`;
- reliability class: `bounded_reliability_reference`;
- transfer class: `bounded_transfer_fragile_reference`;
- lookup write effect exact: True;
- strict answer-absent nulls clean: False;
- combined null flip count: 5;
- same-route repair allowed: False.

## Decisive Metrics

- V29 delta recovery versus direct source masking: `1.0`;
- V29 target-win-loss recovery versus direct source masking: `1.0`;
- V31 lookup target write mean delta: `-6.19775390625`;
- V31 lookup target-win loss: `11`;
- V31 lookup loss margin bands: `{"0p25_0p5": 0, "0p5_1": 0, "1_2": 0, "gt_2": 11, "le_0p25": 0}`;
- V31 null flip margin bands: `{"0p25_0p5": 0, "0p5_1": 1, "1_2": 0, "gt_2": 0, "le_0p25": 4}`;
- V30 imported flips inside strict 0.5 margin cutoff: `False`.

## Decision Rules

- work order: `close_mc005_reference_specimen`;
- iteration budget: Two serious repair variants, then promote, bound, or kill.
- promotion rule: Promote only if lookup mediation remains strong and answer-absent null flips disappear or fall below a predeclared trivial threshold across locality, side-row, and holdout panels.
- bound rule: Bound if source-value mediation remains high-margin and localized but answer-absent low-margin rows continue to flip after the planned repair attempts.
- kill rule: Kill the write-replacement route if null flips or side effects persist across two materially different local intervention variants.
- containment rule: The surviving claim may say source-value attention-write mediation exists in the tested lookup contract; it may not say full reliability, transfer, or general knowledge control.
- export rule: Export NULL_ROW_LOW_MARGIN_FLIP and MODEL_SIZE_NULL_FRAGILITY as named reliability diagnostics if promotion fails.

## Future Work Admission

- same-route repair allowed: False;
- allowed future work: `["materially new intervention family with preregistered null-locality rationale", "matched transfer panel with null reliability as a first-class gate", "comparative baseline for a new bridge or mechanism-card family"]`;
- forbidden future work: `["Do not add unlimited V-number repair attempts.", "Do not average null rows into primary lookup success.", "Do not treat coarse source deletion as circuit locality."]`.

## Claim Boundary

MC005 is a bounded internal-causal reference specimen for Qwen3-1.7B associative lookup under the tested Response: contract: layers 24-26 final-query attention writes exactly mediate high-margin lookup rows.

MC005 is not a promoted mechanism card, not fully reliable, not transfer-ready, and not evidence for general factual or knowledge control.

## What This Proves

It proves that the atlas has one calibrated bounded internal-causal
reference specimen: a local intervention exactly mediates the primary
lookup behavior while reliability gates still stop full promotion.

## What It Does Not Prove

It does not prove full mechanism-card reliability, transfer to other
models, or general control of factual knowledge.
