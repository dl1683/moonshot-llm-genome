# MC005 Write-Replacement Closeout Status

Status: bounded mechanism preserved; same write route closed for promotion.

Date: 2026-07-01

## Artifact

- preregistration:
  `research/prereg/MC005_WRITE_REPLACEMENT_CLOSEOUT.md`
- runner:
  `code/mc005_write_replacement_closeout_audit.py`
- result:
  `results/cards/MC005/mc005_write_replacement_closeout_audit_20260701T200017.json`
- result SHA256:
  `5f91b526d8eecd8ccbfbbb938085bc2e198f93d6b1ac4391e2e55e9422bee874`

## Verdict

The closeout audit returns:

```text
bounded_mechanism_card
```

with route status:

```text
bounded_frozen_not_promoted
```

The same write-replacement promotion route is closed. MC005 remains the atlas
reference specimen, but it is not a promoted mechanism card.

## Evidence

The audit validated V27-V31 sentinels and source hashes before making the
decision.

| Route | Diagnostic | Result |
| --- | --- | --- |
| V27 additive residual steering | `plus_target_no_row_effect` | row-signature steering did not produce the intended row effect |
| V28 donor replacement | `source_control_failed` | donor replacement moved the row metric but failed source controls and nulls |
| V29 attention-write replacement | `null_failed` | lookup write mediation exactly matched direct source masking, but strict null locality failed |
| V30 write null sweep | `fresh_write_null_failed` | the V29 null failure reproduced and fresh null row flips appeared |
| V31 margin boundary | `null_boundary_broad` | lookup losses were high-margin, but null flips exceeded the strict `0.5` explanation |

Key metrics:

- V29 target write mean-delta recovery versus direct masking: `1.0`.
- V29 target write target-win-loss recovery versus direct masking: `1.0`.
- V31 lookup target write mean delta: `-6.19775390625`.
- V31 lookup target-win loss: `11`.
- V31 lookup-loss margin band: `11/11` in `>2`.
- V31 combined null flip count: `5`.
- V31 combined null flip margin bands: `4` in `<=0.25`, `1` in `0.5-1`, `0`
  in `>2`.

## Interpretation

This is a stronger closeout than the earlier prose-only bounded status.

MC005 passes the internal-causal standard for the primary lookup surface:
layers 24-26 final-query attention writes exactly mediate the tested
high-margin source-value lookup effect.

MC005 fails full reliability:
the same write route can move low-to-moderate-margin answer-absent null rows
across the target-versus-distractor threshold while aggregate mean deltas stay
clean.

The important lesson is not just "MC005 failed promotion." The useful law is:

> a local intervention can exactly mediate a high-margin behavior surface while
> still being too dirty on low-margin null rows for full mechanism-card
> promotion.

## Allowed Claims

- MC005 remains a bounded mechanism card for Qwen3-1.7B associative lookup
  under the tested `Response:` prompt contract.
- Layers 24-26 final-query attention writes exactly mediate the tested
  high-margin lookup effect.
- The exact write-replacement route is frozen as bounded, not promoted.
- V27/V28 show that the row-signature intervention family is not a substitute
  for the V29 write route.
- V30/V31 show that the answer-absent null boundary is reproducible and broader
  than the strict `0.5` absolute-margin explanation.

## Forbidden Claims

- MC005 is a full promoted mechanism card.
- The write-replacement null boundary is fixed.
- The write-replacement null boundary is fully explained by the preregistered
  `0.5` absolute-margin cutoff.
- MC005 generalizes to factual recall, knowledge control, or smaller models.
- Another same-route write-replacement repair is licensed without a materially
  new intervention family.

## Next Decision

Do not continue MC005 as an indefinite V-number repair line.

Future MC005 work is justified only as:

- a materially new intervention family with a preregistered reason to preserve
  the lookup effect while suppressing answer-absent null flips;
- a width/transfer reliability probe where null locality is the first-class
  gate;
- or a calibration baseline for another candidate mechanism card.
