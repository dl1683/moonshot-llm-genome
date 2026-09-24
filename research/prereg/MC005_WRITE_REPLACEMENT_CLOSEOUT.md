# MC005 Write-Replacement Closeout Preregistration

Date: 2026-07-01

## Purpose

MC005 is the atlas calibration specimen: it has the strongest internal causal
surface found so far, but it is not promoted because strict answer-absent null
locality failed.

This closeout is not another repair attempt. It is an executable judgment over
the existing MC005 intervention evidence, with the goal of deciding whether the
write-replacement route should be promoted, bounded, or killed as a promotion
path.

## Source Artifacts

- V27 additive residual steering:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v27_signature_intervention_20260630T212004.json`
- V28 donor replacement:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v28_donor_replacement_20260630T213704.json`
- V29 attention-write replacement:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v29_attention_write_replacement_20260630T214645.json`
- V30 write null sweep:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v30_write_null_sweep_20260630T215937.json`
- V31 margin-boundary audit:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v31_margin_boundary_20260630T220916.json`

## Runner

`code/mc005_write_replacement_closeout_audit.py`

The runner must validate artifact sentinels before emitting a verdict.

Required sentinels:

- V27 diagnostic class is `plus_target_no_row_effect`.
- V28 diagnostic class is `source_control_failed`.
- V29 diagnostic class is `null_failed`.
- V29 target write recovery versus direct source masking is exactly `1.0` for
  mean-delta recovery and target-win-loss recovery.
- V30 diagnostic class is `fresh_write_null_failed`.
- V30 reproduces the V29 seed-251 null failure and finds fresh target-win
  changes while keeping fresh mean deltas small.
- V31 diagnostic class is `null_boundary_broad`.
- V31 reproduces the lookup target effect, keeps lookup losses high-margin,
  and rejects the strict `0.5` absolute-margin explanation for all null flips.

## Decision Rules

Promotion rule:

> Promote MC005 only if lookup mediation remains exact, source controls pass,
> and strict answer-absent null locality passes across V29-V31.

Bound rule:

> Bound MC005 if lookup mediation remains exact and high-margin while rare
> low-to-moderate-margin answer-absent null flips persist.

Kill rule:

> Kill further same-route write-replacement promotion attempts if V30-V31 keep
> reproducing null flips and the margin-boundary explanation remains broader
> than the preregistered tight threshold.

Containment rule:

> The surviving claim may only cover source-visible associative lookup in
> Qwen3-1.7B under the tested `Response:` prompt contract.

Export rule:

> Export `NULL_ROW_LOW_MARGIN_FLIP`, `MODEL_SIZE_NULL_FRAGILITY`, and
> `ATTENTION_WRITE_MEDIATION_BOUNDED` as reusable diagnostics if promotion
> fails.

## Forbidden Moves

- Do not average null rows into lookup success.
- Do not relax the null-margin threshold after seeing the result.
- Do not treat V26 row-signature steering as causal after V27/V28.
- Do not start another MC005 V-number repair unless the intervention family is
  materially different from the exact write-replacement route.
