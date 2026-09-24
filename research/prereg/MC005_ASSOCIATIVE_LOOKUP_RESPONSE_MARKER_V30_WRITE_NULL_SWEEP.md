# MC005 Associative Lookup Response-Marker V30 Write Null Sweep Preregistration

Date: 2026-06-30

## Purpose

V29 found the strongest MC005 control surface so far: on V25 holdout lookup
rows, replacing only the final-query self-attention output writes in layers
24-26 exactly recovered the direct layers-24-26 target source-mask effect.

V29 still failed the full reliability gate because one answer-absent null arm
changed one row:

- seed 251;
- `non_source_control_value`;
- `target_win_loss = -1`;
- mean delta `+0.1406`.

V30 asks the narrow follow-up:

> Is the V29 answer-absent write-replacement null failure a reproducible
> reliability boundary, or a sample-fragile one-row event under the strict
> null convention?

## Source Artifacts

V30 uses:

- V25 rows:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v25_factorial_row_heterogeneity_20260630T205454.json`
- V29 write-replacement result:
  `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v29_attention_write_replacement_20260630T214645.json`

The V25 artifact must validate as
`associative_lookup_response_marker_v25_factorial_row_heterogeneity` with
diagnostic class `factorial_cell_contrast_failed`.

The V29 artifact must validate as
`associative_lookup_response_marker_v29_attention_write_replacement` with
diagnostic class `null_failed`, source hash matching V25, exact lookup
write-replacement recovery, and the seed-251 `non_source_control_value` null
failure recorded.

## Panels

V30 has two panels.

1. `v25_replay_96`: replay the V25 answer-absent null rows for seeds 251 and
   257, 96 rows each. This panel must reproduce the V29 failure.
2. `fresh_128`: generate fresh V25-compatible answer-absent null rows with
   `build_null_rows(tokenizer, seed, 128)` for seeds
   263, 269, 271, 277, 281, 283, 293, and 307.

## Intervention

For every seed and every null source key:

- collect the masked final-query self-attention output writes from layers
  24-26 using the same V29 source-mask capture path;
- score the original prompt while replacing only the final-query attention
  writes in layers 24-26;
- compare against the unmodified baseline.

Null source keys:

- `source_value`;
- `non_source_control_value`;
- `final_colon`.

## Row Audit

For each seed and source key, V30 records every row where the baseline
`target_wins` boolean differs from the write-replacement `target_wins`
boolean. Each changed row records:

- row id;
- baseline and arm margins;
- margin delta;
- baseline and arm target-win booleans;
- target-win-loss contribution;
- baseline and arm greedy labels/tokens;
- target value, distractor value, source-focus value, and control value when
  present.

## Clean Null Rule

An arm is clean only if:

- baseline target wins are at least 75 percent for the seed;
- `target_win_loss` is exactly 0;
- absolute mean delta is at most 0.25.

A seed is clean only if all three source-key arms are clean.

## Criteria

V30 records these criteria:

1. V25 source artifact validates.
2. V29 source artifact validates.
3. The V29 seed-251 `non_source_control_value` failure is reproduced in the
   `v25_replay_96` panel.
4. Every fresh seed is clean.
5. No fresh arm has absolute mean delta greater than 0.25.
6. No fresh arm has a nonzero target-win change.

## Diagnostic Labels

- `write_null_sweep_clean`: source artifacts validate, the V29 failure is
  reproduced, and all fresh seeds pass cleanly.
- `v29_replay_failed`: source artifacts validate but the expected V29 replay
  failure is not reproduced.
- `fresh_write_null_failed`: at least one fresh arm has a nonzero target-win
  change or exceeds the mean-delta tolerance.
- `source_artifact_invalid`: V25 or V29 validation fails.
- `mixed_null_boundary`: any other mixed state.

## Allowed Interpretation

If V30 returns `write_null_sweep_clean`:

> The V29 null failure is reproduced on its original rows but does not persist
> across the fresh 8-seed, 1,024-row answer-absent null sweep. The V29 failure
> remains a documented strict-null violation, but the current evidence favors a
> sample-fragile row event over a broad write-replacement side effect.

If V30 returns `fresh_write_null_failed`:

> The V29 null failure marks a real answer-absent write-replacement reliability
> boundary. The MC005 attention-write surface remains lookup-causal but cannot
> be promoted without a stronger null-locality explanation or intervention
> refinement.

If V30 returns `v29_replay_failed`:

> The V29 null failure is not reproducible under the current runner and must be
> treated as an artifact-consistency problem before interpretation.
