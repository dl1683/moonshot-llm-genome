# MC005 Associative Lookup Response-Marker V29 Attention Write Replacement Preregistration

Date: 2026-06-30

## Purpose

V18-V21 support a reliable Qwen3-1.7B MC005 source-mask control surface in
layers 24-26. V26 found a predictive row signature, but V27 additive steering
and V28 donor replacement failed to turn that signature into causal row
control.

V29 shifts from the failed row-signature branch back to the supported parent
surface:

> Is the layers-24-26 source-mask effect mediated by the final-query
> self-attention output writes in those layers?

## Source Artifacts

V29 uses the V25 row artifact because it contains the current parent-effect
lookup rows and answer-absent null rows:

- `results/cards/MC005/mc005_qwen3_1p7b_response_marker_v25_factorial_row_heterogeneity_20260630T205454.json`

V29 treats V27 and V28 as negative row-signature controls, not as direct inputs
to the write-path intervention.

## Split

Lookup evaluation:

- V25 lookup seed 239;
- parent-effect rows only;
- same `Response:` pair16 layout as V25.

Null evaluation:

- V25 answer-absent null seeds 251 and 257.

## Intervention

For each row and source key, V29 first runs a counterfactual source-mask pass
over layers 24-26 and captures the self-attention module output at the final
query token for each of layers 24, 25, and 26.

Then V29 runs the original prompt with no attention mask intervention, but
replaces only the final-query self-attention output in the requested layers
with the cached counterfactual write.

Lookup source keys:

- `target_value`;
- `distractor_value`;
- `random_value`.

Lookup write-replacement layer sets:

- layers 24-26 together;
- each single layer 24, 25, and 26.

Null source keys:

- `source_value`;
- `non_source_control_value`;
- `final_colon`.

The direct layers-24-26 source mask remains the reference effect. V29 is a
mediation/sufficiency diagnostic: it asks whether replacing the attention write
recovers the direct mask effect without applying the attention mask during the
scored run.

## Success Criteria

V29 supports an attention-write mediation claim only if all criteria pass:

1. V25 source artifact validates.
2. The direct parent target source-mask reference is valid: baseline target wins
   are at least 75 percent, target source-mask mean delta is at most -5.0, and
   target-win loss is at least 8.
3. Direct source controls pass: target source-mask mean delta is at least 0.50
   more negative than distractor and random source-mask mean deltas.
4. Layers-24-26 target write replacement recovers at least 70 percent of the
   direct target source-mask mean-delta magnitude.
5. Layers-24-26 target write replacement recovers at least 50 percent of the
   direct target source-mask target-win loss.
6. Write replacement source controls pass: target write replacement is at least
   0.50 more negative than distractor and random write replacements.
7. Answer-absent write-replacement nulls are clean on seeds 251 and 257: every
   null arm has baseline target wins at least 75 percent, target-win loss
   exactly 0, and absolute mean delta at most 0.25.

## Diagnostic Labels

- `attention_write_replacement_supported`: all criteria pass.
- `source_artifact_invalid`: source artifact validation fails.
- `direct_parent_invalid`: direct source-mask parent reference is invalid.
- `direct_source_control_failed`: direct source controls fail.
- `write_delta_recovery_failed`: target write replacement fails mean-delta
  recovery.
- `write_win_loss_recovery_failed`: target write replacement fails target-win
  loss recovery.
- `write_source_control_failed`: write replacement controls match or exceed the
  target effect.
- `null_failed`: answer-absent write-replacement nulls fail.

## Allowed Interpretation

If V29 passes:

> The MC005 layers-24-26 source-mask control surface is mediated, to the tested
> threshold, by final-query self-attention output writes in those layers.

If V29 fails:

> The direct source-mask effect remains reliable, but final-query attention
> write replacement did not prove a clean mediation surface under the required
> source controls and nulls.
