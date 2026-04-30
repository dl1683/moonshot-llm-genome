# Pre-registration: g196 Anchor-Residue Factorial

**Status:** LOCKED (g195 PASS_OUTPUT_DOMINANT → surface=output, lm_head only. Launch gated on g192 completion.)

## Motivation

g191/g194 established that exact-string matched trained token-row directions carry the within-family gain. g194 specifically resolved the scalar-vs-direction confound: correct trained unit directions with shuffled or uniform norms recover 95-97% of the full +0.465 nats signal, while wrong/random directions are harmful.

A18 SEV-10 #2 remains unresolved: g191 found `direct_anchor_only` recovers 98% of the signal, while `direct_init_only` recovers only 19%. The live question is whether the trained row directions change the recipient optimization basin, leaving a persistent residue after the tether is removed, or whether they only help as an active regularizer while the anchor term remains in the loss.

A18 SEV-10 #3 also remains live: "direction" could mean a well-conditioned angular scaffold, not token-specific trained row content. g196 therefore includes angular-scaffold controls, not just anchor cutoff schedules.

## Launch Gate

Run g196 only after g195 completes.

- If g195 returns `PASS_INPUT`, `PASS_INPUT_DOMINANT`, `PASS_OUTPUT`, `PASS_OUTPUT_DOMINANT`, or `PASS_BOTH_NEEDED`, run the **untied primary branch** below using the g195-winning intervention surface.
- If g195 is ambiguous but `tied_reference` cleanly reproduces the g191/g194 signal while untied decomposition is not interpretable, run the **tied fallback branch** and label the result as resolving regularization only for the tied interface.
- If g195 `FAIL`s with no arm >= +0.10 nats and no tied-reference replication, do not launch g196. First diagnose why the known g191/g194 effect failed to replicate.

## Hypotheses

**H1: PASS_RESIDUE.** Correct trained row directions alter the optimization basin. Early anchor exposure leaves a final-step gain after the anchor is removed for thousands of SGD steps.

**H2: PASS_REGULARIZATION.** Correct trained row directions help only as an active tether. Once the anchor is removed, gains decay back to the no-anchor/init-only regime.

**H3: PASS_SCAFFOLD_ALT.** The apparent direction effect is not token-specific trained content. A geometry-preserving or covariance-conditioned scaffold produces a comparable active-anchor gain.

## Common Protocol

- Architecture: 8-layer Qwen3-architecture shell with GPT-2 tokenizer, same base setup as g191/g194/g195.
- Training: 5000 steps, C4 train windows and C4 validation windows identical to g191/g194/g195.
- Seeds: `[42, 7, 13]`.
- Eval cadence: every 500 steps plus final.
- Anchor lambda: `0.01` whenever the schedule says anchor is active.
- Anchor mask: exact-string matched GPT-2/Qwen3 token rows only.
- Primary target matrix: g194 direction-only target, `correct_dir_uniform_norm`, not full trained norms. Matched rows use trained unit directions with uniform matched-row mean norm, Frobenius-normalized to the g194 matched-row Fro norm. Unmatched rows are not anchored.
- Score: seed-matched final validation NLL gain vs scratch: `gain_arm_seed = scratch_seed_final_nll - arm_seed_final_nll`. Positive is better.

## Branch Selection From g195

### Untied Primary Branch

Use `tie_word_embeddings=False`. The intervention surface is determined by g195:

| g195 verdict | g196 intervention surface |
|---|---|
| `PASS_INPUT` or `PASS_INPUT_DOMINANT` | Apply init/anchor schedules only to `model.model.embed_tokens.weight`. |
| `PASS_OUTPUT` or `PASS_OUTPUT_DOMINANT` | Apply init/anchor schedules only to `model.lm_head.weight`. |
| `PASS_BOTH_NEEDED` | Apply the same target and schedule to both `embed_tokens` and `lm_head`. |

The scratch baseline is always `scratch_untied`.

### Tied Fallback Branch

Use `tie_word_embeddings=True`, same as g191/g194. The single tied matrix receives the schedules below. This branch cannot distinguish input embedding from output classifier geometry; it only decides persistence vs active regularization for the tied interface prior.

## Arms

Primary run: 10 arms x 3 seeds = 30 cells.

| Arm | Step-0 init | Anchor schedule | Purpose |
|---|---|---|---|
| `scratch` | None | lambda=0 for steps 1-5000 | Seed-matched baseline |
| `init_only` | Target rows injected into selected surface | lambda=0 for steps 1-5000 | Measures pure initialization residue |
| `anchor_only_full` | None | lambda=0.01 for steps 1-5000 | Active-regularization positive reference |
| `init_anchor_full` | Target rows injected into selected surface | lambda=0.01 for steps 1-5000 | g191/g194-style positive reference |
| `cutoff_50` | None | lambda=0.01 for steps 1-50, then 0 for steps 51-5000 | Tests whether a tiny early tether changes basin |
| `cutoff_500` | None | lambda=0.01 for steps 1-500, then 0 for steps 501-5000 | Tests early-training basin residue |
| `cutoff_2000` | None | lambda=0.01 for steps 1-2000, then 0 for steps 2001-5000 | Primary persistence test: 3000 post-cutoff steps |
| `late_anchor_only_2000` | None | lambda=0 for steps 1-2000, then 0.01 for steps 2001-5000 | Tests whether active tether helps even when introduced late |
| `orthogonal_scaffold_full` | None | lambda=0.01 for steps 1-5000 to orthogonally rotated target | Preserves all trained row-row angles; destroys trained coordinate basis |
| `cov_scaffold_full` | None | lambda=0.01 for steps 1-5000 to covariance-matched random target | Preserves second-order conditioning; destroys token-specific row content |

`cutoff_*` arms deliberately use no step-0 injection. They test whether the active anchor, starting from random init, leaves a basin residue. `init_only` and `init_anchor_full` separately measure whether step-0 placement matters.

## Scaffold Control Construction

All scaffold controls use the same matched-row mask and same uniform row norm/Frobenius normalization as the primary target.

### `orthogonal_scaffold_full`

Let `T` be the primary matched-row target. Draw a fixed random orthogonal matrix `Q` from QR decomposition of a standard normal `(d, d)` matrix using scaffold seed `19601`. Use `T_rot = T @ Q` on matched rows.

This preserves every pairwise cosine between token rows exactly, including the global angular graph, while destroying the trained coordinate basis.

### `cov_scaffold_full`

Let `T_m` be the matched-row target matrix. Estimate its empirical mean and covariance. Draw `X_m ~ N(mean(T_m), cov(T_m) + eps I)` using scaffold seed `19602`, then row-normalize to the same uniform norm and Frobenius-normalize to matched Fro.

This preserves the row-cloud conditioning and coordinate covariance structure but destroys token-specific trained row identity and most pairwise row relations.

## Metrics

For each arm:

- `final_gain_mean`: mean seed-matched gain at step 5000.
- `final_gain_per_seed`: seed-matched gains for seeds 42, 7, 13.
- `active_reference_gain`: `anchor_only_full.final_gain_mean`.
- `init_anchor_reference_gain`: `init_anchor_full.final_gain_mean`.
- `residue_fraction_2000`: `cutoff_2000.final_gain_mean / anchor_only_full.final_gain_mean`.
- `control_max_gain`: max of `orthogonal_scaffold_full.final_gain_mean`, `cov_scaffold_full.final_gain_mean`.
- Drift diagnostics for cutoff arms: gain at cutoff eval step, gain at 5000, and post-cutoff decay.

## Pass/Fail Criteria

### Anchor Effect Replication Gate

Before interpreting residue:

- `anchor_only_full.final_gain_mean >= +0.30` and 3/3 seeds positive, OR
- `init_anchor_full.final_gain_mean >= +0.30` and 3/3 seeds positive.

If neither condition holds, verdict is `FAIL_REPLICATION`; do not interpret residue.

### PASS_RESIDUE

All must hold:

1. Anchor Effect Replication Gate passes.
2. `cutoff_2000.final_gain_mean >= +0.20`.
3. `cutoff_2000` is positive in 3/3 seeds.
4. `residue_fraction_2000 >= 0.45`.
5. `cutoff_2000.final_gain_mean >= init_only.final_gain_mean + 0.10`.
6. `control_max_gain <= +0.15` and `anchor_only_full.final_gain_mean - control_max_gain >= +0.20`.

Interpretation: correct trained directions leave a persistent basin/content residue after 3000 steps without active tether. This resolves A18 #2 in favor of basin change and eliminates A18 #3 for this run.

### PASS_PARTIAL_RESIDUE

All must hold:

1. Anchor Effect Replication Gate passes.
2. `cutoff_2000.final_gain_mean >= +0.12` and positive in at least 2/3 seeds.
3. `residue_fraction_2000 >= 0.25`.
4. Scaffold controls do not pass `PASS_SCAFFOLD_ALT`.

Interpretation: weak basin residue exists, but not enough to support the strong +6.5 section-0.1 move without replication or full-depth follow-up.

### PASS_REGULARIZATION

All must hold:

1. Anchor Effect Replication Gate passes.
2. `cutoff_500.final_gain_mean < +0.12`.
3. `cutoff_2000.final_gain_mean < +0.12`.
4. `residue_fraction_2000 < 0.25`.
5. `late_anchor_only_2000.final_gain_mean >= +0.20` and at least 50% of `anchor_only_full.final_gain_mean`.

Interpretation: directions help mainly while actively tethering the selected rows. The anchor is a regularizer, not evidence of a persistent content/basin residue.

### PASS_EARLY_WINDOW

All must hold:

1. Anchor Effect Replication Gate passes.
2. `cutoff_50.final_gain_mean < +0.10`.
3. `cutoff_500.final_gain_mean < +0.15`.
4. `cutoff_2000.final_gain_mean >= +0.20` and 3/3 seeds positive.

Interpretation: not pure regularization; the basin residue exists but requires a long early tether. This is weaker than PASS_RESIDUE only if retention is below 45%.

### PASS_SCAFFOLD_ALT

Either scaffold control has:

- `final_gain_mean >= +0.20`, OR
- `final_gain_mean >= 0.50 * anchor_only_full.final_gain_mean`.

Interpretation: A18 #3 remains alive. A well-conditioned angular/covariance scaffold is sufficient to explain a substantial part of the benefit. Even if cutoff arms retain gain, the result cannot be claimed as token-specific trained-row content.

### FAIL

Use `FAIL` if no above verdict applies. Report the closest pattern explicitly:

- active effect absent: `FAIL_REPLICATION`;
- active effect present but no residue and late anchor fails: `FAIL_TIMING_AMBIGUOUS`;
- controls positive: `PASS_SCAFFOLD_ALT`;
- mixed seed behavior: `FAIL_NOISY`.

## Compute Envelope

Expected runtime from g194/g195:

- 8-layer tied cells: about 6.9 min/cell.
- 8-layer untied cells: about 7.3 min/cell from current g195 scratch cells.
- 30 cells x 7.3 min = about 219 min = 3.65 hours.
- Smoke test: all 10 arms, seed 42, 50 steps, non-verdict only.

The main run stays under the COMPUTE.md 4h target on the 8-layer shell. Do not run this exact 10-arm design at 28 layers: 30 cells x about 17 min would exceed 8h. If g192 passes and full-depth residue becomes necessary, use a separate compressed g196b with 5 arms (`scratch`, `anchor_only_full`, `cutoff_500`, `cutoff_2000`, `best_scaffold_full`) x 2-3 seeds.

## What Moves Section 0.1

- `PASS_RESIDUE` with scaffold controls dead: section 0.1 can move toward about 6.5/10. The finding becomes "trained interface row directions alter the optimization basin, not merely active regularization."
- `PASS_REGULARIZATION`: cap remains around 6.0 even if g195/g192 pass. The result is useful but should be framed as anchor-conditioned interface regularization.
- `PASS_SCAFFOLD_ALT`: do not claim trained token-row content. Reframe toward generic angular codebook conditioning and design a scaffold predictor.
- `FAIL_REPLICATION`: pause the g191/g194/g195 chain and audit implementation/data drift.
