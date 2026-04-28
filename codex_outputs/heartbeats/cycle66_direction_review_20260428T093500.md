# g180 Genome Forecast Design Gate

## Features (5-8 picks with justification)

Extract at `min(200, ceil(0.03 * final_steps))`, using the same C4 probe batch across cells. If legacy cells lack an early checkpoint, rerun only the early prefix with the same seed/arm and attach the already-known final label.

1. **Mid-layer spectral invariant:** centered hidden-state SVD, reporting tail slope `alpha`, participation ratio, and `sqrt(PR) * alpha`. This is the repo's strongest architecture-portable activation coordinate; it works on token-pooled hidden clouds, not Qwen-specific weights.
2. **Depth spectral drift:** slope of the spectral invariant across early/mid/late blocks. This tests whether a run is forming trained-text geometry through depth or just fitting local loss. Qwen/Llama/RWKV/Falcon-H1 all expose ordered hidden states or recurrent/block states.
3. **TwoNN intrinsic dimension:** demoted as a universal atlas coordinate, but still useful as a forecast diagnostic. It captures manifold expansion/collapse and is cheap on 800-1000 pooled activations.
4. **kNN-10 clustering coefficient:** keep the existing primitive. It measures local neighborhood organization, is invariant to rotation/isotropic scale, and has already passed the strongest portability gate across Qwen, RWKV, Falcon-H1, reasoning, and vision systems.
5. **PCA-64 Procrustes/RSA distance to trained reference:** project the checkpoint cloud and the Qwen3 teacher/reference cloud to 64 PCs, then report orthogonal Procrustes residual plus RDM correlation. PCA/RDM avoids hidden-size mismatch, so it can compare Qwen, Llama, RWKV, and Falcon-H1 without assuming matching coordinates.
6. **Gradient-noise scale:** compute variance of gradient norms over 4 microbatches divided by squared mean norm. This captures optimizer stability and bad-run risk; it is architecture-neutral because it uses scalar gradients after the model's own loss.
7. **Curvature proxy:** one top-eigenvalue estimate from 4-6 Hessian-vector products on the loss, restricted to final block + lm head if full-model HVP exceeds 30s. This gives sharpness without a full Hessian top-k run.
8. **Norm/variance depth ratios:** median RMSNorm/LayerNorm scale and hidden-activation variance ratios from early to late depth. Use normalized ratios, not raw module names, so RWKV and Falcon-H1 equivalents remain comparable.

Do not include raw parameter distances or tokenizer-specific embedding norms in the primary feature set; those would mostly relearn architecture identity.

## Labels (1 pick with justification)

Use **paired final C4 validation NLL gain vs scratch**:

`label = final_c4_nll(same_seed_scratch) - final_c4_nll(arm)`.

This is continuous, already available for the Qwen-family transfer runs and g173, and directly supports a compute-saving claim: predict whether an arm is worth continuing before spending the remaining 97% of training. C3_macro is too sparse across old runs, and arm-winner identity is too g173-specific.

## Architecture span (use of existing logs vs fresh runs)

Use existing completed Qwen-family cells for most labels: g165, g167, g172, g174, g177/g177v2, plus g173 Qwen/Llama cells as they complete. For missing early activations, replay only the first <=3% of the exact cell; do not rerun full training.

For cross-architecture validity, the decisive split is **train on Qwen-family labeled cells, test on g173 Llama-family cells**. Reverse split is a sensitivity check if enough Llama cells finish. RWKV/Falcon-H1 atlas results should be used only to confirm feature computability and distribution sanity, not as supervised labels. Fresh RWKV/Falcon-H1 recipient-training runs are not needed for g180 v0 and would break the 4h envelope without adding enough labels.

## Pass criteria (sharp threshold)

PASS if, at <=3% training compute, `early_loss + geometry` reduces held-out-family MSE on final C4 NLL gain by **>=25%** versus an `early_loss_only` baseline, with paired bootstrap improvement above 0 over held-out cells.

Secondary actionability guard: the model must not assign a stop recommendation to any arm whose true final gain is `>= +0.5` nats. The headline claim is the MSE lift; the guard prevents a fake product win that kills the best runs.

## Envelope estimate

Label extraction from existing JSON/logs: 5-10 min. Early-prefix replay for missing checkpoints: ~15-30s per 500M-class cell at <=3% steps, plus <=30s feature extraction; 50 cells is ~40-60 min. g173 one-seed/partial cells can be included immediately, then refreshed after completion. Regression fit, baselines, bootstrap, and report: <=15 min. With dataset/model load overhead and failures, budget **2.0-2.8h**, hard cap **4h**.

## Risks

1. The dataset is Qwen-heavy; a Qwen-to-Llama holdout can fail from label scarcity rather than feature weakness.
2. <=3% checkpoints may be too noisy for geometry; if feature stability fails, move the checkpoint to exactly step 200 but mark the claim as "early" rather than "<=3%".
