# Derivation: Early Geometry Predicts Training Health

**Status:** SKELETON → DEEPENING (cycle 111, 2026-04-29). Three routes proposed by Codex Architecture Theorist. Route 3 has Verdict Matrix (cycle 105). Route 2 has formal feature-to-rate mapping + discriminators (cycle 106). Route 3 has RMT theory anchor (cycle 111, arXiv 2604.18450).

**Claim to derive:** At <= 3% of training, activation geometry features predict final run quality better than early loss alone.

---

## Route 1: Fisher/NTK Spectrum

Early hidden geometry estimates the learning operator. In a local linearization around the initial parameters, future loss decrease is:

    L_T - L_* ~ sum_i a_i^2 exp(-2 eta lambda_i T)

where lambda_i are learnable mode speeds (eigenvalues of the NTK/Fisher at initialization + early training) and a_i is the task energy projected into each mode.

**Key insight:** Activation geometry gives noisy early estimates of this spectrum:
- Spectral slope alpha ~ tail decay rate of the eigenvalue distribution
- Participation ratio ~ effective rank of the representation (how many modes carry energy)
- Intrinsic dimension ~ manifold dimensionality (geometric lower bound on expressible modes)
- kNN clustering ~ local manifold regularity (smooth = learnable; fractal = stuck)
- Depth drift ~ how the spectrum changes across layers (should concentrate, not disperse)

**Prediction:** If geometry features are proxies for {lambda_i, a_i}, then they predict L_T. If early loss alone is just L_0, it tells you where you started but not the learning-speed spectrum. Geometry tells you HOW FAST you'll learn, not just WHERE you are.

**Testable:** Compute actual NTK eigenvalues at step 108 for a subset of cells. Correlate with our geometry features. If correlation is high, the proxy claim holds.

**Gap:** NTK linearization breaks for deep nets beyond early training. Need to argue that the geometry proxy survives nonlinear dynamics (or that 3% is early enough for linearization to hold).

**Related work:** arXiv 2604.14500 uses Fisher Information Metric on MoE routing simplex to predict training failure (AUC=0.89 at 10% completion). Their Fisher approach is architecture-specific (routing geometry); ours would measure the Fisher/NTK of the representation space, which is architecture-agnostic. Validates that Fisher-based features carry strong predictive signal for training health.

---

## Route 2: Rate-Distortion (Water-Filling)

The representation is a code for next-token-relevant structure. From rate-distortion theory, optimal coding allocates bits across semantic modes by a water-filling rule:

    R(D) = sum_i max(0, 1/2 log(sigma_i^2 / theta))

where sigma_i^2 are source variances per mode and theta is the distortion threshold.

**Healthy training:** Allocates rate (representation capacity) across many semantic modes proportional to their relevance. The hidden state has high effective dimensionality, balanced spectral energy, smooth local geometry.

**Bad training (waste/collapse):**
- Under-allocation: Low ID, low PR, collapsed spectrum -> model hasn't found meaningful modes
- Tokenizer noise: Energy allocated to tokenizer-specific structure, not semantic content -> high norm ratio but low clustering
- Unigram collapse: All capacity in position/frequency statistics, not meaning -> mid_spectral_alpha extreme

**Prediction:** Model C's spectral/rank/local-neighborhood features are direct proxies for the water-filling allocation. Healthy allocation at 3% predicts healthy final outcome. Model D's telemetry features (loss, gradients) tell you the CURRENT distortion but not the allocation strategy.

**Connection to Umwelt:** Different tokenizers define different source distributions. Cross-tokenizer transfer fails (g180b) because the optimal water-filling for one tokenizer's source distribution is suboptimal for another's. This is exactly the Umwelt Representation Hypothesis: alignment arises from overlapping ecological constraints.

**Theoretical anchors from literature (cycle 106):**

- **D'Amato, Lancia & Pezzulo (PLOS Comp Bio 2025):** Under rate-distortion constraints, optimal codes exhibit prototypization (clustering around centroids), specialization (rare categories get disproportionate capacity), and orthogonalization (class representations become mutually orthogonal at low capacity). These are the optimal solutions, not learned heuristics. Validates that RD theory makes specific, testable geometric predictions about learned representations. Orthogonalization rate across layers could be an additional diagnostic feature.

- **"From SGD to Spectra" (ICML 2025):** Squared singular values of weight matrices under SGD follow Dyson Brownian motion with eigenvalue repulsion, yielding gamma-type stationary distributions with power-law tails. This gives the FIRST principled derivation of why mid_spectral_alpha is a meaningful quantity — the tail exponent reflects the equilibrium of SGD dynamics, not arbitrary power-law fitting. Connects Route 1 (Fisher/NTK eigenvalue spectrum) to Route 2 (rate allocation across modes): SGD naturally performs a stochastic analog of water-filling, spreading singular values to maximize effective rank under noise.

- **Coverage principle (Chen et al., ICLR 2026):** The probability mass a pre-trained model places on high-quality responses is necessary and sufficient for post-training success. Coverage generalizes faster than cross-entropy. Potential extension: coverage at 3% of training could be an additional early-training diagnostic, complementary to manifold geometry features.

- **No paper directly applies Shannon water-filling to neural representation learning (as of 2026-04).** The connection is implicit in the above work but has not been formalized. This is the specific novelty opportunity for Route 2.

- **g195 output-dominance finding (cycle 193):** The lm_head (output classifier) carries 65% of the tied embedding signal (+0.362 vs +0.190 nats for input, additivity 99.3%). The lm_head defines the target coordinate system for the Fisher metric — it IS the codebook against which the network optimizes. Under Route 2, this means lm_head geometry directly determines the water-filling allocation: the output basis defines which modes receive rate. Early lm_head geometry is therefore the PRIMARY diagnostic signal, not hidden-state or input-embedding geometry. This shifts the Route 2 prediction: the diagnostic should focus on output-interface features (spectral properties of lm_head rows, angular coverage, ETF distance) rather than mid-layer activation features.

- **g197 canary arena design (cycle 194):** g197 directly tests the Route 2 prediction. By constructing 10 deliberately varied lm_head initializations — from trained Qwen3 rows (expected healthy) through scaffolds and shuffles to anti-frequency-scaled random (expected doomed) — the arena creates a controlled gradient from optimal to pathological water-filling allocation. If Route 2 is correct, the spectral/angular/reference features of the lm_head at step 0 and step 50 should predict which cells converge well and which waste capacity. The leave-one-condition-out CV design tests whether geometry features generalize to unseen allocation regimes, which is exactly what a water-filling model should do (the same eigenvalue-to-rate mapping applies regardless of initial spectrum shape). PASS_CANARY would validate that output-interface geometry IS the water-filling allocation parameter.

### Route 2 Fisher-Codebook Operator (cycle 198, Codex Architecture-Theorist)

The cross-entropy gradient w.r.t. hidden states is `grad_h L = W_out^T (p - e_y)`, where p is the softmax prediction and e_y the one-hot target. Each row direction `w_y / ||w_y||` is the target gradient prototype for token y. This explains:
- g194 (direction carries 95-97%): the direction IS the gradient signal; norms are temperature scaling.
- g191 (content at exact-match): correct row content = correct gradient prototypes at matched positions.
- g195 (output dominant): lm_head defines the Fisher metric; input embed is secondary.

The natural diagnostic operator is the **output Fisher codebook matrix**:

    M_W = C_h^{1/2} W_out^T (Diag(pi) - pi pi^T) W_out C_h^{1/2}

where C_h is the hidden-state covariance and pi is the token frequency distribution. The eigenvalues of M_W control the effective learning rate per semantic mode. Under water-filling, the diagnostic becomes:

    Phi(W) = sum_i log(1 + lambda_i / theta)

where lambda_i are eigenvalues of M_W. Healthy heads maximize Phi(W); pathological heads (shuffled, anti-frequency) have collapsed or misaligned spectra.

**g192 depth-amplification (cycle 201):** 28-layer matched mean = +0.530 nats vs 8-layer +0.465 = 114% retention (amplification, not attenuation). Route 2 predicts this: each layer receives `W_out^T (p - e_y)` via backprop, so correct output directions propagate well-aligned gradients through ALL layers. More layers = more weight matrices that exploit the aligned codebook → compounding benefit. Testable prediction: effect at 4 layers should be weaker than at 8.

**Dual perspective — gradient bottleneck (Godey & Artzi, arXiv 2603.10145, Mar 2026):** The lm_head W_out projects D-dim hidden states to V-dim logits (D << V). Backprop through this rank-D layer compresses the V-dim loss gradient: everything in ker(W_out^T) is destroyed, suppressing 95-99% of gradient norm. Their analysis is the DUAL of M_W: **M_W captures what the output layer CAN propagate (Fisher range space); ker(W_out^T) captures what it CANNOT propagate (null space).** Together they fully characterize the output-interface bottleneck. Their finding that convergence degrades with V/D ratio validates that lm_head geometry at initialization determines training dynamics. No fix proposed — they diagnose only. Our g197/g199 line TESTS whether initial M_W geometry predicts the severity of this bottleneck.

**Optimal anchor lambda (Codex cycle 201):** Cannot be derived from M_W spectrum alone. In a quadratic mode model, dz_i/dt = -eta[mu_i(z_i - z_i*) + lambda(z_i - a_i)], optimal lambda also needs delta_i = a_i - z_i* (anchor-target error per mode). Honest result: closed-form lambda requires spectrum PLUS an anchor-noise/alignment model.

**Depth scaling law (Codex cycle 201):** 28-layer +0.530 vs 8-layer +0.465 = only +14% despite 3.5x depth → diminishing returns. Pre-registerable form: Delta(L) = Delta_max(1 - exp(-L/L_c)), or if water-filling: Delta(L) ~ sum_i log(1 + L*mu_i/theta) saturating against task/noise limits. Sweep 4/8/16/28/40 layers to distinguish exponential saturation from log growth.

**g197 tests this indirectly** via spectral/angular/scaffold proxies. **g199 (proposed cycle 201): compute M_W eigenvalues directly and test whether Phi(W) predicts final NLL better than the proxy features, LOCO CV.** Also: isospectral wrong-codebook controls (same M_W spectrum, wrong token assignment) — if they fail while trained rows pass, the theory must include a supervised alignment term, not just Fisher spectrum. Expected §0.1 impact of the full derivation: +0.6 to +1.0 if the operator predicts feature rankings.

### Route 2 Formal Feature-to-Rate Mapping (cycle 106)

Let h_l denote the hidden representation at layer l, with covariance Sigma_l = E[h_l h_l^T]. Let sigma_{l,1}^2 >= sigma_{l,2}^2 >= ... >= sigma_{l,d}^2 be the eigenvalues. Water-filling allocates rate to mode i iff sigma_{l,i}^2 > theta (the water level). At training step t, the allocation reflects what the model has learned to encode.

**Feature → Rate-Distortion Quantity:**

1. **mid_spectral_alpha** — The power-law exponent of the eigenvalue tail: sigma_i^2 ~ i^{-alpha}. Under water-filling, rate per mode R_i = max(0, 1/2 log(sigma_i^2/theta)). With a power-law spectrum, the total rate is R_total ~ integral from 1 to i_max of 1/2 log(i^{-alpha}/theta) di. Alpha controls the ALLOCATION SHARPNESS: high alpha → rate concentrated in few top modes (over-specialized); low alpha → rate spread across many modes (under-specialized). The optimal alpha for a given task depends on the source's true semantic dimensionality.

2. **mid_participation_ratio** — PR = (sum sigma_i^2)^2 / sum sigma_i^4. This is exactly the EFFECTIVE NUMBER OF MODES receiving rate above the water level. PR proxies the count of active coding dimensions. In a water-filling interpretation: PR ~ |{i : sigma_i^2 > theta}|. Healthy training should have PR matching the intrinsic dimensionality of the semantic task.

3. **mid_sqrt_pr_alpha** — sqrt(PR) * alpha. This composite captures the RATE CONCENTRATION: how much total rate is packed into how steep a spectrum. High sqrt_pr_alpha = many active modes but sharply decaying → healthy allocation where secondary modes are active but dominated by primary ones. Low = either few modes (collapsed) or flat spectrum (noise).

4. **depth_alpha_drift** — Change in alpha across layers: alpha_deep - alpha_shallow. Under successive refinement (Cover & Thomas), an optimal multi-resolution code should progressively sharpen the spectrum with depth: each layer extracts finer-grained semantic modes. NEGATIVE drift (alpha increasing with depth) = progressive refinement (healthy). POSITIVE drift (alpha decreasing) = representation fragmenting or failing to compress (unhealthy).

5. **depth_pr_drift** — Change in PR across layers. Under successive refinement, PR should DECREASE with depth: deeper layers encode higher-level abstractions in fewer dimensions. Positive drift (PR increasing with depth) = capacity expanding instead of compressing (waste).

6. **depth_sqrt_pr_alpha_drift** — Change in the rate-concentration composite. Should stabilize or slightly decrease with depth if the code is well-structured.

7. **twonn_intrinsic_dim** — Two-nearest-neighbor intrinsic dimensionality (Facco et al. 2017). This estimates the manifold dimensionality of the representation, which under rate-distortion theory is the SOURCE DIMENSIONALITY d_s that the model has learned to code for. d_s should be >> 1 (model found structure) but << d_model (model is compressing). Optimal d_s depends on the data: for C4 next-token, semantic dimensions should be on the order of 10-50.

8. **knn10_clustering_coeff** — k=10 nearest-neighbor clustering coefficient. This measures LOCAL QUANTIZATION QUALITY: how regularly the representation tiles the manifold. High clustering = points form tight local clusters → well-quantized code with good packing. Low clustering = scattered points → noisy code with wasted capacity. Under rate-distortion: clustering ~ inverse of the average local distortion per coding cell.

### Route 2 Quantitative Predictions (pre-registered alongside Route 3)

**R1 (Spectral alpha sweet spot).** If water-filling is the mechanism, there exists an optimal alpha* for C4 next-token-prediction. Cells with mid_spectral_alpha closer to alpha* should have better final outcomes. The relationship outcome(alpha) should be CONCAVE (inverted-U), not monotonic. Testable: fit a quadratic alpha term in Ridge and check if the quadratic coefficient is negative and significant.

**R2 (PR predicts outcome monotonically below saturation).** Higher PR = more active coding modes = better allocation, UP TO the point where PR exceeds the task's intrinsic dimensionality (noise modes get allocated rate). Prediction: in g182 data, partial correlation of PR with outcome (controlling for alpha) should be positive for PR < threshold and flat or negative above.

**R3 (Depth drift direction is diagnostic).** Under successive refinement, depth_alpha_drift < 0 (alpha increases with depth) and depth_pr_drift < 0 (PR decreases with depth) are NECESSARY conditions for healthy training. Cells where both drifts are negative should systematically outperform cells where either is positive. Testable: partition cells by sign of depth drifts and compare mean outcomes.

**R4 (ID should correlate with source complexity, not architecture).** Under rate-distortion, twonn_intrinsic_dim reflects the learned source dimensionality, which is DATA-dependent. Within the same arm, ID should be similar across Qwen3 and GPT-2. This overlaps with Route 3's P3 but for a DIFFERENT reason: Route 3 says "same basin" → same features; Route 2 says "same data" → same optimal source model → same ID.

### Where Route 2 and Route 3 Make Different Predictions

**Critical divergence: functional form of geometry→outcome.**

- Route 3 (phase transition): predicts a STEP FUNCTION. Cells are in discrete basins. Geometry features identify the basin. Within-basin, features have no marginal predictive value. A classification model (basin label → outcome) should match or beat Ridge.

- Route 2 (water-filling): predicts a SMOOTH, CONTINUOUS relationship. Better allocation → better outcome, with diminishing returns. Ridge should capture most of the signal. A classification model that discretizes into basins should LOSE information compared to continuous features.

**Testable discriminator (D1):** Fit both (a) Ridge on continuous features and (b) k-means(k=3) + per-cluster-mean as predictor. If (a) beats (b) by >10% MSE, Route 2's continuous picture is better. If (b) ties or beats (a), Route 3's basin picture is better. This can be evaluated on g182 data without new experiments.

**Critical divergence: depth drift causality.**

- Route 3: depth drift is an ORDER PARAMETER — it identifies the basin but doesn't have causal predictive value beyond basin identity. Removing depth drifts from the feature set should not hurt much if basin-identity features (alpha, PR) remain.

- Route 2: depth drift measures CODING EFFICIENCY across layers — it has independent predictive value beyond spectral features. Removing depth drifts should hurt significantly even when alpha and PR are present.

**Testable discriminator (D2):** Compare Ridge with all 8 features vs Ridge with only features [0,1,2,6,7] (no depth drifts). If the 5-feature model matches the 8-feature model within noise, Route 3 is favored. If the 8-feature model wins by >5% MSE, Route 2 is favored.

---

## Route 3: Statistical Physics (Symmetry Breaking)

Early training is symmetry breaking. By ~3%, the run has often chosen an order-parameter basin:
- Random/noisy basin: geometry indistinguishable from random initialization
- Collapsed basin: low-rank, degenerate, high alignment with initialization
- Trained-attractor basin: structured manifold with intermediate ID and spectral properties

**The geometry features are order parameters** for this phase transition:
- PR and ID: extensive -> trained basin; collapsed -> degenerate basin; random -> random basin
- Spectral alpha: intermediate -> trained; extreme -> collapsed or random
- Depth drift: decreasing alpha with depth -> progressive abstraction (healthy); flat or increasing -> stuck

**Prediction:** The basin chosen by step 108 (3% of training) determines the final outcome with high probability. Geometry features identify which basin. Early loss alone cannot distinguish "random but about to find the attractor" from "random and stuck."

**Testable:** Cluster cells by step-108 geometry features. Verify that clusters map cleanly to final outcome bins (good/bad/waste). Loss-only clustering should be worse.

**Connection to Shesha:** Shesha's RDM stability metrics measure one order parameter (geometric consistency). Our moat requires showing that MULTIPLE order parameters (spectral + rank + local geometry) contain signal beyond generic RDM stability.

**Theory anchor: RMT of Early-Stopped Gradient Flow (arXiv 2604.18450, cycle 110).** First-principles derivation of early stopping as a transient spectral phase transition driven by covariance anisotropy. Under gradient flow on quadratic loss with anisotropic population covariance, the empirical spectral distribution of the weight matrix undergoes a BBP-like transition: outlier eigenvalues emerge and get reabsorbed as training progresses. This provides the missing analytical backbone for Route 3's basin-selection picture: g182 manifold features are empirical order parameters for exactly this BBP-like basin selection in nonlinear LLM training. If C' PASSES, cite this as first-principles support for why early geometry is predictive — the spectrum at 3% captures whether the run has undergone the productive phase transition (outlier emergence = trained basin) or not (bulk-only = random/collapsed basin). Per Codex cycle 111: this makes "spectral phase transition" less novel as a phrase but gives the analytical foundation we lacked.

**Explicit feature-to-RMT mapping (cycle 117, pre-locked before g182 data).** Per Codex §B cycle 117: alpha/PR are well-connected to RMT but ID, kNN, and depth drifts are phenomenological. Mapping them explicitly:

- **mid_spectral_alpha → BBP outlier tail decay.** In the BBP framework, the empirical spectral distribution has a bulk (Marchenko-Pastur) and outliers. Alpha measures the tail decay rate of the covariance eigenvalue distribution. Post-BBP-transition (trained basin): outliers emerge, creating a heavier tail → lower alpha. Pre-transition (random basin): spectrum is bulk-only → alpha near the MP limit. Alpha directly tracks the transition.

- **mid_participation_ratio → number of outlier eigenvalues.** PR = (sum lambda_i)^2 / sum lambda_i^2. In the BBP picture, outlier eigenvalues dominate the denominator. More outliers (= more learned modes) → higher PR. Collapsed basins have one dominant outlier → low PR. This is well-connected.

- **twonn_intrinsic_dim → effective rank of the signal subspace.** ID estimates the dimensionality of the data manifold in representation space. Under BBP, the signal subspace has dimension equal to the number of modes with eigenvalue above the BBP threshold. ID should track this count. GAP: ID is a local geometric measure (nearest-neighbor based), while the BBP signal subspace is global (eigenvalue-based). The connection holds if the manifold is locally flat in signal-relevant dimensions — plausible at early training, less so at convergence.

- **knn10_clustering_coeff → local regularity of the manifold.** High clustering = points cluster tightly around local centroids = well-separated coding regions. Under BBP: post-transition, the representation develops local structure as the signal subspace forms. Pre-transition, points are uniformly distributed in a high-dim ball (low clustering). GAP: BBP theory doesn't directly predict local clustering — this requires extending from spectral properties to geometric regularity of the embedding. The connection is PHENOMENOLOGICAL, not derived.

- **depth_alpha_drift → layerwise progression of the BBP transition.** In deep networks, each layer's weight matrix undergoes its own BBP-like transition. Depth drift measures whether deeper layers have undergone a MORE or LESS complete transition. NEGATIVE drift (alpha increasing with depth = fewer outliers in deeper layers) is the WRONG prediction under Route 3 — it should be that DEEPER layers have MORE outliers (more refined features), so alpha should DECREASE with depth in healthy runs. Correcting: negative depth_alpha_drift (alpha decreases with depth) = progressive refinement = healthy.

- **depth_pr_drift, depth_sqrt_pr_alpha_drift → layerwise spectral health gradients.** Similar to above: healthy training should show increasing PR with depth (more modes activated in deeper layers) under the BBP picture. But Route 2 (successive refinement) predicts DECREASING PR with depth (compression). This is a genuine divergence between Route 2 and Route 3 for depth drifts specifically.

---

## Which Route is Most Promising?

Route 1 (Fisher/NTK) is the most rigorous but hardest to validate empirically (NTK computation is expensive for >100M param models).

Route 2 (rate-distortion) connects most naturally to our manifesto (Intelligence = Geometry = efficient coding) and to the Umwelt paper.

Route 3 (stat-physics) is the most intuitive and directly testable (just cluster and compare).

**Recommendation:** Start with Route 3 (easiest to validate on g182 data). If clustering works, formalize via Route 2. Route 1 for a subset of cells as theoretical anchor. **Cycle 106 update:** Route 2 now has formal feature-to-rate mapping + 4 predictions (R1-R4) + 2 discriminators (D1, D2) that distinguish Route 2 from Route 3. Both discriminators implemented in g182 code. Critical divergence: Route 3 predicts step-function (basins); Route 2 predicts smooth continuous relationship. D1 tests this directly.

---

## Cross-Architecture Generalization Prediction (A15 resolver)

**Why should geometry features transfer to unseen architectures?** Under Route 3, the basins (random / collapsed / trained-attractor) are properties of the DATA + LOSS LANDSCAPE, not the architecture. Any model training on C4 next-token-prediction faces the same phase transition — the order parameters (spectral alpha, PR, ID, depth drift) characterize WHERE in the landscape the run sits, not HOW the architecture got there. A Transformer reaches the trained basin via attention; an SSM reaches it via state space dynamics; but the basin geometry is the same because the data constraint is the same.

**Testable prediction:** g184 frozen-C' (8 manifold features, trained on Qwen3+GPT-2) should predict Falcon-H1-0.5B (hybrid attention+SSM) training outcomes WITHOUT refitting, because the features measure basin identity, not architecture-specific structure. If this prediction fails, Route 3 is falsified as the mechanism (the basins would be architecture-dependent, not data-dependent).

## Route 3 Quantitative Predictions for g182 Data (pre-registered)

If Route 3 is correct, the g182 cells should exhibit specific structure:

**P1 (Basin separation at 3%).** K-means clustering (k=3) on the 8 manifold features at step 108 should produce clusters that align with final-outcome terciles (good/mid/bad) measured by fractional gain. Adjusted Rand Index between geometry clusters and outcome terciles should exceed 0.15 (chance ARI=0). Loss-only clustering (k-means on early_loss alone) should have lower ARI.

**P2 (Feature importance ordering).** Under Route 3, spectral alpha and participation ratio are the primary order parameters (they characterize the eigenvalue distribution most directly). In the Ridge model, |coef(spectral_alpha)| + |coef(participation_ratio)| should exceed the sum of all other 6 feature coefficients. If depth-drift features dominate instead, Route 2 (rate-distortion allocation across layers) is the better explanation.

**P3 (Cross-architecture feature alignment).** ~~FALSIFIED (cycle 121, 30/48 cells): 0/8 features pass KS test. Effect sizes 1.4-24.5 pooled SD. Route 3 strong form ("data-dependent basins → same feature distributions") is dead.~~ If basins are data-dependent, the DISTRIBUTION of each manifold feature should be similar across Qwen3 and GPT-2 cells within the same arm. Two-sample KS test on each feature between architectures within the same arm should yield p > 0.1 for at least 6/8 features. If most features have p < 0.05, the features are architecture-specific (falsifies Route 3).

**P4 (Frozen Ridge transfer asymmetry).** If geometry measures basin identity, the frozen Ridge transfer should be symmetric: train-on-Qwen3→test-on-GPT2 and train-on-GPT2→test-on-Qwen3 should yield similar MSE reductions. Asymmetry ratio |MSE_reduction_fold1 - MSE_reduction_fold2| / mean(MSE_reduction) < 0.5. Large asymmetry suggests the Ridge learns architecture-specific patterns, not universal basin structure.

**P5 (g184 transfer prediction).** Frozen C' trained on g182 (Qwen3+GPT-2) should predict Falcon-H1 outcomes with R² > 0 and MSE reduction ≥ 15% vs arm_mean. This is the A15 resolver. FAIL here kills Route 3.

**P6 (Landau-theory functional form).** In a mean-field Landau description of the training phase transition, the free energy near the critical point expands as F(φ) = a₀ + a₂φ² + a₄φ⁴ + ..., where φ is the order parameter (our manifold features). The equilibrium outcome (final NLL) depends on the order parameter through the equation of state ∂F/∂φ = 0, giving a relationship outcome ~ f(φ) that is generically nonlinear. If the true relationship is quadratic/nonlinear, then:
- A Ridge model with φ² features (squared manifold terms + cross-products) should beat the linear Ridge by ≥10% MSE reduction on at least one LOAO fold
- If linear Ridge already captures all signal (quadratic adds <5% improvement), the underlying relationship is monotonic in the order parameters, consistent with being far from the critical point (deep in one basin)
This is testable as a post-hoc analysis on g182 data without a new experiment — just add polynomial features to the Ridge. NOT pre-registered as a gating criterion; purely diagnostic for mechanism identification.

**P7 (Feature trajectory convergence).** If basins are attractors, manifold features should converge during training: the variance of each feature across seeds within the same arm+arch should DECREASE from step 10 to step 108. Specifically, CV(feature) at step 108 should be < CV(feature) at step 10 for at least 6/8 features. This distinguishes "basins attract trajectories" from "random noise that happens to correlate with outcome."

**Correction (cycle 106):** g182 extracts manifold features ONLY at step 108 (feature_step), not at step 10. TRAJECTORY_STEPS logs LOSSES at [10, 20, 40, 60, 80, 108, 200, 500] but not geometry features. P7 as stated is NOT directly testable on g182 data. **Proxy test (P7b):** CV(loss) across seeds within the same arm+arch should DECREASE from step 10 to step 108. This tests whether trajectories converge (basin attractor), though loss is a weaker signal than manifold geometry. A multi-step feature extraction (g182b or g184 extension) would test P7 directly.

These predictions are testable on g182 data (P1-P4, P6, D1, D2, P7b-proxy) and g184 data (P5) without additional experiments. P7 (manifold-feature convergence) requires multi-step feature extraction not in g182. They can be evaluated as part of the `--reanalyze` pass after g182 cells complete.

---

## Route 3 Verdict Matrix (PRE-LOCKED before g182 results)

**Written cycle 105 (2026-04-29) while g182 teacher gen is running. No cells have trained yet. This interpretation is locked before data arrives to prevent post-hoc narration.**

### 1. C' PASS (manifold-only beats arm_mean + telemetry + Shesha, both LOAO folds)
Route 3 SURVIVES. Claim: **pure manifold order parameters identify training basin at 3%, cross-architecture.** This is the headline finding. §0.1 → 8.6-9.0. Fire g184 Falcon-H1 immediately.

### 2. C PASS but C' FAIL
Route 3 WEAKENED. The norm/variance features in C but not C' are carrying the signal — this is optimization-health monitoring, not pure geometry/basin identification. Claim narrows to: "early activation statistics predict training health" (less distinctive, closer to what existing work does). §0.1 → 7.0-7.5.

### 3. Combined telemetry baseline beats C/C'
PIVOT. The signal is in ordinary training diagnostics (loss trajectory, gradient stats), not geometry. Do not frame as Neural Genome evidence. The geometry-as-instrument thesis is falsified for this application. §0.1 → 4.0-4.5. Redirect to pure theory (Route 1/2 derivations) or g155 energy efficiency.

### 4. Shesha (Model E) ties or beats C/C'
MOAT DIES. The result becomes replication/extension of the Geometric Canary (arXiv 2604.17698), not a distinctive moonshot finding. Honest framing: "we independently confirmed that representational geometry predicts training health, consistent with concurrent work by [Shesha authors]." Credit them. §0.1 → 5.0-6.0 (replication, not discovery).

### 5. Only Model A or B pass (not C/C')
Reference leakage or mixed-feature success. The Qwen3-reference features or the telemetry components of B are doing the work, not pure geometry. Cannot claim cross-architecture geometry diagnostic. §0.1 → 6.0-6.5.

### 6. g182 PASS but P3/P4 FAIL
The predictor works but Route 3's mechanism ("basins are data-dependent") is false. The features may be architecture-specific embeddings that happen to correlate with outcome within each family. LOAO transfer works because Qwen3 and GPT-2 are both English-decoder-Transformers, not because basins are universal. g184 becomes even MORE critical — if Falcon-H1 frozen-C' also fails, the "data-dependent basin" story is dead. §0.1 → 7.0-7.5 (tool works, theory wrong).

### 7. g182 PASS + P3/P4 PASS + g184 frozen-C' PASS (>15% MSE reduction, R²>0, no refit)
Route 3 CONFIRMED at 3-family level. Claim: **manifold geometry at 3% of training identifies data-dependent training basins that predict outcome, generalizing to unseen architecture families without refitting.** This is the flagship result. §0.1 → 9.0+.

### 8. g182 PASS + g184 frozen-C' FAIL
Tool works within Transformer family but doesn't generalize to hybrid-SSM. Basins may be architecture-class-dependent (Transformer basin ≠ SSM basin). Honest framing: cross-Transformer diagnostic. Still valuable but not universal. §0.1 → 8.0-8.5.

### 9. C' PARTIAL PASS (one LOAO fold only) — added cycle 117
C' beats baselines on one fold (e.g. train-Qwen3→test-GPT2) but fails on the other. This means geometry is informative in one direction but not symmetric. Cannot claim cross-architecture generalization — the predictor is family-specific. Investigate which direction fails: if GPT-2→Qwen3 fails, likely Qwen3 features carry more architecture-specific signal. §0.1 → 6.5-7.0.

### 10. C' PASS but P1/D1/D2 contradict mechanism — added cycle 117
C' passes the regression criteria (MSE, R², AUROC) but the mechanism diagnostics fail: D1 shows basin-mean beats continuous Ridge (Route 3 favored over Route 2, but basins don't separate cleanly per P1), or D2 shows depth drifts carry no marginal signal, or P1 ARI is near-zero despite good regression. This means the Ridge works as a black-box predictor but the theory (why it works) is wrong. Claim: "geometry predicts training health" (tool claim) but NOT "because of symmetry-breaking basins" (theory claim). §0.1 → 7.5-8.0.

### 11. Inconclusive: operational failure — added cycle 117
Cells fail to train (NaN, OOM, cache corruption), fewer than 40/48 cells complete, or systematic issues (e.g. all scratch cells have identical NLL, suggesting data loading bug). Cannot draw conclusions. Fix operational issues and re-run. §0.1 → unchanged from prior.

---

## g182 ACTUAL OUTCOME (cycle 124, 2026-04-29)

**Closest scenario: 3 (combined telemetry beats C/C'), but WORSE — ALL models fail including baselines.**

The experiment was underpowered for cross-architecture transfer: within-arm label variance (std=0.002-0.003) was too small for any Ridge to learn a transferable mapping across architectures with completely non-overlapping feature distributions (P3 falsified, 0/8).

**Actual results do not cleanly match any pre-locked scenario** because no scenario anticipated ALL models (including baselines) failing simultaneously. The root cause is not that geometry is uninformative — it is that the binary KD-vs-scratch design produces near-constant labels within each (arch, arm) group, making cross-architecture regression impossible at n=12 per fold.

**Surviving signal (not anticipated in verdict matrix):** Pairwise delta R2=0.518 — seed-matched geometric changes (scratch->KD) predict NLL changes within each architecture. This is an architecture-specific causal-response diagnostic, not a universal basin predictor.

**Verdict: §0.1 = 4.8-5.0/10.** Cross-architecture transfer is dead at this sample size/label variance. Route 3 universal basin language is dead. The remaining claim is: "early geometry of a causal intervention predicts whether that intervention helps or harms final training, within a fixed architecture." Codex-validated: `codex_outputs/heartbeats/cycle124_advisor_g182_final_20260429.md`.

**Next: g186 balanced causal dose-response.** Multiple KD strengths create real within-arm label variance. Kill-or-promote: PASS -> §0.1 ~6.5; FAIL -> retire Forecast direction.

### Intervention Susceptibility Extension (cycle 133, pre-staged)

If g186 PASS: Route 2 predicts the dose-response is SMOOTH and CONTINUOUS (water-filling rebalances gradually with alpha). The Ridge coefficients from g186 estimate the linear component of d(final_NLL)/d(geometry_delta), which under Route 2 is the marginal rate reallocation from KD perturbation. Formalizing this as "intervention susceptibility" (Codex §B cycle 128): at 3% of training, geometry estimates how the system will respond to the KD perturbation at each dose. g185v2 tests whether this estimate is accurate enough to SELECT the optimal dose prospectively (prereg: `research/prereg/genome_185v2_dose_selection_optimization_2026-04-30.md`). Route 2 predicts smooth selection surfaces; Route 3 predicts dose basins with sharp boundaries between beneficial and harmful regimes. The dose-response curve shape from g186 itself is a discriminator (see Codex §B cycle 128).
