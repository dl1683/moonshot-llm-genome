# Derivation: Early Geometry Predicts Training Health

**Status:** SKELETON (cycle 96, 2026-04-29). Three routes proposed by Codex Architecture Theorist.

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

---

## Which Route is Most Promising?

Route 1 (Fisher/NTK) is the most rigorous but hardest to validate empirically (NTK computation is expensive for >100M param models).

Route 2 (rate-distortion) connects most naturally to our manifesto (Intelligence = Geometry = efficient coding) and to the Umwelt paper.

Route 3 (stat-physics) is the most intuitive and directly testable (just cluster and compare).

**Recommendation:** Start with Route 3 (easiest to validate on g182 data). If clustering works, formalize via Route 2. Route 1 for a subset of cells as theoretical anchor.

---

## Cross-Architecture Generalization Prediction (A15 resolver)

**Why should geometry features transfer to unseen architectures?** Under Route 3, the basins (random / collapsed / trained-attractor) are properties of the DATA + LOSS LANDSCAPE, not the architecture. Any model training on C4 next-token-prediction faces the same phase transition — the order parameters (spectral alpha, PR, ID, depth drift) characterize WHERE in the landscape the run sits, not HOW the architecture got there. A Transformer reaches the trained basin via attention; an SSM reaches it via state space dynamics; but the basin geometry is the same because the data constraint is the same.

**Testable prediction:** g184 frozen-C' (8 manifold features, trained on Qwen3+GPT-2) should predict Falcon-H1-0.5B (hybrid attention+SSM) training outcomes WITHOUT refitting, because the features measure basin identity, not architecture-specific structure. If this prediction fails, Route 3 is falsified as the mechanism (the basins would be architecture-dependent, not data-dependent).

## Route 3 Quantitative Predictions for g182 Data (pre-registered)

If Route 3 is correct, the g182 cells should exhibit specific structure:

**P1 (Basin separation at 3%).** K-means clustering (k=3) on the 8 manifold features at step 108 should produce clusters that align with final-outcome terciles (good/mid/bad) measured by fractional gain. Adjusted Rand Index between geometry clusters and outcome terciles should exceed 0.15 (chance ARI=0). Loss-only clustering (k-means on early_loss alone) should have lower ARI.

**P2 (Feature importance ordering).** Under Route 3, spectral alpha and participation ratio are the primary order parameters (they characterize the eigenvalue distribution most directly). In the Ridge model, |coef(spectral_alpha)| + |coef(participation_ratio)| should exceed the sum of all other 6 feature coefficients. If depth-drift features dominate instead, Route 2 (rate-distortion allocation across layers) is the better explanation.

**P3 (Cross-architecture feature alignment).** If basins are data-dependent, the DISTRIBUTION of each manifold feature should be similar across Qwen3 and GPT-2 cells within the same arm. Two-sample KS test on each feature between architectures within the same arm should yield p > 0.1 for at least 6/8 features. If most features have p < 0.05, the features are architecture-specific (falsifies Route 3).

**P4 (Frozen Ridge transfer asymmetry).** If geometry measures basin identity, the frozen Ridge transfer should be symmetric: train-on-Qwen3→test-on-GPT2 and train-on-GPT2→test-on-Qwen3 should yield similar MSE reductions. Asymmetry ratio |MSE_reduction_fold1 - MSE_reduction_fold2| / mean(MSE_reduction) < 0.5. Large asymmetry suggests the Ridge learns architecture-specific patterns, not universal basin structure.

**P5 (g184 transfer prediction).** Frozen C' trained on g182 (Qwen3+GPT-2) should predict Falcon-H1 outcomes with R² > 0 and MSE reduction ≥ 15% vs arm_mean. This is the A15 resolver. FAIL here kills Route 3.

These predictions are testable on g182 data (P1-P4) and g184 data (P5) without additional experiments. They can be evaluated as part of the `--reanalyze` pass after g182 cells complete.
