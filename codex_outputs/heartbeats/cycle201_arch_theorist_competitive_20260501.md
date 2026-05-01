**Verdict**
Current trajectory: **6.0/10 now**, **6.1/10 only after 28-layer row_shuffled finishes cleanly**, **6.8-7.0/10 if g197 PASS_CANARY**, and **7.3+ only if direct `M_W` beats proxy features in g199**.

The repo’s live claim is now exactly the right one: “earliest token/embedding/interface geometry predicts whether training will be healthy” [CLAUDE.md:11](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/CLAUDE.md:11>). The competitive bar is first-principles or compute-saving prediction, not another geometry survey [CLAUDE.md:52](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/CLAUDE.md:52>).

**Architecture Theorist**
Yes, `M_W = C_h^{1/2} W_out^T (Diag(pi)-pi*pi^T) W_out C_h^{1/2}` gives a testable prediction beyond depth amplification. The direct test is: compute the eigen-spectrum of `M_W` for every g197 lm_head condition at step 0/50, then test whether

```text
Phi(W) = sum_i log(1 + mu_i / theta)
```

predicts final NLL better than early loss, norm-only features, spectral proxies, and g197’s hand-built geometry features. This is already implied by the derivation doc [early_geometry_predicts_training_health.md:67](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/derivations/early_geometry_predicts_training_health.md:67>) and by the g197 design [EXPERIMENTS.md:7](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/experiments/EXPERIMENTS.md:7>).

But: you **cannot derive optimal anchor lambda from the spectrum alone**. Spectrum gives mode speeds, not anchor correctness. In a quadratic mode model,

```text
dz_i/dt = -eta[ mu_i(z_i - z_i*) + lambda(z_i - a_i) ]
```

`mu_i` comes from `M_W`, but optimal `lambda` also needs `delta_i = a_i - z_i*`, the anchor-target error per mode. If `delta_i = 0`, the toy optimum goes to infinite lambda; if `delta_i` is large, lambda must shrink. So the honest result is: **closed-form lambda requires spectrum plus an anchor-noise/alignment model**, not `M_W` alone.

Depth scaling should **not** grow without bound. The 28-layer result is only +14% relative over 8-layer despite 3.5x depth: +0.530 vs +0.465 [WIKI.md:33](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/WIKI.md:33>). That already screams diminishing returns. A good law to pre-register:

```text
Delta(L) = Delta_max * (1 - exp(-k * S_L))
S_L = sum_{l=1}^L aligned_gradient_transfer_l
```

If layers are roughly homogeneous, use `Delta(L) ~= Delta_max(1-exp(-L/L_c))`. If Route 2 water-filling is right, an alternate form is `Delta(L) ∝ sum_i log(1 + L*mu_i/theta)` until it saturates against task/noise limits. Run 4/8/16/28/40 layers to distinguish exponential saturation from log growth.

**Competitive Analyst**
I do **not** see a direct last-3-month competitor that says “output-layer geometry predicts pretraining health.” Closest papers:

- **Lost in Backpropagation: The LM Head is a Gradient Bottleneck**, Mar 10 2026. Very relevant: it says the LM head suppresses 95-99% of gradient norm and strongly affects LLM training dynamics, but it is not a health predictor. It supports your `W_out^T(p-e_y)` mechanism. Source: arXiv 2603.10145 lines 34-44.
- **Geometric Metrics for MoE Specialization**, Apr 16 2026. Fisher geometry predicts MoE training failure at 10% completion with AUC 0.89, but it is routing-simplex geometry, not lm_head geometry. Source: arXiv 2604.14500 lines 30-40.
- **Fast and Accurate Probing of In-Training LLMs’ Downstream Performances**, Apr 1 2026. Uses lightweight probes on internal reps, AUROC >0.75, not output-interface geometry. Source: arXiv 2604.01025 lines 30-41.
- **The Geometric Canary**, Apr 20/25 2026. Predicts steerability/drift via representational stability, not pretraining health. Source: arXiv 2604.17698 lines 30-40.
- Grokking papers like ILDR and spectral entropy are nearby but toy/generalization-transition scoped, not LLM output-interface scoped.

Compared to **muP / Tensor Programs / maximal update parametrization**: they solve scale-stable parametrization and hyperparameter transfer. Tensor Programs V says many optimal HPs stay stable under `muP`, enabling small-to-large zero-shot HP transfer, including reported 7% tuning cost for a GPT-3-scale example (Microsoft Research lines 17-21). Your §0 framing is different: **muP asks whether updates scale correctly with width; this project asks whether the output codebook supplies the right gradient geometry for a run to become healthy.** They are compatible, but an adversary will ask whether your effect survives `muP`-clean parametrization.

**Weakest Link**
The weakest empirical link is **g197**, because it is code complete, not evidence. The whole claim becomes a real training-health predictor only if g197 beats step-50 loss with held-out condition generalization [WIKI.md:35](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/WIKI.md:35>).

The weakest theoretical link is sharper: **plain `M_W` may be blind to semantic row assignment.** A frequency-bucket shuffle can preserve much of the spectrum while destroying exact token identity. g191/g194/g195 say content, direction, and lm_head matter [EXPERIMENTS.md:127](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/experiments/EXPERIMENTS.md:127>), [EXPERIMENTS.md:78](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/experiments/EXPERIMENTS.md:78>), [EXPERIMENTS.md:53](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/experiments/EXPERIMENTS.md:53>). But `M_W` as covariance may explain curvature without proving correct codebook semantics.

Invest next in this order:

1. Finish g192 row_shuffled. Do not fully claim depth pass until the 28-layer wrong-row control is harmful.
2. Run g197 exactly as locked.
3. Add g199: direct `M_W` eigenspectrum vs final NLL, leave-one-condition-out.
4. Add isospectral wrong-codebook controls: same `M_W` spectrum, wrong token assignment. If they fail while trained rows pass, the final theory must include a supervised alignment term, not just Fisher spectrum.
5. Run g196 to separate persistent basin residue from active regularization.

That makes the chain hard to attack: content → direction → output dominance → depth persistence → prospective health prediction → direct operator.

Sources: arXiv pages for [LM head bottleneck](https://arxiv.org/abs/2603.10145), [MoE Fisher failure detection](https://arxiv.org/abs/2604.14500), [in-training probes](https://arxiv.org/abs/2604.01025), [Geometric Canary](https://arxiv.org/abs/2604.17698), and [Tensor Programs V](https://www.microsoft.com/en-us/research/publication/tuning-large-neural-networks-via-zero-shot-hyperparameter-transfer/).

