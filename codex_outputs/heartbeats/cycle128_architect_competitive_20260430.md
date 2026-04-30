**Architecture Read**

For g186, the clean discriminator is curve shape.

Route 2, rate-distortion, predicts a **smooth residual dose-response**. Low-to-mid KD should improve allocation; high KD may saturate or harm by over-allocating to teacher/interface artifacts. Best-looking result: continuous geometry Ridge beats cluster/arm means, held-out-dose performance degrades gracefully, depth-drift features add value, and alpha-only cannot explain the curve. Shape: concave, saturating, or inverted-U.

Route 3, stat-physics, predicts **thresholds/basins**. Alpha acts like a control parameter: scratch basin, teacher-guided basin, maybe collapse/interface basin at alpha=2.0. Best-looking result: abrupt kinks, seed/architecture-specific thresholds, geometry clusters explain outcome nearly as well as Ridge. Depth drift should behave more like an order parameter than an independent predictor.

The must-watch checks are: label variance, geometry beating `alpha-only`, `alpha^2`, `(arch, alpha)`, telemetry, and Shesha; both architectures non-negative R2; held-out-dose not collapsing; alpha decodability not explaining everything. If alpha or telemetry ties geometry, the claim dies.

If g186 passes, the strongest next theory move is not “add Falcon.” It is to formalize **intervention susceptibility**: KD alpha is a perturbation, early geometry estimates `d(final_NLL)/d(alpha)` or basin transition probability. Then run g185 as prospective triage-to-action: choose/avoid KD dose early and show compute savings. That turns a predictor into a control law.

If g186 fails, the honest read is severe: g182’s pairwise delta was likely an alpha=1 slice, telemetry artifact, or post-failure salvage. Route 3 universal basin language is already dead; Route 2 survives only as broad inspiration, not as this diagnostic. Forecast should stop being central unless g183 finds a different mechanism.

**Competitive Read**

New/relevant April 28-30 scan: no direct g186 clone found. But several nearby papers tighten the field:

- [Architecture Determines Observability in Transformers](https://arxiv.org/abs/2604.24801): very relevant. Activation monitoring depends on architecture/training recipe; checkpoint dynamics can form then erase observability. This attacks universal diagnostic framing.
- [Gradient-Direction Sensitivity](https://arxiv.org/abs/2604.25143): SVD-on-gradients diagnostic, grokking acceleration around 2.3x. Competes with “spectral/geometry predicts transition,” but in toy grokking space.
- [Teacher Forcing as Generalized Bayes](https://arxiv.org/abs/2604.25904): relevant warning. Intervention losses can create geometry mismatches versus the target objective. High-alpha g186 harm would fit this.
- [Sustained Gradient Alignment](https://arxiv.org/abs/2604.25779), also on [OpenReview](https://openreview.net/forum?id=UJM4H9oLJN): KD effects may be gradient-alignment mechanics, not representation geometry.
- [Knowledge Distillation Must Account for What It Loses](https://arxiv.org/abs/2604.25110): not a geometry competitor, but supports evaluating KD by off-metric losses, not headline scores.

Direct competitor assessment: none exact for **additive KD dose-response + seed-matched delta geometry + held-out seeds**. Closest moat threats are [Geometric Canary/Shesha](https://huggingface.co/papers/2604.17698), [ILDR](https://arxiv-troller.com/?q=paper%3A+2602.23128), and grokking spectral/order-parameter work like [Spectral Entropy Collapse](https://goatstack.ai/articles/2604.13123).

If g186 passes at R2 ~0.3-0.5: good, but not world-beating yet. It is stronger than toy grokking on practical relevance and preregistered falsification, weaker on elegance and headline metrics. Geometric Canary has much bigger correlations and breadth. ILDR has clearer actionability and FLOP savings. g186’s edge is causal LLM-training intervention response. To compete, it needs g185: “we predicted the useful KD dose early and saved compute.” Without that, pass means ~6.5/10, not a breakthrough.

I did not interpret partial 13/60 cells; that would invite exactly the post-hoc drift this experiment is meant to avoid.

Sources used: local [WIKI.md](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/WIKI.md:29>), [theory doc](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/derivations/early_geometry_predicts_training_health.md:85>), [g186 prereg](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/prereg/genome_186_dose_response_2026-04-29.md:143>), [cycle125 adversarial](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/codex_outputs/heartbeats/cycle125_adversarial_20260430.md:1>), and arXiv cs.LG new listings for Apr. 29, 2026.


2026-04-30T01:22:56.327135Z ERROR codex_core::session: failed to record rollout items: thread 019ddbf6-acf7-76a2-9f11-a1081f9dafd1 not found
