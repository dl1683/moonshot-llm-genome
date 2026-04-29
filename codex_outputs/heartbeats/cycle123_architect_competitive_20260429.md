Honest odds of a **finding beyond arm identity**: **2/10**, maybe 15-25%.

1. Ratings:
(a) raw LOAO passes: **8/10**. It can exploit architecture/KD magnitude: GPT-2 KD hurts less than Qwen3 KD.
(b) arm-demeaned LOAO passes: **2/10**. Within-KD LOO R2=-0.52 is the warning sign.
(c) pairwise delta test passes: **2-3/10**. Possible if Qwen3 within-arm feature effects are real, but current evidence says arm identity dominates.

2. If raw passes and arm-demeaned fails, the honest claim is: **early geometry detects architecture-mediated KD susceptibility / intervention identity**, not training health. That is diagnostic, but likely **§0.1 = 4.5-5.5**, not 7+, unless it predicts an actionable held-out decision better than early loss: stop, switch teacher, change tokenizer/init, save compute.

3. RMT still applies, but **conditionally**. The RMT paper’s spectral phase-transition story is compatible with within-architecture basin/order-parameter behavior. P3 falsification says the same spectral/geometric coordinates are not universal cross-architecture predictors here. So: RMT as **architecture-local thermodynamics**, not yet a universal atlas. Source: [RMT spectral learning paper](https://www.sciencedirect.com/science/article/pii/S0378437126002104).

4. If arm-controlled tests fail: stop adding rows. Run a **balanced causal dose experiment**: scratch + weak/medium/strong KD + maybe bad-teacher KD, per architecture, labels as paired deltas versus scratch, pre-registered arm-demeaned/pairwise tests, baseline against early loss. If tests pass: immediately replicate on a **held-out third architecture and non-KD intervention**. The next bar is compute-triage utility, not prettier correlation.

5. Recent adjacent work exists, but I did **not** find a direct 2025-2026 knock-out paper doing cross-architecture geometry prediction of training outcomes. Closest: [Anatomy of Capability Emergence, arXiv 2602.15997](https://papers.cool/arxiv/2602.15997), which finds geometry can precede emergence but explicitly notes prediction limits; [relative embedding alignment across forecaster architectures](https://openreview.net/pdf/c155ec2f41d50f3a13d9e491a8b3a6c6e6c8a762.pdf), accuracy-correlated but domain-specific; [Skean et al. ICML 2025](https://proceedings.mlr.press/v267/skean25a.html), layer geometry/quality, not training-outcome forecasting.

