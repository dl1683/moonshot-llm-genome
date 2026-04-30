**Verdict: g186 FAIL kills Forecast/Diagnostic as the headline.** The result favors Route 2 on *curve shape* only, not on the proposed geometry mechanism.

Five discriminators:
1. Held-out dose: fails badly, so no smooth transferable geometry law.
2. Continuous geometry vs arm/basin mean: geometry R2=0.022; arm_mean R2=0.936. Fatal.
3. Depth-drift value: moot; all 8 features fail.
4. Alpha leakage: D5 R2=0.364, alpha_quad R2=0.774. Geometry mostly encodes dose, weakly.
5. Curve shape: smooth concave peak at alpha=1.0 in both archs. This rejects Route 3 phase-transition/basin language and weakly supports Route 2 water-filling saturation.

Theory read: the KD intervention has a clean rate-distortion-like dose curve, but the 8-feature manifold probe does not measure the controlling variable. Spectral alpha/PR are somewhat grounded; TwoNN/kNN/depth drifts are now mostly fishing unless tied to direct NTK/Fisher/RD quantities.

Competitive scan Apr 28-30: no direct KD dose-optimization or early-geometry diagnostic competitor found. Closest: TIDE cross-architecture distillation, PAINT self-distilled reasoners, Entrocraft entropy-curve training control, and Shorthand-for-Thought diagnostic early-stopping signals. Relevant, not a direct g186 hit. Sources: [arXiv recent Apr 28-30](https://arxiv.org/list/cs.LG/recent), [TIDE](https://arxiv.org/abs/2604.26951), [PAINT](https://arxiv.org/abs/2604.26573), [Entrocraft](https://arxiv.org/abs/2604.26326), [Shorthand](https://arxiv.org/abs/2604.26355).

§0.1 scoring:
- If g186 had PASSed: 6.3-6.7; g185v2 PASS could reach ~7.1.
- Actual FAIL: 3.8-4.1. Not a DeepMind-cannot-produce result.

Rescue: archive g185v2. Do not add architectures. Salvage Route 2 as a direct KD dose-response law: derive alpha* from teacher-student KL/entropy/coverage/Fisher, then beat alpha_quad/arm_mean on held-out doses. Otherwise pivot to g183/corpus-init or efficiency work.