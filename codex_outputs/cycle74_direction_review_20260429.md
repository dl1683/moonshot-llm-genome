Brutal answer: the diagnostic pivot is right, but the current trajectory is **5.2/10**, not breakthrough yet. g180 is a real signal, but still a weak-pass Ridge result: 61.6% MSE reduction, CI crossing zero, n=9 held-out. Also, `CLAUDE.md` appears stale on the score; `WIKI.md`/g180 result/user context are the fresher 5.0-5.3 baseline.

**1. Theory**
The best framework is: **early token/interface geometry is the model’s initial rate-distortion code for language.**

Tokenizer + embedding + lm head define the source alphabet, distortion metric, and first coordinate chart. SGD then moves on a Fisher/information-geometric manifold. Healthy training is not “low early loss”; it is early movement toward the rate-distortion frontier for compressing token contexts into predictive hidden states.

The derivation to aim for:

`source P_tau(x,y)` from tokenizer `tau`  
`representation Z_l = f_l(x)` with effective rate `R_l ~= spectral entropy / effective rank / PR`  
`distortion D = E[-log p(y|Z_L)]`  
healthy early state minimizes:

`F(q_t) = D(q_t) + beta R(q_t) + lambda * geodesic_distance(q_t, q*_RD) + noise`

Then prove final excess loss is bounded by early free-energy/RD-excess:

`L_final - L*_family >= a * Delta_RD(t<=3%) + b * log kappa_Fisher + noise`

That would elevate this from phenomenology to law. The key is to derive a **closed-form scalar**, not another trained Ridge. Use rate-distortion / information bottleneck as the compression frame, Amari-style Fisher geometry for optimizer conditioning, and statistical mechanics for phase boundaries / escape probabilities. Relevant foundations: [Tishby-Zaslavsky IB](https://arxiv.org/abs/1503.02406), [Amari natural gradient](https://direct.mit.edu/neco/article/10/2/251/6143/Natural-Gradient-Works-Efficiently-in-Learning), [Bahri et al. statistical mechanics](https://www.annualreviews.org/doi/10.1146/annurev-conmatphys-031119-050745).

**2. Competitive Reality**
Closest external threat is already here: Google/Mila’s 2025 representation-geometry paper tracks spectral phases during LLM training and explicitly says loss/gradient norms hide meaningful representational phases: [Google Research](https://research.google/pubs/tracing-the-representation-geometry-of-language-models-from-pretraining-to-post-training/). So “geometry changes during training” is not distinctive anymore.

Other neighbors:
- DeepMind/Chinchilla: compute-optimal scaling and waste reduction, but loss/compute frontier, not token-interface geometry: [DeepMind](https://deepmind.google/discover/blog/an-empirical-analysis-of-compute-optimal-large-language-model-training/).
- Anthropic: SAE/feature geometry and universality, but mostly post-hoc interpretability, not early training triage: [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html), [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/).
- MIT/Isola PRH: representational convergence, not training diagnostics: [PRH](https://arxiv.org/abs/2405.07987). The 2026 “Aristotelian” critique also pushes toward local-neighborhood geometry, which is close to your kNN line: [arXiv](https://arxiv.org/abs/2602.14486).
- Pretraining-checkpoint diagnostics exist: [The Fine Line](https://arxiv.org/abs/2404.01204).
- Early-training performance prediction exists outside LLM pretraining, e.g. neural capacitance: [Nature Comm. 2024](https://www.nature.com/articles/s41467-024-48069-8).

Your distinctive angle must be: **frozen prospective geometry forecast at <=3% training, cross-tokenizer, with a compute stop/restart/repair policy**. Big labs can publish geometry papers. What they are less likely to publish is: “our early interface geometry says many expensive runs are doomed/wasteful before loss admits it.”

**3. Highest-Leverage Next Experiment**
After g180b/g181b, do **g182: Prospective RD-Geometry Triage + Rescue**.

Do not just run g180c scale-up. That is useful but likely caps at ~8/10. g182 should be prospective and causal:

- Freeze g180/g180b forecast.
- Add a theory-derived `Delta_RD` scalar before the run, not fitted on g182.
- Run fresh held-out cells: tokenizer x stress-regime x seed.
- At <=3% training, declare each run healthy / wasteful / doomed.
- Shadow-policy: train all cells to completion anyway, but score simulated stop/restart savings.
- Add a randomized rescue arm for predicted-doomed cells: geometry repair vs continue.

PASS bar:
- geometry/RD beats early loss with CI > 0,
- AUROC >= 0.85 for doomed/wasteful,
- simulated compute saved >=30% with no false stop on high-value runs,
- rescue recovers >=40-50% of final NLL gap.

Score if PASS: **8.8-9.2/10**.  
Diagnostic-only version: **7.4-7.8/10**.  
g180c scale-only: **~8/10 ceiling**.  
If g180b fails, the 9+ diagnostic path is dead for now; pivot to tokenizer-prior benchmark and call it a narrower negative result.

**4. Pivot vs Direct Transfer**
The pivot is correct. Direct capability transfer is still the endgame, but the current evidence says the naive route is dead: donor identity mostly collapsed, cross-arch transfer failed its locked criterion, and g181a says transformer-block anchoring harms while embed/lm-head carries the effect.

So: do not chase more anchor/transfusion variants now. Use the diagnostic line to identify the actual control variable. Once you can forecast and rescue geometry, transfer becomes a compiler problem again. Right now, direct transfer mainline is **3/10**; diagnostic-as-observatory is **5.2/10 now, 7.5 if g180b passes, 9 only if it becomes prospective control/compute savings**.

