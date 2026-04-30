Yes, conditionally sound. Frozen-Ridge-from-g186 is valid for prospective dose selection if the whole artifact is frozen: feature order, scaler, imputation, Ridge alpha, coefficients, and “all-negative -> scratch” rule. Training the final Ridge on all 48 g186 rows after PASS is fine for deployment; g185v2 fresh seeds are the actual prospective test.

Best Route 2/3 discriminators on g186:

1. Held-out-dose stress: smooth interpolation/extrapolation favors Route 2; sharp failures at alpha boundaries favor Route 3.
2. Continuous Ridge vs k-means/basin mean: Ridge > basin by >10% MSE favors Route 2; basin ties/beats favors Route 3.
3. Depth-drift ablation: all 8 features beating no-depth model by >5% favors Route 2; no loss favors Route 3.
4. Alpha leakage: if geometry decodes alpha and alpha-only performs, neither theory wins.
5. Curve shape: concave/smooth dose-response favors water-filling; threshold/harm basin at high alpha favors phase transition.

Competitive: I found no direct Apr 28-30 paper on KD dose optimization. Closest new/adjacent items are Apr 30 arXiv listings: PAINT self-distilled reasoners, entropy-curve control for RL training, ReGATE teacher-student token elision, and “Shorthand for Thought” noting diagnostic signals for RL early stopping. Relevant but not a direct g186/g185v2 hit. Existing direct pressure remains in-training probes and distillability/traps, but those are earlier April papers. Sources: [arXiv cs.LG recent](https://arxiv.org/list/cs.LG/recent), [arXiv cs.CL new](https://arxiv.org/list/cs.CL/new), [in-training probes](https://arxiv.org/abs/2604.01025), [distillation traps](https://arxiv.org/abs/2604.18963).

Score assuming g186 PASS: 7/10 expected uplift chain. g186 alone gets ~6.5; g185v2 PASS can move it to ~7.1-7.3 because it becomes compute-saving engineering, not just diagnostic science.

