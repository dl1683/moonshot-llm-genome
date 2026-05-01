**Bottom Line**

Right now this is a **6.0/10 project**, not 6.8, not 8. The repo’s own advisor says g192 moves it to about **6.1** and “the headline is still not proven” until g197 ([advisor](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/codex_outputs/g192_advisor_20260501.md:2>)). I agree, but I’d round down because the main result is still **single-family output-interface mechanics**, not training-health prediction.

**1. Evidence Audit**

What is actually proven:

- Correct trained **output/interface row directions** causally improve Qwen3-family training; wrong row directions are toxic. g194 shows direction carries 95-97% of the signal; g195 shows lm_head carries 65% of the tied signal; g192 shows the effect persists and amplifies at 28 layers ([g194/g195/g192 log](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/experiments/EXPERIMENTS.md:21>), [g192 JSON](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/results/genome_192_28layer_replication.json:216>)).
- Strong-form transfer is dead. Donor identity, zero-step transfer, activation scaffold, cross-architecture efficiency, corpus-derived priors, OT tokenizer bridges, and cross-arch early-geometry prediction all failed ([CLAUDE](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/CLAUDE.md:13>), [claim map](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/CLAIM_EVIDENCE_MAP.md:88>)).
- The central surviving finding is: **tokenizer/output-interface codebook plus architecture decoder**, not universal transferable “genome” ([Mystery 8](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/OPEN_MYSTERIES.md:158>)).

What is being claimed but not proven:

- “Earliest geometry predicts healthy/wasteful/doomed training.” g182 and g186 failed. g197 is only coded, not run ([claim map](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/CLAIM_EVIDENCE_MAP.md:89>), [g197 prereg](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/prereg/genome_197_output_interface_canary_arena_2026-04-30.md:11>)).
- Cross-architecture universality. Current evidence argues against it.
- First-principles law. Current theory is a good mechanistic decomposition, not yet a derivation.

Peer-review survivable claims:

- **Yes:** “Output lm_head/token-row directions are causal training coordinates in Qwen3-family shells.”
- **Yes:** “Cross-architecture representational charts are not commensurate under these primitives.”
- **No:** “Neural Genome maps learning across all AI models.”
- **No:** “Early geometry predicts training health generally.”
- **No:** “Model surgery / capability transfer without retraining.”

Overfit to Qwen3-family:

- C18/C21/C23/C25/C26/C27/g192 are all highly Qwen3-family / exact-token-row dependent.
- The strongest evidence literally depends on exact string overlap and Qwen3-shaped decoder compatibility. That is not a bug; it is the finding.

**2. Theory Assessment**

M_W is **promising but not yet a theoretical contribution**. It is the natural Fisher pullback of cross-entropy through `W_out`; that is mathematically clean, but not yet a new law. The derivation defines `M_W = C_h^{1/2} W_out^T (Diag(pi)-pi pi^T) W_out C_h^{1/2}` and proposes `Phi(W)` as the diagnostic ([derivation](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/derivations/early_geometry_predicts_training_health.md:67>)). But g199 has not tested it.

Score: **M_W now = 5.5/10. If g199 passes with isospectral wrong-codebook controls, 7.5/10.**

Route 2 has one genuinely novel prediction:

- **Novel:** initial/few-step output-Fisher codebook spectrum predicts final NLL better than early loss and hand-built geometry features.
- **Not novel:** “geometry predicts health” broadly. Current arXiv already has Geometric Canary, in-training representation probes, Fisher MoE failure prediction, and spectral training dynamics: [Geometric Canary](https://arxiv.org/abs/2604.17698), [in-training probes](https://arxiv.org/abs/2604.01025), [MoE Fisher metrics](https://arxiv.org/abs/2604.14500), [Spectral Lifecycle](https://arxiv.org/abs/2604.22778).

First-principles status: **not yet**. It derives a plausible operator from CE gradients and output Fisher geometry, but it does not derive trained row directions, token alignment, optimal anchor strength, or cross-chart compatibility from first principles. The derivation itself admits optimal lambda needs an extra anchor-noise/alignment model ([derivation](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/research/derivations/early_geometry_predicts_training_health.md:88>)).

**3. Strategic Assessment**

Actual moat as of May 1, 2026:

- **Moat:** ruthless falsification chain plus a sharp output-interface mechanism.
- **Not moat:** spectral geometry, early probes, Fisher diagnostics, or “geometry predicts health.” Those are crowded.
- **Best wedge:** output-interface codebook as a causal training operator.

If published today:

- arXiv reception: “Interesting, unusually honest, strong negative-result chain.”
- Top venue reception: likely **reject/borderline** unless reframed narrowly. Reviewers would attack single-family scope, manipulated conditions, exact-token overlap, missing natural-run prediction, and lack of M_W test.
- If marketed as Neural Genome universality, it gets torn apart.

Single most valuable product:

> **The output interface is not a passive readout. Its token-row directions define gradient prototypes; correct codebook directions help training, wrong ones poison it, and this persists at full depth.**

If reduced to one finding, cut everything else and keep that.

**4. Gap Analysis**

Critical missing experiments:

- g197 result: controlled canary, geometry vs early loss.
- g196 result: residue vs active regularization.
- g199 direct M_W eigenspectrum, with isospectral wrong-codebook controls.
- Natural-run forecast: vary LR, data quality, tokenizer, depth, width, seed; predict final outcome without deliberately poisoning lm_head.
- Non-Qwen family replication: Llama/Pythia/GPT-2 native output heads, not Qwen3 shell.
- μP-clean parameterization control.
- Strong baselines: Shesha, in-training probes, loss slope, telemetry, arm means.

To move §0.1 from ~6 to 8+:

1. g197 passes.
2. g199 shows `Phi(W)` beats proxy features and early loss.
3. Natural-run prediction works on held-out training regimes.
4. At least one non-Qwen family works.
5. The diagnostic changes a real decision: early stop, choose head init, avoid bad run, or save compute.

Cross-architecture generalization:

- Naive universal-coordinate generalization: **probably no**.
- Architecture-specific chart law: **yes, worth pursuing**.
- Accept the current evidence: architectures/tokenizers impose different coordinate charts. The project should stop pretending the same numeric features transfer cleanly.

**5. Kill Or Continue**

Do **not** continue the current trajectory as “one more Qwen3 cleanup.” That path is saturated.

Continue only if the pivot is:

> **Output-Interface Fisher Codebook Diagnostics:** prove that `M_W` predicts and improves training decisions better than loss/telemetry/probes.

Decision tree after g197:

- **g197 PASS:** continue to g199, then natural-run forecast.
- **g197 WEAK_PASS:** publish narrow mechanism, do not claim breakthrough.
- **g197 FAIL:** kill the training-health headline. Keep the output-interface mechanism as the paper.

There is a path to something genuinely novel, but it is narrow: **not Neural Genome universality, not model surgery, not “geometry is everything.”** The real path is a causal, operator-level theory of the LM head as the training codebook. That could matter. Everything else should be cut until that either survives or dies.

