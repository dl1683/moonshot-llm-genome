**Cycle 108 Architecture-Theorist + Competitive Review (Claude-generated, Codex sessions timed out)**

## Architecture Theorist

**Route coherence assessment:**
- Route 1 (Fisher/NTK): Sound framework but computationally heavy for >100M params. Currently skeletal — no concrete predictions implemented. Lowest priority.
- Route 2 (Rate-Distortion Water-Filling): Most novel. Feature-to-rate mapping is formally specified (cycle 106). The Shannon water-filling connection is plausible: spectral alpha ↔ noise-to-signal ratio, PR ↔ capacity spread, depth drift ↔ coding efficiency across layers. Key gap: the mapping is motivated but not derived from first principles. We're fitting a story, not deriving a law. Would need to show SGD's gradient updates MUST allocate capacity as water-filling.
- Route 3 (Stat-Physics Symmetry Breaking): Most testable. Basin metaphor is intuitive (random/collapsed/trained basins). Verdict Matrix is pre-locked. Main weakness: "basins" is metaphorical — no Landau free energy F(φ) has been derived.

**Route 2 novelty claim: "No prior paper directly applies Shannon water-filling to neural representation learning"**
This is PROBABLY true in the narrow sense. Water-filling is used in: MIMO communications, rate-distortion coding, signal processing. Neural network connections: Saxe et al. 2019 (information-theoretic analysis of deep linear networks), Shwartz-Ziv & Tishby 2017 (information bottleneck), recent spectral analysis papers (From SGD to Spectra ICML 2025). But the specific mapping of manifold features → water-filling quantities → training outcome prediction is novel. The risk: "novel because it's a stretch" vs "novel because nobody thought of it." Currently closer to the former — needs a tighter derivation.

**D1/D2 discriminability:**
D1 (continuous vs basin): Route 2 predicts smooth continuous relationship; Route 3 predicts step-function (basins). With n=24, both could look similar — a smooth function sampled at 24 points looks like clusters if the distribution has gaps. D1 is a WEAK discriminator.
D2 (depth drift value): If depth drifts add >5% MSE improvement, Route 2 is favored (layer-wise coding efficiency matters). If not, Route 3 is favored (only global order parameters matter). This is BETTER as a discriminator because the predictions genuinely diverge.

**Missing Route 4 candidate: Information Geometry (Fisher-Rao)**
The 8 features could be coordinates on a statistical manifold (Fisher-Rao metric). Training trajectories are geodesics on this manifold. Training health = how close the trajectory stays to the minimum-description-length geodesic. This connects to: natural gradient (Amari), neural tangent kernel, and the manifold hypothesis. Not worth implementing now — Route 2/3 discriminators come first.

## Competitive Analyst

**State-of-the-art comparison:**
- Loss-of-plasticity (Lyle et al. 2024, Dohare et al. 2024): Focus on LATE training degradation, not EARLY prediction. Our work predicts at 3% of training.
- Training instability detection (Gilmer et al.): Detects divergence/instability, not differential health within healthy-looking runs.
- In-training probes (2604.01025, OLMo3-7B): AUROC>0.75 for bad-run detection, but at mid-training, not at 3%. Closest competitor to our claim.
- Geometric Canary / Shesha (2604.17698): POST-training steerability, not early-training prediction. Different application but overlapping features.
- CKA / linear probing: Comparison metrics, not predictive diagnostics.

**Gap from 4.5/10 to 9/10:**
- 4.5 → 6.0: g182 C' PASS on both LOAO folds (the experiment itself succeeding)
- 6.0 → 7.5: P3/P4 PASS (cross-architecture feature alignment confirms mechanism)
- 7.5 → 8.5: g184 frozen-C' PASS on Falcon-H1 (3rd architecture family, no refit)
- 8.5 → 9.0: First-principles derivation of WHY these features predict (Route 2 or 3 formalized)
- 9.0+: Demonstrated compute savings on a real training run (kill bad runs early → X% compute saved)

**Survival probability given adversarial review:**
SEV-10 (arm masquerade): Mitigated by arm-identity diagnostics (A16) already in code. If arm_decodability > chance+10% AND within-arm R² < 0.05, the claim dies. If arm_decodability is low OR within-arm R² is high, the claim survives.
SEV-9 (overfit): Mitigated by Ridge regularization + LOAO (genuinely held-out architecture). 8 features / 24 cells is borderline but Ridge with α∈[0.01, 1000] handles it.
Overall survival probability: ~40-50% for C' PASS. The experiment is correctly designed to be hard — that's the point.

## POWER MOVE

**The single highest-leverage action: ensure g182 runs to completion and honestly report.**

Everything else is secondary. The Verdict Matrix is locked. The code is correct. The theory backbone exists. What's missing is DATA. Get g182 stage 1 done. If C' PASS → immediately fire g184 Falcon-H1 → that's the 8.5+ finding.

No theory deepening, no paper drafting, no additional experiments, no refactoring will move the needle as much as getting the 48 cells trained and analyzed. Focus all compute and attention on unblocking the teacher text generation hang and running the cells.
