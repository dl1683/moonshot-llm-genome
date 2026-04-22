# Preprint draft — working notes

Not yet public-ready. Living draft of the narrative, figures, and section outline as the findings land. Reorganize once all experiments stabilize.

## Working title

**"The Spectral Fingerprint of Trained Neural Networks: eff_rank·α² ≈ 18 and Capability as Mode Diversity"**

Alternate: "A training-specific geometric invariant predicts capability in language and vision models."

## Abstract (draft, <300 words)

Across 8 trained machine-learning systems spanning text causal, masked, and contrastive language models, self-supervised vision encoders, and cross-modal encoders, we observe a tight empirical invariant on the singular spectrum of mid-depth activations. Two independent functionals — the participation ratio `eff_rank = (Σσ²)²/Σσ⁴` and the power-law tail slope `α` of the singular spectrum — satisfy `sqrt(eff_rank)·α ≈ 3√2` with coefficient of variation 5.1% (N=5 fresh-extraction validation). The same functional on iid-Gaussian and marginally-shuffled surrogates gives 5.47 (CV 17%, a 5.5σ separation), indicating the invariant is a training-specific spectral attractor rather than a random-matrix baseline. Biological cortex (mouse V1) sits on a different attractor value (0.95), so the invariant distinguishes trained-ML from non-trained-gradient-descent systems specifically.

The invariant is mechanistically capability-coupled. During gradient-based recovery from catastrophic lesion on Qwen3-0.6B (all 28 transformer blocks scrambled), layer-wise feature-matching training produces a U-shaped trajectory of the invariant: training first collapses the activation cloud to a degenerate 5-rank subspace (the "coherence wall" of repetitive generation), then re-expands toward the trained attractor as coherent English emerges. Coherence recovery coincides with mode-diversity recovery — the invariant is not a correlate of capability but a measurement of it.

Combined with a second empirical finding that `c = p·d_rd ≈ d_stim + 1` (a modality-stratified training invariant), the spectrum functional predicts rate-distortion dimension as `d_rd ≈ 18/(α²·(d_stim+1))` with no clustering probe, matching empirical within 5%. This closes the candidate-8 spectral bridge (`c ≈ eff_rank/d_rd`) by reducing all three measured geometric quantities to two spectral measurements plus modality dimension.

Implications: (1) geometry-as-auxiliary-loss becomes a concrete training-efficiency primitive; (2) sparse capability-transfer interventions fail because they compress rather than re-expand mode diversity (a 15-op null catalog); (3) a 28 KB per-layer mean-activation atlas restores unigram prior but not coherence, consistent with the mode-diversity picture.

## Section outline

1. **Introduction** — why training-specific invariants matter; Intelligence = Geometry thesis.
2. **Candidate-8 bridge** — `c ≈ eff_rank/d_rd` empirically holds across text/vision/aligned at 15% threshold.
3. **The new invariant** — `sqrt(eff_rank)·α ≈ 3√2` across 5/5 text systems (CV 5%), trained-specific (5.5σ from shuffled/Gaussian).
4. **Compound prediction** — `d_rd = 18/(α²·(d_stim+1))` matches empirical within 5%.
5. **Mechanism: capability as mode diversity** — U-shaped trajectory of eff_rank and invariant during lesion recovery; coherence emerges at mode-diversity recovery.
6. **Phase transition** — capability recovery from catastrophic lesion requires ~1500 dense-supervision steps and shows phase behavior.
7. **The null catalog** — 15 forward-transfer operations all fail; the atlas restores unigram prior but not mode diversity.
8. **GenomeGuard** — training-health monitor from the bridge.
9. **Aux-loss efficiency** — (pending genome_090) adding the invariant as an aux loss either accelerates or fails to accelerate the phase transition. Either outcome publishable.
10. **Toward derivation** — shifted power-law `σ² ∝ (i+k_head)^(-2α)` with `k_head ≈ 5` reproduces the invariant at empirical α values; derives to `sqrt(er)·α = 3√2` in closed form (pending numerical validation on empirical spectra via genome_091).
11. **Discussion** — why trained networks converge to this specific attractor; the role of gradient descent vs architecture vs objective; biology comparison.

## Key figures (as of 2026-04-22)

- Fig 1 — cross-system bridge scorecard (8 systems, 7/8 PASS at 15% threshold).
- Fig 2 — invariant histogram: trained vs shuffled vs Gaussian (5.5σ separation).
- Fig 3 — invariant trajectory U-shape during lesion recovery (genome_089). [generated: `results/figures/genome_089_invariant_trajectory.png`]
- Fig 4 — GenomeGuard C4→shuffled detection matrix (5 systems × 3 corruption modes).
- Fig 5 — aux-loss A/B time-to-coherence (pending genome_090).
- Fig 6 — spectrum shape fit: empirical vs shifted-power-law (pending genome_091).

## Unresolved before posting

1. **Derive constant 18.** Shifted power-law candidate works numerically. Need closed-form.
2. **Extend invariant N.** Need vision systems (DINOv2, CLIP-vision) + random-init twins + another family (Gemma / Phi / RedPajama) to reach N ≥ 10 at CV ≤ 7%.
3. **Aux-loss outcome.** genome_090 A/B test — pending.
4. **Broader lesion recovery characterization.** One seed (genome_089). Want N ≥ 3 seeds for the U-shape.

## Risks / weak points

- N=8 (retrospective) and N=5 (validated) on the invariant — small for any "universal" claim. Must push to N≥12 minimum.
- "3√2 to 0.6%" is eyecatching but could be coincidence given N=5 CV 5%. Derivation is what turns it into a law.
- The 1500-step phase transition is budget-dependent; unclear how it scales with lesion severity, model size, or objective.
- Sample-size dependence of the invariant value (3√2 at n=800, but 2.88 at n=200) needs characterization.
- Biology distinction (biology at 0.95) could be an α artifact — mouse V1 α=0.20 is very shallow; at matched α, might biology satisfy the invariant?
