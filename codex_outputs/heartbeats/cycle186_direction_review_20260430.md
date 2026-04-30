**Architecture Read**
The g195 seed-42 result weakens “input embedding alone” and strengthens “interface coherence.” If this holds, the codebook hypothesis survives only in the stricter form: the learned prior is a matched input-output chart, not just token vectors. Input rows map tokens into hidden-state space; output rows define the logit/classifier basis. If only one side is trained, the model is speaking into one coordinate system and reading out through another.

If g195 returns `PASS_BOTH_NEEDED`, the trained prior is best described as a coordinate system imposed on both encoder and decoder surfaces. That is meaningfully better than “embedding init helps,” because it says the same row-direction chart must govern token ingestion and token prediction.

For g196, yes: the anchor is a quadratic potential well while active, adding a force back toward trained row directions. The crucial distinction is whether that well only regularizes `W` while present, or whether it steers the whole model into a different CE basin before removal. `cutoff_2000` vs `late_anchor_only_2000` is the key discriminator. Scaffold controls are essential: if orthogonal/covariance scaffolds work, the effect is conditioning, not token-specific content.

**Competitive Read**
Recent work does touch nearby ground, but I do not see a direct published analogue of g195/g196.

Relevant anchors:
- [GTI, Apr 2026](https://arxiv.org/abs/2604.02324): token initialization geometry can persist through fine-tuning; mean-init collapse is hard to recover from. Very relevant, but new-token/recommendation setting.
- [In-training probes, Apr 2026](https://arxiv.org/abs/2604.01025): internal representations predict downstream performance better/faster than loss alone. Supports the diagnostic framing.
- [Geometric Canary, Apr 2026](https://arxiv.org/abs/2604.17698): representational stability predicts steerability/drift. Adjacent geometry-as-diagnostic competitor.
- [RMT early-stopped gradient flow, Apr 2026](https://arxiv.org/abs/2604.18450): gives a theory language for early anisotropy, transient spectral phases, and basin windows.
- [Press & Wolf 2017](https://aclanthology.org/E17-2025/) and [Bertolotti & Cazzola 2024](https://proceedings.mlr.press/v235/bertolotti24a.html): input/output embeddings are not interchangeable; tying imposes a real statistical assumption.

**Section 0.1 Impact**
Do not move the score yet from one input-only cell. But if final g195 is `PASS_BOTH_NEEDED` and g196 is `PASS_RESIDUE` with scaffold controls dead, 5.2 -> ~6.5 is justified. The claim becomes: trained lexical interface directions causally steer the optimization basin.

If g196 is `PASS_REGULARIZATION`, cap around ~6.0. Useful, but less moonshot: “active interface regularization works” is weaker than “trained interface chart changes learning.”

If output-only dominates, rename the story to unembedding/logit-prior geometry. Interesting, but it would kill the current input-codebook framing.

