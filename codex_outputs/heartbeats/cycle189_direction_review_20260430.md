**Verdict**
g195 is important, but do not overclaim it yet: the repo artifact is still `INCOMPLETE` at 7/15 cells. Also, your “output starts behind scratch” trajectory does not match the checked JSON. In [results/genome_195_untied_input_output_factorial.json](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/results/genome_195_untied_input_output_factorial.json:147>), output seed 42 is already ahead at step 500: scratch `6.5777` vs output `6.2599`, and final gain is `+0.3635`. If you have fresher monitor logs, use them; the committed artifact says “immediate output advantage,” not late crossover.

**Architecture Theorist**
Yes: this is a prediction-coordinate-system effect, more specifically a **trained class-prototype / dual-basis effect**.

Why output can dominate:

- The `lm_head` is the direct metric of the loss. Every next-token CE gradient asks hidden states to move toward the target token row and away from competitor rows.
- A trained `lm_head` gives the model useful target directions from step 0. The network can learn to produce hidden states in that basis.
- A trained input embedding helps only through the forward context path. It is mediated by attention/MLP layers and can be overwritten/relearned.
- In untied mode, “good input, random output” means the model may encode tokens well but receives a bad readout geometry. “Random input, good output” means the model has to learn input-side representations, but the target basis is already semantically organized.

This connects cleanly to three frameworks:

1. **Output embedding / weight tying theory.** Press & Wolf already found the tied matrix evolves more like the untied output embedding than the untied input embedding.
2. **Logit lens / tuned lens.** These methods treat the unembedding as the decoder basis for hidden states; tuned lens exists because intermediate states may need basis translation before the unembedding can read them.
3. **Neural collapse / classifier-prototype geometry.** In classifiers, last-layer features and classifier weights tend toward self-dual class geometry. For LMs, token rows are huge-class classifier prototypes.

So the “reading instrument matters more” analogy is basically right, but stronger: the output matrix is not just a reader; it is also the **gradient generator** that teaches the whole network what directions token predictions should occupy.

**g196**
For exact `--surface output`, the current implementation looks conceptually correct.

- g195 `PASS_OUTPUT` / `PASS_OUTPUT_DOMINANT` maps to `surface="output"` in [code/genome_196_anchor_residue_factorial.py](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_196_anchor_residue_factorial.py:467>).
- `surface == "output"` initializes only `lm_head` in [code/genome_196_anchor_residue_factorial.py](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_196_anchor_residue_factorial.py:561>).
- Anchor gradients are applied to `lm_head.weight` when surface is output/both in [code/genome_196_anchor_residue_factorial.py](</C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_196_anchor_residue_factorial.py:237>).

One small guard I would add before launch: make argparse enforce `choices=["input","output","both","tied"]`. A typo currently could silently create a bad branch. Also reconcile the doc mismatch: WIKI says g196 prereg is locked, but the prereg file still says `DRAFT`.

**Competitive Analyst**
Direct prior exists, but not exactly your causal decomposition.

- [Press & Wolf 2017](https://aclanthology.org/E17-2025/) strongly predicts the direction: tied embeddings behave more like output embeddings than input embeddings.
- [Inan et al. 2017](https://openreview.net/forum?id=r1aPbsFle) gives the loss-framework argument for tying word vectors and word classifiers.
- Most direct current threat: [“Weight Tying Biases Token Embeddings Towards the Output Space”](https://arxiv.org/abs/2603.26663), submitted March 27, 2026. It says tied embeddings align more with unembeddings because output gradients dominate early.
- April 2026 adjacent work: [“Predicting Where Steering Vectors Succeed”](https://arxiv.org/abs/2604.15557) uses unembedding/logit-lens accessibility to predict steering success; [“To See the Unseen”](https://arxiv.org/abs/2604.21632) shows unembedding collapse of unseen tokens can cause symbolic generalization failure.
- [“Disentangling Geometry, Performance, and Training”](https://arxiv.org/abs/2602.20433) is a useful caution: unembedding geometry can reflect training choices more than capability.

So novelty is not “output embeddings matter.” Novelty is: **trained token-row directions transplanted into the output classifier causally recover most of the tied-interface gain in a controlled recipient-training experiment.**

**Section 0.1 Score**
Current WIKI score is `5.2/10` post-g194, despite CLAUDE’s older baseline. If g195 finishes as clean `PASS_OUTPUT_DOMINANT`, I’d move to **5.8/10**.

Not 6.5 yet. It is still 8-layer, within-family, and competitively adjacent to March 2026 weight-tying work. Move to **~6.1** if g192 full-depth also passes. Move to **~6.5** only if g196 on `surface=output` shows persistent residue after cutoff and scaffold controls stay dead.

