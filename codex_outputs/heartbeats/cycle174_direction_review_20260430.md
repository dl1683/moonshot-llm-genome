Taking your locked g194 numbers as authoritative over the stale local WIKI/JSON: yes, it moves §0.1.

**Q1. Score**
g194 raises §0.1 from **4.5 → 5.2/10**.

Not higher yet. It kills the scalar-norm confound cleanly: `correct_dir_shuffled_norm` recovers ~95% of full match, while correct norms with wrong/random directions are harmful. But the claim is still capped by tied `lm_head`, 8-layer shell, exact-string overlap, and anchor dependence.

**Q2. random_dir worse than shuffled_dir**
The gap says trained directions have **global manifold validity** even when assigned to the wrong token.

`shuffled_dir_correct_norm = -0.662`: wrong lexical assignment, but directions still come from the trained embedding/codebook manifold.

`random_dir_correct_norm = -1.006`: correct norms plus off-manifold directions. Much worse.

So the decomposition is roughly:

- Correct token direction = lexical/content signal.
- Trained-but-wrong direction = valid architecture-conditioned vector, wrong address.
- Random direction = invalid vector, correct scalar shell.

That supports “direction is content,” but also adds: **content lives inside an architecture-compatible direction manifold.**

**Q3. What gets above 6.0**
If g195 and g192 both pass, §0.1 reaches about **6.0/10**, maybe **6.1** if margins are clean.

To get above that, you need **persistence without the tether**. So yes, **g196 anchor-residue is the right call**, but it must answer this exact question:

Does the correct direction change the optimization basin, or does it only help while the anchor is actively forcing the row?

Run: `init_only`, `anchor_only`, `init+anchor`, `cutoff_50`, `cutoff_500`, `cutoff_2000`, `late_anchor_only`, with g194/g195-winning arms. A clean residue pass pushes this to **~6.5**. To reach 7+, follow with prospective interface-compatibility prediction against early-loss baselines.

**Q4. 2026 papers**
No exact g194 competitor found: I did not find a 2026 paper showing token-row unit directions causally preserve training benefit after norm disruption.

Closest validating/competing papers:

- [Semantic Directions Emerge from Sparse Token Mixtures](https://openreview.net/forum?id=nTEl9XtGEx) (ACL ARR 2026): strongest adjacent validation. Semantic directions are decomposed into sparse mixtures of LM-head token embeddings; knockout tests show selected tokens matter.
- [Semantic Structure in Large Language Model Embeddings](https://openreview.net/forum?id=P9BzyDNLDc) (ICLR 2026 submission): direct support that embedding-matrix semantic directions are meaningful and steering-relevant.
- [Differential syntactic and semantic encoding in LLMs](https://arxiv.org/abs/2601.04765): validates linear semantic/syntactic directions in hidden states, less direct for token rows.
- [Linear representations in language models can change dramatically over a conversation](https://arxiv.org/abs/2601.20834): validates semantic directions but warns static direction interpretations can be context-fragile.
- [Fast and Accurate Probing of In-Training LLMs’ Downstream Performances](https://arxiv.org/abs/2604.01025): direct competitor to Genome Forecast, not direction-is-content specifically.
- [The Geometric Canary](https://arxiv.org/abs/2604.17698): competitor on geometry-based steerability/drift diagnostics.
- [The Umwelt Representation Hypothesis](https://arxiv.org/abs/2604.17960): validates your “architecture-specific charts, not universal coordinates” framing.

Net: g194 is differentiated. The field validates “directions carry semantic information”; your claim is sharper: **token-specific trained unit directions are causal interface content for optimization.**

