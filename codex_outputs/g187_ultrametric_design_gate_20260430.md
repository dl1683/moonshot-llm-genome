**Verdict**
Do **not** approve g187 as a §0.1 training-health predictor as written. Approve it as a **measurement primitive / trajectory diagnostic**.

Reason: one successful Pythia training trajectory per size has no “healthy vs doomed” labels. Monotonic ultrametric convergence would show that embedding geometry changes during training, not that it predicts health beyond `step`, `size`, `current loss`, or token frequency.

The design can become §0.1-relevant only if the prereg includes a predictive gate: ultrametric metrics at early checkpoints must predict future/final loss better than strong baselines.

**Design Fixes**
Primary changes needed:

- Use **angular distance** on L2-normalized rows as primary. `1 - cosine` is not a true metric, so ultrametric claims are mathematically weak if it is primary.
- Do not use strict violation count with zero tolerance. For continuous distances, almost every non-ultrametric triangle has a unique longest side, so the raw violation ratio can be trivial.
- Primary triplet metric should be relative slack:
  `slack = (d_max - d_mid) / max(d_max, 1e-12)`
  then report `mean_slack`, `p90_slack`, `violation_rate_tau_0.01`, `violation_rate_tau_0.05`.
- Cophenetic correlation on 10k tokens is expensive; run triplet slack on 10k, but CCC on a fixed 3k-token subset unless a smoke test proves 10k linkage is under budget.
- “Top 10k frequent tokens” must be frozen before measurement. Best: deterministic Pile-token count cache. If unavailable, call it `tokenizer_rank_top10k`, not “most frequent.”

**Controls**
Required controls:

- `step0` random-init baseline.
- Gaussian random embeddings matched to shape.
- Row-norm matched random embeddings.
- Spectral-matched random embeddings: same singular values, randomized orthogonal basis.
- Token subsets: top 10k, middle 10k, rare 10k.
- Centered and whitened embedding variants to catch anisotropy/norm collapse.
- Token metadata regressions: log frequency, token id/rank, token length, whitespace-prefix, byte/special-token class.
- Baselines for prediction: `log_step`, `log_params`, current validation NLL, recent NLL slope, spectral PR/alpha alone.
- Embed-in/out alignment: rowwise cosine, Procrustes/RSA, norm-frequency correlation.

**Extra Measurements**
At each checkpoint record:

- Validation NLL on a fixed small corpus.
- Row norm mean/std and norm-frequency correlation.
- Spectral alpha, participation ratio, stable rank, top-PC variance.
- Hyperbolicity via sampled quadruples.
- kNN-10 clustering coefficient and TwoNN ID on the same token subset.
- CCC using average linkage and complete linkage.
- Embed-in vs embed-out trajectory distance and rowwise alignment.

**Exact Spec**
Files:

- `research/prereg/genome_187_ultrametric_training_diagnostic_2026-04-30.md`
- `code/genome_187_ultrametric_training_diagnostic.py`
- `results/genome_187_ultrametric_training_diagnostic.json`
- `results/figures/genome_187_ultrametric_training_diagnostic.png`

Models:

- Primary: `EleutherAI/pythia-160m`
- Replication: `EleutherAI/pythia-410m`, `EleutherAI/pythia-1b`

Checkpoints:

```text
step0, step1, step2, step4, step8, step16, step32, step64,
step128, step256, step512, step1000, step2000, step4000,
step8000, step16000, step32000, step64000, step128000, step143000
```

Matrices:

- `embed_in = model.gpt_neox.embed_in.weight`
- `embed_out = model.embed_out.weight`
- verify `tie_word_embeddings == false`

Code structure:

```text
load_token_subset()
load_pythia_embedding(model_id, revision, matrix)
normalize_embeddings()
pairwise_angular_distance()
sample_triplet_slack()
compute_cophenetic_ccc()
compute_controls()
compute_checkpoint_nll()
analyze_monotonicity()
analyze_predictive_value()
save_incremental()
plot_trajectories()
main()
```

Save after every `(model, checkpoint, matrix, subset, control)` cell. No full distance matrices in git.

**Prereg Pass/Fail**
Trajectory PASS, not §0.1 PASS:

- `mean_slack` decreases by >=20% from `step0` to `step143000` for `embed_in` and `embed_out` on Pythia-160m.
- `violation_rate_tau_0.01` decreases by >=20%.
- Spearman rho between `log(step+1)` and `mean_slack` <= -0.75.
- CCC increases by >=0.05 absolute.
- Same direction holds in at least 2 of 3 model sizes.
- Embed-in and embed-out trajectories differ: normalized AUC gap >=0.10 and final slack difference bootstrap CI excludes 0.

§0.1 PASS:

- Using checkpoints `<= step4000`, ultrametric features reduce held-out-size future/final-NLL prediction MSE by >=20% versus `log_step + log_params + current_NLL + NLL_slope`.
- Bootstrap CI for MSE improvement lower bound > 0.
- Shuffled-geometry permutation p <= 0.05.
- Result survives token-frequency and spectral-matched controls.

FAIL if monotonicity exists but predictive value disappears after `step/current loss/frequency` controls. That is the most likely outcome.

**Score**
As proposed: **§0.1 movement +0.1 to +0.3**, Nobel/Turing **1.5/10**.

With the predictive gate above and a real win over loss/step baselines: **§0.1 could move to ~5.0**, Nobel/Turing **2.5/10**. Useful, publishable-adjacent, but not a moonshot unless it becomes a robust early stopping or bad-run detector.

Sources checked: Pythia provides controlled public checkpoints across training ([PMLR](https://proceedings.mlr.press/v202/biderman23a.html), [EleutherAI GitHub](https://github.com/EleutherAI/pythia)); the cited ultrametric paper motivates δ-hyperbolicity/ultrametricity/NJ on embeddings but does not close the text-LLM checkpoint-training gap ([arXiv:2512.20926](https://arxiv.org/abs/2512.20926)).

