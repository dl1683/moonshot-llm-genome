**g189 Design**
Use [g181b](<C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-llm-genome/code/genome_181b_long_horizon.py>) as the harness, but generalize anchor targets by param name, not dummy-model params. This is anchor-only, no step-0 embedding injection.

**Arms**
| Arm | Target | What It Controls |
|---|---|---|
| `scratch_ce` | none | CE baseline |
| `true_trained_anchor` | real Qwen3 trained `embed_tokens/lm_head` | C23 candidate |
| `row_shuffled_anchor` | trained rows globally permuted | exact norm, row norms, spectrum, covariance; destroys token identity |
| `freq_bucket_shuffle_anchor` | rows shuffled within C4 token-frequency quantile buckets | preserves frequency/codebook scale; destroys exact content |
| `spectrum_preserving_random` | random orthogonal matrix with trained singular values | preserves SVD spectrum/Frobenius only |
| `same_frobenius_gaussian` | iid Gaussian rescaled to trained Frobenius norm | tests pure scale/norm |
| `anchor_to_initial` | independent random-init Qwen3 embedding, rescaled | tests generic anchoring to plausible init geometry |

All non-scratch targets must satisfy `||T_arm||_F == ||T_true||_F` within tolerance. Row-shuffle and bucket-shuffle preserve this automatically; spectrum control preserves it via singular values; Gaussian/random-init are explicitly rescaled.

**Lambda / Gradient Matching**
For each recipient seed `s` and anchored arm `a`:

```text
d_true_s = ||E_init_s - T_true||_F
G_ref_s = 2 * lambda_true_s * d_true_s   # same g181b trained-anchor force, about 28.46

d_a_s = ||E_init_s - T_a||_F
lambda_a_s = G_ref_s / (2 * d_a_s)
```

So every anchored arm has identical initial anchor-gradient L2 per seed. Log target Frobenius, init-to-target distance, lambda, and measured `2 * lambda * distance`.

**Criteria**
Primary final metric: C4 validation NLL at step 5000. Lower is better.

Define `gap_c = NLL_control - NLL_true`.

PASS_CONTENT if, for every other arm including scratch:

- mean `gap_c >= +0.20` nats
- true beats that control in at least `5/6` paired seeds
- all target-norm and anchor-gradient diagnostics pass

REPRO_FAIL if true trained does not reproduce a meaningful scratch gap.  
FORMAT_FAIL if any control is within `0.20` nats of true or beats true in `>=2/6` seeds.

Interpretation:
- row-shuffle close to true = format, not content.
- freq-bucket close = frequency/codebook prior, not exact lexical content.
- spectrum close = global spectral format sufficient.
- Gaussian close = norm/regularization artifact.
- random-init close = generic anchor prior, not trained content.

**Wall-Clock**
g181b took `7957s` for 6 cells. g189 is `7 arms x 6 seeds = 42 cells`, so estimate:

- training: ~14.8h
- target preprocessing/SVD/cache: ~0.5-1h
- total: ~15.5-17h

Not compliant as one run. Run as resumable seed blocks: `7 cells/seed ~= 2.6-3.0h`, six invocations.

**Implementation Plan**
Create:

- `research/prereg/genome_189_c23_content_causality_2026-04-30.md`
- `code/genome_189_c23_content_causality.py`
- `results/genome_189_c23_content_causality.json`

Use `SEEDS = [42, 7, 13, 101, 202, 303]`, Qwen/Qwen3-0.6B, same tokenizer, same C4 train/val windows, same batch/seq/lr/eval cadence as g181b.

Key implementation requirements:

- bind anchor targets inside the live `train_cell()` model by parameter name
- use manual Frobenius-sum gradient injection: `grad.add_(param - target, alpha=2*lambda)`
- no `F.mse_loss(..., reduction="mean")`
- save after every cell
- add `--seeds`, `--arms`, `--max-cells`, `--prepare-targets-only`, `--smoke`
- update `WIKI.md`, `EXPERIMENTS.md`, and ledger only after run/verdict

**COMPUTE Compliance**
- VRAM: one Qwen3-0.6B training model plus one target; expected `<10-12GB`, under 22GB.
- RAM: CPU target construction one matrix at a time; expected `<20GB`, under 56GB.
- Wall-clock: full sweep exceeds 4h, so must be chunked by seed with resume.
- Disk: target cache ~4GB float32 worst case, acceptable and gitignored.
- Quantization: Qwen3-0.6B full precision/BF16 per `<1B` rule.
- Save/resume: per-cell JSON checkpoint; smoke test before full launch.

Score: 6.5/10. It is not a new moonshot mover by itself, but it is the necessary SEV-10 integrity gate for C23.

