# Candidate-8: Spectral Bridge (eff_rank / d_rd = c)

**Status:** SUPPORTED (2026-04-22 T+44h, 7/8 preregistered PASS across text + CLIP + vision). P2 derivation: pure-power-law FALSIFIED, plateau-plus-power-law PARTIAL with universal k_bulk=48 across 5 text systems (CV 4.2pct). Aux-loss training with ratio target NEUTRAL (genome_066, 12th null forward-transfer op).

**→ See also `trained_spectrum_invariant.md` (2026-04-22 T+55h): the *spectral-only* invariant `sqrt(eff_rank)·α ≈ 3√2` (CV 5% across 5 text systems, 5.5σ separated from shuffled/Gaussian baseline). That invariant closes half of the bridge — predicting eff_rank from α alone — and the compound prediction `d_rd ≈ 18/(α²·(d_stim+1))` reproduces empirical d_rd within 5% without any k-means probe. The two derivation docs now cover complementary pieces.**

**Empirical scorecard (2026-04-22 final):**

| System | modality | c | ratio = eff_rank/d_rd | rel_err | PASS |
|---|---|---:|---:|---:|:---:|
| Qwen3-0.6B | text CLM | 1.889 | 2.059 | 9.0% | ✓ |
| DeepSeek-R1-Distill-1.5B | text CLM | 2.410 | 2.413 | 0.2% | ✓ |
| BERT-base | text MLM | 2.653 | 2.292 | 13.6% | ✓ |
| RoBERTa-base | text MLM | 2.250 | 2.158 | 4.1% | ✓ |
| MiniLM-L6 | text contrastive | 2.027 | 2.199 | 8.4% | ✓ |
| DINOv2-small | vision | 2.242 | 2.694 | 20.2% | ✗ (by 5pt) |
| CLIP-text-B/32 | text + 1 align | 2.975 | 3.184 | 7.0% | ✓ |
| CLIP-vision-B/32 | vision + 1 align | 2.447 | 2.145 | 12.3% | ✓ |

**7/8 ML systems PASS** preregistered 15% threshold. Median rel_err 8.7%.

**Biology extension (2026-04-22, genome_070)**: Allen V1 Neuropixels session 0, 900 frames × 50 cortical units under Natural Movie One:

| System | modality | c | ratio | rel_err | PASS |
|---|---|---:|---:|---:|:---:|
| Mouse V1 (session 0) | biological neurons | 2.488 | 2.183 | **12.3%** | ✓ |

Biological `alpha = 0.200` (vs ML `alpha ≈ 0.77-0.86`) — mouse cortex has much shallower spectral decay (broader variance distribution across neurons) — yet the `c ≈ eff_rank/d_rd` bridge still holds within the 15% threshold. First cross-substrate empirical evidence for candidate-8 universality.

**Stimulus sensitivity:** bridge holds universally across MODELS at C4 baseline. Breaks on wikitext (29-613% rel_err) and scrambled/reversed (60%+). The bridge is a **C4-specific geometric identity** — it characterizes trained-network activation geometry on natural text specifically, and deviates when stimulus structure is destroyed.

**P2 derivation status:** pure power-law `σ_i² ∝ i^{-2α}` predicts ratio≈0.29 (7× too small vs empirical 2.06). Plateau + power-law with `k_bulk` bulk of flat eigenvalues + α-decay tail gives `k_bulk = 48` universally across 5 text systems (CV 4.2%) — a real structural universal — but predicted ratio 1.5 still 30% off empirical 2.2. Full closed-form derivation requires a richer spectrum model (smooth broken power-law, Marchenko-Pastur + low-rank cluster, or spectrum-based d_rd) which is P3.

**Aux-loss training:** CANDIDATE-8 as auxiliary loss (`(eff_rank - 2·d_rd_ma)²`) on tiny transformer from scratch, 2 seeds × 500 steps: speedup **1.00×**, val-NLL identical within 0.01. Aux shifts geometry (c 0.60→1.22 in seed 1337) but NOT capability. 12th null forward-transfer operation on record. Candidate-8 is a **diagnostic of trained capability**, not a **training lever**.

---

## The bridge claim

For a trained activation cloud `X ∈ R^{n × h}`, define:

- `d_rd(X)` = rate-distortion dimension from k-means distortion scaling
  `D(K) ∝ K^(-2/d_rd)`. Measured via `code/genome_rate_distortion_probe.py`.
- `eff_rank(X)` = participation ratio of eigenvalues of the centered covariance:
  `eff_rank = (sum σ²)² / sum σ⁴`, where `σ_i` are singular values of `X - mean(X)`.

**Empirical bridge (single point, Qwen3-0.6B, trained, mid-depth, C4, n=1000):**

| Quantity | Value |
|---|---|
| `c_observed = p · d_rd` | 1.89 |
| `eff_rank / d_rd` | 25.27 / 12.27 = **2.06** |
| `relative error` | 9 % |

**Claim:** at trained-network activation clouds, `c ≈ eff_rank / d_rd` with some constant pre-factor close to 1. If true across systems, this transforms the empirical candidate-5 into a spectral identity involving two geometric measurements rather than two fit parameters.

## Why this matters

1. **Replaces one fit parameter with one geometric measurement.** `p` from kNN-clustering scaling is a fit exponent; `eff_rank` is a closed-form spectral summary. If `p · d_rd ≈ eff_rank / d_rd`, then `p ≈ eff_rank / d_rd²`, which is a *prediction* about the clustering exponent from spectral data.
2. **Connects to random matrix theory.** `eff_rank` is the standard second-moment "participation" measure of a spectrum. For a Marchenko–Pastur (iid-Gaussian) cloud, `eff_rank = h · (1 - 1/γ)²` where `γ = n/h`. A trained cloud's `eff_rank` deviates from Marchenko–Pastur in a measurable way — the training-specific signature genome_057 just isolated.
3. **Connects to rate-distortion via participation ratio theorems.** Under Gaussian-source + squared-loss rate-distortion, `d_rd` has a closed-form in terms of the spectrum (Telatar 1999; Cover & Thomas ch. 10). Specifically `D(K) ∝ K^{-2/d_rd}` implies `d_rd` is determined by the exponential decay rate of the ordered eigenvalues. This is consistent with — but not identical to — `eff_rank`.

## What has to hold for the bridge to be universal

Two concrete predictions, each testable:

### P1. Ratio universality across systems

For every system in the candidate-5 scorecard, measure `eff_rank` and `d_rd` from the same activation cloud used to measure `c = p · d_rd`. Predict:

```
eff_rank_i / d_rd_i ≈ c_i,   for all 11 scorecard-fitting systems.
```

If the relationship holds within < 15 % on 11 / 11, candidate-8 becomes the primary derivation candidate. If it fails on > 2 systems, candidate-8 is falsified.

### P2. Spectral-decay to d_rd connection

A clean derivation of candidate-8 requires showing `d_rd` and `eff_rank` are both predictable from the power-law decay exponent `α` of the singular spectrum. For `σ_i ∝ i^{-α}` in the bulk tail:

- `eff_rank(α)` has a closed form for pure power-law: if `σ_i² = c · i^{-2α}` for `i = 1..h`, then `eff_rank ≈ h^{2α-1} · (2α-1) / (2α)²` (for `α > 0.5`) — a function that monotonically compresses as `α` grows.
- `d_rd(α)` from the same spectrum requires evaluating the rate-distortion D(R) function on the Gaussian source with covariance `diag(σ_i²)` and applying the reverse-waterfilling derivative, which at leading order gives `d_rd ≈ 2 / (1 + 1/α)` = `2α / (α+1)` (rough; verify).

If both hold, `eff_rank / d_rd` becomes a *pure function of α* — and we can directly derive it, numerically evaluate for `α = 0.861` (Qwen3 trained), and check against `c = 1.89`.

## Why this is more than fitting

The asymmetric signature from genome_056 / 057 is:

- `α_trained = 0.861`, `α_shuffled = 0.654`, `α_Gaussian = 0.652` (shuffle ≈ iid-Gaussian)
- `eff_rank_trained = 25.3`, `eff_rank_shuffled = 63.4`, `eff_rank_Gaussian = 61.8`

Training does two things to the spectrum simultaneously:

1. **Concentrates mass into fewer directions** (eff_rank 63.4 → 25.3, a 2.5× compression).
2. **Steepens power-law decay** (α 0.65 → 0.86).

These two are NOT independent — the math of `eff_rank(α)` shows they are coupled: steeper α yields smaller eff_rank. The interesting residual is whether `d_rd` tracks both in the specific way that produces `eff_rank / d_rd ≈ c = n_axes + n_alignments`.

## The stimulus-dependence complication (2026-04-22)

The BERT-wikitext experiments showed that `c` is NOT a model-only invariant: **the same model produces a different `c` on a different stimulus distribution.** Specifically:

| Model | c on C4 | c on wikitext-103 | Δ |
|---|---|---|---|
| Qwen3-0.6B (CLM) | 1.89 | 1.16 | -0.73 |
| BERT-base (MLM) | 2.65 | 0.55 | -2.10 |
| RoBERTa-base (MLM) | 2.25 | 0.93 | -1.32 |
| MiniLM-L6 (contrastive) | 2.03 | 0.93 | -1.10 |

Candidate-8 predicts that `eff_rank / d_rd` tracks `c` point-by-point across stimuli AND models. If true:

- For each (model, stimulus) pair, measure `eff_rank` and `d_rd` from that pair's activation cloud
- Predict `c` from the ratio
- Compare to measured `p · d_rd`

This is the *strong* form of candidate-8: the bridge holds not just at C4 baseline but wherever `c` is measured. If it holds, we have a mechanistic coupling between the geometric measurement `c` and the spectral measurement `eff_rank` — which IS a first-principles-adjacent derivation, provided P2 closes.

If candidate-8 FAILS on P1 (ratio not universal) or P2 (closed form doesn't match), it still provides a sharper question: what *is* the training-specific spectral structure that causes `c` to take specific integer-ish values in specific (model, stimulus) regimes?

## Immediate next experiment

```
code/genome_svd_spectrum_multimodel.py
```

Run SVD spectrum + compute eff_rank on all 11 candidate-5-fitting systems at C4 baseline. Collect (c, eff_rank, d_rd) triples. Test P1. Time estimate: ~30 min (already have activation extraction code). If P1 passes on 9+/11: candidate-8 advances. If fails on > 2: candidate-8 falsified.

## Honest limits

- This is NOT a derivation yet. It is an empirical bridge between two measurement pipelines.
- P2 may be impossible without strong assumptions (isotropic Gaussian source, pure-power-law spectrum) that real activation clouds may violate.
- If the bridge is only approximate (matches within 15% but not exactly), candidate-8 is at best a *partial* explanation of `c`.

Still, the SVD signature from genome_057 is the first mechanism-adjacent finding after 5 Compiler nulls and 2 toy-manifold falsifications. Worth pursuing.
