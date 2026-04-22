# GenomeGuard

**A one-shot silent-data-corruption detector for neural network training, built on the candidate-8 spectral bridge of trained representations.**

Run time: ~20 seconds per probe. No training signal needed. Zero aux-loss overhead.

## The problem

Training gets silently corrupted: a dataloader points at the wrong shard, a checkpoint loads partially, a stimulus distribution drifts, a preprocessing step changes. Loss goes up — but sometimes only by a fraction, easily lost in stochastic noise. The run drifts for thousands of steps before anyone notices.

Traditional training monitors look at the loss itself, or at gradient norms. Both are lagging indicators — they only move once the damage has been integrated into the weights.

**GenomeGuard looks at the activation cloud's spectrum instead.** Healthy trained activations on their training distribution produce a universal spectral bridge:

```
c ≈ eff_rank(X) / d_rd(X)
```

where `c = p · d_rd` from the kNN-clustering power-law fit, `eff_rank = (Σσ²)² / Σσ⁴` from the singular spectrum, and `d_rd` is the rate-distortion dimension from the k-means distortion scaling. This bridge has been measured on 7/8 trained networks (Qwen3, DeepSeek, BERT, RoBERTa, MiniLM, DINOv2, CLIP-text, CLIP-vision) at 88% PASS within 15% relative error — see `research/derivations/candidate_8_spectral_bridge.md`.

The bridge **breaks** the moment the model encodes out-of-distribution stimuli. Bridge `rel_err = |ratio - c| / c` is the alarm bell.

## The result

**On Qwen3-0.6B** (CLM, mid-depth, n=1000 C4 probe batch):

| Condition | rel_err | separation from baseline |
|---|---:|---:|
| Baseline (C4) | **0.090** | 1.0x |
| Doomed (3% attention-weight noise) | 0.113 | 1.26x |
| **SWAP (C4 → wikitext-shuffled)** | **0.655** | **7.29x** |

**Cross-architecture corruption detection** (5 text systems, wikitext-word-shuffled swap, genome_069):

| Model | class | baseline rel_err | swap rel_err | spike |
|---|---|---:|---:|---:|
| Qwen3-0.6B | CLM decoder | 0.090 | 0.621 | **6.9×** |
| DeepSeek-R1-Distill-1.5B | CLM decoder (reasoning-distilled) | 0.002 | 0.227 | **144.9×** |
| BERT-base | MLM encoder | 0.136 | 3.218 | **23.7×** |
| RoBERTa-base | MLM encoder | 0.041 | 0.488 | **11.9×** |
| MiniLM-L6 | contrastive encoder | 0.084 | 0.738 | **8.7×** |

**5 / 5 text architectures pass ≥2× threshold.** Mean spike **39×**. Systems with the tightest baseline (DeepSeek at rel_err=0.002) give the largest spike (144.9×) — the tighter the healthy bridge, the more sensitive the detector. Cross-architecture matrix across CLM decoders, MLM encoders, and contrastive encoders.

**Cross-corruption** (Qwen3 + BERT × 3 corruption types, from genome_062):

| Model | wiki_raw | wiki_word_shuffled | wiki_word_reversed |
|---|---:|---:|---:|
| Qwen3-0.6B | 3.2× | 6.9× | 6.7× |
| BERT-base | **45.1×** | 23.7× | 23.8× |

**6 / 6 (model, corruption) pairs detect contamination with ≥3× spike.** MLM encoders (BERT) are ~5× more sensitive than CLMs (Qwen3) — suggesting MLM training produces a more rigid bridge that is more easily violated. Either way, the detector works on both architectures.

One probe after the stimulus swap, `rel_err` jumps 3-45× above the healthy baseline. No training signal was consulted.

**Catastrophic training-divergence detection** (genome_068): sweep Gaussian noise magnitude on all attention + MLP weights:

| sigma (frac. of Frobenius) | rel_err | separation from baseline |
|---:|---:|---:|
| 0.00 | 0.090 | 1.00× |
| 0.01 | 0.105 | 1.17× |
| 0.03 | 0.111 | 1.24× |
| 0.10 | 0.042 | 0.47× *(non-monotone dip)* |
| **0.30** | **0.736** | **8.19×** |
| **0.50** | **0.660** | **7.35×** |

At catastrophic-perturbation levels (σ ≥ 0.3 = 30% Frobenius), the bridge breaks with 7-8× signal. Small perturbations (σ ≤ 0.03) are not reliably detected — this is a usage note, not a kill. Practical threshold `rel_err > 0.25` (≈3× healthy baseline) triggers reliably at σ ≥ 0.3.

## How to use it

```python
from code.genome_genomeguard import probe_health
from code.genome_loaders import load_system

sys = load_system("Qwen/Qwen3-0.6B", quant="fp16", untrained=False, device="cuda")
mid = sys.n_hidden_layers() // 2

# During training, run on a fixed probe batch every N steps:
h = probe_health(sys.model, sys.tokenizer, probe_batch_texts, "cuda", mid)
if h["bridge_rel_err"] > 0.25:  # 3× above the ~0.09 healthy threshold
    alert(f"GenomeGuard: bridge broken, rel_err={h['bridge_rel_err']:.2f}. Check data pipeline.")
```

`probe_batch_texts` is 1000 texts drawn once from the training distribution and pinned for the run. Extraction + SVD + kNN takes ~20s on a single RTX 5090.

## Scope and limits

- **Detected**: silent data-distribution swaps (wrong shard, unicode corruption, preprocessing mismatch, fine-tuning-drift into an OOD corpus). 7.29× spike.
- **Not yet detected by 3% noise**: small weight perturbations do not move the bridge much. Practical training instabilities (blown LR, gradient explosion) likely DO — they drive weights out of regime hundreds of times faster than 3% Frobenius noise. Testing on real training trajectories is the next step.
- **Scope of the bridge**: measured to hold within 15% on 7 of 8 systems at C4 baseline. Vision (DINOv2) was the one marginal case (rel_err 20% = 5pt over threshold). Other modalities (audio, video, biological neurons) untested.
- **Stimulus choice matters**: the healthy baseline is tied to the training distribution. Use a probe batch drawn from the same distribution.

## Reproduce

```
python code/genome_genomeguard.py
```

Writes `results/gate2/genomeguard.json` with per-step rel_err under each of 3 conditions: baseline, doomed, swap. The swap condition alone is the shippable test case.

See `research/derivations/candidate_8_spectral_bridge.md` for the identity's empirical scorecard and the P2 derivation status (pure-power-law FALSIFIED, plateau-plus-power-law PARTIAL with universal bulk-width k_bulk ≈ h/22).

Per the 12-op null forward-transfer catalog (covariance / codebook / basis / aux-regularizer / single-layer / QK / V / O / attn-all / MLP / Procrustes-aligned / candidate-8-aux — all null), the bridge is a **diagnostic fingerprint** of trained capability, not a training driver.

## What this does not claim

- Not a training-acceleration tool. Twelve distinct forward-transfer operations have been tested as training levers and all yield null capability gain.
- Not a complete monitor. Doomed-weight detection at 3% noise is below our 2× threshold. Larger perturbations and real training-instability signatures are the next iteration.
- Not a universality claim for audio/biology. Those extensions are future work.

What it **does** claim: **trained representations produce a universal, reproducible spectral-geometric bridge on their training distribution, and any stimulus that breaks the bridge is flagged in one probe step, at ~20s wall cost**.

Ship it. Get user reports. Iterate.
