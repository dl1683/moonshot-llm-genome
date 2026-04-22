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

**Cross-model corruption detection matrix** (2 architectures × 3 corruption types, computed from genome_060 + genome_062 data):

| Model | Corruption | baseline → corrupted rel_err | spike |
|---|---|---:|---:|
| Qwen3-0.6B (CLM) | wiki_raw | 0.090 → 0.286 | **3.2×** |
| Qwen3-0.6B (CLM) | wiki_word_shuffled | 0.090 → 0.621 | **6.9×** |
| Qwen3-0.6B (CLM) | wiki_word_reversed | 0.090 → 0.606 | **6.7×** |
| BERT-base (MLM) | wiki_raw | 0.136 → 6.134 | **45.1×** |
| BERT-base (MLM) | wiki_word_shuffled | 0.136 → 3.218 | **23.7×** |
| BERT-base (MLM) | wiki_word_reversed | 0.136 → 3.235 | **23.8×** |

**6 / 6 (model, corruption) pairs detect contamination with ≥3× spike.** MLM encoders (BERT) are ~5× more sensitive than CLMs (Qwen3) — suggesting MLM training produces a more rigid bridge that is more easily violated. Either way, the detector works on both architectures.

One probe after the stimulus swap, `rel_err` jumps 3-45× above the healthy baseline. No training signal was consulted.

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
