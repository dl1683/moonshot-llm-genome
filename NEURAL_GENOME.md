# The Neural Genome

A 112 KB vector table carries 49% of the capability of a 600 M-parameter model.

## The claim

Take Qwen3-0.6B. Scramble every matmul weight in all 28 transformer blocks (196 weight tensors total, Frobenius-norm preserved per tensor). The model's language-modeling NLL collapses from 3.67 to 16.90 — a complete capability wipeout.

Now install a single per-layer 1024-dimensional bias vector at the output of each block. Each vector is the teacher's mean mid-layer activation, computed on 300 C4 sentences. The 28 vectors together are 28 × 1024 scalars × 4 bytes = **112 KB** — roughly 1 / 10,000 the size of the 1.2 GB weight file.

With the atlas installed, NLL drops to 10.40. **Fraction of the capability gap closed: 49%.**

| Quantity | NLL |
|---|---:|
| Teacher (pretrained Qwen3-0.6B) | 3.67 |
| Student (all 28 layers lesioned) | 16.90 |
| Student + 112 KB atlas (this result) | 10.40 |
| fraction_gap_closed | **+0.491** |

The half-atlas control (patching only the first 14 layers) is null (-1.4% gap closed). The recovery is not layer-additive — all layers contribute jointly. Atlas size is the full 28-layer table or nothing.

## Why it works

The candidate-8 spectral bridge (research/derivations/candidate_8_spectral_bridge.md) establishes that trained activation covariances have a specific universal structure — effective rank ≈ 2 × rate-distortion dimension across 7 of 8 tested architectures. The 12-op null catalog (see README landmark finding 3) established that forward geometric manipulation of weights cannot install capability. Rank-48, rank-256, and rank-1024 linear adapters ALL fail at a single-layer patch (see `genome_capability_patch_k48_v2`).

What DOES transfer is the simplest possible adapter: **additive mean shift**. Lesioned weights produce activations with the wrong DC offset at every position in the residual stream. A single bias vector per layer restores the learned mean. The downstream layers, which are themselves lesioned, can still propagate the corrected signal because the mean encodes the teacher's conditioning on the input distribution at that depth.

## Scope

- Tested on Qwen3-0.6B with C4-clean stimuli. Cross-architecture / cross-size and arbitrary-task benchmarks are future work.
- Recovery is on language-model NLL. Task-specific benchmarks (arithmetic, reasoning, instruction-following) have not been evaluated yet.
- The atlas is per-model (teacher-specific). Whether a teacher atlas transfers to a different-family student is an open question.
- "Lesion" here means Frobenius-matched Gaussian-random weight replacement. Real-world analogs (quantization damage, checkpoint corruption, partial fine-tuning collapse) have not been tested.

## Implications

- **Capability compression.** At least a portion of trained-model capability is captured by O(L × h) scalars where L = depth and h = hidden dim — orders of magnitude smaller than the full weight file.
- **Model surgery primitive.** The simplest possible adapter (1024-dim bias per layer) is a real capability patch. No fine-tuning, no gradient descent, no rank-48 adapter tuning — just transplant the teacher's mean.
- **Direction identity ≠ full weight identity.** The 12-op null catalog and the k_bulk=48 universal plateau width (README finding 4) suggested capability lives in specific direction identity. This result refines that: the *additive direction* of learned activation mean is what propagates through the residual stream. Rotation and per-direction scaling hurt.

## Reproduce

```
python code/genome_full_mean_genome.py
```

Writes `results/gate2/full_mean_genome.json`.

Build-on / vary:
- `code/genome_capability_patch_k48.py` — single-layer rank-48 adapter (control, null)
- `code/genome_capability_patch_k48_v2.py` — single-layer rank sweep + mean-shift variant
- `code/genome_capability_patch_generalize.py` — mean-shift across 3 lesion depths (57-67% each)
- `code/genome_full_mean_genome_sweep.py` — 8-regime coverage sweep (which layers carry the signal)

## Open questions

- Does the atlas transfer across model families within the same hidden size (e.g., Qwen3-0.6B → a different 1024-dim CLM)?
- With a linear projection, does the atlas transfer across sizes (Qwen3-0.6B → Qwen3-1.7B)?
- Does it recover task-specific capabilities (arithmetic, code, reasoning) or only general language-modeling?
- Can the atlas be *learned* on a different teacher corpus and still patch?

## Ledger

Primary result: `experiments/ledger.jsonl` entry `genome_078`. Supporting entries: `genome_074` (single-layer mean-shift surgery) and `genome_075` (3-layer generalization, 61% mean).
