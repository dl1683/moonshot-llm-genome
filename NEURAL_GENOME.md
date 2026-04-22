# The Neural Genome

A 28 KB vector table — covering just the last 7 of 28 transformer layers — recovers **53% of the next-token NLL gap** of a fully-lesioned 600 M-parameter model.

**Important scope — read this before citing the number (`genome_083`–`genome_086`):**

1. The 53% is next-token NLL gap, not functional capability. Generation from atlas-patched models is degenerate: repetitive filler tokens (`" directly directly..."`, `" on on in on change..."`) on every prompt. The atlas restores the unigram frequency prior, not reasoning.
2. The atlas works ONLY on fully-destroyed models (every layer scrambled). Patching a partial lesion — only the last 7 layers lesioned, early/middle intact — recovers **+0.2%** (null). When early layers produce correct context-conditional activations, the teacher-unconditional mean atlas does not match the context-conditional target the lesioned last-7 layers would need to produce. The atlas is fit to unconditional activation means, which only matches when the *entire* stream is mis-shaped.
3. Cross-size transfer shares all the same caveats — it recovers 59% NLL on a fully-wiped Qwen3-1.7B via ridge-projected 0.6B atlas, but generation coherence has the same unigram-only scope.
4. **Three-wall convergence (`genome_085`, `genome_086`): even dense gradient-based interventions hit the same ceiling.** Output-KL distillation (last-7 layers, 200 steps) recovers 66% NLL but 5/5 prompts degenerate. Full-unfreeze layer-wise feature-matching (all 28 layers supervised, MSE on every hidden state + output KL, 200 steps) recovers 65% NLL but 5/5 prompts degenerate (`",,,,,,,,"`, `" the the the the,"`). **NLL-gap recovery is *decoupled* from generation coherence.** Sparse (atlas), moderate (output KL), and maximal (per-layer FM) supervision all plateau at the same coherence wall.

Honest final framing: **a 28 KB per-layer mean-activation table is a distribution-prior restorer, not a capability-surgery primitive — and neither is any other short-horizon intervention we tested.** The three-wall convergence is the real finding: capability is not recoverable from a catastrophically-lesioned model by any sparse or short supervised intervention. Conditioning structure must be re-learned over many gradient steps, effectively retraining from scratch with teacher supervision. This is a publishable *negative* claim about the limits of forward capability transfer, and it echoes (and extends) the 12-op null catalog.

## The claim

Take Qwen3-0.6B. Scramble every matmul weight in all 28 transformer blocks (196 weight tensors total, Frobenius-norm preserved per tensor). The model's language-modeling NLL collapses from 3.67 to 16.90 — a complete capability wipeout.

Now install a single per-layer 1024-dimensional bias vector at the output of each block. Each vector is the teacher's mean mid-layer activation, computed on 300 C4 sentences. The 28 vectors together are 28 × 1024 scalars × 4 bytes = **112 KB** — roughly 1 / 10,000 the size of the 1.2 GB weight file.

With the atlas installed, NLL drops to 10.40. **Fraction of the capability gap closed: 49%.**

| Quantity | NLL |
|---|---:|
| Teacher (pretrained Qwen3-0.6B) | 3.67 |
| Student (all 28 layers lesioned) | 18.10 |
| Student + full 112 KB atlas (28 layers) | 10.28 |
| Student + **28 KB atlas (last 7 layers only)** | **10.49** |
| fraction_gap_closed (last-7) | **+0.527** |
| fraction_gap_closed (all-28) | +0.542 |

**The genome is localized to the last quarter of the network.** Sweeping which layers get patched (`genome_079`):

| Regime | layers patched | fg_closed |
|---|---:|---:|
| none | 0 | +0.003 |
| first-7 (layers 0-6) | 7 | +0.009 |
| **last-7 (layers 21-27)** | **7** | **+0.527** |
| first-14 | 14 | +0.116 |
| last-14 | 14 | +0.445 |
| mid-14 (layers 7-20) | 14 | **-0.012** |
| alternate (every 2nd) | 14 | +0.383 |
| all-28 | 28 | +0.542 |

The middle 14 layers alone contribute essentially zero. The last 7 layers alone carry the entire observed effect. Storage for the landmark result: **28 KB** — 1 / 40,000 of the 1.2 GB weight file.

## The compression curve (`genome_080`)

How small can the atlas get? Sweeping last-N from 1 to 28:

| Last N layers | Atlas size | fg_closed |
|---:|---:|---:|
| 1 (layer 27 only) | 4 KB | **20.4%** |
| 2 | 8 KB | 28.7% |
| 3 | 12 KB | **45.8%** |
| 5 | 20 KB | 49.6% |
| 7 | 28 KB | **53.5%** (peak) |
| 10 | 40 KB | 47.7% |
| 14 | 56 KB | 52.5% |
| 28 | 112 KB | 58.5% |

Thresholds:
- ≥15% recovery at **4 KB** (single last layer)
- ≥30% recovery at **12 KB** (last 3 layers)
- ≥50% recovery at **28 KB** (last 7 layers)

The curve plateaus past last-7. Adding layers 10-20 contributes near-zero or negative return on size.

## Why it works

The candidate-8 spectral bridge (research/derivations/candidate_8_spectral_bridge.md) establishes that trained activation covariances have a specific universal structure — effective rank ≈ 2 × rate-distortion dimension across 7 of 8 tested architectures. The 12-op null catalog (see README landmark finding 3) established that forward geometric manipulation of weights cannot install capability. Rank-48, rank-256, and rank-1024 linear adapters ALL fail at a single-layer patch (see `genome_capability_patch_k48_v2`).

What DOES transfer is the simplest possible adapter: **additive mean shift**. Lesioned weights produce activations with the wrong DC offset at every position in the residual stream. A single bias vector per layer restores the learned mean. The downstream layers, which are themselves lesioned, can still propagate the corrected signal because the mean encodes the teacher's conditioning on the input distribution at that depth.

## Scope

- **Tokenizer + embedding + LM head are NOT lesioned.** Only the 28 transformer blocks' matmul weights are randomized. Because Qwen3's embedding and LM-head are tied, the decode-side mapping (hidden-state → token probabilities) is intact in both teacher and lesioned student. The atlas is therefore carrying "mean residual-stream state per layer" into a decode path that knows how to read it. This is still a real capability-transfer result (every transformer block is destroyed), but the honest framing is "the atlas restores residual-stream conditioning, not the tokenizer/decoder." Extending lesion to embeddings is future work.
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

## Cross-size transfer (`genome_082`)

Can a smaller model's atlas patch a larger model?

Teacher: Qwen3-0.6B (h=1024).
Student: Qwen3-1.7B (h=2048), all 28 layers lesioned.
Method: ridge-regularized pseudoinverse projection fit per layer on ~300 C4 probes of the *pretrained* student to align hidden-space geometries; then project teacher atlas → student space; patch lesioned student.

| | NLL |
|---|---:|
| Qwen3-1.7B teacher (pretrained) | 3.374 |
| Qwen3-1.7B student (all 28 layers lesioned) | 18.673 |
| Student + projected 0.6B atlas | **9.676** |
| fraction_gap_closed | **+0.588** |

**A 0.6B model's 112 KB atlas recovers 59% of a 1.7B model's capability when every transformer layer is destroyed.** Marginally better than same-size transfer. Knowledge projects across model sizes via a simple ridge-fit linear map.

## Three-wall convergence (`genome_085`, `genome_086`)

After the atlas scope correction, the natural follow-up was: would gradient-based supervision break the coherence wall? We tested two densities.

**Wall 2 — output-KL distillation, last-7 layers (`genome_085`).** Student = Qwen3-0.6B, all 28 layers lesioned, cast to fp32. Unfreeze layers 21–27 only (~18.5% of params). KL(student_logits || teacher_logits) on 300 C4 sentences, 200 steps, lr=1e-4, batch=4. Result: **NLL 18.33 → 8.65 (fg_closed 66.2%)** but **5/5 prompts degenerate** (`" of,,,,,,,..."`, `",,,,,,,,,,,,,"`).

**Wall 3 — full layer-wise feature-matching (`genome_086`).** Same lesioned student, but **unfreeze every parameter** (~600M trainable), supervise with MSE on every hidden state + output KL, 200 steps, lr=3e-5. Result: **NLL 17.60 → 8.62 (fg_closed 64.6%)** but **5/5 prompts degenerate** (`",,,,,,,,,,,,,,,,,,,,"`, `" the the the the, the,,,,,,,,,,,,,"`).

| Intervention | params touched | supervision density | steps | fg_closed | coherent? |
|---|---:|---|---:|---:|:-:|
| Atlas (`078`) | 28·1024 scalars | static per-layer means | 0 | 54% | no (5/5 rep) |
| Output-KL distill last-7 (`085`) | ~110 M | output logits only | 200 | 66% | no (5/5 rep) |
| Layerwise FM full-unfreeze (`086`) | ~600 M | every hidden state + logits | 200 | 65% | no (5/5 rep) |

**All three intervention classes hit the same wall.** NLL recovery plateaus at 49–66% and generation stays degenerate regardless of supervision density. Capability is not recoverable from a catastrophically-lesioned model via any sparse or short supervised intervention. The conditioning structure must be relearned over many gradient steps — effectively retraining from scratch with teacher supervision.

This is a publishable *negative* capability-transfer claim and a natural extension of the 12-op null catalog: the null extends from zero-step geometric manipulation into the short-horizon supervised regime.

**Open question:** is the wall a training-budget artifact or fundamental? A ≥5000-step layerwise-FM run would settle it — if coherence eventually emerges, the wall is budget; if not, the wall is structural and points at the re-learning interpretation.

## Open questions

- Is the three-wall ceiling budget-limited (would 5k–50k steps break it?) or structural?
- Cross-family transfer (different trained model, same hidden size)?
- Cross-size + last-N combined (do we still need only the last 7 layers when projecting across sizes)?
- Does the atlas recover task-specific capabilities (arithmetic, code, reasoning) or only general language-modeling?
- Can the atlas be *learned* on a different teacher corpus and still patch?

## Ledger

Atlas line: `genome_078` primary; `genome_074`, `genome_075` generalization. Compression: `genome_080`. Cross-size: `genome_082`. Scope corrections: `genome_083`, `genome_084`. Three-wall convergence: `genome_085`, `genome_086`.
