# Pre-registration: genome_183 Corpus-Derived Interface Prior (Causal Interface-Prior Synthesis)

**STATUS:** DRAFT pre-staged 2026-04-29 cycle 80. **LOCKS** after g182 Codex design gate (gates on g180b/g182 progress). Per Codex g181b advisor pick (#1 at +3.5-4.0 uplift).

- Date: 2026-04-29
- Trigger: g181b advisor — "causal interface-prior synthesis" beats g182 continuation
  because it asks a sharper question: can you CONSTRUCT the useful tokenizer/embed
  geometry from corpus statistics alone?

---

## Hypothesis

**H1 (corpus-derived init recovers donor-anchor benefit):** A vocabulary initialization
constructed from corpus co-occurrence / frequency / spectral statistics — with NO trained
model — recovers >= 50% of the g181b +0.513 nat gap vs scratch at 5000 steps, and beats
all control arms.

**H0 (trained-model structure required):** Corpus-derived initializations produce
negligible or noisy benefit; the +0.513 gap requires structure learned during actual
pre-training on a language modeling objective.

## Universality level claim

Level-0 single-architecture mechanism finding. Upgrades to Level-1 if the recipe
generalizes to >= 3 architecture families (cross-arch transfer of the recipe, not of
the donor model).

## Motivation

g181a showed the +1 nat anchor effect is ~100% embed_tokens + lm_head trained-init
(transformer block weights HARM when anchored). g181b showed this persists at 5000
steps (+0.513 nats). The natural question: is the active ingredient specific learned
token representations from pre-training, or can we derive equivalent structure from
corpus statistics? If the latter, the headline shifts from "transfer from trained model"
to "training-design law: cheap corpus-derived interface priors reduce wasted training."

## Arms (5 treatment + 3 control)

All arms: Qwen3-arch random-init recipient, 768-dim, 8 layers. C4 training, 5000 steps.
3 seeds [42, 7, 13] each. Total: 24 cells.

### Treatment arms (corpus-derived):

1. **frequency_init**: Initialize embed weights proportional to unigram log-frequency
   in C4 train corpus. Normalize to unit norm per row. lm_head tied.
2. **cooccurrence_svd**: Build token co-occurrence matrix from C4 (window=5), run
   truncated SVD to 768 dims. Initialize embed from SVD factors. lm_head tied.
3. **covariance_matched**: Match the row covariance matrix of Qwen3 trained embed
   (per g181a snapshot) using corpus-derived PCA directions. No per-token trained info.
4. **spectral_matched**: Match the singular value spectrum (NOT the singular vectors) of
   the trained Qwen3 embed. Fill singular vectors with random orthonormal basis. Tests
   whether spectrum alone carries the benefit.
5. **shuffled_token_anchor**: Qwen3 trained embed + lm_head but with token-to-row
   mapping randomly permuted. Continuous anchor at lambda=0.01. Tests whether token
   identity matters vs just having "good" weight magnitudes.

### Control arms:

6. **scratch_ce**: Random init, CE only. (Baseline.)
7. **trained_anchor**: Qwen3 embed + lm_head anchor at lambda=0.01 (reproduces g181b).
   Reference ceiling.
8. **random_structured**: Random orthonormal embed scaled to match Qwen3 embed Frobenius
   norm. Tests whether scale matching alone helps.

## Pass / fail criteria

| # | Criterion | Threshold |
|---|---|---|
| P1 | At least 1 corpus-derived arm recovers >= 50% of trained_anchor gap | mean(arm - scratch) >= 0.257 nats (50% of +0.513) |
| P2 | Best corpus-derived arm beats ALL 3 controls (scratch, random_structured, shuffled_token) | paired t-test p < 0.05, 3-seed |
| P3 | Trained_anchor still > best corpus-derived (confirming trained info adds value) | OR if corpus-derived >= trained, that's even more interesting |

**PASS:** P1 AND P2 both satisfied.
**STRONG PASS:** P1 AND P2 AND best corpus-derived >= 80% of trained_anchor gap.
**FAIL:** P1 not satisfied (no corpus arm reaches 50% threshold).
**PARTIAL:** P2 satisfied but P1 not (corpus arms beat controls but not enough).

## Feature extraction

Same 24-feature pipeline as g180/g182 at step 108 (3% of training). Enables cross-
experiment feature comparison.

## Compute envelope (per COMPUTE.md)

- 8 arms x 3 seeds = 24 cells x 5000 steps
- Per-cell estimate: ~6-8 min (same as g181b)
- Total: ~3-4h. Within 4h hard cap.
- VRAM: single Qwen3-768-8L model (~0.5 GB) + data. < 4 GB.
- Co-occurrence SVD: done on CPU (numpy), ~5 min for C4 subset.

## Artifacts

- `code/genome_183_corpus_derived_init.py`
- `results/genome_183_corpus_derived_init.json`
- `experiments/ledger.jsonl` entry
- WIKI + CLAIM_EVIDENCE_MAP patch

## Null result interpretation

If FAIL: the +0.513 gap requires structure that can only be learned by optimizing a
language modeling objective. Corpus statistics are insufficient. This is still publishable
as a negative (honest negative controls are rare). It would mean the "tokenizer prior"
is really a "pre-training prior" — the trained model learned token relationships that
corpus co-occurrence alone doesn't capture.

If PASS: The headline is transformative — "you don't need a donor model, you just need
a good corpus and a frequency/spectral initialization recipe." This is the manifesto
in action: Intelligence = Geometry, not Intelligence = Scale.
