# Pre-registration: genome_183 Corpus-Derived Interface Prior (Causal Interface-Prior Synthesis)

**STATUS:** DRAFT pre-staged 2026-04-30 cycle 138. Revised per Codex design gate (g183_design_gate_20260430.md). Supersedes 2026-04-29 version.

- Date: 2026-04-30
- Trigger: g186 FAIL (geometry does NOT predict KD dose-response). g183 is the highest-leverage fireable rescue per Codex advisor (6.4/10). Tests whether the g181b +0.513 nat interface-prior effect can be synthesized from corpus statistics alone.

---

## Hypothesis

**H1 (corpus-derived init recovers donor-anchor benefit):** A vocabulary initialization constructed from corpus PPMI co-occurrence + SVD -- with NO trained model -- recovers >= 50% of the g181b +0.513 nat gap vs scratch at 5000 steps, and beats all control arms.

**H0 (trained-model structure required):** Corpus-derived initializations produce negligible or noisy benefit; the +0.513 gap requires structure learned during actual pre-training on a language modeling objective.

## Universality level claim

Level-0 single-architecture mechanism finding.

## Motivation

g181a showed the +1 nat anchor effect is ~100% embed_tokens + lm_head trained-init (transformer block weights HARM when anchored). g181b showed this persists at 5000 steps (+0.513 nats). g186 FAIL killed the geometry-based dose-response prediction claim. The natural question: is the active ingredient specific learned token representations from pre-training, or can we derive equivalent structure from corpus statistics? If the latter, the headline shifts from "transfer from trained model" to "training-design law: cheap corpus-derived interface priors reduce wasted training."

## Arms

### Stage A (primary, <=4h wall-clock): 3 arms x 3 seeds = 9 cells

All arms: Qwen3-arch random-init recipient, 768-dim, 8 layers. C4 training, 5000 steps. Seeds [42, 7, 13].

1. **scratch_ce** -- Baseline, random init, CE only. No anchor.
2. **trained_anchor** -- Qwen3 embed + lm_head anchor at lambda=0.01 (reproduces g181b). Reference ceiling.
3. **ppmi_svd_anchor** -- PPMI co-occurrence matrix from C4 train corpus (window=5), truncated SVD to 768 dims. Initialize embed from left singular vectors scaled by sqrt(singular values). lm_head tied to embed. Continuous anchor at lambda=0.01. **Primary treatment arm.**

### Stage B (if Stage A ppmi_svd >= 0.15 nats vs scratch): 4 arms x 3 seeds = 12 cells

4. **frequency_anchor** -- Random orthonormal directions scaled by unigram log-frequency norm. Continuous anchor at lambda=0.01. Tests whether frequency structure alone helps.
5. **random_structured_anchor** -- Random orthonormal embed scaled to match Qwen3 embed Frobenius norm. Tests whether scale matching alone helps.
6. **ppmi_svd_shuffled_rows** -- Same PPMI SVD spectrum and rows as arm 3, but token-to-row mapping randomly permuted. Continuous anchor at lambda=0.01. **Key adversarial control -- if this matches ppmi_svd, token identity doesn't matter.**
7. **covariance_matched_anchor** -- Match row covariance matrix of Qwen3 trained embed using trained aggregate statistics. Anchor at lambda=0.01. **Trained-stat control, not corpus-derived.**
8. **spectral_matched_anchor** -- Match singular value spectrum of trained Qwen3 embed, fill singular vectors with random orthonormal basis. Anchor at lambda=0.01. **Trained-stat control, not corpus-derived.**

## Pass / fail criteria

| # | Criterion | Threshold |
|---|---|---|
| P1 | ppmi_svd_anchor recovers >= 50% of trained_anchor gap vs scratch | mean(ppmi_svd - scratch) >= 0.257 nats |
| P2 | ppmi_svd_anchor beats scratch in 3/3 seeds | per-seed ppmi_svd final NLL < scratch final NLL |
| P3 | ppmi_svd_anchor beats best non-corpus control by >= 0.10 nats | mean gap >= 0.10 |
| P4 | ppmi_svd_anchor beats ppmi_svd_shuffled_rows (Stage B) | 3/3 seeds, mean gap > 0 |

**PASS:** P1 AND P2 AND P3 all satisfied.
**STRONG PASS:** P1+P2+P3 AND ppmi_svd >= 80% of trained_anchor gap AND P4.
**FAIL:** P1 not satisfied (ppmi_svd doesn't reach 50% threshold).
**PARTIAL:** P2 satisfied but P1 not (corpus beats scratch but not enough).

## Staged launch gate

Stage B fires ONLY if Stage A ppmi_svd_anchor shows >= 0.15 nats mean gain vs scratch (low bar to justify more compute). If Stage A ppmi_svd < 0.15 nats vs scratch, Stage B is skipped and the experiment closes with Stage A data only.

## Feature extraction

Same 8-feature manifold pipeline as g186 at step 108 (3% of training). Enables cross-experiment feature comparison.

## Compute envelope (per COMPUTE.md)

### Stage A
- 3 arms x 3 seeds = 9 cells x 5000 steps
- Per-cell estimate: ~22 min (per g181b empirical)
- Total: ~3.3h + SVD preprocessing (~5 min). Within 4h hard cap.
- VRAM: single Qwen3-768-8L model (~0.5 GB) + data + donor params on device for trained_anchor. < 6 GB peak.

### Stage B (conditional)
- 4 arms x 3 seeds = 12 cells x 5000 steps
- Total: ~4.4h. Requires separate launch (cannot combine with Stage A in one 4h window).
- VRAM: same as Stage A.

## Artifacts

- `code/genome_183_corpus_derived_init.py`
- `results/genome_183_corpus_derived_init.json`
- `experiments/ledger.jsonl` entry
- WIKI + CLAIM_EVIDENCE_MAP patch

## Null result interpretation

If FAIL: the +0.513 gap requires structure that can only be learned by optimizing a language modeling objective. Corpus statistics are insufficient. This narrows the tokenizer-prior to "pre-training prior" -- the trained model learned token relationships that corpus co-occurrence alone doesn't capture.

If PASS: "You don't need a donor model, you just need a good corpus and a PPMI/spectral initialization recipe." This is the manifesto in action: Intelligence = Geometry, not Intelligence = Scale. Headline: cheap corpus priors replace expensive donor training.

## COMPUTE.md compliance

- [x] Peak VRAM <= 22 GB (empirical: ~6 GB per cell including donor params)
- [x] System RAM <= 56 GB (sparse co-occurrence matrix estimated ~2 GB for top-50K vocab)
- [x] Wall-clock <= 4h per stage (Stage A ~3.3h, Stage B ~4.4h)
- [x] No cloud compute required
- [x] Save/resume: per-cell atomic writes (same as g181b pattern)
- [x] Windows/CUDA rules: num_workers=0, pin_memory=False, n_jobs=1
