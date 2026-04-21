# Prereg: `genome_knn_k10_batch2` — Gate-1 extension to encoder / contrastive systems

**status: STAGED** (flips to LOCKED when the new systems have completed their
first G1.2 rotation-invariance smoke and dataset hashes are computed for
any new stimulus families).

**Rationale.** The Batch-1 locked prereg (`genome_knn_k10_portability_2026-04-21.md`)
tested kNN-k10 on 3 systems that are all essentially decoder / autoregressive /
self-supervised-ViT. A reviewer could legitimately say the universality
claim is narrow: "autoregressive text + one self-supervised ViT" is not
"cross-architecture." This prereg adds **distinct training objectives**
(MLM, contrastive-text, contrastive-vision) that the Batch-1 scope is
blind to. If kNN-k10 passes on these too, the universality claim
becomes training-objective-invariant, not just architecture-invariant.

**Date.** 2026-04-21.
**Validator.** `code/prereg_validator.py` must exit 0 before promotion-commit.

---

## 1. Primitive + mathematical definition

Same as `genome_knn_k10_portability_2026-04-21.md` §1:
- `C(X, k=10)` = mean per-point clustering coefficient on Euclidean kNN-10 graph
- Analytical SE: `std(C(i)) / sqrt(n)` by CLT
- Class-agnostic: depends only on the point cloud.

Derivation (LOCKED at `62338b8`): `research/derivations/knn_clustering_universality.md`.

---

**Scope declaration.**
```
scope_id = "batch2.encoder_contrastive.v1"
```

## 2. Supported classes (NEW)

Three additions to the 5-class Batch-1 bestiary:

- Class 7 masked-LM encoder: `bert-base-uncased`
  - Training: MLM (bidirectional, not causal). Distinct from CLM.
  - Pooling: `seq_mean` across all token positions (same as Batch-1 text).
- Class 8 contrastive text encoder: `sentence-transformers/all-MiniLM-L6-v2`
  - Training: contrastive sentence-pair loss. Different from both MLM and CLM.
  - Pooling: `seq_mean`.
- Class 10 contrastive vision encoder: `openai/clip-vit-base-patch32`
  - Training: CLIP image-text contrastive. Vision branch only.
  - Pooling: `cls_or_mean` (same as DINOv2).

---

## 3. Invariance group G_f + check protocol (G1.2)

Same as Batch-1 §3. Orthogonal rotations + isotropic rescaling. 10 Haar-
random Q applied to each sentinel-depth × pooling point cloud. Equivalence
criterion `max_k(|Δ_k| + c · SE_k) < δ_relative · median(C)`.

---

## 4. Stimulus families ℱ

**No new ℱ required for BERT + MiniLM** — both use the same
`F_text = text.c4_clean.len256.v1_seeds42_123_456` as Batch-1 (dataset_hash
`6c6ccf844f9ec8b62ed6b0c9e427f921b097435574289121f4596ec4959318f7`).
Pinned pointers:
- generator: `(git_commit=6e33a6f, file_path="code/stimulus_banks.py", symbol="c4_clean_v1")`
- filter: `(git_commit=6e33a6f, file_path="code/stimulus_banks.py", symbol="filter_len_256_english")`
- invariance_check: `(git_commit=6e33a6f, file_path="code/stimulus_banks.py", symbol="in_family")`

**No new ℱ required for CLIP-vision** — uses
`F_vision = vision.imagenet1k_val.v1` from Batch-1 (dataset_hash
`0a3af317f97750442121510b318f5dd199a4c2721f2c9217b1aec5c3061bb02f`).
Pinned pointer:
- generator: `(git_commit=6e33a6f, file_path="code/stimulus_banks.py", symbol="imagenet_val_v1")`

Reusing the locked families is by design: the universality claim is
"the same geometry on the same stimuli regardless of which model saw them."

---

## 5. Noise-calibrated decision rule (§2.5.6)

Same as Batch-1 §5:
- `α_FWER = 0.05`
- K enumeration: 3 systems × 6 decisions = **K = 18**
- c = z_{1 − 0.05/18} ≈ **2.77**
- δ_relative = 0.10 (primary); δ_slope = 0.05; δ_neg-control = 0.20
- Mandatory sensitivity sweep at δ ∈ {0.05, 0.10, 0.20}

The K counted here is over the 3 NEW systems. The combined (Batch-1 + Batch-2)
picture will report per-system pass/fail at δ=0.10 and aggregate.

---

## 6. Estimator variants (G1.4)

k=5 vs k=10. Same rule as Batch-1 §6. k=10 primary.

---

## 7. Quantization ladder (G1.5)

FP16 for primary run. Q8 follow-up deferred (CLIP vision bnb less-tested
on Windows; see Batch-1 prereg §7).

---

## 8. n-sweep for G1.6 subsample asymptote

`n ∈ {500, 1000, 2000, 4000}` if any system fails G1.3 at n=2000; otherwise
n=2000 is primary. Falcon-H1 Batch-1 precedent: n=2000 narrow-failed,
n=4000 tipped clean — so 4000 is the escape hatch if a new system does the
same.

---

## 9. Promotion target

**Gate-1 portability on all 3 new systems at δ=0.10.** If achieved:
- Coordinate: kNN-10 clustering coefficient → scope extended from
  `(modality ∈ {text, vision}, stimulus_family ∈ {c4_clean, imagenet1k_val},
   training_objective ∈ {CLM, self-sup-ViT})`
  to
  `(modality ∈ {text, vision}, stimulus_family ∈ {c4_clean, imagenet1k_val},
   training_objective ∈ {CLM, MLM, contrastive-text, self-sup-ViT, contrastive-vision})`
- Bestiary coverage: 5 classes → 8 classes (adds 7, 8, 10).
- Strengthens the Level-1 claim without re-LOCKing the original 3-system prereg.

**NOT within this prereg's scope:** Gate-2 causal / biology (separate prereg
`genome_knn_k10_causal_2026-04-21.md` covers that).

---

## 10. Kill criteria

- **Primary:** kNN-k10 fails G1.3 at δ=0.10 on ≥ 2 of 3 new systems. The
  universality claim is narrower than hoped: restricted to decoder-LM +
  self-supervised-ViT family. Not a total failure — the Batch-1 claim
  stands — but a publishable scope restriction.
- **Secondary:** kNN-k10 fails on exactly 1 new system. Investigate whether
  it's a loader artifact (like Falcon-H1's n=2000 narrow-fail) or a genuine
  counterexample. Escalate to n=4000 on the failing system.

---

## 11. COMPUTE.md §9 compliance

- [x] Max VRAM ≤ 22 GB — all 3 new systems are small (22M-151M params),
  peak VRAM ~3 GB per system.
- [x] Max RAM ≤ 56 GB — activation buffers ~10 GB at n=2000 for CLIP.
- [x] Wall-clock ≤ 4 h — estimate: BERT ~60s, MiniLM ~40s, CLIP ~50s per
  seed, 3 seeds each → ~9 min wall-clock total. WELL within envelope.
- [x] Disk footprint: ~0.5 MB sufficient statistics.
- [x] Quantization FP16 baseline logged in ledger.
- [ ] Smoke test verified: run BERT at n=50 seed=42 and verify extractor
  produces non-degenerate point clouds. PRE-LOCK BLOCKER.

---

## 12. Scope label

`(modality ∈ {text, vision}, stimulus_family ∈ {c4_clean.len256.v1, imagenet1k_val.v1},
pooling ∈ {seq_mean, cls_or_mean}, training_objective ∈ {CLM, MLM, contrastive-text,
contrastive-vision, self-sup-ViT})`

---

## 13. Sign-off

**Status:** STAGED. Flips to LOCKED when:
1. BERT + MiniLM + CLIP each pass a 50-stimulus smoke test with non-degenerate
   clustering values (i.e., point clouds not collapsed to a single point).
2. `code/prereg_validator.py` exits 0 on this file.
3. This commit's SHA is backfilled into the Locked-at line.

**Locked at commit:** `<pending — backfill post-lock>`.

Post-lock modification invalidates this prereg. Commit-message at lock:
`Lock prereg genome_knn_k10_batch2_2026-04-21 — encoder/contrastive extension`.
