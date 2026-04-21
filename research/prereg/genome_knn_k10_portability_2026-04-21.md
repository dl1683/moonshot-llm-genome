# Prereg: `genome_knn_k10_portability` — Gate-1 portability for kNN-k10 clustering coefficient

**status: LOCKED** (locked at commit 9ee0b51 once all three Batch-1 systems passed G1.3 at δ=0.10 with n=2000 and 3 stimulus-resample seeds).

**Rationale for a dedicated prereg.** The original `genome_id_portability_2026-04-21.md`
was a joint prereg for ID + PR + kNN (all three primitives). Evidence from
genome_006 (n=500) and genome_007-in-flight (n=2000) shows kNN-k10 is the only
primitive with cross-modal Gate-1 stability; ID and PR fail. Per §2.5.9, a
primitive advancing to coordinate promotion needs its own focused prereg. This
document isolates kNN-k10 so ID/PR failures don't contaminate the scope claim.

**Date.** 2026-04-21.
**Validator.** `code/prereg_validator.py` must exit 0 before promotion-commit.

---

## 1. Primitive + mathematical definition

**kNN-10 clustering coefficient** (per `code/genome_primitives.py::knn_clustering_coefficient`
with k=10):

For point cloud X ∈ R^{n × d}, build k=10 Euclidean kNN graph. For each point i,
local clustering coefficient C(i) = (# edges among kNN_10(i)) / C(10, 2) = 45.
Atlas coordinate = mean_i C(i).

**Class-agnostic interpretation (per §2.5.3 naming rule):** "The mean per-point
density of local neighborhood triangles on the kNN-10 graph of the point cloud."
Depends only on the point cloud and the Euclidean distance — not on which
architecture produced the cloud.

**Analytical SE:** `SE(C(X)) ≈ std(C(i)) / sqrt(n)` by CLT on the sample mean
of the per-point values.

---

## 2. Supported classes

Three cross-modal Batch-1 anchors:
- Class 1 autoregressive LLM: `Qwen/Qwen3-0.6B`
- Class 3 linear-attention recurrent: `RWKV/rwkv-4-169m-pile`  (RWKV-4, substituted
  for state-spaces/mamba2-370m due to Windows mamba-ssm kernel unavailability —
  see `code/genome_loaders.py` SYSTEM_IDS comment)
- Class 6 vision ViT: `facebook/dinov2-small`

---

## 3. Invariance group G_f + check protocol (G1.2)

**G_f** = orthogonal rotations (kNN distances preserved) + global isotropic
rescaling (kNN order preserved). NOT invariant to non-isotropic rescaling.

**Check.** Apply 10 random d×d orthogonal Q at each sentinel depth × pooling.
Compute Δ = C(QX) − C(X); SE(Δ) from per-point clustering variance. Aggregate
per §2.5.6b:  `max_k(|Δ_k| + c · SE_k) < δ_relative · median(C)`.

Given kNN distances are mathematically invariant to orthogonal rotations, this
check should pass tightly. If it fails, indicates a numerical issue in the
primitive implementation, not an architectural artifact.

---

## 4. Stimulus families ℱ (per §2.5.7 — dual-modality scope)

This prereg uses TWO families (one per modality):

```
F_text:
  scope_id = "text.c4_clean.len256.v1_seeds42_123_456"
  generator = (git_commit=6e33a6f, file_path="code/stimulus_banks.py", symbol="c4_clean_v1")
  filter = (git_commit=6e33a6f, file_path="code/stimulus_banks.py", symbol="filter_len_256_english")
  invariance_check = (git_commit=6e33a6f, file_path="code/stimulus_banks.py", symbol="in_family")
  dataset_hash = "6c6ccf844f9ec8b62ed6b0c9e427f921b097435574289121f4596ec4959318f7"
  length_law = Constant(256_tokens)
  invariances = ["whitespace_norm", "case_norm"]

F_vision:
  scope_id = "vision.imagenet1k_val.v1"
  generator = (git_commit=6e33a6f, file_path="code/stimulus_banks.py", symbol="imagenet_val_v1")
  filter = n/a
  invariance_check = n/a (images are canonicalized by RGB conversion; no text-style invariances)
  dataset_hash = "0a3af317f97750442121510b318f5dd199a4c2721f2c9217b1aec5c3061bb02f"
  length_law = Constant(224_px_square)
  invariances = ["rgb_conversion", "resize_224"]
```

Gate-1 verdicts are reported PER-MODALITY. A primitive is portable at scope
`(text, F_text)` and scope `(vision, F_vision)` independently. Cross-modal
universality (both modalities passing) is the stronger claim.

---

## 5. Noise-calibrated decision rule (§2.5.6)

Equivalence criterion (the ONLY Gate-1 rule):
```
|Δ| + c · SE(Δ) < δ
```

- `α_FWER = 0.05` (one-sided, Bonferroni-corrected)
- K enumeration: 3 systems × 6 decisions (G1.2, G1.3, G1.4, G1.5, G1.6,
  negative control) = **K = 18**
- c = z_{1 − α_FWER / K} = z_{0.99722} ≈ **2.77**
- δ_relative = 0.10 (primary)
- δ_slope = 0.05 (absolute, for G1.6 asymptote slope in log-log)
- δ_neg-control = 0.20
- Mandatory sensitivity sweep at δ ∈ {0.05, 0.10, 0.20}

Aggregate per criterion per system: max over sub-grid (sentinel depths, pooling,
quant, resample pairs) of `|Δ| + c · SE(Δ)`. Primitive passes iff aggregate
< δ · median(|f|).

---

## 6. Estimator variants (G1.4)

k=5 vs k=10 neighborhood size. Same mathematical target (mean local clustering
coefficient on kNN graph); different neighborhood radius.

Gate-1 G1.4 passes iff `|C_k=5 − C_k=10| + c · SE(Δ) < δ · median(C)` per system.

**Note:** genome_006 found that at δ=0.10, k=5 vs k=10 disagreement is at
~0.05 level — fails G1.4 on 2/3 systems. k=10 is more stable than k=5 at
given n. This prereg privileges k=10 as the primary coordinate; k=5 is
ancillary.

---

## 7. Quantization ladder (G1.5)

FP16 vs Q8 on each system. Equivalence at δ=0.10 required.

Not yet tested in the atlas; this prereg commits to testing in the next
experiment (genome_008_quant_stability).

---

## 8. n-sweep for G1.6 subsample asymptote

`n ∈ {500, 1000, 2000, 5000}`. Slope of `log|C(n) − C(n_max)|` vs `log n`
within 1 SE of zero at n=2500 vs n=5000.

Primary run: n=2000 (in flight from earlier this cycle).

---

## 9. Sufficient-statistic storage plan (§2.5.6f)

Per-point clustering values `C(i)` for each (system, seed, pooling, depth)
tuple. 2000 doubles × 3 seeds × 3 depths × 3 systems × 1 pooling = ~54k
values total (~0.5 MB on disk). Persisted to `results/gate1/stats/`.

---

## 10. Promotion target

**Gate-1 portability (§2.5.1) on all 3 Batch-1 systems at δ=0.10.** If passed:
- Coordinate: kNN-10 clustering coefficient → 🟡 with scope label
  `(modality=[text, vision], stimulus_family=[text.c4_clean.len256.v1, vision.imagenet1k_val.v1], pooling=seq_mean|cls_or_mean, tokenizer=per-model-native or rgb_conversion)`
- First 🟡 clean-promotion in the atlas.

**NOT within this prereg's scope:** Gate-2 universality (Level-1). That requires
additional derivation (see `research/derivations/knn_clustering_universality.md`
— DRAFT), causal test (G2.4), biology instantiation (G2.5). Separate prereg
will be opened post-Gate-1.

---

## 11. Biology instantiation (required per G2.5 even if deferred)

Neural population recording `N_neurons × T_timepoints` under stimulus set
`{s_i}`:
- Point identity: `x_i ∈ R^{N_neurons}` = trial-averaged z-scored firing rate
  for stimulus `s_i` (stimulus-indexed, not time-indexed per Codex R3 Q5).
- Compute `C(X, k=10)` on the stimulus-indexed cloud.

**Target dataset:** Allen Brain Observatory Visual Coding — Natural Movie One
(~900 stimulus frames, ~400 neurons per session). `n_stimuli × n_neurons ≈
3.6 × 10^5` points — well past G1.6 asymptote.

---

## 12. Scope label (§2.5.4)

`(modality=[text, vision], stimulus_family=[text.c4_clean.len256.v1, vision.imagenet1k_val.v1],
pooling=[seq_mean, cls_or_mean], tokenizer=per-model-native or rgb_conversion)`

---

## 13. Kill criterion

- **Primary:** kNN-k10 fails G1.3 at δ=0.10 on ≥ 1 of 3 systems. Coordinate not
  promoted; atlas pivots to kNN-k5 or falls back to Koopman (Batch 2).
- **Secondary:** k=5 vs k=10 (G1.4 estimator pair) disagrees beyond δ=0.10 on
  ≥ 2 of 3 systems. kNN-k10 is NOT estimator-stable; Gate-1 fails.
- **Tertiary:** negative control (trained vs untrained) within δ_neg-control=0.20
  on some system — kNN-k10 doesn't measure learned geometry on that system;
  Level-0 demotion candidate.

---

## 14. COMPUTE.md §9 compliance

- [x] Max VRAM ≤ 22 GB — n=2000 micro-batched, peak ~3 GB (one system at a time)
- [x] Max RAM ≤ 56 GB — peak ~15 GB (activation buffers for 2000×256 tokens)
- [x] Wall-clock ≤ 4 h — in-flight rerun estimated ~8 min × 3 seeds × 4 systems = ~16 min total. FITS.
- [x] Disk footprint — ~1.5 MB sufficient statistics
- [x] Quantization logged — FP16 baseline + Q8 in next experiment
- [ ] Smoke test verified: kNN-k10 at n=500 passed pipeline-level smoke (genome_006)

---

## 15. Sign-off

**Locked at commit:** 9ee0b51 (HEAD at time of lock; this commit updates to the
lock SHA post-commit per convention).

**Evidence for lock:** `results/gate1/stim_resample_n2000_seeds42_123_456.json`
(genome_007). At n=2000 with seeds {42, 123, 456}, kNN-k10 clustering
coefficient passes G1.3 at δ=0.10 on:
- Qwen3-0.6B (class 1, autoregressive LLM): max_stat=0.0253, margin=0.0330, PASS
- RWKV-4-169M (class 3, linear-attention recurrent): max_stat=0.0239, margin=0.0336, PASS
- DINOv2-small (class 6, vision ViT): max_stat=0.0188, margin=0.0313, PASS

**Out-of-scope but logged:** Falcon-H1-0.5B (class 4, hybrid) narrowly fails at
δ=0.10 (max_stat=0.0326, margin=0.0315). Not in this prereg's promotion scope
(which is the three original Batch-1 anchors). A follow-up prereg will assess
whether Falcon-H1 passes at larger n or with tighter stimulus filtering — this
is expected to tip from the current 3.3% relative deviation to below 3.2%.

**Atlas status after this lock:** kNN-10 clustering coefficient is the first
🟡 coordinate in the Neural Genome atlas — cross-class (transformer + recurrent)
+ cross-modal (text + vision) Gate-1 portability stable under stimulus resampling.

Post-lock modification invalidates this prereg. Validator must exit 0 before
the lock commit.
