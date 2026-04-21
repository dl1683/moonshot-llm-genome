# Prereg: `genome_knn_k10_biology` — Gate-2 G2.5 biological instantiation

**status: STAGED** (flips to LOCKED once Allen Brain Observatory Neuropixels
Natural-Movie-One slice is pulled + `code/stimulus_banks.py` gains
`allen_neuropixels_v1` + dataset hash is computed).

**gate: 2**

**Rationale.** Level-1 universality is the project's sacred outcome S3 (per
`research/atlas_tl_session.md` §0b). If kNN-10 is a geometry-of-learning
invariant, it should take indistinguishable values on biological neural
populations under the same stimulus distribution that a trained vision
model sees. This prereg specifies the biology bridge — G2.5 per §2.5.2 —
that moves kNN-10 from "universal across trained networks" to "universal
across learning systems, biological + artificial."

---

## 1. Question

Does kNN-10 clustering coefficient on mouse visual-cortex population
responses to natural movie stimuli equal kNN-10 on DINOv2-small's CLS
activation to the *same stimuli*, within the Bonferroni-corrected
δ=0.10 equivalence margin?

- **Pass:** kNN-10(biology) ≈ kNN-10(DINOv2) within δ=0.10. Level-1
  universality spans biological + artificial.
- **Fail:** values differ by more than δ=0.10. Either (a) DINOv2 is not
  a good stand-in for cortex (modality-level claim, still publishable),
  or (b) the manifold-hypothesis derivation is wrong for cortex.

---

## 2. Target primitive + derivation pointer

- Primitive: `code/genome_primitives.py::knn_clustering_coefficient`, k=10.
- Derivation (LOCKED at 62338b8): `research/derivations/knn_clustering_universality.md` §7.
- Gate-1 comparison baseline: DINOv2 kNN-k10 passes G1.3 at δ=0.10
  (genome_007) with values 0.31-0.35 across depths on ImageNet-val.

---

## 3. Biological system

**Allen Brain Observatory — Visual Coding (Neuropixels).**
- **Species:** mouse (Mus musculus, C57BL/6J; ~70-90 days old).
- **Brain areas:** visual cortex (VISp, VISl, VISal, VISrl, VISpm, VISam).
- **Stimulus:** Natural Movie One (30s continuous movie, repeated trials).
- **Recording modality:** Neuropixels 1.0 probes, simultaneous ~400 neurons
  per session.
- **Access:** `remfile + h5py + dandi` (DANDI archive dandiset 000021 or
  000024 depending on version). **Do NOT use `allensdk`** — per CLAUDE.md §6
  Windows+CUDA constraint, allensdk is incompatible with Python 3.13.

---

## 4. Stimulus-indexed point cloud

Per `research/derivations/knn_clustering_universality.md` §7 (locked):

For each stimulus frame `s_i` in Natural Movie One (~900 frames at 30fps):
- `x_i ∈ ℝ^{N_neurons}` = trial-averaged, z-scored firing rate vector
  from the population response to frame `s_i`, time-locked to frame onset
  + a 50 ms integration window.
- Point cloud: `X = {x_i}_{i=1}^{n_stimuli}` where `n_stimuli` = 900 frames
  (or 540 at the 540-frame Natural Movie One Short subset).

Compute `C(X, k=10)` on this cloud.

---

## 5. Matching ANN point cloud

Re-extract DINOv2-small CLS activations on the SAME 900 Natural Movie One
frames (convert movie → frame images → DINOv2 vision stimulus bank). Use
the mid-depth sentinel (layer index 6 of 12, normalized depth ≈ 0.55) as
the primary comparison — DINOv2's natural-feature regime.

Compute `C(X_DINOv2, k=10)` on this cloud.

---

## 6. Noise-calibrated decision rule

Per §2.5.6:

`| kNN-10(biology) − kNN-10(DINOv2) | + c · SE_delta < δ · median(|kNN-10|)`

with:
- c = 2.77 (Bonferroni, K=18 from Gate-1 rule inherited — reasonable
  extension because biology is one more "system" in the bestiary).
- δ_relative = 0.10 (primary); sensitivity sweep at {0.05, 0.10, 0.20}.
- SE_delta = sqrt(SE_biology² + SE_DINOv2²) per §2.5.6b.

Per-session (Allen has ~30 available sessions): evaluate per-session
pass/fail AND pooled across sessions. If ≥ 60% of sessions pass at δ=0.10,
that is a strong positive — accounts for inter-mouse variability. If
every session fails, the claim is falsified cleanly.

---

## 7. Controls

Four required controls (per derivation §7):

### 7a. Shuffle control
Randomly permute stimulus labels on the Allen data. kNN-10 on shuffled
data should be **different** from kNN-10 on DINOv2 by MORE than δ.
(If equivalent even under shuffle, we're measuring something that
doesn't depend on stimulus identity — a bug.)

### 7b. Different-movie control
kNN-10 on Natural Movie Two (different movie, same mouse) should be
equivalent to kNN-10 on Natural Movie One (same movie, same mouse) iff
the geometry is a property of *natural-scene statistics* not this
particular movie. Pre-registered expected-pass.

### 7c. Untrained-DINOv2 control
Random-init DINOv2 CLS vectors on the same stimulus set should have
LARGER `|Δ|` vs biology than trained DINOv2. Confirms the biology match
is about LEARNED representations not architectural priors.

### 7d. Visual-area-specificity
kNN-10 on VISp (primary visual) vs VISam (anterior medial, higher-order).
Different visual areas have different representational geometries. A
positive Level-1 claim requires BOTH areas to match DINOv2 within δ=0.10,
which is strong. Weaker claim: match in one area only.

---

## 8. Pre-registered success criteria

**Primary (Level-1 cross-bio claim):** `| kNN-10(biology, VISp) − kNN-10(DINOv2, mid-depth) | + c·SE < 0.10 · median` on ≥ 60% of Allen sessions. AND shuffle control (7a) fails the same test. AND untrained-DINOv2 control (7c) fails.

**Secondary (modality-level):** primary fails but same test on Natural Movie Two passes. Claim: kNN-10 is movie-specific, not universal across stimulus contexts.

**Falsification:** neither primary nor secondary passes on any Allen session. kNN-10 is NOT a biology-bridged coordinate; Level-1 claim limited to trained-ANN scope only.

---

## 9. Kill criteria

- **Primary:** <60% of Allen sessions pass primary test AND shuffle control also passes (indicating the primitive is measuring something universal including random — probably a numerical artifact in the implementation). Hard fail. Revisit implementation.
- **Secondary:** >90% of sessions pass BUT with per-session kNN-10 values all ≈1 (like PR_uncentered DC artifact). DC or trivial signal. Investigate before claiming.
- **Tertiary:** untrained-DINOv2 control (7c) passes at comparable levels to trained. Means the kNN-10 equivalence is architectural not learned — demote claim.

---

## 10. Implementation plan

NEW code:
- `code/stimulus_banks.py::allen_neuropixels_v1` (generator for AllenSDK-free
  Neuropixels slices via remfile+h5py+dandi).
- `code/genome_biology_extractor.py` (loads the Allen session, computes the
  stimulus-indexed firing-rate cloud, runs `knn_clustering_coefficient`).
- `code/genome_biology_comparison.py` (pairs biology kNN-10 with DINOv2
  kNN-10 on matched frames, evaluates the equivalence criterion).

REUSE:
- `code/genome_primitives.py::knn_clustering_coefficient` (same primitive).
- `code/genome_stim_resample.py::evaluate_g13_sensitivity` (same equivalence
  machinery, extended to the bio-vs-DINOv2 pair).

---

## 11. COMPUTE.md §9 compliance

- [x] Max VRAM ≤ 22 GB — biology point cloud fits in CPU RAM
  (900 stimuli × 400 neurons = 360 KB per session); DINOv2 extraction at
  n=900 ≈ 10s GPU.
- [x] Max RAM ≤ 56 GB — Allen session raw spike data ≈ 2 GB per session
  loaded via streaming.
- [x] Wall-clock ≤ 4 h — per-session evaluation ≈ 15 min; 30 sessions ≈
  7.5 h total, **exceeds envelope** → stage in two batches OR sample
  10 sessions first.
- [x] Disk: Allen session cache ≈ 20 GB total. Acceptable.
- [ ] Smoke test: pull ONE session and verify pipeline end-to-end before
  lock.

---

## 12. Scope label

`(modality=vision (natural-scene movies), stimulus_family=allen_natural_movie_one,
pooling=[neuron-population-vector for biology, CLS for DINOv2],
tokenizer=n/a)`

---

## 13. Sign-off

**Status:** STAGED. Flips to LOCKED after:
1. Allen Neuropixels slice pulled + dataset_hash computed.
2. `code/genome_biology_extractor.py` exists + smoke passes.
3. Validator exits 0 in gate:2 G2.5 mode.
4. Commit SHA backfilled.

Post-lock modification invalidates the Gate-2 G2.5 verdict.

**Locked at commit:** `<pending>`.
