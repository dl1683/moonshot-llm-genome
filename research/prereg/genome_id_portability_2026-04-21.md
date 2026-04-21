# Prereg: `genome_id_portability` — Gate-1 portability test for Intrinsic Dimension

**status: STAGED** — governance discipline per Codex R6 Part V and `code/prereg_validator.py` placeholder rejection rule.

A STAGED prereg has all design decisions committed but still contains fill-in fields that cannot be pinned until real data is on disk (specifically `dataset_hash = PLACEHOLDER_sha256_...`). To promote to `status: LOCKED`:
1. Pull the C4-clean slice per the `c4_clean_v1` generator spec.
2. Compute `sha256` of that slice.
3. Replace `PLACEHOLDER_sha256_...` with the real hash.
4. Commit with message `Lock prereg genome_id_portability_2026-04-21`.
5. Run `python code/prereg_validator.py research/prereg/genome_id_portability_2026-04-21.md` — must pass with 0 errors AND the placeholder-rejection check against a LOCKED status.

Post-LOCK modifications invalidate this prereg. Current STAGED status allows dataset_hash to be filled in without invalidating the rest.

**Date.** 2026-04-21.

**Canonical rule source.** `research/atlas_tl_session.md` §2.5 (two-gate spec), §2.5.6 (noise-calibrated decision rule), §2.5.9 (prereg template this follows).

**Validator.** `code/prereg_validator.py` — MUST exit 0 against this file prior to launch.

---

## 1. Primitive name + mathematical definition

**Primitive.** Intrinsic dimension via TwoNN (Facco et al. 2017).

**Class-agnostic mathematical definition (per §2.5.3 naming rule).**
For a point cloud `X ∈ R^{n × d_ext}` sampled from a manifold with intrinsic dimension `d`, the nearest-two-neighbor distance ratio `μ_i = r_2(i) / r_1(i)` is Pareto-distributed with scale parameter `d`:
```
P(μ ≥ m) = m^{-d}    for m ≥ 1
```
The intrinsic dimension `d` is the scale parameter. MLE from the sample `{μ_i}_i`:
```
d_hat = n / Σ_i log(μ_i)
```

"The intrinsic dimension of the manifold sampled by X, computed as the scale parameter of the nearest-two-neighbor distance ratio distribution."

---

## 2. Supported classes targeted in this prereg

Three matched-modality language classes:
- Class 1 autoregressive LLM: `Qwen/Qwen3-0.6B`
- Class 3 SSM: `state-spaces/mamba2-370m-hf`
- Class 4 hybrid: `tiiuae/Falcon-H1-0.5B-Instruct`

All from the canonical registry at `../../models/MODEL_DIRECTORY.md`.

---

## 3. Invariance group `G_f` + invariance check protocol (G1.2)

**G_f** = orthogonal rotations + global isotropic rescaling. NOT invariant to token permutation (mitigated by seq-mean pooling) or stimulus resampling (tested separately via G1.3).

**Check.** Apply 10 random d×d orthogonal matrices `Q_k` at each sentinel depth + pooling. Compute `Δ_k = ID(Q_k · X) − ID(X)` and its SE via TwoNN analytical formula. Aggregate per §2.5.6b: `max_k(|Δ_k| + c · SE_k) < δ_relative · median(ID)`.

---

## 4. Stimulus family `ℱ` (machine-checkable per §2.5.7)

```
scope_id = "text.c4_clean.len256.v1_seeds42_123_456"
generator = (git_commit=6edf303, file_path="code/stimulus_banks.py", symbol="c4_clean_v1")
filter = (git_commit=6edf303, file_path="code/stimulus_banks.py", symbol="filter_len_256_english")
invariance_check = (git_commit=6edf303, file_path="code/stimulus_banks.py", symbol="in_family")
dataset_hash = "PLACEHOLDER_sha256_locked_at_first_real_run"
length_law = Constant(256_tokens_per_sentence)
invariances = ["whitespace_norm", "case_norm"]
```

Three seed-disjoint resamples via `ℱ.generator(seed)` with seeds 42, 123, 456.

Pinned-pointer resolution verified at prereg commit time by `code/prereg_validator.py` (file_exists + symbol_defined AST check). `git_commit=6edf303` means "the commit that locks this prereg."

---

## 5. Noise-calibrated decision rule (§2.5.6)

**Equivalence criterion (the ONLY Gate-1 rule — no `τ`, no `|z|<c`):**
```
|Δ| + c · SE(Δ) < δ
```
applied per-criterion after per-system aggregation (max over sub-grid).

- `α_FWER = 0.05` (one-sided, Bonferroni-corrected)
- `K = 18` from `3 systems × 6 decisions`:
  - G1.2 rotation invariance
  - G1.3 stimulus resample
  - G1.4 estimator variant (TwoNN vs MLE)
  - G1.5 quantization (FP16 vs Q8)
  - G1.6 subsample asymptote
  - Negative control (trained vs untrained)
- Untrained twins explicitly enumerated (Codex R6 B3 closure): each Batch-1 system gets one randomly-initialized twin — `Qwen/Qwen3-0.6B` random-init, `state-spaces/mamba2-370m-hf` random-init, `tiiuae/Falcon-H1-0.5B-Instruct` random-init. Each contributes one negative-control decision → 3 total in K.
- `c = z_{1 − α_FWER / K} = z_{0.99722} ≈ 2.77`

**Equivalence margins (prereg LOCKED; no post-hoc):**
- `δ_relative = 0.10` (applied as fraction of primitive's median on the test cloud, for G1.2/G1.3/G1.4/G1.5)
- `δ_slope = 0.05` (absolute, for G1.6 asymptote slope in log-log space)
- `δ_neg-control = 0.20` (relative; trained-vs-untrained must differ by more than this)

**Mandatory sensitivity sweep:** report Gate-1 pass/fail at `δ_relative ∈ {0.05, 0.10, 0.20}` in every ledger entry. If verdict flips between δ=0.10 and δ=0.05, the atlas row is annotated `🟡 (δ-sensitive)` per §2.5.6c.

**Gate-1 checks are NOT run on the full 21-point depth grid** — that balloons compute. They run on sentinel depths `ℓ/L ∈ {0.25, 0.50, 0.75}` (3 points). Full depth curve is Phase-2 descriptive observation, not Gate-1. (Per §2.5.6b revision and R5 audit §V.)

---

## 6. Estimator variants (for G1.4 stability)

- **TwoNN** (Facco et al. 2017, scale parameter of Pareto ratio distribution)
- **MLE** (Levina & Bickel 2004, same target under likelihood formulation)

Both compute the SAME mathematical target (intrinsic dimension), so they are true estimator variants. Equivalence criterion applies to their difference on matched point clouds.

---

## 7. Quantization ladder points (for G1.5 stability)

- **FP16** via HuggingFace Transformers default
- **Q8** via `bitsandbytes` 8-bit quantization at inference

Both on each of the three models (3 × 2 = 6 quantization configurations). Equivalence criterion applied within each model, across FP16 vs Q8.

---

## 8. n-sweep for G1.6 subsample asymptote

`n ∈ {500, 1000, 2000, 5000}` at sentinel depth `ℓ/L = 0.5` on seq-mean pooling. Slope of `log|ID(n) − ID(5000)| vs log n` must be within 1 SE of zero at n=2500 vs n=5000. (SE from TwoNN analytical formula `d/√n`.)

---

## 9. Analytical SE or sufficient-statistic storage plan (§2.5.6f)

**Analytical SE for TwoNN.** Under Pareto MLE asymptotics, `SE(d_hat) = d_hat / √n`. For n=5000, SE ≈ 0.014 · d_hat. No bootstrap needed.

**Sufficient statistics persisted per (system, pooling, resample, quant) tuple (36 tuples total):**
- `log μ_i` for all n=5000 points at all sentinel + full depths → ~40 KB per file

Raw point clouds are NOT persisted — streaming extraction, per-layer sufficient-stat dump, activations freed after compute. Per §3b "SUFFICIENT STATISTICS ONLY" policy.

Total disk footprint for this prereg: 36 tuples × ~40 KB = **~1.5 MB**.

---

## 10. Promotion target

**Gate 1 portability** (§2.5.1) on all three language classes. If passed: status → 🟡 "coordinate (portability gate passed)" with scope label `(modality=text, stimulus_family=text.c4_clean.len256.v1, pooling=seq_mean_and_per_token_subsample, tokenizer=per-model-native)`.

Level-1 universality (Gate 2, §2.5.2) is NOT within this prereg's scope. Requires subsequent prereg with G2.2 derivation + G2.4 causal test + G2.5 biology validation.

---

## 11. Biology instantiation (Codex R2 Q7; §2.5.2 G2.5 required declaration even if execution deferred)

Given neural population recording `N_neurons × T_timepoints` under stimulus conditions `{s_i}`:
- **Point identity:** `x_i ∈ R^{N_neurons}` = population response vector for stimulus condition `s_i`, specifically trial-averaged z-scored firing rate (Neuropixels) or BOLD activation (fMRI) over a stimulus-locked window. Points index STIMULI, not time (Codex R3 Q5).
- **Binning:** stimulus-locked (frame-locked ~33 ms for natural-movie; trial-locked for fixed-duration stimuli).
- **G1.6 asymptote on biology:** requires `n_stimuli × n_neurons ≥ 10^5` for TwoNN to stabilize; Allen Neuropixels Visual Coding Natural Movie datasets meet this.
- **Execution deferred to Batch 4** (see `research/atlas_tl_session.md §3e`).

---

## 12. Scope label (§2.5.4)

`(modality=text, stimulus_family=text.c4_clean.len256.v1, pooling=seq_mean_and_per_token_subsample, tokenizer=per-model-native)`

Any Gate-1 pass is conditional on this scope until cross-modal extension (§2.5.8 governance rule). Cross-modal extension requires a NEW prereg.

---

## 13. Kill criterion

- **Primary (primitive fails Gate 1 on ≥ 1 of 3 classes at δ=0.10).** ID is not portable on language; the atlas pivots to alternative primitives (clustering coefficient remains; Ricci / Koopman / PH move from Batch 2 to Batch 1'). Logged as FAIL in ledger.
- **Secondary (Gate 1 passes at δ=0.10 but fails at δ=0.05).** Portability is weak — annotated `🟡 (δ-sensitive)`; Level-2 universality claim deferred until δ can be tightened.
- **Tertiary (negative control fails: `|ID_trained − ID_untrained| < δ_neg-control`).** ID measures architecture not learned geometry; demoted from 🟡 to ⚪ Level-0 diagnostic.

---

## 14. COMPUTE.md §9 compliance checklist

- [x] Max VRAM ≤ 22 GB — peak ~3 GB (three models FP16 concurrent, ~2.6 GB model weights + ~0.5 GB activation scratch per layer)
- [x] Max RAM ≤ 56 GB — peak ~12 GB (per-token activation buffer for one layer at a time)
- [x] Wall-clock ≤ 4 h — 3-experiment split: Exp A extraction ~3.5 h, Exp B primitive compute ~2 h, Exp C stats ~0.5 h — each ≤ 4 h
- [x] Disk footprint — ~1.5 MB sufficient statistics (see §9); activations NOT persisted
- [x] Quantization logged — FP16 + Q8 per model, logged per-tuple in ledger
- [x] Save-resume path — per-tuple `.npz` for Exp A; per-primitive `.json` for Exp B; summary JSON for Exp C
- [ ] **Smoke test required before full run** (5 sentences × 2 layers × 1 system × 1 resample × 1 quant, end-to-end through Exp A+B+C). Must complete in < 10 min. Prereg blocker.

---

## 15. Sign-off (LOCKED at commit)

**Locked at commit:** `6edf303` (the commit that introduced `code/stimulus_banks.py`, the `code/prereg_validator.py` AST hardening, and this prereg file). All `git_commit=6edf303` pointers in §4 resolve via `git show 6edf303:code/stimulus_banks.py` to the file at that exact commit. This lock-commit step is the follow-up commit anticipated by the original §15 sign-off instruction.

**Post-lock modification rule:** Any change to this prereg after its lock commit invalidates it. A new prereg file with a later date must be created if the design changes.

**Validator verdict required:** `python code/prereg_validator.py research/prereg/genome_id_portability_2026-04-21.md` must exit with code 0 before the probe runs.
