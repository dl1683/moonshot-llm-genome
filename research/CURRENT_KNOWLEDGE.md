# Current Knowledge — Neural Genome Moonshot

*Living snapshot of what we actually know, as of 2026-04-21 session end. Reread before any vision-planning or next-experiment decision. Structured so a fresh agent or Codex review can absorb 38 ledger entries in 5 minutes.*

---

## 1. The coordinate and its status

The primitive is the **mean local clustering coefficient** `C(X, k)` of the Euclidean `k`-nearest-neighbor graph on pooled hidden-state point clouds. Across 9 trained neural networks (text + vision), the scaling `C(X, k) = c_0 · k^p` holds with `R² > 0.989` (mean 0.997) in log-log space across k ∈ {3, 5, 8, 12, 18, 27, 40, 60, 90, 130}.

**Gate status** (from paper Table 5 + latest data):

| Gate | Meaning | Status as of 2026-04-21 |
|---|---|---|
| G1.2 | Rotation / isotropic-scale invariance | PASS by construction, empirically verified |
| G1.3 | Stimulus-resample stability | PASS 7/8 classes at n=2000; Falcon tips at n=4000 |
| G1.5 | Quantization stability FP16↔Q8 | PASS 4/4 text at δ=0.05 |
| G1.7 | Not a random-geometry artifact | PASS 4–7× above Gaussian baseline |
| G2.3 | Functional form identification | v1 FALSIFIED; v2 framework D (rate-distortion) SUPPORTED on text, TEXT-LOCAL on vision |
| G2.4 | Causal ablation | PASS 3/3 text (7.8–443% loss lift, 20–66× specific); DINOv2 method-limit |
| G2.5 | Biological instantiation | PASS 10/10 Allen V1 sessions at δ=0.10; 8/10 at δ=0.05 |

Only G2.3 is partial. Level-1 universality is gated on a modality-universal (not modality-stratified) derivation.

---

## 2. The three strongest distinctive findings

### 2a. Training-convergence (genome_028, 029, 030, 031, 032, 033)

Random-init twins of the same architectures produce exponents spanning **`p ∈ [0, 0.37]` (22× wider)** than trained. Trained text systems cluster at `p ∈ [0.154, 0.171]`, trained vision at `p ∈ [0.210, 0.223]`. Per-system cross-seed CV uniformly < 4%. Training is a **convergence operation** toward modality-specific fixed points. On Qwen3 specifically, random-init doesn't even produce a power law (R² < 0.04 across 3 seeds) — training *creates* the log-linearity, not just the exponent.

### 2b. Trained-manifold invariant `c = p × d_rd` (genome_036, 037, 038, 039)

Rate-distortion dimension `d_rd` (k-means scaling at log-spaced K) obeys `p = c / d_rd` with:

- Text (n=4 systems): c = **2.07 ± 0.23** (11% CV)
- Vision (n=3 systems): c = **3.18 ± 0.69** (22% CV)

Training compresses `d_rd` by 3× (DeepSeek untrained d_rd = 42.90 → trained 14.06) and simultaneously establishes the `c = p × d_rd` identity. The relation **breaks on untrained** manifolds (DeepSeek untrained rel_err 87%). This is a trained-manifold geometric invariant, not a general identity.

### 2c. Biology bridge (genome_027, 034)

10 Allen Brain Observatory Visual Coding Neuropixels sessions × 200 cortical units, Natural Movie One: kNN-10 mean **0.333 ± 0.067**. 10/10 pass at δ=0.10 tolerance vs DINOv2 band [0.30, 0.35] (100%); 8/10 at strict δ=0.05 (80%). Prereg criterion (≥60%) cleared with 40-point and 20-point margin respectively. Mouse V1 and DINOv2 produce statistically-indistinguishable kNN-10 values at the pre-registered tolerance.

---

## 3. Derivation candidate status

Four v2 derivation sketches were attempted this session:

| Framework | Relation | Verdict | Detail |
|---|---|---|---|
| A | `p = d_2/d_int − 1` (fractal gap) | **FALSIFIED** (genome_024) | Structurally non-positive; all 3 systems wrong sign |
| B | `p = (h − d_db)/d_db` (doubling-dim) | **FALSIFIED** (genome_026) | Magnitude absurd (60–90 vs observed 0.17) |
| C | `p = (3-α)/(α-1)` (heavy-tailed NN-degree) | **FALSIFIED** (genome_020) | Wrong sign (α ≈ 3.8 gives p ≈ -0.28) |
| D | `p = c/d_rd` (rate-distortion) | **SUPPORTED on text, TEXT-LOCAL on vision** (genome_036/037) | c ≈ 2 text, c ≈ 3 vision; modality-stratified |

Framework D is the only quantitatively predictive candidate. The **integers 2 and 3 are suggestive** — they look like they should come from something (stimulus intrinsic dim? rate-distortion prefactor?) but we have not derived them from first principles.

---

## 4. What we have NOT done (honestly)

The following are explicitly missing. Any one of them would be the breakthrough-bar per the CLAUDE.md §0.1 competitive-reality framing:

1. **First-principles derivation of c=2 and c=3.** Why those specific integers? No theoretical argument yet.
2. **Electricity-grade efficiency demo.** We have a correlation between R²(Q8) and NLL(Q4), and a blind-test-validated decision rule — but no actual training of a model to match baseline capability at 10× less compute using geometry targets.
3. **Geometry transfusion.** Take a random-init Qwen3, inject trained geometric structure, test capability. Not done.
4. **Biology-vs-ANN derivation.** The empirical equivalence is strong (10/10 sessions). Why it holds theoretically is undefined.
5. **Dynamics/Koopman fork.** Static point-cloud geometry may be a shadow of operator-level invariants. Not tested.
6. **G2.4 on vision.** DINOv2 pooled-delta-add ablation inverted; CLS-only perturbation required. Not attempted.
7. **Hybrid architecture untrained twin.** Falcon-H1 hits trust_remote_code + geqrf blockers on Windows; not resolved.

---

## 5. The paper vs the moonshot

**Paper state** (8914 words, 4 figures, 11 tables): workshop-submittable, rigorous, honest. Reports Gate-1 + G2.4-text + G2.5-biology passing, framework D SUPPORTED-on-text, training-convergence negative control, modality-stratified invariant. Pre-registration discipline throughout.

**Honest assessment vs the moonshot**: the paper is a respectable professional artifact. It is NOT the field-shifting contribution an independent researcher needs to stand out against DeepMind/Anthropic. To become field-shifting, one of items 1–5 in §4 needs to land with a clean result. Paper polish beyond this state has near-zero marginal value per the competitive-reality framing.

---

## 6. Compute + reproducibility

- Single RTX 5090 laptop, ≤22 GB VRAM, ≤56 GB RAM, ≤4 h per experiment.
- Every ledger entry reproducible from commit hash + config path.
- 38 ledger entries span smoke (genome_001) through framework D text-local (genome_037 + cross-modality genome_039).
- Next agent on this repo can re-run any result via `python code/genome_*.py`.

---

## 7. The open vision question

This document describes **what is** as of 2026-04-21. The **long-term vision** — what this moonshot looks like when it is done, and the ladder from here to there — is being asked of a fresh Codex session (fired at 2026-04-21 22:14, output at `.codex/outputs/long_term_vision_2026-04-21-2213.md`). When that verdict lands it will be integrated here and into `research/MANIFESTO.md`.

Until then: the paper is shippable, the atlas instrument is mature, and the critical open question is whether we can do genuine first-principles work (derivation, electricity-grade efficiency demo, geometry transfusion, biology-ANN bridge theory) that separates us from "another professional representational-geometry paper." The user's directive this session: aim for exceptional, not respectable.
