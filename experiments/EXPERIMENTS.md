# EXPERIMENTS — Neural Genome

*Reverse-chronological log of every experiment. One entry per run. Only Codex-validated conclusions appear here. Raw ledger lives in `ledger.jsonl`.*

Format per entry:
```
## <YYYY-MM-DD> — <experiment-id>
**Purpose.** One line.
**Systems.** Model IDs from the canonical registry.
**Primitive.** Named primitive from MEASUREMENT_PRIMITIVES.md.
**Universality level claimed.** 1 / 2 / 3 / null.
**Commit.** <git sha>
**Result.** What we learned.
**Next.** What this unlocks or blocks.
```

---

## 2026-04-20 — genome_000_scaffold

**Purpose.** Scaffold the moonshot — README, CLAUDE.md, MANIFESTO, UNIVERSALITY_LEVELS, MEASUREMENT_PRIMITIVES, SYSTEM_BESTIARY, OPEN_MYSTERIES, EXPERIMENTS stub.
**Systems.** None (documentation only).
**Primitive.** None.
**Universality level claimed.** null.
**Result.** Project operating manual in place. Atlas empty. Bestiary defined at class level.
**Next.** Phase 1 — primitive architecture-agnosticism gates. First candidate: intrinsic dimension across the Phase-1 minimum viable bestiary.

---

## 2026-04-21 — genome_001_smoke

**Purpose.** First end-to-end Batch-1 smoke test. Verify the instrument pipeline runs; do NOT claim science.
**Systems.** `Qwen/Qwen3-0.6B` (FP16, trained).
**Primitive.** ID (TwoNN + MLE), PR (centered + uncentered), kNN clustering coefficient (k=4).
**Universality level claimed.** null (not an atlas coordinate claim — smoke only).
**Commit.** `344498b`
**Result.** Pipeline runs end-to-end in **6.63 seconds**. 12 atlas rows emitted to `results/smoke/atlas_rows.json` covering 2 sentinel depths (layers 7 and 14 of 28). Expected-garbage numbers because n=5 (TwoNN vs MLE disagree 4x; PR saturates at 1 because rank-4 covariance; clustering NaN because n<k+2). **Smoke criterion per prereg §14 PASSED** (<10 min wall-clock, end-to-end runs, no crashes).
**What this proves.** Instrument works. Sacred outcome S1 moves from "design exists" to "instrument measures something real on a real model." Still no atlas-coordinate claim yet — real Gate-1 requires n ≥ 2000 per G1.6 asymptote rule.
**Next.** Scale to n ≥ 500 with real C4 stream; run full Gate-1 check suite; emit first non-smoke ledger entry.

---

## 2026-04-21 — genome_002_n500_c4

**Purpose.** Scale smoke to n=500 sentences streamed from real `allenai/c4` en. First real primitive measurements.
**Systems.** `Qwen/Qwen3-0.6B` (FP16, trained).
**Primitive.** ID (TwoNN + MLE), PR (centered + uncentered), kNN clustering (k=5 + k=10).
**Universality level claimed.** null (1 system only; cross-architecture comparison is the next step).
**Commit.** `cc3a2ee`
**Result.**
- Wall-clock: **26.7 seconds** (20s C4 streaming + 2s model load + 5s forward+primitives).
- Layer 7 (depth 0.259): TwoNN ID = **23.6**, MLE ID = **18.7**, PR_centered = **8.9**, kNN clustering k=5 = **0.36**.
- Layer 14 (depth 0.519): TwoNN ID = **22.3**, MLE ID = **18.2**, PR_centered = **26.9**, kNN clustering k=5 = **0.34**.
- First scientific observations: (1) PR expands ~3× from layer 7 → 14 (mid-layer capacity expansion); (2) TwoNN and MLE disagree by ~25% (would fail G1.4 at δ_relative=0.10 but pass at δ=0.20); (3) clustering coefficient slightly decreases with depth.
**Next.** Repeat on Mamba2-370M and Falcon-H1-0.5B at matched depths. First cross-architecture comparison = first atlas row that actually tests the universality axiom.

---

## 2026-04-21 — genome_003_cross_arch_pilot

**Purpose.** First cross-CLASS atlas comparison. 2 architectures × 3 sentinel depths × matched stimuli.
**Systems.** `Qwen/Qwen3-0.6B` (Class 1, autoregressive LLM) + `RWKV/rwkv-4-169m-pile` (Class 3, linear-attention recurrent — substituted for state-spaces/mamba2-370m due to Windows mamba-ssm kernel unavailability). Falcon-H1 hybrid OOMed in naive fallback and is deferred.
**Primitive.** ID (TwoNN), PR (centered + uncentered), kNN-5 clustering coefficient.
**Universality level claimed.** null — these are Phase-2 observations pending Gate-1 suite.
**Commit.** `571f5b3` (pre-run) — to be updated post-commit.
**Result (FIRST CROSS-CLASS DATA in the repo):**
- **Intrinsic dimension (TwoNN)** decreases monotonically with depth in both classes: Qwen3 23.6 → 22.3 → 17.9; RWKV 24.7 → 16.8 → 15.3. Same sign/shape, different magnitudes → **Level-2 family-local candidate**.
- **Participation ratio (centered)** is OPPOSITE SIGN across classes: Qwen3 expands 8.9 → 26.9 → 33.4; RWKV compresses 25.1 → 7.8 → 4.9. **NOT a cross-class universal**, not even Level-2 as written. Genuine falsification evidence — this primitive's behavior is architecture-specific.
- **kNN-5 clustering coefficient** agrees across classes AND depths within ~0.05: Qwen3 0.358 / 0.337 / 0.382; RWKV 0.326 / 0.351 / 0.387. **Strongest universality candidate in the atlas to date.** Validates Codex Round 1 Intuition 2 ("global similarity collapses; only local neighborhood structure survives cross-architecture").
**What this proves.** The atlas produces real, interpretable, cross-class signal at 45-second wall-clock on 500 C4 sentences. Sacred outcome S2 (architecture-agnostic instrument) moves from policy-only to empirically-tested at N=2 classes. S7 (manifesto: Intelligence = Geometry) gets first evidence that SOME geometric statistics are class-agnostic and SOME are not — exactly the kind of discrimination the atlas is designed to make.
**Next.** (1) Unblock the hybrid class (Falcon-H1 or substitute Granite-4.0-H) to reach ≥3 classes. (2) Add untrained-twin controls to verify the cross-class agreement isn't architectural coincidence. (3) Run Gate-1 suite per prereg — stimulus-resample stability, quantization stability, n-sweep asymptote — on the 2 working classes to establish whether ID and clustering pass Gate 1.

---

*(Future entries above this line, newest first.)*
