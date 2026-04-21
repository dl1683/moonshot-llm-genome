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

*(Future entries above this line, newest first.)*
