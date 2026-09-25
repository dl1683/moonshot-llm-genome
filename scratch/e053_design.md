# E053 design — The cache utility timeline (frontier candidate A, T025)

**Question:** when a cached K/V entry becomes dead weight during generation —
the per-position *causal* utility curve over cache age. Frontier scan
(scratch/frontier_research_20260925.md): sinks are universal (Gu ICLR-2025,
KVSink COLM-2025) but nobody has published a per-position causal utility
timeline at any scale. T025 registered candidate A with the prior "utility
collapses onto sink + recency". Today's CPU probes (below) already sharpen
that prior: at our scale the sink is **not** generation-relevant, so the live
question is the *onset age of dead weight* and whether old entries are merely
dead or actively harmful (dilution).

---

## 1. Measured probes (2026-09-25, CPU, this design session)

All numbers from `e005s_small.pt` (0.84M, 4L/4H/128d), `e001.pt` (2.7M,
6L/6H/192d), `e005s_large.pt` (~10M, 8L/8H/320d) — all trained at
block_size=256 (wpe is 256 everywhere: **a hard constraint, see §4**).
Method: exact manual-attention reimplementation; verified against SDPA
(max softmax-prob deviation 9.6e-06 / 6.7e-06 / 3.6e-06 — G0 passed for all
three). Teacher-forced on 16 val sequences, ctx 256.

### P-A. Sink probe — does the net form a sink? (YES weakly; NO for generation)

Attention mass on position 0, averaged over queries 1..255 / heads / seqs
(uniform = 0.39%):

| model | layer range | mean mass@0 | best layer | max single head | mass@0 from LAST query |
|---|---|---|---|---|---|
| 0.84M | L0-L3 | 1.0-2.3% | L0 2.29% | 4.8% (L1) | 0.00-0.23% |
| 2.7M  | L0-L5 | 1.0-3.0% | L5 2.95% | 5.2% (L1) | 0.00-0.32% |
| 10M   | L0-L7 | 0.9-2.3% | L7 2.34% | 4.8% (L1) | 0.00-0.32% |

First-4-position mass 2.5-7.4% (vs 1.6% uniform) — a mild prefix preference.
**Reading:** a weak, head-concentrated average sink exists in all three nets
(3-8x over uniform, consistent with Gu et al.'s claim that sinks appear by
~14M params — our 10M is just under that and it shows only the precursor).
BUT the *generating* position puts ~0% on position 0 at every layer — the
sink mass is carried by early query rows (which have almost nothing else to
attend to). A 512-token exploratory forward was skipped as meaningless:
wpe=256 makes positions 256+ out-of-distribution for every checkpoint we own.

### P-B. K-norm story — no position-0 outlier at our scale

K-norms: position 0 is *not* an outlier anywhere (pos0 ≈ median, e.g. 11.80
vs 11.28 at L1 of 0.84M). What exists: layer-1 norms ~5-6x layer-0, and a
mild **recency-side** growth (last-8 positions ~1.2-1.6x the position
median). The big-model "massive K outlier at the sink" picture does not hold
at ≤10M — another reason e053 must measure utility causally, not via norms
(honesty reflex: logits/norms alone did not predict it).

### P-C. Causal bin smoke (teacher-forced, 0.84M, next-token CE delta, n=16)

Baseline last-token CE 2.0999. V-zero at all layers (K-drop in parens):

| bin | total dCE | per-position dCE |
|---|---|---|
| pos 0 (sink) | **+0.0033** (NaN*, guarded) | 0.0033 |
| p1-16 | +0.054 (+0.083) | 0.0034 |
| p17-64 | +0.087 (+0.089) | 0.0018 |
| p65-128 | +0.117 (+0.179) | 0.0018 |
| p129-255 | **+1.92 (+1.60)** | 0.0151 |

*NaN: dropping K of pos 0 leaves query row 0 with zero visible keys → all
-inf softmax. Fix registered as G3 (K-drop keeps the self-attention diagonal).
Last-query attention mass: 92.8% on p129-255, 0.06% on the sink.
**Shape:** monotone recency cliff (+8x per-position in the last half) with a
weak primacy bump (p1-16 ≥ p17-64 per-position). Sink ≈ causally dead for
next-token prediction at full context.

### P-D. Free-run micro-timeline (n=1 sequence, fixed anchor, preview only)

64-token prompt, generate to 184; sink = original position 0; dCE = V-zero@0
(all layers) on the actually-sampled next token:

T=65: +4.51 → T=85: +0.39 → T=105: −0.12 → T=125: −0.10 → T=165: −0.03 →
T=184: −0.006. Last-query sink mass decays 0.52% → 0.10%.

**Sink utility DECAYS with age (from large-but-just-"early-context" to
slightly NEGATIVE = lesion helps), contradicting the sink-as-accumulator
intuition at this scale.** Recency-8 lesion stays strongly positive
(+0.13..+10.9). High variance (single sequence; one CE=0.009 easy step) —
this is the prior, not the result.

### P-E. Cost calibration (CPU, 8 threads, B=1, T=256)

Forward: 4ms (0.84M) / 9ms (2.7M) / 26ms (10M); manual+lesion ≈ +1-2ms.
Full per-step protocol (1 baseline + 7 bin lesions ≈ 8 forwards) + final
256-position sweep (256 forwards) ≈ 1800 forwards/sequence → **11s (0.84M),
16s (2.7M), 47s (10M) per 256-token sequence.** Everything below is
CPU-feasible with >2x margin.

---

## 2. E053 Phase 1 — the timeline (CPU-only, main experiment)

**Models:** e005s_small (primary), e001, e005s_large (scale axis);
e048_direct400 / e048_direct800 / e001@4000 (training-exposure axis, 2.7M
arch — lineage caveat: different runs, same arch/data).

**Data:** 8 fixed val prompts of 64 tokens, seed 202. Free-run generation to
T=256 with FIXED anchor (no sliding window — position 0 identity constant,
matching non-windowed KV-cache semantics). Sampling temp 0.8, top-k 40,
seeded.

**Interventions (exact, in manual attention):**
- `V-zero(pos)` — wipe cached content at all layers (primary);
- `K-drop(pos)` — remove addressability (softmax column −inf, diagonal
  preserved, G3);
- `both` — entry deletion (equivalence check: K-drop+V-zero ≈ K-drop).

**Measurements per sequence:**
1. **Per-step binned timeline:** at every generation step, dCE of the
   actually-sampled next token for bins {p0, 1-16, 17-64, 65-128, 129-192,
   193-255} → age×step utility surface. (The brief's 257-1k / 1k+ bins are
   unreachable in Phase 1 — wpe=256; see §4 Phase 2.)
2. **Full per-position profile at T=256:** every position lesioned
   individually (256 forwards, ~1.5s) → the actual unpublished curve, finer
   than bins, incl. the primacy bump.
3. **Static teacher-forced control:** same bins on 64 real-text windows —
   does self-generated context age differently from real context?
4. **Derived:** onset age a* = youngest cache age where mean per-position
   dCE < 0.01 nats; junk fraction = share of positions with dCE ≤ −0.01
   (lesion HELPS); per-layer decomposition (V-zero at L0..L3 only) on the
   final step — where in depth does old-context utility live?

**Aggregate:** mean ± bootstrap CI over 8 sequences per model; onset age per
model on the exposure and scale axes.

## 3. Registered predictions (before any Phase-1 run)

- **P1 — collapse shape.** Per-position utility at T=256 is monotone in
  recency + weak primacy bump (bump ≤ 25% of recent plateau); sink dCE
  < 0.05 nats at age > 100 in ≥ 3 models (sink dead for generation).
  *Alternative kept alive:* sink dCE ≥ 0.3 nats (load-bearing sink, Gu et
  al. big-model picture) — would mean emergence between 10M and 14M.
- **P2 — dead-weight onset.** a* as a fraction of context is scale-INVARIANT
  (0.84M / 2.7M / 10M within ±20%) but grows with training exposure
  (400 → 4000 steps) by ≥ 1.5x in absolute tokens: trained models keep old
  entries useful longer. (Context-length axis = Phase 2; registered there:
  a*/T shrinks as T exceeds the training window — dead weight onset is
  where extrapolation begins.)
- **P3 — sink accumulator (three-way, preview favors decay).** Sink V-zero
  dCE trajectory over generation: GROW / FLAT (|slope| < 10% of initial) /
  DECAY (to < 20% of initial by mid-generation). Micro-preview says DECAY
  with a sign flip to ≤ 0 — the sink is not an accumulator at ≤10M; mass is
  an early-row artifact, utility is just "earliest = oldest useful context"
  until it dies.
- **P4 — junk is mildly harmful.** Among positions older than a*, ≥ 5% have
  dCE ≤ −0.01 (removing them IMPROVES next-token CE) — dead weight is
  dilution, quantified causally. If ~0% negative, old entries are dead but
  harmless (pure evictability).

## 4. Phase 2 (optional, GPU only when free + `gpu_ok()`)

Train one fresh 0.84M at block_size=1024 (params: 4L/128d, wpe 1024×128 →
0.94M total — inside the ≤1M default envelope; ~3000 steps, single ≤30-min
launch, thermal guard, then idle-cool) to test the context-length axis of P2
and extend bins to 257-1k / 1k+. NOT part of Phase-1 budget; separate file
decision at review time. (Candidate B/e054 context-rot shares this model —
train once, use twice.)

## 5. Budget, gates, envelope

- **Budget:** Phase 1 ≈ 8 seq × (11+16+47)s ≈ 10 min + exposure-axis 3×16s
  + static controls ≈ **≤ 12 min wall, 100% CPU.** Fallback if slow: 4
  sequences, every-2nd-step timeline, 10M per-position sweep binned.
- **G0** manual ≡ SDPA (max prob dev < 1e-4) — already verified, re-check in
  file. **G1** ckpt val CE within known anchors (e005s_small 1.5581±0.03).
  **G2** recency-bin V-zero dCE > +0.5 at every step (machinery sanity).
  **G3** K-drop preserves the diagonal (no NaN); any NaN → skip + log.
  **G4** Phase 1 touches no GPU; no training; no new automations.
- **Outputs:** `runs/e053/metrics.json` + `runs/e053/e053_utility_timeline.png`
  (age×step dCE heatmap + per-position curve + onset-age vs scale/exposure);
  NOTES.md entry + THINKING interpretation before anything builds on it.
- **Honesty caveats:** all models are char-level, ctx-256-trained — ages
  >255 and token-level sinks are out of scope until Phase 2; last-token CE
  on sampled tokens is high-variance (n=8, CI reported); free-run n=8 cannot
  claim per-position significance, only per-bin (hence bins are the
  registered unit, per-position curves are descriptive).

**Implementation:** one file, `lab/e053_cache_utility_timeline.py`, reusing
`common.py` model/corpus; manual-attention forward defined in-file (hook-free,
exact); seeds fixed (202 prompts, 7 sampling).
