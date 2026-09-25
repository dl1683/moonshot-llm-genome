# Day Two Report — Consolidation, Scale, and the Edit Law

**Neural Dissection Lab · 2026-09-25 · continues DAY_ONE_REPORT.md ·
~15 further experiments (e012d-e049, e033), 3 reviews (R6-R8+patrols),
thinking entries T014-T023, under a new compute envelope.**

*Day one earned laws; day two tried to break them — and mostly did, in the
most useful way possible.*

---

## 1. What day two did

Day one ended with a card of seven laws and a suspicion (Review 4) that
its foundation instrument was circular. Day two was consolidation under
adversity: audit the instrument, replicate the claims, scale the laws, and
finish the edit doctrine — while absorbing a hardware shutdown that
permanently changed how the lab computes.

## 2. The instrument audit (T012, T014)

The argmax-stability "decision depth" lens — the construct that anchored
day one's stage-invariance claim — was tested causally by activation
patching and found **per-position uncorrelated with causal depth**
(Spearman −0.009; the lens's dominant bin selects causally random
positions). The 4-net causal census then showed **causal depth is NOT
cross-net invariant** (seed axis 0.735, regime axis 0.500 — the lens's
0.82-0.85 "invariance" was a shared-bias artifact). What survives:
**a causal gate exists mid-stack in every net** (suffix-monotone flip
curves; 15→7% distributed decisions as depth capacity grows), at a depth
the anatomy's history chooses. Instrument lesson recorded: two biased
instruments agreeing is not validation.

## 3. The replication sweep (T016, T017)

Under the new min-nets rule (positives at H require ≥3 nets):
- **PROMOTED to H (5/5 nets each):** the L5-calibrator (KL 0.91-1.08 at
  ≤+0.046 ablation cost everywhere) and the MLP-5 energy carrier
  (zero/rotate 0.25-0.34, graceful α-scaling in all cells).
- **DEMOTED to distributional:** the two-factor surgical erasure (the
  "one body head" is net-specific — B43's top residual head is L4H4,
  BDO's L3H1; neither theirs nor B's recipe erases on the other nets) and
  the shared-L0 name machine (the L0 BLOCK is top-1 in 20/20 cells; the
  attn-vs-MLP sublayer split is a lineage lottery).

## 4. The scale capstone (T020; steps-confound flagged)

0.84M / 2.7M / 10M, frozen readouts: front-loading HOLDS and intensifies
(attn-L0/last: 11× → 72× → 116×; at 10M the three deepest attention
blocks cost ≤0.04 nats each); address surgery is scale-invariant in shape
(S_name 1293/332, class-exact, +0.0005 nats); 16-token sufficiency erodes
monotonically with scale (far-value −0.001 → +0.035); the causal gate
exists at 8L (mode 2/8) but its relative depth slides with architecture.
Direction survives; the multipliers carry a steps-confound caveat
(4000/2226/1086 steps anti-correlated with scale).

## 5. The retrieval threshold (T021)

Refrain-density corpora map the boundary where "no far retrieval on
natural data" breaks: a refrain-density threshold (≤5% own-probe; 5–20% shared-probe)
flips far context from net interference (−2.24 nats; R8-flagged partly
net-quality artifact) to net retrieval — graded, not sharp;
compartmentalized (zero leak into ordinary text at any density); retrieval
heads form DISCRETELY, always in the late-attention slot (L5 at 6 layers,
L7 at 8). Day one's T007 "far context can mislead" population was the
negative side of this same tug-of-war.

## 6. The edit law completed (T015, T018, T019)

- **Expression is teacher-forcing-bound:** installed-but-silent knowledge
  stays silent at every dose (4×), temperature, and seeding — 92-96%
  battery accuracy with ZERO free-generation occurrences; the installed
  address is geometry-bound (p collapses 6× with 10 context chars
  deleted) and sub-argmax. The day's single expression event came from
  naturally-trained free contexts.
- **Scar tissue:** re-learning after erasure is 2.08× SLOWER than fresh
  install, but the burned address regrows along its ORIGINAL direction
  (cos 0.760 vs fresh 0.278) — **erasure burns the address, not the
  attractor.** The new route is genuinely new (atlas ρ 0.21; the old
  carrier head flips to anti-carrier) and ~3× more resistant to the
  original surgical key. [n=1, flagged]
- **The law, final:** ADDRESS (concentrated, removable) / ABILITY
  (distributed, train-only) / EXPRESSION (needs free-shaped exposure) /
  HISTORY (re-learned ≠ original).

## 7. The write-equalizer (T023)

Forcing uniform MLP write norms: parity passes (1.5147 vs 1.537; n=1, no CI),
attention untouched, calibrator intact, MLP damage flattens and rises.
**The energy carrier is real but the schedule is decorative** — the
network defends its coarse allocation, not its write norms. (0.84M-scoped.)

## 8. The compute envelope (permanent)

A mid-experiment system shutdown (GPU power/thermal; 76°C at idle) bought
permanent rules: models ≤1M default / 5M ceiling; ≥15% GPU headroom;
thermal guards in the harness (gpu_status/gpu_ok/cooldown); no concurrent
GPU jobs; serial runs with cooldown blocks; fast-small-model burst
throttling (88°C observed at batch 32 — speed is also heat).

## 9. Process ledger

Day two caught: one foundation instrument (the lens), one false-positive
verdict (a cosine of a near-zero vector), one never-executed run flagged
"confirmed" (a smoke env-var), one stale-resume crash chain, one
checkpoint-loading bug, and — after two silent ledger losses — formalized
the integrity rules (verify-in-place; never trust a commit message) and
added push discipline (112-commit backlog synced). The meta-law held all
day: every positive that died was n=1; every survivor was replicated.

## 10. Open at close of day two

e040 (is init-anchoring evolvable under lineage selection?) — RUNNING,
registered P1-P3 (T022). The 2.7M write-equalizer (schedule-necessity at
depth). Content-vs-position binding in the expression gap. The e046-grade
replication of the scar findings. Whether the day-one report's five open
edges are now three.

---

*Artifacts: runs/e012d-e049 + v011/v012, THINKING.md T014-T023, REVIEWS
R6-R11. Everything pushed. The heartbeat continues.*
