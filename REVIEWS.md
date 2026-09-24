# Frontier Review Log

One entry per hourly review. The review is overdue if `last_review` in
STATE.json is older than ~75 minutes — any agent noticing this should run a
review immediately (3 parallel subagents: interpreter / ideator / critic),
then append an entry here and update STATE.json.

---

## Review 1 — full panel (2026-09-24T11:20Z)

Panel: INTERPRETER + IDEATOR + CRITIC (3 parallel subagents). State reviewed:
E001–E014b, V001/V002/V006, T001–T005.

### INTERPRETER findings (both accepted, amendments applied to THINKING.md)
1. **T005 over-claimed:** rarity = one head of 36 (L5.h1 6.89 bits; others
   4.39 vs L4 4.28); re-broadening +0.13 nats below the script's own spread
   criterion; ×76 = ratio of tiny masses; 'O'-match plausibly a vocative
   artifact of one prompt. Survives: L5 abandons local d1-3 (0.234→0.060).
   → e013 GATED on e013a (200-prompt census).
2. **E014b over-claimed:** renorm never capped write norms; the live
   hypothesis is "damage tracks write/stream allocation" — P3 now evaluated
   and PASSES (renorm arm: write/c and damage have identical rank order,
   ρ=1.0; net rebuilt a 9.3× declining write schedule). → e014c write-clamp
   queued (pre-named in the e014b design memo failure table).
3. **T004 depth-6 is definitionally the L5 argmax flip** — circular with the
   calibrator finding. → e018 causal-depth (patching) queued as the
   construct upgrade.

### IDEATOR harvest (top of 10; full list in agent output, added to queue)
e013a census (gates e013); e019 MLP-5 thermostat (write-scale sweep — direct
causal test of the energy-carrier claim, eval-only minutes); e020 context
surgery (rare token near/far — R1 vs R2); e021 task-swap (front-loading
task-dependence); e018 causal depth; e023 entity-granularity forgetting;
e024 reconsolidation window (bio-analogue); v007 funnel film; v008 anatomy
phylogeny; e026 selection-on-depth (evolution thread).

### CRITIC verdicts (applied)
- Queue drift fixed: duplicate e003b rows merged; e013 ID collision resolved
  (old predict-and-poke → e027); v006 status corrected to DONE.
- PARKED (no live-hypothesis discrimination): e004–e010, e015, e016, and
  e011 refolded into e019 (MLP-5 is the live organ question).
- Next-3 READY: **e013a → e019 → e003b**.
- Replication debt registered: e014b anatomy-plasticity is single-seed
  (e014b.1 queued); e011c bootstrap CIs never run (micro-task queued).
- Process: last 3h ratio drifted to ~55/45 doing/thinking. **Correction: the
  next work slot is THINKING — T006 (anatomical plasticity) must be written
  before any new experiment launches.**

### Decisions
1. THINKING amendments applied (T005 weakened, T003 P3 evaluated+passed,
   T004 circularity noted).
2. Queue rewritten: e013a → e019 → e003b; e013 gated; e014c/e018 promoted;
   stale parked.
3. T006 (plasticity interpretation) is the next unit of work — Rule 0
   correction accepted.

---

## Review 0.5 — bootstrap results check (2026-09-24T10:12Z)
