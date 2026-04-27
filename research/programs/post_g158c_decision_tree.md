# Post-g158c decision tree

**Date locked:** 2026-04-27 (before g158c verdict known)
**Purpose:** mechanical verdict-to-action transition for the canonical 3-seed run of context-length inversion. No vibes-based pivots.

## ★ STRATEGIC OVERRIDE 2026-04-27 (cycle 24 Codex direction review)

**The first post-g158c GPU slot is now LOCKED to the annealed-donor / decaying-anchor washout test (g165), regardless of g158c verdict.** This decision overrides the original Path A/B/C structure below for the *next* GPU slot (Path A/B/C still apply for the slot AFTER g165).

**Reason (from `codex_outputs/heartbeats/cycle24_direction_review_20260427T101600.md`):**

- Annealed donor experiment scores PASS=**7.3/10**, FAIL=6.2/10 — higher than Path A (6.8) and Path B (6.4), and unlike Path C (8.2/10) it is NOT hardware-blocked.
- Codex finding: "§0 should replace architecture-prior as the primary research line." The architecture-prior chain (g138-g160) has become a feeder/cash-out branch, not the discovery branch.
- The early-help meta-audit (n=7) made annealed donor a mechanistically motivated test, not a post-hoc rescue. Already-collected data shows 0/7 mechanisms persist; g137 (optimizer-state) is the lone positive-final outlier (+0.08 nats).
- Codex caveat: "**don't run plain anchoring**" (g008 already partly tested static anchoring and washed out). The right test is **decaying anchor / annealed donor schedule**.

**g165 spec (DRAFT):**
- Donor: Qwen3-0.6B (trained); recipient: random-init same-architecture model.
- Anchored regularization: λ(t) = λ_0 · decay_schedule(t), where decay_schedule ∈ {step (drop at step 25), linear (1→0 by step 50), exponential (τ=10), constant (control)}.
- λ_0 ∈ {1.3e-4, 1.3e-3, 1.0e-2}. Compare against scratch (no anchor) baseline. Revised 2026-04-27 after Codex lean pre-flight showed the original grid would collapse all anchored arms into donor-clone behavior.
- PASS: at least one (λ_0, schedule) combination produces final-step NLL advantage > +0.5 nats over scratch with 95% CI excluding zero.
- FAIL: no schedule produces persistent positive advantage; the same washout pattern emerges.
- Compute: ~3-4hr wall, <12 GB VRAM, <24 GB RAM.
- Files (to be created on g158c-completion / launch trigger): `research/prereg/genome_165_annealed_donor_2026-04-27.md`, `code/genome_165_annealed_donor.py`, `results/genome_165_annealed_donor.json`.
- Codex pre-flight required before launch.

**Active queue §0.1 expected uplift before this override:** 5.8/10 (Codex cycle 24).
**After override:** 7.0/10 (g165 first slot, Path A/B/C as fallback).
**If wall-power meter unblocks:** g155 still beats g165 at 8.2 vs 7.3 — meter remains the highest-impact procurement.

## Inputs

g158c verdict is one of: PASS_canonical / WEAK_canonical / PILOT_FRAGILE.

**Locked criteria (carried from g158 PILOT, now over 3 seeds):**

- **PASS_canonical:**
  - Spearman rho(C4 delta vs L) >= +0.8 mean across 3 seeds, AND
  - Delta_256(C4) 95% CI excludes zero AND mean >= +2.0pp, AND
  - Delta_32(C4) sign matches PILOT (negative or zero, never strongly positive)
- **WEAK_canonical:**
  - rho >= +0.5 mean, AND Delta_256 >= +1.0pp, AND Delta_256 CI does not flip sign
  - (everything else but not strong enough to lock)
- **PILOT_FRAGILE:**
  - rho mean drops below +0.5, OR Delta_256 CI crosses zero, OR Delta_32 flips strongly positive (sign inversion was noise)

## g158 PILOT reference (single-seed, seed=42)

- rho_c4 = +1.00 (perfect monotone)
- Delta_256(c4) = +4.10pp (minimal_3L_noMLP > baseline_6L+MLP at L=256)
- Delta_32(c4) = -0.24pp (predicted sign inversion at short L)
- Delta_64 = -0.21pp, Delta_128 = +1.81pp
- LR-selected per arm (baseline 0.0002, minimal 0.0004) at L=128 anchor
- nan_seen=False at all 8 cells
- Verdict text: "PARTIAL_INVERSION: rho_c4=+1.00, rho_ood=+1.00, ..."

The PILOT is unusually clean for a single seed. The canonical question is whether seeds 7 and 13 reproduce.

## Path A: g158c returns PASS_canonical

**Interpretation:** The strongest unique theory prediction (transport-demand monotonicity + short-L sign inversion) survives multi-seed. Theory's input-side prediction is locked at canonical scale.

**Next actions (in order):**
1. Run `python code/integrate_g158c.py --commit` -> appends ledger entry, updates EXPERIMENTS.md, writes WIKI patch.
2. Update `research/CLAIM_EVIDENCE_MAP.md`: promote P13 from PILOT to CANONICAL. New row C17 in confirmed claims.
3. Update WIKI §0.1 score: 6.8 -> 7.2 (per cycle 18 ceiling at PASS).
4. Author `research/THEORY_LOCK_2026-04-27.md`: state that the input-side prediction is locked; the η > δ^mlp mechanism is rejected; the chain now reads: g156 (data-order destruction) + g158c (transport-demand monotonicity) as two independent control axes.
5. **Next experiment LOCKED by Codex direction consult 2026-04-27 (`codex_outputs/heartbeats/post_g158c_design_20260427T090500.md`):**
   **g162 transport-arm capacity sweep** (NOT head ablation — head ablation localizes a toy effect that may not generalize).
   - Setup: noMLP arms with layers `{3, 4, 5, 7}` against `baseline_6L+MLP` at `L ∈ {32, 256}`.
   - Primary statistic: `Δ(L,k) = top1(noMLP_kL) - top1(baseline_6L+MLP)` on C4.
   - **PASS:** at L=256, `rho(k, Δ256) >= +0.8` AND `Δ256(7L) - Δ256(3L) >= +0.8pp`; at L=32, `Δ32(7L) <= Δ32(3L) + 0.1pp`.
   - **FAIL:** `rho(k, Δ256) < +0.3` OR `Δ256(7L) - Δ256(3L) < +0.3pp`.
   - Why highest leverage: turns g158 from a binary pairwise effect into a dose-response architecture intervention. If smooth, closest thing to a design law on this branch. If not smooth, transport story stays a brittle 3L-vs-6L curiosity.
   - Compute: 10 cells (2 contexts × 1 baseline × 4 noMLP depths × 1 seed). ~2.5-3.0h, <4 GB VRAM, <16 GB RAM.
   - Files: `research/prereg/genome_162_transport_arm_capacity_sweep_2026-04-27.md`, `code/genome_162_transport_arm_capacity_sweep.py`, `results/genome_162_transport_arm_capacity_sweep.json`.
   - Codex score: PASS=6.8/10, FAIL=6.1/10.
6. Do NOT rerun g160 (would canonize a null).

## Path B: g158c returns WEAK_canonical

**Interpretation:** Direction is right but signal at canonical scale is softer than PILOT suggested. Pilot was directionally right but slightly amplified.

**Next actions:**
1. Run `python code/integrate_g158c.py --commit`.
2. Update CLAIM_EVIDENCE_MAP: P13 -> WEAK_CANONICAL (new status row, between PILOT and PASS).
3. Update WIKI §0.1 score: 6.8 -> 7.0.
4. **Next experiment LOCKED by Codex direction consult 2026-04-27 — DO NOT pursue g158d budget expansion** (post-hoc, easy to attack as goalpost-moving). Instead:
   **g158e endpoint seed expansion** — same protocol, NEW seeds only, ENDPOINTS only `L ∈ {32, 256}`. Aggregate with existing 3 canonical seeds for N=6.
   - **PASS:** mean Delta_256(c4) >= +1.2pp AND mean Delta_32(c4) <= 0.0pp AND endpoint contrast `C = Delta_256 - Delta_32 >= +1.5pp` AND bootstrap 95% CI(C) excludes 0.
   - **FAIL:** CI(C) crosses 0 OR mean Delta_32 > +0.2pp.
   - Why highest leverage: weak canonical means the live question is variance, not "more train tokens." Endpoint seed expansion keeps the estimand fixed, tests sharpest unique prediction directly, either rescues or kills inversion honestly.
   - Compute: 12 cells (2 contexts × 2 arms × 3 new seeds). ~3.0-3.5h, <4 GB VRAM, <16 GB RAM.
   - Files: `research/prereg/genome_158e_endpoint_seed_expansion_2026-04-27.md`, `code/genome_158e_endpoint_seed_expansion.py`, `results/genome_158e_endpoint_seed_expansion.json`.
   - Codex score: PASS=6.4/10, FAIL=6.0/10.

## Path C: g158c returns PILOT_FRAGILE

**Interpretation:** The PILOT's perfect rho was sampling luck. At canonical scale, transport-demand monotonicity does NOT hold. The theory's input-side prediction joins the η > δ^mlp mechanism in the rejected pile.

**Next actions:**
1. Run `python code/integrate_g158c.py --commit`.
2. Update CLAIM_EVIDENCE_MAP: P13 -> REJECTED (new status row R8b). Mirror R7 (g157) framing.
3. Update WIKI §0.1 score: 6.8 -> ~6.0. Both unique theory predictions (mechanism + input-side) failed canonical confirmation.
4. Author `research/THEORY_REJECTION_2026-04-27.md`: state that the prefix-information transport principle does NOT survive canonical confirmation in either of its derivable predictions.
5. **Next experiment LOCKED by Codex direction consult 2026-04-27 (highest score in entire decision tree):**
   **g155 production distillation + locked C3 TEI/kJ benchmark** (production student fed to the locked benchmark contract).
   - PASS criteria (already locked in `research/prereg/genome_155_edge_benchmark_c3_energy_2026-04-26.md`): `C3_ratio >= 0.90` AND `TEI/kJ vs Qwen3-8B >= 4.0x` AND `TEI/kJ vs best non-distilled sub-2B >= 1.25x` AND no single-dataset gap > 5pp.
   - FAIL: `C3_ratio < 0.85` OR `TEI/kJ < 2.5x` OR student loses to best sub-2B baseline.
   - Why highest leverage: once both unique transport predictions are dead, another small-model mechanism salvage is workshop-grade. The only remaining route on this branch that can still beat the §0.1 competitive bar is an electricity-grade edge result big labs will not publish (conflicts with scale-product narrative).
   - Compute: ~3.5-4.0h wall (student kept in 0.3B-0.6B band), <12 GB VRAM, <24 GB RAM.
   - Files: `research/prereg/genome_155_production_distill_2026-04-27.md` (training half), keep the locked benchmark contract unchanged, `code/genome_155_production_distill.py`, `code/genome_155_edge_benchmark.py`.
   - **HARDWARE PREREQUISITE: external AC wall-power meter must be in hand** (Yokogawa WT310E gold; logging smart plug practical). Without it, this prereg cannot execute honestly. Acquisition is a prerequisite, not a footnote.
   - Codex score: PASS=8.2/10 (BREAKS the §0.1 ceiling that all other paths cap at ~7.0), FAIL=6.7/10.
6. Reframe the project narrative: from "we found a derivable invariant" to "we falsified two derivable invariant candidates honestly + delivered an electricity-grade efficiency demo." Path C with g155 PASS is the ONLY decision-tree branch that exits the workshop-paper trap.

## ★ CYCLE 33 OVERRIDE: g166 deferred; g162 is the FAIL pivot (2026-04-27 14:49)

Codex cycle 33 direction review (`codex_outputs/heartbeats/cycle33_direction_review_20260427T144900.md`) revised the FAIL-pivot ordering after g158c PASS_canonical landed:

- **If g165 FAILs, do NOT auto-run g166.** Move g162 (transport-arm capacity sweep) ahead of g166. g166 is draft-only, higher-envelope, and its own prereg flagged that the required donor optimizer state may not exist yet (no Qwen3-0.6B optimizer-state checkpoint preserved).
- **g162 PASS=6.8/10**, g166 estimated **~6.4/10**. With g158c PASS_canonical, g162 cleanly extends the just-PASSED theory into a dose-response architecture intervention.
- Wall-power-meter procurement remains the parallel priority unblocking g155 (8.2/10).

**Updated FAIL-pivot order:** g165 FAIL → g162 (capacity sweep) → wall-power-meter procurement → g166 (only if a meaningful donor-optimizer-state checkpoint exists).

## Codex direction consult — fired 2026-04-27 09:05, completed 2026-04-27

**Output:** `codex_outputs/heartbeats/post_g158c_design_20260427T090500.md`

**Adjudication summary:**
- Path A: g162 transport-arm capacity sweep (NOT head ablation). PASS=6.8/10.
- Path B: g158e endpoint seed expansion (NOT g158d budget expansion). PASS=6.4/10.
- Path C: g155 production+benchmark. PASS=8.2/10. **HARDWARE-BLOCKED on wall-power meter.**

**One experiment to launch RIGHT NOW that does NOT depend on g158c verdict:** g155 production distillation feeding the locked C3-TEI/kJ benchmark, **assuming the wall-power meter is already available**. Currently it is not (per WIKI §pre-staged + research/prereg/genome_155_edge_benchmark_c3_energy_2026-04-26.md §146). Acquire the meter — that is the highest-impact lift across the entire decision tree.

## Codex audit consult — fired 2026-04-27 09:05, completed 2026-04-27

**Output:** `codex_outputs/heartbeats/g158c_audit_20260427T090500.md`

**SEV findings applied 2026-04-27:**
- SEV8 #1 (integrate_g158c.py over-permissive d32 threshold): FIXED — `mean_d32 <= 0.0pp` (was `<= 0.5pp`).
- SEV8 #2 (no incremental checkpoint): DEFERRED — in-flight run cannot be retro-fitted; future canonicals must add per-seed checkpoint.
- SEV6 #1 (single C4 corpus across seeds): DOCUMENTED in canonical prereg + WIKI patch.
- SEV6 #2 (lr_select_L=128 metadata false): TO PATCH for future re-runs (current run unaffected by metadata bug).
- SEV5 #1 (CUDA non-deterministic): DOCUMENTED.
- SEV5 #2 (canonical prereg missing): FIXED — `research/prereg/genome_158c_3seed_canonical_2026-04-27.md` LOCKED 2026-04-27.

## Codex pre-flight prompt template (for the integrity audit AFTER verdict lands)

When g158c verdict lands, use this prompt for the integrity audit Codex consult:

```
Read CLAUDE.md, COMPUTE.md, research/CLAIM_EVIDENCE_MAP.md,
research/programs/post_g158c_decision_tree.md, the prereg for g158
(research/prereg/genome_158_*.md), and the verdict at
results/genome_158c_3seed_canonical.json.

Question 1 (Research Integrity Auditor): does the verdict text in the
JSON honestly reflect the data? Are the criteria from the decision tree
applied correctly? Any p-hacking risk (e.g., post-hoc seed selection)?

Question 2 (Architecture Theorist): conditional on the verdict, what is
the highest-leverage NEXT experiment that pursues a finding a big lab
would not publish? Score Nobel/Turing/Fields out of 10.

Output a verdict: PASS_canonical / WEAK_canonical / PILOT_FRAGILE
(matching decision tree), the integrity finding, and the next
experiment recommendation in a single page.
```

## Notes on path framing

The cycle 21 direction review framed the §0.1 ceiling at PASS as 7.0/10 (not 7.5+) because:
- g156 + g158c PASS gives **two control axes**, not three.
- g157 mechanism is dead -> no internal measurable mechanism.
- g160 PILOT_KILL closed the design-rule cash-out.
- g159 was supportive null, not positive cross-class confirmation.

So even on the best path (PASS_canonical), the framing is "strong directional theory support, two confirmed control axes" — NOT a validated model-selection law. The bar to claim a model-selection law would require either g160c PASS (after a different theoretical motivation justifies rerunning) or a different design-rule experiment.
