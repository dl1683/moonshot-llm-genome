# Post-g158c decision tree

**Date locked:** 2026-04-27 (before g158c verdict known)
**Purpose:** mechanical verdict-to-action transition for the canonical 3-seed run of context-length inversion. No vibes-based pivots.

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
1. Run `python code/integrate_g158c.py --commit` (helper to be written) -> appends ledger entry, updates EXPERIMENTS.md, writes WIKI patch.
2. Update `research/CLAIM_EVIDENCE_MAP.md`: promote P13 from PILOT to CANONICAL. New row C17 in confirmed claims.
3. Update WIKI §0.1 score: 6.8 -> 7.2 (per cycle 18 ceiling at PASS).
4. Author `research/THEORY_LOCK_2026-04-27.md`: state that the input-side prediction is locked; the η > δ^mlp mechanism is rejected; the chain now reads: g156 (data-order destruction) + g158c (transport-demand monotonicity) as two independent control axes.
5. **Next experiment decision:** highest-leverage move is now a CAUSAL mechanism probe distinct from η/δ. Candidates:
   - **g162 transport-arm capacity sweep:** does the inversion persist at intermediate transport capacities (4L, 5L, 7L noMLP)? Tests whether the effect is smooth in transport budget (theory predicts yes).
   - **g163 attention-head ablation in minimal_3L_noMLP:** which attention heads carry the transport signal? Causal head-level test.
   - Pick via Codex direction review at the next consult cycle.
6. Do NOT rerun g160 unless g162 + g163 both PASS (would only canonize a null).

## Path B: g158c returns WEAK_canonical

**Interpretation:** Direction is right but signal at canonical scale is softer than PILOT suggested. Pilot was directionally right but slightly amplified.

**Next actions:**
1. Run `python code/integrate_g158c.py --commit`.
2. Update CLAIM_EVIDENCE_MAP: P13 -> WEAK_CANONICAL (new status row, between PILOT and PASS).
3. Update WIKI §0.1 score: 6.8 -> 7.0.
4. **Probe-budget consideration:** g158 used n_train_256=32768. At larger N the signal might sharpen. BUT this is post-hoc — only consider if Codex direction review at next consult signs off as not p-hacking.
5. **Next experiment decision:** g158d budget-expansion (matched-FLOPs at 2x training budget) is the cheapest move that would either lock or kill the WEAK signal.

## Path C: g158c returns PILOT_FRAGILE

**Interpretation:** The PILOT's perfect rho was sampling luck. At canonical scale, transport-demand monotonicity does NOT hold. The theory's input-side prediction joins the η > δ^mlp mechanism in the rejected pile.

**Next actions:**
1. Run `python code/integrate_g158c.py --commit`.
2. Update CLAIM_EVIDENCE_MAP: P13 -> REJECTED (new status row R8). Mirror R7 (g157) framing.
3. Update WIKI §0.1 score: 6.8 -> ~6.0. Both unique theory predictions (mechanism + input-side) failed canonical confirmation.
4. Author `research/THEORY_REJECTION_2026-04-27.md`: state that the prefix-information transport principle does NOT survive canonical confirmation in either of its derivable predictions. The g156 PASS stands as a narrow empirical observation; the derivation chain is dead.
5. **Pivot decision:** abandon transport theory entirely. Manifesto-aligned cash-out paths:
   - Distillation product: g160 PILOT_KILL was inconclusive at -0.34pp; consider g160c as a 3-seed CANONICAL only if a different theoretical motivation comes up. Otherwise retire.
   - g159 cross-class lesion: completed as INCOMPLETE/SCALE-LIMITED. Retire.
   - g161 RWKV: still hardware-blocked.
   - **g155 edge benchmark:** if hardware available, is the most manifesto-aligned remaining direction (even without the transport theory).
   - **NEW direction needed:** Codex consult cycle at this point should ask: "given both transport predictions failed canonical confirmation, what is the highest-leverage breakthrough-aligned direction this researcher can still pursue with ≤4h compute envelope?"
6. Reframe the project narrative: from "we found a derivable invariant" to "we falsified two derivable invariant candidates honestly." This is still publishable as a falsification result but not as a manifesto-grade breakthrough.

## Codex pre-flight prompt template

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
