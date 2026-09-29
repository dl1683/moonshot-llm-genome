# g6 — WALL THE STREAM (design draft v1, 2026-09-29 ~21:05Z)
<!-- SUPERSEDED (R56 ideator, 21:22Z): the registered spec is scratch/r56_ideator.md
     (per-input anchor-TUBE projection supersedes the regularizer/tether mechanics;
     the SPLINT framing + frontier discipline adopted; 2.7M economics variant parked as g6b).
     This file is retained as the lineage record of the mechanics problem (fact-free wash
     => no fact-context activations to project) and the replay-control leg, both carried
     into the registered spec. -->

Status: DESIGN-DRAFT — to be reconciled with R56's ideation report before
registration (anti-collision) and then frozen (bars verbatim) for dispatch.
Builds on: g1/g1b/g1bR (the param ball, WALL-TAXES +0.53), g3/g3R (the
cone — damage is directional/low-dimensional), g5 (the falsifier: the kill
is the composed path q = W_q·LN(h); single-site param walls leave the host
fact dead; fragility is TWO-SITE), T129 (the named composition question),
T132 (clock vs floor decomposition). What is NEW: an activation-level
(function-space) wall — every prior wall is parameter-space. Nothing in
QUEUE/THINKING covers it (checked 21:05Z).

## The question

g1's whole-net L2 ball holds the fact at ~0.9 through the wash that kills
the control in 2 steps — but taxes the organism +0.53 nats and, worse,
pins the parameters to a neighborhood of ONE configuration (the critic's
splint attack, pre-registered here). g5 showed the host fact's fragility
lives on the composed path through h, not on any single parameter site.
g3 showed the damaging displacement is a low-dimensional direction.
Composition: if damage flows through h at a few sites, then constraining
THE FUNCTION at those sites — not the parameters anywhere — should hold
the fact while leaving the organism free to wash-adapt everywhere else.
A constraint on activations defines a MANIFOLD of acceptable parameter
configurations; the ball defines a POINT-NEIGHBORHOOD. The manifold is
the conjecture: wash solutions exist on it (TETHER-CHEAP), and the fact
survives on it (TETHER-HOLDS).

## The mechanism (the shadow-probe tether)

MECHANICS PROBLEM (stated before solution): during a fact-free wash there
are no fact-context activations to project — the wash trains on wash
tokens; the damage is that parameters drift so the fact-path misfires when
the fact next appears. A runtime h-projection is therefore vacuous.
Rejected: hook-projection on matched contexts (never fires under fact-free
wash).

SOLUTION — the tether (zero new parameters, training-time only):
1. ANCHOR (at consolidation, once): forward the fixed fact-battery probe
   set (the install battery contexts, no grad) through the consolidated
   net; record h_site = the residual-stream activations at the chosen
   sites, mean-pooled over the battery. Sites (frozen, first cell): ALL
   residual sites (every layer) at the fact position band. The anchor is
   the same information g1's weight snapshot uses — the comparison is fair
   by information budget.
2. TETHER (each wash step, before the optimizer): forward the same probe
   set (no grad on the wash loss; probe forward is eval-mode, k=1 every
   step, CPU-cheap at 0.87M... see host note); compute
   L_tether = lambda * mean_sites || h_site - h_anchor ||^2 / d_site;
   add L_tether to the wash loss (its gradient DOES flow to parameters —
   the probe contexts are treated as a regularizer batch).
3. THE DIAL: lambda, calibrated not shopped — a 50-step pilot on a
   log grid {0.01, 0.1, 1, 10} x c (c = the scale making the tether
   gradient comparable to the wash loss gradient at step 0; c recorded),
   pick the SMALLEST lambda holding fact p(Z) >= 0.9 at +50; the
   registered run is that lambda, +300 steps, verbatim g1b wash/readout.

## The host

The g1b 2.7M continuity family (stated reason for >1M: direct
comparability with the registered WALL-TAXES +0.53 — the whole point is
the economics delta). Same consolidation, same wash seed family, same
checkpoint cadence and fact battery as g1b/g1bR.

## Registered bars (draft — freeze at registration)

  TETHER-HOLDS: "fires if at the calibrated lambda the fact p(Z) stays
      >= 0.9 at every checkpoint through +300 while the unwalled control
      dies (p(Z) <= 0.05 by +50) — identical bars to g1b."
  TETHER-CHEAP: "fires if the organism's wash-stream NLL at +300 is
      within +0.15 nats of the UNWALLED control's (the param ball's
      registered cost is +0.53) — protection without the splint."
  SPLINT-ONLY: "fires if the fact holds ONLY at lambda where wash NLL at
      +300 is within 0.05 of the pre-wash value (learning stopped) — the
      tether is a ball in disguise; report the full lambda frontier."
  TWO-SITE-CLOSED (the g5 composition clause, report-only first cell):
      "co-report the probe-h displacement at each site over time; if the
      tether holds the fact, the h-displacement at the critical sites is
      the single variable that tracked survival — the junction covers
      both parameter sites at once."

Discriminating observation (T-rule: >=2 explanations separated): if the
fact survives with wash NLL near control, parameter-space locality is
REFUTED as necessary (the manifold suffices); if SPLINT-ONLY fires, the
g1 tax was not the ball's locality but protection itself — equally
informative, and the rhythm (g2) remains the only non-splint mechanism.

## Pre-registered predictions

- P1 (from g3's cone): the tether's needed lambda is SMALL — the damaging
  direction is low-dimensional, so a penalty on the function at the sites
  counters the wash's drift with little force. If lambda must be huge
  (gradient-dominated), the damage was NOT site-local — refutes the
  two-site composition reading.
- P2 (from g5): the site-scan readout will show ONE-or-two critical sites
  carry most of the survival (the composed path's junction sites).
- P3 (from the resurrection economy): at sub-holding lambda, the fact's
  decay under tether will be SLOWER than control's (the tether partially
  counteracts drift) — a dose-response bridge to g2's event threshold.

## Honesty pre-registration

- The tether knows the fact (probe contexts from the install battery) —
  same budget as g1's anchor; stated on every card.
- First cell n=1 (one host draw, one wash seed 10902-family); the
  replicate ladder (wash seeds, then host draws) BEFORE any law-grade
  language — g1bR's precedent.
- The probe forward each step costs compute; if the 180s cap binds at
  2.7M, drop to every-2-steps tether (k=2) and DOCUMENT — never silently.
- Confound to watch: L_tether's gradient also stabilizes the probe
  contexts' predictions directly (a form of replay!). The readout must
  separate "tether-as-replay" from "tether-as-wall": a REPLAY CONTROL leg
  (same probe batch added as ordinary replay data, no sites/anchor —
  g2's mechanism) at matched compute. If replay alone holds the fact,
  g6's finding reduces to g2 and is reported as such (this is the
  killer control the critic would demand, pre-empted).

## Envelope

gpu_ok() double-poll; cooldown(120); <=180s per training run (tether pilot
+ registered run are separate runs); CPU for probe forwards if needed;
no concurrent GPU jobs.
