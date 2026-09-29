# g3K — THE KAPPA CELL (design draft v1, 2026-09-29 ~21:22Z)

Status: DESIGN-DRAFT (ripening; W022's named cell). One ripening cycle
before registration. Builds on: W022 (the alignment unification), R56
critic evidence (scratch/r56_critic.md — the store's isotropic threshold
~24-32x and the tilt ladder), g3/g3R (the store organism + lambda-sweep
machinery, eval-only conventions), e185/e187 (the host fact dies to
isotropic noise at displacement-match, n=3 arms). What is NEW: the two
kappas read SIDE BY SIDE on ONE organism under ONE displacement
convention — every prior reading is single-ruler or cross-convention.

## The question

W022's fork, stated as a measurement: kappa = (isotropic kill rung) /
(wash kill rung) at matched MEAN per-coordinate RMS (total L2 over
sqrt(N) equal; the wash's RMS may be unevenly distributed — that
unevenness is itself a reading, see confounds). Read kappa TWICE on the
same saved g3 organism: kappa_host (the host fact's battery ruler) and
kappa_store (the store's g0 ruler).

- KAPPA-SPLIT (kappa_host ~1, kappa_store >= 8): directional immunity
  is PURCHASABLE — attractor readout forgives random displacement while
  the distributed host fact never could. The fourth architectural
  claim's candidate, and the rescoped cone's redemption: the wide cone
  is the store's design win, not a measurement artifact.
- NO-BASIN-UNIVERSAL (both ~1): e185's no-basin generalizes to the
  store; the critic's 24-32x was a convention artifact (the wash's RMS
  concentration vs isotropic's even spread — the store coords receive
  less effective displacement under matched total L2 than under matched
  per-coordinate RMS; the two conventions disagree, and the disagreement
  ITSELF localizes where the wash's energy lives).
- MIXED (store 2-8x): graded immunity; report the pair, claim nothing
  larger.

## The cell (eval-only — no training, CPU-friendly, can fill GPU gaps)

Saved organism: the g3/g3R checkpoint (provenance gates as in g3R).
Arms: direction in {wash (the stored full-wash unit direction),
isotropic (fresh Gaussian draws, n=3 seeds)} x rung in {1, 2, 4, 8, 16,
32, 64} x mean-RMS convention (per-coordinate RMS matched to the wash's
mean). Reads at every rung: the store's g0 (all 8 patterns; kill def
g0 < 0.5, g3R's bar) AND the host fact's battery p(Z) (kill def p(Z) <
0.5, e185's bar). Kill rung = first rung below bar (interpolated in
log2 between adjacent rungs; off-grid-high = >64x flagged).
Co-reads (honesty): the wash direction's RMS concentration profile
(top-decile coordinate share; store-coords share of wash RMS vs of
isotropic RMS — the convention-disagreement reading); both rulers'
curves in full.

## Draft bars (freeze at registration)

  KAPPA-SPLIT: "fires if kappa_host <= 4 and kappa_store >= 8 at
      matched mean per-coordinate RMS — directional immunity is
      purchasable; the store's wide cone is a design property."
  NO-BASIN-UNIVERSAL: "fires if both kappas <= 4 — e185's no-basin
      generalizes; the critic's 24-32x was convention-bound; the
      concentration co-read names where the wash's energy lives."
  MIXED: "any other pattern — graded; the pair reported verbatim."

## Honest failure modes (pre-registered)

- Ruler asymmetry: g0 (store attractor overlap) and battery p(Z)
  (behavioral) have different dynamics — the kappa PAIR is the finding,
  never a single number; state both kill defs on every card.
- Organ n=1 (g3R's scope carried); replicate ladder only if a split
  appears.
- The host-fact ruler's provenance on the g3 organism must be verified
  against the g3 build (which root, which battery) before compute —
  pre-dispatch instrument check (Rule 12; W021's meta-law applies to
  this cell too: compute what each arm GUARANTEES — isotropic at
  matched mean RMS guarantees the store coords receive the same RMS as
  everyone; it does NOT guarantee sparing at any rung; the ladder's
  upper rungs could have failed — that is why this cell is honest).
- The wash direction is ONE trajectory snapshot (g3R's convention);
  the wash drifts — a co-read on the terminal wash direction vs the
  mean wash direction both reported (the tilt ladder already licensed
  tilt robustness ~45 degrees).

## Envelope

Eval-only; CPU permitted (no GPU claim, can run beside GPU cells);
perturb-and-eval per rung (minutes); no checkpoints written beyond
metrics + PNG.
