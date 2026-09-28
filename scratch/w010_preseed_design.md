# W010 falsifier design — the pre-seed ladder + far-jitter rider (ripening on paper; dispatch after e131)

Parent cards: W010 (seed-and-amplify), T076 (error-location), W008-corrected (adapter frame).
Status: DESIGNED, NOT DISPATCHED. Blocked deliberately on e131 probe 1 — the design's
meaning branches on it. This doc exists so whichever way e131 lands, the next experiment
is already thought through.

## The dependency (read this first)

e131 probe 1 reads the e120 splice arms at the 183-geometry. Two worlds:

- **NO-SIGNAL-AT-183** (p_z at floor): the splice genuinely failed at its own teaching
  site. Seed-and-amplify's leg-3 stands. The PRE-SEED LADDER below is the clean
  falsifier: does faint pre-existing expression at 183 convert the identical failing
  fine-tune into a climbing consolidation?
- **SIGNAL-AT-183** (p_z >= 0.5 x install-60 level): the splice never failed — it
  consolidated at 183 where no battery looked. Error-location wins outright; T075's
  verdict inverts to "consolidated where we never looked"; the pre-seed design is MOOT.
  The follow-up becomes CROSS-ADDRESS ROBUSTNESS instead: does an 183-only consolidated
  fact survive band deletion (D-all-band + D-183 + both)? A single-site consolidation
  that dies to its own row deletion is just an ordinary install at a new address; one
  that survives is a field entry that used 183 as its only door.

## Branch A: the pre-seed ladder (if NO-SIGNAL-AT-183)

Rig: e120 arm (b) verbatim (corpus-context splice at 183, seed 12101, 300 steps,
CPU-runnable) — the known-WANDERING arm — plus a preceding mini-install phase.

Mini-install (the seed): e043 install protocol with windows shifted so the fact's key
position lands at wpe ~183 (the e120 splice's own site; shift = +54 from the 129
anchor). Dose ladder, 3 levels, targeting FAINT expression at the 183-geometry:
  S1: ~0.05-0.10 p_z (whisper seed)   S2: ~0.15-0.25 (faint)   S3: ~0.30-0.45 (clear)
Calibrate dose (steps/LR) on one throwaway net per level; record seed strength actually
achieved. Then run the IDENTICAL arm-b splice fine-tune on each pre-seeded net.

Arms (all CPU, ~300 steps each; e131's regenerated e131_arm_b_corpus_spliced.pt is the
no-seed control, already in hand):
  0. no-seed splice (sitting data: wanders, ends 0.008)
  1. S1-seeded splice
  2. S2-seeded splice
  3. S3-seeded splice
  4. S3-seed ONLY, no splice fine-tune (does the seed alone survive? decay control)
  5. S3-seeded + name-free corpus fine-tune (does ordinary drift amplify or erase?)

Measurements:
  - steps-to-liftoff: first traj checkpoint where p_z_mean (at 183-geometry, install-60
    battery construction) exceeds 2x its arm-max in the first 50 steps AND stays above
    for all later checkpoints (climb = sticky).
  - trajectory class: CLIMB (Spearman(p_z, step) >= 0.7 across 6 checkpoints) vs WANDER
    (|rho| < 0.4) vs COLLAPSE (rho <= -0.7).
  - final post-D-all-band AND post-D-183 expression (does amplification at 163-183
    create band-crossing structure, or stay site-locked?).

Registered outcomes (bars, for the eventual docstring):
  SEED-GATED fires if arms 1-3 liftoff monotone in seed dose and arm 0/4 do not climb.
  SEED-INDEPENDENT fires if all arms climb (error-location sufficient; W010's seed
    clause dies — splice failure had another cause, likely context pull).
  SEED-HARMFUL fires if pre-seeded arms collapse where arm 0 wanders (the fine-tune
    UNLEARNS the seed without amplifying it — the seed must be error-bearing, not just
    present; this would split W010's "seed" into expression-seed vs error-seed).
  BAND-ONLY-SEED (sub-outcome of SEED-INDEPENDENT variant): if S3 climbs only when the
    seed sits in the 121-137 band (add arm 6: band-seeded splice for this), seed means
    band-membership, not expression-anywhere.

## Branch B: cross-address robustness (if SIGNAL-AT-183)

Rig: e131's regenerated splice arms (nets exist as runs/checkpoints/e131_arm_{a,b}*.pt).
Battery: expression at 183-geometry under deletions: none / D-183 / D-band(5) /
D-band+183 / D-row-0 (the critic's row rides free here too).
  SITE-LOCKED fires if D-183 kills >= 80% of expression (ordinary install at a new
    address — "consolidation at 183" was a misnomer).
  FIELD-ENTRY fires if expression survives D-183 >= 0.5x its no-deletion level AND
    survives D-band+183 partially (the fact used 183 as a door but lives beyond it).

## Rider either way: far-jitter (W010 falsifier a)

Jitter offsets widened from [-8..8] to ±64 (seedless positions), same budget/mask as
e120 arm (d). Prediction: WANDER not CLIMB. If it climbs, seed-and-amplify dies and
pure error-location stands (error anywhere on the fact suffices; the band is not
special). Rides whichever branch runs — same rig class, one extra arm.

## Compute

Branch A: 3 calibration + 6 arms x 300 steps CPU ≈ 40 min (or one GPU batch after
e119 frees it, honoring cooldown). Branch B: eval-only on existing nets, minutes.
