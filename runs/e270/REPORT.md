# E270 — THE 100K CONCURRENT BRANCH (full report)

**Verdict (the frozen bars, adjudicated): MIXED** — the named
anchor-contamination branch: "the serial anchor failed (cross-session
texture) — the same-session pair stands but any firing read would carry
the contamination caveat — the trajectories verbatim, both readouts, no
inflation."

**The letter and the content must be read together, and both are
reported — e269's precedent, repeated to the letter.** The frozen
composite order (TEXTURE -> MIXED -> CEILING -> TOTAL, committed at
birth) routes any firing read to MIXED when G_SERIAL_ANCHOR fails; it
failed — but ONLY on the landing read: the serial cons lottery drew
root g0 0.6884 vs the committed 0.7276 (|d| 0.0392 > the non-halting
0.02 texture bar; the known cross-session family is 0.0419, e261's
G_ANCHOR law on a bit-identical arm — the miss sits INSIDE the known
family). The WRITE read's anchor is bit-faithful: install L2 6.3e-5
(bar 5e-3), |d post g0| 1.0e-6. The discriminator's primary object —
the SAME-SESSION arm pair on the write read, one instrument, one
session — is untouched by the contamination, and under the dispatch's
own bar letter ("post g0 < 0.5x its serial") the read is unambiguous:

**THE CONCURRENT 100K WRITE DIES. THE TURBULENCE IS TOTAL ACROSS THE
MEASURED LADDER {10k, 40k, 100k}.** Directed survival ratio
**0.0250x** — 20x below the 0.5x bar, 9.4x below the pre-sized edge
window. The TURBULENCE-TOTAL clause's content holds: the flow's story
closes at three rungs; the two laws stay two everywhere measured. The
formal letter is MIXED with the contamination caveat carried, per the
frozen registration. No bar shopping either direction.

All 12 hard gates PASS. n=1 per arm (the lottery note carried); nothing
guaranteed, nothing shopped.

## The reads

| arm | post g0 (the write) | root g0 (the landing) | kept (install dose) | opt steps |
|---|---|---|---|---|
| SERIAL (the committed condition, re-run fresh) | **0.435981** | 0.688417 | 0.1915 | 400 |
| CONCURRENT (the 1:1 interleave) | **0.010897** | 0.810299 | 0.1905 | 800 |
| e264's committed K100K rung (anchor) | 0.435982 | 0.727612 | 0.1915 | 400 |

- directed survival ratio post_C/post_S = 0.010897/0.435981 =
  **0.0250x** (bar: >= 0.5x to survive — dies by 20x; the concurrent
  post is 4.6x below the ladder's own 0.05 expression floor).
- the milestone trajectories: serial 0.0000 / 0.4980 / 0.4267 / 0.3523
  / 0.4360 (bit-faithful to the committed rung: post |d| 1.0e-6, kept
  |d| 1e-9) vs concurrent 0.0000 / 0.0007 / 0.0221 / 0.0065 / 0.0109 —
  the concurrent write NEVER sticks; it flickers late, decays, and ends
  two orders below its serial twin.
- the landing read (CARRIED, never adjudicated — T246's rehearsal-lane
  caveat): concurrent root 0.8103 vs serial 0.6884 — the dead write
  lands ABOVE its own serial's landing (outside the +-10% context
  window on the HIGH side); the cons re-teaches a DEAD write past full
  strength at the same rung where it died, the pattern's third
  occurrence at a third rung (e268: 0.7119 with a dead 10k write; e269:
  0.7047 vs serial 0.7034 with a dead 40k write). Root g-12 0.8405
  (the wild lottery, never adjudicated).

## The transient's scaling — the named open branch's other prong, and
it breaks

T247's dichotomy: "if the transient keeps scaling, somewhere the flash
survives; if it saturates, the turbulence is total." THE ANSWER: it
does NOT keep scaling — and the write dies.

- **the s100 chain collapses**: 10k 0.000272 -> 40k 0.024468 (90.0x)
  -> 100k **0.000731 (0.030x)** — the 100k rung's early flash is 33x
  SMALLER than the 40k's, not 2.2x-larger as the 90x-per-4x-rank
  texture projected. The early-life scaling law breaks at the far side.
- **the flash is DELAYED, not bigger**: the 100k trajectory peaks at
  s200 (0.0221) — 0.90x of the 40k peak (s100, 0.0245). The transient
  SATURATES near ~0.02: three rungs, two 4x rank steps, and the peak
  never passes a fiftieth of the serial write.
- **what keeps growing is only the residue**: the s400 endpoint 4.0e-5
  (10k) -> 3.27e-3 (40k, 82x) -> 1.09e-2 (100k, 3.3x) — a sub-linear
  creep, 40x below the survival bar, reported verbatim as texture (n=1
  per rung; no trend is claimed).
- the fate column: dies / dies / dies — ratio 1.5e-4x (e268), 0.0094x
  (e269), 0.0250x (this cell) — monotone in rank, two orders below the
  bar at every rung.

## The mechanics' texture (measured, never nominal)

- the delivered dose held: kept 0.1905 vs 0.1915 (the projection
  delivers the same fraction of every install gradient in both arms).
- the concurrent displacement again sits in LOW-v coordinates: v-excess
  0.26 vs the serial's 0.80 (e268's 1.01 -> 0.24 and e269's 0.88 ->
  0.26 repeat at 100k — the undertow's supply-channel texture, now
  three-for-three); in-own-room dragged 0.98 -> 0.77 (the free corpus
  steps pull the state off the room between install writes).
- the corpus stream's own ledger: CE median 0.8800, clipped-grad norm
  median 1.000 — the concurrent stream ran at full strength throughout
  (no collapse, no explosion).
- thermal: max 78.0C (exactly the burst-end margin — one burst ended on
  it), zero >= 84C violations, 1800 per-step polls, BOTH arms' rows in
  the one ledger (the e269 instrument gap closed this cell).

## Provenance (the short form)

- the vehicle: e264's committed K100K rung VERBATIM — the room rebuilt
  from seeds 26117/26118 and bit-gated vs e264_rooms.pt (D/S exact
  equality; kept2 0.036488 vs 0.036509 at the 10-sigma bar; span-ovl
  0.1911 vs ~0.1911); e001 base; Dmix s400 gen 24314; the SRCT hook
  (clip 1.0 -> project CPU fp64 -> step, norm not rescaled); e113 cons
  s300 seed 10901 HELD on both arms.
- the arms differ in ONE thing: after every install step, ONE free
  48-window corpus step (16 anchors + 32 random, the g1c Dmix corpus
  convention) at the paired step's lr, through the ONE shared AdamW —
  corpus generator seed 27001, REGISTERED FRESH this cell (e268's was
  26801, e269's was 26901). Install streams bit-identical across arms
  by construction (separate generators; CE_inst s1 1.0702 in both).
- parents hard-bound (Rule 12): e264 (SHARP-THRESHOLD; md5
  a42ff478...), e261 (md5 f460475d...), e268 (DYNAMICAL-CARRIER; md5
  c1149229...), e269 (MIXED; md5 58d33356...), the vehicle
  e264_K100K_inst_resume.pt (md5 8061c599..., step 400, traj
  1/100/200/300/400, ledger max 400).
- machinery: e261's drivers VERBATIM BY IMPORT + e268's interleaved
  driver through e269's committed text, ported whole into this file;
  the committed modules are untouched.
- run: 1241.1s total; smoke PASSED clean beforehand (209.7s, zero
  catches, all gates); progressive metrics written at every phase.

## The honest close

The dispatch asked the far side's question and the far side answered
with the third death: at every measured rung the confined write dies
under the live corpus trajectory, the serial write strengthens
monotonically with rank, and the two laws — rank buys quiet-regime
expression; turbulence forbids concurrent survival — stay two
everywhere measured. The transient's promised escalation did not
materialize: the bigger room buys a DELAYED flicker of the same ~0.02
peak, not a bigger flash, and the residue's creep is two orders shy of
the bar. The ceiling, if one exists, lies beyond the measured ladder's
top rung — the flow story closes here as measured, at three rungs,
total.
