# E269 — THE ABOVE-THRESHOLD INTERLEAVE PAIR (full report)

**Verdict (the frozen bars, adjudicated): MIXED** — the named
anchor-contamination branch: "the serial anchor failed (cross-session
texture) — the same-session pair stands but any firing read would carry
the contamination caveat — the trajectories verbatim, both readouts, no
inflation."

**The letter and the content must be read together, and both are
reported.** The frozen composite order (TEXTURE -> MIXED -> PROOF ->
BLOCKS, committed at birth) routes any firing read to MIXED when
G_SERIAL_ANCHOR fails; it failed — but ONLY on the landing read: the
serial cons lottery drew root g0 0.7034 vs the committed 0.7520 (|d|
0.0486 > the non-halting 0.02 texture bar; the known cross-session
family is 0.0419, e261's G_ANCHOR law on a bit-identical arm). The
WRITE read's anchor is bit-faithful: install L2 5.7e-5 (bar 5e-3),
|d post g0| 9.5e-7. The discriminator's primary object — the
SAME-SESSION arm pair on the write read, one instrument, one session —
is untouched by the contamination, and under the dispatch's own bar
letter ("post g0 < 0.5x its serial") the read is unambiguous:

**THE CONCURRENT 40K WRITE ALSO DIES.** Directed survival ratio
**0.0094x** — 53x below the 0.5x bar, 10x below the pre-sized edge
window. The TURBULENCE-BLOCKS-ALL clause's content holds (the
dynamics-barrier is rank-independent on the write side at the measured
rungs; the interference dominates every confined write tested); the
formal letter is MIXED with the contamination caveat carried, per the
frozen registration. No bar shopping either direction.

All 12 hard gates PASS. n=1 per arm (the lottery note carried); nothing
guaranteed, nothing shopped.

## The reads

| arm | post g0 (the write) | root g0 (the landing) | kept (install dose) | opt steps |
|---|---|---|---|---|
| SERIAL (the committed condition) | **0.346475** | 0.703358 | 0.1211 | 400 |
| CONCURRENT (the 1:1 interleave) | **0.003268** | 0.704679 | 0.1203 | 800 |
| e264's committed K40K rung (anchor) | 0.346476 | 0.751956 | 0.1211 | 400 |

- directed survival ratio post_C/post_S = 0.003268/0.346475 =
  **0.0094x** (bar: >= 0.5x to survive — 0.0094x dies by 53x; the
  concurrent post is also 15x below the ladder's own 0.05 expression
  floor).
- the milestone trajectories: serial 0.0 / 0.3746 / 0.3783 / 0.4843 /
  0.3465 (bit-identical to the committed rung's series) vs concurrent
  0.0 / 0.0245 / 0.0091 / 0.0056 / 0.0033 — the concurrent write NEVER
  sticks; after a transient flicker at s100 it decays monotonically.
- the landing read (CARRIED, never adjudicated — T246's rehearsal-lane
  caveat): concurrent root 0.7047 vs serial 0.7034, |d| 0.0013, inside
  the +-10% context window — the cons re-teaches a DEAD write to full
  landing strength at the same rung where it died, the pattern's second
  occurrence at a different rung (e268: 0.7119 vs a dead 10k write).

## The rank's fingerprint in the transient (the one place the room's
size shows)

The 40k concurrent write's transient is ~80-90x the 10k one's at every
milestone (s100: 0.0245 vs e268's 0.00027; s400: 0.0033 vs 0.00004)
while the kept dose only doubled (0.1203 vs 0.0597) — the bigger room
buys a bigger early life, and the turbulence grinds it down anyway.
Rank writes the SERIAL curve (T242); it does not buy CONCURRENT
survival at these rungs.

## The discriminator's floor

The SERIAL re-run reproduced e264's committed K40K rung essentially
bit-perfectly on everything the discriminator touches: install L2
**5.742e-05** (bar 5e-3), |d post g0| **9.5e-7**, kept median 0.1211
(exact), and the milestone series identical to the committed vehicle's.
The arms share: the SAME room (bit-bound to e264_rooms.pt's K40K D/S,
G_ROOMK40K), the SAME install stream (bit-identical draws from gen
24314 — CE_inst at s1 is 1.0702 in BOTH arms, the family's fingerprint),
the SAME delivered install dose (kept 0.1211 vs 0.1203), the SAME lr
schedule, the SAME cons (e113 VERBATIM, seed 10901 HELD). The ONLY delta
is the corpus stream running concurrently: one free 48-window corpus
step after every install step, through the one shared AdamW (draws from
this cell's REGISTERED FRESH corpus generator, seed 26901 — a different
registered stream from e268's 26801, so the fate cannot be an artifact
of one corpus draw sequence).

## The finding, stated carefully

Above the expression threshold, a confined write with FOUR TIMES the
threshold's dimensions and double its dose still never sticks when the
optimizer is busy. The threshold is NOT the turbulence-proof size: rank
buys serial expression (the capacity curve stands, bit-faithfully
reproduced here) but does not confer concurrent survival. THE CAPACITY
LAW AND THE DYNAMICS-BARRIER DO NOT UNIFY AT THIS RUNG — they are two
different objects: the first is a property of the room's size under a
quiet optimizer; the second is a property of the live trajectory's
interference, and at the measured rungs it is dose/rank-independent on
the write side. The question's alternative phrasing — "above the
threshold the write has enough room to survive the turbulence" — is
refuted at 40k under the 1:1 interleave.

## The measured mechanics (never nominal)

- kept (install steps): 0.1211 serial / 0.1203 concurrent — the install
  dose was delivered identically; the corpus stream did not shrink it.
- corpus ledger: CE ~1.01 -> 0.81 over the run (median 0.892); |g_clipped|
  median 1.000 (full-size steps) — the concurrent stream genuinely
  trained the organism.
- organism health: CE_R 1.633 (serial post) / 1.603 (concurrent post) —
  the concurrent arm's organism is HEALTHIER on corpus; the write's
  death is not organism death.
- displacement at post-install: in-own-room 0.9763 serial vs 0.7164
  concurrent (the corpus dragged ~28% of the displacement off-room; at
  10k it was 0.94 vs 0.67); v-excess 0.88 serial vs **0.26 concurrent**
  — the corpus's free write sits in LOW-v coordinates AGAIN (the
  undertow's supply-channel texture, e268's teaser, now confirmed at a
  second rung; still not adjudicated here).
- displacement at root: in-own-room serial 0.5832 / concurrent 0.6296 —
  the cons grows the fact off-room in both arms (the rehearsal lane's
  own footprint).

## The honest boundaries

1. n=1 per arm, one lineage, one session (the g-series standing caveat).
2. THE MIXED LETTER'S CAUSE IS A LANDING-READ LOTTERY, not any doubt in
   the write read: the serial cons drew root |d| 0.0486 (family law
   0.0419; the gate's 0.02 bar is the session-texture bar, and e268's
   session drew 0.0067 — this session drew wide). The write read's
   anchor is bit-faithful to 9.5e-7. The frozen composite orders the
   caution BEFORE the binary; the caution is carried formally, and the
   content read (dies, 0.0094x, 53x below the bar) is reported verbatim.
3. The corpus-dose tripling (in-batch 48 + interleaved 48 vs serial's
   in-batch 48) is the intervention's body, disclosed at birth: the cell
   cannot separate "the trajectory's curvature" from "more corpus
   gradient" at this dose (inherited from e268; the free-vs-projected
   corpus-step variant remains the separating cell).
4. The 100k rung is unmeasured under concurrency. The transient GREW
   ~90x from 10k to 40k while both died; whether some larger rung
   crosses the survival bar is the honest open branch of this cell (the
   natural next cell if the day wants the dose/rank law's far side).
5. The corpus steps were UNPROJECTED (the registered choice, carried):
   the natural stream, so its trajectory could drag the state through
   and off the room (it did).
6. THERMAL COVERAGE AMENDMENT (e268's precedent, disclosed in
   metrics.json): the summary's 800 polls / max 78.0C cover this file's
   poller (the concurrent install); the serial install + both cons poll
   through e261's ported check (rows in run.log) — TRUE session max
   79.0C (one concurrent-cons margin burst-end), ZERO >= 84C violations
   anywhere. The never-past line held.

## Composition with the day

- T246/e268 (the capstone at 10k): this cell IS its named pair — the
  other side of the cliff measured. Both rungs die under the identical
  interleave; the barrier is rank-independent on the write side at
  {10k, 40k}; the rehearsal-lane finding DOUBLY confirmed (a dead write
  lands at full strength via the cons at both rungs); the low-v corpus
  write repeats (0.24 -> 0.26).
- T242/e264 (the capacity curve): the serial side re-validated
  bit-faithfully at the 40k rung (L2 5.7e-5) — the cliff stands; what
  the concurrency adds is a SECOND, dynamical cliff that the serial
  curve cannot see: expression-vs-rank under a quiet optimizer vs
  survival-vs-concurrency under a busy one.
- T245/e263 + T244/e267 (the static/dynamical split): the split
  deepens — even the room's SIZE (the one variable the static story
  could still own) does not move the dynamical barrier at the measured
  rungs.

## Provenance

- Script: `lab/e269_above_threshold.py` (bars frozen verbatim at birth,
  commit 068b31d, BEFORE any compute; smoke commit c9284dd — 1
  pre-compute catch: the jitter-pool build read train_text where the
  vehicle uses the train_ids tensor slice; bars/gates/arms untouched).
- Machinery: e261's drivers VERBATIM BY IMPORT (serial install + cons);
  e268's `chunked_install_concurrent` PORTED VERBATIM into this file
  (importing lab/e268_room_interface.py would fire its module-level
  side effects — the port is the disclosed route; the committed
  e261/e268 files are NOT modified).
- Parents hard-bound (Rule 12): e264 metrics md5 a42ff478... (the K40K
  rung + G_FREE PASS), e261 metrics md5 f460475d..., e268 metrics md5
  c1149229... (DYNAMICAL-CARRIER), the vehicle e264_K40K_inst_resume.pt
  md5 14097f2a... at s400, the room bit-bound to e264_rooms.pt's K40K
  D/S, e246's span, e258's v-map.
- Artifacts: `runs/e269/{metrics.json, e269_above_threshold.png,
  e269_instrument.png, REPORT.md, DRAFT_NOTES_ENTRY.md, run.log}`;
  checkpoints `runs/checkpoints/e269_{rooms,SERIAL_root,CONCURRENT_root}.pt`
  + resume vehicles.
- Thermal: per-step polls on every opt step (both streams); true session
  max 79.0C; zero >= 84C violations (the coverage amendment above).
- Wall: 1345.7 s total; 2 installs + 2 cons; no washes; no fresh FREE
  (e264's G_FREE PASS + e268's serial-anchor PASS cited as the
  instrument validations).
