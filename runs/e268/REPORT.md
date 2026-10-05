# E268 — THE ROOM-OPTIMIZER INTERFACE (full report)

**Verdict (the frozen bars, adjudicated): DYNAMICAL-CARRIER** — "the
concurrent arm's expression or landing differs from the serial arm's
beyond the cons/session scatter (post g0 ratio >= 2x or the root outside
+-10% of the serial's) — the cliff's carrier is the room-trajecatory
INTERFERENCE: the barrier is written in the dynamics, completing the
static/dynamical split's mechanism story."

The expression read fired at **6609x** (bar: 2x). The landing read agreed
(root inside the +-10% window) — and the SPLIT between the two readouts
is itself a finding (below). All 13 gates PASS. n=1 per arm (the lottery
note carried); nothing guaranteed, nothing shopped.

## The reads

| arm | post g0 (expression) | root g0 (landing) | kept (install dose) | opt steps |
|---|---|---|---|---|
| SERIAL (the committed condition) | **0.264647** | 0.7641 (in band) | 0.0600 | 400 |
| CONCURRENT (the 1:1 interleave) | **0.000040** | 0.7119 (in band) | 0.0597 | 800 |
| e264's committed K10K rung (anchor) | 0.264648 | 0.770788 | 0.0600 | 400 |

- post-g0 ratio 6609.04x (max/min, the 1e-6 floor convention) — the
  serial arm's write is alive at the threshold rung; the concurrent
  arm's write NEVER sticks: its install trajectory reads g0 <= 0.00074
  at EVERY milestone (s100/200/300/400: 0.00027 / 0.00023 / 0.00074 /
  0.00004) while the serial's reads 0.3059 / 0.2184 / 0.2102 / 0.2646.
- root window [0.0764, 0.8405]: concurrent 0.7119 INSIDE (the landing
  agrees within the disclosed scatter: serial-vs-committed |d| 0.0067
  this session; cross-session law 0.0419).
- root g-12 (never adjudicated, the wild lottery): serial 0.657,
  concurrent 0.789 — inside e264's three-draw range law.

## The discriminator's floor is solid

The SERIAL re-run reproduced e264's committed threshold rung almost
bit-perfectly: install L2 vs the committed vehicle **4.729e-05** (bar
5e-3), |d post g0| **0.0000**, |d root g0| **0.0067** (bars 0.02), kept
median exactly 0.0600. The arms share: the SAME room (bit-bound to
e264_rooms.pt's K10K D/S), the SAME install stream (bit-identical draws
from gen 24314 — CE_inst at s1 is 1.0702 in BOTH arms), the SAME
delivered install dose (kept 0.0597 vs 0.0600), the SAME lr schedule,
the SAME cons (e113 VERBATIM, seed 10901 HELD). The ONLY delta is the
corpus stream running concurrently: one free 48-window corpus step after
every install step, through the one shared AdamW.

## The finding, stated carefully

At the expression cliff's threshold rung, whether a room-confined write
sticks depends on what the optimizer is DOING between the writes. The
room's eigenstructure is IDENTICAL across arms by construction — and the
fate differs by four orders of magnitude. A per-state spectral object
cannot see the cause; a trajectory object can. This is exactly T244's
registered non-spectral carrier form, and the frozen bar's own reading:
**the barrier is written in the dynamics.** With T245 (the counterfeit
inert) the static/dynamical split now has its mechanism story at the
barrier: the room does not "hold" the write by geometry; the write
survives only if the live trajectory lets it accumulate.

## The landing's lesson (the split between the readouts)

The concurrent arm's ROOT lands in band (0.7119) despite a dead write —
because the cons is a rehearsal lane (16 jittered install windows per
batch, s300, natural): it re-teaches the fact from scratch, and both
arms' roots are substantially cons-written (the serial root's
in-own-room fraction falls from 0.94 at post-install to 0.41 at root —
the cons grows the fact off-room). The LANDING read is cons-dominated;
the WRITE read carried the discriminator. This is why both readouts were
registered, and it bounds the claim: the dynamical carrier is a fact
about the INSTALL's write at the cliff's edge, not about the cons's
ability to re-teach afterward.

## The measured mechanics (never nominal)

- kept (install steps): 0.0600 serial / 0.0597 concurrent — the install
  dose was delivered identically; the corpus stream did not shrink it.
- corpus ledger: CE 1.004 -> 0.814 over the run (the concurrent stream
  genuinely trained the organism); |g_clipped| median 1.000 (full-size
  steps).
- displacement at post-install: in-own-room 0.9442 serial vs 0.6696
  concurrent (the corpus dragged a third of the displacement off-room);
  v-excess 1.01 serial vs 0.24 concurrent — the corpus's free write sits
  in LOW-v coordinates (a texture worth savoring: the undertow's supply
  channel; not adjudicated here).
- organism health: CE_R 1.592 (serial post) / 1.628 (concurrent post) —
  the concurrent arm's organism is HEALTHIER on corpus (it trained more
  corpus); the fact's death is not organism death.

## The honest boundaries

1. n=1 per arm, one lineage, one session (the g-series standing caveat).
2. The threshold rung is the MARGINAL write — the cliff's edge. A rung
   ABOVE threshold (40k/100k) might cohabit under the same interleave;
   that pair is the natural next cell (if they survive, the interface
   SHARPENS the cliff's edge rather than merely eroding; if they die
   too, the corpus stream flattens the whole ladder). Not registered
   here.
3. The corpus-dose tripling (in-batch 48 + interleaved 48 vs serial's
   in-batch 48) is the intervention's body, disclosed at birth: the
   install dose is identical, and the concurrent stream IS the object
   under test — but the cell cannot separate "the trajectory's
   curvature" from "more corpus gradient" at this dose; the pair at 2
   (and the free-vs-projected corpus-step variant) can.
4. The corpus steps were UNPROJECTED (the registered choice): the
   natural stream, so its trajectory could drag the state through and
   off the room (it did: in-own-room 0.67 vs 0.94).

## Composition with the day

- T244/e267's two surviving carriers: this cell discriminates the cheap
  one and it FIRES — the deep-spectrum/Lanczos branch does NOT inherit
  at this dose.
- T242/e264's located cliff (~10k, the 6%-dose rung): the vehicle — and
  the serial re-run bound it at L2 4.7e-5.
- T245/e263's completed static/dynamical split: the barrier joins the
  killer and the destination in refusing static geometry — and now the
  barrier's dynamical carrier is measured, not just inferred.

## Provenance

- Script: `lab/e268_room_interface.py` (bars frozen verbatim at birth,
  commit 9334437, BEFORE any compute; smoke commit 82c822e — 2 pre-
  compute catches).
- Machinery: e261's drivers VERBATIM BY IMPORT (serial + cons); the one
  new driver = `chunked_install_concurrent`.
- Artifacts: `runs/e268/{metrics.json, e268_room_interface.png,
  e268_instrument.png, run.log}`; checkpoints
  `runs/checkpoints/e268_{rooms,SERIAL_root,CONCURRENT_root}.pt` +
  resume vehicles.
- Thermal: max 76.0C, zero >= 84C violations, per-step polls on both
  streams (the coverage amendment in metrics.json disclosed).
- Wall: 1288.0 s total; 2 installs + 2 cons; no washes; no fresh FREE
  (e264's G_FREE PASS cited as the instrument validation).
