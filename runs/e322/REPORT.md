# E322 — THE SQUATTER'S DEED (the 3-room ladder) + THE GHOST CONTROLS

* PART 1 (the ghost control): **GHOST-FLAT** — the ghost hugs the renorm line (max excess 1.000x <= 1.5) while the parasite holds 2.32x its line (peak 0.079006 vs mechanical ceiling 0.034085) — the recruitment is CONTENT-SELECTIVE; part 2 adjudicates at full strength
* PART 2 (the ladder): **MIXED** — neither bar: vacant 0.4123/3 vs fresh 0.2236/1 — the table verbatim
* CONTENDED co-clause: FIRED (host steps 4 vs fresh 1)
* THE WASH SECONDARY: FELL — retention fresh 0.002 / vacant 0.003 / host 0.005
* P-e322a (GHOST-FLAT): FIRED; P-e322b (MARKET-RATE): FELL; WASH-INVERTED: FELL

## Part 1 — the ghost panel (the replay, gen 31103)

| event | host p(Z) | parasite p(T) | ghost p(Q) | ghost2 p(N) | PN total | renorm line |
|---|---|---|---|---|---|---|
| e0 | 0.436550 | 0.019244 | 0.002262 | — | — | 1.000 |
| e5 | 0.003442 | 0.077081 | 0.000511 | 0.002820 | 0.5769 | 1.764 |
| e10 | 0.001143 | 0.079006 | 0.000398 | 0.002813 | 0.3461 | 1.771 |
| e15 | 0.000559 | 0.056491 | 0.000352 | 0.002797 | 0.2304 | 1.773 |
| e20 | 0.000379 | 0.040028 | 0.000329 | 0.002791 | 0.1773 | 1.774 |
| e25 | 0.000310 | 0.029594 | 0.000318 | 0.002795 | 0.1484 | 1.774 |

* THE PEAK'S ERROR BAR (draw2, gen 32203): draw1 0.079006 / draw2 0.070972 -> **0.074989 +/- 0.004017**; draw2's ghost peak 0.000219 (rise -0.002043).

## Part 2 — the 3-room ladder (25-step cons, TAVIREN pool, identical draws)

| room | t0 p(T) | steps-to-0.05 (stable) | landing s25 | post-wash s100 | retention | host fate (t0 -> s25 -> wash) |
|---|---|---|---|---|---|---|
| VACANT | 0.079006 | 3 | 0.412294 | 0.001166 | 0.003 | 0.0011 -> 0.0002 -> 0.0003 |
| HOST-OCCUPIED | 0.003765 | 4 | 0.369947 | 0.001819 | 0.005 | 0.2646 -> 0.0002 -> 0.0000 |
| FRESH | 0.002304 | 1 | 0.223550 | 0.000385 | 0.002 | 0.0000 -> 0.0001 -> 0.0000 |
| VACANT-E25 | 0.029594 | 3 | 0.442058 | 0.001408 | 0.003 | 0.0003 -> 0.0001 -> 0.0000 |

## The construction (disclosed)
* THE GHOST: QELVARO (e293's committed bank), prior-matched (p(Q)@base 0.002591 vs TAVIREN's own 0.002304); NYSTORA/BUVONDI co-read.
* THE REPLAY: e321's erase verbatim from the committed armO state (flat-md5 bound), gen 31103; every committed panel value within the frozen 2e-03 band; the e10 state captured live (the vacant room's start).
* THE CONS: chunked_consolidate verbatim (seed 10901, 25 steps, natural), the TAVIREN pool, draws bit-identical across rooms (G_CONSIDENT); the wash: the corpus half alone (seed 10902, 100 steps, G_WASHIDENT).
* THE FRESH DISJOINT ROOM: the host room's own frame, index slice [k:2k) (k=10000) — exactly disjoint (cross-projection 2.1e-18); where the teaching wrote: VACANT: host-room 0.066/fresh-room 0.060 (chance 0.060); HOST-OCCUPIED: host-room 0.063/fresh-room 0.060 (chance 0.060); FRESH: host-room 0.061/fresh-room 0.061 (chance 0.060); VACANT-E25: host-room 0.066/fresh-room 0.061 (chance 0.060)

## PART 3 — the composite
* the composite: if part 2 shows the vacant room renting faster, the 'possession is nine-tenths' clause drafts; if market-rate, the room's biography ends at its fact's death (T284's fork resolves).
* RESOLUTION: part 1 **GHOST-FLAT** + part 2 **MIXED** — undetermined — the tables verbatim

## Provenance
* birth commit: e9a9a65950e7703a0315d24fae5e24964a4237e2; final head: 6033954169bcfe6b553798f31e15b5e36597623e
* parents: e321 (metrics + armO + erase_post), e293 (the bank), e281 (the cons), e291 (the rooms), the host fact, the base, rooms264 — all md5-bound (G_PARENTS); all gates in metrics.json.
* envelope: bursts <= 175s, polls to runs/_envelope_log.jsonl tagged e322:*; timestamps UTC only.

*No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator folds).*