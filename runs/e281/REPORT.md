# E281 — THE REHEARSAL DOSE-RESPONSE (the retrieval story's lynchpin)

**VERDICT: FLAT** (all hard gates PASS)

every arm lands in the family's landing band [0.65, 0.82] INCLUDING the no-install zero point — P-281b CONFIRMED (floor 0.6508 >= 0.65): the lane carries zero write information; 'delivery is free' collapses to 'the cons teaches from anything'; T246's rehearsal-lane reading and W042's bridge lose their object; the landing read is a property of the CONS alone — the five points: NOINST(w=0) 0.6508 | R10(w=10) 0.6931 | K1KM(w=1000) 0.6881 | K10KD(w=10000) 0.7118 | K100KD(w=100000) 0.8104

## The five points (one cons stream, one session)

| arm | width | seed post g0 | ROOT g0 (the landing) | in band | committed landing (anchor) |
|---|---|---|---|---|---|
| NOINST | 0 | 0.000013 | **0.6508** | YES | — (fresh arm) |
| R10 | 10 | 0.000040 | **0.6931** | YES | — (fresh arm) |
| K1KM | 1000 | 0.001245 | **0.6881** | YES | 0.6879 (|d| 0.0002) |
| K10KD | 10000 | 0.000040 | **0.7118** | YES | 0.7119 (|d| 0.0001) |
| K100KD | 100000 | 0.010897 | **0.8104** | YES | 0.8103 (|d| 0.0001) |

- **the floor (the cons-only zero point): 0.6508** — P-281a clause (floor < 0.45): does not fire; P-281b clause (floor >= 0.65): FIRES
- the registered bars: FLAT / THRESHOLDED / MASS-SCALED / MIXED (verbatim in the script docstring + metrics.registered)
- the formation overlay: e272's serial curve (edge at (1k, 2k]) — this cell's landing curve is its complement on the same axis

### THE BAND-EDGE STRADDLE, NAMED (the verdict's one knife-edge, disclosed per the registered operationalization)

The floor 0.6508 sits **0.0008 above the letter's hard band floor (0.65)**
and **0.0195 BELOW the family's +-10%-of-g1c context window (0.6703)** —
inside the pre-sized (0.65, 0.6703] window between the two band forms.
The verdict FLAT is adjudicated against the registered primary (the
letter's hard band, exactly as frozen at birth); the straddle is NAMED,
not silently adjudicated. Two honest companions: (1) the floor's own cons
trajectory oscillated 0.39-0.75 across its s25-s300 milestones — the
endpoint lottery's amplitude (~0.15+) dwarfs the 0.0008 margin, so the
floor is statistically indistinguishable from a mid-band draw under a
different cons endpoint; (2) the REGISTERED read is the s300 endpoint of
the registered stream (seed 10901) — 0.6508, in-band. Both statements
are true; the bar binds the second. What is NOT knife-edge: the floor is
0.20 ABOVE P-281a's 0.45 clause — the THRESHOLDED world is far away,
and the cons-only lane reaches ~88% of the seeded arms' mean landing
(0.6508 vs 0.7259) from a fact-free base in 25 steps (g0 0.6870 at
s25).

## The seed states: checkpointed vs regenerated

- **ALL THREE loaded — CHECKPOINTED, no regeneration**: `e272_K1KM_inst_resume.pt` (md5 2306a448..., step 400), `e268_CONCURRENT_inst_resume.pt` (md5 9b66f010..., step 400), `e270_CONCURRENT_inst_resume.pt` (md5 59eeed2b..., step 400) — each is its cell's committed FINAL state, behaviorally bound (|d post g0| vs the committed metrics measured live; the birth desk-check read 0.0 bit-exact on all three; x10's fresh-md5 convention — no committed md5 exists for resume files).
- arm (a) NO-INSTALL: the fresh `e001.pt` base (md5 d114536d..., x6's committed bind) — no install at all.
- arm (b) RANK-10: a FRESH install this session (room k=10, seeds 28111/28112 registered fresh; dead gated (post g0 4.03e-05 < 0.01); e246's ALIGNED 2.86e-5 the precedent context).

## Gates

- HARD: G_NAMEFREE / G_SPLICE / G_BATTERY / G_ANCHOR / G_INSTMASK / G_PARENTS / G_BASE / G_ROOT / G_VMBIND / G_SPANBIND / G_PROJ / G_SEEDSTATE / G_R10DEAD — all PASS (a failure would have halted).
- NON-HALTING G_CONS_ANCHOR: K1KM |d| 0.0002, K10KD |d| 0.0001, K100KD |d| 0.0001 (bar 0.02; the cross-session cons lottery family ~0.0419 — expected to miss ~half the time; never a bar input).
  - **THE ANCHOR SURPRISE (an instrument datum, named for the
    coordinator)**: all three loaded arms' fresh landings reproduce their
    committed landings to <= 0.0002 — 100x tighter than the 0.02 bar and
    ~200x tighter than the ~0.0419 cross-session "cons lottery" family.
    The lottery family belongs to RE-RUN INSTALLS (different final
    states), not to a re-run cons from a BIT-IDENTICAL state: the cons
    lane is deterministic to ~1e-4 cross-session given the same start.
    Every prior "landing-read lottery" caveat (e269/e270/e271's
    anchor-contamination MIXED letters) priced the wrong channel for
    this configuration — the landing read is a stable instrument when
    the seed state is bit-bound.

## Disclosures

- The dispatch letter anticipated REGENERATING e268's/e270's dead writes if not checkpointed; the resume checkpoints ARE the committed final states (step 400 + traj last step 400) — loaded, not re-run (the ~5 GPU min unspent).
- The landing band has two near-coincident forms: the letter's [0.65, 0.82] (HARD, adjudicating) and the family's +-10%-of-g1c [0.6703, 0.8192] (context); any adjudication-relevant point between them is a named band-edge branch.
- n=1 per arm, one session; the curve's SHAPE is the registered object — the within-session design makes it immune to the cross-session cons lottery that scattered the prior landing reads (0.577-0.810) across sessions.
- No NOTES/THINKING/QUEUE/STATE edits (the coordinator folds).

## Envelope: max temp 79.0C over 1900 per-step polls; violations >= 84C: 0; bursts <= 175s, cooldowns 40s (tags e281:ARM:phase).
