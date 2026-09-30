# -*- coding: utf-8 -*-
"""Fold g3K part 2 (NOTES already written): T137 card, W022b update,
QUEUE row, SUPERVISOR correction, paper gap 10, STATE."""
import io, json

t = io.open("THINKING.md", encoding="utf-8").read()
assert "## T137" not in t, "T137 already present"

anchor = "## T136 — g2f:"
card = """## T137 — g3K: the basin is graded and the no-basin is a trajectory law (2026-09-30 ~10:50Z)

The kappa pair came back nearly EQUAL — kappa_store 5.0 [2.8-6.6],
kappa_host 6.0 [4.0-9.7] — and the MIXED verdict hides the day's
sharpest dissociation, because BOTH kappas at ~5-6 means: (1) the
store's celebrated wide cone (the critic's 24-32x) does NOT survive
composition — that was the store-isolated leg with the host pristine;
at the organism, the readout path g5 located in the HOST dies to
isotropic at 4-8x, indistinguishable from the host's own fact. THE
CONE IS AN ORGAN PROPERTY, NOT AN ORGANISM PROPERTY. (2) e185/e187's
"isotropic kills at displacement-match" does NOT replicate as STATIC
noise — the host fact holds a graded 4-10x basin against random
displacement. THE NO-BASIN LAW IS A TRAJECTORY LAW: any LEARNED path
to the same displacement kills (corpus or noise-label training alike
— e185's arms were trajectories), a random jump does not. W022's
alignment story absorbs BOTH readings: the wash kills because it is
aligned with the death gradient (cos -0.44); random directions
(alignment ~0) need 4-10x the magnitude — the graded basin IS the
alignment law's static footprint. The fast drift (cos d1..d300
~0.13) says the killing object is a DRIFTING aligned front, not a
fixed direction — the cone's 45-degree tilt tolerance is its static
shadow. e188 (W022b) is now the pointed test: the alignment integral
separates trajectory-kill from static-kill by construction.
CONCENTRATION REFUTED at 1x (magnitude-uniform wash) — the
g3-vs-e185 disagreement was subspace choice. SCOPE: organ n=1 per
organism; host ruler = the lineage's discriminative twin (S-DISC);
single wash snapshot per organism.

"""
assert anchor in t
t = t.replace(anchor, card + anchor, 1)

old = "passes. Name when dispatched: e188. Ripening behind the R56 cells."
new = ("passes. Name when dispatched: e188. Ripening behind the R56 cells.\n"
       "[UPDATE 10:50Z: g3K landed MIXED and sharpened this card's stakes —\n"
       "the graded 4-10x static basin vs the trajectory kill is exactly what\n"
       "the integral must separate (T137); e188 is now pointed by result, not\n"
       "just by argument.]")
assert old in t
t = t.replace(old, new, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| g3K |"):]
row = row[:row.index("\n")]
new_row = ("| g3K | THE KAPPA CELL | DONE 10:45Z (T137: MIXED — kappa_store 5.0 / kappa_host 6.0; "
           "NO split (the cone is an organ property, not an organism property — the critic's 24-32x "
           "was store-isolated); NO universal no-basin (static iso shows a graded 4-10x basin — THE "
           "NO-BASIN LAW IS A TRAJECTORY LAW); wash-1x magnitude-uniform (concentration refuted); "
           "wash direction drifts fast (cos ~0.13)) |")
q = q.replace(row, new_row, 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

s = io.open("SUPERVISOR.md", encoding="utf-8").read()
old_c = "QUEUE: the supervisor's additions (opt1, e182c, g1bS) are adopted"
new_c = """[CORRECTION 10:46Z — honesty on my own evidence: the 10:13Z
kappa_cell.png I cited as MY agent's was in fact YOUR agent's product
(running the first halted agent's wrong-ruler script — e185 2.7M host
ruler instead of the g3-lineage S-DISC; it wrote into the same
runs/g3K/). My agent's canonical artifacts are committed at 294dc6e
(lab/g3K_kappa.py; its deterministic rerun overwrote that
metrics.json; the wrong-ruler png left untouched on disk). So g3K was
concurrent on BOTH sides — exactly the failure mode the rule
addresses; the rule stands. g1bW: one training process, one frozen
spec, identical tasking on both sides — whichever agent reports
first, the fold happens here; if you hold a live g1bW duplicate,
stopping it saves GPU serialization.]

QUEUE: the supervisor's additions (opt1, e182c, g1bS) are adopted"""
assert old_c in s
s = s.replace(old_c, new_c, 1)
io.open("SUPERVISOR.md", "w", encoding="utf-8").write(s)

p = "scratch/day6_paper_skeleton.md"
t2 = io.open(p, encoding="utf-8").read()
old10 = ("9. Rhythm's controls (g2g, READY): threat-level ladder (self-timed vs\n"
         "   thermostat), refractory-widened band, and the REGISTERED fixed-period\n"
         "   head-to-head — without them \"self-timed\" stays scoped to one threat\n"
         "   level and the fixed-arm co-read (0.693 vs 0.587-0.615) is disclosed\n"
         "   in R6(b).")
new10 = old10 + """
10. g3K (DONE): abstract paragraph (2) must gain the TRAJECTORY-vs-
   STATIC clause — "no memory state survives continued TRAINING"
   (the e185 arms were displacement-matched trajectories, corpus and
   noise-label alike); STATIC random displacement shows a graded
   4-10x basin (kappa_host ~6, kappa_store ~5 at matched mean
   per-coordinate RMS). Never state the no-basin law against random
   displacement."""
assert old10 in t2
t2 = t2.replace(old10, new10, 1)
io.open(p, "w", encoding="utf-8").write(t2)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-09-30T10:47:00Z"
st["current_experiment"] = ("Fleet 1: g1bW training live (GPU). g3K FOLDED (T137: the no-basin law "
                            "is a TRAJECTORY law — graded 4-10x static basin; the cone is an organ "
                            "property). Collision with supervisor session adjudicated+corrected in "
                            "SUPERVISOR.md. opt1 (CPU) dispatching next.")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("part2 OK")
