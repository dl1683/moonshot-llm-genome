# -*- coding: utf-8 -*-
"""g2g fold part 2: paper R6(b) (actual anchors), QUEUE, STATE."""
import io, json

p = "scratch/day6_paper_skeleton.md"
s = io.open(p, encoding="utf-8").read()
old = """RHYTHM — a zero-parameter rehearsal organ (cue pool + onset monitor
   + replay gate) self-times resurrection events in the 20-45 band,
   three seeds one waveform (medians 0.587/0.602/0.615, duty ~70%);
   scoping: timing root-robust 2/2, amplitude root-draw-bound (the root
   recipe is a strength lottery 0.591/0.684/0.711; g2f base-redraw in
   flight);"""
new = """RHYTHM (g2g-controlled) — a zero-parameter rehearsal organ (cue
   pool + onset monitor + replay gate) self-times resurrection
   events: threat-responsive within [0.5x, 2x] (a weak step,
   ceiling-saturated beyond), autonomy worth +0.07 of cycle-median
   over a matched fixed schedule at operating threat (run-stable;
   FIXED-MATCHES-OR-WINS the safer letter beyond), refractory-
   tunable with shorter better (the 20-45 band partly a
   construction floor, disclosed); timing root-robust 3/3,
   amplitude root-draw-bound (T136; the seed ladder licensed);"""
assert old in s, "r6b"
s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8").write(s)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| g2g |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| g2g | THE RHYTHM'S CONTROLS | DONE 16:55Z (T152: THERMOSTAT-ON-A-LEASH — SELF-TIMED-THERMOSTAT fires run-stable but weak/ceiling-saturated; SELF-TIMED-WINS run-stable at 1x only (+0.07 both runs; the 0.693 co-read superseded as endpoint-vs-median); FIXED-MATCHES the safer letter; REFRACTORY-REAL fires (the band partly a floor; maintenance IMPROVES at r8); the W023 mini-shock lives in the monitor channel; the seed ladder LICENSED) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-01T16:56:00Z"
st["current_experiment"] = ("g2g FOLDED (T152: a thermostat on a leash; the wave CLOSED — T144-T152 all folded). "
                            "R60 TRIO LAUNCHING (the inherited R59 audit + this wave). Then g1bS2 (GPU) + the g2g seed ladder.")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("part2 OK")
