# -*- coding: utf-8 -*-
"""Fold opt1b: NOTES, T142, QUEUE (DONE + opt1b2), STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## opt1b — the direct SGD kill: CAP-NEITHER — the fact ALIVE (0.601) at D 1.43, 600 steps in; the gate unreached; at matched displacement the raw-gradient path is GENTLER (2026-09-30 ~12:25Z) — DONE

WHAT WE DID: opt1's A2b (SGD 1e-2) continued by deterministic replay
from its committed 69-step state (no a2b checkpoint existed — SGD
is stateless, the seed-10902 stream carries; gated bit-identical,
max|diff| 0.0) in 8 ckpt-resumable <=180s CPU chunks to the kill /
D=2.6 / 600-step cap, whichever first.

WHAT WE SAW (T142): CAP-NEITHER per the frozen bars. The fact
PEAKED (0.9553 at step 80 / D 0.32 — the small-displacement pump)
then eroded monotonically to 0.6013 at step 600 / D 1.43 — alive
above even the 0.50 line at the displacement where every Adam arm
was 1-2 steps from dead. THE MATCHED-D TEXTURE: at D 1.65 (Adam's
first-step displacement) SGD holds 0.79 vs Adam's 0.678 — at EQUAL
raw displacement the raw-gradient path preserves more.
EXTRAPOLATED (labeled, non-adjudicating): D=2.6 at ~step 761 with
g-12 ~0.56 ALIVE — the SPARED trigger past the cap; kill ~step 1829
(D ~10, OUTSIDE the [2.12, 3.27] gate bracket — if it holds, the
gate is trajectory-class-typed). ALIGNMENT flat-negative, slightly
LESS death-directed as D grows (-0.033 -> -0.026; e188/T141
RAW-WINS consistent); CE_R 1.66 -> 1.72 (the organism barely hurt;
Adam's kills ran at CE_R 2.0-2.2). FREE N=2: the registered run
bit-reproduced an accidentally full-depth shakedown (600/600 steps
+ all reads). WHAT'S NEXT: opt1c (dispatching — the direction-size
factorial) and opt1b2 (registered — cap ~800, the D=2.6 crossing
read DIRECTLY; the projection never adjudicates).
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T141 —"
card = """## T142 — opt1b: CAP-NEITHER honestly — Adam is the guillotine, SGD is the bleed; the gate question moves to opt1c/opt1b2 (2026-09-30 ~12:25Z)

The direct SGD kill honors its frozen cap: at 600 steps the fact is
ALIVE (0.601) at D 1.43 — the D=2.6 gate unreached; neither bar
fired. The labeled projection reads the crossing at ~step 761 with
g-12 ~0.56 ALIVE and the kill at ~step 1829 / D ~10 — OUTSIDE the
[2.12, 3.27] bracket: if it holds, the gate is TRAJECTORY-CLASS-
TYPED (raw-gradient paths kill at ~4x the Adam gate) — but the
projection never adjudicates; opt1b2 (registered: cap ~800, the
crossing read directly) owns it. THE MEASURED GEM: at matched D
1.65 — Adam's first-step displacement — SGD holds 0.79 vs Adam's
0.678: at EQUAL raw displacement the raw-gradient path preserves
more than the sign-normalized path. TEXTURE VOCABULARY (the
trajectory classes get names): ADAM IS THE GUILLOTINE (dead in ~2
steps, any stream, organism shocked); SGD IS THE BLEED (pump to
0.955 at D 0.32, then slow monotone erosion with the organism
nearly unharmed — CE_R 1.66 -> 1.72). The pump-then-erode shape is
the bleed's signature. ALIGNMENT: flat-negative, slightly LESS
death-directed as D grows — RAW-WINS consistent (e188). FREE N=2
determinism (the registered run bit-reproduced an accidental
full-depth shakedown). WITH e188: the picture is now — the GATE is
raw displacement for sign-normalized paths; the raw-gradient class
may carry its own (larger) gate; opt1c splits direction from size
inside the Adam kill; opt1b2 catches the bleed at the crossing.

"""
assert anchor in t and "## T142" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| opt1b |"):]; row = row[:row.index("\n")]
new_row = ("| opt1b | THE DIRECT SGD KILL | DONE 12:25Z (T142: CAP-NEITHER per the frozen cap — alive 0.601 at D "
           "1.43; the pump confirmed (0.955 @ D 0.32) then monotone bleed; MATCHED-D: SGD 0.79 vs Adam 0.678 at D "
           "1.65; projection (non-adjudicating): kill ~D 10 — outside the Adam gate; ADAM=GUILLOTINE, SGD=BLEED; "
           "free n=2 determinism) |")
q = q.replace(row, new_row, 1)
i = q.index("\n", q.index("| opt1c |")) + 1
opt1b2 = ("| opt1b2 | THE GATE-CROSSING READ (opt1b's follow-up: continue the bleed to cap ~800, read the fact AT "
          "the D=2.6 crossing directly — adjudicates SPARED-vs-KILLS at the gate; the projection never does) | "
          "QUEUED — CPU after opt1c; inherits opt1b's checkpoint + trajectory |\n")
q = q[:i] + opt1b2 + q[i:]
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

s = json.load(io.open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = "2026-09-30T12:26:00Z"
s["current_experiment"] = ("opt1b FOLDED (T142: CAP-NEITHER — Adam is the guillotine, SGD is the bleed; matched-D "
                           "gentleness; the gate question to opt1c+opt1b2). Fleet: g1bS (GPU) + opt1c DISPATCHING "
                           "(CPU). opt1b2/e189/e190 queue behind.")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(s, indent=2, ensure_ascii=False))
print("opt1b folded")
