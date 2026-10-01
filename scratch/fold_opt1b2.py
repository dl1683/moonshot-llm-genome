# -*- coding: utf-8 -*-
"""Fold opt1b2: NOTES, T145, QUEUE (DONE + opt1b3), STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## opt1b2 — the gate-crossing read: CAP-NEITHER — and the projection itself falsified: the bleed's D(t) is SUBLINEAR (a diffusive walk); it entered the kill window's lower margin ALIVE and stalled; the kill extrapolates ~14 steps past the cap (2026-10-01 ~08:50Z) — DONE

WHAT WE DID: continued the committed s600 bleed (resume certified
bit-identical: chunk_state == s600.pt == journal, displacement
recompute |diff| 0.0; all 9 gates PASS) in 7 CPU chunks to the
1200-step cap with fine cadence in the window.

WHAT WE SAW (T145): alive 0.3290 at D 2.1455 at cap — the bleed
CROSSED INTO the kill window's lower margin ALIVE (D 2.12 at ~s1167,
g-12 0.324) and stalled; the 2.6 gate unreached; neither bar fired.
THE HEADLINE: opt1b's labeled projection (D=2.6 at s761; kill at
s1829/D~10) is FALSIFIED AS ARITHMETIC — it divided remaining
displacement by the per-step norm (0.0072/step) assuming colinear
steps; measured D(761) = 1.68 and D grows at 0.0012/step (5.9x
sublinear; late-window 0.00075 and decelerating): consecutive raw
gradients are NOT colinear — THE BLEED IS A DIFFUSIVE WALK, and any
linear-rate projection overestimates D growth. TEXTURE: pump-
plateau-erode-stall (0.601 -> ~0.49 plateau at D 1.6-1.75 -> 0.35
at D 2.04 -> stall 0.33); the pump-cliff acceleration criterion
NEVER fired in-window; CE_R 1.72 -> 1.69 (the organism still
improving); cos(g0) flat ~-0.02 (RAW-WINS-consistent). THE OPEN
DOOR: last-slope kill extrapolation ~s1214 — ~14 steps past the
cap; opt1b3 (a tens-of-steps continuation) decides KILLS-AT-GATE vs
SPARED directly. Checkpoint: runs/checkpoints/opt1b2_a2b_sgd_1e-2_
s1200.pt. HONESTY: n=1, CPU fp32, single stream; the new
extrapolations are labeled and inherit the same falsified-model
warning (they use the linear D-model this run just broke).
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T144 —"
card = """## T145 — opt1b2: the bleed is a diffusive walk — the trajectory-class axis is BALLISTIC vs DIFFUSIVE (2026-10-01 ~08:50Z)

CAP-NEITHER with a bonus falsification: the bleed entered the kill
window's lower margin ALIVE (0.324 at D 2.12) and STALLED at 0.329/
D 2.1455 — and the projection that said "kill at D~10" was wrong as
arithmetic: it assumed colinear steps, but consecutive raw
gradients are far from colinear; per-step norm 0.0072 buys only
0.0012 of displacement (5.9x sublinear, decelerating). THE BLEED IS
A DIFFUSIVE WALK. THE TRAJECTORY-CLASS AXIS RENAMES ITSELF:
ballistic-maximal (the annihilation: one huge colinear step, kill at
D 0.92), ballistic-normalized (the guillotine: Adam's ~2 colinear
sign-steps, kill at D ~2.5), and DIFFUSIVE (the bleed: a random-walk
in gradient space whose displacement grows ~sqrt-ish, stalling at
the window's edge ~D 2.1-2.2 at 0.33). THE DECIDING DOOR: the
last-slope extrapolation puts the kill ~14 steps past the cap —
opt1b3 (dispatched, tens of steps) reads the bleed's own kill-D
directly: ~2.2-2.6 reunifies the gate (all classes die near the
same ring, protection = staying diffusive/slow); >>2.6 keeps the
classes separate (each has its own ring). W026's managed-bleed noun
upgrades: the rhythm's candidate protection is KEEPING THE WALK
DIFFUSIVE (replay events re-randomize the step directions); e192's
pinned-ray rider tests the same axis interventionally (a pinned walk
is forced-ballistic — dies at the static cliff iff direction-
randomness, not step size, is the protection). The projection
lesson joins W021's family: an arithmetic model is an instrument —
this one was falsified by the measurement it motivated.

"""
assert anchor in t and "## T145" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| opt1b2 |"):]; row = row[:row.index("\n")]
new_row = ("| opt1b2 | THE GATE-CROSSING READ | DONE 08:50Z (T145: CAP-NEITHER — entered the window's lower margin "
           "ALIVE (0.324 @ D 2.12) and stalled at 0.329/D 2.1455; THE PROJECTION FALSIFIED — the bleed's D(t) is "
           "SUBLINEAR (5.9x; consecutive gradients not colinear: a DIFFUSIVE WALK); texture pump-plateau-erode-stall; "
           "the kill extrapolates ~14 steps past the cap) |")
q = q.replace(row, new_row, 1)
i = q.index("\n", q.index(new_row)) + 1
opt1b3 = ("| opt1b3 | THE LAST FOURTEEN STEPS (opt1b2's door: a tens-of-steps continuation to the bleed's own kill-D "
          "— decides KILLS-AT-GATE (reunifies the gate: protection = staying diffusive) vs SPARED (classes keep "
          "separate rings)) | DISPATCHED 08:51Z — CPU, minutes; inherits the s1200 checkpoint |\n")
q = q[:i] + opt1b3 + q[i:]
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

s = json.load(io.open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = "2026-10-01T08:51:00Z"
s["current_experiment"] = ("opt1b2 FOLDED (T145: the bleed is a DIFFUSIVE WALK — the projection falsified; entered "
                           "the window alive and stalled; the kill ~14 steps past the cap). opt1b3 DISPATCHED (the "
                           "deciding tens-of-steps). Fleet: g1bS + e192 + e182c + opt1b3 + R59-auditor pending.")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(s, indent=2, ensure_ascii=False))
print("opt1b2 folded")
