# -*- coding: utf-8 -*-
"""Fold opt1b3: NOTES, T147, abstract bracket fill, QUEUE DONE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## opt1b3 — the last fourteen steps (recovered twice): CAP-AGAIN (GRADED) — neither bar fires; the stall is a SLOW EROSION; the third linear kill-projection dies; the bleed's own kill-D stays unmeasured (2026-10-01 ~14:35Z) — DONE

WHAT WE DID: third dispatch (second recovery): the killed
predecessor's mid-flight chunk_state @ s1264 certified standalone
(md5 + displacement recomputes exact; tails bit-equal the committed
partial; stream unique); the MID-RUN RESUME GATE added (its v1
aborted once at an in-process chunk boundary — a pure control
failure, zero steps adjudicated, disclosed); v2 resumed from the
saved s1348 chunk, cross-process bind ENFORCED. No committed step
re-executed; bars verbatim; all 10 gates PASS.

WHAT WE SAW (T147): CAP-AGAIN — g-12 read EVERY step for 150 steps:
no read <= 0.27 (min 0.2891 AT the final step; mean 0.3167 +- 0.01)
and D = 2.6 never approached (final D 2.2501; 9.26x sublinear —
walked 6.4e-3 buys 7.0e-4 of D per step, D(t) linear r2 0.9996: a
steady deeply-sublinear drift). THE DOOR CLOSED ON ITS OWN
EXTRAPOLATION: opt1b2's kill ~s1214 was walked through alive — THE
THIRD FALSIFIED LINEAR PROJECTION in this arc (W021's family
grows). THE STALL'S TEXTURE (the graded finding): not a flat
equilibrium but a SLOW EROSION 0.33 -> 0.29 (-2.0e-4/step; margin
0.019 at the end) — a diffusive walk that never dies within any cap
set yet keeps grinding; asymptote unresolved. The organism improves
through it (CE_R 1.677 vs root 1.664); cos(g0) flat -0.0195
(RAW-WINS-consistent). WHERE THE GATE STANDS: dead-in-window and
past-2.6 both unfired — the bleed's own kill-D remains UNMEASURED
(a labeled linear read says ~s1440 at D ~2.32, adjudicating nothing
after three misses of that class); e192/T146's rider answers the
causal axis independently. THE FOUR-CLASS OVERLAY is complete:
guillotine ~2.5 / annihilation 0.92 / random >4.0 flat / the bleed
grinding at 2.25. HONESTY: n=1, CPU fp32, single stream;
CAP-AGAIN graded by registration. Checkpoint s1350 on disk.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T146 —"
card = """## T147 — opt1b3: the walk that will not die — the third projection falsified, the grind named (2026-10-01 ~14:35Z)

CAP-AGAIN, graded honestly: 150 every-step reads, no kill (min
0.2891 at the final step), no 2.6 crossing (final D 2.2501). THE
ARITHMETIC LESSON COMPOUNDS: opt1b2's kill ~s1214 was walked
through ALIVE — the third linear projection this arc has falsified
(opt1b's D~10, opt1b2's s1214, now the same class again); W021's
family of arithmetic-model instruments grows a dedicated shelf.
THE GRIND (the named texture): the stall is not equilibrium but
SLOW EROSION (0.33 -> 0.29 at -2e-4/step) on a 9.26x-sublinear
drift (D(t) linear r2 0.9996 — steady, deeply sublinear): the
bleed never dies within any cap we have set, yet never stops
grinding; its asymptote is UNRESOLVED and now unmeasured-by-design
(three falsified projections means the class is retired, not
retried). THE GATE QUESTION'S HONEST STATE: the walk's own kill-D
is unknown; the CAUSAL axis (why the walk is spared where rays
die) is answered independently by e192's rider — orientation owns
the sparing. THE FOUR-CLASS OVERLAY stands complete for Fig-5's
companion: guillotine ~2.5, annihilation 0.92, random >4.0 flat,
and the grind — three ways to die and one way to erode. THE
ABSTRACT'S BRACKET FILLS: the diffusive walk "grinds below the
ring without dying (asymptote unmeasured)". Savoring: the bleed
is the arc's honest ending — not saved, not dead; grinding.

"""
assert anchor in t and "## T147" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

# abstract bracket fill
p = "scratch/claims_ledger.md"
s = io.open(p, encoding="utf-8").read()
old = "and a diffusive\nsmall-step walk [opt1b3: dies at / survives past] the same ring —"
new = "and a diffusive\nsmall-step walk grinding below the ring without dying (asymptote\nunmeasured; three linear projections falsified) —"
assert old in s, "bracket"
s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8").write(s)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| opt1b3 |"):]; row = row[:row.index("\n")]
new_row = ("| opt1b3 | THE LAST FOURTEEN STEPS | DONE 14:35Z (T147: CAP-AGAIN graded — no kill in 150 every-step "
           "reads (min 0.2891 final); D 2.2501 final, 9.26x sublinear; the THIRD linear projection falsified "
           "(walked s1214 alive); the stall = SLOW EROSION 0.33->0.29; the bleed's kill-D UNMEASURED, projection "
           "class retired; the causal axis carried by e192's rider; four-class overlay complete) |")
q = q.replace(row, new_row, 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-01T14:36:00Z"
st["current_experiment"] = ("opt1b3 FOLDED (T147: CAP-AGAIN — the walk that will not die; the grind named; the "
                            "third projection falsified; abstract bracket FILLED). Fleet: g1bS-r4 (GPU) + e182c-r3 "
                            "(CPU) + THE CHART CELL dispatching (the freed slot).")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("opt1b3 folded")
