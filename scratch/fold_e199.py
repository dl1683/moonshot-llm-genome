# -*- coding: utf-8 -*-
"""Fold e199: NOTES, T163, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e199 — the concentration's onset curve: ONSET-COMMON — A WHEN, NOT A WHETHER; org-1-EARLY, not org-1-only; MIRABEL's drop-in texture (2026-10-01 ~23:00Z) — DONE

WHAT WE DID: both organisms' committed walks extended to t=6 or
death (vacuous: both die first — org1 t=2, MIRABEL t=3; T158's
lesson bounds the question); the sign(g_t) rays from each ROOT at
every alive t; 23/23 gates PASS (org1's chain BIT-EXACT; MIRABEL's
chain at a disclosed TEXTURE tier — a measured ~1e-7 cross-
environment fp drift, four orders below any bar margin); 225.9s
CPU.

WHAT WE SAW (T163): ONSET-COMMON FIRES — both organisms'
fronts concentrate from the root: org1 at t=1 (ratio 0.1707,
bit-exact vs e195), MIRABEL at t=2 (0.4269) — THE SAME CURVE
SHIFTED ONE STEP. The t=1 asymmetry stands but its MEANING FLIPS:
the concentration is not org-1-ONLY (e198's verdict) but
ORG-1-EARLY — the rotation needs one step (org1) or two (MIRABEL)
to land on the fleeing support. MIRABEL'S DROP-IN TEXTURE: its t=1
front is SOFTER THAN STATIC (floor ratio 1.54) before the rotation
finds the lethal direction at t=2 — the rotation first points
away, then onto the target. Org1's post-death t=2 context (0.396,
e195 committed) sits beside MIRABEL's alive t=2 (0.427). HONESTY:
n=1 per organism; the t=6 horizon never reached (both die first —
no ripening past death); the MIRABEL chain's TEXTURE tier stamped.
FOLLOW-ON NAMED: e197's half-step alive lineage (alive t=1..4,
only u1 mapped at ratio 3.17) owes its own t=2..t=4 onset curve —
THE ONE ORGANISM WHOSE CURVE COULD DEEPEN across three alive
steps.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T162 —"
card = """## T163 — e199: the WHEN — the flight question closes at onset-shape (2026-10-01 ~23:00Z)

The timing cut returns the day's cleanest synthesis: ONSET-COMMON.
Both organisms' rotating fronts eventually concentrate from the
root — org1 at t=1 (0.171), MIRABEL at t=2 (0.427) — the same
onset curve shifted one step: THE CONCENTRATION IS A WHEN; THE
TIMING IS THE BIOGRAPHY. The e198 verdict's meaning flips without
its numbers changing: org-1-EARLY, not org-1-only. THE DROP-IN
TEXTURE completes the mechanism picture: MIRABEL's t=1 front is
SOFTER THAN STATIC before its t=2 lands on the lethal direction —
the rotation first points away (e194's anti-rotation read), then
ONTO the fleeing support: two movements, not one. THE ARC CLOSES
AT A SHAPE: every organism so far concentrates before it dies
(org1 t=1; MIRABEL t=2; org2 full-step never — it died at t=1
BEFORE its onset; its half-step lineage alive t=1..4 owes the
deepening test — the one curve that could fall across three alive
steps). THE FLIGHT STORY'S FINAL FORM: the lethal direction is a
ROTATING OBJECT the front tracks; the tracking has an onset (1-2
steps); the onset and the death race — where death wins first, no
concentration ever appears (org2's full-step, T158's alive-window
lesson, now with the timing account).

"""
assert anchor in t and "## T163" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e199 |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| e199 | THE CONCENTRATION'S ONSET CURVE | DONE 23:00Z (T163: ONSET-COMMON — a WHEN not a WHETHER; org-1-EARLY not org-1-only; the same curve shifted one step (0.171@t1 vs 0.427@t2); MIRABEL's drop-in texture (t1 softer than static, t2 concentrated — the rotation points away then onto); the onset-and-death race account) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-01T23:01:00Z"
st["current_experiment"] = ("e199 FOLDED (T163: ONSET-COMMON — a WHEN; the timing is the biography). Fleet: g1bS4 (GPU, "
                            "the movement-matched dose) + e200 DISPATCHING (CPU: the deepening test — e197's alive "
                            "lineage's own onset curve)."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e199 folded")
