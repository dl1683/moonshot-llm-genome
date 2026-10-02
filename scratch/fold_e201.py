# -*- coding: utf-8 -*-
"""Fold e201: NOTES, T166, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e201 — the front-rotation census: ALTERNATION-UNIVERSAL (n=3) — the natural washes rotate the same way; the rotation outlives the organism (2026-10-02 ~08:45Z) — DONE

WHAT WE DID: a desk check on committed data (17 cross-file
identity gates; 7 parent files) plus ONE gated fresh read
(MIRABEL's post-death u3 — 23.7s CPU, threads 4, load-probed;
e198's walk rebuilt under its full gate chain, TEXTURE tiers
stamped); the owner envelope's perfect cell.

WHAT WE SAW (T166): EVERY ALIVE CONSECUTIVE-FRONT PAIR IS
ANTI-CORRELATED below the -0.10 bar on ALL THREE organisms: org1
(u0,u1) -0.1545; MIRABEL (u0,u1) -0.1751 and (u1,u2) -0.1781; the
half-step anchor -0.263/-0.313/-0.341/-0.353. THE ALTERNATION IS
NOT THE WASH'S TEXTURE — the natural washes rotate the same way;
T164's rotation-alternates account and the bleed/front phase
picture STAND AT n=3. POST-DEATH CONTEXT (never adjudicated): the
alternation PERSISTS PAST DEATH on every lineage (org1 -0.203 +
the committed continuation; MIRABEL (u2,u3) -0.2204, the one fresh
read) — THE ROTATION OUTLIVES THE ORGANISM: the front's
alternation is a property of the WASH TRAJECTORY, not of the
living fact's response. ROOT-GRADIENT OVERLAPS (root-point
stated): org1 0.595/-0.162/0.034; MIRABEL 0.600/-0.174/-0.021/
+0.007; half 0.547/-0.269/0.148/-0.077/-0.030. HONESTY: a shape
claim at n=3 biographies, not a mechanism proof; the half-step
anchor's counterfactual caveat rides (org1/MIRABEL are the
natural trajectories — which is what makes the verdict about the
natural washes).
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T165 —"
card = """## T166 — e201: the rotation is the wash's — alternation universal, and it outlives the organism (2026-10-02 ~08:45Z)

The census licenses the phase picture: EVERY alive consecutive-
front pair anti-correlates on all three organisms — the natural
washes rotate exactly as the counterfactual lineage does. THE
DEEPER READ IS THE POST-DEATH PERSISTENCE: the alternation
continues past death on every lineage on record — THE ROTATION
OUTLIVES THE ORGANISM. The front's alternation belongs to the WASH
TRAJECTORY (the stream's gradient sign-structure turning over),
not to the fact's fleeing response; what the LIVING organism adds
is only the DEATH TIMING (T164: death = the rotation's deepest
landing on the support). THE PICTURE, FINAL FORM: the wash drives
a rotating lethal front; the fact dies when the rotation lands on
its support deepest; the bleed survives by turning its own steps
away from each landing; the rhythm (a managed bleed) survives by
re-injecting the right gradients at the right times. THE
OPENNESS: a shape claim at n=3; the mechanism candidates (why the
stream's gradient structure alternates) unnamed — the next
dissection question, ripening.

"""
assert anchor in t and "## T166" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e201 |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| e201 | THE FRONT-ROTATION CENSUS | DONE 08:45Z (T166: ALTERNATION-UNIVERSAL at n=3 — every alive pair anti-correlated (org1 -0.155; MIRABEL -0.175/-0.178; half -0.26..-0.35); the alternation persists PAST DEATH on every lineage — the rotation outlives the organism; the front's alternation belongs to the wash trajectory; the phase picture stands) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T08:46:00Z"
st["current_experiment"] = ("e201 FOLDED (T166: ALTERNATION-UNIVERSAL at n=3 — the rotation outlives the organism). "
                            "Fleet: g1bS5 (GPU, gated) + R61 duo DISPATCHING (auditor+critic; the ideator skipped "
                            "with the deviation noted — the queue self-generates from the follow-on chains)."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e201 folded")
