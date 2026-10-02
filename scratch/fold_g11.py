# -*- coding: utf-8 -*-
"""Fold g11: NOTES, T196, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## g11 — the crush mechanism: CARRIER-CONCENTRATED at the boundary; the crush is NOT first-order-directional (the registered falsifier fired: the locked root's first step is the MORE erase-aligned yet survives); the readable residue: the nonlinear interaction between each root's rotated support and the shared wash hot set (2026-10-02 ~19:00Z) — DONE

WHAT WE DID: at the three cons roots (locked 10901; g1e's 10912;
g1f's 10913 — provenance-gated; the committed first-step deltas
loaded bit-exact; the settled +1 reads reproduced to 2.4e-7): the
first wash gradient vs each root's own sensitivity; the ball
geometry; the delta's composition vs the carrier sets. 18s CPU.

WHAT WE SAW (T196): (1) ERASE-ALIGNED REVERSED — cos(g_0, -s_0):
locked +0.0986 > cons -0.0962/-0.0211 — THE LOCKED ROOT'S FIRST
STEP IS THE MORE ERASE-ALIGNED YET SURVIVES (0.945 vs 0.27/0.26):
the crush is NOT first-order-directional (the registered
falsifier); (2) CARRIER-CONCENTRATED FIRES AT THE BOUNDARY — top-
2000 overlap: locked 14/2000 vs cons 23 (1.64x) and 21 (exactly
1.50x), disclosed with the companions: the L1 mass on carriers at
the k/P null for EVERY root; the overlap 0.0 at k <= 1000; the
ladder not uniformly clearing 1.5x — A WEAK k=2000-SHAPED SET
PREFERENCE, NOT A CONCENTRATION; (3) THE FREE GEOMETRY: the first
deltas share 55% of their top coords across roots (the wash's hot
set is ROOT-INDEPENDENT) while the carriers ROTATE (mutual
overlap 0.23-0.33) — THE CRUSH'S READABLE RESIDUE IS THE
NONLINEAR INTERACTION between each root's rotated support and the
shared hot set; (4) the cons roots' first wash gradients are
20-34% larger in norm (clip binds there, not at the locked). THE
INTERVENTION NAMED (for whenever): the first step with its
s_0-component removed; or the cons delta applied at the locked
root. HONESTY: n=1 per root; the carrier proxy; the T157 class
(no intervention — a firing names where to intervene, proves
nothing); the T178 strength confound co-noted.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T195 —"
card = """## T196 — g11: the crush is a rotated-support x shared-hot-set interaction (2026-10-02 ~19:00Z)

The mechanism cell answers with the falsifier: the locked root's
first step is MORE erase-aligned yet survives — first-order
direction is ruled out. The spatial signature fires only at the
boundary (a weak k=2000-shaped set preference, mass at the null)
— honest but thin. THE REAL FINDING IS THE GEOMETRY: the wash's
first gradient is essentially THE SAME OBJECT at every root (55%
shared top coordinates — the wash has a root-independent hot set)
while each root's fact support sits ROTATED relative to it
(overlap 0.23-0.33). The crush depth is therefore NOT a property
of the gradient or the support alone but of their RELATIVE
GEOMETRY — the nonlinear interaction — plus a norm asymmetry (the
cons roots' first gradients 20-34% larger, the clip binding
differently). THE WALL'S MECHANISM STORY, FINAL FORM: the
consolidation stream decides the support's ORIENTATION relative
to the wash's fixed hot set; the first projected step's damage is
the interaction; the wall re-captures by +2 whatever the
interaction costs. THE INTERVENTION NAMED (the s_0-component-
removed first step; the cons delta at the locked root) — the
cheap decisive cell for whenever the question is wanted again.

"""
assert anchor in t and "## T196" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
q = q.replace("| DISPATCHED 18:20Z — bars: ERASE-ALIGNED / CARRIER-CONCENTRATED / PROJECTION-NEUTRAL / GRADED |",
              "| DONE 19:00Z (T196: CARRIER-CONCENTRATED at the boundary; ERASE-ALIGNED REVERSED (the registered falsifier — the locked root more aligned yet surviving); the deltas share 55% of top coords (the wash's hot set root-independent) while the carriers ROTATE — the crush a rotated-support x shared-hot-set interaction; the intervention named) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T19:01:00Z"
st["current_experiment"] = ("g11 FOLDED (T196: the crush is a rotated-support x shared-hot-set interaction - the "
                            "mechanism story final form). Fleet: e221 (the compositionality test) alone."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("g11 folded")
