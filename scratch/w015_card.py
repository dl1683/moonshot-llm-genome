import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

t = open("THINKING.md", encoding="utf-8").read()
w_anchor = "## W014 — WONDER:"
w015 = """## W015 — WONDER: does the self survive losing its pivot? Connecting the two arcs (2026-09-28 ~08:18Z)

The memory arc and the self arc have never touched
mechanistically. The self-recognition machinery (binary
self/other step 0.40-vs-0.14; k*=7 signature subspace; exclusion
of foreign nets; unfakeable — T060-T062) is measured through the
anchor's V-structure reads, but nobody has asked WHICH LAYER
computes it — and the week just built exactly the instruments to
ask: the fact is ROUTED through row-0 presence (e141) with
direction-independence; the LM is direction-dependent; the
memory tenant and the language tenant keep separate insurance
policies (W014). THE DISSOCIATION MATRIX (savor, three
interventions x three functions): interventions = row-0
presence-removal / row-0 direction-scramble / fact-specific-head
ablation; functions = fact expression / SELF-RECOGNITION (the
self/other binary + k*=7 occupancy) / LM corpus CE. The fact's
row is known (dies, survives, dies). The LM's row is known
(dies, dies, cheap). THE SELF'S ROW IS THE OPEN CELL AND THE
INTERESTING ONE: if self-recognition collapses under
presence-removal, the self is ROUTED — identity rides the same
pivot as memory, and W004's "self is a fixed point" becomes
"self is a fixed point OF the routed read" — the lab's two
deepest findings fuse into one substrate. If the self survives
presence-removal but dies under direction-scramble, the self is
computed UPSTREAM (at the V-manifold source) with the LM's
robustness class — identity is older than the route, a
constitutional layer the routing merely consults. If the self
dies under fact-specific-head ablation, self and fact share
readout machinery (the self is one tenant among tenants).
PREDICTED SAVOR: the middle branch — the unfakeable check reads
V-STRUCTURE, and V-structure is direction; presence never carried
structure. The self should be direction-typed like the LM but
presence-independent unlike the fact. If so, the lab earns a
three-layer stack in one run: constitutional self (direction,
upstream), routed memory (presence, downstream), and the
language function straddling both — and the unlearning
implication sharpens: you can evict a memory without touching
the self, but never scramble directions without both.

""" + w_anchor
assert w_anchor in t
t = t.replace(w_anchor, w015, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

q = open("QUEUE.md", encoding="utf-8").read()
o_q = "| e144 | FROZEN-SINK INSTALL"
row = """| e146 | THE DISSOCIATION MATRIX — does the self survive losing its pivot? (W015; connects the memory and self arcs for the first time) | READY (CPU eval-only; after e142) | interventions x functions: {row-0 presence-removal, row-0 direction-scramble, fact-specific-head ablation} x {fact expression, self-recognition (e111 self/other battery + k*=7 occupancy), corpus CE}. Bars: SELF-ROUTED = self collapses under presence-removal (identity rides the memory pivot — W004 fuses with W013); SELF-CONSTITUTIONAL (predicted) = self survives presence-removal, dies under direction-scramble (computed upstream at the V-source); SELF-TENANT = self dies under fact-head ablation (shares readout machinery with the fact) |
""" + o_q
assert o_q in q
q = q.replace(o_q, row, 1)
open("QUEUE.md", "w", encoding="utf-8").write(q)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("W015 + e146 row in")
