# -*- coding: utf-8 -*-
"""Fold e182c: NOTES, T149, T123 amendment, paper gap, QUEUE DONE, STATE."""
import io, json, re

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e182c — the forgetting control: FORGETTING-GENERIC — T123's GPT-2 erosion is GENERIC forgetting; the surgical signature retires at 124M; a template-locus hint (2026-10-01 ~15:40Z) — DONE (phase 1)

WHAT WE DID: fourth dispatch (the lean brief — three predecessors
died pre-artifact; first artifacts in minutes, progressive
commits). THE PREMISE CORRECTION (disclosed pre-compute): e182
saved NO wash states (time-capped); the agent REPLAYED e182's
frozen 5e-5 wash on CPU fp32 (corpus asserted exactly equal:
331770 tokens / 664 lines; seed 18202; same AdamW recipe) — the
replay is bit-tight vs e182's record (fact-battery dp <= 0.0011;
ppl ratios within 0.03%), which also discharges the deferred
CPU/GPU-numerics item. Matched held-out named-entity cloze
controls (n=12, e182's gate verbatim, zero corpus contamination;
R0 0.727 vs facts' 0.797) evaluated at t=0/+10/+50/+80.

WHAT WE SAW (T149): FORGETTING-GENERIC at both probed depths —
fact decline 0.429 vs control 0.397 at +80 (ratio 0.92, inside the
1.5x band) and 0.339 vs 0.281 at +50 — while bank ppl improves
71.3 -> 34.8. THE SUPERVISOR'S CARRIED OBJECTION (check-ins 10-12,
asked across nine check-ins) RESOLVES AGAINST THE LAB'S OWN CLAIM:
T123's erosion at 124M is GENERIC forgetting; the GPT-2 clause
scopes to "ordinary forgetting with improving perplexity", NOT a
no-basin signature. TWO TEXTURE GEMS: (a) the near-related
co-report (same capital-of template, disjoint US entities)
collapses FASTEST (retention 0.234 at +80) — erosion may live
partly at the TEMPLATE/few-shot-following level, not knowledge
storage; (b) control erosion is heterogeneous (founders/
unique-anchor items hold 0.78-0.94; product items collapse
0.18-0.31) — probe-type structure in forgetting. HONESTY: phase-1
only; n=1 lineage, one seed, one corpus draw; controls
brand-flavored; per-state weights saved (the discipline e182
lacked). PHASE 2 (fresh corpus draws, GPU) owns generality.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T148 —"
card = """## T149 — e182c: the surgical signature dies at 124M — generic forgetting, and the template-locus hint (2026-10-01 ~15:40Z)

FORGETTING-GENERIC fired at both depths: matched held-out controls
erode WITH the installed fact (ratio 0.92 at +80; 0.83 at +50) under
the bit-tight replayed wash, while perplexity improves throughout.
The supervisor's objection — carried across nine check-ins —
resolves against the lab's own claim: THE SURGICAL SIGNATURE
RETIREES AT 124M. What survives of T123: the direction-only
transfer (the corrected title) and the time-constant texture; what
dies: "no-basin signature" as a GPT-2 claim. The honest GPT-2
clause: "ordinary forgetting with improving perplexity" — itself a
nontrivial texture (adaptation and erosion co-occur), but not the
tiny-nets' law. THE TEMPLATE-LOCUS HINT is phase-2's gift: the
near-related battery (same cloze template, disjoint entities)
collapses FASTEST (0.234) — erosion may live at the few-shot-
following/template level rather than knowledge storage; and the
controls' heterogeneity (founders hold 0.78-0.94, products collapse
0.18-0.31) says probe-TYPE structures the forgetting. THE META-
NOTES: four dispatches died for this cell; the lean brief (smoke
first, commit at every stage) got it home — the disruption era's
dispatch pattern. And the replay-that-was-necessary incidentally
discharged the CPU/GPU numerics debt (bit-tight). T123's amendment
follows; the paper's GPT-2 clause rewrites in the fold.

"""
assert anchor in t and "## T149" not in t
t = t.replace(anchor, card + anchor, 1)

m = re.search(r"^## T123 — .*$(.*?)(?=^## T12\d)", t, re.M | re.S)
assert m, "t123"
body = m.group(1).rstrip("\n")
amend = """

[E182C RESOLUTION, 2026-10-01 ~15:40Z]: THE SURGICAL SIGNATURE
RETIRES at 124M — matched held-out controls erode with the fact
(ratio 0.92 at +80) under the bit-tight replayed wash; the clause
becomes "ordinary forgetting with improving perplexity", NOT a
no-basin signature. The direction-only transfer (this card's
corrected title) and the time-constant texture survive as the
weaker form. A template-locus hint opens phase 2 (the near-related
battery collapses fastest)."""
t = t[:m.start(1)] + body + amend + "\n\n" + t[m.end(1):]
io.open("THINKING.md", "w", encoding="utf-8").write(t)

p = "scratch/day6_paper_skeleton.md"
s = io.open(p, encoding="utf-8").read()
old = "10. g3K (DONE): abstract paragraph (2) must gain the TRAJECTORY-vs-"
new = """13. GPT-2 clause (e182c DONE, phase 1): T123's erosion is GENERIC
   forgetting at 124M (controls erode with the fact, ratio 0.92,
   while ppl improves) — every GPT-2 sentence reads "ordinary
   forgetting with improving perplexity", never a no-basin
   signature; the template-locus hint and phase 2 (fresh corpus
   draws) noted in the discussion.
10. g3K (DONE): abstract paragraph (2) must gain the TRAJECTORY-vs-"""
assert old in s
s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8").write(s)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e182c |"):]; row = row[:row.index("\n")]
new_row = ("| e182c | GPT-2 forgetting control | DONE 15:40Z phase-1 (T149: FORGETTING-GENERIC — controls erode with "
           "the fact (ratio 0.92 @ +80, 0.83 @ +50) under the bit-tight replayed wash while ppl improves; the surgical "
           "signature RETIRES at 124M; the template-locus hint (nearrel collapses fastest, 0.234); the replay "
           "discharged the CPU/GPU numerics debt; phase 2 = fresh corpus draws on GPU) |")
q = q.replace(row, new_row, 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-01T15:41:00Z"
st["current_experiment"] = ("e182c FOLDED (T149: FORGETTING-GENERIC — the surgical signature dies at 124M; the "
                            "supervisor's nine-check-in objection resolved; the template-locus hint). Fleet: g2g (GPU) "
                            "+ e_chart (CPU) + x1 DISPATCHING (the neighbor's range census).")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e182c folded")
