import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

t = open("THINKING.md", encoding="utf-8").read()
w_anchor = "## W015 — WONDER:"
w016 = """## W016 — WONDER: born with one organ — the sink is the native memory substrate; addresses are protocol-grown grafts (2026-09-28 ~09:00Z)

T085's deepest reading, savored: the architecture comes with
EXACTLY ONE memory organ from birth — the omnipresent row, the
sink — and every 'address' the lab ever dissected was a graft
grown by a protocol that pinned the fact's position. The five-
day arc dissected the graft (row 129: its mass law, family
typing, recency multiplier, removal surgicality) and nearly
missed the organ. The biological inversion is delicious: what
the lab called the hippocampus (specific, context-bound,
cheaply installed, surgically removable) is the GRAFT — the
trained-in structure; what it called the cortex (the field, the
schematic store) was the native organ all along, present in
every context from step zero. PREDICTED SAVORS: (a) train ANY
new association on an untrained random net with natural
placement and it should land row-0-dominant immediately (the
e142 fresh-family result generalized — trivially testable on an
untrained seed); (b) the share law's r*(k)*k should reprice
differently in row-0-only nets (natural installs) vs graft-carrying
protocol installs — e134's second fact, if installed naturally,
tests whether the constant prices the ORGAN or the GRAFT+organ
system; (c) the parked GPT-2 replication gets its sharpest-ever
first question: does pretrained GPT-2's factual recall show
row-0 presence-keying under the e141/e150 instruments? If yes,
the sink-route is an architecture-scale phenomenon, not a tiny-
net curiosity. CAVEAT (R45's ghost): all presence claims carry
the flat-CE bound until e150 lands; this card's 'native organ'
language inherits it.

""" + w_anchor
assert w_anchor in t
t = t.replace(w_anchor, w016, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# e146 instrument sharpening (avoid the rel-1.000 ceiling family)
q = open("QUEUE.md", encoding="utf-8").read()
import re
m = re.search(r"^\| e146 \|[^\n]*\n", q, re.M)
assert m
q = q[:m.start()] + "| e146 | THE DISSOCIATION MATRIX — does the self survive losing its pivot? (W015; connects the memory and self arcs) | READY (CPU eval-only; after e150) | interventions x functions: {row-0 presence-removal, row-0 direction-scramble, fact-specific-head ablation} x {fact expression, self-recognition (e111 self/other battery + k*=7 occupancy), corpus CE}. [T085 instrument note: use the B43 line (rel 0.878) NOT the e098 fresh family (rel 1.000 — the row-0 dial saturates there by construction, e140's lesson); read fact column at NOVEL geometry primary; dose columns for the self (norm ladder + scramble dose); fourth outcome SELF-INDEPENDENT added per R45 ideator]. Bars: SELF-ROUTED = self collapses under presence-removal; SELF-CONSTITUTIONAL (predicted) = self survives presence-removal, dies under direction-scramble; SELF-TENANT = self dies under fact-head ablation; SELF-INDEPENDENT = survives all three |\n" + q[m.end():]
open("QUEUE.md", "w", encoding="utf-8").write(q)

# paper framing registration
p = open("scratch/paper_integrity_r44.md", encoding="utf-8").read()
p += """
## Addendum (2026-09-28 ~09:00Z, post-T085): P-A framing pass owed.

The draft's numbers survive, but T085 (ROW-0-ALWAYS; address =
protocol-made graft) reframes the LANGUAGE: 'the address faculty'
(abstract, intro, 5.1-5.3) should become 'protocol-sculpted
addressability' wherever it is stated as a net property rather
than a protocol product — the experiments themselves (masked-
replay installs) are unaffected, their interpretation's noun is
not. One careful pass over abstract + intro + section 5 headers
when the second-paper skeleton settles (both papers share the
noun).
"""
open("scratch/paper_integrity_r44.md", "w", encoding="utf-8").write(p)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("W016 + e146 sharpening + paper framing registered")
