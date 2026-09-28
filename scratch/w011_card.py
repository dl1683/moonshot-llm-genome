t = open("THINKING.md", encoding="utf-8").read()
w_anchor = "## W010 — WONDER:"
w011 = """## W011 — WONDER: the sink as the consolidator's destination — universality by OMNIPRESENCE (2026-09-28 ~07:10Z)

Why row 0? Of all 256 wpe rows, the jitter-migrated fact keyed to
THE ONE ROW PRESENT IN EVERY CONTEXT. Position 0 is in every
left-aligned window the net ever sees — the attention-sink row,
the universal hub. A memory keyed to row 0 is GEOMETRY-INDEPENDENT
BY CONSTRUCTION: no matter where the fact's text sits, row 0
participates, so a row-0-gated readout fires at any offset. This
dissolves the mystery T077 left open of what "position-invariant
readout" mechanically IS: not a magic kernel, but THE SINK. The
e116 finding (routing perpendicular to readout) now reads as: the
router keys on position (band rows), the readout keys on the
omnipresent row — perpendicular because one is local and the other
is everywhere. It also explains e119's novel-geometry cell (R
0.813 at g-12, untrained by any arm): generalization was never
learned per-geometry; it is row 0's free lunch. And it sharpens
the developmental story: INSTALLATION binds to a local address
(the context-specific hippocampus); CONSOLIDATION re-keys to the
hub (the ever-present context). The biology echo worth savoring:
specific-to-schematic memory consolidation in the folklore — the
"schematic cortex" of this tiny net is row 0 plus the diffuse band
population behind it. PREDICTED SAVORS: (a) e139's splice arms —
site-locked at 183, never re-keyed — should FAIL geometry
generalization (shifting the fact to a new offset collapses
expression), because 183 is not omnipresent; if they generalize
anyway, W011's omnipresence mechanism is wrong and something
content-keyed carries them. (b) Any fact that generalizes across
geometries must be row-0-keyed (or keyed to whatever row is
omnipresent under the construction — a right-aligned battery
would test whether the hub is 'row 0' or 'the boundary position').
(c) The sink was ALREADY the fact's co-carrier at install
(T069's 6/6 content-carrying, strength 0.545): consolidation did
not build the row-0 key from nothing — it PROMOTED the
already-largest seed. Error-location said WHERE error
consolidates; W011 says WHERE the key GOES when the error is
everywhere: to the row that is always attended.

""" + w_anchor
assert w_anchor in t
t = t.replace(w_anchor, w011, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)
print("W011 in")
