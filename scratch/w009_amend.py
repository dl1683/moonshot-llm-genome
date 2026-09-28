t = open("THINKING.md", encoding="utf-8").read()
o = """Registered savor, not a
bar: read the histogram's shape before reading any single row.
"""
n = """Registered savor, not a
bar: read the histogram's shape before reading any single row.

GRADUATION (e133, ~07:50Z): the frame is now TRI-LEVEL, and the
third level is the deepest. Row level: e088's overlapping supports
(pair cost 0.46x singles). Organ level: e133's joint-vs-parts
failure (0.785 vs 1.98, ~2.5x redundancy — marginals everywhere).
Route level: no single head carries the sink route — row 0's
omnipresence is realized as MANY weak value-channels (no head
sink-adjacent >= 0.25, yet deleting the route is fatal) rather
than one strong attention edge. W009's thesis, completed: the
lab has never found a discrete circuit at any level of
description — rows, organs, routes are all populations, and
every instrument that assumed discreteness (pair slots, organ
partitions, 'the' sink head) returned overlap. Discreteness was
always the metaphor's artifact.
"""
assert o in t, "W009 anchor"
t = t.replace(o, n, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)
print("W009 tri-level amendment in (correct anchor)")
