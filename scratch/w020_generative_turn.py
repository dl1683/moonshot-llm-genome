import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

t = open("THINKING.md", encoding="utf-8").read()
w_anchor = "## W019 — WONDER"
w020 = """## W020 — THE GENERATIVE TURN (the user's standing directive, 2026-09-29 ~12:40Z: from dissection to synthesis — design architectures that test our laws)

The lab's findings are now laws-in-waiting: no-basin memory
(basin ~2.5-5 L2; exit t* ~ lr^-1.16), the error compass,
the variance switch, circuit-selective surgery, the
resurrection economy (9 events; one revival; sticky re-entry).
The dissection has earned the right to ask the generative
question: ARE THESE ARCHITECTURAL NECESSITIES OR CONTINGENT
FACTS OF THE PRE-LN TRANSFORMER? Every law we believe becomes
a DESIGN SPEC for an architecture that should break or embody
it. THE PROGRAM (g-series): g1 BASIN-WIDENING, g2
REHEARSAL-NATIVE, g3 GENERATIVE MEMORY, g4 COMPRESSIBILITY.
Each carries registered predictions IN ADVANCE; wrong
predictions are the point — every law that fails in a new
architecture was contingent; every law that holds is closer
to necessary.

""" + w_anchor
assert w_anchor in t
t = t.replace(w_anchor, w020, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)
print("W020 in")
