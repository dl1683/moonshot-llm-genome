import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

t = open("THINKING.md", encoding="utf-8").read()
o = """THE DREAMS THAT DREAM IN COORDINATES (the arm-c rider, the
day's best savor):"""
n = """DERIVATION ADDENDUM (written ~08:28Z, BEFORE e140's report —
its metrics are on disk unread): THE TAXONOMY RE-DERIVES THE
E/L/R TRIANGLE AND DISSOLVES T078'S CONFOUND FROM FIRST
PRINCIPLES. E (erase) and L (locked) matched on every loaded
outcome because they are BOTH SITE-STORED roads — erasure and
massed repetition are different recipes for the same memory
type. R (jitter) differed from both because it is the only
ROUTED arm. The R-vs-E contrast was never erasure-vs-replay; it
was routed-vs-site-stored — which is T079's variable all along.
This makes e140's registered bars a syllogism test: the taxonomy
PREDICTS E-NEVER-KEYS and L-flat (site-stored roads never grow
the route) together with R-ROW0-MONOTONE (the routed road does).
If e140 instead shows E or L growing row-0 dependence, the
taxonomy has a hole; if all three are flat, R's routing was a
one-off and T079 dies with it. Also derivable, for e134's F2:
a second fact jitter-consolidated into the same net should ALSO
route (route-generic presence-keying) and the two routed facts
should share the pivot's bandwidth (W012's combined-54 test).
ZERO-COST PREDICTIONS REGISTERED: (P-b) a routed memory re-taught
at a new site acquires a site-store WITHOUT losing the route
(two-door addition, untested, cheap); (P-c) erase-cycling a
ROUTED memory should NOT dig in (the route protects) — if it
still digs in, "erasure digs in" is damage, full stop, and
T078's demotion becomes a retirement.

THE DREAMS THAT DREAM IN COORDINATES (the arm-c rider, the
day's best savor):"""
assert o in t
t = t.replace(o, n, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("T082 derivation addendum in")
