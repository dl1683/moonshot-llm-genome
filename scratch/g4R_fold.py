import json, datetime, re
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

n = open("NOTES.md", encoding="utf-8").read()
o = "## g1bR — the wall seed replicates:"
entry = """## g4R — the compass + knife replicates: HONEST BOUNDS both — the SUBSTANCE replicates 3/3; the registered bars' specifics don't (the row band slid; the census-selection rule picks load-bearing heads) (2026-09-29 ~20:50Z) — DONE

WHAT WE DID: 2 fresh install draws per claim on g4's surviving
roots (bit-gated); 8 GPU trainings; censuses CPU.

WHAT WE SAW (T134): COMPASS — HONEST BOUND: r1 PASS (site
+0.376 at rows 5-13, A inert); r2's band FAIL — but the arm
learned at ceiling (0.9994) with the content at rows 3-4
(+0.82): THE SITE SLID LEFT of the registered band. THE
SUBSTANCE IS 3/3: P-floor positional placement, A-floor inert
every time, COMPASS-CONTENT never threatened. What's
draw-brittle is the ROW BAND, not the placement law. KNIFE —
HONEST BOUND: the per-census selection rule (e160 top-2)
picked load-bearing heads both times (kills at +0.97/+1.85 CE
— kills but NOT flat); g4's LITERAL headset {L1H2,L3H0}
flat-CE-killed BOTH new nets (88.8% @ +0.170; 86.7% @ +0.177;
dCE stable to ~0.01 across three installs) — THE KNIFE
REPLICATES WITH THE COMMITTED HEADSET 3/3; the census-RANKING
generality does not (it ranks by drop and finds organism
pillars like L0H3). THE SPINE 4/4: all four new installs
landed carrier P, replicating the pre-teaching prediction.
CLAIMS' FATES: both keep scoped forms — compass-positional
(3/3 substance, draw-brittle band); headset-specific knife
(3/3 flat-CE, selection-rule bounded). Neither mints
unqualified per R55. Honesty: install-draw only (root n=1 —
T113 stands); no wash cell; the fork reported not adjudicated.

---

""" + o
assert o in n
n = n.replace(o, entry, 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

t = open("THINKING.md", encoding="utf-8").read()
anchor = "## T133 — g1bR:"
t134 = """## T134 — g4R: the substance replicates where the bars don't — honest bounds as findings (2026-09-29 ~20:50Z)

Both g4 positives return honest bounds, and each bound TEACHES:
the compass's row BAND is draw-brittle (r2's site slid to rows
3-4 at full strength — the placement law held, the registered
window didn't; the band is a convention of one draw, not a
property), and the knife's census-RANKING rule fails because it
selects by damage — and damage-ranking finds organism pillars
(L0H3 at ~2 nats ablation cost), not fact-specific circuits.
The committed headset kills flat 3/3. THE DEEPER LESSON (both
bounds share it): the SELECTION RULES are the brittle layer;
the PHENOMENA are robust. The spine's install prediction went
4/4 — the architecture predicted its own carrier every time it
was asked. CLAIMS' FINAL FORMS: compass-positional (3/3
substance, one-family, band-scoped); headset-specific knife
(3/3 flat-CE); the wall and the rhythm at n=3 law grade; the
cone and two-site at n=1 (g3R the remaining replicate).

""" + anchor
assert anchor in t
t = t.replace(anchor, t134, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

q = open("QUEUE.md", encoding="utf-8").read()
o_q = "| g1bR | THE WALL SEED REPLICATES"
row = """| g4R | THE COMPASS + KNIFE REPLICATES | DONE 20:50Z (T134: HONEST BOUNDS both — the substance 3/3 (P-placement every draw, A inert every draw, the headset flat-CE 3/3); the bars' specifics draw-brittle (the row band slid to 3-4; the census rule selects organism pillars); the spine 4/4) |
""" + o_q
assert o_q in q
q = q.replace(o_q, row, 1)
open("QUEUE.md", "w", encoding="utf-8").write(q)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 0. g4R DONE: honest bounds both — substance 3/3, selection rules brittle. Remaining: g3R (cone seeds), the base-seed redraw, g6, the paper assembly."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("g4R fold complete")
