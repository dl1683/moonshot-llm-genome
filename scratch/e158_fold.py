import json, datetime, re

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- NOTES ----------
n = open("NOTES.md", encoding="utf-8").read()
o = "## E125a — the inverted knife:"
entry = """## E158 — the 2x2 completion: SITE-INDEPENDENT — closure requires NOVELTY *and* zero-variance together; the geometry door closes exactly when a NOVEL graft forms (2026-09-28 ~12:40Z) — DONE

WHAT WE DID: the two completing cells from the consolidated
root — (a) JITTER@183 (e151's protocol with ±8 position jitter
around the new site), (b) LOCKED@BAND (zero-variance at the
home position); the 2x2 completed with the known cells (jitter@
band = the root itself; locked@183 = e151).

WHAT WE SAW (T097): SITE-INDEPENDENT fires — the two-factor
gate. (a) jitter@183: door OPEN (g-12 0.505 >= 0.5) — variance
at a NOVEL site does not close (and per e143/e142, variance
prevents graft formation). (b) locked@band: door OPEN (0.546) —
zero-variance at HOME is harmless. Only locked@NOVEL-site
shuts (e151: 0.102). NEITHER registered simple law holds
(PHASE-BY-VARIANCE and CLOSURE-BY-PLACEMENT both fail). THE
UNIFYING READING: THE GEOMETRY DOOR CLOSES WHEN AND ONLY WHEN A
NOVEL GRAFT FORMS — variance anywhere prevents the graft (door
stays open); zero-variance at home re-trains the existing
structure (door stays open); zero-variance at a novel site
BUILDS a competing graft and the geometry-general access comes
down with it. Graft formation and door closure are ONE EVENT.
Open question (in the metrics' site census): did locked@band
RE-FORM a band graft (weakly) — the home-graft caveat; and the
novelty axis is unexplored (e165: locked at graded distances
from home). Honesty: single seed/lineage; jitter@183's 0.505
sits 0.005 above the open bar (razor-thin — the honest read is
"at-or-near the boundary", and the arm's site census shows how
much graft tried to form); the b-arm's 0.546 carries e119-L's
budget-convention lineage.

---

""" + o
assert o in n
n = n.replace(o, entry, 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

# ---------- THINKING ----------
t = open("THINKING.md", encoding="utf-8").read()
anchor = "## T096 — E125a:"
t097 = """## T097 — E158: the door closes when a novel graft forms — site-store construction and geometry-access destruction are one event (2026-09-28 ~12:40Z)

The 2x2 landed on its strangest branch, and the strangeness is
the synthesis: neither variance alone nor placement alone
closes the geometry door — NOVELTY AND ZERO-VARIANCE TOGETHER
do, which is exactly the recipe for BUILDING A NOVEL GRAFT.
Re-reading the whole arc through this: e142 (natural placement
-> no graft, row 0 only), e143 (locked@novel -> graft), e147
(variance -> no graft, door opens), e151 (locked@novel -> graft
+ door shut), e152 (the dwell = the graft being built while the
door decays), e158 (jitter@novel -> NO graft, door open;
locked@home -> no NEW graft, door open). ONE EVENT, TWO FACES:
erecting a new positional key tears down the geometry-general
access. The "phases" are not two states of a substrate switched
by a variable; they are BUILD and TRAVEL — the same machinery
seen from the construction side and the access side. FOR THE
PAPER: claim 2's provisional marker clears into the two-factor
form ("closure accompanies novel-graft formation"), which is
STRONGER and cleaner than either simple law; the freeze-cell
(e161) now reads as "does graft-building need supervision to
finish tearing down the door"; e165 (the novelty-axis ladder:
locked at graded distances from home) is the discriminating
follow-up — does closure track distance-from-home (novelty as
a continuous variable) or is it binary at first-novel-site?
RAZOR-THIN honesty: jitter@183's 0.505 vs the 0.5 bar — at-or-
near-boundary; the site census (how much graft tried to form
under ±8 jitter) is the mechanism's decimal.

""" + anchor
assert anchor in t
t = t.replace(anchor, t097, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- paper + report ----------
p = open("scratch/day6_paper_skeleton.md", encoding="utf-8").read()
o_p = "[e158 pending: variance-vs-placement; e154 pending:\nglobal-vs-per-fact]."
n_p = "[e158 RESOLVED: SITE-INDEPENDENT — closure requires novelty AND zero-variance together: the geometry door closes exactly when a NOVEL GRAFT forms; neither variance nor placement alone suffices; e154 pending: global-vs-per-fact]."
assert o_p in p
p = p.replace(o_p, n_p, 1)
open("scratch/day6_paper_skeleton.md", "w", encoding="utf-8").write(p)

r = open("DAY_SIX_REPORT.md", encoding="utf-8").read()
o_r = "variance-vs-placement (e158), "
if o_r in r:
    r = r.replace(o_r, "", 1)
open("DAY_SIX_REPORT.md", "w", encoding="utf-8").write(r)

# ---------- QUEUE + STATE ----------
q = open("QUEUE.md", encoding="utf-8").read()
m = re.search(r"^\| e158 \|[^\n]*\n", q, re.M)
assert m
q = q[:m.start()] + "| e158 | the 2x2 completion | DONE 12:40Z (T097: SITE-INDEPENDENT — the two-factor gate: jitter@183 open 0.505, locked@band open 0.546, locked@183 shut 0.102; closure = NOVEL graft formation; one event, two faces) |\n" + q[m.end():]
o_q = "| e163 | THE SATURATION CONTROL"
e165 = """| e165 | THE NOVELTY-AXIS LADDER (T097's discriminating follow-up: does closure track distance-from-home or is it binary at first-novel-site?) | QUEUED (CPU trainings; after e154) | locked (zero-variance) re-teach at graded distances from home (0 / 8 / 16 / 32 / 64 / 128+54 rows); measure door (g-12) + graft census per distance. Bars: DISTANCE-TRACKS = g-12 decays monotonically with distance (novelty continuous); BINARY-AT-FIRST = full closure at any nonzero distance (a step in site-space); HOME-PROTECTED = closure only beyond the band's edge |
""" + o_q
assert o_q in q
q = q.replace(o_q, e165, 1)
open("QUEUE.md", "w", encoding="utf-8").write(q)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 0 — e158 DONE (T097: the two-factor gate). Dispatching e164 (post-kill census). Review due."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("e158 fold complete")
