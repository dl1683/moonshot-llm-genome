import json, datetime, re

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- NOTES ----------
n = open("NOTES.md", encoding="utf-8").read()
o = "## E154 — two facts, one door:"
entry = """## E161 — the freeze-cell: DISUSE/GENERIC-PRESSURE — the door closes with NO fact teaching; the ENTIRE fact dissolves under ordinary gradient flow (2026-09-28 ~14:45Z) — DONE

WHAT WE DID: 300 steps of plain-corpus continuation from the s32
dwell peak (zero name leakage verified at draw time); full dial
trajectory; gates bit-near.

WHAT WE SAW (T101): DISUSE fires CLEAN — g-12: 0.513 -> 0.040 at
+50 -> 0.0036 at +300 (128x under the bar; no fact teaching, no
graft, no anchors). COMPETITIVE dead (g never near 0.7);
DWELL-PERSISTS dead. THE BIGGER TEXTURE: this is NOT selective
door closure — the ENTIRE fact expression dissolves (g0, g+12,
the functional site read 0.965 -> 0.012, the brake -0.373 ->
+0.001, the row-0 sink 0.101 -> 0.002) while CE stays healthy
(1.61-1.69). THE DWELL-PHASE MEMORY IS NOT YET INCORRIGIBLE:
plain gradient flow washes it out in <50 steps — where e125a's
300-step endpoint site-memory survived everything. CONSOLIDATION
= GRADIENT-RESISTANCE ACQUISITION (the s32->s300 axis), and the
e151 'conversion' re-reads as: F1 washing out (as any
unconsolidated memory does under continued training) WHILE the
re-taught fact builds its graft. The 'phase switch' was
substantially FORGETTING + NEW LEARNING. THE MISSING CONTROL
(e176, queued): freeze the FULLY-CONSOLIDATED ROOT on plain
corpus — if the root's fact survives, consolidation really is
resistance-acquisition (the CLS story, licensed); if it too
dissolves, even 'consolidated' is use-it-or-lose-it and the
edifice reframes again. Honesty: single seed/trajectory; the
collapse margin (128x) dwarfs the known scatter; one
distribution tested (the anchor+random stream); s32 starting
point is one point on one trajectory.

---

""" + o
assert o in n
n = n.replace(o, entry, 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

# ---------- THINKING ----------
t = open("THINKING.md", encoding="utf-8").read()
anchor = "## T100 — [R49:"
t101 = """## T101 — E161: disuse, and the reframing it forces — consolidation as gradient-resistance acquisition (2026-09-28 ~14:45Z)

The fork resolves DISUSE cleanly, and the honest consequence is
the day's largest reframing since the tautology: THE CONVERSION
WAS SUBSTANTIALLY FORGETTING. The dwell-phase memory washes out
under plain corpus in <50 steps — all geometries, the site, the
brake, the sink together — while the organism stays healthy.
There was no switch thrown by teaching; there was an
UNCONSOLIDATED memory being washed by ordinary optimization
pressure while a new one was trained in. What remains genuinely
switch-like: e147's cliff (variance builds geometry-general
access that locked training does not) and e125a's endpoint
incorrigibility. The unified honest story: memories lie on a
GRADIENT-RESISTANCE axis — fresh installs and dwell-phase
memories wash out under any continued training; deep
site-stores (e125a's endpoint) and (pending e176) consolidated
memories resist. THE DECISIVE CONTROL (e176): freeze the
FULLY-CONSOLIDATED ROOT. SURVIVES => consolidation IS
resistance-acquisition (CLS licensed, the paper's claim 2
rewrites to the resistance axis); DISSOLVES => even
'consolidated' is use-it-or-lose-it (the anchor half of every
past fine-tune was quietly maintaining the fact — every
'experiment' was also a rehearsal).

FOR THE PAPER: claim 2's phase language converts to the
resistance axis (phases -> degrees of washout-resistance; the
cliff survives as the ACCESS-building fact; e151 re-reads as
washout-plus-rebuilding). The abstract's "bidirectionally
switchable" dies its final death here — the closing direction
was forgetting. THE SAVOR: every memory the lab ever trained
was being secretly rehearsed by the anchor banks in every
subsequent fine-tune — the lab's own protocol was the memory's
life-support, and e161 is the first time anyone turned it off.

""" + anchor
assert anchor in t
t = t.replace(anchor, t101, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- queue: e176 ----------
q = open("QUEUE.md", encoding="utf-8").read()
o_q = "| e175 | THE RECOVERY-KINETICS CELL"
row = """| e176 | FREEZE THE ROOT (T101's decisive control — does the FULLY-CONSOLIDATED fact survive plain corpus?) | DISPATCHED ~14:50Z (CPU; one training) | 300 steps plain-corpus continuation from e131_consolidated (e161's protocol verbatim; zero name leakage verified). Bars: CONSOLIDATION-IS-RESISTANCE = the fact survives (g-12 >= 0.5, g0 >= 0.5 at +300 — CLS licensed; the resistance axis is real; claim 2 rewrites to it); USE-IT-OR-LOSE-IT = the fact dissolves (matching e161's trajectory — even consolidated memories need the anchor rehearsal; every past fine-tune was secretly maintaining its facts) |
""" + o_q
assert o_q in q
q = q.replace(o_q, row, 1)
m = re.search(r"^\| e161 \|[^\n]*\n", q, re.M)
if m:
    q = q[:m.start()] + "| e161 | the freeze-cell | DONE 14:45Z (T101: DISUSE clean — door closes with no teaching, 128x under bar; the ENTIRE fact dissolves under plain gradient flow; the dwell memory is not yet incorrigible; the conversion re-reads as forgetting + new learning; e176 freeze-the-root is the decisive control) |\n" + q[m.end():]
open("QUEUE.md", "w", encoding="utf-8").write(q)

# ---------- paper: claim 2 honest form ----------
p = open("scratch/day6_paper_skeleton.md", encoding="utf-8").read()
o_p = "bidirectionally switchable at fixed architecture (both directions are 300 trained steps; conversions demonstrated across the lineage — same-net reversibility is e155R's cell)"
n_p = "REWRITTEN per e161/T101: memories lie on a GRADIENT-RESISTANCE axis — unconsolidated (dwell-phase) memories wash out under ANY continued training (e161: plain corpus dissolves the whole fact in <50 steps); variance training builds geometry-general access (the cliff survives); the 'closing' direction was substantially forgetting [e176 = the decisive root-freeze control]"
assert o_p in p
p = p.replace(o_p, n_p, 1)
open("scratch/day6_paper_skeleton.md", "w", encoding="utf-8").write(p)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 3: e152R (GPU) + e170 (anchor-neutral) + e173 (partition) + e176 (freeze-the-root — T101's decisive control, dispatching). e161 DONE: DISUSE — the conversion was substantially forgetting."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("e161 fold complete")
