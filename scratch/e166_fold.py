import json, datetime, re

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- NOTES ----------
n = open("NOTES.md", encoding="utf-8").read()
o = "## E164 — the post-kill census:"
entry = """## E166 — the inverse event: DOOR-STAYS-SHUT — the graft rows carry exactly ZERO of the closure; the third great asymmetry lands (2026-09-28 ~13:50Z) — DONE

WHAT WE DID: 29 CE-priced cells — row restore (3 modes), head
ablations (reader3/N2, both modes), joints, root ceiling +
jitter specificity controls; the agent caught stale pass-1
metrics itself and re-gated on the committed pass-2 (the
FOLD-ON-NOTIFICATION lesson, applied prospectively by an agent).

WHAT WE SAW (T099): DOOR-STAYS-SHUT — the clean graft deletion
(wpe[183:189] := root) KILLED the graft (site onset 0.998 ->
0.599) and moved the door by +0.0000 (g-12 0.1021, dCE -0.0000;
zero/mean modes identical). THE GRAFT ROWS CARRY EXACTLY ZERO
OF THE DOOR'S CLOSURE. Head ablations push the door DOWN
(reader3-zero 0.0041; N2-zero 0.0554); joints == head singles
(rows contribute nothing once heads ablated — clean
sub-additivity). Controls: root ceiling behaves (the knife
works there: N2-zero 0.0378); jitter specificity confirms
no-restore where no graft formed. THE READING: the closure is
NOT active competitive inhibition by the wpe graft — the third
great asymmetry: doors open by training, are killed by surgery,
and cannot be REOPENED by surgery (T037's write-once core
extended to re-opening; T091 bounded on the removal side). THE
HONEST FORK-NARROWING: the conversion's delta is 66.5% MLPs/LNs
(e153) that this surgery cannot touch — STAYS-SHUT cannot
separate 'destructively rewritten' from 'inhibition carried in
the stream state'; the REWRITE-vs-DISUSE fork stays open for
e161's freeze-cell, and a gradient re-teach-the-restore cell
(discriminating MLP/LN carriage) is named. Single lineage/seed.

---

""" + o
assert o in n
n = n.replace(o, entry, 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

# ---------- THINKING ----------
t = open("THINKING.md", encoding="utf-8").read()
anchor = "## T098 — E164:"
t099 = """## T099 — E166: zero of the closure — the third great asymmetry, and the fork narrows honestly (2026-09-28 ~13:50Z)

The inverse event returned the cleanest zero the session has
produced: deleting the graft rows kills the graft and moves the
geometry door by +0.0000 — not small, ZERO. Whatever closes the
door, it is not the wpe graft, not actively, not even a little.
Combined with the corrected T097 (home graft + open door) and
the head ablations pushing the door only down: THE CLOSURE IS
SOMEWHERE ELSE — in the 66.5% MLP/LN delta the knife cannot
reach, or in disuse-decay of the ±12 pathway. The fork:
REWRITE (the conversion rewrote the readout's stream state) vs
DISUSE (the pathway decayed for lack of use) — e161's freeze
cell separates them (does the door close under PLAIN CORPUS,
no fact teaching at all?); a gradient re-teach-the-restore cell
would test MLP/LN carriage directly.

THE THIRD GREAT ASYMMETRY, stated: doors OPEN by training
(variance), are KILLED by surgery (N2), and CANNOT BE REOPENED
by surgery (graft removal = +0.0000; transplants nudge to 1/3
bar at best). T037's write-once core now covers all three
directions: no non-gradient write adds function, and no
gradient-built access can be surgically restored once lost.
FOR THE PAPER: claim 2 says "accompanies" (licensed); the
asymmetry triplet joins split custody and the asymmetry of
existence as the third exhibit of the unlearning section. FOR
W018: the BURY fate is not surgically reversible via the tomb's
rows — re-excavation, if possible at all, is a TRAINING
operation (variance at the buried site); the four fates remain
a phase diagram only if training can move between them.

""" + anchor
assert anchor in t
t = t.replace(t_anchor if False else anchor, t099, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- paper ----------
p = open("scratch/day6_paper_skeleton.md", encoding="utf-8").read()
o_p = "mechanism pending e165/e166/e161"
n_p = "mechanism: NOT the graft (e166 — deleting the graft rows moves the door by exactly zero; the closure lives elsewhere: stream state or disuse, e161 pending)"
assert o_p in p
p = p.replace(o_p, n_p, 1)
open("scratch/day6_paper_skeleton.md", "w", encoding="utf-8").write(p)

# ---------- queue + state ----------
q = open("QUEUE.md", encoding="utf-8").read()
m = re.search(r"^\| e166 \|[^\n]*\n", q, re.M)
assert m
q = q[:m.start()] + "| e166 | the inverse event | DONE 13:50Z (T099: DOOR-STAYS-SHUT — graft deletion kills the graft at +0.0000 door movement; the graft rows carry ZERO closure; third great asymmetry: open-by-training / killed-by-surgery / unreopenable-by-surgery; REWRITE-vs-DISUSE open for e161 + a gradient-restore cell) |\n" + q[m.end():]
open("QUEUE.md", "w", encoding="utf-8").write(q)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 2: e154 (two facts) + e152R (re-seeds). e166 DONE: DOOR-STAYS-SHUT — zero of the closure in the graft rows."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("e166 fold complete")
