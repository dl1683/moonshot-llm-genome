import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- NOTES.md ----------
n = open("NOTES.md", encoding="utf-8").read()
o = "## E131 — the re-keying census:"
entry = """## E119 — migration head-to-head: AMBIGUOUS (1 of 2), leaning DIFFERENT-STORES — jitter re-keys, erasure TIGHTENS the address (2026-09-28 ~07:20Z) — DONE

WHAT WE DID: twin installs, same fact, matched final expression
(R@150 0.5597 vs E@c1 0.5606, gap 0.0009; both dials the
pre-registered freedom); full comparative battery; locked-replay
free-rider arm; 7 phase nets saved runs/checkpoints/e119_*.pt.

WHAT WE SAW (T078): verdict AMBIGUOUS as registered — one clean
dissociation ((c) brake: R +0.210 [+0.153,+0.279] FEEDS vs E -0.267
[-0.297,-0.240] SUPPRESSES, both CI-separated; L brakes -0.509
like E), near-misses on the same side (D-all R 0.769 vs E 0.190 —
E sits 0.01 under the 0.20 bar; held-30-under-D-all 0.709 vs
0.071; novel geometry g-12 R 0.813 vs E 0.092; share E off-grid).
Census: R grows 10 decision-band rows, E grows 2 (+ generic
high-row drift 220-254 — overlapping e131's row-249 census find).
PRE-REGISTRATIONS ALL FIRED (registered ~06:48Z before the
battery): P1 R-beats-E on deletion survival (3/3 geometries);
P2 E-scatters/R-concentrates (2 vs 10 band rows); P3 store-thins-
while-expression-recovers (field-only residue 0.190->0.013->0.001
monotone across cycles while cycle-ENDs go 0.347->0.425) — MORE
ERASE CYCLES MADE THE FACT MORE ADDRESS-BOUND, NOT LESS, the
opposite of T073's migration reading. e083's protocol transferred
cleanly to the e048_repro line (no vacuous erases). Honesty: E@c1
= one 300-step relearn vs R@150 mixed-position steps (matched
expression, different training mass); L's brake shows the R/E
brake difference is confounded with position-diversity, not
erasure per se; E's dall cell is threshold-fragile (0.01 under
bar); census growth-rule fires on generic anchor drift (the
band-restricted view is informative); V-typing mild (~0.30-0.36),
report-only.

---

""" + o
assert o in n
n = n.replace(o, entry, 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

# ---------- THINKING.md ----------
t = open("THINKING.md", encoding="utf-8").read()
t_anchor = "## T077 — E131:"
t078 = """## T078 — E119: the two roads run in OPPOSITE directions — jitter migrates (re-keys to row 0), erasure digs in (tightens the address) (2026-09-28 ~07:20Z)

Read against T077's frame, e119's AMBIGUOUS becomes decisive in
interpretation: R (jitter) survives D-all at 0.769 — exactly what
a row-0-keyed fact does (D-all never touches row 0; e131 showed
the e113-line jitter fact is row-0-keyed); E (erasure) falls to
0.190 and its field-only residue THINS toward zero with more
cycles (0.190->0.013->0.001). The roads are not two routes to
one store: JITTER IS MIGRATION; ERASURE IS ANTI-MIGRATION — each
erase cycle strips field residue and re-tightens the address
binding. T073's reading of e083 ("completion migrates onto
position-keyed machinery" as consolidation-by-erasure) is
INVERTED by its own head-to-head: what migrated was nothing; what
happened was address-dependence deepening under stress. W003's
complementary-systems analogy narrows to its replay half only —
the biological echo of lesion-induced recovery does not hold here
at matched expression.

THE BRAKE IS A RE-KEYING SCAR, NOT A CONSOLIDATION UNIVERSAL:
deleting the original address FEEDS R (+0.210 — the moved-out
tenant's lease, T077) but SUPPRESSES E (-0.267) and L (-0.509).
The agent's confound note is exactly right: L (locked replay, no
erasure) brakes like E, so the sign tracks POSITION-DIVERSITY
(jitter), not erasure. e115's dimmer — measured on the jitter
line — generalizes to re-keyed facts only; a fact that never
moved keeps its address as a crutch, and deleting the crutch
collapses it. Brake sign is therefore a DIAGNOSTIC: + means
moved, - means still living there.

ALL THREE PRE-REGISTRATIONS FIRED as written (06:48Z, before the
battery): the discipline paid — P3's expression/store-depth
dissociation is now the sharpest single number-line in the arc
(recovering expression, vanishing store). TEXTURE ECHO: E's
high-row drift (220-254) overlaps e131's census row-249 — high-row
drift is a real shared texture of anchor relearn, not noise.
OPEN EDGES: (i) E@c1 is one relearn at matched expression — a
mass-matched E (300 mixed steps, no erase) would separate
erasure-per-se from relearn-texture (the L arm partially covers
this; L's D-all was not reported — cheap rider); (ii) does E ALSO
row-0-key at its address-tightened endpoint? (e140: row-0
content test on e119's saved E checkpoints — eval-only, nets on
disk); (iii) R@300's overshoot (0.776) vs R@150's match — does
row-0 key strength grow with jitter dose (same e140 rider on
R@150 vs R@300)? The wiring trace (e132) demotes to optional:
row-0 growth across checkpoints answers its kernel question more
directly and eval-only.

""" + t_anchor
assert t_anchor in t
t = t.replace(t_anchor, t078, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- QUEUE.md ----------
q = open("QUEUE.md", encoding="utf-8").read()
import re
m = re.search(r"^\| e119 \|[^\n]*\n", q, re.M)
assert m
q = q[:m.start()] + "| e119 | migration head-to-head | DONE 07:20Z (T078: AMBIGUOUS-lean-DIFFERENT-STORES; brake dissociation clean, R +0.210 feeds vs E -0.267 suppresses; jitter migrates/erasure tightens; all 3 pre-regs fired; T073 erasure-migration inverted; brake = re-keying scar) |\n" + q[m.end():]

o_q2 = "| e132 | the wiring trace"
e140 = """| e140 | ROW-0 GROWTH TRACE + E-ROAD KEYING (T078 open edges ii+iii; eval-only on e119's saved phase nets) | READY (CPU; dispatch after e139 to avoid CPU pileup) | row-0 content test (e131 instrument) across: twin start / E@c1 / E@c2 / E@c3 / R@150 / R@300 / L@150. Bars: R-ROW0-MONOTONE fires if row-0 strength rises with jitter dose and not with erase cycles; E-NEVER-KEYS fires if all E checkpoints row-0-null (address-dig-in confirmed at the key level); PARTIAL if E keys late |
""" + o_q2
assert o_q2 in q
q = q.replace(o_q2, e140, 1)
open("QUEUE.md", "w", encoding="utf-8").write(q)

# ---------- STATE.json ----------
s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 1: e139 (CPU, row-0 universality). e119 DONE: jitter migrates (row-0), erasure digs in — T073 inverted; e140 READY next."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("e119 fold complete")
