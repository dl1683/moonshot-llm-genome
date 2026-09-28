import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- NOTES.md ----------
n = open("NOTES.md", encoding="utf-8").read()
o = "## E133 — field anatomy census:"
entry = """## E141 — sink-key mechanism battery: ROLE-ROUTED (presence-only) — "re-keyed to row 0" formally dead; the noun is row-0 sink-routed (2026-09-28 ~08:05Z) — DONE

WHAT WE DID: five probes (install-restore t-surgery, presence-
vs-content scramble, rows-2-7 + norm-matched scaffold hardening,
d_r0 at novel geometry g-12 on both R nets, gate-vs-source
interpolation) on gated nets; 87s CPU; all artifact gates
bit-exact.

WHAT WE SAW (T081): ROLE-ROUTED, 3 corroborating votes to 0.
(1) INSTALL-RESTORE: removing the ENTIRE consolidation delta
from wpe[0] costs nothing (t=1 x0.999, CE +0.0002); direction-
scramble (norm-preserving permutation) RAISES fact expression to
0.817 while costing +0.70 CE; only removal-class kills (zero
0.053; mean-replace 0.001 — mean-row norm 0.066 = near-removal);
half-norm survives, double-norm slightly helps. (2) PRESENCE-KEY:
front-window scramble HELPS (0.858 vs mid-control 0.804, base
0.785) — scrambling the sink-region content improves the fact
read. (3) SINK-UNIQUENESS hardened: rows 2-7 + norm-matched
random all cheap (max CE +0.017); only row 0 wrecks (+1.40/+2.00).
(4) NOVEL-GEOMETRY COLLAPSE: d_r0 at g-12 x0.014 on R@150 AND
R@300 — W011's missing cell filled: the generalization itself
routes through row 0's presence (content-keyed alternative dies).
(5) NO-COLLAPSE curve: neither SOURCE-GRADED nor GATE-THRESHOLD
can fire (nothing is lost on the path; the flat-curve R^2 trap
documented before adjudication). MECHANISM: row 0 is the net's
attention-sink pivot (2nd-largest row norm); the consolidated
fact's READOUT WEIGHTS route through row 0 EXISTING — presence,
not content, not direction. SAVOR: the corpus-CE dissociation
(direction-scramble: fact intact, CE +0.70) — the fact's route
is more presence-robust than the net's general LM function.
Honesty: single-seed line for probes 1-3 (probe 4 adds two
independently-trained R nets, agreeing); t-curve probes one path
but perm + norm riders make the surviving manifold >=2-parameter
wide; scramble leakage bounded by matched control + the sign of
the effect.

---

""" + o
assert o in n
n = n.replace(o, entry, 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

# ---------- THINKING.md ----------
t = open("THINKING.md", encoding="utf-8").read()
t_anchor = "## T080 — E133:"
t081 = """## T081 — E141: presence, not content — the fact routes through row 0 EXISTING (2026-09-28 ~08:05Z)

The R44 critic's crack is confirmed and sharpened beyond it.
"Re-keyed to row 0" is formally dead: removing the entire
consolidation delta from wpe[0] costs NOTHING (x0.999, CE flat),
and even scrambling row 0's DIRECTION leaves the fact intact
(0.817, +6%) while degrading the corpus (+0.70 CE). What kills
is only REMOVAL (zero/mean/near-removal) — and rows 2-7 plus a
norm-matched random row are all cheap. The noun, corrected: the
consolidated fact is ROW-0 SINK-ROUTED — its readout weights
need the pivot to EXIST (norm >= ~0.38 suffices), not to say
anything. Three consequences:

(1) T077's second amendment CONFIRMED and SHARPENED: the
migration wrote nothing into the destination row; everything it
wrote lives in readout weights (e133: heads 84.5% of the
fact-specific residue). The "moved out" metaphor reduces to: the
read policy stopped keying on position 129 and started keying on
presence-at-the-pivot + content — W013's protagonist with its
mechanism completed.

(2) W011's omnipresence survives in sharpened form — OMNIPRESENCE
OF PRESENCE: the g-12 collapse (x0.014, both R nets) kills the
content-keyed alternative; geometry-independence is literally
routed through the one row that exists in every context, and
what it contributes is BEING THERE (its norm as the pivot), not
its content. The bio-echo sharpens absurdly and beautifully: the
schematic memory's "cortex" is the fact that position zero
exists.

(3) THE CE DISSOCIATION is the new savor: direction-scrambling
row 0 costs the corpus +0.70 nats but HELPS the fact (+6%) —
the fact's route is more presence-robust than the net's own
language function. A memory that survives what cripples the
net's general machinery: the route's independence from the
pivot's content is exactly what makes it geometry-general. Also
noted: scrambling the sink-region content IMPROVES the read —
the pivot's content is, if anything, competition for the route.

REMAINING OPEN: the route's anatomical finish (e133's L0H3 +
value-channel population) has no e141 cell confirming it
directly; the natural completion is W013's POLICY TRANSPLANT
with a presence-preserving twist — transplant the consolidated
net's readout deltas onto the install net and test whether
presence-keying alone converts address-reads into deletion-
tolerant reads. And e140's question survives REWORDED: does the
ROUTE's row-0 DEPENDENCE grow with jitter dose and stay flat
under locked/erase? (The e131 content test measures presence-
necessity — still the right dial, renamed.)

""" + t_anchor
assert t_anchor in t
t = t.replace(t_anchor, t081, 1)

# W011 sharpening amendment
o_w = """AMENDMENT (R44 critic): the CONTENT-KEYED alternative is alive —"""
n_w = """RESOLUTION (e141, ~08:05Z): the content-keyed alternative is
DEAD — d_r0 at g-12 collapses x0.014 on both R nets; and the hub
is not row 0's content either (direction-scramble spares the
fact): the hub is the pivot's PRESENCE. Omnipresence of
presence. Prior amendment retained below for the record.
AMENDMENT (R44 critic): the CONTENT-KEYED alternative is alive —"""
assert o_w in t
t = t.replace(o_w, n_w, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- QUEUE.md ----------
q = open("QUEUE.md", encoding="utf-8").read()
import re
m = re.search(r"^\| e141 \|[^\n]*\n", q, re.M)
assert m
q = q[:m.start()] + "| e141 | sink-key mechanism battery | DONE 08:05Z (T081: ROLE-ROUTED presence-only — install-restore x0.999 CE-flat; direction-scramble SPARES fact (+6%) while +0.70 CE; removal-class only kills; rows 2-7 cheap; g-12 collapse x0.014 both R nets kills content-keyed; presence >= ~0.38 norm suffices) |\n" + q[m.end():]
# reword e140's premise per T081
o_q = "| e140 | ROW-0 GROWTH TRACE + E-ROAD KEYING (T078 open edges ii+iii; eval-only on e119's saved phase nets) | READY (CPU; dispatch after e139 to avoid CPU pileup) | row-0 content test (e131 instrument) across: twin start / E@c1 / E@c2 / E@c3 / R@150 / R@300 / L@150."
n_q = "| e140 | ROUTE-DEPENDENCE TRACE (reworded per T081: the e131 test measures row-0 PRESENCE-necessity; eval-only on e119's saved phase nets) | READY (CPU) | row-0 presence-dependence (e131 instrument verbatim) across: twin start / E@c1 / E@c2 / E@c3 / R@150 / R@300 / L@150."
assert o_q in q
q = q.replace(o_q, n_q, 1)
open("QUEUE.md", "w", encoding="utf-8").write(q)

# ---------- STATE.json ----------
s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 2: e139 (CPU, finishing) + e140 (dispatching: route-dependence trace + L-CYCLED + T079 adjudication). e141 DONE: ROLE-ROUTED presence-only."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("e141 fold complete")
