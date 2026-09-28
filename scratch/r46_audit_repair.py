import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- NOTES.md: E146 ----------
n = open("NOTES.md", encoding="utf-8").read()
o = "## E151 — the P-b cell:"
entry = """## E146 — the dissociation matrix: INSTRUMENT-INVALID — the self/other battery does not transfer to this line (2026-09-28 ~10:25Z) — DONE (null)

WHAT WE DID: the full intervention x function matrix (mask /
poison ladder / perm / head ablation x fact / self / CE) on the
B43-line consolidated net with the e111/e112 battery ported.

WHAT WE SAW (T089): the battery FAILS AT BASELINE — the foreign
donor does not collapse (gap 0.018 vs the >= 1.0 bar; every cell
reads ACCEPTS-FOREIGN; the printed ROUTED/TENANT clauses are
VOID). A weak occupancy separation survives (k*=1, E_sib 0.235 vs
E_for 0.077, clears nulls) but it is not the e111 holographic
k*=7 signature. Interpretation: INSTRUMENT TRANSFER FAILURE —
either this line lacks the binary self/other step or the donor/
null conventions mismatch; self-recognition, where the lab has
found it, is lineage-particular (consistent with family-typed
anchor physics, e108). W015's question (does the self survive
losing its pivot) is UNADJUDICABLE on this rig; e156 BLOCKED
until a lineage-native battery exists (options: run the matrix
on the e111 home lineage instead — its nets and battery exist).

---

""" + o
assert o in n
n = n.replace(o, entry, 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

# ---------- THINKING.md: T089 + retro-markers ----------
t = open("THINKING.md", encoding="utf-8").read()
t_anchor = "## T088 — E151:"
t089 = """## T089 — E146: the self-instrument does not travel — and that is data (2026-09-28 ~10:25Z)

The dissociation matrix could not run: the e111-lineage self/
other battery fails at baseline on the B43 line (foreign gap
0.018 vs 1.0). Read carefully, the null carries information:
the lab's self-recognition findings are LINEAGE-INDEXED — the
binary step, the k*=7 signature, the exclusion all live on the
nets they were measured on, and the instrument does not just
transfer. Two live readings: (a) the B43 line genuinely lacks
the self/other margin (families differ in whether they verify —
e108's binary two-cluster step was family-typed at 0.40-vs-0.14;
maybe B43 sits below it); (b) the battery's donor/null
conventions are calibration-sensitive (the 1.0 gap bar was tuned
on the home line). DISCRIMINATOR (cheap, queued as the e146
repair): run the SAME matrix on the e111 HOME lineage's nets
(they are on disk) — if the battery works there and fails on
B43 under matched conventions, reading (a) strengthens and
self-recognition joins the family-typed physics; if it fails
both, the battery's conventions need recalibration and W015
stays parked. W015 marked; e156 BLOCKED.

""" + t_anchor
assert t_anchor in t
t = t.replace(t_anchor, t089, 1)

markers = [
 ("## T080 — E133: content is everywhere, routes are the difference — the read-policy frame's first direct support",
  "## T080 — [RENAMED FRAME post-e150: 'routes' -> sink-health coupling; content claims stand] E133: content is everywhere, access differs — the read-policy frame's first direct support"),
 ("## T081 — E141: presence, not content — the fact routes through row 0 EXISTING",
  "## T081 — [HARD-BOUNDED by e150/T086: 'routes through' is sink-HEALTH dependence (poisoning), not information flow; presence claims rest on the perm/halfnorm/mean riders] E141: presence, not content — the fact needs row 0 EXISTING"),
 ("## T082 — E139: two memory types — ROUTED vs SITE-STORED — and the dreams that dream in coordinates",
  "## T082 — [TYPE RENAMED by e150/T086: ROUTED -> SINK-COUPLED; 'body-stored' -> content-in-heads/body; see T088: types are PHASES] E139: two memory types — sink-coupled vs site-stored — and the dreams savor (bounded)"),
 ("## W015 — WONDER: does the self survive losing its pivot? Connecting the two arcs",
  "## W015 — [UNADJUDICABLE on this rig: e146 instrument-invalid — the self-battery does not transfer to B43; home-lineage rerun queued] WONDER: does the self survive losing its pivot? Connecting the two arcs"),
 ("## W008 — WONDER: adapters into a position-invariant readout",
  "## W008 — [SUPERSEDED vocabulary: 'e113's end-stage BODY-STORED' below = sink-coupled content-in-heads; maturation already retracted R44] WONDER: adapters into a position-invariant readout"),
]
for old, new in markers:
    assert old in t, old[:40]
    t = t.replace(old, new, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- QUEUE ----------
q = open("QUEUE.md", encoding="utf-8").read()
import re
m = re.search(r"^\| e146 \|[^\n]*\n", q, re.M)
assert m
q = q[:m.start()] + "| e146 | dissociation matrix | DONE 10:25Z — INSTRUMENT-INVALID (T089: self-battery fails at baseline on B43 — foreign gap 0.018 vs 1.0; weak k*=1 occupancy survives; W015 unadjudicable; e156 BLOCKED; REPAIR queued: rerun matrix on the e111 home lineage) |\n| e146b | SELF-BATTERY HOME-LINEAGE RERUN (the T089 discriminator) | READY (CPU eval-only; e111 nets + battery on disk) | same matrix on the e111 home nets. Bars: BATTERY-HOME-VALID = foreign gap >= 1.0 at baseline there (self-recognition is lineage-indexed — B43 genuinely lacks the margin); BATTERY-MISCALIBRATED = fails on home too under matched conventions (recalibrate; W015 stays parked) |\n" + q[m.end():]
q = q.replace("DISPATCHED 10:25Z (CPU eval-only, low threads)", "DISPATCHED ~10:26Z (CPU eval-only, low threads; stamp led dispatch by ~10 min — noted per R46 audit)", 1)
m2 = re.search(r"^\| e152 \|[^\n]*\n", q, re.M)
if m2 and "DONE" not in q[m2.start():m2.end()]:
    pass  # still running, leave
open("QUEUE.md", "w", encoding="utf-8").write(q)

# ---------- STATE ----------
s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 3: e152 (GPU, conversion trace) + e153 (CPU, phase surgery) + R46 critic pending. e146 DONE: INSTRUMENT-INVALID (self-battery does not transfer; e146b repair queued)."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("e146 fold + audit repairs applied")
