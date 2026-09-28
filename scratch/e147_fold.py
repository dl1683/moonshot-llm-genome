import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- NOTES.md ----------
n = open("NOTES.md", encoding="utf-8").read()
o = "## E150 — the flat-CE route test:"
entry = """## E147 — the width ladder: TEXTURE (a CLIFF, not a dose) — any variance switches the memory type; the address key dies at w=1, never gradually (2026-09-28 ~09:50Z) — DONE

WHAT WE DID: six width arms (w in {1,2,4,16,32,64}) from the
e048_repro root + free endpoints (w0=L, w8=R + a replicate, NEAR,
FAR, root); co-measured A(w) (row-129 address-key delta) and
NR(w) (d_r0 drop at g-12 + g+12); gates bit-exact; 1003s.

WHAT WE SAW (T087): INVARIANCE-CAUSAL did NOT fire as registered
(monotone clause: Spearman -0.381 vs bar <= -0.8; sensitivity
ladders all >= -0.38). DEAD-AGAIN no (A range 0.460).
SEED-COVERAGE no (NR(32)/NR(64) above onset bar — no collapse;
T079-pure fails mildly at 0.67x without W010's cliff). THE
TEXTURE IS THE FINDING: A CLIFF, NOT A DOSE — A(w): L +0.327,
w1 -0.029, then a mildly negative plateau (-0.03..-0.13, no
width trend); NR(w): L 0.071, w1 0.696, plateau 0.6-0.9. ANY
position variance (even +-1, name confined to rows 128-130)
kills the address key and births novel-geometry expression
within 300 steps — co-onset at the ladder's resolution limit,
STEP-FUNCTION form. SECONDARY: locked replay erodes NR BELOW
root (0.071 vs 0.166) while strengthening A (+0.327) — the roads
diverge in OPPOSITE directions from the first rung of variance.
FREE CELL FIRES: FAR-ROUTED-TAIL — e143_far's 0.205 g-12 tail
collapses under d_r0 (x0.069): FAR's hybrid tail was row-0-
coupled, not the band field (T084 item 3 resolved). Honesty:
single lineage; NR bounded by each arm's g-12 expression
(existence-confound; ratio forms co-reported, x0.004-0.023 for
all w>=1); NR now carries e150's POISONING semantics (sink-health
dependence, not information routing); L's 150-step budget
bracketed by sensitivity ladders; e143_jitter vs e119_r300 the
one clean replicate (A -0.1324/-0.1325).

---

""" + o
assert o in n
n = n.replace(o, entry, 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

# ---------- THINKING.md ----------
t = open("THINKING.md", encoding="utf-8").read()
t_anchor = "## T086 — E150:"
t087 = """## T087 — E147: the cliff — variance is a SWITCH, not a dial; the type decision is binary at zero-vs-any (2026-09-28 ~09:50Z)

The registered graded law did not fire; what landed is cleaner:
THE MEMORY TYPE IS DECIDED BY A CLIFF. A(w) at w=1 is already
negative (-0.029 vs L's +0.327) and never trends with width;
NR(w) onsets at w=1 (0.696 vs L's 0.071) and plateaus. ANY
position variance — the name moving across as few as three read
rows — is sufficient and (within 300 steps) saturated. T079's
invariance law returns in STEP-FUNCTION form: variance doesn't
GRADE the competition; it OPENS it. The biology echo sharpens:
systems consolidation in the literature is discussed as graded
transfer; here the DECISION to transfer is all-or-none at the
moment the positional key stops being perfectly predictive.

THE ROADS DIVERGE IN OPPOSITE DIRECTIONS FROM THE FIRST RUNG:
locked replay strengthens the trained-geometry key (+0.327)
while ERODING novel-geometry expression below the untrained root
(0.071 vs 0.166) — massed replay doesn't just fail to
consolidate; it actively CONTRACTS the memory's reach.
Variance-replay does the exact opposite on both dials. Two
opposite developmental trajectories from one binary switch.

E150'S CAVEAT APPLIED: NR is now read as sink-HEALTH dependence
(poisoning semantics), and NR is bounded by g-12 expression
existence — the cliff's NR-side conflates 'expresses at novel
geometry' with 'depends on sink health there.' The A-side
(address-key death at w=1) is clean of both caveats. The honest
composite: variance switches OFF the address key (clean) and
switches ON novel-geometry expression that is sink-coupled
(caveated). FAR-ROUTED-TAIL's fire (x0.069) says e143's FAR
boundary case resolves toward coupling.

FOR THE PAPER: claim 2's switch upgrades from 'error-position
variance' to 'a binary cliff at zero-vs-any variance' — simpler
to state, stronger to show (one figure, two rungs). The
mechanism hunt for WHAT variance does (why does one moved
window cancel the address key?) reopens at the head level
(e133's L0H3 class) with the P-b cell (one net, both types) as
the taxonomy's last structural confound — dispatched next.

""" + t_anchor
assert t_anchor in t
t = t.replace(t_anchor, t087, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- QUEUE ----------
q = open("QUEUE.md", encoding="utf-8").read()
import re
m = re.search(r"^\| e147 \|[^\n]*\n", q, re.M)
assert m
q = q[:m.start()] + "| e147 | width ladder | DONE 09:50Z (T087: TEXTURE = CLIFF — any variance (±1) kills address key (A: +0.327→-0.029) and births novel-geometry expression (NR: 0.071→0.696), no width trend; INVARIANCE-CAUSAL failed as graded form, returns as step-function; FAR-ROUTED-TAIL fires (×0.069); locked replay CONTRACTS reach below root) |\n" + q[m.end():]
open("QUEUE.md", "w", encoding="utf-8").write(q)

# ---------- STATE + report + paper ----------
s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 0 — dispatching e151 (P-b: one net, both types). e147 DONE: the cliff."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)

r = open("DAY_SIX_REPORT.md", encoding="utf-8").read()
o_r = """e143 then WON invariance's last causal stand (COMPASS-CAUSAL),
   and the width ladder (e147) is measuring the law's
   dose-response — whether address-key death and route birth
   CO-ONSET at a critical jitter width — as this report's final
   open cell, alongside e150's flat-CE verdict."""
n_r = """e143 then WON invariance's causal stand, and e147's ladder
   found the final form: a CLIFF, not a dose — ANY position
   variance (even ±1) kills the address key (+0.327 → −0.029)
   and births novel-geometry expression (0.071 → 0.696) within
   300 steps, with no width trend above w=1. The type decision
   is binary at zero-vs-any; locked replay actively CONTRACTS
   the memory's reach below the untrained root. e150's verdict
   closed the last open cell the same hour."""
assert o_r in r
r = r.replace(o_r, n_r, 1)
open("DAY_SIX_REPORT.md", "w", encoding="utf-8").write(r)

p = open("scratch/day6_paper_skeleton.md", encoding="utf-8").read()
o_p = "(2) Two memory types follow: SITE-STORED"
n_p = "(2) Two memory types follow, switched by a BINARY CLIFF at zero-vs-any error-position variance (e147: ±1 suffices — address key +0.327→−0.029, novel-geometry expression 0.071→0.696, no width trend): SITE-STORED"
assert o_p in p
p = p.replace(o_p, n_p, 1)
open("scratch/day6_paper_skeleton.md", "w", encoding="utf-8").write(p)
print("e147 fold complete")
