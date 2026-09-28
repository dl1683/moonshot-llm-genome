import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- NOTES.md ----------
n = open("NOTES.md", encoding="utf-8").read()
o = "## E139 — row-0 universality:"
entry = """## E140 — route-dependence trace: GRADIENT-VOLUME fires — T079 dies on its dial; T078's 'erasure digs in' RETIRED (cycle damage); the trained-geometry dial SATURATES (2026-09-28 ~08:40Z) — DONE

WHAT WE DID: row-0 presence-strength S (e131 instrument verbatim,
install-60 g0) across seven checkpoints + the L-CYCLED arm
(3x300 locked, no reset); all gates bit-exact (twin reproduces
e116's stored seed-42 values; R@300 vs e109 GPU ref 7.1e-4).

WHAT WE SAW (T083): FIRED GRADIENT-VOLUME ONLY — ratio
R@150/L@150 = 0.745 < 1.3 (locked replay MORE row-0-dependent
than jitter at matched steps); R-ROUTE-MONOTONE failed (R@150
0.4687 DIPS below twin 0.5455 before R@300 0.7226); E-NEVER-
ROUTES failed absolutely (E@c2 off 0.228) but passed relatively
(all E rel within 0.10 of twin's 0.981). THE CEILING: twin
starts at rel 0.98 — presence-dependence at the trained geometry
measures SINK-LOAD, which every readout has; it does not
isolate routing. L-CYCLED: D-all residue thins 0.1307 ->
0.0045 -> 0.0003 (fold 0.0022) vs E's 0.1902 -> 0.0132 -> 0.0010
(fold 0.0053) — monotone, same-or-faster, NO erasure; grown rows
explode (209) like E's ~190. ANTI-MIGRATION HAS NO
ERASURE-SPECIFIC EVIDENCE LEFT — 'erasure digs in' retired.
TEXTURE GOLD (row-129 address-key strength): L@150 0.327 >
E@c3 0.280 > twin 0.241, but R@150/R@300 NEGATIVE (-0.21/-0.13)
— deleting the address FEEDS only jittered facts (the brake as
census texture): locked and erased roads STRENGTHEN the address
key; only the position-varied road negates it. Fixed D-all:
twin 0.193, E@c2/c3 0.061, R@300 0.903. Honesty: single lineage
(all arms from one twin install); the e131 dial conflates
route-keying with sink-load (e141's lesson) — the ceiling effect
is WHY gradient-volume fired while no arm separates; the
route-isolating dial is presence-dependence at NOVEL geometry
(e141's g-12 design), which this run did not measure.

---

""" + o
assert o in n
n = n.replace(o, entry, 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

# ---------- THINKING.md ----------
t = open("THINKING.md", encoding="utf-8").read()
t_anchor = "## T082 — E139:"
t083 = """## T083 — E140: the law that died on the wrong dial — T079 killed as registered, T078 retired, and the instrument lesson that saves the taxonomy (2026-09-28 ~08:40Z)

Three adjudications, all honest:

**(1) T079 (credit-assignment) is DEAD ON ITS REGISTERED DIAL.**
GRADIENT-VOLUME fired: R@150/L@150 presence ratio 0.745 — locked
replay is MORE row-0-dependent than jitter. No bar shopping: the
law's registered prediction failed. But the INSTRUMENT LESSON is
load-bearing: the twin starts at rel 0.98 — presence-dependence
at the TRAINED geometry measures sink-load (which every readout
has), not routing. The route-isolating dial is presence-
dependence at NOVEL geometry (e141's g-12 collapse, which this
run did not measure). The law died on a dial that saturates for
everyone. ITS UNREGISTERED SIGNATURE SURVIVES IN THE SAME RUN:
row-129 address-key strength — L 0.327 > E@c3 0.280 > twin
0.241, but R NEGATIVE (-0.21/-0.13). Only the position-varied
road NEGATES the address key; locked and erased both STRENGTHEN
it. That is exactly what an invariance competition would
produce — but it was not the registered bar, so it is TEXTURE,
and reviving T079 requires a NEW pre-registered test (address-key
strength vs jitter-width ladder, novel-geometry presence
co-measured), not a reinterpretation.

**(2) T078's 'erasure digs in' is RETIRED.** L-cycled (no
erasure, just three locked cycles) thins 0.131 -> 0.0045 ->
0.0003, same-or-faster fold than E. Thinning is CYCLE DAMAGE.
The two-roads story in its final form: at matched expression,
position-varied replay produces ROUTED, deletion-tolerant,
geometry-general memories; locked and erased roads both produce
site-stored ones that degrade under cycling. There is no
erasure-specific phenomenon beyond the damage it shares with
any repeated fixed-position intervention.

**(3) THE T082 DERIVATION ADDENDUM'S SYLLOGISM FAILED — recorded
as predicted-then-falsified.** Registered at ~08:28Z (before
reading e140): the taxonomy predicts E-flat + L-flat + R-rising
on the row-0 dial. The data: everything flat-high (rel
0.84-0.98), R non-monotone. The taxonomy itself survives — its
discriminating evidence was never this dial; it is e139's
183-geometry cells (routed 0.84-row-0-dependent vs splice 0.25)
and e141's novel-geometry collapse. But the failure teaches the
taxonomy's boundary: ROUTING is invisible at the trained
geometry, where the sink carries everything; it shows only where
the memory must travel. What you measure WHERE matters more than
what you measure.

STANDING: e143 (in flight) now carries the invariance question's
last causal stand — NEAR vs FAR decides whether position-variance
is necessary for routing by INTERVENTION rather than census. If
NEAR routes, proximity wins and the invariance story ends; if
NEAR stays site-stored, invariance survives its observational
death.

""" + t_anchor
assert t_anchor in t
t = t.replace(t_anchor, t083, 1)

# T078 header marker: digs-in retired
o_h = "## T078 — E119: the two roads run in OPPOSITE directions — jitter migrates (re-keys to row 0), erasure TIGHTENS the address"
n_h = "## T078 — [DIGS-IN CLAUSE RETIRED by e140/T083: cycle damage, not erasure — L-cycled thins identically] E119: the two roads run in OPPOSITE directions — jitter migrates (routes via row 0), locked/erase stay site-stored"
assert o_h in t
t = t.replace(o_h, n_h, 1)

# T079 header marker: killed on dial
o_h2 = "## T079 — the credit-assignment law:"
n_h2 = "## T079 — [KILLED ON ITS REGISTERED DIAL by e140/T083 (gradient-volume, ratio 0.745); unregistered row-129 signature survives as texture — revival requires new pre-registration] the credit-assignment law:"
assert o_h2 in t
t = t.replace(o_h2, n_h2, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- QUEUE.md ----------
q = open("QUEUE.md", encoding="utf-8").read()
import re
m = re.search(r"^\| e140 \|[^\n]*\n", q, re.M)
assert m
q = q[:m.start()] + "| e140 | route-dependence trace | DONE 08:40Z (T083: GRADIENT-VOLUME — T079 dead on its dial, ratio 0.745; trained-geometry presence dial SATURATES (twin rel 0.98 — measures sink-load not routing); L-CYCLED retires T078's digs-in (thinning = cycle damage); row-129 texture: L/E strengthen address key, R NEGATES it (brake as census) — T079's unregistered signature) |\n" + q[m.end():]
open("QUEUE.md", "w", encoding="utf-8").write(q)

# ---------- STATE + day-six report cell ----------
s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 1: e143 (GPU, error-steering — invariance's last causal stand). e140 DONE: T079 dead on dial, T078 digs-in retired."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)

r = open("DAY_SIX_REPORT.md", encoding="utf-8").read()
o_r = """credit-assignment (T079: keys strengthen by invariance across
   error windows) its key-selection rule — [e140 ADJUDICATION
   PENDING: R-vs-L presence ratio, credit-assignment vs
   gradient-volume; L-CYCLED decides whether "erasure digs in"
   had any erasure-specific evidence]."""
n_r = """its key-selection rule was T079 (keys strengthen by invariance
   across error windows) — which e140 KILLED on its registered
   dial (gradient-volume: everything is row-0-dependent at the
   trained geometry; the dial saturates) while its unregistered
   signature survived in the same run's texture (only the
   position-varied road NEGATES the address key: R -0.21 vs
   L +0.33); and L-CYCLED retired "erasure digs in" outright
   (locked cycles thin identically — cycle damage, not erasure).
   e143 carries invariance's last causal stand."""
assert o_r in r
r = r.replace(o_r, n_r, 1)
open("DAY_SIX_REPORT.md", "w", encoding="utf-8").write(r)
print("e140 fold complete (incl. day-six cell)")
