import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- NOTES.md ----------
n = open("NOTES.md", encoding="utf-8").read()
o = "## E146 — the dissociation matrix:"
entry = """## E160 — the head-set escalation: FLAT-CE-FACT-KILL — the surgical surface EXISTS, and it is TYPE-SELECTIVE (2026-09-28 ~10:50Z) — DONE

WHAT WE DID: graded head-set escalation (singles through E4 +
no-L0H3 sets N2-N4, zero and mean-replace, random-4 scatter,
install + site-stored control columns); gates bit-exact incl.
L0H3-zero reproducing e150's anchor exactly.

WHAT WE SAW (T090): FLAT-CE-FACT-KILL fires. THE KILL IS N2
{L1H0, L0H0} — WITHOUT the 'fact-specific' L0H3 — in BOTH modes
at the lowest CE: 79.4%/70.7% g0 drop @ +0.245/+0.28 (NLL3
+0.26). L0H3 is not necessary (its singleton: 58.6% @ +0.21,
zero mode). Random-4 scatter: no kills at CE +0.07-0.13 — the
kills are COORDINATE-SPECIFIC. SUPERADDITIVE: E2-mean 81.3% vs
33.6% summed singles — complementary heads (one suppressor +
route suppliers). THE DISSOCIATION (the paper figure): the same
coordinates kill the INSTALL-PHASE fact too (67.3% @ +0.32 — a
SHARED readout circuit, not a consolidation scar) while the
SITE-STORED fact survives same-coordinate surgery (<= 10.6% @
+0.25; only N4 reaches 30.8% at wreck-adjacent +0.89). HEAD
SURGERY DISSOCIATES THE MEMORY TYPES. Honesty: strongly
superadditive (no kill hidden behind sub-additivity); mean-mode
flips L0H3 alone 12->59% — every claim names its mode, kills
corroborated across modes; single lineage (e157 owes the
replication); g-12 fragile under random sets too (the
pre-registered primary is g0 where the random null is clean);
CE and NLL3 agree in sign on every cell.

---

""" + o
assert o in n
n = n.replace(o, entry, 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

# ---------- THINKING.md ----------
t = open("THINKING.md", encoding="utf-8").read()
t_anchor = "## T089 — E146:"
t090 = """## T090 — E160: the surgical surface exists — row-surgery cannot, head-surgery can, and the knife knows the type (2026-09-28 ~10:50Z)

The R46 critic's coin landed on the paper's best figure, and
the result composes the day better than the reading map's best
branch:

(1) THE COMPOSED CLAIM: consolidation moves DEPENDENCE into
organism-critical coordinates (the sink, whose corruption is
fatal — e150) while the READOUT consolidates into a small
attackable head-set (e160: N2 kills at CE +0.25). Row surgery
cannot remove the memory without organism death; head surgery
can, at priced collateral. The removable-to-irremovable sentence
completes: the memory moved from row-removable tissue into
head-attackable circuitry behind an unremovable coordinate.

(2) THE KNIFE KNOWS THE TYPE: the same N2 coordinates kill the
install-phase fact (67%) but SPARE the site-stored fact (<=
10.6%). Head surgery is TYPE-SELECTIVE — the two memory phases
(and the install state) share a readout circuit that the
site-stored memory does not use. This is the cleanest
single-figure dissociation the taxonomy has: not deletion
hierarchies, not geometry cells — one knife, two outcomes.

(3) L0H3 DEMOTED, N2 PROMOTED: the 'fact-specific' head was the
symptom, not the circuit — {L1H0, L0H0} (neither fact-specific
by e133's table) carries the kill at lower cost. STRONG
SUPERADDITIVITY (81% vs 34% summed): the circuit is COMPLEMENTARY
— consistent with one suppressor + route suppliers — a
two-component motif worth naming in the paper.

(4) W014's ordering graduates: heads > route >> band is now a
DEMONSTRATED capability with collateral prices (NLL3 +0.26), and
e125's design collapses to pricing the N2-class surface on the
phase battery. The bias-check (R46's checklist): this fold
STRENGTHENS a claim — the flipped question is asked: the cell
that would flip it is the site-stored circuit's OWN kill set
(does a symmetric site-stored-killing head-set exist at flat
CE? — queued as e125's first arm), and the lineage replication
(e157).

""" + t_anchor
assert t_anchor in t
t = t.replace(t_anchor, t090, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- Report + paper ----------
r = open("DAY_SIX_REPORT.md", encoding="utf-8").read()
o_r = """and consolidation is the movement
from the removable into machinery whose CORRUPTION is
organism-fatal — whether its readout is surgically attackable
is the open frontier (the L0H3 head-ablation near-miss, 58.6%
at CE +0.21, is one cell from deciding it)."""
n_r = """and consolidation is the movement
from the removable into an architecture of split custody: the
DEPENDENCE moved into organism-critical coordinates (the sink,
whose corruption is fatal) while the READOUT consolidated into
a small head-set that surgery CAN remove (e160: {L1H0,L0H0}
kills the memory at CE +0.25 — and spares the site-stored fact
under the same knife: the surgery is type-selective)."""
assert o_r in r
r = r.replace(o_r, n_r, 1)
open("DAY_SIX_REPORT.md", "w", encoding="utf-8").write(r)

p = open("scratch/day6_paper_skeleton.md", encoding="utf-8").read()
o_p = "L0H3-zero (58.6% drop at CE +0.21)\n    is the leading fact-circuit candidate for the surgical-unlearning surface."
n_p = "RESOLVED by e160: the surgical surface EXISTS — {L1H0,L0H0}\n    (no 'fact-specific' head needed) kills the fact at CE +0.25 in both\n    ablation modes, superadditively, while SPARING the site-stored fact\n    under the same coordinates — type-selective head surgery. Fig 2's\n    killer point; the unlearning ordering (heads > route >> band) is\n    demonstrated."
assert o_p in p
p = p.replace(o_p, n_p, 1)
open("scratch/day6_paper_skeleton.md", "w", encoding="utf-8").write(p)

# ---------- QUEUE ----------
q = open("QUEUE.md", encoding="utf-8").read()
import re
m = re.search(r"^\| e160 \|[^\n]*\n", q, re.M)
assert m
q = q[:m.start()] + "| e160 | head-set escalation | DONE 10:50Z (T090: FLAT-CE-FACT-KILL — N2 {L1H0,L0H0} kills at CE +0.25 both modes, no L0H3 needed; superadditive (81 vs 34 summed); TYPE-SELECTIVE (install fact dies 67%, site-stored survives <=10.6%); random scatter clean) |\n" + q[m.end():]
m2 = re.search(r"^\| e125 \|[^\n]*\n", q, re.M)
if m2:
    q = q[:m2.start()] + "| e125 | attack the moved fact — RESHAPED post-e160: first arm = the site-stored circuit's OWN kill set (does a symmetric site-stored-killing head-set exist at flat CE? the flip-cell for T090); then collateral pricing on the N2-class surface across the phase battery (heads > route >> band DEMONSTRATED; price it) | READY | as reshaped |\n" + q[m2.end():]
open("QUEUE.md", "w", encoding="utf-8").write(q)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 2: e152 (GPU, conversion trace) + e153 (CPU, phase surgery). e160 DONE: FLAT-CE-FACT-KILL — type-selective head surgery."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("e160 fold complete")
