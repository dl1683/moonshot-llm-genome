import json, datetime, re

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- NOTES ----------
n = open("NOTES.md", encoding="utf-8").read()
o = "## E162 — the READ-vs-MASS fork:"
entry = """## E125a — the inverted knife: NO-SITE-KNIFE — an asymmetry of EXISTENCE; the generalizing memory is the removable one (2026-09-28 ~12:30Z) — DONE

WHAT WE DID: arm_b's own census (bit-tight vs e133's unread
table: L1H2 0.215 mode-robust, L0H5, L1H4, L1H1, L0H1 — nearly
DISJOINT from the consolidated net's kill ladder) + the full
escalation (B-ladder, e160 sets re-pointed, L3H5-class, X-sets,
random scatter) with split-window selection; 92 cells, 6/6
gates; cross-check on e143_near (site at 5-13).

WHAT WE SAW (T096): NO-SITE-KNIFE fires — best flat-CE drop
25.7% < 30% bar; STRONGER THAN THE BAR: no cell reached 60% at
ANY CE (ceiling 32.8% @ +0.836; site onset never below ~0.63).
N2 spares the site fact (-0.8%) — e160's control replicated.
The B-ladder is SATURATING/SUB-ADDITIVE (S 21.5% -> B4 32.8%) —
redundant population coding, the OPPOSITE of e160's
superadditive complementary circuit. Cross-check: no kill at
the second site either (best flat 3.6%; N2 spares at 1.2%) —
the un-killability replicates. TWO MEMORY TYPES, TWO FACT-HEAD
POPULATIONS (disjoint except the weakest member L0H1). THE
ASYMMETRY OF EXISTENCE: the sink-coupled fact dies 79-95% at
CE +0.25; the site-stored fact has NO kill set at any price.
Honesty: split-window selection (disjoint prompts/fillers;
sel-vs-bar agree); families registered census-independent
(non-self-fulfilling null); both modes; single lineage (near =
site-geometry replication, not family); scope = head-coordinate
surgery only (33% of arm_b's load is MLP).

---

""" + o
assert o in n
n = n.replace(o, entry, 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

# ---------- THINKING ----------
t = open("THINKING.md", encoding="utf-8").read()
anchor = "## T095 — E162:"
t096 = """## T096 — E125a: the asymmetry of existence — consolidation buys generalization AND surgical removability; the locked-in memory can neither travel nor be excised (2026-09-28 ~12:30Z)

The inverted knife returned the strongest possible null: across
92 cells — two sites, both ablation modes, the site's OWN census
ladder, the e160 sets, address heads, and random controls — the
site-stored fact never dropped 60% at ANY CE. Not "harder to
remove": NO KILL SET EXISTS over the head-coordinate surface.
And the two facts' head populations are DISJOINT (L1H2-led vs
L0H3/L1H0-led, sharing only their weakest member), with the
site-fact's ladder SATURATING where the sink-coupled ladder was
SUPERADDITIVE — redundant population coding versus a
complementary killable circuit. T090's circuit-selectivity
inverts into an ASYMMETRY OF EXISTENCE.

THE POIGNANT INVERSION (the paper's strongest unlearning
sentence): the memory that GENERALIZES — geometry-free,
deletion-tolerant, the one jitter built — is the memory you can
surgically remove at CE +0.25. The memory that STAYS PUT —
context-bound, site-locked, the one locked replay built — is
the memory you cannot remove at any price. Consolidation trades
permanence-of-place for portability, and the price of
portability is vulnerability to the knife. For unlearning
practice the lesson inverts the usual fear: the DANGEROUS
memory (the one that generalizes everywhere) is the EASY one to
excise; the harmless-looking localized memory is the
incorrigible one.

FOR T092: layer 2 forks by phase — a killable complementary
circuit (sink-coupled) versus an unkillable redundant population
(site-stored); the four-layer model's readout layer was one
layer too flat. FOR e164 (post-kill census): now also asks
whether the site-fact's MLP third (33% of load) is the
incorrigible substrate.

""" + anchor
assert anchor in t
t = t.replace(anchor, t096, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- paper + report ----------
p = open("scratch/day6_paper_skeleton.md", encoding="utf-8").read()
o_p = "RESOLVED by e160: the surgical surface EXISTS"
n_p = "COMPLETED by e125a: an ASYMMETRY OF EXISTENCE — the sink-coupled\n    (generalizing) memory dies 79-95% at CE +0.25 via a superadditive\n    complementary circuit; the site-stored (locked-in) memory has NO kill\n    set at ANY CE (92 cells, two sites, both modes; disjoint fact-head\n    populations; saturating redundant ladder) — the memory that\n    generalizes is the memory you can remove. Preceded by e160:"
assert o_p in p
p = p.replace(o_p, n_p, 1)
open("scratch/day6_paper_skeleton.md", "w", encoding="utf-8").write(p)

r = open("DAY_SIX_REPORT.md", encoding="utf-8").read()
o_r = "- e160: the surgical surface EXISTS"
n_r = "- e125a (completing e160): the ASYMMETRY OF EXISTENCE — the\n  site-stored fact has NO kill set at any CE (92 cells, two\n  sites); the generalizing memory is the removable one.\n- e160: the surgical surface EXISTS"
assert o_r in r
r = r.replace(o_r, n_r, 1)
open("DAY_SIX_REPORT.md", "w", encoding="utf-8").write(r)

# ---------- QUEUE + STATE ----------
q = open("QUEUE.md", encoding="utf-8").read()
m = re.search(r"^\| e125a \|[^\n]*\n", q, re.M)
assert m
q = q[:m.start()] + "| e125a | the inverted knife | DONE 12:30Z (T096: NO-SITE-KNIFE — no 60% cell at ANY CE across 92 cells, two sites, both modes; disjoint fact-head populations; site ladder saturating (redundant) vs sink-coupled superadditive (complementary); the asymmetry of existence) |\n" + q[m.end():]
open("QUEUE.md", "w", encoding="utf-8").write(q)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 1 (CPU): e158 (2x2 — the last running cell). e125a DONE: NO-SITE-KNIFE — the asymmetry of existence."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("e125a fold complete")
