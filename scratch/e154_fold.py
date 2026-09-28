import json, datetime, re

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- NOTES ----------
n = open("NOTES.md", encoding="utf-8").read()
o = "## E166 — the inverse event:"
entry = """## E154 — two facts, one door: TEXTURE — OVERWRITE, NOT SHARE; F1 annihilated (not door-closed) under a protocol whose anchors contradict it (2026-09-28 ~14:00Z) — DONE

WHAT WE DID: 300-step locked install of a NONCE fact (MIRABEL,
zero corpus occurrences — the novelty confound avoided) at rows
63-69 into the consolidated net; full F1/F2 dials; N2 + W017
riders; 9 gates PASS (N2 bit-exact vs e160).

WHAT WE SAW (T100): the registered bars cannot fire — F1 was
not untouched-with-door-closed, it was ANNIHILATED: g-12 0.916
-> 0.0002, g0 0.785 -> 0.0017, held30 -> 0.0001, row-0 strength
0.732 -> 0.001, A(129) -> +0.001, D-all -> 0.0003. F2's graft
FORMED (site onset 0.124, site_pos TRUE; F2 expresses 0.993 at
site) and CE_R IMPROVED (1.664 -> 1.649) — the damage is
F1-specific. OVERWRITE, NOT SHARE: no second-fact capacity at
this budget. THE CRITICAL CONFOUND (the agent's catch): the
e151-protocol anchor bank uses INCUMBENT-CONTINUATION host
windows — for a second-fact install these actively CONTRADICT
F1's home expression (8 paired anti-F1 anchors x 300 steps):
F1's demolition may be substantially ANCHOR-DRIVEN
unlearning-by-contradiction, not graft-driven; this run cannot
separate them. RIDERS (both clean): N2 kills NEITHER on the
two-fact net (F2 -3.1% — the knife's circuit-selectivity
REPLICATES on a fresh site-stored fact; F1's killable readout
no longer exists to kill); W017's prediction CONFIRMS on a
fresh fact — F2's census is DIFFUSE (top-1 0.139, 28/36 heads,
entropy 2.958 — the locked-trained redundant signature; F1's
sink-coupled was 0.352/16). Honesty: single seed/lineage;
MIRABEL nonce-clean; site at rows 63-69 carries only 64 tokens
pre-context (NEAR-mirror confound, priced: F2's deletion
footprint on F1-battery negligible — F1 was already gone).

---

""" + o
assert o in n
n = n.replace(o, entry, 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

# ---------- THINKING ----------
t = open("THINKING.md", encoding="utf-8").read()
anchor = "## T099 — E166:"
t100 = """## T100 — E154: overwrite, not share — and the anchors may have done it (2026-09-28 ~14:00Z)

The two-facts cell returned the strongest possible outcome with
the most important confound: F1 annihilated everywhere (not
merely door-closed — the registered GLOBAL-PHASE clause could
not fire because its own premise, an untouched F1, was
destroyed), F2's graft formed, CE improved, and the damage is
F1-SPECIFIC. Three readings:

(1) NO SECOND-FACT CAPACITY AT THIS BUDGET: the shared
substrate was overwritten. If confirmed by the confound-free
rerun, the "one door" question dissolves into "one FACT at a
time" — the consolidated net cannot hold a second locked-installed
fact without demolishing the first. W012's bandwidth reading
gets its answer the hard way: the bandwidth is ~one fact wide,
and installing into it costs the tenant everything.

(2) THE ANCHOR-CONTRADICTION CONFOUND (the agent's catch, now
the load-bearing caveat): the protocol's incumbent-continuation
anchors are an ANTI-F1 signal — 8 paired contradiction anchors
per batch, 300 steps. F1's demolition may be textbook
unlearning-by-contradiction through the anchor channel, with
the graft an innocent bystander. THE DISCHARGE CELL (e170):
the identical F2 install with NEUTRAL anchors (plain corpus,
no incumbent-continuation windows) — if F1 survives, the
demolition was the anchors (and the two-facts question REOPENS
with per-fact doors); if F1 still dies, the overwrite is real
and capacity is one fact.

(3) THE RIDERS ARE CLEAN GOLD: the knife's circuit-selectivity
replicates on a FRESH site-stored fact (F2 -3.1% under N2) and
W017's coding prediction confirms out-of-sample (F2 diffuse:
top-1 0.139, 28/36 heads — the locked-trained signature on a
second fact, third data point on the concentration law).

FOR THE PAPER: claim 2's "globally" must await e170 (the
demolition's channel is unsettled); the riders strengthen
claims 3-4 as-is.

""" + anchor
assert anchor in t
t = t.replace(anchor, t100, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- queue + paper ----------
q = open("QUEUE.md", encoding="utf-8").read()
m = re.search(r"^\| e154 \|[^\n]*\n", q, re.M)
assert m
q = q[:m.start()] + "| e154 | two facts, one door | DONE 14:00Z (T100: TEXTURE — OVERWRITE not share: F1 annihilated everywhere (0.785->0.002) while F2's graft formed and CE improved; CONFOUND: incumbent-continuation anchors = anti-F1 signal (unlearning-by-contradiction suspected); riders: N2 spares F2 (type-selectivity replicates) + F2 diffuse (W017 confirms out-of-sample)) |\n" + q[m.end():]
o_q = "| e167 | PER-LAYER-MATCHED MASS"
e170 = """| e170 | THE ANCHOR-NEUTRAL SECOND INSTALL (e154's confound discharge — was F1's demolition the graft or the anchors?) | QUEUED (CPU one training) | identical F2 (MIRABEL) locked install at rows 63-69 but with NEUTRAL anchors (plain corpus, no incumbent-continuation windows); measure F1's full dial set. Bars: ANCHORS-DID-IT = F1 survives (g0 >= 0.5, g-12 >= 0.5 — the two-facts question REOPENS with per-fact doors; unlearning-by-contradiction through the anchor channel is the demolition mechanism); OVERWRITE-REAL = F1 still dies (capacity is one fact at this budget; W012's bandwidth answered the hard way) |
""" + o_q
assert o_q in q
q = q.replace(o_q, e170, 1)
open("QUEUE.md", "w", encoding="utf-8").write(q)

p = open("scratch/day6_paper_skeleton.md", encoding="utf-8").read()
o_p = "e154 pending:\nglobal-vs-per-fact]"
n_p = "e154 landed TEXTURE with a confound: F1 ANNIHILATED under a protocol whose anchors contradict it — \"globally\" awaits e170 (anchor-neutral rerun); the riders (N2 spares F2; F2 diffuse) strengthen claims 3-4 now]"
assert o_p in p
p = p.replace(o_p, n_p, 1)
open("scratch/day6_paper_skeleton.md", "w", encoding="utf-8").write(p)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 2: e152R (GPU re-seeds) + e161 (freeze-cell). e154 DONE: OVERWRITE-not-share with the anchor-contradiction confound; e170 queued."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("e154 fold complete")
