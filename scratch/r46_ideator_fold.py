import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

q = open("QUEUE.md", encoding="utf-8").read()

# New rows after e152's row
o_q = "| e148 | DREAM-TOPOLOGY CENSUS"
rows = """| e153 | PHASE-SWITCH SURGERY (R46 ideator top pick — the order parameter's identity; T088's 'why global') | DISPATCHED 10:25Z (CPU eval-only, low threads) | e151_twodoor vs e131_consolidated differ by 300 locked steps — diff heads both ways (param-delta rank + re-run e133 census on the e151 net), transplant top-K head-sets (K=1,3,6; L0H3-class + twin-L3H4-class named arms) consolidated->e151 and e151->consolidated, eval-only. Readout: phase dial vector (g-12/g+12, D-all, D-183, A(129)). Bars: PHASE-IN-HEADS = some K<=6 transplant reopens the geometry door (g-12 >= 0.45) and/or reverse closes it (>=70% drop), CE priced per cell; PHASE-DISTRIBUTED = no head-set moves g-12 > 20% (order parameter in LN/MLP-stream state -> aims e135). Rider: e152 dwell checkpoint if TRANSIENT fires |
| e154 | TWO FACTS, ONE DOOR (rewrites e134; decides the day's headline — was e151's closure GLOBAL or self-conversion?) | QUEUED — first GPU slot after e152 | root e131_consolidated (F1 sink-coupled); install NEW fact F2 locked at a fresh site (~rows 60-70), 300 steps; measure F1's phase dials + trained-geometry expression. Bars: GLOBAL-PHASE = F1 g-12 drops >=70% while F1's own expression >=0.5 (door closed on an untouched fact); PER-MEMORY = F1 g-12 within 15% (e151 was self-conversion — the biggest correction since E120; abstract rewrites); CROSSOVER = 30-70% (shared substrate with capacity -> W012 contended-bandwidth). Rider: natural-install F2 arm (W016b) |
| e155 | THE HYSTERESIS LOOP (branch 2 of the phase diagram) | QUEUED — GPU after e154 | on e151_twodoor: jitter F1 (e113 ±8) with the same six budgets as e152; overlay both traces. Bars: SYMMETRIC = crossings in the same step-bin; HYSTERESIS-HARD = opening lags >=2 bins (graft pins the policy — W006 echo); PRIMED = opening leads >=1 bin (W010's ghost) |
| e156 | THE SELF ACROSS THE PHASE FLIP (e146 follow-up fork, pre-registered) | QUEUED — rides e146's rig after its gates pass | e146's self-battery on e151_twodoor vs consolidated (nets differ only by the phase conversion). Fork: if SELF-CONSTITUTIONAL -> predict SELF-INVARIANT (identity upstream of the policy); if SELF-ROUTED -> SELF-MOVES-WITH-PHASE; if SELF-TENANT -> e153's transplant arms localize the self; if SELF-INDEPENDENT -> depth-resolved V-source census |
| e157 | CONVERSION REPLICATION, second lineage (paper debt #1; absorbs e145-narrowed) | QUEUED — GPU | e098 s4305 -> jitter (e113) -> locked re-teach (e151 protocol); same dials. CONVERSION-REPLICATES = g-12 >=70% drop + site growth + CE non-worse on family 2; NO-CONVERSION = family boundary |
""" + o_q
assert o_q in q
q = q.replace(o_q, rows, 1)

# Staleness list
o_e134 = "| e134 | two facts, one sink (first multi-fact ecology; W012) | QUEUED |"
n_e134 = "| e134 | [STALE per R46 — rewritten as e154: the share-grid is a graft instrument (W012-2nd), phase dials primary] two facts | SUPERSEDED by e154 |"
assert o_e134 in q
q = q.replace(o_e134, n_e134, 1)

o_e144 = "| e144 | FROZEN-SINK INSTALL"
n_e144 = "| e144 | [PARKED per R46: bars written under the dead re-keyed noun; e141's no-op-by-norm predicts null — re-aim only at the surviving COMPENSATION bar if ever run] FROZEN-SINK INSTALL"
assert o_e144 in q
q = q.replace(o_e144, n_e144, 1)

o_e149 = "Bars: SCAR-IN-CONTENT = restore removes >=50% of brake AND anti-alignment cos<=-0.3 in variance-trained nets only"
n_e149 = "Add e151_twodoor as the PHASE CONTROL (T088(4): anti-alignment predicted only in variance-phase nets — the site-phase net should show NONE). Bars: SCAR-IN-CONTENT = restore removes >=50% of brake AND anti-alignment cos<=-0.3 in variance-trained nets only"
assert o_e149 in q
q = q.replace(o_e149, n_e149, 1)

open("QUEUE.md", "w", encoding="utf-8").write(q)

# Paper skeleton: R2 discharged, replaced by e154 fork
p = open("scratch/day6_paper_skeleton.md", encoding="utf-8").read()
o_p = "R2 \"Confounded taxonomy\" (R45 critic attack 1) — cite the bounds honestly;\n  the single-net-both-types cell (P-b) is queued; run it before submission."
n_p = "R2 DISCHARGED by e151 (one lineage, both phases, bidirectional). NEW R2:\n  \"was the closure global or self-conversion?\" — e154's two-facts cell decides;\n  run before the abstract's \"globally\" survives."
assert o_p in p
p = p.replace(o_p, n_p, 1)
open("scratch/day6_paper_skeleton.md", "w", encoding="utf-8").write(p)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 3: e146 (CPU) + e152 (GPU) + e153 (CPU, phase-switch surgery). R46: ideator folded (e154 decides global-vs-self), auditor+critic pending."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("ideator harvest folded")
