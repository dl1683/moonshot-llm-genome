import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

t = open("THINKING.md", encoding="utf-8").read()

# T093 amendment: the mass alternative
o1 = """THE NOUN, FINAL FORM (pending replication
and e158/e154): the consolidated memory type is READ-COUPLED to
the sink"""
n1 = """R47-CRITIC AMENDMENT (~11:55Z — the noun is PROVISIONAL; the
fork is live): the joint cell was numerically identical to
mask-alone and removed reads AND mass together — it cannot
distinguish 'dies of what it reads' from 'dies of what the
absorber steals' (MASS-coupled: the poisoned row starves
informative positions of attention). e162 dispatched with the
two ~90s cells (value-restore-under-poison; mass-inflate-on-
healthy). Until it lands, every READ-coupled statement carries
[this fork].
THE NOUN, PROVISIONAL FORM (pending e162, replication,
e158/e154): the consolidated memory type is READ-coupled to
the sink"""
assert o1 in t, "T093"
t = t.replace(o1, n1, 1)

# T090 amendment: circuit-selective + number fix + unread census
o2 = """(2) THE KNIFE KNOWS THE TYPE: the same N2 coordinates kill the
install-phase fact (67%) but SPARE the site-stored fact (<=
10.6%). Head surgery is TYPE-SELECTIVE — the two memory phases
(and the install state) share a readout circuit that the
site-stored memory does not use."""
n2 = """(2) THE KNIFE IS CIRCUIT-SELECTIVE, ONE BOUNDARY [R47-critic
correction: not 'type-selective' — install dies too (N2-zero
66.9% @ +0.299; the 67.3% @ +0.32 cell is E2-mean — number
attribution fixed): the knife separates {sink-coupled, install}
vs {site-stored}; and 'does not use' must read 'does not
DEPEND on' (redundant supply unfalsified)]. The site-stored
fact's own census SAT UNREAD in e133's metrics (L1H2 0.204,
L0H5, L0H1 — a different coordinate set; max single-head drop
20%): the surgical surface may be type-ASYMMETRIC — e125a
(running) decides."""
assert o2 in t, "T090"
t = t.replace(o2, n2, 1)

# T094 n=1 marker + the free mask-column re-read
o3 = """## T094 — E152: the dwell time — the phase transition passes through a MIXED state, and the brake overshoots before it releases (2026-09-28 ~11:25Z)"""
n3 = """## T094 — [R47: n=1 TRAJECTORY — the dwell and the brake
overshoot carry [n=1] until 3 seeds re-run (e152R queued); the
s16->s32 bounce is the sole substance of CLEAN's failure] E152:
the dwell time — the phase transition passes through a MIXED state, and the brake overshoots before it releases (2026-09-28 ~11:25Z)

FREE RE-READ (R47 critic, ~11:55Z — from e152's own metrics, no
compute): the per-checkpoint MASK column splits the conflated
g-12 dial — at s32, mask-retention 0.979 while ladder@0.07
kills to 0.19: the shelf's ~0.5 IS route survival, not sink
health. The dwell verdict SURVIVES the clean dial."""
assert o3 in t, "T094"
t = t.replace(o3, n3, 1)

# Intro provisional marker
p = open("scratch/day6_paper_skeleton.md", encoding="utf-8").read()
o_p = "The lab's organisms are born with ONE memory organ: the omnipresent\nrow (the attention-sink coordinate) carries every naturally-placed\nassociation from the first exposure (13/13 install checkpoints, five\nseeds of the fresh family at rel 1.000)."
n_p = "The lab's organisms are born with ONE memory organ: the omnipresent\nrow (the attention-sink coordinate) carries every naturally-placed\nassociation from the first exposure (13/13 install checkpoints, five\nseeds of the fresh family at rel 1.000) [PROVISIONAL per R47: sits on\nthe trained-geometry dial T083 declared saturating; the licensing cell\n(e163: the same dial on arm_b, 7% row-0-share — reads ~1.0 => collapse\nto truism; 0.7-0.8 => stands) is queued]."
assert o_p in p, "intro"
p = p.replace(o_p, n_p, 1)
o_p2 = "the\n    memory's dependence into READ-coupledness with the sink (e159's double\n    dissociation"
n_p2 = "the\n    memory's dependence into coupling with the sink (e159's double\n    dissociation; READ- vs MASS-coupled fork pending e162"
assert o_p2 in p, "claim4"
p = p.replace(o_p2, n_p2, 1)
open("scratch/day6_paper_skeleton.md", "w", encoding="utf-8").write(p)

# NOTES number-attribution fix
n = open("NOTES.md", encoding="utf-8").read()
o_n = "the same coordinates kill the INSTALL-PHASE fact too (67.3% @ +0.32"
n_n = "the same coordinates kill the INSTALL-PHASE fact too (N2-zero 66.9% @ +0.299; the 67.3% @ +0.32 cell is E2-mean — attribution fixed per R47"
assert o_n in n, "notes fix"
n = n.replace(o_n, n_n, 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

# QUEUE: e162 dispatched, e163/e164/e152R queued
q = open("QUEUE.md", encoding="utf-8").read()
o_q = "| e125a | THE INVERTED KNIFE"
rows = """| e162 | THE READ-vs-MASS FORK (R47 critic's final attack — the frame's newest load-bearing fork) | DISPATCHED ~11:50Z (CPU eval-only, ~90s cells) | (i) VALUE-RESTORE-UNDER-POISON (key poisoned to 0.07, healthy values transplanted — heals => READ-coupled literal; kills => MASS); (ii) MASS-INFLATE-ON-HEALTHY (logit bias on key 0 reproducing the 1.69x absorber — kills => mass alone suffices); (iii) starvation measurement (attention mass on fact positions under poison). Bars: READ-COUPLED-LITERAL = both heal/spare; MASS-COUPLED = either kills; MIXED |
| e163 | THE SATURATION CONTROL (licenses or collapses the intro's first sentence) | READY (CPU eval-only, minutes) | run the e131/e142 row-0 dial on arm_b (fact 7.06% row-0-share, survives row-0 removal at 0.725): reads ~1.0 => the dial cannot detect non-carriage, 13/13 collapses to T083's truism (intro's first sentence rewrites); reads 0.7-0.8 => the intro stands |
| e164 | THE POST-KILL CENSUS (de-circularizes T092's layering: is content still present after the N2 behavioral kill?) | READY (CPU eval-only) | census the N2-killed net (e160's artifacts): fact substance in site/MLP residue present-but-behaviorally-dead => layers 1/2 genuinely separable; content vanished with readout => the model collapses one layer |
| e152R | DWELL RE-SEEDS (3 seeds; ~30 min GPU or 1.75h CPU splittable) + a 10/12/14-step insert | QUEUED | puts n=3 on the cliff timing, the dwell, and the brake overshoot |
""" + o_q
assert o_q in q
q = q.replace(o_q, rows, 1)
open("QUEUE.md", "w", encoding="utf-8").write(q)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 3 (all CPU): e158 (2x2) + e125a (inverted knife) + e162 (READ-vs-MASS fork). R47 critic folded."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)
print("R47 critic repairs applied")
