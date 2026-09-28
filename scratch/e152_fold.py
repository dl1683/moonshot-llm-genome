import json, datetime

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- NOTES.md ----------
n = open("NOTES.md", encoding="utf-8").read()
o = "## E159 — coupled-or-organism:"
entry = """## E152 — the conversion time-trace: TRANSIENT-TWO-DOOR — the cliff has a DWELL TIME; e151's P-b was EARLY, not wrong (2026-09-28 ~11:25Z) — DONE

WHAT WE DID: six-checkpoint sequential trace (8/16/32/64/128/300)
of e151's locked re-teach; one trajectory, seed-fixed; root gates
bit-exact; ran CPU (park-to-CPU: user game held GPU at 86-87C —
thermal rule honored; CPU/CUDA float-path equivalence shown).

WHAT WE SAW (T094): TRANSIENT-TWO-DOOR fires — the cliff runs
between 8 and 16 steps (g-12: 0.990 -> 0.447), then DWELLS on a
~0.5 shelf through step 64 (checkpoints {8, 32, 64} hold BOTH
doors: site clears e151's content bar — genuine, row-183-local,
67x control, site read p_Z 0.962 — AND g-12 >= 0.5) before the
final descent (300: 0.139). CLEAN failed (non-monotone bounce
s16->s32; Spearman -0.60); DELAYED failed. TEXTURES: the A(129)
brake OVERSHOOTS mid-conversion (-0.132 -> -0.466 @s128) before
dissolving (the negative posterior intensifies as the graft
grows, then releases); CE_R's early wobble (1.705 @s8) recovers
to 1.643 — conversion, not damage; the 300-step endpoint
reproduces e151 behaviorally (max diff 0.029; two instrument
cells exceed strict tol, reported verbatim). Honesty: ONE
trajectory (the s16->s32 bounce is this path's, not a law);
single seed/lineage — the dwell time is a point estimate; the
~0.5 shelf partially conflates route survival with sink health
(mask/ladder columns in metrics price it).

---

""" + o
assert o in n
n = n.replace(o, entry, 1)
open("NOTES.md", "w", encoding="utf-8").write(n)

# ---------- THINKING.md ----------
t = open("THINKING.md", encoding="utf-8").read()
t_anchor = "## T093 — E159:"
t094 = """## T094 — E152: the dwell time — the phase transition passes through a MIXED state, and the brake overshoots before it releases (2026-09-28 ~11:25Z)

TRANSIENT-TWO-DOOR resolves T088's fork in the gentlest
possible way: the cliff is real AND the two-door state exists —
for a window. Between steps ~16 and ~64 the net holds BOTH a
genuine site-store (67x control, functionally readable) AND
>=50% geometry retention; then the zero-variance training
consolidates the graft and the shared state abandons the
geometry door. THE MIXED STATE IS A DWELL, not an artifact of
measurement timing: it survives across three checkpoints and a
bounce. FOR THE PAPER: claim 2 gains its most physical sentence
— the phase transition is not a hop but a passage through a
mixed state with a dwell time (~50 steps at this budget);
systems-consolidation language ("gradual transfer") and phase
language ("sudden switch") were both half right: the SWITCH is
sudden (8-16 steps), the SEPARATION is slow (the shelf), and
the memory spends the shelf holding both natures.

THE BRAKE OVERSHOOTS: A(129) intensifies (-0.132 -> -0.466)
mid-conversion before dissolving at the end. Under the
negative-posterior reading (T087), this is the address key
fighting hardest exactly while its replacement is being built —
suppression peaks at maximum competition, then the whole
opposition dissolves when the new phase settles. A tiny,
poignant mechanism: the old address does not fade; it resists,
then is released.

STANDING: the dwell is one trajectory, one seed (point
estimate); e158 (variance@183) still decides whether the
conversion itself keys on variance; e154 decides per-fact vs
global. The GPU is user-occupied — dispatch with park-to-CPU.

""" + t_anchor
assert t_anchor in t
t = t.replace(t_anchor, t094, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- QUEUE + STATE ----------
q = open("QUEUE.md", encoding="utf-8").read()
import re
m = re.search(r"^\| e152 \|[^\n]*\n", q, re.M)
assert m
q = q[:m.start()] + "| e152 | conversion time-trace | DONE 11:25Z (T094: TRANSIENT-TWO-DOOR — cliff at 8-16 steps, dwell on a ~0.5 shelf through s64 with BOTH doors genuine, then descent; brake OVERSHOOTS mid-conversion then dissolves; conversion not damage; one trajectory/seed) |\n" + q[m.end():]
open("QUEUE.md", "w", encoding="utf-8").write(q)

s = json.load(open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = now
s["current_experiment"] = "Fleet 1: e158 (jitter@183 — THE 2x2 cell; park-to-CPU authorized, GPU user-occupied at 86-87C). e152 DONE: TRANSIENT-TWO-DOOR."
json.dump(s, open("STATE.json", "w", encoding="utf-8"), indent=2)

# ---------- paper ----------
p = open("scratch/day6_paper_skeleton.md", encoding="utf-8").read()
o_p = "the types are PHASES of one substrate, converted bidirectionally by\ntraining [e158 pending: variance-vs-placement; e154 pending:\nglobal-vs-per-fact]."
n_p = "the types are PHASES of one substrate, converted bidirectionally by\ntraining — and the conversion PASSES THROUGH A MIXED STATE: the cliff\nfires in 8-16 steps, then a ~50-step dwell holds BOTH natures (site-store\ngenuine at 67x control AND >=50% geometry retention) before separation\ncompletes (e152) [e158 pending: variance-vs-placement; e154 pending:\nglobal-vs-per-fact]."
assert o_p in p
p = p.replace(o_p, n_p, 1)
open("scratch/day6_paper_skeleton.md", "w", encoding="utf-8").write(p)
print("e152 fold complete")
