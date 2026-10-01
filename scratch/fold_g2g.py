# -*- coding: utf-8 -*-
"""Fold g2g: NOTES, T152, T131 amendment, W023 note, day7 C7 fill,
paper R6(b), QUEUE DONE, STATE."""
import io, json, re

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## g2g — the rhythm's controls: A THERMOSTAT ON A LEASH — the organ earns its keep at 1x; the band was partly a floor (2026-10-01 ~16:55Z) — DONE

WHAT WE DID: the R56-critic-forced controls on the locked root (8
arms x 2 runs on cuda; all gates PASS both runs; L1 bit-reproduced
g2c's stored schedule 10/10; run 2 = deterministic rerun after a
rider instrument repair (pre-window parity bug, no bar touched)
doubling as the reproducibility check; two pause-and-wait cycles
under the neighbor's 80-81C job — never migrated).

WHAT WE SAW (T152): (1) SELF-TIMED-THERMOSTAT FIRES (run-stable:
rates [0.0333, 0.0333, 0.0400, 0.0400] across 8x threat, monotone)
BUT ON ITS LETTER ONLY — the response is a WEAK STEP (10 -> 12
events), ceiling-saturated: at 4x EVERY spacing sits at the
refractory floor (frac-at-floor 1.0) and the cycle-median collapses
0.619 -> 0.240 -> 0.0017 — fires without maintaining. At 0.5x =
1x exactly. (2) FIXED-PERIOD-ARTIFACT does not fire (there IS
threat response). (3) SELF-TIMED-WINS IS RUN-STABLE AT 1x ONLY
(+0.072/+0.076 both runs vs the matched-count fixed schedule, same
frozen ruler; the critic's 0.693 co-read superseded — it compared
an endpoint against cycle-medians); the 2x leg is float-fragile
(one near-threshold check flipped the rung 0.240 -> 0.167):
FIXED-MATCHES-OR-WINS is the safer standing letter with the 1x win
co-reported. (4) REFRACTORY-REAL FIRES: at r8, 5/14 spacings land
<20 (min 8) — the old "100% in 20-45" band was partly a
CONSTRUCTION FLOOR (disclosed per the bar's letter) — yet 9/14
stay in-band (the wash's decay clock still shapes most intervals)
and maintenance IMPROVES at the shorter refractory (cycle-median
0.671 vs 0.619; duty 0.74). (5) THE W023 RIDER reads the
mini-shock in monitor language: post-event the monitor jumps
(+0.21 at 1x), keeps rising briefly, then decays with ACCELERATING
steepness (late-half steeper in 60-70% of events) — the replay's
protective mismatch wearing off; at 4x the read is pinned ~0 (the
replay's step is as big as the wash). THE RHYTHM'S CLAIM AFTER
g2g: "threat-responsive within [0.5x, 2x]; autonomy worth +0.07 of
cycle-median at operating threat; refractory-tunable" — n=1 root,
one wash seed per rung; the seed-replicate ladder LICENSED by the
thermostat firing (owed next).
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T151 —"
card = """## T152 — g2g: a thermostat on a leash — the rhythm's honest operating envelope (2026-10-01 ~16:55Z)

The controls land as three teaches. (1) THE ORGAN IS
threat-responsive but WEAKLY: 10 -> 12 events across 8x threat,
ceiling-saturated — at high threat every spacing pins to the
refractory floor and maintenance collapses (0.62 -> 0.002): a
thermostat on a leash, firing without maintaining. The autonomy is
real but bounded by the gate's fixed thresholds. (2) THE HEAD-TO-
HEAD RESOLVES AT 1x: the organ beats the matched-count fixed
schedule by +0.07 in both runs on the registered same-ruler
comparison — the critic's 0.693 co-read was endpoint-vs-median and
is superseded — but the 2x leg rides one float-nondeterministic
check, so FIXED-MATCHES-OR-WINS is the safer letter with the 1x
win co-reported. AUTONOMY IS WORTH +0.07 OF CYCLE-MEDIAN AT
OPERATING THREAT — a small, real, priced number. (3) THE BAND WAS
PARTLY A FLOOR: at refractory 8 five spacings land below 20 — the
"100% in 20-45" construction disclosed — yet the wash's decay
clock still shapes 9/14 intervals AND maintenance IMPROVES at the
shorter refractory (0.671, duty 0.74): the refractory is a TUNABLE,
and shorter is better. THE W023 RIDER: the mini-shock lives in the
monitor channel (jump, brief rise, accelerating decay — the
replay's protective mismatch wearing off) even though the
alignment form died; at 4x the replay's step is as big as the wash
and the signal pins to zero. THE CLAIM'S FINAL FORM THIS CELL:
"threat-responsive within [0.5x, 2x]; autonomy worth +0.07 at
operating threat; refractory-tunable" — and the seed ladder is
licensed. W026's managed-bleed noun gains its price tag: the
re-orientation schedule is worth +0.07 over a fixed schedule, at
the cost of a ceiling.

"""
assert anchor in t and "## T152" not in t
t = t.replace(anchor, card + anchor, 1)

m = re.search(r"^## T131 — .*$(.*?)(?=^## T130)", t, re.M | re.S)
assert m, "t131"
body = m.group(1).rstrip("\n")
amend = """

[G2G RESOLUTION, 2026-10-01 ~16:55Z]: the controls landed. The
band was partly a construction floor (at refractory 8, 5/14
spacings <20 — disclosed); the wash's clock still shapes 9/14;
maintenance IMPROVES at shorter refractory. The head-to-head:
the organ wins at 1x (+0.07 both runs, the registered
same-ruler comparison; the 0.693 co-read was endpoint-vs-median,
superseded); FIXED-MATCHES-OR-WINS is the safer letter beyond 1x.
The final claim: threat-responsive [0.5x, 2x], autonomy +0.07 at
operating threat, refractory-tunable; the seed ladder licensed."""
t = t[:m.start(1)] + body + amend + "\n\n" + t[m.end(1):]

# W023 note
i = t.index("## W023 ")
line = t[i:t.index("\n", i)]
t = t.replace(line, line + " [G2G RIDER UPDATE ~16:55Z: the mini-shock LIVES in the monitor channel — jump, brief rise, accelerating decay — even though the alignment form died; at 4x the replay's step is as big as the wash and the signal pins to zero]", 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

# day7 C7 fill
d = "DAY_SEVEN_REPORT.md"
s = io.open(d, encoding="utf-8").read()
old = "Written with one cell (g2g, the rhythm's controls) still out; its\nverdict amends C7 below and nothing else."
new = """The g2g verdict landed and is folded into C7: a thermostat on a
leash — threat-responsive within [0.5x, 2x], autonomy worth +0.07
of cycle-median at operating threat (run-stable at 1x; the safer
letter elsewhere), refractory-tunable with shorter better; the
20-45 band partly a construction floor (disclosed)."""
assert old in s
s = s.replace(old, new, 1)
io.open(d, "w", encoding="utf-8").write(s)

# paper R6(b) final form
p = "scratch/day6_paper_skeleton.md"
s = io.open(p, encoding="utf-8").read()
old = """(b) THE RHYTHM — a zero-parameter rehearsal organ
   (cue pool + onset monitor + replay gate) self-times resurrection events in the 20-45 band,
   three seeds one waveform (medians 0.587/0.602/0.615, duty ~70%);
   scoping: timing root-robust 2/2, amplitude root-draw-bound (the root
   recipe is a strength lottery 0.591/0.684/0.711; g2f base-redraw in
   flight)"""
if old not in s:
    # fallback: locate the R6(b) block loosely
    old = "self-times resurrection events in the 20-45 band"
    assert old in s, "r6b"
    i2 = s.index("(b) THE RHYTHM")
    j2 = s.index("(c) THE CONE", i2)
    s = s[:i2] + """(b) THE RHYTHM (g2g-controlled): a zero-parameter rehearsal organ
   self-times resurrection events — threat-responsive within
   [0.5x, 2x] (a weak step, ceiling-saturated beyond), autonomy
   worth +0.07 of cycle-median over a matched fixed schedule at
   operating threat (run-stable; the safer letter
   FIXED-MATCHES-OR-WINS beyond), refractory-tunable with shorter
   better (the 20-45 band partly a construction floor, disclosed);
   timing root-robust 3/3, amplitude root-draw-bound (T136);
""" + s[j2:]
else:
    s = s.replace(old, """(b) THE RHYTHM (g2g-controlled): a zero-parameter rehearsal organ
   self-times resurrection events — threat-responsive within
   [0.5x, 2x] (a weak step, ceiling-saturated beyond), autonomy
   worth +0.07 of cycle-median over a matched fixed schedule at
   operating threat (run-stable; the safer letter
   FIXED-MATCHES-OR-WINS beyond), refractory-tunable with shorter
   better (the 20-45 band partly a construction floor, disclosed);
   timing root-robust 3/3, amplitude root-draw-bound (T136)""", 1)
io.open(p, "w", encoding="utf-8").write(s)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| g2g |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| g2g | THE RHYTHM'S CONTROLS | DONE 16:55Z (T152: THERMOSTAT-ON-A-LEASH — SELF-TIMED-THERMOSTAT fires run-stable but weak/ceiling-saturated; SELF-TIMED-WINS run-stable at 1x only (+0.07 both runs; the 0.693 co-read superseded as endpoint-vs-median); FIXED-MATCHES the safer letter; REFRACTORY-REAL fires (the band partly a floor; maintenance IMPROVES at r8); the W023 mini-shock lives in the monitor channel; the seed ladder LICENSED) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-01T16:56:00Z"
st["current_experiment"] = ("g2g FOLDED (T152: a thermostat on a leash; the wave CLOSED — T144-T151 + g2g all folded). "
                            "R60 TRIO LAUNCHING (the inherited R59 audit + this wave). Then g1bS2 (GPU) + the g2g seed ladder.")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("g2g folded")
