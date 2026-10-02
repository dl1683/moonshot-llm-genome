# -*- coding: utf-8 -*-
"""Fold g1bS6: NOTES, T170, ledger C6 final, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## g1bS6 — the fifth take adjudicated: WALL-FADES at 10M — the wall's first scale verdict; the strict 2.74M wall did NOT survive 10x; what survives is DIRECTION (1000x separation at every checkpoint >= +10); the first-step blindness is structural (2026-10-02 ~10:25Z) — DONE

WHAT WE DID: the wall arms on g1bS5's formation-curve PEAK root
(0.20 rms; root g-12 0.7677) loaded BIT-EXACT (the new G-ROOTLOAD
gate |d| 0.0 on all cells); the REGISTERED pre-run deviation
honored (the root 0.0123 below the express bar; the curve licenses
the arms; disclosed on every artifact); the owner envelope held
(4 gated launches, 23 heat pauses, 180s cooldowns).

WHAT WE SAW (T170): WALL-FADES — C died at +1 (D_kill = exactly
one AdamW step, T139 at 10x again) and NO rung on the {1x,2x,4x}
R_rms ladder held the strict every-checkpoint bar: EVERY RUNG
BREACHED AT +1 (W1 shock 0.094; W2 0.0015; W3 tracks C — its ball
contains step 1). NOT the freezing escape (W1's CE@300 0.97 <
root 1.70 — the walled organism ADAPTS; the tax re-priced +0.18
vs +0.53 at 2.74M). THE TEXTURE: (a) THE TIGHTER BALL HOLDS
BETTER — flat-phase retention W1 0.895 >> W2 0.464 >> W3 0.0002
(g1bS4's ordering replicated on the strong root; W1 missed the
0.9x-root secondary by 0.005, a late fade); (b) THE WALL'S
FIRST-STEP BLINDNESS IS STRUCTURAL AT 10M — one AdamW step (3.16
raw) EXCEEDS EVERY RUNG on the registered ladder, and the +1
projection shock breaches the bar before the flat phase ever
starts: the R-dial was minted at 2.74M's step scale. THE 2.74M
WALL DID NOT SURVIVE 10x IN ITS STRICT FORM; WHAT SURVIVES IS
DIRECTION: the rms-matched rung separates the fact from the
control by ~1000x at every checkpoint >= +10. HONESTY: n=1
host/wash-seed/fact — the rung was SCALE; the replicate ladder
owed only per the design's own scoping.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T169 —"
card = """## T170 — g1bS6: the wall's scale verdict — WALL-FADES, direction survives, and the first step is the whole story (2026-10-02 ~10:25Z)

The six-take saga closes with an honest negative and a mechanism:
the strict 2.74M wall did NOT survive 10x — every rung breached at
+1 because ONE AdamW step (3.16 raw) exceeds every radius on the
2.74M-minted ladder: THE WALL'S FIRST-STEP BLINDNESS IS
STRUCTURAL, not a tuning artifact — the commit-then-project design
is blind between commit and the first projection rescale, and at
10M that window is exactly where the kill lands (T139's clock: one
step). WHAT SURVIVES IS DIRECTION: the rms-matched rung holds the
fact ~1000x above the control through the whole flat phase — the
ball still separates memory from death; it just cannot promise
every-checkpoint continuity when the step outruns the radius. THE
TIGHTER-BALL ORDERING replicates on the strong root (0.895 >>
0.464 >> 0.0002) — g1bS4's texture was real. THE TAX RE-PRICED:
+0.18 at 10M (vs +0.53) with the walled organism ADAPTING (CE
below root) — the freeze reading dead at both scales now. THE
SAGA'S LEDGER: divergence -> near-miss -> inversion -> the curve
-> the verdict; three recipe casualties, one cure pattern, one
tuned window, one structural limit. THE WALL'S HONEST FINAL FORM:
a displacement budget that separates memory from death at every
scale tested, holding strict continuity only where the step fits
the radius — the fix candidate (a first-step-aware projection)
named for the next life of the g-series.

"""
assert anchor in t and "## T170" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

c = "scratch/claims_ledger.md"
s = io.open(c, encoding="utf-8").read()
old = "scale OPEN-ENCOURAGING — g1bS BLOCKED at the base gate; g1bS2's base license WORKED but the e113 consolidate recipe is the THIRD scale casualty (the recipe stack re-licenses component-by-component; g1bS3 named); the g0 co-report is g1b-shaped at 10x (W1 flat ~0.5 through +300; the Adam-clock = lr*sqrt(P) confirmed at 10M) — a shape, not an adjudicated bar"
assert old in s, "c6"
s = s.replace(old, "scale ADJUDICATED (g1bS6): WALL-FADES at 10M — the strict 2.74M wall did not survive 10x (every rung breached at +1: one AdamW step exceeds every 2.74M-minted radius; the first-step blindness is structural); WHAT SURVIVES IS DIRECTION (~1000x fact-control separation through the flat phase, W1 retention 0.895); the tighter-ball ordering replicated; the tax re-priced +0.18 with the organism adapting; the recipe stack scale-bound (three casualties, a tuned consolidation window 0.15-0.25 rms)", 1)
io.open(c, "w", encoding="utf-8").write(s)

q = io.open("QUEUE.md", encoding="utf-8").read()
q = q.replace("| DISPATCHING 10:06Z — short bursts; the envelope-log live |",
              "| DONE 10:25Z (T170: WALL-FADES at 10M — the first scale verdict; every rung breached at +1 (one AdamW step exceeds every 2.74M-minted radius: the first-step blindness structural); the direction survives (1000x separation >= +10); the tighter-ball ordering replicated; the tax +0.18, adapting) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T10:26:00Z"
st["current_experiment"] = ("g1bS6 FOLDED (T170: WALL-FADES at 10M - the scale saga CLOSED with an honest negative, "
                            "direction surviving, and a structural limit named). Fleet: e204 (CPU, the support "
                            "measurement) alone."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("g1bS6 folded - the saga closed")
