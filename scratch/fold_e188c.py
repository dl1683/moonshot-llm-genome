# -*- coding: utf-8 -*-
"""e188 THINKING fold (corrected): T141, W022b title retirement, W022 note."""
import io, re

t = io.open("THINKING.md", encoding="utf-8").read()
assert "## T141" not in t

anchor = "## T140 —"
card = """## T141 — e188: RAW-WINS — death is priced in raw displacement; alignment is a passenger (2026-09-30 ~12:05Z)

The death-currency cell answers W022b's fork AGAINST its own
favorite, exactly per pre-registered Branch B (scratch/
e188_interp_prereg.md, written before the fold read the numbers).
CV(D at death) 0.233 vs CV(A) 0.557 across the lr grid; neutral-only
0.026 vs 0.697; and the cleanest texture: same arm, same t*=2,
three wash seeds — D at death {2.489, 2.484, 2.500}, 0.6% SPREAD,
while A spreads 3.7x. DISPLACEMENT IS THE INVARIANT; ALIGNMENT IS A
SEED LOTTERY. DIES: W022b's aligned-drift law (in its letter);
W022's rate-law mechanism candidate (the rate law keeps its
displacement reading); the framing slogan "forgetting is aligned
training" (stamped once by C13-1, now killed by EVIDENCE — the
honest arc). STRENGTHENS: T139's displacement GATE (the tightest
invariance the wash arc has produced). REDUCES: T137 to the
static/learned contrast (g3K's kappas; no integral needed).
POIGNANT: the fast arm's alignment flips POSITIVE post-kill — the
organism's adaptation walks toward the fact readout's ascent
direction; it returns to the grave it dug. VOCABULARY REJECTED:
install-vs-wash cos in [-0.034, -0.018] everywhere — the wash is
the corpus's adaptation direction, not the fact's negation
(T138's defence resolves by measurement). e189's census survives
as the mechanism of the FLIP (W024), not the currency.

"""
assert anchor in t
t = t.replace(anchor, card + anchor, 1)

# W022b title retirement (body kept)
i = t.index("## W022b ")
line = t[i:t.index("\n", i)]
assert "KILLED-BY-E188" not in line
t = t.replace(line, line + (" [KILLED-BY-E188 per its own pre-registered Branch B, 2026-09-30: RAW-WINS — "
                            "death is priced in RAW displacement (0.6% seed spread at matched t*); the alignment "
                            "integral was the wrong currency; retired with honor]"), 1)

# W022 retirement note (its next card is W020)
m = re.search(r"^## W022 — .*$(.*?)(?=^## W020)", t, re.M | re.S)
assert m, "w022 body"
body = m.group(1).rstrip("\n")
amend = """

[E188 VERDICT, 2026-09-30: this card's unification role ENDS — the
alignment integral (the death currency) was measured and LOST to
raw displacement (T141, e188); the rate law keeps its displacement
reading; alignment demoted to passenger (real, 15-100x random,
seed-lottery at death). The card's kappa-contrast thread survives
in T137's reduced form. Kept for the record of a favorite that
died well.]"""
t = t[:m.start(1)] + body + amend + "\n\n" + t[m.end(1):]

def amend_card(t, card, text):
    m = re.search(r"^## " + card + r" .*$(.*?)(?=^## T\d)", t, re.M | re.S)
    assert m, card
    b = m.group(1).rstrip("\n")
    return t[:m.start(1)] + b + "\n\n" + text + "\n\n" + t[m.end(1):]

t = amend_card(t, "T137", """[E188 AMENDMENT — REDUCED, 12:05Z]: the trajectory hypothesis loses
its integral layer (RAW-WINS: the alignment-weighted currency lost
to raw displacement, CV 0.233 vs 0.557; pre-registered Branch B
executed). WHAT REMAINS: the STATIC-vs-LEARNED contrast (g3K's
kappas 4-10x at matched per-coordinate RMS, n=1, replication still
owed) — "learned paths reach the gate at 1x; static jumps need
4-10x" — with no claim about alignment as the currency.""")
t = amend_card(t, "T138", """[E188 RESOLUTION, 12:05Z]: the vocabulary clause RESOLVED BY
MEASUREMENT — install-vs-wash cos in [-0.034, -0.018] on all four
organisms; |cos| < 0.3 everywhere -> task-arithmetic vocabulary
REJECTED; the wash is the corpus's adaptation direction, not the
fact's negation. R3d resolves: report the cosines, reject the
vocabulary.""")
t = amend_card(t, "T139", """[E188 AMENDMENT — THE GATE STRENGTHENED, 12:05Z]: e188's RAW-WINS
is the displacement gate's best evidence: D at death varies 0.6%
across three wash seeds at matched t* (2.489/2.484/2.500) — the
tightest invariance the wash arc has produced — while the aligned
currency spreads 3.7x (a seed lottery). The decomposition reads:
CLOCK = Adam's normalization (opt1); GATE = raw displacement
(e188); CURRENCY-of-reaching = open (opt1b running, opt1c
dispatching); ALIGNMENT = passenger (flips positive post-kill on
the fast arm).""")
io.open("THINKING.md", "w", encoding="utf-8").write(t)
print("THINKING fold OK")
