# -*- coding: utf-8 -*-
"""Fold e188 (RAW-WINS, pre-registered Branch B): NOTES, T141, W022b/W022
retirements, T137/T138/T139 amendments, paper rewrites, QUEUE, STATE."""
import io, json, re

# --- NOTES ---
n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e188 — the death-currency cell: RAW-WINS — death is priced in RAW displacement; alignment is a passenger (2026-09-30 ~12:05Z) — DONE

WHAT WE DID: eval-only CPU over the e180 lr-grid snapshots (42
checkpoints sha1-hashed, every battery read reproduced its committed
trajectory row before use); the trajectory row (alignment integral
A vs raw D at death per arm), the static row (g3K's iso arms
re-priced in the same currency), and the install-vs-wash cosine on
four organisms. Deviation documented: only 3e-5/1e-5 arms live
under e180_*; the 1e-3 arm is e176n_neutral_*, the 1e-4 arm
e176n_lr1e4_* (the original extinction stream, coarse {50,300}
grid — flagged wherever it enters a number); the 1e-5 arm censored
(alive at +300, one-sided bound).

WHAT WE SAW (T141): RAW-WINS — across {1e-3, 1e-4, 3e-5}: CV(D at
death) 0.233 vs CV(A) 0.557; neutral-only 0.026 vs 0.697; and the
cleanest texture: same arm, same t*=2, three wash seeds — D at
death {2.489, 2.484, 2.500} (0.6% spread) while A spreads 3.7x.
DISPLACEMENT IS THE INVARIANT; ALIGNMENT IS A SEED LOTTERY. W022b's
aligned-drift law dies in its letter (its own pre-registered
Branch B); the rate law t* ~ lr^-1.16 keeps its displacement
reading. T139's displacement GATE is STRENGTHENED (the tightest
invariance the wash arc has produced); T137 REDUCES to the
static/learned contrast (no integral needed; g3K kappas n=1,
replication still owed). POIGNANT: the lr 1e-3 arm's per-pair
alignment flips POSITIVE after the kill — post-death adaptation
walks toward the fact readout's ascent direction. VOCABULARY:
install-vs-wash cos in [-0.034, -0.018] on all four organisms —
task arithmetic REJECTED; the wash is the corpus's adaptation
direction, not the fact's negation. TWO-CURRENCY TABLE stands
softer: static kills at |A| <= 0.009 on 4-10x raw displacement.
HONESTY: quadrature (the 1e-4 kill in one 50-step segment); single
ZEPHYRA family, n=1 per arm; pricing not causation (opt1b/opt1c/
opt2 own the mechanism); the store's wash direction remains the
only strongly-aligned object measured (the -0.44 was a different
estimator — informative disagreement, not contradiction).
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

# --- T141 card ---
t = io.open("THINKING.md", encoding="utf-8").read()
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
assert anchor in t and "## T141" not in t
t = t.replace(anchor, card + anchor, 1)

# --- W022b retirement (title marker, body kept) ---
old_title = t[t.index("## W022b "):]
old_title = old_title[:old_title.index("\n")]
new_title = old_title + " [KILLED-BY-E188 per its own pre-registered Branch B, 2026-09-30: RAW-WINS — death is priced in RAW displacement (0.6% seed spread at matched t*); the alignment integral was the wrong currency; retired with honor]"
assert "KILLED-BY-E188" not in t
t = t.replace(old_title, new_title, 1)

# --- W022 partial retirement note (append to card) ---
m = re.search(r"^## W022 — WONDER.*$(.*?)(?=^## W021)", t, re.M | re.S)
assert m
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
io.open("THINKING.md", "w", encoding="utf-8").write(t)
print("NOTES/T141/W022b/W022 done")
