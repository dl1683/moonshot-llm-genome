# -*- coding: utf-8 -*-
"""Fold e204: NOTES, T171, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e204 — the support measurement: GRADED — the survivor's leg is PARTIAL (the rank-order stands, the mechanism does not fully); THE SUPPORT ITSELF ROTATES (monotone decorrelation toward orthogonality at death) (2026-10-02 ~10:45Z) — DONE

WHAT WE DID: the fact's LOCAL sensitivity directions (the g-12
readout gradient, FD-gated — the first fact gradients on any f2
state) at every state of the half-step lineage; all five fronts
md5-BIT; a registered sign-identity resolution (the dispatch's
literal formula vs its intent; both columns; the verdict
orientation-invariant); 17s CPU.

WHAT WE SAW (T171): THE LANDING METRIC TRACKS THE KILL-RATIO ONLY
PARTIALLY — Spearman +0.60 (the e197-committed t1/t2 prefix
PERFECT at +1.0: the softest front the most anti-aligned, the
first concentrated front the max) but THE DEEPEST LANDING (t4,
the killing step) IS NOT THE MOST ERASING-ALIGNED (t2 is; t4
slightly anti-aligned at -0.066). DEATH-AT-DEEPEST-LANDING KEEPS
ITS 4/4 RANK-ORDER BUT NOT THE FACT-DIRECTED MECHANISM LEG;
neither is it pure geometry (+0.60 > 0). THE FREE FINDINGS:
(1) THE SUPPORT ITSELF ROTATES — the consecutive-sensitivity
cosines fall monotonically 0.776 -> 0.680 -> 0.564 -> 0.333 ->
0.192 (nearly orthogonal at death): THE FLEEING SUPPORT'S OWN
ROTATION, now measured — the fact's sensitivity direction
decorrelates steadily under the wash; (2) the static g-ray proxy
is CONFIRMED INADEQUATE (cos(s_t, u_g) ~ +-0.04 everywhere —
T166's suspicion quantified); (3) the death landing-point read
cos(u4, s5) = +0.146 is the table's LARGEST positive alignment
(context only); (4) t3 BREAKS the ordering (the un-formed front
the MOST anti-aligned). HONESTY: n=1 lineage, 4 points (Spearman
in 0.2 quanta); the counterfactual-wash caveat; the
alignment-reads-never-predict stamp (T157) carried.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T170 —"
card = """## T171 — e204: the support measured — the survivor partial, and the second rotation found (2026-10-02 ~10:45Z)

The survivor's missing leg is measured and it is PARTIAL: the
landing metric correlates with the kill-depth (+0.60; the early
prefix perfect) but the killing step's own front is not the most
fact-erasing-aligned — the 4/4 rank-order stands as a SHAPE; the
fact-directed mechanism advances no further than "mostly". THE
DAY'S SECOND ROTATION IS THE FREE GIFT: the fact's sensitivity
direction itself decorrelates monotonically under the wash
(0.78 -> 0.19, nearly orthogonal at death) — THE SUPPORT FLIES
with a steadier rotation than the front's bounce: the fact's
sensitivity ladder rotates smoothly while the sign front
alternates. THE PICTURE'S LAST FORM: two rotators — the wash's
front (period-2 bounce, the algorithm's) and the fact's own
sensitivity (a monotone drift, the fact's) — and death where they
meet under conditions only partially rank-ordered. THE STATIC
PROXY buried properly (±0.04 everywhere); the death landing-point
+0.146 the table's largest positive (context — the one hint that
the ENDPOINT alignment matters more than the along-path).

"""
assert anchor in t and "## T171" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e204 |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| e204 | THE SUPPORT MEASUREMENT | DONE 10:45Z (T171: GRADED — the survivor's leg PARTIAL (Spearman +0.60, the early prefix perfect, the killing front NOT the most aligned); THE SUPPORT ITSELF ROTATES (0.78->0.19 monotone — the second rotation found); the static proxy buried (+-0.04); the death landing-point +0.146 the largest positive, context) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T10:46:00Z"
st["current_experiment"] = ("e204 FOLDED (T171: the support measured - the survivor partial; the second rotation "
                            "found). Fleet: the redrawn interior dose DISPATCHING (GPU, gated - the formation curve's "
                            "fine ordering)."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e204 folded")
