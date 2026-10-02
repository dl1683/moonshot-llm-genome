# -*- coding: utf-8 -*-
"""Fold e200: NOTES, T164, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e200 — the deepening test: GRADED — the concentration ARRIVES, UN-FORMS, and RETURNS at its deepest exactly at the killing step; the rotation is a rotating object, not a converging pursuit (2026-10-02 ~08:30Z) — DONE

WHAT WE DID: e197's half-step walk rebuilt verbatim as the ray
factory (journal gated row-by-row, 9.5e-07 TEXTURE tier — threads
4 vs 8, the owner envelope's cap, disclosed); the sign(g_t) rays
from its root at every ALIVE t; 15/15 gates PASS (u1/u2 md5s BIT;
D_kills BIT at 6.6e-08); 141s CPU, threads 4, load checked.

WHAT WE SAW (T164): THE CURVE — t1 SOFT (3.169, reproducing e197's
committed flight ray to 8 decimals) -> t2 CONCENTRATED (0.663;
e197's unadjudicated root_u2 now read as onset) -> t3 UN-FORMS
(1.292) -> t4 RE-CONCENTRATES AT ITS DEEPEST (0.397 — the killing
step's own direction; depth comparable to org1's 0.171 and
MIRABEL's 0.427) -> death at t5 (recompute 1.597 = e197's
committed ABSENT). ONSET-DEEPENS FAILS (no hold); ONSET-STALLS
FAILS (it arrived); GRADED — and the texture is the finding: THE
CONCENTRATION IS NON-MONOTONE, and consecutive fronts are ALL
mutually anti-correlated (cos -0.26/-0.31/-0.34/-0.35): THE FRONT
ALTERNATES ONTO AND OFF THE SUPPORT — a rotating object, not a
converging pursuit — AND THE ORGANISM DIES EXACTLY WHEN THE FRONT
LANDS ON IT DEEPEST. THE WHEN (T163) SURVIVES as arrival-at-first-
concentration (t=2 here, the second t=2 arriver after MIRABEL);
the monotone-deepening SHAPE amends to alternation. HONESTY: n=1
lineage, one stream; the COUNTERFACTUAL-WASH caveat rides (alive
only at half the natural step; the curve belongs to this
construction's alive window); cross-organism magnitudes never
compared (shape and arrival only). FOLLOW-ONS NAMED: the
front-rotation census (do org1/MIRABEL's consecutive fronts also
anti-correlate — is the alternation universal or the wash's?); the
t4-deepest replicate before it is a noun.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T163 —"
card = """## T164 — e200: alternation — the front rotates onto and off the support, and death is the deepest landing (2026-10-02 ~08:30Z)

Given the longest alive window on record, the onset curve answers
NO to monotone deepening and YES to something better: the
concentration ARRIVES (t2), UN-FORMS (t3), and RETURNS AT ITS
DEEPEST at exactly the killing step (t4, ratio 0.397). The
geometry explains the shape: every consecutive front pair is
mutually ANTI-correlated (-0.26 to -0.35) — the front is a
ROTATING OBJECT that alternates onto and off the fleeing support,
and THE ORGANISM DIES WHEN THE ROTATION LANDS ON IT DEEPEST. THE
MECHANISM PICTURE REWRITES AGAIN: not a pursuit that converges
(e194's reading) but a ROTATION that periodically lands; survival
is the phase of the rotation relative to death. THE WHEN ACCOUNT
(T163) HOLDS (arrival at first concentration: org1 t1, MIRABEL t2,
org2-half t2); the SHAPE account amends: alternation, not
deepening. THE BLEED'S RE-ORIENTATION and the front's rotation are
THE SAME OBJECT SEEN TWICE: the bleed's steps turn away from the
lethal direction and live; the sign front's rotation periodically
lands on it and kills — both are the rotation's phase. THE
COUNTERFACTUAL CAVEAT rides honestly (this lineage lives only at
half the natural step); the census follow-on decides whether the
alternation is universal (org1/MIRABEL's committed fronts' mutual
correlations — a pure desk check on committed data).

"""
assert anchor in t and "## T164" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e200 |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| e200 | THE DEEPENING TEST | DONE 08:30Z (T164: GRADED — the concentration arrives (t2 0.663), UN-FORMS (t3 1.292), and RETURNS AT ITS DEEPEST at the killing step (t4 0.397); consecutive fronts ALL anti-correlated — a rotating object, not a pursuit; death = the deepest landing; the WHEN holds, the shape amends to alternation; the counterfactual-wash caveat rides) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T08:31:00Z"
st["current_experiment"] = ("e200 FOLDED (T164: alternation — the front rotates onto/off the support; death = the "
                            "deepest landing). Fleet: g1bS4-recovery (GPU, gated) + the front-rotation census "
                            "DISPATCHING (a desk check on committed data — zero compute)."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e200 folded")
