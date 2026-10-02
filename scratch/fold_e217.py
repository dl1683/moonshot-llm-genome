# -*- coding: utf-8 -*-
"""Fold e217: NOTES, T192, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e217 — the third wash draw: DEEP-MEAN-REVERTS + GRADED — the +80 drift COLLAPSED toward 1 (wash 1's depth was the outlier; the state function deepens its claim); the relational signature three-draw-stable (rho 0.972/0.966; the decided families rock-steady, the mixed band carrying the lottery) (2026-10-02 ~17:00Z) — DONE

WHAT WE DID: a third independent 5e-5 draw (seed 21703; the
e182c2 machinery verbatim; all gates at dp 0.0; the envelope
held: 3 launch cycles, bursts 2.6-9.0s); the three-wash census
with the relational signature at n=3.

WHAT WE SAW (T192): (1) DEEP-MEAN-REVERTS — THE +80 DRIFT DID
NOT GROW; IT COLLAPSED toward 1 on BOTH deepest batteries (near
r21 0.857 -> r31 0.910; tmpl 0.804 -> 0.955; the distances
0.143->0.090 and 0.196->0.046): WASH 1'S DEPTH WAS THE OUTLIER;
the state function deepens its claim — path-independent at
mid-depth, approximately so at depth. (2) MID-DOSE-TIGHTENS
missed the strict 0.15 bar on the n=3 NEAR battery ALONE (its r31
0.819; fact 0.917 / ctrl 1.085 / tmpl 0.927 all inside) — THE
MID-DOSE STATE FUNCTION HOLDS FOR EVERY 12+-ITEM BATTERY AT n=3;
the 3-item battery carries the draw-noise; GRADED the honest
letter. (3) THE RELATIONAL SIGNATURE AT n=3: the family
hold-ratios ROCK-STEADY where decided (lang 0.87/0.82/0.80;
founder-anchor 0.81/0.81/0.86; cap-cur 0.20/0.24/0.20) and wander
only in the MIXED band (rev-capital spread 0.174; near 0.105;
product 0.074); rho w3xw1 0.972 / w3xw2 0.966 (e215's 0.939
reproduced) — THE SORTING KEY IS THREE-DRAW-STABLE, and the
families' own mid-band is where the path-lottery lives. 26
HOLD-all-3 / 12 COLLAPSE-all-3 / 12 discordant (every discordance
a one-class flicker at the boundaries). (4) The +10 shallows stay
stream-typed (the direction flips across draws). HONESTY: n=3
draws; the wash-1 arm the CPU replay; the plot-only re-pass
disclosed.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T191 —"
card = """## T192 — e217: the state function deepens; the signature is three-draw-stable (2026-10-02 ~17:00Z)

The third draw closes T185's thread with the better answer: the
deep drift MEAN-REVERTED (wash 1 was the extreme deep draw) — the
erosion's state-function claim now covers the mid-dose exactly and
the deep dose approximately, with the path-lottery confined to the
shallows and the mixed families' mid-band. THE RELATIONAL
SIGNATURE'S STRONGEST FORM YET: the sorting key three-draw-stable
at rho 0.97, the decided families' hold-ratios steady to within
0.04-0.07 across three independent streams, and every discordance
a boundary flicker — the wash sorts the same way every time; the
uncertainty lives only where families are undecided. THE 124M
THREAD'S FINAL MAP: the erosion is a mid-dose state function
(approximately deep), stream-typed at the shallows; the sorting
key family x nonlinear-height x the third dimension; the third
dimension beyond height with the hunt owed; and the whole thing
replicating at n=3 draws.

"""
assert anchor in t and "## T192" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e217 |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| e217 | THE THIRD WASH DRAW | DONE 17:00Z (T192: DEEP-MEAN-REVERTS — the +80 drift collapsed toward 1 (wash 1 the outlier; the state function deepens); the mid-dose function holds for every 12+-item battery at n=3; THE SIGNATURE THREE-DRAW-STABLE (rho 0.972/0.966; the decided families steady within 0.04-0.07; the lottery confined to the shallows and the mixed band)) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T17:01:00Z"
st["current_experiment"] = ("e217 FOLDED (T192: the state function deepens; the signature three-draw-stable). Fleet "
                            "0; the named threads: the probe-feature hunt (independent entrenchment first); the "
                            "exposure-immunity follow-ups."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e217 folded")
