# -*- coding: utf-8 -*-
"""Fold g1f: NOTES, T194, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## g1f — the second cons seed: CRUSH-IS-TEXTURE — the +1 crush is the CONS AXIS's own (0.2577 replicating 0.2719 against the wash/root family's 0.82-0.96); THE FLAT PHASE BULLETPROOF (0.90-0.93 flat through +300, the strictest reading of the redraw family); the +1 ledger closes in three tiers (2026-10-02 ~18:10Z) — DONE

WHAT WE DID: the g1e machinery VERBATIM at cons seed 10913 (the
only delta); all gates PASS (the cons draw genuine, L2 17.18; the
root 0.9682 — in-family and strong, no deviation); the envelope
held (12 polls all FREE, bursts 16-22s).

WHAT WE SAW (T194): (1) CRUSH-IS-TEXTURE — the second cons draw's
+1 reads 0.2577, replicating g1e's 0.2719 to the second digit
against the wash/root family's 0.82-0.96: n=2-of-3 cons draws
breaching at +1 is not draw-shaped — THE CONS AXIS CARRIES A
SYSTEMATICALLY DEEPER FIRST-STEP CRUSH; the grid's cons cell
stays BOUND at the every-checkpoint form. (2) THE FLAT PHASE IS
BULLETPROOF — W1 recovers to 0.80 by +2 and holds 0.9021-0.9185
FLAT through +300 (the 0.9xroot bar holds for the first time in
the redraw family; FLAT-AT-PIN |d| 0.016) — the wall re-captures
the fact after ONE projected step and holds it at ~0.95x its own
root. (3) THE CONS LOTTERY STAYS QUIET AT 2.74M (the second root
0.9682, above even the locked 0.9156: expression in-family twice;
the PEAK-LOTTERY remains 10M-only). (4) THE +1 LEDGER, CLOSED IN
TIERS: cons 0.27/0.26 > base 0.48 (unadjudicated) > wash/root
0.82-0.96 — THE CRUSH DEPTH IS AN AXIS PROPERTY; THE FLAT PHASE
UNIVERSAL. THE WALL'S LAW, TWO CLAUSES: THE FIRST STEP IS THE
AXIS'S (the crush carries the stream's texture); THE FLAT PHASE IS
THE WALL'S (re-capture by +2, hold ~0.95x root, every expressed
draw). THE OPEN QUESTION NOW MECHANISM: what does the
jitter-replay consolidation leave that costs the fact one extra
projected step at the first wash gradient?
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T193 —"
card = """## T194 — g1f: the crush is the axis's; the flat phase is the wall's (2026-10-02 ~18:10Z)

The second cons draw replicates the first's crush to the second
digit (0.2577 vs 0.2719) — the +1 breach is the CONS axis's own
texture, not a draw. THE +1 LEDGER'S THREE TIERS: cons ~0.26, base
~0.48, wash/root 0.82-0.96 — the crush depth is a property of
WHICH STREAM built the anchor's neighborhood, while the flat phase
(the wall's own) is universal across every expressed draw (this
cell's the strictest yet: 0.90-0.93 flat, the 0.9xroot bar holding
for the first time in the redraw family). THE WALL'S LAW, FINAL
TWO CLAUSES: the first step is the axis's; the flat phase is the
wall's. THE MECHANISM QUESTION NAMED: what does the jitter-replay
consolidation leave in the weights that costs the fact one extra
projected step at the first wash gradient — the wash/root draws
don't pay it. THE GRID, COMPLETE: wash n=3 HOLDS x root n=2 HOLDS
x cons n=2 BOUND (systematic) x base unadjudicated x 10M
direction-form — every cell adjudicated or honestly fenced, the
saga's last variance closed.

"""
assert anchor in t and "## T194" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
q = q.replace("| DISPATCHED 17:07Z — bars: CRUSH-WAS-A-DRAW / CRUSH-IS-TEXTURE / GRADED |",
              "| DONE 18:10Z (T194: CRUSH-IS-TEXTURE — 0.2577 replicating 0.2719; the crush is the CONS axis's own; the flat phase BULLETPROOF at 0.90-0.93; the +1 ledger in tiers: cons 0.26 > base 0.48 > wash/root 0.82-0.96; the wall's law two clauses: the first step the axis's, the flat phase the wall's; the mechanism question named) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T18:11:00Z"
st["current_experiment"] = ("g1f FOLDED (T194: CRUSH-IS-TEXTURE - the wall grid COMPLETE, every cell adjudicated or "
                            "fenced; the mechanism question named). Fleet: e220 (CPU desk, the token-structure test) "
                            "alone."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("g1f folded - the grid complete")
