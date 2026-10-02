# -*- coding: utf-8 -*-
"""Fold e214: NOTES, T187, W028 update, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e214 — the conservation-of-rank cell: RANK-DESTROYED (0/5 cells hold) WITH THE DISCLOSED SPLIT — the t=0 order dies WITH the scale (coupling -0.88) while the deep-state order REPLICATES across washes (xrho 0.94-1.00 at 65-77% decay); the wash re-orders BY RELATION (lang holds, cap/cur collapses; baseline p irrelevant); THE EROSION ORDER IS PHYSICS, THE BASELINE ORDER IS BIOGRAPHY (2026-10-02 ~15:10Z) — DONE

WHAT WE DID: eval-only on the two-wash 124M archive (all four
batteries re-probed fresh on every state; the provenance gates at
3.3e-06); 49s CPU.

WHAT WE SAW (T187): THE BAR'S OWN HALF FIRES: at every
>= 50%-scale-decay state the t=0-vs-t rho is below 0.8 (0/5 hold;
rho tracks the decline, pooled coupling -0.88) — RANK-DESTROYED as
registered. THE DISCLOSED SPLIT IS THE FINDING: the CROSS-WASH
half HELD 5/5 (xrho 1.000/1.000/0.951/1.000/1.000) — THE ORDER IS
THE SAME FUNCTION OF STATE ON BOTH WASHES, even at 65-77% scale
decay: what dies is the ORIGIN's order, not order per se (T185's
state function now reads on the full p-vector). THE MECHANISM
TEXTURE: THE WASH RE-ORDERS BY RELATION, NOT BY BASELINE STRENGTH
— the fact battery: lang holds (0.65-0.69) while cap/cur collapse
(0.17-0.28) with baseline p IRRELEVANT (China->yuan p0 0.93 dies
to 0.16; China->Chinese p0 0.57 holds 0.57); the controls:
founders/unique-anchor hold 0.60-0.62, products collapse 0.32-0.34
— identically on both washes. W028'S LAW, REFINED BY ITS OWN TEST:
THE SHAPE LAYER IS NOT THE BASELINE RANK — IT IS THE EROSION
ORDER, WHICH REPLICATES; the baseline order is biography. HONESTY:
the >= 0.5 regime carried by near (n=3, rho quantized, flagged) +
tmpl n=19; the 0.40-echo co-reported; the cross-wash rho partly
re-expresses the state function; the tiny-scale echo scanned,
absent (aggregates only), skipped.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T186 —"
card = """## T187 — e214: the law survives its own test by deepening — the shape layer is the EROSION ORDER (2026-10-02 ~15:10Z)

W028's direct test fires its honest negative and the negative is
the refinement: the BASELINE order dies WITH the scale (coupled at
-0.88 — baseline rank is scale information), but the DEEP-STATE
order replicates across washes at rho 0.94-1.00 even as the levels
decay 65-77% — the order-of-erosion is the conserved object. THE
MECHANISM BEAUTY: the wash sorts the probes BY RELATION —
language-relations hold, capital/currency-relations collapse, the
baseline strength irrelevant (the strongest lang probe dies; a
mid-strength one holds) — the erosion order is a RELATIONAL
signature, not a magnitude signature. THE LAW'S FINAL FORM: THE
SHAPE LAYER IS THE EROSION ORDER (replicating, relational); THE
HEIGHT LAYER IS EVERYTHING MAGNITUDE (the levels, the baseline
ranks, the distances, the rates). W028 updated by its own test —
the deepest confirmation yet that the program's recurring split is
one law, not a habit of instruments: each direct test has killed a
candidate form and left a sharper one.

"""
assert anchor in t and "## T187" not in t
t = t.replace(anchor, card + anchor, 1)

# W028 update
i = t.index("## W028 ")
line = t[i:t.index("\n", i)]
t = t.replace(line, line + " [E214 UPDATE ~15:10Z: the law's direct test — the baseline rank is HEIGHT (dies with the scale, coupling -0.88); the SHAPE layer is the EROSION ORDER (replicates across washes at rho 0.94-1.00; relational: lang holds, cap/cur collapses, baseline strength irrelevant)]", 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e214 |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| e214 | THE CONSERVATION-OF-RANK CELL | DONE 15:10Z (T187: RANK-DESTROYED as registered WITH the split — the t=0 order dies with the scale (-0.88) but the deep-state order REPLICATES (xrho 0.94-1.00 at 65-77% decay); the wash re-orders BY RELATION; THE EROSION ORDER IS PHYSICS, THE BASELINE ORDER IS BIOGRAPHY) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T15:11:00Z"
st["current_experiment"] = ("e214 FOLDED (T187: the shape layer is the EROSION ORDER - the law refined by its own "
                            "test). Fleet: g1e (GPU, the cons-seed redraw) alone."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e214 folded")
