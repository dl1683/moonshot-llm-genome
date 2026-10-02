# -*- coding: utf-8 -*-
"""Fold g13: NOTES, T200, T198 amendment, QUEUE, STATE."""
import io, re, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## g13 — the symmetry closure: GRADED — the transplant SYMMETRIC at n=2 (the locked delta holds the cons root 0.809/0.870 as the cons delta crushed the locked 0.109: the crush delta-carried both directions); the registered delta-form removal SPARES g1e (0.708, reproducing g12's co-read exactly) but FAILS at g1f (0.295, delta.s0 3x smaller: the g1f crush is s_0-INDEPENDENT) — T198's "made in the normalizer" is a g1e-draw property, not the axis's law (2026-10-02 ~21:05Z) — DONE

WHAT WE DID: the fourth symmetry cell (the locked delta at g1e's
root), the registered delta-form removal (g12's co-read promoted,
reproduction-gated), and the g1f replicates; 23/23 gates PASS
(the three-root chain verbatim); 30.3s CPU.

WHAT WE SAW (T200): (1) THE 2x2 CLOSES SYMMETRIC — the locked
delta HOLDS the cons root (0.8088; the g1f replicate 0.8701) as
the cons delta crushed the locked root (0.1088): THE CRUSH IS
DELTA-CARRIED IN BOTH DIRECTIONS, n=2. (2) THE REGISTERED
DECISIVE CELL SPARES at g1e (0.7075, reproducing g12's co-read
at |d| 0.0 — the promotion clean). (3) THE OPEN REPLICATE FAILS —
at g1f the same removal reads 0.2951 with delta.s0 only -0.0188
(3x smaller than g1e's -0.0565): THE g1f CRUSH IS ESSENTIALLY
s_0-INDEPENDENT. (4) T198's "made in the normalizer" DOWNGRADED:
the s_0-lethality is a DRAW-SPECIFIC concentration (g1e yes, g1f
no), not the cons axis's law — the mechanism carries a per-draw
component; "the damage is elsewhere in the step" at g1f. (5)
First-order predicts nothing anywhere (H: +0.0000 vs -1.745
actual). SUCCESSORS: what carries the g1f crush (the orthogonal
complement of s_0); a third cons draw; the +2 re-capture phase.
HONESTY: n=1 per cell; cells F the same read promoted; H/G the
only independent draws.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T199 —"
card = """## T200 — g13: the crush delta-carried in both directions; the normalizer clause draw-specific (2026-10-02 ~21:05Z)

The symmetry cell delivers the mechanism table's honest finish:
the 2x2 is symmetric (the crush travels with the delta, both
directions, n=2 — the DELTA-CARRIES clause licensed), but the
s_0-decomposition's lethality is a g1e draw's property, not the
axis's law — g1f's crush survives its s_0-removal untouched.
THE MECHANISM'S FINAL HONEST FORM: the crush is carried by the
AdamW-realized step (not the gradient); within the step, WHICH
component carries the damage VARIES BY DRAW (s_0 at g1e; elsewhere
at g1f); first-order predicts nothing; the wall re-captures by
+2 everywhere. THE LAB'S LAW APPLIED TO THE LAB'S OWN MECHANISM
CLAIM: the delta-carrying (shape) replicates; the s_0-attribution
(height) is a draw. THE SUCCESSORS: the orthogonal complement at
g1f; a third cons draw; the +2 re-capture phase's anatomy.

"""
assert anchor in t and "## T200" not in t
t = t.replace(anchor, card + anchor, 1)

m = re.search(r"^## T198 — .*$(.*?)(?=^## T197)", t, re.M | re.S)
assert m, "t198"
body = m.group(1).rstrip("\n")
amend = """

[G13 AMENDMENT ~21:05Z]: the normalizer clause DOWNGRADED to
draw-specific — the registered delta-form removal spares g1e
(0.708, reproduced) but fails at g1f (0.295, delta.s0 3x smaller
there: the g1f crush s_0-independent). The crush is delta-carried
both directions (licensed); WHICH component carries it varies by
draw. See T200."""
t = t[:m.start(1)] + body + amend + "\n\n" + t[m.end(1):]
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
q = q.replace("| DISPATCHED 20:25Z — bars: SYMMETRY-CLOSED / ASYMMETRY / GRADED |",
              "| DONE 21:05Z (T200: GRADED — the transplant SYMMETRIC n=2 (the crush delta-carried both directions); the registered removal spares g1e (0.708) but FAILS g1f (0.295, s0-independent there); T198's normalizer clause draw-specific; first-order predicts nothing) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T21:06:00Z"
st["current_experiment"] = ("g13 FOLDED (T200: the crush delta-carried both directions; the normalizer clause "
                            "draw-specific). Fleet: e223 (the e131 exposure) alone."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("g13 folded")
