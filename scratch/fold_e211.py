# -*- coding: utf-8 -*-
"""Fold e211: NOTES, T182, T179 amendment, QUEUE, STATE."""
import io, re, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e211 — the walled-band question: GRADED — the wall neither widens the span nor flattens the basin; THE "WIDER WALLED BAND" WAS MOSTLY THE INSTRUMENT'S SHADOW (same-instrument medians match; the committed gap = cross-instrument heterogeneity + the n=3 lottery); safety orders with SV ENERGY (2026-10-02 ~15:45Z) — DONE

WHAT WE DID: the two instruments, same-machinery at all four states
(the pristine e131 root + the three walled s300 roots, settled and
md5-gated): each root's OWN contiguous 20-step wash span spectrum
(+2 realizations/root) and FD curvatures along the top span
directions (eps ladder); the per-direction kill rays on e209's
instrument.

WHAT WE SAW (T182): THE SPECTRA MATCH (the fresh pristine PR
4.117; the walled 1.02-1.04x, inside the ~5% realization spread;
shape 0.058 bits) — CANDIDATE (b) DEAD: the wall does not widen
the exploration subspace. THE CURVATURES MATCH (geomean ratio
1.076; 0/3 under the 2x bar) — CANDIDATE (a) DEAD: the wall does
not flatten the basin along the span. THE INSTRUMENT SHADOW: the
SAME-INSTRUMENT per-direction medians MATCH across states
(pristine 1.03 vs walled 0.85/0.90/0.92) — the committed 2-3x gap
(the walled 1.58-1.71 vs the pristine 0.61) dissolves into
cross-instrument heterogeneity (the pristine row's committed band
was e_chart's mixed-lr-ladder span + a different fine grid) PLUS
the n=3 draw lottery (the walled family's own medians swing
0.91-1.71). T179's free find RETIRED, substantially an artifact.
THE FREE LESSON THAT SURVIVES: per-direction safety orders with SV
ENERGY (Spearman +0.38..+0.81; pooled +0.64) and ANTI-orders with
curvature (-0.60..-0.79) — IDENTICALLY at all four states: SAFETY
LIVES IN THE WASH'S OWN TOP DIRECTIONS; flatness is not the
protective axis. HONESTY: n=3 walled + 1 pristine; single
realization per primary span; the FD instrument a ranker; the
committed join's cross-instrument caveat.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T181 —"
card = """## T182 — e211: the shadow retired — and the real ordering found (2026-10-02 ~15:45Z)

The walled-band question closes with the lab policing its own free
find: same-instrument, the wall's noise ball is NOT wider — the
2-3x committed gap was cross-instrument heterogeneity plus the
draw lottery (THE INSTRUMENT SHADOW, now a named failure class
joining W021's family: instruments compared across conventions
manufacture properties). BOTH registered mechanisms dead cleanly
(no span widening; no basin flattening). THE SURVIVING LESSON IS
THE CELL'S BEST: per-direction safety orders with SV energy and
anti-orders with curvature at every state alike — SAFETY LIVES IN
THE WASH'S OWN TOP DIRECTIONS: an organism's displacement tolerance
is not isotropic within the span but concentrated where the wash
itself puts its energy (the directions the wash has already
visited are the directions further displacement forgives) — an
EXPOSURE-IMMUNITY reading: the wash vaccinates its own span. THE
BAND SCALAR'S honest state: retired as a wall property; the
per-direction ordering the direction-level answer; the same-
instrument pristine band the named debt if the scalar is ever
wanted.

"""
assert anchor in t and "## T182" not in t
t = t.replace(anchor, card + anchor, 1)

m = re.search(r"^## T179 — .*$(.*?)(?=^## T178)", t, re.M | re.S)
assert m, "t179"
body = m.group(1).rstrip("\n")
amend = """

[E211 AMENDMENT ~15:45Z]: the free find (the walled bands grow)
RETIRED — same-instrument the medians match; the gap was the
instrument's shadow + the draw lottery. See T182."""
t = t[:m.start(1)] + body + amend + "\n\n" + t[m.end(1):]
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e211 |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| e211 | THE WALLED-BAND QUESTION | DONE 15:45Z (T182: GRADED — both mechanisms dead; THE INSTRUMENT SHADOW retired (same-instrument medians match; the committed gap cross-instrument + lottery); THE SURVIVING LESSON: safety orders with SV energy (the wash's own top directions forgive — an EXPOSURE-IMMUNITY reading)) |", 1)
i = q.index("\n", q.index("| e211 |")) + 1
row2 = ("| e212 | THE SAME-INSTRUMENT PRISTINE BAND (e211's named debt: the random-draw band at the pristine root "
        "on the onset instrument — the band scalar's honest close) | DISPATCHED 15:46Z — CPU eval |\n")
q = q[:i] + row2 + q[i:]
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T15:46:00Z"
st["current_experiment"] = ("e211 FOLDED (T182: the instrument shadow retired; the exposure-immunity ordering found). "
                            "Fleet: e182c-2 (GPU, the template locus) + e212 DISPATCHED (CPU: the pristine band)."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e211 folded")
