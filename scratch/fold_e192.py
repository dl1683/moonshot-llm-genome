# -*- coding: utf-8 -*-
"""Fold e192: NOTES, T146, T144 stamp, W026 upgrade, paper, QUEUE, STATE."""
import io, json, re

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e192 — the all-ray terrain map: TERRAIN-ONE-PICTURE + RIDER-REORIENTATION-CAUSAL — Fig-5 licensed as ONE picture; re-orientation is causal; the pump tracks gradient STRUCTURE (2026-10-01 ~13:50Z) — DONE

WHAT WE DID: static graded jumps along five ray families on ONE
organism (the e131 consolidated root, 2.74M), ONE ruler (the
install-60 g-12 battery + CE_R), DUAL currency (absolute L2 +
per-coordinate RMS), D grid to 4.0; plus the intervention rider
(300 pinned 0.0055-L2 steps down the FROZEN g-ray — orientation
denied, step size at the bleed's scale). All 10 gates PASS (e191
protocol verbatim; G_T0 bit-exact; the loaded direction
md5-identical + fresh recompute identical; the fresh g-ray reads
bit-exact over 12 shared Ds; A0's endpoint anchor exact; the
rider's step-300 anchor 4.4e-8). Second-dispatch recovery
(predecessor killed pre-artifact); progressive PARTIAL writes
throughout; 194.7s CPU eval-only.

WHAT WE SAW (T146): KILLS — g-ray 0.92 (5.6e-4 RMS) < static
sign(g0) 2.5 (1.5e-3 RMS; 2.72x; THE FIRST static sign map on this
lineage — and A0's committed step-1 read 0.678 sits ON the static
curve, |d| 5.5e-4: the path-length stitch objection dies) < three
Gaussian rays ALL ALIVE at D 4.0 (>4.35x; the fact flat
0.896-0.905 — untouched by isotropic displacement at 4x the g-ray
kill). THE ORDER g < sign < random NOW STANDS ON MAPPED GROUND:
Fig-5 licensed as one picture. THE PUMP IS GRADIENT-SPECIFIC — and
the sign ray pumps too (+0.037, peak 0.953 at 0.5): the pump tracks
gradient STRUCTURE (g and sign(g) both pump; isotropic does not
move the fact at all). THE RIDER: the pinned walk dies inside
[0.827, 0.993] (the static edge bracket) while the re-orienting
bleed lives 0.79-0.86 at the same D — STEP SIZE DENIED; ORIENTATION
OWNS THE SPARING: W026's "protection = re-orientation" is now
INTERVENTIONAL. HONESTY: n=1 organism/fact/root; the random-kill
band is UNRESOLVED-HIGH (>4.0 at the grid cap — a number would
need a wider grid, 8-12); the sign ray is one static direction,
not Adam's adaptive path.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T145 —"
card = """## T146 — e192: the terrain licensed as one picture; re-orientation causal; the pump tracks structure (2026-10-01 ~13:50Z)

Both primary bars fired. THE MAP: on one organism, one ruler, dual
currency — g-ray 0.92 < static sign(g0) 2.5 < Gaussian >4.0 (alive,
flat): the three-band figure stands on mapped ground, and the
middle band's stitch objection DIED by measurement (A0's step-1
read sits ON the static sign curve). THE ISOTROPIC ARM'S FLATNESS
is its own finding: 4x the g-ray's kill displacement leaves the
fact untouched — W025's projection ratio at this organism >= 4.35x,
and the e190 subspace test now has a sharp target. THE RIDER IS THE
DAY'S CAUSAL ANCHOR: pinned small steps (orientation denied, size
at the bleed's scale) die at the static edge while the bleed lives
at the same D — RE-ORIENTATION OWNS THE SPARING; step size owns
nothing. W026's managed-bleed noun graduates from poetry to
interventional mechanism (the R59 critic's one converting cell,
paid). THE PUMP TRACKS GRADIENT STRUCTURE: g pumps (+0.045), sign(g)
pumps (+0.037), isotropic is inert (+0.0002) — the pump is not
displacement, not magnitude, but STRUCTURE; W024's census question
sharpens to "what do g and sign(g) share that isotropic lacks" (the
sign pattern itself?). CAVEATS carried: n=1; random band
unresolved-high (the wider grid is a rider for the chart cell, not
a new dispatch); the sign ray is static, not Adam's adaptive path
(the path-vs-ray gap for the SIGN class remains e191-style
disclosed).

"""
assert anchor in t and "## T146" not in t
t = t.replace(anchor, card + anchor, 1)

# T144 stamp update: Fig-5 licensed
old = "e192 (dispatched) is the license.]"
new = "e192 (DONE 13:50Z): LICENSED — the one-organism map: 0.92 < 2.5 < >4.0, A0's step-1 read ON the static sign curve; re-orientation causal via the rider.]"
assert old in t
t = t.replace(old, new, 1)

# W026 upgrade note
m = re.search(r"^## W026 — .*$(.*?)(?=^## W025)", t, re.M | re.S)
assert m
body = m.group(1).rstrip("\n")
amend = """

[E192 UPGRADE ~13:50Z: the managed-bleed noun GRADUATES — the
pinned-ray rider killed step size as the sparing mechanism and
established re-orientation CAUSALLY (pinned steps die at the static
edge; the re-orienting bleed lives at the same D). The
rhythm-as-managed-bleed sentence remains unmeasured on g2's
organisms (the replay-event displacement read is still owed), but
the MECHANISM it invokes is now interventional on e131. The
wall-as-cage noun remains poetry until g1b's lineage gets its
terrain map.]"""
t = t[:m.start(1)] + body + amend + "\n\n" + t[m.end(1):]
io.open("THINKING.md", "w", encoding="utf-8").write(t)

# paper: Fig-5 licensed + abstract bracket fill
p = "scratch/day6_paper_skeleton.md"
s = io.open(p, encoding="utf-8").read()
old = """Fig 5 (THE terrain figure, from e191/opt1c/opt1b/opt1b2): the
   fact-vs-displacement overlay — the static g-ray profile (pump
   ridge, cliff [0.80,0.92], floor), the dynamic single-step kill,
   Adam's ray (kill ~2.5), the random-ray band (4-10x), and the
   re-orienting bleed alive across all of it: forgetting's terrain
   and the one path that dances on it."""
new = """Fig 5 (THE terrain figure — LICENSED by e192 as one picture): the
   fact-vs-displacement overlay on ONE organism/ruler/dual-currency
   — the g-ray profile (pump ridge, cliff 0.92), the STATIC sign
   ray (kill 2.5; A0's step-1 read on the curve), three Gaussian
   rays (alive/flat at 4.0), and the two walks at matched D (the
   pinned ray dead at the static edge, the re-orienting bleed
   alive): forgetting's terrain, and the one path that dances on
   it. Runs from runs/e192/e192_fig5_terrain.png."""
assert old in s
s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8").write(s)

# QUEUE: e192 DONE
q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e192 |"):]; row = row[:row.index("\n")]
new_row = ("| e192 | THE ONE-ORGANISM ALL-RAY TERRAIN MAP | DONE 13:50Z (T146: TERRAIN-ONE-PICTURE + "
           "RIDER-REORIENTATION-CAUSAL — 0.92 < 2.5 < >4.0 on one organism/ruler/dual-currency; A0's step-1 read ON "
           "the static sign curve; isotropic flat at 4x; THE PUMP TRACKS GRADIENT STRUCTURE (g and sign(g) pump, "
           "isotropic inert); the pinned-ray rider kills step size, establishes re-orientation causally; Fig-5 "
           "LICENSED; random band unresolved-high — wider grid owed as a chart-cell rider) |")
q = q.replace(row, new_row, 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-01T13:51:00Z"
st["current_experiment"] = ("e192 FOLDED (T146: Fig-5 LICENSED as one picture; RE-ORIENTATION CAUSAL (the rider); "
                            "the pump tracks gradient structure). Fleet: g1bS-r3 (GPU) + opt1b3-r2 (CPU) + e182c-r2 "
                            "DISPATCHING (the freed slot). Abstract bracket e192 filled.")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e192 folded")
