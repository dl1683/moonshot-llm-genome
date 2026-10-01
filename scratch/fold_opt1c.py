# -*- coding: utf-8 -*-
"""Fold opt1c: NOTES, T143, W025 amendment, QUEUE (DONE + e191), STATE."""
import io, json, re

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## opt1c — the direction-size factorial: KILL-OUT-OF-WINDOW (no bar fires) — the raw-gradient direction at Adam's size kills BELOW the gate; the pump is local; small steps spare by RE-ORIENTATION (2026-09-30 ~07:55Z true-clock) — DONE

WHAT WE DID: one arm — delta = 1.6543 * (g/||g||), the raw
batch-gradient direction at Adam's measured per-step L2, on the
licensed e185 cell (t=0 bit-gate vs opt1's A0: diffs 0.0; step-1
read + chunk md5 bit-identical to the smoke — free n=2
cross-process determinism; recovery agent: verified the survivor
script line-by-line, fixed a plot-layer key bug, added progressive
PARTIAL writes + resume hardening; 18.1s CPU).

WHAT WE SAW (T143): the fact died INSIDE step 1 — densified
interpolated D_kill 0.920 (bracket [0.662, 0.993]) vs the registered
window [2.12, 3.27]: NO bar fires (graded outcome, curve verbatim;
the pre-registered map's two named branches both miss — the third
reading is the finding). CUMULATIVE DISPLACEMENT IS NOT
DIRECTION-ROBUST: kill-D moved ~2.7x DOWN vs Adam's 2.489. AT
MATCHED D 1.6543: Adam's direction left g-12 0.678 (CE_R 2.21);
this arm left 0.0007 (CE_R 4.79 — the organism devastated too); the
bleed held 0.79. THE ALONG-PATH GEM: the fact PUMPS to 0.955 at
D 0.33 — numerically the bleed's pump at D 0.32 — then cliffs
(f=0.6: 0.135; f=0.8: 0.005): THE PUMP IS LOCAL (a property of the
displacement region), and what spares the bleed is small-step
RE-ORIENTATION, not the gradient direction. e188's RAW-WINS becomes
WITHIN-CLASS invariance (the first cross-direction measurement at
fixed size breaks it). Alignment -0.031 at the kill (within opt1's
band). HONESTY: n=1, CPU fp32, recovery provenance disclosed.
WHAT'S NEXT (dispatched): opt1b2 (the bleed's own crossing — it
already passed 0.92 alive at 0.79 territory; where does IT die?) and
e191 (the pump-cliff map: STATIC single jumps along the g-direction
at graded D — geometry vs dynamics).
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T142 —"
card = """## T143 — opt1c: the third outcome — the raw gradient is the most lethal direction, and the pump-cliff is the terrain (2026-09-30 ~07:55Z)

The factorial's answer was a branch neither name covered (the map
still served: it forced the graded-outcome reading, no bar
shopping). KILL-OUT-OF-WINDOW: the raw-gradient direction at Adam's
step size kills at D ~ 0.920 — 2.7x BELOW the Adam gate, inside the
first step. THE THREE CLASSES AT MATCHED D 1.6543: the guillotine
(Adam's sign direction: 0.678, organism shocked), the ANNIHILATION
(raw gradient at full size: 0.0007, organism devastated), the bleed
(small steps: 0.79, organism intact). LETHALITY PER DISPLACEMENT
ORDERS: raw-gradient > sign-normalized > random (4-10x) — W025
REFINES: not merely "in the subspace" — the direction's projected
effectiveness on the fact's sensitive structure orders the lethality;
the raw gradient IS the steepest effective direction, sign(g) its
flattened shadow (W024's stitches again — flattening LOSES some
lethality, it does not add it). THE PUMP-CLIFF GEOMETRY: the fact
pumps to 0.955 at D ~ 0.33 then cliffs by D 1.0 — a local ridge then
a cliff in the g-direction; the pump is a REGION property (the bleed
pumped at the same D on tiny steps), and the bleed's protection is
RE-ORIENTATION: each tiny step recomputes the gradient and the path
curves around the cliff the big step overshoots. e188's RAW-WINS
RESTATES as within-class invariance across lr along the Adam class.
THE OPEN SPLITS (both dispatched): geometry-vs-dynamics (e191:
static single g-jumps at graded D — if the static profile matches
the dynamic cliff, the terrain is real; if the static jump at 0.92
spares, the kill is overshoot) and the bleed's own crossing
(opt1b2: it passed 0.92 alive at 0.79 — where does the re-orienting
path die? the projection said ~10; the cliff says closer).

"""
assert anchor in t and "## T143" not in t
t = t.replace(anchor, card + anchor, 1)

# W025 refinement note
m = re.search(r"^## W025 — .*$(.*?)(?=^## W024)", t, re.M | re.S)
assert m, "w025"
body = m.group(1).rstrip("\n")
amend = """

[OPT1C REFINEMENT ~07:55Z: the subspace picture gains an ordering —
lethality per displacement: raw-gradient (D~0.9) > sign-normalized
(D~2.5) > random (4-10x). Not all in-subspace directions are equal:
the raw gradient is the steepest effective direction; sign(g) is its
magnitude-flattened shadow (W024 inverted at the top end — flattening
LOSES lethality relative to g, even as it beats random). The
pump-cliff at D 0.33-1.0 is the first MAPPED terrain inside the
subspace.]"""
t = t[:m.start(1)] + body + amend + "\n\n" + t[m.end(1):]
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| opt1c |"):]; row = row[:row.index("\n")]
new_row = ("| opt1c | THE DIRECTION-SIZE FACTORIAL | DONE 07:55Z (T143: KILL-OUT-OF-WINDOW — the raw-gradient "
           "direction at Adam's size kills at D 0.920, 2.7x BELOW the gate, inside step 1; no bar fires, the graded "
           "outcome is the finding; at matched D 1.65: Adam 0.678 / this arm 0.0007 (CE 4.79) / bleed 0.79; the "
           "PUMP is local (0.955 @ D 0.33) then a cliff; the bleed spares by RE-ORIENTATION; lethality orders "
           "g > sign(g) > random; RAW-WINS restates as within-class) |")
q = q.replace(row, new_row, 1)
i = q.index("\n", q.index("| opt1b2 |")) + 1
e191 = ("| e191 | THE PUMP-CLIFF MAP (opt1c's discriminator: STATIC single jumps along the g-direction at graded "
        "D {0.1..1.5} — geometry (the terrain is real) vs dynamics (the kill is overshoot); eval-only CPU) | "
        "DISPATCHED 07:56Z — bars at dispatch: STATIC-CLIFF (static profile matches the dynamic cliff: kills "
        "0.9-1.0) / STATIC-SPARES (static jump at 0.92 alive -> the kill is overshoot through the cliff) / graded |\n")
q = q[:i] + e191 + q[i:]
q = q.replace("| QUEUED — CPU after opt1c; inherits opt1b's checkpoint + trajectory |",
              "| DISPATCHED 07:56Z (with e191; both CPU-light beside g1bS's GPU wait; inherits opt1b's checkpoint + trajectory) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

s = json.load(io.open("STATE.json", encoding="utf-8"))
s["last_heartbeat"] = "2026-09-30T07:56:00Z"
s["current_experiment"] = ("opt1c FOLDED (T143: KILL-OUT-OF-WINDOW — the raw gradient is the most lethal direction "
                           "(D 0.92); the pump-cliff is the terrain; the bleed spares by re-orientation; lethality "
                           "orders g > sign(g) > random). Fleet: g1bS (GPU, paused under external job) + opt1b2 + e191 "
                           "(both CPU-light, dispatched).")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(s, indent=2, ensure_ascii=False))
print("opt1c folded")
