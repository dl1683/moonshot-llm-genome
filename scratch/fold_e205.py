# -*- coding: utf-8 -*-
"""Fold e205: NOTES, T173, T163 amendment, QUEUE, STATE."""
import io, re

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e205 — the onset normalization: ARRIVALS-CONTINGENT — the WHEN was an artifact of self-normalized ratios; onset rescopes to within-organism shape; the edge-multiple emerges as the per-organism scalar (2026-10-02 ~11:15Z) — DONE

WHAT WE DID: each front's kill-D re-expressed as a multiple of its
own organism's in-span random band (org1's and MIRABEL's loaded
committed; the half lineage's band FRESHLY computed — 3 draws,
median 0.8526, G_GRAY-anchored to 5.6e-08); the 2x3 arrival
matrix (self/common axes x bars 0.60/0.70/0.80). THE VERDICT WAS
DESK-FORCED TO CONTINGENT-OR-WORSE AT REGISTRATION (the committed
arithmetic alone sufficed; the fresh compute owned only the open
part; no bar shopping possible).

WHAT WE SAW (T173): ARRIVALS-CONTINGENT — on the common axis the
cross-organism arrivals DO NOT SURVIVE: org1 t1 dissolves at bar
0.60 (0.635x); MIRABEL dissolves at every bar (0.904x); at bar
0.60 the EARLIEST IDENTITY FLIPS org1 -> half. Edge multiples:
org1 3.72x / MIRABEL 2.12x / half 0.616x — THE HALF EDGE SITS
BELOW BAND PARITY (a lineage whose consolidation edge is WEAKER
than its own wash noise), a fact the self axis hid inside
ratio_0=1.0; the 4.3x edge spread the critic flagged was RANK
INFORMATION, not noise. ONE INVERSION: the half t2 (0.408x) is
bar-robust on the common axis where its self-axis arrival was the
contingent one. THE ONSET STORY RESCOPES TO WITHIN-ORGANISM SHAPE
ONLY (the cross-organism WHEN falls). THE NEW SCALAR: the
edge-multiple (the consolidation edge vs the organism's own wash
noise band) — the per-organism quantity the story was
accidentally normalizing away. HONESTY: bands n=3 (org1's one
SOFT-censored); the half fronts' counterfactual currency vs the
root-level bands; MIRABEL's multiples lower bounds a fortiori.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T172 —"
card = """## T173 — e205: the WHEN falls — and the edge-multiple rises (2026-10-02 ~11:15Z)

The normalization cell kills the cross-organism WHEN cleanly: on
a common ruler (each front vs its own organism's wash-noise band)
the arrivals dissolve or flip — org1's celebrated t1 arrival is a
0.635x band-multiple (below parity); MIRABEL's 0.904x dissolves
everywhere; the earliest identity flips at bar 0.60. THE ONSET
STORY'S SURVIVING LAYER: within-organism shape (each organism's
own concentration curve — intact). THE RISE: THE EDGE-MULTIPLE —
the consolidation edge as a multiple of the organism's own wash
noise (3.72 / 2.12 / 0.616) — the per-organism scalar the
self-normalized ratios were accidentally erasing: org1's fact
lives 3.7x above its wash noise; the half lineage's edge is BELOW
its own noise yet still resolves arrivals — the edge-multiple may
be the MEMORY-VERSUS-NOISE MARGIN, a genuinely new quantity (the
fact's signal-to-noise in its own environment). THE CHAPTER'S
PATTERN REPEATS: normalize honestly, and a story dies while a
scalar is born (the flight arc's order-vs-distances; now the
WHEN-vs-the-margin).

"""
assert anchor in t and "## T173" not in t
t = t.replace(anchor, card + anchor, 1)

m = re.search(r"^## T163 — .*$(.*?)(?=^## T162)", t, re.M | re.S)
assert m, "t163"
body = m.group(1).rstrip("\n")
amend = """

[E205 AMENDMENT ~11:15Z]: the cross-organism WHEN FALLS under the
common ruler (e205/T173: the arrivals dissolve or flip on the
in-span-band axis; the ordering was an artifact of self-normalized
ratios) — this card's claim rescopes to WITHIN-ORGANISM arrival
shape; the surviving cross-organism object is the EDGE-MULTIPLE
(the fact's margin over its own wash noise)."""
t = t[:m.start(1)] + body + amend + "\n\n" + t[m.end(1):]
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e205 |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| e205 | THE ONSET NORMALIZATION | DONE 11:15Z (T173: ARRIVALS-CONTINGENT — the WHEN falls on the common ruler (org1 t1 = 0.635x, MIRABEL 0.904x, the earliest flips at bar 0.60); onset rescopes to within-organism shape; THE EDGE-MULTIPLE RISES (3.72/2.12/0.616 — the fact's margin over its own wash noise, the new per-organism scalar); desk-forced at registration) |", 1)
i = q.index("\n", q.index("| e205 |")) + 1
row2 = ("| e206 | THE DRIFT-RATE CLOCK (W027's named cut: does the support's drift rate predict death time? the "
        "fact's own timer from gradients alone) | DISPATCHED 11:16Z |\n")
q = q[:i] + row2 + q[i:]
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

import json
st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T11:16:00Z"
st["current_experiment"] = ("e205 FOLDED (T173: the WHEN falls; the edge-multiple rises). Fleet: g10 + g1bS8 (GPU) "
                            "+ e206 DISPATCHED (CPU: the drift-rate clock)."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e205 folded")
