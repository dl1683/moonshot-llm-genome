# -*- coding: utf-8 -*-
"""Fold e191: NOTES, T144, paper figure line, QUEUE, clock note, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e191 — the pump-cliff map: STATIC-CLIFF (TERRAIN) — the cliff is in the geometry; no overshoot needed; the bleed survives by re-orientation (2026-10-01 ~08:05Z) — DONE

WHAT WE DID: 12 graded STATIC single jumps theta_0 - D*u (u = the
root wash-batch gradient, opt1c's t=0 convention, recomputed: t=0
gate diffs 0.0; direction gate 1-cos 8.3e-14 in fp64, the fp32 dot
artifact documented; 5 committed points reproduced to 1.6e-06);
perturb-and-eval, CPU eval-only 35.3s; progressive PARTIAL writes.

WHAT WE SAW (T144): the static profile MATCHES the dynamic cliff —
pump 0.9555 at D 0.33 (grid peak 0.9605 at 0.20), 0.910 at 0.50,
0.755 at 0.66, 0.492 at 0.80, DEAD 0.2482 at D 0.92, floor 0.0004
at D 2.0; cliff-edge bracket tightened from [0.66, 0.99] to
[0.80, 0.92]. STATIC-SPARES does not fire at its own D — no
trajectory effect is required to explain the kill: THE PUMP-CLIFF
IS TERRAIN in the g-direction. THE BLEED OVERLAY is the punchline:
opt1b's re-orienting tiny-step path holds ~0.83 ALIVE at the same D
where the straight g-ray is dead — sparing is re-orientation OFF
the ray, not gentler displacement. CE_R co-read: the fact cliffs
first (dead at CE_R 3.0 vs root 1.66), the organism wrecks
progressively after (5.28 at D 2.0). HONESTY: within [0, 1.6543]
opt1c's dynamic path WAS the straight ray (single-step kill), so
static = dynamic by construction there — DISCLOSED before compute;
the independent content is the fresh recompute, the D 2.0 extension,
the tighter edge, and the CE_R read; n=1 direction-deterministic,
one organism. WHAT'S NEXT: opt1b2 owns the surviving split (where
does the re-orienting path die?); a second-organism replicate is
the honest replicate axis.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T143 —"
card = """## T144 — e191: the cliff is terrain — the first mapped ground of the forgetting machine (2026-10-01 ~08:05Z)

STATIC-CLIFF fires: the graded static profile along the raw-gradient
ray matches the dynamic kill — pump ridge (0.94-0.96 across D
0.05-0.50, peak 0.9605 at 0.20), cliff edge [0.80, 0.92], dead at
0.92, floor at 2.0. NO OVERSHOOT IS NEEDED: the kill is geometry.
THE BLEED OVERLAY IS THE FIGURE: at D 0.92 the straight ray reads
0.248-dead while the re-orienting path reads ~0.83-alive —
protection = re-orientation off the ray (W026's managed-bleed
mechanism now has its figure). THE LETHALITY ORDERING STANDS ON
MAPPED GROUND: the raw-gradient ray's terrain kills at 0.92; the
sign-normalized ray's at ~2.5 (opt1); random rays' at 4-10x (g3K) —
three terrains of increasing width. CE_R: the fact dies at organism
CE 3.0 — the fact is the canary, not the casualty of general
wreck (the organism wrecks further out, 4.8-5.3). DISCLOSED: on the
single-step interval static = dynamic by construction; the
independent content is the recompute, the extension, the tighter
edge, the CE_R profile. THE SURVIVING SPLIT: opt1b2 (running) —
where does the re-orienting path itself die? If it survives past
every static kill ring, the final law's protective principle is
re-orientation alone and the walls/cages are one family of many.

"""
assert anchor in t and "## T144" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

# paper figure line
p = "scratch/day6_paper_skeleton.md"
s = io.open(p, encoding="utf-8").read()
old = "Fig 4 (THE generative plate, 3 panels):"
new = """Fig 5 (THE terrain figure, from e191/opt1c/opt1b/opt1b2): the
   fact-vs-displacement overlay — the static g-ray profile (pump
   ridge, cliff [0.80,0.92], floor), the dynamic single-step kill,
   Adam's ray (kill ~2.5), the random-ray band (4-10x), and the
   re-orienting bleed alive across all of it: forgetting's terrain
   and the one path that dances on it.
Fig 4 (THE generative plate, 3 panels):"""
assert old in s
s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8").write(s)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e191 |"):]; row = row[:row.index("\n")]
new_row = ("| e191 | THE PUMP-CLIFF MAP | DONE 08:05Z Oct-1 (T144: STATIC-CLIFF (TERRAIN) — the static g-ray "
           "profile matches the dynamic cliff (pump ridge 0.94-0.96 across D 0.05-0.50; edge [0.80, 0.92]; dead at "
           "0.92; floor 2.0); no overshoot needed; the bleed overlay: alive ~0.83 where the ray is dead — sparing "
           "= re-orientation; CE_R: the fact is the canary) |")
q = q.replace(row, new_row, 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-01T08:07:00Z"
st["clock_note"] = ("SECOND CLOCK JUMP: machine moved from 2026-09-30T07:56Z to 2026-10-01T08:06Z (~+24h). "
                    "True date is 2026-10-01 (env-confirmed). All stamps are best-effort labels under an unstable "
                    "clock; ordering + commit hashes are the reliable record.")
st["current_experiment"] = ("e191 FOLDED (T144: STATIC-CLIFF — the pump-cliff is terrain; the bleed survives by "
                            "re-orientation; the terrain figure named for the paper). Fleet: g1bS (GPU, PARTIAL "
                            "metrics) + opt1b2 (CPU, metrics landing). CPU queue: e189 -> e190 -> opt2.")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e191 folded")
