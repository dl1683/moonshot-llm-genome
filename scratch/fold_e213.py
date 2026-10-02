# -*- coding: utf-8 -*-
"""Fold e213: NOTES, T185, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e213 — the path-independence census: PATH-PARTIAL (+GRADED, tables verbatim) — 6/11 non-floor cells match within ±10% (all four at +50; fact+ctrl at +80); 5 wander; the free find re-derived at dp 0.0 (ratio 0.9978); the correlate is DEPTH (2026-10-02 ~17:10Z) — DONE

WHAT WE DID: every battery's decline compared across the two saved
washes (the fact, the phase-1 controls, the near-related, the
reversed template) at every shared state; all gates PASS (the
states re-probed, max dp 3.3e-06 over 24 checks); eval-only CPU.

WHAT WE SAW (T185): PATH-PARTIAL — 6/11 non-floor cells match
within +-10%: ALL FOUR batteries at +50 (the free find re-derived
at dp 0.0: tmpl@+50 0.421187 vs 0.420248, ratio 0.9978 — the
original match is real, not luck) and fact+ctrl at +80. THE 5
WANDERERS: the +10 cells (uniform >1.1 — WASH 2 ERODES EARLIER:
the shallow-decline denominators diverge) and near (0.857) + tmpl
(0.804) at +80 — THE TWO DEEPEST wash-1 declines read SHALLOWER on
wash 2 at depth (Spearman -0.60: the deeper the wash-1 decline,
the more wash 2 spares). THE CORRELATE IS DEPTH: not battery size
(n=3 and n=19 both wander), not base rate (the lowest and highest
R0 both wander). THE READING: the MID-regime (+50) is
path-independent (a state function — the displacement-gate echo
holds there); the EARLY regime is path-typed (wash 2's stream
erodes earlier); the DEEP regime partially mean-reverts (the
deepest declines spare on the second wash). HONESTY: n=2 washes;
the pools' n's; the CPU fp32 texture; the free find stands as
real but not general.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T184 —"
card = """## T185 — e213: the state function holds at mid-depth only (2026-10-02 ~17:10Z)

The census maps the path-independence honestly: the +50 regime is
a STATE FUNCTION across every battery (the original three-decimal
match re-derived at dp 0.0 — real, not luck); the early regime is
path-typed (wash 2's stream erodes earlier — the shallows are the
stream's own texture); the deep regime partially mean-reverts (the
deepest wash-1 declines read shallower on wash 2 — a sparing
correlate, Spearman -0.60). THE 124M PICTURE IN ONE BREATH: the
erosion is a displacement function at mid-depth, a stream function
early, and partially self-correcting late. THE ECHO RESCOPED:
e188's displacement gate (tiny-scale) and the 124M mid-depth state
function agree where they overlap — the STATE-FUNCTION claim is
now two-scale, depth-bounded. The program's law again: the SHAPE
(mid-depth state-functionality) is the physics; the depth
boundaries are the biography.

"""
assert anchor in t and "## T185" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e213 |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| e213 | THE PATH-INDEPENDENCE CENSUS | DONE 17:10Z (T185: PATH-PARTIAL — 6/11 match (ALL FOUR at +50, the free find re-derived at dp 0.0); 5 wander (the +10 shallows path-typed; the deep declines partially spared); THE CORRELATE IS DEPTH; the state function holds at mid-depth only — two-scale with e188 where they overlap) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T17:10:00Z"
st["current_experiment"] = ("e213 FOLDED (T185: the state function holds at mid-depth only - the erosion a "
                            "displacement function at +50, a stream function early, partially self-correcting late). "
                            "Fleet: g1d (GPU, the base redraw) alone."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e213 folded")
