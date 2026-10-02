# -*- coding: utf-8 -*-
"""Fold g1bS7: NOTES, T172, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## g1bS7 — the redrawn interior dose: PEAK-LOTTERY — the peak's HEIGHT is a draw lottery (0.9351 vs 0.7677, |d| 0.167); the SHAPE survives (the redraw still clears SHARP and outranks 0.25); the lottery broke UPWARD — the FIRST 10M root over the express bar (2026-10-02 ~10:55Z) — DONE

WHAT WE DID: one fresh-jitter consolidation at the 0.20 rms peak
(seed 10903 vs 10901; everything else verbatim); the
genuine-redraw gates PASS (s25 divergence 4.6x the same-seed
fuzz; ~100% of elements differing; the seed axis ~6x the device
fuzz — two currencies stamped); the envelope audited every poll.

WHAT WE SAW (T172): PEAK-LOTTERY — the fresh draw landed at root
g-12 0.9351 vs the original's 0.7677 (|d| 0.1674 > 0.10, the
frozen bar): THE PEAK'S HEIGHT IS A DRAW LOTTERY (the g2e lesson
at the consolidation level — the lab's second lottery). THE CURVE
IS ONE BIOGRAPHY (g1bS5's five readings are single-seed draws;
the seed axis dominates the fine structure). WHAT SURVIVES
(texture, registered non-bars): the interior-optimum SHAPE (the
redraw still clears SHARP's 0.6998 threshold and still outranks
the 0.25 reading); AND THE LOTTERY BROKE UPWARD — 0.9351 IS THE
FIRST 10M ROOT OVER THE 0.78 EXPRESS BAR (held30 0.6395, CE_R
1.678; saved at runs/checkpoints/g1bS7_root_m020f.pt — the
strongest root on record). g1bS6's WALL-FADES ran on one draw
(0.7677) of a 0.77-0.94-class distribution: the verdict's
DIRECTION untouched; its root's identity a lottery ticket. THE
FORMATION-DOSE CHAPTER'S RESIDUAL: a peak DISTRIBUTION (not a
peak value), and a 0.94-class root available for the sixth take.
HONESTY: n=1 redraw (two draws total at the peak).
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T171 —"
card = """## T172 — g1bS7: the second lottery — and the 0.94 root (2026-10-02 ~10:55Z)

The redraw answers the honesty ledger with the lab's second
lottery: the formation peak's height is a draw lottery (0.77 vs
0.94 at the same dose, same base, one seed apart) — joining the
root-strength lottery (g2e/T132) as the program's recurring
texture: THE RECIPE LEVELS ARE LOTTERIES; THE SHAPES ARE THE
PHYSICS. What survives the redraw is exactly the shape layer (the
interior optimum; the ordering; the class) — the same split as
the flight arc's (order=physics, distances=biography): EVERY LAYER
OF THIS PROGRAM SEPARATES INTO SHAPE (robust) AND HEIGHT
(lottery). THE UPWARD BREAK IS THE PRACTICAL GIFT: the first 10M
root over the express bar (0.9351, saved) — the sixth take's
substrate: the wall arms on a root that actually clears, the
first 10M adjudication with no deviation needed.

"""
assert anchor in t and "## T172" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| g1bS7 |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| g1bS7 | THE REDRAWN INTERIOR DOSE | DONE 10:55Z (T172: PEAK-LOTTERY — 0.9351 vs 0.7677, the height a draw lottery; the SHAPE survives (SHARP still cleared; 0.25 still outranked); the upward break: the FIRST 10M root over the express bar, saved — the 0.94-class root) |", 1)
i = q.index("\n", q.index("| g10 |")) + 1
row2 = ("| g1bS8 | THE SIXTH TAKE (the wall arms on the 0.94-class root g1bS7_root_m020f — the first 10M "
        "adjudication with NO deviation) | DISPATCHING 10:56Z |\n")
q = q[:i] + row2 + q[i:]
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T10:56:00Z"
st["current_experiment"] = ("g1bS7 FOLDED (T172: PEAK-LOTTERY - the second lottery; the 0.94 root saved). Fleet: "
                            "g10 + e205 + g1bS8 DISPATCHING (the sixth take on the 0.94 root)."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("g1bS7 folded")
