# -*- coding: utf-8 -*-
"""Fold e222: NOTES, T199, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e222 — the exposure-immunity causal test: NULL — T182's reading is NOT intervenable at sub-lethal doses (the same-axis rows land ON the pass-back arithmetic to four decimals (ceiling 1.100, residue +0.0006); every cross row within 7% with no arm-specific immunity; the f2 span's own ordering nearly flat); the honest limits named (2026-10-02 ~20:15Z) — DONE

WHAT WE DID: the span machinery verbatim on the f2 root (873k,
51.5s CPU, all 9 gates PASS); the sub-lethal pre-exposure eps =
0.1 x t_top along +v0 / +v19 / a random in-span control; the
kill-Ds of dirs 0/9/19 re-walked from every exposed state; the
pass-back arithmetic disclosed BEFORE compute.

WHAT WE SAW (T199): NULL — the same-direction rows land ON the
mechanical pass-back arithmetic to four decimals (ratio 1.101 vs
ceiling 1.100, residue +0.0006; the low arm -0.0012; the control
-0.0001); every cross row within 7%, no arm-specific immunity
(the top exposure raises the mid/low rays no more than the control
does); the f2 span's own ordering NEARLY FLAT (kills
0.802/0.811/0.877 by SV rank — slightly reversed: e211's ordering
had little variance to act on at this organism). THE EXPOSURE-
IMMUNITY READING, made causal at one sub-lethal dose here, does
NOT fire: the top-SV tolerance is position-determined along its
own axis and untouched along the orthogonal span axes. HONEST
LIMITS (the honesty-gated next): n=1 organism, one span draw, one
dose, ONE SIGN (the +v side; the wash itself drifts -v0 — the
drift-side exposure, the mechanically sensitizing geometry,
remains untested); a direct displacement is the wash's mechanism
at one remove (eps 0.080 vs the wash's 0.916 step — immunity-to-
TRAINING needs a 1-3 AdamW-step exposure arm); the e131-family
root (where the ordering is strong, +0.38..+0.81) is the
replication that would give the NULL teeth.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T198 —"
card = """## T199 — e222: the vaccination does not fire at one dose (2026-10-02 ~20:15Z)

The exposure-immunity causal test returns the honest NULL: at one
sub-lethal dose on the flat-ordering organism, the tolerance is
position arithmetic, not protection — the same-axis rows track the
pass-back prediction to four decimals, and no cross-axis
immunity separates from the control. T182'S ORDERING STANDS AS A
CORRELATION with no causal handle at this dose/sign/organism; the
three honest limits (the drift-side sign; the training-step
exposure; the strong-ordering e131 root) are named — any of them
could revive or bury the reading, and none is owed today. THE
PROGRAM NOTE: the two mechanism questions of the beat closed
oppositely and well — g12's intervention DECISIVE (the normalizer
convicted), e222's NULL honest (the reading not intervenable where
tested) — both under pre-registered bars, both with their limits
named. THE CENSUS DISCIPLINE'S SHAPE: a decisive positive, an
honest null, and a closed hunt in one beat's work.

"""
assert anchor in t and "## T199" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
q = q.replace("| DISPATCHED 19:25Z — bars: IMMUNITY-CAUSAL / SENSITIZATION / NULL / GRADED |",
              "| DONE 20:15Z (T199: NULL — the tolerance position-determined at this dose/sign/organism; the same-axis rows on the pass-back arithmetic to 4 decimals; no cross-axis immunity vs control; T182's ordering stands as a correlation; the three honest limits named) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T20:16:00Z"
st["current_experiment"] = ("e222 FOLDED (T199: NULL - the exposure-immunity not intervenable at this dose; the "
                            "limits named). Fleet 0; the day's every arc closed."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e222 folded")
