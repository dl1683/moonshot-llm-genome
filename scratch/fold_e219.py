# -*- coding: utf-8 -*-
"""Fold e219: NOTES, T193, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e219 — the independent-entrenchment cell: GRADED — THE THIRD DIMENSION SURVIVED THE BREAK (not instrument texture (rho +0.234/+0.256 > the bar); not entrenchment (zero absorption: family+IE leaves the residual at 0.936 vs 0.934)); a weak positive relation real but unable to name the dimension; the hunt continues past entrenchment (2026-10-02 ~17:50Z) — DONE

WHAT WE DID: the confound break executed — 146 hand-registered
paraphrase forms (2-3 per probe; paraphrased cloze, reversed
cue-swaps, per-item constructions) measured at t=0 through a
channel independent of the wash reads (rho(IE, p0) +0.333); all 7
gates PASS (e216's residuals bit-reproduced at dp 0.0); 30s CPU.

WHAT WE SAW (T193): (1) THE CORRELATION: rho(IE, residual) =
+0.234/+0.256 — above the 0.2 texture bar (NOT instrument
texture; the confound genuinely broken), far below the 0.4 real
bar (NOT plain entrenchment); directionally consistent everywhere
(the reversed-direction forms carry more: +0.238/+0.282). (2) THE
ABSORPTION (the decisive split): family + IE substituted as the
height term leaves the cross-wash residual at 0.936 vs the 0.934
baseline — ZERO absorption; the named splits survive substitution
verbatim (product +3.2/+2.5 SD; tmpl width 0.82/0.80). (3) THE
TEXTURE: a weak positive relation is real (Topeka->Kansas at IE
0.96 anchors high; iPhone at 0.48 low) but cannot name the
dimension — Gmail, the largest positive residual, is only
mid-channel. THE HUNT CONTINUES PAST ENTRENCHMENT: the remaining
named candidates are the probe's internal token structure and the
relation's compositionality. HONESTY: the hand-registered
paraphrases (committed before measurement); n=2 washes; the 8
multi-token drops documented.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T192 —"
card = """## T193 — e219: the dimension survives its first confound break (2026-10-02 ~17:50Z)

The third dimension's first identity test lands in the honest
middle: the paraphrase channel (146 forms, independent of the wash
reads at rho 0.33) correlates with the residual above the texture
bar — the replicating miss is NOT instrument texture — but absorbs
nothing (0.936 vs 0.934): NOT entrenchment through a second
channel either. A WEAK POSITIVE slope is real (entrenched probes
sit high) but Gmail — the archive's biggest residual — is
mid-channel: whatever holds Gmail is not entrenchment. THE HUNT'S
STATE: the dimension is real (survived the break), unnamed
(entrenchment retired), with two candidates left (the internal
token structure; the relation's compositionality). THE PROGRAM
NOTE: the census discipline's pattern holds — each identity test
retires one candidate cleanly and leaves the object sharper.

"""
assert anchor in t and "## T193" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e219 |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| e219 | THE INDEPENDENT-ENTRENCHMENT CELL | DONE 17:50Z (T193: GRADED — the dimension SURVIVED the break (not texture: +0.234/+0.256 > bar; not entrenchment: zero absorption 0.936 vs 0.934)); a weak positive real but Gmail mid-channel; the hunt continues: token structure / compositionality) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T17:51:00Z"
st["current_experiment"] = ("e219 FOLDED (T193: the dimension survived its first confound break - real, unnamed). "
                            "Fleet: g1f (GPU, the second cons seed) alone."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e219 folded")
