# -*- coding: utf-8 -*-
"""Fold e212: NOTES, T184, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e212 — the same-instrument pristine band: GRADED — the shadow's near-last word: the 2-3x gap DEAD at the same instrument (bounded <= 16%, inside the SE window); the band scalar stays closed as a NON-PROPERTY (2026-10-02 ~16:40Z) — DONE

WHAT WE DID: the pristine e131 root's random-draw band on e209's
onset-grid instrument VERBATIM (3+2 draws; the instrument identity
BIT-verified — the spans reproduce e211's committed pristine spans
at rel 0.0; the stream md5-gated); 124s CPU.

WHAT WE SAW (T184): THE PRISTINE MEDIAN 0.756 [0.435-2.128, 0
censored] vs the walled family 0.846/0.896/0.920 -> RATIO 0.844x —
neither inside the registered match window (0.809-0.957) nor
materially below (the 0.7x bar): GRADED, the tables verbatim. THE
SHADOW'S NEAR-LAST WORD: the committed 2-3x "walled bands grow"
gap (built on e_chart's cross-instrument 0.610 pristine row) is
now BOUNDED AT <= 16% SAME-INSTRUMENT, inside the generous
within-root SE window [0.533, 1.233] — THE WALL'S BAND IS NOT
WIDER IN ANY MATERIAL SENSE; the residual 16% sits at the edge of
the draw lottery's reach and of the registered window. THE BAND
SCALAR'S FINAL STATE: CLOSED AS A NON-PROPERTY (with the honest
caveats: 1 pristine root vs 3 walled; the direction-family
asymmetry — random draws weight the safe top-SV directions; both
like-for-like joins co-reported). HONESTY: nothing guaranteed,
nothing shopped.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T183 —"
card = """## T184 — e212: the band closes — the instrument-shadow saga's end (2026-10-02 ~16:40Z)

The named debt pays and the saga ends quietly: same-instrument,
the pristine band reads 0.756 against the walled family's 0.85-0.92
(ratio 0.844) — the celebrated 2-3x gap was the cross-instrument
row plus the lottery, now bounded at <= 16% and inside the noise.
THE BAND SCALAR CLOSES AS A NON-PROPERTY — the wall neither buys
nor costs noise tolerance; its ledger (protection, continuity
bounds, the flat phase, the anatomy-independence) stands complete
without it. THE INSTRUMENT-SHADOW SAGA'S THREE ACTS, for the
record: the free find minted (e209: "the walled bands grow 2-3x");
the shadow named (e211: same-instrument per-direction medians
match; both mechanisms dead); the close (e212: the like-for-like
random join bounds the residual inside the noise) — three cells
from minting to burial, the census discipline's standard operating
speed on its own objects.

"""
assert anchor in t and "## T184" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e212 |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| e212 | THE SAME-INSTRUMENT PRISTINE BAND | DONE 16:40Z (T184: GRADED — the 2-3x gap DEAD at the same instrument (bounded <= 16%, inside the SE window); the band scalar CLOSED as a non-property; the instrument-shadow saga's three acts complete: minted->named->buried) |", 1)
i = q.index("\n", q.index("| e212 |")) + 1
row2 = ("| e213 | THE PATH-INDEPENDENCE CENSUS (T183's open object: is the 124M cross-wash stability (0.4212 vs "
        "0.4202) general beyond one battery? every battery x both saved washes) | DISPATCHED 16:41Z — CPU eval |\n")
q = q[:i] + row2 + q[i:]
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T16:41:00Z"
st["current_experiment"] = ("e212 FOLDED (T184: the band closes as a non-property - the shadow saga ends). Fleet: "
                            "g1d (GPU, the base redraw) + e213 DISPATCHED (CPU: the path-independence census)."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e212 folded")
