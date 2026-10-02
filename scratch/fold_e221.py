# -*- coding: utf-8 -*-
"""Fold e221: NOTES, T197, W028 update, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e221 — the compositionality test: GRADED — the hunt's LAST candidate retires WITHOUT naming the dimension (the separation fires only through its anchor singleton: R 2.56x collapses to R' 0.03x leave-one-out, KW p 0.40; the anchor captured BY REGISTRATION (weight zero); class == family XOR is_Gmail, asserted; no absorption — leave-Gmail-out 0.930: the miss replicates without the anchor); THE THIRD DIMENSION STANDS REAL-AND-UNNAMED (2026-10-02 ~19:15Z) — DONE

WHAT WE DID: the hunt's lightest cell (2.9s, zero loads; the
typology hand-registered and blob-certified BEFORE any test); all
5 gates PASS (e216's residual table reproduced at 1e-9).

WHAT WE SAW (T197): THE FOUR-CLASS TYPOLOGY (38 ATOMIC / 10
TOKEN-IDENTITY / 1 FUNCTION-COMPOSED / 5 MULTI-HOP) SEPARATES THE
RESIDUAL ONLY THROUGH ITS ANCHOR SINGLETON — R 2.562/2.629 (clears
the 1.5 bar) but the leave-singleton R' 0.028/0.029 (COLLAPSES;
Kruskal-Wallis p 0.40); the collinearity asserted in code: CLASS ==
FAMILY XOR is_Gmail. THE ANCHOR captured in direction (Gmail
+0.429 vs iPhone -0.206) but BY REGISTRATION (the
FUNCTION-COMPOSED class is Gmail alone — weight zero, the
instrument-tautology guard registered before compute per the
R49/e166 precedent). NO ABSORPTION anywhere near 0.5 (the best
0.924 vs the 0.934 baseline; the leave-Gmail-out baseline 0.930 —
THE MISS REPLICATES WITHOUT THE ANCHOR). THE UNGUARDED co-report:
R + anchor alone would have read COMPOSITIONALITY-NAMES-IT — the
singleton guard (registered at the typology's commit) is what
blocks it, fully reconstructible in metrics. THE HUNT'S LEDGER,
CLOSED: the third dimension is REAL (e219's confound break),
BEYOND HEIGHT (e218), NOT ENTRENCHMENT (e219), NOT TOKENS (e220),
NOT COMPOSITIONALITY-AS-TYPOLOGY (here) — THE LAB'S HONEST OPEN
OBJECT: REAL-AND-UNNAMED.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T196 —"
card = """## T197 — e221: the hunt closes at real-and-unnamed (2026-10-02 ~19:15Z)

The last registered candidate retires in the hunt's most
instructive way: the typology's apparent separation (R 2.56x) was
Gmail alone — the singleton guard (registered before compute, the
e166 instrument-tautology lesson applied preemptively) caught what
would have been a false naming. The anchor itself is honest but
weight-zero: Gmail is the only FUNCTION-COMPOSED probe BY
DEFINITION, so its capture is registration, not prediction. And
the deepest texture: the miss replicates WITHOUT the anchor (the
leave-Gmail-out baseline 0.930) — the third dimension is not
Gmail's story; it is 54 probes' story, and no registered feature
family names it. THE HUNT'S FINAL LEDGER: real, beyond height,
not entrenchment, not tokens, not compositionality-as-typology —
REAL-AND-UNNAMED, the lab's honest open object. THE META-LESSON
of the hunt (five cells, e216-e221): each test retired one
candidate cleanly, one falsifier fired before it could mislead,
and the object survived everything — the census discipline at its
best is knowing what NOT to claim.

"""
assert anchor in t and "## T197" not in t
t = t.replace(anchor, card + anchor, 1)

# W028 final update
i = t.index("## W028 ")
line = t[i:t.index("\n", i)]
t = t.replace(line, line + " [HUNT CLOSED ~19:15Z: the third dimension REAL-AND-UNNAMED — the five-cell identity hunt (residual -> beyond height -> not entrenchment -> not tokens -> not compositionality-as-typology) retired every candidate; the singleton guard caught the false naming; the miss replicates without its own anchor]", 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
q = q.replace("| DISPATCHED 18:45Z — bars: COMPOSITIONALITY-NAMES-IT / COMPOSITIONALITY-SILENT (real-and-unnamed) / GRADED |",
              "| DONE 19:15Z (T197: GRADED — the singleton guard caught the false naming (R 2.56x collapses to R' 0.03x; the anchor BY REGISTRATION); no absorption (leave-Gmail-out 0.930 — the miss replicates without the anchor); THE HUNT CLOSED: the third dimension REAL-AND-UNNAMED — the lab's honest open object) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T19:16:00Z"
st["current_experiment"] = ("e221 FOLDED (T197: the hunt closes at REAL-AND-UNNAMED - the singleton guard caught the "
                            "false naming; the miss replicates without its own anchor). Fleet 0; all arcs closed."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e221 folded - the hunt closed")
