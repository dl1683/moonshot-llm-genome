# -*- coding: utf-8 -*-
"""Fold e208: NOTES, T177, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e208 — the edge-multiple census: MARGIN-PREDICTS (desk-forced, the fork disclosed) — the 2x noise-margin line separates 4/4 on first-wash survival; the within-organism pair is the scalar's cleanest contrast; the margin is a CLASS separator, not an ordering (2026-10-02 ~13:00Z) — DONE

WHAT WE DID: T173's scalar assembled for every (organism, fact)
pair with committed data; zero fresh compute (no torch import); 3
gates PASS (the e205 multiples reproduce to 0.0; every survival
read re-checked).

WHAT WE SAW (T177): THE TABLE — 3.72x (org1/ZEPHYRA, survived 1
step into the wash), 2.12x (e193b/MIRABEL, survived 2), 1.24x
(e193b/ZEPHYRA under the fallback g+0 ruler, died at t=1), 0.616x
(the half lineage, died at t=1 naturally). MARGIN-PREDICTS FIRES
(desk-forced, disclosed): EVERY margin > 2 ROW OUTLIVED EVERY
margin < 1 ROW; the 2x line separates 4/4 on first-wash survival
(context, unregistered). THE SCALAR'S SHAPE: a CLASS separator,
not an ordering (3.72x died at t=2 while 2.12x reached t=3). THE
WITHIN-ORGANISM GEM (e193b, same walk, same in-span draws, two
rulers): margin 1.24x died at t=1 where 2.12x survived to t=3 —
the margin ordered two facts of ONE organism through their ruler
difference alone (T155's ruler lesson made quantitative). THE
FORK: under the half lineage's counterfactual walk (survived 4)
the verdict flips to DECORRELATED — the fork is on the face of
the adjudication. THE DEBT NAMED: the g1bR 2.74M roots have NO
committed rays (verified) — a static-ray + band cell there breaks
the n=4 table; a second walk realization breaks the
single-realization fragility. HONESTY: n=4; band n=3 with
censoring; three ruler classes; no intervention (a census).
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T176 —"
card = """## T177 — e208: the noise margin earns object status — as a class line (2026-10-02 ~13:00Z)

The census gives T173's scalar its first structure: the 2x line
separates every row on first-wash survival (margin > 2 survives
the first wash step; margin < 1 dies at it) — THE FACT'S
SIGNAL-TO-NOISE MARGIN IS A SURVIVAL CLASS PREDICTOR. The scalar's
honest shape: a CLASS separator (the ordering inside the survivor
class is not margin-driven — a threshold object, like the cliff).
THE WITHIN-ORGANISM CONTRAST is the program's cleanest instrument
move in days: two facts, one organism, one walk, one set of
in-span draws — ordered by their margins through the ruler
difference alone. THE PROGRAM'S LAW GAINS A MEMBER: the margin is
a HEIGHT scalar (per-organism, lottery-flavored — the fork flips
it) that nonetheless PREDICTS A CLASS (the shape layer): the
lottery draws the height, and the height sets the class.

"""
assert anchor in t and "## T177" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e208 |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| e208 | THE EDGE-MULTIPLE CENSUS | DONE 13:00Z (T177: MARGIN-PREDICTS — the 2x line separates 4/4 on first-wash survival; a CLASS separator not an ordering; the within-organism pair the cleanest contrast; the fork disclosed; the debt: the g1bR roots' rays + a second realization) |", 1)
i = q.index("\n", q.index("| e208 |")) + 1
row2 = ("| e209 | THE CENSUS DEBT (the g1bR 2.74M roots' static rays + bands — the e191/e192 machinery; breaking "
        "the n=4 margin table) | DISPATCHED 13:01Z — CPU eval |\n")
q = q[:i] + row2 + q[i:]
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T13:01:00Z"
st["current_experiment"] = ("e208 FOLDED (T177: the noise margin earns object status as a class line). Fleet: g1bS8 "
                            "(GPU) + e209 DISPATCHED (CPU: the census debt - the g1bR roots' rays)."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e208 folded")
