# -*- coding: utf-8 -*-
"""Fold e209: NOTES, T179, T177 amendment, QUEUE, STATE."""
import io, re, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e209 — the census debt: MARGIN-BREAKS at n=7 — the scalar demotes to a DESCRIPTOR; the anatomy: an EPISODE MISMATCH (the margin at the settled state vs the first-episode clock); the walled roots' noise bands GROW 2-3x under the wall (2026-10-02 ~14:00Z) — DONE

WHAT WE DID: the g1b/g1bR 2.74M W1 roots (10902 + 10907/10908)
got their static rays + in-span bands (the e191/e192/e205
machinery; fresh seeds); three deterministic passes bit-identical.

WHAT WE SAW (T179): THE EXTENDED TABLE (n=7) — the new rows R5
0.744x and R6 0.786x sit BELOW 1x yet their seeds' unwalled C-arms
SURVIVED the first wash step (deaths +2/+4): the 2x class line
does not survive the extension; Spearman 0.738 -> 0.225; THE
SCALAR DEMOTES FROM CLASS PREDICTOR TO PER-ORGANISM DESCRIPTOR.
THE ANATOMY (the honest disclosure): AN EPISODE MISMATCH — the
margin is measured at the s300 SETTLED state while the honest
survival column is the lineage's FIRST-episode clock; the frozen
letter joined the episodes and broke (no bar shopping). THE LOOP
CLOSED BY THE CO-READ: at all three s300 roots the fact DIES AT
THE FIRST UNWALLED STEP of the band history (ruler traces
0.0001/0.0266/0.0009) — these organisms are WALL-DEPENDENT, and
their at-or-below-noise margins TRACK EXACTLY THAT: the margin
still describes the s300 organism's OWN unwalled death; it does
not predict another episode's. TWO FREE FINDS: (1) the committed
checkpoints store MID-STRIDE states (settling is the load path;
the settled reads = the committed at ~5e-6); (2) THE WALLED
ROOTS' BANDS ARE 2-3x WIDER than the pristine root's — THE NOISE
BALL GROWS UNDER THE WALL. FORKS REGISTERED: same-episode margins
(at the E131 root per seed — the C-arm clock's own episode); the
walled-band question (does a walled history widen the band?).
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T178 —"
card = """## T179 — e209: the margin demotes — and the demotion's anatomy is the finding (2026-10-02 ~14:00Z)

The census extension breaks the class line honestly, and the
break's anatomy teaches more than the line did: the violated rows
joined a SETTLED-STATE margin to a FIRST-EPISODE clock — an
episode mismatch the frozen letter baked in — and the co-read
shows the walled organisms dying at their first unwalled step
with at-or-below-noise margins tracking exactly that fragility.
THE SCALAR'S HONEST FORM: a SAME-EPISODE descriptor (the
organism's own unwalled death, measured in its own noise units) —
the e208 class line was a within-episode coincidence of the
first four rows. THE FREE FIND MAY OUTLIVE THE SCANDAL: the
walled roots' noise bands are 2-3x WIDER — the wall's protection
GROWS the organism's wash-noise ball (the walled history leaves
the net more displacement-tolerant in random directions? or the
settled mid-stride states carry a wider functional noise floor?)
— the walled-band question, a genuinely new wall property. THE
PROGRAM'S LAW HOLDS THROUGH THE DEMOTION: the margin was minted
at n=4, extended at n=7, and broke — the lab's own census
discipline killing its own scalar's overreach in one cell.

"""
assert anchor in t and "## T179" not in t
t = t.replace(anchor, card + anchor, 1)

m = re.search(r"^## T177 — .*$(.*?)(?=^## T176)", t, re.M | re.S)
assert m, "t177"
body = m.group(1).rstrip("\n")
amend = """

[E209 AMENDMENT ~14:00Z]: the class line BREAKS at n=7 (the
extension's rows violate; the anatomy is an episode mismatch);
the scalar demotes to a SAME-EPISODE descriptor — this card's
"class predictor" claim is withdrawn; the honest residue is
T179's."""
t = t[:m.start(1)] + body + amend + "\n\n" + t[m.end(1):]
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e209 |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| e209 | THE CENSUS DEBT | DONE 14:00Z (T179: MARGIN-BREAKS at n=7 — the scalar demotes to a same-episode descriptor; the anatomy: an episode mismatch; the walled roots' bands GROW 2-3x (the noise ball grows under the wall — a new wall property); the forks registered) |", 1)
i = q.index("\n", q.index("| e209 |")) + 1
row2 = ("| e210 | THE SAME-EPISODE MARGIN (the repair fork: the margin measured at the E131 root per seed — the "
        "C-arm clock's own episode; the class line retested within-episode) | DISPATCHED 14:01Z — CPU eval |\n")
q = q[:i] + row2 + q[i:]
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T14:01:00Z"
st["current_experiment"] = ("e209 FOLDED (T179: the margin demotes - the anatomy the finding; the walled bands GROW). "
                            "Fleet: g1c-root (GPU) + e210 DISPATCHED (CPU: the same-episode margin)."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e209 folded")
