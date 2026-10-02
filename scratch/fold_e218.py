# -*- coding: utf-8 -*-
"""Fold e218: NOTES, T191, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e218 — the nonlinear height test: BEYOND-HEIGHT — the third sorting dimension is NOT height in any registered dress (rank 0.930 / logit 0.913 / quad 0.938 — all far above the 0.5 absorption line; even family-specific slopes leave rho 0.910); the named splits survive under every form; THE PROBE-FEATURE HUNT OWED (2026-10-02 ~16:40Z) — DONE

WHAT WE DID: pure desk (2.3s, zero loads; the linear arm
reproduced e216's fit at dp 0.0 — the new G_BASELINE gate); the
four height forms + two outside-adjudication competitors; the
decisive cross-wash residual test per form.

WHAT WE SAW (T191): NO FORM COMES WITHIN 0.4 OF THE ABSORPTION
LINE: linear 0.934 / rank 0.930 / logit 0.913 (the best absorber —
moves rho by 0.02, buys +0.04 R2) / quad 0.938; even the
family-slopes form (12 params, the richest height functional tested,
outside the adjudication) leaves rho 0.910. THE NAMED SPLITS
SURVIVE under the best form (the product contrast +3.28/+2.64 SD;
the tmpl width 0.80/0.79). THE THIRD DIMENSION IS BEYOND HEIGHT:
no registered function of p0+family absorbs it — the per-probe
idiosyncrasy is genuinely new information. THE HUNT OWED, with
named candidates: independent entrenchment (measured beyond the
wash battery), internal token structure, the relation's
compositionality. HONESTY: the four forms carry no information
beyond p0 (rank strictly less) — BEYOND-HEIGHT means "no
registered function of p0+family," not a proof none exists (the
fslopes disclosure at the boundary); the xwash correlation could
still be shared pipeline texture (the same p0 denominator) — the
hunt's first job is to break that confound.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T190 —"
card = """## T191 — e218: beyond height — the per-probe idiosyncrasy is new information (2026-10-02 ~16:40Z)

The nonlinear test closes the height family cleanly: no function of
p0 (linear, rank, logit, quadratic, even family-specific slopes)
absorbs the replicating miss — the best form moves rho by two
hundredths. THE THIRD DIMENSION IS GENUINELY BEYOND HEIGHT: which
probe holds within a family is information the baseline does not
carry in any dress. THE HONEST BOUNDARY: the confound that the
cross-wash correlation could be shared pipeline texture (both
washes read the same probes through the same p0 denominators) is
named as the hunt's first target — the third dimension's reality
rests on the miss replicating for reasons beyond the instrument;
the first candidate feature (independent entrenchment, measured
through a different channel) would break or confirm exactly that.
THE SIGNATURE'S STATE: family (58%) x height (nonlinear, ~6%) x
THE THIRD DIMENSION (the replicating residual, identity unknown) —
the relational signature now three-layered, its deepest layer
unnamed and the hunt specified.

"""
assert anchor in t and "## T191" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e218 |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| e218 | THE NONLINEAR HEIGHT TEST | DONE 16:40Z (T191: BEYOND-HEIGHT — no function of p0+family absorbs the third dimension (the best form moves rho by 0.02); the named splits survive under every form; THE PROBE-FEATURE HUNT OWED: independent entrenchment / token structure / compositionality; the pipeline-texture confound named as the first target) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T16:41:00Z"
st["current_experiment"] = ("e218 FOLDED (T191: BEYOND-HEIGHT - the third dimension is genuinely new information). "
                            "Fleet: e217 (GPU, the third wash draw) alone."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e218 folded")
