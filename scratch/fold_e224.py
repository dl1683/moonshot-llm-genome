# -*- coding: utf-8 -*-
"""Fold e224: NOTES, T203, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e224 — the training-step exposure: TRAINING-IMMUNITY fires on the RAW clause (b) — and the on-ray decomposition takes most of it back (+0.088 mechanical + +0.046 corrected, sub-bar; clause (a) missed by 0.0008); the real finding: THE NORMALIZER'S OFF-AXIS GEOMETRY TOLERIZES (both arms' same-axis residues positive, scaling with off-axis L2 — the traversal gentler than the displacement) (2026-10-02 ~23:05Z) — DONE

WHAT WE DID: the exposure arc's last named limit discharged —
REAL TRAINING (2 AdamW steps, the wash's own optimizer/batches,
gradients projected onto v0 vs a random in-span control, matched
displacement T=0.827 via the both-alive rung ladder, 3x e223's
dose); 10/10 gates (the span PR at 0.0 rel; the synthetic batches
bitwise the wash's own); 180.4s CPU.

WHAT WE SAW (T203): CLAUSE (b) RAW +0.1341 FIRES
TRAINING-IMMUNITY AS REGISTERED — but the on-ray decomposition
inside the verdict takes most of it back: +0.0881 MECHANICAL (the
control's own on-mid shift, confirmed to +0.010 — the desk's
pre-registered "no mechanical loading at mid" claim was WRONG,
disclosed as a deviation) + +0.0460 ON-RAY-CORRECTED (sub-bar);
clause (a) +0.0492 missed by 0.0008. THE GENUINE, REPLICABLE
TEXTURE: BOTH arms' same-axis residues are POSITIVE (+0.049 at
off-axis 0.334; +0.097 at 0.501) where e222/e223's exact-axis
displacements sat at +-0.0002 — THE NORMALIZER'S OFF-AXIS GEOMETRY
TOLERIZES: a trained traversal is gentler than an equal-L2
displacement, in both arms, scaling with off-axis L2 — T139/g12's
"the normalizer sets the pace/aim" gains a third clause: THE
TRAVERSAL IS GENTLER. Honest strength: A RAW-FORMULA FIRE WEARING
GRADED CLOTHES — the corrected vaccination content is sub-bar at
n=1; nothing was un-fired post-hoc. HONESTY: n=1; the projections
the wash's gradients constrained (one-axis, 2-step), not the wash;
the dose displacement-set (Adam's scale invariance); the shared-T
binding (the control's fragility capped the dose at 0.5x the
wash's step).
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T202 —"
card = """## T203 — e224: the traversal is gentler — the exposure arc's true residue (2026-10-02 ~23:05Z)

The training-step cell closes the exposure arc with the honest
split: the raw formula fired (the registration's letter honored,
nothing un-fired post-hoc) but the agent's own decomposition
shows most of the fire is the control's mechanical geometry, and
the corrected vaccination content is sub-bar. THE FINDING THAT
SURVIVES: both arms' same-axis tolerances ROSE with training
where pure displacement moved them not at all — THE NORMALIZER'S
OFF-AXIS GEOMETRY TOLERIZES. The off-axis (per-coordinate-normalized)
component of a trained step buys tolerance proportional to its L2 —
the traversal is gentler than the displacement, in both directions
tested. THE EXPOSURE ARC'S LEDGER (e211 -> e224, seven cells): a
correlation found, a candidate named, two displacement NULLs, a
training test with a raw fire and an honest retraction — and one
real residue: the gentleness of trained traversal. THE NORMALIZER'S
THREE CLAUSES NOW: sets the pace (T139), sets the aim (g12), and
travels gently (e224).

"""
assert anchor in t and "## T203" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
q = q.replace("| DISPATCHED 21:40Z — bars: TRAINING-IMMUNITY / TRAINING-NULL / SENSITIZATION / GRADED |",
              "| DONE 23:05Z (T203: TRAINING-IMMUNITY on the RAW clause (b) with the on-ray decomposition taking most back (+0.088 mechanical / +0.046 corrected sub-bar; clause (a) by 0.0008); THE REAL RESIDUE: the normalizer's off-axis geometry TOLERIZES (both arms, scaling with off-axis L2) — the traversal gentler than the displacement; the exposure arc closed in seven cells) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T23:06:00Z"
st["current_experiment"] = ("e224 FOLDED (T203: the traversal is gentler - the exposure arc closed). Fleet 0. THE "
                            "SUCCESSOR CHAIN ENDED. Next (per the course correction): the fresh-questions review + "
                            "the day-eight synthesis. THE BEAT GUARD IS LIVE (lab/beat_guard.py; wired into the "
                            "heartbeat)."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e224 folded - the chain ended")
