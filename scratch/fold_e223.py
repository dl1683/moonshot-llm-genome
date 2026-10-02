# -*- coding: utf-8 -*-
"""Fold e223: NOTES, T201, T182 amendment, QUEUE, STATE."""
import io, re, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e223 — the exposure test on the strong-ordering root: NULL-CONFIRMS — every same-axis residue within +-0.0002 of the pass-back arithmetic (BOTH signs; the drift-side floor row included; the control riding a 1.47x mechanical ceiling to 4 decimals); no cross-ray immunity vs control; T182's ORDERING RETIRES AS A CORRELATION, full stop (2026-10-02 ~21:30Z) — DONE

WHAT WE DID: the exposure test on the e131 root (the strong-ordering
organism — e222's stated limit) with BOTH signs (the +v exposure
and the -v0 DRIFT side; the wash drifts -4.73 along v0); the
pass-back arithmetic for both signs disclosed before compute; all
9 gates PASS (the span IS e211's committed root-P span at 0.0 rel
dev); 320.6s CPU.

WHAT WE SAW (T201): NULL-CONFIRMS — every same-axis residue within
+-0.0002 of pure position arithmetic (the ceiling rows at +eps
(1.100x/1.086x/1.473x) AND the drift-side floor row at -eps
(0.900x)); the cross rays move +-4-11% with NO immunity signature
vs control; the CE_R canary clean. T182'S ORDERING RETIRES AS A
CORRELATION, FULL STOP: not intervenable at one dose, either sign,
on the organism where the ordering is STRONGEST. RESIDUAL HONESTY:
the ordering does not extend monotonically to the span's tail
(dir19 out-tolerates dir0 at this root — e211's +0.809 is a top-8
fact, disclosed); the remaining named limit: the TRAINING-STEP
exposure (immunity-to-training untested — a displacement is the
wash's mechanism at one remove, per-step 1.654 vs eps 0.276).
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T200 —"
card = """## T201 — e223: the vaccination retires (2026-10-02 ~21:30Z)

The strong-ordering replication closes the exposure thread with
teeth: on the organism where the SV-energy ordering is strongest,
at both signs (including the mechanically-sensitizing drift
side), the tolerance after sub-lethal pre-exposure is pure
position arithmetic to four decimals — the reading is NOT
intervenable, and T182's exposure-immunity ordering retires as a
correlation, full stop. The extra honesty: the ordering itself is
a top-8 fact (the span's tail inverts locally), so the
correlation being retired is thinner than it looked. THE ONE
REMAINING LIMIT, named: the training-step exposure (a few AdamW
steps along the span — immunity-to-TRAINING, the wash's real
mechanism); it stays unowed until an interpretation demands it.
THE EXPOSURE ARC'S LEDGER: a correlation found (e211), a
mechanism candidate named (T182), a causal test designed with
pre-registered arithmetic (e222), a NULL at one dose, and a
NULL-CONFIRMS at the strongest organism with both signs — a
candidate born and retired cleanly in five cells, the census
discipline's standard arc.

"""
assert anchor in t and "## T201" not in t
t = t.replace(anchor, card + anchor, 1)

m = re.search(r"^## T182 — .*$(.*?)(?=^## T181)", t, re.M | re.S)
assert m, "t182"
body = m.group(1).rstrip("\n")
amend = """

[E222+E223 RESOLUTION ~21:30Z]: the exposure-immunity reading
RETIRED AS A CORRELATION (not intervenable at one dose, either
sign, on both organisms — the strong-ordering root included; the
ordering a top-8 fact, the tail locally inverted). The surviving
lesson of this card: the instrument shadow; the per-direction
safety ordering stands as descriptive geometry only."""
t = t[:m.start(1)] + body + amend + "\n\n" + t[m.end(1):]
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
q = q.replace("| DISPATCHED 20:25Z — bars: IMMUNITY-CAUSAL / SENSITIZATION / NULL-CONFIRMS / GRADED |",
              "| DONE 21:30Z (T201: NULL-CONFIRMS — both signs on the arithmetic to 4 decimals; no cross-ray immunity; T182's ordering RETIRES as a correlation, full stop; the top-8 caveat; the training-step limit named) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T21:31:00Z"
st["current_experiment"] = ("e223 FOLDED (T201: the vaccination retires - NULL-CONFIRMS both signs on the strongest "
                            "organism). Fleet 0; every thread of the entry resolved or honestly fenced."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e223 folded")
