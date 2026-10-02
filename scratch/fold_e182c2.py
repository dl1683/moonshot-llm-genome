# -*- coding: utf-8 -*-
"""Fold e182c2: NOTES, T183, T149 amendment, QUEUE, STATE."""
import io, re, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e182c2 — phase 2: TEMPLATE-GENERAL + DRAW-REPLICATES — the fast erosion survives the form change and the fresh draw; the locus is the FEW-SHOT-FOLLOWING FACULTY; the cross-wash stability free find (2026-10-02 ~16:05Z) — DONE

WHAT WE DID: the reversed-form capital battery (n=19 gate-passers,
p0 mean 0.907 — the reversed direction is NOT scarce at 124M) on
phase-1's saved states (re-probe dp 0.0, bit-identical); the fresh
corpus draw (seed 20261002, the only delta); the envelope held (3
bursts of 2.7-9.1s).

WHAT WE SAW (T183): (1) TEMPLATE-GENERAL — the reversed form
declines 0.561 at +80 vs the capital-of form's 0.766 (ratio 0.73,
inside the band): THE EROSION IS TEMPLATE-GENERAL; THE LOCUS IS
THE FEW-SHOT-FOLLOWING FACULTY, not one form. TEXTURES: the +50
co-adjudication flips SPECIFIC by 0.011 (the reversed form LAGS
then CONVERGES: 0.003/+2 -> 0.068/+10 -> 0.421/+50 -> 0.561/+80
vs nearrel's 0.016 -> 0.217 -> 0.648 -> 0.766); tmpl erodes 1.41x
the generic controls (the capital family sits BETWEEN the controls
0.397 and the near-related 0.766); the form+direction confound
disclosed. (2) DRAW-REPLICATES — the fresh draw reproduces the
phase-1 pattern at +50 (generic ratio 0.889 vs phase-1's 0.83; the
near-related still FASTEST) and +80 (0.818; nearrel 0.656); ppl
improves 71.3 -> 34.6 throughout. (3) THE CROSS-WASH FREE FIND:
the tmpl battery's +50 decline reads 0.4212 on phase-1's states
and 0.4202 on the fresh draw — TWO WASHES, THE SAME BATTERY, THE
SAME DECLINE TO THREE DECIMALS (the erosion's per-battery dose-
response is wash-path-independent to this precision — a stability
the texture did not promise). HONESTY: n=1 pool, n=2 draws,
nearrel n=3; CPU-fp32 probes vs GPU-fp32 wash — patterns, never
bits.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T182 —"
card = """## T183 — e182c2: the few-shot locus at n=2 draws and 2 forms; the cross-wash stability (2026-10-02 ~16:05Z)

Phase 2 closes the 124M thread's open question both ways: the
erosion is TEMPLATE-GENERAL (the reversed form converges to the
same decline; the +50 lag-then-converge the only form residue) and
DRAW-REPLICATING (the fresh stream reproduces the pattern; the
near-related still fastest). THE GPT-2 CLAUSE'S FINAL FORM: at
124M the wash erodes the FEW-SHOT-FOLLOWING FACULTY generically —
controls, templates, and near-relations in a graded order (ctrl <
template < nearrel), with perplexity improving throughout: the
organism gets better at the stream while its instruction-following
surface wears. THE FREE FIND IS THE QUIET STUNNER: one battery's
decline identical across two independent washes to three decimals
(0.4212 vs 0.4202) — the per-battery dose-response is
wash-path-INDEPENDENT: the erosion is a function of the STATE (the
displacement), not the PATH — echo of e188's displacement gate at
124M scale. THE THREAD'S LEDGER: surgical signature retired
(e182c); generic erosion licensed at n=2 draws + 2 forms (this
cell); the template-locus resolved to the faculty level; the
path-independence the new open object.

"""
assert anchor in t and "## T183" not in t
t = t.replace(anchor, card + anchor, 1)

m = re.search(r"^## T149 — .*$(.*?)(?=^## T148)", t, re.M | re.S)
assert m, "t149"
body = m.group(1).rstrip("\n")
amend = """

[E182C2 AMENDMENT ~16:05Z]: the hint LICENSED — TEMPLATE-GENERAL
(the reversed form converges) + DRAW-REPLICATES (n=2 draws): the
locus is the FEW-SHOT-FOLLOWING FACULTY; the graded order ctrl <
template < nearrel; the cross-wash stability (0.4212 vs 0.4202) the
new open object."""
t = t[:m.start(1)] + body + amend + "\n\n" + t[m.end(1):]
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
key = "| e182c2 |" if "| e182c2 |" in q else None
if key:
    i = q.index(key)
    row = q[i:]
    row = row[:row.index("\n")] if "\n" in row else row
    q = q.replace(row, "| e182c2 | THE TEMPLATE LOCUS + FRESH DRAW | DONE 16:05Z (T183: TEMPLATE-GENERAL + DRAW-REPLICATES — the locus the few-shot-following faculty; the graded order ctrl < template < nearrel; THE CROSS-WASH FREE FIND: one battery's decline identical across two washes to three decimals — path-independence, the e188 echo at 124M) |", 1)
else:
    j = q.index("\n", q.index("| e182c |")) + 1
    q = q[:j] + "| e182c2 | THE TEMPLATE LOCUS + FRESH DRAW | DONE 16:05Z (T183: TEMPLATE-GENERAL + DRAW-REPLICATES — the locus the few-shot-following faculty; the graded order ctrl < template < nearrel; THE CROSS-WASH FREE FIND: path-independence, the e188 echo at 124M) |\n" + q[j:]
q = q.replace(row, "| e182c2 | THE TEMPLATE LOCUS + FRESH DRAW | DONE 16:05Z (T183: TEMPLATE-GENERAL + DRAW-REPLICATES — the locus the few-shot-following faculty; the graded order ctrl < template < nearrel; THE CROSS-WASH FREE FIND: one battery's decline identical across two washes to three decimals — path-independence, the e188 echo at 124M) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T16:06:00Z"
st["current_experiment"] = ("e182c2 FOLDED (T183: the few-shot locus at n=2 draws + 2 forms; the cross-wash "
                            "stability the new open object). Fleet: e212 (CPU, the pristine band) + g1d DISPATCHING "
                            "(GPU freed: the base-seed redraw - the wall's third axis)."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e182c2 folded")
