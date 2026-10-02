# -*- coding: utf-8 -*-
"""Fold e220: NOTES, T195, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e220 — the token-structure test: TOKEN-SILENT — the hunt's second candidate retires cleanly (all four varying token features at |rho| <= 0.069; no substitution moves the residual off 0.934; the Gmail/iPhone anchor token-identical with opposite fates — the best token model WIDENS the gap); the hunt's last candidate is COMPOSITIONALITY (2026-10-02 ~18:35Z) — DONE

WHAT WE DID: the six registered token features vs e216's residual
(re-derived at dp 0.0); the absorption battery (family + each
feature replacing p0); the Gmail/iPhone anchor study; 12.5s, zero
model loads; 7/7 gates.

WHAT WE SAW (T195): TOKEN-SILENT — of the six features, two
construction-degenerate (the battery's single-token rule; the
exemplar rotation), the four varying ones at |rho| <= 0.069 (cue
tok count +0.069 the best); EVERY substitution leaves the
cross-wash residual >= 0.946 (vs the 0.934 baseline — token
structure adds nothing); the 4-feature composite +0.018/-0.028.
THE NAMED SPLITS SURVIVE under the best token model (product
+3.08/+2.59 SD; tmpl width 1.00/1.00). THE GMAIL/IPHONE ANCHOR,
THE SHARPEST SINGLE WITNESS: near token-identical probes (both 0/1M
source frequency; frag 0.167 vs 0.143; cue 8 vs 7 tokens) with
OPPOSITE residuals (+0.43 vs -0.21) — THE BEST TOKEN MODEL WIDENS
THE PAIR (+0.45 vs -0.24): whatever separates them, it is not
tokens. THE HUNT'S LEDGER: the dimension is real (T193), beyond
height (T191), not entrenchment (e219), not token structure
(here) — THE LAST REGISTERED CANDIDATE IS COMPOSITIONALITY (the
relation's internal structure: how the probe's knowledge is
composed — e.g. Gmail = a service-brand relation composed of
token-identity + function; iPhone = a product-token relation).
HONESTY: the frequency proxy family-aligned (the source corpus,
not GPT-2's true pretraining distribution — disclosed); the
Bonferroni note moot (everything far below per-test alpha).
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T194 —"
card = """## T195 — e220: token-silent — the Gmail/iPhone anchor at its sharpest (2026-10-02 ~18:35Z)

The second candidate retires as cleanly as the first: every varying
token feature silent (|rho| <= 0.069), every substitution leaving
the residual untouched. THE ANCHOR IS THE STORY: Gmail and iPhone
are near token-identical (both zero-frequency in the source corpus,
near-identical fragmentation and cue lengths) with opposite fates —
and the best token model WIDENS their gap. Whatever holds Gmail and
kills iPhone is invisible at the token level entirely. THE HUNT AT
ITS LAST DOOR: real (survived the confound break), beyond height,
not entrenchment, not tokens — COMPOSITIONALITY, the relation's
internal structure, is the last registered candidate. THE PATTERN
ONE MORE TIME: each test retires a candidate cleanly, the object
stays sharp, and the hunt converges — the third dimension is
either compositionality (the next cell names it) or something the
lab has not yet thought to register (the honest open door).

"""
assert anchor in t and "## T195" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e220 |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| e220 | THE TOKEN-STRUCTURE TEST | DONE 18:35Z (T195: TOKEN-SILENT — all varying features at |rho| <= 0.069; no absorption; the Gmail/iPhone anchor token-identical with opposite fates, the best model WIDENING the gap; the hunt's last candidate: COMPOSITIONALITY) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T18:36:00Z"
st["current_experiment"] = ("e220 FOLDED (T195: TOKEN-SILENT - the hunt's last candidate is compositionality). "
                            "Fleet: g11 (the crush mechanism) alone."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e220 folded")
