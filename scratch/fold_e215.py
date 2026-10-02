# -*- coding: utf-8 -*-
"""Fold e215: NOTES, T189, QUEUE, STATE."""
import io, json

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e215 — the relational signature: GRADED — THE SORTING KEY IS FAMILY-FIRST, NOT FAMILY-ONLY (family carries 58% of the variance; the exposure story dead: founders hold at zero cues, products collapse at 232; the key replicates across washes at rho 0.939; the within-family residue is the ratio device) (2026-10-02 ~15:40Z) — DONE

WHAT WE DID: desk+eval on e214's committed 54-probe records; a
hand-registered 6-family typology; the predictor ladder (family
ANOVA vs frequency/length with family-partials); the re-probes at
dp 0.0; 23.8s CPU.

WHAT WE SAW (T189): FAMILY CARRIES 58% OF THE HOLD-RATIO VARIANCE
(F 13.1, eta2 0.577) yet the registered 2x separation bar fails
(SSb/SSw 1.37 — the product and rev-capital families internally
wide): FAMILY-FIRST, NOT FAMILY-ONLY. THE EXPOSURE STORY IS DEAD
IN THIS CORPUS: founders HOLD at ZERO corpus cues while products
collapse at 232 "made" cues; cap-cur/near/rev share the same four
"capital" cues and split three ways; probe-level exposure is 0/54
by the contamination gates. THE TABLE: lang HOLDS (0.87/0.82) +
founder-anchor HOLDS (0.81/0.81, at zero exposure); cap-cur
COLLAPSES (0.20/0.24); product/near-uscap/rev-capital MIXED. THE
SORT EMERGES +10 -> +50 (all families 0.9-1.0 at +10). THE KEY
REPLICATES: Spearman(hr_w1, hr_w2) = 0.939; the hold-class
agreement 81%. THE WITHIN-FAMILY RESIDUE IS THE RATIO DEVICE: p0
carries it (|partial| 0.45) — T178/T186's law at probe level: THE
SORTING KEY IS FAMILY x HEIGHT, W028's two-layer law meeting
T187's erosion order. HONESTY: the typology hand-registered, not
blind to the outcome (disclosed); n=2 washes; the unbalanced n's
(near=3); the answer-length arm structurally null; the
separation-bar reading frozen literal (F co-reported).
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T188 —"
card = """## T189 — e215: family x height — the two-layer law meets the erosion order (2026-10-02 ~15:40Z)

The sorting key resolves as the law's own product: FAMILY-FIRST
(58% of the variance; lang and founder-anchor hold — cap-cur
collapses) crossed with HEIGHT (the within-family residue carried
by the baseline p0 — the ratio device at probe level). THE
EXPOSURE KILLER: founders hold at ZERO corpus cues while products
collapse at 232 — the wash's sorting is not about what it has
seen; it is about WHAT KIND OF RELATION the probe encodes
(language and unique-anchor relations survive; capital/currency
relations die) — and how strong the probe started. THE EMERGENCE
(+10 -> +50): the sort is not instant — the first ten steps
leave every family intact; the relational signature is the
mid-dose object (echoing T185's mid-dose state function). THE
RELIABILITY (rho 0.939 cross-wash): the key is physics, not
stream. THE NAMED OPEN THREAD: what sorts WITHIN the mixed
families (the product split; the tmpl width) — family x height is
the model, its residual the question.

"""
assert anchor in t and "## T189" not in t
t = t.replace(anchor, card + anchor, 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e215 |"):]; row = row[:row.index("\n")]
q = q.replace(row, "| e215 | THE RELATIONAL SIGNATURE | DONE 15:40Z (T189: GRADED — FAMILY-FIRST not family-only (58% of variance; the 2x bar failed on the wide mixed families); THE EXPOSURE STORY DEAD (founders hold at 0 cues, products collapse at 232); the key replicates (rho 0.939); THE SORTING KEY IS FAMILY x HEIGHT — the two-layer law meeting the erosion order; the sort emerges +10->+50) |", 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-02T15:41:00Z"
st["current_experiment"] = ("e215 FOLDED (T189: the sorting key is FAMILY x HEIGHT - the two-layer law meeting the "
                            "erosion order). Fleet 0; the named threads: the within-family residue; the third wash "
                            "draw; the exposure-immunity follow-ups."
                            )
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("e215 folded")
