# -*- coding: utf-8 -*-
"""Fold e_chart: NOTES, T150, W024 retirement, T139 re-amendment, paper +
claims-ledger corrections, QUEUE DONE, STATE."""
import io, json, re

n = io.open("NOTES.md", encoding="utf-8").read()
sep = "\n---\n"
entry = """
## e_chart — THE CHART CELL: A2 FLAT-POSITIVE | B2 PARTIAL-PROJECTION — the pump has no cuts; the sign-flip was an estimator artifact; the span kills at 1x; the shuffled sign is inert; the random band >12 (2026-10-01 ~16:00Z) — DONE

WHAT WE DID: e189+e190 merged per the frozen design: the census
(21 states; both W024 estimators gated HARD — same-point raw
+0.09862 vs committed +0.09808; opt1-estimator -0.03851 vs
-0.03851, d 5.4e-6) and the subspace test (SVD of the wash-step
history; in-span random vs out-span wash at the rung ladder; e131
primary + both g3K rulers), plus the shuffled-sign and wider-grid
riders. All 9 gates PASS; 39 checkpoints sha1-hashed; e188/e191/
e192/opt1/g3K loaded, never recomputed; 290.8s CPU eval-only.

WHAT WE SAW (T150): A2 FLAT-POSITIVE — at t=0 EVERY magnitude
class of the wash gradient aligns positively with the fact gradient
(top-0.1% +0.054 carrying 40% of ||g||; 0.1-1% +0.083; 1-10%
+0.040; bottom-90% +0.027; complements positive at all cuts; 0/21
states show stitches-and-cuts). THE PUMP HAS NO CUTS — W024's
picture dies. THE FLIP WAS AN ESTIMATOR-POINT ARTIFACT: at a
matched point flattening only ATTENUATES (+0.0986 -> +0.0396);
opt1's -0.0385 evaluated the fact gradient AFTER the 1.6543-L2
step (the trajectory read is real but post-step). B2 PARTIAL-
PROJECTION — THE SPAN KILLS AT 1x EVERYWHERE: in-span random kills
at rung 1 on all three organisms (fine-D 0.56-0.61 on e131, BELOW
the g-ray's 0.91); the out-span arm splits by organism (g3: the
wash delta IS the first history segment, removed 1.0, residual
kills at rung 8 — inside the random band; e131: the sign-flattened
history contains only 64.2% of the raw ray, so the "out-span"
residual retains 59% of the g-ray and kills at rung 1). The
dimension estimates do NOT reconcile (kappa-derived d_eff >=52k/
36k/24k vs SVD rank 20/8/7 vs PR 6.4-15.8) — the finite-span proxy
caveat carried verbatim; W025's projection-ratio account of the
random band remains UNCONFIRMED. RIDERS: the shuffled-sign ray is
INERT (no pump; no kill through D 2.5; CE_R flat) — the pump and
the sign-ray kill both need the EXACT coordinate-sign pairing;
the random band widens to >12 (>13x the g-ray; still
unresolved-high). HONESTY: n=1 per ruler; snapshot quadrature;
the 0.02 floor declared; the lr1e-4 arm excluded per e188's
provenance flag.
"""
i = n.index(sep) + len(sep)
n = n[:i] + entry + sep + n[i:]
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()
anchor = "## T149 —"
card = """## T150 — the chart: no cuts, no flip — the corrections corrected (2026-10-01 ~16:00Z)

THE CHART CELL corrects the correctors. (1) THE PUMP HAS NO CUTS:
every magnitude class of the wash gradient is fact-positive at
t=0 — W024's stitches-and-cuts dies whole (there are no tiny
cuts; the erosion is not a sign-flattened mass of small
coordinates). (2) THE SIGN-FLIP WAS AN ESTIMATOR-POINT ARTIFACT:
at a MATCHED point, normalization ATTENUATES fact-relevance
(+0.0986 -> +0.0396, ~2.5x) but does not flip it; opt1's -0.0385
trajectory read is real but POST-STEP (it evaluates the fact
gradient after the 1.6543-L2 step has already moved it). T139's
adopted gem and the paper's clause correct to "the normalizer
ATTENUATES the stream's fact-relevance; the trajectory-level
negative read is the post-step view". (3) THE SPAN KILLS AT 1x:
in-span random directions kill BELOW the g-ray itself (0.56-0.61
on e131) — the empirical gradient-history span is lethally
sufficient, stronger than the subspace hypothesis needed; but the
out-span arm's split behavior (rung 8 on g3 vs rung-1-with-59%-
retained on e131) and the non-reconciling dimension estimates
(kappa >=52k vs SVD rank 20) leave W025's projection-ratio account
UNCONFIRMED — the span is real and lethal; its size is not yet
measurable by these instruments. (4) THE SHUFFLED SIGN IS INERT:
exact coordinate-sign pairing owns both the pump and the sign
kill — gradient structure is in the PAIRING, the sharpest form
the structure question has taken. (5) THE RANDOM BAND >12: the
isotropic arm of the terrain widens another 3x (Fig-5 updates).
THE META: the chart was built to check two wonder cards and it
killed both pictures while confirming both questions were worth
asking — the favorite-dies-well pattern, twice in one cell.

"""
assert anchor in t and "## T150" not in t
t = t.replace(anchor, card + anchor, 1)

# W024 retirement
i = t.index("## W024 ")
line = t[i:t.index("\n", i)]
t = t.replace(line, line + " [RETIRED-BY-CHART, 2026-10-01: BOTH pictures died — no cuts (A2: every class positive) and no flip (estimator-point artifact; matched-point read: attenuates +0.099->+0.040). The surviving object: exact coordinate-sign pairing (the shuffled-sign rider). Died well.]", 1)

# T139 re-amendment
m = re.search(r"^## T139 — .*$(.*?)(?=^## T138)", t, re.M | re.S)
assert m, "t139"
body = m.group(1).rstrip("\n")
amend = """

[CHART RE-AMENDMENT, 2026-10-01 ~16:00Z — the flip corrects to
ATTENUATION]: the R58-critic gem adopted above ("the normalizer
FLIPS the sign of fact-relevance, -0.0385 vs +0.0981") is an
ESTIMATOR-POINT ARTIFACT per the chart cell's hard-gated double
anchor: at a matched point the flattening only attenuates
(+0.0986 -> +0.0396); opt1's -0.0385 evaluates the fact gradient
AFTER the Adam step. The honest sentence: "the normalizer
attenuates the stream's fact-relevance ~2.5x at a matched point;
the trajectory-level negative alignment is the post-step view."
The paper's clause corrects with it.]"""
t = t[:m.start(1)] + body + amend + "\n\n" + t[m.end(1):]
io.open("THINKING.md", "w", encoding="utf-8").write(t)

# paper + claims-ledger corrections
p = "scratch/day6_paper_skeleton.md"
s = io.open(p, encoding="utf-8").read()
old = "the normalizer flips the SIGN of fact-relevance:\n   -0.0385 vs +0.0981 on the same batch)"
if old in s:
    s = s.replace(old, "the normalizer ATTENUATES the stream's\n   fact-relevance ~2.5x at a matched point; the trajectory-level negative\n   read is the post-step view — the chart cell's estimator correction)", 1)
old2 = "(the normalizer flips the SIGN of fact-relevance: -0.0385 vs +0.0981)"
if old2 in s:
    s = s.replace(old2, "(the normalizer attenuates fact-relevance ~2.5x at a matched point — the earlier sign-flip was an estimator artifact)", 1)
io.open(p, "w", encoding="utf-8").write(s)

c = "scratch/claims_ledger.md"
s = io.open(c, encoding="utf-8").read()
old = "a fresh\nAdamW moves 1683x a matched SGD step and flips the sign of the\nstream's fact-relevance"
assert old in s, "ledger"
s = s.replace(old, "a fresh\nAdamW moves 1683x a matched SGD step and attenuates the stream's\nfact-relevance ~2.5x at a matched point", 1)
io.open(c, "w", encoding="utf-8").write(s)

q = io.open("QUEUE.md", encoding="utf-8").read()
row = q[q.index("| e189+e190 |"):]; row = row[:row.index("\n")]
new_row = ("| e189+e190 | THE CHART CELL | DONE 16:00Z (T150: A2 FLAT-POSITIVE — the pump has NO cuts (every class "
           "positive; W024 retired whole); the sign-flip was an ESTIMATOR-POINT ARTIFACT (matched-point: attenuates "
           "+0.099->+0.040; T139 + paper corrected) | B2 PARTIAL-PROJECTION — the span kills at 1x (in-span random "
           "0.56-0.61, below the g-ray; the out-span split by organism; dimension estimates do NOT reconcile — W025's "
           "ratio account unconfirmed) | riders: the shuffled sign INERT (exact coordinate-sign pairing owns the pump "
           "and the kill); the random band >12) |")
q = q.replace(row, new_row, 1)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-01T16:01:00Z"
st["current_experiment"] = ("THE CHART CELL FOLDED (T150: no cuts, no flip — the corrections corrected; the span kills "
                            "at 1x; shuffled sign inert; band >12; W024 retired; T139 + paper + abstract corrected). "
                            "Fleet: g2g (GPU) + x1 (CPU).")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("chart folded")
