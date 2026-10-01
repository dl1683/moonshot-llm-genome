# -*- coding: utf-8 -*-
"""R60 close: auditor repairs + the REVIEWS entry + stamps."""
import io, json

# --- repairs ---
t = io.open("THINKING.md", encoding="utf-8").read()
old = "e189's census survives\nas the mechanism of the FLIP (W024), not the currency."
if old not in t:
    old = "e189's census survives as the mechanism of the FLIP (W024), not the currency."
assert old in t, "t141 flip"
t = t.replace(old, "the census survives as the mechanism of the ATTENUATION (W024's flip corrected to attenuation by the chart), not the currency.", 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

p = "scratch/day6_paper_skeleton.md"
s = io.open(p, encoding="utf-8").read()
old = "the normalizer flips the SIGN of fact-relevance, -0.0385 vs +0.0981"
if old in s:
    s = s.replace(old, "the normalizer ATTENUATES fact-relevance ~2.5x at a matched point (the earlier sign-flip was an estimator artifact)", 1)
    io.open(p, "w", encoding="utf-8").write(s)

p2 = "scratch/day7_skeleton.md"
s = io.open(p2, encoding="utf-8").read()
old = "the normalizer FLIPS THE SIGN of fact-relevance (-0.0385 vs +0.0981, same batch)"
if old not in s2 if False else True:
    pass
if "FLIPS THE SIGN of fact-relevance" in s:
    s = s.replace("the normalizer\n  FLIPS THE SIGN of fact-relevance (-0.0385 vs +0.0981, same\n  batch)", "the normalizer\n  ATTENUATES fact-relevance ~2.5x at a matched point (the flip was\n  an estimator artifact)", 1)
if "FLIPS THE SIGN" in s:
    s = s.replace("the normalizer FLIPS THE SIGN of fact-relevance (-0.0385 vs +0.0981, same batch)", "the normalizer ATTENUATES fact-relevance ~2.5x at a matched point (the flip was an estimator artifact)", 1)
io.open(p2, "w", encoding="utf-8").write(s)

n = io.open("NOTES.md", encoding="utf-8").read()
old = "(the top-10% front holds 75% of ||g||^2 and the"
if old not in n:
    old = "(the top-10% front holds 75% of ||g||² and the entire lethality"
if old not in n:
    old = "the top-10% front holds 75% of ||g||"
assert old in n, "75pct"
n = n.replace("top-10% front holds 75% of ||g||", "top-10% front holds 86.6% of ||g|| (the committed census read; the agent's 75% had no artifact — corrected per R60-audit)", 1)
old2 = "kappa-derived d_eff <=52k/ (bound direction CORRECTED per R60-critic)/36k/24k vs SVD rank 20/8/7 vs PR 6.4-15.8"
if old2 in n:
    n = n.replace(old2, "kappa-derived d_eff <=52k / 35k / 24k (store 35k; PR range 6.4-15.8 is e131-only; the R60 critic's direction fix carried; 11602's rung-2 kill after its rung-1 stay disclosed)", 1)
else:
    old2 = "kappa-derived d_eff <=52k/ (bound direction CORRECTED per R60-critic)"
    assert old2 in n, "deff2"
    n = n.replace(old2, "kappa-derived d_eff <=52k / 35k / 24k (direction corrected; PR e131-only; 11602 rung-2 disclosed)", 1)
io.open("NOTES.md", "w", encoding="utf-8").write(n)

d = "DAY_SEVEN_REPORT.md"
s = io.open(d, encoding="utf-8").read()
old = "Open: the rhythm's controls (g2g, out); the"
if old in s:
    s = s.replace(old, "Open: the rhythm's seed ladder (licensed); the", 1)
    io.open(d, "w", encoding="utf-8").write(s)

q = io.open("QUEUE.md", encoding="utf-8").read()
lines = q.split("\n")
seen_e193 = 0
out = []
for ln in lines:
    if ln.startswith("| e193 |"):
        seen_e193 += 1
        if seen_e193 > 1:
            continue
    out.append(ln)
q = "\n".join(out)
io.open("QUEUE.md", "w", encoding="utf-8").write(q)

# --- REVIEWS entry ---
rv = io.open("REVIEWS.md", encoding="utf-8").read()
sep = "---\n\n---"
entry = """---

## R60 — the wave audited: the numbers are real; the corrections' copies chased; the next wave shaped (2026-10-01, folded ~17:20Z)

Trigger: the day's wave closed (T144-T152) + the inherited R59
audit mandate. Trio complete; e193 (the lineage replicate)
launched mid-review from the ideator's top rank.

AUDITOR (scratch/r60_auditor.md; ~60 figures recomputed — the
inherited mandate discharged): every headline reproduces except
one unsupported co-read and one backwards inequality — opt2's
"75% of ||g||^2" had NO artifact (the committed census reads
86.6% — corrected) and NOTES's d_eff bound was the uncorrected
flip copy (<=52k/35k/24k — corrected, with the PR scope and the
11602 rung-2 disclosure). THREE SURVIVING FLIP COPIES killed
(skeleton R2b, day7-skeleton, T141 — the wave's own correction
now lives everywhere it is cited). g1bS's hard stop verified as a
model negative; opt1b3's fold carries numbers not phrases; e182c's
replay premise sound; g2g "the best fold of the wave". Ledger
minor drift fixed (a duplicated e193 row; DAY7's stale open item).

CRITIC (scratch/r60_critic.md; absorbed at landing + repairs in
51bb934): the terrain is a slice that misses its own steepest wall
(in-span 0.56-0.61 contradicts "the most lethal direction" — T143
amended; three rays are not a map); the lethal-front rests on
three conventions + one instant of selection (the magnitude-shuffle
breaker queued — sign-pairing was convicted by intervention,
magnitude-pairing never was); g2g's +0.07 is an anecdote until the
seed ladder (protocol specified: >=3 fresh wash seeds x {organ,
count-matched fixed} at 1x, CPU-deterministic, bar same-sign 3/3
with median delta >= +0.03; "worth" struck until then); the
abstract's two weakest sentences repaired (construction preceded
explanation; process not causality-proof). FORCED: the second-
organism replicate — e193b registered (fresh root + TWO facts +
the rider + the in-span range + the magnitude shuffle), composing
with the running e193 (the lineage axis).

IDEATOR (scratch/r60_ideator.md): the next wave ranked — e193 >
g1c-root > g1bS2 > the g2g seed ladder > e194 (the sign-front
mechanism); stranded cells named honestly (g3O's d_eff leg dead,
the span's size ownerless, the retired-by-design set); the three
drafting blockers: the n=1 organism (e193/e193b), g1c-root, the
g2g seed ladder; g1bS2 explicitly NOT a blocker (the scale-bound
negative is writable).

DECISIONS: all repairs applied; e193 running; e193b registered
behind it; the g2g seed ladder DISPATCHING now (the critic's exact
protocol); g1bS2 next GPU slot (staggered); novelty clock stamped
(the ideator + the day's card deaths serve)."""
assert sep in rv
rv = rv.replace(sep, "---\n\n" + entry + "\n\n---", 1)
io.open("REVIEWS.md", "w", encoding="utf-8").write(rv)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-01T17:21:00Z"
st["last_review"] = "2026-10-01T17:20:00Z"
st["last_novelty"] = "2026-10-01T17:20:00Z"
st["current_experiment"] = ("R60 CLOSED (the wave audited: ~60 figures real; the flip copies chased; the next wave "
                            "shaped). Fleet: e193 (CPU, the lineage replicate) + the g2g seed ladder DISPATCHING "
                            "(CPU-deterministic, the +0.03x3 bar). e193b behind e193; g1bS2 next GPU.")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("R60 closed + repairs")
