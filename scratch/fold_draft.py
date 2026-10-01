# -*- coding: utf-8 -*-
"""Apply the drafting agent's five findings + commit the draft."""
import io, json

# 1. C5: 2 of 4 facts
c = "scratch/claims_ledger.md"
s = io.open(c, encoding="utf-8").read()
old = "present on 2 of 3 facts"
assert old in s, "c5 stale"
s = s.replace(old, "present on 2 of 4 facts across 3 organisms (f1 +, e193-f2 -, ZEPHYRA +, MIRABEL -)", 1)
io.open(c, "w", encoding="utf-8").write(s)

# 2. T151 mass note: both instruments real
t = io.open("THINKING.md", encoding="utf-8").read()
old = "top-10% front holds 86.6% of ||g|| (the committed census read; the agent's 75% had no artifact — corrected per R60-audit)"
if old not in t:
    old = "top-10% front holds 86.6% of ||g||"
assert old in t, "t151 mass"
t = t.replace(old, "top-10% front holds ~75% of ||g||^2 at t=0 per opt2's OWN gate (energy_frac 0.7497) — the R60 '86.6% correction' imported e_chart's census at a different convention: TWO instruments, both real, never reconciled; the paper omits mass fractions pending a single-convention read", 1)

# 4. T144 ridge range per e191's metrics
old = "a pump ridge (0.94-0.96 across D\n0.05-0.50, peak 0.9605 at 0.20)"
if old not in t:
    old = "pump ridge 0.94-0.96 across D 0.05-0.50"
if old in t:
    t = t.replace(old, "pump ridge 0.91-0.96 across D 0.05-0.50 (per e191's metrics: 0.910 at D 0.50; peak 0.9605 at 0.20)", 1)
io.open("THINKING.md", "w", encoding="utf-8").write(t)

# Fig-5 caption: fact-level
p = "scratch/day6_paper_skeleton.md"
s = io.open(p, encoding="utf-8").read()
old = "the pump ridge does NOT\n   cross lineages — one panel, two organisms)"
if old not in s:
    old = "the pump ridge does NOT cross lineages — one panel, two organisms)"
assert old in s, "fig5 caption"
s = s.replace(old, "the pump ridge is FACT-LEVEL biography (crosses root-draws, tracks facts: +,-,+,- across 4 facts) — one panel, the organisms)", 1)
io.open(p, "w", encoding="utf-8").write(s)

# NOTES opt2 mass line: reconcile the earlier correction
n = io.open("NOTES.md", encoding="utf-8").read()
old = "top-10% front holds 86.6% of ||g|| (the committed census read; the agent's 75% had no artifact — corrected per R60-audit)"
assert old in n, "notes mass"
n = n.replace(old, "top-10% front holds ~75% of ||g||^2 (opt2's own gate, energy_frac 0.7497; e_chart's census reads 86.6% at its convention — two instruments, unreconciled; the paper omits mass fractions)", 1)
io.open("NOTES.md", "w", encoding="utf-8").write(n)

st = json.load(io.open("STATE.json", encoding="utf-8"))
st["last_heartbeat"] = "2026-10-01T19:55:00Z"
st["current_experiment"] = ("The R2b/R6 draft DELIVERED (scratch/draft_r2b_r6.md) + its five findings applied: C5 "
                            "2-of-4; the mass-fraction pair adjudicated (opt2's own 75% restored — two instruments, "
                            "paper omits); Fig-5's caption fact-level; T144's ridge range per metrics; g1bW's "
                            "metrics-vs-amendment noted. Fleet: g1bS2 (the last bracket).")
io.open("STATE.json", "w", encoding="utf-8").write(json.dumps(st, indent=2, ensure_ascii=False))
print("draft findings applied")
