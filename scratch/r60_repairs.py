# -*- coding: utf-8 -*-
"""R60-critic immediate repairs: d_eff flip, T143 contradiction, 11602
disclosure, 'worth' repair, abstract S3/S4."""
import io, re

# 1. d_eff flip in NOTES
n = io.open("NOTES.md", encoding="utf-8").read()
old = "kappa-derived d_eff >=52k/"
new = "kappa-derived d_eff <=52k/ (bound direction CORRECTED per R60-critic)"
assert old in n, "notes flip"
n = n.replace(old, new, 1)
io.open("NOTES.md", "w", encoding="utf-8").write(n)

t = io.open("THINKING.md", encoding="utf-8").read()

# 2. T150 card: the flip + the 11602 disclosure + the SVD-rank cap
m = re.search(r"^## T150 — .*$(.*?)(?=^## T149)", t, re.M | re.S)
assert m, "t150"
body = m.group(1).rstrip("\n")
amend = """

[R60-CRITIC REPAIRS ~17:15Z]: (a) the d_eff BOUND WAS FLIPPED in
this fold — kappa >= 7.25 implies d_eff <= 52k, not >=; corrected.
(b) The "SVD rank 20" saturates its instrument (only 20 history
vectors exist — a cap, not a measurement); the kappa denominator
is an unregistered convention swinging d_eff 8x — the
"non-reconciliation" is convention-plus-cap, not physics. (c)
SUPPRESSED DISCLOSURE restored: in-span seed 11602 is ALIVE (0.66)
at rung 1 where siblings die at ~1e-4 — a ~3x threshold spread
inside the primary organism; the in-span arm's lethality is
seed-heterogeneous. (d) The un-run breaker named: magnitude-shuffle
within the front (keep top-10% support and signs, permute |g|) —
sign-pairing was convicted by intervention; magnitude-pairing is
asserted, never intervened on (queued for e193b/e194's rider)."""
t = t[:m.start(1)] + body + amend + "\n\n" + t[m.end(1):]

# 3. T143 contradiction amendment
m = re.search(r"^## T143 — .*$(.*?)(?=^## T142)", t, re.M | re.S)
assert m, "t143"
body = m.group(1).rstrip("\n")
amend = """

[R60-CRITIC AMENDMENT ~17:15Z]: "the raw gradient is the most
lethal direction" is CONTRADICTED un-amended by the chart's in-span
arm (sampled span directions kill at 0.56-0.61, BELOW the g-ray's
0.92) — the correct sentence: the g-ray is the most lethal of the
FIVE RAYS SAMPLED; the span contains directions more lethal still;
three rays are not a map. TERRAIN language scopes to biography-of-
rays until e193/e193b replicate or scramble the order."""
t = t[:m.start(1)] + body + amend + "\n\n" + t[m.end(1):]
io.open("THINKING.md", "w", encoding="utf-8").write(t)

# 4. 'worth' repair in the paper R6(b) + the abstract strike-order
p = "scratch/day6_paper_skeleton.md"
s = io.open(p, encoding="utf-8").read()
old = "autonomy worth\n   +0.07 of cycle-median"
if old not in s:
    old = "autonomy\n   worth +0.07 of cycle-median"
if old not in s:
    old = "autonomy worth +0.07 of cycle-median"
new = "autonomy reading\n   +0.07 of cycle-median (single-run; the seed ladder owed — 'worth' struck per R60-critic)"
assert old in s, "worth"
s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8").write(s)

c = "scratch/claims_ledger.md"
s = io.open(c, encoding="utf-8").read()
old = "The same physics is generative: a projection"
assert old in c or old in s, "s3"
s = s.replace(old, "The architectures came first and the physics explains them after (construction preceded explanation — not 'the same physics is generative'): a projection", 1)
old2 = "We pre-register every bar; the paper's\ncorrection chain — three rulers bent, one projection falsified by\nits own measurement — is itself evidence the dissected laws are\ncausal, not descriptive (C9)."
new2 = "We pre-register every bar and publish the correction chain\nitself — three rulers bent, one projection falsified by its own\nmeasurement — as the method's claim (verifiable process, not\nproof of causality) (C9)."
assert old2 in s, "s4"
s = s.replace(old2, new2, 1)
io.open(c, "w", encoding="utf-8").write(s)
print("R60-critic repairs applied")
