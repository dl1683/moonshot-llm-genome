t = open("THINKING.md", encoding="utf-8").read()

# ---- timestamp repair (cards were written against a guessed clock ahead of real time) ----
repairs = [
    ("SECOND AMENDMENT (R43 critic — accepted, ~06:55Z):", "SECOND AMENDMENT (R43 critic — accepted, ~06:10Z):"),
    ("## T076 — the critic's error-location theory taken straight: consolidation follows the error — and road E is its stress test (2026-09-28 ~07:10Z)",
     "## T076 — the critic's error-location theory taken straight: consolidation follows the error — and road E is its stress test (2026-09-28 ~06:12Z)"),
    ("## W009 — WONDER: the population frame — every instrument returns overlap, and discreteness is the metaphor's artifact (2026-09-28 ~07:10Z)",
     "## W009 — WONDER: the population frame — every instrument returns overlap, and discreteness is the metaphor's artifact (2026-09-28 ~06:12Z)"),
    ("RIDER RESULT (zero compute — e120's sitting logs, arm-level proxy,\n~07:35Z):",
     "RIDER RESULT (zero compute — e120's sitting logs, arm-level proxy,\n~06:22Z):"),
]
for old, new in repairs:
    assert old in t, old[:50]
    t = t.replace(old, new, 1)

r = open("REVIEWS.md", encoding="utf-8").read()
old_r = "## Review 43 — the re-keying ambush (2026-09-28T06:55Z; covering 05:50–06:55Z; e119 dispatched mid-review, RUNNING)"
new_r = "## Review 43 — the re-keying ambush (2026-09-28T06:10Z; covering 05:50–06:10Z; e119 dispatched mid-review, RUNNING; timestamps in this window repaired 06:30Z after a clock drift)"
assert old_r in r
r = r.replace(old_r, new_r, 1)
old_r2 = """- AUDIT FINDING, CORRECTED BY ERRATUM (~07:25Z):"""
new_r2 = """- AUDIT FINDING, CORRECTED BY ERRATUM (~06:25Z):"""
assert old_r2 in r
r = r.replace(old_r2, new_r2, 1)
open("REVIEWS.md", "w", encoding="utf-8").write(r)

q = open("QUEUE.md", encoding="utf-8").read()
old_q = "DISPATCHED 06:55Z (CPU, parallel w/ e119 GPU)"
new_q = "DISPATCHED ~06:10Z (CPU, parallel w/ e119 GPU)"
assert old_q in q
q = q.replace(old_q, new_q, 1)
open("QUEUE.md", "w", encoding="utf-8").write(q)

# ---- W010 ----
w_anchor = "## W009 — WONDER:"
w010 = """## W010 — WONDER: seed-and-amplify — consolidation as amplification of existing expression, not construction from nothing (2026-09-28 ~06:30Z)

The T076 rider's trajectory savor pulled a thread: splice arms
WANDER (0.108-0.020-0.151-0.016, no trend) while jitter CLIMBS
from its first checkpoint (0.697 at step 50). Under error-location
alone, both arms place error on the fact; the difference is WHERE
the error lands relative to EXISTING EXPRESSION. Jitter re-teaches
inside install windows the net already expresses — error lands on
SEEDED supports and amplifies them (expression -> captured error ->
growth: positive feedback). The splice teaches at row 183 where
nothing is readable — error lands on a seedless site, and each
batch's differing contexts pull the row in different directions:
wandering. FIVE observations, one mechanism: (1) jitter climbs
(seeded band); (2) locked replay partly works (+0.221: the one
original seed, amplified); (3) splice at seedless 183 fails and
wanders; (4) dreams carry the fact but consolidate nothing —
carried content at novel positions = seeds WITHOUT error; (5) road
E (deletion) destroys the seed and the net regrows from the
population's residual overlap (e088's redundancy) — necessity as
the road when amplification has nothing to amplify. The mechanism
reframes the ingredient question: position diversity was never the
cause — it is HOW ERROR FINDS ALL THE SEEDS. PREDICTED SAVORS
(falsifiers in disguise): (a) far-jitter at ±64 — seedless
positions — should WANDER like splice, not climb (the critic's
missing control, now with a mechanism behind it); (b) a tiny
pre-seed at 183 (mini-install) followed by the identical splice
fine-tune should CLIMB — turning e120's failure into e138's
head-start done surgically; (c) climb onset should track seed
strength (pre-seed dose vs steps-to-liftoff, monotone); (d) road
E's regrowth rate should track residual overlap mass after erase
(e083's cycle-weakening is the downward arm). If (a) climbs
anyway, seed-and-amplify dies and pure error-location stands; if
(b) still wanders, the seed must be BAND-MEMBERSHIP, not
expression-anywhere — a sharper noun than either card has.

""" + w_anchor
assert w_anchor in t
t = t.replace(w_anchor, w010, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)
print("W010 in; timestamps repaired")
