t = open("THINKING.md", encoding="utf-8").read()
o = """loss-texture, not self-vs-corpus identity.
"""
n = """loss-texture, not self-vs-corpus identity.

RIDER RESULT (zero compute — e120's sitting logs, arm-level proxy,
~07:35Z): LOSS-TEXTURE SUPPORTED, weakly. ce_r trajectories: arm b
(corpus) ran consistently ABOVE arm a (self) through the early
window — 1.7996 vs 1.7763 at step 50, 1.7578 vs 1.7369 at 100,
1.7496 vs 1.7004 at 150, crossing only at ~200 — i.e. the same
spliced fact carried MORE error in foreign surroundings (the net
predicts its own dreams; name-rich filler makes the fact more
predictable), and b consolidated ~4x more post-D-all (0.098 vs
0.023). The exposure asymmetry agrees: arm a carried 154 ZEPHYRA
(34 residual in filler) vs b's 120 clean — the arm with MORE
name exposure consolidated LESS, counter to any signal-mass
account, as error-location predicts. TRAJECTORY TEXTURE (savor):
splice arms' fact-expression WANDERS (a: 0.108-0.020-0.151-0.049-
0.182-0.016 — no trend; the critic's within-run range, now seen
as shape not noise) while jitter CLIMBS monotonically to 0.735 —
the roads differ in whether a learning trajectory on the fact
exists AT ALL, not just in endpoint. HONEST BOUNDS: n=1 per arm;
ce_r is total exposure-masked CE, not fact-span-only error; the
registered per-context regression was not computable (per-context
loss not logged) — this is the arm-level proxy. The decisive
version rides e131's regenerated arms: log per-context fact-span
loss, rank-correlate with per-context consolidation.
"""
assert o in t, "T076 rider anchor"
t = t.replace(o, n, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)
print("T076 rider in")
