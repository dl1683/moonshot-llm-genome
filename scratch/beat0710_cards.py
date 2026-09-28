import json

# ---------- THINKING.md: T076 + W009 ----------
t = open("THINKING.md", encoding="utf-8").read()

t_anchor = "## T075 — E120:"
t076 = """## T076 — the critic's error-location theory taken straight: consolidation follows the error — and road E is its stress test (2026-09-28 ~07:10Z)

The R43 critic meant ERROR-LOCATION as an attack on T075; taken
straight it is the lab's first unifying theory of WHERE
consolidation happens: the fact moves to wherever the fine-tune's
error on the fact is placed. Jitter distributes fact-error across
the 121–137 band -> overlapping supports grow there (and the
battery, reading that band, sees consolidation). Splice
concentrates fact-error at row 183 -> an install grows there,
unread by our instruments (e131 probe 1 will read it). Dreams
carry near-zero fact-error (the net already predicts its own name
— that IS the 500x decay-slowing) -> nothing consolidates. All
three E120/E121 arms fall under one compass.

ALTERNATIVE EXPLANATIONS: (1) ROAD E BREAKS IT — deletion
pressure consolidates with NO fact-error anywhere (erase cycles +
corpus fine-tune; the corpus never contains the fact, so fact
error is zero at every position). If road E truly graduates the
fact to D-all-surviving expression (T073's reading), then
error-location is sufficient-not-necessary, and the deeper
variable is ERROR-OR-NECESSITY: consolidation happens where the
fact is either re-learned (error placed) or newly REQUIRED
(deletion makes surviving machinery carry it). (2) The
locked-replay partial road (rescue_b +0.221) fits error-location
loosely — error at one address grows one support — so it does not
discriminate. (3) The corpus>self gap: under error-location,
corpus-contexts beat self-contexts simply because higher-loss
contexts place more fact-error at 183 — CHECKABLE from e120's own
logs (per-context loss vs per-context consolidation), zero new
compute.

DISCRIMINATING OBSERVATION: e119's head-to-head battery, already
running, carries road E's grown-row census and D-all — if E shows
band growth WITHOUT fact-error, error-location dies as a
universal and error-or-necessity inherits; if E shows a different
anatomy entirely (DIFFERENT-STORES), error-location survives for
road R only. REGISTERED PREDICTION (zero-cost, rides existing
runs): within road-R arms, final band-support mass rank-orders
with fact-error mass placement across jitter/locked/splice cells
(Spearman >= 0.8 over the cells e119+e131 regenerate); and the
e120 log regression — per-context consolidation vs per-context
loss — has slope > 0, predicting the corpus>self gap is
loss-texture, not self-vs-corpus identity.

""" + t_anchor
assert t_anchor in t, "T anchor"
t = t.replace(t_anchor, t076, 1)

w_anchor = "## W008 — WONDER:"
w009 = """## W009 — WONDER: the population frame — every instrument returns overlap, and discreteness is the metaphor's artifact (2026-09-28 ~07:10Z)

Three instruments in three languages said the same thing this
week. e088's pair-anchor factorial came back SUB-additive (median
pair/(s1+s2) = 0.464 — single removals already carry most of the
pair's cost: overlapping supports, not discrete slots); e089's
dose-response was MASS-ACTION confirmed (threshold-shaped, not
circuit-switched); e113's deletion hierarchy is graded (single
grown rows cheap, all-five fatal) — and e088 shows WHY: the
supports overlap. Even the LN-share magnitude floor reads as a
population statistic: a floor is what a redundant population's
renormalized total contribution looks like. The lab's whole
spatial vocabulary — home, address, migration, tenant, brake —
smuggles in discreteness; the data keep answering with mass.
SAVOR: if the field is a POPULATION (overlapping, redundant,
mass-action), then "where is the fact?" is a category error — the
right question is "what fraction of the population does any read
draw on?", which is exactly what r*(k)·k prices, and WHY it is a
within-net constant: the population's share, not a slot's. If
W009 is right, the critic's row-0 question dissolves differently:
row 0 would not be a new HOME but the population's densest
overlap — the attention sink as grand central. e131's census
already discriminates W009 vs re-keying for free: the content-
projection histogram over 512 wpe rows should be UNIMODAL-DIFFUSE
(many small contributions) for a population, and show a SHARP
OUT-OF-BAND MODE for a re-keyed address. Registered savor, not a
bar: read the histogram's shape before reading any single row.

""" + w_anchor
assert w_anchor in t, "W anchor"
t = t.replace(w_anchor, w009, 1)
open("THINKING.md", "w", encoding="utf-8").write(t)

# ---------- QUEUE.md: write R43 honesty bars into the four rows ----------
q = open("QUEUE.md", encoding="utf-8").read()
fixes = [
 ("| e122 | self-at-distance (P6) | READY | does the anchor accept the same net's field from another run/window — generator-self vs episode-self |",
  "| e122 | self-at-distance (P6) | READY | does the anchor accept the same net's field from another run/window — generator-self vs episode-self. [R43 bar pre-committed: same-run-different-window MUST stay healthy; if it also collapses, the self/episode framing dies — the anchor is content-specific — and the card must say so] |"),
 ("| e123 | self-drift curve (P6) | READY | identity half-life across checkpoints; doubles as P5's rekeying probe |",
  "| e123 | self-drift curve (P6) | READY | identity half-life across checkpoints; doubles as P5's rekeying probe. [R43 bar pre-committed: anchor-acceptance(t, delta) must decay DIFFERENTLY (slower or different shape) than trivial output-similarity JS between checkpoint samples — rule 7a control; if they coincide the card reads 'identity drift = behavior drift'] |"),
 ("| e125 | attack the graduated fact (P7) | READY | what removes a field-stored fact — segregated ~54 units or woven into self? |",
  "| e125 | attack the graduated fact (P7) | READY | what removes a field-stored fact — segregated ~54 units or woven into self? [R43 bar pre-committed: collateral-matched specificity required — compare removability against the PRE-consolidation fact's removability (e043 asymmetry) at matched collateral damage; otherwise it is a ceiling measurement] |"),
 ("| e128 | inversion census (P8) | READY | dp27 promoted to census — do read-rule inversions cluster into a second mode? per-net rate as a fingerprint |",
  "| e128 | inversion census (P8) | READY | dp27 promoted to census — do read-rule inversions cluster into a second mode? per-net rate as a fingerprint. [R43 bar pre-committed: e095 Monte-Carlo null guard on apparent clustering — a diffuse distribution reads as 'a second mode' by eye] |"),
]
for old, new in fixes:
    assert old in q, old[:40]
    q = q.replace(old, new, 1)
open("QUEUE.md", "w", encoding="utf-8").write(q)
print("T076 + W009 in; 4 queue bars written")
