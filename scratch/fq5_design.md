# FQ5 — THE GMAIL/IPHONE INTERIOR (design note, ripening on paper; 2026-10-04 ~12:25Z)

The fresh-questions review's joyful cut: stop asking WHAT names the
third sorting dimension (e216-e221 retired every WHAT-candidate:
height, entrenchment, tokens, compositionality-as-typology). Ask
WHERE the organism differentiates the two near-token-identical,
opposite-fate relations DURING the wash — and let the measurement
decide whether "unnamed" is the object's correct state.

## The anchor pair, restated

Gmail (holds: +0.43 residual on both washes) and iPhone (dies:
-0.21): near token-identical (both ~0/1M source frequency; frag
0.167 vs 0.143; cue 8 vs 7 tokens), same family (product), same
battery, opposite fates — and e220's best token model WIDENS the
gap. Whatever separates them is invisible to every surface feature
measured. The question moves from the FEATURE side to the
MECHANISM side: what does the wash DO differently to the two
probes' representations?

## The design (ripening, not yet frozen)

The discriminating observation on paper: if the difference is in
the RELATION's internal geometry (however composed), then the
wash's first gradients should treat the two probes' contexts
differently in a readable way — not in magnitude (the levels
decline similarly early) but in DIRECTION: the component of the
wash gradient that overlaps each probe's own support.

ARMS (all eval-only on the two-wash 124M archive + tiny CPU
forwards):
  (1) THE SUPPORT DIRECTION PER PROBE: at t=0, each probe's
context-gradient (the p(answer|context) gradient — a 124M-dim
direction); the pairwise geometry of the 54 probes' supports
(the Gmail/iPhone cos; the within-family structure).
  (2) THE WASH'S TREATMENT: the wash gradient at each state,
decomposed against EACH probe's support — the alignment
cos(g_wash, s_probe) per state. The prediction space: Gmail's
support gets progressively LESS anti-aligned (or the wash's overlap
decays faster for iPhone). The e204 convention (fact-gradient
sensitivity) extends per-probe.
  (3) THE LANDING READ: the settled +80 states' supports vs t=0 —
does Gmail's support ROTATE AWAY from the wash's hot set while
iPhone's does not? (g11's rotated-support geometry, applied
per-probe at 124M.)
  (4) THE NULL THAT MATTERS: if the two probes' geometries are
indistinguishable at every read — identical alignment curves,
identical rotation — then the differentiation happens somewhere
these instruments cannot see, and REAL-AND-UNNAMED is the correct
STATE OF THE OBJECT, not a failure to name it. That null is a
finding: it bounds the mechanism to the nonlinear/unmeasured, and
the honest sentence becomes "the third dimension lives below the
first-order instruments' floor."

Draft bars (freeze at dispatch):
  SUPPORT-DIFFERENTIATES: "fires if Gmail and iPhone separate on
the alignment or rotation curves (>= 2 sigma of the within-family
spread at any state) — the wash treats the two relations'
supports differently; the interior mapped; the dimension's seat
located even if its name stays open."
  GEOMETRY-IDENTICAL: "fires if the curves sit within the family
noise at every read — the differentiation is below the
first-order floor; REAL-AND-UNNAMED confirmed as the correct
state; the honest bound recorded."

Cost: CPU-only, minutes (124M forwards at batch 1; the archived
states loaded). Joy: high — this is the record's sharpest
unopened object, cut where it lives.

Ripening note: (2) has a subtlety — the wash gradient is a batch
aggregate; the per-probe decomposition needs the probe-context
forwards of the same batch (or the per-sample gradients, which
GPT-2 at batch-small can give). The instruments exist; the
conventions (matched-point, the e204 FD gates) port. One beat of
ripening, then freeze.
