# Paper Skeleton 2 — the consolidation paper (STRATEGIST draft, 2026-09-28 ~09:18Z)

Working title: **"Consolidation follows error placement: routed and site-stored
memories in a tiny language model"**

Status: SKELETON. Claims 1-3 are evidence-complete (single-lineage caveat);
claim 4 carries W014's open caveat; the e147/e150 cells are marked and their
reading map is pre-registered (THINKING.md "READING MAP"). Blocked on:
replication seeds (e145-class), the width dose-response (e147), flat-CE
verdict (e150), dream-confound discharge (e148).

## Abstract (draft ~150 words)

We dissect memory consolidation in 0.84-2.7M-parameter char-LMs with
pre-registered interventions and deletion batteries. Four findings. (1)
A fact consolidates where its training error is placed — shown causally by
steering: error locked at positions 5-13 builds a site-store there, with no
routing despite sink adjacency. (2) Two memory types follow: SITE-STORED
(content concentrated at a row, context-general, geometry-bound) and ROUTED
(readout keyed to the omnipresent row's presence, geometry-general,
deletion-tolerant) — switched by the error's position-variance. (3) Content
never moves: all states store in body organs and fact-specific heads;
"migration" is read-policy re-routing, the destination row carrying no
written key (install-restore is a no-op; direction-scramble spares the fact
while costing the LM 0.70 nats). (4) sink-coupling and the
removable-to-irremovable movement (e150): no flat-CE fact-kill exists —
masking all attention to position 0 spares the fact at CE +0.03 while
sub-threshold row-0 norm poisons every read (threshold in (0.07, 0.15));
consolidation moves the memory's dependence from surgically-deletable
tissue into organism-critical tissue. L0H3-zero (58.6% drop at CE +0.21)
is the leading fact-circuit candidate for the surgical-unlearning surface. All corrections in the arc
were caught by pre-registered adversarial review and are reported.

## Contributions (numbered)

C1. Error-placement compass (T076/T084; e120/e131/e143) — observational then
    causal. Key exhibit: the "failed" splice arms expressed at 0.989/0.988 at
    the address the battery never read (instrument-geometry blindness, Rule 12).
C2. The type taxonomy + its switch (T082/T085; e139/e142/e143/[e147 pending]).
    Key exhibit: the ROAD→TYPE plate (fig 1). History rewrite: row-0-always,
    addresses are protocol-made grafts (e142, 13/13 nets).
C3. Content-everywhere/routes-differ (T080/T081; e133/e141). Additivity fails
    at the organ level (0.785 vs 1.98) — populations at rows/organs/routes.
    Presence-not-content routing; probe-power honesty (no-op-by-norm lesson).
C4. The memory tenant + unlearning ordering (W014/[e150 pending]; e125 design
    pre-registered: heads > route >> band, brake-trap as the naive failure).
C5. Methodology: the correction chain itself (three headline verdicts inverted
    in one session; every inversion caught by the lab's adversarial-review
    machinery; pre-registrations git-verified) — reproducible honesty.

## Figure plan

Fig 1 (THE plate): rows = {NEAR locked, FAR locked, jitter ±8, [ladder w=1..64
from e147]}, columns = {site content census, novel-geometry generalization,
D-all survival, brake sign}. e143's numbers already fill the core 3x4:
0.278/0.002/0.003/neg; 0.238/0.205/0.156/~0; 0.722/0.914/0.903/brake.
Fig 2: the flat-CE plane (fact-drop vs CE-cost scatter, every intervention,
flat-CE region shaded) — from e150; the paper's honesty centerpiece.
Fig 3: the correction chain timeline (verdict → attack → discriminator →
inversion), the C5 exhibit.

## Evidence gaps (submission blockers)

1. Single lineage everywhere — need ≥3 seeds/families for C1-C2 (e145-class:
   the e098 ladder + B43 exist on disk).
2. Width dose-response for C2's switch (e147, in flight; INVARIANCE-CAUSAL
   vs SEED-COVERAGE changes the claim from binary to parametric or adds the
   seed-reach constant).
3. Flat-CE verdict for C4 (e150; ALL-KILLS-WRECK triggers the
   removable→irremovable reframe — abstract claim 4 rewrites, not retreats).
4. Dream claim stays OUT of the paper until e148 discharges the harvest
   confound.
5. GPT-2 external-validity probe (parked; now has a sharp first question:
   row-0 presence-keying under the e141/e150 instruments).

## Risks (reviewer-kill, pre-empted)

R1 "Toy scale" — C5's methodology + the GPT-2 probe as external validity.
R2 "Confounded taxonomy" (R45 critic attack 1) — cite the bounds honestly;
  the single-net-both-types cell (P-b) is queued; run it before submission.
R3 "Known phenomenon" (attention sinks; memory types) — the novelty is the
  CAUSAL compass + presence-typed routing + the failure-mode inversion, not
  the sink's existence. Position against sink/StreamLLM and
  complementary-learning-systems literature (scratch/massaction_key_lit.md,
  W003's CLS analogy, now narrowed to replay-only).
R4 "Instrument circularity" — Rule 12 + the probe-power amendment are the
  pre-emptive answers; lead with them.
