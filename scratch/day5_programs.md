# Day-5 Research Programs — IDEATOR, 2026-09-26 late (CPU-only thinking)

Inputs: THINKING.md T037–T048 (day-4 set, all cards closed), QUEUE.md,
scratch/post_paper_programs.md, DAY_FOUR_REPORT.md, checkpoint inventory
(runs/checkpoints/). Premise: day 4 closed its arc — the address is a single
portable row (in-seed), junk is self-generated and dynamically load-bearing,
the crossmatch instrument exists (cos ≥ 0.4459, OOS AUC 0.919), the anchor is
run-specific trajectory content, the template is a fast-training emergent.
Day 5 converts closures into the three programs the closures opened, plus one
registered prediction that must not be left sitting.

**Envelope discipline:** everything runs on existing checkpoints (2.7M/0.84M
families, within the ≤10M preferred band). Exactly TWO sub-150-step training
touches on existing nets are requested (the B43 install, the cycle-3 re-learn),
each minutes-scale — this is the day's entire thermal budget; every other
proposal is eval-only CPU. If the GPU stays thermally constrained, e081 GATE 0
and e082 can wait without invalidating anything; e083/e084 run regardless.

**Decision summary (ranked):**
1. **e081 — P1 phase-2: the cross-seed row-129 transplant** ("a one-row
   organ"). The named next-tier item; dimension-matched assets exist TODAY
   (e048_repro seed-42 install ↔ e028_b43 seed-43 base, same 6L/6H/192 cfg,
   same corpus). One 100-step install touch, then all eval-only.
2. **e082 — canalization cycle 3** (T037 #2's REGISTERED falsifiable
   prediction, still untested). Cheapest decisive test on the table; the lab's
   own promotion policy (registered discriminators outrank new lines) puts it
   above the two new-program openers.
3. **e083 — READ POLICY opener: the argmax-flip census.** T037 #5's unified
   question measured directly for the first time (the rule, not its shadows).
   Eval-only CPU; biggest conceptual upside; new instrument (build cost is the
   main risk).
4. **e084 — P3 anchor-description probe** (T048's pivot from replacement to
   DESCRIPTION). Eval-only CPU on the e075/e080 rig; carries T037-a's
   registered within-net correlation as a free rider.

Free riders and parked items listed at the end. e065 (RMU-vs-surgery,
critic-hardened) stays READY-GATED unchanged — not re-proposed.

---

## 1. e081 — P1 phase-2: the cross-seed row-129 transplant ("a one-row organ")

### Organizing question

Day 4 established, within one seed: the installed fact's address is wpe-129
alone — necessary (knife-edge), ~70% sufficient for rebinding (e068/e078,
n=2 installs), surgically targeted in place (KL 0.2–0.5, vs row-0's
distribution-wide 2.6–7.2). T038 says the read circuit keys on the coordinate;
T024/T027/T046 say the stream basis is init-private and selection cannot move
it. These two findings collide on one object: **the address row's CODE is
seed-specific (grown in the host's init basis), but its FUNCTION is
coordinate-keyed. Does the row itself transplant across seeds?** This is also
the P1↔P2 bridge named in post_paper_programs.md ("a one-row organ — the
minimal graft") and the natural first customer for the crossmatch instrument
(T046 registered prediction: tolerance/compatibility should be VISIBLE to the
cosine rule; here we get it for a one-row graft, zero training).

### Why the assets make this cheap NOW

- Donor: `runs/checkpoints/e048_repro.pt` (seed-42 install, 6L/6H/192, block
  256) — the exact net behind T042/T043/T068, with its install-60/held-30
  batteries rebuildable bit-exact (e068/e071 machinery, protocol seeds 1337 /
  E43.SPLICE_RNG / 202).
- Host base: `runs/checkpoints/e028_b43.pt` (B43, seed 43, SAME config, parity
  baseline) — dimension-matched, same corpus/vocab. Row copy is meaningful
  (192-d rows both sides). No base training needed; the only new training is
  the install exposure itself (e043 protocol: 100 steps, resumable, minutes).
- Note on the ≤1M preference: a fresh 0.84M seed-43 base would cost a FULL
  parity train (~minutes at batch 32 but a new net to validate), strictly more
  thermal budget than 100 install steps on the existing B43. B43 is the
  minimum-cost honest host; the 2.7M family is inside the lab envelope.

### Full registration (frozen before any run)

**GATE 0 — host install (the only training touch):** run e043's exposure
verbatim on B43 (same windows, same splice RNG, same name targets, same LR /
step count) → `e081_b43_install.pt`. Pass bars: install-60 p(Z) within the
e043 gate family; held-30 recorded. If the install does not take within 2×
e043's steps-to-bar (protocol fragility across seeds — a KNOWN risk, T015),
PARK the experiment; do not ad-hoc tune the protocol (a tuning study would be
its own registered experiment).

**GATE 1 — seed-43 address census mini (eval-only, CPU):** perturbation
battery over wpe rows {0, 1, 122–135, 250–255} (row←mean + zero for top rows)
on the B43 install × install-60. Pass bar: the bimodal 0/129 structure
replicates (129 in the top-3 rows by p(Z) drop). If seed-43's install binds a
DIFFERENT row (protocol-geometry drift), re-target every arm below to the
host's own top row — the registration covers that branch explicitly ("host
argmax row" replaces 129 wherever it appears).

**MAIN — six arms, all eval-only, one script (CPU), batteries install-60 +
held-30, readouts p(Z) + KL + Z-rank:**

- **A0 host-intact** — reference.
- **A1 host-own-row destruction** (zero + mean, in place) — the seed-43
  knife-edge replication (e066b analogue). Calibrates the host's own row
  weight.
- **A2 DONOR OVERWRITE** — `wpe[129]_B43 ← wpe[129]_e048_repro`. The one-row
  organ transplant. THE primary arm.
- **A3 donor adjacent-slot add** — `wpe[130] ← donor row` (host row intact).
  Tests whether the circuit accepts a foreign code at a neighbor coordinate
  (the e068 shifted-window analogue, cross-seed).
- **A4 sham surgery** — delete host row, re-copy host row. Surgery-damage
  control; calibrates the no-copy plateau (~0.13–0.15 in-seed).
- **A5 donor-family mean control** — donor row ← mean of donor's wpe rows.
  Separates "specific foreign code" from "any big row change".

**Crossmatch overlay (T046 bridge, computed pre-graft on the SAME graft-input
depths):** stream-cosine for the A2/A3 one-row grafts; apply the 0.4459 rule.
T046's registered prediction extends: the rule's admit/reject verdict should
match the transplant outcome direction. Divergence (rule admits, transplant
fails; or rule rejects, transplant works) is itself registered as informative
— instrument-vs-mechanism separation at the minimal graft.

**FREE RIDER (T043's open discriminator, zero marginal cost):** at any
rebound/plateau geometry encountered, record Z's rank. Plateau-with-rank-2
⇒ sub-argmax residue (b); plateau-with-rank>5 ⇒ content-row co-adaptation
(a). Closes T043's registered tension for free.

### Hypotheses

- **H-organ (portable):** the row is a self-contained address organ; the
  circuit reads the coordinate and accepts any sufficiently installed code.
  Predicts A2 restores expression at ≥ half the in-seed rebind.
- **H-basis-private (P2-frozen extension):** the read circuit co-adapts to the
  host's OWN row code during install; a foreign code at the same coordinate is
  unreadable. Predicts A2 ≤ the A4 plateau AND crossmatch cosine low (rule
  correctly rejects).
- **H-coordinate-plastic (middle):** partial transfer — A2 strictly between
  plateau and half-rebind, or A3 (add, host row present) works while A2
  (overwrite) fails: the circuit needs host-code presence but tolerates a
  foreign second key.

### Registered prediction

Primary: **A2 p(Z) < 0.30 on install-60 ⇒ basis-private; ≥ 0.30 ⇒ portable.**
(0.30 = half the 0.60–0.71 in-seed rebind band.) Secondary: the crossmatch
verdict matches direction. Tertiary: A1 drops p(Z) by ≥ half (knife-edge
replicates at seed 43).

### Kill criteria

- GATE 0 fails → park (protocol-fragility study becomes its own queue item).
- GATE 1 shows no single-row concentration at seed 43 → the single-row
  address is an in-seed coincidence; P1 phase-2 HALTS, re-planned around the
  census object; claim-B wording in paper/README gets an explicit seed-42
  scope flag.
- A2 ≈ A4 (donor indistinguishable from sham) → the one-row-organ question
  closes negative at n=1 pair; register the P2 implication (even the minimal
  graft inherits init privacy) and STOP — do not iterate donors.

---

## 2. e082 — canalization cycle 3 (T037 #2's registered prediction)

### Why this outranks the new-program openers

T037 registered, in writing, a falsifiable prediction that has not been run:
"a third erase/re-learn cycle is SLOWER and more surgical-proof than the
second — monotone closure, never oscillation." The lab's promotion policy
says registered discriminators of live hypotheses outrank new topics; this is
THE registered discriminator of the synthesis card, the cheapest on the table
(~75–100 training steps total on an existing 2.7M net), and it gates how much
prior the READ-POLICY program is allowed to import (the "history = its canal"
faculty). Cycle 2 is n=2 across seeds (e044: 2.08×; e044b: 2.92× slower, cos
0.728/0.760). Nobody has measured cycle 3.

### Hypotheses

- **H-canalization (monotone closure):** cycle-3 re-learn is at least as slow
  as cycle 2 (ratio ≥ 1), the regrown row returns toward the ORIGINAL
  groove (cos ≥ 0.6), and targeted re-erasure resistance is non-decreasing.
- **H-groove-only (weaker):** speed oscillates or flatlines but the cos stays
  high — the groove persists without monotone closure.
- **H-scar-saturation (anti-canalization):** cycle 3 is FASTER (ratio < 0.8)
  — the groove primes re-learning rather than resisting it (a real,
  constructive plasticity finding, opposite sign).

### Cheapest discriminating experiment

Continue e044b's B43 line: if the cycle-2 re-learned checkpoint was not saved,
rebuild it first (erase → re-expose, ~35 steps, e044b machinery verbatim,
protocol-identity gate bit-exact). Then: (a) re-erase the re-learned J-rows
(D2 row-zero, same surgery), re-expose → steps-to-bar + cos(relearned_J,
orig_J); (b) fresh-install control arm in parallel (12–35 steps) for the
denominator; (c) surgical-proofness readout: steps-to-bar after a SECOND
half-dose targeted erase vs the naive erase. Total ≤ ~100 steps.

### Registered prediction

steps ratio (cycle-3 / cycle-2) ≥ 1 AND cos ≥ 0.6 on both arms' tasking bars
(BAR_NLL 1.0 / BAR_ACC 0.8). Anti-canalization signature: ratio < 0.8 OR
cos < 0.5 (oscillation to a new solution).

### Kill criterion

If cycle 3 is faster (ratio < 0.8) or the row re-bases (cos < 0.5),
CANALIZATION-as-monotone-closure is falsified in its first direct test: T037
#2 is downgraded to "groove persistence only" in THINKING/paper wording, and
the read-policy program drops the canal prior. If it passes, register cycle 4
as the extrapolation probe (predicted: diminishing increments, asymptotic
plateau — the closure curve's shape becomes a quotable law).

---

## 3. e083 — READ POLICY opener: the argmax-flip census (the rule, not its shadow)

### The program

T037 #5: every killed hypothesis was categorical, every survivor graded, and
the one component never directly edited is **the per-position rule deciding
which stored coordinate is opened and which candidate wins argmax.** The four
edit-law faculties are its shadow. Day-4 sharpened why the shadows are
shadows: dCE is continuous + teacher-forced + single-read (and T048 proved
static utility does not predict dynamic prunability); free-run is honest but
uncontrolled. The rule itself is a DISCRETE function of the cache state:
decision(cache) → argmax. First direct measurement: perturb one stored
coordinate at a time at a decision point and census the ARGMAX — not the
logit wiggle.

### Design (eval-only, CPU)

Net: `e053c_ctx512.pt` (the window-decided 0.84M-class net, a* = 6 CI [4,8]).
Trajectories: the e053c free-run battery (cached or regenerated greedily,
CPU minutes). Sample N = 200 decision points, MARGIN-STRATIFIED (the honest
confound — decisions with margin below the perturbation scale must be
sampled deliberately, and margin recorded as a covariate). Per decision, for
each cache entry (all 512 where feasible; stratified subsample otherwise):

- Perturbation battery: V-zero, K-drop, V-counterfactual (content swapped in
  from a same-age entry of a different run), each at matched magnitudes
  {0.5, 1, 2}× unit (e011c matched-magnitude discipline).
- Readouts: flip / no-flip; if flip — target token, was target the
  pre-perturbation runner-up, a fixed attractor token, or the
  counterfactual's implied continuation (content-following)?
- Plus the redundancy probe on a 50-decision subsample: pairwise
  perturbations of the two most-load-bearing entries — single-flip rarity
  with pair-flip frequency = the rule's redundancy code.

**Deliverable — the READ KERNEL:** per-decision set of opened coordinates
(flip-causing), its age profile, redundancy structure, and the flip-target
taxonomy. This is the object T037 #5 names, made measurable.

### Hypotheses

- **H-sparse-open:** each decision opens ≤ ~6 recent entries + a handful of
  specific old ones; flip targets are mostly the runner-up (shallow rule).
- **H-dense-averaging:** singles almost never flip; only multi-perturbations
  do — the rule is a statistic over many weak reads, not an address fetch.
- **H-content-following vs H-statistics-only:** V-counterfactual flips the
  decision TOWARD the counterfactual's continuation at opened coordinates
  (content read) vs flips that are target-stable whatever content is written
  (mass/statistics read) — T044's K/V dissociation generalized to the
  decision level.

### Registered prediction

Flip-rate(age) correlates ≥ 0.8 with dCE-load(age) on the same battery (the
rule's opening profile reproduces the a* ≈ 6 spike). **The informative
failure:** if flip-profile and dCE-profile dissociate, the lesion shadow was
misreporting the rule — a headline, and the census becomes the primary
instrument for the whole read-policy program.

### Kill criterion

If, at magnitudes ≤ 2× matched-norm, single-coordinate flips occur in < 0.1%
of (decision × entry × type) cells across the margin-stratified battery, the
discrete census has no dynamic range at char scale: park the program until a
decision-point selector with margin floors is built (registered as
instrument-debt, not a finding). Do NOT rescue by raising magnitudes into
model-breaking territory (that measures damage, not the rule).

---

## 4. e084 — P3 anchor-description probe (T048's pivot)

### The question

T048/e080 closed replacement: deletion, norm-matched noise, and prompt-copy
all fail (prompt-copy recovers only 3/4 and still free-runs off-manifold);
the anchor is RUN-SPECIFIC trajectory content, and P3 pivoted from
replacement to DESCRIPTION. What measurable property distinguishes anchor
entries (the age>96 entries whose removal cost +0.26 nats and flipped the run
into the attractor) from non-anchor entries at the same age?

### Candidate properties (computed BEFORE pruning, from the cached e075/e080
trajectories, B = 8)

- **P1 geometric:** entry V-vector projection on the run's own cache-PCA
  (top-8) / distance to the run's trajectory mean.
- **P2 readership:** accumulated attention-received mass over the trailing
  window (how often the run re-read the entry) + re-read trend (rising /
  falling).
- **P3 predictive:** clean-net surprisal of the entry's token in context;
  self-surprisal kept as the LIAR control (free-run-honesty principle
  predicts it fails — its failure is a positive control for the witness).
- **P4 drift-correlated:** alignment of the entry's V with the free-run drift
  direction (free-run minus teacher-forced state delta).
- **P5 age** — the null every other property must beat (partial r).

### Cheapest discriminating experiment

Jackknife removal cost on a stratified sample (~100 pruned-band entries age >
96, ~50 young controls): static cost = teacher-forced single reads (CPU
minutes); dynamic cost on a 30-entry subsample = free-run continuation with
the clean-judge at horizon +8 (T048's only reliable witness — the static arm
is EXPECTED to mispredict, and its failure rate is itself the T048 replication
at entry granularity). Regress cost on P1–P5, partial on age.

### Hypotheses

- **H-readership:** P2 wins or ties (partial r ≥ 0.35, beats age) — the
  anchor is "where the run has been" = what it kept re-reading.
- **H-manifold:** P1 wins — anchor = trajectory-manifold membership (prompt
  content, sharing corpus statistics, partially overlaps the manifold —
  exactly the observed 3/4 recovery).
- **H-drift:** P4 wins — anchor = drift-aligned content.

### Registered prediction

P2 (readership) ≥ 0.35 partial r and ≥ age's; P3-self fails (positive
control). FREE RIDER (T037-a's registered within-net prediction): per-position
far-context value vs old-entry junk-frac correlate POSITIVELY in this net; a
negative correlation revives the conflict T037-a declared pseudo.

### Kill criterion

All of P1/P2/P4 ≤ 0.15 partial r on BOTH static and dynamic cost ⇒ the anchor
is not entry-describable: P3's description leg pivots to the sequence level
(anchor-as-whole-run-object; the cross-run entry-subset exchange from e080's
template) or closes with the taxonomy + provenance law as its legacy. Either
branch is publishable wording; no further pruning-exploit work (e075 killed
it — that stays dead).

---

## Free riders, parked items, and what is deliberately NOT proposed

- **Free with e081:** T043's Z-rank discriminator (rebound geometry); the
  crossmatch rule's first one-row-graft verdict.
- **Free with e084:** T037-a's within-net positive-correlation prediction.
- **Parked (GPU-gated, unchanged):** e065 RMU-vs-surgery head-to-head
  (critic-hardened design 5f11a79 — resumes when the GPU frees; day 5 does
  not touch it); T046's trained-tolerance cosine-raise prediction (P2
  phase-2 proper — needs fine-tunes; the crossmatch grid viz v014 script
  already exists and can run on CPU any time).
- **Deliberately not proposed:** the GPT-2 anchor external check (GPU; the
  single-net-family caveat stays flagged); the context-rot cliff link (P1
  phase 3 — needs a ctx-512 KV-recall train; wrong thermal day); any new
  pruning intervention (e075's kill stands — T048's principle forbids acting
  on static-utility logic).
- **Honesty note on n's:** e081 is n=1 donor-pair by design (the kill
  criteria prevent donor-shopping); e082 rides an n=2 base; e083/e084 are
  instrument-first runs whose B=8/N=200 samples carry the same B-fragility
  flags day 4 learned (e072: a* 18→7 at B=16) — both register bootstrap CIs
  from the start.

## Sequencing (one possible day)

e083 build (CPU, instrument) and e081 GATE 0 (one GPU touch) can start in
parallel; e082's ~100 steps when the thermal window allows; e084 last (its
inputs are fully cached, zero urgency). Every experiment gets its THINKING
card BEFORE the next starts (Rule 0), and e081's GATE failures park rather
than pivot — day 5's discipline is protecting closed cards, not reopening
them.
