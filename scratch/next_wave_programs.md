# Next-Wave Programs — IDEATOR, 2026-09-26 evening (day-5 arcs closed → day-6)

Inputs: THINKING.md T049–T053 (all five cards closed), NOTES.md E081–E091,
QUEUE.md (day-5 table + parking lot), scratch/day5_programs.md (format
precedent), checkpoint inventory (runs/checkpoints/), rig inventory
(lab/e065, e082, e084–e091). Premise: day 5 closed its arcs — reads are pure
at the fact level (string-level induction only, e087); the read kernel equals
its CE-shadow with a flagged tail (e084); the trajectory anchor is MASS-ACTION
with a threshold dose-response law (e085/e088/e089); the address is
basis-private — structure universal, code orthogonal, crossmatch
instrument validated on a live one-row rejection (e082); RMU seals the
READOUT GATE — d5 knowledge trace survives, reception closed, expression
geometry dead (e065/e091).

**Envelope discipline:** everything below rides existing checkpoints and
existing rigs. Exactly TWO training touches are requested, both minutes-scale
(e083's ~100 cycle steps; e094's ≤200 tolerance steps — CPU-feasible per the
e065 `rmu_finetune_cpu` precedent, GPU optional). Everything else is eval-only
CPU. No new base nets. Numbering: e092–e096 new; e083 keeps its assigned slot.

**Decision summary (ranked):**
1. **e092 — READOUT-GATE CENSUS** (WHERE is the gate). Component-swap
   localization on the RMU net: restore-into-RMU (necessity) and
   poison-into-intact (sufficiency), block → head granularity. Eval-only CPU;
   completes T052's mechanism table with its missing WHERE row.
2. **e095 — T050's registered texture probes** (H-tail, H-donor-voice).
   Registered in THINKING.md before any of this was proposed; analysis-grade
   on the e084 cells; the lab's own policy (registered discriminators outrank
   new lines) puts it above the named newcomers.
3. **e083 — canalization cycle 3** (T037's REGISTERED third-cycle prediction,
   sitting READY since day 4). Cheapest decisive test on the table; discharge
   the debt before it rots.
4. **e093 — THRESHOLD-LAW GENERALITY** (fraction vs absolute count across
   window/scale). Rides existing checkpoints through the e089 rig; upgrades a
   single-net curve to a law or kills the word "law". Absorbs parked e090
   (dose-vs-schedule) as an optional leg.
5. **e094 — ONE-ROW TOLERANCE (key recoding)** — the immunology question.
   The wave's one conceptual gamble; one small training touch (flagged);
   decisive routed-around-vs-recoded control; crossmatch accept-side rider.
6. **e096 — COHERENCE-GAP DOSE LADDER** (P-B, sharpened by T051). Eval-only
   on the e080 rig; connects the threshold law to continuous corruption;
   closes the anchor program's last open promise.

Not re-proposed: e065 follow-on entropy drift (flag stands, descriptive);
v013+ visualizations (v-thread parked by policy); e086 (dead).

---

## 1. e092 — the READOUT-GATE CENSUS: where does RMU seal reception?

### Organizing question

T052/e091 established H-READOUT-GATE: the RMU net refuses even good
no-removal states (reverse-rescue ~45× under bar at EVERY depth d0–d6), while
its own d5 state still carries a half-strength trace (0.362 into the intact
net) and retain-only leaves reception open (0.345). The splicing failed at
every entry depth — so the sealing is either (a) LOCALIZED: one component
downstream of (or between) the splice points destroys the address signal in
transit (late-layer attn, final LN, readout path — note wte/lm_head are
untouched at cos +1.00, so the suspects are the transformer blocks and the
final LN), or (b) DISTRIBUTED: every layer's RMU-shifted transform degrades
the trace additively, no single bottleneck. The mechanism table's four
signatures lack their WHERE row. Nobody has localized an unlearning gate at
component granularity on the same net where the store demonstrably survives.

### Cheapest discriminating experiment (eval-only CPU, no training)

Rebuild the RMU and retain-only nets bit-exact (e091's `rmu_finetune_cpu`
recipe, minutes) alongside the no-removal intact net (e048_repro). Then run
TWO directional sweeps with pure weight swaps:

- **RESTORE (necessity):** into the RMU net, replace one component at a time
  with the intact net's counterpart — 12 blocks (attn-L0..L5, MLP-L0..L5),
  the final LN, and (bonus suspect) the wpe-129 row — and re-measure the
  e091 reverse-rescue (no-removal donor state at d5, net0 onset sites,
  shuffled-donor controls, site bootstrap).
- **POISON (sufficiency):** the mirror image — one RMU component into the
  intact net, same readout. If a single component carries the sealing, the
  poison direction reproduces it in an otherwise-open net.
- **Refinement pass:** head-level swaps (36 heads) inside any block that
  crosses a bar; d0–d6 rescue sweep on the winner.

Bars (frozen, anchored to e091's own numbers): RESTORE fires = site-mean
rescue ≥ 0.17 (half of retain-only's 0.345), shuffled ≤ 0.05, CI excluding 0;
POISON fires = rescue drops ≥ 50% of the intact-net level with CI excluding 0.
Controls: full-intact and full-RMU swaps bracket both ends (0.345 / 0.007).

### Registered prediction

H-gate-localized: a SINGLE late-layer component (the L4/L5 attention family
— the same depth band where e055 found the address rescue and where RMU's u
vector lived at d4/d5) crosses BOTH bars (necessary AND sufficient); restoring
it recovers ≥ 50% of retain-only reception. H-gate-distributed predicts no
single component crosses while the full-swap brackets pass.

### Kill criterion

Both directions fail at block AND head granularity (no RESTORE ≥ 0.17, no
POISON ≥ 50% drop, brackets pass) ⇒ the gate is distributed sealing; the
census closes, the mechanism table keeps "reception sealed (distributed)",
and no further localization is proposed at this scale.

### Cost / flags

Eval-only CPU. ~30 swap arms × the e091 sweep (e091 ran in minutes); the only
overhead is the two bit-exact replica rebuilds. No GPU needed.

---

## 2. e095 — T050's registered open textures: the escape tail and the donor voice

### Organizing question

T050 registered two probes verbatim: **H-tail** — the 24–27% of flips that
ESCAPE the top-5: concentrated on specific destination identities/coordinates
(a readable second channel) or diffuse noise? **H-donor-voice** — the 48
V-swap donor-continuation hits (8.8% vs 2.0% chance): do they cluster on
decisions where the run and donor diverge stylistically (the anchor's
complement — donor trajectory content leaking through)? Both flags currently
ride the read-kernel card as caveats; the lab's policy says registered
discriminators run before new lines, and these are day-5 registrations.

### Cheapest discriminating experiment

Cell-level destination identities (t1/t2/top5/class + donor_pick) are
recorded per cell in the e084 run; where the saved aggregates fall short,
re-run the deterministic kernel on the same seeds/battery (274 s wall,
documented). Then two pure statistics:

- **Escape census:** top-10 destination identities' share of outside-top-5
  escapes vs the uniform expectation under the battery's marginal
  next-token distribution (bootstrap over the 24,000 cells for CI).
- **Divergence clustering:** around each donor-continuation hit, compute a
  run-vs-donor divergence measure (position-wise mismatch / donor-CE under
  run context in a ±32 window); compare hits to matched non-hit decisions
  (same age band, same type arm).

### Registered prediction

H-tail-structured: top-10 escape destinations cover ≥ 3× their uniform
expectation (CI excluding null) — the tail has identity structure. H-donor-
voice-clustered: hits' divergence percentile median ≥ 70th vs matched
non-hits — the donor's trajectory content is readable exactly where the run
leaves it.

### Kill criterion

Both nulls (shares ≈ uniform, percentile ≈ 50) ⇒ the two flags close as
descriptive noise; the read-kernel card stands final with the
tail-misreport flag as a permanent caveat, and no tail model is proposed.

### Cost / flags

Analysis-to-cheap-eval, CPU, minutes. Zero new instrument risk (rides e084).

---

## 3. e083 — canalization cycle 3 (T037's registered prediction, unrun)

### Organizing question

T037 registered a falsifiable third-cycle prediction that has sat READY
through all of day 5: across erase/re-learn cycles, does the install
CONSOLIDATE monotonically (canalization — each re-learn faster/straighter
back to the same basin) or not? The lab's promotion policy is explicit that
registered predictions outrank new lines; this is the oldest unrun one.

### Cheapest discriminating experiment

Exactly as registered in QUEUE.md: 3 erase/re-learn cycles on the install
net, ≤ 180 s/arm, ~100 steps total. Readouts per cycle: steps-to-bar,
ΔW ratio vs cycle 1, and cos(ΔW_cycle_i, ΔW_cycle_1).

### Registered prediction (frozen at T037, unchanged)

Monotone closure: ratio ≥ 1 AND cos ≥ 0.6 on cycle 3 ⇒ canalization stands.
Otherwise canalization is falsified — its second haircut — and T037's
consolidation language is downgraded.

### Kill criterion

ratio < 1 or cos < 0.6 ⇒ canalization dead at this scale; close the thread,
no cycle-4 proposal.

### Cost / flags

One ~100-step training touch (minutes; the wave's first of two). GPU not
required.

---

## 4. e093 — THRESHOLD-LAW GENERALITY: does the mass-action break move with window and scale?

### Organizing question

e089's law is single-net: on the ctx-512 0.87M net, removal cost is flat
through k=32 and breaks convexly 64→128→200 over a 351-entry anchor band
(final-frame ages 97–447). Two incompatible generalizations survive:
(a) FRACTION law — the break sits at a roughly constant fraction of the band
(f* ≈ 0.18–0.37 here), so bigger windows/scale shift k* proportionally;
(b) ABSOLUTE law — the break sits at a roughly constant ENTRY COUNT (k* ≈
64–128), a fixed redundancy pool, so bigger bands get proportionally safer.
A threshold that wanders idiosyncratically kills the word "law". This rides
existing checkpoints only.

### Cheapest discriminating experiment (eval-only CPU)

Replicate the e089 k-ladder (same matched-stream instrument, leg-local clean
judge, leg-local band definition — final-frame ages scaled to the window) on:

- **Window leg:** a block-256 family net (same corpus, same 0.87M class) —
  band roughly halves.
- **Scale legs:** the 8M (e005s_large) and the 10M extreme net from the
  e073/e079 inventory.

Per leg: k ∈ {2..band_max} log-ladder, 10 draws each (e089 convention, 58 s
CPU for the whole ladder on the original net — this is cheap). Report f* =
k*/band with bootstrap CIs; re-fire e089's two clauses per leg
(cost(32)/cost(max) < 0.25; convex ratio > 3).

**Optional folded leg (parked e090, one net only):** at k=128, one-shot
removal vs 8 lumps of 16 vs K=1 trickle — does the threshold MASS depend on
removal SCHEDULE (e089's 8× texture formalized), or only on mass?

### Registered prediction

FRACTION invariance: f* ∈ [0.15, 0.40] with overlapping CIs on every leg —
the law is "how much of the run's self-generated mass", a ratio law.
(Alternative firing: k* CIs overlap across legs at ≈ 64–128 regardless of
band ⇒ absolute-count law — also a publishable outcome, discriminated by the
same table.)

### Kill criterion

Neither invariance holds (clauses fail on ≥ 2 legs, or f* ranges beyond a
2× band with disjoint CIs) ⇒ T051's "law" is demoted to net-specific
texture; THINKING.md wording downgraded at the next review.

### Cost / flags

Eval-only CPU; free-run generations dominate (each leg ≈ minutes). The
e089 rig is frozen and gated — instrument risk minimal.

---

## 5. e094 — ONE-ROW TOLERANCE: can the host learn to read a foreign key? (the immunology question)

### Organizing question

e082 says the address is a one-row KEY cut for one lock: structure universal,
code seed-orthogonal (row-cos 0.059), graft rejected, crossmatch correctly
pre-screened it. The open immunology question at one-row granularity: is
rejection PERMANENT (the lock cannot be recoded — acceptance requires writing
a NEW key, i.e., what "install" actually is), or can the host be trained into
TOLERANCE (fine-tune everything EXCEPT the pinned donor row until the read
circuit decodes the foreign code)? Nobody has trained a host to accept a
specific foreign memory row while holding the row fixed.

### Cheapest discriminating experiment (one small training touch — flagged)

Host: e082_b43_install (seed 43). Donor row: e048_repro wpe-129 (seed 42),
the exact object e082's A2 arm rejected (0.198). Three arms, ≤ 200 steps
(2× the e043 install cost), e043 protocol LR/batch:

- **T1 tolerance:** overwrite wpe-129 with the donor row, PIN it (no grad),
  fine-tune the rest on the install-60. Readout: steps-to p(Z) ≥ 0.2 on
  install-60 + held-30.
- **T2 lock-and-key-free:** same but row unfrozen — track cos(row_t,
  donor_row_0) and cos(row_t, host-native-129): does the row relax back to
  host code (the lock rewrites the key) or stay donor while the net adapts?
- **T3 any-frozen-row control:** row pinned at a spectrum-matched random
  row, same budget — is "tolerance" donor-specific or generic plasticity
  toward any pinned row?

**DECISIVE POST-CELL (eval-only, cheap):** on any T1 success, destroy
wpe-129. p(Z) collapses ⇒ the circuit genuinely reads the donor code (the
lock was recoded — true tolerance). p(Z) survives ⇒ the host routed AROUND
the pinned row (a new key was cut elsewhere; the donor row was bypassed, not
accepted).

**INSTRUMENT RIDER (T046's registered open, accept-side):** crossmatch
cos(d2,d3) pre (0.059, REJECT) vs post-tolerance — does acceptance move cos
toward the 0.4459 rule? If T1 succeeds while cos stays far below the rule,
the instrument is blind on the accept side (it predicts rejection of things
that can be tolerated) — a paper-grade caveat for the crossmatch card.

### Registered prediction

Split. H-tolerance-acquirable: T1 bars within 200 steps AND T3 stays < 0.05
AND the post-cell collapses (donor-code-specific acceptance; the read circuit
can learn a foreign basis in ≤ 2× install cost — cheaper than growing one).
H-lock-rigid: T1 never bars while T2 bars (writing the key is mandatory —
install = key-cutting, not lock-recoding), or T1's success survives the
post-cell (routed-around, not tolerant).

### Kill criterion

T3 fires (any pinned row equally "tolerated") ⇒ generic plasticity, not
immunological matching — kill the tolerance interpretation, record as an
install-mechanism footnote. GATE-0-style: if the e043 install fails to
replicate on this lineage, PARK rather than tune.

### Cost / flags

The wave's second (and larger) training touch: ≤ 200 fine-tune steps at
0.87M scale — minutes; CPU feasible (e065 precedent), GPU optional but
flagged as the wave's only GPU-preferrable item.

---

## 6. e096 — the coherence-gap dose ladder (P-B, sharpened by the threshold law)

### Organizing question

P-B (harvested READY, eval-only) meets T051's law: mid-generation corruption
at increasing dose should NOT damage monotonically per nat — sub-threshold
perturbation mass leaves the run in its basin (drift accumulates but
self-heals), super-threshold mass collapses it toward the off-manifold
attractor, and very large corruption can re-enter a DIFFERENT coherent basin.
This is the continuous-dose face of the same threshold geometry e089 measured
in discrete entries — canalization made causal, and the anchor program's last
open promise.

### Cheapest discriminating experiment (eval-only CPU, e080 rig)

Corrupt a fixed mid-generation band at eps ∈ {0.05, 0.1, 0.25, 0.5, 0.75,
1.0} (noise/scale per the e080 convention), matched-stream controls, B=8,
clean-judge tail CE exactly as e089. Readout: per-nat damage (tail-CE cost
divided by perturbation mass) vs eps.

### Registered prediction

NON-monotone (U-shape): per-nat damage peaks at intermediate eps with CIs
excluding the monotone ordering at ≥ one contrast pair; small-dose arms show
permanent-but-sub-linear drift; the largest eps re-enters a coherent basin
(tail CE partial recovery + fluency spot-check). Secondary (descriptive, not
barred): the peak eps sits near the mass-threshold fraction band (~0.2–0.4
perturbed mass) if the two phenomena are one law.

### Kill criterion

Monotone per-nat damage with CIs compatible ⇒ canalization-continuity claim
dies (P-B's second haircut); the threshold law stays an entry-level fact,
disconnected from continuous corruption.

### Cost / flags

Eval-only CPU on the frozen e080 rig; ~6 arms × minutes.

---

## Free riders and parked items

- e093 optionally absorbs **e090** (dose-vs-schedule at k=128) — one extra
  arm, one net, pre-registered as texture.
- e092 reuses e091's Y-cell (0.362 carriage) as its localization reference
  and can report the entropy-drift flag (e065 R4) per restored component for
  free.
- e094's T2 row-trajectory is the first direct observation of "does the lock
  rewrite the key or the key recode the lock" — keep both cos tracks in
  metrics even if T1 fails.
- Parking lot untouched: e037, e036, v008/v012 synthesis, architecture-
  breaker sweep — none outrank the above under current policy.
