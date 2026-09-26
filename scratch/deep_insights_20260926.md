# Deep Insights — DEEP INSIGHT agent, 2026-09-26

Inputs: THINKING.md T001–T036 (full), DAY_ONE/TWO/THREE_REPORT.md,
scratch/novelty_inventory.md. This is not an audit — the interpreters own
that. This is the cross-cutting read: patterns that exist in the data but
in no entry. Every claim below is traceable to entries; the CONNECTIONS are
the new content.

---

## 0. The thesis in one paragraph

Across 64 experiments this lab has been studying what is, without naming
it, a **coordinate-keyed memory governed by a learned read policy that
only ever closes**. Knowledge is stored as addresses in coordinate systems
chosen at init or by cheap positional accident (wpe-130, vocab rows,
init-basis, recency); being stored is nearly free and happens constantly
(sub-argmax everywhere, 85–95% of cache present-but-unread); being READ
into argmax at the right coordinate is the scarce commodity; and every
longitudinal axis the lab measured (selection, training, re-learning,
exposure, scale) moves in one direction only — **closure**: the reachable
set of reads contracts around whatever the training distribution
exercised. The lab's dozens of inversions and corpses all triangulate the
same structure: the causal core of these nets is write-once, and the
writable surface is causally light.

---

## 1. Recurring motifs nobody has connected

### MOTIF 1 — Coordinate-keying: function is indexed by WHERE, not by WHAT
(five independent subsystems, one law)

| Subsystem | Entry | The coordinate | The collapse when it's missed |
|---|---|---|---|
| Expression gap | T032 | wpe-130 (one position) | one-char shift: p(Z) 0.556→0.12; off-geometry probe read 0 |
| Transplant rejection | T016/e028/e041 | init-basis (alignment ladder 1.0/0.53/0.15/0.00) | diff-init cos≈0.000; organs refine, never re-choose, the basis |
| Entity knowledge | T011 | vocab row indices (wte/lm rows) | 384 params erase; S_letter≈1 (identity won't leave its index class) |
| KV cache | T031 | recency (last ~7-token spike; sink dead) | 13–32% of old entries are NEGATIVE utility — old coordinates poison |
| Scar tissue | T018/T036 | the groove (original row direction, cos 0.76/0.73 n=2) | fresh names land orthogonal (0.278); re-learn refills the old groove |
| New circuitry | T021 | the late-attention SLOT (L5@6L, L7@8L) | head identity is a lottery but the slot is conserved — an address for circuits |

**The law the table states:** lookup is keyed by coordinates that were
cheap to fix (init draw, position embedding, row index, recency). Content-
addressed lookup exists ONLY where exposure paid for it — the COPY task
bought L4-H1 (T009); refrain density ≥5% flipped far context from
interference to retrieval (T021). **Content-addressability is purchased;
coordinate-addressability is the default.** Even the portability results
rhyme: the things that transfer across nets are statistical (energy,
allocations, coarse pipeline, attention's weak anchoring — "reads of
directions all adequate solutions share," T029), and the things that never
transfer are symbolic (bases, addresses, keys, second-factor heads).

### MOTIF 2 — Presence without expression: the unread model
(seven instances of the same gap)

Stored-but-not-read is the lab's most repeated phenomenon under seven
names: (a) sub-argmax persistent knowledge — correct token rank-2/rank-3
everywhere while floor is 1e-7 (T032, the territory T028 claimed); (b)
battery 0.974 with ZERO free-run occurrences at every dose/temperature/
seeding (T019); (c) the L5-calibrator — KL ~1 nat of distribution
reshaping at ≤+0.046 removal cost, 5/5 nets: the most robust mechanism in
the lab is a machine whose entire job is shaping a distribution WITHOUT
flipping argmax (T017); (d) the lens that "saw" L5 decisions where causal
truth was mid-stack — late activity is recalibration, not decision (T012);
(e) 85–95% of cache entries present-but-unread (T031); (f) mid-stack
readouts anti-informative (depth-2 CE 5.19 > unigram 4.17, T005); (g)
TF-completion given 'Z' ≈ 1.00 while the onset CHOICE fails (T032) — the
knowledge is present one token later than it is needed.

**The unnamed system these all point at:** a read policy — the machinery
deciding, per position, which stored candidate gets opened and which gets
to win argmax. The lab has measured its outputs everywhere and edited
everything EXCEPT it (rows, states, weights). Note the parameter-level
proof it exists: zeroing the lm_head WRITE row erases generation while
battery accuracy survives — reading and expression are separable rows of
the same matrix (T011 refinement 2).

### MOTIF 3 — Budgets are conserved, personnel is a lottery
(the meta-law, extended to everything it actually covers)

Every replicated positive is a BUDGET (how much authority where, how much
energy when, how big the live window, which stage exists): front-loading
(9.3× rebuilt under renorm, ρ=1.0 write/damage rank match, T003/T006);
L5-calibrator 5/5; energy carrier 5/5; L0-BLOCK top-1 in 20/20 cells;
pipeline shape across seeds/regimes (T008); late-attention slot for
retrieval heads (T021); coarse allocation defended under the equalizer
while write norms are decorative (T023). Every dead positive is PERSONNEL
(which head, which factor, which depth): L3H5 vs L4H4 vs L3H1 (T016);
attn-vs-MLP sublayer split (T017); gate depth sliding 3→4→4→5 (T014);
rare-token head (one prompt); r≈6 ascent (chaotic event). The lab wrote
"laws are ensemble properties; mechanisms are samples" — the sharper form
is: **the network is reliable about budgets and unreliable about
personnel; it is a schedule-wearing organism, not an organ-collection.**
Even criticality obeys the split: L0 is the most vital organ at 2.7M yet
plugs in cross-seed almost cleanly, while the mid-stack rejection sites
are anti-aligned (cos −0.013) — vitality says nothing about rejection
(T029). Rejection lives where individuality lives (see Motif 5).

### MOTIF 4 — Canalization: every longitudinal axis points toward closure

One direction, six independent instruments: (1) the basis is written once
at init and selection cannot see it (e040: −5.5% trickle, alignment floor)
and directed mutation cannot reach it (e050: −3.6% ≈ random −3.5% — the
limiter is the view, not the reach); (2) training shrinks the live cache
window under the sign statistic (a* 182→32→25, 400→4000 steps; the
registered-threshold statistic says GROWS — see §6, both are moments of
CONCENTRATION); (3) expression requires free-run-shaped exposure —
TF-shaped install at any dose never confers it (T019), i.e. the canal only
admits reads shaped like what flowed through it; (4) re-learning is 2–2.9×
SLOWER than fresh yet regrows the same groove — the channel deepens even
as it narrows (T018/T036, n=2); (5) the network DECLINES late authority
even when renorm makes it purchasable (T003) — closure is not a capacity
limit, it is the optimizer's preference; (6) retrieval stays
compartmentalized — zero leak into ordinary text at any density (T021) —
purchased readers don't generalize their citizenship. Biology has the word
already (Waddington): **canalization**. Nobody in the lab has used it, and
it is the correct parent concept for "init-anchored," "frozen under
selection," "expression-bound," and "history" at once.

### MOTIF 5 — The sovereign middle: all measured variability co-locates
in the stack's interior; the ends are conserved

The default pipeline is a sandwich: L0 = input formation (conserved,
plug-compatible, near-seed-determined), L5 = output calibration
(conserved, 5/5), and the INTERIOR = commitment, suppression, and
individuality: the causal gate lives mid-stack in every net but its depth
slides with seed/regime/architecture (T014/T020); the address dies across
blocks 1→2 (d1-peak/d2-crash, T032/T033); A-residualized geometry-
sensitivity concentrates mid-stack at r=0.916 (T029); cross-seed dW is
ANTI-aligned exactly at L3/L4 — the rejection zone is where solutions
diverge (T029); distributed-decision mass and interference peaks are
interior. e064 killed the claim that the interference instrument tracks
the gate — but what it left standing is stronger than anyone wrote down:
**every instrument that measures where these nets DIFFER from each other
lights up the interior.** The ends are where the task lives; the middle is
where each training run exercises its sovereignty. (Scope note: at 4L the
"middle" degenerates — mode at the final block, "a 4-layer stack has no
mid," T020 — the interior is defined relative to depth budget.)

### MOTIF 6 — Dynamic viability: what separates a memory from a crank

The lab discovered the criterion twice without naming it: a write is real
iff it survives the model's own dynamics. The d4 one-shot state write
persists 32 rows downstream and produces recurrent Z-words; the same
donor state at a non-onset position cranks p(Z) to 0.509 at +1 and is
floor by +2 — a loud logit PASTE, zero recurrence in 288 continuations
(T033/T035). The groove survives erasure (T018); off-geometry states are
actively killed by the model's dynamics (A-rev: 0.775→0.08, T033); early
writes are digested across blocks 1→2 (d1-peak/d2-crash). Free-run
continuation is to states what it already was to installs (the battery
honesty check, T019): **the only incorruptible witness is the model's own
next steps.**

---

## 2. The inversions, taxonomized — and what they collectively say

The lab's expectation inverted at least a dozen times. The inversions are
not noise; they form four families:

**I1 — Write/read asymmetry (subtraction works, addition never does).**
Removal surgical (384 params, class-exact, scale-invariant); installation
impossible surgically, cheap only via on-distribution exposure (T015);
ascent anti-selective (r≈1.08, and 6.6× PAST random into inversion —
ascent doesn't randomize the net, it inverts it, T002); grafts damage
without installing (cross-init rows +1.92 nats, zero install); selection
and directed selection both trickle (T024/T027); off-position state
injection is transient (T035). **Sharp form: the only successful non-
gradient writes are SUBTRACTIONS (row-zero, lesion) or REPLACEMENTS AT
MATCHING COORDINATES (state transplant at its own position); no non-
gradient write has ever ADDED function.** Addition goes through gradient
descent on-distribution or not at all.

**I2 — Scale and training counter-intuitions.** Smaller nets hold cache
entries live longer (0.84M a*=36 vs 2.7M 24; 10M shortest window, 4
tokens); training shrinks the live window (one statistic — conflict
flagged in §6); the deepest attention is literally free at 10M (≤0.04
nats ×3 blocks) while shallow MLP-0 is the keystone (3.40); the
undertrained net uses its whole cache, the trained net a spike; far-value
rises with scale (+0.035 at 10M) while old entries become MORE harmful
(32% negative-utility) — bigger nets bought far-content sensitivity AND
far-entry intolerance (unremarked tension, §6).

**I3 — Activity/importance inversions.** Zeroing ≈ 60°-rotation for most
blocks (direction nearly free; energy is the currency); KL~1-nat of L5
reshaping at ~zero CE cost; the battery overstates install 3.4×; the lens
saw decisions where there was calibration; the a*=63 "invariant" was bin
quantization; raw-D peaked at the gates while D/A manufactured mid-stack
structure (T030). **Family law: every instrument that routes through the
model's own readout inherits the model's read policy and lies in its
direction.**

**I4 — Memory-direction inversions.** Re-learning is slower than fresh
yet lands in the original groove; the old carrier head becomes an
ANTI-carrier (zeroing it now improves the memory); shuffled-far context
hurts MORE than real-far (incoherence destabilizes worse than misleading
coherence); the single expression event came from the control arm, and the
"zero-expression" of e048 was the probe being 10 coordinates off an
install that worked.

**What the four families collectively say:** causality in these nets is
concentrated exactly where intervention cannot reach — coordinates chosen
at init, readers distributed everywhere, argmax adjudication mid-stack —
while everything intervention CAN reach (magnitudes, distributions, late
layers, single heads, logit cranks) is causally light or a lottery.
**The causal core is write-once; the writable surface is causally light.**
That is why the lab's surgical instrument is a scalpel that only burns and
why its best causal intervention is a transplant that only replaces a
state with its own counterfactual self.

---

## 3. THE UNIFIED QUESTION

> **What is the read policy — the rule that decides, at each position,
> which of the model's stored coordinates get opened and which stored
> candidate gets to win argmax — and why can it be written only by
> on-distribution gradient, never by intervention?**

Two faces, one question:
- **The keying face** (why addresses are position/basis/row/recency-bound,
  and content keys must be purchased) — illuminates: the expression gap's
  wpe-130 binding, init-anchoring and its non-evolvability, row surgery's
  selectivity, the cache spike + negative-utility entries, the retrieval
  threshold flip, transplant rejection, scar grooves.
- **The arbiter face** (why presence never becomes expression) —
  illuminates: sub-argmax persistence, TF/free-run dissociation, the
  calibrator's massive-KL-zero-cost profile, the d1-peak/d2-crash
  suppression, transplant rescue vs loud paste, battery overstatement,
  the write-row/generation dissociation.

Answering it touches the majority of the lab's headline results
simultaneously. Note what it does NOT subsume (honesty): the energy/
budget results (Motif 3) and the retrieval anatomy — those are the
schedule the read policy operates on, not the policy itself.

---

## 4. What the negatives map

The corpses: ascent unlearning (all granularities); far retrieval on
natural data; surgical installation; two-factor erasure transfer;
stream-geometry as the cause of front-loading; attention subspace
sharing; lens=causal-depth; causal-depth cross-net invariance;
interference=gate unification; basis evolvability (random AND directed);
sink load-bearingness; directed>random mutation; the a*=63 invariant;
rare-token re-globalization; repetition-interference as the loser cause;
letter-overlap collateral; LN-calibration confound; energy-schedule
necessity.

The shape of what ISN'T there:
1. **No categorical claim survived.** Every corpse is a clean/categorical
   story (erase completely; install surgically; a sharp threshold; an
   invariant; a conductor; a shared subspace; one confound). Every
   survivor is graded/distributional (tug-of-war far-value, spike+
   shoulder+plateau, sub-argmax ranks, distributional card entries).
   Behavior grades; anatomy quantizes (T021) — but the quantized things
   are SLOTS, not modules. **The module-ontology fails at this scale; the
   policy/budget-ontology holds.** The one organ-like universal (the
   calibrator) is itself a distribution-shaper, not a decision-maker.
2. **No conductor.** Sink dead 5/5; late attention free; calibration
   local; decisions interior and partially distributed (7–26%). Control
   is local reading with a commitment point — nothing at the top or at
   special positions runs the show.
3. **The training distribution is the only write interface.** Every
   non-gradient write failed to ADD; every gradient write worked only
   with exposure shaped like the intended read (free-run exposure for
   free-run expression). The system is closed to everything except its
   own training objective. This is the negatives-map version of
   canalization — the corpses are the wall of the canal seen from inside.
4. **No content portability across nets, ever.** Cross-net transfer
   succeeded only for statistics (energy, allocations, weak-anchored
   attention reads). Nothing symbolic ever crossed a seed boundary.

---

## 5. Coins — the concepts the lab keeps circling without naming

1. **CANALIZATION** (deliberate Waddington borrow): the monotone closure
   of the set of reachable reads around the training history. Parent
   concept of: init-anchoring, frozen-under-selection, expression-requires-
   matched-exposure, slower-but-groove-true re-learning, declined-late-
   authority, compartmentalized retrieval. One word replaces five law
   names and predicts the sign of every future longitudinal result.
2. **THE READ POLICY** (aka the arbiter): the per-position rule that
   decides which stored coordinate is opened and which candidate wins
   argmax. The lab's four faculties are its shadow: ADDRESS is what it
   reads, ABILITY is its training cost, EXPRESSION is its verdict,
   HISTORY is its canal. It is the one component the lab has never
   edited — and (§3) the highest-value target.
3. **THE SOVEREIGN MIDDLE**: the conserved-ends/idiosyncratic-interior
   stack structure (Motif 5). e064 killed one unification instrument; the
   co-location of ALL cross-net variability in the interior stands as the
   residue nobody has stated.
   Minor coins worth minting alongside:
   - **DYNAMIC VIABILITY** (Motif 6): survival under the model's own next
     steps as the criterion separating memory from logit crank — the
     state-level generalization of the free-run honesty check.
   - **BUDGET/PERSONNEL SPLIT** (Motif 3): budgets replicate, personnel
     lotteries; the design rule is to pre-register budgets, never heads.

---

## 6. Unremarked tensions and free observations (each is a registered
discriminator waiting to happen)

1. **The 10M far-context tension:** far-value RISES with scale (+0.035,
   p99 2.3) while old-cache negative-utility WORSENS with scale (32%, 4-
   token window). Same far entries, opposite signs by instrument.
   Discriminator: at 10M, intersect the far-value-gain positions with the
   negative-utility positions — disjoint populations rescues both claims;
   overlap means the far-value gain is carried by entries whose removal
   still helps on net (concentration again).
2. **The e053 conflict is itself data:** registered-threshold a* GROWS
   (3→86) while sign-frac live-window SHRINKS (0.68→0.13) with training.
   Unifying reframe: training CONCENTRATES utility — taller spike,
   narrower support — height and width move oppositely, and each
   statistic reads a different moment. Registered decider: plot max/
   mean dCE (concentration) vs steps; monotone rise under BOTH old
   statistics confirms the reframe (and is another canalization
   instance).
3. **The calibrator-as-arbiter candidate:** the most robust mechanism in
   the lab (L5, 5/5, KL~1 nat, free) is exactly the machinery that
   enforces the training prior on the output distribution; the expression
   gap is a failure to get past a prior-enforcer. The d2-crash localizes
   address destruction mid-stack, but PRIOR-ENFORCEMENT late is untested.
   Cheap eval-only discriminator: temperature-scale the L5 readout
   (interpolate L4→L5) at onset positions and watch p(Z) — if the sub-
   argmax prior moves toward argmax as calibration is unwound, the
   suppressor has a late face too.
4. **The one expression event's fingerprint:** direct-trained control, at
   home slots, p(Z) 0.27 with argmax 25% (T019) — natural learning buys
   argmax citizenship; install buys rank-2. The read policy prices
   citizenship by exposure SHAPE, not amount.
5. **Instruments are models:** lens/census shared bias, D/A
   normalization, bin-quantized onset, off-geometry probe, battery 3.4×,
   loud paste — six cases where the instrument's transfer function WAS
   the finding. The lab's only instrument that never lied is free-run
   continuation. Formalize: prefer instruments whose readout does not
   route through the model's own posterior.

---

## 7. What I would register next (cheapest first; none require new
training except where noted)

1. **Concentration curve** (zero-GPU, existing e053 artifacts): max/mean
   dCE vs training steps, both old statistics reconciled under the
   concentration hypothesis (§6.2).
2. **10M far-value × negative-utility intersection** (eval-only on cached
   cells; resolves §6.1).
3. **Calibrator-arbiter probe** (eval-only): L4→L5 interpolation at onset
   positions; does unwinding calibration promote the sub-argmax
   knowledge? (§6.3).
4. **Purchased citizenship test** (one small training): install ZEPHYRA
   at matched dose under teacher forcing vs scheduled-sampling/free-run-
   shaped exposure; prediction from canalization — expression > 0 ONLY in
   the matched-shape arm, at equal battery accuracy.
5. **Third-canal test** (one small training): a third erase→re-learn
   cycle on B43; canalization predicts monotonically slower AND
   monotonically more surgical-proof re-learn (deepening groove), not
   oscillation.
6. **Sovereign-middle census** (needs one 12L run): seed-variance of
   causal-gate depth by layer position; prediction — variance peaks
   interior, near-zero at the ends.

---

*Filed by the DEEP INSIGHT agent, 2026-09-26. No other files were
touched. The sharpest five of these are relayed in the reply; the file is
the full residue.*
