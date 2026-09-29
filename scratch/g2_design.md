# g2 — THE REHEARSAL-NATIVE ARCHITECTURE (the rehearsal organ)

**W020's generative turn, law 2 of 4: THE RESURRECTION ECONOMY as design spec.**
Design agent, 2026-09-29. Status: DESIGN — bars registered BEFORE implementation
(the g-series discipline; W020: "registered predictions IN ADVANCE; wrong
predictions are the point — every law that fails in a new architecture was
contingent; every law that holds is closer to necessary").
Dispatch target: `lab/g2_rehearsal_native.py`; outputs `runs/g2/`.

THE ONE-SENTENCE DESIGN: the base net keeps its wash law untouched, and g2 adds
a **zero-new-parameter organ** that (a) indexes what it was taught, (b) watches
its own read of the index, and (c) when that read decays, re-teaches itself the
index with error-isolated, position-jittered self-replay. Maintenance moves from
the dataloader's accident into the model's own control loop; the sawtooth
becomes architectural.

---

## 0. PROVENANCE (extend, don't repeat — SUPERVISOR directive 1)

BUILDS ON (every clause measured):

- **E179/T120** (the resurrection economy): ≤9 replay batches per 300 wash
  steps maintain the fact (r=1/32 → 0.699 at +300, whole anatomy intact); ONE
  replay event resurrects the dead (0.033 → 0.686 by +50, 18 wash steps after
  the event); non-monotone sawtooth (1/8, 1/4 MISS at +300 by 0.074/0.025 —
  density is not a dial); re-entry is cheap and sticky where residence was
  impossible. The replay batch arithmetic (e174 arm B = e113's jitter pool +
  7-name-char masked CE + anchors) is the maintaining recipe, VERBATIM.
- **E174/T107** (the maintenance budget): without rehearsal the dose curve is
  a cliff (F1 dies at the FIRST F2 gradient — dose two in the smoke, 0.0009);
  1:1 interleaved rehearsal maintains (0.921 at arm A's kill point) at ~zero
  cost to the concurrent task (F2 0.971 vs 0.994). R51's surviving form:
  REHEARSAL MAINTAINS (direction, two independent protocols).
- **The wash law** (E161/T101, E176/T105, E176N/T109, E183/T111, E184/T112,
  E187/T121, E180/T119): no memory state tested retains expression under the
  rehearsal-free stream — stream-invariant, seed-replicated (n=3), 2 families;
  exit is displacement-limited (t* ~ 7.5e-4 x lr^-1.16, R^2 0.975; family 2
  dies at +1). Every past fine-tune's anchor bank was quietly rehearsing the
  fact (T105): the lab's own protocol was the memory's life-support.
- **E121/T074 + T079** (the dreams objection): the net's own dreams CARRIED
  the fact (22.5 ZEPHYRA per 10k own-dream chars) and consolidated NOTHING —
  verbatim dream replay slows forgetting ~500x (0.04–0.07 vs 0.0001) but dies
  under address deletion; 33/34 dream ZEPHYRAs sit at x-col 130, the OLD
  address (zero position variance → the positional key keeps the credit →
  nothing re-routes); replay-without-error left the fact BELOW base at the
  dream positions (0.230 vs 0.391) — echo ERODES. The two roads that work
  (jitter e109/e113; deletion pressure e083) both carry an explicit
  reorganization signal that verbatim replay lacks.
- **E157/T113** (family 2): the ≤1M base and its wash are ALREADY ON DISK —
  the e098 s4305 line (Cfg n_layer=4, n_head=4, n_embd=128, block=256 =
  0.838M params; `e098_base_s4305.pt`, `e157_f2_consolidated.pt`,
  `e157_f2_neutral_s*.pt`). Family-2 root: g-12 0.198, g0 0.578, g+12 0.591;
  dies on the FIRST neutral-wash step (g-12 0.0011 at +1). Trainings ~35 s
  each on GPU.
- **E043** (install machinery): ZEPHYRA spliced at PRE=130 into 60 install
  hosts (SPLICE_RNG 24301), masked name-position CE + anchor interleave —
  deviation-3's measured lesson: token weighting (what carries the loss)
  determines what installs.
- **W019** (the radical view: no archive, only practice) and **W020** (the
  g-series program).

WHAT IS NEW (nothing below exists in the lab yet):

1. **THE GATE** — an internal read-strength monitor replaces the external
   replay schedule: event-triggered, not clock-triggered. e179's clock was the
   experimenter's; g2's is the net's own mismatch signal.
2. **THE ORGAN** — the replay pool moves from the experimenter's scratch
   memory (e179's `jit_pool_x` tensors) into the model's registered buffers,
   written once by an index-on-teach rule at install time. The dataloader
   holds NO fact content at wash time.
3. **THE ANTI-ECHO LOCKS, registered as the active ingredients** — (i)
   gate-guaranteed error, (ii) name-char-masked CE, (iii) position jitter ±8.
4. **THE GHOST CONTROL** (echo cell) — gate-triggered verbatim replay without
   mask/jitter: e121's verdict tested INSIDE the architecture.
5. **Step-parity protocol** — replay events REPLACE wash steps (300 optimizer
   steps in every cell): e179's replacement convention made architectural.

Model-size statement (SUPERVISOR directive 2): 0.838M base + 0 new learnable
parameters — the smallest family the lab owns that already carries the wash
law (e157 family 2), satisfying the ≤1M design constraint with headroom. The
mechanism adds buffers, not weights; if a rhythm needs capacity to work, it
was not the rhythm.

---

## 1. THE LAW AS DESIGN SPEC

T120's clauses, each becoming one spec decision:

| Law clause (E179/T120 + kin) | Spec decision |
|---|---|
| Exit is cheap (t* = 1–2 at lr 1e-3; displacement-limited, E180) | Do NOT resist exit. No anchoring, no dual-rate weights (that is g1's design space). |
| Re-entry is cheaper (ONE event resurrects; 18+ steps of stickiness) | One replay event per gate opening; the event is the gate's close. |
| 9 events / 300 steps maintain; density is NOT a dial (1/8, 1/4 miss) | Gate + refractory must land 5–25 events/300 — the economy's band, never a dense schedule. |
| Maintenance = a re-entry rhythm, not residence (the sawtooth) | The architecture PRODUCES the sawtooth. Dips between events are expected; adjudication at +50/+300 STATES (e179's smoke-corrected convention). |
| Rehearsal costs the concurrent task ~nothing (E174) | Events replace wash steps 1:1 (optimizer-step parity); CE_R must stay in the healthy band. |

The architecture deliberately does NOT widen the basin, harden weights, or
store the fact robustly: g2 bets WITH our own laws (nothing survives the wash;
e176/e177/e180/e185/e187) and implements the one thing the evidence says
works — cheap, error-carrying, event-limited re-entry.

---

## 2. THE E121 OBJECTION — AND THE THREE LOCKS THAT ANSWER IT

**Objection (QUEUE g2 row, verbatim):** "dreams carried the fact but
consolidated nothing — why does INTERNAL replay do better?"

**Why the dreams failed (measured, T074/T079):** (1) ZERO ERROR AT THE
ADDRESS — the dream context at x-col 130 is exactly the cue that keys the
read the net still has; the net predicts its own dream with low surprisal, so
the name tokens carry almost no loss mass, and optimizing the context tokens
moves weights toward generic text at those positions — net effect: the fact
ERODES below base (0.230 vs 0.391). (2) ZERO POSITION VARIANCE — 33/34 dream
ZEPHYRAs at the old address; the positional key keeps the credit, nothing
re-routes (T079). Content was present; error and variance were absent.

**The three locks (each lock cites the lab result that motivates it):**

- **LOCK 1 — THE GATE GUARANTEES ERROR.** The replay fires ONLY when the
  monitor reads the fact below θ_OPEN = 0.5 — so at the moment of replay, the
  name-char CE is by construction large (the read is weak ⇒ the loss on the
  name tokens IS the read's error). e121's schedule replayed dreams regardless
  of state — mostly echo; g2's gate converts the schedule's ignorance of state
  into the architecture's knowledge of state. The clock did not know when
  error existed; the gate does.
- **LOCK 2 — THE MASK ISOLATES THE ERROR.** The event's loss on the name
  windows is the 7 name-char targets ONLY (e174 arm B's masked-CE arithmetic
  VERBATIM); the error mass cannot be diluted by context tokens (e043
  deviation-3: what carries the loss determines what installs). When the read
  is intact this loss is ~0 → no gradient → no cost: the mechanism is
  self-limiting.
- **LOCK 3 — THE JITTER SPREADS THE CREDIT.** Event windows are drawn from
  e113's jitter pool (offsets {-8,-4,0,+4,+8}; the grown address band
  121–137), so re-entry cannot collapse back onto the single positional key —
  the variance channel e109/e113 showed converts replay into geometry-general
  access.

**The economy clause:** the gate only needs to fire ~10 times per 300 steps,
because re-entry is sticky (E179: one event's re-entry outlasted 18+ wash
steps). g2 does not hold the net in the basin; it re-enters cheaply on its own
detective work. If the three locks are the active ingredients, the ghost
control (Section 5, CELL-ECHO) will show it: the same organ and gate WITHOUT
locks 2+3 should erode — e121's signature reproduced inside the architecture.

---

## 3. WHY NOT THE OTHER CANDIDATES

**(b) ATTRACTOR READOUT** — a recurrent attractor layer whose fixed points
encode the fact. Rejected for g2, assigned to g3: (i) it answers REVIVAL, not
MAINTENANCE — the law's surprise is the economy of events, not the basin's
existence; (ii) it is a new parametric store ON the readout path, so the
existing dials would partly measure the new module (instruments conflated),
and as weights it too washes — it needs its own maintenance (circular) unless
frozen (then it is an anchor, i.e. g1's weight-anchoring in disguise); (iii)
a static attractor eliminates the rhythm rather than embodying it — it bets
against the very law we are testing. The modern-Hopfield store is literally
g3's registered candidate.

**(c) SYNAPTIC CONSOLIDATION SCHEDULE** — per-weight fast/slow EMAs in
optimizer state. Rejected for g2, assigned to g1: (i) it is optimizer state,
not architecture (the candidate's own caveat), and g1's registered candidates
are exactly "weight anchoring / dual-rate / gated updates"; (ii) it is a
RESISTANCE mechanism — a slow-EMA copy of fact-bearing weights is a frozen
archive that trivially "survives" the wash by restoration — and the lab KILLED
the resistance story (e176/T105: the consolidated fact has no more
wash-resistance than the dwell peak; "consolidation = gradient-resistance
acquisition" is dead; e187: no robustness basin even against noise). Betting
on resistance bets against our own data.

**The division is itself a design claim:** the three mechanism families map
onto the three laws — resistance/widening → g1, rhythm → g2, generative
storage → g3 — and g2 picks the one the evidence says is load-bearing:
EVENTS, NOT RESISTANCE. If g2 fails while g1-style anchoring succeeds, the
resurrection economy was the contingent one.

---

## 4. THE MECHANISM — the rehearsal organ (0 new learnable parameters)

**The base:** `TinyGPT(Cfg(vocab=65, n_layer=4, n_head=4, n_embd=128,
block_size=256))` = 0.838M params (the e098/e157 family-2 line; instruments
and wash precedent already exist on it).

**The organ** (a wrapper `G2Net` holding the body + four components; the body
itself is byte-identical TinyGPT so every instrument loads it unchanged):

1. **CUE POOL** (`registered_buffer`, write-once at install end): e113's
   jitter pool VERBATIM — 300 windows (60 install hosts x offsets
   {-8,-4,0,+4,+8}), each 256 tokens with ZEPHYRA spliced at home, plus its
   name-position target mask (300 x 255 bool). ~150 KB as int16. **Write
   policy (the index-on-teach rule):** every fact-bearing window whose
   name-char targets entered the install loss is indexed, once, at the end of
   install — the hippocampal one-shot write. Never updated afterward (the
   re-registration/reconsolidation variant is g2b, out of scope).
2. **READ MONITOR** (no-grad, no params): every CADENCE = 4 wash steps (never
   during REFRACTORY), forward K_MON = 8 cue windows (offsets cycling
   j ∈ {0, -4, +4}) and read the mean p(true name char) over their 7 x 8
   name positions — the net's own confidence in its index. This is the
   mismatch/comparator signal; it is NOT the external battery (co-reported
   alongside the ruler at every checkpoint so any decoupling is visible).
3. **REHEARSAL GATE** (a comparator, no params): opens when monitor < θ_OPEN
   = 0.5 (the maintain bar constant, e176n/e179). One event per opening.
   After an event, REFRACTORY = 24 steps of lockout (≥ e179's measured 18-step
   stickiness + margin — the refractory IS the stickiness, made architectural).
4. **ERROR REPLAY** (the event): that step's wash batch is REPLACED by e174
   arm B's replay batch VERBATIM — 16 windows drawn fresh from the 300-pool +
   16 anchors (8 neutral-bank + 8 random corpus, full CE), union CE with the
   7-name-char mask on the 16 name windows. Batch size parity (32 = the wash
   batch's 32). Same optimizer (AdamW 0.9/0.95, wd 0.1, lr 1e-3 constant,
   clip 1.0), same step count.

**Anti-triviality clauses (the honesty reflex, pre-registered):**

- **THE ORGAN IS NOT IN THE FORWARD PATH.** The dials measure the body alone;
  the organ contributes ONLY gradients at events. Check: at +300, a dial
  battery run with the organ's buffers zeroed must read BIT-IDENTICAL
  (gate G_ORGAN-INERT). No lookup-table shortcut exists by construction.
- **THE ORGAN IS NOT AN ARCHIVE OF THE FACT** in the sense W019 killed: it
  stores the RETRIEVAL CUE (context + name tokens), not the fact; the fact
  lives in the body's weights and is measured there. If the body's dials die,
  g2 has failed even though the organ still "contains" the name — the
  experiment adjudicates the BODY.
- **THE TRENCH-COAT TEST:** what distinguishes g2 from "a replay buffer in
  the dataloader"? (i) The pool lives in the model (state_dict, checkpoints);
  (ii) the schedule is gone — no external r, no clock, no experimenter; the
  net decides WHEN from its own state; (iii) the stream is verifiably
  fact-free (G_NAMEFREE on every wash draw). What remains external is only
  the wash itself — exactly the environment's part.
- **Wash stream UNCHANGED:** e176n arm A's neutral wash verbatim (16
  neutral-bank windows, seed 170, rejection on FLORIZEL/ELIZABETH/ZEPH/
  MIRABEL, 0/16 junctions + 16 random corpus windows, full-token CE).

**Biology echo (one paragraph, for the paper's discussion):** the organ is
complementary-learning-systems theory implemented in 0.838M params — a fast,
one-shot, content-addressable episodic index (hippocampus) maintaining a slow
cortical store through mismatch-gated, error-driven reconsolidation — with
the lab's INVERSION made explicit: the cortex never hardens (no systems
consolidation into resistance — e176/T105 killed it), the hippocampus never
hands off, and the rhythm IS the memory (W019's radical view, weaponized:
the architecture concedes there is no archive and installs a practice organ
instead).

---

## 5. THE EXPERIMENT — cells, instruments, protocol

**Lineage:** the e098 0.84M family. Root = a fresh e043-Dmix install + e113
jitter consolidation (e157's ported recipe) on an e098 base, reaching
ROOT-STRENGTH (below); if the e157 s4305 root already passes ROOT-STRENGTH,
reuse it (`e157_f2_consolidated.pt`). Organ registered at install end.

**ROOT-STRENGTH GATE (frozen before compute):** the ruler (below) at root
must be ≥ 0.7. If a fresh install lands below, re-install (another e098
seed / more exposure steps) rather than adjudicate on a thin root. Never
lower the bar.

**RULER (frozen rule, no shopping):** among the three battery geometries
{-12, 0, +12} measured on the root, the ruler is the geo with maximum root
mean_pz (family 1 → g-12 0.916; e157's family-2 root → g+12 0.591). Die bar
0.27, maintain bar 0.5 — e158/e161/e176/e176n/e179 constants verbatim. All
three geos + held30 co-reported at every adjudication point.

**Cells (each: 300 optimizer steps, lr 1e-3, AdamW(0.9,0.95) wd 0.1, clip
1.0, batch 32, seed 10902, checkpoints {1,2,4,10,25,50,100,200,300}, light
evals per checkpoint + FULL dial at +50 and +300 — e179's grid verbatim):**

- **CELL-BASE** (the dying side; the key contrast's control): the same net,
  organ inert (gate never opens — the ablation identity: with the gate
  disabled, g2 IS the base). Expected: replicates family 2's stored wash
  (e157: dead at +1); G_REPRO vs `e157_f2_neutral_s*` (e179's r=0-replicate
  rider convention). **This cell is a GATE, not an assumption:** if the base
  does NOT die by +50, there is no contrast — record a size/family bound on
  the wash law and stop (also informative).
- **CELL-G2** (the key contrast cell): gate live, all three locks on. THE
  REGISTERED QUESTION: under the SAME rehearsal-free stream that kills the
  base, does g2 survive?
- **CELL-ECHO** (the e121 ghost control): gate live, but the event's loss is
  full-token CE on 16 verbatim j=0 cue windows + the same 16 anchors — NO
  mask, NO jitter (locks 2+3 off; lock 1 on). Isolates the anti-echo delta
  as the active ingredient.
- **CELL-SCHED** (the schedule anchor; optional-cheap): organ-less base +
  EXTERNAL r=1/32 schedule (e179's protocol verbatim at this family/seed) —
  anchors the maintaining band at ≤1M, so "gate matches schedule" is an
  apples-to-apples claim.

**Instruments (all existing, zero new):** the 3-geo batteries + held30
(install-60/held-30 contexts), CE_R (e065 val windows, seed 26502), site read
@183 (onset/span), old-band row census (row-0 sink strength, A129 brake),
D-all/D183 wpe deletion table (e180's `measure()` verbatim). NEW LOGGED
QUANTITIES (logging only, not new instruments): monitor trace, gate openings,
per-event {step, pre-event monitor, post-refractory monitor, ruler at next
checkpoint}, event-spacing histogram, realized r = n_events/300.

---

## 6. REGISTERED PREDICTION (frozen before compute; no bar shopping)

**Primary (CELL-G2 vs CELL-BASE, the key contrast):** under the same
rehearsal-free neutral wash that kills the base by +50 (CELL-BASE ruler ≤
0.27 — the contrast gate), **g2 MAINTAINS: ruler ≥ 0.5 at BOTH horizons +50
AND +300** (e179's state-based maintain convention verbatim). The trajectory
is a self-triggered SAWTOOTH whose dips stay bounded and whose every event is
followed within ≤ 18–25 steps by a ruler ≥ 0.5 (the e179 resurrection
signature, now self-detected). MID-CYCLE DIPS BELOW 0.27 DO NOT UN-MAINTAIN
(e179's own smoke-corrected convention; the checkpoints sit at unknowable
cycle phases — the late-grid mean/min over {100,200,300} is co-reported for
phase honesty).

**Secondary outcomes, each registered:**

1. **ECONOMY:** 5 ≤ n_events ≤ 25 (realized r ∈ [1/60, 1/12]); event spacing
   concentrates in 20–45 steps (refractory-bounded rhythm). If it maintains
   at n_events ≤ 4, the gate is cheaper than the schedule — a STRONGER
   economy than e179's (texture, recorded).
2. **ANATOMY INTACT at +300 (full dial):** every dial ≥ 50% of ITS OWN root
   value (self-referential bar — no cross-family assumption): held30, site
   read onset/span, row-0 sink, A129 brake, and **D-all deletion survival
   (D-all g0 ≥ 0.5 x root's)** — g2 must maintain the GENERALIZING readout,
   not an address echo (e179's r=1/32 cell kept the whole anatomy; e121's
   echo died under address deletion).
3. **ORGANISM HEALTH:** CE_R at +300 ≤ root + 0.10 (e174's ~zero-cost
   precedent); no sustained concussion (the base wash's own +1/+2 transient
   ~2.0–2.2 recovers; g2's events must not hold CE_R above the healthy band).
4. **THE GHOST ERRODES (CELL-ECHO):** late-grid sustained level (mean ruler
   over {100,200,300}) < 50% of CELL-G2's, declining across the three
   checkpoints, ending < 0.4 at +300 despite the gate firing — e121's
   erosion signature inside the architecture. This cell is the mechanism
   attribution: if ECHO maintains too, locks 2+3 are NOT the active
   ingredients and e121's verdict must be re-localized (recorded either way).
5. **SCHEDULE PARITY (CELL-SCHED):** maintains at r=1/32 on this family
   (e179's result replicating at ≤1M) — expected; if it FAILS here, the
   family-2 lineage is schedule-fragile and every g2 verdict gets that bound.
6. **MONITOR-RULER COUPLING:** monitor and ruler track at every checkpoint
   (both normalized to root). A divergence — monitor high, ruler low — is the
   CUE-OVERFIT texture (the net maintains only its index); caught
   independently by the D-all/held30 bars.

**The law-level stake (W020's frame):** if the primary fires, the
resurrection economy is ARCHITECTURALLY SUFFICIENT — the rhythm was the
missing organ, and e121's failure is localized to echo (no isolated error, no
variance), not to self-generation; the law moves toward necessity. If it
fails, the economy was SCHEDULE-CONTINGENT — the dataloader's phase carried
information the net cannot self-supply — and the law was contingent.

---

## 7. FALSIFIERS (frozen; adjudicate against exactly these)

With the contrast gate passed (base dies by +50) and the gate demonstrably
firing (n_events ≥ 3, each event's batch verified by G_REPLAY):

- **F1 — SCHEDULE-BOUND (the main falsifier):** CELL-G2 ruler < 0.5 at +50
  OR +300. Internal, gate-triggered, error-carrying replay does NOT maintain
  where the external schedule did. The resurrection economy is not
  self-triggerable at this threshold/cadence.
- **F2 — EROSION (the e121 ghost):** maintains at +50 but the late-grid
  {100,200,300} mean < 0.5 AND monotonically declining — the events are echo
  despite the mask; internal replay erodes like dreams did; e121's verdict
  extends to architectural replay.
- **F3 — GATE-NEVER-CLOSES:** every post-refractory monitor check fires
  (n_events at the refractory ceiling ~12–13) AND maintain still fails —
  re-entry never sticks on this family; the rhythm cannot bootstrap.
- **F4 — ORGANISM PRICE:** CE_R > root + 0.25 at +300 (sustained) —
  maintenance bought at the corpus's price is a fail (the lab's
  organism-health reflex).
- **F5 — ANATOMY FAIL (secondary):** ruler maintains but D-all deletion at
  +300 kills (D-all g0 < 0.27 vs root's survival) — g2 maintained an address
  echo, not the fact; the ghost in anatomy form.
- **STRUCTURAL VOID:** CELL-BASE does not die by +50 → no contrast; record a
  size/family bound on the wash law itself (informative, not a verdict on g2).

Any of F1–F3 kills the design claim; F4/F5 bound it. No bar shopping; every
sub-boolean reported regardless.

---

## 8. COST HYPOTHESIS

- **Parameters:** +0 learnable. Base 0.838M ≤ 1M. Buffers: cue pool 300x256
  int16 + mask 300x255 bool ≈ 165 KB in the state_dict.
- **Compute per wash step:** monitor = one no-grad batch-8 forward every 4th
  step ≈ +2–5% wall-clock. Events replace wash steps 1:1 (batch-size parity
  32 = 32) — no extra optimizer steps, no extra wall-clock beyond the
  monitor.
- **Maintenance events:** predicted 5–25 per 300 steps (e179's economy: 9
  suffice at r=1/32; stickiness 18+ steps ⇒ the refractory-bound rhythm
  undershoots the ceiling).
- **Total compute:** 4 wash cells x 300 steps at 0.84M ≈ 35–90 s each on GPU
  (e157 measured 35 s max) or a few minutes each on CPU; + install and
  jitter-consolidation IF the e157 root fails ROOT-STRENGTH (2 more
  trainings). Whole grid ≈ 6–8 trainings, well inside the e179 dispatch
  envelope (park-once GPU policy, 90 s cooldowns, 1800 s per-training caps,
  torch threads 8, no concurrent GPU). CPU-feasible end to end.
- **Data:** nothing new — data/input.txt + e098 bases + e157's ported
  recipe.

---

## 9. IMPLEMENTATION SKETCH

One file: `lab/g2_rehearsal_native.py` (the lab's one-file-per-experiment
rule). Provenance discipline: instruments copied VERBATIM from
`lab/e179_rehearsal_law.py` (= e180 = e176n/e161/e152/e151/e143/e131/e119/
e113/e068/e043): `load_cpu/evl_load/battery_cell/battery_pz/ce_fixed_cpu/
val_windows/deleted_wpe/read_fact_at/row_census/measure/flat_cells/gate_vs`,
`wash_loss`, `replay_loss`, the neutral-bank builder (G_ANCHOR), the jitter
pool builder (G_JIT), `pick_dev/migrate_to_cpu` (e184's device policy).
Copied, not imported, to own the device policy (e179's convention).

```python
class G2Net(nn.Module):
    # body: TinyGPT (Cfg 4L/4H/128d/256) — byte-identical, all dials run on it
    # cue_pool:   registered_buffer (300, 256) int16   [e113 jitter pool]
    # cue_mask:   registered_buffer (300, 255) bool    [name-target mask]
    # j0_idx:     the 60 j=0 indices (monitor population)
    # gate state: last_event_step, n_events, monitor_trace, event_log

    def monitor(self) -> float:            # no_grad, K_MON=8, j in {0,-4,+4}
        ...                                 # mean p(true name char) over 7*8

    def step_kind(self, step) -> str:      # "wash" | "event"
        if step - self.last_event_step < REFRACTORY: return "wash"
        if step % CADENCE == 0 and self.monitor() < THETA_OPEN:
            return "event"                  # the opening IS the close: 1 event
        return "wash"
```

The training loop = e179's `finetune_rate` with ONE predicate replaced:
`replay = (k >= 1 and step % k == 0)` → `replay = (g2.step_kind(step) ==
"event")`; the event branch calls `replay_loss` on draws from the ORGAN's
pool (identical arithmetic, RNG shapes ix(16,)+aj(8,)+rj(8,) at seed 10902);
the wash branch is `wash_loss` VERBATIM (aj(16,)+rj(16,)). CELL-BASE = the
same loop with the gate hard-disabled; CELL-ECHO = the event branch with
`jit_pool[j=0]` slices and full-token CE (no mask); CELL-SCHED = e179's
`finetune_rate` untouched at k=32.

**Gates (all must PASS before any cell adjudicates):** G_NAMEFREE (zero ZEPH
in every wash draw, checked at draw time), G_ANCHOR (neutral bank 0/16 host
content, 0/16 junctions, seed 170), G_JIT (pool: 300 windows, name in place,
masks vary with jitter, onset band 121–137), G_ROOT (root dials vs own step-0
measurement, self-consistent), G_REPLAY (per event: exactly 16 name windows,
7 masked targets each, 8+8 anchors present), G_STEP_PARITY (300 optimizer
steps, batch 32, EVERY cell), G_ORGAN-INERT (a +300 dial battery with zeroed
organ buffers is bit-identical), G_REPRO (CELL-BASE vs `e157_f2_neutral_s*`
stored cells; device-mixing recorded, e179's rider convention).

**Outputs:** `runs/g2/metrics.json` (per-cell trajectories, full dials,
event logs, monitor traces, all gates, registered prediction verbatim,
deviations list) + `runs/g2/g2_wash.png` — panel A: ruler vs step for all
four cells with gate-event markers (the sawtooth made visible); panel B:
event spacing histogram + per-event pre/post monitor; panel C: anatomy bars
at +300 as % of own root (D-all, row-0, A129, held30, site read).
Checkpoints: `runs/checkpoints/g2_{root,base,g2,echo,sched}[_s{50,300}].pt`.
Smoke mode via `G2_SMOKE=1` (36 steps, trimmed grid, nothing adjudicated).

---

## 10. HONESTY CLAUSES + HANDOFFS

- Single seed (10902), single lineage, n=1 per cell — point estimates until
  replicated (e179's own clause; the wash's seed lottery lives in the tail).
- The monitor's cue forward is a READ, not a rehearsal: no gradient, no
  optimizer contact; only events write.
- RNG: g2's event steps consume extra draws, so CELL-G2's wash windows
  diverge from CELL-BASE's after the first event BY DESIGN (same stream
  DISTRIBUTION, same seed, same step count — e179's cross-cell RNG note
  applies); CELL-BASE's own stream is arm-A-identical for G_REPRO.
- e121's dreams were SELF-GENERATED; CELL-ECHO replays VERBATIM cues — the
  purest in-scope echo control. Whether a GENERATIVE cue source (the decoder
  head, candidate (a)'s engine) can carry the locks too is exactly g3's
  registered question (GENERATIVE MEMORY: hybrid AE store / modern-Hopfield).
  g2's design deliberately keeps the cue explicit so the generative question
  stays clean.
- **g2b extensions (parking lot, in priority order):** (1) re-registration
  at each event (reconsolidation: does a refreshed index maintain better than
  a frozen one?); (2) two-fact cohabitation under g2 (does the internal gate
  double the maintenance budget the way external rehearsal doubled capacity —
  e174's law, made architectural); (3) continuous index-on-surprisal (register
  any high-error window, not just taught facts — the scalable organ).
- **The paper's paragraph if g2 succeeds:** "the one-fact limit and the
  two-step death were never storage failures; a zero-parameter organ that
  indexes its own teaching and re-teaches itself on detected decay keeps the
  whole memory anatomy alive through a stream that kills the base a hundred
  times over — catastrophic forgetting is not a failure mode of these
  networks; it is the absence of an organ they were never given."
