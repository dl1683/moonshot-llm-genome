# g4 — THE DUAL-ADDRESS NET: two address floors, and a gate that predicts which one will carry the fact (design + registration, 2026-09-29, W020 generative turn)

STATUS: design only. Registered predictions BELOW are the commitments —
adjudicate against exactly this text, no bar shopping (e143 convention).
Implementation is a separate dispatch (lab/g4_dual_address.py) AFTER this
doc is committed. No compute was used for this design.

---

## 0. What this is, what it extends (Directive 1 ledger)

The g-series question (W020): are THE ERROR COMPASS and THE VARIANCE
SWITCH architectural necessities or pre-LN-transformer contingencies?

The lab's laws under test (verbatim scopes):
- COMPASS (T076 observational, T084/e143 CAUSAL): facts consolidate where
  their training error is placed; parking the error at rows 5-13 builds a
  site-store AT 5-13.
- SWITCH/CLIFF (T087/e147): memory TYPE flips at zero-vs-any position
  variance — A(w=0)=+0.327 vs A(w=1)=-0.029, NR 0.071→0.696, no width
  trend. LINEAGE-1-SCOPED (T113/e157: family 2's doors all shut; the wash
  replicates, the phase structure does not).
- BASIN (T119/e176N/e184/e187): no robustness basin; the consolidated fact
  dissolves on the first 1-2 gradient steps of a fact-free stream
  (t* ~ lr^-1.16). The lab's most replicated law (2 families x 3 seeds x
  all types x all streams).
- SURGERY (e125a/e160/e164): the deletion hierarchy — table-graft facts
  are surgically removable at flat CE; head-carried fields are not
  (the asymmetry of existence); N2 {L1H0,L0H0} kills at CE +0.25.

BUILDS ON: e043 (install), e048 (install-phase root), e113 (jitter pool +
battery), e119 (locked protocol), e131 (row census), e133 (head census),
e139/e140 (site-test + A-dial census conventions), e141 (d_r0), e142
(the address is protocol-made; row-0-always), e143 (NEAR/FAR compass
arms), e147 (width ladder, A/NR dials, the cliff), e150 (sink-HEALTH
semantics), e160 (N2 flat-CE head kill), e157 (the 0.84M port + the
lineage bound), e176N (neutral wash), W013 (the read policy is the
protagonist — never directly edited), W016 (one native organ, the sink;
addresses are protocol-grown grafts), W017 (variance concentrates head
load), T084/T087/T113 (the laws and their scoping).

WHAT IS NEW (nothing below exists in any prior era; checked against
NOTES/THINKING/REVIEWS/runs/lab/scratch):
1. The lab's first architecture with a SECOND dedicated discrete key
   channel that is NON-POSITIONAL (a content-conditional address codebook
   selected per-token by a small gate). In every net the lab has ever
   dissected, "address" was CONFOUNDED with "position" (the address WAS a
   wpe row). g4 deconfounds them: P = position-only floor (the old wpe,
   instruments verbatim), A = content-conditional floor (token-class x
   position code, K discrete rows).
2. The pre-teaching spine: the gate's position-vs-content sensitivity
   (PSI) and the codebook's slot-usage mass (M) are measured BEFORE any
   fact teaching, and the registered table predicts WHICH FLOOR the fact
   will consolidate onto. The architecture predicts its own memory type —
   literally: an architectural measurement, made before the fact exists,
   foretells the type. No lab instrument has ever read a type-predictor
   pre-teaching.
3. The key-selection rule's INTERFERENCE TERM, made measurable: the
   single-table net could only show reliability (T079-revived, T087); the
   dual net prices the content channel by its corpus usage mass, so
   "which key wins" becomes "reliability x conditionality".
4. W013's read policy made architectural: the gate IS the per-token rule
   deciding which stored coordinate is opened — directly readable and
   directly editable (the GATE-FREEZE knife; the component T037 named
   "never directly edited", edited).
5. The matched-control discipline: a same-size, same-seed, same-recipe
   SINGLE-table control root runs the identical program, so the
   architecture claim is separable from the lineage/scale bound e157
   taught us to fear.

## 1. The necessity question, sharpened

The compass and the switch were measured on nets where the ONLY dedicated
extra channel was the absolute wpe table. Two readings coexist:

- CONTINGENT READING: the laws describe the wpe table's economics. A
  positional lookup row is a perfect predictor under zero variance; any
  variance makes it an anti-predictor (T087's negative posterior). No
  table, no cliff. The memory type is a property of the CHANNEL
  INVENTORY.
- NECESSARY READING: the laws describe credit assignment in any attention
  net. The compass places memory at the error's read-state; the switch is
  a flight from low-order keys (position, single token) to the only
  substrate that can host the invariant CONTEXT key — attention heads.
  Channel inventory changes the rungs, never the flight.

The discriminating intervention is NOT "remove the table" (candidate a,
rotary — see rejection below) but "ADD a competing discrete channel that
is non-positional". If the locked fact still grows on the positional
floor and the variance-flipped fact still lands in heads while a viable
content-address sits unused, the flight is substrate-forced — necessity.
If the fact rides the content floor in any arm, the carrier is
inventory-arbitrated — the laws are real but their CARRIER is contingent,
and the lab gains a new object (the content-address; W016's organ/graft
vocabulary gains a third member).

The structural argument I commit to (the thing g4 tests): a TABLE FLOOR
sees exactly one token (its own char, its own position). The junction
conjunction the jitter regime demands ("this 6-char window-type is a
ZEPHYRA junction") is only expressible through attention. Therefore the
switch's destination is the heads in ANY architecture whose extra
channels are per-token — the law is necessary in the strong sense that
its carrier is forced by what channels can REPRESENT.

## 2. Why candidate (b), and why not (a) rotary or (c) sweep

(a) POSITION-INSENSITIVE (rotary) — REJECTED, three reasons:
  (i) It deletes the instruments. The cliff's CLEAN side is the A-dial
  ("clean of both caveats" — T087); rotary removes wpe, so the row
  census, D-all, the brake, and the address-key dial have no object. The
  experiment would stand on the caveated dials only.
  (ii) The switch question in pure rotary is close to analysis, not
  experiment: the install task's jitter is INVISIBLE in relative
  coordinates. The junction-local content is jitter-invariant by
  construction (batteries at g0 vs g-12 share their last 118 chars;
  jitter pools differ only in far-context chars at distances >118, which
  e013 priced at ~0 nats at char level). Predicting "no switch" would be
  a derivation dressed as a result.
  (iii) Double-removal confound: rotary removes the absolute channel AND
  the discrete-channel-in-general at once; a null cannot say which
  removal mattered.
  (The compass side of (a) IS sharp — NEAR/FAR become context-length
  keys — and is parked as the g4b follow-up if g4's committed branch
  fires: rotary then asks whether the compass survives with NO discrete
  floor at all.)

(c) ARCHITECTURE SWEEP — REJECTED as the primary: e157's own lesson is
that the phase structure is lineage-fragile (family 2's doors all shut
with NO architecture change); a 3-4 delta sweep at n=1 seed/cell mostly
measures lineage lottery. The sweep's minimal informative subset is
exactly the dual-vs-matched-control pair g4 runs (same size, same seed,
same recipe, ONE variable: the A-floor). post-LN and register-token
deltas queue behind the channel-inventory answer.

CHOSEN: (b) DUAL-ADDRESS, with (d)'s spine — the pre-teaching
predictability measurement that makes "the architecture that predicts its
own memory type" literal.

## 3. The architecture (spec)

DualGPT — the lab TinyGPT with ONE change: the input floor carries two
address tables and a gate.

```
class DualGPT(nn.Module):           # base config: 4L / 4H / 128d / 256 ctx
    wte    = Embedding(65, 128)     # unchanged
    wpe    = Embedding(256, 128)    # THE P FLOOR — attribute name kept so
                                    # every state-dict wpe instrument runs
                                    # VERBATIM (row census, D-all, A-dial)
    slots  = Embedding(32, 128)     # THE A FLOOR — K=32 discrete rows,
                                    # content-conditional address codebook
    gate   = Sequential(Linear(256, 64), GELU(), Linear(64, 32))
                                    # input = [wte(x_t) ; wpe(t)] — the
                                    # CURRENT token + its position row.
                                    # NO attention, NO context: the gate
                                    # sees one token (this is the point).
    # forward:
    #   a_t    = softmax(gate([wte[idx_t], wpe[t]]))      # (B,T,32)
    #   x      = wte(idx) + wpe(pos) + a @ slots.weight   # two floors
    #   ... blocks / ln_f / lm_head IDENTICAL to TinyGPT (pre-LN, causal)
```

- Params: base 840,704 (identical to a 4L/4H/128d/256 TinyGPT) + slots
  4,096 + gate 18,528 = 863,328 <= 1M. Control = plain TinyGPT at
  840,704. Delta = 22,624 (2.7%) — capacity confound implausible, stated.
- The gate is TRAINABLE in every stage (no freezes — the architecture
  must be free to use its floors; we test what it does, not what we
  force).
- What the A floor can and cannot represent (load-bearing for the
  predictions): slots can host POSITION keys (if the gate reads wpe) or
  TOKEN-CLASS keys (if it reads wte), including the interaction
  (token x position). Slots CANNOT host context keys — the gate never
  sees neighbours. Only attention heads can. This asymmetry is the
  experiment's engine.
- P keeps row 0 (the sink candidate — e150 semantics apply verbatim) and
  the site rows the instruments know (129/band conventions).

## 4. The pre-teaching spine (measured BEFORE any fact teaching)

On each pretrained root (dual; control has no gate, spine is dual-only):

- PSI (gate position-vs-content sensitivity): over 4,000 corpus
  positions, PSI = median KL(a(x,t) || a(x,t+1)) / median KL(a(x,t) ||
  a(x',t)) with x' a random other char; reported GLOBAL and at the READ
  BAND (cols 120-140, the instruments' read region).
- M(k) (slot-usage mass): fraction of corpus tokens whose argmax slot is
  k; report the mass of the top-3 read-band slots (the interference
  price of loading fact content onto them).
- Registered outcome table (the spine — committed BEFORE measurement):

| pre-teach gate class | predicted INSTALL carrier (locked) | predicted w>=1 carrier |
|---|---|---|
| PSI_read < 0.25 AND M(top read slot) > 2%  | P (positional graft) | HEADS |
| PSI_read < 0.25 AND M < 0.2%               | A (low-interference slot) | A |
| PSI_read > 1 (position-tuned at read band) | P | HEADS (A key position-tied, dies with P) |
| mixed / borderline | spine unsharp as registered — report TEXTURE, no post-hoc bars |

- Structural prior for the first row (why I expect it): the corpus task
  under random crops (common.get_batch) is TRANSLATION-INVARIANT —
  mid-window position carries no corpus information, so pretraining
  should leave the gate content-dominated at the read band, and the read
  char (the char before a host junction — space/punct class) selects a
  HIGH-MASS slot. The spine is a real fork: a position-tuned gate or a
  rare read-slot would flip the table, and both are live mechanisms.

## 5. The program (stages; CONTROL ROOT RUNS FIRST — the scale gate)

Stage 0 — ROOTS (2 pretrains, GPU, chunked):
  dual + control, corpus seed 1337 (data/input.txt), net seed 4305,
  train_model defaults (AdamW 1e-3/wd 0.1, batch 64, cosine), ckpt-resume
  chunks of <=180 s to >=3,000 steps. Gates: val CE <= 1.65 each;
  parity |CE_dual - CE_ctrl| <= 0.05 (remedy if violated: ONE gate-only
  re-seed, recorded; if still violated, record and proceed — the A-floor
  costs CE and the comparison carries the bound). Then the SPINE (sec 4)
  on the dual root — its numbers are committed to the table BEFORE stage 1.

Stage 1 — INSTALL (both roots; e043/e048 protocol ported to 256-ctx/0.86M):
  ZEPHYRA splice, install60/held30 split (SPLICE_RNG 24301), masked
  name-position loss + anchors (batch 32 = 16 install + 16 anchor, 8
  paired + 8 random), 300 steps, lr 1e-3. Gates: p(Z)@g0 in [0.35, 0.75]
  each; dual-vs-control parity <= 0.15. Readouts: P-site census (row
  121-137 band, e139 site-test), A-slot census at read positions, row-0
  presence (e131), head census (e133). THE FIRST REGISTERED CELL: where
  does the install land? (spine table row 1.)

Stage 2 — THE SWITCH LADDER (both roots; e147 minimal ladder):
  arms w in {0 (e119 locked), 1, 8 (e113 jitter pools)} from each
  installed root; 300 steps each; +w=4 if budget permits. Readouts per
  arm: A_P(w) (e140 census VERBATIM on wpe: wpe[129] <- mean-of-rows /
  <- 0, drop on install-60 g0, strength = min(mean-drop, zero-drop)),
  A_A(w) (the SAME census ported to slots: top-3 read-band slots <-
  mean-of-slots / <- 0), NR(w) (e141 d_r0 at g-12 primary, g+12
  robustness; + D-A-top at g-12 as the slot-sink co-report), brake
  (p(D-P-site, g0) - p(none, g0)), head census (top-2 share + positive-
  head count — W017's dial), gate drift (re-measure PSI_read per arm).

Stage 3 — THE COMPASS (dual root primary; control NEAR = the P-replication cell):
  NEAR / FAR arms verbatim (e143: name at x-cols 6-12 / 138-144, zero
  diversity, 300 steps) from the installed dual root. Readouts: e143's
  battery (i) site content test on P rows 5-13 / 137-143 (+ the SAME
  census on A slots), (ii) row-0 presence at install baseline, (iii)
  D-all{121..137}, (iv) novel-geometry g-12/-2/+2/+12.

Stage 4 — THE BASIN (wash; e176N arm A on each w=8 consolidated net):
  neutral stream (plain corpus, no fact windows), lr 1e-3, checkpoints
  {+1, +2, +10, +50}; readouts g-12 / g0 p(Z) + CE per checkpoint.

Stage 5 — THE SURGERY (dual root, locked (w=0) and jitter (w=8) nets):
  cells: D-P-site (trained-band rows <- 0) / D-P-0 (row 0 — the poison
  cell, e150 semantics, CE co-report, never a fact-kill claim) /
  D-A-top3 (top fact-selected slots <- 0) / D-A-all (whole codebook <-
  0) / GATE-FREEZE (swap the gate's weights back to the PRE-TEACHING
  root values at eval — a confined ~18K-param policy edit) / N2-class
  top-2 head ablation (e160 escalation procedure run per-net). CE on
  EVERY cell; flat-CE bar <= +0.35 for any fact-kill claim. Also the
  Z-slot continuation rider (see honesty 9.5).

## 6. REGISTERED PREDICTIONS, PER LAW (the commitments)

P0 — SPINE (the title claim). The gate's pre-teaching class (sec 4
  table) predicts the install carrier AND the w>=1 carrier. FIRES if the
  census-observed carriers match the table's row. DIES if the carrier
  contradicts the table (e.g., PSI_read < 0.25 & M > 2% but the install
  lands A-carried) — then the gate is epiphenomenal to the type decision
  and "the architecture predicts its own memory type" is false as
  stated.

P1 — COMPASS (committed: COMPASS-IS-POSITIONAL). NEAR builds P-site
  content at rows 5-13 (e116 criterion AND strength >= 2x shared-control
  max, e143 convention), FAR at 137-143; A-slots carry no fact content
  (A-arm strength < 0.5 x P-site strength); row-0 presence at/below
  install baseline; NEAR novel-geometry ~0 (site-bound). Falsifiers:
  (a) COMPASS-CONTENT — the NEAR memory rides A-slots (A-arm >= 2x
  P-site): placement steers the content-address; the compass keys on the
  local read-STATE, not on position. (b) COMPASS-DEAD — NEAR ~= FAR on
  every floor (all site strengths < control band): placement inert in
  the dual net; the compass was a single-channel contingency.

P2 — SWITCH (committed: CLIFF-ON-P, FLIGHT-TO-HEADS). On BOTH roots:
  A_P(0) >= +0.15; A_P(1) <= 0 (w* = 1, the step function on the P
  floor); A_A(w) fact-free at every w (strength < 0.10, no slot's usage
  collapses onto junction reads); NR onsets at w=1 (>= 2x root); head
  top-2 share rises w0 -> w8 (W017's form). MECHANISM COMMITTED: the
  flight is P -> HEADS because no table can host the junction
  conjunction (tables see one token). Falsifiers:
  (a) SLOT-SUCCESSION — at any w >= 1 the fact rides A (A_A >= 0.15 at
  CE <= +0.35): a table CAN host the invariant key; "field = heads"
  (e133's 84.5%) dies as a necessity claim; the switch generalizes as a
  flight to whatever non-positional dedicated channel exists. This is
  the outcome that would most reshape the taxonomy — and it is fully
  live: a rare-char slot (Z is corpus-absent) is a low-interference
  invariant key for the CONTINUATION reads.
  (b) NO-SWITCH — A_P flat (max-min <= 0.10) or w0/w1 same carrier ON
  THE CONTROL: the scale/lineage bound (e157 extends to family 3); the
  necessity question re-opens at 2.7M, NOT re-run here.
  (c) NO-SWITCH on dual WHILE control cliffs: the A-floor's mere
  presence unpolarized the competition — inventory-SENSITIVE contingency.
  Adjudication order: committed branch -> SLOT-SUCCESSION -> (c) -> (b).

P3 — BASIN (committed: DISSOLVES-BY-TWO-STEPS, BOTH roots, ANY carrier).
  g-12 p(Z) crosses < 0.05 within (1, 2] steps of the e176N neutral
  stream at lr 1e-3, with the family-1 CE transient shape. The no-basin
  law is optimizer-level and transfers untouched to the new
  architecture. Falsifier: ANY-SURVIVOR (g-12 >= 0.5 at +50 on either
  root) — the first wash-resistant store; W019's falsifier fires; the
  law was architecture-contingent after all. (This cell is W019's
  standing debt — g4 pays it in the new architecture for free.)

P4 — SURGERY (committed: THE HIERARCHY SEGREGATES BY FLOOR).
  (a) locked/P-carried fact dies under D-P-site (>= 60% drop at CE
  <= +0.35) — the graft is deletable, replicating the original address
  surgery. (b) jitter fact survives every TABLE surgery (D-P-site,
  D-A-top3, D-A-all: < 30% drop at any CE) and dies under the N2-class
  head ablation (>= 60% at CE <= +0.35) — the asymmetry of existence
  replicates, now floor-clean. (c) GATE-FREEZE spares the fact whenever
  the carrier is P or heads (< 10% drop; the policy edit is inert on a
  table-carried memory) — and is the killing knife ONLY in the
  SLOT-SUCCESSION world. (d) D-A-all costs the organism (+0.2..+1.0 CE)
  but kills no fact — the codebook serves the corpus, not the memory.
  Falsifiers: any table surgery killing the jitter fact at flat CE
  (e125a's asymmetry inverts); or D-P-site failing to kill the locked
  fact (the graft stops being deletable).

SECONDARY (co-reports, registered as texture, never bars): gate drift
  (does locked teaching RAISE PSI_read and jitter LOWER it? — the switch
  visible inside W013's read policy); slot-sink (does an omnipresent
  content-slot emerge — a content-class sink? D-A-top at g-12 answers);
  Z-slot continuation rider (does the name-CONTINUATION ride the rare
  Z-slot while the onset goes heads? — a split carrier would be the
  first two-floor memory; measured by the per-position census breakdown
  + the Z-slot arm).

## 7. The falsifier, stated once, plainly

The global verdict table (pre-registered):

| control cliff? | dual outcome | verdict |
|---|---|---|
| yes | committed branch (P2) | NECESSITY: compass + switch are channel-inventory-robust; carrier forced by what channels can represent; paper claims 1-2 upgrade |
| yes | SLOT-SUCCESSION | LAWS REAL, CARRIER CONTINGENT: the content-address exists (new object); switch = flight to any non-positional dedicated channel; taxonomy gains a floor |
| yes | NO-SWITCH on dual | INVENTORY-SENSITIVE: offering a channel unpolarizes the competition — contingency of a subtler kind |
| no  | anything | SCALE/LINEAGE BOUND (e157 extends): g4's verdict is the bound itself; necessity re-opens at 2.7M, is NOT re-run |

The deepest single falsifier of the committed branch: SLOT-SUCCESSION.
If a discrete content-conditional table can carry the variance-flipped
fact, then "consolidation is a flight into attention because only
attention sees the conjunction" is false, and the lab's head-centric
field story (e133/e160) was an artifact of the single-table inventory.

## 8. Implementation sketch

- ONE file: lab/g4_dual_address.py (e-series conventions: SMOKE env,
  gates, run_dir("g4"), metrics.json + 3 PNGs: the ladder figure
  (A_P(w), A_A(w), NR(w) — dual vs control), the compass figure, the
  wash trace).
- DualGPT as in sec 3; control = TinyGPT(Cfg(vocab=65, n_layer=4,
  n_head=4, n_embd=128, block_size=256)). Keep `wpe` as the P attribute
  name — deleted_wpe / row census / D-all / A-dial instruments from the
  e113/e131/e139/e140/e147 lineage run UNMODIFIED on state dicts.
- NEW readers (~60 lines, mirroring existing conventions, the only new
  instrument code): slot census (gate distribution at read cols; entropy;
  argmax), A-arm surgery (slots.weight[k] <- mean/0 with the D2
  confinement gate), PSI probe, M(k) usage mass, GATE-FREEZE swap.
- Recipes imported/ported verbatim: e043 splice + find_occ + battery
  builders; e113 jitter pools (JITTERS = {-8,-4,0,4,8} and the ±1 grid
  {-1,0,1}); e119 locked; e143 NEAR/FAR truncation; e147 census
  operationalizations; e176N wash; e160 head escalation.
- Compute: 2 pretrains (<= 3 chunks x 180 s each) + 2 installs + 8-10
  teaching arms (<= 90 s each at 0.86M on GPU; e157 precedent 35 s) +
  2 washes + eval-only censuses (CPU). Total ~35-45 min GPU, every stage
  CPU-parkable (e157/e185c/e187 precedent; SUPERVISOR check-in 10 notes
  outside CPU load — the resource guard pauses rather than migrates).
  NO concurrent GPU (e182 owns the GPU lane until it lands; g4 queues
  behind it or parks to CPU).
- Gates: bit-exact protocol rebuilds (install60 mix
  {FLORIZEL:19, ELIZABETH:41}); CE parity; install parity; confinement
  gates on every surgery cell; fail-late G_INPUTS pattern avoided
  (metrics written after every stage).

## 9. Honesty notes

1. n=1 seed per cell. The CONTROL root is the within-experiment
   replicator for the phase structure; cross-seed replication of any
   firing branch is queued as g4R (the e147R precedent), not assumed.
2. The compass/cliff instruments (e143/e147 A-dial) have NEVER run at
   0.86M/256-ctx — family 1 is 2.7M, family 2's port covered only
   consolidation + wash + the (failed) 2x2 rider. The control root is
   the scale gate and runs FIRST; if it fails, no dual-net claim is
   read as architecture.
3. The committed branch is argued (sec 1, sec 4) but genuinely at risk:
   the rare-char slot loophole (continuation reads on a Z-slot) and the
   gate's freedom to sharpen during teaching are real mechanisms for
   SLOT-SUCCESSION. The prediction is registered as committed precisely
   because it can fail.
4. Single-lineage scope of the laws themselves (T113): g4's dual and
   control roots are a NEW lineage pair (fresh seeds, new architecture);
   any firing branch inherits the n=1-family bound until replicated.
5. The instruments' object is the ONSET dial (p(Z) at the last context
   position). The continuation sub-task could ride different floors
   (the Z-slot rider measures it) — the census reports per-position
   breakdown so a split carrier cannot hide inside the mean.
6. No new automations; no NOTES/THINKING/QUEUE/STATE edits by the
   implementation agent without folding per lab convention; commit this
   design BEFORE implementation dispatch (the registration IS the commit
   ordering — the T082 lesson).
7. If the GPU lane stays contended (outside builds), the whole program
   runs CPU-parked at ~3-4x wall cost; no scientific cell changes.

## 10. The one-line form (for QUEUE.md when promoted)

g4: the dual-address net — a positional table, a content-conditional
codebook, and a gate between them; the gate read BEFORE teaching
predicts which floor carries the fact, and the cliff is tested for
whether its flight (P -> heads) survives an offered alternative
(P -> A). Committed: it survives — no table can host a conjunction.
