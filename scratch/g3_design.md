# g3 — THE GENERATIVE-MEMORY ARCHITECTURE (design, registered 2026-09-29)

W020's generative turn, row 3 of the g-series. THE CHALLENGE (the
active-inference / predictive-processing objection to the lab's lead
finding): our memories are DISCRIMINATIVE READOUTS — attention + MLP
pathways that map context -> p(Z), content smeared into the weights of a
6-layer (here 4-layer) composed function. PP lore says memory should be a
GENERATIVE MODEL — prediction IS storage — and a generative store should
hold a BASIN (an energy landscape with the fact as an attractor), not the
basinless first-order-fragile object T114/T119 described. THE REGISTERED
QUESTION: does explicitly-generative storage change the basin physics, or
is the no-basin law substrate-independent (optimization pressure washes
any single-net store)? Predictions registered IN ADVANCE for both the
wash and the noise kill (below, frozen).

## What it builds on (and what is new)

BUILDS ON: the basin law (T114/T119: no robustness basin; basin ~2.5-5 L2
over 2.7M params; exit t* ~ lr^-1.16, displacement-limited); the noise
kill (E185/T114 + E187/T121: content-free displacement kills at match,
collateral; the no-basin noun now n=3/arm); the resurrection economy
(E179/T120: one replay event resurrects; sticky re-entry); the tenant
frame (W014: memory is a semi-independent tenant; route vs direction
failure modes); the sink-organ view (W016: one memory organ from birth);
the energy-carrier era (T023/e019; e011c's matched-energy ladder — damage
tracks perturbation energy, not content); the closure partition
(e173/e178: class-swap restores decompose a kill into located rewrites);
e157's family-2 port (the 0.87M e098 s4305 line, the neutral wash
RNG-matched, consolidated + washed cells ON DISK); e176N/e185's wash and
noise instrument lineage; the mass-action attractor physics
(scratch/massaction_key_lit.md Claim A — the lab's own prior attractor
stability result). Mechanism reference for the store: modern Hopfield /
RPMH (Ramsauer et al., "Hopfield Networks is All You Need" — retrieval is
one softmax step; attention IS a Hopfield update).

NEW (nothing like it in the lab's history or the git prior-era): (1) the
first DESIGNED memory organ — the fact stored as an explicit PATTERN
MATRIX (a 7-step generative program for the name's token sequence) in a
Hopfield retrieval layer grafted into the trunk, rather than as a
distributed pathway; (2) the TWO-BASIN MAP — the experiment that
separates the ENERGY LANDSCAPE'S STATE BASIN (query-space, where PP lore
says memory lives) from the PARAMETER BASIN (weight displacement, where
forgetting actually happens); (3) the KILL-SITE CENSUS at organ
granularity (gain / query / gate / route) via a store-class closure
partition; (4) the DECOUPLING CONTROL (arm D) that delineates the law's
boundary: survival by blindness vs survival by basin.

## Why the Hopfield spine (and how candidates (a)/(c) were disposed)

- (a) the hybrid AUTOENCODER store: rejected as the spine because its
  decoder puts content IN WEIGHTS — a trained regenerator with
  first-order parameter sensitivity is the discriminative basin physics
  wearing a generative costume; its spirit is kept as the S-SHAL control
  (below), which isolates exactly that confound.
- (b) the ENERGY-BASED ATTRACTOR: adopted as the spine — it is the only
  candidate whose storage is an EXPLICIT OBJECT (the pattern matrix) with
  a measurable landscape, and whose retrieval is locally constant in
  parameter space (an argmax-preservation plateau) — the one mechanism
  that could plausibly buy basin width. The sequential character of (a)
  is absorbed: the store's patterns form a 7-step generative program of
  the name (see below), so the fact is stored AS the prediction that
  regenerates it — prediction-as-storage, implemented literally.
- (c) own mechanism: the composite below (pattern store + shallow-discriminative
  twin + decoupling control + two-basin map) IS the custom design; every
  degree of freedom except the storage mechanism is matched between arms.

## THE SPEC

### The host (frozen choice)

Family 2: the e098 s4305 line (F2_CFG = 4L/4H/128d/512-ctx,
F2_PARAMS = 873,472). WHY: (i) the <=1M total envelope: host 0.87M +
store ~17.4k = ~0.89M <= 1M; (ii) the DISCRIMINATIVE CONTROL IS ALREADY
ON DISK AT ZERO COST — e157_f2_consolidated.pt + its neutral-wash cells
(the fact dissolves at +1: g0 0.578 -> dead; g+12 0.591 -> dead); (iii)
family-2 keeps the claim clear of lineage-1's site idiosyncrasies (its
rider doors all shut; no 183-site confound at root); (iv) e157 already
RNG-matched the wash protocol to this family. The host base for the
graft: e098_base_s4305.pt (the pristine fact-free base — the fact must
be carried by the STORE, not by a pre-installed discriminative pathway).

### The store (the generative organ) — g3-GEN

One position-wise Hopfield block grafted AFTER block 3, BEFORE ln_f
(the output-side organ; the read from injection to logits is exactly
ln_f + lm_head — the shallowest possible route, which deliberately
removes routing depth from the kill candidates and centers the question
on the store itself):

  q_t  = W_q . LN_noaffine(h_t)              # 128 -> 64, position-wise,
                                             # causal by construction
                                             # (h_t sees only tokens <= t)
  a_t  = softmax( beta * cos(q_t, K) )       # K: p x 64 pattern keys,
                                             # cosine Hopfield; beta = 8
                                             # FIXED (a design constant,
                                             # not a washable param)
  r_t  = a_t . V                             # V: p x 64 pattern values
  inj_t = W_o . r_t                          # 64 -> 128
  h_t  <- h_t + inj_t                        # residual injection, every
                                             # position

PATTERNS (p = 8): SEVEN FACT KEYS — one per name step j = 0..6. Key j is
the context prototype at name position j (the query distribution of
install windows at the position that predicts name char j); value j
decodes (through W_o) to the content that raises name char j's logit at
that position. The store is therefore a MICRO-GENERATIVE MODEL OF THE
NAME: given the pre-host context it generates Z; given ...Z it generates
E; ... ZEPHYR -> A. The battery's p(Z) reads step 0; the site-read span
instrument (read_fact_at, rows 183..189) reads all 7 steps verbatim.
ONE NULL PATTERN: key = the anchor-query prototype, value = 0 — the
gate. Non-matching (neutral) queries retrieve null and inject ~nothing.
SOFT gate (softmax tails): the honest single-net store — the tails leak,
and the leak is load-bearing for the registered mechanism (below).

INIT (the one-shot write): W_q small random; compute the install-window
queries; K_j init = the class mean of step-j queries, null key = the
anchor-query mean, V_null = 0; V_j random. Storage begins as a WRITING
operation (Hebbian-style closed form), then gradient-refined — the
Hopfield way.

PARAMS: W_q 8,192 + K 512 + V 512 + W_o 8,192 = 17,410 (~17.4k).
TOTAL NET: 873,472 + 17,410 = 890,882 (~0.89M <= 1M). All store tensors
form one parameter CLASS ("store": the 4 tensor keys) for the e173
class-swap instrument.

### The shallow-discriminative twin — g3-SHAL (the contrast control)

IDENTICAL graft point, query source (LN -> W_q', 128->64), injection
form (W_o', 64->128, pre-ln_f residual), training data, loss, and
budget. ONLY THE MIDDLE DIFFERS: a 2-layer MLP (64 -> 48 GELU -> 64)
replaces the pattern retrieval — the map context -> next-name-char is
computed DIRECTLY, content IN WEIGHTS, no explicit stored pattern, no
retrieval plateau. Params ~22.6k (30% over the store; exact counts
reported — the asymmetry is recorded rather than padded away: filler
patterns would change the retrieval geometry). S-SHAL is what candidate
(a) reduces to once its decoder is honest — the AE's spirit, controlled.

### The decoupling control — g3-HARD (arm D)

The g3-GEN construction with the gate HARDENED: top-1 argmax retrieval
(straight-through gradients). Neutral queries select the null branch
EXACTLY (zero leak); the fact key/value receive gradient only from fact
windows — under the wash they are gradient-BLIND (weight decay only).
Arm D does not test a wider basin; it tests the COUPLING: if the soft
store dies by the leak channel, the decoupled store must survive; if arm
D dies anyway, the kill entered through a shared door (W_q drift / null-
key encroachment) and the law's entry point relocates.

### The interface guarantee (the constraint that binds)

The store lives INSIDE the forward pass, position-wise and causal —
every existing instrument reads the g3 net UNCHANGED: battery_cell on
the install-60/held-30 g-12/g0/g+12 contexts (p(Z) at the last position
is fed by the store's step-0 generation at exactly that position);
read_fact_at site/span reads; CE_R via val_windows; the neutral bank +
finetune_freeze wash (e157's family-2 port, seed 10902); e185's noise
arms + the displacement currency (now over ~0.89M params + a store-
subspace co-currency). The ONLY mechanical deviation: the model
constructor (GenMemGPT, a TinyGPT subclass carrying the store) —
copied-not-imported per house convention, recorded. The primary dial is
g0 (family-2's honest dial — its g-12 root was under-bar even
discriminatively, e157's recorded asymmetry); g-12/g+12/held30/span are
co-reports.

## THE PROTOCOL

STAGE 0 — CONSTRUCTIONS (3 trainings, each <= 180 s, host frozen):
store-only training on the e043 recipe verbatim (16 install windows +
32 random corpus anchors per step, token-weighted union CE — 112
name-char targets vs 8,160 anchor targets; the anchors train the null
gate for free); AdamW lr 1e-3, 200-400 steps, seed 43050; held30 never
trained. If the root gate misses on expression, ONE brief joint
calibration (host+store, <= 100 steps, lr 3e-4, CE_R gate) is allowed
and recorded. Root gates (frozen):
  - G_ROOT_EXPR: g0 >= 0.50 (install-60); held30 g0, span onset/over-7
    co-reported.
  - G_STOREOFF: with the store output zeroed at eval, g0 <= 0.27 — the
    fact is STORE-carried; the host did not learn it (this gate is what
    makes the generative/discriminative contrast valid).
  - G_CECLEAN: CE_R <= base + 0.10 (the tenant pays its rent — W014).
  - G_NAMEFREE / G_SPLICE / G_POOL / G_ANCHOR: e176n/e157 verbatim.
  - CO-REPORT (no bar): the wpe row census (no new fact site — the
    discriminative graft must not have regrown) and the GEO PROFILE
    (predicted FLAT-ish for the store: the keys are trailing-content
    keyed, so g-12/g+12 should track g0 — the store's wild state-basin
    read; the discriminative profile was position-textured).

STAGE A — THE WASH (3 x 300 steps, e157's ported e176N arm A verbatim:
16 neutral anchors + 16 random, full-token CE, AdamW (0.9,0.95) wd 0.1,
lr 1e-3, clip 1.0, seed 10902; checkpoints {1,2,4,10,50,100,200,300} —
+10 added for the noise displacement-match grid; lean battery per
checkpoint, full dial at root/+10/+300):
  A1 g3-GEN (THE cell), A2 g3-SHAL (contrast), A3 g3-HARD (decoupling).
  The discriminative baseline (S-DISC) needs NO compute: e157_f2 cells.

STAGE B — THE NOISE KILL (e185 ported, on g3-GEN): NOISE-LABELS (iid
uniform targets, seed 18501) and SHUFFLED-TARGET (permuted targets, seed
18502), 10 steps, checkpoints {1,2,4,10}, inputs bit-identical to A1
(per-step hash gates), displacement-matched to A1's own D_kill (the
cumulative all-param displacement at A1's first-under-bar step);
damage read at M and every later matched checkpoint through +10.

STAGE C — THE EVAL BATTERY (eval-only):
  C1 THE KILL-SITE CENSUS (at t* = A1's first-under-bar): retrieval
     state on fact queries (argmax identity, a_fact mass, ||inj||
     gain vs root); per-tensor displacement (K, V, W_q, W_o — which
     tensor spent the displacement budget); the QUERY ATTRIBUTION 2x2
     (root-q x washed-K vs washed-q x root-K — did the query leave the
     key or the key leave the query).
  C2 THE CLOSURE PARTITION (e173's restore_class with CLASSES['store'],
     weight-level, 2x2): {root, washed} store x {root, washed} host on
     the t* checkpoint. Root-store-into-washed-host = THE BYPASS PROBE:
     restores >= 0.5 => the route is intact, the STORE died;
     fails => ROUTE-KILL (the alphabet itself moved — ln_f/lm_head).
     All-restored = the bit-exact sanity gate.
  C3 THE TWO-BASIN MAP: (i) STATE BASIN — interpolate root fact queries
     toward corpus queries (gamma sweep) + gaussian query noise (sigma
     sweep); retrieval fidelity and readout vs perturbation. (ii)
     PARAMETER BASIN — lambda-sweep the store's tensors along A1's
     MEASURED wash direction (theta_root + lambda . (theta_t* -
     theta_root), lambda in {0,.25,.5,1,2,4}) AND isotropic parameter
     noise at matched L2 (e011c's matched-energy ladder on the store's
     subspace); readout vs store-displacement. Output: the overlay map
     (S-DISC survival-vs-displacement from the e157/e180 cells; g3-GEN
     store-subspace; g3-GEN host-subspace; g3-SHAL) — the paper's
     exhibit.
  C4 THE RESURRECTION RIDER (report-only, e179's protocol): at t*+1,
     ONE replay batch of the fact's windows, then continue the wash 50
     steps; read g0. The re-entry economy on the generative substrate.

## REGISTERED PREDICTION (frozen before compute)

THE WASH — G3-DIES, THE LAW HOLDS: the soft-gated generative store
dissolves with the law — g0 <= 0.27 by +50 (the arc's NEUTRAL-DISSOLVES
bar), point prediction for the clock: first-under-bar in (2, 20] — the
plateau buys at most a small-factor discount off the discriminative +1,
never survival. MECHANISM (the point prediction): the kill site is the
GAIN — the soft gate's exponential tails leak name content onto every
neutral token; the neutral CE pays for the leak; and because AdamW
moves at ~lr per coordinate REGARDLESS of gradient magnitude (the
e185 displacement-matching methodology's own observation), the small
but sign-consistent leak gradient collapses W_o/V at the standard
clock. THE SAVOR PREDICTION: at the kill the RETRIEVAL IS STILL INTACT
(the fact key still wins the argmax on fact queries; the energy
landscape still holds the pattern) — STORAGE WITHOUT EXPRESSION, the
exact inverse of the lab's oldest dissociation. THE TWO-BASIN
DISSOCIATION: the state basin will measure WIDE (retrieval survives
large query perturbation — Hopfield basins by construction) while the
parameter basin measures NARROW (the readout dies at wash-direction
store-displacement comparable to the discriminative kill displacement).
One line for the paper: THE WASH DOES NOT TRAVERSE THE ENERGY
LANDSCAPE — IT RE-SCULPTS IT. PP lore conflates the basin where
retrieval lives (state space) with the basin where forgetting happens
(parameter space); g3's map separates them.

THE NOISE KILL — NOISE-KILLS: both noise arms kill g3-GEN at
displacement-match (a fortiori below), COLLATERAL (CE >= 3.0, the
organism dies with the fact) — no generative immunity: the plateau
protects against small displacements, and matched displacement exceeds
it; undirected damage is not gain-selective. NOISE-SPARES at match
would resurrect a corpus-directed mechanism clause for the generative
substrate (the no-basin law's content-free clause breaking).

ARM D — D-SURVIVES (g0 >= 0.50 through +300) IF the GAIN channel is
the kill (the leak removed, the fact pattern gradient-blind, weight
decay ~3% over 300 steps is sub-lethal); D-DIES relocates the law's
entry point to a shared door (W_q drift or null-key encroachment —
the census adjudicates). Arm D's survival is SURVIVAL BY BLINDNESS,
not by basin — it delineates the law's boundary: the no-basin law
binds what the gradient can see.

S-SHAL — dies in the same bracket as g3-GEN (within ~2x of its clock):
the leak/gain channel applies identically to a continuously-firing MLP;
no plateau, no discount. If g3-GEN >> g3-SHAL survival, generativity
bought real time (the PLATEAU DISCOUNT becomes the finding).

THE RESURRECTION RIDER — fires: one replay event restores g0 >= 0.5
within 50 wash steps (the re-entry economy is substrate-independent;
memory as RHYTHM holds for the organ too).

## THE FALSIFIERS

- THE LAW BREAKS (my prediction dies) if: g3-GEN's g0 >= 0.50 at EVERY
  continuation checkpoint {1,2,4,10,50,100,200,300} at lr 1e-3 —
  explicitly-generative storage changed the basin physics. The
  follow-up is then built in: g3-SHAL's fate discriminates GENERATIVITY
  (S-GEN alive, S-SHAL dead) from SHALLOWNESS (both alive — the
  no-basin law was a deep-pathway law all along, and W020's
  contingency question answers "contingent on pathway depth").
- The MECHANISM prediction dies if: the kill site is not the gain (the
  census shows retrieval collapse, query drift, gate flip, or route
  death instead), or the retrieval is NOT intact at the kill (no
  storage-without-expression texture).
- The NOISE prediction dies if: either noise arm spares g3-GEN at
  displacement-match.
- The COUPLING claim dies if: the gain channel fires on g3-GEN but
  g3-HARD also dies (the leak was not the channel).
- The state-basin claim dies if: the query sweeps show a narrow state
  basin (then the Hopfield store was never an attractor store in
  practice — a construction finding, honestly reportable).

## COST HYPOTHESIS

5 trainings, each far under the 180 s cap: 3 constructions (~120 s
each, host frozen — store-only is cheap) + 3 washes (300 steps; e157's
family-2 trainings ran 35-60 s) + 2 noise runs (10 steps, ~seconds).
Everything else is eval-only: ~30 checkpoint batteries (lean ~30-60 s
CPU each), the closure partition, the two-basin sweeps, the census, the
rider. ZERO new data; the S-DISC control is free (e157's stored cells).
TOTAL: ~60-90 min wall-clock, GPU sequential with gpu_ok gates + 60-90 s
cooldowns (e157 precedent) or CPU fallback (e176n precedent, threads 4).
Params ~0.89M <= the 1M design envelope, and 2 orders under the 100M
free tier. Outputs: runs/g3/{metrics.json, g3_basin_map.png};
checkpoints runs/checkpoints/g3_{gen,shal,hard}{,_sN}.pt.

## IMPLEMENTATION SKETCH

One file: lab/g3_generative_memory.py (one file per experiment; the
docstring freezes the bars + this registration verbatim; no bar
shopping).

  - class StoreBlock(nn.Module): W_q/LN/K/V/beta(8, buffer)/W_o;
    forward(x) -> x + W_o(softmax(beta*cos(x@W_q_normed, K_normed))@V);
    hard=True variant (top-1 + straight-through) for arm D.
  - class GenMemGPT(TinyGPT-subclass): the family-2 trunk from
    e098_base_s4305.pt; store called between the block loop and ln_f;
    state dict = trunk keys + 4 store keys (the "store" class for
    e173's restore_class, copied verbatim; CLASSES['all'] extended and
    the coverage gate re-run).
  - Instruments copied verbatim from e176n/e157/e185 (battery_cell,
    battery_pz, ce_fixed_cpu, val_windows, read_fact_at, row_census,
    finetune_freeze with lr/ckpt params, the neutral bank + junction
    accounting, the noise-arm target generators + per-step input-hash
    gates, displacement + cosine accounting) with the recorded
    deviations: (1) the model class; (2) displacement additionally
    reported on the store subspace + per store tensor; (3) the +10
    checkpoint; (4) the store-off eval hook (zero inj via a forward
    flag, no weight edits — the G_STOREOFF gate).
  - Root gates before any wash (G_ROOT_EXPR, G_STOREOFF, G_CECLEAN,
    G_NAMEFREE, G_SPLICE, G_POOL, G_ANCHOR) — a failed root gate aborts
    that arm to CONSTRUCTION-CEILING (reported; no wash adjudication).
  - Adjudication exactly against the registered clauses; the census
    definitions frozen in the docstring (GAIN: argmax intact, a_fact
    >= 0.5 on fact queries, ||inj|| <= 30% of root; QUERY: a_fact <
    0.5 or argmax flipped with the key geometry intact (root-q
    attribution); GATE: the null key encroached (washed-K attribution);
    ROUTE: bypass fails to restore >= 0.5 with the store's isolated
    forward intact; else MIXED).
  - Executor duties per house convention: NOTES.md entry, the T-card
    (2+ alternative explanations + discriminating observation +
    registered prediction — this document supplies the prediction),
    QUEUE g3 row -> DONE, STATE.json, single commit + push.

## HONESTY NOTES (pre-registered)

Single seed per construction/wash (replicates owed before any noun
moves); one lineage (family 2; a family-1 g3 follow-up owed only if the
law breaks); beta/d_key/placement are design constants, not swept (a
sweep is g4's architecture-axis business); the S-SHAL count asymmetry
(+30%, reported); S-DISC is a cross-construction baseline (e157's
consolidated root vs our grafted roots — S-SHAL is the
same-construction control); displacement currency is unweighted L2
(the store-subspace co-currency quantifies what the aggregate hides);
the corpus direction's surgicality is expected to hold for the g3 wash
(CE transient co-reported); if the joint calibration runs, G_STOREOFF
is the gate that keeps the contrast honest.
