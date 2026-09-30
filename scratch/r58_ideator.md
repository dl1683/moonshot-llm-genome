# R58 IDEATOR — the three cells the freshest results demand (opt2 / e188-confirmed / g9)

2026-09-30, frontier review R58. Inputs read: AGENTS.md, THINKING.md
(T135–T140, W020–W023 incl. W022b), NOTES.md (g1bW, opt1, g3K, the
restart note), QUEUE.md, REVIEWS.md (R56/R57), STATE.json (fleet:
g1bS GPU + opt1b CPU + R58 trio), scratch/r56_ideator.md in full
(g6/g7/g8 specs — the collision baseline for the architecture cell).

## What the three freshest results opened (one paragraph, so the cells are legible)

opt1 (T139) split the wash kill into CLOCK (Adam's arithmetic:
1.6543/step at lr·sqrt(N), sign-normalized — 1687x matched-lr SGD)
and GATE (displacement D ~ 2.49–2.84 in every Adam arm), and found
the killer stream TEACHES under SGD (0.916 -> 0.955) — "the stream
chooses signs, the normalizer chooses the clock." g3K (T137) found
the no-basin law is a TRAJECTORY law: static isotropic needs 4–10x
(a graded basin exists vs random directions), every learned path
kills at 1x, and the killing object is a DRIFTING aligned front
(cos d1..d300 ~ 0.13). g1bW (T140) found the wall's tax relocated to
the ONSET channel: A survived an ACTIVE second install (0.83, min
0.65) while B's new-structure formation was taxed (partial-form 0.21
walled vs 0.53 unwalled) — inside-the-ball protected,
outside-the-ball resisted. Three open doors: (1) WHICH property of
the sign-normalized step is lethal (sign? density? fresh-state
uniformity?); (2) is death priced in ALIGNED displacement (the
integral W022b specified is now pointed by result, not argument);
(3) can a wall protect the old fact WITHOUT blocking the new one —
is the full ball's point-locality the design error?

## Collision ledger (checked 2026-09-30 ~12:00Z)

DISPATCHED: g1bS (GPU, C13-2 slot 1), opt1b (CPU, the direct SGD
kill). QUEUED/SPEC'D: g1bW2 (GPU, behind g1bS/g2g — the B-dose
finder), e182c (CPU eval-only), g2g (GPU, C13-2 slot 2), g2h, g6
(READY, bars frozen — the FUNCTION-space tube on the g3 store
organism), g7 (behind g6), g8 (CPU-friendly census legs), g1c-root,
g3O, e188 (spec'd in W022b + T137/T138/T139 additions, unregistered
as a cell). NONE of the three cells below collides: opt2 is the
mechanism decomposition of the Adam step itself (opt1b asks whether
the gate generalizes to raw-gradient paths — orthogonal; no
signSGD/top-k/masked-update cell exists anywhere in QUEUE or
THINKING); e188 consumes stored checkpoints only; g9 walls
PARAMETERS on the g1b host-fact lineage — g6 walls ACTIVATIONS on
the g3 store organism (different space, different organism,
different question: cheap protection + noise-wound vs sequential
admission), and g7 composes wall+rhythm+cone with no admission
question. g9's prerequisite is g1bW2's dose finding — it queues
behind it, not beside it.

---

## opt2 — THE SIGN CARRIER: which property of the Adam step kills?

### (a) Builds on

- **opt1/T139** (verbatim numbers): fresh AdamW's first step is
  exactly ±lr on every coordinate (m̂/sqrt(v̂) = sign(g) at step 1,
  bias-corrected); per-step L2 1.6543 = lr·sqrt(N); warmup stretched
  the clock 10.08x with the kill at the same D within 15% (the
  stretch-invariance precedent this cell reuses); beta2 and moment
  inheritance carry NOTHING (A4 indistinguishable, A5 bit-identical).
- **opt1b (dispatched)**: whether the D ~ 2.5 gate generalizes to
  raw-gradient (SGD) paths. opt2 is the orthogonal decomposition:
  GIVEN a fast path kills, WHICH structural property of it is the
  carrier? Either opt1b answer sharpens opt2's framing (if SGD is
  spared at the gate, SIGN is the bridge arm between the classes).
- **g3K/T137**: the killing object is a drifting aligned front;
  wash-1x is magnitude-uniform. The TOPK arms test whether a
  |g|-selected sparse front — the optimizer-real version of g3K's
  front — suffices.
- **W022**: cos(grad g0, wash) = -0.44 — the alignment convention
  the co-reads reuse. **W023**: the shock-and-recover curve wants
  per-step alignment (the summary read was flat-negative, -0.015..
  -0.105; the curves decide ADAPTATION-SHARPENS).
- **e185/e187 + opt1's CPU precedent**: the licensed wash cell
  verbatim, optimizer swapped, 7–8 arms CPU-only, A0 bit-repro gate.

### (b) What is new

Nobody in the lab has intervened on the optimizer step's STRUCTURE.
opt1 swapped optimizers whole; opt1b changes trajectory class via
lr. opt2 holds the corpus, the net, the seed, and the per-step L2
budget FIXED and varies only WHICH coordinates carry the update and
in what arithmetic — separating three candidate carriers that
opt1's arms could not separate: (i) the sign DIRECTION (magnitude-
blind ± movement), (ii) the DENSITY (every-coordinate motion vs a
sparse front), (iii) the fresh-state UNIFORMITY (±lr blindness to
per-coordinate scale, vs install-calibrated adaptive scaling).
Collision check: no signSGD, top-k, masked-update, or warm-moment
arm exists in QUEUE.md or THINKING.md; A5 (moment RESET) was the
complement (state ignored) — WARMV is the anti-A5 (state CALIBRATED
to the install); fresh.

### (c) The single knob and the discriminating observation

**Knob: the update MASK/arithmetic structure at matched per-step
L2.** Arms (e185 wash cell verbatim, same inputs md5-parity, same
seed; D is the registered clock per T139; all arms at lr/8 so the
gate is crossed at ~step 12–16 with dense checkpoints — the warmup
invariance licenses the stretch, one arm at lr_base verifies):

| arm | update | question |
|---|---|---|
| REF | AdamW fresh (lr/8) | the stretch control (G-STRETCH) |
| REF-RAW | AdamW fresh (lr_base) | kills ~step 1.6; the D-band anchor |
| SIGN | -lr·sign(g) (lr matched: same per-step L2 by construction) | is step-1 arithmetic the WHOLE lethal object? |
| TOPK-10 | Adam step applied to top-10% \|g\| coords only, lr/sqrt(0.10) to match L2 | sparse front suffices? |
| TOPK-50 | same at 50%, lr/sqrt(0.50) | the density ladder's midpoint |
| WARMV | AdamW with v initialized from the install battery's E[g²] (one CPU pass at the anchor) | does calibrated scaling defuse ±lr? |

SIGN is exactly fresh-Adam's step-1 rule forever; if SIGN kills
identically in D, everything Adam computes beyond sign is innocent.
**Discriminating observation**: D-at-kill per arm against the
2.0–3.2 band, PLUS the per-step alignment curves |cos(δθ, grad g0)|
against CE_R recovery — W023's knife-sharpening question rides the
same runs for free.

### (d) Registered bars (draft)

- GATES: G-A0 (bit-repro of e185's stored control, opt1's standard);
  G-INPUTS (per-step md5 across arms); G-STRETCH ("REF kills in the
  D-band 2.0–3.2 — the warmup invariance replicates at lr/8; if it
  fails, REF-RAW becomes the reference and the stretch caveat is
  reported, not shopped").
- SIGN-SUFFICES: "fires if SIGN kills in the D-band — the ±lr
  arithmetic is the entire lethal object; moments and adaptive
  scaling carry nothing (completing A4/A5)."
- SPARSE-FRONT-KILLS: "fires if TOPK-10 kills in the D-band at
  matched per-step L2 — lethality is CONCENTRATED in a |g|-selected
  front; g3K's drifting front becomes optimizer-real and the
  displacement gate gains a subspace rider."
- DENSE-REQUIRED: "fires if TOPK-10 and TOPK-50 hold the fact >= 0.5
  through D = 3x the band while SIGN kills — the basin dies of
  every-coordinate TOTAL displacement; the mechanism noun is
  'everything moved', not a front."
- CALIBRATED-DEFUSE: "fires if WARMV holds the fact >= 0.5 through
  D = 2x the band — the fresh-state ±lr uniformity is the carrier
  (blindness to per-coordinate scale), and install-calibrated
  adaptive scaling is protection."
- Co-read ADAPTATION-SHARPENS (W023): "fires if per-step
  |cos(δθ, grad g0)| RISES with CE_R recovery within any killing
  arm — the knife sharpens; flat alignment through the recover
  means the clock was magnitude, not alignment."

### (e) Honest failure modes

- Matching per-step L2 via lr rescaling shifts LATER Adam dynamics
  (moments warm differently) — mitigated by registering D-at-kill,
  never step-at-kill, and co-reporting realized per-step L2.
- The top-k set churns step to step (the mask follows g) — that is
  the drifting front, a read not a bug; report inter-step set
  overlap.
- WARMV's v is estimated from the install BATTERY, not the full
  install history — report the per-coordinate ratio distribution;
  if v is badly off, CALIBRATED-DEFUSE's null is ambiguous between
  "scaling protects" and "our v was wrong" — stated.
- n=1 per arm, one root, one stream, CPU fp32 texture (A0-gated) —
  opt1's standing honesty clause, inherited verbatim.
- If opt1b lands SGD-KILLS-AT-GATE, DENSE-REQUIRED becomes the
  leading hypothesis (SGD is dense and kills) and SIGN vs TOPK
  still separates direction from density; if SGD-SPARED, SIGN is
  the bridge arm — the bars need no edit either way.

**Lane/model**: CPU only (opt1 precedent, 6 arms, chunk-resumable);
the e185 2.7M host verbatim — the smallest net that carries the
licensed wash cell. Zero new parameters. ~100–200 steps/arm at lr/8.

---

## e188 — CONFIRMED (with one amendment): the alignment integral, the death currency, and the vocabulary

### Verdict: still the right next e-cell — dispatch first (see order below)

W022b + T137/T138/T139 have left it fully spec'd AND pointed by
result: g3K's trajectory-vs-static split is exactly what the
integral separates "by construction" (T137's words); T138 added the
install-vs-wash cosine that decides the paper's vocabulary (Ilharco
task arithmetic vs corpus-adaptation); W023's per-pair alignment
curves fall out of the same estimator. Costs: eval-only, CPU, tens
of backward passes on the e180 lr-grid snapshots (s2/s10/s50/s100/
s200 per arm, on disk). No training, no lane conflict.

### (a) Builds on

- **W022b**: the estimator (a_t = cos(d_theta, grad g0_t);
  A = Σ a_t·||d_theta||), the three-way fork, the snapshot
  inventory. **W022**: cos -0.44 — the quantity the integral
  time-integrates.
- **e180/T119**: the lr grid {1e-5, 3e-5, ...} with t* known per
  arm; t* ~ lr^-1.16, displacement-limited — the rate law the
  integral either explains or demotes to corollary.
- **g3K/T137 + stored wash/iso snapshots**: the static-kill rungs
  are constructible eval-only (seeds verbatim in metrics) — see the
  amendment.
- **T138/T139**: the alignment co-read's summary values (-0.015..
  -0.105) vs the store's -0.44 — the per-arm curves adjudicate.

### (b) What is new (the amendment)

The spec'd cell answers "is death priced in aligned displacement?"
on the TRAJECTORY side only. ADD THE STATIC CONTRAST ROW: compute A
for g3K's isotropic rungs (theta_washed + rung·iso_vector, seeds
verbatim): predicted A_iso ~ 0.001·rung ~ 0.004–0.008 vs a
trajectory A* of order 1–2.5 — the two-currency structure of T137
in ONE plot: trajectory death priced in aligned units, static death
in raw collateral units. This converts g3K's qualitative
"trajectory law" into the paper's quantitative centerpiece figure
and is the cheapest possible discharge of g3K's WHAT'S-NEXT (c).
Collision check: nothing queued computes A on static rungs; e188 is
named but unregistered — this confirms and completes, it does not
duplicate.

### (c) Knob and discriminating observation

No intervention (eval-only): the "knob" is the CURRENCY in which
per-arm t* is expressed (A vs raw ||d_theta||). Discriminating
observation: the spread of A(t*) across lr arms vs the spread of
raw D(t*) — and the iso row's near-zero A against whichever
currency wins.

### (d) Registered bars (draft, consolidating W022b/T138)

- ALIGNED-CURRENCY: "fires if all e180 arms die at lr-independent
  A* (spread <= 25%) — the rate law is a corollary of constant-
  speed aligned drift; no-basin sharpens to no-basin-on-the-
  aligned-ray."
- RAW-CURRENCY: "fires if t* tracks raw ||d_theta|| better (R²
  gain >= 0.1 over A) — alignment is epiphenomenal; e185's
  displacement story was already the whole law."
- ALIGNMENT-DRIFTS: "fires if per-arm alignment shifts monotonically
  with lr — the organism's adaptation changes fact-erodingness with
  speed; its own finding."
- TWO-CURRENCY (the amendment): "fires if the g3K iso rungs kill at
  A < 2% of A* — static and trajectory kills priced in different
  currencies; the trajectory law's quantitative statement."
- TASK-ARITHMETIC co-read (T138): "fires if cos(install_dir,
  wash_dir) <= -0.7 — forgetting here IS task arithmetic; the paper
  adopts the vocabulary. Else the wash is the corpus's adaptation
  direction, not the fact's negation — either answer decides the
  framing."
- ADAPTATION-SHARPENS co-read (W023): per-pair |cos| vs CE_R
  recovery on the e180 trajectories.

### (e) Honest failure modes

- The integral is a 5-point quadrature (s2..s200) — coarse between
  s2 and s10 where the action is; bounded and stated (no re-run
  shopping; the e180 checkpoints are what they are).
- grad g0 is computed at SNAPSHOTTED theta, not continuously — the
  estimator assumes piecewise-linear adaptation; the assumption is
  named on the figure.
- Single lineage (e180's); the iso regeneration depends on g3K's
  committed seeds (stored verbatim — regenerable, G-SEEDS gate).
- If ALIGNED-CURRENCY and TWO-CURRENCY both fire, the temptation to
  say "alignment CAUSES death" — the cell shows pricing, not
  causation; the causal cell is opt2's masked arms. Stated in the
  card.

**Lane/model**: CPU eval-only, minutes-to-a-quarter-hour; the e180
2.7M grid + g3K's 0.89M organism snapshots. Zero parameters, zero
training.

---

## g9 — THE ADMISSION BALL: a cone-shaped wall that protects the old fact without blocking the new one

### (a) Builds on

- **g1bW/T140** (verbatim numbers): the wall's real sequential cost
  is NEW structure formation — B partial-form 0.21 walled vs 0.53
  unwalled; A held 0.83 through an ACTIVE second install; A's row0
  protected (0.62–0.69 vs reference 0.006); "inside-the-ball
  protected, outside-the-ball resisted."
- **g1b/g1bR/T125/T133**: the wall machinery (commit-then-project,
  R=0.7, settle-at-checkpoint semantics, the +0.53 wash tax) — g9
  keeps the machinery and changes the ball's SHAPE.
- **g1bW2 (queued, the prerequisite)**: locates the B-dose where B
  installs on the unwashed control (600+ steps expected; A needed
  ~400). g9 runs AT THAT operating point or it has nothing to
  admit. Dispatch gated on it.
- **g3K/T137 + g3R/T135**: the death currency is ALIGNED
  displacement, and the lethal set is a low-dimensional
  adaptation-like span with a wide-but-bounded cone (45° tilts
  kill; isotropic needs 4–10x). This DESIGNS the wall's correct
  geometry: cap the aligned subspace, let the isotropic remainder
  ride the natural graded basin.
- **W022**: cos(grad g0, wash) = -0.44 — the rank-1 ball's axis is
  the lab's existing death ruler.
- **e154/T100 + e160/T090**: two facts annihilate under normal
  training (all-or-nothing at the first F2 gradient) yet their
  head populations are DISJOINT — the leading prediction that B's
  formation lives OUTSIDE A's anchored subspace (ADMISSION-WORKS)
  and the leading NO-DOOR mechanism (row-0 collision, e142/T082's
  ROW-0-ALWAYS) named in advance.
- **Collision, stated**: g6 walls the FUNCTION space (per-input
  activation tube at the graft site) on the g3 STORE organism —
  g9 walls PARAMETER subspaces on the g1b HOST-FACT lineage;
  g7 composes wall+rhythm+cone with no admission question. No
  overlap in instrument, organism, or question.

### (b) What is new

The wall's geometry has never been a design axis: g1/g1b confined
ALL directions (a point-neighborhood); g1bW then measured the tax
of full confinement on new learning. g9 makes the ball's SCOPE the
single knob: the projection-back-to-anchor fires ONLY on the
anchored subspace spanned by the install fact's own gradient
structure; all orthogonal motion is FREE. FULL-BALL and FREE are
the two endpoints of the same knob (P = I and P = 0), so g1bW's
walled and unwashed references are literally this cell's endpoints
— the cell is one continuum with three interior rungs. The
architectural claim at stake: protection and plasticity are
SPATIALLY SEPARABLE (the cone-shaped wall spends its entire budget
on the death currency and nothing on new-structure directions).
Zero new trainable parameters; the basis is computed from A's
install battery gradients (state = k basis vectors + the anchor —
g1's anchor budget).

### (c) The single knob and the discriminating observation

**Knob: k — the rank of the anchored subspace** (top-k PCs of the
install-battery gradient covariance at the anchor; k ∈ {0 (FREE),
8, 64, ∞ (FULL, = g1bW's walled arm)}; R = 0.7 on the in-subspace
component, inherited). Mechanics: θ <- θ - lr·u; decompose
δ = θ - θ_anchor into Pδ and (I-P)δ; project Pδ back to radius R;
orthogonal component untouched. **Discriminating observation**: the
(A-survival, B-formation) PAIR across k — the cell is interesting
ONLY in the joint read; A alone is g1b again, B alone is g1bW
again. Pre-registered basis rule (no shopping): G-BASIS must show
the span covers >= 0.8 of grad g0's energy at k=8 before any arm
runs; if it fails, the cell reports TEXTURE with the coverage
number and the basis is NOT swapped.

Arms (2.7M seed-10907 lineage, stated reason: direct comparability
with the minted onset-tax numbers; g1bW2's licensed B-dose and
draws, bit-identical inputs across arms; installs 600+ steps,
chunk-resumable <= 180 s; GPU): FREE / FULL (references, G-REPRO
against g1bW2's cells) / GATED k=8 / GATED k=64 / + one B-draw
replicate at the best k.

### (d) Registered bars (draft)

- GATES: G-REPRO (walled and unwashed references reproduce g1bW2's
  cells bit-comparably); G-BASIS (>= 0.8 grad-g0 coverage at k=8);
  G-PIN (in-subspace displacement <= R + one-step fuzz at every
  checkpoint); G-INPUTS (md5 parity).
- ADMISSION-WORKS: "fires if at some k ∈ {8, 64}: A >= 0.70 at
  EVERY checkpoint through B's install AND B's partial-form peak
  >= 0.8x the unwashed reference — protection and plasticity are
  spatially separable; the wall's correct denomination is
  cone-shaped; the onset tax was the full ball's point-locality."
- CONE-BYPASS: "fires if A <= 0.27 by install end at EVERY k while
  B forms freely — the kill paths tilt outside the install-gradient
  span (g3R's 45-degree lesson transplanted to the sequential
  setting); the FULL ball is necessary and its tax structural."
- NO-DOOR: "fires if B's partial-form stays <= 0.5x unwalled at
  every k with A holding — B's formation needs the anchored
  subspace itself (carrier overlap; the row-0 collision the
  ROW-0-ALWAYS law predicts); sequential memory is a genuine
  either/or at this architecture."
- ONSET-TAX-CURVE (co-read): "B's partial-form TIME trace per k —
  where in the install the block acts (formation onset vs
  consolidation phase); feeds T140's onset-channel mechanism."
- TAX-FRONTIER (co-read): "in-batch install CE per k at install
  end vs FREE — the protection-per-unit-tax frontier; the number
  g6's STREAM-WALL-CHEAP will be compared against."

### (e) Honest failure modes

- The PC basis may simply miss the cone (CONE-BYPASS) — that is an
  outcome with a name, not a failed instrument; g3R's tilt ladder
  predicts it as live.
- The operating point is borrowed from g1bW2 — if B never installs
  unwalled even at 900 steps, g9 has no object and stands down
  (the gate, not a shopping loop).
- Row-0 collision between A and B (e142/T082) is the leading
  NO-DOOR mechanism — pre-registered here so the null reads as the
  address law's prediction, not a surprise.
- k=64's basis approaches the full ball's state cost — the
  TAX-FRONTIER co-read keeps the economics honest.
- A held at 0.83 through an ACTIVE install under the FULL ball
  (g1bW) — so CONE-BYPASS must beat a high bar: A dying under a
  WEAKER wall while it survived the stronger one is informative
  about the cone's width precisely because the endpoint held.
- n=1 lineage, one wash/install seed, B-draw replicate at one k;
  the standing convention, stated.

**Lane/model**: GPU (installs are trainings; g1bW precedent),
behind g1bW2 (which sits behind g1bS/g2g per C13-2 and g6 per
R56's order); 2.7M with stated reason; zero new trainable
parameters; every run chunk-resumable at <= 180 s.

---

## Dispatch order (the ideator's recommendation)

1. **e188 NOW** — CPU eval-only, minutes, zero lane conflict with
   g1bS (GPU) or opt1b (CPU, light coexistence per the e182c
   "run beside" precedent); it is the only cell of the three that
   is BOTH fully spec'd (W022b + three amendments landed) AND
   pointed by two results (T137 names it; T139's flat summary
   alignment demands its curves); it decides the paper's
   vocabulary (task arithmetic) and the rate law's mechanism in
   one pass. The TWO-CURRENCY amendment (g3K's static row) makes
   it the trajectory law's quantitative statement — the cheapest
   centerpiece figure the lab can buy.
2. **opt2 next on the CPU lane, after opt1b lands** — its bars are
   stable under either opt1b outcome (stated in the failure
   modes), but its FRAMING consumes opt1b's answer, and the lane
   is single-threaded by convention. This is the optimizer arc's
   natural terminal cell: opt1 decomposed clock/gate, opt1b tests
   the gate's class-generality, opt2 names the carrier.
3. **g9 specs freeze now; dispatch on GPU after g1bW2** — the C13-2
   order (g1bS then g2g) and R56's g6 hold the GPU lane regardless;
   g9 is GATED on g1bW2's dose finding by construction. Freezing
   the spec now (basis rule, bars, gates) lets the next review
   register it without a redesign, and G-BASIS can be pre-computed
   eval-only on CPU the moment g1bW2's operating point exists.

The one-sentence program view: the lab now holds the wash's clock
(opt1), its gate (opt1/opt1b), its currency candidate (e188), its
carrier candidate (opt2), and the wall that survives it (g1b);
e188 prices the kill, opt2 names the weapon, g9 reshapes the
shield — and none of them touches the C13-2 lane order.
