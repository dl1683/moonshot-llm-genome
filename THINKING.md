---

## T010 — MECHANISM CARD v1: the functional anatomy of a 2.7M char-transformer (2026-09-24T13:47Z)

The lab's first formal findings artifact (README: "a mechanism card another
researcher could attack"). Every claim carries its evidence chain, known
debt, and falsifier. Scope: ONE architecture (6L/6H/192 pre-LN char-GPT,
2.7M params), Shakespeare + synthetic variants, 6 trained nets (B, R, B43,
R43, e021 task+control), ~5 GPU-hours, single lab.

**C1 (H) — Function is staged; stages are the organism.** Pipeline:
token-formation → local completion → late distribution-calibration; on
retrieval tasks a retrieval stage appears at L4. Evidence: depth-census
invariance across 2 regimes × 2 seeds (4-net table, cross-seed 0.849 ≥
cross-regime 0.828, e012/e012b/e012c); decision-depth class ordering
(letters>structural) in all 6 nets; task-dependent L4 mode (e021).
Debt: single architecture/corpus family; v008 multi-seed lesion maps.
Falsifier: stage structure failing at other scales or on far-retrieval
natural data (e005s ladder, gated on this card).

**C2 (H) — Anatomy is plastic; damage tracks write allocation.** Lesion
maps reorganize under constraints (keystone dissolved +4.08→+0.10; MLP
profile inverted) with parity-or-better loss; replicated on R43; write/c
rank-order = damage rank-order (ρ=1.0); damage tracks perturbation ENERGY
not content (e011c ladder, CIs clean); MLP-5 is an energy carrier (e019
causal: graceful α, zero=4×rotate). Debt: e014c write-clamp parked.
Falsifier: write-clamp training leaving lesion maps unchanged.

**C3 (H) — The seed-anchored object is the residual-stream basis.** Organ
graft compatibility follows init lineage, not regime (e028/e029 2×2); ΔW
motion from different inits is near-orthogonal (cos≈0.000 vs same-init
+0.15); stream-FACING matrices (W_in reads, W_out writes) are the violent
grafts, W_in dominant (e031: L5 W_in alone +3.28 > whole organ +1.46 —
donor W_out partially rescues); attention matrices are mild (c_attn
mildest). Debt PAID (e041): alignment ladder complete — 1.0 → 0.534 (same-init
diff-order CEILING) → 0.152 (same-init diff-regime) → 0.000 (diff-init).
PARTIAL ANCHORING: regime change exceeds batch-order noise 3.6×, yet
same-init stays far above the floor; early organs order-robust, depth
erodes. Former debt line (superseded): ΔW ceiling null unrun —
e031 LN-statistics alternative unexcluded; "attention portable" is
weak-anchoring, NOT shared subspaces (v009).
Falsifier: e030 Procrustes-style basis-realignment rescuing cross-seed
grafts would confirm basis-geometry (upgrade); spectrum-matched W_out
control rescuing e031's pair effect would demote the coherence reading.

**C4 (H, scoped) — No far-context retrieval on natural char data; retrieval
is task-elicited, not architectural.** Shakespeare: L5 calibration local
(KL −6.9% under truncation), far attention idle (200-prompt census),
far-value bimodal but structure-insensitive (gains shuffle-robust, hurts
shuffle-amplified; e013a/c/d). Task: noiseless retrieval (CE 0.007,
far-value ln 26, head L4-H1 95.1% ID-mass, control clean; e021).
RESOLVED (e035/e038): L4-H1 is causally the LARGEST retrieval channel
(~51% of distance-to-chance) inside a REDUNDANT cooperative fan (copy
survives at 58%=15× chance with perfect locality) — not a dedicated organ.
One net holds TWO stage profiles (JS(filler,Shakespeare)=0.010 vs
JS(filler,COPY)=0.272), selected per-position. Lesion maps are blind to
task circuits (COPY ~5% of tokens). Init-anchoring is task-independent
(slightly stronger than B↔R).

**C5 (CLOSED, negative) — First-order ascent cannot content-selectively
forget.** At the 0.66-nat bar: naive r=1.23, projection r=1.43 (margin
1.17×), and r vs train-B = 1.08 — memorization-symmetric damage; apparent
selectivity was a measurement-axis artifact. The r≈5-6 head-start did not
reproduce (chaotic event). Next family: weight surgery (e023) / second-
order. MLP-5 energy-carrier confirmed causally along the way (e019).

**Cross-cutting instruments validated:** decision depth, locality funnel,
KL(L5‖L4), write/stream ratios, transplant R-bands, ΔW subspace atlas —
with known lens caveats (mid-stack readouts anti-informative; argmax-stability
robust).

**Card debt summary (blocking v2):** ΔW ceiling null; e035/e038 verdicts;
e003d; v008 multi-seed maps; e005s scaling ladder (gated on this card).

---

# Thinking Journal — the 80%

The lab's operating ratio is 80% thinking / 20% doing. Every result gets an
entry here BEFORE the next experiment that builds on it: what it could mean
(multiple hypotheses), how the hypotheses differ, which cheap observation
would discriminate them, and a registered prediction so we can't retrofit.
New experiments are gated on this file: if the latest result has no
interpretation entry, the next heartbeat thinks instead of runs.

---

## T003 — What E011b says about authority, redundancy, and geometry (2026-09-24T10:4xZ)

**Observed (E011b, eval-only):**
1. **L0 head subsets:** singles mean +0.05 (max +0.165, one slightly
   negative); damage by subset size 1→6: 0.05, 0.36, 1.01, 1.77, 2.27, 2.40.
   Sum of singles 0.317 vs all-6 2.403 = **7.6× superadditivity**. Internal
   validation: all-6 head-zero (2.403) ≈ whole-block zero (2.401) — the hook
   implementation is exact (closes critique #8).
2. **Orthogonal-innovation:** same-norm random writes damage MORE than zeroing
   everywhere: attn [3.66 vs 2.40, 2.41 vs 1.74, 1.78 vs 1.06, 0.78 vs 0.38,
   0.38 vs 0.19, 0.10 vs 0.03]; mlp [4.44 vs 4.08, **0.82 vs 0.15**, 0.72 vs
   0.27, 0.84 vs 0.47, 0.89 vs 0.60, 1.00 vs 0.59].
3. **Stream norms at block inputs:** [0.67, 5.66, 6.14, 5.12, 5.20, 5.62] —
   an 8.4× jump at block 0, then a plateau.

**Reading A — cooperative ensemble, not backups (finding 1):** L0's heads are
a graceful-degradation fan: each is nearly dispensable alone, but damage grows
steeply with subset size (keep 1 head → 94% of full-ablation damage). They
are small parallel contributions, not redundant copies of each other.
Single-lesion anatomy maps CANNOT see this; joint criticality is ~7.6× the
sum of marginal criticalities.

**Reading B — the authority-schedule hypothesis (findings 2+3, the big one):**
downstream blocks read LN(x), which is scale-invariant — what a write can
change is the ANGULAR position of LN(x), bounded by ~‖w‖/‖x‖. The stream
grows 8.4× at block 0 and then plateaus, so **write-norm/stream-norm falls
~12× from L0 to L5** (attn-L0 writes 4× its incoming stream; attn-L5 writes
0.34×). Under this reading the front-loaded lesion map is partly ARCHITECTURE,
not learning: layer 0 structurally holds the most authority per parameter,
late blocks are architecturally incapable of large angular moves. The network
schedules authority by stream growth.

**Reading C — noise poisons more than absence (finding 2):** zeroing removes
information; a same-norm random write injects misinformation. Perturbation
norms differ only by ~√2, but observed zero→random ratios run 1.4× (attn L0-L1
≈ pure geometry) to 5.4× (MLP L1: 0.15→0.82) — far beyond √2 for MLPs and
late attention. Where the ratio exceeds √2, downstream computation is
calibrated to the write's DIRECTION (content matters, and actively).

**Discriminating observations:**
1. **Matched-perturbation control (e011c, minutes):** replace write w by w′
   rotated exactly 60° so ‖w′−w‖ = ‖w‖ = perturbation of zeroing. If damage
   ≈ zero-damage → geometry; if ≫ → content. Settles B vs C per component.
2. **Authority-schedule intervention (e014b, the real test of B):** train a
   fresh net with the stream renormalized to constant norm at every block
   input (hook). B predicts the lesion map flattens (late damage rises, L0
   dominance drops). If the map stays front-loaded without stream growth,
   learning, not architecture, owns the front-loading.

**P1 RESOLVED (E011c, 2026-09-24T10:45Z): REFUTED in both parts — and the
truth is cleaner.** At matched perturbation energy (60° rotation, verified
‖w′−w‖=‖w‖=1.0000), rotate-damage vs zero-damage: attn [1.38, 0.74, 0.89,
0.87, 0.71, 1.17]; mlp [0.80, **2.95**, 1.20, 0.70, 0.47, **0.24**].

- **Dominant factor = perturbation ENERGY, not content:** across all 12
  components, damage ranks {zero ≈ rotate60} < {random ≈ √2·energy}. Most
  blocks tolerate scrambled content about as well as — or better than —
  removal. This STRENGTHENS Reading B (authority schedule): what matters
  most is how much a block can move the stream, i.e., the geometric
  schedule, not the specific meaning of its write.
- **Three real exceptions (content-sensitive or energy-carrier):** MLP-L1 is
  direction-sensitive (×2.95); attn-L0 mildly direction-sensitive (×1.38);
  MLP-L5 is an ENERGY CARRIER — zero costs +0.59 but rotate only +0.14, so
  its value is mostly magnitude in the stream, not information. (Connects to
  E011a: MLP-L5 writes the largest residual and matters least per unit.)
- **Caveat before over-reading:** single damage estimates over 20 eval
  batches; ratios 0.7–1.2 may be within noise. Needed observation (cheap):
  bootstrap CIs over eval batches for the rotate/zero ratios; only MLP-L1,
  attn-L0, MLP-L5 look safely beyond noise.
- **The decisive test of Reading B remains e014b** (stream-renorm training).
  P2 prediction unchanged.

**Still-registered predictions (T003):**
- P2 (e014b): stream-renormalized training flattens the attention damage
  profile by ≥50% (L5 damage rises well above +0.03; L0 falls below +2.0).
- P3: damage-per-write across attention layers correlates ≥0.8 with
  write/stream ratio (the geometric authority term) — checkable now from
  existing numbers.

**T003 FINAL RESOLUTION (E014b, 2026-09-24T11:08Z): Reading B REFUTED at
full parity — and the deeper finding is anatomical plasticity.**

Facts: renorm arm (stream pinned to c=5.6 at every block input, train+eval)
reached val 1.610 — BETTER than baseline 1.622, parity gate passed
decisively. Lesion map under renorm: attn damage [2.80, 1.69, 0.48, 0.81,
0.21, 0.03] — still strongly front-loaded (spread 2.77 vs baseline 2.37;
D'5 +0.031 ≈ baseline 0.034). P2 criteria: REFUTED (parity ✓, D'0 ≥ 2.0,
D'5 ≤ 0.05).

1. **Front-loading is NOT caused by stream-norm geometry.** With the
   norm-growth schedule deleted from the architecture, the network still
   organizes early-critical/late-cheap structure. Front-loading is
   functional allocation (early layers do the coarse work), not a
   geometric artifact. The E011c energy-dominance result stands, but the
   CAUSE of the energy allocation is learned task structure, not stream
   growth.
2. **Anatomy is plastic.** Baseline keystones MOVED: MLP-0's keystone role
   (+4.08 nats) dissolved under renorm (+0.10) — plausibly because baseline
   MLP-0 was bootstrapping the 0.67→5.7 norm jump, and renorm does that for
   free. Attention-L0 became MORE critical (+2.80 vs +2.40) with a 2.9×
   larger write (7.8 vs 2.7). Mid-stack reorganized (L2 damage halved, L3
   doubled). Multiple anatomies reach the same function.
3. **Optimization declines late authority even when purchasable.** Renorm
   constrains stream norm at CONSUMPTION points, not write size — any layer
   could still steer strongly by writing big (L0 does: 7.8). Yet L4-L5
   writes DEFLATED (ρ 0.82, 0.70) while L1-L3 partially re-inflated
   (ρ 1.25-1.47; mean 1.10 < 1.3 criterion → re-inflation hypothesis also
   not met as stated). The network allocates authority where the work is.
4. **The norm profile is per-net load-bearing but task-optional:** eval-only
   renorm on baseline weights +3.56 nats (E014b control) vs renorm-trained
   parity — a given net's wiring depends on its norm profile, but learning
   does not require the canonical profile.

Status of T003 after E014b: geometry-schedule story dead; live questions
move to WHY early layers hold the coarse work (curriculum of abstraction?
token-formation bottleneck at L0?) and WHY late MLPs write big-but-cheap
(the MLP-L5 energy-carrier mystery persists and deepened — renorm MLP
damage is now LATE-heavy [0.10, 0.08, 0.23, 0.45, 0.78, 0.70], the
opposite arrangement from baseline). Stream-norm growth is documented (TurnTrout 2023: ~1.045×/
layer in GPT-2-XL, "overshadowing not deletion", NO causal interventions);
front-loaded criticality is widely observed (Gromov 2024, ShortGPT, BERT
pruning) but always explained functionally, never geometrically; renorm
training exists (nGPT trains unit-norm streams, post-LN literally renorms)
but nobody has measured depth-wise lesion profiles under it. **Our
contribution claim: the write/stream→damage link + the causal flattening
test.** Falsifier on record: Pythia/GPT-Neo streams SHRINK with depth
(schedule is regime-dependent), and Gromov/Nepal suggest importance profiles
can be sticky across training interventions.
**e014b design upgrades from the check:** (a) add a post-LN arm alongside the
renorm arm; (b) verify the renormed net reaches comparable loss before
comparing lesion maps (else the comparison is confounded); (c) LOG the
renormed net's learned write/stream ratios — if training RE-INFLATES writes
under renormalization, that alone shows optimization *wants* an authority
schedule, independent of the lesion result.

---

**REVIEW 1 AMENDMENTS (2026-09-24T11:20Z, INTERPRETER findings — accepted):**

1. **T005 WEAKENED.** The "rare-token" signal is ONE head of 36 (L5.h1
   surprisal 6.89 bits; other five L5 heads 4.39 vs L4's 4.28 — below the
   script's own 0.15-bit "rarer" threshold). Re-broadening is +0.13 nats,
   below v002's own "similarly spread" criterion. The ×76 distant jump is a
   ratio of tiny masses, and 62% of L5's non-local mass is mid-range
   (d17-64), not ancient. The 'O'-matching may be a dialogue-vocative
   artifact of a single prompt containing ≥3 'O's. **What survives: L5
   abandons the local d1-3 window (0.234→0.060).** The rare-token framing
   is a hypothesis, not a finding — e013 (causal mask) is GATED on e013a,
   a 200-prompt all-head attention census, so we don't spend causal budget
   on a mechanism that may exist in one head of one prompt.
2. **T003/E014b "functional, not geometric" was OVER-CLAIMED.** Renorm
   pinned block inputs, never write norms — angular authority was never
   capped. The un-refuted live hypothesis: **damage tracks the write/stream
   allocation itself.** P3 now EVALUATED (was pending): renorm-arm attn
   write/c [1.40, 0.58, 0.39, 0.56, 0.31, 0.15] vs damage [2.80, 1.69,
   0.48, 0.81, 0.21, 0.03] — identical rank order (ρ=1.0). The network
   REBUILT a 9.3× declining write-allocation schedule under renorm, and
   front-loading strengthened. Decisive test now queued as **e014c
   write-clamp** (pre-named in the design memo's failure table): clamp
   ‖w‖ ≤ α·‖x_in‖ during training, or eval-time rescale L0/L5 writes by
   k∈{0.5, 2, 4} and measure damage.
3. **T004 depth-6 is definitionally the L5 argmax flip** (depth=6 ⟺
   top1(L4-readout) ≠ top1(L5-readout)) — the "54% finalize at L5" stat
   partially RESTATES the calibrator observation. Upgrade queued as e018:
   CAUSAL depth (activation patching from counterfactual contexts; the
   shallowest depth where splicing switches the final decision).

**T005 FINAL ADJUDICATION (E013a census, 2026-09-24T11:47Z): L5 is a
DIFFUSE re-globalizer — the rare-token framing is dead.** Across 200
prompts, all 36 heads: the locality funnel replicates (far-mass U-shape
0.80 → 0.09 at L2 → 0.54 at L5; local-mass peaks at L2 0.42); L5 abandons
the local window in **82.5%** of prompts (criterion 70%). But attended-token
surprisal is flat across ALL layers (~5 bits) and **zero** heads show
consistent distant-concentration — v002's 'O'-matching head was a one-prompt
artifact, exactly as Review 1 suspected. The causal question shifts: does
L5's calibration (KL(L5‖L4 readout) ≈ 1 nat) actually DEPEND on far context?
**e013 REDESIGNED as context truncation:** last-16 vs full-96 contexts →
measure KL(L5‖L4). Registered: KL shrinks ≥50% with truncated context (L5's
reshaping draws on far information); if unchanged, L5's calibration is
locally derived and "re-globalization" is epiphenomenal attention shape.

**T005 CLOSED (E013 truncation, 2026-09-24T12:05Z): L5's calibration is
LOCAL; far attention is idle grazing.** KL(L5‖L4 readout) barely moves under
256→16 truncation (0.997→0.928, −6.9% vs registered ≥50%); flip rate
unchanged (49→52%). Combined with the census (shape real, no rare-token
selectivity) the full honest picture: **L5 reshapes the output distribution
using LOCAL information, while attending diffusely far for nothing.** The
sharper corpus fact underneath: **16-token sufficiency** — next-char CE at
1.648 (full) vs 1.644 (16 tokens): at char level on Shakespeare, far context
has ≈ zero marginal value for this 2.7M net. Consequences: (a) T004's D2
(integration-of-range) is refuted — late decisions are deep LEXICAL
computation; (b) any "long-range" mechanism claim at this scale must first
show far context matters at its positions (follow-up: per-position far-value
tail — uniform ≈0, or a rare-position minority carrying all of it?);
(c) mid-stack readouts are anti-informative (depth-2 CE 5.19 > unigram 4.17)
— absolute mid-stream distribution claims need a tuned lens; argmax-stability
claims are robust.

## T006 — Anatomical plasticity: what actually persists when the organs move? (2026-09-24T11:21Z)

**Observed (E014b, single seed — replication debt registered):** under
constant-norm streams the lesion anatomy reorganized while the function did
not: renorm arm reached BETTER val loss, yet MLP-0's keystone role dissolved
(+4.08→+0.10 nats), attention-L0 strengthened (+2.40→+2.80), and the MLP
damage profile INVERTED to late-heavy [0.10, 0.08, 0.23, 0.45, 0.78, 0.70].
Write allocation rebuilt a 9.3× declining schedule whose rank order matches
damage exactly (T003 P3, ρ=1.0). Eval-only renorm on baseline: +3.56 nats —
each anatomy presumes its own geometry.

**The question:** when the organs move, does anything stay put? Four layers
of "what persists":

- **PL1 — role migration (organs follow necessity):** baseline MLP-0's
  keystone role was largely "norm bootstrapper" (writing 4.3 into a 0.67
  stream); when the architecture does that for free, the organ's importance
  collapses. Prediction: in the renorm arm, MLP-0's write barely STEERS —
  its angular displacement per token should be far below baseline's.
- **PL2 — functional invariants (the stages persist, organs don't):** both
  anatomies implement the same pipeline — early token-formation, mid-stack
  local completion, late global calibration (the E012 decision-depth profile
  and the V002 locality funnel). Prediction: the renorm arm's decision-depth
  census ≈ baseline's — same depth modes, same letters-late/structure-early
  class ordering.
- **PL3 — anatomy-level degeneracy:** the neuro-ai-lab degeneracy concept
  scaled from routes to whole organ arrangements. Prediction: cross-anatomy
  transplants (baseline organ into renorm net, and vice versa) fail much
  harder than same-anatomy swaps — organs are interchangeable within an
  anatomy (E011b head redundancy) but not across anatomies.
- **PL4 — regime clustering:** optimization pressure + constraints choose
  the organ assignment; assignment is regime-dependent more than
  seed-dependent. Test deferred to v008 (multi-seed lesion-map phylogeny).

**Why this matters for the lab's identity:** if PL2 holds, the correct
dissection units are FUNCTIONAL STAGES (measured by decision depth, locality,
calibration KL), with lesion maps as implementation detail — the instruments
we built this morning (E012, V002) would be measuring the real organs, and
"which layer does X" is the wrong question; "which stage does X, and where
did it land this time" is the right one.

**Discriminating observations (cheap → expensive):**
1. **e012b (minutes, eval-only):** rerun the decision-depth census + angular
   profile on the renorm checkpoint (hooks active). Tests P1+P2 at once.
2. **e028 transplant (design memo queued):** within- vs cross-anatomy organ
   swaps at matched sites. Tests P3.
3. **v008 (later):** multi-seed lesion-map embedding. Tests P4.

**Registered predictions:**
- P1: renorm-arm MLP-0 angular displacement ≤ ⅓ of baseline's.
- P2: renorm depth histogram matches baseline's (bin-wise correlation
  ≥ 0.8; identical class ordering: letters > punct > newline ≈ space).
- P3: cross-anatomy swaps cost ≥ 2× same-anatomy swap damage.
**T006 PARTIAL RESOLUTION (E012b, 2026-09-24T11:32Z): stages are the
organisms; lesion maps are their current addresses.**

- **P2 CONFIRMED (corr 0.822):** the renorm anatomy reproduces the baseline
  functional profile almost exactly — L5-finalization mode identical (1084
  vs 1082 of 2000 positions), same L1 dip, same depth↔entropy relation
  (Spearman +0.344 vs +0.322). The pipeline (early token-formation →
  mid-stack local completion → late finalization) is the INVARIANT; where
  exactly the mid-stack work sits (baseline spreads L0-L4 [158,46,132,160];
  renorm concentrates L3-L4 [348,524]) is implementation detail.
- **P1 missed the strict threshold, direction strong:** block-0 angular
  displacement 0.746 → 0.288 (ratio 0.386 vs predicted ≤0.33); ALL layers'
  angular displacement roughly halved in the renorm anatomy (calmer net).
  Since renorm attn-L0 grew MORE important, MLP-0's own angular role shrank
  further than the block total indicates. PL1 (norm-bootstrapper role)
  supported, threshold pedantically missed.
- **Lab-identity consequence adopted:** the primary dissection instruments
  are now decision depth, locality, and calibration KL (they measure the
  invariants); lesion maps are secondary (they measure where the stages
  currently live in THIS net). "Which layer does X" is deprecated in favor
  of "which stage does X, and where did it land this time."
- Open: P3 (cross-anatomy transplant, e028 design memo in progress) and P4
  (v008 multi-seed phylogeny) test whether whole-organ degeneracy respects
  anatomy boundaries.

**T007 CLOSED (E013d, 2026-09-24T12:12Z): no specific far-context retrieval;
far context acts through bulk statistics.** P1 refuted (+0.22σ < 0.5; 91% of
hurt positions have NO divergent repeat — repetition interference dead). P2
refuted informatively: shuffled-far makes hurt WORSE (−2.26 vs −1.68) while
gains survive (+1.50 vs +1.60) — gains are shuffle-robust (statistical:
char mix/length), and incoherent far text destabilizes more than real far
text. Combined with E013 (L5 calibration local; far attention idle): this
model's long-range behavior is bulk-statistics + noise, not information
retrieval. Long-range claims must be re-tested on tasks that provably
require retrieval (copy spans; e021 task-swap).

**T006 P3 RESOLUTION AMENDED (E029, 2026-09-24T12:16Z): mechanism CONFIRMED,
claim refined.** ΔW-alignment is decisive: same-init organ pairs cos = +0.152,
different-init ≈ 0.000 (max |cos| 0.017) — **training motion from different
inits is almost perfectly orthogonal in parameter space**; organs refine
init-anchored directions. Seed dominance is organ-type specific: **MLP organs
are seed-anchored (ρ 2.0-3.6), attention organs are portable across both
axes** (ρ 0.54-1.29); the one regime-dominant organ is R-host MLP-L0 (the
keystone asymmetry). T008 claim 3 upgraded to H (mechanism confirmed) and
rewritten: "MLP-organ compatibility follows initialization lineage; attention
organs are anatomy- and init-portable." New open question: why? Candidate:
attention READS stream directions that all adequate solutions share; MLPs
WRITE into seed-specific subspaces.

**T008 REVIEW-2 AMENDMENTS (2026-09-24T12:26Z):**
- **Claim 1 → M (downgraded):** the two anatomies compared (B, R) share
  seed 42; e029 showed same-init nets share ΔW directions — stage
  invariance was never tested across seeds. e012c (census on B43/R43,
  running) de-confounds: cross-seED histogram match restores H;
  seed-clustering keeps it init-bound.
- **Claim 3 sub-claim "attention portable" flagged:** MLP ΔW alignment
  (.26/.10/.20) exceeds attention's (.21/.06/.08) — cosine cannot mediate
  the organ-type difference; portability may partly be small-denominator
  artifact (late-attn ablation refs 0.016-0.034; R-host L5-attn is
  regime-dominant). Solid at L0/L3 only. ΔW gap (+0.155) lacks a
  same-init/data-order-replicate ceiling null — registered as needed.

**T008 DEBT RESOLUTION (e012c + e014b.1 + e011c-ci, 2026-09-24T12:40Z):
claims 1 and 2 upgraded to H.**
- **Claim 1 RESTORED at H (init-independent):** 4-net depth-histogram table —
  cross-seed same-regime (B–B43 0.855, R–R43 0.842; mean 0.849) ≥
  cross-regime same-seed (B–R 0.822, B43–R43 0.834; mean 0.828); no seed
  clustering (delta −0.021). Supporting invariants replicate in both new
  nets (L5-finalization 1027/1088; Spearman +0.323/+0.326; class ordering
  letters > punct > structural). Stages are the organism — across seeds AND
  regimes.
- **Claim 2 REPLICATED (second renorm seed R43):** keystone dissolution
  (MLP-0 +0.21 vs B +4.08), late-heavy MLP flip (trend ρ +0.74), attention
  front-load (spread 2.56), and a THIRD independent rebuild of the declining
  write schedule (6.27→0.96 into the pinned stream).
- **e011c exceptions all real:** attn-L0 1.38±0.006, MLP-L1 3.03±0.044,
  MLP-L5 0.25±0.009 — beyond noise by 20-100× sd.

**T009 RESOLVED (E021, 2026-09-24T13:02Z): all four predictions landed.
Retrieval is task-elicited, not architecturally absent.**
- P1: 100% copy accuracy (control 4.1%); CE at COPY 0.007 nats — the copy
  is noiseless at 2.7M params.
- P2: far-value +3.269 ≈ ln 26 at COPY (control −0.002). **T008 claim 4
  AMENDED (stays H, narrowed scope): "no far-context retrieval on natural
  char data at this scale." When the task demands retrieval, this exact
  architecture delivers it exactly.**
- P3: dedicated retrieval head L4-H1 (95.1% mass on the ID nonce; control
  15%). The idle-grazing signature was a property of the corpus, not the
  architecture.
- P4: NEW decision mode — 88.3% of COPY decisions at L4 (Shakespeare 8.0%;
  JS 0.265), one layer earlier than Shakespeare's L5 mode: retrieval
  completes before final calibration. Claim 1 (stages) intact and enriched:
  the pipeline reorganizes around task demands; stage membership is
  task-dependent, stage *existence* is not.

**T008 claim-3 mechanism note — AMENDED (V009 full report, 2026-09-24T13:10Z):
the seed-anchored object is the residual-STREAM basis itself.** Right-singular
(input-space) gaps: reads W_in +0.260 / c_attn +0.235 vs W_out-right +0.091
(which initially suggested "reads private, writes shared"). But the
left-singular (stream-space) supplement REVERSES the second half: **W_out-left
gap +0.267 (same 0.562 / diff 0.295) vs c_proj-left +0.071 — MLP stream-writes
are 3.8× more init-anchored than attention's**, matching e029's transplant ρ
(MLP L3/L5 3.08/2.27 vs attn 0.93/1.29); W_in-right peaks at L5 (0.661) where
MLP grafts are most seed-dominant. And diff-seed alignment sits AT the random
floor everywhere (excess ≤ +0.04) — attention portability is NOT shared
subspaces; it is weak anchoring (c_proj barely anchored on both sides).
Refined mechanism: every stream-FACING interface (reads and MLP writes) is
init-anchored; MLP hidden space is barely anchored; c_proj is the
insensitivity exception. **e031 RE-REGISTERED: individual-matrix grafts at
L3/L5 — W_in and W_out each predicted VIOLENT (stream-facing); c_proj
predicted MILDEST of the four.**

## T009 — e021 registration: does a retrieval-required task break the no-retrieval picture? (2026-09-24T12:40Z)

**Design (adopted from Review-2 ideator):** synthetic corpus (~1MB) of
documents: "ID: [5-char uppercase nonce] … Shakespeare filler (>16 tokens)
… COPY: [nonce repeated]". Retrieval is provably required at COPY positions
(nonce is >16 tokens back, unpredictable without the ID). Control corpus:
same shape, nonces shuffled at COPY (cue uncorrelated). Train fresh 2.7M
nets on each (252s cap, ckpt, parity-style val gates); readouts: copy
accuracy at nonce positions; far-value (CE full-256 vs trunc-16) at COPY
positions; decision-depth census at COPY positions; locality funnel
(e013a-style mini-census) on the task net.

**Registered predictions:**
- P1 (learnability): task net reaches ≥80% next-char accuracy on nonce
  positions at COPY. If not: capacity/budget limit — informative negative.
- P2 (retrieval exists when required): far-value at COPY positions ≥ +1.0
  nat, and the control net shows ≈0. **If confirmed, T008 claim 4 narrows
  to "no retrieval on natural char data at this scale" — NOT an
  architectural limit.**
- P3 (mechanism): attention at COPY positions CONCENTRATES on the ID nonce
  positions (local mass collapses; a real retrieval head appears — the
  Shakespeare L5 idle-grazing signature should be gone).
- P4 (stages): if a new "retrieval depth mode" appears at COPY positions
  (decisions later than any Shakespeare position), the stage picture gains
  a task-dependent member; if depth profile is byte-identical to Shakespeare
  despite retrieval, stages are corpus-trivial — a serious blow to claim 1's
  interpretation.

**CLAIM 5 CLOSED (E003c full report, 2026-09-24T14:05Z): first-order ascent
cannot content-selectively forget.** r at the 0.66-nat bar = 1.43 (2 seeds)
vs 2.0 bar; step-matched naive 1.23 (margin 1.17×); **r vs train-B = 1.08**
— memorization-symmetric damage, zero content selectivity; val_B
"selectivity" was a measurement artifact. e003b's r≈6 head-start failed to
reproduce against its own code+seed (chaotic event, not mechanism). Next
family: weight surgery (e023) or second-order methods. MLP-5 energy-carrier
causally confirmed (e019: zero 4× rotate; α=0.5 improves CE; removal
spikes entropy +0.62).

**CLAIM 5 — FINAL CORRECTED STATE (E003b full report + Review 3, 2026-09-24T13:50Z).**
The registered instruments FAILED: exact projection peak r=1.455 (<1.5,
decays to 1.09); masked 1.80 (<2.0); combined 1.93. Naive is r=1.22 under
corrected labels (the historical "r≈1.0 anti-selectivity" was partly eval
mislabeling). The striking r=3-6 came from an ACCIDENTAL 70%-strength
projection whose residual B-component acts as an implicit retain-descent
anchor through Adam's sign updates — projection-before-Adam ≠
projection-of-the-step; it is E002's explicit anchor rediscovered at the
right dose. Claim 5 stands as: "no first-order instrument cleanly clears
its selectivity bar; a partial-projection/implicit-anchor hybrid shows
r=3-6 at mild doses and is the live candidate." e003d registered: the
deliberate anchor variant, dosed to the 0.66-nat forgetting bar, with
step-norm-matched naive and train-B collateral. Also registered: mean-level
cos(g_A,g_B)=0.799 vs batch-level 0.345 — gradient averaging collapses onto
the fluency direction.


