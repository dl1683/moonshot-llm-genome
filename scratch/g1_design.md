# g1 — THE ANCHORED BALL: the basin-widening architecture (design doc, W020 slot 1)

Design agent, 2026-09-29. This file is the design agent's single write; the
executor turns it into `lab/g1_basin_wall.py` + `runs/g1/`. Everything here
is frozen at dispatch: the bars below are registered IN ADVANCE; no bar
shopping at adjudication.

## 0. The law g1 tests (verbatim numbers, with provenance)

Memory has NO ROBUSTNESS BASIN in the pre-LN char-transformer:

- The consolidated fact dies at ~2.5–5 L2 displacement over 2,739,072
  trainable params (per-coordinate RMS ~1.5e-3) — T119/E180.
- Content-free noise kills identically at displacement-match; the kill is
  generic optimizer fragility, not corpus-directed — T114/E185, n=3/arm E187.
- Survival follows t* ~ 7.5e-4 · lr^-1.16 (R^2 0.975), lr·t* ~ const: the
  exit is DISPLACEMENT-LIMITED — T119/E180.
- The corpus stream's one gift is SURGERY, not direction (the fact dies, the
  organism recovers) — T114; re-entry is event-cheap and sticky (one replay
  resurrects the +2-dead fact; 9 replay events per 300 wash steps maintain)
  — T120/E179.

The g1 question: is the no-basin property an ARCHITECTURAL NECESSITY of the
pre-LN transformer, or does one minimal architectural change install a well?

## 1. A size correction, registered up front (the honest discrepancy)

The dispatch calls TinyGPT "~0.84M pre-LN". The memory line's organism — the
e131 root (`runs/checkpoints/e131_consolidated_e113.pt`), the source of
every number in §0 — is `Cfg(n_layer=6, n_head=6, n_embd=192)` = 2,739,072
trainable params. The 0.84M figure is the lab's OTHER lineage:
`Cfg(n_layer=4, n_head=4, n_embd=128)` = 840,704 params (e033 "0.84M" /
e040; its corpus-pretrained base `runs/checkpoints/e005s_small.pt` exists on
disk; installs at this size have precedent in e033/e040).

g1 runs at the **0.84M lineage** to honor the ≤1M-param constraint, and owns
the consequence explicitly: the stored wash numbers (root g-12 0.9156; +1
0.678; +2 0.0271) are 2.74M-organism numbers and are **priors, not bars**.
g1 therefore carries its own **size-matched control arm** — plain TinyGPT,
same config, same init seed, same phase-0 recipe — through the identical
battery. Every g1 verdict is adjudicated against the in-run control. The
basin width transfers only by sqrt(P) scaling: 2.5–5 L2 at 2.74M →
**1.4–2.8 L2 at 0.84M** (per-coordinate RMS assumed size-invariant;
registered as an estimate, calibrated in-run by the control's measured
D_kill).

CONTINGENCY CELL (one config line, no redesign): if the supervisor waives
the size cap, the identical design at 6/6/192 reuses the stored root
directly (load `e131_consolidated_e113.pt`, `commit()`, wash) making g1's
wash bit-comparable to e176N/e179's stored traces. Registered as the
preferred continuity cell; not required for any verdict below.

## 2. Mechanism choice: (a), refined by the law's own structure

**THE ANCHORED BALL** — a committed snapshot plus a hard L2 wall at radius
R, flat interior, all directions, enforced as forward semantics. The law's
four clauses each contribute one design decision:

| Law clause | Design consequence |
|---|---|
| thin in EVERY direction (noise kills identically) | the wall bounds ALL parameters jointly — no subset, no direction exemption |
| death tracks cumulative L2 (t* ~ lr^-1.16) | R is denominated in the law's own currency (L2 over all params); the wall's only dial is R |
| corpus direction is GENTLE per unit displacement (T114) | FLAT interior — zero force below R; within-ball adaptation is untaxed (plasticity preserved where the law says nothing dies) |
| re-entry is event-cheap and sticky (E179) | the commit event mirrors consolidation: one protocol event installs the well; the WELL is what the dataloader's rhythm was doing, made structural |

Honesty note on "basin-widening": g1 does NOT widen the region where the
fact is expressed (that region is fixed by the install; widening the
readout's tolerance is g3 territory). It makes the DYNAMICS unable to leave
the expressed region — the effective robustness basin (against continued
optimization) becomes unbounded in time. The task's own framing concedes
this: "the basin becomes a WELL". The R ladder then measures the expressed
region's width dynamically — architecture as instrument.

Why not (b) dual-rate: in any instrument-compatible form (one forward pass,
one residual stream) the effective weights are slow+fast jointly, and the
law applies to the SUM — e185's noise displaces the sum identically. Pure
freezing of the slow half is the trivial infinite well with no dial (and
"freeze" in e161/e176 meant a frozen STREAM, not frozen parameters —
parameter-space confinement has never been tried in this lab). Replay-gated
slow updates re-import the dataloader (g2's territory).

Why not (c) gated updates: the gate reads the INPUT, but e185's noise arms
are LABEL-side by construction (inputs bit-identical to the control's) — an
input-reading gate cannot see the noise kill coming; it fails the battery's
sharpest instrument by design. Loss-side gating cannot separate the fact's
own install loss from noise loss at onset. (c) is exactly W020's g2
REHEARSAL-NATIVE, where dataloader coupling is the point.

Prior-art delta vs EWC: EWC's Fisher-weighted quadratic penalty taxes ALL
movement (no flat interior) with an uncalibrated λ. The anchored ball has
ONE parameter, R, pre-registered in the law's measured currency, a flat
interior justified by the gentleness clause, and a hard wall that
GUARANTEES confinement (making the readout's fate a pure test of the law's
causal claim). Vs e161: e161 froze the stream, never the weights.

## 3. The design spec

### 3.1 Architecture delta from TinyGPT (the whole thing)

`class CommittedGPT(TinyGPT)` adds, and only adds:

1. `commit(R: float)` — snapshot every trainable tensor into registered
   non-trainable buffers (`anch__<name>`, 45 tensors); set `self.R`.
   One protocol event, called at the end of consolidation.
2. Forward-entry wall: at the start of EVERY `forward()` (train and eval),
   if committed and `||θ − θ_anchor||₂ > R`, project all params in-place
   (under `no_grad`) back onto the ball surface: `θ ← θ_a + (θ − θ_a)·R/d`.
   The model's function is thereby DEFINED on the ball; the optimizer may
   step one step outside between forwards (≈0.9 L2 at lr 1e-3, the
   registered wall fuzz) and the next forward pulls it back.
3. `wall_report()` — returns (raw, projected) L2 displacement vs anchor.

**Parameter count: 840,704 trainable — bit-identical to the control
organism. ZERO new parameters.** The delta is one buffer set (840,704 fp32
≈ 3.4 MB — optimizer-grade state, the same order as AdamW's existing m/v;
honestly co-reported; if the supervisor counts state, the number is
1,681,408 floats, still ≤1M trainable). Compute overhead: one norm per
forward (<5%). Before `commit()`, `CommittedGPT` is behaviorally
bit-identical to TinyGPT — the control arm uses the same class UNCOMMITTED,
killing any subclass confound.

### 3.2 The R grid (frozen)

Scaled-basin estimate R̂ = 1.4 L2 (low end 2.5 × sqrt(840704/2739072)).
Wall radii: **R ∈ {0.7, 1.4, 4.2} L2** = {0.5·R̂, 1.0·R̂, 3·R̂ (1.5× the
scaled high end 2.8)}. R_tight=0.7 should sit inside any plausible basin at
this size; R_mid=1.4 is the predicted knife edge; R_wide=4.2 is beyond any
plausible basin — the wash must cross the expressed region on the way to
that wall.

### 3.3 Training recipe (phase 0, shared by all arms; one chain)

All pieces are the lineage's own protocols, verbatim, at the 0.84M organism:

1. **Base**: `runs/checkpoints/e005s_small.pt` (0.84M corpus base, e005s).
2. **Install ZEPHYRA**: e043's Dmix exposure (16 paired originals + 32
   random corpus anchors, token-weighted union CE), dose s400, house cosine
   schedule (common.py `cosine_lr`, warmup 100), seed 42 — run through
   `common.py train_model` by wrapping the Dmix stream as a `CharCorpus`
   subclass whose `get_batch` emits the install mix (the literal-loop
   compliance note; ≤180 s cap).
3. **Consolidate**: e113 VERBATIM — jittered replay {-8,-4,0,+4,+8}, 300
   steps, batch 16 install + 16 anchor (8 paired + 8 random), AdamW
   (0.9,0.95) wd 0.1 constant lr 1e-3 clip 1.0, seed 10901 (≤180 s GPU;
   1800 s CPU fallback per e176n precedent).
4. **Root + commit**: the consolidated net is θ₀ for BOTH arms. Control
   arms proceed uncommitted; wall arms `deepcopy` + `commit(R)`.
5. **Wash**: `finetune_freeze` VERBATIM (e176n's copy of the e161/e176
   lineage loop) — e170's neutral anchor bank (seed 170, 0/16 junctions),
   batch 32, full-token CE, AdamW (0.9,0.95) wd 0.1 constant lr 1e-3 clip
   1.0, seed 10902, 300 steps, checkpoints {1,2,4,10,50,100,200,300}.
6. **Noise kill**: e185's construction VERBATIM at this size — labels-noise
   arm (y ~ uniform vocab, RNG seed 18501) and shuffled-target arm (seed
   18502), 10 steps, inputs bit-identical to the control's draws; plus the
   labels-noise CONTROL (uncommitted) as the size-replicate gate.

Instruments (copied, not imported, per e176n's device-policy convention):
`battery_cell/battery_pz/ce_fixed_cpu/val_windows/deleted_wpe/read_fact_at/
row_census/finetune_freeze` from e176_freeze_root; the e131 dial set
(base/held batteries at offsets −12/0/+12, site read, row census incl. row
0 and the 121–129 band, del_table) from e176n's `measure()`; the
displacement table from e185. ONE instrument adaptation, registered:
`evl_load` constructs `CommittedGPT` (so anchors load from state_dict), and
the displacement instrument co-reports raw AND projected displacement (the
g1 model's semantics are the projected weights). Full dial set at
{0, +2, +50, +300}; light g-12 at all checkpoints.

### 3.4 Arms (one file, one run; sequential, cooldowns per lab envelope)

| arm | organism | protocol | role |
|---|---|---|---|
| C  | uncommitted | neutral wash 300 | size-matched reference: the clock, D_kill, CE adaptation curve |
| W1 | commit(0.7) | neutral wash 300 | primary: does confinement hold the fact? |
| W2 | commit(1.4) | neutral wash 300 | knife edge: cliff localization |
| W3 | commit(4.2) | neutral wash 300 | wall-too-far: kill en route |
| N0 | uncommitted | labels-noise 10 | size-replicate gate for the noise kill |
| N1 | commit(0.7) | labels-noise 10 | noise under the wall |
| N2 | commit(0.7) | shuffled-target 10 | noise under the wall, arm 2 |

Trainings: 2 (phase 0) + 4 washes + 3 noise = 9, each ≤180 s GPU
(~0.2 s/step at 0.84M; total stepping ≈ 8 min) + ~40 dial sets. GPU-gated
(`gpu_ok()`, cooldowns, no concurrency — e182 may hold the GPU;
thermal-migrate per e179 precedent). Outputs: `runs/g1/metrics.json`,
`runs/g1/wall_ladder.png`, checkpoints `runs/checkpoints/g1_*.pt`.

## 4. The registered prediction, IN ADVANCE (with bars)

Conventions verbatim from the arc: **dies/dissolve** = g-12 ≤ 0.27;
**maintains** = g-12 ≥ 0.50 at EVERY checkpoint {1,2,4,10,50,100,200,300};
"dies by +50" = the +50 state. Adjudication order: GATES → WALL verdicts →
NOISE verdicts → PIN verdict → COSTS. No bar shopping; texture ⇒ TEXTURE
with numbers.

**GATES (any failure ⇒ ABORT TO TEXTURE, nothing adjudicated):**
- G-ROOT: consolidated root g-12 ≥ 0.78 (the arc's express bar; 2.74M read
  0.9156; if the 0.84M install cannot express, the organism cannot test).
- G-CTRL: arm C kills by +50 (the size-replicated wash; e185's convention).
- G-PIN: every wall arm's raw displacement ≤ R + 1.5 L2 at every
  checkpoint (the wall works mechanically; else implementation failure).
- G-NOISE: N0 kills by +10 at displacement-match (e185 at 0.84M, n=1).

**PREDICTED (the registered prior): WALL-HOLDS.**

1. **W1 maintains** — g-12 ≥ 0.50 through +300, predicted floor band
   0.55–0.85 (a dip from root, then flat; e179's r=1/32 sawtooth endpoints
   ~0.70 are the precedent shape). The two-step clock DIES: no checkpoint
   ≤ 0.27 in W1.
2. **W3 dies by +50** — the wall beyond the basin does not save; the kill
   happens en route (walk-to-wall ≈ 4.2/0.9 ≈ 5 steps ≫ the ~2-step kill).
3. **WALL-CLIFF-INSIDE** = W1 maintains AND W3 dies by +50 (the survival
   cliff lives in (0.7, 4.2]); W2's outcome locates it finer (maintains ⇒
   cliff ∈ (1.4, 4.2]; dies ⇒ cliff ∈ (0.7, 1.4]) and is co-reported
   against the control's measured D_kill — the cliff should sit at ≈D_kill.
   THE HEADLINE QUANTITATIVE TEST: the wall re-measures the basin width
   dynamically, and the two measurements (e180's t* extrapolation; g1's
   survival cliff) must agree or the law's currency is wrong.
4. **NOISE-SPARED-BY-WALL** — N1 and N2 both ≥ 0.50 at {1,2,4,10} with
   pinned displacement; the noise kill defuses exactly as the corpus kill.
   Co-bar (the organism survives too): N1/N2 CE_R within +0.3 nats of their
   root CE_R — the wall cures e185's collateral devastation as well.
   PRE-REGISTERED ANISOTROPY FORK: if N1 dips ≥0.15 below W1's floor at the
   same R, the basin is direction-thin for noise (T114's "gentleness"
   quantified: noise directions kill at smaller radius than corpus ones).
5. **FLAT-AT-PIN** — W1's g-12 from +50 to +300 changes by ≤0.05. This is
   the law's sharpest single readout: at PINNED displacement there is no
   displacement left to blame; a flat readout confirms the pure
   displacement law, an erosion exposes a second clock (see falsifier F4).

**WHAT DOES NOT CHANGE (registered invariants):**
- Step-0 dial set BIT-IDENTICAL between control root and every wall root
  (same tensors; G_ROOT-style gate max|diff| = 0.0 — the anchor is a copy).
- Install and consolidation histories identical by construction (one
  shared phase-0 chain; the wall is inert before commit).
- Anatomy at W1's +300 (the maintained state): row-0 strength ≥ 0.5× its
  root value; site_read span ≥ 0.7; band rows stay content=True — the
  e179 r=1/32 maintained template, not the e176N wreckage pattern.
- Held-30 generalization rides along: W1 held30_gm12 at +300 ≥ 0.40.
- The resurrection economy is BYPASSED, not refuted: r=0 now maintains
  (nothing to resurrect); e179's sawtooth was the un-walled organism's
  answer. The rate law's SIGNATURE VANISHES: under the wall, t* = never at
  every lr (prediction registered without a cell; lr 1e-3 is the only lr
  run — the gentle regime needed no wall anyway, e180).

## 5. The falsifiers (what shows the law is architecturally necessary even here)

- **F1 WALL-DEAF**: W1 dies (g-12 ≤ 0.27 by +50) despite G-PIN-verified
  confinement at R = 0.7 ≤ half the scaled basin. The readout died WITHOUT
  displacement: the kill is not displacement-limited — the law's MECHANISM
  (not its phenomenon) is falsified, and the no-basin law upgrades to a
  stronger architectural necessity: no static state survives optimization
  at all; only the rhythm does (W019/T120 becomes necessity, not reading).
  Candidate killers it would implicate: Adam second-moment poisoning of the
  readout pathway, LN-gain drift within the ball, or the wash gradient's
  specific action on load-bearing coordinates.
- **F2 WALL-CLIFF-MISPLACED**: survival does not order with R against
  measured D_kill (e.g., W3 maintains, or the cliff sits far from D_kill).
  The L2-position currency of the law fails to transfer across
  size/architecture — the 2.74M numbers were organism-specific (the law's
  phenomenon contingent, its width estimate non-portable).
- **F3 NOISE-PENETRATES**: N1 or N2 kills at pinned R — noise kills WITHOUT
  displacement, contradicting e185's displacement-matched mechanism at
  2.74M; the noise kill is (partly) non-geometric.
- **F4 ERODES-AT-PIN**: W1's g-12 declines monotonically ≥0.10 from +50 to
  +300 at pinned displacement — a second, residence-time clock at fixed
  geometry; "displacement-limited" was incomplete.

Every falsifier is a paper-grade finding; that is the point of W020 — wrong
predictions are the product.

## 6. The cost hypothesis (what gets worse, with bars)

- **WALL-TAXES-ADAPTATION** (predicted: FIRES): W1's corpus CE at +300 ≥
  arm C's + 0.05 nats. The wash's own adaptation (e176N drove in-batch CE
  1.36 → 0.62 at 2.74M) lives outside any small ball — the same
  displacement that kills the fact IS the stream's learning. WALL-FREE if
  |ΔCE| < 0.05.
- **Install speed: unchanged** (the wall is inert before commit; phase-0 is
  the control's own history).
- **Generalization of the fact: unchanged-or-better** (held-30 rides the
  confinement; bar in §4).
- **New-fact plasticity (the squatter problem), predicted BLOCKED, deferred
  to g1b**: a post-commit F2 install must displace the net (e174's dose
  ladder killed F1 by displacement); a wall sized to F1's basin should
  block or badly delay F2-onset. Registered as g1b's cell (F2 install under
  commit(0.7), onset vs control) — the stability–plasticity dial made
  quantitative against our own laws. NOT run in g1 (arm budget).
- **Memory-state cost**: one fp32 snapshot (3.4 MB) + <5% step compute.

## 7. Implementation sketch (against lab/common.py's interface)

```python
# lab/g1_basin_wall.py — sketch; ~full file per spec §3
import copy, torch
from common import Cfg, TinyGPT, CharCorpus, train_model, set_seed, gpu_ok, cooldown

G1_CFG = dict(n_layer=4, n_head=4, n_embd=128, block_size=256)   # 840,704 params

class CommittedGPT(TinyGPT):
    """TinyGPT + a commitment well. Zero new trainable parameters."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.R, self.anchored = None, False

    @torch.no_grad()
    def commit(self, R: float):
        for n, p in self.named_parameters():                       # snapshot
            self.register_buffer("anch__" + n.replace(".", "_"), p.detach().clone())
        self.R, self.anchored = float(R), True

    @torch.no_grad()
    def _enforce_wall(self):
        if not self.anchored: return
        anchors = {n: getattr(self, "anch__" + n.replace(".", "_"))
                   for n, _ in self.named_parameters()}
        d = sum(((p - anchors[n]) ** 2).sum() for n, p in self.named_parameters()).sqrt()
        if d > self.R:                                            # hard projection
            s = self.R / d
            for n, p in self.named_parameters():
                p.copy_(anchors[n] + (p - anchors[n]) * s)

    def forward(self, idx, targets=None):
        self._enforce_wall()                                      # forward semantics
        return super().forward(idx, targets)

    @torch.no_grad()
    def wall_report(self):                                        # g1 instrument add
        anchors = {n: getattr(self, "anch__" + n.replace(".", "_"))
                   for n, _ in self.named_parameters()} if self.anchored else {}
        d = (sum(((p - anchors[n]) ** 2).sum() for n, p in self.named_parameters())
             .sqrt().item() if self.anchored else None)
        return {"d_raw": d, "d_proj": min(d, self.R) if d is not None else None, "R": self.R}

class DmixCorpus(CharCorpus):                                     # install via train_model
    """get_batch emits e043's Dmix stream: 16 paired install windows + 32 random."""
    def get_batch(self, split, block_size, batch_size, gen=None):
        ...                                                        # e043 construction verbatim

# ---- phase 0 (ONE chain, shared by every arm; = the control's own history) ----
corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
net = CommittedGPT(Cfg(vocab=corpus.vocab_size, **G1_CFG))
net.load_state_dict(torch.load("runs/checkpoints/e005s_small.pt", map_location="cpu"))
train_model(net, DmixCorpus(...), steps=400, lr=1e-3, max_seconds=180)      # install
consolidate(net)               # e113 verbatim: jittered replay 300, seed 10901 (≤180 s)
assert root_battery(net)["g_m12"] >= 0.78                                    # G-ROOT
theta0 = copy.deepcopy(net)                                                 # shared root
full_dial_set(theta0, tag="g1_root")                                        # step-0 invariant ref

# ---- arms (finetune_freeze VERBATIM; e185 noise VERBATIM; batteries VERBATIM) ----
for tag, R in [("C", None), ("W1", 0.7), ("W2", 1.4), ("W3", 4.2)]:
    net = copy.deepcopy(theta0)
    if R: net.commit(R)
    finetune_freeze(tag, net, neutral_bank_e170(), ...seed=10902)           # 300 steps
    # per checkpoint: light g-12 + wall_report(); full dial set at {2, 50, 300}
for tag, R, ymode in [("N0", None, "labels"), ("N1", 0.7, "labels"), ("N2", 0.7, "shuffled")]:
    net = copy.deepcopy(theta0)
    if R: net.commit(R)
    noise_wash_e185(tag, net, ymode, steps=10)                              # seeds 18501/2
# verdicts per §4: gates -> WALL-HOLDS/DEAF/CLIFF-MISPLACED -> NOISE -> FLAT-AT-PIN -> costs
```

## 8. Registration discipline

Frozen at dispatch: this doc's bars, R grid, seeds (42 install / 10901
consolidation / 10902 wash draws / 170 neutral bank / 18501-2 noise),
checkpoints, adjudication order. Single seed per arm (the arc's honesty
convention; the replication debt is g1b's first cell if WALL-HOLDS). g1b
follow-ups, in priority order: (1) seeds ×3 on W1 and C; (2) F2-under-the-
wall (the squatter cost); (3) subset anchoring (which parameter classes
suffice — the architecture's factor analysis); (4) commit-point dial
(pre-consolidation commit); (5) gentle-lr wall cell; (6) the 2.74M
continuity cell (§1). No NOTES/THINKING/QUEUE/STATE edits by the executor
beyond the lab's standard entry for the run.

Builds on: e185 (displacement matching), e176N (neutral wash), e170
(neutral bank), e113 (consolidation), e043 (install), e005s_small (base),
e033/e040 (0.84M lineage), e131 (dial set), e179 (rehearsal conventions),
e180 (rate law). New: the CommittedGPT class, the R ladder, the WALL-PIN /
FLAT-AT-PIN readouts, the size-matched control arm, noise-under-wall cells.
Parameter-space confinement has never been run in this lab (e161/e176
"froze" streams, not weights).
