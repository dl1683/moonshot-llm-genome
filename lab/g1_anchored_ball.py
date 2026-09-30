"""G1 — THE ANCHORED BALL: the basin-widening architecture (W020 slot 1).

Design spec: scratch/g1_design.md (COMMITTED; bars registered IN ADVANCE —
frozen at dispatch; no bar shopping at adjudication). This docstring carries
the spec's registered prediction, gates, falsifiers and costs VERBATIM in
condensed form; adjudicate against exactly this.

THE DESIGN (spec sections 2-3): CommittedGPT(TinyGPT) — a commit(R) event
snapshots all weights into non-trainable buffers; every forward opens with a
hard in-place L2 projection onto the ball ||theta - theta_anchor||_2 <= R
(flat interior, all directions, enforced as forward semantics). ZERO new
trainable parameters: 840,704 trainable, bit-identical to the control
organism; the delta is one fp32 anchor buffer set (3.4 MB optimizer-grade
state, honestly co-reported). The dial is R in {0.7, 1.4, 4.2} L2 = {0.5,
1.0, 3} x R_hat, R_hat = 1.4 = the sqrt(P)-scaled prior basin (2.5 L2 at
2.74M -> 1.4 L2 at 0.84M; per-coordinate RMS assumed size-invariant — a
registered ESTIMATE, calibrated in-run by the control's measured D_kill).

THE SIZE CORRECTION (spec section 1, registered up front): g1 runs at the
0.84M lineage (Cfg 4L/4H/128d, block 256 = 840,704 params; base
runs/checkpoints/e005s_small.pt) to honor the <=1M cap; the stored wash
numbers (root g-12 0.9156; +1 0.678; +2 0.0271; e185 D_kill 2.489) are
2.74M-organism numbers and are PRIORS, not bars. g1 carries its own
SIZE-MATCHED CONTROL ARM (plain uncommitted CommittedGPT — behaviorally
bit-identical to TinyGPT before commit; kills any subclass confound) through
the identical battery; every verdict is adjudicated against the in-run
control.

THE RUN (spec section 3; one file, one run, one shared phase-0 chain):
  phase 0: base e005s_small.pt -> install ZEPHYRA (e043's Dmix stream: 16
  spliced install windows + 16 paired originals + 32 random corpus, dose
  s400, house cosine (warmup 100), seed 42, via common.train_model wrapping
  the mix as a CharCorpus subclass — the spec's literal-loop note) ->
  consolidate (e113 VERBATIM: jittered replay {-8,-4,0,+4,+8}, 300 steps,
  batch 16 install + 16 anchor (8 paired + 8 random), token-weighted union
  CE, AdamW (0.9,0.95) wd 0.1 constant lr 1e-3 clip 1.0, seed 10901).
  Root = theta0 for BOTH arm families.
  wash arms (e176N arm A VERBATIM: e170's neutral bank seed 170, batch 32 =
  16 neutral + 16 random, full-token CE, AdamW (0.9,0.95) wd 0.1 constant
  lr 1e-3 clip 1.0, seed 10902, 300 steps, ckpts {1,2,4,10,50,100,200,300}):
    C  uncommitted  — the size-matched reference (the clock, D_kill, CE curve)
    W1 commit(0.7)  — primary: does confinement hold the fact?
    W2 commit(1.4)  — the knife edge
    W3 commit(4.2)  — wall-too-far: the kill en route
  noise arms (e185 VERBATIM at this size; inputs bit-identical, 10 steps,
  ckpts {1,2,4,10}; targets from a dedicated generator drawn AFTER the
  aj/rj input draws):
    N0 uncommitted labels-noise (y ~ iid uniform, seed 18501) — the
       size-replicate gate for the noise kill
    N1 commit(0.7)  labels-noise  — noise under the wall
    N2 commit(0.7)  shuffled-target (flat randperm, seed 18502) — arm 2
  Instruments: the e131 dial set (base/held batteries at offsets -12/0/+12,
  site read, row census incl. row 0 + the 121-129 band, del_table) from
  e176n's measure(); the displacement table from e185 (cumulative + per-step
  L2, checkpoint deltas + cosines); full dial at {0, +2, +50, +300} (wash
  arms) and at every noise ckpt; light g-12 at all checkpoints.

========================= REGISTERED (spec section 4-6; frozen) ================
Conventions verbatim from the arc: dies/dissolve = g-12 <= 0.27; maintains
= g-12 >= 0.50 at EVERY checkpoint {1,2,4,10,50,100,200,300}; "dies by +50"
= the +50 state. Adjudication order: GATES -> WALL verdicts -> NOISE
verdicts -> PIN verdict -> COSTS. No bar shopping; texture => TEXTURE with
numbers.

GATES (any failure => ABORT TO TEXTURE, nothing adjudicated):
  G-ROOT  consolidated root g-12 >= 0.78 (the arc's express bar; 2.74M read
          0.9156; if the 0.84M install cannot express, the organism cannot
          test).
  G-CTRL  arm C kills by +50 (the size-replicated wash; e185's convention).
  G-PIN   every wall arm's raw displacement <= R + 1.5 L2 at every
          checkpoint (the wall works mechanically; else implementation
          failure).
  G-NOISE N0 kills by +10 at displacement-match (e185 at 0.84M, n=1).

PREDICTED: WALL-HOLDS.
  1. W1 maintains — g-12 >= 0.50 through +300, predicted floor band
     0.55-0.85 (a dip from root, then flat); the two-step clock DIES: no
     checkpoint <= 0.27 in W1.
  2. W3 dies by +50 — the wall beyond the basin does not save; the kill
     happens en route (walk-to-wall ~ 4.2/0.9 ~ 5 steps >> the ~2-step
     kill).
  3. WALL-CLIFF-INSIDE = W1 maintains AND W3 dies by +50 (the survival
     cliff lives in (0.7, 4.2]); W2 locates it finer (maintains => cliff in
     (1.4, 4.2]; dies => cliff in (0.7, 1.4]), co-reported against the
     control's measured D_kill — the cliff should sit at ~ D_kill. THE
     HEADLINE QUANTITATIVE TEST: the wall re-measures the basin width
     dynamically; the two measurements (e180's t* extrapolation; g1's
     survival cliff) must agree or the law's currency is wrong.
  4. NOISE-SPARED-BY-WALL — N1 and N2 both >= 0.50 at {1,2,4,10} with
     pinned displacement; co-bar: N1/N2 CE_R within +0.3 nats of their root
     CE_R. ANISOTROPY FORK: if N1 dips >= 0.15 below W1's floor at the same
     R, the basin is direction-thin for noise.
  5. FLAT-AT-PIN — W1's g-12 from +50 to +300 changes by <= 0.05.

FALSIFIERS (each a paper-grade finding):
  F1 WALL-DEAF          W1 dies (g-12 <= 0.27 by +50) despite G-PIN-verified
                        confinement at R = 0.7 <= half the scaled basin: the
                        readout died WITHOUT displacement — the kill is not
                        displacement-limited; no-basin upgrades to a stronger
                        architectural necessity (only the rhythm survives).
  F2 WALL-CLIFF-MISPLACED survival does not order with R against measured
                        D_kill (e.g. W3 maintains, or the cliff sits far
                        from D_kill): the L2-position currency fails to
                        transfer across size/architecture.
  F3 NOISE-PENETRATES   N1 or N2 kills at pinned R — noise kills WITHOUT
                        displacement; the noise kill is (partly)
                        non-geometric.
  F4 ERODES-AT-PIN      W1's g-12 declines monotonically >= 0.10 from +50
                        to +300 at pinned displacement — a second,
                        residence-time clock at fixed geometry.

COSTS: WALL-TAXES-ADAPTATION (predicted FIRES) = W1's corpus CE at +300 >=
arm C's + 0.05 nats (WALL-FREE if |dCE| < 0.05); install speed unchanged
(the wall is inert before commit); held30 generalization: W1 held30_gm12 at
+300 >= 0.40; anatomy at W1's +300: row-0 strength >= 0.5x root, site_read
span >= 0.7, band rows content=True (the e179 r=1/32 template); memory
cost 3.4 MB + <5% step compute.

INVARIANTS: step-0 tensors bit-identical between the control root and every
wall root (the anchor is a copy; max|diff| = 0.0); install/consolidation
histories identical by construction (one shared phase-0 chain; the wall is
inert before commit); per-step input batches bit-identical across ALL
SEVEN arms (same seed-10902 draw sequence; md5-gated).

INSTRUMENT PROVENANCE (spec section 3.3): battery_cell/battery_pz/
ce_fixed_cpu/val_windows/deleted_wpe/read_fact_at/row_census/measure()/
flat_cells/gate_vs are lab/e176n_neutral_wash.py VERBATIM (= e185's copies
= the e176/e161/e152/e151/e143/e131/e119/e113/e068/e065/e043 lineage); the
wash/noise trainer is e185's noise_wash VERBATIM (= e176n's finetune_freeze
+ target substitution + displacement bookkeeping) + the wall bookkeeping;
the consolidation is e113's finetune_arm VERBATIM; the neutral bank is
e170 VERBATIM via e176n's copy; pick_dev/migrate_to_cpu are e184's via
e179/g2. Copied, not imported, to own the device policy. TWO instrument
adaptations, registered by the spec: (i) evl_load constructs CommittedGPT
(so anchors load from state_dict); (ii) the displacement instrument
co-reports raw AND projected displacement (the g1 model's semantics are
the projected weights).

COMPUTE ENVELOPE (dispatch): GPU allowed (gpu_ok() gate, park-once +
mid-run poll every 25 steps; e182 may hold the GPU — park to CPU, the
0.84M net runs fine there); torch threads 8; cooldown 60 s before each
training (the dispatch's 60-120 s band); per-training cap 180 s GPU /
1800 s CPU (e176n precedent); NO concurrent GPU.

Outputs: runs/g1/{metrics.json, anchored_ball.png}; checkpoints
runs/checkpoints/g1_*.pt. No NOTES/THINKING/QUEUE/STATE edits; single
commit, no push.

Run:  cd lab && python g1_anchored_ball.py    (G1_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # GPU allowed, gated below

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

torch.set_num_threads(8)                              # e152R/e143/e184/e179

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT, cooldown, gpu_ok,     # noqa: E402
                    gpu_status, run_dir, save_json, set_seed,
                    train_model)
import e043_install as E43                             # noqa: E402 (REPO, find_occ,
                                                      # SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("G1_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
BASE_CK = "e005s_small.pt"        # the 0.84M corpus base (e005s)
CKPT_DIR = E43.REPO / "runs" / "checkpoints"

# ---- the 0.84M organism (e033/e040 lineage; e005s_small's own cfg) -------------
G1_CFG = Cfg(vocab=65, n_layer=4, n_head=4, n_embd=128, block_size=256)
G1_PARAMS = 840_704

# ---- placement constants (e152/e176n/e185 verbatim) ---------------------------
RETEACH_J = 54                    # measurement pool: name x-cols 184..190
SITE_ADDR_ROW = 183               # the onset read row (= e120's SPLICE_ADDR_ROW)
SITE_Z_XCOL = PRE + RETEACH_J     # 184
SITE_CONT = BLOCK - PRE - RETEACH_J - len(NAME)     # 65-token continuation
D_ALL = (121, 125, 129, 133, 137)                   # e113's fixed set
GEOS = (-12, 0, 12)               # novel x2 + the trained g0
JITTERS = (-8, -4, 0, 4, 8)       # e109/e113's registered jitter set

ROWS_OLD = (0, 1, 2, 3, 4, 5, 6, 60, 100, 118, 119, 120) + tuple(range(121, 130))
if SMOKE:
    ROWS_OLD = (0, 1, 60, 121, 125, 129)

# ---- the R ladder (spec section 3.2, FROZEN) -----------------------------------
R_HAT = 2.5 * (G1_PARAMS / 2_739_072) ** 0.5         # ~1.38 -> prior basin
R_LADDER = (0.7, 1.4, 4.2)         # {0.5, 1.0, 3} x R_hat
SCALED_BASIN = (1.4, 2.8)         # sqrt(P)-scaled 2.5-5 L2 (registered estimate)

# ---- phase 0 (spec section 3.3, one shared chain) ------------------------------
INSTALL_SEED = 42                 # the Dmix stream seed (spec section 8)
INSTALL_STEPS = 400 if not SMOKE else 8
CONS_SEED = 10901                 # e109/e113 arm (a) seed VERBATIM
CONS_STEPS = 300 if not SMOKE else 8
NAME_BS = 16                      # e043: install windows per exposure step
CORP_BS = 48                      # e043 Dmix: 16 paired originals + 32 random
MIX_RANDOM = 32
FT_LR = 1e-3
TRAIN_CAP_GPU, TRAIN_CAP_CPU = 180.0, 1800.0

# ---- the wash / noise envelope (e176N arm A / e185 VERBATIM) -------------------
FREEZE_SEED = 10902               # the locked seed lineage
NOISE_SEED_A = 18501              # labels-noise target RNG (e185)
NOISE_SEED_B = 18502              # shuffled-target RNG (e185)
CK_WASH: tuple[int, ...] = (1, 2, 4, 10, 50, 100, 200, 300) if not SMOKE \
    else (1, 2, 4)
CK_NOISE: tuple[int, ...] = (1, 2, 4, 10) if not SMOKE else (1, 2)
FULL_DIAL_WASH: tuple[int, ...] = (2, 50, 300) if not SMOKE else ()
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 neutral + 16 random
COOLDOWN_S = 60.0                 # dispatch band 60-120 s (before each training)
MIDRUN_POLL_EVERY = 25            # mid-run GPU contention poll cadence

# ---- e170's neutral anchor bank (e176N arm A's stream, VERBATIM) ---------------
E170_ANCHOR_SEED = 170
ANCHOR_FORBIDDEN = ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")

# ---- gates / bars (frozen) ------------------------------------------------------
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
G_BIT_TOL = 5e-6
SHUT_BAR = 0.27                   # dies (e158/e161/e176/e176N/e184/e185)
MAINTAIN_BAR = 0.50               # maintains
EXPRESS_BAR = 0.78                # G-ROOT (the arc's express bar)
PIN_FUZZ_BAR = 1.5                # G-PIN: raw displacement <= R + this
FLAT_BAR = 0.05                   # FLAT-AT-PIN |g(+300) - g(+50)|
ERODE_BAR = 0.10                  # F4 monotone decline from +50 to +300
CE_NOISE_BAR = 0.30               # N1/N2 CE_R within root + this
CE_TAX_BAR = 0.05                 # WALL-TAXES-ADAPTATION
ANISO_BAR = 0.15                  # N1 floor dips >= this below W1's floor
HELD30_BAR = 0.40                 # W1 held30_gm12 at +300

# e185/e176N stored 2.74M cells (PRIORS, not bars — the size correction)
PRIORS_274M = {
    "source": "runs/e176n + runs/e185 metrics (the 2.74M e131 organism)",
    "root_gm12": 0.9155886769294739,
    "plus1_gm12": 0.6780440807342529,
    "plus2_gm12": 0.027077054604887962,
    "e185_D_kill": 2.4892616271972656,
    "e185_t_kill": 2,
    "e180_fit": "t* ~ 7.5e-4 * lr^-1.16 (R^2 0.975); lr*t* ~ const",
}

REGISTERED_PREDICTION = {
    "gates": "G-ROOT root g-12 >= 0.78 | G-CTRL arm C kills by +50 | "
             "G-PIN every wall arm raw displacement <= R + 1.5 at every "
             "checkpoint | G-NOISE N0 kills by +10 at displacement-match. "
             "Any failure => ABORT TO TEXTURE, nothing adjudicated.",
    "predicted": "WALL-HOLDS: (1) W1 maintains (g-12 >= 0.50 through +300, "
                 "floor band 0.55-0.85; no checkpoint <= 0.27); (2) W3 dies "
                 "by +50 (the kill en route to the wall); (3) "
                 "WALL-CLIFF-INSIDE with W2 locating the cliff, which "
                 "should sit at ~ D_kill (the headline quantitative test "
                 "vs e180's t* extrapolation); (4) NOISE-SPARED-BY-WALL "
                 "(N1, N2 >= 0.50 at {1,2,4,10} pinned; CE_R within root + "
                 "0.3); (5) FLAT-AT-PIN (|W1 g-12(+300) - g-12(+50)| <= "
                 "0.05).",
    "falsifiers": "F1 WALL-DEAF (W1 dies by +50 at verified pin) | F2 "
                  "WALL-CLIFF-MISPLACED (survival does not order with R "
                  "against D_kill, e.g. W3 maintains) | F3 NOISE-PENETRATES "
                  "(N1/N2 kill at pinned R) | F4 ERODES-AT-PIN (W1 declines "
                  "monotonically >= 0.10 from +50 to +300).",
    "costs": "WALL-TAXES-ADAPTATION predicted FIRES (W1 corpus CE at +300 "
             ">= C's + 0.05); WALL-FREE if |dCE| < 0.05; install speed "
             "unchanged; W1 held30_gm12(+300) >= 0.40; W1 +300 anatomy: "
             "row0 >= 0.5x root, site span >= 0.7, band rows content=True.",
    "invariants": "step-0 tensors bit-identical control root vs every wall "
                  "root (max|diff| = 0.0); one shared phase-0 chain; "
                  "per-step inputs bit-identical across all seven arms.",
    "registration": "scratch/g1_design.md sections 4-6 + section 8 (seeds, "
                    "checkpoints, adjudication order) VERBATIM, committed "
                    "before this implementation; frozen — no bar shopping.",
}

trims: list[str] = []
device_events: list[dict] = []
deviations: list[str] = [
    "FILENAME: the spec sketch names lab/g1_basin_wall.py + "
    "runs/g1/wall_ladder.png; the dispatch mission names lab/"
    "g1_anchored_ball.py + runs/g1/anchored_ball.png — implemented under "
    "the mission's names (same spec).",
    "INSTALL VIA train_model (the spec's literal-loop note): the e043 Dmix "
    "stream is wrapped as a CharCorpus subclass (DmixCorpus) and runs "
    "through common.train_model — full-token CE over the 64-window union "
    "batch (16 spliced install windows incl. their 7 name-char targets at "
    "natural token weight 112/16320 = 0.69%, + 16 paired originals + 32 "
    "random corpus). e043's name-position mask (which EXCLUDED the install "
    "windows' non-name tokens; name weight 112/12352 = 0.91%) is thereby "
    "dropped — the registered adaptation; its consequence is owned by "
    "G-ROOT: if the 0.84M install cannot express (g-12 < 0.78), the run "
    "aborts to TEXTURE per the spec's own gate. Optimizer/schedule/clip "
    "are train_model's = e043 exposure's (AdamW (0.9,0.95) wd 0.1, house "
    "cosine warmup 100, clip 1.0); the draw order ix(16)-aj(16)-rj(32) is "
    "e043's; the generator is train_model's own, seeded by the corpus seed "
    "= 42 (the spec's install seed).",
    "anch__R: R joins the anchor buffers as a 1-element fp32 tensor "
    "(anch__R) so the commitment state (radius + anchors) round-trips "
    "through state_dict; the spec's sketch kept R as a python float which "
    "does not survive a checkpoint. 45 anchor tensors + 1 scalar, "
    "registered.",
    "EVAL SEMANTICS PIVOT (the second instrument adaptation, extended from "
    "the spec's 'evl_load constructs CommittedGPT'): evl_load SETTLES the "
    "loaded net onto the ball first (one no-grad _enforce_wall — the "
    "model's defined state), then DISARMS the wall for the measurement. "
    "For non-perturbative dials (batteries, site read, CE_R) settle+disarm "
    "is EXACTLY armed semantics (a settled net never projects again: d <= "
    "R at every forward). For PERTURBATIVE instruments (row census, wpe "
    "deletion table) the disarm is necessary: their probes deliberately "
    "leave the ball, and projecting a zeroed row back would measure the "
    "projection, not the probe — and would silently mutate the twin. The "
    "deletion table therefore slices the SETTLED state dict (bit-equal to "
    "the raw sd for uncommitted arms). The in-loop light evals (inside the "
    "trainer) run on an ARMED twin (full wall semantics); on-disk "
    "checkpoints store the RAW post-step weights (loading into "
    "CommittedGPT re-settles at eval).",
    "DEVICE: batches are built on CPU and moved to the training device; "
    "common.DEVICE is set to the training device for the install's "
    "train_model call; GPU is gated (pick_dev park-once double-poll + "
    "mid-run contention poll every 25 steps with in-place migration, "
    "e184/e179/g2 precedent); per-training caps 180 s GPU / 1800 s CPU.",
    "CHECKPOINTS: g1_root + per-arm finals + every noise-arm checkpoint "
    "are written; the wash arms' intermediate checkpoints live in "
    "metrics.json (sds are 6.8 MB each with anchors; e185's save-all "
    "convention would add ~200 MB).",
    "Cooldowns are 60 s (the dispatch's 60-120 s band), BEFORE each "
    "training only — the next training's pre-cooldown and the CPU dial "
    "time between trainings cover the after-gap (g2's recorded trim).",
    "G-ROOT failure handling: the arms still run and are reported, but the "
    "verdict is TEXTURE (GATE FAILURE: G-ROOT) — nothing adjudicated (the "
    "spec's own abort clause; running the arms costs minutes and completes "
    "the record).",
    "GPU float nondeterminism vs the CPU-eval'd 2.74M references: every "
    "bar is adjudicated against the IN-RUN size-matched control; the "
    "stored 2.74M numbers are priors only (the size correction).",
    "Single seed per arm (install 42 / consolidation 10901 / wash draws "
    "10902 / noise 18501-2; one trajectory per arm, n=1 per cell) — the "
    "arc's honesty convention; the replication debt is g1b's first cell.",
    "Smoke mode trims: 8-step install/consolidation, 4-step washes, "
    "2-step noise, lean dials, no cooldowns — nothing adjudicated.",
]


# ------------------------------------------------------------------ device pick
# PROVENANCE: lab/e184_seed_replicates.py VERBATIM via e179/g2.

GPU_PARKED = False
PARK_REASON = None


def pick_dev(tag: str) -> torch.device:
    """Strict pre-training quick check: gpu_ok() double-poll 5 s apart;
    PARK-ONCE — any failure parks every remaining training to CPU."""
    global GPU_PARKED, PARK_REASON
    if GPU_PARKED:
        log(f"[gpu] '{tag}' CPU (PARKED: {PARK_REASON})")
        return CPU
    if not torch.cuda.is_available():
        GPU_PARKED, PARK_REASON = True, "no CUDA"
        return CPU
    s1 = gpu_status()
    if gpu_ok():
        time.sleep(5)
        if gpu_ok():
            s2 = gpu_status()
            log(f"[gpu] '{tag}' may use GPU (util {s2['util']:.0f}% temp "
                f"{s2['temp']:.0f}C mem {s2['mem_used']:.0f}/"
                f"{s2['mem_total']:.0f}MB)")
            return torch.device("cuda")
    GPU_PARKED = True
    PARK_REASON = f"quick check failed: {s1}"
    log(f"[gpu] PARK — '{tag}' and all remaining trainings run CPU ({s1})")
    return CPU


MIDRUN_PAUSE = False   # default off (legacy behaviour = migrate); g1bW sets True


def midrun_pause_wait(tag: str, temp_resume: float = 75.0) -> None:
    """Block until outside GPU load / heat clears (mem <= 85%, temp <= resume)."""
    while True:
        s = gpu_status()
        if s["mem_total"] == 0 or (s["mem_used"] <= 0.85 * s["mem_total"]
                                   and s["temp"] <= temp_resume):
            log(f"  [{tag}] contention cleared ({s}); resuming")
            return
        log(f"  [{tag}] PAUSED for outside load/heat ({s})")
        time.sleep(30)


def migrate_to_cpu(net, opt) -> None:
    """Move net + optimizer state to CPU in place (params persist)."""
    net.to("cpu")
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state.get(p, {})
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to("cpu")


# ------------------------------------------------------------------ the wall
# PROVENANCE: scratch/g1_design.md section 7 sketch, extended only by the
# anch__R buffer (state_dict round-trip) + the settle/disarm instrument API.

class CommittedGPT(TinyGPT):
    """TinyGPT + a commitment well. ZERO new trainable parameters.

    Before commit() this class is behaviorally bit-identical to TinyGPT
    (the control arm uses it UNCOMMITTED — no subclass confound). After
    commit(R): every forward opens with a hard in-place L2 projection onto
    the ball ||theta - theta_anchor||_2 <= R (flat interior, all
    directions); the optimizer may step one step outside between forwards
    (the registered wall fuzz ~ lr*sqrt(P) = 0.92 L2 at lr 1e-3, 0.84M)
    and the next forward pulls the parameters back onto the surface.
    """

    def __init__(self, cfg: Cfg):
        super().__init__(cfg)
        self.R: float | None = None
        self.anchored = False

    @torch.no_grad()
    def commit(self, R: float) -> None:
        """Snapshot every trainable tensor into registered non-trainable
        buffers (anch__<name>); set the radius. One protocol event."""
        n = 0
        for name, p in self.named_parameters():
            self.register_buffer("anch__" + name.replace(".", "_"),
                                 p.detach().clone())
            n += 1
        self.register_buffer("anch__R", torch.tensor(float(R)))
        self.R, self.anchored = float(self.anch__R.item()), True
        self._n_anchor_tensors = n

    def _anchor(self, name: str):
        return getattr(self, "anch__" + name.replace(".", "_"))

    @torch.no_grad()
    def _enforce_wall(self) -> None:
        """THE WALL: hard in-place L2 projection onto the ball surface if
        outside. No-op when uncommitted (the control arms)."""
        if not self.anchored:
            return
        d = None
        for name, p in self.named_parameters():
            a = self._anchor(name)
            dd = ((p - a) ** 2).sum()
            d = dd if d is None else (d + dd)
        d = float(d.sqrt())
        if d > self.R:
            s = self.R / d
            for name, p in self.named_parameters():
                a = self._anchor(name)
                p.copy_(a + (p - a) * s)

    def forward(self, idx, targets=None):
        self._enforce_wall()                       # forward semantics
        return super().forward(idx, targets)

    @torch.no_grad()
    def wall_report(self) -> dict:
        """The g1 instrument add: raw + projected L2 displacement vs anchor."""
        if not self.anchored:
            d = float(torch.norm(torch.cat(
                [p.detach().reshape(-1) for p in self.parameters()])))
            return {"d_raw": None, "d_proj": None, "R": None,
                    "theta_norm": d}
        d = None
        for name, p in self.named_parameters():
            dd = ((p - self._anchor(name)) ** 2).sum()
            d = dd if d is None else (d + dd)
        d = float(d.sqrt())
        return {"d_raw": d, "d_proj": min(d, self.R), "R": self.R}


def split_anchored_sd(sd: dict) -> tuple[dict, dict]:
    """(body sd, anch__ buffers incl. anch__R)."""
    body = {k: v for k, v in sd.items() if not k.startswith("anch__")}
    anch = {k: v for k, v in sd.items() if k.startswith("anch__")}
    return body, anch


def load_g1(path) -> CommittedGPT:
    """Load a plain TinyGPT-lineage checkpoint into CommittedGPT (CPU)."""
    m = CommittedGPT(G1_CFG)
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    body, anch = split_anchored_sd(sd)
    m.load_state_dict(body)
    if anch:
        _restore_anchors(m, anch)
    m.eval()
    return m


def _restore_anchors(m: CommittedGPT, anch: dict) -> None:
    for k, v in anch.items():
        m.register_buffer(k, v.clone())     # incl. anch__R (state_dict parity)
    m.R = float(anch["anch__R"])
    m.anchored = True


def evl_load(sd: dict) -> CommittedGPT:
    """THE registered instrument adaptation: constructs CommittedGPT (so
    anchors load from state_dict), SETTLES onto the ball (the model's
    defined state), then DISARMS for measurement — see the PIVOT deviation.
    For uncommitted sds this is a plain TinyGPT load."""
    m = CommittedGPT(G1_CFG)
    body, anch = split_anchored_sd(sd)
    m.load_state_dict(body)
    if anch:
        _restore_anchors(m, anch)
        m._enforce_wall()          # settle: the model's defined configuration
        m.anchored = False         # disarm perturbative instruments (PIVOT)
    m.eval()
    return m


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/e176n_neutral_wash.py VERBATIM (= e185's copies; the
# e176/e161/e152/e151/e143/e131/e119/e113/e068/e065/e043 lineage), on the
# 0.84M cfg. Copied rather than imported to own the device policy.

@torch.no_grad()
def battery_cell(net, ids: torch.Tensor, zid: int, bs=30) -> dict:
    """e068/e113/e120/e151 battery on CPU: p(Z) at the last position."""
    net.eval()
    pzs, amax = [], 0.0
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs.append(pr[:, zid])
        amax += float((lg[:, -1].argmax(-1) == zid).float().sum())
    p = torch.cat(pzs)
    return {"mean_pz": float(p.mean()), "median_pz": float(p.median()),
            "std_pz": float(p.std()),
            "frac_pz_ge_0.5": float((p >= 0.5).float().mean()),
            "frac_argmax_z": amax / ids.shape[0]}


@torch.no_grad()
def battery_pz(net, ids: torch.Tensor, zid: int, bs=30) -> float:
    """e116's scalar battery (census readout)."""
    net.eval()
    pzs = []
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs += [float(pr[k, zid]) for k in range(pr.shape[0])]
    return float(np.mean(pzs))


@torch.no_grad()
def ce_fixed_cpu(net, x, y, bs=64) -> float:
    net.eval()
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        _, loss = net(x[i:i + bs], y[i:i + bs])
        tot += float(loss.item()) * len(x[i:i + bs])
        n += len(x[i:i + bs])
    return tot / max(n, 1)


def val_windows(val_ids, val_text, n, seed, block=256):
    """e065 val_windows verbatim: name-free val-split windows."""
    g = torch.Generator().manual_seed(seed)
    out_x, out_y = [], []
    tries = 0
    while len(out_y) < n and tries < 500 * n:
        i = int(torch.randint(len(val_ids) - block - 1, (1,), generator=g))
        txt = val_text[i: i + block + 1]
        if "ZEPHYRA" in txt or "ZEPH" in txt:
            continue
        out_x.append(val_ids[i: i + block])
        out_y.append(val_ids[i + 1: i + 1 + block])
    return torch.stack(out_x), torch.stack(out_y)


def deleted_wpe(sd: dict, rows: tuple[int, ...]) -> tuple[dict, dict]:
    """D2 subtractive row-zero with the e065/e113 confinement gate."""
    out = {k: v.clone() for k, v in sd.items()}
    for r in rows:
        out["wpe.weight"][r] = 0.0
    d = out["wpe.weight"] != sd["wpe.weight"]
    changed_rows = sorted(set(int(r) for r in torch.nonzero(d)[:, 0].tolist()))
    n = int(d.sum().item())
    others = all(torch.equal(out[k], sd[k]) for k in out if k != "wpe.weight")
    gate = {"rows": list(rows), "n_elements_changed": n,
            "expected": len(rows) * sd["wpe.weight"].shape[1],
            "changed_rows": changed_rows,
            "confined": bool(changed_rows == sorted(rows)),
            "others_bit_identical": bool(others),
            "pass": bool(n == len(rows) * sd["wpe.weight"].shape[1]
                         and changed_rows == sorted(rows) and others)}
    return out, gate


@torch.no_grad()
def read_fact_at(net, pool_x: torch.Tensor, name_ids, zid: int,
                 addr_row: int, xcol: int, bs=30) -> dict:
    """e131's read_fact_position VERBATIM ARITHMETIC: p(true name char) at
    positions addr_row..addr_row+6 over the pool windows."""
    net.eval()
    n_name = len(name_ids)
    per_pos = [[] for _ in range(n_name)]
    onset = []
    for i in range(0, pool_x.shape[0], bs):
        w = pool_x[i:i + bs]
        lg, _ = net(w)
        pr = F.softmax(lg, -1)
        for k in range(w.shape[0]):
            onset.append(float(pr[k, addr_row, int(zid)]))
            for j in range(n_name):
                per_pos[j].append(
                    float(pr[k, addr_row + j, int(w[k, xcol + j])]))
    onset_t = torch.tensor(onset)
    allp_t = torch.tensor([p for pos in per_pos for p in pos])
    return {"pz_onset_mean": float(onset_t.mean()),
            "pz_onset_median": float(onset_t.median()),
            "pz_onset_frac_ge_0.5": float((onset_t >= 0.5).float().mean()),
            "pname_mean_over7": float(allp_t.mean()),
            "pname_frac_ge_0.5": float((allp_t >= 0.5).float().mean()),
            "per_position_mean": [float(np.mean(pos)) for pos in per_pos]}


def row_census(net, rows, readout, *rargs) -> dict:
    """e139's row_census VERBATIM (mean-arm / zero-arm / restore)."""
    net.eval()
    base = readout(net, *rargs)
    w = net.wpe.weight.data
    orig = w.clone()
    mean_row = orig.mean(0)
    m_d, z_d = {}, {}
    for r in rows:
        w.copy_(orig); w[r] = mean_row
        m_d[r] = base - readout(net, *rargs)
        w.copy_(orig); w[r] = 0.0
        z_d[r] = base - readout(net, *rargs)
    w.copy_(orig)
    rows_d = {str(r): {"mean": float(m_d[r]), "zero": float(z_d[r]),
                       "ratio": float(min(m_d[r], z_d[r]) /
                                      max(m_d[r], z_d[r]))
                              if max(m_d[r], z_d[r]) > 0 else 0.0,
                       "strength": float(min(m_d[r], z_d[r])),
                       "content": bool(m_d[r] > 0 and z_d[r] > 0 and
                                       min(m_d[r], z_d[r]) /
                                       max(m_d[r], z_d[r]) >= 0.5)}
              for r in rows}
    assert torch.equal(w, orig), "census failed to restore wpe"
    return {"base_readout": base, "rows": rows_d}


# ------------------------------------------------------------------ streams

class DmixCorpus(CharCorpus):
    """e043's Dmix stream wrapped for common.train_model (the spec's
    literal-loop note). get_batch draws e043's exposure composition IN
    ORDER — ix(16) spliced install windows, aj(16) paired original host
    windows, rj(32) random corpus windows — from the passed generator
    (train_model seeds it with self.seed = the install seed 42); the batch
    is (64, 255) x/y. train_model's full-token CE trains the union (the
    mask drop is the registered adaptation; see deviations). estimate_loss
    reads self.train/self.val directly (real corpus) — unaffected."""

    def __init__(self, path, seed: int, inst_x: torch.Tensor,
                 anchor_full: torch.Tensor, train_ids: torch.Tensor):
        super().__init__(path, seed=seed)
        self.inst_x = inst_x                # (60, 256) spliced install windows
        self.anchor_full = anchor_full      # (60, 256) original host windows
        self.train_ids = train_ids

    def get_batch(self, split, block_size, batch_size, gen=None):
        g = gen if gen is not None else torch.Generator().manual_seed(self.seed)
        ix = torch.randint(self.inst_x.shape[0], (NAME_BS,), generator=g)
        aj = torch.randint(self.anchor_full.shape[0], (CORP_BS - MIX_RANDOM,),
                           generator=g)
        rj = torch.randint(len(self.train_ids) - BLOCK - 1, (MIX_RANDOM,),
                           generator=g)
        corp = torch.cat([self.anchor_full[aj],
                          torch.stack([self.train_ids[s: s + BLOCK]
                                       for s in rj])], 0)
        nw = self.inst_x[ix]
        x = torch.cat([nw[:, :-1], corp[:, :-1]], 0)
        y = torch.cat([nw[:, 1:], corp[:, 1:]], 0)
        return x.to(common.DEVICE), y.to(common.DEVICE)


# ------------------------------------------------------------------ training
# PROVENANCE: the consolidation is e113's finetune_arm VERBATIM arithmetic
# (device-parameterized; e176n/g2 precedent); the wash/noise trainer is
# e185's noise_wash VERBATIM (= e176n's finetune_freeze + target
# substitution + displacement bookkeeping) + the wall bookkeeping.

def consolidate(net0: CommittedGPT, pool_x, pool_mask, anchor, train_ids,
                f_eval_ids, r_eval_xy, zid, tag) -> dict:
    """e113's finetune_arm VERBATIM (recipe/seed/batch composition): batch
    32 = 16 install windows from the jittered pool + 16 anchors (8 paired
    originals + 8 random); e043 token-level union CE (name-masked on pool
    windows); constant lr 1e-3 AdamW (0.9,0.95) wd 0.1 clip 1.0; 300 steps,
    seed 10901. In-loop CPU evals every 25 steps."""
    dev = pick_dev(tag)
    cap = TRAIN_CAP_GPU if dev.type == "cuda" else TRAIN_CAP_CPU
    net = copy.deepcopy(net0).to(dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(CONS_SEED)
    n_pool, n_anc = pool_x.shape[0], anchor.shape[0]
    traj, t_start = [], time.time()
    evl = copy.deepcopy(net0).to(CPU)          # CPU eval twin (uncommitted)
    step = 0
    for step in range(1, CONS_STEPS + 1):
        ix = torch.randint(n_pool, (16,), generator=gen)
        aj = torch.randint(n_anc, (8,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (8,), generator=gen)
        nw = pool_x[ix].to(dev)
        anc = torch.cat([anchor[aj],
                         torch.stack([train_ids[s: s + BLOCK] for s in rj])],
                        0).to(dev)
        x = torch.cat([nw[:, :-1], anc[:, :-1]], 0)
        y = torch.cat([nw[:, 1:], anc[:, 1:]], 0)
        m = torch.zeros(32, x.shape[1], dtype=torch.bool, device=dev)
        m[:16] = pool_mask[ix].to(dev)
        logits, _ = net(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                              y.reshape(-1), reduction="none"
                              ).view(x.shape[0], x.shape[1])
        nm = nll[:16][m[:16]]
        cm = nll[16:]
        loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == CONS_STEPS or \
                (time.time() - t_start) > cap:
            sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            evl.load_state_dict(sd_cpu)
            evl.eval()
            bz = battery_cell(evl, f_eval_ids, zid)
            ce_r = ce_fixed_cpu(evl, *r_eval_xy)
            traj.append({"step": step, "install60_g0_pz": bz["mean_pz"],
                         "ce_r": ce_r,
                         "elapsed_s": round(time.time() - t_start, 1)})
            log(f"  [{tag}] s{step:4d} g0 {bz['mean_pz']:.4f} CE_R {ce_r:.4f} "
                f"({traj[-1]['elapsed_s']:.0f}s)")
        if (time.time() - t_start) > cap:
            log(f"  [{tag}] time cap {cap:.0f}s at s{step}")
            trims.append(f"{tag}: time cap at step {step}")
            break
        if dev.type == "cuda" and step % MIDRUN_POLL_EVERY == 0:
            s = gpu_status()
            if s["mem_total"] > 0 and (s["mem_used"] > 0.85 * s["mem_total"]
                                       or s["temp"] > 80):
                device_events.append({"tag": tag, "step": step,
                                      "event": "MID-RUN MIGRATION", "status": s})
                log(f"  [{tag}] MID-RUN GPU contention at s{step} ({s}) -> CPU")
                migrate_to_cpu(net, opt)
                dev = CPU
    net.eval()
    sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"sd": sd_cpu, "traj": traj, "steps_ran": step,
            "device": str(dev), "seed": CONS_SEED}


def flat_params_cpu(net) -> torch.Tensor:
    """The fp32 flat parameter vector (all trainable tensors,
    net.parameters() order — the optimizer's own currency; 840,704
    elements on this line)."""
    return torch.cat([p.detach().reshape(-1).cpu()
                      for p in net.parameters()]).clone()


def g1_wash(tag: str, net0: CommittedGPT, anchor: torch.Tensor,
            train_ids: torch.Tensor, itos, r_eval_xy, gm12_ids, g0_ids,
            zid: int, target_mode: str = "true", noise_seed: int = 0,
            ckpt_steps: tuple[int, ...] = CK_WASH, lr: float = FT_LR,
            seed: int = FREEZE_SEED):
    """THE NEUTRAL PLAIN-CORPUS WASH / NOISE WASH (e185's noise_wash
    VERBATIM arithmetic + the wall bookkeeping). Per step: aj = randint(16)
    anchor draws, rj = randint(16) random corpus offsets (the seed-10902
    generator — BIT-IDENTICAL across ALL arms); THEN the target
    substitution from the dedicated noise generator ('true' = the real
    wash; 'iid' = y ~ uniform(vocab); 'perm' = flat randperm of the true
    targets). Batch 32 full-token CE; AdamW (0.9,0.95) wd 0.1 constant lr,
    clip 1.0. Wall: every forward projects (committed arms); displacement
    co-reports raw + projected. In-batch CE at EVERY step; snapshots +
    ARMED-twin light CPU evals (g-12, g0, CE_R — no RNG consumed) at the
    checkpoint steps; checkpoint delta vectors kept for the cosines."""
    assert target_mode in ("true", "iid", "perm")
    dev = pick_dev(tag)
    cap = TRAIN_CAP_GPU if dev.type == "cuda" else TRAIN_CAP_CPU
    ckpt_set = set(ckpt_steps)
    n_steps = ckpt_steps[-1]
    net = copy.deepcopy(net0).to(dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    ngen = (torch.Generator().manual_seed(noise_seed)
            if target_mode != "true" else None)
    vocab = len(itos)
    n_anc = anchor.shape[0]
    traj, sds, t_start = [], {}, time.time()
    step = 0
    zeph_checks = 0
    evl = copy.deepcopy(net0).to(CPU)          # CPU eval twin (ARMED)
    theta0 = flat_params_cpu(net)              # the displacement origin
    prev = theta0.clone()
    deltas: dict[int, torch.Tensor] = {}
    x_hashes: dict[int, str] = {}
    y_stats: list[dict] = []
    wall_R = net0.R
    for step in range(1, n_steps + 1):
        aj = torch.randint(n_anc, (ANCH_BS,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                           generator=gen)
        anc = anchor[aj]
        rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj])
        # name-free VERIFY (no-op by corpus construction; hard-fail if not)
        for w in rnd:
            txt = "".join(itos[int(c)] for c in w[:64]) + \
                  "".join(itos[int(c)] for c in w[192:])
            if "ZEPH" in txt:
                zeph_checks += 1
        x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
        y_true = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
        # ---- THE TARGET SUBSTITUTION (dedicated RNG, drawn AFTER aj/rj)
        if target_mode == "true":
            y = y_true
        elif target_mode == "iid":
            y = torch.randint(vocab, y_true.shape, generator=ngen)
        else:  # perm
            perm = torch.randperm(y_true.numel(), generator=ngen)
            y = y_true.reshape(-1)[perm].reshape(y_true.shape)
        x_hashes[step] = hashlib.md5(
            x.contiguous().numpy().tobytes()).hexdigest()
        if target_mode != "true":
            y_stats.append({
                "step": step,
                "frac_targets_changed": float((y != y_true).float().mean()),
                "multiset_equals_true": bool(
                    torch.equal(torch.sort(y.reshape(-1))[0],
                                torch.sort(y_true.reshape(-1))[0])),
            })
        xd, yd = x.to(dev), y.to(dev)
        logits, _ = net(xd)                    # <- the wall projects here
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               yd.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        cur = flat_params_cpu(net)
        cum_disp = float(torch.norm(cur - theta0))
        inc_disp = float(torch.norm(cur - prev))
        prev = cur
        if step in ckpt_set:
            deltas[step] = cur - theta0
        row = {"step": step, "ce_batch": float(loss.item()),
               "cum_disp": cum_disp, "step_disp": inc_disp,
               "d_proj": min(cum_disp, wall_R) if wall_R else None,
               "elapsed_s": round(time.time() - t_start, 1)}
        if step in ckpt_set:
            sd_cpu = {k: v.detach().cpu().clone()
                      for k, v in net.state_dict().items()}
            sds[step] = sd_cpu
            evl.load_state_dict(sd_cpu)
            evl.eval()
            gz = battery_cell(evl, gm12_ids, zid)
            gz0 = battery_cell(evl, g0_ids, zid)
            ce_r = ce_fixed_cpu(evl, *r_eval_xy)
            row.update({"g_m12_mean_pz": gz["mean_pz"],
                        "g0_mean_pz": gz0["mean_pz"],
                        "frac_argmax_z": gz["frac_argmax_z"],
                        "ce_r": ce_r})
            log(f"  [{tag}] CKPT +{step:4d} g-12 {gz['mean_pz']:.4f} "
                f"g0 {gz0['mean_pz']:.4f} CE_R {ce_r:.4f} "
                f"|d| {cum_disp:.4f} (CE {float(loss.item()):.4f})")
        traj.append(row)
        if step % 50 == 0 and step not in ckpt_set:
            log(f"  [{tag}] s{step:4d} CE {float(loss.item()):.4f} "
                f"|d| {cum_disp:.4f} ({row['elapsed_s']:.0f}s)")
        if (time.time() - t_start) > cap:
            log(f"  [{tag}] time cap {cap:.0f}s at s{step}")
            trims.append(f"{tag}: time cap at step {step}")
            break
        if dev.type == "cuda" and step % MIDRUN_POLL_EVERY == 0:
            s = gpu_status()
            if s["mem_total"] > 0 and (s["mem_used"] > 0.85 * s["mem_total"]
                                       or s["temp"] > 80):
                device_events.append({"tag": tag, "step": step,
                                      "event": "MID-RUN MIGRATION", "status": s})
                log(f"  [{tag}] MID-RUN GPU contention at s{step} ({s}) -> CPU")
                if MIDRUN_PAUSE:      # g1bW: pause and wait, never migrate
                    t_p = time.time()
                    midrun_pause_wait(tag)
                    t_start += time.time() - t_p     # paused time is not cap
                    continue
                migrate_to_cpu(net, opt)
                dev = CPU
    net.eval()
    return {"sds": sds, "traj": traj, "steps_ran": step,
            "seed": seed, "lr": lr, "target_mode": target_mode,
            "noise_seed": noise_seed if ngen is not None else None,
            "zeph_violations": zeph_checks, "x_hashes": x_hashes,
            "y_stats": y_stats, "deltas": deltas, "wall_R": wall_R,
            "theta0_norm": float(torch.norm(theta0)),
            "device": str(dev)}


# ------------------------------------------------------------------ checkpoints

CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "g1", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name}")


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("g1_smoke" if SMOKE else "g1")
    log(f"G1 THE ANCHORED BALL (smoke={SMOKE}) -> {rd}")
    set_seed(INSTALL_SEED)

    # ---------------- protocol rebuild (e143/e151/e152/e161/e176/e176n/e185)
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    corpus_zeph = train_text.count("ZEPH")
    G_NAMEFREE = {"corpus_zeph_count": corpus_zeph,
                  "pass": bool(corpus_zeph == 0)}
    assert G_NAMEFREE["pass"], f"corpus contains ZEPH x{corpus_zeph}"

    host_occ = []
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix, "pass": bool(mix == {"FLORIZEL": 19,
                                                         "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"
    log(f"protocol rebuilt: install60 {mix}, held30 (SPLICE_RNG {E43.SPLICE_RNG})")
    name_ids = corpus.encode(NAME)

    # install windows (e043 build_win verbatim) + the original-host anchor bank
    def build_win(p, host):
        return torch.cat([train_ids[p - PRE: p], name_ids,
                          train_ids[p + len(host): p + len(host) + POST_CAP]])

    win_i = torch.stack([build_win(p, h) for p, h in install_occ])
    inst_x = win_i.clone()                                  # (60, 256)
    anchor_full = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                               for p, _ in install_occ])    # (60, 256) originals

    # measurement pool: e152's locked j=54 windows (instrument only)
    wins = []
    for p, h in install_occ:
        pre = train_ids[p - PRE - RETEACH_J: p]
        post = train_ids[p + len(h): p + len(h) + SITE_CONT]
        if len(pre) != PRE + RETEACH_J or len(post) != SITE_CONT:
            raise RuntimeError(f"pool window short at p={p}")
        w = torch.cat([pre, name_ids, post])
        if len(w) != BLOCK:
            raise RuntimeError(f"pool window len {len(w)} != {BLOCK}")
        wins.append(w)
    pool_x = torch.stack(wins)
    G_POOL = {"shape": list(pool_x.shape),
              "name_xcols": [SITE_Z_XCOL, SITE_Z_XCOL + len(NAME) - 1],
              "all_windows_name_in_place": bool(
                  all(torch.equal(w[SITE_Z_XCOL: SITE_Z_XCOL + len(NAME)],
                                  name_ids) for w in pool_x)),
              "note": "MEASUREMENT instrument only (e152's locked j=54 pool); "
                      "NO window from this pool enters any training"}
    G_POOL["pass"] = G_POOL["all_windows_name_in_place"]
    assert G_POOL["pass"], f"pool gate FAILED: {G_POOL}"

    # jittered install pool (e113's construction VERBATIM) for consolidation
    jit_x, jit_mask = {}, {}
    for j in JITTERS:
        jwins = []
        for p, h in install_occ:
            pre = train_ids[p - PRE - j: p]
            post = train_ids[p + len(h): p + len(h) + POST_CAP - j]
            w = torch.cat([pre, name_ids, post])
            if len(w) != BLOCK:
                raise RuntimeError(f"jit window len {len(w)} != {BLOCK} at {j}")
            jwins.append(w)
        jit_x[j] = torch.stack(jwins)
        m = torch.zeros(len(jwins), BLOCK - 1, dtype=torch.bool)
        m[:, PRE - 1 + j: PRE - 1 + j + len(NAME)] = True
        jit_mask[j] = m
    pool_a_x = torch.cat([jit_x[j] for j in JITTERS])          # (300, 256)
    pool_a_mask = torch.cat([jit_mask[j] for j in JITTERS])
    cons_anchor = anchor_full[:16]      # e113: first-16-install original bank

    # =====================================================================
    # THE NEUTRAL STREAM (e170's construction VERBATIM via e176n arm A)
    # =====================================================================
    arng = random.Random(E170_ANCHOR_SEED)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        tries += 1
        txt = train_text[s: s + BLOCK + 1]
        if any(f in txt for f in ANCHOR_FORBIDDEN):
            rejections += 1
            continue
        n_starts.append(s)
    if len(n_starts) != 16:
        raise RuntimeError(f"neutral bank incomplete: {len(n_starts)}/16")
    anchor_neutral = torch.stack([train_ids[s: s + BLOCK] for s in n_starts])

    host_positions = [p for p in E43.find_occ(train_text, HOSTS[0])]
    host_positions += [p for p in E43.find_occ(train_text, HOSTS[1])]
    jc_neutral = sum(1 for s in n_starts
                     if any(s <= p < s + BLOCK + 1 for p in host_positions))
    host_occ_total = len(host_positions)
    bg_rate = host_occ_total * (BLOCK + 1) / len(train_ids)
    G_ANCHOR = {
        "neutral_bank": {
            "construction": ("16 plain corpus windows from train_ids, RNG seed "
                             f"{E170_ANCHOR_SEED}, rejection if [s, s+257) "
                             "contains FLORIZEL/ELIZABETH/ZEPH/MIRABEL — "
                             "e170's construction VERBATIM (= e176n arm A's "
                             "stream; FIXED content)"),
            "n_windows": 16, "block": BLOCK, "seed": E170_ANCHOR_SEED,
            "starts": n_starts, "tries": tries, "rejections": rejections,
            "forbidden": list(ANCHOR_FORBIDDEN),
            "windows_with_host_content": sum(
                1 for s in n_starts
                if any(f in train_text[s: s + BLOCK + 1] for f in HOSTS)),
            "junctions_covered": jc_neutral,
        },
        "budget_identical_to_e176n": bool(anchor_neutral.shape == (16, BLOCK)),
        "rng_note": ("g1_wash verbatim; the seed-10902 aj/rj draw sequence is "
                     "IDENTICAL across ALL SEVEN arms (same shapes/moduli, "
                     "drawn BEFORE any noise draw) — the input stream is "
                     "bit-identical (gated by per-step md5); the deltas are "
                     "the wall (wash arms) and each step's targets (noise)"),
        "random_channel_background": {
            "host_occurrences_in_train": host_occ_total,
            "est_window_hit_rate": bg_rate,
        },
    }
    G_ANCHOR["pass"] = bool(
        G_ANCHOR["neutral_bank"]["windows_with_host_content"] == 0
        and G_ANCHOR["neutral_bank"]["junctions_covered"] == 0
        and G_ANCHOR["budget_identical_to_e176n"])
    assert G_ANCHOR["pass"], f"anchor gate FAILED: {G_ANCHOR}"
    log(f"G_ANCHOR: neutral bank 16x{BLOCK} (seed {E170_ANCHOR_SEED}, "
        f"{rejections} rejections/{tries} tries) — host 0/16, junctions "
        f"0/16: PASS")

    # ---------------- batteries (e119/e176n verbatim)
    bat_ids, held_ids = {}, {}
    for j in GEOS:
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
        hs = [train_text[p - PRE - j: p] for p, _ in held_occ]
        held_ids[j] = torch.stack([corpus.encode(c) for c in hs])
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    gm12_ids = bat_ids[-12]
    g0_ids = bat_ids[0]

    gates_surg: dict = {}

    def measure(sd: dict, tag: str, lean: bool = False) -> dict:
        """e176n's measure() VERBATIM dial: base 3-geos + held30 + CE_R +
        site read + old-band census (row-0 sink / A129 brake) + deletion
        table (D-all, D-183) — on evl_load (settle+disarm; see PIVOT). The
        perturbative probes (census, deletions) read the SETTLED weights:
        evl_load settles the net in place, and sd_local (the settled state
        dict, bit-equal to sd for uncommitted arms) is what the deletion
        table slices — the probe must not mix raw wall-fuzz with the
        deletion."""
        net = evl_load(sd)
        sd_local = {k: v.detach().clone() for k, v in net.state_dict().items()}
        out: dict = {"tag": tag}
        out["base"] = {j: battery_cell(net, bat_ids[j], zid) for j in GEOS}
        out["base_held"] = {j: battery_cell(net, held_ids[j], zid)
                            for j in GEOS}
        out["ce_r"] = ce_fixed_cpu(net, *r_eval_xy)
        log(f"[{tag}] base: " + " ".join(f"g{j:+d} {out['base'][j]['mean_pz']:.4f}"
                                         for j in GEOS)
            + " | held30: " + " ".join(
                f"g{j:+d} {out['base_held'][j]['mean_pz']:.4f}" for j in GEOS)
            + f" | CE_R {out['ce_r']:.4f}")
        out["site_read"] = read_fact_at(net, pool_x, name_ids, zid,
                                        SITE_ADDR_ROW, SITE_Z_XCOL)
        log(f"[{tag}] site read @183: onset "
            f"{out['site_read']['pz_onset_mean']:.4f} span "
            f"{out['site_read']['pname_mean_over7']:.4f}")
        if not lean and not SMOKE:
            out["census_old"] = row_census(net, ROWS_OLD,
                                           lambda n: battery_pz(n, bat_ids[0],
                                                                zid))
            co = out["census_old"]["rows"]
            out["old_band"] = {
                "base_pz": out["census_old"]["base_readout"],
                "row0_strength": co["0"]["strength"],
                "A129": co["129"]["strength"],
                "band121_129_max": max(co[str(r)]["strength"]
                                       for r in range(121, 130)
                                       if str(r) in co)}
            log(f"[{tag}] old band: row0 S "
                f"{out['old_band']['row0_strength']:+.4f} | A(129) "
                f"{out['old_band']['A129']:+.4f}")
            DELS = {"d_all": D_ALL, "d183": (SITE_ADDR_ROW,)}
            out["del_table"] = {}
            for dl, rows_ in DELS.items():
                sd_d, gate = deleted_wpe(sd_local, rows_)
                gates_surg[f"{tag}__{dl}"] = gate
                if not gate["pass"]:
                    raise RuntimeError(f"deletion gate FAILED {tag}/{dl}: "
                                       f"{gate}")
                net.load_state_dict(sd_d)
                cell = {"g0": battery_cell(net, bat_ids[0], zid)["mean_pz"]}
                if dl == "d183":
                    cell["gm12"] = battery_cell(net, bat_ids[-12],
                                                zid)["mean_pz"]
                out["del_table"][dl] = cell
            net.load_state_dict(sd_local)
            log(f"[{tag}] deletions g0: " + " | ".join(
                f"{dl} {out['del_table'][dl]['g0']:.3f}" for dl in DELS))
        else:
            w = net.wpe.weight.data
            orig = w.clone()
            mean_row = orig.mean(0)
            bp = battery_pz(net, bat_ids[0], zid)
            w[129] = mean_row
            m129 = bp - battery_pz(net, bat_ids[0], zid)
            w.copy_(orig)
            w[129] = 0.0
            z129 = bp - battery_pz(net, bat_ids[0], zid)
            w.copy_(orig)
            assert torch.equal(w, orig), "lean A129 failed to restore wpe"
            out["A129_quick"] = float(min(m129, z129))
        del net
        return out

    def flat_cells(m: dict) -> dict:
        c = {"gm12": m["base"][-12]["mean_pz"],
             "g0": m["base"][0]["mean_pz"],
             "gp12": m["base"][12]["mean_pz"],
             "held30_gm12": m["base_held"][-12]["mean_pz"],
             "held30_g0": m["base_held"][0]["mean_pz"],
             "ce_r": m["ce_r"],
             "site_read_onset": m["site_read"]["pz_onset_mean"],
             "site_read_span": m["site_read"]["pname_mean_over7"]}
        if "old_band" in m:
            c["A129"] = m["old_band"]["A129"]
            c["row0_strength"] = m["old_band"]["row0_strength"]
            c["dall_g0"] = m["del_table"]["d_all"]["g0"]
            c["d183_g0"] = m["del_table"]["d183"]["g0"]
            c["d183_gm12"] = m["del_table"]["d183"]["gm12"]
        else:
            c["A129"] = m["A129_quick"]
        return c

    # =====================================================================
    # PHASE 0 — ONE SHARED CHAIN (the control's own history)
    # =====================================================================
    log("=" * 78)
    base = load_g1(CKPT_DIR / BASE_CK)
    assert base.num_params() == G1_PARAMS, \
        f"base param count {base.num_params()} != {G1_PARAMS}"
    base_cells = flat_cells(measure(base.state_dict(), "base_e005s",
                                    lean=SMOKE))
    log(f"base {BASE_CK}: g-12 {base_cells['gm12']:.4f} CE_R "
        f"{base_cells['ce_r']:.4f} ({G1_PARAMS} params)")

    # ---- stage 1: the Dmix install via common.train_model (seed 42) -------
    log(f"PHASE 0a — INSTALL: e043 Dmix stream ({NAME_BS} install + 16 "
        f"paired + {MIX_RANDOM} random = 64 windows/step), dose s"
        f"{INSTALL_STEPS}, house cosine (warmup 100), seed {INSTALL_SEED}")
    if not SMOKE:
        cooldown(COOLDOWN_S)
    dev = pick_dev("install")
    common.DEVICE = str(dev)
    install_net = copy.deepcopy(base).to(dev)
    dmix = DmixCorpus(E43.REPO / "data" / "input.txt", seed=INSTALL_SEED,
                      inst_x=inst_x, anchor_full=anchor_full,
                      train_ids=train_ids)
    cap = TRAIN_CAP_GPU if dev.type == "cuda" else TRAIN_CAP_CPU
    ck_install = CKPT_DIR / ("smoke_g1_install.pt" if SMOKE else "g1_install.pt")
    install_hist = train_model(install_net, dmix, steps=INSTALL_STEPS,
                               lr=1e-3, batch_size=64, max_seconds=cap,
                               eval_every=100 if not SMOKE else 4,
                               ckpt=ck_install)
    sd_install = {k: v.detach().cpu().clone()
                  for k, v in install_net.state_dict().items()}
    inst_cells = flat_cells(measure(sd_install, "post_install", lean=SMOKE))
    log(f"post-install: g-12 {inst_cells['gm12']:.4f} g0 "
        f"{inst_cells['g0']:.4f} CE_R {inst_cells['ce_r']:.4f}")
    del install_net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- stage 2: the e113 jitter consolidation (seed 10901) --------------
    log(f"PHASE 0b — CONSOLIDATE: e113 VERBATIM (jitters {list(JITTERS)}, "
        f"{CONS_STEPS} steps, seed {CONS_SEED}, lr 1e-3 constant)")
    if not SMOKE:
        cooldown(COOLDOWN_S)
    cons = consolidate(evl_load(sd_install), pool_a_x, pool_a_mask,
                       cons_anchor, train_ids, g0_ids, r_eval_xy, zid,
                       "consolidate")
    sd_root = cons["sd"]

    # ---- the root battery + G-ROOT ----------------------------------------
    log("=" * 78)
    root = measure(sd_root, "g1_root", lean=SMOKE)
    root_cells = flat_cells(root)
    G_ROOT0 = {"bar": EXPRESS_BAR, "gm12": root_cells["gm12"],
               "pass": bool(root_cells["gm12"] >= EXPRESS_BAR)}
    log(f"G-ROOT: root g-12 {root_cells['gm12']:.4f} "
        f"(bar >= {EXPRESS_BAR}): {'PASS' if G_ROOT0['pass'] else 'FAIL'}")
    if not G_ROOT0["pass"] and not SMOKE:
        log("G-ROOT FAILED — arms still run for the record; verdict will be "
            "TEXTURE (gate failure), nothing adjudicated")
    save_ckpt("g1_root", sd_root,
              {"desc": f"e005s_small + Dmix install s{INSTALL_STEPS} (seed "
                       f"{INSTALL_SEED}) + e113 jitter consolidation "
                       f"(seed {CONS_SEED}) — the shared phase-0 root "
                       f"(theta0 for every g1 arm)",
               "install_steps": INSTALL_STEPS, "install_seed": INSTALL_SEED,
               "cons_steps": cons["steps_ran"], "cons_seed": CONS_SEED,
               "base": f"runs/checkpoints/{BASE_CK}"})

    theta0 = sd_root

    # =====================================================================
    # THE ARMS (one training each; cooldown before each; ALL inputs
    # bit-identical — the deltas are the wall and the targets)
    # =====================================================================
    ARM_SPECS = ([("C", None, "true", 0, CK_WASH,
                   "CONTROL — uncommitted neutral wash (e176N arm A verbatim "
                   "at 0.84M): the size-matched reference (the clock, "
                   "D_kill, the CE adaptation curve)")]
                  + [(f"W{i+1}", R, "true", 0, CK_WASH,
                      f"WALL R={R} — commit({R}) then the identical neutral "
                      f"wash") for i, R in enumerate(R_LADDER)]
                  + [("N0", None, "iid", NOISE_SEED_A, CK_NOISE,
                      "NOISE CONTROL — uncommitted labels-noise (e185 "
                      "verbatim at 0.84M): the size-replicate gate"),
                     ("N1", 0.7, "iid", NOISE_SEED_A, CK_NOISE,
                      "NOISE UNDER THE WALL — commit(0.7), labels-noise"),
                     ("N2", 0.7, "perm", NOISE_SEED_B, CK_NOISE,
                      "NOISE UNDER THE WALL, arm 2 — commit(0.7), "
                      "shuffled-target")])

    arms: dict = {}
    batteries_all: dict = {}
    G_BITROOT = {}
    for tag, R, mode, nseed, cks, desc in ARM_SPECS:
        log("=" * 78)
        if not SMOKE:
            log(f"[thermal] cooldown {COOLDOWN_S:.0f}s before {tag}")
            cooldown(COOLDOWN_S)
        log(f"ARM {tag} — {desc}")
        net0 = evl_load(theta0) if R is None else CommittedGPT(G1_CFG)
        if R is not None:
            net0.load_state_dict(theta0)
            net0.commit(R)
            # G_BITROOT: the wall root's tensors are bit-identical copies of
            # theta0 (the anchor is a copy; the wall is inert before commit)
            body, _ = split_anchored_sd(net0.state_dict())
            md = max(float((body[k].float() - theta0[k].float()).abs().max())
                     for k in theta0)
            anch_ok = all(torch.equal(net0._anchor(n), p.detach())
                          for n, p in net0.named_parameters())
            G_BITROOT[tag] = {"max_abs_diff": md, "anchors_bit_equal": bool(anch_ok),
                              "n_anchor_tensors": net0._n_anchor_tensors,
                              "pass": bool(md == 0.0 and anch_ok)}
            assert G_BITROOT[tag]["pass"], f"{tag}: wall root != theta0"
            log(f"G_BITROOT[{tag}]: max|diff| {md:.1e}, anchors bit-equal: "
                f"PASS")
        arm = g1_wash(tag, net0, anchor_neutral, train_ids, itos,
                      r_eval_xy, gm12_ids, g0_ids, zid, target_mode=mode,
                      noise_seed=nseed, ckpt_steps=cks)
        G_DRAWFREE = {"zeph_violations": arm["zeph_violations"],
                      "pass": bool(arm["zeph_violations"] == 0)}
        assert G_DRAWFREE["pass"], f"{tag}: name token leaked into a window"
        gates_surg[f"G_DRAWFREE_{tag}"] = G_DRAWFREE
        arms[tag] = arm

        # checkpoints: every noise ckpt + each arm's final
        for s in sorted(arm["sds"]):
            if tag.startswith("N") or s == max(arm["sds"]):
                save_ckpt(f"g1_{tag}_s{s}", arm["sds"][s],
                          {"desc": f"g1 root + {s}-step {mode}-target neutral "
                                   f"{'wash' if mode == 'true' else 'noise'} "
                                   f"(R={R}, input seed {FREEZE_SEED}, lr "
                                   f"{FT_LR})",
                           "steps": int(s), "R": R, "target_mode": mode,
                           "input_seed": FREEZE_SEED, "lr": FT_LR,
                           "noise_seed": nseed or None,
                           "base": "runs/checkpoints/g1_root.pt"})

        # full dials: wash arms at {2, 50, 300}; noise arms at every ckpt
        full_at = [s for s in cks if s in arm["sds"]
                   and (mode == "true" and s in FULL_DIAL_WASH
                        or mode != "true")]
        batteries = {}
        for s in full_at:
            log(f"{tag} +{s} full dial")
            batteries[str(s)] = measure(arm["sds"][s], f"{tag}{s}", lean=SMOKE)
        batteries_all[tag] = batteries

    # =====================================================================
    # CONSTRUCTION-FIDELITY GATES
    # =====================================================================
    G_INPUTS = {"per_step": {}, "pass": None, "note":
                "the seed-10902 aj/rj draw sequence is shared by ALL arms — "
                "per-step input batches are bit-identical (md5-gated) across "
                "wash AND noise arms through +10"}
    noise_tags = [t for t, *_ in ARM_SPECS if t.startswith("N")]
    for step in range(1, CK_NOISE[-1] + 1):
        hs = {t: arms[t]["x_hashes"].get(step) for t in arms}
        same = all(h is not None for h in hs.values()) and \
            len(set(hs.values())) == 1
        G_INPUTS["per_step"][step] = {t: hs[t] for t in arms}
        G_INPUTS["per_step"][step]["identical"] = bool(same)
    G_INPUTS["pass"] = bool(all(v["identical"]
                                for v in G_INPUTS["per_step"].values()))
    assert G_INPUTS["pass"], "input streams diverged across arms"
    log(f"G_INPUTS: per-step inputs bit-identical across all "
        f"{len(arms)} arms through +{CK_NOISE[-1]}: PASS")

    G_TARGETS = {"per_arm": {}}
    for tag in noise_tags:
        ys = arms[tag]["y_stats"]
        G_TARGETS["per_arm"][tag] = {
            "frac_targets_changed_mean": float(
                np.mean([r["frac_targets_changed"] for r in ys])),
            "multiset_equals_true_all_steps": bool(
                all(r["multiset_equals_true"] for r in ys)),
        }
    G_TARGETS["per_arm"]["N0"]["expectation"] = (
        "iid labels change ~64/65 positions; multiset NOT preserved")
    G_TARGETS["per_arm"]["N1"]["expectation"] = (
        "iid labels change ~64/65 positions; multiset NOT preserved")
    G_TARGETS["per_arm"]["N2"]["expectation"] = (
        "permutation changes ~63/65 positions; multiset preserved at EVERY step")
    G_TARGETS["pass"] = bool(
        all(G_TARGETS["per_arm"][t]["frac_targets_changed_mean"] > 0.9
            for t in noise_tags)
        and not G_TARGETS["per_arm"]["N0"]["multiset_equals_true_all_steps"]
        and not G_TARGETS["per_arm"]["N1"]["multiset_equals_true_all_steps"]
        and G_TARGETS["per_arm"]["N2"]["multiset_equals_true_all_steps"])
    assert G_TARGETS["pass"], f"target gate FAILED: {G_TARGETS}"
    log("G_TARGETS: iid arms break the target multiset, perm arm preserves "
        "it at every step: PASS")

    # =====================================================================
    # DISPLACEMENT TABLES + COSINES (e185's currency, measured)
    # =====================================================================
    def disp_table_for(tag):
        rows = []
        for t in arms[tag]["traj"]:
            s = t["step"]
            row = {"step": s, "ce_batch": t["ce_batch"],
                   "cum_disp": t["cum_disp"], "step_disp": t["step_disp"],
                   "d_proj": t["d_proj"]}
            if "g_m12_mean_pz" in t:
                row["g_m12_light"] = t["g_m12_mean_pz"]
                row["ce_r_light"] = t["ce_r"]
            if s in arms[tag]["deltas"] and s in arms["C"]["deltas"]:
                d_a = arms[tag]["deltas"][s]
                d_c = arms["C"]["deltas"][s]
                row["cos_vs_C"] = float(torch.dot(d_a, d_c)
                                        / (torch.norm(d_a) * torch.norm(d_c)
                                           + 1e-30))
            rows.append(row)
        return rows

    disp_table = {tag: disp_table_for(tag) for tag in arms}

    # =====================================================================
    # ADJUDICATION (registered clauses; order GATES -> WALL -> NOISE -> PIN
    # -> COSTS; no shopping)
    # =====================================================================
    def light_gm12(tag):
        return {t["step"]: t["g_m12_mean_pz"] for t in arms[tag]["traj"]
                if "g_m12_mean_pz" in t}

    def full_gm12(tag):
        return {int(s): flat_cells(batteries_all[tag][s])["gm12"]
                for s in batteries_all.get(tag, {})}

    # ---- G-CTRL: arm C kills by +50 (the +50 state)
    c_gm12 = light_gm12("C")
    G_CTRL = {"bar": SHUT_BAR, "gm12_at_50": c_gm12.get(50),
              "earliest_le_bar": next((s for s in CK_WASH if s > 0
                                       and c_gm12.get(s, 1.0) <= SHUT_BAR),
                                      None),
              "pass": bool(c_gm12.get(50, 1.0) <= SHUT_BAR)}
    log(f"G-CTRL: arm C g-12 at +50 = {c_gm12.get(50)} "
        f"(earliest <= {SHUT_BAR}: +{G_CTRL['earliest_le_bar']}): "
        f"{'PASS' if G_CTRL['pass'] else 'FAIL'}")

    # ---- G-PIN: every wall arm's raw displacement <= R + 1.5 at every ckpt
    G_PIN = {"per_arm": {}}
    for tag in ("W1", "W2", "W3", "N1", "N2"):
        R = arms[tag]["wall_R"]
        rows = [r for r in disp_table[tag] if "g_m12_light" in r]
        mx = max(r["cum_disp"] for r in rows) if rows else None
        G_PIN["per_arm"][tag] = {
            "R": R, "bound": R + PIN_FUZZ_BAR,
            "max_raw_disp_at_ckpt": mx,
            "per_ckpt": {r["step"]: r["cum_disp"] for r in rows},
            "pass": bool(mx is not None and mx <= R + PIN_FUZZ_BAR)}
        log(f"G-PIN[{tag}]: max raw |d| at ckpt {mx:.4f} <= "
            f"{R + PIN_FUZZ_BAR:.2f}: "
            f"{'PASS' if G_PIN['per_arm'][tag]['pass'] else 'FAIL'}")
    G_PIN["pass"] = bool(all(v["pass"] for v in G_PIN["per_arm"].values()))

    # ---- G-NOISE: N0 kills by +10 at displacement-match (e185's logic)
    n0_gm12 = light_gm12("N0")
    t_kill = next((s for s in CK_NOISE if c_gm12.get(s, 1.0) <= SHUT_BAR), None)
    ctrl_kills_fine = t_kill is not None
    if ctrl_kills_fine:
        D_kill = next(r["cum_disp"] for r in disp_table["C"]
                      if r["step"] == t_kill)
    else:
        # the control kills only past the noise horizon: D_kill at the +50
        # state (coarse; co-reported) — the e185 abort clause does NOT apply
        # to g1 (G-CTRL is g1's own control gate and has already adjudicated)
        t_kill_eff = next((s for s in CK_WASH
                           if s > 0 and c_gm12.get(s, 1.0) <= SHUT_BAR), None)
        D_kill = next(r["cum_disp"] for r in disp_table["C"]
                      if r["step"] == t_kill_eff) if t_kill_eff else None
    G_NOISE = {"t_kill_C": t_kill, "D_kill": D_kill, "coarse": None,
               "match_step_N0": None, "gm12_at_match": None, "pass": None}
    if D_kill is not None:
        G_NOISE["coarse"] = not ctrl_kills_fine
        M = next((r["step"] for r in disp_table["N0"]
                  if r["cum_disp"] >= D_kill), None)
        G_NOISE["match_step_N0"] = M
        G_NOISE["gm12_at_match"] = n0_gm12.get(M) if M is not None \
            else n0_gm12.get(CK_NOISE[-1])
        G_NOISE["displacement_unmatched"] = M is None
        G_NOISE["pass"] = bool(G_NOISE["gm12_at_match"] is not None
                               and G_NOISE["gm12_at_match"] <= SHUT_BAR)
    log(f"G-NOISE: N0 g-12 at match {G_NOISE['gm12_at_match']} "
        f"(D_kill {D_kill}): "
        f"{'PASS' if G_NOISE['pass'] else 'FAIL'}")

    gates_pass = bool(G_ROOT0["pass"] and G_CTRL["pass"] and G_PIN["pass"]
                      and G_NOISE["pass"])

    # ---- WALL verdicts
    def arm_verdict(tag, cks):
        g = light_gm12(tag)
        vals = [g[s] for s in cks if s in g]
        maintains = bool(vals and all(v >= MAINTAIN_BAR for v in vals))
        dies_by_50 = bool(g.get(50, 1.0) <= SHUT_BAR)
        first_under = next((s for s in cks if g.get(s, 1.0) <= SHUT_BAR), None)
        return {"g_m12": g, "min_gm12": min(vals) if vals else None,
                "maintains": maintains, "dies_by_50": dies_by_50,
                "first_ck_le_bar": first_under}

    wall = {tag: arm_verdict(tag, CK_WASH) for tag in ("C", "W1", "W2", "W3")}
    F1_WALL_DEAF = bool(wall["W1"]["dies_by_50"])
    W3_maintains = wall["W3"]["maintains"]
    WALL_CLIFF_INSIDE = bool(wall["W1"]["maintains"] and
                             wall["W3"]["dies_by_50"])
    if wall["W2"]["maintains"]:
        cliff = (1.4, 4.2)
    elif wall["W2"]["dies_by_50"]:
        cliff = (0.7, 1.4)
    else:
        cliff = None
    # F2: survival must order with R against measured D_kill
    survival_order = [wall[t]["min_gm12"] for t in ("W1", "W2", "W3")]
    orders_with_R = all(survival_order[i] >= survival_order[i + 1] - 1e-12
                        for i in range(2))
    cliff_ok = True
    if cliff is not None and D_kill is not None:
        cliff_ok = bool(cliff[0] <= 2.0 * D_kill and cliff[1] >= 0.5 * D_kill)
    F2_CLIFF_MISPLACED = bool(W3_maintains or not orders_with_R
                              or not cliff_ok)

    # ---- NOISE verdicts (at pinned R)
    noise = {tag: arm_verdict(tag, CK_NOISE) for tag in noise_tags}
    ce_r_root = root_cells["ce_r"]
    for tag in noise_tags:
        g = arms[tag]["traj"]
        noise[tag]["ce_r_at_10"] = next((t["ce_r"] for t in g
                                         if t["step"] == CK_NOISE[-1]), None)
        noise[tag]["ce_r_within_bar"] = bool(
            noise[tag]["ce_r_at_10"] is not None
            and noise[tag]["ce_r_at_10"] <= ce_r_root + CE_NOISE_BAR)
    F3_NOISE_PENETRATES = bool(
        any(light_gm12(t).get(s, 1.0) <= SHUT_BAR for t in ("N1", "N2")
            for s in CK_NOISE))
    NOISE_SPARED = bool(
        all(light_gm12(t).get(s, 0.0) >= MAINTAIN_BAR
            for t in ("N1", "N2") for s in CK_NOISE
            if s in light_gm12(t))
        and noise["N1"]["ce_r_within_bar"] and noise["N2"]["ce_r_within_bar"]
        and G_PIN["per_arm"]["N1"]["pass"] and G_PIN["per_arm"]["N2"]["pass"])
    # the anisotropy fork: N1's floor vs W1's floor over the SAME horizon
    w1_short = [light_gm12("W1").get(s) for s in CK_NOISE
                if s in light_gm12("W1")]
    n1_short = [light_gm12("N1").get(s) for s in CK_NOISE
                if s in light_gm12("N1")]
    aniso = {"W1_floor_h10": min(w1_short) if w1_short else None,
             "N1_floor_h10": min(n1_short) if n1_short else None}
    aniso["dip"] = (aniso["W1_floor_h10"] - aniso["N1_floor_h10"]) \
        if None not in aniso.values() else None
    aniso["direction_thin_for_noise"] = bool(
        aniso["dip"] is not None and aniso["dip"] >= ANISO_BAR)

    # ---- PIN verdict (FLAT-AT-PIN / F4)
    w1g = light_gm12("W1")
    flat_delta = abs(w1g.get(300, float("nan")) - w1g.get(50, float("nan"))) \
        if 50 in w1g and 300 in w1g else None
    FLAT_AT_PIN = bool(flat_delta is not None and flat_delta <= FLAT_BAR)
    seq = [w1g[s] for s in (50, 100, 200, 300) if s in w1g]
    monotone_decline = all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))
    F4_ERODES = bool(len(seq) >= 2 and monotone_decline
                     and (seq[0] - seq[-1]) >= ERODE_BAR)

    # ---- COSTS
    def ce_at(tag, step):
        return next((t["ce_batch"] for t in arms[tag]["traj"]
                     if t["step"] == step), None)

    ce_tax = {"W1_ce300": ce_at("W1", 300), "C_ce300": ce_at("C", 300)}
    ce_tax["delta"] = (ce_tax["W1_ce300"] - ce_tax["C_ce300"]) \
        if None not in ce_tax.values() else None
    WALL_TAXES = bool(ce_tax["delta"] is not None
                      and ce_tax["delta"] >= CE_TAX_BAR)
    WALL_FREE = bool(ce_tax["delta"] is not None
                     and abs(ce_tax["delta"]) < CE_TAX_BAR)

    w1_300 = flat_cells(batteries_all["W1"]["300"]) \
        if "300" in batteries_all.get("W1", {}) else None
    anatomy = None
    if w1_300 is not None:
        band = batteries_all["W1"]["300"]["census_old"]["rows"] \
            if "census_old" in batteries_all["W1"]["300"] else {}
        anatomy = {
            "row0_strength": w1_300.get("row0_strength"),
            "row0_ratio_vs_root": (w1_300.get("row0_strength")
                                   / root_cells.get("row0_strength"))
            if w1_300.get("row0_strength") is not None
            and root_cells.get("row0_strength") else None,
            "row0_ge_half_root": bool(
                w1_300.get("row0_strength") is not None
                and root_cells.get("row0_strength") is not None
                and w1_300["row0_strength"]
                >= 0.5 * root_cells["row0_strength"]),
            "site_read_span": w1_300.get("site_read_span"),
            "span_ge_0p7": bool((w1_300.get("site_read_span") or 0)
                                >= 0.7),
            "band121_129_content": {r: band[str(r)]["content"]
                                    for r in range(121, 130)
                                    if str(r) in band} if band else None,
            "band_all_content": bool(band and all(
                band[str(r)]["content"] for r in range(121, 130)
                if str(r) in band)),
            "held30_gm12": w1_300.get("held30_gm12"),
            "held30_ge_bar": bool((w1_300.get("held30_gm12") or 0)
                                  >= HELD30_BAR),
            "ce_r_at_300": w1_300.get("ce_r"),
        }

    # ---- the composed verdict (order: gates -> F1 -> F2 -> cliff -> texture)
    fmt = lambda v: "n/a" if v is None else f"{v:.4f}"

    if not gates_pass:
        failed = [k for k, g in (("G-ROOT", G_ROOT0), ("G-CTRL", G_CTRL),
                                 ("G-PIN", G_PIN), ("G-NOISE", G_NOISE))
                  if not g["pass"]]
        verdict = f"TEXTURE (GATE FAILURE: {', '.join(failed)})"
        clause = ("a registered gate failed — nothing adjudicated per the "
                  f"spec's abort clause; failed: {failed}; the full record "
                  f"is reported (root g-12 {fmt(root_cells['gm12'])}, "
                  f"C g-12 trace "
                  + " -> ".join(f"+{s}:{fmt(v)}" for s, v in
                                sorted(c_gm12.items())) + ")")
    elif F1_WALL_DEAF:
        verdict = "F1 WALL-DEAF"
        clause = (f"W1 DIED (g-12 {fmt(wall['W1']['g_m12'].get(50))} <= "
                  f"{SHUT_BAR} at +50; first checkpoint under bar: "
                  f"+{wall['W1']['first_ck_le_bar']}) DESPITE G-PIN-verified "
                  f"confinement (max raw |d| "
                  f"{fmt(G_PIN['per_arm']['W1']['max_raw_disp_at_ckpt'])} <= "
                  f"{0.7 + PIN_FUZZ_BAR:.2f}) at R = 0.7 <= half the scaled "
                  f"basin — the readout died WITHOUT displacement: the kill "
                  f"is not displacement-limited; the no-basin law upgrades "
                  f"to a stronger architectural necessity.")
    elif F2_CLIFF_MISPLACED:
        verdict = "F2 WALL-CLIFF-MISPLACED"
        clause = (f"survival does not order with R against measured D_kill "
                  f"{fmt(D_kill)}: W1 min {fmt(survival_order[0])}, W2 min "
                  f"{fmt(survival_order[1])}, W3 min {fmt(survival_order[2])} "
                  f"(W3 maintains: {W3_maintains}; orders with R: "
                  f"{orders_with_R}; cliff interval {cliff} vs D_kill) — "
                  f"the L2-position currency fails to transfer across "
                  f"size/architecture.")
    elif WALL_CLIFF_INSIDE:
        verdict = "WALL-HOLDS (WALL-CLIFF-INSIDE)"
        clause = (f"W1 maintains (min g-12 {fmt(wall['W1']['min_gm12'])} >= "
                  f"{MAINTAIN_BAR} at every checkpoint; +300 "
                  f"{fmt(wall['W1']['g_m12'].get(300))}) while W3 dies by "
                  f"+50 (g-12 {fmt(wall['W3']['g_m12'].get(50))}; first "
                  f"under +{wall['W3']['first_ck_le_bar']}) — the survival "
                  f"cliff lives inside (0.7, 4.2]; W2 "
                  + ("MAINTAINS" if wall["W2"]["maintains"] else "dies")
                  + f" => cliff in {cliff}; the control's measured D_kill = "
                  f"{fmt(D_kill)} L2 (the scaled prior was "
                  f"{SCALED_BASIN}) — the wall re-measures the basin width "
                  f"dynamically against e180's t* extrapolation.")
    else:
        verdict = "TEXTURE"
        clause = (f"no wall clause fired cleanly: W1 min "
                  f"{fmt(wall['W1']['min_gm12'])}, W2 min "
                  f"{fmt(wall['W2']['min_gm12'])}, W3 min "
                  f"{fmt(wall['W3']['min_gm12'])}, C +50 "
                  f"{fmt(c_gm12.get(50))} — full trajectories reported, no "
                  f"bar shopping.")

    log("=" * 78)
    log(f"G1 VERDICT: {verdict}")
    for tag in ("C", "W1", "W2", "W3"):
        g = wall[tag]["g_m12"]
        d = disp_table[tag]
        log(f"  {tag}: g-12 " + " -> ".join(f"+{s}:{v:.4f}"
                                            for s, v in sorted(g.items())))
        log(f"  {tag}: |d| " + " -> ".join(
            f"+{r['step']}:{r['cum_disp']:.3f}" for r in d if "g_m12_light" in r))
    for tag in noise_tags:
        g = light_gm12(tag)
        log(f"  {tag}: g-12 " + " -> ".join(f"+{s}:{v:.4f}"
                                            for s, v in sorted(g.items()))
            + f" | CE_R@10 {noise[tag]['ce_r_at_10']:.4f} "
              f"(root {ce_r_root:.4f})")
    log(f"  FLAT-AT-PIN: {FLAT_AT_PIN} (|delta(+300,+50)| = {flat_delta})"
        f" | F4 ERODES: {F4_ERODES}")
    log(f"  NOISE-SPARED-BY-WALL: {NOISE_SPARED} | F3 NOISE-PENETRATES: "
        f"{F3_NOISE_PENETRATES} | aniso dip {aniso['dip']}")
    log(f"  WALL-TAXES-ADAPTATION: {WALL_TAXES} (dCE@300 "
        f"{ce_tax['delta']})")
    log(f"  {clause}")
    log("=" * 78)

    # =====================================================================
    # OUTPUTS
    # =====================================================================
    trace = {}
    for tag in arms:
        rows = [{"freeze_steps": 0, **{k: root_cells[k] for k in
                 ("gm12", "g0", "gp12", "held30_gm12", "held30_g0", "ce_r",
                  "site_read_onset", "site_read_span")}}]
        g0_light = {t["step"]: t["g0_mean_pz"] for t in arms[tag]["traj"]
                    if "g0_mean_pz" in t}
        for s in sorted(arms[tag]["sds"]):
            c = None
            if str(s) in batteries_all.get(tag, {}):
                c = flat_cells(batteries_all[tag][str(s)])
            lg = light_gm12(tag).get(s)
            if c is not None:
                rows.append({"freeze_steps": s, **c})
            elif lg is not None:
                rows.append({"freeze_steps": s, "gm12": lg,
                             "g0": g0_light.get(s),
                             "ce_r": next((t["ce_r"] for t in arms[tag]["traj"]
                                           if t["step"] == s), None)})
        trace[tag] = rows

    metrics = {
        "experiment": "g1_anchored_ball",
        "date": common.now_iso(),
        "design": "scratch/g1_design.md (committed; bars frozen at dispatch)",
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("is the no-basin property an ARCHITECTURAL NECESSITY "
                     "of the pre-LN transformer, or does one minimal "
                     "architectural change (a committed snapshot + a hard "
                     "L2 wall at radius R, flat interior, all directions, "
                     "forward semantics) install a well? The R ladder "
                     "{0.7, 1.4, 4.2} re-measures the basin width "
                     "dynamically against e180's t* extrapolation."),
        "size_correction": {
            "organism": f"Cfg 4L/4H/128d block 256 = {G1_PARAMS} params "
                        f"(the e033/e040 0.84M lineage; base "
                        f"runs/checkpoints/{BASE_CK})",
            "prior_274M": PRIORS_274M,
            "scaled_basin_estimate": list(SCALED_BASIN),
            "R_hat": R_HAT,
            "note": "the stored wash numbers are 2.74M-organism PRIORS, not "
                    "bars; every g1 verdict is adjudicated against the "
                    "IN-RUN size-matched control arm C",
        },
        "phase0": {
            "base_cells": base_cells,
            "install": {"steps": INSTALL_STEPS, "seed": INSTALL_SEED,
                        "stream": "e043 Dmix via common.train_model "
                                  "(DmixCorpus; the literal-loop note)",
                        "history": install_hist,
                        "post_install_cells": inst_cells},
            "consolidation": {"steps": cons["steps_ran"],
                              "seed": cons["seed"], "device": cons["device"],
                              "traj": cons["traj"]},
            "root_cells": root_cells,
        },
        "arms": {
            tag: {
                "desc": desc, "R": R, "target_mode": mode,
                "noise_seed": nseed or None,
                "ckpt_steps": list(cks),
                "steps_ran": arms[tag]["steps_ran"],
                "device": arms[tag]["device"],
                "traj": [{k: v for k, v in t.items() if k != "d_proj"}
                         for t in arms[tag]["traj"]],
                "y_stats": arms[tag]["y_stats"],
                "missing_checkpoints": [s for s in cks
                                        if s not in arms[tag]["sds"]],
            } for (tag, R, mode, nseed, cks, desc) in ARM_SPECS
        },
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
                     "neutral_bank": G_ANCHOR["neutral_bank"],
                     "measure_dial": "e176n's measure() (the e131 dial set) "
                                     "on evl_load (settle+disarm PIVOT)"},
        "displacement": {
            "currency": ("cumulative ||theta_t - theta_0||_2 over all "
                         f"{G1_PARAMS} trainable parameters (fp32, CPU, "
                         "measured per step) + per-step increments + the "
                         "projected displacement min(d, R); cosines vs arm "
                         "C at shared checkpoints"),
            "table": disp_table,
            "theta0_norm": {t: arms[t]["theta0_norm"] for t in arms},
            "wall_fuzz_registered": "one AdamW step ~ lr*sqrt(P) = 0.92 L2 "
                                    "at lr 1e-3, 0.84M params",
        },
        "gates": {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                  "G_POOL": G_POOL, "G_ANCHOR": G_ANCHOR,
                  "G_ROOT": G_ROOT0, "G_CTRL": G_CTRL, "G_PIN": G_PIN,
                  "G_NOISE": G_NOISE, "G_BITROOT": G_BITROOT,
                  "G_INPUTS": {k: v for k, v in G_INPUTS.items()
                               if k != "per_step"} | {"per_step": {
                                   s: v["identical"] for s, v in
                                   G_INPUTS["per_step"].items()}},
                  "G_TARGETS": G_TARGETS, "G_SURG": gates_surg},
        "traces": trace,
        "batteries": batteries_all,
        "adjudication": {
            "order": "GATES -> WALL -> NOISE -> PIN -> COSTS (frozen)",
            "gates_pass": gates_pass,
            "wall": wall, "noise": noise,
            "F1_WALL_DEAF": F1_WALL_DEAF,
            "F2_WALL_CLIFF_MISPLACED": F2_CLIFF_MISPLACED,
            "WALL_CLIFF_INSIDE": WALL_CLIFF_INSIDE,
            "cliff_interval": cliff,
            "survival_order_with_R": orders_with_R,
            "F3_NOISE_PENETRATES": F3_NOISE_PENETRATES,
            "NOISE_SPARED_BY_WALL": NOISE_SPARED,
            "anisotropy_fork": aniso,
            "FLAT_AT_PIN": FLAT_AT_PIN, "flat_delta": flat_delta,
            "F4_ERODES_AT_PIN": F4_ERODES,
            "WALL_TAXES_ADAPTATION": WALL_TAXES,
            "WALL_FREE": WALL_FREE, "ce_tax": ce_tax,
            "W1_anatomy_at_300": anatomy,
            "verdict": verdict, "clause": clause,
        },
        "honesty_reflex": {
            "projection_implementation": ("the wall is enforced IN THE "
                "FORWARD (hard in-place L2 projection under no_grad when "
                "||theta - theta_anchor|| > R); between forwards one AdamW "
                "step may sit outside (the registered fuzz ~0.92 L2) — "
                "G-PIN gates the raw displacement <= R + 1.5 at every "
                "checkpoint and the displacement table co-reports raw and "
                "projected. The g1 model's measured function is the "
                "PROJECTED weights: light evals run on an armed twin; full "
                "dials settle then disarm (a settled net never projects "
                "again — exact for non-perturbative dials; the disarm "
                "exists so the census/deletion probes measure the probe, "
                "not a projection of the probe)."),
            "size_matched_control": ("every verdict is adjudicated against "
                "arm C — the same class (uncommitted CommittedGPT = "
                "bit-identical TinyGPT), same phase-0 chain, same seed, "
                "same instruments, run in the same process on the same "
                "device; the 2.74M stored numbers (root 0.9156, +2 0.0271, "
                "e185 D_kill 2.489) are priors only. The control supplies "
                "t_kill/D_kill for the noise displacement-match and the CE "
                "adaptation baseline for the wall-tax bar."),
            "single_seed": ("one trajectory per arm (install 42 / "
                "consolidation 10901 / wash draws 10902 / noise 18501-2), "
                "n=1 per cell — the arc's honesty convention; the "
                "replication debt is g1b's first cell if WALL-HOLDS. "
                "e152R showed seed-to-seed timing can span an order of "
                "magnitude on related washes."),
            "wall_blind_spot": ("the wall never protects the FIRST step: "
                "commit happens at d=0, so step 1's AdamW update (the full "
                "~0.92 L2) always lands before the first projection; the "
                "wall caps CUMULATIVE displacement at R + fuzz, it does "
                "not shrink any single step — this is the design's own "
                "registered semantics, not a bug."),
            "bars_anchored": ("dies/maintain reuse the arc's absolute "
                "home-battery bars (0.27 / 0.50) on the e131 dial set's "
                "ruler; the R grid, seeds, checkpoints and adjudication "
                "order were frozen in scratch/g1_design.md before this "
                "implementation; no bar shopping."),
        },
        "trims": trims, "deviations": deviations,
        "device_events": device_events,
        "ckpt_inventory": CKPT_INVENTORY,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 4, "n_head": 4, "n_embd": 128,
                   "block_size": 256, "params": G1_PARAMS,
                   "trainable_params": G1_PARAMS,
                   "anchor_state_bytes": G1_PARAMS * 4,
                   "R_ladder": list(R_LADDER),
                   "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "anchored_ball.png", trace, disp_table, wall, noise, c_gm12,
         D_kill, verdict, clause, ce_tax, ce_r_root, root_cells,
         wall_taxes=WALL_TAXES, flat=(FLAT_AT_PIN, flat_delta),
         f1=F1_WALL_DEAF, f3=F3_NOISE_PENETRATES, f4=F4_ERODES,
         noise_spared=(NOISE_SPARED, aniso.get("dip")), cliff=cliff)

    log(f"outputs: {rd / 'metrics.json'}, {rd / 'anchored_ball.png'}, "
        f"{len(CKPT_INVENTORY)} checkpoints in runs/checkpoints/g1_*.pt")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot

def plot(path, trace, disp_table, wall, noise, c_gm12, D_kill, verdict,
         clause, ce_tax, ce_r_root, root_cells, *, wall_taxes=False,
         flat=(False, None), f1=False, f3=False, f4=False,
         noise_spared=(False, None), cliff=None):
    """THE R-DIAL figure: fact survival vs wash steps, survival + CE-tax vs
    R (the ladder), the cost panel, the displacement trajectories vs the
    walls, the noise panel, and the verdict."""
    import textwrap
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.5))
    cols = {"C": "crimson", "W1": "seagreen", "W2": "royalblue",
            "W3": "darkorange", "N0": "crimson", "N1": "seagreen",
            "N2": "mediumseagreen"}
    lbls = {"C": "C — control (no wall)", "W1": "W1 — commit(0.7)",
            "W2": "W2 — commit(1.4)", "W3": "W3 — commit(4.2)",
            "N0": "N0 — labels-noise, no wall",
            "N1": "N1 — labels-noise, R=0.7",
            "N2": "N2 — shuffled-target, R=0.7"}
    marks = {"C": "o", "W1": "s", "W2": "^", "W3": "v",
             "N0": "o", "N1": "s", "N2": "^"}

    def gm12_series(tag):
        pts = [(r["freeze_steps"], r["gm12"]) for r in trace[tag]]
        return [p[0] for p in pts], [p[1] for p in pts]

    # (0,0) THE HEADLINE: g-12 vs wash steps
    ax = axes[0, 0]
    for tag in ("C", "W1", "W2", "W3"):
        xs, ys = gm12_series(tag)
        ax.plot(xs, ys, marks[tag] + "-", ms=8, lw=2.2, color=cols[tag],
                alpha=0.9, label=lbls[tag])
    for yv, col in ((MAINTAIN_BAR, "seagreen"), (SHUT_BAR, "tab:purple")):
        ax.axhline(yv, ls="--", lw=1.1, color=col, alpha=0.8)
    ax.axhline(EXPRESS_BAR, ls=":", lw=1.0, color="gray", alpha=0.7)
    ax.annotate(f"root {trace['C'][0]['gm12']:.3f}", (0, trace["C"][0]["gm12"]),
                textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    ax.set_xlabel("neutral-wash steps from the committed root")
    ax.set_ylabel("g-12 (absolute mean p(Z), install-60 battery)")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7.5, loc="center right")
    ax.set_title("THE ANCHORED BALL vs the wash — fact survival", fontsize=10)

    # (0,1) THE R LADDER: survival vs R at the three horizons
    ax = axes[0, 1]
    Rs = [0.0] + list(R_LADDER)
    tags_by_R = ["C", "W1", "W2", "W3"]
    for step, col, mk in ((2, "gray", "o"), (50, "tab:purple", "s"),
                          (300, "seagreen", "D")):
        ys = []
        for tag in tags_by_R:
            g = wall[tag]["g_m12"]
            ys.append(g.get(step))
        ax.plot(Rs, ys, mk + "-", ms=9, lw=2.0, color=col,
                label=f"g-12 at +{step}")
    ax.axvspan(SCALED_BASIN[0], SCALED_BASIN[1], color="gold", alpha=0.18,
               label=f"scaled basin prior {SCALED_BASIN}")
    if D_kill is not None:
        ax.axvline(D_kill, ls=":", lw=1.8, color="k", alpha=0.8,
                   label=f"measured D_kill {D_kill:.3f}")
    for yv, col in ((MAINTAIN_BAR, "seagreen"), (SHUT_BAR, "tab:purple")):
        ax.axhline(yv, ls="--", lw=1.0, color=col, alpha=0.8)
    ax.set_xticks(Rs)
    ax.set_xticklabels(["C\n(no wall)"] + [f"R={r}" for r in R_LADDER])
    ax.set_xlabel("the wall dial R (L2 over all 840,704 params)")
    ax.set_ylabel("g-12")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7.5, loc="center right")
    ax.set_title("THE R-DIAL: fact survival vs the wall radius", fontsize=10)

    # (0,2) THE COST PANEL: adaptation CE (in-batch corpus CE + CE_R)
    ax = axes[0, 2]
    for tag in ("C", "W1", "W2", "W3"):
        rows = [r for r in disp_table[tag]]
        xs = [r["step"] for r in rows]
        ys = [r["ce_batch"] for r in rows]
        ax.plot(xs, ys, "-", lw=1.6, color=cols[tag], alpha=0.8,
                label=f"{tag} in-batch CE")
    if ce_tax["delta"] is not None:
        ax.annotate(f"WALL-TAX dCE@300 {ce_tax['delta']:+.3f}",
                    (0.03, 0.05), xycoords="axes fraction", fontsize=8,
                    weight="bold",
                    color="darkred" if wall_taxes else "seagreen")
    ax.set_xlabel("wash step")
    ax.set_ylabel("in-batch corpus CE (the wash's own adaptation)")
    ax.legend(fontsize=7.5)
    ax.set_title("WALL-TAXES-ADAPTATION — the stream's learning vs the wall",
                 fontsize=9.5)

    # (1,0) DISPLACEMENT trajectories vs the walls
    ax = axes[1, 0]
    for tag in ("C", "W1", "W2", "W3"):
        rows = [r for r in disp_table[tag] if "g_m12_light" in r]
        ax.plot([r["step"] for r in rows], [r["cum_disp"] for r in rows],
                marks[tag] + "-", ms=6, lw=1.8, color=cols[tag],
                alpha=0.9, label=lbls[tag])
    for R, col in zip(R_LADDER, ("seagreen", "royalblue", "darkorange")):
        ax.axhline(R, ls="--", lw=1.0, color=col, alpha=0.6)
        ax.axhline(R + PIN_FUZZ_BAR, ls=":", lw=0.8, color=col, alpha=0.5)
        ax.annotate(f"R={R} (+pin {R + PIN_FUZZ_BAR})", (0.99, R),
                    xycoords=("axes fraction", "data"), ha="right",
                    fontsize=7, color=col)
    if D_kill is not None:
        ax.axhline(D_kill, ls=":", lw=1.8, color="k", alpha=0.8)
        ax.annotate(f"D_kill {D_kill:.3f}", (0.01, D_kill),
                    xycoords=("axes fraction", "data"), fontsize=7.5)
    ax.set_xlabel("wash step")
    ax.set_ylabel(r"raw $\|\theta_t-\theta_0\|_2$ at checkpoints")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("DISPLACEMENT TRAJECTORIES vs the walls (G-PIN)", fontsize=10)

    # (1,1) THE NOISE PANEL
    ax = axes[1, 1]
    for tag in ("N0", "N1", "N2"):
        xs, ys = gm12_series(tag)
        ax.plot(xs, ys, marks[tag] + "-", ms=9, lw=2.2, color=cols[tag],
                alpha=0.9, label=lbls[tag])
    w1x, w1y = gm12_series("W1")
    w1f = [(x, y) for x, y in zip(w1x, w1y) if x <= 10]
    if w1f:
        ax.plot([p[0] for p in w1f], [p[1] for p in w1f], "kx--", ms=8,
                lw=1.2, alpha=0.6, label="W1 (corpus, same R) — aniso ref")
    for yv, col in ((MAINTAIN_BAR, "seagreen"), (SHUT_BAR, "tab:purple")):
        ax.axhline(yv, ls="--", lw=1.0, color=col, alpha=0.8)
    for tag in ("N1", "N2"):
        c10 = noise[tag].get("ce_r_at_10")
        if c10 is not None:
            ax.annotate(f"{tag} CE_R@10 {c10:.2f} "
                        f"(root {ce_r_root:.2f}+0.3)",
                        (0.03, 0.12 if tag == "N1" else 0.05),
                        xycoords="axes fraction", fontsize=7,
                        color=cols[tag])
    ax.set_xlabel("noise-wash steps")
    ax.set_ylabel("g-12")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7.5, loc="center right")
    ax.set_title("THE NOISE KILL under the wall (e185's battery, 0.84M)",
                 fontsize=10)

    # (1,2) THE VERDICT PANEL
    ax = axes[1, 2]
    ax.axis("off")
    y = 0.97
    ax.text(0.02, y, "G1 — THE ANCHORED BALL", fontsize=11, va="top",
            family="monospace", weight="bold")
    y -= 0.052
    gm = {t: wall[t]["g_m12"] for t in ("C", "W1", "W2", "W3")}
    for tag in ("C", "W1", "W2", "W3"):
        seq = " -> ".join(f"+{s}:{v:.4f}" for s, v in sorted(gm[tag].items()))
        ax.text(0.02, y, f"  {tag}: {seq}", fontsize=6.8, va="top",
                family="monospace", color=cols[tag])
        y -= 0.030
    y -= 0.008
    ax.text(0.02, y, f"  FLAT-AT-PIN {flat[0]} (|d| {flat[1]}) | "
            f"F1 {f1} | F4 {f4}", fontsize=7.2, va="top",
            family="monospace")
    y -= 0.034
    ax.text(0.02, y, f"  NOISE-SPARED {noise_spared[0]} | F3 {f3} | aniso "
            f"dip {noise_spared[1]}", fontsize=7.2, va="top",
            family="monospace")
    y -= 0.034
    ax.text(0.02, y, f"  D_kill {D_kill if D_kill is None else round(D_kill, 3)}"
            f" | cliff {cliff} | WALL-TAX "
            f"{ce_tax['delta'] if ce_tax['delta'] is None else round(ce_tax['delta'], 3)}",
            fontsize=7.2, va="top", family="monospace")
    y -= 0.048
    ax.text(0.02, y, f"VERDICT: {verdict}", fontsize=9.5, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.042
    for wd in textwrap.wrap(clause, width=88, break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=6.6, va="top", family="monospace")
        y -= 0.026

    fig.suptitle("G1 — THE ANCHORED BALL: commit(R) + a hard L2 wall — "
                 f"does confinement install a well? -> {verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
