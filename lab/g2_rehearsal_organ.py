"""G2 — THE REHEARSAL ORGAN (the generative turn's first built architecture;
zero-new-parameter wrapper on the e098/e157 family-2 line).

Design spec: scratch/g2_design.md (COMMITTED; bars registered BEFORE
implementation — the g-series discipline). This docstring carries the spec's
registered prediction + falsifiers VERBATIM; adjudicate against exactly this.

THE ONE-SENTENCE DESIGN (spec): the base net keeps its wash law untouched, and
g2 adds a zero-new-parameter organ that (a) indexes what it was taught, (b)
watches its own read of the index, and (c) when that read decays, re-teaches
itself the index with error-isolated, position-jittered self-replay.

THE ORGAN (0 new learnable parameters; G2Net wraps a byte-identical TinyGPT
body so every instrument runs unchanged):
  1. CUE POOL — registered_buffer, write-once at install end: e113's jitter
     pool VERBATIM (300 windows = 60 install hosts x offsets {-8,-4,0,+4,+8},
     ZEPHYRA spliced at home, 256 tokens) + its name-position target mask
     (300x255 bool). ~165 KB in the state_dict. Never updated afterward.
  2. READ MONITOR — no-grad, no params: every CADENCE=4 wash steps (never
     during REFRACTORY) forward K_MON=8 cue windows (offsets cycling
     j in {0,-4,+4}) and read mean p(true name char) over the 7x8 name
     positions — the net's own confidence in its index.
  3. REHEARSAL GATE — a comparator: opens when monitor < THETA_OPEN = 0.5.
     One event per opening; after an event, REFRACTORY = 24 steps of lockout.
  4. ERROR REPLAY — the event REPLACES that step's wash batch with e174 arm
     B's replay batch VERBATIM: 16 windows drawn fresh from the 300-pool +
     16 anchors (8 neutral-bank + 8 random corpus, full CE), union CE with the
     7-name-char mask. Batch-size parity (32 = the wash batch's 32).

CELLS (each: 300 optimizer steps, lr 1e-3 constant, AdamW(0.9,0.95) wd 0.1,
clip 1.0, batch 32, seed 10902, checkpoints {1,2,4,10,25,50,100,200,300},
light evals per checkpoint + FULL dial at +50 and +300 — e179's grid):
  CELL-BASE — gate hard-disabled (the ablation identity: with the gate
    disabled, g2 IS the base). Must reproduce family 2's stored wash (e157:
    dead at +1). This cell is a GATE: if the base does NOT die by +50 there is
    no contrast (STRUCTURAL VOID — record a size/family bound, stop).
  CELL-G2   — the key contrast: gate live, all three locks on.
  CELL-ECHO — the e121 ghost: gate live, event = full-token CE on 16 verbatim
    j=0 cue windows + the same 16 anchors (NO mask, NO jitter; lock 1 only).
  CELL-SCHED — organ-less base + EXTERNAL r=1/32 (e179's finetune_rate
    untouched at k=32) — the schedule anchor at this family/seed.

ROOT: the e157 s4305 root FAILED the spec's ROOT-STRENGTH gate (its ruler
g+12 = 0.591 < 0.7) — re-installed per the spec's own clause ("another e098
seed / more exposure steps"): fresh e043-Dmix install (more exposure steps)
+ e113 jitter consolidation (e157's ported recipe, seed 10901) on an e098
base. RULER (frozen rule, no shopping): among {-12, 0, +12} measured on the
chosen root, the ruler is the geo with maximum root mean_pz; die bar 0.27,
maintain bar 0.5. All three geos + held30 co-reported everywhere.

================================= REGISTERED PREDICTION (spec section 6, VERBATIM)
Primary (CELL-G2 vs CELL-BASE, the key contrast): under the same
rehearsal-free neutral wash that kills the base by +50 (CELL-BASE ruler <=
0.27 -- the contrast gate), g2 MAINTAINS: ruler >= 0.5 at BOTH horizons +50
AND +300 (e179's state-based maintain convention verbatim). The trajectory
is a self-triggered SAWTOOTH whose dips stay bounded and whose every event is
followed within <= 18-25 steps by a ruler >= 0.5 (the e179 resurrection
signature, now self-detected). MID-CYCLE DIPS BELOW 0.27 DO NOT UN-MAINTAIN
(e179's own smoke-corrected convention; the checkpoints sit at unknowable
cycle phases -- the late-grid mean/min over {100,200,300} is co-reported for
phase honesty).

Secondary outcomes, each registered:
1. ECONOMY: 5 <= n_events <= 25 (realized r in [1/60, 1/12]); event spacing
   concentrates in 20-45 steps (refractory-bounded rhythm). If it maintains
   at n_events <= 4, the gate is cheaper than the schedule -- a STRONGER
   economy than e179's (texture, recorded).
2. ANATOMY INTACT at +300 (full dial): every dial >= 50% of ITS OWN root
   value (self-referential bar -- no cross-family assumption): held30, site
   read onset/span, row-0 sink, A129 brake, and D-all deletion survival
   (D-all g0 >= 0.5 x root's) -- g2 must maintain the GENERALIZING readout,
   not an address echo.
3. ORGANISM HEALTH: CE_R at +300 <= root + 0.10; no sustained concussion.
4. THE GHOST ERRODES (CELL-ECHO): late-grid sustained level (mean ruler over
   {100,200,300}) < 50% of CELL-G2's, declining across the three checkpoints,
   ending < 0.4 at +300 despite the gate firing -- e121's erosion signature
   inside the architecture. If ECHO maintains too, locks 2+3 are NOT the
   active ingredients and e121's verdict must be re-localized (recorded
   either way).
5. SCHEDULE PARITY (CELL-SCHED): maintains at r=1/32 on this family —
   expected; if it FAILS here, the family-2 lineage is schedule-fragile and
   every g2 verdict gets that bound.
6. MONITOR-RULER COUPLING: monitor and ruler track at every checkpoint (both
   normalized to root). A divergence — monitor high, ruler low — is the
   CUE-OVERFIT texture; caught independently by the D-all/held30 bars.

The law-level stake: if the primary fires, the resurrection economy is
ARCHITECTURALLY SUFFICIENT — the rhythm was the missing organ, and e121's
failure is localized to echo, not to self-generation; the law moves toward
necessity. If it fails, the economy was SCHEDULE-CONTINGENT — the
dataloader's phase carried information the net cannot self-supply.

================================= FALSIFIERS (spec section 7, VERBATIM)
With the contrast gate passed (base dies by +50) and the gate demonstrably
firing (n_events >= 3, each event's batch verified by G_REPLAY):
- F1 — SCHEDULE-BOUND (the main falsifier): CELL-G2 ruler < 0.5 at +50 OR
  +300. Internal, gate-triggered, error-carrying replay does NOT maintain
  where the external schedule did.
- F2 — EROSION (the e121 ghost): maintains at +50 but the late-grid
  {100,200,300} mean < 0.5 AND monotonically declining — the events are echo
  despite the mask; internal replay erodes like dreams did.
- F3 — GATE-NEVER-CLOSES: every post-refractory monitor check fires
  (n_events at the refractory ceiling ~12-13) AND maintain still fails —
  re-entry never sticks on this family; the rhythm cannot bootstrap.
- F4 — ORGANISM PRICE: CE_R > root + 0.25 at +300 (sustained) — maintenance
  bought at the corpus's price is a fail.
- F5 — ANATOMY FAIL (secondary): ruler maintains but D-all deletion at +300
  kills (D-all g0 < 0.27 vs root's survival) — g2 maintained an address
  echo, not the fact.
- STRUCTURAL VOID: CELL-BASE does not die by +50 -> no contrast; record a
  size/family bound on the wash law itself (informative, not a verdict).
Any of F1-F3 kills the design claim; F4/F5 bound it. No bar shopping; every
sub-boolean reported regardless.

INSTRUMENT PROVENANCE (spec section 9): copied VERBATIM from
lab/e179_rehearsal_lab (= e180 = e176n/e161/e152/e151/e143/e131/e119/e113/
e068/e043): load/evl_load/battery_cell/battery_pz/ce_fixed_cpu/val_windows/
deleted_wpe/read_fact_at/row_census/measure-dial/flat_cells/gate_vs, wash_loss,
replay_loss, the neutral-bank builder (G_ANCHOR), the jitter pool builder
(G_JIT); pick_dev/migrate_to_cpu are e184's via e179; the root builders are
e098's E43.exposure call (e043-Dmix install) + e157's finetune_replay (the
ported e113 consolidation). Copied, not imported, to own the device policy.

COMPUTE ENVELOPE (dispatch): GPU allowed (gpu_status()/gpu_ok() gate,
park-once + mid-run guard every 25 steps; e182 may hold the GPU — park to
CPU, the 0.84M net runs fine there); torch threads 8; cooldown(60 s; the
dispatch's 60-120 s band) before each training; per-training cap 1800 s;
no concurrent GPU. Checkpoints runs/checkpoints/g2_*.pt.

Outputs: runs/g2/{metrics.json, rehearsal_organ.png}. No NOTES/THINKING/
QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python g2_rehearsal_organ.py    (G2_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
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

import torch.nn as nn                                 # noqa: E402
import torch.nn.functional as F                       # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT, cooldown, gpu_ok,     # noqa: E402
                    gpu_status, run_dir, save_json)
import e043_install as E43                             # noqa: E402 (REPO, find_occ,
                                                      # SPLICE_RNG, jsonable,
                                                      # exposure, eval_seq)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import matplotlib.gridspec as gridspec                # noqa: E402

SMOKE = os.environ.get("G2_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256                       # training-window block (e098/e157 convention)
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E157_METRICS = E43.REPO / "runs" / "e157" / "metrics.json"
G2_DESIGN = E43.REPO / "scratch" / "g2_design.md"

# ---- the family-2 net (e098 s4305 line: 4L/4H/128d; e157's ACTUAL cfg) --------
# NOTE (recorded deviation): the on-disk family-2 line is block_size=512
# (873,472 params; e157's F2_CFG) — the design doc's "block=256 = 0.838M"
# misstates the line's wpe. All training/anchor/random windows remain
# 256-token windows (e157's convention); <=1M holds with headroom.
F2_CFG = Cfg(vocab=65, n_layer=4, n_head=4, n_embd=128, block_size=512)
F2_PARAMS = 873_472

# ---- placement constants (e152/e157/e179 verbatim) ----------------------------
RETEACH_J = 54                    # measurement pool: name x-cols 184..190
SITE_ADDR_ROW = 183               # the onset read row
SITE_Z_XCOL = PRE + RETEACH_J     # 184
SITE_CONT = BLOCK - PRE - RETEACH_J - len(NAME)     # 65
D_ALL = (121, 125, 129, 133, 137)                   # e113's fixed set
GEOS = (-12, 0, 12)               # the ruler's three battery geometries
JITTERS = (-8, -4, 0, 4, 8)       # e109/e113's registered jitter set
J0_BLOCK = 2                      # pool layout: JITTERS order -> j=0 is block 2
                                  # (windows 120..179)

ROWS_OLD = (0, 1, 2, 3, 4, 5, 6, 60, 100, 118, 119, 120) + tuple(range(121, 130))
if SMOKE:
    ROWS_OLD = (0, 1, 60, 121, 125, 129)

# ---- THE ORGAN's constants (spec section 4, frozen) ---------------------------
CADENCE = 4                       # monitor check every 4 wash steps
K_MON = 8                         # cue windows per check
MON_OFFSETS = (0, -4, 4)          # offsets cycling (spec: j in {0,-4,+4})
THETA_OPEN = 0.5                  # the maintain-bar constant (e176n/e179)
REFRACTORY = 24                   # >= e179's 18-step stickiness + margin

# ---- cells (spec section 5) ----------------------------------------------------
CK_MAIN: tuple[int, ...] = (1, 2, 4, 10, 25, 50, 100, 200, 300) if not SMOKE \
    else (1, 2, 4, 36)
FULL_DIAL_AT: tuple[int, ...] = (50, 300) if not SMOKE else ()
N_STEPS = CK_MAIN[-1]
FT_LR = 1e-3
FREEZE_SEED = 10902               # the locked seed lineage
CONS_SEED = 10901                 # e109/e113 arm (a) seed (e157's stage A)
TRAIN_CAP_S = 1800.0              # dispatch: <=1800 s per training
WASH_ANCH_BS, WASH_RAND_BS = 16, 16       # e176n arm A's wash batch: 16 + 16
RP_NAME_BS, RP_ANCH_BS = 16, 16           # e174 arm B's replay batch: 16 + (8+8)
COOLDOWN_S = 60.0                 # dispatch band 60-120 s (pre-training only;
                                  # the next training's pre-cooldown + CPU dial
                                  # time covers the post-gap)
MIDRUN_POLL_EVERY = 25            # mid-run GPU contention poll cadence
SCHED_K = 32                      # CELL-SCHED's external r=1/32

# ---- root construction (the spec's re-install clause; frozen ladder) ----------
# e157_f2_consolidated's ruler (g+12 0.591) < ROOT_BAR 0.7 -> fresh roots.
# Each attempt: e043-Dmix install (E43.exposure, mix 16 paired + 32 random,
# house cosine at total=steps, gen = the e098 seed convention) + e113 jitter
# consolidation (e157's finetune_replay VERBATIM, 300 steps, seed 10901).
ROOT_BAR = 0.7                    # ROOT-STRENGTH (frozen; never lowered)
ROOT_ATTEMPTS: tuple[tuple[str, str, int, int], ...] = (
    ("r1_s4305_i300", "e098_base_s4305.pt", 4305, 300),
    ("r2_s4305_i600", "e098_base_s4305.pt", 4305, 600),
    ("r3_s4307_i300", "e098_base_s4307.pt", 4307, 300),
) if not SMOKE else (
    ("r1s_s4305_i8", "e098_base_s4305.pt", 4305, 8),
)
GATE_PZ_FLOOR, GATE_R1I_MAX = 0.20, 4.5     # e098 GATE-0 bars verbatim

# ---- e170's neutral anchor bank (the wash stream, VERBATIM) --------------------
E170_ANCHOR_SEED = 170
ANCHOR_FORBIDDEN = ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")

# ---- gates / references --------------------------------------------------------
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05

# e157's stored family-2 cells (embedded; verified vs runs/e157/metrics.json):
# the root that FAILED ROOT-STRENGTH (ruler g+12 0.5911 < 0.7) and the wash
# trajectory CELL-BASE must reproduce in shape (dies at +1).
E157_ROOT = {                     # runs/e157 metrics stage B trace row 0
    "gm12": 0.19826222956180573, "g0": 0.5784125924110413,
    "gp12": 0.5911492705345154, "ce_r": 1.9504578113555908,
}
E157_WASH = {                     # runs/e157 metrics stage B trace (seed 10902)
    "label": "e157 stage B: family-2 neutral wash (e176n arm A verbatim), "
             "lr 1e-3, seed 10902 — the stored lineage cell",
    "freeze_steps": [0, 1, 2, 4, 50, 100, 200, 300],
    "gm12": [0.19826222956180573, 0.001129803480580449,
             0.0011759375920519233, 0.003656684886664152,
             0.007448808290064335, 0.004423586186021566,
             0.00012963238987140357, 0.00013348007632885128],
    "g0": [0.5784125924110413, 0.0040224287658929825,
           0.00459279865026474, 0.015774426981806755,
           0.020724201574921608, 0.002758212387561798,
           0.00029339289176277816, 0.00019846257055178285],
    "ce_r": [1.9504578113555908, 2.840304136276245,
             2.7520101070404053, 2.7039923667907715,
             2.0670039653778076, 1.9851480722427368,
             1.8773754835128784, 1.8591028451919556],
}

# ---- registered bar constants (spec sections 5-7; frozen) ----------------------
SHUT_BAR = 0.27                   # die bar (e158/e161/e176/e176n/e179)
MAINTAIN_BAR = 0.50               # maintain bar (e176n/e179)
ECON_MIN, ECON_MAX = 5, 25        # the economy band (events per 300)
SPACING_BAND = (20, 45)           # event-spacing concentration band
CE_HEALTH = 0.10                  # CE_R@300 <= root + this (F4 at +0.25)
GHOST_END_BAR = 0.4               # echo ends below this at +300
COUPLING_DIV = 0.30               # monitor_norm - ruler_norm divergence flag

REGISTERED_PREDICTION = {
    "primary": "under the same rehearsal-free neutral wash that kills the "
        "base by +50 (CELL-BASE ruler <= 0.27 — the contrast gate), g2 "
        "MAINTAINS: ruler >= 0.5 at BOTH horizons +50 AND +300 (e179's "
        "state-based maintain convention verbatim). The trajectory is a "
        "self-triggered SAWTOOTH whose dips stay bounded and whose every "
        "event is followed within <= 18-25 steps by a ruler >= 0.5. "
        "MID-CYCLE DIPS BELOW 0.27 DO NOT UN-MAINTAIN (the late-grid "
        "mean/min over {100,200,300} is co-reported for phase honesty).",
    "secondary": {
        "economy": "5 <= n_events <= 25 (realized r in [1/60,1/12]); event "
            "spacing concentrates in 20-45 steps; maintaining at n_events "
            "<= 4 = a STRONGER economy than e179's (texture).",
        "anatomy": "every dial >= 50% of ITS OWN root value at +300: held30, "
            "site read onset/span, row-0 sink, A129 brake, D-all deletion "
            "survival (D-all g0 >= 0.5 x root's).",
        "organism_health": "CE_R at +300 <= root + 0.10; no sustained "
            "concussion.",
        "ghost_erodes": "CELL-ECHO late-grid mean (over {100,200,300}) < 50% "
            "of CELL-G2's, declining across the three checkpoints, ending "
            "< 0.4 at +300 despite the gate firing; if ECHO maintains too, "
            "locks 2+3 are NOT the active ingredients (recorded either way).",
        "schedule_parity": "CELL-SCHED maintains at r=1/32 on this family "
            "(expected); a failure bounds every g2 verdict with family-2 "
            "schedule-fragility.",
        "coupling": "monitor and ruler track at every checkpoint (both "
            "normalized to root); monitor-high/ruler-low divergence = the "
            "CUE-OVERFIT texture.",
    },
    "falsifiers": {
        "F1_schedule_bound": "CELL-G2 ruler < 0.5 at +50 OR +300 (with the "
            "contrast gate passed and the gate firing n_events >= 3, events "
            "G_REPLAY-verified).",
        "F2_erosion": "maintains at +50 but the late-grid {100,200,300} mean "
            "< 0.5 AND monotonically declining.",
        "F3_gate_never_closes": "every post-refractory monitor check fires "
            "(n_events at the refractory ceiling ~12-13) AND maintain still "
            "fails.",
        "F4_organism_price": "CE_R > root + 0.25 at +300 (sustained).",
        "F5_anatomy_fail": "ruler maintains but D-all deletion at +300 kills "
            "(D-all g0 < 0.27 vs root's survival).",
        "structural_void": "CELL-BASE does not die by +50 -> no contrast; a "
            "size/family bound on the wash law (informative).",
    },
    "law_level_stake": "primary fires -> the resurrection economy is "
        "ARCHITECTURALLY SUFFICIENT (the rhythm was the missing organ; e121 "
        "localized to echo); fails -> the economy was SCHEDULE-CONTINGENT "
        "(the dataloader's phase carried information the net cannot "
        "self-supply).",
    "registration": "scratch/g2_design.md sections 6-7 VERBATIM (committed "
        "before this implementation); frozen — no bar shopping.",
}

trims: list[str] = []
device_events: list[dict] = []
deviations: list[str] = [
    "FILENAME: the design doc names lab/g2_rehearsal_native.py; the dispatch "
    "mission names lab/g2_rehearsal_organ.py — implemented under the "
    "mission's name (same spec). The PNG is runs/g2/rehearsal_organ.png "
    "(the doc's g2_wash.png), same panels.",
    "F2 NET: the on-disk family-2 line is block_size=512 = 873,472 params "
    "(e157's F2_CFG, the checkpoints the spec cites); the doc's 'block=256 "
    "= 0.838M' misstates the line's wpe. All training windows remain "
    "256-token (e157's convention); <=1M holds.",
    "FRESH ROOT (the spec's own clause): e157_f2_consolidated FAILED "
    "ROOT-STRENGTH (ruler g+12 0.5911 < 0.7) — root re-built as e043-Dmix "
    "install (MORE exposure steps; e098's E43.exposure call verbatim) + "
    "e113 jitter consolidation (e157's finetune_replay verbatim, seed "
    "10901) on an e098 base; the frozen attempt ladder is in-file; every "
    "attempt is recorded; the bar was never lowered.",
    "G_REPRO is SHAPE-level by construction: CELL-BASE's root differs from "
    "e157's (the re-install clause), so bit-reproduction of the stored "
    "e157_f2_neutral cells is impossible; the gate checks the registered "
    "SHAPE (dead at +1/+50 on the same wash stream/arm-A-identical RNG "
    "shapes) with e157's stored trace embedded + verified vs file.",
    "MONITOR POPULATION (registered here): K_MON=8 windows, offsets cycling "
    "{0,-4,+4} per the spec; the within-offset window rotation is this "
    "implementation's frozen choice (check c, window k -> block(k%3 "
    "offset), index (c*8+k) % 60) — deterministic, no RNG, covers the pool "
    "over time.",
    "GATE INIT: last_event_step = 0 (install end = the last teaching), so "
    "the refractory covers steps 1..23 and the FIRST possible event is "
    "step 24 (24 % CADENCE == 0).",
    "CELL-ECHO's event draws keep the g2 RNG shapes (ix(16,)+aj(8,)+rj(8,) "
    "at seed 10902) and map ix -> the j=0 block as 120 + (ix % 60) — "
    "verbatim j=0 cue windows, full-token CE, NO mask.",
    "Extra texture evals: for gate cells, a light ruler read at each "
    "event's refractory-expiry step (event + 24) — no-RNG, no-grad; "
    "measures the registered 'event followed within <= 18-25 steps by "
    "ruler >= 0.5' sawtooth clause directly.",
    "Cooldowns are 60 s (the dispatch's 60-120 s band), BEFORE each "
    "training only — the next training's pre-cooldown and the CPU dial "
    "time between trainings cover the after-gap (e179 used 90 s both "
    "sides; recorded trim, not a protocol change).",
    "ANATOMY sign handling: the >= 50%-of-own-root bar is applied to "
    "positive root dials; a root dial <= 0 (e.g. a negative A129) is "
    "reported as ratio texture with the sub-boolean marked trivial.",
    "GPU float nondeterminism vs the CPU-eval'd e157 references: every bar "
    "lives at order-of-magnitude separations; the G_ROOT self-consistency "
    "gate reports both the 5e-6 bit flag and the 0.05 fallback.",
    "Single seed (10902), single lineage, n=1 per cell — point estimates "
    "until replicated (e179's own clause).",
    "Smoke mode trims: one 8-step root attempt, 36-step cells, checkpoints "
    "{1,2,4,36}, lean dials, no cooldowns — nothing adjudicated.",
]


# ------------------------------------------------------------------ device pick
# PROVENANCE: lab/e184_seed_replicates.py VERBATIM via e179/e157.

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


def migrate_to_cpu(net, opt) -> None:
    """Move net + optimizer state to CPU in place (params persist)."""
    net.to("cpu")
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state.get(p, {})
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to("cpu")


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/e179_rehearsal_law.py VERBATIM (= e180 = e176n/e161/e152/
# e151/e143/e131/e119/e113/e068/e043), on the family-2 cfg. Copied rather
# than imported to own the device policy.

def load_f2(path) -> TinyGPT:
    m = TinyGPT(F2_CFG)
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


def evl_load(sd: dict) -> TinyGPT:
    m = TinyGPT(F2_CFG)
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def battery_cell(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> dict:
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
def battery_pz(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> float:
    """e116's scalar battery (census readout)."""
    net.eval()
    pzs = []
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs += [float(pr[k, zid]) for k in range(pr.shape[0])]
    return float(np.mean(pzs))


@torch.no_grad()
def ce_fixed_cpu(net: TinyGPT, x, y, bs=64) -> float:
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
        tries += 1
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
def read_fact_at(net: TinyGPT, pool_x: torch.Tensor, name_ids, zid: int,
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


def row_census(net: TinyGPT, rows, readout, *rargs) -> dict:
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


# ------------------------------------------------------------------ batches
# PROVENANCE: lab/e179_rehearsal_law.py VERBATIM (device-parameterized);
# echo_loss is the ghost control's full-token form.

def wash_loss(net, anchor, aj, train_ids, rj, dev):
    """e176n arm A's WASH batch VERBATIM arithmetic: 16 neutral-bank draws +
    16 random corpus windows, full-token CE."""
    anc = anchor[aj]
    rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj])
    x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0).to(dev)
    y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0).to(dev)
    logits, _ = net(x)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


def replay_loss(net, jit_x, jit_mask, ix, anchor, aj, train_ids, rj, dev):
    """e174 arm B's F1-REPLAY batch VERBATIM (make_batch): 16 F1 jitter
    windows (7 name-char masked CE) + 16 anchors (8 paired neutral + 8
    random, full CE), union CE — the ONLY delta vs a wash step is the
    replay channel."""
    nw = jit_x[ix]
    anc = torch.cat([anchor[aj],
                     torch.stack([train_ids[s: s + BLOCK] for s in rj])], 0)
    x = torch.cat([nw[:, :-1], anc[:, :-1]], 0).to(dev)
    y = torch.cat([nw[:, 1:], anc[:, 1:]], 0).to(dev)
    m = torch.zeros(RP_NAME_BS + RP_ANCH_BS, x.shape[1], dtype=torch.bool,
                    device=dev)
    m[:RP_NAME_BS] = jit_mask[ix].to(dev)
    logits, _ = net(x)
    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                          reduction="none").view(x.shape[0], x.shape[1])
    nm = nll[:RP_NAME_BS][m[:RP_NAME_BS]]
    cm = nll[RP_NAME_BS:]
    return (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())


def echo_loss(net, nw, anchor, aj, train_ids, rj, dev):
    """CELL-ECHO's event (the e121 ghost): full-token CE on 16 verbatim j=0
    cue windows + the same 16 anchors (8 neutral + 8 random) — NO mask, NO
    jitter (locks 2+3 off; lock 1 on)."""
    anc = torch.cat([anchor[aj],
                     torch.stack([train_ids[s: s + BLOCK] for s in rj])], 0)
    x = torch.cat([nw[:, :-1], anc[:, :-1]], 0).to(dev)
    y = torch.cat([nw[:, 1:], anc[:, 1:]], 0).to(dev)
    logits, _ = net(x)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


# ------------------------------------------------------------------ the organ
# G2Net: the body (byte-identical TinyGPT — every dial runs on it unchanged)
# + the cue pool in registered buffers. ZERO new learnable parameters: the
# organ contributes only gradients at events; it is NOT in the read path.

class G2Net(nn.Module):
    """THE REHEARSAL ORGAN (spec section 4). state_dict carries the pool
    (the trench-coat clause: the pool lives in the model, checkpoints
    included); the body keys are prefixed 'body.'."""

    def __init__(self, body: TinyGPT, cue_pool: torch.Tensor,
                 cue_mask: torch.Tensor):
        super().__init__()
        assert cue_pool.shape == (300, BLOCK) and cue_mask.shape == (300, BLOCK - 1)
        assert int(cue_pool.max()) < 32768, "cue ids overflow int16"
        self.body = body
        self.register_buffer("cue_pool", cue_pool.to(torch.int16))
        self.register_buffer("cue_mask", cue_mask.to(torch.bool))

    def forward(self, idx, targets=None):     # delegation (dials use .body)
        return self.body(idx, targets)

    # ---- the read monitor (no-grad, no params, no RNG) ----------------------
    @torch.no_grad()
    def monitor(self, c: int, dev=None) -> float:
        """K_MON=8 cue windows, offsets cycling MON_OFFSETS; rotation by the
        check counter c (frozen rule — see deviations). Mean p(true name
        char) over the 7x8 masked name positions."""
        dev = dev or next(self.body.parameters()).device
        pool = self.cue_pool.to(dev).long()
        mask = self.cue_mask.to(dev)
        blocks = {j: JITTERS.index(j) for j in MON_OFFSETS}
        idx = []
        for k in range(K_MON):
            b = blocks[MON_OFFSETS[k % len(MON_OFFSETS)]] * 60
            idx.append(b + ((c * K_MON + k) % 60))
        idx = torch.tensor(idx, device=dev)
        w = pool[idx]
        x, y = w[:, :-1], w[:, 1:]
        logits, _ = self.body(x)
        pr = F.softmax(logits, -1)
        ptrue = pr.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        return float(ptrue[mask[idx]].mean())

    # ---- the rehearsal gate (a comparator, no params) -----------------------
    def step_kind(self, step: int, last_event: int, c: int) -> tuple[str, float | None, bool]:
        """'wash' | 'event'. The refractory first (never check inside it);
        then every CADENCE-th step reads the monitor; the opening IS the
        close (one event per opening)."""
        if step - last_event < REFRACTORY:
            return "wash", None, False
        if step % CADENCE == 0:
            mv = self.monitor(c)
            return ("event" if mv < THETA_OPEN else "wash"), mv, mv < THETA_OPEN
        return "wash", None, False


def organ_body_sd(wrapper_sd: dict) -> dict:
    """Strip the 'body.' prefix — every instrument loads the body alone."""
    return {k[len("body."):]: v for k, v in wrapper_sd.items()
            if k.startswith("body.")}


# ------------------------------------------------------------------ root build
# PROVENANCE: the install is e098's E43.exposure call VERBATIM (e043-Dmix:
# 16 paired originals + 32 random anchors, token-weighted masked union CE,
# house cosine at total=steps); the consolidation is e157's finetune_replay
# VERBATIM (e113's ported recipe: 300-window jitter pool + first-16-install
# original-host anchor bank, masked union CE, constant lr 1e-3, seed 10901).

def build_root(tag: str, base_ck: str, gen_seed: int, inst_steps: int,
               net_base: TinyGPT, win_i, inst_mask, anchor_full, train_ids,
               jit_pool_x, jit_pool_mask, ids130, r_eval_xy, zid):
    """One root attempt: fresh e043-Dmix install + e113 jitter consolidation.
    Returns {install_sd-traj, root body sd, jit geos, ruler geos, device}."""
    dev = pick_dev(f"root_{tag}")
    # ---- stage 1: the fresh install (e043-Dmix; more exposure steps) -------
    E43.DEVICE = str(dev)
    inst = copy.deepcopy(net_base).to(dev)
    eval_at = {inst_steps // 2, inst_steps} if not SMOKE else {inst_steps}
    install_traj: list[dict] = []

    def on_eval(net_, step, _dev=dev):
        was_training = net_.training
        net_.eval()
        bz = battery_cell_cpu_dev(net_, ids130.to(_dev), zid)
        r1i = E43.eval_seq(net_, win_i[:, :PRE + len(NAME)].to(_dev),
                           len(NAME), PRE - 1)
        if was_training:
            net_.train()
        install_traj.append({"step": step, "install60_pz": bz["mean_pz"],
                             "r1i_nll": r1i["nll"], "r1i_acc": r1i["acc"]})
        return install_traj[-1]

    tA = time.time()
    E43.exposure(inst, win_i.to(dev), inst_mask.to(dev),
                 anchor_full.to(dev), steps=inst_steps, total=inst_steps,
                 gen=torch.Generator().manual_seed(gen_seed),
                 tag=f"g2_{tag}_install", ckpt=None, eval_at=eval_at,
                 on_eval=on_eval, mix_random=E43.MIX_RANDOM,
                 train_ids=train_ids, log=None)
    sd_inst = {k: v.detach().cpu().clone() for k, v in inst.state_dict().items()}
    del inst
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    log(f"  [{tag}] install {inst_steps} steps in {time.time() - tA:.0f}s "
        f"({dev}): {install_traj[-1] if install_traj else 'no eval'}")

    # ---- stage 2: the e113 jitter consolidation (e157's ported recipe) -----
    if not SMOKE:
        cooldown(COOLDOWN_S)
    cons = consolidate(evl_load(sd_inst), jit_pool_x, jit_pool_mask,
                       anchor_full[:16], train_ids, ids130, r_eval_xy, zid,
                       f"root_{tag}_cons")
    return {"install_sd": sd_inst, "install_traj": install_traj,
            "install_wall_s": round(time.time() - tA, 1),
            "root_sd": cons["sd"], "cons_traj": cons["traj"],
            "cons_device": cons["device"], "install_device": str(dev),
            "gen_seed": gen_seed, "inst_steps": inst_steps}


@torch.no_grad()
def battery_cell_cpu_dev(net, ids, zid):
    """battery_cell arithmetic on the training device (install readouts)."""
    net.eval()
    lg, _ = net(ids)
    pr = F.softmax(lg[:, -1], -1)
    amax = float((lg[:, -1].argmax(-1) == zid).float().sum())
    return {"mean_pz": float(pr[:, zid].mean()),
            "frac_argmax_z": amax / ids.shape[0]}


def consolidate(net0: TinyGPT, pool_x, pool_mask, anchor, train_ids,
                f_eval_ids, r_eval_xy, zid, tag) -> dict:
    """e157's finetune_replay VERBATIM (e113's ported recipe; seed 10901):
    batch 32 = 16 jitter-pool draws + 16 anchors (8 paired originals + 8
    random), e043 token-level union CE (name-masked on pool windows),
    constant lr 1e-3 AdamW (0.9,0.95) wd 0.1 clip 1.0, 300 steps."""
    dev = pick_dev(tag)
    cap = TRAIN_CAP_S
    n_steps = 300 if not SMOKE else 8
    seed = CONS_SEED
    net = copy.deepcopy(net0).to(dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    n_pool, n_anc = pool_x.shape[0], anchor.shape[0]
    traj, t_start = [], time.time()
    evl = copy.deepcopy(net0).to(CPU)          # CPU eval twin
    step = 0
    for step in range(1, n_steps + 1):
        ix = torch.randint(n_pool, (RP_NAME_BS,), generator=gen)
        aj = torch.randint(n_anc, (RP_ANCH_BS // 2,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (RP_ANCH_BS // 2,),
                           generator=gen)
        loss = replay_loss(net, pool_x, pool_mask, ix, anchor, aj,
                           train_ids, rj, dev)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == n_steps:
            sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            evl.load_state_dict(sd_cpu)
            evl.eval()
            bz = battery_cell(evl, f_eval_ids, zid)
            ce_r = ce_fixed_cpu(evl, *r_eval_xy)
            traj.append({"step": step, "install60_pz": bz["mean_pz"],
                         "ce_r": ce_r,
                         "elapsed_s": round(time.time() - t_start, 1)})
            log(f"  [{tag}] s{step:4d} install60 pZ {bz['mean_pz']:.4f} "
                f"CE_R {ce_r:.4f} ({traj[-1]['elapsed_s']:.0f}s)")
        if (time.time() - t_start) > cap:
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
            "device": str(dev), "seed": seed}


# ------------------------------------------------------------------ the cells
# The training loop = e179's finetune_rate with ONE predicate replaced:
# replay = (k >= 1 and step % k == 0)  ->  replay = (step_kind(step) == 'event').
# CELL-BASE = gate hard-disabled (all wash); CELL-SCHED = e179's finetune_rate
# UNTOUCHED at k=32 on an organ-less base.

def run_cell(mode: str, root_body_sd: dict, cue_pool: torch.Tensor,
             cue_mask: torch.Tensor, anchor_neutral, train_ids, itos,
             r_eval_xy, bat_ids, zid, seed: int, ckpt_steps: tuple[int, ...]):
    """mode in {'base','g2','echo','sched'}. 300 optimizer steps; the event
    branch draws e174-arm-B shapes ix(16,)+aj(8,)+rj(8,) and REPLACES the
    wash batch; the wash branch is e176n arm A VERBATIM (aj(16,)+rj(16,)).
    Light CPU evals (3 geos + CE_R + monitor co-report) at the checkpoint
    steps; +24-post-event ruler texture reads for the gate cells."""
    assert mode in ("base", "g2", "echo", "sched")
    tag = {"base": "CELL-BASE", "g2": "CELL-G2", "echo": "CELL-ECHO",
           "sched": "CELL-SCHED"}[mode]
    dev = pick_dev(tag)
    cap = TRAIN_CAP_S
    ckpt_set = set(ckpt_steps)
    n_steps = ckpt_steps[-1]
    gen = torch.Generator().manual_seed(seed)
    n_anc = anchor_neutral.shape[0]
    n_jit = cue_pool.shape[0]

    gate_live = mode in ("g2", "echo")
    if mode == "sched":
        net = TinyGPT(F2_CFG)
        net.load_state_dict(root_body_sd)
        net = net.to(dev)
        body = net
        pool_x, pool_m = cue_pool, cue_mask      # experimenter-held (organ-less)
    else:
        body0 = TinyGPT(F2_CFG)
        body0.load_state_dict(root_body_sd)
        net = G2Net(body0, cue_pool, cue_mask).to(dev)   # organ rides along
        body = net.body
        # the ORGAN's own pool (bit-identical CPU copy of the registered
        # buffers — e179's batch arithmetic builds on CPU, moves to dev)
        pool_x = net.cue_pool.to("cpu").long()
        pool_m = net.cue_mask.to("cpu")
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    evl = TinyGPT(F2_CFG)                        # CPU eval twin (body only)
    evl.load_state_dict(root_body_sd)
    evl.eval()

    traj, sds = [], {}
    monitor_trace, event_log = [], []
    last_event = 0
    n_events = n_checks = 0
    zeph_checks = 0
    replay_checks = {"events": 0, "mask7_ok": 0, "anchors_8_8_ok": 0,
                     "echo_j0_ok": 0}
    batch_sizes = set()
    extra_evals: set[int] = set()
    t_start = time.time()
    step = 0

    def light_eval(step_, was_event, why):
        sd_cpu = {k: v.detach().cpu().clone() for k, v in body.state_dict().items()}
        if why == "ckpt":
            sds[step_] = sd_cpu
        evl.load_state_dict(sd_cpu)
        evl.eval()
        cells = {j: battery_cell(evl, bat_ids[j], zid) for j in GEOS}
        ce_r = ce_fixed_cpu(evl, *r_eval_xy)
        mon = None
        if gate_live:
            mon = net.monitor(step_ // CADENCE)
        row = {"step": step_, "why": why,
               "gm12": cells[-12]["mean_pz"], "g0": cells[0]["mean_pz"],
               "gp12": cells[12]["mean_pz"],
               "frac_argmax_z": cells[0]["frac_argmax_z"],
               "ce_r": ce_r, "monitor": mon,
               "n_events_so_far": n_events, "was_event": bool(was_event),
               "elapsed_s": round(time.time() - t_start, 1)}
        traj.append(row)
        log(f"  [{tag}] {why} +{step_:4d} ruler-cells "
            f"g-12 {row['gm12']:.4f} g0 {row['g0']:.4f} g+12 "
            f"{row['gp12']:.4f} CE_R {ce_r:.4f}"
            + (f" mon {mon:.4f}" if mon is not None else "")
            + f" (events {n_events})")

    for step in range(1, n_steps + 1):
        # ---- the one replaced predicate ---------------------------------
        pre_monitor, fired = None, False
        if gate_live:
            kind, mv, fired = net.step_kind(step, last_event,
                                            c=step // CADENCE)
            if mv is not None:
                n_checks += 1
                monitor_trace.append({"step": step, "check": step // CADENCE,
                                      "monitor": mv, "fired": bool(fired),
                                      "became_event": kind == "event"})
        elif mode == "sched":
            kind = "event" if (step % SCHED_K == 0) else "wash"
        else:
            kind = "wash"

        if kind == "event":
            ix = torch.randint(n_jit, (RP_NAME_BS,), generator=gen)
            aj = torch.randint(n_anc, (RP_ANCH_BS // 2,), generator=gen)
            rj = torch.randint(len(train_ids) - BLOCK - 1, (RP_ANCH_BS // 2,),
                               generator=gen)
            for s in rj:                     # name-free verify (random half)
                txt = "".join(itos[int(c)] for c in
                              train_ids[s: s + 64]) + \
                      "".join(itos[int(c)] for c in
                              train_ids[s + 192: s + BLOCK])
                if "ZEPH" in txt:
                    zeph_checks += 1
            replay_checks["events"] += 1
            replay_checks["anchors_8_8_ok"] += int(
                aj.numel() == RP_ANCH_BS // 2
                and rj.numel() == RP_ANCH_BS // 2)
            if mode == "echo":
                jx = J0_BLOCK * 60 + (ix % 60)     # verbatim j=0 cue windows
                nw = pool_x[jx]
                replay_checks["echo_j0_ok"] += int(bool(
                    jx.numel() == RP_NAME_BS
                    and int(((jx >= J0_BLOCK * 60)
                             & (jx < (J0_BLOCK + 1) * 60)).all()) == 1))
                loss = echo_loss(body, nw, anchor_neutral, aj, train_ids,
                                 rj, dev)
            else:
                per_win = pool_m[ix].sum(-1)
                replay_checks["mask7_ok"] += int(bool(
                    (per_win == len(NAME)).all().item()))
                loss = replay_loss(body, pool_x, pool_m, ix, anchor_neutral,
                                   aj, train_ids, rj, dev)
            last_event = step
            n_events += 1
            pre_monitor = mv if gate_live else None
            event_log.append({"n": n_events, "step": step, "mode": mode,
                              "pre_event_monitor": pre_monitor})
            extra_evals.add(min(step + REFRACTORY, n_steps))
            log(f"  [{tag}] EVENT {n_events} @s{step} "
                f"(pre-monitor {pre_monitor if pre_monitor is None else round(pre_monitor, 4)})")
        else:
            aj = torch.randint(n_anc, (WASH_ANCH_BS,), generator=gen)
            rj = torch.randint(len(train_ids) - BLOCK - 1, (WASH_RAND_BS,),
                               generator=gen)
            for s in rj:
                txt = "".join(itos[int(c)] for c in
                              train_ids[s: s + 64]) + \
                      "".join(itos[int(c)] for c in
                              train_ids[s + 192: s + BLOCK])
                if "ZEPH" in txt:
                    zeph_checks += 1
            loss = wash_loss(body, anchor_neutral, aj, train_ids, rj, dev)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        batch_sizes.add(32)              # both branches are 32 by
                                          # construction (16+16 wash / 16+16
                                          # event); shapes fixed in the losses
        if step in ckpt_set:
            light_eval(step, kind == "event", "ckpt")
        elif step in extra_evals and gate_live:
            light_eval(step, False, "post-event+24")
            extra_evals.discard(step)
        if (time.time() - t_start) > cap:
            log(f"  [{tag}] time cap {cap:.0f}s at s{step}")
            trims.append(f"{tag}: time cap at step {step}")
            break
        if dev.type == "cuda" and step % MIDRUN_POLL_EVERY == 0:
            s = gpu_status()
            if s["mem_total"] > 0 and (s["mem_used"] > 0.85 * s["mem_total"]
                                       or s["temp"] > 80):
                device_events.append({"tag": tag, "step": step,
                                      "event": "MID-RUN MIGRATION",
                                      "status": s})
                log(f"  [{tag}] MID-RUN GPU contention at s{step} ({s}) -> CPU")
                migrate_to_cpu(net, opt)
                dev = CPU
                body = net.body if hasattr(net, "body") else net
                # (pool_x/pool_m are CPU copies already; the monitor follows
                # the wrapper's own device)
    net.eval()
    final_sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # post-event bookkeeping from the traces
    for ev in event_log:
        post = next((t for t in monitor_trace if t["step"] >= ev["step"] + REFRACTORY),
                    None)
        ev["post_refractory_monitor"] = post["monitor"] if post else None
        ev["next_check_fired"] = post["fired"] if post else None
        plus = next((r for r in traj if r["step"] == min(ev["step"] + REFRACTORY, n_steps)), None)
        ev["ruler_cells_at_plus_refractory"] = (
            {k: plus[k] for k in ("gm12", "g0", "gp12")} if plus else None)
        nxt = next((r for r in traj if r["why"] == "ckpt"
                    and r["step"] >= ev["step"]), None)
        ev["ruler_cells_at_next_ckpt"] = (
            {k: nxt[k] for k in ("gm12", "g0", "gp12")} if nxt else None)
    spacings = [b["step"] - a["step"] for a, b in zip(event_log, event_log[1:])]
    return {"mode": mode, "final_sd": final_sd, "sds": sds, "traj": traj,
            "monitor_trace": monitor_trace, "event_log": event_log,
            "event_spacings": spacings, "n_events": n_events,
            "n_checks": n_checks, "realized_r": n_events / max(step, 1),
            "steps_ran": step, "seed": seed,
            "zeph_violations": zeph_checks, "replay_checks": replay_checks,
            "initial_device": "cuda" if not GPU_PARKED else "cpu",
            "final_device": str(dev), "time_cap_s": cap}


# monitor is a method on G2Net (no-grad, device-following); no helper needed


# ------------------------------------------------------------------ checkpoints
CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "g2", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name}")


def verify_e157(path: Path) -> dict:
    """Verify the embedded e157 references vs runs/e157/metrics.json."""
    src = {"source": "embedded verbatim copies (runs/e157/metrics.json)",
           "file_present": path.exists(), "verified_vs_embedded": None,
           "max_abs_diff": None}
    if path.exists():
        mm = json.loads(path.read_text(encoding="utf-8"))
        tr = mm["stages"]["B_wash"]["trace"]
        got = {"freeze_steps": [r["freeze_steps"] for r in tr],
               "gm12": [r["gm12"] for r in tr],
               "g0": [r["g0"] for r in tr],
               "ce_r": [r["ce_r"] for r in tr]}
        diffs = [abs(a - b) for k in ("gm12", "g0", "ce_r")
                 for a, b in zip(got[k], E157_WASH[k])]
        root = tr[0]
        rd = [abs(root["gm12"] - E157_ROOT["gm12"]),
              abs(root["g0"] - E157_ROOT["g0"]),
              abs(root["gp12"] - E157_ROOT["gp12"])]
        steps_ok = got["freeze_steps"] == E157_WASH["freeze_steps"]
        src["max_abs_diff"] = max(diffs + rd)
        src["verified_vs_embedded"] = bool(steps_ok and src["max_abs_diff"] < 1e-9)
        if src["verified_vs_embedded"]:
            src["source"] = (f"runs/e157/metrics.json stage B (embedded "
                             f"copies verified, max|diff| {src['max_abs_diff']:.1e})")
    return src


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("g2_smoke" if SMOKE else "g2")
    common.DEVICE = "cpu"     # ALL readouts CPU-side; trainers own devices
    log(f"G2 THE REHEARSAL ORGAN (family 2 = e098/e157 0.84M line; "
        f"smoke={SMOKE}) -> {rd}")
    log(f"compute: GPU allowed (park-once + mid-run guard every "
        f"{MIDRUN_POLL_EVERY} steps), cooldown {COOLDOWN_S:.0f}s pre-training, "
        f"per-training cap {TRAIN_CAP_S:.0f}s; gpu at start: {gpu_status()}; "
        f"threads {torch.get_num_threads()}")

    # ---------------- protocol rebuild (e143/e151/e157/e179 verbatim)
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    corpus_zeph = train_text.count("ZEPH")
    G_NAMEFREE0 = {"corpus_zeph_count": corpus_zeph,
                   "pass": bool(corpus_zeph == 0)}
    assert G_NAMEFREE0["pass"], f"corpus contains ZEPH x{corpus_zeph}"

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
    L = len(NAME)

    # ---------------- pools (e157's offset_pool machinery, VERBATIM)
    def offset_pool(j: int):
        wins = []
        for p, h in install_occ:
            pre = train_ids[p - PRE - j: p]
            post = train_ids[p + len(h): p + len(h) + POST_CAP - j]
            if len(pre) != PRE + j or len(post) != POST_CAP - j:
                raise RuntimeError(f"window short at p={p} j={j}")
            w = torch.cat([pre, name_ids, post])
            if len(w) != BLOCK:
                raise RuntimeError(f"window len {len(w)} != {BLOCK} at j={j}")
            wins.append(w)
        px = torch.stack(wins)
        pm = torch.zeros(len(wins), BLOCK - 1, dtype=torch.bool)
        pm[:, PRE - 1 + j: PRE - 1 + j + L] = True
        return px, pm

    jit_pools = {j: offset_pool(j) for j in JITTERS}
    jit_pool_x = torch.cat([jit_pools[j][0] for j in JITTERS])   # (300,256)
    jit_pool_mask = torch.cat([jit_pools[j][1] for j in JITTERS])
    pool_183_x, _ = offset_pool(RETEACH_J)          # the j=54 measurement pool
    G_JIT = {
        "jitters": list(JITTERS), "pool_shape": list(jit_pool_x.shape),
        "name_in_place_all": bool(all(
            all(torch.equal(w[PRE + j: PRE + j + L], name_ids)
                for w in jit_pools[j][0]) for j in JITTERS)),
        "mask_targets_per_window": int(jit_pool_mask[0].sum()),
        "masks_vary_with_jitter": bool(
            len({int(jit_pools[j][1][0].nonzero()[0]) for j in JITTERS}) ==
            len(JITTERS)),
        "stays_in_band": bool(all(121 <= PRE - 1 + j <= 137 for j in JITTERS)),
        "note": "e113's jitter set reads the name at onset rows 129+j in "
                "[121,137] — e113/e174/e179 VERBATIM arithmetic; this pool "
                "becomes the ORGAN's cue pool (registered_buffer, write-once)",
    }
    G_JIT["pass"] = bool(G_JIT["name_in_place_all"]
                         and G_JIT["mask_targets_per_window"] == L
                         and G_JIT["masks_vary_with_jitter"]
                         and G_JIT["stays_in_band"]
                         and jit_pool_x.shape[0] == 300)
    assert G_JIT["pass"], f"G_JIT FAILED: {G_JIT}"
    log(f"G_JIT: cue pool {tuple(jit_pool_x.shape)} (offsets {list(JITTERS)}), "
        f"7 name targets/window, masks at y-cols "
        f"{[int(jit_pools[j][1][0].nonzero()[0]) for j in JITTERS]}: PASS")

    # the install windows + anchor bank for the e043-Dmix install (e098)
    win_i = torch.stack([torch.cat([train_ids[p - PRE: p], name_ids,
                                    train_ids[p + len(h):
                                              p + len(h) + POST_CAP]])
                         for p, h in install_occ])
    inst_mask = torch.zeros(60, BLOCK - 1, dtype=torch.bool)
    inst_mask[:, PRE - 1: PRE - 1 + L] = True
    anchor_full = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                               for p, _ in install_occ])   # 60 originals

    # ---------------- e170's NEUTRAL anchor bank (the wash stream, VERBATIM)
    arng = random.Random(E170_ANCHOR_SEED)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
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

    def junctions_covered(starts):
        cov = 0
        for s in starts:
            if any(s <= p < s + BLOCK + 1 for p in host_positions):
                cov += 1
        return cov

    jc_neutral = junctions_covered(n_starts)
    host_occ_total = len(host_positions)
    bg_rate = host_occ_total * (BLOCK + 1) / len(train_ids)
    G_ANCHOR = {
        "construction": ("16 plain corpus windows from train_ids, RNG seed "
                         f"{E170_ANCHOR_SEED}, rejection if [s, s+257) "
                         "contains FLORIZEL/ELIZABETH/ZEPH/MIRABEL — e170's "
                         "construction VERBATIM (e176n arm A / e157 / e179)"),
        "n_windows": 16, "seed": E170_ANCHOR_SEED, "starts": n_starts,
        "tries": tries, "rejections": rejections,
        "windows_with_host_content": sum(
            1 for s in n_starts
            if any(f in train_text[s: s + BLOCK + 1] for f in HOSTS)),
        "junctions_covered": jc_neutral,
        "random_channel_background": {
            "host_occurrences_in_train": host_occ_total,
            "est_window_hit_rate": bg_rate},
    }
    G_ANCHOR["pass"] = bool(
        G_ANCHOR["windows_with_host_content"] == 0
        and G_ANCHOR["junctions_covered"] == 0
        and anchor_neutral.shape == (16, BLOCK))
    assert G_ANCHOR["pass"], f"G_ANCHOR FAILED: {G_ANCHOR}"
    log(f"G_ANCHOR: neutral bank 16x{BLOCK} (seed {E170_ANCHOR_SEED}, "
        f"{rejections} rejections/{tries} tries) — host 0/16, junctions "
        f"0/16; random-channel background ~{100 * bg_rate:.1f}%/window: PASS")

    # ---------------- batteries (e119/e157/e179 construction verbatim)
    bat_ids, held_bat, jit_bat = {}, {}, {}
    for j in GEOS:
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
        hs = [train_text[p - PRE - j: p] for p, _ in held_occ]
        held_bat[j] = torch.stack([corpus.encode(c) for c in hs])
    for j in JITTERS:                    # the consolidation geometry battery
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        jit_bat[j] = torch.stack([corpus.encode(c) for c in cs])
    ids130 = bat_ids[0]
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

    # ---------------- the measurement dial (e179's measure, family-2 cfg)
    gates_surg: dict = {}

    def measure(sd: dict, tag: str, lean: bool = False) -> dict:
        net = evl_load(sd)
        out: dict = {"tag": tag}
        out["base"] = {j: battery_cell(net, bat_ids[j], zid) for j in GEOS}
        out["base_held"] = {j: battery_cell(net, held_bat[j], zid)
                            for j in GEOS}
        out["ce_r"] = ce_fixed_cpu(net, *r_eval_xy)
        log(f"[{tag}] base: " + " ".join(f"g{j:+d} {out['base'][j]['mean_pz']:.4f}"
                                         for j in GEOS)
            + " | held30: " + " ".join(
                f"g{j:+d} {out['base_held'][j]['mean_pz']:.4f}" for j in GEOS)
            + f" | CE_R {out['ce_r']:.4f}")
        out["site_read"] = read_fact_at(net, pool_183_x, name_ids, zid,
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
                sd_d, gate = deleted_wpe(sd, rows_)
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
            net.load_state_dict(sd)
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

    def gate_vs(cells: dict, refs: dict, name: str) -> dict:
        keys = [k for k in refs if k in cells]
        missing = [k for k in refs if k not in cells]
        diffs = {k: cells[k] - refs[k] for k in keys}
        max_abs = max(abs(v) for v in diffs.values())
        g = {"cells": {k: cells[k] for k in keys}, "refs": refs,
             "skipped_missing": missing, "diffs": diffs,
             "max_abs_diff": max_abs, "bit_tol": G_BIT_TOL,
             "tol": G_FALLBACK_TOL, "bit": bool(max_abs < G_BIT_TOL),
             "pass": bool(max_abs < G_FALLBACK_TOL)}
        log(f"GATE {name}: max|diff| {max_abs:.2e} (tol {G_FALLBACK_TOL}): "
            + ("PASS" if g["pass"] else "FAIL")
            + (" (bit)" if g["bit"] else ""))
        return g

    src_e157 = verify_e157(E157_METRICS)

    # =====================================================================
    # THE ROOT — e157's failed ROOT-STRENGTH check, then the fresh ladder
    # =====================================================================
    log("=" * 78)
    log("STEP 0 — the e157 s4305 root's ROOT-STRENGTH check (spec: reuse "
        "only if ruler >= 0.7)")
    e157_root = load_f2(CKPT_DIR / "e157_f2_consolidated.pt")
    e157_geos = {j: battery_cell(e157_root, bat_ids[j], zid)["mean_pz"]
                 for j in GEOS}
    e157_ruler_geo = max(GEOS, key=lambda j: e157_geos[j])
    e157_ruler = e157_geos[e157_ruler_geo]
    del e157_root
    e157_check = {"geos": {f"g{j:+d}": e157_geos[j] for j in GEOS},
                  "ruler_geo": e157_ruler_geo, "ruler": e157_ruler,
                  "root_bar": ROOT_BAR,
                  "passes": bool(e157_ruler >= ROOT_BAR)}
    log(f"e157_f2_consolidated ruler g{e157_ruler_geo:+d} = {e157_ruler:.4f} "
        f"-> {'REUSE' if e157_check['passes'] else 'FAILS ROOT-STRENGTH (fresh root required)'}")
    root_bound = False
    if e157_check["passes"]:
        root_sd = {k: v.clone() for k, v in
                   load_f2(CKPT_DIR / "e157_f2_consolidated.pt").state_dict().items()}
        root_attempts_rec = [{"attempt": "e157_f2_consolidated (reused)",
                               **{k: v for k, v in e157_check.items()
                                  if k != "geos"}}]
        root_meta = {"desc": "e157_f2_consolidated reused (ROOT-STRENGTH "
                             "pass)", "base": "runs/checkpoints/"
                             "e157_f2_consolidated.pt"}
    else:
        attempts_rec: list[dict] = []
        recs_kept: list[dict] = []          # root_sds kept (no rebuilds)
        chosen = None
        for (atag, base_ck, gseed, isteps) in ROOT_ATTEMPTS:
            if not SMOKE:
                cooldown(COOLDOWN_S)
            log(f"ROOT ATTEMPT {atag}: {base_ck} + e043-Dmix install "
                f"{isteps} steps (gen {gseed}) + e113 consolidation")
            net_base = load_f2(CKPT_DIR / base_ck)
            rec = build_root(atag, base_ck, gseed, isteps, net_base, win_i,
                             inst_mask, anchor_full, train_ids, jit_pool_x,
                             jit_pool_mask, ids130, r_eval_xy, zid)
            del net_base
            geos = {j: battery_cell(evl_load(rec["root_sd"]),
                                    bat_ids[j], zid)["mean_pz"] for j in GEOS}
            jitg = {j: battery_cell(evl_load(rec["root_sd"]),
                                    jit_bat[j], zid)["mean_pz"]
                    for j in JITTERS} if not SMOKE else {}
            rgeo = max(GEOS, key=lambda j: geos[j])
            a = {"attempt": atag, "base": base_ck, "gen_seed": gseed,
                 "install_steps": isteps,
                 "install_final": rec["install_traj"][-1]
                 if rec["install_traj"] else None,
                 "geos": {f"g{j:+d}": geos[j] for j in GEOS},
                 "jit_geos": {f"g{j:+d}": jitg[j] for j in jitg},
                 "ruler_geo": rgeo, "ruler": geos[rgeo],
                 "passes": bool(geos[rgeo] >= ROOT_BAR),
                 "cons_traj": rec["cons_traj"],
                 "devices": {"install": rec["install_device"],
                             "consolidation": rec["cons_device"]}}
            attempts_rec.append(a)
            recs_kept.append(rec)
            log(f"  [{atag}] root geos "
                + " ".join(f"g{j:+d} {geos[j]:.4f}" for j in GEOS)
                + f" -> ruler g{rgeo:+d} = {geos[rgeo]:.4f} "
                f"({'PASS' if a['passes'] else 'below 0.7'})")
            if a["passes"]:
                chosen = (atag, rec, rgeo, geos, base_ck)
                break
        if chosen is None:
            root_bound = True
            best_i = max(range(len(attempts_rec)),
                         key=lambda i: attempts_rec[i]["ruler"])
            best = attempts_rec[best_i]
            log(f"!! NO attempt reached ROOT_BAR {ROOT_BAR} — adjudicating on "
                f"the BEST root ({best['attempt']}, ruler {best['ruler']:.4f}) "
                f"with the ROOT-BOUND flag (bar never lowered)")
            chosen = (best["attempt"], recs_kept[best_i], best["ruler_geo"],
                      {j: best["geos"][f"g{j:+d}"] for j in GEOS},
                      best["base"])
        root_sd = chosen[1]["root_sd"]
        ruler_geo = chosen[2]
        root_attempts_rec = attempts_rec
        root_meta = {
            "desc": (f"fresh root {chosen[0]}: e043-Dmix install "
                     f"({chosen[1]['inst_steps']} steps, gen "
                     f"{chosen[1]['gen_seed']}) + e113 jitter consolidation "
                     f"(300 steps, seed {CONS_SEED}) on "
                     f"{chosen[1]['install_device']}/"
                     f"{chosen[1]['cons_device']}"),
            "base": f"runs/checkpoints/{chosen[4]}",
        }

    # ruler freeze (no shopping): the geo with max root mean_pz
    RULER_GEO = e157_ruler_geo if e157_check["passes"] else ruler_geo
    RULER_KEY = {-12: "gm12", 0: "g0", 12: "gp12"}[RULER_GEO]
    log(f"RULER frozen: g{RULER_GEO:+d} ({RULER_KEY})")

    # ---------------- root dial + G_ROOT (self-consistency from disk)
    log("=" * 78)
    log("STEP-0 battery (the chosen root)")
    root = measure(root_sd, "root", lean=SMOKE)
    root_cells = flat_cells(root)
    root_monitor_pool = G2Net(evl_load(root_sd), jit_pool_x, jit_pool_mask)
    root_monitor = root_monitor_pool.monitor(0, dev=CPU)
    log(f"root monitor (organ read, c=0): {root_monitor:.4f}")
    save_ckpt("g2_root",
              {f"body.{k}": v for k, v in root_sd.items()}
              | {"cue_pool": jit_pool_x.to(torch.int16),
                 "cue_mask": jit_pool_mask},
              {"desc": root_meta["desc"] + " — organ registered at install "
                                        "end (cue pool 300x256 int16 + mask)",
               "ruler_geo": int(RULER_GEO),
               "root_bound": bool(root_bound), **{
                   k: v for k, v in root_meta.items() if k != "desc"}})
    # G_ROOT: re-load the root from the saved checkpoint; dials self-consistent
    ck = torch.load(CKPT_DIR / ("smoke_g2_root.pt" if SMOKE else "g2_root.pt"),
                    map_location="cpu", weights_only=False)
    root_reload = evl_load(organ_body_sd(ck["model"]))
    re_cells = {j: battery_cell(root_reload, bat_ids[j], zid)["mean_pz"]
                for j in GEOS}
    re_ce = ce_fixed_cpu(root_reload, *r_eval_xy)
    del root_reload, ck
    G_ROOT = {
        "geos_reloaded": {f"g{j:+d}": re_cells[j] for j in GEOS},
        "ce_r_reloaded": re_ce,
        "geos_step0": {"g-12": root_cells["gm12"], "g0": root_cells["g0"],
                       "g+12": root_cells["gp12"]},
        "ce_r_step0": root_cells["ce_r"],
        "max_abs_diff": max(max(abs(re_cells[j] -
                                    {-12: root_cells["gm12"],
                                     0: root_cells["g0"],
                                     12: root_cells["gp12"]}[j]) for j in GEOS),
                            abs(re_ce - root_cells["ce_r"])),
        "bit_tol": G_BIT_TOL, "tol": G_FALLBACK_TOL,
    }
    G_ROOT["bit"] = bool(G_ROOT["max_abs_diff"] < G_BIT_TOL)
    G_ROOT["pass"] = bool(G_ROOT["max_abs_diff"] < G_FALLBACK_TOL)
    log(f"G_ROOT (reload self-consistency): max|diff| "
        f"{G_ROOT['max_abs_diff']:.2e}: "
        + ("PASS" if G_ROOT["pass"] else "FAIL")
        + (" (bit)" if G_ROOT["bit"] else ""))
    if not G_ROOT["pass"]:
        raise RuntimeError("G_ROOT FAILED — root dial not self-consistent")

    # =====================================================================
    # THE FOUR CELLS (BASE first — the contrast gate validates the rig)
    # =====================================================================
    cells_out: dict = {}
    for mode in ("base", "g2", "echo", "sched"):
        log("=" * 78)
        if not SMOKE:
            cooldown(COOLDOWN_S)
        desc = {"base": "gate hard-disabled — the ablation identity",
                "g2": "gate live, all three locks on (THE KEY CONTRAST)",
                "echo": "gate live, verbatim j=0 full-CE events (the ghost)",
                "sched": "organ-less + external r=1/32 (e179 verbatim at k=32)"}
        log(f"CELL {mode.upper()}: {desc[mode]} — {N_STEPS} steps, seed "
            f"{FREEZE_SEED}, checkpoints +{list(CK_MAIN)}")
        cells_out[mode] = run_cell(mode, root_sd, jit_pool_x, jit_pool_mask,
                                   anchor_neutral, train_ids, itos, r_eval_xy,
                                   bat_ids, zid, FREEZE_SEED, CK_MAIN)
        cells_out[mode]["dials"] = {}
        for s in FULL_DIAL_AT:
            if s in cells_out[mode]["sds"]:
                log(f"CELL {mode.upper()} +{s} FULL dial")
                cells_out[mode]["dials"][str(s)] = measure(
                    cells_out[mode]["sds"][s], f"{mode}{s}", lean=SMOKE)
        save_ckpt(f"g2_{mode}",
                  cells_out[mode]["final_sd"],
                  {"desc": f"g2 root + {N_STEPS}-step cell ({desc[mode]}), "
                           f"seed {FREEZE_SEED}, "
                           f"{cells_out[mode]['n_events']} events",
                   "mode": mode, "steps": int(cells_out[mode]["steps_ran"]),
                   "n_events": int(cells_out[mode]["n_events"]),
                   "realized_r": cells_out[mode]["realized_r"],
                   "seed": FREEZE_SEED})
        if 50 in cells_out[mode]["sds"] and 50 != max(cells_out[mode]["sds"]):
            save_ckpt(f"g2_{mode}_s50", cells_out[mode]["sds"][50],
                      {"desc": f"g2 root + 50-step cell ({mode}), seed "
                               f"{FREEZE_SEED}", "mode": mode, "steps": 50})

    # =====================================================================
    # GATES
    # =====================================================================
    log("=" * 78)
    G_NAMEFREE = {"corpus_zeph_count": corpus_zeph,
                  "cell_zeph_violations": {m: cells_out[m]["zeph_violations"]
                                           for m in cells_out},
                  "pass": bool(all(cells_out[m]["zeph_violations"] == 0
                                   for m in cells_out) and corpus_zeph == 0)}
    assert G_NAMEFREE["pass"], "name token leaked into a wash/replay window"
    log("G_NAMEFREE: PASS (0 ZEPH in corpus + every drawn random window)")

    G_REPLAY = {}
    for m in cells_out:
        rc = cells_out[m]["replay_checks"]
        ev = rc["events"]
        if m == "echo":
            ok = bool(ev > 0 and rc["echo_j0_ok"] == ev
                      and rc["anchors_8_8_ok"] == ev) or ev == 0
        else:
            ok = bool(ev == 0 or (rc["mask7_ok"] == ev
                                  and rc["anchors_8_8_ok"] == ev))
        G_REPLAY[m] = {"n_events": ev, "checks": rc, "pass": ok}
    g_replay_all = all(G_REPLAY[m]["pass"] for m in G_REPLAY)
    log(f"G_REPLAY (per-event batch composition): "
        + " ".join(f"{m} {G_REPLAY[m]['pass']}" for m in G_REPLAY)
        + f" -> {'PASS' if g_replay_all else 'FAIL'}")

    G_STEP = {m: {"steps_ran": cells_out[m]["steps_ran"],
                  "expected": N_STEPS,
                  "batch": 32, "n_events": cells_out[m]["n_events"],
                  "pass": bool(cells_out[m]["steps_ran"] == N_STEPS)}
              for m in cells_out}
    log("G_STEP_PARITY: "
        + " ".join(f"{m}={G_STEP[m]['steps_ran']}" for m in G_STEP)
        + f" -> {'PASS' if all(g['pass'] for g in G_STEP.values()) else 'FAIL'}")

    # G_ORGAN-INERT: +300 dial with zeroed organ buffers is bit-identical
    goi = {"cell": "g2", "step": 300, "bit": None, "pass": None}
    if "300" in cells_out["g2"]["dials"]:
        wrap = G2Net(TinyGPT(F2_CFG), jit_pool_x, jit_pool_mask)
        wrap.load_state_dict(cells_out["g2"]["final_sd"])
        wrap.cue_pool.zero_()
        wrap.cue_mask.zero_()
        zbat = {j: battery_cell(wrap.body, bat_ids[j], zid)["mean_pz"]
                for j in GEOS}
        zce = ce_fixed_cpu(wrap.body, *r_eval_xy)
        d300 = cells_out["g2"]["dials"]["300"]
        ref = {-12: d300["base"][-12]["mean_pz"], 0: d300["base"][0]["mean_pz"],
               12: d300["base"][12]["mean_pz"]}
        goi["max_abs_diff"] = max(max(abs(zbat[j] - ref[j]) for j in GEOS),
                                  abs(zce - d300["ce_r"]))
        goi["bit"] = bool(goi["max_abs_diff"] < G_BIT_TOL)
        goi["pass"] = bool(goi["max_abs_diff"] < G_FALLBACK_TOL)
        del wrap
    G_ORGAN_INERT = goi
    log(f"G_ORGAN-INERT (zeroed buffers, +300 dial): "
        f"max|diff| {goi.get('max_abs_diff')}: "
        + ("PASS" if goi["pass"] else ("SKIPPED (no +300 dial)" if goi["pass"]
                                       is None else "FAIL")))

    # G_REPRO: CELL-BASE reproduces the family-2 stored wash SHAPE (the root
    # is fresh by the spec's own clause — bit-repro impossible; recorded)
    base_traj = cells_out["base"]["traj"]
    base_ruler = {r["step"]: r[RULER_KEY] for r in base_traj}

    def val_at(traj, step):
        return next((r[RULER_KEY] for r in traj if r["step"] == step), None)

    g_repro = {"form": "shape-level (fresh root; bit-repro N/A by design)",
               "e157_ref": src_e157,
               "base_ruler_at": {str(s): val_at(base_traj, s)
                                 for s in (1, 2, 4, 50, 100, 200, 300)
                                 if val_at(base_traj, s) is not None},
               "e157_stored_gm12_at": {str(s): v for s, v in
                                       zip(E157_WASH["freeze_steps"][1:],
                                           E157_WASH["gm12"][1:])},
               "dies_like_e157": None, "pass": None}
    if SMOKE:
        g_repro["pass"] = True
        g_repro["note"] = "smoke: not adjudicated"
    else:
        b1 = val_at(base_traj, 1)
        b50 = val_at(base_traj, 50)
        g_repro["base_ruler_at_1"] = b1
        g_repro["base_ruler_at_50"] = b50
        g_repro["dies_like_e157"] = bool(
            b1 is not None and b1 <= SHUT_BAR and b50 is not None
            and b50 <= SHUT_BAR)
        g_repro["pass"] = g_repro["dies_like_e157"]
    G_REPRO = g_repro
    log(f"G_REPRO (BASE vs e157 stored wash, shape-level): "
        + ("PASS" if G_REPRO["pass"] else "FAIL")
        + f" (base ruler +1 {g_repro.get('base_ruler_at_1')}, +50 "
        f"{g_repro.get('base_ruler_at_50')}; e157 stored +1 "
        f"{E157_WASH['gm12'][1]:.4f})")

    # hard gate enforcement (spec: all gates PASS before any cell adjudicates;
    # G_REPRO's failure is the STRUCTURAL VOID outcome, not a crash — it
    # adjudicates below)
    if not SMOKE:
        hard = {"G_NAMEFREE": G_NAMEFREE["pass"], "G_JIT": G_JIT["pass"],
                "G_ANCHOR": G_ANCHOR["pass"], "G_ROOT": G_ROOT["pass"],
                "G_REPLAY": g_replay_all,
                "G_STEP_PARITY": all(g["pass"] for g in G_STEP.values()),
                "G_ORGAN_INERT": G_ORGAN_INERT["pass"] is not False}
        bad = [k for k, v in hard.items() if not v]
        if bad:
            raise RuntimeError(f"gate(s) FAILED: {bad}")

    # =====================================================================
    # ADJUDICATION (registered clauses; no shopping)
    # =====================================================================
    log("=" * 78)
    ruler_at = {m: {r["step"]: r[RULER_KEY] for r in cells_out[m]["traj"]}
                for m in cells_out}
    late_steps = [s for s in (100, 200, 300) if s in ruler_at["g2"]]
    late_vals = [ruler_at["g2"][s] for s in late_steps]
    late_mean = float(np.mean(late_vals)) if late_vals else float("nan")

    contrast = {"base_ruler_50": ruler_at["base"].get(50),
                "bar": SHUT_BAR,
                "pass": bool(ruler_at["base"].get(50, 1.0) <= SHUT_BAR)}
    gate_firing = {"n_events": cells_out["g2"]["n_events"],
                   "bar": 3,
                   "g_replay_ok": bool(G_REPLAY["g2"]["pass"]),
                   "pass": bool(cells_out["g2"]["n_events"] >= 3
                                and G_REPLAY["g2"]["pass"])}
    primary = {"g2_ruler_50": ruler_at["g2"].get(50),
               "g2_ruler_300": ruler_at["g2"].get(300),
               "bar": MAINTAIN_BAR,
               "maintains": bool(
                   (ruler_at["g2"].get(50, 0.0) >= MAINTAIN_BAR)
                   and (ruler_at["g2"].get(300, 0.0) >= MAINTAIN_BAR)),
               "late_grid_mean_100_200_300": late_mean,
               "late_grid_min": float(min(late_vals)) if late_vals else None}

    F = {}
    F["F1_schedule_bound"] = bool(
        contrast["pass"] and gate_firing["pass"]
        and not primary["maintains"])
    F["F2_erosion"] = bool(
        contrast["pass"] and gate_firing["pass"]
        and (ruler_at["g2"].get(50, 0.0) >= MAINTAIN_BAR)
        and late_mean < MAINTAIN_BAR
        and all(a > b for a, b in zip(late_vals, late_vals[1:])))
    trace = cells_out["g2"]["monitor_trace"]
    all_fired = bool(len(trace) > 0 and all(t["fired"] for t in trace))
    F["F3_gate_never_closes"] = bool(
        contrast["pass"] and gate_firing["pass"] and all_fired
        and not primary["maintains"])
    root_ce = root_cells["ce_r"]
    ce300 = next((r["ce_r"] for r in cells_out["g2"]["traj"]
                  if r["step"] == 300), None)
    F["F4_organism_price"] = bool(
        ce300 is not None and ce300 > root_ce + 0.25)
    g2d300 = cells_out["g2"]["dials"].get("300")
    F["F5_anatomy_fail"] = bool(
        primary["maintains"] and g2d300 is not None
        and flat_cells(g2d300)["dall_g0"] < SHUT_BAR
        and root_cells.get("dall_g0", 0.0) >= SHUT_BAR)
    structural_void = bool(not contrast["pass"])

    # secondary outcomes
    n_ev = cells_out["g2"]["n_events"]
    spac = cells_out["g2"]["event_spacings"]
    S1 = {"n_events": n_ev, "realized_r": cells_out["g2"]["realized_r"],
          "in_band": bool(ECON_MIN <= n_ev <= ECON_MAX),
          "spacings": spac,
          "frac_in_20_45": float(np.mean([SPACING_BAND[0] <= s <= SPACING_BAND[1]
                                          for s in spac])) if spac else None,
          "cheaper_than_schedule": bool(n_ev <= 4 and primary["maintains"])}
    S2 = {"root": {k: root_cells.get(k) for k in
                   ("held30_g0", "site_read_onset", "site_read_span",
                    "row0_strength", "A129", "dall_g0")}}
    if g2d300 is not None:
        c300 = flat_cells(g2d300)
        S2["g2_300"] = {k: c300.get(k) for k in S2["root"]}
        S2["ratios"] = {}
        for k, rv in S2["root"].items():
            v300 = S2["g2_300"].get(k)
            if rv is None or v300 is None:
                S2["ratios"][k] = None
            elif rv > 0:
                S2["ratios"][k] = v300 / rv
            else:
                S2["ratios"][k] = "root<=0 (trivial; texture)"
        S2["all_ge_50pct"] = bool(S2["ratios"]) and all(
            isinstance(r, float) and r >= 0.5 for r in S2["ratios"].values())
        S2["dall_bar"] = {"g2_300_dall": c300.get("dall_g0"),
                          "root_dall": root_cells.get("dall_g0"),
                          "pass": bool(c300.get("dall_g0", 0.0)
                                       >= 0.5 * max(root_cells.get("dall_g0",
                                                                   0.0), 1e-9)
                                       if root_cells.get("dall_g0", 0) > 0
                                       else True)}
    S3 = {"root_ce_r": root_ce, "g2_ce_r_300": ce300,
          "health_bar": root_ce + CE_HEALTH, "pass": bool(
              ce300 is not None and ce300 <= root_ce + CE_HEALTH),
          "max_ce_r_over_ckpts": max((r["ce_r"] for r in cells_out["g2"]["traj"]),
                                     default=None),
          "concussion_note": "the base wash's own +1/+2 transient ~2.0-2.8 "
                             "recovers; sustained elevation fails F4"}
    e_late = [ruler_at["echo"][s] for s in (100, 200, 300)
              if s in ruler_at["echo"]]
    g_late = [ruler_at["g2"][s] for s in (100, 200, 300)
              if s in ruler_at["g2"]]
    echo_n = cells_out["echo"]["n_events"]
    echo_maintains = bool(
        ruler_at["echo"].get(50, 0.0) >= MAINTAIN_BAR
        and ruler_at["echo"].get(300, 0.0) >= MAINTAIN_BAR)
    S4 = {"echo_n_events": echo_n, "gate_firing": bool(echo_n >= 3),
          "echo_late_mean": float(np.mean(e_late)) if e_late else None,
          "g2_late_mean": float(np.mean(g_late)) if g_late else None,
          "echo_declining": bool(all(a > b for a, b in zip(e_late, e_late[1:]))),
          "echo_ruler_300": ruler_at["echo"].get(300),
          "echo_maintains": echo_maintains}
    if S4["echo_late_mean"] is not None and S4["g2_late_mean"]:
        S4["ghost_erodes"] = bool(
            echo_n >= 3
            and S4["echo_late_mean"] < 0.5 * S4["g2_late_mean"]
            and S4["echo_declining"]
            and ruler_at["echo"].get(300, 1.0) < GHOST_END_BAR)
    else:
        S4["ghost_erodes"] = None
    S4["reading"] = ("GHOST-ERODES (e121's verdict reproduced inside the "
                     "architecture — locks 2+3 are the active ingredients)"
                     if S4.get("ghost_erodes") else
                     ("ECHO-MAINTAINS — locks 2+3 are NOT the active "
                      "ingredients; e121's verdict must be re-localized "
                      "(recorded)" if echo_maintains else "MIXED/texture"))
    S5 = {"r": 1.0 / SCHED_K, "k": SCHED_K,
          "sched_ruler_50": ruler_at["sched"].get(50),
          "sched_ruler_300": ruler_at["sched"].get(300),
          "n_events": cells_out["sched"]["n_events"],
          "realized_r": cells_out["sched"]["realized_r"]}
    S5["maintains"] = bool(
        (ruler_at["sched"].get(50, 0.0) >= MAINTAIN_BAR)
        and (ruler_at["sched"].get(300, 0.0) >= MAINTAIN_BAR))
    S6 = {"root_monitor": root_monitor, "root_ruler": root_cells[RULER_KEY],
          "points": [], "max_divergence": None, "cue_overfit_flag": False}
    for r in cells_out["g2"]["traj"]:
        if r.get("monitor") is None:
            continue
        mn = r["monitor"] / max(root_monitor, 1e-9)
        rn = r[RULER_KEY] / max(root_cells[RULER_KEY], 1e-9)
        S6["points"].append({"step": r["step"], "monitor_norm": mn,
                             "ruler_norm": rn, "divergence": mn - rn})
    if S6["points"]:
        S6["max_divergence"] = max(p["divergence"] for p in S6["points"])
        S6["cue_overfit_flag"] = bool(
            S6["max_divergence"] > COUPLING_DIV)

    # verdict composition
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)

    if structural_void:
        verdict = "STRUCTURAL VOID"
        clause = (f"CELL-BASE did NOT die by +50 (ruler "
                  f"{fmt(ruler_at['base'].get(50))} > {SHUT_BAR}) — no "
                  f"contrast; a size/family bound on the wash law itself "
                  f"(informative, not a verdict on g2).")
    elif not gate_firing["pass"]:
        verdict = "GATE-SILENT (texture)"
        clause = (f"the gate did not demonstrably fire (n_events "
                  f"{n_ev} vs bar 3 / G_REPLAY {G_REPLAY['g2']['pass']}) — "
                  f"the falsifiers cannot adjudicate; primary reported "
                  f"regardless: maintains={primary['maintains']}.")
    elif primary["maintains"]:
        verdict = "MAINTAINED"
        bounds = [k for k in ("F4_organism_price", "F5_anatomy_fail") if F[k]]
        clause = (f"under the same wash that killed the base by +50, g2 "
                  f"MAINTAINED: ruler {fmt(ruler_at['g2'].get(50))} at +50 "
                  f"and {fmt(ruler_at['g2'].get(300))} at +300 (bar "
                  f"{MAINTAIN_BAR}); {n_ev} events, realized r "
                  f"{cells_out['g2']['realized_r']:.4f}. The resurrection "
                  f"economy is ARCHITECTURALLY SUFFICIENT — the rhythm was "
                  f"the missing organ."
                  + (f" BOUNDED by {bounds}." if bounds else "")
                  + (" ROOT-BOUND (root < 0.7)." if root_bound else ""))
    else:
        fired = [k for k in ("F1_schedule_bound", "F2_erosion",
                             "F3_gate_never_closes") if F[k]]
        verdict = (" + ".join(fired) if fired else "MAINTAIN-FAILED (no "
                                                              "falsifier form)")
        clause = (f"g2 did NOT maintain (ruler "
                  f"{ruler_at['g2'].get(50)} at +50, "
                  f"{ruler_at['g2'].get(300)} at +300 vs bar {MAINTAIN_BAR}) "
                  f"under the contrast gate; falsifiers fired: {fired}. The "
                  f"resurrection economy was SCHEDULE-CONTINGENT at this "
                  f"threshold/cadence — the dataloader's phase carried "
                  f"information the net cannot self-supply."
                  + (" ROOT-BOUND (root < 0.7)." if root_bound else ""))
    if root_bound:
        verdict += " [ROOT-BOUND]"
    log("=" * 78)
    log(f"G2 VERDICT: {verdict}")
    log(f"  {clause}")
    log(f"  falsifiers: " + " ".join(f"{k}={F[k]}" for k in F))
    log(f"  secondary: economy {S1['in_band']} | anatomy "
        f"{S2.get('all_ge_50pct')} | health {S3['pass']} | ghost "
        f"'{S4['reading']}' | sched-parity {S5['maintains']} | coupling "
        f"div {S6['max_divergence']}")
    log("=" * 78)

    # =====================================================================
    # PLOT — panel A: ruler sawtooth; B: event spacing + pre/post monitor;
    # C: anatomy bars at +300 as % of own root
    # =====================================================================
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.25, 1.0])
    axA = fig.add_subplot(gs[0, :])
    colors = {"base": "#777777", "g2": "#d62728", "echo": "#9467bd",
              "sched": "#1f77b4"}
    labels = {"base": "CELL-BASE (gate off)", "g2": "CELL-G2 (the organ)",
              "echo": "CELL-ECHO (ghost)", "sched": "CELL-SCHED (r=1/32)"}
    for m in ("base", "sched", "echo", "g2"):
        xs = sorted(ruler_at[m])
        axA.plot(xs, [ruler_at[m][s] for s in xs], "-o", ms=3,
                 color=colors[m], label=labels[m], lw=1.8, zorder=3)
    for m, ls in (("g2", "-"), ("echo", ":")):
        evs = [e["step"] for e in cells_out[m]["event_log"]]
        for e in evs:
            axA.axvline(e, color=colors[m], ls=ls, lw=0.8, alpha=0.45,
                        zorder=1)
    axA.axhline(MAINTAIN_BAR, color="k", ls="--", lw=0.8, alpha=0.6)
    axA.axhline(SHUT_BAR, color="k", ls=":", lw=0.8, alpha=0.6)
    axA.text(302, MAINTAIN_BAR, " maintain 0.5", va="bottom", fontsize=8)
    axA.text(302, SHUT_BAR, " die 0.27", va="bottom", fontsize=8)
    axA.axhline(root_cells[RULER_KEY], color="#2ca02c", lw=0.8, alpha=0.5)
    axA.set_xlabel("wash step"); axA.set_ylabel(f"ruler g{RULER_GEO:+d} mean p(Z)")
    axA.set_title(f"G2 THE REHEARSAL ORGAN — ruler vs step (root "
                  f"{root_cells[RULER_KEY]:.3f}; vlines = gate events)")
    axA.legend(loc="center right", fontsize=8)
    axA.set_xlim(0, 315)

    axB = fig.add_subplot(gs[1, 0])
    esp = cells_out["echo"]["event_spacings"]
    hi = max([50] + spac + esp)
    bins = range(20, hi + 4, 2)
    if spac:
        axB.hist(spac, bins=bins, color="#d62728", alpha=0.7,
                 label="G2 spacing")
    if esp:
        axB.hist(esp, bins=bins, color="#9467bd", alpha=0.45,
                 label="ECHO spacing")
    axB.axvspan(SPACING_BAND[0], SPACING_BAND[1], color="k", alpha=0.08)
    axB.set_xlabel("event spacing (steps)"); axB.set_ylabel("count")
    axB.set_title(f"event spacing (band 20-45; G2 n={n_ev}, "
                  f"ECHO n={echo_n})")
    if spac or esp:
        axB.legend(fontsize=8)

    axC = fig.add_subplot(gs[1, 1])
    ev_g2 = [e for e in cells_out["g2"]["event_log"]
             if e.get("pre_event_monitor") is not None]
    steps_ev = [e["step"] for e in ev_g2]
    pre = [e["pre_event_monitor"] for e in ev_g2]
    post = [e["post_refractory_monitor"] for e in ev_g2]
    axC.plot(steps_ev, pre, "o-", color="#d62728", label="pre-event monitor")
    axC.plot(steps_ev, post, "s--", color="#1f77b4",
             label="post-refractory monitor")
    axC.axhline(THETA_OPEN, color="k", ls="--", lw=0.8)
    axC.set_xlabel("event step"); axC.set_ylabel("monitor p(name char)")
    axC.set_title("per-event pre/post monitor (the re-entry test)")
    axC.legend(fontsize=8)

    fig.tight_layout()
    png = rd / "rehearsal_organ.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    log(f"[plot] {png}")

    # anatomy panel C lives in metrics + a second figure for clarity
    if g2d300 is not None:
        fig2, ax = plt.subplots(figsize=(8, 4.2))
        keys = ["held30_g0", "site_read_onset", "site_read_span",
                "row0_strength", "A129", "dall_g0"]
        c300 = flat_cells(g2d300)
        x = np.arange(len(keys))
        for i, (m, col) in enumerate((("g2", "#d62728"), ("echo", "#9467bd"),
                                      ("sched", "#1f77b4"))):
            d = cells_out[m]["dials"].get("300")
            if d is None:
                continue
            cc = flat_cells(d)
            vals = []
            for k in keys:
                rv, v = root_cells.get(k), cc.get(k)
                vals.append(100 * v / rv if (rv and v is not None and rv > 0)
                            else 0.0)
            ax.bar(x + (i - 1) * 0.25, vals, width=0.25, color=col, label=m)
        ax.axhline(50, color="k", ls="--", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace("_", "\n") for k in keys], fontsize=8)
        ax.set_ylabel("% of own root value")
        ax.set_title("G2 anatomy at +300 as % of own root (bar = 50%)")
        ax.legend()
        fig2.tight_layout()
        png2 = rd / "rehearsal_organ_anatomy.png"
        fig2.savefig(png2, dpi=130)
        plt.close(fig2)
        log(f"[plot] {png2}")

    # =====================================================================
    # metrics.json
    # =====================================================================
    def strip_cell(mrec: dict) -> dict:
        return {"mode": mrec["mode"], "steps_ran": mrec["steps_ran"],
                "n_events": mrec["n_events"], "n_checks": mrec["n_checks"],
                "realized_r": mrec["realized_r"],
                "event_spacings": mrec["event_spacings"],
                "traj": mrec["traj"],
                "monitor_trace": mrec["monitor_trace"],
                "event_log": mrec["event_log"],
                "devices": {"initial": mrec["initial_device"],
                            "final": mrec["final_device"]},
                "dials_flat": {s: flat_cells(d) for s, d in
                               mrec["dials"].items()},
                "dials_full": mrec["dials"]}

    metrics = {
        "experiment": "g2",
        "date": common.now_iso(),
        "purpose": "THE REHEARSAL ORGAN — the generative turn's first built "
                   "architecture (zero-new-parameter wrapper: cue pool + "
                   "read monitor + rehearsal gate + error replay) vs the "
                   "gate-disabled base under the e176N neutral wash, with "
                   "the e121 ghost control and the r=1/32 schedule anchor.",
        "design_spec": str(G2_DESIGN.relative_to(E43.REPO)).replace("\\", "/"),
        "smoke": SMOKE,
        "threads": torch.get_num_threads(),
        "cfg": {**{k: v for k, v in F2_CFG.__dict__.items()},
                "params": F2_PARAMS,
                "note": "family-2 line (e098/e157); training windows 256"},
        "organ": {"cadence": CADENCE, "k_mon": K_MON,
                  "mon_offsets": list(MON_OFFSETS),
                  "theta_open": THETA_OPEN, "refractory": REFRACTORY,
                  "cue_pool": [300, BLOCK], "cue_mask": [300, BLOCK - 1],
                  "new_learnable_params": 0,
                  "root_monitor": root_monitor},
        "compute": {"gpu_at_start": gpu_status(),
                    "parked": GPU_PARKED, "park_reason": PARK_REASON,
                    "device_events": device_events,
                    "cooldown_s": COOLDOWN_S, "train_cap_s": TRAIN_CAP_S},
        "root": {"e157_check": e157_check, "attempts": root_attempts_rec,
                 "root_bound": root_bound, "root_bar": ROOT_BAR,
                 "ruler_geo": int(RULER_GEO), "ruler_key": RULER_KEY,
                 "cells": root_cells, "meta": root_meta,
                 "cons_seed": CONS_SEED},
        "cells": {m: strip_cell(cells_out[m]) for m in cells_out},
        "gates": {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                  "G_JIT": G_JIT, "G_ANCHOR": G_ANCHOR, "G_ROOT": G_ROOT,
                  "G_REPLAY": G_REPLAY, "G_STEP_PARITY": G_STEP,
                  "G_ORGAN_INERT": G_ORGAN_INERT, "G_REPRO": G_REPRO,
                  "gates_surgical": gates_surg},
        "registered_prediction": REGISTERED_PREDICTION,
        "adjudication": {
            "ruler": {"geo": int(RULER_GEO), "key": RULER_KEY,
                      "root_value": root_cells[RULER_KEY],
                      "die_bar": SHUT_BAR, "maintain_bar": MAINTAIN_BAR},
            "contrast_gate": contrast, "gate_firing": gate_firing,
            "primary": primary, "falsifiers": F,
            "structural_void": structural_void,
            "secondary": {"S1_economy": S1, "S2_anatomy": S2,
                          "S3_organism_health": S3, "S4_ghost": S4,
                          "S5_schedule_parity": S5, "S6_coupling": S6},
            "verdict": verdict, "clause": clause,
            "root_bound_flag": root_bound,
        },
        "references": {"e157": src_e157,
                       "e157_wash_embedded": E157_WASH,
                       "e157_root_embedded": E157_ROOT},
        "checkpoints": CKPT_INVENTORY,
        "trims": trims,
        "deviations": deviations,
        "timing_s": round(time.time() - T0, 1),
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))
    log(f"[done] metrics -> {rd / 'metrics.json'} "
        f"({metrics['timing_s']:.0f}s total); verdict: {verdict}")


if __name__ == "__main__":
    main()
