"""G10 — THE FIRST-STEP-AWARE WALL (T170's named structural fix, dispatched
2026-10-02). THE WALL'S BLINDNESS BETWEEN COMMIT AND THE FIRST PROJECTION
RESCALE IS WHERE THE 10M KILL LANDS — THIS CELL TESTS THE THREE REGISTERED
FIXES.

==================== THE QUESTION =========================================
T170 (g1bS6's WALL-FADES verdict) located the 10M kill mechanically: the
commit-and-project wall commits at d=0 and projects at the NEXT forward, so
ONE AdamW step (3.16 raw at lr 1e-3, sqrt(P) = 3158.7) lands BEFORE the
first projection — and it exceeds every rung on the 2.74M-minted ladder
(W1 = 1.336 raw). The first projection rescale then shrinks that step by
0.423 uniform — a scramble that put W1's +1 settled read at 0.0940 vs the
scaled bar 0.7546 (the +1 breach; the fact recovered to 0.87 at +2 but the
every-checkpoint bar was already dead). THE QUESTION: does ANY of the three
registered fix variants hold the strict every-checkpoint bar (the scaled
0.9eq form, bar_0p9eq = 0.7546) where the original wall breached at +1?

==================== THE THREE FIX VARIANTS (dispatch, verbatim) ==========
All on the LOADED PEAK ROOT (runs/checkpoints/g1bS5_root_m020.pt, bit-exact
per g1bS6's G-ROOTLOAD); the wash/projection conventions verbatim from
g1bS6 (e176N arm A wash, seed 10902, AdamW (0.9,0.95) wd 0.1 const lr 1e-3
clip 1.0, ckpts {1,2,4,10,50,100,200,300}, light reads via the ARMED eval
twin = settled reads; both R conventions everywhere):

  F1 STEP-CLIP: the FIRST optimizer step's delta is clipped to the rung
     radius before applying (an lr-scale on step 1 only; then the standard
     commit-and-project cadence) — the wall never sees an out-of-ball first
     step. ARITHMETIC: net committed at theta_0 (R = 1x rung); after
     opt.step() of step 1 ONLY, the wall's own projection
     (net._enforce_wall()) runs once — identical code path and rescale
     arithmetic to what forward 2 would have done. Registered mechanical
     note: the clip produces EXACTLY the state the original wall's +1
     settled read measures (theta_0 + R*dir_1), and from forward 2 onward
     the cadence is identical to W1's — F1 is predicted to REPRODUCE W1's
     loaded trace point-for-point (the discriminating observation; cross-
     run CUDA nondeterminism fuzz expected ~1e-3, gate tol 0.05).

  F2 ANCHOR-AT-ONE: commit at theta_1 (take one unwalled wash step, THEN
     snapshot the anchor and project thereafter) — the anchor includes the
     formation shock. ARITHMETIC: net starts UNCOMMITTED; step 1 runs the
     standard unwalled update (bit-identical to C's step 1 — the loaded
     control's step-1 body, never rerun, is the reference class); after
     opt.step() of step 1, net.commit(R) snapshots theta_1 (d=0) and the
     standard wall cadence runs from forward 2. The +1 checkpoint IS the
     anchor → its settled read = the RAW read of theta_1 ≈ C's loaded +1
     read (0.00067; registered prediction). The ball lives around a state
     the wash already killed.

  F3 DELTA-PROJECTION: project each per-step DELTA into the ball (a trust
     region on steps, not on cumulative displacement) — every step is
     in-ball by construction. ARITHMETIC: net UNCOMMITTED throughout (no
     anchor semantics); after EVERY opt.step(), the step delta
     du_t = theta_t - theta_{t-1} is contracted with the wall's own rescale
     arithmetic (per-tensor accumulation, p <- prev + (p - prev)*(R/||du||)
     when ||du|| > R) so ||du_t|| <= R always. Cumulative displacement is
     UNBOUNDED (no G-PIN; G-STEPREGION replaces it). Reads are RAW reads
     (no settle needed — every state is the model's own defined state).
     Registered mechanical note: step 1's contracted state is again
     theta_0 + R*dir_1 → F3's +1 read is predicted ≈ 0.0940 as well; the
     OPEN bit is the tail (the walk leaves the basin at its own pace).

ARMS: the three variants at the 1x rung (R_rms match, R = 1.3360 raw) +
the g1bS6 C as the LOADED CONTROL (never rerun — parsed from
runs/g1bS6/metrics.json) + the g1bS6 W1 as the LOADED ORIGINAL (the +1
breach record). One 300-step wash per variant, the standard cadence/reads.

==================== REGISTERED BARS (frozen, dispatch verbatim) ==========
  FIX-HOLDS: "fires if any variant's 1x arm holds the scaled bar at EVERY
      checkpoint through +300 — the structural blindness is curable; the
      wall's 10M continuity restored; the winning variant named."
  FIX-PARTIAL: "fires if any variant beats the original W1's +1 breach
      materially (min@+1 > 0.5) but none holds every checkpoint — the
      blindness is reducible, not curable at this dial; the texture
      reported."
  FIX-IMPOTENT: "fires if all variants breach like the original — the +1
      kill is not about the projection timing; the WALL-FADES verdict
      stands untouched."
  The scaled bar itself is g1bS6's committed form VERBATIM:
  bar_0p9eq = root_gm12 * 0.9 / 0.9155886769294739 = 0.7546 (recomputed
  from THIS run's G-ROOTLOAD-gated root dial and cross-checked against
  g1bS6's committed 0.754589897590845 BEFORE any arm trains). No bar
  shopping: FIX-HOLDS takes precedence over FIX-PARTIAL over
  FIX-IMPOTENT.

==================== CHECKS (Rule 12) =====================================
  G-CONFIG    the 10M band + the R-convention assertion (verbatim g1bS6).
  G-ROOTLOAD  the peak-root load gate (g1bS6's G-ROOTLOAD form verbatim:
              |d gm12|, |d ce_r|, |d g0|, |d held30| <= 0.005 vs g1bS5's
              committed record).
  G-CTRL-LOADED  (read, not rerun) the loaded C's committed death record.
  G-BITROOT   F1's committed root bit-identical to theta_0.
  G-COMMITAT1 F2's anchors == its own +1 body (bit) and anch__R == R.
  G-INPUTS    per-step input md5 identical across all three variants.
  G-STEP1F1   F1's +1 body == the wall-projection of the unwalled step-1
              body (F2's +1 body serves as the in-run unwalled reference;
              tol 1e-4) — the clip IS the projection, verified.
  G-STEP1F2   F2's +1 raw read == the loaded C's +1 read (tol 0.02,
              behavioral; the unwalled step-1 body class).
  G-EQUIVF1   (read, tol 0.05/ckpt) F1's trace vs the loaded W1's trace —
              the registered isomorphism, verified empirically.
  G-PIN       F1, F2: raw displacement vs the arm's OWN anchor at ckpts
              <= R + pin fuzz (rms-carried; verbatim R+1.5 co-reported).
              F3 exempt (no cumulative ball) — G-STEPREGION instead:
              every recorded per-step |du| <= R + 1e-3.
  Nothing guaranteed — the openness is the point.

==================== THE OWNER ENVELOPE (STATE.json compute_directive) ===
Mindful, not timid: util AND temp checked before every launch (double-poll
5 s apart; launch at util <= 20% / temp <= 70C / mem <= 60%; every poll
logged); short bursts <= 90 s; cooldown >= 180 s between bursts; free
windows used decisively; wait only when actually contended. The wash arms
NEVER run on CPU; if no window opens within GPU_WAIT_MAX the run STOPS
honestly (partial metrics + resume ckpts hold the record).

==================== LINEAGE ==============================================
Builds on: g1bS6 (the arms machinery, the +1 breach record, the scaled
bar, the loaded control; T170's WALL-FADES + the named fix), g1bS5 (the
peak root artifact), g1bS2 (base+install), g1/g1b/g1bR (the
commit-and-project wall). What is NEW: the first INTERVENTION on the
wall's first-step blindness — three fix variants, one dial (the 1x rung),
the strict every-checkpoint adjudication.

Outputs: runs/g10/{metrics.json, first_step_wall.png}; checkpoints
runs/checkpoints/g10_*.pt (resume ckpts transient, gitignored). No
NOTES/THINKING/QUEUE/STATE edits (the coordinator folds). Progressive
metrics writes after every phase; commit + push per phase.

Run:  cd lab && python g10_first_step_wall.py    (G10_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
import random
import sys
import time
from pathlib import Path

SMOKE = os.environ.get("G10_SMOKE") == "1"
if SMOKE:
    os.environ["G1_SMOKE"] = "1"     # trims G1.ROWS_OLD only (machinery smoke)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
import torch.nn.functional as F                       # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT, cooldown, gpu_ok,   # noqa: E402
                    gpu_status, run_dir, save_json, set_seed)
import e043_install as E43                             # noqa: E402 (REPO,
                                                      # find_occ, SPLICE_RNG,
                                                      # jsonable)
import g1_anchored_ball as G1                          # noqa: E402 — g1's
                                                      # machinery (CommittedGPT,
                                                      # instruments, streams)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

CPU = torch.device("cpu")
torch.set_num_threads(4)     # shared machine (g1bW's adopted trim; g1's
                             # import resets to 8 — reset after import)

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)
G1.log = log                                          # unify the timeline

# ======================================================================
# THE CONFIG (the same 10M host; the owner envelope)
# ======================================================================
G10_CFG = Cfg(vocab=65, n_layer=8, n_head=8, n_embd=320, block_size=256)
G10_PARAMS = 9_977_600          # e005s LARGE's verified 10M-class config
HOST_SEED = 1337                # the "1337 family" (host init + base corpus)
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
SRC_ROOT_CK = "g1bS5_root_m020.pt"
                                 # THE PEAK ROOT (g1bS5's m020: s500 @ 4e-4 =
                                 # 0.20 rms movement, seed 10901; committed
                                 # root g-12 0.7676599621772766) — LOADED
                                 # VERBATIM (g1bS6's G-ROOTLOAD form); the
                                 # licensed theta0 of every g10 arm
G1BS6_METRICS = E43.REPO / "runs" / "g1bS6" / "metrics.json"
                                 # the loaded control C + the loaded original
                                 # W1 (NEVER rerun; parsed, not retrained)

# ---- THE R LADDER (frozen convention): R_rms = 0.7/sqrt(2.74e6) ----------
G1B_PARAMS = 2_739_072
G1B_ROOT_GM12 = 0.9155886769294739          # g1b's stored root (the bar's
                                            # denominator, frozen)
R_RMS_REF = 0.7 / math.sqrt(G1B_PARAMS)     # 4.2301e-4 per-coordinate RMS
SQRT_P = math.sqrt(G10_PARAMS)
R1_RAW = 1.0 * R_RMS_REF * SQRT_P           # THE 1x RUNG (rms match)
R1_RMS = R1_RAW / SQRT_P                    # == R_RMS_REF (asserted)
PIN_FUZZ_RMS_CARRIED = (G1.PIN_FUZZ_BAR / math.sqrt(G1B_PARAMS)) * SQRT_P
ONE_STEP_FUZZ_RAW = G1.FT_LR * SQRT_P       # one AdamW step ~ lr*sqrt(P)

WASH_SEED = G1.FREEZE_SEED                    # 10902 (the locked lineage)
CK_WASH: tuple[int, ...] = G1.CK_WASH if not SMOKE else (1, 2, 4)
VARIANTS = ("F1", "F2", "F3")

# ---- g1bS5's committed peak-root record (the G-ROOTLOAD reference) --------
G1BS5_PEAK_ROOT = {               # runs/g1bS5/metrics.json (committed,
                                  # 39a730f): provenance.consolidations.m020
    "ckpt": "runs/checkpoints/g1bS5_root_m020.pt",
    "movement_rms": 0.20, "steps": 500, "lr": 4e-4, "cons_seed": 10901,
    "root_gm12": 0.7676599621772766,
    "g0": 0.7367021441459656, "gp12": 0.8544988036155701,
    "held30_gm12": 0.5573378801345825, "held30_g0": 0.6574546694755554,
    "ce_r": 1.6968417167663574,
    "site_read_onset": 0.6970594525337219,
    "site_read_span": 0.9553064107894897,
    "source": ("runs/g1bS5/metrics.json provenance.consolidations.m020."
               "root_cells (T167's SHARP-OPTIMUM peak root)"),
}

# ---- g1bS6's committed record: the loaded C + the loaded original W1 -----
G1BS6_LOADED = {          # runs/g1bS6/metrics.json (committed; WALL-FADES)
    "source": "runs/g1bS6/metrics.json (parsed at run start; never rerun)",
    "bar_0p9eq": 0.754589897590845,
    "root_gm12": 0.7676599621772766,
    "verdict": "WALL-FADES",
    "gates_pass": True,
    "D_kill_raw": 3.1567881107330322,
    "W1_g_m12": {1: 0.09395793825387955, 2: 0.8709184527397156,
                 4: 0.5674092769622803, 10: 0.8379992842674255,
                 50: 0.7405008673667908, 100: 0.7344890236854553,
                 200: 0.8255654573440552, 300: 0.6870275735855103},
    "W1_min": 0.09395793825387955, "W1_argmin_step": 1,
    "W1_first_below_bar": 1,
    "W1_cum_disp_at_1": 3.1568,
    "C_g_m12": {1: 0.0006727818981744349, 2: 8.457228977931663e-05,
                4: 2.8809463401557878e-06, 10: 1.4179910067468882e-05,
                50: 0.00033003868884406984, 100: 0.002657693810760975,
                200: 0.0003904156037606299, 300: 0.00015885834000073373},
    "C_dead_at": 1,
    "note": ("W1's +1 settled read = the read of theta_0 + R*dir_1 (the "
             "projected first step) — the +1 breach record this cell's "
             "variants are built against"),
}

# ---- THE OWNER ENVELOPE (STATE.json compute_directive) -------------------
OWNER_UTIL_C = 20.0              # launch only when util <= 20%
OWNER_TEMP_C = 70.0              # AND temp <= 70C (double-poll)
OWNER_MEM_FRAC = 0.60            # resident-neighbor caution (mem <= 60%)
COOLDOWN_S = 180.0               # >= 180 s between bursts
TRAIN_CAP_GPU = 90.0             # <= 90 s GPU bursts
GPU_WAIT_MAX = float(os.environ.get("G10_GPU_WAIT_MAX", "7200"))

REGISTERED = {
    "question": ("does any of the three registered first-step-aware fixes "
                 "(F1 STEP-CLIP / F2 ANCHOR-AT-ONE / F3 DELTA-PROJECTION) "
                 "hold the strict every-checkpoint scaled bar (bar_0p9eq "
                 "0.7546) through +300 at the 1x rung on the loaded peak "
                 "root, where the original wall breached at +1 (W1 0.0940)?"),
    "bars_verbatim": {
        "FIX-HOLDS": ("fires if any variant's 1x arm holds the scaled bar at "
                      "EVERY checkpoint through +300 — the structural "
                      "blindness is curable; the wall's 10M continuity "
                      "restored; the winning variant named."),
        "FIX-PARTIAL": ("fires if any variant beats the original W1's +1 "
                        "breach materially (min@+1 > 0.5) but none holds "
                        "every checkpoint — the blindness is reducible, not "
                        "curable at this dial; the texture reported."),
        "FIX-IMPOTENT": ("fires if all variants breach like the original — "
                         "the +1 kill is not about the projection timing; "
                         "the WALL-FADES verdict stands untouched."),
        "precedence": "FIX-HOLDS > FIX-PARTIAL > FIX-IMPOTENT (no shopping)",
    },
    "scaled_bar": {
        "formula": "bar_0p9eq = root_gm12(this run's G-ROOTLOAD-gated dial) "
                   "* 0.9 / 0.9155886769294739",
        "g1b_root_gm12": G1B_ROOT_GM12,
        "g1bs6_committed": G1BS6_LOADED["bar_0p9eq"],
        "holds": ("g-12 >= bar_0p9eq at EVERY checkpoint "
                  "{1,2,4,10,50,100,200,300} (light battery read, g1b/g1bS6's "
                  "convention; settled reads for F1/F2, raw reads for F3 — "
                  "each variant's own defined semantics, registered below)"),
        "computed_before_arms": True,
    },
    "fix_arithmetic": {
        "common": (f"theta_0 = the loaded peak root; R = the 1x rung = "
                   f"{R1_RAW:.6f} raw = {R1_RMS:.6e} rms; the wash VERBATIM "
                   f"e176N arm A (seed {WASH_SEED}, batch 32 = 16 neutral + "
                   f"16 random, full-token CE, AdamW (0.9,0.95) wd 0.1 const "
                   f"lr 1e-3 clip 1.0, 300 steps, ckpts {list(CK_WASH)}); "
                   f"the expected unwalled step-1 raw delta = "
                   f"{ONE_STEP_FUZZ_RAW:.4f} (the loaded record: 3.1568); "
                   f"the step-1 rescale factor s = R/||du_1|| = "
                   f"{R1_RAW / ONE_STEP_FUZZ_RAW:.4f}"),
        "F1_STEP_CLIP": ("net committed at theta_0 with R BEFORE the wash; "
                         "after opt.step() of step 1 ONLY, net._enforce_wall()"
                         " runs once (the wall's own projection arithmetic — "
                         "identical code path to forward 2's); then the "
                         "standard commit-and-project cadence. The +1 body "
                         "is IN-ball (d = R within float fuzz). Settled "
                         "reads (= raw reads, in-ball)."),
        "F2_ANCHOR_AT_ONE": ("net starts UNCOMMITTED; step 1 = the standard "
                             "unwalled wash step (the loaded C's step-1 body "
                             "class); after opt.step() of step 1, "
                             "net.commit(R) snapshots theta_1 as the anchor "
                             "(d = 0); the standard wall cadence runs from "
                             "forward 2 around theta_1. The +1 checkpoint IS "
                             "the anchor: settled read = raw read of "
                             "theta_1."),
        "F3_DELTA_PROJECTION": ("net UNCOMMITTED throughout; after EVERY "
                                "opt.step(), the per-step delta du_t is "
                                "contracted with the wall's own rescale "
                                "arithmetic (p <- prev + (p - prev)*(R/||du||) "
                                "iff ||du_t|| > R) — a trust region on steps; "
                                "cumulative displacement unbounded (G-PIN "
                                "replaced by G-STEPREGION); raw reads."),
        "registered_predictions": {
            "mechanical": ("registered BEFORE compute (Rule 12; nothing "
                           "guaranteed — the openness is the point): "
                           "(a) F1's +1 state = theta_0 + R*dir_1 = EXACTLY "
                           "the state W1's +1 settled read measured -> F1@+1 "
                           "predicted ~= 0.0940, and from forward 2 the "
                           "cadence is identical to W1's -> F1 predicted to "
                           "reproduce W1's loaded trace (G-EQUIVF1); "
                           "(b) F2's +1 = the raw read of theta_1 ~= C's "
                           "loaded +1 read 0.00067 (the anchor absorbs a "
                           "dead state; the ball never reaches theta_0's "
                           "basin) -> F2 predicted dead throughout; "
                           "(c) F3's +1 = theta_0 + R*dir_1 ~= 0.0940 as "
                           "well; F3's TAIL is the genuinely open bit (the "
                           "per-step trust region slows the walk ~2.4x; "
                           "whether the fact survives +300 in-basin is not "
                           "mechanically determined by the loaded record)."),
            "bar_expectation": ("if (a)-(c) hold, the +1 leg breaches for "
                                "all three -> FIX-IMPOTENT fires with the "
                                "equivalence texture; the +1 kill would be "
                                "about the SIZE of the first step at the "
                                "1x rung, not its timing. A min@+1 > 0.5 "
                                "anywhere, or a flat-phase hold, falsifies "
                                "this expectation — reported as found."),
        },
    },
    "wash": {
        "recipe": ("e176N arm A VERBATIM via g1bS6's chunked wash (neutral "
                   "bank seed 170, batch 32 = 16 neutral + 16 random, "
                   "full-token CE, AdamW (0.9,0.95) wd 0.1 const lr 1e-3 "
                   "clip 1.0)"),
        "seed": WASH_SEED, "ckpt_steps": list(CK_WASH),
        "note": ("the wash is the treatment carrier, untouched; the "
                 "first-step mechanics are the ONLY delta between F1/F2/F3 "
                 "and the loaded arms"),
    },
    "R_convention": {
        "R_rms_ref": R_RMS_REF, "derivation": "0.7/sqrt(2739072)",
        "rung_1x_raw": R1_RAW, "rung_1x_rms": R1_RMS,
        "pin_fuzz_rms_carried_raw": PIN_FUZZ_RMS_CARRIED,
        "one_adamw_step_fuzz_raw": ONE_STEP_FUZZ_RAW,
    },
    "owner_envelope": {
        "launch_gate": f"util <= {OWNER_UTIL_C:.0f}% AND temp <= "
                       f"{OWNER_TEMP_C:.0f}C (double-poll 5 s) AND mem <= "
                       f"{OWNER_MEM_FRAC:.0%}",
        "burst_cap_s": TRAIN_CAP_GPU, "cooldown_s": COOLDOWN_S,
        "no_cpu_hop": ("the wash arms NEVER run on CPU; no window within "
                       "GPU_WAIT_MAX => STOP honestly (partial metrics + "
                       "resume ckpts hold the record)"),
    },
    "sources_loaded": {
        "peak_root": (f"runs/checkpoints/{SRC_ROOT_CK} — g1bS5's m020 "
                      "formation-curve peak root (base+install LOADED + "
                      "e113 jitter consolidation s500 @ 4e-4 = 0.20 rms, "
                      "seed 10901); committed root g-12 0.7677; the licensed "
                      "theta0 of every g10 arm — LOADED VERBATIM, identity-"
                      "gated by G-ROOTLOAD (g1bS6's form)"),
        "control_C": ("runs/g1bS6/metrics.json arms.C — the 10M kill clock "
                      "(dead at +1, D_kill 3.1568 raw) — NEVER rerun "
                      "(the dispatch's frozen decision); parsed at start"),
        "original_W1": ("runs/g1bS6/metrics.json arms.W1 / adjudication."
                        "ladder.W1 — the +1 breach record (0.0940 at +1, "
                        "min over the whole trace) — NEVER rerun; parsed"),
        "base_install": ("carried inside the peak-root artifact (g1bS2's "
                         "PASSED base + movement-matched install; their "
                         "identity gates are g1bS6's committed PASS, "
                         "re-derived there on the load — this cell's "
                         "G-ROOTLOAD on the root artifact subsumes the "
                         "lineage)"),
    },
}

deviations: list[str] = [
    "THE BASE/INSTALL GATES RIDE ON THE LOADED ROOT (documented trim): this "
    "cell loads g1bS5's peak-root artifact directly and gates it with "
    "G-ROOTLOAD (g1bS6's form, tol 0.005 vs g1bS5's committed record); "
    "g1bS2's G-BASE/G-BASE-QUAL/G-INST are g1bS6's committed PASS on the "
    "same lineage and are NOT re-derived here (nothing downstream of the "
    "root depends on them beyond the root itself; re-deriving would add "
    "CPU minutes without a new check).",
    "THE CONTROL C AND THE ORIGINAL W1 ARE LOADED, NOT RERUN (the "
    "dispatch's frozen decision): every comparison table carries g1bS6's "
    "committed numbers verbatim; the only rerun arithmetic in this cell is "
    "the three fix washes (each sharing the bit-identical input stream, "
    "md5-gated).",
    "F2's UNWALLED STEP 1 doubles as the in-run unwalled reference (the "
    "loaded C's step-1 body was not saved by g1bS6): G-STEP1F2 checks "
    "F2's +1 read against C's loaded +1 READ (behavioral, tol 0.02) and "
    "G-STEP1F1 checks F1's +1 body against the wall-projection of F2's "
    "+1 body (tol 1e-4) — together these pin the clip-is-the-projection "
    "identity without rerunning C.",
    "THE OWNER ENVELOPE (STATE.json compute_directive) replaces the "
    "lineage's standing GPU policy (verbatim from g1bS6): launch gate "
    "util <= 20% AND temp <= 70C double-polled, bursts <= 90 s, cooldown "
    ">= 180 s, pause-and-wait mid-burst, no CPU fallback.",
    "READ CONVENTIONS PER VARIANT (registered): F1/F2 read via the ARMED "
    "eval twin (settled reads, g1bS6's convention — for F1 the settle is a "
    "no-op at +1 by construction); F3 reads RAW (no anchor semantics — "
    "every state is the model's own defined state). The loaded W1/C reads "
    "are g1bS6's settled/raw conventions respectively.",
    "Smoke mode trims: washes 4 steps with ckpts {1,2,4}, the REAL peak "
    "root loaded (no g1bS5 smoke artifact exists; read-only load), lean "
    "dials, no cooldowns, gate waits capped at 30 s — nothing adjudicated.",
]

device_events: list[dict] = []
trims: list[str] = []

_progressive = {"n": 0, "phases": []}


class OwnerWindowShut(Exception):
    """No owner-envelope GPU window opened within GPU_WAIT_MAX — the run
    stops honestly (partial metrics + resume ckpts hold the record)."""


def write_partial(rd: Path, phase: str, payload: dict) -> None:
    """The outage lesson, mechanized: metrics.json exists from the first
    phase onward and is rewritten after every phase/arm."""
    _progressive["n"] += 1
    _progressive["phases"].append(phase)
    try:
        out = {
            "experiment": "g10_first_step_wall",
            "date": common.now_iso(),
            "partial": True,
            "phase": phase,
            "progressive_writes": _progressive["n"],
            "phases": list(_progressive["phases"]),
            "device_events": device_events,
        }
        out.update(E43.jsonable(payload))
        save_json(rd / "metrics.json", out)
        log(f"[partial] metrics.json updated (phase '{phase}', write "
            f"#{_progressive['n']})")
    except Exception as e:         # bookkeeping must never kill compute
        log(f"[partial] WRITE FAILED at '{phase}' ({e}) — continuing")


# ------------------------------------------------------------------ device
# THE OWNER ENVELOPE gate: util AND temp AND mem, double-poll, park-and-wait
# (verbatim g1bS6 policy).

def owner_gpu_ok() -> bool:
    s = gpu_status()
    return bool(s["util"] <= OWNER_UTIL_C and s["temp"] <= OWNER_TEMP_C
                and (s["mem_total"] == 0
                     or s["mem_used"] <= OWNER_MEM_FRAC * s["mem_total"]))


def wait_gpu_owner(tag: str, max_wait: float | None = None) -> torch.device:
    mw = 30.0 if SMOKE else (GPU_WAIT_MAX if max_wait is None else max_wait)
    if not torch.cuda.is_available():
        raise OwnerWindowShut(
            f"'{tag}': no CUDA device — this cell has no CPU form under the "
            f"owner envelope (stop honestly; resume ckpts hold the record)")
    t0, waited = time.time(), 0.0
    while (time.time() - t0) <= mw:
        if owner_gpu_ok():
            time.sleep(5)                      # the double-poll
            if owner_gpu_ok():
                s = gpu_status()
                device_events.append(
                    {"tag": tag, "event": "LAUNCH WINDOW OPEN",
                     "waited_s": round(waited, 1), "status": s})
                log(f"[gpu] '{tag}' OWNER window open (util {s['util']:.0f}% "
                    f"temp {s['temp']:.0f}C mem {s['mem_used']:.0f}/"
                    f"{s['mem_total']:.0f}MB"
                    + (f"; waited {waited:.0f}s" if waited > 0 else "") + ")")
                return torch.device("cuda")
        else:
            s = gpu_status()
            log(f"[gpu] '{tag}' window shut (util {s['util']:.0f}% temp "
                f"{s['temp']:.0f}C mem {s['mem_used']:.0f}MB) — WAITING "
                f"(owner envelope: util<={OWNER_UTIL_C:.0f}% "
                f"temp<={OWNER_TEMP_C:.0f}C)")
        time.sleep(30)
        waited = time.time() - t0
    device_events.append({"tag": tag, "event": "OWNER-WINDOW TIMEOUT — STOP",
                          "waited_s": round(waited, 1),
                          "status": gpu_status()})
    raise OwnerWindowShut(
        f"'{tag}': no owner window in {mw:.0f}s — stopping honestly "
        f"(resume ckpts + partial metrics hold the record; re-launch "
        f"continues)")


def midrun_pause_wait(tag: str) -> None:
    """g1bW policy: outside load/heat => PAUSE, never migrate."""
    while True:
        s = gpu_status()
        if s["mem_total"] == 0 or (s["mem_used"] <= 0.85 * s["mem_total"]
                                   and s["temp"] <= 75.0):
            log(f"  [{tag}] contention cleared ({s}); resuming")
            return
        log(f"  [{tag}] PAUSED for outside load/heat ({s})")
        time.sleep(30)


def flat_params(net) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()])


def coherence_stats(text: str) -> dict:       # house helper (unused trim kept)
    n = max(1, len(text))
    words = [w for w in text.split() if w]
    runs = max((sum(1 for _ in g) for _, g in itertools.groupby(text)),
               default=0)
    return {"len": len(text),
            "lower_space_fraction": sum(1 for c in text
                                        if c.islower() or c == " ") / n}


# ------------------------------------------------------------------ the wash
# g1bS6_wash VERBATIM + the three variant hooks (the ONLY deltas; each hook
# runs immediately after opt.step(), before any displacement bookkeeping —
# every recorded displacement is the variant's actually-applied state).

def g10_wash(variant: str, net0, anchor: torch.Tensor,
             train_ids: torch.Tensor, itos, r_eval_xy, gm12_ids, g0_ids,
             zid: int, resume_ck: Path | None = None,
             lr: float = G1.FT_LR, R: float = R1_RAW) -> dict:
    seed = WASH_SEED
    ckpt_steps = CK_WASH
    ckpt_set = set(ckpt_steps)
    n_steps = ckpt_steps[-1]
    n_anc = anchor.shape[0]
    step, traj, sds, x_hashes, deltas = 0, [], {}, {}, {}
    zeph_checks = 0
    net, opt, gen = None, None, None
    theta0 = None
    wall_R = R if variant in ("F1", "F2") else None   # bookkeeping display
    n_chunks, chunk_table, devices = 0, [], []
    active_s = 0.0
    committed_f2 = False
    # edge: a prior process saved the FINAL state but died before returning
    if resume_ck is not None and Path(resume_ck).exists():
        _pre = torch.load(resume_ck, map_location="cpu", weights_only=False)
        if int(_pre.get("step", 0)) >= n_steps:
            log(f"  [{variant}] resume ckpt already COMPLETE at "
                f"s{_pre['step']} — returning the saved final state")
            return {"sds": _pre["sds"], "traj": _pre.get("traj", []),
                    "steps_ran": n_steps, "seed": seed, "lr": lr,
                    "zeph_violations": _pre.get("zeph", 0),
                    "x_hashes": _pre.get("x_hashes", {}),
                    "deltas": _pre.get("deltas", {}),
                    "wall_R": _pre.get("wall_R", wall_R),
                    "theta0_norm": float(torch.norm(flat_params(net0))),
                    "device": "resumed (final state)",
                    "n_chunks": _pre.get("n_chunks", 0),
                    "chunk_table": _pre.get("chunk_table", []),
                    "f2_anchors_sd": _pre.get("f2_anchors_sd")}
    while step < n_steps:
        n_chunks += 1
        dev = wait_gpu_owner(f"{variant}-chunk{n_chunks}")   # OWNER envelope
        cap = TRAIN_CAP_GPU
        if net is None:
            net = copy.deepcopy(net0).to(dev)
            net.train()
            opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95),
                                    weight_decay=0.1)
            gen = torch.Generator().manual_seed(seed)
            theta0 = flat_params(net)          # displacement origin (device)
            prev = theta0.clone()
            prev_list = [p.detach().clone() for p in net.parameters()]
            if resume_ck is not None and Path(resume_ck).exists():
                state = torch.load(resume_ck, map_location="cpu",
                                   weights_only=False)
                body, anch = G1.split_anchored_sd(state["model"])
                if variant == "F1":
                    net.load_state_dict(state["model"])  # committed twin:
                    # the full sd (anchors included) matches bit-for-bit
                else:
                    net.load_state_dict(body)
                    if anch:                    # a committed F2 resume
                        G1._restore_anchors(net, anch)
                        if variant == "F2":
                            committed_f2 = True
                net.to(dev)
                net.train()
                opt.load_state_dict(state["opt"])
                gen.set_state(state["gen_state"])
                step = state["step"]
                traj, sds = state.get("traj", []), state.get("sds", {})
                deltas = state.get("deltas", {})
                x_hashes = state.get("x_hashes", {})
                zeph_checks = state.get("zeph", 0)
                active_s = state.get("active_s", 0.0)
                committed_f2 = committed_f2 or state.get("f2_committed",
                                                         committed_f2)
                prev = flat_params(net)
                prev_list = [p.detach().clone() for p in net.parameters()]
                log(f"  [{variant}] RESUMED from {Path(resume_ck).name} at "
                    f"step {step}/{n_steps} ({len(traj)} traj rows)")
        else:
            _pd = str(next(net.parameters()).device)
            if _pd != str(dev):
                _osd = opt.state_dict()
                net = net.to(dev)
                opt = torch.optim.AdamW(net.parameters(), lr=lr,
                                        betas=(0.9, 0.95), weight_decay=0.1)
                opt.load_state_dict(_osd)
                log(f"  [{variant}] chunk {n_chunks}: device moved {_pd} -> "
                    f"{dev} (optimizer state recast)")
            net.train()
            prev = flat_params(net)
            prev_list = [p.detach().clone() for p in net.parameters()]
        devices.append(str(dev))
        evl = copy.deepcopy(net).to(dev)    # eval twin (committed iff net is)
        t_start = time.time()
        chunk_capped = False
        for step in range(step + 1, n_steps + 1):
            aj = torch.randint(n_anc, (G1.ANCH_BS,), generator=gen)
            rj = torch.randint(len(train_ids) - G1.BLOCK - 1, (G1.RAND_BS,),
                               generator=gen)
            anc = anchor[aj]
            rnd = torch.stack([train_ids[s: s + G1.BLOCK] for s in rj])
            for w in rnd:                    # name-free VERIFY (hard-fail)
                txt = "".join(itos[int(c)] for c in w[:64]) + \
                      "".join(itos[int(c)] for c in w[192:])
                if "ZEPH" in txt:
                    zeph_checks += 1
            x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
            y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
            x_hashes[step] = hashlib.md5(
                x.contiguous().numpy().tobytes()).hexdigest()
            xd, yd = x.to(dev), y.to(dev)
            logits, _ = net(xd)                    # <- the wall projects here
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   yd.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            # ============ THE VARIANT HOOKS (the ONLY deltas) =============
            du_raw = None
            if variant == "F1" and step == 1:
                net._enforce_wall()     # THE STEP-CLIP: the wall's own
                                        # projection, one step early
            elif variant == "F2" and step == 1 and not committed_f2:
                net.commit(R)           # THE ANCHOR-AT-ONE: theta_1 snapshot
                committed_f2 = True
                evl = copy.deepcopy(net).to(dev)   # twin becomes committed
                log(f"  [{variant}] committed at theta_1 (the formation "
                    f"shock is inside the anchor; R = {R:.6f} raw)")
            elif variant == "F3":
                # THE DELTA-PROJECTION: trust region on the per-step delta,
                # the wall's own rescale arithmetic with prev as anchor
                with torch.no_grad():
                    d = None
                    for p, pp in zip(net.parameters(), prev_list):
                        dd = ((p - pp) ** 2).sum()
                        d = dd if d is None else (d + dd)
                    d = float(d.sqrt())
                    du_raw = d
                    if d > R:
                        s = R / d
                        for p, pp in zip(net.parameters(), prev_list):
                            p.copy_(pp + (p - pp) * s)
            # =================================================================
            cur = flat_params(net)
            cum_disp = float(torch.norm(cur - theta0))
            inc_disp = float(torch.norm(cur - prev)) \
                if prev is not None else None
            d_anchor = None
            if getattr(net, "anchored", False):
                with torch.no_grad():
                    da = None
                    for name, p in net.named_parameters():
                        a = net._anchor(name)
                        dd = ((p - a) ** 2).sum()
                        da = dd if da is None else (da + dd)
                    d_anchor = float(da.sqrt())
            prev = cur
            prev_list = [p.detach().clone() for p in net.parameters()]
            if step in ckpt_set:
                deltas[step] = (cur - theta0).cpu()
            row = {"step": step, "ce_batch": float(loss.item()),
                   "cum_disp": cum_disp, "cum_disp_rms": cum_disp / SQRT_P,
                   "step_disp": inc_disp,
                   "step_disp_raw_opt": du_raw,
                   "d_anchor": d_anchor,
                   "d_proj": (min(d_anchor, wall_R) if (wall_R
                              and d_anchor is not None) else None),
                   "elapsed_s": round(active_s + time.time() - t_start, 1)}
            if step in ckpt_set:
                sd_cpu = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
                sds[step] = sd_cpu
                evl.load_state_dict(sd_cpu)
                evl.eval()
                gz = G1.battery_cell(evl, gm12_ids.to(dev), zid)
                gz0 = G1.battery_cell(evl, g0_ids.to(dev), zid)
                ce_r = G1.ce_fixed_cpu(evl, r_eval_xy[0].to(dev),
                                       r_eval_xy[1].to(dev))
                row.update({"g_m12_mean_pz": gz["mean_pz"],
                            "g0_mean_pz": gz0["mean_pz"],
                            "frac_argmax_z": gz["frac_argmax_z"],
                            "ce_r": ce_r})
                wr = net.wall_report() if getattr(net, "anchored", False) \
                    else {"d_raw": cum_disp, "d_proj": None, "R": None}
                log(f"  [{variant}] CKPT +{step:4d} g-12 {gz['mean_pz']:.4f} "
                    f"g0 {gz0['mean_pz']:.4f} CE_R {ce_r:.4f} | "
                    f"|d| {cum_disp:.4f} raw = {cum_disp / SQRT_P:.4e} rms "
                    + (f"| d_anchor {d_anchor:.4f}" if d_anchor is not None
                       else "")
                    + (f" | R {R:.4f} raw = {R / SQRT_P:.4e} rms "
                       f"(d_proj {wr['d_proj']})" if wall_R else " (no ball)")
                    + f" (CE {float(loss.item()):.4f})")
            traj.append(row)
            if step % 50 == 0 and step not in ckpt_set:
                log(f"  [{variant}] s{step:4d} CE {float(loss.item()):.4f} "
                    f"|d| {cum_disp:.4f} ({row['elapsed_s']:.0f}s)")
            if (time.time() - t_start) > cap:
                log(f"  [{variant}] chunk {n_chunks}: owner-envelope burst "
                    f"cap {cap:.0f}s at s{step} — resume ckpt saved")
                chunk_capped = True
                break
            if dev.type == "cuda" and step % G1.MIDRUN_POLL_EVERY == 0:
                s = gpu_status()
                if s["mem_total"] > 0 and (s["mem_used"] > 0.85 * s["mem_total"]
                                           or s["temp"] > 80):
                    device_events.append({"tag": variant, "step": step,
                                          "event": "MID-RUN PAUSE (no "
                                                   "migration)", "status": s})
                    t_p = time.time()
                    midrun_pause_wait(variant)
                    t_start += time.time() - t_p
        chunk_secs = round(time.time() - t_start, 1)
        active_s += chunk_secs
        chunk_table.append({"chunk": n_chunks, "device": str(dev),
                            "seconds": chunk_secs,
                            "steps_done": step, "capped": chunk_capped})
        if resume_ck is not None:
            torch.save({"model": {k: v.detach().cpu()
                                  for k, v in net.state_dict().items()},
                        "opt": opt.state_dict(),
                        "gen_state": gen.get_state(),
                        "step": step, "traj": traj, "sds": sds,
                        "deltas": deltas, "x_hashes": x_hashes,
                        "zeph": zeph_checks,
                        "active_s": active_s, "wall_R": wall_R,
                        "n_chunks": n_chunks, "chunk_table": chunk_table,
                        "f2_committed": committed_f2},
                       resume_ck)
        if step >= n_steps:
            break
        if n_chunks >= 24:                # pathological-cap guard
            trims.append(f"{variant}: stopped at chunk {n_chunks} (cap-loop "
                         f"guard) at step {step} of {n_steps}")
            break
        if not SMOKE:
            cooldown(COOLDOWN_S)          # >= 180 s between bursts (owner)
        del evl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    net.eval()
    f2_anchors_sd = None
    if variant == "F2":
        f2_anchors_sd = {k: v.detach().cpu().clone()
                         for k, v in net.state_dict().items()
                         if k.startswith("anch__")}
    return {"sds": sds, "traj": traj, "steps_ran": step,
            "seed": seed, "lr": lr,
            "zeph_violations": zeph_checks, "x_hashes": x_hashes,
            "deltas": deltas, "wall_R": wall_R,
            "theta0_norm": float(torch.norm(theta0.cpu())),
            "device": devices[0] if devices else "n/a", "devices": devices,
            "n_chunks": n_chunks, "chunk_table": chunk_table,
            "f2_anchors_sd": f2_anchors_sd}


CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "g10", **meta}}, path)
    CKPT_INVENTORY[name] = {
        "path": str(path.relative_to(E43.REPO)).replace("\\", "/"), **meta}
    log(f"[ckpt] saved {path.name}")


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("g10_smoke" if SMOKE else "g10")
    log(f"G10 — THE FIRST-STEP-AWARE WALL (T170's named structural fix): "
        f"F1 STEP-CLIP / F2 ANCHOR-AT-ONE / F3 DELTA-PROJECTION at the 1x "
        f"rung ({R1_RAW:.4f} raw = {R1_RMS:.4e} rms) on the LOADED peak root "
        f"({SRC_ROOT_CK}, committed g-12 {G1BS5_PEAK_ROOT['root_gm12']:.4f}); "
        f"C + W1 loaded from g1bS6 (never rerun); bar_0p9eq "
        f"{G1BS6_LOADED['bar_0p9eq']:.4f}; wash seed {WASH_SEED}; owner "
        f"envelope: bursts <= {TRAIN_CAP_GPU:.0f}s, cooldown "
        f"{COOLDOWN_S:.0f}s, gate util <= {OWNER_UTIL_C:.0f}% temp <= "
        f"{OWNER_TEMP_C:.0f}C; smoke={SMOKE} -> {rd}")
    write_partial(rd, "start", {
        "design": ("the g10 dispatch (bars verbatim; fix arithmetic "
                   "registered BEFORE compute; no bar shopping) on T170's "
                   "named fix; the wash/projection conventions verbatim "
                   "from g1bS6"),
        "registered": REGISTERED, "deviations": deviations, "smoke": SMOKE,
        "loaded_g1bs6_record": G1BS6_LOADED,
        "device_events": device_events})
    set_seed(HOST_SEED)      # host init + base corpus (the 1337 family)

    # ---- patch g1's machinery to the 10M family (g1b's pattern) ---------
    G1.G1_CFG = G10_CFG
    G1.G1_PARAMS = G10_PARAMS

    # ---------------- protocol rebuild (g1bS6's main VERBATIM) -------------
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
    for host in G1.HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + G1.POST_CAP <= len(train_ids):
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
    name_ids = corpus.encode(G1.NAME)

    # measurement pool: e152's locked j=54 windows (instrument only)
    wins = []
    for p, h in install_occ:
        pre = train_ids[p - G1.PRE - G1.RETEACH_J: p]
        post = train_ids[p + len(h): p + len(h) + G1.SITE_CONT]
        if len(pre) != G1.PRE + G1.RETEACH_J or len(post) != G1.SITE_CONT:
            raise RuntimeError(f"pool window short at p={p}")
        w = torch.cat([pre, name_ids, post])
        if len(w) != G1.BLOCK:
            raise RuntimeError(f"pool window len {len(w)} != {G1.BLOCK}")
        wins.append(w)
    pool_x = torch.stack(wins)
    G_POOL = {"shape": list(pool_x.shape),
              "name_xcols": [G1.SITE_Z_XCOL, G1.SITE_Z_XCOL + len(G1.NAME) - 1],
              "all_windows_name_in_place": bool(
                  all(torch.equal(w[G1.SITE_Z_XCOL: G1.SITE_Z_XCOL + len(G1.NAME)],
                                  name_ids) for w in pool_x)),
              "note": "MEASUREMENT instrument only; NO window from this pool "
                      "enters any training"}
    G_POOL["pass"] = G_POOL["all_windows_name_in_place"]
    assert G_POOL["pass"], f"pool gate FAILED: {G_POOL}"

    def build_win(p, host):
        return torch.cat([train_ids[p - G1.PRE: p], name_ids,
                          train_ids[p + len(host): p + len(host) + G1.POST_CAP]])

    _ = torch.stack([build_win(p, h) for p, h in install_occ])  # identity

    # ---- e170's neutral bank (e176N arm A's stream, VERBATIM) -----------
    arng = random.Random(G1.E170_ANCHOR_SEED)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - G1.BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        tries += 1
        txt = train_text[s: s + G1.BLOCK + 1]
        if any(f in txt for f in G1.ANCHOR_FORBIDDEN):
            rejections += 1
            continue
        n_starts.append(s)
    if len(n_starts) != 16:
        raise RuntimeError(f"neutral bank incomplete: {len(n_starts)}/16")
    anchor_neutral = torch.stack([train_ids[s: s + G1.BLOCK] for s in n_starts])

    host_positions = [p for p in E43.find_occ(train_text, G1.HOSTS[0])]
    host_positions += [p for p in E43.find_occ(train_text, G1.HOSTS[1])]
    jc_neutral = sum(1 for s in n_starts
                     if any(s <= p < s + G1.BLOCK + 1 for p in host_positions))
    G_ANCHOR = {
        "neutral_bank": {
            "construction": (f"16 plain corpus windows from train_ids, RNG "
                             f"seed {G1.E170_ANCHOR_SEED}, rejection if "
                             f"[s, s+257) contains FLORIZEL/ELIZABETH/ZEPH/"
                             f"MIRABEL — e170's construction VERBATIM (="
                             f" e176n arm A's stream; FIXED content, shared "
                             f"by ALL arms)"),
            "n_windows": 16, "block": G1.BLOCK, "seed": G1.E170_ANCHOR_SEED,
            "starts": n_starts, "tries": tries, "rejections": rejections,
            "forbidden": list(G1.ANCHOR_FORBIDDEN),
            "windows_with_host_content": sum(
                1 for s in n_starts
                if any(f in train_text[s: s + G1.BLOCK + 1] for f in G1.HOSTS)),
            "junctions_covered": jc_neutral,
        },
        "budget_identical_to_e176n": bool(
            anchor_neutral.shape == (16, G1.BLOCK)),
    }
    G_ANCHOR["pass"] = bool(
        G_ANCHOR["neutral_bank"]["windows_with_host_content"] == 0
        and G_ANCHOR["neutral_bank"]["junctions_covered"] == 0
        and G_ANCHOR["budget_identical_to_e176n"])
    assert G_ANCHOR["pass"], f"anchor gate FAILED: {G_ANCHOR}"
    log(f"G_ANCHOR: neutral bank 16x{G1.BLOCK} (seed {G1.E170_ANCHOR_SEED}, "
        f"{rejections} rejections/{tries} tries) — host 0/16, junctions "
        f"0/16: PASS")

    # ---------------- batteries (e119/e176n verbatim) --------------------
    bat_ids, held_ids = {}, {}
    for j in G1.GEOS:
        cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
        hs = [train_text[p - G1.PRE - j: p] for p, _ in held_occ]
        held_ids[j] = torch.stack([corpus.encode(c) for c in hs])
    r_eval_x, r_eval_y = G1.val_windows(val_ids, val_text, 60, G1.R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    gm12_ids = bat_ids[-12]
    g0_ids = bat_ids[0]

    def measure(sd: dict, tag: str) -> dict:
        """g1bS6's measure() LEAN dial (batteries + held + CE_R + site read)
        on evl_load — CPU-only, offline (no cap)."""
        net = G1.evl_load(sd)
        out: dict = {"tag": tag}
        out["base"] = {j: G1.battery_cell(net, bat_ids[j], zid)
                       for j in G1.GEOS}
        out["base_held"] = {j: G1.battery_cell(net, held_ids[j], zid)
                            for j in G1.GEOS}
        out["ce_r"] = G1.ce_fixed_cpu(net, *r_eval_xy)
        log(f"[{tag}] base: " + " ".join(
            f"g{j:+d} {out['base'][j]['mean_pz']:.4f}" for j in G1.GEOS)
            + " | held30: " + " ".join(
                f"g{j:+d} {out['base_held'][j]['mean_pz']:.4f}"
                for j in G1.GEOS)
            + f" | CE_R {out['ce_r']:.4f}")
        out["site_read"] = G1.read_fact_at(net, pool_x, name_ids, zid,
                                           G1.SITE_ADDR_ROW, G1.SITE_Z_XCOL)
        log(f"[{tag}] site read @183: onset "
            f"{out['site_read']['pz_onset_mean']:.4f} span "
            f"{out['site_read']['pname_mean_over7']:.4f}")
        del net
        return out

    def flat_cells(m: dict) -> dict:
        return {"gm12": m["base"][-12]["mean_pz"],
                "g0": m["base"][0]["mean_pz"],
                "gp12": m["base"][12]["mean_pz"],
                "held30_gm12": m["base_held"][-12]["mean_pz"],
                "held30_g0": m["base_held"][0]["mean_pz"],
                "ce_r": m["ce_r"],
                "site_read_onset": m["site_read"]["pz_onset_mean"],
                "site_read_span": m["site_read"]["pname_mean_over7"]}

    # =====================================================================
    # G-CONFIG: the 10M band + the R-convention assertion (Rule 12)
    # =====================================================================
    torch.manual_seed(HOST_SEED)
    host_net = TinyGPT(G10_CFG)
    n_params = host_net.num_params()
    del host_net
    G_CONFIG = {
        "params": n_params, "band": [8.5e6, 11.5e6],
        "vocab": G10_CFG.vocab, "block_size": G10_CFG.block_size,
        "n_layer": G10_CFG.n_layer, "n_head": G10_CFG.n_head,
        "n_embd": G10_CFG.n_embd,
        "R_rms_ref": R_RMS_REF, "rung_1x_raw": R1_RAW, "rung_1x_rms": R1_RMS,
        "pin_fuzz_rms_carried_raw": PIN_FUZZ_RMS_CARRIED,
        "one_adamw_step_fuzz_raw": ONE_STEP_FUZZ_RAW,
        "assertions": {
            "rung_1x_rms": {"mult": 1.0, "rms": R1_RMS,
                            "expected_rms": R_RMS_REF,
                            "abs_err_rms": abs(R1_RMS - R_RMS_REF),
                            "pass": bool(abs(R1_RMS - R_RMS_REF) < 1e-15)},
            "params_match_lineage": {"expected": G10_PARAMS, "got": n_params,
                                     "pass": bool(n_params == G10_PARAMS)},
        },
        "battery_geometry": {
            "vocab_unchanged": bool(G10_CFG.vocab == corpus.vocab_size == 65),
            "block_unchanged": bool(G10_CFG.block_size == G1.BLOCK == 256),
        },
    }
    G_CONFIG["pass"] = bool(
        8.5e6 <= n_params <= 11.5e6
        and all(a["pass"] for a in G_CONFIG["assertions"].values())
        and G_CONFIG["battery_geometry"]["vocab_unchanged"]
        and G_CONFIG["battery_geometry"]["block_unchanged"])
    assert G_CONFIG["pass"], f"G-CONFIG FAILED: {G_CONFIG}"
    log(f"G-CONFIG: {n_params:,} params in [8.5M, 11.5M]; the 1x rung "
        f"{R1_RAW:.6f} raw = {R1_RMS:.6e} rms (asserted = R_rms_ref); pin "
        f"fuzz rms-carried {PIN_FUZZ_RMS_CARRIED:.3f} raw; one-step "
        f"{ONE_STEP_FUZZ_RAW:.3f} raw: PASS")
    write_partial(rd, "G-CONFIG", {"gates_partial": {"G_CONFIG": G_CONFIG},
                                   "config": {"n_layer": 8, "n_head": 8,
                                              "n_embd": 320,
                                              "block_size": 256,
                                              "params": n_params,
                                              "rung_1x_raw": R1_RAW,
                                              "smoke": SMOKE}})

    # =====================================================================
    # THE LOADED RECORD (g1bS6: the control C + the original W1; never rerun)
    # =====================================================================
    if not G1BS6_METRICS.exists():
        raise SystemExit(
            f"g1bS6 metrics missing: {G1BS6_METRICS} (the loaded control "
            f"and the +1 breach record are this cell's frozen references)")
    g6 = json.loads(G1BS6_METRICS.read_text(encoding="utf-8"))
    lad6 = g6["adjudication"]["ladder"]
    loaded_C = {int(k): v for k, v in lad6["C"]["g_m12"].items()}
    loaded_W1 = {int(k): v for k, v in lad6["W1"]["g_m12"].items()}
    loaded_W1_ce = {t["step"]: t["ce_batch"]
                    for t in g6["arms"]["W1"]["traj"]}
    loaded_C_ce = {t["step"]: t["ce_batch"] for t in g6["arms"]["C"]["traj"]}
    loaded_W1_disp = {t["step"]: t["cum_disp"]
                      for t in g6["arms"]["W1"]["traj"]
                      if "cum_disp" in t and t["step"] in loaded_W1}
    loaded_C_disp = {t["step"]: t["cum_disp"]
                     for t in g6["arms"]["C"]["traj"]
                     if "cum_disp" in t and t["step"] in loaded_C}
    # cross-check the embedded frozen record vs the parsed one (drift guard)
    drift = {"W1": max(abs(loaded_W1[s] - G1BS6_LOADED["W1_g_m12"][s])
                       for s in loaded_W1),
             "C": max(abs(loaded_C[s] - G1BS6_LOADED["C_g_m12"][s])
                      for s in loaded_C),
             "bar": abs(g6["adjudication"]["bar_0p9eq"]
                        - G1BS6_LOADED["bar_0p9eq"])}
    assert drift["W1"] < 1e-9 and drift["C"] < 1e-9 \
        and drift["bar"] < 1e-9, f"loaded-record drift: {drift}"
    G_CTRL_LOADED = {
        "source": REGISTERED["sources_loaded"]["control_C"],
        "dead_at": G1BS6_LOADED["C_dead_at"],
        "gm12_at_1": loaded_C[1], "bar": G1.SHUT_BAR,
        "pass": bool(loaded_C[1] <= G1.SHUT_BAR and g6["adjudication"]
                     ["gates_pass"] is True),
        "note": ("a READ on g1bS6's committed record (C never rerun — the "
                 "dispatch's frozen decision); the kill clock at 10M"),
    }
    log(f"G-CTRL-LOADED: C dead at +{G_CTRL_LOADED['dead_at']} "
        f"(g-12 {loaded_C[1]:.4f} <= {G1.SHUT_BAR}) on g1bS6's committed "
        f"record (verdict {g6['adjudication']['verdict']}, gates "
        f"{g6['adjudication']['gates_pass']}): "
        f"{'PASS' if G_CTRL_LOADED['pass'] else 'FAIL'}")

    # =====================================================================
    # THE PEAK ROOT — LOADED VERBATIM; G-ROOTLOAD (g1bS6's form)
    # =====================================================================
    log("=" * 78)
    log(f"THE PEAK ROOT: LOADED VERBATIM from {SRC_ROOT_CK} (g1bS5's m020 "
        f"formation-curve peak: committed root g-12 "
        f"{G1BS5_PEAK_ROOT['root_gm12']:.4f}; the licensed theta0 of every "
        f"g10 arm) — G-ROOTLOAD identity gate (g1bS6's form)")
    root_ck_path = CKPT_DIR / SRC_ROOT_CK
    if not root_ck_path.exists():
        raise SystemExit(
            f"g10 source root missing: {root_ck_path} (the licensed cell "
            f"LOADS g1bS5's committed peak-root checkpoint)")
    rstate = torch.load(root_ck_path, map_location="cpu", weights_only=False)
    theta0 = {k: v.detach().clone() for k, v in rstate["model"].items()}
    root_meta = dict(rstate.get("meta", {}))
    del rstate

    root_m = measure(theta0, "g10_root")
    root_cells = flat_cells(root_m)

    tol_rootload = 0.005
    _rootload_keys = {"gm12": "root_gm12", "g0": "g0",
                      "held30_gm12": "held30_gm12", "ce_r": "ce_r"}
    deltas_root = {k: abs(root_cells[k] - G1BS5_PEAK_ROOT[ref])
                   for k, ref in _rootload_keys.items()}
    G_ROOTLOAD = {
        "tol": tol_rootload,
        "g1bs5_committed": {"gm12": G1BS5_PEAK_ROOT["root_gm12"],
                            "g0": G1BS5_PEAK_ROOT["g0"],
                            "gp12": G1BS5_PEAK_ROOT["gp12"],
                            "held30_gm12": G1BS5_PEAK_ROOT["held30_gm12"],
                            "held30_g0": G1BS5_PEAK_ROOT["held30_g0"],
                            "ce_r": G1BS5_PEAK_ROOT["ce_r"]},
        "this_run_remeasured": root_cells,
        "abs_deltas": deltas_root,
        "ckpt_meta": root_meta,
        "pass": bool(SMOKE or all(d <= tol_rootload
                                  for d in deltas_root.values())),
        "rationale": ("the loaded checkpoint must BE g1bS5's committed peak "
                      "root (and thereby g1bS6's theta0 — same artifact, "
                      "same committed cells): the root dial is a "
                      "deterministic CPU measure on the same tensors (expect "
                      "~0; 0.005 tolerates float-reduction fuzz); a mismatch "
                      "would mean the wrong artifact or a corrupted load"),
    }
    log(f"G-ROOTLOAD: re-measured root g-12 {root_cells['gm12']:.6f} vs "
        f"g1bS5 committed {G1BS5_PEAK_ROOT['root_gm12']:.6f} "
        f"(|d| {deltas_root['gm12']:.2e}); |d CE_R| {deltas_root['ce_r']:.2e}; "
        f"|d g0| {deltas_root['g0']:.2e}; |d held30| "
        f"{deltas_root['held30_gm12']:.2e} (tol {tol_rootload}): "
        f"{'PASS' if G_ROOTLOAD['pass'] else 'FAIL'}")
    assert G_ROOTLOAD["pass"], f"G-ROOTLOAD FAILED: {G_ROOTLOAD}"

    # ---- THE SCALED BAR, computed before any arm trains ------------------
    bar_0p9eq = root_cells["gm12"] * 0.9 / G1B_ROOT_GM12
    bar_x_g6 = abs(bar_0p9eq - G1BS6_LOADED["bar_0p9eq"])
    SCALED_BAR = {
        "definition": REGISTERED["scaled_bar"],
        "root_gm12_measured": root_cells["gm12"],
        "bar_0p9eq": bar_0p9eq,
        "bar_09x_root": 0.9 * root_cells["gm12"],
        "abs_d_vs_g1bs6_committed": bar_x_g6,
        "computed_before_arms": True,
    }
    assert bar_x_g6 <= 1e-6, (f"scaled bar drifted vs g1bS6's committed "
                              f"{G1BS6_LOADED['bar_0p9eq']}: {bar_0p9eq}")
    log(f"SCALED BAR (frozen formula, root measured): bar_0p9eq = "
        f"{root_cells['gm12']:.4f} x 0.9 / {G1B_ROOT_GM12:.4f} = "
        f"{bar_0p9eq:.4f} (g1bS6 committed "
        f"{G1BS6_LOADED['bar_0p9eq']:.4f}; |d| {bar_x_g6:.2e})")
    write_partial(rd, "root-load+scaled-bar+loaded-record", {
        "G_ROOTLOAD": G_ROOTLOAD, "root_cells": root_cells,
        "scaled_bar": SCALED_BAR, "G_CTRL_LOADED": G_CTRL_LOADED,
        "loaded": {"C_g_m12": loaded_C, "W1_g_m12": loaded_W1,
                   "W1_ce": loaded_W1_ce, "C_ce": loaded_C_ce,
                   "W1_disp": loaded_W1_disp, "C_disp": loaded_C_disp,
                   "record_drift": drift,
                   "g1bs6_verdict": g6["adjudication"]["verdict"]},
        "theta0_ckpt": f"runs/checkpoints/{SRC_ROOT_CK}"})
    CKPT_INVENTORY[f"src_{SRC_ROOT_CK[:-3]}"] = {
        "path": f"runs/checkpoints/{SRC_ROOT_CK}",
        "desc": ("g1bS5's m020 formation-curve PEAK root — theta0 for every "
                 "g10 arm, LOADED VERBATIM (identity-gated by G-ROOTLOAD)"),
        "params": n_params, "root_gm12": root_cells["gm12"]}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =====================================================================
    # THE THREE FIX ARMS (the registered variants; the wash verbatim; the
    # variant hooks are the only deltas)
    # =====================================================================
    VARIANT_DESCS = {
        "F1": ("STEP-CLIP — net committed at theta_0; the FIRST optimizer "
               "step's delta clipped to the rung radius (the wall's own "
               "projection arithmetic, one step early); then the standard "
               "commit-and-project cadence. The wall never sees an "
               "out-of-ball first step."),
        "F2": ("ANCHOR-AT-ONE — one unwalled wash step, THEN commit at "
               "theta_1 (the anchor includes the formation shock) and "
               "project thereafter around theta_1."),
        "F3": ("DELTA-PROJECTION — a trust region on steps: every per-step "
               "delta contracted into the rung ball (cumulative "
               "displacement unbounded); every step in-ball by "
               "construction."),
    }
    net0_by_variant = {
        "F1": None,     # built below (committed at theta_0)
        "F2": None,     # uncommitted CommittedGPT (commits at theta_1)
        "F3": None,     # uncommitted CommittedGPT (no anchor semantics)
    }
    arms: dict = {}
    for variant in VARIANTS:
        log("=" * 78)
        if not SMOKE:
            log(f"[owner envelope] cooldown {COOLDOWN_S:.0f}s before "
                f"{variant} (no back-to-back bursts)")
            cooldown(COOLDOWN_S)
        log(f"FIX ARM {variant} — {VARIANT_DESCS[variant]}")
        if variant == "F1":
            net0 = G1.CommittedGPT(G10_CFG)
            net0.load_state_dict(theta0)
            net0.commit(R1_RAW)
            body, _ = G1.split_anchored_sd(net0.state_dict())
            md = max(float((body[k].float() - theta0[k].float()).abs().max())
                     for k in theta0)
            anch_ok = all(torch.equal(net0._anchor(n), p.detach())
                          for n, p in net0.named_parameters())
            G_BITROOT = {"max_abs_diff": md,
                         "anchors_bit_equal": bool(anch_ok),
                         "n_anchor_tensors": net0._n_anchor_tensors,
                         "pass": bool(md == 0.0 and anch_ok)}
            assert G_BITROOT["pass"], f"F1: wall root != theta0: {G_BITROOT}"
            log(f"G-BITROOT[F1]: max|diff| {md:.1e}, anchors bit-equal, "
                f"R = {R1_RAW:.6f} raw = {R1_RMS:.6e} rms: PASS")
            net0_by_variant["F1"] = net0
        else:
            net0 = G1.CommittedGPT(G10_CFG)
            net0.load_state_dict(theta0)          # UNCOMMITTED (inert wall)
            net0_by_variant[variant] = net0
        try:
            arm = g10_wash(
                variant, net0, anchor_neutral, train_ids, itos, r_eval_xy,
                gm12_ids, g0_ids, zid,
                resume_ck=(CKPT_DIR /
                           (f"smoke_g10_wash_{variant}_resume.pt" if SMOKE
                            else f"g10_wash_{variant}_resume.pt")))
        except OwnerWindowShut as e:
            write_partial(rd, f"owner-window-shut-{variant}",
                          {"stopped_at": variant, "reason": str(e),
                           "arms_partial": {t: {
                               "steps_ran": arms[t]["steps_ran"],
                               "device": arms[t]["device"]}
                               for t in arms},
                           "device_events": device_events})
            log(f"OWNER WINDOW SHUT at {variant}: {e} — stopping honestly")
            raise SystemExit(3)
        G_DRAWFREE = {"zeph_violations": arm["zeph_violations"],
                      "pass": bool(arm["zeph_violations"] == 0)}
        assert G_DRAWFREE["pass"], f"{variant}: name token leaked"
        arm["drawfree_gate"] = G_DRAWFREE
        arms[variant] = arm
        write_partial(rd, f"arm-{variant}", {"arms_partial": {variant: {
            "R_raw": R1_RAW, "R_rms": R1_RMS,
            "steps_ran": arm["steps_ran"], "device": arm["device"],
            "wall_R": arm["wall_R"],
            "g_m12": {t["step"]: t["g_m12_mean_pz"] for t in arm["traj"]
                      if "g_m12_mean_pz" in t},
            "cum_disp": {t["step"]: t["cum_disp"] for t in arm["traj"]
                         if "g_m12_mean_pz" in t},
            "traj": arm["traj"],
            "x_hashes_head": {s: arm["x_hashes"][s]
                              for s in list(arm["x_hashes"])[:2]},
        }}})

        # checkpoints: each arm's final only (g1b/g1bS6's wash-arm convention)
        for s in sorted(arm["sds"]):
            if s == max(arm["sds"]):
                save_ckpt(f"g10_{variant}_s{s}", arm["sds"][s],
                          {"desc": f"g10 {variant} ({VARIANT_DESCS[variant]}"
                                   f") — peak root + {s}-step true-target "
                                   f"neutral wash (R = {R1_RAW} raw, input "
                                   f"seed {WASH_SEED}, lr {G1.FT_LR})",
                           "steps": int(s), "R_raw": R1_RAW, "R_rms": R1_RMS,
                           "input_seed": WASH_SEED, "lr": G1.FT_LR,
                           "base": f"runs/checkpoints/{SRC_ROOT_CK}"})
        del net0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =====================================================================
    # CONSTRUCTION-FIDELITY GATES
    # =====================================================================
    G_INPUTS = {"per_step": {}, "pass": None, "note":
                "the seed-10902 aj/rj draw sequence is shared by ALL "
                "variants — per-step input batches bit-identical (md5-gated)"}
    for step in range(1, CK_WASH[-1] + 1):
        hs = {v: arms[v]["x_hashes"].get(step) for v in VARIANTS}
        same = all(h is not None for h in hs.values()) and \
            len(set(hs.values())) == 1
        G_INPUTS["per_step"][step] = {v: hs[v] for v in VARIANTS}
        G_INPUTS["per_step"][step]["identical"] = bool(same)
    G_INPUTS["pass"] = bool(all(v["identical"]
                                for v in G_INPUTS["per_step"].values()))
    assert G_INPUTS["pass"], "input streams diverged across variants"
    log(f"G-INPUTS: per-step inputs bit-identical across all "
        f"{len(VARIANTS)} variants through +{CK_WASH[-1]}: PASS")

    # F1's +1 body == the wall-projection of the unwalled step-1 body
    # (F2's +1 body is the in-run unwalled reference)
    f1_body, _ = G1.split_anchored_sd(arms["F1"]["sds"][1])
    f2_body, _ = G1.split_anchored_sd(arms["F2"]["sds"][1])
    d_unwalled = float(torch.norm(
        torch.cat([f2_body[k].float().reshape(-1) for k in sorted(f2_body)])
        - torch.cat([theta0[k].float().reshape(-1) for k in sorted(f2_body)])))
    s_expected = min(1.0, R1_RAW / d_unwalled)
    proj_max_diff = max(
        float((f1_body[k].float()
               - (theta0[k].float()
                  + (f2_body[k].float() - theta0[k].float()) * s_expected)
               ).abs().max()) for k in sorted(f2_body))
    G_STEP1F1 = {
        "unwalled_step1_raw": d_unwalled,
        "expected_scale": s_expected,
        "max_abs_diff_F1_vs_projection_of_F2step1": proj_max_diff,
        "tol": 1e-4,
        "pass": bool(proj_max_diff <= 1e-4),
        "rationale": ("the clip IS the projection: F1's step-1-clipped body "
                      "must equal theta_0 + (theta_1_unwalled - theta_0) * "
                      "min(1, R/||...||) — the same arithmetic the wall "
                      "would have applied one forward later"),
    }
    assert G_STEP1F1["pass"], f"G-STEP1F1 FAILED: {G_STEP1F1}"
    log(f"G-STEP1F1: unwalled step-1 delta {d_unwalled:.4f} raw -> clip "
        f"scale {s_expected:.4f}; F1@+1 == projection of the unwalled step "
        f"(max|diff| {proj_max_diff:.1e} <= 1e-4): PASS")

    # F2's +1 read == the loaded C's +1 read (the unwalled step-1 class)
    f2_g1 = next(t["g_m12_mean_pz"] for t in arms["F2"]["traj"]
                 if t["step"] == 1)
    G_STEP1F2 = {"f2_gm12_at_1": f2_g1,
                 "loaded_C_gm12_at_1": loaded_C[1],
                 "abs_d": abs(f2_g1 - loaded_C[1]), "tol": 0.02,
                 "pass": bool(abs(f2_g1 - loaded_C[1]) <= 0.02),
                 "rationale": ("F2's step 1 is the standard unwalled wash "
                               "step (C's step-1 body class); its raw read "
                               "must match the loaded C's committed +1 read "
                               "(behavioral tolerance; CUDA atomics fuzz)")}
    assert G_STEP1F2["pass"], f"G-STEP1F2 FAILED: {G_STEP1F2}"
    log(f"G-STEP1F2: F2@+1 raw read {f2_g1:.4f} vs loaded C@+1 "
        f"{loaded_C[1]:.4f} (|d| {abs(f2_g1 - loaded_C[1]):.2e} <= 0.02): "
        f"PASS")

    # F2's anchors == its own +1 body (bit) and anch__R == R
    f2_anch = arms["F2"]["f2_anchors_sd"] or {}
    key_of = {"anch__" + bk.replace(".", "_"): bk for bk in f2_body}
    anch_pairs = [(ak, key_of[ak]) for ak in f2_anch
                  if ak != "anch__R" and ak in key_of]
    anch_bit = all(torch.equal(f2_anch[ak], f2_body[bk])
                   for ak, bk in anch_pairs)
    G_COMMITAT1 = {
        "n_anchor_tensors": len(anch_pairs),
        "anchors_bit_equal_own_plus1_body": bool(anch_bit),
        "anch_R": float(f2_anch["anch__R"]) if "anch__R" in f2_anch else None,
        "expected_R": R1_RAW,
        "pass": bool(anch_bit and "anch__R" in f2_anch
                     and abs(float(f2_anch["anch__R"]) - R1_RAW) < 1e-9),
        "rationale": ("F2's anchor must BE theta_1 (the post-shock state "
                      "itself), bit-equal, at the 1x radius"),
    }
    assert G_COMMITAT1["pass"], f"G-COMMITAT1 FAILED: {G_COMMITAT1}"
    log(f"G-COMMITAT1: F2's anchors bit-equal its own +1 body "
        f"({len(anch_pairs)} tensors), anch__R == {R1_RAW:.6f}: PASS")

    # G-PIN (F1, F2: displacement vs the arm's OWN anchor) + G-STEPREGION (F3)
    G_PIN = {"per_arm": {}, "primary": "rms_carried",
             "fuzz_rms_carried_raw": PIN_FUZZ_RMS_CARRIED,
             "fuzz_verbatim": G1.PIN_FUZZ_BAR,
             "one_step_fuzz_raw": ONE_STEP_FUZZ_RAW}
    for variant in ("F1", "F2"):
        rows = [t for t in arms[variant]["traj"]
                if t.get("d_anchor") is not None and t["step"] in CK_WASH]
        mx = max(t["d_anchor"] for t in rows) if rows else None
        G_PIN["per_arm"][variant] = {
            "R_raw": R1_RAW, "R_rms": R1_RMS,
            "anchor": ("theta_0" if variant == "F1" else "theta_1 (its own "
                       "+1 body)"),
            "bound_rms_carried": R1_RAW + PIN_FUZZ_RMS_CARRIED,
            "bound_verbatim": R1_RAW + G1.PIN_FUZZ_BAR,
            "max_d_anchor_at_ckpt": mx,
            "per_ckpt_d_anchor": {t["step"]: t["d_anchor"] for t in rows},
            "pass_rms_carried": bool(
                mx is not None and mx <= R1_RAW + PIN_FUZZ_RMS_CARRIED),
            "pass_verbatim": bool(
                mx is not None and mx <= R1_RAW + G1.PIN_FUZZ_BAR),
        }
        log(f"G-PIN[{variant}]: max |d_anchor| {mx:.4f} <= "
            f"{R1_RAW + PIN_FUZZ_RMS_CARRIED:.3f} (rms-carried): "
            f"{'PASS' if G_PIN['per_arm'][variant]['pass_rms_carried'] else 'FAIL'}"
            f" | verbatim R+1.5 = {R1_RAW + G1.PIN_FUZZ_BAR:.2f}: "
            f"{'PASS' if G_PIN['per_arm'][variant]['pass_verbatim'] else 'FAIL'}"
            f" (co-reported)")
    G_PIN["pass"] = bool(all(v["pass_rms_carried"]
                             for v in G_PIN["per_arm"].values()))
    f3_steps = [t["step_disp"] for t in arms["F3"]["traj"]
                if t["step_disp"] is not None]
    G_STEPREGION = {
        "variant": "F3", "R_raw": R1_RAW,
        "max_step_disp": max(f3_steps) if f3_steps else None,
        "tol": R1_RAW + 1e-3,
        "n_steps_over_tol": sum(1 for d in f3_steps if d > R1_RAW + 1e-3),
        "max_cum_disp": max(t["cum_disp"] for t in arms["F3"]["traj"]),
        "note": ("a trust region on steps: every applied per-step delta <= R "
                 "+ float fuzz; cumulative displacement UNBOUNDED by design "
                 "(max reported)"),
        "pass": bool(f3_steps and max(f3_steps) <= R1_RAW + 1e-3),
    }
    assert G_STEPREGION["pass"], f"G-STEPREGION FAILED: {G_STEPREGION}"
    log(f"G-STEPREGION[F3]: max per-step |du| {G_STEPREGION['max_step_disp']:.4f}"
        f" <= R + 1e-3 = {R1_RAW + 1e-3:.4f}; max cumulative |d| "
        f"{G_STEPREGION['max_cum_disp']:.4f} (unbounded by design): PASS")
    write_partial(rd, "construction-gates", {
        "G_INPUTS_pass": G_INPUTS["pass"], "G_BITROOT": G_BITROOT,
        "G_COMMITAT1": G_COMMITAT1, "G_STEP1F1": G_STEP1F1,
        "G_STEP1F2": G_STEP1F2, "G_PIN": G_PIN, "G_STEPREGION": G_STEPREGION,
        "G_CTRL_LOADED": G_CTRL_LOADED})

    # ---- G-EQUIVF1 (read): F1's trace vs the loaded W1's trace ------------
    f1_g = {t["step"]: t["g_m12_mean_pz"] for t in arms["F1"]["traj"]
            if "g_m12_mean_pz" in t}
    shared = sorted(set(f1_g) & set(loaded_W1))
    equiv_rows = {s: {"F1": f1_g[s], "W1_loaded": loaded_W1[s],
                      "delta": f1_g[s] - loaded_W1[s]} for s in shared}
    G_EQUIVF1 = {
        "tol_per_ckpt": 0.05, "shared_ckpts": shared,
        "max_abs_delta": max(abs(r["delta"]) for r in equiv_rows.values()),
        "mean_abs_delta": float(np.mean([abs(r["delta"])
                                         for r in equiv_rows.values()])),
        "rows": equiv_rows,
        "tracks_W1": bool(all(abs(r["delta"]) <= 0.05
                              for r in equiv_rows.values())),
        "note": ("a READ (registered prediction verified empirically): F1's "
                 "step-1 clip produces exactly the state W1's +1 settled "
                 "read measured, and the cadence from forward 2 is "
                 "identical — cross-run CUDA nondeterminism sets the fuzz"),
    }
    log(f"G-EQUIVF1 (read): F1 vs loaded W1 — max |d| "
        f"{G_EQUIVF1['max_abs_delta']:.4f}, mean |d| "
        f"{G_EQUIVF1['mean_abs_delta']:.4f} over {len(shared)} ckpts "
        f"(tol 0.05): tracks={G_EQUIVF1['tracks_W1']}")

    # =====================================================================
    # ADJUDICATION (the frozen bars; GATES -> the fixes vs the bar)
    # =====================================================================
    fmt = lambda v: "n/a" if v is None else f"{v:.4f}"

    def variant_verdict(variant):
        g = {t["step"]: t["g_m12_mean_pz"] for t in arms[variant]["traj"]
             if "g_m12_mean_pz" in t}
        vals = [g[s] for s in CK_WASH if s in g]
        complete = bool(len(vals) == len(CK_WASH))
        return {
            "g_m12": g,
            "all_checkpoints_present": complete,
            "missing": [s for s in CK_WASH if s not in g],
            "min_gm12": min(vals) if vals else None,
            "argmin_step": (min(g, key=lambda s: g[s]) if vals else None),
            "gm12_at_1": g.get(1),
            "holds_0p9eq_bar": bool(complete and vals
                                    and all(v >= bar_0p9eq for v in vals)),
            "first_ck_below_bar": next(
                (s for s in CK_WASH if g.get(s, 1.0) < bar_0p9eq), None),
            "beats_plus1_materially": bool(g.get(1, 0.0) > 0.5),
            "maintains_g1_bar": bool(complete and vals
                                     and all(v >= G1.MAINTAIN_BAR
                                             for v in vals)),
            "retention_min": (min(v / root_cells["gm12"] for v in vals)
                              if vals else None),
            "flat_phase_min": (min([g[s] for s in (10, 50, 100, 200, 300)
                                    if s in g] or [None])),
            "ce_r_at_300": next((t.get("ce_r") for t in arms[variant]["traj"]
                                 if t["step"] == CK_WASH[-1]), None),
            "ce_batch_at_300": next((t["ce_batch"] for t in arms[variant]
                                     ["traj"] if t["step"] == CK_WASH[-1]),
                                    None),
        }

    fixes = {v: variant_verdict(v) for v in VARIANTS}

    gates_pass = bool(G_CONFIG["pass"] and G_ROOTLOAD["pass"]
                      and G_CTRL_LOADED["pass"] and G_BITROOT["pass"]
                      and G_COMMITAT1["pass"] and G_INPUTS["pass"]
                      and G_STEP1F1["pass"] and G_STEP1F2["pass"]
                      and G_PIN["pass"] and G_STEPREGION["pass"]
                      and all(a["drawfree_gate"]["pass"] for a in arms.values()))

    holders = [v for v in VARIANTS if fixes[v]["holds_0p9eq_bar"]]
    partials = [v for v in VARIANTS if fixes[v]["beats_plus1_materially"]]
    stamp = (f"[root {root_cells['gm12']:.4f} (g1bS5 peak, G-ROOTLOAD-passed); "
             f"bar_0p9eq {bar_0p9eq:.4f} = g1bS6's committed scaled bar; "
             f"loaded W1 +1 breach {loaded_W1[1]:.4f}, min over its trace "
             f"{min(loaded_W1.values()):.4f}; loaded C dead at "
             f"+{G_CTRL_LOADED['dead_at']}]")
    if not gates_pass:
        failed = [k for k, g in (("G-CONFIG", G_CONFIG),
                                 ("G-ROOTLOAD", G_ROOTLOAD),
                                 ("G-CTRL-LOADED", G_CTRL_LOADED),
                                 ("G-BITROOT", G_BITROOT),
                                 ("G-COMMITAT1", G_COMMITAT1),
                                 ("G-INPUTS", G_INPUTS),
                                 ("G-STEP1F1", G_STEP1F1),
                                 ("G-STEP1F2", G_STEP1F2),
                                 ("G-PIN", G_PIN),
                                 ("G-STEPREGION", G_STEPREGION))
                  if not g["pass"]]
        verdict = f"TEXTURE (GATE FAILURE: {', '.join(failed)})"
        clause = ("a registered gate failed — nothing adjudicated; failed: "
                  f"{failed} " + stamp)
        FIX_HOLDS_G = FIX_PARTIAL_G = FIX_IMPOTENT_G = None
    else:
        FIX_HOLDS_G = bool(holders)
        FIX_PARTIAL_G = bool(not holders and partials)
        FIX_IMPOTENT_G = bool(not holders and not partials)
        if FIX_HOLDS_G:
            verdict = ("FIX-HOLDS ("
                       + " + ".join(f"{v} ({'STEP-CLIP' if v == 'F1' else ('ANCHOR-AT-ONE' if v == 'F2' else 'DELTA-PROJECTION')})"
                                    for v in holders) + ")")
            clause = (f"the structural blindness is CURABLE: "
                      + " and ".join(
                          f"{v} held the scaled bar ({bar_0p9eq:.4f}) at "
                          f"EVERY checkpoint through +{CK_WASH[-1]} (min "
                          f"{fmt(fixes[v]['min_gm12'])} at "
                          f"+{fixes[v]['argmin_step']}, +1 read "
                          f"{fmt(fixes[v]['gm12_at_1'])})"
                          for v in holders)
                      + f" where the loaded W1 breached at +1 "
                      f"({loaded_W1[1]:.4f}); the wall's 10M continuity "
                      f"restored. " + stamp)
        elif FIX_PARTIAL_G:
            verdict = ("FIX-PARTIAL ("
                       + " + ".join(partials) + ")")
            clause = (f"the blindness is REDUCIBLE, not curable at this "
                      f"dial: "
                      + " and ".join(
                          f"{v}'s +1 read {fmt(fixes[v]['gm12_at_1'])} "
                          f"> 0.5 (vs the loaded W1's {loaded_W1[1]:.4f}) but "
                          f"no variant held every checkpoint (first below "
                          f"bar: "
                          + ", ".join(f"{v} +{fixes[v]['first_ck_below_bar']}"
                                      for v in VARIANTS) + ")"
                          for v in partials)
                      + f"; full texture reported. " + stamp)
        else:
            verdict = "FIX-IMPOTENT"
            clause = (f"all three variants breached like the original: "
                      + " | ".join(
                          f"{v} ({'STEP-CLIP' if v == 'F1' else ('ANCHOR-AT-ONE' if v == 'F2' else 'DELTA-PROJECTION')}): "
                          f"+1 {fmt(fixes[v]['gm12_at_1'])}, min "
                          f"{fmt(fixes[v]['min_gm12'])} at "
                          f"+{fixes[v]['argmin_step']}, first below bar "
                          f"+{fixes[v]['first_ck_below_bar']}"
                          for v in VARIANTS)
                      + f" vs the loaded W1's +1 {loaded_W1[1]:.4f} (its own "
                      f"trace-min) — the +1 kill is NOT about the projection "
                      f"timing; the WALL-FADES verdict stands untouched. "
                      f"Texture: F1-vs-W1 max |d| "
                      f"{G_EQUIVF1['max_abs_delta']:.4f} (the clip reproduces "
                      f"the wall's own trajectory — the registered "
                      f"isomorphism); F3's flat-phase min "
                      f"{fmt(fixes['F3']['flat_phase_min'])} with unbounded "
                      f"cumulative walk (max "
                      f"{G_STEPREGION['max_cum_disp']:.3f} raw); F2 anchored "
                      f"at its own dead +1 state. " + stamp)

    log("=" * 78)
    log(f"G10 VERDICT: {verdict}")
    for v in VARIANTS:
        g = fixes[v]["g_m12"]
        log(f"  {v}: g-12 " + " -> ".join(f"+{s}:{val:.4f}"
                                          for s, val in sorted(g.items())))
    log(f"  loaded W1:  " + " -> ".join(f"+{s}:{val:.4f}" for s, val
                                        in sorted(loaded_W1.items())))
    log(f"  loaded C:   " + " -> ".join(f"+{s}:{val:.4f}" for s, val
                                        in sorted(loaded_C.items())))
    log(f"  bar_0p9eq {bar_0p9eq:.4f} | +1 reads: "
        + " ".join(f"{v} {fmt(fixes[v]['gm12_at_1'])}" for v in VARIANTS)
        + f" | W1 {loaded_W1[1]:.4f} | C {loaded_C[1]:.4f}")
    log(f"  {clause}")
    log("=" * 78)

    # =====================================================================
    # OUTPUTS
    # =====================================================================
    loaded_table = {
        "bar_0p9eq": bar_0p9eq,
        "steps": list(CK_WASH),
        "loaded_C": {s: loaded_C.get(s) for s in CK_WASH},
        "loaded_W1": {s: loaded_W1.get(s) for s in CK_WASH},
        **{v: {s: fixes[v]["g_m12"].get(s) for s in CK_WASH}
           for v in VARIANTS},
    }
    metrics = {
        "experiment": "g10_first_step_wall",
        "date": common.now_iso(),
        "partial": False,
        "progressive_writes": _progressive["n"],
        "phases": list(_progressive["phases"]),
        "design": ("the g10 dispatch (T170's named structural fix; bars "
                   "verbatim; fix arithmetic registered BEFORE compute; no "
                   "bar shopping) — the wash/projection conventions verbatim "
                   "from g1bS6"),
        "question": REGISTERED["question"],
        "registered": REGISTERED,
        "provenance": {
            "host": {"config": {"n_layer": 8, "n_head": 8, "n_embd": 320,
                                "block_size": 256, "vocab": 65},
                     "params": n_params, "band": "[8.5M, 11.5M]",
                     "host_seed": HOST_SEED, "corpus_seed": 1337,
                     "config_provenance": "e005s LARGE's 10M-class config "
                                          "(the lineage's standing host)",
                     "base_install": REGISTERED["sources_loaded"][
                         "base_install"]},
            "root": {"source": REGISTERED["sources_loaded"]["peak_root"],
                     "movement_rms": G1BS5_PEAK_ROOT["movement_rms"],
                     "steps": G1BS5_PEAK_ROOT["steps"],
                     "cons_seed": G1BS5_PEAK_ROOT["cons_seed"],
                     "root_cells": root_cells},
            "wash": {"recipe": REGISTERED["wash"]["recipe"],
                     "seed": WASH_SEED, "ckpt_steps": list(CK_WASH),
                     "devices": {v: arms[v]["device"] for v in VARIANTS},
                     "n_chunks": {v: arms[v].get("n_chunks") for v in VARIANTS},
                     "chunk_tables": {v: arms[v].get("chunk_table")
                                      for v in VARIANTS},
                     "runtimes_s": {v: round(arms[v]["traj"][-1]["elapsed_s"], 1)
                                    for v in VARIANTS}},
            "loaded_g1bs6": {"C": REGISTERED["sources_loaded"]["control_C"],
                             "W1": REGISTERED["sources_loaded"]["original_W1"],
                             "verdict": g6["adjudication"]["verdict"],
                             "gates_pass": g6["adjudication"]["gates_pass"],
                             "bar_0p9eq": g6["adjudication"]["bar_0p9eq"],
                             "D_kill_raw": g6["adjudication"]["D_kill_raw"]},
            "R_convention": REGISTERED["R_convention"],
            "owner_envelope": {
                "launch_gate": REGISTERED["owner_envelope"]["launch_gate"],
                "burst_cap_s": TRAIN_CAP_GPU, "cooldown_s": COOLDOWN_S,
                "events": device_events},
            "seeds": {"host_init_base_corpus": HOST_SEED,
                      "wash": WASH_SEED, "protocol_corpus": 1337},
        },
        "scaled_bar": SCALED_BAR,
        "loaded_control": {
            "table": loaded_table,
            "W1_breach_record": {
                "gm12_at_1": loaded_W1[1],
                "min_over_trace": min(loaded_W1.values()),
                "first_ck_below_bar": 1,
                "cum_disp_at_1": loaded_W1_disp.get(1),
                "note": G1BS6_LOADED["note"]},
            "C_kill_record": {"dead_at": G_CTRL_LOADED["dead_at"],
                              "gm12_at_1": loaded_C[1],
                              "D_kill_raw": g6["adjudication"]["D_kill_raw"]},
        },
        "fixes": {
            v: {"desc": VARIANT_DESCS[v], "R_raw": R1_RAW, "R_rms": R1_RMS,
                "ckpt_steps": list(CK_WASH),
                "steps_ran": arms[v]["steps_ran"],
                "device": arms[v]["device"],
                "n_chunks": arms[v].get("n_chunks"),
                "chunk_table": arms[v].get("chunk_table"),
                "traj": arms[v]["traj"],
                "missing_checkpoints": [s for s in CK_WASH
                                        if s not in arms[v]["sds"]],
                "verdict": fixes[v]}
            for v in VARIANTS},
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": G1.PRE, "post_cap":
                     G1.POST_CAP, "neutral_bank": G_ANCHOR["neutral_bank"],
                     "measure_dial": "g1bS6's measure() LEAN dial on evl_load, "
                                     "CPU-only; ckpt light reads via the "
                                     "ARMED eval twin (F1/F2 settled, F3 raw "
                                     "- registered conventions)"},
        "gates": {"G_CONFIG": G_CONFIG, "G_ROOTLOAD": G_ROOTLOAD,
                  "G_CTRL_LOADED": G_CTRL_LOADED, "G_BITROOT": G_BITROOT,
                  "G_COMMITAT1": G_COMMITAT1,
                  "G_INPUTS": {k: v for k, v in G_INPUTS.items()
                               if k != "per_step"} | {"per_step": {
                                   s: v["identical"] for s, v in
                                   G_INPUTS["per_step"].items()}},
                  "G_STEP1F1": G_STEP1F1, "G_STEP1F2": G_STEP1F2,
                  "G_PIN": G_PIN, "G_STEPREGION": G_STEPREGION,
                  "G_EQUIVF1_read_NOT_A_GATE": G_EQUIVF1,
                  "G_DRAWFREE": {v: arms[v]["drawfree_gate"]
                                 for v in VARIANTS}},
        "adjudication": {
            "bars_verbatim": REGISTERED["bars_verbatim"],
            "order": "GATES -> the three fixes vs the scaled bar at every "
                     "checkpoint -> FIX-HOLDS > FIX-PARTIAL > FIX-IMPOTENT",
            "gates_pass": gates_pass,
            "bar_0p9eq": bar_0p9eq,
            "fixes": fixes,
            "G_EQUIVF1": G_EQUIVF1,
            "FIX_HOLDS": (FIX_HOLDS_G if gates_pass else None),
            "FIX_PARTIAL": (FIX_PARTIAL_G if gates_pass else None),
            "FIX_IMPOTENT": (FIX_IMPOTENT_G if gates_pass else None),
            "holders": (holders if gates_pass else None),
            "material_plus1": (partials if gates_pass else None),
            "verdict": verdict, "clause": clause,
        },
        "honesty_reflex": {
            "n1_scope": ("n=1 root (loaded, identity-gated), one wash seed "
                         "(10902), each variant n=1 — the rung is the FIX "
                         "FORM, not seeds; replicate ladders follow only if "
                         "a bar fires"),
            "loaded_not_rerun": ("C and W1 are g1bS6's committed numbers, "
                                 "parsed (drift-guarded to 1e-12) — every "
                                 "cross-run comparison (G-STEP1F2, "
                                 "G-EQUIVF1) carries CUDA-atomics "
                                 "nondeterminism fuzz; the behavioral "
                                 "tolerances (0.02 / 0.05) bound it"),
            "intervention_not_logits": ("the fix IS the intervention: all "
                "three variants share bit-identical step-0 weights and "
                "bit-identical per-step inputs (md5-gated, G-INPUTS); the "
                "ONLY deltas are the registered first-step mechanics "
                "(G-STEP1F1 pins the clip to the wall's own projection "
                "arithmetic; G-COMMITAT1 pins the anchor to theta_1; "
                "G-STEPREGION pins every F3 step in-ball)"),
            "read_conventions": ("F1/F2 read settled (the armed twin, "
                                 "g1bS6's convention); F3 reads raw — each "
                                 "variant's own defined semantics, "
                                 "registered before compute; the loaded W1 "
                                 "reads are settled, C's raw (g1bS6's "
                                 "conventions)"),
            "f1_isomorphism": ("the registered prediction that F1 "
                               "reproduces W1's loaded trace is a MECHANICAL "
                               "claim (the clip = the projection the wall "
                               "applies one forward later; identical "
                               "optimizer state), verified empirically by "
                               "G-EQUIVF1 — if it holds, F1's breach is not "
                               "a new fact about the wall but the same fact "
                               "seen from inside the first step"),
            "no_bar_shopping": ("the three bars and their precedence were "
                                "frozen in the dispatch; the +1 materiality "
                                "threshold (0.5) and the every-checkpoint "
                                "form are used exactly as registered"),
        },
        "trims": trims, "deviations": deviations,
        "device_events": device_events,
        "ckpt_inventory": CKPT_INVENTORY,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 8, "n_head": 8, "n_embd": 320,
                   "block_size": 256, "params": n_params,
                   "rung_1x_raw": R1_RAW, "rung_1x_rms": R1_RMS,
                   "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "first_step_wall.png", loaded_table, fixes, arms, verdict,
         clause, gates_pass, bar_0p9eq, root_cells, loaded_W1, loaded_C,
         G_EQUIVF1, G_STEPREGION)

    log(f"outputs: {rd / 'metrics.json'}, {rd / 'first_step_wall.png'}, "
        f"{len(CKPT_INVENTORY)} checkpoints in runs/checkpoints/g10_*.pt")
    log(f"VERDICT: {verdict}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot
# THE FIRST-STEP WALL figure: the headline (five traces vs the bar), the
# early breach zoom, the F1-vs-W1 equivalence panel, displacement vs the
# rung (F3's walk), F3's per-step trust region, the verdict panel.

def plot(path, loaded_table, fixes, arms, verdict, clause, gates_pass,
         bar_0p9eq, root_cells, loaded_W1, loaded_C, G_EQUIVF1,
         G_STEPREGION):
    import textwrap
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.5))
    cols = {"C": "crimson", "W1": "black", "F1": "seagreen",
            "F2": "royalblue", "F3": "darkorange"}
    lbls = {"C": "C — loaded control (g1bS6, no wall)",
            "W1": "W1 — loaded original wall (g1bS6, 1x)",
            "F1": "F1 — STEP-CLIP (this run)",
            "F2": "F2 — ANCHOR-AT-ONE (this run)",
            "F3": "F3 — DELTA-PROJECTION (this run)"}
    marks = {"C": "o", "W1": "x", "F1": "s", "F2": "^", "F3": "v"}

    def series(tag):
        key = {"C": "loaded_C", "W1": "loaded_W1"}.get(tag, tag)
        pts = [(0, root_cells["gm12"])] + sorted(loaded_table[key].items())
        return [p[0] for p in pts], [p[1] for p in pts]

    # (0,0) THE HEADLINE: g-12 vs wash steps — five traces vs the bar
    ax = axes[0, 0]
    for tag in ("C", "W1", "F1", "F2", "F3"):
        xs, ys = series(tag)
        ax.plot(xs, ys, marks[tag] + "-" if tag != "W1" else "x--",
                ms=8, lw=2.2, color=cols[tag],
                alpha=0.55 if tag in ("C", "W1") else 0.95,
                label=lbls[tag], mew=2 if tag == "W1" else 1)
    ax.axhline(bar_0p9eq, ls="--", lw=1.6, color="black", alpha=0.85,
               label=f"0.9eq bar {bar_0p9eq:.3f}")
    ax.axhline(0.9 * root_cells["gm12"], ls="-.", lw=1.1, color="gray",
               alpha=0.8, label=f"0.9 x root {0.9 * root_cells['gm12']:.3f}")
    for yv, col in ((G1.MAINTAIN_BAR, "seagreen"), (G1.SHUT_BAR,
                                                    "tab:purple")):
        ax.axhline(yv, ls=":", lw=1.0, color=col, alpha=0.7)
    ax.annotate(f"root {root_cells['gm12']:.3f} (loaded peak)",
                (0, root_cells["gm12"]), textcoords="offset points",
                xytext=(6, 4), fontsize=7.5)
    ax.set_xlabel("neutral-wash steps from the loaded peak root")
    ax.set_ylabel("g-12 (mean p(Z), install-60 battery)")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7.5, loc="center right")
    ax.set_title("THE FIRST-STEP-AWARE WALL — three fixes vs the +1 breach",
                 fontsize=10)

    # (0,1) THE EARLY ZOOM: the breach structure at +1..+10
    ax = axes[0, 1]
    for tag in ("C", "W1", "F1", "F2", "F3"):
        xs, ys = series(tag)
        ax.plot(xs, ys, marks[tag] + "-" if tag != "W1" else "x--",
                ms=9, lw=2.0, color=cols[tag],
                alpha=0.55 if tag in ("C", "W1") else 0.95,
                label=lbls[tag], mew=2 if tag == "W1" else 1)
    ax.axhline(bar_0p9eq, ls="--", lw=1.6, color="black", alpha=0.85,
               label=f"0.9eq bar {bar_0p9eq:.3f}")
    ax.axhline(0.5, ls=":", lw=1.3, color="tab:purple", alpha=0.8,
               label="FIX-PARTIAL materiality (0.5)")
    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("wash step (zoom)")
    ax.set_ylabel("g-12")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("THE BREACH ZOOM (+1..+10): where the fixes land", fontsize=10)

    # (0,2) THE EQUIVALENCE PANEL: F1 - loaded W1 per checkpoint
    ax = axes[0, 2]
    rows = G_EQUIVF1["rows"]
    ss = sorted(int(s) for s in rows)
    ax.plot(ss, [rows[s]["delta"] for s in ss], "s-", ms=8, lw=2.0,
            color="seagreen", label="F1 - W1(loaded)")
    ax.axhline(0.0, ls="-", lw=1.2, color="black", alpha=0.8)
    ax.axhspan(-0.05, 0.05, color="seagreen", alpha=0.12,
               label="equivalence tol 0.05")
    for v, col in (("F2", "royalblue"), ("F3", "darkorange")):
        g = {int(k): val for k, val in fixes[v]["g_m12"].items()}
        ax.plot(sorted(g), [g[s] - loaded_W1[s] for s in sorted(g)
                            if s in loaded_W1], "^-", ms=6, lw=1.5,
                color=col, alpha=0.85, label=f"{v} - W1(loaded)")
    ax.set_xlabel("wash step")
    ax.set_ylabel("g-12 delta vs the loaded original W1")
    ax.legend(fontsize=7.5)
    ax.set_title("THE ISOMORPHISM READ: each fix vs the original breach",
                 fontsize=10)

    # (1,0) DISPLACEMENT vs theta_0 (raw) + the rung
    ax = axes[1, 0]
    for v, col in (("F1", "seagreen"), ("F2", "royalblue"),
                   ("F3", "darkorange")):
        rows_d = [t for t in arms[v]["traj"] if "g_m12_mean_pz" in t]
        ax.plot([r["step"] for r in rows_d], [r["cum_disp"] for r in rows_d],
                marks[v] + "-", ms=6, lw=1.8, color=col, alpha=0.9,
                label=f"{v} ||theta_t - theta_0||")
    ax.axhline(R1_RAW, ls="--", lw=1.4, color="seagreen", alpha=0.7,
               label=f"the 1x rung R = {R1_RAW:.3f} raw")
    ax.axhline(R1_RAW + PIN_FUZZ_RMS_CARRIED, ls=":", lw=1.0,
               color="seagreen", alpha=0.5,
               label=f"R + rms-carried fuzz ({PIN_FUZZ_RMS_CARRIED:.2f})")
    ax.axhline(ONE_STEP_FUZZ_RAW, ls=":", lw=1.8, color="k", alpha=0.8)
    ax.annotate(f"one AdamW step = {ONE_STEP_FUZZ_RAW:.2f} raw "
                f"(the structural excess)", (0.99, ONE_STEP_FUZZ_RAW),
                xycoords=("axes fraction", "data"), ha="right",
                fontsize=7.5)
    ax.set_xlabel("wash step")
    ax.set_ylabel(r"raw $\|\theta_t-\theta_0\|_2$ at checkpoints")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("DISPLACEMENT FROM THE ROOT (F3's walk is unbounded)",
                 fontsize=10)

    # (1,1) F3's TRUST REGION: per-step deltas vs R
    ax = axes[1, 1]
    f3 = [t for t in arms["F3"]["traj"]]
    ax.plot([t["step"] for t in f3], [t["step_disp"] for t in f3],
            "-", lw=1.2, color="darkorange", alpha=0.85,
            label="F3 applied per-step |du|")
    raws = [(t["step"], t["step_disp_raw_opt"]) for t in f3
            if t.get("step_disp_raw_opt") is not None]
    if raws:
        ax.plot([r[0] for r in raws], [r[1] for r in raws], ":", lw=1.0,
                color="gray", alpha=0.9,
                label="pre-clip |du| (the raw optimizer step)")
    ax.axhline(R1_RAW, ls="--", lw=1.4, color="seagreen", alpha=0.8,
               label=f"R = {R1_RAW:.3f} raw")
    f1r = [t for t in arms["F1"]["traj"] if t["step"] <= 12]
    ax.plot([t["step"] for t in f1r], [t["step_disp"] for t in f1r],
            "s-", ms=4, lw=1.0, color="seagreen", alpha=0.7,
            label="F1 per-step |du| (step 1 clipped)")
    ax.set_xlabel("wash step")
    ax.set_ylabel("per-step raw displacement")
    ax.set_xlim(-2, 60)
    ax.legend(fontsize=7.5)
    ax.set_title(f"F3's TRUST REGION (every step in-ball; max cum "
                 f"{G_STEPREGION['max_cum_disp']:.2f} raw)", fontsize=10)

    # (1,2) THE VERDICT PANEL
    ax = axes[1, 2]
    ax.axis("off")
    y = 0.97
    ax.text(0.02, y, "G10 — THE FIRST-STEP-AWARE WALL: F1 STEP-CLIP / F2 "
            "ANCHOR-AT-ONE / F3 DELTA-PROJECTION", fontsize=9.5,
            va="top", family="monospace", weight="bold")
    y -= 0.048
    ax.text(0.02, y, f"host {G10_PARAMS:,} | theta0 = {SRC_ROOT_CK} (loaded, "
            f"G-ROOTLOAD) | bar_0p9eq {bar_0p9eq:.4f}",
            fontsize=7.2, va="top", family="monospace")
    y -= 0.030
    hdr = (f"  step:      " + " ".join(f"+{s:<7d}" for s in
                                       sorted(loaded_W1)))
    ax.text(0.02, y, hdr, fontsize=6.4, va="top", family="monospace")
    y -= 0.027
    for tag in ("C", "W1", "F1", "F2", "F3"):
        g = loaded_table[{"C": "loaded_C", "W1": "loaded_W1"}.get(tag, tag)]
        seq = " ".join(f"{(fmt2(g[s]) if g.get(s) is not None else 'n/a'):<8}"
                       for s in sorted(loaded_W1))
        ax.text(0.02, y, f"  {tag} ({lbls[tag].split('—')[0].strip()}): "
                f"{seq}", fontsize=6.4, va="top", family="monospace",
                color=cols[tag])
        y -= 0.026
    y -= 0.008
    ax.text(0.02, y, f"  gates {'ALL PASS' if gates_pass else 'FAILURE'} | "
            f"holds0.9eq: "
            + ", ".join(f"{v}={fixes[v]['holds_0p9eq_bar']}" for v in
                        VARIANTS)
            + f" | F1-vs-W1 max|d| {G_EQUIVF1['max_abs_delta']:.4f}",
            fontsize=7.0, va="top", family="monospace")
    y -= 0.040
    ax.text(0.02, y, f"VERDICT: {verdict}", fontsize=9.5, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.038
    for wd in textwrap.wrap(clause, width=88, break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=6.6, va="top", family="monospace")
        y -= 0.025

    fig.suptitle("G10 — THE FIRST-STEP-AWARE WALL (T170's named fix): the "
                 "three registered variants at the 1x rung on the loaded "
                 "peak root vs the loaded +1 breach -> " + verdict,
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fmt2(v):
    return f"{v:.4f}"


if __name__ == "__main__":
    sys.exit(main())
