"""G1BS6 — THE WALL AT 10x, TAKE 5: THE ARMS ON THE PEAK ROOT (T167's
follow-on; the coordinator's ninth-dispatch lineage, 2026-10-02). THE ACTUAL
SCALE ADJUDICATION.

==================== THE QUESTION (the design's, finally in reach) =========
Does the commit-and-project L2 ball hold a consolidated fact through a wash
that kills the control, at ~10x the parameters? Four takes bought the
prerequisite: a live consolidated root. g1bS5's formation curve (T167,
SHARP-OPTIMUM) located the 10M formation optimum IN THE INTERIOR — 0.20 rms
(s500 @ 4e-4), root g-12 0.7677 — and its registered clause named THIS cell:
"a fifth take at the peak movement would finally adjudicate the wall."

==================== THE REGISTERED DEVIATION ON THE GATE ===================
THE PEAK ROOT SITS 0.012 BELOW THE 0.78 EXPRESS BAR (0.7677 < 0.78).
REGISTERED PRE-RUN DECISION (frozen in the dispatch BEFORE this script ran;
documented here verbatim): THE ARMS RUN — this is the fifth take, not a gate
stop. The wall bars adjudicate against THE SCALED-BAR DEFINITION (the
root-strength-relative form, computed from THIS root's measured gm12), with
the sub-bar root disclosed on every artifact. THE DEVIATION'S JUSTIFICATION:
the express gate's purpose (g1bS2/g1bS3's own failure_action semantics) was
preventing arms on a DEAD channel — 0.0010-class roots where the wall assay
measures noise; 0.7677 is a LIVE channel 1.6% under the bar, and g1bS5's
sweep licenses the arms (the curve's peak, CE_R healthy 1.697, held30 0.557
— the strongest root the 10M lineage has produced). G-ROOT is therefore a
READ recorded on every artifact, NOT a member of the adjudicating gate set
(G-ROOTLOAD — the loaded checkpoint's identity vs g1bS5's committed record —
replaces it as the load gate). No other gate, dial, bar or stream is touched.

=============== THE FROZEN CONVENTION (design, verbatim) ================
The R-dial travels in PER-COORDINATE RMS units, not raw L2.
R_rms = 0.7/sqrt(2.74e6) = 4.2301e-4. The ladder: {1x, 2x, 4x} x R_rms
(dimension-matched to g1's {1,2,6}x shape). All readouts report BOTH
conventions (R_rms and raw L2). THE WASH IS UNTOUCHED (the e1xx recipe
verbatim — the wash is the treatment, not the dial).

=============== THE REGISTERED BARS (scratch/g1bS_design.md, verbatim) ======
  WALL-SCALES: "fires if at R_rms_match the fact's battery-channel p(Z)
      stays >= 0.9-equivalent (scaled to this host's root strength) at every
      checkpoint through +300 while C dies by +50."
  WALL-FADES: "fires if C dies but no R on the ladder holds the bar (or only
      the tightest R holds it while freezing CE)."
  WALL-TIGHTENS: "fires if a LOOSER R than rms-match holds (the basin grew
      with dimension) — co-report the CE tax curve at scale (the +0.53
      reference re-priced)."
No bar shopping: the ladder is fixed; if no rung holds, WALL-FADES is the
verdict, not a dial search.

====== THE SCALED-BAR DEFINITION (g1bS2/3/4's committed convention,
====== computed BEFORE any arm trains, from the frozen prior + this run's
====== own measured root):
  g1b_root_gm12 := 0.9155886769294739          (g1b's stored root, frozen)
  root_gm12     := THIS host's loaded peak-root g-12 (measured at the root
                   dial; cross-checked vs g1bS5's committed 0.7677)
  bar_0p9eq     := root_gm12 * 0.9 / g1b_root_gm12   (= 0.982955 x root)
  "R holds the bar" := g-12 >= bar_0p9eq at EVERY checkpoint
      {1,2,4,10,50,100,200,300}.
  "freezing CE" (the WALL-FADES escape clause) := W1's in-batch corpus CE at
      +300 >= the root's CE_R. Co-reported: dCE(W1,C)@300 vs the +0.53 2.74M
      reference.
  CO-REPORTED SECONDARY BARS (context only, never adjudicated as primary):
  g1-verbatim maintains (>= 0.50 at every ckpt) / dies (<= 0.27 by +50);
  post-transient flat-phase retention := min g-12 over {10,50,100,200,300}
  >= 0.9 x root_gm12.

THE CELL (g1b's conventions verbatim in form; the root is LOADED, not
trained — g1bS5's committed peak artifact):
  HOST   the same 10M TinyGPT char-LM family (8L/8H/320d block 256 =
         9,977,600 params); base+install = g1bS2's PASSED ckpts, LOADED
         read-only (identity gates re-derived on the load; takes 3-5's
         standing license: never retrained).
  ROOT   runs/checkpoints/g1bS5_root_m020.pt — g1bS5's m020 peak root
         (base+install LOADED + e113 jitter consolidation s500 @ 4e-4, seed
         10901, movement 0.20 rms) LOADED VERBATIM; the full root dial
         re-measured THIS run and cross-checked against g1bS5's committed
         root_cells (G-ROOTLOAD).
  WALL   commit(theta_anchor) at the RMS ladder, then the UNTOUCHED wash
         (e176N arm A verbatim: neutral bank seed 170, batch 32 = 16 neutral
         + 16 random, full-token CE, AdamW (0.9,0.95) wd 0.1 constant lr
         1e-3 clip 1.0, seed 10902, 300 steps, ckpts {1,2,4,10,50,100,200,300}):
         C uncommitted / W1 = 1x R_rms / W2 = 2x / W3 = 4x. Checkpoint
         cadence, fact battery, CE reads VERBATIM from g1b's conventions.

ARMS GUARANTEE (Rule 12 pre-dispatch): all four arms share bit-identical
step-0 weights (G-BITROOT), bit-identical per-step input streams (md5-gated,
G-INPUTS through +300) and identical targets; the ONLY delta is the commit
radius. C guarantees the kill clock at 10M (G-CTRL); the W arms guarantee a
G-PIN-verified ball at their radius.

GATES (any failure => TEXTURE, nothing adjudicated — G-ROOT IS NOT IN THIS
SET per the registered deviation above):
  G-CONFIG  params in [8.5M, 11.5M]; vocab 65; block 256; the R-convention
            assertion (|R_i_raw/sqrt(P) - k_i x R_rms_ref| < 1e-12).
  G-BASE / G-BASE-QUAL / G-INST  the loaded-lineage identity gates, re-derived
            on the load (g1bS4/g1bS5's standing convention).
  G-ROOTLOAD the loaded peak root's cells == g1bS5's committed record
            (|d gm12|, |d ce_r|, |d g0|, |d held30| <= 0.005; the root dial
            is deterministic CPU on the same tensor — expect ~0).
  G-CTRL    arm C dies by +50 (g-12 <= 0.27, verbatim).
  G-PIN     every wall arm's raw displacement <= R_raw + pin_fuzz_rms-carried
            (primary; the frozen 1.5 fuzz constant carried in the SAME rms
            convention as the dial — g1bS4's committed convention).
  G-BITROOT / G-INPUTS / G-STEP1  verbatim (wall roots bit-identical to
            theta0; per-step inputs md5-identical across all four arms;
            wall arms' step-1 bodies == C's step-1 body within 1e-4).

==================== THE OWNER ENVELOPE (STATE.json compute_directive) ======
The tightest constraint (2026-10-02, permanent): the lab is the LOWEST
compute priority. (1) GPU util AND temperature checked BEFORE every launch
(nvidia-smi); launch only when util <= 20% AND temp <= 70C (double-poll,
5 s apart; plus mem <= 60% resident-neighbor caution); (2) SHORT bursts
<= 90 s GPU (TRAIN_CAP_GPU = 90 — the wash arms are split at the cap with
full-state resume ckpts); (3) cooldown >= 180 s between bursts (COOLDOWN_S
= 180; the CPU full dials add further natural cooling on top); (4) NO
back-to-back; (5) when in doubt WAIT (the gate parks in 30 s polls; after
GPU_WAIT_MAX it STOPS honestly — partial metrics + resume ckpts hold the
record; NEVER a silent CPU hop). Outside load mid-burst => PAUSE-AND-WAIT
(the g1bW policy), never migrate. torch threads 4 (shared machine).

==================== LINEAGE (the standing headers, abridged) ===============
G1BS5 — the formation curve (T165/T167): consolidation-only dose sweep; the
  optimum IN THE INTERIOR at 0.20 rms (0.7677) — THIS cell's root.
G1BS4 — the movement-matched dose (T161): 0.30 rms -> 0.2523; the ladder's
  recorded texture at a WEAK root (W1 +1 shock then recovery above root; W3
  late fade) — context priors, never adjudicated against.
G1BS3 — the width-scaled e113 license (T159): 4e-4; 0.12 rms -> 0.6498.
G1BS2 — the val-min-anchored base + the 1e-3 consolidation casualty.
Builds on: g1/g1b/g1bR (the commit-and-project wall; WALL-REPLICATES n=3
seeds; WALL-TAXES +0.53), e182 (the pretrained-LM lane). What is NEW: the
wall's first ADJUDICATION at 10M — the first take where the arms carry a
live, curve-licensed root.

Outputs: runs/g1bS6/{metrics.json, scale_wall.png}; checkpoints
runs/checkpoints/g1bS6_*.pt (all prior takes' artifacts untouched; the
base+install are g1bS2's and the root is g1bS5's, LOADED read-only).
No NOTES/THINKING/QUEUE/STATE edits (the coordinator folds). Progressive
metrics writes after every phase (the outage lesson); commit + push per
phase.

Run:  cd lab && python g1bS6_peak_wall.py    (G1BS6_SMOKE=1 shakedown)
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

SMOKE = os.environ.get("G1BS6_SMOKE") == "1"
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
# THE CONFIG (the same 10M host; the arms machinery; the owner envelope)
# ======================================================================
G1BS_CFG = Cfg(vocab=65, n_layer=8, n_head=8, n_embd=320, block_size=256)
G1BS_PARAMS = 9_977_600          # e005s LARGE's verified 10M-class config
HOST_SEED = 1337                 # the "1337 family" (host init + base corpus)
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
SRC_BASE_CK, SRC_INSTALL_CK = "g1bS2_base.pt", "g1bS2_install.pt"
                                 # the lineage license (takes 3-5): g1bS2's
                                 # PASSED base + movement-matched install,
                                 # LOADED VERBATIM (never retrained;
                                 # read-only sources)
SRC_ROOT_CK = "g1bS5_root_m020.pt"
                                 # THE PEAK ROOT (g1bS5's m020: s500 @ 4e-4 =
                                 # 0.20 rms movement, seed 10901; committed
                                 # root g-12 0.7676599621772766) — LOADED
                                 # VERBATIM; the licensed theta0 of every
                                 # g1bS6 arm
BASE_STEPS = 1200 if not SMOKE else 24
INSTALL_STEPS = 1000 if not SMOKE else 8
                                 # the loaded artifacts' own completed doses
                                 # (identity-gate references only; nothing
                                 # retrains in this cell)

# ---- THE R LADDER (frozen convention): R_rms = 0.7/sqrt(2.74e6) ----------
G1B_PARAMS = 2_739_072
G1B_ROOT_GM12 = 0.9155886769294739          # g1b's stored root (the bar's
                                            # denominator, frozen)
R_RMS_REF = 0.7 / math.sqrt(G1B_PARAMS)     # 4.2301e-4 per-coordinate RMS
R_LADDER_MULT = (1.0, 2.0, 4.0)             # {1x, 2x, 4x} x R_rms (frozen)
SQRT_P = math.sqrt(G1BS_PARAMS)
R_LADDER_RAW = tuple(k * R_RMS_REF * SQRT_P for k in R_LADDER_MULT)
R_LADDER_RMS = tuple(r / SQRT_P for r in R_LADDER_RAW)   # == k*R_RMS_REF
PIN_FUZZ_RMS_CARRIED = (G1.PIN_FUZZ_BAR / math.sqrt(G1B_PARAMS)) * SQRT_P
ONE_STEP_FUZZ_RAW = G1.FT_LR * SQRT_P        # one AdamW step ~ lr*sqrt(P)

WASH_SEED = G1.FREEZE_SEED                    # 10902 (the locked lineage)
CK_WASH: tuple[int, ...] = G1.CK_WASH if not SMOKE else (1, 2, 4)
FULL_DIAL_STEPS = (2, 50, 300) if not SMOKE else ()

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

# ---- the g1b stored reference (priors, not bars) -------------------------
PRIORS_G1B = {
    "source": "runs/g1b/metrics.json (the 2.74M continuity cell)",
    "root_gm12": G1B_ROOT_GM12,
    "W1_g_m12": {1: 0.9452, 2: 0.7768, 4: 0.8174, 10: 0.9227, 50: 0.9155,
                 100: 0.9222, 200: 0.9137, 300: 0.9181},
    "C_dead_at": 2,
    "e185_D_kill_raw_274M": 2.4892616271972656,
    "D_kill_rms_274M": 2.4892616271972656 / math.sqrt(G1B_PARAMS),
    "wall_tax_274M": 0.5263230800628662,    # the +0.53 reference
    "W2_g_m12": {1: 0.8087, 2: 0.5031, 4: 0.3203, 10: 0.7301, 50: 0.7767,
                 100: 0.7436, 200: 0.6736, 300: 0.6465},
}

# ---- g1bS4's committed ladder texture at the WEAK root (0.2523) — context
# priors only, never adjudicated against (no bar shopping) ------------------
PRIORS_G1BS4_LADDER = {
    "source": "runs/g1bS4/metrics.json (the take-4 arms-for-the-record at the "
              "0.2523 root)",
    "root_gm12": 0.2523, "C_dead_at": 1, "D_kill_raw": 3.157,
    "W1": {"gm12_at_1": 0.0400, "flat_range": "0.375-0.485",
           "flat_retention_min": 1.486, "tax_dCE_300_vs_C": 0.232},
    "W3": {"gm12_at_1": None, "gm12_at_200": 0.306, "gm12_at_300": 0.1233,
           "flat_retention_min": 0.49},
    "note": ("at the weak root the +1 formation shock breached every strict "
             "bar (W1 0.0400@+1) before recovery ABOVE root; the texture "
             "context for THIS cell's stronger-root dip"),
}

# ---- THE OWNER ENVELOPE (tighter than the lab's standing guards) ----------
OWNER_UTIL_C = 20.0              # launch only when util <= 20%
OWNER_TEMP_C = 70.0              # AND temp <= 70C (double-poll)
OWNER_MEM_FRAC = 0.60            # resident-neighbor caution (mem <= 60%)
COOLDOWN_S = 180.0               # >= 180 s between bursts (the dispatch)
TRAIN_CAP_GPU = 90.0             # <= 90 s GPU bursts (the dispatch)
GPU_WAIT_MAX = float(os.environ.get("G1BS6_GPU_WAIT_MAX", "7200"))
                                 # when in doubt WAIT; after 2 h of no
                                 # windows STOP honestly (resumable) —
                                 # never a silent CPU hop

REGISTERED = {
    "bars_verbatim": {
        "WALL-SCALES": ("fires if at R_rms_match the fact's battery-channel "
                        "p(Z) stays >= 0.9-equivalent (scaled to this host's "
                        "root strength) at every checkpoint through +300 "
                        "while C dies by +50."),
        "WALL-FADES": ("fires if C dies but no R on the ladder holds the bar "
                       "(or only the tightest R holds it while freezing CE)."),
        "WALL-TIGHTENS": ("fires if a LOOSER R than rms-match holds (the "
                          "basin grew with dimension) — co-report the CE tax "
                          "curve at scale (the +0.53 reference re-priced)."),
    },
    "scaled_bar_definition": {
        "formula": "bar_0p9eq = root_gm12(this host) * 0.9 / 0.9155886769294739",
        "g1b_root_gm12": G1B_ROOT_GM12,
        "factor_vs_root": 0.9 / G1B_ROOT_GM12,
        "root_source": ("THIS run's measured dial on the LOADED g1bS5 peak "
                        "root (committed 0.7677; identity-gated by "
                        "G-ROOTLOAD)"),
        "holds": ("g-12 >= bar_0p9eq at EVERY checkpoint "
                  "{1,2,4,10,50,100,200,300} (light battery read, g1b's "
                  "convention)"),
        "freezing_CE": ("W1 in-batch corpus CE at +300 >= root CE_R (the "
                        "walled organism never improved on its own root "
                        "corpus CE through the wash)"),
        "secondary_co_reported": [
            "g1-verbatim maintains (>=0.50 every ckpt) / dies (<=0.27 by +50)",
            "post-transient flat-phase retention: min g-12 over "
            "{10,50,100,200,300} >= 0.9 x root_gm12",
        ],
        "note": ("defined in the dispatch BEFORE compute; the primary bar is "
                 "intentionally strict (0.983x root): g1b's own W1 dipped to "
                 "0.848x root at +2 then sat flat at ~1.00x — at 10x the dip "
                 "discriminates a raw-geometry leak (dip widens, no rung "
                 "holds) from an rms-matched basin (dip shallow, rung holds)"),
    },
    "gate_deviation": {
        "registered_before_compute": True,
        "the_decision": ("THE ARMS RUN (the fifth take, not a gate stop): the "
                         "peak root sits 0.012 below the 0.78 express bar "
                         "(0.7677 < 0.78); the wall bars adjudicate against "
                         "the scaled-bar definition (the root-strength-"
                         "relative form computed from THIS root's measured "
                         "gm12), with the sub-bar root disclosed on every "
                         "artifact"),
        "root_bar": G1.EXPRESS_BAR,
        "expected_root_gm12": G1BS5_PEAK_ROOT["root_gm12"],
        "rationale": ("the express gate's purpose (g1bS2/g1bS3's own "
                      "failure_action semantics) was preventing arms on a "
                      "dead channel — 0.0010-class roots where the wall "
                      "assay measures noise; 0.7677 is a live channel 1.6% "
                      "under the bar; the sweep's curve licenses the arms "
                      "(g1bS5's SHARP-OPTIMUM peak, CE_R 1.697 healthy, "
                      "held30 0.557)"),
        "mechanics": ("G-ROOT is recorded as a READ on every artifact and is "
                      "NOT a member of the adjudicating gate set; G-ROOTLOAD "
                      "(the loaded checkpoint's identity vs g1bS5's committed "
                      "record) replaces it as the load gate; no other gate, "
                      "dial, bar or stream is touched"),
    },
    "sources_loaded": {
        "base": ("runs/checkpoints/g1bS2_base.pt — g1bS2's PASSED base "
                 "(val-min-anchored 1200-step cosine, lr 4e-4, seed 1337); "
                 "G-BASE-QUAL PASS on record, re-derived ON THE LOAD (takes "
                 "3-5's standing license: never retrained)"),
        "install": ("runs/checkpoints/g1bS2_install.pt — the "
                    "movement-matched s1000 @ 4e-4 install (seed 42); "
                    "post-install state REUSABLE (re-verified by takes 3-5 "
                    "and re-checked here)"),
        "peak_root": ("runs/checkpoints/g1bS5_root_m020.pt — g1bS5's m020 "
                      "formation-curve peak root (base+install LOADED + "
                      "e113 jitter consolidation s500 @ 4e-4 = 0.20 rms, "
                      "seed 10901); committed root g-12 0.7677 (T167's "
                      "SHARP-OPTIMUM peak); the licensed theta0 of every "
                      "g1bS6 arm — LOADED VERBATIM, never retrained"),
    },
    "wash": {
        "recipe": ("e176N arm A VERBATIM (neutral bank seed 170, batch 32 = "
                   "16 neutral + 16 random, full-token CE, AdamW (0.9,0.95) "
                   "wd 0.1 const lr 1e-3 clip 1.0)"),
        "seed": WASH_SEED, "ckpt_steps": list(CK_WASH),
        "note": ("the wash is the treatment, not the dial — UNTOUCHED; the "
                 "Adam-clock (T139) and ONE_STEP_FUZZ_RAW are priced at 1e-3 "
                 "and stay there"),
    },
    "R_convention": {
        "R_rms_ref": R_RMS_REF,
        "derivation": "0.7/sqrt(2739072)",
        "ladder_mult": list(R_LADDER_MULT),
        "ladder_raw": list(R_LADDER_RAW),
        "ladder_rms": list(R_LADDER_RMS),
        "assertion": "every rung's rms == mult x R_rms_ref (G-CONFIG, "
                     "asserted)",
        "both_conventions": ("every checkpoint log + readout reports raw L2 "
                             "AND per-coordinate rms; G-PIN primary = "
                             "rms-carried fuzz (g1bS4's committed "
                             "convention), verbatim R+1.5 co-reported"),
    },
    "owner_envelope": {
        "launch_gate": f"util <= {OWNER_UTIL_C:.0f}% AND temp <= "
                       f"{OWNER_TEMP_C:.0f}C (double-poll 5 s) AND mem <= "
                       f"{OWNER_MEM_FRAC:.0%}",
        "burst_cap_s": TRAIN_CAP_GPU, "cooldown_s": COOLDOWN_S,
        "no_cpu_hop": ("the wash arms NEVER run on CPU (a 10M CPU wash is "
                       "not a viable burst; if no CUDA or no window: STOP "
                       "honestly, partial metrics + resume ckpts hold the "
                       "record)"),
    },
    "predicted": ("mechanical expectation only (registered before compute): "
                  "C dead by +1..+4 (the T139 Adam clock at 10M; g1bS4's C "
                  "died at +1 with D_kill ~= one AdamW step rms); the wall's "
                  "first-step blindness (commit at d=0; one AdamW step = "
                  f"{ONE_STEP_FUZZ_RAW:.2f} raw > every rung except W3) "
                  "makes a +1/+2 dip the registered discriminator — at the "
                  "WEAK 0.2523 root (g1bS4) W1's +1 reading fell to 0.0400 "
                  "then recovered ABOVE root; whether the dip survives at "
                  "the STRONGER 0.7677 root, and which rung (if any) holds "
                  "the strict every-checkpoint 0.9eq bar, is the open bit. "
                  "If the dip breaches every rung, WALL-FADES fires with the "
                  "dip texture (reported, not re-tuned)."),
    "falsifiers": ("C survives (+50 > 0.27) => G-CTRL fails => TEXTURE (a "
                   "scale finding about the WASH, registered as texture); "
                   "no rung holds while C dies => WALL-FADES; a looser rung "
                   "holds => WALL-TIGHTENS with the tax curve."),
}

deviations: list[str] = [
    "THE GATE DEVIATION (registered pre-run, frozen in the dispatch): the "
    "arms RUN on the sub-bar peak root (0.7677 vs the 0.78 express bar, "
    "-0.0123) — G-ROOT is a disclosed READ, not an adjudicating gate; the "
    "wall bars adjudicate against the scaled-bar definition computed from "
    "THIS root's measured gm12; G-ROOTLOAD (identity vs g1bS5's committed "
    "record, tol 0.005) is the load gate. Justification: the gate's purpose "
    "was preventing arms on a dead channel; 0.7677 is a live channel 1.6% "
    "under the bar and g1bS5's curve licenses the arms. No other gate, dial, "
    "bar or stream touched.",
    "THE OWNER ENVELOPE (STATE.json compute_directive 2026-10-02, permanent) "
    "replaces the lineage's standing GPU policy for this run — TIGHTER on "
    "every axis: launch gate util <= 20% AND temp <= 70C (double-poll; the "
    "lineage's gpu_ok() was util <= 85% / temp <= 80C), burst cap 90 s (was "
    "180 s), cooldown 180 s (was 120 s), no CPU fallback for the wash arms "
    "(pause-and-wait / stop-honestly instead). Wash arithmetic, streams and "
    "seeds UNTOUCHED (g1bS4_wash's verbatim e176N arithmetic, re-capped).",
    "CHUNK-RESUMABLE WASH (mechanical addition to g1bS4_wash, mandated by "
    "the 90 s burst cap; device policy only, no training semantics): the "
    "300-step wash runs in <= 90 s counted bursts with a full-state resume "
    "ckpt ({model, opt, gen_state, step, traj, sds, deltas, x_hashes}) at "
    "every burst boundary — a cap or a process kill CONTINUES from the "
    "boundary instead of trimming (bit-identical stream: the generator "
    "state carries it; same precedent as the consolidation chunks). The "
    "draw order, batch composition, targets, optimizer and clip are "
    "VERBATIM g1bS4_wash.",
    "FULL DIALS TRIMMED to root + W1@{2,50,300} (g1b's FULL_DIAL_WASH "
    "verbatim): W2/W3/C run light-only (their roles are the ladder and the "
    "kill clock — g1bR's precedent); at 10M one full dial is ~4-5 CPU-min, "
    "so 12 dials would dominate the cell.",
    "NO NOISE ARMS: the g1bS design's arms are C + the R ladder only; the "
    "noise split (N0/N1/N2) is g1b's cell, not re-run here.",
    "LIGHT-EVAL TWIN ON THE TRAINING DEVICE (g1bS4's registered scale "
    "adaptation, verbatim): ALL full dials (measure()) stay CPU-only; "
    "cross-device float fuzz is quantified by G-STEP1 and the G-ROOTLOAD "
    "dual-read co-report.",
    "DISPLACEMENT DEVICE-RESIDENT PER STEP (g1bS4's registered scale "
    "adaptation, verbatim): the flat-parameter L2/increment arithmetic is "
    "computed on the training device; cross-arm cosines move deltas to CPU "
    "first.",
    "torch threads 4 (shared machine; g1bW's adopted trim; g1's import "
    "resets to 8, reset after import).",
    "COMMON.PY INFRA REPAIR (this dispatch's pre-flight, commit 722f163): "
    "the envelope-log fix committed at 8d6be37 was syntactically invalid "
    "(literal newline in the f.write string) and never wired into gpu_ok(); "
    "repaired before this cell could import common at all. No training "
    "semantics touched.",
    "Smoke mode trims: washes 4 steps with ckpts {1,2,4}, the smoke root "
    "(g1bS5's smoke artifact), lean dials, no cooldowns, gate waits capped "
    "at 30 s — nothing adjudicated.",
]

device_events: list[dict] = []
trims: list[str] = []

_progressive = {"n": 0, "phases": []}


class OwnerWindowShut(Exception):
    """No owner-envelope GPU window opened within GPU_WAIT_MAX — the run
    stops honestly (partial metrics + resume ckpts hold the record)."""


def write_partial(rd: Path, phase: str, payload: dict) -> None:
    """The outage lesson, mechanized: metrics.json exists from the first
    phase onward and is rewritten after every phase/arm. Superseded by the
    final full write (partial=false). Bookkeeping only — never allowed to
    kill compute."""
    _progressive["n"] += 1
    _progressive["phases"].append(phase)
    try:
        out = {
            "experiment": "g1bS6_peak_wall",
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
# THE OWNER ENVELOPE gate: util AND temp AND mem, double-poll, park-and-wait.

def owner_gpu_ok() -> bool:
    s = gpu_status()
    return bool(s["util"] <= OWNER_UTIL_C and s["temp"] <= OWNER_TEMP_C
                and (s["mem_total"] == 0
                     or s["mem_used"] <= OWNER_MEM_FRAC * s["mem_total"]))


def wait_gpu_owner(tag: str, max_wait: float | None = None) -> torch.device:
    """THE OWNER ENVELOPE gate (STATE.json compute_directive — the tightest
    constraint): launch only when util <= 20% AND temp <= 70C, double-poll
    5 s apart (+ mem <= 60% resident-neighbor caution). Parks in 30 s polls
    (when in doubt WAIT); after max_wait STOPS honestly via OwnerWindowShut
    — never migrates to CPU."""
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


def coherence_stats(text: str) -> dict:
    """The g4 coherence dial (house stat; g1bS4/S5's verbatim helper)."""
    n = max(1, len(text))
    words = [w for w in text.split() if w]
    runs = max((sum(1 for _ in g) for _, g in itertools.groupby(text)),
               default=0)
    return {
        "len": len(text),
        "lower_space_fraction": sum(1 for c in text
                                    if c.islower() or c == " ") / n,
        "mean_word_len": (sum(len(w) for w in words) / len(words))
                         if words else None,
        "max_char_run": runs,
        "distinct_chars": len(set(text)),
    }


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("g1bS6_smoke" if SMOKE else "g1bS6")
    log(f"G1BS6 THE WALL AT 10x, TAKE 5 — THE ARMS ON THE PEAK ROOT (theta0 "
        f"= {SRC_ROOT_CK}, committed root g-12 "
        f"{G1BS5_PEAK_ROOT['root_gm12']:.4f} = 0.012 UNDER the 0.78 express "
        f"bar — REGISTERED DEVIATION: the arms RUN, the scaled bar "
        f"adjudicates; ladder "
        + " | ".join(f"W{i+1}: {r:.4f} raw = {m:.0f}x R_rms"
                     for i, (r, m) in enumerate(zip(R_LADDER_RAW,
                                                    R_LADDER_MULT)))
        + f"; wash lr {G1.FT_LR} seed {WASH_SEED}; owner envelope: bursts <= "
        f"{TRAIN_CAP_GPU:.0f}s, cooldown {COOLDOWN_S:.0f}s, gate util <= "
        f"{OWNER_UTIL_C:.0f}% temp <= {OWNER_TEMP_C:.0f}C; smoke={SMOKE}) "
        f"-> {rd}")
    write_partial(rd, "start", {
        "design": ("scratch/g1bS_design.md (convention frozen at dispatch; "
                   "bars verbatim; no bar shopping) + the dispatch's "
                   "registered gate deviation (the arms run on the sub-bar "
                   "peak root)"),
        "question": ("does the commit-and-project L2 ball hold a "
                     "consolidated fact through a wash that kills the "
                     "control, at ~10x the parameters (9,977,600 vs the "
                     "2,739,072 every g-law was minted on), with the R-dial "
                     "carried in per-coordinate RMS units — TAKE 5: the arms "
                     "on g1bS5's PEAK ROOT (0.20 rms formation optimum, "
                     "root g-12 0.7677), the strongest root the 10M lineage "
                     "has produced?"),
        "registered": REGISTERED, "deviations": deviations, "smoke": SMOKE,
        "priors": {"g1b_274M": PRIORS_G1B,
                   "g1bs4_ladder_weak_root": PRIORS_G1BS4_LADDER,
                   "g1bs5_peak_root": G1BS5_PEAK_ROOT},
        "device_events": device_events})
    set_seed(HOST_SEED)      # host init + base corpus (the 1337 family)

    # ---- patch g1's machinery to the 10M family (g1b's pattern) ---------
    G1.G1_CFG = G1BS_CFG
    G1.G1_PARAMS = G1BS_PARAMS

    # ---------------- protocol rebuild (g1bS4's main VERBATIM) -------------
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
              "note": "MEASUREMENT instrument only (e152's locked j=54 pool); "
                      "NO window from this pool enters any training"}
    G_POOL["pass"] = G_POOL["all_windows_name_in_place"]
    assert G_POOL["pass"], f"pool gate FAILED: {G_POOL}"

    # install windows (e043 build_win verbatim) + original-host anchor bank
    def build_win(p, host):
        return torch.cat([train_ids[p - G1.PRE: p], name_ids,
                          train_ids[p + len(host): p + len(host) + G1.POST_CAP]])

    _ = torch.stack([build_win(p, h) for p, h in install_occ])  # identity
    anchor_full = torch.stack([train_ids[p - G1.PRE: p - G1.PRE + G1.BLOCK]
                               for p, _ in install_occ])

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
            "construction": ("16 plain corpus windows from train_ids, RNG seed "
                             f"{G1.E170_ANCHOR_SEED}, rejection if [s, s+257) "
                             "contains FLORIZEL/ELIZABETH/ZEPH/MIRABEL — "
                             "e170's construction VERBATIM (= e176n arm A's "
                             "stream; FIXED content, shared by ALL arms)"),
            "n_windows": 16, "block": G1.BLOCK, "seed": G1.E170_ANCHOR_SEED,
            "starts": n_starts, "tries": tries, "rejections": rejections,
            "forbidden": list(G1.ANCHOR_FORBIDDEN),
            "windows_with_host_content": sum(
                1 for s in n_starts
                if any(f in train_text[s: s + G1.BLOCK + 1] for f in G1.HOSTS)),
            "junctions_covered": jc_neutral,
        },
        "budget_identical_to_e176n": bool(anchor_neutral.shape == (16, G1.BLOCK)),
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

    gates_surg: dict = {}

    def measure(sd: dict, tag: str, lean: bool = False) -> dict:
        """g1b's measure() VERBATIM dial (e176n's = the e131 dial set) on
        evl_load (settle+disarm; g1's PIVOT). CPU-only, offline (no cap)."""
        net = G1.evl_load(sd)
        sd_local = {k: v.detach().clone() for k, v in net.state_dict().items()}
        out: dict = {"tag": tag}
        out["base"] = {j: G1.battery_cell(net, bat_ids[j], zid) for j in G1.GEOS}
        out["base_held"] = {j: G1.battery_cell(net, held_ids[j], zid)
                            for j in G1.GEOS}
        out["ce_r"] = G1.ce_fixed_cpu(net, *r_eval_xy)
        log(f"[{tag}] base: " + " ".join(f"g{j:+d} {out['base'][j]['mean_pz']:.4f}"
                                         for j in G1.GEOS)
            + " | held30: " + " ".join(
                f"g{j:+d} {out['base_held'][j]['mean_pz']:.4f}" for j in G1.GEOS)
            + f" | CE_R {out['ce_r']:.4f}")
        out["site_read"] = G1.read_fact_at(net, pool_x, name_ids, zid,
                                           G1.SITE_ADDR_ROW, G1.SITE_Z_XCOL)
        log(f"[{tag}] site read @183: onset "
            f"{out['site_read']['pz_onset_mean']:.4f} span "
            f"{out['site_read']['pname_mean_over7']:.4f}")
        if not lean and not SMOKE:
            out["census_old"] = G1.row_census(net, G1.ROWS_OLD,
                                              lambda n: G1.battery_pz(
                                                  n, bat_ids[0], zid))
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
            DELS = {"d_all": G1.D_ALL, "d183": (G1.SITE_ADDR_ROW,)}
            out["del_table"] = {}
            for dl, rows_ in DELS.items():
                sd_d, gate = G1.deleted_wpe(sd_local, rows_)
                gates_surg[f"{tag}__{dl}"] = gate
                if not gate["pass"]:
                    raise RuntimeError(f"deletion gate FAILED {tag}/{dl}: "
                                       f"{gate}")
                net.load_state_dict(sd_d)
                cell = {"g0": G1.battery_cell(net, bat_ids[0], zid)["mean_pz"]}
                if dl == "d183":
                    cell["gm12"] = G1.battery_cell(net, bat_ids[-12],
                                                   zid)["mean_pz"]
                out["del_table"][dl] = cell
            net.load_state_dict(sd_local)
            log(f"[{tag}] deletions g0: " + " | ".join(
                f"{dl} {out['del_table'][dl]['g0']:.3f}" for dl in DELS))
        else:
            w = net.wpe.weight.data
            orig = w.clone()
            mean_row = orig.mean(0)
            bp = G1.battery_pz(net, bat_ids[0], zid)
            w[129] = mean_row
            m129 = bp - G1.battery_pz(net, bat_ids[0], zid)
            w.copy_(orig)
            w[129] = 0.0
            z129 = bp - G1.battery_pz(net, bat_ids[0], zid)
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
    # G-CONFIG: the 10M band + the R-convention assertion (Rule 12)
    # =====================================================================
    torch.manual_seed(HOST_SEED)
    host_net = TinyGPT(G1BS_CFG)
    n_params = host_net.num_params()
    G_CONFIG = {
        "params": n_params,
        "band": [8.5e6, 11.5e6],
        "vocab": G1BS_CFG.vocab, "block_size": G1BS_CFG.block_size,
        "n_layer": G1BS_CFG.n_layer, "n_head": G1BS_CFG.n_head,
        "n_embd": G1BS_CFG.n_embd,
        "R_rms_ref": R_RMS_REF,
        "R_ladder_raw": list(R_LADDER_RAW),
        "R_ladder_rms": list(R_LADDER_RMS),
        "R_ladder_mult": list(R_LADDER_MULT),
        "pin_fuzz_rms_carried_raw": PIN_FUZZ_RMS_CARRIED,
        "one_adamw_step_fuzz_raw": ONE_STEP_FUZZ_RAW,
        "assertions": {
            f"rung_{i+1}": {
                "mult": R_LADDER_MULT[i],
                "rms": R_LADDER_RMS[i],
                "expected_rms": R_LADDER_MULT[i] * R_RMS_REF,
                "raw": R_LADDER_RAW[i],
                "abs_err_rms": abs(R_LADDER_RMS[i]
                                   - R_LADDER_MULT[i] * R_RMS_REF),
                "pass": bool(abs(R_LADDER_RMS[i]
                                 - R_LADDER_MULT[i] * R_RMS_REF) < 1e-15),
            } for i in range(3)},
        "battery_geometry": {
            "vocab_unchanged": bool(G1BS_CFG.vocab == corpus.vocab_size == 65),
            "block_unchanged": bool(G1BS_CFG.block_size == G1.BLOCK == 256),
            "positions_read": {"site_rows": [G1.SITE_ADDR_ROW,
                                             G1.SITE_ADDR_ROW + 6],
                               "wpe_band": [121, 129],
                               "inside_block": True},
            "tokenizer": "same data/input.txt char stoi (65 chars)",
        },
    }
    G_CONFIG["pass"] = bool(
        8.5e6 <= n_params <= 11.5e6
        and all(a["pass"] for a in G_CONFIG["assertions"].values())
        and G_CONFIG["battery_geometry"]["vocab_unchanged"]
        and G_CONFIG["battery_geometry"]["block_unchanged"])
    assert G_CONFIG["pass"], f"G-CONFIG FAILED: {G_CONFIG}"
    log(f"G-CONFIG: {n_params:,} params in [8.5M, 11.5M]; ladder "
        + " | ".join(f"W{i+1}: {r:.4f} raw = {m:.0f}x {R_RMS_REF:.4e} rms"
                    for i, (r, m) in enumerate(zip(R_LADDER_RAW,
                                                   R_LADDER_MULT)))
        + f"; pin fuzz rms-carried {PIN_FUZZ_RMS_CARRIED:.3f} raw "
        f"(verbatim 1.5 co-reported; one-step {ONE_STEP_FUZZ_RAW:.3f}): PASS")
    del host_net
    write_partial(rd, "G-CONFIG", {"gates_partial": {"G_CONFIG": G_CONFIG},
                                   "config": {"n_layer": 8, "n_head": 8,
                                              "n_embd": 320,
                                              "block_size": 256,
                                              "params": n_params,
                                              "R_ladder_raw": list(R_LADDER_RAW),
                                              "R_ladder_rms": list(R_LADDER_RMS),
                                              "smoke": SMOKE}})

    # =====================================================================
    # PHASE 0a-host — THE BASE: g1bS2's PASSED base, LOADED VERBATIM (the
    # lineage license (takes 3-6): never retrain; identity gates ON THE LOAD)
    # =====================================================================
    log("=" * 78)
    log(f"HOST BASE: LOADED VERBATIM from {SRC_BASE_CK} (g1bS2's PASSED "
        f"base — {n_params:,} params, {BASE_STEPS}-step val-min-anchored "
        f"cosine, lr 4e-4, seed {HOST_SEED}); G-BASE-QUAL re-derived ON THE "
        f"LOAD (a corrupted load cannot slip through); never retrained")
    base_ck = CKPT_DIR / ("smoke_" + SRC_BASE_CK if SMOKE else SRC_BASE_CK)
    base_sample_prompt = val_text[:64]      # the fixed g4 prompt (recorded)
    if not base_ck.exists():
        raise SystemExit(
            f"G1BS6 source base missing: {base_ck} (the licensed cell LOADS "
            f"g1bS2's PASSED base; it cannot be retrained)")
    bstate = torch.load(base_ck, map_location="cpu", weights_only=False)
    hist: list[dict] = list(bstate.get("history", []))
    if not hist:
        raise SystemExit(f"loaded base carries no history: {base_ck}")
    bstep = int(bstate.get("step", hist[-1]["step"]))
    theta_base = {k: v.detach().clone() for k, v in bstate["model"].items()}
    del bstate
    # g1bS2's own committed record (read-only provenance cross-check)
    g1bs2_rec = {}
    _rec = E43.REPO / "runs" / ("g1bS2_smoke" if SMOKE else "g1bS2") / \
        "metrics.json"
    if _rec.exists():
        try:
            _r = json.loads(_rec.read_text(encoding="utf-8"))
            g1bs2_rec = {
                "G_BASE_QUAL": {k: v for k, v in
                                _r.get("gates", {}).get("G_BASE_QUAL",
                                                        {}).items()
                                if not isinstance(v, (dict, list))},
                "inst_cells": _r.get("provenance", {}).get("install", {})
                               .get("post_install_cells", {}),
            }
        except Exception as e:                                     # noqa: BLE001
            log(f"[base] g1bS2 record unreadable ({e}) — continuing without "
                f"the cross-check (recorded)")
    G_BASE = {"steps": bstep, "final_val_loss": hist[-1]["val_loss"],
              "loaded_from": f"runs/checkpoints/{SRC_BASE_CK}",
              "pass": bool(bstep == BASE_STEPS)}
    assert G_BASE["pass"], f"G-BASE FAILED (cosine incomplete): {G_BASE}"
    log(f"G-BASE: loaded {bstep}/{BASE_STEPS} steps, final val "
        f"{hist[-1]['val_loss']:.4f} — cosine COMPLETE (g1bS2's own PASS on "
        f"record at runs/g1bS2/metrics.json): PASS")
    CKPT_INVENTORY[f"src_{SRC_BASE_CK[:-3]}"] = {
        "path": f"runs/checkpoints/{SRC_BASE_CK}",
        "desc": ("g1bS2's PASSED 10M base (val-min-anchored 1200-step "
                 "cosine, lr 4e-4, seed 1337) — LOADED read-only by g1bS6 "
                 "(the license: never retrained)"),
        "step": bstep, "final_val_loss": hist[-1]["val_loss"]}
    base_cells = flat_cells(measure(theta_base, "g1bS6_base", lean=True))
    log(f"base cells: g-12 {base_cells['gm12']:.4f} CE_R "
        f"{base_cells['ce_r']:.4f}")

    # ---- G-BASE-QUAL (the registered legs re-derived ON THE LOADED STATE)
    eval_vals = [e["val_loss"] for e in hist]
    first_eval_val, final_val = eval_vals[0], eval_vals[-1]
    mono_ok = all(eval_vals[i + 1] <= eval_vals[i] + 0.02
                  for i in range(len(eval_vals) - 1))
    g2 = bool(mono_ok and final_val <= first_eval_val - 0.15)
    g3 = bool(final_val <= 1.70)
    common.DEVICE = "cpu"            # deterministic CPU sample (house tool)
    bnet = TinyGPT(G1BS_CFG)
    bnet.load_state_dict(theta_base)
    bnet.to(common.DEVICE)
    base_sample = common.generate(bnet, corpus, base_sample_prompt,
                                  max_new_tokens=300)
    del bnet
    base_gen = base_sample[len(base_sample_prompt):]
    bstats = coherence_stats(base_gen)
    mwl = bstats["mean_word_len"]
    g4 = bool(bstats["lower_space_fraction"] >= 0.70 and mwl is not None
              and 2.0 <= mwl <= 8.0 and bstats["max_char_run"] <= 10
              and bstats["distinct_chars"] >= 18)
    G_BASE_QUAL = {
        "cosine_complete": G_BASE["pass"],
        "eval_vals": eval_vals,
        "val_monotone_decreasing_noise0.02": mono_ok,
        "first_eval_val": first_eval_val, "final_val": final_val,
        "g2_val_decreasing": g2,
        "g3_final_val<=1.70": g3,
        "sample_prompt": base_sample_prompt,
        "sample_generated": base_gen,
        "sample_stats": bstats,
        "g4_coherence_sample": g4,
        "pass": bool(G_BASE["pass"] and g2 and g3 and g4),
        "enforced": bool(not SMOKE),
        "verification_of": ("the LOADED g1bS2 base (its own G-BASE-QUAL "
                            "PASS is on record at runs/g1bS2/metrics.json; "
                            "these legs re-derive on the carried history + "
                            "a fresh coherence sample so a corrupted load "
                            "cannot slip through)"),
        "g1bs2_committed_record": g1bs2_rec.get("G_BASE_QUAL", {}),
    }
    log(f"G-BASE-QUAL (on the loaded state): cosine {G_BASE['pass']} | val "
        f"decreasing {g2} ({first_eval_val:.4f} -> {final_val:.4f}) | "
        f"final<=1.70 {g3} | coherence {g4}: "
        f"{'PASS' if G_BASE_QUAL['pass'] else 'FAIL'}"
        + ("" if not SMOKE else " (smoke: NOT enforced)"))
    write_partial(rd, "G-BASE-QUAL", {"G_BASE": G_BASE,
                                      "base_cells": base_cells,
                                      "G_BASE_QUAL": G_BASE_QUAL})
    if not G_BASE_QUAL["pass"] and not SMOKE:
        raise SystemExit(
            "G-BASE-QUAL FAILED on the loaded base (a corrupted load or a "
            "record mismatch — the dispatch bar: NO arms; report and stop, "
            "no dial search)")

    # =====================================================================
    # PHASE 0a — INSTALL: g1bS2's movement-matched install, LOADED VERBATIM
    # (the standing license: the post-install state is REUSABLE)
    # =====================================================================
    log("=" * 78)
    log(f"INSTALL: LOADED VERBATIM from {SRC_INSTALL_CK} (g1bS2's "
        f"movement-matched e043-Dmix install: 16 install + 16 paired + 32 "
        f"random = 64 windows/step, dose s{INSTALL_STEPS}, house cosine, lr "
        f"4e-4 width-scaled, seed {G1.INSTALL_SEED}); post-install cells "
        f"re-measured THIS run and cross-checked against g1bS2's record")
    inst_ck = CKPT_DIR / ("smoke_" + SRC_INSTALL_CK if SMOKE
                          else SRC_INSTALL_CK)
    if not inst_ck.exists():
        raise SystemExit(
            f"G1BS6 source install missing: {inst_ck} (the licensed cell "
            f"LOADS g1bS2's install checkpoint — the REUSABLE post-install "
            f"state)")
    istate = torch.load(inst_ck, map_location="cpu", weights_only=False)
    istep = int(istate.get("step", 0))
    theta_install = {k: v.detach().clone()
                     for k, v in istate["model"].items()}
    del istate
    G_INST = {"steps": istep,
              "loaded_from": f"runs/checkpoints/{SRC_INSTALL_CK}",
              "recipe": (f"e043 Dmix s{INSTALL_STEPS} @ lr 4e-4 "
                         f"(movement-matched, RECOVERY-3), seed "
                         f"{G1.INSTALL_SEED} — g1bS2's run"),
              "pass": bool(istep == INSTALL_STEPS)}
    assert G_INST["pass"], f"INSTALL incomplete: {G_INST}"
    inst_cells = flat_cells(measure(theta_install, "post_install", lean=True))
    prev_inst = g1bs2_rec.get("inst_cells") or {}
    d_gm12 = (abs(inst_cells["gm12"] - prev_inst["gm12"])
              if prev_inst.get("gm12") is not None else None)
    reuse_ok = bool(SMOKE or (d_gm12 is not None and d_gm12 <= 0.02))
    G_INST["reuse_check"] = {
        "g1bs2_committed": prev_inst,
        "this_run_remeasured": inst_cells,
        "abs_d_gm12": d_gm12, "tol": 0.02,
        "reuse_ok": reuse_ok,
        "rationale": ("the post-install state is the licensed grandparent of "
                      "the peak root; g1bS2's G-ROOT failure was caused by "
                      "its lr-1e-3 consolidation itself, strictly "
                      "downstream of this state; smoke skips the check "
                      "(machinery only)"),
    }
    assert reuse_ok, f"post-install drift vs g1bS2 record: {G_INST['reuse_check']}"
    pv = prev_inst.get("gm12")
    log(f"post-install (re-measured): g-12 {inst_cells['gm12']:.4f} g0 "
        f"{inst_cells['g0']:.4f} CE_R {inst_cells['ce_r']:.4f} "
        + (f"(g1bS2 record g-12 {pv:.4f}; |d| {d_gm12:.2e} <= 0.02): "
           f"REUSABLE" if pv is not None else "(no g1bS2 record read)"))
    write_partial(rd, "G-INST", {"G_INST": G_INST, "inst_cells": inst_cells})
    CKPT_INVENTORY[f"src_{SRC_INSTALL_CK[:-3]}"] = {
        "path": f"runs/checkpoints/{SRC_INSTALL_CK}",
        "desc": ("g1bS2's movement-matched install (e043 Dmix s1000 @ 4e-4, "
                 "seed 42) — LOADED read-only by g1bS6; the licensed "
                 "grandparent of the peak root"),
        "step": istep}

    # =====================================================================
    # THE PEAK ROOT — LOADED VERBATIM (g1bS5's m020 artifact); G-ROOTLOAD
    # identity vs g1bS5's committed record; G-ROOT read (THE REGISTERED
    # DEVIATION: a read, not a stop)
    # =====================================================================
    log("=" * 78)
    log(f"THE PEAK ROOT: LOADED VERBATIM from {SRC_ROOT_CK} (g1bS5's m020 "
        f"formation-curve peak: base+install LOADED + e113 jitter "
        f"consolidation s{G1BS5_PEAK_ROOT['steps']} @ lr "
        f"{G1BS5_PEAK_ROOT['lr']} = {G1BS5_PEAK_ROOT['movement_rms']:.2f} rms "
        f"(seed {G1BS5_PEAK_ROOT['cons_seed']}); committed root g-12 "
        f"{G1BS5_PEAK_ROOT['root_gm12']:.4f} = the curve's interior PEAK, "
        f"0.012 below the 0.78 express bar)")
    root_ck_path = CKPT_DIR / ("smoke_" + SRC_ROOT_CK if SMOKE
                               else SRC_ROOT_CK)
    if not root_ck_path.exists():
        raise SystemExit(
            f"G1BS6 source root missing: {root_ck_path} (the licensed cell "
            f"LOADS g1bS5's committed peak-root checkpoint; it cannot be "
            f"retrained)")
    rstate = torch.load(root_ck_path, map_location="cpu", weights_only=False)
    theta0 = {k: v.detach().clone() for k, v in rstate["model"].items()}
    root_meta = dict(rstate.get("meta", {}))
    del rstate

    # ---- THE ROOT DIAL (full measure, CPU; the prior takes' convention) ---
    root_m = measure(theta0, "g1bS6_root", lean=SMOKE)
    root_cells = flat_cells(root_m)

    # ---- G-ROOTLOAD: identity vs g1bS5's committed record ----------------
    tol_rootload = 0.005
    _rootload_keys = {"gm12": "root_gm12", "g0": "g0",
                      "held30_gm12": "held30_gm12", "ce_r": "ce_r"}
    deltas_root = {
        k: abs(root_cells[k] - G1BS5_PEAK_ROOT[ref])
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
                      "root: the root dial is a deterministic CPU measure "
                      "on the same tensors (expect ~0; 0.005 tolerates "
                      "float-reduction fuzz); a mismatch would mean the "
                      "wrong artifact or a corrupted load — the arms would "
                      "measure something else entirely"),
    }
    log(f"G-ROOTLOAD: re-measured root g-12 {root_cells['gm12']:.6f} vs "
        f"g1bS5 committed {G1BS5_PEAK_ROOT['root_gm12']:.6f} "
        f"(|d| {deltas_root['gm12']:.2e}); |d CE_R| {deltas_root['ce_r']:.2e}; "
        f"|d g0| {deltas_root['g0']:.2e}; |d held30| "
        f"{deltas_root['held30_gm12']:.2e} (tol {tol_rootload}): "
        f"{'PASS' if G_ROOTLOAD['pass'] else 'FAIL'}")
    assert G_ROOTLOAD["pass"], f"G-ROOTLOAD FAILED: {G_ROOTLOAD}"

    # ---- G-ROOT: the READ + THE REGISTERED DEVIATION ---------------------
    G_ROOT_read = {
        "bar": G1.EXPRESS_BAR,
        "gm12": root_cells["gm12"],
        "pass": bool(root_cells["gm12"] >= G1.EXPRESS_BAR),
        "note": ("A READ, NOT A STOP — THE REGISTERED PRE-RUN DEVIATION "
                 "(frozen in the dispatch): the arms RUN on the sub-bar "
                 "peak root; the wall bars adjudicate against the scaled-bar "
                 "definition computed from THIS root; see "
                 "REGISTERED.gate_deviation"),
        "not_in_gate_set": True,
    }
    log(f"G-ROOT (read): root g-12 {root_cells['gm12']:.4f} vs bar >= "
        f"{G1.EXPRESS_BAR} -> "
        f"{'PASS' if G_ROOT_read['pass'] else 'BELOW BAR by ' + format(G1.EXPRESS_BAR - root_cells['gm12'], '.4f')}"
        f" — REGISTERED DEVIATION: the arms RUN (a live channel 1.6% under "
        f"the bar; g1bS5's curve licenses the arms); the scaled bar "
        f"adjudicates")

    # ---- THE SCALED BAR, computed from the root BEFORE any arm trains ----
    bar_0p9eq = root_cells["gm12"] * 0.9 / G1B_ROOT_GM12
    SCALED_BAR = {
        "definition": REGISTERED["scaled_bar_definition"],
        "root_gm12_measured": root_cells["gm12"],
        "bar_0p9eq": bar_0p9eq,
        "bar_09x_root": 0.9 * root_cells["gm12"],
        "g1b_W1_reference": {"min": 0.7768, "argmin_step": 2,
                             "min_retention_vs_root": 0.7768 / G1B_ROOT_GM12,
                             "note": "g1b's own W1 dipped below BOTH "
                                     "0.9x-root and the 0.9eq bar at +2/+4"},
        "sub_bar_root_disclosure": (f"the root is {root_cells['gm12']:.4f} = "
                                    f"{G1.EXPRESS_BAR - root_cells['gm12']:.4f} "
                                    f"BELOW the 0.78 express bar (the "
                                    f"registered deviation; disclosed on "
                                    f"every artifact)"),
        "computed_before_arms": True,
    }
    log(f"SCALED BAR (frozen formula, root measured): bar_0p9eq = "
        f"{root_cells['gm12']:.4f} x 0.9 / {G1B_ROOT_GM12:.4f} = "
        f"{bar_0p9eq:.4f}  (= {0.9 / G1B_ROOT_GM12:.5f} x root; secondary "
        f"0.9x root = {0.9 * root_cells['gm12']:.4f})")

    write_partial(rd, "root-load+scaled-bar", {
        "G_ROOTLOAD": G_ROOTLOAD, "G_ROOT_read": G_ROOT_read,
        "root_cells": root_cells, "scaled_bar": SCALED_BAR,
        "theta0_ckpt": f"runs/checkpoints/{SRC_ROOT_CK}",
        "gates_partial": {"G_ROOTLOAD": G_ROOTLOAD}})
    CKPT_INVENTORY[f"src_{SRC_ROOT_CK[:-3]}"] = {
        "path": f"runs/checkpoints/{SRC_ROOT_CK}",
        "desc": ("g1bS5's m020 formation-curve PEAK root (base+install "
                 "LOADED read-only + e113 jitter consolidation s500 @ 4e-4 = "
                 "0.20 rms, seed 10901) — theta0 for every g1bS6 arm, LOADED "
                 "VERBATIM (never retrained)"),
        "params": n_params, "movement_rms": G1BS5_PEAK_ROOT["movement_rms"],
        "steps": G1BS5_PEAK_ROOT["steps"], "lr": G1BS5_PEAK_ROOT["lr"],
        "cons_seed": G1BS5_PEAK_ROOT["cons_seed"],
        "root_gm12": root_cells["gm12"]}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =====================================================================
    # THE ARMS — C + the RMS ladder (cooldown 180 s before each; ALL inputs
    # bit-identical; the ONLY delta is the commit radius)
    # =====================================================================
    ARM_SPECS = ([("C", None,
                   "CONTROL — uncommitted neutral wash (e176N arm A "
                   "VERBATIM at 10M): the kill clock, D_kill, the CE "
                   "adaptation curve")]
                  + [(f"W{i+1}", R_LADDER_RAW[i], R_LADDER_MULT[i],
                      f"WALL {R_LADDER_MULT[i]:.0f}x R_rms = {R_LADDER_RAW[i]:.4f} "
                      f"raw L2 = {R_LADDER_RMS[i]:.4e} rms — commit then the "
                      f"identical neutral wash; step-1 weights equal to C's "
                      f"(the wall first acts at forward 2)") for i in range(3)])

    arms: dict = {}
    batteries_all: dict = {}
    G_BITROOT = {}
    for spec in ARM_SPECS:
        tag, R, *rest = spec
        desc = rest[-1]
        log("=" * 78)
        if not SMOKE:
            log(f"[owner envelope] cooldown {COOLDOWN_S:.0f}s before {tag} "
                f"(no back-to-back bursts)")
            cooldown(COOLDOWN_S)
        log(f"ARM {tag} — {desc}")
        net0 = G1.evl_load(theta0) if R is None else G1.CommittedGPT(G1BS_CFG)
        if R is not None:
            net0.load_state_dict(theta0)
            net0.commit(R)
            body, _ = G1.split_anchored_sd(net0.state_dict())
            md = max(float((body[k].float() - theta0[k].float()).abs().max())
                     for k in theta0)
            anch_ok = all(torch.equal(net0._anchor(n), p.detach())
                          for n, p in net0.named_parameters())
            G_BITROOT[tag] = {"max_abs_diff": md,
                              "anchors_bit_equal": bool(anch_ok),
                              "n_anchor_tensors": net0._n_anchor_tensors,
                              "pass": bool(md == 0.0 and anch_ok)}
            assert G_BITROOT[tag]["pass"], f"{tag}: wall root != theta0"
            log(f"G_BITROOT[{tag}]: max|diff| {md:.1e}, anchors bit-equal, "
                f"R = {R:.6f} raw = {R / SQRT_P:.6e} rms: PASS")
        try:
            arm = g1bS6_wash(tag, net0, anchor_neutral, train_ids, itos,
                             r_eval_xy, gm12_ids, g0_ids, zid,
                             resume_ck=(CKPT_DIR /
                                        (f"smoke_g1bS6_wash_{tag}_resume.pt"
                                         if SMOKE
                                         else f"g1bS6_wash_{tag}_resume.pt")))
        except OwnerWindowShut as e:
            write_partial(rd, f"owner-window-shut-{tag}",
                          {"stopped_at": tag, "reason": str(e),
                           "arms_partial": {t: {
                               "steps_ran": arms[t]["steps_ran"],
                               "device": arms[t]["device"]}
                               for t in arms},
                           "device_events": device_events})
            log(f"OWNER WINDOW SHUT at arm {tag}: {e} — stopping honestly "
                f"(partial metrics + resume ckpts hold the record)")
            raise SystemExit(3)
        G_DRAWFREE = {"zeph_violations": arm["zeph_violations"],
                      "pass": bool(arm["zeph_violations"] == 0)}
        assert G_DRAWFREE["pass"], f"{tag}: name token leaked into a window"
        gates_surg[f"G_DRAWFREE_{tag}"] = G_DRAWFREE
        arms[tag] = arm
        write_partial(rd, f"arm-{tag}", {"arms_partial": {tag: {
            "R_raw": R, "R_rms": (R / SQRT_P if R else None),
            "steps_ran": arm["steps_ran"], "device": arm["device"],
            "wall_R": arm["wall_R"],
            "g_m12": {t["step"]: t["g_m12_mean_pz"] for t in arm["traj"]
                      if "g_m12_mean_pz" in t},
            "traj": [{k: v for k, v in t.items() if k != "d_proj"}
                     for t in arm["traj"]],
            "x_hashes_head": {s: arm["x_hashes"][s]
                              for s in list(arm["x_hashes"])[:2]},
        }}})

        # checkpoints: each arm's final only (g1b's wash-arm convention)
        for s in sorted(arm["sds"]):
            if s == max(arm["sds"]):
                save_ckpt(f"g1bS6_{tag}_s{s}", arm["sds"][s],
                          {"desc": f"g1bS6 peak root + {s}-step true-target "
                                   f"neutral wash (R={R} raw = "
                                   f"{R / SQRT_P if R else None} rms, input "
                                   f"seed {WASH_SEED}, lr {G1.FT_LR})",
                           "steps": int(s), "R_raw": R,
                           "R_rms": R / SQRT_P if R else None,
                           "input_seed": WASH_SEED, "lr": G1.FT_LR,
                           "base": f"runs/checkpoints/{SRC_ROOT_CK}"})

        # full dials: W1 at g1b's FULL_DIAL_WASH {2,50,300}; others light-only
        full_at = [s for s in FULL_DIAL_STEPS if s in arm["sds"]] \
            if tag == "W1" else []
        batteries = {}
        for s in full_at:
            log(f"{tag} +{s} full dial (CPU)")
            batteries[str(s)] = measure(arm["sds"][s], f"{tag}{s}", lean=SMOKE)
        batteries_all[tag] = batteries

    # =====================================================================
    # CONSTRUCTION-FIDELITY GATES
    # =====================================================================
    G_INPUTS = {"per_step": {}, "pass": None, "note":
                "the seed-10902 aj/rj draw sequence is shared by ALL arms — "
                "per-step input batches are bit-identical (md5-gated) "
                "through the wash horizon"}
    for step in range(1, CK_WASH[-1] + 1):
        hs = {t: arms[t]["x_hashes"].get(step) for t in arms}
        same = all(h is not None for h in hs.values()) and \
            len(set(hs.values())) == 1
        G_INPUTS["per_step"][step] = {t: hs[t] for t in arms}
        G_INPUTS["per_step"][step]["identical"] = bool(same)
    G_INPUTS["pass"] = bool(all(v["identical"]
                                for v in G_INPUTS["per_step"].values()))
    assert G_INPUTS["pass"], "input streams diverged across arms"
    log(f"G-INPUTS: per-step inputs bit-identical across all {len(arms)} "
        f"arms through +{CK_WASH[-1]}: PASS")

    G_STEP1 = {"per_arm": {}}
    for tag in ("W1", "W2", "W3"):
        body_w, _ = G1.split_anchored_sd(arms[tag]["sds"][1])
        sd_c = arms["C"]["sds"][1]
        md1 = max(float((body_w[k].float().cpu() - sd_c[k].float()).abs().max())
                  for k in sd_c)
        G_STEP1["per_arm"][tag] = {"max_abs_diff": md1, "tol": 1e-4,
                                   "pass": bool(md1 <= 1e-4)}
    G_STEP1["pass"] = bool(all(v["pass"] for v in G_STEP1["per_arm"].values()))
    assert G_STEP1["pass"], f"G-STEP1 FAILED (wall acted at forward 1?): {G_STEP1}"
    log("G-STEP1: wall arms' step-1 bodies == C's step-1 body (max|diff| "
        + ", ".join(f"{v['max_abs_diff']:.1e}"
                    for v in G_STEP1["per_arm"].values())
        + " <= 1e-4): PASS — the wall is inert until forward 2")
    write_partial(rd, "input-gates", {
        "G_INPUTS_pass": G_INPUTS["pass"], "G_STEP1": G_STEP1,
        "G_BITROOT": G_BITROOT})

    # =====================================================================
    # DISPLACEMENT TABLES + COSINES (e185's currency, measured; both
    # conventions at every checkpoint)
    # =====================================================================
    def disp_rows(tag):
        rows = []
        for t in arms[tag]["traj"]:
            s = t["step"]
            row = {"step": s, "ce_batch": t["ce_batch"],
                   "cum_disp_raw": t["cum_disp"],
                   "cum_disp_rms": t["cum_disp"] / SQRT_P,
                   "step_disp_raw": t["step_disp"],
                   "step_disp_rms": (t["step_disp"] / SQRT_P
                                     if t["step_disp"] is not None else None),
                   "d_proj_raw": t["d_proj"],
                   "d_proj_rms": (t["d_proj"] / SQRT_P
                                  if t["d_proj"] is not None else None)}
            if "g_m12_mean_pz" in t:
                row["g_m12_light"] = t["g_m12_mean_pz"]
                row["ce_r_light"] = t["ce_r"]
            if s in arms[tag]["deltas"] and s in arms["C"]["deltas"]:
                d_a = arms[tag]["deltas"][s].float().cpu()
                d_c = arms["C"]["deltas"][s].float().cpu()
                row["cos_vs_C"] = float(torch.dot(d_a, d_c)
                                        / (torch.norm(d_a) * torch.norm(d_c)
                                           + 1e-30))
            rows.append(row)
        return rows

    disp_table = {tag: disp_rows(tag) for tag in arms}

    # =====================================================================
    # ADJUDICATION (the frozen bars; order GATES -> WALL -> COSTS; no
    # shopping; the scaled-bar definition was computed before the arms)
    # =====================================================================
    def light_gm12(tag):
        return {t["step"]: t["g_m12_mean_pz"] for t in arms[tag]["traj"]
                if "g_m12_mean_pz" in t}

    fmt = lambda v: "n/a" if v is None else f"{v:.4f}"

    # ---- G-CTRL: arm C dies by +50
    c_gm12 = light_gm12("C")
    G_CTRL = {"bar": G1.SHUT_BAR, "gm12_at_50": c_gm12.get(50),
              "earliest_le_bar": next((s for s in CK_WASH if s > 0
                                       and c_gm12.get(s, 1.0) <= G1.SHUT_BAR),
                                      None),
              "pass": bool(c_gm12.get(50, 1.0) <= G1.SHUT_BAR)}
    log(f"G-CTRL: arm C g-12 at +50 = {fmt(G_CTRL['gm12_at_50'])} "
        f"(earliest <= {G1.SHUT_BAR}: +{G_CTRL['earliest_le_bar']}): "
        f"{'PASS' if G_CTRL['pass'] else 'FAIL'}")
    t_kill = G_CTRL["earliest_le_bar"]
    D_kill_raw = next((r["cum_disp_raw"] for r in disp_table["C"]
                       if r["step"] == t_kill), None) if t_kill else None

    # ---- G-PIN: both conventions; primary = rms-carried
    G_PIN = {"per_arm": {}, "primary": "rms_carried",
             "fuzz_rms_carried_raw": PIN_FUZZ_RMS_CARRIED,
             "fuzz_verbatim": G1.PIN_FUZZ_BAR,
             "one_step_fuzz_raw": ONE_STEP_FUZZ_RAW}
    for tag in ("W1", "W2", "W3"):
        R = arms[tag]["wall_R"]
        rows = [r for r in disp_table[tag] if "g_m12_light" in r]
        mx = max(r["cum_disp_raw"] for r in rows) if rows else None
        G_PIN["per_arm"][tag] = {
            "R_raw": R, "R_rms": R / SQRT_P,
            "bound_rms_carried": R + PIN_FUZZ_RMS_CARRIED,
            "bound_verbatim": R + G1.PIN_FUZZ_BAR,
            "max_raw_disp_at_ckpt": mx,
            "per_ckpt_raw": {r["step"]: r["cum_disp_raw"] for r in rows},
            "per_ckpt_rms": {r["step"]: r["cum_disp_rms"] for r in rows},
            "pass_rms_carried": bool(
                mx is not None and mx <= R + PIN_FUZZ_RMS_CARRIED),
            "pass_verbatim": bool(
                mx is not None and mx <= R + G1.PIN_FUZZ_BAR),
        }
        log(f"G-PIN[{tag}]: max raw |d| {fmt(mx)} <= "
            f"{R + PIN_FUZZ_RMS_CARRIED:.3f} (rms-carried): "
            f"{'PASS' if G_PIN['per_arm'][tag]['pass_rms_carried'] else 'FAIL'}"
            f" | verbatim R+1.5 = {R + G1.PIN_FUZZ_BAR:.2f}: "
            f"{'PASS' if G_PIN['per_arm'][tag]['pass_verbatim'] else 'FAIL'}"
            f" (co-reported)")
    G_PIN["pass"] = bool(all(v["pass_rms_carried"]
                             for v in G_PIN["per_arm"].values()))

    # ---- the adjudicating gate set: G-ROOT IS NOT A MEMBER (the registered
    # deviation); G-ROOTLOAD (the load's identity) stands in its place
    gates_pass = bool(G_CONFIG["pass"] and G_BASE["pass"]
                      and G_BASE_QUAL["pass"] and G_ROOTLOAD["pass"]
                      and G_CTRL["pass"] and G_PIN["pass"]
                      and G_INPUTS["pass"] and G_STEP1["pass"]
                      and all(v["pass"] for v in G_BITROOT.values())
                      and G_INST["pass"])

    # ---- the ladder verdicts against the frozen scaled bar
    def arm_verdict(tag):
        g = light_gm12(tag)
        vals = [g[s] for s in CK_WASH if s in g]
        complete = bool(len(vals) == len(CK_WASH))   # anti-erosion: a
        # truncated arm (missing checkpoints) cannot hold a bar
        holds_bar = bool(complete and vals
                         and all(v >= bar_0p9eq for v in vals))
        maintains = bool(complete and vals
                         and all(v >= G1.MAINTAIN_BAR for v in vals))
        dies_by_50 = bool(g.get(50, 1.0) <= G1.SHUT_BAR)
        flat_phase = [g[s] for s in (10, 50, 100, 200, 300) if s in g]
        return {
            "g_m12": g,
            "all_checkpoints_present": complete,
            "missing": [s for s in CK_WASH if s not in g],
            "min_gm12": min(vals) if vals else None,
            "argmin_step": (min(g, key=lambda s: g[s]) if vals else None),
            "holds_0p9eq_bar": holds_bar,
            "first_ck_below_bar": next((s for s in CK_WASH
                                        if g.get(s, 1.0) < bar_0p9eq), None),
            "maintains_g1_bar": maintains, "dies_by_50": dies_by_50,
            "retention_min": (min(v / root_cells["gm12"] for v in vals)
                              if vals else None),
            "flat_phase_min": (min(flat_phase) if flat_phase else None),
            "flat_phase_retention_min": (
                min(flat_phase) / root_cells["gm12"]
                if flat_phase else None),
            "flat_phase_holds_09x_root": bool(
                flat_phase and min(flat_phase) >= 0.9 * root_cells["gm12"]),
        }

    ladder = {tag: arm_verdict(tag) for tag in ("C", "W1", "W2", "W3")}

    # ---- COSTS: the tax curve (the +0.53 reference re-priced)
    def ce_at(tag, step):
        return next((t["ce_batch"] for t in arms[tag]["traj"]
                     if t["step"] == step), None)

    ce_tax = {
        "W1_ce300": ce_at("W1", CK_WASH[-1]), "C_ce300": ce_at("C", CK_WASH[-1]),
        "W2_ce300": ce_at("W2", CK_WASH[-1]), "W3_ce300": ce_at("W3", CK_WASH[-1]),
        "root_ce_r": root_cells["ce_r"],
        "reference_274M": PRIORS_G1B["wall_tax_274M"],
    }
    ce_tax["W1_minus_C"] = (ce_tax["W1_ce300"] - ce_tax["C_ce300"]) \
        if None not in (ce_tax["W1_ce300"], ce_tax["C_ce300"]) else None
    ce_tax["W1_minus_root"] = (ce_tax["W1_ce300"] - ce_tax["root_ce_r"]) \
        if ce_tax["W1_ce300"] is not None else None
    ce_tax["per_step_curve"] = {
        tag: {t["step"]: t["ce_batch"] for t in arms[tag]["traj"]}
        for tag in arms}
    FREEZING_CE = bool(ce_tax["W1_ce300"] is not None
                       and ce_tax["W1_ce300"] >= ce_tax["root_ce_r"])

    # ---- the composed verdict (the frozen three; no shopping) -------------
    WALL_SCALES_G = WALL_TIGHTENS_G = WALL_FADES_G = None
    deviation_stamp = (f"[adjudicated at the SUB-BAR peak root "
                       f"{root_cells['gm12']:.4f} — "
                       f"{G1.EXPRESS_BAR - root_cells['gm12']:.4f} below the "
                       f"0.78 express bar; the registered pre-run deviation; "
                       f"the scaled bar {bar_0p9eq:.4f} = 0.983x THIS root]")
    if not gates_pass:
        failed = [k for k, g in (("G-CONFIG", G_CONFIG), ("G-BASE", G_BASE),
                                 ("G-BASE-QUAL", G_BASE_QUAL),
                                 ("G-INST", G_INST),
                                 ("G-ROOTLOAD", G_ROOTLOAD),
                                 ("G-CTRL", G_CTRL), ("G-PIN", G_PIN),
                                 ("G-INPUTS", G_INPUTS),
                                 ("G-STEP1", G_STEP1),
                                 ("G-BITROOT", {"pass": all(
                                     v["pass"] for v in G_BITROOT.values())})
                                 ) if not g["pass"]]
        verdict = f"TEXTURE (GATE FAILURE: {', '.join(failed)})"
        clause = ("a registered gate failed — nothing adjudicated; failed: "
                  f"{failed}; full record reported (root g-12 "
                  f"{fmt(root_cells['gm12'])}, C trace "
                  + " -> ".join(f"+{s}:{fmt(v)}" for s, v in
                                sorted(c_gm12.items())) + ") "
                  + deviation_stamp)
    else:
        w1_holds = ladder["W1"]["holds_0p9eq_bar"]
        looser_hold = [t for t in ("W2", "W3") if ladder[t]["holds_0p9eq_bar"]]
        WALL_SCALES = bool(w1_holds and G_CTRL["pass"])
        WALL_TIGHTENS = bool(looser_hold and G_CTRL["pass"])
        WALL_FADES = bool(G_CTRL["pass"] and not WALL_SCALES
                          and (not any(ladder[t]["holds_0p9eq_bar"]
                                       for t in ("W1", "W2", "W3"))
                               or (w1_holds and FREEZING_CE)))
        if WALL_SCALES and WALL_TIGHTENS:
            verdict = "WALL-SCALES + WALL-TIGHTENS"
            clause = (f"at the rms-matched rung (R = {R_LADDER_RAW[0]:.4f} raw "
                      f"= {R_LADDER_RMS[0]:.4e} rms) the fact HELD the "
                      f"0.9-equivalent bar ({bar_0p9eq:.4f} = "
                      f"{0.9 / G1B_ROOT_GM12:.4f} x root) at every checkpoint "
                      f"through +{CK_WASH[-1]} (min "
                      f"{fmt(ladder['W1']['min_gm12'])} at "
                      f"+{ladder['W1']['argmin_step']}) while C died by +50 "
                      f"(dead at +{t_kill}) — AND looser rungs also held "
                      f"({','.join(looser_hold)}): the basin GREW with "
                      f"dimension; tax curve re-priced: W1-C dCE@300 "
                      f"{fmt(ce_tax['W1_minus_C'])} vs the 2.74M +0.53 "
                      f"reference. " + deviation_stamp)
        elif WALL_SCALES:
            verdict = "WALL-SCALES"
            clause = (f"at the rms-matched rung (R = {R_LADDER_RAW[0]:.4f} raw "
                      f"= {R_LADDER_RMS[0]:.4e} rms) the fact HELD the "
                      f"0.9-equivalent bar ({bar_0p9eq:.4f}) at every "
                      f"checkpoint through +{CK_WASH[-1]} (min "
                      f"{fmt(ladder['W1']['min_gm12'])} at "
                      f"+{ladder['W1']['argmin_step']}, retention "
                      f"{fmt(ladder['W1']['retention_min'])}x root) while C "
                      f"died by +50 (dead at +{t_kill}, D_kill "
                      f"{fmt(D_kill_raw)} raw = "
                      f"{D_kill_raw / SQRT_P if D_kill_raw else float('nan'):.4e}"
                      f" rms vs the 2.74M {PRIORS_G1B['D_kill_rms_274M']:.4e})"
                      f" — the wall generalizes at 10x with the dial carried "
                      f"in per-coordinate RMS units; tax: W1-C dCE@300 "
                      f"{fmt(ce_tax['W1_minus_C'])} (2.74M reference +0.53); "
                      f"looser rungs did NOT hold the strict bar ("
                      + " | ".join(f"{t}: min {fmt(ladder[t]['min_gm12'])}, "
                                   f"first below bar +"
                                   f"{ladder[t]['first_ck_below_bar']}"
                                   for t in ("W2", "W3")) + "). "
                      + deviation_stamp)
        elif WALL_TIGHTENS:
            verdict = "WALL-TIGHTENS"
            clause = (f"C died by +50 (dead at +{t_kill}) and the rms-matched "
                      f"rung did NOT hold the strict bar (W1 min "
                      f"{fmt(ladder['W1']['min_gm12'])} at "
                      f"+{ladder['W1']['argmin_step']}, first below "
                      f"+{ladder['W1']['first_ck_below_bar']}) but a LOOSER "
                      f"rung did ({','.join(looser_hold)}) — the basin grew "
                      f"with dimension; tax curve: W1-C dCE@300 "
                      f"{fmt(ce_tax['W1_minus_C'])} vs +0.53; full ladder "
                      "traces reported. " + deviation_stamp)
        elif WALL_FADES:
            if w1_holds and FREEZING_CE:
                verdict = "WALL-FADES (only the tightest R holds, CE frozen)"
                clause = (f"only W1 held the bar and the wall FROZE adaptation "
                          f"(W1 CE@300 {fmt(ce_tax['W1_ce300'])} >= root CE_R "
                          f"{fmt(ce_tax['root_ce_r'])}); C died at +{t_kill}. "
                          + deviation_stamp)
            else:
                verdict = "WALL-FADES"
                clause = (f"C died by +50 (dead at +{t_kill}) but NO rung on "
                          f"the RMS ladder held the 0.9-equivalent bar "
                          f"({bar_0p9eq:.4f} = "
                          f"{0.9 / G1B_ROOT_GM12:.4f} x root): "
                          + " | ".join(
                              f"{t} ({R_LADDER_RAW[i]:.3f} raw = "
                              f"{R_LADDER_MULT[i]:.0f}x rms): min "
                              f"{fmt(ladder[t]['min_gm12'])} at +"
                              f"{ladder[t]['argmin_step']}, first below bar "
                              f"+{ladder[t]['first_ck_below_bar']}, g1-verbatim "
                              f"maintains={ladder[t]['maintains_g1_bar']}, "
                              f"flat-phase retention "
                              f"{fmt(ladder[t]['flat_phase_retention_min'])}"
                              for i, t in enumerate(("W1", "W2", "W3")))
                          + f" — honest verdict per the frozen bars, no dial "
                          f"search; the flat-phase retention and g1-verbatim "
                          f"maintain bars are co-reported for the "
                          f"raw-vs-rms-geometry read. " + deviation_stamp)
        else:
            verdict = "TEXTURE"
            clause = ("no clause composed cleanly (gates passed; report the "
                      "full ladder traces) " + deviation_stamp)
        WALL_SCALES_G, WALL_TIGHTENS_G, WALL_FADES_G = \
            WALL_SCALES, WALL_TIGHTENS, WALL_FADES

    log("=" * 78)
    log(f"G1BS6 VERDICT: {verdict}")
    for tag in ("C", "W1", "W2", "W3"):
        g = ladder[tag]["g_m12"]
        log(f"  {tag}: g-12 " + " -> ".join(f"+{s}:{v:.4f}"
                                            for s, v in sorted(g.items())))
        log(f"  {tag}: |d| raw/rms " + " -> ".join(
            f"+{r['step']}:{r['cum_disp_raw']:.3f}/{r['cum_disp_rms']:.2e}"
            for r in disp_table[tag] if "g_m12_light" in r))
    log(f"  bar_0p9eq {bar_0p9eq:.4f} | root {root_cells['gm12']:.4f} "
        f"(0.78-express {'PASS' if G_ROOT_read['pass'] else 'below bar'}) | "
        f"D_kill raw {fmt(D_kill_raw)}")
    log(f"  tax: W1-C dCE@300 {fmt(ce_tax['W1_minus_C'])} (ref +0.53) | "
        f"W1-root {fmt(ce_tax['W1_minus_root'])} | freezing {FREEZING_CE}")
    log(f"  {clause}")
    log("=" * 78)

    # =====================================================================
    # OUTPUTS
    # =====================================================================
    trace = {}
    for tag in arms:
        rows = [{"freeze_steps": 0,
                 **{k: root_cells[k] for k in
                    ("gm12", "g0", "gp12", "held30_gm12", "held30_g0",
                     "ce_r", "site_read_onset", "site_read_span")}}]
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
        "experiment": "g1bS6_peak_wall",
        "date": common.now_iso(),
        "partial": False,
        "progressive_writes": _progressive["n"],
        "phases": list(_progressive["phases"]),
        "design": ("scratch/g1bS_design.md (convention frozen at dispatch; "
                   "bars verbatim; no bar shopping) + the dispatch's "
                   "REGISTERED gate deviation (the arms run on g1bS5's "
                   "sub-bar peak root; the scaled bar adjudicates)"),
        "question": ("does the commit-and-project L2 ball hold a "
                     "consolidated fact through a wash that kills the "
                     "control, at ~10x the parameters (9,977,600 vs the "
                     "2,739,072 every g-law was minted on), with the R-dial "
                     "carried in per-coordinate RMS units — TAKE 5: the "
                     "arms on g1bS5's PEAK ROOT (0.20 rms formation "
                     "optimum, root g-12 0.7677), the actual scale "
                     "adjudication?"),
        "registered": REGISTERED,
        "provenance": {
            "host": {"config": {"n_layer": 8, "n_head": 8, "n_embd": 320,
                                "block_size": 256, "vocab": 65},
                     "params": n_params, "band": "[8.5M, 11.5M]",
                     "config_provenance": "e005s LARGE's 10M-class config "
                                          "(the lineage's standing host)",
                     "host_seed": HOST_SEED, "corpus_seed": 1337,
                     "base": ("g1bS2's PASSED base LOADED VERBATIM from "
                              f"runs/checkpoints/{SRC_BASE_CK} (never "
                              "retrained; val-min-anchored 1200-step cosine, "
                              "lr 4e-4; G-BASE-QUAL re-derived on the load)"),
                     "final_val_loss": G_BASE["final_val_loss"]},
            "install": {"steps": INSTALL_STEPS,
                        "lr": 4e-4,
                        "seed": G1.INSTALL_SEED,
                        "source": ("g1bS2's movement-matched install LOADED "
                                   f"VERBATIM from runs/checkpoints/"
                                   f"{SRC_INSTALL_CK} (post-install state "
                                   "REUSABLE; cells re-measured and "
                                   "cross-checked by this run)"),
                        "post_install_cells": inst_cells},
            "root": {"source": ("g1bS5's m020 formation-curve PEAK root, "
                                f"LOADED VERBATIM from runs/checkpoints/"
                                f"{SRC_ROOT_CK} (base+install LOADED + "
                                "e113 jitter consolidation s500 @ 4e-4 = "
                                "0.20 rms, seed 10901) — theta0 for every "
                                "g1bS6 arm; identity-gated by G-ROOTLOAD "
                                "vs g1bS5's committed record"),
                     "movement_rms": G1BS5_PEAK_ROOT["movement_rms"],
                     "steps": G1BS5_PEAK_ROOT["steps"],
                     "cons_seed": G1BS5_PEAK_ROOT["cons_seed"],
                     "root_read": G_ROOT_read,
                     "root_cells": root_cells},
            "wash": {"recipe": REGISTERED["wash"]["recipe"],
                     "seed": WASH_SEED, "ckpt_steps": list(CK_WASH),
                     "devices": {t: arms[t]["device"] for t in arms},
                     "n_chunks": {t: arms[t].get("n_chunks") for t in arms},
                     "chunk_tables": {t: arms[t].get("chunk_table")
                                      for t in arms},
                     "runtimes_s": {t: round(
                         arms[t]["traj"][-1]["elapsed_s"], 1) for t in arms}},
            "R_convention": REGISTERED["R_convention"],
            "owner_envelope": {
                "launch_gate": REGISTERED["owner_envelope"]["launch_gate"],
                "burst_cap_s": TRAIN_CAP_GPU, "cooldown_s": COOLDOWN_S,
                "events": device_events},
            "seeds": {"host_init_base_corpus": HOST_SEED,
                      "install": G1.INSTALL_SEED,
                      "consolidation": G1BS5_PEAK_ROOT["cons_seed"],
                      "wash": WASH_SEED, "protocol_corpus": 1337},
        },
        "scaled_bar": SCALED_BAR,
        "priors_g1b": PRIORS_G1B,
        "priors_g1bs4_ladder_weak_root": PRIORS_G1BS4_LADDER,
        "priors_g1bs5_peak_root": G1BS5_PEAK_ROOT,
        "arms": {
            tag: {"desc": desc, "R_raw": R,
                  "R_rms": (R / SQRT_P if R else None),
                  "R_mult": (R / (R_RMS_REF * SQRT_P) if R else None),
                  "ckpt_steps": list(CK_WASH),
                  "steps_ran": arms[tag]["steps_ran"],
                  "device": arms[tag]["device"],
                  "n_chunks": arms[tag].get("n_chunks"),
                  "chunk_table": arms[tag].get("chunk_table"),
                  "traj": [{k: v for k, v in t.items() if k != "d_proj"}
                           for t in arms[tag]["traj"]],
                  "missing_checkpoints": [s for s in CK_WASH
                                          if s not in arms[tag]["sds"]]}
            for i, (tag, R, *rest) in enumerate(ARM_SPECS)
            for desc in [rest[-1]]},
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": G1.PRE, "post_cap":
                     G1.POST_CAP, "neutral_bank": G_ANCHOR["neutral_bank"],
                     "measure_dial": "g1b's measure() (the e131 dial set) on "
                                     "evl_load (settle+disarm PIVOT), "
                                     "CPU-only"},
        "displacement": {
            "currency": ("cumulative ||theta_t - theta_0||_2 over all "
                         f"{n_params} trainable parameters (fp32, "
                         "device-resident per step — registered scale "
                         "adaptation) in BOTH conventions (raw L2 and "
                         "per-coordinate rms = raw/sqrt(P)) + per-step "
                         "increments + projected displacement min(d, R) + "
                         "cosines vs arm C"),
            "table": disp_table,
            "theta0_norm": {t: arms[t]["theta0_norm"] for t in arms},
            "wall_fuzz_registered": {
                "one_step_lr_sqrtP_raw": ONE_STEP_FUZZ_RAW,
                "pin_fuzz_rms_carried_raw": PIN_FUZZ_RMS_CARRIED,
                "note": "the frozen 1.5 fuzz constant carried in the rms "
                        "convention (primary; g1bS4's committed convention); "
                        "the verbatim R+1.5 form is co-reported per arm and "
                        "cannot hold at 10M (one AdamW step = 3.159 raw > "
                        "any R+1.5 on the ladder)"},
        },
        "gates": {"G_CONFIG": G_CONFIG, "G_BASE": G_BASE,
                  "G_BASE_QUAL": G_BASE_QUAL, "G_INST": G_INST,
                  "G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                  "G_POOL": G_POOL, "G_ANCHOR": G_ANCHOR,
                  "G_ROOTLOAD": G_ROOTLOAD, "G_ROOT_read_NOT_A_GATE":
                  G_ROOT_read, "G_CTRL": G_CTRL, "G_PIN": G_PIN,
                  "G_BITROOT": G_BITROOT,
                  "G_INPUTS": {k: v for k, v in G_INPUTS.items()
                               if k != "per_step"} | {"per_step": {
                                   s: v["identical"] for s, v in
                                   G_INPUTS["per_step"].items()}},
                  "G_STEP1": G_STEP1, "G_SURG": gates_surg,
                  "note": ("G-ROOT is NOT in the adjudicating set (the "
                           "REGISTERED pre-run deviation: the arms run on "
                           "the sub-bar peak root); G-ROOTLOAD (the loaded "
                           "checkpoint's identity vs g1bS5's committed "
                           "record) stands in as the load gate")},
        "traces": trace,
        "batteries": batteries_all,
        "adjudication": {
            "bars_verbatim": REGISTERED["bars_verbatim"],
            "order": "GATES -> WALL ladder vs the scaled bar -> COSTS",
            "gates_pass": gates_pass,
            "gate_deviation": REGISTERED["gate_deviation"],
            "ladder": ladder,
            "bar_0p9eq": bar_0p9eq,
            "WALL_SCALES": (WALL_SCALES_G if gates_pass else None),
            "WALL_TIGHTENS": (WALL_TIGHTENS_G if gates_pass else None),
            "WALL_FADES": (WALL_FADES_G if gates_pass else None),
            "D_kill_raw": D_kill_raw,
            "D_kill_rms": (D_kill_raw / SQRT_P if D_kill_raw else None),
            "D_kill_rms_274M_prior": PRIORS_G1B["D_kill_rms_274M"],
            "ce_tax": ce_tax, "freezing_CE": FREEZING_CE,
            "verdict": verdict, "clause": clause,
        },
        "honesty_reflex": {
            "n1_scope": ("n=1 host, one wash seed (10902), one fact — the "
                         "rung is SCALE, not seeds; the replicate ladder "
                         "follows only if a bar fires (the design's own "
                         "first-cell scoping)."),
            "sub_bar_root": ("THE REGISTERED DEVIATION is disclosed on every "
                             "artifact: the root is "
                             f"{root_cells['gm12']:.4f}, 0.012 below the "
                             "0.78 express bar; the bars adjudicate against "
                             "the root-strength-relative scaled bar "
                             "(0.983x THIS root), the form every prior take "
                             "committed; the absolute-bar comparison is "
                             "NOT re-run post hoc (no bar shopping)"),
            "intervention_not_logits": ("the wall IS the intervention: all "
                "four arms share bit-identical step-0 weights (G-BITROOT "
                "max|diff| = 0.0), bit-identical per-step inputs (md5-gated, "
                "G-INPUTS) and identical targets; the only delta is the "
                "commit radius, and G-STEP1 shows the commit is inert until "
                "forward 2. G-PIN verifies the ball's mechanics in both "
                "conventions BEFORE any behavioral clause is read."),
            "scaled_bar_strictness": ("the primary bar (0.983x root at every "
                "checkpoint) is STRICTER than g1's maintain bar by design; "
                "g1b's own W1 would have failed it at +2/+4 (retention "
                "0.848/0.893) — the co-reported flat-phase retention and "
                "g1-verbatim maintain/dies bars carry the dip structure so "
                "the fold can separate 'the dip' from 'the fade'."),
            "device_texture": ("the 2.74M references (root 0.9156, +0.53 "
                "tax) were CPU/mixed-device runs; this cell is GPU-trained "
                "with CPU full dials; every bar adjudicates against the "
                "in-run control C, and cross-device float fuzz is bounded "
                "by G-STEP1 (~1e-5 measured on same-stream step-1 bodies) "
                "and G-ROOTLOAD (~0 on the loaded-root dual read)."),
            "wall_blind_spot": ("the wall never protects the FIRST step: "
                "commit happens at d=0, so step 1's AdamW update (~3.16 raw "
                "= lr*sqrt(P)) always lands before the first projection; "
                "W3's ball (5.345 raw) even contains step 1 unfree — its "
                "+1 read necessarily tracks C's. The wall caps cumulative "
                "displacement at R + fuzz — the design's registered "
                "semantics, not a bug."),
            "cpu_gpu_tax_caveat": ("the tax comparison to +0.53 carries the "
                "device-texture caveat (g1b's dCE was measured on its own "
                "device mix); the in-run W1-vs-C delta is the primary "
                "number."),
            "no_bar_shopping": ("the ladder was fixed by the frozen design; "
                "if no rung holds, WALL-FADES is the verdict; the secondary "
                "bars are co-reported context, never adjudicated."),
        },
        "trims": trims, "deviations": deviations,
        "device_events": device_events,
        "ckpt_inventory": CKPT_INVENTORY,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 8, "n_head": 8, "n_embd": 320,
                   "block_size": 256, "params": n_params,
                   "R_ladder_raw": list(R_LADDER_RAW),
                   "R_ladder_rms": list(R_LADDER_RMS),
                   "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "scale_wall.png", trace, disp_table, ladder, verdict, clause,
         gates_pass, bar_0p9eq, root_cells, ce_tax, D_kill_raw)

    log(f"outputs: {rd / 'metrics.json'}, {rd / 'scale_wall.png'}, "
        f"{len(CKPT_INVENTORY)} checkpoints in runs/checkpoints/g1bS6_*.pt")
    log(f"VERDICT: {verdict}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ driver

def g1bS6_wash(tag: str, net0, anchor: torch.Tensor, train_ids: torch.Tensor,
               itos, r_eval_xy, gm12_ids, g0_ids, zid: int,
               resume_ck: Path | None = None,
               lr: float = G1.FT_LR, seed: int = None) -> dict:
    """THE NEUTRAL PLAIN-CORPUS WASH — g1bS4_wash VERBATIM arithmetic
    (g1_wash = e185's noise_wash = e176n's finetune_freeze; draw order
    aj/rj, batch 32 = 16 neutral + 16 random, full-token CE, AdamW
    (0.9,0.95) wd 0.1 const lr, clip 1.0, wall bookkeeping, in-batch CE
    every step, ckpt snapshots + light evals, x-hash gate) + the
    OWNER-ENVELOPE adaptations ONLY (device policy; no training-semantics
    change): the launch gate is util<=20% AND temp<=70C double-polled
    (wait_gpu_owner), the burst cap is 90 s (TRAIN_CAP_GPU), the
    between-chunk cooldown is 180 s, outside load mid-burst =>
    PAUSE-AND-WAIT (never migrate), no CPU fallback. CHUNK-RESUMABLE (the
    90 s cap made mandatory what g1bS4 trimmed): full-state resume ckpt
    ({model, opt, gen_state, step, traj, sds, deltas, x_hashes}) at every
    burst boundary — bit-identical to an uninterrupted run because the
    generator + optimizer state carry the whole stream."""
    seed = WASH_SEED if seed is None else seed
    ckpt_steps = CK_WASH
    ckpt_set = set(ckpt_steps)
    n_steps = ckpt_steps[-1]
    n_anc = anchor.shape[0]
    vocab = len(itos)
    step, traj, sds, x_hashes, deltas = 0, [], {}, {}, {}
    zeph_checks = 0
    net, opt, gen = None, None, None
    theta0 = None
    wall_R = getattr(net0, "R", None)
    n_chunks, chunk_table, devices = 0, [], []
    active_s = 0.0
    # edge: a prior process saved the FINAL state but died before returning
    if resume_ck is not None and Path(resume_ck).exists():
        _pre = torch.load(resume_ck, map_location="cpu", weights_only=False)
        if int(_pre.get("step", 0)) >= n_steps:
            log(f"  [{tag}] resume ckpt already COMPLETE at s{_pre['step']} — "
                f"returning the saved final state (no steps re-run)")
            return {"sds": _pre["sds"], "traj": _pre.get("traj", []),
                    "steps_ran": n_steps, "seed": seed, "lr": lr,
                    "target_mode": "true",
                    "zeph_violations": _pre.get("zeph", 0),
                    "x_hashes": _pre.get("x_hashes", {}),
                    "deltas": _pre.get("deltas", {}),
                    "wall_R": _pre.get("wall_R", wall_R),
                    "theta0_norm": float(torch.norm(flat_params(net0))),
                    "device": "resumed (final state)",
                    "n_chunks": _pre.get("n_chunks", 0),
                    "chunk_table": _pre.get("chunk_table", [])}
    while step < n_steps:
        n_chunks += 1
        dev = wait_gpu_owner(f"{tag}-chunk{n_chunks}")   # OWNER envelope
        cap = TRAIN_CAP_GPU                             # <= 90 s, always
        if net is None:
            net = copy.deepcopy(net0).to(dev)
            net.train()
            opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95),
                                    weight_decay=0.1)
            gen = torch.Generator().manual_seed(seed)
            theta0 = flat_params(net)          # displacement origin (device)
            prev = theta0.clone()
            if resume_ck is not None and Path(resume_ck).exists():
                state = torch.load(resume_ck, map_location="cpu",
                                   weights_only=False)
                net.load_state_dict(state["model"])
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
                prev = flat_params(net)        # increments resume cleanly
                log(f"  [{tag}] RESUMED from {Path(resume_ck).name} at step "
                    f"{step}/{n_steps} ({len(traj)} traj rows, "
                    f"{len(sds)} ckpts carried)")
        else:
            _pd = str(next(net.parameters()).device)
            if _pd != str(dev):
                _osd = opt.state_dict()
                net = net.to(dev)
                opt = torch.optim.AdamW(net.parameters(), lr=lr,
                                        betas=(0.9, 0.95), weight_decay=0.1)
                opt.load_state_dict(_osd)
                log(f"  [{tag}] chunk {n_chunks}: device moved {_pd} -> "
                    f"{dev} (optimizer state recast)")
            net.train()
            prev = flat_params(net)
        devices.append(str(dev))
        evl = copy.deepcopy(net0).to(dev)      # ARMED eval twin on the device
        t_start = time.time()
        chunk_capped = False
        for step in range(step + 1, n_steps + 1):
            aj = torch.randint(n_anc, (G1.ANCH_BS,), generator=gen)
            rj = torch.randint(len(train_ids) - G1.BLOCK - 1, (G1.RAND_BS,),
                               generator=gen)
            anc = anchor[aj]
            rnd = torch.stack([train_ids[s: s + G1.BLOCK] for s in rj])
            # name-free VERIFY (no-op by corpus construction; hard-fail if not)
            for w in rnd:
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
            cur = flat_params(net)
            cum_disp = float(torch.norm(cur - theta0))
            inc_disp = (float(torch.norm(cur - prev)) if prev is not None
                        else None)
            prev = cur
            if step in ckpt_set:
                deltas[step] = (cur - theta0).cpu()
            row = {"step": step, "ce_batch": float(loss.item()),
                   "cum_disp": cum_disp, "cum_disp_rms": cum_disp / SQRT_P,
                   "step_disp": inc_disp,
                   "d_proj": min(cum_disp, wall_R) if wall_R else None,
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
                # BOTH CONVENTIONS at every checkpoint (the frozen co-report)
                wr = net.wall_report()
                log(f"  [{tag}] CKPT +{step:4d} g-12 {gz['mean_pz']:.4f} "
                    f"g0 {gz0['mean_pz']:.4f} CE_R {ce_r:.4f} | "
                    f"|d| {cum_disp:.4f} raw = {cum_disp / SQRT_P:.4e} rms "
                    + (f"| R {wall_R:.4f} raw = {wall_R / SQRT_P:.4e} rms "
                       f"(d_proj {wr['d_proj']})" if wall_R else "(uncommitted)")
                    + f" (CE {float(loss.item()):.4f})")
            traj.append(row)
            if step % 50 == 0 and step not in ckpt_set:
                log(f"  [{tag}] s{step:4d} CE {float(loss.item()):.4f} "
                    f"|d| {cum_disp:.4f} ({row['elapsed_s']:.0f}s)")
            if (time.time() - t_start) > cap:
                log(f"  [{tag}] chunk {n_chunks}: owner-envelope burst cap "
                    f"{cap:.0f}s at s{step} — resume ckpt saved; next chunk "
                    f"continues after the cooldown")
                chunk_capped = True
                break
            if dev.type == "cuda" and step % G1.MIDRUN_POLL_EVERY == 0:
                s = gpu_status()
                if s["mem_total"] > 0 and (s["mem_used"] > 0.85 * s["mem_total"]
                                           or s["temp"] > 80):
                    device_events.append({"tag": tag, "step": step,
                                          "event": "MID-RUN PAUSE (no "
                                                   "migration)", "status": s})
                    t_p = time.time()
                    midrun_pause_wait(tag)
                    t_start += time.time() - t_p     # paused time is not cap
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
                        "n_chunks": n_chunks, "chunk_table": chunk_table},
                       resume_ck)
        if step >= n_steps:
            break
        if n_chunks >= 24:                # pathological-cap guard (~2h of
            trims.append(f"{tag}: stopped at chunk {n_chunks} (cap-loop "
                         f"guard) at step {step} of {n_steps}")
            break                          # chunks; documented, not silent)
        if not SMOKE:
            cooldown(COOLDOWN_S)          # >= 180 s between bursts (owner)
        del evl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    net.eval()
    return {"sds": sds, "traj": traj, "steps_ran": step,
            "seed": seed, "lr": lr, "target_mode": "true",
            "zeph_violations": zeph_checks, "x_hashes": x_hashes,
            "deltas": deltas, "wall_R": wall_R,
            "theta0_norm": float(torch.norm(theta0.cpu())),
            "device": devices[0] if devices else "n/a", "devices": devices,
            "n_chunks": n_chunks, "chunk_table": chunk_table}


CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "g1bS6", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name}")


# ------------------------------------------------------------------ plot
# THE SCALE WALL figure: fact survival vs steps (the ladder vs C), the
# R-dial (rms axis), the TAX CURVE, displacement vs the walls (both
# conventions), the retention panel, the verdict (+ the deviation stamp).

def plot(path, trace, disp_table, ladder, verdict, clause, gates_pass,
         bar_0p9eq, root_cells, ce_tax, D_kill_raw):
    import textwrap
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.5))
    cols = {"C": "crimson", "W1": "seagreen", "W2": "royalblue",
            "W3": "darkorange"}
    lbls = {"C": "C — no wall (control)",
            "W1": f"W1 — 1x R_rms ({R_LADDER_RAW[0]:.3f} raw)",
            "W2": f"W2 — 2x R_rms ({R_LADDER_RAW[1]:.3f} raw)",
            "W3": f"W3 — 4x R_rms ({R_LADDER_RAW[2]:.3f} raw)"}
    marks = {"C": "o", "W1": "s", "W2": "^", "W3": "v"}
    dev_stamp = (f"root {root_cells['gm12']:.4f} = "
                 f"{G1.EXPRESS_BAR - root_cells['gm12']:.4f} below the 0.78 "
                 f"express bar (REGISTERED deviation: arms run)")

    def series(tag):
        pts = [(r["freeze_steps"], r["gm12"]) for r in trace[tag]]
        return [p[0] for p in pts], [p[1] for p in pts]

    # (0,0) THE HEADLINE: g-12 vs wash steps
    ax = axes[0, 0]
    for tag in ("C", "W1", "W2", "W3"):
        xs, ys = series(tag)
        ax.plot(xs, ys, marks[tag] + "-", ms=8, lw=2.2, color=cols[tag],
                alpha=0.9, label=lbls[tag])
    ax.axhline(bar_0p9eq, ls="--", lw=1.6, color="black", alpha=0.85,
               label=f"0.9-equivalent bar {bar_0p9eq:.3f}")
    ax.axhline(0.9 * root_cells["gm12"], ls="-.", lw=1.1, color="gray",
               alpha=0.8, label=f"0.9 x root {0.9 * root_cells['gm12']:.3f}")
    for yv, col in ((G1.MAINTAIN_BAR, "seagreen"), (G1.SHUT_BAR, "tab:purple")):
        ax.axhline(yv, ls=":", lw=1.0, color=col, alpha=0.7)
    ax.annotate(f"root {root_cells['gm12']:.3f}\n(0.78-express: below)",
                (0, root_cells["gm12"]),
                textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    ax.set_xlabel("neutral-wash steps from the committed peak root")
    ax.set_ylabel("g-12 (mean p(Z), install-60 battery)")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7.5, loc="center right")
    ax.set_title(f"THE WALL AT 10x, ADJUDICATED — fact survival "
                 f"({G1BS_PARAMS:,} params)", fontsize=10)

    # (0,1) THE R-DIAL (rms axis): survival at three horizons
    ax = axes[0, 1]
    rms_pts = [0.0] + list(R_LADDER_RMS)
    tags_by_R = ["C", "W1", "W2", "W3"]
    for step, col, mk in ((2, "gray", "o"), (50, "tab:purple", "s"),
                          (300, "seagreen", "D")):
        ys = [ladder[t]["g_m12"].get(step) for t in tags_by_R]
        ax.plot([r / R_RMS_REF for r in rms_pts], ys, mk + "-", ms=9, lw=2.0,
                color=col, label=f"g-12 at +{step}")
    ax.axhline(bar_0p9eq, ls="--", lw=1.6, color="black", alpha=0.85,
               label=f"0.9eq bar {bar_0p9eq:.3f}")
    for yv, col in ((G1.MAINTAIN_BAR, "seagreen"), (G1.SHUT_BAR, "tab:purple")):
        ax.axhline(yv, ls=":", lw=1.0, color=col, alpha=0.7)
    ax.set_xticks([r / R_RMS_REF for r in rms_pts])
    ax.set_xticklabels(["C\n(no wall)"] + [f"{m:.0f}x R_rms\n({r:.3f} raw)"
                                           for m, r in zip(R_LADDER_MULT,
                                                           R_LADDER_RAW)])
    ax.set_xlabel(f"the wall dial (rms units; R_rms = {R_RMS_REF:.2e})")
    ax.set_ylabel("g-12")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7, loc="center right")
    ax.set_title("THE R-DIAL in rms units (both conventions on the ticks)",
                 fontsize=10)

    # (0,2) THE TAX CURVE
    ax = axes[0, 2]
    for tag in ("C", "W1", "W2", "W3"):
        rows = disp_table[tag]
        ax.plot([r["step"] for r in rows], [r["ce_batch"] for r in rows],
                "-", lw=1.6, color=cols[tag], alpha=0.8,
                label=f"{tag} in-batch CE")
    ax.axhline(ce_tax["root_ce_r"], ls=":", lw=1.3, color="black",
               alpha=0.7, label=f"root CE_R {ce_tax['root_ce_r']:.2f}")
    if ce_tax["W1_minus_C"] is not None:
        frz = bool(ce_tax["W1_ce300"] >= ce_tax["root_ce_r"])
        ax.annotate(f"W1-C dCE@300 {ce_tax['W1_minus_C']:+.3f}\n"
                    f"(2.74M ref +{PRIORS_G1B['wall_tax_274M']:.2f}; "
                    f"freezing={frz})",
                    (0.03, 0.05), xycoords="axes fraction", fontsize=8,
                    weight="bold")
    ax.set_xlabel("wash step")
    ax.set_ylabel("in-batch corpus CE")
    ax.legend(fontsize=7.5)
    ax.set_title("THE TAX CURVE AT SCALE (the +0.53 re-priced)", fontsize=9.5)

    # (1,0) DISPLACEMENT vs the walls (raw; rms twin axis)
    ax = axes[1, 0]
    for tag in ("C", "W1", "W2", "W3"):
        rows = [r for r in disp_table[tag] if "g_m12_light" in r]
        ax.plot([r["step"] for r in rows], [r["cum_disp_raw"] for r in rows],
                marks[tag] + "-", ms=6, lw=1.8, color=cols[tag], alpha=0.9,
                label=lbls[tag])
    for R, col in zip(R_LADDER_RAW, ("seagreen", "royalblue", "darkorange")):
        ax.axhline(R, ls="--", lw=1.0, color=col, alpha=0.6)
        ax.axhline(R + PIN_FUZZ_RMS_CARRIED, ls=":", lw=0.8, color=col,
                   alpha=0.5)
    if D_kill_raw is not None:
        ax.axhline(D_kill_raw, ls=":", lw=1.8, color="k", alpha=0.8)
        ax.annotate(f"D_kill {D_kill_raw:.2f} raw "
                    f"(= {D_kill_raw / SQRT_P / R_RMS_REF:.1f}x R_rms)",
                    (0.01, D_kill_raw), xycoords=("axes fraction", "data"),
                    fontsize=7.5)
    ax.annotate(f"one AdamW step = {ONE_STEP_FUZZ_RAW:.2f} raw",
                (0.99, 0.03), xycoords="axes fraction", ha="right",
                fontsize=7.5)
    ax.set_xlabel("wash step")
    ax.set_ylabel(r"raw $\|\theta_t-\theta_0\|_2$ at checkpoints")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("DISPLACEMENT vs the walls (G-PIN, rms-carried fuzz)",
                 fontsize=10)

    # (1,1) THE RETENTION PANEL (the dip structure, honestly)
    ax = axes[1, 1]
    for tag in ("W1", "W2", "W3"):
        xs, ys = series(tag)
        ax.plot(xs, [y / root_cells["gm12"] for y in ys], marks[tag] + "-",
                ms=7, lw=2.0, color=cols[tag], alpha=0.9,
                label=f"{tag} ({lbls[tag].split('—')[1].strip()})")
    ax.axhline(0.9 / G1B_ROOT_GM12, ls="--", lw=1.6, color="black",
               alpha=0.85, label=f"0.9eq = {0.9 / G1B_ROOT_GM12:.3f} x root")
    ax.axhline(0.9, ls="-.", lw=1.1, color="gray", alpha=0.8,
               label="0.9 x root (secondary)")
    # g1b's W1 reference shape, faint, for the cross-scale eye
    ref = {int(k): v for k, v in PRIORS_G1B["W1_g_m12"].items()}
    ax.plot(sorted(ref), [ref[s] / G1B_ROOT_GM12 for s in sorted(ref)],
            "kx--", ms=5, lw=1.0, alpha=0.45,
            label="g1b W1 @2.74M (reference)")
    ax.set_xlabel("wash step")
    ax.set_ylabel("retention (g-12 / root)")
    ax.set_ylim(-0.03, 1.15)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("RETENTION vs root — the dip structure across scale",
                 fontsize=10)

    # (1,2) THE VERDICT PANEL
    ax = axes[1, 2]
    ax.axis("off")
    y = 0.97
    ax.text(0.02, y, "G1BS6 — THE WALL AT 10x, TAKE 5: the arms on the "
            "PEAK ROOT (g1bS5 m020)", fontsize=10.5,
            va="top", family="monospace", weight="bold")
    y -= 0.052
    ax.text(0.02, y, f"host {G1BS_PARAMS:,} params | theta0 = "
            f"{SRC_ROOT_CK} | bar_0p9eq {bar_0p9eq:.4f}",
            fontsize=7.2, va="top", family="monospace")
    y -= 0.030
    ax.text(0.02, y, f"  REGISTERED DEVIATION: {dev_stamp}",
            fontsize=6.8, va="top", family="monospace", color="darkred")
    y -= 0.034
    for tag in ("C", "W1", "W2", "W3"):
        seq = " -> ".join(f"+{s}:{v:.4f}" for s, v in
                          sorted(ladder[tag]["g_m12"].items()))
        ax.text(0.02, y, f"  {tag}: {seq}", fontsize=6.6, va="top",
                family="monospace", color=cols[tag])
        y -= 0.028
    y -= 0.006
    gates_lbl = ("ALL PASS (G-ROOT excluded per the registered deviation)"
                 if gates_pass else "FAILURE")
    ax.text(0.02, y, f"  gates {gates_lbl} | "
            f"D_kill {D_kill_raw if D_kill_raw is None else round(D_kill_raw, 3)}"
            f" raw | holds0.9eq: "
            + ", ".join(f"{t}={ladder[t]['holds_0p9eq_bar']}"
                        for t in ("W1", "W2", "W3")), fontsize=7.0,
            va="top", family="monospace")
    y -= 0.032
    ax.text(0.02, y, f"  tax W1-C {ce_tax['W1_minus_C']} | freezing_CE "
            f"{bool((ce_tax['W1_ce300'] or 9e9) >= ce_tax['root_ce_r'])} "
            f"(ref +0.53)",
            fontsize=7.0, va="top", family="monospace")
    y -= 0.046
    ax.text(0.02, y, f"VERDICT: {verdict}", fontsize=9.5, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.042
    for wd in textwrap.wrap(clause, width=88, break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=6.6, va="top", family="monospace")
        y -= 0.026

    fig.suptitle("G1BS6 — THE WALL AT 10x, ADJUDICATED: the arms on g1bS5's "
                 "peak root (0.7677; the registered sub-bar deviation) — "
                 "commit + project at the rms-matched ladder vs the 10M "
                 f"neutral wash -> {verdict}", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
