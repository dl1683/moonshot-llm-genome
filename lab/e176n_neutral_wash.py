"""E176N — THE NEUTRAL WASH CONTROL (the R50 critic's critical discharge;
NO QUEUE ROW — the bars in this docstring ARE the registration; e177's fold
waits on this cell).

WHY (the R50 critique of T105/e176): e176's "the consolidated fact washes
out in two steps" ran e161's stream whose ANCHOR half is the install's OWN
256-token windows with the name deleted (train_ids[p-130:p+126] at the
first 16 install positions — the TRUE host text: FLORIZEL/ELIZABETH at
x-col 130 plus its true continuation). Under full-token CE those windows
train "after this exact 130-token pre-host context, predict F/E (not Z)"
at precisely the junctions where the fact battery reads p(Z) — 16 draws
per batch x 300 steps ~ targeted EXTINCTION (paired-associate cue with
the target absent), NOT disuse. THE QUESTION THIS CELL DECIDES: does the
fact ALSO dissolve under a truly NEUTRAL stream (e170's neutral anchors —
16 plain-corpus windows, 0/16 junction coverage)?

REGISTERED BARS (frozen here before compute; the dispatch's registration
verbatim; no bar shopping — adjudicate against exactly this):
  - NEUTRAL-DISSOLVES = g-12 <= 0.27 by +50 (the wash licenses the
    activity-dependence noun — W019 strengthened as a bounded reading).
  - NEUTRAL-SURVIVES = g-12 >= 0.5 through +300 (the wash was EXTINCTION —
    the disuse noun dies; the honest finding becomes "memories cannot
    survive their own teaching contexts shown without the name").
  - SLOW-WASH texture in between (full trajectory reported).

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * g-12 / g0 / g+12 / held30 = ABSOLUTE install-60 / held-30 battery
    mean p(Z) at ctx offsets -12 / 0 / +12 (e176's convention verbatim —
    the same batteries, the same corpus rebuild, the same ruler).
  * NEUTRAL-DISSOLVES = the neutral arm's g-12 at the +50 checkpoint
    <= 0.27 (the dispatch's "by +50"); the earliest checkpoint <= 0.27
    (fine steps {1,2,4} included) is CO-REPORTED as the collapse timing.
  * NEUTRAL-SURVIVES = the neutral arm's g-12 >= 0.50 at EVERY
    continuation checkpoint {1,2,4,50,100,200,300} (the dispatch's
    "through +300"; step 0 = the root, 0.9156, trivially above).
  * The two clauses are disjoint (surviving requires +50 >= 0.50 >
    0.27); order NEUTRAL-DISSOLVES -> NEUTRAL-SURVIVES -> SLOW-WASH;
    every sub-boolean reported regardless.
  * CE-AT-DISSOLUTION (the R50 splice critique): per stream, the CE_R
    (and the in-batch corpus CE) at the FIRST checkpoint under the 0.27
    bar is reported as its own record — recovered CE values alone do not
    characterize the wash.

THREE ARMS (one file, one run):
  (A) THE NEUTRAL WASH (the critical cell): e176's protocol VERBATIM
      except the anchor bank = e170's NEUTRAL construction (16
      plain-corpus windows, RNG seed 170, rejection on FLORIZEL /
      ELIZABETH / ZEPH / MIRABEL in [s, s+257); G_ANCHOR: 0/16 host
      content, 0/16 junctions covered). Batch 32 = 16 neutral-anchor
      draws + 16 random corpus windows, full-token CE (NO fact windows,
      NO name tokens, NO mask), AdamW (0.9,0.95) wd 0.1 constant lr 1e-3
      clip 1.0, seed 10902 — the RNG draw sequence is BIT-IDENTICAL to
      e176's (same seed, same shapes/moduli: randint(16,(16,)) +
      randint(len-BLOCK-1,(16,)) per step; only anchor[aj]'s CONTENT
      differs — e170's ANCHOR-DELTA convention). Snapshots + FULL dial
      set + CE_R at fine steps {1,2,4} (the smoke rig's fine steps,
      measured in the MAIN run this time — e176's two-step claim rests
      on its smoke) and main {50,100,200,300}.
  (B) THE LR RIDER (attack 2): the ORIGINAL stream (e161/e176's
      original anchor bank, rebuilt for the delta record) at lr 1e-4,
      300 steps, seed 10902 — does the wash rate scale with the
      optimizer? Light in-run evals at {50,100,200,300} + one full dial
      at +300 (anatomy co-report).
  (C) THE RESTORE-INTO-+50 EVAL (attack 4, eval-only, minutes):
      restore the root's MLP+LN class (e173's CLASSES['mlp_ln'], 50
      tensors, via e178's restore_class VERBATIM) into e176's ON-DISK
      +50 checkpoint (runs/checkpoints/e176_root_freeze_s50.pt; gated vs
      e176's stored +50 cells BEFORE any surgery). READING (the
      dispatch's fork, operationalized): ~42% again (g-12 within
      e178's +300 band 0.4224 +/- 0.08) => the "half-fact" is
      interface/gain-set (recovery ceiling set by the hybrid context,
      not by wash depth); ~80% (>= 0.65, toward e173's forward 0.7818)
      => WASH-DEPTH ARTIFACT (e178's 42% was the deeper +300 wash);
      in between => MIXED, numbers reported. Annotation fork, no
      re-adjudication of e178. All-restored sanity gate bounds the
      machinery (sd must equal the root bit-exactly).

NETS: root runs/checkpoints/e131_consolidated_e113.pt (gate bit-exact vs
e151's stored before-cells, e176's G_ROOT set); the ORIGINAL stream's
trajectory = e176's stored trace (embedded, verified vs file); e176's
fine steps = its SMOKE trace (embedded, verified; the main run never
measured {1,2,4}); e178's +300 restore cells + e173's forward 0.7818
loaded/embedded for the inset. New checkpoints: runs/checkpoints/
e176n_neutral{,_sN}.pt (arm A) + e176n_lr1e4{,_s50}.pt (arm B).

INSTRUMENT PROVENANCE: load_cpu/evl_load/battery_cell/battery_pz/
ce_fixed_cpu/val_windows/deleted_wpe/read_fact_at/row_census/
finetune_freeze are lab/e176_freeze_root.py VERBATIM (the e161/e152/
e151/e143/e131/e119/e113/e068/e065/e043 lineage; finetune_freeze gains
lr/ckpt-step parameters — arithmetic otherwise identical); the neutral
bank + junction accounting + G_ANCHOR are lab/e170_anchor_neutral.py
VERBATIM; restore_class/CLASSES/_mlp_keys/_ln_keys are lab/
e178_reverse_restore.py (= e173) VERBATIM. The measure dial is e176's
measure() minus the 183-span census (e178's recorded deviation: the
consolidated root has NO 183 site — 0.00027 < 2x control — and no arm
here touches that conclusion; the functional site read @183 is kept).
Copied, not imported, to own the device policy.

COMPUTE ENVELOPE: CPU-ONLY by dispatch (CUDA_VISIBLE_DEVICES=-1 forced
before torch; e174/e177/e175 also on the CPU — LOW threads <= 4), one
25 s launch stagger (single sleep, no busy-waiting anywhere), cooldown
60 s before and after EACH of the two trainings, per-training CPU cap
1800 s (e176 precedent: 414.6 s actual for 300 steps at 4 threads).

Outputs: runs/e176n/{metrics.json, neutral_wash.png}; checkpoints
runs/checkpoints/e176n_*.pt. No NOTES/THINKING/QUEUE/STATE edits;
single commit, no push.

Run:  cd lab && python e176n_neutral_wash.py    (E176N_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e174/e177/
# e175 also on the CPU — threads capped at 4 below)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

torch.set_num_threads(4)                              # dispatch: LOW <= 4

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT, cooldown,     # noqa: E402
                    run_dir, save_json)
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E176N_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e176n is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
ROOT_CK = "e131_consolidated_e113.pt"   # THE FULLY-CONSOLIDATED ROOT
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E176_METRICS = E43.REPO / "runs" / "e176" / "metrics.json"
E176_SMOKE_METRICS = E43.REPO / "runs" / "e176_smoke" / "metrics.json"
E178_METRICS = E43.REPO / "runs" / "e178" / "metrics.json"
S50_CK = CKPT_DIR / "e176_root_freeze_s50.pt"      # arm C target (on disk)

# ---- e152's placement constants (the measurement instruments rebuild these) ----
RETEACH_J = 54                    # offset: name x-cols 184..190, read rows 183..189
SITE_ADDR_ROW = 183               # the onset read row (= e120's SPLICE_ADDR_ROW)
SITE_Z_XCOL = PRE + RETEACH_J     # 184 (= e120's Z_XCOL)
SITE_CONT = BLOCK - PRE - RETEACH_J - len(NAME)     # 65-token continuation
D_ALL = (121, 125, 129, 133, 137)                   # e113's fixed set
GEOS = (-12, 0, 12)               # novel x2 + the trained g0

# ---- census row set (e158/e161/e176 old-band convention; read-visible rows) ----
ROWS_OLD = (0, 1, 2, 3, 4, 5, 6, 60, 100, 118, 119, 120) + tuple(range(121, 130))
if SMOKE:
    ROWS_OLD = (0, 1, 60, 121, 125, 129)

# ---- the arms' freeze schedules ----------------------------------------------
CK_A: tuple[int, ...] = (1, 2, 4, 50, 100, 200, 300) if not SMOKE else (1, 2, 4)
CK_B: tuple[int, ...] = (50, 100, 200, 300) if not SMOKE else (2, 4)

# ---- fine-tune envelope (e109 arm-b / e119-L / e143 / e151 / e152 / e161/e176) --
LR_A = 1e-3                        # arm A: e176 verbatim
LR_B = 1e-4                        # arm B: the LR rider
FT_TIME_CAP = 1800.0               # CPU cap per training (e176 precedent)
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
FREEZE_SEED = 10902               # the locked seed lineage (= e161/e176's)
COOLDOWN_S = 60.0                  # around EACH of the two trainings
STAGGER_S = 25.0                   # launch stagger vs the CPU fleet

# ---- e170's neutral anchor bank (THE delta; see the docstring) ----------------
E170_ANCHOR_SEED = 170             # dedicated RNG for the plain-corpus bank
ANCHOR_FORBIDDEN = ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")

# ---- parameter classes (e173 via e178 VERBATIM; arm C) ------------------------
N_LAYER, N_HEAD, N_EMBD = 6, 6, 192

def _mlp_keys(l):
    return [f"h.{l}.mlp.0.weight", f"h.{l}.mlp.0.bias",
            f"h.{l}.mlp.2.weight", f"h.{l}.mlp.2.bias"]

def _ln_keys(l):
    return [f"h.{l}.ln1.weight", f"h.{l}.ln1.bias",
            f"h.{l}.ln2.weight", f"h.{l}.ln2.bias"]

def _attn_keys(l):
    return [f"h.{l}.attn.c_attn.weight", f"h.{l}.attn.c_proj.weight"]

_MLP = [k for l in range(N_LAYER) for k in _mlp_keys(l)]
_LN_BLOCK = [k for l in range(N_LAYER) for k in _ln_keys(l)]
_LN = _LN_BLOCK + ["ln_f.weight", "ln_f.bias"]
_ATTN = [k for l in range(N_LAYER) for k in _attn_keys(l)]
_WPE = ["wpe.weight"]
_IO = ["wte.weight", "lm_head.weight"]

CLASSES = {                     # e173's registered/sanity key sets verbatim
    "mlp_ln": _MLP + _LN,       # arm C: the class the dispatch names — 50 tensors
    "all": _MLP + _LN + _ATTN + _WPE + _IO,   # sanity gate: must cover every key
}

# ---- gates / references (full precision, = stored metrics) --------------------
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05

E151_ROOT = {                     # runs/e151 'before' battery (e176's gate set)
    "base_gm12": 0.9155886769294739,
    "base_g0": 0.7850371599197388,
    "base_gp12": 0.9478210210800171,
    "ce_r": 1.663516640663147,
    "site_read_onset": 0.8898659348487854,
    "site_read_span": 0.982668936252594,
    "A129": -0.13237020391970877,
    "row0_strength": 0.7316772222270098,
    "dall_g0": 0.9047248959541321,
}

# e176's stored MAIN trajectory (the original extinction-confounded stream,
# lr 1e-3): runs/e176/metrics.json trace_summary, VERBATIM, re-verified at
# plot time.
E176_MAIN = {
    "freeze_steps": [0, 50, 100, 200, 300],
    "base_gm12": [0.9155886173248291, 0.020882638171315193,
                  0.006246014963835478, 0.0006464376347139478,
                  0.0011210207594558597],
    "base_g0": [0.7850371599197388, 0.027392588555812836,
                0.016695411875844002, 0.001976940780878067,
                0.0012388104805722833],
    "base_gp12": [0.9478210210800171, 0.02393173798918724,
                  0.008211151696741581, 0.001558798598125577,
                  0.0017126891762018204],
    "held30_gm12": [0.6361417174339294, 0.00599030451849103,
                    0.001938893343321979, 0.00025415088748559356,
                    0.0003840079589281231],
    "held30_g0": [0.7233286499977112, 0.01260960940271616,
                  0.005210408940911293, 0.0009895081166177988,
                  0.0005695995059795678],
    "row0_strength": [0.7316772222270098, 0.02203246613395701,
                      0.015375644575343964, 0.0015484262148422432,
                      -0.0005032020296397376],
    "dall_g0": [0.9047248959541321, 0.03675490990281105,
                0.008526976220809734, 0.0017805545357988285,
                0.001604252029210329],
    "ce_r": [1.663516640663147, 1.6478883028030396,
             1.6605137586593628, 1.6117957830429077,
             1.6334049701690674],
}

# e176's SMOKE fine steps (the ONLY fine-step trajectory the original stream
# has — the main run never measured {1,2,4}; runs/e176_smoke/metrics.json
# trace_summary VERBATIM; the R50 auditor attributed the two-step claim to
# this run): g-12 0.9156 -> 0.0882 (s2, CE_R 2.0001) -> 0.0026 (s4).
E176_SMOKE = {
    "freeze_steps": [0, 2, 4],
    "base_gm12": [0.9155886173248291, 0.08816977590322495,
                  0.002640149090439081],
    "base_g0": [0.7850371599197388, 0.08683783560991287,
                0.0034518486354500055],
    "ce_r": [1.663516640663147, 2.0001156330108643, 1.7823567390441895],
}

# e176's stored +50 cells (trace index 1) — the G_S50 gate for arm C's target.
E176_S50 = {
    "gm12": 0.020882638171315193,
    "g0": 0.027392588555812836,
    "gp12": 0.02393173798918724,
    "held30_gm12": 0.00599030451849103,
    "held30_g0": 0.01260960940271616,
    "ce_r": 1.6478883028030396,
    "site_read_onset": 0.02520560473203659,
    "site_read_span": 0.8252765536308289,
    "A129": -0.012971208266814457,
    "row0_strength": 0.02203246613395701,
    "dall_g0": 0.03675490990281105,
}

# e178's stored +300 MLP+LN restore cells (runs/e178/metrics.json arm.cells)
# + e173's FORWARD single-class restore (runs/e173/metrics.json
# adjudication.class_gm12['mlp_ln']) — the inset references for arm C.
E178_RESTORE_REF = {
    "target": "+300 washed (e176_root_freeze.pt)",
    "gm12": 0.4223632514476776,
    "g0": 0.3274501860141754,
    "held30_gm12": 0.2527451813220978,
    "held30_g0": 0.25866401195526123,
    "ce_r": 1.694811463356018,
    "retention_gm12": 0.46064205256736845,
    "verdict": "TEXTURE (half-fact: all dials 40-45%, CE-cheap)",
}
E173_FORWARD_REF = {
    "desc": "e173's FORWARD single-class restore (conversion direction)",
    "gm12": 0.7817810773849487,
    "reading": "84% of ceiling — the wash-depth-artifact pole of arm C's fork",
}

# ---- registered bar constants (frozen; see OPERATIONALIZATIONS above) --------
SHUT_BAR = 0.27                   # NEUTRAL-DISSOLVES by +50 (e158/e161/e176)
SURVIVE_BAR = 0.50                # NEUTRAL-SURVIVES through +300
IFACE_BAND = 0.08                 # arm C: e178's gm12 +/- this = "~42% again"
DEEP_CKPT_BAR = 0.65              # arm C: at/above this = "~80%" wash-depth pole
ROOT_GM12 = E151_ROOT["base_gm12"]
ROOT_G0 = E151_ROOT["base_g0"]
ROOT_ROW0 = E151_ROOT["row0_strength"]

REGISTERED_PREDICTION = {
    "neutral_dissolves": "NEUTRAL-DISSOLVES fires if: g-12 <= 0.27 by +50 "
        "(the wash licenses the activity-dependence noun — W019 strengthened "
        "as a bounded reading).",
    "neutral_survives": "NEUTRAL-SURVIVES fires if: g-12 >= 0.5 through +300 "
        "(the wash was EXTINCTION — the disuse noun dies; the honest finding "
        "becomes 'memories cannot survive their own teaching contexts shown "
        "without the name').",
    "slow_wash": "SLOW-WASH texture in between (full trajectory reported).",
    "operationalizations": "g-12/g0/g+12/held30 = absolute install-60/held-30 "
        "battery mean p(Z) at ctx offsets -12/0/+12 (e176's convention); "
        "NEUTRAL-DISSOLVES = neutral arm g-12 at +50 <= 0.27 (earliest "
        "checkpoint <= 0.27 over {1,2,4,50,100,200,300} CO-REPORTED); "
        "NEUTRAL-SURVIVES = neutral arm g-12 >= 0.50 at EVERY continuation "
        "checkpoint {1,2,4,50,100,200,300}; clauses disjoint; order "
        "NEUTRAL-DISSOLVES -> NEUTRAL-SURVIVES -> SLOW-WASH; CE-at-dissolution "
        "reported per stream (CE_R + in-batch corpus CE at the first "
        "checkpoint under 0.27); arm B (lr 1e-4) and arm C (restore-into-+50) "
        "are riders — reported, not bar-adjudicated (arm C's fork "
        "operationalized: gm12 in e178's band 0.4224 +/- 0.08 = "
        "interface/gain-set; >= 0.65 = wash-depth artifact; between = MIXED).",
    "registration": "NO QUEUE ROW EXISTS — the dispatch registered these bars "
        "in the mission text; this docstring freezes them verbatim before "
        "compute. Adjudicate against exactly this; no bar shopping.",
}

trims: list[str] = []
deviations: list[str] = [
    "CPU-ONLY by dispatch (CUDA_VISIBLE_DEVICES=-1 forced before torch; "
    "e174/e177/e175 also on the CPU): torch threads 4, one 25 s launch "
    "stagger (single sleep, no busy-waiting), cooldown 60 s before/after "
    "EACH of the two trainings (arm A + arm B), per-training CPU cap 1800 s "
    "(e176 precedent: 414.6 s actual / 300 steps).",
    "NO QUEUE ROW: the registered bars are the dispatch's registration, "
    "frozen verbatim in the module docstring before compute.",
    "The measure dial is e176's measure() minus the 183-span census (e178's "
    "recorded deviation): the consolidated root has NO 183 site (0.00027 < "
    "2x control) and no arm here touches that conclusion; the functional "
    "site read @183 is kept as the instrument co-report.",
    "Arm A measures the fine steps {1,2,4} in the MAIN run (e176's two-step "
    "claim rests on its smoke run — the R50 auditor's splice critique); "
    "arm A's smoke schedule IS the fine-step schedule (4 steps, ckpts "
    "{1,2,4}).",
    "Arm B carries light in-run evals only, plus ONE full dial at +300 "
    "(anatomy co-report); its question (wash-rate scaling) is answered by "
    "the g-12 trajectory, not the full battery.",
    "Arm C is eval-only surgery on the ON-DISK e176 +50 checkpoint (gated "
    "vs e176's stored +50 cells in G_S50 before any surgery); its fork is "
    "an ANNOTATION classification of the dispatch's '~42% again vs ~80%' "
    "reading — it does not re-adjudicate e178.",
    "finetune_freeze gains lr / ckpt-steps parameters vs e176's verbatim "
    "copy (arm B needs lr 1e-4; arm A needs the fine-step schedule); the "
    "per-step arithmetic and the RNG draw sequence are unchanged — at seed "
    "10902 both arms draw the SAME aj/rj sequences as e176's original run "
    "(only anchor content / lr differ).",
    "Eval thread count is 4 (dispatch) vs e151's stored cells — CPU "
    "reduction order can drift low-order bits; the G_ROOT gate reports both "
    "the 5e-6 bit flag and the 0.05 fallback tolerance (e161/e176/e178 "
    "precedent).",
    "Single seed (10902), one trajectory per training arm, one root "
    "lineage, n=1 per cell — point estimates until replicated.",
    "Smoke mode trims: arm A 4 steps / arm B 4 steps, lean measures "
    "(no censuses, no deletion table), arm C gates on base cells + CE "
    "only; nothing adjudicated.",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/e176_freeze_root.py VERBATIM (see the module docstring) +
# e170's anchor gate + e178's restore_class. Copied rather than imported to
# own the device policy.

def load_cpu(path) -> TinyGPT:
    m = TinyGPT(Cfg())
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


def evl_load(sd: dict) -> TinyGPT:
    m = TinyGPT(Cfg())
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
    while len(out_x) < n and tries < 500 * n:
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
    """e131's read_fact_position VERBATIM ARITHMETIC (e151/e152/e161/e176/e178
    copy): p(true name char) at positions addr_row..addr_row+6 over the pool."""
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
    """e139's row_census_at183 VERBATIM (mean-arm / zero-arm / restore)."""
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


# ------------------------------------------------------------------ fine-tune

def finetune_freeze(tag: str, net0: TinyGPT, anchor: torch.Tensor,
                    train_ids: torch.Tensor, itos, r_eval_xy, gm12_ids,
                    g0_ids, zid: int, seed: int, lr: float,
                    ckpt_steps: tuple[int, ...]):
    """THE PLAIN-CORPUS FREEZE (e176's finetune_freeze VERBATIM arithmetic,
    parameterized lr / ckpt steps). Per step: aj = randint(16) anchor draws,
    rj = randint(16) random corpus offsets; batch 32 full-token CE; AdamW
    (0.9,0.95) wd 0.1 constant lr, clip 1.0. The RNG draw sequence is
    IDENTICAL to e176's at seed 10902 (same shapes/moduli) — only anchor
    content (arm A) or lr (arm B) differ. Snapshots (deep-copy out) + light
    CPU evals (g-12, g0, CE_R — no RNG consumed) at the checkpoint steps;
    the in-batch corpus CE is recorded at every checkpoint and every 50."""
    ckpt_set = set(ckpt_steps)
    n_steps = ckpt_steps[-1]
    net = copy.deepcopy(net0)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    n_anc = anchor.shape[0]
    traj, sds, t_start = [], {}, time.time()
    step = 0
    zeph_checks = 0
    evl = copy.deepcopy(net0)          # CPU eval twin
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
        y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
        logits, _ = net(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step in ckpt_set or step % 50 == 0:
            log(f"  [{tag}] s{step:4d} corpus CE {float(loss.item()):.4f} "
                f"({time.time() - t_start:.0f}s)")
        if step in ckpt_set:
            sd_cpu = {k: v.detach().cpu().clone()
                      for k, v in net.state_dict().items()}
            sds[step] = sd_cpu
            evl.load_state_dict(sd_cpu)
            evl.eval()
            gz = battery_cell(evl, gm12_ids, zid)
            gz0 = battery_cell(evl, g0_ids, zid)
            ce_r = ce_fixed_cpu(evl, *r_eval_xy)
            traj.append({"step": step, "g_m12_mean_pz": gz["mean_pz"],
                         "g0_mean_pz": gz0["mean_pz"],
                         "frac_argmax_z": gz["frac_argmax_z"],
                         "corpus_ce": float(loss.item()), "ce_r": ce_r,
                         "elapsed_s": round(time.time() - t_start, 1)})
            log(f"  [{tag}] CKPT +{step:4d} g-12 {gz['mean_pz']:.4f} "
                f"g0 {gz0['mean_pz']:.4f} CE_R {ce_r:.4f} "
                f"(in-batch CE {float(loss.item()):.4f})")
        if (time.time() - t_start) > FT_TIME_CAP:
            log(f"  [{tag}] time cap {FT_TIME_CAP:.0f}s at s{step}")
            trims.append(f"{tag}: time cap at step {step}")
            break
    net.eval()
    return {"sds": sds, "traj": traj, "steps_ran": step, "seed": seed,
            "lr": lr, "zeph_violations": zeph_checks}


# ------------------------------------------------------------------ surgery (e173)

def restore_class(sd_base: dict, sd_src: dict, keys: list, name: str
                  ) -> tuple[dict, dict]:
    """Replace the TARGET's tensors for `keys` with the SOURCE's; confinement
    gate = only class keys may change, everything else bit-identical (e153's
    transplant gate generalized to key sets; e173/e178 VERBATIM)."""
    out = {k: v.clone() for k, v in sd_base.items()}
    for k in keys:
        out[k] = sd_src[k].clone()
    ks = set(keys)
    per_key = {k: int((out[k] != sd_base[k]).sum().item()) for k in keys}
    n_changed = sum(per_key.values())
    n_elems = sum(sd_base[k].numel() for k in keys)
    others = all(torch.equal(out[k], sd_base[k]) for k in out if k not in ks)
    gate = {"class": name, "n_tensors": len(keys),
            "tensors_changed": int(sum(1 for v in per_key.values() if v > 0)),
            "n_elements_changed": n_changed,
            "expected_elements": int(n_elems),
            "per_key_changed": {k: v for k, v in per_key.items() if v > 0},
            "identity": bool(n_changed == 0),
            "others_bit_identical": bool(others),
            "pass": bool(others and (n_changed == 0
                                     or n_changed >= 0.95 * n_elems))}
    return out, gate


CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "e176n", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name} "
        f"({', '.join(f'{k}={v}' for k, v in meta.items())})")


def verify_ref(embedded: dict, path: Path, ts_keys: list, src_name: str) -> dict:
    """Verify an embedded reference copy against its stored metrics file when
    present (no silent divergence; e176's load_e161_ref convention)."""
    src = {"source": f"embedded verbatim copy ({src_name})",
           "file_present": path.exists(), "verified_vs_embedded": None,
           "max_abs_diff": None}
    if path.exists():
        mm = json.loads(path.read_text(encoding="utf-8"))
        ts = mm["trace_summary"] if "trace_summary" in mm else mm
        diffs = [abs(a - b) for k in ts_keys
                 for a, b in zip(ts[k], embedded[k])]
        steps_ok = list(ts[ts_keys[0]]) == embedded[ts_keys[0]]
        src["max_abs_diff"] = max(diffs) if diffs else None
        src["verified_vs_embedded"] = bool(steps_ok and max(diffs) < 1e-9)
        if src["verified_vs_embedded"]:
            src["source"] = f"{src_name} (embedded copy verified, max|diff| " \
                            f"{max(diffs):.1e})"
    return src


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e176n_smoke" if SMOKE else "e176n")
    log(f"E176N THE NEUTRAL WASH CONTROL (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()}), stagger "
        f"{STAGGER_S:.0f}s, cooldown {COOLDOWN_S:.0f}s around each training")
    time.sleep(STAGGER_S)            # launch stagger vs the CPU fleet

    # ---------------- protocol rebuild (e143/e151/e152/e161/e176 verbatim)
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

    # ---------------- measurement pool: e152's locked j=54 windows (instrument)
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

    # =====================================================================
    # THE ANCHOR DELTA (e170's construction VERBATIM): the original
    # extinction-confounded bank (rebuilt for the delta record) vs e170's
    # NEUTRAL plain-corpus bank (arm A's only protocol change).
    # =====================================================================
    # e161/e176 bank verbatim: the first 16 install positions' ORIGINAL host
    # windows — the teaching junctions shown WITHOUT the name (the true host
    # + continuation): targeted extinction.
    anchor_orig = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                               for p, _ in install_occ[:16]])
    n_orig_hostwins = sum(
        1 for p, _ in install_occ[:16]
        if any(f in train_text[p - PRE: p - PRE + BLOCK + 1]
               for f in HOSTS))
    anchor_orig_zeph = sum(1 for w in anchor_orig if "ZEPH" in corpus.decode(w))
    G_ANCHFREE_ORIG = {"anchor_zeph_windows": anchor_orig_zeph,
                       "pass": bool(anchor_orig_zeph == 0)}
    assert G_ANCHFREE_ORIG["pass"], "original anchor bank contains ZEPH"

    # e170's neutral bank VERBATIM: plain corpus windows, rejection on
    # host/nonce content in [s, s+257) — window PLUS first target.
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

    # junction accounting (e170 VERBATIM): a window "covers" a junction if
    # its [s, s+257) span contains a host occurrence's onset position p.
    host_positions = [p for p in E43.find_occ(train_text, HOSTS[0])]
    host_positions += [p for p in E43.find_occ(train_text, HOSTS[1])]

    def junctions_covered(starts):
        cov = 0
        for s in starts:
            if any(s <= p < s + BLOCK + 1 for p in host_positions):
                cov += 1
        return cov

    jc_orig = junctions_covered([p - PRE for p, _ in install_occ[:16]])
    jc_neutral = junctions_covered(n_starts)
    host_occ_total = len(host_positions)
    bg_rate = host_occ_total * (BLOCK + 1) / len(train_ids)
    G_ANCHOR = {
        "neutral_bank": {
            "construction": ("16 plain corpus windows from train_ids, RNG seed "
                             f"{E170_ANCHOR_SEED}, rejection if [s, s+257) "
                             "contains FLORIZEL/ELIZABETH/ZEPH/MIRABEL — "
                             "e170's construction VERBATIM"),
            "n_windows": 16, "block": BLOCK, "seed": E170_ANCHOR_SEED,
            "starts": n_starts, "tries": tries, "rejections": rejections,
            "forbidden": list(ANCHOR_FORBIDDEN),
            "windows_with_host_content": sum(
                1 for s in n_starts
                if any(f in train_text[s: s + BLOCK + 1] for f in HOSTS)),
            "junctions_covered": jc_neutral,
        },
        "original_bank": {
            "construction": ("e161/e176 verbatim: the first 16 install "
                             "positions' ORIGINAL host windows "
                             "(train_ids[p-130 : p+126]) — the teaching "
                             "junctions shown with the TRUE host + "
                             "continuation and no ZEPHYRA (targeted "
                             "EXTINCTION; the R50 critic's finding)"),
            "n_windows": 16,
            "windows_with_host_content": n_orig_hostwins,
            "junctions_covered": jc_orig,
        },
        "budget_identical": bool(anchor_neutral.shape == anchor_orig.shape),
        "rng_stream_identical_to_e176": True,
        "rng_note": ("finetune_freeze verbatim; draw shapes/moduli identical "
                     "(n_anc=16, len(train_ids)); seed 10902 — the same "
                     "aj/rj sequences as e176's run; only anchor[aj] content "
                     "(arm A) or lr (arm B) differs"),
        "random_channel_background": {
            "host_occurrences_in_train": host_occ_total,
            "est_window_hit_rate": bg_rate,
            "note": ("the 16-per-batch random corpus windows are e161/e176 "
                     "VERBATIM and unfiltered in ALL THREE streams — "
                     "identical background (~1-2%/window), not part of the "
                     "delta"),
        },
    }
    G_ANCHOR["pass"] = bool(
        G_ANCHOR["neutral_bank"]["windows_with_host_content"] == 0
        and G_ANCHOR["neutral_bank"]["junctions_covered"] == 0
        and G_ANCHOR["original_bank"]["windows_with_host_content"] == 16
        and G_ANCHOR["original_bank"]["junctions_covered"] == 16
        and G_ANCHOR["budget_identical"]
        and anchor_neutral.shape == (16, BLOCK))
    assert G_ANCHOR["pass"], f"anchor gate FAILED: {G_ANCHOR}"
    log(f"G_ANCHOR: neutral bank 16x{BLOCK} plain-corpus windows (seed "
        f"{E170_ANCHOR_SEED}, {rejections} rejections/{tries} tries) — host "
        f"content 0/16, junctions 0/16; original bank (rebuilt): host "
        f"{n_orig_hostwins}/16, junctions {jc_orig}/16; random-channel "
        f"background ~{100 * bg_rate:.1f}%/window: PASS")

    # ---------------- batteries: ctx = train_text[p-PRE-j : p] (e119 verbatim)
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

    # ---------------- root net + gate vs e151
    net0 = load_cpu(CKPT_DIR / ROOT_CK)
    sd_root = {k: v.clone() for k, v in net0.state_dict().items()}
    root_meta = None
    st_raw = torch.load(CKPT_DIR / ROOT_CK, map_location="cpu",
                        weights_only=False)
    if isinstance(st_raw, dict) and "meta" in st_raw:
        root_meta = E43.jsonable(st_raw["meta"])
    log(f"root: {ROOT_CK} (meta: {root_meta})")

    gates_surg: dict = {}

    def measure(sd: dict, tag: str, lean: bool = False) -> dict:
        """e176's measure() minus the 183-span census (recorded deviation):
        base 3-geos + held30 + CE_R + site read + old-band census (row-0
        sink / A129 brake) + deletion table (D-all, D-183). lean=True (the
        smoke/graded convention) drops the census + deletions, keeps a
        quick A(129)."""
        net = evl_load(sd)
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

    log("=" * 78)
    log("STEP-0 battery (root = e131_consolidated_e113; 'before')")
    root = measure(sd_root, "root", lean=SMOKE)
    root_cells = flat_cells(root)
    keymap = {"base_gm12": "gm12", "base_g0": "g0", "base_gp12": "gp12",
              "ce_r": "ce_r", "site_read_onset": "site_read_onset",
              "site_read_span": "site_read_span", "A129": "A129",
              "row0_strength": "row0_strength", "dall_g0": "dall_g0"}
    root_refs = {keymap[k]: v for k, v in E151_ROOT.items()}
    G_ROOT = gate_vs(root_cells, root_refs, "G_ROOT (vs e151 before-cells)")
    if not G_ROOT["pass"]:
        raise RuntimeError("consolidated-root gate FAILED vs e151 stored "
                           "before-cells")
    log("gates: G_SPLICE, G_NAMEFREE, G_POOL, G_ANCHOR, G_ANCHFREE_ORIG, "
        "G_ROOT all PASS")

    # reference provenance (embedded copies verified vs the stored files)
    TS_KEYS = ["freeze_steps", "base_gm12", "base_g0", "base_gp12",
               "held30_gm12", "held30_g0", "row0_strength", "dall_g0",
               "ce_r"]
    src_e176 = verify_ref(E176_MAIN, E176_METRICS, TS_KEYS,
                          "runs/e176/metrics.json trace_summary")
    src_e176s = verify_ref(E176_SMOKE, E176_SMOKE_METRICS,
                           ["freeze_steps", "base_gm12", "base_g0", "ce_r"],
                           "runs/e176_smoke/metrics.json trace_summary")
    src_e178 = {"file_present": E178_METRICS.exists(),
                "verified_vs_embedded": None}
    if E178_METRICS.exists():
        mm = json.loads(E178_METRICS.read_text(encoding="utf-8"))
        cells = mm["arm"]["cells"]
        src_e178["verified_vs_embedded"] = bool(
            abs(cells["gm12"] - E178_RESTORE_REF["gm12"]) < 1e-9
            and abs(cells["g0"] - E178_RESTORE_REF["g0"]) < 1e-9)
        src_e178["source"] = ("runs/e178/metrics.json arm.cells (embedded "
                              "copy verified)"
                              if src_e178["verified_vs_embedded"]
                              else "EMBEDDED COPY DIVERGED — using file")
        if src_e178["verified_vs_embedded"]:
            E178_RESTORE_REF.update({k: cells[k] for k in
                                     ("held30_gm12", "held30_g0", "ce_r")})
            E178_RESTORE_REF["retention_gm12"] = mm["arm"]["retention_gm12"]
    else:
        src_e178["source"] = "embedded copy (file absent)"

    # =====================================================================
    # ARM A — THE NEUTRAL WASH (the critical cell; ONE training; cooldown)
    # =====================================================================
    log("=" * 78)
    if not SMOKE:
        log(f"[thermal] cooldown {COOLDOWN_S:.0f}s before arm A")
        cooldown(COOLDOWN_S)
    log(f"ARM A — THE NEUTRAL WASH: {CK_A[-1]}-step plain-corpus freeze with "
        f"e170's NEUTRAL anchors (batch {ANCH_BS} neutral + {RAND_BS} random, "
        f"full-token CE, lr {LR_A}, seed {FREEZE_SEED} = e176's draw "
        f"sequence), checkpoints +{list(CK_A)}")
    armA = finetune_freeze("neutral", net0, anchor_neutral, train_ids, itos,
                           r_eval_xy, gm12_ids, g0_ids, zid, FREEZE_SEED,
                           LR_A, CK_A)
    if not SMOKE:
        log(f"[thermal] cooldown {COOLDOWN_S:.0f}s after arm A")
        cooldown(COOLDOWN_S)
    G_DRAWFREE_A = {"zeph_violations": armA["zeph_violations"],
                    "pass": bool(armA["zeph_violations"] == 0)}
    assert G_DRAWFREE_A["pass"], "arm A: name token leaked into a window"

    save_ckpt("e176n_neutral", armA["sds"][max(armA["sds"])],
              {"desc": f"e131_consolidated_e113 + {max(armA['sds'])}-step "
                       f"NEUTRAL-anchor plain-corpus freeze (e170's neutral "
                       f"bank seed {E170_ANCHOR_SEED}: 16 plain-corpus "
                       f"windows, 0/16 junctions; batch 32 = 16 neutral "
                       f"anchors + 16 random, full-token CE), lr {LR_A}, "
                       f"seed {FREEZE_SEED}",
               "steps": int(max(armA["sds"])), "seed": FREEZE_SEED,
               "lr": LR_A, "anchor_bank": f"neutral (seed {E170_ANCHOR_SEED})",
               "base": f"runs/checkpoints/{ROOT_CK}"})
    for s in sorted(armA["sds"]):
        if s == max(armA["sds"]):
            continue
        save_ckpt(f"e176n_neutral_s{s}", armA["sds"][s],
                  {"desc": f"e131_consolidated_e113 + {s}-step NEUTRAL-anchor "
                           f"freeze (intermediate), seed {FREEZE_SEED}",
                   "steps": int(s), "seed": FREEZE_SEED, "lr": LR_A,
                   "anchor_bank": f"neutral (seed {E170_ANCHOR_SEED})",
                   "base": f"runs/checkpoints/{ROOT_CK}"})

    batteriesA: dict = {"root": root}
    for s in sorted(armA["sds"]):
        log(f"ARM A +{s} battery")
        batteriesA[str(s)] = measure(armA["sds"][s], f"n{s}", lean=SMOKE)

    # =====================================================================
    # ARM B — THE LR RIDER (original extinction stream at lr 1e-4)
    # =====================================================================
    log("=" * 78)
    if not SMOKE:
        log(f"[thermal] cooldown {COOLDOWN_S:.0f}s before arm B")
        cooldown(COOLDOWN_S)
    log(f"ARM B — THE LR RIDER: the ORIGINAL stream (e161/e176's anchor "
        f"bank) at lr {LR_B}, {CK_B[-1]} steps, seed {FREEZE_SEED} (same "
        f"draw sequence), checkpoints +{list(CK_B)}")
    armB = finetune_freeze("lr1e4", net0, anchor_orig, train_ids, itos,
                           r_eval_xy, gm12_ids, g0_ids, zid, FREEZE_SEED,
                           LR_B, CK_B)
    if not SMOKE:
        log(f"[thermal] cooldown {COOLDOWN_S:.0f}s after arm B")
        cooldown(COOLDOWN_S)
    G_DRAWFREE_B = {"zeph_violations": armB["zeph_violations"],
                    "pass": bool(armB["zeph_violations"] == 0)}
    assert G_DRAWFREE_B["pass"], "arm B: name token leaked into a window"

    save_ckpt("e176n_lr1e4", armB["sds"][max(armB["sds"])],
              {"desc": f"e131_consolidated_e113 + {max(armB['sds'])}-step "
                       f"ORIGINAL-anchor (extinction) freeze at lr {LR_B} "
                       f"(the LR rider), seed {FREEZE_SEED}",
               "steps": int(max(armB["sds"])), "seed": FREEZE_SEED,
               "lr": LR_B, "anchor_bank": "original (e161/e176 host windows)",
               "base": f"runs/checkpoints/{ROOT_CK}"})
    if 50 in armB["sds"] and 50 != max(armB["sds"]):
        save_ckpt("e176n_lr1e4_s50", armB["sds"][50],
                  {"desc": "e131_consolidated_e113 + 50-step ORIGINAL-anchor "
                           f"freeze at lr {LR_B} (intermediate), seed "
                           f"{FREEZE_SEED}",
                   "steps": 50, "seed": FREEZE_SEED, "lr": LR_B,
                   "anchor_bank": "original (e161/e176 host windows)",
                   "base": f"runs/checkpoints/{ROOT_CK}"})

    batteriesB: dict = {}
    for s in sorted(armB["sds"]):
        log(f"ARM B +{s} battery (lean; +300 upgraded to full in the main "
            f"run)")
        batteriesB[str(s)] = measure(armB["sds"][s], f"b{s}", lean=True)
    if not SMOKE:
        smax = max(armB["sds"])
        batteriesB[str(smax)] = measure(armB["sds"][smax], f"b{smax}_full",
                                        lean=False)

    # =====================================================================
    # ARM C — THE RESTORE-INTO-+50 EVAL (attack 4; eval-only, minutes)
    # =====================================================================
    log("=" * 78)
    log("ARM C — THE RESTORE-INTO-+50 EVAL (eval-only): root's MLP+LN class "
        "into e176's on-disk +50 checkpoint")
    armC: dict = {"ran": False}
    if S50_CK.exists():
        armC["ran"] = True
        m_s50 = load_cpu(S50_CK)
        sd_s50 = {k: v.clone() for k, v in m_s50.state_dict().items()}
        st50 = torch.load(S50_CK, map_location="cpu", weights_only=False)
        armC["target"] = {
            "path": str(S50_CK.relative_to(E43.REPO)).replace("\\", "/"),
            "meta": E43.jsonable(st50.get("meta", {}))
            if isinstance(st50, dict) else None}
        del m_s50, st50
        log(f"arm C target loaded: {armC['target']['path']} "
            f"(meta {armC['target']['meta']})")
        base50 = measure(sd_s50, "s50_base", lean=SMOKE)
        armC["base"] = base50
        G_S50 = gate_vs(flat_cells(base50), E176_S50, "G_S50 (vs e176 +50)")
        if not G_S50["pass"]:
            raise RuntimeError("arm C: the +50 checkpoint failed its gate vs "
                               "e176's stored +50 cells")
        # THE RESTORE: root's MLP+LN into the +50 net (e178's arm, new target)
        sd_armC, g_armC = restore_class(sd_s50, sd_root, CLASSES["mlp_ln"],
                                        "mlp_ln")
        gates_surg["armC_mlp_ln"] = g_armC
        if not g_armC["pass"]:
            raise RuntimeError(f"class gate FAILED arm C: {g_armC}")
        armC["dial"] = measure(sd_armC, "s50_mlp_ln", lean=SMOKE)
        # all-restored sanity gate (e178's artifact-bounding convention)
        sd_allC, g_allC = restore_class(sd_s50, sd_root, CLASSES["all"], "all")
        gates_surg["armC_all"] = g_allC
        if not g_allC["pass"]:
            raise RuntimeError(f"all-restored gate FAILED arm C: {g_allC}")
        bit_allC = all(torch.equal(sd_allC[k], sd_root[k]) for k in sd_allC)
        dial_allC = measure(sd_allC, "s50_all", lean=SMOKE)
        diffs_all = {k: abs(flat_cells(dial_allC)[k] - root_cells[k])
                     for k in root_cells if k in flat_cells(dial_allC)}
        G_ALLC = {"sd_bit_equal_root": bool(bit_allC),
                  "max_cell_diff": max(diffs_all.values()),
                  "cell_diffs": diffs_all,
                  "pass": bool(bit_allC
                               and max(diffs_all.values()) < G_FALLBACK_TOL)}
        log(f"ARM C all-restored sanity: sd==root bits {bit_allC}, max dial "
            f"diff {max(diffs_all.values()):.2e} -> "
            + ("PASS" if G_ALLC["pass"] else "FAIL"))
        if not G_ALLC["pass"]:
            raise RuntimeError("arm C all-restored sanity gate FAILED")
        armC.update({
            "cells_base": flat_cells(base50),
            "cells_restore": flat_cells(armC["dial"]),
            "cells_all": flat_cells(dial_allC),
            "gate_restore": g_armC, "gate_all": g_allC, "G_S50": G_S50,
            "G_ALL_RESTORED": G_ALLC,
            "refs": {"e178_plus300": E178_RESTORE_REF,
                     "e173_forward": E173_FORWARD_REF},
        })
        c50 = armC["cells_restore"]
        r50b = armC["cells_base"]
        armC["retention_gm12"] = ((c50["gm12"] - r50b["gm12"])
                                  / max(root_cells["gm12"] - r50b["gm12"],
                                        1e-12))
        armC["delta_ce_vs_base"] = c50["ce_r"] - r50b["ce_r"]
        e178_g = E178_RESTORE_REF["gm12"]
        in_iface = bool(abs(c50["gm12"] - e178_g) <= IFACE_BAND)
        deep = bool(c50["gm12"] >= DEEP_CKPT_BAR)
        if deep:
            armC["reading"] = "WASH-DEPTH ARTIFACT"
            armC["reading_clause"] = (
                f"the +50 restore reopened g-12 to {c50['gm12']:.4f} "
                f">= {DEEP_CKPT_BAR} (toward e173's forward 0.7818) while "
                f"e178's +300 restore sat at {e178_g:.4f} — e178's half-fact "
                f"was the DEEPER wash: at +50 the class-external context "
                f"still supports the restored MLP+LN; recovery scales with "
                f"wash depth, not with the interface alone.")
        elif in_iface:
            armC["reading"] = "INTERFACE/GAIN-SET (the half-fact)"
            armC["reading_clause"] = (
                f"the +50 restore landed at g-12 {c50['gm12']:.4f}, inside "
                f"e178's +300 band ({e178_g:.4f} +/- {IFACE_BAND}) — the "
                f"recovery ceiling is set by the HYBRID INTERFACE (root "
                f"MLP+LN computing inside a washed attn/wpe/io context), "
                f"NOT by wash depth: the 'half-fact' is an interface/gain-"
                f"set property of the surgery, present at both wash depths.")
        else:
            armC["reading"] = "MIXED"
            armC["reading_clause"] = (
                f"the +50 restore landed at g-12 {c50['gm12']:.4f} — between "
                f"e178's band ({e178_g:.4f} +/- {IFACE_BAND}) and the "
                f"{DEEP_CKPT_BAR} wash-depth pole; both factors contribute; "
                f"numbers reported, no clean fork.")
        log(f"ARM C: +50 base g-12 {r50b['gm12']:.4f} -> MLP+LN restore "
            f"{c50['gm12']:.4f} (retention {100 * armC['retention_gm12']:.1f}%; "
            f"e178 +300 ref {e178_g:.4f}; e173 forward ref "
            f"{E173_FORWARD_REF['gm12']:.4f}) -> {armC['reading']}")
    else:
        armC["skip_reason"] = f"{S50_CK} not on disk"
        log(f"ARM C SKIPPED: {armC['skip_reason']}")

    # =====================================================================
    # TRAJECTORIES + ADJUDICATION (registered clauses; no shopping)
    # =====================================================================
    def trace_from(batteries: dict, steps: list) -> list:
        rows = []
        for s in steps:
            b = batteries["root" if s == 0 else str(s)]
            c = flat_cells(b)
            row = {"freeze_steps": s, **c,
                   "retention_vs_root_gm12": c["gm12"] / ROOT_GM12,
                   "retention_vs_root_g0": c["g0"] / ROOT_G0}
            rows.append(row)
        return rows

    stepsA = [0] + sorted(armA["sds"])
    traceA = trace_from(batteriesA, stepsA)
    stepsB = [0] + sorted(armB["sds"])
    # arm B's step-0 cells = the root's (same net, same batteries)
    traceB = trace_from({"root": root, **batteriesB}, stepsB)

    gA_seq = [r["gm12"] for r in traceA]
    g0A_seq = [r["g0"] for r in traceA]
    ck_idxA = [i for i, r in enumerate(traceA) if r["freeze_steps"] > 0]
    gA_50 = next((r["gm12"] for r in traceA if r["freeze_steps"] == 50),
                 traceA[-1]["gm12"])
    gA_min = min(gA_seq[i] for i in ck_idxA)

    neutral_dissolves = bool(gA_50 <= SHUT_BAR)
    neutral_survives = bool(gA_min >= SURVIVE_BAR)
    earliest_under = next((traceA[i]["freeze_steps"] for i in ck_idxA
                           if gA_seq[i] <= SHUT_BAR), None)

    # CE-AT-DISSOLUTION per stream (the R50 splice critique)
    def ce_at_dissolution(traj_light, trace):
        step_u = None
        for r in trace:
            if r["freeze_steps"] > 0 and r["gm12"] <= SHUT_BAR:
                step_u = r["freeze_steps"]
                break
        if step_u is None:
            return {"dissolved": False}
        light = next((t for t in traj_light if t["step"] == step_u), None)
        row = next(r for r in trace if r["freeze_steps"] == step_u)
        return {"dissolved": True, "step": step_u,
                "gm12": row["gm12"], "g0": row["g0"],
                "ce_r": row["ce_r"],
                "in_batch_corpus_ce": light["corpus_ce"] if light else None,
                "note": "CE_R + the in-batch corpus CE at the FIRST "
                        "checkpoint under the 0.27 bar — the wash's price at "
                        "the moment of dissolution, not the recovered value"}

    ceD_A = ce_at_dissolution(armA["traj"], traceA)
    ceD_B = ce_at_dissolution(armB["traj"], traceB)
    # the original stream's dissolution: main run first-under-bar = +50; the
    # SMOKE fine steps (the only fine-step data that stream has) co-reported
    ceD_orig = ce_at_dissolution(
        [], [{"freeze_steps": s, "gm12": g, "g0": g0, "ce_r": c}
             for s, g, g0, c in zip(E176_MAIN["freeze_steps"],
                                    E176_MAIN["base_gm12"],
                                    E176_MAIN["base_g0"],
                                    E176_MAIN["ce_r"])])
    ceD_orig["smoke_fine_steps"] = {
        "steps": E176_SMOKE["freeze_steps"],
        "gm12": E176_SMOKE["base_gm12"], "ce_r": E176_SMOKE["ce_r"],
        "note": "e176's only fine-step trajectory is its SMOKE run "
                "(s2 g-12 0.0882 at CE_R 2.0001) — the R50 auditor's splice "
                "attribution; arm A measures {1,2,4} in the MAIN run"}

    cond = {
        "NEUTRAL_DISSOLVES": {
            "bar": SHUT_BAR, "gA_at_50": gA_50, "clause": gA_50 <= SHUT_BAR,
            "earliest_ck_le_bar": earliest_under,
            "fires": neutral_dissolves},
        "NEUTRAL_SURVIVES": {
            "bar": SURVIVE_BAR, "gA_min_over_ckpts": gA_min,
            "all_ckpts_ge_bar": [r["freeze_steps"] for r in traceA
                                 if r["freeze_steps"] > 0
                                 and r["gm12"] < SURVIVE_BAR] == [],
            "below_bar_ckpts": [r["freeze_steps"] for r in traceA
                                if r["freeze_steps"] > 0
                                and r["gm12"] < SURVIVE_BAR],
            "fires": neutral_survives},
    }
    lastA = traceA[-1]
    if neutral_dissolves:
        verdict = "NEUTRAL-DISSOLVES"
        clause = (f"the fact dissolves under the truly neutral stream too: "
                  f"g-12 {gA_50:.4f} <= {SHUT_BAR} at +50 (earliest "
                  f"checkpoint <= bar: +{earliest_under}; trace "
                  + " -> ".join(f"+{r['freeze_steps']}:{r['gm12']:.4f}"
                                for r in traceA)
                  + f") — the wash was NOT extinction-specific; the "
                  f"activity-dependence noun is licensed (W019 strengthened "
                  f"as a bounded reading): remove rehearsal and ordinary "
                  f"corpus pressure alone dissolves the consolidated fact.")
    elif neutral_survives:
        verdict = "NEUTRAL-SURVIVES"
        clause = (f"the fact survives the neutral stream: g-12 >= "
                  f"{SURVIVE_BAR} at every checkpoint (min {gA_min:.4f}; "
                  f"+300 {lastA['gm12']:.4f}; held30 "
                  f"{lastA['held30_gm12']:.4f}/{lastA['held30_g0']:.4f}) — "
                  f"e176's wash was EXTINCTION (the install's own "
                  f"name-deleted junction windows), not disuse; the honest "
                  f"finding becomes: memories cannot survive their own "
                  f"teaching contexts shown without the name.")
    else:
        verdict = "SLOW-WASH"
        clause = (f"neither registered bar fired: g-12 at +50 {gA_50:.4f} "
                  f"(> {SHUT_BAR}) but the minimum over checkpoints "
                  f"{gA_min:.4f} < {SURVIVE_BAR}; trace "
                  + " -> ".join(f"+{r['freeze_steps']}:{r['gm12']:.4f}"
                                for r in traceA)
                  + f" — SLOW-WASH texture: the neutral stream erodes the "
                  f"fact on a slower clock than the extinction stream (full "
                  f"trajectory reported, no bar shopping).")

    # arm B rider reading (wash-rate scaling; report-only)
    gB_50 = next((r["gm12"] for r in traceB if r["freeze_steps"] == 50),
                 None)
    gB_300 = traceB[-1]["gm12"] if traceB else None
    b_first_under = next((r["freeze_steps"] for r in traceB
                          if r["freeze_steps"] > 0
                          and r["gm12"] <= SHUT_BAR), None)
    fmtB = lambda v: "n/a" if v is None else f"{v:.4f}"
    if traceB:
        if b_first_under is not None:
            armB_read = (f"the ORIGINAL stream at lr {LR_B} still dissolves "
                         f"(first checkpoint <= {SHUT_BAR}: +{b_first_under}; "
                         f"+50 {fmtB(gB_50)}, +300 {fmtB(gB_300)}) — the wash "
                         f"survives a 10x smaller step size; e176's "
                         f"two-step clock was the lr-1e-3 optimizer's, not "
                         f"the stream's alone")
        elif gB_300 is not None and gB_300 >= SURVIVE_BAR:
            armB_read = (f"the ORIGINAL stream at lr {LR_B} does NOT "
                         f"dissolve within 300 steps (g-12 +50 {fmtB(gB_50)}, "
                         f"+300 {fmtB(gB_300)} >= {SURVIVE_BAR}) — the wash "
                         f"rate scales with the optimizer: extinction needs "
                         f"the lr-1e-3 step size to complete in <=300 steps")
        else:
            armB_read = (f"the ORIGINAL stream at lr {LR_B} lands in the gap "
                         f"band (g-12 +50 {fmtB(gB_50)}, +300 {fmtB(gB_300)})"
                         f" — partial wash at 1/10 the step size; rate scales "
                         f"with the optimizer without a clean bar")
    else:
        armB_read = "arm B trajectory unavailable"
    log("=" * 78)
    log(f"E176N VERDICT: {verdict}")
    log(f"  arm A (neutral) g-12 trace: " + " -> ".join(
        f"+{r['freeze_steps']}:{r['gm12']:.4f}" for r in traceA))
    log(f"  arm A g0 trace:             " + " -> ".join(
        f"+{r['freeze_steps']}:{r['g0']:.4f}" for r in traceA))
    log(f"  arm B (lr1e-4) g-12 trace:  " + " -> ".join(
        f"+{r['freeze_steps']}:{r['gm12']:.4f}" for r in traceB))
    log(f"  CE-at-dissolution: A {ceD_A} | B {ceD_B}")
    log(f"  arm C: {armC.get('reading', 'SKIPPED')} — "
        f"{armC.get('reading_clause', armC.get('skip_reason', ''))}")
    log(f"  {clause}")
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e176n_neutral_wash",
        "date": common.now_iso(),
        "registration": ("NO QUEUE ROW — the R50 critic's critical discharge "
                         "(commit e5df61c/b4c6a4c) dispatched the bars in "
                         "the mission text; frozen verbatim in the module "
                         "docstring before compute; e177's fold waits on "
                         "this cell"),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("does the consolidated fact ALSO dissolve under a truly "
                     "NEUTRAL stream (e170's anchors: 16 plain-corpus "
                     "windows, 0/16 junction coverage), or was e176's wash "
                     "targeted EXTINCTION by the install's own name-deleted "
                     "junction windows?"),
        "root": f"runs/checkpoints/{ROOT_CK} (gated vs e151 before-cells, "
                f"max|diff| {G_ROOT['max_abs_diff']:.2e})",
        "root_meta": root_meta,
        "armA_neutral": {
            "desc": "ONE plain-corpus freeze with e170's NEUTRAL anchors: "
                    "batch 32 = 16 neutral-bank draws + 16 random corpus "
                    "windows, full-token CE (NO fact windows, NO name "
                    "tokens, NO mask), AdamW (0.9,0.95) wd 0.1 lr 1e-3 "
                    "constant, clip 1.0, seed 10902 (= e176's draw "
                    "sequence; only anchor content differs)",
            "ckpt_steps": list(CK_A), "steps_ran": armA["steps_ran"],
            "seed": armA["seed"], "lr": armA["lr"],
            "traj": armA["traj"], "zeph_violations": armA["zeph_violations"],
            "missing_checkpoints": [s for s in CK_A
                                    if s not in armA["sds"]],
        },
        "armB_lr1e4": {
            "desc": "the LR RIDER: the ORIGINAL stream (e161/e176's anchor "
                    "bank — the install's own name-deleted host windows) at "
                    "lr 1e-4, 300 steps, seed 10902 (same draw sequence as "
                    "e176 at lr 1e-3)",
            "ckpt_steps": list(CK_B), "steps_ran": armB["steps_ran"],
            "seed": armB["seed"], "lr": armB["lr"],
            "traj": armB["traj"], "zeph_violations": armB["zeph_violations"],
            "missing_checkpoints": [s for s in CK_B
                                    if s not in armB["sds"]],
            "reading": armB_read,
            "ce_at_dissolution": ceD_B,
        },
        "armC_restore50": armC,
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
                     "geometries_measured": {"novel": [-12, 12],
                                             "trained_g0": 0},
                     "measure_dial": "e176's measure() minus the 183-span "
                                     "census (e178's recorded deviation)"},
        "gates": {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                  "G_POOL": G_POOL, "G_ANCHOR": G_ANCHOR,
                  "G_ANCHFREE_ORIG": G_ANCHFREE_ORIG, "G_ROOT": G_ROOT,
                  "G_DRAWFREE_A": G_DRAWFREE_A, "G_DRAWFREE_B": G_DRAWFREE_B,
                  "G_SURG": gates_surg},
        "trace_armA": traceA,
        "trace_armB": traceB,
        "trace_summary": {
            "armA_steps": stepsA,
            "armA_gm12": [r["gm12"] for r in traceA],
            "armA_g0": [r["g0"] for r in traceA],
            "armA_gp12": [r["gp12"] for r in traceA],
            "armA_held30_gm12": [r["held30_gm12"] for r in traceA],
            "armA_held30_g0": [r["held30_g0"] for r in traceA],
            "armA_row0": [r.get("row0_strength") for r in traceA],
            "armA_dall_g0": [r.get("dall_g0") for r in traceA],
            "armA_ce_r": [r["ce_r"] for r in traceA],
            "armB_steps": stepsB,
            "armB_gm12": [r["gm12"] for r in traceB],
            "armB_g0": [r["g0"] for r in traceB],
            "armB_ce_r": [r["ce_r"] for r in traceB],
            "e176_main_steps": E176_MAIN["freeze_steps"],
            "e176_main_gm12": E176_MAIN["base_gm12"],
            "e176_main_g0": E176_MAIN["base_g0"],
            "e176_smoke_steps": E176_SMOKE["freeze_steps"],
            "e176_smoke_gm12": E176_SMOKE["base_gm12"],
            "e176_smoke_ce_r": E176_SMOKE["ce_r"],
        },
        "batteries_armA": batteriesA,
        "batteries_armB": batteriesB,
        "references": {
            "e176_main": {"ref": E176_MAIN, "provenance": src_e176,
                          "note": "the original extinction-confounded stream "
                                  "(lr 1e-3) — the comparison arm"},
            "e176_smoke": {"ref": E176_SMOKE, "provenance": src_e176s,
                           "note": "e176's ONLY fine-step trajectory (the "
                                   "smoke run; the R50 splice attribution)"},
            "e178_plus300_restore": {"ref": E178_RESTORE_REF,
                                     "provenance": src_e178},
            "e173_forward_restore": E173_FORWARD_REF,
        },
        "ce_at_dissolution": {"armA_neutral": ceD_A,
                              "armB_lr1e4": ceD_B,
                              "e176_original": ceD_orig,
                              "critique": "the R50 splice critique: report "
                                          "the CE at the DISSOLUTION moment, "
                                          "not just recovered values"},
        "adjudication": {"conditions": cond, "verdict": verdict,
                         "clause": clause,
                         "armB_reading": armB_read,
                         "armC_reading": armC.get("reading"),
                         "armC_clause": armC.get("reading_clause")},
        "honesty_reflex": {
            "anchor_fidelity": (f"the neutral bank is e170's construction "
                                f"VERBATIM (seed {E170_ANCHOR_SEED}, "
                                f"rejection on FLORIZEL/ELIZABETH/ZEPH/"
                                f"MIRABEL in [s, s+257)): 0/16 host-content "
                                f"windows, 0/16 junctions covered (G_ANCHOR); "
                                f"the original bank rebuilt for the delta "
                                f"record: 16/16 host windows, 16/16 junctions "
                                f"— the delta between arm A and e176 is "
                                f"EXACTLY the anchor half's informational "
                                f"content; budget (16x256), structure, "
                                f"optimizer, seed and RNG draw sequence are "
                                f"identical"),
            "shared_stream_components": (
                f"the random-corpus half (16/32 per batch, unfiltered) is "
                f"IDENTICAL across e176/arm A/arm B and can itself draw "
                f"host-junction windows at ~{100 * bg_rate:.1f}%/window "
                f"(e170's background estimate) — a small extinction "
                f"background PRESENT IN ALL THREE STREAMS, not part of the "
                f"delta; host names are corpus-native; a NEUTRAL-SURVIVES "
                f"verdict therefore survives this background, while a "
                f"NEUTRAL-DISSOLVES verdict cannot fully exclude it as the "
                f"driver (the ~4800 random draws x ~{100 * bg_rate:.1f}% "
                f"chance each leave a non-trivial cumulative junction "
                f"exposure through the random half alone)"),
            "single_seed": "ONE trajectory per training arm (seed 10902, the "
                "same draw sequence as e176 — the arms differ from e176 only "
                "in anchor content (A) or lr (B)); n=1 per cell; the fine "
                "steps {1,2,4} are this trajectory's, not a replicated law "
                "(e152R showed seed-to-seed timing spans an order of "
                "magnitude on related washes)",
            "original_stream_fine_steps": "e176's fine-step trajectory comes "
                "from its SMOKE run (4 steps, different total schedule than "
                "the main run's 300); arm A's fine steps are measured inside "
                "its own main run — the fine-step COMPARISON across streams "
                "is smoke-vs-main and carries that caveat",
            "arm_c_n1": "arm C is one deterministic surgery on one +50 "
                "checkpoint (n=1); its fork is an annotation of e178's "
                "TEXTURE, not a re-adjudication; the all-restored gate "
                "(sd==root bit-exact) bounds the machinery",
            "thread_bit_drift": "evals at 4 threads vs e151's stored cells "
                "can drift low-order bits; gates report both the 5e-6 bit "
                "flag and the 0.05 fallback tolerance",
            "bars_anchored": "the dissolve/survive bars are the same "
                "absolute home-battery bars e176/e161/e158 used (0.27 SHUT / "
                "0.50 SURVIVE) — the original stream's death and the neutral "
                "stream's fate are measured on one ruler",
        },
        "trims": trims, "deviations": deviations,
        "ckpt_inventory": CKPT_INVENTORY,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": int(net0.num_params()),
                   "train_device": "cpu", "eval_device": "cpu",
                   "torch_threads": torch.get_num_threads(), "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "neutral_wash.png", traceA, traceB, armA["traj"], armB["traj"],
         armC, cond, verdict, clause, armB_read, ceD_A, ceD_B)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'neutral_wash.png'}, "
        f"{len(CKPT_INVENTORY)} checkpoints in runs/checkpoints/e176n_*.pt")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot

def plot(path, traceA, traceB, trajA, trajB, armC, cond, verdict, clause,
         armB_read, ceD_A, ceD_B):
    """THE figure: the three-stream comparison (original / neutral / lr1e-4),
    the anatomy + CE-at-dissolution, the fine-step zoom, and the
    restore-into-+50 inset with the verdict."""
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 10.5))
    xsA = [r["freeze_steps"] for r in traceA]
    xsB = [r["freeze_steps"] for r in traceB]
    xe = E176_MAIN["freeze_steps"]
    xes = E176_SMOKE["freeze_steps"]

    # (0,0) THE THREE-STREAM COMPARISON (g-12 headline + g0)
    ax = axes[0, 0]
    ax.plot(xe, E176_MAIN["base_gm12"], "v--", lw=1.6, ms=7, color="crimson",
            alpha=0.65, label="e176 ORIGINAL g-12 (extinction anchors, lr 1e-3)")
    ax.plot(xe, E176_MAIN["base_g0"], "v--", lw=1.1, ms=5, color="tab:blue",
            alpha=0.45, label="e176 original g0")
    ax.plot(xsB, [r["gm12"] for r in traceB], "s-", ms=7, lw=2.0,
            color="darkorange", label=f"e176N LR RIDER g-12 (original stream, lr {LR_B})")
    ax.plot(xsB, [r["g0"] for r in traceB], "s-", ms=4, lw=1.0,
            color="darkorange", alpha=0.45, label="lr rider g0")
    ax.plot(xsA, [r["gm12"] for r in traceA], "o-", ms=8, lw=2.6,
            color="seagreen", label="e176N NEUTRAL g-12 (0/16 junctions, lr 1e-3) HEADLINE")
    ax.plot(xsA, [r["g0"] for r in traceA], "^-", ms=6, lw=1.6,
            color="mediumseagreen", label="neutral g0")
    for yv, col, lbl in ((SURVIVE_BAR, "seagreen",
                          f"{SURVIVE_BAR} SURVIVES bar (through +300)"),
                         (SHUT_BAR, "tab:purple",
                          f"{SHUT_BAR} DISSOLVES bar (by +50)")):
        ax.axhline(yv, ls="--", lw=1.1, color=col, alpha=0.8, label=lbl)
    ax.annotate(f"root g-12 {traceA[0]['gm12']:.3f}",
                (0, traceA[0]["gm12"]), textcoords="offset points",
                xytext=(6, 4), fontsize=7.5, color="seagreen")
    ax.annotate(f"e176 orig +50 {E176_MAIN['base_gm12'][1]:.3f}",
                (E176_MAIN["freeze_steps"][1], E176_MAIN["base_gm12"][1]),
                textcoords="offset points", xytext=(-10, 8), fontsize=7,
                color="crimson", alpha=0.8)
    ax.set_xlabel("plain-corpus freeze steps from the root (step 0 = root; "
                  "same seed/draw sequence in every stream)")
    ax.set_ylabel("absolute mean p(Z), install-60 battery")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7.0, loc="center right")
    ax.set_title("THE THREE-STREAM COMPARISON — the neutral wash control "
                 f"-> {verdict}", fontsize=10)

    # (0,1) neutral-stream anatomy + CE_R (with the dissolution marker)
    ax = axes[0, 1]
    has_anatomy = traceA[0].get("row0_strength") is not None
    if has_anatomy:
        ax.plot(xsA, [r["row0_strength"] for r in traceA], "^-", ms=6,
                lw=1.8, color="tab:cyan", label="neutral row-0 (the sink)")
        ax.plot(xe, E176_MAIN["row0_strength"], "v--", ms=5, lw=1.1,
                color="tab:cyan", alpha=0.45, label="e176 row-0")
        ax.plot(xsA, [r["dall_g0"] for r in traceA], "s-", ms=5, lw=1.6,
                color="tab:blue", label="neutral D-all g0")
        ax.plot(xe, E176_MAIN["dall_g0"], "v--", ms=5, lw=1.1,
                color="tab:blue", alpha=0.45, label="e176 D-all")
        ax.plot(xsA, [r["held30_gm12"] for r in traceA], "D-", ms=5, lw=1.4,
                color="crimson", alpha=0.8, label="neutral held30 g-12")
    axr = ax.twinx()
    axr.plot(xsA, [r["ce_r"] for r in traceA], "k:o", ms=5, lw=1.3,
             label="neutral CE_R")
    axr.plot(xsB, [r["ce_r"] for r in traceB], "k:s", ms=4, lw=1.0,
             alpha=0.5, label="lr-rider CE_R")
    axr.set_ylabel("CE_R")
    if ceD_A.get("dissolved"):
        ax.axvline(ceD_A["step"], color="tab:purple", ls=":", lw=1.6,
                   alpha=0.85)
        cce = ceD_A.get("in_batch_corpus_ce")
        axr.annotate(
            f"DISSOLUTION +{ceD_A['step']}: g-12 {ceD_A['gm12']:.4f}, "
            f"CE_R {ceD_A['ce_r']:.3f}"
            + (f", in-batch CE {cce:.3f}" if cce is not None else ""),
            (ceD_A["step"], ceD_A["ce_r"]), textcoords="offset points",
            xytext=(8, 10), fontsize=7.2, color="tab:purple",
            arrowprops=dict(arrowstyle="->", color="tab:purple", lw=0.8))
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("plain-corpus freeze steps (neutral stream)")
    ax.set_ylabel("strength / p(Z)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.0, loc="center right")
    ax.set_title("the neutral stream's anatomy + CE_R — the dotted line "
                 "marks DISSOLUTION (CE reported AT the moment, per the R50 "
                 "splice critique)", fontsize=9.5)

    # (1,0) the fine-step zoom (symlog): neutral vs e176's smoke fine steps
    ax = axes[1, 0]
    ax.plot(xes, E176_SMOKE["base_gm12"], "v--", ms=9, lw=1.8,
            color="crimson", alpha=0.7,
            label="e176 SMOKE g-12 (original stream, fine steps)")
    ax.plot(xes, E176_SMOKE["base_g0"], "v--", ms=6, lw=1.2,
            color="tab:blue", alpha=0.4, label="e176 smoke g0")
    fine = [r for r in traceA if r["freeze_steps"] <= 50]
    ax.plot([r["freeze_steps"] for r in fine], [r["gm12"] for r in fine],
            "o-", ms=9, lw=2.4, color="seagreen",
            label="e176N MAIN-RUN g-12 (neutral, fine steps)")
    ax.plot([r["freeze_steps"] for r in fine], [r["g0"] for r in fine],
            "^-", ms=7, lw=1.6, color="mediumseagreen",
            label="neutral g0 (main run)")
    for s, g in zip(xes, E176_SMOKE["base_gm12"]):
        if s > 0:
            ax.annotate(f"{g:.3f}", (s, g), textcoords="offset points",
                        xytext=(4, 6), fontsize=7, color="crimson")
    for r in fine:
        if 0 < r["freeze_steps"] <= 4:
            ax.annotate(f"{r['gm12']:.3f}", (r["freeze_steps"], r["gm12"]),
                        textcoords="offset points", xytext=(4, 6),
                        fontsize=7, color="seagreen")
    ax.axhline(SHUT_BAR, ls="--", lw=1.0, color="tab:purple", alpha=0.7)
    ax.axhline(SURVIVE_BAR, ls="--", lw=1.0, color="seagreen", alpha=0.7)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel("freeze steps (zoom 0..50; symlog scale)")
    ax.set_ylabel("mean p(Z) (symlog)")
    ax.legend(fontsize=7.0, loc="lower left")
    ax.set_title("the two-step collapse, fine steps: neutral (main run) vs "
                 "e176's smoke (its only fine-step data)", fontsize=10)

    # (1,1) the restore-into-+50 inset + the verdict panel
    ax = axes[1, 1]
    ax.axis("off")
    ytxt = 0.97
    ax.text(0.02, ytxt, "ARM C — THE RESTORE-INTO-+50 EVAL (attack 4):",
            fontsize=8.5, va="top", family="monospace", weight="bold")
    ytxt -= 0.045
    if armC.get("ran"):
        c50 = armC["cells_restore"]
        b50 = armC["cells_base"]
        ax.text(0.02, ytxt,
                f"  +50 base (on disk, gated)   g-12 {b50['gm12']:.4f}",
                fontsize=7.6, va="top", family="monospace")
        ytxt -= 0.034
        ax.text(0.02, ytxt,
                f"  +50 + root MLP+LN restore   g-12 {c50['gm12']:.4f}  "
                f"g0 {c50['g0']:.4f}  CE {c50['ce_r']:.4f}  "
                f"(retention {100 * armC['retention_gm12']:.1f}%)  "
                f"<- THIS CELL", fontsize=7.6, va="top", family="monospace",
                color="darkorange")
        ytxt -= 0.034
        ax.text(0.02, ytxt,
                f"  +300 + root MLP+LN (e178)   g-12 "
                f"{E178_RESTORE_REF['gm12']:.4f}  (the '~42%' pole)",
                fontsize=7.6, va="top", family="monospace")
        ytxt -= 0.034
        ax.text(0.02, ytxt,
                f"  e173 FORWARD restore        g-12 "
                f"{E173_FORWARD_REF['gm12']:.4f}  (the '~80%' pole)",
                fontsize=7.6, va="top", family="monospace")
        ytxt -= 0.048
        ax.text(0.02, ytxt, f"  READING: {armC['reading']}", fontsize=8.0,
                va="top", family="monospace", weight="bold")
        ytxt -= 0.036
        for wd in [armC["reading_clause"][i:i + 86]
                   for i in range(0, len(armC["reading_clause"]), 86)]:
            ax.text(0.02, ytxt, f"    {wd}", fontsize=6.8, va="top",
                    family="monospace")
            ytxt -= 0.028
    else:
        ax.text(0.02, ytxt, f"  SKIPPED: {armC.get('skip_reason')}",
                fontsize=7.6, va="top", family="monospace")
        ytxt -= 0.05
    ytxt -= 0.02
    ax.text(0.02, ytxt, "ARM B (LR RIDER):", fontsize=6.8, va="top",
            family="monospace")
    ytxt -= 0.028
    for wd in [armB_read[i:i + 86] for i in range(0, len(armB_read), 86)]:
        ax.text(0.02, ytxt, f"  {wd}", fontsize=6.8, va="top",
                family="monospace")
        ytxt -= 0.027
    ytxt -= 0.025
    ax.text(0.02, ytxt, f"E176N VERDICT: {verdict}", fontsize=9.0, va="top",
            family="monospace", weight="bold", color="darkred")
    ytxt -= 0.038
    for wd in [clause[i:i + 88] for i in range(0, len(clause), 88)]:
        ax.text(0.02, ytxt, f"  {wd}", fontsize=6.8, va="top",
                family="monospace")
        ytxt -= 0.027

    fig.suptitle("E176N — THE NEUTRAL WASH CONTROL: does the consolidated "
                 "fact dissolve under a truly neutral stream? "
                 f"-> {verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
