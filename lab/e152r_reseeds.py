"""E152R — THE DWELL RE-SEEDS (QUEUE row e152R): put n=3 on e152's session
n=1 trajectory claims — the 8-16-step cliff, the ~50-step dwell shelf, the
brake overshoot to -0.466 — plus settle e158's STRADDLING CELL (locked@band
g-12 read 0.546 on CPU / 0.458 on GPU in e158's two passes — device scatter
beyond the cross-device bound; a second and third seed canonize its
OPEN-vs-MID label).

WHAT RUNS (dispatch design):
  (A) TRACES x2 NEW SEEDS — the FULL e152 conversion trace protocol with
      re-teach seeds 10903 and 10904 (e152's n=1 used 10902): sequential
      continuation from the consolidated root, ONE 300-step training per
      seed, CPU state-dict snapshots at steps {8,16,32,64,128,300}, and the
      per-checkpoint dial battery (g-12/g+12/g0 base, site span census,
      A(129), D-all, CE_R).
  (B) THE STRADDLE SETTLER — e158 arm (b)'s locked@band protocol (offset-0
      pool, ZEPHYRA locked at x-cols 130..136, onset read row 129) at TWO
      additional seeds (10905, 10906); g-12 reported per seed together with
      the pass/device history (e158 pass 1 CPU 0.5464 OPEN / committed pass
      2 GPU 0.4584 MID).
  (C) THE INSERT (folded, cheap) — 10/12/14-step checkpoints ON ONE SEED
      (10903): folded INTO seed 10903's sequential trace (its ladder becomes
      {8,10,12,14,16,32,64,128,300}). Sequential continuation with the same
      seed coincides with an independent N-step run BY CONSTRUCTION (one
      seed-fixed generator, RNG-free evals — e152's own argument), so the
      fold costs one trajectory nothing and tightens the cliff's bracket
      between s8 and s16. Insert rows get the LIGHT battery and are EXCLUDED
      from the canonical bar arithmetic.

ROOT (mandated): runs/checkpoints/e131_consolidated_e113.pt — gated
bit-exact vs the stored e131/e151 root cells before any compute is trusted
(G_CONS/G_ROW0/G_A129/G_DALL refs are e151's stored values, which are
e131's).

RE-TEACH (e151/e152 VERBATIM = e143 locked replay = e109 arm-b = e119-L):
batch 32 = 16 install windows + 16 anchors (8 paired + 8 random), e043
token-level union CE on the 7 name-char targets (name-only mask), AdamW
(0.9,0.95) wd 0.1 constant lr 1e-3 clip 1.0, PLACEMENT offset j=+54:
ZEPHYRA locked at x-cols 184..190 (onset READ ROW 183; e120/e121/e133
splice-site geometry), pre-context 184 true tokens, continuation 65, zero
position variance by construction. Only the SEED changes (10903/10904).
Straddle arms: e158 arm (b) verbatim — offset 0, x-cols 130..136, read row
129, continuation 119, seeds 10905/10906.

REGISTERED PREDICTION (dispatch VERBATIM; adjudicate against exactly this;
no bar shopping):
  - DWELL-REPLICATES fires if: both new traces show cliff-at-8-16 + a
    shelf >= 0.4 through s32-64 + final <= 0.27 (the dwell is a phenomenon,
    not a path).
  - DWELL-SEED-DEPENDENT fires if: shelf/cliff timings or shapes differ
    qualitatively across seeds (the dwell is trajectory-specific texture).
  - BRAKE-OVERSHOOT-REPLICATES fires if: A(129) deepens below -0.35 at
    some mid-conversion checkpoint in >= 2/3 seeds.
  - STRADDLE-SETTLED: report the settled label for locked@band (OPEN if
    >= 2/3 new seeds >= 0.5; MID if straddling persists; the honest note
    either way).

OPERATIONALIZATIONS (frozen here BEFORE compute — the registered clauses
name cliff/shelf/shape without numbers; these fix them, they do not move
the bars):
  * retention(s) = g-12 install-60 battery mean_pz at checkpoint s / the
    root's 0.9155886769294739 (e152's denominator; the root is retention
    1.0 at step 0 by construction, excluded from checkpoint statistics).
    Retention (not absolute) is the trace variable — e152's headline
    instrument; absolutes co-reported everywhere.
  * "cliff-at-8-16" = ret(8) >= 0.70 AND ret(16) <= 0.50 (the door falls
    between the 8 and 16 checkpoints; e152 seed 10902 read 0.990/0.447).
  * "shelf >= 0.4 through s32-64" = min(ret(32), ret(64)) >= 0.40.
  * "final <= 0.27" = ret(300) <= 0.27.
  * DWELL-REPLICATES adjudicates ONLY on the two NEW seeds, and only at the
    six canonical checkpoints (insert rows excluded).
  * per-seed "cliff bracket" (the seed-dependence feature) = the adjacent
    canonical pair with the most negative retention diff, counted only if
    that diff <= -0.30, else "none". "shelf present" = min(ret32,ret64) >=
    0.40. "shape" feature = (cliff bracket, shelf present, final <= 0.27).
    DWELL-SEED-DEPENDENT fires iff the shape feature vector is NOT
    identical across the three seeds {10902 (e152, from
    runs/e152/metrics.json), 10903, 10904}. REPLICATES and SEED-DEPENDENT
    key on different seed sets and may BOTH fire (replicated on the two new
    seeds; 10902 differs) — both reported, never shopped.
  * "mid-conversion checkpoint" (the brake clause) = s in {8,16,32,64,128}
    (excludes the root and the released s300; e152's overshoot lived at
    s16..s128, min -0.466 at s128). Per-seed brake-overshoot = min A(129)
    over those checkpoints < -0.35. The 3 seeds counted = 10902 (e152's
    stored trace, which HAS it) + the two new traces; fires at >= 2/3.
  * STRADDLE-SETTLED: from the TWO NEW locked@band arms (absolute g-12,
    e158's instrument/bars): OPEN if >= 2/3 of new seeds >= 0.50 (with two
    new seeds: both); MID if straddling persists (not OPEN and every new
    seed > 0.27 — the cell keeps living between the bars); any new seed
    <= 0.27 is OUTSIDE the registered fork -> reported as SHUT-APPEARS with
    numbers (the honest note). Full pass/device history co-reported:
    e158 pass 1 CPU 0.5464 OPEN / committed pass 2 GPU 0.4584 MID / these
    arms with their devices.
  * Adjudication order for the OVERALL verdict string: DWELL-REPLICATES
    (with SEED-DEPENDENT co-fire noted) -> DWELL-SEED-DEPENDENT -> texture;
    BRAKE and STRADDLE adjudicate independently and join the composite.

DEVICE POLICY (dispatch, strict): GPU allowed ONLY on a pre-training quick
check — gpu_status() with util <= 85 AND temp <= 80 C (double-poll 5 s
apart, plus the lab's mem-headroom guard <= 85% of total). If ANY check
fails: PARK — every remaining training runs CPU for the rest of the
experiment (park-once, no re-probing; the traces run fine on CPU, ~35 min
each per e152's calibration 2079 s total). No bounded wait, never
contention with a returning user. cooldown(90 s) between trainings; <=180 s
GPU cap / 1500 s CPU cap per training; ALL readouts CPU-side (torch
threads 8, e143 convention), sequential. Nets are the mandated 2.7M
e131_consolidated line (the dispatch's '<=1M family' note is an envelope
statement — e143/e151/e152/e158 precedent; every gate reference and
lineage number of these cells lives on the 2.7M line). <=4 concurrent
jobs: this rig is ONE process (3 CPU agents live elsewhere).

BATTERY TRIM (disclosed, frozen): e152's per-checkpoint battery minus the
census183_onset census, the e150 mask probes, the norm ladder, and the
d129/d_r0 deletion rows — texture cells that feed no e152R bar. KEPT (the
dial list): base g-12/g0/g+12, CE_R, census183_span (site span strength +
site_pos, e151 convention), census_old (A(129), row-0, band121-129 max),
deletions {none, d_all, d183}. Straddle arms: base g-12/g0/g+12 + CE_R
only (the bar instrument + wreckage guard). Insert rows: light battery
(base + CE_R).

INSTRUMENT PROVENANCE: load_cpu/evl_load/battery_cell/battery_pz/
ce_fixed_cpu/val_windows/deleted_wpe/read_fact_at/row_census are
lab/e151_twodoor.py VERBATIM (e143/e065/e068/e109/e113/e116/e119/e131
lineage); finetune_trace is lab/e152_conversion_trace.py's finetune_trace
(e151's finetune_arm + RNG-free checkpoint snapshots) with the checkpoint
set and device parameterized — the per-step body VERBATIM (same draw
order/sizes: ix(16) then aj(8) then rj(8)). offset_pool is
lab/e158_2x2_completion.py's VERBATIM. Copied, not imported, to own the
device policy. Protocol rebuild: corpus seed 1337, SPLICE_RNG 24301 host
shuffle, install60/held30 split, mix gate — e143/e151 verbatim.

Outputs: runs/e152r/{metrics.json, reseeds.png}; checkpoints
runs/checkpoints/e152r_s{seed}_steps{N}.pt (traces),
runs/checkpoints/e152r_locked_band_s{seed}.pt (straddle arms). No NOTES/
THINKING/QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python e152r_reseeds.py    (E152R_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # GPU only via the
# strict per-training quick check below; park-once to CPU on any failure.

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

torch.set_num_threads(8)                              # e143 convention (24 cores)

import torch.nn.functional as F                       # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT, cooldown, gpu_status,    # noqa: E402
                    run_dir, save_json)
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E152R_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
ROOT_CK = "e131_consolidated_e113.pt"
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E152_METRICS = E43.REPO / "runs" / "e152" / "metrics.json"

# ---- seeds (the only deliberate change vs e152/e158) ---------------------------
TRACE_SEEDS = (10903, 10904)       # (A): the two new e152-protocol traces
INSERT_SEED = 10903                # (C): 10/12/14 folded into this trace
INSERT_STEPS = (10, 12, 14)
STRADDLE_SEEDS = (10905, 10906)    # (B): e158 arm-b protocol re-seeds
E152_SEED = 10902                  # the n=1 being replicated (e152's own)

# ---- re-teach placement (e152 verbatim; offset j=+54) --------------------------
RETEACH_J = 54                    # name x-cols 184..190, read rows 183..189
SITE_ADDR_ROW = 183               # the onset read row (= e120 SPLICE_ADDR_ROW)
SITE_Z_XCOL = PRE + RETEACH_J     # 184
SITE_CONT = BLOCK - PRE - RETEACH_J - len(NAME)     # 65-token continuation
SITE_ROWS = tuple(range(183, 190))                  # the 7 trained read rows
SHARED_CTR = (60, 100, 150, 160, 170, 200, 220)     # outside every trained band

# ---- straddle placement (e158 arm b verbatim; the home band, offset 0) ---------
BAND_J = 0
BAND_ADDR_ROW = PRE - 1 + BAND_J  # 129: onset read row of the home position
BAND_Z_XCOL = PRE + BAND_J        # 130: Z x-col

D_ALL = (121, 125, 129, 133, 137)                     # e113's fixed set
GEOS = (-12, 0, 12)               # novel x2 + the trained g0 (root trained {-8..+8})

ROWS_183 = (0, 1, 2) + (180, 181, 182) + SITE_ROWS + SHARED_CTR
ROWS_OLD = (0, 1, 2, 3, 4, 5, 6, 60, 100, 118, 119, 120) + tuple(range(121, 130))
if SMOKE:
    ROWS_183 = (0, 1, 2, 182) + SITE_ROWS + (60, 100)
    ROWS_OLD = (0, 1, 60, 121, 125, 129)

# ---- the trace: canonical checkpoint ladder + the folded insert ----------------
CKPT_CANON = (8, 16, 32, 64, 128, 300)
CKPT_BY_SEED = {
    TRACE_SEEDS[0]: tuple(sorted(set(CKPT_CANON) | set(INSERT_STEPS))),
    TRACE_SEEDS[1]: CKPT_CANON,
}
if SMOKE:                          # 4-step shakedown, nothing adjudicated
    CKPT_CANON = (2, 4)
    CKPT_BY_SEED = {s: (2, 4) for s in TRACE_SEEDS}
    INSERT_STEPS = ()

# ---- fine-tune envelope (e109 arm-b / e119-L / e143 / e151 verbatim) -----------
FT_LR = 1e-3
FT_STEPS = CKPT_CANON[-1]
NAME_BS, ANCH_BS = 16, 16
COOLDOWN_S = 90.0                 # dispatch envelope 60-120 s between trainings
GPU_CAP_S = 180.0                 # dispatch: <=180 s caps
CPU_CAP_S = 1500.0                # e152 calibration (300 CPU steps ~ 716 s)

# ---- gates / references (full precision, = e151's stored root cells) -----------
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05
G_CONS_REF_PZ = 0.7850371599197388            # e131/e151 none__g+0__install60
G_CONS_REF_CE = 1.663516640663147             # e131/e151 none__ce_r
G_R0_REF_MEAN = 0.7842019017236945            # e131/e151 consolidated census row 0
G_R0_REF_ZERO = 0.7316772222270098
G_A129_REF_MEAN = -0.1084650677318375         # e131/e151 consolidated census row 129
G_A129_REF_ZERO = -0.13237020391970877
G_DALL_REF = 0.9047248959541321               # e131/e151 d_all_e113__g+0__install60

# ---- e152's stored n=1 trace (the replication target; loaded at runtime) -------
ROOT_REF_GM12 = 0.9155886769294739            # e151 before g-12 (the denominator)
ROOT_REF_GP12 = 0.9478210210800171            # e151 before g+12 (co-report den.)
ROOT_REF_G0 = 0.7850371599197388              # e151 before g0
E152_A129_MIN_MID = None                      # filled from e152 metrics (expect ~-0.466)

# ---- registered bar constants (frozen; see OPERATIONALIZATIONS) ----------------
CLIFF_HI = 0.70                   # cliff-at-8-16: ret(8) >= this
CLIFF_LO = 0.50                   # cliff-at-8-16: ret(16) <= this
CLIFF_DROP = 0.30                 # cliff bracket feature: adjacent diff <= -this
SHELF_BAR = 0.40                  # shelf: min(ret32, ret64) >= this
FINAL_BAR = 0.27                  # final: ret(300) <= this
BRAKE_BAR = -0.35                 # brake overshoot: A(129) < this
MID_STEPS = (8, 16, 32, 64, 128)  # mid-conversion checkpoints for the brake
OPEN_BAR = 0.50                   # straddle cell bars (e158 verbatim)
SHUT_BAR = 0.27
SITE_CTRL_MULT = 2.0              # e139/e151 site convention: >= 2x control-max

# e158's stored locked@band history (the cell being settled)
E158_STRADDLE_HISTORY = [
    {"source": "e158 pass 1 (discarded, clause-prose bug)", "device": "cpu",
     "gm12": 0.5464, "label": "OPEN"},
    {"source": "e158 pass 2 (COMMITTED)", "device": "cuda",
     "gm12": 0.45835810899734497, "label": "MID"},
]

REGISTERED_PREDICTION = {
    "dwell_replicates": "DWELL-REPLICATES fires if: both new traces show "
        "cliff-at-8-16 + a shelf >= 0.4 through s32-64 + final <= 0.27 (the "
        "dwell is a phenomenon, not a path).",
    "dwell_seed_dependent": "DWELL-SEED-DEPENDENT fires if: shelf/cliff "
        "timings or shapes differ qualitatively across seeds (the dwell is "
        "trajectory-specific texture).",
    "brake_overshoot_replicates": "BRAKE-OVERSHOOT-REPLICATES fires if: "
        "A(129) deepens below -0.35 at some mid-conversion checkpoint in "
        ">= 2/3 seeds.",
    "straddle_settled": "STRADDLE-SETTLED: report the settled label for "
        "locked@band (OPEN if >= 2/3 new seeds >= 0.5; MID if straddling "
        "persists; the honest note either way).",
    "operationalizations": "retention(s) = g-12 battery mean_pz(s) / root "
        f"{ROOT_REF_GM12:.16f}; cliff-at-8-16 = ret8 >= {CLIFF_HI} AND ret16 "
        f"<= {CLIFF_LO}; shelf = min(ret32, ret64) >= {SHELF_BAR}; final = "
        f"ret300 <= {FINAL_BAR}; DWELL-REPLICATES adjudicates on the two NEW "
        "seeds at the six canonical checkpoints only (insert rows excluded); "
        "per-seed shape feature = (cliff bracket = adjacent canonical pair "
        f"with diff <= -{CLIFF_DROP}, shelf present, final shut); SEED-"
        "DEPENDENT fires iff the feature differs across {10902, 10903, "
        "10904}; brake mid-conversion checkpoints = " +
        f"{list(MID_STEPS)} (root and s300 excluded), per-seed overshoot = "
        f"min A(129) over them < {BRAKE_BAR}, seeds = e152's 10902 + the two "
        "new; STRADDLE from the two NEW locked@band arms (absolute g-12): "
        f"OPEN if >= 2/3 new seeds >= {OPEN_BAR}, MID if straddling persists "
        f"(not OPEN, all > {SHUT_BAR}), any <= {SHUT_BAR} outside the fork "
        "(reported); the four bars adjudicate independently.",
    "committed": "QUEUE e152R registered no committed branch (replication "
                 "run; T094 marks the dwell/brake [n=1] pending this run).",
}

trims: list[str] = []
deviations: list[str] = [
    "Battery trim (disclosed, frozen): e152's per-checkpoint battery minus "
    "census183_onset, the e150 mask probes, the norm ladder, and the d129/"
    "d_r0 deletion rows — texture cells feeding no e152R bar; kept the full "
    "dial list (g-12/g+12/g0, site span + site_pos, A(129), D-all, CE_R, "
    "plus d183). Straddle arms: base + CE_R only. Insert rows: light battery.",
    "The 10/12/14-step insert is FOLDED into seed 10903's trace (its ladder "
    "becomes {8,10,12,14,16,32,64,128,300}) rather than run as a separate "
    "training: sequential continuation with a seed-fixed generator coincides "
    "with an independent N-step run BY CONSTRUCTION (e152's own argument, "
    "RNG-free snapshots/evals), so the fold changes nothing about the "
    "trajectory and saves one training; insert rows carry the light battery "
    "and are excluded from the canonical bar arithmetic.",
    "Nets are the mandated 2.7M e131_consolidated line (the dispatch's "
    "'<=1M family' note is an envelope statement; e143/e151/e152/e158 "
    "precedent — every gate reference lives on this line).",
    "Trace seeds 10903/10904 and straddle seeds 10905/10906 (adjacent to the "
    "lineage's 10902; distinct across arms so no cell shares a seed with "
    "another cell of this run).",
    "GPU float nondeterminism: e152's seed-10902 trace ran CPU (parked) and "
    "e158's committed cells ran GPU; if this run parks mid-way, devices mix "
    "across seeds — recorded per cell in the honesty reflex (e158's "
    "two-pass scatter 0.546/0.458 is exactly the sensitivity motivating the "
    "settler).",
    "Smoke mode trims: 4-step trainings, checkpoints {2,4}, reduced census "
    "rows, nothing adjudicated.",
]


# ------------------------------------------------------------------ device pick

GPU_PARKED = False
PARK_REASON = None


def gpu_quick_ok(s: dict) -> bool:
    """The dispatch's strict gate: util <= 85 AND temp <= 80C (plus the lab's
    mem-headroom guard)."""
    return bool(s["util"] <= 85 and s["temp"] <= 80
                and (s["mem_total"] == 0 or s["mem_used"] <= 0.85 * s["mem_total"]))


def pick_dev(tag: str) -> torch.device:
    """Quick pre-training check; PARK-ONCE: any failure parks every remaining
    training to CPU (no re-probing, no bounded wait, never contention)."""
    global GPU_PARKED, PARK_REASON
    if GPU_PARKED:
        log(f"[gpu] '{tag}' CPU (PARKED: {PARK_REASON})")
        return CPU
    if not torch.cuda.is_available():
        GPU_PARKED, PARK_REASON = True, "no CUDA"
        return CPU
    s1 = gpu_status()
    if gpu_quick_ok(s1):
        time.sleep(5)
        s2 = gpu_status()
        if gpu_quick_ok(s2):
            log(f"[gpu] '{tag}' may use GPU (util {s2['util']:.0f}% temp "
                f"{s2['temp']:.0f}C mem {s2['mem_used']:.0f}/"
                f"{s2['mem_total']:.0f}MB)")
            return torch.device("cuda")
        s1 = s2
    GPU_PARKED = True
    PARK_REASON = f"quick check failed: {s1}"
    log(f"[gpu] PARK — '{tag}' and all remaining trainings run CPU ({s1})")
    return CPU


# ------------------------------------------------------------------ instruments
# PROVENANCE: verbatim lineage — see the module docstring. Copied rather than
# imported (e131/e150 rigs force CUDA_VISIBLE_DEVICES=-1; this one owns the
# device policy). finetune_trace = e152's with ckpt set + device parameterized.

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
    """e131's read_fact_position VERBATIM ARITHMETIC (e151 copy): p(true name
    char) at positions addr_row..addr_row+6 over the pool windows."""
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


# ------------------------------------------------------------------ fine-tune

def finetune_trace(tag: str, net0: TinyGPT, pool_x: torch.Tensor,
                   pool_mask: torch.Tensor, anchor: torch.Tensor,
                   train_ids: torch.Tensor, r_eval_xy, f_eval_ids, zid: int,
                   seed: int, ckpt_steps: tuple[int, ...]):
    """e152's finetune_trace VERBATIM per-step body (e151's finetune_arm +
    RNG-free CPU snapshots at ckpt_steps), with the checkpoint set and the
    device parameterized (e152's module-global CKPT_SET/DEV passed in).
    The trajectory for a given seed is e152's BY CONSTRUCTION."""
    dev = pick_dev(tag)
    cap = GPU_CAP_S if dev.type == "cuda" else CPU_CAP_S
    ckpt_set = set(ckpt_steps)
    net = copy.deepcopy(net0).to(dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    n_pool = pool_x.shape[0]
    n_anc = anchor.shape[0]
    traj, sds, t_start = [], {}, time.time()
    step = 0
    evl = copy.deepcopy(net0)          # CPU eval twin
    for step in range(1, FT_STEPS + 1):
        ix = torch.randint(n_pool, (NAME_BS,), generator=gen)
        aj = torch.randint(n_anc, (ANCH_BS // 2,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (ANCH_BS // 2,),
                           generator=gen)
        nw = pool_x[ix].to(dev)
        anc = torch.cat([anchor[aj],
                         torch.stack([train_ids[s: s + BLOCK] for s in rj])], 0).to(dev)
        x = torch.cat([nw[:, :-1], anc[:, :-1]], 0)
        y = torch.cat([nw[:, 1:], anc[:, 1:]], 0)
        m = torch.zeros(NAME_BS + ANCH_BS, x.shape[1], dtype=torch.bool, device=dev)
        m[:NAME_BS] = pool_mask[ix].to(dev)
        logits, _ = net(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                              reduction="none").view(x.shape[0], x.shape[1])
        nm = nll[:NAME_BS][m[:NAME_BS]]
        cm = nll[NAME_BS:]
        loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step in ckpt_set:
            sd_cpu = {k: v.detach().cpu().clone()
                      for k, v in net.state_dict().items()}
            sds[step] = sd_cpu
            evl.load_state_dict(sd_cpu)
            evl.eval()
            bz = battery_cell(evl, f_eval_ids, zid)
            ce_r = ce_fixed_cpu(evl, *r_eval_xy)
            traj.append({"step": step, "p_z_mean": bz["mean_pz"],
                         "frac_argmax_z": bz["frac_argmax_z"], "ce_r": ce_r,
                         "elapsed_s": round(time.time() - t_start, 1)})
            log(f"  [{tag}] CKPT s{step:4d} p_z(site) {bz['mean_pz']:.4f} "
                f"argmaxZ {bz['frac_argmax_z']:.2f} CE_R {ce_r:.4f} "
                f"({traj[-1]['elapsed_s']:.0f}s)")
        if (time.time() - t_start) > cap:
            log(f"  [{tag}] time cap {cap:.0f}s at s{step}")
            trims.append(f"{tag}: time cap at step {step}")
            break
    net.eval()
    del net
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return {"sds": sds, "traj": traj, "steps_ran": step, "seed": seed,
            "device": str(dev), "time_cap_s": cap}


CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "e152r", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name}")


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e152r_smoke" if SMOKE else "e152r")
    log(f"E152R THE DWELL RE-SEEDS (smoke={SMOKE}) -> {rd}")
    log(f"compute: strict per-training GPU gate (park-once); gpu at start: "
        f"{gpu_status()}, cpu threads {torch.get_num_threads()}")
    log(f"traces: seeds {TRACE_SEEDS} at canonical ckpts {CKPT_CANON}"
        + (f" + insert {INSERT_STEPS} folded into seed {INSERT_SEED}"
           if INSERT_STEPS else ""))
    log(f"straddle: e158 arm-b protocol at seeds {STRADDLE_SEEDS}")

    # ---- e152's stored n=1 (the replication target)
    e152 = json.loads(E152_METRICS.read_text(encoding="utf-8"))
    e152_trace = e152["trace"]                      # rows incl. step 0
    e152_sum = e152["trace_summary"]
    global E152_A129_MIN_MID
    E152_A129_MIN_MID = min(r["A129"] for r in e152_trace
                            if r["steps"] in MID_STEPS)
    log(f"e152 n=1 loaded (seed {E152_SEED}): ret "
        + " ".join(f"s{r['steps']}:{r['retention_gm12']:.3f}"
                   for r in e152_trace)
        + f" | A(129) min over mid = {E152_A129_MIN_MID:+.4f}")

    # ---------------- protocol rebuild (e143/e151/e152 verbatim)
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)

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

    # ---------------- pools (e158's offset_pool verbatim, parameterized)
    def offset_pool(j: int):
        """e143/e158's offset-pool construction VERBATIM at total offset j:
        pre = 130+j true tokens, ZEPHYRA at x-cols (130+j)..(136+j),
        continuation 119-j; mask targets = read rows (129+j)..(135+j)."""
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

    pool_x, pool_mask = offset_pool(RETEACH_J)          # trace pool (j=+54)
    pool_band_x, pool_band_mask = offset_pool(BAND_J)   # straddle pool (j=0)

    G_GEO = {
        "trace_name_xcols": [SITE_Z_XCOL, SITE_Z_XCOL + L - 1],
        "trace_all_windows_name_in_place": bool(
            all(torch.equal(w[SITE_Z_XCOL: SITE_Z_XCOL + L], name_ids)
                for w in pool_x)),
        "trace_mask_targets_per_window": int(pool_mask[0].sum()),
        "trace_zero_variance": True,
        "band_name_xcols": [BAND_Z_XCOL, BAND_Z_XCOL + L - 1],
        "band_all_windows_name_in_place": bool(
            all(torch.equal(w[BAND_Z_XCOL: BAND_Z_XCOL + L], name_ids)
                for w in pool_band_x)),
        "band_mask_targets_per_window": int(pool_band_mask[0].sum()),
        "band_zero_variance": True,
        "pool_sizes": {"trace": int(pool_x.shape[0]),
                       "band": int(pool_band_x.shape[0])},
    }
    G_GEO["pass"] = bool(G_GEO["trace_all_windows_name_in_place"]
                         and G_GEO["trace_mask_targets_per_window"] == L
                         and G_GEO["band_all_windows_name_in_place"]
                         and G_GEO["band_mask_targets_per_window"] == L)
    assert G_GEO["pass"], f"pool geometry gate FAILED: {G_GEO}"
    log(f"pools: trace {tuple(pool_x.shape)} (locked, x-cols 184..190, read "
        f"row {SITE_ADDR_ROW}) | band {tuple(pool_band_x.shape)} (locked, "
        f"x-cols 130..136, read row {BAND_ADDR_ROW})")

    # anchor bank (e065/e109/e143/e151/e158 verbatim)
    anchor = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                          for p, _ in install_occ[:16]])

    # ---------------- batteries: ctx = train_text[p-PRE-j : p] (e119 verbatim)
    bat_ids = {}
    for j in GEOS:
        for tag_, occ in (("install60", install_occ), ("held30", held_occ)):
            cs = [train_text[p - PRE - j: p] for p, _ in occ]
            bat_ids[(j, tag_)] = torch.stack([corpus.encode(c) for c in cs])
    ids130 = bat_ids[(0, "install60")]             # e116/e131 battery verbatim
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    f_eval_trace = pool_x[:, :SITE_Z_XCOL]         # p(Z) read at row 183
    f_eval_band = ids130                           # e119-L convention

    # ---------------- root net + gates
    net0 = load_cpu(CKPT_DIR / ROOT_CK)
    sd_root = {k: v.clone() for k, v in net0.state_dict().items()}
    ev = copy.deepcopy(net0)
    bz0 = battery_cell(ev, ids130, zid)
    ce0 = ce_fixed_cpu(ev, *r_eval_xy)
    G_CONS = {"battery_pz": bz0["mean_pz"], "ref_pz": G_CONS_REF_PZ,
              "ce_r": ce0, "ref_ce": G_CONS_REF_CE,
              "bit_reproducible": bool(
                  abs(bz0["mean_pz"] - G_CONS_REF_PZ) < G_BIT_TOL
                  and abs(ce0 - G_CONS_REF_CE) < G_BIT_TOL),
              "pass": bool(abs(bz0["mean_pz"] - G_CONS_REF_PZ) < G_FALLBACK_TOL
                           and abs(ce0 - G_CONS_REF_CE) < G_FALLBACK_TOL)}
    log(f"G_CONS root: p(Z) {bz0['mean_pz']:.10f} (ref {G_CONS_REF_PZ:.10f}) "
        f"CE_R {ce0:.6f} (ref {G_CONS_REF_CE:.6f}): "
        f"{'PASS' if G_CONS['pass'] else 'FAIL'}")
    if not G_CONS["pass"]:
        raise RuntimeError("root checkpoint failed its gate vs e151 stored cells")
    del ev

    # =====================================================================
    # the measurement batteries (trimmed per the frozen disclosure)
    # =====================================================================
    DELS = {"none": (), "d_all": D_ALL, "d183": (SITE_ADDR_ROW,)}
    gates_surg: dict = {}

    def site_span_summary(cen):
        cm = max(cen["rows"][str(r)]["strength"] for r in SHARED_CTR
                 if str(r) in cen["rows"])
        sstr = max(cen["rows"][str(r)]["strength"] for r in SITE_ROWS)
        spos = any(cen["rows"][str(r)]["content"] and
                   cen["rows"][str(r)]["strength"] >= SITE_CTRL_MULT * cm
                   for r in SITE_ROWS)
        best = max(SITE_ROWS, key=lambda r: cen["rows"][str(r)]["strength"])
        return {"control_max": cm, "site_strength": sstr,
                "bar_2x_control": SITE_CTRL_MULT * cm, "site_pos": spos,
                "peak_row": int(best)}

    def measure_full(sd: dict, tag: str) -> dict:
        """The dial battery (trimmed e152 battery; see docstring)."""
        net = evl_load(sd)
        out: dict = {"tag": tag}
        out["base"] = {j: battery_cell(net, bat_ids[(j, "install60")], zid)
                       for j in GEOS}
        out["ce_r"] = ce_fixed_cpu(net, *r_eval_xy)
        log(f"[{tag}] base: " + " ".join(f"g{j:+d} {out['base'][j]['mean_pz']:.4f}"
                                         for j in GEOS)
            + f" | CE_R {out['ce_r']:.4f}")

        def span_fn(n):
            return read_fact_at(n, pool_x, name_ids, zid, SITE_ADDR_ROW,
                                SITE_Z_XCOL)["pname_mean_over7"]

        out["site_read"] = read_fact_at(net, pool_x, name_ids, zid,
                                        SITE_ADDR_ROW, SITE_Z_XCOL)
        cen = row_census(net, ROWS_183, span_fn)
        out["census183_span"] = cen
        out["site_span"] = site_span_summary(cen)
        log(f"[{tag}] site(span) strength {out['site_span']['site_strength']:+.4f} "
            f"@r{out['site_span']['peak_row']} (2x-ctrl "
            f"{out['site_span']['bar_2x_control']:.4f}) -> site_pos "
            f"{out['site_span']['site_pos']}")

        out["census_old"] = row_census(net, ROWS_OLD,
                                       lambda n: battery_pz(n, ids130, zid))
        co = out["census_old"]["rows"]
        out["old_band"] = {
            "base_pz": out["census_old"]["base_readout"],
            "row0_strength": co["0"]["strength"],
            "A129": co["129"]["strength"],
            "row0": co["0"], "row129": co["129"],
            "band121_129_max": max(co[str(r)]["strength"] for r in range(121, 130)
                                   if str(r) in co)}
        log(f"[{tag}] old band: row0 S {out['old_band']['row0_strength']:+.4f} | "
            f"A(129) {out['old_band']['A129']:+.4f} | band121-129 max "
            f"{out['old_band']['band121_129_max']:+.4f}")

        out["del_table"] = {}
        for dl, rows_ in DELS.items():
            if dl == "none":
                sd_d = {k: v.clone() for k, v in sd.items()}
                gate = {"rows": [], "pass": True, "note": "no deletion"}
            else:
                sd_d, gate = deleted_wpe(sd, rows_)
            gates_surg[f"{tag}__{dl}"] = gate
            if not gate["pass"]:
                raise RuntimeError(f"deletion gate FAILED {tag}/{dl}: {gate}")
            net.load_state_dict(sd_d)
            cell = {"g0": battery_cell(net, bat_ids[(0, "install60")], zid)}
            if dl == "d183":
                cell["gm12"] = battery_cell(net, bat_ids[(-12, "install60")], zid)
            out["del_table"][dl] = cell
        net.load_state_dict(sd)                    # restore
        log(f"[{tag}] deletions g0: " + " | ".join(
            f"{dl} {out['del_table'][dl]['g0']['mean_pz']:.3f}" for dl in DELS))
        del net
        return out

    def measure_light(sd: dict, tag: str) -> dict:
        """Base + CE_R only (insert rows; straddle arms)."""
        net = evl_load(sd)
        out: dict = {"tag": tag}
        out["base"] = {j: battery_cell(net, bat_ids[(j, "install60")], zid)
                       for j in GEOS}
        out["ce_r"] = ce_fixed_cpu(net, *r_eval_xy)
        log(f"[{tag}] base: " + " ".join(f"g{j:+d} {out['base'][j]['mean_pz']:.4f}"
                                         for j in GEOS)
            + f" | CE_R {out['ce_r']:.4f}")
        del net
        return out

    log("=" * 78)
    log("STEP-0 battery (root; also computes the G_ROW0/G_A129/G_DALL gates)")
    root = measure_full(sd_root, "root")
    G_ROW0 = {"mean": root["census_old"]["rows"]["0"]["mean"],
              "zero": root["census_old"]["rows"]["0"]["zero"],
              "ref_mean": G_R0_REF_MEAN, "ref_zero": G_R0_REF_ZERO,
              "pass": bool(abs(root["census_old"]["rows"]["0"]["mean"] - G_R0_REF_MEAN)
                           < G_FALLBACK_TOL
                           and abs(root["census_old"]["rows"]["0"]["zero"] - G_R0_REF_ZERO)
                           < G_FALLBACK_TOL)}
    G_A129 = {"mean": root["census_old"]["rows"]["129"]["mean"],
              "zero": root["census_old"]["rows"]["129"]["zero"],
              "ref_mean": G_A129_REF_MEAN, "ref_zero": G_A129_REF_ZERO,
              "pass": bool(abs(root["census_old"]["rows"]["129"]["mean"] - G_A129_REF_MEAN)
                           < G_FALLBACK_TOL
                           and abs(root["census_old"]["rows"]["129"]["zero"] - G_A129_REF_ZERO)
                           < G_FALLBACK_TOL)}
    G_DALL = {"dall_g0": root["del_table"]["d_all"]["g0"]["mean_pz"],
              "ref": G_DALL_REF,
              "pass": bool(abs(root["del_table"]["d_all"]["g0"]["mean_pz"]
                               - G_DALL_REF) < G_FALLBACK_TOL)}
    for gname, g in (("G_ROW0", G_ROW0), ("G_A129", G_A129), ("G_DALL", G_DALL)):
        log(f"{gname}: {'PASS' if g['pass'] else 'FAIL'}")
        if not g["pass"]:
            raise RuntimeError(f"{gname} failed vs e151 stored cells")
    log("gates: G_SPLICE, G_GEO, G_CONS, G_ROW0, G_A129, G_DALL all PASS")

    # =====================================================================
    # (A) THE TRACES x2 (+ the folded insert on seed 10903)
    # =====================================================================
    trace_runs: dict[int, dict] = {}
    for si, seed in enumerate(TRACE_SEEDS):
        if si > 0 and not SMOKE:
            log(f"[thermal] cooldown {COOLDOWN_S:.0f}s between trainings")
            cooldown(COOLDOWN_S)
        cks = CKPT_BY_SEED[seed]
        log("=" * 78)
        log(f"TRACE seed {seed}: {FT_STEPS}-step locked replay of ZEPHYRA at "
            f"rows 183..189 (x-cols 184..190), checkpoints at {list(cks)}")
        run = finetune_trace(f"trace_s{seed}", net0, pool_x, pool_mask, anchor,
                             train_ids, r_eval_xy, f_eval_trace, zid, seed, cks)
        for s in sorted(run["sds"]):
            save_ckpt(f"e152r_s{seed}_steps{s}", run["sds"][s],
                      {"desc": f"e131_consolidated_e113 + {s}-step locked "
                               f"replay of ZEPHYRA at read rows 183..189 "
                               f"(x-cols 184..190), name-only mask, seed "
                               f"{seed} — sequential-continuation snapshot "
                               f"of the single {FT_STEPS}-step run (e152R "
                               f"re-seed)",
                       "steps": s, "seed": seed,
                       "base": f"runs/checkpoints/{ROOT_CK}"})
        missing = [s for s in cks if s not in run["sds"]]
        if missing:
            trims.append(f"trace_s{seed}: checkpoints not reached {missing}")
        trace_runs[seed] = run

    # per-checkpoint batteries
    batteries: dict[int, dict] = {}
    for seed in TRACE_SEEDS:
        run = trace_runs[seed]
        batteries[seed] = {}
        log("=" * 78)
        for s in sorted(run["sds"]):
            is_insert = (not SMOKE) and s in INSERT_STEPS and seed == INSERT_SEED
            log(f"TRACE seed {seed} STEP-{s} battery "
                f"({'insert/light' if is_insert else 'full'})")
            batteries[seed][s] = (measure_light(run["sds"][s], f"s{seed}_{s}")
                                  if is_insert
                                  else measure_full(run["sds"][s], f"s{seed}_{s}"))

    # trace tables
    def trace_table(seed: int) -> list[dict]:
        rows = [{"steps": 0, "insert": False,
                 "base_gm12": ROOT_REF_GM12, "base_g0": ROOT_REF_G0,
                 "base_gp12": ROOT_REF_GP12, "retention_gm12": 1.0,
                 "retention_g0": 1.0, "retention_gp12": 1.0,
                 "ce_r": G_CONS_REF_CE,
                 "site_span_strength": root["site_span"]["site_strength"],
                 "site_pos_span": root["site_span"]["site_pos"],
                 "A129": G_A129_REF_ZERO,
                 "row0_strength": G_R0_REF_ZERO,
                 "dall_g0": G_DALL_REF}]
        for s in sorted(trace_runs[seed]["sds"]):
            b = batteries[seed][s]
            is_insert = (not SMOKE) and s in INSERT_STEPS and seed == INSERT_SEED
            row = {
                "steps": s, "insert": is_insert,
                "base_gm12": b["base"][-12]["mean_pz"],
                "base_g0": b["base"][0]["mean_pz"],
                "base_gp12": b["base"][12]["mean_pz"],
                "retention_gm12": b["base"][-12]["mean_pz"] / ROOT_REF_GM12,
                "retention_g0": b["base"][0]["mean_pz"] / ROOT_REF_G0,
                "retention_gp12": b["base"][12]["mean_pz"] / ROOT_REF_GP12,
                "ce_r": b["ce_r"],
            }
            if not is_insert:
                row.update({
                    "site_read_onset": b["site_read"]["pz_onset_mean"],
                    "site_span_strength": b["site_span"]["site_strength"],
                    "site_span_peak_row": b["site_span"]["peak_row"],
                    "site_span_bar_2x_control": b["site_span"]["bar_2x_control"],
                    "site_pos_span": b["site_span"]["site_pos"],
                    "A129": b["old_band"]["A129"],
                    "row0_strength": b["old_band"]["row0_strength"],
                    "band121_129_max": b["old_band"]["band121_129_max"],
                    "dall_g0": b["del_table"]["d_all"]["g0"]["mean_pz"],
                    "d183_g0": b["del_table"]["d183"]["g0"]["mean_pz"],
                    "d183_gm12": b["del_table"]["d183"]["gm12"]["mean_pz"],
                })
            rows.append(row)
        return rows

    tables = {seed: trace_table(seed) for seed in TRACE_SEEDS}

    # =====================================================================
    # (B) THE STRADDLE SETTLER x2 (e158 arm-b protocol, new seeds)
    # =====================================================================
    straddle_runs: dict[int, dict] = {}
    straddle_bat: dict[int, dict] = {}
    for si, seed in enumerate(STRADDLE_SEEDS):
        if si > 0 and not SMOKE:
            log(f"[thermal] cooldown {COOLDOWN_S:.0f}s between trainings")
            cooldown(COOLDOWN_S)
        log("=" * 78)
        log(f"STRADDLE seed {seed}: {FT_STEPS}-step LOCKED replay at the home "
            f"band (x-cols 130..136, onset read row 129; e158 arm-b verbatim)")
        run = finetune_trace(f"locked_band_s{seed}", net0, pool_band_x,
                             pool_band_mask, anchor, train_ids, r_eval_xy,
                             f_eval_band, zid, seed, (FT_STEPS,))
        save_ckpt(f"e152r_locked_band_s{seed}", run["sds"][FT_STEPS],
                  {"desc": f"e131_consolidated_e113 + {FT_STEPS}-step LOCKED "
                           f"replay of ZEPHYRA at the home band (x-cols "
                           f"130..136, onset read row 129, e119-L / e158 "
                           f"arm-b convention), name-only mask, seed {seed} "
                           f"(e152R straddle settler)",
                   "steps": run["steps_ran"], "seed": seed,
                   "device": run["device"],
                   "base": f"runs/checkpoints/{ROOT_CK}"})
        straddle_runs[seed] = run
        log(f"STRADDLE seed {seed} AFTER battery")
        straddle_bat[seed] = measure_light(run["sds"][FT_STEPS],
                                           f"band_s{seed}")

    # =====================================================================
    # SMOKE EXIT (nothing adjudicated; shakedown only)
    # =====================================================================
    if SMOKE:
        save_json(rd / "metrics.json", E43.jsonable({
            "experiment": "e152r_reseeds", "date": common.now_iso(),
            "smoke": True, "gates": {"G_SPLICE": G_SPLICE, "G_GEO": G_GEO,
                                     "G_CONS": G_CONS, "G_ROW0": G_ROW0,
                                     "G_A129": G_A129, "G_DALL": G_DALL},
            "trace_runs": {str(s): {"steps_ran": trace_runs[s]["steps_ran"],
                                    "device": trace_runs[s]["device"],
                                    "traj": trace_runs[s]["traj"]}
                           for s in TRACE_SEEDS},
            "straddle_runs": {str(s): {"steps_ran": straddle_runs[s]["steps_ran"],
                                       "device": straddle_runs[s]["device"],
                                       "traj": straddle_runs[s]["traj"],
                                       "gm12": straddle_bat[s]["base"][-12]["mean_pz"]}
                              for s in STRADDLE_SEEDS},
            "ckpt_inventory": CKPT_INVENTORY, "trims": trims,
            "timing": {"total_s": round(time.time() - T0, 1)},
        }))
        log("SMOKE complete — gates PASS, trainings + batteries exercised, "
            "nothing adjudicated")
        log(f"total {time.time() - T0:.1f}s")
        return 0

    # =====================================================================
    # ADJUDICATION (registered clauses; frozen operationalizations)
    # =====================================================================
    def canon_rows(seed: int) -> list[dict]:
        return [r for r in tables[seed] if not r["insert"]]

    def seed_features(rows: list[dict], name: str) -> dict:
        by = {r["steps"]: r for r in rows}
        rets = [by[s]["retention_gm12"] for s in CKPT_CANON]
        diffs = [(CKPT_CANON[i], CKPT_CANON[i + 1],
                  rets[i + 1] - rets[i]) for i in range(len(rets) - 1)]
        big = [d for d in diffs if d[2] <= -CLIFF_DROP]
        bracket = (f"{big[0][0]}->{big[0][1]}" if big else "none")
        # most-negative adjacent diff (ties -> earliest), reported regardless
        worst = min(diffs, key=lambda d: d[2])
        ret8, ret16 = by[8]["retention_gm12"], by[16]["retention_gm12"]
        shelf = min(by[32]["retention_gm12"], by[64]["retention_gm12"])
        fin = by[300]["retention_gm12"]
        a129_mid = [by[s]["A129"] for s in MID_STEPS]
        return {
            "seed": name,
            "retention_gm12": {str(s): by[s]["retention_gm12"] for s in CKPT_CANON},
            "abs_gm12": {str(s): by[s]["base_gm12"] for s in CKPT_CANON},
            "cliff_at_8_16": bool(ret8 >= CLIFF_HI and ret16 <= CLIFF_LO),
            "ret8": ret8, "ret16": ret16,
            "cliff_bracket": bracket,
            "worst_adjacent_drop": {"pair": f"{worst[0]}->{worst[1]}",
                                    "diff": worst[2]},
            "shelf_min_ret_32_64": shelf,
            "shelf_present": bool(shelf >= SHELF_BAR),
            "final_ret_300": fin,
            "final_shut": bool(fin <= FINAL_BAR),
            "A129_mid": {str(s): by[s]["A129"] for s in MID_STEPS},
            "A129_min_mid": min(a129_mid),
            "brake_overshoot": bool(min(a129_mid) < BRAKE_BAR),
            "dwell_all_three": bool(ret8 >= CLIFF_HI and ret16 <= CLIFF_LO
                                    and shelf >= SHELF_BAR and fin <= FINAL_BAR),
            "shape_feature": (bracket, bool(shelf >= SHELF_BAR),
                              bool(fin <= FINAL_BAR)),
        }

    # e152's seed-10902 features from its stored trace (canonical steps)
    e152_rows = [r for r in e152_trace if r["steps"] in (0,) + CKPT_CANON]
    f10902 = seed_features(e152_rows, str(E152_SEED))
    feats = {seed: seed_features(canon_rows(seed), str(seed))
             for seed in TRACE_SEEDS}

    new_seeds_all_three = [feats[s]["dwell_all_three"] for s in TRACE_SEEDS]
    dwell_replicates = bool(all(new_seeds_all_three))
    shapes = [f10902["shape_feature"]] + [feats[s]["shape_feature"]
                                          for s in TRACE_SEEDS]
    dwell_seed_dependent = bool(len(set(shapes)) > 1)

    brake_seeds = {str(E152_SEED): f10902["brake_overshoot"]}
    brake_seeds.update({str(s): feats[s]["brake_overshoot"] for s in TRACE_SEEDS})
    brake_overshoot_fires = bool(sum(brake_seeds.values()) >= 2)

    # straddle label
    gm12_str = {s: straddle_bat[s]["base"][-12]["mean_pz"]
                for s in STRADDLE_SEEDS}
    n_open = sum(1 for v in gm12_str.values() if v >= OPEN_BAR)
    n_shut = sum(1 for v in gm12_str.values() if v <= SHUT_BAR)
    if n_open >= 2:                      # >= 2/3 of new seeds (2 new seeds)
        straddle_label = "OPEN"
        straddle_note = (f"both new seeds >= {OPEN_BAR}: the cell canonizes "
                         "OPEN — e158's committed MID was the GPU-side of a "
                         "device/seed scatter, not the cell's center.")
    elif n_shut == 0:
        straddle_label = "MID"
        straddle_note = (f"straddling persists ({n_open}/2 new seeds >= "
                         f"{OPEN_BAR}, none <= {SHUT_BAR}): the cell lives "
                         "between the bars across seeds/devices — label MID, "
                         "honestly unstable at the 0.5 bar.")
    else:
        straddle_label = "SHUT-APPEARS"
        straddle_note = (f"OUTSIDE the registered fork: {n_shut}/2 new seeds "
                         f"<= {SHUT_BAR} — numbers reported, no bar shopping.")

    cond = {
        "DWELL_REPLICATES": {
            "per_new_seed_all_three": {str(s): feats[s]["dwell_all_three"]
                                       for s in TRACE_SEEDS},
            "bars": {"cliff": f"ret8 >= {CLIFF_HI} AND ret16 <= {CLIFF_LO}",
                     "shelf": f"min(ret32,ret64) >= {SHELF_BAR}",
                     "final": f"ret300 <= {FINAL_BAR}"},
            "per_seed_detail": {str(s): {
                "ret8": feats[s]["ret8"], "ret16": feats[s]["ret16"],
                "shelf": feats[s]["shelf_min_ret_32_64"],
                "final": feats[s]["final_ret_300"]} for s in TRACE_SEEDS},
            "fires": dwell_replicates},
        "DWELL_SEED_DEPENDENT": {
            "shape_features": {str(E152_SEED): list(f10902["shape_feature"]),
                               **{str(s): list(feats[s]["shape_feature"])
                                  for s in TRACE_SEEDS}},
            "feature_names": ["cliff_bracket", "shelf_present", "final_shut"],
            "fires": dwell_seed_dependent},
        "BRAKE_OVERSHOOT_REPLICATES": {
            "per_seed_min_A129_mid": {str(E152_SEED): f10902["A129_min_mid"],
                                      **{str(s): feats[s]["A129_min_mid"]
                                         for s in TRACE_SEEDS}},
            "bar": BRAKE_BAR, "mid_steps": list(MID_STEPS),
            "n_true": int(sum(brake_seeds.values())), "n_seeds": 3,
            "fires": brake_overshoot_fires},
        "STRADDLE_SETTLED": {
            "new_seeds_gm12": {str(s): gm12_str[s] for s in STRADDLE_SEEDS},
            "devices": {str(s): straddle_runs[s]["device"]
                        for s in STRADDLE_SEEDS},
            "bars": {"open": OPEN_BAR, "shut": SHUT_BAR},
            "history": E158_STRADDLE_HISTORY,
            "label": straddle_label, "note": straddle_note},
    }

    if dwell_replicates and not dwell_seed_dependent:
        verdict = ("DWELL-REPLICATES (uniform: the dwell is a phenomenon, "
                   "not a path)")
    elif dwell_replicates:
        verdict = ("DWELL-REPLICATES on both new seeds (SEED-DEPENDENT "
                   "co-fires: 10902 differs in a shape feature)")
    elif dwell_seed_dependent:
        verdict = "DWELL-SEED-DEPENDENT (trajectory-specific texture)"
    else:
        verdict = "TEXTURE (no registered dwell bar fired; numbers reported)"
    verdict += (" | BRAKE-OVERSHOOT-"
                + ("REPLICATES" if brake_overshoot_fires else "FAILS"))
    verdict += f" | STRADDLE: {straddle_label}"

    clause = (
        f"per-seed retention (canonical steps {list(CKPT_CANON)}): "
        + " || ".join(
            f"s{f['seed']}: " + "/".join(f"{f['retention_gm12'][str(s)]:.3f}"
                                         for s in CKPT_CANON)
            for f in [f10902] + [feats[s] for s in TRACE_SEEDS])
        + f" — cliff brackets " + ", ".join(
            f"s{f['seed']}={f['cliff_bracket']}"
            for f in [f10902] + [feats[s] for s in TRACE_SEEDS])
        + f"; shelves " + ", ".join(
            f"s{f['seed']}={f['shelf_min_ret_32_64']:.3f}"
            for f in [f10902] + [feats[s] for s in TRACE_SEEDS])
        + f"; finals " + ", ".join(
            f"s{f['seed']}={f['final_ret_300']:.3f}"
            for f in [f10902] + [feats[s] for s in TRACE_SEEDS])
        + f"; A(129) min-over-mid " + ", ".join(
            f"s{f['seed']}={f['A129_min_mid']:+.3f}"
            for f in [f10902] + [feats[s] for s in TRACE_SEEDS])
        + f"; locked@band g-12 " + ", ".join(
            f"s{s}={v:.4f}({straddle_runs[s]['device']})"
            for s, v in gm12_str.items())
        + f" vs e158 history "
        + "/".join(f"{h['gm12']:.4f}({h['device']})"
                   for h in E158_STRADDLE_HISTORY)
        + ". No bar shopping; every sub-boolean reported.")

    log("=" * 78)
    log(f"E152R VERDICT: {verdict}")
    for f in [f10902] + [feats[s] for s in TRACE_SEEDS]:
        log(f"  s{f['seed']}: cliff@8-16 {f['cliff_at_8_16']} (bracket "
            f"{f['cliff_bracket']}, worst {f['worst_adjacent_drop']['pair']} "
            f"{f['worst_adjacent_drop']['diff']:+.3f}) shelf "
            f"{f['shelf_min_ret_32_64']:.3f} final {f['final_ret_300']:.3f} "
            f"A129min {f['A129_min_mid']:+.4f}")
    log(f"  straddle: {json.dumps({str(s): round(v, 4) for s, v in gm12_str.items()})}"
        f" -> {straddle_label}")
    log(f"  {clause}")
    log("=" * 78)

    # ---------------- insert read (the tightened cliff bracket)
    insert_read = None
    if INSERT_STEPS:
        ins = {r["steps"]: r for r in tables[INSERT_SEED] if r["insert"]}
        by = {r["steps"]: r for r in tables[INSERT_SEED]}
        insert_read = {
            "seed": INSERT_SEED, "steps": list(INSERT_STEPS),
            "retention_gm12": {str(s): ins[s]["retention_gm12"]
                               for s in INSERT_STEPS},
            "abs_gm12": {str(s): ins[s]["base_gm12"] for s in INSERT_STEPS},
            "bracket_8_to_16": [by[s]["retention_gm12"]
                                for s in (8,) + INSERT_STEPS + (16,)],
            "note": "folded sequential-continuation checkpoints (trajectory "
                    "identical by construction); light battery; excluded "
                    "from the canonical bar arithmetic.",
        }
        log(f"insert (seed {INSERT_SEED}) bracket 8/10/12/14/16: "
            + "/".join(f"{v:.3f}" for v in insert_read["bracket_8_to_16"]))

    # ---------------- honesty reflex
    devices_used = {f"trace_s{s}": trace_runs[s]["device"] for s in TRACE_SEEDS}
    devices_used.update({f"locked_band_s{s}": straddle_runs[s]["device"]
                         for s in STRADDLE_SEEDS})
    device_mixed = len(set(devices_used.values())) > 1
    honesty = {
        "device_mixing": (
            f"devices: {devices_used}; PARKED={GPU_PARKED}"
            + (f" ({PARK_REASON})" if PARK_REASON else "")
            + (". MIXED across trainings — e158's two-pass scatter "
               "(0.546 CPU / 0.458 GPU on this exact cell) shows device "
               "sensitivity of order the bar width, so any cross-seed "
               "difference within ~0.1 of a bar is not adjudicable as seed "
               "structure; flagged wherever it applies."
               if device_mixed else
               " — homogeneous this run; NOTE the seed-10902 n=1 being "
               "replicated ran CPU (e152 parked) and e158's committed "
               "straddle cell ran GPU, so cross-experiment comparisons carry "
               "the e119-precedent ~1e-2 float-drift regardless.")),
        "sequential_continuation": "one 300-step trajectory per seed with "
            "RNG-free CPU snapshots (e152's design); the six checkpoints are "
            "ONE path per seed — re-seeds replicate over TRAJECTORIES "
            "(seed-level n=3 with 10902), not over snapshots; the folded "
            "10/12/14 insert adds bracket resolution without a new "
            "trajectory (coincides by construction).",
        "root_single_lineage": "all seeds re-teach the SAME e131 consolidated "
            "root (one lineage, one install): seed variance here is re-teach "
            "trajectory variance only — the n=3 is n=3 paths, not n=3 nets.",
        "battery_trim": "e152's texture cells (onset census, mask/ladder "
            "probes, d129/d_r0 deletions) not re-run — no e152R bar reads "
            "them; the dials that feed bars are instrument-identical to "
            "e152. e161/e169-class dissections on the new-seed checkpoints "
            "would need their own runs.",
        "straddle_baseline": "e158's committed cell is the GPU pass (0.4584); "
            "the settler's comparison set is {0.5464 CPU, 0.4584 GPU, this "
            "run's arms} — the label is seed-AND-device indexed; the honest "
            "note records which side each observation came from.",
        "reproduction_tolerance": "root gates bit-exact (G_CONS bit flag); "
            "GPU-vs-CPU trajectory drift precedent e119/e158 (~1e-2 on "
            "cells, up to ~0.09 on this exact straddle cell).",
    }

    # ---------------- outputs
    metrics = {
        "experiment": "e152r_reseeds",
        "date": common.now_iso(),
        "registration": ("QUEUE row e152R + the dispatch (verbatim bars); "
                         "operationalizations frozen in the module docstring "
                         "before compute"),
        "registered_prediction": REGISTERED_PREDICTION,
        "committed_prediction": None,
        "question": ("do the 8-16-step cliff, the ~50-step dwell shelf, and "
                     "the brake overshoot to -0.466 replicate across re-teach "
                     "seeds (dwell: phenomenon vs trajectory-specific "
                     "texture)? and does e158's straddling locked@band cell "
                     "(0.546 CPU / 0.458 GPU) settle OPEN or MID?"),
        "root": f"runs/checkpoints/{ROOT_CK} (loaded, gated bit-exact vs "
                f"e151 stored root cells)",
        "design": {
            "traces": {"seeds": list(TRACE_SEEDS),
                       "canonical_ckpt_steps": list(CKPT_CANON),
                       "protocol": "e152 verbatim (sequential continuation, "
                                   "offset +54, name-only mask, batch 32/16/"
                                   "16, AdamW 1e-3 (0.9,0.95) wd 0.1 clip 1.0)"},
            "straddle": {"seeds": list(STRADDLE_SEEDS),
                         "protocol": "e158 arm-b verbatim (offset 0, x-cols "
                                     "130..136, read row 129, 60-window "
                                     "locked pool)"},
            "insert": {"seed": INSERT_SEED if INSERT_STEPS else None,
                       "steps": list(INSERT_STEPS),
                       "mode": "folded into the seed's sequential trace"},
            "device_policy": "strict per-training quick check (util<=85 AND "
                             "temp<=80C, double-poll 5s, mem guard); "
                             "park-once to CPU; cooldown(90) between "
                             "trainings; caps 180s GPU / 1500s CPU",
        },
        "e152_n1_loaded": {
            "source": "runs/e152/metrics.json (seed 10902, CPU run)",
            "retention_gm12": e152_sum["retention_gm12"],
            "A129": e152_sum["A129"],
            "features": {k: f10902[k] for k in
                         ("cliff_at_8_16", "cliff_bracket",
                          "shelf_min_ret_32_64", "shelf_present",
                          "final_ret_300", "final_shut", "A129_min_mid",
                          "brake_overshoot", "dwell_all_three",
                          "shape_feature")},
        },
        "reteach_traces": {
            str(seed): {
                "seed": seed, "steps_ran": trace_runs[seed]["steps_ran"],
                "device": trace_runs[seed]["device"],
                "time_cap_s": trace_runs[seed]["time_cap_s"],
                "ckpt_steps": sorted(trace_runs[seed]["sds"]),
                "traj": trace_runs[seed]["traj"],
                "trace": tables[seed],
                "features": {k: feats[seed][k] for k in
                             ("retention_gm12", "abs_gm12", "cliff_at_8_16",
                              "ret8", "ret16", "cliff_bracket",
                              "worst_adjacent_drop", "shelf_min_ret_32_64",
                              "shelf_present", "final_ret_300", "final_shut",
                              "A129_mid", "A129_min_mid", "brake_overshoot",
                              "dwell_all_three", "shape_feature")},
            } for seed in TRACE_SEEDS},
        "straddle_cells": {
            str(seed): {
                "seed": seed, "protocol": "e158 arm-b (locked@band)",
                "steps_ran": straddle_runs[seed]["steps_ran"],
                "device": straddle_runs[seed]["device"],
                "traj": straddle_runs[seed]["traj"],
                "gm12": gm12_str[seed],
                "gm12_label": ("OPEN" if gm12_str[seed] >= OPEN_BAR else
                               ("SHUT" if gm12_str[seed] <= SHUT_BAR else "MID")),
                "g0": straddle_bat[seed]["base"][0]["mean_pz"],
                "gp12": straddle_bat[seed]["base"][12]["mean_pz"],
                "ce_r": straddle_bat[seed]["ce_r"],
            } for seed in STRADDLE_SEEDS},
        "straddle_history_full": E158_STRADDLE_HISTORY + [
            {"source": f"e152r seed {s}", "device": straddle_runs[s]["device"],
             "gm12": gm12_str[s],
             "label": ("OPEN" if gm12_str[s] >= OPEN_BAR else
                       ("SHUT" if gm12_str[s] <= SHUT_BAR else "MID"))}
            for s in STRADDLE_SEEDS],
        "insert_read": insert_read,
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
                     "trace_placement": {"offset": RETEACH_J,
                                         "name_xcols": [SITE_Z_XCOL,
                                                        SITE_Z_XCOL + 6],
                                         "read_rows": [183, 189],
                                         "continuation_len": SITE_CONT,
                                         "zero_variance": True},
                     "straddle_placement": {"offset": BAND_J,
                                            "name_xcols": [BAND_Z_XCOL,
                                                           BAND_Z_XCOL + 6],
                                            "read_rows": [129, 135],
                                            "zero_variance": True},
                     "mask": "7 name-char targets per window (name-only)",
                     "geometries_measured": {"novel": [-12, 12],
                                             "trained_g0": 0}},
        "gates": {"G_SPLICE": G_SPLICE, "G_GEO": G_GEO, "G_CONS": G_CONS,
                  "G_ROW0": G_ROW0, "G_A129": G_A129, "G_DALL": G_DALL,
                  "G_SURG": gates_surg,
                  "gpu": {"parked": GPU_PARKED, "reason": PARK_REASON,
                          "status_at_start": gpu_status(),
                          "devices_used": devices_used}},
        "root_battery": root,
        "batteries": {str(seed): {str(s): batteries[seed][s]
                                  for s in batteries[seed]}
                      for seed in TRACE_SEEDS},
        "straddle_batteries": {str(seed): straddle_bat[seed]
                               for seed in STRADDLE_SEEDS},
        "adjudication": {"conditions": cond, "verdict": verdict,
                         "clause": clause},
        "honesty_reflex": honesty,
        "trims": trims, "deviations": deviations,
        "ckpt_inventory": CKPT_INVENTORY,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": int(net0.num_params()),
                   "eval_device": "cpu",
                   "torch_threads": torch.get_num_threads(), "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "reseeds.png", e152_trace, tables, feats, f10902,
         gm12_str, straddle_runs,
         {str(s): trace_runs[s]["device"] for s in TRACE_SEEDS},
         cond, verdict, clause, insert_read)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'reseeds.png'}, "
        f"{len(CKPT_INVENTORY)} checkpoints in runs/checkpoints/")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot

def plot(path, e152_trace, tables, feats, f10902, gm12_str, straddle_runs,
         trace_devs, cond, verdict, clause, insert_read):
    """THE n=3 OVERLAY: g-12 retention vs steps per seed with the shelf band
    shaded; brake panel; straddle settler; verdict."""
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 10.5))
    seed_colors = {10902: "dimgray", 10903: "crimson", 10904: "tab:blue"}

    # (0,0) THE OVERLAY
    ax = axes[0, 0]
    ax.axhspan(SHELF_BAR, 0.6, color="seagreen", alpha=0.10,
               label="shelf band 0.4-0.6 (the dwell)")
    for yv, col, lbl in ((CLIFF_HI, "tab:blue", "0.70 cliff-hi"),
                         (CLIFF_LO, "tab:orange", "0.50 cliff-lo"),
                         (FINAL_BAR, "gray", "0.27 final bar")):
        ax.axhline(yv, ls="--", lw=0.9, color=col, alpha=0.7, label=lbl)
    xs2 = [7.0 if r["steps"] == 0 else float(r["steps"]) for r in e152_trace]
    ax.plot(xs2, [r["retention_gm12"] for r in e152_trace], "o--", ms=6,
            lw=1.6, color=seed_colors[10902], alpha=0.9,
            label=f"seed 10902 (e152 n=1, CPU)")
    for seed in TRACE_SEEDS:
        rows = tables[seed]
        xs = [7.0 if r["steps"] == 0 else float(r["steps"]) for r in rows]
        ax.plot(xs, [r["retention_gm12"] for r in rows], "o-", ms=6, lw=2.2,
                color=seed_colors[seed],
                label=f"seed {seed} (train {trace_devs[str(seed)]})")
        ins = [r for r in rows if r["insert"]]
        if ins:
            ax.plot([float(r["steps"]) for r in ins],
                    [r["retention_gm12"] for r in ins], "o", ms=9, mfc="none",
                    mec=seed_colors[seed], mew=1.8,
                    label=f"seed {seed} insert 10/12/14")
    ax.set_xscale("log")
    ax.set_xlim(6, 420)
    ck_x = sorted({float(s) for s in CKPT_CANON}
                  | ({float(s) for s in INSERT_STEPS} if INSERT_STEPS else set()))
    ax.set_xticks(ck_x)
    ax.set_xticklabels([str(int(x)) for x in ck_x])
    ax.set_xlabel("locked re-teach steps (root at left; open markers = insert)")
    ax.set_ylabel("g-12 retention (vs root 0.9156)")
    ax.set_ylim(-0.03, 1.1)
    ax.legend(fontsize=7.0, loc="lower left")
    ax.set_title("(A) THE DWELL RE-SEEDS — n=3 overlay (retention vs steps); "
                 f"cliff brackets " + ", ".join(
                     f"s{f['seed']}={f['cliff_bracket']}"
                     for f in [f10902] + [feats[s] for s in TRACE_SEEDS]),
                 fontsize=10)

    # (0,1) the brake
    ax = axes[0, 1]
    ax.axhline(BRAKE_BAR, ls="--", lw=1.0, color="crimson", alpha=0.8,
               label=f"{BRAKE_BAR} brake-overshoot bar")
    xs2m = [float(r["steps"]) for r in e152_trace if r["steps"] in MID_STEPS]
    ax.plot(xs2m, [r["A129"] for r in e152_trace if r["steps"] in MID_STEPS],
            "o--", ms=6, lw=1.4, color=seed_colors[10902], alpha=0.9,
            label="seed 10902 (e152)")
    for seed in TRACE_SEEDS:
        rows = [r for r in tables[seed] if r["steps"] in MID_STEPS]
        ax.plot([float(r["steps"]) for r in rows], [r["A129"] for r in rows],
                "o-", ms=6, lw=2.0, color=seed_colors[seed],
                label=f"seed {seed}: min {feats[seed]['A129_min_mid']:+.3f}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlim(6, 200)
    ax.set_xticks([float(s) for s in MID_STEPS])
    ax.set_xticklabels([str(s) for s in MID_STEPS])
    ax.set_xlabel("locked re-teach steps (mid-conversion)")
    ax.set_ylabel("A(129) census strength")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title(f"(brake) A(129) overshoot — fires at >= 2/3 seeds "
                 f"(currently {cond['BRAKE_OVERSHOOT_REPLICATES']['n_true']}/3)",
                 fontsize=10)

    # (1,0) the straddle settler
    ax = axes[1, 0]
    labels, vals, cols = [], [], []
    for h in E158_STRADDLE_HISTORY:
        labels.append(f"e158 {h['source'].split('(')[0].strip()}\n({h['device']})")
        vals.append(h["gm12"])
        cols.append("seagreen" if h["gm12"] >= OPEN_BAR else
                    ("lightgray" if h["gm12"] <= SHUT_BAR else "gold"))
    for s in STRADDLE_SEEDS:
        labels.append(f"e152r s{s}\n({straddle_runs[s]['device']})")
        vals.append(gm12_str[s])
        cols.append("seagreen" if gm12_str[s] >= OPEN_BAR else
                    ("lightgray" if gm12_str[s] <= SHUT_BAR else "gold"))
    bars = ax.bar(range(len(vals)), vals, color=cols, edgecolor="k", lw=0.6)
    ax.axhline(OPEN_BAR, ls="--", color="seagreen", lw=1.4,
               label=f"OPEN bar {OPEN_BAR}")
    ax.axhline(SHUT_BAR, ls="--", color="gray", lw=1.2,
               label=f"SHUT bar {SHUT_BAR}")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.4f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("locked@band g-12 (absolute)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"(B) THE STRADDLE SETTLER — settled label: "
                 f"{cond['STRADDLE_SETTLED']['label']}", fontsize=10)

    # (1,1) verdict panel
    ax = axes[1, 1]
    ax.axis("off")
    vlines = [
        "REGISTERED (dispatch verbatim; frozen operationalizations):",
        f"  DWELL-REPLICATES: {'FIRES' if cond['DWELL_REPLICATES']['fires'] else 'does not fire'}"
        f" — cliff(8-16) + shelf>=0.4@s32-64 + final<=0.27, both new seeds",
        f"  DWELL-SEED-DEPENDENT: {'FIRES' if cond['DWELL_SEED_DEPENDENT']['fires'] else 'does not fire'}"
        f" — shape features differ across seeds",
        f"  BRAKE-OVERSHOOT-REPLICATES: {'FIRES' if cond['BRAKE_OVERSHOOT_REPLICATES']['fires'] else 'does not fire'}"
        f" — A(129) < {BRAKE_BAR} mid-conversion in >= 2/3 seeds",
        f"  STRADDLE-SETTLED: {cond['STRADDLE_SETTLED']['label']}",
        "",
        "PER-SEED (canonical steps 8/16/32/64/128/300):",
    ] + [
        f"  s{f['seed']}: " + "/".join(f"{f['retention_gm12'][str(s)]:.3f}"
                                       for s in CKPT_CANON)
        + f"  A129min {f['A129_min_mid']:+.3f}  shelf {f['shelf_min_ret_32_64']:.3f}"
        + f"  final {f['final_ret_300']:.3f}  bracket {f['cliff_bracket']}"
        for f in [f10902] + [feats[s] for s in TRACE_SEEDS]] + [
        "",
        "STRADDLE g-12: " + ", ".join(
            f"s{s}={v:.4f}({straddle_runs[s]['device']})"
            for s, v in gm12_str.items()),
        "  e158 history: 0.5464(cpu) / 0.4584(cuda, committed)",
        f"  {cond['STRADDLE_SETTLED']['note']}",
    ]
    if insert_read:
        vlines += ["",
                   f"INSERT (s{insert_read['seed']}) bracket 8/10/12/14/16: "
                   + "/".join(f"{v:.3f}"
                              for v in insert_read["bracket_8_to_16"])]
    vlines += ["", f"VERDICT: {verdict}"]
    vlines += [f"  {wd}" for wd in
               [clause[i:i + 78] for i in range(0, len(clause), 78)]]
    for i, tx in enumerate(vlines):
        ax.text(0.02, 0.97 - i * 0.036, tx, fontsize=7.0, va="top",
                family="monospace")

    fig.suptitle("E152R — THE DWELL RE-SEEDS: n=3 on the cliff / shelf / "
                 "brake + the locked@band straddle settler", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
