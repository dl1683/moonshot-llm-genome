"""E157 — THE LINEAGE REPLICATION (T112's last bound; the paper's debt #1).

WHY (T112 / QUEUE row e157): the lead finding — "no memory state tested
retains expression under continued training without the fact's windows" —
is evidenced at 3 streams x 2 lrs x 3 seeds x every memory type, but ALL
of it lives on ONE LINEAGE (the e131/e065 line: e048 install -> e109/e113
jitter consolidation -> e176n wash). The bound the paper still carries is
"one lineage". This run replicates the conversion + wash arc on the
SECOND family — the e098 fresh 0.84M line (4L/4H/128d/512-ctx, nets on
disk: runs/checkpoints/e098_base_s4305.pt + e098_install_s4305.pt) — and
runs its 2x2 rider (e157r).

DESIGN (3 stages, 5 short trainings + evals):
  (A) CONSOLIDATE — the e113 jitter-replay recipe ported to the e098
      install's conventions: jittered pool {-8,-4,0,+4,+8} (300 windows),
      anchor bank = the first 16 install positions' original windows,
      batch 32 = 16 pool draws + 16 anchors (8 paired + 8 random), e043
      token-level union CE (name-masked on pool windows), AdamW (0.9,0.95)
      wd 0.1 constant lr 1e-3 clip 1.0, 300 steps, seed 10901 (e113 arm
      (a) VERBATIM; the RNG draw shapes/moduli are identical: pool 300 /
      anchor 16 / len(train_ids)-257). GATE (registered): the consolidated
      state shows geometry-general access (novel-geometry read high) and
      the install's dials.
  (B) WASH — e176N arm A VERBATIM on the new consolidated net (seed
      10902): e170's NEUTRAL anchor bank (16 plain-corpus windows, RNG
      seed 170, rejection on FLORIZEL/ELIZABETH/ZEPH/MIRABEL in
      [s, s+257); G_ANCHOR 0/16 host content, 0/16 junctions) + 16 random
      corpus windows per step, full-token CE, AdamW (0.9,0.95) wd 0.1
      constant lr 1e-3 clip 1.0, 300 steps, checkpoints {1,2,4,50,100,200,
      300} with snapshots + full dials — the lineage cell of the lead
      finding, family 2.
  (C) 2x2 RIDER (e157r) — e158's conventions on the new consolidated net:
      jitter@183 (offsets {46,50,54,58,62} = e113 set as deltas on +54,
      300-window pool), locked@band (offset-0, 60-window pool), and the
      conversion cell locked@183 (e151's offset-54 locked pool) — budget
      allowed it. All seed 10902, 300 steps, same recipe as (A).

REGISTERED PREDICTION (dispatch verbatim; no bar shopping):
  - LEAD-FINDING-REPLICATES fires if: the second family's consolidated
    fact dissolves under the neutral wash — g-12 <= 0.27 by +50 (e176n's
    NEUTRAL-DISSOLVES bar) — the lineage bound discharges; the finding is
    n=2 families.
  - LINEAGE-BOUND fires if: the second family's fact survives the wash —
    g-12 >= 0.5 at +300 — the finding is lineage-specific; the bound
    stands and the paper says so.
  - RIDER (report, not gating): the 2x2 cells' directions on family 2;
    QUEUE e157r's bars co-adjudicated: GATE-REPLICATES = jitter@183 open
    (>= 0.5) AND locked@band open (>= 0.5) AND locked@183 shut (<= 0.27);
    LINEAGE-BOUND = any door flips vs family 1's committed cells
    (jitter@183 OPEN 0.7885 / locked@band MID 0.4584 / locked@183 SHUT
    0.1021).
  - Texture => TEXTURE with numbers.

OPERATIONALIZATIONS (frozen before compute):
  * g-12 / g0 / g+12 / held30 = ABSOLUTE install-60 / held-30 battery
    mean p(Z) at ctx offsets -12 / 0 / +12 (e176's ruler, same corpus
    rebuild). Dissolve bar 0.27 at the +50 checkpoint; survive bar 0.5 at
    the +300 checkpoint (dispatch verbatim); e176n's stricter
    every-checkpoint survival form CO-REPORTED.
  * G_CONS (stage A gate): novel-geometry read high = pZ >= 0.5 at BOTH
    jitter endpoints g-8 / g+8 (novel at install time) AND >= 4/5 jitter
    geometries >= 0.5; the geometry door = g-12 >= 0.5 (e158's OPEN bar);
    the install's dials = install60 p(Z) >= 0.20 AND R1i NLL <= 4.5
    (e098's GATE-0 bars verbatim).
  * Doors (rider): g-12 after the arm; OPEN >= 0.5, SHUT <= 0.27.

RECIPE PORT DELTAS (e113 -> e157 stage A; everything else VERBATIM):
  1. NET FAMILY: the e098 s4305 install net (4L/4H/128d, block 512, wpe
     512x128, 873,472 params) instead of e048_repro (6L/6H/192d, block
     256, ~2.7M). This is the point of the experiment.
  2. BLOCK GEOMETRY: all training/anchor/random windows remain 256-token
     windows (BLOCK=256) — the e098 line's OWN install convention (e043
     exposure verbatim used 256-token windows on the 512-ctx net); the
     family-2 net's wpe rows 256..511 receive no gradient in ANY protocol
     here (install, consolidation, wash, riders) — the wash tests the same
     rows the fact lives in, exactly as on family 1.
  3. DEVICE: GPU-gated per training (park-once + mid-run guard) instead
     of e113's forced CPU — float arithmetic differs by device; the e113
     family-1 numbers were themselves a CPU rebuild of a cuda original
     (gated 0.05/cell), and every bar here is an absolute value, not a
     bit-parity claim. Draw sequences are device-independent (CPU
     torch.Generator).

COMPUTE ENVELOPE (dispatch): GPU allowed — strict pre-training quick
check per training (gpu_ok(): util <= 85 AND temp <= 80C AND mem <= 85%,
double-poll 5 s), PARK-ONCE to CPU on failure; mid-run guard every 25
steps (mem > 85% or temp > 80C -> migrate net+optimizer to CPU and finish
there — stricter than the dispatch's 83C note, e184's precedent verbatim);
cooldown(90 s) before each training; caps 180 s (GPU) / 1500 s (CPU) per
training; ALL readouts CPU-side; torch threads 8. Checkpoints ~10 x 11 MB
(disk-fine).

INSTRUMENT PROVENANCE: battery_cell / battery_pz / ce_fixed_cpu /
val_windows / deleted_wpe / read_fact_at / row_census are lab/
e176n_neutral_wash.py VERBATIM (the e176/e161/e152/e151/e143/e131/e119/
e113/e068/e065/e043 lineage); the neutral bank + junction accounting are
e170's via e176n's copy; offset_pool is e158's (e143's machinery);
finetune_replay is e158's finetune_arm with e184's device/mid-run guard;
finetune_freeze is e184's device-parameterized copy of e176n arm A.
Copied, not imported, to own the device policy.

Outputs: runs/e157/{metrics.json, lineage_replication.png}; checkpoints
runs/checkpoints/e157_f2_*.pt. No NOTES/THINKING/QUEUE/STATE edits;
single commit, no push.

Run:  cd lab && python e157_lineage_replication.py   (E157_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # GPU allowed, gated below

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                           # noqa: E402

torch.set_num_threads(8)                              # e143/e151/e158 convention

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT, cooldown, gpu_ok,     # noqa: E402
                    gpu_status, run_dir, save_json)
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable, eval_seq)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E157_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256                       # training-window block (the e098 line's own
                                  # install convention: 256-token windows on a
                                  # 512-context net)
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
BASE_CK = "e098_base_s4305.pt"
INST_CK = "e098_install_s4305.pt"

# ---- the family-2 net (e098 s4305 line: fresh 0.84M, 4L/4H/128d/512-ctx) ----
F2_CFG = Cfg(vocab=65, n_layer=4, n_head=4, n_embd=128, block_size=512)
F2_PARAMS = 873_472

# ---- placement constants (e152/e158 verbatim) ---------------------------------
RETEACH_J = 54                    # offset: name x-cols 184..190, read rows 183..189
SITE_ADDR_ROW = 183               # onset read row (= e120's SPLICE_ADDR_ROW)
SITE_Z_XCOL = PRE + RETEACH_J     # 184
SITE_ROWS = tuple(range(183, 190))
JIT_DELTAS = (-8, -4, 0, 4, 8)    # e109/e113 registered jitter set
ARM_C_J = tuple(RETEACH_J + d for d in JIT_DELTAS)    # {46,50,54,58,62}
P8_ADDR_ROW, P8_XCOL = SITE_ADDR_ROW + 8, SITE_Z_XCOL + 8
BAND_ADDR_ROW, BAND_Z_XCOL = PRE - 1, PRE            # 129, 130 (home position)
D_ALL = (121, 125, 129, 133, 137)                    # e113's fixed set
GEOS = (-12, 0, 12)               # wash ruler: novel x2 + the trained g0
JITTERS = (-8, -4, 0, 4, 8)       # consolidation geometries (e113's set)

ROWS_OLD = (0, 1, 2, 3, 4, 5, 6, 60, 100, 118, 119, 120) + tuple(range(121, 130))
ROWS_183 = (0, 1, 2) + (181, 182) + SITE_ROWS + (60, 100, 150, 160, 170)
ROWS_JIT8 = (0,) + tuple(range(175, 198)) + (60, 100, 150)
ROWS_BAND = (0,) + tuple(range(118, 138)) + (60,)
if SMOKE:
    ROWS_OLD = (0, 1, 60, 121, 125, 129)
    ROWS_183 = (0, 1, 182) + (183, 185, 189) + (60, 100)
    ROWS_JIT8 = (0, 175, 183, 191, 197) + (60,)
    ROWS_BAND = (0, 121, 129, 135) + (60,)

# ---- trainings ----------------------------------------------------------------
FT_LR = 1e-3
FT_STEPS = 300 if not SMOKE else 8
EVAL_EVERY = 25 if not SMOKE else 2
NAME_BS, ANCH_BS = 16, 16
CONS_SEED = 10901                 # e109/e113 arm (a) seed VERBATIM
ARM_SEED = 10902                  # e158 arms / e176n wash seed VERBATIM
CK_WASH: tuple[int, ...] = (1, 2, 4, 50, 100, 200, 300) if not SMOKE else (1, 2, 4)
COOLDOWN_S = 90.0                 # dispatch envelope 60-120 s
GPU_CAP_S, CPU_CAP_S = 180.0, 1500.0
MIDRUN_POLL_EVERY = 25

# ---- e170's neutral anchor bank (e176N arm A's stream, VERBATIM) --------------
E170_ANCHOR_SEED = 170
ANCHOR_FORBIDDEN = ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")

# ---- gates / references (full precision, = stored metrics) --------------------
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05
G_INST_STRICT_TOL = 0.008         # e091's 0.005 widened one notch for the
                                  # cpu-vs-cuda eval drift (flagged either way)

# e098 s4305 stored cells (runs/e098/metrics.json per_seed['4305'])
E098_REF = {
    "install60_pz": 0.5678548689931631,
    "held30_pz": 0.5380222407480081,
    "r1i_nll": 0.1528928577899933,
    "r1i_acc": 0.9547619223594666,
    "base_val_ce": 1.6445075869560242,
    "census_base_pz": 0.5678548759470383,
    "params": 873472,
}

# e113's stored post-none table (runs/e113/metrics.json battery_table; the
# family-1 consolidated net's five-geometry read) — family 1's stage-A result.
E113_NONE = {-8: 0.9933038353919983, -4: 0.9476708769798279,
             0: 0.7850371599197388, 4: 0.9860305190086365,
             8: 0.9811912775039673}

# e176N arm A's stored trajectory (runs/e176n/metrics.json trace_armA) —
# family 1's lineage cell, seed 10902 (the exact cell being replicated).
E176N_A = {
    "steps": [0, 1, 2, 4, 50, 100, 200, 300],
    "gm12": [0.9155886173248291, 0.6780440807342529, 0.027077054604887962,
             0.010940761305391788, 0.022122304886579514, 0.014922752045094967,
             0.00204725144430995, 0.0038193254731595516],
    "g0": [0.7850371599197388, 0.4619811177253723, 0.1147073358297348,
           0.06042749062180519, 0.16835635900497437, 0.05249874293804169,
           0.011475668288767338, 0.016238771378993988],
    "ce_r": [1.663516640663147, 2.2113420963287354, 2.032074451446533,
             1.8213403224945068, 1.708834171295166, 1.6700899600982666,
             1.6468615531921387, 1.642844796180725],
}

# e158's committed 2x2 (family 1) + e151's locked@183
F1_2X2 = {
    "jitter_band": 0.9155886769294739,   # e131 root (in-run BEFORE)
    "locked_183": 0.1020549014210701,    # e151 stored
    "jitter_183": 0.7884896397590637,    # e158 committed pass (GPU)
    "locked_band": 0.45835810899734497,  # e158 committed pass (GPU)
}

# ---- registered bar constants (frozen) ----------------------------------------
SHUT_BAR = 0.27                   # dissolve by +50 (e176n's NEUTRAL-DISSOLVES)
SURVIVE_BAR = 0.50                # survive at +300 (dispatch verbatim)
OPEN_BAR = 0.50                   # e158 door bars
CONS_PZ_BAR = 0.50                # stage-A novel-geometry read bar
GATE_PZ_FLOOR = 0.20              # e098 GATE-0 bars verbatim
GATE_R1I_MAX = 4.5

REGISTERED_PREDICTION = {
    "lead_finding_replicates": "LEAD-FINDING-REPLICATES fires if: the second "
        "family's consolidated fact dissolves under the neutral wash (g-12 "
        f"<= {SHUT_BAR} by +50) — the lineage bound discharges; the finding "
        "is n=2 families.",
    "lineage_bound": "LINEAGE-BOUND fires if: the second family's fact "
        f"survives the wash (g-12 >= {SURVIVE_BAR} at +300) — the finding is "
        "lineage-specific; the bound stands and the paper says so.",
    "rider_report": "RIDER (report, not gating): the 2x2 cells' directions on "
        "family 2; QUEUE e157r bars co-adjudicated: GATE-REPLICATES = "
        "jitter@183 open (>= 0.5) AND locked@band open (>= 0.5) AND "
        "locked@183 shut (<= 0.27); LINEAGE-BOUND = any door flips vs family "
        "1's committed cells (OPEN 0.7885 / MID 0.4584 / SHUT 0.1021).",
    "texture": "Texture => TEXTURE with numbers.",
    "operationalizations": "g-12/g0/g+12/held30 = absolute install-60/held-30 "
        "battery mean p(Z) at ctx offsets -12/0/+12 (e176's ruler); "
        "LEAD-FINDING-REPLICATES = wash g-12 at the +50 checkpoint <= 0.27 "
        "(earliest checkpoint under the bar co-reported); LINEAGE-BOUND = "
        "wash g-12 at +300 >= 0.5 (e176n's every-checkpoint form "
        "co-reported); order LEAD-FINDING-REPLICATES -> LINEAGE-BOUND -> "
        "TEXTURE; G_CONS (stage A) = pZ >= 0.5 at BOTH g-8/g+8 AND >= 4/5 "
        "jitter geometries >= 0.5 AND g-12 >= 0.5 AND install60 pZ >= 0.20 "
        "AND R1i NLL <= 4.5 — a G_CONS failure bounds every downstream "
        "verdict (reported, not silently dropped); rider doors: g-12 after "
        "the arm, OPEN >= 0.5 / SHUT <= 0.27.",
    "registration": "QUEUE rows e157 + e157r / dispatch registration, frozen "
        "verbatim in this docstring before compute. Adjudicate against "
        "exactly this; no bar shopping.",
}

trims: list[str] = []
deviations: list[str] = [
    "RECIPE PORT (stage A): e113's jitter consolidation on the e098 s4305 "
    "install (4L/4H/128d/512-ctx, 873,472 params) instead of e048_repro "
    "(6L/6H/192d/256, ~2.7M) — the experiment's point; recipe/seed/batch "
    "composition/RNG moduli otherwise VERBATIM (pool 300 / anchor 16 / "
    "len(train_ids)-257 at seed 10901).",
    "BLOCK GEOMETRY: all training/anchor/random windows are 256-token "
    "windows — the e098 line's own install convention (e043 exposure "
    "verbatim on the 512-ctx net); the family-2 wpe rows 256..511 receive "
    "no gradient in ANY protocol here (the wash tests the rows the fact "
    "lives in, as on family 1).",
    "DEVICE: GPU-gated per training (park-once + mid-run guard at mem > 85% "
    "or temp > 80C — stricter than the dispatch's 83C note, e184 precedent "
    "verbatim); family-1's references mix devices themselves (e109 arm a "
    "cuda, e113 rebuild cpu, e176n arm A cpu, e158 committed pass gpu) — "
    "every bar here is absolute, none claims bit parity; draw sequences "
    "are device-independent (CPU torch.Generator).",
    "The measure dial is e176n's measure() (already minus the 183-span "
    "census per e178's recorded deviation) MINUS e158's mask/ladder probes "
    "on the rider arms (e150-informed extras; the rider's registered "
    "readout is the g-12 directions) — recorded trim, not a bar.",
    "Stage C includes the conversion cell (locked@183, e151's protocol) — "
    "the dispatch's 'if budget allows' branch: the four trainings each ran "
    "far under the 180 s GPU cap, so the full 2x2 exists on family 2.",
    "Single seed per cell (10901 consolidation / 10902 wash + riders), one "
    "lineage per family — n=1 per family cell; the replication claim is "
    "family-level (n=2 families), not seed-level.",
    "G_GATE-0's strict instrument tolerance is 0.008 (e091's 0.005 widened "
    "one notch for cpu-vs-cuda eval drift on the install battery; the "
    "0.05 fallback pass and the 5e-6 bit flag are both reported).",
    "Smoke mode trims: 8-step trainings, checkpoints {1,2,4}, reduced "
    "census rows, nothing adjudicated.",
]

# ------------------------------------------------------------------ device pick
# PROVENANCE: lab/e184_seed_replicates.py's pick_dev/migrate (e152R park-once
# adapted to the lab's gpu_ok()) VERBATIM.

GPU_PARKED = False
PARK_REASON = None
device_events: list[dict] = []


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
    GPU_PARKED, PARK_REASON = True, f"quick check failed: {s1}"
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
# PROVENANCE: lab/e176n_neutral_wash.py / e158 VERBATIM (see docstrings there).
# Copied rather than imported to own the device policy.

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


# ------------------------------------------------------------------ fine-tune

def finetune_replay(tag: str, net0: TinyGPT, pool_x: torch.Tensor,
                    pool_mask: torch.Tensor, anchor: torch.Tensor,
                    train_ids: torch.Tensor, r_eval_xy, f_eval_ids, zid: int,
                    seed: int):
    """e158's finetune_arm (e109 arm-b / e119-L / e113 recipe) with e184's
    device policy: batch 32 = 16 install windows from the pool + 16 anchors
    (8 paired + 8 random); e043 token-level union CE on the name-char
    targets; constant lr 1e-3 AdamW (0.9,0.95) wd 0.1 clip 1.0; in-loop CPU
    evals every 25 (no RNG consumed). Mid-run GPU contention guard every 25
    steps: mem > 85% or temp > 80C -> migrate to CPU and finish there."""
    dev = pick_dev(tag)
    cap = GPU_CAP_S if dev.type == "cuda" else CPU_CAP_S
    net = copy.deepcopy(net0).to(dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    n_pool = pool_x.shape[0]
    n_anc = anchor.shape[0]
    traj, t_start = [], time.time()
    step = 0
    evl = copy.deepcopy(net0).to(CPU)      # CPU eval twin
    for step in range(1, FT_STEPS + 1):
        ix = torch.randint(n_pool, (NAME_BS,), generator=gen)
        aj = torch.randint(n_anc, (ANCH_BS // 2,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (ANCH_BS // 2,),
                           generator=gen)
        nw = pool_x[ix].to(dev)
        anc = torch.cat([anchor[aj],
                         torch.stack([train_ids[s: s + BLOCK] for s in rj])],
                        0).to(dev)
        x = torch.cat([nw[:, :-1], anc[:, :-1]], 0)
        y = torch.cat([nw[:, 1:], anc[:, 1:]], 0)
        m = torch.zeros(NAME_BS + ANCH_BS, x.shape[1], dtype=torch.bool,
                        device=dev)
        m[:NAME_BS] = pool_mask[ix].to(dev)
        logits, _ = net(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                              y.reshape(-1), reduction="none"
                              ).view(x.shape[0], x.shape[1])
        nm = nll[:NAME_BS][m[:NAME_BS]]
        cm = nll[NAME_BS:]
        loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0 or step == FT_STEPS or \
                (time.time() - t_start) > cap:
            evl.load_state_dict({k: v.detach().cpu().clone()
                                 for k, v in net.state_dict().items()})
            evl.eval()
            bz = battery_cell(evl, f_eval_ids, zid)
            ce_r = ce_fixed_cpu(evl, *r_eval_xy)
            traj.append({"step": step, "p_z_mean": bz["mean_pz"],
                         "frac_argmax_z": bz["frac_argmax_z"], "ce_r": ce_r,
                         "elapsed_s": round(time.time() - t_start, 1)})
            log(f"  [{tag}] s{step:4d} p_z {bz['mean_pz']:.4f} argmaxZ "
                f"{bz['frac_argmax_z']:.2f} CE_R {ce_r:.4f} "
                f"({traj[-1]['elapsed_s']:.0f}s)")
        if (time.time() - t_start) > cap:
            log(f"  [{tag}] time cap {cap:.0f}s at s{step}")
            trims.append(f"{tag}: time cap at step {step}")
            break
        if dev.type == "cuda" and step % MIDRUN_POLL_EVERY == 0:
            s = gpu_status()
            if s["mem_total"] > 0 and (s["mem_used"] > 0.85 * s["mem_total"]
                                       or s["temp"] > 80):
                device_events.append(
                    {"tag": tag, "step": step, "event": "MID-RUN MIGRATION",
                     "status": s})
                log(f"  [{tag}] MID-RUN GPU contention at s{step} ({s}) -> "
                    f"migrating to CPU")
                migrate_to_cpu(net, opt)
                dev = CPU
                cap = CPU_CAP_S
    net.eval()
    sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"sd": sd_cpu, "traj": traj, "steps_ran": step, "seed": seed,
            "device": str(dev), "time_cap_s": cap}


def finetune_freeze(tag: str, net0: TinyGPT, anchor: torch.Tensor,
                    train_ids: torch.Tensor, itos, r_eval_xy, gm12_ids,
                    g0_ids, zid: int, seed: int,
                    ckpt_steps: tuple[int, ...]):
    """THE NEUTRAL PLAIN-CORPUS FREEZE (e176n arm A VERBATIM arithmetic,
    e184's device policy). Per step: aj = randint(16) anchor draws, rj =
    randint(16) random corpus offsets; batch 32 full-token CE; AdamW
    (0.9,0.95) wd 0.1 constant lr 1e-3, clip 1.0. The RNG draw sequence is
    IDENTICAL to e176n arm A's at seed 10902 (same shapes/moduli) — only
    the net family (and possibly device) differ. Snapshots + light CPU
    evals (g-12, g0, CE_R — no RNG consumed) at the checkpoint steps."""
    dev = pick_dev(tag)
    cap = GPU_CAP_S if dev.type == "cuda" else CPU_CAP_S
    ckpt_set = set(ckpt_steps)
    n_steps = ckpt_steps[-1]
    net = copy.deepcopy(net0).to(dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    n_anc = anchor.shape[0]
    traj, sds, t_start = [], {}, time.time()
    step = 0
    zeph_checks = 0
    evl = copy.deepcopy(net0).to(CPU)      # CPU eval twin
    for step in range(1, n_steps + 1):
        aj = torch.randint(n_anc, (16,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (16,), generator=gen)
        anc = anchor[aj]
        rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj])
        # name-free VERIFY (no-op by corpus construction; hard-fail if not)
        for w in rnd:
            txt = "".join(itos[int(c)] for c in w[:64]) + \
                  "".join(itos[int(c)] for c in w[192:])
            if "ZEPH" in txt:
                zeph_checks += 1
        x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0).to(dev)
        y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0).to(dev)
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
        if (time.time() - t_start) > cap:
            log(f"  [{tag}] time cap {cap:.0f}s at s{step}")
            trims.append(f"{tag}: time cap at step {step}")
            break
        if dev.type == "cuda" and step % MIDRUN_POLL_EVERY == 0:
            s = gpu_status()
            if s["mem_total"] > 0 and (s["mem_used"] > 0.85 * s["mem_total"]
                                       or s["temp"] > 80):
                device_events.append(
                    {"tag": tag, "step": step, "event": "MID-RUN MIGRATION",
                     "status": s})
                log(f"  [{tag}] MID-RUN GPU contention at s{step} ({s}) -> "
                    f"migrating to CPU")
                migrate_to_cpu(net, opt)
                dev = CPU
                cap = CPU_CAP_S
    net.eval()
    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"sds": sds, "traj": traj, "steps_ran": step, "seed": seed,
            "lr": FT_LR, "zeph_violations": zeph_checks, "device": str(dev),
            "time_cap_s": cap}


CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "e157", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name} "
        f"({', '.join(f'{k}={v}' for k, v in meta.items())})")


def verify_ref(embedded: dict, path: Path, getter, name: str) -> dict:
    """Verify an embedded reference copy against its stored metrics file when
    present (e176n's verify_ref convention — no silent divergence)."""
    src = {"source": f"embedded verbatim copy ({name})",
           "file_present": path.exists(), "verified_vs_embedded": None,
           "max_abs_diff": None}
    if path.exists():
        try:
            got = getter(json.loads(path.read_text(encoding="utf-8")))
            diffs = []
            for k, v in embedded.items():
                if isinstance(v, list):
                    diffs += [abs(a - b) for a, b in zip(got[k], v)]
                else:
                    diffs.append(abs(got[k] - v))
            src["max_abs_diff"] = max(diffs) if diffs else None
            src["verified_vs_embedded"] = bool(max(diffs) < 1e-9)
            if src["verified_vs_embedded"]:
                src["source"] = (f"{name} (embedded copy verified, max|diff| "
                                 f"{max(diffs):.1e})")
        except Exception as e:  # noqa: BLE001 — report, don't crash on refs
            src["verify_error"] = str(e)
    return src


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e157_smoke" if SMOKE else "e157")
    common.DEVICE = "cpu"     # ALL readouts CPU-side (e098's eval convention;
                              # the trainers below manage their own devices)
    log(f"E157 THE LINEAGE REPLICATION (family 2 = e098 s4305 0.84M line; "
        f"smoke={SMOKE}) -> {rd}")
    log(f"compute: strict per-training GPU gate (park-once) + mid-run guard "
        f"every {MIDRUN_POLL_EVERY} steps; gpu at start: {gpu_status()}; "
        f"threads {torch.get_num_threads()}")

    # ---------------- protocol rebuild (e143/e151/e158/e176n verbatim)
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
    L = len(NAME)

    # ---------------- pools (e158's offset_pool machinery, VERBATIM)
    def offset_pool(j: int):
        """e143's offset-pool construction at total offset j: pre = 130+j
        true tokens, ZEPHYRA at x-cols (130+j)..(136+j), continuation
        119-j; mask targets = read rows (129+j)..(135+j)."""
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
    pool_cons_x = torch.cat([jit_pools[j][0] for j in JITTERS])   # (300,256)
    pool_cons_mask = torch.cat([jit_pools[j][1] for j in JITTERS])
    pool_band_x, pool_band_mask = jit_pools[0]                    # locked j=0
    pool_183_x, pool_183_mask = offset_pool(RETEACH_J)            # locked j=54
    jit183_pools = {j: offset_pool(j) for j in ARM_C_J}
    pool_jit183_x = torch.cat([jit183_pools[j][0] for j in ARM_C_J])
    pool_jit183_mask = torch.cat([jit183_pools[j][1] for j in ARM_C_J])
    pool_p8_x = jit183_pools[RETEACH_J + 8][0]                    # d=+8 read

    G_GEO = {
        "cons_jit_offsets": list(JITTERS),
        "cons_all_in_place": bool(all(
            all(torch.equal(w[PRE + j: PRE + j + L], name_ids)
                for w in jit_pools[j][0]) for j in JITTERS)),
        "cons_mask_targets": int(pool_cons_mask[0].sum()),
        "band_locked": bool(
            all(torch.equal(w[BAND_Z_XCOL: BAND_Z_XCOL + L], name_ids)
                for w in pool_band_x)
            and int(pool_band_mask[0].sum()) == L),
        "p183_locked": bool(
            all(torch.equal(w[SITE_Z_XCOL: SITE_Z_XCOL + L], name_ids)
                for w in pool_183_x)
            and int(pool_183_mask[0].sum()) == L),
        "jit183_offsets": list(ARM_C_J),
        "jit183_all_in_place": bool(all(
            all(torch.equal(w[PRE + j: PRE + j + L], name_ids)
                for w in jit183_pools[j][0]) for j in ARM_C_J)),
        "jit183_mask_targets": int(pool_jit183_mask[0].sum()),
        "pool_sizes": {"cons": int(pool_cons_x.shape[0]),
                       "band": int(pool_band_x.shape[0]),
                       "p183": int(pool_183_x.shape[0]),
                       "jit183": int(pool_jit183_x.shape[0])},
    }
    G_GEO["pass"] = bool(G_GEO["cons_all_in_place"] and G_GEO["band_locked"]
                         and G_GEO["p183_locked"]
                         and G_GEO["jit183_all_in_place"]
                         and G_GEO["cons_mask_targets"] == L
                         and G_GEO["jit183_mask_targets"] == L)
    assert G_GEO["pass"], f"pool geometry gate FAILED: {G_GEO}"
    log(f"pools: cons jitter {tuple(pool_cons_x.shape)} (offsets "
        f"{list(JITTERS)}) | band {tuple(pool_band_x.shape)} (locked) | "
        f"p183 {tuple(pool_183_x.shape)} (locked j=54) | jit183 "
        f"{tuple(pool_jit183_x.shape)} (offsets {list(ARM_C_J)})")

    # anchor bank (e065/e109/e113/e158 verbatim): first 16 install-position
    # ORIGINAL host windows (incumbent continuations, no ZEPHYRA)
    anchor = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                          for p, _ in install_occ[:16]])

    # ---------------- batteries (e119/e158/e176n construction verbatim)
    bat_ids = {}
    for j in set(GEOS) | set(JITTERS):
        for tag, occ in (("install60", install_occ), ("held30", held_occ)):
            cs = [train_text[p - PRE - j: p] for p, _ in occ]
            bat_ids[(j, tag)] = torch.stack([corpus.encode(c) for c in cs])
    ids130 = bat_ids[(0, "install60")]             # the g0 battery
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

    # ---------------- THE NEUTRAL ANCHOR BANK (e170 via e176n VERBATIM)
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
                         "construction VERBATIM (e176n arm A's only stream)"),
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
    assert G_ANCHOR["pass"], f"anchor gate FAILED: {G_ANCHOR}"
    log(f"G_ANCHOR: neutral bank 16x{BLOCK} plain-corpus windows (seed "
        f"{E170_ANCHOR_SEED}, {rejections} rejections/{tries} tries) — host "
        f"content 0/16, junctions 0/16; random-channel background "
        f"~{100 * bg_rate:.1f}%/window: PASS")

    # ---------------- family-2 nets + GATE-0 vs e098 stored cells
    net_base = load_f2(CKPT_DIR / BASE_CK)
    net_inst = load_f2(CKPT_DIR / INST_CK)
    n_params = net_inst.num_params()
    assert n_params == F2_PARAMS, f"params {n_params} != {F2_PARAMS}"
    sd_inst = {k: v.clone() for k, v in net_inst.state_dict().items()}
    base_val_ce = common.estimate_loss(net_base, corpus, "val", n_batches=12)
    del net_base, net_inst

    bz0 = battery_cell(evl_load(sd_inst), ids130, zid)
    bzh0 = battery_cell(evl_load(sd_inst), bat_ids[(0, "held30")], zid)
    E43.DEVICE = "cpu"                              # eval_seq device
    bat_i_seq = pool_band_x[:, :PRE + L]            # e098's bat_i_seq
    r1i0 = E43.eval_seq(evl_load(sd_inst), bat_i_seq, L, PRE - 1)
    val_ce0 = common.estimate_loss(evl_load(sd_inst), corpus, "val",
                                   n_batches=12)
    G_GATE0 = {
        "cells": {"install60_pz": bz0["mean_pz"], "held30_pz": bzh0["mean_pz"],
                  "r1i_nll": r1i0["nll"], "r1i_acc": r1i0["acc"],
                  "base_val_ce": base_val_ce,
                  "install_val_ce": val_ce0},
        "refs": E098_REF,
        "diffs": {"install60_pz": bz0["mean_pz"] - E098_REF["install60_pz"],
                  "r1i_nll": r1i0["nll"] - E098_REF["r1i_nll"]},
        "strict_tol": G_INST_STRICT_TOL, "bit_tol": G_BIT_TOL,
        "tol": G_FALLBACK_TOL,
        "strict": bool(abs(bz0["mean_pz"] - E098_REF["install60_pz"])
                       < G_INST_STRICT_TOL),
        "bit": bool(abs(bz0["mean_pz"] - E098_REF["install60_pz"]) < G_BIT_TOL),
        "pass": bool(abs(bz0["mean_pz"] - E098_REF["install60_pz"])
                     < G_FALLBACK_TOL
                     and abs(r1i0["nll"] - E098_REF["r1i_nll"]) < 0.05
                     and abs(base_val_ce - E098_REF["base_val_ce"]) < 0.05),
    }
    log(f"G_GATE-0 install battery p(Z) {bz0['mean_pz']:.6f} (ref "
        f"{E098_REF['install60_pz']:.6f}, diff "
        f"{bz0['mean_pz'] - E098_REF['install60_pz']:+.6f}) | R1i "
        f"{r1i0['nll']:.4f} (ref {E098_REF['r1i_nll']:.4f}) | held30 "
        f"{bzh0['mean_pz']:.4f} | base val CE {base_val_ce:.4f} (ref "
        f"{E098_REF['base_val_ce']:.4f}) | install val CE {val_ce0:.4f}: "
        f"{'PASS' if G_GATE0['pass'] else 'FAIL'}"
        + (" (strict)" if G_GATE0["strict"] else ""))
    if not G_GATE0["pass"] and not SMOKE:
        raise RuntimeError("family-2 install failed GATE-0 vs e098 stored")

    # =====================================================================
    # the measurement dial (e176n's measure() conventions, compact census)
    # =====================================================================
    gates_surg: dict = {}

    def measure(sd: dict, tag: str, rider: bool = False) -> dict:
        net = evl_load(sd)
        out: dict = {"tag": tag}
        out["base"] = {j: battery_cell(net, bat_ids[(j, "install60")], zid)
                       for j in GEOS}
        out["base_held"] = {j: battery_cell(net, bat_ids[(j, "held30")], zid)
                            for j in GEOS}
        out["ce_r"] = ce_fixed_cpu(net, *r_eval_xy)
        log(f"[{tag}] base: " + " ".join(f"g{j:+d} {out['base'][j]['mean_pz']:.4f}"
                                         for j in GEOS)
            + " | held30: " + " ".join(
                f"g{j:+d} {out['base_held'][j]['mean_pz']:.4f}" for j in GEOS)
            + f" | CE_R {out['ce_r']:.4f}")
        out["site_read"] = read_fact_at(net, pool_183_x, name_ids, zid,
                                        SITE_ADDR_ROW, SITE_Z_XCOL)
        out["site_read_band"] = read_fact_at(net, pool_band_x, name_ids, zid,
                                             BAND_ADDR_ROW, BAND_Z_XCOL)
        log(f"[{tag}] site read @183: onset "
            f"{out['site_read']['pz_onset_mean']:.4f} span "
            f"{out['site_read']['pname_mean_over7']:.4f} | @band(129): onset "
            f"{out['site_read_band']['pz_onset_mean']:.4f} span "
            f"{out['site_read_band']['pname_mean_over7']:.4f}")
        out["census_old"] = row_census(net, ROWS_OLD,
                                       lambda n: battery_pz(n, ids130, zid))
        co = out["census_old"]["rows"]
        out["old_band"] = {
            "base_pz": out["census_old"]["base_readout"],
            "row0_strength": co["0"]["strength"],
            "A129": co["129"]["strength"],
            "band121_129_max": max(co[str(r)]["strength"]
                                   for r in range(121, 130)
                                   if str(r) in co)}
        log(f"[{tag}] old band: row0 S {out['old_band']['row0_strength']:+.4f}"
            f" | A(129) {out['old_band']['A129']:+.4f}")
        DELS = {"d_all": D_ALL, "d183": (SITE_ADDR_ROW,)}
        if rider:
            DELS["d129"] = (PRE - 1,)
        out["del_table"] = {}
        for dl, rows_ in DELS.items():
            sd_d, gate = deleted_wpe(sd, rows_)
            gates_surg[f"{tag}__{dl}"] = gate
            if not gate["pass"]:
                raise RuntimeError(f"deletion gate FAILED {tag}/{dl}: {gate}")
            net.load_state_dict(sd_d)
            cell = {"g0": battery_cell(net, bat_ids[(0, "install60")],
                                      zid)["mean_pz"]}
            if dl == "d183":
                cell["gm12"] = battery_cell(net, bat_ids[(-12, "install60")],
                                            zid)["mean_pz"]
            out["del_table"][dl] = cell
        net.load_state_dict(sd)
        log(f"[{tag}] deletions g0: " + " | ".join(
            f"{dl} {out['del_table'][dl]['g0']:.3f}" for dl in DELS))
        if rider:                                   # the 2x2 site censuses
            out["census183_span"] = row_census(
                net, ROWS_183,
                lambda n: read_fact_at(n, pool_183_x, name_ids, zid,
                                       SITE_ADDR_ROW,
                                       SITE_Z_XCOL)["pname_mean_over7"])
            out["census_jit8_span"] = row_census(
                net, ROWS_JIT8,
                lambda n: read_fact_at(n, pool_p8_x, name_ids, zid,
                                       P8_ADDR_ROW, P8_XCOL)["pname_mean_over7"])
            out["census_band_span"] = row_census(
                net, ROWS_BAND,
                lambda n: read_fact_at(n, pool_band_x, name_ids, zid,
                                       BAND_ADDR_ROW,
                                       BAND_Z_XCOL)["pname_mean_over7"])
            for key, rows_ in (("site_183", SITE_ROWS),
                               ("site_jit8", tuple(range(175, 198))),
                               ("site_band", tuple(range(121, 138)))):
                cen = out[f"census{'_jit8' if 'jit8' in key else '183' if '183' in key else '_band'}_span"]
                cm = max(cen["rows"][str(r)]["strength"] for r in (60, 100, 150)
                         if str(r) in cen["rows"])
                brows = [r for r in rows_ if str(r) in cen["rows"]]
                out[key] = {
                    "control_max": cm,
                    "site_strength": max(cen["rows"][str(r)]["strength"]
                                         for r in brows),
                    "site_pos": any(
                        cen["rows"][str(r)]["content"]
                        and cen["rows"][str(r)]["strength"] >= 2.0 * cm
                        for r in brows)}
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
                "site_read_span": m["site_read"]["pname_mean_over7"],
                "band_read_onset": m["site_read_band"]["pz_onset_mean"],
                "A129": m["old_band"]["A129"],
                "row0_strength": m["old_band"]["row0_strength"],
                "dall_g0": m["del_table"]["d_all"]["g0"],
                "d183_g0": m["del_table"]["d183"]["g0"]}

    # =====================================================================
    # STAGE A — CONSOLIDATE (the e113 jitter replay ported; seed 10901)
    # =====================================================================
    log("=" * 78)
    log("STAGE A BEFORE (the family-2 install net, jitter-geometry battery)")
    inst_jit = {j: battery_cell(evl_load(sd_inst), bat_ids[(j, "install60")],
                                zid)["mean_pz"] for j in JITTERS}
    inst_door = {j: battery_cell(evl_load(sd_inst), bat_ids[(j, "install60")],
                                 zid)["mean_pz"] for j in GEOS}
    log("  install jitter-geos: "
        + " ".join(f"g{j:+d} {inst_jit[j]:.4f}" for j in JITTERS)
        + " | ruler: " + " ".join(f"g{j:+d} {inst_door[j]:.4f}" for j in GEOS))

    if not SMOKE:
        log(f"[thermal] cooldown {COOLDOWN_S:.0f}s before stage A training")
        cooldown(COOLDOWN_S)
    log(f"STAGE A CONSOLIDATE: e113 jittered replay VERBATIM (pool "
        f"{pool_cons_x.shape[0]}, offsets {list(JITTERS)}), {FT_STEPS} steps, "
        f"seed {CONS_SEED}")
    armA = finetune_replay("consolidate", evl_load(sd_inst), pool_cons_x,
                           pool_cons_mask, anchor, train_ids, r_eval_xy,
                           ids130, zid, CONS_SEED)
    sd_cons = armA["sd"]
    save_ckpt("e157_f2_consolidated", sd_cons,
              {"desc": "e098_install_s4305 + 300-step e113 jittered replay "
                       f"(offsets {list(JITTERS)}, seed {CONS_SEED}) — the "
                       "family-2 consolidated net",
               "steps": int(armA["steps_ran"]), "seed": CONS_SEED,
               "device": armA["device"],
               "base": f"runs/checkpoints/{INST_CK}"})

    log("STAGE A AFTER (consolidated dial)")
    cons = measure(sd_cons, "consolidated", rider=True)
    cons_jit = {j: battery_cell(evl_load(sd_cons), bat_ids[(j, "install60")],
                                zid)["mean_pz"] for j in JITTERS}
    r1i_c = E43.eval_seq(evl_load(sd_cons), bat_i_seq, L, PRE - 1)
    log("  consolidated jitter-geos: "
        + " ".join(f"g{j:+d} {cons_jit[j]:.4f}" for j in JITTERS)
        + f" | R1i {r1i_c['nll']:.4f}")
    n_ge_05 = sum(v >= CONS_PZ_BAR for v in cons_jit.values())
    G_CONS = {
        "jitter_geos_pz": {f"g{j:+d}": cons_jit[j] for j in JITTERS},
        "n_ge_05": n_ge_05, "bar_n": 4,
        "novel_endpoints_ge_05": bool(cons_jit[-8] >= CONS_PZ_BAR
                                      and cons_jit[8] >= CONS_PZ_BAR),
        "door_gm12": flat_cells(cons)["gm12"],
        "door_open": bool(flat_cells(cons)["gm12"] >= OPEN_BAR),
        "dials": {"install60_pz": flat_cells(cons)["g0"],
                  "r1i_nll": r1i_c["nll"]},
        "dials_pass": bool(flat_cells(cons)["g0"] >= GATE_PZ_FLOOR
                           and r1i_c["nll"] <= GATE_R1I_MAX),
        "family1_e113_ref": {f"g{j:+d}": E113_NONE[j] for j in JITTERS},
    }
    G_CONS["pass"] = bool(G_CONS["novel_endpoints_ge_05"] and n_ge_05 >= 4
                          and G_CONS["door_open"] and G_CONS["dials_pass"])
    door_txt = "open" if G_CONS["door_open"] else "NOT open"
    log(f"G_CONS (stage A gate): novel endpoints "
        f"{G_CONS['novel_endpoints_ge_05']} | n>=0.5 {n_ge_05}/5 | door g-12 "
        f"{G_CONS['door_gm12']:.4f} ({door_txt}) | dials pZ "
        f"{G_CONS['dials']['install60_pz']:.4f} / R1i "
        f"{G_CONS['dials']['r1i_nll']:.3f} -> "
        f"{'PASS' if G_CONS['pass'] else 'FAIL'}")
    if not G_CONS["pass"]:
        log("  !! G_CONS FAILED — downstream verdicts are BOUNDED by this "
            "(reported, not silently dropped)")
        deviations.append("G_CONS (stage A) FAILED — the family-2 "
                          "consolidated state did not meet the registered "
                          "gate; wash/rider numbers reported but their "
                          "verdicts carry this bound")

    # =====================================================================
    # STAGE B — THE WASH (e176N arm A VERBATIM; seed 10902)
    # =====================================================================
    log("=" * 78)
    if not SMOKE:
        log(f"[thermal] cooldown {COOLDOWN_S:.0f}s before stage B training")
        cooldown(COOLDOWN_S)
    log(f"STAGE B THE NEUTRAL WASH: e176n arm A VERBATIM on the family-2 "
        f"consolidated net ({CK_WASH[-1]} steps, neutral bank seed "
        f"{E170_ANCHOR_SEED}, batch 16+16 full-token CE, lr 1e-3, seed "
        f"{ARM_SEED}), checkpoints +{list(CK_WASH)}")
    wash = finetune_freeze("neutral", evl_load(sd_cons), anchor_neutral,
                           train_ids, itos, r_eval_xy,
                           bat_ids[(-12, "install60")], ids130, zid,
                           ARM_SEED, CK_WASH)
    G_DRAWFREE = {"zeph_violations": wash["zeph_violations"],
                  "pass": bool(wash["zeph_violations"] == 0)}
    assert G_DRAWFREE["pass"], "wash: name token leaked into a window"
    if not SMOKE:
        log(f"[thermal] cooldown {COOLDOWN_S:.0f}s after stage B training")
        cooldown(COOLDOWN_S)

    save_ckpt("e157_f2_neutral", wash["sds"][max(wash["sds"])],
              {"desc": "e157_f2_consolidated + "
                       f"{max(wash['sds'])}-step NEUTRAL-anchor plain-corpus "
                       f"freeze (e176n arm A verbatim; bank seed "
                       f"{E170_ANCHOR_SEED}), lr 1e-3, seed {ARM_SEED}",
               "steps": int(max(wash["sds"])), "seed": ARM_SEED,
               "device": wash["device"],
               "base": "runs/checkpoints/e157_f2_consolidated.pt"})
    if not SMOKE:
        for s in sorted(wash["sds"]):
            if s == max(wash["sds"]):
                continue
            save_ckpt(f"e157_f2_neutral_s{s}", wash["sds"][s],
                      {"desc": "e157_f2_consolidated + "
                               f"{s}-step NEUTRAL freeze (intermediate)",
                       "steps": int(s), "seed": ARM_SEED,
                       "device": wash["device"],
                       "base": "runs/checkpoints/e157_f2_consolidated.pt"})

    wash_batteries: dict = {"0": cons}
    for s in sorted(wash["sds"]):
        log(f"STAGE B +{s} dial")
        wash_batteries[str(s)] = measure(wash["sds"][s], f"n{s}")
    trace_wash = [{"freeze_steps": s, **flat_cells(wash_batteries["0" if s == 0 else str(s)])}
                  for s in [0] + sorted(wash["sds"])]

    # ---- stage B adjudication (registered clauses; no shopping)
    g50 = next((r["gm12"] for r in trace_wash if r["freeze_steps"] == 50),
               trace_wash[-1]["gm12"])
    g300 = trace_wash[-1]["gm12"]
    ck_gm12 = [r["gm12"] for r in trace_wash if r["freeze_steps"] > 0]
    lead_replicates = bool(g50 <= SHUT_BAR)
    lineage_bound = bool(g300 >= SURVIVE_BAR)
    earliest_under = next((r["freeze_steps"] for r in trace_wash
                           if r["freeze_steps"] > 0 and r["gm12"] <= SHUT_BAR),
                          None)
    every_ckpt_survive = bool(min(ck_gm12) >= SURVIVE_BAR)
    if lead_replicates:
        verdict_b = "LEAD-FINDING-REPLICATES"
        clause_b = (f"the family-2 consolidated fact dissolved under the "
                    f"neutral wash: g-12 {g50:.4f} <= {SHUT_BAR} at +50 "
                    f"(earliest checkpoint under the bar: "
                    f"{earliest_under}); family 1 (e176n arm A) read "
                    f"{E176N_A['gm12'][4]:.4f} at +50 — the lead finding is "
                    f"n=2 families and the lineage bound discharges.")
    elif lineage_bound:
        verdict_b = "LINEAGE-BOUND"
        clause_b = (f"the family-2 fact SURVIVED the wash: g-12 {g300:.4f} "
                    f">= {SURVIVE_BAR} at +300 (every-checkpoint form: "
                    f"{every_ckpt_survive}) — the finding is "
                    f"lineage-specific; the bound stands and the paper says "
                    f"so.")
    else:
        verdict_b = "TEXTURE"
        clause_b = (f"neither registered bar fired: g-12 {g50:.4f} at +50 "
                    f"(dissolve bar <= {SHUT_BAR}), {g300:.4f} at +300 "
                    f"(survive bar >= {SURVIVE_BAR}) — numbers reported, no "
                    f"bar shopping.")
    if not G_CONS["pass"]:
        clause_b += (" BOUNDED: stage A's G_CONS failed — this is not a "
                     "clean consolidated state.")
    log("=" * 78)
    log(f"STAGE B VERDICT: {verdict_b}")
    log(f"  wash g-12 trace: "
        + " ".join(f"+{r['freeze_steps']} {r['gm12']:.4f}" for r in trace_wash))
    log(f"  {clause_b}")
    log("=" * 78)

    # =====================================================================
    # STAGE C — THE 2x2 RIDER (e158's conventions on family 2; seed 10902)
    # =====================================================================
    rider_arms: dict[str, dict] = {}
    rider_plan = [
        ("jitter183", "jitter@183", pool_jit183_x, pool_jit183_mask,
         pool_183_x[:, :SITE_Z_XCOL],
         "e151's protocol with the position JITTERED (e113 set as deltas on "
         "+54): offsets {46,50,54,58,62}, 300-window pool"),
        ("locked_band", "locked@band", pool_band_x, pool_band_mask,
         ids130,
         "locked (zero-variance) replay at the home band (e119-L): offset 0, "
         "60-window pool"),
        ("locked183", "locked@183", pool_183_x, pool_183_mask,
         pool_183_x[:, :SITE_Z_XCOL],
         "e151's conversion protocol: locked replay at the novel site 183 "
         "(offset +54, 60-window pool)"),
    ]
    for i, (tag, label, px, pm, fev, desc) in enumerate(rider_plan):
        log("=" * 78)
        if not SMOKE:
            log(f"[thermal] cooldown {COOLDOWN_S:.0f}s before rider {label}")
            cooldown(COOLDOWN_S)
        log(f"STAGE C ARM {label}: {FT_STEPS}-step replay, seed {ARM_SEED} "
            f"({desc})")
        arm = finetune_replay(tag, evl_load(sd_cons), px, pm, anchor,
                              train_ids, r_eval_xy, fev, zid, ARM_SEED)
        save_ckpt(f"e157_f2_{tag}", arm["sd"],
                  {"desc": f"e157_f2_consolidated + {FT_STEPS}-step {desc}, "
                           f"name-only mask, seed {ARM_SEED}",
                   "steps": int(arm["steps_ran"]), "seed": ARM_SEED,
                   "device": arm["device"],
                   "base": "runs/checkpoints/e157_f2_consolidated.pt"})
        rider_arms[tag] = {"arm": arm, "desc": desc}
    if not SMOKE:
        log(f"[thermal] cooldown {COOLDOWN_S:.0f}s after the last training")
        cooldown(COOLDOWN_S)

    for tag, label, _px, _pm, _fev, _desc in rider_plan:
        log(f"STAGE C AFTER battery, {label}")
        rider_arms[tag]["dial"] = measure(rider_arms[tag]["arm"]["sd"],
                                          f"after_{tag}", rider=True)

    gm12_j183 = flat_cells(rider_arms["jitter183"]["dial"])["gm12"]
    gm12_lband = flat_cells(rider_arms["locked_band"]["dial"])["gm12"]
    gm12_l183 = flat_cells(rider_arms["locked183"]["dial"])["gm12"]
    root_gm12 = flat_cells(cons)["gm12"]

    def door(v: float) -> str:
        return "OPEN" if v >= OPEN_BAR else ("SHUT" if v <= SHUT_BAR else "MID")

    cells_2x2_f2 = {
        "jitter_band": {"g_m12": root_gm12, "door": door(root_gm12),
                        "source": "the family-2 consolidated root by "
                                  "construction (stage A; in-run dial)"},
        "locked_183": {"g_m12": gm12_l183, "door": door(gm12_l183),
                       "source": "this run (conversion cell)"},
        "jitter_183": {"g_m12": gm12_j183, "door": door(gm12_j183),
                       "source": "this run"},
        "locked_band": {"g_m12": gm12_lband, "door": door(gm12_lband),
                        "source": "this run"},
    }
    f1_doors = {k: ("OPEN" if v >= OPEN_BAR else
                    ("SHUT" if v <= SHUT_BAR else "MID"))
                for k, v in F1_2X2.items()}
    flips = {k: bool(cells_2x2_f2[k]["door"] != f1_doors[k])
             for k in cells_2x2_f2}
    rider_gate_replicates = bool(gm12_j183 >= OPEN_BAR
                                 and gm12_lband >= OPEN_BAR
                                 and gm12_l183 <= SHUT_BAR)
    rider_lineage_bound = bool(any(flips.values()))
    if rider_gate_replicates:
        verdict_c = "GATE-REPLICATES (rider)"
        clause_c = (f"jitter@183 OPEN ({gm12_j183:.4f}) AND locked@band OPEN "
                    f"({gm12_lband:.4f}) AND locked@183 SHUT ({gm12_l183:.4f})"
                    f" — family 2 reproduces the e158/e151 pattern with the "
                    f"locked@band cell on the open side this time.")
    elif rider_lineage_bound:
        verdict_c = "LINEAGE-BOUND (rider)"
        clause_c = ("door flips vs family 1's committed cells: "
                    + ", ".join(f"{k} {f1_doors[k]} "
                                f"({F1_2X2[k]:.4f}) -> "
                                f"{cells_2x2_f2[k]['door']} "
                                f"({cells_2x2_f2[k]['g_m12']:.4f})"
                                for k in cells_2x2_f2 if flips[k])
                    + " — the rider's directions are family-specific.")
    else:
        verdict_c = "RIDER-TEXTURE"
        clause_c = (f"no rider bar fired cleanly: jitter@183 {gm12_j183:.4f} "
                    f"({door(gm12_j183)}), locked@band {gm12_lband:.4f} "
                    f"({door(gm12_lband)}), locked@183 {gm12_l183:.4f} "
                    f"({door(gm12_l183)}); no door flipped vs family 1 "
                    f"({', '.join(f'{k}:{f1_doors[k]}' for k in f1_doors)}) — "
                    f"directions match, magnitudes reported.")
    log("=" * 78)
    log(f"STAGE C (rider, report-only): {verdict_c}")
    log(f"  THE 2x2 family 2 (g-12 after): jitter@band(root) {root_gm12:.4f} "
        f"| locked@band {gm12_lband:.4f} | jitter@183 {gm12_j183:.4f} | "
        f"locked@183 {gm12_l183:.4f}")
    log(f"  family 1 committed:            jitter@band "
        f"{F1_2X2['jitter_band']:.4f} | locked@band "
        f"{F1_2X2['locked_band']:.4f} | jitter@183 "
        f"{F1_2X2['jitter_183']:.4f} | locked@183 "
        f"{F1_2X2['locked_183']:.4f}")
    log(f"  {clause_c}")
    log("=" * 78)

    # ---------------- reference provenance (embedded vs stored files)
    refs_prov = {
        "e176n_trace_armA": verify_ref(
            E176N_A, E43.REPO / "runs" / "e176n" / "metrics.json",
            lambda m: {"steps": [r["freeze_steps"] for r in m["trace_armA"]],
                       "gm12": [r["gm12"] for r in m["trace_armA"]],
                       "g0": [r["g0"] for r in m["trace_armA"]],
                       "ce_r": [r["ce_r"] for r in m["trace_armA"]]},
            "runs/e176n/metrics.json trace_armA"),
        "e113_none_table": verify_ref(
            E113_NONE, E43.REPO / "runs" / "e113" / "metrics.json",
            lambda m: {k: m["battery_table"][f"none__g{int(k):+d}__install60"]
                       ["mean_pz"] for k in E113_NONE},
            "runs/e113/metrics.json battery_table (none/install60)"),
        "e158_2x2": verify_ref(
            F1_2X2, E43.REPO / "runs" / "e158" / "metrics.json",
            lambda m: {k: m["cells_2x2"][k]["g_m12"] for k in F1_2X2},
            "runs/e158/metrics.json cells_2x2 (committed pass)"),
        "e098_s4305": verify_ref(
            E098_REF, E43.REPO / "runs" / "e098" / "metrics.json",
            lambda m: m["per_seed"]["4305"]["install"]["final"]
            | {"base_val_ce": m["per_seed"]["4305"]["base"]["val_ce"],
               "params": m["per_seed"]["4305"]["base"]["params"],
               "census_base_pz": m["per_seed"]["4305"]["census"]["base_pz"]},
            "runs/e098/metrics.json per_seed.4305"),
    }
    for k, v in refs_prov.items():
        log(f"ref {k}: verified={v['verified_vs_embedded']} "
            f"({v['source']})")

    # ---------------- outputs
    metrics = {
        "experiment": "e157_lineage_replication",
        "date": common.now_iso(),
        "purpose": ("THE LINEAGE REPLICATION: the conversion+wash arc on the "
                    "SECOND family (the e098 fresh 0.84M line, s4305) — "
                    "discharges or confirms the lead finding's one-lineage "
                    "bound (T112); absorbs the e157r 2x2 rider"),
        "registered_prediction": REGISTERED_PREDICTION,
        "family2": {"base": f"runs/checkpoints/{BASE_CK}",
                    "install": f"runs/checkpoints/{INST_CK}",
                    "cfg": {"n_layer": 4, "n_head": 4, "n_embd": 128,
                            "block_size": 512, "params": n_params},
                    "lineage": "e098 s4305 (e053c recipe corpus-1337 base + "
                               "e043 install verbatim; e098 GATE-0 passed, "
                               "no patience branch)"},
        "stages": {
            "A_consolidate": {
                "recipe": "e113 jittered replay VERBATIM (offsets {-8..+8}, "
                          "300-window pool, batch 16 pool + 16 anchors (8 "
                          "paired + 8 random), e043 union CE, AdamW "
                          "(0.9,0.95) wd 0.1 lr 1e-3 clip 1.0)",
                "seed": CONS_SEED, "steps_ran": armA["steps_ran"],
                "device": armA["device"], "traj": armA["traj"],
                "install_jitter_geos": {f"g{j:+d}": inst_jit[j]
                                        for j in JITTERS},
                "consolidated_jitter_geos": {f"g{j:+d}": cons_jit[j]
                                             for j in JITTERS},
                "family1_e113_ref": {f"g{j:+d}": E113_NONE[j]
                                     for j in JITTERS},
                "r1i_consolidated": r1i_c,
                "G_CONS": G_CONS,
                "dial": {k: v for k, v in cons.items() if k != "tag"},
            },
            "B_wash": {
                "recipe": "e176N arm A VERBATIM (e170 neutral bank seed 170, "
                          "batch 16 neutral + 16 random, full-token CE, "
                          "AdamW (0.9,0.95) wd 0.1 lr 1e-3 clip 1.0)",
                "seed": ARM_SEED, "steps_ran": wash["steps_ran"],
                "device": wash["device"], "traj": wash["traj"],
                "checkpoints": list(CK_WASH),
                "trace": trace_wash,
                "family1_e176n_armA_ref": E176N_A,
                "adjudication": {
                    "g50": g50, "g300": g300, "shut_bar": SHUT_BAR,
                    "survive_bar": SURVIVE_BAR,
                    "earliest_ck_le_bar": earliest_under,
                    "every_ckpt_survive_form": every_ckpt_survive,
                    "LEAD_FINDING_REPLICATES": {"clause": lead_replicates},
                    "LINEAGE_BOUND": {"clause": lineage_bound},
                    "verdict": verdict_b, "clause": clause_b,
                    "bounded_by_G_cons_failure": bool(not G_CONS["pass"]),
                },
                "dials": {s: {k: v for k, v in m.items() if k != "tag"}
                          for s, m in wash_batteries.items()},
            },
            "C_rider": {
                "recipe": "e158's conventions (arms from the family-2 "
                          "consolidated net; batch/optimizer as stage A)",
                "seed": ARM_SEED,
                "arms": {tag: {"desc": r["desc"], "steps_ran":
                               r["arm"]["steps_ran"],
                               "device": r["arm"]["device"],
                               "traj": r["arm"]["traj"],
                               "dial": {k: v for k, v in r["dial"].items()
                                        if k != "tag"}}
                         for tag, r in rider_arms.items()},
                "cells_2x2_family2": cells_2x2_f2,
                "cells_2x2_family1": {k: {"g_m12": F1_2X2[k],
                                          "door": f1_doors[k]}
                                      for k in F1_2X2},
                "flips_vs_family1": flips,
                "adjudication": {
                    "GATE_REPLICATES": {"clause": rider_gate_replicates,
                                        "bars": "jitter@183 >= 0.5 AND "
                                                "locked@band >= 0.5 AND "
                                                "locked@183 <= 0.27"},
                    "LINEAGE_BOUND": {"clause": rider_lineage_bound},
                    "verdict": verdict_c, "clause": clause_c,
                    "note": "report-only (the mission's rider clause); QUEUE "
                            "e157r's bars co-adjudicated",
                },
            },
        },
        "gates": {"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                  "G_GEO": G_GEO, "G_ANCHOR": G_ANCHOR, "G_GATE0": G_GATE0,
                  "G_DRAWFREE": G_DRAWFREE, "G_SURG": gates_surg,
                  "gpu_at_start": gpu_status(), "gpu_parked": GPU_PARKED,
                  "park_reason": PARK_REASON,
                  "device_events": device_events},
        "references": refs_prov,
        "checkpoints": CKPT_INVENTORY,
        "trims": trims,
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "smoke": SMOKE, "threads": torch.get_num_threads(),
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))
    log("metrics.json written")
    plot(rd / "lineage_replication.png", metrics)
    log(f"plot written -> {rd}")
    return 0


# ------------------------------------------------------------------ plot

def plot(path: Path, M: dict):
    stA, stB, stC = M["stages"]["A_consolidate"], M["stages"]["B_wash"], \
        M["stages"]["C_rider"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))

    # (0,0) THE WASH — family 2 (all three ruler geometries) vs family 1
    ax = axes[0, 0]
    t2 = [r["freeze_steps"] for r in stB["trace"]]
    g2 = [r["gm12"] for r in stB["trace"]]
    ax.plot(t2, g2, "o-", color="crimson", lw=2.4, ms=6,
            label="fam 2 g-12 (the bar's ruler)")
    ax.plot(t2, [r["g0"] for r in stB["trace"]], "^-", color="darkred",
            lw=1.4, ms=4.5, alpha=0.8, label="fam 2 g0 (well-expressed "
            "co-report)")
    ax.plot(t2, [r["gp12"] for r in stB["trace"]], "v-", color="salmon",
            lw=1.4, ms=4.5, alpha=0.8, label="fam 2 g+12 (co-report)")
    f1 = stB["family1_e176n_armA_ref"]
    ax.plot(f1["steps"], f1["gm12"], "s--", color="steelblue", lw=1.6, ms=5,
            label="family 1 (e176n arm A, 2.7M line)")
    ax.axhline(SHUT_BAR, color="gray", ls=":", lw=1.2,
               label=f"dissolve bar {SHUT_BAR}")
    ax.axhline(SURVIVE_BAR, color="seagreen", ls="--", lw=1.2,
               label=f"survive bar {SURVIVE_BAR}")
    ax.set_xscale("symlog", linthresh=2)
    ax.set_xticks([0, 1, 2, 4, 50, 100, 200, 300])
    ax.set_xticklabels(["0", "+1", "+2", "+4", "+50", "+100", "+200", "+300"])
    ax.set_xlabel("neutral-wash steps (e176n arm A verbatim, seed 10902)")
    ax.set_ylabel("install-60 battery p(Z)")
    adj = stB["adjudication"]
    bound_txt = (" | BOUNDED: root g-12 was 0.198 (< 0.5 G_CONS door); "
                 "g0/g+12 were well-expressed and ALSO dissolved at +1"
                 if adj.get("bounded_by_G_cons_failure") else "")
    ax.set_title(f"THE LINEAGE CELL — {adj['verdict']}{bound_txt}\n"
                 f"(g-12 {adj['g50']:.4f} @+50 vs bar {SHUT_BAR}; "
                 f"{adj['g300']:.4f} @+300 vs bar {SURVIVE_BAR})",
                 fontsize=9)
    ax.legend(fontsize=7)

    # (0,1) STAGE A — consolidation gate, jitter geometries
    ax = axes[0, 1]
    js = sorted(int(k[1:]) for k in stA["install_jitter_geos"])
    kfmt = lambda j: f"g{j:+d}"  # noqa: E731
    before = [stA["install_jitter_geos"][kfmt(j)] for j in js]
    after = [stA["consolidated_jitter_geos"][kfmt(j)] for j in js]
    ref = [stA["family1_e113_ref"][kfmt(j)] for j in js]
    xs = np.arange(len(js))
    ax.bar(xs - 0.26, before, 0.25, color="lightgray", edgecolor="k",
           linewidth=0.4, label="family-2 install (before)")
    ax.bar(xs, after, 0.25, color="seagreen", edgecolor="k", linewidth=0.4,
           label="family-2 consolidated (after)")
    ax.bar(xs + 0.26, ref, 0.25, color="whitesmoke", edgecolor="k",
           linewidth=0.6, hatch="//", label="family-1 consolidated (e113)")
    ax.axhline(CONS_PZ_BAR, color="crimson", ls="--", lw=1.1,
               label=f"gate bar {CONS_PZ_BAR}")
    for x, v in zip(xs, after):
        ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{j:+d}\n(row {129 + j})" for j in js], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("install-60 battery p(Z)")
    gc = stA["G_CONS"]
    ax.set_title(f"STAGE A consolidation — G_CONS "
                 f"{'PASS' if gc['pass'] else 'FAIL'} (door g-12 "
                 f"{gc['door_gm12']:.3f}; R1i {gc['dials']['r1i_nll']:.2f})",
                 fontsize=10)
    ax.legend(fontsize=7)

    # (1,0) THE 2x2 — family 2 vs family 1
    ax = axes[1, 0]
    keys = ["jitter_band", "locked_band", "jitter_183", "locked_183"]
    lbls = ["jitter@band\n(root)", "locked@band", "jitter@183", "locked@183"]
    f2v = [stC["cells_2x2_family2"][k]["g_m12"] for k in keys]
    f1v = [stC["cells_2x2_family1"][k]["g_m12"] for k in keys]
    xs = np.arange(len(keys))
    ax.bar(xs - 0.2, f2v, 0.38, color="crimson", edgecolor="k", linewidth=0.4,
           label="family 2 (this run)")
    ax.bar(xs + 0.2, f1v, 0.38, color="steelblue", edgecolor="k",
           linewidth=0.4, alpha=0.75, label="family 1 (e158/e151 committed)")
    ax.axhline(OPEN_BAR, color="seagreen", ls="--", lw=1.1,
               label=f"open {OPEN_BAR}")
    ax.axhline(SHUT_BAR, color="gray", ls=":", lw=1.1,
               label=f"shut {SHUT_BAR}")
    for x, v, c2 in zip(xs, f2v, stC["cells_2x2_family2"].values()):
        ax.text(x - 0.2, v + 0.012, f"{v:.3f}\n{c2['door']}", ha="center",
                fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(lbls, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("g-12 install-60 battery p(Z) after the arm")
    ax.set_title(f"THE 2x2 RIDER — {stC['adjudication']['verdict']}",
                 fontsize=10)
    ax.legend(fontsize=7)

    # (1,1) CE_R + held30 traces (the wash's price / generality)
    ax = axes[1, 1]
    ce2 = [r["ce_r"] for r in stB["trace"]]
    ax.plot(t2, ce2, "o-", color="darkorange", lw=2, ms=5,
            label="CE_R family 2")
    ax.plot(f1["steps"], f1["ce_r"], "s--", color="navy", lw=1.4, ms=4,
            label="CE_R family 1")
    ax.set_xscale("symlog", linthresh=2)
    ax.set_xticks([0, 1, 2, 4, 50, 100, 200, 300])
    ax.set_xticklabels(["0", "+1", "+2", "+4", "+50", "+100", "+200", "+300"])
    ax.set_xlabel("neutral-wash steps")
    ax.set_ylabel("CE_R (60 name-free val windows)")
    h2 = [r["held30_gm12"] for r in stB["trace"]]
    ax2 = ax.twinx()
    ax2.plot(t2, h2, "^:", color="purple", lw=1.6, ms=5,
             label="held30 g-12 (fam 2)")
    ax2.set_ylabel("held30 g-12 p(Z) (family 2)", color="purple")
    ax.legend(fontsize=7, loc="center left")
    ax2.legend(fontsize=7, loc="center right")
    ax.set_title("the wash's price (CE_R transient) and the held-30 mirror",
                 fontsize=10)

    fig.suptitle(
        f"E157 THE LINEAGE REPLICATION — family 2 = e098 s4305 (0.84M) | "
        f"wash: {M['stages']['B_wash']['adjudication']['verdict']} | rider: "
        f"{M['stages']['C_rider']['adjudication']['verdict']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
