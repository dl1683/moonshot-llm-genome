"""E147 — THE WIDTH LADDER (T079's pre-registered revival; bars registered in
THINKING.md card T084 at ~09:00Z BEFORE this dispatch — the bars below are
VERBATIM from that pre-registration; adjudicate against exactly this, no bar
shopping).

WHY: T076/e143 made consolidation's compass CAUSAL (choose the error's site,
choose the store's site) and killed proximity as the key-selection
alternative, leaving T079's invariance law alive by intervention after dying
on its observational dial (e140/T083: the trained-geometry presence dial
saturates for everyone). The law's remaining unknown is the WIDTH
dose-response: as jitter width grows, does the address key die exactly when
the route is born? This experiment makes that a co-measured ladder.

REGISTERED PREDICTION (VERBATIM, THINKING.md T084, ~09:00Z):
  "PRE-REGISTERED (e147, BEFORE dispatch): the width ladder w in
  {1,2,4,16,32,64} from the same root, co-measuring address-key strength
  A(w) (row-129 replacement delta) and novel-geometry routing NR(w) (row-0
  presence at g-12 + one more novel geometry). INVARIANCE-CAUSAL fires if
  A(w) is monotone decreasing (Spearman <= -0.8), crosses <= 0 at w*, with
  NR(w) onsetting (>= 2x install baseline) within one bin of the same w* —
  the co-onset of address-key death and route birth is the causal joint.
  DEAD-AGAIN if A(w) flat, or routing onsets while A(w) still >= +0.15.
  SEED-COVERAGE rider (W010's ghost): if routing peaks at +-8 and collapses
  at +-32/64, seeds have finite reach; T079-pure predicts +-64 routes at
  least as well as +-8."

OPERATIONALIZATIONS (frozen here before compute):
  * LADDER (primary, bars carried here): w = {0 (L@150, e119's locked net),
    1, 2, 4, 8 (e143_jitter — protocol-identical e113 recipe), 16, 32, 64},
    eight points. Sensitivity ladders (report-only): (a) w=0 -> e143_far
    (300-step locked, budget-matched, site 137-143); (b) near+far added as
    extra w=0 points (tie-corrected Spearman); (c) root (e048_repro, 0
    replay steps) as w=0.
  * WIDTH ARMS: pools of install windows at balanced integer offset grids
    spanning +-w — the e113 ±8 recipe's form {-8,-4,0,+4,+8} generalized:
    w=2 {-2,-1,0,1,2}, w=4 {-4,-2,0,2,4}, w=16 {-16,-8,0,8,16}, w=32
    {-32,-16,0,16,32}, w=64 {-64,-32,0,32,64}; w=1 {-1,0,1} (3-point grid,
    pool 180 windows vs 300 — recorded deviation; sampling is with
    replacement and the 300-step budget is matched). "Uniform offsets
    within +-w" = every host appears at every grid offset (balanced, like
    e113). g0 (offset 0) is in EVERY grid, so the A readout battery is
    in-distribution for all arms. g-12 and g+12 are trained geometries for
    NO ladder point (no grid contains +-12).
  * A(w) = row-129 replacement delta, e140's convention verbatim: census
    arms wpe[129] <- mean-of-all-rows / <- 0, drop = base - arm on the
    install-60 g0 battery (130-token contexts, p(Z) at last position,
    battery_pz/ids130 lineage), strength = min(mean-drop, zero-drop).
    NEGATIVE strength = the replacement HELPED (key turned suppressive) —
    e140's R texture (-0.21/-0.13).
  * NR(w) = row-0 presence-dependence at NOVEL geometry, e141's d_r0
    construction: zero-arm drop = pz_none(g) - pz_dr0(g) on the install-60
    battery at g-12 (PRIMARY, the registered headline geometry) and g+12
    (robustness, registered "one more novel geometry"). Install baseline =
    the ROOT's (e048_repro) zero-arm drop at the same geometry. ONSET(w) =
    NR(w) >= 2 x NR(root). The e131 mean-arm (wpe[0] <- mean row) is
    co-reported (convention supplement); the e141 ratio x = pz_dr0/pz_none
    is co-reported (collapse convention texture).
  * w* = the SMALLEST primary-ladder w with A(w) <= 0 (the crossing). Bins =
    ladder indices {0:0, 1:1, 2:2, 4:3, 8:4, 16:5, 32:6, 64:7}; "within one
    bin" = |idx(w_on) - idx(w*)| <= 1 with w_on = the FIRST primary-ladder
    onset width. Co-onset clause read on the g-12 primary; g+12 agreement
    reported as robustness texture.
  * "A(w) flat" (DEAD-AGAIN clause 1) = max(A) - min(A) <= 0.10 over the
    primary ladder. "Routing onsets while A(w) still >= +0.15" (DEAD-AGAIN
    clause 2) = onset exists AND A(w_on) >= +0.15 at the FIRST onset width.
  * SEED-COVERAGE = NR_g12(8) is the primary-ladder MAX AND NR_g12(32),
    NR_g12(64) both < 2 x NR(root) (peaked at ±8, collapsed at the wide
    end). T079-pure counter-read: NR(64) >= NR(8).
  * Adjudication order: INVARIANCE-CAUSAL -> DEAD-AGAIN -> SEED-COVERAGE ->
    TEXTURE (numbers). All sub-clause booleans reported regardless.
  * T084 FREE CELL (registered in T084 item 3, "Registered before
    running"): d_r0 at g-12 on e143_far — FAR-ROUTED-TAIL if pz_dr0 <=
    0.5 x pz_none (e141 die bar), CONTENT-TAIL if >= 0.8 x (survive bar),
    else AMBIGUOUS. Adjudicates whether FAR's 0.205 g-12 tail is routed
    through row-0 presence or the old band field expressing.

ROOT (mandated): runs/checkpoints/e048_repro.pt for ALL six new arms
(corpus seed 1337, SPLICE_RNG 24301, install60 battery p(Z) g0 = 0.556313;
row-0 install baseline 0.5455324 gated at 5e-6 with the 0.05 fallback
convention). FREE endpoints are LOADED AND DIALED, never retrained:
  w=0  L     runs/checkpoints/e119_l_locked_s150.pt  (e119 arm-b recipe,
             seed 10902, 150 steps — step-budget mismatch vs the 300-step
             ladder recorded as a comparability caveat)
  w=0  NEAR  runs/checkpoints/e143_near.pt   (300 steps locked, rows 5-11)
  w=0  FAR   runs/checkpoints/e143_far.pt    (300 steps locked, rows 137-143)
  w=8  R     runs/checkpoints/e143_jitter.pt (e113 recipe verbatim, 300
             steps, seed 10901 — protocol-identical to the six new arms)
  w=8  R8b   runs/checkpoints/e119_r_jittered_s300.pt (replicate, texture)

FINE-TUNE RECIPE (e143's finetune_arm = e109 arm-b = e119-L verbatim): batch
32 = 16 install windows from the arm's pool + 16 anchors (8 paired + 8
random), e043 token-level union CE on the 7 name-char targets, AdamW
(0.9,0.95) wd 0.1 constant lr 1e-3 clip 1.0, 300 steps, <=180 s GPU cap,
in-loop CPU evals every 25 steps (install-60 g0 battery + CE_R), seed 10901
for all six arms (e113 CONS_SEED convention), cooldown(90) between launches
(dispatch envelope 60-120 s), gpu_ok() double-poll gate per launch.

READOUTS per ladder point (all CPU, 8 threads):
  (i)   A(w): row-129 census strength (above), inside the full e140
        CENSUS_ROWS census (0 + controls 1-6 + band 121..137) for the
        band-spectrum texture.
  (ii)  NR(w): none/d_r0/mean_r0 cells at g-12 and g+12, install-60
        (primary) + held30 (texture, none/d_r0).
  (iii) TERTIARY: D-all{121,125,129,133,137} battery at g0 (none/d129/
        d_all/d_r0 cells, install-60).
  (iv)  Base-expression calibration column: none-cell g0 / g-12 / g+12 and
        CE_R (the drop ceilings: A <= base g0, NR <= base g).

COMPUTE ENVELOPE: GPU for the six fine-tunes (idle RTX 5090 checked via
gpu_ok(); NO concurrent GPU; e142 owns CPU lanes and is never touched;
torch threads 8, e143's convention on a 24-logical-core host). All readouts
CPU-side, sequential, no busy-wait.

Outputs: runs/e147/{metrics.json, width_ladder.png}; arm nets
runs/checkpoints/e147_w{1,2,4,16,32,64}.pt (ckpt_inventory in metrics).
No NOTES/THINKING/QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python e147_width_ladder.py    (E147_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import random
import sys
import time
from pathlib import Path

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")     # GPU for fine-tunes

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

torch.set_num_threads(8)                              # e143 convention (24 cores)

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT, cooldown, gpu_ok,     # noqa: E402
                    gpu_status, run_dir, save_json)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E147_SMOKE") == "1"
CPU = torch.device("cpu")
_USE_GPU = torch.cuda.is_available() and gpu_ok()
DEV = torch.device("cuda") if _USE_GPU else CPU

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
INSTALLED_CK = "e048_repro.pt"
CKPT_DIR = E43.REPO / "runs" / "checkpoints"

# ---- the width ladder --------------------------------------------------------
WIDTHS = (1, 2, 4, 16, 32, 64)     # the six NEW arms (registered set)
W8_FREE = "e143_jitter.pt"         # ±8 free endpoint (e113 recipe, same rig)
W0_FREE = "e119_l_locked_s150.pt"  # w=0 free endpoint (locked/L)
NEAR_FREE = "e143_near.pt"         # w=0 site-variant (rows 5-11)
FAR_FREE = "e143_far.pt"           # w=0 site-variant (rows 137-143)
W8B_FREE = "e119_r_jittered_s300.pt"   # ±8 replicate (texture)


def offset_grid(w: int) -> tuple[int, ...]:
    """Balanced integer offset grid spanning +-w (e113's {-8,-4,0,+4,+8}
    generalized). w=1 gets the 3-point grid {-1,0,1} (recorded deviation)."""
    if w == 1:
        return (-1, 0, 1)
    h = w // 2
    return (-w, -h, 0, h, w)


LADDER_W = (0, 1, 2, 4, 8, 16, 32, 64)          # primary ladder (8 points)
LADDER_IDX = {w: i for i, w in enumerate(LADDER_W)}

ADDR = PRE - 1                    # 129, the install address row
D_ALL = (121, 125, 129, 133, 137)                 # e113's D-all set
CONTROL_ROWS = (1, 2, 3, 4, 5, 6, 60, 100, 118, 119, 120)   # e140 verbatim
ADDR_BAND = tuple(r for r in range(121, 138))
CENSUS_ROWS = (0,) + CONTROL_ROWS + ADDR_BAND     # e140's row set verbatim
NOVEL_GEO = (-12, 12)             # g-12 primary; g+12 robustness (registered)

# ---- fine-tune envelope (e109 arm-b / e119-L / e143 verbatim) ----------------
FT_LR = 1e-3
FT_STEPS = 300 if not SMOKE else 8
FT_TIME_CAP = 180.0 if _USE_GPU else 1500.0
EVAL_EVERY = 25 if not SMOKE else 2
NAME_BS, ANCH_BS = 16, 16
ARM_SEED = 10901                   # e113 CONS_SEED / e143-JITTER convention
COOLDOWN_S = 90.0                  # dispatch envelope 60-120 s

# ---- gates / references -------------------------------------------------------
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
G_INST_REF = 0.556313             # e065/e091/e113/e119 install battery ref
G_INST_TOL = 0.005
R0_BASE_REF = 0.5455324279477395  # e131 stored install-phase row-0 strength
R0_MEAN_REF = 0.5549507188142645  # e131 stored install row-0 mean-arm drop
R0_TOL = 5e-6                     # e131 G_E116 convention
G_FALLBACK_TOL = 0.05             # e140's thread-order fallback convention

# free-endpoint gates (stored CPU cells from e143/e119 metrics; the 0.05
# fallback convention applies — this rig runs the same instruments on
# possibly different thread counts than the reference rigs)
FREE_GATES = {
    "L0":  {"path": W0_FREE, "ref_g0": 0.7376871109008789, "src": "e140 CKPTS L@150 cell"},
    "w8":  {"path": W8_FREE, "ref_g0": 0.7751880884, "src": "e143 del_table jitter__none"},
    "w8b": {"path": W8B_FREE, "ref_g0": 0.776076078414917, "src": "e109 GPU none-cell (tol 0.02)"},
    "near": {"path": NEAR_FREE, "ref_g0": 0.0004731952, "src": "e143 del_table near__none"},
    "far": {"path": FAR_FREE, "ref_g0": 0.0543758348, "src": "e143 del_table far__none"},
}
FAR_G12_REF = 0.20483240485191345  # e143 novel iv far g-12 (free-cell anchor)
NEAR_G12_REF = 0.001829148386605084
W8_G12_REF = 0.9143660068511963

# ---- registered bar constants (frozen) ----------------------------------------
RHO_BAR = -0.8                     # INVARIANCE-CAUSAL monotone clause
DEAD_FLAT_RANGE = 0.10             # DEAD-AGAIN clause 1: range(A) <= 0.10
DEAD_A_ALIVE = 0.15                # DEAD-AGAIN clause 2: A(w_on) >= +0.15
ONSET_MULT = 2.0                   # NR onset = >= 2x install baseline
ONE_BIN = 1                        # co-onset window (ladder indices)
DIE_BAR, SURVIVE_BAR = 0.5, 0.8    # e131/e141 collapse/survive convention

REGISTERED_PREDICTION = {
    "verbatim_T084": (
        "PRE-REGISTERED (e147, BEFORE dispatch): the width ladder w in "
        "{1,2,4,16,32,64} from the same root, co-measuring address-key "
        "strength A(w) (row-129 replacement delta) and novel-geometry "
        "routing NR(w) (row-0 presence at g-12 + one more novel geometry). "
        "INVARIANCE-CAUSAL fires if A(w) is monotone decreasing (Spearman "
        "<= -0.8), crosses <= 0 at w*, with NR(w) onsetting (>= 2x install "
        "baseline) within one bin of the same w* — the co-onset of "
        "address-key death and route birth is the causal joint. DEAD-AGAIN "
        "if A(w) flat, or routing onsets while A(w) still >= +0.15. "
        "SEED-COVERAGE rider (W010's ghost): if routing peaks at +-8 and "
        "collapses at +-32/64, seeds have finite reach; T079-pure predicts "
        "+-64 routes at least as well as +-8."),
    "free_cell_T084_item3": (
        "The free cell (d_r0@g-12 on e143_far) adjudicates: FAR-ROUTED-TAIL "
        "vs CONTENT-TAIL. Registered before running."),
    "no_bar_shopping": "No bar shopping; texture => TEXTURE with numbers.",
    "operationalizations": (
        "primary ladder w={0(L@150),1,2,4,8(e143_jitter),16,32,64}; A = "
        "min(mean,zero) drop at row 129 on ids130 (e140 convention); NR = "
        "zero-arm row-0 drop at g-12 (primary) / g+12 (robustness) on the "
        "install-60 battery (e141 d_r0 construction); onset = NR >= 2x "
        "NR(root); w* = smallest w with A <= 0; w_on = first onset; "
        "co-onset = |idx(w_on) - idx(w*)| <= 1 (g-12 carries the bar); "
        "flat = range(A) <= 0.10; DEAD clause 2 = A(w_on) >= +0.15; "
        "SEED-COVERAGE = NR(8) ladder-max AND NR(32),NR(64) < 2x root; "
        "order INVARIANCE-CAUSAL -> DEAD-AGAIN -> SEED-COVERAGE -> TEXTURE."),
}

trims: list[str] = []
deviations: list[str] = [
    "w=1's balanced grid is the 3-point {-1,0,1} (a 5-point integer grid "
    "cannot span +-1) — pool 180 windows vs 300; the 300-step budget is "
    "matched, sampling is with replacement, and offset 0 remains in the "
    "grid so the A readout stays in-distribution.",
    "Free endpoints are cross-rig dials by construction: L@150 (e119, 150 "
    "steps, seed 10902) and e143_jitter (300 steps, seed 10901) both used "
    "the e109 arm-b finetune_arm recipe this run clones; e119_r_jittered_"
    "s300 is a same-recipe replicate dialed as texture. L@150's step-budget "
    "mismatch (150 vs 300) is the widest comparability gap on the primary "
    "ladder; sensitivity ladders with e143_far (300-step locked) and the "
    "root (0-step) bracket it.",
    "Gate tolerances: this rig runs torch threads 8; free-endpoint gate "
    "cells were produced by e140/e143 (threads 8) and e109 (GPU). The 5e-6 "
    "bit tolerance is attempted first, the 0.05 thread-order fallback "
    "convention (e140 precedent) applies otherwise, per-gate recorded.",
    "GPU float nondeterminism precedent (e119's R rerun deviated 0.216 at "
    "g0, same seed/recipe): the six new arms are fresh GPU runs with no "
    "bit-repro expectations; internal gates only (loss finite, 300 steps).",
    "Smoke mode trims: 8-step fine-tunes, 2 arms, reduced census; nothing "
    "adjudicated.",
]


# ------------------------------------------------------------------ gpu guard

def gate_launch(tag: str) -> None:
    """e119/e143's bounded gate_launch: gpu_ok() double-poll, wait, PARK."""
    t0 = time.time()
    while True:
        if gpu_ok():
            time.sleep(10)
            s2 = gpu_status()
            if gpu_ok():
                log(f"[gpu] launch '{tag}' ok (util {s2['util']:.0f}% temp "
                    f"{s2['temp']:.0f}C mem {s2['mem_used']:.0f}/"
                    f"{s2['mem_total']:.0f}MB)")
                return
        if time.time() - t0 > 1200.0:
            raise RuntimeError(f"PARK: GPU busy/hot for 1200s "
                               f"({gpu_status()}) — refusing '{tag}'")
        time.sleep(30.0)


# ------------------------------------------------------------------ instruments
# PROVENANCE: load_cpu / evl_load / battery_cell / battery_pz / ce_fixed_cpu /
# val_windows / deleted_wpe are e143's copies of the e131 instruments
# (e068/e109/e113/e116 lineage) — copied rather than imported because
# importing e131 force-sets CUDA_VISIBLE_DEVICES=-1. finetune_arm is
# e143's VERBATIM (e109 arm-b recipe with the bounded gate_launch and
# DEV parameterized). row_census is e139/e143's generalized census.

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


def row_census(net: TinyGPT, rows, readout, *rargs) -> dict:
    """e139/e140's census VERBATIM (mean-arm / zero-arm / restore), scalar
    readout passed as a callable (battery_pz on ids130 — the e140 A dial)."""
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
                       "strength": float(min(m_d[r], z_d[r]))}
              for r in rows}
    assert torch.equal(w, orig), "census failed to restore wpe"
    return {"base_readout": base, "rows": rows_d}


# ------------------------------------------------------------------ fine-tune

def finetune_arm(tag: str, net0: TinyGPT, pool_x: torch.Tensor,
                 pool_mask: torch.Tensor, anchor: torch.Tensor,
                 train_ids: torch.Tensor, r_eval_xy, f_eval_ids, zid: int,
                 seed: int):
    """e143's finetune_arm VERBATIM (e109 arm-b / e119-L recipe)."""
    if DEV.type == "cuda":
        gate_launch(tag)
    net = copy.deepcopy(net0).to(DEV)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    n_pool = pool_x.shape[0]
    n_anc = anchor.shape[0]
    traj, t_start = [], time.time()
    step = 0
    evl = copy.deepcopy(net0)          # CPU eval twin
    for step in range(1, FT_STEPS + 1):
        ix = torch.randint(n_pool, (NAME_BS,), generator=gen)
        aj = torch.randint(n_anc, (ANCH_BS // 2,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (ANCH_BS // 2,),
                           generator=gen)
        nw = pool_x[ix].to(DEV)
        anc = torch.cat([anchor[aj],
                         torch.stack([train_ids[s: s + BLOCK] for s in rj])], 0).to(DEV)
        x = torch.cat([nw[:, :-1], anc[:, :-1]], 0)
        y = torch.cat([nw[:, 1:], anc[:, 1:]], 0)
        m = torch.zeros(NAME_BS + ANCH_BS, x.shape[1], dtype=torch.bool, device=DEV)
        m[:NAME_BS] = pool_mask[ix].to(DEV)
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
        if step % EVAL_EVERY == 0 or step == FT_STEPS or \
                (time.time() - t_start) > FT_TIME_CAP:
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
        if (time.time() - t_start) > FT_TIME_CAP:
            log(f"  [{tag}] time cap {FT_TIME_CAP:.0f}s at s{step}")
            trims.append(f"{tag}: time cap at step {step}")
            break
    net.eval()
    sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    del net
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    return {"sd": sd_cpu, "traj": traj, "steps_ran": step, "seed": seed,
            "final_loss": float(loss.item())}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"e147_{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "e147", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name} "
        f"({', '.join(f'{k}={v}' for k, v in meta.items())})")


CKPT_INVENTORY: dict = {}


# ------------------------------------------------------------------ stats

def _ranks(v):
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=float)
    sv = np.asarray(v, dtype=float)[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman rho with average-rank tie correction (scipy-free)."""
    rx, ry = _ranks(x), _ranks(y)
    rx -= rx.mean(); ry -= ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e147_smoke" if SMOKE else "e147")
    log(f"E147 THE WIDTH LADDER (T079 revival; bars T084 verbatim; "
        f"smoke={SMOKE}) -> {rd}")
    log(f"compute: train device {DEV} (gpu_ok at start: {_USE_GPU}), "
        f"cpu threads {torch.get_num_threads()}")

    if not _USE_GPU and not SMOKE:
        deviations.append("GPU parked (gpu_ok() failed at startup or no CUDA) "
                          "— fine-tunes ran CPU-side under the 1500 s cap; "
                          "reported, adjudication unchanged.")

    # ---------------- protocol rebuild (e065/e091/e109/e113/e119/e143 verbatim)
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

    # ---------------- width-arm pools (e113's jit machinery, width grids)
    arm_widths = WIDTHS if not SMOKE else (1, 16)
    pools_x, pools_mask, grids = {}, {}, {}
    for w in arm_widths:
        grid = offset_grid(w)
        grids[w] = grid
        wins, mrow = [], []
        for j in grid:
            for p, h in install_occ:
                pre = train_ids[p - PRE - j: p]
                post = train_ids[p + len(h): p + len(h) + POST_CAP - j]
                win = torch.cat([pre, name_ids, post])
                if len(win) != BLOCK:
                    raise RuntimeError(f"window len {len(win)} != {BLOCK} at w={w} j={j}")
                if not torch.equal(win[PRE + j: PRE + j + L], name_ids):
                    raise RuntimeError(f"name not at x-col {PRE + j} (w={w} j={j})")
                wins.append(win)
                m = torch.zeros(BLOCK - 1, dtype=torch.bool)
                m[PRE - 1 + j: PRE - 1 + j + L] = True
                mrow.append(m)
        pools_x[w] = torch.stack(wins)
        pools_mask[w] = torch.stack(mrow)
        log(f"pool w={w:2d}: offsets {list(grid)} -> {len(wins)} windows "
            f"(name x-cols {PRE + grid[0]}..{PRE + grid[-1]})")

    anchor = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                          for p, _ in install_occ[:16]])       # e065/e109 bank

    # ---------------- batteries (e068 construction, novel geos included)
    bat_ids = {}
    for j in (0,) + NOVEL_GEO:
        for tag, occ in (("install60", install_occ), ("held30", held_occ)):
            cs = [train_text[p - PRE - j: p] for p, _ in occ]
            bat_ids[(j, tag)] = torch.stack([corpus.encode(c) for c in cs])
    f_eval = bat_ids[(0, "install60")]             # the G_INST battery
    ids130 = bat_ids[(0, "install60")]             # e116 130-token battery
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

    # ---------------- root net + instrument gates
    net0 = load_cpu(CKPT_DIR / INSTALLED_CK)
    evl = copy.deepcopy(net0)
    bz0 = battery_cell(evl, f_eval, zid)
    G_INST = {"battery_pz": bz0["mean_pz"], "ref": G_INST_REF, "tol": G_INST_TOL,
              "pass": bool(abs(bz0["mean_pz"] - G_INST_REF) < G_INST_TOL)}
    log(f"G_INST root battery p(Z) {bz0['mean_pz']:.6f} (ref {G_INST_REF}): "
        f"{'PASS' if G_INST['pass'] else 'FAIL'}")
    if not G_INST["pass"]:
        raise RuntimeError("instrument broken vs e065/e091/e113/e119/e143")

    cen0 = row_census(evl, (0, 1), lambda n: battery_pz(n, ids130, zid))
    G_R0BASE = {"row0_mean": cen0["rows"]["0"]["mean"],
                "row0_zero": cen0["rows"]["0"]["zero"],
                "row0_strength": cen0["rows"]["0"]["strength"],
                "ref_mean": R0_MEAN_REF, "ref_zero": R0_BASE_REF,
                "tol": R0_TOL,
                "bit_reproducible": bool(
                    abs(cen0["rows"]["0"]["mean"] - R0_MEAN_REF) < R0_TOL
                    and abs(cen0["rows"]["0"]["zero"] - R0_BASE_REF) < R0_TOL)}
    G_R0BASE["pass"] = bool(
        G_R0BASE["bit_reproducible"] or
        max(abs(cen0["rows"]["0"]["mean"] - R0_MEAN_REF),
            abs(cen0["rows"]["0"]["zero"] - R0_BASE_REF)) < G_FALLBACK_TOL)
    if not G_R0BASE["bit_reproducible"]:
        deviations.append(f"G_R0BASE bit-repro failed (diffs "
                          f"{abs(cen0['rows']['0']['mean'] - R0_MEAN_REF):.2e}/"
                          f"{abs(cen0['rows']['0']['zero'] - R0_BASE_REF):.2e}) "
                          f"— 0.05 thread-order convention pass")
    log(f"G_R0BASE root row-0 m/z {cen0['rows']['0']['mean']:.7f}/"
        f"{cen0['rows']['0']['zero']:.7f}: "
        f"{'PASS' if G_R0BASE['pass'] else 'FAIL'}")
    if not G_R0BASE["pass"]:
        raise RuntimeError("row-0 baseline drifted vs e131 stored values")
    sd_base = {k: v.clone() for k, v in net0.state_dict().items()}
    del evl

    # ---------------- SIX WIDTH FINE-TUNES (GPU, cooldown between)
    arms_sd, arms_meta = {"root": sd_base}, {"root": {
        "desc": "e048_repro install-phase root (the ladder's origin)",
        "steps": 0, "seed": None}}
    for k, w in enumerate(arm_widths):
        if k > 0:
            log("[thermal] cooldown(90) between training launches")
            cooldown(COOLDOWN_S)
        log(f"ARM w={w}: balanced grid {list(grids[w])}, seed {ARM_SEED}, "
            f"{FT_STEPS} steps from e048_repro")
        res = finetune_arm(f"w{w}", net0, pools_x[w], pools_mask[w], anchor,
                           train_ids, r_eval_xy, f_eval, zid, ARM_SEED)
        tag = f"w{w}"
        arms_sd[tag] = res["sd"]
        arms_meta[tag] = {"desc": f"width-{w} jittered replay (grid "
                                  f"{list(grids[w])})",
                          "traj": res["traj"], "seed": ARM_SEED,
                          "steps_ran": res["steps_ran"],
                          "final_loss": res["final_loss"]}
        save_ckpt(f"w{w}", res["sd"],
                  {"desc": f"width-{w} jittered replay, offsets "
                           f"{list(grids[w])}, 300 steps, seed {ARM_SEED}",
                   "width": w, "offsets": list(grids[w]),
                   "steps": res["steps_ran"], "seed": ARM_SEED,
                   "base": f"runs/checkpoints/{INSTALLED_CK}"})

    # ---------------- free endpoints (loaded + gated, never retrained)
    gates_free = {}
    free_map = {"L0": 0, "w8": 8, "near": 0, "far": 0, "w8b": 8}
    for tag, w in free_map.items():
        path = CKPT_DIR / FREE_GATES[tag]["path"]
        net_f = load_cpu(path)
        bz = battery_cell(net_f, f_eval, zid)["mean_pz"]
        tol = 0.02 if tag == "w8b" else G_FALLBACK_TOL
        g = {"battery_pz_g0": bz, "ref": FREE_GATES[tag]["ref_g0"],
             "src": FREE_GATES[tag]["src"], "tol": tol,
             "diff": abs(bz - FREE_GATES[tag]["ref_g0"]),
             "pass": bool(abs(bz - FREE_GATES[tag]["ref_g0"]) < tol)}
        gates_free[tag] = g
        log(f"G_FREE {tag:5s} ({path.name}): g0 {bz:.10f} (ref "
            f"{FREE_GATES[tag]['ref_g0']:.10f}): "
            f"{'PASS' if g['pass'] else 'FAIL'}")
        if not g["pass"]:
            raise RuntimeError(f"free endpoint {tag} failed its gate: {g}")
        arms_sd[tag] = {k: v.clone() for k, v in net_f.state_dict().items()}
        arms_meta[tag] = {"desc": f"FREE endpoint {path.name} (w={w}), "
                                  f"{FREE_GATES[tag]['src']}",
                          "steps": 150 if tag == "L0" else 300,
                          "loaded_from": f"runs/checkpoints/{FREE_GATES[tag]['path']}"}
        del net_f

    # secondary free-endpoint anchors (g-12 cells from e143, report-only)
    G_G12_ANCHORS = {}
    for tag, ref in (("far", FAR_G12_REF), ("near", NEAR_G12_REF),
                     ("w8", W8_G12_REF)):
        net_f = evl_load(arms_sd[tag])
        pz = battery_cell(net_f, bat_ids[(-12, "install60")], zid)["mean_pz"]
        G_G12_ANCHORS[tag] = {"pz_g12": pz, "ref": ref,
                              "diff": abs(pz - ref),
                              "tol": G_FALLBACK_TOL,
                              "pass": bool(abs(pz - ref) < G_FALLBACK_TOL)}
        log(f"G_G12  {tag:5s} g-12 {pz:.10f} (ref {ref:.10f}): "
            f"{'PASS' if G_G12_ANCHORS[tag]['pass'] else 'DRIFT'}")
        del net_f

    # ================= DIALS (eval-only, all CPU) =================
    log("=" * 78)
    log("DIALS (CPU): (i) A = row-129 census @ ids130 (+band spectra) | "
        "(ii) NR = d_r0 @ g-12/g+12 | (iii) D-all g0 | (iv) base calib")

    nets = {t: evl_load(sd) for t, sd in arms_sd.items()}
    dial_order = (["root", "L0", "near", "far"] +
                  [f"w{w}" for w in arm_widths[:2]] + ["w8", "w8b"] +
                  [f"w{w}" for w in arm_widths[2:]])
    dial_order = [t for t in dict.fromkeys(dial_order) if t in nets]

    census, A, r0_g0, ctrl_max = {}, {}, {}, {}
    for t in dial_order:
        cen = row_census(nets[t], CENSUS_ROWS,
                         lambda n: battery_pz(n, ids130, zid))
        census[t] = cen
        A[t] = cen["rows"]["129"]["strength"]
        r0_g0[t] = cen["rows"]["0"]["strength"]
        ctrl_max[t] = max(cen["rows"][str(r)]["strength"]
                          for r in CONTROL_ROWS if str(r) in cen["rows"])
        log(f"(i) {t:5s} A(129) {A[t]:+.4f} (m {cen['rows']['129']['mean']:+.4f} "
            f"/ z {cen['rows']['129']['zero']:+.4f}) | row0@g0 {r0_g0[t]:+.4f} "
            f"| ctrl-max {ctrl_max[t]:.4f} | base {cen['base_readout']:.4f}")

    # (ii) NR: none / d_r0 / mean_r0 at g-12 and g+12
    NR, nr_detail = {}, {}
    for t in dial_order:
        net = nets[t]
        sd_t = arms_sd[t]
        nr_detail[t] = {}
        for g in NOVEL_GEO:
            cells = {}
            for arm in ("none", "d_r0", "mean_r0"):
                if arm == "none":
                    sd_a = sd_t
                    gate = {"rows": [], "pass": True}
                else:
                    sd_a, gate = deleted_wpe(sd_t, (0,))
                    if arm == "mean_r0":          # e131 mean-arm convention
                        sd_a = {k: v.clone() for k, v in sd_a.items()}
                        sd_a["wpe.weight"][0] = sd_t["wpe.weight"].mean(0)
                    if not gate["pass"]:
                        raise RuntimeError(f"{t} {arm} gate FAILED")
                net.load_state_dict(sd_a)
                cells[arm] = {
                    "install60": battery_cell(net, bat_ids[(g, "install60")],
                                              zid)["mean_pz"],
                    "held30": (battery_cell(net, bat_ids[(g, "held30")],
                                            zid)["mean_pz"]
                               if arm in ("none", "d_r0") else None)}
            net.load_state_dict(sd_t)
            n12, d12, m12 = (cells["none"]["install60"],
                             cells["d_r0"]["install60"],
                             cells["mean_r0"]["install60"])
            det = {"cells": cells,
                   "pz_none": n12, "pz_dr0": d12, "pz_mean_r0": m12,
                   "NR_zero_drop": n12 - d12,
                   "NR_mean_drop": n12 - m12,
                   "ratio_dr0_over_none": d12 / max(n12, 1e-12),
                   "held30_NR_zero_drop": (cells["none"]["held30"]
                                           - cells["d_r0"]["held30"])}
            nr_detail[t][g] = det
            log(f"(ii){t:5s} g{g:+3d}: none {n12:.4f} d_r0 {d12:.4f} "
                f"(drop {n12 - d12:+.4f}, x{d12 / max(n12, 1e-12):.3f}) "
                f"mean {m12:.4f} (drop {n12 - m12:+.4f})")
        NR[t] = nr_detail[t][-12]["NR_zero_drop"]      # g-12 primary

    # (iii) D-all battery at g0 + (iv) base calibration
    dall, base_calib = {}, {}
    DELS = {"none": (), "d129": (ADDR,), "d_all": D_ALL, "d_r0": (0,)}
    for t in dial_order:
        net = nets[t]
        sd_t = arms_sd[t]
        cells = {}
        for dl, rows_ in DELS.items():
            if dl == "none":
                sd_del = sd_t
                gate = {"rows": [], "pass": True}
            else:
                sd_del, gate = deleted_wpe(sd_t, rows_)
            if not gate["pass"]:
                raise RuntimeError(f"{t} {dl} deletion gate FAILED")
            net.load_state_dict(sd_del)
            cells[dl] = battery_cell(net, bat_ids[(0, "install60")], zid)[
                "mean_pz"]
        net.load_state_dict(sd_t)
        dall[t] = cells
        base_calib[t] = {"g0_none": cells["none"],
                         "g-12_none": nr_detail[t][-12]["pz_none"],
                         "g+12_none": nr_detail[t][12]["pz_none"],
                         "ce_r": ce_fixed_cpu(net, *r_eval_xy)}
        log(f"(iii){t:5s}: " + " | ".join(f"{dl} {cells[dl]:.3f}"
                                          for dl in DELS)
            + f" | CE_R {base_calib[t]['ce_r']:.4f}")

    del nets

    # ================= ADJUDICATION (registered, verbatim clauses) ==========
    # primary ladder, explicitly width-sorted (free endpoints interleaved)
    _ladder_pts = [(0, "L0")] + [(w, f"w{w}") for w in arm_widths] + \
        [(8, "w8")]
    _ladder_pts.sort(key=lambda p: p[0])
    prim_ws = [w for w, _ in _ladder_pts]
    prim_tags = [t for _, t in _ladder_pts]
    A_vec = [A[t] for t in prim_tags]
    NR_vec = [NR[t] for t in prim_tags]
    NR_root = NR["root"]

    rho = spearman(prim_ws, A_vec)
    # sensitivity ladders (report-only)
    sens = {
        "far_for_L0": spearman([0] + prim_ws[1:], [A["far"]] + A_vec[1:]),
        "near_far_added": spearman(
            [0, 0, 0] + prim_ws[1:],
            [A["L0"], A["near"], A["far"]] + A_vec[1:]),
        "root_for_L0": spearman([0] + prim_ws[1:], [A["root"]] + A_vec[1:]),
    }

    crossing_ws = [w for w, a in zip(prim_ws, A_vec) if a <= 0.0]
    w_star = crossing_ws[0] if crossing_ws else None
    onset_ws = [w for w, n in zip(prim_ws, NR_vec) if n >= ONSET_MULT * NR_root]
    w_on = onset_ws[0] if onset_ws else None
    onset_set = onset_ws

    mono_ok = bool(rho <= RHO_BAR)
    cross_ok = w_star is not None
    onset_ok = w_on is not None
    co_onset = bool(w_star is not None and w_on is not None and
                    abs(LADDER_IDX[w_on] - LADDER_IDX[w_star]) <= ONE_BIN)
    g12_plus_agree = None
    if w_on is not None:
        nr_p = nr_detail[f"w{w_on}" if w_on != 0 else "L0"][12]["NR_zero_drop"]
        g12_plus_agree = bool(nr_p >= ONSET_MULT *
                              nr_detail["root"][12]["NR_zero_drop"])

    invariance_causal = bool(mono_ok and cross_ok and onset_ok and co_onset)

    a_range = max(A_vec) - min(A_vec)
    dead_flat = bool(a_range <= DEAD_FLAT_RANGE)
    dead_early = bool(w_on is not None and A[f"w{w_on}" if w_on != 0 else "L0"]
                      >= DEAD_A_ALIVE)
    dead_again = bool(dead_flat or dead_early)

    nr8 = NR["w8"] if "w8" in NR else None
    seed_coverage = bool(
        nr8 is not None and nr8 >= max(NR_vec)
        and NR[f"w{arm_widths[-2]}"] < ONSET_MULT * NR_root
        and NR[f"w{arm_widths[-1]}"] < ONSET_MULT * NR_root)

    # T084 free cell: d_r0 @ g-12 on e143_far
    far_det = nr_detail["far"][-12]
    far_cell = {"pz_none": far_det["pz_none"], "pz_dr0": far_det["pz_dr0"],
                "ratio": far_det["ratio_dr0_over_none"],
                "die_bar": DIE_BAR, "survive_bar": SURVIVE_BAR,
                "verdict": ("FAR-ROUTED-TAIL"
                            if far_det["pz_dr0"] <= DIE_BAR * far_det["pz_none"]
                            else "CONTENT-TAIL"
                            if far_det["pz_dr0"] >= SURVIVE_BAR * far_det["pz_none"]
                            else "AMBIGUOUS")}

    bars = {
        "invariance_causal": {
            "fired": invariance_causal,
            "spearman_rho": rho, "rho_bar": RHO_BAR, "mono_ok": mono_ok,
            "crossing_w_star": w_star, "cross_ok": cross_ok,
            "first_onset_w": w_on, "onset_set": onset_set, "onset_ok": onset_ok,
            "co_onset_within_one_bin": co_onset,
            "g+12_agrees_with_onset": g12_plus_agree,
            "clause": ("A(w) monotone decreasing (rho<=-0.8), crosses <=0 at "
                       "w*, NR onsets (>=2x install baseline) within one bin "
                       "of the same w*")},
        "dead_again": {
            "fired": dead_again,
            "flat_clause": {"fired": dead_flat, "range_A": a_range,
                            "bar": DEAD_FLAT_RANGE},
            "early_onset_clause": {"fired": dead_early,
                                   "A_at_first_onset": (
                                       A[f"w{w_on}"] if w_on not in (None, 0)
                                       else A.get("L0")) if w_on is not None
                                   else None,
                                   "bar": DEAD_A_ALIVE}},
        "seed_coverage": {
            "fired": seed_coverage,
            "NR_at_8": nr8, "NR_ladder_max": max(NR_vec),
            "NR_root": NR_root, "onset_bar": ONSET_MULT * NR_root,
            "clause": "routing peaks at +-8 and collapses at +-32/64"},
        "free_cell_T084_far": far_cell,
        "sensitivity_ladders_report_only": sens,
    }

    if invariance_causal:
        verdict = "INVARIANCE-CAUSAL"
        clause = (f"A(w) monotone decreasing (rho {rho:+.3f} <= {RHO_BAR}), "
                  f"crosses <= 0 at w*={w_star} (A {A[f'w{w_star}' if w_star != 0 else 'L0']:+.4f}), "
                  f"NR onsets at w={w_on} (bins |{LADDER_IDX[w_on]}-"
                  f"{LADDER_IDX[w_star]}| <= 1) — the co-onset of address-key "
                  f"death and route birth; the causal joint holds.")
    elif dead_again:
        verdict = "DEAD-AGAIN"
        why = ("A(w) flat (range "
               f"{a_range:.4f} <= {DEAD_FLAT_RANGE})" if dead_flat else
               f"routing onsets at w={w_on} while A is still "
               f"{A[f'w{w_on}' if w_on != 0 else 'L0']:+.4f} >= +{DEAD_A_ALIVE}")
        clause = (f"{why} — two independent switches; credit-competition "
                  f"falsified (rho {rho:+.3f}, w* {w_star}, w_on {w_on}).")
    elif seed_coverage:
        verdict = "SEED-COVERAGE"
        clause = (f"routing peaks at ±8 (NR {nr8:.4f} = ladder max) and "
                  f"collapses at ±{arm_widths[-2]}/{arm_widths[-1]} "
                  f"(NR {NR[f'w{arm_widths[-2]}']:.4f}/"
                  f"{NR[f'w{arm_widths[-1]}']:.4f} < onset bar "
                  f"{ONSET_MULT * NR_root:.4f}) — seeds have finite reach; "
                  f"T079-pure (±64 >= ±8) falsified in this lineage.")
    else:
        verdict = "TEXTURE"
        clause = (f"no registered bar fired cleanly: rho {rho:+.3f} "
                  f"(bar <= {RHO_BAR}), A range {a_range:.4f}, w* {w_star}, "
                  f"w_on {w_on}, onset set {onset_ws}, NR(8) "
                  f"{nr8 if nr8 is None else round(nr8, 4)} vs ladder max "
                  f"{max(NR_vec):.4f} — numbers reported, no bar shopping.")

    log("=" * 78)
    log(f"E147 VERDICT: {verdict}")
    log(f"  A(w):  " + " ".join(f"{t}({w}) {A[t]:+.4f}" for t, w
                                in zip(prim_tags, prim_ws)))
    log(f"  NR(w): " + " ".join(f"{t} {NR[t]:+.4f}" for t in prim_tags)
        + f" | root baseline {NR_root:+.4f} (onset bar "
          f"{ONSET_MULT * NR_root:.4f})")
    log(f"  rho {rho:+.3f} | w* {w_star} | first onset {w_on} "
        f"(set {onset_ws}) | co-onset {co_onset} | g+12 agrees "
        f"{g12_plus_agree}")
    log(f"  DEAD clauses: flat {dead_flat} (range {a_range:.4f}), "
        f"early-onset {dead_early} | SEED-COVERAGE {seed_coverage}")
    log(f"  free cell far d_r0@g-12: {far_det['pz_none']:.4f} -> "
        f"{far_det['pz_dr0']:.4f} (x{far_det['ratio_dr0_over_none']:.3f}) -> "
        f"{far_cell['verdict']}")
    log(f"  {clause}")
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e147_width_ladder",
        "date": common.now_iso(),
        "registration": ("THINKING.md card T084 pre-registration ~09:00Z, "
                         "BEFORE dispatch; bars verbatim below; "
                         "operationalizations frozen in the module docstring "
                         "before compute"),
        "registered_prediction": REGISTERED_PREDICTION,
        "committed_prediction": "INVARIANCE-CAUSAL (T079-pure form)",
        "prediction_held": bool(verdict == "INVARIANCE-CAUSAL"),
        "question": ("as jitter width grows, does the address key die exactly "
                     "when the route is born? (the invariance law's "
                     "dose-response; T079's pre-registered revival)"),
        "root": f"runs/checkpoints/{INSTALLED_CK} (all six width arms)",
        "ladder": {
            "primary": [{"tag": t, "w": w, "net": (
                "loaded:" + arms_meta[t].get("loaded_from",
                                             f"runs/checkpoints/e147_{t}.pt")
                if t in ("L0", "w8", "near", "far", "w8b")
                else f"runs/checkpoints/e147_{t}.pt"),
                "A": A[t], "NR_g12": NR[t],
                "NR_g12_plus": nr_detail[t][12]["NR_zero_drop"],
                "base_g0": base_calib[t]["g0_none"],
                "d_all_g0": dall[t]["d_all"], "ce_r": base_calib[t]["ce_r"]}
                for t, w in zip(prim_tags, prim_ws)],
            "extras": {"root": {"w": 0, "A": A["root"], "NR_g12": NR["root"]},
                       "near": {"w": 0, "A": A["near"], "NR_g12": NR["near"]},
                       "far": {"w": 0, "A": A["far"], "NR_g12": NR["far"]},
                       "w8b": {"w": 8, "A": A["w8b"], "NR_g12": NR["w8b"]}},
            "width_grids": {str(w): list(grids[w]) for w in grids},
        },
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
                     "finetune": "e109 arm-b / e143 finetune_arm verbatim "
                                 "(batch 32, union CE on 7 name targets, "
                                 "AdamW 1e-3 const wd 0.1 clip 1.0, 300 "
                                 f"steps, seed {ARM_SEED})",
                     "battery_construction": "ctx = train_text[p-PRE-j:p], "
                                             "readout p(Z) at last position "
                                             "(wpe row 129+j); g-12/g+12 are "
                                             "trained geometries for NO "
                                             "ladder point"},
        "gates": {"G_SPLICE": G_SPLICE, "G_INST": G_INST, "G_R0BASE": G_R0BASE,
                  "G_FREE": gates_free, "G_G12_anchors_report_only":
                      G_G12_ANCHORS,
                  "gpu_at_start": {"use_gpu": _USE_GPU,
                                   "status": gpu_status()}},
        "arms": arms_meta,
        "dial_A_address_key": {
            "convention": "e140: row-129 min(mean,zero) drop on ids130 "
                          "(install-60 g0 battery)",
            "values": A,
            "row0_g0_texture_e140_S": r0_g0,
            "control_max": ctrl_max},
        "dial_NR_novel_geometry": {
            "convention": "e141 d_r0: zero-arm row-0 drop at g-12 (primary) "
                          "and g+12 (robustness), install-60 battery; "
                          "install baseline = root",
            "NR_root_g12": NR_root,
            "onset_bar": ONSET_MULT * NR_root,
            "values": NR,
            "detail": nr_detail},
        "dial_D_all_g0": dall,
        "base_calibration": base_calib,
        "census_rows": census,
        "bars": bars,
        "adjudication": {"verdict": verdict, "clause": clause,
                         "fired_order": "INVARIANCE-CAUSAL -> DEAD-AGAIN -> "
                                        "SEED-COVERAGE -> TEXTURE"},
        "honesty_reflex": [
            "SINGLE LINEAGE: every ladder point descends from the ONE "
            "e048_repro install (corpus 1337 / SPLICE_RNG 24301); width "
            "effects are width-caused WITHIN this lineage, but the "
            "install's idiosyncrasies are not sampled (no seed replication "
            "anywhere on the ladder).",
            "NR CEILING BEHAVIOR: NR is a drop bounded by the arm's own "
            "novel-geometry expression (pz_none); the root's g-12 base is "
            "near floor, so the 2x-baseline onset bar is easy to clear once "
            "ANY novel-geometry expression exists — the onset's TIMING "
            "(vs w*) is the informative half of the joint, and the "
            "ratio-form dial (pz_dr0/pz_none, e141 convention) is "
            "co-reported for every point.",
            "FREE-ENDPOINT COMPARABILITY ACROSS RIGS: L@150 was trained "
            "150 steps (e119's mass-matching) vs the ladder's 300 — the "
            "w=0 A value inherits a half-budget; sensitivity ladders with "
            "e143_far (300-step locked) and the root (0-step) bracket the "
            "w=0 point. e143_jitter (w=8) is protocol-identical to the new "
            "arms; e119_r300 (w8b) is a same-recipe replicate.",
            "A-DIAL READOUT: the g0 battery is in-distribution for every "
            "arm (offset 0 in every grid) — A measures the OLD address "
            "key's fate under each width's replay, not an out-of-distribution "
            "artifact; negative A (replacement helps) counts as key death "
            "per the registered 'crossing <= 0'.",
            "g±12 DISTANCE TO THE TRAINED SET varies with width (nearest "
            "trained offset is 1..12 rows away depending on grid); the "
            "novel-geometry read is genuinely untrained everywhere but not "
            "equidistant from the trained manifold across arms.",
        ],
        "ckpt_inventory": {"saved": CKPT_INVENTORY,
                           "external_used": [
                               f"runs/checkpoints/{INSTALLED_CK}"]
                           + [f"runs/checkpoints/{FREE_GATES[t]['path']}"
                              for t in FREE_GATES],
                           "note": "*.pt gitignored — on-disk persistence"},
        "trims": trims, "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": int(net0.num_params()),
                   "train_device": str(DEV), "eval_device": "cpu",
                   "torch_threads": torch.get_num_threads(), "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    # ---------------- plot (THE dose-response figure + supports)
    plot(rd / "width_ladder.png", prim_tags, prim_ws, A, NR, nr_detail, dall,
         base_calib, census, bars, verdict, clause, arm_widths)

    log(f"outputs: {rd / 'metrics.json'}, {rd / 'width_ladder.png'}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot

def plot(path, prim_tags, prim_ws, A, NR, nr_detail, dall, base_calib,
         census, bars, verdict, clause, arm_widths):
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.5))

    def xw(ws):
        return np.array(ws, dtype=float)

    # (0,0) THE A(w) PANEL — address-key death
    ax = axes[0, 0]
    ax.axhline(0, color="k", lw=0.8)
    ax.plot(xw(prim_ws), [A[t] for t in prim_tags], "o-", ms=8, lw=1.8,
            color="crimson", label="A(w) primary ladder")
    extras = [("root", 0, "gray"), ("near", 0, "tab:purple"),
              ("far", 0, "darkorange"), ("w8b", 8, "seagreen")]
    for t, w, c in extras:
        ax.plot([w], [A[t]], "x", ms=9, mew=2, color=c, alpha=0.8,
                label=f"{t} (free, w={w})")
    if bars["invariance_causal"]["crossing_w_star"] is not None:
        w_star = bars["invariance_causal"]["crossing_w_star"]
        ax.axvline(w_star, color="crimson", ls="--", lw=1.6, alpha=0.7)
        ax.annotate(f"w* = {w_star}", xy=(w_star, 0.05),
                    xytext=(6, 10), textcoords="offset points", fontsize=10,
                    color="crimson", fontweight="bold")
    for t, w in zip(prim_tags, prim_ws):
        ax.annotate(f"{A[t]:+.3f}", xy=(w, A[t]), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=7.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0] + list(prim_ws[1:]))
    ax.set_xticklabels(["0(L)"] + [str(w) for w in prim_ws[1:]])
    ax.set_xlabel("jitter width w (log2)")
    ax.set_ylabel("A(w) = row-129 replacement delta (e140)")
    rho = bars["invariance_causal"]["spearman_rho"]
    ax.set_title(f"(i) ADDRESS KEY vs WIDTH — Spearman rho {rho:+.3f} "
                 f"(bar <= {bars['invariance_causal']['rho_bar']})",
                 fontsize=10)
    ax.legend(fontsize=7)

    # (0,1) THE NR(w) PANEL — route birth
    ax = axes[0, 1]
    onb = bars["seed_coverage"]["onset_bar"]
    ax.axhline(onb, color="darkorange", ls="--", lw=1.5,
               label=f"onset bar 2x install baseline = {onb:.4f}")
    ax.axhline(0, color="k", lw=0.6)
    ax.plot(xw(prim_ws), [NR[t] for t in prim_tags], "o-", ms=8, lw=1.8,
            color="steelblue", label="NR(w) @ g-12 (primary)")
    ax.plot(xw(prim_ws), [nr_detail[t][12]["NR_zero_drop"]
                          for t in prim_tags], "s--", ms=5, lw=1.1,
            color="cadetblue", alpha=0.8, label="NR(w) @ g+12 (robustness)")
    for t, w in zip(prim_tags, prim_ws):
        ax.annotate(f"{NR[t]:.3f}", xy=(w, NR[t]), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=7.5)
    w_star = bars["invariance_causal"]["crossing_w_star"]
    if w_star is not None:
        ax.axvline(w_star, color="crimson", ls="--", lw=1.4, alpha=0.6,
                   label=f"w* = {w_star} (A crosses 0)")
    w_on = bars["invariance_causal"]["first_onset_w"]
    if w_on is not None:
        ax.axvspan(max(w_on / 2.0, 0.4), w_on * 2, color="steelblue",
                   alpha=0.08)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0] + list(prim_ws[1:]))
    ax.set_xticklabels(["0(L)"] + [str(w) for w in prim_ws[1:]])
    ax.set_xlabel("jitter width w (log2)")
    ax.set_ylabel("NR(w) = d_r0 drop at novel geometry (e141)")
    ax.set_title("(ii) ROUTE BIRTH vs WIDTH — co-onset with w* is the "
                 "causal joint", fontsize=10)
    ax.legend(fontsize=7)

    # (0,2) A and NR OVERLAID (the joint, twin-axis)
    ax = axes[0, 2]
    ax.axhline(0, color="k", lw=0.7)
    ax.plot(xw(prim_ws), [A[t] for t in prim_tags], "o-", ms=7, lw=1.6,
            color="crimson", label="A(w) (left)")
    ax.set_ylabel("A(w)", color="crimson")
    ax.tick_params(axis="y", labelcolor="crimson")
    ax2 = ax.twinx()
    ax2.plot(xw(prim_ws), [NR[t] for t in prim_tags], "s-", ms=7, lw=1.6,
             color="steelblue", label="NR(w) (right)")
    ax2.axhline(onb, color="darkorange", ls="--", lw=1.2)
    ax2.set_ylabel("NR(w)", color="steelblue")
    ax2.tick_params(axis="y", labelcolor="steelblue")
    if w_star is not None:
        for a_ in (ax, ax2):
            a_.axvline(w_star, color="k", ls=":", lw=1.2, alpha=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0] + list(prim_ws[1:]))
    ax.set_xticklabels(["0(L)"] + [str(w) for w in prim_ws[1:]])
    ax.set_xlabel("jitter width w (log2)")
    co = bars["invariance_causal"]["co_onset_within_one_bin"]
    ax.set_title(f"(iii) THE JOINT: key death vs route birth — co-onset "
                 f"{'HELD' if co else 'did not hold'}"
                 + (f" (w*={w_star}, w_on={w_on})" if w_star is not None
                    and w_on is not None else ""), fontsize=9.5)

    # (1,0) tertiary: D-all survival + base g0 calibration
    ax = axes[1, 0]
    xs = np.arange(len(prim_tags))
    ax.bar(xs - 0.19, [dall[t]["d_all"] for t in prim_tags], 0.36,
           color="mediumseagreen", edgecolor="k", lw=0.4, label="D-all g0")
    ax.bar(xs + 0.19, [base_calib[t]["g0_none"] for t in prim_tags], 0.36,
           color="lightgray", edgecolor="k", lw=0.4, label="base g0 (none)")
    for x, t in zip(xs, prim_tags):
        ax.text(x - 0.19, dall[t]["d_all"] + 0.01, f"{dall[t]['d_all']:.2f}",
                ha="center", fontsize=6.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{t}\n(w={w})" for t, w in zip(prim_tags, prim_ws)],
                       fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_title("(iv) TERTIARY: D-all survival at g0 (calibration: base)",
                 fontsize=10)
    ax.legend(fontsize=7.5)

    # (1,1) address-band census spectra (e140 texture)
    ax = axes[1, 1]
    cmap = plt.get_cmap("viridis")
    for k, t in enumerate(prim_tags):
        cen = census[t]
        rows_r = [int(r) for r in cen["rows"] if 118 <= int(r) <= 140]
        ys = [cen["rows"][str(r)]["strength"] for r in rows_r]
        ax.plot(rows_r, ys, "o-", ms=2.5, lw=1.0, alpha=0.85,
                color=cmap(k / max(len(prim_tags) - 1, 1)), label=f"{t}")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(129, color="crimson", ls=":", lw=1.2, alpha=0.8)
    ax.text(129.3, ax.get_ylim()[1] * 0.92, "row 129 (the install address)",
            fontsize=7.5, color="crimson")
    ax.set_xlabel("wpe row (band 118-140)")
    ax.set_ylabel("strength = min(mean-drop, zero-drop)")
    ax.set_title("(v) address-band census spectra per width (e140 texture)",
                 fontsize=10)
    ax.legend(fontsize=6.2, ncol=2)

    # (1,2) verdict panel
    ax = axes[1, 2]
    ax.axis("off")
    vlines = [
        "E147 — THE WIDTH LADDER (T084 pre-registration, verbatim bars):",
        f"  A(w):  " + "  ".join(f"{t} {A[t]:+.3f}" for t in prim_tags),
        f"  NR(w): " + "  ".join(f"{t} {NR[t]:+.3f}" for t in prim_tags)
        + f"  [root {NR['root']:+.4f}]",
        f"  Spearman(A, w) = {bars['invariance_causal']['spearman_rho']:+.3f} "
        f"(bar <= -0.8) | sensitivity: "
        + " ".join(f"{k} {v:+.2f}" for k, v in
                   bars["sensitivity_ladders_report_only"].items()),
        f"  w* (A crosses 0) = {w_star} | first NR onset = {w_on} "
        f"(set {bars['invariance_causal']['onset_set']})",
        f"  co-onset within one bin: "
        f"{bars['invariance_causal']['co_onset_within_one_bin']} | "
        f"g+12 agrees: {bars['invariance_causal']['g+12_agrees_with_onset']}",
        f"  DEAD-AGAIN: flat {bars['dead_again']['flat_clause']} | "
        f"early-onset {bars['dead_again']['early_onset_clause']}",
        f"  SEED-COVERAGE: {bars['seed_coverage']['fired']} "
        f"(NR8 {bars['seed_coverage']['NR_at_8']}, max "
        f"{bars['seed_coverage']['NR_ladder_max']:.4f})",
        f"  T084 free cell (far d_r0@g-12): "
        f"{bars['free_cell_T084_far']['verdict']} "
        f"(x{bars['free_cell_T084_far']['ratio']:.3f})",
        "",
        f"VERDICT: {verdict}",
    ] + [f"  {clause[i:i+66]}" for i in range(0, len(clause), 66)]
    for i, tx in enumerate(vlines):
        ax.text(0.02, 0.97 - i * 0.042, tx, fontsize=7.2, va="top",
                family="monospace",
                bbox=dict(facecolor="lightyellow", alpha=0.9, edgecolor="gray")
                if tx.startswith("VERDICT") else None)

    fig.suptitle(f"E147 — the width ladder: address-key death A(w) vs route "
                 f"birth NR(w) (root e048_repro) -> {verdict}", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
