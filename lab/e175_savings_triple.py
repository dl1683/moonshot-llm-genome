"""E175 — THE SAVINGS TRIPLE (R50's expanded design; QUEUE.md row e175,
DISPATCHED ~15:50Z; bars are the QUEUE row VERBATIM, written before compute).

WHY: the lab now holds two dissected states of the SAME consolidated fact,
each with a known lesion class:
  * the KILLED state — e160's N2 head-ablation (mean-replace {L1H0,L0H0}):
    71% readout kill at CE +0.245, with e164 showing the machinery behind
    the kill INTACT (row-0 full, MLP headroom, all heads load-bearing) —
    an ACCESS lesion, storage suspected intact (T098's bound: the kill was
    71%, not 100% — the residual readout is 29% alive).
  * the WASHED state — e176's plain-corpus freeze of the consolidated root:
    the whole fact gone in two steps (g-12 0.916 -> 0.0011 by +300), with
    e178 showing it only ~40-45% class-recoverable (storage HALF-degraded;
    T106: the brake scar survives in the MLP+LN stream at near-full
    strength — a scar that outlasts the memory it scarred).
THE TEST (the classical Ebbinghaus savings paradigm, applied to a
dissected network): identical short locked re-teaches on both states —
do they re-learn FASTER than naive (savings — a residue exists), SLOWER
(negative savings — the surviving brake scar actively interferes), or
equal (no residue — the archive truly empty)?

REGISTERED PREDICTION (QUEUE.md e175 row, VERBATIM — no bar shopping):
  "Bars: FAST-RECOVERY = killed <=30 steps to 0.78 (thin access lesion —
  e164's strong reading licensed); POSITIVE-SAVINGS = washed re-learns
  faster than naive (a residue exists); NEGATIVE-SAVINGS = slower (T106's
  surviving brake scar actively interferes — the delicious branch);
  NO-SAVINGS = equal (the archive truly empty)"

OPERATIONALIZATIONS (frozen here before compute):
  * RE-TEACH = the e119-L locked replay VERBATIM (e109 arm-b / e143 / e151
    recipe) at the HOME site (offset j=0: ZEPHYRA at x-cols 130..136,
    name-onset read row 129, 7-target name-only mask, zero position
    variance): batch 32 = 16 install windows + 16 anchors (8 paired + 8
    random), e043 token-level union CE, AdamW (0.9,0.95) wd 0.1 constant
    lr 1e-3 clip 1.0, 300 steps, seed 10902, snapshots at {10,30,100,300}.
    All three arms see the IDENTICAL draw sequence (evals consume no RNG).
  * ARMS: (i) killed = e131_consolidated_e113 + the PERSISTENT N2/mean
    clamp (forward pre-hooks on c_proj inputs; the L1H0/L0H0 slices
    replaced by their CE_R-bank means from the INTACT base net, e160's
    convention: fixed vectors, never recomputed — here also never
    recomputed under training). The clamp is active during TRAINING and
    every eval: recovery must route AROUND the lesion, exactly what
    "re-learn after an access lesion" means. Step-0 gate must reproduce
    e160's stored N2/mean cells bit-exact (g0 0.22963003814220428).
    (ii) washed = runs/checkpoints/e176_root_freeze.pt (gated vs e176's
    stored +300 cells). (iii) naive = runs/checkpoints/e001.pt (the
    pre-install base of the whole e048/e109/e113/e131 lineage — e048's
    own BASE; verified fact-free at g0 (p(Z) 1.34e-05, e048's stored
    baseline R1i acc 0.0976 = chance).
  * "~0.78" = the consolidated root's g0 (0.7850); the bar reads g0
    (install-60 battery, the e133/e150/e160 drop-table convention).
    steps-to-0.78 = the FIRST checkpoint in {0,10,30,100,300} with
    g0 >= 0.78; resolution is the checkpoint grid (stated, not shopped).
  * FAST-RECOVERY fires iff the KILLED arm's steps-to-0.78 <= 30.
  * POSITIVE-/NEGATIVE-/NO-SAVINGS compare the WASHED arm to the NAIVE
    arm on steps-to-0.78: washed < naive -> POSITIVE; washed > naive ->
    NEGATIVE; equal -> NO-SAVINGS. If either side is censored (>300),
    no savings bar fires: TEXTURE-CENSORED with the full curves (mean g0
    over the 4 checkpoints, g0@300, held30 co-dials) — reported, never
    shopped. The bars are NOT mutually exclusive across the two states
    (FAST-RECOVERY can co-fire with any savings verdict).
  * DIALS per checkpoint (step 0 included; e176's measure() convention
    verbatim): install-60 batteries g-12/g0/g+12 + held30 counterparts;
    CE_R (e065 bank, seed 26502); the 183-site functional read + span
    census over e152's locked j=54 measurement pool (instrument only);
    the old-band census (A129 brake, row-0 sink strength); D-all g0;
    D-183 co-report. Killed-arm dials run UNDER the clamp.

NETS (on disk, gated; CPU-ONLY — CUDA_VISIBLE_DEVICES=-1, threads <= 4,
e174/e177 share the CPU; staggered, no busy-waiting):
  * runs/checkpoints/e131_consolidated_e113.pt (root: the killed state's
    base; naive-retrain lineage anchor)
  * runs/checkpoints/e176_root_freeze.pt (washed)
  * runs/checkpoints/e001.pt (naive pre-install base)
  saved: runs/checkpoints/e175_killed_reteach.pt (+ s10/s30/s100),
  e175_washed_reteach.pt (+ s10/s30/s100), e175_naive_reteach.pt.

COMPUTE ENVELOPE: CPU-only, torch threads 4, three short trainings (300
steps each; CPU cap 1800 s per training — e161/e176 precedent for the
dispatch's <=180 s GPU cap), cooldown(75) between trainings, 10 s
staggers between phases, no busy-waiting. Outputs:
runs/e175/{metrics.json, savings_triple.png}.

Run:  cd lab && python e175_savings_triple.py    (E175_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # CPU-ONLY (e174/e177 share)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                           # noqa: E402

torch.set_num_threads(4)                              # modest (shared CPU)

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
common.DEVICE = "cpu"
from common import Cfg, CharCorpus, TinyGPT, cooldown, run_dir, save_json  # noqa: E402
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E175_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
ROOT_CK = "e131_consolidated_e113.pt"     # killed-state base
WASH_CK = "e176_root_freeze.pt"           # washed state
NAIVE_CK = "e001.pt"                      # pre-install base (e048's BASE)

# ---- the N2 kill (e160's bar-winning cell, mean-replace mode) ----------------
KILL_HEADS = ((1, 0), (0, 0))     # {L1H0, L0H0} = e160's N2:top2-noL0H3
KILL_TAG = "N2mean:L1H0+L0H0"

# ---- re-teach placement: HOME site (offset j=0, e119-L/e109 arm-b) -----------
RETEACH_J = 0                     # name x-cols 130..136, onset read row 129
HOME_ADDR_ROW = PRE - 1           # 129: the x-position predicting the first Z
HOME_Z_XCOL = PRE + RETEACH_J     # 130
NAME_ROWS = tuple(range(HOME_ADDR_ROW, HOME_ADDR_ROW + len(NAME)))  # 129..135

# ---- site-read instrument (e152's locked j=54 pool; measurement only) --------
SITE_J = 54                       # e152/e161/e176 measurement offset
SITE_ADDR_ROW = 183
SITE_Z_XCOL = PRE + SITE_J        # 184
SITE_CONT = BLOCK - PRE - SITE_J - len(NAME)     # 65
SITE_ROWS = tuple(range(183, 190))
SHARED_CTR = (60, 100, 150, 160, 170, 200, 220)
ROWS_183 = (0,) + (181, 182) + SITE_ROWS + SHARED_CTR           # 17 rows
ROWS_OLD = (0, 60, 100) + tuple(range(121, 130))                # 12 rows
D_ALL = (121, 125, 129, 133, 137)               # e113's fixed set
GEOS = (-12, 0, 12)
if SMOKE:
    ROWS_183 = (0, 1, 182) + (183, 185, 189) + (60, 100)
    ROWS_OLD = (0, 1, 60, 121, 125, 129)

# ---- fine-tune envelope (e109 arm-b / e119-L / e143 / e151 verbatim) ---------
FT_LR = 1e-3
FT_STEPS = 300 if not SMOKE else 6
CKPT_STEPS = [10, 30, 100, 300] if not SMOKE else [2, 4, 6]
CKPT_SET = set(CKPT_STEPS)
FT_TIME_CAP = 1800.0              # CPU cap (e161/e176 precedent)
NAME_BS, ANCH_BS = 16, 16
RETEACH_SEED = 10902              # e119's L_SEED / e143/e151 locked seed
COOLDOWN_S = 75.0 if not SMOKE else 5.0    # dispatch envelope 60-120 s between trainings
STAGGER_S = 10.0 if not SMOKE else 2.0     # stagger between eval phases

# ---- seeds / gates ------------------------------------------------------------
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05

# root (e131 consolidated) — e151/e160/e176 bit anchors
G_ROOT_PZ = 0.7850371599197388
G_ROOT_CE = 1.663516640663147
G_ROOT_GM12 = 0.9155886173248291

# the killed state — e160's stored N2/mean cells (the reproduction gate)
G_KILL_PZ = 0.22963003814220428
G_KILL_GM12 = 0.05206625908613205
G_KILL_CE = 1.9085687398910522

# the washed state — e176's stored +300 cells (the load gate)
G_WASH_PZ = 0.0012388104805722833
G_WASH_GM12 = 0.0011210207594558597
G_WASH_GP12 = 0.0017126891762018204
G_WASH_CE = 1.6334049701690674

# naive base — fact-free gate (measured pre-run: p(Z) 1.34e-05; e048's stored
# baseline R1i acc 0.0976 = chance) — no bit reference exists for this battery
G_NAIVE_PZ_MAX = 0.15

# ---- registered bars (numeric, frozen) ----------------------------------------
RECOVERY_BAR = 0.78               # "~0.78" = the consolidated root's g0
FAST_STEPS = 30                   # FAST-RECOVERY: killed <= 30 steps to 0.78

REGISTERED_PREDICTION = {
    "queue_row_verbatim": (
        "identical short locked re-teach (10/30/100/300 steps) on (i) the "
        "N2-kILLED net (access lesion, storage intact per e164) and (ii) the "
        "e176-WASHED net (storage half-degraded per e178). Bars: "
        "FAST-RECOVERY = killed <=30 steps to 0.78 (thin access lesion — "
        "e164's strong reading licensed); POSITIVE-SAVINGS = washed re-learns "
        "faster than naive (a residue exists); NEGATIVE-SAVINGS = slower "
        "(T106's surviving brake scar actively interferes — the delicious "
        "branch); NO-SAVINGS = equal (the archive truly empty)"),
    "operationalizations": (
        "re-teach = e119-L locked replay verbatim at the HOME site (offset 0, "
        "7-target name-only mask): batch 32 = 16 install + 16 anchors (8 "
        "paired + 8 random), e043 union CE, AdamW (0.9,0.95) wd 0.1 lr 1e-3 "
        "constant clip 1.0, seed 10902, ckpts {10,30,100,300}. Killed arm "
        "trains AND evaluates under the persistent e160 N2/mean clamp (fixed "
        "CE_R-bank vectors from the intact base, never recomputed). '~0.78' "
        "read at g0 (install-60). steps-to-0.78 = first grid checkpoint with "
        "g0 >= 0.78. FAST-RECOVERY iff killed steps-to-0.78 <= 30. Savings "
        "bars compare washed vs naive steps-to-0.78: < POSITIVE, > NEGATIVE, "
        "== NO-SAVINGS; any censored side (never crosses within 300) -> no "
        "bar fires, TEXTURE-CENSORED with the curves. Bars can co-fire "
        "across the two states."),
    "no_bar_shopping": "No bar shopping; texture => TEXTURE with the curves.",
}

trims: list[str] = []
deviations: list[str] = [
    "CPU-ONLY by dispatch (CUDA_VISIBLE_DEVICES=-1 forced before torch; "
    "e174/e177 share the CPU): torch threads 4, cooldown(90) between the "
    "trainings, 25 s staggers, no busy-waiting.",
    "The dispatch's '<=180 s cap' is the lab's GPU single-training cap; the "
    "CPU path runs under a 1800 s per-training cap (e161/e176 precedent "
    "verbatim).",
    "Nets are the 2.7M e131/e176/e001 line (the dispatch's '<=1M family' "
    "note is an envelope statement; e143/e151/e152/e161/e176 precedent — "
    "every gate reference and lineage number of this cell lives on the "
    "2.7M line).",
    "The naive reference is RUN, not cited: e109's arm-b and e119-L stored "
    "trajectories start from the INSTALL net (g0 0.5563 — already "
    "half-learned, not naive), so the dispatch's 'if cheap' clause is "
    "taken: a third identical arm on e001.pt (the pre-install base of the "
    "entire lineage, e048's own BASE, measured fact-free). The stored "
    "trajectories are cited as context in metrics.reference_stored.",
    "The killed state is a PERSISTENT INPUT-SIDE CLAMP (forward pre-hooks), "
    "not a weight edit: the lesion stays during training and every eval — "
    "recovery must route around it. Consequence recorded: the clamped "
    "heads' qkv receive zero gradient through the replaced slice (their "
    "weights only decay via AdamW's decoupled wd), while c_proj and all "
    "downstream weights train normally.",
    "steps-to-0.78 resolution is the checkpoint grid {0,10,30,100,300}: a "
    "crossing between two checkpoints is only bounded, not interpolated "
    "(stated; the FAST bar uses <=30 which the grid resolves exactly).",
    "held30 batteries co-reported per checkpoint (e152's held30 "
    "construction; a CO-DIAL, not a bar clause).",
    "DIAL DENSITY: the two bar-carrying arms (killed, washed) get FULL "
    "dials at every checkpoint {0,10,30,100,300}; the naive control gets "
    "full at {0, final} and LIGHT (batteries + CE + site read + deletion "
    "table; censuses skipped) at {10,30,100} — the savings bars read g0 "
    "only, and CPU envelope is shared (e174/e177). Re-teach protocol is "
    "IDENTICAL across all three arms regardless.",
    "CENSUS ROWS trimmed vs e176's tables (ROWS_183 17 rows, ROWS_OLD 12): "
    "the dropped rows (1-6, 118-120, 183-lead controls beyond 181/182) were "
    "content-null in every stored census of this lineage; A(129), row-0, "
    "the 121-129 band and the 2x-control convention are unchanged.",
    "Thermal envelope: cooldown 75 s between trainings, 10 s staggers "
    "between eval phases (both inside the dispatch's 60-120 s / stagger "
    "envelope; no busy-waiting).",
    "Smoke mode trims: 6-step re-teach, ckpts {2,4,6}, reduced census rows, "
    "nothing adjudicated.",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: load_cpu/evl_load/battery_cell/ce_fixed_cpu/val_windows/deleted_wpe
# are lab/e143/e151 VERBATIM (e065/e068/e109/e113/e116/e119 lineage);
# read_fact_at/row_census are lab/e151 VERBATIM; HeadReplace/head_mean_vec are
# lab/e160 VERBATIM (the N2 kill instrument). Copied, not imported, to keep
# this rig self-contained and CPU-forced.

def evl_load(sd: dict) -> TinyGPT:
    m = TinyGPT(Cfg())
    m.load_state_dict(sd)
    m.eval()
    return m


def load_cpu(path) -> TinyGPT:
    m = TinyGPT(Cfg())
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def battery_cell(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> dict:
    """e068/e113/e120 battery on CPU: p(Z) at the last position."""
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
    """e131's read_fact_position VERBATIM ARITHMETIC, geometry parameterized:
    p(true name char) at positions addr_row..addr_row+6 over the pool windows."""
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


def head_mean_vec(net: TinyGPT, layer: int, head: int, bank_x, bs=30):
    """e160 verbatim: mean of the head's 32-dim slice of c_proj's input over
    the CE_R bank (corpus windows only)."""
    outs = []
    hd = net.cfg.n_embd // net.cfg.n_head

    def pre(m, args):
        x = args[0].detach()
        outs.append(x[..., head * hd:(head + 1) * hd].clone())
        return None
    h = net.h[layer].attn.c_proj.register_forward_pre_hook(pre)
    with torch.no_grad():
        for i in range(0, bank_x.shape[0], bs):
            net(bank_x[i:i + bs])
    h.remove()
    o = torch.cat([t.reshape(-1, t.shape[-1]) for t in outs], 0)
    return o.mean(0)


class HeadReplace:
    """e160 verbatim: replace head slices of c_proj's input with fixed vectors
    (mean-replace). Works under autograd (input-side persistent clamp): grads
    flow to downstream weights; the replaced head's own qkv gets none."""

    def __init__(self, net: TinyGPT, replace: dict):
        # replace: {(layer, head): vector}
        self.net = net
        self.replace = replace
        self.handles = []
        self.hd = net.cfg.n_embd // net.cfg.n_head

    def __enter__(self):
        by_layer: dict[int, list] = {}
        for (l, hd_), vec in self.replace.items():
            by_layer.setdefault(l, []).append((hd_, vec))
        for l, items in by_layer.items():
            def pre(m, args, items=items):
                x = args[0].clone()
                for hd_, vec in items:
                    x[..., hd_ * self.hd:(hd_ + 1) * self.hd] = vec
                return (x,)
            self.handles.append(
                self.net.h[l].attn.c_proj.register_forward_pre_hook(pre))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []


# ------------------------------------------------------------------ fine-tune

def reteach_arm(tag: str, net0: TinyGPT, pool_x: torch.Tensor,
                pool_mask: torch.Tensor, anchor: torch.Tensor,
                train_ids: torch.Tensor, itos, r_eval_xy, gm12_ids,
                g0_ids, zid: int, seed: int, clamp=None):
    """THE LOCKED RE-TEACH (e109 arm-b / e119-L / e143 verbatim; CPU; the
    killed arm passes clamp={ (l,h): vec } — the persistent N2/mean lesion,
    active in training and in the light in-loop evals). Per step: ix =
    randint(16) pool draws, aj = randint(8) paired anchors, rj = randint(8)
    random corpus offsets; union CE (7-target name mask on pool windows +
    full-token CE on anchors); AdamW (0.9,0.95) wd 0.1 lr 1e-3 constant,
    clip 1.0. Snapshots + light CPU evals at the checkpoint steps (no RNG
    consumed — the draw sequence is e119-L's exactly)."""
    net = copy.deepcopy(net0)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    n_pool = pool_x.shape[0]
    n_anc = anchor.shape[0]
    traj, sds, t_start = [], {}, time.time()
    step = 0
    zeph_checks = 0
    evl = copy.deepcopy(net0)          # CPU eval twin
    cm = None
    if clamp is not None:
        cm = HeadReplace(net, clamp)
        cm.__enter__()
    try:
        for step in range(1, FT_STEPS + 1):
            ix = torch.randint(n_pool, (NAME_BS,), generator=gen)
            aj = torch.randint(n_anc, (ANCH_BS // 2,), generator=gen)
            rj = torch.randint(len(train_ids) - BLOCK - 1, (ANCH_BS // 2,),
                               generator=gen)
            nw = pool_x[ix]
            rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj])
            for w in rnd:              # name-free VERIFY (no-op by construction)
                txt = "".join(itos[int(c)] for c in w[:64]) + \
                      "".join(itos[int(c)] for c in w[192:])
                if "ZEPH" in txt:
                    zeph_checks += 1
            anc = torch.cat([anchor[aj], rnd], 0)
            x = torch.cat([nw[:, :-1], anc[:, :-1]], 0)
            y = torch.cat([nw[:, 1:], anc[:, 1:]], 0)
            m = torch.zeros(NAME_BS + ANCH_BS, x.shape[1], dtype=torch.bool)
            m[:NAME_BS] = pool_mask[ix]
            logits, _ = net(x)
            nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                  y.reshape(-1),
                                  reduction="none").view(x.shape[0], x.shape[1])
            nm = nll[:NAME_BS][m[:NAME_BS]]
            am = nll[NAME_BS:]
            loss = (nm.sum() + am.sum()) / (nm.numel() + am.numel())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if step in CKPT_SET or step % 50 == 0:
                log(f"  [{tag}] s{step:4d} union CE {float(loss.item()):.4f} "
                    f"({time.time() - t_start:.0f}s)")
            if step in CKPT_SET:
                sd_cpu = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
                sds[step] = sd_cpu
                evl.load_state_dict(sd_cpu)
                evl.eval()
                if clamp is not None:
                    with HeadReplace(evl, clamp):
                        gz = battery_cell(evl, gm12_ids, zid)
                        gz0 = battery_cell(evl, g0_ids, zid)
                        ce_r = ce_fixed_cpu(evl, *r_eval_xy)
                else:
                    gz = battery_cell(evl, gm12_ids, zid)
                    gz0 = battery_cell(evl, g0_ids, zid)
                    ce_r = ce_fixed_cpu(evl, *r_eval_xy)
                traj.append({"step": step, "g_m12_mean_pz": gz["mean_pz"],
                             "g0_mean_pz": gz0["mean_pz"],
                             "frac_argmax_z": gz0["frac_argmax_z"],
                             "ce_r": ce_r,
                             "elapsed_s": round(time.time() - t_start, 1)})
                log(f"  [{tag}] CKPT +{step:4d} g0 {gz0['mean_pz']:.4f} "
                    f"g-12 {gz['mean_pz']:.4f} CE_R {ce_r:.4f} "
                    f"({traj[-1]['elapsed_s']:.0f}s)")
            if (time.time() - t_start) > FT_TIME_CAP:
                log(f"  [{tag}] time cap {FT_TIME_CAP:.0f}s at s{step}")
                trims.append(f"{tag}: time cap at step {step}")
                break
    finally:
        if cm is not None:
            cm.__exit__(None, None, None)
    net.eval()
    return {"sds": sds, "traj": traj, "steps_ran": step, "seed": seed,
            "zeph_violations": zeph_checks}


CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "e175", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name}")


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e175_smoke" if SMOKE else "e175")
    log(f"E175 THE SAVINGS TRIPLE (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-only, torch threads {torch.get_num_threads()}, "
        f"cuda available = {torch.cuda.is_available()}")

    # ---------------- protocol rebuild (e143/e151/e176 verbatim)
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

    # ---------------- the HOME-site locked re-teach pool (e119-L / e109 arm-b)
    wins = []
    for p, h in install_occ:
        pre = train_ids[p - PRE - RETEACH_J: p]
        post = train_ids[p + len(h): p + len(h) + POST_CAP - RETEACH_J]
        if len(pre) != PRE + RETEACH_J or len(post) != POST_CAP - RETEACH_J:
            raise RuntimeError(f"re-teach window short at p={p}")
        w = torch.cat([pre, name_ids, post])
        if len(w) != BLOCK:
            raise RuntimeError(f"re-teach window len {len(w)} != {BLOCK}")
        wins.append(w)
    pool_x = torch.stack(wins)
    pool_mask = torch.zeros(len(wins), BLOCK - 1, dtype=torch.bool)
    pool_mask[:, HOME_ADDR_ROW: HOME_ADDR_ROW + L] = True
    G_POOL = {
        "shape": list(pool_x.shape),
        "name_xcols": [HOME_Z_XCOL, HOME_Z_XCOL + L - 1],
        "read_rows": [HOME_ADDR_ROW, HOME_ADDR_ROW + L - 1],
        "all_windows_name_in_place": bool(
            all(torch.equal(w[HOME_Z_XCOL: HOME_Z_XCOL + L], name_ids)
                for w in pool_x)),
        "mask_targets_per_window": int(pool_mask[0].sum()),
        "zero_variance": True,
        "note": "TRAINING pool (the e119-L HOME-site locked replay)",
    }
    G_POOL["pass"] = bool(G_POOL["all_windows_name_in_place"]
                          and G_POOL["mask_targets_per_window"] == L)
    assert G_POOL["pass"], f"re-teach pool gate FAILED: {G_POOL}"
    log(f"re-teach pool: {tuple(pool_x.shape)} — ZEPHYRA locked at x-cols "
        f"{HOME_Z_XCOL}..{HOME_Z_XCOL + L - 1} (onset read row "
        f"{HOME_ADDR_ROW}), {L}-target name mask, zero position variance")

    # anchor bank (e065/e109/e143 verbatim): first 16 install-position
    # ORIGINAL host windows (incumbent continuations, no ZEPHYRA)
    anchor = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                          for p, _ in install_occ[:16]])
    anchor_zeph = sum(1 for w in anchor if "ZEPH" in corpus.decode(w))
    G_ANCHFREE = {"anchor_zeph_windows": anchor_zeph,
                  "pass": bool(anchor_zeph == 0)}
    assert G_ANCHFREE["pass"], "anchor bank contains ZEPH"

    # ---------------- site-read instrument (e152's locked j=54 pool)
    site_wins = []
    for p, h in install_occ:
        pre = train_ids[p - PRE - SITE_J: p]
        post = train_ids[p + len(h): p + len(h) + SITE_CONT]
        if len(pre) != PRE + SITE_J or len(post) != SITE_CONT:
            raise RuntimeError(f"site pool window short at p={p}")
        site_wins.append(torch.cat([pre, name_ids, post]))
    site_pool = torch.stack(site_wins)
    G_SITE = {"shape": list(site_pool.shape),
              "name_xcols": [SITE_Z_XCOL, SITE_Z_XCOL + L - 1],
              "all_windows_name_in_place": bool(
                  all(torch.equal(w[SITE_Z_XCOL: SITE_Z_XCOL + L], name_ids)
                      for w in site_pool)),
              "note": "MEASUREMENT instrument only (e152's locked j=54 pool); "
                      "NO window from this pool enters the re-teach training"}
    G_SITE["pass"] = G_SITE["all_windows_name_in_place"]
    assert G_SITE["pass"], f"site pool gate FAILED: {G_SITE}"

    # ---------------- batteries (e119 construction) + CE bank
    bat_ids, held_ids = {}, {}
    for j in GEOS:
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
        hs = [train_text[p - PRE - j: p] for p, _ in held_occ]
        held_ids[j] = torch.stack([corpus.encode(c) for c in hs])
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    gm12_ids, g0_ids = bat_ids[-12], bat_ids[0]
    log(f"batteries: install60/held30 x geos {list(GEOS)}; CE_R bank "
        f"{tuple(r_eval_x.shape)} (seed {R_EVAL_SEED})")

    # =====================================================================
    # PHASE 0: load + gate the three bases; build the kill
    # =====================================================================
    log("--- PHASE 0: bases + gates + the N2/mean clamp ---")
    net_root = load_cpu(CKPT_DIR / ROOT_CK)
    sd_root = {k: v.clone() for k, v in net_root.state_dict().items()}

    ev = copy.deepcopy(net_root)
    bz_r = battery_cell(ev, g0_ids, zid)
    ce_r0 = ce_fixed_cpu(ev, *r_eval_xy)
    bz_r12 = battery_cell(ev, gm12_ids, zid)
    G_ROOT = {"battery_pz": bz_r["mean_pz"], "ref_pz": G_ROOT_PZ,
              "battery_gm12": bz_r12["mean_pz"], "ref_gm12": G_ROOT_GM12,
              "ce_r": ce_r0, "ref_ce": G_ROOT_CE,
              "bit_reproducible": bool(
                  abs(bz_r["mean_pz"] - G_ROOT_PZ) < G_BIT_TOL
                  and abs(ce_r0 - G_ROOT_CE) < G_BIT_TOL),
              "pass": bool(abs(bz_r["mean_pz"] - G_ROOT_PZ) < G_FALLBACK_TOL
                           and abs(ce_r0 - G_ROOT_CE) < G_FALLBACK_TOL
                           and abs(bz_r12["mean_pz"] - G_ROOT_GM12)
                           < G_FALLBACK_TOL)}
    log(f"G_ROOT: p(Z) {bz_r['mean_pz']:.10f} (ref {G_ROOT_PZ:.10f}) "
        f"g-12 {bz_r12['mean_pz']:.10f} CE_R {ce_r0:.6f}: "
        f"{'PASS' if G_ROOT['pass'] else 'FAIL'}")
    if not G_ROOT["pass"]:
        raise RuntimeError("consolidated root failed its gate")

    # the KILL: fixed mean-replace vectors from the INTACT base over the CE_R
    # bank (e160's N2/mean convention verbatim)
    clamp = {}
    for (l, h) in KILL_HEADS:
        clamp[(l, h)] = head_mean_vec(net_root, l, h, r_eval_x)
    with HeadReplace(net_root, clamp):
        kz0 = battery_cell(net_root, g0_ids, zid)
        kz12 = battery_cell(net_root, gm12_ids, zid)
        kce = ce_fixed_cpu(net_root, *r_eval_xy)
    G_KILL = {"tag": KILL_TAG, "heads": [f"L{l}H{h}" for l, h in KILL_HEADS],
              "mode": "mean-replace (e160 N2/mean convention)",
              "battery_pz": kz0["mean_pz"], "ref_pz": G_KILL_PZ,
              "battery_gm12": kz12["mean_pz"], "ref_gm12": G_KILL_GM12,
              "ce_r": kce, "ref_ce": G_KILL_CE,
              "drop_g0_pct": 100.0 * (1.0 - kz0["mean_pz"] / G_ROOT_PZ),
              "bit_reproducible": bool(
                  abs(kz0["mean_pz"] - G_KILL_PZ) < G_BIT_TOL
                  and abs(kz12["mean_pz"] - G_KILL_GM12) < G_BIT_TOL
                  and abs(kce - G_KILL_CE) < G_BIT_TOL),
              "pass": bool(abs(kz0["mean_pz"] - G_KILL_PZ) < G_FALLBACK_TOL
                           and abs(kz12["mean_pz"] - G_KILL_GM12)
                           < G_FALLBACK_TOL
                           and abs(kce - G_KILL_CE) < G_FALLBACK_TOL)}
    log(f"G_KILL ({KILL_TAG}): g0 {kz0['mean_pz']:.10f} (ref {G_KILL_PZ:.10f}) "
        f"g-12 {kz12['mean_pz']:.10f} CE {kce:.6f} "
        f"[drop {G_KILL['drop_g0_pct']:.1f}%]: "
        f"{'PASS' if G_KILL['pass'] else 'FAIL'}"
        + ("  [bit-exact]" if G_KILL["bit_reproducible"] else ""))
    if not G_KILL["pass"]:
        raise RuntimeError("in-memory N2/mean kill failed to reproduce e160")

    net_wash = load_cpu(CKPT_DIR / WASH_CK)
    sd_wash = {k: v.clone() for k, v in net_wash.state_dict().items()}
    ev = copy.deepcopy(net_wash)
    wz = {j: battery_cell(ev, bat_ids[j], zid) for j in GEOS}
    wce = ce_fixed_cpu(ev, *r_eval_xy)
    G_WASH = {"base_g0": wz[0]["mean_pz"], "ref_g0": G_WASH_PZ,
              "base_gm12": wz[-12]["mean_pz"], "ref_gm12": G_WASH_GM12,
              "base_gp12": wz[12]["mean_pz"], "ref_gp12": G_WASH_GP12,
              "ce_r": wce, "ref_ce": G_WASH_CE,
              "bit_reproducible": bool(
                  abs(wz[0]["mean_pz"] - G_WASH_PZ) < G_BIT_TOL
                  and abs(wz[-12]["mean_pz"] - G_WASH_GM12) < G_BIT_TOL
                  and abs(wz[12]["mean_pz"] - G_WASH_GP12) < G_BIT_TOL
                  and abs(wce - G_WASH_CE) < G_BIT_TOL),
              "pass": bool(abs(wz[0]["mean_pz"] - G_WASH_PZ) < G_FALLBACK_TOL
                           and abs(wz[-12]["mean_pz"] - G_WASH_GM12)
                           < G_FALLBACK_TOL
                           and abs(wce - G_WASH_CE) < G_FALLBACK_TOL)}
    log(f"G_WASH: g0 {wz[0]['mean_pz']:.10f} g-12 {wz[-12]['mean_pz']:.10f} "
        f"g+12 {wz[12]['mean_pz']:.10f} CE {wce:.6f}: "
        f"{'PASS' if G_WASH['pass'] else 'FAIL'}"
        + ("  [bit-exact]" if G_WASH["bit_reproducible"] else ""))
    if not G_WASH["pass"]:
        raise RuntimeError("washed checkpoint failed its gate vs e176 +300 cells")

    net_naive = load_cpu(CKPT_DIR / NAIVE_CK)
    sd_naive = {k: v.clone() for k, v in net_naive.state_dict().items()}
    ev = copy.deepcopy(net_naive)
    nz = battery_cell(ev, g0_ids, zid)
    nce = ce_fixed_cpu(ev, *r_eval_xy)
    G_NAIVE = {"ckpt": f"runs/checkpoints/{NAIVE_CK}",
               "provenance": "e048's BASE (runs/e048/metrics.json baseline: "
                             "R1i name-window acc 0.0976 = chance, NLL 6.67) "
                             "— the pre-install base of the whole "
                             "e043/e048/e109/e113/e131 lineage",
               "battery_pz": nz["mean_pz"],
               "battery_frac_argmax_z": nz["frac_argmax_z"],
               "ce_r": nce, "pz_max_bar": G_NAIVE_PZ_MAX,
               "pass": bool(nz["mean_pz"] < G_NAIVE_PZ_MAX
                            and nz["frac_argmax_z"] <= 0.2)}
    log(f"G_NAIVE: g0 p(Z) {nz['mean_pz']:.2e} argmaxZ "
        f"{nz['frac_argmax_z']:.2f} CE_R {nce:.4f}: "
        f"{'PASS' if G_NAIVE['pass'] else 'FAIL'} (fact-free check)")
    if not G_NAIVE["pass"]:
        raise RuntimeError("naive base is not fact-free")

    # =====================================================================
    # the measurement battery (e176's measure() convention; clamp-aware)
    # =====================================================================
    gates_surg: dict = {}

    def measure(sd: dict, tag: str, use_clamp: bool, full: bool = True) -> dict:
        """e176's measure() convention; full=False skips the two row censuses
        (bar dials — batteries, CE, site read, deletions — always run)."""
        net = evl_load(sd)
        out: dict = {"tag": tag, "full": bool(full)}
        ctx = HeadReplace(net, clamp) if use_clamp else contextlib_null()

        with ctx:
            # (0) base expression + held30 + CE
            out["base"] = {j: battery_cell(net, bat_ids[j], zid) for j in GEOS}
            out["base_held"] = {j: battery_cell(net, held_ids[j], zid)
                                for j in GEOS}
            out["ce_r"] = ce_fixed_cpu(net, *r_eval_xy)
            log(f"[{tag}] base: " + " ".join(f"g{j:+d} "
                                             f"{out['base'][j]['mean_pz']:.4f}"
                                             for j in GEOS)
                + " | held30: " + " ".join(
                    f"g{j:+d} {out['base_held'][j]['mean_pz']:.4f}"
                    for j in GEOS)
                + f" | CE_R {out['ce_r']:.4f}")

            # (i) 183-site functional read (+ span census when full)
            def span_fn(n):
                return read_fact_at(n, site_pool, name_ids, zid,
                                    SITE_ADDR_ROW, SITE_Z_XCOL
                                    )["pname_mean_over7"]

            out["site_read"] = read_fact_at(net, site_pool, name_ids, zid,
                                            SITE_ADDR_ROW, SITE_Z_XCOL)
            if full:
                out["census183_span"] = row_census(net, ROWS_183, span_fn)
                cen = out["census183_span"]
                site_rows_present = [r for r in SITE_ROWS
                                     if str(r) in cen["rows"]]
                cmx = max(cen["rows"][str(r)]["strength"] for r in SHARED_CTR
                          if str(r) in cen["rows"])
                sstr = max(cen["rows"][str(r)]["strength"]
                           for r in site_rows_present)
                spos = any(cen["rows"][str(r)]["content"] and
                           cen["rows"][str(r)]["strength"] >= 2.0 * cmx
                           for r in site_rows_present)
                best = max(site_rows_present,
                           key=lambda r: cen["rows"][str(r)]["strength"])
                out["site_span"] = {"control_max": cmx, "site_strength": sstr,
                                    "bar_2x_control": 2.0 * cmx,
                                    "site_pos": spos, "peak_row": int(best)}
                log(f"[{tag}] site(span@183) strength {sstr:+.4f} @r{best} "
                    f"(2x-ctrl {2.0 * cmx:.4f}) | read onset "
                    f"{out['site_read']['pz_onset_mean']:.4f} span "
                    f"{out['site_read']['pname_mean_over7']:.4f}")
            else:
                out["census183_span"] = None
                out["site_span"] = None
                log(f"[{tag}] site read onset "
                    f"{out['site_read']['pz_onset_mean']:.4f} span "
                    f"{out['site_read']['pname_mean_over7']:.4f} (light: "
                    f"censuses skipped)")

            # (ii) old-band census (g0 readout): A(129) + row 0 (full only)
            if full:
                out["census_old"] = row_census(
                    net, ROWS_OLD, lambda n: battery_pz(n, g0_ids, zid))
                co = out["census_old"]["rows"]
                out["old_band"] = {
                    "base_pz": out["census_old"]["base_readout"],
                    "row0": co["0"], "row129": co["129"],
                    "row0_strength": co["0"]["strength"],
                    "A129": co["129"]["strength"],
                    "band121_129_max": max(co[str(r)]["strength"]
                                           for r in range(121, 130)
                                           if str(r) in co)}
                log(f"[{tag}] old band: row0 S "
                    f"{out['old_band']['row0_strength']:+.4f} | A(129) "
                    f"{out['old_band']['A129']:+.4f} "
                    f"(base {out['old_band']['base_pz']:.4f})")
            else:
                out["census_old"] = None
                out["old_band"] = None

            # (iii) deletion table (D-all + D-183), still under the clamp
            DELS = {"none": (), "d_all": D_ALL, "d183": (SITE_ADDR_ROW,)}
            out["del_table"] = {}
            for dl, rows_ in DELS.items():
                if dl == "none":
                    sd_d = {k: v.clone() for k, v in sd.items()}
                    gate = {"rows": [], "pass": True, "note": "no deletion"}
                else:
                    sd_d, gate = deleted_wpe(sd, rows_)
                gates_surg[f"{tag}__{dl}"] = gate
                if not gate["pass"]:
                    raise RuntimeError(f"deletion gate FAILED {tag}/{dl}")
                net.load_state_dict(sd_d)
                cell = {"g0": battery_cell(net, bat_ids[0], zid)}
                if dl == "d183":
                    cell["gm12"] = battery_cell(net, bat_ids[-12], zid)
                    cell["site_span"] = read_fact_at(
                        net, site_pool, name_ids, zid, SITE_ADDR_ROW,
                        SITE_Z_XCOL)["pname_mean_over7"]
                out["del_table"][dl] = cell
            net.load_state_dict(sd)                    # restore
            log(f"[{tag}] deletions g0: " + " | ".join(
                f"{dl} {out['del_table'][dl]['g0']['mean_pz']:.3f}"
                for dl in DELS))
        del net
        return out

    # =====================================================================
    # PHASE 1: step-0 batteries (the three states; killed = root + clamp)
    # =====================================================================
    log("--- PHASE 1: step-0 batteries ---")
    time.sleep(STAGGER_S)                              # stagger (shared CPU)
    batteries = {"killed": {}, "washed": {}, "naive": {}}
    batteries["killed"]["0"] = measure(sd_root, "killed_s0", use_clamp=True)
    time.sleep(STAGGER_S)
    batteries["washed"]["0"] = measure(sd_wash, "washed_s0", use_clamp=False)
    time.sleep(STAGGER_S)
    batteries["naive"]["0"] = measure(sd_naive, "naive_s0", use_clamp=False)

    # hook-confinement honesty gates: (a) the kill touches no weights (the
    # killed arm trains FROM the root's exact weights — hook-based by
    # construction, asserted here); (b) the measurement battery under the
    # clamp reproduces the Phase-0 gate read (instrument continuity).
    k0_g0 = batteries["killed"]["0"]["base"][0]["mean_pz"]
    G_HOOKCONF = {
        "note": "the kill is hook-based (input-side clamp): the killed "
                "arm's step-0 weights are the root's, bit-identical by "
                "construction; clamp vectors fixed from the intact base, "
                "never recomputed",
        "measure_killed_s0_g0": k0_g0, "phase0_gate_g0": G_KILL_PZ,
        "measure_matches_gate": bool(abs(k0_g0 - G_KILL_PZ) < G_BIT_TOL),
        "pass": bool(abs(k0_g0 - G_KILL_PZ) < G_BIT_TOL)}
    log(f"G_HOOKCONF: killed s0 g0 {k0_g0:.10f} vs e160 {G_KILL_PZ:.10f}: "
        f"{'PASS' if G_HOOKCONF['pass'] else 'FAIL'}")

    # =====================================================================
    # PHASE 2: the THREE re-teaches (CPU; cooldowns between)
    # =====================================================================
    log("=" * 78)
    arms_run = [
        ("killed", net_root, True,
         "e131_consolidated_e113 + persistent e160 N2/mean clamp "
         "{L1H0,L0H0} (fixed CE_R-bank mean vectors)"),
        ("washed", net_wash, False,
         "e176_root_freeze (the plain-corpus-washed consolidated root)"),
        ("naive", net_naive, False,
         "e001 (the pre-install base; fact-free) — the Ebbinghaus control"),
    ]
    results = {}
    for i, (tag, net_base, use_clamp, desc) in enumerate(arms_run):
        if i > 0:
            log(f"[thermal] cooldown {COOLDOWN_S:.0f}s between trainings")
            cooldown(COOLDOWN_S)
        log(f"RE-TEACH[{tag}]: {FT_STEPS}-step HOME-site locked replay "
            f"(seed {RETEACH_SEED}, clamp={use_clamp}) — base: {desc}")
        clamp_arg = clamp if use_clamp else None
        res = reteach_arm(f"reteach_{tag}", net_base, pool_x, pool_mask,
                          anchor, train_ids, itos, r_eval_xy, gm12_ids,
                          g0_ids, zid, RETEACH_SEED, clamp=clamp_arg)
        results[tag] = {"desc": desc, "res": res, "clamped": use_clamp}
        time.sleep(STAGGER_S)

    G_DRAWFREE = {"zeph_violations": sum(
        r["res"]["zeph_violations"] for r in results.values()),
        "pass": bool(all(r["res"]["zeph_violations"] == 0
                         for r in results.values()))}
    assert G_DRAWFREE["pass"], "name token leaked into a training window"

    # save checkpoints: finals for all three; intermediates for killed/washed
    for tag in ("killed", "washed", "naive"):
        res = results[tag]["res"]
        smax = max(res["sds"]) if res["sds"] else None
        if smax is None:
            trims.append(f"{tag}: no snapshot reached (time cap)")
            continue
        save_ckpt(f"e175_{tag}_reteach", res["sds"][smax],
                  {"desc": f"{results[tag]['desc']} + {smax}-step HOME-site "
                           f"locked re-teach (e119-L recipe, seed "
                           f"{RETEACH_SEED})",
                   "steps": int(smax), "seed": RETEACH_SEED,
                   "clamped_during_training": results[tag]["clamped"],
                   "base": f"runs/checkpoints/"
                           f"{ROOT_CK if tag == 'killed' else WASH_CK if tag == 'washed' else NAIVE_CK}"})
        if tag in ("killed", "washed"):
            for s in sorted(res["sds"]):
                if s == smax:
                    continue
                save_ckpt(f"e175_{tag}_reteach_s{s}", res["sds"][s],
                          {"desc": f"{results[tag]['desc']} + {s}-step "
                                   f"locked re-teach (intermediate snapshot)",
                           "steps": int(s), "seed": RETEACH_SEED,
                           "base": f"runs/checkpoints/"
                                   f"{ROOT_CK if tag == 'killed' else WASH_CK}"})
    missing = {tag: [s for s in CKPT_STEPS if s not in results[tag]["res"]["sds"]]
               for tag in results}
    if any(missing.values()):
        trims.append(f"checkpoints not reached (time cap): {missing}")

    # =====================================================================
    # PHASE 3: full dial batteries per checkpoint (identical instruments)
    # =====================================================================
    log("--- PHASE 3: per-checkpoint dial batteries ---")
    for tag in ("killed", "washed", "naive"):
        use_clamp = results[tag]["clamped"]
        smax = max(results[tag]["res"]["sds"]) if results[tag]["res"]["sds"] else 0
        for s in sorted(results[tag]["res"]["sds"]):
            time.sleep(STAGGER_S)
            # the two bar-carrying arms get FULL dials at every checkpoint;
            # the naive control gets full at {0, final}, light between (the
            # savings bars read g0 only — documented deviation)
            full = (tag in ("killed", "washed")) or (s == smax)
            batteries[tag][str(s)] = measure(results[tag]["res"]["sds"][s],
                                             f"{tag}_s{s}",
                                             use_clamp=use_clamp, full=full)

    # =====================================================================
    # trajectories + ADJUDICATION (registered; no bar shopping)
    # =====================================================================
    def trace_of(tag):
        rows = []
        steps = [0] + sorted(results[tag]["res"]["sds"])
        for s in steps:
            b = batteries[tag]["0" if s == 0 else str(s)]
            ob = b.get("old_band") or {}
            sp = b.get("site_span") or {}
            rows.append({
                "reteach_steps": s,
                "base_g0": b["base"][0]["mean_pz"],
                "base_gm12": b["base"][-12]["mean_pz"],
                "base_gp12": b["base"][12]["mean_pz"],
                "held30_g0": b["base_held"][0]["mean_pz"],
                "held30_gm12": b["base_held"][-12]["mean_pz"],
                "held30_gp12": b["base_held"][12]["mean_pz"],
                "ce_r": b["ce_r"],
                "site_read_onset": b["site_read"]["pz_onset_mean"],
                "site_read_span": b["site_read"]["pname_mean_over7"],
                "site_span_strength": sp.get("site_strength"),
                "site_pos_span": sp.get("site_pos"),
                "A129": ob.get("A129"),
                "row0_strength": ob.get("row0_strength"),
                "band121_129_max": ob.get("band121_129_max"),
                "dall_g0": b["del_table"]["d_all"]["g0"]["mean_pz"],
                "d183_g0": b["del_table"]["d183"]["g0"]["mean_pz"],
                "d183_gm12": b["del_table"]["d183"]["gm12"]["mean_pz"],
                "d183_site_span": b["del_table"]["d183"]["site_span"],
                "dial_density": "full" if b.get("full") else "light",
            })
        return rows

    traces = {tag: trace_of(tag) for tag in ("killed", "washed", "naive")}

    def steps_to_078(tag):
        """first grid checkpoint with g0 >= 0.78 (None = censored)."""
        for r in traces[tag]:
            if r["reteach_steps"] > 0 and r["base_g0"] >= RECOVERY_BAR:
                return r["reteach_steps"]
        return None

    st_killed = steps_to_078("killed")
    st_washed = steps_to_078("washed")
    st_naive = steps_to_078("naive")
    ck_g0 = {tag: {r["reteach_steps"]: r["base_g0"]
                   for r in traces[tag] if r["reteach_steps"] > 0}
             for tag in traces}
    mean_g0 = {tag: float(np.mean([r["base_g0"] for r in traces[tag]
                                   if r["reteach_steps"] > 0]))
               for tag in traces}
    g0_300 = {tag: next((r["base_g0"] for r in traces[tag]
                         if r["reteach_steps"] == max(
                             x["reteach_steps"] for x in traces[tag])),
                        traces[tag][-1]["base_g0"]) for tag in traces}

    fast_fires = bool(st_killed is not None and st_killed <= FAST_STEPS)
    censored = (st_washed is None or st_naive is None)
    if censored:
        savings_verdict = "TEXTURE-CENSORED"
        savings_clause = (f"washed steps-to-{RECOVERY_BAR}: "
                          f"{st_washed}, naive: {st_naive} — at least one "
                          f"censored (> {FT_STEPS} steps); no savings bar "
                          f"fires. Curve texture: mean ckpt g0 washed "
                          f"{mean_g0['washed']:.3f} vs naive "
                          f"{mean_g0['naive']:.3f}; g0@final washed "
                          f"{g0_300['washed']:.3f} vs naive "
                          f"{g0_300['naive']:.3f}.")
    elif st_washed < st_naive:
        savings_verdict = "POSITIVE-SAVINGS"
        savings_clause = (f"the washed net re-learned FASTER than naive "
                          f"(steps-to-{RECOVERY_BAR}: washed {st_washed} < "
                          f"naive {st_naive}) — a residue exists.")
    elif st_washed > st_naive:
        savings_verdict = "NEGATIVE-SAVINGS"
        savings_clause = (f"the washed net re-learned SLOWER than naive "
                          f"(steps-to-{RECOVERY_BAR}: washed {st_washed} > "
                          f"naive {st_naive}) — the surviving brake scar "
                          f"actively interferes (T106's branch).")
    else:
        savings_verdict = "NO-SAVINGS"
        savings_clause = (f"washed and naive re-learned at the same checkpoint "
                          f"bin (steps-to-{RECOVERY_BAR}: both {st_washed}) — "
                          f"the archive truly empty.")

    fired = [v for v, b in (("FAST-RECOVERY", fast_fires),
                            (savings_verdict,
                             savings_verdict in ("POSITIVE-SAVINGS",
                                                 "NEGATIVE-SAVINGS",
                                                 "NO-SAVINGS"))) if b]
    if fast_fires:
        fast_clause = (f"the KILLED net restored g0 to {RECOVERY_BAR} within "
                       f"<= {FAST_STEPS} steps (steps-to-bar: {st_killed}; "
                       f"g0 trajectory "
                       + ", ".join(f"s{s}={ck_g0['killed'][s]:.3f}"
                                   for s in sorted(ck_g0['killed']))
                       + f") — thin access lesion: e164's reading licensed.")
    else:
        fast_clause = (f"the KILLED net did NOT restore g0 to {RECOVERY_BAR} "
                       f"within <= {FAST_STEPS} steps (steps-to-bar: "
                       f"{st_killed}; g0 trajectory "
                       + ", ".join(f"s{s}={ck_g0['killed'][s]:.3f}"
                                   for s in sorted(ck_g0['killed']))
                       + f") — FAST-RECOVERY fails.")

    adjudication = {
        "bars_verbatim": REGISTERED_PREDICTION["queue_row_verbatim"],
        "recovery_bar": RECOVERY_BAR,
        "steps_to_078": {"killed": st_killed, "washed": st_washed,
                         "naive": st_naive,
                         "resolution": "the {10,30,100,300} checkpoint grid"},
        "conditions": {
            "FAST_RECOVERY": {
                "fires": fast_fires, "bar_steps": FAST_STEPS,
                "killed_steps_to_078": st_killed,
                "killed_g0_traj": ck_g0["killed"],
                "clause": fast_clause},
            "POSITIVE_SAVINGS": {
                "fires": savings_verdict == "POSITIVE-SAVINGS",
                "washed_steps": st_washed, "naive_steps": st_naive},
            "NEGATIVE_SAVINGS": {
                "fires": savings_verdict == "NEGATIVE-SAVINGS",
                "washed_steps": st_washed, "naive_steps": st_naive},
            "NO_SAVINGS": {
                "fires": savings_verdict == "NO-SAVINGS",
                "washed_steps": st_washed, "naive_steps": st_naive},
        },
        "censored": censored,
        "co_reported_texture": {
            "mean_ckpt_g0": mean_g0, "g0_at_final_ckpt": g0_300,
            "killed_vs_naive_note": (
                f"the killed arm's steps-to-{RECOVERY_BAR} vs naive "
                f"({st_killed} vs {st_naive}) is CO-REPORTED texture, not a "
                f"registered bar — savings of the access-lesioned state."),
        },
        "verdicts_fired": fired,
        "verdict_clause": f"FAST-RECOVERY: {fast_clause} | SAVINGS: "
                          f"{savings_clause}",
    }
    log("=" * 78)
    log(f"E175 VERDICT(S): {fired if fired else 'NONE (texture)'}")
    log(f"  steps-to-{RECOVERY_BAR}: killed {st_killed} | washed "
        f"{st_washed} | naive {st_naive}")
    log(f"  {fast_clause}")
    log(f"  {savings_clause}")
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e175_savings_triple",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": ("QUEUE.md e175 row (R50's expanded design), "
                         "dispatched ~15:50Z; docstring + bars written "
                         "before compute."),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("identical short HOME-site locked re-teaches on the "
                     "N2-kILLED net (access lesion) and the e176-WASHED net "
                     "(storage half-degraded): do they re-learn faster than "
                     "naive (savings), slower (negative savings — the "
                     "surviving brake scar interferes), or equal (no "
                     "residue)?"),
        "nets": {
            "killed_base": f"runs/checkpoints/{ROOT_CK} + the in-memory "
                           f"persistent {KILL_TAG} clamp (e160's N2/mean "
                           f"instrument, gated bit-exact)",
            "washed": f"runs/checkpoints/{WASH_CK} (gated vs e176 +300 cells)",
            "naive": f"runs/checkpoints/{NAIVE_CK} (e048's BASE; fact-free "
                     f"gate)",
            "eval_clamp": "killed-arm dials run UNDER the clamp",
        },
        "reteach": {
            "desc": "e119-L locked replay VERBATIM at the HOME site (offset "
                    "0: ZEPHYRA at x-cols 130..136, onset read row 129, "
                    "7-target name-only mask, zero position variance)",
            "recipe": "batch 32 = 16 install windows + 16 anchors (8 paired "
                      "+ 8 random), e043 token-level union CE, AdamW "
                      "(0.9,0.95) wd 0.1, lr 1e-3 constant, clip 1.0",
            "steps": FT_STEPS, "ckpt_steps": CKPT_STEPS,
            "seed": RETEACH_SEED,
            "identical_draws": "all three arms see the identical batch "
                               "sequence (evals consume no RNG)",
            "time_cap_s": FT_TIME_CAP, "cooldown_s": COOLDOWN_S,
        },
        "reference_stored": {
            "e109_arm_b_traj": ("locked replay from the INSTALL net (g0 "
                                "0.5563, NOT naive): 0.731@s25, 0.788@s75, "
                                "0.832@s100 (runs/e109/metrics.json) — "
                                "cited as context only"),
            "e119_L_traj": ("locked replay from the INSTALL net at R's 150 "
                            "steps: 0.788@s75, 0.790@s125 (runs/e119/"
                            "metrics.json) — context only"),
            "e113": ("the root's own consolidation (JITTERED replay, "
                     "different protocol) needed the full 300 steps to "
                     "reach 0.785 — context only"),
            "primary_reference": "the RUN naive arm (e001 base, identical "
                                 "protocol/seed)",
        },
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
                     "placement": {"offset": RETEACH_J,
                                   "name_xcols": [HOME_Z_XCOL,
                                                  HOME_Z_XCOL + 6],
                                   "read_rows": [129, 135]},
                     "measurement_pool": {"offset": SITE_J,
                                          "name_xcols": [184, 190],
                                          "note": "e152's locked j=54 pool, "
                                                  "instrument only"},
                     "geometries": {"novel": [-12, 12], "trained_g0": 0}},
        "gates": {"G_SPLICE": G_SPLICE, "G_POOL": G_POOL, "G_SITE": G_SITE,
                  "G_ANCHFREE": G_ANCHFREE, "G_ROOT": G_ROOT,
                  "G_KILL": G_KILL, "G_WASH": G_WASH, "G_NAIVE": G_NAIVE,
                  "G_HOOKCONF": G_HOOKCONF, "G_DRAWFREE": G_DRAWFREE,
                  "G_SURG": gates_surg},
        "arms": {tag: {"desc": results[tag]["desc"],
                       "clamped": results[tag]["clamped"],
                       "steps_ran": results[tag]["res"]["steps_ran"],
                       "seed": results[tag]["res"]["seed"],
                       "traj": results[tag]["res"]["traj"]}
                 for tag in results},
        "traces": traces,
        "batteries": batteries,
        "adjudication": adjudication,
        "honesty_reflex": {
            "in_memory_kill_fidelity": ("the killed state is a forward "
                "pre-hook clamp, not a weight edit: e160's stored N2/mean "
                "cells reproduced "
                f"{'bit-exact' if G_KILL['bit_reproducible'] else 'within tol'}"
                f" (g0 {kz0['mean_pz']:.10f} vs {G_KILL_PZ:.10f}); the clamp "
                "vectors are the INTACT base's CE_R-bank means, fixed and "
                "never recomputed — neither under the joint ablation (e160's "
                "own caveat) nor under training here; the killed arm's "
                "weights at step 0 are the root's, bit-identical."),
            "kill_training_dynamics": ("under the clamp the two heads' qkv "
                "receive zero gradient through the replaced slice (their "
                "weights only shrink via AdamW's decoupled weight decay); "
                "c_proj and all downstream weights train — recovery must "
                "route AROUND the lesion, which is the operational meaning "
                "of 're-learn after an access lesion'"),
            "naive_reference_provenance": ("e109's arm-b and e119-L stored "
                "trajectories start from the INSTALL net (g0 0.5563 — "
                "half-learned, not naive) and e113's is a different "
                "(jittered) protocol, so the naive reference is the RUN "
                "arm on e001.pt (e048's own BASE, the lineage's pre-install "
                "state, measured fact-free at p(Z) 1.34e-05); caveat: e001 "
                "never saw the 400-step install, so its corpus-CE and "
                "anchor-window familiarity differ from the root line's "
                "post-install bases — the naive arm prices the fact-free "
                "learning curve under the identical teaching"),
            "single_seed": ("ONE seed (10902, e119-L's), ONE lineage, three "
                "trajectories — every steps-to-bar number is a point "
                "estimate on the checkpoint grid {10,30,100,300}; crossings "
                "between checkpoints are bounded, not interpolated"),
            "threshold_read": (f"'~0.78' read at g0 (install-60), the "
                f"e133/e150/e160 drop-table convention; the consolidated "
                f"root's own g0 is {G_ROOT_PZ:.4f}, so 0.78 is a "
                f"conservative (harder) bar for re-teach recovery"),
            "site_183_dial": ("the 183-site census is a CO-REPORT on an "
                "instrument (e152's j=54 pool) whose site the home-site "
                "re-teach never trains — growth there would be transport, "
                "not teaching"),
        },
        "trims": trims,
        "recipe_deviations": deviations,
        "ckpt_inventory": CKPT_INVENTORY,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": 2739072,
                   "device": "cpu", "torch_threads": torch.get_num_threads(),
                   "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "savings_triple.png", traces, adjudication, fired)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'savings_triple.png'}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def contextlib_null():
    return _NullCtx()


# ------------------------------------------------------------------ plot

def plot(path, traces, adj, fired):
    """THE Ebbinghaus figure: three re-learn curves overlaid (+ dials)."""
    fig = plt.figure(figsize=(16.0, 10.0))
    gs = fig.add_gridspec(2, 3)
    sty = {"killed": ("tab:red", "o", "KILLED (N2/mean clamp)"),
           "washed": ("tab:blue", "s", "WASHED (e176 freeze)"),
           "naive": ("dimgray", "^", "NAIVE (e001 base)")}

    # (0) THE g0 savings curves
    ax = fig.add_subplot(gs[0, :2])
    for tag, (col, mk, lbl) in sty.items():
        xs = [r["reteach_steps"] for r in traces[tag]]
        ys = [r["base_g0"] for r in traces[tag]]
        ax.plot(xs, ys, f"{mk}-", color=col, lw=2.0, ms=7, label=lbl,
                zorder=3)
        for x, y in zip(xs, ys):
            if x > 0:
                ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                            xytext=(0, 7), fontsize=7, ha="center", color=col)
    ax.axhline(RECOVERY_BAR, ls="--", color="seagreen", lw=1.4)
    ax.text(1.0, RECOVERY_BAR + 0.012, f"recovery bar {RECOVERY_BAR} "
            "(root 0.785)", fontsize=8, color="seagreen")
    ax.axvline(FAST_STEPS, ls=":", color="tab:red", lw=1.2)
    ax.text(FAST_STEPS - 1, 0.06, "FAST bar\n<=30 steps", fontsize=7.5,
            color="tab:red", ha="right")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xticks([0, 10, 30, 100, 300])
    ax.set_xlabel("locked re-teach steps (HOME site, seed 10902)")
    ax.set_ylabel("fact expression g0 (install-60 battery p(Z))")
    ax.set_ylim(-0.03, 1.02)
    ax.set_title("THE EBBINGHAUS SAVINGS TRIPLE — g0 re-learn curves "
                 f"(fired: {fired if fired else 'NONE/TEXTURE'})",
                 fontsize=10.5)
    ax.legend(fontsize=8.5, loc="center right")
    ax.grid(alpha=0.25)

    # (1) g-12 curves
    ax = fig.add_subplot(gs[0, 2])
    for tag, (col, mk, lbl) in sty.items():
        ax.plot([r["reteach_steps"] for r in traces[tag]],
                [r["base_gm12"] for r in traces[tag]], f"{mk}-", color=col,
                lw=1.6, ms=5, label=lbl)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xticks([0, 10, 30, 100, 300])
    ax.set_xlabel("re-teach steps")
    ax.set_ylabel("g-12 (novel geometry)")
    ax.set_title("novel-geometry door (g-12)", fontsize=9.5)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25)

    # (2) held30 g0
    ax = fig.add_subplot(gs[1, 0])
    for tag, (col, mk, lbl) in sty.items():
        ax.plot([r["reteach_steps"] for r in traces[tag]],
                [r["held30_g0"] for r in traces[tag]], f"{mk}-", color=col,
                lw=1.6, ms=5, label=lbl)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xticks([0, 10, 30, 100, 300])
    ax.set_xlabel("re-teach steps")
    ax.set_ylabel("held30 g0 p(Z)")
    ax.set_title("held-30 generalization (co-dial)", fontsize=9.5)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25)

    # (3) steps-to-bar ladder + CE
    ax = fig.add_subplot(gs[1, 1])
    labels, vals, cols = [], [], []
    for tag, (col, mk, lbl) in sty.items():
        st = adj["steps_to_078"][tag]
        labels.append(lbl.split(" ")[0])
        vals.append(st if st is not None else FT_STEPS * 1.15)
        cols.append(col)
        if st is None:
            pass
    ax.bar(labels, vals, 0.55, color=cols, edgecolor="k", lw=0.6)
    for i, (l, v) in enumerate(zip(labels, vals)):
        st = adj["steps_to_078"][["killed", "washed", "naive"][i]]
        ax.text(i, v + 6, str(st) if st is not None else ">300 (censored)",
                ha="center", fontsize=9, fontweight="bold")
    ax.axhline(FAST_STEPS, ls=":", color="tab:red", lw=1.2)
    ax.set_ylabel(f"steps to g0 >= {RECOVERY_BAR}")
    ax.set_title("steps-to-threshold (checkpoint-grid resolution)",
                 fontsize=9.5)
    ax.grid(alpha=0.25, axis="y")

    # (4) verdict panel
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    vl = ["REGISTERED (QUEUE e175 verbatim):",
          "  FAST-RECOVERY: killed <=30 steps to 0.78",
          "  POSITIVE-SAVINGS: washed faster than naive (residue)",
          "  NEGATIVE-SAVINGS: slower (brake scar interferes)",
          "  NO-SAVINGS: equal (archive empty)",
          "",
          "STEPS-TO-0.78:",
          f"  killed: {adj['steps_to_078']['killed']}",
          f"  washed: {adj['steps_to_078']['washed']}",
          f"  naive:  {adj['steps_to_078']['naive']}",
          "",
          f"FIRED: {fired if fired else 'NONE (texture)'}", ""]
    vl += [f"  {w}" for w in
           [adj["verdict_clause"][i:i + 60]
            for i in range(0, len(adj["verdict_clause"]), 60)]]
    for i, tx in enumerate(vl):
        ax.text(0.02, 0.97 - i * 0.045, tx, fontsize=7.0, va="top",
                family="monospace")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
