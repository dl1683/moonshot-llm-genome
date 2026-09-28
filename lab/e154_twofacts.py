"""E154 — TWO FACTS, ONE DOOR (QUEUE.md row e154, rewritten from e134; the
paper's last structural blocker). T097's frame: the geometry door closes
exactly when a NOVEL GRAFT forms (e158: locked@novel shuts it; jitter@novel
and locked@home don't). But e151/e158 re-taught the SAME fact — the open
question: does building a SECOND fact's graft tear down the FIRST fact's
travel? GLOBAL-PHASE would confirm one shared read policy; PER-MEMORY
corrects the abstract (e151 was self-conversion); CROSSOVER prices shared
substrate capacity (W012's contended-bandwidth reading).

ROOT (mandated): runs/checkpoints/e131_consolidated_e113.pt — F1 = ZEPHYRA,
the sink-coupled, geometry-general consolidated fact (row-0 census strength
0.7317 at the trained g0 readout; e113 jitter recipe {-8..+8}; D-all g0
0.9047; g-12 0.9156). Gates below reproduce e131's/e151's/e160's stored cells
before any compute is trusted.

F2 (the new fact): MIRABEL — a 7-char NONCE (0 occurrences in data/input.txt,
verified before compute; onset char M distinct from Z; every char common in
the vocab), constructed per e134's nonce-fact design. The harness's other
candidate (e044's JULIET) is a NATURAL corpus name (e044b's G1 gate: B43
knows JULIET from the corpus) and its checkpoints are B43-lineage — unusable
under the mandated e131 root and confounded as a "novel" graft. A nonce
matches F1's own construction (ZEPHYRA was a nonce install).

INSTALL F2 (the ONE training): e151's locked-re-teach protocol VERBATIM
(e143 = e109 arm-b = e119-L recipe): batch 32 = 16 install windows + 16
anchors (8 paired + 8 random), token-level union CE on the 7 name-char
targets (name-only mask), AdamW (0.9,0.95) wd 0.1 constant lr 1e-3 clip 1.0,
300 steps, in-loop CPU evals every 25, seed 10902 (e151's locked seed).
PLACEMENT: the e109/e151 offset-pool machinery at offset j=-66 — MIRABEL
locked at x-cols 64..70, onset read row 63, SITE BAND rows 63..69 (the
dispatch's ~rows 60-70, clear of the F1 band 121..137 and of 183..189);
pre-context = the 64 TRUE train tokens before each install host,
continuation = the 185 true tokens after it. Zero position variance by
construction. Anchor bank = e151 verbatim (first 16 install-position
ORIGINAL host windows, incumbent continuations, no name).

MEASURE F1 BEFORE(root)/AFTER(e154_twofacts.pt) — identical instruments:
  (i)   the DOOR: base batteries at g-12 / g0 / g+12 (install-60, e119
        construction ctx = train_text[p-PRE-j:p]) + held30@g0; retention =
        after/before per geometry.
  (ii)  F1's phase dials: old-band census (rows 121..129 + row 0, readout
        battery_pz on ids130) => A(129) + row-0 strength; deletion table
        {none, d129, D-all{121,125,129,133,137}, d_r0, dF2site{63..69}} at
        g0 (dF2 also read at g-12 — does F2's graft sit in F1's read path?);
        e150-informed probes: forced-off-sink MASK at g0/g-12 + NORM-LADDER
        0.07/0.15 (the poison bracket).
MEASURE F2: site reads (onset p(M) at row 63 + name-span mean over rows
63..69, install pool primary + held30-host pool texture); site-content
census at the F2 band (e139/e143/e151 span-primary convention; controls
{20,40,80,100,150,160,200,220} — row 60 falls in the F2 site's causal past
and is NOT a control here); cross-leak cells (p(M) on F1's g0 battery;
p(Z) at row 63 on F2's pool).

N2 RIDER (eval-only, e160's set): N2 = {L1H0, L0H0} (e160's top-2-noL0H3,
zero + mean modes; singles as texture) on BEFORE (instrument gate vs e160's
stored cells: zero 0.16161009669303894, mean 0.22963003814220428) and AFTER
— pricing F1 (g0 primary, g-12 co-report) and F2 (site onset, e160's
site-column convention) separately: the layer-attribution question (is the
read policy one global head circuit or per-fact readouts?).

W017 RIDER (free): F2's 36-head zero-mode census on the after-net (readout =
F2 site onset on the held30 pool + CE per head, e133's convention) —
locked-trained F2 should grow a DIFFUSE load (top-1 low, many heads) like
arm_b's (site-stored top-1 share 0.257, install 0.117), not concentrated
like F1's (0.352).

REGISTERED PREDICTION (QUEUE e154 / dispatch VERBATIM — adjudicate against
exactly this; no bar shopping; texture => TEXTURE with numbers):
  - GLOBAL-PHASE fires if: F1 g-12 drops >= 70% (0.916 -> <= 0.27) while
    F1's own trained-geometry expression stays >= 0.5 — the door closed on
    an untouched fact; T097's one-event reading generalizes across facts.
  - PER-MEMORY fires if: F1 g-12 within 15% — e151/e158 were
    self-conversion; the abstract's "globally" rewrites; doors are per-fact.
  - CROSSOVER fires if: 30-70% drop — shared substrate with capacity (feeds
    W012's contended-bandwidth reading).
  - Rider bars: SHARED-READOUT = N2 kills both facts jointly;
    PER-FACT-READOUT = N2 kills F2 while F1 holds.
  - No bar shopping; texture => TEXTURE.

OPERATIONALIZATIONS (frozen here before compute):
  * door_drop = 1 - g12_after/g12_before (mean p(Z), install-60 battery).
  * GLOBAL-PHASE: door_drop >= 0.70 AND g0_after >= 0.50 (absolute mean p at
    the trained geometry — "stays >= 0.5"; the g0 retention ratio is
    co-reported). door_drop >= 0.70 with g0_after < 0.50 is TEXTURE (the
    fact was damaged everywhere, not door-specifically).
  * PER-MEMORY: |door_drop| <= 0.15. CROSSOVER: 0.30 <= door_drop < 0.70
    (the >= 0.70 edge belongs to GLOBAL-PHASE). Gap (0.15, 0.30) => TEXTURE.
  * Adjudication order: GLOBAL-PHASE -> PER-MEMORY -> CROSSOVER -> TEXTURE.
  * Rider kill = >= 60% drop (e160's KILL_DROP); F1 read = g0 battery
    (e160's primary), F2 read = F2 site onset over the install pool (e160's
    site-column convention); N2 ZERO mode primary, mean co-reported.
    SHARED-READOUT = F1 and F2 both >= 60%; PER-FACT-READOUT = F2 >= 60% AND
    F1 < 60%; the third pattern (F1 >= 60%, F2 < 60% — e160's
    type-selective shape) = TEXTURE named F1-ONLY-KILL, numbers reported.
  * W017 rider (free, no kill bar): load shares = positive drops / sum of
    positive drops (e133 convention); DIFFUSE-AS-PREDICTED if F2's top-1
    share < 0.352 (F1's stored sink-coupled top-1); CONCENTRATED otherwise.

INSTRUMENT PROVENANCE: load_cpu/evl_load/battery_cell/battery_pz/
ce_fixed_cpu/val_windows/deleted_wpe/modified_wpe/read_fact_at/row_census/
finetune_arm are lab/e151_twodoor.py VERBATIM (which are e143/e150 verbatim
= the e065/e068/e109/e113/e116/e119/e131 lineage) with the GPU gate removed
(CPU mandated); forward_custom/fwd_causal/fwd_offsink/battery_fwd/ce_fwd are
e150's VERBATIM (via e151); head_mean_vec/HeadReplace are lab/
e160_headset_escalation.py VERBATIM. Copied, not imported, per lab
convention (self-contained experiment files). Protocol rebuild: corpus seed
1337, SPLICE_RNG 24301 host shuffle, install60/held30 split, mix gate —
e143/e151 verbatim.

COMPUTE ENVELOPE: CPU-ONLY MANDATED (CUDA_VISIBLE_DEVICES=-1 before torch
import; GPU user-occupied; e164 also on CPU) — torch threads 4, all evals
sequential, 3 s staggers between phases, cooldown(60) around the ONE
training, no busy-waiting; training cap 1500 s (e151's CPU-fallback
convention), in-loop evals every 25 steps. Single seed, single lineage.

Outputs: runs/e154/{metrics.json, twofacts.png}; the two-fact net
runs/checkpoints/e154_twofacts.pt (ckpt_inventory in metrics). No NOTES/
THINKING/QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python e154_twofacts.py    (E154_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # CPU-ONLY (mandated)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                           # noqa: E402

torch.set_num_threads(4)                              # LOW (e164 co-runs)

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
common.DEVICE = "cpu"
from common import (Cfg, CharCorpus, TinyGPT, cooldown, run_dir,    # noqa: E402
                    save_json)
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E154_SMOKE") == "1"
CPU = torch.device("cpu")
DEV = CPU                                             # CPU mandated

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME1 = "ZEPHYRA"                    # F1 (the root's sink-coupled fact)
NAME2 = "MIRABEL"                    # F2 (the nonce; 0 corpus occurrences)
PRE = 130
POST_CAP = 119                       # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
ROOT_CK = "e131_consolidated_e113.pt"
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E133_METRICS = E43.REPO / "runs" / "e133" / "metrics.json"  # W017 overlay

# ---- F2 placement (fresh site rows 63..69, clear of band 121..137 and 183) ----
F2_J = -66                           # offset: name x-cols 64..70, read rows 63..69
F2_ADDR_ROW = 63                     # onset read row
F2_Z_XCOL = PRE + F2_J               # 64
F2_CONT = BLOCK - PRE - F2_J - len(NAME2)     # 185-token continuation
F2_ROWS = tuple(range(63, 70))       # the 7 trained read rows
CTR2 = (20, 40, 80, 100, 150, 160, 200, 220)  # outside every band; clear of
                                              # the F2 site's causal past
ROWS_F2 = (0, 1, 2) + (60, 61, 62) + F2_ROWS + CTR2          # span census
ROWS_F2_ONSET = (0,) + (60, 61, 62) + (63,) + CTR2           # onset census
ROWS_F2_BEFORE = (0,) + F2_ROWS + CTR2                       # before: null ctl
ROWS_OLD = (0, 1, 2, 3, 4, 5, 6, 60, 100, 118, 119, 120) + tuple(range(121, 130))
D_ALL = (121, 125, 129, 133, 137)    # e113's fixed set
GEOS = (-12, 0, 12)                  # novel x2 + trained g0 (root {-8..+8})
if SMOKE:
    ROWS_F2 = (0, 1) + F2_ROWS + (20, 40)
    ROWS_F2_ONSET = (0, 63) + (20, 40)
    ROWS_F2_BEFORE = (0,) + F2_ROWS + (20, 40)
    ROWS_OLD = (0, 1, 60, 121, 125, 129)

# ---- fine-tune envelope (e109 arm-b / e119-L / e143 / e151 verbatim) ------------
FT_LR = 1e-3
FT_STEPS = 300 if not SMOKE else 8
FT_TIME_CAP = 1500.0                 # CPU cap (e151's CPU-fallback convention)
EVAL_EVERY = 25 if not SMOKE else 2
NAME_BS, ANCH_BS = 16, 16
F2_SEED = 10902                      # e119-L / e143 locked / e151's seed
COOLDOWN_S = 60.0                    # CPU envelope around the ONE training
STAGGER_S = 3.0                      # between phases (e160 convention)

# ---- gates / references (full precision, from stored metrics) -------------------
R_EVAL_SEED = 26502                  # e065 CE_R eval-bank seed (verbatim)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05
G_CONS_REF_PZ = 0.7850371599197388            # e131/e151 none__g+0__install60
G_CONS_REF_CE = 1.663516640663147             # e131/e151 none__ce_r
G_CONS_REF_G12 = 0.9155886173248291           # e150/e160 g-12 base
G_R0_REF_MEAN = 0.7842019017236945            # e131 consolidated census row 0
G_R0_REF_ZERO = 0.7316772222270098
G_A129_REF_MEAN = -0.1084650677318375         # e131 consolidated census row 129
G_A129_REF_ZERO = -0.13237020391970877
G_DALL_REF = 0.9047248959541321               # e131 d_all_e113__g+0__install60
G_N2Z_REF = 0.16161009669303894               # e160 N2 zero @g0 (before net)
G_N2M_REF = 0.22963003814220428               # e160 N2 mean @g0 (before net)

# ---- e150-informed probe constants ----------------------------------------------
LADDER_NORMS = (0.07, 0.15)          # the poison-probe bracket (e150 verdict)

# ---- registered bar constants (frozen) -------------------------------------------
GLOBAL_DROP_BAR = 0.70               # GLOBAL-PHASE door-drop clause
F1_G0_FLOOR = 0.50                   # GLOBAL-PHASE trained-geometry floor
PERMEM_TOL = 0.15                    # PER-MEMORY |door_drop| clause
XOVER_LO = 0.30                      # CROSSOVER lower edge
KILL_DROP = 0.60                     # rider kill bar (e160's KILL_DROP)
SITE_CTRL_MULT = 2.0                 # e139 site convention: >= 2x control-max
W017_TOP1_F1 = 0.352                 # F1's stored sink-coupled top-1 share
W017_TOP1_SITE = 0.257               # e133 site-stored top-1 share
W017_TOP1_INST = 0.117               # e133 install top-1 share

REGISTERED_PREDICTION = {
    "global_phase": "GLOBAL-PHASE fires if: F1 g-12 drops >= 70% (0.916 -> "
        "<= 0.27) while F1's own trained-geometry expression stays >= 0.5 — "
        "the door closed on an untouched fact; T097's one-event reading "
        "generalizes across facts.",
    "per_memory": "PER-MEMORY fires if: F1 g-12 within 15% — e151/e158 were "
        "self-conversion; the abstract's 'globally' rewrites; doors are "
        "per-fact.",
    "crossover": "CROSSOVER fires if: 30-70% drop — shared substrate with "
        "capacity (feeds W012's contended-bandwidth reading).",
    "rider_bars": "SHARED-READOUT = N2 kills both facts jointly; "
                  "PER-FACT-READOUT = N2 kills F2 while F1 holds.",
    "no_bar_shopping": "No bar shopping; texture => TEXTURE.",
    "operationalizations": "door_drop = 1 - g12_after/g12_before (mean p(Z), "
        "install-60 battery); GLOBAL-PHASE = door_drop >= 0.70 AND g0_after "
        ">= 0.50 absolute (retention co-reported; >= 0.70 with g0 < 0.50 is "
        "TEXTURE); PER-MEMORY = |door_drop| <= 0.15; CROSSOVER = 0.30 <= "
        "door_drop < 0.70; gap (0.15, 0.30) => TEXTURE; order GLOBAL-PHASE "
        "-> PER-MEMORY -> CROSSOVER -> TEXTURE. Rider kill = >= 60% drop, "
        "F1 read g0 (e160 primary), F2 read = site onset install pool "
        "(e160 site-column convention), N2 zero primary / mean co-report; "
        "SHARED-READOUT = both >= 60%; PER-FACT-READOUT = F2 >= 60% AND "
        "F1 < 60%; F1 >= 60% & F2 < 60% = TEXTURE named F1-ONLY-KILL. W017 "
        "rider free: shares = positive drops / sum positive; "
        "DIFFUSE-AS-PREDICTED if top-1 < 0.352 (F1's stored), else "
        "CONCENTRATED.",
    "committed": "NONE REGISTERED on the main fork (the dispatch left it "
                 "open — 'decides the day's headline'); T088's shared-state "
                 "argument informally leans GLOBAL-PHASE. Recorded, not "
                 "adjudicated against.",
}

trims: list[str] = []
deviations: list[str] = [
    "Nets are the mandated 2.7M e131_consolidated line (e143's precedent "
    "note; the dispatch's '<=1M family' phrase is an envelope statement — "
    "every gate reference and lineage number of this cell lives on the 2.7M "
    "line).",
    "F2 = MIRABEL, a nonce (0 corpus occurrences, verified pre-compute): "
    "e134's nonce-fact design. The harness's other candidate (e044's "
    "JULIET) is a natural corpus name known to the base net (e044b's G1) "
    "and lives on the B43 lineage — a novelty confound under the mandated "
    "e131 root. A nonce matches F1's own construction.",
    "F2 site rows 63..69 (offset j=-66): pre-context 64 tokens vs g0's 130 "
    "— the NEAR-extreme of the placement-statistics axis (e151's FAR "
    "confound mirrored); and the F2 band sits INSIDE the causal past of "
    "F1's battery reads (positions 117..143 attend keys 63..69), so "
    "graft-interference with F1's read is real and priced by the dF2 "
    "deletion arm (the F1 battery text itself never contains MIRABEL).",
    "Census controls CTR2 = {20,40,80,100,150,160,200,220} replace e151's "
    "{60,100,...}: row 60 falls in the F2 site's causal past and cannot "
    "serve as a control for the F2 census.",
    "Before-net F2 census is reduced (span readout, fewer rows — the null "
    "control; the AFTER census carries the 2x-control bar); the onset "
    "census runs a reduced row set (rows > 63 are causally invisible to "
    "the onset read — e151's own exactly-zero cells).",
    "N2 rider adds the two singles (L1H0, L0H0; zero mode) as texture "
    "cells beyond the dispatched N2 set — denser attribution, same bars.",
    "CPU-ONLY mandated (GPU user-occupied; e164 co-running): training under "
    "the 1500 s CPU cap (e151's fallback convention), threads 4, sequential "
    "evals, 3 s staggers, cooldown(60) around the ONE training.",
    "Single seed (10902), single lineage — no replication arm (dispatch "
    "envelope: ONE training).",
    "Smoke mode trims: 8-step install, reduced census rows, 6-head W017 "
    "census, reduced rider cells; nothing adjudicated.",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: load_cpu/evl_load/battery_cell/battery_pz/ce_fixed_cpu/
# val_windows/deleted_wpe/modified_wpe/read_fact_at/row_census/finetune_arm are
# lab/e151_twodoor.py VERBATIM (e143/e150 lineage) with the GPU gate removed;
# forward_custom/fwd_causal/fwd_offsink/battery_fwd/ce_fwd are e150's VERBATIM
# (via e151); head_mean_vec/HeadReplace are lab/e160_headset_escalation.py
# VERBATIM.

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
def battery_fwd(net: TinyGPT, ids: torch.Tensor, zid: int, fwd=None,
                bs=30) -> dict:
    """e150's battery_fwd: battery_cell with a pluggable forward."""
    net.eval()
    f = fwd if fwd is not None else net
    pzs, amax = [], 0.0
    for i in range(0, ids.shape[0], bs):
        lg, _ = f(ids[i:i + bs])
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


@torch.no_grad()
def ce_fwd(net: TinyGPT, x, y, fwd=None, bs=64) -> float:
    f = fwd if fwd is not None else net
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        _, loss = f(x[i:i + bs], y[i:i + bs])
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


def modified_wpe(sd: dict, row: int, value) -> tuple[dict, dict]:
    """e141's row-value surgery with e131's confinement gate."""
    out = {k: v.clone() for k, v in sd.items()}
    out["wpe.weight"][row] = value
    d = out["wpe.weight"] != sd["wpe.weight"]
    changed_rows = sorted(set(int(r) for r in torch.nonzero(d)[:, 0].tolist()))
    n = int(d.sum().item())
    others = all(torch.equal(out[k], sd[k]) for k in out if k != "wpe.weight")
    ok_rows = changed_rows in ([], [row])
    gate = {"row": row, "n_elements_changed": n,
            "changed_rows": changed_rows,
            "identity": bool(n == 0),
            "confined": bool(ok_rows),
            "others_bit_identical": bool(others),
            "pass": bool(ok_rows and others)}
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


@torch.no_grad()
def forward_custom(net: TinyGPT, idx, targets=None, block_key0=False):
    """e150's custom forward VERBATIM (the forced-off-sink mask)."""
    B, T = idx.shape
    C = net.cfg.n_embd
    H = net.cfg.n_head
    D = C // H
    pos = torch.arange(T, device=idx.device)
    x = net.wte(idx) + net.wpe(pos)
    mask = torch.zeros(T, T, device=idx.device)
    mask.masked_fill_(torch.triu(torch.ones(T, T, device=idx.device,
                                            dtype=torch.bool), 1),
                      float("-inf"))
    if block_key0:
        mask[:, 0] = float("-inf")
        mask[0, 0] = 0.0                 # keep row 0's only legal key
    for block in net.h:
        xin = block.ln1(x)
        q, k, v = block.attn.c_attn(xin).split(C, dim=2)
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + block.attn.c_proj(y)
        x = x + block.mlp(block.ln2(x))
    logits = net.lm_head(net.ln_f(x))
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.reshape(-1))
    return logits, loss


def fwd_causal(net):
    return lambda idx, targets=None: forward_custom(
        net, idx, targets=targets, block_key0=False)


def fwd_offsink(net):
    return lambda idx, targets=None: forward_custom(
        net, idx, targets=targets, block_key0=True)


def head_mean_vec(net: TinyGPT, layer: int, head: int, bank_x, bs=30):
    """e133/e160: mean of the head's 32-dim slice of c_proj's input over the
    CE_R bank (corpus windows only)."""
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
    """e160 VERBATIM: replace head slices of c_proj's input with fixed
    vectors (mean-replace) or zeros, eval-only forward pre-hooks."""

    def __init__(self, net: TinyGPT, replace: dict):
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
                    sl = x[..., hd_ * self.hd:(hd_ + 1) * self.hd]
                    if vec is None:
                        sl.zero_()
                    else:
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

def finetune_arm(tag: str, net0: TinyGPT, pool_x: torch.Tensor,
                 pool_mask: torch.Tensor, anchor: torch.Tensor,
                 train_ids: torch.Tensor, r_eval_xy, f_eval_ids, zid: int,
                 seed: int):
    """e151's finetune_arm VERBATIM (e109 arm-b / e119-L recipe), CPU-mandated
    (GPU gate removed): batch 32 = 16 install windows + 16 anchors (8 paired
    + 8 random); token-level union CE on the name-char targets; AdamW
    (0.9,0.95) wd 0.1 lr 1e-3 constant clip 1.0; 300 steps / CPU cap;
    in-loop evals every 25 (evals consume no RNG)."""
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
            traj.append({"step": step, "p_name_mean": bz["mean_pz"],
                         "frac_argmax": bz["frac_argmax_z"], "ce_r": ce_r,
                         "elapsed_s": round(time.time() - t_start, 1)})
            log(f"  [{tag}] s{step:4d} p(M@site) {bz['mean_pz']:.4f} argmax "
                f"{bz['frac_argmax_z']:.2f} CE_R {ce_r:.4f} "
                f"({traj[-1]['elapsed_s']:.0f}s)")
        if (time.time() - t_start) > FT_TIME_CAP:
            log(f"  [{tag}] time cap {FT_TIME_CAP:.0f}s at s{step}")
            trims.append(f"{tag}: time cap at step {step}")
            break
    net.eval()
    sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    del net
    return {"sd": sd_cpu, "traj": traj, "steps_ran": step, "seed": seed}


CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "e154", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name} "
        f"({', '.join(f'{k}={v}' for k, v in meta.items())})")


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e154_smoke" if SMOKE else "e154")
    log(f"E154 TWO FACTS, ONE DOOR (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY mandated, torch threads {torch.get_num_threads()}, "
        f"cuda available = {torch.cuda.is_available()}")

    # ---------------- protocol rebuild (e143/e151 verbatim)
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    mid = stoi["M"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    G_NONCE = {"name": NAME2, "corpus_occurrences": train_text.count(NAME2)
               + val_text.count(NAME2),
               "len": len(NAME2), "onset_char": "M",
               "pass": bool(train_text.count(NAME2) == 0
                            and val_text.count(NAME2) == 0
                            and len(NAME2) == len(NAME1))}
    assert G_NONCE["pass"], f"F2 nonce gate FAILED: {G_NONCE}"
    log(f"F2 nonce gate: {NAME2} absent from corpus "
        f"({G_NONCE['corpus_occurrences']} occurrences), 7 chars like F1")

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
    name1_ids = corpus.encode(NAME1)
    name2_ids = corpus.encode(NAME2)
    L = len(NAME2)

    # ---------------- F2 install pool: offset j=-66 (fresh site rows 63..69)
    def build_f2_pool(occ):
        wins = []
        for p, h in occ:
            pre = train_ids[p - PRE - F2_J: p]
            post = train_ids[p + len(h): p + len(h) + F2_CONT]
            if len(pre) != PRE + F2_J or len(post) != F2_CONT:
                raise RuntimeError(f"F2 window short at p={p}")
            w = torch.cat([pre, name2_ids, post])
            if len(w) != BLOCK:
                raise RuntimeError(f"F2 window len {len(w)} != {BLOCK}")
            wins.append(w)
        return torch.stack(wins)

    pool2 = build_f2_pool(install_occ)
    pool2_held = build_f2_pool(held_occ)
    pool2_mask = torch.zeros(len(pool2), BLOCK - 1, dtype=torch.bool)
    pool2_mask[:, F2_ADDR_ROW: F2_ADDR_ROW + L] = True
    G_GEO2 = {
        "name_xcols": [F2_Z_XCOL, F2_Z_XCOL + L - 1],
        "all_windows_name_in_place": bool(
            all(torch.equal(w[F2_Z_XCOL: F2_Z_XCOL + L], name2_ids)
                for w in pool2)),
        "held_windows_name_in_place": bool(
            all(torch.equal(w[F2_Z_XCOL: F2_Z_XCOL + L], name2_ids)
                for w in pool2_held)),
        "mask_targets_per_window": int(pool2_mask[0].sum()),
        "mask_cols": [F2_ADDR_ROW, F2_ADDR_ROW + L - 1],
        "zero_variance": True,
        "clear_of_f1_band": bool(F2_ADDR_ROW + L - 1 < 121),
        "clear_of_183": bool(F2_ADDR_ROW + L - 1 < 183),
    }
    G_GEO2["pass"] = bool(G_GEO2["all_windows_name_in_place"]
                          and G_GEO2["held_windows_name_in_place"]
                          and G_GEO2["mask_targets_per_window"] == L
                          and G_GEO2["clear_of_f1_band"])
    assert G_GEO2["pass"], f"F2 geometry gate FAILED: {G_GEO2}"
    log(f"F2 pool: {tuple(pool2.shape)} + held {tuple(pool2_held.shape)} — "
        f"{NAME2} locked at x-cols {F2_Z_XCOL}..{F2_Z_XCOL + L - 1} (onset "
        f"read row {F2_ADDR_ROW}), {L}-target name mask, zero variance")

    # anchor bank (e065/e109/e143/e151 verbatim): first 16 install-position
    # ORIGINAL host windows (incumbent continuations, no name)
    anchor = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                          for p, _ in install_occ[:16]])

    # ---------------- F1 batteries: ctx = train_text[p-PRE-j : p] (e119)
    bat1 = {}
    for j in GEOS:
        for tag, occ in (("install60", install_occ), ("held30", held_occ)):
            cs = [train_text[p - PRE - j: p] for p, _ in occ]
            bat1[(j, tag)] = torch.stack([corpus.encode(c) for c in cs])
    ids130 = bat1[(0, "install60")]             # e116/e131 battery verbatim
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    f2_eval_ids = pool2[:, :F2_Z_XCOL]          # p(M) read at row 63

    # ---------------- root net + gates
    net0 = load_cpu(CKPT_DIR / ROOT_CK)
    sd_root = {k: v.clone() for k, v in net0.state_dict().items()}
    ev = copy.deepcopy(net0)
    bz0 = battery_cell(ev, ids130, zid)
    bz012 = battery_cell(ev, bat1[(-12, "install60")], zid)
    ce0 = ce_fixed_cpu(ev, *r_eval_xy)
    G_CONS = {"battery_pz": bz0["mean_pz"], "ref_pz": G_CONS_REF_PZ,
              "battery_pz_g12": bz012["mean_pz"], "ref_pz_g12": G_CONS_REF_G12,
              "ce_r": ce0, "ref_ce": G_CONS_REF_CE,
              "bit_reproducible": bool(
                  abs(bz0["mean_pz"] - G_CONS_REF_PZ) < G_BIT_TOL
                  and abs(ce0 - G_CONS_REF_CE) < G_BIT_TOL
                  and abs(bz012["mean_pz"] - G_CONS_REF_G12) < G_BIT_TOL),
              "pass": bool(abs(bz0["mean_pz"] - G_CONS_REF_PZ) < G_FALLBACK_TOL
                           and abs(ce0 - G_CONS_REF_CE) < G_FALLBACK_TOL
                           and abs(bz012["mean_pz"] - G_CONS_REF_G12)
                           < G_FALLBACK_TOL)}
    log(f"G_CONS root: p(Z) {bz0['mean_pz']:.10f} (ref {G_CONS_REF_PZ:.10f}) "
        f"g-12 {bz012['mean_pz']:.10f} CE_R {ce0:.6f} (ref {G_CONS_REF_CE:.6f}):"
        f" {'PASS' if G_CONS['pass'] else 'FAIL'}")
    if not G_CONS["pass"]:
        raise RuntimeError("root checkpoint failed its gate vs e131/e150 cells")

    # mask-instrument gate (e150): custom-causal must reproduce the standard
    with torch.no_grad():
        lg_std, _ = net0(ids130[:6])
        lg_cus, _ = forward_custom(net0, ids130[:6], block_key0=False)
        dmax = float((lg_std - lg_cus).abs().max())
    pz_cus = battery_fwd(net0, ids130, zid, fwd=fwd_causal(net0))["mean_pz"]
    dpz = abs(pz_cus - bz0["mean_pz"])
    G_MASK = {"max_abs_logit_diff_batch6": dmax, "abs_dpz_install60_g0": dpz,
              "tol_logit": 1e-2, "tol_pz": 1e-3,
              "pass": bool(dmax < 1e-2 and dpz < 1e-3)}
    log(f"G_MASK custom-causal vs standard: max|dlogit| {dmax:.3e}, |dp(Z)| "
        f"{dpz:.3e}: {'PASS' if G_MASK['pass'] else 'FAIL'}")
    if not G_MASK["pass"]:
        raise RuntimeError("custom forward does not reproduce the standard one")
    del ev

    # =====================================================================
    # the BEFORE/AFTER measurement battery (identical instruments)
    # =====================================================================
    DELS = {"none": (), "d129": (PRE - 1,), "d_all": D_ALL, "d_r0": (0,),
            "dF2": tuple(F2_ROWS)}
    gates_surg: dict = {}

    def measure(sd: dict, tag: str, full_f2: bool) -> dict:
        net = evl_load(sd)
        out: dict = {"tag": tag}
        # (i) the door: F1 base expression + CE
        out["base"] = {j: battery_cell(net, bat1[(j, "install60")], zid)
                       for j in GEOS}
        out["base_held30_g0"] = battery_cell(net, bat1[(0, "held30")], zid)
        out["ce_r"] = ce_fixed_cpu(net, *r_eval_xy)
        log(f"[{tag}] F1 base: " + " ".join(
            f"g{j:+d} {out['base'][j]['mean_pz']:.4f}" for j in GEOS)
            + f" | held30 g0 {out['base_held30_g0']['mean_pz']:.4f} "
            f"| CE_R {out['ce_r']:.4f}")

        # cross-leak cells (texture): p(M) on F1's battery; p(Z)@63 on F2 pool
        out["f2_leak_g0"] = battery_cell(net, ids130, mid)["mean_pz"]
        out["f1_leak_site"] = battery_cell(net, f2_eval_ids, zid)["mean_pz"]

        # F2 site reads (install pool primary, held30 texture)
        out["f2_site_install"] = read_fact_at(net, pool2, name2_ids, mid,
                                              F2_ADDR_ROW, F2_Z_XCOL)
        out["f2_site_held"] = read_fact_at(net, pool2_held, name2_ids, mid,
                                           F2_ADDR_ROW, F2_Z_XCOL)
        log(f"[{tag}] F2 site: onset {out['f2_site_install']['pz_onset_mean']:.4f}"
            f" span {out['f2_site_install']['pname_mean_over7']:.4f} "
            f"(held {out['f2_site_held']['pz_onset_mean']:.4f}) | leaks: "
            f"p(M)@129 {out['f2_leak_g0']:.4f} p(Z)@63 "
            f"{out['f1_leak_site']:.4f}")

        # F2 site-content census (span-primary; onset co-report on after)
        def span2_fn(n):
            return read_fact_at(n, pool2, name2_ids, mid, F2_ADDR_ROW,
                                F2_Z_XCOL)["pname_mean_over7"]

        def onset2_fn(n):
            return read_fact_at(n, pool2, name2_ids, mid, F2_ADDR_ROW,
                                F2_Z_XCOL)["pz_onset_mean"]

        rows_span = ROWS_F2 if full_f2 else ROWS_F2_BEFORE
        out["censusF2_span"] = row_census(net, rows_span, span2_fn)
        cen = out["censusF2_span"]
        cm = max(cen["rows"][str(r)]["strength"] for r in CTR2
                 if str(r) in cen["rows"])
        sstr = max(cen["rows"][str(r)]["strength"] for r in F2_ROWS)
        spos = any(cen["rows"][str(r)]["content"] and
                   cen["rows"][str(r)]["strength"] >= SITE_CTRL_MULT * cm
                   for r in F2_ROWS)
        best = max(F2_ROWS, key=lambda r: cen["rows"][str(r)]["strength"])
        out["site63_span"] = {"control_max": cm, "site_strength": sstr,
                              "bar_2x_control": SITE_CTRL_MULT * cm,
                              "site_pos": spos, "peak_row": int(best)}
        log(f"[{tag}] F2 site(span) strength {sstr:+.4f} @r{best} "
            f"(2x-ctrl {SITE_CTRL_MULT * cm:.4f}) -> site_pos {spos} "
            f"| base readout {cen['base_readout']:.4f}")
        if full_f2:
            out["censusF2_onset"] = row_census(net, ROWS_F2_ONSET, onset2_fn)
            ceno = out["censusF2_onset"]
            cmo = max(ceno["rows"][str(r)]["strength"] for r in CTR2
                      if str(r) in ceno["rows"])
            sstro = max(ceno["rows"][str(r)]["strength"] for r in (63,))
            out["site63_onset"] = {
                "control_max": cmo, "site_strength": sstro,
                "bar_2x_control": SITE_CTRL_MULT * cmo,
                "site_pos": bool(ceno["rows"]["63"]["content"]
                                 and ceno["rows"]["63"]["strength"]
                                 >= SITE_CTRL_MULT * cmo)}
            log(f"[{tag}] F2 site(onset) strength {sstro:+.4f} "
                f"(2x-ctrl {SITE_CTRL_MULT * cmo:.4f}) -> "
                f"{out['site63_onset']['site_pos']}")

        # (ii) F1 old-band census => A(129), row-0 strength
        out["census_old"] = row_census(net, ROWS_OLD,
                                       lambda n: battery_pz(n, ids130, zid))
        co = out["census_old"]["rows"]
        r0, a129 = co["0"], co["129"]
        out["old_band"] = {
            "base_pz": out["census_old"]["base_readout"],
            "row0": r0, "row129": a129,
            "row0_strength": r0["strength"], "A129": a129["strength"],
            "band121_129_max": max(co[str(r)]["strength"]
                                   for r in range(121, 130)
                                   if str(r) in co)}
        log(f"[{tag}] F1 old band: row0 S {r0['strength']:+.4f} | A(129) "
            f"{a129['strength']:+.4f} | band121-129 max "
            f"{out['old_band']['band121_129_max']:+.4f} "
            f"(base {out['census_old']['base_readout']:.4f})")

        # (ii) deletion table (dF2 = the graft's footprint on F1's reads)
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
            cell = {"g0": battery_cell(net, bat1[(0, "install60")], zid)}
            if dl == "dF2":
                cell["gm12"] = battery_cell(net, bat1[(-12, "install60")], zid)
                cell["f2_onset"] = read_fact_at(
                    net, pool2, name2_ids, mid, F2_ADDR_ROW,
                    F2_Z_XCOL)["pz_onset_mean"]
                cell["f2_span"] = read_fact_at(
                    net, pool2, name2_ids, mid, F2_ADDR_ROW,
                    F2_Z_XCOL)["pname_mean_over7"]
            out["del_table"][dl] = cell
        net.load_state_dict(sd)                    # restore
        log(f"[{tag}] deletions g0: " + " | ".join(
            f"{dl} {out['del_table'][dl]['g0']['mean_pz']:.3f}"
            for dl in DELS))

        # (ii) e150-informed probes at both geometries
        f0, f1 = fwd_causal(net), fwd_offsink(net)
        mask = {}
        for j in (0, -12):
            b = battery_fwd(net, bat1[(j, "install60")], zid, fwd=f0)
            m = battery_fwd(net, bat1[(j, "install60")], zid, fwd=f1)
            bce = ce_fwd(net, *r_eval_xy, fwd=f0)
            mce = ce_fwd(net, *r_eval_xy, fwd=f1)
            mask[f"g{j:+d}"] = {"base_pz": b["mean_pz"], "mask_pz": m["mean_pz"],
                                "retention": m["mean_pz"] / max(b["mean_pz"], 1e-12),
                                "ce_base": bce, "ce_mask": mce,
                                "ce_cost": mce - bce}
            log(f"[{tag}] mask@g{j:+d}: {b['mean_pz']:.4f} -> {m['mean_pz']:.4f} "
                f"(x{mask[f'g{j:+d}']['retention']:.3f}) CE cost "
                f"{mce - bce:+.4f}")
        out["mask"] = mask

        ladder = []
        n0 = float(sd["wpe.weight"][0].norm())
        for tgt in LADDER_NORMS:
            row = sd["wpe.weight"][0] * (tgt / n0)
            sd_t, gate_t = modified_wpe(sd, 0, row)
            if not gate_t["pass"]:
                raise RuntimeError(f"ladder gate FAILED at norm {tgt}")
            net.load_state_dict(sd_t)
            ent = {"target_norm": tgt, "actual_norm": float(row.norm()),
                   "gate": gate_t}
            for j in (0, -12):
                ent[f"g{j:+d}"] = battery_cell(net, bat1[(j, "install60")],
                                               zid)["mean_pz"]
            ent["ce_r"] = ce_fixed_cpu(net, *r_eval_xy)
            ladder.append(ent)
            log(f"[{tag}] ladder |wpe0|={tgt:.2f}: g0 {ent['g+0']:.4f} "
                f"g-12 {ent['g-12']:.4f} CE {ent['ce_r']:.4f}")
        net.load_state_dict(sd)                    # restore
        out["ladder"] = ladder
        del net
        return out

    log("=" * 78)
    log("BEFORE battery (root: e131_consolidated_e113)")
    before = measure(sd_root, "before", full_f2=False)
    time.sleep(STAGGER_S)

    # instrument gates against e151's stored root cells
    G_ROW0 = {"mean": before["old_band"]["row0"]["mean"],
              "zero": before["old_band"]["row0"]["zero"],
              "ref_mean": G_R0_REF_MEAN, "ref_zero": G_R0_REF_ZERO,
              "pass": bool(abs(before["old_band"]["row0"]["mean"] - G_R0_REF_MEAN)
                           < G_FALLBACK_TOL
                           and abs(before["old_band"]["row0"]["zero"] - G_R0_REF_ZERO)
                           < G_FALLBACK_TOL)}
    G_A129 = {"mean": before["old_band"]["row129"]["mean"],
              "zero": before["old_band"]["row129"]["zero"],
              "ref_mean": G_A129_REF_MEAN, "ref_zero": G_A129_REF_ZERO,
              "pass": bool(abs(before["old_band"]["row129"]["mean"] - G_A129_REF_MEAN)
                           < G_FALLBACK_TOL
                           and abs(before["old_band"]["row129"]["zero"] - G_A129_REF_ZERO)
                           < G_FALLBACK_TOL)}
    G_DALL = {"dall_g0": before["del_table"]["d_all"]["g0"]["mean_pz"],
              "ref": G_DALL_REF,
              "pass": bool(abs(before["del_table"]["d_all"]["g0"]["mean_pz"]
                               - G_DALL_REF) < G_FALLBACK_TOL)}
    for gname, g in (("G_ROW0", G_ROW0), ("G_A129", G_A129), ("G_DALL", G_DALL)):
        log(f"{gname}: {'PASS' if g['pass'] else 'FAIL'}")
        if not g["pass"]:
            raise RuntimeError(f"{gname} failed vs stored root cells")
    log("gates: G_NONCE, G_SPLICE, G_GEO2, G_CONS, G_MASK, G_ROW0, G_A129, "
        "G_DALL all PASS")

    # ---------------- N2 rider BEFORE (instrument gate vs e160's stored cells)
    log("--- N2 rider on the BEFORE net (gate vs e160) ---")
    N2_HEADS = [(1, 0), (0, 0)]                    # e160's N2: {L1H0, L0H0}

    def n2_cell(net, heads, mode, f1_base_g0, f1_base_g12, f2_site_base=None,
                with_f2=False):
        mvcache = {}

        def mv(l, h):
            if (l, h) not in mvcache:
                mvcache[(l, h)] = head_mean_vec(net, l, h, r_eval_x)
            return mvcache[(l, h)]

        replace = {(l, h): (None if mode == "zero" else mv(l, h))
                   for (l, h) in heads}
        with HeadReplace(net, replace):
            c = {"heads": [f"L{l}H{h}" for (l, h) in heads], "mode": mode,
                 "f1_g0": battery_cell(net, bat1[(0, "install60")], zid)["mean_pz"],
                 "f1_gm12": battery_cell(net, bat1[(-12, "install60")], zid)["mean_pz"],
                 "ce_r": ce_fixed_cpu(net, *r_eval_xy)}
            c["f1_drop_g0"] = 1.0 - c["f1_g0"] / max(f1_base_g0, 1e-12)
            c["f1_drop_gm12"] = 1.0 - c["f1_gm12"] / max(f1_base_g12, 1e-12)
            c["ce_cost"] = c["ce_r"] - before["ce_r"] if not with_f2 else None
            if with_f2:
                sr = read_fact_at(net, pool2, name2_ids, mid, F2_ADDR_ROW,
                                  F2_Z_XCOL)
                srh = read_fact_at(net, pool2_held, name2_ids, mid,
                                   F2_ADDR_ROW, F2_Z_XCOL)
                c["f2_onset"] = sr["pz_onset_mean"]
                c["f2_span"] = sr["pname_mean_over7"]
                c["f2_onset_held"] = srh["pz_onset_mean"]
                c["f2_drop_onset"] = (1.0 - c["f2_onset"]
                                      / max(f2_site_base, 1e-12))
        return c

    n2_before = {
        "zero": n2_cell(net0, N2_HEADS, "zero", before["base"][0]["mean_pz"],
                        before["base"][-12]["mean_pz"]),
        "mean": n2_cell(net0, N2_HEADS, "mean", before["base"][0]["mean_pz"],
                        before["base"][-12]["mean_pz"]),
    }
    for c in n2_before.values():
        c.pop("ce_cost", None)
    G_N2 = {
        "zero_pz": n2_before["zero"]["f1_g0"], "ref_zero": G_N2Z_REF,
        "mean_pz": n2_before["mean"]["f1_g0"], "ref_mean": G_N2M_REF,
        "pass": bool(abs(n2_before["zero"]["f1_g0"] - G_N2Z_REF) < G_BIT_TOL
                     and abs(n2_before["mean"]["f1_g0"] - G_N2M_REF)
                     < G_BIT_TOL)}
    log(f"G_N2 before-net: zero {n2_before['zero']['f1_g0']:.10f} "
        f"(ref {G_N2Z_REF:.10f}) mean {n2_before['mean']['f1_g0']:.10f} "
        f"(ref {G_N2M_REF:.10f}): {'PASS' if G_N2['pass'] else 'FAIL'}")
    if not G_N2["pass"]:
        raise RuntimeError("N2 instrument drifted from e160's stored cells")
    time.sleep(STAGGER_S)

    # =====================================================================
    # INSTALL F2 (the ONE training; CPU; cooldown around it)
    # =====================================================================
    log("=" * 78)
    if not SMOKE:
        log(f"[cooldown] {COOLDOWN_S:.0f}s before the ONE training")
        cooldown(COOLDOWN_S)
    log(f"INSTALL F2: 300-step locked replay of {NAME2} at read rows 63..69 "
        f"(seed {F2_SEED}, device {DEV})")
    install = finetune_arm("installF2", net0, pool2, pool2_mask, anchor,
                           train_ids, r_eval_xy, f2_eval_ids, mid, F2_SEED)
    sd_after = install["sd"]
    if not SMOKE:
        log(f"[cooldown] {COOLDOWN_S:.0f}s after the ONE training")
        cooldown(COOLDOWN_S)
    save_ckpt("e154_twofacts", sd_after,
              {"desc": "e131_consolidated_e113 + 300-step locked install of "
                       f"{NAME2} (nonce) at read rows 63..69 (x-cols 64..70), "
                       "name-only mask, seed 10902",
               "steps": install["steps_ran"], "seed": F2_SEED,
               "base": f"runs/checkpoints/{ROOT_CK}"})
    time.sleep(STAGGER_S)

    log("=" * 78)
    log("AFTER battery (e154_twofacts)")
    after = measure(sd_after, "after", full_f2=True)
    time.sleep(STAGGER_S)

    # ---------------- N2 rider AFTER (price F1 and F2 separately)
    log("--- N2 rider on the AFTER net (F1 and F2 priced separately) ---")
    net_a = evl_load(sd_after)                       # the two-fact net
    f2_site_base = after["f2_site_install"]["pz_onset_mean"]
    n2_after = {
        "N2:top2-noL0H3/zero": n2_cell(net_a, N2_HEADS, "zero",
                                       after["base"][0]["mean_pz"],
                                       after["base"][-12]["mean_pz"],
                                       f2_site_base=f2_site_base, with_f2=True),
        "N2:top2-noL0H3/mean": n2_cell(net_a, N2_HEADS, "mean",
                                       after["base"][0]["mean_pz"],
                                       after["base"][-12]["mean_pz"],
                                       f2_site_base=f2_site_base, with_f2=True),
        "S:L1H0/zero": n2_cell(net_a, [N2_HEADS[0]], "zero",
                               after["base"][0]["mean_pz"],
                               after["base"][-12]["mean_pz"],
                               f2_site_base=f2_site_base, with_f2=True),
        "S:L0H0/zero": n2_cell(net_a, [N2_HEADS[1]], "zero",
                               after["base"][0]["mean_pz"],
                               after["base"][-12]["mean_pz"],
                               f2_site_base=f2_site_base, with_f2=True),
    }
    for c in n2_after.values():
        c["ce_cost"] = c["ce_r"] - after["ce_r"]
        c["f1_kills60"] = bool(c["f1_drop_g0"] >= KILL_DROP)
        c["f2_kills60"] = bool(c.get("f2_drop_onset", 0.0) >= KILL_DROP)
        log(f"  [{c['heads']} {c['mode']:4s}] F1 g0 {c['f1_g0']:.4f} "
            f"(drop {100 * c['f1_drop_g0']:.1f}%) F2 onset {c['f2_onset']:.4f} "
            f"(drop {100 * c['f2_drop_onset']:.1f}%) CE {c['ce_cost']:+.4f}")
    time.sleep(STAGGER_S)

    # ---------------- W017 rider: F2's 36-head census (diffuse vs concentrated)
    log("--- W017 rider: F2 head census (36 heads, zero mode) ---")
    census_base = read_fact_at(net_a, pool2_held, name2_ids, mid, F2_ADDR_ROW,
                               F2_Z_XCOL)
    w017_base = {"onset": census_base["pz_onset_mean"],
                 "span": census_base["pname_mean_over7"],
                 "ce_r": ce_fixed_cpu(net_a, *r_eval_xy)}
    head_rows = ([(l, h) for l in range(2) for h in range(6)]
                 if SMOKE else
                 [(l, h) for l in range(6) for h in range(6)])
    w017_cells = []
    for (l, h) in head_rows:
        with HeadReplace(net_a, {(l, h): None}):
            sr = read_fact_at(net_a, pool2_held, name2_ids, mid, F2_ADDR_ROW,
                              F2_Z_XCOL)
            ce = ce_fixed_cpu(net_a, *r_eval_xy)
        c = {"head": f"L{l}H{h}", "onset": sr["pz_onset_mean"],
             "span": sr["pname_mean_over7"], "ce_r": ce,
             "drop": w017_base["onset"] - sr["pz_onset_mean"],
             "ce_cost": ce - w017_base["ce_r"]}
        w017_cells.append(c)
    drops_pos = [max(c["drop"], 0.0) for c in w017_cells]
    tot = sum(drops_pos) if sum(drops_pos) > 0 else 1.0
    shares = sorted((d / tot for d in drops_pos), reverse=True)
    ent = float(-sum(s * np.log(s) for s in shares if s > 0))
    ranked = sorted(w017_cells, key=lambda c: -c["drop"])
    w017 = {
        "readout": "F2 site onset on the held30 pool (e133 site-column "
                   "convention); zero mode; CE on the e065 bank",
        "base": w017_base,
        "cells": w017_cells,
        "top1_share": shares[0], "top2_share": shares[0] + shares[1],
        "n_positive": int(sum(1 for d in drops_pos if d > 0)),
        "entropy_nat": ent, "effective_heads": float(np.exp(ent)),
        "top5": [{"head": c["head"], "drop": c["drop"],
                  "share": max(c["drop"], 0.0) / tot} for c in ranked[:5]],
        "refs_w017_card": {"top1": {"sink_coupled_F1": W017_TOP1_F1,
                                    "site_stored": W017_TOP1_SITE,
                                    "install": W017_TOP1_INST},
                           "top2": {"sink_coupled_F1": 0.467,
                                    "site_stored": 0.389, "install": 0.224}},
        "verdict": ("DIFFUSE-AS-PREDICTED" if shares[0] < W017_TOP1_F1
                    else "CONCENTRATED"),
    }
    log(f"W017: top-1 share {shares[0]:.3f} (F1 stored {W017_TOP1_F1}, site "
        f"{W017_TOP1_SITE}, install {W017_TOP1_INST}) | top-2 "
        f"{shares[0] + shares[1]:.3f} | n_pos {w017['n_positive']} | "
        f"H {ent:.3f} (eff {np.exp(ent):.1f}) -> {w017['verdict']}; "
        f"top-5: " + ", ".join(f"{c['head']} {c['drop']:.3f}"
                               for c in ranked[:5]))
    del net_a
    time.sleep(STAGGER_S)

    # =====================================================================
    # ADJUDICATION (registered clauses, verbatim; no bar shopping)
    # =====================================================================
    g12b = before["base"][-12]["mean_pz"]
    g12a = after["base"][-12]["mean_pz"]
    g0b = before["base"][0]["mean_pz"]
    g0a = after["base"][0]["mean_pz"]
    door_drop = 1.0 - g12a / max(g12b, 1e-12)
    ret = {j: after["base"][j]["mean_pz"] / max(before["base"][j]["mean_pz"],
                                                1e-12) for j in GEOS}
    ret_g0 = ret[0]

    cond = {
        "GLOBAL_PHASE": {
            "door_drop": door_drop, "bar": GLOBAL_DROP_BAR,
            "g0_after": g0a, "g0_floor": F1_G0_FLOOR, "g0_retention": ret_g0,
            "fires": bool(door_drop >= GLOBAL_DROP_BAR and g0a >= F1_G0_FLOOR)},
        "PER_MEMORY": {
            "door_drop": door_drop, "tol": PERMEM_TOL,
            "fires": bool(abs(door_drop) <= PERMEM_TOL)},
        "CROSSOVER": {
            "door_drop": door_drop, "lo": XOVER_LO, "hi": GLOBAL_DROP_BAR,
            "fires": bool(XOVER_LO <= door_drop < GLOBAL_DROP_BAR)},
    }
    if cond["GLOBAL_PHASE"]["fires"]:
        verdict = "GLOBAL-PHASE"
        clause = (f"F1 g-12 dropped {100 * door_drop:.1f}% ({g12b:.4f} -> "
                  f"{g12a:.4f}, bar >= 70%) while F1's own trained-geometry "
                  f"expression stayed >= 0.5 (g0 {g0a:.4f}, retention "
                  f"{ret_g0:.3f}) — the door closed on an untouched fact; "
                  f"T097's one-event reading generalizes across facts.")
    elif cond["PER_MEMORY"]["fires"]:
        verdict = "PER-MEMORY"
        clause = (f"F1 g-12 within 15% ({100 * door_drop:+.1f}%; {g12b:.4f} -> "
                  f"{g12a:.4f}) — e151/e158 were self-conversion; the "
                  f"abstract's 'globally' rewrites; doors are per-fact.")
    elif cond["CROSSOVER"]["fires"]:
        verdict = "CROSSOVER"
        clause = (f"F1 g-12 dropped {100 * door_drop:.1f}% ({g12b:.4f} -> "
                  f"{g12a:.4f}; the 30-70% band) — shared substrate with "
                  f"capacity (feeds W012's contended-bandwidth reading).")
    else:
        verdict = "TEXTURE"
        if door_drop >= GLOBAL_DROP_BAR:
            clause = (f"door_drop {100 * door_drop:.1f}% >= 70% BUT F1's own "
                      f"trained-geometry expression fell below 0.5 (g0 "
                      f"{g0a:.4f}, retention {ret_g0:.3f}) — the fact was "
                      f"damaged everywhere, not door-specifically; numbers "
                      f"reported, no bar shopping.")
        else:
            clause = (f"door_drop {100 * door_drop:+.1f}% lands in no "
                      f"registered band (gap (15,30)%% or a gain); F1 g-12 "
                      f"{g12b:.4f} -> {g12a:.4f}, g0 {g0b:.4f} -> {g0a:.4f} "
                      f"— TEXTURE with numbers.")
    log("=" * 78)
    log(f"E154 VERDICT: {verdict}")
    log(f"  deciding: F1 g-12 {g12b:.4f} -> {g12a:.4f} (drop "
        f"{100 * door_drop:.1f}%) | F1 g0 {g0b:.4f} -> {g0a:.4f} (ret "
        f"{ret_g0:.3f}) | F1 g+12 {before['base'][12]['mean_pz']:.4f} -> "
        f"{after['base'][12]['mean_pz']:.4f}")
    log(f"  F2 graft: site63 span {before['site63_span']['site_strength']:+.4f}"
        f" -> {after['site63_span']['site_strength']:+.4f} (2x-ctrl "
        f"{after['site63_span']['bar_2x_control']:.4f}) site_pos "
        f"{before['site63_span']['site_pos']} -> "
        f"{after['site63_span']['site_pos']} | F2 expression onset "
        f"{before['f2_site_install']['pz_onset_mean']:.4f} -> "
        f"{after['f2_site_install']['pz_onset_mean']:.4f}")
    log(f"  F1 dials: A(129) {before['old_band']['A129']:+.4f} -> "
        f"{after['old_band']['A129']:+.4f} | row0 S "
        f"{before['old_band']['row0_strength']:+.4f} -> "
        f"{after['old_band']['row0_strength']:+.4f} | D-all g0 "
        f"{before['del_table']['d_all']['g0']['mean_pz']:.4f} -> "
        f"{after['del_table']['d_all']['g0']['mean_pz']:.4f} | CE_R "
        f"{before['ce_r']:.4f} -> {after['ce_r']:.4f}")
    log(f"  {clause}")
    log("=" * 78)

    # rider adjudication (zero mode primary)
    n2z = n2_after["N2:top2-noL0H3/zero"]
    rider = {
        "primary_cell": "N2:top2-noL0H3/zero (e160's N2, zero mode primary)",
        "f1_drop_g0": n2z["f1_drop_g0"], "f2_drop_onset": n2z["f2_drop_onset"],
        "kill_bar": KILL_DROP,
        "cells": n2_after, "before_gate_cells": n2_before,
        "SHARED_READOUT": {"fires": bool(n2z["f1_drop_g0"] >= KILL_DROP
                                         and n2z["f2_drop_onset"] >= KILL_DROP)},
        "PER_FACT_READOUT": {"fires": bool(n2z["f2_drop_onset"] >= KILL_DROP
                                           and n2z["f1_drop_g0"] < KILL_DROP)},
        "F1_ONLY_KILL_texture": {"fires": bool(
            n2z["f1_drop_g0"] >= KILL_DROP
            and n2z["f2_drop_onset"] < KILL_DROP)},
    }
    if rider["SHARED_READOUT"]["fires"]:
        rider["verdict"] = ("SHARED-READOUT — N2 kills both facts jointly "
                            f"(F1 {100 * n2z['f1_drop_g0']:.1f}%, F2 "
                            f"{100 * n2z['f2_drop_onset']:.1f}% at CE "
                            f"{n2z['ce_cost']:+.3f})")
    elif rider["PER_FACT_READOUT"]["fires"]:
        rider["verdict"] = ("PER-FACT-READOUT — N2 kills F2 while F1 holds "
                            f"(F2 {100 * n2z['f2_drop_onset']:.1f}%, F1 "
                            f"{100 * n2z['f1_drop_g0']:.1f}% at CE "
                            f"{n2z['ce_cost']:+.3f})")
    elif rider["F1_ONLY_KILL_texture"]["fires"]:
        rider["verdict"] = ("TEXTURE (F1-ONLY-KILL, the e160-consistent "
                            "type-selective shape): N2 kills F1 "
                            f"({100 * n2z['f1_drop_g0']:.1f}%) while F2 holds "
                            f"({100 * n2z['f2_drop_onset']:.1f}%) at CE "
                            f"{n2z['ce_cost']:+.3f} — readouts are per-fact "
                            "with F1's circuit intact; not a registered bar")
    else:
        rider["verdict"] = (f"TEXTURE — neither fact killed by N2/zero (F1 "
                            f"{100 * n2z['f1_drop_g0']:.1f}%, F2 "
                            f"{100 * n2z['f2_drop_onset']:.1f}%); numbers "
                            f"reported, no bar shopping")
    log(f"RIDER (N2, zero primary): {rider['verdict']}")
    log(f"W017 RIDER: {w017['verdict']} (top-1 {w017['top1_share']:.3f} vs "
        f"F1's stored {W017_TOP1_F1})")

    # ---------------- outputs
    metrics = {
        "experiment": "e154_twofacts",
        "date": common.now_iso(),
        "registration": ("QUEUE.md row e154 (rewrites e134) + the dispatch's "
                         "registered prediction VERBATIM; operationalizations "
                         "frozen in the module docstring before compute"),
        "registered_prediction": REGISTERED_PREDICTION,
        "committed_prediction": REGISTERED_PREDICTION["committed"],
        "prediction_held": None,
        "question": ("does building a SECOND fact's locked graft (F2 = "
                     "MIRABEL, nonce, site rows 63..69) tear down the FIRST "
                     "fact's travel (F1 = ZEPHYRA, sink-coupled, g-12 0.916)? "
                     "GLOBAL-PHASE vs PER-MEMORY vs CROSSOVER"),
        "root": f"runs/checkpoints/{ROOT_CK} (loaded, gated vs e131/e151/"
                "e150/e160 cells)",
        "f2": {"name": NAME2, "kind": "nonce (e134 design)",
               "corpus_occurrences": G_NONCE["corpus_occurrences"],
               "onset_char": "M", "onset_char_id": mid,
               "install": {"desc": "e151 locked-replay protocol verbatim at "
                                   "offset -66: MIRABEL locked at x-cols "
                                   "64..70, onset read row 63, name-only "
                                   "7-target mask, zero position variance",
                           "recipe": "batch 32 = 16 pool + 16 anchors (8 "
                                     "paired + 8 random), token-weighted "
                                     "union CE, AdamW (0.9,0.95) wd 0.1, "
                                     "lr 1e-3 constant, clip 1.0",
                           "steps_ran": install["steps_ran"],
                           "seed": F2_SEED, "traj": install["traj"],
                           "device": str(DEV), "time_cap_s": FT_TIME_CAP,
                           "cooldown_s": COOLDOWN_S},
               "pre_install_site_error": before["f2_site_install"],
               },
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
                     "f2_placement": {"offset": F2_J,
                                      "name_xcols": [F2_Z_XCOL,
                                                     F2_Z_XCOL + 6],
                                      "read_rows": [63, 69],
                                      "site_band": list(F2_ROWS),
                                      "continuation_len": F2_CONT},
                     "mask": "7 name-char targets per window (name-only)",
                     "geometries_measured": {"novel": [-12, 12],
                                             "trained_g0": 0,
                                             "note": "root trained "
                                                     "{-8,-4,0,+4,+8}; +-12 "
                                                     "novel for it"}},
        "gates": {"G_NONCE": G_NONCE, "G_SPLICE": G_SPLICE, "G_GEO2": G_GEO2,
                  "G_CONS": G_CONS, "G_MASK": G_MASK, "G_ROW0": G_ROW0,
                  "G_A129": G_A129, "G_DALL": G_DALL, "G_N2": G_N2,
                  "G_SURG": gates_surg},
        "before": before, "after": after,
        "retentions": {f"g{j:+d}": ret[j] for j in GEOS},
        "door_drop": door_drop,
        "deltas": {
            "A129": after["old_band"]["A129"] - before["old_band"]["A129"],
            "row0_strength": (after["old_band"]["row0_strength"]
                              - before["old_band"]["row0_strength"]),
            "site63_span": (after["site63_span"]["site_strength"]
                            - before["site63_span"]["site_strength"]),
            "d_all_g0": (after["del_table"]["d_all"]["g0"]["mean_pz"]
                         - before["del_table"]["d_all"]["g0"]["mean_pz"]),
            "dF2_g0": (after["del_table"]["dF2"]["g0"]["mean_pz"]
                       - before["del_table"]["dF2"]["g0"]["mean_pz"]),
            "ce_r": after["ce_r"] - before["ce_r"],
        },
        "adjudication": {"conditions": cond, "verdict": verdict,
                         "clause": clause},
        "n2_rider": rider,
        "w017_rider": w017,
        "honesty_reflex": {
            "f2_choice": "MIRABEL is a nonce (0 corpus occurrences; the "
                "harness's JULIET is a natural corpus name known to base "
                "nets — a novelty confound under the e131 root, and its "
                "e044 checkpoints are B43-lineage, unusable here). F2's "
                "error statistics: pre-install root site read "
                f"p(M)@63 = {before['f2_site_install']['pz_onset_mean']:.4f} "
                f"(span {before['f2_site_install']['pname_mean_over7']:.4f}); "
                "post-install "
                f"{after['f2_site_install']['pz_onset_mean']:.4f} "
                "(the full traj is in f2.install.traj). The install learned "
                "the nonce at the site; generalization beyond the site is "
                "the f2_site_held cell.",
            "single_seed_single_lineage": "one install (seed 10902) on one "
                "root (e131 consolidated, one seed) — n=1; the fork verdict "
                "is lineage-specific until replicated (e157 class)",
            "site_placement": "F2's site rows 63..69 carry only 64 tokens of "
                "pre-context (vs g0's 130 — the NEAR mirror of e151's FAR "
                "confound) AND sit inside the causal past of F1's battery "
                "reads (positions 117..143 attend keys 63..69): graft-"
                "interference with F1's read is real, priced by the dF2 "
                "deletion arm; F1's battery text never contains MIRABEL",
            "install_budget": "300 locked steps vs the root's 300 jittered + "
                "4000-step pretraining + install — a fine-tune of a "
                "consolidated net (recorded, not a bar)",
            "novel_geometry_caveat_T087": "F1 g-12 retention conflates "
                "'the route survived' with 'sink health at the novel "
                "geometry survived' (T087/E150 caveat); the mask/ladder "
                "columns separate the two only partially",
            "w017_convention": "F2's census readout is the held30-pool site "
                "onset (out-of-install-host generalization); shares over "
                "positive drops only — comparable to e133's stored "
                "conventions but not identical batteries",
        },
        "trims": trims, "deviations": deviations,
        "ckpt_inventory": CKPT_INVENTORY,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": int(net0.num_params()),
                   "train_device": str(DEV), "eval_device": "cpu",
                   "torch_threads": torch.get_num_threads(), "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "twofacts.png", before, after, ret, door_drop, verdict, clause,
         cond, n2_after, rider, w017)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'twofacts.png'}, "
        f"ckpt runs/checkpoints/e154_twofacts.pt")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot

def plot(path, before, after, ret, door_drop, verdict, clause, cond,
         n2_after, rider, w017):
    """Legible two-fact dials + riders."""
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.5))
    GEOS = (-12, 0, 12)

    # (0,0) F1 door dials before/after + the three registered zones
    ax = axes[0, 0]
    xs = np.arange(3)
    lbls = ["g-12 (NOVEL,\nthe door)", "g0 (trained,\ninterference ctl)",
            "g+12 (NOVEL)"]
    for k, j in enumerate(GEOS):
        b = before["base"][j]["mean_pz"]
        a = after["base"][j]["mean_pz"]
        ax.bar(k - 0.19, b, 0.36, color="steelblue", edgecolor="k", lw=0.5,
               label="before (root)" if k == 0 else None)
        ax.bar(k + 0.19, a, 0.36, color="crimson", edgecolor="k", lw=0.5,
               label="after (F2 installed)" if k == 0 else None)
        ax.text(k - 0.19, b + 0.012, f"{b:.3f}", ha="center", fontsize=8.5)
        ax.text(k + 0.19, a + 0.012, f"{a:.3f}", ha="center", fontsize=8.5)
        if j == -12:
            for frac, col, tag in ((0.30, "firebrick", "GLOBAL bar -70%"),
                                   (0.70, "darkgoldenrod", "CROSSOVER hi"),
                                   (0.85, "seagreen", "PER-MEM floor -15%")):
                ax.plot([k - 0.30, k + 0.30], [frac * b] * 2, ls="--", lw=1.1,
                        color=col)
                ax.text(k + 0.31, frac * b, tag, fontsize=6.4, color=col,
                        va="center")
            ax.text(k + 0.19, 0.45, f"x{ret[j]:.2f}", ha="center", fontsize=10,
                    fontweight="bold",
                    color="firebrick" if ret[j] <= 0.30 else "dimgray")
    ax.set_xticks(xs)
    ax.set_xticklabels(lbls, fontsize=8.5)
    ax.set_ylabel("F1 battery p(Z) install-60")
    ax.set_ylim(0, 1.12)
    ax.set_title(f"(i) THE DOOR — F1 travel before/after F2's install\n"
                 f"door_drop {100 * door_drop:.1f}% -> {verdict}", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower right")

    # (0,1) F2 site census spectrum
    ax = axes[0, 1]
    for tag, d, col, mk in (("before", before, "steelblue", "o"),
                            ("after", after, "crimson", "s")):
        cen = d["censusF2_span"]["rows"]
        xs_r = [int(r) for r in cen]
        ax.plot(xs_r, [cen[str(r)]["strength"] for r in xs_r], f"{mk}-",
                ms=4, lw=1.2, color=col, alpha=0.9, label=f"{tag} (span)")
        cm = d["site63_span"]["control_max"]
        ax.axhline(2.0 * cm, ls=":", lw=1.0, color=col, alpha=0.7,
                   label=f"{tag} 2x-ctrl {2.0 * cm:.4f}")
    if "censusF2_onset" in after:
        ceno = after["censusF2_onset"]["rows"]
        xs_o = [int(r) for r in ceno]
        ax.plot(xs_o, [ceno[str(r)]["strength"] for r in xs_o], "^--", ms=3,
                lw=0.9, color="crimson", alpha=0.4, label="after (onset)")
    ax.axvspan(63, 69, color="crimson", alpha=0.07)
    ax.axvspan(121, 137, color="seagreen", alpha=0.05)
    ax.axvspan(183, 189, color="gray", alpha=0.05)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("wpe row (red: F2 band 63-69; green: F1 band; gray: 183)")
    ax.set_ylabel("strength = min(mean-drop, zero-drop)")
    ax.set_title(f"(F2) SITE-CONTENT TEST at rows 63..69 — the graft\n"
                 f"span {before['site63_span']['site_strength']:+.4f} -> "
                 f"{after['site63_span']['site_strength']:+.4f} | site_pos "
                 f"{before['site63_span']['site_pos']} -> "
                 f"{after['site63_span']['site_pos']} | F2 onset "
                 f"{before['f2_site_install']['pz_onset_mean']:.3f} -> "
                 f"{after['f2_site_install']['pz_onset_mean']:.3f}",
                 fontsize=9)
    ax.legend(fontsize=6.2)

    # (0,2) verdict panel
    ax = axes[0, 2]
    ax.axis("off")
    vlines = [
        "REGISTERED (QUEUE e154 / dispatch verbatim; no bar shopping):",
        "  GLOBAL-PHASE: g-12 drop >= 70% AND F1 g0 stays >= 0.5",
        "  PER-MEMORY: g-12 within 15%",
        "  CROSSOVER: 30-70% drop",
        "  Riders: SHARED-READOUT = N2 kills both; PER-FACT-READOUT =",
        "           N2 kills F2 while F1 holds",
        "",
        "DECIDING NUMBERS:",
        f"  F1 g-12: {before['base'][-12]['mean_pz']:.4f} -> "
        f"{after['base'][-12]['mean_pz']:.4f}  (door_drop "
        f"{100 * door_drop:.1f}%)",
        f"  F1 g0:   {before['base'][0]['mean_pz']:.4f} -> "
        f"{after['base'][0]['mean_pz']:.4f}  (ret {ret[0]:.3f})",
        f"  F1 g+12: {before['base'][12]['mean_pz']:.4f} -> "
        f"{after['base'][12]['mean_pz']:.4f}  (ret {ret[12]:.3f})",
        f"  F2 graft span: {before['site63_span']['site_strength']:+.4f} -> "
        f"{after['site63_span']['site_strength']:+.4f} (2x-ctrl "
        f"{after['site63_span']['bar_2x_control']:.4f})",
        f"  F2 onset: {before['f2_site_install']['pz_onset_mean']:.4f} -> "
        f"{after['f2_site_install']['pz_onset_mean']:.4f} | held "
        f"{after['f2_site_held']['pz_onset_mean']:.4f}",
        f"  A(129): {before['old_band']['A129']:+.4f} -> "
        f"{after['old_band']['A129']:+.4f}  row0 S: "
        f"{before['old_band']['row0_strength']:+.4f} -> "
        f"{after['old_band']['row0_strength']:+.4f}",
        f"  D-all g0: {before['del_table']['d_all']['g0']['mean_pz']:.4f} -> "
        f"{after['del_table']['d_all']['g0']['mean_pz']:.4f} | dF2 g0 "
        f"{after['del_table']['dF2']['g0']['mean_pz']:.4f} | CE_R "
        f"{before['ce_r']:.4f} -> {after['ce_r']:.4f}",
        "",
        f"N2 RIDER: {rider['verdict']}",
        f"W017 RIDER: {w017['verdict']} (top-1 {w017['top1_share']:.3f} vs "
        f"F1's stored 0.352)",
        "",
        f"VERDICT: {verdict}",
    ] + [f"  {wd}" for wd in
         [clause[i:i + 64] for i in range(0, len(clause), 64)]]
    for i, tx in enumerate(vlines):
        ax.text(0.02, 0.97 - i * 0.036, tx, fontsize=7.0, va="top",
                family="monospace")

    # (1,0) N2 rider bars
    ax = axes[1, 0]
    tags = list(n2_after.keys())
    f1d = [100 * n2_after[t]["f1_drop_g0"] for t in tags]
    f2d = [100 * n2_after[t].get("f2_drop_onset", 0.0) for t in tags]
    ys = np.arange(len(tags))
    ax.barh(ys - 0.19, f1d, 0.36, color="steelblue", edgecolor="k", lw=0.4,
            label="F1 drop (g0 battery)")
    ax.barh(ys + 0.19, f2d, 0.36, color="crimson", edgecolor="k", lw=0.4,
            label="F2 drop (site onset)")
    for y, t in zip(ys, tags):
        ax.text(102, y, f"CE {n2_after[t]['ce_cost']:+.3f}", va="center",
                fontsize=6.6)
    ax.axvline(100 * 0.60, ls="--", color="gray", lw=1.2, label="60% kill bar")
    ax.set_yticks(ys)
    ax.set_yticklabels([t.replace("N2:top2-noL0H3", "N2") for t in tags],
                       fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 135)
    ax.set_xlabel("fact drop (%) under the head-set — after net")
    ax.set_title(f"(rider) N2 = L1H0+L0H0, layer attribution: "
                 f"{rider['verdict'].split(' — ')[0]}", fontsize=9.5)
    ax.legend(fontsize=7, loc="lower right")

    # (1,1) W017 census distribution + e133 overlays
    ax = axes[1, 1]
    drops_pos = sorted((max(c["drop"], 0.0) for c in w017["cells"]),
                       reverse=True)
    tot = sum(drops_pos) if sum(drops_pos) > 0 else 1.0
    f2_curve = np.cumsum(drops_pos) / tot
    ax.plot(np.arange(1, len(f2_curve) + 1), f2_curve, "o-", ms=3, lw=1.4,
            color="crimson", label=f"F2 MIRABEL (top-1 "
            f"{w017['top1_share']:.3f})")
    try:
        e133 = json.loads(E133_METRICS.read_text(encoding="utf-8"))
        for nk, col, rtag in (("graduated", "steelblue", "F1 sink-coupled"),
                              ("site_locked", "seagreen", "arm_b site")):
            n = e133["nets"][nk]
            hd = []
            for t, v in n["arms"].items():
                if t.startswith("head_"):
                    key = ("std_install60" if nk == "graduated"
                           else "site_onset")
                    base = (n["base"]["std_install60"]["mean_pz"]
                            if nk == "graduated"
                            else n["base"]["site"]["onset_pz"])
                    hd.append(max(base - v[key], 0.0))
            s = sorted(hd, reverse=True)
            tt = sum(s) if sum(s) > 0 else 1.0
            ax.plot(np.arange(1, len(s) + 1), np.cumsum(s) / tt, "-", lw=1.0,
                    color=col, alpha=0.8, label=f"e133 {rtag}")
    except Exception as e:  # noqa: BLE001
        ax.text(0.3, 0.2, f"e133 overlay unavailable: {e}", fontsize=6.5)
    ax.axhline(w017["top1_share"], ls=":", color="crimson", lw=0.8)
    ax.axhline(1.0 / 36, ls=":", color="gray", lw=0.8)
    ax.set_xlabel("head rank (36 heads, zero mode)")
    ax.set_ylabel("cumulative load share")
    ax.set_title(f"(W017) F2's head census: {w017['verdict']}\n"
                 f"top-1 {w017['top1_share']:.3f} | top-2 "
                 f"{w017['top2_share']:.3f} | n_pos {w017['n_positive']} | "
                 f"H {w017['entropy_nat']:.3f} (eff "
                 f"{w017['effective_heads']:.1f})", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.25)

    # (1,2) deletion table + phase dials
    ax = axes[1, 2]
    dls = ("none", "d129", "d_all", "d_r0", "dF2")
    xs = np.arange(len(dls))
    for k, (tag, d, col) in enumerate((("before", before, "steelblue"),
                                       ("after", after, "crimson"))):
        vals = [d["del_table"][dl]["g0"]["mean_pz"] for dl in dls]
        ax.bar(xs + (k - 0.5) * 0.36, vals, 0.34, color=col, edgecolor="k",
               lw=0.4, label=tag)
        for x, v in zip(xs + (k - 0.5) * 0.36, vals):
            ax.text(x, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.6,
                    rotation=90, va="bottom")
    ax.text(1.62, 0.62,
            f"dF2 after: F1 g-12 {after['del_table']['dF2']['gm12']['mean_pz']:.3f}"
            f" | F2 onset {after['del_table']['dF2']['f2_onset']:.3f}"
            f" span {after['del_table']['dF2']['f2_span']:.3f}\n"
            f"mask@g-12 ret: bef "
            f"{before['mask']['g-12']['retention']:.3f} -> aft "
            f"{after['mask']['g-12']['retention']:.3f}\n"
            f"ladder@0.07 g-12: bef "
            f"{before['ladder'][0]['g-12']:.3f} -> aft "
            f"{after['ladder'][0]['g-12']:.3f}\n"
            f"ladder@0.15 g-12: bef "
            f"{before['ladder'][1]['g-12']:.3f} -> aft "
            f"{after['ladder'][1]['g-12']:.3f}\n"
            f"leaks after: p(M)@129 {after['f2_leak_g0']:.4f} | "
            f"p(Z)@63 {after['f1_leak_site']:.4f}",
            fontsize=6.8, family="monospace",
            bbox=dict(facecolor="lightyellow", alpha=0.9, edgecolor="gray"))
    ax.set_xticks(xs)
    ax.set_xticklabels(["none", "D129", "D-all\n{121..137}", "D-row-0",
                        "D-F2site\n{63..69}"], fontsize=8)
    ax.set_ylabel("F1 battery p(Z) g0 install-60")
    ax.set_ylim(0, 1.15)
    ax.set_title("(ii) deletions at g0 + F1 phase dials", fontsize=9.5)
    ax.legend(fontsize=7.5)

    fig.suptitle(f"E154 — TWO FACTS, ONE DOOR (root e131_consolidated + "
                 f"locked MIRABEL at rows 63..69) -> {verdict} | rider: "
                 f"{rider['verdict'].split(' — ')[0]} | W017: "
                 f"{w017['verdict']}", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
