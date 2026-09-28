"""E170 — THE ANCHOR-NEUTRAL SECOND INSTALL (QUEUE.md row e170; e154's
confound discharge; T100's reading (2) put to the test). e154's locked
install of the nonce fact MIRABEL at rows 63..69 ANNIHILATED F1 (0.785 ->
0.0017 at g0, everywhere) — but the e151-protocol's anchor bank uses
INCUMBENT-CONTINUATION host windows that actively CONTRADICT F1's home
expression: the 16 bank windows are the ORIGINAL host texts at the first
16 install positions (FLORIZEL/ELIZABETH at x-col 130 + their true
continuations), and the anchor loss is FULL CE — so 8 paired draws per
batch x 300 steps teach "after this exact pre-host context, predict F/E
(not Z)" at precisely the junctions where F1's battery reads p(Z). The
demolition may be anchor-driven unlearning-by-contradiction, with the
graft an innocent bystander. This cell reruns the IDENTICAL install with
NEUTRAL anchors and watches whether F1 survives.

ROOT (mandated): runs/checkpoints/e131_consolidated_e113.pt — F1 = ZEPHYRA,
the sink-coupled consolidated fact (gates below reproduce e131's/e151's/
e150's/e160's stored cells, bit-exact, before any compute is trusted).

================================= ANCHOR-DELTA ==============================
The ONLY change vs e154's protocol is the informational content of the
16-window paired-anchor bank. Everything else — corpus rebuild, install
pool, placement, mask, optimizer, step count, seed, eval schedule, the
8-paired + 8-random per-batch structure, and the RNG STREAM ITSELF — is
e154 verbatim (the finetune RNG draws have identical moduli: n_pool=60,
n_anc=16, len(train_ids); with seed 10902 the very same ix/aj/rj index
sequences are drawn; only anchor[aj]'s CONTENT differs).

e154 BANK (the confound, rebuilt here for the delta record):
    anchor = stack([train_ids[p - 130 : p - 130 + 256]
                    for p, _ in install_occ[:16]])
  = the first 16 install positions' ORIGINAL host windows: each contains
    the true host name (FLORIZEL/ELIZABETH) at x-cols 130..137/8 and its
    true 118/119-token continuation. Under full CE each window's
    prediction target at x-col 129 is the host ONSET char (F/E) after
    exactly the 130-token context on which F1's g0 battery reads p(Z) ->
    16/16 windows carry the anti-F1 contradiction; 8 draws/batch x 300
    steps = up to ~2400 junction exposures.

e170 BANK (neutral, this cell):
    16 PLAIN CORPUS windows of 256 tokens drawn from train_ids with a
    dedicated RNG (E170_ANCHOR_SEED = 170), REJECTING any candidate whose
    text [s, s+257) — the window PLUS its first prediction target —
    contains FLORIZEL, ELIZABETH, ZEPH or MIRABEL. No accepted window
    covers any context->host junction, so the contradiction signal at
    F1's readout positions is removed by construction; the windows remain
    genuine corpus text performing the anchor's actual job (holding
    general corpus behavior during the fine-tune).
  Budget identical: 16 windows x 256 tokens, full-token CE (no mask),
  8 paired + 8 random draws per batch of 32.
  The random-anchor channel (8/batch from raw train_ids) is e154
  VERBATIM and can itself rarely draw host-containing windows (~1-2% of
  draws, estimated and recorded below): that background is IDENTICAL in
  both cells and is not part of the delta.
============================================================================

INSTALL F2 (the ONE training): e151's locked-re-teach protocol VERBATIM
(= e143 = e109 arm-b = e119-L; the e154 clone): batch 32 = 16 install
windows + 16 anchors (8 paired + 8 random), token-level union CE on the
7 name-char targets (name-only mask on the install half; full CE on the
anchor half), AdamW (0.9,0.95) wd 0.1 constant lr 1e-3 clip 1.0, 300
steps, in-loop CPU evals every 25, seed 10902 (e151's locked seed).
PLACEMENT: MIRABEL locked at x-cols 64..70, onset read row 63, SITE BAND
rows 63..69 (j=-66), pre-context = the 64 TRUE train tokens before each
install host, continuation = the 185 true tokens after it.

MEASURE F1 BEFORE(root)/AFTER(e170_neutral.pt) — e154's identical
instruments: (i) the dial set g-12/g0/g+12 (install-60 battery) +
held30@g0 + CE_R; (ii) F1 phase dials: old-band census (rows 121..129 +
row 0) => A(129) + row-0 strength; deletion table {none, d129,
D-all{121,125,129,133,137}, d_r0, dF2site{63..69}} at g0 (dF2 also at
g-12); e150-informed probes: forced-off-sink MASK at g0/g-12 + NORM-
LADDER 0.07/0.15. MEASURE F2: site reads (onset p(M)@63 + name-span over
63..69; install pool primary + held30 texture); site-content census at
the F2 band (span + onset conventions, CTR2 controls); cross-leak cells.
The e154 AFTER numbers (runs/e154/metrics.json) are loaded at runtime
and reported alongside as the incumbent-anchor arm of the 2x2.

N2 RIDER (conditional, eval-only, e160's set): runs ONLY if F1 survives
per the registered clause (g0 >= 0.5 AND g-12 >= 0.5) — "does F1's
killable circuit persist alongside F2's graft?" N2 = {L1H0, L0H0} (zero +
mean modes; singles as texture), pricing F1 (g0 primary, g-12 co-report)
and F2 (site onset) separately. The BEFORE-net N2 instrument gate vs
e160's stored cells runs unconditionally. No registered rider bar in
e170 — informative cells only (e154's kills60 flags co-reported).

REGISTERED PREDICTION (QUEUE e170 VERBATIM; no bar shopping):
  - ANCHORS-DID-IT fires if: F1 survives (g0 >= 0.5 AND g-12 >= 0.5) —
    the demolition was the anchor channel (unlearning-by-contradiction);
    the two-facts question REOPENS with per-fact doors.
  - OVERWRITE-REAL fires if: F1 still dies (g0 <= 0.2) — capacity is one
    fact at this budget; W012's bandwidth answered the hard way.
  - No bar shopping; texture (partial survival, differential geometry
    damage) => TEXTURE with numbers.

OPERATIONALIZATIONS (frozen before compute):
  * g0/g-12 = mean p(Z) on the install-60 battery at j=0/-12, AFTER net.
  * Adjudication order: ANCHORS-DID-IT -> OVERWRITE-REAL -> TEXTURE (the
    first two are disjoint on g0: >= 0.5 vs <= 0.2; everything else —
    including g0 in (0.2, 0.5), or g0 >= 0.5 with g-12 < 0.5 — is
    TEXTURE with numbers).
  * N2 rider precondition = the ANCHORS-DID-IT survival clause itself.
  * e154 comparison cells are read from runs/e154/metrics.json at
    runtime (full precision); the BEFORE battery must reproduce e154's
    stored before-values (same root, same instruments — recorded as a
    sanity note, gated by G_CONS bit-exactness).

INSTRUMENT PROVENANCE: load_cpu/evl_load/battery_cell/battery_pz/
ce_fixed_cpu/val_windows/deleted_wpe/modified_wpe/read_fact_at/row_census/
finetune_arm are lab/e154_twofacts.py VERBATIM (which are e151/e143/e150
verbatim = the e065/e068/e109/e113/e116/e119/e131 lineage, GPU gate
already removed there); battery_fwd/ce_fwd/forward_custom/fwd_causal/
fwd_offsink are e150's VERBATIM (via e154); head_mean_vec/HeadReplace are
lab/e160_headset_escalation.py VERBATIM (via e154). Copied, not imported,
per lab convention.

COMPUTE ENVELOPE: CPU-ONLY MANDATED (CUDA_VISIBLE_DEVICES=-1 before torch
import; e152R owns the GPU; e161 co-runs on CPU) — torch threads 4, all
evals sequential, 3 s staggers between phases, cooldown(60) around the
ONE training, no busy-waiting (no single wait exceeds 180 s: cooldown 60
s, staggers 3 s); training cap 1500 s (e151's CPU-fallback convention).
Single seed, single lineage.

Outputs: runs/e170/{metrics.json, anchor_neutral.png}; the two-fact net
runs/checkpoints/e170_neutral.pt (ckpt_inventory in metrics). No NOTES/
THINKING/QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python e170_anchor_neutral.py    (E170_SMOKE=1 shakedown)
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

torch.set_num_threads(4)                              # LOW (e161 co-runs)

import torch.nn.functional as F                       # noqa: E402

import common                                          # noqa: E402
common.DEVICE = "cpu"
from common import (Cfg, CharCorpus, TinyGPT, cooldown, run_dir,    # noqa: E402
                    save_json)
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E170_SMOKE") == "1"
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
E154_METRICS = E43.REPO / "runs" / "e154" / "metrics.json"   # the confound arm

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

# ---- fine-tune envelope (e109 arm-b / e119-L / e143 / e151 / e154 verbatim) ----
FT_LR = 1e-3
FT_STEPS = 300 if not SMOKE else 8
FT_TIME_CAP = 1500.0                 # CPU cap (e151's CPU-fallback convention)
EVAL_EVERY = 25 if not SMOKE else 2
NAME_BS, ANCH_BS = 16, 16
F2_SEED = 10902                      # e119-L / e143 locked / e151 / e154's seed
COOLDOWN_S = 60.0                    # CPU envelope around the ONE training
STAGGER_S = 3.0                      # between phases (e160 convention)

# ---- neutral anchor bank (THE delta; see ANCHOR-DELTA in the docstring) -------
E170_ANCHOR_SEED = 170               # dedicated RNG for the plain-corpus bank
ANCHOR_FORBIDDEN = ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")

# ---- gates / references (full precision, from stored metrics) -----------------
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

# ---- registered bar constants (frozen; QUEUE e170 verbatim) --------------------
SURVIVE_FLOOR = 0.50                 # ANCHORS-DID-IT clause (g0 AND g-12 >= .5)
OVERWRITE_CEIL = 0.20                # OVERWRITE-REAL clause (g0 <= 0.2)
KILL_DROP = 0.60                     # e160's KILL_DROP (informative rider flags)
SITE_CTRL_MULT = 2.0                 # e139 site convention: >= 2x control-max

REGISTERED_PREDICTION = {
    "anchors_did_it": "ANCHORS-DID-IT fires if: F1 survives (g0 >= 0.5 AND "
        "g-12 >= 0.5) — the demolition was the anchor channel (unlearning-by-"
        "contradiction); the two-facts question REOPENS with per-fact doors.",
    "overwrite_real": "OVERWRITE-REAL fires if: F1 still dies (g0 <= 0.2) — "
        "capacity is one fact at this budget; W012's bandwidth answered the "
        "hard way.",
    "no_bar_shopping": "No bar shopping; texture (partial survival, "
        "differential geometry damage) => TEXTURE with numbers.",
    "operationalizations": "g0/g-12 = mean p(Z) on the install-60 battery at "
        "j=0/-12 on the AFTER net; order ANCHORS-DID-IT -> OVERWRITE-REAL -> "
        "TEXTURE (clauses disjoint on g0: >= 0.5 vs <= 0.2; everything else — "
        "g0 in (0.2, 0.5), or g0 >= 0.5 with g-12 < 0.5 — is TEXTURE with "
        "numbers); N2 rider precondition = the ANCHORS-DID-IT survival clause "
        "itself (informative cells, no registered rider bar in e170; e160's "
        "kills60 flags co-reported).",
    "committed": "The dispatch registered BOTH forks explicitly (T100's (1) "
                 "vs (2)); adjudicated against exactly these bars.",
}

trims: list[str] = []
deviations: list[str] = [
    "Nets are the mandated 2.7M e131_consolidated line (e143/e154's precedent "
    "note; the dispatch's '<=1M family' phrase is an envelope statement — "
    "every gate reference and lineage number of this cell lives on the 2.7M "
    "line, and the mandated root IS e131_consolidated_e113.pt).",
    "THE ANCHOR-DELTA (the only protocol change vs e154): the 16-window "
    "paired-anchor bank rebuilt as PLAIN CORPUS windows (seed 170, rejection "
    "on host/nonce content in [s, s+257) — window plus first target). "
    "Budget/structure/RNG-stream identical; see the ANCHOR-DELTA docstring "
    "section and the anchor_delta record in metrics.json.",
    "The random-anchor channel (8/batch from raw train_ids, inside "
    "finetune_arm) is e154 VERBATIM — it can rarely draw host-containing "
    "windows (background rate estimated and recorded); identical in both "
    "cells, hence not part of the delta.",
    "N2 rider is CONDITIONAL on the registered survival clause (g0 >= 0.5 AND "
    "g-12 >= 0.5); if F1 does not survive, the rider is skipped and recorded "
    "as such (its question — does F1's killable circuit persist alongside "
    "F2's graft — is moot). No W017 rider (not dispatched for e170).",
    "CPU-ONLY mandated (e152R owns the GPU; e161 co-runs on CPU): training "
    "under the 1500 s CPU cap (e151's fallback convention), threads 4, "
    "sequential evals, 3 s staggers, cooldown(60) around the ONE training; "
    "'<=180 s cap' read as the no-busy-waiting bound on any single wait "
    "(cooldown 60 s, staggers 3 s — compliant).",
    "Single seed (10902), single lineage — no replication arm (dispatch "
    "envelope: ONE training).",
    "Smoke mode trims: 8-step install, reduced census rows; nothing "
    "adjudicated.",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/e154_twofacts.py VERBATIM (= e151/e143/e150/e160 lineage)
# with the GPU gate already removed there; head_mean_vec/HeadReplace are
# e160's VERBATIM (via e154).

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
    """e151's/e154's finetune_arm VERBATIM (e109 arm-b / e119-L recipe):
    batch 32 = 16 install windows + 16 anchors (8 paired from the bank +
    8 random corpus windows); token-level union CE on the name-char targets
    (install half) + full CE (anchor half); AdamW (0.9,0.95) wd 0.1 lr 1e-3
    constant clip 1.0; 300 steps / CPU cap; in-loop evals every 25 (evals
    consume no RNG). The RNG stream is IDENTICAL to e154's given seed 10902
    (same draw shapes and moduli: n_pool=60, n_anc=16, len(train_ids)) —
    only the CONTENT of anchor[aj] differs (the neutral bank)."""
    net = copy.deepcopy(net0).to(DEV)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    n_pool = pool_x.shape[0]
    n_anc = anchor.shape[0]
    assert n_anc == 16, f"anchor bank must stay 16 windows (got {n_anc})"
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
    torch.save({"model": sd, "meta": {"experiment": "e170", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name} "
        f"({', '.join(f'{k}={v}' for k, v in meta.items())})")


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e170_smoke" if SMOKE else "e170")
    log(f"E170 ANCHOR-NEUTRAL SECOND INSTALL (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY mandated, torch threads {torch.get_num_threads()}, "
        f"cuda available = {torch.cuda.is_available()}")

    # ---------------- protocol rebuild (e143/e151/e154 verbatim)
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

    # =====================================================================
    # THE ANCHOR-DELTA: e154's incumbent-continuation bank (rebuilt for the
    # record) vs e170's NEUTRAL plain-corpus bank
    # =====================================================================
    # e154 bank verbatim: first 16 install-position ORIGINAL host windows
    anchor_e154 = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                               for p, _ in install_occ[:16]])
    n_e154_hostwins = sum(
        1 for p, _ in install_occ[:16]
        if any(f in train_text[p - PRE: p - PRE + BLOCK + 1]
               for f in ANCHOR_FORBIDDEN[:2]))

    # e170 neutral bank: plain corpus windows, rejection on host/nonce content
    arng = random.Random(E170_ANCHOR_SEED)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        tries += 1
        txt = train_text[s: s + BLOCK + 1]      # window + FIRST TARGET token
        if any(f in txt for f in ANCHOR_FORBIDDEN):
            rejections += 1
            continue
        n_starts.append(s)
    if len(n_starts) != 16:
        raise RuntimeError(f"neutral bank incomplete: {len(n_starts)}/16")
    anchor = torch.stack([train_ids[s: s + BLOCK] for s in n_starts])

    # junction accounting: a window "covers" a context->host junction if its
    # [s, s+257) span contains a host occurrence's onset position p (the
    # window then trains the host-onset target at/after that context).
    host_positions = [p for p in E43.find_occ(train_text, HOSTS[0])]
    host_positions += [p for p in E43.find_occ(train_text, HOSTS[1])]

    def junctions_covered(starts):
        cov = 0
        for s in starts:
            if any(s <= p < s + BLOCK + 1 for p in host_positions):
                cov += 1
        return cov

    jc_e154 = junctions_covered([p - PRE for p, _ in install_occ[:16]])
    jc_e170 = junctions_covered(n_starts)
    host_occ_total = len(host_positions)
    bg_rate = host_occ_total * (BLOCK + 1) / len(train_ids)   # random channel
    G_ANCHOR = {
        "e170_bank": {
            "construction": ("16 plain corpus windows from train_ids, RNG seed "
                             f"{E170_ANCHOR_SEED}, rejection if [s, s+257) "
                             "contains FLORIZEL/ELIZABETH/ZEPH/MIRABEL"),
            "n_windows": 16, "block": BLOCK, "seed": E170_ANCHOR_SEED,
            "starts": n_starts, "tries": tries, "rejections": rejections,
            "forbidden": list(ANCHOR_FORBIDDEN),
            "windows_with_host_content": sum(
                1 for s in n_starts
                if any(f in train_text[s: s + BLOCK + 1]
                       for f in ANCHOR_FORBIDDEN)),
            "junctions_covered": jc_e170,
            "overlaps_e154_bank_windows": sum(
                1 for s in n_starts
                if any(abs(s - (p - PRE)) < BLOCK for p, _ in install_occ[:16])),
        },
        "e154_bank": {
            "construction": ("first 16 install-position ORIGINAL host windows "
                             "(incumbent continuations, no name swap) — "
                             "train_ids[p-130 : p+126]"),
            "n_windows": 16,
            "windows_with_host_content": n_e154_hostwins,
            "junctions_covered": jc_e154,
        },
        "budget_identical": bool(anchor.shape == anchor_e154.shape),
        "rng_stream_identical_to_e154": True,
        "rng_note": ("finetune_arm verbatim; draw shapes/moduli identical "
                     "(n_pool=60, n_anc=16, len(train_ids)); seed 10902 — the "
                     "same ix/aj/rj sequences as e154; only anchor[aj] content "
                     "differs"),
        "random_channel_background": {
            "host_occurrences_in_train": host_occ_total,
            "est_window_hit_rate": bg_rate,
            "note": ("the 8-per-batch random corpus anchors are e154 VERBATIM "
                     "and unfiltered in both cells — identical background, "
                     "not part of the delta"),
        },
    }
    G_ANCHOR["pass"] = bool(
        G_ANCHOR["e170_bank"]["windows_with_host_content"] == 0
        and G_ANCHOR["e170_bank"]["junctions_covered"] == 0
        and G_ANCHOR["e154_bank"]["windows_with_host_content"] == 16
        and G_ANCHOR["e154_bank"]["junctions_covered"] == 16
        and G_ANCHOR["budget_identical"]
        and anchor.shape == (16, BLOCK))
    assert G_ANCHOR["pass"], f"anchor gate FAILED: {G_ANCHOR}"
    log(f"G_ANCHOR: neutral bank 16x{BLOCK} plain-corpus windows (seed "
        f"{E170_ANCHOR_SEED}, {rejections} rejections/{tries} tries) — "
        f"host-content windows 0/16 (e154: {n_e154_hostwins}/16), junctions "
        f"covered 0/16 (e154: {jc_e154}/16); random-channel background "
        f"~{100 * bg_rate:.1f}%/window: PASS")

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
        raise RuntimeError("root checkpoint failed its gate vs e131/e151 cells")
    del ev

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

    # =====================================================================
    # the BEFORE/AFTER measurement battery (e154's identical instruments)
    # =====================================================================
    DELS = {"none": (), "d129": (PRE - 1,), "d_all": D_ALL, "d_r0": (0,),
            "dF2": tuple(F2_ROWS)}
    gates_surg: dict = {}

    def measure(sd: dict, tag: str, full_f2: bool) -> dict:
        net = evl_load(sd)
        out: dict = {"tag": tag}
        # (i) the dial set: F1 base expression + CE
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

    # instrument gates against e151's/e131's stored root cells
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
    log("gates: G_NONCE, G_SPLICE, G_GEO2, G_ANCHOR, G_CONS, G_MASK, G_ROW0, "
        "G_A129, G_DALL all PASS")

    # ---------------- N2 instrument gate on the BEFORE net (always; vs e160)
    log("--- N2 instrument gate on the BEFORE net (gate vs e160) ---")
    N2_HEADS = [(1, 0), (0, 0)]                    # e160's N2: {L1H0, L0H0}

    def n2_cell(net, heads, mode, f1_base_g0, f1_base_g12, base_ce=None,
                f2_site_base=None, with_f2=False):
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
            c["ce_cost"] = (None if base_ce is None
                            else c["ce_r"] - base_ce)
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
    # INSTALL F2 with NEUTRAL anchors (the ONE training; CPU; cooldown)
    # =====================================================================
    log("=" * 78)
    if not SMOKE:
        log(f"[cooldown] {COOLDOWN_S:.0f}s before the ONE training")
        cooldown(COOLDOWN_S)
    log(f"INSTALL F2 (NEUTRAL anchors): 300-step locked replay of {NAME2} at "
        f"read rows 63..69 (seed {F2_SEED}, device {DEV})")
    install = finetune_arm("installF2neutral", net0, pool2, pool2_mask,
                           anchor, train_ids, r_eval_xy, f2_eval_ids, mid,
                           F2_SEED)
    sd_after = install["sd"]
    if not SMOKE:
        log(f"[cooldown] {COOLDOWN_S:.0f}s after the ONE training")
        cooldown(COOLDOWN_S)
    save_ckpt("e170_neutral", sd_after,
              {"desc": "e131_consolidated_e113 + 300-step locked install of "
                       f"{NAME2} (nonce) at read rows 63..69 (x-cols 64..70), "
                       "name-only mask, seed 10902, NEUTRAL plain-corpus "
                       "anchor bank (seed 170, host-free) — e154's protocol "
                       "with only the anchor content changed",
               "steps": install["steps_ran"], "seed": F2_SEED,
               "anchor_bank": f"neutral (E170_ANCHOR_SEED={E170_ANCHOR_SEED})",
               "base": f"runs/checkpoints/{ROOT_CK}"})
    time.sleep(STAGGER_S)

    log("=" * 78)
    log("AFTER battery (e170_neutral)")
    after = measure(sd_after, "after", full_f2=True)
    time.sleep(STAGGER_S)

    # =====================================================================
    # ADJUDICATION (registered clauses, QUEUE e170 verbatim; no bar shopping)
    # =====================================================================
    g12b = before["base"][-12]["mean_pz"]
    g12a = after["base"][-12]["mean_pz"]
    g0b = before["base"][0]["mean_pz"]
    g0a = after["base"][0]["mean_pz"]
    ret = {j: after["base"][j]["mean_pz"] / max(before["base"][j]["mean_pz"],
                                                1e-12) for j in GEOS}
    f1_survives = bool(g0a >= SURVIVE_FLOOR and g12a >= SURVIVE_FLOOR)
    f1_dies = bool(g0a <= OVERWRITE_CEIL)

    cond = {
        "ANCHORS_DID_IT": {
            "g0_after": g0a, "g12_after": g12a, "floor": SURVIVE_FLOOR,
            "fires": f1_survives},
        "OVERWRITE_REAL": {
            "g0_after": g0a, "ceil": OVERWRITE_CEIL, "fires": f1_dies},
    }
    if cond["ANCHORS_DID_IT"]["fires"]:
        verdict = "ANCHORS-DID-IT"
        clause = (f"F1 survives under neutral anchors (g0 {g0a:.4f} >= 0.5 AND "
                  f"g-12 {g12a:.4f} >= 0.5; before {g0b:.4f}/{g12b:.4f}) — "
                  f"the demolition was the anchor channel (unlearning-by-"
                  f"contradiction); the two-facts question REOPENS with "
                  f"per-fact doors.")
    elif cond["OVERWRITE_REAL"]["fires"]:
        verdict = "OVERWRITE-REAL"
        clause = (f"F1 still dies under neutral anchors (g0 {g0a:.4f} <= 0.2; "
                  f"g-12 {g12a:.4f}; before {g0b:.4f}/{g12b:.4f}) — capacity "
                  f"is one fact at this budget; W012's bandwidth answered "
                  f"the hard way.")
    else:
        verdict = "TEXTURE"
        clause = (f"F1 lands in no registered band (g0 {g0a:.4f}, g-12 "
                  f"{g12a:.4f}; bars: survive g0 AND g-12 >= 0.5, die g0 <= "
                  f"0.2; before {g0b:.4f}/{g12b:.4f}) — partial survival / "
                  f"differential geometry damage; TEXTURE with numbers.")
    log("=" * 78)
    log(f"E170 VERDICT: {verdict}")
    log(f"  deciding: F1 g0 {g0b:.4f} -> {g0a:.4f} (ret {ret[0]:.3f}) | F1 "
        f"g-12 {g12b:.4f} -> {g12a:.4f} (ret {ret[-12]:.3f}) | F1 g+12 "
        f"{before['base'][12]['mean_pz']:.4f} -> "
        f"{after['base'][12]['mean_pz']:.4f}")
    log(f"  F2 graft: site63 span {before['site63_span']['site_strength']:+.4f}"
        f" -> {after['site63_span']['site_strength']:+.4f} (2x-ctrl "
        f"{after['site63_span']['bar_2x_control']:.4f}) site_pos "
        f"{before['site63_span']['site_pos']} -> "
        f"{after['site63_span']['site_pos']} | F2 expression onset "
        f"{before['f2_site_install']['pz_onset_mean']:.4f} -> "
        f"{after['f2_site_install']['pz_onset_mean']:.4f} (held "
        f"{after['f2_site_held']['pz_onset_mean']:.4f})")
    log(f"  F1 dials: A(129) {before['old_band']['A129']:+.4f} -> "
        f"{after['old_band']['A129']:+.4f} | row0 S "
        f"{before['old_band']['row0_strength']:+.4f} -> "
        f"{after['old_band']['row0_strength']:+.4f} | D-all g0 "
        f"{before['del_table']['d_all']['g0']['mean_pz']:.4f} -> "
        f"{after['del_table']['d_all']['g0']['mean_pz']:.4f} | held30 g0 "
        f"{before['base_held30_g0']['mean_pz']:.4f} -> "
        f"{after['base_held30_g0']['mean_pz']:.4f} | CE_R "
        f"{before['ce_r']:.4f} -> {after['ce_r']:.4f}")
    log(f"  {clause}")
    log("=" * 78)

    # ---------------- N2 rider (CONDITIONAL on the survival clause)
    n2_after: dict = {}
    rider: dict = {
        "precondition": ("ran only if F1 survives per the registered clause "
                         "(g0 >= 0.5 AND g-12 >= 0.5)"),
        "ran": False, "cells": {}, "before_gate_cells": n2_before,
        "note": ("informative in e170 (no registered rider bar): does F1's "
                 "killable circuit persist alongside F2's graft — per-fact "
                 "readouts? e160's kills60 flags co-reported"),
    }
    if f1_survives:
        log("--- N2 rider on the AFTER net (F1 and F2 priced separately) ---")
        net_a = evl_load(sd_after)                   # the two-fact net
        f2_site_base = after["f2_site_install"]["pz_onset_mean"]
        n2_after = {
            "N2:top2-noL0H3/zero": n2_cell(
                net_a, N2_HEADS, "zero", after["base"][0]["mean_pz"],
                after["base"][-12]["mean_pz"], base_ce=after["ce_r"],
                f2_site_base=f2_site_base, with_f2=True),
            "N2:top2-noL0H3/mean": n2_cell(
                net_a, N2_HEADS, "mean", after["base"][0]["mean_pz"],
                after["base"][-12]["mean_pz"], base_ce=after["ce_r"],
                f2_site_base=f2_site_base, with_f2=True),
            "S:L1H0/zero": n2_cell(
                net_a, [N2_HEADS[0]], "zero", after["base"][0]["mean_pz"],
                after["base"][-12]["mean_pz"], base_ce=after["ce_r"],
                f2_site_base=f2_site_base, with_f2=True),
            "S:L0H0/zero": n2_cell(
                net_a, [N2_HEADS[1]], "zero", after["base"][0]["mean_pz"],
                after["base"][-12]["mean_pz"], base_ce=after["ce_r"],
                f2_site_base=f2_site_base, with_f2=True),
        }
        for c in n2_after.values():
            c["f1_kills60"] = bool(c["f1_drop_g0"] >= KILL_DROP)
            c["f2_kills60"] = bool(c.get("f2_drop_onset", 0.0) >= KILL_DROP)
            log(f"  [{c['heads']} {c['mode']:4s}] F1 g0 {c['f1_g0']:.4f} "
                f"(drop {100 * c['f1_drop_g0']:.1f}%) F2 onset "
                f"{c['f2_onset']:.4f} (drop {100 * c['f2_drop_onset']:.1f}%) "
                f"CE {c['ce_cost']:+.4f}")
        n2z = n2_after["N2:top2-noL0H3/zero"]
        rider.update({
            "ran": True, "cells": n2_after,
            "primary_cell": "N2:top2-noL0H3/zero (e160's N2, zero mode)",
            "f1_drop_g0": n2z["f1_drop_g0"],
            "f2_drop_onset": n2z["f2_drop_onset"],
            "kill_bar": KILL_DROP,
            "summary": (f"F1 {100 * n2z['f1_drop_g0']:.1f}% / F2 "
                        f"{100 * n2z['f2_drop_onset']:.1f}% drops under "
                        f"N2/zero at CE {n2z['ce_cost']:+.3f} — F1's killable "
                        f"circuit {'PERSISTS' if n2z['f1_drop_g0'] >= KILL_DROP else 'does not clearly persist'} "
                        f"alongside F2's graft"),
        })
        del net_a
        time.sleep(STAGGER_S)
    else:
        rider["skip_reason"] = (
            "F1 did not survive (the registered ANCHORS-DID-IT clause failed) "
            "— the rider's question (does F1's killable circuit persist "
            "alongside F2's graft) is moot on a dead F1; skipped, no bar "
            "shopped")

    # ---------------- e154 comparison triangle (runtime read, full precision)
    e154_cmp: dict = {"path": "runs/e154/metrics.json", "loaded": False}
    try:
        e154m = json.loads(E154_METRICS.read_text(encoding="utf-8"))
        e154_cmp["loaded"] = True
        e154_cmp["after"] = {
            "g0": e154m["after"]["base"]["0"]["mean_pz"],
            "g-12": e154m["after"]["base"]["-12"]["mean_pz"],
            "g+12": e154m["after"]["base"]["12"]["mean_pz"],
            "held30_g0": e154m["after"]["base_held30_g0"]["mean_pz"],
            "ce_r": e154m["after"]["ce_r"],
            "row0_strength": e154m["after"]["old_band"]["row0_strength"],
            "A129": e154m["after"]["old_band"]["A129"],
            "d_all_g0": e154m["after"]["del_table"]["d_all"]["g0"]["mean_pz"],
            "f2_onset": e154m["after"]["f2_site_install"]["pz_onset_mean"],
            "f2_span": e154m["after"]["f2_site_install"]["pname_mean_over7"],
            "site63_span_strength":
                e154m["after"]["site63_span"]["site_strength"],
        }
        e154_cmp["before"] = {
            "g0": e154m["before"]["base"]["0"]["mean_pz"],
            "g-12": e154m["before"]["base"]["-12"]["mean_pz"],
        }
        e154_cmp["sanity_before_match"] = bool(
            abs(e154m["before"]["base"]["0"]["mean_pz"] - g0b) < 1e-9
            and abs(e154m["before"]["base"]["-12"]["mean_pz"] - g12b) < 1e-9)
        log(f"e154 comparison loaded: incumbent-anchor after g0 "
            f"{e154_cmp['after']['g0']:.6f} g-12 "
            f"{e154_cmp['after']['g-12']:.6f} | before-match "
            f"{e154_cmp['sanity_before_match']}")
    except Exception as e:  # noqa: BLE001
        e154_cmp["error"] = str(e)
        log(f"e154 metrics unavailable for comparison: {e}")

    # ---------------- outputs
    metrics = {
        "experiment": "e170_anchor_neutral",
        "date": common.now_iso(),
        "registration": ("QUEUE.md row e170 + the dispatch's registered "
                         "prediction VERBATIM (T100 reading (2) discharge); "
                         "operationalizations frozen in the module docstring "
                         "before compute"),
        "registered_prediction": REGISTERED_PREDICTION,
        "committed_prediction": REGISTERED_PREDICTION["committed"],
        "prediction_held": None,
        "question": ("was e154's annihilation of F1 (0.785 -> 0.0017 at g0) "
                     "the MIRABEL graft or the incumbent-continuation ANCHORS "
                     "(8 anti-F1 junction anchors per batch x 300 steps)? "
                     "Identical install, NEUTRAL plain-corpus anchors: "
                     "ANCHORS-DID-IT vs OVERWRITE-REAL"),
        "root": f"runs/checkpoints/{ROOT_CK} (loaded, gated bit-exact vs "
                "e131/e151/e150/e160 cells)",
        "anchor_delta": dict(G_ANCHOR, doc_ref="ANCHOR-DELTA section of the "
                            "module docstring"),
        "f2": {"name": NAME2, "kind": "nonce (e134 design; e154's F2)",
               "corpus_occurrences": G_NONCE["corpus_occurrences"],
               "onset_char": "M", "onset_char_id": mid,
               "install": {"desc": "e154's e151 locked-replay protocol "
                                   "VERBATIM at offset -66 (MIRABEL locked at "
                                   "x-cols 64..70, onset read row 63, "
                                   "name-only 7-target mask, zero position "
                                   "variance) — ONLY the anchor bank's "
                                   "content changed (neutral)",
                           "recipe": "batch 32 = 16 pool + 16 anchors (8 "
                                     "paired NEUTRAL + 8 random corpus), "
                                     "token-weighted union CE, AdamW "
                                     "(0.9,0.95) wd 0.1, lr 1e-3 constant, "
                                     "clip 1.0",
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
                                             "trained_g0": 0}},
        "gates": {"G_NONCE": G_NONCE, "G_SPLICE": G_SPLICE, "G_GEO2": G_GEO2,
                  "G_ANCHOR": G_ANCHOR, "G_CONS": G_CONS, "G_MASK": G_MASK,
                  "G_ROW0": G_ROW0, "G_A129": G_A129, "G_DALL": G_DALL,
                  "G_N2": G_N2, "G_SURG": gates_surg},
        "before": before, "after": after,
        "e154_comparison": e154_cmp,
        "retentions": {f"g{j:+d}": ret[j] for j in GEOS},
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
                         "clause": clause,
                         "f1_survives": f1_survives, "f1_dies": f1_dies},
        "n2_rider": rider,
        "honesty_reflex": {
            "anchor_construction_fidelity": (
                "the neutral bank removed the contradiction signal by "
                "construction (G_ANCHOR: 0/16 windows with host content, 0/16 "
                "context->host junctions covered, vs e154's 16/16 and 16/16; "
                "rejection checked over [s, s+257) = window + first target "
                "token, so no window can train a host-onset target either); "
                "budget kept identical (16x256, full CE, 8 paired + 8 random "
                "per batch, same moduli => identical RNG stream at seed "
                "10902). REMAINING DIFFERENCES BEYOND THE MISSION'S 'ONLY "
                "CHANGE': (a) register shift — e154's bank windows are "
                "install-position drama text (name-heavy scenes), e170's are "
                "uniform corpus draws; (b) the 8-per-batch random-anchor "
                "channel is e154 verbatim and unfiltered — it draws "
                f"host-containing windows at ~{100 * bg_rate:.1f}%/window in "
                "BOTH cells (shared background, ~1-2% of that channel), so "
                "the contradiction signal is reduced to a low uniform "
                "background, not to absolute zero"),
            "single_seed_single_lineage": (
                "one install (seed 10902) on one root (e131 consolidated, "
                "one seed) — n=1; the fork verdict is lineage-specific until "
                "replicated (e157 class)"),
            "before_reproduction": (
                "the BEFORE battery re-measured the same root with e154's "
                "instruments; G_CONS is bit-exact vs stored cells"
                f"{' and the g0/g-12 before-values match e154 stored before' if e154_cmp.get('sanity_before_match') else ''}"),
            "graft_formation_is_measured": (
                "F2's formation is adjudicated by the site census (2x-control "
                "bar) and expression cells, not assumed; the install traj "
                "(in-loop p(M@site)) is recorded in f2.install.traj"),
            "novel_geometry_caveat_T087": (
                "F1 g-12 retention conflates 'the route survived' with 'sink "
                "health at the novel geometry survived' (T087/E150 caveat); "
                "the mask/ladder columns separate the two only partially"),
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

    plot(rd / "anchor_neutral.png", before, after, ret, verdict, clause, cond,
         rider, G_ANCHOR, install, e154_cmp)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'anchor_neutral.png'}, "
        f"ckpt runs/checkpoints/e170_neutral.pt")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot

def plot(path, before, after, ret, verdict, clause, cond, rider, g_anchor,
         install, e154_cmp):
    """Legible anchor-delta dial set + verdict."""
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.5))
    GEOS = (-12, 0, 12)
    e154a = e154_cmp.get("after", {}) if e154_cmp.get("loaded") else {}

    # (0,0) F1 dial set: before / after(neutral) / after(e154 incumbent)
    ax = axes[0, 0]
    xs = np.arange(3)
    lbls = ["g-12 (NOVEL,\nthe door)", "g0 (trained)", "g+12 (NOVEL)"]
    for k, j in enumerate(GEOS):
        b = before["base"][j]["mean_pz"]
        a = after["base"][j]["mean_pz"]
        ax.bar(k - 0.27, b, 0.25, color="steelblue", edgecolor="k", lw=0.5,
               label="before (root)" if k == 0 else None)
        ax.bar(k, a, 0.25, color="seagreen", edgecolor="k", lw=0.5,
               label="after NEUTRAL anchors (e170)" if k == 0 else None)
        if e154a:
            ax.bar(k + 0.27, e154a[f"g{j:+d}" if j != 0 else "g0"], 0.25,
                   color="crimson", edgecolor="k", lw=0.5,
                   label="after INCUMBENT anchors (e154)" if k == 0 else None)
            ax.text(k + 0.27, e154a[f"g{j:+d}" if j != 0 else "g0"] + 0.012,
                    f"{e154a[f'g{j:+d}' if j != 0 else 'g0']:.3f}",
                    ha="center", fontsize=8)
        ax.text(k - 0.27, b + 0.012, f"{b:.3f}", ha="center", fontsize=8)
        ax.text(k, a + 0.012, f"{a:.3f}", ha="center", fontsize=8,
                fontweight="bold")
        ax.text(k, 0.45, f"x{ret[j]:.2f}", ha="center", fontsize=9,
                fontweight="bold",
                color="seagreen" if ret[j] >= 0.5 else "firebrick")
    ax.axhline(0.5, ls="--", lw=1.1, color="seagreen")
    ax.text(2.42, 0.515, "survive bar 0.5", fontsize=6.5, color="seagreen")
    ax.axhline(0.2, ls="--", lw=1.1, color="firebrick")
    ax.text(2.42, 0.215, "die bar 0.2", fontsize=6.5, color="firebrick")
    ax.set_xticks(xs)
    ax.set_xticklabels(lbls, fontsize=8.5)
    ax.set_ylabel("F1 battery p(Z) install-60")
    ax.set_ylim(0, 1.12)
    ax.set_title(f"(i) F1's DIAL SET — the anchor fork -> {verdict}",
                 fontsize=9.5)
    ax.legend(fontsize=7, loc="lower right")

    # (0,1) F2 site census spectrum
    ax = axes[0, 1]
    for tag, d, col, mk in (("before", before, "steelblue", "o"),
                            ("after", after, "seagreen", "s")):
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
                lw=0.9, color="seagreen", alpha=0.4, label="after (onset)")
    ax.axvspan(63, 69, color="crimson", alpha=0.07)
    ax.axvspan(121, 137, color="seagreen", alpha=0.05)
    ax.axvspan(183, 189, color="gray", alpha=0.05)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("wpe row (red: F2 band 63-69; green: F1 band; gray: 183)")
    ax.set_ylabel("strength = min(mean-drop, zero-drop)")
    ax.set_title(f"(F2) did the graft still form? span "
                 f"{before['site63_span']['site_strength']:+.4f} -> "
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
        "REGISTERED (QUEUE e170 verbatim; no bar shopping):",
        "  ANCHORS-DID-IT: F1 survives (g0 >= 0.5 AND g-12 >= 0.5)",
        "  OVERWRITE-REAL: F1 still dies (g0 <= 0.2)",
        "  texture (partial survival, differential geometry",
        "   damage) => TEXTURE with numbers",
        "",
        "DECIDING NUMBERS:",
        f"  F1 g-12: {before['base'][-12]['mean_pz']:.4f} -> "
        f"{after['base'][-12]['mean_pz']:.4f}",
        f"  F1 g0:   {before['base'][0]['mean_pz']:.4f} -> "
        f"{after['base'][0]['mean_pz']:.4f}",
        f"  F1 g+12: {before['base'][12]['mean_pz']:.4f} -> "
        f"{after['base'][12]['mean_pz']:.4f}",
        f"  F1 held30 g0: {before['base_held30_g0']['mean_pz']:.4f} -> "
        f"{after['base_held30_g0']['mean_pz']:.4f}",
        f"  row0 S: {before['old_band']['row0_strength']:+.4f} -> "
        f"{after['old_band']['row0_strength']:+.4f} | A(129): "
        f"{before['old_band']['A129']:+.4f} -> "
        f"{after['old_band']['A129']:+.4f}",
        f"  D-all g0: {before['del_table']['d_all']['g0']['mean_pz']:.4f} -> "
        f"{after['del_table']['d_all']['g0']['mean_pz']:.4f} | CE_R "
        f"{before['ce_r']:.4f} -> {after['ce_r']:.4f}",
        f"  F2 graft span: {before['site63_span']['site_strength']:+.4f} -> "
        f"{after['site63_span']['site_strength']:+.4f} | F2 onset -> "
        f"{after['f2_site_install']['pz_onset_mean']:.3f} (held "
        f"{after['f2_site_held']['pz_onset_mean']:.3f})",
    ]
    if e154a:
        vlines += [
            "",
            "e154 arm (incumbent anchors, identical install):",
            f"  g0 {e154a['g0']:.4f} | g-12 {e154a['g-12']:.4f} | held30 "
            f"{e154a['held30_g0']:.4f} | CE_R {e154a['ce_r']:.4f}",
        ]
    rider_line = (rider["summary"] if rider.get("ran")
                  else "SKIPPED — " + rider.get("skip_reason", ""))
    vlines += ["", f"N2 rider: {rider_line}", "", f"VERDICT: {verdict}"]
    vlines += [f"  {wd}" for wd in
               [clause[i:i + 64] for i in range(0, len(clause), 64)]]
    for i, tx in enumerate(vlines):
        ax.text(0.02, 0.97 - i * 0.036, tx, fontsize=7.0, va="top",
                family="monospace")

    # (1,0) the ANCHOR-DELTA panel
    ax = axes[1, 0]
    cats = ["windows with host\ncontent (of 16)",
            "context->host junctions\ncovered (of 16)",
            "bank size\n(windows)"]
    e154v = [g_anchor["e154_bank"]["windows_with_host_content"],
             g_anchor["e154_bank"]["junctions_covered"],
             g_anchor["e154_bank"]["n_windows"]]
    e170v = [g_anchor["e170_bank"]["windows_with_host_content"],
             g_anchor["e170_bank"]["junctions_covered"],
             g_anchor["e170_bank"]["n_windows"]]
    xs = np.arange(3)
    ax.bar(xs - 0.19, e154v, 0.36, color="crimson", edgecolor="k", lw=0.5,
           label="e154 bank (incumbent-continuation)")
    ax.bar(xs + 0.19, e170v, 0.36, color="seagreen", edgecolor="k", lw=0.5,
           label="e170 bank (plain corpus, seed 170)")
    for x, v in zip(xs - 0.19, e154v):
        ax.text(x, v + 0.25, str(v), ha="center", fontsize=9)
    for x, v in zip(xs + 0.19, e170v):
        ax.text(x, v + 0.25, str(v), ha="center", fontsize=9,
                fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylim(0, 19)
    bg = g_anchor["random_channel_background"]["est_window_hit_rate"]
    ax.text(0.02, 0.40,
            f"ONLY protocol change: the 16-window paired-anchor bank\n"
            f"rejection over [s, s+257): window + FIRST TARGET token\n"
            f"(no window can train a host-onset target)\n"
            f"budget/RNG identical: 16x256 full-CE, 8 paired + 8 random,\n"
            f"same draw moduli @ seed 10902 (same ix/aj/rj streams)\n"
            f"random-anchor channel e154-verbatim: host-window background\n"
            f"~{100 * bg:.1f}%/window in BOTH cells (shared, not delta)",
            fontsize=6.8, family="monospace", transform=ax.transAxes,
            bbox=dict(facecolor="lightyellow", alpha=0.9, edgecolor="gray"))
    ax.set_title("(delta) THE ANCHOR-DELTA — e154 vs e170 bank", fontsize=9.5)
    ax.legend(fontsize=7, loc="upper right")

    # (1,1) deletion table + phase dials
    ax = axes[1, 1]
    dls = ("none", "d129", "d_all", "d_r0", "dF2")
    xs = np.arange(len(dls))
    for k, (tag, d, col) in enumerate((("before", before, "steelblue"),
                                       ("after", after, "seagreen"))):
        vals = [d["del_table"][dl]["g0"]["mean_pz"] for dl in dls]
        ax.bar(xs + (k - 0.5) * 0.36, vals, 0.34, color=col, edgecolor="k",
               lw=0.4, label=tag)
        for x, v in zip(xs + (k - 0.5) * 0.36, vals):
            ax.text(x, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.6,
                    rotation=90, va="bottom")
    ax.text(1.62, 0.52,
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

    # (1,2) install trajectory (F2 formation) vs e154's
    ax = axes[1, 2]
    traj = install["traj"]
    ax.plot([t["step"] for t in traj], [t["p_name_mean"] for t in traj],
            "o-", ms=3, lw=1.3, color="seagreen", label="e170 p(M@63) neutral")
    ax.plot([t["step"] for t in traj], [t["ce_r"] for t in traj], "s--",
            ms=3, lw=1.0, color="dimgray", label="e170 CE_R")
    if e154_cmp.get("loaded"):
        try:
            e154m = json.loads(E154_METRICS.read_text(encoding="utf-8"))
            et = e154m["f2"]["install"]["traj"]
            ax.plot([t["step"] for t in et], [t["p_name_mean"] for t in et],
                    "^:", ms=3, lw=1.0, color="crimson", alpha=0.8,
                    label="e154 p(M@63) incumbent")
            ax.plot([t["step"] for t in et], [t["ce_r"] for t in et],
                    "v:", ms=3, lw=1.0, color="crimson", alpha=0.5,
                    label="e154 CE_R")
        except Exception as e:  # noqa: BLE001
            ax.text(0.3, 0.2, f"e154 traj unavailable: {e}", fontsize=6.5)
    ax.set_xlabel("install step")
    ax.set_ylabel("p(M@63) / CE_R")
    ax.set_ylim(0, 2.0)
    ax.set_title(f"(F2 install) formation under neutral anchors — "
                 f"{install['steps_ran']} steps, seed {install['seed']}",
                 fontsize=9.5)
    ax.legend(fontsize=7, loc="center right")
    ax.grid(alpha=0.25)

    fig.suptitle(f"E170 — ANCHOR-NEUTRAL SECOND INSTALL (root "
                 f"e131_consolidated + locked MIRABEL at rows 63..69, "
                 f"neutral anchors) -> {verdict}", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
