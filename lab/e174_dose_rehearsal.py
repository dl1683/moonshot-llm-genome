"""E174 — THE F2-DOSE LADDER + REHEARSAL CELL (e170's gated follow-up; the
gate fired: OVERWRITE-REAL). QUEUE.md row e174; T103's capacity claim put on
the budget axis. e170 settled that a second locked fact's install
DEMOLISHES the first under NEUTRAL anchors (F1 0.7850 -> 0.0001 at g0;
graft formed, corpus CE flat) — capacity is ~one fact wide at the FULL
300-step budget. Two questions here: (1) the BUDGET AXIS — is capacity a
DIAL (a smaller F2 dose cohabits) or a CLIFF (F1 dies at every
graft-forming dose)? (2) REHEARSAL — does interleaved F1-replay during
F2's install save the tenant (T101's disuse frame: the replay channel is
exactly the "life-support" channel that every past fine-tune's anchors
may have been quietly providing)? Both quotable either way.

ROOT (mandated): runs/checkpoints/e131_consolidated_e113.pt — F1 = ZEPHYRA,
the sink-coupled consolidated fact. F2 = MIRABEL at rows 63..69 (e170's
conventions: offset j=-66, name x-cols 64..70, onset read row 63, 7-target
name-only mask, zero position variance).

================================= ARM (A) ====================================
THE DOSE LADDER — ONE training, free checkpoints. e170's NEUTRAL-anchor
install protocol VERBATIM (corpus rebuild, install/held pools, F2
placement, neutral anchor bank seed 170, batch 32 = 16 install + 16
anchors (8 paired neutral + 8 random), token-level union CE, AdamW
(0.9,0.95) wd 0.1 constant lr 1e-3 clip 1.0, seed 10902, threads 4) with
in-run state snapshots at F2-doses 25/50/75/150/300 steps (snapshots
consume NO training RNG — the stream is e170-identical; G_REPRO compares
this run's in-loop p(M@site) trajectory to e170's stored one). Each
checkpoint is measured with: F1's full dial set (g-12/g0/g+12 install-60 +
held30@g0 + CE_R + cross-leak cells), F2's expression (site onset + span,
install pool primary + held30), and F2's graft strength (site-content
census, span-primary, e139's 2x-control bar) — the capacity trade-off
curve.
================================= ARM (B) ====================================
THE REHEARSAL ARM — second training, ~600 steps alternating F2-install
batches with F1-replay batches, 1:1 interleave: odd optimizer steps are
e170's install batch VERBATIM; even steps swap the 16 install windows for
16 F1 windows drawn from e113's jitter pool (60 install hosts x offsets
{-8,-4,0,+4,+8}, ZEPHYRA spliced, 7 name-char mask — the e113 arm-(a)
consolidation recipe, i.e. F1 replayed AT ITS HOME geometry); anchors stay
NEUTRAL (the e170 bank) in BOTH batch types — the ONLY delta vs arm (A)
is the interleaved F1-replay channel (e113's incumbent-continuation anchor
bank is deliberately NOT used: it would reintroduce e154's contradiction
channel). Total F2 exposure is matched to arm (A)'s 300 install batches
(600 steps = 300 F2 + 300 F1). Checkpoints at steps 50/100/150/300/600 =
F2-doses 25/50/75/150/300, the SAME matched-dose ladder as arm (A); F1
and F2 measured identically per checkpoint.

REGISTERED PREDICTION (QUEUE e174 / the dispatch VERBATIM; no bar shopping):
  - WIN-WIN fires if: a dose exists with F2 site onset >= 0.5 AND F1 g0
    >= 0.5 — capacity is a budget dial.
  - ALL-OR-NOTHING fires if: F1 dies (<= 0.2) at every graft-forming dose
    (F2 onset >= 0.5) — the overwrite is formation-triggered, a cliff.
  - REHEARSAL-RESOLVES fires if: in arm (B) F1 holds >= 0.5 at the
    F2-dose where (A) killed it.
  - REHEARSAL-FAILS if not.
  - No bar shopping; texture => TEXTURE with the curve.

OPERATIONALIZATIONS (frozen before compute):
  * dose = COUNT OF F2-INSTALL BATCHES (optimizer steps of F2 exposure).
    Arm (A) checkpoint at step= dose; arm (B) checkpoint at step= 2*dose.
  * F1 g0 = battery p(Z) mean on the install-60 battery at j=0 on the
    checkpoint net; F2 site onset = read_fact_at pz_onset_mean over the
    F2 install pool (row 63).
  * graft-forming dose := F2 site onset >= 0.5 at that checkpoint.
  * dose_kill(A) := the SMALLEST measured dose where arm-(A) F1 g0 <= 0.2
    (the die bar). If no measured dose kills F1, the rehearsal fork is
    MOOT (no kill to rescue) and recorded as such.
  * REHEARSAL-RESOLVES := arm-(B) F1 g0 >= 0.5 at dose_kill(A) (F2 onset
    in arm (B) at that dose co-reported — a rescue at a dose where the
    graft has NOT yet formed in (B) is slower-kill texture, not
    cohabitation; the bars stay as registered either way).
  * Ladder adjudication order: WIN-WIN -> ALL-OR-NOTHING -> TEXTURE (on
    arm (A) alone); the rehearsal fork adjudicates separately and both
    verdicts are reported.

INSTRUMENT PROVENANCE: load_cpu/evl_load/battery_cell/battery_pz/
ce_fixed_cpu/val_windows/read_fact_at/row_census are lab/e170_anchor_
neutral.py VERBATIM (which are e154/e151/e143/e150/e139/e131/e065
verbatim); finetune_ladder is e170's finetune_arm VERBATIM + no-RNG
snapshot saves; the F1 jitter pool builder is lab/e113_all_addresses.py
VERBATIM arithmetic. Copied, not imported, per lab convention.

COMPUTE ENVELOPE: CPU-ONLY MANDATED (CUDA_VISIBLE_DEVICES=-1 before torch
import; e176 co-runs on CPU) — torch threads 4, all evals sequential,
3 s staggers between phases, cooldown(90) between the TWO trainings
(60-120 s mandated band), no busy-waiting (no single wait exceeds 180 s:
cooldown 90 s, staggers 3 s); training caps: arm (A) 1740 s, arm (B)
3600 s (the ~600-step interleaved arm needs ~2x e151's 1500 s CPU
convention — recorded as a deviation; it is compute, not a wait).
Single seed, single lineage.

Outputs: runs/e174/{metrics.json, dose_rehearsal.png}; checkpoints
runs/checkpoints/e174_dose_s{25,50,75,150}.pt + e174_dose.pt (arm A) +
e174_rehearsal_s{50,100,150,300}.pt + e174_rehearsal.pt (arm B). No
NOTES/THINKING/QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python e174_dose_rehearsal.py    (E174_SMOKE=1 shakedown)
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

torch.set_num_threads(4)                              # LOW (e176 co-runs)

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
common.DEVICE = "cpu"
from common import (Cfg, CharCorpus, TinyGPT, cooldown, run_dir,    # noqa: E402
                    save_json)
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E174_SMOKE") == "1"
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
E170_METRICS = E43.REPO / "runs" / "e170" / "metrics.json"

# ---- F2 placement (e170's conventions: fresh site rows 63..69) ----------------
F2_J = -66                           # offset: name x-cols 64..70, read rows 63..69
F2_ADDR_ROW = 63                     # onset read row
F2_Z_XCOL = PRE + F2_J               # 64
F2_CONT = BLOCK - PRE - F2_J - len(NAME2)     # 185-token continuation
F2_ROWS = tuple(range(63, 70))       # the 7 trained read rows
CTR2 = (20, 40, 80, 100, 150, 160, 200, 220)  # e170's control set (finals)
CTRL_LIGHT = (20, 40, 80, 150, 220)  # checkpoint censuses (5 of the 8)
ROWS_F2_FULL = (0, 1, 2) + (60, 61, 62) + F2_ROWS + CTR2        # e170 verbatim
ROWS_F2_ONSET_FULL = (0,) + (60, 61, 62) + (63,) + CTR2         # e170 verbatim
ROWS_CEN = (0,) + (60, 61, 62) + F2_ROWS + CTRL_LIGHT           # per-checkpoint
GEOS = (-12, 0, 12)                  # novel x2 + trained g0

# ---- F1 replay pool (e113 arm-(a) jitter recipe VERBATIM arithmetic) ----------
JITTERS = (-8, -4, 0, 4, 8)          # e109/e113's registered jitter set

# ---- dose ladders (frozen) ---------------------------------------------------
CKPT_DOSES = (25, 50, 75, 150, 300)  # F2-batch counts (arm A step = dose)
CKPT_B_STEPS = (50, 100, 150, 300, 600)   # arm B steps -> same doses

# ---- fine-tune envelope (e109 arm-b / e119-L / e143 / e151 / e154 / e170) ----
FT_LR = 1e-3
FT_STEPS_A = 300 if not SMOKE else 8
FT_STEPS_B = 600 if not SMOKE else 12
FT_TIME_CAP_A = 1740.0               # arm A cap (< 30 min single-step bound)
FT_TIME_CAP_B = 3600.0               # arm B cap (~600 steps; deviation noted)
EVAL_EVERY_A = 25 if not SMOKE else 2
EVAL_EVERY_B = 50 if not SMOKE else 4
NAME_BS, ANCH_BS = 16, 16
F2_SEED = 10902                      # e119-L / e143 / e151 / e154 / e170's seed
COOLDOWN_S = 90.0                    # between the TWO trainings (60-120 band)
STAGGER_S = 3.0                      # between phases (e160 convention)

# ---- neutral anchor bank (e170's, seed 170; verified vs e170's stored starts) -
E170_ANCHOR_SEED = 170
ANCHOR_FORBIDDEN = ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")
E170_BANK_STARTS_REF = [825650, 361746, 954106, 856844, 615070, 335787,
                        329879, 96582, 253994, 341757, 608690, 127521,
                        228843, 834931, 15774, 408603]

# ---- gates / references (full precision, from stored metrics) -----------------
R_EVAL_SEED = 26502                  # e065 CE_R eval-bank seed (verbatim)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05
G_CONS_REF_PZ = 0.7850371599197388            # e131/e151/e170 none g0 install60
G_CONS_REF_CE = 1.663516640663147             # e131/e151/e170 ce_r
G_CONS_REF_G12 = 0.9155886173248291           # e150/e160/e170 g-12 base
G_F2INST_REF_ONSET = 0.04224947467446327      # e170 before f2_site_install
G_F2INST_REF_SPAN = 0.18264618515968323       # e170 before pname_mean_over7
G_REPRO_TOL = 0.05                            # e113's cross-rebuild tolerance
G_REPRO_TIGHT = 0.02

# ---- registered bar constants (frozen; QUEUE e174 / dispatch verbatim) --------
SURVIVE_FLOOR = 0.50                 # WIN-WIN / REHEARSAL-RESOLVES clause
GRAFT_BAR = 0.50                     # F2 site onset "graft-forming" bar
DIE_CEIL = 0.20                      # F1 dies bar
SITE_CTRL_MULT = 2.0                 # e139 site convention: >= 2x control-max

REGISTERED_PREDICTION = {
    "win_win": "WIN-WIN fires if: a dose exists with F2 site onset >= 0.5 AND "
        "F1 g0 >= 0.5 — capacity is a budget dial.",
    "all_or_nothing": "ALL-OR-NOTHING fires if: F1 dies (<= 0.2) at every "
        "graft-forming dose (F2 onset >= 0.5) — the overwrite is "
        "formation-triggered, a cliff.",
    "rehearsal_resolves": "REHEARSAL-RESOLVES fires if: in arm (B) F1 holds "
        ">= 0.5 at the F2-dose where (A) killed it.",
    "rehearsal_fails": "REHEARSAL-FAILS if not.",
    "no_bar_shopping": "No bar shopping; texture => TEXTURE with the curve.",
    "operationalizations": "dose = count of F2-install batches (arm A step = "
        "dose; arm B step = 2*dose); F1 g0 = battery p(Z) install-60 at j=0; "
        "F2 site onset = read_fact_at pz_onset_mean over the F2 install pool "
        "(row 63); graft-forming := onset >= 0.5; dose_kill(A) = smallest "
        "measured dose with arm-(A) F1 g0 <= 0.2 (none => rehearsal fork "
        "MOOT); REHEARSAL-RESOLVES := arm-(B) F1 g0 >= 0.5 at dose_kill(A) "
        "(arm-B F2 onset there co-reported); ladder order WIN-WIN -> "
        "ALL-OR-NOTHING -> TEXTURE on arm (A) alone, rehearsal fork separate.",
    "committed": "Both forks registered before compute (T103's budget-axis "
                 "question); adjudicated against exactly these bars.",
}

trims: list[str] = []
deviations: list[str] = [
    "Nets are the mandated 2.7M e131_consolidated line (e143/e154/e170's "
    "precedent note; the dispatch's '<=1M family' phrase is an envelope "
    "statement — every gate reference and lineage number of this cell lives "
    "on the 2.7M line, and the mandated root IS e131_consolidated_e113.pt).",
    "Arm (B) time cap 3600 s (~600 optimizer steps at e170's observed ~5 "
    "s/step CPU rate; e151's 1500 s convention covers 300 steps): the "
    "mission mandates the ~600-step 1:1 interleaved arm with F2 exposure "
    "matched to arm (A)'s 300 — that is 600 optimizer steps, one training. "
    "All WAITS stay <= 180 s (cooldown 90 s, staggers 3 s).",
    "DOSE = F2-install batch count as the proxy for exposure (the dispatch's "
    "own axis). Arm (B) matches arm (A)'s 300 F2 batches but necessarily has "
    "2x optimizer steps and the F1 channel's own gradients — matched-dose is "
    "NOT matched optimizer state; the arm-(B) F2-onset column at each "
    "matched dose adjudicates whether the graft formed at the same rate.",
    "In-run snapshot saves consume NO training RNG; arm (A)'s RNG stream is "
    "e170-identical at seed 10902 (G_REPRO vs e170's stored trajectory); "
    "arm (B) is its own stream by design (alternating ix moduli 60/300).",
    "Neutral anchors in BOTH arms (arm B's F1-replay batches use the same "
    "e170 neutral bank, NOT e113's incumbent-continuation bank — the only "
    "delta vs arm (A) is the interleaved F1 windows; e154's contradiction "
    "channel stays discharged).",
    "Per-checkpoint site censuses use a reduced control set (5 of e170's 8 "
    "CTR2 rows, incl. rows 20/40 which carry e170's before-control max); "
    "the two finals use e170's full row sets incl. the onset census — the "
    "2x-control bar convention is preserved everywhere.",
    "Single seed (10902), single lineage — no replication arm.",
    "Smoke mode trims: 8/12-step arms, reduced censuses; nothing adjudicated.",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/e170_anchor_neutral.py VERBATIM (= e154/e151/e143/e150/
# e139/e131/e065 lineage), GPU gate already removed there.

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


def site_summary(census: dict, site_rows, ctrl_rows) -> dict:
    """e170's site63_span summary convention from a census dict."""
    rows = census["rows"]
    cm = max(rows[str(r)]["strength"] for r in ctrl_rows
             if str(r) in rows)
    sstr = max(rows[str(r)]["strength"] for r in site_rows
               if str(r) in rows)
    spos = any(rows[str(r)]["content"] and
               rows[str(r)]["strength"] >= SITE_CTRL_MULT * cm
               for r in site_rows if str(r) in rows)
    best = max(site_rows, key=lambda r: rows[str(r)]["strength"]
               if str(r) in rows else -1e9)
    return {"control_max": cm, "site_strength": sstr,
            "bar_2x_control": SITE_CTRL_MULT * cm,
            "site_pos": bool(spos), "peak_row": int(best)}


CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "e174", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name} "
        f"({', '.join(f'{k}={v}' for k, v in meta.items())})")


def snap_sd(net: TinyGPT) -> dict:
    """No-RNG state snapshot (e170's sd_cpu convention)."""
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


# ------------------------------------------------------------------ trainers

def make_batch(net: TinyGPT, pool_x, pool_mask, ix, anchor, aj, train_ids, rj):
    """e170's finetune_arm batch construction VERBATIM (16 install windows +
    16 anchors: 8 paired bank + 8 random corpus), union CE (name mask on the
    install half, full CE on the anchor half)."""
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
    return (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())


def opt_step(net, opt, loss):
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()


def finetune_ladder(tag: str, net0: TinyGPT, pool_x: torch.Tensor,
                    pool_mask: torch.Tensor, anchor: torch.Tensor,
                    train_ids: torch.Tensor, r_eval_xy, f_eval_ids, zid: int,
                    seed: int):
    """ARM (A): e170's finetune_arm VERBATIM (same draws/shapes/moduli at
    seed 10902 -> e170-identical RNG stream) + no-RNG in-run snapshots at
    CKPT_DOSES. In-loop evals consume no RNG (e170 convention)."""
    net = copy.deepcopy(net0).to(DEV)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    n_pool = pool_x.shape[0]
    n_anc = anchor.shape[0]
    assert n_anc == 16, f"anchor bank must stay 16 windows (got {n_anc})"
    traj, snaps, t_start = [], {}, time.time()
    step = 0
    evl = copy.deepcopy(net0)          # CPU eval twin
    ckpt_at = set(CKPT_DOSES if not SMOKE else (2, 4, 8))
    for step in range(1, FT_STEPS_A + 1):
        ix = torch.randint(n_pool, (NAME_BS,), generator=gen)
        aj = torch.randint(n_anc, (ANCH_BS // 2,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (ANCH_BS // 2,),
                           generator=gen)
        loss = make_batch(net, pool_x, pool_mask, ix, anchor, aj,
                          train_ids, rj)
        opt_step(net, opt, loss)
        if step in ckpt_at and step not in snaps:
            sd = snap_sd(net)
            snaps[step] = sd
            nm = f"e174_dose_s{step}" if step != max(ckpt_at) else "e174_dose"
            save_ckpt(nm, sd, {"arm": "A_dose_ladder", "f2_dose": step,
                               "steps": step, "seed": seed,
                               "desc": f"e131_consolidated + {step} F2-install "
                                       f"batches (MIRABEL rows 63..69, neutral "
                                       f"anchors, seed 10902) — e174 dose "
                                       f"ladder checkpoint",
                               "base": f"runs/checkpoints/{ROOT_CK}"})
        if step % EVAL_EVERY_A == 0 or step == FT_STEPS_A or \
                (time.time() - t_start) > FT_TIME_CAP_A:
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
        if (time.time() - t_start) > FT_TIME_CAP_A:
            log(f"  [{tag}] time cap {FT_TIME_CAP_A:.0f}s at s{step}")
            trims.append(f"{tag}: time cap at step {step}")
            break
    net.eval()
    if step not in snaps:               # cap landed off-ladder: save the final
        sd = snap_sd(net)
        snaps[step] = sd
        save_ckpt("e174_dose", sd, {"arm": "A_dose_ladder", "f2_dose": step,
                                    "steps": step, "seed": seed,
                                    "desc": "e174 arm A final (time-cap trim)",
                                    "base": f"runs/checkpoints/{ROOT_CK}"})
    del net
    return {"snaps": snaps, "traj": traj, "steps_ran": step, "seed": seed}


def finetune_rehearsal(tag: str, net0: TinyGPT,
                       pool2_x: torch.Tensor, pool2_mask: torch.Tensor,
                       jit_x: torch.Tensor, jit_mask: torch.Tensor,
                       anchor: torch.Tensor, train_ids: torch.Tensor,
                       r_eval_xy, f_eval_ids, f1_eval_ids, zid: int,
                       mid: int, seed: int):
    """ARM (B): ~600 steps, 1:1 interleave — odd steps: e170's F2-install
    batch VERBATIM; even steps: the same batch structure with the 16 install
    windows drawn from e113's F1 jitter pool (ZEPHYRA at home, 7-char mask).
    Anchors NEUTRAL in both. Same optimizer/clip/schedule. F2 exposure
    matched to arm (A)'s 300 install batches."""
    net = copy.deepcopy(net0).to(DEV)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    n_anc = anchor.shape[0]
    assert n_anc == 16, f"anchor bank must stay 16 windows (got {n_anc})"
    traj, snaps, t_start = [], {}, time.time()
    step = 0
    evl = copy.deepcopy(net0)
    ckpt_at = set(CKPT_B_STEPS if not SMOKE else (4, 8, 12))
    n_f2 = 0
    for step in range(1, FT_STEPS_B + 1):
        f2_step = (step % 2 == 1)
        pool_x, pool_mask = (pool2_x, pool2_mask) if f2_step else (jit_x, jit_mask)
        ix = torch.randint(pool_x.shape[0], (NAME_BS,), generator=gen)
        aj = torch.randint(n_anc, (ANCH_BS // 2,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (ANCH_BS // 2,),
                           generator=gen)
        loss = make_batch(net, pool_x, pool_mask, ix, anchor, aj,
                          train_ids, rj)
        opt_step(net, opt, loss)
        if f2_step:
            n_f2 += 1
        if step in ckpt_at and step not in snaps:
            sd = snap_sd(net)
            snaps[step] = sd
            save_ckpt(f"e174_rehearsal_s{step}", sd,
                      {"arm": "B_rehearsal", "step": step, "f2_dose": n_f2,
                       "f1_replay_batches": step - n_f2, "seed": seed,
                       "desc": f"e131_consolidated + interleaved install "
                               f"({n_f2} F2 batches + {step - n_f2} F1-replay "
                               f"batches, 1:1, neutral anchors) — e174 "
                               f"rehearsal checkpoint",
                       "base": f"runs/checkpoints/{ROOT_CK}"})
        if step % EVAL_EVERY_B == 0 or step == FT_STEPS_B or \
                (time.time() - t_start) > FT_TIME_CAP_B:
            evl.load_state_dict({k: v.detach().cpu().clone()
                                 for k, v in net.state_dict().items()})
            evl.eval()
            bm = battery_cell(evl, f_eval_ids, mid)
            bz = battery_cell(evl, f1_eval_ids, zid)
            ce_r = ce_fixed_cpu(evl, *r_eval_xy)
            traj.append({"step": step, "f2_dose": n_f2,
                         "p_name_mean": bm["mean_pz"],
                         "p_f1_g0": bz["mean_pz"],
                         "frac_argmax_f2": bm["frac_argmax_z"],
                         "ce_r": ce_r,
                         "elapsed_s": round(time.time() - t_start, 1)})
            log(f"  [{tag}] s{step:4d} (F2x{n_f2:3d}) p(M@site) "
                f"{bm['mean_pz']:.4f} p(Z@g0) {bz['mean_pz']:.4f} "
                f"CE_R {ce_r:.4f} ({traj[-1]['elapsed_s']:.0f}s)")
        if (time.time() - t_start) > FT_TIME_CAP_B:
            log(f"  [{tag}] time cap {FT_TIME_CAP_B:.0f}s at s{step}")
            trims.append(f"{tag}: time cap at step {step} (F2 dose {n_f2})")
            break
    net.eval()
    if step not in snaps:
        sd = snap_sd(net)
        snaps[step] = sd
        save_ckpt("e174_rehearsal", sd,
                  {"arm": "B_rehearsal", "step": step, "f2_dose": n_f2,
                   "f1_replay_batches": step - n_f2, "seed": seed,
                   "desc": "e174 arm B final (time-cap trim)",
                   "base": f"runs/checkpoints/{ROOT_CK}"})
    del net
    return {"snaps": snaps, "traj": traj, "steps_ran": step,
            "f2_batches": n_f2, "seed": seed}


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e174_smoke" if SMOKE else "e174")
    log(f"E174 F2-DOSE LADDER + REHEARSAL (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY mandated, torch threads {torch.get_num_threads()}, "
        f"cuda available = {torch.cuda.is_available()}")

    # ---------------- protocol rebuild (e143/e151/e154/e170 verbatim)
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

    # ---------------- F2 install pool: offset j=-66 (e170 verbatim)
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

    # ---------------- F1 replay pool: e113's jitter recipe VERBATIM
    jit_x, jit_mask = {}, {}
    for j in JITTERS:
        wins = []
        for p, h in install_occ:
            pre = train_ids[p - PRE - j: p]
            post = train_ids[p + len(h): p + len(h) + POST_CAP - j]
            w = torch.cat([pre, name1_ids, post])
            if len(w) != BLOCK:
                raise RuntimeError(f"F1 jitter window len {len(w)} != {BLOCK} "
                                   f"at offset {j}")
            wins.append(w)
        jit_x[j] = torch.stack(wins)
        m = torch.zeros(len(wins), BLOCK - 1, dtype=torch.bool)
        m[:, PRE - 1 + j: PRE - 1 + j + len(NAME1)] = True
        jit_mask[j] = m
    jit_pool_x = torch.cat([jit_x[j] for j in JITTERS])       # (300, 256)
    jit_pool_mask = torch.cat([jit_mask[j] for j in JITTERS])
    G_JIT = {
        "jitters": list(JITTERS), "pool_shape": list(jit_pool_x.shape),
        "name_in_place_all": bool(all(
            torch.equal(w[PRE + j: PRE + j + len(NAME1)], name1_ids)
            for j in JITTERS for w in jit_x[j])),
        "mask_targets_per_window": int(jit_pool_mask[0].sum()),
        "masks_vary_with_jitter": bool(
            len({int(jit_mask[j][0].nonzero()[0]) for j in JITTERS}) ==
            len(JITTERS)),
        "stays_in_f1_band": bool(all(
            121 <= PRE - 1 + j <= 137 for j in JITTERS)),   # e113's onset-row
                                                            # span 121..137
        "note": ("e113's jitter set reads F1 at onset rows 129+j in "
                 "[121,137] (the grown-address band); the +8 span reaches "
                 "row 143 — e113 VERBATIM arithmetic"),
    }
    G_JIT["pass"] = bool(G_JIT["name_in_place_all"]
                         and G_JIT["mask_targets_per_window"] == len(NAME1)
                         and G_JIT["masks_vary_with_jitter"]
                         and G_JIT["stays_in_f1_band"]
                         and jit_pool_x.shape[0] == 300)
    assert G_JIT["pass"], f"F1 jitter pool gate FAILED: {G_JIT}"
    log(f"F1 replay pool (e113 recipe): {tuple(jit_pool_x.shape)} "
        f"(offsets {list(JITTERS)}), name masks at y-cols "
        f"{[int(jit_mask[j][0].nonzero()[0]) for j in JITTERS]}, all inside "
        f"the F1 band 121..137")

    # ---------------- neutral anchor bank (e170's, seed 170)
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
    host_positions = [p for p in E43.find_occ(train_text, HOSTS[0])]
    host_positions += [p for p in E43.find_occ(train_text, HOSTS[1])]

    def junctions_covered(starts):
        return sum(1 for s in starts
                   if any(s <= p < s + BLOCK + 1 for p in host_positions))

    jc = junctions_covered(n_starts)
    G_ANCHOR = {
        "construction": ("16 plain corpus windows from train_ids, RNG seed "
                         f"{E170_ANCHOR_SEED}, rejection if [s, s+257) "
                         "contains FLORIZEL/ELIZABETH/ZEPH/MIRABEL (e170 "
                         "verbatim)"),
        "n_windows": 16, "block": BLOCK, "seed": E170_ANCHOR_SEED,
        "starts": n_starts, "tries": tries, "rejections": rejections,
        "forbidden": list(ANCHOR_FORBIDDEN),
        "windows_with_host_content": sum(
            1 for s in n_starts
            if any(f in train_text[s: s + BLOCK + 1]
                   for f in ANCHOR_FORBIDDEN)),
        "junctions_covered": jc,
        "starts_match_e170_stored": bool(n_starts == E170_BANK_STARTS_REF),
        "e170_starts_ref": E170_BANK_STARTS_REF,
    }
    G_ANCHOR["pass"] = bool(
        G_ANCHOR["windows_with_host_content"] == 0
        and G_ANCHOR["junctions_covered"] == 0
        and G_ANCHOR["starts_match_e170_stored"]
        and anchor.shape == (16, BLOCK))
    assert G_ANCHOR["pass"], f"anchor gate FAILED: {G_ANCHOR}"
    log(f"G_ANCHOR: neutral bank 16x{BLOCK} (seed {E170_ANCHOR_SEED}) — "
        f"host-content 0/16, junctions 0/16, starts MATCH e170's stored bank "
        f"bit-for-bit: PASS")

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

    # ---------------- root net + gate
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
        raise RuntimeError("root checkpoint failed its gate vs e131/e151/e170 cells")
    del ev

    # =====================================================================
    # the per-checkpoint measurement battery (light = dials + F2 site +
    # span census; full adds e170's full row sets + onset census)
    # =====================================================================
    def measure(sd: dict, tag: str, rows_cen, ctrl_rows, full: bool) -> dict:
        net = evl_load(sd)
        out: dict = {"tag": tag}
        out["base"] = {j: battery_cell(net, bat1[(j, "install60")], zid)
                       for j in GEOS}
        out["base_held30_g0"] = battery_cell(net, bat1[(0, "held30")], zid)
        out["ce_r"] = ce_fixed_cpu(net, *r_eval_xy)
        out["f2_leak_g0"] = battery_cell(net, ids130, mid)["mean_pz"]
        out["f1_leak_site"] = battery_cell(net, f2_eval_ids, zid)["mean_pz"]
        out["f2_site_install"] = read_fact_at(net, pool2, name2_ids, mid,
                                              F2_ADDR_ROW, F2_Z_XCOL)
        out["f2_site_held"] = read_fact_at(net, pool2_held, name2_ids, mid,
                                           F2_ADDR_ROW, F2_Z_XCOL)

        def span2_fn(n):
            return read_fact_at(n, pool2, name2_ids, mid, F2_ADDR_ROW,
                                F2_Z_XCOL)["pname_mean_over7"]

        out["censusF2_span"] = row_census(net, rows_cen, span2_fn)
        out["site63_span"] = site_summary(out["censusF2_span"], F2_ROWS,
                                          ctrl_rows)
        if full:
            def onset2_fn(n):
                return read_fact_at(n, pool2, name2_ids, mid, F2_ADDR_ROW,
                                    F2_Z_XCOL)["pz_onset_mean"]
            out["censusF2_onset"] = row_census(net, ROWS_F2_ONSET_FULL,
                                               onset2_fn)
            out["site63_onset"] = site_summary(out["censusF2_onset"], (63,),
                                               CTR2)
        g0 = out["base"][0]["mean_pz"]
        log(f"[{tag}] F1 g-12/g0/g+12 " + " ".join(
            f"{out['base'][j]['mean_pz']:.4f}" for j in GEOS)
            + f" | held30 {out['base_held30_g0']['mean_pz']:.4f} CE_R "
            f"{out['ce_r']:.4f} | F2 onset "
            f"{out['f2_site_install']['pz_onset_mean']:.4f} span "
            f"{out['f2_site_install']['pname_mean_over7']:.4f} (held "
            f"{out['f2_site_held']['pz_onset_mean']:.4f}) | graft S "
            f"{out['site63_span']['site_strength']:+.4f} @r"
            f"{out['site63_span']['peak_row']} (2x-ctrl "
            f"{out['site63_span']['bar_2x_control']:.4f}) site_pos "
            f"{out['site63_span']['site_pos']} | leaks p(M)@129 "
            f"{out['f2_leak_g0']:.4f} p(Z)@63 {out['f1_leak_site']:.4f}")
        del net
        return out

    log("=" * 78)
    log("BEFORE battery (root: e131_consolidated_e113)")
    rows_before = ROWS_CEN if not SMOKE else (0,) + F2_ROWS + (20, 40)
    before = measure(sd_root, "before", rows_before, CTRL_LIGHT, full=False)
    G_F2INST = {"onset": before["f2_site_install"]["pz_onset_mean"],
                "ref_onset": G_F2INST_REF_ONSET,
                "span": before["f2_site_install"]["pname_mean_over7"],
                "ref_span": G_F2INST_REF_SPAN,
                "tol": G_BIT_TOL,
                "pass": bool(
                    abs(before["f2_site_install"]["pz_onset_mean"]
                        - G_F2INST_REF_ONSET) < G_FALLBACK_TOL
                    and abs(before["f2_site_install"]["pname_mean_over7"]
                            - G_F2INST_REF_SPAN) < G_FALLBACK_TOL)}
    log(f"G_F2INST (F2 site instrument vs e170 before): onset "
        f"{G_F2INST['onset']:.10f} (ref {G_F2INST_REF_ONSET:.10f}) span "
        f"{G_F2INST['span']:.10f} (ref {G_F2INST_REF_SPAN:.10f}): "
        f"{'PASS' if G_F2INST['pass'] else 'FAIL'}")
    if not G_F2INST["pass"]:
        raise RuntimeError("F2 site instrument drifted from e170's before cells")
    time.sleep(STAGGER_S)
    log("gates so far: G_NONCE, G_SPLICE, G_GEO2, G_JIT, G_ANCHOR, G_CONS, "
        "G_F2INST all PASS")

    # =====================================================================
    # ARM (A): the dose ladder (training 1 of 2; CPU; cooldown-wrapped)
    # =====================================================================
    log("=" * 78)
    if not SMOKE:
        log(f"[cooldown] {COOLDOWN_S:.0f}s before training 1/2")
        cooldown(COOLDOWN_S)
    log(f"ARM (A) DOSE LADDER: e170's neutral install, in-run checkpoints at "
        f"F2-doses {list(CKPT_DOSES)} (seed {F2_SEED}, device {DEV})")
    armA = finetune_ladder("ladderA", net0, pool2, pool2_mask, anchor,
                           train_ids, r_eval_xy, f2_eval_ids, mid, F2_SEED)
    if not SMOKE:
        log(f"[cooldown] {COOLDOWN_S:.0f}s after training 1/2")
        cooldown(COOLDOWN_S)
    time.sleep(STAGGER_S)

    # ---- G_REPRO: arm A's in-loop traj vs e170's stored install traj
    e170_cmp: dict = {"path": str(E170_METRICS), "loaded": False}
    try:
        e170m = json.loads(E170_METRICS.read_text(encoding="utf-8"))
        e170_cmp["loaded"] = True
        e170_traj = {t["step"]: t["p_name_mean"]
                     for t in e170m["f2"]["install"]["traj"]}
        cells = {}
        for t in armA["traj"]:
            if t["step"] in e170_traj:
                cells[t["step"]] = {"this_run": t["p_name_mean"],
                                    "e170": e170_traj[t["step"]],
                                    "diff": t["p_name_mean"]
                                            - e170_traj[t["step"]]}
        maxdiff = max((abs(c["diff"]) for c in cells.values()), default=0.0)
        G_REPRO = {"cells": cells, "n_cells": len(cells), "max_abs_diff": maxdiff,
                   "tol": G_REPRO_TOL, "tight_tol": G_REPRO_TIGHT,
                   "bit_reproducible": bool(0 < len(cells) and maxdiff < G_BIT_TOL),
                   "tight": bool(0 < len(cells) and maxdiff < G_REPRO_TIGHT),
                   "pass": bool(len(cells) > 0 and maxdiff < G_REPRO_TOL)}
        log(f"G_REPRO arm-A traj vs e170 stored ({len(cells)} common steps): "
            f"max|diff| {maxdiff:.2e} (tol {G_REPRO_TOL}): "
            f"{'PASS' if G_REPRO['pass'] else 'FAIL'}"
            f"{' [bit-exact]' if G_REPRO['bit_reproducible'] else ''}")
        e170_cmp["after_g0"] = e170m["after"]["base"]["0"]["mean_pz"]
        e170_cmp["after_steps"] = e170m["f2"]["install"]["steps_ran"]
        e170_cmp["before_match"] = bool(
            abs(e170m["before"]["base"]["0"]["mean_pz"]
                - before["base"][0]["mean_pz"]) < 1e-9)
    except Exception as e:  # noqa: BLE001
        G_REPRO = {"pass": None, "error": str(e),
                   "note": "e170 metrics unavailable; stream identity rests "
                           "on seed/moduli/code lineage"}
        log(f"e170 metrics unavailable for G_REPRO: {e}")
    if G_REPRO.get("pass") is False and not SMOKE:
        raise RuntimeError(f"arm A failed to reproduce e170's install "
                           f"trajectory: {G_REPRO['max_abs_diff']}")
    time.sleep(STAGGER_S)

    # ---- measure the ladder checkpoints
    log("--- ARM (A) checkpoint measurements (the dose ladder) ---")
    ladder_A: dict = {}
    for dose in sorted(armA["snaps"]):
        ladder_A[dose] = measure(armA["snaps"][dose], f"A_dose{dose}",
                                 ROWS_CEN, CTRL_LIGHT,
                                 full=(dose == max(armA["snaps"])))
        time.sleep(STAGGER_S)

    # =====================================================================
    # ARM (B): the rehearsal cell (training 2 of 2; CPU; cooldown-wrapped)
    # =====================================================================
    log("=" * 78)
    if not SMOKE:
        log(f"[cooldown] {COOLDOWN_S:.0f}s before training 2/2")
        cooldown(COOLDOWN_S)
    log(f"ARM (B) REHEARSAL: 600 steps alternating F2-install / F1-replay "
        f"(1:1, e113 jitter windows, neutral anchors, F2 exposure matched to "
        f"arm A's 300; seed {F2_SEED})")
    armB = finetune_rehearsal("rehearsalB", net0, pool2, pool2_mask,
                              jit_pool_x, jit_pool_mask, anchor, train_ids,
                              r_eval_xy, f2_eval_ids, ids130, zid, mid,
                              F2_SEED)
    if not SMOKE:
        log(f"[cooldown] {COOLDOWN_S:.0f}s after training 2/2")
        cooldown(COOLDOWN_S)
    time.sleep(STAGGER_S)

    log("--- ARM (B) checkpoint measurements (matched F2-doses) ---")
    ladder_B: dict = {}
    for stp in sorted(armB["snaps"]):
        dose = (stp + 1) // 2                       # F2 batches by step stp
        ladder_B[dose] = measure(armB["snaps"][stp], f"B_dose{dose}_s{stp}",
                                 ROWS_CEN, CTRL_LIGHT,
                                 full=(stp == max(armB["snaps"])))
        time.sleep(STAGGER_S)

    # =====================================================================
    # ADJUDICATION (registered clauses, QUEUE e174 verbatim; no bar shopping)
    # =====================================================================
    def g0(m):
        return m["base"][0]["mean_pz"]

    def g12(m):
        return m["base"][-12]["mean_pz"]

    def f2on(m):
        return m["f2_site_install"]["pz_onset_mean"]

    doses_A = sorted(ladder_A)
    doses_B = sorted(ladder_B)
    curve_A = {d: {"f1_g0": g0(ladder_A[d]), "f1_gm12": g12(ladder_A[d]),
                   "f2_onset": f2on(ladder_A[d]),
                   "f2_span": ladder_A[d]["f2_site_install"]["pname_mean_over7"],
                   "f2_graft_strength":
                       ladder_A[d]["site63_span"]["site_strength"],
                   "site_pos": ladder_A[d]["site63_span"]["site_pos"]}
               for d in doses_A}
    curve_B = {d: {"f1_g0": g0(ladder_B[d]), "f1_gm12": g12(ladder_B[d]),
                   "f2_onset": f2on(ladder_B[d]),
                   "f2_span": ladder_B[d]["f2_site_install"]["pname_mean_over7"],
                   "f2_graft_strength":
                       ladder_B[d]["site63_span"]["site_strength"],
                   "site_pos": ladder_B[d]["site63_span"]["site_pos"]}
               for d in doses_B}

    graft_forming = [d for d in doses_A if curve_A[d]["f2_onset"] >= GRAFT_BAR]
    winwin_doses = [d for d in doses_A
                    if curve_A[d]["f2_onset"] >= GRAFT_BAR
                    and curve_A[d]["f1_g0"] >= SURVIVE_FLOOR]
    allornothing = (len(graft_forming) > 0
                    and all(curve_A[d]["f1_g0"] <= DIE_CEIL
                            for d in graft_forming))
    if len(winwin_doses) > 0:
        ladder_verdict = "WIN-WIN"
        ladder_clause = (f"a dose exists with F2 onset >= 0.5 AND F1 g0 >= 0.5 "
                         f"(win-win doses: {winwin_doses}; e.g. dose "
                         f"{winwin_doses[0]}: F2 "
                         f"{curve_A[winwin_doses[0]]['f2_onset']:.4f}, F1 "
                         f"{curve_A[winwin_doses[0]]['f1_g0']:.4f}) — capacity "
                         f"is a budget dial.")
    elif allornothing:
        ladder_verdict = "ALL-OR-NOTHING"
        worst = graft_forming[-1]
        ladder_clause = (f"F1 dies (<= 0.2) at every graft-forming dose "
                         f"({graft_forming}; e.g. dose {worst}: F2 onset "
                         f"{curve_A[worst]['f2_onset']:.4f} vs F1 g0 "
                         f"{curve_A[worst]['f1_g0']:.4f}) — the overwrite is "
                         f"formation-triggered, a cliff.")
    else:
        ladder_verdict = "TEXTURE"
        ladder_clause = ("the ladder lands in no registered band (graft-"
                         f"forming doses {graft_forming}; F1 g0 per dose: "
                         + ", ".join(f"{d}:{curve_A[d]['f1_g0']:.4f}"
                                     for d in doses_A)
                         + ") — texture with the curve.")

    kill_doses_A = [d for d in doses_A if curve_A[d]["f1_g0"] <= DIE_CEIL]
    dose_kill_A = min(kill_doses_A) if kill_doses_A else None
    if dose_kill_A is None:
        rehearsal_verdict = "MOOT-NO-KILL"
        rehearsal_clause = ("arm (A) never killed F1 at any measured dose "
                            "(no F1 g0 <= 0.2) — the registered rehearsal "
                            "fork has no kill to rescue; recorded, not "
                            "shopped.")
    elif dose_kill_A in curve_B:
        b_g0 = curve_B[dose_kill_A]["f1_g0"]
        b_on = curve_B[dose_kill_A]["f2_onset"]
        if b_g0 >= SURVIVE_FLOOR:
            rehearsal_verdict = "REHEARSAL-RESOLVES"
            rehearsal_clause = (f"in arm (B) F1 holds >= 0.5 at the F2-dose "
                                f"where (A) killed it (dose {dose_kill_A}: "
                                f"A g0 {curve_A[dose_kill_A]['f1_g0']:.4f} -> "
                                f"B g0 {b_g0:.4f}; F2 onset in B there "
                                f"{b_on:.4f}"
                                + ("" if b_on >= GRAFT_BAR else
                                   " — graft NOT yet formed in B at that "
                                   "dose: slower-kill texture, bars as "
                                   "registered")
                                + ").")
        else:
            rehearsal_verdict = "REHEARSAL-FAILS"
            rehearsal_clause = (f"in arm (B) F1 does NOT hold >= 0.5 at the "
                                f"F2-dose where (A) killed it (dose "
                                f"{dose_kill_A}: A g0 "
                                f"{curve_A[dose_kill_A]['f1_g0']:.4f} -> B "
                                f"g0 {b_g0:.4f}) — interleaved F1-replay does "
                                f"not save the tenant at that dose.")
    else:
        rehearsal_verdict = "MOOT-DOSE-MISS"
        rehearsal_clause = (f"dose_kill(A) = {dose_kill_A} has no matched "
                            f"arm-(B) checkpoint (arm B doses measured: "
                            f"{doses_B}) — recorded, not shopped.")

    cohab_B = [d for d in doses_B
               if curve_B[d]["f2_onset"] >= GRAFT_BAR
               and curve_B[d]["f1_g0"] >= SURVIVE_FLOOR]
    verdict = f"{ladder_verdict} + {rehearsal_verdict}"
    log("=" * 78)
    log(f"E174 VERDICT: {verdict}")
    log(f"  ladder: {ladder_clause}")
    log(f"  rehearsal: {rehearsal_clause}")
    log(f"  curve (A) dose: " + " | ".join(
        f"{d}: F1 {curve_A[d]['f1_g0']:.4f} F2 {curve_A[d]['f2_onset']:.4f} "
        f"S {curve_A[d]['f2_graft_strength']:+.4f}" for d in doses_A))
    log(f"  curve (B) dose: " + " | ".join(
        f"{d}: F1 {curve_B[d]['f1_g0']:.4f} F2 {curve_B[d]['f2_onset']:.4f} "
        f"S {curve_B[d]['f2_graft_strength']:+.4f}" for d in doses_B))
    if cohab_B:
        log(f"  arm (B) cohabitation doses (F2 >= 0.5 AND F1 >= 0.5): "
            f"{cohab_B}")
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e174_dose_rehearsal",
        "date": common.now_iso(),
        "registration": ("QUEUE.md row e174 + the dispatch's registered "
                         "prediction VERBATIM (e170's gated follow-up; gate "
                         "fired OVERWRITE-REAL); operationalizations frozen "
                         "in the module docstring before compute"),
        "registered_prediction": REGISTERED_PREDICTION,
        "committed_prediction": REGISTERED_PREDICTION["committed"],
        "prediction_held": None,
        "question": ("is capacity a DIAL (a smaller F2 dose cohabits with "
                     "F1) or a CLIFF (F1 dies at every graft-forming dose), "
                     "and does interleaved F1-replay during F2's install "
                     "save the tenant? — e170's neutral-anchor install "
                     "laddered over F2 dose, plus a 1:1 F1-replay arm at "
                     "matched F2 exposure"),
        "root": f"runs/checkpoints/{ROOT_CK} (loaded, gated vs e131/e151/"
                "e170 cells)",
        "f2": {"name": NAME2, "kind": "nonce (e134 design; e154/e170's F2)",
               "corpus_occurrences": G_NONCE["corpus_occurrences"],
               "placement": {"offset": F2_J,
                             "name_xcols": [F2_Z_XCOL, F2_Z_XCOL + 6],
                             "read_rows": [63, 69], "site_band": list(F2_ROWS),
                             "continuation_len": F2_CONT}},
        "f1_replay": {"name": NAME1, "recipe": "e113 arm-(a) jitter pool "
                      "VERBATIM (60 install hosts x offsets "
                      f"{list(JITTERS)}, ZEPHYRA spliced, 7 name-char mask, "
                      "home geometry rows 121..137)",
                      "pool_shape": list(jit_pool_x.shape)},
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
                     "mask": "7 name-char targets per window (name-only)",
                     "anchor_bank": "neutral (e170's, seed 170, verified "
                                    "bit-identical starts)",
                     "optimizer": "AdamW (0.9,0.95) wd 0.1 lr 1e-3 constant "
                                  "clip 1.0; batch 32 = 16 install + 16 "
                                  "anchors (8 paired neutral + 8 random)",
                     "geometries_measured": {"novel": [-12, 12],
                                             "trained_g0": 0}},
        "gates": {"G_NONCE": G_NONCE, "G_SPLICE": G_SPLICE, "G_GEO2": G_GEO2,
                  "G_JIT": G_JIT, "G_ANCHOR": G_ANCHOR, "G_CONS": G_CONS,
                  "G_F2INST": G_F2INST, "G_REPRO": G_REPRO},
        "before": before,
        "arms": {
            "A_dose_ladder": {
                "desc": "e170's neutral-anchor install VERBATIM (seed 10902, "
                        "e170-identical RNG stream) + no-RNG snapshots",
                "steps_ran": armA["steps_ran"], "seed": armA["seed"],
                "time_cap_s": FT_TIME_CAP_A, "traj": armA["traj"],
                "ckpt_doses": doses_A},
            "B_rehearsal": {
                "desc": "600 steps 1:1 alternating F2-install (e170 batch) / "
                        "F1-replay (e113 jitter windows); neutral anchors in "
                        "both; F2 exposure matched to arm A's 300 batches",
                "steps_ran": armB["steps_ran"], "seed": armB["seed"],
                "f2_batches": armB["f2_batches"],
                "time_cap_s": FT_TIME_CAP_B, "traj": armB["traj"],
                "ckpt_doses": doses_B},
        },
        "ladder_A": {str(d): ladder_A[d] for d in doses_A},
        "ladder_B": {str(d): ladder_B[d] for d in doses_B},
        "curve_A": {str(d): curve_A[d] for d in doses_A},
        "curve_B": {str(d): curve_B[d] for d in doses_B},
        "e170_comparison": e170_cmp,
        "adjudication": {
            "bars": {"SURVIVE_FLOOR": SURVIVE_FLOOR, "GRAFT_BAR": GRAFT_BAR,
                     "DIE_CEIL": DIE_CEIL},
            "ladder": {"graft_forming_doses": graft_forming,
                       "winwin_doses": winwin_doses,
                       "allornothing": bool(allornothing),
                       "verdict": ladder_verdict, "clause": ladder_clause},
            "rehearsal": {"dose_kill_A": dose_kill_A,
                          "kill_doses_A": kill_doses_A,
                          "armB_cohabitation_doses": cohab_B,
                          "verdict": rehearsal_verdict,
                          "clause": rehearsal_clause},
            "verdict": verdict,
        },
        "honesty_reflex": {
            "dose_steps_proxy": ("dose = COUNT of F2-install batches "
                                 "(optimizer steps of F2 exposure), the "
                                 "dispatch's own axis; a 'dose' is not a "
                                 "quantity of parameter change — the "
                                 "per-checkpoint F2 onset/graft columns "
                                 "measure what the dose actually bought"),
            "interleaved_accounting": ("arm (B) matches arm (A)'s 300 F2 "
                                       "batches but runs 600 optimizer steps "
                                       "total and adds the F1 channel's own "
                                       "gradients (its own wash/interference "
                                       "pressure) — matched-dose is NOT "
                                       "matched optimizer state; if F1 "
                                       "survives in (B) while F2's onset at "
                                       "the same dose lags (A)'s, part of "
                                       "the 'rescue' is slower F2 "
                                       "acquisition, and the co-reported "
                                       "arm-B F2-onset column separates the "
                                       "two only partially"),
            "single_seed_single_lineage": ("both arms single seed (10902) on "
                                           "one root (e131 consolidated) — "
                                           "n=1 per arm; e152R showed "
                                           "trajectory texture is "
                                           "seed-dependent (though e170's "
                                           "demolition and e152R's conversion "
                                           "outcome were invariant)"),
            "graft_formation_is_measured": ("F2's formation at every dose is "
                                            "adjudicated by the site census "
                                            "(2x-control bar) and expression "
                                            "cells, not assumed; arm A's "
                                            "e170-lineage is additionally "
                                            "gated by G_REPRO"),
            "novel_geometry_caveat_T087": ("F1 g-12 retention conflates 'the "
                                           "route survived' with 'sink "
                                           "health at the novel geometry "
                                           "survived' (T087/E150 caveat)"),
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

    plot(rd / "dose_rehearsal.png", before, ladder_A, ladder_B, curve_A,
         curve_B, verdict, ladder_verdict, ladder_clause, rehearsal_verdict,
         rehearsal_clause, armA, armB, graft_forming, dose_kill_A, cohab_B)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'dose_rehearsal.png'}, "
        f"ckpts {len(CKPT_INVENTORY)} (e174_dose*, e174_rehearsal*)")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot

def plot(path, before, ladder_A, ladder_B, curve_A, curve_B, verdict,
         ladder_verdict, ladder_clause, rehearsal_verdict, rehearsal_clause,
         armA, armB, graft_forming, dose_kill_A, cohab_B):
    """THE capacity trade-off curve + the rehearsal overlay + verdict."""
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.5))
    dA = sorted(curve_A)
    dB = sorted(curve_B)
    g0b = before["base"][0]["mean_pz"]
    onb = before["f2_site_install"]["pz_onset_mean"]

    # (0,0) THE capacity trade-off curve: F1 g0 vs F2 site onset
    ax = axes[0, 0]
    ax.add_patch(plt.Rectangle((GRAFT_BAR, SURVIVE_FLOOR), 1 - GRAFT_BAR,
                               1 - SURVIVE_FLOOR, color="seagreen",
                               alpha=0.10))
    ax.text(0.75, 0.93, "WIN-WIN quadrant", fontsize=7.5, color="seagreen",
            ha="center")
    xs = [curve_A[d]["f2_onset"] for d in dA]
    ys = [curve_A[d]["f1_g0"] for d in dA]
    ax.plot(xs, ys, "o-", ms=7, lw=1.6, color="crimson", label="arm A: dose ladder")
    for d in dA:
        ax.annotate(f"{d}", (curve_A[d]["f2_onset"], curve_A[d]["f1_g0"]),
                    textcoords="offset points", xytext=(7, -3), fontsize=8)
    xb = [curve_B[d]["f2_onset"] for d in dB]
    yb = [curve_B[d]["f1_g0"] for d in dB]
    ax.plot(xb, yb, "s--", ms=6, lw=1.4, color="royalblue",
            label="arm B: +F1 rehearsal (matched F2 dose)")
    for d in dB:
        ax.annotate(f"{d}", (curve_B[d]["f2_onset"], curve_B[d]["f1_g0"]),
                    textcoords="offset points", xytext=(7, 5), fontsize=8,
                    color="royalblue")
    ax.plot([onb], [g0b], "*", ms=15, color="dimgray", label="before (root)")
    if dose_kill_A is not None and dose_kill_A in curve_B:
        ax.annotate("", xy=(curve_B[dose_kill_A]["f2_onset"],
                            curve_B[dose_kill_A]["f1_g0"]),
                    xytext=(curve_A[dose_kill_A]["f2_onset"],
                            curve_A[dose_kill_A]["f1_g0"]),
                    arrowprops=dict(arrowstyle="->", lw=1.4,
                                    color="royalblue", alpha=0.55))
        ax.text(0.02, 0.03, f"arrow: dose {dose_kill_A} (A kills F1) under "
                f"rehearsal", fontsize=7, color="royalblue",
                transform=ax.transAxes)
    ax.axhline(SURVIVE_FLOOR, ls="--", lw=1.1, color="seagreen")
    ax.text(1.01, SURVIVE_FLOOR, "F1 bar 0.5", fontsize=6.5,
            color="seagreen", va="center")
    ax.axhline(DIE_CEIL, ls="--", lw=1.1, color="firebrick")
    ax.text(1.01, DIE_CEIL, "F1 dies 0.2", fontsize=6.5, color="firebrick",
            va="center")
    ax.axvline(GRAFT_BAR, ls=":", lw=1.2, color="black")
    ax.text(GRAFT_BAR, 1.03, "graft-forming 0.5", fontsize=6.5, ha="center")
    ax.set_xlabel("F2 site onset p(M)@63 (install pool)")
    ax.set_ylabel("F1 g0 p(Z) install-60")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.08)
    ax.set_title(f"THE CAPACITY TRADE-OFF CURVE — F1 vs F2 across doses "
                 f"(labels = F2 dose)", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="center right")
    ax.grid(alpha=0.25)

    # (0,1) dose curves: F1 g0 and F2 onset vs F2 dose, both arms
    ax = axes[0, 1]
    ax.plot(dA, [curve_A[d]["f1_g0"] for d in dA], "o-", ms=5, lw=1.5,
            color="crimson", label="A: F1 g0")
    ax.plot(dB, [curve_B[d]["f1_g0"] for d in dB], "s--", ms=5, lw=1.3,
            color="royalblue", label="B: F1 g0 (rehearsal)")
    ax.axhline(SURVIVE_FLOOR, ls="--", lw=1.0, color="seagreen")
    ax.axhline(DIE_CEIL, ls="--", lw=1.0, color="firebrick")
    ax.axhline(g0b, ls=":", lw=1.0, color="dimgray")
    ax.text(max(dA) * 1.01, g0b, f"before {g0b:.3f}", fontsize=6.5,
            color="dimgray", va="center")
    ax.set_xlabel("F2 dose (install batches)")
    ax.set_ylabel("F1 g0 p(Z)")
    ax.set_ylim(-0.03, 1.05)
    ax2 = ax.twinx()
    ax2.plot(dA, [curve_A[d]["f2_onset"] for d in dA], "o:", ms=4, lw=1.1,
             color="darkorange", label="A: F2 onset")
    ax2.plot(dB, [curve_B[d]["f2_onset"] for d in dB], "s:", ms=4, lw=1.1,
             color="gold", label="B: F2 onset")
    ax2.axhline(GRAFT_BAR, ls=":", lw=1.0, color="black")
    ax2.set_ylabel("F2 site onset p(M)@63", color="darkorange")
    ax2.set_ylim(-0.03, 1.05)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
    ax.set_title("dose curves — F1 (left axis) and F2 (right axis) per arm",
                 fontsize=9.5)
    ax.grid(alpha=0.25)

    # (0,2) verdict panel
    ax = axes[0, 2]
    ax.axis("off")
    vlines = [
        "REGISTERED (QUEUE e174 verbatim; no bar shopping):",
        "  WIN-WIN: a dose with F2 onset >= 0.5 AND F1 g0 >= 0.5",
        "  ALL-OR-NOTHING: F1 dies (<= 0.2) at every graft-forming",
        "   dose (F2 onset >= 0.5) — the overwrite is a cliff",
        "  REHEARSAL-RESOLVES: in (B) F1 holds >= 0.5 at the dose",
        "   where (A) killed it; REHEARSAL-FAILS if not",
        "",
        "LADDER (arm A): " + ladder_verdict,
    ] + [f"  {wd}" for wd in
         [ladder_clause[i:i + 62] for i in range(0, len(ladder_clause), 62)]]
    vlines += ["", f"REHEARSAL (arm B): {rehearsal_verdict}"]
    vlines += [f"  {wd}" for wd in
               [rehearsal_clause[i:i + 62]
                for i in range(0, len(rehearsal_clause), 62)]]
    vlines += ["", f"VERDICT: {verdict}"]
    for i, tx in enumerate(vlines):
        ax.text(0.02, 0.97 - i * 0.030, tx, fontsize=6.9, va="top",
                family="monospace")

    # (1,0) graft strength (site census) vs dose
    ax = axes[1, 0]
    w = 0.36
    xs = np.arange(len(dA))
    ax.bar(xs - w / 2, [curve_A[d]["f2_graft_strength"] for d in dA], w,
           color="crimson", edgecolor="k", lw=0.4, label="A graft strength")
    ax.bar(xs + w / 2, [curve_B[d]["f2_graft_strength"] for d in dB], w,
           color="royalblue", edgecolor="k", lw=0.4,
           label="B graft strength")
    for k, d in enumerate(dA):
        ax.text(k - w / 2, curve_A[d]["f2_graft_strength"] + 0.0006,
                f"{curve_A[d]['f2_graft_strength']:.3f}", ha="center",
                fontsize=6.4, rotation=90, va="bottom")
    for k, d in enumerate(dB):
        ax.text(k + w / 2, curve_B[d]["f2_graft_strength"] + 0.0006,
                f"{curve_B[d]['f2_graft_strength']:.3f}", ha="center",
                fontsize=6.4, rotation=90, va="bottom", color="royalblue")
    ax.plot(xs, [2.0 * ladder_A[d]["site63_span"]["control_max"]
                 for d in dA], "k^--", ms=4, lw=0.9,
            label="2x control (A censuses)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"dose {d}" for d in dA], fontsize=8)
    ax.set_ylabel("site census strength min(mean,zero) drop")
    ax.set_title("F2 graft strength (span census, rows 63..69) per dose",
                 fontsize=9.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, axis="y")

    # (1,1) F1 full dial set per dose (arm A) + rehearsal overlay
    ax = axes[1, 1]
    for j, col, mk in ((-12, "seagreen", "o"), (0, "crimson", "o"),
                       (12, "darkviolet", "o")):
        ax.plot(dA, [ladder_A[d]["base"][j]["mean_pz"] for d in dA],
                f"{mk}-", ms=4, lw=1.2, color=col, label=f"A g{j:+d}")
    ax.plot(dA, [ladder_A[d]["base_held30_g0"]["mean_pz"] for d in dA],
            "v-", ms=4, lw=1.0, color="gray", label="A held30 g0")
    ax.plot(dB, [ladder_B[d]["base"][0]["mean_pz"] for d in dB], "s--",
            ms=5, lw=1.4, color="royalblue", label="B g0 (rehearsal)")
    ax.plot(dB, [ladder_B[d]["base"][-12]["mean_pz"] for d in dB], "s--",
            ms=4, lw=1.0, color="teal", label="B g-12 (rehearsal)")
    ax.axhline(SURVIVE_FLOOR, ls="--", lw=1.0, color="seagreen")
    ax.axhline(DIE_CEIL, ls="--", lw=1.0, color="firebrick")
    ax.set_xlabel("F2 dose (install batches)")
    ax.set_ylabel("F1 battery p(Z)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("F1 dial set across the ladder (g-12/g0/g+12/held30) "
                 "+ rehearsal overlay", fontsize=9.5)
    ax.legend(fontsize=7, loc="center right")
    ax.grid(alpha=0.25)

    # (1,2) in-loop trajectories
    ax = axes[1, 2]
    ta = armA["traj"]
    ax.plot([t["step"] for t in ta], [t["p_name_mean"] for t in ta], "o-",
            ms=3, lw=1.2, color="crimson", label="A p(M@63) in-loop")
    tb = armB["traj"]
    ax.plot([t["step"] for t in tb], [t["p_name_mean"] for t in tb], "s--",
            ms=3, lw=1.1, color="royalblue", label="B p(M@63) (x = step)")
    ax.plot([t["step"] for t in tb], [t["p_f1_g0"] for t in tb], "^:",
            ms=3, lw=1.1, color="teal", label="B p(Z@g0) in-loop")
    ax.axhline(GRAFT_BAR, ls=":", lw=1.0, color="black")
    ax.axhline(SURVIVE_FLOOR, ls="--", lw=0.9, color="seagreen")
    ax.set_xlabel("optimizer step (B: 1:1 interleave, so F2 dose = step/2)")
    ax.set_ylabel("p(M@63) / p(Z)@g0")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(f"in-loop trajectories — A {armA['steps_ran']} steps / "
                 f"B {armB['steps_ran']} steps ({armB.get('f2_batches', '?')} "
                 f"F2 batches), seed 10902", fontsize=9.5)
    ax.legend(fontsize=7, loc="center right")
    ax.grid(alpha=0.25)

    fig.suptitle(f"E174 — THE F2-DOSE LADDER + REHEARSAL CELL (root "
                 f"e131_consolidated + MIRABEL @ 63..69, neutral anchors) "
                 f"-> {verdict}", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
