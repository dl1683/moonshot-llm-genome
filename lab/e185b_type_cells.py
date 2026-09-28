"""E185B — THE NEUTRAL-STREAM TYPE CELLS (R52's grid gap-closers; the lead
finding's "all types" claim rides an inference no run discharged).

WHY (T112/R52 + T113/T114, the grid): the lead finding — "no memory state
tested retains expression under continued training without the fact's
windows (3 streams, 2 lrs, 3 seeds, all types; one lineage)" — crosses the
type axis on an INFERENCE. The dwell type (e161, root e152_steps32) and the
site type (e177, root e151_twodoor) were each washed only on their OWN
single stream, and that stream was the EXTINCTION-GRADE one (e161/e176's
original anchor bank: the install's own name-deleted host windows, 16/16
junctions covered). The neutral-stream cell (e176N arm A: e170's plain-
corpus anchors, 0/16 junctions) exists ONLY for the consolidated root
(the sink-coupled type). THIS CELL fills the two empty cells: e176N arm
A's protocol VERBATIM on BOTH types. Both dissolve => the sparse union
becomes an honest cross and "all types" earns its name; either survives
=> "all types" is FALSE in the direction that matters (a memory type
that survives neutral streams).

REGISTERED BARS (frozen here before compute; the dispatch's registration
verbatim — QUEUE row e185b; no bar shopping; adjudicate against exactly
this):
  - BOTH-DISSOLVE fires if: the site type's onset and the dwell type's
    g-12 both fall under their bars by +50 — the sparse union becomes an
    honest cross; "all types" earns its name.
  - EITHER-SURVIVES fires if: either type retains (site >= 0.5 onset /
    dwell >= 0.5 g-12) through +300 — "all types" is FALSE in the
    direction that matters; a survivor exists.
  - No bar shopping; texture (different rates than their extinction runs)
    => TEXTURE with both curves.

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * "their bars" = each type's OWN prior adjudication's SHUT bar — both
    are 0.27 (e161's DISUSE primary g-12 <= 0.27; e177's SCRATCH-MEMORY
    primary onset(300) <= 0.27; e176N's NEUTRAL-DISSOLVES g-12 <= 0.27 by
    +50). Dissolve = primary dial <= 0.27 AT THE +50 CHECKPOINT (e176N's
    "by +50" convention); the earliest checkpoint <= 0.27 over
    {1,2,4,50,100,200,300} is CO-REPORTED as collapse timing.
  * "retains through +300" = primary dial >= 0.50 at EVERY continuation
    checkpoint {1,2,4,50,100,200,300} (step 0 = the root, trivially
    above; the dispatch's parenthetical defines the retention bars).
  * Primary dials: SITE type = read_fact_at onset (pz_onset_mean) on
    e152's locked j=54 pool at addr_row 183 / x-col 184 (e177's site
    battery verbatim); DWELL type = absolute install-60 battery mean
    p(Z) at ctx offset -12 (e161's g-12 verbatim). The other dial is
    co-reported on BOTH nets each checkpoint (free, same instruments).
  * The clauses are disjoint (dissolve requires <= 0.27 by +50; survive
    requires >= 0.50 at every checkpoint including +50); order
    BOTH-DISSOLVE -> EITHER-SURVIVES -> TEXTURE; every sub-boolean
    reported regardless; a split outcome that fires neither clause is
    TEXTURE with both curves and per-type fates stated.
  * CE_R per checkpoint on both nets (the wash's price channel); the
    in-batch corpus CE recorded at every checkpoint.

NETS (both gated vs their stored cells before any training):
  * runs/checkpoints/e151_twodoor.pt — the SITE type (e151's buried
    site-endpoint; primary dial onset, root 0.9981); gated vs e151's
    stored after-cells.
  * runs/checkpoints/e152_steps32.pt — the DWELL peak (e152's 32-step
    locked replay snapshot; primary dial g-12, root 0.5133); gated vs
    e152's stored s32 cells.
  New checkpoints: runs/checkpoints/e185b_site_neutral{,_sN}.pt and
  e185b_dwell_neutral{,_sN}.pt.

THE STREAM (e176N arm A VERBATIM — the only protocol this cell runs):
anchor bank = e170's NEUTRAL construction (16 plain-corpus windows, RNG
seed 170, rejection on FLORIZEL / ELIZABETH / ZEPH / MIRABEL in [s,
s+257); G_ANCHOR: 0/16 host content, 0/16 junctions covered). Batch 32 =
16 neutral-anchor draws + 16 random corpus windows, full-token CE (NO
fact windows, NO name tokens, NO mask), AdamW (0.9,0.95) wd 0.1 constant
lr 1e-3 clip 1.0, seed 10902 — the RNG draw sequence is BIT-IDENTICAL to
e161's/e177's/e176's/e176N's (same seed, same shapes/moduli:
randint(16,(16,)) + randint(len-BLOCK-1,(16,)) per step; only the anchor
CONTENT differs — e170's ANCHOR-DELTA convention). Snapshots + light
evals (primary dials + CE_R, no RNG consumed) at fine steps {1,2,4} (in
the MAIN run — e176N arm A's convention) and main {50,100,200,300}.

PRIORS (each type's own extinction-stream trajectory, embedded verbatim
and verified vs the stored files at runtime): e161's trace_summary for
the dwell type (g-12 0.5133 -> 0.0398 -> 0.0194 -> 0.0018 -> 0.0036;
first <= 0.27 at +50) and e177's trace_summary for the site type (onset
0.9981 -> 0.6717 -> 0.1946 -> 0.1409 -> 0.0115; first <= 0.27 at +100).
Neither prior has fine steps ({1,2,4} is new texture both types).

INSTRUMENT PROVENANCE: load_cpu/evl_load/battery_cell/ce_fixed_cpu/
val_windows/read_fact_at/finetune_freeze are lab/e176n_neutral_wash.py
VERBATIM (the e161/e152/e151/e143/e131/e119/e113/e068/e065/e043 lineage;
finetune_freeze's in-run light eval GAINS the site-read dial — it
consumes no RNG, so the draw sequence is unchanged); the neutral bank +
junction accounting + G_ANCHOR are lab/e170_anchor_neutral.py VERBATIM
(via e176N's copy). Copied, not imported, to own the device policy.

COMPUTE ENVELOPE: CPU-ONLY by dispatch (CUDA_VISIBLE_DEVICES=-1 forced
before torch; LOW threads <= 4), one 25 s launch stagger (single sleep,
no busy-waiting anywhere), cooldown 60 s before and after EACH of the two
trainings, per-training CPU cap 1800 s (e176N arm-A precedent: 829.7 s
actual for 300 steps with in-run evals at 4 threads; the dispatch's
"<=180s caps" is read as e176N's 1800 s per-training cap — a 300-step
CPU training cannot fit 180 s; the SMOKE (4 steps) fits 180 s naturally).

Outputs: runs/e185b/{metrics.json, type_cells.png}; checkpoints
runs/checkpoints/e185b_*.pt. No NOTES/THINKING/QUEUE/STATE edits; single
commit, no push.

Run:  cd lab && python e185b_type_cells.py    (E185B_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e174/
# e176n/e177/e185 also on the CPU — threads capped at 4 below)

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

SMOKE = os.environ.get("E185B_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e185b is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E161_METRICS = E43.REPO / "runs" / "e161" / "metrics.json"
E177_METRICS = E43.REPO / "runs" / "e177" / "metrics.json"
E152_METRICS = E43.REPO / "runs" / "e152" / "metrics.json"
E151_METRICS = E43.REPO / "runs" / "e151" / "metrics.json"
E176N_METRICS = E43.REPO / "runs" / "e176n" / "metrics.json"

SITE_CK = "e151_twodoor.pt"        # the SITE type (primary: onset @183)
DWELL_CK = "e152_steps32.pt"       # the DWELL peak (primary: g-12)

# ---- e152's placement constants (the measurement instruments rebuild these) ----
RETEACH_J = 54                    # offset: name x-cols 184..190, read rows 183..189
SITE_ADDR_ROW = 183               # the onset read row (= e120's SPLICE_ADDR_ROW)
SITE_Z_XCOL = PRE + RETEACH_J     # 184 (= e120's Z_XCOL)
SITE_CONT = BLOCK - PRE - RETEACH_J - len(NAME)     # 65-token continuation

# ---- the arms' checkpoint schedules (e176N arm A verbatim) ---------------------
CK: tuple[int, ...] = (1, 2, 4, 50, 100, 200, 300) if not SMOKE else (1, 2, 4)

# ---- fine-tune envelope (e109 arm-b / e119-L / e143 / e151 / e152 / e161/e176/
# e176n arm A VERBATIM) ---------------------------------------------------------
LR = 1e-3                          # e176N arm A verbatim
FT_TIME_CAP = 1800.0               # e176N per-training CPU cap (see docstring)
SMOKE_TIME_CAP = 180.0             # the smoke's own <=180 s cap (dispatch)
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
FREEZE_SEED = 10902               # the locked seed lineage (= e161/e177/e176/e176n)
COOLDOWN_S = 60.0                  # around EACH of the two trainings
STAGGER_S = 25.0                   # launch stagger vs the CPU fleet

# ---- e170's neutral anchor bank (e176N arm A's ONLY protocol delta) ------------
E170_ANCHOR_SEED = 170             # dedicated RNG for the plain-corpus bank
ANCHOR_FORBIDDEN = ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")

# ---- gates / references (full precision, = stored metrics) --------------------
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05

# e152's stored s32 cells (runs/e152/metrics.json trace row steps=32) — the
# DWELL net's gate refs (primary: base_gm12; co-dial: site_read_onset).
E152_S32 = {
    "gm12": 0.513336181640625,
    "g0": 0.14117614924907684,
    "gp12": 0.7145950198173523,
    "ce_r": 1.6883095502853394,
    "site_read_onset": 0.9647804498672485,
    "site_read_span": 0.9945025444030762,
}

# e151's stored after-cells (runs/e151/metrics.json 'after'; = e177's G_ROOT
# e151_stored refs) — the SITE net's gate refs (primary: site_read_onset).
E151_AFTER = {
    "gm12": 0.1020549014210701,
    "g0": 0.10710463672876358,
    "gp12": 0.12079524248838425,
    "ce_r": 1.6489683389663696,
    "site_read_onset": 0.9980737566947937,
    "site_read_span": 0.9993568658828735,
}

# e161's stored trace (the DWELL type's OWN extinction-stream wash; its stream
# WAS the extinction-grade one): runs/e161/metrics.json trace_summary VERBATIM.
E161_PRIOR = {
    "steps": [0, 50, 100, 200, 300],
    "gm12": [0.5133363008499146, 0.03981111943721771,
             0.019376089796423912, 0.0018337517976760864,
             0.00364150432869792],
    "g0": [0.14117614924907684, 0.023069579154253006,
           0.021433841437101364, 0.0064969733357429504,
           0.005843465682119131],
    "site_read_onset": [0.9647805094718933, 0.1495015025138855,
                        0.051231108605861664, 0.010209816507995129,
                        0.012098240666091442],
    "ce_r": [1.6883095502853394, 1.6653131246566772,
             1.6886937618255615, 1.6153916120529175,
             1.6463027000427246],
}

# e177's stored trace (the SITE type's OWN extinction-stream wash): runs/e177/
# metrics.json trace_summary VERBATIM.
E177_PRIOR = {
    "steps": [0, 50, 100, 200, 300],
    "site_read_onset": [0.9980737566947937, 0.6716808676719666,
                        0.1945670247077942, 0.1409446746110916,
                        0.011479347012937069],
    "site_read_span": [0.9993568658828735, 0.9501108527183533,
                       0.8833314180374146, 0.800300657749176,
                       0.8000056743621826],
    "gm12": [0.10205486416816711, 0.03960553556680679,
             0.0019761284347623587, 0.002991028130054474,
             0.0014502160483971238],
    "g0": [0.10710463672876358, 0.07036908715963364,
           0.0036823058035224676, 0.0009324398706667125,
           0.0008762489887885749],
    "ce_r": [1.6489683389663696, 1.6404904127120972,
             1.6416882276535034, 1.6352111101150513,
             1.6433320045471191],
}

# ---- registered bar constants (frozen; see OPERATIONALIZATIONS above) --------
SHUT_BAR = 0.27                   # both types' own prior SHUT bars (identical)
SURVIVE_BAR = 0.50                # the dispatch's retention bars (identical)
ROOT_SITE_ONSET = E151_AFTER["site_read_onset"]
ROOT_DWELL_GM12 = E152_S32["gm12"]

REGISTERED_PREDICTION = {
    "both_dissolve": "BOTH-DISSOLVE fires if: the site type's onset and the "
        "dwell type's g-12 both fall under their bars by +50 — the sparse "
        "union becomes an honest cross; 'all types' earns its name.",
    "either_survives": "EITHER-SURVIVES fires if: either type retains "
        "(site >= 0.5 onset / dwell >= 0.5 g-12) through +300 — 'all types' "
        "is FALSE in the direction that matters; a survivor exists.",
    "texture": "No bar shopping; texture (different rates than their "
        "extinction runs) => TEXTURE with both curves.",
    "operationalizations": "'their bars' = each type's own prior SHUT bar, "
        "both 0.27 (e161 DISUSE / e177 SCRATCH-MEMORY / e176N NEUTRAL-"
        "DISSOLVES); dissolve = primary dial <= 0.27 AT +50 (earliest "
        "checkpoint <= 0.27 over {1,2,4,50,100,200,300} CO-REPORTED); "
        "'retains through +300' = primary dial >= 0.50 at EVERY continuation "
        "checkpoint {1,2,4,50,100,200,300}; SITE primary = read_fact_at "
        "onset on e152's locked j=54 pool @183/184; DWELL primary = "
        "install-60 battery mean p(Z) at ctx offset -12; clauses disjoint; "
        "order BOTH-DISSOLVE -> EITHER-SURVIVES -> TEXTURE; every sub-boolean "
        "reported regardless; CE_R + in-batch corpus CE per checkpoint on "
        "both nets.",
    "registration": "QUEUE row e185b (DISPATCHED ~19:40Z CPU) + the dispatch "
        "mission text; frozen verbatim in this docstring before compute. "
        "Adjudicate against exactly this; no bar shopping.",
}

trims: list[str] = []
deviations: list[str] = [
    "CPU-ONLY by dispatch (CUDA_VISIBLE_DEVICES=-1 forced before torch): "
    "torch threads 4, one 25 s launch stagger (single sleep, no "
    "busy-waiting), cooldown 60 s before/after EACH of the two trainings, "
    "per-training CPU cap 1800 s (e176N arm-A precedent 829.7 s actual; the "
    "dispatch's '<=180s caps' read as e176N's 1800 s per-training cap — a "
    "300-step CPU training cannot fit 180 s; the SMOKE keeps its own 180 s "
    "cap).",
    "finetune_freeze's in-run light eval GAINS the site-read dial "
    "(read_fact_at on the j=54 pool @183/184) vs e176N's copy — it consumes "
    "no RNG, so the draw sequence at seed 10902 is unchanged (verified by "
    "the identical-shapes/moduli assertion of the copy); the per-step "
    "arithmetic is e176N arm A VERBATIM.",
    "The two priors have NO fine-step trajectories (e161/e177 checkpointed "
    "only {50,100,200,300}); {1,2,4} is new texture for both types and is "
    "reported as such, not adjudicated.",
    "Eval thread count is 4 (dispatch) vs the stored cells' original "
    "threads — CPU reduction order can drift low-order bits; the two G_ROOT "
    "gates report both the 5e-6 bit flag and the 0.05 fallback tolerance "
    "(e161/e176/e176n/e177 precedent).",
    "Single seed (10902), one trajectory per type, one lineage per net "
    "(both nets descend from e131_consolidated_e113) — n=1 per cell, point "
    "estimates until replicated; the dwell peak itself is a single "
    "trajectory (e152R: DWELL-SEED-DEPENDENT) and the site endpoint is a "
    "lineage-1 object (T113: phase structure lineage-specific).",
    "Smoke mode trims: 4 steps per net, ckpts {1,2,4}, no cooldowns, 180 s "
    "cap, checkpoints/dirs prefixed smoke_; nothing adjudicated.",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/e176n_neutral_wash.py VERBATIM (see the module docstring) +
# e170's anchor gate (via e176N's copy). Copied rather than imported to own
# the device policy.

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


@torch.no_grad()
def read_fact_at(net: TinyGPT, pool_x: torch.Tensor, name_ids, zid: int,
                 addr_row: int, xcol: int, bs=30) -> dict:
    """e131's read_fact_position VERBATIM ARITHMETIC (e151/e152/e161/e176/
    e176n/e177 copy): p(true name char) at positions addr_row..addr_row+6
    over the pool."""
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


# ------------------------------------------------------------------ fine-tune

def finetune_freeze(tag: str, net0: TinyGPT, anchor: torch.Tensor,
                    train_ids: torch.Tensor, itos, r_eval_xy, gm12_ids,
                    g0_ids, pool_x, name_ids, zid: int, seed: int,
                    lr: float, ckpt_steps: tuple[int, ...]):
    """THE NEUTRAL PLAIN-CORPUS FREEZE (e176N arm A's finetune_freeze
    VERBATIM arithmetic, the light eval gaining the site-read dial). Per
    step: aj = randint(16) anchor draws, rj = randint(16) random corpus
    offsets; batch 32 full-token CE; AdamW (0.9,0.95) wd 0.1 constant lr,
    clip 1.0. The RNG draw sequence is IDENTICAL to e161's/e177's/e176N's
    at seed 10902 (same shapes/moduli) — only the anchor bank's CONTENT
    differs (e170's neutral construction). Snapshots (deep-copy out) +
    light CPU evals (g-12, g0, site onset/span, CE_R — no RNG consumed)
    at the checkpoint steps; the in-batch corpus CE recorded at every
    checkpoint and every 50."""
    ckpt_set = set(ckpt_steps)
    n_steps = ckpt_steps[-1]
    cap = SMOKE_TIME_CAP if SMOKE else FT_TIME_CAP
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
            sr = read_fact_at(evl, pool_x, name_ids, zid,
                              SITE_ADDR_ROW, SITE_Z_XCOL)
            ce_r = ce_fixed_cpu(evl, *r_eval_xy)
            traj.append({"step": step, "g_m12_mean_pz": gz["mean_pz"],
                         "g0_mean_pz": gz0["mean_pz"],
                         "site_onset": sr["pz_onset_mean"],
                         "site_span": sr["pname_mean_over7"],
                         "corpus_ce": float(loss.item()), "ce_r": ce_r,
                         "elapsed_s": round(time.time() - t_start, 1)})
            log(f"  [{tag}] CKPT +{step:4d} g-12 {gz['mean_pz']:.4f} "
                f"onset {sr['pz_onset_mean']:.4f} CE_R {ce_r:.4f} "
                f"(in-batch CE {float(loss.item()):.4f})")
        if (time.time() - t_start) > cap:
            log(f"  [{tag}] time cap {cap:.0f}s at s{step}")
            trims.append(f"{tag}: time cap at step {step}")
            break
    net.eval()
    return {"sds": sds, "traj": traj, "steps_ran": step, "seed": seed,
            "lr": lr, "zeph_violations": zeph_checks}


CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "e185b", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name} "
        f"({', '.join(f'{k}={v}' for k, v in meta.items())})")


def verify_ref(embedded: dict, path: Path, keymap: dict, src_name: str) -> dict:
    """Verify an embedded reference copy against its stored metrics file when
    present (no silent divergence; e176N's verify_ref convention). `keymap`
    maps each embedded key to its trace_summary key in the stored file."""
    src = {"source": f"embedded verbatim copy ({src_name})",
           "file_present": path.exists(), "verified_vs_embedded": None,
           "max_abs_diff": None}
    if path.exists():
        mm = json.loads(path.read_text(encoding="utf-8"))
        ts = mm["trace_summary"]
        diffs = [abs(a - b) for k, file_key in keymap.items()
                 for a, b in zip(ts[file_key], embedded[k])]
        steps_key = next(iter(keymap))          # the steps key (embedded name)
        steps_ok = list(ts[keymap[steps_key]]) == embedded[steps_key]
        src["max_abs_diff"] = max(diffs) if diffs else None
        src["verified_vs_embedded"] = bool(steps_ok and max(diffs) < 1e-9)
        if src["verified_vs_embedded"]:
            src["source"] = f"{src_name} (embedded copy verified, max|diff| " \
                            f"{max(diffs):.1e})"
    return src


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


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e185b_smoke" if SMOKE else "e185b")
    log(f"E185B THE NEUTRAL-STREAM TYPE CELLS (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()}), stagger "
        f"{STAGGER_S:.0f}s, cooldown {COOLDOWN_S:.0f}s around each training")
    time.sleep(STAGGER_S)            # launch stagger vs the CPU fleet

    # ---------------- protocol rebuild (e143/e151/e152/e161/e176/e176n verbatim)
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
    # THE NEUTRAL ANCHOR BANK (e170's construction VERBATIM via e176N arm A
    # — the ONLY protocol delta vs each type's own extinction run).
    # =====================================================================
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

    jc_neutral = junctions_covered(n_starts)
    host_occ_total = len(host_positions)
    bg_rate = host_occ_total * (BLOCK + 1) / len(train_ids)
    anchor_neutral_zeph = sum(1 for w in anchor_neutral
                              if "ZEPH" in corpus.decode(w))
    G_ANCHOR = {
        "neutral_bank": {
            "construction": ("16 plain corpus windows from train_ids, RNG seed "
                             f"{E170_ANCHOR_SEED}, rejection if [s, s+257) "
                             "contains FLORIZEL/ELIZABETH/ZEPH/MIRABEL — "
                             "e170's construction VERBATIM (= e176N arm A's "
                             "bank)"),
            "n_windows": 16, "block": BLOCK, "seed": E170_ANCHOR_SEED,
            "starts": n_starts, "tries": tries, "rejections": rejections,
            "forbidden": list(ANCHOR_FORBIDDEN),
            "windows_with_host_content": sum(
                1 for s in n_starts
                if any(f in train_text[s: s + BLOCK + 1] for f in HOSTS)),
            "junctions_covered": jc_neutral,
            "anchor_zeph_windows": anchor_neutral_zeph,
        },
        "prior_stream_note": ("both types' prior washes (e161 dwell / e177 "
                              "site) ran the ORIGINAL bank — the install's "
                              "own name-deleted host windows, 16/16 junctions "
                              "covered (targeted EXTINCTION; the R50 critic's "
                              "finding) — at the SAME seed/lr/optimizer; only "
                              "the anchor content differs here"),
        "budget_identical_to_priors": True,
        "rng_stream_identical_to_e161_e177_e176n": True,
        "rng_note": ("finetune_freeze verbatim; draw shapes/moduli identical "
                     "(n_anc=16, len(train_ids)); seed 10902 — the same "
                     "aj/rj sequences as e161's/e177's/e176N's runs; only "
                     "anchor[aj] content differs"),
        "random_channel_background": {
            "host_occurrences_in_train": host_occ_total,
            "est_window_hit_rate": bg_rate,
            "note": ("the 16-per-batch random corpus windows are e161/e176/"
                     "e176n VERBATIM and unfiltered — identical background "
                     "(~1-2%/window), not part of the delta"),
        },
    }
    G_ANCHOR["pass"] = bool(
        G_ANCHOR["neutral_bank"]["windows_with_host_content"] == 0
        and G_ANCHOR["neutral_bank"]["junctions_covered"] == 0
        and anchor_neutral_zeph == 0
        and anchor_neutral.shape == (16, BLOCK))
    assert G_ANCHOR["pass"], f"anchor gate FAILED: {G_ANCHOR}"
    log(f"G_ANCHOR: neutral bank 16x{BLOCK} plain-corpus windows (seed "
        f"{E170_ANCHOR_SEED}, {rejections} rejections/{tries} tries) — host "
        f"content 0/16, junctions 0/16; random-channel background "
        f"~{100 * bg_rate:.1f}%/window: PASS")

    # ---------------- batteries: ctx = train_text[p-PRE-j : p] (e119 verbatim)
    bat_ids = {}
    for j in (-12, 0, 12):
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    gm12_ids = bat_ids[-12]
    g0_ids = bat_ids[0]

    # ---------------- the two type nets + gates vs their stored cells
    def measure_root(sd: dict, tag: str) -> dict:
        net = evl_load(sd)
        out = {"tag": tag,
               "base": {j: battery_cell(net, bat_ids[j], zid)
                        for j in (-12, 0, 12)},
               "ce_r": ce_fixed_cpu(net, *r_eval_xy),
               "site_read": read_fact_at(net, pool_x, name_ids, zid,
                                         SITE_ADDR_ROW, SITE_Z_XCOL)}
        log(f"[{tag}] base: " + " ".join(
                f"g{j:+d} {out['base'][j]['mean_pz']:.4f}" for j in (-12, 0, 12))
            + f" | onset {out['site_read']['pz_onset_mean']:.4f} "
            f"span {out['site_read']['pname_mean_over7']:.4f} "
            f"| CE_R {out['ce_r']:.4f}")
        del net
        return out

    def flat_cells(m: dict) -> dict:
        return {"gm12": m["base"][-12]["mean_pz"],
                "g0": m["base"][0]["mean_pz"],
                "gp12": m["base"][12]["mean_pz"],
                "ce_r": m["ce_r"],
                "site_read_onset": m["site_read"]["pz_onset_mean"],
                "site_read_span": m["site_read"]["pname_mean_over7"]}

    net_site = load_cpu(CKPT_DIR / SITE_CK)
    sd_site = {k: v.clone() for k, v in net_site.state_dict().items()}
    st_site = torch.load(CKPT_DIR / SITE_CK, map_location="cpu",
                         weights_only=False)
    site_meta = E43.jsonable(st_site.get("meta", {})) \
        if isinstance(st_site, dict) else None
    log(f"site net: {SITE_CK} (meta: {site_meta})")
    root_site = measure_root(sd_site, "root_site")
    G_ROOT_SITE = gate_vs(flat_cells(root_site), E151_AFTER,
                          "G_ROOT_SITE (vs e151 stored after-cells)")
    if not G_ROOT_SITE["pass"]:
        raise RuntimeError("site-net gate FAILED vs e151 stored after-cells")

    net_dwell = load_cpu(CKPT_DIR / DWELL_CK)
    sd_dwell = {k: v.clone() for k, v in net_dwell.state_dict().items()}
    st_dwell = torch.load(CKPT_DIR / DWELL_CK, map_location="cpu",
                          weights_only=False)
    dwell_meta = E43.jsonable(st_dwell.get("meta", {})) \
        if isinstance(st_dwell, dict) else None
    log(f"dwell net: {DWELL_CK} (meta: {dwell_meta})")
    root_dwell = measure_root(sd_dwell, "root_dwell")
    G_ROOT_DWELL = gate_vs(flat_cells(root_dwell), E152_S32,
                           "G_ROOT_DWELL (vs e152 stored s32 cells)")
    if not G_ROOT_DWELL["pass"]:
        raise RuntimeError("dwell-net gate FAILED vs e152 stored s32 cells")
    log("gates: G_SPLICE, G_NAMEFREE, G_POOL, G_ANCHOR, G_ROOT_SITE, "
        "G_ROOT_DWELL all PASS")

    # reference provenance (embedded copies verified vs the stored files)
    src_e161 = verify_ref(E161_PRIOR, E161_METRICS,
                          {"steps": "freeze_steps", "gm12": "base_gm12",
                           "g0": "base_g0",
                           "site_read_onset": "site_read_onset",
                           "ce_r": "ce_r"},
                          "runs/e161/metrics.json trace_summary")
    src_e177 = verify_ref(E177_PRIOR, E177_METRICS,
                          {"steps": "freeze_steps",
                           "site_read_onset": "site_read_onset",
                           "site_read_span": "site_read_span",
                           "gm12": "base_gm12", "g0": "base_g0",
                           "ce_r": "ce_r"},
                          "runs/e177/metrics.json trace_summary")
    src_e152 = {"file_present": E152_METRICS.exists(),
                "verified_vs_embedded": None}
    if E152_METRICS.exists():
        mm = json.loads(E152_METRICS.read_text(encoding="utf-8"))
        row = next(r for r in mm["trace"] if r.get("steps") == 32)
        file_keys = {"gm12": "base_gm12", "g0": "base_g0",
                     "gp12": "base_gp12", "ce_r": "ce_r",
                     "site_read_onset": "site_read_onset",
                     "site_read_span": "site_read_span"}
        src_e152["verified_vs_embedded"] = bool(
            all(abs(row[fk] - E152_S32[k]) < 1e-9 for k, fk in
                file_keys.items()))
        src_e152["source"] = ("runs/e152/metrics.json trace row s32 "
                              "(embedded copy verified)"
                              if src_e152["verified_vs_embedded"]
                              else "EMBEDDED COPY DIVERGED — using file")
        if src_e152["verified_vs_embedded"] is False:
            E152_S32.update({k: row[fk] for k, fk in file_keys.items()})
    src_e151 = {"file_present": E151_METRICS.exists(),
                "verified_vs_embedded": None}
    if E151_METRICS.exists():
        mm = json.loads(E151_METRICS.read_text(encoding="utf-8"))
        af = mm["after"]
        e151_ref = {"gm12": af["base"]["-12"]["mean_pz"],
                    "g0": af["base"]["0"]["mean_pz"],
                    "gp12": af["base"]["12"]["mean_pz"],
                    "ce_r": af["ce_r"],
                    "site_read_onset": af["site_read"]["pz_onset_mean"],
                    "site_read_span": af["site_read"]["pname_mean_over7"]}
        src_e151["verified_vs_embedded"] = bool(
            all(abs(e151_ref[k] - E151_AFTER[k]) < 1e-9 for k in E151_AFTER))
        src_e151["source"] = ("runs/e151/metrics.json after cells "
                              "(embedded copy verified)"
                              if src_e151["verified_vs_embedded"]
                              else "EMBEDDED COPY DIVERGED — using file")
        if src_e151["verified_vs_embedded"] is False:
            E151_AFTER.update(e151_ref)

    # =====================================================================
    # THE TWO TRAININGS (e176N arm A verbatim, one per type; cooldowns)
    # =====================================================================
    arms: dict = {}

    for tag, net0, ckname in (("site", net_site, "e185b_site_neutral"),
                              ("dwell", net_dwell, "e185b_dwell_neutral")):
        log("=" * 78)
        if not SMOKE:
            log(f"[thermal] cooldown {COOLDOWN_S:.0f}s before arm {tag}")
            cooldown(COOLDOWN_S)
        log(f"ARM {tag.upper()} — THE NEUTRAL WASH on the "
            f"{'SITE' if tag == 'site' else 'DWELL'} type: {CK[-1]}-step "
            f"plain-corpus freeze with e170's NEUTRAL anchors (batch "
            f"{ANCH_BS} neutral + {RAND_BS} random, full-token CE, lr {LR}, "
            f"seed {FREEZE_SEED} = e161/e177/e176N's draw sequence), "
            f"checkpoints +{list(CK)}")
        arm = finetune_freeze(tag, net0, anchor_neutral, train_ids, itos,
                              r_eval_xy, gm12_ids, g0_ids, pool_x, name_ids,
                              zid, FREEZE_SEED, LR, CK)
        if not SMOKE:
            log(f"[thermal] cooldown {COOLDOWN_S:.0f}s after arm {tag}")
            cooldown(COOLDOWN_S)
        g_free = {"zeph_violations": arm["zeph_violations"],
                  "pass": bool(arm["zeph_violations"] == 0)}
        assert g_free["pass"], f"arm {tag}: name token leaked into a window"
        arms[tag] = {"arm": arm, "G_DRAWFREE": g_free, "ckname": ckname,
                     "root": root_site if tag == "site" else root_dwell}

        smax = max(arm["sds"])
        save_ckpt(ckname, arm["sds"][smax],
                  {"desc": f"{'e151_twodoor (SITE type)' if tag == 'site' else 'e152_steps32 (DWELL peak)'} "
                           f"+ {smax}-step NEUTRAL-anchor plain-corpus freeze "
                           f"(e170's neutral bank seed {E170_ANCHOR_SEED}: 16 "
                           f"plain-corpus windows, 0/16 junctions; batch 32 = "
                           f"16 neutral anchors + 16 random, full-token CE), "
                           f"lr {LR}, seed {FREEZE_SEED} (= e176N arm A "
                           f"protocol verbatim)",
                   "steps": int(smax), "seed": FREEZE_SEED, "lr": LR,
                   "anchor_bank": f"neutral (seed {E170_ANCHOR_SEED})",
                   "primary_dial": "site_read_onset" if tag == "site"
                                   else "base_gm12",
                   "base": f"runs/checkpoints/"
                           f"{SITE_CK if tag == 'site' else DWELL_CK}"})
        for s in sorted(arm["sds"]):
            if s == smax:
                continue
            save_ckpt(f"{ckname}_s{s}", arm["sds"][s],
                      {"desc": f"{tag} type + {s}-step NEUTRAL-anchor freeze "
                               f"(intermediate), seed {FREEZE_SEED}",
                       "steps": int(s), "seed": FREEZE_SEED, "lr": LR,
                       "anchor_bank": f"neutral (seed {E170_ANCHOR_SEED})",
                       "base": f"runs/checkpoints/"
                               f"{SITE_CK if tag == 'site' else DWELL_CK}"})

    # =====================================================================
    # TRAJECTORIES + ADJUDICATION (registered clauses; no shopping)
    # =====================================================================
    def trace_from(tag: str) -> list:
        root_c = flat_cells(arms[tag]["root"])
        rows = [{"freeze_steps": 0, **root_c,
                 "retention_vs_root_onset":
                     root_c["site_read_onset"] / ROOT_SITE_ONSET,
                 "retention_vs_root_gm12":
                     root_c["gm12"] / ROOT_DWELL_GM12}]
        for t in arms[tag]["arm"]["traj"]:
            rows.append({"freeze_steps": t["step"],
                         "gm12": t["g_m12_mean_pz"], "g0": t["g0_mean_pz"],
                         "site_read_onset": t["site_onset"],
                         "site_read_span": t["site_span"],
                         "ce_r": t["ce_r"], "corpus_ce": t["corpus_ce"],
                         "retention_vs_root_onset":
                             t["site_onset"] / ROOT_SITE_ONSET,
                         "retention_vs_root_gm12":
                             t["g_m12_mean_pz"] / ROOT_DWELL_GM12})
        return rows

    trace_site = trace_from("site")
    trace_dwell = trace_from("dwell")

    def dial_seq(trace, key):
        return [r[key] for r in trace]

    def step_of(trace, key, pred):
        return next((r["freeze_steps"] for r in trace
                     if r["freeze_steps"] > 0 and pred(r[key])), None)

    site_on = dial_seq(trace_site, "site_read_onset")
    dwell_g = dial_seq(trace_dwell, "gm12")
    ck_steps = [r["freeze_steps"] for r in trace_site if r["freeze_steps"] > 0]

    site_50 = next((r["site_read_onset"] for r in trace_site
                    if r["freeze_steps"] == 50), trace_site[-1]["site_read_onset"])
    dwell_50 = next((r["gm12"] for r in trace_dwell
                     if r["freeze_steps"] == 50), trace_dwell[-1]["gm12"])

    site_dissolves = bool(site_50 <= SHUT_BAR)
    dwell_dissolves = bool(dwell_50 <= SHUT_BAR)
    site_survives = bool(all(v >= SURVIVE_BAR for v in site_on[1:]))
    dwell_survives = bool(all(v >= SURVIVE_BAR for v in dwell_g[1:]))

    both_dissolve = bool(site_dissolves and dwell_dissolves)
    either_survives = bool(site_survives or dwell_survives)
    if both_dissolve:
        verdict = "BOTH-DISSOLVE"
    elif either_survives:
        verdict = "EITHER-SURVIVES"
    else:
        verdict = "TEXTURE"

    # per-type fate vs its own extinction prior (co-report; not a bar)
    def fate_vs_prior(trace, key, prior, pname):
        first_le = step_of(trace, key, lambda v: v <= SHUT_BAR)
        first_lt50 = step_of(trace, key, lambda v: v < SURVIVE_BAR)
        prior_first = next((s for s, v in zip(prior["steps"], prior[key])
                            if s > 0 and v <= SHUT_BAR), None)
        return {"neutral_first_le_0.27": first_le,
                "neutral_first_lt_0.5": first_lt50,
                f"{pname}_first_le_0.27": prior_first,
                "neutral_50": next((r[key] for r in trace
                                    if r["freeze_steps"] == 50), None),
                "neutral_300": trace[-1][key],
                "prior_300": prior[key][-1]}

    fate_site = fate_vs_prior(trace_site, "site_read_onset", E177_PRIOR,
                              "e177_extinction")
    fate_dwell = fate_vs_prior(trace_dwell, "gm12", E161_PRIOR,
                               "e161_extinction")

    cond = {
        "BOTH_DISSOLVE": {
            "bar": SHUT_BAR, "site_onset_at_50": site_50,
            "dwell_gm12_at_50": dwell_50,
            "site_clause": site_dissolves, "dwell_clause": dwell_dissolves,
            "fires": both_dissolve},
        "EITHER_SURVIVES": {
            "bar": SURVIVE_BAR,
            "site_onset_seq": site_on, "dwell_gm12_seq": dwell_g,
            "site_all_ge_bar": site_survives, "dwell_all_ge_bar": dwell_survives,
            "ck_steps": ck_steps,
            "fires": either_survives},
        "TEXTURE": {"fires": verdict == "TEXTURE"},
        "site_earliest_le_0.27": step_of(trace_site, "site_read_onset",
                                         lambda v: v <= SHUT_BAR),
        "dwell_earliest_le_0.27": step_of(trace_dwell, "gm12",
                                          lambda v: v <= SHUT_BAR),
        "site_earliest_lt_0.5": step_of(trace_site, "site_read_onset",
                                        lambda v: v < SURVIVE_BAR),
        "dwell_earliest_lt_0.5": step_of(trace_dwell, "gm12",
                                         lambda v: v < SURVIVE_BAR),
    }
    if not SMOKE:
        log(f"ADJUDICATION: site onset@+50 {site_50:.4f} (bar {SHUT_BAR}) "
            f"| dwell g-12@+50 {dwell_50:.4f} (bar {SHUT_BAR}) -> "
            f"{verdict}")

    # =====================================================================
    # THE FOUR-WAY OVERLAY (both types x both streams)
    # =====================================================================
    fig, axs = plt.subplots(3, 1, figsize=(9, 12))
    ax1, ax2, ax3 = axs
    for ax, title in ((ax1, "SITE type — read onset @183 (e151_twodoor)"),
                      (ax2, "DWELL type — install-60 g-12 (e152_steps32)")):
        ax.axhline(SURVIVE_BAR, color="gray", lw=0.8, ls=":")
        ax.axhline(SHUT_BAR, color="gray", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("freeze steps (neutral stream, log scale)")
        ax.set_ylabel("mean p(Z)")
        ax.set_title(title)
        ax.set_ylim(-0.03, 1.05)
    ns = [r["freeze_steps"] for r in trace_site]
    nd = [r["freeze_steps"] for r in trace_dwell]
    ax1.plot(ns, site_on, "o-", color="tab:blue",
             label="e185b NEUTRAL stream (this run)")
    ax1.plot(E177_PRIOR["steps"], E177_PRIOR["site_read_onset"], "s--",
             color="tab:red",
             label="e177 EXTINCTION stream (prior, own stream)")
    ax1.set_title(ax1.get_title()
                  + f"\nroot {ROOT_SITE_ONSET:.4f} -> +50 {site_50:.4f} "
                    f"| +300 {site_on[-1]:.4f}")
    ax1.legend(loc="best", fontsize=8)
    ax2.plot(nd, dwell_g, "o-", color="tab:blue",
             label="e185b NEUTRAL stream (this run)")
    ax2.plot(E161_PRIOR["steps"], E161_PRIOR["gm12"], "s--",
             color="tab:red",
             label="e161 EXTINCTION stream (prior, own stream)")
    ax2.set_title(ax2.get_title()
                  + f"\nroot {ROOT_DWELL_GM12:.4f} -> +50 {dwell_50:.4f} "
                    f"| +300 {dwell_g[-1]:.4f}")
    ax2.legend(loc="best", fontsize=8)
    ax3.plot(ns, [r["ce_r"] for r in trace_site], "o-", color="tab:blue",
             label="site type, neutral (e185b)")
    ax3.plot(E177_PRIOR["steps"], E177_PRIOR["ce_r"], "s--", color="tab:red",
             label="site type, extinction (e177)")
    ax3.plot(nd, [r["ce_r"] for r in trace_dwell], "o-", color="tab:cyan",
             label="dwell type, neutral (e185b)")
    ax3.plot(E161_PRIOR["steps"], E161_PRIOR["ce_r"], "s--", color="tab:orange",
             label="dwell type, extinction (e161)")
    ax3.set_xscale("log")
    ax3.set_xlabel("freeze steps (log scale)")
    ax3.set_ylabel("CE_R (val windows)")
    ax3.set_title("the wash's price channel (CE_R)")
    ax3.legend(loc="best", fontsize=8)
    fig.suptitle(f"E185B — the neutral-stream type cells: {verdict} "
                 f"(dissolve bar {SHUT_BAR} by +50; survive bar "
                 f"{SURVIVE_BAR} through +300)\nsite onset@+50 "
                 f"{site_50:.4f} / dwell g-12@+50 {dwell_50:.4f}; "
                 f"e176N arm-A protocol verbatim on both types",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = rd / "type_cells.png"
    fig.savefig(png, dpi=140)
    log(f"[plot] saved {png}")

    metrics = {
        "experiment": "e185b_type_cells",
        "date": common.now_iso(),
        "registration": REGISTERED_PREDICTION["registration"],
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("does each memory type dissolve under the NEUTRAL stream "
                     "it never touched (e176N arm A verbatim), or does either "
                     "type survive it — closing R52's two empty grid cells "
                     "under the lead finding's 'all types' claim?"),
        "nets": {
            "site": {"path": f"runs/checkpoints/{SITE_CK}",
                     "meta": site_meta,
                     "primary_dial": "site_read_onset @183/184 (e177 battery)",
                     "prior_wash": "e177 (its OWN extinction stream, seed "
                                   "10902, lr 1e-3)"},
            "dwell": {"path": f"runs/checkpoints/{DWELL_CK}",
                      "meta": dwell_meta,
                      "primary_dial": "base_gm12 install-60 battery (e161 "
                                      "convention)",
                      "prior_wash": "e161 (its OWN extinction stream, seed "
                                    "10902, lr 1e-3)"},
        },
        "stream": {
            "desc": G_ANCHOR["neutral_bank"]["construction"],
            "protocol": ("e176N arm A VERBATIM: batch 32 = 16 neutral-anchor "
                         "draws + 16 random corpus windows, full-token CE (NO "
                         "fact windows, NO name tokens, NO mask), AdamW "
                         "(0.9,0.95) wd 0.1 constant lr 1e-3 clip 1.0, seed "
                         "10902 — the RNG draw sequence identical to "
                         "e161/e177/e176/e176N's; only the anchor bank's "
                         "CONTENT differs (0/16 junctions vs 16/16)"),
            "ckpt_steps": list(CK),
            "device": "cpu", "torch_threads": torch.get_num_threads(),
            "time_cap_s": FT_TIME_CAP, "cooldown_s": COOLDOWN_S,
            "stagger_s": STAGGER_S,
        },
        "gates": {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                  "G_POOL": G_POOL, "G_ANCHOR": G_ANCHOR,
                  "G_ROOT_SITE": G_ROOT_SITE, "G_ROOT_DWELL": G_ROOT_DWELL,
                  "G_DRAWFREE_SITE": arms["site"]["G_DRAWFREE"],
                  "G_DRAWFREE_DWELL": arms["dwell"]["G_DRAWFREE"]},
        "priors": {
            "e161_dwell_extinction": {**E161_PRIOR, "provenance": src_e161},
            "e177_site_extinction": {**E177_PRIOR, "provenance": src_e177},
            "e152_s32_gate_refs": {"refs": E152_S32, "provenance": src_e152},
            "e151_after_gate_refs": {"refs": E151_AFTER, "provenance": src_e151},
        },
        "trace_site": trace_site,
        "trace_dwell": trace_dwell,
        "traj_site_light": arms["site"]["arm"]["traj"],
        "traj_dwell_light": arms["dwell"]["arm"]["traj"],
        "trace_summary": {
            "site_steps": [r["freeze_steps"] for r in trace_site],
            "site_onset": site_on,
            "site_span": [r["site_read_span"] for r in trace_site],
            "site_gm12": dial_seq(trace_site, "gm12"),
            "site_ce_r": [r["ce_r"] for r in trace_site],
            "dwell_steps": [r["freeze_steps"] for r in trace_dwell],
            "dwell_gm12": dwell_g,
            "dwell_onset": [r["site_read_onset"] for r in trace_dwell],
            "dwell_ce_r": [r["ce_r"] for r in trace_dwell],
            "e177_prior_steps": E177_PRIOR["steps"],
            "e177_prior_onset": E177_PRIOR["site_read_onset"],
            "e161_prior_steps": E161_PRIOR["steps"],
            "e161_prior_gm12": E161_PRIOR["gm12"],
        },
        "fate_vs_prior": {"site": fate_site, "dwell": fate_dwell},
        "adjudication": {
            "verdict": None if SMOKE else verdict,
            "clauses": cond,
            "sub_booleans": {
                "site_dissolves_by_50": site_dissolves,
                "dwell_dissolves_by_50": dwell_dissolves,
                "site_survives_through_300": site_survives,
                "dwell_survives_through_300": dwell_survives},
            "clause": (None if SMOKE else
                       {"BOTH-DISSOLVE": "the sparse union becomes an honest "
                                         "cross; 'all types' earns its name",
                        "EITHER-SURVIVES": "'all types' is FALSE in the "
                                           "direction that matters; a "
                                           "survivor exists",
                        "TEXTURE": "different rates than their extinction "
                                   "runs; both curves reported"}[verdict]),
            "smoke_note": "SMOKE run — nothing adjudicated" if SMOKE else None,
        },
        "ckpt_inventory": CKPT_INVENTORY,
        "trims": trims,
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1),
                   "stagger_s": STAGGER_S,
                   "trainings": {t: {"steps_ran": arms[t]["arm"]["steps_ran"],
                                     "traj_elapsed_final_s":
                                         arms[t]["arm"]["traj"][-1]["elapsed_s"]
                                     if arms[t]["arm"]["traj"] else None}
                                 for t in ("site", "dwell")}},
    }
    save_json(rd / "metrics.json", metrics)
    log(f"saved {rd / 'metrics.json'} (verdict "
        f"{'SMOKE' if SMOKE else verdict}) — total "
        f"{time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
