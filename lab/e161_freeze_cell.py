"""E161 — THE FREEZE-CELL (re-registered post-e166 per the R48 honesty note;
now THE mechanism decider for the session's central open fork).

WHY (T099 + the MAP UPDATE): e151's locked re-teach built a graft and shut
the geometry door (g-12 0.916 -> 0.102); e166 proved the graft rows carry
ZERO of the closure (deleting them moves the door +0.0000). So the closure
lives in the stream state (REWRITE — the conversion rewrote the readout) or
in disuse-decay of the +/-12 pathway (DISUSE — the door decayed for lack of
use while training pressed elsewhere). THE CELL: take the DWELL PEAK
(e152_steps32 — both doors genuine: site store clears the census bar AND
geometry retention ~0.56) and continue it on PLAIN CORPUS with NO fact
replay, NO name tokens. Does the door close with no fact teaching at all?

ROOT (mandated): runs/checkpoints/e152_steps32.pt — the e152 dwell-peak
snapshot (sequential-continuation step 32 of the single 300-step locked
re-teach, seed 10902). Gated below vs e152's stored s32 cells before any
compute is trusted (G_S32 refs = runs/e152/metrics.json trace row steps=32).

PLAIN-CORPUS FREEZE (the design): ONE ~300-step fine-tune of the s32 net on
the lab's standard corpus stream — the same corpus the erase-road used
(e119 road E / e083 relearn corpus half): batch 32 = 16 anchor-bank draws
(e152's anchor bank: 256-token windows at the install occurrences' left
context, name-free by construction) + 16 random corpus windows from
train_ids (name-free — the raw corpus contains ZERO 'ZEPH' substrings,
grep-verified; every drawn window re-verified at draw time). NO fact
windows, NO name mask, NO name tokens. Full-token CE (every position of
every window — e152's anchor-half treatment generalized to the whole batch).
Optimizer VERBATIM e109 arm-b / e119-L / e143 / e151 / e152: AdamW (0.9,
0.95) wd 0.1, constant lr 1e-3, clip 1.0; seed 10902 (the locked fine-tune
seed lineage); snapshots at freeze steps {50, 100, 200, 300} (continuation
steps from the dwell peak).

MEASURE PER CHECKPOINT (+ step 0 = the s32 root itself, the 'before'):
g-12 AND g+12 (the geometry doors, install-60 batteries, ctx =
train_text[p-PRE-j:p], e119 construction), g0; the 183-site census
(span-primary, e139/e143/e151/e152 lineage — does the site-store decay
without supervision?) + the functional site read (onset/span means over the
e152 locked j=54 pool); A(129) (the brake, old-band census); D-all g0; CE_R
(wreckage guard); d183 co-report (the new door's row-necessity); row-0
strength (the sink dial).

REGISTERED PREDICTION (verbatim from QUEUE e161 / the dispatch; adjudicate
against exactly this; no bar shopping; texture => TEXTURE with numbers):
  - DISUSE/GENERIC-PRESSURE fires if: g-12 falls <= 0.27 with NO fact
    teaching — any off-distribution training closes doors; the closure
    needs no graft, no fact; the three-way fork resolves DISUSE.
  - COMPETITIVE-FLAVORED fires if: g-12 recovers >= 0.7 while the site
    decays — the door was held shut by graft-building under supervision;
    without it, both revert.
  - DWELL-PERSISTS fires if: both doors hold >= 80% of s32's levels — a
    metastable mixed phase.
  - No bar shopping; texture => TEXTURE with numbers.

OPERATIONALIZATIONS (frozen here before compute — the registered clauses
name falls/recovers/holds without endpoint definitions; these fix them,
they do not move the bars):
  * g-12 = ABSOLUTE mean p(Z) on the install-60 battery at ctx offset -12
    (not a retention ratio; e158's convention). Freeze steps are
    CONTINUATION steps from the dwell peak: {0, 50, 100, 200, 300}.
  * DISUSE: primary = g-12(300) <= 0.27 (the SHUT bar); co-report the
    earliest checkpoint at/below 0.27 and the trajectory minimum.
  * COMPETITIVE: primary = max over checkpoints {50,100,200,300} of g-12
    >= 0.70 AND site content at that argmax checkpoint strictly below
    s32's site_span_strength 0.013817965984344482 ('the site decays' =
    the span-census band-max below the dwell peak's level); endpoint
    co-report; functional co-form (site_read_span below s32's 0.9945025)
    co-reported.
  * DWELL: primary (census form) = g-12(300) >= 0.8 * 0.513336181640625
    (= 0.4106689453125) AND site_span_strength(300) >= 0.8 *
    0.013817965984344482 (= 0.011054372787475586); functional co-form =
    site_read_span(300) >= 0.7956020 AND site_read_onset(300) >=
    0.7718244; a split (census fails, functional holds) is REPORTED, does
    not flip the bar.
  * 'both doors' = the geometry door (g-12) and the site door (site
    content); g+12 is a co-measured dial, not a bar clause.
  * Adjudication order: DISUSE -> COMPETITIVE-FLAVORED -> DWELL-PERSISTS
    -> TEXTURE; every sub-boolean reported regardless; co-fires reported.

COMPUTE ENVELOPE: CPU-ONLY by dispatch (CUDA_VISIBLE_DEVICES=-1; e152R owns
the GPU, e154 also on CPU): torch threads 4 (LOW <= 4), a 25 s launch
stagger against e154's CPU bursts (one sleep, no busy-waiting anywhere),
cooldown(60) before and after the ONE training, CPU training time cap
1800 s. The dispatch's '<=180 s cap' is the GPU single-training cap; the
CPU path follows e152's CPU-fallback precedent (1500 s at 8 threads gave
716 s / 300 steps; the mandated 4 threads need the headroom — QUEUE's own
row prices this cell's CPU training at 15-25 min). Nets are the 2.7M
e131_consolidated line (the '<=1M family' note is an envelope statement;
e143/e151/e152 precedent — the dwell peak lives on this line).

INSTRUMENT PROVENANCE: load_cpu / evl_load / battery_cell / battery_pz /
ce_fixed_cpu / val_windows / deleted_wpe / read_fact_at / row_census are
lab/e152_conversion_trace.py VERBATIM (e151/e143/e065/e068/e109/e113/e116/
e119/e131/e139 lineage). Copied, not imported, to own the device policy
(e152 imports force CUDA_VISIBLE_DEVICES=0; this rig forces -1).
finetune_freeze is e152's finetune_trace with the name-pool half REMOVED
(anchors + random corpus only, full-token CE) and freeze-step snapshots.
Protocol rebuild: corpus seed 1337, SPLICE_RNG 24301 host shuffle,
install60/held30 split, mix gate — e143/e151/e152 verbatim. Battery trimmed
vs e152's full battery to the dispatch's dial list (no mask/ladder/onset-
census probes; the span census is the registered site-content instrument).

Outputs: runs/e161/{metrics.json, freeze_cell.png}; checkpoints
runs/checkpoints/e161_freeze32.pt (the s32+300 endpoint) +
e161_freeze32_s{50,100,200}.pt (intermediates). No NOTES/THINKING/QUEUE/
STATE edits; single commit, no push.

Run:  cd lab && python e161_freeze_cell.py    (E161_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e152R owns
# the GPU; e154 shares the CPU — threads capped at 4 below)

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

SMOKE = os.environ.get("E161_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e161 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
ROOT_CK = "e152_steps32.pt"       # THE DWELL PEAK (both doors genuine)
CKPT_DIR = E43.REPO / "runs" / "checkpoints"

# ---- e152's placement constants (the measurement instruments rebuild these) ----
RETEACH_J = 54                    # offset: name x-cols 184..190, read rows 183..189
SITE_ADDR_ROW = 183               # the onset read row (= e120's SPLICE_ADDR_ROW)
SITE_Z_XCOL = PRE + RETEACH_J     # 184 (= e120's Z_XCOL)
SITE_CONT = BLOCK - PRE - RETEACH_J - len(NAME)     # 65-token continuation
SITE_ROWS = tuple(range(183, 190))                  # the 7 trained read rows
SHARED_CTR = (60, 100, 150, 160, 170, 200, 220)     # outside every trained band
D_ALL = (121, 125, 129, 133, 137)                   # e113's fixed set
GEOS = (-12, 0, 12)               # novel x2 + the trained g0

# ---- census row sets (e158's battery convention; read-visible rows only) ----
ROWS_183 = (0, 1, 2) + (181, 182) + SITE_ROWS + (60, 100, 150, 160, 170)
ROWS_OLD = (0, 1, 2, 3, 4, 5, 6, 60, 100, 118, 119, 120) + tuple(range(121, 130))
if SMOKE:
    ROWS_183 = (0, 1, 182) + (183, 185, 189) + (60, 100)
    ROWS_OLD = (0, 1, 60, 121, 125, 129)

# ---- the freeze: continuation steps from the dwell peak ------------------------
CKPT_STEPS: tuple[int, ...] = (50, 100, 200, 300) if not SMOKE else (2, 4)
CKPT_SET = set(CKPT_STEPS)

# ---- fine-tune envelope (e109 arm-b / e119-L / e143 / e151 / e152 verbatim) ----
FT_LR = 1e-3
FT_STEPS = CKPT_STEPS[-1]
FT_TIME_CAP = 1800.0              # CPU cap (e152 CPU precedent 1500 s @ 8 thr)
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
FREEZE_SEED = 10902               # the locked fine-tune seed lineage
COOLDOWN_S = 60.0                 # around the ONE training (CPU thermal)
STAGGER_S = 25.0                  # launch stagger vs e154's CPU bursts

# ---- gates / references (full precision, = e152's stored s32 cells) ------------
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05
E152_S32 = {                      # runs/e152/metrics.json trace row steps=32
    "base_gm12": 0.513336181640625,
    "base_g0": 0.14117614924907684,
    "base_gp12": 0.7145950198173523,
    "ce_r": 1.6883095502853394,
    "site_span_strength": 0.013817965984344482,
    "site_read_onset": 0.9647804498672485,
    "site_read_span": 0.9945025440030762,
    "A129": -0.3725217759458853,
    "row0_strength": 0.10077391087021775,
    "dall_g0": 0.5828545689582825,
}

# ---- registered bar constants (frozen; see OPERATIONALIZATIONS above) ----------
SHUT_BAR = 0.27                   # DISUSE g-12 bar (e158's SHUT convention)
RECOVER_BAR = 0.70                # COMPETITIVE g-12 recovery bar
DWELL_FRAC = 0.80                 # DWELL-PERSISTS fraction of s32's levels
SITE_CTRL_MULT = 2.0              # e139/e151 site convention: >= 2x control-max
S32_GM12 = E152_S32["base_gm12"]                 # 0.513336181640625
S32_SITE = E152_S32["site_span_strength"]        # 0.013817965984344482
S32_SITE_READ_SPAN = E152_S32["site_read_span"]  # 0.9945025440030762
S32_SITE_READ_ONSET = E152_S32["site_read_onset"]  # 0.9647804498672485
DWELL_GM12_BAR = DWELL_FRAC * S32_GM12           # 0.4106689453125
DWELL_SITE_BAR = DWELL_FRAC * S32_SITE           # 0.011054372787475586
DWELL_FUNC_SPAN_BAR = DWELL_FRAC * S32_SITE_READ_SPAN    # 0.7956020...
DWELL_FUNC_ONSET_BAR = DWELL_FRAC * S32_SITE_READ_ONSET  # 0.7718244...

REGISTERED_PREDICTION = {
    "disuse_generic_pressure": "DISUSE/GENERIC-PRESSURE fires if: g-12 falls "
        "<= 0.27 with NO fact teaching — any off-distribution training closes "
        "doors; the closure needs no graft, no fact; the three-way fork "
        "resolves DISUSE.",
    "competitive_flavored": "COMPETITIVE-FLAVORED fires if: g-12 recovers "
        ">= 0.7 while the site decays — the door was held shut by "
        "graft-building under supervision; without it, both revert.",
    "dwell_persists": "DWELL-PERSISTS fires if: both doors hold >= 80% of "
        "s32's levels — a metastable mixed phase.",
    "no_bar_shopping": "No bar shopping; texture => TEXTURE with numbers.",
    "operationalizations": "g-12 = absolute install-60 battery mean p(Z) at "
        "ctx offset -12 per checkpoint; freeze steps are CONTINUATION steps "
        f"from the dwell peak {{0, {list(CKPT_STEPS)}}}; DISUSE primary = "
        f"g-12(300) <= {SHUT_BAR} (earliest-below + min co-reported); "
        f"COMPETITIVE primary = max_{{50..300}} g-12 >= {RECOVER_BAR} AND "
        f"site_span_strength at the argmax checkpoint < s32's {S32_SITE:.16f}"
        " (functional co-form co-reported); DWELL primary (census form) = "
        f"g-12(300) >= {DWELL_GM12_BAR:.16f} AND site_span_strength(300) >= "
        f"{DWELL_SITE_BAR:.16f} (functional co-form {DWELL_FUNC_SPAN_BAR:.4f}"
        f"/{DWELL_FUNC_ONSET_BAR:.4f} co-reported; a split is reported, does "
        "not flip the bar); 'both doors' = geometry (g-12) + site content; "
        "order DISUSE -> COMPETITIVE -> DWELL -> TEXTURE; every sub-boolean "
        "reported regardless.",
    "committed": "QUEUE e161 registered no committed branch (three-bar fork; "
                 "T099 frames this cell as the REWRITE-vs-DISUSE decider).",
}

trims: list[str] = []
deviations: list[str] = [
    "CPU-ONLY by dispatch (CUDA_VISIBLE_DEVICES=-1 forced before torch; "
    "e152R owns the GPU, e154 shares the CPU): torch threads 4, one 25 s "
    "launch stagger (single sleep, no busy-waiting), cooldown(60) before/"
    "after the ONE training.",
    "The dispatch's '<=180 s cap' is the lab's GPU single-training cap; the "
    "CPU path runs under a 1800 s training cap (e152's CPU-fallback "
    "precedent: 1500 s at 8 threads, 716 s actual / 300 steps; the mandated "
    "4 threads need the headroom — QUEUE's own row prices this cell at "
    "15-25 min CPU training).",
    "Nets are the 2.7M e131_consolidated line (the dispatch's '<=1M family' "
    "note is an envelope statement; e143/e151/e152 precedent — the dwell "
    "peak, every gate reference, and every lineage number of this cell "
    "lives on the 2.7M line).",
    "Plain-corpus stream = 16 anchor-bank draws + 16 random corpus windows "
    "per step (batch 32): e152's anchor-half batch treatment generalized to "
    "the whole batch (full-token CE, no name mask — there are no fact "
    "windows to mask); the e119/E-road corpus convention (anchors + random "
    "windows from train_ids).",
    "Name-free guarantee implemented as a VERIFY, not a redraw: the raw "
    "corpus contains zero 'ZEPH' substrings (grep data/input.txt = 0 hits), "
    "and every drawn window is decoded and checked at draw time (violation "
    "would hard-fail, consuming no hidden RNG).",
    "Battery trimmed vs e152's full battery to the dispatch's dial list "
    "(no mask/ladder/onset-census probes; the span census is the registered "
    "site-content instrument, the onset functional read co-reported via "
    "site_read; d183 kept as the new door's row-necessity co-report).",
    "Eval thread count is 4 (dispatch) vs e152's 8 — CPU reduction order "
    "can drift low-order bits; the G_S32 gate therefore reports both the "
    "5e-6 bit flag and the 0.05 fallback tolerance (e152/e158 precedent).",
    "Single seed (10902), single lineage, ONE trajectory — the freeze "
    "outcome is a point estimate (n=1 path through continuation-step "
    "space).",
    "Smoke mode trims: 4-step freeze with checkpoints at {2,4}, reduced "
    "census rows, nothing adjudicated.",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: verbatim lineage — see the module docstring. Copied rather than
# imported (e152's rig sets CUDA_VISIBLE_DEVICES=0; this one forces -1).

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
    """e131's read_fact_position VERBATIM ARITHMETIC (e151/e152 copy): p(true
    name char) at positions addr_row..addr_row+6 over the pool windows."""
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
                    zid: int, seed: int):
    """THE PLAIN-CORPUS FREEZE: e152's finetune_trace with the name-pool half
    REMOVED. Per step: aj = randint(16) anchor draws, rj = randint(16) random
    corpus offsets; batch 32 full-token CE; AdamW (0.9,0.95) wd 0.1 lr 1e-3
    constant, clip 1.0. Snapshots (deep-copy out; nothing loaded into the
    training net) + light CPU evals at the freeze steps (no RNG consumed)."""
    net = copy.deepcopy(net0)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    n_anc = anchor.shape[0]
    traj, sds, t_start = [], {}, time.time()
    step = 0
    zeph_checks = 0
    evl = copy.deepcopy(net0)          # CPU eval twin
    for step in range(1, FT_STEPS + 1):
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
        if step in CKPT_SET or step % 50 == 0:
            log(f"  [{tag}] s{step:4d} corpus CE {float(loss.item()):.4f} "
                f"({time.time() - t_start:.0f}s)")
        if step in CKPT_SET:
            sd_cpu = {k: v.detach().cpu().clone()
                      for k, v in net.state_dict().items()}
            sds[step] = sd_cpu
            evl.load_state_dict(sd_cpu)
            evl.eval()
            gz = battery_cell(evl, gm12_ids, zid)
            ce_r = ce_fixed_cpu(evl, *r_eval_xy)
            traj.append({"step": step, "g_m12_mean_pz": gz["mean_pz"],
                         "frac_argmax_z": gz["frac_argmax_z"], "ce_r": ce_r,
                         "elapsed_s": round(time.time() - t_start, 1)})
            log(f"  [{tag}] CKPT +{step:4d} g-12 {gz['mean_pz']:.4f} "
                f"argmaxZ {gz['frac_argmax_z']:.2f} CE_R {ce_r:.4f} "
                f"({traj[-1]['elapsed_s']:.0f}s)")
        if (time.time() - t_start) > FT_TIME_CAP:
            log(f"  [{tag}] time cap {FT_TIME_CAP:.0f}s at s{step}")
            trims.append(f"{tag}: time cap at step {step}")
            break
    net.eval()
    return {"sds": sds, "traj": traj, "steps_ran": step, "seed": seed,
            "zeph_violations": zeph_checks}


CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "e161", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                             **meta}
    log(f"[ckpt] saved {path.name} "
        f"({', '.join(f'{k}={v}' for k, v in meta.items())})")


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e161_smoke" if SMOKE else "e161")
    log(f"E161 THE FREEZE-CELL (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()}), stagger "
        f"{STAGGER_S:.0f}s, cooldown {COOLDOWN_S:.0f}s around the ONE "
        f"training")
    time.sleep(STAGGER_S)            # launch stagger vs e154 (no busy-wait)

    # ---------------- protocol rebuild (e143/e151/e152 verbatim)
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
              "name_xcols": [SITE_Z_XCOL, SITE_Z_XCOL + L - 1],
              "all_windows_name_in_place": bool(
                  all(torch.equal(w[SITE_Z_XCOL: SITE_Z_XCOL + L], name_ids)
                      for w in pool_x)),
              "note": "MEASUREMENT instrument only (e152's locked j=54 pool); "
                      "NO window from this pool enters the freeze training"}
    G_POOL["pass"] = G_POOL["all_windows_name_in_place"]
    assert G_POOL["pass"], f"pool gate FAILED: {G_POOL}"

    # anchor bank (e065/e109/e143/e151/e152 verbatim) — the corpus half's
    # paired component; verified name-free
    anchor = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                          for p, _ in install_occ[:16]])
    anchor_zeph = sum(1 for w in anchor if "ZEPH" in corpus.decode(w))
    G_ANCHFREE = {"anchor_zeph_windows": anchor_zeph, "pass": bool(anchor_zeph == 0)}
    assert G_ANCHFREE["pass"], "anchor bank contains ZEPH"

    # ---------------- batteries: ctx = train_text[p-PRE-j : p] (e119 verbatim)
    bat_ids = {}
    for j in GEOS:
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    gm12_ids = bat_ids[-12]

    # ---------------- root net (THE DWELL PEAK) + gate vs e152's stored cells
    net0 = load_cpu(CKPT_DIR / ROOT_CK)
    sd_root = {k: v.clone() for k, v in net0.state_dict().items()}
    root_meta = None
    st_raw = torch.load(CKPT_DIR / ROOT_CK, map_location="cpu",
                        weights_only=False)
    if isinstance(st_raw, dict) and "meta" in st_raw:
        root_meta = E43.jsonable(st_raw["meta"])
    log(f"root: {ROOT_CK} (meta: {root_meta})")

    gates_surg: dict = {}

    def measure(sd: dict, tag: str) -> dict:
        net = evl_load(sd)
        out: dict = {"tag": tag}
        # (0) base expression + CE
        out["base"] = {j: battery_cell(net, bat_ids[j], zid) for j in GEOS}
        out["ce_r"] = ce_fixed_cpu(net, *r_eval_xy)
        log(f"[{tag}] base: " + " ".join(f"g{j:+d} {out['base'][j]['mean_pz']:.4f}"
                                         for j in GEOS)
            + f" | CE_R {out['ce_r']:.4f}")

        # (ii) site content at 183 — span-primary census + functional read
        def span_fn(n):
            return read_fact_at(n, pool_x, name_ids, zid, SITE_ADDR_ROW,
                                SITE_Z_XCOL)["pname_mean_over7"]

        out["site_read"] = read_fact_at(net, pool_x, name_ids, zid,
                                        SITE_ADDR_ROW, SITE_Z_XCOL)
        out["census183_span"] = row_census(net, ROWS_183, span_fn)
        cen = out["census183_span"]
        site_rows_present = [r for r in SITE_ROWS if str(r) in cen["rows"]]
        cm = max(cen["rows"][str(r)]["strength"] for r in SHARED_CTR
                 if str(r) in cen["rows"])
        sstr = max(cen["rows"][str(r)]["strength"] for r in site_rows_present)
        spos = any(cen["rows"][str(r)]["content"] and
                   cen["rows"][str(r)]["strength"] >= SITE_CTRL_MULT * cm
                   for r in site_rows_present)
        best = max(site_rows_present,
                   key=lambda r: cen["rows"][str(r)]["strength"])
        out["site_span"] = {"control_max": cm, "site_strength": sstr,
                            "bar_2x_control": SITE_CTRL_MULT * cm,
                            "site_pos": spos, "peak_row": int(best)}
        log(f"[{tag}] site(span) strength {sstr:+.4f} @r{best} "
            f"(2x-ctrl {SITE_CTRL_MULT * cm:.4f}) -> site_pos {spos} "
            f"| read onset {out['site_read']['pz_onset_mean']:.4f} "
            f"span {out['site_read']['pname_mean_over7']:.4f}")

        # old-band census (g0 readout) — the brake A(129) + row 0
        out["census_old"] = row_census(net, ROWS_OLD,
                                       lambda n: battery_pz(n, bat_ids[0], zid))
        co = out["census_old"]["rows"]
        out["old_band"] = {
            "base_pz": out["census_old"]["base_readout"],
            "row0": co["0"], "row129": co["129"],
            "row0_strength": co["0"]["strength"], "A129": co["129"]["strength"],
            "band121_129_max": max(co[str(r)]["strength"] for r in range(121, 130)
                                   if str(r) in co)}
        log(f"[{tag}] old band: row0 S {out['old_band']['row0_strength']:+.4f} "
            f"| A(129) {out['old_band']['A129']:+.4f} "
            f"(base {out['old_band']['base_pz']:.4f})")

        # deletion table: D-all (e113 set) + D-183 (the new door's necessity)
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
                raise RuntimeError(f"deletion gate FAILED {tag}/{dl}: {gate}")
            net.load_state_dict(sd_d)
            cell = {"g0": battery_cell(net, bat_ids[0], zid)}
            if dl == "d183":
                cell["gm12"] = battery_cell(net, bat_ids[-12], zid)
                cell["site_span"] = read_fact_at(
                    net, pool_x, name_ids, zid, SITE_ADDR_ROW,
                    SITE_Z_XCOL)["pname_mean_over7"]
            out["del_table"][dl] = cell
        net.load_state_dict(sd)                    # restore
        log(f"[{tag}] deletions g0: " + " | ".join(
            f"{dl} {out['del_table'][dl]['g0']['mean_pz']:.3f}"
            for dl in DELS))
        del net
        return out

    log("=" * 78)
    log("STEP-0 battery (root = the dwell peak e152_steps32; 'before')")
    root = measure(sd_root, "s32")
    root_cells = {
        "base_gm12": root["base"][-12]["mean_pz"],
        "base_g0": root["base"][0]["mean_pz"],
        "base_gp12": root["base"][12]["mean_pz"],
        "ce_r": root["ce_r"],
        "site_span_strength": root["site_span"]["site_strength"],
        "site_read_onset": root["site_read"]["pz_onset_mean"],
        "site_read_span": root["site_read"]["pname_mean_over7"],
        "A129": root["old_band"]["A129"],
        "row0_strength": root["old_band"]["row0_strength"],
        "dall_g0": root["del_table"]["d_all"]["g0"]["mean_pz"],
    }
    diffs = {k: root_cells[k] - E152_S32[k] for k in root_cells}
    max_abs = max(abs(v) for v in diffs.values())
    G_S32 = {"cells": root_cells, "e152_stored": E152_S32, "diffs": diffs,
             "max_abs_diff": max_abs, "bit_tol": G_BIT_TOL,
             "tol": G_FALLBACK_TOL,
             "bit_reproducible": bool(max_abs < G_BIT_TOL),
             "pass": bool(max_abs < G_FALLBACK_TOL)}
    log(f"G_S32 dwell-peak gate: max|diff| {max_abs:.2e} "
        f"(tol {G_FALLBACK_TOL}, bit {G_BIT_TOL}): "
        f"{'PASS' if G_S32['pass'] else 'FAIL'}")
    if not G_S32["pass"]:
        raise RuntimeError("dwell-peak checkpoint failed its gate vs e152 "
                           "stored s32 cells")
    log("gates: G_SPLICE, G_NAMEFREE, G_POOL, G_ANCHFREE, G_S32 all PASS")

    # =====================================================================
    # THE FREEZE (ONE plain-corpus training; CPU; cooldown around it)
    # =====================================================================
    log("=" * 78)
    log(f"[thermal] cooldown {COOLDOWN_S:.0f}s before the ONE training")
    cooldown(COOLDOWN_S)
    log(f"FREEZE: {FT_STEPS}-step plain-corpus fine-tune of the dwell peak "
        f"(batch {ANCH_BS} anchors + {RAND_BS} random, full-token CE, seed "
        f"{FREEZE_SEED}, CPU {torch.get_num_threads()} threads), "
        f"snapshots at +{list(CKPT_STEPS)}")
    freeze = finetune_freeze("freeze_plain", net0, anchor, train_ids, itos,
                             r_eval_xy, gm12_ids, zid, FREEZE_SEED)
    log(f"[thermal] cooldown {COOLDOWN_S:.0f}s after the ONE training")
    cooldown(COOLDOWN_S)
    G_DRAWFREE = {"zeph_violations": freeze["zeph_violations"],
                  "pass": bool(freeze["zeph_violations"] == 0)}
    assert G_DRAWFREE["pass"], "name token leaked into a training window"

    save_ckpt("e161_freeze32", freeze["sds"][max(freeze["sds"])],
              {"desc": f"e152_steps32 (the dwell peak) + "
                       f"{max(freeze['sds'])}-step PLAIN-CORPUS freeze "
                       f"(no fact windows, no name tokens; batch 32 = 16 "
                       f"anchors + 16 random corpus, full-token CE), seed "
                       f"{FREEZE_SEED}",
               "steps": int(max(freeze["sds"])), "seed": FREEZE_SEED,
               "base": f"runs/checkpoints/{ROOT_CK}"})
    for s in sorted(freeze["sds"]):
        if s == max(freeze["sds"]):
            continue
        save_ckpt(f"e161_freeze32_s{s}", freeze["sds"][s],
                  {"desc": f"e152_steps32 + {s}-step plain-corpus freeze "
                           f"(intermediate snapshot), seed {FREEZE_SEED}",
                   "steps": int(s), "seed": FREEZE_SEED,
                   "base": f"runs/checkpoints/{ROOT_CK}"})
    missing = [s for s in CKPT_STEPS if s not in freeze["sds"]]
    if missing:
        trims.append(f"checkpoints not reached (time cap): {missing}")

    log("=" * 78)
    batteries = {"s32": root}
    for s in sorted(freeze["sds"]):
        log(f"FREEZE+{s} battery")
        batteries[str(s)] = measure(freeze["sds"][s], f"f{s}")

    # =====================================================================
    # THE FREEZE TRAJECTORY + ADJUDICATION (registered clauses; no shopping)
    # =====================================================================
    steps_meas = [0] + sorted(s for s in freeze["sds"])
    trace = []
    for s in steps_meas:
        b = batteries["s32" if s == 0 else str(s)]
        row = {
            "freeze_steps": s,
            "base_gm12": b["base"][-12]["mean_pz"],
            "base_g0": b["base"][0]["mean_pz"],
            "base_gp12": b["base"][12]["mean_pz"],
            "retention_vs_s32_gm12": b["base"][-12]["mean_pz"] / S32_GM12,
            "retention_vs_s32_gp12": b["base"][12]["mean_pz"] / E152_S32["base_gp12"],
            "retention_vs_s32_g0": b["base"][0]["mean_pz"] / E152_S32["base_g0"],
            "ce_r": b["ce_r"],
            "site_read_onset": b["site_read"]["pz_onset_mean"],
            "site_read_span": b["site_read"]["pname_mean_over7"],
            "site_span_strength": b["site_span"]["site_strength"],
            "site_span_peak_row": b["site_span"]["peak_row"],
            "site_span_bar_2x_control": b["site_span"]["bar_2x_control"],
            "site_pos_span": b["site_span"]["site_pos"],
            "row183_span_strength": b["census183_span"]["rows"]["183"]["strength"]
                if "183" in b["census183_span"]["rows"] else None,
            "A129": b["old_band"]["A129"],
            "row0_strength": b["old_band"]["row0_strength"],
            "band121_129_max": b["old_band"]["band121_129_max"],
            "dall_g0": b["del_table"]["d_all"]["g0"]["mean_pz"],
            "d183_g0": b["del_table"]["d183"]["g0"]["mean_pz"],
            "d183_gm12": b["del_table"]["d183"]["gm12"]["mean_pz"],
            "d183_site_span": b["del_table"]["d183"]["site_span"],
        }
        trace.append(row)

    g_seq = [r["base_gm12"] for r in trace]
    site_seq = [r["site_span_strength"] for r in trace]
    ck_idx = [i for i, r in enumerate(trace) if r["freeze_steps"] > 0]
    last = trace[-1]
    g_ck = [(trace[i]["freeze_steps"], g_seq[i]) for i in ck_idx]
    g_final = g_seq[-1]
    g_max_i = max(ck_idx, key=lambda i: g_seq[i])
    earliest_below = next((trace[i]["freeze_steps"] for i in ck_idx
                           if g_seq[i] <= SHUT_BAR), None)
    g_min = min(g_seq[i] for i in ck_idx)

    disuse_fires = bool(g_final <= SHUT_BAR)
    comp_g = any(g_seq[i] >= RECOVER_BAR for i in ck_idx)
    comp_site_decay = bool(site_seq[g_max_i] < S32_SITE)
    comp_fires = bool(comp_g and comp_site_decay)
    dwell_g = bool(g_final >= DWELL_GM12_BAR)
    dwell_site = bool(site_seq[-1] >= DWELL_SITE_BAR)
    dwell_census = bool(dwell_g and dwell_site)
    dwell_func_span = bool(last["site_read_span"] >= DWELL_FUNC_SPAN_BAR)
    dwell_func_onset = bool(last["site_read_onset"] >= DWELL_FUNC_ONSET_BAR)
    dwell_func = bool(dwell_func_span and dwell_func_onset)
    dwell_split = bool(dwell_g and not dwell_site and dwell_func)

    cond = {
        "DISUSE": {
            "bar": SHUT_BAR, "g_final": g_final,
            "earliest_ck_le_bar": earliest_below, "g_min": g_min,
            "fires": disuse_fires,
        },
        "COMPETITIVE": {
            "g_recover_bar": RECOVER_BAR,
            "g_max": g_seq[g_max_i], "g_max_at": trace[g_max_i]["freeze_steps"],
            "site_at_g_max": site_seq[g_max_i], "s32_site": S32_SITE,
            "site_decay_clause": comp_site_decay,
            "endpoint_site_below_s32": bool(site_seq[-1] < S32_SITE),
            "endpoint_site_read_span": last["site_read_span"],
            "s32_site_read_span": S32_SITE_READ_SPAN,
            "fires": comp_fires,
        },
        "DWELL": {
            "frac": DWELL_FRAC,
            "gm12_bar": DWELL_GM12_BAR, "site_bar": DWELL_SITE_BAR,
            "g_final": g_final, "site_final": site_seq[-1],
            "g_clause": dwell_g, "site_clause": dwell_site,
            "census_form_fires": dwell_census,
            "func_span_bar": DWELL_FUNC_SPAN_BAR,
            "func_onset_bar": DWELL_FUNC_ONSET_BAR,
            "func_read_span": last["site_read_span"],
            "func_read_onset": last["site_read_onset"],
            "functional_form_fires": dwell_func,
            "split_reported": dwell_split,
            "fires": dwell_census,
        },
    }
    if disuse_fires:
        verdict = "DISUSE/GENERIC-PRESSURE"
        clause = (f"g-12 falls to {g_final:.4f} <= {SHUT_BAR} at +300 "
                  f"plain-corpus steps with NO fact teaching (earliest "
                  f"checkpoint <= bar: {earliest_below}; trajectory min "
                  f"{g_min:.4f}) — any off-distribution training closes "
                  f"doors; the closure needs no graft, no fact; the "
                  f"three-way fork resolves DISUSE.")
    elif comp_fires:
        verdict = "COMPETITIVE-FLAVORED"
        clause = (f"g-12 recovers to {g_seq[g_max_i]:.4f} >= {RECOVER_BAR} "
                  f"(at +{trace[g_max_i]['freeze_steps']}) while the site "
                  f"decays (census {site_seq[g_max_i]:+.4f} < s32's "
                  f"{S32_SITE:.4f}) — the door was held shut by "
                  f"graft-building under supervision; without it, both "
                  f"revert.")
    elif dwell_census:
        verdict = "DWELL-PERSISTS"
        clause = (f"both doors hold >= {DWELL_FRAC:.0%} of s32's levels at "
                  f"+300: g-12 {g_final:.4f} >= {DWELL_GM12_BAR:.4f} AND "
                  f"site census {site_seq[-1]:+.4f} >= {DWELL_SITE_BAR:.4f} "
                  f"(functional read span {last['site_read_span']:.4f}, "
                  f"onset {last['site_read_onset']:.4f}) — a metastable "
                  f"mixed phase.")
    else:
        verdict = "TEXTURE"
        clause = (f"no registered bar fired cleanly: g-12 trace "
                  f"{['%.4f' % g for g in g_seq]} (final {g_final:.4f} vs "
                  f"DISUSE<= {SHUT_BAR}, COMPETITIVE>= {RECOVER_BAR}, "
                  f"DWELL>= {DWELL_GM12_BAR:.4f}); site census trace "
                  f"{['%+.4f' % s for s in site_seq]} (DWELL bar "
                  f"{DWELL_SITE_BAR:.4f}); functional site read "
                  f"{last['site_read_span']:.4f}/{last['site_read_onset']:.4f}"
                  f" (bars {DWELL_FUNC_SPAN_BAR:.4f}/"
                  f"{DWELL_FUNC_ONSET_BAR:.4f}); split={dwell_split} — "
                  f"numbers reported, no bar shopping.")
    log("=" * 78)
    log(f"E161 VERDICT: {verdict}")
    log(f"  g-12 trace (freeze steps {steps_meas}): " + " -> ".join(
        f"+{r['freeze_steps']}:{r['base_gm12']:.4f}" for r in trace))
    log(f"  site census trace: " + " -> ".join(
        f"+{r['freeze_steps']}:{r['site_span_strength']:+.4f}" for r in trace))
    log(f"  A(129) trace: " + " -> ".join(
        f"{r['A129']:+.3f}" for r in trace))
    log(f"  D-all g0 trace: " + " -> ".join(f"{r['dall_g0']:.3f}" for r in trace))
    log(f"  CE_R trace: " + " -> ".join(f"{r['ce_r']:.3f}" for r in trace))
    log(f"  {clause}")
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e161_freeze_cell",
        "date": common.now_iso(),
        "registration": ("QUEUE e161 row (re-registered post-e166 per the "
                         "R48 honesty note) + T099 + the MAP UPDATE, "
                         "dispatched ~14:05Z with verbatim bars; "
                         "operationalizations frozen in the module docstring "
                         "before compute"),
        "registered_prediction": REGISTERED_PREDICTION,
        "committed_prediction": None,
        "question": ("does the geometry door close with NO fact teaching at "
                     "all — i.e., does plain-corpus pressure on the dwell "
                     "peak (e152_steps32, both doors genuine) close g-12 "
                     "(DISUSE), recover it while the site decays "
                     "(COMPETITIVE-FLAVORED), or leave both doors standing "
                     "(DWELL-PERSISTS)?"),
        "root": f"runs/checkpoints/{ROOT_CK} (the dwell peak; loaded, gated "
                f"vs e152's stored s32 cells, max|diff| "
                f"{G_S32['max_abs_diff']:.2e})",
        "root_meta": root_meta,
        "freeze": {"desc": "ONE plain-corpus fine-tune of the dwell peak: "
                           "batch 32 = 16 anchor-bank draws + 16 random "
                           "corpus windows, full-token CE (NO fact windows, "
                           "NO name tokens, NO mask), AdamW (0.9,0.95) wd "
                           "0.1, constant lr 1e-3, clip 1.0",
                   "recipe_lineage": "e109 arm-b / e119-L / e143 / e151 / "
                                     "e152 optimizer verbatim; corpus half "
                                     "= e119 road-E corpus convention",
                   "steps_ran": freeze["steps_ran"], "seed": FREEZE_SEED,
                   "ckpt_steps": list(CKPT_STEPS),
                   "missing_checkpoints": missing,
                   "traj": freeze["traj"], "device": "cpu",
                   "torch_threads": torch.get_num_threads(),
                   "time_cap_s": FT_TIME_CAP,
                   "cooldown_s": COOLDOWN_S, "stagger_s": STAGGER_S,
                   "zeph_violations": freeze["zeph_violations"]},
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
                     "measurement_pool": {"offset": RETEACH_J,
                                          "name_xcols": [SITE_Z_XCOL,
                                                         SITE_Z_XCOL + 6],
                                          "read_rows": [183, 189],
                                          "note": "e152's locked j=54 pool, "
                                                  "instrument only"},
                     "geometries_measured": {"novel": [-12, 12],
                                             "trained_g0": 0}},
        "gates": {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                  "G_POOL": G_POOL, "G_ANCHFREE": G_ANCHFREE,
                  "G_S32": G_S32, "G_DRAWFREE": G_DRAWFREE,
                  "G_SURG": gates_surg},
        "trace": trace,
        "trace_summary": {
            "freeze_steps": steps_meas,
            "base_gm12": [r["base_gm12"] for r in trace],
            "base_gp12": [r["base_gp12"] for r in trace],
            "base_g0": [r["base_g0"] for r in trace],
            "site_span_strength": [r["site_span_strength"] for r in trace],
            "row183_span_strength": [r["row183_span_strength"] for r in trace],
            "site_read_span": [r["site_read_span"] for r in trace],
            "site_read_onset": [r["site_read_onset"] for r in trace],
            "A129": [r["A129"] for r in trace],
            "row0_strength": [r["row0_strength"] for r in trace],
            "dall_g0": [r["dall_g0"] for r in trace],
            "ce_r": [r["ce_r"] for r in trace],
        },
        "batteries": batteries,
        "adjudication": {"conditions": cond, "verdict": verdict,
                         "clause": clause},
        "honesty_reflex": {
            "name_leakage": (
                f"raw corpus contains {corpus_zeph} 'ZEPH' substrings; all "
                f"{freeze['steps_ran'] * RAND_BS} random training windows "
                f"verified name-free at draw time (violations: "
                f"{freeze['zeph_violations']}); anchor bank verified "
                "name-free — zero name-token leakage into the freeze; host "
                "names (FLORIZEL/ELIZABETH) are corpus-native and appear in "
                "anchors as in e152's own anchor half"),
            "single_seed": "ONE trajectory (seed 10902) from ONE dwell-peak "
                "snapshot; the freeze outcome is a point estimate — "
                "non-monotonic texture is this trajectory's, not a "
                "replicated law",
            "dwell_peak_variance": "the starting point e152_steps32 is one "
                "point on ONE e152 trajectory (seed 10902 of the e152R "
                "re-seed fleet); e152's own s16->s32 bounce (0.447->0.561) "
                "and e158's cross-device scatter (~+-0.029, locked@band "
                "straddling the 0.5 bar across passes) price the family's "
                "noise scale — a different dwell peak might freeze "
                "differently; e152R's 3 seeds are the standing variance "
                "cell",
            "thread_bit_drift": "evals at 4 threads vs e152's 8 can drift "
                "low-order bits; the G_S32 gate reports both the 5e-6 bit "
                "flag and the 0.05 fallback tolerance",
            "off_distribution_definition": "plain corpus IS off-distribution "
                "for the +/-12 geometry reads (no battery-context windows "
                "are trained); 'generic pressure' therefore means "
                "ordinary-corpus gradient flow, not matched-distribution "
                "replay — the DISUSE clause's 'any off-distribution "
                "training' is tested at exactly one such distribution",
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

    plot(rd / "freeze_cell.png", trace, batteries, cond, verdict, clause, G_S32)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'freeze_cell.png'}, ckpts "
        f"runs/checkpoints/e161_freeze32{{,_s{','.join(str(s) for s in sorted(freeze['sds']) if s != max(freeze['sds']))}}}.pt")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot

def plot(path, trace, batteries, cond, verdict, clause, g_s32):
    """THE figure: g-12 vs plain-corpus freeze steps with the site census
    overlaid; plus census spectra, the brake/wreckage dials, and the verdict
    panel."""
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 10.5))
    xs = [r["freeze_steps"] for r in trace]
    ck = [r for r in trace if r["freeze_steps"] > 0]

    # (0,0) THE freeze trace: g-12 (left) vs site census (right)
    ax = axes[0, 0]
    ax.plot(xs, [r["base_gm12"] for r in trace], "o-", ms=8, lw=2.4,
            color="crimson", label="g-12 (the geometry door, HEADLINE)")
    ax.plot(xs, [r["base_gp12"] for r in trace], "s-", ms=5, lw=1.1,
            color="darkorange", alpha=0.8, label="g+12 (co-dial)")
    ax.plot(xs, [r["base_g0"] for r in trace], "^-", ms=5, lw=1.1,
            color="dimgray", alpha=0.8, label="g0 (home)")
    for yv, col, lbl in ((SHUT_BAR, "tab:purple",
                          f"{SHUT_BAR} DISUSE shut bar"),
                         (RECOVER_BAR, "seagreen",
                          f"{RECOVER_BAR} COMPETITIVE recovery bar"),
                         (DWELL_GM12_BAR, "tab:blue",
                          f"{DWELL_GM12_BAR:.3f} 80% of s32 g-12 (DWELL)"),
                         (S32_GM12, "lightgray", "s32 g-12 = 0.5133")):
        ax.axhline(yv, ls="--", lw=1.0, color=col, alpha=0.75, label=lbl)
    axr = ax.twinx()
    axr.plot(xs, [r["site_span_strength"] for r in trace], "D-.", ms=6,
             lw=1.8, color="seagreen", label="site content: band-max "
             "strength (span census)")
    axr.plot(xs, [r["row183_span_strength"] for r in trace], "v:", ms=5,
             lw=1.0, color="mediumseagreen", alpha=0.85,
             label="row-183 strength (co-report)")
    axr.plot(xs, [r["site_span_bar_2x_control"] for r in trace], "x--",
             ms=5, lw=1.0, color="k", alpha=0.6,
             label="2x-control content bar (per checkpoint)")
    axr.axhline(DWELL_SITE_BAR, ls=":", lw=1.2, color="seagreen",
                alpha=0.9, label=f"{DWELL_SITE_BAR:.5f} 80% of s32 site")
    for r in ck:
        if r["site_pos_span"]:
            axr.plot([r["freeze_steps"]], [r["site_span_strength"]],
                     marker="*", ms=17, color="gold", mec="k", mew=0.7,
                     zorder=5)
    ax.set_xlabel("plain-corpus freeze steps from the dwell peak "
                  "(NO fact teaching; * = site content clears the bar; "
                  "step 0 = e152_steps32 itself)")
    ax.set_ylabel("absolute mean p(Z), install-60 battery")
    axr.set_ylabel("site content @183 (census strength)")
    ax.set_ylim(-0.03, 1.05)
    axr.set_ylim(-0.004, max(0.03, 1.4 * max(
        r["site_span_strength"] for r in trace)))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.0, loc="center left")
    ax.set_title("THE FREEZE-CELL — g-12 (left) vs site census (right) under "
                 f"plain corpus\nverdict: {verdict}", fontsize=10)

    # (0,1) site census spectra across checkpoints (span readout)
    ax = axes[0, 1]
    cmap = plt.cm.coolwarm
    n_lines = len(trace)
    for i, r in enumerate(trace):
        b = batteries["s32" if r["freeze_steps"] == 0 else str(r["freeze_steps"])]
        cen = b["census183_span"]["rows"]
        xs_r = sorted(int(k) for k in cen
                      if int(k) not in SHARED_CTR and int(k) >= 100)
        col = "k" if r["freeze_steps"] == 0 else cmap(i / max(1, n_lines - 1))
        ax.plot(xs_r, [cen[str(x)]["strength"] for x in xs_r], "-o",
                ms=3.5, lw=1.3 if r["freeze_steps"] in (0, CKPT_STEPS[-1]) else 0.9,
                color=col, alpha=0.95,
                label="s32 root" if r["freeze_steps"] == 0 else
                      (f"+{r['freeze_steps']}" if r["freeze_steps"] in
                       (CKPT_STEPS[0], CKPT_STEPS[-1]) else None))
    ax.axvspan(183, 189, color="crimson", alpha=0.08)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("wpe row (red shade: site band 183-189)")
    ax.set_ylabel("strength (span census)")
    ax.set_title("(ii) SITE-CONTENT CENSUS spectra — s32 root (black) "
                 f"through +{CKPT_STEPS[-1]} (warm): does the site-store "
                 "decay without supervision?", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")

    # (1,0) the brake, the sink, the wreckage guard vs freeze steps
    ax = axes[1, 0]
    ax.plot(xs, [r["A129"] for r in trace], "o-", ms=6, lw=1.8,
            color="tab:purple", label="A(129) address-key strength (the brake)")
    ax.plot(xs, [r["dall_g0"] for r in trace], "s-", ms=6, lw=1.8,
            color="tab:blue", label="D-all g0 (deletion tolerance)")
    ax.plot(xs, [r["row0_strength"] for r in trace], "^-", ms=6, lw=1.5,
            color="tab:cyan", label="row-0 strength (sink)")
    ax.plot(xs, [r["d183_site_span"] for r in trace], "d--", ms=5, lw=1.0,
            color="crimson", alpha=0.8,
            label="D-183 site span (new door's row-necessity)")
    axr = ax.twinx()
    axr.plot(xs, [r["ce_r"] for r in trace], "k:o", ms=5, lw=1.2,
             alpha=0.8, label="CE_R (wreckage guard)")
    ax.set_xlabel("plain-corpus freeze steps")
    ax.set_ylabel("strength / p(Z)")
    axr.set_ylabel("CE_R")
    ax.axhline(0, color="k", lw=0.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="center right")
    ax.set_title("(iii) the brake / sink / deletion dials — what plain "
                 "corpus does to the rest of the anatomy", fontsize=10)

    # (1,1) verdict panel
    ax = axes[1, 1]
    ax.axis("off")
    g_txt = "  ".join(f"+{r['freeze_steps']}:{r['base_gm12']:.4f}" for r in trace)
    site_txt = "  ".join(
        f"+{r['freeze_steps']}:{r['site_span_strength']:+.4f}" for r in trace)
    vlines = [
        "REGISTERED (QUEUE e161 verbatim; frozen operationalizations):",
        f"  DISUSE: g-12(300) <= {SHUT_BAR} with NO fact teaching",
        f"  COMPETITIVE: max g-12 >= {RECOVER_BAR} while the site decays",
        f"  DWELL: both doors >= {DWELL_FRAC:.0%} of s32 "
        f"(g-12 {DWELL_GM12_BAR:.4f} / site {DWELL_SITE_BAR:.5f})",
        "",
        "TRACE (freeze steps from the dwell peak):",
        f"  g-12:          {g_txt}",
        f"  site census:   {site_txt}",
        f"  site read:     " + "  ".join(
            f"+{r['freeze_steps']}:{r['site_read_span']:.3f}" for r in trace),
        f"  A(129):        " + " ".join(f"{r['A129']:+.3f}" for r in trace),
        f"  D-all g0:      " + " ".join(f"{r['dall_g0']:.3f}" for r in trace),
        f"  CE_R:          " + " ".join(f"{r['ce_r']:.3f}" for r in trace),
        "",
        f"  DISUSE fires={cond['DISUSE']['fires']} "
        f"(final {cond['DISUSE']['g_final']:.4f}, earliest<=bar "
        f"{cond['DISUSE']['earliest_ck_le_bar']}, min {cond['DISUSE']['g_min']:.4f})",
        f"  COMPETITIVE fires={cond['COMPETITIVE']['fires']} "
        f"(max {cond['COMPETITIVE']['g_max']:.4f} @"
        f"+{cond['COMPETITIVE']['g_max_at']}, site there "
        f"{cond['COMPETITIVE']['site_at_g_max']:+.4f} vs s32 "
        f"{S32_SITE:+.4f})",
        f"  DWELL fires={cond['DWELL']['fires']} "
        f"(g {cond['DWELL']['g_clause']} site {cond['DWELL']['site_clause']} "
        f"func {cond['DWELL']['functional_form_fires']} "
        f"split {cond['DWELL']['split_reported']})",
        "",
        f"G_S32 dwell-peak gate: max|diff| {g_s32['max_abs_diff']:.2e} "
        f"({'PASS' if g_s32['pass'] else 'FAIL'})",
        "",
        f"VERDICT: {verdict}",
    ] + [f"  {wd}" for wd in
         [clause[i:i + 78] for i in range(0, len(clause), 78)]]
    for i, tx in enumerate(vlines):
        ax.text(0.02, 0.97 - i * 0.038, tx, fontsize=7.3, va="top",
                family="monospace")

    fig.suptitle(f"E161 — THE FREEZE-CELL (root e152_steps32, the dwell "
                 f"peak; +{CKPT_STEPS[-1]} plain-corpus steps, no fact "
                 f"teaching) -> {verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
