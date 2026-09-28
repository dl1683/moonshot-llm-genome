"""E119 — THE MIGRATION HEAD-TO-HEAD (P5 road 1v2; REGISTERED).

MISSION (scratch/next_arc_programs.md, "1. P5 — MIGRATION ANATOMY", the
registered design — verbatim authority): when the same fact consolidates
from the address-store into the distributed field via JITTERED REPLAY
(road R) vs DELETION PRESSURE / erase cycles (road E), is it the same fact
in the same store, or different stores with different anatomy?

TWIN INSTALLS (registered choice): "e043 protocol on B43-class bases, or
two same-seed installs". This run uses the SAME-SEED limit of that clause:
both roads start from the ONE bit-identical installed net —
runs/checkpoints/e048_repro.pt (the e043 Dmix@s400 install line, corpus
seed 1337, SPLICE_RNG 24301, install60 battery p(Z) 0.556313, the e109
line). Two same-seed installs are bit-identical by construction, so the
shared start IS the twin pair, maximally matched: every arm difference is
route-caused, nothing is install-caused. Same fact (ZEPHYRA@row129), same
family (2.7M), same seed.

ARMS (all start from the shared install):
  R (road R)  — jittered replay VERBATIM (e109 arm a): pool of 300 windows
                at offsets {-8,-4,0,+4,+8}, batch 32 = 16 install + 16
                anchor (8 paired + 8 random), e043 token-level union CE,
                AdamW (0.9,0.95) wd 0.1 constant lr 1e-3 clip 1.0, seed
                10901, default 300 steps (<=180 s GPU wall, in-loop evals
                every 25).
  E (road E)  — three erase cycles VERBATIM (e083): per cycle, D2 row-reset
                erase wte[Z]=0 & lm_head[Z]=0 (confinement gate 2*192
                elements + dCE <= 0.01 + strictly-off-bar gate), then the
                e044b Dmix re-learn battery (batch 8 name + 24 corpus = 8
                paired anchors of 60 + 16 random, cosine total 1000 / warmup
                100, token-weighted union CE, exposure seed 24401 EVERY
                cycle, FULL 300-step battery, <=180 s of TRAINING compute,
                cycles chain from their END states).
  L (locked)  — the free-rider third arm: matched-mass LOCKED replay
                (e109 arm b verbatim: offset-0 pool only, seed 10902, same
                step count as R — e109's own control).

GATE (do not skip; pre-registered): match final fact-expression at the
trained geometry across arms — the trained geometry COMMON to all arms is
offset 0 (E and L train only there; R trains there among its five).
Calibration dials (pre-registered freedom ONLY): R's step count on its
25-step eval grid (deterministic same-seed reruns) and E's cycle count in
{1,2,3} (each cycle's END state is kept). The gate PASSES if
|p_R(g0) - p_E(g0)| <= 0.20 on the install-60 battery; the calibration
search and its record are reported. If the gate CANNOT be met within that
freedom, PARK: write the calibration record, no comparative battery, no
adjudication. L rides at R's step count (matched mass), reported, never
gated (it is the free-rider control).

THEN the comparative battery (eval-only, all CPU):
  (a) deletion hierarchy x geometries: per arm, none -> D129 (original
      address row) -> D-grown (that arm's census-grown rows, 129 excluded)
      -> D-all (129 + grown), x 9 geometries (5 trained {-8..+8} + 4 novel
      {-12,-2,+2,+12}) x {install-60, held-30}. Novel-offset batteries use
      e068 left-extension verbatim. Honesty note (e109's): under row-zero
      deletions, j>0 batteries contain the deleted rows as CONTEXT rows;
      j<0 batteries never touch them.
  (b) grown-row census: which wpe rows grew (pre-registered rule: row r in
      1..255, r != 129, |wpe_arm[r]| - |wpe_base[r]| >= +0.04), each grown
      row's cos to the ORIGINAL address direction (wpe_base[129]); full
      256-row delta-norm spectrum kept for the plate; threshold
      sensitivity 0.03/0.05 report-only.
  (c) brake sign at full field (e115's dimmer switch): per arm,
      brake = p(D129, g0, install60) - p(none, g0, install60) with paired
      bootstrap CI. Does deleting the original address row SUPPRESS or
      FEED the field? Sign classes with a 0.02 magnitude floor.
  (d) share measurement r*(k)*k at the fact's collapse boundary: the
      e115 graded field instrument (V <- r*V entry-wise on the retained
      k-subset of context band positions 1..128, complement zeroed, every
      layer/head/query, K untouched, row 0 and the address coordinate 129
      never scaled) crossed with the e110 k-ladder (k in {32,64,128};
      r-grid 1.0/0.9/0.8/0.7/0.6/0.5/0.35/0.2/0.1; 2 subsets,
      subset-paired across arms, RNG default_rng(119) consumed
      k-major/subset-minor, REUSED across r and arms). PRIMARY read with
      the arm's FULL address set zeroed (e113 convention: field-only
      expression); D129-only secondary at r=1. r*(k) = min r with mean
      battery p(Z) >= 0.20 (the lab's survive bar); share product
      r*(k)*k; off-grid-high (never survives) recorded censored.
  (e) family-cos typing of the migrated content (e108 V-geometry): per
      arm, dV = (arm net's value vectors) - (base install's) at band
      positions on the install-60 g0 battery, per layer/head; readout =
      mean |cos(dV_X, dV_Y)| for road pairs (R,E) (R,L) (E,L) vs the null
      (within-layer/head position derangement + seeded permutation).

REGISTERED PREDICTION (adjudicate honestly against exactly this):
- SAME-STORE fires if: routes match on (c) brake sign and (d) share
  product within per-net band, AND their (b) grown-row censuses overlap in
  position-band.
- DIFFERENT-STORES fires if: >=2 readouts dissociate (e.g., R grows new
  address rows where E grows none; R survives scaffold loss at held-30
  where E doesn't; opposite brake signs).
- Bars must fire only as registered. No bar shopping. If the result is
  ambiguous, say AMBIGUOUS with the numbers.

OPERATIONALIZATIONS (frozen before compute):
  (c) match  = same sign class (both |brake| < 0.02, or same sign with
               both |brake| >= 0.02); dissociate = opposite signs, both
               above the 0.02 floor.
  (d) match  = at >= 2 of the 3 ks where BOTH roads are on-grid, the
               share products agree within a factor < 2 (the per-net
               band: products live on a coarse r-grid whose adjacent
               rungs differ by ~1.1-1.5x; a factor-2 gap is a full
               boundary shift). dissociate = products differ by >= 2x at
               >= 2 on-grid ks, OR one road off-grid-high at >= 2 ks while
               the other is on-grid there.
  (b) overlap = the grown-row position bands [min,max] of R and E
               intersect or the sets share a row (both-empty also counts:
               no road grew anything); dissociate = one road grows
               decision-window rows ([121,137]) the other does not.
  held-30 rider (from (a)) = dissociates if under D-all one road survives
               held-30 (max_g >= 0.20) while the other collapses
               (max_g <= 0.05).
  (e) is REPORT-ONLY (typing texture; alignment >> null = same content
               family, at null = different writing; never gates a bar).
  SAME-STORE requires (c) match AND (d) match AND (b) overlap.
  DIFFERENT-STORES requires >= 2 dissociations among {(b), (c), (d),
  held-30 rider}. Anything else -> AMBIGUOUS with the numbers.

CALIBRATION REFERENCES (runs/e109, runs/e083, runs/e113, runs/e115):
  base install p(Z) g0 0.556313 (G_INST tol 0.005); e109 arm a (GPU)
  none-table 0.992/0.950/0.776/0.985/0.985, D129 g0 0.909 (G_R_REPRO tol
  0.05/cell when R runs its registered 300 steps); e113's CPU rebuild of
  that arm: D-all max 0.930 -> BODY-STORED; e115 on that arm: brake +0.132
  (deleting the address FEEDS the field), band r=0.56 collapse; e083 on
  the B43 install line: cycle-END p(Z) 0.542/0.426/0.440 vs install 0.320,
  post-erase NLL 2.724->1.441->1.369, steps-to-bar 20/10/9.

COMPUTE ENVELOPE: the harness TinyGPT is 2.7M params (lab default for
this whole line; every e109/e113/e115 reference lives on it — the
registered program's own envelope names these checkpoints). Trainings on
GPU, EACH gated by gpu_ok() double-poll (e109 gate_launch verbatim) with
cooldown(120) between launches; every training <= 180 s (e109 wall cap /
e083 training-compute cap, verbatim); batch 32. No concurrent GPU jobs.
ALL battery evals CPU-side. If the GPU is busy/hot beyond the 20-min
bounded wait on an E-cycle launch, PARK politely.

Outputs: runs/e119/{metrics.json, migration_headtohead.png}; phase nets
runs/checkpoints/e119_<phase>.pt (R43 erratum convention — twin start,
every E cycle END, the calibrated R and its 300-step registered default
when calibration moves R, L; inventory in metrics.ckpt_inventory; per-cycle
end expression AND per-cycle D-outcomes surfaced in metrics.e_cycle_summary).
No NOTES/THINKING/QUEUE/STATE edits; commit by the experiment agent.

Run:  cd lab && python e119_migration_headtohead.py   (E119_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
import torch.nn.functional as F                       # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT, cooldown, gpu_ok,     # noqa: E402
                    gpu_status, run_dir, save_json)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)
import e083_cycle3 as E83                              # noqa: E402 (relearn / eval_bat / ce_val / fixed_blocks / mk_cpu_net / row_probe — road E verbatim)
import e109_consolidation as E109                      # noqa: E402 (gate_launch / finetune_arm / battery_cell / deleted_wpe / load_cpu / val_windows — road R verbatim)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E119_SMOKE") == "1"
CPU = torch.device("cpu")
DEV = "cuda"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
INSTALLED_CK = "e048_repro.pt"

JITTERS = (-8, -4, 0, 4, 8)       # e109's registered jitter set (road R)
NOVEL_GEO = (-12, -2, 2, 12)      # never trained by ANY arm (road E/L train g0 only)
GEO_ORDER = [-8, -4, 0, 4, 8]     # trained geometries, display order
DECISION_BAND = (121, 137)        # e109's grown-row band

# road R / L fine-tune envelope (e109 registered)
FT_LR = 1e-3
FT_STEPS = 300 if not SMOKE else 8
FT_TIME_CAP = 180.0
EVAL_EVERY = 25 if not SMOKE else 2
NAME_BS = 16
ANCH_BS = 16
R_SEED, L_SEED = 10901, 10902

# road E erase-cycle envelope (e083 registered)
N_CYCLES = 3 if not SMOKE else 1
E_STEPS = 300 if not SMOKE else 8
BAR_NLL, BAR_ACC = 1.0, 0.8        # e044b's tasking bar
DCE_GATE = 0.01
GEN_EXP = 24401                    # e044 paired-draw seed, ALL cycles
CE_SEED, N_CE_BLOCKS = 202, 30

# batteries / guards
R_EVAL_SEED = 26502
G_INST_REF = 0.556313
G_INST_TOL = 0.005
E109_REF_NONE = {-8: 0.9924036860466003, -4: 0.9496323466300964,
                 0: 0.776076078414917, 4: 0.9848979115486145,
                 8: 0.9854525923728943}
G_R_REPRO_TOL = 0.05

# gate + bars
GATE_BAND = 0.20                   # |p_R(g0) - p_E(g0)| after calibration
GROWN_DNORM = 0.04                 # census growth threshold (0.03/0.05 sensitivity)
BRAKE_FLOOR = 0.02                 # sign class floor for (c)
SHARE_BAR = 0.20                   # collapse boundary bar for r*(k)
SHARE_FACTOR = 2.0                 # product ratio counted as a boundary shift
KS = [32, 64, 128]                 # e110 k-ladder (band = 128 positions)
R_GRID = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.35, 0.2, 0.1]   # descending (display)
R_GRID_ASC = sorted(R_GRID)        # ascending — the r* scan order
N_SUBSET = 2
SUBSET_SEED = 119
FIELD_BAND = list(range(1, PRE - 1))          # positions 1..128 (e115 band)
BOOT_N = 1000

GPU_WAIT_MAX_S, GPU_POLL_S = 1200.0, 30.0

trims: list[str] = []
deviations: list[str] = [
    "Twin reading: the registered 'two same-seed installs' clause is used "
    "in its degenerate limit — both roads start from the ONE shared "
    "installed net (e048_repro.pt). Same-seed installs are bit-identical, "
    "so the shared start IS the twin pair; every arm difference is "
    "route-caused. (The alternative reading — a second install on another "
    "base — would ADD install noise between arms, not remove it.)",
    "Trainings on GPU (like the e109/e083 originals), gated by gpu_ok() "
    "double-poll with cooldown(120) between launches; every battery eval "
    "CPU-side on state-dict snapshots (e083 deviation-3 convention). E "
    "cycle launches use a 20-min-bounded wait then PARK; R/L launches use "
    "e109's own unbounded internal gate (verbatim).",
    "R's step count and E's cycle count are the pre-registered calibration "
    "dials for the expression-matching gate; R reruns are same-seed "
    "prefixes of the e109 recipe (the module's finetune_arm reads FT_STEPS "
    "at call time, so the dial is set via that module attribute; recipe, "
    "seed and batch composition verbatim; GPU float nondeterminism makes "
    "prefix reruns match the in-loop trajectory to device tolerance only).",
    "Share ladder (d): the e115 graded-V instrument crossed with the e110 "
    "k-subset ladder, read on the FACT's battery p(Z) (not e110's stream "
    "CE — that rig belongs to the 0.84M ctx-512 family; the r-grid is "
    "refined to 0.1 steps near 1.0 because e115 showed the boundary sits "
    "in (0.56, 1.0) on this net), primary read with the arm's FULL "
    "address set zeroed (e113's field-only convention), D129-only "
    "secondary at r=1. 2 subsets (not e110's 24) — the readout is the "
    "coarse r*(k) boundary, subset-paired across arms; the per-subset "
    "spread is the reported band.",
    "(e) V-geometry typing is report-only (never gates a bar); the "
    "registered SAME/DIFFERENT clauses name (b), (c), (d) and the held-30 "
    "example only.",
    "Smoke mode (E119_SMOKE=1) trims: 1 erase cycle, 8-step fine-tunes, "
    "novel geos {-2,+2}, share grid {1.0, 0.7, 0.4} x {32,128} x 1 "
    "subset; nothing adjudicated.",
]


# ------------------------------------------------------------------ gpu guard

CKPT_DIR = E43.REPO / "runs" / "checkpoints"      # gitignored, lab convention
CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    """Phase-net checkpoint (R43 erratum convention): every state this
    experiment trains is saved as runs/checkpoints/e119_<phase>.pt in the
    {'model': sd, 'meta': ...} house format (load_cpu-compatible). Smoke
    runs write smoke_-prefixed names so they never clobber real ckpts."""
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"e119_{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "e119", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)), **meta}
    log(f"[ckpt] saved {path.name} ({', '.join(f'{k}={v}' for k, v in meta.items())})")


def gate_launch(tag: str) -> None:
    """e109's gate_launch with e083's bounded wait; PARK on timeout."""
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
        if time.time() - t0 > GPU_WAIT_MAX_S:
            raise RuntimeError(
                f"PARK: GPU busy/hot for {GPU_WAIT_MAX_S:.0f}s "
                f"({gpu_status()}) — refusing to launch '{tag}'")
        time.sleep(GPU_POLL_S)


# ------------------------------------------------------------------ stats

def boot_mean_ci(d, n=BOOT_N, seed=1):
    """e115's bootstrap 95% CI of the mean (context resampling)."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    vals = [float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(n)]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# ------------------------------------------------------------------ field instrument

@torch.no_grad()
def field_forward(net: TinyGPT, ids: torch.Tensor, zid: int, mult=None,
                  wpe_zero_rows=(), bs=30, vstats=False):
    """Manual TinyGPT forward (e114/e115's instrument, generalized): p(Z) at
    the last position per context. `mult` = per-position multiplier on V
    (length T; every layer, every head, all queries; K untouched; caller
    guarantees mult=1 at row 0 / address coordinates). `wpe_zero_rows` =
    eval-time address zeroing (same arithmetic as deleted_wpe, in-forward).
    vstats collects the vector-space identity of the scaling (norm-ratio
    dev, min cos over scaled positions) on the first batch/layer."""
    net.eval()
    wpe_w = net.wpe.weight
    if wpe_zero_rows:
        w2 = wpe_w.clone()
        for r in wpe_zero_rows:
            w2[r] = 0.0
        wpe_w = w2
    outs = []
    stats = {"norm_ratio_dev": None, "min_cos": None}
    for i in range(0, ids.shape[0], bs):
        idxs = ids[i:i + bs]
        N, T = idxs.shape
        pos = torch.arange(T)
        x = net.wte(idxs) + wpe_w[pos].unsqueeze(0)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        for li, blk in enumerate(net.h):
            xh = blk.ln1(x)
            qkv = blk.attn.c_attn(xh)
            C = qkv.shape[-1] // 3
            d = C // net.cfg.n_head
            q, k, v = qkv.split(C, dim=2)
            q = q.view(N, T, net.cfg.n_head, d).transpose(1, 2)
            k = k.view(N, T, net.cfg.n_head, d).transpose(1, 2)
            v = v.view(N, T, net.cfg.n_head, d).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
            att = att.masked_fill(causal, float("-inf"))
            probs = torch.softmax(att, dim=-1)
            if mult is not None:
                if vstats and li == 0 and i == 0 and bool((mult != 1.0).any()):
                    old = v.clone()
                    v = v * mult.view(1, 1, T, 1)
                    sel = mult != 1.0
                    nrm = old[:, :, sel, :].norm(dim=-1)
                    nnew = v[:, :, sel, :].norm(dim=-1)
                    ratio = nnew / nrm.clamp_min(1e-12)
                    stats["norm_ratio_dev"] = float(
                        (ratio - mult[sel].view(1, 1, -1)).abs().max())
                    cos = (old[:, :, sel, :] * v[:, :, sel, :]).sum(-1) / \
                        (nrm * nnew).clamp_min(1e-12)
                    stats["min_cos"] = float(cos.min())
                else:
                    v = v * mult.view(1, 1, T, 1)
            y = (probs @ v).transpose(1, 2).reshape(N, T, C)
            x = x + blk.attn.c_proj(y)
            x = x + blk.mlp(blk.ln2(x))
        pr = F.softmax(net.lm_head(net.ln_f(x[:, -1, :])), -1)[:, zid]
        outs.append(pr)
    p = torch.cat(outs)
    return (p, stats) if vstats else (p, None)


@torch.no_grad()
def capture_band_v(net: TinyGPT, ids: torch.Tensor, band, bs=30):
    """Per-layer value vectors at `band` positions, list over layers of
    (N, H, len(band), d) — the (e) V-geometry probe (e108 lineage)."""
    net.eval()
    L = net.cfg.n_layer
    band_t = torch.as_tensor(band, dtype=torch.long)
    out = [[] for _ in range(L)]
    for i in range(0, ids.shape[0], bs):
        idxs = ids[i:i + bs]
        N, T = idxs.shape
        pos = torch.arange(T)
        x = net.wte(idxs) + net.wpe.weight[pos].unsqueeze(0)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        for li, blk in enumerate(net.h):
            xh = blk.ln1(x)
            qkv = blk.attn.c_attn(xh)
            C = qkv.shape[-1] // 3
            d = C // net.cfg.n_head
            q, k, v = qkv.split(C, dim=2)
            q = q.view(N, T, net.cfg.n_head, d).transpose(1, 2)
            k = k.view(N, T, net.cfg.n_head, d).transpose(1, 2)
            v = v.view(N, T, net.cfg.n_head, d).transpose(1, 2)
            out[li].append(v[:, :, band_t, :])
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
            att = att.masked_fill(causal, float("-inf"))
            probs = torch.softmax(att, dim=-1)
            y = (probs @ v).transpose(1, 2).reshape(N, T, C)
            x = x + blk.attn.c_proj(y)
            x = x + blk.mlp(blk.ln2(x))
    return [torch.cat(chunks, 0) for chunks in out]


def vgeo_pair(dvx: list, dvy: list) -> dict:
    """mean |cos| between two arms' dV at matching (layer, head, ctx, pos),
    plus nulls: cyclic position roll + seeded permutation within (l,h,ctx)."""
    cs_same, cs_roll, cs_perm = [], [], []
    rng = np.random.default_rng(1190)
    for ax, ay in zip(dvx, dvy):                          # (N,H,P,d)
        a = F.normalize(ax, dim=-1)
        b = F.normalize(ay, dim=-1)
        cs_same.append((a * b).sum(-1))
        b_roll = torch.roll(b, shifts=1, dims=2)          # derange positions
        cs_roll.append((a * b_roll).sum(-1))
        P = ax.shape[2]
        pm = torch.as_tensor(rng.permutation(P), dtype=torch.long)
        cs_perm.append((a * b[:, :, pm, :]).sum(-1))
    same = torch.cat([c.flatten() for c in cs_same])
    roll = torch.cat([c.flatten() for c in cs_roll])
    perm = torch.cat([c.flatten() for c in cs_perm])
    return {"mean_abs_cos": float(same.abs().mean()),
            "null_roll": float(roll.abs().mean()),
            "null_perm": float(perm.abs().mean()),
            "n_cosines": int(same.numel())}


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e119_smoke" if SMOKE else "e119")
    torch.set_num_threads(min(16, os.cpu_count() or 8))
    log(f"E119 MIGRATION HEAD-TO-HEAD (P5 road R vs road E; smoke={SMOKE}) -> {rd}")

    novel_geo = NOVEL_GEO
    r_grid, ks, n_sub = R_GRID, KS, N_SUBSET
    if SMOKE:
        novel_geo = (-2, 2)
        r_grid, ks, n_sub = [1.0, 0.7, 0.4], [32, 128], 1
        E83.TRAIN_CAP_S = 60.0
        E83.EVAL_STEPS = list(range(1, 9))
        E83.EVAL_SET = set(E83.EVAL_STEPS)
        E83.SPARSE_AT = {4, 8}
    geo_all = sorted(set(GEO_ORDER) | set(novel_geo))

    # ---------------- protocol rebuild (e065/e091/e109/e113 verbatim)
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
    name_ids = torch.tensor([stoi[c] for c in NAME], dtype=torch.long)
    L = len(NAME)

    # ---------------- training pools + batteries
    jit_x, jit_mask = {}, {}
    for j in JITTERS:
        wins = []
        for p, h in install_occ:
            pre = train_ids[p - PRE - j: p]
            post = train_ids[p + len(h): p + len(h) + POST_CAP - j]
            w = torch.cat([pre, name_ids, post])
            if len(w) != BLOCK:
                raise RuntimeError(f"window len {len(w)} != {BLOCK} at offset {j}")
            wins.append(w)
        jit_x[j] = torch.stack(wins)
        m = torch.zeros(len(wins), BLOCK - 1, dtype=torch.bool)
        m[:, PRE - 1 + j: PRE - 1 + j + L] = True
        jit_mask[j] = m
    pool_r_x = torch.cat([jit_x[j] for j in JITTERS])       # (300, 256)
    pool_r_mask = torch.cat([jit_mask[j] for j in JITTERS])
    pool_l_x, pool_l_mask = jit_x[0], jit_mask[0]
    anchor16 = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                            for p, _ in install_occ[:16]])  # e109 anchor bank
    anchor60 = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                            for p, _ in install_occ])       # e083 anchor bank

    # batteries: ctx = train_text[p-PRE-j : p], readout p(Z) at last position
    bat_ids = {}
    for j in geo_all:
        for tag, occ in (("install60", install_occ), ("held30", held_occ)):
            cs = [train_text[p - PRE - j: p] for p, _ in occ]
            bat_ids[(j, tag)] = torch.stack([corpus.encode(c) for c in cs])
    f_eval_ids = bat_ids[(0, "install60")]                   # the G_INST battery

    # road E instruments (e083 verbatim): spliced battery + CE bank
    bat_i = jit_x[0][:, :PRE + L]                            # (60, 137)
    vx, vy = E83.fixed_blocks(val_ids, BLOCK, N_CE_BLOCKS, CE_SEED)
    r_eval_x, r_eval_y = E109.val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

    # ---------------- shared twin start + instrument gate
    net0 = E109.load_cpu(E43.REPO / "runs" / "checkpoints" / INSTALLED_CK)
    sd_base = {k: v.clone() for k, v in net0.state_dict().items()}
    evl = copy.deepcopy(net0)
    bz0 = E109.battery_cell(evl, f_eval_ids, zid)
    G_INST = {"battery_pz": bz0["mean_pz"], "ref": G_INST_REF, "tol": G_INST_TOL,
              "pass": bool(abs(bz0["mean_pz"] - G_INST_REF) < G_INST_TOL)}
    log(f"G_INST installed battery p(Z) {bz0['mean_pz']:.6f} (ref {G_INST_REF}): "
        f"{'PASS' if G_INST['pass'] else 'FAIL'}")
    if not G_INST["pass"]:
        raise RuntimeError("instrument broken vs e065/e091/e109/e113 (net/battery mismatch)")
    save_ckpt("twin_start", sd_base,
              {"desc": "shared bit-identical install (= runs/checkpoints/"
                       "e048_repro.pt, seed-42 line) — the twin start for "
                       "all arms", "pz_g0": bz0["mean_pz"]})

    # ---------------- ARM E: three erase cycles (e083 verbatim, GPU-gated)
    log(f"ARM E: {N_CYCLES} erase cycles (e083 protocol verbatim on the twin start)")
    orig_w = sd_base["wte.weight"][zid].clone()
    orig_l = sd_base["lm_head.weight"][zid].clone()
    orig_wpe129 = sd_base["wpe.weight"][PRE - 1].clone()
    cycles, e_ends, e_end_pz = [], {}, {}
    cur_sd = {k: v.clone() for k, v in sd_base.items()}
    ce_pre = E83.ce_val(E83.mk_cpu_net(cur_sd), vx, vy)
    for c in range(1, N_CYCLES + 1):
        if c > 1:
            log("[thermal] cooldown(120) between training launches")
            cooldown(120.0)
        gate_launch(f"armE_cycle{c}")
        # ---- ERASE (e044b D2 row-reset verbatim + gates)
        er_sd = {k: v.clone() for k, v in cur_sd.items()}
        er_sd["wte.weight"][zid] = 0.0
        er_sd["lm_head.weight"][zid] = 0.0
        n_diff, confined = 0, True
        for key in ("wte.weight", "lm_head.weight"):
            d = er_sd[key] != cur_sd[key]
            nd = int(d.sum().item())
            n_diff += nd
            rows = torch.nonzero(d)[:, 0].unique()
            if not (len(rows) == 1 and int(rows[0]) == zid and nd == Cfg().n_embd):
                confined = False
        others = all(torch.equal(er_sd[k], cur_sd[k]) for k in er_sd
                     if k not in ("wte.weight", "lm_head.weight"))
        netE = E83.mk_cpu_net(er_sd)
        r_er = E83.eval_bat(netE, bat_i, L, PRE - 1)
        ce_er = E83.ce_val(netE, vx, vy)
        G2 = {"n_elements_changed": n_diff, "expected": 2 * Cfg().n_embd,
              "confined_to_Z_rows": bool(confined),
              "others_bit_identical": bool(others), "dce": ce_er - ce_pre,
              "dce_gate": DCE_GATE,
              "pass": bool(n_diff == 2 * Cfg().n_embd and confined and others
                           and abs(ce_er - ce_pre) <= DCE_GATE)}
        G3 = {"nll": r_er["nll"], "acc": r_er["acc"],
              "off_bar": bool(r_er["nll"] > BAR_NLL or r_er["acc"] < BAR_ACC),
              "note": "e083 re-based strictly-off-bar gate (install line)",
              "pass": bool(r_er["nll"] > BAR_NLL or r_er["acc"] < BAR_ACC)}
        log(f"  cycle {c} erase: {n_diff} elems, dCE {ce_er - ce_pre:+.5f}, "
            f"spliced {r_er['nll']:.3f}/{r_er['acc']:.3f} -> G2 {G2['pass']} "
            f"G3 {G3['pass']}")
        assert G2["pass"], f"erase confinement FAILED: {G2}"
        if not G3["pass"]:
            log(f"  cycle {c} ERASE VACUOUS (state still AT bar after D2) — "
                f"lesion-tolerant tasking; steps_to_bar=0, no re-learn")
            cycles.append({"cycle": c, "erase": {"G2": G2, "G3": G3,
                                                 "post_erase_spliced": r_er},
                           "relearn": None, "erase_vacuous": True,
                           "steps_to_bar": 0, "traj": []})
            continue

        # ---- RE-LEARN (e083 relearn verbatim; on_eval adds p(Z)@g0 sparse)
        net = TinyGPT(Cfg()).to(DEV)
        net.load_state_dict({k: v.to(DEV) for k, v in er_sd.items()})

        def on_eval(sd_cpu, step, c=c):
            rec = {"step": step,
                   "spliced_i": E83.eval_bat(E83.mk_cpu_net(sd_cpu), bat_i, L, PRE - 1),
                   "rows": E83.row_probe(sd_cpu, zid, orig_w, orig_l, orig_wpe129)}
            if step in E83.SPARSE_AT:
                m_ = E83.mk_cpu_net(sd_cpu)
                rec["ce"] = E83.ce_val(m_, vx, vy)
                rec["pz_g0"] = E109.battery_cell(m_, f_eval_ids, zid)["mean_pz"]
            sp = rec["spliced_i"]
            rec["bar"] = bool(sp["nll"] <= BAR_NLL and sp["acc"] >= BAR_ACC)
            log(f"  [c{c} s{step:3d}] spliced {sp['nll']:6.3f}/{sp['acc']:.3f}"
                + (f" ce {rec['ce']:.4f} pz_g0 {rec['pz_g0']:.3f}"
                   if "pz_g0" in rec else "")
                + f" |wte| {rec['rows']['wte_norm']:.3f}"
                  f" cos {rec['rows']['cos_wte']:+.3f}"
                + ("  BAR" if rec["bar"] else ""))
            return rec

        gen = torch.Generator().manual_seed(GEN_EXP)   # paired draws, all cycles
        run = E83.relearn(net, DEV, jit_x[0].to(DEV), jit_mask[0].to(DEV),
                          anchor60.to(DEV), train_ids, steps=E_STEPS,
                          gen=gen, log=log, on_eval=on_eval)
        final_sd = run["final_sd"] if run["final_sd"] is not None else \
            {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pz_end = E109.battery_cell(E83.mk_cpu_net(final_sd), f_eval_ids,
                                   zid)["mean_pz"]
        ce_end = E83.ce_val(E83.mk_cpu_net(final_sd), vx, vy)
        # per-cycle D-outcomes at g0 (surfaced for EVERY cycle END, not just
        # the chosen one — W010 falsifier-(d) wants regrowth tracked against
        # residual overlap mass cycle-over-cycle)
        w_end = final_sd["wpe.weight"]
        dn_end = (w_end.norm(dim=1) - sd_base["wpe.weight"].norm(dim=1))
        grown_end = sorted(int(r) for r in range(1, w_end.shape[0])
                           if r != PRE - 1 and float(dn_end[r]) >= GROWN_DNORM)
        dall_end = tuple(sorted(set(grown_end) | {PRE - 1}))
        net_end = E83.mk_cpu_net(final_sd)
        del_g0 = {"grown_rows": grown_end, "dall": list(dall_end), "cells": {}}
        for dl, rows_ in (("none", ()), ("d129", (PRE - 1,)), ("dall", dall_end)):
            sd_x, _gate_x = E109.deleted_wpe(final_sd, rows_)
            net_end.load_state_dict(sd_x)
            for bt in ("install60", "held30"):
                del_g0["cells"][f"{dl}__{bt}"] = E109.battery_cell(
                    net_end, bat_ids[(0, bt)], zid)["mean_pz"]
        save_ckpt(f"e_erase_c{c}_end", final_sd,
                  {"cycle": c, "end_pz_g0": pz_end,
                   "steps_to_bar": run["bar_step"],
                   "grown_rows_end": grown_end})
        cycles.append({"cycle": c, "erase": {"G2": G2, "G3": G3,
                                             "post_erase_spliced": r_er,
                                             "post_erase_ce": ce_er},
                       "relearn": {k: v for k, v in run.items()
                                   if k not in ("final_sd", "at_bar_sd")},
                       "erase_vacuous": False,
                       "steps_to_bar": run["bar_step"],
                       "traj": run["traj"], "end_pz_g0": pz_end,
                       "end_ce": ce_end, "end_deletions_g0": del_g0})
        e_ends[c] = final_sd
        e_end_pz[c] = pz_end
        cur_sd = {k: v.clone() for k, v in final_sd.items()}
        ce_pre = ce_end
        log(f"  cycle {c} END: pz_g0 {pz_end:.4f}, bar at "
            f"{run['bar_step']}, train {run['train_s']}s")

    # ---------------- ARM R: jittered replay (e109 verbatim, GPU-gated)
    log("[thermal] cooldown(120) between training launches")
    cooldown(120.0)
    log(f"ARM R: jittered replay {list(JITTERS)} (e109 arm a verbatim, "
        f"seed {R_SEED}, {FT_STEPS} steps default)")
    E109.FT_STEPS = FT_STEPS
    E109.EVAL_EVERY = EVAL_EVERY
    r_res = E109.finetune_arm("r_jittered", net0, pool_r_x, pool_r_mask,
                              anchor16, train_ids, r_eval_xy, f_eval_ids,
                              zid, R_SEED)
    r_traj = r_res["traj"]

    def pz_at_step(traj, step):
        for t in traj:
            if t["step"] == step:
                return t["p_z_mean"]
        return None

    # ---------------- EXPRESSION-MATCHING GATE + calibration (pre-registered)
    p_r300 = pz_at_step(r_traj, r_res["steps_ran"])
    e_choices = {c: p for c, p in e_end_pz.items()}
    log(f"GATE inputs: R@{r_res['steps_ran']} pz_g0 {p_r300:.4f} | "
        f"E cycle-ENDs " + " ".join(f"c{c} {p:.4f}" for c, p in e_choices.items()))
    r_steps_grid = sorted({t["step"] for t in r_traj})
    cands = []
    for rc, pe in e_choices.items():
        for rs in r_steps_grid:
            pv = pz_at_step(r_traj, rs)
            if pv is not None:
                cands.append({"r_steps": rs, "e_cycles": rc, "p_r": pv,
                              "p_e": pe, "gap": abs(pv - pe)})
    if e_choices and N_CYCLES in e_choices:
        default = {"r_steps": r_res["steps_ran"], "e_cycles": N_CYCLES,
                   "p_r": p_r300, "p_e": e_choices[N_CYCLES],
                   "gap": abs(p_r300 - e_choices[N_CYCLES])}
    elif cands:
        default = dict(min(cands, key=lambda d: d["gap"]))
    else:
        default = {"r_steps": r_res["steps_ran"], "e_cycles": N_CYCLES,
                   "p_r": p_r300, "p_e": None, "gap": float("inf")}
    if SMOKE or default["gap"] <= GATE_BAND:
        chosen = default
    else:
        chosen = min(cands, key=lambda d: d["gap"]) if cands else default
    calibration = {"dial_default": default, "candidates": cands,
                   "chosen": chosen, "band": GATE_BAND,
                   "freedom": "R steps on its 25-step eval grid (deterministic "
                              "same-seed reruns) x E cycle count in "
                              "{1,2,3} (cycle END states kept)"}
    gate_pass = bool(chosen["gap"] <= GATE_BAND)
    calibration["pass"] = gate_pass
    log(f"GATE: chosen R@{chosen['r_steps']} pz {chosen['p_r']:.4f} vs "
        f"E@c{chosen['e_cycles']} pz {chosen['p_e'] if chosen['p_e'] is not None else float('nan'):.4f} "
        f"(gap {chosen['gap']:.4f} <= {GATE_BAND}): "
        f"{'PASS' if gate_pass else 'FAIL -> PARK'}")

    # R at calibrated steps (deterministic prefix rerun if needed)
    sd_r = r_res["sd"]
    r_steps_ran = r_res["steps_ran"]
    if chosen["r_steps"] != r_steps_ran and not SMOKE:
        log(f"[calibration] rerunning R deterministically to step "
            f"{chosen['r_steps']} (seed {R_SEED} prefix)")
        sd_r_default = sd_r                       # the registered 300-step net
        E109.FT_STEPS = chosen["r_steps"]
        r_res = E109.finetune_arm("r_jittered_cal", net0, pool_r_x,
                                  pool_r_mask, anchor16, train_ids,
                                  r_eval_xy, f_eval_ids, zid, R_SEED)
        sd_r = r_res["sd"]
        save_ckpt(f"r_jittered_s{chosen['r_steps']}", sd_r,
                  {"steps": chosen["r_steps"], "seed": R_SEED,
                   "pz_g0": chosen["p_r"], "role": "calibrated road-R arm"})
        save_ckpt(f"r_jittered_s{r_steps_ran}", sd_r_default,
                  {"steps": r_steps_ran, "seed": R_SEED,
                   "role": "registered default run (calibration moved R)"})
    else:
        save_ckpt(f"r_jittered_s{chosen['r_steps']}", sd_r,
                  {"steps": chosen["r_steps"], "seed": R_SEED,
                   "pz_g0": chosen["p_r"], "role": "road-R arm"})
    sd_e = e_ends.get(chosen["e_cycles"], cur_sd)

    # ---------------- ARM L: matched-mass locked replay (free rider)
    log("[thermal] cooldown(120) between training launches")
    cooldown(120.0)
    l_steps = chosen["r_steps"]
    log(f"ARM L: locked replay (e109 arm b verbatim, seed {L_SEED}, "
        f"{l_steps} steps — matched mass to R)")
    E109.FT_STEPS = l_steps
    l_res = E109.finetune_arm("l_locked", net0, pool_l_x, pool_l_mask,
                              anchor16, train_ids, r_eval_xy, f_eval_ids,
                              zid, L_SEED)
    sd_l = l_res["sd"]
    p_l = E109.battery_cell(E83.mk_cpu_net(sd_l), f_eval_ids, zid)["mean_pz"]
    save_ckpt(f"l_locked_s{l_steps}", sd_l,
              {"steps": l_steps, "seed": L_SEED, "pz_g0": p_l,
               "role": "locked-replay free-rider arm (matched mass to R)"})

    # explicit per-cycle surfacing (coordinator ask; W010 falsifier-(d))
    e_cycle_summary = []
    for cyc in cycles:
        tr = cyc.get("traj") or []
        e_cycle_summary.append({
            "cycle": cyc["cycle"], "erase_vacuous": cyc["erase_vacuous"],
            "steps_to_bar": cyc["steps_to_bar"],
            "post_erase_spliced": cyc["erase"].get("post_erase_spliced"),
            "end_pz_g0": cyc.get("end_pz_g0"), "end_ce": cyc.get("end_ce"),
            "end_rows": (tr[-1].get("rows") if tr else None),
            "end_deletions_g0": cyc.get("end_deletions_g0")})

    arms_sd = {"r_jittered": sd_r, "e_erase": sd_e, "l_locked": sd_l}
    arm_order = ["r_jittered", "e_erase", "l_locked"]
    short_lbl = {"r_jittered": "R (jittered)", "e_erase": "E (erase)",
                 "l_locked": "L (locked)"}
    arms_meta = {
        "r_jittered": {"desc": f"jittered replay {list(JITTERS)}, seed {R_SEED}, "
                               f"{chosen['r_steps']} steps (calibrated)",
                       "traj": r_res["traj"], "seed": R_SEED},
        "e_erase": {"desc": f"{chosen['e_cycles']} erase cycle(s) (e083 verbatim: "
                            f"D2 Z-row reset + Dmix 300-step battery, GEN {GEN_EXP})",
                    "cycles": cycles, "end_pz_g0": chosen["p_e"]},
        "l_locked": {"desc": f"locked replay offset-0 only, seed {L_SEED}, "
                             f"{l_steps} steps (matched mass to R)",
                     "traj": l_res["traj"], "seed": L_SEED, "pz_g0": p_l},
    }

    # ---------------- PARK branch (gate failed: no battery, no adjudication)
    if not gate_pass and not SMOKE:
        metrics = {
            "experiment": "e119_migration_headtohead", "date": common.now_iso(),
            "smoke": SMOKE, "verdict": "PARKED",
            "park_reason": f"expression-matching gate unachievable within the "
                           f"pre-registered freedom (best gap "
                           f"{chosen['gap']:.4f} > {GATE_BAND}) — protocol "
                           f"fragility on the twin install; no ad-hoc tuning",
            "gates": {"G_SPLICE": G_SPLICE, "G_INST": G_INST,
                      "G_EXP": calibration},
            "arms": arms_meta,
            "ckpt_inventory": CKPT_INVENTORY,
            "e_cycle_summary": e_cycle_summary,
            "deviations": deviations, "trims": trims,
            "timing": {"total_s": round(time.time() - T0, 1)},
            "config": {"params": int(net0.num_params())},
        }
        save_json(rd / "metrics.json", E43.jsonable(metrics))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.axis("off")
        ax.text(0.5, 0.6, "PARKED — expression gate unmet", ha="center",
                fontsize=16, family="monospace")
        ax.text(0.5, 0.45,
                f"R pz {chosen['p_r']:.3f} vs E pz "
                f"{chosen['p_e'] if chosen['p_e'] is not None else float('nan'):.3f} "
                f"(gap {chosen['gap']:.3f} > {GATE_BAND})",
                ha="center", fontsize=11, family="monospace")
        fig.savefig(rd / "migration_headtohead.png", dpi=130)
        plt.close(fig)
        log("PARKED: metrics + png written; no comparative battery run")
        return 2

    # ================= COMPARATIVE BATTERY (eval-only, all CPU) ==============
    log("=" * 78)
    log("COMPARATIVE BATTERY (CPU): (a) deletion hierarchy x geometries, "
        "(b) census, (c) brake, (d) share, (e) V-geometry")

    # ---------------- (b) grown-row census
    census = {}
    for an in arm_order:
        w = arms_sd[an]["wpe.weight"]
        dn = (w.norm(dim=1) - sd_base["wpe.weight"].norm(dim=1))
        grown = sorted(int(r) for r in range(1, w.shape[0])
                       if r != PRE - 1 and float(dn[r]) >= GROWN_DNORM)
        rows = {}
        for r in grown:
            rows[str(r)] = {
                "dnorm": float(dn[r]), "norm": float(w[r].norm()),
                "base_norm": float(sd_base["wpe.weight"][r].norm()),
                "cos_to_base_row": float(F.cosine_similarity(
                    w[r], sd_base["wpe.weight"][r], dim=0)),
                "cos_to_address_dir": float(F.cosine_similarity(
                    w[r], orig_wpe129, dim=0))}
        top = sorted(((float(dn[r]), r) for r in range(w.shape[0])
                      if r != PRE - 1), reverse=True)[:10]
        census[an] = {"grown_rows": grown, "grown_detail": rows,
                      "dnorm_spectrum": [float(v) for v in dn.tolist()],
                      "top10_dnorm": [{"row": int(r), "dnorm": float(v)}
                                      for v, r in top],
                      "sensitivity": {
                          f"thr_{t:g}": sorted(int(r) for r in range(1, w.shape[0])
                                               if r != PRE - 1
                                               and float(dn[r]) >= t)
                          for t in (0.03, 0.04, 0.05)}}
        log(f"census {an:12s}: grown rows {grown} "
            f"(top dnorm " + ", ".join(f"r{r} {v:+.3f}" for v, r in top[:5])
            + ")")
    r_grown, e_grown = census["r_jittered"]["grown_rows"], census["e_erase"]["grown_rows"]

    b_overlap = False
    if not r_grown and not e_grown:
        b_overlap = True                       # both empty: no growth anywhere
    elif r_grown and e_grown:
        br_, be_ = (min(r_grown), max(r_grown)), (min(e_grown), max(e_grown))
        b_overlap = bool(not (br_[1] < be_[0] or be_[1] < br_[0])
                         or (set(r_grown) & set(e_grown)))
    one_sided_growth = bool((r_grown and not e_grown)
                            or (e_grown and not r_grown))
    in_dec_band = {an: [r for r in census[an]["grown_rows"]
                        if DECISION_BAND[0] <= r <= DECISION_BAND[1]]
                   for an in arm_order}

    # ---------------- (a) deletion hierarchy x geometries
    deletions = {}
    for an in arm_order:
        g = census[an]["grown_rows"]
        deletions[an] = {"none": (), "d129": (PRE - 1,),
                         "dgrown": tuple(r for r in g if r != PRE - 1),
                         "dall": tuple(sorted(set(g) | {PRE - 1}))}
    table, gates_surg, per_ctx = {}, {}, {}
    evl = copy.deepcopy(net0)
    for an in arm_order:
        for dl_name in ("none", "d129", "dgrown", "dall"):
            rows = deletions[an][dl_name]
            if dl_name != "none" and not rows:
                # degenerate (no grown rows): D-grown is a no-op, D-all = D129
                for j in geo_all:
                    for bt in ("install60", "held30"):
                        table[(an, dl_name, j, bt)] = table[(an, "none", j, bt)] \
                            if dl_name == "dgrown" else table[(an, "d129", j, bt)]
                per_ctx[f"{an}__{dl_name}"] = per_ctx[
                    f"{an}__{'none' if dl_name == 'dgrown' else 'd129'}"]
                gates_surg[f"{an}__{dl_name}"] = {"degenerate_noop": True,
                                                  "pass": True}
                continue
            sd_del, gate = E109.deleted_wpe(arms_sd[an], rows)
            gates_surg[f"{an}__{dl_name}"] = gate
            if not gate["pass"]:
                raise RuntimeError(f"deletion gate FAILED {an}/{dl_name}: {gate}")
            evl.load_state_dict(sd_del)
            for j in geo_all:
                for bt in ("install60", "held30"):
                    cell = E109.battery_cell(evl, bat_ids[(j, bt)], zid,
                                             keep_per_ctx=(bt == "install60"))
                    table[(an, dl_name, j, bt)] = cell
                    if bt == "install60" and j == 0:
                        per_ctx[f"{an}__{dl_name}"] = cell.get("pz_per_ctx")
        log(f"(a) {an:12s}: " + " | ".join(
            f"{dl} g0 {table[(an, dl, 0, 'install60')]['mean_pz']:.3f}"
            for dl in ("none", "d129", "dgrown", "dall")))

    def post(an, dl, j, bt="install60"):
        return table[(an, dl, j, bt)]["mean_pz"]

    # ---------------- (c) brake sign at full field
    brake = {}
    for an in arm_order:
        pa = np.asarray(per_ctx[f"{an}__none"])
        pn = np.asarray(per_ctx[f"{an}__d129"])
        br = pn - pa
        ci = boot_mean_ci(br, seed=1)
        cls = ("no_brake" if abs(br.mean()) < BRAKE_FLOOR
               else ("feeds" if br.mean() > 0 else "suppresses"))
        brake[an] = {"brake_mean": float(br.mean()), "brake_ci95": list(ci),
                     "class": cls, "floor": BRAKE_FLOOR,
                     "held30_mirror": post(an, "d129", 0, "held30")
                     - post(an, "none", 0, "held30")}
        log(f"(c) brake {an:12s}: {br.mean():+.4f} CI [{ci[0]:+.3f},{ci[1]:+.3f}] "
            f"-> {cls} (held30 {brake[an]['held30_mirror']:+.3f})")
    c_match = brake["r_jittered"]["class"] == brake["e_erase"]["class"]
    c_dissoc = (brake["r_jittered"]["class"] in ("feeds", "suppresses")
                and brake["e_erase"]["class"] in ("feeds", "suppresses")
                and brake["r_jittered"]["class"] != brake["e_erase"]["class"])

    # ---------------- (d) share r*(k)*k at the collapse boundary
    log(f"(d) share ladder: k in {ks}, r-grid {r_grid}, {n_sub} subset(s), "
        f"primary = FULL address set zeroed (e113 convention)")
    sub_rng = np.random.default_rng(SUBSET_SEED)
    subsets = {k: [sorted(int(x) for x in sub_rng.choice(FIELD_BAND, size=k,
                                                         replace=False))
                   for _ in range(n_sub)] for k in ks}
    evl_r = E83.mk_cpu_net(arms_sd["r_jittered"])
    evl_e = E83.mk_cpu_net(arms_sd["e_erase"])
    evl_l = E83.mk_cpu_net(arms_sd["l_locked"])

    # instrument gates: net()-agreement + identity + vector-space scaling check
    p_man, _ = field_forward(evl_r, f_eval_ids, zid)
    p_net = []
    with torch.no_grad():
        for i in range(0, f_eval_ids.shape[0], 30):
            lg, _ = evl_r(f_eval_ids[i:i + 30])
            p_net.append(F.softmax(lg[:, -1], -1)[:, zid])
    g_fwd = float((p_man - torch.cat(p_net)).abs().max())
    mult_id = torch.ones(f_eval_ids.shape[1])
    p_id, _ = field_forward(evl_r, f_eval_ids, zid, mult=mult_id)
    g_id = float((p_id - p_man).abs().max())
    mult_vs = torch.ones(f_eval_ids.shape[1])
    mult_vs[FIELD_BAND] = 0.9
    _, vst = field_forward(evl_r, f_eval_ids, zid, mult=mult_vs, vstats=True)
    G_FIELD = {"fwd_max_dev": g_fwd, "identity_max_dev": g_id,
               "vscale_norm_ratio_dev": vst["norm_ratio_dev"],
               "vscale_min_cos": vst["min_cos"],
               "tol": 1e-4, "vs_tol": 1e-6,
               "pass": bool(g_fwd < 1e-4 and g_id < 1e-4
                            and vst["norm_ratio_dev"] < 1e-6
                            and vst["min_cos"] > 1.0 - 1e-6)}
    log(f"G_FIELD manual-forward: dev {g_fwd:.2e}, identity {g_id:.2e}, "
        f"norm-ratio dev {vst['norm_ratio_dev']:.2e}, min cos "
        f"{vst['min_cos']:.9f} -> {'PASS' if G_FIELD['pass'] else 'FAIL'}")
    if not G_FIELD["pass"]:
        raise RuntimeError("field instrument broken vs TinyGPT net()")

    share = {}
    for an, net_ in (("r_jittered", evl_r), ("e_erase", evl_e), ("l_locked", evl_l)):
        addr_full = deletions[an]["dall"]
        share[an] = {"addr_set_full": list(addr_full), "grid": {},
                     "n_subsets_used": (n_sub if an != "l_locked" else 1)}
        T = f_eval_ids.shape[1]
        for k in ks:
            for si, sub in enumerate(subsets[k]):
                if an == "l_locked" and si > 0:
                    continue                      # free rider: 1 subset
                sub_set = set(sub)
                comp = [p_ for p_ in FIELD_BAND if p_ not in sub_set]
                for r in r_grid:
                    mult = torch.ones(T)
                    mult[torch.as_tensor(sub, dtype=torch.long)] = r
                    mult[torch.as_tensor(comp, dtype=torch.long)] = 0.0
                    p, _ = field_forward(net_, f_eval_ids, zid, mult=mult,
                                         wpe_zero_rows=addr_full)
                    share[an]["grid"][f"k{k}_s{si}_r{r:g}"] = float(p.mean())
        # secondary: D129-only at r=1 (full band retained)
        mult = torch.ones(T)
        p, _ = field_forward(net_, f_eval_ids, zid, mult=mult,
                             wpe_zero_rows=(PRE - 1,))
        share[an]["secondary_d129_only_r1"] = float(p.mean())
        log(f"(d) {an:12s} ladder done (full-addr r=1 k={ks[-1]}: "
            f"{share[an]['grid'].get(f'k{ks[-1]}_s0_r1', float('nan')):.3f}, "
            f"D129-only r=1: {share[an]['secondary_d129_only_r1']:.3f})")

    grid_asc = sorted(r_grid, reverse=False)          # ascending retention

    def rstar(an, k, si=None):
        """min r (ascending scan) with mean-over-subsets p >= bar; None=censored."""
        sis = [si] if si is not None else range(share[an]["n_subsets_used"])
        for r in grid_asc:
            vals = [share[an]["grid"].get(f"k{k}_s{s}_r{r:g}") for s in sis]
            vals = [v for v in vals if v is not None]
            if vals and float(np.mean(vals)) >= SHARE_BAR:
                return r
        return None

    share_summary = {}
    for an in arm_order:
        share_summary[an] = {}
        for k in ks:
            rs = rstar(an, k)
            share_summary[an][f"k{k}"] = {
                "r_star": rs,
                "product": (rs * k if rs is not None else None),
                "censored_offgrid_high": rs is None,
                "per_subset_r_star": [rstar(an, k, si=s)
                                      for s in range(share[an]["n_subsets_used"])]}
        log(f"(d) {an:12s}: " + " | ".join(
            f"k{k} r* {share_summary[an][f'k{k}']['r_star']} "
            f"prod {share_summary[an][f'k{k}']['product']}" for k in ks))

    d_ongrid = [k for k in ks
                if share_summary["r_jittered"][f"k{k}"]["r_star"] is not None
                and share_summary["e_erase"][f"k{k}"]["r_star"] is not None]
    d_ratios = []
    for k in d_ongrid:
        pr = share_summary["r_jittered"][f"k{k}"]["product"]
        pe = share_summary["e_erase"][f"k{k}"]["product"]
        d_ratios.append(max(pr, pe) / max(min(pr, pe), 1e-9))
    n_offgap = sum(
        1 for k in ks
        if (share_summary["r_jittered"][f"k{k}"]["r_star"] is None)
        != (share_summary["e_erase"][f"k{k}"]["r_star"] is None))
    d_match = bool(len(d_ongrid) >= 2
                   and sum(1 for x in d_ratios if x < SHARE_FACTOR) >= 2
                   and n_offgap == 0)
    d_dissoc = bool((len(d_ongrid) >= 2
                     and sum(1 for x in d_ratios if x >= SHARE_FACTOR) >= 2)
                    or n_offgap >= 2)

    # held-30 rider under D-all
    h30 = {an: {"dall_held30_max": max(post(an, "dall", j, "held30")
                                       for j in geo_all),
                "dall_held30_min": min(post(an, "dall", j, "held30")
                                       for j in geo_all)} for an in arm_order}
    h30_rider_dissoc = bool(
        (h30["r_jittered"]["dall_held30_max"] >= SHARE_BAR
         and h30["e_erase"]["dall_held30_max"] <= 0.05)
        or (h30["e_erase"]["dall_held30_max"] >= SHARE_BAR
            and h30["r_jittered"]["dall_held30_max"] <= 0.05))
    log(f"held-30 rider under D-all: R max {h30['r_jittered']['dall_held30_max']:.3f} "
        f"vs E max {h30['e_erase']['dall_held30_max']:.3f} "
        f"(dissociates={h30_rider_dissoc})")

    # ---------------- (e) V-geometry typing (report-only)
    log("(e) V-geometry typing of the migrated content")
    band_pos = FIELD_BAND
    v_base = capture_band_v(net0, f_eval_ids, band_pos)
    v_arms = {an: capture_band_v(E83.mk_cpu_net(arms_sd[an]), f_eval_ids, band_pos)
              for an in arm_order}
    dv = {an: [(va - vb) for va, vb in zip(v_arms[an], v_base)]
          for an in arm_order}
    vgeo = {"pairs": {f"{a}|{b}": vgeo_pair(dv[a], dv[b])
                      for a, b in (("r_jittered", "e_erase"),
                                   ("r_jittered", "l_locked"),
                                   ("e_erase", "l_locked"))},
            "note": "mean |cos(dV_X, dV_Y)| at matching (layer,head,ctx,pos); "
                    "nulls = cyclic position roll + seeded permutation; "
                    "REPORT-ONLY (never gates a bar)"}
    for pr_, d_ in vgeo["pairs"].items():
        log(f"(e) {pr_:26s}: |cos| {d_['mean_abs_cos']:.4f} "
            f"(roll {d_['null_roll']:.4f}, perm {d_['null_perm']:.4f})")

    # ---------------- G_R_REPRO (only if R ran its registered 300 steps)
    g_r_repro = None
    if chosen["r_steps"] == 300 and not SMOKE:
        cells = {f"g{g:+d}": {"this_run": post("r_jittered", "none", g),
                              "e109_ref": E109_REF_NONE[g],
                              "diff": post("r_jittered", "none", g) - E109_REF_NONE[g]}
                 for g in GEO_ORDER}
        g_r_repro = {"cells": cells, "tol": G_R_REPRO_TOL,
                     "pass": bool(all(abs(c["diff"]) < G_R_REPRO_TOL
                                      for c in cells.values()))}
        log(f"G_R_REPRO vs e109 none-table (tol {G_R_REPRO_TOL}): "
            + " ".join(f"g{g:+d} {cells[f'g{g:+d}']['this_run']:.4f}/"
                       f"{E109_REF_NONE[g]:.4f}" for g in GEO_ORDER)
            + f" -> {'PASS' if g_r_repro['pass'] else 'FAIL'}")

    # ---------------- adjudication (registered, verbatim clauses)
    dissoc = {"b_one_sided_growth": bool(one_sided_growth and not b_overlap),
              "c_opposite_brake_signs": bool(c_dissoc),
              "d_share_boundary": bool(d_dissoc),
              "held30_dall_survival": bool(h30_rider_dissoc)}
    n_dissoc = sum(dissoc.values())
    if c_match and d_match and b_overlap:
        verdict = "SAME-STORE"
        clause = ("routes match on (c) brake sign and (d) share product within "
                  "band, AND their (b) grown-row censuses overlap in "
                  "position-band — one consolidation attractor, many roads")
    elif n_dissoc >= 2:
        verdict = "DIFFERENT-STORES"
        clause = (f"{n_dissoc} readouts dissociate "
                  f"({', '.join(k for k, v in dissoc.items() if v)}) — the "
                  f"'field' is a family of stores; the lab's newest noun splits")
    else:
        verdict = "AMBIGUOUS"
        clause = (f"neither registered bar fires cleanly (matches: (c) {c_match}, "
                  f"(d) {d_match}, (b) overlap {b_overlap}; dissociations: "
                  f"{n_dissoc} of the named forms) — report the numbers, no "
                  f"bar shopping")
    log("=" * 78)
    log(f"E119 VERDICT: {verdict}")
    log(f"  (c) brake: R {brake['r_jittered']['brake_mean']:+.4f} "
        f"({brake['r_jittered']['class']}) vs E "
        f"{brake['e_erase']['brake_mean']:+.4f} ({brake['e_erase']['class']}) "
        f"-> match {c_match}, dissociate {c_dissoc}")
    log(f"  (d) share: R " + " ".join(f"k{k} r* {share_summary['r_jittered'][f'k{k}']['r_star']}"
                                      for k in ks)
        + " | E " + " ".join(f"k{k} r* {share_summary['e_erase'][f'k{k}']['r_star']}"
                             for k in ks)
        + f" -> match {d_match}, dissociate {d_dissoc} "
          f"(ratios {[round(x, 2) for x in d_ratios]}, off-gap {n_offgap})")
    log(f"  (b) census: R grown {r_grown} vs E grown {e_grown} -> overlap "
        f"{b_overlap}, one-sided {one_sided_growth}")
    log(f"  held-30 D-all: R max {h30['r_jittered']['dall_held30_max']:.3f} "
        f"vs E max {h30['e_erase']['dall_held30_max']:.3f} -> "
        f"{h30_rider_dissoc}")
    log(f"  {clause}")
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e119_migration_headtohead",
        "date": common.now_iso(),
        "registration": "scratch/next_arc_programs.md P5 e119 (the registered "
                        "design, verbatim authority); registered prediction "
                        "and operationalizations frozen in the module "
                        "docstring before compute",
        "registered_prediction": {
            "same_store": "routes match on (c) brake sign and (d) share "
                          "product within per-net band, AND their (b) "
                          "grown-row censuses overlap in position-band",
            "different_stores": ">=2 readouts dissociate (e.g., R grows new "
                                "address rows where E grows none; R survives "
                                "scaffold loss at held-30 where E doesn't; "
                                "opposite brake signs)",
            "ambiguous": "say AMBIGUOUS with the numbers; no bar shopping"},
        "twin_start": f"runs/checkpoints/{INSTALLED_CK} (shared bit-identical "
                      f"install = the 'two same-seed installs' limit; both "
                      f"roads start from the same weights)",
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
                     "geometries": {"trained": list(GEO_ORDER),
                                    "novel": list(novel_geo)},
                     "battery_construction": "ctx = train_text[p-PRE-j:p], "
                                             "readout p(Z) at last position "
                                             "(wpe row 129+j)"},
        "gates": {"G_SPLICE": G_SPLICE, "G_INST": G_INST,
                  "G_EXP_calibration": calibration, "G_SURG": gates_surg,
                  "G_FIELD": G_FIELD, "G_R_REPRO": g_r_repro},
        "arms": arms_meta,
        "ckpt_inventory": CKPT_INVENTORY,
        "e_cycle_summary": e_cycle_summary,
        "deletions_per_arm": {an: {dl: list(rows) for dl, rows in d.items()}
                              for an, d in deletions.items()},
        "census": {an: {k: v for k, v in census[an].items()
                        if k != "dnorm_spectrum"} for an in arm_order},
        "census_spectra": {an: census[an]["dnorm_spectrum"] for an in arm_order},
        "battery_table": {f"{an}__{dl}__g{g:+d}__{bt}":
                          {k: v for k, v in table[(an, dl, g, bt)].items()
                           if k != "pz_per_ctx"}
                          for an in arm_order for dl in deletions[an]
                          for g in geo_all for bt in ("install60", "held30")},
        "brake_c": brake,
        "share_d": {"subsets": {str(k): subsets[k] for k in ks},
                    "r_grid": r_grid, "bar": SHARE_BAR,
                    "primary": "arm's FULL address set zeroed (e113 "
                               "convention); D129-only secondary at r=1",
                    "grid": share, "summary": share_summary,
                    "ongrid_ks": d_ongrid, "ratios": d_ratios,
                    "offgrid_gap_count": n_offgap},
        "vgeo_e": vgeo,
        "held30_dall": h30,
        "adjudication": {"c_match": bool(c_match), "c_dissociate": bool(c_dissoc),
                         "d_match": bool(d_match), "d_dissociate": bool(d_dissoc),
                         "b_overlap": bool(b_overlap),
                         "b_one_sided_growth": bool(one_sided_growth),
                         "b_in_decision_band": in_dec_band,
                         "held30_rider_dissociate": bool(h30_rider_dissoc),
                         "dissociations": dissoc, "n_dissociations": n_dissoc,
                         "verdict": verdict, "clause": clause},
        "trims": trims, "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192, "block_size": 256,
                   "params": int(net0.num_params()),
                   "train_device": "cuda (gated)", "eval_device": "cpu",
                   "torch_threads": torch.get_num_threads(), "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    # ---------------- plot: the migration plate
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.5))
    acol = {"r_jittered": "crimson", "e_erase": "darkorange",
            "l_locked": "steelblue"}

    # (0,0) census strip: dnorm spectrum, arms stacked (rows 100..160 zoom)
    ax = axes[0, 0]
    lo_r, hi_r = 100, 160
    mat = np.stack([np.asarray(census[an]["dnorm_spectrum"][lo_r:hi_r + 1])
                    for an in arm_order])
    im = ax.imshow(mat, aspect="auto", cmap="viridis",
                   extent=[lo_r - 0.5, hi_r + 0.5, 2.5, -0.5])
    ax.axvline(129, color="w", ls="--", lw=1.2)
    for ai_, an in enumerate(arm_order):
        for r in census[an]["grown_rows"]:
            if lo_r <= r <= hi_r:
                ax.text(r, ai_, "*", ha="center", va="center", fontsize=12,
                        color="w" if mat[ai_, r - lo_r] < 0.6 * mat.max() else "k")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([short_lbl[a] for a in arm_order])
    ax.set_xlabel("wpe row")
    ax.set_ylabel("arm")
    ax.set_title(f"(b) grown-row census: dnorm vs shared install (thr +{GROWN_DNORM})\n"
                 f"R grown {r_grown} | E grown {e_grown} | * = census row; "
                 f"white dashed = address row 129", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (0,1) deletion hierarchy at g0 (install-60)
    ax = axes[0, 1]
    dl_names = ("none", "d129", "dgrown", "dall")
    xs = np.arange(len(dl_names))
    bw = 0.26
    for k_i, an in enumerate(arm_order):
        vals = [post(an, dl, 0) for dl in dl_names]
        ax.bar(xs + (k_i - 1) * bw, vals, bw, color=acol[an], edgecolor="k",
               linewidth=0.4, label=short_lbl[an])
        for x, vv in zip(xs + (k_i - 1) * bw, vals):
            ax.text(x, vv + 0.006, f"{vv:.3f}", ha="center", fontsize=6.6,
                    rotation=90, va="bottom")
    ax.axhline(SHARE_BAR, color="seagreen", ls="--", lw=1.1,
               label="survive bar 0.20")
    ax.set_xticks(xs)
    ax.set_xticklabels(["none", "D129\n(original address)",
                        "D-grown\n(arm's census)", "D-all\n(129+grown)"],
                       fontsize=8)
    ax.set_ylabel("battery p(Z) g0 (install-60)")
    ax.set_ylim(0, 1.15)
    ax.set_title("(a) deletion hierarchy at the trained geometry", fontsize=10)
    ax.legend(fontsize=7.5)

    # (0,2) brake (c)
    ax = axes[0, 2]
    xs = np.arange(len(arm_order))
    vals = [brake[an]["brake_mean"] for an in arm_order]
    cis = [brake[an]["brake_ci95"] for an in arm_order]
    ax.bar(xs, vals, 0.5, color=[acol[a] for a in arm_order], edgecolor="k")
    for x, v, c in zip(xs, vals, cis):
        ax.errorbar(x, v, yerr=[[max(0, v - c[0])], [max(0, c[1] - v)]],
                    fmt="none", ecolor="k", capsize=5, lw=1.3)
        ax.text(x, v + (0.02 if v >= 0 else -0.07),
                f"{v:+.3f}\n{brake[arm_order[x]]['class']}", ha="center",
                fontsize=8.5)
    ax.axhline(0, color="k", lw=1.2)
    ax.axhline(BRAKE_FLOOR, color="gray", ls=":", lw=1)
    ax.axhline(-BRAKE_FLOOR, color="gray", ls=":", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([short_lbl[a] for a in arm_order])
    ax.set_ylabel("brake = p(D129) - p(none)  (install-60, g0)")
    ax.set_title("(c) brake sign at full field — deleting the address\n"
                 "SUPPRESSES (negative) or FEEDS (positive) the field?",
                 fontsize=9.5)

    # (1,0) share ladder (primary, full-address)
    ax = axes[1, 0]
    for an in arm_order:
        for k, sty in zip(ks, ("o-", "s--", "^:")):
            xs_r = sorted(r_grid, reverse=True)
            ys = []
            for r in xs_r:
                vals = [share[an]["grid"][f"k{k}_s{si}_r{r:g}"]
                        for si in range(share[an]["n_subsets_used"])
                        if f"k{k}_s{si}_r{r:g}" in share[an]["grid"]]
                ys.append(float(np.mean(vals)) if vals else float("nan"))
            ax.plot(xs_r, ys, sty, color=acol[an], ms=4, lw=1.3,
                    label=f"{short_lbl[an].split(' (')[0]} k={k}")
    ax.axhline(SHARE_BAR, color="seagreen", ls="--", lw=1.2,
               label="collapse bar 0.20")
    ax.set_xscale("log")
    ax.set_xticks([float(v) for v in sorted(r_grid)])
    ax.set_xticklabels([f"{v:g}" for v in sorted(r_grid)], fontsize=7.5)
    ax.set_xlabel("retention r on the retained k-subset (log; complement zeroed)")
    ax.set_ylabel("battery p(Z) (install-60, g0, full-address-deleted)")
    ax.set_title("(d) share ladder r*(k)*k — the field's collapse boundary\n"
                 + " | ".join(f"{short_lbl[a].split(' (')[0]}: "
                              + " ".join(f"k{k}*{share_summary[a][f'k{k}']['r_star']}"
                                         for k in ks)
                              for a in ("r_jittered", "e_erase")), fontsize=8.5)
    ax.legend(fontsize=6.2, ncol=2)

    # (1,1) V-geometry (e)
    ax = axes[1, 1]
    prs = list(vgeo["pairs"].keys())
    xs = np.arange(len(prs))
    vals = [vgeo["pairs"][p_]["mean_abs_cos"] for p_ in prs]
    nul1 = [vgeo["pairs"][p_]["null_roll"] for p_ in prs]
    nul2 = [vgeo["pairs"][p_]["null_perm"] for p_ in prs]
    ax.bar(xs - 0.22, vals, 0.22, color="seagreen", edgecolor="k",
           linewidth=0.4, label="mean |cos(dV_X, dV_Y)|")
    ax.bar(xs, nul1, 0.22, color="lightgray", edgecolor="k", linewidth=0.4,
           label="null (position roll)")
    ax.bar(xs + 0.22, nul2, 0.22, color="whitesmoke", edgecolor="k",
           linewidth=0.6, hatch="//", label="null (permutation)")
    for x, v in zip(xs - 0.22, vals):
        ax.text(x, v + 0.004, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([p_.replace("r_jittered", "R").replace("e_erase", "E")
                        .replace("l_locked", "L").replace("|", "\nvs ")
                        for p_ in prs], fontsize=8)
    ax.set_title("(e) V-geometry typing of the migrated content (report-only)",
                 fontsize=9.5)
    ax.legend(fontsize=7.5)

    # (1,2) verdict panel
    ax = axes[1, 2]
    ax.axis("off")
    p_e_str = f"{chosen['p_e']:.4f}" if chosen["p_e"] is not None else "n/a"
    vlines = [
        "GATE (calibration):",
        f"  R@{chosen['r_steps']} pz(g0) {chosen['p_r']:.4f} | "
        f"E@c{chosen['e_cycles']} pz(g0) {p_e_str}",
        f"  gap {chosen['gap']:.4f} <= {GATE_BAND} : "
        f"{'PASS' if gate_pass else 'FAIL'}"
        f"   | L pz(g0) {p_l:.4f}",
        "",
        "DECIDING NUMBERS:",
        f"  (c) brake R {brake['r_jittered']['brake_mean']:+.4f} "
        f"[{brake['r_jittered']['class']}] vs E "
        f"{brake['e_erase']['brake_mean']:+.4f} "
        f"[{brake['e_erase']['class']}]",
        f"  (d) r*: R " + " ".join(f"k{k}={share_summary['r_jittered'][f'k{k}']['r_star']}"
                                   for k in ks)
        + " | E " + " ".join(f"k{k}={share_summary['e_erase'][f'k{k}']['r_star']}"
                             for k in ks),
        f"      products R " + " ".join(str(share_summary['r_jittered'][f'k{k}']['product'])
                                       for k in ks)
        + " | E " + " ".join(str(share_summary['e_erase'][f'k{k}']['product'])
                             for k in ks),
        f"  (b) grown: R {r_grown} | E {e_grown} (overlap {b_overlap})",
        f"  held-30 D-all max: R {h30['r_jittered']['dall_held30_max']:.3f} "
        f"| E {h30['e_erase']['dall_held30_max']:.3f}",
        "",
        f"DISSOCIATIONS ({n_dissoc}): "
        + (", ".join(k for k, v in dissoc.items() if v) or "none"),
        "",
        f"VERDICT: {verdict}",
    ] + [f"  {wd}" for wd in
         [clause[i:i + 64] for i in range(0, len(clause), 64)]]
    for i, tx in enumerate(vlines):
        ax.text(0.02, 0.97 - i * 0.045, tx, fontsize=7.6, va="top",
                family="monospace")

    fig.suptitle(f"E119 — the migration head-to-head: road R (jittered replay) "
                 f"vs road E (erase cycles), same fact, same twin start -> "
                 f"{verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(rd / "migration_headtohead.png", dpi=130)
    plt.close(fig)

    log(f"outputs: {rd / 'metrics.json'}, {rd / 'migration_headtohead.png'}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
