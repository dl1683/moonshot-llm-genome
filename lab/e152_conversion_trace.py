"""E152 — THE CONVERSION TIME-TRACE (T088's central open number: where in
8-300 steps does the phase convert, and does a TRANSIENT two-door state exist
mid-conversion?).

WHY (T088 (2)+(3)): e151 showed ONE 300-step locked re-teach converts a
sink-coupled (geometry-general) memory to site-stored (g-12 0.916 -> 0.102,
retention 0.112) — but the 8-step smoke held g-12 at 0.99. The conversion
lives somewhere in 8-300 steps. If a two-door state exists mid-conversion,
e151's committed P-b prediction was EARLY, not wrong — the cliff has a DWELL
TIME. This experiment traces it: the e151 re-teach protocol run as ONE
trajectory with checkpoints at steps {8, 16, 32, 64, 128, 300}, the FULL
e151 before/after battery measured at every checkpoint.

ROOT (mandated): runs/checkpoints/e131_consolidated_e113.pt — gated bit-exact
vs e151's stored root cells (G_CONS/G_ROW0/G_A129/G_DALL refs below are
e151's stored values, which are e131's stored values).

CONTINUATION CHOICE (frozen before compute, per the dispatch's instruction):
SEQUENTIAL CONTINUATION — one 300-step training from the root (seed 10902,
generator never reset), CPU state-dict snapshots saved at steps 8/16/32/64/
128/300 as runs/checkpoints/e152_steps{N}.pt. This IS e151's protocol (e151
ran ONE continuous 300-step fine-tune); an independent N-step run with the
same seed coincides with the first N steps BY CONSTRUCTION (per-step randint
draws come from one seed-fixed generator sequence and the in-loop CPU evals
consume no RNG), so the two designs differ only by GPU float nondeterminism —
and sequential removes that run-to-run noise from the monotonicity read. The
300-step endpoint therefore reproduces e151's conversion up to float
nondeterminism (quantified in the reproduction block). Cost recorded in the
honesty reflex: the six checkpoints are ONE trajectory, not six replicates.

RE-TEACH (e151 VERBATIM = e143 locked replay = e109 arm-b = e119-L): batch
32 = 16 install windows + 16 anchors (8 paired + 8 random), e043 token-level
union CE on the 7 name-char targets (name-only mask), AdamW (0.9,0.95) wd 0.1
constant lr 1e-3 clip 1.0, <=180 s GPU cap, seed 10902. PLACEMENT: offset
j=+54, ZEPHYRA locked at x-cols 184..190 (onset READ ROW 183; e120/e121/
e133 splice-site geometry), pre-context 184 true tokens, continuation 65,
zero position variance by construction. In-loop CPU evals at each checkpoint
step (evals consume no RNG and do not touch the trajectory).

MEASURE PER CHECKPOINT (the full e151 battery, identical instruments; root
measured as step-0 "before"): (i) g-12 AND g+12 retention (novel-geometry
expression vs the root's 0.916/0.948; install-60 batteries, ctx = train_text
[p-PRE-j:p]); (ii) 183-site content census (span-primary readout reads
183..189; row-183 strength + band-max strength + site_pos at the 2x-control
bar); (iii) A(129) address-key delta; (iv) D-all g0; (v) base expression g0
+ CE_R (wreckage guard); plus e151's texture cells (d183 deletion, old-band
census, e150-informed mask/ladder probes) co-reported in full.

REGISTERED PREDICTION (verbatim from QUEUE e152 / the dispatch; adjudicate
against exactly this; no bar shopping; texture => TEXTURE with numbers):
  - CLEAN-CONVERSION fires if: g-12 retention decays monotonically with no
    plateau, and site growth anti-correlates (Spearman <= -0.8 between g-12
    retention and site content across checkpoints).
  - TRANSIENT-TWO-DOOR fires if: some intermediate checkpoint has BOTH site
    content clearing the content bar AND g-12 retention >= 0.5, followed by
    decay — the cliff has a dwell time; e151's P-b prediction was EARLY, not
    wrong.
  - DELAYED-CONVERSION fires if: g-12 stays flat (>= 0.8) until a step
    threshold then cliffs — a critical mass of zero-variance steps.
  - Texture => TEXTURE with numbers.

OPERATIONALIZATIONS (frozen here before compute — the registered clauses name
monotone/plateau/threshold without numbers; these fix them, they do not move
the bars):
  * retention(s) = g-12 install-60 battery mean_pz at checkpoint s / the
    root's 0.9155886769294739 (the root is retention 1.0 at step 0 by
    construction, excluded from the checkpoint statistics).
  * "site content" = the e151 span-primary census BAND-MAX strength over
    rows 183..189 (e151's site_strength instrument); row-183 strength
    co-reported everywhere. "clears the content bar" = e151's site_pos
    (any band row content=True AND strength >= 2x the same-census
    shared-control-max).
  * "decays monotonically" = every successive retention difference across
    the six checkpoints in step order is <= +0.01 (numerical slack).
  * "no plateau" = at most ONE adjacent checkpoint pair with |Dret| <= 0.02
    (two or more near-zero gaps = a dwell/shelf).
  * Spearman primary = rank correlation between g-12 retention and site
    content across the SIX checkpoints; the 7-point version including the
    root is co-reported.
  * TRANSIENT: "intermediate checkpoint" = any checkpoint with s < 300;
    "followed by decay" = ret(300) < ret(s) - 0.20.
  * DELAYED: "flat until a threshold" = a leading prefix of >= 2 checkpoints
    with retention >= 0.80 and within-prefix adjacent |Dret| <= 0.05;
    "then cliffs" = some checkpoint after the prefix has retention <= 0.30.
  * Adjudication order: CLEAN-CONVERSION -> TRANSIENT-TWO-DOOR ->
    DELAYED-CONVERSION -> TEXTURE. Every sub-boolean reported regardless.

INSTRUMENT PROVENANCE: load_cpu/evl_load/battery_cell/battery_pz/
ce_fixed_cpu/val_windows/deleted_wpe/read_fact_at/row_census/gate_launch are
lab/e151_twodoor.py VERBATIM (e143/e065/e068/e109/e113/e116/e119/e131
lineage); battery_fwd/ce_fwd/forward_custom/fwd_causal/fwd_offsink are
lab/e150_flatce_route.py VERBATIM (the forced-off-sink mask); modified_wpe
is e141's row-value surgery with the e131 confinement gate. finetune_trace
is e151's finetune_arm with (only) checkpoint snapshots + evals at the six
checkpoint steps (no RNG consumed, trajectory-preserving). Copied, not
imported, because importing e131/e150 force-sets CUDA_VISIBLE_DEVICES=-1
(CPU-only rigs) and this experiment owns the GPU. Protocol rebuild: corpus
seed 1337, SPLICE_RNG 24301 host shuffle, install60/held30 split, mix gate —
e143/e151 verbatim.

COMPUTE ENVELOPE: GPU for the ONE fine-tune (idle check gpu_ok() double-poll
via gate_launch, 20-min bounded wait then PARK; cooldown(90) before and
after — e151's envelope; the dispatch's 60-120 s between-trainings cooldowns
are moot under sequential continuation: there is ONE training, and its
trajectory cannot be interrupted). <=180 s training cap. ALL readouts
CPU-side (torch threads 8, e143's convention), sequential, no busy-waiting.

Outputs: runs/e152/{metrics.json, conversion_trace.png}; checkpoints
runs/checkpoints/e152_steps{8,16,32,64,128,300}.pt (ckpt_inventory in
metrics). No NOTES/THINKING/QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python e152_conversion_trace.py    (E152_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")     # GPU for the re-teach

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

SMOKE = os.environ.get("E152_SMOKE") == "1"
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
ROOT_CK = "e131_consolidated_e113.pt"
CKPT_DIR = E43.REPO / "runs" / "checkpoints"

# ---- re-teach placement (183-style, FAR's offset machinery; e151 verbatim) ----
RETEACH_J = 54                    # offset: name x-cols 184..190, read rows 183..189
SITE_ADDR_ROW = 183               # the onset read row (= e120's SPLICE_ADDR_ROW)
SITE_Z_XCOL = PRE + RETEACH_J     # 184 (= e120's Z_XCOL)
SITE_CONT = BLOCK - PRE - RETEACH_J - len(NAME)     # 65-token continuation
SITE_ROWS = tuple(range(183, 190))                  # the 7 trained read rows
SHARED_CTR = (60, 100, 150, 160, 170, 200, 220)     # outside every trained band
ROWS_183 = (0, 1, 2) + (180, 181, 182) + SITE_ROWS + SHARED_CTR
ROWS_OLD = (0, 1, 2, 3, 4, 5, 6, 60, 100, 118, 119, 120) + tuple(range(121, 130))
D_ALL = (121, 125, 129, 133, 137)                   # e113's fixed set
GEOS = (-12, 0, 12)               # novel x2 + the trained g0 (root trained {-8..+8})
if SMOKE:
    ROWS_183 = (0, 1, 2, 182) + SITE_ROWS + (60, 100)
    ROWS_OLD = (0, 1, 60, 121, 125, 129)

# ---- the trace: checkpoint steps (sequential continuation, one trajectory) ----
CKPT_STEPS: tuple[int, ...] = (8, 16, 32, 64, 128, 300) if not SMOKE else (2, 4)
CKPT_SET = set(CKPT_STEPS)

# ---- fine-tune envelope (e109 arm-b / e119-L / e143 / e151 verbatim) -----------
FT_LR = 1e-3
FT_STEPS = CKPT_STEPS[-1]
FT_TIME_CAP = 180.0 if _USE_GPU else 1500.0
NAME_BS, ANCH_BS = 16, 16
RETEACH_SEED = 10902              # e119's L_SEED / e143's NEAR+FAR locked / e151
COOLDOWN_S = 90.0                 # e151's envelope, around the ONE training

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

# ---- e151's stored AFTER cells (the 300-step reproduction targets) -------------
E151_AFTER = {
    "base_gm12": 0.1020549014210701, "base_g0": 0.10710463672876358,
    "base_gp12": 0.12079524248838425, "ce_r": 1.6489683389663696,
    "site_span_strength": 0.07310783863067627,
    "site_onset_strength": 0.5117409825325012,
    "row183_span_strength": 0.07310783863067627,   # band max WAS row 183
    "A129": 0.005159098243098015,
    "row0_strength": 0.10635086206680928,
    "dall_g0": 0.09776780754327774,
    "site_read_onset": 0.9980737566947937,
}

# ---- e150-informed probe constants (e151 verbatim) ------------------------------
LADDER_NORMS = (0.07, 0.15)       # the poison-probe bracket (e150's verdict)

# ---- registered bar constants (frozen; see OPERATIONALIZATIONS above) ----------
SITE_CTRL_MULT = 2.0              # e139/e151 site convention: >= 2x control-max
ROOT_REF_GM12 = 0.9155886769294739            # e151 before g-12 (the denominator)
ROOT_REF_GP12 = 0.9478210210800171            # e151 before g+12 (co-reported den.)
ROOT_REF_G0 = 0.7850371599197388              # e151 before g0
MONO_SLACK = 0.01                 # successive retention diff allowed above zero
PLATEAU_GAP = 0.02                # adjacent |Dret| <= this counts as a plateau gap
MAX_PLATEAU_GAPS = 1              # CLEAN allows at most one
SPEARMAN_BAR = -0.8               # CLEAN anti-correlation bar (six checkpoints)
TRANSIENT_RET_BAR = 0.5           # TRANSIENT g-12 retention clause
TRANSIENT_DECAY = 0.20            # TRANSIENT "followed by decay" clause
DELAYED_HI = 0.8                  # DELAYED flat-prefix level
DELAYED_FLAT = 0.05               # DELAYED within-prefix adjacent |Dret|
DELAYED_LO = 0.3                  # DELAYED post-threshold cliff level

REGISTERED_PREDICTION = {
    "clean_conversion": "CLEAN-CONVERSION fires if: g-12 retention decays "
        "monotonically with no plateau, and site growth anti-correlates "
        "(Spearman <= -0.8 between g-12 retention and site content across "
        "checkpoints).",
    "transient_two_door": "TRANSIENT-TWO-DOOR fires if: some intermediate "
        "checkpoint has BOTH site content clearing the content bar AND g-12 "
        "retention >= 0.5, followed by decay — the cliff has a dwell time; "
        "e151's P-b prediction was EARLY, not wrong.",
    "delayed_conversion": "DELAYED-CONVERSION fires if: g-12 stays flat "
        "(>= 0.8) until a step threshold then cliffs — a critical mass of "
        "zero-variance steps.",
    "texture": "Texture => TEXTURE with numbers.",
    "operationalizations": "retention(s) = g-12 battery mean_pz(s) / root "
        f"{ROOT_REF_GM12:.16f}; site content = e151 span-primary census "
        "band-max strength over rows 183..189 (row-183 co-reported); clears "
        "the content bar = e151 site_pos (band row content=True AND strength "
        ">= 2x same-census control-max); monotone = successive diffs <= "
        "+0.01; no plateau = at most one adjacent |Dret| <= 0.02; Spearman "
        "primary across the SIX checkpoints (7-point incl. root co-reported); "
        "TRANSIENT intermediate = s < 300 with ret(300) < ret(s) - 0.20; "
        "DELAYED = leading prefix of >= 2 checkpoints ret >= 0.80 with "
        "within-prefix |Dret| <= 0.05 AND a later checkpoint <= 0.30; order "
        "CLEAN -> TRANSIENT -> DELAYED -> TEXTURE.",
    "committed": "QUEUE e152 registered no committed branch (three-bar fork; "
                 "T088 frames the transient as the open question).",
}

trims: list[str] = []
deviations: list[str] = [
    "Sequential continuation chosen (one 300-step run, snapshots at "
    "8/16/32/64/128/300) — e151's own protocol is ONE continuous 300-step "
    "training; independent same-seed N-step runs coincide with the first N "
    "steps by construction (seed-fixed generator, RNG-free evals), so the "
    "designs differ only by GPU float nondeterminism; recorded per the "
    "dispatch's instruction. The dispatch's between-trainings cooldowns are "
    "moot under continuation (one trajectory); e151's cooldown(90) before/ "
    "after envelope kept.",
    "Nets are the mandated 2.7M e131_consolidated line (the dispatch's "
    "'<=1M family' note is an envelope statement; every instrument, gate "
    "reference, and lineage number of this cell lives on the 2.7M line — "
    "e143/e151 precedent, root mandate honored).",
    "The registered clauses name monotone/plateau/threshold without numbers; "
    "the OPERATIONALIZATIONS in the module docstring freeze them BEFORE "
    "compute (slack +0.01, plateau gap 0.02 with max 1, transient decay 0.20, "
    "delayed prefix 0.80/0.05 with cliff 0.30) — fixed definitions, not bar "
    "shopping; every sub-boolean reported regardless.",
    "Single seed (10902), single lineage, ONE trajectory — the dwell time, "
    "if any, is a point estimate (n=1 path through step-space).",
    "GPU float nondeterminism may move battery cells ~1e-2 vs e151's stored "
    "AFTER cells (e119 precedent); the reproduction block reports max abs "
    "diff at the lab's 0.05 tolerance with the 5e-6 bit flag.",
    "Smoke mode trims: 4-step re-teach with checkpoints at {2,4}, reduced "
    "census rows, nothing adjudicated.",
]


# ------------------------------------------------------------------ gpu guard

def gate_launch(tag: str) -> None:
    """e119/e143/e151's bounded gate_launch: gpu_ok() double-poll, 20-min wait,
    PARK."""
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


GPU_WAIT_MAX_S, GPU_POLL_S = 1200.0, 30.0


# ------------------------------------------------------------------ instruments
# PROVENANCE: verbatim lineage — see the module docstring. Copied rather than
# imported (e131/e150 rigs force CUDA_VISIBLE_DEVICES=-1; this one owns the
# GPU). finetune_trace = e151's finetune_arm + checkpoint snapshots (RNG-free).

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
def battery_fwd(net: TinyGPT, ids: torch.Tensor, zid: int, fwd=None,
                bs=30) -> dict:
    """e150's battery_fwd: battery_cell with a pluggable forward (mask cells)."""
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


def modified_wpe(sd: dict, row: int, value) -> tuple[dict, dict]:
    """e141's row-value surgery with e131's confinement gate (the ladder arm)."""
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


@torch.no_grad()
def forward_custom(net: TinyGPT, idx, targets=None, block_key0=False):
    """e150's custom forward VERBATIM: common.TinyGPT.forward replicated with
    an explicit additive attention mask. block_key0=True: attention TO key
    position 0 blocked for queries 1..T-1, ALL layers, ALL heads."""
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


# ------------------------------------------------------------------ fine-tune

def finetune_trace(tag: str, net0: TinyGPT, pool_x: torch.Tensor,
                   pool_mask: torch.Tensor, anchor: torch.Tensor,
                   train_ids: torch.Tensor, r_eval_xy, f_eval_ids, zid: int,
                   seed: int):
    """e151's finetune_arm with (only) checkpoint snapshots at CKPT_STEPS:
    the per-step body is VERBATIM (same generator, same draw order/sizes —
    ix(16) then aj(8) then rj(8) per step), so the trajectory is e151's.
    At each checkpoint step: CPU sd snapshot (deep-copy out; nothing loaded
    into the training net) + in-loop CPU eval on the twin (no RNG consumed).
    """
    if DEV.type == "cuda":
        gate_launch(tag)
    net = copy.deepcopy(net0).to(DEV)
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
        if step in CKPT_SET:
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
        if (time.time() - t_start) > FT_TIME_CAP:
            log(f"  [{tag}] time cap {FT_TIME_CAP:.0f}s at s{step}")
            trims.append(f"{tag}: time cap at step {step}")
            break
    net.eval()
    del net
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    return {"sds": sds, "traj": traj, "steps_ran": step, "seed": seed}


CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "e152", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name} "
        f"({', '.join(f'{k}={v}' for k, v in meta.items())})")


# ------------------------------------------------------------------ statistics

def _ranks(v) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    v = np.asarray(v, dtype=float)
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v), dtype=float)
    sv = v[order]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    ra, rb = _ranks(a), _ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else 0.0


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e152_smoke" if SMOKE else "e152")
    log(f"E152 THE CONVERSION TIME-TRACE (smoke={SMOKE}) -> {rd}")
    log(f"compute: train device {DEV} (gpu_ok at start: {_USE_GPU}), "
        f"cpu threads {torch.get_num_threads()}")
    log(f"checkpoints at steps {CKPT_STEPS} (sequential continuation, "
        f"seed {RETEACH_SEED})")

    if not _USE_GPU and not SMOKE:
        deviations.append("GPU parked (gpu_ok() failed at startup or no CUDA) "
                          "— the re-teach ran CPU-side under the 1500 s cap; "
                          "reported, adjudication unchanged. This run's cause: "
                          "a user-space graphics process held the GPU at "
                          "86-87C (above the lab's 80C launch ceiling); "
                          "thermal guard honored, no lab process touched. "
                          "Trajectory fidelity: the E152 smoke's CPU in-loop "
                          "traj matched e151's CUDA smoke traj to 3-4 "
                          "decimals (s2 p_z 0.0579/0.058, s4 0.6749/0.675; "
                          "CE 2.0962/2.096 -> 1.9205/1.920), so the CPU "
                          "float path is equivalent for this workload; the "
                          "reproduction block prices the residual.")

    # ---------------- protocol rebuild (e143/e151 verbatim)
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

    # ---------------- re-teach pool: offset j=+54 (FAR's machinery, 183 site)
    wins = []
    for p, h in install_occ:
        pre = train_ids[p - PRE - RETEACH_J: p]
        post = train_ids[p + len(h): p + len(h) + SITE_CONT]
        if len(pre) != PRE + RETEACH_J or len(post) != SITE_CONT:
            raise RuntimeError(f"re-teach window short at p={p}")
        w = torch.cat([pre, name_ids, post])
        if len(w) != BLOCK:
            raise RuntimeError(f"re-teach window len {len(w)} != {BLOCK}")
        wins.append(w)
    pool_x = torch.stack(wins)
    pool_mask = torch.zeros(len(wins), BLOCK - 1, dtype=torch.bool)
    pool_mask[:, SITE_ADDR_ROW: SITE_ADDR_ROW + L] = True
    G_GEO = {
        "name_xcols": [SITE_Z_XCOL, SITE_Z_XCOL + L - 1],
        "all_windows_name_in_place": bool(
            all(torch.equal(w[SITE_Z_XCOL: SITE_Z_XCOL + L], name_ids)
                for w in pool_x)),
        "mask_targets_per_window": int(pool_mask[0].sum()),
        "mask_cols": [SITE_ADDR_ROW, SITE_ADDR_ROW + L - 1],
        "zero_variance": True,
    }
    G_GEO["pass"] = bool(G_GEO["all_windows_name_in_place"]
                         and G_GEO["mask_targets_per_window"] == L)
    assert G_GEO["pass"], f"re-teach geometry gate FAILED: {G_GEO}"
    log(f"re-teach pool: {tuple(pool_x.shape)} — ZEPHYRA locked at x-cols "
        f"{SITE_Z_XCOL}..{SITE_Z_XCOL + L - 1} (onset read row "
        f"{SITE_ADDR_ROW}), {L}-target name mask, zero position variance")

    # anchor bank (e065/e109/e143/e151 verbatim)
    anchor = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                          for p, _ in install_occ[:16]])

    # ---------------- batteries: ctx = train_text[p-PRE-j : p] (e119 verbatim)
    bat_ids = {}
    for j in GEOS:
        for tag, occ in (("install60", install_occ), ("held30", held_occ)):
            cs = [train_text[p - PRE - j: p] for p, _ in occ]
            bat_ids[(j, tag)] = torch.stack([corpus.encode(c) for c in cs])
    ids130 = bat_ids[(0, "install60")]             # e116/e131 battery verbatim
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    f_eval_ids = pool_x[:, :SITE_Z_XCOL]           # p(Z) read at row 183

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
    # the measurement battery (e151's measure VERBATIM; run per net)
    # =====================================================================
    DELS = {"none": (), "d129": (PRE - 1,), "d_all": D_ALL, "d_r0": (0,),
            "d183": (SITE_ADDR_ROW,)}
    gates_surg: dict = {}

    def measure(sd: dict, tag: str) -> dict:
        net = evl_load(sd)
        out: dict = {"tag": tag}
        # (0) base expression + CE
        out["base"] = {j: battery_cell(net, bat_ids[(j, "install60")], zid)
                       for j in GEOS}
        out["ce_r"] = ce_fixed_cpu(net, *r_eval_xy)
        log(f"[{tag}] base: " + " ".join(f"g{j:+d} {out['base'][j]['mean_pz']:.4f}"
                                         for j in GEOS)
            + f" | CE_R {out['ce_r']:.4f}")

        # (i)+(ii) site content test at 183 — span-primary + onset co-report
        def span_fn(n):
            return read_fact_at(n, pool_x, name_ids, zid, SITE_ADDR_ROW,
                                SITE_Z_XCOL)["pname_mean_over7"]

        def onset_fn(n):
            return read_fact_at(n, pool_x, name_ids, zid, SITE_ADDR_ROW,
                                SITE_Z_XCOL)["pz_onset_mean"]

        out["site_read"] = read_fact_at(net, pool_x, name_ids, zid,
                                        SITE_ADDR_ROW, SITE_Z_XCOL)
        out["census183_span"] = row_census(net, ROWS_183, span_fn)
        out["census183_onset"] = row_census(net, ROWS_183, onset_fn)
        for ro in ("span", "onset"):
            cen = out[f"census183_{ro}"]
            cm = max(cen["rows"][str(r)]["strength"] for r in SHARED_CTR
                     if str(r) in cen["rows"])
            sstr = max(cen["rows"][str(r)]["strength"] for r in SITE_ROWS)
            spos = any(cen["rows"][str(r)]["content"] and
                       cen["rows"][str(r)]["strength"] >= SITE_CTRL_MULT * cm
                       for r in SITE_ROWS)
            best = max(SITE_ROWS, key=lambda r: cen["rows"][str(r)]["strength"])
            out[f"site_{ro}"] = {"control_max": cm, "site_strength": sstr,
                                 "bar_2x_control": SITE_CTRL_MULT * cm,
                                 "site_pos": spos, "peak_row": int(best)}
            log(f"[{tag}] site({ro}) strength {sstr:+.4f} @r{best} "
                f"(2x-ctrl {SITE_CTRL_MULT * cm:.4f}) -> site_pos {spos} "
                f"| base readout {cen['base_readout']:.4f}")

        # (vii) old-band census (g0 readout) — the original trace + A(129)
        out["census_old"] = row_census(net, ROWS_OLD,
                                       lambda n: battery_pz(n, ids130, zid))
        co = out["census_old"]["rows"]
        r0, a129 = co["0"], co["129"]
        ctrl_old = max(co[str(r)]["strength"] for r in (1, 2, 3, 4, 5, 6)
                       if str(r) in co)
        out["old_band"] = {
            "base_pz": out["census_old"]["base_readout"],
            "row0": r0, "row129": a129,
            "row0_strength": r0["strength"], "A129": a129["strength"],
            "band121_129_max": max(co[str(r)]["strength"] for r in range(121, 130)
                                   if str(r) in co),
            "control_max_1_6": ctrl_old}
        log(f"[{tag}] old band: row0 S {r0['strength']:+.4f} | A(129) "
            f"{a129['strength']:+.4f} | band121-129 max "
            f"{out['old_band']['band121_129_max']:+.4f} "
            f"(base {out['census_old']['base_readout']:.4f})")

        # (iv)+(vi) deletion table
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
            if dl == "d183":                      # the new door's necessity
                cell["gm12"] = battery_cell(net, bat_ids[(-12, "install60")], zid)
                cell["site_onset"] = read_fact_at(
                    net, pool_x, name_ids, zid, SITE_ADDR_ROW,
                    SITE_Z_XCOL)["pz_onset_mean"]
                cell["site_span"] = read_fact_at(
                    net, pool_x, name_ids, zid, SITE_ADDR_ROW,
                    SITE_Z_XCOL)["pname_mean_over7"]
            out["del_table"][dl] = cell
        net.load_state_dict(sd)                    # restore
        log(f"[{tag}] deletions g0: " + " | ".join(
            f"{dl} {out['del_table'][dl]['g0']['mean_pz']:.3f}"
            for dl in DELS))

        # (v) e150-informed probes at both geometries
        f0, f1 = fwd_causal(net), fwd_offsink(net)
        mask = {}
        for j in (0, -12):
            b = battery_fwd(net, bat_ids[(j, "install60")], zid, fwd=f0)
            m = battery_fwd(net, bat_ids[(j, "install60")], zid, fwd=f1)
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
                ent[f"g{j:+d}"] = battery_cell(net, bat_ids[(j, "install60")],
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
    log("STEP-0 battery (root: e131_consolidated_e113 = e151's 'before')")
    root = measure(sd_root, "root")

    # instrument gates against e131/e151's stored census cells (root = before)
    G_ROW0 = {"mean": root["old_band"]["row0"]["mean"],
              "zero": root["old_band"]["row0"]["zero"],
              "ref_mean": G_R0_REF_MEAN, "ref_zero": G_R0_REF_ZERO,
              "pass": bool(abs(root["old_band"]["row0"]["mean"] - G_R0_REF_MEAN)
                           < G_FALLBACK_TOL
                           and abs(root["old_band"]["row0"]["zero"] - G_R0_REF_ZERO)
                           < G_FALLBACK_TOL)}
    G_A129 = {"mean": root["old_band"]["row129"]["mean"],
              "zero": root["old_band"]["row129"]["zero"],
              "ref_mean": G_A129_REF_MEAN, "ref_zero": G_A129_REF_ZERO,
              "pass": bool(abs(root["old_band"]["row129"]["mean"] - G_A129_REF_MEAN)
                           < G_FALLBACK_TOL
                           and abs(root["old_band"]["row129"]["zero"] - G_A129_REF_ZERO)
                           < G_FALLBACK_TOL)}
    G_DALL = {"dall_g0": root["del_table"]["d_all"]["g0"]["mean_pz"],
              "ref": G_DALL_REF,
              "pass": bool(abs(root["del_table"]["d_all"]["g0"]["mean_pz"]
                               - G_DALL_REF) < G_FALLBACK_TOL)}
    for gname, g in (("G_ROW0", G_ROW0), ("G_A129", G_A129), ("G_DALL", G_DALL)):
        log(f"{gname}: {'PASS' if g['pass'] else 'FAIL'} ({json.dumps({k: round(v, 6) if isinstance(v, float) else v for k, v in g.items() if k not in ('pass',)})})")
        if not g["pass"]:
            raise RuntimeError(f"{gname} failed vs e151 stored cells")
    log("gates: G_SPLICE, G_GEO, G_CONS, G_MASK, G_ROW0, G_A129, G_DALL all PASS")

    # =====================================================================
    # RE-TEACH (the ONE training; GPU; sequential continuation with
    # checkpoints; cooldown around it — e151's envelope)
    # =====================================================================
    log("=" * 78)
    if not SMOKE and _USE_GPU:
        log(f"[thermal] cooldown {COOLDOWN_S:.0f}s before the ONE training")
        cooldown(COOLDOWN_S)
    elif not SMOKE:
        log("[thermal] CPU-fallback run: GPU cooldown skipped (no GPU launch "
            "to space; the hot GPU is a user-space graphics process, "
            "untouched)")
    log(f"RE-TEACH: {FT_STEPS}-step locked replay of ZEPHYRA at rows 183..189 "
        f"(seed {RETEACH_SEED}, device {DEV}), checkpoints at {CKPT_STEPS}")
    reteach = finetune_trace("reteach183_trace", net0, pool_x, pool_mask,
                             anchor, train_ids, r_eval_xy, f_eval_ids, zid,
                             RETEACH_SEED)
    if not SMOKE and _USE_GPU:
        log(f"[thermal] cooldown {COOLDOWN_S:.0f}s after the ONE training")
        cooldown(COOLDOWN_S)
    for s in sorted(reteach["sds"]):
        save_ckpt(f"e152_steps{s}", reteach["sds"][s],
                  {"desc": f"e131_consolidated_e113 + {s}-step locked replay "
                           f"of ZEPHYRA at read rows 183..189 (x-cols "
                           f"184..190), name-only mask, seed 10902 — "
                           f"sequential-continuation snapshot of the single "
                           f"{FT_STEPS}-step run",
                   "steps": s, "seed": RETEACH_SEED,
                   "base": f"runs/checkpoints/{ROOT_CK}"})
    missing = [s for s in CKPT_STEPS if s not in reteach["sds"]]
    if missing:
        trims.append(f"checkpoints not reached (time cap): {missing}")

    log("=" * 78)
    batteries = {"root": root}
    for s in sorted(reteach["sds"]):
        log(f"STEP-{s} battery")
        batteries[str(s)] = measure(reteach["sds"][s], f"s{s}")

    # =====================================================================
    # THE TRACE TABLE + ADJUDICATION (registered clauses; no bar shopping)
    # =====================================================================
    steps_meas = [0] + sorted(s for s in reteach["sds"])
    trace = []
    for s in steps_meas:
        b = batteries["root" if s == 0 else str(s)]
        row = {
            "steps": s,
            "base_gm12": b["base"][-12]["mean_pz"],
            "base_g0": b["base"][0]["mean_pz"],
            "base_gp12": b["base"][12]["mean_pz"],
            "retention_gm12": b["base"][-12]["mean_pz"] / ROOT_REF_GM12,
            "retention_g0": b["base"][0]["mean_pz"] / ROOT_REF_G0,
            "retention_gp12": b["base"][12]["mean_pz"] / ROOT_REF_GP12,
            "ce_r": b["ce_r"],
            "site_read_onset": b["site_read"]["pz_onset_mean"],
            "site_read_span": b["site_read"]["pname_mean_over7"],
            "site_span_strength": b["site_span"]["site_strength"],
            "site_span_peak_row": b["site_span"]["peak_row"],
            "site_span_bar_2x_control": b["site_span"]["bar_2x_control"],
            "site_pos_span": b["site_span"]["site_pos"],
            "site_onset_strength": b["site_onset"]["site_strength"],
            "site_pos_onset": b["site_onset"]["site_pos"],
            "row183_span_strength": b["census183_span"]["rows"]["183"]["strength"]
                if "183" in b["census183_span"]["rows"] else None,
            "A129": b["old_band"]["A129"],
            "row0_strength": b["old_band"]["row0_strength"],
            "band121_129_max": b["old_band"]["band121_129_max"],
            "dall_g0": b["del_table"]["d_all"]["g0"]["mean_pz"],
            "d129_g0": b["del_table"]["d129"]["g0"]["mean_pz"],
            "dr0_g0": b["del_table"]["d_r0"]["g0"]["mean_pz"],
            "d183_g0": b["del_table"]["d183"]["g0"]["mean_pz"],
            "d183_gm12": b["del_table"]["d183"]["gm12"]["mean_pz"],
            "d183_site_span": b["del_table"]["d183"]["site_span"],
            "mask_retention_g0": b["mask"]["g+0"]["retention"],
            "mask_retention_gm12": b["mask"]["g-12"]["retention"],
            "ladder007_gm12": b["ladder"][0]["g-12"],
            "ladder015_gm12": b["ladder"][1]["g-12"],
        }
        trace.append(row)
    ck_steps = [r["steps"] for r in trace if r["steps"] > 0]
    ret_seq = [r["retention_gm12"] for r in trace if r["steps"] > 0]
    site_seq = [r["site_span_strength"] for r in trace if r["steps"] > 0]
    ret300 = trace[-1]["retention_gm12"]

    diffs = [ret_seq[i + 1] - ret_seq[i] for i in range(len(ret_seq) - 1)]
    monotone = bool(all(d <= MONO_SLACK for d in diffs))
    plateau_gaps = int(sum(1 for d in diffs if abs(d) <= PLATEAU_GAP))
    no_plateau = bool(plateau_gaps <= MAX_PLATEAU_GAPS)
    rho6 = spearman(ret_seq, site_seq)
    rho7 = spearman([r["retention_gm12"] for r in trace],
                    [r["site_span_strength"] for r in trace])

    transient_peaks = [r["steps"] for r in trace
                       if 0 < r["steps"] < CKPT_STEPS[-1]
                       and r["site_pos_span"] and r["retention_gm12"] >= TRANSIENT_RET_BAR
                       and ret300 < r["retention_gm12"] - TRANSIENT_DECAY]

    k = 0
    while k < len(ret_seq) and ret_seq[k] >= DELAYED_HI:
        k += 1
    delayed_prefix_steps = ck_steps[:k]
    prefix_flat = bool(k >= 2 and all(
        abs(ret_seq[i + 1] - ret_seq[i]) <= DELAYED_FLAT for i in range(k - 1)))
    post_prefix = ret_seq[k:]
    cliff = bool(any(r <= DELAYED_LO for r in post_prefix)) if k < len(ret_seq) else False

    cond = {
        "CLEAN": {
            "successive_diffs": [round(d, 4) for d in diffs],
            "monotone": monotone, "monotone_slack": MONO_SLACK,
            "plateau_gaps": plateau_gaps, "plateau_gap_bar": PLATEAU_GAP,
            "max_plateau_gaps": MAX_PLATEAU_GAPS, "no_plateau": no_plateau,
            "spearman_6ck": rho6, "spearman_bar": SPEARMAN_BAR,
            "spearman_7pt_incl_root": rho7,
            "fires": bool(monotone and no_plateau and rho6 <= SPEARMAN_BAR)},
        "TRANSIENT": {
            "peaks_steps": transient_peaks,
            "retention_bar": TRANSIENT_RET_BAR, "decay_bar": TRANSIENT_DECAY,
            "site_pos_at_peak": {str(r["steps"]): r["site_pos_span"]
                                 for r in trace if r["steps"] > 0},
            "fires": bool(len(transient_peaks) > 0)},
        "DELAYED": {
            "prefix_steps": delayed_prefix_steps, "prefix_flat": prefix_flat,
            "hi_bar": DELAYED_HI, "flat_bar": DELAYED_FLAT,
            "cliff": cliff, "cliff_bar": DELAYED_LO,
            "fires": bool(prefix_flat and cliff)},
    }
    if cond["CLEAN"]["fires"]:
        verdict = "CLEAN-CONVERSION"
        clause = (f"g-12 retention decays monotonically with no plateau "
                  f"({['%.3f' % r for r in ret_seq]}, diffs "
                  f"{['%+.3f' % d for d in diffs]}, {plateau_gaps} plateau "
                  f"gap(s) <= {MAX_PLATEAU_GAPS}) and site growth "
                  f"anti-correlates (Spearman {rho6:.3f} <= {SPEARMAN_BAR}) "
                  f"— the conversion is a smooth handoff, no dwell.")
    elif cond["TRANSIENT"]["fires"]:
        verdict = "TRANSIENT-TWO-DOOR"
        clause = (f"checkpoint(s) {transient_peaks} hold BOTH doors — site "
                  f"content clears the e151 content bar AND g-12 retention "
                  f">= {TRANSIENT_RET_BAR} — followed by decay to "
                  f"{ret300:.3f} (>= {TRANSIENT_DECAY} fall): the cliff has "
                  f"a DWELL TIME; e151's P-b prediction was EARLY, not "
                  f"wrong.")
    elif cond["DELAYED"]["fires"]:
        verdict = "DELAYED-CONVERSION"
        clause = (f"g-12 stays flat >= {DELAYED_HI} over steps "
                  f"{delayed_prefix_steps} (within-prefix |dret| <= "
                  f"{DELAYED_FLAT}) then cliffs to <= {DELAYED_LO} — a "
                  f"critical mass of zero-variance steps.")
    else:
        verdict = "TEXTURE"
        clause = (f"no registered bar fired cleanly: monotone={monotone}, "
                  f"plateau_gaps={plateau_gaps}, Spearman={rho6:.3f}, "
                  f"transient_peaks={transient_peaks}, delayed prefix "
                  f"{delayed_prefix_steps} flat={prefix_flat} cliff={cliff}; "
                  f"retention trace {['%.3f' % r for r in ret_seq]}, site "
                  f"trace {['%.4f' % s for s in site_seq]} — numbers "
                  f"reported, no bar shopping.")
    log("=" * 78)
    log(f"E152 VERDICT: {verdict}")
    log(f"  g-12 retention trace: " + " -> ".join(
        f"s{r['steps']}:{r['retention_gm12']:.3f}" for r in trace))
    log(f"  site content (span band-max): " + " -> ".join(
        f"s{r['steps']}:{r['site_span_strength']:+.4f}" for r in trace))
    log(f"  monotone={monotone} plateau_gaps={plateau_gaps} "
        f"Spearman(6ck)={rho6:.3f} transient_peaks={transient_peaks} "
        f"delayed_prefix={delayed_prefix_steps}")
    log(f"  {clause}")
    log("=" * 78)

    # ---- e151 reproduction check (the 300-step endpoint vs stored cells)
    last = trace[-1]
    repro_cells = {
        "base_gm12": last["base_gm12"], "base_g0": last["base_g0"],
        "base_gp12": last["base_gp12"], "ce_r": last["ce_r"],
        "site_span_strength": last["site_span_strength"],
        "site_onset_strength": last["site_onset_strength"],
        "A129": last["A129"], "row0_strength": last["row0_strength"],
        "dall_g0": last["dall_g0"], "site_read_onset": last["site_read_onset"],
    }
    repro_diffs = {k: repro_cells[k] - E151_AFTER[k] for k in repro_cells}
    max_abs = max(abs(v) for v in repro_diffs.values())
    reproduction_e151 = {
        "steps_compared": last["steps"], "cells": repro_cells,
        "e151_stored": E151_AFTER, "diffs": repro_diffs,
        "max_abs_diff": max_abs, "tol": G_FALLBACK_TOL,
        "bit_tol": G_BIT_TOL,
        "bit_reproducible": bool(max_abs < G_BIT_TOL),
        "pass": bool(max_abs < G_FALLBACK_TOL),
        "note": "sequential continuation = e151's own protocol; residual "
                "differences are GPU float nondeterminism (e119 precedent).",
    }
    log(f"e151 reproduction (s{last['steps']}): max|diff| {max_abs:.2e} "
        f"(tol {G_FALLBACK_TOL}, bit {G_BIT_TOL}): "
        f"{'PASS' if reproduction_e151['pass'] else 'FAIL'}")

    # ---------------- outputs
    metrics = {
        "experiment": "e152_conversion_trace",
        "date": common.now_iso(),
        "registration": ("QUEUE e152 row + T088 (2)/(3), dispatched 10:10Z "
                         "with verbatim bars; operationalizations frozen in "
                         "the module docstring before compute"),
        "registered_prediction": REGISTERED_PREDICTION,
        "committed_prediction": None,
        "question": ("where in 8-300 locked re-teach steps does the phase "
                     "convert, and does a TRANSIENT two-door state (site "
                     "content present AND g-12 retention >= 0.5) exist "
                     "mid-conversion?"),
        "root": f"runs/checkpoints/{ROOT_CK} (loaded, gated bit-exact vs "
                f"e151 stored root cells)",
        "continuation_choice": {
            "choice": "sequential continuation",
            "detail": "ONE {FT_STEPS}-step training from the root (seed "
                      "{seed}, generator never reset), CPU state-dict "
                      "snapshots at steps {steps}. This IS e151's protocol "
                      "(one continuous 300-step fine-tune); an independent "
                      "N-step run with the same seed coincides with the "
                      "first N steps BY CONSTRUCTION (per-step randint draws "
                      "from one seed-fixed generator; in-loop CPU evals "
                      "consume no RNG), so the designs differ only by GPU "
                      "float nondeterminism — quantified in "
                      "reproduction_e151. The dispatch's between-trainings "
                      "cooldowns are moot (one trajectory, uninterruptible); "
                      "e151's cooldown(90) before/after kept.".format(
                          FT_STEPS=FT_STEPS, seed=RETEACH_SEED,
                          steps=list(CKPT_STEPS)),
        },
        "reteach": {"desc": "e151/e143 locked-replay protocol verbatim at "
                            "offset +54 (FAR-style): ZEPHYRA locked at "
                            "x-cols 184..190, onset read row 183, name-only "
                            "7-target mask, zero position variance",
                    "recipe": "batch 32 = 16 pool + 16 anchors (8 paired + 8 "
                              "random), token-weighted union CE, AdamW "
                              "(0.9,0.95) wd 0.1, lr 1e-3 constant, clip 1.0",
                    "steps_ran": reteach["steps_ran"], "seed": RETEACH_SEED,
                    "ckpt_steps": list(CKPT_STEPS),
                    "missing_checkpoints": missing,
                    "traj": reteach["traj"], "device": str(DEV),
                    "time_cap_s": FT_TIME_CAP,
                    "cooldown_s": COOLDOWN_S},
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
                     "placement": {"offset": RETEACH_J,
                                   "name_xcols": [SITE_Z_XCOL,
                                                  SITE_Z_XCOL + 6],
                                   "read_rows": [183, 189],
                                   "site_band": list(SITE_ROWS),
                                   "continuation_len": SITE_CONT},
                     "mask": "7 name-char targets per window (name-only)",
                     "geometries_measured": {"novel": [-12, 12],
                                             "trained_g0": 0,
                                             "note": "root trained "
                                                     "{-8,-4,0,+4,+8}; +-12 "
                                                     "novel for it"}},
        "gates": {"G_SPLICE": G_SPLICE, "G_GEO": G_GEO, "G_CONS": G_CONS,
                  "G_MASK": G_MASK, "G_ROW0": G_ROW0, "G_A129": G_A129,
                  "G_DALL": G_DALL, "G_SURG": gates_surg,
                  "gpu_at_start": {"use_gpu": _USE_GPU,
                                   "status": gpu_status()}},
        "trace": trace,
        "trace_summary": {
            "steps": steps_meas,
            "retention_gm12": [r["retention_gm12"] for r in trace],
            "retention_gp12": [r["retention_gp12"] for r in trace],
            "retention_g0": [r["retention_g0"] for r in trace],
            "site_span_strength": [r["site_span_strength"] for r in trace],
            "row183_span_strength": [r["row183_span_strength"] for r in trace],
            "A129": [r["A129"] for r in trace],
            "dall_g0": [r["dall_g0"] for r in trace],
            "ce_r": [r["ce_r"] for r in trace],
        },
        "batteries": batteries,
        "adjudication": {"conditions": cond, "verdict": verdict,
                         "clause": clause},
        "reproduction_e151": reproduction_e151,
        "honesty_reflex": {
            "continuation_vs_independent": "sequential continuation and "
                "independent same-seed runs coincide BY CONSTRUCTION here "
                "(seed-fixed generator draws; RNG-free evals; optimizer "
                "replays identically), so the choice costs nothing in "
                "protocol fidelity — but the six checkpoints remain ONE "
                "trajectory: step effects and trajectory idiosyncrasy are "
                "confounded (n=1 path through step-space), and any "
                "non-monotonicity observed is this trajectory's, not a "
                "replicated law",
            "checkpoint_interference": "snapshots deep-copy weights OUT of "
                "the training net only; in-loop evals run on the CPU twin "
                "and consume no RNG — the trajectory is e151's up to GPU "
                "float nondeterminism (reproduction_e151 prices it)",
            "single_seed_single_lineage": "one trajectory (seed 10902) on "
                "one root (e131 consolidated, one seed); the dwell time / "
                "threshold, if found, is a point estimate",
            "reproduction_tolerance": "the 300-step endpoint is compared to "
                "e151's stored AFTER cells at the lab's 0.05 tolerance with "
                "the 5e-6 bit flag; GPU float nondeterminism precedent e119 "
                "(~1e-2 cell drift)",
            "sink_coupling_caveat_T087_E150": "g-12 retention conflates "
                "'the route survived' with 'sink health at the novel "
                "geometry survived' (T087's E150-applied caveat); the "
                "per-checkpoint mask (information) and ladder (poison) "
                "columns co-report the split",
            "site_content_noise_floor": "span census strengths at early "
                "checkpoints sit near the ~1e-4 control noise floor (the "
                "2x-control bar moves with each census); site_pos at early "
                "steps is fragile — the row-183 co-report and the strength "
                "trajectory (not the boolean alone) carry the reading",
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

    plot(rd / "conversion_trace.png", trace, batteries, cond, verdict, clause,
         reproduction_e151)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'conversion_trace.png'}, "
        f"ckpts runs/checkpoints/e152_steps{{{','.join(str(s) for s in sorted(reteach['sds']))}}}.pt")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot

def plot(path, trace, batteries, cond, verdict, clause, repro):
    """THE figure: retention and site-content vs steps (dual axis), any
    plateau/transient marked; plus spectra, old-trace, and verdict panels."""
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 10.5))
    xs_all = [r["steps"] for r in trace]
    xp_all = [7.0 if s == 0 else float(s) for s in xs_all]   # root placed at 7
    ck = [r for r in trace if r["steps"] > 0]
    xp_ck = [float(r["steps"]) for r in ck]

    # (0,0) THE trace: retention (left) vs site content (right)
    ax = axes[0, 0]
    for val, lbl, col, mk, lw in (
            ([r["retention_gm12"] for r in trace], "g-12 retention (HEADLINE)",
             "crimson", "o", 2.4),
            ([r["retention_gp12"] for r in trace], "g+12 retention",
             "darkorange", "s", 1.1),
            ([r["retention_g0"] for r in trace], "g0 retention",
             "dimgray", "^", 1.1)):
        ax.plot(xp_all, val, mk + "-", ms=7 if "HEADLINE" in lbl else 5,
                lw=lw, color=col, label=lbl,
                alpha=1.0 if "HEADLINE" in lbl else 0.8)
    for yv, col, lbl in ((DELAYED_HI, "tab:blue", "0.80 DELAYED flat bar"),
                         (TRANSIENT_RET_BAR, "seagreen", "0.50 TRANSIENT bar"),
                         (0.20, "gray", "0.20 e151 collapse bar")):
        ax.axhline(yv, ls="--", lw=1.0, color=col, alpha=0.75, label=lbl)
    axr = ax.twinx()
    axr.plot(xp_all, [r["site_span_strength"] for r in trace], "D-.",
             ms=6, lw=1.8, color="seagreen", label="site content: band-max "
             "strength (span census)")
    axr.plot(xp_all, [r["row183_span_strength"] for r in trace], "v:",
             ms=5, lw=1.0, color="mediumseagreen", alpha=0.85,
             label="row-183 strength (co-report)")
    axr.plot(xp_all, [r["site_span_bar_2x_control"] for r in trace], "x--",
             ms=5, lw=1.0, color="k", alpha=0.6,
             label="2x-control content bar (per checkpoint)")
    for r, x in zip(ck, xp_ck):
        if r["site_pos_span"]:
            axr.plot([x], [r["site_span_strength"]], marker="*", ms=17,
                     color="gold", mec="k", mew=0.7, zorder=5)
    if cond["TRANSIENT"]["peaks_steps"]:
        pk = min(cond["TRANSIENT"]["peaks_steps"])
        ax.axvspan(pk, max(xp_ck), color="gold", alpha=0.08)
        ax.annotate("TRANSIENT dwell", (np.sqrt(pk * max(xp_ck)), 0.72),
                    ha="center", fontsize=9, color="goldenrod",
                    fontweight="bold")
    if cond["DELAYED"]["prefix_steps"] and cond["DELAYED"]["prefix_flat"]:
        p0, p1 = cond["DELAYED"]["prefix_steps"][0], cond["DELAYED"]["prefix_steps"][-1]
        ax.plot([p0, p1], [0.84, 0.84], lw=5, color="tab:blue",
                solid_capstyle="butt", alpha=0.6)
        ax.annotate("flat prefix", (np.sqrt(p0 * p1), 0.86), ha="center",
                    fontsize=9, color="tab:blue")
    ax.set_xscale("log")
    ax.set_xlim(6, 400)
    ax.set_xticks(xp_ck)
    ax.set_xticklabels([str(int(x)) for x in xp_ck])
    ax.set_xlabel("locked re-teach steps (sequential continuation; * = site "
                  "content clears the bar; root plotted at left)")
    ax.set_ylabel("retention (vs root: g-12 0.9156 / g+12 0.9478 / g0 0.7850)")
    axr.set_ylabel("site content @183 (census strength = min(mean-drop, "
                   "zero-drop))")
    ax.set_ylim(-0.03, 1.1)
    axr.set_ylim(-0.005, max(0.08, 1.25 * max(
        r["site_span_strength"] for r in trace)))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.0, loc="center left")
    ax.set_title("THE CONVERSION TIME-TRACE — retention (left) vs site "
                 f"content (right)\nSpearman(g-12 ret, site) = "
                 f"{cond['CLEAN']['spearman_6ck']:.3f} (6ck) / "
                 f"{cond['CLEAN']['spearman_7pt_incl_root']:.3f} (7pt) — "
                 f"verdict: {verdict}", fontsize=10)

    # (0,1) site census spectra across checkpoints (span readout)
    ax = axes[0, 1]
    cmap = plt.cm.coolwarm
    n_lines = len(trace)
    for i, r in enumerate(trace):
        b = batteries["root" if r["steps"] == 0 else str(r["steps"])]
        cen = b["census183_span"]["rows"]
        xs_r = sorted(int(k) for k in cen
                      if int(k) not in SHARED_CTR and int(k) >= 100)
        col = "k" if r["steps"] == 0 else cmap(i / max(1, n_lines - 1))
        ax.plot(xs_r, [cen[str(x)]["strength"] for x in xs_r], "-o",
                ms=3.5, lw=1.3 if r["steps"] in (0, CKPT_STEPS[-1]) else 0.9,
                color=col, alpha=0.95,
                label=f"root" if r["steps"] == 0 else
                      (f"s{r['steps']}" if r["steps"] in
                       (CKPT_STEPS[0], CKPT_STEPS[-1]) else None))
    ax.axvspan(183, 189, color="crimson", alpha=0.08)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("wpe row (red shade: site band 183-189)")
    ax.set_ylabel("strength (span census)")
    ax.set_title("(ii) SITE-CONTENT CENSUS spectra — root (black) through "
                 f"s{CKPT_STEPS[-1]} (warm)", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")

    # (1,0) old trace + wreckage guard vs steps
    ax = axes[1, 0]
    ax.plot(xp_all, [r["A129"] for r in trace], "o-", ms=6, lw=1.8,
            color="tab:purple", label="A(129) address-key strength")
    ax.plot(xp_all, [r["dall_g0"] for r in trace], "s-", ms=6, lw=1.8,
            color="tab:blue", label="D-all g0 (deletion tolerance)")
    ax.plot(xp_all, [r["row0_strength"] for r in trace], "^-", ms=6, lw=1.5,
            color="tab:cyan", label="row-0 strength (sink)")
    ax.plot(xp_all, [r["d183_g0"] for r in trace], "d--", ms=5, lw=1.0,
            color="crimson", alpha=0.8, label="D-183 g0 (new door inert?)")
    axr = ax.twinx()
    axr.plot(xp_all, [r["ce_r"] for r in trace], "k:o", ms=5, lw=1.2,
             alpha=0.8, label="CE_R (wreckage guard)")
    ax.set_xscale("log")
    ax.set_xlim(6, 400)
    ax.set_xticks(xp_ck)
    ax.set_xticklabels([str(int(x)) for x in xp_ck])
    ax.set_xlabel("locked re-teach steps")
    ax.set_ylabel("strength / p(Z)")
    axr.set_ylabel("CE_R")
    ax.axhline(0, color="k", lw=0.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="center right")
    ax.set_title("(iii)+(iv)+(v) old trace dissolves as the site grows — "
                 "CE_R stays flat (conversion, not damage)", fontsize=10)

    # (1,1) verdict panel
    ax = axes[1, 1]
    ax.axis("off")
    tr_txt = "  ".join(
        f"s{r['steps']}:{r['retention_gm12']:.3f}" for r in trace)
    site_txt = "  ".join(
        f"s{r['steps']}:{r['site_span_strength']:+.4f}" for r in trace)
    vlines = [
        "REGISTERED (QUEUE e152 verbatim; frozen operationalizations):",
        "  CLEAN: monotone decay, no plateau, Spearman <= -0.8",
        "  TRANSIENT: site clears bar AND g-12 >= 0.5 mid-run, then decay",
        "  DELAYED: flat >= 0.8 until a threshold, then cliff",
        "",
        "TRACE:",
        f"  g-12 retention: {tr_txt}",
        f"  site content:   {site_txt}",
        f"  diffs: {cond['CLEAN']['successive_diffs']}  "
        f"plateau_gaps={cond['CLEAN']['plateau_gaps']}",
        f"  monotone={cond['CLEAN']['monotone']} "
        f"no_plateau={cond['CLEAN']['no_plateau']} "
        f"Spearman={cond['CLEAN']['spearman_6ck']:.3f}",
        f"  transient peaks={cond['TRANSIENT']['peaks_steps']}  "
        f"delayed prefix={cond['DELAYED']['prefix_steps']} "
        f"flat={cond['DELAYED']['prefix_flat']} cliff={cond['DELAYED']['cliff']}",
        f"  A(129): " + " ".join(f"{r['A129']:+.3f}" for r in trace),
        f"  D-all g0: " + " ".join(f"{r['dall_g0']:.3f}" for r in trace),
        f"  CE_R: " + " ".join(f"{r['ce_r']:.3f}" for r in trace),
        "",
        f"E151 REPRODUCTION (s{repro['steps_compared']}): max|diff| "
        f"{repro['max_abs_diff']:.2e} "
        f"({'PASS' if repro['pass'] else 'FAIL'} at {repro['tol']}; "
        f"bit {'YES' if repro['bit_reproducible'] else 'no'})",
        "",
        f"VERDICT: {verdict}",
    ] + [f"  {wd}" for wd in
         [clause[i:i + 78] for i in range(0, len(clause), 78)]]
    for i, tx in enumerate(vlines):
        ax.text(0.02, 0.97 - i * 0.040, tx, fontsize=7.3, va="top",
                family="monospace")

    fig.suptitle(f"E152 — THE CONVERSION TIME-TRACE (root "
                 f"e131_consolidated_e113 + locked replay at 183, sequential "
                 f"checkpoints {list(CKPT_STEPS)}) -> {verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
