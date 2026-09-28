"""E184 — THE SEED REPLICATES: the lead finding's final clause (T111).

WHY (T111's discriminator, verbatim): "the dissolution's invariance makes
the seed question CHEAPER than feared (three streams already triangulate
the noise), but the lab's own >=3 rule stands." Every wash in the session's
lead finding — extinction (e176), neutral (e176N arm A), filtered (e183),
both lrs — ran seed 10902. e152R showed seed-to-seed wash timing spans an
order of magnitude on related washes; mechanisms are samples. This cell
runs the NEUTRAL wash (e176N arm A — the cleanest stream: 0/16 junctions,
0/16 host content) at TWO additional seeds (10903, 10904; the e152R seed
convention: adjacent to the lineage's 10902, distinct across cells).

REGISTERED BARS (frozen here before compute; the dispatch's registration
verbatim; no bar shopping — adjudicate against exactly this):
  - ALL-DISSOLVE = both new seeds under the 0.27 bar by +50 (the lead
    finding's evidence completes at n=3 across seeds).
  - ANY-SURVIVOR = any seed >= 0.5 through +300 (re-opened: seed-dependent
    persistence exists — the lottery includes winners).

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * g-12 / g0 / g+12 / held30 = ABSOLUTE install-60 / held-30 battery
    mean p(Z) at ctx offsets -12 / 0 / +12 (e176/e176N's convention
    verbatim — the same batteries, the same corpus rebuild, the same ruler).
  * per seed: DISSOLVES = g-12 at the +50 checkpoint <= 0.27 (the
    dispatch's "under the 0.27 bar by +50"); the earliest checkpoint
    <= 0.27 over {1,2,4,50,100,200,300} is CO-REPORTED as the seed's
    steps-to-under-bar (the timing lottery's readout).
  * per seed: SURVIVES = g-12 >= 0.50 at EVERY continuation checkpoint
    {1,2,4,50,100,200,300} (the dispatch's "through +300"; step 0 = the
    root, 0.9156, trivially above).
  * the two per-seed clauses are disjoint (+50 <= 0.27 vs +50 >= 0.50).
  * ALL-DISSOLVE = DISSOLVES fires for BOTH new seeds (10903 AND 10904);
    seed 10902's cell is already dissolved on disk (e176n trace_armA +50
    0.0221 — embedded, verified vs file) and merely joins the n=3 table.
  * ANY-SURVIVOR = SURVIVES fires for ANY of the three seeds {10902,
    10903, 10904}.
  * composite verdict order: ALL-DISSOLVE -> ANY-SURVIVOR -> MIXED (a new
    seed lands in the gap band: above 0.27 at +50 yet not >= 0.5 at every
    checkpoint; full trajectories reported, no bar shopping).
  * THE TWO-STEP CLOCK (co-report, not a bar): seed 10902 went under the
    bar at +2 (0.6780 -> 0.0271). The clock REPLICATES if both new seeds'
    steps-to-under-bar <= 2; any slower seed is reported as seed-textured
    timing (e152R's order-of-magnitude prior makes this the live question
    the bars do not adjudicate).
  * CE_R (+ the in-batch corpus CE) reported at every checkpoint; the
    CE-at-first-under-bar is highlighted per seed (e176N's R50 convention).

DESIGN: e176N arm A VERBATIM at seeds 10903 and 10904 — the neutral bank
is e170's FIXED construction (16 plain-corpus windows, RNG seed 170,
rejection on FLORIZEL/ELIZABETH/ZEPH/MIRABEL in [s, s+257); 0/16 host
content, 0/16 junctions); batch 32 = 16 neutral-anchor draws + 16 random
corpus windows, full-token CE (NO fact windows, NO name tokens, NO mask),
AdamW (0.9,0.95) wd 0.1 constant lr 1e-3 clip 1.0. ONLY THE SEED CHANGES:
the per-step draws (aj = randint(16,(16,)) anchor picks + rj =
randint(len-BLOCK-1,(16,)) random offsets) come from
torch.Generator().manual_seed(seed) — a CPU generator, device-independent
— so each seed sees a different SAMPLING ORDER over the SAME neutral bank
plus different random windows: exactly the lottery being tested.
Checkpoints {1,2,4} (the fine-step smoke schedule, measured in the MAIN
run — e176N's convention) + {50,100,200,300}; full dial set + CE_R per
checkpoint.

COMPUTE ENVELOPE (dispatch): GPU ALLOWED — strict pre-training quick check
per training (gpu_status()/gpu_ok(): util <= 85 AND temp <= 80 C, plus the
lab's mem-headroom guard <= 85% of total; double-poll 5 s apart), PARK-ONCE
to CPU on any failure (no re-probing, never contention with a returning
user; e152R's policy verbatim). MID-RUN contention guard: every 25 steps
of a GPU training, re-poll; mem > 85% of total or temp > 80 C -> migrate
net + optimizer state to CPU and FINISH THERE (the dispatch's e152
precedent; any device mixing recorded per cell). cooldown(90 s) before and
after EACH training; caps 180 s (GPU) / 1500 s (CPU) per training; ALL
readouts CPU-side; sequential; NO concurrent GPU. Torch threads 8 (e152R/
e143 convention). Nets are the mandated 2.7M e131_consolidated line (the
dispatch's '<=1M family' note is an envelope statement — e143/e151/e152/
e158/e176n precedent; every gate reference and lineage number of these
cells lives on the 2.7M line).

INSTRUMENT PROVENANCE: load_cpu/evl_load/battery_cell/battery_pz/
ce_fixed_cpu/val_windows/deleted_wpe/read_fact_at/row_census/finetune_
freeze/measure()/flat_cells/gate_vs are lab/e176n_neutral_wash.py VERBATIM
(the e176/e161/e152/e151/e143/e131/e119/e113/e068/e065/e043 lineage;
finetune_freeze gains the device parameter + mid-run guard — per-step
arithmetic and the CPU-generator RNG draw sequence otherwise unchanged);
the neutral bank + junction accounting are lab/e170_anchor_neutral.py
VERBATIM via e176n's copy; pick_dev is lab/e152r_reseeds.py's quick-check
park-once policy adapted to the lab's gpu_ok(). The measure dial is
e176n's measure() (already minus the 183-span census — e178's recorded
deviation). Copied, not imported, to own the device policy.

NETS: root runs/checkpoints/e131_consolidated_e113.pt (gate bit-exact vs
e151's stored before-cells, e176n's G_ROOT set). Seed 10902's trajectory =
e176n's stored arm-A trace (embedded, verified vs file at plot time). New
checkpoints: runs/checkpoints/e184_neutral_s{10903,10904}.pt (+ _s{1,2,4,
50,100,200} intermediates) + smoke_* twins.

Outputs: runs/e184/{metrics.json, seed_replicates.png}; checkpoints
runs/checkpoints/e184_*.pt. No NOTES/THINKING/QUEUE/STATE edits; single
commit, no push.

Run:  cd lab && python e184_seed_replicates.py    (E184_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

torch.set_num_threads(8)                              # e152R/e143 convention

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT, cooldown,     # noqa: E402
                    gpu_ok, gpu_status, run_dir, save_json)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E184_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
ROOT_CK = "e131_consolidated_e113.pt"   # THE FULLY-CONSOLIDATED ROOT
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E176N_METRICS = E43.REPO / "runs" / "e176n" / "metrics.json"

# ---- e152's placement constants (the measurement instruments rebuild these) ----
RETEACH_J = 54                    # offset: name x-cols 184..190, read rows 183..189
SITE_ADDR_ROW = 183               # the onset read row (= e120's SPLICE_ADDR_ROW)
SITE_Z_XCOL = PRE + RETEACH_J     # 184 (= e120's Z_XCOL)
SITE_CONT = BLOCK - PRE - RETEACH_J - len(NAME)     # 65-token continuation
D_ALL = (121, 125, 129, 133, 137)                   # e113's fixed set
GEOS = (-12, 0, 12)               # novel x2 + the trained g0

# ---- census row set (e158/e161/e176 old-band convention; read-visible rows) ----
ROWS_OLD = (0, 1, 2, 3, 4, 5, 6, 60, 100, 118, 119, 120) + tuple(range(121, 130))
if SMOKE:
    ROWS_OLD = (0, 1, 60, 121, 125, 129)

# ---- the two replicate trainings -----------------------------------------------
CK_MAIN: tuple[int, ...] = (1, 2, 4, 50, 100, 200, 300) if not SMOKE else (1, 2, 4)
SEEDS: tuple[int, ...] = (10903, 10904)     # the e152R seed convention

# ---- fine-tune envelope (e176N arm A VERBATIM; only the seeds differ) ----------
LR = 1e-3                          # e176N arm A verbatim
GPU_CAP_S = 180.0                  # dispatch: <=180 s caps on GPU
CPU_CAP_S = 1500.0                 # e152R CPU calibration precedent
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
COOLDOWN_S = 90.0                  # dispatch: 60-120 s; e152R's 90
MIDRUN_POLL_EVERY = 25             # mid-run GPU contention poll cadence (steps)

# ---- e170's neutral anchor bank (e176N arm A's stream, VERBATIM) --------------
E170_ANCHOR_SEED = 170             # dedicated RNG for the plain-corpus bank
ANCHOR_FORBIDDEN = ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")

# ---- gates / references (full precision, = stored metrics) --------------------
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05

E151_ROOT = {                     # runs/e151 'before' battery (e176n's gate set)
    "base_gm12": 0.9155886769294739,
    "base_g0": 0.7850371599197388,
    "base_gp12": 0.9478210210800171,
    "ce_r": 1.663516640663147,
    "site_read_onset": 0.8898659348487854,
    "site_read_span": 0.982668936252594,
    "A129": -0.13237020391970877,
    "row0_strength": 0.7316772222270098,
    "dall_g0": 0.9047248959541321,
}

# e176N arm A's stored FULL-DIAL trajectory (seed 10902; runs/e176n/metrics.json
# trace_armA VERBATIM, re-verified at plot time) — the n=3 table's third member.
E176N_TRACE = {
    "seed": 10902,
    "freeze_steps": [0, 1, 2, 4, 50, 100, 200, 300],
    "gm12": [0.9155886173248291, 0.6780440807342529,
             0.027077054604887962, 0.010940761305391788,
             0.022122304886579514, 0.014922752045094967,
             0.00204725144430995, 0.0038193254731595516],
    "g0": [0.7850371599197388, 0.4619811177253723,
           0.1147073358297348, 0.06042749062152519,
           0.16835635900497437, 0.05249874293804169,
           0.011475668288767338, 0.016238771378993988],
    "gp12": [0.9478210210800171, 0.630972683429718,
             0.0140452291816473, 0.01570577546954155,
             0.051125336438417435, 0.016041822731904904,
             0.003007616149261594, 0.0074020992033183575],
    "held30_gm12": [0.6361417174339294, 0.4820752739906311,
                    0.0053826989606022835, 0.001151302014477551,
                    0.008003094233572483, 0.008265051059424877,
                    0.0012307859724387527, 0.003200115170329809],
    "held30_g0": [0.7233286499977112, 0.46198493242263794,
                  0.052937380969524384, 0.008801662363111973,
                  0.0722624883051407, 0.018951624631881714,
                  0.009389182542695274, 0.010610799305140972],
    "ce_r": [1.663516640663147, 2.2113420963287354,
             2.032074451446533, 1.8213403224945068,
             1.708834171295166, 1.6700899600982666,
             1.6468615531921387, 1.642844796180725],
    "site_read_onset": [0.8898658156394958, 0.5253196954727173,
                        0.004276960156857967, 0.01323858741670847,
                        0.03920356556773186, 0.006887354888021946,
                        0.003228061832487583, 0.0064509568673318],
    "site_read_span": [0.982668936252594, 0.9041284918785095,
                       0.25527775287628174, 0.8370789289474875,
                       0.7883875966072083, 0.7708680629730227,
                       0.6387351153281067, 0.6367783546447754],
    "A129": [-0.13237020391970877, -0.18496511500949658,
             0.06476628839979337, 0.03774139005108736,
             0.06142527761985549, 0.021514282444527135,
             0.0026059946973267262, 0.0024257128311243534],
    "row0_strength": [0.7316772222270098, 0.44394606062541586,
                      0.11226377164743061, 0.05891638144703677,
                      0.16746244587458062, 0.05185848949654896,
                      0.011329283959191792, 0.016176560471552647],
    "dall_g0": [0.9047248959541321, 0.6481041312217712,
                0.02991652674973011, 0.013727720826864243,
                0.07909166037838669, 0.027866331860423088,
                0.00567732285708189, 0.010615414804498192],
}

# e176N arm A's stored LIGHT in-run trajectory (corpus CE at checkpoints —
# the CE-at-dissolution co-report's in-batch column for seed 10902).
E176N_TRAJ = [
    {"step": 1, "corpus_ce": 1.356567621231079},
    {"step": 2, "corpus_ce": 1.8263245820999146},
    {"step": 4, "corpus_ce": 1.3631858825683594},
    {"step": 50, "corpus_ce": 0.6575877070426941},
    {"step": 100, "corpus_ce": 0.660504937171936},
    {"step": 200, "corpus_ce": 0.6505676507949829},
    {"step": 300, "corpus_ce": 0.6238144636154175},
]

# ---- registered bar constants (frozen; see OPERATIONALIZATIONS above) ---------
SHUT_BAR = 0.27                   # per-seed DISSOLVES by +50 (e158/e161/e176/e176N)
SURVIVE_BAR = 0.50                # per-seed SURVIVES through +300
TWO_STEP_N = 2                    # 10902's steps-to-under-bar (the clock)
ROOT_GM12 = E151_ROOT["base_gm12"]
ROOT_G0 = E151_ROOT["base_g0"]

REGISTERED_PREDICTION = {
    "all_dissolve": "ALL-DISSOLVE fires if: both new seeds (10903, 10904) "
        "under the 0.27 bar by +50 (the lead finding's evidence completes at "
        "n=3 across seeds).",
    "any_survivor": "ANY-SURVIVOR fires if: any seed (10902/10903/10904) "
        ">= 0.5 through +300 (re-opened: seed-dependent persistence exists — "
        "the lottery includes winners).",
    "two_step_clock": "CO-REPORT, not a bar: 10902 went under at +2 "
        "(0.6780 -> 0.0271); the clock replicates if both new seeds' "
        "steps-to-under-bar <= 2; slower timings are seed texture (e152R's "
        "order-of-magnitude prior).",
    "operationalizations": "g-12/g0/g+12/held30 = absolute install-60/held-30 "
        "battery mean p(Z) at ctx offsets -12/0/+12 (e176N's convention); "
        "per-seed DISSOLVES = g-12 at +50 <= 0.27 (earliest checkpoint <= 0.27 "
        "over {1,2,4,50,100,200,300} co-reported as steps-to-under-bar); "
        "per-seed SURVIVES = g-12 >= 0.50 at EVERY continuation checkpoint "
        "{1,2,4,50,100,200,300}; clauses disjoint; ALL-DISSOLVE = both new "
        "seeds DISSOLVE; ANY-SURVIVOR = any of the three SURVIVES; composite "
        "order ALL-DISSOLVE -> ANY-SURVIVOR -> MIXED; CE_R + in-batch corpus "
        "CE at every checkpoint, highlighted at first-under-bar.",
    "registration": "QUEUE has no e184 row at dispatch (commit 98b8b30 "
        "dispatched the bars in the mission text); this docstring freezes "
        "them verbatim before compute. Adjudicate against exactly this; no "
        "bar shopping.",
}

trims: list[str] = []
device_events: list[dict] = []
deviations: list[str] = [
    "GPU ALLOWED by dispatch (unlike e176n's forced CPU): strict pre-training "
    "quick check per training (gpu_ok() gates: util <= 85, temp <= 80 C, mem "
    "<= 85% of total; double-poll 5 s apart), PARK-ONCE to CPU on failure "
    "(e152R's policy verbatim); mid-run contention guard polls every "
    f"{MIDRUN_POLL_EVERY} steps and migrates net + optimizer state to CPU on "
    "mem/temp breach, finishing the training there (the dispatch's e152 "
    "precedent; any mixing recorded per cell in device_events).",
    "GPU float nondeterminism: seed 10902's stored trajectory ran CPU; if the "
    "new seeds run GPU (or park mid-run), devices mix ACROSS seeds — recorded "
    "per cell (e152R's precedent note; e158's two-pass scatter 0.546/0.458 is "
    "the sensitivity, and it lived at the bar — the wash collapses to "
    "~0.00-0.03, two orders below both bars, so device drift cannot flip a "
    "verdict unless a seed is genuinely near-bar; if one is, that is itself "
    "reported, not smoothed).",
    "finetune_freeze gains a device parameter + the mid-run guard vs e176n's "
    "verbatim copy; the per-step arithmetic and the CPU-generator RNG draw "
    "sequence are unchanged — at a given seed the draw sequence is "
    "device-independent by construction (torch.Generator, CPU).",
    "The original (extinction) anchor bank is NOT rebuilt (e176n's arm-B "
    "business): this cell's delta is the SEED, and the stream is e176n arm "
    "A's neutral bank verbatim (re-gated: 0/16 host content, 0/16 junctions).",
    "Nets are the mandated 2.7M e131_consolidated line (the dispatch's "
    "'<=1M family' note is an envelope statement; e143/e151/e152/e158/e176n "
    "precedent — every gate reference lives on this line).",
    "Eval thread count is 8 (e152R/e143 convention) vs e151's stored cells — "
    "CPU reduction order can drift low-order bits; the G_ROOT gate reports "
    "both the 5e-6 bit flag and the 0.05 fallback tolerance (e161/e176/e176n "
    "precedent).",
    "n=3 is three point estimates, one trajectory per seed (the >=3 rule "
    "satisfied at its floor); the timing lottery is read out at checkpoint "
    "resolution {1,2,4,...}, not per-step.",
    "Smoke mode trims: 4-step trainings per seed, checkpoints {1,2,4}, lean "
    "measures (no censuses, no deletion table), no cooldowns; nothing "
    "adjudicated.",
]


# ------------------------------------------------------------------ device pick
# PROVENANCE: lab/e152r_reseeds.py's pick_dev (quick check, PARK-ONCE) adapted
# to the lab's gpu_ok() gate; plus the dispatch's mid-run migration guard.

GPU_PARKED = False
PARK_REASON = None


def pick_dev(tag: str) -> torch.device:
    """Strict pre-training quick check (dispatch): gpu_ok() double-poll 5 s
    apart; PARK-ONCE — any failure parks every remaining training to CPU."""
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
    GPU_PARKED = True
    PARK_REASON = f"quick check failed: {s1}"
    log(f"[gpu] PARK — '{tag}' and all remaining trainings run CPU ({s1})")
    return CPU


def migrate_to_cpu(net, opt) -> None:
    """Move net + optimizer state to CPU in place (params persist, so the
    opt.state keys stay valid). The dispatch's mid-run contention exit."""
    net.to("cpu")
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state.get(p, {})
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to("cpu")


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/e176n_neutral_wash.py VERBATIM (see the module docstring).
# Copied rather than imported to own the device policy.

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
    """e131's read_fact_position VERBATIM ARITHMETIC (e151/e152/e161/e176/
    e176n/e178 copy): p(true name char) at positions addr_row..addr_row+6."""
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
                    g0_ids, zid: int, seed: int,
                    ckpt_steps: tuple[int, ...]):
    """THE NEUTRAL PLAIN-CORPUS FREEZE (e176n's finetune_freeze VERBATIM
    arithmetic, device-parameterized). Per step: aj = randint(16) anchor
    draws, rj = randint(16) random corpus offsets; batch 32 full-token CE;
    AdamW (0.9,0.95) wd 0.1 constant lr, clip 1.0. The draw sequence at a
    given seed is device-independent (CPU torch.Generator); only the seed
    (the sampling order over the FIXED neutral bank) differs across cells.
    Snapshots (deep-copy out, CPU) + light CPU evals (g-12, g0, CE_R — no
    RNG consumed) at the checkpoint steps; in-batch corpus CE recorded at
    every checkpoint and every 50. Mid-run GPU contention guard every
    MIDRUN_POLL_EVERY steps: mem/temp breach -> migrate to CPU and finish
    there (recorded in device_events)."""
    dev = pick_dev(tag)
    cap = GPU_CAP_S if dev.type == "cuda" else CPU_CAP_S
    ckpt_set = set(ckpt_steps)
    n_steps = ckpt_steps[-1]
    net = copy.deepcopy(net0).to(dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=LR, betas=(0.9, 0.95),
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
                     "status": s,
                     "note": "contention guard fired (mem > 85% or temp > 80C)"
                             " — net + optimizer state moved to CPU; training"
                             " finishes on CPU (e152 precedent)"})
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
            "lr": LR, "zeph_violations": zeph_checks,
            "initial_device": "cuda" if not GPU_PARKED else "cpu",
            "final_device": str(dev), "time_cap_s": cap}


CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "e184", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name} "
        f"({', '.join(f'{k}={v}' for k, v in meta.items())})")


def verify_e176n(embedded: dict, traj_light: list, path: Path) -> dict:
    """Verify the embedded e176n arm-A copies against the stored metrics file
    (no silent divergence; e176n's verify_ref convention)."""
    src = {"source": "embedded verbatim copy (runs/e176n trace_armA/traj)",
           "file_present": path.exists(), "verified_vs_embedded": None,
           "max_abs_diff": None}
    if path.exists():
        mm = json.loads(path.read_text(encoding="utf-8"))
        rows = mm["trace_armA"]
        diffs = []
        for r in rows:
            s = r["freeze_steps"]
            i = embedded["freeze_steps"].index(s)
            for k in ("gm12", "g0", "gp12", "held30_gm12", "held30_g0",
                      "ce_r", "site_read_onset", "site_read_span", "A129",
                      "row0_strength", "dall_g0"):
                diffs.append(abs(r[k] - embedded[k][i]))
        steps_ok = ([r["freeze_steps"] for r in rows]
                    == embedded["freeze_steps"])
        src["max_abs_diff"] = max(diffs) if diffs else None
        src["verified_vs_embedded"] = bool(steps_ok and max(diffs) < 1e-9)
        if src["verified_vs_embedded"]:
            src["source"] = ("runs/e176n/metrics.json trace_armA (embedded "
                             f"copy verified, max|diff| {max(diffs):.1e})")
        # the light traj's corpus CE (in-batch CE column for seed 10902)
        stored_traj = mm["armA_neutral"]["traj"]
        ce_ok = all(abs(a["corpus_ce"] - b["corpus_ce"]) < 1e-9
                    for a, b in zip(stored_traj, traj_light)) and \
            len(stored_traj) == len(traj_light)
        src["traj_corpus_ce_verified"] = bool(ce_ok)
    return src


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e184_smoke" if SMOKE else "e184")
    log(f"E184 THE SEED REPLICATES (smoke={SMOKE}) -> {rd}")
    log(f"compute: strict per-training GPU gate (park-once) + mid-run "
        f"contention guard every {MIDRUN_POLL_EVERY} steps; gpu at start: "
        f"{gpu_status()}, cpu threads {torch.get_num_threads()}")

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
    # THE NEUTRAL STREAM (e170's construction VERBATIM via e176n arm A):
    # 16 plain-corpus windows, rejection on host/nonce content in [s, s+257).
    # FIXED content (seed 170) — only the per-step SAMPLING seed differs.
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
    jc_neutral = sum(
        1 for s in n_starts
        if any(s <= p < s + BLOCK + 1 for p in host_positions))
    host_occ_total = len(host_positions)
    bg_rate = host_occ_total * (BLOCK + 1) / len(train_ids)
    G_ANCHOR = {
        "neutral_bank": {
            "construction": ("16 plain corpus windows from train_ids, RNG seed "
                             f"{E170_ANCHOR_SEED}, rejection if [s, s+257) "
                             "contains FLORIZEL/ELIZABETH/ZEPH/MIRABEL — "
                             "e170's construction VERBATIM (= e176n arm A's "
                             "stream; FIXED content, not reseeded)"),
            "n_windows": 16, "block": BLOCK, "seed": E170_ANCHOR_SEED,
            "starts": n_starts, "tries": tries, "rejections": rejections,
            "forbidden": list(ANCHOR_FORBIDDEN),
            "windows_with_host_content": sum(
                1 for s in n_starts
                if any(f in train_text[s: s + BLOCK + 1] for f in HOSTS)),
            "junctions_covered": jc_neutral,
        },
        "budget_identical_to_e176n": bool(anchor_neutral.shape == (16, BLOCK)),
        "rng_note": ("finetune_freeze verbatim; draw shapes/moduli identical "
                     "(n_anc=16, len(train_ids)); ONLY the generator seed "
                     "differs across cells (10902 stored / 10903 / 10904) — "
                     "same fixed bank, different sampling order + random "
                     "windows: the seed lottery under test"),
        "random_channel_background": {
            "host_occurrences_in_train": host_occ_total,
            "est_window_hit_rate": bg_rate,
            "note": ("the 16-per-batch random corpus windows are e161/e176/"
                     "e176n VERBATIM and unfiltered — identical background "
                     "(~3.8%/window), present in all three seeds; not part "
                     "of the delta"),
        },
    }
    G_ANCHOR["pass"] = bool(
        G_ANCHOR["neutral_bank"]["windows_with_host_content"] == 0
        and G_ANCHOR["neutral_bank"]["junctions_covered"] == 0
        and G_ANCHOR["budget_identical_to_e176n"])
    assert G_ANCHOR["pass"], f"anchor gate FAILED: {G_ANCHOR}"
    log(f"G_ANCHOR: neutral bank 16x{BLOCK} plain-corpus windows (seed "
        f"{E170_ANCHOR_SEED}, {rejections} rejections/{tries} tries) — host "
        f"content 0/16, junctions 0/16; random-channel background "
        f"~{100 * bg_rate:.1f}%/window: PASS")

    # ---------------- batteries: ctx = train_text[p-PRE-j : p] (e119 verbatim)
    bat_ids, held_ids = {}, {}
    for j in GEOS:
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
        hs = [train_text[p - PRE - j: p] for p, _ in held_occ]
        held_ids[j] = torch.stack([corpus.encode(c) for c in hs])
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    gm12_ids = bat_ids[-12]
    g0_ids = bat_ids[0]

    # ---------------- root net + gate vs e151
    net0 = load_cpu(CKPT_DIR / ROOT_CK)
    sd_root = {k: v.clone() for k, v in net0.state_dict().items()}
    root_meta = None
    st_raw = torch.load(CKPT_DIR / ROOT_CK, map_location="cpu",
                        weights_only=False)
    if isinstance(st_raw, dict) and "meta" in st_raw:
        root_meta = E43.jsonable(st_raw["meta"])
    log(f"root: {ROOT_CK} (meta: {root_meta})")

    gates_surg: dict = {}

    def measure(sd: dict, tag: str, lean: bool = False) -> dict:
        """e176n's measure() VERBATIM: base 3-geos + held30 + CE_R + site
        read + old-band census (row-0 sink / A129 brake) + deletion table
        (D-all, D-183). lean=True (smoke convention) drops the census +
        deletions, keeps a quick A(129)."""
        net = evl_load(sd)
        out: dict = {"tag": tag}
        out["base"] = {j: battery_cell(net, bat_ids[j], zid) for j in GEOS}
        out["base_held"] = {j: battery_cell(net, held_ids[j], zid)
                            for j in GEOS}
        out["ce_r"] = ce_fixed_cpu(net, *r_eval_xy)
        log(f"[{tag}] base: " + " ".join(f"g{j:+d} {out['base'][j]['mean_pz']:.4f}"
                                         for j in GEOS)
            + " | held30: " + " ".join(
                f"g{j:+d} {out['base_held'][j]['mean_pz']:.4f}" for j in GEOS)
            + f" | CE_R {out['ce_r']:.4f}")
        out["site_read"] = read_fact_at(net, pool_x, name_ids, zid,
                                        SITE_ADDR_ROW, SITE_Z_XCOL)
        log(f"[{tag}] site read @183: onset "
            f"{out['site_read']['pz_onset_mean']:.4f} span "
            f"{out['site_read']['pname_mean_over7']:.4f}")
        if not lean and not SMOKE:
            out["census_old"] = row_census(net, ROWS_OLD,
                                           lambda n: battery_pz(n, bat_ids[0],
                                                                zid))
            co = out["census_old"]["rows"]
            out["old_band"] = {
                "base_pz": out["census_old"]["base_readout"],
                "row0_strength": co["0"]["strength"],
                "A129": co["129"]["strength"],
                "band121_129_max": max(co[str(r)]["strength"]
                                       for r in range(121, 130)
                                       if str(r) in co)}
            log(f"[{tag}] old band: row0 S "
                f"{out['old_band']['row0_strength']:+.4f} | A(129) "
                f"{out['old_band']['A129']:+.4f}")
            DELS = {"d_all": D_ALL, "d183": (SITE_ADDR_ROW,)}
            out["del_table"] = {}
            for dl, rows_ in DELS.items():
                sd_d, gate = deleted_wpe(sd, rows_)
                gates_surg[f"{tag}__{dl}"] = gate
                if not gate["pass"]:
                    raise RuntimeError(f"deletion gate FAILED {tag}/{dl}: "
                                       f"{gate}")
                net.load_state_dict(sd_d)
                cell = {"g0": battery_cell(net, bat_ids[0], zid)["mean_pz"]}
                if dl == "d183":
                    cell["gm12"] = battery_cell(net, bat_ids[-12],
                                                zid)["mean_pz"]
                out["del_table"][dl] = cell
            net.load_state_dict(sd)
            log(f"[{tag}] deletions g0: " + " | ".join(
                f"{dl} {out['del_table'][dl]['g0']:.3f}" for dl in DELS))
        else:
            w = net.wpe.weight.data
            orig = w.clone()
            mean_row = orig.mean(0)
            bp = battery_pz(net, bat_ids[0], zid)
            w[129] = mean_row
            m129 = bp - battery_pz(net, bat_ids[0], zid)
            w.copy_(orig)
            w[129] = 0.0
            z129 = bp - battery_pz(net, bat_ids[0], zid)
            w.copy_(orig)
            assert torch.equal(w, orig), "lean A129 failed to restore wpe"
            out["A129_quick"] = float(min(m129, z129))
        del net
        return out

    def flat_cells(m: dict) -> dict:
        c = {"gm12": m["base"][-12]["mean_pz"],
             "g0": m["base"][0]["mean_pz"],
             "gp12": m["base"][12]["mean_pz"],
             "held30_gm12": m["base_held"][-12]["mean_pz"],
             "held30_g0": m["base_held"][0]["mean_pz"],
             "ce_r": m["ce_r"],
             "site_read_onset": m["site_read"]["pz_onset_mean"],
             "site_read_span": m["site_read"]["pname_mean_over7"]}
        if "old_band" in m:
            c["A129"] = m["old_band"]["A129"]
            c["row0_strength"] = m["old_band"]["row0_strength"]
            c["dall_g0"] = m["del_table"]["d_all"]["g0"]
            c["d183_g0"] = m["del_table"]["d183"]["g0"]
            c["d183_gm12"] = m["del_table"]["d183"]["gm12"]
        else:
            c["A129"] = m["A129_quick"]
        return c

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

    log("=" * 78)
    log("STEP-0 battery (root = e131_consolidated_e113; 'before')")
    root = measure(sd_root, "root", lean=SMOKE)
    root_cells = flat_cells(root)
    keymap = {"base_gm12": "gm12", "base_g0": "g0", "base_gp12": "gp12",
              "ce_r": "ce_r", "site_read_onset": "site_read_onset",
              "site_read_span": "site_read_span", "A129": "A129",
              "row0_strength": "row0_strength", "dall_g0": "dall_g0"}
    root_refs = {keymap[k]: v for k, v in E151_ROOT.items()}
    G_ROOT = gate_vs(root_cells, root_refs, "G_ROOT (vs e151 before-cells)")
    if not G_ROOT["pass"]:
        raise RuntimeError("consolidated-root gate FAILED vs e151 stored "
                           "before-cells")
    log("gates: G_SPLICE, G_NAMEFREE, G_POOL, G_ANCHOR, G_ROOT all PASS")

    # reference provenance (embedded copies verified vs the stored file)
    src_e176n = verify_e176n(E176N_TRACE, E176N_TRAJ, E176N_METRICS)
    log(f"e176n arm-A reference: {src_e176n['source']}")

    # =====================================================================
    # THE TWO SEED REPLICATES (e176N arm A verbatim; only the seed differs)
    # =====================================================================
    arms: dict = {}
    batteries_all: dict = {}
    for k, seed in enumerate(SEEDS):
        tag = f"s{seed}"
        log("=" * 78)
        if not SMOKE:
            log(f"[thermal] cooldown {COOLDOWN_S:.0f}s before {tag}")
            cooldown(COOLDOWN_S)
        log(f"SEED REPLICATE {k + 1}/2 — {tag}: {CK_MAIN[-1]}-step neutral "
            f"freeze (e176N arm A verbatim: batch {ANCH_BS} neutral + "
            f"{RAND_BS} random, full-token CE, lr {LR}), checkpoints "
            f"+{list(CK_MAIN)}")
        arm = finetune_freeze(tag, net0, anchor_neutral, train_ids, itos,
                              r_eval_xy, gm12_ids, g0_ids, zid, seed, CK_MAIN)
        if not SMOKE:
            log(f"[thermal] cooldown {COOLDOWN_S:.0f}s after {tag}")
            cooldown(COOLDOWN_S)
        G_DRAWFREE = {f"zeph_violations_{tag}": arm["zeph_violations"],
                      "pass": bool(arm["zeph_violations"] == 0)}
        assert G_DRAWFREE["pass"], f"{tag}: name token leaked into a window"
        gates_surg[f"G_DRAWFREE_{tag}"] = G_DRAWFREE
        arms[tag] = arm

        smax = max(arm["sds"])
        save_ckpt(f"e184_neutral_s{seed}", arm["sds"][smax],
                  {"desc": f"e131_consolidated_e113 + {smax}-step NEUTRAL-"
                           f"anchor plain-corpus freeze (e176N arm A "
                           f"protocol verbatim; e170's neutral bank seed "
                           f"{E170_ANCHOR_SEED}: 16 plain-corpus windows, "
                           f"0/16 junctions; batch 32 = 16 neutral anchors + "
                           f"16 random, full-token CE), lr {LR}, seed {seed}",
                   "steps": int(smax), "seed": seed, "lr": LR,
                   "anchor_bank": f"neutral (seed {E170_ANCHOR_SEED})",
                   "device": f"{arm['initial_device']}->{arm['final_device']}",
                   "base": f"runs/checkpoints/{ROOT_CK}"})
        for s in sorted(arm["sds"]):
            if s == smax:
                continue
            save_ckpt(f"e184_neutral_s{seed}_s{s}", arm["sds"][s],
                      {"desc": f"e131_consolidated_e113 + {s}-step NEUTRAL-"
                               f"anchor freeze (intermediate), seed {seed}",
                       "steps": int(s), "seed": seed, "lr": LR,
                       "anchor_bank": f"neutral (seed {E170_ANCHOR_SEED})",
                       "base": f"runs/checkpoints/{ROOT_CK}"})

        batteries = {"root": root}
        for s in sorted(arm["sds"]):
            log(f"{tag} +{s} battery")
            batteries[str(s)] = measure(arm["sds"][s], f"{tag}n{s}",
                                        lean=SMOKE)
        batteries_all[tag] = batteries

    # =====================================================================
    # TRAJECTORIES + ADJUDICATION (registered clauses; no shopping)
    # =====================================================================
    def trace_from(batteries: dict, steps: list) -> list:
        rows = []
        for s in steps:
            b = batteries["root" if s == 0 else str(s)]
            c = flat_cells(b)
            row = {"freeze_steps": s, **c,
                   "retention_vs_root_gm12": c["gm12"] / ROOT_GM12,
                   "retention_vs_root_g0": c["g0"] / ROOT_G0}
            rows.append(row)
        return rows

    traces = {}
    per_seed: dict = {}
    for tag, seed in [(f"s{s}", s) for s in SEEDS]:
        steps = [0] + sorted(arms[tag]["sds"])
        traces[str(seed)] = trace_from(batteries_all[tag], steps)
        seq = [r["gm12"] for r in traces[str(seed)]]
        ck_idx = [i for i, r in enumerate(traces[str(seed)])
                  if r["freeze_steps"] > 0]
        g50 = next((r["gm12"] for r in traces[str(seed)]
                    if r["freeze_steps"] == 50),
                   traces[str(seed)][-1]["gm12"] if SMOKE else None)
        earliest_under = next(
            (traces[str(seed)][i]["freeze_steps"] for i in ck_idx
             if seq[i] <= SHUT_BAR), None)
        per_seed[str(seed)] = {
            "seed": seed, "trace": traces[str(seed)],
            "g50": g50, "min_over_ckpts": min(seq[i] for i in ck_idx),
            "earliest_ck_le_bar": earliest_under,
            "DISSOLVES": (None if g50 is None else bool(g50 <= SHUT_BAR)),
            "SURVIVES": bool(min(seq[i] for i in ck_idx) >= SURVIVE_BAR),
            "traj_light": arms[tag]["traj"],
            "device": f"{arms[tag]['initial_device']}->"
                      f"{arms[tag]['final_device']}",
            "steps_ran": arms[tag]["steps_ran"],
            "missing_checkpoints": [s for s in CK_MAIN
                                    if s not in arms[tag]["sds"]],
        }

    # seed 10902's stored cells join the table (embedded, verified)
    seq02 = E176N_TRACE["gm12"]
    steps02 = E176N_TRACE["freeze_steps"]
    ck_idx02 = [i for i, s in enumerate(steps02) if s > 0]
    g50_02 = next((g for s, g in zip(steps02, seq02) if s == 50), None)
    earliest_under_02 = next((s for i, s in enumerate(steps02)
                              if i in ck_idx02 and seq02[i] <= SHUT_BAR), None)
    per_seed["10902"] = {
        "seed": 10902, "trace": "e176n trace_armA (embedded, verified)",
        "g50": g50_02, "min_over_ckpts": min(seq02[i] for i in ck_idx02),
        "earliest_ck_le_bar": earliest_under_02,
        "DISSOLVES": bool(g50_02 is not None and g50_02 <= SHUT_BAR),
        "SURVIVES": bool(min(seq02[i] for i in ck_idx02) >= SURVIVE_BAR),
        "traj_light": E176N_TRAJ, "device": "cpu->cpu (e176n, CPU-only)",
        "steps_ran": 300, "missing_checkpoints": [],
    }

    new_dissolve = [per_seed[str(s)]["DISSOLVES"] for s in SEEDS]
    all_dissolve = bool(all(d is True for d in new_dissolve)
                        and per_seed["10902"]["DISSOLVES"])
    any_survivor = bool(any(per_seed[str(s)]["SURVIVES"]
                            for s in (10902,) + SEEDS))

    steps_under = {str(s): per_seed[str(s)]["earliest_ck_le_bar"]
                   for s in (10902,) + SEEDS}
    clock_replicates = bool(
        all(steps_under[str(s)] is not None and steps_under[str(s)] <= TWO_STEP_N
            for s in SEEDS))

    # CE-at-dissolution per seed (e176N's R50 convention)
    def ce_at_dissolution(seed_key: str) -> dict:
        ps = per_seed[seed_key]
        step_u = ps["earliest_ck_le_bar"]
        if step_u is None:
            return {"dissolved": False}
        light = next((t for t in ps["traj_light"] if t["step"] == step_u),
                     None)
        if seed_key == "10902":
            row = {"gm12": E176N_TRACE["gm12"][
                E176N_TRACE["freeze_steps"].index(step_u)],
                   "ce_r": E176N_TRACE["ce_r"][
                E176N_TRACE["freeze_steps"].index(step_u)]}
        else:
            r = next(r for r in ps["trace"] if r["freeze_steps"] == step_u)
            row = {"gm12": r["gm12"], "g0": r["g0"], "ce_r": r["ce_r"]}
        return {"dissolved": True, "step": step_u, **row,
                "in_batch_corpus_ce": light["corpus_ce"] if light else None,
                "note": "CE_R + the in-batch corpus CE at the FIRST "
                        "checkpoint under the 0.27 bar"}

    ceD = {k: ce_at_dissolution(k) for k in [str(s) for s in (10902,) + SEEDS]}

    cond = {
        "ALL_DISSOLVE": {
            "bar": SHUT_BAR,
            "per_seed_g50": {k: per_seed[k]["g50"] for k in per_seed},
            "per_seed_dissolves": {k: per_seed[k]["DISSOLVES"]
                                   for k in per_seed},
            "fires": all_dissolve},
        "ANY_SURVIVOR": {
            "bar": SURVIVE_BAR,
            "per_seed_min_over_ckpts": {k: per_seed[k]["min_over_ckpts"]
                                        for k in per_seed},
            "per_seed_survives": {k: per_seed[k]["SURVIVES"] for k in per_seed},
            "fires": any_survivor},
        "TWO_STEP_CLOCK": {
            "steps_to_under_bar": steps_under,
            "replicates": clock_replicates,
            "bar": "co-report (not a bar): both new seeds <= +2"},
    }

    if all_dissolve:
        verdict = "ALL-DISSOLVE"
        clause = (f"both new seeds dissolved under the neutral wash: 10903 "
                  f"g-12 at +50 {per_seed['10903']['g50']:.4f}, 10904 "
                  f"{per_seed['10904']['g50']:.4f} (both <= {SHUT_BAR}; "
                  f"stored 10902: {per_seed['10902']['g50']:.4f}) — with "
                  f"e176n's stored cell the lead finding's evidence "
                  f"completes at n=3 across seeds: no seed of the "
                  f"consolidated line retains expression under continued "
                  f"training without the fact's windows; T111's seed clause "
                  f"discharges.")
    elif any_survivor:
        surv = [k for k in per_seed if per_seed[k]["SURVIVES"]]
        verdict = "ANY-SURVIVOR"
        clause = (f"seed-dependent persistence EXISTS: seed(s) {surv} held "
                  f"g-12 >= {SURVIVE_BAR} at EVERY continuation checkpoint "
                  f"through +300 — the lead finding RE-OPENS: the lottery "
                  f"includes winners; dissolution is not seed-invariant.")
    else:
        verdict = "MIXED"
        clause = ("neither registered bar fired: the new seeds landed in the "
                  "gap band (above the dissolve bar at +50 yet not above the "
                  "survive bar at every checkpoint) — full trajectories "
                  "reported, no bar shopping.")

    log("=" * 78)
    log(f"E184 VERDICT: {verdict}")
    for k in ("10902", "10903", "10904"):
        ps = per_seed.get(k, {})
        if ps:
            log(f"  seed {k}: g-12 trace "
                + (" -> ".join(f"+{r['freeze_steps']}:{r['gm12']:.4f}"
                               for r in ps["trace"])
                   if isinstance(ps["trace"], list) else "(stored: "
                   + " -> ".join(f"+{s}:{g:.4f}"
                                 for s, g in zip(steps02, seq02)) + ")")
                + f" | earliest <= {SHUT_BAR}: +{ps['earliest_ck_le_bar']} "
                f"| min {ps['min_over_ckpts']:.4f} "
                f"| DISSOLVES {ps['DISSOLVES']} SURVIVES {ps['SURVIVES']}")
    log(f"  steps-to-under-bar: {steps_under} | two-step clock replicates: "
        f"{clock_replicates}")
    log(f"  CE-at-dissolution: {ceD}")
    log(f"  {clause}")
    log(f"  device events: {device_events or 'none'}")
    log("=" * 78)

    # ---------------- outputs
    seed_table = {}
    for k in sorted(per_seed):
        ps = per_seed[k]
        gm12_seq = (E176N_TRACE["gm12"] if k == "10902"
                    else [r["gm12"] for r in ps["trace"]])
        st_seq = (E176N_TRACE["freeze_steps"] if k == "10902"
                  else [r["freeze_steps"] for r in ps["trace"]])
        seed_table[k] = {
            "seed": ps["seed"], "device": ps["device"],
            "steps": st_seq, "gm12": gm12_seq,
            "g50": ps["g50"], "min_over_ckpts": ps["min_over_ckpts"],
            "steps_to_under_bar": ps["earliest_ck_le_bar"],
            "DISSOLVES": ps["DISSOLVES"], "SURVIVES": ps["SURVIVES"],
            "ce_at_dissolution": ceD[k],
        }

    metrics = {
        "experiment": "e184_seed_replicates",
        "date": common.now_iso(),
        "registration": ("commit 98b8b30 dispatched the bars in the mission "
                         "text (no QUEUE row); frozen verbatim in the module "
                         "docstring before compute"),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("does the neutral wash's dissolution replicate across "
                     "seeds (e176N arm A verbatim at 10903/10904 vs the "
                     "stored 10902), or does the seed lottery include "
                     "winners?"),
        "root": f"runs/checkpoints/{ROOT_CK} (gated vs e151 before-cells, "
                f"max|diff| {G_ROOT['max_abs_diff']:.2e})",
        "root_meta": root_meta,
        "arms": {
            f"s{seed}": {
                "desc": "e176N arm A VERBATIM at a new seed: batch 32 = 16 "
                        "neutral-bank draws + 16 random corpus windows, "
                        "full-token CE (NO fact windows, NO name tokens, NO "
                        "mask), AdamW (0.9,0.95) wd 0.1 lr 1e-3 constant, "
                        "clip 1.0 — only the generator seed differs from "
                        "e176n's stored cell",
                "ckpt_steps": list(CK_MAIN),
                "steps_ran": arms[f"s{seed}"]["steps_ran"],
                "seed": seed, "lr": arms[f"s{seed}"]["lr"],
                "device": f"{arms[f's{seed}']['initial_device']}->"
                          f"{arms[f's{seed}']['final_device']}",
                "time_cap_s": arms[f"s{seed}"]["time_cap_s"],
                "traj": arms[f"s{seed}"]["traj"],
                "zeph_violations": arms[f"s{seed}"]["zeph_violations"],
                "missing_checkpoints": per_seed[str(seed)][
                    "missing_checkpoints"],
            } for seed in SEEDS
        },
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
                     "geometries_measured": {"novel": [-12, 12],
                                             "trained_g0": 0},
                     "neutral_bank": G_ANCHOR["neutral_bank"],
                     "measure_dial": "e176n's measure() (minus the 183-span "
                                     "census; e178's recorded deviation)"},
        "gates": {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                  "G_POOL": G_POOL, "G_ANCHOR": G_ANCHOR, "G_ROOT": G_ROOT,
                  "G_SURG": gates_surg},
        "traces": {k: v["trace"] for k, v in per_seed.items()
                   if isinstance(v["trace"], list)},
        "seed_comparison_table": seed_table,
        "trace_summary": {
            "steps": E176N_TRACE["freeze_steps"],
            "gm12_10902_stored": E176N_TRACE["gm12"],
            "gm12_10903": [r["gm12"] for r in per_seed["10903"]["trace"]],
            "gm12_10904": [r["gm12"] for r in per_seed["10904"]["trace"]],
            "g0_10902_stored": E176N_TRACE["g0"],
            "g0_10903": [r["g0"] for r in per_seed["10903"]["trace"]],
            "g0_10904": [r["g0"] for r in per_seed["10904"]["trace"]],
            "ce_r_10902_stored": E176N_TRACE["ce_r"],
            "ce_r_10903": [r["ce_r"] for r in per_seed["10903"]["trace"]],
            "ce_r_10904": [r["ce_r"] for r in per_seed["10904"]["trace"]],
        },
        "batteries": batteries_all,
        "references": {
            "e176n_armA": {
                "ref": {"seed": 10902, "desc": "the neutral wash's stored "
                         "n=1 (e176N arm A, CPU-only run)"},
                "provenance": src_e176n},
        },
        "ce_at_dissolution": ceD,
        "adjudication": {"conditions": cond, "verdict": verdict,
                         "clause": clause,
                         "two_step_clock_replicates": clock_replicates},
        "honesty_reflex": {
            "seed_is_the_delta": ("the neutral bank is FIXED (e170's seed-170 "
                                  "construction, re-gated 0/16 host content, "
                                  "0/16 junctions); ONLY the per-step "
                                  "sampling order over that bank + the random "
                                  "corpus windows differ across seeds "
                                  "(torch.Generator on CPU — device-"
                                  "independent draw sequences); the delta "
                                  "between cells is EXACTLY the seed"),
            "device_mixing": ("seed 10902 ran CPU (e176n was CPU-only by "
                              "dispatch); the new seeds run GPU when the "
                              "quick gate passes and PARK (or migrate "
                              "mid-run) when it does not — devices across "
                              "seeds: "
                              + "; ".join(f"{k}: {per_seed[k]['device']}"
                                          for k in per_seed)
                              + (f"; mid-run events: {device_events}"
                                 if device_events else "; no mid-run events")),
            "rng_divergence": ("different seeds diverge from step 1 (different "
                               "aj/rj draws); the trajectories are "
                               "INDEPENDENT samples of the wash process, not "
                               "perturbations of one path — the steps-to-"
                               "under-bar spread is the honest readout of the "
                               "timing lottery at checkpoint resolution"),
            "checkpoint_resolution": ("steps-to-under-bar is bracketed by the "
                                      "checkpoint set {1,2,4,50,...}; a seed "
                                      "under bar at +2 could have crossed "
                                      "anywhere in (1,2] — same resolution as "
                                      "the stored 10902 cell (its +1 read "
                                      "0.6780 brackets the crossing in "
                                      "(1,2])"),
            "single_root": ("all three trajectories start from the SAME "
                            "bit-gated root (e131_consolidated_e113, G_ROOT "
                            "PASS) — the seed lottery tested is the WASH "
                            "dynamics', not the install's"),
            "bars_anchored": ("the dissolve/survive bars are the same "
                              "absolute home-battery bars e158/e161/e176/"
                              "e176n used (0.27 SHUT / 0.50 SURVIVE) — all "
                              "three seeds measured on one ruler"),
        },
        "trims": trims, "device_events": device_events,
        "deviations": deviations,
        "ckpt_inventory": CKPT_INVENTORY,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": int(net0.num_params()),
                   "train_device": "gpu-gated (park-once) per training",
                   "eval_device": "cpu",
                   "torch_threads": torch.get_num_threads(),
                   "gpu_parked": GPU_PARKED, "park_reason": PARK_REASON,
                   "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "seed_replicates.png", per_seed, cond, verdict, clause,
         clock_replicates, seed_table, ceD)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'seed_replicates.png'}, "
        f"{len(CKPT_INVENTORY)} checkpoints in runs/checkpoints/e184_*.pt")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot

def plot(path, per_seed, cond, verdict, clause, clock_replicates,
         seed_table, ceD):
    """THE figure: the n=3 overlay (the headline), the whole-battery collapse
    across seeds, the fine-step zoom (the two-step clock), and the seed
    comparison table + verdict."""
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 10.5))
    cols = {"10902": "crimson", "10903": "seagreen", "10904": "darkorange"}
    lbls = {"10902": "seed 10902 (e176n stored, CPU)", "10903": "seed 10903",
            "10904": "seed 10904"}

    def seq(k, key):
        if k == "10902":
            return E176N_TRACE["freeze_steps"], E176N_TRACE[key]
        return ([r["freeze_steps"] for r in per_seed[k]["trace"]],
                [r[key] for r in per_seed[k]["trace"]])

    # (0,0) THE n=3 OVERLAY (g-12 headline)
    ax = axes[0, 0]
    for k in ("10902", "10903", "10904"):
        xs, ys = seq(k, "gm12")
        ax.plot(xs, ys, "o--" if k == "10902" else "o-", ms=8,
                lw=2.4 if k != "10902" else 1.8, color=cols[k],
                alpha=0.9 if k != "10902" else 0.75, label=lbls[k])
    for yv, col, lbl in ((SURVIVE_BAR, "seagreen",
                          f"{SURVIVE_BAR} SURVIVES bar (through +300)"),
                         (SHUT_BAR, "tab:purple",
                          f"{SHUT_BAR} DISSOLVES bar (by +50)")):
        ax.axhline(yv, ls="--", lw=1.1, color=col, alpha=0.8, label=lbl)
    ax.annotate(f"root g-12 {E176N_TRACE['gm12'][0]:.3f}", (0, E176N_TRACE["gm12"][0]),
                textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    ax.set_xlabel("plain-corpus freeze steps from the root (step 0 = root; "
                  "same neutral bank, different seeds)")
    ax.set_ylabel("absolute mean p(Z), install-60 battery")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7.2, loc="center right")
    ax.set_title(f"THE n=3 OVERLAY — the neutral wash across seeds "
                 f"-> {verdict}", fontsize=10)

    # (0,1) whole-battery collapse across seeds (g0 + held30 + CE_R)
    ax = axes[0, 1]
    for k in ("10902", "10903", "10904"):
        xs, ys = seq(k, "g0")
        ax.plot(xs, ys, "^-", ms=6, lw=1.6, color=cols[k], alpha=0.75,
                label=f"{lbls[k]} g0")
        xs, ys = seq(k, "held30_gm12")
        ax.plot(xs, ys, "D-", ms=4, lw=1.2, color=cols[k], alpha=0.45,
                label=f"{lbls[k]} held30 g-12")
    axr = ax.twinx()
    for k in ("10902", "10903", "10904"):
        xs, ys = seq(k, "ce_r")
        axr.plot(xs, ys, "k:o" if k == "10902" else "k:s", ms=4, lw=1.0,
                 alpha=0.35 if k != "10902" else 0.5)
    axr.set_ylabel("CE_R (k markers)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("plain-corpus freeze steps")
    ax.set_ylabel("mean p(Z)")
    ax.legend(fontsize=6.6, loc="center right")
    ax.set_title("the whole battery collapses in every seed (g0 / held30; "
                 "CE_R transient then recovery)", fontsize=9.5)

    # (1,0) the fine-step zoom (symlog): the two-step clock
    ax = axes[1, 0]
    for k in ("10902", "10903", "10904"):
        xs, ys = seq(k, "gm12")
        fine = [(s, g) for s, g in zip(xs, ys) if s <= 4]
        ax.plot([s for s, _ in fine], [g for _, g in fine],
                "o--" if k == "10902" else "o-", ms=10, lw=2.4,
                color=cols[k], label=lbls[k])
        for s, g in fine:
            if s > 0:
                ax.annotate(f"{g:.3f}", (s, g), textcoords="offset points",
                            xytext=(5, 6), fontsize=7.5, color=cols[k])
    ax.axhline(SHUT_BAR, ls="--", lw=1.0, color="tab:purple", alpha=0.8,
               label=f"{SHUT_BAR} bar")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel("freeze steps (zoom 0..4; symlog)")
    ax.set_ylabel("mean p(Z) (symlog)")
    ax.legend(fontsize=7.2, loc="lower left")
    ax.set_title(f"the two-step clock across seeds — replicates: "
                 f"{clock_replicates} (under-bar "
                 + ", ".join(f"{k}:+{seed_table[k]['steps_to_under_bar']}"
                             for k in sorted(seed_table)) + ")", fontsize=9.5)

    # (1,1) the seed comparison table + verdict panel
    ax = axes[1, 1]
    ax.axis("off")
    ytxt = 0.97
    ax.text(0.02, ytxt, "THE SEED COMPARISON (n=3, one ruler):",
            fontsize=8.5, va="top", family="monospace", weight="bold")
    ytxt -= 0.038
    ax.text(0.02, ytxt,
            "  seed   device      +1      +2      +4     +50    +300   "
            "under-bar  DISSOLV  SURVIV",
            fontsize=7.2, va="top", family="monospace")
    ytxt -= 0.032
    for k in sorted(seed_table):
        r = seed_table[k]
        def g(step):
            i = r["steps"].index(step)
            return f"{r['gm12'][i]:.4f}" if i < len(r["gm12"]) else "  n/a "
        try:
            g1, g2, g4 = g(1), g(2), g(4)
        except ValueError:
            g1 = g2 = g4 = "  n/a "
        try:
            g50, g300 = g(50), g(300)
        except ValueError:
            g50 = g300 = "  n/a "
        ub = r["steps_to_under_bar"]
        ax.text(0.02, ytxt,
                f"  {k}  {r['device'].split(' ')[0]:<10}  {g1}  {g2}  {g4}  "
                f"{g50}  "
                f"{g300}   +{ub if ub is not None else '-':<6}  "
                f"{str(r['DISSOLVES']):<7}  {str(r['SURVIVES']):<5}",
                fontsize=7.2, va="top", family="monospace",
                color=cols[k] if k != "10902" else "crimson")
        ytxt -= 0.03
    ytxt -= 0.02
    ax.text(0.02, ytxt, f"TWO-STEP CLOCK replicates: {clock_replicates} "
            f"(10902 crossed in (1,2])", fontsize=7.6, va="top",
            family="monospace")
    ytxt -= 0.04
    for k in sorted(ceD):
        c = ceD[k]
        if c.get("dissolved"):
            ibce = c['in_batch_corpus_ce']
            ax.text(0.02, ytxt,
                    f"  seed {k} under-bar +{c['step']}: g-12 {c['gm12']:.4f},"
                    f" CE_R {c['ce_r']:.3f}"
                    + (f", in-batch CE {ibce:.3f}" if ibce is not None else ""),
                    fontsize=7.0, va="top", family="monospace")
            ytxt -= 0.028
    ytxt -= 0.015
    ax.text(0.02, ytxt, f"E184 VERDICT: {verdict}", fontsize=9.0, va="top",
            family="monospace", weight="bold", color="darkred")
    ytxt -= 0.038
    for wd in textwrap.wrap(clause, width=84, break_long_words=False):
        ax.text(0.02, ytxt, f"  {wd}", fontsize=6.8, va="top",
                family="monospace")
        ytxt -= 0.027

    fig.suptitle("E184 — THE SEED REPLICATES: the lead finding's final "
                 f"clause (neutral wash, seeds 10902/10903/10904) -> "
                 f"{verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
