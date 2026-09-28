"""E185C — THE CPU-ONLY TAIL RE-RUN: R52's device-confound discharge.

WHY (T112/R52's verdict, verbatim finding): "the tail lottery is DEVICE-
confounded at the comparison points". e184's two seed replicates (10903/
10904) both started cuda and MIGRATED to CPU mid-run (s10903 at s175, s10904
at s275 — thermal guard, 83 C), so e184's tail checkpoints straddle devices:
+100 is GPU for both seeds, +200 is CPU(10903) vs GPU(10904) — a CROSS-
DEVICE comparison, and +300 is CPU for both. T112's tail sentence ("the
timing lottery lives in the DEPTH/TAIL, not the clock: seed 10904's tail
runs 2-34x slower") therefore rides numbers measured on mismatched
arithmetic. This cell re-runs BOTH seeds CPU END-TO-END (no device event
possible) under e184's exact protocol and adjudicates the tail ordering
against e184's stored mixed-device result.

REGISTERED BARS (frozen here before compute; the dispatch's registration
verbatim; no bar shopping — adjudicate against exactly this):
  - TAIL-REPRODUCES fires if: the tail ordering matches e184's (10904
    slower than 10903 at the tail checkpoints) — the lottery is real;
    T112's tail sentence stands.
  - TAIL-SHUFFLES fires if: the ordering changes or both tails converge —
    the tail sentence dies (a device artifact); T112's texture struck.
  - No bar shopping; texture (both die on the same clock regardless) =>
    note that the DISSOLUTION itself is device-robust either way.

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * the run: e184's protocol VERBATIM at seeds 10903 and 10904 — the same
    corpus rebuild, the same e170 neutral bank (seed 170; 0/16 host
    content, 0/16 junctions), the same batch 32 = 16 neutral draws + 16
    random corpus windows, full-token CE, AdamW (0.9,0.95) wd 0.1 constant
    lr 1e-3 clip 1.0, the same CPU torch.Generator draw sequence (device-
    independent), the same checkpoint grid {1,2,4,50,100,200,300}, the
    same measure() dial — ONLY the device policy changes (CPU from step 0).
  * TAIL = checkpoints {100, 200, 300} (the dispatch's comparison grid).
  * RETENTION DIALS (10, the adjudicated set): gm12, g0, gp12, held30_gm12,
    held30_g0, site_read_onset, site_read_span, A129 (ABSOLUTE — its sign
    flips near zero at the tail; strength is the magnitude), row0_strength,
    dall_g0. CE_R is REPORTED but NOT adjudicated: it is a corpus-recovery
    dial — at the tail both seeds sit at the same corpus level (1.60-1.63)
    in BOTH runs; the retention question does not live there.
  * per tail checkpoint: slower_count = #{dials: v(10904) > v(10903)}
    (higher residual = slower after-death decay); class "10904-SLOWER"
    iff slower_count >= 7/10, else "NOT-SLOWER". ONE classification rule,
    applied IDENTICALLY to e184's stored mixed-device trace (embedded,
    verified vs runs/e184/metrics.json) and to this cell's CPU-only trace.
  * e184's stored reference classes (computed from the stored file, quoted
    here for the record): +100 SLOWER (10/10, median 2.40x, max 7.15x),
    +200 SLOWER (8/10, median 1.80x, max 3.32x), +300 NOT-SLOWER (0/10,
    median 0.52x, max 0.99x — both seeds at the ~1e-3 wash floor).
  * TAIL-REPRODUCES iff this cell's class equals e184's stored class at
    EVERY tail checkpoint {100,200,300}; TAIL-SHUFFLES otherwise (any
    class change; a checkpoint where e184 had separation collapsing to
    parity is the "both tails converge" clause).
  * magnitude co-report (not a bar): per checkpoint the median and max
    10904/10903 ratio over the 10 dials, this cell vs e184's — the
    "2-34x"-scale separation's honest size.
  * DISSOLUTION CLOCK co-report (not a bar; the dispatch's texture note):
    per seed the earliest checkpoint with gm12 <= 0.27; the DISSOLUTION is
    device-robust iff both CPU seeds cross in the same bracket as e184's
    ((1,2]: under at +2, above at +1).
  * DEVICE-EFFECT co-report (honesty reflex; not a bar): per seed, per
    checkpoint, the cpu/mixed ratio of every dial — how much device
    arithmetic alone moved the SAME seed's numbers (the confound's own
    size, measured).

COMPUTE ENVELOPE (dispatch): CPU-ONLY — CUDA_VISIBLE_DEVICES="-1" is set
before torch is imported and asserted (cuda.is_available() must be False);
torch threads 4 (the dispatch's LOW <= 4; e184 used 8); sequential
trainings with cooldown(75 s) before/after each (the dispatch's 60-120 s);
time cap 1800 s per training; no mid-run guard (nothing to guard — no
device events possible); ALL readouts CPU-side. Nets are the mandated
2.7M e131_consolidated line (the dispatch's '<=1M family' note is an
envelope statement — e143/e151/e152/e158/e176n/e184 precedent; every gate
reference and lineage number of these cells lives on the 2.7M line).

INSTRUMENT PROVENANCE: load_cpu/evl_load/battery_cell/battery_pz/
ce_fixed_cpu/val_windows/deleted_wpe/read_fact_at/row_census/finetune_
freeze/measure()/flat_cells/gate_vs are lab/e184_seed_replicates.py
VERBATIM (the e176n/e176/e161/e152/e151/e143/e131/e119/e113/e068/e065/
e043 lineage); finetune_freeze LOSES the device parameter and the mid-run
guard (CPU-only policy) — the per-step arithmetic and the CPU-generator
RNG draw sequence are unchanged. Copied, not imported, to own the device
policy.

NETS: root runs/checkpoints/e131_consolidated_e113.pt (gate bit-exact vs
e151's stored before-cells, e176n/e184's G_ROOT set). e184's mixed-device
trajectories = runs/e184/metrics.json traces (embedded, verified vs file
at plot time). New checkpoints: runs/checkpoints/e185c_neutral_s{10903,
10904}.pt (+ _s{1,2,4,50,100,200} intermediates) + smoke_* twins.

Outputs: runs/e185c/{metrics.json, tail_rerun.png}; checkpoints
runs/checkpoints/e185c_*.pt. No NOTES/THINKING/QUEUE/STATE edits; single
commit, no push.

Run:  cd lab && python e185c_tail_rerun.py    (E185C_SMOKE=1 shakedown)
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

# ---- CPU-ONLY POLICY (dispatch): set BEFORE any torch import --------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

assert not torch.cuda.is_available(), (
    "e185c is CPU-ONLY by dispatch; CUDA_VISIBLE_DEVICES=-1 failed")
torch.set_num_threads(4)            # the dispatch's LOW <= 4 threads

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT, cooldown,     # noqa: E402
                    run_dir, save_json)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E185C_SMOKE") == "1"
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
E184_METRICS = E43.REPO / "runs" / "e184" / "metrics.json"

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

# ---- the two CPU-only re-runs ------------------------------------------------
CK_MAIN: tuple[int, ...] = (1, 2, 4, 50, 100, 200, 300) if not SMOKE else (1, 2, 4)
SEEDS: tuple[int, ...] = (10903, 10904)     # e184's replicate seeds, unchanged

# ---- fine-tune envelope (e184 VERBATIM; only the device policy changes) ------
LR = 1e-3                          # e176N arm A verbatim
CPU_CAP_S = 1800.0                 # dispatch: <=1800 s caps, CPU-only
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
COOLDOWN_S = 75.0                  # dispatch: 60-120 s

# ---- e170's neutral anchor bank (e176N arm A's stream, VERBATIM) --------------
E170_ANCHOR_SEED = 170             # dedicated RNG for the plain-corpus bank
ANCHOR_FORBIDDEN = ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")

# ---- gates / references (full precision, = stored metrics) --------------------
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05

E151_ROOT = {                     # runs/e151 'before' battery (e176n/e184's gate set)
    "base_gm12": 0.9155886769294739,
    "base_g0": 0.7850371599197388,
    "base_gp12": 0.9478210210800171,
    "ce_r": 1.663516640663147,
    "site_read_onset": 0.8898659348487854,
    "site_read_span": 0.982668936252594,
    "A129": -0.13237020391970877,
    "row0_strength": 0.7316772222770098,
    "dall_g0": 0.9047248959541321,
}

# e184's stored MIXED-DEVICE trajectories (runs/e184/metrics.json traces
# VERBATIM, re-verified at plot time) — the comparison's reference arm.
# Device events in that run: s10903 cuda->cpu at s175, s10904 cuda->cpu at
# s275 (thermal guard, 83 C) — so +100 is GPU/GPU, +200 is CPU/GPU (the
# confounded comparison), +300 is CPU/CPU.
E184_TRACE = {
    "10903": {
        "freeze_steps": [0, 1, 2, 4, 50, 100, 200, 300],
        "gm12": [0.9155886769294739, 0.9414604902267456,
                 0.0012103066546842456, 0.002691009547561407,
                 0.017371630296111107, 0.006901818327605724,
                 0.0011056294897571206, 0.0014312716666609049],
        "g0": [0.7850371599197388, 0.7742742896080017,
               0.03131471574306488, 0.016779916360974312,
               0.07963299751281738, 0.010602777823805809,
               0.005421841982752085, 0.002145472913980484],
        "gp12": [0.9478210210800171, 0.9105220437049866,
                 0.0007421242189593613, 0.0035336578730493784,
                 0.03288742154836655, 0.0061796666122972965,
                 0.0026814802549779415, 0.0037908523809164762],
        "held30_gm12": [0.636141836643219, 0.7853401899337769,
                        0.0005616897833533585, 0.0017917196964845061,
                        0.003652523970231414, 0.0032577381934970617,
                        0.0008765784441493452, 0.001160718034952879],
        "held30_g0": [0.7233286499977112, 0.8278920650482178,
                      0.004254986997693777, 0.010120860300958157,
                      0.029445599764585495, 0.004297696985304356,
                      0.0038885220419615507, 0.0015575072029605508],
        "ce_r": [1.663516640663147, 2.3094489574432373,
                 2.2162535190582275, 1.8131717443466187,
                 1.6601144075393677, 1.6374231576919556,
                 1.6055161952972412, 1.6305550336837769],
        "site_read_onset": [0.8898659348487854, 0.7769210934638977,
                            0.0004843590431846678, 0.0026645001489669085,
                            0.02792367711663246, 0.006057294085621834,
                            0.0012391885975375772, 0.0009866261389106512],
        "site_read_span": [0.982668936252594, 0.9329555630683899,
                           0.10127590596675873, 0.8255504965782166,
                           0.7924992442131042, 0.7424058318138123,
                           0.5711501836776733, 0.6271534562110901],
        "A129": [-0.13237020391970877, -0.13476216706136857,
                 0.023748736632842337, 0.009077680735693625,
                 0.01598753137125944, -0.00445624619994002,
                 0.0023396291847651205, -0.0007033789048364269],
        "row0_strength": [0.7316772222770098, 0.6947010270132978,
                          0.026552071558125095, 0.01563638785519288,
                          0.0793856572446245, 0.010511997488208635,
                          0.005175752295633629, 0.0019520004118750952],
        "dall_g0": [0.9047248959541321, 0.9117472767829895,
                    0.0023086972068995237, 0.0063136545941233635,
                    0.04849929362535477, 0.013582943007349968,
                    0.0026710412587310076, 0.0023896275088191032],
    },
    "10904": {
        "freeze_steps": [0, 1, 2, 4, 50, 100, 200, 300],
        "gm12": [0.9155886769294739, 0.7062474489212036,
                 0.08502934128046036, 0.04554390907287598,
                 0.041999075561761856, 0.006971660070121288,
                 0.0036725152749568224, 0.001418216503225267],
        "g0": [0.7850371599197388, 0.5195467472076416,
               0.3203027546405792, 0.21303822100162506,
               0.1337602585554123, 0.04275732859969139,
               0.010678130201995373, 0.00109452148899436],
        "gp12": [0.9478210210800171, 0.7527174353599548,
                 0.17854999005794525, 0.08628150820732117,
                 0.049249399453401566, 0.013239486142992973,
                 0.001770209171809256, 0.0009967173682525754],
        "held30_gm12": [0.636141836643219, 0.5653285980224609,
                        0.03169308975338936, 0.02285507135093212,
                        0.015852700918912888, 0.0032966190483421087,
                        0.0018227447289973497, 0.0008544281008653343],
        "held30_g0": [0.7233286499977112, 0.5462616086006165,
                      0.20292720198631287, 0.1382521092891693,
                      0.08721010386943817, 0.030726807191967964,
                      0.006324071902781725, 0.0005635803681798279],
        "ce_r": [1.663516640663147, 2.1426706314086914,
                 1.924661636352539, 1.799659252166748,
                 1.6943918466567993, 1.6291100978851318,
                 1.6297920942306519, 1.6145808696746826],
        "site_read_onset": [0.8898659348487854, 0.5938538312911987,
                            0.08672204613685608, 0.04597408324480057,
                            0.06940940022468567, 0.02432984858751297,
                            0.001725569716654718, 0.0006519577000290155],
        "site_read_span": [0.982668936252594, 0.9320135712623596,
                           0.7148771286010742, 0.8479496836662292,
                           0.8287115097045898, 0.8322591781616211,
                           0.5458841323852539, 0.6045894622802734],
        "A129": [-0.13237020391970877, -0.2236907600269964,
                 0.11744463199477953, 0.09747628482679527,
                 0.006052696846503142, 0.011837428566650487,
                 0.003934658945945556, -0.00022405162344512068],
        "row0_strength": [0.7316772222770098, 0.48658570824668834,
                          0.3035019068236115, 0.20919532310842898,
                          0.13207145412095447, 0.04188441713594955,
                          0.009946092482247574, 0.0010139477353504843],
        "dall_g0": [0.9047248959541321, 0.7461299300193787,
                    0.16410207748413086, 0.09194029867649078,
                    0.10970885306596756, 0.024938814422091942,
                    0.00546565605327487, 0.0012289943406358361],
    },
}

E184_DEVICE_EVENTS = [
    {"tag": "s10903", "step": 175, "event": "MID-RUN MIGRATION",
     "status": {"util": 91.0, "mem_used": 1104.0, "mem_total": 24463.0,
                "temp": 83.0, "power": 99.12}},
    {"tag": "s10904", "step": 275, "event": "MID-RUN MIGRATION",
     "status": {"util": 77.0, "mem_used": 1104.0, "mem_total": 24463.0,
                "temp": 83.0, "power": 111.35}},
]

# ---- registered bar constants (frozen; see OPERATIONALIZATIONS above) ---------
SHUT_BAR = 0.27                   # the dissolution clock's bar (co-report)
TAIL_CKPTS: tuple[int, ...] = (100, 200, 300)
TAIL_DIALS: tuple[str, ...] = ("gm12", "g0", "gp12", "held30_gm12",
                               "held30_g0", "site_read_onset",
                               "site_read_span", "A129", "row0_strength",
                               "dall_g0")
SLOWER_MIN = 7                    # of 10 dials -> class "10904-SLOWER"
ROOT_GM12 = E151_ROOT["base_gm12"]
ROOT_G0 = E151_ROOT["base_g0"]

REGISTERED_PREDICTION = {
    "tail_reproduces": "TAIL-REPRODUCES fires if: the tail ordering matches "
        "e184's (10904 slower than 10903 at the tail checkpoints) — the "
        "lottery is real; T112's tail sentence stands.",
    "tail_shuffles": "TAIL-SHUFFLES fires if: the ordering changes or both "
        "tails converge — the tail sentence dies (a device artifact); "
        "T112's texture struck.",
    "no_shopping": "No bar shopping; texture (both die on the same clock "
        "regardless) => note that the DISSOLUTION itself is device-robust "
        "either way.",
    "operationalizations": "TAIL = {100,200,300}; retention dials (10) = "
        "gm12/g0/gp12/held30_gm12/held30_g0/site_read_onset/site_read_span/"
        "A129(|.|)/row0_strength/dall_g0 (CE_R reported, not adjudicated: a "
        "corpus-recovery dial); per checkpoint slower_count = #{dials: "
        "v10904 > v10903}, class 10904-SLOWER iff >= 7/10 — ONE rule applied "
        "identically to e184's stored mixed-device trace (embedded, "
        "verified) and this cell's CPU-only trace; TAIL-REPRODUCES iff the "
        "classes match at ALL THREE tail checkpoints; magnitudes (median/"
        "max 10904/10903 ratios) co-reported; dissolution clock (earliest "
        "gm12 <= 0.27) and per-seed cpu-vs-mixed device-effect ratios "
        "co-reported, not adjudicated.",
    "registration": "Commit a0974ab dispatched e185c (bars in the mission "
        "text; QUEUE row: 'tail ordering reproduces => the lottery stands; "
        "shuffles => the tail sentence dies'); this docstring freezes them "
        "verbatim before compute. Adjudicate against exactly this; no bar "
        "shopping.",
}

trims: list[str] = []
deviations: list[str] = [
    "CPU-ONLY by dispatch: CUDA_VISIBLE_DEVICES=-1 set before torch import "
    "and asserted; no device pick, no mid-run guard — device events are "
    "impossible by construction (the point of the cell).",
    "Torch threads 4 (the dispatch's LOW <= 4) vs e184's 8 — CPU reduction "
    "order can drift low-order bits; the G_ROOT gate reports the 5e-6 bit "
    "flag and the 0.05 fallback tolerance (e161/e176/e176n/e184 precedent).",
    "Nets are the mandated 2.7M e131_consolidated line (the dispatch's "
    "'<=1M family' note is an envelope statement; e143/e151/e152/e158/"
    "e176n/e184 precedent — every gate reference lives on this line).",
    "The original (extinction) anchor bank is NOT rebuilt (e176n's arm-B "
    "business): the stream is e176n arm A's neutral bank verbatim (re-gated: "
    "0/16 host content, 0/16 junctions), identical to e184's cells.",
    "Same-seed CPU-vs-e184 differences are EXPECTED, not drift: e184's "
    "arithmetic ran cuda until s175/s275 — the per-seed cpu/mixed ratio "
    "table measures exactly this (the confound's own size).",
    "n=2 per run arm (one trajectory per seed per device policy); the tail "
    "readout is at checkpoint resolution {100,200,300} with 10 dials each "
    "(30 ordered comparisons per run) — no per-step resolution claimed.",
    "Smoke mode trims: 4-step trainings per seed, checkpoints {1,2,4}, lean "
    "measures (no censuses, no deletion table), no cooldowns; nothing "
    "adjudicated.",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/e184_seed_replicates.py VERBATIM (see the module docstring).
# Copied rather than imported to own the CPU-only device policy.

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
    """e131's read_fact_position VERBATIM ARITHMETIC (e184 copy): p(true name
    char) at positions addr_row..addr_row+6."""
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
    """THE NEUTRAL PLAIN-CORPUS FREEZE, CPU FROM STEP 0 (e184's
    finetune_freeze with the device parameter and mid-run guard REMOVED —
    per-step arithmetic and the CPU-generator RNG draw sequence unchanged).
    Per step: aj = randint(16) anchor draws, rj = randint(16) random corpus
    offsets; batch 32 full-token CE; AdamW (0.9,0.95) wd 0.1 constant lr,
    clip 1.0. Snapshots (deep-copy out) + light CPU evals (g-12, g0, CE_R —
    no RNG consumed) at the checkpoint steps; in-batch corpus CE recorded
    at every checkpoint and every 50."""
    cap = CPU_CAP_S
    ckpt_set = set(ckpt_steps)
    n_steps = ckpt_steps[-1]
    net = copy.deepcopy(net0)
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
    net.eval()
    del net
    return {"sds": sds, "traj": traj, "steps_ran": step, "seed": seed,
            "lr": LR, "zeph_violations": zeph_checks,
            "initial_device": "cpu", "final_device": "cpu",
            "time_cap_s": cap}


CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "e185c", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name} "
        f"({', '.join(f'{k}={v}' for k, v in meta.items())})")


def verify_e184(embedded: dict, path: Path) -> dict:
    """Verify the embedded e184 trace copies against the stored metrics file
    (no silent divergence; e184's verify_e176n convention)."""
    src = {"source": "embedded verbatim copy (runs/e184 traces)",
           "file_present": path.exists(), "verified_vs_embedded": None,
           "max_abs_diff": None}
    if path.exists():
        mm = json.loads(path.read_text(encoding="utf-8"))
        diffs = []
        for sk in ("10903", "10904"):
            rows = mm["traces"][sk]
            for r in rows:
                s = r["freeze_steps"]
                i = embedded[sk]["freeze_steps"].index(s)
                for k in TAIL_DIALS + ("ce_r",):
                    diffs.append(abs(r[k] - embedded[sk][k][i]))
        steps_ok = all(
            [r["freeze_steps"] for r in mm["traces"][sk]]
            == embedded[sk]["freeze_steps"] for sk in ("10903", "10904"))
        src["max_abs_diff"] = max(diffs) if diffs else None
        src["verified_vs_embedded"] = bool(steps_ok and max(diffs) < 1e-9)
        if src["verified_vs_embedded"]:
            src["source"] = ("runs/e184/metrics.json traces (embedded copy "
                             f"verified, max|diff| {max(diffs):.1e})")
        # the stored device events (the confound under discharge)
        dev_ok = mm.get("device_events") == E184_DEVICE_EVENTS
        src["device_events_match"] = bool(dev_ok)
    return src


# ------------------------------------------------------------------ adjudication

def tail_pattern(tr: dict) -> dict:
    """ONE classification rule for both arms: per tail checkpoint, the count
    of retention dials where seed 10904 sits ABOVE seed 10903 (slower
    after-death decay), the class (>= 7/10 -> '10904-SLOWER'), and the
    magnitude (median/max ratio)."""
    out = {}
    for ck in TAIL_CKPTS:
        i3 = tr["10903"]["freeze_steps"].index(ck)
        i4 = tr["10904"]["freeze_steps"].index(ck)
        ratios = {}
        for d in TAIL_DIALS:
            a, b = tr["10903"][d][i3], tr["10904"][d][i4]
            if d == "A129":
                a, b = abs(a), abs(b)
            ratios[d] = float(b / a) if a != 0 else float("inf")
        slower = sum(1 for r in ratios.values() if r > 1.0)
        finite = [r for r in ratios.values() if r != float("inf")]
        out[str(ck)] = {
            "slower_count": slower, "n_dials": len(TAIL_DIALS),
            "class": "10904-SLOWER" if slower >= SLOWER_MIN else "NOT-SLOWER",
            "median_ratio": float(np.median(finite)) if finite else None,
            "max_ratio": max(finite) if finite else None,
            "min_ratio": min(finite) if finite else None,
            "ratios": ratios,
        }
    return out


def steps_to_under(tr: dict, seed_key: str, bar: float = SHUT_BAR):
    """Earliest continuation checkpoint with gm12 <= bar (e184's clock)."""
    fs = tr[seed_key]["freeze_steps"]
    gm = tr[seed_key]["gm12"]
    return next((s for s, g in zip(fs, gm) if s > 0 and g <= bar), None)


def device_effect(tr_cpu: dict) -> dict:
    """Per seed, per checkpoint: the cpu/mixed ratio of every dial — the
    device confound's own size (e184's arithmetic ran cuda until s175/s275)."""
    out = {}
    for sk in ("10903", "10904"):
        fs = tr_cpu[sk]["freeze_steps"]
        rows = []
        for i, s in enumerate(fs):
            row = {"step": s}
            for d in TAIL_DIALS + ("ce_r",):
                v_cpu = tr_cpu[sk][d][i]
                v_mix = E184_TRACE[sk][d][i]
                row[d] = {"cpu": v_cpu, "e184_mixed": v_mix,
                          "ratio": float(v_cpu / v_mix) if v_mix != 0
                          else None}
            rows.append(row)
        out[sk] = rows
    return out


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e185c_smoke" if SMOKE else "e185c")
    log(f"E185C THE CPU-ONLY TAIL RE-RUN (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r}, "
        f"cuda_available={torch.cuda.is_available()}), threads "
        f"{torch.get_num_threads()}, cooldown {COOLDOWN_S:.0f}s, cap "
        f"{CPU_CAP_S:.0f}s/training")

    # ---------------- protocol rebuild (e143/e151/e152/e161/e176/e176n/e184 verbatim)
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
    # THE NEUTRAL STREAM (e170's construction VERBATIM via e176n arm A /
    # e184): 16 plain-corpus windows, rejection on host/nonce content.
    # FIXED content (seed 170) — identical to e184's cells.
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

    # junction accounting (e170 VERBATIM)
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
                             "e170's construction VERBATIM (= e176n arm A = "
                             "e184's stream; FIXED content, not reseeded)"),
            "n_windows": 16, "block": BLOCK, "seed": E170_ANCHOR_SEED,
            "starts": n_starts, "tries": tries, "rejections": rejections,
            "forbidden": list(ANCHOR_FORBIDDEN),
            "windows_with_host_content": sum(
                1 for s in n_starts
                if any(f in train_text[s: s + BLOCK + 1] for f in HOSTS)),
            "junctions_covered": jc_neutral,
        },
        "budget_identical_to_e184": bool(anchor_neutral.shape == (16, BLOCK)),
        "rng_note": ("finetune_freeze verbatim (CPU-only); draw shapes/moduli "
                     "identical; the generator seeds are UNCHANGED from e184 "
                     "(10903/10904) — same draw sequences, same fixed bank; "
                     "ONLY the arithmetic's device differs (cuda-prefix -> "
                     "none): the device confound under discharge"),
    }
    G_ANCHOR["pass"] = bool(
        G_ANCHOR["neutral_bank"]["windows_with_host_content"] == 0
        and G_ANCHOR["neutral_bank"]["junctions_covered"] == 0
        and G_ANCHOR["budget_identical_to_e184"])
    assert G_ANCHOR["pass"], f"anchor gate FAILED: {G_ANCHOR}"
    log(f"G_ANCHOR: neutral bank 16x{BLOCK} plain-corpus windows (seed "
        f"{E170_ANCHOR_SEED}, {rejections} rejections/{tries} tries) — host "
        f"content 0/16, junctions 0/16: PASS")

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
        """e184's measure() VERBATIM: base 3-geos + held30 + CE_R + site
        read + old-band census + deletion table (D-all, D-183)."""
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
    src_e184 = verify_e184(E184_TRACE, E184_METRICS)
    log(f"e184 mixed-device reference: {src_e184['source']}")

    # =====================================================================
    # THE TWO CPU-ONLY RE-RUNS (e184 verbatim; only the device policy differs)
    # =====================================================================
    arms: dict = {}
    batteries_all: dict = {}
    for k, seed in enumerate(SEEDS):
        tag = f"s{seed}"
        log("=" * 78)
        if not SMOKE:
            log(f"[thermal] cooldown {COOLDOWN_S:.0f}s before {tag}")
            cooldown(COOLDOWN_S)
        log(f"CPU-ONLY RE-RUN {k + 1}/2 — {tag}: {CK_MAIN[-1]}-step neutral "
            f"freeze (e184 protocol verbatim: batch {ANCH_BS} neutral + "
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
        save_ckpt(f"e185c_neutral_s{seed}", arm["sds"][smax],
                  {"desc": f"e131_consolidated_e113 + {smax}-step NEUTRAL-"
                           f"anchor plain-corpus freeze, CPU END-TO-END "
                           f"(e184's protocol verbatim; e170's neutral bank "
                           f"seed {E170_ANCHOR_SEED}: 16 plain-corpus "
                           f"windows, 0/16 junctions; batch 32 = 16 neutral "
                           f"anchors + 16 random, full-token CE), lr {LR}, "
                           f"seed {seed} — R52's device-confound discharge",
                   "steps": int(smax), "seed": seed, "lr": LR,
                   "anchor_bank": f"neutral (seed {E170_ANCHOR_SEED})",
                   "device": "cpu->cpu",
                   "base": f"runs/checkpoints/{ROOT_CK}"})
        for s in sorted(arm["sds"]):
            if s == smax:
                continue
            save_ckpt(f"e185c_neutral_s{seed}_s{s}", arm["sds"][s],
                      {"desc": f"e131_consolidated_e113 + {s}-step NEUTRAL-"
                               f"anchor freeze (intermediate), CPU-only, "
                               f"seed {seed}",
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
    # THE TAIL ADJUDICATION (registered clauses; no shopping)
    # =====================================================================
    tr_cpu = {}
    for tag, seed in [(f"s{s}", s) for s in SEEDS]:
        steps = [0] + sorted(arms[tag]["sds"])
        rows = []
        for s in steps:
            b = batteries_all[tag]["root" if s == 0 else str(s)]
            c = flat_cells(b)
            rows.append(c)
        dial_keys = TAIL_DIALS + ("ce_r", "d183_g0", "d183_gm12")
        if SMOKE:
            dial_keys = tuple(d for d in dial_keys if d in rows[0])
        tr_cpu[str(seed)] = {"freeze_steps": steps,
                             **{d: [r[d] for r in rows]
                                for d in dial_keys}}

    if SMOKE:
        log("SMOKE: no tail checkpoints measured — nothing adjudicated")
        metrics = {
            "experiment": "e185c_tail_rerun", "date": common.now_iso(),
            "smoke": True, "gates": {"G_ROOT": G_ROOT},
            "arms": {f"s{s}": {"traj": arms[f"s{s}"]["traj"],
                               "steps_ran": arms[f"s{s}"]["steps_ran"]}
                     for s in SEEDS},
            "timing": {"total_s": round(time.time() - T0, 1)},
        }
        save_json(rd / "metrics.json", E43.jsonable(metrics))
        log(f"outputs: {rd / 'metrics.json'} (smoke; nothing adjudicated)")
        return 0

    pat_e184 = tail_pattern(E184_TRACE)
    pat_cpu = tail_pattern(tr_cpu)
    reproduces = all(pat_cpu[str(ck)]["class"] == pat_e184[str(ck)]["class"]
                     for ck in TAIL_CKPTS)
    verdict = "TAIL-REPRODUCES" if reproduces else "TAIL-SHUFFLES"

    # the shuffle's texture, per checkpoint (which class moved, convergence?)
    shuffle_detail = {}
    for ck in TAIL_CKPTS:
        a, b = pat_e184[str(ck)], pat_cpu[str(ck)]
        shuffle_detail[str(ck)] = {
            "e184_class": a["class"], "cpu_class": b["class"],
            "e184_slower_count": a["slower_count"],
            "cpu_slower_count": b["slower_count"],
            "e184_median_ratio": a["median_ratio"],
            "cpu_median_ratio": b["median_ratio"],
            "cpu_converged": bool(0.8 <= (b["median_ratio"] or 0) <= 1.25),
        }

    # dissolution clock (co-report): device-robust iff same bracket as e184
    clock = {}
    for sk in ("10903", "10904"):
        clock[sk] = {
            "e184_steps_to_under_bar": steps_to_under(E184_TRACE, sk),
            "cpu_steps_to_under_bar": steps_to_under(tr_cpu, sk),
        }
    clock_device_robust = all(
        c["cpu_steps_to_under_bar"] is not None
        and c["cpu_steps_to_under_bar"] == c["e184_steps_to_under_bar"]
        for c in clock.values())

    # device effect (co-report): same-seed cpu vs e184-mixed ratios
    deffect = device_effect(tr_cpu)
    deffect_summary = {}
    for sk in ("10903", "10904"):
        rows = []
        for row in deffect[sk]:
            if row["step"] == 0:
                continue
            finite = [row[d]["ratio"] for d in TAIL_DIALS
                      if row[d]["ratio"] is not None]
            rows.append({"step": row["step"],
                         "median_ratio_cpu_vs_e184mixed":
                             float(np.median(finite)),
                         "max_ratio": max(finite),
                         "min_ratio": min(finite)})
        deffect_summary[sk] = rows

    if reproduces:
        clause = (f"the tail ordering REPRODUCES CPU-only: at every tail "
                  f"checkpoint {{100,200,300}} the CPU classes match e184's "
                  f"mixed-device classes ("
                  + "; ".join(f"+{ck}: {pat_cpu[str(ck)]['class']} "
                              f"({pat_cpu[str(ck)]['slower_count']}/10, "
                              f"median {pat_cpu[str(ck)]['median_ratio']:.2f}x)"
                              for ck in TAIL_CKPTS)
                  + f") — seed 10904's slower after-death decay is NOT a "
                  f"device artifact; T112's tail sentence stands (the timing "
                  f"lottery lives in the depth/tail, not the clock)")
    else:
        moved = [f"+{ck}: {pat_e184[str(ck)]['class']}->"
                 f"{pat_cpu[str(ck)]['class']}"
                 for ck in TAIL_CKPTS
                 if pat_cpu[str(ck)]['class'] != pat_e184[str(ck)]['class']]
        conv = [f"+{ck}" for ck in TAIL_CKPTS
                if shuffle_detail[str(ck)]["cpu_converged"]
                and pat_e184[str(ck)]['class'] == '10904-SLOWER'
                and pat_cpu[str(ck)]['class'] != '10904-SLOWER']
        clause = (f"the tail ordering SHUFFLES CPU-only ({'; '.join(moved)})"
                  + (f" — both tails converge at {', '.join(conv)}: the "
                     f"separation e184 showed was the device artifact (the "
                     f"confounded +200 sat CPU(10903) vs GPU(10904))"
                     if conv else "")
                  + f"; T112's tail sentence dies; the texture is struck")

    log("=" * 78)
    log(f"E185C VERDICT: {verdict}")
    for ck in TAIL_CKPTS:
        a, b = pat_e184[str(ck)], pat_cpu[str(ck)]
        log(f"  +{ck}: e184 {a['class']} ({a['slower_count']}/10, median "
            f"{a['median_ratio']:.2f}x, max {a['max_ratio']:.2f}x)  ->  cpu "
            f"{b['class']} ({b['slower_count']}/10, median "
            f"{b['median_ratio']:.2f}x, max {b['max_ratio']:.2f}x)")
    log(f"  dissolution clock: {clock} | device-robust: {clock_device_robust}")
    log(f"  {clause}")
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e185c_tail_rerun",
        "date": common.now_iso(),
        "registration": ("commit a0974ab dispatched e185c (bars in the "
                         "mission text + QUEUE row); frozen verbatim in the "
                         "module docstring before compute"),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("is T112's tail sentence (seed 10904's 2-34x slower "
                     "after-death decay) a device artifact of e184's "
                     "cuda->cpu migrations (s175/s275), or does the tail "
                     "ordering reproduce CPU end-to-end?"),
        "root": f"runs/checkpoints/{ROOT_CK} (gated vs e151 before-cells, "
                f"max|diff| {G_ROOT['max_abs_diff']:.2e})",
        "root_meta": root_meta,
        "arms": {
            f"s{seed}": {
                "desc": "e184's protocol VERBATIM at the same seed, CPU "
                        "END-TO-END (no device events possible): batch 32 = "
                        "16 neutral-bank draws + 16 random corpus windows, "
                        "full-token CE, AdamW (0.9,0.95) wd 0.1 lr 1e-3 "
                        "constant, clip 1.0 — the RNG draw sequence is "
                        "seed-identical to e184's cell; only the arithmetic's "
                        "device differs",
                "ckpt_steps": list(CK_MAIN),
                "steps_ran": arms[f"s{seed}"]["steps_ran"],
                "seed": seed, "lr": arms[f"s{seed}"]["lr"],
                "device": "cpu->cpu",
                "time_cap_s": arms[f"s{seed}"]["time_cap_s"],
                "traj": arms[f"s{seed}"]["traj"],
                "zeph_violations": arms[f"s{seed}"]["zeph_violations"],
                "missing_checkpoints": [s for s in CK_MAIN
                                        if s not in arms[f"s{seed}"]["sds"]],
            } for seed in SEEDS
        },
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
                     "geometries_measured": {"novel": [-12, 12],
                                             "trained_g0": 0},
                     "neutral_bank": G_ANCHOR["neutral_bank"],
                     "measure_dial": "e184's measure() (= e176n's, minus the "
                                     "183-span census; e178's recorded "
                                     "deviation)"},
        "gates": {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                  "G_POOL": G_POOL, "G_ANCHOR": G_ANCHOR, "G_ROOT": G_ROOT,
                  "G_SURG": gates_surg},
        "traces_cpu": tr_cpu,
        "tail_adjudication": {
            "tail_ckpts": list(TAIL_CKPTS),
            "dials": list(TAIL_DIALS),
            "slower_rule": f"class 10904-SLOWER iff >= {SLOWER_MIN}/10 dials "
                           f"have v(10904) > v(10903) (A129 absolute; CE_R "
                           f"reported not adjudicated)",
            "e184_reference_pattern": pat_e184,
            "cpu_pattern": pat_cpu,
            "shuffle_detail": shuffle_detail,
            "verdict": verdict,
            "clause": clause,
        },
        "dissolution_clock": {
            "bar": SHUT_BAR,
            "per_seed": clock,
            "device_robust": clock_device_robust,
            "note": "co-report, not a bar: both seeds' earliest gm12 <= 0.27 "
                    "checkpoint, CPU vs e184's mixed run — the DISSOLUTION "
                    "is device-robust iff the bracket matches ((1,2])",
        },
        "device_effect_same_seed": {
            "per_seed_rows": deffect,
            "summary": deffect_summary,
            "note": "co-report, not a bar: the SAME seed's dial values, "
                    "CPU-only vs e184's mixed-device run — the confound's "
                    "own size (e184 ran cuda until s175/s275)",
        },
        "batteries": batteries_all,
        "references": {
            "e184_mixed": {
                "ref": {"desc": "e184's stored seed replicates (the tail "
                         "sentence's source run; cuda->cpu at s175/s275)"},
                "provenance": src_e184,
                "device_events": E184_DEVICE_EVENTS,
            },
        },
        "honesty_reflex": {
            "device_policy": ("CPU-ONLY from step 0 at BOTH seeds — no "
                              "device event is possible; the comparison "
                              "grid {100,200,300} is same-device "
                              "cpu/cpu at every checkpoint (e184's +200 was "
                              "cpu(10903) vs cuda(10904) — the confound "
                              "under discharge)"),
            "same_seed_not_bit_reproducible": ("the RNG draw sequences are "
                               "seed-identical to e184's (CPU torch."
                               "Generator, device-independent), but e184's "
                               "arithmetic ran cuda until s175/s275 — the "
                               "CPU re-run at the same seed is a DIFFERENT "
                               "float trajectory, not a bit-reproducibility "
                               "check; the per-seed cpu/mixed ratio table "
                               "measures exactly this divergence"),
            "one_ruler": ("the tail classification rule is applied "
                          "identically to e184's stored trace (embedded, "
                          "verified vs file) and the fresh CPU trace — the "
                          "reference classes are COMPUTED, not asserted"),
            "tail_floor": ("at +300 both seeds sit at the ~1e-3 wash floor "
                           "in e184; ordering statements there are "
                           "floor-noise-limited — the class rule handles "
                           "this identically on both arms, and the "
                           "convergence clause names it"),
            "single_lineage": ("both trajectories start from the SAME "
                               "bit-gated root (e131_consolidated_e113, "
                               "G_ROOT PASS) on the 2.7M line — the device "
                               "question tested is the WASH dynamics', one "
                               "organism; e157's second family remains the "
                               "lineage bound"),
            "n_of_one": ("one trajectory per seed per device policy; the "
                         "tail readout is 30 ordered dial comparisons per "
                         "run — device-robustness of the ORDERING is the "
                         "claim adjudicated, not per-dial magnitudes"),
        },
        "trims": trims, "device_events": [],
        "deviations": deviations,
        "ckpt_inventory": CKPT_INVENTORY,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": int(net0.num_params()),
                   "train_device": "cpu (CUDA_VISIBLE_DEVICES=-1, asserted)",
                   "eval_device": "cpu",
                   "cuda_available": bool(torch.cuda.is_available()),
                   "torch_threads": torch.get_num_threads(),
                   "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "tail_rerun.png", tr_cpu, pat_e184, pat_cpu, verdict, clause,
         clock, clock_device_robust, deffect_summary)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'tail_rerun.png'}, "
        f"{len(CKPT_INVENTORY)} checkpoints in runs/checkpoints/e185c_*.pt")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot

def plot(path, tr_cpu, pat_e184, pat_cpu, verdict, clause, clock,
         clock_robust, deffect_summary):
    """THE figure: the n=2 overlay (cpu-only vs e184's mixed), the tail zoom,
    the ratio panel (one ruler, both arms), and the verdict table."""
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 10.5))
    cols = {"10903": "seagreen", "10904": "darkorange"}

    def seq(tr, sk, key):
        return tr[sk]["freeze_steps"], tr[sk][key]

    # (0,0) THE n=2 OVERLAY (g-12 headline; cpu solid vs e184 mixed dashed)
    ax = axes[0, 0]
    for sk in ("10903", "10904"):
        xs, ys = seq(E184_TRACE, sk, "gm12")
        ax.plot(xs, ys, "o--", ms=6, lw=1.8, color=cols[sk], alpha=0.55,
                label=f"seed {sk} e184 MIXED (cuda->cpu s175/s275)")
        xs, ys = seq(tr_cpu, sk, "gm12")
        ax.plot(xs, ys, "o-", ms=8, lw=2.4, color=cols[sk],
                label=f"seed {sk} e185c CPU-ONLY")
    ax.axvspan(100, 300, color="tab:purple", alpha=0.06)
    ax.text(200, 0.45, "TAIL {100,200,300}", ha="center", fontsize=8,
            color="tab:purple")
    ax.axhline(SHUT_BAR, ls="--", lw=1.0, color="tab:purple", alpha=0.7,
               label=f"{SHUT_BAR} dissolution bar")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel("plain-corpus freeze steps (step 0 = root)")
    ax.set_ylabel("mean p(Z) g-12 (symlog)")
    ax.legend(fontsize=6.8, loc="lower left")
    ax.set_title(f"THE n=2 OVERLAY — g-12, same seeds: CPU-only vs e184's "
                 f"mixed -> {verdict}", fontsize=9.5)

    # (0,1) TAIL ZOOM: gm12 + g0 at {50..300}, both arms
    ax = axes[0, 1]
    for sk in ("10903", "10904"):
        for key, mk in (("gm12", "o"), ("g0", "^")):
            xs, ys = seq(E184_TRACE, sk, key)
            fine = [(s, v) for s, v in zip(xs, ys) if s >= 50]
            ax.plot([s for s, _ in fine], [v for _, v in fine], f"{mk}--",
                    ms=5, lw=1.3, color=cols[sk], alpha=0.45,
                    label=f"{sk} {key} e184")
            xs, ys = seq(tr_cpu, sk, key)
            fine = [(s, v) for s, v in zip(xs, ys) if s >= 50]
            ax.plot([s for s, _ in fine], [v for _, v in fine], f"{mk}-",
                    ms=7, lw=2.0, color=cols[sk], label=f"{sk} {key} CPU")
    ax.set_yscale("symlog", linthresh=0.005)
    ax.set_xlabel("freeze steps (tail zoom, 50..300)")
    ax.set_ylabel("mean p(Z) (symlog)")
    ax.legend(fontsize=6.2, loc="lower left", ncol=2)
    ax.set_title("tail zoom — g-12 and g0, both arms", fontsize=9.5)

    # (1,0) THE RATIO PANEL: 10904/10903 per tail checkpoint, one ruler
    ax = axes[1, 0]
    xs = np.arange(len(TAIL_CKPTS))
    w = 0.35
    for off, pat, lbl, a in ((-w / 2 - 0.02, pat_e184, "e184 MIXED", 0.55),
                             (w / 2 + 0.02, pat_cpu, "e185c CPU-ONLY", 1.0)):
        meds = [pat[str(ck)]["median_ratio"] for ck in TAIL_CKPTS]
        maxs = [pat[str(ck)]["max_ratio"] for ck in TAIL_CKPTS]
        mins = [pat[str(ck)]["min_ratio"] for ck in TAIL_CKPTS]
        ax.bar(xs + off, meds, width=w, alpha=a, color="slategray",
               label=f"{lbl}: median 10904/10903")
        ax.plot(xs + off, maxs, "rv", ms=7, alpha=a, label=f"{lbl}: max")
        ax.plot(xs + off, mins, "b^", ms=6, alpha=a, label=f"{lbl}: min")
        for x, m in zip(xs + off, meds):
            ax.text(x, m + 0.06, f"{m:.2f}x", ha="center", fontsize=7.5)
    for i, ck in enumerate(TAIL_CKPTS):
        ax.text(xs[i] - w / 2 - 0.02, -0.28,
                f"{pat_e184[str(ck)]['slower_count']}/10",
                ha="center", fontsize=7, color="dimgray")
        ax.text(xs[i] + w / 2 + 0.02, -0.28,
                f"{pat_cpu[str(ck)]['slower_count']}/10",
                ha="center", fontsize=7, color="k")
    ax.axhline(1.0, ls="-", lw=1.0, color="k", alpha=0.6)
    ax.axhspan(0.8, 1.25, color="seagreen", alpha=0.08)
    ax.text(2.42, 1.02, "parity band", fontsize=6.5, color="seagreen")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"+{ck}" for ck in TAIL_CKPTS])
    ax.set_xlabel("tail checkpoint (slower-counts below: e184 | cpu)")
    ax.set_ylabel("10904/10903 ratio over 10 retention dials")
    ax.set_ylim(-0.35, max(8.5, max(pat_cpu[str(ck)]["max_ratio"]
                                    for ck in TAIL_CKPTS) * 1.15))
    ax.legend(fontsize=6.5, loc="upper right")
    ax.set_title("the tail ordering, one ruler — median/max/min dial ratio "
                 "(>=7/10 above 1.0 => 10904-SLOWER)", fontsize=9)

    # (1,1) the verdict table + clock + device effect
    ax = axes[1, 1]
    ax.axis("off")
    ytxt = 0.97
    ax.text(0.02, ytxt, "THE TAIL COMPARISON (one rule, both arms):",
            fontsize=8.5, va="top", family="monospace", weight="bold")
    ytxt -= 0.042
    ax.text(0.02, ytxt,
            "  ckpt   e184 MIXED (slow/n, med)      CPU-ONLY (slow/n, med)",
            fontsize=7.2, va="top", family="monospace")
    ytxt -= 0.034
    for ck in TAIL_CKPTS:
        a, b = pat_e184[str(ck)], pat_cpu[str(ck)]
        match = "=" if a["class"] == b["class"] else "!="
        ax.text(0.02, ytxt,
                f"  +{ck:<5} {a['class']:<12} {a['slower_count']}/10 "
                f"{a['median_ratio']:.2f}x   {match}  "
                f"{b['class']:<12} {b['slower_count']}/10 "
                f"{b['median_ratio']:.2f}x",
                fontsize=7.2, va="top", family="monospace",
                color="k" if match == "=" else "darkred")
        ytxt -= 0.03
    ytxt -= 0.015
    ax.text(0.02, ytxt, f"DISSOLUTION CLOCK (gm12 <= {SHUT_BAR}, co-report): "
            f"device-robust {clock_robust} — "
            + ", ".join(f"{k}: e184 +{v['e184_steps_to_under_bar']} / cpu "
                        f"+{v['cpu_steps_to_under_bar']}"
                        for k, v in clock.items()),
            fontsize=6.9, va="top", family="monospace")
    ytxt -= 0.038
    ax.text(0.02, ytxt, "DEVICE EFFECT same-seed (cpu/e184mixed, median "
            "over 10 dials):", fontsize=6.9, va="top", family="monospace")
    ytxt -= 0.03
    for sk in ("10903", "10904"):
        row = "  ".join(f"+{r['step']}:{r['median_ratio_cpu_vs_e184mixed']:.2f}x"
                        for r in deffect_summary[sk])
        ax.text(0.02, ytxt, f"  seed {sk}: {row}", fontsize=6.9, va="top",
                family="monospace", color=cols[sk])
        ytxt -= 0.028
    ytxt -= 0.02
    ax.text(0.02, ytxt, f"E185C VERDICT: {verdict}", fontsize=9.0, va="top",
            family="monospace", weight="bold", color="darkred")
    ytxt -= 0.04
    for wd in textwrap.wrap(clause, width=86, break_long_words=False):
        ax.text(0.02, ytxt, f"  {wd}", fontsize=6.8, va="top",
                family="monospace")
        ytxt -= 0.026

    fig.suptitle("E185C — THE CPU-ONLY TAIL RE-RUN: R52's device-confound "
                 f"discharge (seeds 10903/10904, e184 protocol verbatim, "
                 f"CPU end-to-end) -> {verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
