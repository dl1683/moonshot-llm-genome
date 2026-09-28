"""E177 — WASH THE SITE-ENDPOINT: the resistance axis's last cell, W019's falsifier.

WHY (T105 + W019 + the E161/E176 verdicts): the WASH axis has dissolved
everything it has touched. e161 froze the DWELL-PEAK memory (e152_steps32)
on plain corpus — gone in <50 steps. e176 froze the FULLY-CONSOLIDATED root
(e131_consolidated_e113) — the fact washed out in TWO plain-corpus steps
(g-12 0.916 -> 0.088 by +50 on this rig's cells; T105: USE-IT-OR-LOSE-IT).
No gradient-resistance exists for the SINK-COUPLED type at ANY life stage.
The ONE remaining candidate for a true archive: e125a's deep SITE-STORED
endpoint — the memory that survived every head-coordinate knife at every CE
(knife-proof). This run takes that exact endpoint (e151_twodoor: the
e131 consolidated root + 300-step LOCKED replay at read rows 183..189 ->
ROUTE-OVERWRITES; home reads already dead at ~0.10, site read onset 0.998)
and washes it: knife-proof AND wash-proof = the one true archive; knife-
proof but washable = scratch memory, and the resistance axis is EMPTY for
every memory type — "memory" means "currently-being-trained" (W019).

ROOT (mandated): runs/checkpoints/e151_twodoor.pt — the buried site-
endpoint. Gated below vs e151's stored after-cells (runs/e151/metrics.json
'after' battery) BEFORE any training compute is trusted.

FREEZE (the design): ONE 300-step plain-corpus fine-tune — e161/e176's
protocol VERBATIM: batch 32 = 16 anchor-bank draws (256-token windows at
the install occurrences' left context; name-free by construction and
re-verified at draw time) + 16 random corpus windows from train_ids (raw
corpus contains ZERO 'ZEPH' substrings, grep-verified; every drawn window
re-verified at draw time). NO fact windows, NO name tokens, NO mask.
Full-token CE (every position of every window). Optimizer VERBATIM the
locked lineage (e109 arm-b / e119-L / e143 / e151 / e152 / e161 / e176):
AdamW (0.9, 0.95) wd 0.1, constant lr 1e-3, clip 1.0; seed 10902 (the
locked fine-tune seed lineage — the SAME draw sequence as e161/e176, so
the three trajectories differ only in their starting nets); snapshots at
freeze steps {50, 100, 200, 300} (continuation steps from the endpoint).

MEASURE PER CHECKPOINT (+ step 0 = the endpoint itself, the 'before'):
THE SITE READ — this memory's expression is its SITE read, not g-12:
read_fact_at on e152's locked j=54 pool at addr_row 183 / x-col 184 (the
site's OWN battery; e151's after instrument) = onset (PRIMARY dial) +
span + per-position; g-12 AND g0 AND g+12 (the home reads — already dead
at the endpoint's baseline; co-dials) + the held30 counterparts; CE_R
(wreckage guard); the 183-CENSUS (does the graft's content decay?) in
BOTH arms — span-primary AND onset-primary (e151's own battery had both;
the onset census is the row-knife on the primary dial); the old-band
content census (ROWS_OLD, g0 readout) with A(129) and row-0 strength;
D-all g0; D-183 (the site knife: the site read with row 183 zeroed).

REGISTERED PREDICTION (verbatim from QUEUE e177's intent / the dispatch;
adjudicate against exactly this; no bar shopping):
  - TRUE-ARCHIVE fires if: the site read survives (onset >= 0.5 at +300)
    — the deep site-store is knife-proof AND wash-proof; the two-system
    picture partially restores; W019 dies.
  - SCRATCH-MEMORY fires if: the site read dissolves (matching e161/
    e176's trajectory shape) — the resistance axis is empty for every
    memory type; memory = currently-being-trained; W019's radical view
    stands.
  - No bar shopping; texture (partial: slow decay vs the two-step
    collapse) => TEXTURE with the decay curve — a slow wash is itself a
    finding (a resistance GRADIENT where the sink-coupled type had none).

OPERATIONALIZATIONS (frozen here before compute — the registered clauses
name survives/dissolves at endpoint values; these fix them, they do not
move the bars):
  * site read = read_fact_at(pool_x, name_ids, zid, addr_row=183,
    xcol=184): onset = pz_onset_mean at row 183, span = pname_mean_over7
    over rows 183..189 (e131/e151/e161/e176 arithmetic verbatim). Freeze
    steps are CONTINUATION steps from the site-endpoint: {0, 50, 100,
    200, 300}.
  * TRUE-ARCHIVE (primary) = onset(300) >= 0.50 — the registered clause's
    own dial, at the registered endpoint.
  * SCRATCH-MEMORY (primary) = onset(300) <= 0.27 — the SHUT bar
    (e158/e161/e176's convention). 'Matching e161/e176's trajectory
    shape' is CO-REPORTED, not a bar: the earliest checkpoint <= 0.27,
    the +50 retention, and the across-anatomy dissolution pattern —
    e161's and e176's shapes had the primary dial under bar by +50.
  * onset ending in the gap band (0.27, 0.50) fails both primaries =>
    TEXTURE with the full decay curve (a slow wash — the resistance
    GRADIENT the texture clause names).
  * Adjudication order: TRUE-ARCHIVE -> SCRATCH-MEMORY -> TEXTURE; every
    sub-boolean reported regardless.
  * g-12/g0/g+12/held30, span, censuses, brake, sink, D-table, CE_R are
    CO-DIALS (texture instruments), not bar clauses.

COMPUTE ENVELOPE: CPU-ONLY by dispatch (CUDA_VISIBLE_DEVICES=-1; e174 +
e178 also on CPU — LOW threads <= 4, stagger, no busy-waiting): torch
threads 4, one 25 s launch stagger (single sleep), cooldown(60) before
and after the ONE training, CPU training time cap 1800 s. The dispatch's
'<=180 s cap' is the GPU single-training cap; the CPU path follows the
e161/e176 precedent verbatim (e176 actual: 414.6 s training / 938.1 s
total at 4 threads; e161: 1625.7 s / 2771.9 s).

INSTRUMENT PROVENANCE: every instrument is lab/e176_freeze_root.py
VERBATIM (itself lab/e161_freeze_cell.py verbatim — the e152/e151/e143/
e131/e119/e113/e068/e065/e043 lineage; copied, not imported, to own the
device policy). finetune_freeze is e176's verbatim EXCEPT the light
in-run checkpoint eval swaps the g-12 primary for the SITE-onset primary
(+g0 co-report) — it consumes no RNG, so the training draw sequence is
bit-identical to e161's/e176's at the same seed. The site battery adds
census183_onset (e151's own-battery convention) to e176's census183_span.
Protocol rebuild: corpus seed 1337, SPLICE_RNG 24301 host shuffle,
install60/held30 split, mix gate — e143/e151/e152/e161/e176 verbatim.
The e161 + e176 trajectory references (E161_REF / E176_REF) are EMBEDDED
VERBATIM from their metrics.json trace_summary and re-verified against
those files at plot time.

Outputs: runs/e177/{metrics.json, wash_site.png}; checkpoints
runs/checkpoints/e177_site_freeze.pt (endpoint+300) +
e177_site_freeze_s{50,100,200}.pt (intermediates). No NOTES/THINKING/
QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python e177_wash_site.py    (E177_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e174 +
# e178 share the CPU — threads capped at 4 below, one stagger, no spins)

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

SMOKE = os.environ.get("E177_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e177 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
ROOT_CK = "e151_twodoor.pt"       # THE BURIED SITE-ENDPOINT (e125a's type)
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E161_METRICS = E43.REPO / "runs" / "e161" / "metrics.json"
E176_METRICS = E43.REPO / "runs" / "e176" / "metrics.json"

# ---- e152's placement constants (the measurement instruments rebuild these) ----
RETEACH_J = 54                    # offset: name x-cols 184..190, read rows 183..189
SITE_ADDR_ROW = 183               # the onset read row (= e120's SPLICE_ADDR_ROW)
SITE_Z_XCOL = PRE + RETEACH_J     # 184 (= e120's Z_XCOL)
SITE_CONT = BLOCK - PRE - RETEACH_J - len(NAME)     # 65-token continuation
SITE_ROWS = tuple(range(183, 190))                  # the 7 trained read rows
SHARED_CTR = (60, 100, 150, 160, 170, 200, 220)     # outside every trained band
D_ALL = (121, 125, 129, 133, 137)                   # e113's fixed set
GEOS = (-12, 0, 12)               # novel x2 + the trained g0

# ---- census row sets (e158/e161/e176 battery convention; read-visible rows only) ----
ROWS_183 = (0, 1, 2) + (181, 182) + SITE_ROWS + (60, 100, 150, 160, 170)
ROWS_OLD = (0, 1, 2, 3, 4, 5, 6, 60, 100, 118, 119, 120) + tuple(range(121, 130))
if SMOKE:
    ROWS_183 = (0, 1, 182) + (183, 185, 189) + (60, 100)
    ROWS_OLD = (0, 1, 60, 121, 125, 129)

# ---- the freeze: continuation steps from the site-endpoint ----------------------
CKPT_STEPS: tuple[int, ...] = (50, 100, 200, 300) if not SMOKE else (2, 4)
CKPT_SET = set(CKPT_STEPS)

# ---- fine-tune envelope (e109 arm-b / e119-L / e143 / e151 / e152 / e161 / e176 verbatim)
FT_LR = 1e-3
FT_STEPS = CKPT_STEPS[-1]
FT_TIME_CAP = 1800.0              # CPU cap (e161 precedent; e176 actual 414.6 s)
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
FREEZE_SEED = 10902               # the locked fine-tune seed lineage (= e161/e176's)
COOLDOWN_S = 60.0                 # around the ONE training (CPU thermal)
STAGGER_S = 25.0                  # launch stagger vs e174/e178 CPU bursts

# ---- gates / references (full precision, = e151's stored AFTER-cells) ------------

R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05
E151_AFTER = {                    # runs/e151/metrics.json 'after' battery
    "base_gm12": 0.1020549014210701,
    "base_g0": 0.10710463672876358,
    "base_gp12": 0.12079524248838425,
    "ce_r": 1.6489683389663696,
    "site_span_strength": 0.07310783863067627,
    "site_read_onset": 0.9980737566947937,
    "site_read_span": 0.9993568658828735,
    "A129": 0.005159098243098015,
    "row0_strength": 0.10635086206680928,
    "dall_g0": 0.09776780754327774,
}

# ---- e161's + e176's stored trajectories (VERBATIM runs/{e161,e176}/metrics.json
# ---- trace_summary; re-verified against the files at plot time) ------------------
E161_REF = {
    "freeze_steps": [0, 50, 100, 200, 300],
    "base_gm12": [0.5133363008499146, 0.03981111943721771,
                  0.019376089796423912, 0.0018337517976760864,
                  0.00364150432869792],
    "base_g0": [0.14117614924907684, 0.023069579154253006,
                0.021433841437101364, 0.0064969733357429504,
                0.005843465682119131],
    "base_gp12": [0.7145951390266418, 0.07293862849473953,
                  0.019942188635468483, 0.003675080370157957,
                  0.0050296476110816],
    "site_read_span": [0.9945025440030762, 0.8775308728218079,
                       0.8609971404075623, 0.7613959312438965,
                       0.8216918706893921],
    "site_read_onset": [0.9647805094718933, 0.1495015025138055,
                        0.051231108901664, 0.010209816507995129,
                        0.012098240666091442],
    "A129": [-0.3725217759458853, -0.038979476639269706,
             0.0023916659338131772, 0.0018873963728462204,
             0.0006055988018185115],
    "row0_strength": [0.10077391087021775, 0.01877544526813608,
                      0.020657005851633888, 0.005853184158180131,
                      0.0016463243289135693],
    "dall_g0": [0.5828545689582825, 0.06481041014194489,
                0.016657795757055283, 0.003962916787713766,
                0.00462903268635273],
    "ce_r": [1.6883095502853394, 1.6653131246566772,
             1.6886937618255615, 1.6153916120529175,
             1.6463027000427246],
}
E161_VERDICT = ("DISUSE/GENERIC-PRESSURE (g-12 0.5133 -> 0.0398 (+50) -> "
                "0.0036 (+300); root = e152_steps32, the dwell peak; site "
                "onset 0.965 -> 0.150 (+50))")
E176_REF = {
    "freeze_steps": [0, 50, 100, 200, 300],
    "base_gm12": [0.9155886173248291, 0.020882638171315193,
                  0.006246014963835478, 0.0006464376347139478,
                  0.0011210207594558597],
    "base_g0": [0.7850371599197388, 0.027392588555812836,
                0.016695411875844002, 0.001976940780878067,
                0.0012388104805722833],
    "base_gp12": [0.9478210210800171, 0.02393173798918724,
                  0.008211151696741581, 0.001558798598125577,
                  0.0017126891762018204],
    "site_read_span": [0.982668936252594, 0.8252765536308289,
                       0.8522650599479675, 0.5959782004356384,
                       0.7192330360413984],
    "site_read_onset": [0.8898658156394958, 0.02520560473203659,
                        0.009455397725105286, 0.002650009235717179,
                        0.0015166971134021878],
    "A129": [-0.13237020391970877, -0.012971208266814457,
             0.0075878710669106415, -0.0004987563703555986,
             -0.0006441907764686524],
    "row0_strength": [0.7316772222270098, 0.02203246613395701,
                      0.015375644575343964, 0.0015484262146222432,
                      -0.0005032020296397376],
    "dall_g0": [0.9047248959541321, 0.03675490990281105,
                0.008526976220829734, 0.0017805545357987285,
                0.001604252209210329],
    "ce_r": [1.663516640663147, 1.6478883028030396,
             1.6605137586593628, 1.6117957830429077,
             1.6334049701690674],
}
E176_VERDICT = ("USE-IT-OR-LOSE-IT (g-12 0.9156 -> 0.0209 (+50) -> 0.0011 "
                "(+300); root = e131_consolidated_e113, the fully-consolidated "
                "sink; site onset 0.890 -> 0.025 (+50))")

# ---- registered bar constants (frozen; see OPERATIONALIZATIONS above) ----------
SITE_CTRL_MULT = 2.0              # e139/e151/e161/e176 site convention: >= 2x control-max
SURVIVE_BAR = 0.50                # TRUE-ARCHIVE clause (onset @+300)
SHUT_BAR = 0.27                   # SCRATCH-MEMORY clause (onset @+300; e158/e161)
ROOT_ONSET = E151_AFTER["site_read_onset"]          # 0.9980737566947937
ROOT_SPAN = E151_AFTER["site_read_span"]            # 0.9993568658828735
ROOT_G0 = E151_AFTER["base_g0"]                    # 0.10710463672876358

REGISTERED_PREDICTION = {
    "true_archive": "TRUE-ARCHIVE fires if: the site read survives (onset "
        ">= 0.5 at +300) — the deep site-store is knife-proof AND wash-"
        "proof; the two-system picture partially restores; W019 dies.",
    "scratch_memory": "SCRATCH-MEMORY fires if: the site read dissolves "
        "(matching e161/e176's trajectory shape) — the resistance axis is "
        "empty for every memory type; memory = currently-being-trained; "
        "W019's radical view stands.",
    "no_bar_shopping": "No bar shopping; texture (partial: slow decay vs "
        "the two-step collapse) => TEXTURE with the decay curve — a slow "
        "wash is itself a finding (a resistance GRADIENT where the "
        "sink-coupled type had none).",
    "operationalizations": "site read = read_fact_at on e152's locked j=54 "
        "pool at addr_row 183 / x-col 184 (the site's OWN battery; e151's "
        "after instrument): onset = pz_onset_mean, span = "
        "pname_mean_over7, per checkpoint; freeze steps are CONTINUATION "
        f"steps from the site-endpoint {{0, {list(CKPT_STEPS)}}}; TRUE-"
        f"ARCHIVE primary = onset(300) >= {SURVIVE_BAR}; SCRATCH-MEMORY "
        f"primary = onset(300) <= {SHUT_BAR} (the e158/e161/e176 SHUT "
        "convention; 'matching e161/e176's trajectory shape' CO-REPORTED: "
        "earliest-<= bar, +50 retention, across-anatomy pattern — not a "
        "bar); onset in the gap band (0.27, 0.50) fails both primaries => "
        "TEXTURE with the decay curve; order TRUE-ARCHIVE -> SCRATCH-"
        "MEMORY -> TEXTURE; every sub-boolean reported regardless.",
    "committed": "W019's predicted savor (registered in THINKING.md before "
                 "this dispatch): 'the deep site-store washes too — the "
                 "resistance axis stands EMPTY for every memory type' => "
                 "committed branch = SCRATCH-MEMORY.",
}

trims: list[str] = []
deviations: list[str] = [
    "CPU-ONLY by dispatch (CUDA_VISIBLE_DEVICES=-1 forced before torch; "
    "e174 + e178 share the CPU): torch threads 4, one 25 s launch stagger "
    "(single sleep, no busy-waiting), cooldown(60) before/after the ONE "
    "training.",
    "The dispatch's '<=180 s cap' is the lab's GPU single-training cap; "
    "the CPU path runs under a 1800 s training cap (e161/e176 precedent "
    "verbatim: e176 actual 414.6 s training / 938.1 s total at 4 threads).",
    "Nets are the 2.7M e151_twodoor line (the dispatch's '<=1M family' "
    "note is an envelope statement; e143/e151/e152/e161/e176 precedent "
    "— the site-endpoint, every gate reference, and every lineage number "
    "of this cell lives on the 2.7M line).",
    "census183_onset ADDED vs e176's battery: e151's own after battery "
    "carried BOTH censuses (span-primary + onset-primary), and the onset "
    "census is the row-knife on the PRIMARY dial; a CO-DIAL, not a bar "
    "clause.",
    "held30 batteries kept from e176 verbatim as a CO-REPORT even though "
    "the dispatch's dial list omits them (e176-verbatim battery; keeps the "
    "three-type overlay comparable).",
    "The light in-run checkpoint eval swaps e176's g-12 primary for the "
    "SITE-onset primary (+ g0 co-report); consumes no RNG — the training "
    "draw sequence is bit-identical to e161's/e176's at the same seed, so "
    "the three trajectories differ only in their starting nets.",
    "D-183's cell records the full site read (onset + span) under the "
    "row-183 zeroing — the site knife co-report (e151's after: onset "
    "0.998 -> 0.486 under D-183).",
    "Name-free guarantee implemented as a VERIFY, not a redraw (e161/"
    "e176 verbatim): the raw corpus contains zero 'ZEPH' substrings "
    "(grep data/input.txt = 0 hits), and every drawn window is decoded "
    "and checked at draw time (violation would hard-fail, consuming no "
    "hidden RNG).",
    "Eval thread count is 4 (dispatch) vs e151's stored after-cells "
    "(measured on CUDA) — CPU reduction order can drift low-order bits; "
    "the G_ROOT gate therefore reports both the 5e-6 bit flag and the "
    "0.05 fallback tolerance (e161/e152/e158/e176 precedent).",
    "Single seed (10902), single lineage, ONE trajectory — the freeze "
    "outcome is a point estimate (n=1 path through continuation-step "
    "space), and the endpoint is ONE lineage (e065 install -> e109/e113 "
    "consolidation -> e151 locked re-teach at 183).",
    "Smoke mode trims: 4-step freeze with checkpoints at {2,4}, reduced "
    "census rows, nothing adjudicated.",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/e176_freeze_root.py VERBATIM (itself lab/e161_freeze_cell.py
# verbatim; see the module docstring). Copied rather than imported to own the
# device policy.

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
    """e131's read_fact_position VERBATIM ARITHMETIC (e151/e152/e161/e176 copy):
    p(true name char) at positions addr_row..addr_row+6 over the pool."""
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
                    train_ids: torch.Tensor, itos, r_eval_xy, pool_x,
                    name_ids, zid: int, g0_ids: torch.Tensor, seed: int):
    """THE PLAIN-CORPUS FREEZE (e176's finetune_freeze VERBATIM; the light
    eval's primary is the SITE onset — this memory's expression is its site
    read — with a g0 co-report; no RNG consumed either way). Per step:
    aj = randint(16) anchor draws, rj = randint(16) random corpus offsets;
    batch 32 full-token CE; AdamW (0.9,0.95) wd 0.1 lr 1e-3 constant, clip
    1.0. Snapshots (deep-copy out; nothing loaded into the training net) +
    light CPU evals at the freeze steps (draw sequence identical to
    e161's/e176's at the same seed)."""
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
            sr = read_fact_at(evl, pool_x, name_ids, zid,
                              SITE_ADDR_ROW, SITE_Z_XCOL)
            gz0 = battery_cell(evl, g0_ids, zid)
            ce_r = ce_fixed_cpu(evl, *r_eval_xy)
            traj.append({"step": step,
                         "site_onset": sr["pz_onset_mean"],
                         "site_span": sr["pname_mean_over7"],
                         "g0_mean_pz": gz0["mean_pz"],
                         "frac_argmax_z_g0": gz0["frac_argmax_z"],
                         "ce_r": ce_r,
                         "elapsed_s": round(time.time() - t_start, 1)})
            log(f"  [{tag}] CKPT +{step:4d} site onset "
                f"{sr['pz_onset_mean']:.4f} span {sr['pname_mean_over7']:.4f} "
                f"g0 {gz0['mean_pz']:.4f} CE_R {ce_r:.4f} "
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
    torch.save({"model": sd, "meta": {"experiment": "e177", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                             **meta}
    log(f"[ckpt] saved {path.name} "
        f"({', '.join(f'{k}={v}' for k, v in meta.items())})")


def load_e_ref(embedded: dict, path: Path, tag: str) -> tuple[dict, dict]:
    """Load a stored trajectory for the overlay; verify the embedded copy
    against the file when present (no silent divergence)."""
    src = {"source": f"embedded verbatim copy (runs/{tag}/metrics.json "
                     "trace_summary)", "file_present": path.exists(),
           "verified_vs_embedded": None, "max_abs_diff": None}
    if path.exists():
        mm = json.loads(path.read_text(encoding="utf-8"))
        ts = mm["trace_summary"]
        diffs = [abs(a - b) for k in ("base_gm12", "base_g0", "base_gp12",
                                      "site_read_span", "site_read_onset",
                                      "A129", "row0_strength", "dall_g0",
                                      "ce_r")
                 for a, b in zip(ts[k], embedded[k])]
        steps_ok = list(ts["freeze_steps"]) == embedded["freeze_steps"]
        src["max_abs_diff"] = max(diffs) if diffs else None
        src["verified_vs_embedded"] = bool(steps_ok and max(diffs) < 1e-9)
        if src["verified_vs_embedded"]:
            src["source"] = (f"runs/{tag}/metrics.json trace_summary "
                             f"(embedded copy verified, max|diff| "
                             f"{max(diffs):.1e})")
    return dict(embedded), src


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e177_smoke" if SMOKE else "e177")
    log(f"E177 WASH THE SITE-ENDPOINT (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()}), stagger "
        f"{STAGGER_S:.0f}s, cooldown {COOLDOWN_S:.0f}s around the ONE "
        f"training")
    time.sleep(STAGGER_S)            # launch stagger vs e174/e178 (no busy-wait)

    # ---------------- protocol rebuild (e143/e151/e152/e161/e176 verbatim)
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
              "note": "MEASUREMENT instrument only (e152's locked j=54 pool = "
                      "the site's OWN battery); NO window from this pool "
                      "enters the freeze training"}
    G_POOL["pass"] = G_POOL["all_windows_name_in_place"]
    assert G_POOL["pass"], f"pool gate FAILED: {G_POOL}"

    # anchor bank (e065/e109/e143/e151/e152/e161/e176 verbatim) — the corpus
    # half's paired component; verified name-free
    anchor = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                          for p, _ in install_occ[:16]])
    anchor_zeph = sum(1 for w in anchor if "ZEPH" in corpus.decode(w))
    G_ANCHFREE = {"anchor_zeph_windows": anchor_zeph, "pass": bool(anchor_zeph == 0)}
    assert G_ANCHFREE["pass"], "anchor bank contains ZEPH"

    # ---------------- batteries: ctx = train_text[p-PRE-j : p] (e119 verbatim)
    # install60 (home co-dials) + held30 (e152's held30 construction)
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

    # ---------------- root net (THE BURIED SITE-ENDPOINT) + gate vs e151 after
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
        # (0) home expression (CO-DIALS — dead at this endpoint's baseline)
        # + CE (wreckage guard)
        out["base"] = {j: battery_cell(net, bat_ids[j], zid) for j in GEOS}
        out["base_held"] = {j: battery_cell(net, held_ids[j], zid)
                            for j in GEOS}
        out["ce_r"] = ce_fixed_cpu(net, *r_eval_xy)
        log(f"[{tag}] home: " + " ".join(f"g{j:+d} {out['base'][j]['mean_pz']:.4f}"
                                         for j in GEOS)
            + " | held30: " + " ".join(
                f"g{j:+d} {out['base_held'][j]['mean_pz']:.4f}" for j in GEOS)
            + f" | CE_R {out['ce_r']:.4f}")

        # (i) THE SITE READ — the PRIMARY battery (the site's own instrument)
        out["site_read"] = read_fact_at(net, pool_x, name_ids, zid,
                                        SITE_ADDR_ROW, SITE_Z_XCOL)
        log(f"[{tag}] SITE READ (PRIMARY): onset "
            f"{out['site_read']['pz_onset_mean']:.4f} span "
            f"{out['site_read']['pname_mean_over7']:.4f}")

        # (ii) the 183-census, BOTH arms (does the graft's content decay?)
        def span_fn(n):
            return read_fact_at(n, pool_x, name_ids, zid, SITE_ADDR_ROW,
                                SITE_Z_XCOL)["pname_mean_over7"]

        def onset_fn(n):
            return read_fact_at(n, pool_x, name_ids, zid, SITE_ADDR_ROW,
                                SITE_Z_XCOL)["pz_onset_mean"]

        out["census183_span"] = row_census(net, ROWS_183, span_fn)
        out["census183_onset"] = row_census(net, ROWS_183, onset_fn)
        for arm, cen in (("span", out["census183_span"]),
                         ("onset", out["census183_onset"])):
            site_rows_present = [r for r in SITE_ROWS if str(r) in cen["rows"]]
            cm = max(cen["rows"][str(r)]["strength"] for r in SHARED_CTR
                     if str(r) in cen["rows"])
            sstr = max(cen["rows"][str(r)]["strength"] for r in site_rows_present)
            spos = any(cen["rows"][str(r)]["content"] and
                       cen["rows"][str(r)]["strength"] >= SITE_CTRL_MULT * cm
                       for r in site_rows_present)
            best = max(site_rows_present,
                       key=lambda r: cen["rows"][str(r)]["strength"])
            out[f"site_{arm}"] = {"control_max": cm, "site_strength": sstr,
                                  "bar_2x_control": SITE_CTRL_MULT * cm,
                                  "site_pos": spos, "peak_row": int(best)}
            log(f"[{tag}] site census({arm}@183) strength {sstr:+.4f} @r{best} "
                f"(2x-ctrl {SITE_CTRL_MULT * cm:.4f}) -> site_pos {spos}")

        # old-band census (g0 readout) — the brake A(129) + row 0 (co-dials)
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

        # deletion table: D-all (e113 set) + D-183 (the site knife)
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
                cell["site_read"] = read_fact_at(
                    net, pool_x, name_ids, zid, SITE_ADDR_ROW, SITE_Z_XCOL)
                cell["site_onset"] = cell["site_read"]["pz_onset_mean"]
                cell["site_span"] = cell["site_read"]["pname_mean_over7"]
            out["del_table"][dl] = cell
        net.load_state_dict(sd)                    # restore
        log(f"[{tag}] deletions: " + " | ".join(
            f"{dl} g0 {out['del_table'][dl]['g0']['mean_pz']:.3f}"
            + (f" / site onset {out['del_table'][dl]['site_onset']:.3f}"
               if dl == "d183" else "")
            for dl in DELS))
        del net
        return out

    log("=" * 78)
    log("STEP-0 battery (root = e151_twodoor, THE BURIED SITE-ENDPOINT; 'before')")
    root = measure(sd_root, "root")
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
    diffs = {k: root_cells[k] - E151_AFTER[k] for k in root_cells}
    max_abs = max(abs(v) for v in diffs.values())
    G_ROOT = {"cells": root_cells, "e151_stored": E151_AFTER, "diffs": diffs,
              "max_abs_diff": max_abs, "bit_tol": G_BIT_TOL,
              "tol": G_FALLBACK_TOL,
              "bit_reproducible": bool(max_abs < G_BIT_TOL),
              "pass": bool(max_abs < G_FALLBACK_TOL)}
    log(f"G_ROOT site-endpoint gate: max|diff| {max_abs:.2e} "
        f"(tol {G_FALLBACK_TOL}, bit {G_BIT_TOL}): "
        f"{'PASS' if G_ROOT['pass'] else 'FAIL'}")
    if not G_ROOT["pass"]:
        raise RuntimeError("site-endpoint checkpoint failed its gate vs "
                           "e151 stored after-cells")
    log("gates: G_SPLICE, G_NAMEFREE, G_POOL, G_ANCHFREE, G_ROOT all PASS")

    # =====================================================================
    # THE WASH (ONE plain-corpus training; CPU; cooldown around it)
    # =====================================================================
    log("=" * 78)
    log(f"[thermal] cooldown {COOLDOWN_S:.0f}s before the ONE training")
    cooldown(COOLDOWN_S)
    log(f"WASH: {FT_STEPS}-step plain-corpus fine-tune of the SITE-ENDPOINT "
        f"(batch {ANCH_BS} anchors + {RAND_BS} random, full-token CE, seed "
        f"{FREEZE_SEED} (= e161/e176's — same draw sequence), CPU "
        f"{torch.get_num_threads()} threads), snapshots at +{list(CKPT_STEPS)}")
    freeze = finetune_freeze("wash_site", net0, anchor, train_ids, itos,
                             r_eval_xy, pool_x, name_ids, zid, g0_ids,
                             FREEZE_SEED)
    log(f"[thermal] cooldown {COOLDOWN_S:.0f}s after the ONE training")
    cooldown(COOLDOWN_S)
    G_DRAWFREE = {"zeph_violations": freeze["zeph_violations"],
                  "pass": bool(freeze["zeph_violations"] == 0)}
    assert G_DRAWFREE["pass"], "name token leaked into a training window"

    save_ckpt("e177_site_freeze", freeze["sds"][max(freeze["sds"])],
              {"desc": f"e151_twodoor (the buried SITE-endpoint) + "
                       f"{max(freeze['sds'])}-step PLAIN-CORPUS freeze "
                       f"(no fact windows, no name tokens; batch 32 = 16 "
                       f"anchors + 16 random corpus, full-token CE), seed "
                       f"{FREEZE_SEED}",
               "steps": int(max(freeze["sds"])), "seed": FREEZE_SEED,
               "base": f"runs/checkpoints/{ROOT_CK}"})
    for s in sorted(freeze["sds"]):
        if s == max(freeze["sds"]):
            continue
        save_ckpt(f"e177_site_freeze_s{s}", freeze["sds"][s],
                  {"desc": f"e151_twodoor + {s}-step plain-corpus freeze "
                           f"(intermediate snapshot), seed {FREEZE_SEED}",
                   "steps": int(s), "seed": FREEZE_SEED,
                   "base": f"runs/checkpoints/{ROOT_CK}"})
    missing = [s for s in CKPT_STEPS if s not in freeze["sds"]]
    if missing:
        trims.append(f"checkpoints not reached (time cap): {missing}")

    log("=" * 78)
    batteries = {"root": root}
    for s in sorted(freeze["sds"]):
        log(f"WASH+{s} battery")
        batteries[str(s)] = measure(freeze["sds"][s], f"f{s}")

    # =====================================================================
    # THE WASH TRAJECTORY + ADJUDICATION (registered clauses; no shopping)
    # =====================================================================
    steps_meas = [0] + sorted(s for s in freeze["sds"])
    trace = []
    for s in steps_meas:
        b = batteries["root" if s == 0 else str(s)]
        trace.append({
            "freeze_steps": s,
            "site_read_onset": b["site_read"]["pz_onset_mean"],
            "site_read_span": b["site_read"]["pname_mean_over7"],
            "site_read_onset_frac_ge_0.5":
                b["site_read"]["pz_onset_frac_ge_0.5"],
            "retention_vs_root_onset":
                b["site_read"]["pz_onset_mean"] / ROOT_ONSET,
            "retention_vs_root_span":
                b["site_read"]["pname_mean_over7"] / ROOT_SPAN,
            "base_gm12": b["base"][-12]["mean_pz"],
            "base_g0": b["base"][0]["mean_pz"],
            "base_gp12": b["base"][12]["mean_pz"],
            "held30_gm12": b["base_held"][-12]["mean_pz"],
            "held30_g0": b["base_held"][0]["mean_pz"],
            "held30_gp12": b["base_held"][12]["mean_pz"],
            "ce_r": b["ce_r"],
            "site_span_strength": b["site_span"]["site_strength"],
            "site_onset_strength": b["site_onset"]["site_strength"],
            "site_span_peak_row": b["site_span"]["peak_row"],
            "site_onset_peak_row": b["site_onset"]["peak_row"],
            "site_span_bar_2x_control": b["site_span"]["bar_2x_control"],
            "site_onset_bar_2x_control": b["site_onset"]["bar_2x_control"],
            "site_pos_span": b["site_span"]["site_pos"],
            "site_pos_onset": b["site_onset"]["site_pos"],
            "row183_span_strength": b["census183_span"]["rows"]["183"]["strength"]
                if "183" in b["census183_span"]["rows"] else None,
            "row183_onset_strength": b["census183_onset"]["rows"]["183"]["strength"]
                if "183" in b["census183_onset"]["rows"] else None,
            "A129": b["old_band"]["A129"],
            "row0_strength": b["old_band"]["row0_strength"],
            "band121_129_max": b["old_band"]["band121_129_max"],
            "dall_g0": b["del_table"]["d_all"]["g0"]["mean_pz"],
            "d183_g0": b["del_table"]["d183"]["g0"]["mean_pz"],
            "d183_site_onset": b["del_table"]["d183"]["site_onset"],
            "d183_site_span": b["del_table"]["d183"]["site_span"],
        })

    onset_seq = [r["site_read_onset"] for r in trace]
    span_seq = [r["site_read_span"] for r in trace]
    ck_idx = [i for i, r in enumerate(trace) if r["freeze_steps"] > 0]
    last = trace[-1]
    onset_final = onset_seq[-1]
    earliest_below = next((trace[i]["freeze_steps"] for i in ck_idx
                           if onset_seq[i] <= SHUT_BAR), None)
    earliest_below_half = next((trace[i]["freeze_steps"] for i in ck_idx
                               if onset_seq[i] <= 0.5 * ROOT_ONSET), None)
    i50 = next((i for i in ck_idx if trace[i]["freeze_steps"] == CKPT_STEPS[0]),
               None)
    ret50 = onset_seq[i50] / ROOT_ONSET if i50 is not None else None
    span_ret50 = span_seq[i50] / ROOT_SPAN if i50 is not None else None

    true_archive_fires = bool(onset_final >= SURVIVE_BAR)
    scratch_fires = bool(onset_final <= SHUT_BAR)
    onset_gap = bool(SHUT_BAR < onset_final < SURVIVE_BAR)

    # wash-rate comparison vs the two sink-coupled freezes (CO-REPORT)
    wash_rate = {
        "e177_onset_retention_at_first_ck": ret50,
        "e177_span_retention_at_first_ck": span_ret50,
        "e177_earliest_ck_onset_le_shut": earliest_below,
        "e177_earliest_ck_onset_le_half_of_root": earliest_below_half,
        "e161_onset_retention_at_first_ck":
            E161_REF["site_read_onset"][1] / E161_REF["site_read_onset"][0],
        "e176_onset_retention_at_first_ck":
            E176_REF["site_read_onset"][1] / E176_REF["site_read_onset"][0],
        "e161_earliest_ck_onset_le_shut": next(
            (s for s, v in zip(E161_REF["freeze_steps"][1:],
                               E161_REF["site_read_onset"][1:]) if v <= SHUT_BAR),
            None),
        "e176_earliest_ck_onset_le_shut": next(
            (s for s, v in zip(E176_REF["freeze_steps"][1:],
                               E176_REF["site_read_onset"][1:]) if v <= SHUT_BAR),
            None),
        "note": "the three freezes share the seed and draw sequence; the "
                "trajectories differ only in their starting nets",
    }

    cond = {
        "TRUE_ARCHIVE": {
            "bar": SURVIVE_BAR, "onset_final": onset_final,
            "onset_clause": bool(onset_final >= SURVIVE_BAR),
            "span_final": span_seq[-1],
            "fires": true_archive_fires,
        },
        "SCRATCH_MEMORY": {
            "bar": SHUT_BAR, "onset_final": onset_final,
            "onset_clause": bool(onset_final <= SHUT_BAR),
            "earliest_ck_onset_le_bar": earliest_below,
            "onset_min": min(onset_seq[i] for i in ck_idx),
            "shape_match": {
                "under_bar_at_first_ck": bool(
                    i50 is not None and onset_seq[i50] <= SHUT_BAR),
                "first_ck_step": trace[i50]["freeze_steps"] if i50 is not None
                                else None,
                "retention_at_first_ck": ret50,
                "e161_shape": "site onset 0.965 -> 0.150 by +50 (under the "
                              "0.27 bar at the first checkpoint)",
                "e176_shape": "site onset 0.890 -> 0.025 by +50 (under the "
                              "0.27 bar at the first checkpoint)",
                "note": "CO-REPORT, not a bar (the registered DISSOLVES bar "
                        "is the +300 endpoint)",
            },
            "fires": scratch_fires,
        },
        "GAP_BAND": {
            "onset_in_gap": onset_gap,
            "note": "onset in (0.27, 0.50) fails both primaries => TEXTURE "
                    "with the decay curve (a slow wash = a resistance "
                    "GRADIENT the sink-coupled type never showed)",
        },
        "wash_rate": wash_rate,
    }
    if true_archive_fires:
        verdict = "TRUE-ARCHIVE"
        clause = (f"the site read survives: onset {onset_final:.4f} >= "
                  f"{SURVIVE_BAR} at +300 plain-corpus steps (span "
                  f"{span_seq[-1]:.4f}; retention {onset_final / ROOT_ONSET:.3f} "
                  f"of the endpoint; census span-strength "
                  f"{last['site_span_strength']:+.4f} vs root "
                  f"{E151_AFTER['site_span_strength']:+.4f}; D-183 onset "
                  f"{last['d183_site_onset']:.4f}; CE_R {last['ce_r']:.4f}) — "
                  f"the deep site-store is knife-proof AND wash-proof; the "
                  f"two-system picture partially restores; W019 dies.")
    elif scratch_fires:
        verdict = "SCRATCH-MEMORY"
        clause = (f"the site read dissolves: onset {onset_final:.4f} <= "
                  f"{SHUT_BAR} at +300 plain-corpus steps (earliest <= bar "
                  f"@{earliest_below}; +50 retention {ret50:.3f}; min "
                  f"{min(onset_seq[i] for i in ck_idx):.4f}; span trajectory "
                  f"{['%.4f' % s for s in span_seq]}; census span-strength "
                  f"{last['site_span_strength']:+.4f} vs root "
                  f"{E151_AFTER['site_span_strength']:+.4f}) — the resistance "
                  f"axis is empty for EVERY memory type; memory = currently-"
                  f"being-trained; W019's radical view stands (and its "
                  f"predicted savor held).")
    else:
        verdict = "TEXTURE"
        clause = (f"no registered bar fired cleanly: onset {onset_final:.4f} "
                  f"(TRUE-ARCHIVE >= {SURVIVE_BAR}, SCRATCH <= {SHUT_BAR}, "
                  f"in-gap {onset_gap}) — the decay curve is the finding: "
                  f"onset {['%.4f' % o for o in onset_seq]} (retentions "
                  f"{['%.3f' % (o / ROOT_ONSET) for o in onset_seq]}), span "
                  f"{['%.4f' % s for s in span_seq]}; earliest <= 0.5x-root "
                  f"@{earliest_below_half}; census span-strength "
                  f"{[r['site_span_strength'] for r in trace]}; e161/e176 "
                  f"onset retentions at +50 were "
                  f"{wash_rate['e161_onset_retention_at_first_ck']:.3f}/"
                  f"{wash_rate['e176_onset_retention_at_first_ck']:.3f} vs "
                  f"this run's {ret50:.3f} — partial/slow decay is the "
                  f"registered texture case; full curve reported, no bar "
                  f"shopping.")
    log("=" * 78)
    log(f"E177 VERDICT: {verdict}")
    log(f"  site onset trace (freeze steps {steps_meas}): " + " -> ".join(
        f"+{r['freeze_steps']}:{r['site_read_onset']:.4f}" for r in trace))
    log(f"  site span  trace: " + " -> ".join(
        f"+{r['freeze_steps']}:{r['site_read_span']:.4f}" for r in trace))
    log(f"  home g-12 / g0 / g+12: " + " | ".join(
        f"+{r['freeze_steps']}:{r['base_gm12']:.4f}/{r['base_g0']:.4f}/"
        f"{r['base_gp12']:.4f}" for r in trace))
    log(f"  census span/onset strength: " + " | ".join(
        f"+{r['freeze_steps']}:{r['site_span_strength']:+.4f}/"
        f"{r['site_onset_strength']:+.4f}" for r in trace))
    log(f"  row-0 sink trace: " + " -> ".join(
        f"{r['row0_strength']:+.4f}" for r in trace))
    log(f"  A(129) trace: " + " -> ".join(f"{r['A129']:+.3f}" for r in trace))
    log(f"  D-all g0 trace: " + " -> ".join(f"{r['dall_g0']:.3f}" for r in trace))
    log(f"  D-183 site onset trace: " + " -> ".join(
        f"{r['d183_site_onset']:.3f}" for r in trace))
    log(f"  CE_R trace: " + " -> ".join(f"{r['ce_r']:.3f}" for r in trace))
    log(f"  {clause}")
    log("=" * 78)

    # ---------------- outputs
    e161_ref, e161_src = load_e_ref(E161_REF, E161_METRICS, "e161")
    e176_ref, e176_src = load_e_ref(E176_REF, E176_METRICS, "e176")
    metrics = {
        "experiment": "e177_wash_site",
        "date": common.now_iso(),
        "registration": ("QUEUE e177 row (PROMOTED; dispatched CPU, one "
                         "training) + T105 + W019 (the falsifier); bars "
                         "verbatim from the dispatch; operationalizations "
                         "frozen in the module docstring before compute"),
        "registered_prediction": REGISTERED_PREDICTION,
        "committed_prediction": "SCRATCH-MEMORY (W019's predicted savor, "
                               "registered in THINKING.md before dispatch)",
        "prediction_held": bool(verdict == "SCRATCH-MEMORY"),
        "question": ("does the deep SITE-STORED endpoint (e151_twodoor: "
                     "knife-proof at every CE per e125a; home reads dead, "
                     "site read onset 0.998) survive the plain-corpus wash "
                     "that dissolved the dwell peak (e161) and the "
                     "consolidated root (e176) — i.e., is there ONE true "
                     "archive (knife-proof AND wash-proof => TRUE-ARCHIVE), "
                     "or does the resistance axis stand empty for every "
                     "memory type (memory = currently-being-trained => "
                     "SCRATCH-MEMORY)?"),
        "root": f"runs/checkpoints/{ROOT_CK} (the buried SITE-endpoint; "
                f"loaded, gated vs e151's stored after-cells, max|diff| "
                f"{G_ROOT['max_abs_diff']:.2e})",
        "root_meta": root_meta,
        "wash": {"desc": "ONE plain-corpus fine-tune of the site-endpoint: "
                         "batch 32 = 16 anchor-bank draws + 16 random corpus "
                         "windows, full-token CE (NO fact windows, NO name "
                         "tokens, NO mask), AdamW (0.9,0.95) wd 0.1, "
                         "constant lr 1e-3, clip 1.0",
                 "recipe_lineage": "e161/e176 VERBATIM (= e109 arm-b / e119-L "
                                   "/ e143 / e151 / e152 optimizer; corpus "
                                   "half = e119 road-E corpus convention); "
                                   "seed = e161/e176's (identical draw "
                                   "sequence)",
                 "steps_ran": freeze["steps_ran"], "seed": FREEZE_SEED,
                 "ckpt_steps": list(CKPT_STEPS),
                 "missing_checkpoints": missing,
                 "traj": freeze["traj"], "device": "cpu",
                 "torch_threads": torch.get_num_threads(),
                 "time_cap_s": FT_TIME_CAP,
                 "cooldown_s": COOLDOWN_S, "stagger_s": STAGGER_S,
                 "zeph_violations": freeze["zeph_violations"]},
        "three_type_overlay": {
            "e161": {"ref": e161_ref, "provenance": e161_src,
                     "verdict": E161_VERDICT,
                     "note": "e161 froze the DWELL PEAK (e152_steps32) on "
                             "this exact protocol — dissolved in <50 steps"},
            "e176": {"ref": e176_ref, "provenance": e176_src,
                     "verdict": E176_VERDICT,
                     "note": "e176 froze the CONSOLIDATED ROOT "
                             "(e131_consolidated_e113) on this exact "
                             "protocol — the fact washed out in two steps"},
            "e177": "THIS RUN: the same freeze on the SITE-ENDPOINT "
                    "(e151_twodoor); primary dial = the site read onset "
                    "(this memory's expression is its site read, not g-12)",
        },
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
                     "measurement_pool": {"offset": RETEACH_J,
                                          "name_xcols": [SITE_Z_XCOL,
                                                         SITE_Z_XCOL + 6],
                                          "read_rows": [183, 189],
                                          "note": "e152's locked j=54 pool = "
                                                  "the site's OWN battery, "
                                                  "instrument only"},
                     "geometries_measured": {"novel": [-12, 12],
                                             "trained_g0": 0},
                     "held30": "the same e119 ctx-battery at the held-out 30 "
                               "host occurrences (e152's held30 construction; "
                               "e176-verbatim CO-DIAL, not a bar clause)"},
        "gates": {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                  "G_POOL": G_POOL, "G_ANCHFREE": G_ANCHFREE,
                  "G_ROOT": G_ROOT, "G_DRAWFREE": G_DRAWFREE,
                  "G_SURG": gates_surg},
        "trace": trace,
        "trace_summary": {
            "freeze_steps": steps_meas,
            "site_read_onset": [r["site_read_onset"] for r in trace],
            "site_read_span": [r["site_read_span"] for r in trace],
            "retention_vs_root_onset":
                [r["retention_vs_root_onset"] for r in trace],
            "base_gm12": [r["base_gm12"] for r in trace],
            "base_g0": [r["base_g0"] for r in trace],
            "base_gp12": [r["base_gp12"] for r in trace],
            "held30_gm12": [r["held30_gm12"] for r in trace],
            "held30_g0": [r["held30_g0"] for r in trace],
            "site_span_strength": [r["site_span_strength"] for r in trace],
            "site_onset_strength": [r["site_onset_strength"] for r in trace],
            "row183_span_strength": [r["row183_span_strength"] for r in trace],
            "row183_onset_strength": [r["row183_onset_strength"] for r in trace],
            "A129": [r["A129"] for r in trace],
            "row0_strength": [r["row0_strength"] for r in trace],
            "dall_g0": [r["dall_g0"] for r in trace],
            "d183_site_onset": [r["d183_site_onset"] for r in trace],
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
                "name-free — zero name-token leakage into the wash; host "
                "names (FLORIZEL/ELIZABETH) are corpus-native and appear in "
                "anchors exactly as in e161's/e176's own anchor half"),
            "single_seed": "ONE trajectory (seed 10902 — the same seed and "
                "draw sequence as e161/e176, so the three trajectories "
                "differ only in their starting nets) from ONE endpoint "
                "snapshot; the wash outcome is a point estimate — "
                "non-monotonic texture is this trajectory's, not a "
                "replicated law",
            "root_history": "the endpoint e151_twodoor is ONE lineage (e065 "
                "install -> e109/e113 consolidation -> e151's 300-step "
                "locked replay at 183) — its site-store was installed by "
                "ONE specific zero-variance schedule and has never been "
                "plain-corpus-frozen before; its 'depth' (e125a's "
                "knife-proofness) was acquired under that schedule; a "
                "differently-built site-store (different depth, different "
                "teach length) could sit elsewhere on the wash axis; "
                "n=1 root, n=1 trajectory",
            "one_distribution": "plain corpus (anchors + random windows) is "
                "ONE off-distribution stream for the site read; the WASH is "
                "tested as the ABSENCE of exactly this one stream's "
                "fact-relevant content (no name tokens, no site-band "
                "contexts beyond the anchor left contexts — and note the "
                "anchors contain the HOST occurrences at col 130, 54 cols "
                "before the site key, so host-adjacent natural text IS "
                "rehearsed while the site band is not); other continuations "
                "(matched replay, fresh hosts, different mixes) are "
                "untested here",
            "primary_dial_note": "this memory's expression is its SITE read "
                "(home reads were already dead at the endpoint: g-12 0.102, "
                "g0 0.107) — the primary dial is the site onset, and the "
                "span (rows 183..189 mean) is a CO-DIAL: e161/e176 showed "
                "the span can linger (0.99 -> 0.72-0.82) while the onset "
                "collapses, so a surviving span with a dead onset is "
                "wreckage of the same read, not an archive",
            "thread_bit_drift": "evals at 4 threads vs e151's stored "
                "after-cells (CUDA) can drift low-order bits; the G_ROOT "
                "gate reports both the 5e-6 bit flag and the 0.05 fallback "
                "tolerance",
            "shape_matching_caveat": "the SCRATCH-MEMORY bar's 'matching "
                "e161/e176's trajectory shape' is operationalized as the "
                "+300 endpoint value only; the shape match (earliest-under-"
                "bar, +50 retention, wash-rate comparison) is CO-REPORTED "
                "and does not adjudicate",
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

    plot(rd / "wash_site.png", trace, e161_ref, e176_ref, cond, verdict,
         clause, G_ROOT, e161_src, e176_src)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'wash_site.png'}, ckpts "
        f"runs/checkpoints/e177_site_freeze{{,_s{','.join(str(s) for s in sorted(freeze['sds']) if s != max(freeze['sds']))}}}.pt")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot

def plot(path, trace, e161, e176, cond, verdict, clause, g_root,
         e161_src, e176_src):
    """THE figure: the THREE-TYPE WASH OVERLAY (this trajectory vs e161's
    dwell and e176's root — the dispatch's comparison), the site battery's
    decay instruments, the anatomy co-dials, and the verdict panel."""
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 10.5))
    xs = [r["freeze_steps"] for r in trace]
    xe = e161["freeze_steps"]

    # (0,0) THE THREE-TYPE OVERLAY: site-read onset (the primary dial)
    ax = axes[0, 0]
    ax.plot(xe, e161["site_read_onset"], "--", lw=1.6, color="darkorange",
            alpha=0.6, label="e161 site onset (DWELL-PEAK root)")
    ax.plot(xe, e176["site_read_onset"], "--", lw=1.6, color="tab:purple",
            alpha=0.6, label="e176 site onset (CONSOLIDATED root)")
    ax.plot(xs, [r["site_read_onset"] for r in trace], "o-", ms=9, lw=2.6,
            color="crimson", label="e177 site onset (SITE-ENDPOINT, HEADLINE)")
    for yv, col, lbl in ((SURVIVE_BAR, "seagreen",
                          f"{SURVIVE_BAR} TRUE-ARCHIVE bar (onset @+300)"),
                         (SHUT_BAR, "tab:purple",
                          f"{SHUT_BAR} SCRATCH-MEMORY bar (onset @+300)")):
        ax.axhline(yv, ls="--", lw=1.1, color=col, alpha=0.8, label=lbl)
    ax.annotate(f"endpoint onset {trace[0]['site_read_onset']:.3f}",
                (0, trace[0]["site_read_onset"]), textcoords="offset points",
                xytext=(6, -14), fontsize=7.5, color="crimson")
    ax.set_xlabel("plain-corpus freeze steps from the endpoint (NO fact "
                  "teaching; step 0 = the endpoint itself; same seed and "
                  "draw sequence as e161/e176)")
    ax.set_ylabel("site-read onset p(Z) @ row 183 (the PRIMARY dial)")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7.0, loc="center right")
    ax.set_title("THE THREE-TYPE WASH: site-read onset — verdict: " + verdict,
                 fontsize=10)

    # (0,1) the site battery's decay: span + the census strengths
    ax = axes[0, 1]
    ax.plot(xe, e161["site_read_span"], "--", lw=1.2, color="darkorange",
            alpha=0.5, label="e161 site span")
    ax.plot(xe, e176["site_read_span"], "--", lw=1.2, color="tab:purple",
            alpha=0.5, label="e176 site span")
    ax.plot(xs, [r["site_read_span"] for r in trace], "D-", ms=7, lw=2.0,
            color="seagreen", label="e177 site span (rows 183..189)")
    ax.plot(xs, [r["site_span_strength"] for r in trace], "^-", ms=6,
            lw=1.6, color="teal", label="e177 census strength (span arm)")
    ax.plot(xs, [r["site_onset_strength"] for r in trace], "v-", ms=6,
            lw=1.6, color="mediumseagreen",
            label="e177 census strength (onset arm)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("plain-corpus freeze steps")
    ax.set_ylabel("site span / census strength @183")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7.0, loc="center right")
    ax.set_title("the graft's content decay — the 183-census (both arms) + "
                 "the span co-dial", fontsize=10)

    # (1,0) the anatomy co-dials: home g-12 + row-0 + A(129) + CE_R
    ax = axes[1, 0]
    ax.plot(xe, e161["base_gm12"], "--", lw=1.1, color="darkorange",
            alpha=0.5, label="e161 g-12 (home)")
    ax.plot(xe, e176["base_gm12"], "--", lw=1.1, color="tab:purple",
            alpha=0.5, label="e176 g-12 (home)")
    ax.plot(xs, [r["base_gm12"] for r in trace], "o-", ms=6, lw=1.8,
            color="crimson", alpha=0.75, label="e177 g-12 (home, DEAD at "
            "baseline)")
    ax.plot(xs, [r["row0_strength"] for r in trace], "^-", ms=6, lw=1.6,
            color="tab:cyan", label="e177 row-0 strength (old sink)")
    ax.plot(xs, [r["A129"] for r in trace], "s-", ms=5, lw=1.4,
            color="tab:purple", label="e177 A(129) brake")
    axr = ax.twinx()
    axr.plot(xe, e161["ce_r"], "--", lw=0.9, color="k", alpha=0.3)
    axr.plot(xe, e176["ce_r"], "--", lw=0.9, color="k", alpha=0.3)
    axr.plot(xs, [r["ce_r"] for r in trace], "k:o", ms=5, lw=1.2,
             alpha=0.85, label="e177 CE_R (wreckage guard)")
    ax.set_xlabel("plain-corpus freeze steps")
    ax.set_ylabel("home p(Z) / strength")
    axr.set_ylabel("CE_R")
    ax.axhline(0, color="k", lw=0.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.0, loc="center right")
    ax.set_title("the home reads + anatomy co-dials (already dead at this "
                 "endpoint) vs e161/e176's home collapse", fontsize=10)

    # (1,1) verdict panel
    ax = axes[1, 1]
    ax.axis("off")
    onset_txt = "  ".join(f"+{r['freeze_steps']}:{r['site_read_onset']:.4f}"
                          for r in trace)
    span_txt = "  ".join(f"+{r['freeze_steps']}:{r['site_read_span']:.4f}"
                         for r in trace)
    wr = cond["wash_rate"]
    vlines = [
        "REGISTERED (QUEUE e177 intent / dispatch verbatim; frozen "
        "operationalizations):",
        f"  TRUE-ARCHIVE: site onset(300) >= {SURVIVE_BAR}",
        f"  SCRATCH-MEMORY: site onset(300) <= {SHUT_BAR} (e158/e161 SHUT "
        f"bar; shape CO-REPORTED)",
        "  texture (partial: slow decay vs the two-step collapse) => "
        "TEXTURE with the decay curve",
        "",
        "TRACE (freeze steps from the SITE-ENDPOINT, e151_twodoor):",
        f"  onset:  {onset_txt}",
        f"  span:   {span_txt}",
        f"  home:   " + "  ".join(
            f"+{r['freeze_steps']}:{r['base_gm12']:.3f}/{r['base_g0']:.3f}"
            for r in trace) + "  (g-12/g0)",
        f"  census: " + " ".join(
            f"{r['site_span_strength']:+.3f}/{r['site_onset_strength']:+.3f}"
            for r in trace) + "  (span/onset arm)",
        f"  D-183:  " + " ".join(f"{r['d183_site_onset']:.3f}" for r in trace)
            + "  (site onset under the row-183 knife)",
        f"  CE_R:   " + " ".join(f"{r['ce_r']:.3f}" for r in trace),
        "",
        f"  TRUE-ARCHIVE fires={cond['TRUE_ARCHIVE']['fires']} "
        f"(onset {cond['TRUE_ARCHIVE']['onset_final']:.4f} "
        f"[{cond['TRUE_ARCHIVE']['onset_clause']}], span "
        f"{cond['TRUE_ARCHIVE']['span_final']:.4f})",
        f"  SCRATCH-MEMORY fires={cond['SCRATCH_MEMORY']['fires']} "
        f"(earliest<=bar @{cond['SCRATCH_MEMORY']['earliest_ck_onset_le_bar']}; "
        f"under-bar@first-ck "
        f"{cond['SCRATCH_MEMORY']['shape_match']['under_bar_at_first_ck']})",
        f"  gap band (=> TEXTURE): {cond['GAP_BAND']['onset_in_gap']}",
        "",
        "WASH RATE (onset retention at the first checkpoint; same seed,",
        "same draw sequence — the trajectories differ only in their nets):",
        f"  e161 dwell {wr['e161_onset_retention_at_first_ck']:.3f} | e176 "
        f"consolidated {wr['e176_onset_retention_at_first_ck']:.3f} | e177 "
        f"site {wr['e177_onset_retention_at_first_ck']:.3f}",
        f"  earliest ck onset <= 0.27: e161 @{wr['e161_earliest_ck_onset_le_shut']}, "
        f"e176 @{wr['e176_earliest_ck_onset_le_shut']}, e177 "
        f"@{wr['e177_earliest_ck_onset_le_shut']}",
        "",
        f"G_ROOT site-endpoint gate: max|diff| {g_root['max_abs_diff']:.2e} "
        f"({'PASS' if g_root['pass'] else 'FAIL'})",
        f"overlay refs: e161 {e161_src.get('source', 'embedded copy')}",
        f"             e176 {e176_src.get('source', 'embedded copy')}",
        "",
        f"VERDICT: {verdict}",
    ] + [f"  {wd}" for wd in
         [clause[i:i + 78] for i in range(0, len(clause), 78)]]
    for i, tx in enumerate(vlines):
        ax.text(0.02, 0.97 - i * 0.030, tx, fontsize=7.1, va="top",
                family="monospace")

    fig.suptitle(f"E177 — WASH THE SITE-ENDPOINT (e151_twodoor, the buried "
                 f"deep site-store; +{CKPT_STEPS[-1]} plain-corpus steps, "
                 f"same protocol/seed as e161/e176) -> {verdict}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
