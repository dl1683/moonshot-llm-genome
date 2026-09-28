"""E183 — THE FILTERED STREAM (QUEUE.md row e183; T109's named residue — the
last gate on the session's biggest finding's unbounded form).

WHY (T109's residue): e176N proved the consolidated fact dissolves under the
truly NEUTRAL stream (0/16 junction anchors) on the same two-step clock —
BUT both dissolving streams (e176's extinction anchors AND e176N's neutral
anchors) share a random-corpus half that is UNFILTERED in all prior cells:
16 random train_ids windows per step, ~3.84%/window of which carry a
host-junction (e176N's measurement: 150 host occurrences x 257-token span /
1.0M train chars = 0.0384/window; ~4800 draws over 300 steps ~ a non-
trivial cumulative junction exposure through the random half alone). THE
QUESTION THIS CELL DECIDES: does the fact STILL dissolve when that
background is FILTERED OUT — the truly clean stream — or was the 3.84%
background the driver all along?

REGISTERED BARS (QUEUE e183 verbatim, frozen here before compute; no bar
shopping — adjudicate against exactly this):
  - STILL-DISSOLVES fires if: g-12 <= 0.27 by +50 — the activity-dependence
    noun goes UNBOUNDED (no memory survives continued training without
    fact-bearing windows, on any stream composition tested).
  - BACKGROUND-CARRIED fires if: survival (g-12 >= 0.5 through +300) — the
    3.84% background was the driver; the noun stays bounded.
  - No bar shopping; texture (slower wash) => TEXTURE with the curve.

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * g-12 / g0 / g+12 / held30 = ABSOLUTE install-60 / held-30 battery
    mean p(Z) at ctx offsets -12 / 0 / +12 (e176/e176N's convention
    verbatim — the same batteries, the same corpus rebuild, the same ruler).
  * STILL-DISSOLVES = the filtered arm's g-12 at the +50 checkpoint
    <= 0.27 (the dispatch's "by +50"); the earliest checkpoint <= 0.27
    (fine steps {1,2,4} included, measured in the MAIN run) is CO-REPORTED
    as the collapse timing.
  * BACKGROUND-CARRIED = the filtered arm's g-12 >= 0.50 at EVERY
    continuation checkpoint {1,2,4,50,100,200,300} (the dispatch's
    "through +300"; step 0 = the root, 0.9156, trivially above).
  * The two clauses are disjoint (surviving requires +50 >= 0.50 > 0.27);
    order STILL-DISSOLVES -> BACKGROUND-CARRIED -> TEXTURE; every
    sub-boolean reported regardless.
  * CE-AT-DISSOLUTION (the R50 splice discipline, e176N's convention): per
    stream, the CE_R (and the in-batch corpus CE) at the FIRST checkpoint
    under the 0.27 bar is reported as its own record — recovered CE values
    alone do not characterize the wash. Co-reported for all three streams
    (e176 original / e176N neutral / e183 filtered).

THE ONE PROTOCOL CHANGE (e176N arm A VERBATIM otherwise): the random-corpus
half's windows are FILTERED at draw time. SPEC:
  * A candidate start s (drawn exactly as e161/e176/e176N draw it:
    torch.randint(len(train_ids) - BLOCK - 1, ...) on the SAME generator,
    seed 10902) is REJECTED and redrawn (single draws from the same
    generator, same modulus) if EITHER
      (a) its span text train_text[s : s+257) — the window PLUS its first
          prediction target, e170's span convention — CONTAINS a host-name
          token: the substring "FLORIZEL" or "ELIZABETH" (grep-verified at
          draw time; the e161 ZEPH-check convention extended to the host
          class), OR
      (b) it COVERS a host-name junction: some host occurrence onset p
          satisfies s <= p < s+257 (e170's junction accounting; the
          e176N-measured 3.84%/window class — (b) is the wider clause and
          (a) u (b) = (b) up to right-edge name cutoffs, which (b) also
          catches).
  * Expected rejection rate = the measured background itself: 150 host
    occurrences x 257 / len(train_ids) ~ 3.84% of candidates (~184 of
    ~4816 main-run draws); the REALIZED rate, the per-category counts, and
    the redraw tally are recorded in metrics.json filter_record and gated
    (G_FILTER: every accepted window passes an independent post-hoc grep +
    onset check; zero host-substring leaks allowed).
  * What the filter does NOT remove (recorded, not hidden): left-edge
    straddles — a host name whose onset falls BEFORE s but whose tail
    enters the window (not in the 3.84% onset-in-span class; carries no
    pre-host context, hence no context->onset junction signal) — are
    COUNTED in the record as the accepted residue; and the stream remains
    full-token CE plain-corpus training at lr 1e-3: ordinary corpus
    pressure (the disuse mechanism itself) is the thing under test and
    stays by design.
  * RNG consequence (documented deviation): rejection redraws consume
    extra generator draws, so after the FIRST rejection the draw sequence
    necessarily diverges from e176N arm A's (same seed, same modulus; only
    the number of draws differs). The anchor half (e170's neutral bank,
    seed 170) is rebuilt bit-identically and GATED against e176N's stored
    starts.

PROVENANCE: load_cpu/evl_load/battery_cell/battery_pz/ce_fixed_cpu/
val_windows/deleted_wpe/read_fact_at/row_census are lab/e176n_neutral_wash.py
VERBATIM (= lab/e176_freeze_root.py; the e161/e152/e151/e143/e131/e119/
e113/e068/e065/e043 lineage); finetune_freeze_filtered is e176N's
finetune_freeze (= e176's) with ONLY the filter inserted into the random
half (per-step arithmetic otherwise identical); the neutral anchor bank +
junction accounting + G_ANCHOR are lab/e170_anchor_neutral.py VERBATIM;
the measure dial is e176N's measure() (e176's minus the 183-span census —
e178's recorded deviation, inherited). The three reference trajectories
(e176 main, e176 smoke, e176N arm A) are embedded verbatim and re-verified
against their stored metrics files at plot time. Copied, not imported, to
own the device policy.

NETS: root runs/checkpoints/e131_consolidated_e113.pt (gated bit-exact vs
e151's stored before-cells, e176N's G_ROOT set). New checkpoints:
runs/checkpoints/e183_filtered{,_sN}.pt (smoke_ prefixed in smoke mode).

COMPUTE ENVELOPE: CPU-ONLY by dispatch (CUDA_VISIBLE_DEVICES=-1 forced
before torch; threads 4, LOW <= 4, the machine otherwise quiet), one 25 s
launch stagger (single sleep, no busy-waiting anywhere), cooldown 60 s
before and after the ONE 300-step training, per-training CPU cap 1800 s
(e176N precedent: ~830 s actual for 300 steps), plus the 4-step fine-step
smoke shakedown (E183_SMOKE=1, no cooldowns, nothing adjudicated). No
single wait exceeds 180 s.

Outputs: runs/e183/{metrics.json, filtered_stream.png} (runs/e183_smoke/
for the shakedown); checkpoints runs/checkpoints/e183_filtered*.pt. No
NOTES/THINKING/QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python e183_filtered_stream.py    (E183_SMOKE=1 shakedown)

RUN RECORD (added POST-COMPUTE, 2026-09-28; the registration above was
frozen before any training): realized rejection rate 3.26% (162 rejections /
4962 candidates; expected 3.84%; 153 substring+junction, 9 junction-only,
0 substring-only; 162 redraws, max 4 on one slot; 4800 accepted windows, 7
left-edge straddle residue, 0 post-hoc grep leaks). VERDICT: STILL-DISSOLVES
— g-12 0.9156 -> 0.8033 (+1) -> 0.0398 (+2) -> 0.0147 (+4) -> 0.0042 (+50)
-> 0.0187 (+100) -> 0.0003 (+200) -> 0.0002 (+300); g-12 at +50 = 0.0042
<= 0.27, earliest checkpoint under the bar +2 (the same two-step clock as
the neutral stream: 0.6780/0.0271/0.0109); CE-at-dissolution +2: CE_R
1.9949, in-batch CE 1.8234 (neutral's +2: CE_R 2.0321, in-batch 1.8263).
Full record: runs/e183/metrics.json.
"""
from __future__ import annotations

import bisect
import copy
import json
import random
import sys
import time
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY by dispatch (threads
# capped at 4 below; the machine is otherwise quiet)

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

SMOKE = os.environ.get("E183_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e183 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
ROOT_CK = "e131_consolidated_e113.pt"   # THE FULLY-CONSOLIDATED ROOT
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E176_METRICS = E43.REPO / "runs" / "e176" / "metrics.json"
E176_SMOKE_METRICS = E43.REPO / "runs" / "e176_smoke" / "metrics.json"
E176N_METRICS = E43.REPO / "runs" / "e176n" / "metrics.json"

# ---- e152's placement constants (the measurement instruments rebuild these) ----
RETEACH_J = 54                    # offset: name x-cols 184..190, read rows 183..189
SITE_ADDR_ROW = 183               # the onset read row (= e120's SPLICE_ADDR_ROW)
SITE_Z_XCOL = PRE + RETEACH_J     # 184 (= e120's Z_XCOL)
SITE_CONT = BLOCK - PRE - RETEACH_J - len(NAME)     # 65-token continuation
D_ALL = (121, 125, 129, 133, 137)                   # e113's fixed set
GEOS = (-12, 0, 12)               # novel x2 + the trained g0

# ---- census row set (e158/e161/e176/e176N old-band convention) ----------------
ROWS_OLD = (0, 1, 2, 3, 4, 5, 6, 60, 100, 118, 119, 120) + tuple(range(121, 130))
if SMOKE:
    ROWS_OLD = (0, 1, 60, 121, 125, 129)

# ---- the arm's freeze schedule -------------------------------------------------
CK_F: tuple[int, ...] = (1, 2, 4, 50, 100, 200, 300) if not SMOKE else (1, 2, 4)

# ---- fine-tune envelope (e109 arm-b / e119-L / e143 / e151 / e152 / e161/e176/
# e176N arm A verbatim) ----------------------------------------------------------
LR_F = 1e-3                        # e176N arm A verbatim
FT_TIME_CAP = 1800.0               # CPU cap per training (e176N precedent)
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
FREEZE_SEED = 10902               # the locked seed lineage (= e161/e176/e176N)
COOLDOWN_S = 60.0                  # around the ONE training
STAGGER_S = 25.0                   # launch stagger vs the CPU fleet

# ---- e170's neutral anchor bank (rebuilt bit-identically; gated vs e176N) ------
E170_ANCHOR_SEED = 170             # dedicated RNG for the plain-corpus bank
ANCHOR_FORBIDDEN = ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")
# e176N's stored neutral-bank starts (runs/e176n/metrics.json
# gates.G_ANCHOR.neutral_bank.starts, VERBATIM) — the anchor half of this run
# must reproduce them EXACTLY (same seed, same corpus rebuild).
E176N_NEUTRAL_STARTS = [825650, 361746, 954106, 856844, 615070, 335787, 329879,
                        96582, 253994, 341757, 608690, 127521, 228843, 834931,
                        15774, 408603]

# ---- gates / references (full precision, = stored metrics) --------------------
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05

E151_ROOT = {                     # runs/e151 'before' battery (e176N's gate set)
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

# e176's stored MAIN trajectory (the ORIGINAL extinction-confounded stream,
# lr 1e-3): runs/e176/metrics.json trace_summary, VERBATIM, re-verified at
# plot time.
E176_MAIN = {
    "freeze_steps": [0, 50, 100, 200, 300],
    "base_gm12": [0.9155886173248291, 0.020882638171315193,
                  0.006246014963835478, 0.0006464376347139478,
                  0.0011210207594558597],
    "base_g0": [0.7850371599197388, 0.027392588555812836,
                0.016695411875844002, 0.001976940780878067,
                0.0012388104805722833],
    "ce_r": [1.663516640663147, 1.6478883028030396,
             1.6605137586593628, 1.6117957830429077,
             1.6334049701690674],
}

# e176's SMOKE fine steps (the extinction stream's ONLY fine-step trajectory):
# runs/e176_smoke/metrics.json trace_summary VERBATIM.
E176_SMOKE = {
    "freeze_steps": [0, 2, 4],
    "base_gm12": [0.9155886173248291, 0.08816977590322495,
                  0.002640149090439081],
    "base_g0": [0.7850371599197388, 0.08683783560991287,
                0.0034518486354500055],
    "ce_r": [1.663516640663147, 2.0001156330108643, 1.7823567390441895],
}

# e176N arm A's stored trajectory (the NEUTRAL stream, lr 1e-3): runs/
# e176n/metrics.json trace_summary armA_* arrays, VERBATIM, re-verified at
# plot time. Fine steps {1,2,4} measured in its MAIN run.
E176N_ARM_A = {
    "armA_steps": [0, 1, 2, 4, 50, 100, 200, 300],
    "armA_gm12": [0.9155886173248291, 0.6780440807342529,
                  0.027077054604887962, 0.010940761305391788,
                  0.022122304886579514, 0.014922752045094967,
                  0.00204725144430995, 0.0038193254731595516],
    "armA_g0": [0.7850371599197388, 0.4619811177253723,
                0.1147073358297348, 0.06042749062151019,
                0.16835635900497437, 0.05249874293804169,
                0.011475668288767338, 0.016238771378993988],
    "armA_held30_gm12": [0.6361417174339294, 0.4820752739906311,
                         0.0053826989606022835, 0.001151302014477551,
                         0.008003094233572483, 0.008265051619424877,
                         0.0012307859724387527, 0.003200115170329809],
    "armA_ce_r": [1.663516640663147, 2.2113420963287354,
                  2.032074451446533, 1.8213403224945068,
                  1.708834171295166, 1.6700899600982666,
                  1.6468615531921387, 1.642844796180725],
}
# e176N arm A's in-run light trajectory (for its CE-at-dissolution record):
# runs/e176n/metrics.json armA_neutral.traj VERBATIM (step, corpus_ce).
E176N_TRAJ_CE = {1: 1.356567621231079, 2: 1.8263245820999146,
                 4: 1.3631858825683594, 50: 0.6575877070426941,
                 100: 0.660504937171936, 200: 0.6505676507949829,
                 300: 0.6238144636154175}

# ---- registered bar constants (frozen; see OPERATIONALIZATIONS above) --------
SHUT_BAR = 0.27                   # STILL-DISSOLVES by +50 (e158/e161/e176/e176N)
SURVIVE_BAR = 0.50                # BACKGROUND-CARRIED through +300
ROOT_GM12 = E151_ROOT["base_gm12"]
ROOT_G0 = E151_ROOT["base_g0"]

REGISTERED_PREDICTION = {
    "still_dissolves": "STILL-DISSOLVES fires if: g-12 <= 0.27 by +50 — the "
        "activity-dependence noun goes UNBOUNDED (no memory survives continued "
        "training without fact-bearing windows, on any stream composition "
        "tested).",
    "background_carried": "BACKGROUND-CARRIED fires if: survival (g-12 >= 0.5 "
        "through +300) — the 3.84% background was the driver; the noun stays "
        "bounded.",
    "no_bar_shopping": "No bar shopping; texture (slower wash) => TEXTURE "
        "with the curve.",
    "operationalizations": "g-12/g0/g+12/held30 = absolute install-60/held-30 "
        "battery mean p(Z) at ctx offsets -12/0/+12 (e176/e176N's "
        "convention); STILL-DISSOLVES = filtered arm g-12 at +50 <= 0.27 "
        "(earliest checkpoint <= 0.27 over {1,2,4,50,100,200,300} "
        "CO-REPORTED); BACKGROUND-CARRIED = filtered arm g-12 >= 0.50 at "
        "EVERY continuation checkpoint {1,2,4,50,100,200,300}; clauses "
        "disjoint; order STILL-DISSOLVES -> BACKGROUND-CARRIED -> TEXTURE; "
        "CE-at-dissolution reported per stream (CE_R + in-batch corpus CE at "
        "the first checkpoint under 0.27) for all three streams.",
    "registration": "QUEUE.md row e183 (T109's named residue, the dispatch's "
        "registration); frozen verbatim in this docstring before compute. "
        "Adjudicate against exactly this; no bar shopping.",
}

trims: list[str] = []
deviations: list[str] = [
    "CPU-ONLY by dispatch (CUDA_VISIBLE_DEVICES=-1 forced before torch; the "
    "machine otherwise quiet): torch threads 4, one 25 s launch stagger "
    "(single sleep, no busy-waiting), cooldown 60 s before/after the ONE "
    "300-step training, per-training CPU cap 1800 s (e176N precedent: ~830 s "
    "actual / 300 steps). The 4-step smoke shakedown runs without cooldowns "
    "and adjudicates nothing.",
    "Nets are the mandated e131_consolidated line (2.7M; e170/e176N's "
    "precedent note): the dispatch's '<=1M family' phrase is an envelope "
    "statement — every gate reference and lineage number of this cell lives "
    "on the 2.7M line, and the mandated root IS e131_consolidated_e113.pt.",
    "THE FILTER (the only protocol change vs e176N arm A): the 16-per-step "
    "random-corpus windows are rejection-sampled at draw time — reject any "
    "candidate whose [s, s+257) span contains FLORIZEL/ELIZABETH as a "
    "substring (grep, the e161 convention extended to the host class) OR "
    "covers a host junction (onset p with s <= p < s+257; the e176N-measured "
    "3.84%/window class); redraw singles from the same torch generator until "
    "pass. Spec + realized rejection rate in the docstring RUN RECORD and in "
    "metrics.json filter_record (G_FILTER gates zero leaks).",
    "RNG DIVERGENCE (necessary consequence of the filter): rejection "
    "redraws consume extra generator draws, so after the first rejection the "
    "draw sequence diverges from e176N arm A's (same seed 10902, same "
    "modulus, same per-step draw structure; only the redraw count differs). "
    "The anchor half is unaffected content-wise: e170's neutral bank is "
    "rebuilt bit-identically and gated vs e176N's stored starts (G_ANCHOR).",
    "LEFT-EDGE RESIDUE (recorded, not filtered): host names whose onset "
    "falls before s but whose tail enters the window are NOT in the 3.84% "
    "onset-in-span class and are accepted; they carry no pre-host context "
    "(no context->onset junction signal). COUNTED in filter_record as "
    "accepted_left_edge_straddle.",
    "Fine steps {1,2,4} are measured in the MAIN run (e176N's recorded "
    "convention — e176's two-step claim rested on its smoke); the smoke run "
    "is a machinery shakedown only.",
    "The measure dial is e176N's measure() (= e176's minus the 183-span "
    "census, e178's recorded deviation, inherited): the functional site read "
    "@183 is kept as the instrument co-report.",
    "Eval thread count is 4 (dispatch) vs e151's stored cells — CPU "
    "reduction order can drift low-order bits; the G_ROOT gate reports both "
    "the 5e-6 bit flag and the 0.05 fallback tolerance (e161/e176/e176N "
    "precedent).",
    "Single seed (10902), one trajectory, one root lineage, n=1 per cell — "
    "point estimates until replicated (e152R showed seed-to-seed timing "
    "spans an order of magnitude on related washes).",
    "Smoke mode trims: 4 steps, lean measures (no censuses, no deletion "
    "table), filter machinery live (its tallies sanity-check the rejection "
    "rate); nothing adjudicated.",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/e176n_neutral_wash.py VERBATIM (= lab/e176_freeze_root.py;
# see the module docstring). Copied rather than imported to own the device
# policy.

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
    e176N copy): p(true name char) at positions addr_row..addr_row+6."""
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


# ------------------------------------------------------------------ the filter

class HostJunctionFilter:
    """THE FILTER (this cell's only protocol change): reject any random-
    corpus window start whose [s, s+257) span (window + first target, e170's
    span convention) contains a host-name token (substring grep — the e161
    convention extended to the host class) or covers a host-name junction
    (some host onset p with s <= p < s+257 — the e176N-measured 3.84%/window
    class). Tally kept for the filter record; left-edge straddles among the
    ACCEPTED windows are counted as the recorded residue."""

    def __init__(self, train_text: str, hosts: list[str]):
        self.text = train_text
        self.hosts = list(hosts)
        self.occ = sorted(
            (p, len(h)) for h in hosts for p in E43.find_occ(train_text, h))
        self.onsets = sorted(p for p, _ in self.occ)
        self.tally = {
            "candidates_tested": 0,
            "rejected_substring_only": 0,
            "rejected_junction_only": 0,
            "rejected_substring_and_junction": 0,
            "redraws": 0,
            "accepted": 0,
            "accepted_left_edge_straddle": 0,
            "max_redraws_one_slot": 0,
        }

    def _junction(self, s: int) -> bool:
        # any host onset p with s <= p < s + BLOCK + 1
        lo = bisect.bisect_left(self.onsets, s)
        hi = bisect.bisect_right(self.onsets, s + BLOCK)
        return hi > lo

    def _left_straddle(self, s: int) -> bool:
        # host onset before s whose tail enters the window (residue class)
        return any(p < s < p + L for p, L in self.occ)

    def test(self, s: int) -> bool:
        """Test one candidate; tally; return True if ACCEPTED."""
        self.tally["candidates_tested"] += 1
        span = self.text[s: s + BLOCK + 1]
        sub = any(h in span for h in self.hosts)
        junc = self._junction(s)
        if sub or junc:
            if sub and junc:
                self.tally["rejected_substring_and_junction"] += 1
            elif sub:
                self.tally["rejected_substring_only"] += 1
            else:
                self.tally["rejected_junction_only"] += 1
            return False
        self.tally["accepted"] += 1
        if self._left_straddle(s):
            self.tally["accepted_left_edge_straddle"] += 1
        return True

    @property
    def rejected(self) -> int:
        t = self.tally
        return (t["rejected_substring_only"] + t["rejected_junction_only"]
                + t["rejected_substring_and_junction"])

    def record(self, bg_rate: float) -> dict:
        t = self.tally
        n = t["candidates_tested"]
        return {
            "spec": ("reject any random-window start s whose span "
                     "[s, s+257) contains FLORIZEL/ELIZABETH as a substring "
                     "(grep at draw time — the e161 convention extended to "
                     "the host class) OR covers a host junction (onset p "
                     "with s <= p < s+257 — e170's convention; the "
                     "e176N-measured 3.84%/window class); redraw singles "
                     "from the same torch generator (seed 10902, same "
                     "modulus) until pass"),
            **{k: v for k, v in t.items()},
            "rejections_total": self.rejected,
            "realized_rejection_rate": self.rejected / max(n, 1),
            "expected_rejection_rate": bg_rate,
            "expected_class": "the e176N measurement: 150 host occurrences "
                              "x 257-token span / len(train_ids) = "
                              "3.84%/window",
            "residue_note": ("left-edge straddles (host onset before s, "
                             "tail inside) are NOT in the 3.84% onset-in-"
                             "span class, are accepted, and are counted in "
                             "accepted_left_edge_straddle — they carry no "
                             "pre-host context, hence no context->onset "
                             "junction signal"),
        }


# ------------------------------------------------------------------ fine-tune

def finetune_freeze_filtered(tag: str, net0: TinyGPT, anchor: torch.Tensor,
                             train_ids: torch.Tensor, itos, train_text: str,
                             filt: HostJunctionFilter, r_eval_xy,
                             gm12_ids, g0_ids, zid: int, seed: int,
                             lr: float, ckpt_steps: tuple[int, ...]):
    """THE FILTERED PLAIN-CORPUS FREEZE (e176N arm A's finetune_freeze
    VERBATIM arithmetic with THE FILTER inserted into the random half). Per
    step: aj = randint(16) anchor draws, rj = randint(16) random offsets;
    each random start is FILTER-TESTED at draw time and REDRAWN (single
    draws, same generator/modulus) until it passes; batch 32 full-token CE;
    AdamW (0.9,0.95) wd 0.1 constant lr, clip 1.0. Snapshots (deep-copy out)
    + light CPU evals (g-12, g0, CE_R — no RNG consumed) at the checkpoint
    steps; the in-batch corpus CE is recorded at every checkpoint and every
    50. The accepted windows are additionally verified by an independent
    post-hoc grep (the e161 hard-fail convention, host class)."""
    ckpt_set = set(ckpt_steps)
    n_steps = ckpt_steps[-1]
    net = copy.deepcopy(net0)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    n_anc = anchor.shape[0]
    hi = len(train_ids) - BLOCK - 1
    traj, sds, t_start = [], {}, time.time()
    step = 0
    zeph_checks = 0
    host_leaks = 0
    evl = copy.deepcopy(net0)          # CPU eval twin
    for step in range(1, n_steps + 1):
        aj = torch.randint(n_anc, (ANCH_BS,), generator=gen)
        rj = torch.randint(hi, (RAND_BS,), generator=gen)
        starts, r_redraws = [], 0
        for v in rj.tolist():
            s = int(v)
            while not filt.test(s):
                s = int(torch.randint(hi, (1,), generator=gen))
                r_redraws += 1
            starts.append(s)
        filt.tally["redraws"] += r_redraws
        filt.tally["max_redraws_one_slot"] = max(
            filt.tally["max_redraws_one_slot"], r_redraws)
        anc = anchor[aj]
        rnd = torch.stack([train_ids[s: s + BLOCK] for s in starts])
        # name-free + host-free VERIFY (no-op by construction; hard-fail if
        # not — the e161 convention extended to the host class)
        for s in starts:
            txt = "".join(itos[int(c)] for c in train_ids[s: s + BLOCK + 1])
            if "ZEPH" in txt:
                zeph_checks += 1
            if any(h in txt for h in HOSTS):
                host_leaks += 1
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
        if (time.time() - t_start) > FT_TIME_CAP:
            log(f"  [{tag}] time cap {FT_TIME_CAP:.0f}s at s{step}")
            trims.append(f"{tag}: time cap at step {step}")
            break
    net.eval()
    return {"sds": sds, "traj": traj, "steps_ran": step, "seed": seed,
            "lr": lr, "zeph_violations": zeph_checks,
            "host_leaks": host_leaks}


CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "e183", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name} "
        f"({', '.join(f'{k}={v}' for k, v in meta.items())})")


def verify_ref(embedded: dict, path: Path, ts_keys: list, src_name: str) -> dict:
    """Verify an embedded reference copy against its stored metrics file when
    present (no silent divergence; e176N's verify_ref convention)."""
    src = {"source": f"embedded verbatim copy ({src_name})",
           "file_present": path.exists(),
           "verified_vs_embedded": None,
           "max_abs_diff": None}
    if path.exists():
        mm = json.loads(path.read_text(encoding="utf-8"))
        ts = mm["trace_summary"] if "trace_summary" in mm else mm
        diffs = [abs(a - b) for k in ts_keys
                 for a, b in zip(ts[k], embedded[k])]
        steps_ok = list(ts[ts_keys[0]]) == embedded[ts_keys[0]]
        src["max_abs_diff"] = max(diffs) if diffs else None
        src["verified_vs_embedded"] = bool(steps_ok and max(diffs) < 1e-9)
        if src["verified_vs_embedded"]:
            src["source"] = f"{src_name} (embedded copy verified, max|diff| " \
                            f"{max(diffs):.1e})"
    return src


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e183_smoke" if SMOKE else "e183")
    log(f"E183 THE FILTERED STREAM (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()}), stagger "
        f"{STAGGER_S:.0f}s, cooldown {COOLDOWN_S:.0f}s around the ONE training")
    time.sleep(STAGGER_S)            # launch stagger vs the CPU fleet

    # ---------------- protocol rebuild (e143/e151/e152/e161/e176/e176N verbatim)
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
    # THE ANCHOR HALF (e170's neutral bank, rebuilt bit-identically to
    # e176N arm A's — the ONLY half this run does NOT change): 16 plain-
    # corpus windows, rejection on host/nonce content in [s, s+257).
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

    def junctions_covered(starts):
        cov = 0
        for s in starts:
            if any(s <= p < s + BLOCK + 1 for p in host_positions):
                cov += 1
        return cov

    jc_neutral = junctions_covered(n_starts)
    host_occ_total = len(host_positions)
    bg_rate = host_occ_total * (BLOCK + 1) / len(train_ids)

    # THE FILTER (the delta): instantiated on the same corpus text
    filt = HostJunctionFilter(train_text, HOSTS)

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
            "bit_identical_to_e176n": bool(n_starts == E176N_NEUTRAL_STARTS),
        },
        "random_channel_background": {
            "host_occurrences_in_train": host_occ_total,
            "est_window_hit_rate": bg_rate,
            "note": ("the 16-per-batch random corpus windows carried this "
                     "~3.84%/window host-junction background UNFILTERED in "
                     "e176 and e176N — THIS cell filters it (the delta); "
                     "~4800 draws x 3.84% ~ 184 expected rejections over the "
                     "300-step main run"),
        },
    }
    G_ANCHOR["pass"] = bool(
        G_ANCHOR["neutral_bank"]["windows_with_host_content"] == 0
        and G_ANCHOR["neutral_bank"]["junctions_covered"] == 0
        and G_ANCHOR["neutral_bank"]["bit_identical_to_e176n"]
        and anchor_neutral.shape == (16, BLOCK))
    assert G_ANCHOR["pass"], f"anchor gate FAILED: {G_ANCHOR}"
    log(f"G_ANCHOR: neutral bank 16x{BLOCK} rebuilt bit-identical to e176N's "
        f"(seed {E170_ANCHOR_SEED}, {rejections} rejections/{tries} tries) — "
        f"host content 0/16, junctions 0/16; random-channel background "
        f"{100 * bg_rate:.2f}%/window NOW FILTERED: PASS")

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
        """e176N's measure() VERBATIM (e176's minus the 183-span census):
        base 3-geos + held30 + CE_R + site read + old-band census (row-0
        sink / A129 brake) + deletion table (D-all, D-183). lean=True (the
        smoke convention) drops the census + deletions, keeps a quick
        A(129)."""
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

    # reference provenance (embedded copies verified vs the stored files)
    TS_KEYS = ["freeze_steps", "base_gm12", "base_g0", "ce_r"]
    src_e176 = verify_ref(E176_MAIN, E176_METRICS,
                          ["freeze_steps", "base_gm12", "base_g0", "ce_r"],
                          "runs/e176/metrics.json trace_summary")
    src_e176s = verify_ref(E176_SMOKE, E176_SMOKE_METRICS,
                           ["freeze_steps", "base_gm12", "base_g0", "ce_r"],
                           "runs/e176_smoke/metrics.json trace_summary")
    src_e176n = verify_ref(E176N_ARM_A, E176N_METRICS,
                           ["armA_steps", "armA_gm12", "armA_g0",
                            "armA_held30_gm12", "armA_ce_r"],
                           "runs/e176n/metrics.json trace_summary armA_*")
    log(f"references: e176 main {src_e176['verified_vs_embedded']}, "
        f"e176 smoke {src_e176s['verified_vs_embedded']}, "
        f"e176N armA {src_e176n['verified_vs_embedded']}")

    # =====================================================================
    # THE FILTERED WASH (the critical cell; ONE training; cooldown)
    # =====================================================================
    log("=" * 78)
    if not SMOKE:
        log(f"[thermal] cooldown {COOLDOWN_S:.0f}s before the training")
        cooldown(COOLDOWN_S)
    log(f"THE FILTERED WASH: {CK_F[-1]}-step plain-corpus freeze with e170's "
        f"NEUTRAL anchors (bit-identical to e176N arm A's bank) + the random "
        f"half FILTERED of host-junction windows (batch {ANCH_BS} neutral + "
        f"{RAND_BS} filtered-random, full-token CE, lr {LR_F}, seed "
        f"{FREEZE_SEED}), checkpoints +{list(CK_F)}")
    armF = finetune_freeze_filtered("filtered", net0, anchor_neutral, train_ids,
                                    itos, train_text, filt, r_eval_xy,
                                    gm12_ids, g0_ids, zid, FREEZE_SEED,
                                    LR_F, CK_F)
    if not SMOKE:
        log(f"[thermal] cooldown {COOLDOWN_S:.0f}s after the training")
        cooldown(COOLDOWN_S)
    G_DRAWFREE = {"zeph_violations": armF["zeph_violations"],
                  "host_leaks": armF["host_leaks"],
                  "pass": bool(armF["zeph_violations"] == 0
                               and armF["host_leaks"] == 0)}
    assert G_DRAWFREE["pass"], ("name/host token leaked into an accepted "
                                "window")
    filter_record = filt.record(bg_rate)
    # rate tolerance: 3-sigma binomial bound around the measured class rate,
    # floored at 0.02 (the smoke's ~64 candidates need the wider bound; the
    # main run's ~4816 pin it near the floor)
    n_cand = filter_record["candidates_tested"]
    rate_tol = max(0.02, 3.0 * (bg_rate * (1.0 - bg_rate) / max(n_cand, 1))
                   ** 0.5)
    G_FILTER = {
        "candidates_tested": filter_record["candidates_tested"],
        "rejections_total": filter_record["rejections_total"],
        "realized_rejection_rate": filter_record["realized_rejection_rate"],
        "expected_rejection_rate": bg_rate,
        "rate_tol_3sigma": rate_tol,
        "post_hoc_grep_leaks": armF["host_leaks"],
        "accepted_windows": filter_record["accepted"],
        "accepted_left_edge_straddle": filter_record[
            "accepted_left_edge_straddle"],
        "pass": bool(armF["host_leaks"] == 0
                     and abs(filter_record["realized_rejection_rate"]
                             - bg_rate) < rate_tol),
    }
    log(f"G_FILTER: {filter_record['rejections_total']} rejections / "
        f"{filter_record['candidates_tested']} candidates = "
        f"{100 * filter_record['realized_rejection_rate']:.2f}% "
        f"(expected {100 * bg_rate:.2f}%); left-edge residue "
        f"{filter_record['accepted_left_edge_straddle']} accepted windows; "
        f"post-hoc grep leaks {armF['host_leaks']}: "
        + ("PASS" if G_FILTER["pass"] else "FAIL"))
    assert G_FILTER["pass"], f"filter gate FAILED: {G_FILTER}"
    log("gates: G_SPLICE, G_NAMEFREE, G_POOL, G_ANCHOR, G_ROOT, G_DRAWFREE, "
        "G_FILTER all PASS")

    save_ckpt("e183_filtered", armF["sds"][max(armF["sds"])],
              {"desc": f"e131_consolidated_e113 + {max(armF['sds'])}-step "
                       f"FILTERED plain-corpus freeze (e170's neutral anchor "
                       f"bank bit-identical to e176N arm A + the random half "
                       f"rejection-filtered of host-junction windows: no "
                       f"FLORIZEL/ELIZABETH substring, no host onset in "
                       f"[s, s+257)), full-token CE), lr {LR_F}, seed "
                       f"{FREEZE_SEED}",
               "steps": int(max(armF["sds"])), "seed": FREEZE_SEED,
               "lr": LR_F,
               "anchor_bank": f"neutral (seed {E170_ANCHOR_SEED}, "
                              f"bit-identical to e176N arm A)",
               "random_half": "FILTERED (host-junction class removed)",
               "base": f"runs/checkpoints/{ROOT_CK}"})
    for s in sorted(armF["sds"]):
        if s == max(armF["sds"]):
            continue
        save_ckpt(f"e183_filtered_s{s}", armF["sds"][s],
                  {"desc": f"e131_consolidated_e113 + {s}-step FILTERED "
                           f"freeze (intermediate), seed {FREEZE_SEED}",
                   "steps": int(s), "seed": FREEZE_SEED, "lr": LR_F,
                   "anchor_bank": f"neutral (seed {E170_ANCHOR_SEED})",
                   "random_half": "FILTERED (host-junction class removed)",
                   "base": f"runs/checkpoints/{ROOT_CK}"})

    batteriesF: dict = {"root": root}
    for s in sorted(armF["sds"]):
        log(f"FILTERED +{s} battery")
        batteriesF[str(s)] = measure(armF["sds"][s], f"f{s}", lean=SMOKE)

    # =====================================================================
    # TRAJECTORY + ADJUDICATION (registered clauses; no shopping)
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

    stepsF = [0] + sorted(armF["sds"])
    traceF = trace_from(batteriesF, stepsF)

    gF_seq = [r["gm12"] for r in traceF]
    ck_idxF = [i for i, r in enumerate(traceF) if r["freeze_steps"] > 0]
    gF_50 = next((r["gm12"] for r in traceF if r["freeze_steps"] == 50),
                 traceF[-1]["gm12"])
    gF_min = min(gF_seq[i] for i in ck_idxF)

    still_dissolves = bool(gF_50 <= SHUT_BAR)
    background_carried = bool(gF_min >= SURVIVE_BAR)
    earliest_under = next((traceF[i]["freeze_steps"] for i in ck_idxF
                           if gF_seq[i] <= SHUT_BAR), None)

    # CE-at-dissolution per stream (the R50 splice discipline)
    def ce_at_dissolution(traj_light, trace, light_ce=None):
        step_u = None
        for r in trace:
            if r["freeze_steps"] > 0 and r["gm12"] <= SHUT_BAR:
                step_u = r["freeze_steps"]
                break
        if step_u is None:
            return {"dissolved": False}
        light = next((t for t in traj_light if t["step"] == step_u), None)
        row = next(r for r in trace if r["freeze_steps"] == step_u)
        cce = (light["corpus_ce"] if light else
               (light_ce or {}).get(step_u))
        return {"dissolved": True, "step": step_u,
                "gm12": row["gm12"], "g0": row["g0"],
                "ce_r": row["ce_r"],
                "in_batch_corpus_ce": cce,
                "note": "CE_R + the in-batch corpus CE at the FIRST "
                        "checkpoint under the 0.27 bar — the wash's price at "
                        "the moment of dissolution, not the recovered value"}

    ceD_F = ce_at_dissolution(armF["traj"], traceF)
    ceD_neutral = ce_at_dissolution(
        [], [{"freeze_steps": s, "gm12": g, "g0": g0, "ce_r": c}
             for s, g, g0, c in zip(E176N_ARM_A["armA_steps"],
                                    E176N_ARM_A["armA_gm12"],
                                    E176N_ARM_A["armA_g0"],
                                    E176N_ARM_A["armA_ce_r"])],
        light_ce=E176N_TRAJ_CE)
    ceD_orig = ce_at_dissolution(
        [], [{"freeze_steps": s, "gm12": g, "g0": g0, "ce_r": c}
             for s, g, g0, c in zip(E176_MAIN["freeze_steps"],
                                    E176_MAIN["base_gm12"],
                                    E176_MAIN["base_g0"],
                                    E176_MAIN["ce_r"])])
    ceD_orig["smoke_fine_steps"] = {
        "steps": E176_SMOKE["freeze_steps"],
        "gm12": E176_SMOKE["base_gm12"], "ce_r": E176_SMOKE["ce_r"],
        "note": "e176's only fine-step trajectory is its SMOKE run (s2 g-12 "
                "0.0882 at CE_R 2.0001) — the R50 auditor's splice "
                "attribution; e176N arm A and this cell measure {1,2,4} in "
                "their MAIN runs"}

    cond = {
        "STILL_DISSOLVES": {
            "bar": SHUT_BAR, "gF_at_50": gF_50, "clause": gF_50 <= SHUT_BAR,
            "earliest_ck_le_bar": earliest_under,
            "fires": still_dissolves},
        "BACKGROUND_CARRIED": {
            "bar": SURVIVE_BAR, "gF_min_over_ckpts": gF_min,
            "below_bar_ckpts": [r["freeze_steps"] for r in traceF
                                if r["freeze_steps"] > 0
                                and r["gm12"] < SURVIVE_BAR],
            "fires": background_carried},
    }
    lastF = traceF[-1]
    if still_dissolves:
        verdict = "STILL-DISSOLVES"
        clause = (f"the fact dissolves under the FILTERED stream too: g-12 "
                  f"{gF_50:.4f} <= {SHUT_BAR} at +50 (earliest checkpoint "
                  f"<= bar: +{earliest_under}; trace "
                  + " -> ".join(f"+{r['freeze_steps']}:{r['gm12']:.4f}"
                                for r in traceF)
                  + f") — the 3.84%/window host-junction background was NOT "
                  f"the driver: with the anchor half neutral AND the random "
                  f"half filtered, the consolidated fact still dies. The "
                  f"activity-dependence noun goes UNBOUNDED: no memory state "
                  f"tested retains expression under continued training "
                  f"without fact-bearing windows, on any stream composition "
                  f"tested (extinction anchors / neutral anchors / filtered "
                  f"random half).")
    elif background_carried:
        verdict = "BACKGROUND-CARRIED"
        clause = (f"the fact survives the FILTERED stream: g-12 >= "
                  f"{SURVIVE_BAR} at every checkpoint (min {gF_min:.4f}; "
                  f"+50 {gF_50:.4f}; +300 {lastF['gm12']:.4f}; held30 "
                  f"{lastF['held30_gm12']:.4f}/{lastF['held30_g0']:.4f}) — "
                  f"the 3.84%/window host-junction background WAS the "
                  f"driver of both prior dissolutions (e176's extinction "
                  f"stream AND e176N's neutral stream carried it through "
                  f"the unfiltered random half); the noun stays bounded: "
                  f"memory dies only when the stream carries fact-adjacent "
                  f"(junction) content.")
    else:
        verdict = "TEXTURE"
        clause = (f"neither registered bar fired: g-12 at +50 {gF_50:.4f} "
                  f"(> {SHUT_BAR}) but the minimum over checkpoints "
                  f"{gF_min:.4f} < {SURVIVE_BAR}; trace "
                  + " -> ".join(f"+{r['freeze_steps']}:{r['gm12']:.4f}"
                                for r in traceF)
                  + f" — SLOWER WASH texture: the filtered stream erodes the "
                  f"fact on a slower clock than the unfiltered streams (the "
                  f"background contributed rate, not the whole kill); full "
                  f"curve reported, no bar shopping.")
    log("=" * 78)
    log(f"E183 VERDICT: {verdict}")
    log(f"  filtered g-12 trace: " + " -> ".join(
        f"+{r['freeze_steps']}:{r['gm12']:.4f}" for r in traceF))
    log(f"  filtered g0 trace:   " + " -> ".join(
        f"+{r['freeze_steps']}:{r['g0']:.4f}" for r in traceF))
    log(f"  vs e176 original +50 {E176_MAIN['base_gm12'][1]:.4f} | e176N "
        f"neutral +50 {E176N_ARM_A['armA_gm12'][4]:.4f}")
    log(f"  CE-at-dissolution: filtered {ceD_F} | neutral {ceD_neutral}")
    log(f"  filter: {filter_record['rejections_total']} rejections / "
        f"{filter_record['candidates_tested']} candidates = "
        f"{100 * filter_record['realized_rejection_rate']:.2f}% "
        f"(expected {100 * bg_rate:.2f}%)")
    log(f"  {clause}")
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e183_filtered_stream",
        "date": common.now_iso(),
        "registration": ("QUEUE.md row e183 (T109's named residue — the last "
                         "gate on the unbounded noun); bars frozen verbatim "
                         "in the module docstring before compute"),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("does the consolidated fact STILL dissolve when the "
                     "random-corpus half's 3.84%/window host-junction "
                     "background is FILTERED OUT (the truly clean stream), "
                     "or was that background the driver of both prior "
                     "dissolutions?"),
        "root": f"runs/checkpoints/{ROOT_CK} (gated vs e151 before-cells, "
                f"max|diff| {G_ROOT['max_abs_diff']:.2e})",
        "root_meta": root_meta,
        "armF_filtered": {
            "desc": "ONE plain-corpus freeze, e176N arm A VERBATIM except the "
                    "random half is FILTERED: batch 32 = 16 neutral-bank "
                    "draws (e170's bank, bit-identical to e176N arm A's) + "
                    "16 random corpus windows rejection-sampled at draw time "
                    "(no FLORIZEL/ELIZABETH substring in [s, s+257), no host "
                    "onset in [s, s+257)); full-token CE (NO fact windows, NO "
                    "name tokens, NO mask), AdamW (0.9,0.95) wd 0.1 lr 1e-3 "
                    "constant, clip 1.0, seed 10902 (redraws diverge the draw "
                    "sequence from e176N's after the first rejection — "
                    "necessary consequence of the filter)",
            "ckpt_steps": list(CK_F), "steps_ran": armF["steps_ran"],
            "seed": armF["seed"], "lr": armF["lr"],
            "traj": armF["traj"],
            "zeph_violations": armF["zeph_violations"],
            "host_leaks": armF["host_leaks"],
            "missing_checkpoints": [s for s in CK_F if s not in armF["sds"]],
        },
        "filter_record": filter_record,
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
                     "geometries_measured": {"novel": [-12, 12],
                                             "trained_g0": 0},
                     "measure_dial": "e176N's measure() (= e176's minus the "
                                     "183-span census, e178's recorded "
                                     "deviation, inherited)"},
        "gates": {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                  "G_POOL": G_POOL, "G_ANCHOR": G_ANCHOR, "G_ROOT": G_ROOT,
                  "G_DRAWFREE": G_DRAWFREE, "G_FILTER": G_FILTER,
                  "G_SURG": gates_surg},
        "trace_armF": traceF,
        "trace_summary": {
            "armF_steps": stepsF,
            "armF_gm12": [r["gm12"] for r in traceF],
            "armF_g0": [r["g0"] for r in traceF],
            "armF_gp12": [r["gp12"] for r in traceF],
            "armF_held30_gm12": [r["held30_gm12"] for r in traceF],
            "armF_held30_g0": [r["held30_g0"] for r in traceF],
            "armF_row0": [r.get("row0_strength") for r in traceF],
            "armF_dall_g0": [r.get("dall_g0") for r in traceF],
            "armF_ce_r": [r["ce_r"] for r in traceF],
            "e176_main_steps": E176_MAIN["freeze_steps"],
            "e176_main_gm12": E176_MAIN["base_gm12"],
            "e176_main_g0": E176_MAIN["base_g0"],
            "e176_smoke_steps": E176_SMOKE["freeze_steps"],
            "e176_smoke_gm12": E176_SMOKE["base_gm12"],
            "e176_smoke_ce_r": E176_SMOKE["ce_r"],
            "e176n_armA_steps": E176N_ARM_A["armA_steps"],
            "e176n_armA_gm12": E176N_ARM_A["armA_gm12"],
            "e176n_armA_g0": E176N_ARM_A["armA_g0"],
            "e176n_armA_ce_r": E176N_ARM_A["armA_ce_r"],
        },
        "batteries_armF": batteriesF,
        "references": {
            "e176_main": {"ref": E176_MAIN, "provenance": src_e176,
                          "note": "the original extinction-confounded stream "
                                  "(lr 1e-3) — stream 1 of 3"},
            "e176_smoke": {"ref": E176_SMOKE, "provenance": src_e176s,
                           "note": "e176's ONLY fine-step trajectory (the "
                                   "smoke run)"},
            "e176n_armA": {"ref": E176N_ARM_A, "provenance": src_e176n,
                           "note": "the neutral stream (lr 1e-3, 0/16 "
                                   "junction anchors, UNFILTERED random "
                                   "half) — stream 2 of 3"},
        },
        "ce_at_dissolution": {"armF_filtered": ceD_F,
                              "e176n_neutral": ceD_neutral,
                              "e176_original": ceD_orig,
                              "critique": "the R50 splice discipline: report "
                                          "the CE at the DISSOLUTION moment, "
                                          "not just recovered values"},
        "adjudication": {"conditions": cond, "verdict": verdict,
                         "clause": clause},
        "honesty_reflex": {
            "filter_fidelity": (f"the filter removes EXACTLY the registered "
                                f"class (host substring in [s, s+257) OR "
                                f"host onset in [s, s+257)); realized "
                                f"{100 * filter_record['realized_rejection_rate']:.2f}% "
                                f"over {filter_record['candidates_tested']} "
                                f"candidates vs the {100 * bg_rate:.2f}% "
                                f"measurement; every accepted window passed "
                                f"an independent post-hoc grep (0 leaks)"),
            "what_the_filter_could_not_remove": (
                f"(1) LEFT-EDGE STRADDLES: {filter_record['accepted_left_edge_straddle']} "
                f"accepted windows carry a host-name TAIL at their left edge "
                f"(onset before the window) — not in the 3.84% class, no "
                f"pre-host context, hence no context->onset junction "
                f"signal, but they are host-adjacent text and are COUNTED, "
                f"not hidden. (2) ORDINARY CORPUS PRESSURE: the stream "
                f"remains full-token CE plain-corpus training at lr 1e-3 — "
                f"the disuse mechanism itself is the object under test and "
                f"cannot be filtered without emptying the stream. (3) The "
                f"CE_R eval bank and the neutral anchor bank are host-free "
                f"but still genuine corpus text."),
            "rng_divergence": ("rejection redraws consume extra generator "
                               "draws: after the first rejection the draw "
                               "sequence diverges from e176N arm A's (same "
                               "seed/modulus; the anchor bank content is "
                               "bit-identical and gated) — the filtered arm "
                               "is a sibling, not a paired draw-sequence "
                               "twin"),
            "single_seed": ("ONE trajectory (seed 10902), n=1 per cell; "
                            "e152R showed seed-to-seed timing spans an order "
                            "of magnitude on related washes — the fine-step "
                            "clock is this trajectory's, not a replicated "
                            "law"),
            "stream_composition": ("the filtered stream = 16 neutral-anchor "
                                   "draws + 16 filtered-random draws per "
                                   "batch; the UNFILTERED streams it is "
                                   "compared against carried ~3.84%/window "
                                   "junction content in their random half "
                                   "(~184 junction windows over 300 steps) "
                                   "— the delta is exactly that content"),
            "thread_bit_drift": "evals at 4 threads vs e151's stored cells "
                                "can drift low-order bits; gates report both "
                                "the 5e-6 bit flag and the 0.05 fallback "
                                "tolerance",
            "bars_anchored": "the dissolve/survive bars are the same "
                             "absolute home-battery bars e158/e161/e176/"
                             "e176N used (0.27 SHUT / 0.50 SURVIVE) — all "
                             "three streams' deaths are measured on one "
                             "ruler",
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

    plot(rd / "filtered_stream.png", traceF, armF["traj"], cond, verdict,
         clause, ceD_F, ceD_neutral, filter_record, bg_rate)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'filtered_stream.png'}, "
        f"{len(CKPT_INVENTORY)} checkpoints in runs/checkpoints/e183_*.pt")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot

def plot(path, traceF, trajF, cond, verdict, clause, ceD_F, ceD_neutral,
         filter_record, bg_rate):
    """THE figure: the three-stream comparison COMPLETED (extinction /
    neutral / FILTERED), the filtered anatomy + CE_R + the dissolution
    marker, the fine-step zoom, and the filter record + verdict panel."""
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 10.5))
    xsF = [r["freeze_steps"] for r in traceF]
    xe = E176_MAIN["freeze_steps"]
    xes = E176_SMOKE["freeze_steps"]
    xn = E176N_ARM_A["armA_steps"]

    # (0,0) THE THREE-STREAM COMPARISON (g-12 headline)
    ax = axes[0, 0]
    ax.plot(xe, E176_MAIN["base_gm12"], "v--", lw=1.8, ms=7, color="crimson",
            alpha=0.75,
            label="e176 EXTINCTION g-12 (host anchors + 3.84% background)")
    ax.plot(xn, E176N_ARM_A["armA_gm12"], "s-", ms=7, lw=2.0,
            color="darkorange",
            label="e176N NEUTRAL g-12 (plain anchors + 3.84% background)")
    ax.plot(xsF, [r["gm12"] for r in traceF], "o-", ms=9, lw=2.8,
            color="seagreen",
            label="e183 FILTERED g-12 (plain anchors + background REMOVED) HEADLINE")
    ax.plot(xsF, [r["g0"] for r in traceF], "^-", ms=6, lw=1.6,
            color="mediumseagreen", label="filtered g0")
    for yv, col, lbl in ((SURVIVE_BAR, "seagreen",
                          f"{SURVIVE_BAR} BACKGROUND-CARRIED bar (survive through +300)"),
                         (SHUT_BAR, "tab:purple",
                          f"{SHUT_BAR} STILL-DISSOLVES bar (by +50)")):
        ax.axhline(yv, ls="--", lw=1.1, color=col, alpha=0.8, label=lbl)
    ax.annotate(f"root g-12 {traceF[0]['gm12']:.3f}",
                (0, traceF[0]["gm12"]), textcoords="offset points",
                xytext=(6, 4), fontsize=7.5, color="seagreen")
    ax.annotate(f"e176 +50 {E176_MAIN['base_gm12'][1]:.3f}",
                (E176_MAIN["freeze_steps"][1], E176_MAIN["base_gm12"][1]),
                textcoords="offset points", xytext=(-14, 8), fontsize=7,
                color="crimson", alpha=0.85)
    ax.annotate(f"e176N +50 {E176N_ARM_A['armA_gm12'][4]:.3f}",
                (E176N_ARM_A["armA_steps"][4], E176N_ARM_A["armA_gm12"][4]),
                textcoords="offset points", xytext=(-4, 10), fontsize=7,
                color="darkorange", alpha=0.9)
    ax.set_xlabel("plain-corpus freeze steps from the root (step 0 = root)")
    ax.set_ylabel("absolute mean p(Z), install-60 battery")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7.0, loc="center right")
    ax.set_title("THE THREE-STREAM COMPARISON, COMPLETED — extinction / "
                 f"neutral / FILTERED -> {verdict}", fontsize=10)

    # (0,1) filtered anatomy + CE_R (with the dissolution marker)
    ax = axes[0, 1]
    has_anatomy = traceF[0].get("row0_strength") is not None
    if has_anatomy:
        ax.plot(xsF, [r["row0_strength"] for r in traceF], "^-", ms=6,
                lw=1.8, color="tab:cyan", label="filtered row-0 (the sink)")
        ax.plot(xsF, [r["dall_g0"] for r in traceF], "s-", ms=5, lw=1.6,
                color="tab:blue", label="filtered D-all g0")
        ax.plot(xsF, [r["held30_gm12"] for r in traceF], "D-", ms=5, lw=1.4,
                color="crimson", alpha=0.8, label="filtered held30 g-12")
    axr = ax.twinx()
    axr.plot(xsF, [r["ce_r"] for r in traceF], "k:o", ms=5, lw=1.3,
             label="filtered CE_R")
    axr.set_ylabel("CE_R")
    if ceD_F.get("dissolved"):
        ax.axvline(ceD_F["step"], color="tab:purple", ls=":", lw=1.6,
                   alpha=0.85)
        cce = ceD_F.get("in_batch_corpus_ce")
        axr.annotate(
            f"DISSOLUTION +{ceD_F['step']}: g-12 {ceD_F['gm12']:.4f}, "
            f"CE_R {ceD_F['ce_r']:.3f}"
            + (f", in-batch CE {cce:.3f}" if cce is not None else ""),
            (ceD_F["step"], ceD_F["ce_r"]), textcoords="offset points",
            xytext=(8, 10), fontsize=7.2, color="tab:purple",
            arrowprops=dict(arrowstyle="->", color="tab:purple", lw=0.8))
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("plain-corpus freeze steps (filtered stream)")
    ax.set_ylabel("strength / p(Z)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.0, loc="center right")
    ax.set_title("the FILTERED stream's anatomy + CE_R — the dotted line "
                 "marks DISSOLUTION (CE reported AT the moment, per the R50 "
                 "splice discipline)", fontsize=9.5)

    # (1,0) the fine-step zoom (symlog): all three streams' fine steps
    ax = axes[1, 0]
    ax.plot(xes, E176_SMOKE["base_gm12"], "v--", ms=9, lw=1.8,
            color="crimson", alpha=0.75,
            label="e176 SMOKE g-12 (extinction stream's only fine steps)")
    ax.plot(xn[:5], E176N_ARM_A["armA_gm12"][:5], "s-", ms=8, lw=2.0,
            color="darkorange", label="e176N MAIN-RUN g-12 (neutral, fine steps)")
    fine = [r for r in traceF if r["freeze_steps"] <= 50]
    ax.plot([r["freeze_steps"] for r in fine], [r["gm12"] for r in fine],
            "o-", ms=9, lw=2.4, color="seagreen",
            label="e183 MAIN-RUN g-12 (filtered, fine steps) HEADLINE")
    for r in fine:
        if 0 < r["freeze_steps"] <= 4:
            ax.annotate(f"{r['gm12']:.3f}", (r["freeze_steps"], r["gm12"]),
                        textcoords="offset points", xytext=(4, 6),
                        fontsize=7, color="seagreen")
    for s, g in zip(xn[1:4], E176N_ARM_A["armA_gm12"][1:4]):
        ax.annotate(f"{g:.3f}", (s, g), textcoords="offset points",
                    xytext=(4, -10), fontsize=7, color="darkorange")
    ax.axhline(SHUT_BAR, ls="--", lw=1.0, color="tab:purple", alpha=0.7)
    ax.axhline(SURVIVE_BAR, ls="--", lw=1.0, color="seagreen", alpha=0.7)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel("freeze steps (zoom 0..50; symlog scale)")
    ax.set_ylabel("mean p(Z) (symlog)")
    ax.legend(fontsize=7.0, loc="lower left")
    ax.set_title("the two-step clock under the filter: e183 (main run) vs "
                 "e176N neutral vs e176 smoke", fontsize=10)

    # (1,1) the filter record + CE-at-dissolution + the verdict panel
    ax = axes[1, 1]
    ax.axis("off")
    ytxt = 0.97
    ax.text(0.02, ytxt, "THE FILTER (the only protocol change vs e176N arm A):",
            fontsize=8.5, va="top", family="monospace", weight="bold")
    ytxt -= 0.040
    fr = filter_record
    for line in [
        f"  spec: reject s if [s, s+257) contains FLORIZEL/ELIZABETH (grep)",
        f"        or covers a host junction (onset p in [s, s+257))",
        f"  candidates tested: {fr['candidates_tested']}   redraws: {fr['redraws']}",
        f"  rejections: {fr['rejections_total']} = {100 * fr['realized_rejection_rate']:.2f}% "
        f"(expected {100 * bg_rate:.2f}% — the e176N measurement)",
        f"    by class: substring-only {fr['rejected_substring_only']}, "
        f"junction-only {fr['rejected_junction_only']},",
        f"    both {fr['rejected_substring_and_junction']}; post-hoc grep leaks 0",
        f"  accepted: {fr['accepted']} windows; left-edge straddle residue "
        f"(host tail, no context): {fr['accepted_left_edge_straddle']}",
    ]:
        ax.text(0.02, ytxt, line, fontsize=7.2, va="top", family="monospace")
        ytxt -= 0.030
    ytxt -= 0.012
    ax.text(0.02, ytxt, "CE-AT-DISSOLUTION (the R50 splice discipline):",
            fontsize=7.6, va="top", family="monospace", weight="bold")
    ytxt -= 0.032

    def ce_line(name, ceD):
        if ceD.get("dissolved"):
            cce = ceD.get("in_batch_corpus_ce")
            return (f"  {name}: +{ceD['step']} g-12 {ceD['gm12']:.4f} "
                    f"CE_R {ceD['ce_r']:.3f}"
                    + (f" in-batch {cce:.3f}" if cce is not None else ""))
        return f"  {name}: NOT dissolved (no checkpoint <= {SHUT_BAR})"

    for line in [ce_line("e183 filtered ", ceD_F),
                 ce_line("e176N neutral ", ceD_neutral),
                 ce_line("e176 extinction", {
                     "dissolved": True, "step": 50,
                     "gm12": E176_MAIN["base_gm12"][1],
                     "ce_r": E176_MAIN["ce_r"][1]})]:
        ax.text(0.02, ytxt, line, fontsize=7.2, va="top", family="monospace")
        ytxt -= 0.030
    ytxt -= 0.015
    ax.text(0.02, ytxt, f"E183 VERDICT: {verdict}", fontsize=9.0, va="top",
            family="monospace", weight="bold", color="darkred")
    ytxt -= 0.036
    for wd in [clause[i:i + 88] for i in range(0, len(clause), 88)]:
        ax.text(0.02, ytxt, f"  {wd}", fontsize=6.8, va="top",
                family="monospace")
        ytxt -= 0.027

    fig.suptitle("E183 — THE FILTERED STREAM: does the fact survive the "
                 "truly clean stream (the 3.84% host-junction background "
                 f"removed)? -> {verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
