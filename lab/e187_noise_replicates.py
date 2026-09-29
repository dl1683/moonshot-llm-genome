"""E187 — THE NOISE REPLICATES (R52's flagged debt, the mechanism noun's
last formality; NO QUEUE ROW — these bars are the registration, frozen here
before compute; dispatched post-T114).

WHY (T114): E185's NOISE-KILLS verdict — content-free noise at
displacement-match dissolved the consolidated fact IDENTICALLY to the
corpus (in ORTHOGONAL directions: cos ~ -0.09 at +1, ~ -0.04 after) — was
drawn at ONE seed per noise arm (labels-iid 18501; shuffled-target 18502;
recorded deviation: "replicate seeds are owed before the paper's noun
moves"). The skeleton's mechanism paragraph has ALREADY moved to the
no-basin form ("these memories have no basin; what keeps them is the
dataloader's direction — and even that kills, just neatly"); the lab's own
rule says replicates are owed FIRST. A non-killing replicate re-inverts
the mechanism toward corpus-directed and re-bounds the noun.

WHAT THIS CELL DOES: E185's noise arms VERBATIM at 2 additional noise
seeds per arm — labels-iid at 18503/18504 (original 18501);
shuffled-target at 18505/18506 (original 18502). Same root, same input
stream (seed 10902 — gated bit-identical to E185's stored per-step input
hashes, so the replicate isolates the NOISE DRAW, not the input draw),
same optimizer/lr/clip, same steps {1,2,4,10}, same dial, same ruler.

REGISTERED BARS (frozen here before compute; the dispatch's registration
verbatim; no bar shopping — adjudicate against exactly this):
  - NOISE-KILLS-REPLICATES fires if: both arms kill (g-12 <= 0.27 by +2)
    at BOTH new seeds (the mechanism formally licensed — the no-basin
    noun stands).
  - ANY-SPARES fires if: any arm/seed combination leaves the fact
    >= 0.5 (the mechanism re-inverts toward corpus-directed; the noun
    re-bounds).

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * g-12 / g0 / g+12 / held30 = ABSOLUTE install-60 / held-30 battery
    mean p(Z) at ctx offsets -12 / 0 / +12 (e176/e176N/e184/e185's
    convention verbatim — the same batteries, the same corpus rebuild,
    the same ruler); "the fact" = g-12.
  * KILL "by +2" = g-12 at checkpoint +2 <= 0.27 (the SHUT bar,
    e158/e161/e176/e176N/e184/e185). +2 is where the corpus control died
    (e176N stored 0.0271; e185 re-run bit-identical) AND where e185's
    measured displacement-match landed (M=+2, D_kill 2.4893). g-12@+1
    co-reported everywhere (e185: both arms were already dead BELOW
    match at +1 — the fortiori read).
  * SPARE = g-12 >= 0.50 (the SURVIVE bar) at ANY matched checkpoint in
    {+2,+4,+10} (matched = cumulative displacement >= D_kill, e185's
    operationalization verbatim; +1 is pre-match — the corpus control
    itself read 0.678 there). The ANY-quantifier is taken at the
    broadest honest scope (a spare anywhere post-match re-opens the
    mechanism).
  * COMPOSITE ORDER (frozen): ANY-SPARES -> NOISE-KILLS-REPLICATES ->
    TEXTURE. NOISE-KILLS-REPLICATES = all four new cells kill by +2 AND
    no cell spares. ANY-SPARES = at least one cell spares. Anything
    else (e.g. partial damage, displacement-unmatched) = TEXTURE with
    numbers; no bar shopping.
  * DISPLACEMENT = cumulative parameter delta norm ||theta_t - theta_0||_2
    over ALL 2,739,072 trainable parameters (fp32, CPU, measured every
    step — never assumed) + per-step increments, exactly e185's currency.
  * D_kill = 2.4892616271972656 (e185's stored control displacement at
    its kill step +2) — RE-DERIVED on this device from e185's stored
    control checkpoints (runs/checkpoints/e185_ctrl_s*.pt) and GATED vs
    the stored displacement table (tol 5e-6; see G_E185CKPTS, which also
    re-derives e185's stored noise/shuf displacements AND cosines from
    the stored checkpoints — the corpus-direction reference is therefore
    verified, not trusted).
  * THE CORPUS DIRECTION (for the cosines) = e185's control delta
    vectors at {1,2,4,10}, recomputed from the stored control
    checkpoints on THIS device (no cross-device displacement/cosine
    comparison anywhere: all four new trainings run this process, this
    CPU, 4 threads; the control never re-trains — its tensors are the
    gated stored ones).
  * per arm/seed: the e185 formality is CO-ADJUDICATED: M = earliest
    checkpoint with ||delta|| >= D_kill; KILLS@M = g-12 at M <= 0.27;
    a kill first firing below D_kill co-reported as fortiori.
  * CE at each step (the arm's own in-batch training CE) + CE_R at every
    checkpoint (the noise's price), the full dial (base/held/site-read/
    old-band census/deletion table) at every checkpoint.

THE FOUR CELLS (one file, one run; all training-light / eval-heavy;
CPU-ONLY by dispatch — e182 owns the GPU, never touched):
  noise18503 = NOISE-LABELS seed 18503 (y ~ i.i.d. uniform over the
               65-token vocab; destroys mapping AND target statistics)
  noise18504 = NOISE-LABELS seed 18504
  shuf18505  = SHUFFLED-TARGET seed 18505 (flat randperm of the true
               targets per step; preserves the exact target multiset,
               destroys only the mapping)
  shuf18506  = SHUFFLED-TARGET seed 18506
  All four: batch 32 = 16 neutral-anchor draws + 16 random corpus windows
  (e170's FIXED neutral bank, seed 170), full-token CE, AdamW (0.9,0.95)
  wd 0.1 constant lr 1e-3 clip 1.0, generator seed 10902 — the ONLY
  inter-cell delta is the TARGET tensor of each step (and across seeds,
  only the noise generator's seed; the seed-10902 aj/rj input draws
  precede any noise draw, so every input batch is bit-identical across
  all four cells AND vs e185's stored per-step hashes — gated by md5).
  10 steps, checkpoints {1,2,4,10}; full dial + CE_R per checkpoint.

COMPUTE ENVELOPE (dispatch): CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced
before torch, as e176n/e185), LOW threads <= 4 (torch.set_num_threads(4)),
one 25 s launch stagger (single sleep, no busy-waiting anywhere),
cooldown 60 s before and after EACH of the four trainings, per-training
cap 180 s (e185 measured ~1.5-2.5 s/step at 4 threads; 10 steps ~ 15-25 s).

INSTRUMENT PROVENANCE: every instrument is lab/e185_noise_wash.py VERBATIM
(load_cpu/evl_load/battery_cell/battery_pz/ce_fixed_cpu/val_windows/
deleted_wpe/read_fact_at/row_census/measure()/flat_cells/gate_vs = the
e176n lineage; noise_wash = e185's trainer verbatim, target substitution +
displacement bookkeeping unchanged; only the ARM SET and the registration
differ). Copied, not imported, to own the device policy.

NETS: root runs/checkpoints/e131_consolidated_e113.pt (gate bit-exact vs
e151's stored before-cells, e176n's G_ROOT set). References:
runs/e185/metrics.json + runs/checkpoints/e185_{ctrl,noise,shuf}_s*.pt
(the n=1 originals; embedded copies verified vs the stored file at run
time — e185's verify convention). New checkpoints:
runs/checkpoints/e187_{noise,shuf}<seed>_s{1,2,4,10}.pt.

Outputs: runs/e187/{metrics.json, noise_replicates.png} (the n=3 overlay
per arm: e185's stored original seed + the two new seeds). No NOTES/
THINKING/QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python e187_noise_replicates.py    (E187_SMOKE=1 shakedown)
      E187_RECOVERY=1: finish from the checkpoints that survived the
      2026-09-28 outage (eval-only where the grid is complete; the minimal
      re-run where it is not; G_FIDELITY gates the lineage; the registration
      above is untouched — see the deviations list)
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import sys
import textwrap
import time
from pathlib import Path

import os as _os
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e176n/e185)

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

SMOKE = os.environ.get("E187_SMOKE") == "1"
# RECOVERY (2026-09-29): finish from the survivors of the 2026-09-28 outage
# (the previous process completed the trainings but died before any output;
# see deviations + metrics["recovery"]). Cells whose checkpoint grid is
# complete are EVAL-ONLY; an incomplete cell is re-run in full (the minimal
# completion — AdamW state is not checkpointed) and G_FIDELITY gates the
# re-run's deltas vs its surviving checkpoints (bit-identity expected).
RECOVERY = os.environ.get("E187_RECOVERY") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e187 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
ROOT_CK = "e131_consolidated_e113.pt"   # THE FULLY-CONSOLIDATED ROOT
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E185_METRICS = E43.REPO / "runs" / "e185" / "metrics.json"

# ---- e152's placement constants (the measurement instruments rebuild these) ----
RETEACH_J = 54                    # offset: name x-cols 184..190, read rows 183..189
SITE_ADDR_ROW = 183               # the onset read row (= e120's SPLICE_ADDR_ROW)
SITE_Z_XCOL = PRE + RETEACH_J     # 184 (= e120's Z_XCOL)
SITE_CONT = BLOCK - PRE - RETEACH_J - len(NAME)     # 65-token continuation
D_ALL = (121, 125, 129, 133, 137)                   # e113's fixed set
GEOS = (-12, 0, 12)               # novel x2 + the trained g0

# ---- census row set (e158/e161/e176/e176n/e184/e185 old-band convention) ------
ROWS_OLD = (0, 1, 2, 3, 4, 5, 6, 60, 100, 118, 119, 120) + tuple(range(121, 130))
if SMOKE:
    ROWS_OLD = (0, 1, 60, 121, 125, 129)

# ---- the four cells --------------------------------------------------------------
CK_MAIN: tuple[int, ...] = (1, 2, 4, 10) if not SMOKE else (1, 2)
N_STEPS = CK_MAIN[-1]

# ---- fine-tune envelope (e185 VERBATIM; only the targets differ) ----------------
LR = 1e-3                          # the wash's own lr (all cells)
FT_TIME_CAP = 180.0                # dispatch: <=180 s per training
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
FREEZE_SEED = 10902               # the locked seed lineage (= e161/e176/e176n/e185)
NOISE_SEEDS = {"noise": (18503, 18504), "shuf": (18505, 18506)}
E185_ORIG_SEEDS = {"noise": 18501, "shuf": 18502}
COOLDOWN_S = 60.0                  # around EACH of the four trainings
STAGGER_S = 25.0                   # launch stagger vs the CPU fleet

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

# ---- E185's stored n=1 originals (embedded verbatim from runs/e185/metrics.json;
# verified vs the file at run time — verify_e185; the n=3 overlay's first member).
E185_TRACE_STEPS = [0, 1, 2, 4, 10]
E185_GM12 = {                     # full-dial g-12 traces (freeze_steps above)
    "ctrl": [0.9155886173248291, 0.6780440807342529,
             0.027077054604887962, 0.010940761305391788,
             0.019353048875927925],
    "noise": [0.9155886173248291, 0.0002546094765421003,
              0.0007630126783624291, 0.021637430414557457,
              0.010607769712805748],
    "shuf": [0.9155886173248291, 3.487885260256007e-05,
             9.778873391041998e-06, 6.469085928983986e-05,
             0.00016334644169546664],
}
E185_DISP = {                     # cumulative displacement at the checkpoints
    "ctrl": {1: 1.6542880535125732, 2: 2.4892616271972656,
             4: 3.526789665222168, 10: 5.195915222167969},
    "noise": {1: 1.6541621685028076, 2: 2.6485204696655273,
              4: 4.002531051635742, 10: 6.385190010070801},
    "shuf": {1: 1.6541498899459839, 2: 2.650214433670044,
             4: 4.0620503425598145, 10: 6.566292762756348},
}
E185_COS = {                      # cos(arm delta, ctrl delta) at the checkpoints
    "ctrl": {1: 0.9999352097511292, 2: 1.0000027418136597,
             4: 1.0000380277633667, 10: 1.0000596046447754},
    "noise": {1: -0.09102325141429901, 2: -0.04463526979088783,
              4: -0.033226702362297965, 10: -0.03346420079469681},
    "shuf": {1: -0.09880252927541733, 2: -0.03536924719810486,
             4: -0.03541024029254913, 10: -0.03878234326839447},
}
E185_XHASH = {                    # per-step input-batch md5 (seed-10902 stream)
    1: "1ea27bffde6c4a53be8badf5ab453d64",
    2: "1d6f0e55cc6a25ece947d2040528225e",
    3: "b5c0b670270406a94aca63071b051468",
    4: "cdccea0c413e603dc52d1873e37b9844",
    5: "4da7b67a7fd80b4e9729731fb27bec0c",
    6: "1aa4f9f250f14acad52d3b969343090a",
    7: "1e9e3373028935fe272d86683280e4b5",
    8: "3535a9db2d1aa2e6e0655208ff26b3d9",
    9: "2c3260242cd38e60cffafe0c91495c5e",
    10: "688062bb39f486e563b091124a7231a1",
}
E185_YSTATS = {"noise": 0.9839338183403015, "shuf": 0.9433578550815582}
T_KILL_E185 = 2                    # e185's control kill step (g-12 0.0271)
CTRL_GM12_AT_TKILL = 0.027077054604887962
D_KILL_STORED = 2.4892616271972656  # e185's control displacement at +2

# ---- registered bar constants (frozen; see OPERATIONALIZATIONS above) ---------
SHUT_BAR = 0.27                   # DISSOLVE (e158/e161/e176/e176N/e184/e185)
SURVIVE_BAR = 0.50                # SPARE
ROOT_GM12 = E151_ROOT["base_gm12"]
ROOT_G0 = E151_ROOT["base_g0"]
MATCH_STEPS = tuple(s for s in CK_MAIN if s >= T_KILL_E185)   # {2,4,10}

REGISTERED_PREDICTION = {
    "noise_kills_replicates": "NOISE-KILLS-REPLICATES fires if: both arms "
        "kill (g-12 <= 0.27 by +2) at BOTH new seeds (the mechanism formally "
        "licensed — the no-basin noun stands).",
    "any_spares": "ANY-SPARES fires if: any arm/seed combination leaves the "
        "fact >= 0.5 (the mechanism re-inverts toward corpus-directed; the "
        "noun re-bounds).",
    "operationalizations": "g-12/g0/g+12/held30 = absolute install-60/held-30 "
        "battery mean p(Z) at ctx offsets -12/0/+12 (e185's convention); "
        "KILL by +2 = g-12 at checkpoint +2 <= 0.27 (g-12@+1 co-reported; the "
        "e185 displacement-match formality co-adjudicated at D_kill "
        "2.4892616271972656, re-derived from e185's stored ctrl checkpoints "
        "and gated); SPARE = g-12 >= 0.50 at ANY matched checkpoint "
        "{+2,+4,+10} (matched = cum disp >= D_kill; +1 is pre-match — the "
        "corpus control read 0.678 there); COMPOSITE ORDER frozen: "
        "ANY-SPARES -> NOISE-KILLS-REPLICATES -> TEXTURE; NOISE-KILLS-"
        "REPLICATES = all four new cells kill by +2 AND no cell spares; "
        "displacement measured per step (cumulative + increment, all "
        "2,739,072 params, fp32, CPU); cosines vs the corpus direction "
        "(e185 ctrl deltas recomputed from stored checkpoints on this "
        "device, gated); CE at each step + CE_R and the full dial at every "
        "checkpoint.",
    "registration": "NO QUEUE ROW — the dispatch's registration IS the "
        "registration, frozen verbatim in this docstring before compute. "
        "Adjudicate against exactly this; no bar shopping.",
}

trims: list[str] = []
deviations: list[str] = [
    "CPU-ONLY by dispatch (CUDA_VISIBLE_DEVICES=-1 forced before torch, "
    "e176n/e185 precedent): torch threads 4, one 25 s launch stagger (single "
    "sleep, no busy-waiting), cooldown 60 s before/after EACH of the four "
    "trainings, per-training CPU cap 180 s (e185 measured ~1.5-2.5 s/step at "
    "4 threads; 10 steps ~ 15-25 s).",
    "NO control re-training: e185's stored control checkpoints "
    "(runs/checkpoints/e185_ctrl_s{1,2,4,10}.pt) supply BOTH the corpus "
    "direction (delta vectors recomputed vs the root on this device) and "
    "D_kill (re-derived and gated vs the stored displacement table, tol "
    "5e-6). G_E185CKPTS additionally re-derives e185's stored noise/shuf "
    "displacements AND their stored cosines from the stored checkpoints — "
    "the entire e185 reference structure is verified on this device, not "
    "trusted; no cross-device displacement/cosine comparison exists "
    "anywhere in this cell.",
    "The replicate isolates the NOISE DRAW, not the input draw: the "
    "seed-10902 aj/rj sequence is unchanged, so every input batch is "
    "bit-identical across the four new cells AND vs e185's stored per-step "
    "md5s (G_INPUTS gates all three ways). The four noise generators "
    "(18503/18504/18505/18506) are independent dedicated generators drawn "
    "AFTER the input draws.",
    "Seed numbering convention: e185's original noise seeds were 18501 "
    "(labels-iid) / 18502 (shuffled); the replicates take the next four "
    "integers in arm order — 18503/18504 (labels-iid), 18505/18506 "
    "(shuffled) — per the dispatch's explicit example.",
    "KILL 'by +2' is operationalized at checkpoint +2 (the corpus kill "
    "step AND e185's measured displacement-match step); g-12@+1 is "
    "co-reported everywhere (e185: both arms were already dead below match "
    "at +1 — the fortiori read).",
    "SPARE is operationalized at the broadest honest scope — any matched "
    "checkpoint {+2,+4,+10} — because ANY-SPARES is the re-opening bar; a "
    "cell that dies by +2 and revives to >= 0.50 later still fires "
    "ANY-SPARES (the fact survived the noise somewhere post-match).",
    "Nets are the mandated 2.7M e131_consolidated line (e143/e151/e152/"
    "e158/e176n/e184/e185 precedent — every gate reference lives on this "
    "line).",
    "Eval thread count is 4 (dispatch) vs e151's stored cells — CPU "
    "reduction order can drift low-order bits; the G_ROOT gate reports both "
    "the 5e-6 bit flag and the 0.05 fallback tolerance (e161/e176/e176n/"
    "e184/e185 precedent).",
    "n=3 per arm after this cell (18501+18503+18504; 18502+18505+18506) — "
    "still ONE input stream (10902) and ONE root: the replicate samples "
    "the noise draw, not the organism; recorded as the standing scope "
    "limit on the noun's license.",
    "Smoke mode trims: 2-step trainings, checkpoints {1,2}, lean measures "
    "(no censuses, no deletion table), no cooldowns; nothing adjudicated.",
]

if RECOVERY:
    deviations.append(
        "RECOVERY RUN (E187_RECOVERY=1, 2026-09-29): the 2026-09-28 system "
        "outage killed the previous process AFTER all four trainings but "
        "BEFORE any output (survivors: this script verbatim, 15/16 "
        "checkpoints under runs/checkpoints/e187_*.pt, and "
        "runs/e187_run.log). Diagnosis from the log: shuf18505's training "
        "hit the registered 180 s CPU cap at step 9 (outage onset — the "
        "same 9 steps that took noise18503 ~17 s took shuf18505 184 s), so "
        "its s10 checkpoint is missing, and the process then died at the "
        "G_INPUTS assert on that cell's MISSING step-10 hash — NOT on any "
        "divergence: the reconstructed seed-10902 stream matches e185's "
        "stored per-step md5s bit-exactly (verified before this recovery "
        "and re-gated live by G_INPUTS below). Policy per dispatch: "
        "EVAL-ONLY from survivors where the checkpoint grid is complete "
        "(noise18503, noise18504, shuf18506); the minimal re-run where it "
        "is not (shuf18505 in full — optimizer state is not checkpointed, "
        "so a fresh 10-step cell is the smallest completion). G_FIDELITY "
        "gates the lineage: the re-run's checkpoint deltas vs the "
        "surviving s{1,2,4} files (bit-identity expected; 5e-6 fallback "
        "flagged) plus every survivor's embedded meta vs the registered "
        "cell spec. Eval-only cells' per-step training CE and "
        "intra-interval displacement increments are NOT recoverable from "
        "checkpoints (null in traj; checkpoint cum_disp exact); their x/y "
        "streams are reconstructed by exact generator replay (a pure "
        "function of seeds 10902/<noise_seed>, gated by G_INPUTS/G_TARGETS) "
        "and the re-run cell carries the full live telemetry as the "
        "machinery reference. The registered bars and operationalizations "
        "above are untouched; survivors are loaded, never overwritten.")


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/e185_noise_wash.py VERBATIM (= the e176n lineage; see the
# module docstrings). Copied rather than imported to own the device policy.

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
    e176n/e178/e184/e185 copy): p(true name char) at positions
    addr_row..addr_row+6."""
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


# ------------------------------------------------------------------ the wash

def flat_params(net: TinyGPT) -> torch.Tensor:
    """The fp32 flat parameter vector (all trainable tensors, net.parameters()
    order — the optimizer's own currency; 2,739,072 elements on this line)."""
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()]).clone()


def noise_wash(tag: str, net0: TinyGPT, anchor: torch.Tensor,
               train_ids: torch.Tensor, itos, r_eval_xy, gm12_ids,
               g0_ids, zid: int, target_mode: str, noise_seed: int,
               ckpt_steps: tuple[int, ...]):
    """E185's noise_wash VERBATIM ARITHMETIC (e176n's finetune_freeze with
    target substitution + displacement bookkeeping). Per step: aj = randint(16)
    anchor draws, rj = randint(16) random corpus offsets (the seed-10902
    generator — BIT-IDENTICAL across all cells and vs e185); THEN the target
    substitution from the dedicated noise generator: 'iid' = y ~ uniform(vocab);
    'perm' = flat randperm of the true targets. Batch 32 full-token CE; AdamW
    (0.9,0.95) wd 0.1 constant lr, clip 1.0. Displacement: cumulative
    ||theta_t - theta_0||_2 + per-step ||theta_t - theta_{t-1}||_2 measured
    every step; checkpoint delta vectors kept (CPU) for the cosine co-report.
    Snapshots + light CPU evals (g-12, g0, CE_R — no RNG consumed) at the
    checkpoint steps; in-batch CE recorded at EVERY step."""
    assert target_mode in ("true", "iid", "perm")
    ckpt_set = set(ckpt_steps)
    n_steps = ckpt_steps[-1]
    net = copy.deepcopy(net0)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(FREEZE_SEED)
    ngen = (torch.Generator().manual_seed(noise_seed)
            if target_mode != "true" else None)
    vocab = len(itos)
    n_anc = anchor.shape[0]
    traj, sds, t_start = [], {}, time.time()
    step = 0
    zeph_checks = 0
    evl = copy.deepcopy(net0)          # CPU eval twin
    theta0 = flat_params(net)           # the displacement origin
    prev = theta0.clone()
    deltas: dict[int, torch.Tensor] = {}   # checkpoint delta vectors
    x_hashes: dict[int, str] = {}
    y_stats: list[dict] = []
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
        y_true = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
        # ---- THE TARGET SUBSTITUTION (the cell's only delta; dedicated RNG,
        # drawn AFTER the aj/rj draws so the input stream is untouched)
        if target_mode == "true":
            y = y_true
        elif target_mode == "iid":
            y = torch.randint(vocab, y_true.shape, generator=ngen)
        else:  # perm
            perm = torch.randperm(y_true.numel(), generator=ngen)
            y = y_true.reshape(-1)[perm].reshape(y_true.shape)
        x_hashes[step] = hashlib.md5(
            x.contiguous().numpy().tobytes()).hexdigest()
        y_stats.append({
            "step": step,
            "frac_targets_changed": float((y != y_true).float().mean()),
            "multiset_equals_true": bool(
                torch.equal(torch.sort(y.reshape(-1))[0],
                            torch.sort(y_true.reshape(-1))[0])),
        })
        logits, _ = net(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        cur = flat_params(net)
        cum_disp = float(torch.norm(cur - theta0))
        inc_disp = float(torch.norm(cur - prev))
        prev = cur
        if step in ckpt_set:
            deltas[step] = cur - theta0
        traj.append({"step": step, "ce_batch": float(loss.item()),
                     "cum_disp": cum_disp, "step_disp": inc_disp,
                     "elapsed_s": round(time.time() - t_start, 1)})
        log(f"  [{tag}] s{step:4d} CE {float(loss.item()):.4f} "
            f"|d| {cum_disp:.4f} (step |d| {inc_disp:.4f})")
        if step in ckpt_set:
            sd_cpu = {k: v.detach().cpu().clone()
                      for k, v in net.state_dict().items()}
            sds[step] = sd_cpu
            evl.load_state_dict(sd_cpu)
            evl.eval()
            gz = battery_cell(evl, gm12_ids, zid)
            gz0 = battery_cell(evl, g0_ids, zid)
            ce_r = ce_fixed_cpu(evl, *r_eval_xy)
            traj[-1].update({"g_m12_mean_pz": gz["mean_pz"],
                             "g0_mean_pz": gz0["mean_pz"],
                             "frac_argmax_z": gz["frac_argmax_z"],
                             "ce_r": ce_r})
            log(f"  [{tag}] CKPT +{step:4d} g-12 {gz['mean_pz']:.4f} "
                f"g0 {gz0['mean_pz']:.4f} CE_R {ce_r:.4f}")
        if (time.time() - t_start) > FT_TIME_CAP:
            log(f"  [{tag}] time cap {FT_TIME_CAP:.0f}s at s{step}")
            trims.append(f"{tag}: time cap at step {step}")
            break
    net.eval()
    return {"sds": sds, "traj": traj, "steps_ran": step,
            "seed": FREEZE_SEED, "lr": LR, "target_mode": target_mode,
            "noise_seed": noise_seed if ngen is not None else None,
            "zeph_violations": zeph_checks, "x_hashes": x_hashes,
            "y_stats": y_stats, "deltas": deltas,
            "theta0_norm": float(torch.norm(theta0))}


CKPT_INVENTORY: dict = {}


# ------------------------------------------------- recovery helpers (2026-09-29)
# Exact reconstruction of the per-step data stream WITHOUT training: the
# aj/rj draws are a pure function of the seed-10902 generator (drawn before
# any noise draw) and the target substitution a pure function of the
# dedicated noise generator — so x_hashes/y_stats/zeph are bit-recoverable
# for eval-only cells (gated vs e185's stored md5s by G_INPUTS below).

def replay_stream(target_mode: str, noise_seed: int, anchor: torch.Tensor,
                  train_ids: torch.Tensor, itos) -> dict:
    """noise_wash's data construction replayed step-for-step (no model, no
    training): identical draw order, identical tensors, identical md5."""
    assert target_mode in ("true", "iid", "perm")
    gen = torch.Generator().manual_seed(FREEZE_SEED)
    ngen = (torch.Generator().manual_seed(noise_seed)
            if target_mode != "true" else None)
    vocab = len(itos)
    n_anc = anchor.shape[0]
    x_hashes: dict[int, str] = {}
    y_stats: list[dict] = []
    zeph_checks = 0
    for step in range(1, N_STEPS + 1):
        aj = torch.randint(n_anc, (ANCH_BS,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                           generator=gen)
        anc = anchor[aj]
        rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj])
        for w in rnd:
            txt = "".join(itos[int(c)] for c in w[:64]) + \
                  "".join(itos[int(c)] for c in w[192:])
            if "ZEPH" in txt:
                zeph_checks += 1
        x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
        y_true = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
        if target_mode == "true":
            y = y_true
        elif target_mode == "iid":
            y = torch.randint(vocab, y_true.shape, generator=ngen)
        else:  # perm
            perm = torch.randperm(y_true.numel(), generator=ngen)
            y = y_true.reshape(-1)[perm].reshape(y_true.shape)
        x_hashes[step] = hashlib.md5(
            x.contiguous().numpy().tobytes()).hexdigest()
        y_stats.append({
            "step": step,
            "frac_targets_changed": float((y != y_true).float().mean()),
            "multiset_equals_true": bool(
                torch.equal(torch.sort(y.reshape(-1))[0],
                            torch.sort(y_true.reshape(-1))[0])),
        })
    return {"x_hashes": x_hashes, "y_stats": y_stats,
            "zeph_violations": zeph_checks}


def load_survivors(tag: str, mode: str, nseed: int) -> tuple[dict, dict]:
    """Load the previous run's surviving checkpoints for a cell and verify
    each file's embedded meta against the registered cell spec (experiment,
    arm, noise seed, input seed, lr, steps). Returns (checkpoints-by-step,
    meta-table). Survivors are NEVER overwritten by the recovery."""
    pref = "smoke_" if SMOKE else ""
    ckpts: dict[int, dict] = {}
    meta_tbl: dict[int, dict] = {}
    for s in CK_MAIN:
        p = CKPT_DIR / f"{pref}e187_{tag}_s{s}.pt"
        if not p.exists():
            continue
        st = torch.load(p, map_location="cpu", weights_only=False)
        mt = st.get("meta", {}) if isinstance(st, dict) else {}
        want = {"experiment": "e187", "target_mode": mode,
                "noise_seed": nseed, "input_seed": FREEZE_SEED,
                "lr": LR, "steps": int(s)}
        meta_tbl[s] = {
            "path": str(p.relative_to(E43.REPO)).replace("\\", "/"),
            "mtime": time.strftime("%Y-%m-%dT%H:%M:%S",
                                   time.localtime(p.stat().st_mtime)),
            "meta_match": {k: {"stored": mt.get(k), "expected": v}
                           for k, v in want.items()},
            "pass": bool(all(mt.get(k) == v for k, v in want.items())),
        }
        ckpts[s] = st
    return ckpts, meta_tbl


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "e187", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name} "
        f"({', '.join(f'{k}={v}' for k, v in meta.items())})")


def verify_e185(path: Path) -> dict:
    """Verify the embedded e185 copies against the stored metrics file
    (no silent divergence; e185's verify convention)."""
    src = {"source": "embedded verbatim copy (runs/e185 metrics)",
           "file_present": path.exists(), "verified_vs_embedded": None,
           "max_abs_diff": None, "checks": {}}
    if path.exists():
        mm = json.loads(path.read_text(encoding="utf-8"))
        diffs = []
        # full-dial g-12 traces
        for tag in ("ctrl", "noise", "shuf"):
            rows = mm["traces"][tag]
            got = [r["gm12"] for r in rows]
            want = E185_GM12[tag]
            src["checks"][f"trace_gm12_{tag}"] = bool(
                [r["freeze_steps"] for r in rows] == E185_TRACE_STEPS)
            diffs += [abs(a - b) for a, b in zip(got, want)]
        # displacement + cosines at the checkpoints
        for tag in ("ctrl", "noise", "shuf"):
            for r in mm["displacement"]["table"][tag]:
                s = r["step"]
                if s in E185_DISP[tag]:
                    diffs.append(abs(r["cum_disp"] - E185_DISP[tag][s]))
                if "cos_vs_ctrl" in r and s in E185_COS[tag]:
                    diffs.append(abs(r["cos_vs_ctrl"] - E185_COS[tag][s]))
        # per-step input hashes (the ctrl's == every arm's; G_INPUTS stored)
        for s_str, rec in mm["gates"]["G_INPUTS"]["per_step"].items():
            s = int(s_str)
            diffs.append(0.0 if rec["ctrl"] == E185_XHASH[s] else 1.0)
        # y_stats means
        for tag in ("noise", "shuf"):
            ym = mm["gates"]["G_TARGETS"]["per_arm"][tag][
                "frac_targets_changed_mean"]
            diffs.append(abs(ym - E185_YSTATS[tag]))
        # the stored adjudication anchors
        adj = mm["adjudication"]["conditions"]["CONTROL_KILL"]
        diffs.append(0.0 if adj["t_kill"] == T_KILL_E185 else 1.0)
        diffs.append(abs(adj["D_kill"] - D_KILL_STORED))
        diffs.append(abs(adj["ctrl_gm12"][str(T_KILL_E185)]
                         - CTRL_GM12_AT_TKILL))
        src["max_abs_diff"] = max(diffs)
        src["verified_vs_embedded"] = bool(max(diffs) < 1e-9)
        if src["verified_vs_embedded"]:
            src["source"] = ("runs/e185/metrics.json (embedded copy verified, "
                             f"max|diff| {max(diffs):.1e})")
    return src


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e187_smoke" if SMOKE else "e187")
    log(f"E187 THE NOISE REPLICATES (smoke={SMOKE}, recovery={RECOVERY}) "
        f"-> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()}), stagger "
        f"{STAGGER_S:.0f}s, cooldown {COOLDOWN_S:.0f}s around each training")
    time.sleep(STAGGER_S)            # launch stagger vs the CPU fleet

    # ---------------- protocol rebuild (e143/e151/e152/e161/e176/e176n/e185)
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
    # FIXED content (seed 170) — the cells differ ONLY in each step's targets.
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
                             "e170's construction VERBATIM (= e176n arm A's / "
                             "e185's stream; FIXED content, not reseeded)"),
            "n_windows": 16, "block": BLOCK, "seed": E170_ANCHOR_SEED,
            "starts": n_starts, "tries": tries, "rejections": rejections,
            "forbidden": list(ANCHOR_FORBIDDEN),
            "windows_with_host_content": sum(
                1 for s in n_starts
                if any(f in train_text[s: s + BLOCK + 1] for f in HOSTS)),
            "junctions_covered": jc_neutral,
        },
        "budget_identical_to_e185": bool(anchor_neutral.shape == (16, BLOCK)),
        "rng_note": ("noise_wash verbatim; the seed-10902 aj/rj draw sequence "
                     "is IDENTICAL across the four new cells AND vs e185's "
                     "stored arms (same shapes/moduli, drawn BEFORE any noise "
                     "draw) — the input stream is bit-identical (gated by "
                     "per-step md5 vs e185's stored hashes); ONLY each step's "
                     "target tensor differs (iid / perm at 4 new seeds)"),
        "random_channel_background": {
            "host_occurrences_in_train": host_occ_total,
            "est_window_hit_rate": bg_rate,
            "note": ("the 16-per-batch random corpus windows are e161/e176/"
                     "e176n/e184/e185 VERBATIM and unfiltered — identical "
                     "background (~3.8%/window) in all four cells; not part "
                     "of the delta"),
        },
    }
    G_ANCHOR["pass"] = bool(
        G_ANCHOR["neutral_bank"]["windows_with_host_content"] == 0
        and G_ANCHOR["neutral_bank"]["junctions_covered"] == 0
        and G_ANCHOR["budget_identical_to_e185"])
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
        """e185's measure() VERBATIM: base 3-geos + held30 + CE_R + site
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
    src_e185 = verify_e185(E185_METRICS)
    log(f"e185 reference: {src_e185['source']}")
    if not src_e185["verified_vs_embedded"]:
        raise RuntimeError("embedded e185 references diverged from the "
                           "stored metrics file — the overlay/reference is "
                           "untrustworthy")

    # =====================================================================
    # THE CORPUS DIRECTION + D_kill (re-derived from e185's stored ctrl
    # checkpoints on THIS device; gated vs the stored displacement table)
    # =====================================================================
    theta_root = flat_params(net0)
    e185_deltas: dict[str, dict[int, torch.Tensor]] = {}
    for tag in ("ctrl", "noise", "shuf"):
        e185_deltas[tag] = {}
        for s in CK_MAIN:
            m = load_cpu(CKPT_DIR / f"e185_{tag}_s{s}.pt")
            e185_deltas[tag][s] = flat_params(m) - theta_root
            del m
    G_E185CKPTS = {"per_tag": {}, "pass": None}
    for tag in ("ctrl", "noise", "shuf"):
        disp_diffs, cos_diffs = [], []
        for s in CK_MAIN:
            d_norm = float(torch.norm(e185_deltas[tag][s]))
            disp_diffs.append(abs(d_norm - E185_DISP[tag][s]))
            if tag != "ctrl":
                cos = float(torch.dot(e185_deltas[tag][s],
                                      e185_deltas["ctrl"][s])
                            / (torch.norm(e185_deltas[tag][s])
                               * torch.norm(e185_deltas["ctrl"][s]) + 1e-30))
                cos_diffs.append(abs(cos - E185_COS[tag][s]))
        G_E185CKPTS["per_tag"][tag] = {
            "max_disp_diff": max(disp_diffs),
            "max_cos_diff": (max(cos_diffs) if cos_diffs else None),
        }
    G_E185CKPTS["tol"] = G_BIT_TOL
    G_E185CKPTS["pass"] = bool(
        all(v["max_disp_diff"] < G_BIT_TOL
            for v in G_E185CKPTS["per_tag"].values())
        and all((v["max_cos_diff"] is not None
                 and v["max_cos_diff"] < G_BIT_TOL)
                or v["max_cos_diff"] is None
                for v in G_E185CKPTS["per_tag"].values()))
    D_KILL = float(torch.norm(e185_deltas["ctrl"][T_KILL_E185]))
    G_E185CKPTS["D_kill_rederived"] = D_KILL
    G_E185CKPTS["D_kill_stored"] = D_KILL_STORED
    G_E185CKPTS["D_kill_diff"] = abs(D_KILL - D_KILL_STORED)
    if not G_E185CKPTS["pass"] or G_E185CKPTS["D_kill_diff"] > G_BIT_TOL:
        raise RuntimeError(f"e185 stored checkpoints failed to reproduce the "
                           f"stored displacement/cosine table: "
                           f"{G_E185CKPTS}")
    log(f"G_E185CKPTS: e185's stored ctrl/noise/shuf checkpoints reproduce "
        f"the stored displacement table AND cosines on this device "
        f"(max|diff| {G_E185CKPTS['per_tag']['ctrl']['max_disp_diff']:.1e}); "
        f"D_kill re-derived {D_KILL:.10f} vs stored {D_KILL_STORED:.10f}: "
        f"PASS")

    # =====================================================================
    # THE FOUR CELLS (one training each; cooldown around each)
    # =====================================================================
    ARM_SPECS = []
    for base, mode in (("noise", "iid"), ("shuf", "perm")):
        for ns in NOISE_SEEDS[base]:
            ARM_SPECS.append((
                f"{base}{ns}", mode, ns,
                f"{'NOISE-LABELS' if mode == 'iid' else 'SHUFFLED-TARGET'} "
                f"seed {ns} — e185's arm VERBATIM at a fresh noise draw ("
                + ("y ~ i.i.d. uniform over the 65-token vocab; destroys "
                   "mapping AND target statistics" if mode == "iid" else
                   "y = a flat random permutation of the true targets; "
                   "preserves the exact target multiset, destroys only the "
                   "mapping") + ")"))
    arms: dict = {}
    batteries_all: dict = {}
    recovery_cells: dict = {}
    survivors_meta: dict = {}

    def eval_from_survivors(tag: str, mode: str, nseed: int,
                            surv: dict, theta_root_v: torch.Tensor) -> dict:
        """EVAL-ONLY cell (dispatch policy: no retraining where the grid
        survives): the dial is measured from the surviving checkpoints.
        Deltas/displacement recomputed on this device; the per-step stream
        (input md5s, target stats, name-free check) reconstructed EXACTLY by
        generator replay; per-step training CE and intra-interval increments
        are NOT recoverable from checkpoints (null in traj — the re-run cell
        carries the live telemetry reference)."""
        sds = {s: surv[s]["model"] for s in sorted(surv)}
        rep = replay_stream(mode, nseed, anchor_neutral, train_ids, itos)
        deltas: dict[int, torch.Tensor] = {}
        traj: list[dict] = []
        prev: torch.Tensor | None = None
        for s in sorted(sds):
            d = flat_params(evl_load(sds[s])) - theta_root_v
            deltas[s] = d
            traj.append({"step": s, "ce_batch": None,
                         "cum_disp": float(torch.norm(d)),
                         "step_disp": (float(torch.norm(d - prev))
                                       if prev is not None else None),
                         "elapsed_s": None,
                         "recovered_from": ("checkpoint-eval (per-step CE / "
                                            "increments not recoverable; "
                                            "cum_disp exact)")})
            prev = d
        return {"sds": sds, "traj": traj, "steps_ran": max(sds),
                "seed": FREEZE_SEED, "lr": LR, "target_mode": mode,
                "noise_seed": nseed,
                "zeph_violations": rep["zeph_violations"],
                "x_hashes": rep["x_hashes"], "y_stats": rep["y_stats"],
                "deltas": deltas,
                "theta0_norm": float(torch.norm(theta_root_v)),
                "eval_only": True}

    for tag, mode, nseed, desc in ARM_SPECS:
        log("=" * 78)
        surv, surv_meta = ({}, {}) if not RECOVERY else load_survivors(
            tag, mode, nseed)
        survivors_meta[tag] = surv_meta
        rerun = bool(RECOVERY and set(surv) != set(CK_MAIN))
        recovery_cells[tag] = {
            "mode": ("live run" if not RECOVERY else
                     ("re-run (survivor grid incomplete: "
                      f"{sorted(surv)} vs {list(CK_MAIN)})" if rerun else
                      "eval-only from survivors")),
            "survivor_steps": sorted(surv),
        }
        if rerun:
            trims.append(
                f"{tag}: previous process's cell incomplete (2026-09-28 "
                "outage; runs/e187_run.log records 'time cap 180s at s9') — "
                "re-run in full this process (the minimal completion: "
                "optimizer state is not checkpointed)")
        if not SMOKE:
            log(f"[thermal] cooldown {COOLDOWN_S:.0f}s before {tag}")
            cooldown(COOLDOWN_S)
        log(f"CELL {tag.upper()} — {desc}; {N_STEPS} steps, checkpoints "
            f"+{list(CK_MAIN)}"
            + (f" [{recovery_cells[tag]['mode']}]" if RECOVERY else ""))
        if rerun or not RECOVERY:
            arm = noise_wash(tag, net0, anchor_neutral, train_ids, itos,
                             r_eval_xy, gm12_ids, g0_ids, zid, mode,
                             nseed, CK_MAIN)
        else:
            arm = eval_from_survivors(tag, mode, nseed, surv, theta_root)
        if not SMOKE:
            log(f"[thermal] cooldown {COOLDOWN_S:.0f}s after {tag}")
            cooldown(COOLDOWN_S)
        G_DRAWFREE = {f"zeph_violations_{tag}": arm["zeph_violations"],
                      "pass": bool(arm["zeph_violations"] == 0)}
        assert G_DRAWFREE["pass"], f"{tag}: name token leaked into a window"
        gates_surg[f"G_DRAWFREE_{tag}"] = G_DRAWFREE
        arms[tag] = arm

        pref = "smoke_" if SMOKE else ""
        for s in sorted(arm["sds"]):
            name = f"{pref}e187_{tag}_s{s}"
            if RECOVERY and (CKPT_DIR / f"{name}.pt").exists():
                CKPT_INVENTORY[name] = {
                    "path": f"runs/checkpoints/{name}.pt",
                    "provenance": ("survivor of the 2026-09-28 outage "
                                   "(loaded for eval, NOT overwritten)"),
                    "steps": int(s), "target_mode": mode,
                    "noise_seed": nseed, "input_seed": FREEZE_SEED,
                    "lr": LR,
                }
                log(f"[ckpt] survivor kept intact: {name}.pt")
                continue
            save_ckpt(f"e187_{tag}_s{s}",
                      arm["sds"][s],
                      {"desc": f"e131_consolidated_e113 + {s}-step "
                               f"{mode.upper()}-target neutral freeze "
                               f"(target_mode={mode}), lr {LR}, input seed "
                               f"{FREEZE_SEED}, noise seed {nseed}",
                       "steps": int(s), "target_mode": mode,
                       "input_seed": FREEZE_SEED, "noise_seed": nseed,
                       "lr": LR,
                       "anchor_bank": f"neutral (seed {E170_ANCHOR_SEED})",
                       "base": f"runs/checkpoints/{ROOT_CK}"})

        batteries = {"root": root}
        for s in sorted(arm["sds"]):
            log(f"{tag} +{s} battery")
            batteries[str(s)] = measure(arm["sds"][s], f"{tag}{s}",
                                        lean=SMOKE)
        batteries_all[tag] = batteries

    # ---- G_FIDELITY (recovery lineage gate; no-op PASS when not RECOVERY)
    G_FIDELITY: dict = {"cells": {}, "tol": G_BIT_TOL, "pass": None,
                        "applicable": bool(RECOVERY)}
    for tag, mode, nseed, _ in ARM_SPECS:
        if not RECOVERY:
            break
        entry = {"mode": recovery_cells[tag]["mode"],
                 "survivor_meta": survivors_meta[tag],
                 "survivor_meta_all_match": bool(
                     (len(survivors_meta[tag]) == 0) or
                     all(m["pass"] for m in survivors_meta[tag].values()))}
        if recovery_cells[tag]["mode"].startswith("re-run"):
            pref = "smoke_" if SMOKE else ""
            per = {}
            for s in CK_MAIN:
                p = CKPT_DIR / f"{pref}e187_{tag}_s{s}.pt"
                if not p.exists():
                    continue   # was missing; the re-run just saved it
                surv_sd = torch.load(p, map_location="cpu",
                                     weights_only=False)["model"]
                d_old = flat_params(evl_load(surv_sd)) - theta_root
                d_new = arms[tag]["deltas"][s]
                per[s] = {"bit_identical": bool(torch.equal(d_new, d_old)),
                          "max_abs_param_diff": float(
                              (d_new - d_old).abs().max())}
            entry["rerun_vs_survivor_deltas"] = per
            if per:
                entry["bit"] = bool(all(v["bit_identical"]
                                        for v in per.values()))
                entry["max_abs_param_diff"] = max(
                    v["max_abs_param_diff"] for v in per.values())
        G_FIDELITY["cells"][tag] = entry
    if RECOVERY:
        G_FIDELITY["pass"] = bool(
            all(c.get("survivor_meta_all_match", False)
                and c.get("bit", True)
                and ((c.get("max_abs_param_diff") or 0.0) < G_BIT_TOL)
                for c in G_FIDELITY["cells"].values()))
        if not G_FIDELITY["pass"]:
            raise RuntimeError(f"recovery fidelity gate FAILED "
                               f"(re-run vs survivors / meta): {G_FIDELITY}")
        rerun_tags = [t for t, c in G_FIDELITY["cells"].items()
                      if c["mode"].startswith("re-run")]
        log(f"G_FIDELITY: survivor meta verified for all cells; "
            + (f"re-run {'/'.join(rerun_tags)} reproduces its surviving "
               f"checkpoint deltas (max|diff| "
               f"{max((c.get('max_abs_param_diff') or 0.0)
                      for c in G_FIDELITY['cells'].values()):.1e}): PASS"
               if rerun_tags else "no cell required a re-run: PASS"))
    else:
        G_FIDELITY["pass"] = True

    # =====================================================================
    # CONSTRUCTION-FIDELITY GATES (inputs bit-identical incl. vs e185)
    # =====================================================================
    G_INPUTS = {"per_step": {}, "vs_e185_stored": {}, "pass": None}
    tags = [t for t, _, _, _ in ARM_SPECS]
    for step in range(1, N_STEPS + 1):
        hs = {t: arms[t]["x_hashes"].get(step) for t in tags}
        same = bool(all(h is not None for h in hs.values())
                    and len(set(hs.values())) == 1)
        vs185 = bool(hs[tags[0]] == E185_XHASH.get(step))
        G_INPUTS["per_step"][step] = {**hs, "identical": same}
        G_INPUTS["vs_e185_stored"][step] = {
            "e185_stored": E185_XHASH.get(step), "match": vs185}
    G_INPUTS["pass"] = bool(
        all(v["identical"] for v in G_INPUTS["per_step"].values())
        and all(v["match"] for v in G_INPUTS["vs_e185_stored"].values()))
    assert G_INPUTS["pass"], ("input streams diverged across cells (or vs "
                              "e185's stored hashes) — the replicate is not "
                              "at matched inputs")
    log(f"G_INPUTS: per-step input batches bit-identical across all four "
        f"cells AND vs e185's stored hashes ({N_STEPS}/{N_STEPS}): PASS")

    G_TARGETS = {"per_cell": {}}
    for tag, mode, nseed, _ in ARM_SPECS:
        ys = arms[tag]["y_stats"]
        G_TARGETS["per_cell"][tag] = {
            "target_mode": mode,
            "frac_targets_changed_mean": float(
                np.mean([r["frac_targets_changed"] for r in ys])),
            "multiset_equals_true_all_steps": bool(
                all(r["multiset_equals_true"] for r in ys)),
        }
    for tag, mode, nseed, _ in ARM_SPECS:
        if mode == "iid":
            G_TARGETS["per_cell"][tag]["expectation"] = (
                "iid labels change ~64/65 of positions; multiset NOT "
                "preserved (e185's 18501 changed "
                f"{E185_YSTATS['noise']:.3f})")
        else:
            G_TARGETS["per_cell"][tag]["expectation"] = (
                "permutation changes ~63/65 positions; multiset preserved "
                f"at EVERY step (e185's 18502 changed "
                f"{E185_YSTATS['shuf']:.3f})")
    G_TARGETS["pass"] = bool(all(
        c["frac_targets_changed_mean"] > 0.9
        and (not c["multiset_equals_true_all_steps"]
             if c["target_mode"] == "iid"
             else c["multiset_equals_true_all_steps"])
        for c in G_TARGETS["per_cell"].values()))
    assert G_TARGETS["pass"], f"target gate FAILED: {G_TARGETS}"
    log("G_TARGETS: all four cells' targets constructed as registered "
        "(iid multiset-broken / perm multiset-preserved, changed > 0.9): "
        "PASS")

    # =====================================================================
    # DISPLACEMENT TABLE + COSINES vs THE CORPUS DIRECTION (measured)
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

    disp_table = {}
    for tag in tags:
        rows = []
        for t in arms[tag]["traj"]:
            s = t["step"]
            row = {"step": s, "ce_batch": t["ce_batch"],
                   "cum_disp": t["cum_disp"], "step_disp": t["step_disp"]}
            if "g_m12_mean_pz" in t:
                row["g_m12_light"] = t["g_m12_mean_pz"]
                row["ce_r_light"] = t["ce_r"]
            if s in arms[tag]["deltas"] and s in e185_deltas["ctrl"]:
                d_a = arms[tag]["deltas"][s]
                d_c = e185_deltas["ctrl"][s]
                cos = float(torch.dot(d_a, d_c)
                            / (torch.norm(d_a) * torch.norm(d_c) + 1e-30))
                row["cos_vs_corpus"] = cos
            rows.append(row)
        disp_table[tag] = rows

    # =====================================================================
    # ADJUDICATION (registered clauses; no shopping)
    # =====================================================================
    trace = {tag: trace_from(batteries_all[tag], [0] + sorted(arms[tag]["sds"]))
             for tag in tags}
    cells = {}
    for tag, mode, nseed, _ in ARM_SPECS:
        gm = {r["freeze_steps"]: r["gm12"] for r in trace[tag]}
        disp = {r["step"]: r["cum_disp"] for r in disp_table[tag]}
        M = next((s for s in CK_MAIN if disp.get(s, 0.0) >= D_KILL), None)
        matched_ckpts = [s for s in CK_MAIN
                         if disp.get(s, 0.0) >= D_KILL and s in gm]
        kill_by2 = bool(gm.get(2, 1.0) <= SHUT_BAR)
        kills_at_M = bool(M is not None and gm.get(M, 1.0) <= SHUT_BAR)
        spares = bool(matched_ckpts
                      and any(gm.get(s, 0.0) >= SURVIVE_BAR
                              for s in matched_ckpts))
        below_match_kill = next(
            (s for s in CK_MAIN
             if disp.get(s, float("inf")) < D_KILL
             and gm.get(s, 1.0) <= SHUT_BAR), None)
        cells[tag] = {
            "target_mode": mode, "noise_seed": nseed,
            "M": M, "matched_ckpts": matched_ckpts, "D_kill": D_KILL,
            "gm12_at_1": gm.get(1), "gm12_at_2": gm.get(2),
            "gm12_at_M": gm.get(M) if M is not None else None,
            "matched_gm12": {s: gm.get(s) for s in matched_ckpts},
            "max_matched_gm12": (max(gm.get(s, 0.0) for s in matched_ckpts)
                                 if matched_ckpts else None),
            "kill_by2": kill_by2, "kills_at_M": kills_at_M,
            "spares": spares, "below_match_kill": below_match_kill,
        }

    any_spares = any(c["spares"] for c in cells.values())
    all_kill_by2 = all(c["kill_by2"] for c in cells.values())
    all_kills_at_M = all(c["kills_at_M"] for c in cells.values())

    if any_spares:
        spared = [f"{t} (max matched g-12 {c['max_matched_gm12']:.4f})"
                  for t, c in cells.items() if c["spares"]]
        verdict = "ANY-SPARES"
        clause = ("at least one arm/seed combination left the fact >= "
                  f"{SURVIVE_BAR} at a matched checkpoint — " + "; ".join(spared)
                  + f" (D_kill {D_KILL:.4f}; the corpus control was dead at "
                  f"+{T_KILL_E185} with {CTRL_GM12_AT_TKILL:.4f}) — the "
                  "mechanism RE-INVERTS toward corpus-directed; the noun "
                  "RE-BOUNDS; the no-basin paragraph is withdrawn pending "
                  "re-adjudication.")
    elif all_kill_by2:
        seq = "; ".join(f"{t} g-12@+2 {c['gm12_at_2']:.2e}"
                        + (f" (@+1 {c['gm12_at_1']:.2e})"
                           if c["gm12_at_1"] is not None else "")
                        for t, c in cells.items())
        verdict = "NOISE-KILLS-REPLICATES"
        clause = ("ALL FOUR replicate cells killed by +2 — " + seq
                  + f"; e185's displacement-match formality co-adjudicates "
                  f"{'KILLS at M for every cell' if all_kills_at_M else 'mixed (see cells)'}"
                  + f" (D_kill {D_KILL:.4f}) — THE MECHANISM IS FORMALLY "
                  "LICENSED: the no-basin noun STANDS (n=3 noise draws per "
                  "arm, all content-free kills; the cosines vs the corpus "
                  "direction report whether the orthogonal-direction texture "
                  "replicates too).")
    else:
        bits = []
        for t, c in cells.items():
            g2 = c["gm12_at_2"]
            bits.append(f"{t}: g-12@+2 "
                        + (f"{g2:.4f}" if g2 is not None else "n/a")
                        + (" KILL" if c["kill_by2"] else " no-kill")
                        + (f" (max matched {c['max_matched_gm12']:.4f})"
                           if c["max_matched_gm12"] is not None else ""))
        verdict = "TEXTURE"
        clause = ("partial damage — neither bar fired cleanly; " + "; ".join(bits)
                  + f"; D_kill {D_KILL:.4f}; full dial trajectories "
                  "reported, no bar shopping.")

    log("=" * 78)
    log(f"E187 VERDICT: {verdict}")
    for tag in tags:
        seq = " -> ".join(f"+{r['freeze_steps']}:{r['gm12']:.4f}"
                          for r in trace[tag])
        disp = " -> ".join(f"+{r['step']}:{r['cum_disp']:.4f}"
                           for r in disp_table[tag])
        cos = " -> ".join(f"+{r['step']}:{r['cos_vs_corpus']:+.3f}"
                          for r in disp_table[tag] if "cos_vs_corpus" in r)
        log(f"  {tag}: g-12 {seq}")
        log(f"  {tag}: |d(theta)| {disp}")
        log(f"  {tag}: cos vs corpus {cos}")
    log(f"  D_kill={D_KILL:.4f} (re-derived, gated); cells: "
        + "; ".join(f"{t}: kill_by2={c['kill_by2']}, spares={c['spares']}"
                    for t, c in cells.items()))
    log(f"  {clause}")
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e187_noise_replicates",
        "date": common.now_iso(),
        "registration": ("NO QUEUE ROW — the dispatch's registration IS the "
                         "registration (the bars quoted verbatim in the "
                         "module docstring, frozen before compute)"),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("does e185's content-free kill REPLICATE at fresh noise "
                     "draws? Both arms (labels-iid, shuffled-target) at 2 "
                     "additional noise seeds each, same input stream (10902, "
                     "bit-identical to e185's — gated), same ruler. A spare "
                     "anywhere re-inverts the mechanism toward "
                     "corpus-directed; four kills formally license the "
                     "no-basin noun."),
        "root": f"runs/checkpoints/{ROOT_CK} (gated vs e151 before-cells, "
                f"max|diff| {G_ROOT['max_abs_diff']:.2e})",
        "root_meta": root_meta,
        "arms": {
            tag: {
                "desc": desc,
                "target_mode": mode,
                "noise_seed": nseed,
                "orig_e185_seed": E185_ORIG_SEEDS["noise" if mode == "iid"
                                                 else "shuf"],
                "ckpt_steps": list(CK_MAIN),
                "steps_ran": arms[tag]["steps_ran"],
                "seed": FREEZE_SEED, "lr": LR,
                "traj": arms[tag]["traj"],
                "zeph_violations": arms[tag]["zeph_violations"],
                "y_stats": arms[tag]["y_stats"],
                "missing_checkpoints": [s for s in CK_MAIN
                                        if s not in arms[tag]["sds"]],
                "provenance": recovery_cells[tag],
                "eval_only": bool(arms[tag].get("eval_only", False)),
            } for (tag, mode, nseed, desc) in ARM_SPECS
        },
        "recovery": {
            "event": ("the 2026-09-28 system outage killed the previous "
                      "process after the trainings but before any output "
                      "(survivors: lab/e187_noise_replicates.py verbatim, "
                      "15/16 checkpoints under runs/checkpoints/e187_*.pt, "
                      "runs/e187_run.log; runs/e187 was empty). From the "
                      "log: shuf18505's training hit the registered 180 s "
                      "CPU cap at step 9 (outage onset) so its s10 "
                      "checkpoint is missing, and the process then died at "
                      "the G_INPUTS assert — on the capped cell's MISSING "
                      "step-10 hash, not on a divergence (the reconstructed "
                      "seed-10902 stream matches e185's stored per-step "
                      "md5s bit-exactly; re-gated live by G_INPUTS)"),
            "mode": ("E187_RECOVERY=1: eval-only from survivors where the "
                     "grid is complete + the minimal re-run (shuf18505, in "
                     "full — optimizer state is not checkpointed) where it "
                     "is not" if RECOVERY else
                     "single live process; no recovery applied"),
            "cells": recovery_cells,
            "stream_reconstruction": ("eval-only cells' x_hashes/y_stats/"
                                      "zeph reconstructed by exact generator "
                                      "replay — a pure function of seeds "
                                      "10902/<noise_seed> — gated "
                                      "bit-identical to e185's stored "
                                      "per-step md5s by G_INPUTS and to the "
                                      "e185 target statistics by G_TARGETS; "
                                      "the re-run cell's LIVE stream "
                                      "additionally anchors the replay "
                                      "path"),
            "not_recoverable_from_checkpoints": ("per-step training CE and "
                                                 "per-step displacement "
                                                 "increments inside "
                                                 "checkpoint intervals "
                                                 "(null in eval-only traj "
                                                 "rows); checkpoint cum_disp "
                                                 "is exact; the re-run cell "
                                                 "carries full live "
                                                 "telemetry"),
        },
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
                     "geometries_measured": {"novel": [-12, 12],
                                             "trained_g0": 0},
                     "neutral_bank": G_ANCHOR["neutral_bank"],
                     "measure_dial": "e185's measure() VERBATIM (the e176n "
                                     "dial: base/held/site-read/old-band "
                                     "census/deletion table)"},
        "displacement": {
            "currency": ("cumulative ||theta_t - theta_0||_2 over all "
                         "2,739,072 trainable parameters (fp32, CPU, "
                         "measured per step); per-step increments "
                         "||theta_t - theta_{t-1}||_2; cosine vs THE CORPUS "
                         "DIRECTION (e185's stored control deltas, "
                         "recomputed from checkpoints on this device, "
                         "gated) at checkpoints"),
            "D_kill": D_KILL,
            "D_kill_stored_e185": D_KILL_STORED,
            "t_kill_e185": T_KILL_E185,
            "ctrl_gm12_at_tkill_e185": CTRL_GM12_AT_TKILL,
            "table": disp_table,
            "theta0_norm": {t: arms[t]["theta0_norm"] for t in tags},
            "e185_originals": {"disp": E185_DISP, "cos": E185_COS},
        },
        "gates": {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                  "G_POOL": G_POOL, "G_ANCHOR": G_ANCHOR, "G_ROOT": G_ROOT,
                  "G_E185CKPTS": G_E185CKPTS, "G_INPUTS": G_INPUTS,
                  "G_TARGETS": G_TARGETS, "G_FIDELITY": G_FIDELITY,
                  "G_SURG": gates_surg},
        "traces": trace,
        "batteries": batteries_all,
        "references": {
            "e185": {
                "ref": {"desc": "e185's stored n=1 originals (the overlay's "
                         "first member) + the control's displacement/cosine "
                         "columns + the per-step input hashes",
                        "metrics": "runs/e185/metrics.json",
                        "checkpoints": "runs/checkpoints/e185_{ctrl,noise,"
                                       "shuf}_s{1,2,4,10}.pt"},
                "provenance": src_e185,
            },
        },
        "adjudication": {
            "conditions": {
                "NOISE_KILLS_REPLICATES": {"fires": all_kill_by2
                                           and not any_spares},
                "ANY_SPARES": {"fires": any_spares},
                "ALL_KILLS_AT_M_E185_FORMALITY": {"fires": all_kills_at_M},
            },
            "cells": cells,
            "verdict": verdict, "clause": clause,
        },
        "honesty_reflex": {
            "seed_independence": ("the four noise draws (18503/18504 iid, "
                                  "18505/18506 perm) are independent "
                                  "dedicated generators drawn AFTER the "
                                  "shared seed-10902 input draws; the input "
                                  "stream is bit-identical across cells AND "
                                  "vs e185's stored per-step md5s (G_INPUTS) "
                                  "— the replicate samples the NOISE DRAW, "
                                  "not the input draw; n=3 per arm after "
                                  "this cell, but still ONE input stream and "
                                  "ONE root (the organism is not resampled)"),
            "displacement_matching": ("matching is at MEASURED displacement: "
                                      "per-cell cumulative norms reported "
                                      "per step and adjudicated against "
                                      "D_kill re-derived from e185's stored "
                                      "control checkpoints ON THIS DEVICE "
                                      "(gated vs the stored table to "
                                      f"{G_BIT_TOL:.0e}); AdamW's first "
                                      "step is lr-signed per coordinate, so "
                                      "step-1 displacement matches the e185 "
                                      "arms to weight-decay arithmetic "
                                      "(measured, not assumed)"),
            "device_consistency": ("all four trainings ran in ONE process on "
                                   "ONE CPU (4 threads); the corpus "
                                   "direction is recomputed from stored "
                                   "checkpoints on the same device and "
                                   "G_E185CKPTS re-derives e185's stored "
                                   "noise/shuf cosines from those "
                                   "checkpoints before any new cosine is "
                                   "read — no cross-device comparison "
                                   "anywhere"),
            "bars_anchored": ("KILL reuses the arc's absolute DISSOLVE bar "
                              "(0.27) at the corpus kill step +2; SPARE "
                              "reuses the SURVIVE bar (0.50) at any matched "
                              "checkpoint — the broadest honest scope for "
                              "the re-opening bar; composite order frozen "
                              "ANY-SPARES -> NOISE-KILLS-REPLICATES -> "
                              "TEXTURE before compute"),
            "logit_prediction_check": ("the noise kills annihilate the "
                                       "ENTIRE dial (g0/gp12/held30/site-"
                                       "read all collapse together — "
                                       "collateral devastation), unlike the "
                                       "corpus kill's surgical profile; "
                                       "co-reported per checkpoint so the "
                                       "verdict never rests on g-12 alone"),
            "recovery_fidelity": (
                "checkpoint-eval vs fresh-run: the re-run cell (the one "
                "incomplete grid) is compared PARAMETER-BY-PARAMETER "
                "against its surviving s{1,2,4} checkpoints (G_FIDELITY: "
                "bit-identity expected — the training is a pure function "
                "of the registered seeds on this device/torch build; "
                "5e-6 fallback flagged, never silent), which transitively "
                "anchors the eval-only cells (same script path, same "
                "seeds, same machine); eval-only cells' dial is measured "
                "directly FROM the survivor tensors (no reconstruction of "
                "weights involved — only the data stream is replayed, and "
                "it is gated bit-exact vs e185's stored md5s). Single "
                "lineage: one root, one input stream (10902), four noise "
                "draws; no cell was re-trained where its grid survived, "
                "and survivors were never overwritten."
                if RECOVERY else
                "single live process — no recovery involved"),
        },
        "trims": trims, "deviations": deviations,
        "ckpt_inventory": CKPT_INVENTORY,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": int(net0.num_params()),
                   "train_device": "cpu (CUDA_VISIBLE_DEVICES=-1)",
                   "eval_device": "cpu",
                   "torch_threads": torch.get_num_threads(),
                   "smoke": SMOKE, "recovery": bool(RECOVERY),
                   "torch": torch.__version__},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "noise_replicates.png", trace, disp_table, cells, verdict,
         clause)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'noise_replicates.png'}, "
        f"{len(CKPT_INVENTORY)} checkpoints in runs/checkpoints/e187_*.pt")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot

def plot(path, trace, disp_table, cells, verdict, clause):
    """THE figure: the n=3 overlay per arm — (a) the g-12 trajectories in
    step space (3 seeds per noise arm + the corpus control), (b) the
    displacement-vs-damage plane, (c) the cosines vs the corpus direction
    (n=3 per arm), (d) the per-cell displacement-match table + verdict."""
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 10.5))
    arm_col = {"noise": "royalblue", "shuf": "darkorange"}
    seed_marks = {18501: "x", 18503: "s", 18504: "D",
                  18502: "x", 18505: "^", 18506: "v"}
    new_cells = list(cells.keys())            # e.g. noise18503, shuf18506
    base_of = lambda t: "".join(c for c in t if not c.isdigit())
    seed_of = lambda t: int("".join(c for c in t if c.isdigit()))
    cell_col = {t: arm_col[base_of(t)] for t in new_cells}
    cell_mk = {t: seed_marks[seed_of(t)] for t in new_cells}

    # (0,0) the g-12 trajectories in step space (the n=3 overlay per arm)
    ax = axes[0, 0]
    st = E185_TRACE_STEPS
    ax.plot(st, E185_GM12["ctrl"], "o-", ms=8, lw=2.2, color="crimson",
            alpha=0.95, label="CONTROL (e185 stored, true targets)")
    for base in ("noise", "shuf"):
        ax.plot(st, E185_GM12[base], seed_marks[E185_ORIG_SEEDS[base]] + "--",
                ms=8, lw=1.4, color=arm_col[base], alpha=0.45,
                label=f"e185 orig {base} (seed {E185_ORIG_SEEDS[base]})")
    for t in new_cells:
        ax.plot([r["freeze_steps"] for r in trace[t]],
                [r["gm12"] for r in trace[t]], cell_mk[t] + "-",
                ms=7, lw=1.8, color=cell_col[t], alpha=0.9,
                label=f"{base_of(t)} seed {seed_of(t)} (new)")
    for yv, col, lbl in ((SURVIVE_BAR, "seagreen",
                          f"{SURVIVE_BAR} SPARE bar"),
                         (SHUT_BAR, "tab:purple",
                          f"{SHUT_BAR} DISSOLVE bar")):
        ax.axhline(yv, ls="--", lw=1.1, color=col, alpha=0.8, label=lbl)
    ax.set_xlabel("freeze steps from the root")
    ax.set_ylabel("g-12 (absolute mean p(Z), install-60 battery)")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=6.8, loc="center right")
    ax.set_title("THE n=3 OVERLAY PER ARM — g-12 trajectories (3 noise "
                 "draws per arm, 1 input stream)", fontsize=9.5)

    # (0,1) the displacement-vs-damage plane
    ax = axes[0, 1]
    xs = [E185_DISP["ctrl"][s] for s in CK_MAIN]
    ys = [E185_GM12["ctrl"][E185_TRACE_STEPS.index(s)] for s in CK_MAIN]
    ax.plot([0.0] + xs, [E185_GM12["ctrl"][0]] + ys, "o-", ms=8, lw=2.2,
            color="crimson", alpha=0.95, label="CONTROL (e185 stored)")
    for s, x, y in zip(CK_MAIN, xs, ys):
        ax.annotate(f"+{s}", (x, y), textcoords="offset points",
                    xytext=(5, 4), fontsize=7.5, color="crimson")
    for base in ("noise", "shuf"):
        ax.plot([E185_DISP[base][s] for s in CK_MAIN],
                [E185_GM12[base][E185_TRACE_STEPS.index(s)]
                 for s in CK_MAIN],
                seed_marks[E185_ORIG_SEEDS[base]] + "--", ms=7, lw=1.2,
                color=arm_col[base], alpha=0.45,
                label=f"e185 orig {base}")
    for t in new_cells:
        gm_of = {r["freeze_steps"]: r["gm12"] for r in trace[t]}
        # checkpoint rows only: live-trained cells carry per-step displacement
        # rows whose steps have no battery reading (fix of the latent v1 bug
        # that crashed panel (b) on the first full run)
        rows_c = [r for r in disp_table[t] if r["step"] in gm_of]
        xs2 = [r["cum_disp"] for r in rows_c]
        ys2 = [gm_of[r["step"]] for r in rows_c]
        ax.plot(xs2, ys2, cell_mk[t] + "-", ms=7, lw=1.6,
                color=cell_col[t], alpha=0.9,
                label=f"{base_of(t)} {seed_of(t)} (new)")
    ax.plot([0.0], [E185_GM12["ctrl"][0]], "k*", ms=14,
            label=f"root (g-12 {E185_GM12['ctrl'][0]:.3f})")
    for yv, col in ((SURVIVE_BAR, "seagreen"), (SHUT_BAR, "tab:purple")):
        ax.axhline(yv, ls="--", lw=1.0, color=col, alpha=0.8)
    ax.axvline(D_KILL_STORED, ls=":", lw=1.8, color="k", alpha=0.7,
               label=f"D_kill {D_KILL_STORED:.3f} (control dead at "
                     f"+{T_KILL_E185})")
    ax.set_xlabel(r"cumulative $\|\theta_t-\theta_0\|_2$ (all 2.74M params)")
    ax.set_ylabel("g-12 (absolute mean p(Z))")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=6.8, loc="center right")
    ax.set_title("the displacement-vs-damage plane (n=3 per arm; the bars "
                 "read at displacement-match)", fontsize=9.5)

    # (1,0) cosines vs the corpus direction (n=3 per arm)
    ax = axes[1, 0]
    width = 0.13
    series = []
    for base in ("noise", "shuf"):
        series.append((f"{base} e185 orig ({E185_ORIG_SEEDS[base]})",
                       [E185_COS[base][s] for s in CK_MAIN],
                       arm_col[base], 0.35, seed_marks[E185_ORIG_SEEDS[base]]))
    for t in new_cells:
        series.append((f"{base_of(t)} {seed_of(t)} (new)",
                       [next((r["cos_vs_corpus"] for r in disp_table[t]
                              if r["step"] == s), float("nan"))
                        for s in CK_MAIN],
                       cell_col[t], 0.9, cell_mk[t]))
    for k, (lbl, vals, col, al, mk) in enumerate(series):
        ax.bar([s + (k - (len(series) - 1) / 2) * width for s in CK_MAIN],
               vals, width=width, color=col, alpha=al, label=lbl)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(1.0, ls="--", color="crimson", lw=1.0, alpha=0.6,
               label="cos=1 (the corpus itself)")
    ax.set_xticks(list(CK_MAIN))
    ax.set_xticklabels([f"+{s}" for s in CK_MAIN])
    ax.set_xlabel("freeze step (checkpoint)")
    ax.set_ylabel("cos(arm delta, corpus-direction delta)")
    ax.legend(fontsize=6.8, loc="lower right", ncol=2)
    ax.set_title("the cosines vs THE CORPUS DIRECTION (orthogonal kills "
                 "replicate?)", fontsize=9.5)

    # (1,1) the per-cell table + verdict panel
    ax = axes[1, 1]
    ax.axis("off")
    ytxt = 0.97
    ax.text(0.02, ytxt, "THE REPLICATE TABLE (targets are the only delta; "
            "D_kill " + f"{D_KILL_STORED:.4f} re-derived+gated):",
            fontsize=8.5, va="top", family="monospace", weight="bold")
    ytxt -= 0.042
    ax.text(0.02, ytxt,
            "  cell            step   |d|    |d|/Dk  cos    g-12    read",
            fontsize=7.0, va="top", family="monospace")
    ytxt -= 0.032
    for t in new_cells:
        c = cells[t]
        ck_steps = {rr["freeze_steps"] for rr in trace[t]}
        for r in disp_table[t]:
            s = r["step"]
            if s not in ck_steps:
                continue   # per-step displacement rows (no battery reading)
            g = next((rr["gm12"] for rr in trace[t]
                      if rr["freeze_steps"] == s), None)
            cos = r.get("cos_vs_corpus")
            dk = r["cum_disp"] / D_KILL_STORED
            inmatch = r["cum_disp"] >= D_KILL_STORED
            if not inmatch:
                rd_ = "pre-match"
            elif g is None:
                rd_ = "n/a"
            elif g <= SHUT_BAR:
                rd_ = "KILLS" + (" (by+2)" if s == 2 else "")
            elif g >= SURVIVE_BAR:
                rd_ = "SPARES"
            else:
                rd_ = "partial"
            ax.text(0.02, ytxt,
                    f"  {t:<13} +{s:<4} {r['cum_disp']:.4f}  {dk:5.2f}  "
                    + (f"{cos:+.3f} " if cos is not None else "  n/a  ")
                    + (f"{g:.4f}  {rd_}" if g is not None
                       else f"{'n/a':>7}  {rd_}"),
                    fontsize=7.0, va="top", family="monospace",
                    color=cell_col[t])
            ytxt -= 0.026
        ax.text(0.02, ytxt,
                f"  => {'KILL by +2' if c['kill_by2'] else 'NO kill by +2'}"
                + f" (g-12@+2 {c['gm12_at_2']:.2e}, @+1 "
                + (f"{c['gm12_at_1']:.2e}" if c["gm12_at_1"] is not None
                   else "n/a")
                + "; max matched "
                + (f"{c['max_matched_gm12']:.2e}"
                   if c["max_matched_gm12"] is not None else "n/a")
                + f"; spares={c['spares']})", fontsize=7.0, va="top",
                family="monospace", weight="bold", color=cell_col[t])
        ytxt -= 0.030
    ytxt -= 0.008
    ax.text(0.02, ytxt, f"E187 VERDICT: {verdict}", fontsize=9.0, va="top",
            family="monospace", weight="bold", color="darkred")
    ytxt -= 0.038
    for wd in textwrap.wrap(clause, width=88, break_long_words=False):
        ax.text(0.02, ytxt, f"  {wd}", fontsize=6.7, va="top",
                family="monospace")
        ytxt -= 0.026

    fig.suptitle("E187 — THE NOISE REPLICATES: e185's content-free kill at "
                 f"2 fresh noise seeds per arm -> {verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
