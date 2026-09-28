"""E185 — THE NOISE-GRADIENT WASH (R52's missing discriminator; QUEUE row
e185, DISPATCHED ~19:00Z, commit da1689a).

WHY (T112's R52 verdict, the owed cell): the wash arc's lead finding — a
well-expressed consolidated fact dissolves on the FIRST gradient step of a
fact-free neutral stream (n=2 families, 3 streams, 3 wash-seeds, 2 lrs) —
has streamed through EXTINCTION (e176), NEUTRAL (e176N arm A) and FILTERED
(e183) compositions without changing its clock, and R52 flagged exactly
the hole: "the mechanism noun undiscriminated from generic two-step
optimizer fragility (the noise-gradient cell owed)". T111's mechanism
candidate is "ORDINARY CORPUS GRADIENT FLOW at lr 1e-3 — the stream's own
pressure". THE QUESTION THIS CELL DECIDES: is the kill CORPUS-DIRECTED
(the stream's gradients specifically destroy the readout) or GENERIC
two-step optimizer fragility (any sufficiently large AdamW step — even on
NOISE — destroys a basinless read)? If noise kills at matched
displacement, the noun dies and the finding becomes a BASIN-WIDTH
statement; if noise spares, corpus pressure is real.

REGISTERED BARS (frozen here before compute; the QUEUE e185 row + the
dispatch's registration verbatim; no bar shopping — adjudicate against
exactly this):
  - NOISE-KILLS fires if: the noise arms dissolve the fact at
    displacement-match (the kill is generic optimizer fragility — "no
    robustness basin"; the mechanism noun DIES; the stream-invariance
    becomes evidence AGAINST corpus-pressure; the paper's mechanism
    sentence rewrites to the basin-width form).
  - NOISE-SPARES fires if: the noise arms leave the fact >= 0.5 at
    matched displacement (corpus-directed pressure is REAL; the clause
    earns its noun).
  - No bar shopping; texture (partial damage) => TEXTURE with numbers.

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they do
not move the bars):
  * g-12 / g0 / g+12 / held30 = ABSOLUTE install-60 / held-30 battery
    mean p(Z) at ctx offsets -12 / 0 / +12 (e176/e176N/e184's convention
    verbatim — the same batteries, the same corpus rebuild, the same
    ruler); "the fact" = g-12.
  * DISSOLVE = g-12 <= 0.27 (the arc's SHUT bar, e158/e161/e176/e176N/
    e184); SPARE = g-12 >= 0.50 (the arc's SURVIVE bar).
  * DISPLACEMENT (the matching currency) = the cumulative parameter
    delta norm ||theta_t - theta_0||_2 over ALL 2,739,072 trainable
    parameters (65 tensors, fp32, CPU, measured — never assumed),
    plus the per-step increment ||theta_t - theta_{t-1}||_2. Reported
    per arm per step; the co-reported cosine(arm delta, control delta)
    at checkpoints quantifies direction divergence (a norm cannot).
  * THE CONTROL (arm C) = the real neutral wash re-run from the same
    bit-gated root with the same seed-10902 draw sequence at lr 1e-3;
    t_kill = the control's FIRST checkpoint {1,2,4,10} with g-12 <= 0.27
    (expected +2 by e176N's stored 0.0271; gated), D_kill = the control's
    cumulative displacement at t_kill. If the control does NOT kill by
    +10 the cell ABORTS to TEXTURE (control failure — nothing adjudicated;
    e176N's stored trajectory is the falsifier).
  * DISPLACEMENT-MATCH per noise arm = the earliest checkpoint M in
    {1,2,4,10} with ||delta|| >= D_kill ("at displacement-match"); the
    arm's damage is read at M AND at every later matched checkpoint
    through +10. If no checkpoint reaches D_kill: DISPLACEMENT-UNMATCHED
    (reported; the arm cannot fire either bar — expected not to happen:
    matched lr + matched inputs make the step-1 AdamW update
    norm-arm-independent up to weight decay; measured, not assumed).
  * per noise arm: KILLS = g-12 at M <= 0.27 (a kill that FIRST fires at
    a checkpoint with displacement < D_kill is CO-REPORTED as a fortiori);
    SPARES = g-12 >= 0.50 at EVERY matched checkpoint through +10.
  * NOISE-KILLS = BOTH noise arms KILLS. NOISE-SPARES = BOTH noise arms
    SPARES. Composite order NOISE-KILLS -> NOISE-SPARES -> TEXTURE; one
    kills + one spares = TEXTURE with numbers (the label-statistics
    contrast then carries the reading).
  * CE at each step (the arm's own in-batch training CE — the noise arms'
    price of their noise) + CE_R at every checkpoint.

THREE ARMS (one file, one run; all training-light / eval-heavy):
  (A) NOISE-LABELS: the neutral stream's windows with RANDOM labels —
      y ~ i.i.d. uniform over the 65-token vocabulary (destroys BOTH the
      input->output mapping AND the target statistics; the gradients are
      content-free noise at matched displacement). Noise RNG: dedicated
      torch.Generator seed 18501, drawn AFTER the aj/rj input draws so
      the input stream stays bit-identical to the control's.
  (B) SHUFFLED-TARGET: the same windows, targets = a random permutation
      (flat randperm per step) of the true next-tokens — preserves the
      batch's exact target multiset, destroys only the mapping. Noise
      RNG: seed 18502 (same discipline).
  (C) CONTROL: the REAL neutral wash (true targets) re-run — the
      reference displacement + the known kill, gated against e176N's
      stored arm-A cells at +1/+2 (tol 0.05; bit flag co-reported).
  All three: batch 32 = 16 neutral-anchor draws + 16 random corpus
  windows (e170's FIXED neutral bank, seed 170), full-token CE, AdamW
  (0.9,0.95) wd 0.1 constant lr 1e-3 clip 1.0, generator seed 10902 —
  the ONLY inter-arm delta is the TARGET tensor of each step (gated:
  per-step input hashes bit-identical across arms; permutation arm's
  target multiset == control's; noise arm's targets diverge). 10 steps,
  checkpoints {1,2,4,10}; full dial set + CE_R per checkpoint.

THE DISPLACEMENT-MATCHING METHODOLOGY (explicit): lr is held at the
wash's own 1e-3 for every arm (the dispatch's design), so the arms share
inputs, optimizer, lr and step count — the gradient CONTENT (direction
x per-coordinate consistency) is the only free variable. AdamW's first
bias-corrected step is +-lr per coordinate regardless of gradient
magnitude, so step-1 displacement is expected to match across arms to
within weight-decay arithmetic; from step 2 the second-moment estimates
differentiate (consistent corpus gradients keep steering; noise does
not). The bars therefore adjudicate at MEASURED displacement-match, and
the per-step displacement table (both noise arms' norms vs the real
wash's at the same steps — the dispatch's explicit report requirement)
is a first-class output, not an appendix.

COMPUTE ENVELOPE (dispatch): CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced
before torch, as e176n), LOW threads <= 4 (torch.set_num_threads(4)),
one 25 s launch stagger (single sleep, no busy-waiting anywhere),
cooldown 60 s before and after EACH of the three trainings, per-training
cap 180 s (e176N precedent: ~6 s/step at 4 threads; 10 steps ~ 60 s).

INSTRUMENT PROVENANCE: load_cpu/evl_load/battery_cell/battery_pz/
ce_fixed_cpu/val_windows/deleted_wpe/read_fact_at/row_census/measure()/
flat_cells/gate_vs are lab/e176n_neutral_wash.py VERBATIM (= e184's
copies; the e176/e161/e152/e151/e143/e131/e119/e113/e068/e065/e043
lineage); the neutral bank + junction accounting are e170 VERBATIM via
e176n's copy; the trainer is e176n's finetune_freeze with the target
substitution + the displacement bookkeeping added (per-step arithmetic
and the CPU-generator RNG draw sequence otherwise unchanged — the
aj/rj draws precede any noise draw). Copied, not imported, to own the
device policy.

NETS: root runs/checkpoints/e131_consolidated_e113.pt (gate bit-exact vs
e151's stored before-cells, e176n's G_ROOT set). e176N arm A's stored
trajectory = the control's reference (embedded, verified vs file at plot
time; it stored NO parameter-delta record — the control re-run supplies
the displacement column on this device). New checkpoints:
runs/checkpoints/e185_{noise,shuf,ctrl}_s{1,2,4,10}.pt.

Outputs: runs/e185/{metrics.json, noise_wash.png}. No NOTES/THINKING/
QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python e185_noise_wash.py    (E185_SMOKE=1 shakedown)
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
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e176n)

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

SMOKE = os.environ.get("E185_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e185 is CPU-only by dispatch"

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

# ---- census row set (e158/e161/e176/e176n/e184 old-band convention) -------------
ROWS_OLD = (0, 1, 2, 3, 4, 5, 6, 60, 100, 118, 119, 120) + tuple(range(121, 130))
if SMOKE:
    ROWS_OLD = (0, 1, 60, 121, 125, 129)

# ---- the three arms -------------------------------------------------------------
CK_MAIN: tuple[int, ...] = (1, 2, 4, 10) if not SMOKE else (1, 2)
N_STEPS = CK_MAIN[-1]

# ---- fine-tune envelope (e176N arm A VERBATIM; only the targets differ) ---------
LR = 1e-3                          # the wash's own lr (all arms)
FT_TIME_CAP = 180.0                # dispatch: <=180 s per training
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
FREEZE_SEED = 10902               # the locked seed lineage (= e161/e176/e176n)
NOISE_SEED_A = 18501              # arm A's target RNG (dedicated generator)
NOISE_SEED_B = 18502              # arm B's target RNG (dedicated generator)
COOLDOWN_S = 60.0                  # around EACH of the three trainings
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

# e176N arm A's stored FULL-DIAL trajectory (seed 10902, CPU-only; runs/e176n/
# metrics.json trace_armA VERBATIM, re-verified at plot time) — the control's
# reference cells (+1/+2 gate) and the damage-panel's lineage overlay. It stored
# NO parameter-delta record — the control re-run supplies the displacement column.
E176N_TRACE = {
    "seed": 10902,
    "freeze_steps": [0, 1, 2, 4, 50, 100, 200, 300],
    "gm12": [0.9155886173248291, 0.6780440807342529,
             0.027077054604887962, 0.010940761305391788,
             0.022122304886579514, 0.014922752045094967,
             0.002047251444299595, 0.0038193254731595516],
    "g0": [0.7850371599197388, 0.4619811177253723,
           0.1147073358297348, 0.06042749062152519,
           0.16835635900497437, 0.05249874293884169,
           0.011475668287767338, 0.016238771378993988],
    "gp12": [0.9478210210800171, 0.630972683429718,
             0.0140452291816473, 0.01570577546954155,
             0.051125336438417435, 0.016041822731594904,
             0.003007616149261594, 0.007402099203357575],
    "held30_gm12": [0.6361417174339294, 0.4820752739906311,
                    0.0053826989606022835, 0.001151302014477551,
                    0.008003094483572483, 0.008265051479424877,
                    0.0012307859724387527, 0.003200115170298809],
    "held30_g0": [0.7233286499977112, 0.46198493242263794,
                  0.052937380969524384, 0.008801662363111973,
                  0.0722624883055687, 0.018951624631881714,
                  0.009389182857285274, 0.010610799305140972],
    "ce_r": [1.663516640663147, 2.2113420963287354,
             2.032074451446533, 1.8213403224945068,
             1.708834171295166, 1.6700899600982666,
             1.6468615531921387, 1.642844796180725],
    "site_read_onset": [0.8898658156394958, 0.5253196954727173,
                        0.004276960156857967, 0.01323858741670847,
                        0.03920356547773186, 0.006887354888021946,
                        0.003228061832487583, 0.0064509566873312],
    "site_read_span": [0.982668936252594, 0.9041284918785095,
                       0.25527775287628174, 0.8370789289474487,
                       0.7883875966072083, 0.7708680629730227,
                       0.6387351153281067, 0.6367780546447754],
    "A129": [-0.13237020391970877, -0.18496511500949658,
             0.06476628839979337, 0.03774139005108736,
             0.06142527761985549, 0.021514282444527135,
             0.0026059946973267262, 0.0024257128311243534],
    "row0_strength": [0.7316772222270098, 0.44394606062541586,
                      0.11226377164743061, 0.05891658144703677,
                      0.16746244587458062, 0.05185848949654896,
                      0.011329283391791792, 0.016176560471552647],
    "dall_g0": [0.9047248959541321, 0.6481041312217712,
                0.02991652674973011, 0.013727720826864243,
                0.07909166067838669, 0.027866331860423088,
                0.00567732285708189, 0.010615414804498192],
}

# e176N arm A's stored LIGHT in-run trajectory (in-batch corpus CE at the
# checkpoints — the control's CE column reference).
E176N_TRAJ = [
    {"step": 1, "corpus_ce": 1.356567621231079},
    {"step": 2, "corpus_ce": 1.8263245820999146},
    {"step": 4, "corpus_ce": 1.3631858825683594},
    {"step": 50, "corpus_ce": 0.6575877070426941},
]

# ---- registered bar constants (frozen; see OPERATIONALIZATIONS above) ---------
SHUT_BAR = 0.27                   # DISSOLVE (e158/e161/e176/e176N/e184)
SURVIVE_BAR = 0.50                # SPARE
ROOT_GM12 = E151_ROOT["base_gm12"]
ROOT_G0 = E151_ROOT["base_g0"]

REGISTERED_PREDICTION = {
    "noise_kills": "NOISE-KILLS fires if: the noise arms dissolve the fact at "
        "displacement-match (the kill is generic optimizer fragility — 'no "
        "robustness basin'; the mechanism noun DIES; the stream-invariance "
        "becomes evidence AGAINST corpus-pressure; the paper's mechanism "
        "sentence rewrites to the basin-width form).",
    "noise_spares": "NOISE-SPARES fires if: the noise arms leave the fact "
        ">= 0.5 at matched displacement (corpus-directed pressure is REAL; "
        "the clause earns its noun).",
    "texture": "No bar shopping; texture (partial damage) => TEXTURE with "
        "numbers.",
    "operationalizations": "g-12/g0/g+12/held30 = absolute install-60/held-30 "
        "battery mean p(Z) at ctx offsets -12/0/+12 (e176N's convention); "
        "DISSOLVE = g-12 <= 0.27, SPARE = g-12 >= 0.50; DISPLACEMENT = "
        "cumulative ||theta_t - theta_0||_2 over all 2,739,072 trainable "
        "parameters (fp32, CPU, measured per step) + per-step increments; "
        "the CONTROL (arm C, the real neutral wash re-run, seed 10902 draw "
        "sequence, lr 1e-3) supplies t_kill = first checkpoint {1,2,4,10} "
        "with g-12 <= 0.27 (else the cell ABORTS to TEXTURE, control "
        "failure) and D_kill = its cumulative displacement at t_kill; "
        "per noise arm DISPLACEMENT-MATCH M = earliest checkpoint with "
        "||delta|| >= D_kill (none by +10 => DISPLACEMENT-UNMATCHED, "
        "co-reported); KILLS = g-12 at M <= 0.27 (kill first firing below "
        "D_kill co-reported as fortiori); SPARES = g-12 >= 0.50 at EVERY "
        "matched checkpoint through +10; NOISE-KILLS = both arms KILLS; "
        "NOISE-SPARES = both arms SPARES; order NOISE-KILLS -> NOISE-SPARES "
        "-> TEXTURE; CE at each step + CE_R at every checkpoint reported.",
    "registration": "QUEUE row e185 (DISPATCHED ~19:00Z, commit da1689a) + "
        "the dispatch's registration, frozen verbatim in this docstring "
        "before compute. Adjudicate against exactly this; no bar shopping.",
}

trims: list[str] = []
deviations: list[str] = [
    "CPU-ONLY by dispatch (CUDA_VISIBLE_DEVICES=-1 forced before torch, "
    "e176n precedent): torch threads 4, one 25 s launch stagger (single "
    "sleep, no busy-waiting), cooldown 60 s before/after EACH of the three "
    "trainings, per-training CPU cap 180 s (e176N precedent ~6 s/step at 4 "
    "threads; 10 steps ~ 60 s).",
    "The CONTROL runs to +10 (checkpoints {1,2,4,10}); the dispatch named 2 "
    "steps — the displacement-vs-damage PLANE needs the reference curve, "
    "and the +2 kill is gated against e176N's stored cells either way; the "
    "extra 8 steps change no bar.",
    "The trainer is e176n's finetune_freeze with (i) the per-step target "
    "substitution (arm A i.i.d. uniform over the vocab / arm B flat "
    "randperm of the true targets / control true) drawn from a DEDICATED "
    "generator (seeds 18501/18502) AFTER the aj/rj input draws, and (ii) "
    "the displacement bookkeeping (flat fp32 parameter vector, cumulative "
    "and per-step L2 norms, checkpoint delta vectors for the cosine "
    "co-report). The per-step arithmetic and the seed-10902 aj/rj draw "
    "sequence are unchanged — the input stream is bit-identical across "
    "arms (gated by per-step md5).",
    "NOISE-LABELS operationalized as y ~ i.i.d. uniform over the 65-token "
    "vocabulary (destroys mapping AND target statistics); SHUFFLED-TARGET "
    "as a flat random permutation of the batch's true targets (preserves "
    "the exact target multiset, destroys the mapping). The dispatch's "
    "parenthetical '(shuffled targets)' under arm A is read against arm "
    "B's explicit 'preserves token statistics' — the two arms bracket "
    "'noise' from both sides.",
    "e176N's stored arm A has NO parameter-delta record (displacement was "
    "never measured in the arc); the CONTROL re-run supplies the reference "
    "displacement on this device — cross-device displacement comparison is "
    "therefore avoided by construction (all three arms run this device, "
    "this process).",
    "Nets are the mandated 2.7M e131_consolidated line (the dispatch's "
    "'<=1M family' note is an envelope statement; e143/e151/e152/e158/"
    "e176n/e184 precedent — every gate reference lives on this line).",
    "Eval thread count is 4 (dispatch) vs e151's stored cells — CPU "
    "reduction order can drift low-order bits; the G_ROOT gate reports both "
    "the 5e-6 bit flag and the 0.05 fallback tolerance (e161/e176/e176n/"
    "e184 precedent).",
    "Single seed (10902 input stream; single draws of the target noise at "
    "18501/18502), one trajectory per arm, one root lineage, n=1 per cell "
    "— point estimates until replicated; the R52-owed DISCRIMINATION is "
    "within-trajectory (targets are the only delta), not across-trajectory.",
    "Smoke mode trims: 2-step trainings, checkpoints {1,2}, lean measures "
    "(no censuses, no deletion table), no cooldowns; nothing adjudicated.",
]


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
    e176n/e178/e184 copy): p(true name char) at positions addr_row..addr_row+6."""
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
    """THE NEUTRAL PLAIN-CORPUS FREEZE with target substitution + displacement
    bookkeeping (e176n's finetune_freeze VERBATIM arithmetic). Per step:
    aj = randint(16) anchor draws, rj = randint(16) random corpus offsets (the
    seed-10902 generator — BIT-IDENTICAL across the three arms); THEN the
    target substitution from the dedicated noise generator: 'true' = the real
    wash; 'iid' = y ~ uniform(vocab); 'perm' = flat randperm of the true
    targets. Batch 32 full-token CE; AdamW (0.9,0.95) wd 0.1 constant lr,
    clip 1.0. Displacement: cumulative ||theta_t - theta_0||_2 + per-step
    ||theta_t - theta_{t-1}||_2 measured every step; checkpoint delta vectors
    kept (CPU) for the cosine co-report. Snapshots + light CPU evals (g-12,
    g0, CE_R — no RNG consumed) at the checkpoint steps; in-batch CE recorded
    at EVERY step (the dispatch's CE-at-each-step)."""
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
        # ---- THE TARGET SUBSTITUTION (the arm's only delta; dedicated RNG,
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


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "e185", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name} "
        f"({', '.join(f'{k}={v}' for k, v in meta.items())})")


def verify_e176n(embedded: dict, traj_light: list, path: Path) -> dict:
    """Verify the embedded e176n arm-A copies against the stored metrics file
    (no silent divergence; e176n/e184's verify convention)."""
    src = {"source": "embedded verbatim copy (runs/e176n trace_armA/traj)",
           "file_present": path.exists(), "verified_vs_embedded": None,
           "max_abs_diff": None}
    if path.exists():
        mm = json.loads(path.read_text(encoding="utf-8"))
        rows = mm["trace_armA"]
        diffs = []
        for r in rows:
            s = r["freeze_steps"]
            if s not in embedded["freeze_steps"]:
                continue
            i = embedded["freeze_steps"].index(s)
            for k in ("gm12", "g0", "gp12", "held30_gm12", "held30_g0",
                      "ce_r", "site_read_onset", "site_read_span", "A129",
                      "row0_strength", "dall_g0"):
                diffs.append(abs(r[k] - embedded[k][i]))
        steps_ok = all(s in embedded["freeze_steps"]
                       for s in [r["freeze_steps"] for r in rows])
        src["max_abs_diff"] = max(diffs) if diffs else None
        src["verified_vs_embedded"] = bool(steps_ok and max(diffs) < 1e-9)
        if src["verified_vs_embedded"]:
            src["source"] = ("runs/e176n/metrics.json trace_armA (embedded "
                             f"copy verified, max|diff| {max(diffs):.1e})")
        stored_traj = mm["armA_neutral"]["traj"]
        ce_ok = all(abs(a["corpus_ce"] - b["corpus_ce"]) < 1e-9
                    for a, b in zip(stored_traj, traj_light)) and \
            len(stored_traj) >= len(traj_light)
        src["traj_corpus_ce_verified"] = bool(ce_ok)
    return src


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e185_smoke" if SMOKE else "e185")
    log(f"E185 THE NOISE-GRADIENT WASH (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()}), stagger "
        f"{STAGGER_S:.0f}s, cooldown {COOLDOWN_S:.0f}s around each training")
    time.sleep(STAGGER_S)            # launch stagger vs the CPU fleet

    # ---------------- protocol rebuild (e143/e151/e152/e161/e176/e176n/e184)
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
    # FIXED content (seed 170) — the arms differ ONLY in each step's targets.
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
        "rng_note": ("noise_wash verbatim; the seed-10902 aj/rj draw sequence "
                     "is IDENTICAL across the three arms (same shapes/moduli, "
                     "drawn BEFORE any noise draw) — the input stream is "
                     "bit-identical (gated by per-step md5); ONLY each step's "
                     "target tensor differs (true / iid / perm)"),
        "random_channel_background": {
            "host_occurrences_in_train": host_occ_total,
            "est_window_hit_rate": bg_rate,
            "note": ("the 16-per-batch random corpus windows are e161/e176/"
                     "e176n/e184 VERBATIM and unfiltered — identical "
                     "background (~3.8%/window) in all three arms; not part "
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
    # THE THREE ARMS (one training each; cooldown around each; the control
    # RUNS FIRST — its t_kill / D_kill define the match the bars read on)
    # =====================================================================
    ARM_SPECS = [
        ("ctrl", "true", None, "CONTROL — the real neutral wash (true "
         "targets), seed 10902 draw sequence, lr 1e-3: the reference "
         "displacement + the known kill"),
        ("noise", "iid", NOISE_SEED_A, "NOISE-LABELS — the same windows, "
         "y ~ i.i.d. uniform over the 65-token vocab (destroys mapping AND "
         "target statistics; content-free gradients at matched displacement)"),
        ("shuf", "perm", NOISE_SEED_B, "SHUFFLED-TARGET — the same windows, "
         "y = a flat random permutation of the true targets (preserves the "
         "exact target multiset, destroys only the mapping)"),
    ]
    arms: dict = {}
    batteries_all: dict = {}
    for tag, mode, nseed, desc in ARM_SPECS:
        log("=" * 78)
        if not SMOKE:
            log(f"[thermal] cooldown {COOLDOWN_S:.0f}s before {tag}")
            cooldown(COOLDOWN_S)
        log(f"ARM {tag.upper()} — {desc}; {N_STEPS} steps, checkpoints "
            f"+{list(CK_MAIN)}")
        arm = noise_wash(tag, net0, anchor_neutral, train_ids, itos,
                         r_eval_xy, gm12_ids, g0_ids, zid, mode,
                         nseed if nseed is not None else 0, CK_MAIN)
        if not SMOKE:
            log(f"[thermal] cooldown {COOLDOWN_S:.0f}s after {tag}")
            cooldown(COOLDOWN_S)
        G_DRAWFREE = {f"zeph_violations_{tag}": arm["zeph_violations"],
                      "pass": bool(arm["zeph_violations"] == 0)}
        assert G_DRAWFREE["pass"], f"{tag}: name token leaked into a window"
        gates_surg[f"G_DRAWFREE_{tag}"] = G_DRAWFREE
        arms[tag] = arm

        for s in sorted(arm["sds"]):
            save_ckpt(f"e185_{tag}_s{s}", arm["sds"][s],
                      {"desc": f"e131_consolidated_e113 + {s}-step "
                               f"{mode.upper()}-target neutral freeze "
                               f"(target_mode={mode}), lr {LR}, input seed "
                               f"{FREEZE_SEED}"
                               + (f", noise seed {nseed}" if nseed else ""),
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

    # =====================================================================
    # CONSTRUCTION-FIDELITY GATES (the noise arms' inputs/targets)
    # =====================================================================
    G_INPUTS = {"per_step": {}, "pass": None}
    for step in range(1, N_STEPS + 1):
        hc = arms["ctrl"]["x_hashes"].get(step)
        hn = arms["noise"]["x_hashes"].get(step)
        hs = arms["shuf"]["x_hashes"].get(step)
        same = bool(hc is not None and hc == hn == hs)
        G_INPUTS["per_step"][step] = {"ctrl": hc, "noise": hn, "shuf": hs,
                                      "identical": same}
    G_INPUTS["pass"] = bool(all(v["identical"]
                                for v in G_INPUTS["per_step"].values()))
    assert G_INPUTS["pass"], ("input streams diverged across arms — the "
                              "delta is not the targets alone")
    log(f"G_INPUTS: per-step input batches bit-identical across all three "
        f"arms ({N_STEPS}/{N_STEPS}): PASS")

    G_TARGETS = {"per_arm": {}}
    for tag in ("noise", "shuf"):
        ys = arms[tag]["y_stats"]
        G_TARGETS["per_arm"][tag] = {
            "frac_targets_changed_mean": float(
                np.mean([r["frac_targets_changed"] for r in ys])),
            "multiset_equals_true_all_steps": bool(
                all(r["multiset_equals_true"] for r in ys)),
        }
    G_TARGETS["per_arm"]["noise"]["expectation"] = (
        "iid labels change ~64/65 of positions; multiset NOT preserved")
    G_TARGETS["per_arm"]["shuf"]["expectation"] = (
        "permutation changes ~63/65 positions; multiset preserved at EVERY "
        "step")
    G_TARGETS["pass"] = bool(
        G_TARGETS["per_arm"]["noise"]["frac_targets_changed_mean"] > 0.9
        and not G_TARGETS["per_arm"]["noise"]["multiset_equals_true_all_steps"]
        and G_TARGETS["per_arm"]["shuf"]["frac_targets_changed_mean"] > 0.9
        and G_TARGETS["per_arm"]["shuf"]["multiset_equals_true_all_steps"])
    assert G_TARGETS["pass"], f"target gate FAILED: {G_TARGETS}"
    log(f"G_TARGETS: noise i.i.d. (changed "
        f"{G_TARGETS['per_arm']['noise']['frac_targets_changed_mean']:.3f}, "
        f"multiset broken), shuf permutation (changed "
        f"{G_TARGETS['per_arm']['shuf']['frac_targets_changed_mean']:.3f}, "
        f"multiset preserved at every step): PASS")

    # control vs e176n stored cells (the known kill, on this device)
    def ctrl_cells(step):
        return flat_cells(batteries_all["ctrl"][str(step)])
    i1 = E176N_TRACE["freeze_steps"].index(1)
    i2 = E176N_TRACE["freeze_steps"].index(2)
    G_CTRL = gate_vs(
        {**{f"+{s}_{k}": ctrl_cells(s)[k] for s in (1, 2)
            for k in ("gm12", "g0", "ce_r")},
         },
        {f"+{s}_{k}": E176N_TRACE[k][i]
         for s, i in ((1, i1), (2, i2))
         for k in ("gm12", "g0", "ce_r")},
        "G_CTRL (control vs e176n arm-A stored +1/+2)")
    if not G_CTRL["pass"]:
        raise RuntimeError("the control re-run diverged from e176N's stored "
                           "arm-A cells — the displacement reference is "
                           "untrustworthy on this device")

    # =====================================================================
    # DISPLACEMENT TABLE + COSINES (the matching currency, measured)
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
    for tag in ("ctrl", "noise", "shuf"):
        rows = []
        for t in arms[tag]["traj"]:
            s = t["step"]
            row = {"step": s, "ce_batch": t["ce_batch"],
                   "cum_disp": t["cum_disp"], "step_disp": t["step_disp"]}
            if "g_m12_mean_pz" in t:
                row["g_m12_light"] = t["g_m12_mean_pz"]
                row["ce_r_light"] = t["ce_r"]
            if s in arms[tag]["deltas"] and s in arms["ctrl"]["deltas"]:
                d_a = arms[tag]["deltas"][s]
                d_c = arms["ctrl"]["deltas"][s]
                cos = float(torch.dot(d_a, d_c)
                            / (torch.norm(d_a) * torch.norm(d_c) + 1e-30))
                row["cos_vs_ctrl"] = cos
            rows.append(row)
        disp_table[tag] = rows

    # =====================================================================
    # ADJUDICATION (registered clauses; no shopping)
    # =====================================================================
    trace = {tag: trace_from(batteries_all[tag], [0] + sorted(arms[tag]["sds"]))
             for tag in ("ctrl", "noise", "shuf")}
    ctrl_gm12 = {r["freeze_steps"]: r["gm12"] for r in trace["ctrl"]}
    t_kill = next((s for s in CK_MAIN if ctrl_gm12.get(s, 1.0) <= SHUT_BAR),
                  None)
    ctrl_kills = t_kill is not None
    if ctrl_kills:
        D_kill = next(r["cum_disp"] for r in disp_table["ctrl"]
                      if r["step"] == t_kill)
    else:
        D_kill = None

    match, per_arm = {}, {}
    if ctrl_kills:
        for tag in ("noise", "shuf"):
            gm = {r["freeze_steps"]: r["gm12"] for r in trace[tag]}
            M = next((r["step"] for r in disp_table[tag]
                      if r["cum_disp"] >= D_kill), None)
            # matched CHECKPOINTS = measured dial steps ({1,2,4,10}) whose
            # displacement is >= D_kill (the frozen operationalization reads
            # "every matched checkpoint" — the dial exists only at the
            # checkpoint set; per-step light g-12 co-reported in traj)
            matched_ckpts = [r["step"] for r in disp_table[tag]
                             if r["cum_disp"] >= D_kill
                             and r["step"] in gm]
            below_match_kill = next(
                (r["step"] for r in disp_table[tag]
                 if r["cum_disp"] < D_kill and gm.get(r["step"], 1.0) <= SHUT_BAR),
                None)
            kills = bool(M is not None and gm.get(M, 1.0) <= SHUT_BAR)
            spares = bool(matched_ckpts and all(gm.get(s, 0.0) >= SURVIVE_BAR
                                                for s in matched_ckpts))
            match[tag] = {"M": M, "matched_ckpts": matched_ckpts,
                          "D_kill": D_kill,
                          "gm12_at_M": gm.get(M) if M is not None else None,
                          "matched_gm12": {s: gm.get(s)
                                           for s in matched_ckpts},
                          "below_match_kill": below_match_kill}
            per_arm[tag] = {"KILLS": kills, "SPARES": spares}
    else:
        match = {"ctrl_failure": True}
        per_arm = {"noise": {"KILLS": None, "SPARES": None},
                   "shuf": {"KILLS": None, "SPARES": None}}

    noise_kills = bool(ctrl_kills and per_arm["noise"]["KILLS"]
                       and per_arm["shuf"]["KILLS"])
    noise_spares = bool(ctrl_kills and per_arm["noise"]["SPARES"]
                        and per_arm["shuf"]["SPARES"])

    if not ctrl_kills:
        verdict = "TEXTURE (CONTROL FAILURE)"
        clause = (f"the CONTROL did not kill by +10 (g-12 trace "
                  + " -> ".join(f"+{s}:{ctrl_gm12.get(s, float('nan')):.4f}"
                                for s in CK_MAIN if s in ctrl_gm12)
                  + f") — e176N's stored arm A killed at +2 (0.0271); "
                  f"nothing adjudicated; the run is flagged for a device/"
                  f"protocol audit before any noun moves.")
    elif noise_kills:
        mN, mS = match["noise"], match["shuf"]
        verdict = "NOISE-KILLS"
        clause = (f"BOTH noise arms dissolved the fact at displacement-match: "
                  f"noise-labels g-12 {mN['gm12_at_M']:.4f} at +{mN['M']} "
                  f"(disp >= D_kill {D_kill:.4f}), shuffled-target "
                  f"{mS['gm12_at_M']:.4f} at +{mS['M']} — the kill is GENERIC "
                  f"optimizer fragility, not corpus-directed: any AdamW step "
                  f"of the wash's size destroys a basinless read. THE "
                  f"MECHANISM NOUN DIES ('no robustness basin'); the "
                  f"stream-invariance becomes evidence AGAINST corpus-"
                  f"pressure; the paper's mechanism sentence rewrites to the "
                  f"basin-width form.")
    elif noise_spares:
        verdict = "NOISE-SPARES"
        clause = (f"BOTH noise arms spared the fact at matched displacement "
                  f"(>= {SURVIVE_BAR} at every checkpoint with disp >= "
                  f"D_kill {D_kill:.4f}; control dead at +{t_kill} with "
                  f"{ctrl_gm12[t_kill]:.4f}) — corpus-directed pressure is "
                  f"REAL: the kill needs the stream's gradient DIRECTION, "
                  f"not merely its step size; the clause earns its noun.")
    else:
        verdict = "TEXTURE"
        bits = []
        for tag, nm in (("noise", "noise-labels"), ("shuf", "shuffled-target")):
            pa = per_arm[tag]
            mt = match[tag]
            gM = mt["gm12_at_M"]
            bits.append(f"{nm}: "
                        + ("KILLS" if pa["KILLS"] else
                           ("SPARES" if pa["SPARES"] else "partial"))
                        + (f" g-12 {gM:.4f} at +{mt['M']}" if gM is not None
                           else " displacement-unmatched"))
        clause = ("partial damage at displacement-match — " + "; ".join(bits)
                  + f"; D_kill {D_kill:.4f} (control dead at +{t_kill}); "
                  f"full dial trajectories reported, no bar shopping.")

    log("=" * 78)
    log(f"E185 VERDICT: {verdict}")
    for tag in ("ctrl", "noise", "shuf"):
        seq = " -> ".join(f"+{r['freeze_steps']}:{r['gm12']:.4f}"
                          for r in trace[tag])
        disp = " -> ".join(f"+{r['step']}:{r['cum_disp']:.4f}"
                           for r in disp_table[tag])
        log(f"  {tag}: g-12 {seq}")
        log(f"  {tag}: |d(theta)| {disp}")
    if ctrl_kills:
        log(f"  t_kill=+{t_kill}  D_kill={D_kill:.4f}  match: "
            + "; ".join(f"{t}: M=+{match[t]['M']}, g-12@M="
                        + (f"{match[t]['gm12_at_M']:.4f}"
                           if match[t]["gm12_at_M"] is not None else "n/a")
                        for t in ("noise", "shuf")))
    log(f"  {clause}")
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e185_noise_wash",
        "date": common.now_iso(),
        "registration": ("QUEUE row e185 (DISPATCHED ~19:00Z, commit da1689a) "
                         "+ the dispatch's registration; frozen verbatim in "
                         "the module docstring before compute"),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("is the two-step kill CORPUS-DIRECTED (the stream's "
                     "gradients specifically destroy the readout) or GENERIC "
                     "two-step optimizer fragility (any sufficiently large "
                     "AdamW step — even on NOISE — destroys a basinless "
                     "read)? Targets are the only inter-arm delta; "
                     "displacement is the matching currency."),
        "root": f"runs/checkpoints/{ROOT_CK} (gated vs e151 before-cells, "
                f"max|diff| {G_ROOT['max_abs_diff']:.2e})",
        "root_meta": root_meta,
        "arms": {
            tag: {
                "desc": desc,
                "target_mode": mode,
                "noise_seed": nseed,
                "ckpt_steps": list(CK_MAIN),
                "steps_ran": arms[tag]["steps_ran"],
                "seed": FREEZE_SEED, "lr": LR,
                "traj": arms[tag]["traj"],
                "zeph_violations": arms[tag]["zeph_violations"],
                "y_stats": arms[tag]["y_stats"],
                "missing_checkpoints": [s for s in CK_MAIN
                                        if s not in arms[tag]["sds"]],
            } for (tag, mode, nseed, desc) in ARM_SPECS
        },
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
                     "geometries_measured": {"novel": [-12, 12],
                                             "trained_g0": 0},
                     "neutral_bank": G_ANCHOR["neutral_bank"],
                     "measure_dial": "e176n's measure() (minus the 183-span "
                                     "census; e178's recorded deviation)"},
        "displacement": {
            "currency": ("cumulative ||theta_t - theta_0||_2 over all "
                         "2,739,072 trainable parameters (fp32, CPU, "
                         "measured per step); per-step increments "
                         "||theta_t - theta_{t-1}||_2; cosine vs the "
                         "control's same-step delta at checkpoints"),
            "table": disp_table,
            "theta0_norm": {t: arms[t]["theta0_norm"]
                            for t in ("ctrl", "noise", "shuf")},
        },
        "gates": {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                  "G_POOL": G_POOL, "G_ANCHOR": G_ANCHOR, "G_ROOT": G_ROOT,
                  "G_INPUTS": G_INPUTS, "G_TARGETS": G_TARGETS,
                  "G_CTRL": G_CTRL, "G_SURG": gates_surg},
        "traces": trace,
        "batteries": batteries_all,
        "references": {
            "e176n_armA": {
                "ref": {"seed": 10902, "desc": "the neutral wash's stored "
                         "cell (e176N arm A, CPU-only run) — the control's "
                         "damage reference; NO displacement record existed"},
                "provenance": src_e176n},
        },
        "adjudication": {
            "conditions": {
                "NOISE_KILLS": {"fires": noise_kills,
                                "per_arm": per_arm, "match": match},
                "NOISE_SPARES": {"fires": noise_spares,
                                 "per_arm": per_arm, "match": match},
                "CONTROL_KILL": {"t_kill": t_kill, "D_kill": D_kill,
                                 "ctrl_gm12": ctrl_gm12,
                                 "fires": ctrl_kills},
            },
            "verdict": verdict, "clause": clause,
            "displacement_match": match,
        },
        "honesty_reflex": {
            "noise_arm_construction": ("the targets are the ONLY delta: "
                                       "per-step input batches are "
                                       "bit-identical across arms (md5-gated, "
                                       "G_INPUTS) because the noise draws "
                                       "come from a dedicated generator "
                                       "AFTER the shared seed-10902 aj/rj "
                                       "draws; the permutation arm's target "
                                       "multiset equals the true one at "
                                       "every step (G_TARGETS); the i.i.d. "
                                       "arm breaks it by design"),
            "displacement_currency": ("a single unweighted L2 scalar over "
                                      "2.74M parameters: two steps of equal "
                                      "norm but different direction are "
                                      "'matched' by this ruler while being "
                                      "different interventions — which is "
                                      "exactly the question; the cosine "
                                      "co-report quantifies the direction "
                                      "divergence the norm hides"),
            "single_seed": ("one input stream (10902), one draw of each "
                            "target-noise seed (18501/18502), one root — "
                            "n=1 per cell; the discrimination is "
                            "within-trajectory (huge expected effects on "
                            "both sides of the fork), but replicate seeds "
                            "are owed before the paper's noun moves"),
            "same_device": ("all three arms ran in ONE process on ONE CPU "
                            "(4 threads) — displacement comparisons are "
                            "within-device by construction; e176N's stored "
                            "arm A supplies only DAMAGE reference cells "
                            "(gated), never displacement"),
            "control_extension": ("the dispatch named a 2-step control; it "
                                  "ran to +10 to draw the plane's reference "
                                  "curve (recorded deviation; no bar "
                                  "depends on steps 4/10 of the control)"),
            "bars_anchored": ("DISSOLVE/SPARE reuse the arc's absolute "
                              "home-battery bars (0.27 / 0.50) on the same "
                              "ruler as e158/e161/e176/e176N/e184"),
        },
        "trims": trims, "deviations": deviations,
        "ckpt_inventory": CKPT_INVENTORY,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": int(net0.num_params()),
                   "train_device": "cpu (CUDA_VISIBLE_DEVICES=-1)",
                   "eval_device": "cpu",
                   "torch_threads": torch.get_num_threads(),
                   "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "noise_wash.png", trace, disp_table, match, per_arm, verdict,
         clause, t_kill, D_kill, ctrl_gm12)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'noise_wash.png'}, "
        f"{len(CKPT_INVENTORY)} checkpoints in runs/checkpoints/e185_*.pt")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot

def plot(path, trace, disp_table, match, per_arm, verdict, clause,
         t_kill, D_kill, ctrl_gm12):
    """THE figure: the displacement-vs-damage plane (the headline), the
    step-space trajectories, the per-step displacement norms, and the
    displacement-match table + verdict."""
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 10.5))
    cols = {"ctrl": "crimson", "noise": "royalblue", "shuf": "darkorange"}
    lbls = {"ctrl": "CONTROL — real neutral wash (true targets)",
            "noise": "NOISE-LABELS (iid uniform targets)",
            "shuf": "SHUFFLED-TARGET (permuted true targets)"}
    marks = {"ctrl": "o", "noise": "s", "shuf": "^"}

    def disp_of(tag, step):
        return next((r["cum_disp"] for r in disp_table[tag]
                     if r["step"] == step), None)

    # (0,0) THE DISPLACEMENT-VS-DAMAGE PLANE (the headline)
    ax = axes[0, 0]
    for tag in ("ctrl", "noise", "shuf"):
        xs, ys, ss = [], [], []
        for r in trace[tag]:
            if r["freeze_steps"] == 0:
                continue
            d = disp_of(tag, r["freeze_steps"])
            if d is not None:
                xs.append(d)
                ys.append(r["gm12"])
                ss.append(r["freeze_steps"])
        ax.plot(xs, ys, marks[tag] + "-", ms=9, lw=2.0, color=cols[tag],
                alpha=0.9, label=lbls[tag])
        for x, y, s in zip(xs, ys, ss):
            ax.annotate(f"+{s}", (x, y), textcoords="offset points",
                        xytext=(5, 5), fontsize=7.5, color=cols[tag])
    ax.plot([0.0], [trace["ctrl"][0]["gm12"]], "k*", ms=14,
            label=f"root (g-12 {trace['ctrl'][0]['gm12']:.3f})")
    for yv, col, lbl in ((SURVIVE_BAR, "seagreen",
                          f"{SURVIVE_BAR} SPARE bar"),
                         (SHUT_BAR, "tab:purple",
                          f"{SHUT_BAR} DISSOLVE bar")):
        ax.axhline(yv, ls="--", lw=1.1, color=col, alpha=0.8, label=lbl)
    if D_kill is not None:
        ax.axvline(D_kill, ls=":", lw=1.8, color="k", alpha=0.7,
                   label=f"D_kill {D_kill:.3f} (control dead at "
                         f"+{t_kill})")
        ax.annotate("displacement-match\n(the bars read here)", (D_kill, 0.5),
                    textcoords="offset points", xytext=(6, 0), fontsize=7.5)
    ax.set_xlabel("cumulative parameter displacement "
                  r"$\|\theta_t-\theta_0\|_2$ (all 2.74M params, fp32)")
    ax.set_ylabel("g-12 (absolute mean p(Z), install-60 battery)")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7.2, loc="center right")
    ax.set_title("THE DISPLACEMENT-VS-DAMAGE PLANE — same step size, "
                 "different gradient content", fontsize=10)

    # (0,1) the step-space trajectories + e176n stored reference
    ax = axes[0, 1]
    for tag in ("ctrl", "noise", "shuf"):
        ax.plot([r["freeze_steps"] for r in trace[tag]],
                [r["gm12"] for r in trace[tag]], marks[tag] + "-", ms=8,
                lw=2.0, color=cols[tag], alpha=0.9, label=lbls[tag])
    fine = [(s, g) for s, g in zip(E176N_TRACE["freeze_steps"],
                                   E176N_TRACE["gm12"]) if s <= 10]
    ax.plot([s for s, _ in fine], [g for _, g in fine], "kx--", ms=7, lw=1.2,
            alpha=0.6, label="e176N arm A stored (seed 10902, CPU)")
    for yv, col in ((SURVIVE_BAR, "seagreen"), (SHUT_BAR, "tab:purple")):
        ax.axhline(yv, ls="--", lw=1.0, color=col, alpha=0.8)
    ax.set_xlabel("freeze steps from the root")
    ax.set_ylabel("g-12 (absolute mean p(Z))")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7.2, loc="center right")
    ax.set_title("the dial trajectories in step space (control overlays "
                 "the stored wash)", fontsize=9.5)

    # (1,0) per-step displacement norms (incremental + cumulative)
    ax = axes[1, 0]
    for tag in ("ctrl", "noise", "shuf"):
        stps = [r["step"] for r in disp_table[tag]]
        incs = [r["step_disp"] for r in disp_table[tag]]
        ax.bar([s + 0.13 * ("ctrl", "noise", "shuf").index(tag) - 0.13
                for s in stps], incs, width=0.24, color=cols[tag],
               alpha=0.55, label=f"{tag} per-step " +
               r"$\|\Delta\theta_t\|$")
    axr = ax.twinx()
    for tag in ("ctrl", "noise", "shuf"):
        axr.plot([r["step"] for r in disp_table[tag]],
                 [r["cum_disp"] for r in disp_table[tag]], marks[tag] + "-",
                 ms=6, lw=1.8, color=cols[tag],
                 label=f"{tag} cumulative |d|")
    if D_kill is not None:
        axr.axhline(D_kill, ls=":", lw=1.5, color="k", alpha=0.7)
        axr.annotate(f"D_kill {D_kill:.3f}", (0.02, D_kill),
                     xycoords=("axes fraction", "data"),
                     textcoords="offset points", xytext=(4, 4), fontsize=7.5)
    ax.set_xlabel("freeze step")
    ax.set_ylabel("per-step displacement (bars)")
    axr.set_ylabel("cumulative displacement (lines)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.8, loc="lower right")
    ax.set_title("displacement norms per arm per step (the matching "
                 "currency, measured)", fontsize=9.5)

    # (1,1) the displacement-match table + verdict panel
    ax = axes[1, 1]
    ax.axis("off")
    ytxt = 0.97
    ax.text(0.02, ytxt, "THE DISPLACEMENT-MATCH TABLE (targets are the "
            "only delta):", fontsize=8.5, va="top", family="monospace",
            weight="bold")
    ytxt -= 0.042
    ax.text(0.02, ytxt,
            "  arm        step   |d|    |d|/Dk  cos-vs-ctrl   g-12    "
            "cell",
            fontsize=7.2, va="top", family="monospace")
    ytxt -= 0.034
    gm = {tag: {r["freeze_steps"]: r["gm12"] for r in trace[tag]}
          for tag in ("ctrl", "noise", "shuf")}
    for tag in ("ctrl", "noise", "shuf"):
        for r in disp_table[tag]:
            s = r["step"]
            dk = (r["cum_disp"] / D_kill) if D_kill else float("nan")
            cos = r.get("cos_vs_ctrl")
            g = gm[tag].get(s)
            if tag == "ctrl":
                cell = ("KILL" if (ctrl_gm12.get(s, 1.0) <= SHUT_BAR)
                        else "alive")
            else:
                inmatch = D_kill is not None and r["cum_disp"] >= D_kill
                if not inmatch:
                    cell = "pre-match"
                elif g is None:
                    cell = "n/a"
                elif g <= SHUT_BAR:
                    cell = "KILLS"
                elif g >= SURVIVE_BAR:
                    cell = "spares"
                else:
                    cell = "partial"
            ax.text(0.02, ytxt,
                    f"  {tag:<9}  +{s:<4}  {r['cum_disp']:.4f}  {dk:5.2f}  "
                    + (f"{cos:+.3f}      " if cos is not None else "  n/a      ")
                    + (f"{g:.4f}  {cell}" if g is not None
                       else f"{'n/a':>7}  {cell}"),
                    fontsize=7.2, va="top", family="monospace",
                    color=cols[tag])
            ytxt -= 0.028
        ytxt -= 0.008
    ytxt -= 0.012
    if t_kill is not None:
        ax.text(0.02, ytxt, f"t_kill=+{t_kill}  D_kill={D_kill:.4f}  "
                f"control g-12@+{t_kill}={ctrl_gm12[t_kill]:.4f}",
                fontsize=7.6, va="top", family="monospace")
        ytxt -= 0.036
    ax.text(0.02, ytxt, f"E185 VERDICT: {verdict}", fontsize=9.0, va="top",
            family="monospace", weight="bold", color="darkred")
    ytxt -= 0.04
    for wd in textwrap.wrap(clause, width=86, break_long_words=False):
        ax.text(0.02, ytxt, f"  {wd}", fontsize=6.8, va="top",
                family="monospace")
        ytxt -= 0.027

    fig.suptitle("E185 — THE NOISE-GRADIENT WASH: corpus-directed kill or "
                 f"generic optimizer fragility? -> {verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
