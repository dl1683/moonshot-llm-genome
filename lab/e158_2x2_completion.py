"""E158 — THE 2x2 COMPLETION (R46 critic attack 1: the single most important
missing cell in the lab). e151 showed a LOCKED re-teach at the NEW site 183
closes the geometry door (g-12: 0.916 -> 0.102). But "locked" and
"at-a-new-site" were ENTANGLED — the decisive cell, VARIANCE training at 183,
has never been run. Does the phase conversion key on VARIANCE or on
PLACEMENT?

THE 2x2 (variance x site; g-12 after the 300-step arm, install-60 battery):
  jitter@band   = the ROOT ITSELF by construction (e131_consolidated_e113 IS
                  300-step e113 jittered replay {-8..+8} at the home band on
                  the e048 install) -> 0.916 OPEN   [known, in-run BEFORE]
  locked@183    = e151's re-teach                     -> 0.102 SHUT  [known]
  jitter@183    = ARM (a) of this run                                 [NEW]
  locked@band   = ARM (b) of this run                                 [NEW]

ROOT (mandated): runs/checkpoints/e131_consolidated_e113.pt — the sink-
coupled, geometry-general consolidated net (row-0 census strength 0.7317 at
the trained g0 readout; e113 jitter recipe {-8..+8}; D-all g0 0.9047). Gates
below reproduce e131's stored cells bit-exact before any compute is trusted
(e151/e152 CPU precedent).

ARM (a) JITTER@183 — e151's protocol VERBATIM but with the fact's position
JITTERED across the replay windows: the e113/e143 offset-pool machinery with
the registered jitter set {-8,-4,0,+4,+8} as deltas around e151's base offset
j=+54 — name x-cols (184+d)..(190+d), onset read rows (183+d)..(189+d), d in
{-8..+8} (the dispatch's +-1-8 window; the lab's canonical variance
parameterization; T087: any variance suffices and saturates, so the exact set
is not the switch). Pool = 5 x 60 = 300 windows; per-window mask tracks the
name (7 targets). Fine-tune = e109 arm-b / e119-L recipe VERBATIM: batch 32 =
16 install windows + 16 anchors (8 paired + 8 random), e043 token-level union
CE on the name-char targets, AdamW (0.9,0.95) wd 0.1 constant lr 1e-3 clip
1.0, 300 steps, seed 10902, in-loop CPU evals every 25.

ARM (b) LOCKED@BAND — locked (zero-variance) replay at the ORIGINAL band, the
fact's home position (e119-L convention: offset-0 pool only — pre = 130 true
tokens, ZEPHYRA locked at x-cols 130..136, onset read row 129, continuation
119; the DECISION_BAND 121..137 is the home band). Same recipe, same seed
10902, same anchors, 300 steps. This isolates zero-variance from novel-site:
if locking the HOME site also closes the door, "new-site" is unnecessary for
closure; if it stays open, closure keys on novel-site teaching specifically.

MEASURE (identical instruments BEFORE(root) / AFTER(a) / AFTER(b)):
  base batteries g-12 / g0 / g+12 (install-60, e119 construction), CE_R;
  site content at 183 (span-primary census, e139/e143/e151 lineage, readout =
  the name-span read over e151's locked j=54 pool) AND at the band (span
  census over the offset-0 pool, read at rows 129..135) AND the full jitter
  span 175..197 (span census over the d=+8 pool, read at rows 191..197 — the
  only read that causally sees the whole jittered span); old-band census
  (rows 121..129 + row 0, readout ids130) => A(129) + row-0 strength;
  deletion table {none, D129, D-all{121,125,129,133,137}, D-r0, D-183} at g0;
  e150-informed probes: forced-off-sink MASK at g0/g-12 + NORM-LADDER
  0.07/0.15 (cheap; the poison-vs-information bracket).

REGISTERED PREDICTION (QUEUE.md row e158 / dispatch VERBATIM — adjudicate
against exactly this; no bar shopping; texture => TEXTURE with numbers):
  - PHASE-BY-VARIANCE fires if: (a) jitter@183 keeps the geometry door open
    (g-12 >= 0.5) AND (b) locked@band closes it (g-12 <= 0.27) — variance is
    the switch, site is irrelevant.
  - CLOSURE-BY-PLACEMENT fires if: (a) also shuts (g-12 <= 0.27) — any
    new-site teaching closes the door; the phase claim's variance clause
    dies.
  - SITE-INDEPENDENT fires if: (b) stays open (g-12 >= 0.5) — locking the
    home site is harmless; closure keys on NOVEL-SITE teaching specifically.
  - No bar shopping; texture => TEXTURE with numbers.

OPERATIONALIZATIONS (frozen here before compute):
  * g-12 = ABSOLUTE mean p(Z) on the install-60 battery at ctx offset -12
    AFTER the arm (not a retention ratio); OPEN bar >= 0.5, SHUT bar <= 0.27.
  * Known cells carried at their stored values: locked@183 = 0.1020549014210701
    (runs/e151/metrics.json after.base['-12'].mean_pz), jitter@band = the
    in-run BEFORE battery (the root by construction).
  * Adjudication order: PHASE-BY-VARIANCE -> CLOSURE-BY-PLACEMENT ->
    SITE-INDEPENDENT -> TEXTURE; every sub-boolean reported regardless;
    co-fired conditions reported (e.g. (a) shut AND (b) open fires
    CLOSURE-BY-PLACEMENT first with SITE-INDEPENDENT co-fired).
  * site_pos = e139/e151 span-primary convention: any site-band row
    content=True (e116: both drops > 0 and min/max ratio >= 0.5) AND
    strength >= 2x same-census shared-control-max; site bands: 183..189
    (census183), 175..197 (census_jit8), 121..137 (census_band).
  * MIXED/DWELL readout (T094's open question — does the mixed state persist
    under variance): mixed_state(a) = site_pos@183-span(a) AND a_open; also
    recorded over the full jitter span.

COMPUTE ENVELOPE: CPU-ONLY by dispatch (the user's game holds the GPU at
86-87C, above the 80C launch ceiling; CUDA_VISIBLE_DEVICES=-1 semantics).
A QUICK, non-blocking GPU check (util low AND temp <= 75C, double-poll 5s)
before EACH training may switch that one arm to GPU — never contention, no
bounded wait (e151's 20-min gate_launch is deliberately NOT used). CPU cap
1500 s/arm (e152 calibration: 300 CPU steps = 716 s; the two arms ~24 min).
cooldown(90) between trainings; torch threads 8; ALL readouts CPU-side,
sequential, no busy-waiting. Nets are the mandated 2.7M e131_consolidated
line (the dispatch's '<=1M family' note is an envelope statement — e143/e151
precedent; every gate reference lives on this line).

INSTRUMENT PROVENANCE: battery_cell / battery_fwd / battery_pz / ce_fixed_cpu
/ ce_fwd / val_windows / deleted_wpe / modified_wpe / read_fact_at /
row_census / finetune_arm / load_cpu / evl_load are lab/e151_twodoor.py
VERBATIM (which are e143/e109/e119/e113/e116/e068/e065/e139/e131 lineage);
forward_custom / fwd_causal / fwd_offsink are lab/e150_flatce_route.py
VERBATIM (the forced-off-sink mask). Copied, not imported, to own the device
policy. Protocol rebuild: corpus seed 1337, SPLICE_RNG 24301 host shuffle,
install60/held30 split, mix gate — e143/e151 verbatim.

Outputs: runs/e158/{metrics.json, twobytwo.png}; arm nets
runs/checkpoints/e158_jitter183.pt + e158_locked_band.pt. No NOTES/
THINKING/QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python e158_2x2_completion.py    (E158_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # GPU allowed ONLY via
# the quick per-arm check below; the dispatch default is CPU-only.

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

torch.set_num_threads(8)                              # e143/e151 convention

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT, cooldown, gpu_ok,     # noqa: E402
                    gpu_status, run_dir, save_json)
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E158_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
ROOT_CK = "e131_consolidated_e113.pt"
CKPT_DIR = E43.REPO / "runs" / "checkpoints"

# ---- arm (a) placement: e151's offset j=54 jittered by the e113 set --------
RETEACH_J = 54                    # base offset: name x-cols 184..190, rows 183..189
JIT_DELTAS = (-8, -4, 0, 4, 8)    # e109/e113 registered jitter set (dispatch: +-1-8)
ARM_A_J = tuple(RETEACH_J + d for d in JIT_DELTAS)   # {46,50,54,58,62}
SITE_ADDR_ROW = 183               # onset read row at d=0 (= e120 SPLICE_ADDR_ROW)
SITE_Z_XCOL = PRE + RETEACH_J     # 184
SITE_ROWS = tuple(range(183, 190))
JIT_ROWS_FULL = tuple(range(183 - 8, 190 + 8))       # trained read rows 175..197
P8_ADDR_ROW, P8_XCOL = SITE_ADDR_ROW + 8, SITE_Z_XCOL + 8    # +8 pool read

# ---- arm (b) placement: the home band, e119-L convention (offset 0) --------
BAND_J = 0
BAND_ADDR_ROW = PRE - 1 + BAND_J  # 129: onset read row of the home position
BAND_Z_XCOL = PRE + BAND_J        # 130: Z x-col
BAND_ROWS = tuple(range(129, 136))                    # trained read rows 129..135
DECISION_BAND = (121, 137)        # e109/e119 home band

D_ALL = (121, 125, 129, 133, 137)                     # e113's fixed set
GEOS = (-12, 0, 12)               # novel x2 + the trained g0
SHARED_CTR = (60, 100, 150, 160, 170, 200, 220)       # outside every band

ROWS_183 = (0, 1, 2) + (181, 182) + SITE_ROWS + (60, 100, 150, 160, 170)
ROWS_JIT8 = (0,) + JIT_ROWS_FULL + (60, 100, 150)
ROWS_BAND = (0,) + tuple(range(118, 138)) + (60,)
ROWS_OLD = (0, 1, 2, 3, 4, 5, 6, 60, 100, 118, 119, 120) + tuple(range(121, 130))
if SMOKE:
    ROWS_183 = (0, 1, 182) + (183, 185, 189) + (60, 100)
    ROWS_JIT8 = (0, 175, 183, 191, 197) + (60,)
    ROWS_BAND = (0, 121, 129, 135) + (60,)
    ROWS_OLD = (0, 1, 60, 121, 125, 129)

# ---- fine-tune envelope (e109 arm-b / e119-L / e143 / e151 verbatim) --------
FT_LR = 1e-3
FT_STEPS = 300 if not SMOKE else 8
EVAL_EVERY = 25 if not SMOKE else 2
NAME_BS, ANCH_BS = 16, 16
ARM_SEED = 10902                  # e119 L_SEED / e143 NEAR+FAR / e151 re-teach
COOLDOWN_S = 90.0                 # dispatch envelope 60-120 s between trainings

# ---- gates / references (full precision, from runs/e131 + runs/e151) --------
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05
G_CONS_REF_PZ = 0.7850371599197388            # e131 none__g+0__install60
G_CONS_REF_CE = 1.663516640663147             # e131 none__ce_r
G_R0_REF_MEAN = 0.7842019017236945            # e131 consolidated census row 0
G_R0_REF_ZERO = 0.7316772222270098
G_A129_REF_MEAN = -0.1084650677318375         # e131 consolidated census row 129
G_A129_REF_ZERO = -0.13237020391970877
G_DALL_REF = 0.9047248959541321               # e131 d_all_e113__g+0__install60
E151_LOCKED183_GM12 = 0.1020549014210701      # e151 after base g-12 (KNOWN cell)

# ---- e150-informed probe constants --------------------------------------------
LADDER_NORMS = (0.07, 0.15)       # the poison-probe bracket (e150's verdict)

# ---- registered bar constants (frozen) -----------------------------------------
OPEN_BAR = 0.50                   # geometry door OPEN (g-12 after)
SHUT_BAR = 0.27                   # geometry door SHUT (g-12 after)
SITE_CTRL_MULT = 2.0              # e139 site convention: >= 2x control-max

REGISTERED_PREDICTION = {
    "phase_by_variance": "PHASE-BY-VARIANCE fires if: (a) jitter@183 keeps "
        "the geometry door open (g-12 >= 0.5) AND (b) locked@band closes it "
        "(g-12 <= 0.27) — variance is the switch, site is irrelevant.",
    "closure_by_placement": "CLOSURE-BY-PLACEMENT fires if: (a) also shuts "
        "(g-12 <= 0.27) — any new-site teaching closes the door; the phase "
        "claim's variance clause dies.",
    "site_independent": "SITE-INDEPENDENT fires if: (b) stays open "
        "(g-12 >= 0.5) — locking the home site is harmless; closure keys on "
        "NOVEL-SITE teaching specifically.",
    "no_bar_shopping": "No bar shopping; texture => TEXTURE with numbers.",
    "operationalizations": "g-12 = ABSOLUTE mean p(Z) on the install-60 "
        "battery at ctx offset -12 AFTER the arm; open >= 0.5, shut <= 0.27; "
        "known cells: locked@183 = 0.1021 (e151 stored), jitter@band = the "
        "root by construction (in-run BEFORE battery); order PHASE-BY-"
        "VARIANCE -> CLOSURE-BY-PLACEMENT -> SITE-INDEPENDENT -> TEXTURE; "
        "site_pos = span-primary census, any band row content=True AND "
        "strength >= 2x same-census control-max; bands 183..189 (census183), "
        "175..197 (census_jit8), 121..137 (census_band); mixed_state(a) = "
        "site_pos@183(a) AND a_open (T094 dwell question); every sub-boolean "
        "reported regardless; co-fires reported.",
    "committed": "Committed prediction on record (T088 phase claim / QUEUE "
                 "e158): PHASE-BY-VARIANCE.",
}

trims: list[str] = []
deviations: list[str] = [
    "CPU-ONLY run by dispatch: the user's game held the GPU at 86-87C "
    "(above the 80C launch ceiling). A quick non-blocking check (util low "
    "AND temp <= 75C, double-poll 5s) before each training may switch that "
    "one arm to GPU; no bounded wait, never contention. e152 precedent: the "
    "CPU path reproduced e151's GPU arm behaviorally to <= 0.029.",
    "Nets are the mandated 2.7M e131_consolidated line (the dispatch's "
    "'<=1M family' note is an envelope statement; e143/e151 precedent — "
    "every instrument, gate reference, and lineage number of this cell "
    "lives on the 2.7M line).",
    "Arm (a) jitter set = e113's registered {-8,-4,0,+4,+8} as deltas "
    "around e151's base offset +54 — within the dispatch's +-1-8 window; "
    "T087's cliff says any variance suffices (saturated at w=1), so the "
    "exact delta set is not the switch. Pool = 300 windows (5x60, the e113 "
    "construction) vs e151's locked 60 — same batch composition (16 pool "
    "draws + 16 anchors), same seed 10902.",
    "The jitter@band cell of the 2x2 is the ROOT ITSELF by construction "
    "(e131_consolidated IS e113 jittered replay at the home band on the "
    "e048 install); reported from the in-run BEFORE battery, not re-trained.",
    "Site-span censuses: the census183 readout (span over e151's locked j=54 "
    "pool, reads 183..189) is causally blind to rows > 189, so arm (a) also "
    "carries a census over the d=+8 pool (reads 191..197 — the only read "
    "that sees the whole jittered span 175..197); the band census reads "
    "129..135 over the offset-0 pool (sees 121..135). Rows 200/220 controls "
    "dropped from the 183 census (read-invisible by construction, exactly "
    "0.0 in e151).",
    "Old-band census covers rows 121..129 only (the rows causally visible "
    "to the g0 battery read at position 129); e131's rows 130..137 cells "
    "were read-invisible by construction (exactly 0.0) and are not rerun "
    "(e151 convention).",
    "Single seed (10902), single lineage — one arm per cell, n=1; the 2x2 "
    "verdict is lineage-specific until replicated (T088's standing note).",
    "Arm (a)'s re-teach windows at offset j carry 130+j tokens of true "
    "pre-context (176..192 across the jitter pool) vs g0's 130 — the "
    "FAR-class placement-statistics confound of e143/e151, inherited "
    "verbatim from the protocol being completed (recorded, not a bar).",
    "Smoke mode trims: 8-step arms, reduced census rows, nothing adjudicated.",
    "Post-first-pass clause fix + TWO FULL PASSES, both disclosed (bars and "
    "adjudication logic UNCHANGED): pass 1 (both arms CPU — GPU user-held "
    "at 86-87C) auto-generated a factually wrong phrase in the "
    "SITE-INDEPENDENT clause (described cell (a) as under the open bar when "
    "its g-12 was above it); the clause generator was parameterized on "
    "(a)'s actual bar status and the rig was rerun end-to-end. Pass 2 = the "
    "committed run (BOTH arms GPU — the user's GPU freed before the rerun "
    "started: util 0%, temp 69C at t=0; each arm passed the dispatch's "
    "quick switch check, util <= 20 AND temp <= 75, double-poll — so the "
    "committed 2x2 is device-homogeneous with e151's locked@183 and the "
    "root's GPU lineage). The two passes' g-12 cells: (a) 0.5048 OPEN (CPU) "
    "/ 0.7885 OPEN (GPU); (b) 0.5464 OPEN (CPU) / 0.4584 MID (GPU) — cell "
    "(b) straddles the 0.5 open bar across passes and the generated verdict "
    "differs (SITE-INDEPENDENT vs TEXTURE); the committed verdict is the "
    "committed pass's bars (TEXTURE), pass 1 is reported as scatter in "
    "two_pass_disclosure, and NO further passes were run (with a "
    "straddling cell, rerunning until a preferred side appears would be "
    "bar shopping).",
]


# ------------------------------------------------------------------ device pick

def pick_dev(tag: str) -> torch.device:
    """The dispatch's quick switch check: GPU ONLY if util drops AND temp
    <= 75C on a double-poll (5 s apart). Never waits, never contends."""
    if not torch.cuda.is_available():
        return CPU
    s1 = gpu_status()
    if s1["util"] <= 20 and s1["temp"] <= 75:
        time.sleep(5)
        s2 = gpu_status()
        if s2["util"] <= 20 and s2["temp"] <= 75:
            log(f"[gpu] '{tag}' may use GPU (util {s2['util']:.0f}% temp "
                f"{s2['temp']:.0f}C)")
            return torch.device("cuda")
    log(f"[gpu] '{tag}' stays CPU (status {s1})")
    return CPU


# ------------------------------------------------------------------ instruments
# PROVENANCE: everything in this section is lab/e151_twodoor.py VERBATIM
# (which is e143/e109/e119/e113/e116/e068/e065/e139/e131 lineage; the custom
# forwards are e150's). Copied rather than imported to own the device policy.

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
    """e068/e113/e120 battery on CPU: p(Z) at the last position."""
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
    """e141's row-value surgery with e131's confinement gate (the ladder)."""
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
    """e131's read_fact_position VERBATIM ARITHMETIC, geometry parameterized:
    p(true name char) at positions addr_row..addr_row+6 over the pool windows
    (position t predicts window col t+1; the name occupies x-cols
    xcol..xcol+6). Headline = p(Z) at the onset position addr_row."""
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
    """e139's row_census_at183 VERBATIM (mean-arm / zero-arm / restore), the
    scalar readout passed as a callable."""
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
    position 0 blocked for queries 1..T-1, ALL layers, ALL heads (row 0 keeps
    its self-attention). Eval only — never trained through."""
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

def finetune_arm(tag: str, net0: TinyGPT, pool_x: torch.Tensor,
                 pool_mask: torch.Tensor, anchor: torch.Tensor,
                 train_ids: torch.Tensor, r_eval_xy, f_eval_ids, zid: int,
                 seed: int):
    """e143/e151's finetune_arm VERBATIM (e109 arm-b / e119-L recipe) with the
    device decided by the caller's quick pick_dev() (no bounded wait). Batch
    32 = 16 install windows from pool + 16 anchors (8 paired + 8 random);
    e043 token-level union CE on the name-char targets; constant lr 1e-3
    AdamW (0.9,0.95) wd 0.1 clip 1.0; 300 steps; in-loop CPU evals every 25
    (evals consume no RNG and do not touch the trajectory)."""
    dev = pick_dev(tag)
    cap = 180.0 if dev.type == "cuda" else 1500.0
    net = copy.deepcopy(net0).to(dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    n_pool = pool_x.shape[0]
    n_anc = anchor.shape[0]
    traj, t_start = [], time.time()
    step = 0
    evl = copy.deepcopy(net0)          # CPU eval twin
    for step in range(1, FT_STEPS + 1):
        ix = torch.randint(n_pool, (NAME_BS,), generator=gen)
        aj = torch.randint(n_anc, (ANCH_BS // 2,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (ANCH_BS // 2,),
                           generator=gen)
        nw = pool_x[ix].to(dev)
        anc = torch.cat([anchor[aj],
                         torch.stack([train_ids[s: s + BLOCK] for s in rj])], 0).to(dev)
        x = torch.cat([nw[:, :-1], anc[:, :-1]], 0)
        y = torch.cat([nw[:, 1:], anc[:, 1:]], 0)
        m = torch.zeros(NAME_BS + ANCH_BS, x.shape[1], dtype=torch.bool, device=dev)
        m[:NAME_BS] = pool_mask[ix].to(dev)
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
        if step % EVAL_EVERY == 0 or step == FT_STEPS or \
                (time.time() - t_start) > cap:
            evl.load_state_dict({k: v.detach().cpu().clone()
                                 for k, v in net.state_dict().items()})
            evl.eval()
            bz = battery_cell(evl, f_eval_ids, zid)
            ce_r = ce_fixed_cpu(evl, *r_eval_xy)
            traj.append({"step": step, "p_z_mean": bz["mean_pz"],
                         "frac_argmax_z": bz["frac_argmax_z"], "ce_r": ce_r,
                         "elapsed_s": round(time.time() - t_start, 1)})
            log(f"  [{tag}] s{step:4d} p_z {bz['mean_pz']:.4f} argmaxZ "
                f"{bz['frac_argmax_z']:.2f} CE_R {ce_r:.4f} "
                f"({traj[-1]['elapsed_s']:.0f}s)")
        if (time.time() - t_start) > cap:
            log(f"  [{tag}] time cap {cap:.0f}s at s{step}")
            trims.append(f"{tag}: time cap at step {step}")
            break
    net.eval()
    sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    del net
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return {"sd": sd_cpu, "traj": traj, "steps_ran": step, "seed": seed,
            "device": str(dev), "time_cap_s": cap}


CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "e158", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name} "
        f"({', '.join(f'{k}={v}' for k, v in meta.items())})")


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e158_smoke" if SMOKE else "e158")
    log(f"E158 THE 2x2 COMPLETION: variance x site (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-first (quick per-arm GPU check; gpu at start: "
        f"{gpu_status()}), threads {torch.get_num_threads()}")

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

    # ---------------- pools (the e143 offset machinery, parameterized)
    def offset_pool(j: int):
        """e143's offset-pool construction VERBATIM at total offset j:
        pre = 130+j true tokens, ZEPHYRA at x-cols (130+j)..(136+j),
        continuation 119-j; mask targets = read rows (129+j)..(135+j)."""
        wins = []
        for p, h in install_occ:
            pre = train_ids[p - PRE - j: p]
            post = train_ids[p + len(h): p + len(h) + POST_CAP - j]
            if len(pre) != PRE + j or len(post) != POST_CAP - j:
                raise RuntimeError(f"window short at p={p} j={j}")
            w = torch.cat([pre, name_ids, post])
            if len(w) != BLOCK:
                raise RuntimeError(f"window len {len(w)} != {BLOCK} at j={j}")
            wins.append(w)
        px = torch.stack(wins)
        pm = torch.zeros(len(wins), BLOCK - 1, dtype=torch.bool)
        pm[:, PRE - 1 + j: PRE - 1 + j + L] = True
        return px, pm

    # arm (b) pool: locked at the home band (offset 0, e119-L)
    pool_band_x, pool_band_mask = offset_pool(BAND_J)
    # instrument pool: e151's locked j=54 windows (census183 readout + in-loop)
    pool_183_x, pool_183_mask = offset_pool(RETEACH_J)
    # arm (a) pool: the jitter concatenation around +54 (e113 construction)
    jit_pools = {j: offset_pool(j) for j in ARM_A_J}
    pool_jit_x = torch.cat([jit_pools[j][0] for j in ARM_A_J])       # (300,256)
    pool_jit_mask = torch.cat([jit_pools[j][1] for j in ARM_A_J])
    pool_p8_x = jit_pools[RETEACH_J + 8][0]        # d=+8 instrument pool

    G_GEO = {
        "band_locked": bool(
            all(torch.equal(w[BAND_Z_XCOL: BAND_Z_XCOL + L], name_ids)
                for w in pool_band_x)
            and int(pool_band_mask[0].sum()) == L),
        "p183_locked": bool(
            all(torch.equal(w[SITE_Z_XCOL: SITE_Z_XCOL + L], name_ids)
                for w in pool_183_x)
            and int(pool_183_mask[0].sum()) == L),
        "jit_offsets": list(ARM_A_J),
        "jit_name_positions": len({int(j) for j in ARM_A_J}),
        "jit_zero_variance": False,
        "jit_all_in_place": bool(all(
            all(torch.equal(w[PRE + j: PRE + j + L], name_ids)
                for w in jit_pools[j][0]) for j in ARM_A_J)),
        "jit_mask_targets": int(pool_jit_mask[0].sum()),
        "jit_read_rows_d0": [SITE_ADDR_ROW - 8, SITE_ADDR_ROW + 6 + 8],
        "band_zero_variance": True,
        "pool_sizes": {"band": int(pool_band_x.shape[0]),
                       "jit": int(pool_jit_x.shape[0]),
                       "p183_instrument": int(pool_183_x.shape[0])},
    }
    G_GEO["pass"] = bool(G_GEO["band_locked"] and G_GEO["p183_locked"]
                         and G_GEO["jit_all_in_place"]
                         and G_GEO["jit_name_positions"] == len(ARM_A_J)
                         and G_GEO["jit_mask_targets"] == L)
    assert G_GEO["pass"], f"pool geometry gate FAILED: {G_GEO}"
    log(f"pools: band {tuple(pool_band_x.shape)} (locked, x-cols 130..136) | "
        f"jitter {tuple(pool_jit_x.shape)} (offsets {list(ARM_A_J)}, name "
        f"x-cols {SITE_Z_XCOL - 8}..{SITE_Z_XCOL + 6 + 8}) | p183 "
        f"{tuple(pool_183_x.shape)} (instrument)")

    # anchor bank (e065/e109/e143/e151 verbatim): first 16 install-position
    # ORIGINAL host windows (incumbent continuations, no ZEPHYRA)
    anchor = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                          for p, _ in install_occ[:16]])

    # ---------------- batteries (e119/e151 construction verbatim)
    bat_ids = {}
    for j in GEOS:
        for tag, occ in (("install60", install_occ), ("held30", held_occ)):
            cs = [train_text[p - PRE - j: p] for p, _ in occ]
            bat_ids[(j, tag)] = torch.stack([corpus.encode(c) for c in cs])
    ids130 = bat_ids[(0, "install60")]             # e116/e131 battery verbatim
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    feval_a = pool_183_x[:, :SITE_Z_XCOL]          # p(Z) read at row 183 (d=0)
    feval_b = ids130                               # e119-L convention

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
        raise RuntimeError("root checkpoint failed its gate vs e131 cells")

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

    # =====================================================================
    # the measurement battery (identical instruments, three calls)
    # =====================================================================
    DELS = {"none": (), "d129": (PRE - 1,), "d_all": D_ALL, "d_r0": (0,),
            "d183": (SITE_ADDR_ROW,)}
    gates_surg: dict = {}

    def site_summary(cen, band_rows, tag):
        cm = max(cen["rows"][str(r)]["strength"] for r in SHARED_CTR
                 if str(r) in cen["rows"])
        brows = [r for r in band_rows if str(r) in cen["rows"]]
        sstr = max(cen["rows"][str(r)]["strength"] for r in brows)
        spos = any(cen["rows"][str(r)]["content"] and
                   cen["rows"][str(r)]["strength"] >= SITE_CTRL_MULT * cm
                   for r in brows)
        best = max(brows, key=lambda r: cen["rows"][str(r)]["strength"])
        out = {"control_max": cm, "site_strength": sstr,
               "bar_2x_control": SITE_CTRL_MULT * cm, "site_pos": spos,
               "peak_row": int(best), "base_readout": cen["base_readout"]}
        log(f"[{tag}] site({band_rows[0]}..{band_rows[-1]}) strength "
            f"{sstr:+.4f} @r{best} (2x-ctrl {SITE_CTRL_MULT * cm:.4f}) -> "
            f"site_pos {spos} | base readout {cen['base_readout']:.4f}")
        return out

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

        # (i) site content at 183 (span-primary; e151's instrument verbatim)
        out["site_read"] = read_fact_at(net, pool_183_x, name_ids, zid,
                                        SITE_ADDR_ROW, SITE_Z_XCOL)
        out["census183_span"] = row_census(
            net, ROWS_183,
            lambda n: read_fact_at(n, pool_183_x, name_ids, zid,
                                   SITE_ADDR_ROW, SITE_Z_XCOL)["pname_mean_over7"])
        out["site_183"] = site_summary(out["census183_span"], SITE_ROWS, tag)

        # (i-b) site content over the FULL jitter span (d=+8 pool read)
        out["census_jit8_span"] = row_census(
            net, ROWS_JIT8,
            lambda n: read_fact_at(n, pool_p8_x, name_ids, zid,
                                   P8_ADDR_ROW, P8_XCOL)["pname_mean_over7"])
        out["site_jit8"] = site_summary(out["census_jit8_span"],
                                        JIT_ROWS_FULL, tag)

        # (i-c) site content at the home band (offset-0 pool, reads 129..135)
        out["census_band_span"] = row_census(
            net, ROWS_BAND,
            lambda n: read_fact_at(n, pool_band_x, name_ids, zid,
                                   BAND_ADDR_ROW, BAND_Z_XCOL)["pname_mean_over7"])
        out["site_band"] = site_summary(out["census_band_span"],
                                        tuple(range(DECISION_BAND[0],
                                                    DECISION_BAND[1] + 1)), tag)

        # (ii) old-band census (g0 readout) — the original trace + A(129)
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

        # (iii) deletion table at g0 (+ d183 texture cells)
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
            if dl == "d183":
                cell["gm12"] = battery_cell(net, bat_ids[(-12, "install60")], zid)
                cell["site_onset"] = read_fact_at(
                    net, pool_183_x, name_ids, zid, SITE_ADDR_ROW,
                    SITE_Z_XCOL)["pz_onset_mean"]
            out["del_table"][dl] = cell
        net.load_state_dict(sd)                    # restore
        log(f"[{tag}] deletions g0: " + " | ".join(
            f"{dl} {out['del_table'][dl]['g0']['mean_pz']:.3f}"
            for dl in DELS))

        # (iv) e150-informed probes at both geometries
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
    log("BEFORE battery (root e131_consolidated_e113 = the jitter@band cell "
        "BY CONSTRUCTION)")
    before = measure(sd_root, "before")

    # instrument gates against e131's stored census cells (root = before)
    G_ROW0 = {"mean": before["old_band"]["row0"]["mean"],
              "zero": before["old_band"]["row0"]["zero"],
              "ref_mean": G_R0_REF_MEAN, "ref_zero": G_R0_REF_ZERO,
              "pass": bool(abs(before["old_band"]["row0"]["mean"] - G_R0_REF_MEAN)
                           < G_FALLBACK_TOL
                           and abs(before["old_band"]["row0"]["zero"] - G_R0_REF_ZERO)
                           < G_FALLBACK_TOL)}
    G_A129 = {"mean": before["old_band"]["row129"]["mean"],
              "zero": before["old_band"]["row129"]["zero"],
              "ref_mean": G_A129_REF_MEAN, "ref_zero": G_A129_REF_ZERO,
              "pass": bool(abs(before["old_band"]["row129"]["mean"] - G_A129_REF_MEAN)
                           < G_FALLBACK_TOL
                           and abs(before["old_band"]["row129"]["zero"] - G_A129_REF_ZERO)
                           < G_FALLBACK_TOL)}
    G_DALL = {"dall_g0": before["del_table"]["d_all"]["g0"]["mean_pz"],
              "ref": G_DALL_REF,
              "pass": bool(abs(before["del_table"]["d_all"]["g0"]["mean_pz"]
                               - G_DALL_REF) < G_FALLBACK_TOL)}
    for gname, g in (("G_ROW0", G_ROW0), ("G_A129", G_A129), ("G_DALL", G_DALL)):
        log(f"{gname}: {'PASS' if g['pass'] else 'FAIL'} ({json.dumps({k: round(v, 6) if isinstance(v, float) else v for k, v in g.items() if k not in ('pass',)})})")
        if not g["pass"]:
            raise RuntimeError(f"{gname} failed vs e131 stored cells")
    root_gm12 = before["base"][-12]["mean_pz"]     # jitter@band cell (in-run)
    log(f"gates: G_SPLICE, G_GEO, G_CONS, G_MASK, G_ROW0, G_A129, G_DALL all "
        f"PASS | known cells: jitter@band(root) g-12 {root_gm12:.4f} OPEN, "
        f"locked@183(e151) g-12 {E151_LOCKED183_GM12:.4f} SHUT")

    # =====================================================================
    # THE TWO ARMS (sequential; cooldown between; CPU unless GPU freed)
    # =====================================================================
    log("=" * 78)
    if not SMOKE:
        log(f"[thermal] cooldown {COOLDOWN_S:.0f}s before arm (a)")
        cooldown(COOLDOWN_S)
    log(f"ARM (a) JITTER@183: {FT_STEPS}-step replay, name at x-cols "
        f"{SITE_Z_XCOL - 8}..{SITE_Z_XCOL + 6 + 8} across offsets "
        f"{list(ARM_A_J)} (seed {ARM_SEED})")
    arm_a = finetune_arm("jitter183", net0, pool_jit_x, pool_jit_mask, anchor,
                         train_ids, r_eval_xy, feval_a, zid, ARM_SEED)
    sd_a = arm_a["sd"]
    save_ckpt("e158_jitter183", sd_a,
              {"desc": "e131_consolidated_e113 + 300-step JITTERED replay of "
                       "ZEPHYRA around the 183 site (offsets 46..62, "
                       "e113 set {-8,-4,0,+4,+8} as deltas on +54), name-only "
                       "mask, seed 10902",
               "steps": arm_a["steps_ran"], "seed": ARM_SEED,
               "device": arm_a["device"],
               "base": f"runs/checkpoints/{ROOT_CK}"})

    if not SMOKE:
        log(f"[thermal] cooldown {COOLDOWN_S:.0f}s between trainings")
        cooldown(COOLDOWN_S)
    log(f"ARM (b) LOCKED@BAND: {FT_STEPS}-step locked replay at the home "
        f"position (x-cols 130..136, onset read row 129; seed {ARM_SEED})")
    arm_b = finetune_arm("locked_band", net0, pool_band_x, pool_band_mask,
                         anchor, train_ids, r_eval_xy, feval_b, zid, ARM_SEED)
    sd_b = arm_b["sd"]
    save_ckpt("e158_locked_band", sd_b,
              {"desc": "e131_consolidated_e113 + 300-step LOCKED replay of "
                       "ZEPHYRA at the home band (x-cols 130..136, onset read "
                       "row 129, e119-L convention), name-only mask, "
                       "seed 10902",
               "steps": arm_b["steps_ran"], "seed": ARM_SEED,
               "device": arm_b["device"],
               "base": f"runs/checkpoints/{ROOT_CK}"})

    log("=" * 78)
    log("AFTER battery, arm (a) JITTER@183")
    after_a = measure(sd_a, "after_a")
    log("=" * 78)
    log("AFTER battery, arm (b) LOCKED@BAND")
    after_b = measure(sd_b, "after_b")

    # =====================================================================
    # ADJUDICATION (registered clauses verbatim; no bar shopping)
    # =====================================================================
    gm12_a = after_a["base"][-12]["mean_pz"]
    gm12_b = after_b["base"][-12]["mean_pz"]
    a_open, a_shut = bool(gm12_a >= OPEN_BAR), bool(gm12_a <= SHUT_BAR)
    b_open, b_shut = bool(gm12_b >= OPEN_BAR), bool(gm12_b <= SHUT_BAR)

    cond = {
        "PHASE_BY_VARIANCE": {
            "a_jitter183_gm12": gm12_a, "a_open": a_open,
            "b_locked_band_gm12": gm12_b, "b_shut": b_shut,
            "open_bar": OPEN_BAR, "shut_bar": SHUT_BAR,
            "fires": bool(a_open and b_shut)},
        "CLOSURE_BY_PLACEMENT": {
            "a_jitter183_gm12": gm12_a, "a_shut": a_shut,
            "shut_bar": SHUT_BAR,
            "fires": bool(a_shut)},
        "SITE_INDEPENDENT": {
            "b_locked_band_gm12": gm12_b, "b_open": b_open,
            "open_bar": OPEN_BAR,
            "fires": bool(b_open)},
    }
    cofires = [k for k, v in cond.items() if v["fires"]]
    if cond["PHASE_BY_VARIANCE"]["fires"]:
        verdict = "PHASE-BY-VARIANCE"
        clause = (f"jitter@183 kept the geometry door OPEN (g-12 {gm12_a:.4f} "
                  f">= {OPEN_BAR}) AND locked@band closed it (g-12 "
                  f"{gm12_b:.4f} <= {SHUT_BAR}) — variance is the switch, "
                  f"site is irrelevant: the 2x2 reads {root_gm12:.3f}/"
                  f"{gm12_b:.3f} under locking (open at the variance-trained "
                  f"root, shut at home) and {gm12_a:.3f}/"
                  f"{E151_LOCKED183_GM12:.3f} at 183 (open under jitter, "
                  f"shut under locking). T088's phase claim survives its "
                  f"sharpest test.")
    elif cond["CLOSURE_BY_PLACEMENT"]["fires"]:
        verdict = "CLOSURE-BY-PLACEMENT"
        clause = (f"jitter@183 ALSO shut the door (g-12 {gm12_a:.4f} <= "
                  f"{SHUT_BAR}) — any new-site teaching closes it, variance "
                  f"or not; the phase claim's variance clause dies at the 183 "
                  f"site (locked@band g-12 {gm12_b:.4f} co-reported"
                  f"{'; SITE-INDEPENDENT co-fired (b stayed open)' if b_open else ''}).")
    elif cond["SITE_INDEPENDENT"]["fires"]:
        verdict = "SITE-INDEPENDENT"
        a_txt = (f"OPEN (>= {OPEN_BAR}: variance at the new site PREVENTED "
                 f"closure — CLOSURE-BY-PLACEMENT's 'any new-site teaching' "
                 f"clause is dead)" if a_open else
                 (f"SHUT (<= {SHUT_BAR})" if a_shut else
                  f"MID (between {SHUT_BAR} and {OPEN_BAR}) — texture with "
                  f"numbers"))
        clause = (f"locked@band left the door OPEN (g-12 {gm12_b:.4f} >= "
                  f"{OPEN_BAR}) — locking the home site is harmless; closure "
                  f"keys on NOVEL-SITE teaching specifically, and only in "
                  f"conjunction with zero-variance: e151's locked@183 "
                  f"{E151_LOCKED183_GM12:.3f} is the ONLY shutting cell of "
                  f"the 2x2; jitter@183 g-12 {gm12_a:.4f} is {a_txt} — "
                  f"neither factor alone closes the door (degradation "
                  f"texture: both new arms dragged g-12 down from the root's "
                  f"{root_gm12:.3f} to {gm12_a:.3f}/{gm12_b:.3f}, ~45%, "
                  f"without shutting it).")
    else:
        verdict = "TEXTURE"
        clause = (f"no registered bar fired cleanly: jitter@183 g-12 "
                  f"{gm12_a:.4f} (open >= {OPEN_BAR}, shut <= {SHUT_BAR}), "
                  f"locked@band g-12 {gm12_b:.4f} — the middle band; numbers "
                  f"reported, no bar shopping.")
    prediction_held = bool(verdict == "PHASE-BY-VARIANCE")

    # the four cells
    cells_2x2 = {
        "jitter_band": {"g_m12": root_gm12, "source": "root by construction "
                        "(e131 = e113 jittered replay at the home band; "
                        "in-run BEFORE battery)", "door": "OPEN" if root_gm12 >= OPEN_BAR else ("SHUT" if root_gm12 <= SHUT_BAR else "MID")},
        "locked_183": {"g_m12": E151_LOCKED183_GM12,
                       "source": "runs/e151/metrics.json after.base g-12",
                       "door": "SHUT"},
        "jitter_183": {"g_m12": gm12_a, "source": "this run arm (a)",
                       "door": "OPEN" if a_open else ("SHUT" if a_shut else "MID")},
        "locked_band": {"g_m12": gm12_b, "source": "this run arm (b)",
                        "door": "OPEN" if b_open else ("SHUT" if b_shut else "MID")},
    }

    # T094's dwell question: does the mixed state persist under variance?
    mixed_state_a = bool(after_a["site_183"]["site_pos"] and a_open)
    mixed_state_a_jit8 = bool(after_a["site_jit8"]["site_pos"] and a_open)
    dwell_read = {
        "mixed_state_a_183": mixed_state_a,
        "mixed_state_a_jitspan": mixed_state_a_jit8,
        "site_pos_183_before": before["site_183"]["site_pos"],
        "site_pos_183_after_a": after_a["site_183"]["site_pos"],
        "site_pos_jitspan_after_a": after_a["site_jit8"]["site_pos"],
        "note": "T094's dwell was measured under ZERO-variance teaching; "
                "under variance the phase claim predicts NO site-store "
                "(variance kills the graft, T088) AND an open door — a "
                "mixed state here (site_pos AND open) would extend the "
                "dwell to the variance regime; a clean no-site+open is the "
                "variance-phase signature.",
    }

    log("=" * 78)
    log(f"E158 VERDICT: {verdict} (committed PHASE-BY-VARIANCE [T088 phase "
        f"claim]: {'HELD' if prediction_held else 'FAILED'})")
    log(f"  THE 2x2 (g-12 after): jitter@band(root) {root_gm12:.4f} | "
        f"locked@band {gm12_b:.4f} | jitter@183 {gm12_a:.4f} | "
        f"locked@183 {E151_LOCKED183_GM12:.4f} (e151)")
    log(f"  bars: open >= {OPEN_BAR}, shut <= {SHUT_BAR} | cofires: {cofires}")
    log(f"  site@183 (a): {before['site_183']['site_strength']:+.4f} -> "
        f"{after_a['site_183']['site_strength']:+.4f} (site_pos "
        f"{after_a['site_183']['site_pos']}) | jit-span 175-197 peak "
        f"{after_a['site_jit8']['site_strength']:+.4f} "
        f"(site_pos {after_a['site_jit8']['site_pos']})")
    log(f"  site@band (b): {before['site_band']['site_strength']:+.4f} -> "
        f"{after_b['site_band']['site_strength']:+.4f} (site_pos "
        f"{after_b['site_band']['site_pos']}) | A(129) "
        f"{before['old_band']['A129']:+.4f} -> {after_b['old_band']['A129']:+.4f}")
    log(f"  mixed/dwell under variance: {dwell_read['mixed_state_a_183']} "
        f"(183-span) / {dwell_read['mixed_state_a_jitspan']} (jit-span)")
    log(f"  {clause}")
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e158_2x2_completion",
        "date": common.now_iso(),
        "registration": ("QUEUE.md row e158 / dispatch VERBATIM; bars frozen "
                         "before compute; T088 phase claim (variance opens "
                         "doors, zero-variance closes them) vs the placement "
                         "alternative (any new-site teaching closes)"),
        "registered_prediction": REGISTERED_PREDICTION,
        "committed_prediction": "PHASE-BY-VARIANCE (T088 phase claim)",
        "prediction_held": prediction_held,
        "question": ("does the phase conversion key on VARIANCE or on "
                     "PLACEMENT? — the 2x2 completion: jitter@183 (variance "
                     "at the new site) and locked@band (zero-variance at "
                     "home) fill the two missing cells around e151's "
                     "locked@183 and the root's jitter@band"),
        "root": f"runs/checkpoints/{ROOT_CK} (loaded, gated vs e131 cells)",
        "cells_2x2": cells_2x2,
        "arms": {
            "a_jitter183": {"desc": "e151's protocol verbatim with the fact's "
                            "position jittered (e113 set {-8,-4,0,+4,+8} as "
                            "deltas on offset +54): name x-cols 176..198 "
                            "across the pool, read rows 175..197, 300-window "
                            "pool, name-only mask",
                            "recipe": "batch 32 = 16 pool + 16 anchors (8 "
                                      "paired + 8 random), token-weighted "
                                      "union CE, AdamW (0.9,0.95) wd 0.1, "
                                      "lr 1e-3 constant, clip 1.0",
                            "steps_ran": arm_a["steps_ran"],
                            "seed": ARM_SEED, "traj": arm_a["traj"],
                            "device": arm_a["device"],
                            "time_cap_s": arm_a["time_cap_s"]},
            "b_locked_band": {"desc": "locked (zero-variance) replay at the "
                              "fact's home position (e119-L convention): "
                              "ZEPHYRA locked at x-cols 130..136, onset read "
                              "row 129, 60-window pool, name-only mask",
                              "recipe": "identical to arm (a) (same seed "
                                        "stream, same anchors)",
                              "steps_ran": arm_b["steps_ran"],
                              "seed": ARM_SEED, "traj": arm_b["traj"],
                              "device": arm_b["device"],
                              "time_cap_s": arm_b["time_cap_s"]},
        },
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
                     "placement_a": {"base_offset": RETEACH_J,
                                     "jitter_deltas": list(JIT_DELTAS),
                                     "total_offsets": list(ARM_A_J),
                                     "name_xcols": [SITE_Z_XCOL - 8,
                                                    SITE_Z_XCOL + 6 + 8],
                                     "read_rows": [175, 197],
                                     "pool": 300},
                     "placement_b": {"offset": BAND_J,
                                     "name_xcols": [BAND_Z_XCOL,
                                                    BAND_Z_XCOL + 6],
                                     "read_rows": [129, 135],
                                     "decision_band": list(DECISION_BAND),
                                     "pool": 60, "zero_variance": True},
                     "mask": "7 name-char targets per window (name-only)",
                     "geometries_measured": {"novel": [-12, 12],
                                             "trained_g0": 0,
                                             "note": "root trained "
                                                     "{-8,-4,0,+4,+8}; +-12 "
                                                     "novel for it"}},
        "gates": {"G_SPLICE": G_SPLICE, "G_GEO": G_GEO, "G_CONS": G_CONS,
                  "G_MASK": G_MASK, "G_ROW0": G_ROW0, "G_A129": G_A129,
                  "G_DALL": G_DALL, "G_SURG": gates_surg,
                  "gpu_at_start": {"cuda_visible": bool(torch.cuda.is_available()),
                                   "status": gpu_status(),
                                   "note": "CPU-only by dispatch at tasking "
                                           "time (GPU user-held at 86-87C, "
                                           "above the 80C ceiling); if the "
                                           "GPU frees, the per-arm quick "
                                           "check (util <= 20 AND temp <= "
                                           "75C, double-poll) may switch an "
                                           "arm — in the committed pass the "
                                           "GPU was free from the start and "
                                           "BOTH arms ran GPU"}},
        "before": before, "after_a": after_a, "after_b": after_b,
        "adjudication": {"conditions": cond, "cofires": cofires,
                         "verdict": verdict, "clause": clause,
                         "bars": {"open": OPEN_BAR, "shut": SHUT_BAR,
                                  "measure": "absolute g-12 mean p(Z) "
                                             "install-60 after the arm"},
                         "committed_prediction": "PHASE-BY-VARIANCE",
                         "prediction_held": prediction_held},
        "dwell_readout": dwell_read,
        "two_pass_disclosure": {
            "what": "two full end-to-end passes were run; the committed "
                    "artifacts are pass 2's. Documentation-only record — "
                    "no measurement, bar, or verdict below was altered.",
            "pass1": {"devices": "a: cpu / b: cpu", "gm12_a": 0.5048,
                      "gm12_b": 0.5464,
                      "generated_verdict": "SITE-INDEPENDENT",
                      "discarded_reason": "clause-prose bug (cell (a) "
                                          "misdescribed as under the open "
                                          "bar); bars/logic unchanged"},
            "pass2_committed": {"devices": "a: cuda / b: cuda (the user's "
                                "GPU freed before the rerun: util 0%, temp "
                                "69C at t=0; both arms passed the dispatch's "
                                "quick switch check — device-homogeneous "
                                "with e151's cell and the root's lineage)",
                                "gm12_a": None, "gm12_b": None,
                                "generated_verdict": None,
                                "note": "filled by the run; see "
                                        "cells_2x2 / adjudication"},
            "composite_reading": "neither variance alone (a OPEN in both "
                                 "passes: 0.505/0.789) nor home-locking "
                                 "alone (b 0.46-0.55, never shut) closes "
                                 "the door; both degrade it (~50% of the "
                                 "root's 0.916); the only SHUT cell is "
                                 "locked@183 — closure requires the "
                                 "CONJUNCTION novel-site x zero-variance. "
                                 "The committed pass's bars give TEXTURE "
                                 "(b mid); pass 1's SITE-INDEPENDENT is "
                                 "scatter, not shopping; replicate before "
                                 "canonizing (b)'s side of the 0.5 bar.",
        },
        "honesty_reflex": {
            "single_seed_single_lineage": "one arm per cell (seed 10902) on "
                "one root (e131 consolidated, one seed) — each new cell is "
                "n=1; the 2x2 verdict is lineage-specific until replicated "
                "(T088's standing note)",
            "jitter_magnitude_choice": "the e113 registered set "
                "{-8,-4,0,+4,+8} as deltas (within the dispatch's +-1-8); "
                "T087 says the cliff is saturated at ANY variance (w=1 "
                "already negative A, NR onset), so the set choice is not "
                "the switch — but the 183-site jitter is offset-54-based "
                "(FAR-class context statistics), and no finer-grain (e.g. "
                "+-1 only) arm was run",
            "cpu_float_path": "pass 1 ran both arms CPU (GPU user-held "
                "86-87C); the GPU freed before pass 2, which ran BOTH arms "
                "GPU — the committed 2x2 is device-homogeneous with e151's "
                "cell and the root's lineage. e152 precedent: cross-device "
                "behavioral drift ~0.03 on g-12 (site-census up to 0.111); "
                "OBSERVED ACROSS PASSES, larger: cell (a) 0.505 (CPU) vs "
                "0.789 (GPU) — both OPEN, scatter 0.28 (CPU pass under user "
                "game load; run-to-run CPU nondeterminism cannot be "
                "separated from the device effect with n=1 each); cell (b) "
                "0.546 (CPU) vs 0.458 (GPU) — straddles the 0.5 bar; see "
                "two_pass_disclosure.",
            "budget_vs_consolidation": "300 steps per arm, step-matched to "
                "the root's consolidation and e151's re-teach; a longer arm "
                "could convert more (the dwell decays with steps, T094)",
            "known_cells_carry": "locked@183 is carried at e151's stored "
                "value (GPU-trained) while the two new cells are CPU-trained "
                "— device asymmetry inside the 2x2, priced by e152's "
                "behavioral repro (<= 0.029 on g-12)",
            "mask_off_distribution": "the forced-off-sink mask puts every "
                "context off its training distribution (e150's caveat) — "
                "the CE column prices the generic part; the BEFORE net's "
                "mask cells are the in-run differential control",
            "norm_ladder_is_poison_not_information": "per T086/E150, ladder "
                "kills measure SINK-HEALTH (poisoning), not routing; a "
                "differential there is texture, not a route readout",
            "bar_marginal_cells": "OBSERVED: cell (b) locked@band straddles "
                "the 0.5 open bar across the two passes (0.5464 CPU pass 1 "
                "/ 0.4584 GPU pass 2); the committed pass adjudicates it MID "
                "-> the TEXTURE branch, pass 1's generated verdict was "
                "SITE-INDEPENDENT. Cell (a) jitter@183 is OPEN in both "
                "passes (0.5048 / 0.7885). Composite across passes: neither "
                "factor alone SHUTS the door (a open twice; b never shut, "
                "0.46-0.55); both degrade it to ~50% of the root's 0.916; "
                "the only SHUT cell of the 2x2 remains locked@183 (0.102, "
                "e151) — closure requires the CONJUNCTION novel-site x "
                "zero-variance. Replicate (second seed, matched device) "
                "before canonizing (b)'s open-vs-mid classification; the "
                "conjunction reading is robust to the scatter.",
        },
        "trims": trims, "deviations": deviations,
        "ckpt_inventory": CKPT_INVENTORY,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": int(net0.num_params()),
                   "train_device": f"a:{arm_a['device']}/b:{arm_b['device']}",
                   "eval_device": "cpu",
                   "torch_threads": torch.get_num_threads(), "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "twobytwo.png", before, after_a, after_b, cells_2x2, cond,
         verdict, clause, prediction_held, dwell_read)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'twobytwo.png'}, ckpts "
        f"runs/checkpoints/e158_jitter183.pt + e158_locked_band.pt")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot

def plot(path, before, after_a, after_b, cells, cond, verdict, clause,
         prediction_held, dwell_read):
    """The 2x2 grid (variance x site) + before/after dials per arm."""
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.5))

    # (0,0) THE 2x2 GRID — the headline
    ax = axes[0, 0]
    grid = {(0, 0): ("jitter@band\n(root by construction)", cells["jitter_band"], "tab:blue"),
            (0, 1): ("jitter@183\nARM (a) — the missing cell", cells["jitter_183"], "crimson"),
            (1, 0): ("locked@band\nARM (b)", cells["locked_band"], "crimson"),
            (1, 1): ("locked@183\n(e151, known)", cells["locked_183"], "gray")}
    for (vr, sr), (lbl, c, col) in grid.items():
        x, y = sr, 1 - vr
        v = c["g_m12"]
        open_ = v >= OPEN_BAR
        shut = v <= SHUT_BAR
        fc = "seagreen" if open_ else ("dimgray" if shut else "gold")
        ax.add_patch(plt.Rectangle((x - 0.44, y - 0.40), 0.88, 0.80,
                                   facecolor=fc, alpha=0.22, edgecolor="k",
                                   lw=0.8))
        ax.text(x, y + 0.20, lbl, ha="center", va="center", fontsize=8.6,
                fontweight="bold")
        ax.text(x, y - 0.02, f"g-12 = {v:.3f}", ha="center", va="center",
                fontsize=12.5, family="monospace", fontweight="bold",
                color=col)
        ax.text(x, y - 0.22, "OPEN" if open_ else ("SHUT" if shut else "MID"),
                ha="center", va="center", fontsize=9,
                color="seagreen" if open_ else ("dimgray" if shut else "darkgoldenrod"))
    ax.axhline(0.5, color="k", lw=0.7)
    ax.axvline(0.5, color="k", lw=0.7)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["HOME band\n(x-cols 130..136)", "NEW site 183\n(x-cols 184..190)"], fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["locked\n(zero-var)", "jitter\n(VARIANCE)"], fontsize=9)
    ax.set_title("THE 2x2 — variance x site (g-12 after the 300-step arm;\n"
                 f"open >= {OPEN_BAR} green, shut <= {SHUT_BAR} gray)",
                 fontsize=10)

    # (0,1) g-12 / g+12 / g0 dials before vs after, both arms
    ax = axes[0, 1]
    xs = np.arange(3)
    labels = ["g-12 (NOVEL)", "g0 (trained)", "g+12 (NOVEL)"]
    for k, j in enumerate((-12, 0, 12)):
        b = before["base"][j]["mean_pz"]
        aa = after_a["base"][j]["mean_pz"]
        ab = after_b["base"][j]["mean_pz"]
        ax.bar(k - 0.25, b, 0.24, color="steelblue", edgecolor="k", lw=0.5,
               label="before (root)" if k == 0 else None)
        ax.bar(k, aa, 0.24, color="crimson", edgecolor="k", lw=0.5,
               label="after (a) jitter@183" if k == 0 else None)
        ax.bar(k + 0.25, ab, 0.24, color="darkorange", edgecolor="k", lw=0.5,
               label="after (b) locked@band" if k == 0 else None)
        for x0, v in ((k - 0.25, b), (k, aa), (k + 0.25, ab)):
            ax.text(x0, v + 0.012, f"{v:.3f}", ha="center", fontsize=7)
    ax.axhline(OPEN_BAR, ls="--", lw=1.1, color="seagreen")
    ax.axhline(SHUT_BAR, ls="--", lw=1.1, color="gray")
    ax.text(2.45, OPEN_BAR + 0.01, f"open {OPEN_BAR}", fontsize=7, color="seagreen", ha="right")
    ax.text(2.45, SHUT_BAR + 0.01, f"shut {SHUT_BAR}", fontsize=7, color="gray", ha="right")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("battery p(Z) install-60")
    ax.set_ylim(0, 1.12)
    ax.set_title("GEOMETRY DOOR DIALS — before vs both arms", fontsize=9.5)
    ax.legend(fontsize=7, loc="lower right")

    # (0,2) site-content spectra: census183 span (before/a) + census_band (before/b)
    ax = axes[0, 2]
    for tag, d, col, mk in (("before", before, "steelblue", "o"),
                            ("after_a", after_a, "crimson", "s")):
        cen = d["census183_span"]["rows"]
        xs_r = [int(r) for r in cen if int(r) not in SHARED_CTR]
        ax.plot(xs_r, [cen[str(r)]["strength"] for r in xs_r], f"{mk}-",
                ms=4, lw=1.2, color=col, alpha=0.9,
                label=f"{tag} (183-span read)")
    for tag, d, col, mk in (("before", before, "steelblue", "o"),
                            ("after_b", after_b, "darkorange", "D")):
        cen = d["census_band_span"]["rows"]
        xs_r = [int(r) for r in cen if int(r) not in SHARED_CTR and int(r) >= 118]
        ax.plot(xs_r, [cen[str(r)]["strength"] for r in xs_r], f"{mk}--",
                ms=3.5, lw=1.0, color=col, alpha=0.75,
                label=f"{tag} (band-span read)")
    ax.axvspan(183, 189, color="crimson", alpha=0.07)
    ax.axvspan(121, 137, color="seagreen", alpha=0.06)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("wpe row (red: 183 site; green: home band 121-137)")
    ax.set_ylabel("strength = min(mean-drop, zero-drop)")
    ax.set_title("SITE CONTENT at 183 (arm a) and at the band (arm b)",
                 fontsize=9.5)
    ax.legend(fontsize=6.4)

    # (1,0) e150-informed probes
    ax = axes[1, 0]
    rows = []
    for tag, d in (("bef", before), ("a", after_a), ("b", after_b)):
        for j in (0, -12):
            c = d["mask"][f"g{j:+d}"]
            rows.append((f"{tag} mask@g{j:+d}", c["retention"] * 100,
                         c["ce_cost"], "tab:red"))
            for e in d["ladder"]:
                b = d["base"][j]["mean_pz"]
                rows.append((f"{tag} |wpe0|={e['target_norm']:.2f}@g{j:+d}",
                             100 * e[f"g{j:+d}"] / max(b, 1e-12),
                             e["ce_r"] - d["ce_r"], "tab:blue"))
    ys = np.arange(len(rows))
    ax.barh(ys, [r[1] for r in rows], 0.62, color=[r[3] for r in rows],
            edgecolor="k", lw=0.4)
    for y, r in zip(ys, rows):
        ax.text(max(r[1], 0) + 1.5, y, f"{r[1]:.0f}%  CE{r[2]:+.2f}",
                va="center", fontsize=6.2)
    ax.axvline(100 * OPEN_BAR, ls="--", color="seagreen", lw=1.1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xlim(0, 145)
    ax.set_xlabel("fact retention (%) — red = mask (info probe), blue = ladder (poison)")
    ax.set_title("(e150 probes) mask should SPARE; 0.07 kill by poison; 0.15 survive",
                 fontsize=9)

    # (1,1) deletion table + old-trace panel
    ax = axes[1, 1]
    dls = ("none", "d129", "d_all", "d_r0", "d183")
    xs = np.arange(len(dls))
    for k, (tag, d, col) in enumerate((("before", before, "steelblue"),
                                       ("after_a", after_a, "crimson"),
                                       ("after_b", after_b, "darkorange"))):
        vals = [d["del_table"][dl]["g0"]["mean_pz"] for dl in dls]
        ax.bar(xs + (k - 1) * 0.26, vals, 0.24, color=col, edgecolor="k",
               lw=0.4, label=tag)
        for x, v in zip(xs + (k - 1) * 0.26, vals):
            ax.text(x, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.2,
                    rotation=90, va="bottom")
    ax.text(-0.4, 0.62,
            f"A(129): bef {before['old_band']['A129']:+.3f} | "
            f"a {after_a['old_band']['A129']:+.3f} | "
            f"b {after_b['old_band']['A129']:+.3f}\n"
            f"row0 S: bef {before['old_band']['row0_strength']:+.3f} | "
            f"a {after_a['old_band']['row0_strength']:+.3f} | "
            f"b {after_b['old_band']['row0_strength']:+.3f}\n"
            f"site@183 span: bef {before['site_183']['site_strength']:+.4f} -> "
            f"a {after_a['site_183']['site_strength']:+.4f} "
            f"(pos {after_a['site_183']['site_pos']})\n"
            f"site@jit-span: a {after_a['site_jit8']['site_strength']:+.4f} "
            f"(pos {after_a['site_jit8']['site_pos']}, peak "
            f"{after_a['site_jit8']['peak_row']})\n"
            f"site@band: bef {before['site_band']['site_strength']:+.4f} -> "
            f"b {after_b['site_band']['site_strength']:+.4f} "
            f"(pos {after_b['site_band']['site_pos']})\n"
            f"CE_R: bef {before['ce_r']:.3f} | a {after_a['ce_r']:.3f} | "
            f"b {after_b['ce_r']:.3f}\n"
            f"mixed/dwell under variance: 183={dwell_read['mixed_state_a_183']} "
            f"jit-span={dwell_read['mixed_state_a_jitspan']}",
            fontsize=6.8, family="monospace",
            bbox=dict(facecolor="lightyellow", alpha=0.9, edgecolor="gray"))
    ax.set_xticks(xs)
    ax.set_xticklabels(["none", "D129", "D-all\n{121..137}", "D-row-0",
                        "D-183"], fontsize=8)
    ax.set_ylabel("battery p(Z) g0 install-60")
    ax.set_ylim(0, 1.15)
    ax.set_title("deletions at g0 + old-trace / site panel", fontsize=9.5)
    ax.legend(fontsize=7, loc="upper right")

    # (1,2) verdict panel
    ax = axes[1, 2]
    ax.axis("off")
    vlines = [
        "REGISTERED (QUEUE e158 verbatim; no bar shopping):",
        "  PHASE-BY-VARIANCE: (a) open (>=0.5) AND (b) shut (<=0.27)",
        "  CLOSURE-BY-PLACEMENT: (a) also shuts (<=0.27)",
        "  SITE-INDEPENDENT: (b) stays open (>=0.5)",
        "",
        "THE FOUR CELLS (g-12 after):",
        f"  jitter@band  (root by construction): {cells['jitter_band']['g_m12']:.4f}  OPEN",
        f"  locked@band  (ARM b, this run):      {cells['locked_band']['g_m12']:.4f}  {cells['locked_band']['door']}",
        f"  jitter@183   (ARM a, this run):      {cells['jitter_183']['g_m12']:.4f}  {cells['jitter_183']['door']}",
        f"  locked@183   (e151, known):          {cells['locked_183']['g_m12']:.4f}  SHUT",
        "",
        f"cofires: {[k for k, v in cond.items() if v['fires']]}",
        f"mixed/dwell under variance: 183-span={dwell_read['mixed_state_a_183']}"
        f" jit-span={dwell_read['mixed_state_a_jitspan']}",
        "",
        f"VERDICT: {verdict}  (committed PHASE-BY-VARIANCE "
        f"[T088]: {'HELD' if prediction_held else 'FAILED'})",
    ] + [f"  {wd}" for wd in
         [clause[i:i + 64] for i in range(0, len(clause), 64)]]
    for i, tx in enumerate(vlines):
        ax.text(0.02, 0.97 - i * 0.052, tx, fontsize=7.0, va="top",
                family="monospace")

    fig.suptitle(f"E158 — THE 2x2 COMPLETION: variance x placement "
                 f"(root e131_consolidated + 300-step arms) -> {verdict}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
