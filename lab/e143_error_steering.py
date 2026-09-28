"""E143 — ERROR-PLACEMENT STEERING (T076's only causal test; the T079
proximity-vs-invariance fork, PRE-REGISTERED in THINKING.md's T079 card at
~07:58Z BEFORE this dispatch — the fork bars below are VERBATIM from that
pre-registration; adjudicate against exactly this, no bar shopping).

WHY: T076 says THE FACT CONSOLIDATES WHERE ITS TRAINING ERROR IS PLACED, but
every support so far is observational (e120 splice, e131 blindness, e139
taxonomy). The one causal cell never run: park the error NEXT TO the
omnipresent sink row (positions 5-13, zero position diversity) and ask
whether proximity alone inherits row-0 routing (W011's mechanism) or whether
T079's invariance clause (routing forms only when the positional key LOSES
the credit competition under position diversity) is genuinely necessary.

ROOT (mandated): runs/checkpoints/e048_repro.pt — the install-phase net of
the e113/e119/e131 line (corpus seed 1337, SPLICE_RNG 24301, install60
battery p(Z) g0 = 0.556313; row-0 presence-strength install baseline
0.5455324, gated vs e131's stored value at 5e-6). ALL THREE arms start from
this one bit-identical net; every arm difference is placement-caused.

ARMS (all = e119's LOCKED-replay protocol = e109 arm-b fine-tune recipe:
batch 32 = 16 install windows + 16 anchor (8 paired + 8 random), e043
token-level union CE on the 16x7 name-char targets, AdamW (0.9,0.95) wd 0.1
constant lr 1e-3 clip 1.0, 300 steps, <=180 s GPU wall, in-loop evals every
25 steps):
  NEAR   — install windows TRUNCATED so ZEPHYRA always occupies x-cols 6..12
           (pre-context = the LAST 6 corpus tokens before the host slot;
           continuation extended to 243 tokens so the window stays 256).
           The name's read runs through wpe rows 5..11 — just past the sink
           row 0, no position diversity. THE PLACEMENT TREATMENT. Seed 10902.
  FAR    — the e109 +8-offset pool VERBATIM (pre = 138 tokens, ZEPHYRA at
           x-cols 138..144, read rows 137..143, continuation 111 tokens).
           Identical budget/steps/mask; fact locked away from the sink.
           Seed 10902 (same RNG stream as NEAR: the two arms differ ONLY in
           the placement treatment, batch-index for batch-index).
  JITTER — the e113 recipe VERBATIM on the same root (pool of 300 windows at
           offsets {-8,-4,0,+4,+8}, seed 10901, 300 steps) — the known
           ROUTED reference (row-0 strength 0.732, novel-geometry ~0.8,
           D-all survivor per e113/e119/e131).

READOUTS per arm (all CPU-side on state-dict snapshots):
  (i)   SITE CONTENT TEST at the trained site — e139's site-test instrument
        (row_census_at183 lineage: per-row mean-arm [wpe[r] <- mean of all
        rows] / zero-arm [wpe[r] <- 0] / restore; strength = min(mean-drop,
        zero-drop); e116 criterion content = both drops > 0 and min/max
        ratio >= 0.5), readout at the arm's OWN trained geometry (e139's
        registered readout adaptation), on the arm's own training pool.
        Site bands: NEAR rows 5..13 (trained read rows 5..11 + 2 margin),
        FAR rows 137..143, JITTER band 121..137 (readout at the g0 battery,
        e131's Phase-C convention for the jitter line). Control band =
        shared rows {60,100,150,160,170,200,220} (outside every trained
        band; e131's CONTROL_ROWS rows 1-6 overlap NEAR's trained site and
        cannot serve as NEAR controls).
  (ii)  ROW-0 PRESENCE-DEPENDENCE — the e131 instrument (T081: the content
        test measures presence-necessity): row 0's mean/zero-arm strength,
        PRIMARY at the arm's trained-geometry readout (e139 adaptation),
        SECONDARY at the e131-verbatim install-60 g0 battery readout.
        Install baseline = 0.5455324 (e131 stored install-phase row-0
        strength; gated in-run at 5e-6).
  (iii) D-all{121,125,129,133,137} battery at install-60 g0 (D2 subtractive
        row-zero + confinement gate; none / D129 / D-all / D-row-0 cells).
  (iv)  NOVEL-GEOMETRY read (g-12 headline; -2/+2/+12 texture) — the routed
        type's signature (e119 R: ~0.80 at all four under none AND D-all).

REGISTERED PREDICTION (VERBATIM, THINKING.md T079 card, ~07:58Z):
  - COMPASS-CAUSAL fires if: NEAR consolidates site-stored at 5-13 (site
    content-positive there; row-0 presence-dependence at install baseline)
    — invariance is necessary for routing; T079 survives.
  - PROXIMITY-PIGGYBACK fires if: NEAR becomes row-0-routed
    (presence-dependence >= 2x install baseline) while FAR stays
    site-stored — proximity inherits the route; T079's invariance clause
    dies (W011's mechanism wins).
  - UNIFORM fires if: NEAR ~= FAR everywhere.
  - Committed prediction on record: COMPASS-CAUSAL. No bar shopping;
    texture => TEXTURE with numbers.

OPERATIONALIZATIONS (frozen here before compute):
  * site content-positive = e116 criterion AND strength >= 2x shared-
    control-max (e139's site convention); site_str(arm) = max strength over
    the arm's site band; site_pos(arm) = any site-band row content-positive
    at >= 2x control-max.
  * r0_str(arm) = row-0 strength, min(mean-drop, zero-drop), at the arm's
    trained-geometry readout (primary). R0_BASE = 0.5455324279477395.
  * PIGGY bar (literal): r0_str(NEAR) >= 2 * R0_BASE = 1.0911 — kept
    verbatim even though it sits at/above the readout's arithmetic ceiling
    when the arm's base p(Z) approaches 1.0; a near-miss is reported as
    numbers, never re-barred.
  * "row-0 presence-dependence at install baseline" (COMPASS clause) =
    r0_str(NEAR) <= R0_BASE + 0.5 * (r0_str(JITTER) - R0_BASE) — NEAR must
    sit in the LOWER HALF of the install->routed span, the routed reference
    measured in-run (e131/e139 measured install 0.545 vs routed 0.732).
  * "FAR stays site-stored" (PIGGY clause) = site_pos(FAR) at 137..143 AND
    r0_str(FAR) <= the same midpoint bar.
  * UNIFORM = |NEAR - FAR| <= 0.10 on all five aggregates {own-geometry
    p(Z), r0_str, site_str, D-all g0, g-12} AND the census peak rows
    (argmax strength over the censused region) of NEAR and FAR coincide —
    placement inert. Differing peak rows = the placement steered, not
    uniform, however similar the aggregates.
  * Adjudication order: COMPASS-CAUSAL -> PROXIMITY-PIGGYBACK -> UNIFORM ->
    TEXTURE (with numbers).

COMPUTE ENVELOPE: GPU for the three fine-tunes (idle RTX 5090; gpu_ok()
double-poll with a 20-min bounded wait then PARK; NO concurrent GPU — e140
owns the CPU and is not touched; cooldown(120) between launches; each
training <= 180 s; batch 32). ALL readouts CPU-side (8 threads, lab
convention). Nets are the mandated 2.7M e048_repro line (the tasking's
"0.84M family" note is the ctx-512 rig's envelope — every cited instrument
and reference number of this fork lives on the 2.7M line; recorded as a
deviation). Single seed per arm (honesty note: no seed replication).

Outputs: runs/e143/{metrics.json, error_steering.png}; phase nets
runs/checkpoints/e143_{near,far,jitter}.pt (listed in ckpt_inventory).
No NOTES/THINKING/QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python e143_error_steering.py    (E143_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import random
import sys
import time
from pathlib import Path

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")     # GPU for fine-tunes

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

torch.set_num_threads(8)                              # CPU-considerate (e140 runs)

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT, cooldown, gpu_ok,     # noqa: E402
                    gpu_status, run_dir, save_json)
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E143_SMOKE") == "1"
CPU = torch.device("cpu")
# GPU is permitted but must pass gpu_ok() at startup; else PARK to CPU.
_USE_GPU = torch.cuda.is_available() and gpu_ok()
DEV = torch.device("cuda") if _USE_GPU else CPU

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
INSTALLED_CK = "e048_repro.pt"
CKPT_DIR = E43.REPO / "runs" / "checkpoints"

# ---- placement geometry ----------------------------------------------------
NEAR_PRE = 6                      # ZEPHYRA x-cols 6..12; read rows 5..11
NEAR_ADDR_ROW = NEAR_PRE - 1      # 5: wpe row predicting 'Z'
NEAR_Z_XCOL = NEAR_PRE            # 6
NEAR_CONT = BLOCK - NEAR_PRE - len(NAME)          # 243-token continuation
NEAR_SITE_ROWS = tuple(range(5, 14))              # trained site band (dispatch)
FAR_J = 8                         # e109's +8 offset pool verbatim
FAR_ADDR_ROW = PRE - 1 + FAR_J                    # 137
FAR_Z_XCOL = PRE + FAR_J                          # 138
FAR_SITE_ROWS = tuple(range(137, 144))            # read rows 137..143
JITTERS = (-8, -4, 0, 4, 8)       # e109/e113 registered jitter set
GEO_ORDER = [-8, -4, 0, 4, 8]
JIT_SITE_ROWS = tuple(range(121, 138))            # e119 DECISION_BAND
D_ALL = (121, 125, 129, 133, 137)                 # e113's D-all set
NOVEL_GEO = (-12, -2, 2, 12)                      # e119 novel set (g-12 headline)

SHARED_CTR = (60, 100, 150, 160, 170, 200, 220)   # outside every trained band
NEAR_ROWS = (0, 1, 2) + tuple(range(3, 18)) + SHARED_CTR      # 1,2 = scaffold ref
FAR_ROWS = (0, 1, 2) + (134, 135, 136) + tuple(range(137, 147)) + SHARED_CTR
JIT_ROWS = (0,) + tuple(range(121, 138)) + (1, 2, 3, 4, 60, 100, 118, 119, 120) + SHARED_CTR

# ---- fine-tune envelope (e109 arm-b / e119 L protocol verbatim) -------------
FT_LR = 1e-3
FT_STEPS = 300 if not SMOKE else 8
FT_TIME_CAP = 180.0 if _USE_GPU else 1500.0        # GPU thermal / CPU safety
EVAL_EVERY = 25 if not SMOKE else 2
NAME_BS, ANCH_BS = 16, 16
NEAR_SEED = FAR_SEED = 10902                       # e119 L_SEED; same stream
JIT_SEED = 10901                                   # e113 CONS_SEED verbatim

# ---- gates / references -----------------------------------------------------
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
G_INST_REF = 0.556313             # e065/e091/e113/e119 install battery ref
G_INST_TOL = 0.005
R0_BASE_REF = 0.5455324279477395  # e131 stored install-phase row-0 strength
R0_MEAN_REF = 0.5549507188142645  # e131 stored install row-0 mean-arm drop
R0_TOL = 5e-6                     # e131 G_E116 convention
E109_REF_NONE = {-8: 0.9924036860466003, -4: 0.9496323466300964,
                 0: 0.776076078414917, 4: 0.9848979115486145,
                 8: 0.9854525923728943}
G_JREPRO_TOL = 0.05               # report-only (GPU float nondeterminism)
UNIFORM_BAND = 0.10

GPU_WAIT_MAX_S, GPU_POLL_S = 1200.0, 30.0

REGISTERED_PREDICTION = {
    "compass_causal": "COMPASS-CAUSAL fires if: NEAR consolidates site-stored "
                      "at 5-13 (site content-positive there; row-0 "
                      "presence-dependence at install baseline) — invariance "
                      "is necessary for routing; T079 survives.",
    "proximity_piggyback": "PROXIMITY-PIGGYBACK fires if: NEAR becomes "
                           "row-0-routed (presence-dependence >= 2x install "
                           "baseline) while FAR stays site-stored — proximity "
                           "inherits the route; T079's invariance clause dies "
                           "(W011's mechanism wins).",
    "uniform": "UNIFORM fires if: NEAR ~= FAR everywhere.",
    "committed": "Committed prediction on record: COMPASS-CAUSAL. No bar "
                 "shopping; texture => TEXTURE with numbers.",
    "operationalizations": "site_pos = e116 criterion AND strength >= 2x "
                           "shared-control-max (e139 site convention); r0_str "
                           "= min(mean-drop, zero-drop) at the arm's "
                           "trained-geometry readout (primary); PIGGY bar "
                           "literal r0_str(NEAR) >= 2x0.5455324 = 1.0911 "
                           "(kept verbatim though near the readout ceiling); "
                           "'at install baseline' = r0_str(NEAR) <= R0_BASE + "
                           "0.5*(r0_str(JITTER)-R0_BASE); FAR site-stored = "
                           "site_pos(FAR) AND r0_str(FAR) below the same "
                           "midpoint; UNIFORM = |NEAR-FAR| <= 0.10 on all five "
                           "aggregates AND census peak rows coincide; order "
                           "COMPASS -> PIGGY -> UNIFORM -> TEXTURE.",
}

trims: list[str] = []
deviations: list[str] = [
    "Placement construction: NEAR truncates the e043 install windows to a "
    "6-token pre-context (name x-cols 6..12, read rows 5..11) and extends the "
    "host continuation to 243 tokens to keep the 256-token window — the "
    "e068 left-retraction limit of the jitter machinery (j = -124). Side "
    "effect recorded honestly: NEAR's context statistics differ (shorter "
    "left context, longer continuation); the name-target mask stays exactly "
    "7 targets/window like every arm.",
    "NEAR and FAR share seed 10902 (e119's L_SEED) and pool size 60, so "
    "their RNG streams (install draws, anchor draws, random windows) are "
    "identical draw-for-draw — the arms differ ONLY in the placement "
    "treatment. JITTER keeps e113's seed 10901 verbatim.",
    "In-loop eval ids: JITTER uses the install-60 g0 battery (e113 "
    "convention verbatim); NEAR/FAR use their own trained-geometry context "
    "sets (diagnostics only — evals consume no RNG and do not touch the "
    "gradient trajectory, e131 note).",
    "Site-test readout adaptation (e139's registered precedent): the census "
    "readout for NEAR/FAR is the arm's own trained-geometry p(Z) read; the "
    "JITTER line keeps e131's Phase-C g0-battery readout (its trained "
    "geometry among the five). Criterion arithmetic (mean/zero arms, "
    "strength=min, ratio>=0.5) is e116/e131 verbatim.",
    "Control rows: e131's CONTROL_ROWS (1..6) overlap NEAR's trained site "
    "band and cannot serve as NEAR controls; the control band is the shared "
    "set {60,100,150,160,170,200,220} (outside [0..20] and [121..146]) for "
    "all arms, plus per-arm adjacency rows reported for localization only.",
    "Nets are the mandated 2.7M e048_repro line (all fork references — "
    "0.556313, 0.5455, 0.732, e113 recipe — live on it). The dispatch's "
    "'0.84M family' note belongs to the ctx-512 rig; recorded here, root "
    "mandate honored.",
    "JITTER vs e109's none-table (tol 0.05) is REPORT-ONLY: GPU float "
    "nondeterminism precedent (e119's R rerun deviated 0.216 at g0 with the "
    "same seed/recipe); the fork's JITTER role is the in-run ROUTED "
    "reference, not a bit reproduction.",
    "Smoke mode trims: 8-step fine-tunes, reduced census rows/novel geos, "
    "nothing adjudicated.",
    "Post-first-pass instrument hardening BEFORE commit (bars and "
    "adjudication logic unchanged): the onset-only site readout leaves the "
    "non-onset site rows causally invisible (drops exactly 0 by causal "
    "masking — position t never sees rows > t), so a SUPPLEMENTARY name-span "
    "readout (mean p of the true name char over read positions "
    "addr_row..addr_row+6, which DO see the whole trained span) was added "
    "for NEAR/FAR and recorded in metrics; the onset-row readout (e131/e139 "
    "convention) stays the bar-carrying primary. Rows 1-2 added to the "
    "NEAR/FAR censuses as scaffold references (e131's d_r1 convention).",
]


# ------------------------------------------------------------------ gpu guard

def gate_launch(tag: str) -> None:
    """e119's bounded gate_launch: gpu_ok() double-poll, 20-min wait, PARK."""
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


# ------------------------------------------------------------------ instruments
# PROVENANCE: battery_cell / battery_pz / ce_fixed_cpu / val_windows /
# deleted_wpe / load_cpu / evl_load are lab/e131_rekeying_census.py's
# instruments VERBATIM (which are e068/e109/e113/e116 lineage). They are
# copied rather than imported because importing e131 force-sets
# CUDA_VISIBLE_DEVICES=-1 (CPU-only run) and this experiment owns the GPU.
# finetune_arm is lab/e109_consolidation.py's VERBATIM with (a) the bounded
# e119 gate_launch and (b) DEV parameterized for the park-to-CPU fallback.
# read_fact_at generalizes e131's read_fact_position to (addr_row, xcol);
# row_census generalizes e139's row_census_at183 the same way.

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
def battery_cell(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30,
                 keep_per_ctx=False) -> dict:
    """e068/e113/e120 battery on CPU: p(Z) at the last position."""
    net.eval()
    pzs, amax = [], 0.0
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs.append(pr[:, zid])
        amax += float((lg[:, -1].argmax(-1) == zid).float().sum())
    p = torch.cat(pzs)
    out = {"mean_pz": float(p.mean()), "median_pz": float(p.median()),
           "std_pz": float(p.std()),
           "frac_pz_ge_0.5": float((p >= 0.5).float().mean()),
           "frac_argmax_z": amax / ids.shape[0]}
    if keep_per_ctx:
        out["pz_per_ctx"] = [float(v) for v in p.tolist()]
    return out


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
    """e139's row_census_at183 VERBATIM (mean-arm / zero-arm / restore), with
    the scalar readout passed as a callable (arm's own trained-geometry
    read for NEAR/FAR; e131's battery_pz on ids130 for the JITTER line)."""
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

def finetune_arm(tag: str, net0: TinyGPT, pool_x: torch.Tensor,
                 pool_mask: torch.Tensor, anchor: torch.Tensor,
                 train_ids: torch.Tensor, r_eval_xy, f_eval_ids, zid: int,
                 seed: int):
    """e109 arm-(b)/e119-L fine-tune recipe VERBATIM (composition/seed/loss):
    batch 32 = 16 install windows from pool + 16 anchors (8 paired + 8
    random); e043 token-level union CE; constant lr 1e-3 AdamW (0.9,0.95)
    wd 0.1 clip 1.0; 300 steps / 180 s GPU cap; in-loop CPU evals every 25."""
    if DEV.type == "cuda":
        gate_launch(tag)
    net = copy.deepcopy(net0).to(DEV)
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
        if step % EVAL_EVERY == 0 or step == FT_STEPS or \
                (time.time() - t_start) > FT_TIME_CAP:
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
        if (time.time() - t_start) > FT_TIME_CAP:
            log(f"  [{tag}] time cap {FT_TIME_CAP:.0f}s at s{step}")
            trims.append(f"{tag}: time cap at step {step}")
            break
    net.eval()
    sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    del net
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    return {"sd": sd_cpu, "traj": traj, "steps_ran": step, "seed": seed}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"e143_{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "e143", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name} "
        f"({', '.join(f'{k}={v}' for k, v in meta.items())})")


CKPT_INVENTORY: dict = {}


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e143_smoke" if SMOKE else "e143")
    log(f"E143 ERROR-PLACEMENT STEERING (T079 fork; smoke={SMOKE}) -> {rd}")
    log(f"compute: train device {DEV} (gpu_ok at start: {_USE_GPU}), "
        f"cpu threads {torch.get_num_threads()}")

    if not _USE_GPU and not SMOKE:
        deviations.append("GPU parked (gpu_ok() failed at startup or no CUDA) "
                          "— fine-tunes ran CPU-side under the 1500 s cap; "
                          "reported, adjudication unchanged.")

    # ---------------- protocol rebuild (e065/e091/e109/e113/e119 verbatim)
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

    # ---------------- training pools
    # JITTER (e113 verbatim): windows at offsets {-8..+8}; name targets
    # y-cols [129+j, 136+j).
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
    pool_jit_x = torch.cat([jit_x[j] for j in JITTERS])       # (300, 256)
    pool_jit_mask = torch.cat([jit_mask[j] for j in JITTERS])

    # FAR = the +8 offset pool (locked at read rows 137..143).
    pool_far_x, pool_far_mask = jit_x[FAR_J], jit_mask[FAR_J]

    # NEAR = install windows truncated to a 6-token pre-context; name
    # x-cols 6..12; continuation extended to 243 tokens; name targets
    # y-cols [6, 13) -> mask x-cols [5, 12).
    near_wins = []
    for p, h in install_occ:
        pre = train_ids[p - NEAR_PRE: p]
        post = train_ids[p + len(h): p + len(h) + NEAR_CONT]
        if len(post) != NEAR_CONT:
            raise RuntimeError(
                f"NEAR continuation short at p={p}: {len(post)} < {NEAR_CONT} "
                f"(corpus end within {NEAR_CONT} of the host)")
        w = torch.cat([pre, name_ids, post])
        if len(w) != BLOCK:
            raise RuntimeError(f"NEAR window len {len(w)} != {BLOCK}")
        near_wins.append(w)
    pool_near_x = torch.stack(near_wins)
    pool_near_mask = torch.zeros(len(near_wins), BLOCK - 1, dtype=torch.bool)
    pool_near_mask[:, NEAR_ADDR_ROW: NEAR_ADDR_ROW + L] = True
    G_GEO = {
        "near": all(torch.equal(w[NEAR_Z_XCOL: NEAR_Z_XCOL + L], name_ids)
                    for w in pool_near_x),
        "far": all(torch.equal(w[FAR_Z_XCOL: FAR_Z_XCOL + L], name_ids)
                   for w in pool_far_x),
        "jitter": all(all(torch.equal(w[PRE + j: PRE + j + L], name_ids)
                          for w in jit_x[j]) for j in JITTERS),
    }
    G_GEO["pass"] = all(G_GEO.values())
    assert G_GEO["pass"], f"geometry gate FAILED: {G_GEO}"
    log(f"pools: near {tuple(pool_near_x.shape)} (name x-cols 6..12, rows 5..11) | "
        f"far {tuple(pool_far_x.shape)} (x-cols 138..144, rows 137..143) | "
        f"jitter {tuple(pool_jit_x.shape)} (offsets {list(JITTERS)})")

    anchor = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                          for p, _ in install_occ[:16]])       # e065/e109 bank

    # ---------------- batteries (e068 construction verbatim)
    bat_ids = {}
    geo_all = sorted(set(GEO_ORDER) | set(NOVEL_GEO))
    for j in geo_all:
        for tag, occ in (("install60", install_occ), ("held30", held_occ)):
            cs = [train_text[p - PRE - j: p] for p, _ in occ]
            bat_ids[(j, tag)] = torch.stack([corpus.encode(c) for c in cs])
    f_eval = bat_ids[(0, "install60")]             # the G_INST battery
    ids130 = bat_ids[(0, "install60")]             # e116 130-token battery
    feval_near = pool_near_x[:, :NEAR_PRE]         # p(Z) read at row 5
    feval_far = pool_far_x[:, :PRE + FAR_J]        # p(Z) read at row 137

    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

    # ---------------- root net + instrument gates
    net0 = load_cpu(CKPT_DIR / INSTALLED_CK)
    evl = copy.deepcopy(net0)
    bz0 = battery_cell(evl, f_eval, zid)
    G_INST = {"battery_pz": bz0["mean_pz"], "ref": G_INST_REF, "tol": G_INST_TOL,
              "pass": bool(abs(bz0["mean_pz"] - G_INST_REF) < G_INST_TOL)}
    log(f"G_INST installed battery p(Z) {bz0['mean_pz']:.6f} (ref {G_INST_REF}): "
        f"{'PASS' if G_INST['pass'] else 'FAIL'}")
    if not G_INST["pass"]:
        raise RuntimeError("instrument broken vs e065/e091/e113/e119")

    # G_R0BASE: e131 Phase-C census verbatim on the ROOT (install phase),
    # row 0 mean/zero drops on ids130, gated vs e131's stored values.
    cen0 = row_census(evl, (0, 1), lambda n: battery_pz(n, ids130, zid))
    G_R0BASE = {"row0_mean": cen0["rows"]["0"]["mean"],
                "row0_zero": cen0["rows"]["0"]["zero"],
                "row0_strength": cen0["rows"]["0"]["strength"],
                "ref_mean": R0_MEAN_REF, "ref_zero": R0_BASE_REF,
                "ref_strength": R0_BASE_REF, "tol": R0_TOL,
                "pass": bool(abs(cen0["rows"]["0"]["mean"] - R0_MEAN_REF) < R0_TOL
                             and abs(cen0["rows"]["0"]["zero"] - R0_BASE_REF) < R0_TOL)}
    log(f"G_R0BASE root row-0 m/z {cen0['rows']['0']['mean']:.7f}/"
        f"{cen0['rows']['0']['zero']:.7f} (strength "
        f"{cen0['rows']['0']['strength']:.7f} vs baseline {R0_BASE_REF:.7f}): "
        f"{'PASS' if G_R0BASE['pass'] else 'FAIL'}")
    if not G_R0BASE["pass"]:
        raise RuntimeError("row-0 baseline drifted vs e131 stored values")
    sd_base = {k: v.clone() for k, v in net0.state_dict().items()}
    ce_r0 = ce_fixed_cpu(evl, *r_eval_xy)
    log(f"CE_R root (60 name-free val windows): {ce_r0:.4f}")
    del evl

    # ---------------- ARM NEAR (placement treatment)
    log(f"ARM NEAR: locked replay, fact at rows 5..11 (seed {NEAR_SEED}, "
        f"{FT_STEPS} steps)")
    near = finetune_arm("near", net0, pool_near_x, pool_near_mask, anchor,
                        train_ids, r_eval_xy, feval_near, zid, NEAR_SEED)
    sd_near = near["sd"]
    save_ckpt("near", sd_near,
              {"desc": "locked replay, fact truncated to rows 5..11 (NEAR)",
               "steps": near["steps_ran"], "seed": NEAR_SEED,
               "base": f"runs/checkpoints/{INSTALLED_CK}"})

    # ---------------- ARM FAR (matched placement control)
    log("[thermal] cooldown(120) between training launches")
    cooldown(120.0)
    log(f"ARM FAR: locked replay, fact at rows 137..143 (seed {FAR_SEED}, "
        f"{FT_STEPS} steps, same RNG stream)")
    far = finetune_arm("far", net0, pool_far_x, pool_far_mask, anchor,
                       train_ids, r_eval_xy, feval_far, zid, FAR_SEED)
    sd_far = far["sd"]
    save_ckpt("far", sd_far,
              {"desc": "locked replay, fact locked at rows 137..143 (FAR)",
               "steps": far["steps_ran"], "seed": FAR_SEED,
               "base": f"runs/checkpoints/{INSTALLED_CK}"})

    # ---------------- ARM JITTER (routed reference, e113 verbatim)
    log("[thermal] cooldown(120) between training launches")
    cooldown(120.0)
    log(f"ARM JITTER: e113 recipe verbatim, offsets {list(JITTERS)} "
        f"(seed {JIT_SEED}, {FT_STEPS} steps)")
    jit = finetune_arm("jitter", net0, pool_jit_x, pool_jit_mask, anchor,
                       train_ids, r_eval_xy, f_eval, zid, JIT_SEED)
    sd_jit = jit["sd"]
    save_ckpt("jitter", sd_jit,
              {"desc": "e113 jittered replay verbatim (routed reference)",
               "steps": jit["steps_ran"], "seed": JIT_SEED,
               "base": f"runs/checkpoints/{INSTALLED_CK}"})

    arms_sd = {"near": sd_near, "far": sd_far, "jitter": sd_jit}
    arm_order = ["near", "far", "jitter"]
    arms_meta = {
        "near": {"desc": f"locked replay, fact at rows 5..11, seed {NEAR_SEED}, "
                         f"{near['steps_ran']} steps", "traj": near["traj"],
                 "seed": NEAR_SEED, "steps_ran": near["steps_ran"]},
        "far": {"desc": f"locked replay, fact at rows 137..143, seed {FAR_SEED}, "
                        f"{far['steps_ran']} steps (same RNG stream as NEAR)",
                "traj": far["traj"], "seed": FAR_SEED,
                "steps_ran": far["steps_ran"]},
        "jitter": {"desc": f"e113 jittered replay {{-8,-4,0,+4,+8}} verbatim, seed "
                           f"{JIT_SEED}, {jit['steps_ran']} steps",
                   "traj": jit["traj"], "seed": JIT_SEED,
                   "steps_ran": jit["steps_ran"]},
    }

    # ================= READOUTS (eval-only, all CPU) =================
    log("=" * 78)
    log("READOUTS (CPU): (0) own-geometry expression, (i) site content test, "
        "(ii) row-0 presence, (iii) D-all g0, (iv) novel geometry")

    arm_geo = {"near": (NEAR_ADDR_ROW, NEAR_Z_XCOL, pool_near_x),
               "far": (FAR_ADDR_ROW, FAR_Z_XCOL, pool_far_x)}
    own_read, nets = {}, {}
    for an in arm_order:
        nets[an] = evl_load(arms_sd[an])
        if an in arm_geo:
            a_r, x_c, pool = arm_geo[an]
            own_read[an] = read_fact_at(nets[an], pool, name_ids, zid, a_r, x_c)
        else:
            cells = {j: battery_cell(nets[an], bat_ids[(j, "install60")],
                                     zid)["mean_pz"] for j in GEO_ORDER}
            vals = list(cells.values())
            own_read[an] = {"pz_onset_mean": float(np.mean(vals)),
                            "pz_onset_median": float(np.median(vals)),
                            "pz_onset_frac_ge_0.5": None,
                            "per_geometry": cells,
                            "note": "jitter: mean over its 5 trained "
                                    "geometries (battery readout)"}
        o = own_read[an]
        log(f"(0) {an:7s} own-geometry p(Z) {o['pz_onset_mean']:.4f}"
            + (f" p(name7) {o['pname_mean_over7']:.4f}" if "pname_mean_over7" in o else ""))

    # (i) site content test + (ii) row-0 presence (primary readout)
    census = {}
    site_str, site_pos, r0_str, ctrl_max = {}, {}, {}, {}
    for an in arm_order:
        if an in arm_geo:
            a_r, x_c, pool = arm_geo[an]
            def readout_fn(n, a_r=a_r, x_c=x_c, pool=pool):
                return read_fact_at(n, pool, name_ids, zid, a_r,
                                    x_c)["pz_onset_mean"]
            rows = NEAR_ROWS if an == "near" else FAR_ROWS
        else:
            def readout_fn(n):                       # e131 Phase-C verbatim
                return battery_pz(n, ids130, zid)
            rows = JIT_ROWS
        census[an] = row_census(nets[an], rows, readout_fn)
        cen = census[an]
        ctrl_max[an] = max(cen["rows"][str(r)]["strength"] for r in SHARED_CTR)
        site_rows = {"near": NEAR_SITE_ROWS, "far": FAR_SITE_ROWS,
                     "jitter": JIT_SITE_ROWS}[an]
        site_str[an] = max(cen["rows"][str(r)]["strength"] for r in site_rows)
        site_pos[an] = any(
            cen["rows"][str(r)]["content"] and
            cen["rows"][str(r)]["strength"] >= 2.0 * max(ctrl_max[an], 0.0)
            for r in site_rows)
        r0_str[an] = cen["rows"]["0"]["strength"]
        best_site = max(site_rows,
                        key=lambda r: cen["rows"][str(r)]["strength"])
        log(f"(i){an:7s} site {site_str[an]:+.4f} @r{best_site} "
            f"(2x-ctrl bar {2 * ctrl_max[an]:.4f}) -> site_pos {site_pos[an]} "
            f"| (ii) row0 strength {r0_str[an]:+.4f} "
            f"(base readout {cen['base_readout']:.4f})")

    # (i-supplement) name-span site census for NEAR/FAR: readout = mean p of
    # the true name char over read positions addr_row..addr_row+6 — makes the
    # WHOLE trained span causally visible (the onset-only readout cannot see
    # rows after the onset). Report-only hardening; the onset readout above
    # carries the bars.
    site_span = {}
    for an in ("near", "far"):
        a_r, x_c, pool = arm_geo[an]

        def span_fn(n, a_r=a_r, x_c=x_c, pool=pool):
            return read_fact_at(n, pool, name_ids, zid, a_r,
                                x_c)["pname_mean_over7"]
        rows = NEAR_ROWS if an == "near" else FAR_ROWS
        site_span[an] = row_census(nets[an], rows, span_fn)
        sp = site_span[an]["rows"]
        site_rows = NEAR_SITE_ROWS if an == "near" else FAR_SITE_ROWS
        log(f"(i-s) {an:7s} name-span census: site " +
            " ".join(f"r{r} {sp[str(r)]['strength']:+.4f}" for r in site_rows)
            + f" | r0 {sp['0']['strength']:+.4f} r1 {sp['1']['strength']:+.4f}"
              f" r2 {sp['2']['strength']:+.4f}"
              f" (base span-readout {site_span[an]['base_readout']:.4f})")

    # (ii-secondary) e131-verbatim row-0 read at the g0 battery readout
    r0_g0 = {}
    for an in arm_order:
        cen = row_census(nets[an], (0, 1), lambda n: battery_pz(n, ids130, zid))
        r0_g0[an] = {"base_pz": cen["base_readout"],
                     "row0": cen["rows"]["0"], "row1": cen["rows"]["1"]}
        log(f"(ii-s) {an:7s} g0-battery row0 strength "
            f"{r0_g0[an]['row0']['strength']:+.4f} (base "
            f"{r0_g0[an]['base_pz']:.4f})")

    # (iii) deletion battery at install-60 g0
    DELS = {"none": (), "d129": (PRE - 1,), "d_all": D_ALL, "d_r0": (0,)}
    del_table, gates_surg = {}, {}
    evl2 = copy.deepcopy(net0)
    for an in arm_order:
        for dl, rows_ in DELS.items():
            if dl == "none":
                sd_del = {k: v.clone() for k, v in arms_sd[an].items()}
                gate = {"rows": [], "pass": True, "note": "no deletion"}
            else:
                sd_del, gate = deleted_wpe(arms_sd[an], rows_)
            gates_surg[f"{an}__{dl}"] = gate
            if not gate["pass"]:
                raise RuntimeError(f"deletion gate FAILED {an}/{dl}: {gate}")
            evl2.load_state_dict(sd_del)
            del_table[(an, dl)] = battery_cell(evl2, bat_ids[(0, "install60")], zid)
        log(f"(iii) {an:7s}: " + " | ".join(
            f"{dl} {del_table[(an, dl)]['mean_pz']:.3f}"
            for dl in ("none", "d129", "d_all", "d_r0")))

    # (iv) novel-geometry battery (no deletion)
    novel = {}
    for an in arm_order:
        novel[an] = {j: battery_cell(nets[an], bat_ids[(j, "install60")], zid)
                     ["mean_pz"] for j in NOVEL_GEO}
        novel[an]["ce_r"] = ce_fixed_cpu(nets[an], *r_eval_xy)
        log(f"(iv) {an:7s}: " + " ".join(f"g{j:+d} {novel[an][j]:.3f}"
                                         for j in NOVEL_GEO)
            + f" | CE_R {novel[an]['ce_r']:.4f}")

    # JITTER repro vs e109 (report-only)
    g_jrepro = None
    if not SMOKE:
        cells = {f"g{g:+d}": {"this_run": battery_cell(nets["jitter"],
                                                       bat_ids[(g, "install60")],
                                                       zid)["mean_pz"],
                              "e109_ref": E109_REF_NONE[g]}
                 for g in GEO_ORDER}
        for c in cells.values():
            c["diff"] = c["this_run"] - c["e109_ref"]
        g_jrepro = {"cells": cells, "tol": G_JREPRO_TOL, "report_only": True,
                    "pass": bool(all(abs(c["diff"]) < G_JREPRO_TOL
                                     for c in cells.values()))}
        log(f"G_JREPRO vs e109 none-table (report-only, tol {G_JREPRO_TOL}): "
            + " ".join(f"g{g:+d} {cells[f'g{g:+d}']['this_run']:.4f}/"
                       f"{E109_REF_NONE[g]:.4f}" for g in GEO_ORDER))

    # ================= ADJUDICATION (registered, verbatim clauses) ==========
    piggy_bar = 2.0 * R0_BASE_REF
    midpoint_bar = R0_BASE_REF + 0.5 * max(r0_str["jitter"] - R0_BASE_REF, 0.0)
    near_site_pos = site_pos["near"]
    near_r0_at_baseline = bool(r0_str["near"] <= midpoint_bar)
    near_piggy = bool(r0_str["near"] >= piggy_bar)
    far_site_stored = bool(site_pos["far"] and r0_str["far"] <= midpoint_bar)

    def peak_row(an):
        cen = census[an]
        region = [int(r) for r in cen["rows"]
                  if 3 <= int(r) <= 17 or 121 <= int(r) <= 146]
        return max(region, key=lambda r: cen["rows"][str(r)]["strength"])

    aggregates = {
        "own_pz": {an: own_read[an]["pz_onset_mean"] for an in arm_order},
        "r0_str": r0_str, "site_str": site_str,
        "d_all_g0": {an: del_table[(an, "d_all")]["mean_pz"] for an in arm_order},
        "novel_g12": {an: novel[an][-12] for an in arm_order},
    }
    gaps = {k: abs(v["near"] - v["far"]) for k, v in aggregates.items()}
    uniform_agg = bool(max(gaps.values()) <= UNIFORM_BAND)
    uniform_peak = bool(peak_row("near") == peak_row("far"))

    cond = {
        "COMPASS": {"near_site_pos": near_site_pos,
                    "near_r0_at_baseline": near_r0_at_baseline,
                    "fires": bool(near_site_pos and near_r0_at_baseline)},
        "PIGGY": {"near_r0_str": r0_str["near"], "piggy_bar": piggy_bar,
                  "near_ge_bar": near_piggy, "far_site_stored": far_site_stored,
                  "fires": bool(near_piggy and far_site_stored)},
        "UNIFORM": {"max_aggregate_gap": max(gaps.values()),
                    "gaps": gaps, "peaks": {"near": peak_row("near"),
                                            "far": peak_row("far")},
                    "uniform_aggregates": uniform_agg, "uniform_peak": uniform_peak,
                    "fires": bool(uniform_agg and uniform_peak)},
    }
    if cond["COMPASS"]["fires"]:
        verdict = "COMPASS-CAUSAL"
        clause = ("NEAR consolidated site-stored at rows 5-13 (site "
                  f"content-positive, strength {site_str['near']:+.4f} vs 2x-ctrl "
                  f"{2 * ctrl_max['near']:.4f}) with row-0 presence-dependence at "
                  f"install baseline ({r0_str['near']:.4f} <= midpoint bar "
                  f"{midpoint_bar:.4f}; baseline {R0_BASE_REF:.4f}, routed "
                  f"reference JITTER {r0_str['jitter']:.4f}) — invariance is "
                  "necessary for routing; T079 survives its strongest attack.")
    elif cond["PIGGY"]["fires"]:
        verdict = "PROXIMITY-PIGGYBACK"
        clause = (f"NEAR row-0 presence {r0_str['near']:.4f} >= 2x baseline "
                  f"({piggy_bar:.4f}) while FAR stays site-stored (site "
                  f"{site_str['far']:+.4f}, row-0 {r0_str['far']:.4f}) — "
                  "proximity inherits the route; T079's invariance clause dies.")
    elif cond["UNIFORM"]["fires"]:
        verdict = "UNIFORM"
        clause = (f"NEAR ~= FAR everywhere (max aggregate gap "
                  f"{max(gaps.values()):.4f} <= {UNIFORM_BAND}; census peaks "
                  f"coincide at row {peak_row('near')}) — placement inert.")
    else:
        verdict = "TEXTURE"
        clause = ("no registered bar fires cleanly: NEAR site_pos "
                  f"{near_site_pos} (strength {site_str['near']:+.4f} vs 2x-ctrl "
                  f"{2 * ctrl_max['near']:.4f}); NEAR row-0 {r0_str['near']:.4f} "
                  f"vs baseline-at-bar {midpoint_bar:.4f} and piggy-bar "
                  f"{piggy_bar:.4f}; FAR site_pos {site_pos['far']}; "
                  f"aggregate gaps {dict((k, round(v, 4)) for k, v in gaps.items())}"
                  f"; peaks near r{peak_row('near')} / far r{peak_row('far')}"
                  " — numbers reported, no bar shopping.")
    prediction_held = bool(verdict == "COMPASS-CAUSAL")
    log("=" * 78)
    log(f"E143 VERDICT: {verdict} "
        f"(committed prediction COMPASS-CAUSAL: "
        f"{'HELD' if prediction_held else 'FAILED'})")
    log(f"  deciding: NEAR site {site_str['near']:+.4f} site_pos {near_site_pos} | "
        f"NEAR r0 {r0_str['near']:.4f} (baseline {R0_BASE_REF:.4f}, midpoint-bar "
        f"{midpoint_bar:.4f}, piggy-bar {piggy_bar:.4f}) | FAR site "
        f"{site_str['far']:+.4f} r0 {r0_str['far']:.4f} | JITTER r0 "
        f"{r0_str['jitter']:.4f} (routed ref) novel g-12 {novel['jitter'][-12]:.3f}")
    log(f"  {clause}")
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e143_error_steering",
        "date": common.now_iso(),
        "registration": ("T079 card pre-registration ~07:58Z (THINKING.md), "
                         "BEFORE dispatch; bars verbatim below; "
                         "operationalizations frozen in the module docstring "
                         "before compute"),
        "registered_prediction": REGISTERED_PREDICTION,
        "committed_prediction": "COMPASS-CAUSAL",
        "prediction_held": prediction_held,
        "question": ("does the fact consolidate WHERE its error is placed even "
                     "next to the sink (NEAR rows 5-13, no diversity), or does "
                     "proximity to row 0 alone inherit routing "
                     "(proximity-vs-invariance, T079's last open cell)?"),
        "root": f"runs/checkpoints/{INSTALLED_CK} (install-phase, all arms)",
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
                     "placement": {
                         "near": {"pre_len": NEAR_PRE, "name_xcols": [6, 12],
                                  "read_rows": [5, 11], "site_band": list(NEAR_SITE_ROWS),
                                  "continuation_len": NEAR_CONT},
                         "far": {"offset": FAR_J, "name_xcols": [138, 144],
                                 "read_rows": [137, 143], "site_band": list(FAR_SITE_ROWS),
                                 "continuation_len": POST_CAP - FAR_J},
                         "jitter": {"offsets": list(JITTERS),
                                    "band": list(JIT_SITE_ROWS)}},
                     "mask": "7 name-char targets per window, all arms",
                     "battery_construction": "ctx = train_text[p-PRE-j:p], "
                                             "readout p(Z) at last position "
                                             "(wpe row 129+j)"},
        "gates": {"G_SPLICE": G_SPLICE, "G_GEO": G_GEO, "G_INST": G_INST,
                  "G_R0BASE": G_R0BASE, "G_SURG": gates_surg,
                  "G_JREPRO_report_only": g_jrepro,
                  "gpu_at_start": {"use_gpu": _USE_GPU,
                                   "status": gpu_status()}},
        "arms": arms_meta,
        "own_geometry_read": own_read,
        "site_content_test_i": {
            an: {"census": census[an],
                 "site_rows": {"near": list(NEAR_SITE_ROWS),
                               "far": list(FAR_SITE_ROWS),
                               "jitter": list(JIT_SITE_ROWS)}[an],
                 "site_strength": site_str[an],
                 "control_max": ctrl_max[an],
                 "bar_2x_control": 2.0 * ctrl_max[an],
                 "site_pos": site_pos[an],
                 "peak_row_census_region": peak_row(an),
                 "name_span_supplement": (
                     {r: site_span[an]["rows"][r] for r in site_span[an]["rows"]}
                     if an in site_span else
                     {"note": "jitter line: no single-site span (5 trained "
                              "geometries); the band census above is the "
                              "readout"})}
            for an in arm_order},
        "row0_presence_ii": {
            "primary_trained_geometry": {an: census[an]["rows"]["0"]
                                         for an in arm_order},
            "base_readouts": {an: census[an]["base_readout"] for an in arm_order},
            "secondary_g0_battery": {**r0_g0, "note":
                                     "e131-verbatim readout at the install-60 "
                                     "g0 battery (the ROOT's trained geometry). "
                                     "NEAR/FAR never train at g0, so this "
                                     "secondary mostly measures the inherited "
                                     "root route, not the arm's new storage; "
                                     "the PRIMARY (trained-geometry) readout "
                                     "carries the fork"},
            "install_baseline": R0_BASE_REF,
            "piggy_bar_2x": piggy_bar,
            "midpoint_bar": midpoint_bar,
            "routed_reference_e131": 0.7316772229270098},
        "d_all_battery_iii": {f"{an}__{dl}": del_table[(an, dl)]
                              for an in arm_order for dl in DELS},
        "novel_geometry_iv": novel,
        "adjudication": {"conditions": cond, "verdict": verdict,
                         "clause": clause,
                         "aggregates": aggregates, "aggregate_gaps": gaps,
                         "committed_prediction": "COMPASS-CAUSAL",
                         "prediction_held": prediction_held},
        "ckpt_inventory": CKPT_INVENTORY,
        "trims": trims, "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192, "block_size": 256,
                   "params": int(net0.num_params()),
                   "train_device": str(DEV), "eval_device": "cpu",
                   "torch_threads": torch.get_num_threads(), "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    # ---------------- plot
    acol = {"near": "crimson", "far": "steelblue", "jitter": "seagreen"}
    albl = {"near": "NEAR (rows 5-13)", "far": "FAR (rows 137-143)",
            "jitter": "JITTER (routed ref)"}
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.5))

    # (0,0) own-geometry expression
    ax = axes[0, 0]
    xs = np.arange(3)
    for k, an in enumerate(arm_order):
        v = own_read[an]["pz_onset_mean"]
        ax.bar(k, v, 0.55, color=acol[an], edgecolor="k", lw=0.5)
        extra = ""
        if an == "jitter":
            cells = own_read[an]["per_geometry"]
            extra = f"\n[{min(cells.values()):.2f}-{max(cells.values()):.2f}]"
        ax.text(k, v + 0.01, f"{v:.3f}{extra}", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([albl[a] for a in arm_order], fontsize=8)
    ax.set_ylabel("p(Z) at the arm's trained geometry")
    ax.set_ylim(0, 1.15)
    ax.set_title("(0) trained-geometry expression (JITTER = mean over its "
                 "5 trained geos, [min-max])", fontsize=9.5)

    # (0,1) site content census spectra
    ax = axes[0, 1]
    for an in arm_order:
        cen = census[an]
        xs_r = [int(r) for r in cen["rows"] if int(r) != 0]
        ys = [cen["rows"][str(r)]["strength"] for r in xs_r]
        ax.plot(xs_r, ys, "o-", ms=3, lw=1.1, color=acol[an], alpha=0.85,
                label=albl[an])
        ax.plot(0, cen["rows"]["0"]["strength"], "*", ms=13, color=acol[an])
        if an in site_span:
            sp = site_span[an]["rows"]
            ax.plot(xs_r, [sp[str(r)]["strength"] for r in xs_r], "x--",
                    ms=3, lw=0.9, color=acol[an], alpha=0.55,
                    label=albl[an].split(" (")[0] + " name-span" if an == "near"
                    else None)
    ax.axvspan(5, 13, color="crimson", alpha=0.07)
    ax.axvspan(121, 137, color="seagreen", alpha=0.07)
    ax.axvspan(137, 143, color="steelblue", alpha=0.10)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("wpe row (stars = row 0)")
    ax.set_ylabel("strength = min(mean-drop, zero-drop)")
    ax.set_title("(i) site content test at the trained site (readout = arm's "
                 "own geometry)\nshaded: NEAR site 5-13 | FAR site 137-143 | "
                 "JITTER band 121-137", fontsize=8.5)
    ax.legend(fontsize=7)

    # (0,2) row-0 presence-dependence
    ax = axes[0, 2]
    xs = np.arange(3)
    vals = [r0_str[an] for an in arm_order]
    ax.bar(xs, vals, 0.55, color=[acol[a] for a in arm_order], edgecolor="k", lw=0.5)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.012, f"{v:.4f}", ha="center", fontsize=8)
    ax.axhline(R0_BASE_REF, color="k", ls="-.", lw=1.2,
               label=f"install baseline {R0_BASE_REF:.4f}")
    ax.axhline(piggy_bar, color="darkorange", ls="--", lw=1.3,
               label=f"PIGGY bar 2x baseline = {piggy_bar:.4f}")
    ax.axhline(midpoint_bar, color="gray", ls=":", lw=1.3,
               label=f"'at baseline' midpoint bar {midpoint_bar:.4f}")
    ax.axhline(0.7317, color="seagreen", ls=":", lw=1.1, alpha=0.8,
               label="e131 routed reference 0.7317")
    ax.set_xticks(xs)
    ax.set_xticklabels([albl[a] for a in arm_order], fontsize=8)
    ax.set_ylabel("row-0 strength (min of mean/zero drop)")
    ax.set_title("(ii) row-0 presence-dependence (T081 dial)", fontsize=9.5)
    ax.legend(fontsize=6.6)

    # (1,0) deletion battery at g0
    ax = axes[1, 0]
    dls = ("none", "d129", "d_all", "d_r0")
    xs = np.arange(len(dls))
    bw = 0.26
    for k, an in enumerate(arm_order):
        vals = [del_table[(an, dl)]["mean_pz"] for dl in dls]
        ax.bar(xs + (k - 1) * bw, vals, bw, color=acol[an], edgecolor="k",
               lw=0.4, label=albl[an])
        for x, vv in zip(xs + (k - 1) * bw, vals):
            ax.text(x, vv + 0.006, f"{vv:.3f}", ha="center", fontsize=6.4,
                    rotation=90, va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels(["none", "D129", "D-all\n{121,125,129,133,137}",
                        "D-row-0\n(report)"], fontsize=8)
    ax.set_ylabel("battery p(Z) g0 (install-60)")
    ax.set_ylim(0, 1.15)
    ax.set_title("(iii) D-all battery at install-60 g0", fontsize=10)
    ax.legend(fontsize=7)

    # (1,1) novel geometry
    ax = axes[1, 1]
    for an in arm_order:
        ys = [novel[an][j] for j in NOVEL_GEO]
        ax.plot(np.arange(len(NOVEL_GEO)), ys, "o-", ms=5, lw=1.2,
                color=acol[an], label=albl[an])
    ax.set_xticks(np.arange(len(NOVEL_GEO)))
    ax.set_xticklabels([f"g{j:+d} (row {129 + j})" for j in NOVEL_GEO],
                       fontsize=8)
    ax.set_ylabel("battery p(Z) (install-60, no deletion)")
    ax.set_ylim(0, 1.05)
    ax.set_title("(iv) novel-geometry read — the routed type's signature "
                 "(g-12 headline)", fontsize=9.5)
    ax.legend(fontsize=7)

    # (1,2) verdict panel (the fork annotated)
    ax = axes[1, 2]
    ax.axis("off")
    vlines = [
        "THE FORK (T079 pre-reg 07:58Z; committed: COMPASS-CAUSAL):",
        f"  NEAR site 5-13 strength {site_str['near']:+.4f} "
        f"(2x-ctrl {2 * ctrl_max['near']:.4f}) site_pos={near_site_pos}",
        f"  NEAR row-0 presence {r0_str['near']:.4f} vs baseline "
        f"{R0_BASE_REF:.4f},",
        f"      midpoint bar {midpoint_bar:.4f}, piggy bar 2x={piggy_bar:.4f}",
        f"  FAR  site 137-143 strength {site_str['far']:+.4f} site_pos="
        f"{site_pos['far']}, row-0 {r0_str['far']:.4f}",
        f"  JITTER (routed ref) row-0 {r0_str['jitter']:.4f}, "
        f"g-12 {novel['jitter'][-12]:.3f}, D-all g0 "
        f"{del_table[('jitter', 'd_all')]['mean_pz']:.3f}",
        "",
        f"  NEAR g-12 {novel['near'][-12]:.3f} | FAR g-12 "
        f"{novel['far'][-12]:.3f} | NEAR D-all g0 "
        f"{del_table[('near', 'd_all')]['mean_pz']:.3f} | FAR D-all g0 "
        f"{del_table[('far', 'd_all')]['mean_pz']:.3f}",
        f"  census peaks: NEAR r{peak_row('near')} / FAR r{peak_row('far')}",
        "",
        f"VERDICT: {verdict}  "
        f"(committed prediction {'HELD' if prediction_held else 'FAILED'})",
    ] + [f"  {wd}" for wd in
         [clause[i:i + 68] for i in range(0, len(clause), 68)]]
    for i, tx in enumerate(vlines):
        ax.text(0.02, 0.97 - i * 0.042, tx, fontsize=7.4, va="top",
                family="monospace")

    fig.suptitle(f"E143 — error-placement steering: NEAR vs FAR vs JITTER "
                 f"(root e048_repro) -> {verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(rd / "error_steering.png", dpi=130)
    plt.close(fig)

    log(f"outputs: {rd / 'metrics.json'}, {rd / 'error_steering.png'}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
