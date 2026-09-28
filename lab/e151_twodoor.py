"""E151 — THE P-b CELL: TWO DOORS IN ONE NET (T082's zero-cost prediction P-b;
the taxonomy's last structural confound — R45 critic attack 1: the routed-vs-
site-stored discriminator crosses NET lineages; no single net has ever held
both memory types).

WHY (T082 P-b, registered before e140's report): "ZERO-COST PREDICTIONS
REGISTERED: (P-b) a routed memory re-taught at a new site acquires a
site-store WITHOUT losing the route (two-door addition, untested, cheap)."
T087's cliff makes the fork sharp: variance is a SWITCH at zero-vs-any — so
what happens when a net that ALREADY switched (sink-coupled, geometry-
general) is re-taught with LOCKED (zero-variance) replay at a NEW site?
T087's pre-registered reading fork: TWO-DOOR => the cliff is PER-MEMORY;
ROUTE-OVERWRITES => PER-NET (a shared substrate the locked re-teach
destroys); SITE-REJECTED => the cliff ran once and cannot run again here.

ROOT (mandated): runs/checkpoints/e131_consolidated_e113.pt — the net whose
ZEPHYRA fact is sink-coupled (row-0 census strength 0.7317 at the trained g0
readout) and geometry-general (e113 jitter recipe {-8..+8}; d_r0 kills at
every battery geometry; D-all g0 0.9047). Gates below reproduce e131's
stored cells before any compute is trusted.

RE-TEACH (the ONE training): e143's locked-replay protocol VERBATIM
(finetune_arm = e109 arm-b = e119-L recipe): batch 32 = 16 install windows +
16 anchors (8 paired + 8 random), e043 token-level union CE on the 7
name-char targets (name-only mask), AdamW (0.9,0.95) wd 0.1 constant lr 1e-3
clip 1.0, 300 steps, <=180 s GPU cap, in-loop CPU evals every 25 steps,
seed 10902 (e119's L_SEED / e143's NEAR+FAR locked-arm seed). PLACEMENT: the
e109 offset-pool machinery at offset j=+54 — FAR's construction with the
offset chosen so ZEPHYRA sits at x-cols 184..190 (name-onset READ ROW 183,
the e120/e121/e133 splice-site geometry; onset read row 183 = SPLICE_ADDR_ROW
in e120's coordinates): pre-context = the 184 TRUE train tokens before each
install host, continuation = the 65 true tokens after it. Zero position
variance by construction (every window locks the name to x-cols 184..190).

MEASURE BEFORE (root) and AFTER (e151_twodoor.pt) — identical instruments:
  (i)   SITE CONTENT TEST at 183 (does a site-store grow?): e139/e143's
        row_census convention (per row r: mean arm [wpe[r] <- mean of all
        rows] / zero arm [wpe[r] <- 0]; strength = min(mean-drop, zero-drop);
        e116 content criterion). BAR-CARRYING readout = the NAME-SPAN read
        (mean p of the true name char over reads 183..189, which causally
        see the whole trained band 183..189) — e143's own instrument-
        hardening lesson: "the onset-only site readout leaves the non-onset
        site rows causally invisible (position t never sees rows > t)". The
        ONSET readout (p(Z) at row 183; e131/e139's convention, sees row 183
        only) is co-reported in full. site_pos = e139's site convention:
        any site-band row content-positive at >= 2x shared-control-max
        (controls {60,100,150,160,170,200,220}, outside every band).
  (ii)  NOVEL-GEOMETRY expression at g-12 AND g+12 (install-60 batteries,
        ctx = train_text[p-PRE-j:p]; +-12 are OUTSIDE the root's {-8..+8}
        jitter training — novel for it). Retention = after/before per geo.
  (iii) A(129) address-key delta: row-129 census strength (e140's convention
        verbatim; readout = battery_pz on ids130, the g0 battery).
  (iv)  D-all {121,125,129,133,137} battery at g0 (+ d129 / d_r0 / d_183
        texture cells, same table).
  (v)   E150-INFORMED PROBES at BOTH geometries (g0 and g-12):
        forced-off-sink MASK (the information probe — blocks attention to
        key 0, all layers/heads, eval-only custom forward; e150: spares the
        fact at CE ~+0.03 if no information route) and NORM-LADDER 0.07 /
        0.15 (the poison probe — direction-kept rescale of wpe[0]; e150:
        0.07 kills at CE ~+0.84, 0.15 survives at CE ~+0.31).
  (vi)  D-183 DELETION (the new door's necessity): zero wpe[183]; read at
        the site (pool onset), at g0, and at g-12.
  (vii) SITE CONTENT TEST AT THE OLD BAND (does re-teaching at 183 disturb
        the original trace?): census rows 121..129 (the rows causally
        visible to the g0 read at position 129; e131's 130..137 cells are
        read-invisible by construction, not rerun) + row 0, readout ids130.

REGISTERED PREDICTION (dispatch VERBATIM — adjudicate against exactly this;
no bar shopping; texture => TEXTURE with numbers):
  - TWO-DOOR-ADDITION fires if: site content at 183 clears the content bar
    AND novel-geometry expression retains >= 50% of its pre-reteach level —
    one net, both types; the taxonomy is a property of memories, not
    protocols.
  - ROUTE-OVERWRITES fires if: g-12 collapses (<= 20% of pre-reteach) while
    the 183 site grows — locked re-teaching CONVERTS the memory to site-only
    (variance's cliff runs in reverse: removing variance re-sites the
    memory).
  - SITE-REJECTED fires if: no site-store grows at 183 (the sink-coupled
    memory resists zero-variance re-siting — the geometry-general structure
    absorbs the teaching).

OPERATIONALIZATIONS (frozen here before compute):
  * "site content at 183 clears the content bar" = site_pos (span-primary
    census): any row r in the site band 183..189 with content=True (e116:
    both drops > 0 and min/max ratio >= 0.5) AND strength >= 2x the
    shared-control-max of the SAME census. site_grows = site_pos(after) AND
    site-band max strength strictly greater than the BEFORE net's.
  * "novel-geometry expression retains >= 50%" = min(retention at g-12,
    retention at g+12) >= 0.50, retention = expr_after/expr_before on the
    install-60 battery at that geometry (the dispatch measures BOTH novel
    geometries; the min carries the clause).
  * "g-12 collapses" = retention at g-12 <= 0.20.
  * Adjudication order: TWO-DOOR-ADDITION -> ROUTE-OVERWRITES ->
    SITE-REJECTED -> TEXTURE. Every sub-boolean reported regardless.
  * Committed prediction on record (T082 P-b): TWO-DOOR-ADDITION.

INSTRUMENT PROVENANCE: battery_cell / battery_pz / ce_fixed_cpu /
val_windows / deleted_wpe / load_cpu / evl_load / read_fact_at / row_census
/ finetune_arm / gate_launch are lab/e143_error_steering.py VERBATIM (which
are e065/e068/e109/e113/e116/e119/e131 lineage); forward_custom / fwd_causal
/ fwd_offsink / battery_fwd / ce_fwd are lab/e150_flatce_route.py VERBATIM
(the forced-off-sink mask); modified_wpe is e141's row-value surgery with
the e131 confinement gate (the ladder). Copied, not imported, because
importing e131/e150 force-sets CUDA_VISIBLE_DEVICES=-1 (CPU-only rigs) and
this experiment owns the GPU. Protocol rebuild: corpus seed 1337,
SPLICE_RNG 24301 host shuffle, install60/held30 split, mix gate — e143
verbatim.

COMPUTE ENVELOPE: GPU for the ONE fine-tune (idle check gpu_ok() double-
poll via gate_launch, 20-min bounded wait then PARK; cooldown(90) before
and after — the dispatch's 60-120 s envelope; <=180 s training cap). ALL
readouts CPU-side (torch threads 8, e143's convention on this 24-logical-
core host), sequential, no busy-waiting.

Outputs: runs/e151/{metrics.json, twodoor.png}; the re-taught net
runs/checkpoints/e151_twodoor.pt (ckpt_inventory in metrics). No NOTES/
THINKING/QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python e151_twodoor.py    (E151_SMOKE=1 shakedown)
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

SMOKE = os.environ.get("E151_SMOKE") == "1"
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

# ---- re-teach placement (183-style, FAR's offset machinery) ------------------
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

# ---- fine-tune envelope (e109 arm-b / e119-L / e143 verbatim) -----------------
FT_LR = 1e-3
FT_STEPS = 300 if not SMOKE else 8
FT_TIME_CAP = 180.0 if _USE_GPU else 1500.0
EVAL_EVERY = 25 if not SMOKE else 2
NAME_BS, ANCH_BS = 16, 16
RETEACH_SEED = 10902              # e119's L_SEED / e143's NEAR+FAR locked seed
COOLDOWN_S = 90.0                 # dispatch envelope 60-120 s, around the ONE run

# ---- gates / references (full precision, from runs/e131/metrics.json) ---------
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

# ---- e150-informed probe constants --------------------------------------------
LADDER_NORMS = (0.07, 0.15)       # the poison-probe bracket (e150's verdict)
PERM_DIM_SEED = 14101             # unused here (no perm arm) — kept for lineage

# ---- registered bar constants (frozen) -----------------------------------------
RETAIN_BAR = 0.50                 # TWO-DOOR novel-geometry retention clause
COLLAPSE_BAR = 0.20               # ROUTE-OVERWRITES g-12 collapse clause
SITE_CTRL_MULT = 2.0              # e139 site convention: >= 2x control-max

REGISTERED_PREDICTION = {
    "two_door_addition": "TWO-DOOR-ADDITION fires if: site content at 183 "
        "clears the content bar AND novel-geometry expression retains >= 50% "
        "of its pre-reteach level — one net, both types; the taxonomy is a "
        "property of memories, not protocols.",
    "route_overwrites": "ROUTE-OVERWRITES fires if: g-12 collapses (<= 20% of "
        "pre-reteach) while the 183 site grows — locked re-teaching CONVERTS "
        "the memory to site-only (variance's cliff runs in reverse: removing "
        "variance re-sites the memory).",
    "site_rejected": "SITE-REJECTED fires if: no site-store grows at 183 (the "
        "sink-coupled memory resists zero-variance re-siting — the "
        "geometry-general structure absorbs the teaching).",
    "no_bar_shopping": "No bar shopping; texture => TEXTURE with numbers.",
    "operationalizations": "site_pos = span-primary census (reads 183..189, "
        "whole band causally visible — e143's instrument-hardening lesson; "
        "onset readout co-reported): any band row content=True AND strength "
        ">= 2x same-census control-max; site_grows = site_pos(after) AND "
        "band-max strength > before's; retention = expr_after/expr_before "
        "per geometry on install-60 batteries; TWO-DOOR needs "
        "min(ret_g-12, ret_g+12) >= 0.50; ROUTE-OVERWRITES needs ret_g-12 "
        "<= 0.20 AND site_grows; SITE-REJECTED = not site_pos(after); order "
        "TWO-DOOR -> ROUTE-OVERWRITES -> SITE-REJECTED -> TEXTURE.",
    "committed": "Committed prediction on record (T082 P-b): "
                 "TWO-DOOR-ADDITION.",
}

trims: list[str] = []
deviations: list[str] = [
    "Nets are the mandated 2.7M e131_consolidated line (the dispatch's "
    "'<=1M family' note is an envelope statement; every instrument, gate "
    "reference, and lineage number of this cell lives on the 2.7M line — "
    "e143's precedent note, root mandate honored).",
    "Site-readout primary = the NAME-SPAN read (reads 183..189): e143's "
    "documented instrument lesson is that the onset-only readout is causally "
    "blind to rows 184..189 (position t never sees rows > t), so an "
    "onset-primary bar could only ever test row 183 of the 7-row site band. "
    "Fixed here BEFORE compute; the onset readout (e131/e139 convention) is "
    "co-reported in full, and the onset-only site_pos is recorded as "
    "site_pos_onset for continuity.",
    "Old-band census covers rows 121..129 only (the rows causally visible to "
    "the g0 battery read at position 129); e131's rows 130..137 cells were "
    "read-invisible by construction (exactly 0.0) and are not rerun.",
    "Placement-induced context statistics (the FAR-style confound, recorded): "
    "the re-teach windows carry 184 tokens of true pre-context and 65 of "
    "continuation (vs g0's 130/119) — the fact is read after a longer "
    "context at a site 46 rows past the trained band's edge (137). Same "
    "construction class as e143's FAR arm (offset pool, name-only mask, "
    "locked geometry), offset 54 instead of 8.",
    "Re-teach budget vs original consolidation: 300 locked steps here vs the "
    "root's 300 jittered steps (e113 recipe) — step-matched, but the root "
    "also carries the 4000-step corpus pre-training + install phase; the "
    "re-teach is a fine-tune of a fine-tune (recorded for the honesty "
    "reflex, not a bar).",
    "Single seed (10902), single lineage — no replication arm (dispatch "
    "envelope: ONE training).",
    "GPU float nondeterminism may move battery cells ~1e-2 vs stored CPU "
    "references (e119 precedent); every gate uses the lab's 0.05 fallback "
    "convention with the 5e-6 bit flag reported.",
    "Smoke mode trims: 8-step re-teach, reduced census rows, 2 ladder norms "
    "kept, nothing adjudicated.",
]


# ------------------------------------------------------------------ gpu guard

def gate_launch(tag: str) -> None:
    """e119/e143's bounded gate_launch: gpu_ok() double-poll, 20-min wait, PARK."""
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
# PROVENANCE: load_cpu/evl_load/battery_cell/battery_pz/ce_fixed_cpu/
# val_windows/deleted_wpe/read_fact_at/row_census are lab/e143_error_steering.py
# VERBATIM (e065/e068/e109/e113/e116/e119/e131 lineage); battery_fwd/ce_fwd/
# forward_custom/fwd_causal/fwd_offsink are lab/e150_flatce_route.py VERBATIM
# (the forced-off-sink mask); modified_wpe is e141's row-value surgery with the
# e131 confinement gate. Copied rather than imported (those rigs force
# CUDA_VISIBLE_DEVICES=-1; this one owns the GPU).
# finetune_arm is e143's VERBATIM with (a) bounded gate_launch and (b) DEV
# parameterized for the park-to-CPU fallback.

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
    """e141's row-value surgery with e131's confinement gate: at most `row`'s
    elements change, every other tensor bit-identical (the ladder arm)."""
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
    """e143's finetune_arm VERBATIM (e109 arm-b / e119-L recipe): batch 32 =
    16 install windows from pool + 16 anchors (8 paired + 8 random); e043
    token-level union CE on the name-char targets; constant lr 1e-3 AdamW
    (0.9,0.95) wd 0.1 clip 1.0; 300 steps / 180 s GPU cap; in-loop CPU evals
    every 25 (evals consume no RNG and do not touch the trajectory)."""
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
            log(f"  [{tag}] s{step:4d} p_z(site) {bz['mean_pz']:.4f} argmaxZ "
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


CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "e151", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name} "
        f"({', '.join(f'{k}={v}' for k, v in meta.items())})")


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e151_smoke" if SMOKE else "e151")
    log(f"E151 THE P-b CELL: TWO DOORS IN ONE NET (smoke={SMOKE}) -> {rd}")
    log(f"compute: train device {DEV} (gpu_ok at start: {_USE_GPU}), "
        f"cpu threads {torch.get_num_threads()}")

    if not _USE_GPU and not SMOKE:
        deviations.append("GPU parked (gpu_ok() failed at startup or no CUDA) "
                          "— the re-teach ran CPU-side under the 1500 s cap; "
                          "reported, adjudication unchanged.")

    # ---------------- protocol rebuild (e143 verbatim)
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

    # anchor bank (e065/e109/e143 verbatim): first 16 install-position
    # ORIGINAL host windows (incumbent continuations, no ZEPHYRA)
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
    # the BEFORE/AFTER measurement battery (identical instruments)
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

        # (i) site content test at 183 — span-primary + onset co-report
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
    log("BEFORE battery (root: e131_consolidated_e113)")
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
    log("gates: G_SPLICE, G_GEO, G_CONS, G_MASK, G_ROW0, G_A129, G_DALL all PASS")

    # =====================================================================
    # RE-TEACH (the ONE training; GPU; cooldown around it)
    # =====================================================================
    log("=" * 78)
    if not SMOKE:
        log(f"[thermal] cooldown {COOLDOWN_S:.0f}s before the ONE training")
        cooldown(COOLDOWN_S)
    log(f"RE-TEACH: 300-step locked replay of ZEPHYRA at rows 183..189 "
        f"(seed {RETEACH_SEED}, device {DEV})")
    reteach = finetune_arm("reteach183", net0, pool_x, pool_mask, anchor,
                           train_ids, r_eval_xy, f_eval_ids, zid, RETEACH_SEED)
    sd_after = reteach["sd"]
    if not SMOKE:
        log(f"[thermal] cooldown {COOLDOWN_S:.0f}s after the ONE training")
        cooldown(COOLDOWN_S)
    save_ckpt("e151_twodoor", sd_after,
              {"desc": "e131_consolidated_e113 + 300-step locked replay of "
                       "ZEPHYRA at read rows 183..189 (x-cols 184..190), "
                       "name-only mask, seed 10902",
               "steps": reteach["steps_ran"], "seed": RETEACH_SEED,
               "base": f"runs/checkpoints/{ROOT_CK}"})

    log("=" * 78)
    log("AFTER battery (e151_twodoor)")
    after = measure(sd_after, "after")

    # =====================================================================
    # ADJUDICATION (registered clauses, verbatim; no bar shopping)
    # =====================================================================
    ret = {j: after["base"][j]["mean_pz"] / max(before["base"][j]["mean_pz"], 1e-12)
           for j in GEOS}
    site_pos_after = after["site_span"]["site_pos"]
    site_pos_before = before["site_span"]["site_pos"]
    site_grows = bool(site_pos_after and after["site_span"]["site_strength"]
                      > before["site_span"]["site_strength"])
    ret_min_novel = min(ret[-12], ret[12])

    cond = {
        "TWO_DOOR": {
            "site_pos_at_183": site_pos_after,
            "retention_gm12": ret[-12], "retention_gp12": ret[12],
            "retention_min_novel": ret_min_novel, "bar": RETAIN_BAR,
            "fires": bool(site_pos_after and ret_min_novel >= RETAIN_BAR)},
        "ROUTE_OVERWRITES": {
            "retention_gm12": ret[-12], "collapse_bar": COLLAPSE_BAR,
            "site_grows": site_grows,
            "fires": bool(ret[-12] <= COLLAPSE_BAR and site_grows)},
        "SITE_REJECTED": {
            "site_pos_at_183": site_pos_after,
            "fires": bool(not site_pos_after)},
    }
    if cond["TWO_DOOR"]["fires"]:
        verdict = "TWO-DOOR-ADDITION"
        clause = (f"site content at 183 cleared the bar (span strength "
                  f"{after['site_span']['site_strength']:+.4f} vs 2x-ctrl "
                  f"{after['site_span']['bar_2x_control']:.4f}, peak row "
                  f"{after['site_span']['peak_row']}) AND novel-geometry "
                  f"retention g-12 {ret[-12]:.3f} / g+12 {ret[12]:.3f} "
                  f"(min {ret_min_novel:.3f} >= {RETAIN_BAR}) — one net, both "
                  f"types; the taxonomy is a property of memories, not "
                  f"protocols. T087's cliff is PER-MEMORY.")
    elif cond["ROUTE_OVERWRITES"]["fires"]:
        verdict = "ROUTE-OVERWRITES"
        clause = (f"g-12 collapsed to {ret[-12]:.3f} of pre-reteach "
                  f"(<= {COLLAPSE_BAR}) while the 183 site grew (span "
                  f"{before['site_span']['site_strength']:+.4f} -> "
                  f"{after['site_span']['site_strength']:+.4f}) — locked "
                  f"re-teaching CONVERTED the memory to site-only; the cliff "
                  f"is PER-NET (a shared substrate the locked re-teach "
                  f"destroys).")
    elif cond["SITE_REJECTED"]["fires"]:
        verdict = "SITE-REJECTED"
        clause = (f"no site-store grew at 183 (span strength "
                  f"{after['site_span']['site_strength']:+.4f} vs 2x-ctrl "
                  f"{after['site_span']['bar_2x_control']:.4f}; onset "
                  f"{after['site_onset']['site_strength']:+.4f}) — the "
                  f"sink-coupled memory resists zero-variance re-siting; the "
                  f"geometry-general structure absorbs the teaching. The "
                  f"cliff ran once and cannot run again on this net.")
    else:
        verdict = "TEXTURE"
        clause = (f"no registered bar fired cleanly: site_pos {site_pos_after} "
                  f"(span {after['site_span']['site_strength']:+.4f} vs bar "
                  f"{after['site_span']['bar_2x_control']:.4f}; onset "
                  f"{after['site_onset']['site_pos']}); retention g-12 "
                  f"{ret[-12]:.3f}, g+12 {ret[12]:.3f} — numbers reported, no "
                  f"bar shopping.")
    prediction_held = bool(verdict == "TWO-DOOR-ADDITION")
    log("=" * 78)
    log(f"E151 VERDICT: {verdict} (committed prediction TWO-DOOR-ADDITION "
        f"[T082 P-b]: {'HELD' if prediction_held else 'FAILED'})")
    log(f"  deciding: site@183 span {before['site_span']['site_strength']:+.4f} -> "
        f"{after['site_span']['site_strength']:+.4f} (2x-ctrl bar "
        f"{after['site_span']['bar_2x_control']:.4f}) site_pos "
        f"{site_pos_before} -> {site_pos_after}")
    log(f"  novel-geometry: g-12 {before['base'][-12]['mean_pz']:.4f} -> "
        f"{after['base'][-12]['mean_pz']:.4f} (x{ret[-12]:.3f}) | g+12 "
        f"{before['base'][12]['mean_pz']:.4f} -> "
        f"{after['base'][12]['mean_pz']:.4f} (x{ret[12]:.3f})")
    log(f"  old trace: A(129) {before['old_band']['A129']:+.4f} -> "
        f"{after['old_band']['A129']:+.4f} | row0 S "
        f"{before['old_band']['row0_strength']:+.4f} -> "
        f"{after['old_band']['row0_strength']:+.4f} | D-all g0 "
        f"{before['del_table']['d_all']['g0']['mean_pz']:.4f} -> "
        f"{after['del_table']['d_all']['g0']['mean_pz']:.4f}")
    log(f"  {clause}")
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e151_twodoor",
        "date": common.now_iso(),
        "registration": ("T082's zero-cost prediction (P-b) + T087's "
                         "pre-registered reading fork (per-memory vs per-net "
                         "cliff), THINKING.md commit 7c69f9d BEFORE this "
                         "dispatch; bars verbatim below; operationalizations "
                         "frozen in the module docstring before compute"),
        "registered_prediction": REGISTERED_PREDICTION,
        "committed_prediction": "TWO-DOOR-ADDITION (T082 P-b)",
        "prediction_held": prediction_held,
        "question": ("can ONE net hold BOTH memory types — a sink-coupled "
                     "geometry-general memory re-taught with LOCKED "
                     "(zero-variance) replay at a NEW site (read row 183) "
                     "gains a site-store WITHOUT losing its route?"),
        "root": f"runs/checkpoints/{ROOT_CK} (loaded, gated vs e131 cells)",
        "reteach": {"desc": "e143 locked-replay protocol verbatim at offset "
                            "+54 (FAR-style): ZEPHYRA locked at x-cols "
                            "184..190, onset read row 183, name-only 7-target "
                            "mask, zero position variance",
                    "recipe": "batch 32 = 16 pool + 16 anchors (8 paired + 8 "
                              "random), token-weighted union CE, AdamW "
                              "(0.9,0.95) wd 0.1, lr 1e-3 constant, clip 1.0",
                    "steps_ran": reteach["steps_ran"], "seed": RETEACH_SEED,
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
        "before": before, "after": after,
        "retentions": {f"g{j:+d}": ret[j] for j in GEOS},
        "deltas": {
            "A129": after["old_band"]["A129"] - before["old_band"]["A129"],
            "row0_strength": (after["old_band"]["row0_strength"]
                              - before["old_band"]["row0_strength"]),
            "site183_span": (after["site_span"]["site_strength"]
                             - before["site_span"]["site_strength"]),
            "d_all_g0": (after["del_table"]["d_all"]["g0"]["mean_pz"]
                         - before["del_table"]["d_all"]["g0"]["mean_pz"]),
            "ce_r": after["ce_r"] - before["ce_r"],
        },
        "adjudication": {"conditions": cond, "verdict": verdict,
                         "clause": clause,
                         "committed_prediction": "TWO-DOOR-ADDITION",
                         "prediction_held": prediction_held},
        "honesty_reflex": {
            "single_seed_single_lineage": "one re-teach (seed 10902) on one "
                "root (e131 consolidated, one seed) — the P-b cell is n=1; "
                "T087's fork verdicts are lineage-specific until replicated",
            "reteach_budget_vs_consolidation": "300 locked steps vs the "
                "root's 300 jittered steps (step-matched) — but the root "
                "also carries the 4000-step corpus pre-training + install "
                "phase, so the re-teach is a fine-tune of a fine-tune; a "
                "longer locked re-teach could convert more",
            "placement_statistics": "the 183-site windows carry 184 tokens "
                "of pre-context vs g0's 130 (FAR-class confound) and sit 46 "
                "rows past the trained band's edge; the site-vs-route "
                "attribution is placement-confounded the same way e143's "
                "FAR arm is",
            "readout_ceiling": "site strengths and row-0 strength are "
                "bounded by their base readouts; retentions are ratios of "
                "the same battery, ceiling-free",
            "mask_off_distribution": "the forced-off-sink mask puts every "
                "context off its training distribution (e150's caveat) — "
                "the CE column prices the generic part; the BEFORE net's "
                "mask cells are the in-run differential control",
            "norm_ladder_is_poison_not_information": "per T086/E150, ladder "
                "kills measure SINK-HEALTH (poisoning), not routing; 0.07 "
                "should kill both nets at high CE cost, 0.15 spare both — "
                "a differential there is texture, not a route readout",
            "novel_geometry_caveat_T087": "novel-geometry expression is "
                "sink-coupled (T087's E150-applied caveat): retention "
                "conflates 'the route survived' with 'sink health at the "
                "novel geometry survived'; the mask/ladder columns separate "
                "the two only partially",
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

    plot(rd / "twodoor.png", before, after, ret, cond, verdict, clause,
         prediction_held)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'twodoor.png'}, "
        f"ckpt runs/checkpoints/e151_twodoor.pt")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot

def plot(path, before, after, ret, cond, verdict, clause, prediction_held):
    """Legible before/after dials + the two-door grid."""
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.5))

    # (0,0) novel-geometry dials (g-12 / g+12), before vs after + bars
    ax = axes[0, 0]
    xs = np.arange(2)
    for k, (j, lbl) in enumerate(((-12, "g-12 (row 117)"),
                                  (12, "g+12 (row 141)"))):
        b = before["base"][j]["mean_pz"]
        a = after["base"][j]["mean_pz"]
        ax.bar(k - 0.19, b, 0.36, color="steelblue", edgecolor="k", lw=0.5,
               label="before (root)" if k == 0 else None)
        ax.bar(k + 0.19, a, 0.36, color="crimson", edgecolor="k", lw=0.5,
               label="after (re-taught)" if k == 0 else None)
        ax.text(k - 0.19, b + 0.012, f"{b:.3f}", ha="center", fontsize=8.5)
        ax.text(k + 0.19, a + 0.012, f"{a:.3f}", ha="center", fontsize=8.5)
        for x0, bar_v, col, tag2 in ((k, RETAIN_BAR * b, "seagreen", "50%"),
                                     (k + 0.42, COLLAPSE_BAR * b, "gray",
                                      "20%")):
            ax.plot([x0 - 0.28, x0 + 0.28], [bar_v] * 2, ls="--", lw=1.1,
                    color=col)
        ax.text(k + 0.19, RETAIN_BAR * b + 0.015, f"x{ret[j]:.2f}",
                ha="center", fontsize=9, fontweight="bold",
                color="seagreen" if ret[j] >= RETAIN_BAR else "gray")
    ax.set_xticks(xs)
    ax.set_xticklabels(["g-12  (NOVEL, headline)", "g+12  (NOVEL)"],
                       fontsize=9)
    ax.set_ylabel("battery p(Z) install-60")
    ax.set_ylim(0, 1.12)
    ax.set_title("(ii) NOVEL-GEOMETRY DIALS — the route door\n"
                 f"retention g-12 x{ret[-12]:.3f} | g+12 x{ret[12]:.3f} "
                 f"(TWO-DOOR bar: min >= {RETAIN_BAR}; collapse bar "
                 f"{COLLAPSE_BAR})", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower right")

    # (0,1) site-content spectrum at 183 (span census) + onset overlay
    ax = axes[0, 1]
    for tag, d, col, mk in (("before", before, "steelblue", "o"),
                            ("after", after, "crimson", "s")):
        cen = d["census183_span"]["rows"]
        xs_r = [int(r) for r in cen if int(r) not in SHARED_CTR]
        ax.plot(xs_r, [cen[str(r)]["strength"] for r in xs_r],
                f"{mk}-", ms=4, lw=1.2, color=col, alpha=0.9,
                label=f"{tag} (span read)")
        ceno = d["census183_onset"]["rows"]
        ax.plot(xs_r, [ceno[str(r)]["strength"] for r in xs_r], f"{mk}--",
                ms=3, lw=0.9, color=col, alpha=0.45,
                label=f"{tag} (onset read)" if tag == "before" else None)
    for tag, d, col in (("before", before, "steelblue"),
                        ("after", after, "crimson")):
        cm = d["site_span"]["control_max"]
        ax.axhline(SITE_CTRL_MULT * cm, ls=":", lw=1.0, color=col, alpha=0.7,
                   label=f"{tag} 2x-ctrl bar {SITE_CTRL_MULT * cm:.4f}")
    ax.axvspan(183, 189, color="crimson", alpha=0.07)
    ax.axvspan(121, 137, color="seagreen", alpha=0.05)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("wpe row (shaded: site band 183-189; green = old band)")
    ax.set_ylabel("strength = min(mean-drop, zero-drop)")
    ax.set_title(f"(i) SITE-CONTENT TEST at 183 — the site door\n"
                 f"before {before['site_span']['site_strength']:+.4f} -> "
                 f"after {after['site_span']['site_strength']:+.4f} | "
                 f"site_pos {before['site_span']['site_pos']} -> "
                 f"{after['site_span']['site_pos']}", fontsize=9.5)
    ax.legend(fontsize=6.2)

    # (0,2) the two-door grid
    ax = axes[0, 2]
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axvline(0.5, color="k", lw=0.8)
    ax.axhline(0.5, color="k", lw=0.8)
    ax.axvspan(0.5, 1.05, ymin=0.5, color="seagreen", alpha=0.10)
    ax.axvspan(0.5, 1.05, ymax=0.5, color="gray", alpha=0.06)
    def _pt(d, tag, col):
        sx = min(1.0, max(0.0, d["site_span"]["site_strength"]
                          / max(d["site_span"]["bar_2x_control"], 1e-9))) / 2 + \
            (0.5 if d["site_span"]["site_pos"] else 0.0)
        sy = min(1.0, d["base"][-12]["mean_pz"])
        ax.scatter(sx, sy, s=130 if tag == "after" else 90, color=col,
                   edgecolor="k", lw=0.8, marker=("o" if tag == "before"
                                                 else "*"), zorder=3)
        ax.annotate(f"{tag}\n(site {d['site_span']['site_strength']:+.3f},"
                    f"\ng-12 {d['base'][-12]['mean_pz']:.3f})", (sx, sy),
                    textcoords="offset points", xytext=(8, -6), fontsize=7)
        ax.arrow(sx, sy, 0, 0, head_width=0.02, color=col)
    _pt(before, "before", "steelblue")
    _pt(after, "after", "crimson")
    ax.text(0.77, 0.97, "TWO-DOOR\n(one net, both types)", ha="center",
            va="top", fontsize=8, color="seagreen", fontweight="bold")
    ax.text(0.25, 0.97, "route only\n(SITE-REJECTED)", ha="center", va="top",
            fontsize=8, color="dimgray")
    ax.text(0.77, 0.05, "site only\n(ROUTE-OVERWRITES)", ha="center",
            va="bottom", fontsize=8, color="dimgray")
    ax.set_xlabel("site-store at 183 (strength / 2x-control bar, clipped)")
    ax.set_ylabel("geometry-generalization (g-12 expression)")
    ax.set_title("THE TWO-DOOR GRID (before o / after *)", fontsize=10)

    # (1,0) e150-informed probes at both geometries
    ax = axes[1, 0]
    rows = []
    for tag, d in (("bef", before), ("aft", after)):
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
                va="center", fontsize=6.6)
    ax.axvline(100 * RETAIN_BAR, ls="--", color="seagreen", lw=1.1)
    ax.axvline(100 * COLLAPSE_BAR, ls="--", color="gray", lw=1.1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.4)
    ax.invert_yaxis()
    ax.set_xlim(0, 145)
    ax.set_xlabel("fact retention (%) — red = mask (info probe), blue = ladder (poison probe)")
    ax.set_title("(v) E150-INFORMED PROBES at both geometries, before/after\n"
                 "mask should SPARE (no info route); 0.07 should kill by "
                 "poison (CE high), 0.15 survive", fontsize=9)

    # (1,1) deletion table
    ax = axes[1, 1]
    dls = ("none", "d129", "d_all", "d_r0", "d183")
    xs = np.arange(len(dls))
    for k, (tag, d, col) in enumerate((("before", before, "steelblue"),
                                       ("after", after, "crimson"))):
        vals = [d["del_table"][dl]["g0"]["mean_pz"] for dl in dls]
        ax.bar(xs + (k - 0.5) * 0.36, vals, 0.34, color=col, edgecolor="k",
               lw=0.4, label=tag)
        for x, v in zip(xs + (k - 0.5) * 0.36, vals):
            ax.text(x, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.8,
                    rotation=90, va="bottom")
    d183 = after["del_table"]["d183"]
    ax.text(3.5, 0.30,
            f"D-183 site onset: {before['del_table']['d183']['site_onset']:.3f}"
            f" -> {d183['site_onset']:.3f}\n"
            f"D-183 g-12: {before['del_table']['d183']['gm12']['mean_pz']:.3f}"
            f" -> {d183['gm12']['mean_pz']:.3f}\n"
            f"old band 121-129 max: {before['old_band']['band121_129_max']:.4f}"
            f" -> {after['old_band']['band121_129_max']:.4f}\n"
            f"A(129): {before['old_band']['A129']:+.4f} -> "
            f"{after['old_band']['A129']:+.4f} | row0 S "
            f"{before['old_band']['row0_strength']:+.3f} -> "
            f"{after['old_band']['row0_strength']:+.3f}\nCE_R "
            f"{before['ce_r']:.3f} -> {after['ce_r']:.3f}",
            fontsize=7, family="monospace",
            bbox=dict(facecolor="lightyellow", alpha=0.9, edgecolor="gray"))
    ax.set_xticks(xs)
    ax.set_xticklabels(["none", "D129", "D-all\n{121..137}", "D-row-0",
                        "D-183\n(new door)"], fontsize=8)
    ax.set_ylabel("battery p(Z) g0 install-60")
    ax.set_ylim(0, 1.15)
    ax.set_title("(iv)+(vi)+(vii) deletions at g0 + old-trace panel",
                 fontsize=9.5)
    ax.legend(fontsize=7.5)

    # (1,2) verdict panel
    ax = axes[1, 2]
    ax.axis("off")
    vlines = [
        "REGISTERED (dispatch verbatim; T087 fork pre-reg 7c69f9d):",
        "  TWO-DOOR-ADDITION: site@183 clears bar AND novel retention >= 50%",
        "  ROUTE-OVERWRITES: g-12 <= 20% while the 183 site grows",
        "  SITE-REJECTED: no site-store grows at 183",
        "",
        "DECIDING NUMBERS:",
        f"  site@183 (span): {before['site_span']['site_strength']:+.4f} -> "
        f"{after['site_span']['site_strength']:+.4f}"
        f"  (bar {after['site_span']['bar_2x_control']:.4f})",
        f"  site_pos: {before['site_span']['site_pos']} -> "
        f"{after['site_span']['site_pos']} "
        f"(onset {after['site_onset']['site_pos']})",
        f"  g-12: {before['base'][-12]['mean_pz']:.4f} -> "
        f"{after['base'][-12]['mean_pz']:.4f}  (x{ret[-12]:.3f})",
        f"  g+12: {before['base'][12]['mean_pz']:.4f} -> "
        f"{after['base'][12]['mean_pz']:.4f}  (x{ret[12]:.3f})",
        f"  A(129): {before['old_band']['A129']:+.4f} -> "
        f"{after['old_band']['A129']:+.4f}   row0 S: "
        f"{before['old_band']['row0_strength']:+.4f} -> "
        f"{after['old_band']['row0_strength']:+.4f}",
        f"  D-all g0: {before['del_table']['d_all']['g0']['mean_pz']:.4f} -> "
        f"{after['del_table']['d_all']['g0']['mean_pz']:.4f}   "
        f"D-183 g0: "
        f"{before['del_table']['d183']['g0']['mean_pz']:.4f} -> "
        f"{after['del_table']['d183']['g0']['mean_pz']:.4f}",
        "",
        f"VERDICT: {verdict}  (committed TWO-DOOR-ADDITION "
        f"[T082 P-b]: {'HELD' if prediction_held else 'FAILED'})",
    ] + [f"  {wd}" for wd in
         [clause[i:i + 66] for i in range(0, len(clause), 66)]]
    for i, tx in enumerate(vlines):
        ax.text(0.02, 0.97 - i * 0.048, tx, fontsize=7.2, va="top",
                family="monospace")

    fig.suptitle(f"E151 — the P-b cell: TWO DOORS IN ONE NET "
                 f"(root e131_consolidated_e113 + locked replay at 183) -> "
                 f"{verdict}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
