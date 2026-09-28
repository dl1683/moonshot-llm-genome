"""E173 — THE CLOSURE PARTITION: where does the shut geometry door live in
parameter space? (QUEUE.md row e173, DISPATCHED ~14:15Z; bars are the QUEUE
row VERBATIM, written before compute. Fills T099's named hole.)

WHY (the elimination so far): e151's locked re-teach at novel site 183 shut
the geometry door (g-12 0.916 -> 0.102) in the same 300 steps that built the
graft. e166's inverse event reported the wpe graft rows carry EXACTLY ZERO of
the closure (wpe[183:189] := root moved g-12 by +0.0000). e153 proved no
K<=6 head-set transplant switches the phase (best = reader_K3 g-12 0.1505 vs
bar 0.45). The remaining suspects: the MLP+LN stream state (66.5% of the
conversion's delta energy, late-layer-skewed — T091) or distributed
interaction between classes.

INSTRUMENT CORRECTION (R49, received at design time BEFORE any compute; it
rebukes e166, and this experiment inherits the fix): e166's "+0.0000" was an
INSTRUMENT TAUTOLOGY — the install-60 door battery's windows are PRE+j =
118/130/142 chars, so wpe rows 183-189 NEVER enter the computation (causal:
positions <= 141 never attend to positions >= 142); e166's row restores were
bit-identical even on the root's OPEN door. Consequences baked in here:
  * DOOR LADDER: every arm is measured on FOUR door cells with nested wpe-row
    visibility — home g-12/g0/g+12 (rows 0-141), door_long[-12] (onset col
    172; rows 0-171), door_long[0] (onset col 184, the site readout row 183;
    rows 0-183), door_long[+12] (onset col 196; rows 0-195). door_long[j] =
    battery mean p(Z) at the last position of windows of length 184+j ending
    at the install hosts (the e151 re-teach geometry offset by +-12; the
    name itself is NOT in the window — the cell reads onset prediction).
  * ROW CLAIMS are licensed ONLY by ladder cells whose row range covers the
    rows in question (never by home cells); "the rows carry nothing" is NOT
    citable from home cells anywhere in this experiment's outputs.
  * WPE-BLOCK CONTROL (report-only, unregistered, labeled here before
    compute): e166's EXACT surgery (wpe[183:189] := root) re-measured on the
    ladder. By construction its home AND door_long[-12] cells MUST be
    bit-equal to the twodoor base (the tautology, demonstrated in-run as a
    gate); door_long[0]/[+12] and the 183-site census are its informative
    cells.

DESIGN (eval-only class restores, e153's conventions; no training): restore
weight CLASSES from the PRE-conversion net (root = e131_consolidated) into
the POST-conversion net (twodoor = e151_twodoor), class by class:
  (a) MLP+LN   : all MLP weights + all LayerNorm params (ln1/ln2 per layer,
                 ln_f) — 50 tensors, ~67.3% of the conversion's delta energy.
  (b) ATTENTION: all attention weights (c_attn QKV + c_proj O, every layer)
                 — 12 tensors, ~30.0% of the delta energy.
  (c) WPE-ROWS : the full positional table (superset of e166's 183:189 null;
                 home cells see only its rows 0-141 diff — the ladder reads
                 the rest).
  (d) ALL-RESTORED sanity: every class restored (incl. wte/lm_head) — the
      state dict MUST equal the root's bit-exactly and the dial must
      reproduce the root's bit-close (the artifact-bounding gate; if it does
      not, swap path-dependence contaminates and every single-class read
      below carries that caveat).
  (+io)  report-only completeness control: wte + lm_head (the residual ~0.9%
      of delta energy) — NOT bar-eligible (unregistered arm, labeled here
      before compute).
  (+wpe_block) report-only instrument-corrected re-read of e166's null (see
      above) — NOT bar-eligible.
Per arm: home g0/g-12/g+12 batteries, the door_long ladder, the 183-site
census (e151's span/onset row-census convention), D-183, A(129), CE_R (every
cell CE-priced). IF MLP+LN alone reopens: graded per-layer restores of the
MLP+LN class (layer-local mlp+ln1+ln2 slices; ln_f not layer-local —
excluded; diagnostic only, no registered bars).

REGISTERED PREDICTION (QUEUE e173 VERBATIM — no bar shopping):
  "Bars: MLP-LN-CARRIES = single-class MLP+LN restore reopens g-12 >= 0.45
  with graft intact (closure is a located weight rewrite); NOTHING-CARRIES =
  every class alone <= 0.27 (interactive/stream-state; gradient-restore
  becomes the only probe)"

OPERATIONALIZATIONS (frozen here before compute):
  * g-12 = install-60 battery mean p(Z) at ctx offset j=-12 (e119/e151);
    twodoor base 0.1021, root base 0.9156 (gates below). The registered bars
    stay anchored on the HOME short battery (unchanged per the R49 note);
    the ladder co-reports and licenses row claims.
  * "single-class" arms = the three registered classes (a)-(c) alone on the
    twodoor net; all-restored is the sanity gate (IS the root net —
    trivially open, not bar-eligible); io and wpe_block are report-only.
  * "graft intact" = the arm leaves wpe bit-untouched (confinement gate) —
    construction-level guarantee; the 183-site census co-reports whether the
    graft still EXPRESSES, and a collapse of site span below 0.5 on a
    bar-crossing arm is FLAGGED as a graft-silent caveat in honesty_reflex,
    never re-adjudicated.
  * "every class alone <= 0.27" = max over the three registered class arms
    (home g-12). If a NON-MLP+LN class crosses 0.45 on home g-12, neither
    registered bar fires -> TEXTURE with numbers (the registered prediction
    names only MLP+LN as the carries-candidate).
  * graded arms: layer-l slice {mlp_l, ln1_l, ln2_l} of the MLP+LN class,
    run ONLY if the mlp_ln arm's home g-12 >= 0.45; no bars, location only.
  * Adjudication order: MLP-LN-CARRIES -> NOTHING-CARRIES -> TEXTURE.
    Raw bars decide; CE annotates (a door reopened at CE cost >> +0.1 nats
    is a wreckage flag in honesty_reflex, not a re-adjudication).
  * Internal consistency: door_long[0] must reproduce site_onset bit-close
    on the base nets (same arithmetic, different window tail — causal
    attention makes the tail invisible); recorded, gated at 0.05 fallback.

INSTRUMENT PROVENANCE: load_cpu / evl_load / battery_cell / battery_pz /
ce_fixed_cpu / val_windows / deleted_wpe / read_fact_at are lab/
e153_phase_surgery.py VERBATIM (e151/e143/e131 lineage); site_census is
e139/e151's row_census arithmetic with BOTH readouts (onset+span) taken from
one forward per (row, mode) — an efficiency adaptation, same subtractive
mean/zero arms and restore assert; restore_class is e153's transplant gate
generalized from head-slices to arbitrary key sets; row_block_surgery is
e166's row_surgery (the exact e166 null surgery). Copied, not imported
(thread/device pinning differs).

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1; three other agents on
this host) — torch threads 4, single staggered launch, no busy-waiting.
Eval-only, no training, 2.7M-param nets; expected 10-20 min.

NETS (on disk, gated vs runs/e151 stored cells BEFORE any surgery):
  * TWODOOR  runs/checkpoints/e151_twodoor.pt            (post; door SHUT)
  * ROOT     runs/checkpoints/e131_consolidated_e113.pt  (pre; door OPEN;
             the restore source)

Outputs: runs/e173/{metrics.json, closure_partition.png}. No NOTES/THINKING/
QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python e173_closure_partition.py    (E173_SMOKE=1 shakedown)
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # three other agents live

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch

torch.set_num_threads(4)                              # CPU-shared host

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import Cfg, CharCorpus, TinyGPT, run_dir, save_json   # noqa: E402
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E173_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
NETS = {
    "twodoor": CKPT_DIR / "e151_twodoor.pt",          # post; door SHUT
    "root": CKPT_DIR / "e131_consolidated_e113.pt",   # pre; door OPEN
}

# ---- site geometry (e151 verbatim) --------------------------------------------
RETEACH_J = 54                    # name x-cols 184..190, read rows 183..189
SITE_ADDR_ROW = 183
SITE_Z_XCOL = PRE + RETEACH_J     # 184
SITE_CONT = BLOCK - PRE - RETEACH_J - len(NAME)     # 65
SITE_ROWS = tuple(range(183, 190))                  # the 7 grafted rows
SHARED_CTR = (60, 100, 150, 160, 170, 200, 220)     # outside trained bands
CTRL_ROWS = SHARED_CTR                             # e151 site census controls
D_ALL = (121, 125, 129, 133, 137)                   # e113's fixed set
GEOS = (-12, 0, 12)
SITE_CTRL_MULT = 2.0              # e139 site convention: >= 2x control-max

# ---- parameter classes (frozen before compute) --------------------------------
N_LAYER, N_HEAD, N_EMBD = 6, 6, 192

def _mlp_keys(l):
    return [f"h.{l}.mlp.0.weight", f"h.{l}.mlp.0.bias",
            f"h.{l}.mlp.2.weight", f"h.{l}.mlp.2.bias"]

def _ln_keys(l):
    return [f"h.{l}.ln1.weight", f"h.{l}.ln1.bias",
            f"h.{l}.ln2.weight", f"h.{l}.ln2.bias"]

def _attn_keys(l):
    return [f"h.{l}.attn.c_attn.weight", f"h.{l}.attn.c_proj.weight"]

_MLP = [k for l in range(N_LAYER) for k in _mlp_keys(l)]
_LN_BLOCK = [k for l in range(N_LAYER) for k in _ln_keys(l)]
_LN = _LN_BLOCK + ["ln_f.weight", "ln_f.bias"]
_ATTN = [k for l in range(N_LAYER) for k in _attn_keys(l)]
_WPE = ["wpe.weight"]
_IO = ["wte.weight", "lm_head.weight"]

CLASSES = {                     # registered (a)-(c) + sanity (d) + report-only
    "mlp_ln": _MLP + _LN,       # (a) 50 tensors
    "attn": _ATTN,              # (b) 12 tensors
    "wpe": _WPE,                # (c) 1 tensor (full table)
    "io": _IO,                  # report-only completeness control
    "all": _MLP + _LN + _ATTN + _WPE + _IO,   # (d) must cover every key
}

def graded_keys(l):
    """Layer-local slice of the MLP+LN class (ln_f not layer-local)."""
    return _mlp_keys(l) + _ln_keys(l)

# ---- gate references (runs/e151 metrics before/after, verbatim from e153) ------
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05
REFS = {
    "root": {         # e151 "before"
        "g0": 0.7850371599197388, "gm12": 0.9155886769294739,
        "gp12": 0.9478210210800171, "ce_r": 1.663516640663147,
        "site_onset": 0.8898659348487854, "site_span": 0.982668936252594,
        "A129": -0.13237020391970877,
        "d_all_g0": 0.9047248959541321, "d_r0_g0": 0.0533599890768528,
        "d183_site_onset": 0.896581768989563,
    },
    "twodoor": {      # e151 "after"
        "g0": 0.10710463672876358, "gm12": 0.1020549014210701,
        "gp12": 0.12079524248838425, "ce_r": 1.6489683389663696,
        "site_onset": 0.9980737566947937, "site_span": 0.9993568658828735,
        "A129": 0.005159098383098015,
        "d_all_g0": 0.09776780754327774, "d_r0_g0": 0.0007308662170544267,
        "d183_site_onset": 0.4863327741622925,
    },
}
# e151's stored 183-site census summary cells (same SHARED_CTR controls)
CENSUS_REFS = {
    "root": {"site_span_strength": 0.000274658203125,
             "site_onset_strength": 0.0,
             "site_span_pos": False, "site_onset_pos": False},
    "twodoor": {"site_span_strength": 0.07310783863067627,
                "site_onset_strength": 0.5117409825325012,
                "site_span_pos": True, "site_onset_pos": True},
}
# e153's stored class energy shares of the conversion (T091 provenance)
E153_SHARES = {"heads": 0.2999729803489722, "mlps": 0.6647335032129783,
               "lns": 0.008505192635032622, "wpe": 0.01773844020721051,
               "wte": 0.003683935172347457, "lm_head": 0.005365948423459058,
               "total_fro": 14.798053699343638}

# ---- registered bars (frozen) -------------------------------------------------
REOPEN_BAR = 0.45                 # home g-12 absolute (mlp_ln arm)
NOTHING_CEIL = 0.27               # every registered class arm ceiling
CE_WRECK_FLAG = 0.1               # dCE above this flags wreckage (annotation)
CE_EVAL_SEED = 26502              # e065 CE_R bank seed

REGISTERED_PREDICTION = {
    "mlp_ln_carries": "MLP-LN-CARRIES = single-class MLP+LN restore reopens "
        "g-12 >= 0.45 with graft intact (closure is a located weight "
        "rewrite).",
    "nothing_carries": "NOTHING-CARRIES = every class alone <= 0.27 "
        "(interactive/stream-state; gradient-restore becomes the only "
        "probe).",
    "texture": "Texture (partial reopens) => TEXTURE with numbers.",
    "operationalizations": "g-12 = install-60 battery mean p(Z) at j=-12 "
        "(HOME short battery, rows 0-141 — registered bars anchored here, "
        "unchanged per R49); single-class arms = mlp_ln/attn/wpe alone on "
        "twodoor; all-restored = sanity gate (not bar-eligible); io + "
        "wpe_block = report-only; graft intact = wpe bit-untouched "
        "(confinement gate), site-expression co-reported and a span<0.5 "
        "collapse on a bar-crossing arm is a FLAG, not re-adjudication; "
        "nothing-carries = max home g-12 over the three class arms <= 0.27; "
        "a non-MLP+LN class >= 0.45 fires no bar -> TEXTURE; graded layer-l "
        "slices run only if mlp_ln home g-12 >= 0.45, no bars; order "
        "MLP-LN-CARRIES -> NOTHING-CARRIES -> TEXTURE; raw bars decide, CE "
        "annotates (dCE > +0.1 = wreckage flag, never re-adjudication); ROW "
        "CLAIMS licensed only by door-ladder cells (home battery is blind "
        "to wpe rows >= 142 — the e166 tautology, corrected here).",
    "no_bar_shopping": "No bar shopping; texture => TEXTURE with numbers.",
    "instrument_correction": "R49 (received at design time, before compute): "
        "e166's '+0.0000 rows carry nothing' was an instrument tautology "
        "(home battery windows never reach rows 183-189). This experiment "
        "adds the door ladder (home rows 0-141 / long-12 rows 0-171 / "
        "long0 rows 0-183 / long+12 rows 0-195) to every arm, re-measures "
        "e166's exact row surgery as the report-only wpe_block control "
        "(home and long-12 cells MUST be bit-equal to base — the tautology "
        "demonstrated in-run as a gate), and licenses row claims ONLY from "
        "ladder cells.",
}

deviations: list[str] = [
    "CPU-only at 4 torch threads (dispatch: three other agents on this "
    "host); e151's stored cells were computed at 8 threads, so bit flags "
    "may fall back to the lab's 0.05 tolerance — recorded per gate (e153's "
    "4-thread precedent reproduced e151 cells bit-exact).",
    "R49 instrument correction applied at design time (before compute): the "
    "home door battery is causally blind to wpe rows >= 142; the door "
    "ladder is co-measured for every arm and is the ONLY license for row "
    "claims; e166's null is re-read as the wpe_block control.",
    "door_long cells are a NEW instrument (no stored refs): windows of "
    "length 184+j ending at the install hosts, p(Z) at the last position; "
    "door_long[0] is cross-checked against site_onset on the base nets "
    "(same arithmetic, causally invisible tail).",
    "The io (wte+lm_head) and wpe_block arms are report-only completeness/"
    "instrument controls, NOT bar-eligible (registered classes are mlp_ln, "
    "attn, wpe only); graded arms are diagnostics without bars.",
    "Census rows trimmed vs e151's ROWS_183: SITE_ROWS + SHARED_CTR only "
    "(e151's report rows 0,1,2,180,181,182 dropped; the control set "
    "SHARED_CTR and site rows are identical, so the summary cells gate "
    "against e151's stored values).",
    "A(129) measured with mean/zero arms at row 129 only (e151's full "
    "old-band census not re-run); d_all/d_r0 deletions measured on the two "
    "base nets and the all-restored arm only (gate lineage), not on class "
    "arms (mission's lean cell list).",
    "Graded arms restore the layer-local slice {mlp_l, ln1_l, ln2_l} of the "
    "MLP+LN class; ln_f is not layer-local and is excluded (recorded).",
    "Single lineage, single seed (10902) — n=1 until replicated.",
    "Smoke mode trims: census rows (183,184)+(60,100), arms {mlp_ln, "
    "wpe_block} + all-restored, no graded; nothing adjudicated.",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: e153_phase_surgery.py VERBATIM (load_cpu/evl_load/battery_cell/
# battery_pz/ce_fixed_cpu/val_windows/deleted_wpe/read_fact_at); site_census =
# e139/e151 row_census arithmetic (dual readout per pass); restore_class =
# e153's transplant gate generalized to key sets; row_block_surgery = e166's
# row_surgery. Copied (thread/device pinning differs).

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
    while len(out_y) < n and tries < 500 * n:
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
    (position t predicts window col t+1)."""
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


def site_census(net: TinyGPT, pool_x, name_ids, zid, site_rows, ctrl_rows) -> dict:
    """e139/e151 row-census arithmetic, dual readout per pass: per row r,
    wpe[r] := mean-row (mean arm) then := 0 (zero arm); per readout
    (onset/span) drop = base - arm, strength = min(mean,zero), content =
    e139's rule. Same restore assert as e151's row_census."""
    net.eval()
    base = read_fact_at(net, pool_x, name_ids, zid, SITE_ADDR_ROW, SITE_Z_XCOL)
    w = net.wpe.weight.data
    orig = w.clone()
    mean_row = orig.mean(0)
    rows_out: dict = {}
    for r in site_rows + ctrl_rows:
        w.copy_(orig); w[r] = mean_row
        m = read_fact_at(net, pool_x, name_ids, zid, SITE_ADDR_ROW, SITE_Z_XCOL)
        w.copy_(orig); w[r] = 0.0
        z = read_fact_at(net, pool_x, name_ids, zid, SITE_ADDR_ROW, SITE_Z_XCOL)
        rec = {}
        for ro in ("span", "onset"):
            mv = base["pz_onset_mean"] - m["pz_onset_mean"] if ro == "onset" \
                else base["pname_mean_over7"] - m["pname_mean_over7"]
            zv = base["pz_onset_mean"] - z["pz_onset_mean"] if ro == "onset" \
                else base["pname_mean_over7"] - z["pname_mean_over7"]
            mx = max(mv, zv)
            rec[ro] = {"mean": float(mv), "zero": float(zv),
                       "strength": float(min(mv, zv)),
                       "content": bool(mv > 0 and zv > 0 and
                                       (min(mv, zv) / mx if mx > 0 else 0.0)
                                       >= 0.5)}
        rows_out[str(r)] = rec
    w.copy_(orig)
    assert torch.equal(w, orig), "site census failed to restore wpe"
    summary = {}
    for ro in ("span", "onset"):
        cm = max(rows_out[str(r)][ro]["strength"] for r in ctrl_rows)
        sstr = max(rows_out[str(r)][ro]["strength"] for r in site_rows)
        spos = any(rows_out[str(r)][ro]["content"] and
                   rows_out[str(r)][ro]["strength"] >= SITE_CTRL_MULT * cm
                   for r in site_rows)
        peak = max(site_rows, key=lambda r: rows_out[str(r)][ro]["strength"])
        summary[ro] = {"control_max": cm, "site_strength": sstr,
                       "bar_2x_control": SITE_CTRL_MULT * cm,
                       "site_pos": bool(spos), "peak_row": int(peak)}
    return {"base_onset": base["pz_onset_mean"],
            "base_span": base["pname_mean_over7"],
            "rows": rows_out, "summary": summary}


# ------------------------------------------------------------------ surgery

def restore_class(sd_base: dict, sd_src: dict, keys: list, name: str
                  ) -> tuple[dict, dict]:
    """Replace the TARGET's (twodoor) tensors for `keys` with the SOURCE's
    (root); confinement gate = only class keys may change, everything else
    bit-identical (e153's transplant gate generalized to key sets)."""
    out = {k: v.clone() for k, v in sd_base.items()}
    for k in keys:
        out[k] = sd_src[k].clone()
    ks = set(keys)
    per_key = {k: int((out[k] != sd_base[k]).sum().item()) for k in keys}
    n_changed = sum(per_key.values())
    n_elems = sum(sd_base[k].numel() for k in keys)
    others = all(torch.equal(out[k], sd_base[k]) for k in out if k not in ks)
    gate = {"class": name, "n_tensors": len(keys),
            "tensors_changed": int(sum(1 for v in per_key.values() if v > 0)),
            "n_elements_changed": n_changed,
            "expected_elements": int(n_elems),
            "per_key_changed": {k: v for k, v in per_key.items() if v > 0},
            "identity": bool(n_changed == 0),
            "others_bit_identical": bool(others),
            "pass": bool(others and (n_changed == 0
                                     or n_changed >= 0.95 * n_elems))}
    return out, gate


def row_block_surgery(sd: dict, rows: tuple[int, ...], values: torch.Tensor
                      ) -> tuple[dict, dict]:
    """e166's row_surgery VERBATIM: wpe[rows] := values, confinement gate."""
    out = {k: v.clone() for k, v in sd.items()}
    for i, r in enumerate(rows):
        out["wpe.weight"][r] = values[i]
    d = out["wpe.weight"] != sd["wpe.weight"]
    changed_rows = sorted(set(int(r) for r in torch.nonzero(d)[:, 0].tolist()))
    n = int(d.sum().item())
    others = all(torch.equal(out[k], sd[k]) for k in out if k != "wpe.weight")
    gate = {"rows": list(rows), "n_elements_changed": n,
            "max_expected": len(rows) * sd["wpe.weight"].shape[1],
            "changed_rows": changed_rows, "identity": bool(n == 0),
            "confined": bool(set(changed_rows) <= set(rows)),
            "others_bit_identical": bool(others),
            "pass": bool(set(changed_rows) <= set(rows) and others)}
    return out, gate


def class_energy(sd_a: dict, sd_b: dict) -> dict:
    """e153's wiring-diff class energy shares (root=a vs twodoor=b),
    recomputed for self-containment; per-layer MLP/LN deltas for the graded
    context."""
    groups = {"attn": _ATTN, "mlps": _MLP,
              "lns": _LN_BLOCK + ["ln_f.weight", "ln_f.bias"],
              "wpe": _WPE, "wte": _IO[0:1], "lm_head": _IO[1:2]}
    E = {g: sum(float((sd_b[k] - sd_a[k]).norm() ** 2) for k in ks)
         for g, ks in groups.items()}
    tot = sum(E.values())
    per_layer = {f"mlp_l{l}": float(sum((sd_b[k] - sd_a[k]).norm() ** 2
                                        for k in _mlp_keys(l)) ** 0.5)
                 for l in range(N_LAYER)}
    per_layer_ln = {f"ln_l{l}": float(sum((sd_b[k] - sd_a[k]).norm() ** 2
                                          for k in _ln_keys(l)) ** 0.5)
                    for l in range(N_LAYER)}
    return {"fro": {g: v ** 0.5 for g, v in E.items()},
            "energy_shares": {g: v / tot for g, v in E.items()},
            "total_fro": tot ** 0.5,
            "e153_stored_shares": E153_SHARES,
            "per_layer_mlp_fro": per_layer, "per_layer_ln_fro": per_layer_ln,
            "ln_f_fro": float((sd_b["ln_f.weight"] - sd_a["ln_f.weight"]).norm()
                              ** 2 + (sd_b["ln_f.bias"]
                                      - sd_a["ln_f.bias"]).norm() ** 2) ** 0.5}


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e173_smoke" if SMOKE else "e173")
    log(f"E173 THE CLOSURE PARTITION (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-only, torch threads {torch.get_num_threads()}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # ---------------- protocol rebuild (e151/e153/e166 verbatim) --------------
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
    assert mix == {"FLORIZEL": 19, "ELIZABETH": 41}, f"splice drift {mix}"
    log(f"protocol rebuilt: install60 {mix} (SPLICE_RNG {E43.SPLICE_RNG})")
    name_ids = corpus.encode(NAME)

    # site battery pool (e151's re-teach windows, verbatim construction)
    wins = []
    for p, h in install_occ:
        pre = train_ids[p - PRE - RETEACH_J: p]
        post = train_ids[p + len(h): p + len(h) + SITE_CONT]
        assert len(pre) == PRE + RETEACH_J and len(post) == SITE_CONT
        wins.append(torch.cat([pre, name_ids, post]))
    pool_x = torch.stack(wins)
    assert all(torch.equal(w[SITE_Z_XCOL:SITE_Z_XCOL + len(NAME)].cpu(),
                           name_ids) for w in pool_x)

    # batteries: home (rows 0-141) + the door ladder (long windows)
    bat_ids = {}
    for j in GEOS:                                   # home: len PRE+j
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    bat_long = {}
    for j in GEOS:                                   # ladder: len 184+j
        cs = [train_text[p - (PRE + RETEACH_J) - j: p] for p, _ in install_occ]
        bat_long[j] = torch.stack([corpus.encode(c) for c in cs])
    for j in GEOS:
        assert bat_long[j].shape[1] == PRE + RETEACH_J + j
    LADDER = {"home_gm12": ("home", -12, "rows 0-141"),
              "long_m12": ("long", -12, "rows 0-171"),
              "long_0": ("long", 0, "rows 0-183"),
              "long_p12": ("long", 12, "rows 0-195")}
    ids130 = bat_ids[0]
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, CE_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    log(f"batteries: home {bat_ids[0].shape[1]}-char (rows 0-141) | ladder "
        + " ".join(f"{k}={bat_long[j].shape[1]}ch" for k, (_, j, _)
                   in LADDER.items())
        + " | site pool " + str(tuple(pool_x.shape)))

    # ---------------- nets ----------------------------------------------------
    sd = {}
    for tag in NETS:
        m = load_cpu(NETS[tag])
        sd[tag] = {k: v.clone() for k, v in m.state_dict().items()}
        del m
    # class coverage check: 'all' must cover every state-dict key exactly
    assert set(CLASSES["all"]) == set(sd["twodoor"].keys()), \
        "CLASSES['all'] does not cover the state dict exactly"
    for cname, keys in CLASSES.items():
        n = sum(sd["twodoor"][k].numel() for k in keys)
        log(f"class {cname}: {len(keys)} tensors, {n} params "
            f"({n / 2739072 * 100:.1f}% of the net)")
    log("nets loaded: twodoor (post, SHUT), root (pre, OPEN; restore source)")

    wdiff = class_energy(sd["root"], sd["twodoor"])
    log("conversion delta energy: "
        + ", ".join(f"{k} {v * 100:.1f}%"
                    for k, v in wdiff["energy_shares"].items())
        + f" | total fro {wdiff['total_fro']:.2f} (e153 stored "
          f"{E153_SHARES['total_fro']:.2f})")

    gates_surg: dict = {}

    # ---------------- the dial (every arm, same instrument) -------------------
    def dial(sd_in: dict, tag: str, full: bool = False,
             census: bool = True) -> dict:
        """Home 3-geos + door ladder + site read + 183-census + A(129) +
        D-183 + CE_R (CE-priced); full=True adds d_all/d_r0 (base gates)."""
        net = evl_load(sd_in)
        d: dict = {"tag": tag}
        d["base"] = {j: battery_cell(net, bat_ids[j], zid) for j in GEOS}
        d["door_long"] = {j: battery_cell(net, bat_long[j], zid) for j in GEOS}
        d["ce_r"] = ce_fixed_cpu(net, *r_eval_xy)
        sr = read_fact_at(net, pool_x, name_ids, zid, SITE_ADDR_ROW,
                          SITE_Z_XCOL)
        d["site_onset"] = sr["pz_onset_mean"]
        d["site_span"] = sr["pname_mean_over7"]
        # A(129): mean/zero census arms (e140/e151 convention)
        w = net.wpe.weight.data
        orig = w.clone()
        mean_row = orig.mean(0)
        bp = battery_pz(net, ids130, zid)
        w[129] = mean_row
        m129 = bp - battery_pz(net, ids130, zid)
        w.copy_(orig)
        w[129] = 0.0
        z129 = bp - battery_pz(net, ids130, zid)
        w.copy_(orig)
        assert torch.equal(w, orig), "A129 census failed to restore wpe"
        d["A129_mean_drop"] = float(m129)
        d["A129_zero_drop"] = float(z129)
        d["A129"] = float(min(m129, z129))
        if census:
            site_rows = SITE_ROWS if not SMOKE else (183, 184)
            ctrl = CTRL_ROWS if not SMOKE else (60, 100)
            d["census"] = site_census(net, pool_x, name_ids, zid,
                                      site_rows, ctrl)
        if full:
            for dl, rows_ in (("d_all", D_ALL), ("d_r0", (0,))):
                sd_d, g = deleted_wpe(sd_in, rows_)
                gates_surg[f"{tag}__{dl}"] = g
                if not g["pass"]:
                    raise RuntimeError(f"deletion gate FAILED {tag}/{dl}: {g}")
                net.load_state_dict(sd_d)
                d[dl] = {"g0": battery_cell(net, bat_ids[0], zid)["mean_pz"],
                         "ce_r": ce_fixed_cpu(net, *r_eval_xy)}
        # D-183 always (site + ladder-m12 readout + CE price)
        sd_d, g = deleted_wpe(sd_in, (183,))
        gates_surg[f"{tag}__d183"] = g
        if not g["pass"]:
            raise RuntimeError(f"D-183 gate FAILED {tag}: {g}")
        net.load_state_dict(sd_d)
        s2 = read_fact_at(net, pool_x, name_ids, zid, SITE_ADDR_ROW,
                          SITE_Z_XCOL)
        d["d183"] = {"site_onset": s2["pz_onset_mean"],
                     "site_span": s2["pname_mean_over7"],
                     "gm12": battery_cell(net, bat_ids[-12], zid)["mean_pz"],
                     "long_p12": battery_cell(net, bat_long[12], zid)["mean_pz"],
                     "ce_r": ce_fixed_cpu(net, *r_eval_xy)}
        net.load_state_dict(sd_in)
        del net
        return d

    def flat_cells(d: dict) -> dict:
        c = {"g0": d["base"][0]["mean_pz"], "gm12": d["base"][-12]["mean_pz"],
             "gp12": d["base"][12]["mean_pz"], "ce_r": d["ce_r"],
             "site_onset": d["site_onset"], "site_span": d["site_span"],
             "A129": d["A129"], "d183_site_onset": d["d183"]["site_onset"],
             "long_m12": d["door_long"][-12]["mean_pz"],
             "long_0": d["door_long"][0]["mean_pz"],
             "long_p12": d["door_long"][12]["mean_pz"]}
        if "d_all" in d:
            c["d_all_g0"] = d["d_all"]["g0"]
            c["d_r0_g0"] = d["d_r0"]["g0"]
        return c

    # ---------------- base dials + gates vs e151 stored cells -----------------
    base = {}
    gates: dict = {"G_SPLICE": {"install_mix": mix, "pass": True}}
    for tag in ("twodoor", "root"):
        base[tag] = dial(sd[tag], f"base_{tag}", full=True)
        b = base[tag]
        cells = flat_cells(b)
        g = {"cells": {k: cells[k] for k in REFS[tag]},
             "refs": REFS[tag], "tol_bit": G_BIT_TOL,
             "fallback_tol": G_FALLBACK_TOL,
             "bit": bool(all(abs(cells[k] - REFS[tag][k]) < G_BIT_TOL
                             for k in REFS[tag])),
             "pass": bool(all(abs(cells[k] - REFS[tag][k]) < G_FALLBACK_TOL
                              for k in REFS[tag]))}
        # census summary cells vs e151's stored site census
        cs = b["census"]["summary"]
        cens_cells = {"site_span_strength": cs["span"]["site_strength"],
                      "site_onset_strength": cs["onset"]["site_strength"],
                      "site_span_pos": cs["span"]["site_pos"],
                      "site_onset_pos": cs["onset"]["site_pos"]}
        g["census_cells"] = cens_cells
        g["census_refs"] = CENSUS_REFS[tag]
        g["census_bit"] = bool(
            abs(cens_cells["site_span_strength"]
                - CENSUS_REFS[tag]["site_span_strength"]) < G_BIT_TOL
            and abs(cens_cells["site_onset_strength"]
                    - CENSUS_REFS[tag]["site_onset_strength"]) < G_BIT_TOL
            and cens_cells["site_span_pos"] == CENSUS_REFS[tag]["site_span_pos"]
            and cens_cells["site_onset_pos"] == CENSUS_REFS[tag]["site_onset_pos"])
        g["pass"] = bool(g["pass"] and g["census_bit"])
        # ladder internal consistency: door_long[0] == site_onset (same
        # arithmetic, causally invisible tail)
        g["ladder_consistency"] = {
            "long0_minus_siteonset": b["door_long"][0]["mean_pz"]
                                      - b["site_onset"],
            "pass": bool(abs(b["door_long"][0]["mean_pz"] - b["site_onset"])
                         < G_FALLBACK_TOL)}
        gates[f"G_{tag.upper()}"] = g
        log(f"GATE {tag}: " + " ".join(f"{k} {cells[k]:.4f}"
                                       for k in REFS[tag])
            + f" | census span {cs['span']['site_strength']:+.4f} onset "
              f"{cs['onset']['site_strength']:+.4f} -> "
            + ("PASS" if g["pass"] else "FAIL")
            + (" (bit)" if g["bit"] and g["census_bit"] else "")
            + f" | long0-siteonset {g['ladder_consistency']['long0_minus_siteonset']:+.2e}")
        if not g["pass"]:
            raise RuntimeError(f"{tag} failed its gate vs e151 stored cells")
        log(f"  ladder {tag}: "
            + " ".join(f"long{j:+d} {b['door_long'][j]['mean_pz']:.4f}"
                       for j in GEOS))

    # ---------------- class restores (the partition) --------------------------
    log("=" * 78)
    arm_names = ["mlp_ln", "attn", "wpe", "io"] if not SMOKE \
        else ["mlp_ln", "wpe"]
    arms: dict = {}

    def run_arm(cname: str, sd_arm: dict, gate: dict, registered: bool) -> dict:
        d = dial(sd_arm, f"arm_{cname}", full=False, census=True)
        bt = base["twodoor"]
        rec = {"arm": cname, "registered": registered,
               "bar_eligible": bool(registered), "gate": gate, "dial": d,
               "gm12": d["base"][-12]["mean_pz"],
               "delta_gm12": d["base"][-12]["mean_pz"]
                             - bt["base"][-12]["mean_pz"],
               "delta_ce": d["ce_r"] - bt["ce_r"],
               "door_fraction": (d["base"][-12]["mean_pz"]
                                 - bt["base"][-12]["mean_pz"])
               / max(base["root"]["base"][-12]["mean_pz"]
                     - bt["base"][-12]["mean_pz"], 1e-12)}
        arms[cname] = rec
        lab = "" if registered else " [report-only]"
        log(f"ARM {cname}{lab}: home g-12 {rec['gm12']:.4f} (base "
            f"{bt['base'][-12]['mean_pz']:.4f}, d {rec['delta_gm12']:+.4f}, "
            f"frac {rec['door_fraction'] * 100:+.1f}%) g0 "
            f"{d['base'][0]['mean_pz']:.4f} | ladder "
            + " ".join(f"{j:+d} {d['door_long'][j]['mean_pz']:.4f}"
                       for j in GEOS)
            + f" | site {d['site_onset']:.4f}/{d['site_span']:.4f} A129 "
              f"{d['A129']:+.4f} CE {d['ce_r']:.4f} "
              f"(dCE {rec['delta_ce']:+.4f})")
        return rec

    for cname in arm_names:
        sd_a, g_a = restore_class(sd["twodoor"], sd["root"], CLASSES[cname],
                                  cname)
        gates_surg[f"arm_{cname}"] = g_a
        if not g_a["pass"]:
            raise RuntimeError(f"class gate FAILED {cname}: {g_a}")
        run_arm(cname, sd_a, g_a, registered=cname in ("mlp_ln", "attn", "wpe"))

    # wpe_block: e166's exact surgery, re-read on the ladder (report-only)
    sd_b, g_b = row_block_surgery(sd["twodoor"], SITE_ROWS,
                                  sd["root"]["wpe.weight"][list(SITE_ROWS)]
                                  .clone())
    gates_surg["arm_wpe_block"] = g_b
    if not g_b["pass"]:
        raise RuntimeError(f"wpe_block gate FAILED: {g_b}")
    blk = dial(sd_b, "arm_wpe_block", full=False, census=True)
    bt = base["twodoor"]
    taut = {"home_gm12_bit_equal": bool(blk["base"][-12]["mean_pz"]
                                        == bt["base"][-12]["mean_pz"]),
            "home_g0_bit_equal": bool(blk["base"][0]["mean_pz"]
                                      == bt["base"][0]["mean_pz"]),
            "home_gp12_bit_equal": bool(blk["base"][12]["mean_pz"]
                                        == bt["base"][12]["mean_pz"]),
            "long_m12_bit_equal": bool(blk["door_long"][-12]["mean_pz"]
                                       == bt["door_long"][-12]["mean_pz"]),
            "ce_bit_equal": bool(blk["ce_r"] == bt["ce_r"]),
            "ce_note": "CE_R windows are 256 chars and DO read rows "
                       "183-189, so dCE may be nonzero — informational, "
                       "not part of the tautology claim (only cells whose "
                       "windows stop below row 173 are causally blind)"}
    taut["pass"] = bool(taut["home_gm12_bit_equal"] and taut["home_g0_bit_equal"]
                        and taut["home_gp12_bit_equal"]
                        and taut["long_m12_bit_equal"])
    gates["G_TAUTOLOGY_WPE_BLOCK"] = taut
    arms["wpe_block"] = {"arm": "wpe_block", "registered": False,
                         "bar_eligible": False, "gate": g_b, "dial": blk,
                         "gm12": blk["base"][-12]["mean_pz"],
                         "delta_gm12": blk["base"][-12]["mean_pz"]
                                       - bt["base"][-12]["mean_pz"],
                         "delta_ce": blk["ce_r"] - bt["ce_r"],
                         "tautology_check": taut}
    log(f"ARM wpe_block [report-only; e166 null re-read]: home g-12 "
        f"{blk['base'][-12]['mean_pz']:.4f} (d {arms['wpe_block']['delta_gm12']:+.4f}) "
        f"| ladder "
        + " ".join(f"{j:+d} {blk['door_long'][j]['mean_pz']:.4f}" for j in GEOS)
        + f" | site {blk['site_onset']:.4f}/{blk['site_span']:.4f} | "
          f"tautology gate (home+long-12 bit-equal; CE exempt, 256ch): "
        + ("PASS" if taut["pass"] else "FAIL"))
    if not taut["pass"]:
        raise RuntimeError("wpe_block tautology gate FAILED — home cells of a "
                           "rows>=183-only surgery must be bit-equal to base")

    # ---------------- (d) ALL-RESTORED sanity gate ----------------------------
    sd_all, g_all = restore_class(sd["twodoor"], sd["root"], CLASSES["all"],
                                  "all")
    gates_surg["arm_all"] = g_all
    if not g_all["pass"]:
        raise RuntimeError(f"all-restored gate FAILED: {g_all}")
    sd_bit_equal_root = all(torch.equal(sd_all[k], sd["root"][k]) for k in sd_all)
    dial_all = dial(sd_all, "arm_all", full=True, census=True)
    c_all, c_root = flat_cells(dial_all), flat_cells(base["root"])
    diffs = {k: abs(c_all[k] - c_root[k]) for k in c_root}
    max_diff = max(diffs.values())
    cen_a = dial_all["census"]["summary"]
    cen_r = base["root"]["census"]["summary"]
    cen_diff = max(abs(cen_a[ro]["site_strength"] - cen_r[ro]["site_strength"])
                   for ro in ("span", "onset"))
    bit_all = max_diff < G_BIT_TOL and cen_diff < G_BIT_TOL
    pass_all = max_diff < G_FALLBACK_TOL and cen_diff < G_FALLBACK_TOL
    gates["G_ALL_RESTORED"] = {
        "sd_bit_equal_root": bool(sd_bit_equal_root),
        "max_cell_diff": max_diff, "census_max_diff": cen_diff,
        "cell_diffs": diffs, "bit": bool(bit_all), "pass": bool(pass_all),
        "note": "artifact-bounding gate: classes compose to the root; if "
                "this fails, swap path-dependence contaminates and every "
                "single-class read carries that caveat"}
    all_rec = {"arm": "all", "registered": False, "bar_eligible": False,
               "gate": g_all, "dial": dial_all,
               "gm12": dial_all["base"][-12]["mean_pz"],
               "delta_gm12": dial_all["base"][-12]["mean_pz"]
                             - base["twodoor"]["base"][-12]["mean_pz"],
               "delta_ce": dial_all["ce_r"] - base["twodoor"]["ce_r"],
               "door_fraction": (dial_all["base"][-12]["mean_pz"]
                                 - base["twodoor"]["base"][-12]["mean_pz"])
               / max(base["root"]["base"][-12]["mean_pz"]
                     - base["twodoor"]["base"][-12]["mean_pz"], 1e-12),
               "sd_bit_equal_root": bool(sd_bit_equal_root)}
    arms["all"] = all_rec
    log(f"ARM all-restored [sanity gate]: home g-12 "
        f"{dial_all['base'][-12]['mean_pz']:.4f} CE "
        f"{dial_all['ce_r']:.4f} | sd==root bits "
        f"{sd_bit_equal_root} | max dial diff vs root {max_diff:.2e} "
        f"(census {cen_diff:.2e}) -> "
        + ("PASS" if pass_all else "FAIL")
        + (" (bit)" if bit_all else ""))
    if not sd_bit_equal_root:
        raise RuntimeError("all-restored state dict is NOT bit-equal to root "
                           "- class key sets are wrong")

    # ---------------- graded per-layer MLP+LN (conditional) -------------------
    graded: list[dict] = []
    mlp_gm12 = arms["mlp_ln"]["gm12"]
    if (not SMOKE) and mlp_gm12 >= REOPEN_BAR:
        log(f"MLP+LN reopened ({mlp_gm12:.4f} >= {REOPEN_BAR}) -> graded "
            "per-layer restores for location detail")
        for l in range(N_LAYER):
            sd_g, g_g = restore_class(sd["twodoor"], sd["root"],
                                      graded_keys(l), f"mlp_ln_L{l}")
            gates_surg[f"graded_L{l}"] = g_g
            if not g_g["pass"]:
                raise RuntimeError(f"graded gate FAILED L{l}: {g_g}")
            d = dial(sd_g, f"graded_L{l}", full=False, census=False)
            rec = {"arm": f"mlp_ln_L{l}", "layer": l, "gate": g_g, "dial": d,
                   "gm12": d["base"][-12]["mean_pz"],
                   "delta_ce": d["ce_r"] - base["twodoor"]["ce_r"]}
            graded.append(rec)
            log(f"GRADED L{l}: home g-12 {rec['gm12']:.4f} | ladder "
                + " ".join(f"{j:+d} {d['door_long'][j]['mean_pz']:.4f}"
                           for j in GEOS)
                + f" | CE {d['ce_r']:.4f} (d {rec['delta_ce']:+.4f})")
    elif not SMOKE:
        log(f"graded variant NOT triggered (mlp_ln home g-12 {mlp_gm12:.4f} "
            f"< {REOPEN_BAR})")

    # ---------------- adjudication (registered; no shopping) ------------------
    log("=" * 78)
    cls = [arms[c]["gm12"] for c in ("mlp_ln", "attn", "wpe")]
    max_cls = max(cls)
    mlp_arm = arms["mlp_ln"]
    graft_silent = bool(mlp_arm["dial"]["site_span"] < 0.5)
    carries = bool(mlp_gm12 >= REOPEN_BAR)
    nothing = bool(max_cls <= NOTHING_CEIL)
    non_mlp_crossed = [c for c in ("attn", "wpe")
                       if arms[c]["gm12"] >= REOPEN_BAR]
    if carries:
        verdict = "MLP-LN-CARRIES"
        clause = (f"the single-class MLP+LN restore reopened the geometry "
                  f"door: home g-12 {mlp_gm12:.4f} >= {REOPEN_BAR} (from "
                  f"twodoor base "
                  f"{base['twodoor']['base'][-12]['mean_pz']:.4f}; root open "
                  f"{base['root']['base'][-12]['mean_pz']:.4f}) at CE "
                  f"{mlp_arm['delta_ce']:+.4f} — the closure is a located "
                  f"weight rewrite in the MLP+LN stream state"
                  + (" [GRAFT-SILENT FLAG: site span collapsed below 0.5 "
                     "while the door reopened — door and graft decoupled at "
                     "expression level; reported, not re-adjudicated]"
                     if graft_silent else "")
                  + (" [WRECKAGE FLAG: dCE > +0.1]" if mlp_arm["delta_ce"]
                     > CE_WRECK_FLAG else "")
                  + ". Graded per-layer restores below give the location.")
    elif nothing:
        verdict = "NOTHING-CARRIES"
        clause = (f"every registered class alone stayed <= {NOTHING_CEIL} on "
                  f"home g-12 (max {max_cls:.4f}; mlp_ln "
                  f"{arms['mlp_ln']['gm12']:.4f}, attn "
                  f"{arms['attn']['gm12']:.4f}, wpe "
                  f"{arms['wpe']['gm12']:.4f}) while the all-restored net "
                  f"IS the root ({arms['all']['gm12']:.4f}) — the closure is "
                  f"interactive/stream-state: no single weight class carries "
                  f"it in isolation; the gradient-restore cell becomes the "
                  f"only remaining probe. Ladder cells co-reported; row "
                  f"claims licensed only there.")
    else:
        verdict = "TEXTURE"
        extra = (f"; non-MLP+LN class crossed the reopen bar: {non_mlp_crossed}"
                 if non_mlp_crossed else "")
        clause = (f"no registered bar fired cleanly: mlp_ln home g-12 "
                  f"{mlp_gm12:.4f} (< {REOPEN_BAR}) but the class ceiling "
                  f"{NOTHING_CEIL} was exceeded (max class {max_cls:.4f}) — "
                  f"partial carriage without a single-class switch{extra}; "
                  f"numbers reported, no bar shopping.")
    adjudication = {
        "bars": {"reopen_gm12_home": REOPEN_BAR,
                 "nothing_carries_ceiling": NOTHING_CEIL,
                 "ce_wreck_flag": CE_WRECK_FLAG,
                 "note": "bars anchored on the HOME short battery (rows "
                         "0-141), unchanged per R49; ladder is co-evidence"},
        "class_gm12": {"mlp_ln": arms["mlp_ln"]["gm12"],
                       "attn": arms["attn"]["gm12"],
                       "wpe": arms["wpe"]["gm12"],
                       "io_report_only": arms["io"]["gm12"] if "io" in arms
                       else None,
                       "wpe_block_report_only": arms["wpe_block"]["gm12"],
                       "all_restored": arms["all"]["gm12"]},
        "max_class_gm12": max_cls, "mlp_ln_carries": carries,
        "nothing_carries": nothing, "graft_silent_flag": graft_silent,
        "non_mlp_class_crossings": non_mlp_crossed,
        "all_restored_gate": {"pass": gates["G_ALL_RESTORED"]["pass"],
                              "bit": gates["G_ALL_RESTORED"]["bit"],
                              "sd_bit_equal_root": sd_bit_equal_root},
        "graded_run": bool(graded), "verdict": verdict, "clause": clause,
        "instrument_note": "home battery is causally blind to wpe rows >= "
                           "142 (e166 tautology, R49); the door ladder "
                           "(home/long-12/long0/long+12 = wpe rows 0-141/"
                           "0-171/0-183/0-195) co-measured on every arm; "
                           "'the rows carry nothing' is citable ONLY from "
                           "ladder cells",
    }
    log(f"E173 VERDICT: {verdict}")
    log(f"  class home g-12: mlp_ln {arms['mlp_ln']['gm12']:.4f} | attn "
        f"{arms['attn']['gm12']:.4f} | wpe {arms['wpe']['gm12']:.4f} | "
        f"io {arms['io']['gm12']:.4f} | wpe_block "
        f"{arms['wpe_block']['gm12']:.4f} | all-restored "
        f"{arms['all']['gm12']:.4f} (bars: reopen >= {REOPEN_BAR}; "
        f"ceiling <= {NOTHING_CEIL})")
    log(f"  {clause}")

    # ---------------- outputs -------------------------------------------------
    metrics = {
        "experiment": "e173_closure_partition",
        "date": common.now_iso(),
        "registration": ("T099's named hole; QUEUE.md row e173 dispatch "
                         "~14:15Z; R49 instrument correction (e166 "
                         "tautology) applied at design time BEFORE compute. "
                         "Registered prediction verbatim below; "
                         "operationalizations frozen in the module docstring "
                         "before compute."),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("where does the shut geometry door live in parameter "
                     "space — does restoring a single weight CLASS "
                     "(MLP+LN / attention / wpe) from the pre-conversion "
                     "net into the converted net reopen it, or is the "
                     "closure interactive/stream-state?"),
        "nets": {k: str(NETS[k].relative_to(E43.REPO)).replace("\\", "/")
                 for k in NETS},
        "compute": {"device": "cpu", "torch_threads": torch.get_num_threads(),
                    "eval_only": True,
                    "note": "three other agents on this host — 4 threads, "
                            "single staggered launch, no busy-waiting"},
        "instrument": {
            "door_ladder": {k: {"battery": b, "offset": j, "wpe_rows": r,
                                "window_len": (PRE if b == "home" else
                                               PRE + RETEACH_J) + j}
                            for k, (b, j, r) in LADDER.items()},
            "r49_correction": "e166's '+0.0000' was an instrument tautology "
                              "(home battery never reads wpe rows 183-189); "
                              "row claims below are licensed ONLY by ladder "
                              "cells; wpe_block re-reads e166's exact surgery "
                              "with home/-12 bit-equality gated as the "
                              "in-run tautology demonstration",
            "ladder_consistency": {t: gates[f"G_{t.upper()}"]
                                   ["ladder_consistency"]
                                   for t in ("twodoor", "root")}},
        "gates": gates,
        "gates_surgery": gates_surg,
        "class_energy": wdiff,
        "class_table": {c: {"n_tensors": len(CLASSES[c]),
                            "n_params": sum(sd["twodoor"][k].numel()
                                            for k in CLASSES[c]),
                            "registered": c in ("mlp_ln", "attn", "wpe")}
                        for c in CLASSES},
        "base_dial": base,
        "arms": arms,
        "graded": graded,
        "adjudication": adjudication,
        "honesty_reflex": {
            "instrument_tautology": "the home door battery reads only wpe "
                "rows 0-141; e166's row-restore null was vacuous on it and "
                "this experiment's row claims rest ONLY on ladder cells "
                "(long-12: rows 0-171; long0: 0-183; long+12: 0-195) and "
                "the 183-site census; the wpe_block control demonstrates "
                "the tautology in-run (home and long-12 bit-equal to base "
                "by causal construction, gated)",
            "swap_path_dependence": "bounded by the all-restored gate: the "
                "class restores compose to the root state dict bit-exactly "
                "and reproduce its dial "
                + ("bit-close" if gates["G_ALL_RESTORED"]["pass"]
                   else "ONLY at fallback tolerance — CAVEAT ATTACHED to "
                        "every single-class read")
                + "; single-class failures are therefore about isolation, "
                  "not machinery",
            "isolation_vs_interaction": "a class that fails alone in a "
                "hybrid net does not show the closure is absent from that "
                "class's weights — the restored class computes in the "
                "OTHER classes' (converted) context; NOTHING-CARRIES is "
                "precisely the interactive/stream-state reading, not an "
                "absence claim",
            "ce_pricing": "every arm carries its CE price; door movements "
                "bought at dCE > +0.1 nats are flagged as wreckage "
                "candidates in the verdict clause, never re-adjudicated",
            "graft_intact": "construction-level (wpe bit-untouched, "
                "confinement gates); the census co-reports graft expression "
                "— a span collapse on a bar-crossing arm is a FLAG "
                "(graft-silent), not a re-adjudication",
            "single_lineage": "one net pair (one 300-step conversion, seed "
                "10902, one root seed) — n=1 until replicated",
            "ladder_blindness_map": "each ladder cell is blind to wpe rows "
                "above its window: home > 141, long-12 > 171, long0 > 183 "
                "(row 183 itself IS its readout row), long+12 > 195; rows "
                "196-255 are read by no door cell here (only corpus windows "
                "touch them via CE)",
        },
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": 2739072, "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "closure_partition.png", base, arms, graded, adjudication,
         wdiff, gates)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'closure_partition.png'}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot

def plot(path, base, arms, graded, adj, wdiff, gates):
    """Legible: (top) the bar panel, partition accounting, dial grid;
    (bottom) door ladder, graft census, verdict box."""
    fig = plt.figure(figsize=(19.5, 12.5))
    gs = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.2), hspace=0.34,
                          wspace=0.28)
    root_gm12 = base["root"]["base"][-12]["mean_pz"]
    td_gm12 = base["twodoor"]["base"][-12]["mean_pz"]

    # (0,0) THE BAR PANEL: class arms' home g-12 vs the registered bars
    ax = fig.add_subplot(gs[0, 0])
    names, vals, cols = [], [], []
    names.append("twodoor\nbase"); vals.append(td_gm12); cols.append("dimgray")
    for c, col in (("mlp_ln", "darkorange"), ("attn", "indianred"),
                   ("wpe", "tab:blue"), ("io", "lightgray"),
                   ("wpe_block", "lavender")):
        if c in arms:
            lab = c + ("\n[rep-only]" if not arms[c]["bar_eligible"] else "")
            names.append(lab); vals.append(arms[c]["gm12"]); cols.append(col)
    names.append("ALL-restored\n[sanity gate]")
    vals.append(arms["all"]["gm12"]); cols.append("steelblue")
    names.append("root\nbase"); vals.append(root_gm12); cols.append("seagreen")
    xs = np.arange(len(names))
    ax.bar(xs, vals, 0.62, color=cols, edgecolor="k", lw=0.5)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=6.6)
    ax.axhline(REOPEN_BAR, color="seagreen", ls="--", lw=1.4)
    ax.axhline(NOTHING_CEIL, color="gray", ls="--", lw=1.2)
    ax.text(len(names) - 0.4, REOPEN_BAR + 0.015,
            f"MLP-LN-CARRIES bar {REOPEN_BAR}", fontsize=7, color="seagreen",
            ha="right")
    ax.text(len(names) - 0.4, NOTHING_CEIL + 0.015,
            f"NOTHING-CARRIES ceiling {NOTHING_CEIL}", fontsize=7,
            color="gray", ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=6.6)
    ax.set_ylabel("HOME battery p(Z) g-12 (wpe rows 0-141)")
    ax.set_ylim(0, 1.08)
    ax.set_title("(a-c) THE CLOSURE PARTITION — class restores on "
                 "e151_twodoor\nhome g-12 (green >= bar | orange 0.27-0.45 | "
                 "gray <= 0.27)", fontsize=9)

    # (0,1) partition accounting: delta-energy share vs door fraction
    ax = fig.add_subplot(gs[0, 1])
    es = wdiff["energy_shares"]
    cls_e = {"mlp_ln": es["mlps"] + es["lns"], "attn": es["attn"],
             "wpe": es["wpe"], "io": es["wte"] + es["lm_head"]}
    ks = [k for k in cls_e if k in arms]
    xs2 = np.arange(len(ks))
    ax.bar(xs2 - 0.19, [cls_e[k] * 100 for k in ks], 0.36, color="dimgray",
           edgecolor="k", lw=0.4, label="% of conversion delta energy (e153 "
                                       "wiring diff, recomputed)")
    ax.bar(xs2 + 0.19, [np.clip(arms[k]["door_fraction"], -1.5, 1.5) * 100
                        for k in ks], 0.36, color="darkorange", edgecolor="k",
           lw=0.4, label="% of the shut door reopened (home g-12, clipped)")
    for x, k in zip(xs2, ks):
        ax.text(x - 0.19, cls_e[k] * 100 + 1.2, f"{cls_e[k] * 100:.1f}%",
                ha="center", fontsize=6.6)
        ax.text(x + 0.19, np.clip(arms[k]["door_fraction"], -1.5, 1.5) * 100
                + (1.2 if arms[k]["door_fraction"] >= 0 else -4.0),
                f"{arms[k]['door_fraction'] * 100:+.1f}%", ha="center",
                fontsize=6.6)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs2)
    ax.set_xticklabels(ks, fontsize=8)
    ax.set_ylabel("%")
    ax.set_title("(A) PARTITION ACCOUNTING — where the 300 steps went vs "
                 "what each class restores alone\n(all-restored = 100% by "
                 "construction; single-class bars are isolation reads)",
                 fontsize=8.5)
    ax.legend(fontsize=6.4, loc="upper right")

    # (0,2) dial grid heatmap
    ax = fig.add_subplot(gs[0, 2])
    cols_h = ["g0", "g-12", "g+12", "long\n-12", "long\n0", "long\n+12",
              "site\nonset", "site\nspan", "A129", "CE_R", "dCE"]
    rows = [("twodoor base", base["twodoor"])]
    order = [c for c in ("mlp_ln", "attn", "wpe", "io", "wpe_block", "all")
             if c in arms]
    for c in order:
        rows.append((f"{c}" + ("" if arms[c]["bar_eligible"]
                               or c == "all" else " *"), arms[c]["dial"]))
    rows.append(("root base", base["root"]))
    M = np.zeros((len(rows), len(cols_h)))
    for i, (_, d) in enumerate(rows):
        M[i] = [d["base"][0]["mean_pz"], d["base"][-12]["mean_pz"],
                d["base"][12]["mean_pz"], d["door_long"][-12]["mean_pz"],
                d["door_long"][0]["mean_pz"], d["door_long"][12]["mean_pz"],
                d["site_onset"], d["site_span"], d["A129"], d["ce_r"],
                np.nan]
        if 0 < i < len(rows) - 1:
            M[i, 10] = rows[i][1]["ce_r"] - base["twodoor"]["ce_r"]
    lo = np.nanmin(M)
    norm = (M - lo) / (np.nanmax(M) - lo + 1e-12)
    norm[:, 9] = 1 - (M[:, 9] - M[:, 9].min()) / (M[:, 9].max() - M[:, 9].min()
                                                  + 1e-12)
    norm[:, 10] = 1 - np.clip((M[:, 10] + 0.1) / 0.8, 0, 1)
    ax.imshow(np.clip(norm, 0, 1), cmap="RdYlGn", aspect="auto", vmin=0,
              vmax=1)
    for i in range(len(rows)):
        for j in range(len(cols_h)):
            v = M[i, j]
            ax.text(j, i, f"{v:.3f}" if not np.isnan(v) else "-", ha="center",
                    va="center", fontsize=6.4)
    ax.set_xticks(range(len(cols_h)))
    ax.set_xticklabels(cols_h, fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_title("(B) DIAL GRID — * = report-only arm; ladder cols long-12/0/"
                 "+12 see wpe rows 0-171/0-183/0-195\n(home cols see 0-141 "
                 "only — the e166 tautology, corrected)", fontsize=8)

    # (1,0) door ladder per arm
    ax = fig.add_subplot(gs[1, 0])
    ladder_keys = [("home g-12", lambda d: d["base"][-12]["mean_pz"], "rows 0-141"),
                   ("long -12", lambda d: d["door_long"][-12]["mean_pz"], "rows 0-171"),
                   ("long 0", lambda d: d["door_long"][0]["mean_pz"], "rows 0-183"),
                   ("long +12", lambda d: d["door_long"][12]["mean_pz"], "rows 0-195")]
    show = [("twodoor", base["twodoor"], "dimgray", "twodoor base"),
            ("root", base["root"], "seagreen", "root base")] + \
           [(c, arms[c]["dial"],
             {"mlp_ln": "darkorange", "attn": "indianred", "wpe": "tab:blue",
              "io": "lightgray", "wpe_block": "purple",
              "all": "steelblue"}[c], c) for c in order]
    xs3 = np.arange(len(ladder_keys))
    for k, (lab, fn, rr) in enumerate(ladder_keys):
        for i, (_, d, col, nm) in enumerate(show):
            ax.bar(k + (i - len(show) / 2) * (0.8 / len(show)), fn(d),
                   0.8 / len(show) - 0.03, color=col, edgecolor="k", lw=0.3)
    ax.set_xticks(range(len(ladder_keys)))
    ax.set_xticklabels([f"{lab}\n({rr})" for lab, _, rr in ladder_keys],
                       fontsize=7)
    ax.set_ylabel("battery p(Z) at onset prediction")
    ax.set_ylim(0, 1.08)
    ax.set_title("(C) DOOR LADDER — nested wpe-row visibility per arm\n"
                 "row claims are licensed ONLY by these cells (R49)",
                 fontsize=8.5)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=col, ec="k")
                       for _, _, col, nm in show],
              labels=[nm for _, _, _, nm in show], fontsize=6.0, ncol=2,
              loc="upper center")

    # (1,1) graft census per arm + graded panel
    ax = fig.add_subplot(gs[1, 1])
    if graded:
        gs_l = [g["layer"] for g in graded]
        vs = [g["gm12"] for g in graded]
        ax.bar([f"L{l}" for l in gs_l], vs, 0.6, color="darkorange",
               edgecolor="k", lw=0.5)
        for x, v in zip(range(len(gs_l)), vs):
            ax.text(x, v + 0.015, f"{v:.3f}", ha="center", fontsize=7)
        ax.axhline(REOPEN_BAR, color="seagreen", ls="--", lw=1.2)
        ax.axhline(NOTHING_CEIL, color="gray", ls="--", lw=1.0)
        per = wdiff["per_layer_mlp_fro"]
        ax2 = ax.twinx()
        ax2.plot(range(6), [per[f"mlp_l{l}"] for l in range(6)], "ko--",
                 lw=0.9, ms=4, label="per-layer MLP ||delta|| (e153 context)")
        ax2.set_ylabel("MLP ||delta||_F root->twodoor", fontsize=7.5)
        ax2.legend(fontsize=6.2, loc="upper left")
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("home g-12 p(Z)")
        ax.set_title("(D) GRADED per-layer MLP+LN restores (triggered: "
                     "MLP+LN reopened)", fontsize=9)
    else:
        cens = ([("twodoor", base["twodoor"]), ("root", base["root"])]
                + [(c, arms[c]["dial"]) for c in order])
        xs4 = np.arange(len(cens))
        for off, ro, col in ((-0.17, "span", "crimson"),
                             (+0.17, "onset", "darkorange")):
            vals2 = []
            for nm, d in cens:
                s = d.get("census", {}).get("summary", {})
                vals2.append(s.get(ro, {}).get("site_strength", np.nan))
            ax.bar(xs4 + off, vals2, 0.32, color=col, edgecolor="k", lw=0.4,
                   label=f"183-census strength ({ro})")
            for x, v in zip(xs4 + off, vals2):
                if not np.isnan(v):
                    ax.text(x, v + (0.008 if v >= 0 else -0.02),
                            f"{v:+.3f}", ha="center", fontsize=5.8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(xs4)
        ax.set_xticklabels([nm for nm, _ in cens], rotation=38, ha="right",
                           fontsize=6.6)
        ax.set_ylabel("census strength (min of mean/zero arms)")
        ax.set_title("(D) 183-SITE CENSUS per arm — is the graft still "
                     "there / still read?\n(graft-intact check; graded "
                     "variant not triggered: mlp_ln "
                     f"{adj['class_gm12']['mlp_ln']:.3f} < {REOPEN_BAR})",
                     fontsize=8.2)
        ax.legend(fontsize=6.6)

    # (1,2) verdict panel
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    cg = adj["class_gm12"]
    vlines = [
        "REGISTERED (QUEUE e173 verbatim):",
        f"  MLP-LN-CARRIES: mlp_ln home g-12 >= {REOPEN_BAR} w/ graft",
        f"  NOTHING-CARRIES: every class alone <= {NOTHING_CEIL}",
        "  texture => TEXTURE with numbers",
        "",
        "DECIDING NUMBERS (home g-12):",
        f"  twodoor base {td_gm12:.4f} | root base {root_gm12:.4f}",
        f"  mlp_ln  {cg['mlp_ln']:.4f}   attn {cg['attn']:.4f}",
        f"  wpe     {cg['wpe']:.4f}   io {cg['io_report_only']:.4f} *",
        f"  wpe_block {cg['wpe_block_report_only']:.4f} *  all-restored "
        f"{cg['all_restored']:.4f}",
        f"  all-restored gate: "
        f"{'PASS' if adj['all_restored_gate']['pass'] else 'FAIL'}"
        f" (sd==root {adj['all_restored_gate']['sd_bit_equal_root']})",
        f"  tautology gate (wpe_block home==-base): "
        f"{gates['G_TAUTOLOGY_WPE_BLOCK']['pass']}",
        "",
        f"VERDICT: {adj['verdict']}",
    ] + [f"  {wd}" for wd in
         [adj["clause"][i:i + 60] for i in range(0, len(adj["clause"]), 60)]]
    for i, tx in enumerate(vlines):
        ax.text(0.02, 0.97 - i * 0.043, tx, fontsize=6.9, va="top",
                family="monospace")

    fig.suptitle("E173 — THE CLOSURE PARTITION: where does the shut geometry "
                 "door live in parameter space? — " + adj["verdict"],
                 fontsize=11.5)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
