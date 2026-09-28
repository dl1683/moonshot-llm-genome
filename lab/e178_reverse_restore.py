"""E178 — THE REVERSE CLASS-RESTORE: e176's mandatory rider.

WHY (T105 + T104): e176 froze the FULLY-CONSOLIDATED root (e131) on 300
steps of plain corpus and the fact DISSOLVED — g-12 0.916 -> 0.001, a
two-step collapse, at healthy CE; no gradient-resistance was ever
acquired (T105: the net has no archive, only rehearsal). SEPARATELY,
e173 localized the conversion's door-closure in parameter space: the
MLP+LN class carries it (single-class restore reopened g-12 to 0.782 =
84% of ceiling; graded band L2-L4 with L3 peaking). THE OPEN CELL this
run decides: does restoring the ROOT's MLP+LN class into the WASHED net
rescue the fact? RESCUE => the washout IS the located MLP+LN rewrite
operating BIDIRECTIONALLY (T104's disuse-connection claim gets its
second direction), and class-surgery restoration gains its second
demonstration. NO-RESCUE => the wash destroyed something the class
cannot rebuild (irreversible at the class level) — and the two-step
collapse's speed already hints the wash is fast and total.

DESIGN (eval-only class restore, e173's instrument run in reverse; no
training): TARGET = the +300 washed net (runs/checkpoints/
e176_root_freeze.pt — ON DISK, meta steps=300 seed=10902; gated below vs
e176's stored +300 cells BEFORE any surgery is trusted, so no
reconstruction of the washed state is needed or performed). DONOR = the
root (runs/checkpoints/e131_consolidated_e113.pt — the pre-wash state
AND the class donor; gated vs e151's stored before-cells, e176's
convention). ARMS:
  (a) MLP+LN  : the REGISTERED arm — all MLP weights + all LayerNorm
                params (ln1/ln2 per layer + ln_f; 50 tensors, e173's
                CLASSES['mlp_ln'] verbatim) restored root -> washed.
  (g) L2-L4   : report-only graded arm — the layer-local band e173
                localized (mlp_l + ln1_l + ln2_l for l in {2,3,4};
                ln_f not layer-local, excluded per e173's graded
                convention). No bars.
  (d) ALL     : sanity gate — every class restored; the state dict MUST
                equal the root bit-exactly and the dial must reproduce
                the root's bit-close (e173's artifact-bounding gate; not
                bar-eligible).
Per arm (full dial, e176's dial set): g-12/g+12/g0 (install-60
batteries, ctx = train_text[p-PRE-j:p]) + the held30 counterparts + the
functional site read @183 (instrument co-report) + the old-band census
(row-0 strength = the consolidated fact's sink; A(129) = the brake) +
D-all g0 + D-183 + CE_R (wreckage guard).

REGISTERED PREDICTION (QUEUE.md row e178 VERBATIM — no bar shopping):
  "Bars: RESCUE = the fact returns (g-12/g0 >= 0.5) — washout = the
  located MLP+LN rewrite bidirectionally; class-surgery restoration's
  second demo; NO-RESCUE = the wash destroyed what the class cannot
  rebuild (irreversible at the class level)"

OPERATIONALIZATIONS (frozen here before compute):
  * g-12 / g0 / g+12 / held30 = ABSOLUTE install-60 / held-30 battery
    mean p(Z) at ctx offsets -12 / 0 / +12 (e176's convention, same
    batteries, same corpus rebuild).
  * RESCUE (primary) = the mlp_ln arm's g-12 >= 0.50 AND g0 >= 0.50 —
    the registered clause's own two dials, on the same home battery the
    e176 SURVIVES clause used.
  * NO-RESCUE (primary) = the mlp_ln arm's g-12 <= 0.27 (the e158/e161/
    e176 SHUT bar; the QUEUE row's single named dial).
  * Anything else — g-12 in the gap band (0.27, 0.50), or g-12 >= 0.50
    with g0 < 0.50 (partial recovery) — fires neither primary =>
    TEXTURE with the numbers.
  * Adjudication order: RESCUE -> NO-RESCUE -> TEXTURE; every
    sub-boolean reported regardless.
  * CE annotates (e173's convention): dCE > +0.1 nats vs the washed
    base = WRECKAGE FLAG in the clause/honesty_reflex, never a
    re-adjudication.
  * FACT-SITE-SILENT FLAG (the e173 graft-silent convention, renamed
    for this lineage): the consolidated fact's site is ROW 0 (strength
    0.7317 at the root); a bar-crossing arm whose row-0 strength
    collapses below 0.5 is FLAGGED (the door reopened without the
    fact's sink), never re-adjudicated.
  * all-restored = the sanity gate (IS the root — trivially alive, not
    bar-eligible); graded L2-L4 = report-only location detail.
  * The e173 door ladder is NOT re-measured (no arm touches wpe; the
    mission's dial set is e176's home/held set; no row claims are made
    here). The e176 183-span site census is NOT re-run (the fact has
    no 183 site at either endpoint — root 0.00027 < 2x control, +300
    site_pos false; the functional site read @183 is kept as the
    instrument co-report).

INSTRUMENT PROVENANCE: restore_class / CLASSES / graded_keys /
class_energy are lab/e173_closure_partition.py VERBATIM (itself e153's
transplant gate generalized to key sets); load_cpu / evl_load /
battery_cell / battery_pz / ce_fixed_cpu / val_windows / deleted_wpe /
read_fact_at / row_census are lab/e176_freeze_root.py VERBATIM (the
e161/e152/e151/e143/e131/e119/e113/e068/e065/e043 lineage). The measure
dial is e176's measure() minus the 183-span census (dropped, recorded
above). Copied, not imported (thread/device pinning differs).

COMPUTE ENVELOPE: CPU-ONLY by dispatch (CUDA_VISIBLE_DEVICES=-1; e174
also on the CPU — LOW threads <= 4), one 25 s launch stagger (single
sleep, no busy-waiting anywhere), no training (eval-only, minutes).

Outputs: runs/e178/{metrics.json, reverse_restore.png}. No NOTES/
THINKING/QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python e178_reverse_restore.py    (E178_SMOKE=1 shakedown)
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e174 also
# on the CPU — threads capped at 4 below)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

torch.set_num_threads(4)                              # dispatch: LOW <= 4

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import Cfg, CharCorpus, TinyGPT, run_dir, save_json   # noqa: E402
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E178_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e178 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E176_METRICS = E43.REPO / "runs" / "e176" / "metrics.json"
NETS = {
    "washed": CKPT_DIR / "e176_root_freeze.pt",        # post-wash; fact DEAD
    "root": CKPT_DIR / "e131_consolidated_e113.pt",    # pre-wash; fact ALIVE;
}                                                       # class donor

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

# ---- parameter classes (e173 VERBATIM; frozen before compute) ------------------
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

CLASSES = {                     # e173's registered/sanity key sets verbatim
    "mlp_ln": _MLP + _LN,       # (a) the registered arm — 50 tensors
    "attn": _ATTN,              # (defined for coverage; NOT run here)
    "wpe": _WPE,                # (defined for coverage; NOT run here)
    "io": _IO,                  # (defined for coverage; NOT run here)
    "all": _MLP + _LN + _ATTN + _WPE + _IO,   # (d) must cover every key
}

def graded_keys(l):
    """e173's layer-local slice of the MLP+LN class (ln_f not layer-local)."""
    return _mlp_keys(l) + _ln_keys(l)

GRADED_BAND = (2, 3, 4)          # the band e173 localized (L3 peak 0.2946)

def band_keys(layers):
    return [k for l in layers for k in graded_keys(l)]

# ---- gates / references (full precision) ----------------------------------------
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05

E151_ROOT = {                     # runs/e151 'before' battery (e176's gate set)
    "gm12": 0.9155886769294739,
    "g0": 0.7850371599197388,
    "gp12": 0.9478210210800171,
    "ce_r": 1.663516640663147,
    "site_read_onset": 0.8898659348487854,
    "site_read_span": 0.982668936252594,
    "A129": -0.13237020391970877,
    "row0_strength": 0.7316772222270098,
    "dall_g0": 0.9047248959541321,
    # held30 refs from e176's own 4-thread step-0 measurement (bit-close)
    "held30_gm12": 0.6361417174339294,
    "held30_g0": 0.7233286499977112,
}

E176_WASHED = {                   # runs/e176 trace_summary @ +300 (the endpoint)
    "gm12": 0.0011210207594558597,
    "g0": 0.0012388104805722833,
    "gp12": 0.0017126891762018204,
    "held30_gm12": 0.0003840079589281231,
    "held30_g0": 0.0005695995059795678,
    "ce_r": 1.6334049701692674,
    "site_read_onset": 0.0015166971134021878,
    "site_read_span": 0.7192330360412598,
    "A129": -0.0006441907764686524,
    "row0_strength": -0.0005032020296397376,
    "dall_g0": 0.001604252029210329,
}

# ---- registered bar constants (frozen; see OPERATIONALIZATIONS above) ----------
RESCUE_BAR = 0.50                 # RESCUE clause: g-12 AND g0 (e176's SURVIVES bar)
NO_RESCUE_BAR = 0.27              # NO-RESCUE clause: g-12 (the e158/e161/e176 SHUT bar)
CE_WRECK_FLAG = 0.1               # dCE above this flags wreckage (annotation)
SITE_SILENT_BAR = 0.5             # row-0 strength below this on a bar-crossing arm
ROOT_GM12 = E151_ROOT["gm12"]
ROOT_G0 = E151_ROOT["g0"]
ROOT_ROW0 = E151_ROOT["row0_strength"]

REGISTERED_PREDICTION = {
    "rescue": "RESCUE fires if: the fact returns (g-12 >= 0.5 AND g0 >= 0.5) "
        "— washout is the located MLP+LN rewrite bidirectionally; "
        "class-surgery restoration's second demo.",
    "no_rescue": "NO-RESCUE fires if: the fact stays dead (g-12 <= 0.27) — "
        "the wash destroyed what the class cannot rebuild (irreversible at "
        "the class level).",
    "no_bar_shopping": "No bar shopping; texture (partial recovery) => "
        "TEXTURE with numbers.",
    "queue_row_verbatim": "Bars: RESCUE = the fact returns (g-12/g0 >= 0.5) "
        "— washout = the located MLP+LN rewrite bidirectionally; "
        "class-surgery restoration's second demo; NO-RESCUE = the wash "
        "destroyed what the class cannot rebuild (irreversible at the "
        "class level)",
    "operationalizations": "g-12/g0/g+12/held30 = absolute install-60 / "
        "held-30 battery mean p(Z) at ctx offsets -12/0/+12 (e176's "
        "convention, same batteries); RESCUE = mlp_ln arm g-12 >= 0.50 AND "
        "g0 >= 0.50; NO-RESCUE = mlp_ln arm g-12 <= 0.27; anything else "
        "(gap band, or g-12 >= 0.50 with g0 < 0.50) => TEXTURE with the "
        "numbers; order RESCUE -> NO-RESCUE -> TEXTURE; CE annotates (dCE > "
        "+0.1 vs washed base = wreckage flag, never re-adjudicated); row-0 "
        "strength < 0.5 on a bar-crossing arm = FACT-SITE-SILENT flag (the "
        "e173 graft-silent convention renamed); all-restored = sanity gate "
        "(not bar-eligible); graded L2-L4 = report-only.",
}

deviations: list[str] = [
    "CPU-ONLY by dispatch (CUDA_VISIBLE_DEVICES=-1 forced before torch; "
    "e174 also on the CPU): torch threads 4, one 25 s launch stagger "
    "(single sleep, no busy-waiting), eval-only — no training, no "
    "cooldowns needed.",
    "The washed net is the ON-DISK checkpoint runs/checkpoints/"
    "e176_root_freeze.pt (meta steps=300 seed=10902, base e131), gated vs "
    "e176's stored +300 cells BEFORE any surgery — the deterministic "
    "reconstruction path in the dispatch was NOT taken (the checkpoint "
    "inventory supported the direct path); fidelity recorded in G_WASH.",
    "Only the MLP+LN class arm is registered and adjudicated; e173's other "
    "partition arms (attn/wpe/io) are NOT re-run — e178 is the reverse "
    "direction on the WASH axis with the single class the QUEUE row names; "
    "the class key sets are kept verbatim for the coverage assert.",
    "The e173 door ladder is not re-measured (no arm touches wpe — the "
    "confinement gates prove it — and no row claims are made); the e176 "
    "183-span site census is not re-run (the fact has no 183 site at "
    "either endpoint: root 0.00027 < 2x control, +300 site_pos false); the "
    "functional site read @183 is kept as the instrument co-report.",
    "The graded L2-L4 arm carries e173's lean dial (base + held30 + CE + "
    "site read + quick A(129); no old-band census, no deletion table) — "
    "e173's graded convention (full=False, census=False); report-only, no "
    "bars.",
    "Eval thread count is 4 (dispatch) vs e151's stored 8-thread root "
    "cells — CPU reduction order can drift low-order bits; gates report "
    "both the 5e-6 bit flag and the 0.05 fallback tolerance (e153/e158/"
    "e161/e176 precedent; e176's own 4-thread cells gate near-bit).",
    "Single seed (10902) wash trajectory, single lineage (e065 install -> "
    "e109/e113 consolidation -> e176 wash) — the restore is deterministic "
    "surgery on ONE washed endpoint; n=1 until replicated.",
    "Smoke mode trims: lean measures everywhere (no censuses, no deletion "
    "table), graded arm skipped, gates on base cells + CE only; nothing "
    "adjudicated.",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: e173 (restore_class / class_energy) + e176 (everything else)
# VERBATIM — see the module docstring. Copied rather than imported to own
# the device policy.

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


# ------------------------------------------------------------------ surgery (e173)

def restore_class(sd_base: dict, sd_src: dict, keys: list, name: str
                  ) -> tuple[dict, dict]:
    """Replace the TARGET's (washed) tensors for `keys` with the SOURCE's
    (root); confinement gate = only class keys may change, everything else
    bit-identical (e153's transplant gate generalized to key sets, e173
    VERBATIM — only the roles of base/src reversed onto the washed net)."""
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


def class_energy(sd_a: dict, sd_b: dict) -> dict:
    """e173's wiring-diff class energy shares (root=a vs washed=b), VERBATIM
    arithmetic — here measuring WHERE THE WASH's 300 plain-corpus steps went
    (co-report for the bidirectionality claim)."""
    groups = {"attn": _ATTN, "mlps": _MLP,
              "lns": _LN_BLOCK + ["ln_f.weight", "ln_f.bias"],
              "wpe": _WPE, "wte": _IO[0:1], "lm_head": _IO[1:2]}
    E = {g: sum(float((sd_b[k] - sd_a[k]).norm() ** 2) for k in ks)
         for g, ks in groups.items()}
    tot = sum(E.values())
    per_layer = {f"mlp_l{l}": float(sum((sd_b[k] - sd_a[k]).norm() ** 2
                                        for k in _mlp_keys(l)) ** 0.5)
                 for l in range(N_LAYER)}
    return {"fro": {g: v ** 0.5 for g, v in E.items()},
            "energy_shares": {g: v / tot for g, v in E.items()},
            "total_fro": tot ** 0.5,
            "per_layer_mlp_fro": per_layer,
            "e153_conversion_shares": {"heads": 0.2999729803489722,
                                       "mlps": 0.6647335032129783,
                                       "lns": 0.008505192635032622,
                                       "wpe": 0.01773844020721051,
                                       "wte": 0.003683935172347457,
                                       "lm_head": 0.005365948423459058},
            "note": "energy shares of the WASH (root -> washed, 300 "
                    "plain-corpus steps) vs e153's stored shares of the "
                    "CONVERSION (root -> twodoor, 300 re-teach steps)"}


def load_e176_washed_ref() -> tuple[dict, dict]:
    """Verify the embedded E176_WASHED copy against runs/e176/metrics.json
    when present (no silent divergence; e176's load_e161_ref convention)."""
    src = {"source": "embedded verbatim copy (runs/e176/metrics.json "
                     "trace_summary @ +300)", "file_present":
           E176_METRICS.exists(), "verified_vs_embedded": None,
           "max_abs_diff": None}
    if E176_METRICS.exists():
        mm = json.loads(E176_METRICS.read_text(encoding="utf-8"))
        ts = mm["trace_summary"]
        keymap = {"gm12": "base_gm12", "g0": "base_g0", "gp12": "base_gp12",
                  "held30_gm12": "held30_gm12", "held30_g0": "held30_g0",
                  "ce_r": "ce_r", "site_read_onset": "site_read_onset",
                  "site_read_span": "site_read_span", "A129": "A129",
                  "row0_strength": "row0_strength", "dall_g0": "dall_g0"}
        diffs = [abs(ts[keymap[k]][-1] - E176_WASHED[k]) for k in keymap]
        src["max_abs_diff"] = max(diffs)
        src["verified_vs_embedded"] = bool(max(diffs) < 1e-9)
        if src["verified_vs_embedded"]:
            src["source"] = ("runs/e176/metrics.json trace_summary @ +300 "
                             "(embedded copy verified, max|diff| "
                             f"{max(diffs):.1e})")
    return dict(E176_WASHED), src


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e178_smoke" if SMOKE else "e178")
    log(f"E178 THE REVERSE CLASS-RESTORE (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()}), stagger "
        "25 s, eval-only (no training)")
    time.sleep(25.0)                  # launch stagger vs e174 (no busy-wait)

    # ---------------- protocol rebuild (e143/e151/e152/e161/e176 verbatim)
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

    # ---------------- batteries: ctx = train_text[p-PRE-j : p] (e119 verbatim)
    bat_ids, held_ids = {}, {}
    for j in GEOS:
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
        hs = [train_text[p - PRE - j: p] for p, _ in held_occ]
        held_ids[j] = torch.stack([corpus.encode(c) for c in hs])
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

    # ---------------- nets (washed + root) ------------------------------------
    sd, metas = {}, {}
    for tag in NETS:
        m = load_cpu(NETS[tag])
        sd[tag] = {k: v.clone() for k, v in m.state_dict().items()}
        st_raw = torch.load(NETS[tag], map_location="cpu", weights_only=False)
        if isinstance(st_raw, dict) and "meta" in st_raw:
            metas[tag] = E43.jsonable(st_raw["meta"])
        del m, st_raw
    # class coverage check: 'all' must cover every state-dict key exactly
    assert set(CLASSES["all"]) == set(sd["washed"].keys()), \
        "CLASSES['all'] does not cover the state dict exactly"
    n_mlp = sum(sd["washed"][k].numel() for k in CLASSES["mlp_ln"])
    log(f"nets loaded: washed {metas['washed']} | root {metas['root']}")
    log(f"class mlp_ln: {len(CLASSES['mlp_ln'])} tensors, {n_mlp} params "
        f"({n_mlp / 2739072 * 100:.1f}% of the net)")

    wdiff = class_energy(sd["root"], sd["washed"])
    log("wash delta energy (root -> washed): "
        + ", ".join(f"{k} {v * 100:.1f}%" for k, v in
                    wdiff["energy_shares"].items())
        + f" | total fro {wdiff['total_fro']:.2f}")

    gates_surg: dict = {}

    # ---------------- the dial (every arm, same instrument) -------------------
    def measure(sd_in: dict, tag: str, lean: bool = False) -> dict:
        """e176's measure() minus the 183-span census (recorded deviation):
        base 3-geos + held30 + CE_R + site read + old-band census (row-0
        sink / A129 brake) + deletion table (D-all, D-183). lean=True (the
        graded convention) drops the census + deletions, keeps a quick
        A(129)."""
        net = evl_load(sd_in)
        out: dict = {"tag": tag}
        out["base"] = {j: battery_cell(net, bat_ids[j], zid) for j in GEOS}
        out["base_held"] = {j: battery_cell(net, held_ids[j], zid)
                            for j in GEOS}
        out["ce_r"] = ce_fixed_cpu(net, *r_eval_xy)
        log(f"[{tag}] base: " + " ".join(f"g{j:+d} "
                                         f"{out['base'][j]['mean_pz']:.4f}"
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
                sd_d, gate = deleted_wpe(sd_in, rows_)
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
            net.load_state_dict(sd_in)
            log(f"[{tag}] deletions g0: " + " | ".join(
                f"{dl} {out['del_table'][dl]['g0']:.3f}" for dl in DELS))
        else:
            # quick A(129): mean/zero arms at row 129 on the g0 battery
            # (e140/e151 convention; the lean graded dial's brake read)
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
        c = {"g0": m["base"][0]["mean_pz"], "gm12": m["base"][-12]["mean_pz"],
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

    def gate_vs(cells: dict, refs: dict, name: str,
                require_all: bool = True) -> dict:
        missing = [k for k in refs if k not in cells]
        if require_all and missing:
            raise RuntimeError(f"{name}: gate refs missing from cells: "
                               f"{missing}")
        keys = [k for k in refs if k in cells]
        diffs = {k: cells[k] - refs[k] for k in keys}
        max_abs = max(abs(v) for v in diffs.values())
        g = {"cells": {k: cells[k] for k in keys}, "refs": refs,
             "skipped_missing": missing, "diffs": diffs,
             "max_abs_diff": max_abs,
             "bit_tol": G_BIT_TOL, "tol": G_FALLBACK_TOL,
             "bit": bool(max_abs < G_BIT_TOL),
             "pass": bool(max_abs < G_FALLBACK_TOL)}
        log(f"GATE {name}: max|diff| {max_abs:.2e} (tol {G_FALLBACK_TOL}): "
            + ("PASS" if g["pass"] else "FAIL") + (" (bit)" if g["bit"] else ""))
        return g

    # ---------------- base dials + gates --------------------------------------
    gates: dict = {"G_SPLICE": {"install_mix": mix, "pass": True}}
    e176_ref, e176_src = load_e176_washed_ref()
    gates["G_E176_REF"] = {"provenance": e176_src, "pass": bool(
        e176_src["verified_vs_embedded"] is not False)}

    base = {}
    base["root"] = measure(sd["root"], "root")
    g_root = gate_vs(flat_cells(base["root"]), E151_ROOT, "G_ROOT",
                     require_all=not SMOKE)
    gates["G_ROOT"] = g_root
    if not g_root["pass"]:
        raise RuntimeError("root failed its gate vs e151 stored before-cells")

    base["washed"] = measure(sd["washed"], "washed")
    g_wash = gate_vs(flat_cells(base["washed"]), e176_ref, "G_WASH",
                     require_all=not SMOKE)
    gates["G_WASH"] = g_wash
    if not g_wash["pass"]:
        raise RuntimeError("washed net failed its gate vs e176 stored +300 "
                           "cells — checkpoint/reconstruction fidelity broken")

    # ---------------- (a) THE REGISTERED ARM: root's MLP+LN into the washed net
    log("=" * 78)
    sd_arm, g_arm = restore_class(sd["washed"], sd["root"],
                                  CLASSES["mlp_ln"], "mlp_ln")
    gates_surg["arm_mlp_ln"] = g_arm
    if not g_arm["pass"]:
        raise RuntimeError(f"class gate FAILED mlp_ln: {g_arm}")
    arm = measure(sd_arm, "arm_mlp_ln")
    arm_cells = flat_cells(arm)
    w_cells, r_cells = flat_cells(base["washed"]), flat_cells(base["root"])
    arm_rec = {
        "arm": "mlp_ln", "registered": True, "bar_eligible": True,
        "gate": g_arm, "dial": arm, "cells": arm_cells,
        "gm12": arm_cells["gm12"], "g0": arm_cells["g0"],
        "delta_gm12": arm_cells["gm12"] - w_cells["gm12"],
        "delta_g0": arm_cells["g0"] - w_cells["g0"],
        "delta_ce_vs_washed": arm_cells["ce_r"] - w_cells["ce_r"],
        "delta_ce_vs_root": arm_cells["ce_r"] - r_cells["ce_r"],
        "retention_gm12": (arm_cells["gm12"] - w_cells["gm12"])
                          / max(r_cells["gm12"] - w_cells["gm12"], 1e-12),
        "retention_g0": (arm_cells["g0"] - w_cells["g0"])
                        / max(r_cells["g0"] - w_cells["g0"], 1e-12),
    }
    log(f"ARM mlp_ln [REGISTERED]: g-12 {arm_cells['gm12']:.4f} (washed "
        f"{w_cells['gm12']:.4f}, root {r_cells['gm12']:.4f}; retention "
        f"{arm_rec['retention_gm12'] * 100:+.1f}%) g0 {arm_cells['g0']:.4f} "
        f"(retention {arm_rec['retention_g0'] * 100:+.1f}%) | held30 "
        f"{arm_cells['held30_gm12']:.4f}/{arm_cells['held30_g0']:.4f} | "
        f"row0 {arm_cells.get('row0_strength', float('nan')):+.4f} A(129) "
        f"{arm_cells['A129']:+.4f} D-all {arm_cells.get('dall_g0', float('nan')):.4f} "
        f"| CE {arm_cells['ce_r']:.4f} (dCE vs washed "
        f"{arm_rec['delta_ce_vs_washed']:+.4f})")

    # ---------------- (g) graded L2-L4 band (report-only) ---------------------
    graded: dict = {}
    if not SMOKE:
        keys_band = band_keys(GRADED_BAND)
        sd_band, g_band = restore_class(sd["washed"], sd["root"], keys_band,
                                        f"mlp_ln_L{GRADED_BAND[0]}L{GRADED_BAND[-1]}")
        gates_surg["graded_L2_L4"] = g_band
        if not g_band["pass"]:
            raise RuntimeError(f"graded gate FAILED L2-L4: {g_band}")
        gd = measure(sd_band, "graded_L2_L4", lean=True)
        gc_ = flat_cells(gd)
        graded = {"arm": f"mlp_ln_L{GRADED_BAND[0]}..L{GRADED_BAND[-1]}",
                  "keys": keys_band, "n_tensors": len(keys_band),
                  "registered": False, "bar_eligible": False, "gate": g_band,
                  "dial": gd, "cells": gc_, "gm12": gc_["gm12"],
                  "g0": gc_["g0"],
                  "delta_ce_vs_washed": gc_["ce_r"] - w_cells["ce_r"]}
        log(f"ARM graded L{GRADED_BAND[0]}-L{GRADED_BAND[-1]} [report-only; "
            f"e173's band]: g-12 {gc_['gm12']:.4f} g0 {gc_['g0']:.4f} | "
            f"held30 {gc_['held30_gm12']:.4f}/{gc_['held30_g0']:.4f} | "
            f"A(129) {gc_['A129']:+.4f} | CE {gc_['ce_r']:.4f} (dCE "
            f"{graded['delta_ce_vs_washed']:+.4f})")

    # ---------------- (d) ALL-RESTORED sanity gate ----------------------------
    sd_all, g_all = restore_class(sd["washed"], sd["root"], CLASSES["all"],
                                  "all")
    gates_surg["arm_all"] = g_all
    if not g_all["pass"]:
        raise RuntimeError(f"all-restored gate FAILED: {g_all}")
    sd_bit_equal_root = all(torch.equal(sd_all[k], sd["root"][k])
                            for k in sd_all)
    dial_all = measure(sd_all, "arm_all")
    c_all, c_root = flat_cells(dial_all), flat_cells(base["root"])
    shared = [k for k in c_root if k in c_all]
    diffs_all = {k: abs(c_all[k] - c_root[k]) for k in shared}
    max_diff = max(diffs_all.values())
    bit_all = max_diff < G_BIT_TOL
    pass_all = max_diff < G_FALLBACK_TOL
    gates["G_ALL_RESTORED"] = {
        "sd_bit_equal_root": bool(sd_bit_equal_root),
        "max_cell_diff": max_diff, "cell_diffs": diffs_all,
        "bit": bool(bit_all), "pass": bool(pass_all),
        "note": "artifact-bounding gate (e173 convention): the classes "
                "compose to the root; if this fails, swap path-dependence "
                "contaminates and the single-class read carries that caveat"}
    all_rec = {"arm": "all", "registered": False, "bar_eligible": False,
               "gate": g_all, "dial": dial_all, "cells": c_all,
               "gm12": c_all["gm12"], "g0": c_all["g0"],
               "delta_ce_vs_washed": c_all["ce_r"] - w_cells["ce_r"],
               "sd_bit_equal_root": bool(sd_bit_equal_root)}
    log(f"ARM all-restored [sanity gate]: g-12 {c_all['gm12']:.4f} CE "
        f"{c_all['ce_r']:.4f} | sd==root bits {sd_bit_equal_root} | max "
        f"dial diff vs root {max_diff:.2e} -> "
        + ("PASS" if pass_all else "FAIL") + (" (bit)" if bit_all else ""))
    if not sd_bit_equal_root:
        raise RuntimeError("all-restored state dict is NOT bit-equal to root "
                           "— class key sets are wrong")
    if not pass_all:
        raise RuntimeError("all-restored dial does not reproduce the root "
                           "at fallback tolerance — machinery contaminated")

    # ---------------- adjudication (registered; no shopping) ------------------
    log("=" * 78)
    m_gm12, m_g0 = arm_rec["gm12"], arm_rec["g0"]
    rescue = bool(m_gm12 >= RESCUE_BAR and m_g0 >= RESCUE_BAR)
    no_rescue = bool(m_gm12 <= NO_RESCUE_BAR and not rescue)
    gap_g = bool(NO_RESCUE_BAR < m_gm12 < RESCUE_BAR)
    g0_lag = bool(m_gm12 >= RESCUE_BAR and m_g0 < RESCUE_BAR)
    site_silent = bool(rescue and arm_cells.get("row0_strength", 1.0)
                       < SITE_SILENT_BAR)
    wreck = bool(arm_rec["delta_ce_vs_washed"] > CE_WRECK_FLAG)
    if rescue:
        verdict = "RESCUE"
        clause = (f"the fact returns: the single-class MLP+LN restore "
                  f"brought g-12 {m_gm12:.4f} >= {RESCUE_BAR} AND g0 "
                  f"{m_g0:.4f} >= {RESCUE_BAR} (from washed base "
                  f"{w_cells['gm12']:.4f}/{w_cells['g0']:.4f}; root "
                  f"{r_cells['gm12']:.4f}/{r_cells['g0']:.4f}; retentions "
                  f"{arm_rec['retention_gm12'] * 100:.1f}%/"
                  f"{arm_rec['retention_g0'] * 100:.1f}%) — the washout IS "
                  f"the located MLP+LN rewrite operating bidirectionally "
                  f"(T104's disuse-connection gets its second direction), "
                  f"and class-surgery restoration gains its second "
                  f"demonstration"
                  + (" [FACT-SITE-SILENT FLAG: row-0 strength "
                     f"{arm_cells['row0_strength']:+.4f} < {SITE_SILENT_BAR} "
                     "while the doors reopened — door and fact-site "
                     "decoupled; reported, not re-adjudicated]"
                     if site_silent else "")
                  + (" [WRECKAGE FLAG: dCE > +0.1 vs washed]" if wreck
                     else "")
                  + ".")
    elif no_rescue:
        verdict = "NO-RESCUE"
        clause = (f"the fact stays dead: the MLP+LN restore left g-12 "
                  f"{m_gm12:.4f} <= {NO_RESCUE_BAR} (g0 {m_g0:.4f}; washed "
                  f"base {w_cells['gm12']:.4f}; root "
                  f"{r_cells['gm12']:.4f}) while the all-restored net IS "
                  f"the root ({all_rec['gm12']:.4f}) — the wash destroyed "
                  f"something the class cannot rebuild: irreversible at "
                  f"the class level (the restored MLP+LN computes in the "
                  f"washed attention/wpe/io context and that context no "
                  f"longer supports the fact)."
                  + (" [WRECKAGE FLAG]" if wreck else ""))
    else:
        verdict = "TEXTURE"
        clause = (f"no registered bar fired cleanly: MLP+LN restore g-12 "
                  f"{m_gm12:.4f} (RESCUE >= {RESCUE_BAR} needs BOTH g-12 "
                  f"and g0; NO-RESCUE <= {NO_RESCUE_BAR}), g0 {m_g0:.4f} "
                  f"(in-gap {gap_g}; g0-lag {g0_lag}); held30 "
                  f"{arm_cells['held30_gm12']:.4f}/"
                  f"{arm_cells['held30_g0']:.4f}; row-0 "
                  f"{arm_cells.get('row0_strength', float('nan')):+.4f} vs "
                  f"root {ROOT_ROW0:+.4f}; A(129) {arm_cells['A129']:+.4f}; "
                  f"D-all {arm_cells.get('dall_g0', float('nan')):.4f}; "
                  f"CE {arm_cells['ce_r']:.4f} (dCE vs washed "
                  f"{arm_rec['delta_ce_vs_washed']:+.4f}) — partial "
                  f"recovery is the registered texture case; numbers "
                  f"reported, no bar shopping.")
    adjudication = {
        "bars": {"rescue_g_and_g0": RESCUE_BAR, "no_rescue_g": NO_RESCUE_BAR,
                 "ce_wreck_flag": CE_WRECK_FLAG,
                 "site_silent_bar": SITE_SILENT_BAR,
                 "note": "g-12/g0 = absolute install-60 battery mean p(Z) "
                         "(e176's home battery convention — the same cells "
                         "the e176 SURVIVES/SHUT clauses used)"},
        "arm_cells": arm_cells, "washed_cells": w_cells, "root_cells": r_cells,
        "rescue_fires": rescue, "no_rescue_fires": no_rescue,
        "gap_band_g": gap_g, "g0_lag": g0_lag,
        "retention_gm12": arm_rec["retention_gm12"],
        "retention_g0": arm_rec["retention_g0"],
        "site_silent_flag": site_silent, "wreck_flag": wreck,
        "all_restored_gate": {"pass": gates["G_ALL_RESTORED"]["pass"],
                              "bit": gates["G_ALL_RESTORED"]["bit"],
                              "sd_bit_equal_root": sd_bit_equal_root},
        "graded_report_only": graded if graded else None,
        "verdict": verdict, "clause": clause,
    }
    log(f"E178 VERDICT: {verdict}")
    log(f"  mlp_ln restore: g-12 {m_gm12:.4f} g0 {m_g0:.4f} (bars: rescue "
        f">= {RESCUE_BAR} both; no-rescue <= {NO_RESCUE_BAR}) | washed "
        f"{w_cells['gm12']:.4f}/{w_cells['g0']:.4f} | root "
        f"{r_cells['gm12']:.4f}/{r_cells['g0']:.4f} | all-restored "
        f"{all_rec['gm12']:.4f}/{all_rec['g0']:.4f}")
    if graded:
        log(f"  graded L2-L4: g-12 {graded['gm12']:.4f} g0 {graded['g0']:.4f}")
    log(f"  {clause}")

    # ---------------- outputs -------------------------------------------------
    metrics = {
        "experiment": "e178_reverse_restore",
        "date": common.now_iso(),
        "registration": ("QUEUE.md row e178 (DISPATCHED ~15:25Z, CPU "
                         "eval-only, minutes) — e176's mandatory rider; "
                         "T105/T104 claims; registered prediction verbatim "
                         "below; operationalizations frozen in the module "
                         "docstring before compute"),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("does restoring the ROOT's MLP+LN class into the "
                     "+300 washed net rescue the fact — is the washout the "
                     "located MLP+LN rewrite operating bidirectionally "
                     "(RESCUE), or did the wash destroy something the class "
                     "cannot rebuild (NO-RESCUE)?"),
        "nets": {k: {"path": str(NETS[k].relative_to(E43.REPO)).replace("\\", "/"),
                     "meta": metas.get(k)}
                 for k in NETS},
        "compute": {"device": "cpu", "torch_threads": torch.get_num_threads(),
                    "eval_only": True, "stagger_s": 25.0,
                    "note": "e174 also on this host — 4 threads, single "
                            "staggered launch, no busy-waiting, no training"},
        "instrument": {
            "restore": "e173's restore_class VERBATIM, roles reversed: "
                       "target = the +300 washed net (e176_root_freeze.pt), "
                       "source = the root (e131_consolidated_e113.pt); "
                       "confinement gate per arm (only class keys change)",
            "dial": "e176's measure() minus the 183-span census (no 183 "
                    "site at either endpoint — recorded deviation): base "
                    "3-geos + held30 + CE_R + site read @183 + old-band "
                    "census (row-0 sink / A129 brake) + D-all + D-183",
            "washed_state": "the ON-DISK e176 endpoint checkpoint, gated vs "
                            "e176's stored +300 cells (G_WASH) — the "
                            "deterministic reconstruction path was not "
                            "needed",
        },
        "gates": gates,
        "gates_surgery": gates_surg,
        "class_energy_wash": wdiff,
        "class_table": {"mlp_ln": {"n_tensors": len(CLASSES["mlp_ln"]),
                                   "n_params": n_mlp, "registered": True},
                        "graded_L2_L4": {"n_tensors": len(band_keys(GRADED_BAND)),
                                         "layers": list(GRADED_BAND),
                                         "registered": False},
                        "all": {"n_tensors": len(CLASSES["all"]),
                                "n_params": sum(sd["washed"][k].numel()
                                                for k in CLASSES["all"]),
                                "registered": False}},
        "base_dials": base,
        "arm": arm_rec,
        "graded": graded if graded else None,
        "all_restored": all_rec,
        "adjudication": adjudication,
        "honesty_reflex": {
            "washed_state_fidelity": "the washed net is e176's saved "
                f"+300 endpoint (meta steps=300, seed=10902), gated vs "
                f"e176's stored +300 cells at max|diff| "
                f"{g_wash['max_abs_diff']:.2e} "
                + ("(bit-level)" if g_wash["bit"] else "(fallback tol)")
                + " — no reconstruction was performed, so reconstruction "
                  "fidelity is not at issue; the gate proves the checkpoint "
                  "IS the cell e176 reported",
            "single_seed": "ONE wash trajectory (seed 10902) from ONE root "
                "lineage; the restore is deterministic surgery on that one "
                "endpoint — the verdict is a point estimate (n=1) until "
                "replicated on other wash endpoints (e.g. +50, other seeds)",
            "donor_history": "the class donor (root MLP+LN) is ONE lineage "
                "(e065 install seed lineage -> e109/e113 consolidation) — "
                "its weights are the pre-wash state of exactly this net; a "
                "differently-consolidated root's class might rescue "
                "differently; e173's forward demo used the same donor on a "
                "different target (twodoor), so the 'second demonstration' "
                "shares the donor (not independent of it)",
            "isolation_vs_interaction": "the restored MLP+LN computes in "
                "the WASHED attention/wpe/io context (a hybrid net): a "
                "NO-RESCUE is a claim about what the class cannot rebuild "
                "IN THAT CONTEXT, not an absence claim about the class's "
                "weights (they demonstrably carry the fact when the "
                "context is intact — they ARE the root's); symmetric to "
                "e173's isolation caveat",
            "swap_path_dependence": "bounded by the all-restored gate: the "
                "class restores compose to the root state dict "
                f"bit-exactly ({sd_bit_equal_root}) and reproduce its dial "
                + ("bit-close" if bit_all else "at fallback tolerance")
                + "; single-class reads are about isolation, not machinery",
            "ce_pricing": "the arm carries its CE price; a rescue bought at "
                "dCE > +0.1 nats vs the washed base is flagged as wreckage "
                "in the verdict clause, never re-adjudicated",
            "bars_anchored": "the rescue/shut bars are the same absolute "
                "home-battery bars e176's registered clauses used (0.50 "
                "SURVIVES / 0.27 SHUT) — the wash's death and the restore's "
                "life are measured on one ruler",
        },
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": 2739072, "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "reverse_restore.png", base, arm_rec, graded, all_rec,
         adjudication, gates)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'reverse_restore.png'}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot

def plot(path, base, arm_rec, graded, all_rec, adj, gates):
    """THE figure: the bar panel (the registered fork), the anatomy dials,
    the generalization + site dials, and the verdict panel."""
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 10.5))

    def flat(m):
        c = {"gm12": m["base"][-12]["mean_pz"],
             "g0": m["base"][0]["mean_pz"],
             "gp12": m["base"][12]["mean_pz"],
             "held30_gm12": m["base_held"][-12]["mean_pz"],
             "held30_g0": m["base_held"][0]["mean_pz"],
             "ce_r": m["ce_r"],
             "onset": m["site_read"]["pz_onset_mean"],
             "span": m["site_read"]["pname_mean_over7"]}
        if "old_band" in m:
            c["row0_strength"] = m["old_band"]["row0_strength"]
            c["A129"] = m["old_band"]["A129"]
            c["dall_g0"] = m["del_table"]["d_all"]["g0"]
        elif "A129_quick" in m:
            c["A129"] = m["A129_quick"]
        return c

    w = flat(base["washed"])
    r = flat(base["root"])
    a = flat(arm_rec["dial"])
    al = flat(all_rec["dial"])
    gr = flat(graded["dial"]) if graded else None

    # (0,0) THE BAR PANEL: g-12 and g0 per arm vs the registered bars
    ax = axes[0, 0]
    names = ["washed\nbase", "MLP+LN\nrestore\n[REGISTERED]"]
    vals_g, vals_g0, cols = [w["gm12"]], [w["g0"]], ["dimgray"]
    vals_g.append(a["gm12"]); vals_g0.append(a["g0"]); cols.append("darkorange")
    if gr is not None:
        names.append(f"L2-L4 band\n[report-only]")
        vals_g.append(gr["gm12"]); vals_g0.append(gr["g0"])
        cols.append("moccasin")
    names.append("ALL-restored\n[sanity gate]")
    vals_g.append(al["gm12"]); vals_g0.append(al["g0"]); cols.append("steelblue")
    names.append("root\nbase")
    vals_g.append(r["gm12"]); vals_g0.append(r["g0"]); cols.append("seagreen")
    xs = np.arange(len(names))
    ax.bar(xs - 0.19, vals_g, 0.36, color=cols, edgecolor="k", lw=0.5,
           label="g-12 (install-60)")
    ax.bar(xs + 0.19, vals_g0, 0.36, color=cols, edgecolor="k", lw=0.5,
           alpha=0.55, hatch="//", label="g0 (home)")
    for x, v1, v2 in zip(xs, vals_g, vals_g0):
        ax.text(x - 0.19, v1 + 0.012, f"{v1:.3f}", ha="center", fontsize=6.8)
        ax.text(x + 0.19, v2 + 0.012, f"{v2:.3f}", ha="center", fontsize=6.8)
    ax.axhline(RESCUE_BAR, color="seagreen", ls="--", lw=1.4)
    ax.axhline(NO_RESCUE_BAR, color="tab:purple", ls="--", lw=1.2)
    ax.text(len(names) - 0.4, RESCUE_BAR + 0.015,
            f"RESCUE bar {RESCUE_BAR} (g-12 AND g0)", fontsize=7,
            color="seagreen", ha="right")
    ax.text(len(names) - 0.4, NO_RESCUE_BAR + 0.015,
            f"NO-RESCUE bar {NO_RESCUE_BAR} (g-12)", fontsize=7,
            color="tab:purple", ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("absolute mean p(Z), install-60 battery")
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("(a) THE REVERSE CLASS-RESTORE — root's MLP+LN into the "
                 "+300 washed net\n(did the fact come back?)", fontsize=10)

    # (0,1) the anatomy: row-0 sink / A(129) brake / D-all / CE_R
    ax = axes[0, 1]
    an = [t for t in (a, al, r) if "row0_strength" in t]
    labs = ["MLP+LN" if t is a else ("ALL" if t is al else "root")
            for t in an]
    labs = ["washed"] + labs
    row0 = [w.get("row0_strength", np.nan)] + [t["row0_strength"] for t in an]
    a129 = [w.get("A129", np.nan)] + [t["A129"] for t in an]
    dall = [w.get("dall_g0", np.nan)] + [t["dall_g0"] for t in an]
    xs2 = np.arange(len(labs))
    ax.bar(xs2 - 0.26, row0, 0.24, color="tab:cyan", edgecolor="k", lw=0.4,
           label="row-0 strength (the fact's sink)")
    ax.bar(xs2, a129, 0.24, color="tab:purple", edgecolor="k", lw=0.4,
           label="A(129) (the brake)")
    ax.bar(xs2 + 0.26, dall, 0.24, color="tab:blue", edgecolor="k", lw=0.4,
           label="D-all g0 (deletion tolerance)")
    for x, v in zip(xs2 - 0.26, row0):
        if not np.isnan(v):
            ax.text(x, v + (0.012 if v >= 0 else -0.03), f"{v:+.3f}",
                    ha="center", fontsize=6.2)
    axr = ax.twinx()
    ces = [w["ce_r"]] + [t["ce_r"] for t in an]
    axr.plot(xs2, ces, "k:o", ms=6, lw=1.2, label="CE_R (wreckage guard)")
    axr.set_ylabel("CE_R")
    axr.set_ylim(min(ces) - 0.08, max(ces) + 0.08)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(xs2)
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("strength / p(Z)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    ax.set_title("(b) THE ANATOMY — sink / brake / deletion tolerance "
                 "(the dials that dissolved together in e176's two-step "
                 "collapse)", fontsize=10)

    # (1,0) generalization (held30) + the 183-site instrument read
    ax = axes[1, 0]
    hg = [w["held30_gm12"], a["held30_gm12"]]
    h0 = [w["held30_g0"], a["held30_g0"]]
    labs3 = ["washed", "MLP+LN"]
    if gr is not None:
        hg.append(gr["held30_gm12"]); h0.append(gr["held30_g0"])
        labs3.append("L2-L4*")
    hg += [al["held30_gm12"], r["held30_gm12"]]
    h0 += [al["held30_g0"], r["held30_g0"]]
    labs3 += ["ALL", "root"]
    xs3 = np.arange(len(labs3))
    ax.bar(xs3 - 0.19, hg, 0.36, color="crimson", edgecolor="k", lw=0.4,
           label="held30 g-12")
    ax.bar(xs3 + 0.19, h0, 0.36, color="tab:blue", edgecolor="k", lw=0.4,
           alpha=0.6, hatch="//", label="held30 g0")
    for x, v1, v2 in zip(xs3, hg, h0):
        ax.text(x - 0.19, max(v1, 0) + 0.012, f"{v1:.3f}", ha="center",
                fontsize=6.4)
        ax.text(x + 0.19, max(v2, 0) + 0.012, f"{v2:.3f}", ha="center",
                fontsize=6.4)
    ax.set_xticks(xs3)
    ax.set_xticklabels(labs3, fontsize=8)
    ax.set_ylabel("held30 mean p(Z)")
    ax.set_ylim(-0.03, 1.05)
    axr2 = ax.twinx()
    spans = [w["span"], a["span"], al["span"], r["span"]]
    onsets = [w["onset"], a["onset"], al["onset"], r["onset"]]
    axr2.plot(np.arange(4), spans, "D-", ms=6, lw=1.4, color="seagreen",
              label="site-read span @183")
    axr2.plot(np.arange(4), onsets, "v-", ms=6, lw=1.4,
              color="mediumseagreen", label="site-read onset @183")
    axr2.set_ylabel("site read @183 (instrument)")
    axr2.set_ylim(-0.03, 1.05)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    ax.set_title("(c) GENERALIZATION (held30) + the 183-site instrument "
                 "read (* = lean dial)", fontsize=10)

    # (1,1) verdict panel
    ax = axes[1, 1]
    ax.axis("off")
    vlines = [
        "REGISTERED (QUEUE e178 verbatim; frozen operationalizations):",
        f"  RESCUE: g-12 >= {RESCUE_BAR} AND g0 >= {RESCUE_BAR} (the fact "
        f"returns)",
        f"  NO-RESCUE: g-12 <= {NO_RESCUE_BAR} (the fact stays dead)",
        "  texture (partial recovery) => TEXTURE with numbers",
        "",
        "DECIDING NUMBERS (install-60 battery):",
        f"  washed  g-12 {w['gm12']:.4f}  g0 {w['g0']:.4f}  CE "
        f"{w['ce_r']:.4f}",
        f"  MLP+LN  g-12 {a['gm12']:.4f}  g0 {a['g0']:.4f}  CE "
        f"{a['ce_r']:.4f} (dCE {arm_rec['delta_ce_vs_washed']:+.4f})",
        f"  held30  {a['held30_gm12']:.4f}/{a['held30_g0']:.4f} | row-0 "
        f"{a.get('row0_strength', float('nan')):+.4f} | A(129) "
        f"{a['A129']:+.4f} | D-all {a.get('dall_g0', float('nan')):.4f}",
        f"  root    g-12 {r['gm12']:.4f}  g0 {r['g0']:.4f}  CE "
        f"{r['ce_r']:.4f}",
        f"  ALL-restored g-12 {al['gm12']:.4f} g0 {al['g0']:.4f} | sd==root "
        f"{all_rec['sd_bit_equal_root']} | gate "
        f"{'PASS' if adj['all_restored_gate']['pass'] else 'FAIL'}",
    ]
    if gr is not None:
        vlines.append(f"  L2-L4*  g-12 {gr['gm12']:.4f}  g0 {gr['g0']:.4f} "
                      f"(e173's band; report-only)")
    vlines += [
        f"  G_ROOT {gates['G_ROOT']['max_abs_diff']:.1e} | G_WASH "
        f"{gates['G_WASH']['max_abs_diff']:.1e}",
        "",
        f"VERDICT: {adj['verdict']}",
    ] + [f"  {wd}" for wd in
         [adj["clause"][i:i + 78] for i in range(0, len(adj["clause"]), 78)]]
    for i, tx in enumerate(vlines):
        ax.text(0.02, 0.97 - i * 0.036, tx, fontsize=7.3, va="top",
                family="monospace")

    fig.suptitle("E178 — THE REVERSE CLASS-RESTORE: root's MLP+LN into "
                 f"e176's washed net (the rider) -> {adj['verdict']}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
