"""E166 — THE INVERSE EVENT: the graft-removal door-restore test (QUEUE.md row
e166, DISPATCHED ~13:00Z; bars are the QUEUE row VERBATIM, written before
compute).

WHY (T097, the claim under test): e151's locked re-teach at novel site 183
built a graft (span census strength +0.073, onset +0.512 at row 183) AND shut
the geometry door (g-12 0.916 -> 0.102) — "ONE EVENT, TWO FACES: erecting a
new positional key tears down the geometry-general access." But that is a
TEMPORAL COINCIDENCE until inverted: does REMOVING the graft RESTORE the
door? T091 (e153) already showed the shut door is movable by transplant only
to ~1/3 of the reopen bar (reader_K3 g-12 0.1505 vs bar 0.45), and T037's
write-once core says surgery cannot CREATE access. The inverse event decides
between:
  * closure = ACTIVE COMPETITIVE INHIBITION by the graft (removal restores —
    build-and-travel is CAUSAL, one mechanism);
  * closure = DESTRUCTIVE REWRITE (removal restores nothing — the door is
    gone, not held shut; "one event" downgrades to same-step correlation;
    extending T037: doors open by training, are killed by surgery, and
    cannot be REOPENED by surgery — the third great asymmetry).

REGISTERED PREDICTION (QUEUE e166 VERBATIM — no bar shopping):
  "Bars: DOOR-RESTORES = any cell g-12 >= 0.5 at CE <= +0.1 (closure is
  ACTIVE competitive inhibition — build-and-travel CAUSAL); DOOR-STAYS-SHUT
  = all <= 0.27 (destructive rewrite; 'one event' downgrades to same-step
  correlation; third great asymmetry: unreopenable-by-surgery); PARTIAL
  0.27-0.5"

DESIGN (eval-only surgery, CPU-only, minutes):
  (a) ROW RESTORE: wpe[183:189] := the root's (e131_consolidated) values —
      the clean graft deletion; plus zero and mean comparison modes (the
      e139/e151 census arms, net-local values).
  (b) HEAD ABLATION of the door-movers: e153's reader set {L3H5, L0H3,
      L1H0} (twodoor census site-drop top-3; the set whose consolidated->
      twodoor transplant moved the shut door most, g-12 0.1505) and N2
      {L1H0, L0H0} (e160's top2-no-L0H3 kill set); zero (e133 lesion) and
      mean-replace (e150 CE_R-bank mean) modes.
  (c) JOINT: row restore x ALL FOUR head-ablation cells (the dispatch's
      "best head ablation" operationalized as the full 2x2 set-x-mode grid
      composed with the restore — decided by the same bars, no data-picked
      selection before compute).
  (d) CONTROLS: the same surgeries on the ROOT (ceiling — an open door
      cannot open further; its row-restore is a bit-identity no-op) and on
      e158_jitter183 (jittered replay around 183: NO graft formed, door at
      g-12 0.505 — no restore predicted; specificity).
  Dials per cell: g-12 AND g+12 (install-60 batteries), site expression at
  183 (onset + span read on the e151 pool), D-183 (zero row 183 on top of
  the cell), CE_R (the flat-CE discipline — every cell priced).

OPERATIONALIZATIONS (frozen here before compute):
  * g-12 = battery mean p(Z) install-60 at ctx offset j=-12 (e119/e151);
    twodoor base 0.1021, root base 0.9156, jitter base 0.7885 (e158 pass 2,
    commit 28ad653 — see deviations).
  * "CE <= +0.1" = cell CE_R minus the SAME net's intact base CE_R
    (twodoor 1.6490 / root 1.6635 / jitter 1.6602).
  * BAR-ELIGIBLE CELLS = the 11 surgery cells on the PRIMARY net
    (e151_twodoor): row{root,zero,mean} + head{reader3,n2}x{zero,mean} +
    joint{restore}x{reader3,n2}x{zero,mean}. The root/jitter columns are
    CONTROLS (report-only): the root's intact base already exceeds the
    restore bar, so controls cannot be bar-eligible by construction.
  * "any cell" / "all cells" quantified over the 11 eligible cells plus
    the primary base row (0.102 — inside every clause either way).
  * row mode "root" = wpe[183:189] := root's values (identity on the root
    itself — bit no-op gate); "zero" = rows := 0; "mean" = each row := the
    net's OWN wpe mean row (e139 census convention).
  * head mean-replace vectors are computed per cell on the CELL's state
    (row surgery applied first — the head's mean activity in the operated
    organism; e160 computed them on intact nets, recorded convention).
  * D-183 is degenerate (recorded, not rerun) where the cell already holds
    row 183 at zero (row__zero cells); the root's joint cells are degenerate
    (row-restore is identity there) and are skipped in its control column.
  * Specificity guard (pre-registered): if DOOR-RESTORES fires on the
    primary but the jitter control's row__root cell gains > max(0.5 x the
    primary's best gain, +0.10) g-12, the verdict DOWNGRADES to TEXTURE —
    a comparable move on a net with NO graft means the move is generic
    row-content effect, not graft removal.
  * Adjudication order: DOOR-RESTORES -> DOOR-STAYS-SHUT -> PARTIAL ->
    TEXTURE. Wreck note: a cell reaching g-12 >= 0.5 only at CE > +0.1
    blocks neither STAYS-SHUT's "<= 0.27" read nor PARTIAL — it is
    reported as a wreck-flagged cell inside PARTIAL.

NETS (on disk, gated vs stored cells BEFORE any surgery; tol 5e-6 bit /
0.05 fallback, e153's 4-thread precedent):
  * TWODOOR  runs/checkpoints/e151_twodoor.pt      (primary; e151 after)
  * ROOT     runs/checkpoints/e131_consolidated_e113.pt (restore source +
             ceiling control; e151 before)
  * JITTER   runs/checkpoints/e158_jitter183.pt    (specificity control;
             e158 after_a)

INSTRUMENT PROVENANCE: load_cpu / evl_load / battery_cell / battery_pz /
ce_fixed_cpu / val_windows / deleted_wpe / read_fact_at are lab/
e153_phase_surgery.py VERBATIM (the e151/e143/e131 lineage); HeadReplace /
head_mean_vec are lab/e160_headset_escalation.py VERBATIM (e150/e133
lineage); row_surgery is e141's modified_wpe generalized to a row block
with the same confinement gate. Copied, not imported (thread/device
pinning differs).

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1; GPU user-occupied;
e164 + e154 also on the CPU) — torch threads 4, single staggered launch,
no busy-waiting. Eval-only, no training. Well under 30 min.

Outputs: runs/e166/{metrics.json, inverse_event.png}. No NOTES/THINKING/
QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python e166_inverse_event.py     (E166_SMOKE=1 shakedown)
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # GPU user-occupied

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

SMOKE = os.environ.get("E166_SMOKE") == "1"
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
    "twodoor": CKPT_DIR / "e151_twodoor.pt",
    "root": CKPT_DIR / "e131_consolidated_e113.pt",
    "jitter": CKPT_DIR / "e158_jitter183.pt",
}

# ---- site geometry (e151 verbatim) --------------------------------------------
RETEACH_J = 54                    # name x-cols 184..190, read rows 183..189
SITE_ADDR_ROW = 183
SITE_Z_XCOL = PRE + RETEACH_J     # 184
SITE_CONT = BLOCK - PRE - RETEACH_J - len(NAME)     # 65
SITE_ROWS = tuple(range(183, 190))                  # the 7 grafted rows
D_ALL = (121, 125, 129, 133, 137)                   # e113's fixed set
GEOS = (-12, 0, 12)

# ---- pre-registered head sets (provenance: e153 reader_K3 / e160 N2) ----------
READER3 = ("L3H5", "L0H3", "L1H0")     # e153 reader order top-3 (site-drop)
N2SET = ("L1H0", "L0H0")               # e160 top2-no-L0H3
HEAD_SETS = {"reader3": READER3, "n2": N2SET}
HEAD_MODES = ("zero", "mean")
ROW_MODES = ("root", "zero", "mean")

# ---- gate references (runs/e151 metrics before/after + runs/e158 after_a) -----
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05
REFS = {
    "root": {         # e151 "before"
        "g0": 0.7850371599197388, "gm12": 0.9155886769294739,
        "gp12": 0.9478210210800171, "ce_r": 1.663516640663147,
        "site_onset": 0.8898659348487854, "site_span": 0.982668936252594,
        "A129": -0.13237020391970877,
        "d_all_g0": 0.9047248959541321, "d_r0_g0": 0.0533599890768524,
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
    "jitter": {       # e158 after_a PASS 2 (correction commit 28ad653; the
                      # checkpoint reproduces these, not pass 1's 0.505 — see
                      # deviations)
        "g0": 0.14524763822555542, "gm12": 0.7884896397590637,
        "gp12": 0.8614740967750549, "ce_r": 1.6601638793945312,
        "site_onset": 0.9983755946159363, "site_span": 0.999580442905426,
        "A129": -0.6100223067487055,
        "d_all_g0": 0.6698154211044312, "d_r0_g0": 0.0017258950974792242,
        "d183_site_onset": 0.988865315914154,
    },
}

# ---- registered bars (frozen) -------------------------------------------------
RESTORE_BAR = 0.5                 # g-12 absolute (primary net)
RESTORE_CE_BAR = 0.1              # dCE vs the net's own base
SHUT_BAR = 0.27                   # all-cells ceiling (= 0.30 x root base)
SPEC_GAIN_FLOOR = 0.10            # jitter specificity guard absolute part
CE_EVAL_SEED = 26502              # e065 CE_R bank seed

REGISTERED_PREDICTION = {
    "door_restores": "DOOR-RESTORES = any cell g-12 >= 0.5 at CE <= +0.1 "
        "(closure is ACTIVE competitive inhibition - build-and-travel "
        "CAUSAL).",
    "door_stays_shut": "DOOR-STAYS-SHUT = all <= 0.27 (destructive rewrite; "
        "'one event' downgrades to same-step correlation; third great "
        "asymmetry: unreopenable-by-surgery).",
    "partial": "PARTIAL 0.27-0.5.",
    "texture": "Texture => TEXTURE with numbers.",
    "operationalizations": "g-12 = install-60 battery mean p(Z) at j=-12; "
        "CE clause = cell CE_R - same-net intact base CE_R <= +0.1; "
        "bar-eligible = the 11 primary (twodoor) surgery cells (controls "
        "report-only); 'any/all cells' over eligible + primary base; row "
        "modes root/zero/mean as frozen in the docstring; joint = restore "
        "x all four head cells; head mean-replace vectors per cell state; "
        "D-183 degenerate cells recorded not rerun; specificity guard: "
        "RESTORES downgrades to TEXTURE if jitter row__root gains > "
        "max(0.5 x primary best gain, +0.10); order DOOR-RESTORES -> "
        "DOOR-STAYS-SHUT -> PARTIAL -> TEXTURE; no bar shopping.",
    "no_bar_shopping": "No bar shopping; texture => TEXTURE with numbers.",
}

deviations: list[str] = [
    "CPU-only at 4 torch threads (dispatch: GPU user-occupied; e164 + e154 "
    "share the CPU); e151/e158 stored cells were computed at 8 threads, so "
    "bit flags may fall back to the lab's 0.05 tolerance — recorded per "
    "gate (e153's 4-thread precedent reproduced e151 cells bit-exact).",
    "Bar-eligibility frozen: only the 11 twodoor surgery cells (+ the "
    "primary base row) can gate the bars; the root and jitter columns are "
    "controls (the root's intact base already exceeds the restore bar, so "
    "controls cannot be eligible by construction).",
    "The dispatch's 'joint: row restore + best head ablation' is "
    "operationalized as restore x ALL FOUR head cells — no data-picked "
    "'best' before compute; the bars decide.",
    "Head mean-replace vectors computed on the cell's (row-operated) state, "
    "not the intact net (e160 computed intact) — convention recorded.",
    "Cell dials skip g0 (dispatch lists g-12, g+12, site, D-183, CE_R); g0 "
    "is recorded in every net's base dial and gate.",
    "Root joint cells are degenerate (row-restore is identity on the root) "
    "and skipped in its control column; D-183 degenerate cells (row already "
    "zero) are recorded, not rerun.",
    "The jitter control's row__root surgery restores ROOT values into a net "
    "whose jittered training touched rows 175-197 — it is not a pure no-op; "
    "it removes whatever training wrote in 183:189 of a net that formed no "
    "graft (the specificity arm's exact meaning).",
    "Single lineage, single seed (10902) — n=1 until replicated.",
    "JITTER refs are e158 PASS 2 (correction commit 28ad653: jitter@183 "
    "g-12 0.789, GPU arms — the committed pass). An early read of runs/"
    "e158/metrics.json during THIS dispatch returned stale PASS-1 content "
    "(g-12 0.505); the pre-surgery gate caught it (checkpoint measures "
    "0.7885, not 0.5048) and the refs were re-extracted from git HEAD "
    "(disk == HEAD verified). The jitter control's door therefore sits at "
    "0.789 OPEN (T097's 'razor-thin 0.505' remark referenced pass 1; the "
    "pass-2 correction superseded it) — MORE ceiling room, same "
    "no-graft/no-restore prediction.",
    "Smoke mode trims: 2 nets, 3 cells, no D-183 / g+12 in cells; nothing "
    "adjudicated.",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: e153_phase_surgery.py VERBATIM (load_cpu/evl_load/battery_cell/
# battery_pz/ce_fixed_cpu/val_windows/deleted_wpe/read_fact_at); HeadReplace +
# head_mean_vec from e160_headset_escalation.py VERBATIM; row_surgery is
# e141's modified_wpe generalized to a row block. Copied (pinning differs).

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


def row_surgery(sd: dict, rows: tuple[int, ...],
                values: torch.Tensor) -> tuple[dict, dict]:
    """e141's modified_wpe generalized to a row block: wpe[rows] := values
    (one 192-dim vector per row); confinement gate = only those rows may
    change, every other tensor bit-identical. Identity when values already
    equal the rows (root-mode on the root itself)."""
    out = {k: v.clone() for k, v in sd.items()}
    for i, r in enumerate(rows):
        out["wpe.weight"][r] = values[i]
    d = out["wpe.weight"] != sd["wpe.weight"]
    changed_rows = sorted(set(int(r) for r in torch.nonzero(d)[:, 0].tolist()))
    n = int(d.sum().item())
    others = all(torch.equal(out[k], sd[k]) for k in out if k != "wpe.weight")
    confined = set(changed_rows) <= set(rows)
    gate = {"rows": list(rows), "n_elements_changed": n,
            "max_expected": len(rows) * sd["wpe.weight"].shape[1],
            "changed_rows": changed_rows,
            "identity": bool(n == 0),
            "confined": bool(confined),
            "others_bit_identical": bool(others),
            "pass": bool(confined and others)}
    return out, gate


@torch.no_grad()
def read_fact_at(net: TinyGPT, pool_x: torch.Tensor, name_ids, zid: int,
                 addr_row: int, xcol: int, bs=30) -> dict:
    """e131's read_fact_position VERBATIM ARITHMETIC, geometry parameterized:
    p(true name char) at positions addr_row..addr_row+6 over the pool windows."""
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


def head_mean_vec(net: TinyGPT, layer: int, head: int, bank_x, bs=30):
    """Mean of the head's 32-dim slice of c_proj's input over the CE_R bank
    (corpus windows only — no fact contexts). e133 organ_stats / e150 / e160
    convention."""
    outs = []
    hd = net.cfg.n_embd // net.cfg.n_head

    def pre(m, args):
        x = args[0].detach()
        outs.append(x[..., head * hd:(head + 1) * hd].clone())
        return None
    h = net.h[layer].attn.c_proj.register_forward_pre_hook(pre)
    with torch.no_grad():
        for i in range(0, bank_x.shape[0], bs):
            net(bank_x[i:i + bs])
    h.remove()
    o = torch.cat([t.reshape(-1, t.shape[-1]) for t in outs], 0)
    return o.mean(0)


class HeadReplace:
    """Replace head slices of c_proj's input with fixed vectors (mean-replace)
    or zeros (e133's zero arm), eval-only forward pre-hooks. e160 verbatim."""

    def __init__(self, net: TinyGPT, replace: dict):
        # replace: {(layer, head): vector-or-None}; None -> zero
        self.net = net
        self.replace = replace
        self.handles = []
        self.hd = net.cfg.n_embd // net.cfg.n_head

    def __enter__(self):
        by_layer: dict[int, list] = {}
        for (l, hd_), vec in self.replace.items():
            by_layer.setdefault(l, []).append((hd_, vec))
        for l, items in by_layer.items():
            def pre(m, args, items=items):
                x = args[0].clone()
                for hd_, vec in items:
                    sl = x[..., hd_ * self.hd:(hd_ + 1) * self.hd]
                    if vec is None:
                        sl.zero_()
                    else:
                        x[..., hd_ * self.hd:(hd_ + 1) * self.hd] = vec
                return (x,)
            self.handles.append(
                self.net.h[l].attn.c_proj.register_forward_pre_hook(pre))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []


def parse(hn: str) -> tuple[int, int]:
    return int(hn[1]), int(hn[3:])


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e166_smoke" if SMOKE else "e166")
    log(f"E166 THE INVERSE EVENT: graft-removal door-restore "
        f"(smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-only, torch threads {torch.get_num_threads()}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # ---------------- protocol rebuild (e151/e153 verbatim) ------------------
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)

    import random
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
    log(f"site battery: {tuple(pool_x.shape)} - ZEPHYRA at x-col {SITE_Z_XCOL}"
        f" (onset read row {SITE_ADDR_ROW}) in all windows")

    # batteries at the three geometries + CE_R bank
    bat_ids = {}
    for j in GEOS:
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    ids130 = bat_ids[0]
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, CE_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

    # ---------------- nets + pre-surgery gates -------------------------------
    sd = {}
    for tag in NETS:
        m = load_cpu(NETS[tag])
        sd[tag] = {k: v.clone() for k, v in m.state_dict().items()}
        del m
    log("nets loaded: twodoor (primary), root (restore source + ceiling), "
        "jitter (specificity)")

    gates_surg: dict = {}

    def site_read(net):
        return read_fact_at(net, pool_x, name_ids, zid, SITE_ADDR_ROW,
                            SITE_Z_XCOL)

    def base_dial(sd_in: dict, tag: str) -> dict:
        """Full base dial + the gate cells (e153's 10-cell convention)."""
        net = evl_load(sd_in)
        d: dict = {"tag": tag}
        d["base"] = {j: battery_cell(net, bat_ids[j], zid) for j in GEOS}
        d["ce_r"] = ce_fixed_cpu(net, *r_eval_xy)
        sr = site_read(net)
        d["site_onset"] = sr["pz_onset_mean"]
        d["site_span"] = sr["pname_mean_over7"]
        # A(129) census strength (e140/e151 convention: min of mean/zero arms)
        w = net.wpe.weight.data
        orig = w.clone()
        mean_row = orig.mean(0)
        base_pz = battery_pz(net, ids130, zid)
        w[129] = mean_row
        m129 = base_pz - battery_pz(net, ids130, zid)
        w.copy_(orig)
        w[129] = 0.0
        z129 = base_pz - battery_pz(net, ids130, zid)
        w.copy_(orig)
        assert torch.equal(w, orig), "A129 census failed to restore wpe"
        d["A129_mean_drop"] = float(m129)
        d["A129_zero_drop"] = float(z129)
        d["A129"] = float(min(m129, z129))
        # deletions
        for dl, rows in (("d_all", D_ALL), ("d_r0", (0,)), ("d183", (183,))):
            sd_d, gate = deleted_wpe(sd_in, rows)
            gates_surg[f"{tag}__{dl}"] = gate
            if not gate["pass"]:
                raise RuntimeError(f"deletion gate FAILED {tag}/{dl}: {gate}")
            net.load_state_dict(sd_d)
            d[dl] = {"g0": battery_cell(net, bat_ids[0], zid)["mean_pz"],
                     "ce_r": ce_fixed_cpu(net, *r_eval_xy)}
            if dl == "d183":
                s2 = site_read(net)
                d[dl]["site_onset"] = s2["pz_onset_mean"]
                d[dl]["site_span"] = s2["pname_mean_over7"]
        net.load_state_dict(sd_in)
        del net
        return d

    base = {}
    gates: dict = {"G_SPLICE": {"install_mix": mix, "pass": True}}
    for tag in ("twodoor", "root", "jitter"):
        base[tag] = base_dial(sd[tag], f"base_{tag}")
        b = base[tag]
        cells = {"g0": b["base"][0]["mean_pz"], "gm12": b["base"][-12]["mean_pz"],
                 "gp12": b["base"][12]["mean_pz"], "ce_r": b["ce_r"],
                 "site_onset": b["site_onset"], "site_span": b["site_span"],
                 "A129": b["A129"],
                 "d_all_g0": b["d_all"]["g0"], "d_r0_g0": b["d_r0"]["g0"],
                 "d183_site_onset": b["d183"]["site_onset"]}
        g = {"cells": cells, "refs": REFS[tag], "tol_bit": G_BIT_TOL,
             "fallback_tol": G_FALLBACK_TOL,
             "bit": bool(all(abs(cells[k] - REFS[tag][k]) < G_BIT_TOL
                             for k in cells)),
             "pass": bool(all(abs(cells[k] - REFS[tag][k]) < G_FALLBACK_TOL
                              for k in cells))}
        gates[f"G_{tag.upper()}"] = g
        log(f"GATE {tag}: " + " ".join(f"{k} {cells[k]:.4f}" for k in cells)
            + f" -> {'PASS' if g['pass'] else 'FAIL'}"
            f"{' (bit)' if g['bit'] else ''}")
        if not g["pass"]:
            raise RuntimeError(f"{tag} failed its gate vs stored cells")

    # ---------------- surgery grid -------------------------------------------
    log("=" * 78)
    n_dim = sd["root"]["wpe.weight"].shape[1]
    ROW_VALUES = {
        "root": sd["root"]["wpe.weight"][list(SITE_ROWS)].clone(),
        "zero": torch.zeros(len(SITE_ROWS), n_dim),
    }

    def row_values(net_tag: str, mode: str) -> torch.Tensor:
        if mode == "mean":                       # net-local census convention
            m = sd[net_tag]["wpe.weight"].mean(0)
            return m.unsqueeze(0).expand(len(SITE_ROWS), n_dim).clone()
        return ROW_VALUES[mode]

    if SMOKE:
        cell_grid = [("twodoor", "row", "root", None, None),
                     ("twodoor", "head", None, "n2", "zero"),
                     ("root", "row", "root", None, None)]
    else:
        cell_grid = []
        for net_tag in ("twodoor", "root", "jitter"):
            for mode in ROW_MODES:
                cell_grid.append((net_tag, "row", mode, None, None))
            for hname in HEAD_SETS:
                for hm in HEAD_MODES:
                    cell_grid.append((net_tag, "head", None, hname, hm))
            for hname in HEAD_SETS:              # joint = restore x all four
                for hm in HEAD_MODES:
                    if net_tag == "root":
                        continue                 # degenerate: restore=identity
                    cell_grid.append((net_tag, "joint", "root", hname, hm))

    cells_out: list[dict] = []

    def run_cell(net_tag: str, kind: str, row_mode, hname, hm) -> dict:
        sd_base = sd[net_tag]
        name = (f"{net_tag}:{kind}" +
                (f"__{row_mode}" if row_mode else "") +
                (f"__{hname}__{hm}" if hname else ""))
        if kind in ("row", "joint"):
            values = row_values(net_tag, row_mode)
            sd_c, gate = row_surgery(sd_base, SITE_ROWS, values)
        else:
            sd_c = {k: v.clone() for k, v in sd_base.items()}
            gate = {"rows": [], "identity": True, "confined": True,
                    "others_bit_identical": True, "pass": True,
                    "note": "no row surgery"}
        gates_surg[name] = gate
        if not gate["pass"]:
            raise RuntimeError(f"row-surgery gate FAILED {name}: {gate}")
        net = evl_load(sd_c)

        heads = list(HEAD_SETS[hname]) if hname else []
        rec: dict = {"net": net_tag, "kind": kind, "cell": name,
                     "row_mode": row_mode, "head_set": hname, "head_mode": hm,
                     "heads": heads, "gate": gate, "bar_eligible":
                         bool(net_tag == "twodoor")}

        if heads:
            if hm == "mean":
                mvs = {(l, h): head_mean_vec(net, l, h, r_eval_x)
                       for l, h in map(parse, heads)}
            else:
                mvs = {(l, h): None for l, h in map(parse, heads)}
            with HeadReplace(net, mvs):
                d_gm12 = battery_cell(net, bat_ids[-12], zid)
                d_sr = site_read(net)
                d_ce = ce_fixed_cpu(net, *r_eval_xy)
                if not SMOKE:
                    d_gp12 = battery_cell(net, bat_ids[12], zid)
            rec["gm12"] = d_gm12["mean_pz"]
            rec["site_onset"] = d_sr["pz_onset_mean"]
            rec["site_span"] = d_sr["pname_mean_over7"]
            rec["ce_r"] = d_ce
            if not SMOKE:
                rec["gp12"] = d_gp12["mean_pz"]
            rec["head_mean_vec_norms"] = {f"L{l}H{h}": float(v.norm())
                                          for (l, h), v in mvs.items()
                                          if v is not None}
        else:
            d_gm12 = battery_cell(net, bat_ids[-12], zid)
            d_sr = site_read(net)
            d_ce = ce_fixed_cpu(net, *r_eval_xy)
            if not SMOKE:
                d_gp12 = battery_cell(net, bat_ids[12], zid)
            rec["gm12"] = d_gm12["mean_pz"]
            rec["site_onset"] = d_sr["pz_onset_mean"]
            rec["site_span"] = d_sr["pname_mean_over7"]
            rec["ce_r"] = d_ce
            if not SMOKE:
                rec["gp12"] = d_gp12["mean_pz"]

        bt = base[net_tag]
        rec["delta_gm12"] = rec["gm12"] - bt["base"][-12]["mean_pz"]
        rec["delta_ce"] = rec["ce_r"] - bt["ce_r"]

        # no-op gate: root's row-root cell must be bit-equal to its base dial
        if net_tag == "root" and kind == "row" and row_mode == "root":
            rec_flat = [rec["gm12"], rec["ce_r"], rec["site_onset"],
                        rec["site_span"]]
            bt_flat = [bt["base"][-12]["mean_pz"], bt["ce_r"],
                       bt["site_onset"], bt["site_span"]]
            bit = gate["identity"] and all(a == b for a, b in
                                           zip(rec_flat, bt_flat))
            gates["G_NOOP_ROOT_RESTORE"] = {"identity": gate["identity"],
                                            "dial_bit_equal_base": bool(bit),
                                            "pass": bool(bit)}
            log(f"NOOP root row-restore: identity {gate['identity']}, dial "
                f"bit-equal {bit}")
            if not bit:
                raise RuntimeError("root row-restore no-op gate FAILED")

        # D-183 on top of the cell (degenerate where row 183 already zero)
        if not SMOKE:
            row183_zero = bool(not sd_c["wpe.weight"][183].any())
            if row183_zero:
                rec["d183"] = {"degenerate": True,
                               "note": "row 183 already zero in this cell",
                               "site_onset": rec["site_onset"],
                               "site_span": rec["site_span"],
                               "gm12": rec["gm12"]}
            else:
                sd_d, gate_d = deleted_wpe(sd_c, (183,))
                gates_surg[f"{name}__d183"] = gate_d
                if not gate_d["pass"]:
                    raise RuntimeError(f"D-183 gate FAILED {name}: {gate_d}")
                net.load_state_dict(sd_d)
                if heads:
                    with HeadReplace(net, mvs):
                        s3 = site_read(net)
                        g3 = battery_cell(net, bat_ids[-12], zid)["mean_pz"]
                else:
                    s3 = site_read(net)
                    g3 = battery_cell(net, bat_ids[-12], zid)["mean_pz"]
                rec["d183"] = {"degenerate": False,
                               "site_onset": s3["pz_onset_mean"],
                               "site_span": s3["pname_mean_over7"],
                               "gm12": g3}
        del net
        cells_out.append(rec)
        log(f"CELL {name}: g-12 {rec['gm12']:.4f} (base "
            f"{bt['base'][-12]['mean_pz']:.4f}, d {rec['delta_gm12']:+.4f}) "
            + (f"g+12 {rec['gp12']:.4f} " if not SMOKE else "")
            + f"site {rec['site_onset']:.4f} "
            + (f"D183 {rec['d183']['site_onset']:.4f} "
               if not SMOKE else "")
            + f"CE {rec['ce_r']:.4f} (dCE {rec['delta_ce']:+.4f})")
        return rec

    for spec in cell_grid:
        run_cell(*spec)

    # ---------------- adjudication (registered; no shopping) -----------------
    log("=" * 78)
    elig = [c for c in cells_out if c["bar_eligible"]]
    prim_base_gm12 = base["twodoor"]["base"][-12]["mean_pz"]
    all_vals = [prim_base_gm12] + [c["gm12"] for c in elig]
    restore_cells = [c for c in elig
                     if c["gm12"] >= RESTORE_BAR
                     and c["delta_ce"] <= RESTORE_CE_BAR]
    wreck_cells = [c for c in elig
                   if c["gm12"] >= RESTORE_BAR
                   and c["delta_ce"] > RESTORE_CE_BAR]
    stays_shut = bool(max(all_vals) <= SHUT_BAR)
    best = max(elig, key=lambda c: c["gm12"]) if elig else None

    jit_row_root = next((c for c in cells_out if c["net"] == "jitter"
                         and c["kind"] == "row" and c["row_mode"] == "root"),
                        None)
    spec_triggered = False
    if restore_cells and jit_row_root is not None:
        best_gain = max(c["delta_gm12"] for c in restore_cells)
        jit_gain = jit_row_root["delta_gm12"]
        spec_triggered = bool(jit_gain > max(0.5 * best_gain, SPEC_GAIN_FLOOR))

    if restore_cells and not spec_triggered:
        verdict = "DOOR-RESTORES"
        clause = (f"graft removal restored the geometry door: "
                  f"{[c['cell'] for c in restore_cells]} reached g-12 >= "
                  f"{RESTORE_BAR} at dCE <= +{RESTORE_CE_BAR} (best "
                  f"{best['cell']} g-12 {best['gm12']:.4f}, CE "
                  f"{best['delta_ce']:+.4f}) — the closure is ACTIVE "
                  f"competitive inhibition by the graft; build-and-travel "
                  f"is CAUSAL (T097's one-event reading confirmed by "
                  f"intervention).")
    elif restore_cells and spec_triggered:
        verdict = "TEXTURE"
        clause = (f"primary cells crossed the restore bar "
                  f"({[c['cell'] for c in restore_cells]}) BUT the jitter "
                  f"control (no graft) gained {jit_row_root['delta_gm12']:+.4f}"
                  f" from the SAME row surgery (> max(0.5 x primary best "
                  f"gain, {SPEC_GAIN_FLOOR})) — the move is a generic "
                  f"row-content effect, not graft removal; pre-registered "
                  f"specificity guard downgrades to TEXTURE.")
    elif stays_shut:
        verdict = "DOOR-STAYS-SHUT"
        clause = (f"every bar-eligible cell (and the base) stayed <= "
                  f"{SHUT_BAR} (max {max(all_vals):.4f}, {best['cell']}) — "
                  f"removing the graft does NOT restore the door: the "
                  f"closure was a destructive rewrite, not active "
                  f"inhibition; 'one event' downgrades to same-step "
                  f"correlation; third great asymmetry: doors open by "
                  f"training, are killed by surgery, and cannot be "
                  f"REOPENED by surgery (T037's write-once core extended "
                  f"to re-opening).")
    elif max(all_vals) > SHUT_BAR:
        verdict = "PARTIAL"
        wtxt = (f"; wreck-flagged: {[c['cell'] for c in wreck_cells]} "
                f"reached the bar only at CE > +{RESTORE_CE_BAR}"
                if wreck_cells else "")
        clause = (f"the best cell moved the door into the 0.27-0.5 band "
                  f"({best['cell']} g-12 {best['gm12']:.4f}, CE "
                  f"{best['delta_ce']:+.4f}) without clearing the restore "
                  f"bar — partial re-opening{wtxt}; numbers reported, no "
                  f"bar shopping.")
    else:
        verdict = "TEXTURE"
        clause = (f"no registered bar fired cleanly (max eligible g-12 "
                  f"{max(all_vals):.4f}); numbers reported.")
    adjudication = {
        "bars": {"restore_gm12": RESTORE_BAR, "restore_dce": RESTORE_CE_BAR,
                 "shut_all_cells": SHUT_BAR},
        "eligible_cells": [c["cell"] for c in elig],
        "restore_cells": [c["cell"] for c in restore_cells],
        "wreck_cells": [c["cell"] for c in wreck_cells],
        "best_cell": best["cell"] if best else None,
        "best_gm12": best["gm12"] if best else None,
        "max_all_vals": max(all_vals),
        "specificity_guard": {
            "jitter_row_root_delta_gm12": (jit_row_root["delta_gm12"]
                                           if jit_row_root else None),
            "triggered": spec_triggered},
        "verdict": verdict, "clause": clause,
    }
    log(f"E166 VERDICT: {verdict}")
    log(f"  eligible g-12 range: {min(all_vals):.4f} .. {max(all_vals):.4f} "
        f"(bars: restore >= {RESTORE_BAR} @ dCE <= +{RESTORE_CE_BAR}; "
        f"shut all <= {SHUT_BAR})")
    jit_txt = (f"{jit_row_root['delta_gm12']:+.4f}" if jit_row_root
               else "absent (smoke)")
    log(f"  best cell: {best['cell']} g-12 {best['gm12']:.4f} dCE "
        f"{best['delta_ce']:+.4f} | jitter row-root d {jit_txt} "
        f"(guard {spec_triggered})")
    log(f"  {clause}")

    # ---------------- outputs -------------------------------------------------
    metrics = {
        "experiment": "e166_inverse_event",
        "date": common.now_iso(),
        "registration": ("T097 build-and-travel unification; QUEUE.md row "
                         "e166 dispatch ~13:00Z. Registered prediction "
                         "verbatim below; operationalizations frozen in the "
                         "module docstring before compute."),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("does REMOVING the 183-site graft (wpe[183:189] := "
                     "root values; door-mover head ablation; joint) RESTORE "
                     "the geometry door e151's locked re-teach shut "
                     "(g-12 0.916 -> 0.102)?"),
        "nets": {k: str(NETS[k].relative_to(E43.REPO)).replace("\\", "/")
                 for k in NETS},
        "compute": {"device": "cpu", "torch_threads": torch.get_num_threads(),
                    "eval_only": True,
                    "note": "GPU user-occupied; e164 + e154 share the CPU — "
                            "4 threads, single staggered launch, no "
                            "busy-waiting"},
        "gates": gates,
        "gates_surgery": gates_surg,
        "head_sets": {"reader3": list(READER3), "n2": list(N2SET),
                      "provenance": "reader3 = e153 reader order top-3 "
                                    "(twodoor census site-drop: L3H5, L0H3, "
                                    "L1H0; the best transplant set, g-12 "
                                    "0.1505); n2 = e160 top2-no-L0H3 "
                                    "(L1H0, L0H0)"},
        "bars": {"restore_gm12": RESTORE_BAR, "restore_dce": RESTORE_CE_BAR,
                 "shut_all_cells": SHUT_BAR,
                 "shut_derivation": "0.30 x root base g-12 (0.9156) — "
                                    "e153's close-bar convention"},
        "base_dial": base,
        "cells": cells_out,
        "adjudication": adjudication,
        "honesty_reflex": {
            "graft_is_not_only_wpe": "e153's wiring diff put 66.5% of the "
                "conversion's delta energy in MLPs (late-layer-skewed) with "
                "LN changes around it; removing the wpe part of the graft "
                "leaves the re-taught MLP/LN stream intact, so DOOR-STAYS-"
                "SHUT cannot distinguish 'the door was destructively "
                "rewritten' from 'the inhibition is carried outside wpe "
                "rows 183:189'. The head-ablation and joint cells probe the "
                "circuit part; nothing here can ablate MLP/LN state. A "
                "follow-up re-teach-the-restore cell (gradient step, not "
                "surgery) would separate the two readings.",
            "off_manifold_values": "zero/mean row modes put positions "
                "183-189 off any trained manifold; head mean-replace is "
                "off-manifold in head space (e150 caveat) — the CE column "
                "prices both; bars are raw g-12 with the CE clause only on "
                "RESTORES.",
            "row_head_interaction": "joint cells are the interaction probe; "
                "the composition is nonlinear (e153's additivity note), so "
                "row+head may not add; a joint cell BELOW both singles is "
                "recorded as interaction texture, not adjudicated.",
            "edit_law_asymmetry": "T091: transplants (addition) moved the "
                "shut door only to ~1/3 of bar; ablation (subtraction) is "
                "the edit direction the lab's law says works — the head "
                "cells test exactly that direction on the door-movers.",
            "controls": "root column: row-restore is a bit no-op (gate "
                "G_NOOP_ROOT_RESTORE); its zero/mean rows and head cells "
                "show the knife's effect on an OPEN door (e160's N2 kill "
                "expected). jitter column: no graft formed; row-restore "
                "there also removes jitter-drift on 183:189 (not a pure "
                "no-op) — the specificity arm's reading is the DELTA, "
                "guarded by the pre-registered clause.",
            "single_lineage": "one primary net (one 300-step conversion, "
                "seed 10902, one root seed) — n=1 until replicated.",
            "readout_ceiling": "g-12 cells are bounded by the net's own "
                "scale; the root ceiling control bounds the restore's "
                "maximum (an open door cannot open further).",
        },
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": 2739072, "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "inverse_event.png", base, cells_out, adjudication)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'inverse_event.png'}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot

def plot(path, base, cells, adj):
    """Legible: (top) the bar panel + row-mode trio + site/D-183;
    (bottom) controls + CE-vs-door scatter + verdict box."""
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.5))
    prim = [c for c in cells if c["net"] == "twodoor"]

    # (0,0) THE BAR PANEL: primary cells' g-12 vs the registered bars
    ax = axes[0, 0]
    names = ["base"] + [c["cell"].split(":", 1)[1] for c in prim]
    vals = [base["twodoor"]["base"][-12]["mean_pz"]] + [c["gm12"] for c in prim]
    ces = [0.0] + [c["delta_ce"] for c in prim]
    cols = ["steelblue"] + [("seagreen" if v >= 0.5 and ce <= 0.1 else
                             "crimson" if v >= 0.5 else
                             "darkorange" if v > 0.27 else "lightgray")
                            for v, ce in zip(vals[1:], ces[1:])]
    xs = np.arange(len(names))
    ax.bar(xs, vals, 0.62, color=cols, edgecolor="k", lw=0.5)
    for x, v, ce in zip(xs, vals, ces):
        ax.text(x, v + 0.012, f"{v:.3f}\nCE{ce:+.2f}", ha="center",
                fontsize=6.2)
    ax.axhline(0.5, color="seagreen", ls="--", lw=1.4)
    ax.axhline(0.27, color="gray", ls="--", lw=1.2)
    ax.text(len(names) - 0.4, 0.515, "RESTORE bar 0.50 @ dCE<=+0.1",
            fontsize=7, color="seagreen", ha="right")
    ax.text(len(names) - 0.4, 0.285, "SHUT bar 0.27", fontsize=7,
            color="gray", ha="right")
    ax.axhline(base["root"]["base"][-12]["mean_pz"], color="steelblue",
               ls=":", lw=1.0)
    ax.text(0.0, base["root"]["base"][-12]["mean_pz"] + 0.012,
            f"root (open-door) base {base['root']['base'][-12]['mean_pz']:.3f}",
            fontsize=6.5, color="steelblue")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=42, ha="right", fontsize=6.4)
    ax.set_ylabel("battery p(Z) g-12 install-60")
    ax.set_ylim(0, 1.08)
    ax.set_title("(a-c) THE INVERSE EVENT — graft-removal cells on "
                 "e151_twodoor\n(green: >=0.5@CE ok | red: >=0.5@wreck CE | "
                 "orange: 0.27-0.5 | gray: shut)", fontsize=9)

    # (0,1) row-mode trio across nets (primary vs controls)
    ax = axes[0, 1]
    modes = ("intact", "root", "zero", "mean")
    for k, net_tag in enumerate(("twodoor", "root", "jitter")):
        vals2 = [base[net_tag]["base"][-12]["mean_pz"]]
        for m in ("root", "zero", "mean"):
            c = next((x for x in cells if x["net"] == net_tag
                      and x["kind"] == "row" and x["row_mode"] == m), None)
            vals2.append(c["gm12"] if c else np.nan)
        xs2 = np.arange(len(modes)) + (k - 1) * 0.26
        ax.bar(xs2, vals2, 0.24,
               color=["steelblue", "crimson", "darkorange", "dimgray"][k],
               edgecolor="k", lw=0.4,
               label=f"{net_tag} (base g-12 "
                     f"{base[net_tag]['base'][-12]['mean_pz']:.3f})")
        for x, v in zip(xs2, vals2):
            if not np.isnan(v):
                ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=6.2)
    ax.axhline(0.5, color="seagreen", ls="--", lw=1.2)
    ax.axhline(0.27, color="gray", ls="--", lw=1.0)
    ax.set_xticks(np.arange(len(modes)))
    ax.set_xticklabels(["intact", "restore\n:=root", "zero", "mean"],
                       fontsize=8)
    ax.set_ylabel("battery p(Z) g-12")
    ax.set_ylim(0, 1.08)
    ax.set_title("(a)+(d) ROW SURGERY across nets — primary / root ceiling / "
                 "jitter specificity", fontsize=9)
    ax.legend(fontsize=6.6, loc="upper right")

    # (0,2) site expression at 183 + D-183 (primary cells)
    ax = axes[0, 2]
    names3 = ["base"] + [c["cell"].split(":", 1)[1] for c in prim]
    onset = [base["twodoor"]["site_onset"]] + [c["site_onset"] for c in prim]
    d183 = [base["twodoor"]["d183"]["site_onset"]] + [
        c.get("d183", {}).get("site_onset", np.nan) for c in prim]
    xs3 = np.arange(len(names3))
    ax.bar(xs3 - 0.19, onset, 0.36, color="crimson", edgecolor="k", lw=0.4,
           label="site onset p(Z) @183")
    ax.bar(xs3 + 0.19, d183, 0.36, color="lightgray", edgecolor="k", lw=0.4,
           label="after D-183")
    for x, v in zip(xs3 - 0.19, onset):
        ax.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=5.8)
    for x, v in zip(xs3 + 0.19, d183):
        if not np.isnan(v):
            ax.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=5.8)
    ax.set_xticks(xs3)
    ax.set_xticklabels(names3, rotation=42, ha="right", fontsize=6.4)
    ax.set_ylim(0, 1.12)
    ax.set_title("(a-c) GRAFT SIDE — site expression at 183 with/without "
                 "row 183\ndoes the graft die under the surgery that tested "
                 "the door?", fontsize=9)
    ax.legend(fontsize=7)

    # (1,0) control columns: head + joint cells on root and jitter
    ax = axes[1, 0]
    rows = []
    for net_tag in ("root", "jitter"):
        cc = [x for x in cells if x["net"] == net_tag and x["kind"] != "row"]
        rows.append((f"{net_tag} base", base[net_tag]["base"][-12]["mean_pz"]))
        rows += [(x["cell"].split(":", 1)[1], x["gm12"]) for x in cc]
    ys = np.arange(len(rows))
    ax.barh(ys, [r[1] for r in rows], 0.62,
            color=["dimgray" if "base" in r[0] else
                   "steelblue" if r[0].startswith(("root", "jitter")) and
                   "joint" in r[0] else "darkorange"
                   for r in rows], edgecolor="k", lw=0.4)
    for y, r in zip(ys, rows):
        ax.text(max(r[1], 0) + 0.015, y, f"{r[1]:.3f}", va="center",
                fontsize=6.4)
    ax.axvline(0.5, color="seagreen", ls="--", lw=1.2)
    ax.axvline(0.27, color="gray", ls="--", lw=1.0)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("battery p(Z) g-12")
    ax.set_title("(d) CONTROLS — head/joint cells on root (knife on an open "
                 "door)\nand jitter (no graft; specificity)", fontsize=9)

    # (1,1) CE-priced scatter: dCE vs g-12 for primary cells
    ax = axes[1, 1]
    for c in prim:
        col = ("seagreen" if c["gm12"] >= 0.5 and c["delta_ce"] <= 0.1 else
               "crimson" if c["gm12"] >= 0.5 else
               "darkorange" if c["gm12"] > 0.27 else "lightgray")
        ax.scatter(c["delta_ce"], c["gm12"], s=90, color=col,
                   edgecolor="k", lw=0.6, zorder=3)
        ax.annotate(c["cell"].split(":", 1)[1], (c["delta_ce"], c["gm12"]),
                    textcoords="offset points", xytext=(6, -3), fontsize=5.8)
    ax.scatter(0.0, base["twodoor"]["base"][-12]["mean_pz"], s=110,
               marker="*", color="steelblue", edgecolor="k", lw=0.6,
               zorder=3)
    ax.annotate("twodoor base", (0.0, base["twodoor"]["base"][-12]["mean_pz"]),
                textcoords="offset points", xytext=(6, 4), fontsize=6.5)
    ax.axhline(0.5, color="seagreen", ls="--", lw=1.2)
    ax.axhline(0.27, color="gray", ls="--", lw=1.0)
    ax.axvline(0.1, color="seagreen", ls=":", lw=1.2)
    ax.set_xlabel("dCE_R vs intact twodoor (flat-CE discipline)")
    ax.set_ylabel("g-12 p(Z)")
    ax.set_title("CE-PRICED DOOR MOVES — restore quadrant = upper-left\n"
                 "(g-12 >= 0.5 at dCE <= +0.1)", fontsize=9)

    # (1,2) verdict panel
    ax = axes[1, 2]
    ax.axis("off")
    jit = adj["specificity_guard"]["jitter_row_root_delta_gm12"]
    vlines = [
        "REGISTERED (QUEUE e166 verbatim):",
        "  DOOR-RESTORES: any cell g-12 >= 0.5 at CE <= +0.1",
        "  DOOR-STAYS-SHUT: all <= 0.27",
        "  PARTIAL: 0.27-0.5   |   texture => TEXTURE with numbers",
        "",
        "DECIDING NUMBERS (primary e151_twodoor):",
        f"  base g-12: {base['twodoor']['base'][-12]['mean_pz']:.4f} "
        f"(root open-door base "
        f"{base['root']['base'][-12]['mean_pz']:.4f})",
        f"  best eligible cell: {adj['best_cell']} g-12 "
        f"{adj['best_gm12']:.4f}",
        f"  eligible range: {adj['max_all_vals']:.4f} max",
        f"  restore cells: {adj['restore_cells'] or 'none'}"
        f"  wreck cells: {adj['wreck_cells'] or 'none'}",
        f"  jitter row-restore delta: "
        f"{jit if jit is None else round(jit, 4)} (guard "
        f"{adj['specificity_guard']['triggered']})",
        "",
        f"VERDICT: {adj['verdict']}",
    ] + [f"  {wd}" for wd in
         [adj["clause"][i:i + 64] for i in range(0, len(adj["clause"]), 64)]]
    for i, tx in enumerate(vlines):
        ax.text(0.02, 0.97 - i * 0.045, tx, fontsize=7.0, va="top",
                family="monospace")

    fig.suptitle(f"E166 — THE INVERSE EVENT: graft-removal door-restore on "
                 f"e151_twodoor (g-12 0.916 -> 0.102) -> {adj['verdict']}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
