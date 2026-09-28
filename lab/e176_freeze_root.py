"""E176 — FREEZE THE ROOT: the decisive control e161/T101 demands.

WHY (T101 + the E161 verdict): e161 froze the DWELL-PEAK memory
(e152_steps32) on plain corpus and it DISSOLVED completely in <50 steps
— the door, all geometries, the site, the brake, the sink — at healthy
CE. The honest reframing: e151's "conversion" was substantially
FORGETTING — an unconsolidated memory washed out by ordinary gradient
flow while a new one trained in. Memories lie on a GRADIENT-RESISTANCE
axis; e161 filled the washable end (fresh installs, dwell-phase). THE
OPEN CELL this run decides: does the FULLY-CONSOLIDATED memory survive
the same plain-corpus freeze? SURVIVES => consolidation IS
resistance-acquisition (CLS licensed, claim 2 rewrites to the
resistance axis). DISSOLVES => even "consolidated" is use-it-or-lose-it
(the anchor half of every past fine-tune was quietly rehearsing the
fact — every 'experiment' was also a life-support).

ROOT (mandated): runs/checkpoints/e131_consolidated_e113.pt — the
FULLY-CONSOLIDATED root (e065 install -> e109/e113 consolidation
lineage), i.e. the exact pre-e151 state. Gated below vs e151's stored
before-cells (runs/e151/metrics.json 'before' battery) BEFORE any
training compute is trusted.

FREEZE (the design): ONE 300-step plain-corpus fine-tune — e161's
protocol VERBATIM: batch 32 = 16 anchor-bank draws (256-token windows
at the install occurrences' left context; e152's anchor bank, name-free
by construction and re-verified at draw time) + 16 random corpus
windows from train_ids (raw corpus contains ZERO 'ZEPH' substrings,
grep-verified; every drawn window re-verified at draw time). NO fact
windows, NO name tokens, NO mask. Full-token CE (every position of
every window). Optimizer VERBATIM the locked lineage (e109 arm-b /
e119-L / e143 / e151 / e152 / e161): AdamW (0.9, 0.95) wd 0.1,
constant lr 1e-3, clip 1.0; seed 10902 (the locked fine-tune seed
lineage — the SAME draw sequence as e161, so the two trajectories
differ only in their starting nets); snapshots at freeze steps {50,
100, 200, 300} (continuation steps from the root).

MEASURE PER CHECKPOINT (+ step 0 = the root itself, the 'before'):
g-12 AND g+12 AND g0 (the geometry doors, install-60 batteries, ctx =
train_text[p-PRE-j:p], e119 construction) + the held30 counterparts
(the dispatch's generalization dial; e152's held30 construction);
the 183-site span census (co-report — the consolidated root has NO
183 site: 0.00027 < 2x control; kept because the registered texture
clause names 'site survives') + the functional site read; the 129-band
content census (ROWS_OLD, g0 readout) with A(129) (the brake) and
row-0 strength (the consolidated fact's sink); D-all g0; CE_R
(wreckage guard); d183 co-report.

REGISTERED PREDICTION (verbatim from QUEUE e176 / the dispatch;
adjudicate against exactly this; no bar shopping):
  - CONSOLIDATION-IS-RESISTANCE fires if: the fact survives (g-12 >= 0.5
    AND g0 >= 0.5 at +300) — CLS licensed; the resistance axis is real;
    claim 2 rewrites to it.
  - USE-IT-OR-LOSE-IT fires if: the fact dissolves (matching e161's
    trajectory shape) — even consolidated memories need the anchor
    rehearsal.
  - No bar shopping; texture (partial: e.g., site survives but geometry
    dies) => TEXTURE with the full trajectory.

OPERATIONALIZATIONS (frozen here before compute — the registered
clauses name survives/dissolves at endpoint values; these fix them,
they do not move the bars):
  * g-12 / g0 = ABSOLUTE install-60 battery mean p(Z) at ctx offsets
    -12 / 0 (e158's convention). Freeze steps are CONTINUATION steps
    from the consolidated root: {0, 50, 100, 200, 300}.
  * SURVIVES (primary) = g-12(300) >= 0.50 AND g0(300) >= 0.50 — the
    registered clause's own two dials, at the registered endpoint.
  * DISSOLVES (primary) = g-12(300) <= 0.27 AND g0(300) <= 0.27 — both
    fact dials under the SHUT bar (e158/e161's convention).
    'Matching e161's trajectory shape' is CO-REPORTED, not a bar: the
    earliest checkpoint <= 0.27 per dial, the +50 retention per dial,
    and the across-anatomy dissolution pattern (gp12, held30, A129,
    row-0, D-all) — e161's shape had every dial under bar by +50.
  * A dial ending in the gap band (0.27, 0.50) fails both primaries =>
    TEXTURE with the full trajectory (a partial fact — one dial
    survived, one dissolved — is exactly the registered 'texture'
    case; so is a surviving site/sink with dead geometries).
  * Adjudication order: CONSOLIDATION-IS-RESISTANCE -> USE-IT-OR-LOSE-
    IT -> TEXTURE; every sub-boolean reported regardless.
  * The 183 site census / held30 / brake / sink are CO-DIALS (texture
    instruments), not bar clauses — the consolidated fact's site is
    row 0 (strength 0.7317 at the root), reported as its own dial.

COMPUTE ENVELOPE: CPU-ONLY by dispatch (CUDA_VISIBLE_DEVICES=-1;
e152R owns the GPU, e170/e173 share the CPU): torch threads 4 (LOW
<= 4), one 25 s launch stagger (single sleep, no busy-waiting
anywhere), cooldown(60) before and after the ONE training, CPU
training time cap 1800 s. The dispatch's '<=180 s cap' is the GPU
single-training cap; the CPU path follows e161's precedent verbatim
(e161 actual: 1625.7 s training / 2771.9 s total at 4 threads).

INSTRUMENT PROVENANCE: every instrument is lab/e161_freeze_cell.py
VERBATIM (itself the e152/e151/e143/e131/e119/e113/e068/e065/e043
lineage — load_cpu / evl_load / battery_cell / battery_pz /
ce_fixed_cpu / val_windows / deleted_wpe / read_fact_at / row_census;
copied, not imported, to own the device policy — e161's rig already
forces CUDA_VISIBLE_DEVICES=-1; this one re-forces it). finetune_freeze
is e161's verbatim (anchors + random corpus only, full-token CE,
freeze-step snapshots; the light in-run eval gains a g0 co-report that
consumes no RNG — the training draw sequence is bit-identical to
e161's). Protocol rebuild: corpus seed 1337, SPLICE_RNG 24301 host
shuffle, install60/held30 split, mix gate — e143/e151/e152/e161
verbatim. The e161 trajectory references (E161_REF) are EMBEDDED
VERBATIM from runs/e161/metrics.json trace_summary and re-verified
against that file at plot time.

Outputs: runs/e176/{metrics.json, freeze_root.png}; checkpoints
runs/checkpoints/e176_root_freeze.pt (root+300 endpoint) +
e176_root_freeze_s{50,100,200}.pt (intermediates). No NOTES/THINKING/
QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python e176_freeze_root.py    (E176_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e152R owns
# the GPU; e170/e173 share the CPU — threads capped at 4 below)

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

SMOKE = os.environ.get("E176_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e176 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
ROOT_CK = "e131_consolidated_e113.pt"   # THE FULLY-CONSOLIDATED ROOT
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E161_METRICS = E43.REPO / "runs" / "e161" / "metrics.json"

# ---- e152's placement constants (the measurement instruments rebuild these) ----
RETEACH_J = 54                    # offset: name x-cols 184..190, read rows 183..189
SITE_ADDR_ROW = 183               # the onset read row (= e120's SPLICE_ADDR_ROW)
SITE_Z_XCOL = PRE + RETEACH_J     # 184 (= e120's Z_XCOL)
SITE_CONT = BLOCK - PRE - RETEACH_J - len(NAME)     # 65-token continuation
SITE_ROWS = tuple(range(183, 190))                  # the 7 trained read rows
SHARED_CTR = (60, 100, 150, 160, 170, 200, 220)     # outside every trained band
D_ALL = (121, 125, 129, 133, 137)                   # e113's fixed set
GEOS = (-12, 0, 12)               # novel x2 + the trained g0

# ---- census row sets (e158/e161 battery convention; read-visible rows only) ----
ROWS_183 = (0, 1, 2) + (181, 182) + SITE_ROWS + (60, 100, 150, 160, 170)
ROWS_OLD = (0, 1, 2, 3, 4, 5, 6, 60, 100, 118, 119, 120) + tuple(range(121, 130))
if SMOKE:
    ROWS_183 = (0, 1, 182) + (183, 185, 189) + (60, 100)
    ROWS_OLD = (0, 1, 60, 121, 125, 129)

# ---- the freeze: continuation steps from the consolidated root -----------------
CKPT_STEPS: tuple[int, ...] = (50, 100, 200, 300) if not SMOKE else (2, 4)
CKPT_SET = set(CKPT_STEPS)

# ---- fine-tune envelope (e109 arm-b / e119-L / e143 / e151 / e152 / e161 verbatim)
FT_LR = 1e-3
FT_STEPS = CKPT_STEPS[-1]
FT_TIME_CAP = 1800.0              # CPU cap (e161 precedent: 1625.7 s actual)
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
FREEZE_SEED = 10902               # the locked fine-tune seed lineage (= e161's)
COOLDOWN_S = 60.0                 # around the ONE training (CPU thermal)
STAGGER_S = 25.0                  # launch stagger vs e170/e173 CPU bursts

# ---- gates / references (full precision, = e151's stored before-cells) ----------

R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05
E151_ROOT = {                     # runs/e151/metrics.json 'before' battery
    "base_gm12": 0.9155886769294739,
    "base_g0": 0.7850371599197388,
    "base_gp12": 0.9478210210800171,
    "ce_r": 1.663516640663147,
    "site_span_strength": 0.000274658203125,
    "site_read_onset": 0.8898659348487854,
    "site_read_span": 0.982668936252594,
    "A129": -0.13237020391970877,
    "row0_strength": 0.7316772222270098,
    "dall_g0": 0.9047248959541321,
}

# ---- e161's stored trajectory (VERBATIM runs/e161/metrics.json trace_summary;
# re-verified against the file at plot time — the overlay references) ------------
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
                "0.0036 (+300); root = e152_steps32, the dwell peak)")

# ---- registered bar constants (frozen; see OPERATIONALIZATIONS above) ----------
SITE_CTRL_MULT = 2.0              # e139/e151/e161 site convention: >= 2x control-max
SURVIVE_BAR = 0.50                # CONSOLIDATION-IS-RESISTANCE clause (both dials)
SHUT_BAR = 0.27                   # USE-IT-OR-LOSE-IT clause (both dials; e158/e161)
ROOT_GM12 = E151_ROOT["base_gm12"]                # 0.9155886769294739
ROOT_G0 = E151_ROOT["base_g0"]                    # 0.7850371599197388
ROOT_ROW0 = E151_ROOT["row0_strength"]            # 0.7316772222270098

REGISTERED_PREDICTION = {
    "consolidation_is_resistance": "CONSOLIDATION-IS-RESISTANCE fires if: "
        "the fact survives (g-12 >= 0.5 AND g0 >= 0.5 at +300) — CLS "
        "licensed; the resistance axis is real; claim 2 rewrites to it.",
    "use_it_or_lose_it": "USE-IT-OR-LOSE-IT fires if: the fact dissolves "
        "(matching e161's trajectory shape) — even consolidated memories "
        "need the anchor rehearsal.",
    "no_bar_shopping": "No bar shopping; texture (partial: e.g., site "
        "survives but geometry dies) => TEXTURE with the full trajectory.",
    "operationalizations": "g-12 / g0 = absolute install-60 battery mean "
        "p(Z) at ctx offsets -12 / 0 per checkpoint; freeze steps are "
        f"CONTINUATION steps from the consolidated root {{0, {list(CKPT_STEPS)}}}; "
        f"SURVIVES primary = g-12(300) >= {SURVIVE_BAR} AND g0(300) >= "
        f"{SURVIVE_BAR}; DISSOLVES primary = g-12(300) <= {SHUT_BAR} AND "
        f"g0(300) <= {SHUT_BAR} (the e158/e161 SHUT convention; "
        "'matching e161's trajectory shape' CO-REPORTED: earliest-<= bar "
        "per dial, +50 retention per dial, across-anatomy pattern — not a "
        "bar); a dial in the gap band (0.27, 0.50) fails both primaries => "
        "TEXTURE; order CONSOLIDATION-IS-RESISTANCE -> USE-IT-OR-LOSE-IT "
        "-> TEXTURE; every sub-boolean reported regardless.",
    "committed": "QUEUE e176 registered no committed branch (the two-bar "
                 "fork T101 frames as the decisive control).",
}

trims: list[str] = []
deviations: list[str] = [
    "CPU-ONLY by dispatch (CUDA_VISIBLE_DEVICES=-1 forced before torch; "
    "e152R owns the GPU, e170/e173 share the CPU): torch threads 4, one "
    "25 s launch stagger (single sleep, no busy-waiting), cooldown(60) "
    "before/after the ONE training.",
    "The dispatch's '<=180 s cap' is the lab's GPU single-training cap; "
    "the CPU path runs under a 1800 s training cap (e161 precedent "
    "verbatim: 1625.7 s actual / 300 steps at 4 threads; 2771.9 s total).",
    "Nets are the 2.7M e131_consolidated line (the dispatch's '<=1M "
    "family' note is an envelope statement; e143/e151/e152/e161 precedent "
    "— the consolidated root, every gate reference, and every lineage "
    "number of this cell lives on the 2.7M line).",
    "held30 batteries ADDED vs e161's battery (the dispatch's dial list "
    "names held30): the same e119 ctx-battery at the held-out 30 host "
    "occurrences (e152's held30 construction); a CO-DIAL, not a bar "
    "clause.",
    "The 183-span site census kept from e161 verbatim as a CO-REPORT "
    "even though the dispatch's dial list omits it: the registered "
    "texture clause names 'site survives', and this is the instrument "
    "that would show it (the consolidated root has NO 183 site: 0.00027 "
    "< 2x control — its fact-site is row 0, strength 0.732).",
    "The light in-run checkpoint eval gains a g0 co-report vs e161's "
    "g-12-only light eval (consumes no RNG; the training draw sequence "
    "is bit-identical to e161's — same seed, same shapes).",
    "Name-free guarantee implemented as a VERIFY, not a redraw (e161 "
    "verbatim): the raw corpus contains zero 'ZEPH' substrings (grep "
    "data/input.txt = 0 hits), and every drawn window is decoded and "
    "checked at draw time (violation would hard-fail, consuming no "
    "hidden RNG).",
    "Eval thread count is 4 (dispatch) vs e151's stored cells — CPU "
    "reduction order can drift low-order bits; the G_ROOT gate therefore "
    "reports both the 5e-6 bit flag and the 0.05 fallback tolerance "
    "(e161/e152/e158 precedent).",
    "Single seed (10902), single lineage, ONE trajectory — the freeze "
    "outcome is a point estimate (n=1 path through continuation-step "
    "space), and the root is ONE consolidated lineage (e065 install -> "
    "e109/e113 consolidation), never plain-corpus-frozen before.",
    "Smoke mode trims: 4-step freeze with checkpoints at {2,4}, reduced "
    "census rows, nothing adjudicated.",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/e161_freeze_cell.py VERBATIM (see the module docstring).
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
    """e131's read_fact_position VERBATIM ARITHMETIC (e151/e152/e161 copy):
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
                    train_ids: torch.Tensor, itos, r_eval_xy, gm12_ids,
                    g0_ids, zid: int, seed: int):
    """THE PLAIN-CORPUS FREEZE (e161's finetune_freeze VERBATIM; the light
    eval gains a no-RNG g0 co-report). Per step: aj = randint(16) anchor
    draws, rj = randint(16) random corpus offsets; batch 32 full-token CE;
    AdamW (0.9,0.95) wd 0.1 lr 1e-3 constant, clip 1.0. Snapshots
    (deep-copy out; nothing loaded into the training net) + light CPU
    evals at the freeze steps (no RNG consumed — draw sequence identical
    to e161's at the same seed)."""
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
            gz = battery_cell(evl, gm12_ids, zid)
            gz0 = battery_cell(evl, g0_ids, zid)
            ce_r = ce_fixed_cpu(evl, *r_eval_xy)
            traj.append({"step": step, "g_m12_mean_pz": gz["mean_pz"],
                         "g0_mean_pz": gz0["mean_pz"],
                         "frac_argmax_z": gz["frac_argmax_z"], "ce_r": ce_r,
                         "elapsed_s": round(time.time() - t_start, 1)})
            log(f"  [{tag}] CKPT +{step:4d} g-12 {gz['mean_pz']:.4f} "
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
    torch.save({"model": sd, "meta": {"experiment": "e176", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                             **meta}
    log(f"[ckpt] saved {path.name} "
        f"({', '.join(f'{k}={v}' for k, v in meta.items())})")


def load_e161_ref() -> tuple[dict, dict]:
    """Load e161's stored trajectory for the overlay; verify the embedded
    E161_REF copy against the file when present (no silent divergence)."""
    src = {"source": "embedded verbatim copy (runs/e161/metrics.json "
                     "trace_summary)", "file_present": E161_METRICS.exists(),
           "verified_vs_embedded": None, "max_abs_diff": None}
    if E161_METRICS.exists():
        mm = json.loads(E161_METRICS.read_text(encoding="utf-8"))
        ts = mm["trace_summary"]
        diffs = [abs(a - b) for k in ("base_gm12", "base_g0", "base_gp12",
                                      "site_read_span", "site_read_onset",
                                      "A129", "row0_strength", "dall_g0",
                                      "ce_r")
                 for a, b in zip(ts[k], E161_REF[k])]
        steps_ok = list(ts["freeze_steps"]) == E161_REF["freeze_steps"]
        src["max_abs_diff"] = max(diffs) if diffs else None
        src["verified_vs_embedded"] = bool(steps_ok and max(diffs) < 1e-9)
        if src["verified_vs_embedded"]:
            src["source"] = "runs/e161/metrics.json trace_summary " \
                            "(embedded copy verified, max|diff| " \
                            f"{max(diffs):.1e})"
    return dict(E161_REF), src


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e176_smoke" if SMOKE else "e176")
    log(f"E176 FREEZE THE ROOT (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()}), stagger "
        f"{STAGGER_S:.0f}s, cooldown {COOLDOWN_S:.0f}s around the ONE "
        f"training")
    time.sleep(STAGGER_S)            # launch stagger vs e170/e173 (no busy-wait)

    # ---------------- protocol rebuild (e143/e151/e152/e161 verbatim)
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
              "note": "MEASUREMENT instrument only (e152's locked j=54 pool); "
                      "NO window from this pool enters the freeze training"}
    G_POOL["pass"] = G_POOL["all_windows_name_in_place"]
    assert G_POOL["pass"], f"pool gate FAILED: {G_POOL}"

    # anchor bank (e065/e109/e143/e151/e152/e161 verbatim) — the corpus
    # half's paired component; verified name-free
    anchor = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                          for p, _ in install_occ[:16]])
    anchor_zeph = sum(1 for w in anchor if "ZEPH" in corpus.decode(w))
    G_ANCHFREE = {"anchor_zeph_windows": anchor_zeph, "pass": bool(anchor_zeph == 0)}
    assert G_ANCHFREE["pass"], "anchor bank contains ZEPH"

    # ---------------- batteries: ctx = train_text[p-PRE-j : p] (e119 verbatim)
    # install60 (the bar dials) + held30 (the dispatch's generalization dial;
    # e152's held30 construction)
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

    # ---------------- root net (THE FULLY-CONSOLIDATED ROOT) + gate vs e151
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
        # (0) base expression + CE (install60 = the bar dials; held30 = co-dial)
        out["base"] = {j: battery_cell(net, bat_ids[j], zid) for j in GEOS}
        out["base_held"] = {j: battery_cell(net, held_ids[j], zid)
                            for j in GEOS}
        out["ce_r"] = ce_fixed_cpu(net, *r_eval_xy)
        log(f"[{tag}] base: " + " ".join(f"g{j:+d} {out['base'][j]['mean_pz']:.4f}"
                                         for j in GEOS)
            + " | held30: " + " ".join(
                f"g{j:+d} {out['base_held'][j]['mean_pz']:.4f}" for j in GEOS)
            + f" | CE_R {out['ce_r']:.4f}")

        # (ii) site content at 183 — span-primary census + functional read
        def span_fn(n):
            return read_fact_at(n, pool_x, name_ids, zid, SITE_ADDR_ROW,
                                SITE_Z_XCOL)["pname_mean_over7"]

        out["site_read"] = read_fact_at(net, pool_x, name_ids, zid,
                                        SITE_ADDR_ROW, SITE_Z_XCOL)
        out["census183_span"] = row_census(net, ROWS_183, span_fn)
        cen = out["census183_span"]
        site_rows_present = [r for r in SITE_ROWS if str(r) in cen["rows"]]
        cm = max(cen["rows"][str(r)]["strength"] for r in SHARED_CTR
                 if str(r) in cen["rows"])
        sstr = max(cen["rows"][str(r)]["strength"] for r in site_rows_present)
        spos = any(cen["rows"][str(r)]["content"] and
                   cen["rows"][str(r)]["strength"] >= SITE_CTRL_MULT * cm
                   for r in site_rows_present)
        best = max(site_rows_present,
                   key=lambda r: cen["rows"][str(r)]["strength"])
        out["site_span"] = {"control_max": cm, "site_strength": sstr,
                            "bar_2x_control": SITE_CTRL_MULT * cm,
                            "site_pos": spos, "peak_row": int(best)}
        log(f"[{tag}] site(span@183) strength {sstr:+.4f} @r{best} "
            f"(2x-ctrl {SITE_CTRL_MULT * cm:.4f}) -> site_pos {spos} "
            f"| read onset {out['site_read']['pz_onset_mean']:.4f} "
            f"span {out['site_read']['pname_mean_over7']:.4f}")

        # old-band census (g0 readout) — the brake A(129) + row 0 (the sink)
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

        # deletion table: D-all (e113 set) + D-183 (the new door's necessity)
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
                cell["site_span"] = read_fact_at(
                    net, pool_x, name_ids, zid, SITE_ADDR_ROW,
                    SITE_Z_XCOL)["pname_mean_over7"]
            out["del_table"][dl] = cell
        net.load_state_dict(sd)                    # restore
        log(f"[{tag}] deletions g0: " + " | ".join(
            f"{dl} {out['del_table'][dl]['g0']['mean_pz']:.3f}"
            for dl in DELS))
        del net
        return out

    log("=" * 78)
    log("STEP-0 battery (root = e131_consolidated_e113, the FULLY-CONSOLIDATED "
        "fact; 'before')")
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
    diffs = {k: root_cells[k] - E151_ROOT[k] for k in root_cells}
    max_abs = max(abs(v) for v in diffs.values())
    G_ROOT = {"cells": root_cells, "e151_stored": E151_ROOT, "diffs": diffs,
              "max_abs_diff": max_abs, "bit_tol": G_BIT_TOL,
              "tol": G_FALLBACK_TOL,
              "bit_reproducible": bool(max_abs < G_BIT_TOL),
              "pass": bool(max_abs < G_FALLBACK_TOL)}
    log(f"G_ROOT consolidated-root gate: max|diff| {max_abs:.2e} "
        f"(tol {G_FALLBACK_TOL}, bit {G_BIT_TOL}): "
        f"{'PASS' if G_ROOT['pass'] else 'FAIL'}")
    if not G_ROOT["pass"]:
        raise RuntimeError("consolidated-root checkpoint failed its gate vs "
                           "e151 stored before-cells")
    log("gates: G_SPLICE, G_NAMEFREE, G_POOL, G_ANCHFREE, G_ROOT all PASS")

    # =====================================================================
    # THE FREEZE (ONE plain-corpus training; CPU; cooldown around it)
    # =====================================================================
    log("=" * 78)
    log(f"[thermal] cooldown {COOLDOWN_S:.0f}s before the ONE training")
    cooldown(COOLDOWN_S)
    log(f"FREEZE: {FT_STEPS}-step plain-corpus fine-tune of the CONSOLIDATED "
        f"ROOT (batch {ANCH_BS} anchors + {RAND_BS} random, full-token CE, "
        f"seed {FREEZE_SEED} (= e161's — same draw sequence), CPU "
        f"{torch.get_num_threads()} threads), snapshots at +{list(CKPT_STEPS)}")
    freeze = finetune_freeze("freeze_root", net0, anchor, train_ids, itos,
                             r_eval_xy, gm12_ids, g0_ids, zid, FREEZE_SEED)
    log(f"[thermal] cooldown {COOLDOWN_S:.0f}s after the ONE training")
    cooldown(COOLDOWN_S)
    G_DRAWFREE = {"zeph_violations": freeze["zeph_violations"],
                  "pass": bool(freeze["zeph_violations"] == 0)}
    assert G_DRAWFREE["pass"], "name token leaked into a training window"

    save_ckpt("e176_root_freeze", freeze["sds"][max(freeze["sds"])],
              {"desc": f"e131_consolidated_e113 (the FULLY-consolidated root) "
                       f"+ {max(freeze['sds'])}-step PLAIN-CORPUS freeze "
                       f"(no fact windows, no name tokens; batch 32 = 16 "
                       f"anchors + 16 random corpus, full-token CE), seed "
                       f"{FREEZE_SEED}",
               "steps": int(max(freeze["sds"])), "seed": FREEZE_SEED,
               "base": f"runs/checkpoints/{ROOT_CK}"})
    for s in sorted(freeze["sds"]):
        if s == max(freeze["sds"]):
            continue
        save_ckpt(f"e176_root_freeze_s{s}", freeze["sds"][s],
                  {"desc": f"e131_consolidated_e113 + {s}-step plain-corpus "
                           f"freeze (intermediate snapshot), seed {FREEZE_SEED}",
                   "steps": int(s), "seed": FREEZE_SEED,
                   "base": f"runs/checkpoints/{ROOT_CK}"})
    missing = [s for s in CKPT_STEPS if s not in freeze["sds"]]
    if missing:
        trims.append(f"checkpoints not reached (time cap): {missing}")

    log("=" * 78)
    batteries = {"root": root}
    for s in sorted(freeze["sds"]):
        log(f"FREEZE+{s} battery")
        batteries[str(s)] = measure(freeze["sds"][s], f"f{s}")

    # =====================================================================
    # THE FREEZE TRAJECTORY + ADJUDICATION (registered clauses; no shopping)
    # =====================================================================
    steps_meas = [0] + sorted(s for s in freeze["sds"])
    trace = []
    for s in steps_meas:
        b = batteries["root" if s == 0 else str(s)]
        row = {
            "freeze_steps": s,
            "base_gm12": b["base"][-12]["mean_pz"],
            "base_g0": b["base"][0]["mean_pz"],
            "base_gp12": b["base"][12]["mean_pz"],
            "held30_gm12": b["base_held"][-12]["mean_pz"],
            "held30_g0": b["base_held"][0]["mean_pz"],
            "held30_gp12": b["base_held"][12]["mean_pz"],
            "retention_vs_root_gm12": b["base"][-12]["mean_pz"] / ROOT_GM12,
            "retention_vs_root_g0": b["base"][0]["mean_pz"] / ROOT_G0,
            "ce_r": b["ce_r"],
            "site_read_onset": b["site_read"]["pz_onset_mean"],
            "site_read_span": b["site_read"]["pname_mean_over7"],
            "site_span_strength": b["site_span"]["site_strength"],
            "site_span_peak_row": b["site_span"]["peak_row"],
            "site_span_bar_2x_control": b["site_span"]["bar_2x_control"],
            "site_pos_span": b["site_span"]["site_pos"],
            "row183_span_strength": b["census183_span"]["rows"]["183"]["strength"]
                if "183" in b["census183_span"]["rows"] else None,
            "A129": b["old_band"]["A129"],
            "row0_strength": b["old_band"]["row0_strength"],
            "band121_129_max": b["old_band"]["band121_129_max"],
            "dall_g0": b["del_table"]["d_all"]["g0"]["mean_pz"],
            "d183_g0": b["del_table"]["d183"]["g0"]["mean_pz"],
            "d183_gm12": b["del_table"]["d183"]["gm12"]["mean_pz"],
            "d183_site_span": b["del_table"]["d183"]["site_span"],
        }
        trace.append(row)

    g_seq = [r["base_gm12"] for r in trace]
    g0_seq = [r["base_g0"] for r in trace]
    ck_idx = [i for i, r in enumerate(trace) if r["freeze_steps"] > 0]
    last = trace[-1]
    g_final, g0_final = g_seq[-1], g0_seq[-1]
    earliest_below_g = next((trace[i]["freeze_steps"] for i in ck_idx
                             if g_seq[i] <= SHUT_BAR), None)
    earliest_below_g0 = next((trace[i]["freeze_steps"] for i in ck_idx
                              if g0_seq[i] <= SHUT_BAR), None)
    i50 = next((i for i in ck_idx if trace[i]["freeze_steps"] == CKPT_STEPS[0]),
               None)

    resistance_fires = bool(g_final >= SURVIVE_BAR and g0_final >= SURVIVE_BAR)
    dissolve_fires = bool(g_final <= SHUT_BAR and g0_final <= SHUT_BAR)
    g_gap = bool(SHUT_BAR < g_final < SURVIVE_BAR)
    g0_gap = bool(SHUT_BAR < g0_final < SURVIVE_BAR)
    shape_both_under_at_first_ck = (i50 is not None and
                                    g_seq[i50] <= SHUT_BAR and
                                    g0_seq[i50] <= SHUT_BAR)

    cond = {
        "CONSOLIDATION_IS_RESISTANCE": {
            "bar": SURVIVE_BAR, "g_final": g_final, "g0_final": g0_final,
            "g_clause": bool(g_final >= SURVIVE_BAR),
            "g0_clause": bool(g0_final >= SURVIVE_BAR),
            "fires": resistance_fires,
        },
        "USE_IT_OR_LOSE_IT": {
            "bar": SHUT_BAR, "g_final": g_final, "g0_final": g0_final,
            "g_clause": bool(g_final <= SHUT_BAR),
            "g0_clause": bool(g0_final <= SHUT_BAR),
            "earliest_ck_g_le_bar": earliest_below_g,
            "earliest_ck_g0_le_bar": earliest_below_g0,
            "g_min": min(g_seq[i] for i in ck_idx),
            "g0_min": min(g0_seq[i] for i in ck_idx),
            "shape_match": {
                "both_under_bar_at_first_ck": shape_both_under_at_first_ck,
                "first_ck_step": trace[i50]["freeze_steps"] if i50 is not None
                                else None,
                "retention_at_first_ck_g": g_seq[i50] / ROOT_GM12
                    if i50 is not None else None,
                "retention_at_first_ck_g0": g0_seq[i50] / ROOT_G0
                    if i50 is not None else None,
                "e161_shape": "every dial under the 0.27 bar by +50 "
                              "(g-12 0.513->0.0398, g0 0.141->0.0231, "
                              "g+12 0.715->0.0729)",
                "note": "CO-REPORT, not a bar (the registered DISSOLVES bar "
                        "is the +300 endpoint pair)",
            },
            "fires": dissolve_fires,
        },
        "GAP_BAND": {
            "g_in_gap": g_gap, "g0_in_gap": g0_gap,
            "note": "a dial in (0.27, 0.50) fails both primaries => TEXTURE",
        },
    }
    if resistance_fires:
        verdict = "CONSOLIDATION-IS-RESISTANCE"
        clause = (f"the fact survives: g-12 {g_final:.4f} >= {SURVIVE_BAR} AND "
                  f"g0 {g0_final:.4f} >= {SURVIVE_BAR} at +300 plain-corpus "
                  f"steps (retentions {g_final / ROOT_GM12:.3f} / "
                  f"{g0_final / ROOT_G0:.3f} of the root; held30 "
                  f"{last['held30_gm12']:.4f}/{last['held30_g0']:.4f}; "
                  f"row-0 sink {last['row0_strength']:+.4f} vs root "
                  f"{ROOT_ROW0:+.4f}; CE_R {last['ce_r']:.4f}) — CLS licensed; "
                  f"the resistance axis is real; claim 2 rewrites to it.")
    elif dissolve_fires:
        verdict = "USE-IT-OR-LOSE-IT"
        clause = (f"the fact dissolves: g-12 {g_final:.4f} AND g0 "
                  f"{g0_final:.4f} both <= {SHUT_BAR} at +300 plain-corpus "
                  f"steps (earliest <= bar: g-12 @{earliest_below_g}, g0 "
                  f"@{earliest_below_g0}; mins {min(g_seq[i] for i in ck_idx):.4f}"
                  f"/{min(g0_seq[i] for i in ck_idx):.4f}; both-under-bar at "
                  f"first checkpoint: {shape_both_under_at_first_ck}) — even "
                  f"consolidated memories need the anchor rehearsal; every "
                  f"past fine-tune's anchor bank was the memory's "
                  f"life-support.")
    else:
        verdict = "TEXTURE"
        clause = (f"no registered bar fired cleanly: g-12 {g_final:.4f} "
                  f"(SURVIVE >= {SURVIVE_BAR}, DISSOLVE <= {SHUT_BAR}, "
                  f"in-gap {g_gap}), g0 {g0_final:.4f} (in-gap {g0_gap}); "
                  f"trajectory g-12 {['%.4f' % g for g in g_seq]}, g0 "
                  f"{['%.4f' % g for g in g0_seq]}; held30 "
                  f"{last['held30_gm12']:.4f}/{last['held30_g0']:.4f}; "
                  f"row-0 sink {last['row0_strength']:+.4f} vs root "
                  f"{ROOT_ROW0:+.4f}; A(129) {last['A129']:+.4f}; site@183 "
                  f"{last['site_span_strength']:+.4f} (pos "
                  f"{last['site_pos_span']}); CE_R {last['ce_r']:.4f} — "
                  f"partial survival is the registered texture case; full "
                  f"trajectory reported, no bar shopping.")
    log("=" * 78)
    log(f"E176 VERDICT: {verdict}")
    log(f"  g-12 trace (freeze steps {steps_meas}): " + " -> ".join(
        f"+{r['freeze_steps']}:{r['base_gm12']:.4f}" for r in trace))
    log(f"  g0   trace: " + " -> ".join(
        f"+{r['freeze_steps']}:{r['base_g0']:.4f}" for r in trace))
    log(f"  g+12 trace: " + " -> ".join(
        f"+{r['freeze_steps']}:{r['base_gp12']:.4f}" for r in trace))
    log(f"  held30 g-12 / g0: " + " / ".join(
        f"+{r['freeze_steps']}:{r['held30_gm12']:.4f}/{r['held30_g0']:.4f}"
        for r in trace))
    log(f"  row-0 sink trace: " + " -> ".join(
        f"{r['row0_strength']:+.4f}" for r in trace))
    log(f"  A(129) trace: " + " -> ".join(f"{r['A129']:+.3f}" for r in trace))
    log(f"  D-all g0 trace: " + " -> ".join(f"{r['dall_g0']:.3f}" for r in trace))
    log(f"  CE_R trace: " + " -> ".join(f"{r['ce_r']:.3f}" for r in trace))
    log(f"  {clause}")
    log("=" * 78)

    # ---------------- outputs
    e161_ref, e161_src = load_e161_ref()
    metrics = {
        "experiment": "e176_freeze_root",
        "date": common.now_iso(),
        "registration": ("QUEUE e176 row (dispatched ~14:50Z, CPU, one "
                         "training) + T101 + the E161 DISUSE verdict; "
                         "operationalizations frozen in the module docstring "
                         "before compute"),
        "registered_prediction": REGISTERED_PREDICTION,
        "committed_prediction": None,
        "question": ("does the FULLY-CONSOLIDATED memory survive the same "
                     "plain-corpus freeze that dissolved the dwell-peak "
                     "memory in <50 steps — i.e., is consolidation the "
                     "acquisition of gradient-resistance "
                     "(CONSOLIDATION-IS-RESISTANCE), or does even the "
                     "consolidated fact need the anchor rehearsal every "
                     "past fine-tune was secretly providing "
                     "(USE-IT-OR-LOSE-IT)?"),
        "root": f"runs/checkpoints/{ROOT_CK} (the FULLY-consolidated root; "
                f"loaded, gated vs e151's stored before-cells, max|diff| "
                f"{G_ROOT['max_abs_diff']:.2e})",
        "root_meta": root_meta,
        "freeze": {"desc": "ONE plain-corpus fine-tune of the consolidated "
                           "root: batch 32 = 16 anchor-bank draws + 16 random "
                           "corpus windows, full-token CE (NO fact windows, "
                           "NO name tokens, NO mask), AdamW (0.9,0.95) wd "
                           "0.1, constant lr 1e-3, clip 1.0",
                   "recipe_lineage": "e161 VERBATIM (= e109 arm-b / e119-L / "
                                     "e143 / e151 / e152 optimizer; corpus "
                                     "half = e119 road-E corpus convention); "
                                     "seed = e161's (identical draw sequence)",
                   "steps_ran": freeze["steps_ran"], "seed": FREEZE_SEED,
                   "ckpt_steps": list(CKPT_STEPS),
                   "missing_checkpoints": missing,
                   "traj": freeze["traj"], "device": "cpu",
                   "torch_threads": torch.get_num_threads(),
                   "time_cap_s": FT_TIME_CAP,
                   "cooldown_s": COOLDOWN_S, "stagger_s": STAGGER_S,
                   "zeph_violations": freeze["zeph_violations"]},
        "e161_overlay": {"ref": e161_ref, "provenance": e161_src,
                         "e161_verdict": E161_VERDICT,
                         "note": "e161 froze the DWELL PEAK (e152_steps32) "
                                 "on this exact protocol and it dissolved in "
                                 "<50 steps; this cell is the same freeze on "
                                 "the consolidated root"},
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": PRE, "post_cap": POST_CAP,
                     "measurement_pool": {"offset": RETEACH_J,
                                          "name_xcols": [SITE_Z_XCOL,
                                                         SITE_Z_XCOL + 6],
                                          "read_rows": [183, 189],
                                          "note": "e152's locked j=54 pool, "
                                                  "instrument only"},
                     "geometries_measured": {"novel": [-12, 12],
                                             "trained_g0": 0},
                     "held30": "the same e119 ctx-battery at the held-out 30 "
                               "host occurrences (e152's held30 construction; "
                               "dispatch dial; CO-DIAL, not a bar clause)"},
        "gates": {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                  "G_POOL": G_POOL, "G_ANCHFREE": G_ANCHFREE,
                  "G_ROOT": G_ROOT, "G_DRAWFREE": G_DRAWFREE,
                  "G_SURG": gates_surg},
        "trace": trace,
        "trace_summary": {
            "freeze_steps": steps_meas,
            "base_gm12": [r["base_gm12"] for r in trace],
            "base_g0": [r["base_g0"] for r in trace],
            "base_gp12": [r["base_gp12"] for r in trace],
            "held30_gm12": [r["held30_gm12"] for r in trace],
            "held30_g0": [r["held30_g0"] for r in trace],
            "held30_gp12": [r["held30_gp12"] for r in trace],
            "site_span_strength": [r["site_span_strength"] for r in trace],
            "row183_span_strength": [r["row183_span_strength"] for r in trace],
            "site_read_span": [r["site_read_span"] for r in trace],
            "site_read_onset": [r["site_read_onset"] for r in trace],
            "A129": [r["A129"] for r in trace],
            "row0_strength": [r["row0_strength"] for r in trace],
            "dall_g0": [r["dall_g0"] for r in trace],
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
                "name-free — zero name-token leakage into the freeze; host "
                "names (FLORIZEL/ELIZABETH) are corpus-native and appear in "
                "anchors exactly as in e161's/e152's own anchor half"),
            "single_seed": "ONE trajectory (seed 10902 — the same seed and "
                "draw sequence as e161, so the two trajectories differ only "
                "in their starting nets) from ONE root snapshot; the freeze "
                "outcome is a point estimate — non-monotonic texture is "
                "this trajectory's, not a replicated law",
            "root_history": "the root e131_consolidated_e113 is ONE lineage "
                "(e065's install seed lineage -> e109/e113 consolidation) — "
                "its 'resistance' was acquired under ONE specific "
                "consolidation schedule and has never been plain-corpus-"
                "frozen before; a differently-consolidated root (different "
                "consolidation depth/schedule) could sit elsewhere on the "
                "resistance axis; n=1 root, n=1 trajectory",
            "one_distribution": "plain corpus (anchors + random windows) is "
                "ONE off-distribution stream for the fact reads; the "
                "USE-IT-OR-LOSE-IT clause's 'rehearsal' is therefore tested "
                "as the ABSENCE of exactly this one stream's fact-relevant "
                "content (no name, no site contexts beyond the anchor left "
                "contexts) — other continuations (matched replay, fresh "
                "hosts, different mixes) are untested here",
            "thread_bit_drift": "evals at 4 threads vs e151's stored cells "
                "can drift low-order bits; the G_ROOT gate reports both the "
                "5e-6 bit flag and the 0.05 fallback tolerance",
            "shape_matching_caveat": "the DISSOLVES bar's 'matching e161's "
                "trajectory shape' is operationalized as the +300 endpoint "
                "pair only; the shape match (earliest-under-bar, +50 "
                "retention) is CO-REPORTED and does not adjudicate",
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

    plot(rd / "freeze_root.png", trace, e161_ref, batteries, cond, verdict,
         clause, G_ROOT, e161_src)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'freeze_root.png'}, ckpts "
        f"runs/checkpoints/e176_root_freeze{{,_s{','.join(str(s) for s in sorted(freeze['sds']) if s != max(freeze['sds']))}}}.pt")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot

def plot(path, trace, e161, batteries, cond, verdict, clause, g_root,
         e161_src):
    """THE figure: THIS trajectory OVERLAID ON e161's (the direct visual
    comparison the dispatch demands), the anatomy dials, the generalization
    dials, and the verdict panel."""
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 10.5))
    xs = [r["freeze_steps"] for r in trace]
    xe = e161["freeze_steps"]
    ck = [r for r in trace if r["freeze_steps"] > 0]

    # (0,0) THE OVERLAY: g-12 / g0 / g+12 vs e161's (dashed)
    ax = axes[0, 0]
    ax.plot(xe, e161["base_gm12"], "--", lw=1.4, color="crimson",
            alpha=0.55, label="e161 g-12 (dwell-peak root, DISSOLVED)")
    ax.plot(xe, e161["base_g0"], "--", lw=1.4, color="tab:blue",
            alpha=0.55, label="e161 g0")
    ax.plot(xe, e161["base_gp12"], "--", lw=1.2, color="darkorange",
            alpha=0.5, label="e161 g+12")
    ax.plot(xs, [r["base_gm12"] for r in trace], "o-", ms=8, lw=2.4,
            color="crimson", label="e176 g-12 (CONSOLIDATED root, HEADLINE)")
    ax.plot(xs, [r["base_g0"] for r in trace], "^-", ms=7, lw=2.0,
            color="tab:blue", label="e176 g0 (home)")
    ax.plot(xs, [r["base_gp12"] for r in trace], "s-", ms=5, lw=1.2,
            color="darkorange", alpha=0.85, label="e176 g+12 (co-dial)")
    for yv, col, lbl in ((SURVIVE_BAR, "seagreen",
                          f"{SURVIVE_BAR} SURVIVES bar (g-12 AND g0 @+300)"),
                         (SHUT_BAR, "tab:purple",
                          f"{SHUT_BAR} DISSOLVES bar (g-12 AND g0 @+300)")):
        ax.axhline(yv, ls="--", lw=1.1, color=col, alpha=0.8, label=lbl)
    ax.annotate(f"root g-12 {trace[0]['base_gm12']:.3f}",
                (0, trace[0]["base_gm12"]), textcoords="offset points",
                xytext=(6, 6), fontsize=7.5, color="crimson")
    ax.annotate(f"e161 s32 root 0.513",
                (0, e161["base_gm12"][0]), textcoords="offset points",
                xytext=(6, -12), fontsize=7.5, color="crimson", alpha=0.6)
    ax.set_xlabel("plain-corpus freeze steps from the root (NO fact "
                  "teaching; step 0 = the root itself; same seed and draw "
                  "sequence as e161)")
    ax.set_ylabel("absolute mean p(Z), install-60 battery")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7.0, loc="center right")
    ax.set_title("FREEZE THE ROOT vs e161's freeze of the dwell peak — "
                 f"verdict: {verdict}", fontsize=10)

    # (0,1) the anatomy: brake / sink / D-all / CE_R vs e161's (dashed)
    ax = axes[0, 1]
    ax.plot(xe, e161["A129"], "--", lw=1.2, color="tab:purple", alpha=0.55,
            label="e161 A(129)")
    ax.plot(xe, e161["row0_strength"], "--", lw=1.2, color="tab:cyan",
            alpha=0.55, label="e161 row-0 (sink)")
    ax.plot(xe, e161["dall_g0"], "--", lw=1.2, color="tab:blue", alpha=0.55,
            label="e161 D-all g0")
    ax.plot(xs, [r["A129"] for r in trace], "o-", ms=6, lw=1.8,
            color="tab:purple", label="e176 A(129) brake")
    ax.plot(xs, [r["row0_strength"] for r in trace], "^-", ms=6, lw=1.8,
            color="tab:cyan", label="e176 row-0 strength (the fact's sink)")
    ax.plot(xs, [r["dall_g0"] for r in trace], "s-", ms=6, lw=1.8,
            color="tab:blue", label="e176 D-all g0")
    axr = ax.twinx()
    axr.plot(xe, e161["ce_r"], "--", lw=1.0, color="k", alpha=0.45,
             label="e161 CE_R")
    axr.plot(xs, [r["ce_r"] for r in trace], "k:o", ms=5, lw=1.2,
             alpha=0.85, label="e176 CE_R (wreckage guard)")
    ax.set_xlabel("plain-corpus freeze steps")
    ax.set_ylabel("strength / p(Z)")
    axr.set_ylabel("CE_R")
    ax.axhline(0, color="k", lw=0.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.0, loc="center right")
    ax.set_title("the brake / sink / deletion dials vs e161 — what plain "
                 "corpus does to the consolidated anatomy", fontsize=10)

    # (1,0) generalization (held30) + the site dials vs e161's (dashed)
    ax = axes[1, 0]
    ax.plot(xs, [r["base_gm12"] for r in trace], "o-", ms=4, lw=1.0,
            color="crimson", alpha=0.5, label="e176 install60 g-12 (ref)")
    ax.plot(xs, [r["held30_gm12"] for r in trace], "o--", ms=7, lw=1.8,
            color="crimson", mfc="none", label="e176 held30 g-12")
    ax.plot(xs, [r["held30_g0"] for r in trace], "^--", ms=7, lw=1.8,
            color="tab:blue", mfc="none", label="e176 held30 g0")
    for yv, col in ((SURVIVE_BAR, "seagreen"), (SHUT_BAR, "tab:purple")):
        ax.axhline(yv, ls=":", lw=1.0, color=col, alpha=0.7)
    axr = ax.twinx()
    axr.plot(xe, e161["site_read_span"], "--", lw=1.1, color="seagreen",
             alpha=0.5, label="e161 site-read span")
    axr.plot(xs, [r["site_read_span"] for r in trace], "D-", ms=5, lw=1.5,
             color="seagreen", label="e176 site-read span @183")
    axr.plot(xs, [r["site_read_onset"] for r in trace], "v-", ms=5, lw=1.5,
             color="mediumseagreen", label="e176 site-read onset @183")
    ax.set_xlabel("plain-corpus freeze steps")
    ax.set_ylabel("held30 mean p(Z)")
    axr.set_ylabel("site read @183 (instrument)")
    ax.set_ylim(-0.03, 1.05)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.0, loc="center right")
    ax.set_title("generalization (held30) + the 183-site instrument (the "
                 "consolidated root has NO 183 site — its fact-site is "
                 "row 0, panel above)", fontsize=10)

    # (1,1) verdict panel
    ax = axes[1, 1]
    ax.axis("off")
    g_txt = "  ".join(f"+{r['freeze_steps']}:{r['base_gm12']:.4f}" for r in trace)
    g0_txt = "  ".join(f"+{r['freeze_steps']}:{r['base_g0']:.4f}" for r in trace)
    vlines = [
        "REGISTERED (QUEUE e176 verbatim; frozen operationalizations):",
        f"  CONSOLIDATION-IS-RESISTANCE: g-12(300) >= {SURVIVE_BAR} AND "
        f"g0(300) >= {SURVIVE_BAR}",
        f"  USE-IT-OR-LOSE-IT: g-12(300) <= {SHUT_BAR} AND g0(300) <= "
        f"{SHUT_BAR} (e161's SHUT bar; shape CO-REPORTED)",
        "  texture (partial) => TEXTURE with the full trajectory",
        "",
        "TRACE (freeze steps from the CONSOLIDATED root):",
        f"  g-12:   {g_txt}",
        f"  g0:     {g0_txt}",
        f"  g+12:   " + "  ".join(
            f"+{r['freeze_steps']}:{r['base_gp12']:.4f}" for r in trace),
        f"  held30: " + "  ".join(
            f"+{r['freeze_steps']}:{r['held30_gm12']:.3f}"
            f"/{r['held30_g0']:.3f}" for r in trace),
        f"  row-0:  " + " ".join(f"{r['row0_strength']:+.3f}" for r in trace),
        f"  A(129): " + " ".join(f"{r['A129']:+.3f}" for r in trace),
        f"  D-all:  " + " ".join(f"{r['dall_g0']:.3f}" for r in trace),
        f"  CE_R:   " + " ".join(f"{r['ce_r']:.3f}" for r in trace),
        "",
        f"  RESISTANCE fires={cond['CONSOLIDATION_IS_RESISTANCE']['fires']} "
        f"(g {cond['CONSOLIDATION_IS_RESISTANCE']['g_final']:.4f} "
        f"[{cond['CONSOLIDATION_IS_RESISTANCE']['g_clause']}], g0 "
        f"{cond['CONSOLIDATION_IS_RESISTANCE']['g0_final']:.4f} "
        f"[{cond['CONSOLIDATION_IS_RESISTANCE']['g0_clause']}])",
        f"  USE-IT-LOSE-IT fires={cond['USE_IT_OR_LOSE_IT']['fires']} "
        f"(earliest<=bar g-12 @{cond['USE_IT_OR_LOSE_IT']['earliest_ck_g_le_bar']}, "
        f"g0 @{cond['USE_IT_OR_LOSE_IT']['earliest_ck_g0_le_bar']}; "
        f"shape@first-ck "
        f"{cond['USE_IT_OR_LOSE_IT']['shape_match']['both_under_bar_at_first_ck']})",
        f"  gap band (=> TEXTURE): g {cond['GAP_BAND']['g_in_gap']}, "
        f"g0 {cond['GAP_BAND']['g0_in_gap']}",
        "",
        f"G_ROOT consolidated-root gate: max|diff| "
        f"{g_root['max_abs_diff']:.2e} "
        f"({'PASS' if g_root['pass'] else 'FAIL'})",
        f"e161 overlay refs: {e161_src.get('source', 'embedded copy')}",
        "",
        f"VERDICT: {verdict}",
    ] + [f"  {wd}" for wd in
         [clause[i:i + 78] for i in range(0, len(clause), 78)]]
    for i, tx in enumerate(vlines):
        ax.text(0.02, 0.97 - i * 0.036, tx, fontsize=7.3, va="top",
                family="monospace")

    fig.suptitle(f"E176 — FREEZE THE ROOT (e131_consolidated_e113, the "
                 f"fully-consolidated fact; +{CKPT_STEPS[-1]} plain-corpus "
                 f"steps, same protocol/seed as e161) -> {verdict}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
