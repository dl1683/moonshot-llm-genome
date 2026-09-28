"""E153 — PHASE-SWITCH SURGERY: the memory phase's order parameter at the
wiring level (T088's 'why global?' made testable by e151's conversion).

WHY (T088 verbatim frame): memory types are PHASES of one substrate —
e131_consolidated (geometry-general phase: g-12 0.916, D-all 0.905, brake
A(129) -0.132) and e151_twodoor (site phase: g-12 0.102, D-183 kills half,
A(129) +0.005) are THE SAME LINEAGE differing by 300 locked re-teach steps
(seed 10902). The wiring diff between them IS the conversion. Question: is
the phase carried by a SMALL SET OF HEADS (transplantable) or distributed
(LN/MLP-stream state)?

NETS (on disk, gated vs runs/e151/metrics.json stored cells BEFORE surgery):
  * CONSOLIDATED  runs/checkpoints/e131_consolidated_e113.pt  (geometry phase)
  * TWODOOR       runs/checkpoints/e151_twodoor.pt            (site phase)
  * TWIN          runs/checkpoints/e119_twin_start.pt  (address-phase
    reference for the L3H4-class reader: e133's twin census has L3H4 as the
    top fact-drop head, 0.319 — loaded here only to gate + re-confirm the
    class namesake; NO transplants involve the twin).

DESIGN (eval-only; no training):
  (A) WIRING DIFF: parameter-delta ranking between the phase nets — per-head
      (W_Q, W_K, W_V row-slices of c_attn; W_O column-slice of c_proj, per
      layer-head), per-MLP, per-LN (ln1/ln2/ln_f), wpe rows, wte/lm_head.
      Report top movers + energy shares by class. Plus the e133 census
      RE-RUN on the twodoor net (same code, CPU): 36 heads zero-ablated,
      fact-drop at BOTH geometries (std g0 battery at pos 129; site onset
      read at pos 183 on the e151 re-teach pool) + attention band/sink/site
      adjacency — to find ITS fact-heads and any re-grown positional/
      address reader (the L3H4-class). Consolidated census re-run alongside
      for same-code symmetry (the FACT arm order stays e133's stored loads,
      fixed below BEFORE compute).
  (B) TRANSPLANTS: swap top-K head PARAMETER SETS (Q,K,V,O) between the
      nets — consolidated->twodoor (does the geometry door reopen?) and
      twodoor->consolidated (does it close?) — K = 1, 3, 6, named arms:
        FACT   e133 consolidated census load order (pre-registered):
               L0H3, L1H0, L0H0, L1H3, L0H1, L0H5 (nested prefixes)
        READER this run's twodoor census ranking by site-onset fact-drop
               (rule fixed before compute)
        MOVER  this run's wiring-delta ranking (rule fixed before compute)
        RAND   seeded nested permutation of the 36 heads (seed 15301)
      Full phase dial per transplant (e151's instruments verbatim): base
      expression g0, g-12, g+12, D-all {121,125,129,133,137} g0, D-r0 g0,
      D-183 (site onset+span), A(129) census strength, CE_R — with CE priced
      for every surgery cell (e150 discipline).
  (C) CONTROLS: same-K random-head transplants (both directions); a no-op
      gate per direction (transplant a head with itself -> state dict
      bit-identical, dial bit-equal).

REGISTERED PREDICTION (QUEUE e153 verbatim — no bar shopping):
  * PHASE-IN-HEADS fires if: some K<=6 head-set transplant reopens the
    geometry door in the site-phase net (g-12 >= 0.45) and/or the reverse
    transplant closes it (g-12 drops >= 70%), with CE priced per cell.
  * PHASE-DISTRIBUTED fires if: no head-set moves g-12 > 20% — the order
    parameter lives in LN/MLP-stream state (aims e135).
  * Texture => TEXTURE with numbers.

OPERATIONALIZATIONS (frozen before compute):
  * g-12 = battery mean p(Z) install-60 at ctx offset j=-12 (e119/e151
    convention); consolidated base 0.9156, twodoor base 0.1021.
  * "closes it (g-12 drops >= 70%)" = arm g-12 <= 0.30 x consolidated base
    g-12 (= 0.2747).
  * "reopens (g-12 >= 0.45)" = absolute bar on the twodoor-side arm.
  * "no head-set moves g-12 > 20%" = over ALL transplant arms (named +
    random + noop, both directions): |arm g-12 - base g-12(net)| <= 0.20 x
    base g-12(net). Strictest reading (random wreckage also blocks the
    verdict); the named-only variant is co-reported. Raw bars decide; CE
    annotates (a door moved at CE cost >> 0.1 nats is flagged wreckage in
    the honesty section, not re-adjudicated).
  * transplant = replace the TARGET net's per-head Q/K/V/O slices with the
    DONOR's at the SAME (layer, head) slot; confinement gate: only those
    slices change, all else bit-identical; no-op gate: nothing changes.
  * Adjudication order: PHASE-IN-HEADS -> PHASE-DISTRIBUTED -> TEXTURE.

INSTRUMENT PROVENANCE: load_cpu / evl_load / battery_cell / battery_pz /
ce_fixed_cpu / val_windows / deleted_wpe / read_fact_at / row_census are
lab/e151_twodoor.py VERBATIM (e143/e150/e131 lineage); Hooks.head_zero and
head_attention_geometry are lab/e133_field_anatomy.py VERBATIM (e001/e038
head-lesion convention; measurement-only softmax replication). Copied, not
imported, because those rigs pin different CUDA/thread settings and this
experiment is CPU-ONLY (e152 owns the GPU; e146 shares the CPU — threads
pinned to 4, single staggered launch, no busy-waiting).

COMPUTE ENVELOPE: eval-only, no training, CPU (CUDA_VISIBLE_DEVICES=-1),
torch threads 4. Nets 2.7M params. Single run well under 30 min.

Outputs: runs/e153/{metrics.json, phase_surgery.png}. No NOTES/THINKING/
QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python e153_phase_surgery.py     (E153_SMOKE=1 -> shakedown)
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # e152 owns the GPU

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

SMOKE = os.environ.get("E153_SMOKE") == "1"
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
    "consolidated": CKPT_DIR / "e131_consolidated_e113.pt",
    "twodoor": CKPT_DIR / "e151_twodoor.pt",
    "twin": CKPT_DIR / "e119_twin_start.pt",       # reference only
}

# ---- re-teach / site geometry (e151 verbatim) --------------------------------
RETEACH_J = 54                    # name x-cols 184..190, read rows 183..189
SITE_ADDR_ROW = 183
SITE_Z_XCOL = PRE + RETEACH_J     # 184
SITE_CONT = BLOCK - PRE - RETEACH_J - len(NAME)     # 65
D_ALL = (121, 125, 129, 133, 137)                   # e113's fixed set
GEOS = (-12, 0, 12)
KS = (1, 3, 6)
C_EMB, HD = 192, 32

# ---- gate references (runs/e151/metrics.json before/after cells, verbatim) ---
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05
REFS = {
    "consolidated": {   # e151 "before"
        "g0": 0.7850371599197388, "gm12": 0.9155886769294739,
        "gp12": 0.9478210210800171, "ce_r": 1.663516640663147,
        "site_onset": 0.8898659348487854, "site_span": 0.982668936252594,
        "A129": -0.13237020391970877,
        "d_all_g0": 0.9047248959541321, "d_r0_g0": 0.0533599890768528,
        "d183_site_onset": 0.896581768989563,
    },
    "twodoor": {        # e151 "after"
        "g0": 0.10710463672876358, "gm12": 0.1020549014210701,
        "gp12": 0.12079524248838425, "ce_r": 1.6489683389663696,
        "site_onset": 0.9980737566947937, "site_span": 0.9993568658828735,
        "A129": 0.005159098383098015,
        "d_all_g0": 0.09776780754327774, "d_r0_g0": 0.0007308662170544267,
        "d183_site_onset": 0.4863327741622925,
    },
}
TWIN_REF_G0 = 0.5563086867332458                   # e133 stored twin gate

# ---- pre-registered arm sets --------------------------------------------------
FACT_ORDER = ["L0H3", "L1H0", "L0H0", "L1H3", "L0H1", "L0H5"]
# (e133 graduated head loads: 0.4599, 0.1494, 0.1258, 0.1020, 0.0894, 0.0730)
RAND_SEED = 15301
ATT_BAR = 0.25                                     # e133 adjacency bar
CE_EVAL_SEED = 26502                               # e065 CE_R bank seed

# ---- registered bars (frozen) -------------------------------------------------
REOPEN_BAR = 0.45                                  # g-12 absolute (twodoor side)
CLOSE_FRAC = 0.30                                  # <= 30% of consolidated base
MOVE_FRAC = 0.20                                   # >20% relative move bar

REGISTERED_PREDICTION = {
    "phase_in_heads": "PHASE-IN-HEADS fires if: some K<=6 head-set transplant "
        "reopens the geometry door in the site-phase net (g-12 >= 0.45) "
        "and/or the reverse transplant closes it (g-12 drops >= 70%), with "
        "CE priced per cell.",
    "phase_distributed": "PHASE-DISTRIBUTED fires if: no head-set moves g-12 "
        "> 20% - the order parameter lives in LN/MLP-stream state (aims "
        "e135).",
    "texture": "Texture => TEXTURE with numbers.",
    "operationalizations": "g-12 = install-60 battery mean p(Z) at j=-12; "
        "closes = arm g-12 <= 0.30 x consolidated base g-12 (0.2747); "
        "reopens = arm g-12 >= 0.45 (twodoor side); moves >20% = |arm g-12 - "
        "base g-12(net)| > 0.20 x base g-12(net) over ALL arms (named+random+"
        "noop, both directions; named-only co-reported); transplant = target "
        "net's Q/K/V/O slices at the SAME (layer,head) slot replaced by the "
        "donor's, confinement-gated; FACT order = e133 stored loads "
        "[L0H3,L1H0,L0H0,L1H3,L0H1,L0H5]; READER = this run's twodoor census "
        "site-drop ranking; MOVER = this run's wiring-delta ranking; RAND = "
        "seeded nested permutation (15301); K prefixes 1/3/6; order "
        "PHASE-IN-HEADS -> PHASE-DISTRIBUTED -> TEXTURE; raw bars decide, CE "
        "annotates.",
    "no_bar_shopping": "No bar shopping; texture => TEXTURE with numbers.",
}

deviations: list[str] = [
    "CPU-only at 4 torch threads (dispatch: e152 owns the GPU, e146 shares "
    "the CPU); e151's stored cells were computed at 8 threads, so bit-flags "
    "may fall back to the lab's 0.05 tolerance — recorded per gate.",
    "READER and MOVER arm memberships are data-dependent (rules fixed in the "
    "docstring before compute: site-onset census drop; wiring delta). FACT "
    "order is e133's stored consolidated loads, fixed verbatim above.",
    "RAND arms use one seeded permutation's nested prefixes (K=1 ⊂ K=3 ⊂ "
    "K=6), mirroring the nested named arms; the draw may include named "
    "heads — seeded, not shopped.",
    "Twin net loaded for reference only (gate + L3H4 class namesake "
    "re-confirmation); no transplant arm involves it — the dispatch's "
    "'L3H4-class reader' enters as the READER rule's expected shape, with "
    "e133's stored twin census as provenance.",
    "Strict PHASE-DISTRIBUTED reading counts random/noop arms in the >20% "
    "move test (named-only variant co-reported); CE is annotative, never "
    "re-adjudicating a raw bar (wreckage flags live in honesty_reflex).",
    "Smoke mode trims: census over layers 0-1 only, arms trimmed to K=1 per "
    "family + noop, nothing adjudicated.",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: e151_twodoor.py VERBATIM (load_cpu/evl_load/battery_cell/
# battery_pz/ce_fixed_cpu/val_windows/deleted_wpe/read_fact_at/row_census);
# Hooks.head_zero + head_attention_geometry from e133_field_anatomy.py
# VERBATIM. Copied, not imported (thread/device pinning differs).

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
    """e131's read_fact_position VERBATIM ARITHMETIC, geometry parameterized:
    p(true name char) at positions addr_row..addr_row+6 over the pool windows
    (position t predicts window col t+1). Headline = p(Z) at the onset."""
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
    """e139's row_census VERBATIM (mean-arm / zero-arm / restore)."""
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


class Hooks:
    """e133's organ-ablation hooks (eval-only); head_zero = e001/e038 lesion."""

    def __init__(self, model: TinyGPT):
        self.model = model
        self.handles = []
        self.head_dim = model.cfg.n_embd // model.cfg.n_head

    def head_zero(self, layer, head):
        blk = self.model.h[layer]
        hd = self.head_dim

        def pre(m, args):
            x = args[0].clone()
            x[..., head * hd:(head + 1) * hd] = 0.0
            return (x,)
        self.handles.append(blk.attn.c_proj.register_forward_pre_hook(pre))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def head_attention_geometry(net: TinyGPT, ids: torch.Tensor, pos: int,
                            band_rows=(121, 122, 123, 124, 125, 126, 127, 128,
                                       129),
                            site_rows=tuple(range(183, 191)), bs=30) -> dict:
    """e133's per-head attention mass at the scored position, BEFORE any
    ablation (pure forward measurement; manual softmax replication —
    measurement only, never gates bit-reproduction)."""
    net.eval()
    cfg = net.cfg
    L, H, D = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head
    T0 = int(ids.shape[1])
    band_ok = [r for r in band_rows if r < T0]
    site_ok = [r for r in site_rows if r < T0]
    band_ix = torch.tensor(band_ok, device=ids.device)
    site_ix = torch.tensor(site_ok, device=ids.device)
    acc = {(l, h): {"sink": [], "band": [], "site": []}
           for l in range(L) for h in range(H)}
    for i in range(0, ids.shape[0], bs):
        x = ids[i:i + bs]
        B, Tt = x.shape
        emb = net.wte(x) + net.wpe(torch.arange(Tt, device=x.device))
        mask = torch.triu(torch.ones(Tt, Tt, device=x.device,
                                     dtype=torch.bool), 1)
        for l, block in enumerate(net.h):
            xin = block.ln1(emb)
            q, k, v = block.attn.c_attn(xin).split(cfg.n_embd, dim=2)
            q = q.view(B, Tt, H, D).transpose(1, 2)
            k = k.view(B, Tt, H, D).transpose(1, 2)
            v = v.view(B, Tt, H, D).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) / (D ** 0.5)
            att = F.softmax(att.masked_fill(mask, float("-inf")), dim=-1)
            row = att[:, :, pos, :].mean(0)          # (H, T)
            for h in range(H):
                acc[(l, h)]["sink"].append(float(row[h, 0]))
                if len(band_ok):
                    acc[(l, h)]["band"].append(float(row[h, band_ix].sum()))
                if len(site_ok):
                    acc[(l, h)]["site"].append(float(row[h, site_ix].sum()))
            y = (att @ v).transpose(1, 2).contiguous().view(B, Tt, H * D)
            emb = emb + block.attn.c_proj(y)         # same residual path
            emb = emb + block.mlp(block.ln2(emb))
    return {f"L{l}H{h}": {k: (float(np.mean(v[k])) if v[k] else 0.0)
                          for k in v}
            for (l, h), v in acc.items()}


# ------------------------------------------------------------------ surgery

def transplant_heads(sd_target: dict, sd_donor: dict,
                     heads: list[tuple[int, int]]) -> tuple[dict, dict]:
    """Replace the TARGET's per-head Q/K/V/O parameter slices with the
    DONOR's at the same (layer, head) slots; confinement gate: only those
    slices may change, all else bit-identical. No-op (donor==target) =>
    n_changed == 0."""
    out = {k: v.clone() for k, v in sd_target.items()}
    for (l, h) in heads:
        kt = f"h.{l}.attn.c_attn.weight"
        kd = f"h.{l}.attn.c_proj.weight"
        for seg in range(3):
            r0 = seg * C_EMB + h * HD
            out[kt][r0:r0 + HD, :] = sd_donor[kt][r0:r0 + HD, :]
        out[kd][:, h * HD:(h + 1) * HD] = sd_donor[kd][:, h * HD:(h + 1) * HD]
    changed = {k: int((out[k] != sd_target[k]).sum().item()) for k in out}
    n_changed = sum(changed.values())
    per_head_elems = 3 * HD * C_EMB + C_EMB * HD     # 24576
    confined, viol = True, []
    for (l, h) in heads:
        kt = f"h.{l}.attn.c_attn.weight"
        kd = f"h.{l}.attn.c_proj.weight"
        d = (out[kt] != sd_target[kt])
        rows = set(int(r) for r in torch.nonzero(d)[:, 0].tolist())
        allowed = set()
        for (ll, hh) in heads:
            if ll == l:
                for seg in range(3):
                    allowed.update(range(seg * C_EMB + hh * HD,
                                         seg * C_EMB + (hh + 1) * HD))
        if not rows <= allowed:
            confined = False; viol.append(f"{kt} rows {sorted(rows - allowed)}")
        d2 = (out[kd] != sd_target[kd])
        cols = set(int(c) for c in torch.nonzero(d2)[:, 1].tolist())
        allowed_c = set()
        for (ll, hh) in heads:
            if ll == l:
                allowed_c.update(range(hh * HD, (hh + 1) * HD))
        if not cols <= allowed_c:
            confined = False; viol.append(f"{kd} cols {sorted(cols - allowed_c)}")
    others_zero = all(v == 0 for k, v in changed.items()
                      if ".attn.c_attn.weight" not in k
                      and ".attn.c_proj.weight" not in k)
    identity = (n_changed == 0)
    gate = {"heads": [f"L{l}H{h}" for l, h in heads],
            "n_elements_changed": n_changed,
            "expected_if_not_identity": len(heads) * per_head_elems,
            "identity": bool(identity),
            "confined": bool(confined and others_zero),
            "violations": viol,
            "pass": bool(confined and others_zero
                         and (identity or n_changed
                              >= 0.95 * len(heads) * per_head_elems))}
    return out, gate


def wiring_diff(sd_a: dict, sd_b: dict) -> dict:
    """Parameter-delta ranking consolidated (a) vs twodoor (b): per-head
    (dQ,dK,dV,dO + combined), per-MLP, per-LN, wpe/wte/lm_head texture,
    energy shares by class."""
    L, H = 6, 6
    heads = {}
    for l in range(L):
        ca = sd_b[f"h.{l}.attn.c_attn.weight"] - sd_a[f"h.{l}.attn.c_attn.weight"]
        cp = sd_b[f"h.{l}.attn.c_proj.weight"] - sd_a[f"h.{l}.attn.c_proj.weight"]
        ca_a = sd_a[f"h.{l}.attn.c_attn.weight"]
        cp_a = sd_a[f"h.{l}.attn.c_proj.weight"]
        for h in range(H):
            dq = float(ca[h * HD:(h + 1) * HD].norm())
            dk = float(ca[C_EMB + h * HD:C_EMB + (h + 1) * HD].norm())
            dv = float(ca[2 * C_EMB + h * HD:2 * C_EMB + (h + 1) * HD].norm())
            do = float(cp[:, h * HD:(h + 1) * HD].norm())
            n_a = float((ca_a[h * HD:(h + 1) * HD].norm() ** 2
                         + ca_a[C_EMB + h * HD:C_EMB + (h + 1) * HD].norm() ** 2
                         + ca_a[2 * C_EMB + h * HD:
                                2 * C_EMB + (h + 1) * HD].norm() ** 2
                         + cp_a[:, h * HD:(h + 1) * HD].norm() ** 2) ** 0.5)
            heads[f"L{l}H{h}"] = {"dQ": dq, "dK": dk, "dV": dv, "dO": do,
                                  "delta": (dq ** 2 + dk ** 2 + dv ** 2
                                            + do ** 2) ** 0.5,
                                  "rel_delta": ((dq ** 2 + dk ** 2 + dv ** 2
                                                 + do ** 2) ** 0.5)
                                  / max(n_a, 1e-12)}
    mlps = {}
    for l in range(L):
        parts = [f"h.{l}.mlp.0.weight", f"h.{l}.mlp.0.bias",
                 f"h.{l}.mlp.2.weight", f"h.{l}.mlp.2.bias"]
        e = sum(float((sd_b[k] - sd_a[k]).norm() ** 2) for k in parts)
        mlps[f"mlp_l{l}"] = {"delta": e ** 0.5,
                             "d_in_w": float((sd_b[parts[0]]
                                              - sd_a[parts[0]]).norm()),
                             "d_out_w": float((sd_b[parts[2]]
                                               - sd_a[parts[2]]).norm())}
    lns = {}
    for l in range(L):
        parts = [f"h.{l}.ln1.weight", f"h.{l}.ln1.bias",
                 f"h.{l}.ln2.weight", f"h.{l}.ln2.bias"]
        e = sum(float((sd_b[k] - sd_a[k]).norm() ** 2) for k in parts)
        lns[f"ln_l{l}"] = e ** 0.5
    ln_f = float((sd_b["ln_f.weight"] - sd_a["ln_f.weight"]).norm() ** 2
                 + (sd_b["ln_f.bias"] - sd_a["ln_f.bias"]).norm() ** 2) ** 0.5
    wpe_d = sd_b["wpe.weight"] - sd_a["wpe.weight"]
    row_n = wpe_d.norm(dim=1)
    wte_d = float((sd_b["wte.weight"] - sd_a["wte.weight"]).norm())
    lm_d = float((sd_b["lm_head.weight"] - sd_a["lm_head.weight"]).norm())

    def fro(*keys):
        return float(sum(float((sd_b[k] - sd_a[k]).norm() ** 2)
                         for k in keys) ** 0.5)

    heads_E = sum(v["delta"] ** 2 for v in heads.values())
    mlp_E = sum(v["delta"] ** 2 for v in mlps.values())
    ln_E = sum(v ** 2 for v in lns.values()) + ln_f ** 2
    wpe_E = float(wpe_d.norm() ** 2)
    wte_E, lm_E = wte_d ** 2, lm_d ** 2
    tot = heads_E + mlp_E + ln_E + wpe_E + wte_E + lm_E
    return {
        "heads": heads,
        "head_rank": sorted(heads, key=lambda k: -heads[k]["delta"]),
        "mlps": mlps,
        "lns_block": lns, "ln_f": ln_f,
        "wpe": {"row_norms": {str(r): float(row_n[r]) for r in
                              (0, 1, 60, 121, 125, 129, 133, 137, 182, 183,
                               184, 189, 190, 220)},
                "mean_row_norm": float(row_n.mean()),
                "max_row": int(row_n.argmax()),
                "max_row_norm": float(row_n.max()),
                "fro": float(wpe_d.norm())},
        "wte_fro": wte_d, "lm_head_fro": lm_d,
        "energy_shares": {"heads": heads_E / tot, "mlps": mlp_E / tot,
                          "lns": ln_E / tot, "wpe": wpe_E / tot,
                          "wte": wte_E / tot, "lm_head": lm_E / tot,
                          "total_fro": tot ** 0.5},
    }


def head_census(sd: dict, tag: str, ids130, pool_x, name_ids, zid,
                r_eval_xy, layers=6) -> dict:
    """e133's head census re-run (zero-ablation, same-code CPU): fact-drop at
    BOTH geometries (std g0 battery; site onset read at 183) + CE + the
    attention geometry (sink/band at pos 129; sink/band/site at pos 183)."""
    net = evl_load(sd)
    base_g0 = battery_pz(net, ids130, zid)
    base_site = read_fact_at(net, pool_x, name_ids, zid, SITE_ADDR_ROW,
                             SITE_Z_XCOL)["pz_onset_mean"]
    base_ce = ce_fixed_cpu(net, *r_eval_xy)
    geo_std = head_attention_geometry(net, ids130, 129)
    geo_site = head_attention_geometry(net, pool_x, SITE_ADDR_ROW)
    arms = {}
    for l in range(layers):
        for h in range(6):
            hk = Hooks(net)
            hk.head_zero(l, h)
            g0 = battery_pz(net, ids130, zid)
            st = read_fact_at(net, pool_x, name_ids, zid, SITE_ADDR_ROW,
                              SITE_Z_XCOL)["pz_onset_mean"]
            ce = ce_fixed_cpu(net, *r_eval_xy)
            hk.remove()
            arms[f"L{l}H{h}"] = {"drop_g0": base_g0 - g0,
                                 "drop_site": base_site - st,
                                 "ce_r": ce}
    for k in arms:
        arms[k]["geo_std_pos129"] = geo_std[k]
        arms[k]["geo_site_pos183"] = geo_site[k]
        arms[k]["band_adjacent"] = (geo_std[k]["band"] >= ATT_BAR
                                    or geo_site[k]["band"] >= ATT_BAR)
        arms[k]["site_adjacent"] = geo_site[k]["site"] >= ATT_BAR
        arms[k]["sink_adjacent"] = (geo_std[k]["sink"] >= ATT_BAR
                                    or geo_site[k]["sink"] >= ATT_BAR)
    del net
    return {"tag": tag, "base_g0": base_g0, "base_site_onset": base_site,
            "base_ce": base_ce, "arms": arms,
            "rank_site_drop": sorted(arms, key=lambda k: -arms[k]["drop_site"]),
            "rank_g0_drop": sorted(arms, key=lambda k: -arms[k]["drop_g0"])}


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e153_smoke" if SMOKE else "e153")
    log(f"E153 PHASE-SWITCH SURGERY (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-only, torch threads {torch.get_num_threads()}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # ---------------- protocol rebuild (e151 verbatim) ----------------------
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

    # site battery pool (e151's re-teach windows verbatim construction)
    wins = []
    for p, h in install_occ:
        pre = train_ids[p - PRE - RETEACH_J: p]
        post = train_ids[p + len(h): p + len(h) + SITE_CONT]
        assert len(pre) == PRE + RETEACH_J and len(post) == SITE_CONT
        wins.append(torch.cat([pre, name_ids, post]))
    pool_x = torch.stack(wins)
    assert all(torch.equal(w[SITE_Z_XCOL:SITE_Z_XCOL + len(NAME)].cpu(),
                           name_ids) for w in pool_x)
    log(f"site battery: {tuple(pool_x.shape)} - ZEPHYRA at x-col {SITE_Z_XCOL} "
        f"(onset read row {SITE_ADDR_ROW}) in all windows")

    # batteries at the three geometries + CE_R bank
    bat_ids = {}
    for j in GEOS:
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    ids130 = bat_ids[0]
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, CE_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

    # ---------------- nets + gates ------------------------------------------
    sd = {}
    for tag in ("consolidated", "twodoor", "twin"):
        m = load_cpu(NETS[tag])
        sd[tag] = {k: v.clone() for k, v in m.state_dict().items()}
        del m
    log("nets loaded: consolidated, twodoor, twin (reference only)")

    gates_surg: dict = {}

    def phase_dial(sd_in: dict, tag: str) -> dict:
        """The full phase dial (e151 instruments verbatim): base expression
        (g0/g-12/g+12), site read, A(129) census, D-all/D-r0/D-183 deletions
        — CE priced for the base state and every surgery cell."""
        net = evl_load(sd_in)
        d: dict = {"tag": tag}
        d["base"] = {j: battery_cell(net, bat_ids[j], zid)["mean_pz"]
                     for j in GEOS}
        d["ce_r"] = ce_fixed_cpu(net, *r_eval_xy)
        sr = read_fact_at(net, pool_x, name_ids, zid, SITE_ADDR_ROW,
                          SITE_Z_XCOL)
        d["site_onset"] = sr["pz_onset_mean"]
        d["site_span"] = sr["pname_mean_over7"]
        cen = row_census(net, (129,), lambda n: battery_pz(n, ids130, zid))
        d["A129"] = cen["rows"]["129"]
        for dl, rows in (("d_all", D_ALL), ("d_r0", (0,)), ("d183", (183,))):
            sd_d, gate = deleted_wpe(sd_in, rows)
            gates_surg[f"{tag}__{dl}"] = gate
            if not gate["pass"]:
                raise RuntimeError(f"deletion gate FAILED {tag}/{dl}: {gate}")
            net.load_state_dict(sd_d)
            d[dl] = {"g0": battery_cell(net, bat_ids[0], zid)["mean_pz"],
                     "ce_r": ce_fixed_cpu(net, *r_eval_xy)}
            if dl == "d183":
                s2 = read_fact_at(net, pool_x, name_ids, zid, SITE_ADDR_ROW,
                                  SITE_Z_XCOL)
                d[dl]["site_onset"] = s2["pz_onset_mean"]
                d[dl]["site_span"] = s2["pname_mean_over7"]
        net.load_state_dict(sd_in)
        del net
        return d

    # base dials double as the pre-surgery gates vs e151's stored cells
    base_dial = {}
    gates: dict = {"G_SPLICE": {"install_mix": mix, "pass": True}}
    for tag in ("consolidated", "twodoor"):
        base_dial[tag] = phase_dial(sd[tag], f"base_{tag}")
        b = base_dial[tag]
        cells = {"g0": b["base"][0], "gm12": b["base"][-12],
                 "gp12": b["base"][12], "ce_r": b["ce_r"],
                 "site_onset": b["site_onset"], "site_span": b["site_span"],
                 "A129": b["A129"]["strength"],
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
            raise RuntimeError(f"{tag} failed its gate vs e151 stored cells")

    # twin reference gate (class namesake provenance; no surgery on it)
    net_t = evl_load(sd["twin"])
    tg0 = battery_pz(net_t, ids130, zid)
    tgeo = head_attention_geometry(net_t, ids130, 129)
    hk = Hooks(net_t); hk.head_zero(3, 4)
    t_l3h4_drop = tg0 - battery_pz(net_t, ids130, zid)
    hk.remove(); del net_t
    gates["G_TWIN"] = {"g0": tg0, "ref": TWIN_REF_G0,
                       "pass": bool(abs(tg0 - TWIN_REF_G0) < G_FALLBACK_TOL),
                       "L3H4_drop": t_l3h4_drop,
                       "L3H4_band_mass": tgeo["L3H4"]["band"],
                       "note": "address-phase reference; e133 stored L3H4 "
                               "drop 0.319 (top twin head)"}
    log(f"GATE twin: g0 {tg0:.6f} (ref {TWIN_REF_G0:.6f}) | L3H4 drop "
        f"{t_l3h4_drop:.4f} band {tgeo['L3H4']['band']:.3f}")

    # ---------------- (A) wiring diff + census re-runs ----------------------
    log("=" * 78)
    wdiff = wiring_diff(sd["consolidated"], sd["twodoor"])
    top = wdiff["head_rank"][:10]
    log("WIRING DIFF top-10 head movers: "
        + ", ".join(f"{k} {wdiff['heads'][k]['delta']:.3f}" for k in top))
    log("energy shares: "
        + ", ".join(f"{k} {v * 100:.1f}%"
                    for k, v in wdiff["energy_shares"].items()
                    if k != "total_fro")
        + f" | total fro {wdiff['energy_shares']['total_fro']:.2f}")

    census_layers = 2 if SMOKE else 6
    census = {}
    for tag in ("twodoor", "consolidated"):
        census[tag] = head_census(sd[tag], tag, ids130, pool_x, name_ids,
                                  zid, r_eval_xy, layers=census_layers)
        c = census[tag]
        log(f"CENSUS {tag}: base g0 {c['base_g0']:.4f} site "
            f"{c['base_site_onset']:.4f} | top site-drops: "
            + ", ".join(f"{k} {c['arms'][k]['drop_site']:.3f}"
                        for k in c["rank_site_drop"][:6])
            + " | top g0-drops: "
            + ", ".join(f"{k} {c['arms'][k]['drop_g0']:.3f}"
                        for k in c["rank_g0_drop"][:6]))

    reader_order = census["twodoor"]["rank_site_drop"]
    mover_order = wdiff["head_rank"]
    g = torch.Generator().manual_seed(RAND_SEED)
    perm = torch.randperm(36, generator=g).tolist()
    rand_order = [f"L{i // 6}H{i % 6}" for i in perm]

    def parse(hn: str) -> tuple[int, int]:
        return int(hn[1]), int(hn[3:])

    # ---------------- (B)+(C) transplant grid --------------------------------
    log("=" * 78)
    families = [("fact", FACT_ORDER), ("reader", reader_order),
                ("mover", mover_order), ("rand", rand_order)]
    if SMOKE:
        families = [(n, o) for n, o in families]
        ks = (1,)
    else:
        ks = KS
    arms: list[dict] = []
    noop_checks = []

    def run_arm(direction: str, arm_name: str, heads: list[str]) -> dict:
        tgt_tag = "twodoor" if direction == "c2s" else "consolidated"
        don_tag = "consolidated" if direction == "c2s" else "twodoor"
        sd_t, gate = transplant_heads(sd[tgt_tag], sd[don_tag],
                                      [parse(h) for h in heads])
        if not gate["pass"]:
            raise RuntimeError(f"transplant gate FAILED {direction}/"
                               f"{arm_name}: {gate}")
        dial = phase_dial(sd_t, f"{direction}_{arm_name}")
        bt = base_dial[tgt_tag]
        rec = {"direction": direction, "arm": arm_name, "K": len(heads),
               "heads": list(heads), "donor": don_tag, "target": tgt_tag,
               "gate": gate, "dial": dial,
               "delta_gm12": dial["base"][-12] - bt["base"][-12],
               "rel_move_gm12": abs(dial["base"][-12] - bt["base"][-12])
               / max(bt["base"][-12], 1e-12),
               "delta_ce": dial["ce_r"] - bt["ce_r"]}
        arms.append(rec)
        log(f"ARM {direction}:{arm_name} (K={len(heads)}, {','.join(heads)}): "
            f"g-12 {dial['base'][-12]:.4f} (base {bt['base'][-12]:.4f}, "
            f"{rec['rel_move_gm12'] * 100:+.1f}%) g0 {dial['base'][0]:.4f} "
            f"D-all {dial['d_all']['g0']:.4f} D-183on "
            f"{dial['d183']['site_onset']:.4f} A129 "
            f"{dial['A129']['strength']:+.4f} CE {dial['ce_r']:.4f} "
            f"(dCE {rec['delta_ce']:+.4f})")
        return rec

    for direction in ("c2s", "s2c"):
        for fam, order in families:
            for k in ks:
                run_arm(direction, f"{fam}_K{k}", order[:k])
        # no-op gate: transplant a head with itself
        sd_n, gate_n = transplant_heads(sd["twodoor" if direction == "c2s"
                                            else "consolidated"],
                                        sd["twodoor" if direction == "c2s"
                                           else "consolidated"],
                                        [parse(FACT_ORDER[0])])
        dial_n = phase_dial(sd_n, f"noop_{direction}")
        bt = base_dial["twodoor" if direction == "c2s" else "consolidated"]
        flat = lambda d: [d["base"][j] for j in GEOS] + [d["ce_r"], d["site_onset"], d["site_span"], d["A129"]["strength"], d["d_all"]["g0"], d["d_r0"]["g0"], d["d183"]["g0"], d["d183"]["site_onset"], d["d183"]["ce_r"]]
        bit = all(a == b for a, b in zip(flat(dial_n), flat(bt))) \
            and gate_n["identity"]
        noop_checks.append({"direction": direction, "gate": gate_n,
                            "dial_bit_equal_base": bool(bit)})
        log(f"NOOP {direction}: identity {gate_n['identity']}, dial "
            f"bit-equal {bit}")
        if not (gate_n["identity"] and bit):
            raise RuntimeError(f"no-op gate FAILED for {direction}")

    # ---------------- adjudication (registered; no shopping) ----------------
    log("=" * 78)
    cons_gm12 = base_dial["consolidated"]["base"][-12]
    td_gm12 = base_dial["twodoor"]["base"][-12]
    close_bar = CLOSE_FRAC * cons_gm12
    reopened = [a for a in arms if a["direction"] == "c2s"
                and a["dial"]["base"][-12] >= REOPEN_BAR]
    closed = [a for a in arms if a["direction"] == "s2c"
              and a["dial"]["base"][-12] <= close_bar]
    movers = [a for a in arms if a["rel_move_gm12"] > MOVE_FRAC]
    movers_named = [a for a in movers if a["arm"].split("_")[0] != "rand"]
    if reopened or closed:
        verdict = "PHASE-IN-HEADS"
        clause = (f"transplant(s) moved the geometry door across the "
                  f"registered bar: reopen "
                  f"{[a['direction'] + ':' + a['arm'] for a in reopened]} "
                  f"(g-12 >= {REOPEN_BAR}); close "
                  f"{[a['direction'] + ':' + a['arm'] for a in closed]} "
                  f"(g-12 <= {close_bar:.4f}) — CE prices attached per arm.")
    elif not movers:
        verdict = "PHASE-DISTRIBUTED"
        clause = (f"no transplant arm (named+random, both directions, "
                  f"K in {ks}) moved g-12 by more than {MOVE_FRAC * 100:.0f}% "
                  f"of its net's base (max move "
                  f"{max(a['rel_move_gm12'] for a in arms) * 100:.1f}%) — the "
                  f"order parameter is not carried by any K<=6 head set; it "
                  f"lives in LN/MLP-stream state (aims e135).")
    else:
        verdict = "TEXTURE"
        clause = (f"transplants move g-12 (max "
                  f"{max(a['rel_move_gm12'] for a in arms) * 100:.1f}%, "
                  f"{len(movers_named)} named arms >20%) but no arm crosses "
                  f"the reopen bar (>= {REOPEN_BAR} on the twodoor side) or "
                  f"the close bar (<= {close_bar:.4f} on the consolidated "
                  f"side) — partial head-carriage without the switch; "
                  f"numbers reported, no bar shopping.")
    adjudication = {
        "reopen_bar": REOPEN_BAR, "close_bar": close_bar,
        "move_frac": MOVE_FRAC,
        "reopen_arms": [f"{a['direction']}:{a['arm']}" for a in reopened],
        "close_arms": [f"{a['direction']}:{a['arm']}" for a in closed],
        "arms_moving_gm12_gt20pct": [f"{a['direction']}:{a['arm']}"
                                     for a in movers],
        "arms_moving_gm12_gt20pct_named_only":
            [f"{a['direction']}:{a['arm']}" for a in movers_named],
        "max_rel_move_any_arm": max(a["rel_move_gm12"] for a in arms),
        "verdict": verdict, "clause": clause,
        "ce_note": "raw bars decide; CE column annotates (wreckage flags in "
                   "honesty_reflex, never re-adjudicated)",
    }
    log(f"E153 VERDICT: {verdict}")
    log(f"  reopen arms: {adjudication['reopen_arms']} | close arms: "
        f"{adjudication['close_arms']}")
    log(f"  arms moving g-12 >20%: {adjudication['arms_moving_gm12_gt20pct']}"
        f" (named-only: {adjudication['arms_moving_gm12_gt20pct_named_only']})")
    log(f"  {clause}")

    # ---------------- outputs ------------------------------------------------
    metrics = {
        "experiment": "e153_phase_surgery",
        "date": common.now_iso(),
        "registration": ("T088 phase frame; QUEUE e153 dispatch. Registered "
                         "prediction verbatim below; operationalizations "
                         "frozen in the module docstring before compute."),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("is the memory phase (geometry-general vs site) carried "
                     "by a small set of heads (transplantable between the "
                     "two same-lineage nets) or distributed in LN/MLP-stream "
                     "state?"),
        "compute": {"device": "cpu", "torch_threads": torch.get_num_threads(),
                    "eval_only": True,
                    "note": "e152 owns the GPU; e146 shares the CPU — 4 "
                            "threads, single staggered launch, no "
                            "busy-waiting"},
        "gates": gates,
        "gates_surgery": gates_surg,
        "fact_head_provenance": {
            "fact_order": FACT_ORDER,
            "e133_graduated_loads": {"L0H3": 0.4598899483680725,
                                     "L1H0": 0.14939802885055542,
                                     "L0H0": 0.12578976154327393,
                                     "L1H3": 0.10202407814086914,
                                     "L0H1": 0.08939898014068604,
                                     "L0H5": 0.07301414012908936},
            "twin_L3H4_reader": {"e133_drop": 0.3194, "class": "address-phase "
                                 "top fact-drop head (L3H4-class)"}},
        "wiring_diff": wdiff,
        "census": census,
        "arm_orders": {"fact": FACT_ORDER, "reader": reader_order,
                       "mover": mover_order, "rand": rand_order},
        "base_dial": base_dial,
        "transplants": arms,
        "noop_checks": noop_checks,
        "adjudication": adjudication,
        "honesty_reflex": {
            "parameter_swap_path_dependence": "a transplanted head enters a "
                "foreign LN context: ln1/ln2 gains and biases differ between "
                "the phase nets (wiring_diff table), so Q/K/V/O computed for "
                "the donor's pre-LN distribution are off-manifold in the "
                "host. A failed transplant therefore cannot separate 'phase "
                "not in heads' from 'head incompatible with host LN state'; "
                "the LN deltas and the no-op/random controls bound the "
                "generic-damage part, and the asymmetry is recorded, not "
                "assumed away.",
            "single_lineage": "one net pair (one 300-step conversion, seed "
                "10902, one root seed) — the wiring diff and every "
                "transplant verdict is n=1 until replicated",
            "ce_pricing": "every dial cell carries its CE price; door "
                "movements bought at large CE cost are flagged as wreckage "
                "candidates, not phase surgery",
            "wpe_is_not_a_head": "wpe row-0 content is known-necessary in "
                "BOTH phases (e151 d_r0 kills both nets); head transplants "
                "cannot move it — a PHASE-DISTRIBUTED verdict is consistent "
                "with the phase living partly in wpe row 0's interaction "
                "with LN-stream state, which wiring_diff.wpe prices",
            "additivity": "head parameter sets are disjoint (clean swaps) "
                "but the function is nonlinear; K-prefix nesting (1<3<6) is "
                "the in-run additivity probe",
        },
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": 2739072, "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "phase_surgery.png", wdiff, base_dial, arms, adjudication,
         reader_order, census)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'phase_surgery.png'}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot

def plot(path, wdiff, base_dial, arms, adjudication, reader_order, census):
    """Legible: (top) wiring-delta ranking + class energy shares;
    (bottom) transplant dial grid with numbers."""
    fig = plt.figure(figsize=(19.5, 12.5))
    gs = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.25), hspace=0.34,
                          wspace=0.28)

    fact6 = set(FACT_ORDER[:6])
    read6 = set(reader_order[:6])
    move6 = set(wdiff["head_rank"][:6])

    # (0,0) top-20 head movers
    ax = fig.add_subplot(gs[0, 0])
    top = wdiff["head_rank"][:20][::-1]
    vals = [wdiff["heads"][k]["delta"] for k in top]
    cols, labs = [], []
    for k in top:
        tags = []
        if k in fact6: tags.append("f")
        if k in read6: tags.append("r")
        if k in move6: tags.append("m")
        cols.append("crimson" if "f" in tags else
                    "steelblue" if "r" in tags else
                    "darkorange" if "m" in tags else "lightgray")
        labs.append(k + ("" if not tags else "*" + "".join(tags)))
    ax.barh(range(len(top)), vals, 0.72, color=cols, edgecolor="k", lw=0.4)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labs, fontsize=7)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, f"{v:.2f}", va="center", fontsize=6.5)
    ax.set_xlabel("||Δ head params||_F  (Q,K,V,O combined)")
    ax.set_title("(A) WIRING DIFF — top-20 head movers\n"
                 "*f = fact-head (e133), *r = reader (e153 census), "
                 "*m = mover", fontsize=9)

    # (0,1) class energy shares
    ax = fig.add_subplot(gs[0, 1])
    es = wdiff["energy_shares"]
    cls = ["heads\n(QKVO)", "MLPs", "LANs", "wpe", "wte", "lm_head"]
    vals = [es["heads"], es["mlps"], es["lns"], es["wpe"], es["wte"],
            es["lm_head"]]
    ax.bar(range(6), [v * 100 for v in vals], 0.6,
           color=["indianred", "darkorange", "seagreen", "tab:blue",
                  "lightgray", "slategray"], edgecolor="k", lw=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v * 100 + 0.8, f"{v * 100:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(range(6))
    ax.set_xticklabels(cls, fontsize=8)
    ax.set_ylabel("% of total ||Δ||² energy")
    ax.set_ylim(0, max(105, max(vals) * 115))
    ln_txt = ("per-LN ||Δ||: " + ", ".join(
        f"L{l} {wdiff['lns_block'][f'ln_l{l}']:.3f}" for l in range(6))
        + f" | ln_f {wdiff['ln_f']:.3f}\n"
        + "wpe rows: r0 {:.3f} | r129 {:.3f} | r183 {:.3f} | mean {:.4f} | "
          "max r{} {:.3f}".format(
              wdiff["wpe"]["row_norms"]["0"], wdiff["wpe"]["row_norms"]["129"],
              wdiff["wpe"]["row_norms"]["183"],
              wdiff["wpe"]["mean_row_norm"], wdiff["wpe"]["max_row"],
              wdiff["wpe"]["max_row_norm"]))
    ax.set_title("(A) where the 300 locked steps went — class shares\n"
                 + ln_txt, fontsize=8)

    # (0,2) census panel: the re-grown reader + verdict box
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    td = census["twodoor"]
    co = census["consolidated"]
    lines = ["(A) CENSUS RE-RUN (this code, CPU)",
             "twodoor top site-drops (reader candidates):"]
    for k in td["rank_site_drop"][:6]:
        ar = td["arms"][k]
        lines.append(f"  {k}  site {ar['drop_site']:+.3f}  g0 "
                     f"{ar['drop_g0']:+.3f}  band "
                     f"{ar['geo_site_pos183']['band']:.2f}  site-adj "
                     f"{ar['site_adjacent']}")
    lines.append("consolidated top g0-drops (fact-heads):")
    for k in co["rank_g0_drop"][:4]:
        lines.append(f"  {k}  g0 {co['arms'][k]['drop_g0']:+.3f}  site "
                     f"{co['arms'][k]['drop_site']:+.3f}")
    ax.text(0.02, 0.98, "\n".join(lines), fontsize=6.8, va="top",
            family="monospace")
    a = adjudication
    ax.text(0.02, 0.42,
            f"VERDICT: {a['verdict']}\n"
            f"reopen bar g-12 >= {a['reopen_bar']}: "
            f"{a['reopen_arms'] or 'none'}\n"
            f"close bar g-12 <= {a['close_bar']:.4f}: "
            f"{a['close_arms'] or 'none'}\n"
            f"arms moving g-12 >20%: "
            f"{a['arms_moving_gm12_gt20pct'] or 'none'}\n"
            f"max move {a['max_rel_move_any_arm'] * 100:.1f}%", fontsize=7.5,
            va="top", family="monospace",
            bbox=dict(facecolor="lightyellow", alpha=0.9, edgecolor="gray"))

    # (1,:): the dial grid
    ax = fig.add_subplot(gs[1, :])
    cols = ["g0", "g-12", "g+12", "D-all\ng0", "D-r0\ng0", "D-183\nonset",
            "A(129)", "CE_R", "dCE"]
    rows = [("BASE cons", base_dial["consolidated"]),
            ("BASE twodoor", base_dial["twodoor"])]
    for a_ in arms:
        rows.append((f"{a_['direction']}:{a_['arm']}", a_["dial"]))
    bt = {"c2s": base_dial["twodoor"]["ce_r"],
          "s2c": base_dial["consolidated"]["ce_r"]}
    M = np.zeros((len(rows), len(cols)))
    for i, (name, d) in enumerate(rows):
        M[i] = [d["base"][0], d["base"][-12], d["base"][12],
                d["d_all"]["g0"], d["d_r0"]["g0"], d["d183"]["site_onset"],
                d["A129"]["strength"], d["ce_r"], np.nan]
        if i >= 2:
            M[i, 8] = d["ce_r"] - bt[arms[i - 2]["direction"]]
    norm = (M - M[:2].min(axis=0)) / (M[:2].max(axis=0) - M[:2].min(axis=0)
                                      + 1e-12)
    norm[:, 7] = (M[:, 7] - (M[:, 7].min() - 0.02)) / \
        (M[:, 7].max() - M[:, 7].min() + 0.04)
    c8 = np.clip((M[:, 8] + 0.05) / 0.6, 0, 1)
    norm[:, 8] = 1 - c8
    ax.imshow(np.clip(norm, 0, 1), cmap="RdYlGn", aspect="auto", vmin=0,
              vmax=1)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i, j]
            ax.text(j, i, f"{v:.3f}" if not np.isnan(v) else "-",
                    ha="center", va="center", fontsize=6.8)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{n} [{a_['heads'][0]}...]" if i >= 2 else n
                        for i, (n, _) in enumerate(rows)
                        for a_ in [arms[i - 2] if i >= 2 else None]],
                       fontsize=6.5)
    ax.axhline(1.5, color="k", lw=1.4)
    mid = 2 + sum(1 for a_ in arms if a_["direction"] == "c2s")
    ax.axhline(mid - 0.5 if mid > 2 else 1.5, color="gray", lw=0.8, ls="--")
    ax.set_title("(B) TRANSPLANT DIAL GRID — consolidated->twodoor (top) / "
                 "twodoor->consolidated (bottom); green/red scaled between "
                 "the two BASE rows per column (dCE: green=cheap)", fontsize=9)

    fig.suptitle("E153 — PHASE-SWITCH SURGERY: the phase's order parameter at "
                 "the wiring level — " + adjudication["verdict"],
                 fontsize=11.5)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
