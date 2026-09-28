"""E164 — THE POST-KILL CENSUS (de-circularizes T092's four-layer model;
R47 critic; QUEUE.md row e164; bars below are the dispatch wording VERBATIM,
written before compute).

WHY (R47 CRITIC): each "layer" of T092's four-layer model is defined by what
removes it — circular — unless the layers are shown SEPARABLE. e160's N2
head-ablation kills the sink-coupled fact's BEHAVIOR at flat CE (70.7% drop
at CE +0.245, mean-replace of {L1H0, L0H0}). Is the fact's SUBSTANCE still
present behind the dead readout (layers 1/2 genuinely separable), or did
content vanish with the readout (the model collapses a layer)? Second
question (T096): the site-stored fact's MLP third (33% of its locality-
filtered load, e133) — is THAT the incorrigible substrate behind its
un-killability (e125a NO-SITE-KNIFE)?

REGISTERED PREDICTION (dispatch, VERBATIM — no bar shopping):
  * SUBSTANCE-SURVIVES fires if: the killed net's content probes match the
    root (row-0/wpe unchanged, MLP ablation still costs) — layers separable;
    the kill severed ACCESS, not STORAGE.
  * CONTENT-VANISHED fires if: the content probes degrade with the kill —
    the model collapses; "content" and "readout" were one thing.
  * MLP-INCORRIGIBLE fires if: arm_b's remainder survives head-ablation but
    dies under graded MLP ablation at matched-or-lower CE — the site-fact's
    incorrigible substrate is the MLP third.
  * Texture => TEXTURE with numbers.

OPERATIONALIZATIONS (fixed before compute):
  * KILLED STATE: e131_consolidated with e160's N2 mean-replace (the head's
    32-dim c_proj-input slice -> its CE_R-bank mean over corpus windows,
    intact-net bank, e160 convention) applied IN-MEMORY to {L1H0, L0H0}.
    GATES (e160 stored, tol 5e-6 CPU->CPU, 0.05 e113-fallback recorded):
    expr_g0 = 0.22963003814220428 (drop 70.749%), CE_arm = 1.9085687398910522
    (cost +0.24505), expr_gm12 = 0.05206625908613205 (drop 94.31%).
  * PART A content probes (killed net vs SAME-RUN CPU root; e133/e131
    stored values audited with cross-device tol 5e-3):
      W1 (storage tautology): all params bit-identical root vs killed — the
          kill is hook-based, so weight-space "unchanged" holds BY
          CONSTRUCTION; recorded as such, never counted as evidence.
      W2 (row-0 functional role): e131 probe-2 instrument (wpe[0] -> mean of
          all rows / -> 0; install-60 g0 battery). PRESERVED = killed-net
          content criterion (mean-drop>0, zero-drop>0, ratio>=0.5) AND
          headroom-normalized strength >= 0.5 x root's, headroom = base -
          0.007800613064318895 (arm_b's std floor, the line's fact-free
          reference at this geometry).
      W3 (MLP-cost preserved): for BOTH of the root's top-2 MLP layers by
          load (expected l0, l5): killed headroom-normalized cost >= 0.5.
      W4 (remaining fact-heads load-bearing): >=3 of the root's top-5
          non-N2 heads (by same-run root census load) with killed-net load
          > 0.005 (noise floor; e131 control-row strengths < 0.002).
      SUBSTANCE-SURVIVES = W1 and W2 and W3 and W4.
      CONTENT-VANISHED = (row-0 content criterion FAILS on the killed net)
          AND (both top-2 MLP normalized costs < 0.2). Neither => TEXTURE.
  * PART A recovery (redundancy): (a) un-ablate one N2 head at a time —
    algebraically identical to e160's single cells (hook statelessness);
    reported as instrument continuity, NOT new evidence; (b) third-head
    sweep: additionally mean-replace EACH of the 34 remaining heads on top
    of the kill (e160 joint-mean convention, intact-bank vectors) and each
    MLP layer (killed-net census cells) — RE-OPEN = expression rises above
    the killed baseline by >= +0.05 absolute (a suppressor whose removal
    re-routes the read).
  * PART A controls: matched-RMS Gaussian noise shams (e133 convention,
    seeds 16401-3) on the killed net for mlp_l0 / mlp_l5 / the top head;
    compact install-net column (e048_repro base + N2-kill gate vs e160's
    stored install cell + 4 census probes) — report-only.
  * PART B (arm_b, the site-stored fact): B-ladder heads taken from e125a's
    STORED split-window selection (pool_sel; expected L1H2, L0H5, L1H4,
    L1H1, L0H1) — NOT re-selected; e164's PRIMARY bar read = pool_bar site
    onset (held prompts 15-29, filler seed 12512; virgin to selection,
    e125a convention). pool_full (e133's exact battery) = audit/companion.
    Sets: B1..B5 ladder prefixes (mean mode; B2/B4 zero companions);
    MLP singles l0..l5 (mean mode, e133 convention); graded by e133
    site_locked load (G2={l0,l5}, G4=+l2,l4, G6=all); graded EXCLUDING the
    general-critical l0 (E2={l5,l2}, E4=+l3,l4, E5=+l1); flat-budget pair
    F2={l1,l2}; combined B4+{l5}, B4+{l2,l3,l4}, B4+E4, B5+E4.
    MLP-INCORRIGIBLE = (best head-only pool_bar drop at CE<=+0.35 < 60%)
      AND (some MLP-containing cell pool_bar drop >= 60% at CE <= +0.35 —
      "matched-or-lower CE" = matched to the head-plane's flat-CE class
      where e125a probed and found no knife). Kills only at CE>=+0.70 =>
      MLP-WRECK-ONLY; strictly inside (+0.35,+0.70) => AMBIGUOUS-GAP with
      numbers; the min-CE >=60% MLP kill is recorded either way (the
      "MLP-coordinate kill at ANY CE" answer).
  * No bar shopping. Ambiguous => say AMBIGUOUS/TEXTURE with numbers.

NETS (on disk, gated; eval-only; CPU-ONLY — GPU user-occupied, threads<=4,
staggered):
  * CONSOLIDATED  runs/checkpoints/e131_consolidated_e113.pt (primary; base
    gates g0 0.7850371599197388 / CE_R 1.663516640663147 / g-12
    0.9155886173248291, tol 5e-6).
  * SITE-STORED   runs/checkpoints/e131_arm_b_corpus_spliced.pt (part B;
    gates: pool_full onset 0.9880021214485168, CE_R 1.682660698890686,
    NLL3 1.8478997945785522, std floor 0.007800613064318895).
  * INSTALL       runs/checkpoints/e048_repro.pt (control column; gate g0
    0.5563086867332458; N2-kill gated vs e160's stored install cell).

INSTRUMENT PROVENANCE: load/eval/head-replace/CE instruments are e160's
(lab/e160_headset_escalation.py) verbatim — themselves e150's/e141's with
e133's c_proj pre-hook conventions; the organ-census instruments (Hooks,
organ_stats, head geometry conventions) are e133's
(lab/e133_field_anatomy.py) verbatim; the row-0 content test and wpe D2
row-arms are e131's (lab/e131_rekeying_census.py); the split-window site
batteries and B-ladder are e125a's (lab/e125a_inverted_knife.py). Protocol
rebuild: corpus seed 1337, SPLICE_RNG host shuffle, install-60/held-30.

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 before torch import),
torch.set_num_threads(4), sequential evals, tiny staggers. Outputs:
runs/e164/{metrics.json, postkill_census.png}. No checkpoints written.

Run:  cd lab && python e164_postkill_census.py   (E164_SMOKE=1 -> shakedown)
"""
from __future__ import annotations

import json
import random
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # CPU-ONLY (GPU user-occupied)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                           # noqa: E402

torch.set_num_threads(4)                              # modest (shared CPU)

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
common.DEVICE = "cpu"
from common import Cfg, CharCorpus, TinyGPT, run_dir, save_json   # noqa: E402
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E164_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
CONS_CK = CKPT_DIR / "e131_consolidated_e113.pt"
INST_CK = CKPT_DIR / "e048_repro.pt"
ARMB_CK = CKPT_DIR / "e131_arm_b_corpus_spliced.pt"
E160_M = E43.REPO / "runs" / "e160" / "metrics.json"
E133_M = E43.REPO / "runs" / "e133" / "metrics.json"
E125A_M = E43.REPO / "runs" / "e125a" / "metrics.json"
E131_M = E43.REPO / "runs" / "e131" / "metrics.json"

GEOS = (0, -12)                   # Rule 12: g0 primary, g-12 robustness
N2_HEADS = ((1, 0), (0, 0))       # e160's N2 = top-2 no-L0H3 = {L1H0, L0H0}

# seeds (fixed before compute; all lineage-verbatim)
R_EVAL_SEED = 26502               # e065 CE_R bank seed
SKILL_SEED = 31337                # e160 base-skill windows
CORP_CONT_SEED = 12103            # e133 pool_full filler seed
SEL_SEED = 12511                  # e125a pool_sel filler seed
BAR_SEED = 12512                  # e125a pool_bar filler seed
RAND5_SEED = 13301                # e133 rand-5 wpe control rows
SHAM_BASE_SEED = 16401

# site battery geometry (e133/e125a verbatim)
FACT_PRE, FACT_POST = 12, 12
FACT_LEN = FACT_PRE + len(NAME) + FACT_POST      # 31
SPLICE_AT = 42
Z_XCOL = PRE + SPLICE_AT + FACT_PRE             # 184
SPLICE_ADDR_ROW = Z_XCOL - 1                    # 183
HALF = 2 if SMOKE else 15
N_DRAWS = 1 if SMOKE else 4     # sel/bar draws (pool_full always 30x4)

# organs / bars
BAND5 = (121, 125, 129, 133, 137)               # e113 D-all set (verbatim)
FLOOR_PZ = 0.007800613064318895                 # arm_b std floor (headroom ref)
G_BIT_TOL = 5e-6                                # CPU->CPU bit gate
G_CROSS_TOL = 5e-3                              # cross-device/device-mix audit
G_FALLBACK_TOL = 0.05                           # e113 convention
KILL_DROP = 60.0
CE_FLAT = 0.35
CE_WRECK = 0.70
REOPEN_MARGIN = 0.05                            # recovery re-open bar (abs p)
HEAD_LOAD_FLOOR = 0.005                         # W4 noise floor
NORM_BAR = 0.5                                  # W2/W3 normalized-cost bar
VANISH_BAR = 0.2                                # CONTENT-VANISHED normalized bar

# gates (full precision, from stored metrics)
G_CONS_REF_PZ = 0.7850371599197388
G_CONS_REF_CE = 1.663516640663147
G_CONS_REF_G12 = 0.9155886173248291
G_KILL_REF_G0 = 0.22963003814220428             # e160 N2/mean consolidated
G_KILL_REF_CE = 1.9085687398910522
G_KILL_REF_G12 = 0.05206625908613205
G_KILL_REF_DROP = 70.74915050318364
G_KILL_REF_CE_COST = 0.24505209922790527
G_INST_REF = 0.5563086867332458
G_ARMB_REF_SITE = 0.9880021214485168
G_ARMB_REF_CE = 1.682660698890686
G_ARMB_REF_NLL3 = 1.8478997945785522
G_ARMB_REF_STD = 0.007800613064318895
G_E131_R0_MEAN = 0.7842019017236945             # e131 probe-2 consolidated
G_E131_R0_ZERO = 0.7316772222270098

REGISTERED_PREDICTION = {
    "substance_survives": "SUBSTANCE-SURVIVES fires if: the killed net's "
        "content probes match the root (row-0/wpe unchanged, MLP ablation "
        "still costs) — layers separable; the kill severed ACCESS, not "
        "STORAGE.",
    "content_vanished": "CONTENT-VANISHED fires if: the content probes "
        "degrade with the kill — the model collapses; 'content' and "
        "'readout' were one thing.",
    "mlp_incorrigible": "MLP-INCORRIGIBLE fires if: arm_b's remainder "
        "survives head-ablation but dies under graded MLP ablation at "
        "matched-or-lower CE — the site-fact's incorrigible substrate is "
        "the MLP third.",
    "texture": "Texture => TEXTURE with numbers.",
    "operationalizations": (
        "killed state = e160 N2 mean-replace {L1H0,L0H0} in-memory on "
        "e131_consolidated (intact-bank mean vectors), gated vs e160 stored "
        "g0/CE/g-12. W1 = params bit-identical (construction tautology, "
        "recorded not counted). W2 = killed row-0 content criterion + "
        "headroom-normalized strength >= 0.5 x root's (headroom = base - "
        "0.0078006). W3 = both root top-2 MLP layers' killed normalized "
        "cost >= 0.5. W4 = >=3 of root top-5 non-N2 heads with killed load "
        "> 0.005. SUBSTANCE-SURVIVES = W1..W4; CONTENT-VANISHED = row-0 "
        "criterion fails AND both top-2 MLP normalized costs < 0.2. "
        "Recovery: un-ablate singles (continuity, tautological); third-head "
        "mean-mode sweep + MLP census cells re-open if expression rises "
        ">= +0.05 over the killed baseline. Part B: B-ladder from e125a "
        "stored pool_sel selection; PRIMARY read pool_bar (seed 12512); "
        "MLP-INCORRIGIBLE = best head-only flat-CE drop < 60% AND some "
        "MLP-containing cell >= 60% drop at CE <= +0.35; wreck-only / "
        "ambiguous-gap / min-CE kill recorded."),
    "no_bar_shopping": "No bar shopping. Ambiguous => say AMBIGUOUS/TEXTURE "
                       "with numbers.",
}

recipe_deviations: list[str] = [
    "The killed state is hook-constructed (eval-only forward pre-hooks on "
    "c_proj) — the weight-space 'wpe unchanged' probe is a construction "
    "TAUTOLOGY (params bit-identical by definition); recorded as W1, never "
    "counted as separability evidence. The SUBSTANCE bars rest on the "
    "FUNCTIONAL probes (W2/W3/W4).",
    "Un-ablate-one-head recovery cells are algebraically identical to "
    "e160's single-ablation cells (hook statelessness: any hook-set state "
    "is the same computation regardless of construction order) — reported "
    "as instrument continuity vs e160's stored singles, NOT as new "
    "evidence; the new recovery evidence is the third-head sweep (34 "
    "mean-mode additions on top of the kill) and the killed-net MLP census "
    "cells (re-open = rise >= +0.05 over the killed baseline).",
    "Mean-vector conventions follow each instrument's lineage: head "
    "mean-replace vectors (the kill and the third-head sweep) come from "
    "the INTACT net's CE_R bank (e160's joint-mean convention, vectors "
    "never recomputed under joint ablation); the killed-net MLP census "
    "uses the KILLED net's own bank means (e133's census convention — "
    "organ_stats on the net under study); the root MLP census uses the "
    "root bank. Mode disagreement is reported texture.",
    "e133's stored root arms were GPU-computed (cross-device); audits use "
    "tol 5e-3 and the SAME-RUN CPU root census is the comparison control "
    "for every W-bar. e131's stored row-0 values were CPU 8-thread; audit "
    "tol 5e-4, recorded deltas.",
    "B-ladder heads are taken from e125a's STORED split-window selection "
    "(pool_sel, prompts 0-14, seed 12511) — NOT re-selected by e164; "
    "e164's bar read pool_bar (prompts 15-29, seed 12512) remains virgin "
    "to selection (e125a's registration discipline preserved). Audit "
    "cell: B1 zero-mode on pool_sel vs e125a's stored onset.",
    "ADDED densifier cells beyond the dispatched wording (same bars): "
    "F2={mlp_l1,l2} (the flat-budget MLP pair), the E-graded ladder "
    "EXCLUDING the general-critical l0 (e133: mlp_l0 alone costs CE "
    "+4.14 — including it in every graded set would make every graded "
    "cell a wreck by construction), and the four B+MLP combined cells.",
    "The install-net control column is compact (base + N2-kill gate + 4 "
    "census probes), not a full census — the dispatch lists e048_repro as "
    "a net, and this column prices instrument line-specificity; it is "
    "report-only.",
    "head-load noise floor for W4 set at 0.005 p(Z) points (e131 "
    "control-row strengths < 0.002; battery std ~0.03) — below single "
    "sham/noise effects, above control-row content strengths.",
]

trims: list[str] = []


# ------------------------------------------------------------------ instruments
# (e160 verbatim: load_cpu/evl_load/battery_fwd/ce_fwd/val_windows/
#  head_mean_vec/HeadReplace; e133 verbatim: Hooks/organ_stats; e131
#  verbatim: wpe row arms; e125a verbatim: split-window splice_pool)

def evl_load(sd: dict) -> TinyGPT:
    m = TinyGPT(Cfg())
    m.load_state_dict(sd)
    m.eval()
    return m


def load_cpu(path) -> TinyGPT:
    m = TinyGPT(Cfg())
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def battery_fwd(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> dict:
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
def ce_fwd(net: TinyGPT, x, y, bs=64) -> float:
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        _, loss = net(x[i:i + bs], y[i:i + bs])
        tot += float(loss.item()) * len(x[i:i + bs])
        n += len(x[i:i + bs])
    return tot / max(n, 1)


def val_windows(val_ids, val_text, n, seed, block=256):
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


def head_mean_vec(net: TinyGPT, layer: int, head: int, bank_x, bs=30):
    """Mean of the head's 32-dim slice of c_proj's input over the CE_R bank
    (e133 organ_stats / e150 / e160 convention)."""
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
    or zeros, eval-only forward pre-hooks (e160 verbatim)."""

    def __init__(self, net: TinyGPT, replace: dict):
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


class Hooks:
    """Organ ablation hooks (e133 verbatim conventions, eval-only)."""

    def __init__(self, model: TinyGPT):
        self.model = model
        self.handles = []
        self.head_dim = model.cfg.n_embd // model.cfg.n_head

    def _mlp_zero(self, layer):
        blk = self.model.h[layer]

        def h(m, a, o):
            return torch.zeros_like(o)
        self.handles.append(blk.mlp.register_forward_hook(h))

    def _mlp_mean(self, layer, mean_vec):
        blk = self.model.h[layer]

        def h(m, a, o):
            return torch.zeros_like(o) + mean_vec
        self.handles.append(blk.mlp.register_forward_hook(h))

    def head_zero(self, layer, head):
        blk = self.model.h[layer]
        hd = self.head_dim

        def pre(m, args):
            x = args[0].clone()
            x[..., head * hd:(head + 1) * hd] = 0.0
            return (x,)
        self.handles.append(blk.attn.c_proj.register_forward_pre_hook(pre))

    def head_noise(self, layer, head, std, gen_seed):
        blk = self.model.h[layer]
        hd = self.head_dim
        g = torch.Generator(device="cpu").manual_seed(gen_seed)

        def pre(m, args):
            x = args[0].clone()
            noise = torch.randn(x.shape[:-1] + (hd,),
                                generator=g).to(x.device, x.dtype) * std
            x[..., head * hd:(head + 1) * hd] += noise
            return (x,)
        self.handles.append(blk.attn.c_proj.register_forward_pre_hook(pre))

    def mlp_noise(self, layer, std, gen_seed):
        blk = self.model.h[layer]
        g = torch.Generator(device="cpu").manual_seed(gen_seed)

        def h(m, a, o):
            noise = torch.randn(o.shape, generator=g).to(o.device, o.dtype) * std
            return o + noise
        self.handles.append(blk.mlp.register_forward_hook(h))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def organ_stats(net: TinyGPT, bank_x, which: str, layer: int, head=None,
                bs=30):
    """RMS/mean of an organ's output over the CE_R bank (e133 verbatim)."""
    outs = []
    blk = net.h[layer]
    hd = net.cfg.n_embd // net.cfg.n_head

    def cap_m(m, a, o):
        outs.append(o.detach())
        return None
    if which == "mlp":
        handle = blk.mlp.register_forward_hook(cap_m)
    elif which == "head":
        def cap_h(m, args):
            x = args[0].detach()
            outs.append(x[..., head * hd:(head + 1) * hd].clone())
            return None
        handle = blk.attn.c_proj.register_forward_pre_hook(cap_h)
    else:
        raise ValueError(which)
    for i in range(0, bank_x.shape[0], bs):
        net(bank_x[i:i + bs])
    handle.remove()
    o = torch.cat([t.reshape(-1, t.shape[-1]) for t in outs], 0)
    return {"rms": float(o.pow(2).mean().sqrt()),
            "mean_vec": o.mean(0).clone() if which == "mlp" else None}


@torch.no_grad()
def site_reads(net: TinyGPT, pool: torch.Tensor, name_ids: torch.Tensor,
               zid: int, bs=30) -> dict:
    """e133 battery_site / e125a site_reads: onset p(Z) at row 183 + mean
    p(true name char) over the 7 name positions."""
    net.eval()
    onset, per_pos = [], [[] for _ in range(len(name_ids))]
    for i in range(0, pool.shape[0], bs):
        w = pool[i:i + bs]
        lg, _ = net(w)
        pr = F.softmax(lg, -1)
        for k in range(pr.shape[0]):
            onset.append(float(pr[k, SPLICE_ADDR_ROW, int(zid)]))
            for j in range(len(name_ids)):
                per_pos[j].append(
                    float(pr[k, SPLICE_ADDR_ROW + j,
                           int(w[k, Z_XCOL + j])]))
    on = torch.tensor(onset)
    allp = torch.tensor([q for pos in per_pos for q in pos])
    return {"onset_pz": float(on.mean()),
            "onset_frac_ge_0.5": float((on >= 0.5).float().mean()),
            "pname_mean_over7": float(allp.mean())}


@contextmanager
def wpe_row_arm(net: TinyGPT, row: int, mode: str):
    """e131 probe-2 wpe row arm: mode 'mean' -> wpe[r] = mean(all rows);
    'zero' -> wpe[r] = 0. In-place with restore."""
    w = net.wpe.weight.data
    orig = w[row].clone()
    if mode == "mean":
        w[row] = w.mean(0)
    else:
        w[row] = 0.0
    try:
        yield
    finally:
        w[row] = orig


@contextmanager
def wpe_rows_zero(net: TinyGPT, rows: tuple[int, ...]):
    """e133/e131 D2 subtractive row-zero, in-place with restore."""
    w = net.wpe.weight.data
    orig = w[list(rows)].clone()
    w[list(rows)] = 0.0
    try:
        yield
    finally:
        w[list(rows)] = orig


def hx(l, h):
    return f"L{l}H{h}"


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e164_smoke" if SMOKE else "e164")
    log(f"E164 THE POST-KILL CENSUS (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-only, torch threads {torch.get_num_threads()}, "
        f"cuda available = {torch.cuda.is_available()}")

    # stored references (auditable, loaded at runtime)
    e160 = json.loads(E160_M.read_text(encoding="utf-8"))
    e133 = json.loads(E133_M.read_text(encoding="utf-8"))
    e125a = json.loads(E125A_M.read_text(encoding="utf-8"))
    e131 = json.loads(E131_M.read_text(encoding="utf-8"))
    e160_kill_cell = next(c for c in e160["grid"]
                          if c["net"] == "consolidated"
                          and c["set"] == "N2:top2-noL0H3"
                          and c["mode"] == "mean")
    e160_kill_inst = next(c for c in e160["grid"]
                          if c["net"] == "install"
                          and c["set"] == "N2:top2-noL0H3"
                          and c["mode"] == "mean")
    sl133 = e133["nets"]["site_locked"]
    gr133 = e133["nets"]["graduated"]

    # ---------------- protocol rebuild (e160 verbatim)
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
    log(f"protocol rebuilt: install60 {mix}, held30")

    name_ids = corpus.encode(NAME)
    n_held = 12 if SMOKE else 30
    bat_ids = {}
    for j in GEOS:
        for tag, occ in (("install60", install_occ),
                         ("held30", held_occ[:n_held])):
            cs = [train_text[p - PRE - j: p] for p, _ in occ]
            bat_ids[(j, tag)] = torch.stack([corpus.encode(c) for c in cs])
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    skill_x, skill_y = val_windows(val_ids, val_text, 3, SKILL_SEED)
    skill_xy = (skill_x, skill_y)
    log(f"CE_R bank {tuple(r_eval_x.shape)} (seed {R_EVAL_SEED}); "
        f"batteries install{bat_ids[(0, 'install60')].shape[0]} "
        f"g0/g-12 + held{n_held}")

    # rand-5 matched control rows (e133 verbatim, seed 13301)
    ordinary = [r for r in range(256)
                if r != 0 and r not in range(121, 138) and r != 183]
    gr_ = torch.Generator().manual_seed(RAND5_SEED)
    rand5 = tuple(sorted(ordinary[i] for i in
                         torch.randperm(len(ordinary), generator=gr_)[:5]
                         .tolist()))

    # ---------------- site pools (e125a split-window construction verbatim)
    def splice_pool(prompt_list, n_draws, seed):
        pid = torch.stack([corpus.encode(c) for c in prompt_list])
        gc_ = torch.Generator().manual_seed(seed)
        src = torch.randint(len(train_ids) - (BLOCK - PRE) - 1,
                            (n_draws, len(prompt_list)), generator=gc_)
        filler = torch.stack([train_ids[s: s + BLOCK - PRE]
                              for s in src.flatten()])
        segs = []
        for p, h in install_occ:
            s = (train_text[p - FACT_PRE: p] + NAME
                 + train_text[p + len(h): p + len(h) + FACT_POST])
            assert len(s) == FACT_LEN
            segs.append(corpus.encode(s))
        fact_segs = torch.stack(segs)
        fs = fact_segs[torch.arange(filler.shape[0]) % fact_segs.shape[0]]
        cont = torch.cat([filler[:, :SPLICE_AT], fs,
                          filler[:, SPLICE_AT + FACT_LEN:]], 1)
        pool = torch.cat([torch.stack(
            [pid[k % pid.shape[0]] for k in range(filler.shape[0])]),
            cont], 1)
        assert all(torch.equal(w[Z_XCOL:Z_XCOL + len(NAME)], name_ids)
                   for w in pool)
        return pool

    # pool_full is ALWAYS the full e133 battery (30 prompts x 4 draws) —
    # it exists for stored-value gates; sel/bar carry the smoke trims.
    prompts_all = [train_text[p - PRE: p] for p, _ in held_occ]
    pool_full = splice_pool(prompts_all, 4, CORP_CONT_SEED)
    pool_sel = splice_pool(prompts_all[:HALF], N_DRAWS, SEL_SEED)
    pool_bar = splice_pool(prompts_all[HALF:HALF + HALF], N_DRAWS, BAR_SEED)
    log(f"site pools: full {tuple(pool_full.shape)} (audit) | sel "
        f"{tuple(pool_sel.shape)} | bar {tuple(pool_bar.shape)} (PRIMARY B "
        f"read) — ZEPHYRA at x-col {Z_XCOL}, address row {SPLICE_ADDR_ROW}")

    # =====================================================================
    # PART A — the post-kill content census on the consolidated net
    # =====================================================================
    log("--- PHASE A1: consolidated base + gates ---")
    net_cons = load_cpu(CONS_CK)
    sd_root = {k: v.clone() for k, v in net_cons.state_dict().items()}
    bz_root = battery_fwd(net_cons, bat_ids[(0, "install60")], zid)
    ce_root = ce_fwd(net_cons, *r_eval_xy)
    bz_root12 = battery_fwd(net_cons, bat_ids[(-12, "install60")], zid)
    held_root = battery_fwd(net_cons, bat_ids[(0, "held30")], zid)
    nll3_root = ce_fwd(net_cons, *skill_xy)
    gates = {"consolidated_base": {
        "battery_pz": bz_root["mean_pz"], "ref_pz": G_CONS_REF_PZ,
        "ce_r": ce_root, "ref_ce": G_CONS_REF_CE,
        "g12": bz_root12["mean_pz"], "ref_g12": G_CONS_REF_G12,
        "tol": G_BIT_TOL, "fallback_tol": G_FALLBACK_TOL,
        "pass": bool(abs(bz_root["mean_pz"] - G_CONS_REF_PZ) < G_FALLBACK_TOL
                     and abs(ce_root - G_CONS_REF_CE) < G_FALLBACK_TOL
                     and abs(bz_root12["mean_pz"] - G_CONS_REF_G12)
                     < G_FALLBACK_TOL),
        "bit": bool(abs(bz_root["mean_pz"] - G_CONS_REF_PZ) < G_BIT_TOL
                    and abs(ce_root - G_CONS_REF_CE) < G_BIT_TOL
                    and abs(bz_root12["mean_pz"] - G_CONS_REF_G12)
                    < G_BIT_TOL)}}
    log(f"G_CONS: g0 {bz_root['mean_pz']:.10f} CE {ce_root:.6f} g-12 "
        f"{bz_root12['mean_pz']:.10f}: "
        f"{'PASS' if gates['consolidated_base']['pass'] else 'FAIL'}"
        f"{' (bit)' if gates['consolidated_base']['bit'] else ''}")
    if not gates["consolidated_base"]["pass"]:
        raise RuntimeError("consolidated base failed its gate")

    # ---------------- PHASE A2: construct + gate the N2-KILLED state
    log("--- PHASE A2: the N2-killed state (e160 mean-replace {L1H0,L0H0}) ---")
    mv_root = {hh: head_mean_vec(net_cons, hh[0], hh[1], r_eval_x)
               for hh in N2_HEADS}
    kill_replace = {hh: mv_root[hh] for hh in N2_HEADS}

    @contextmanager
    def killed():
        with HeadReplace(net_cons, kill_replace):
            yield net_cons

    with killed():
        bz_kill = battery_fwd(net_cons, bat_ids[(0, "install60")], zid)
        ce_kill = ce_fwd(net_cons, *r_eval_xy)
        bz_kill12 = battery_fwd(net_cons, bat_ids[(-12, "install60")], zid)
        held_kill = battery_fwd(net_cons, bat_ids[(0, "held30")], zid)
        nll3_kill = ce_fwd(net_cons, *skill_xy)
    drop_kill_pct = 100.0 * (1.0 - bz_kill["mean_pz"] / bz_root["mean_pz"])
    gates["killed_state"] = {
        "expr_g0": bz_kill["mean_pz"], "ref_g0": G_KILL_REF_G0,
        "ce_arm": ce_kill, "ref_ce": G_KILL_REF_CE,
        "expr_gm12": bz_kill12["mean_pz"], "ref_gm12": G_KILL_REF_G12,
        "drop_g0_pct": drop_kill_pct, "ref_drop_pct": G_KILL_REF_DROP,
        "ce_cost": ce_kill - ce_root, "ref_ce_cost": G_KILL_REF_CE_COST,
        "tol": G_BIT_TOL, "fallback_tol": G_FALLBACK_TOL,
        "pass": bool(abs(bz_kill["mean_pz"] - G_KILL_REF_G0) < G_FALLBACK_TOL
                     and abs(ce_kill - G_KILL_REF_CE) < G_FALLBACK_TOL
                     and abs(bz_kill12["mean_pz"] - G_KILL_REF_G12)
                     < G_FALLBACK_TOL),
        "bit": bool(abs(bz_kill["mean_pz"] - G_KILL_REF_G0) < G_BIT_TOL
                    and abs(ce_kill - G_KILL_REF_CE) < G_BIT_TOL
                    and abs(bz_kill12["mean_pz"] - G_KILL_REF_G12)
                    < G_BIT_TOL),
        "e160_cell_tag": "N2:top2-noL0H3/mean (consolidated)"}
    log(f"KILLED: g0 {bz_kill['mean_pz']:.10f} (drop {drop_kill_pct:.4f}% vs "
        f"ref {G_KILL_REF_DROP:.4f}%) CE_arm {ce_kill:.10f} (cost "
        f"{ce_kill - ce_root:+.6f} vs ref {G_KILL_REF_CE_COST:+.6f}) g-12 "
        f"{bz_kill12['mean_pz']:.10f}: "
        f"{'PASS' if gates['killed_state']['pass'] else 'FAIL'}"
        f"{' (bit)' if gates['killed_state']['bit'] else ''}")
    if not gates["killed_state"]["pass"]:
        raise RuntimeError("killed state failed its e160 reproduction gate")

    # W1: the storage tautology
    sd_killed = {k: v.clone() for k, v in net_cons.state_dict().items()}
    w1 = all(torch.equal(sd_root[k], sd_killed[k]) for k in sd_root)
    gates["W1_storage_tautology"] = {
        "params_bit_identical": bool(w1),
        "note": "hook-based kill: weights untouched BY CONSTRUCTION — "
                "recorded, not counted as separability evidence"}
    log(f"W1 storage tautology: params bit-identical = {w1} (by construction)")

    # ---------------- PHASE A3: ROOT profile census (same-run CPU control)
    log("--- PHASE A3: root organ census (same-run control) ---")
    root_headroom = bz_root["mean_pz"] - FLOOR_PZ
    kill_headroom = bz_kill["mean_pz"] - FLOOR_PZ

    def full_cell(tag, apply=None, wpe_arm=None):
        """One census cell on the ROOT: battery g0 + CE."""
        hk = Hooks(net_cons) if apply else None
        cm = wpe_arm if wpe_arm is not None else nullcontext()
        with cm:
            if apply:
                apply(hk)
            bz = battery_fwd(net_cons, bat_ids[(0, "install60")], zid)
            ce = ce_fwd(net_cons, *r_eval_xy)
        if hk:
            hk.remove()
        drop = bz_root["mean_pz"] - bz["mean_pz"]
        return {"tag": tag, "expr": bz["mean_pz"], "drop": drop,
                "drop_pct": 100.0 * drop / bz_root["mean_pz"],
                "norm_cost": drop / root_headroom,
                "ce_arm": ce, "ce_cost": ce - ce_root}

    root_cells: dict[str, dict] = {}

    # row-0 / row-1 content test (e131 probe-2 instrument) — root
    for r in (0, 1):
        for mode in ("mean", "zero"):
            with wpe_row_arm(net_cons, r, mode):
                v = battery_fwd(net_cons, bat_ids[(0, "install60")],
                                zid)["mean_pz"]
                ce_v = ce_fwd(net_cons, *r_eval_xy)
            root_cells[f"row{r}_{mode}"] = {
                "tag": f"row{r}_{mode}", "expr": v,
                "drop": bz_root["mean_pz"] - v,
                "drop_pct": 100.0 * (bz_root["mean_pz"] - v)
                            / bz_root["mean_pz"],
                "norm_cost": (bz_root["mean_pz"] - v) / root_headroom,
                "ce_arm": ce_v, "ce_cost": ce_v - ce_root}
    gates["e131_row0_root_audit"] = {
        "mean_drop": root_cells["row0_mean"]["drop"],
        "ref_mean": G_E131_R0_MEAN, "zero_drop": root_cells["row0_zero"]["drop"],
        "ref_zero": G_E131_R0_ZERO, "tol": 5e-4,
        "pass": bool(abs(root_cells["row0_mean"]["drop"] - G_E131_R0_MEAN)
                     < 5e-4
                     and abs(root_cells["row0_zero"]["drop"] - G_E131_R0_ZERO)
                     < 5e-4)}
    log(f"root row0 m/z drops {root_cells['row0_mean']['drop']:+.4f}/"
        f"{root_cells['row0_zero']['drop']:+.4f} vs e131 "
        f"{G_E131_R0_MEAN:+.4f}/{G_E131_R0_ZERO:+.4f}: "
        f"{'PASS' if gates['e131_row0_root_audit']['pass'] else 'DRIFT'}")

    # wpe row-set arms (D2 zero) — root
    for tag, rows in (("wpe_band5", BAND5), ("wpe_rand5", rand5),
                      ("wpe_183", (183,))):
        root_cells[tag] = full_cell(tag, wpe_arm=wpe_rows_zero(net_cons, rows))

    # MLP layers (mean mode, root bank vectors) — root
    root_mlp_vec = {l: organ_stats(net_cons, r_eval_x, "mlp", l)["mean_vec"]
                    for l in range(6)}
    for l in range(6):
        root_cells[f"mlp_l{l}"] = full_cell(
            f"mlp_l{l}",
            apply=lambda hk, l=l, v=root_mlp_vec[l]: hk._mlp_mean(l, v))

    # all 36 heads (zero mode) — root
    head_sweep = [(l, h) for l in range(6) for h in range(6)]
    if SMOKE:
        head_sweep = head_sweep[:8] + [(1, 0), (0, 0)]
        head_sweep = sorted(set(head_sweep))
    for (l, h) in head_sweep:
        root_cells[f"head_{hx(l, h)}"] = full_cell(
            f"head_{hx(l, h)}", apply=lambda hk, l=l, h=h: hk.head_zero(l, h))
    # e133 GPU-stored audits (cross-device); stored arm keys are lowercase
    def e133_arm_key(t):
        if t.startswith("head_"):
            return t.lower()                    # head_L0H3 -> head_l0h3
        return t
    e133_audit = {}
    for t in ("mlp_l0", "mlp_l5", "head_L0H3", "wpe_band5", "wpe_183",
              "wpe_rand5"):
        k133 = e133_arm_key(t)
        ref_arm = gr133["arms"].get(k133)
        if ref_arm is not None:
            e133_audit[t] = {"mine_expr": root_cells[t]["expr"],
                             "e133_expr": ref_arm["std_install60"],
                             "diff": abs(root_cells[t]["expr"]
                                         - ref_arm["std_install60"]),
                             "tol": G_CROSS_TOL}
    gates["e133_root_arms_audit"] = {
        "cells": e133_audit,
        "pass": bool(all(v["diff"] < G_CROSS_TOL for v in e133_audit.values()))
        if e133_audit else None,
        "note": "e133 ran GPU; this run CPU — tol 5e-3, deltas recorded"}
    log(f"root census: {len(root_cells)} cells; e133 arm audit "
        + ("PASS" if gates["e133_root_arms_audit"]["pass"] else "DRIFT "
           + str({k: round(v['diff'], 5) for k, v in e133_audit.items()})))
    time.sleep(2.0)                                   # stagger (shared CPU)

    # ---------------- PHASE A4: KILLED-net content census
    log("--- PHASE A4: killed-net content census ---")
    kill_cells: dict[str, dict] = {}

    def kill_cell(tag, apply=None, wpe_arm=None):
        """One census cell ON TOP of the N2 kill."""
        hk = Hooks(net_cons) if apply else None
        cm = wpe_arm if wpe_arm is not None else nullcontext()
        with killed():
            with cm:
                if apply:
                    apply(hk)
                bz = battery_fwd(net_cons, bat_ids[(0, "install60")], zid)
                ce = ce_fwd(net_cons, *r_eval_xy)
        if hk:
            hk.remove()
        drop = bz_kill["mean_pz"] - bz["mean_pz"]
        return {"tag": tag, "expr": bz["mean_pz"], "drop": drop,
                "drop_pct": 100.0 * drop / bz_kill["mean_pz"],
                "norm_cost": drop / kill_headroom,
                "ce_arm": ce, "ce_cost": ce - ce_kill,
                "delta_vs_root_expr": bz["mean_pz"]
                - root_cells[tag]["expr"] if tag in root_cells else None}

    for r in (0, 1):
        for mode in ("mean", "zero"):
            kill_cells[f"row{r}_{mode}"] = kill_cell(
                f"row{r}_{mode}", wpe_arm=wpe_row_arm(net_cons, r, mode))
    for tag, rows in (("wpe_band5", BAND5), ("wpe_rand5", rand5),
                      ("wpe_183", (183,))):
        kill_cells[tag] = kill_cell(tag, wpe_arm=wpe_rows_zero(net_cons, rows))

    # MLP census on the killed net (killed-net bank means — e133 convention)
    with killed():
        kill_mlp_vec = {l: organ_stats(net_cons, r_eval_x, "mlp", l)["mean_vec"]
                        for l in range(6)}
    for l in range(6):
        kill_cells[f"mlp_l{l}"] = kill_cell(
            f"mlp_l{l}",
            apply=lambda hk, l=l, v=kill_mlp_vec[l]: hk._mlp_mean(l, v))

    # remaining 34 heads (zero mode) — killed net
    rem_heads = [(l, h) for (l, h) in head_sweep
                 if (l, h) not in N2_HEADS]
    for (l, h) in rem_heads:
        kill_cells[f"head_{hx(l, h)}"] = kill_cell(
            f"head_{hx(l, h)}",
            apply=lambda hk, l=l, h=h: hk.head_zero(l, h))

    log(f"killed census: {len(kill_cells)} cells; row0 m/z drops "
        f"{kill_cells['row0_mean']['drop']:+.4f}/"
        f"{kill_cells['row0_zero']['drop']:+.4f} (norm "
        f"{kill_cells['row0_mean']['norm_cost']:.3f}/"
        f"{kill_cells['row0_zero']['norm_cost']:.3f}); mlp_l0/l5 norm "
        f"{kill_cells['mlp_l0']['norm_cost']:.3f}/"
        f"{kill_cells['mlp_l5']['norm_cost']:.3f}")
    time.sleep(2.0)                                   # stagger (shared CPU)

    # ---------------- PHASE A5: shams + recovery
    log("--- PHASE A5: shams (matched-RMS noise) + recovery ---")
    shams = {}
    sham_specs = [("mlp_l0", "mlp"), ("mlp_l5", "mlp"),
                  ("head_L0H3", "head")]
    for si, (tag, kind) in enumerate(sham_specs):
        if SMOKE and si == 2:
            continue
        if kind == "mlp":
            l = int(tag.split("_l")[1])
            with killed():
                st = organ_stats(net_cons, r_eval_x, "mlp", l)
            hk = Hooks(net_cons)
            with killed():
                hk.mlp_noise(l, st["rms"], SHAM_BASE_SEED + si)
                v = battery_fwd(net_cons, bat_ids[(0, "install60")], zid)
            hk.remove()
        else:
            l, h = 0, 3
            with killed():
                st = organ_stats(net_cons, r_eval_x, "head", l, head=h)
            hk = Hooks(net_cons)
            with killed():
                hk.head_noise(l, h, st["rms"], SHAM_BASE_SEED + si)
                v = battery_fwd(net_cons, bat_ids[(0, "install60")], zid)
            hk.remove()
        shams[tag] = {"rms": st["rms"], "expr": v["mean_pz"],
                      "drop": bz_kill["mean_pz"] - v["mean_pz"],
                      "vs_ablated_drop": kill_cells[tag]["drop"]}
        log(f"  SHAM {tag} (rms {st['rms']:.4f}): expr {v['mean_pz']:.4f} "
            f"(noise drop {shams[tag]['drop']:+.4f} vs ablation drop "
            f"{kill_cells[tag]['drop']:+.4f})")

    # recovery: (a) un-ablate singles == e160 singles (continuity, tautology)
    # NB: HeadReplace({X}) = ONLY X ablated = the N2-kill with the OTHER
    # head restored; it equals e160's single-ablation cell S:X verbatim.
    recovery = {"un_ablate": {}, "third_head": {}, "note_tautology": (
        "un-ablate cells are hook-stateless re-labels of e160's single-"
        "ablation cells — instrument continuity only")}
    for (l, h) in N2_HEADS:
        other = tuple(o for o in N2_HEADS if o != (l, h))[0]
        with HeadReplace(net_cons, {(l, h): mv_root[(l, h)]}):
            v = battery_fwd(net_cons, bat_ids[(0, "install60")], zid)["mean_pz"]
        tag = f"single_{hx(l, h)}_restores_{hx(*other)}"
        e160_single = next((c for c in e160["grid"]
                            if c["net"] == "consolidated"
                            and c["set"] == "S:" + hx(l, h)
                            and c["mode"] == "mean"), None)
        recovery["un_ablate"][tag] = {
            "expr": v,
            "e160_ref": e160_single["expr_g0"] if e160_single else None,
            "reproduces_e160": bool(e160_single is not None
                                    and abs(v - e160_single["expr_g0"])
                                    < G_FALLBACK_TOL),
            "gap_closed_pct": 100.0 * (v - bz_kill["mean_pz"])
                              / (bz_root["mean_pz"] - bz_kill["mean_pz"])}
        log(f"  {tag}: expr {v:.4f} (e160 S:{hx(l, h)} ref "
            f"{e160_single['expr_g0'] if e160_single else 'n/a'}) — closes "
            f"{recovery['un_ablate'][tag]['gap_closed_pct']:.1f}% of the "
            f"kill gap")

    # (b) third-head sweep: add one more mean-replace on top of the kill
    mv_cache = dict(mv_root)

    def mv(l, h):
        if (l, h) not in mv_cache:
            mv_cache[(l, h)] = head_mean_vec(net_cons, l, h, r_eval_x)
        return mv_cache[(l, h)]

    reopen_cells = []
    for (l, h) in rem_heads:
        rep = {n2: mv_root[n2] for n2 in N2_HEADS}
        rep[(l, h)] = mv(l, h)
        with HeadReplace(net_cons, rep):
            v = battery_fwd(net_cons, bat_ids[(0, "install60")], zid)["mean_pz"]
            ce_v = ce_fwd(net_cons, *r_eval_xy)
        c = {"tag": f"+{hx(l, h)}", "expr": v,
             "reopen": v - bz_kill["mean_pz"],
             "ce_cost": ce_v - ce_kill}
        recovery["third_head"][hx(l, h)] = c
        reopen_cells.append(c)
    best_reopen = max(reopen_cells, key=lambda c: c["reopen"]) \
        if reopen_cells else None
    mlp_reopen = {f"mlp_l{l}": {"expr": kill_cells[f"mlp_l{l}"]["expr"],
                                "reopen": kill_cells[f"mlp_l{l}"]["expr"]
                                - bz_kill["mean_pz"]}
                  for l in range(6)}
    best_reopen_all = max(
        [{"tag": "MLP:" + k, **v} for k, v in mlp_reopen.items()]
        + [{"tag": "HEAD:" + c["tag"], **{kk: c[kk] for kk in
                                          ("expr", "reopen")}}
           for c in reopen_cells],
        key=lambda c: c["reopen"], default=None)
    any_reopen = bool(best_reopen_all is not None
                      and best_reopen_all["reopen"] >= REOPEN_MARGIN)
    log(f"  third-head sweep: best head re-open "
        f"{best_reopen['tag'] if best_reopen else 'n/a'} "
        f"{best_reopen['reopen']:+.4f}" if best_reopen else "  (none)")
    log(f"  best re-open anywhere (heads+MLP): "
        f"{best_reopen_all['tag'] if best_reopen_all else 'n/a'} "
        f"{best_reopen_all['reopen']:+.4f} (bar >= +{REOPEN_MARGIN}) -> "
        f"{'RE-OPENS' if any_reopen else 'no re-opening'}")

    # ---------------- PHASE A6: install-net compact control (report-only)
    log("--- PHASE A6: install-net control column (report-only) ---")
    net_inst = load_cpu(INST_CK)
    bz_inst = battery_fwd(net_inst, bat_ids[(0, "install60")], zid)
    ce_inst = ce_fwd(net_inst, *r_eval_xy)
    gates["install_base"] = {"battery_pz": bz_inst["mean_pz"],
                             "ref": G_INST_REF, "tol": G_BIT_TOL,
                             "pass": bool(abs(bz_inst["mean_pz"] - G_INST_REF)
                                          < G_FALLBACK_TOL)}
    inst_col = {"base": {"expr": bz_inst["mean_pz"], "ce_r": ce_inst}}
    mv_inst = {hh: head_mean_vec(net_inst, hh[0], hh[1], r_eval_x)
               for hh in N2_HEADS}
    with HeadReplace(net_inst, {hh: mv_inst[hh] for hh in N2_HEADS}):
        ik = battery_fwd(net_inst, bat_ids[(0, "install60")], zid)["mean_pz"]
        ik_ce = ce_fwd(net_inst, *r_eval_xy)
    inst_col["n2_mean_kill"] = {
        "expr": ik, "ce_cost": ik_ce - ce_inst,
        "e160_ref_expr": e160_kill_inst["expr_g0"],
        "e160_ref_ce_cost": e160_kill_inst["ce_cost"],
        "reproduces_e160": bool(abs(ik - e160_kill_inst["expr_g0"])
                                < G_FALLBACK_TOL)}
    log(f"install N2/mean: {ik:.4f} (e160 {e160_kill_inst['expr_g0']:.4f})"
        f" CE {ik_ce - ce_inst:+.4f} (e160 "
        f"{e160_kill_inst['ce_cost']:+.4f})")
    with HeadReplace(net_inst, {hh: mv_inst[hh] for hh in N2_HEADS}):
        with wpe_row_arm(net_inst, 0, "zero"):
            v = battery_fwd(net_inst, bat_ids[(0, "install60")], zid)
        hk = Hooks(net_inst)
        with HeadReplace(net_inst, {hh: mv_inst[hh] for hh in N2_HEADS}):
            hk._mlp_zero(0)
            m0 = battery_fwd(net_inst, bat_ids[(0, "install60")], zid)
        hk.remove()
    inst_col["kill_plus_probes"] = {"row0_zero_expr": v["mean_pz"],
                                    "mlp_l0_zero_expr": m0["mean_pz"]}
    del net_inst

    # ---------------- PART A adjudication (registered)
    def row_criterion(cell_m, cell_z):
        d_m, d_z = cell_m["drop"], cell_z["drop"]
        ratio = min(d_m, d_z) / max(d_m, d_z) if max(d_m, d_z) > 0 else 0.0
        return bool(d_m > 0 and d_z > 0 and ratio >= 0.5), ratio, \
            min(d_m, d_z)

    r0_content_kill, r0_ratio_kill, r0_strength_kill = row_criterion(
        kill_cells["row0_mean"], kill_cells["row0_zero"])
    r0_content_root, r0_ratio_root, r0_strength_root = row_criterion(
        root_cells["row0_mean"], root_cells["row0_zero"])
    r0_norm_kill = r0_strength_kill / kill_headroom
    r0_norm_root = r0_strength_root / root_headroom
    W2 = bool(r0_content_kill and r0_norm_kill >= NORM_BAR * r0_norm_root)

    root_mlp_load = {l: max(0.0, root_cells[f"mlp_l{l}"]["drop"])
                     for l in range(6)}
    top2_mlp = sorted(root_mlp_load, key=lambda l: -root_mlp_load[l])[:2]
    mlp_norm_kill = {l: kill_cells[f"mlp_l{l}"]["norm_cost"] for l in range(6)}
    W3 = bool(all(mlp_norm_kill[l] >= NORM_BAR for l in top2_mlp))

    root_head_load = {t.replace("head_", ""):
                      max(0.0, root_cells[t]["drop"])
                      for t in root_cells if t.startswith("head_")}
    top5_nonN2 = sorted((k for k in root_head_load
                         if tuple(int(x) for x in
                                  k.replace("L", "").split("H"))
                         not in [tuple(n2) for n2 in N2_HEADS]),
                        key=lambda k: -root_head_load[k])[:5]
    top5_kill_load = {k: kill_cells[f"head_{k}"]["drop"] for k in top5_nonN2}
    W4 = bool(sum(1 for v in top5_kill_load.values()
                  if v > HEAD_LOAD_FLOOR) >= 3)

    W = {"W1_params_bit_identical": bool(w1),
         "W2_row0_role_preserved": W2,
         "W3_mlp_cost_preserved": W3,
         "W4_remaining_heads_load_bearing": W4}
    W_detail = {
        "W2": {"root": {"content": r0_content_root, "ratio": r0_ratio_root,
                        "strength": r0_strength_root,
                        "norm_strength": r0_norm_root},
               "killed": {"content": r0_content_kill, "ratio": r0_ratio_kill,
                          "strength": r0_strength_kill,
                          "norm_strength": r0_norm_kill},
               "bar": f"content criterion AND norm strength >= "
                      f"{NORM_BAR} x root's ({r0_norm_root:.3f})"},
        "W3": {"top2_mlp_by_root_load": [f"mlp_l{l}" for l in top2_mlp],
               "root_norm": {f"mlp_l{l}":
                             root_cells[f"mlp_l{l}"]["norm_cost"]
                             for l in top2_mlp},
               "killed_norm": {f"mlp_l{l}": mlp_norm_kill[l]
                               for l in top2_mlp},
               "bar": f"both >= {NORM_BAR} (root ~1.0 = arm drops below "
                      f"floor)"},
        "W4": {"root_top5_nonN2": top5_nonN2,
               "root_loads": {k: root_head_load[k] for k in top5_nonN2},
               "killed_loads": top5_kill_load,
               "floor": HEAD_LOAD_FLOOR},
        "sham_contrast": shams}
    vanished = bool((not r0_content_kill)
                    and all(mlp_norm_kill[l] < VANISH_BAR for l in top2_mlp))
    if all(W.values()):
        verdict_a = ("SUBSTANCE-SURVIVES (content probes match the root — "
                     "layers separable; the N2 kill severed ACCESS, not "
                     "STORAGE)")
    elif vanished:
        verdict_a = ("CONTENT-VANISHED (content probes degraded with the "
                     "kill — the model collapses; 'content' and 'readout' "
                     "were one thing)")
    else:
        verdict_a = "TEXTURE (mixed probe outcomes — numbers below)"
    # profile correlation texture
    common = [t for t in root_cells if t in kill_cells]
    lr = np.argsort([root_cells[t]["drop"] for t in common]).argsort()
    lk = np.argsort([kill_cells[t]["drop"] for t in common]).argsort()
    rank_corr = float(np.corrcoef(lr, lk)[0, 1]) if len(common) > 2 else None

    log("=" * 78)
    log(f"E164 PART A VERDICT: {verdict_a}")
    log(f"  W1 {W['W1_params_bit_identical']} | W2 {W2} (row0 norm "
        f"{r0_norm_kill:.3f} vs root {r0_norm_root:.3f}) | W3 {W3} "
        f"(mlp {[f'l{l}: {mlp_norm_kill[l]:.3f}' for l in top2_mlp]}) | "
        f"W4 {W4} ({ {k: round(v, 4) for k, v in top5_kill_load.items()} })")
    log(f"  re-open anywhere: "
        f"{best_reopen_all['tag'] if best_reopen_all else 'n/a'} "
        f"{best_reopen_all['reopen']:+.4f} -> "
        f"{'RE-OPENS' if any_reopen else 'no re-opening'} | profile rank "
        f"corr {rank_corr if rank_corr is None else round(rank_corr, 3)}")
    log("=" * 78)

    # =====================================================================
    # PART B — the site-fact's MLP third on arm_b
    # =====================================================================
    log("--- PHASE B1: arm_b base + gates ---")
    net_armb = load_cpu(ARMB_CK)
    st_full = site_reads(net_armb, pool_full, name_ids, zid)
    st_sel = site_reads(net_armb, pool_sel, name_ids, zid)
    st_bar = site_reads(net_armb, pool_bar, name_ids, zid)
    ce_armb = ce_fwd(net_armb, *r_eval_xy)
    nll3_armb = ce_fwd(net_armb, *skill_xy)
    std_armb = battery_fwd(net_armb, bat_ids[(0, "install60")], zid)["mean_pz"]
    gates["arm_b"] = {
        "site_full_onset": st_full["onset_pz"], "ref": G_ARMB_REF_SITE,
        "ce_r": ce_armb, "ref_ce": G_ARMB_REF_CE,
        "nll3": nll3_armb, "ref_nll3": G_ARMB_REF_NLL3,
        "std_floor": std_armb, "ref_std": G_ARMB_REF_STD,
        "tol": G_CROSS_TOL,
        "pass": bool(abs(st_full["onset_pz"] - G_ARMB_REF_SITE) < G_CROSS_TOL
                     and abs(ce_armb - G_ARMB_REF_CE) < G_CROSS_TOL
                     and abs(std_armb - G_ARMB_REF_STD) < G_CROSS_TOL)}
    log(f"G_ARMB: full onset {st_full['onset_pz']:.10f} CE {ce_armb:.6f} "
        f"std floor {std_armb:.6f}: "
        f"{'PASS' if gates['arm_b']['pass'] else 'FAIL'}")
    if not gates["arm_b"]["pass"]:
        raise RuntimeError("arm_b failed its gate")
    base_bar = st_bar["onset_pz"]

    # B-ladder from e125a's stored split-window selection (not re-selected)
    def parse_head(s):
        return (int(s[1:s.find("H")]), int(s[s.find("H") + 1:]))
    blad = [parse_head(d_["head"]) for d_ in e125a["census"]
            ["ladder_ce_class_top5"]]
    log("B-ladder (e125a stored pool_sel selection): "
        + ", ".join(hx(*c) for c in blad))
    gates["e125a_bladder_audit"] = {
        "stored": [d_["head"] for d_ in e125a["census"]
                   ["ladder_ce_class_top5"]],
        "b1_zero_sel": None, "e125a_ref": e125a["census"]["table"][0]["onset_sel"],
        "tol": 5e-4}

    # MLP ranking from e133's stored site_locked loads (auditable)
    mlp_rank = sorted(range(6), key=lambda l: -sl133["loads"][f"mlp_l{l}"])
    log(f"MLP ranking by e133 site_locked load: "
        + ", ".join(f"l{l} ({sl133['loads'][f'mlp_l{l}']:.3f})"
                    for l in mlp_rank))

    armb_mlp_vec = {l: organ_stats(net_armb, r_eval_x, "mlp", l)["mean_vec"]
                    for l in range(6)}
    armb_head_mv = {}
    BCELLS: list[dict] = []

    def bcell(tag, heads=(), mlps=(), mode="mean"):
        """One part-B cell: head mean/zero + MLP mean arms; reads pool_bar
        (PRIMARY) + pool_full (audit) + CE + NLL3."""
        rep = {}
        for (l, h) in heads:
            if mode == "mean":
                if (l, h) not in armb_head_mv:
                    armb_head_mv[(l, h)] = head_mean_vec(net_armb, l, h,
                                                         r_eval_x)
                rep[(l, h)] = armb_head_mv[(l, h)]
            else:
                rep[(l, h)] = None
        hk = Hooks(net_armb) if (mlps or rep) else None
        with HeadReplace(net_armb, rep) if rep else nullcontext():
            if hk:
                for l in mlps:
                    hk._mlp_mean(l, armb_mlp_vec[l])
            sb = site_reads(net_armb, pool_bar, name_ids, zid)
            sf = site_reads(net_armb, pool_full, name_ids, zid)
            ce_v = ce_fwd(net_armb, *r_eval_xy)
            nll_v = ce_fwd(net_armb, *skill_xy)
        if hk:
            hk.remove()
        c = {"tag": tag, "heads": [hx(*c_) for c_ in heads],
             "mlps": [f"l{l}" for l in mlps], "mode": mode,
             "has_mlp": bool(mlps), "head_only": bool(heads and not mlps),
             "onset_bar": sb["onset_pz"],
             "drop_bar_pct": 100.0 * (1.0 - sb["onset_pz"] / base_bar),
             "onset_full": sf["onset_pz"], "pname_bar": sb["pname_mean_over7"],
             "ce_cost": ce_v - ce_armb, "nll3_delta": nll_v - nll3_armb,
             "kills60_bar": bool(100.0 * (1.0 - sb["onset_pz"] / base_bar)
                                 >= KILL_DROP)}
        BCELLS.append(c)
        log(f"  [{tag:26s}] bar {sb['onset_pz']:.4f} (drop "
            f"{c['drop_bar_pct']:5.1f}%) full {sf['onset_pz']:.4f} "
            f"| CE {c['ce_cost']:+.4f} | NLL3 {c['nll3_delta']:+.4f}")
        return c

    log("--- PHASE B2: the cell machine (bar read = pool_bar) ---")
    # audit cell first: B1 zero on pool_sel vs e125a stored
    with HeadReplace(net_armb, {blad[0]: None}):
        sa = site_reads(net_armb, pool_sel, name_ids, zid)
    gates["e125a_bladder_audit"]["b1_zero_sel"] = sa["onset_pz"]
    gates["e125a_bladder_audit"]["pass"] = bool(
        abs(sa["onset_pz"]
            - e125a["census"]["table"][0]["onset_sel"]) < 5e-4)
    log(f"B1 zero on pool_sel {sa['onset_pz']:.6f} vs e125a "
        f"{e125a['census']['table'][0]['onset_sel']:.6f}: "
        f"{'PASS' if gates['e125a_bladder_audit']['pass'] else 'DRIFT'}")

    # MLP singles (mean mode; audit mlp_l5 vs e133 stored pool_full onset)
    for l in range(6):
        bcell(f"MLP:l{l}", mlps=(l,))
    gates["e133_armb_mlp_audit"] = {
        "mine": next(c for c in BCELLS if c["tag"] == "MLP:l5")["onset_full"],
        "e133_ref": sl133["arms"]["mlp_l5"]["site_onset"],
        "diff": abs(next(c for c in BCELLS if c["tag"] == "MLP:l5")
                    ["onset_full"] - sl133["arms"]["mlp_l5"]["site_onset"]),
        "tol": G_CROSS_TOL,
        "pass": bool(abs(next(c for c in BCELLS if c["tag"] == "MLP:l5")
                         ["onset_full"]
                         - sl133["arms"]["mlp_l5"]["site_onset"])
                     < G_CROSS_TOL)}
    log(f"mlp_l5 pool_full audit: "
        f"{gates['e133_armb_mlp_audit']['mine']:.6f} vs e133 "
        f"{sl133['arms']['mlp_l5']['site_onset']:.6f} "
        f"({'PASS' if gates['e133_armb_mlp_audit']['pass'] else 'DRIFT'})")

    # graded ladders
    bcell("G2:l0+l5", mlps=(mlp_rank[0], mlp_rank[1]))
    bcell("G4:+l2+l4", mlps=tuple(mlp_rank[:4]))
    bcell("G6:all6", mlps=tuple(range(6)))
    ex0 = mlp_rank[0]
    E2 = (mlp_rank[1], mlp_rank[2])
    E4 = tuple(sorted(set(E2 + (mlp_rank[3], mlp_rank[4]))))
    E5 = tuple(sorted(set(E4 + (mlp_rank[5],))))
    bcell("E2:no-l0-top2", mlps=E2)
    bcell("E4:no-l0-top4", mlps=E4)
    bcell("E5:no-l0-top5", mlps=E5)
    bcell("F2:l1+l2-flat", mlps=(1, 2))

    # B-ladder (mean mode) + zero companions
    for k in range(1, len(blad) + 1):
        bcell(f"B{k}:" + "+".join(hx(*c_) for c_ in blad[:k]),
              heads=tuple(blad[:k]))
    bcell("B2z:top2-zero", heads=tuple(blad[:2]), mode="zero")
    bcell("B4z:top4-zero", heads=tuple(blad[:4]), mode="zero")

    # combined: heads + MLP
    B4 = tuple(blad[:4])
    B5 = tuple(blad[:5])
    bcell("B4+l5", heads=B4, mlps=(mlp_rank[1],))
    bcell("B4+Mflat{l2,l3,l4}", heads=B4, mlps=(2, 3, 4))
    bcell("B4+E4", heads=B4, mlps=E4)
    bcell("B5+E4", heads=B5, mlps=E4)

    time.sleep(2.0)                                   # stagger (shared CPU)

    # ---------------- PART B adjudication (registered)
    head_cells = [c for c in BCELLS if c["head_only"]]
    head_flat = [c for c in head_cells if c["ce_cost"] <= CE_FLAT]
    best_head_flat = max(head_flat, key=lambda c: c["drop_bar_pct"]) \
        if head_flat else None
    remainder_survives = bool(best_head_flat is None
                              or best_head_flat["drop_bar_pct"] < KILL_DROP)
    mlp_cells = [c for c in BCELLS if c["has_mlp"]]
    mlp_flat_kills = [c for c in mlp_cells
                      if c["kills60_bar"] and c["ce_cost"] <= CE_FLAT]
    mlp_kills = [c for c in mlp_cells if c["kills60_bar"]]
    min_ce_kill = min(mlp_kills, key=lambda c: c["ce_cost"]) if mlp_kills \
        else None
    mlp_wreck_only = bool(mlp_kills and not mlp_flat_kills
                          and all(c["ce_cost"] >= CE_WRECK for c in mlp_kills))
    mlp_gap = [c for c in mlp_kills if CE_FLAT < c["ce_cost"] < CE_WRECK]
    combo_flat_kills = [c for c in mlp_flat_kills
                        if not c["head_only"] and c["heads"]]

    if remainder_survives and mlp_flat_kills:
        verdict_b = ("MLP-INCORRIGIBLE (arm_b's remainder survives "
                     "head-ablation but dies under graded MLP ablation at "
                     "matched-or-lower CE — the site-fact's incorrigible "
                     "substrate is the MLP third): "
                     + "; ".join(f"{c['tag']} drop {c['drop_bar_pct']:.1f}% "
                                 f"at CE {c['ce_cost']:+.3f}"
                                 for c in mlp_flat_kills))
    elif mlp_wreck_only:
        verdict_b = ("MLP-WRECK-ONLY (the remainder dies under MLP ablation "
                     "but every >=60% MLP cell costs CE >= +0.70 — the MLP "
                     "third carries the un-killable remainder at organism "
                     "prices; un-killability at flat CE holds on BOTH "
                     f"surfaces) | min-CE kill: "
                     f"{min_ce_kill['tag']} "
                     f"{min_ce_kill['drop_bar_pct']:.1f}% @ "
                     f"{min_ce_kill['ce_cost']:+.3f}")
    elif mlp_gap:
        verdict_b = ("AMBIGUOUS-GAP (MLP kills exist strictly inside CE "
                     f"(+{CE_FLAT},+{CE_WRECK}): "
                     + "; ".join(f"{c['tag']} {c['drop_bar_pct']:.1f}% @ "
                                 f"{c['ce_cost']:+.3f}" for c in mlp_gap)
                     + ")")
    elif not mlp_kills:
        verdict_b = ("NO-MLP-KILL (no probed MLP-containing cell reaches "
                     "60% site-drop — the frontier plateaus below the kill "
                     "bar on the MLP surface too)")
    else:
        verdict_b = "ADJUDICATION-ERROR (unreachable)"

    log("=" * 78)
    log(f"E164 PART B VERDICT: {verdict_b}")
    log(f"  best flat head-only: "
        f"{best_head_flat['tag']} {best_head_flat['drop_bar_pct']:.1f}% @ "
        f"{best_head_flat['ce_cost']:+.3f}" if best_head_flat else
        "  (no flat head cell)")
    log(f"  MLP kills: "
        + str([(c["tag"], round(c["drop_bar_pct"], 1),
                round(c["ce_cost"], 3)) for c in mlp_kills]))
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e164_postkill_census",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": ("R47 critic (T092 circularity) + T096 second "
                         "question; QUEUE.md row e164. Docstring + bars "
                         "written before compute."),
        "registered_prediction": REGISTERED_PREDICTION,
        "questions": {
            "A": ("is the sink-coupled fact's SUBSTANCE still present behind "
                  "e160's N2-dead readout (layers 1/2 separable), or did "
                  "content vanish with the readout?"),
            "B": ("is the site-stored fact's MLP third (33% of its "
                  "locality-filtered load) the incorrigible substrate behind "
                  "its un-killability — is there an MLP-coordinate kill at "
                  "ANY CE?")},
        "nets": {
            "consolidated": f"runs/checkpoints/{CONS_CK.name} (primary; "
                            "N2-killed state built in-memory)",
            "site_stored": f"runs/checkpoints/{ARMB_CK.name} (part B)",
            "install_control": f"runs/checkpoints/{INST_CK.name} "
                               f"(report-only)",
            "eval_only": True},
        "gates": {"G_SPLICE": G_SPLICE, **gates},
        "part_a": {
            "bases": {"root": {"expr_g0": bz_root["mean_pz"],
                               "expr_gm12": bz_root12["mean_pz"],
                               "held30_g0": held_root["mean_pz"],
                               "ce_r": ce_root, "nll3": nll3_root,
                               "battery_detail": bz_root},
                      "killed": {"expr_g0": bz_kill["mean_pz"],
                                 "expr_gm12": bz_kill12["mean_pz"],
                                 "held30_g0": held_kill["mean_pz"],
                                 "ce_arm": ce_kill, "nll3": nll3_kill,
                                 "battery_detail": bz_kill,
                                 "drop_g0_pct": drop_kill_pct,
                                 "headroom": kill_headroom}},
            "root_headroom": root_headroom,
            "kill_construction": {"heads": [hx(*h) for h in N2_HEADS],
                                  "mode": "mean (CE_R-bank, intact net)",
                                  "e160_cell": e160_kill_cell["set"]},
            "root_census": root_cells,
            "killed_census": kill_cells,
            "shams": shams,
            "recovery": {"un_ablate": recovery["un_ablate"],
                         "third_head": recovery["third_head"],
                         "mlp_reopen": mlp_reopen,
                         "best_reopen_any": best_reopen_all,
                         "any_reopen": any_reopen,
                         "reopen_bar": REOPEN_MARGIN,
                         "note": recovery["note_tautology"]},
            "install_control_column": inst_col,
            "W_bars": W, "W_detail": W_detail,
            "profile_rank_correlation_root_vs_killed": rank_corr,
            "verdict": verdict_a},
        "part_b": {
            "bases": {"onset_full": st_full["onset_pz"],
                      "onset_sel": st_sel["onset_pz"],
                      "onset_bar": st_bar["onset_pz"],
                      "ce_r": ce_armb, "nll3": nll3_armb,
                      "std_floor": std_armb,
                      "detail_bar": st_bar},
            "b_ladder_source": "e125a stored pool_sel selection (verbatim)",
            "b_ladder": [hx(*c) for c in blad],
            "mlp_rank_e133": [f"l{l}" for l in mlp_rank],
            "cells": BCELLS,
            "head_plane": {"best_flat_head_only":
                           None if best_head_flat is None else
                           {"tag": best_head_flat["tag"],
                            "drop_bar_pct": best_head_flat["drop_bar_pct"],
                            "ce_cost": best_head_flat["ce_cost"]},
                           "remainder_survives_heads": remainder_survives},
            "mlp_plane": {"flat_kills": [c["tag"] for c in mlp_flat_kills],
                          "n_kills": len(mlp_kills),
                          "min_ce_kill60":
                          None if min_ce_kill is None else
                          {"tag": min_ce_kill["tag"],
                           "drop_bar_pct": min_ce_kill["drop_bar_pct"],
                           "ce_cost": min_ce_kill["ce_cost"]},
                          "mlp_wreck_only": mlp_wreck_only,
                          "combo_flat_kills": [c["tag"]
                                               for c in combo_flat_kills]},
            "verdict": verdict_b},
        "adjudication": {
            "SUBSTANCE_SURVIVES": {"fires": all(W.values()), "bars": W},
            "CONTENT_VANISHED": {"fires": vanished},
            "MLP_INCORRIGIBLE": {
                "fires": bool(remainder_survives and mlp_flat_kills),
                "remainder_survives_heads": remainder_survives,
                "mlp_flat_kill_cells": [c["tag"] for c in mlp_flat_kills]},
            "verdict_a": verdict_a,
            "verdict_b": verdict_b},
        "honesty_reflex": {
            "kill_state_fidelity": "the killed state is eval-only hooks "
                "reproducing e160's stored N2/mean cell at 5e-6 (g0, CE, "
                "g-12) — but hook statelessness means un-ablate/re-ablate "
                "sequences cannot test history-dependence: any hook-set is "
                "the same computation. The census's new evidence is "
                "interactions (organs ON TOP of the kill), not dynamics.",
            "weight_space_tautology": "W1 (params identical) is true by "
                "construction; the SUBSTANCE claim rests on functional "
                "probes (W2/W3/W4), which are behavior-level — logits alone "
                "do not establish representational storage, only that the "
                "residual expression still depends on the fact's organs "
                "the way the root's did.",
            "floor_compression": "the killed baseline (0.230) sits near the "
                "0.0078 floor, so normalized-cost reads saturate near 1.0; "
                "both absolute and normalized numbers are reported and the "
                "W3/W2 bars require only >=50% of the REMAINING headroom — "
                "a conservative read. Sham cells price generic noise "
                "sensitivity against the same headroom.",
            "ablation_interplay": "mean-replace vectors for heads come from "
                "the intact bank (e160 joint convention) — replaced slices "
                "interact downstream and the measured drop is the NET "
                "effect; the killed-net MLP census uses killed-bank means "
                "(e133 census convention). Mode/vector disagreement is "
                "itself reported texture, not smoothed over.",
            "part_b_selection": "the B-ladder was selected by e125a on "
                "pool_sel; e164 reads pool_bar (disjoint prompts AND filler "
                "draws) — but the MLP ranking comes from e133's pool_full "
                "battery, which overlaps pool_bar's prompts (both held-30) "
                "with different filler seeds; MLP cells are therefore "
                "mildly selection-favored relative to head cells. The "
                "pool_full companion column prices that per cell.",
            "single_lineage": "one consolidated lineage and one site-stored "
                "lineage (seeds 1337-chain / 12101); phase conclusions are "
                "line-specific until replication lands.",
        },
        "trims": trims,
        "recipe_deviations": recipe_deviations,
        "ckpt_inventory": {"saved": {},
                           "external_used": [
                               f"runs/checkpoints/{p.name}"
                               for p in (CONS_CK, INST_CK, ARMB_CK)],
                           "note": "eval-only: no checkpoints written"},
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": 2739072,
                   "device": "cpu", "torch_threads": torch.get_num_threads(),
                   "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "postkill_census.png", root_cells, kill_cells, W, W_detail,
         recovery, mlp_reopen, bz_root, bz_kill, BCELLS, verdict_a,
         verdict_b, best_reopen_all)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'postkill_census.png'}")
    log(f"total {time.time() - T0:.1f}s")


# ------------------------------------------------------------------ plot

def plot(path, root_cells, kill_cells, W, W_detail, recovery, mlp_reopen,
         bz_root, bz_kill, bcells, verdict_a, verdict_b, best_reopen_all):
    fig = plt.figure(figsize=(16.5, 12.5))
    gs = fig.add_gridspec(2, 2)

    # ---- (0,0): organ profile root vs killed (normalized costs)
    ax = fig.add_subplot(gs[0, 0])
    tags = [t for t in root_cells if t in kill_cells]
    order = sorted(tags, key=lambda t: -root_cells[t]["drop"])
    xr = np.arange(len(order))
    rv = [max(0.0, root_cells[t]["norm_cost"]) for t in order]
    kv = [max(0.0, kill_cells[t]["norm_cost"]) for t in order]
    ax.bar(xr - 0.2, rv, 0.4, color="steelblue", edgecolor="k", lw=0.4,
           label="root (normalized cost)")
    ax.bar(xr + 0.2, kv, 0.4, color="crimson", edgecolor="k", lw=0.4,
           label="N2-killed (normalized cost)")
    labs = [t.replace("head_", "").replace("wpe_", "w") for t in order]
    ax.set_xticks(xr)
    ax.set_xticklabels(labs, rotation=90, fontsize=5.2)
    ax.set_ylabel("(base - arm) / (base - floor)")
    ax.set_title(f"PART A — organ-dependency profile: root vs N2-killed\n"
                 f"(sorted by root load; killed baseline "
                 f"{bz_kill['mean_pz']:.3f} vs root "
                 f"{bz_root['mean_pz']:.3f})", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, axis="y")

    # ---- (0,1): the W-bars summary
    ax = fig.add_subplot(gs[0, 1])
    w2d = W_detail["W2"]
    w3d = W_detail["W3"]
    w4d = W_detail["W4"]
    rows = [
        ("row0 strength\n(norm)", w2d["root"]["norm_strength"],
         w2d["killed"]["norm_strength"]),
        ("row1 strength\n(norm)",
         max(0.0, min(root_cells["row1_mean"]["drop"],
                      root_cells["row1_zero"]["drop"])),
         max(0.0, min(kill_cells["row1_mean"]["drop"],
                      kill_cells["row1_zero"]["drop"]))),
    ]
    for l in w3d["top2_mlp_by_root_load"]:
        li = int(l.split("_l")[1])
        rows.append((f"{l} cost\n(norm)",
                     root_cells[l]["norm_cost"],
                     kill_cells[l]["norm_cost"]))
    xs = np.arange(len(rows))
    ax.bar(xs - 0.2, [r[1] for r in rows], 0.4, color="steelblue",
           edgecolor="k", lw=0.4, label="root")
    ax.bar(xs + 0.2, [max(0.0, r[2]) for r in rows], 0.4, color="crimson",
           edgecolor="k", lw=0.4, label="N2-killed")
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=7)
    ax.axhline(0.5, ls="--", color="seagreen", lw=1.2,
               label="W2/W3 bar (>=0.5 x root)")
    ax.set_ylabel("normalized cost / strength")
    w4txt = "W4 killed loads: " + ", ".join(
        f"{k}:{v:.3f}" for k, v in w4d["killed_loads"].items())
    ax.set_title("PART A — content probes (the W bars)\n" + w4txt,
                 fontsize=8.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, axis="y")

    # ---- (1,0): recovery sweep
    ax = fig.add_subplot(gs[1, 0])
    th = recovery["third_head"]
    keys = sorted(th, key=lambda k: -th[k]["reopen"])
    ys = [th[k]["reopen"] for k in keys]
    ax.bar(np.arange(len(keys)), ys, 0.7, color="darkorange",
           edgecolor="k", lw=0.3)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(REOPEN_MARGIN, ls="--", color="seagreen",
               lw=1.2, label="re-open bar (+0.05)")
    ax.axhline(bz_root["mean_pz"] - bz_kill["mean_pz"], ls=":", color="gray",
               lw=1.2, label="full recovery (root base)")
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, rotation=90, fontsize=5.2)
    ax.set_ylabel("expression delta vs killed baseline (p(Z))")
    bo = (f"best re-open: {best_reopen_all['tag']} "
          f"{best_reopen_all['reopen']:+.4f}" if best_reopen_all else "n/a")
    ax.set_title(f"PART A — PROBE RECOVERY: one more head mean-replaced on "
                 f"top of the kill\n{bo} | MLP re-opens: "
                 + ", ".join(f"{k}:{v['reopen']:+.3f}"
                             for k, v in sorted(mlp_reopen.items(),
                                                key=lambda kv: -kv[1]
                                                ["reopen"])[:3]),
                 fontsize=8.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, axis="y")

    # ---- (1,1): part B frontier
    ax = fig.add_subplot(gs[1, 1])
    xmax = max(1.3, max((c["ce_cost"] for c in bcells), default=0.8) + 0.25)
    ax.axvspan(0, CE_FLAT, color="seagreen", alpha=0.07)
    ax.fill_between([-0.05, CE_FLAT], KILL_DROP, 118, color="seagreen",
                    alpha=0.18)
    ax.fill_between([CE_WRECK, xmax], KILL_DROP, 118, color="firebrick",
                    alpha=0.10)
    ax.axhline(KILL_DROP, ls="--", color="gray", lw=1.1)
    ax.axvline(CE_FLAT, ls="--", color="seagreen", lw=1.2)
    ax.axvline(CE_WRECK, ls="--", color="firebrick", lw=1.2)
    for c in bcells:
        if c["head_only"]:
            m, col, s = ("^", "tab:blue", 46)
        elif not c["heads"]:
            m, col, s = ("o", "darkorange", 46)
        else:
            m, col, s = ("D", "purple", 46)
        ax.scatter(c["ce_cost"], c["drop_bar_pct"], s=s, color=col,
                   edgecolor="k", lw=0.5, marker=m, zorder=4)
        ax.annotate(c["tag"].split(":")[0], (c["ce_cost"], c["drop_bar_pct"]),
                    textcoords="offset points", xytext=(4, 3), fontsize=5.5)
    ax.scatter([], [], marker="^", color="tab:blue", label="B-heads only")
    ax.scatter([], [], marker="o", color="darkorange", label="MLP only")
    ax.scatter([], [], marker="D", color="purple", label="heads + MLP")
    ax.set_xlabel("CE cost vs arm_b base (nats)")
    ax.set_ylabel("site-fact drop (% pool_bar onset)")
    ax.set_xlim(-0.08, xmax)
    ax.set_ylim(-15, 118)
    ax.set_title("PART B — the MLP third: site-drop vs CE frontier on "
                 "arm_b\n(green region = MLP-INCORRIGIBLE; "
                 "MLP-coordinate kill at any CE = any orange/purple above "
                 "60%)", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.25)

    short_a = verdict_a.split(" (")[0]
    short_b = verdict_b.split(" (")[0]
    fig.suptitle(f"E164 — THE POST-KILL CENSUS\nA: {short_a} | W-bars "
                 f"{W} | B: {short_b}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
