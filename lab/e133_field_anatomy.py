"""E133 — the FIELD ANATOMY CENSUS: where does a memory fact physically
live, before and after consolidation? (REGISTERED; T077/T078/W011 frame.)

WHY (T077's honesty reflex, verbatim): "row 0's necessity is proven, its
sufficiency is not — row 0 may be the readout GATE with content in body
weights; the census's diffuse OOB texture is consistent with a distributed
content store behind a row-0 door." e131 proved necessity (D-all+row0
collapses expression -97%); nobody has made the causal-LOAD map that
separates ROUTE from SUBSTRATE. This experiment is that map: a per-organ
causal ablation sweep on three nets — the graduated (row-0-keyed) net, its
address-phase twin (the same line pre-consolidation), and the site-locked
183-consolidated splice arm as texture.

NETS (on disk, gated):
  * GRADUATED      runs/checkpoints/e131_consolidated_e113.pt
                   (e113 jitter-consolidated, regenerated + gated in e131;
                   std battery install-60 g0 = 0.7850371599197388)
  * ADDRESS TWIN   runs/checkpoints/e119_twin_start.pt
                   (e119's twin start = bit-identical e048_repro install-phase
                   net per e119 metrics ckpt_inventory; g0 = 0.5563086867332458)
  * SITE-LOCKED    runs/checkpoints/e131_arm_b_corpus_spliced.pt
                   (e120 arm (b) verbatim, 183-consolidated, no re-key; std
                   battery g0 = 0.007800613064318895 — FLOOR: its fact lives
                   at the 183 splice geometry, onset p(Z) = 0.9880021214485168
                   per e131 probe-1 — so its PRIMARY battery is the 183-geometry
                   read, e131's probe-1 instrument verbatim).

INSTRUMENTS (reused verbatim, provenance in comments):
  * battery: e131/e113/e120 install-60 g0 primary + held-30 report (host
    occurrences FLORIZEL/ELIZABETH, 130-token preceding contexts; e043
    protocol rebuild, SPLICE_RNG shuffle, install60/held30 split).
  * 183-geometry battery: e120 arm-b pool (held-30 prompts, corpus filler
    seed 12103, fact segments from install windows, splice at col 42 ->
    ZEPHYRA x-cols 184..190, address row 183); score = p(Z) at position 183
    (onset) + mean p(name char) over the 7 name positions (e131 probe-1).
  * wpe deletion: D2 subtractive row-zero with the e065/e113 confinement
    gate (e131's deleted_wpe).
  * CE_R bank: e065 val-windows seed 26502, 60 windows (locality control).
  * organ ablation: zero-hook conventions — MLP/attn block zeroed at the
    block output (e001/e035 lesion map); single head zeroed as its 32-dim
    slice of the c_proj input (e001/e038 head-lesion hook == common.lesion).
  * grown-row caveat (e119): grown rows defined there by dnorm >= 0.04 with
    a drift warning; the address set here stays e113's D-all rows
    {121,125,129,133,137} — the pre-registered set every necessity number
    in this line was measured against.

ORGANS (per net; census = 6 MLP layers + 36 heads + 4 wpe row-sets = 46):
  * per-layer MLP (6 layers — line config is 6x6x192, see deviations),
    ablate = zero OR mean-replace the block's MLP output. MODE PILOT
    (pre-registered): on the graduated net's L3 MLP (fallback: the max
    zero-load MLP layer if L3 drop < 0.05), run BOTH modes, record std
    battery drop AND CE_R damage; select the mode with smaller CE_R damage
    (spare general LM, isolate the fact); tie within 0.02 nats -> ZERO (the
    e018/e035 incumbent). Mean-replace vector = the layer's mean MLP output
    over the CE_R bank (corpus windows, no fact contexts).
  * per-head attention (6x6), ablate = zero the head slice.
  * whole-attn-block per layer (6) — redundancy texture (block drop vs sum
    of its heads' drops), NOT in the attribution denominator.
  * wpe row sets (positive controls): {0} | band {121,125,129,133,137} |
    rand-5 matched (seeded sample from ordinary rows, excludes 0, band,
    183) | {183} (the site address).

CONTROLS: no-ablation baseline; sham ablation (add matched-magnitude
Gaussian noise — std = the organ output's RMS over the CE_R bank — instead
of ablating) on 3 organs per decisive net (top-load MLP, top-load head,
min-load MLP, picked by rank after the main sweep, seeds fixed); rand-5 wpe
set as the wpe sham-equivalent; CE_R per arm (locality); ONE joint arm per
decisive net (top-MLP + top-head + the net's key wpe set) vs the sum of its
parts — the additivity assumption's honesty check.

SCORE / CAUSAL LOAD (precise):
  * primary score per net = std battery install-60 g0 mean p(Z) (graduated,
    twin) or 183-onset p(Z) (site-locked); both batteries run for all nets.
  * drop(arm) = base_primary - arm_primary (signed; negative = ablation
    HELPS expression, e.g. the e115 brake at the old address).
  * attributable load L(organ) = max(0, drop) for the 46 census organs;
    total T = sum(L); class share = sum over class / T. ADDITIVE
    ATTRIBUTION IS AN ASSUMPTION — single-organ drops need not sum to the
    joint; the joint arm quantifies this (report-only).

REGISTERED PREDICTION (coordinator dispatch, VERBATIM — no bar shopping):
  * SUBSTRATE-IN-BODY fires if: graduated net shows >=60% of attributable
    causal load in non-address organs (MLPs + non-address heads), AND the
    address-phase twin shows the mirror (>=60% in address-adjacent organs:
    wpe band rows + heads at the installing layers).
  * ROUTE-ONLY fires if: graduated net's body organs carry <30% and
    row-0/wpe carry the load (row 0 is a gate to... row 0 itself; content
    positional).
  * W005-ALT (address-spreading) is killed if grown-row-adjacent attention
    carries <50% post-graduation.
  * Report the site-locked (183) net's map as texture; no bar.
  * No bar shopping; texture => TEXTURE with numbers.

OPERATIONALIZATIONS (fixed before compute):
  * head ADDRESS classification — attention geometry at the scored position
    of the net's PRIMARY battery, measured BEFORE any ablation (a pure
    forward measurement, blind to ablation results):
      twin:     band-adjacent  = mean attn mass on positions 121..129 >= 0.25
                (the fed part of the grown band in a 130-token window,
                uniform 9/130 = 0.069);
      graduated: address head   = sink-adjacent (mass on position 0 >= 0.25)
                OR band-adjacent (>= 0.25)  — the address system = old band
                + new row-0 hub (W011);
      site-locked: site-adjacent = mean attn mass on positions 183..190 of
                the 256-token site battery >= 0.25 (uniform 8/256 = 0.031).
    "heads at the installing layers" (dispatch wording) is operationalized
    as these geometry-classified heads: the heads that read the address.
  * SUBSTRATE-IN-BODY: share(MLPs + non-address heads | graduated) >= 0.60
    AND share(band5 wpe arm + address heads | twin) >= 0.60.
  * ROUTE-ONLY: share(MLPs + ALL heads | graduated) < 0.30 (conservative:
    every head counts as body for this bar; wpe share is then the rest).
  * W005-ALT killed: share(band-adjacent heads | graduated) < 0.50.
  * Neither fires => TEXTURE with numbers.

COMPUTE ENVELOPE: eval-only, no training. GPU (common.gpu_ok() gate; park to
CPU if it fails — nets are 2.7M params). Batch 30. No concurrent GPU jobs
(e139 is CPU-only; verified idle before launch). Single steps well < 30 min.

Outputs: runs/e133/{metrics.json, field_anatomy.png}.
No NOTES/THINKING/QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python e133_field_anatomy.py     (E133_SMOKE=1 -> shakedown)
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import os

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                           # noqa: E402
import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E133_SMOKE") == "1"

# ---- device gate (GPU allowed and recommended; park to CPU if it fails) ----
GPU = common.gpu_ok() and torch.cuda.is_available()
DEVICE = torch.device("cuda" if GPU else "cpu")
if GPU:
    torch.cuda.init()

from common import Cfg, CharCorpus, TinyGPT, run_dir, save_json   # noqa: E402

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
NETS = {  # tag -> (path, std-g0 ref, std-held ref or None, site-onset ref or None)
    "graduated": (CKPT_DIR / "e131_consolidated_e113.pt",
                  0.7850371599197388, 0.7233286499977112, None),
    "twin": (CKPT_DIR / "e119_twin_start.pt",
             0.5563086867332458, None, None),
    "site_locked": (CKPT_DIR / "e131_arm_b_corpus_spliced.pt",
                    0.007800613064318895, None, 0.9880021214485168),
}
PRIMARY_BATTERY = {"graduated": "std", "twin": "std", "site_locked": "site"}

# splice geometry (e120 verbatim — reused for the site battery)
FACT_PRE, FACT_POST = 12, 12
FACT_LEN = FACT_PRE + len(NAME) + FACT_POST      # 31
SPLICE_AT = 42
Z_XCOL = PRE + SPLICE_AT + FACT_PRE             # 184
SPLICE_ADDR_ROW = Z_XCOL - 1                    # 183
CORP_CONT_SEED = 12103
N_PROMPTS = 8 if SMOKE else 30

# organs
BAND5 = (121, 125, 129, 133, 137)               # e113's D-all set (verbatim)
RAND5_SEED = 13301
ATT_BAR = 0.25                                  # address-adjacency bar
GATE_TOL = 5e-6                                 # e131 bit-reproduction tol
GATE_FALLBACK = 0.05                            # e113 convention
SHAM_BASE_SEED = 13300
CE_EVAL_SEED = 26502                            # e065 CE_R bank seed

REGISTERED_PREDICTION = {
    "substrate_in_body": "graduated >=60% of attributable causal load in "
                         "non-address organs (MLPs + non-address heads), AND "
                         "twin >=60% in address-adjacent organs (wpe band "
                         "rows + heads at the installing layers).",
    "route_only": "graduated net's body organs (MLPs + heads) carry <30% and "
                  "row-0/wpe carry the load.",
    "w005_alt": "W005-ALT (address-spreading) killed if grown-row-adjacent "
                "attention carries <50% post-graduation.",
    "site_locked": "site-locked (183) net's map reported as texture; no bar.",
    "no_bar_shopping": "No bar shopping; texture => TEXTURE with numbers.",
    "operationalizations": "L(o)=max(0, base-primary - arm-primary); T = sum "
                           "over 46 census organs (6 MLP + 36 heads + 4 wpe "
                           "sets); head address label = attention mass at the "
                           "scored position of the primary battery (band "
                           "121..129 >= 0.25 [twin/grad]; sink pos-0 >= 0.25 "
                           "OR band >= 0.25 [grad]; site 183..190 >= 0.25 "
                           "[site]); ROUTE-ONLY body = MLPs + ALL heads.",
}

recipe_deviations: list[str] = [
    "Tasking said 'all 4 layers' / '4 layers x 4 heads'; this line's config "
    "is 6 layers x 6 heads x 192 embd (e131/e119 metrics config) — the census "
    "covers ALL 6 layers and ALL 36 heads. No organ skipped.",
    "Site-locked net: the standard install-60 g0 battery is FLOOR (0.0078, "
    "e131 stored) — its fact lives at the 183 splice geometry (onset p(Z) "
    "0.988). Primary battery for the site-locked net = the 183-geometry "
    "pool-b onset read (e131 probe-1 instrument verbatim); the standard "
    "battery is still run and reported for all nets.",
    "Address-head classification on the standard battery uses fed band rows "
    "121..129 only (rows 130..137 are never fed in a 130-token window — "
    "e116's fed-row note); uniform reference 9/130 = 0.069.",
    "e119 twin_start is bit-identical to e048_repro (its own metrics "
    "ckpt_inventory) — the address-phase twin IS the install-phase net; not "
    "duplicated or regenerated.",
    "Additive attribution (shares of T = sum of single-organ positive drops) "
    "is an assumption; the joint arm (top-MLP + top-head + key wpe set per "
    "net) quantifies interaction and is report-only.",
    "e131 ran CPU; this sweep runs GPU for speed. Gate tolerance 5e-6 "
    "primary (e131 bit-tol) with the 0.05 e113-convention fallback — GPU "
    "float non-identity may force the fallback; whichever fires is recorded.",
    "wpe organs use D2 row-zero (the e131 necessity instrument) rather than "
    "e116's mean-arm, for comparability with every necessity number in this "
    "line; negative drops (ablation helps) are recorded as brake texture, "
    "clamped out of the attribution denominator.",
]


# ------------------------------------------------------------------ instruments

def load_net(path) -> TinyGPT:
    m = TinyGPT(Cfg())
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval().to(DEVICE)
    return m


def wpe_zero_rows(model: TinyGPT, rows: tuple[int, ...]):
    """D2 subtractive row-zero (e065/e113/e131); returns (orig_rows, gate)."""
    w = model.wpe.weight.data
    orig = w[list(rows)].clone()
    w[list(rows)] = 0.0
    gate = {"rows": list(rows),
            "n_elements_changed": int(len(rows) * w.shape[1]),
            "pass": bool(all(torch.equal(w[r], torch.zeros_like(w[r]))
                             for r in rows))}
    return orig, gate


def wpe_restore(model: TinyGPT, rows, orig):
    model.wpe.weight.data[list(rows)] = orig


class Hooks:
    """Organ ablation hooks (e001/e035/e038 conventions, eval-only)."""

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

    def attn_block_zero(self, layer):
        blk = self.model.h[layer]

        def h(m, a, o):
            return torch.zeros_like(o)
        self.handles.append(blk.attn.register_forward_hook(h))

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
def battery_std(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> dict:
    """e131 battery_cell on DEVICE: p(Z) at the last position."""
    net.eval()
    pzs, amax = [], 0.0
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs.append(pr[:, zid])
        amax += float((lg[:, -1].argmax(-1) == zid).float().sum())
    p = torch.cat(pzs)
    return {"mean_pz": float(p.mean()),
            "frac_pz_ge_0.5": float((p >= 0.5).float().mean()),
            "frac_argmax_z": amax / ids.shape[0]}


@torch.no_grad()
def battery_site(net: TinyGPT, pool_x: torch.Tensor, name_ids: torch.Tensor,
                 zid: int, bs=30) -> dict:
    """e131 read_fact_position verbatim logic: p at the 183-geometry onset +
    mean p(true name char) over the 7 name positions."""
    net.eval()
    n_name = len(name_ids)
    onset, per_pos = [], [[] for _ in range(n_name)]
    for i in range(0, pool_x.shape[0], bs):
        w = pool_x[i:i + bs]
        lg, _ = net(w)
        pr = F.softmax(lg, -1)
        for k in range(w.shape[0]):
            onset.append(float(pr[k, SPLICE_ADDR_ROW, int(zid)]))
            for j in range(n_name):
                per_pos[j].append(
                    float(pr[k, SPLICE_ADDR_ROW + j, int(w[k, Z_XCOL + j])]))
    on = torch.tensor(onset)
    allp = torch.tensor([p for pos in per_pos for p in pos])
    return {"onset_pz": float(on.mean()),
            "onset_frac_ge_0.5": float((on >= 0.5).float().mean()),
            "pname_mean_over7": float(allp.mean())}


@torch.no_grad()
def ce_fixed(net: TinyGPT, x, y, bs=30) -> float:
    net.eval()
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        _, loss = net(x[i:i + bs], y[i:i + bs])
        tot += float(loss.item()) * len(x[i:i + bs])
        n += len(x[i:i + bs])
    return tot / max(n, 1)


@torch.no_grad()
def organ_stats(net: TinyGPT, bank_x, which: str, layer: int, head=None,
                bs=30):
    """RMS of an organ's output over the CE_R bank (sham magnitude + MLP
    mean-replace vector). which in {'mlp','head'}; for 'mlp' also returns the
    mean output vector."""
    outs = []
    blk = net.h[layer]
    handle = None
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
def head_attention_geometry(net: TinyGPT, ids: torch.Tensor, pos: int,
                            band_rows=(121, 122, 123, 124, 125, 126, 127, 128,
                                       129),
                            site_rows=tuple(range(183, 191)), bs=30) -> dict:
    """Per-head attention mass at the scored position `pos`, BEFORE any
    ablation (pure forward measurement; softmax(qk) replicated manually —
    measurement only, never gates bit-reproduction)."""
    net.eval()
    cfg = net.cfg
    L, H, D = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head
    T0 = int(ids.shape[1])
    band_ok = [r for r in band_rows if r < T0]     # std battery: only rows
    site_ok = [r for r in site_rows if r < T0]     # 121..129 of the band are
    band_ix = torch.tensor(band_ok, device=ids.device)   # fed (e116 note);
    site_ix = torch.tensor(site_ok, device=ids.device)   # site 183+ unseen @130
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


# ------------------------------------------------------------------ main

def parse_organ(t: str) -> tuple[int, int | None]:
    """'mlp_l0' / 'attnblock_l2' / 'head_l3h5' -> (layer, head or None)."""
    p = t.split("_")[1]
    i = p.find("h")
    if i < 0:
        return int(p[1:]), None
    return int(p[1:i]), int(p[i + 1:])


def organ_geo_key(t: str) -> str:
    """'head_l3h5' -> 'L3H5' (geo_class key)."""
    l, h = parse_organ(t)
    return f"L{l}H{h}"


def main():
    rd = run_dir("e133_smoke" if SMOKE else "e133")
    log(f"E133 FIELD ANATOMY CENSUS (smoke={SMOKE}) -> {rd}")
    log(f"compute: {'GPU' if GPU else 'CPU (parked — gpu gate failed)'} "
        f"{common.gpu_status()}")
    if GPU and not common.gpu_ok():
        raise RuntimeError("GPU gate failed mid-flight — rerun parked")

    # ---------------- protocol rebuild (e131 verbatim) ----------------
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
    log(f"protocol rebuilt: install60 {mix}, held30 (e131 verbatim)")

    name_ids = corpus.encode(NAME)
    bat = {}
    for tag, occ in (("install60", install_occ), ("held30", held_occ)):
        cs = [train_text[p - PRE: p] for p, _ in occ]
        bat[tag] = torch.stack([corpus.encode(c) for c in cs]).to(DEVICE)
    # g0 only — the primary geometry (e131 f_eval convention)

    # CE_R bank (e065 verbatim, seed 26502)
    g = torch.Generator().manual_seed(CE_EVAL_SEED)
    rx, ry = [], []
    tries = 0
    while len(rx) < 60 and tries < 500 * 60:
        i = int(torch.randint(len(val_ids) - BLOCK - 1, (1,), generator=g))
        txt = val_text[i: i + BLOCK + 1]
        tries += 1
        if "ZEPHYRA" in txt or "ZEPH" in txt:
            continue
        rx.append(val_ids[i: i + BLOCK])
        ry.append(val_ids[i + 1: i + 1 + BLOCK])
    r_eval_x = torch.stack(rx).to(DEVICE)
    r_eval_y = torch.stack(ry).to(DEVICE)

    # site battery pool (e120 arm-b verbatim construction, corpus filler)
    prompts = [train_text[p - PRE: p] for p, _ in held_occ][:N_PROMPTS]
    prompt_ids = torch.stack([corpus.encode(c) for c in prompts])
    gc = torch.Generator().manual_seed(CORP_CONT_SEED)
    src = torch.randint(len(train_ids) - (BLOCK - PRE) - 1,
                        (4 if not SMOKE else 1, len(prompts)), generator=gc)
    filler = torch.stack([train_ids[s: s + BLOCK - PRE] for s in src.flatten()])
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
    pool_b = torch.cat([torch.stack(
        [prompt_ids[k % prompt_ids.shape[0]] for k in range(filler.shape[0])]),
        cont], 1).to(DEVICE)
    assert all(torch.equal(w[Z_XCOL:Z_XCOL + len(NAME)].cpu(), name_ids)
               for w in pool_b)
    log(f"site battery: {tuple(pool_b.shape)}, ZEPHYRA at x-col {Z_XCOL} "
        f"(address row {SPLICE_ADDR_ROW}) in all windows")

    # rand-5 matched control rows (seeded; excludes 0, band 121..137, 183)
    ordinary = [r for r in range(256)
                if r != 0 and r not in range(121, 138) and r != 183]
    gr = torch.Generator().manual_seed(RAND5_SEED)
    rand5 = tuple(sorted(ordinary[i] for i in
                         torch.randperm(len(ordinary), generator=gr)[:5]
                         .tolist()))
    log(f"rand-5 matched control rows: {rand5}")

    # ---------------- per-net sweep ----------------
    results: dict = {}
    gates: dict = {}
    mlp_mode = "zero"          # pilot may switch to "mean"
    pilot = {}

    def score_all(net) -> dict:
        return {
            "std_install60": battery_std(net, bat["install60"], zid),
            "std_held30": battery_std(net, bat["held30"], zid),
            "site": battery_site(net, pool_b, name_ids, zid),
            "ce_r": ce_fixed(net, r_eval_x, r_eval_y),
        }

    def primary(sc: dict, net_tag: str) -> float:
        return sc["std_install60"]["mean_pz"] if PRIMARY_BATTERY[net_tag] == "std" \
            else sc["site"]["onset_pz"]

    # ---- phase 0: mode pilot on the graduated net (pre-registered) -------
    net_g = load_net(NETS["graduated"][0])
    base_g = score_all(net_g)
    ref_g = NETS["graduated"][1]
    gates["graduated"] = {
        "std_g0": base_g["std_install60"]["mean_pz"], "ref": ref_g,
        "tol": GATE_TOL, "fallback_tol": GATE_FALLBACK,
        "bit": bool(abs(base_g["std_install60"]["mean_pz"] - ref_g) < GATE_TOL),
        "pass": bool(abs(base_g["std_install60"]["mean_pz"] - ref_g) < GATE_FALLBACK)}
    log(f"GATE graduated: g0 {base_g['std_install60']['mean_pz']:.10f} vs ref "
        f"{ref_g:.10f} -> {'PASS' if gates['graduated']['pass'] else 'FAIL'}"
        f"{' (bit)' if gates['graduated']['bit'] else ''}")
    if not gates["graduated"]["pass"]:
        raise RuntimeError("graduated checkpoint failed its gate")

    stats_bank = organ_stats(net_g, r_eval_x, "mlp", 3)
    pilot_layer = 3

    def mlp_arm(net, layer, mode, mean_vec=None):
        hk = Hooks(net)
        if mode == "zero":
            hk._mlp_zero(layer)
        else:
            hk._mlp_mean(layer, mean_vec)
        return hk

    sc_zero = None
    hk = mlp_arm(net_g, pilot_layer, "zero")
    sc_zero = score_all(net_g)
    hk.remove()
    hk = mlp_arm(net_g, pilot_layer, "mean", stats_bank["mean_vec"].to(DEVICE))
    sc_mean = score_all(net_g)
    hk.remove()
    pilot = {
        "layer": pilot_layer,
        "zero": {"drop_std": base_g["std_install60"]["mean_pz"]
                 - sc_zero["std_install60"]["mean_pz"],
                 "ce_r": sc_zero["ce_r"]},
        "mean": {"drop_std": base_g["std_install60"]["mean_pz"]
                 - sc_mean["std_install60"]["mean_pz"],
                 "ce_r": sc_mean["ce_r"]},
        "note": "pre-registered selection: smaller CE_R damage wins; tie "
                "<=0.02 nats -> zero (e018/e035 incumbent)",
    }
    dz, dm = pilot["zero"], pilot["mean"]
    if dz["drop_std"] < 0.05:
        recipe_deviations.append(
            f"pilot fallback fired: L3 zero-drop {dz['drop_std']:.4f} < 0.05 "
            f"— pilot re-run on the max zero-load MLP layer after the sweep "
            f"(selection rule unchanged)")
    if abs(dz["ce_r"] - dm["ce_r"]) <= 0.02:
        mlp_mode = "zero"
    else:
        mlp_mode = "zero" if dz["ce_r"] < dm["ce_r"] else "mean"
    pilot["selected_mode"] = mlp_mode
    log(f"PILOT L{pilot_layer} MLP: zero drop {dz['drop_std']:.4f} CE_R "
        f"{dz['ce_r']:.4f} | mean drop {dm['drop_std']:.4f} CE_R "
        f"{dm['ce_r']:.4f} -> mode {mlp_mode}")
    del net_g, sc_zero, sc_mean

    # ---- phase 1: the sweep on each net ----------------
    for net_tag, (path, ref_std, ref_held, ref_site) in NETS.items():
        net = load_net(path)
        base = score_all(net)
        gt = {"std_g0": base["std_install60"]["mean_pz"], "ref": ref_std,
              "tol": GATE_TOL, "fallback_tol": GATE_FALLBACK,
              "pass": bool(abs(base["std_install60"]["mean_pz"] - ref_std)
                           < GATE_FALLBACK)}
        if net_tag in gates:      # graduated already gated (pre-pilot)
            gt = gates[net_tag]
        if ref_site is not None:
            gt["site_onset"] = base["site"]["onset_pz"]
            gt["site_ref"] = ref_site
            gt["site_pass"] = bool(abs(base["site"]["onset_pz"] - ref_site)
                                   < GATE_FALLBACK)
        gates[net_tag] = gt
        log(f"NET {net_tag}: base g0 {base['std_install60']['mean_pz']:.6f} "
            f"| site onset {base['site']['onset_pz']:.6f} | CE_R "
            f"{base['ce_r']:.4f} | gate "
            f"{'PASS' if gt['pass'] else 'FAIL'}")
        if not gt["pass"] or (ref_site is not None and not gt["site_pass"]):
            raise RuntimeError(f"{net_tag} failed its gate")

        # attention geometry (pre-ablation, primary battery)
        geo_ids = bat["install60"] if PRIMARY_BATTERY[net_tag] == "std" else pool_b
        geo_pos = 129 if PRIMARY_BATTERY[net_tag] == "std" else SPLICE_ADDR_ROW
        geo = head_attention_geometry(net, geo_ids, geo_pos)
        geo_class = {}
        for k, v in geo.items():
            band_adj = v["band"] >= ATT_BAR
            sink_adj = v["sink"] >= ATT_BAR
            site_adj = v["site"] >= ATT_BAR
            if net_tag == "twin":
                addr = band_adj
            elif net_tag == "graduated":
                addr = sink_adj or band_adj
            else:
                addr = site_adj
            geo_class[k] = {"sink_mass": v["sink"], "band_mass": v["band"],
                            "site_mass": v["site"], "band_adjacent": band_adj,
                            "sink_adjacent": sink_adj,
                            "site_adjacent": site_adj, "address_head": addr}
        n_addr = sum(1 for v in geo_class.values() if v["address_head"])
        n_band = sum(1 for v in geo_class.values() if v["band_adjacent"])
        log(f"  head geometry: {n_addr} address heads "
            f"({n_band} band-adjacent) at bar {ATT_BAR}")

        # ---- organ arms ----
        arms: dict = {}

        def run_arm(tag, apply=None, wpe_rows=None):
            hk = Hooks(net) if apply else None
            orig = None
            if wpe_rows is not None:
                orig, _ = wpe_zero_rows(net, wpe_rows)
            if apply:
                apply(hk)
            sc = score_all(net)
            if hk:
                hk.remove()
            if wpe_rows is not None:
                wpe_restore(net, wpe_rows, orig)
            arms[tag] = {"std_install60": sc["std_install60"]["mean_pz"],
                         "std_held30": sc["std_held30"]["mean_pz"],
                         "site_onset": sc["site"]["onset_pz"],
                         "site_pname": sc["site"]["pname_mean_over7"],
                         "ce_r": sc["ce_r"]}
            return arms[tag]

        # MLP layers (selected mode)
        for l in range(net.cfg.n_layer):
            if mlp_mode == "zero":
                run_arm(f"mlp_l{l}", apply=lambda hk, l=l: hk._mlp_zero(l))
            else:
                st = organ_stats(net, r_eval_x, "mlp", l)
                run_arm(f"mlp_l{l}",
                        apply=lambda hk, l=l, m=st["mean_vec"].to(DEVICE):
                        hk._mlp_mean(l, m))
        # whole-attn blocks (zero; redundancy texture — NOT in denominator)
        for l in range(net.cfg.n_layer):
            run_arm(f"attnblock_l{l}",
                    apply=lambda hk, l=l: hk.attn_block_zero(l))
        # per-head
        for l in range(net.cfg.n_layer):
            for h in range(net.cfg.n_head):
                run_arm(f"head_l{l}h{h}",
                        apply=lambda hk, l=l, h=h: hk.head_zero(l, h))
        # wpe row sets
        run_arm("wpe_r0", wpe_rows=(0,))
        run_arm("wpe_band5", wpe_rows=BAND5)
        run_arm("wpe_rand5", wpe_rows=rand5)
        run_arm("wpe_183", wpe_rows=(183,))

        # ---- attribution (primary battery) ----
        head_key = organ_geo_key

        base_p = primary(base, net_tag)
        pkey = "std_install60" if PRIMARY_BATTERY[net_tag] == "std" \
            else "site_onset"
        cen = {t: base_p - v[pkey] for t, v in arms.items()}
        census_tags = [t for t in cen if t.startswith("mlp_")
                       or t.startswith("head_") or t.startswith("wpe_")]
        load = {t: max(0.0, cen[t]) for t in census_tags}
        T = sum(load.values())

        def share(pred):
            return sum(v for t, v in load.items() if pred(t)) / T if T > 0 else 0.0

        mlp_share = share(lambda t: t.startswith("mlp_"))
        head_addr_share = share(
            lambda t: t.startswith("head_") and geo_class[head_key(t)]["address_head"])
        head_non_share = share(
            lambda t: t.startswith("head_")
            and not geo_class[head_key(t)]["address_head"])
        head_all_share = share(lambda t: t.startswith("head_"))
        wpe_r0_share = share(lambda t: t == "wpe_r0")
        wpe_band5_share = share(lambda t: t == "wpe_band5")
        wpe_rand5_share = share(lambda t: t == "wpe_rand5")
        wpe_183_share = share(lambda t: t == "wpe_183")
        band_head_share = share(
            lambda t: t.startswith("head_")
            and geo_class[head_key(t)]["band_adjacent"])

        # sham organs picked by rank on `load`
        mlp_ts = sorted((t for t in load if t.startswith("mlp_")),
                        key=lambda t: load[t])
        head_ts = sorted((t for t in load if t.startswith("head_")),
                         key=lambda t: load[t])
        sham_organs = [mlp_ts[-1], head_ts[-1], mlp_ts[0]]
        sham = {}
        for si, t in enumerate(sham_organs):
            l, h = parse_organ(t)
            if h is None:
                st = organ_stats(net, r_eval_x, "mlp", l)
                hk = Hooks(net)
                hk.mlp_noise(l, st["rms"], SHAM_BASE_SEED + si)
                sc = score_all(net)
                hk.remove()
            else:
                st = organ_stats(net, r_eval_x, "head", l, head=h)
                hk = Hooks(net)
                hk.head_noise(l, h, st["rms"], SHAM_BASE_SEED + si)
                sc = score_all(net)
                hk.remove()
            sham[t] = {"rms": st["rms"], "std_g0": sc["std_install60"]["mean_pz"],
                       "site_onset": sc["site"]["onset_pz"],
                       "ce_r": sc["ce_r"]}
            log(f"  SHAM {t} (rms {st['rms']:.4f}): g0 "
                f"{sc['std_install60']['mean_pz']:.4f} vs ablated "
                f"{arms[t]['std_install60']:.4f} vs base "
                f"{base['std_install60']['mean_pz']:.4f}")

        # joint arm (additivity honesty check): top-MLP + top-head + key wpe
        key_rows = {"graduated": (0,), "twin": BAND5,
                    "site_locked": (183,)}[net_tag]
        top_mlp, top_head = mlp_ts[-1], head_ts[-1]
        ml, _ = parse_organ(top_mlp)
        hl, hh = parse_organ(top_head)
        hk = Hooks(net)
        if mlp_mode == "zero":
            hk._mlp_zero(ml)
        else:
            st = organ_stats(net, r_eval_x, "mlp", ml)
            hk._mlp_mean(ml, st["mean_vec"].to(DEVICE))
        hk.head_zero(hl, hh)
        orig, _ = wpe_zero_rows(net, key_rows)
        sc = score_all(net)
        hk.remove()
        wpe_restore(net, key_rows, orig)
        joint_parts = [load[top_mlp], load[top_head], load["wpe_"
                       + ("r0" if key_rows == (0,) else
                          "band5" if key_rows == BAND5 else "183")]]
        joint = {"parts": [top_mlp, top_head,
                           "wpe_" + ("r0" if key_rows == (0,) else
                                     "band5" if key_rows == BAND5 else "183")],
                 "sum_parts": float(sum(joint_parts)),
                 "joint_drop": float(base_p - (sc["std_install60"]["mean_pz"]
                                               if PRIMARY_BATTERY[net_tag] == "std"
                                               else sc["site"]["onset_pz"])),
                 "ce_r": sc["ce_r"]}
        log(f"  JOINT {top_mlp}+{top_head}+wpe{key_rows}: drop "
            f"{joint['joint_drop']:.4f} vs sum(parts) "
            f"{joint['sum_parts']:.4f}")

        results[net_tag] = {
            "ckpt": f"runs/checkpoints/{path.name}",
            "primary_battery": PRIMARY_BATTERY[net_tag],
            "base": {"std_install60": base["std_install60"],
                     "std_held30": base["std_held30"],
                     "site": base["site"], "ce_r": base["ce_r"]},
            "geo": geo_class,
            "arms": arms,
            "drops_primary": cen,
            "loads": load,
            "total_attributable_T": T,
            "shares": {
                "mlp": mlp_share, "heads_address": head_addr_share,
                "heads_nonaddress": head_non_share, "heads_all": head_all_share,
                "wpe_r0": wpe_r0_share, "wpe_band5": wpe_band5_share,
                "wpe_rand5": wpe_rand5_share, "wpe_183": wpe_183_share,
                "band_adjacent_heads": band_head_share,
            },
            "per_layer": {
                f"mlp_l{l}": load[f"mlp_l{l}"] for l in range(net.cfg.n_layer)},
            "attn_block_drops": {f"attnblock_l{l}":
                                 cen[f"attnblock_l{l}"]
                                 for l in range(net.cfg.n_layer)},
            "sham": sham,
            "joint": joint,
        }
        del net

    # pilot fallback (if L3 was null): re-run both modes on max-load layer
    if pilot["zero"]["drop_std"] < 0.05:
        net_g = load_net(NETS["graduated"][0])
        top = max((t for t in results["graduated"]["loads"]
                   if t.startswith("mlp_")),
                  key=lambda t: results["graduated"]["loads"][t])
        l, _ = parse_organ(top)
        hk = Hooks(net_g); hk._mlp_zero(l)
        sz = score_all(net_g); hk.remove()
        st = organ_stats(net_g, r_eval_x, "mlp", l)
        hk = Hooks(net_g); hk._mlp_mean(l, st["mean_vec"].to(DEVICE))
        sm = score_all(net_g); hk.remove()
        pilot["fallback_layer"] = l
        pilot["zero_fb"] = {"drop_std": base_g["std_install60"]["mean_pz"]
                            - sz["std_install60"]["mean_pz"],
                            "ce_r": sz["ce_r"]}
        pilot["mean_fb"] = {"drop_std": base_g["std_install60"]["mean_pz"]
                            - sm["std_install60"]["mean_pz"],
                            "ce_r": sm["ce_r"]}
        dz, dm = pilot["zero_fb"], pilot["mean_fb"]
        mlp_mode_fb = "zero" if (abs(dz["ce_r"] - dm["ce_r"]) <= 0.02
                                 or dz["ce_r"] < dm["ce_r"]) else "mean"
        pilot["selected_mode_fb"] = mlp_mode_fb
        if mlp_mode_fb != mlp_mode:
            recipe_deviations.append(
                f"pilot fallback on L{l} selected {mlp_mode_fb} != {mlp_mode} "
                f"(L3) — recorded; sweep ran with {mlp_mode} (L3 decision). "
                f"Mode disagreement is itself reported texture.")
        del net_g

    # ---------------- adjudication (registered) ----------------
    sg = results["graduated"]["shares"]
    st_t = results["twin"]["shares"]
    body_nonaddr = sg["mlp"] + sg["heads_nonaddress"]
    body_all = sg["mlp"] + sg["heads_all"]
    twin_addr = st_t["wpe_band5"] + st_t["heads_address"]
    cond_sub_1 = body_nonaddr >= 0.60
    cond_sub_2 = twin_addr >= 0.60
    cond_route = body_all < 0.30
    w005_killed = sg["band_adjacent_heads"] < 0.50
    if cond_sub_1 and cond_sub_2:
        verdict = "SUBSTRATE-IN-BODY"
    elif cond_route:
        verdict = "ROUTE-ONLY"
    else:
        verdict = "TEXTURE"
    adjudication = {
        "substrate_body_share_graduated_nonaddress": body_nonaddr,
        "substrate_bar_60pct": cond_sub_1,
        "twin_address_share_band_plus_address_heads": twin_addr,
        "twin_bar_60pct": cond_sub_2,
        "route_body_share_graduated_allheads": body_all,
        "route_bar_lt_30pct": cond_route,
        "w005_alt_band_adjacent_head_share_graduated": sg["band_adjacent_heads"],
        "w005_alt_killed_lt_50pct": w005_killed,
        "verdict": verdict,
        "site_locked_note": "texture only; no bar (dispatch)",
    }
    log("=" * 78)
    log(f"E133 VERDICT: {verdict}")
    log(f"  graduated body(non-address) share {body_nonaddr:.3f} "
        f"(bar >=0.60: {cond_sub_1}) | body(all heads) {body_all:.3f} "
        f"(bar <0.30: {cond_route})")
    log(f"  twin address share {twin_addr:.3f} (bar >=0.60: {cond_sub_2})")
    log(f"  band-adjacent-head share (graduated) "
        f"{sg['band_adjacent_heads']:.3f} -> W005-ALT "
        f"{'KILLED' if w005_killed else 'alive'}")
    log("=" * 78)

    # ---- report-only sensitivities (NOT gating; post-smoke, labeled) ----
    # (i) locality-filtered shares: exclude arms whose CE_R damage says
    #     "general LM wreckage" (drop_primary / ce_r_damage ratio).
    # (ii) attention-bar sensitivity: address-head shares under stricter bars.
    sens = {}
    for tag, r in results.items():
        base_ce = r["base"]["ce_r"]
        spec = {}
        for t, v in r["arms"].items():
            ce_d = v["ce_r"] - base_ce
            drop_p = r["drops_primary"].get(t)
            spec[t] = {
                "ce_damage": ce_d,
                "fact_specificity": (drop_p / ce_d) if (ce_d is not None
                                                        and drop_p is not None
                                                        and ce_d > 1e-3) else
                                    (None if drop_p is None else
                                     float("inf") if drop_p > 0 else 0.0)}
        r["fact_specificity_per_arm"] = spec
        # locality-filtered attribution: keep only arms with ce_damage <= 0.75
        keep = {t: v for t, v in r["loads"].items()
                if r["arms"][t]["ce_r"] - base_ce <= 0.75}
        Tk = sum(keep.values())
        if Tk > 0:
            mlp_k = sum(v for t, v in keep.items() if t.startswith("mlp_"))
            hh_k = sum(v for t, v in keep.items() if t.startswith("head_"))
            wpe_k = sum(v for t, v in keep.items() if t.startswith("wpe_"))
            sens[tag] = {"ce_filter": 0.75, "T_filtered": Tk,
                         "mlp_share": mlp_k / Tk, "heads_share": hh_k / Tk,
                         "wpe_share": wpe_k / Tk}
    # bar sensitivity (recompute address-head shares at 0.40 / 0.50)
    bar_sens = {}
    for tag, r in results.items():
        geo = r["geo"]
        for bar in (0.40, 0.50):
            if tag == "twin":
                addr = {k: v["band_mass"] >= bar for k, v in geo.items()}
            elif tag == "graduated":
                addr = {k: (v["sink_mass"] >= bar or v["band_mass"] >= bar)
                        for k, v in geo.items()}
            else:
                addr = {k: v["site_mass"] >= bar for k, v in geo.items()}
            n = sum(addr.values())
            sh = sum(v for t, v in r["loads"].items()
                     if t.startswith("head_")
                     and addr[organ_geo_key(t)]) \
                / r["total_attributable_T"] if r["total_attributable_T"] > 0 else 0.0
            bar_sens[f"{tag}@{bar}"] = {"n_address_heads": n,
                                        "heads_address_share": sh}
    metrics_sens = {"locality_filtered": sens,
                    "attention_bar_sensitivity": bar_sens,
                    "note": "REPORT-ONLY, post-smoke sensitivity; registered "
                            "bars use ATT_BAR=0.25 and unfiltered shares "
                            "exactly as pre-registered in the docstring."}

    # ---------------- outputs ----------------
    metrics = {
        "experiment": "e133_field_anatomy",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": ("T077/T078/W011 frame; coordinator dispatch. "
                         "Docstring + bars written before compute."),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("where does the memory fact physically live, before and "
                     "after consolidation — body organs (substrate) or the "
                     "positional route (row 0 / wpe)?"),
        "compute": {"device": "gpu" if GPU else "cpu",
                    "gpu_status_at_launch": common.gpu_status(),
                    "eval_only": True},
        "gates": gates,
        "sensitivities_report_only": metrics_sens,
        "mlp_mode_pilot": pilot,
        "mlp_mode_used": mlp_mode,
        "rand5_rows": list(rand5),
        "nets": results,
        "adjudication": adjudication,
        "recipe_deviations": recipe_deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": 2739072,
                   "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "field_anatomy.png", results, adjudication, mlp_mode, rand5)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'field_anatomy.png'}")
    log(f"total {time.time() - T0:.1f}s")


# ------------------------------------------------------------------ plot

def plot(path, results, adjudication, mlp_mode, rand5):
    tags = ["graduated", "twin", "site_locked"]
    titles = {"graduated": "GRADUATED (e131 consolidated, row-0-keyed)",
              "twin": "ADDRESS-PHASE TWIN (e119 twin_start = install)",
              "site_locked": "SITE-LOCKED (e120 arm b, 183)"}
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.6))

    for c, tag in enumerate(tags):
        r = results[tag]
        sh = r["shares"]
        # row 1: class-share bars
        ax = axes[0, c]
        cls = ["MLP\n(all)", "HEADS\naddress", "HEADS\nnon-addr",
               "wpe\nrow0", "wpe\nband5", "wpe\nrand5", "wpe\n183"]
        vals = [sh["mlp"], sh["heads_address"], sh["heads_nonaddress"],
                sh["wpe_r0"], sh["wpe_band5"], sh["wpe_rand5"], sh["wpe_183"]]
        cols = ["darkorange", "crimson", "indianred",
                "tab:blue", "steelblue", "lightgray", "slateblue"]
        ax.bar(range(len(cls)), [v * 100 for v in vals], color=cols,
               edgecolor="k", lw=0.5)
        for i, v in enumerate(vals):
            ax.text(i, v * 100 + 1.2, f"{v * 100:.1f}%", ha="center",
                    fontsize=7.5)
        ax.set_xticks(range(len(cls)))
        ax.set_xticklabels(cls, fontsize=6.6)
        ax.set_ylabel("% of attributable causal load" if c == 0 else "")
        base_p = r["base"]["std_install60"]["mean_pz"] \
            if r["primary_battery"] == "std" else r["base"]["site"]["onset_pz"]
        ax.set_title(f"{titles[tag]}\nbase p={base_p:.3f} | T="
                     f"{r['total_attributable_T']:.3f}", fontsize=9)
        ax.set_ylim(0, max(105, max(vals) * 110))

        # row 2: per-layer detail + wpe arms
        ax = axes[1, c]
        L = 6
        xs = np.arange(L)
        w = 0.38
        mlp = [r["per_layer"][f"mlp_l{l}"] for l in range(L)]
        blk = [max(0.0, r["attn_block_drops"][f"attnblock_l{l}"]) for l in range(L)]
        hsum = [sum(v for t, v in r["loads"].items()
                    if t.startswith(f"head_l{l}")) for l in range(L)]
        ax.bar(xs - w / 2, mlp, w, color="darkorange",
               label=f"MLP load ({mlp_mode})", edgecolor="k", lw=0.4)
        ax.bar(xs + w / 2, blk, w, color="crimson", alpha=0.85,
               label="attn block (zero, texture)", edgecolor="k", lw=0.4)
        ax.plot(xs, hsum, "D", ms=5, color="k",
                label="sum of 6 heads (attribution)")
        wx = np.arange(L, L + 4)
        wv = [r["loads"]["wpe_r0"], r["loads"]["wpe_band5"],
              r["loads"]["wpe_rand5"], r["loads"]["wpe_183"]]
        ax.bar(wx, wv, 0.55, color=["tab:blue", "steelblue", "lightgray",
                                    "slateblue"],
               edgecolor="k", lw=0.5)
        ax.set_xticks(list(xs) + list(wx))
        ax.set_xticklabels([f"L{l}" for l in range(L)]
                           + ["row0", "band5", f"rand5\n{rand5[0]}+4", "183"],
                           fontsize=7)
        ax.set_ylabel("causal load (p(Z) points)" if c == 0 else "")
        ax.legend(fontsize=6.5, loc="upper right")
        ax.set_title(f"per-organ detail — primary battery: "
                     f"{r['primary_battery']}", fontsize=9)

    a = adjudication
    txt = (f"VERDICT: {a['verdict']}\n"
           f"graduated body(non-addr) {a['substrate_body_share_graduated_nonaddress']:.3f} "
           f"(>=0.60: {a['substrate_bar_60pct']}) | "
           f"body(all heads) {a['route_body_share_graduated_allheads']:.3f} "
           f"(<0.30: {a['route_bar_lt_30pct']})\n"
           f"twin address share {a['twin_address_share_band_plus_address_heads']:.3f} "
           f"(>=0.60: {a['twin_bar_60pct']}) | W005-ALT band-head share "
           f"{a['w005_alt_band_adjacent_head_share_graduated']:.3f} "
           f"(killed<0.50: {a['w005_alt_killed_lt_50pct']})")
    fig.suptitle("E133 — field anatomy census: where the fact lives "
                 f"(MLP mode: {mlp_mode})\n{txt}", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
