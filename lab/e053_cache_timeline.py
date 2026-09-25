"""E053 — the CACHE UTILITY TIMELINE (frontier candidate A / T025, Phase 1).

Design: scratch/e053_design.md (REGISTERED before this run — follow exactly).
Question: when does a cached K/V entry become dead weight during generation?
Per-position CAUSAL utility over cache age — the curve nobody has published
at any scale (sinks are known; the age timeline is not).

Protocol (Phase 1, 100% CPU, G4):
- Cells: scale axis {e005s_small 0.84M / e001 2.7M @4000 / e005s_large 10M}
  x exposure axis {e048_direct400 / e048_direct800 / e001@4000} (2.7M arch;
  lineage caveat registered in the design: different runs, same arch family).
- 8 fixed val prompts (64 tok, seed 202); free-run generation to T=256 with
  FIXED anchor (no sliding window; position identity constant — matches
  non-windowed KV-cache semantics; wpe=256 hard limit). Sampling T=0.8,
  top-k 40, seed 7 (re-init per cell -> paired across cells).
- Interventions, exact, in a manual-attention forward (hook-free):
  V-zero(pos)  = wipe cached content at all layers        [primary]
  K-drop(pos)  = softmax column -inf, DIAGONAL PRESERVED   [secondary, G3]
  both         = entry deletion (equivalence: both ~= K-drop)
- Per step: binned dCE of the actually-sampled token, bins
  {p0, 1-16, 17-64, 65-128, 129-192, 193-255}.
- Final step (t=255): full per-position sweep (V-zero + K-drop), per-layer
  V-zero decomposition, binned {vzero, kdrop, both}.
- Static teacher-forced control: same bins on 64 real-text windows.
- Derived: onset age a* (youngest cache age with mean per-position dCE <
  0.01 nats; naive + 5-pt-sustained robust), junk fraction (dCE <= -0.01:
  lesion HELPS), primacy/recency structure.

Registered predictions (from the design, REVISED after the CPU probes):
- P1 shape: per-position utility at T=256 monotone in recency + weak primacy
  bump (bump <= 25% of recent plateau); sink dCE < 0.05 at age>100 in >=3
  models (sink DEAD for generation). Alt kept alive: sink dCE >= 0.3.
- P2 onset: a* as a fraction of context is scale-INVARIANT (within +-20%)
  but grows with exposure (400 -> 4000 steps) by >= 1.5x in tokens.
- P3 sink accumulator, three-way: sink V-zero dCE trajectory over generation
  GROW / FLAT (|change| < 10% of initial) / DECAY (< 20% of initial by
  mid-generation). Micro-preview (n=1) said DECAY with sign flip.
- P4 junk: among positions older than a*, >= 5% with dCE <= -0.01 (dilution)
  else old entries are dead-but-harmless (pure evictability).

Gates: G0 manual == SDPA (max prob dev < 1e-4); G1 ckpt val CE within known
anchors (small 1.5581+-0.03, e001 1.6224+-0.03); G2 newest-bin V-zero dCE >
+0.5 whenever that bin has >= 32 filled positions; G3 K-drop diagonal -> no
NaN (any NaN skipped+counted); G4 no GPU (enforced via CUDA_VISIBLE_DEVICES
before torch import), no training, no new automations.

Run:   python lab/e053_cache_timeline.py
Smoke: E053_SMOKE=1 python lab/e053_cache_timeline.py  (2 seqs, quick)
Outputs: runs/e053/metrics.json + runs/e053/e053_utility_timeline.png
Budget <= 15 min CPU. No NOTES/THINKING/QUEUE/STATE edits here; no commit.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # G4: Phase 1 is 100% CPU (set pre-torch)

import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import REPO, Cfg, CharCorpus, TinyGPT, estimate_loss, run_dir, save_json

# ------------------------------------------------------------------ constants
SMOKE = os.environ.get("E053_SMOKE") == "1"
N_SEQ = 2 if SMOKE else 8
PROMPT_TOK = 64
T_TOTAL = 256                     # wpe=256 hard limit (design §1)
TEMP, TOPK = 0.8, 40
SEED_PROMPT, SEED_SAMPLE, SEED_STATIC = 202, 7, 203
N_STATIC = 8 if SMOKE else 64
BINS = [(0, 0), (1, 16), (17, 64), (65, 128), (129, 192), (193, 255)]
BIN_NAMES = ["p0", "1-16", "17-64", "65-128", "129-192", "193-255"]
THRESH = 0.01                     # dead-weight threshold (nats)
A_STAR_K = 5                       # sustained window for robust onset
BUDGET_S = 840.0                   # soft guard (14 min)
THREADS = 8                        # leave CPU headroom for e050's CPU assay
torch.set_num_threads(THREADS)

CK = REPO / "runs" / "checkpoints"
CELLS = [
    # name, ckpt, arch overrides, axis tag, training steps (exposure axis)
    ("small_0.84M", CK / "e005s_small.pt", dict(n_layer=4, n_head=4, n_embd=128), "scale", None),
    ("mid_2.7M",    CK / "e001.pt",        dict(n_layer=6, n_head=6, n_embd=192), "scale+exposure", 4000),
    ("large_10M",   CK / "e005s_large.pt", dict(n_layer=8, n_head=8, n_embd=320), "scale", None),
    ("exp_d400",    CK / "e048_direct400.pt", dict(n_layer=6, n_head=6, n_embd=192), "exposure", 400),
    ("exp_d800",    CK / "e048_direct800.pt", dict(n_layer=6, n_head=6, n_embd=192), "exposure", 800),
]
ANCHORS = {"small_0.84M": (1.5581, 0.03), "mid_2.7M": (1.622391, 0.03)}
CHUNK = {4: 255, 6: 96, 8: 48}     # sweep batch chunk by n_head (memory guard)
T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# ------------------------------------------------------------- exact machinery

@torch.no_grad()
def _manual_chunk(net: TinyGPT, idxs, vzero, kdrop, layers):
    """Exact manual forward, returns LAST-position logits (N, vocab).

    vzero/kdrop: (N, T) bool. layers: None = all, else set of layer indices.
    K-drop masks the softmax column of pos p for all query rows q != p
    (diagonal preserved -> G3: every row keeps >= 1 visible key).
    """
    N, T = idxs.shape
    pos = torch.arange(T)
    x = net.wte(idxs) + net.wpe(pos).unsqueeze(0)
    H = net.cfg.n_head
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    nondiag = ~torch.eye(T, dtype=torch.bool)
    for li, blk in enumerate(net.h):
        active = layers is None or li in layers
        xh = blk.ln1(x)
        qkv = blk.attn.c_attn(xh)
        C = qkv.shape[-1] // 3
        d = C // H
        q, k, v = qkv.split(C, dim=2)
        q = q.view(N, T, H, d).transpose(1, 2)
        k = k.view(N, T, H, d).transpose(1, 2)
        v = v.view(N, T, H, d).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
        att = att.masked_fill(causal, float("-inf"))
        if active and kdrop is not None and bool(kdrop.any()):
            att = att.masked_fill(kdrop[:, None, None, :] & nondiag, float("-inf"))
        probs = torch.softmax(att, dim=-1)
        if active and vzero is not None and bool(vzero.any()):
            v = v.masked_fill(vzero[:, None, :, None], 0.0)
        y = (probs @ v).transpose(1, 2).reshape(N, T, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
    return net.lm_head(net.ln_f(x)[:, -1, :])


@torch.no_grad()
def manual_logits(net, idxs, vzero=None, kdrop=None, layers=None, chunk=64):
    outs = []
    for i in range(0, idxs.shape[0], chunk):
        outs.append(_manual_chunk(net, idxs[i:i + chunk],
                                  vzero[i:i + chunk] if vzero is not None else None,
                                  kdrop[i:i + chunk] if kdrop is not None else None,
                                  layers))
    return torch.cat(outs, 0)


def bin_positions(bi: int, t_ctx: int) -> list[int]:
    a, b = BINS[bi]
    hi = min(b, t_ctx - 1)
    return list(range(a, hi + 1)) if a <= hi else []


def sample_and_ce(logits_clean: torch.Tensor, gen: torch.Generator):
    """Sample (temp 0.8, top-k 40) from filtered probs; CE from FULL softmax."""
    lg = logits_clean / TEMP
    v, _ = torch.topk(lg, TOPK)
    lg_f = lg.masked_fill(lg < v[-1], float("-inf"))
    tok = int(torch.multinomial(torch.softmax(lg_f, -1), 1, generator=gen))
    p_full = torch.softmax(logits_clean, -1)
    return tok, float(-math.log(max(p_full[tok].item(), 1e-12)))


def boot_ci(X, n: int = 1000, seed: int = 0):
    """Bootstrap CI over the sequence axis. X: (n_seq, ...) -> (lo, hi)."""
    X = np.asarray(X, float)
    if X.shape[0] < 2:
        return X[0], X[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, X.shape[0], size=(n, X.shape[0]))
    means = X[idx].mean(1)
    return np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)


# ----------------------------------------------------------------- per-sequence

@torch.no_grad()
def run_sequence(net: TinyGPT, prompt: torch.Tensor, gen: torch.Generator, ch: int,
                 with_per_layer: bool = True):
    """Free-run one 64->256 sequence, fixed anchor; returns measurement dict."""
    n_steps = T_TOTAL - PROMPT_TOK
    L = net.cfg.n_layer
    idx = prompt.clone()
    tl = np.full((n_steps, len(BINS)), np.nan)      # per-step binned V-zero dCE
    nan_count = 0
    g2_viol = 0
    for si, t in enumerate(range(PROMPT_TOK, T_TOTAL)):
        ctx = idx[:t]
        # batch: row0 clean (sampling), rows 1..: binned V-zero lesions
        rows, cols = [], []
        for bi in range(len(BINS)):
            ps = bin_positions(bi, t)
            if ps:
                rows.append(ps)
                cols.append(bi)
        N = 1 + len(rows)
        idxs = ctx[None].repeat(N, 1)
        vz = torch.zeros(N, t, dtype=torch.bool)
        for r, ps in enumerate(rows, start=1):
            vz[r, ps] = True
        logits = manual_logits(net, idxs, vz, None, None, chunk=ch)
        if bool(torch.isnan(logits).any()):
            nan_count += int(torch.isnan(logits).any(-1).sum())
            logits = torch.nan_to_num(logits, nan=0.0)
        tok, ce_clean = sample_and_ce(logits[0], gen)
        lg = torch.log_softmax(logits[1:].float(), -1)
        for r, bi in enumerate(cols):
            tl[si, bi] = ce_clean + float(lg[r, tok])
            # G2: newest bin with >=32 filled positions must hurt (> +0.5)
        newest = None
        for bi in reversed(range(len(BINS))):
            ps = bin_positions(bi, t)
            if len(ps) >= 32:
                newest = bi
                break
        if newest is not None and tl[si, newest] <= 0.5:
            g2_viol += 1
        idx = torch.cat([idx, torch.tensor([tok])])
    # ---- final step (t=255, target = position 255): full measurements
    ctx, tgt = idx[:T_TOTAL - 1], int(idx[T_TOTAL - 1])
    t_ctx = ctx.shape[0]
    ce_clean = float(-math.log(max(torch.softmax(
        manual_logits(net, ctx[None], None, None, chunk=ch)[0], -1)[tgt].item(), 1e-12)))
    # full per-position sweep: V-zero and K-drop
    P = t_ctx
    eye = torch.eye(P, dtype=torch.bool)
    idxs = ctx[None].repeat(P, 1)
    sweep_v = ce_clean + torch.gather(
        torch.log_softmax(manual_logits(net, idxs, eye, None, None, chunk=ch).float(), -1),
        1, torch.full((P, 1), tgt)).squeeze(1).numpy()
    sweep_k = ce_clean + torch.gather(
        torch.log_softmax(manual_logits(net, idxs, None, eye, None, chunk=ch).float(), -1),
        1, torch.full((P, 1), tgt)).squeeze(1).numpy()
    # final bins: vzero / kdrop / both
    fb = {}
    for kind in ("vzero", "kdrop", "both"):
        idxs = ctx[None].repeat(len(BINS), 1)
        vz = torch.zeros(len(BINS), t_ctx, dtype=torch.bool)
        kd = torch.zeros(len(BINS), t_ctx, dtype=torch.bool)
        for bi in range(len(BINS)):
            ps = bin_positions(bi, t_ctx)
            if kind in ("vzero", "both"):
                vz[bi, ps] = True
            if kind in ("kdrop", "both"):
                kd[bi, ps] = True
        lg = torch.log_softmax(manual_logits(net, idxs, vz, kd, None, chunk=ch).float(), -1)
        fb[kind] = ce_clean + lg[torch.arange(len(BINS)), tgt].numpy()
    # per-layer V-zero decomposition (final step)
    per_layer = np.full((L, len(BINS)), np.nan)
    if with_per_layer:
        for li in range(L):
            idxs = ctx[None].repeat(len(BINS), 1)
            vz = torch.zeros(len(BINS), t_ctx, dtype=torch.bool)
            for bi in range(len(BINS)):
                vz[bi, bin_positions(bi, t_ctx)] = True
            lg = torch.log_softmax(
                manual_logits(net, idxs, vz, None, layers={li}, chunk=ch).float(), -1)
            per_layer[li] = ce_clean + lg[torch.arange(len(BINS)), tgt].numpy()
    return dict(tl=tl, sweep_v=sweep_v, sweep_k=sweep_k, final_bins=fb,
                per_layer=per_layer, nan_count=nan_count, g2_viol=g2_viol,
                text=idx.tolist())


# ------------------------------------------------------------------- per-cell

def load_cell(name: str, ckpt: Path, arch: dict):
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    cfg = Cfg(vocab=65, block_size=256, **arch)
    net = TinyGPT(cfg)
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net


@torch.no_grad()
def g0_check(net, window: torch.Tensor) -> float:
    idx = window[:256][None]
    _, sdpa_logits = net(idx)
    man = manual_logits(net, idx, None, None, chunk=1)[0]
    p0 = torch.softmax(man, -1)
    p1 = torch.softmax(sdpa_logits[0, -1].float(), -1)
    return float((p0 - p1).abs().max().item())


def onset_age(dce_by_age: np.ndarray, k: int = A_STAR_K, thresh: float = THRESH):
    """dce_by_age: ages ascending (1=youngest ... 255=oldest/sink).
    Returns (naive, robust): youngest age whose mean dCE < thresh; robust
    requires the k-point moving mean to stay < thresh for k consecutive ages."""
    m = np.asarray(dce_by_age, float)
    below = m < thresh
    naive = int(np.argmax(below)) + 1 if below.any() else None
    if len(m) < k or not below.any():
        return naive, None
    ma = np.convolve(m, np.ones(k) / k, mode="valid")
    ok = ma < thresh
    # first index i (scanning from the end) whose k-length run is fully below
    run = 0
    robust_i = None
    for i in range(len(ok) - 1, -1, -1):
        run = run + 1 if ok[i] else 0
        if run >= k:
            robust_i = i
    robust = int(robust_i) + 1 if robust_i is not None else None
    return naive, robust


def classify_p3(traj: np.ndarray, steps: np.ndarray):
    init = float(np.nanmean(traj[:10]))
    mid = float(np.nanmean(traj[np.abs(steps - 160) <= 8]))
    fin = float(np.nanmean(traj[-10:]))
    if init <= 1e-9:
        cls = "FLAT"
    elif mid < 0.2 * init:
        cls = "DECAY"
    elif fin > 1.1 * init:
        cls = "GROW"
    elif abs(fin - init) < 0.1 * init:
        cls = "FLAT"
    else:
        cls = "PARTIAL_DECAY"
    return cls, dict(initial=init, mid=mid, final=fin, sign_flip=bool(fin < 0))


# --------------------------------------------------------------------- main

def main():
    assert not torch.cuda.is_available(), "G4 violated: CUDA visible"
    out_dir = run_dir("e053")
    corp = CharCorpus(REPO / "data" / "input.txt")
    assert corp.vocab_size == 65

    # fixed prompts (seed 202)
    gen_p = torch.Generator().manual_seed(SEED_PROMPT)
    ix = torch.randint(len(corp.val) - PROMPT_TOK - 1, (N_SEQ,), generator=gen_p)
    prompts = [corp.val[i:i + PROMPT_TOK] for i in ix]
    g0_window = corp.val[corp.seed % (len(corp.val) - T_TOTAL - 1):][:T_TOTAL]

    results, gates = {}, dict(
        G0={}, G1={}, G2_violations=0, G3_nan=0, G4_cpu_only=True,
        threads=THREADS, smoke=SMOKE)
    skipped = []

    for name, ckpt, arch, axis, steps_trained in CELLS:
        if elapsed() > BUDGET_S:
            skipped.append(name)
            log(f"SKIP {name}: budget guard")
            continue
        net = load_cell(name, ckpt, arch)
        ch = CHUNK[net.cfg.n_head]
        dev0 = g0_check(net, g0_window)
        gates["G0"][name] = dev0
        val_ce = estimate_loss(net, corp, "val", n_batches=12)
        if name in ANCHORS:
            a, tol = ANCHORS[name]
            gates["G1"][name] = dict(val_ce=val_ce, anchor=a, ok=abs(val_ce - a) <= tol)
        else:
            gates["G1"][name] = dict(val_ce=val_ce, anchor=None, ok=None)
        log(f"{name}: G0 prob-dev {dev0:.2e} | val CE {val_ce:.4f} | "
            f"steps={steps_trained} | axis={axis}")

        gen_s = torch.Generator().manual_seed(SEED_SAMPLE)   # paired across cells
        seqs = []
        for si, pr in enumerate(prompts):
            per_layer = (not SMOKE) and (elapsed() < BUDGET_S * 0.9)
            r = run_sequence(net, pr, gen_s, ch, with_per_layer=per_layer)
            seqs.append(r)
            gates["G3_nan"] += r["nan_count"]
            gates["G2_violations"] += r["g2_viol"]
            log(f"  {name} seq{si}: g2viol={r['g2_viol']} nan={r['nan_count']} "
                f"recency-bin dCE(final)={np.nanmean([s['final_bins']['vzero'][5] for s in seqs[:si+1]]):+.3f}")

        # static teacher-forced control
        st_v, st_k = None, None
        if elapsed() < BUDGET_S * 0.95:
            gen_st = torch.Generator().manual_seed(SEED_STATIC)
            ix2 = torch.randint(len(corp.val) - T_TOTAL - 1, (N_STATIC,), generator=gen_st)
            accv = np.zeros(len(BINS)); acck = np.zeros(len(BINS)); nw = 0
            for w0 in range(0, N_STATIC, 4):
                wins = [corp.val[ix2[w]:ix2[w] + T_TOTAL] for w in range(w0, min(w0 + 4, N_STATIC))]
                rows = 1 + 2 * len(BINS)
                idxs = torch.stack([w[:T_TOTAL - 1] for w in wins for _ in range(rows)])
                vz = torch.zeros(idxs.shape[0], T_TOTAL - 1, dtype=torch.bool)
                kd = torch.zeros_like(vz)
                for wi in range(len(wins)):
                    base = wi * rows
                    for bi in range(len(BINS)):
                        ps = bin_positions(bi, T_TOTAL - 1)
                        vz[base + 1 + bi, ps] = True
                        kd[base + 1 + len(BINS) + bi, ps] = True
                lp = torch.log_softmax(manual_logits(net, idxs, vz, kd, None, ch).float(), -1)
                for wi, win in enumerate(wins):
                    base = wi * rows
                    tgt = int(win[T_TOTAL - 1])
                    ce = -float(lp[base, tgt])
                    for bi in range(len(BINS)):
                        accv[bi] += ce + float(lp[base + 1 + bi, tgt])
                        acck[bi] += ce + float(lp[base + 1 + len(BINS) + bi, tgt])
                nw += len(wins)
            st_v, st_k = accv / nw, acck / nw
        else:
            skipped.append(f"{name}:static")

        # ---- aggregate
        tl = np.stack([s["tl"] for s in seqs])                       # (S, steps, bins)
        steps_axis = np.arange(PROMPT_TOK, T_TOTAL)
        sweep_v = np.stack([s["sweep_v"] for s in seqs])             # (S, 255) pos 0..254
        sweep_k = np.stack([s["sweep_k"] for s in seqs])
        fbp = {k: np.stack([s["final_bins"][k] for s in seqs]) for k in ("vzero", "kdrop", "both")}
        per_layer = np.stack([s["per_layer"] for s in seqs])         # (S, L, bins)
        age_axis = np.arange(1, T_TOTAL)                             # age of pos p = 255-p
        dce_age = sweep_v[:, ::-1]                                   # young(1) -> old(255)
        mean_age = dce_age.mean(0)
        ci_lo, ci_hi = boot_ci(dce_age)
        naive_a, robust_a = onset_age(mean_age)
        a_star = robust_a if robust_a is not None else naive_a
        junk_all = float((sweep_v <= -THRESH).mean())
        if a_star is not None:
            old = dce_age[:, a_star:]                                # ages > a_star
            junk_old = float((old <= -THRESH).mean())
        else:
            junk_old = None
        primacy = float(sweep_v[:, 1:17].mean())
        plateau = float(sweep_v[:, 224:255].mean())                  # youngest 31 pos
        bin_v = fbp["vzero"].mean(0)
        monotone = float(np.mean(np.diff(bin_v[1:]) > 0))            # 4 pairs, utility rising w/ recency
        p3_cls, p3_nums = classify_p3(tl[:, :, 0].mean(0), steps_axis)
        eq_gap = float(np.abs(fbp["both"] - fbp["kdrop"]).max())

        results[name] = dict(
            ckpt=str(ckpt), arch=dict(**arch, block_size=256, vocab=65),
            axis=axis, steps_trained=steps_trained, val_ce=val_ce,
            timeline=dict(steps=steps_axis.tolist(), bins=BIN_NAMES,
                          vzero_dce_mean=tl.mean(0).tolist()),
            sink_traj=dict(mean=tl[:, :, 0].mean(0).tolist(),
                           ci=[boot_ci(tl[:, :, 0])[0].tolist(), boot_ci(tl[:, :, 0])[1].tolist()],
                          p3_class=p3_cls, p3_numbers=p3_nums),
            final_profile=dict(positions=list(range(T_TOTAL - 1)),
                               age=list(age_axis),
                               vzero_dce_mean=sweep_v.mean(0).tolist(),
                               vzero_ci=[ci_lo.tolist(), ci_hi.tolist()],
                               kdrop_dce_mean=sweep_k.mean(0).tolist()),
            final_bins={k: dict(mean=v.mean(0).tolist(),
                                ci=[boot_ci(v)[0].tolist(), boot_ci(v)[1].tolist()])
                        for k, v in fbp.items()},
            per_layer_vzero=dict(mean=per_layer.mean(0).tolist(),
                                 ci=[boot_ci(per_layer)[0].tolist(), boot_ci(per_layer)[1].tolist()]),
            static_control=dict(vzero=st_v.tolist() if st_v is not None else None,
                                kdrop=st_k.tolist() if st_k is not None else None,
                                n_windows=N_STATIC),
            derived=dict(a_star_naive=naive_a, a_star_robust=robust_a, a_star=a_star,
                         a_star_frac=(a_star / (T_TOTAL - 1)) if a_star else None,
                         junk_frac_all=junk_all, junk_frac_beyond_ast=junk_old,
                         sink_dce_final=float(sweep_v[:, 0].mean()),
                         primacy_bump=primacy, recent_plateau=plateau,
                         primacy_ratio=(primacy / plateau) if plateau > 0 else None,
                         monotone_frac=monotone, equivalence_gap=eq_gap,
                         bin_vzero_final=bin_v.tolist()),
            sample_text_first160=corp.decode(torch.tensor(seqs[0]["text"][:160])),
        )
        log(f"{name} DONE a*={a_star} naive={naive_a} robust={robust_a} "
            f"sink_final={sweep_v[:, 0].mean():+.4f} p3={p3_cls} junk_old={junk_old}")

    # ---------------------------------------------------------------- verdicts
    sc = [n for n in results if results[n]["axis"].startswith("scale")]
    ex = [("exp_d400", 400), ("exp_d800", 800), ("mid_2.7M", 4000)]
    fr = {n: results[n]["derived"]["a_star_frac"] for n in results
          if results[n]["derived"]["a_star_frac"] is not None}
    scale_f = [fr[n] for n in sc if n in fr]
    p2_scale_ok = (len(scale_f) == len(sc)) and (max(scale_f) / min(scale_f) <= 1.5) \
        if scale_f else None
    p2_scale_ratio = (max(scale_f) / min(scale_f)) if len(scale_f) == len(sc) and scale_f else None
    p2_exp_ratio = (fr.get("mid_2.7M") / fr.get("exp_d400")
                    if fr.get("exp_d400") not in (None, 0) and fr.get("mid_2.7M") else None)
    sink_dce = {n: results[n]["derived"]["sink_dce_final"] for n in results}
    p1 = dict(
        sink_dead_count=int(sum(1 for v in sink_dce.values() if v < 0.05)),
        sink_loadbearing=[n for n, v in sink_dce.items() if v >= 0.3],
        primacy_ratio={n: results[n]["derived"]["primacy_ratio"] for n in results},
        monotone_frac={n: results[n]["derived"]["monotone_frac"] for n in results},
    )
    ratios = [r for r in p1["primacy_ratio"].values() if r is not None]
    p1["pass"] = bool(p1["sink_dead_count"] >= 3 and ratios
                      and all(r <= 0.25 for r in ratios)
                      and all(results[n]["derived"]["monotone_frac"] >= 0.75 for n in results))
    p2 = dict(scale_fractions={n: fr.get(n) for n in sc},
              scale_invariant=p2_scale_ok, scale_maxmin_ratio=p2_scale_ratio,
              exposure_a_star={n: results[n]["derived"]["a_star"] for n, _ in ex},
              exposure_growth_ratio=p2_exp_ratio,
              pass_=bool(p2_scale_ok and p2_exp_ratio is not None and p2_exp_ratio >= 1.5))
    p3 = dict(per_cell={n: results[n]["sink_traj"]["p3_class"] for n in results},
              numbers={n: results[n]["sink_traj"]["p3_numbers"] for n in results},
              sign_flip={n: results[n]["sink_traj"]["p3_numbers"]["sign_flip"] for n in results})
    junk = [results[n]["derived"]["junk_frac_beyond_ast"] for n in results
            if results[n]["derived"]["junk_frac_beyond_ast"] is not None]
    p4 = dict(junk_beyond_ast_per_cell={n: results[n]["derived"]["junk_frac_beyond_ast"]
                                        for n in results},
              pooled=float(np.mean(junk)) if junk else None,
              pass_=bool(junk and np.mean(junk) >= 0.05))

    metrics = dict(
        experiment="e053_cache_timeline", phase=1,
        started=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        wall_s=elapsed(), smoke=SMOKE, n_seq=N_SEQ, bins=BIN_NAMES,
        seeds=dict(prompts=SEED_PROMPT, sampling=SEED_SAMPLE, static=SEED_STATIC),
        protocol=dict(prompt_tokens=PROMPT_TOK, t_total=T_TOTAL, temp=TEMP, topk=TOPK,
                      fixed_anchor=True, threshold=THRESH, a_star_window=A_STAR_K),
        gates=gates, skipped=skipped, cells=results,
        verdicts=dict(P1_shape=p1, P2_onset=p2, P3_sink=p3, P4_junk=p4),
    )
    save_json(out_dir / "metrics.json", metrics)
    log(f"metrics.json written ({len(results)} cells, skipped={skipped})")

    # ------------------------------------------------------------------- plot
    plot(out_dir / "e053_utility_timeline.png", results)
    log(f"plot written; total wall {elapsed():.0f}s")


# ---------------------------------------------------------------------- plot

def plot(path: Path, R: dict):
    names = list(R)
    fig = plt.figure(figsize=(19, 12.5))
    gs = fig.add_gridspec(3, 6, hspace=0.42, wspace=0.42)
    colors = {"small_0.84M": "tab:blue", "mid_2.7M": "tab:green", "large_10M": "tab:red",
              "exp_d400": "tab:orange", "exp_d800": "tab:purple"}

    # row 1: THE curve — per-position V-zero dCE vs cache age, one panel per cell + overlay
    for i, n in enumerate(names):
        ax = fig.add_subplot(gs[0, i])
        fp = R[n]["final_profile"]
        age = np.asarray(fp["age"]); m = np.asarray(fp["vzero_dce_mean"])
        lo, hi = np.asarray(fp["vzero_ci"][0]), np.asarray(fp["vzero_ci"][1])
        ax.fill_between(age, lo, hi, alpha=0.25, color=colors[n])
        ax.plot(age, m, lw=1.4, color=colors[n])
        ax.axhline(0, color="k", lw=0.6); ax.axhline(THRESH, color="gray", ls=":", lw=0.8)
        a = R[n]["derived"]["a_star"]
        if a: ax.axvline(a, color=colors[n], ls="--", lw=1)
        ax.set_title(f"{n}\na*={a} (dash)", fontsize=9)
        ax.set_xlabel("cache age (tokens)"); ax.set_ylabel("V-zero dCE (nats)")
        ax.set_xlim(0, 256)
    ax = fig.add_subplot(gs[0, 5])
    for n in names:
        fp = R[n]["final_profile"]
        ax.plot(fp["age"], fp["vzero_dce_mean"], lw=1.3, color=colors[n],
                ls="--" if R[n]["axis"] == "exposure" else "-", label=n)
    ax.axhline(0, color="k", lw=0.6); ax.axhline(THRESH, color="gray", ls=":", lw=0.8)
    ax.legend(fontsize=7); ax.set_title("overlay (solid=scale, dash=exposure)", fontsize=9)
    ax.set_xlabel("cache age (tokens)"); ax.set_xlim(0, 256)

    # row 2: age x step heatmap (primary + largest), sink trajectory (P3)
    for j, n in enumerate([x for x in ("small_0.84M", "large_10M") if x in R][:2]):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
        tl = np.asarray(R[n]["timeline"]["vzero_dce_mean"])     # (steps, bins)
        M = np.ma.masked_invalid(tl.T)
        vmax = np.nanpercentile(tl, 98)
        im = ax.imshow(M, aspect="auto", origin="lower", cmap="magma",
                       extent=[64, 256, -0.5, 5.5], vmin=0, vmax=max(vmax, 0.1))
        ax.set_yticks(range(6), BIN_NAMES)
        ax.set_xlabel("generation step t"); ax.set_title(f"age-bin x step dCE — {n}", fontsize=9)
        plt.colorbar(im, ax=ax, label="dCE (nats)")
    ax = fig.add_subplot(gs[1, 4:6])
    for n in names:
        st = R[n]["sink_traj"]
        steps = np.asarray(R[n]["timeline"]["steps"])
        ax.plot(steps, st["mean"], color=colors[n],
                ls="--" if R[n]["axis"] == "exposure" else "-", lw=1.2, label=f"{n}:{st['p3_class']}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("generation step t"); ax.set_ylabel("sink (pos 0) V-zero dCE")
    ax.set_title("P3 — sink utility timeline", fontsize=9); ax.legend(fontsize=7)

    # row 3: per-layer decomposition (scale cell with most old-bin structure), onset ages, static vs free
    sc = [n for n in R if R[n]["axis"].startswith("scale")]
    pick = max(sc, key=lambda n: float(np.nanmax(np.asarray(R[n]["per_layer_vzero"]["mean"])[:, 3])))
    ax = fig.add_subplot(gs[2, 0:2])
    pl = np.asarray(R[pick]["per_layer_vzero"]["mean"])
    M = np.ma.masked_invalid(pl)
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(6), BIN_NAMES); ax.set_ylabel("layer")
    ax.set_title(f"per-layer V-zero dCE — {pick}", fontsize=9)
    plt.colorbar(im, ax=ax, label="dCE (nats)")
    ax = fig.add_subplot(gs[2, 2:4])
    xs, ys, cs = [], [], []
    for n in names:
        a = R[n]["derived"]["a_star"]
        if a:
            xs.append(n); ys.append(a); cs.append(colors[n])
    ax.bar(xs, ys, color=cs)
    ax.set_ylabel("onset age a* (tokens)")
    ax.set_title("P2 — dead-weight onset (scale vs exposure)", fontsize=9)
    ax.tick_params(axis="x", rotation=30)
    ax = fig.add_subplot(gs[2, 4:6])
    for n in names:
        stf = R[n]["static_control"]["vzero"]
        fr = R[n]["derived"]["bin_vzero_final"]
        if stf is None: continue
        ax.scatter(np.abs(stf[1:]), np.abs(fr[1:]), color=colors[n], label=n, s=28)
    lim = [1e-4, 10]
    ax.plot(lim, lim, "k--", lw=0.8); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("|static (real text) bin dCE|"); ax.set_ylabel("|free-run final bin dCE|")
    ax.set_title("self-generated vs real context aging", fontsize=9); ax.legend(fontsize=7)

    fig.suptitle("E053 — cache utility timeline: per-position causal dCE of V-zero lesions "
                 "during fixed-anchor generation (8 seeded seqs, T=256)", fontsize=12)
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
