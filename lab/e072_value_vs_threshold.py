"""E072 — T044's REGISTERED value-side vs threshold discriminator.

Trigger (THINKING.md T044, 2026-09-26 ~10:25Z): e070 showed the ages-4-17
liveness GREW under eval-256 truncation with UNCHANGED final-query attention
(ages 1-20 mass ratio 1.006 [0.97,1.03]) and unchanged clean CE. Two live
stories: H-value-side (the value/read pathway carries more load in the
shorter window) vs H-threshold-artifact (the 0.01-nat 5-pt robust rule is
straddled by a fragile a* statistic — B=4, 2 of 4 sequences driving it).

This run rebuilds the e069/e070 protocol VERBATIM (frozen e053c ctx-512
checkpoint runs/checkpoints/e053c_ctx512.pt, seeds 202/7, same B=4
sequences, same generation/sweep/onset machinery; eval-256 = last-256
positions re-indexed wpe 0..254) and measures, on the SAME sequences, both
windows:

1. V-VECTOR NORMS at ages 4-17: the value-projection output vectors (per
   layer, the v-path input to attention) at those key positions — mean norm
   per age and per layer, eval-256 vs eval-512, ratio + bootstrap CI. Plus
   the post-attention aggregated residual contribution: the norm of
   c_proj(sum over ages-4-17 keys of p_final[k] * v[k]) per layer (the
   actual residual-stream write those keys make through the final query).

2. B=16 BOOTSTRAP of a*: regenerate 16 sequences under the same prompt-draw
   rules (battery = e069's exact 4 [seeds 202/7] + 12 fresh [seeds 302/17,
   the 202/7 family extended +100/+10, documented]); fit a* on eval-256
   with B=16; report the a* CI and the bootstrap distribution vs the 5-pt
   robust rule's edge (fraction of resamples landing AT 6 vs inside 12-24).
   SECONDARY (unregistered contrast): the same fit on eval-512 with B=16.

REGISTERED NUMBERS (frozen before running):
  - V-norm ratio at ages 4-17 outside [0.9, 1.1]  => H-VALUE-SIDE (the load
    genuinely grew in the value pathway);
  - V-norms stable (ratio inside [0.9,1.1]) AND the B=16 a* CI still
    wide/straddling  => H-THRESHOLD (instrument fragility);
  - V-norms stable AND B=16 CI tight around 12-20  => NEITHER simple story —
    report honestly (a real, stable, attention-invariant, value-norm-
    invariant liveness shift; residual-scale/LN story to register next).

Run:     python lab/e072_value_vs_threshold.py
Outputs: runs/e072/metrics.json + runs/e072/value_vs_threshold.png
Envelope: NO training, NO new automations; CPU-only, single step, minutes.
No NOTES/THINKING/QUEUE/STATE edits; no commit (the dispatcher owns those).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # task spec: CPU-only (pre-torch)
import torch

# this torch build reports is_available()==True even with an empty
# CUDA_VISIBLE_DEVICES; force CPU so common.DEVICE=="cpu" (e053b/e069/e070)
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import common  # noqa: E402
from common import (  # noqa: E402
    REPO,
    Cfg,
    CharCorpus,
    TinyGPT,
    estimate_loss,
    run_dir,
    save_json,
)

# ------------------------------------------------------------------ constants
PROMPT_TOK = 64
T_TOTAL = 512                             # e053c's ctx-512 window
TEMP, TOPK = 0.8, 40
SEED_PROMPT, SEED_SAMPLE = 202, 7         # e053b/e053c/e069/e070 exact seeds
N_PROMPTS = 8
B = 4                                     # e069/e070 uniform B (battery A)
BOOT_N = 1000
THREADS = 12
torch.set_num_threads(THREADS)

EVAL256 = 256
P256 = EVAL256 - 1                        # 255 positions in the eval-256 window
P512 = T_TOTAL - 1                        # 511 positions in the eval-512 window
THRESH = 0.01                             # dead-weight threshold (nats)
A_STAR_K = 5                              # sustained window (e053b verbatim)
CHUNK_512 = 64                            # e069/e070 measured chunks
CHUNK_256 = 128

# battery B (B=16): e069's exact 4 + 12 fresh from the extended family
B_X = 12
SEED_PROMPT_X = 302                       # 202-family + 100
SEED_SAMPLE_X = 17                        # 7-family + 10

# ages whose liveness grew in e069's D1 (the registered band)
A_LO, A_HI = 4, 17
AGES_BAND = list(range(A_LO, A_HI + 1))

# ---- REGISTERED decision numbers (FROZEN, task spec / T044) -----------------
VN_LO, VN_HI = 0.9, 1.1                   # V-norm ratio band (ages 4-17)
STRADDLE_LO, STRADDLE_HI = 8.0, 12.0      # CI covering both neighborhoods
WIDE_BAR = 12.0                           # CI width bar ("still wide")
TIGHT_LO, TIGHT_HI = 12.0, 20.0           # "tight around 12-20"

# ---- e069 stored readouts (protocol-identity instrument gates) -------------
# NOTE: eval512's winstart ref is the AGE-255 entry (+0.0069; wpe row 256).
# e070's E069_REF table mistakenly stored -0.0655 there — that is the
# eval-512 curve's LAST entry = age 511 = the true sequence start (a
# different object; e070's own replica read +0.0069 and its G3 note says so).
E069_REF = {
    "eval512": {"a_star": 6, "ci": [4.0, 8.0], "clean_ce": 0.4528,
                "ages1_5": [1.742, 4.467, 2.319, 0.914, 0.321],
                "per_seq": [8, 4, 6, 3], "winstart_dce": 0.0069},
    "eval256": {"a_star": 18, "ci": [7.0, 30.0], "clean_ce": 0.4587,
                "ages1_5": [2.715, 4.888, 2.349, 0.064, 0.131],
                "per_seq": [8, 26, 30, 7], "winstart_dce": 0.6986},
}
CKPT = REPO / "runs" / "checkpoints" / "e053c_ctx512.pt"
E053C_VAL_CE = 1.5226792494455974         # the run being interrogated
T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# ------------------------------------------------- machinery (VERBATIM e069/e070)

@torch.no_grad()
def _manual_chunk(net: TinyGPT, idxs, vzero, kdrop, layers, pos_offset=0):
    """Exact manual forward, returns LAST-position logits (N, vocab).
    pos_offset lets the same tokens be fed at shifted wpe indices."""
    N, T = idxs.shape
    pos = torch.arange(T) + int(pos_offset)
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
    return net.lm_head(net.ln_f(x[:, -1, :]))


@torch.no_grad()
def manual_logits(net, idxs, vzero=None, kdrop=None, layers=None, chunk=64,
                  pos_offset=0):
    outs = []
    for i in range(0, idxs.shape[0], chunk):
        outs.append(_manual_chunk(net, idxs[i:i + chunk],
                                  vzero[i:i + chunk] if vzero is not None else None,
                                  kdrop[i:i + chunk] if kdrop is not None else None,
                                  layers, pos_offset))
    return torch.cat(outs, 0)


def sample_and_ce(logits_clean: torch.Tensor, gen: torch.Generator):
    """Sample (temp 0.8, top-k 40) from filtered probs; CE from FULL softmax.
    [VERBATIM e053b — CPU tensors + CPU generator, stream-identical]"""
    lg = logits_clean / TEMP
    v, _ = torch.topk(lg, TOPK)
    lg_f = lg.masked_fill(lg < v[-1], float("-inf"))
    tok = int(torch.multinomial(torch.softmax(lg_f, -1), 1, generator=gen))
    p_full = torch.softmax(logits_clean, -1)
    return tok, float(-math.log(max(p_full[tok].item(), 1e-12)))


def boot_ci(X, n: int = BOOT_N, seed: int = 0):
    """Bootstrap CI over the sequence axis. X: (n_seq, ...) -> (lo, hi)."""
    X = np.asarray(X, float)
    if X.shape[0] < 2:
        return X[0], X[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, X.shape[0], size=(n, X.shape[0]))
    means = X[idx].mean(1)
    return np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)


def onset_age(dce_by_age: np.ndarray, k: int = A_STAR_K, thresh: float = THRESH):
    """dce_by_age: ages ascending (1=youngest ... Pw=oldest/sink).
    Returns (naive, robust). [VERBATIM e053b/E053 — the statistic]"""
    m = np.asarray(dce_by_age, float)
    below = m < thresh
    naive = int(np.argmax(below)) + 1 if below.any() else None
    if len(m) < k or not below.any():
        return naive, None
    ma = np.convolve(m, np.ones(k) / k, mode="valid")
    ok = ma < thresh
    run, robust_i = 0, None
    for i in range(len(ok) - 1, -1, -1):
        run = run + 1 if ok[i] else 0
        if run >= k:
            robust_i = i
    robust = int(robust_i) + 1 if robust_i is not None else None
    return naive, robust


def a_star_rule(mean_age: np.ndarray):
    """e053b's reported rule: robust preferred, naive fallback."""
    naive, robust = onset_age(mean_age)
    a = robust if robust is not None else naive
    return a, naive, robust


# ------------------------------------------- batched generation (e053b math)

@torch.no_grad()
def prefill_batch(net: TinyGPT, idx: torch.Tensor):
    """Batched prefill, (B, Tp) -> last-position logits (B, V) + KV cache."""
    B, T = idx.shape
    x = net.wte(idx) + net.wpe(torch.arange(T))
    H = net.cfg.n_head
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    kv = []
    for blk in net.h:
        xh = blk.ln1(x)
        qkv = blk.attn.c_attn(xh)
        C = qkv.shape[-1] // 3
        d = C // H
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, H, d).transpose(1, 2)
        k = k.view(B, T, H, d).transpose(1, 2)
        v = v.view(B, T, H, d).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
        att = att.masked_fill(causal, float("-inf"))
        y = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(B, T, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
        kv.append((k, v))
    return net.lm_head(net.ln_f(x[:, -1, :])), kv


@torch.no_grad()
def decode_step_batch(net: TinyGPT, toks: torch.Tensor, pos: int, kv: list):
    """Batched incremental decode: (B,) tokens at position pos -> (B, V)."""
    B = toks.shape[0]
    x = net.wte(toks) + net.wpe(torch.full((B,), pos))
    H = net.cfg.n_head
    for li, blk in enumerate(net.h):
        xh = blk.ln1(x)
        qkv = blk.attn.c_attn(xh)
        C = qkv.shape[-1] // 3
        d = C // H
        q, k, v = qkv.split(C, dim=1)
        q = q.view(B, 1, H, d).transpose(1, 2)
        k = k.view(B, 1, H, d).transpose(1, 2)
        v = v.view(B, 1, H, d).transpose(1, 2)
        kp, vp = kv[li]
        k = torch.cat([kp, k], dim=2)
        v = torch.cat([vp, v], dim=2)
        kv[li] = (k, v)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))   # (B,H,1,t+1)
        y = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(B, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
    return net.lm_head(net.ln_f(x))


@torch.no_grad()
def generate_batch(net: TinyGPT, prompts, gen: torch.Generator):
    """Free-run 64->T_TOTAL for B sequences at once, fixed anchor. Per-row
    draws from the shared CPU generator in row order per step
    (e053b-identical stream math). Returns idx (B, T) and the final
    (position-T_TOTAL-1) logits (B, V). [CPU-only, e069/e070 verbatim]"""
    B = len(prompts)
    idx = torch.stack(prompts)
    logits, kv = prefill_batch(net, idx)
    for t in range(PROMPT_TOK, T_TOTAL):
        toks = torch.zeros(B, dtype=torch.long)
        for j in range(B):
            tok, _ = sample_and_ce(logits[j], gen)
            toks[j] = tok
        idx = torch.cat([idx, toks[:, None]], 1)
        if t < T_TOTAL - 1:
            logits = decode_step_batch(net, toks, t, kv)
    return idx, logits


@torch.no_grad()
def fine_sweep(net: TinyGPT, ctx: torch.Tensor, tgt: torch.Tensor,
               clean_logits: torch.Tensor, ch: int, pos_offset: int = 0):
    """Final-step fine V-zero sweep for all B sequences at once, on whatever
    (ctx, tgt, clean_logits) window is handed in. Returns dce (B, Pw) indexed
    by POSITION (0..Pw-1); age of pos p = Pw - p, so the age-ascending view
    is dce[:, ::-1], plus the per-sequence clean CE. [e053c math, verbatim]"""
    Bs, Pw = ctx.shape
    p_clean = torch.softmax(clean_logits.float(), -1)
    ce_clean = -torch.log(p_clean[torch.arange(Bs), tgt].clamp_min(1e-12))
    idxs = ctx.repeat_interleave(Pw, dim=0)                # (B*Pw, Pw)
    vz = torch.eye(Pw, dtype=torch.bool).repeat(Bs, 1)     # row b*Pw+p: zero p
    les = manual_logits(net, idxs, vz, None, None, chunk=ch, pos_offset=pos_offset)
    lp = torch.log_softmax(les.float(), -1)
    rows = torch.arange(Bs * Pw)
    dce = (-ce_clean.repeat_interleave(Pw) - lp[rows, tgt.repeat_interleave(Pw)])
    return dce.view(Bs, Pw).numpy(), ce_clean.numpy()


def bootstrap_stat(dce_age: np.ndarray, stat, n: int = BOOT_N, seed: int = 0):
    """Bootstrap any scalar stat over the sequence axis."""
    S = dce_age.shape[0]
    if S < 2:
        v = stat(dce_age)
        return (v, v) if v is not None else (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sel = rng.integers(0, S, S)
        vals.append(stat(dce_age[sel]))
    vals = np.asarray([np.nan if v is None else v for v in vals], float)
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def bootstrap_a_star_dist(dce_age: np.ndarray, n: int = BOOT_N, seed: int = 0):
    """Full bootstrap distribution (n resamples) of the a* rule over the
    sequence axis — same rng stream as bootstrap_stat, so percentiles of this
    array reproduce analyze_window's a_star_ci exactly."""
    S = dce_age.shape[0]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sel = rng.integers(0, S, S)
        vals.append(a_star_rule(dce_age[sel].mean(0))[0])
    return np.asarray([np.nan if v is None else v for v in vals], float)


def analyze_window(sweep_pos: np.ndarray):
    """sweep_pos: (B, Pw) V-zero dCE by position. e053c's analyze_cell
    generalized to any window length Pw. Returns the derived block."""
    dce_age = sweep_pos[:, ::-1]                       # (B, Pw) young->old
    Pw = dce_age.shape[1]
    mean_age = dce_age.mean(0)
    a_star, naive, robust = a_star_rule(mean_age)
    a_lo, a_hi = bootstrap_stat(dce_age, lambda m: a_star_rule(m.mean(0))[0])
    live = mean_age >= THRESH
    live_frac = float(live.mean())
    lf_lo, lf_hi = bootstrap_stat(dce_age, lambda m: float((m.mean(0) >= THRESH).mean()))
    frac = (a_star / Pw) if a_star is not None else None
    return dict(
        a_star=a_star, a_star_naive=naive, a_star_robust=robust,
        a_star_ci=[a_lo, a_hi], a_star_frac=frac,
        live_frac=live_frac, live_frac_ci=[lf_lo, lf_hi],
        per_seq_a_star=[a_star_rule(dce_age[s])[0] for s in range(dce_age.shape[0])],
        mean_age_curve=mean_age, ci=boot_ci(dce_age, BOOT_N),
        dce_age=dce_age,
    )


def sweep_block(net, ctx, tgt, clean_logits, ch, pos_offset=0):
    """Clean forward -> full V-zero sweep -> analysis, plus bookkeeping."""
    dce, ce_clean = fine_sweep(net, ctx, tgt, clean_logits, ch, pos_offset)
    nan_n = int(np.isnan(dce).sum())
    if nan_n:
        dce = np.nan_to_num(dce, nan=0.0)
    d = analyze_window(dce)
    log(f"  a*={d['a_star']} CI [{d['a_star_ci'][0]:.0f},{d['a_star_ci'][1]:.0f}] "
        f"naive={d['a_star_naive']} robust={d['a_star_robust']} "
        f"live_frac={d['live_frac']:.3f} | per-seq {d['per_seq_a_star']} "
        f"| clean CE {ce_clean.mean():.3f} (nan->0: {nan_n})")
    return d, ce_clean


def age_keys(Pw: int, a_lo: int, a_hi: int) -> torch.Tensor:
    """Context key positions for ages a_lo..a_hi (age of key p = Pw - p)."""
    return torch.arange(Pw - a_hi, Pw - a_lo + 1)


def keys_for_ages(Pw: int, ages) -> torch.Tensor:
    """Key positions listed in AGE-ASCENDING column order (age a -> Pw - a)."""
    return torch.tensor([Pw - int(a) for a in ages], dtype=torch.long)


# ------------------------------------- NEW: value-pathway probe (e072's own)

@torch.no_grad()
def value_probe(net: TinyGPT, idxs: torch.Tensor):
    """Manual forward in the EXACT _manual_chunk op order that ALSO records,
    per layer, the value-projection outputs v (N, T, H, d) — the v-path input
    to attention, BEFORE any softmax weighting — and the final query's
    attention probability row (N, H, T). Returns (v_by_layer, probs_by_layer,
    final logits). G4 checks the logits against manual_logits."""
    N, T = idxs.shape
    pos = torch.arange(T)
    x = net.wte(idxs) + net.wpe(pos).unsqueeze(0)
    H = net.cfg.n_head
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    vs, ps = [], []
    for blk in net.h:
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
        p = torch.softmax(att, dim=-1)
        vs.append(v)
        ps.append(p[:, :, -1, :])                       # final query's row
        y = (p @ v).transpose(1, 2).reshape(N, T, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
    return vs, ps, net.lm_head(net.ln_f(x[:, -1, :]))


def vnorm_tensors(vs, keys: torch.Tensor):
    """vs: per-layer v (N, H, T, d) — the manual forward keeps v head-major
    after transpose(1,2); key positions live on axis 2. keys: key positions,
    age-ascending. Returns (full, head): concatenated-value norms (N, L, K)
    and per-head norms (N, L, K, H), columns aligned with keys."""
    sel = [v.index_select(2, keys) for v in vs]                  # (N, H, K, d)
    full = torch.stack([s.norm(dim=(1, 3)) for s in sel], 1)     # (N, L, K)
    head = torch.stack([s.norm(dim=3).transpose(1, 2) for s in sel], 1)  # (N,L,K,H)
    return full.numpy().astype(np.float64), head.numpy().astype(np.float64)


@torch.no_grad()
def resid_contrib(net: TinyGPT, vs, ps, keys: torch.Tensor):
    """Post-attention aggregated residual contribution of the given key set:
    per layer, sum_k p_final[k] * v[k] (the keys' share of the final query's
    attention output), then through c_proj — the actual residual-stream write.
    Returns (pre_cproj, post_cproj): (N, L) norms."""
    pre, post = [], []
    for li, blk in enumerate(net.h):
        w = ps[li].index_select(-1, keys)               # (N, H, K)
        vv = vs[li].index_select(2, keys)               # (N, H, K, d)
        c = torch.einsum("nhk,nhkd->nhd", w, vv).flatten(1)   # (N, C), head-major
        pre.append(c.norm(dim=1))
        post.append(blk.attn.c_proj(c).norm(dim=1))
    return (torch.stack(pre, 1).numpy().astype(np.float64),
            torch.stack(post, 1).numpy().astype(np.float64))


def ratio_of_means(x256: np.ndarray, x512: np.ndarray, n: int = BOOT_N, seed: int = 0):
    """Paired bootstrap of mean(x256)/mean(x512) over the sequence axis.
    Inputs (S, ...) — any trailing axes are aggregated by MEAN within each
    sequence first (norms aggregate by mean, not sum)."""
    x = np.asarray(x256, float).reshape(len(x256), -1).mean(1)
    y = np.asarray(x512, float).reshape(len(x512), -1).mean(1)
    ratio = float(x.mean() / max(y.mean(), 1e-30))
    rng = np.random.default_rng(seed)
    S = len(x)
    vals = []
    for _ in range(n):
        sel = rng.integers(0, S, S)
        vals.append(x[sel].mean() / max(y[sel].mean(), 1e-30))
    return ratio, [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def dist_stats(dist: np.ndarray) -> dict:
    v = dist[~np.isnan(dist)]
    return dict(
        n_boot=int(BOOT_N), n_valid=int(len(v)),
        frac_eq_6=float((v == 6).mean()),
        frac_in_7_11=float(((v >= 7) & (v <= 11)).mean()),
        frac_in_12_24=float(((v >= 12) & (v <= 24)).mean()),
        frac_le_8=float((v <= 8).mean()),
        frac_gt_24=float((v > 24).mean()),
        frac_nan=float(np.isnan(dist).mean()),
        median=float(np.nanmedian(v)),
        pcts={str(p): float(np.nanpercentile(v, p)) for p in (2.5, 25, 50, 75, 97.5)},
    )


# ---------------------------------------------------------------------- main


def main():
    assert torch.cuda.device_count() == 0, "CPU-only violated: CUDA visible"
    assert common.DEVICE == "cpu", f"common.DEVICE={common.DEVICE} != cpu"
    out_dir = run_dir("e072")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gates = dict(cpu_only=True, threads=THREADS, smoke=False)

    # ---- battery: rebuild corpus + net EXACTLY as e053c/e069/e070 did
    corp = CharCorpus(REPO / "data" / "input.txt")       # seed 1337
    assert corp.vocab_size == 65
    st = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    cfg = Cfg(vocab=corp.vocab_size, n_layer=4, n_head=4, n_embd=128,
              block_size=T_TOTAL)
    net = TinyGPT(cfg)
    net.load_state_dict(sd, strict=True)
    net.eval()
    n_params = net.num_params()
    steps_in_ckpt = int(st.get("step", -1)) if isinstance(st, dict) else -1
    gates["G2_params"] = dict(params=n_params, expected=873472,
                              ok=bool(n_params == 873472))
    val_ce = estimate_loss(net, corp, "val", n_batches=12)
    gates["G1_val_ce"] = dict(val_ce=val_ce, ref=E053C_VAL_CE, tol=0.02,
                              ok=bool(abs(val_ce - E053C_VAL_CE) <= 0.02))
    log(f"e053c net loaded (ckpt step {steps_in_ckpt}, {n_params:,} params) | "
        f"val CE {val_ce:.4f} vs e053c {E053C_VAL_CE:.4f} "
        f"-> G1 {'PASS' if gates['G1_val_ce']['ok'] else 'FAIL'}")

    # ---- battery A sequences: e069/e070 protocol verbatim (seeds 202/7, B=4)
    gen_p = torch.Generator().manual_seed(SEED_PROMPT)
    ix = torch.randint(len(corp.val) - PROMPT_TOK - 1, (N_PROMPTS,),
                       generator=gen_p)
    prompts = [corp.val[i:i + PROMPT_TOK] for i in ix]
    log(f"battery A: {N_PROMPTS} prompts (seed {SEED_PROMPT}); using first "
        f"B={B}; prompt0 prefix: {corp.decode(prompts[0])[:32]!r}")
    gen = torch.Generator().manual_seed(SEED_SAMPLE)     # per-cell seed-7 re-init
    idx, final_logits = generate_batch(net, prompts[:B], gen)
    log(f"battery A: generated {B} seqs 64->{T_TOTAL} (fixed anchor, CPU)")

    # G0b: batched incremental-KV final logits vs full recompute (prob dev)
    with torch.no_grad():
        full = manual_logits(net, idx[:, :-1], None, None, None, chunk=B)
    dev_b = float((torch.softmax(final_logits.float(), -1)
                   - torch.softmax(full.float(), -1)).abs().max())
    gates["G0b_kvs_vs_full"] = dict(max_prob_dev=dev_b, ok=bool(dev_b < 1e-4))
    log(f"G0b incremental-KV vs full recompute: max prob dev {dev_b:.2e} "
        f"-> {'PASS' if dev_b < 1e-4 else 'FAIL'}")

    # ---- frames (eval-window protocol verbatim e069 sweep A / sweep B)
    tgt = idx[:, -1]
    ctx512 = idx[:, :-1]                                 # seq 0..510, wpe 0..510
    ctx256 = idx[:, T_TOTAL - EVAL256:-1]                # seq 256..510, wpe 0..254
    assert torch.equal(ctx256[:, 0], idx[:, T_TOTAL - EVAL256])
    assert torch.equal(tgt, idx[:, -1])

    # ---- replica sweeps (protocol-identity gates vs e069's stored readouts)
    log("replica sweep 1/2: eval-512 fine onset fit (e069 sweep A replica)")
    d512, ce512 = sweep_block(net, ctx512, tgt, final_logits, CHUNK_512)
    log("replica sweep 2/2: eval-256 re-indexed fine onset fit (e069 sweep B)")
    clean256 = manual_logits(net, ctx256, None, None, None, chunk=CHUNK_256)
    d256, ce256 = sweep_block(net, ctx256, tgt, clean256, CHUNK_256,
                              pos_offset=0)
    for key, d, ce in [("eval512", d512, ce512), ("eval256", d256, ce256)]:
        ref = E069_REF[key]
        ages15 = d["mean_age_curve"][:5]
        dev15 = float(np.max(np.abs(ages15 - np.asarray(ref["ages1_5"]))))
        ws_dev = abs(float(d["mean_age_curve"][P256 - 1]) - ref["winstart_dce"])
        a_match = d["a_star"] == ref["a_star"]
        ps_match = list(d["per_seq_a_star"]) == list(ref["per_seq"])
        ce_dev = abs(float(ce.mean()) - ref["clean_ce"])
        ok = bool(a_match and dev15 < 0.05 and ws_dev < 0.05 and ce_dev < 0.02
                  and ps_match)
        gates[f"G3_{key}_vs_e069"] = dict(
            a_star=d["a_star"], a_star_ref=ref["a_star"], a_star_match=a_match,
            per_seq_a_star=list(d["per_seq_a_star"]),
            per_seq_ref=ref["per_seq"], per_seq_match=ps_match,
            ages1_5=[float(a) for a in ages15], max_dev_ages1_5=dev15,
            winstart_dce=float(d["mean_age_curve"][P256 - 1]),
            winstart_dce_ref=ref["winstart_dce"], winstart_dev=ws_dev,
            clean_ce=float(ce.mean()), clean_ce_ref=ref["clean_ce"],
            clean_ce_dev=ce_dev,
            note=("eval512 ref = age-255 entry (+0.0069); e070's stored "
                  "-0.0655 was the age-511 entry (true seq start)") if key == "eval512" else "",
            ok=ok)
        log(f"G3 {key}: a* {d['a_star']} (ref {ref['a_star']}), per-seq "
            f"{list(d['per_seq_a_star'])} (ref {ref['per_seq']}), ages1-5 "
            f"maxdev {dev15:.4f}, age-255 dev {ws_dev:.4f}, CE dev "
            f"{ce_dev:.4f} -> {'PASS' if ok else 'FAIL'}")

    # ============================================================ PART 1
    # ---- V-vector norms at ages 4-17, eval-256 vs eval-512, same tokens
    log(f"PART 1: value-probe both frames (ages {A_LO}-{A_HI})")
    vs512, ps512, lg512p = value_probe(net, ctx512)
    vs256, ps256, lg256p = value_probe(net, ctx256)

    # G4: probe forward logits == manual recompute (same op order)
    with torch.no_grad():
        ref512 = manual_logits(net, ctx512, None, None, None, chunk=CHUNK_512)
        ref256 = manual_logits(net, ctx256, None, None, None, chunk=CHUNK_256)
    g4 = float(max((lg512p - ref512).abs().max(), (lg256p - ref256).abs().max()))
    gates["G4_probe_vs_manual"] = dict(max_logit_dev=g4, ok=bool(g4 < 1e-5))
    log(f"G4 value-probe vs manual logits dev {g4:.2e} "
        f"-> {'PASS' if g4 < 1e-5 else 'FAIL'}")

    # G5: age-aligned token identity between the frames (same underlying
    # tokens at every registered age, different wpe/context length)
    idn_ok = True
    for a in AGES_BAND:
        if not torch.equal(ctx512[:, P512 - a], ctx256[:, P256 - a]):
            idn_ok = False
            break
    gates["G5_age_token_identity"] = dict(
        ages=[A_LO, A_HI], ok=bool(idn_ok),
        wpe_eval512=[P512 - A_HI, P512 - A_LO], wpe_eval256=[P256 - A_HI, P256 - A_LO])
    log(f"G5 token identity across frames at ages {A_LO}-{A_HI}: "
        f"{'PASS' if idn_ok else 'FAIL'} (wpe {P512 - A_HI}..{P512 - A_LO} vs "
        f"{P256 - A_HI}..{P256 - A_LO})")

    keys512 = keys_for_ages(P512, AGES_BAND)
    keys256 = keys_for_ages(P256, AGES_BAND)
    full512, head512 = vnorm_tensors(vs512, keys512)
    full256, head256 = vnorm_tensors(vs256, keys256)

    # headline: per-seq mean over (layers x ages) of the concatenated v-norm
    vn_ratio, vn_ci = ratio_of_means(full256, full512)
    # secondary aggregate: mean of PER-HEAD norms (catches head redistribution
    # the concatenated norm can hide)
    ph_ratio, ph_ci = ratio_of_means(head256, head512)

    # per-layer and per-age tables
    per_layer = []
    for li in range(4):
        r, ci = ratio_of_means(full256[:, li], full512[:, li])
        per_layer.append(dict(layer=li, ratio=r, ci=ci,
                              mean_eval512=float(full512[:, li].mean()),
                              mean_eval256=float(full256[:, li].mean())))
    per_age = []
    for j, a in enumerate(AGES_BAND):
        r, ci = ratio_of_means(full256[:, :, j], full512[:, :, j])
        per_age.append(dict(age=a, ratio=r, ci=ci,
                            mean_eval512=float(full512[:, :, j].mean()),
                            mean_eval256=float(full256[:, :, j].mean())))

    # per layer x head ratio (mean over the age band, per-seq first)
    lh256 = head256.mean(2)                              # (N, L, H)
    lh512 = head512.mean(2)
    lh_ratio = (lh256.mean(0) / lh512.mean(0)).tolist()

    # ages 1-20 curve (context for the plot; the registered band is 4-17)
    ages20 = list(range(1, 21))
    c512, ch512 = vnorm_tensors(vs512, keys_for_ages(P512, ages20))
    c256, ch256 = vnorm_tensors(vs256, keys_for_ages(P256, ages20))
    cur_ratio = (c256.mean(0).mean(0) / c512.mean(0).mean(0))
    per_seq_ratio_curve = c256.mean(1) / np.maximum(c512.mean(1), 1e-30)
    rq_lo = np.percentile(per_seq_ratio_curve, 25, axis=0)
    rq_hi = np.percentile(per_seq_ratio_curve, 75, axis=0)

    # post-attention aggregated residual contribution of the ages-4-17 keys
    pre512, post512 = resid_contrib(net, vs512, ps512, keys512)
    pre256, post256 = resid_contrib(net, vs256, ps256, keys256)
    rc_ratio, rc_ci = ratio_of_means(post256, post512)
    rc_pre_ratio, rc_pre_ci = ratio_of_means(pre256, pre512)
    rc_per_layer = []
    for li in range(4):
        r, ci = ratio_of_means(post256[:, li], post512[:, li])
        rc_per_layer.append(dict(layer=li, ratio=r, ci=ci,
                                 mean_eval512=float(post512[:, li].mean()),
                                 mean_eval256=float(post256[:, li].mean())))

    log(f"V-NORM (ages 4-17, all layers, concatenated): 512 "
        f"{full512.mean():.4f} -> 256 {full256.mean():.4f} | ratio "
        f"{vn_ratio:.4f} CI [{vn_ci[0]:.4f},{vn_ci[1]:.4f}] | per-head-mean "
        f"ratio {ph_ratio:.4f} [{ph_ci[0]:.4f},{ph_ci[1]:.4f}]")
    log(f"per-layer V-norm ratios: "
        + ", ".join(f"L{p['layer']} {p['ratio']:.3f}" for p in per_layer))
    log(f"per-age V-norm ratios (4..17): "
        + ", ".join(f"{p['age']}:{p['ratio']:.3f}" for p in per_age))
    log(f"RESID-CONTRIB post-c_proj (ages 4-17): 512 {post512.mean():.4f} -> "
        f"256 {post256.mean():.4f} | ratio {rc_ratio:.4f} CI "
        f"[{rc_ci[0]:.4f},{rc_ci[1]:.4f}] | pre-c_proj {rc_pre_ratio:.4f}")
    log(f"resid-contrib per-layer ratios: "
        + ", ".join(f"L{p['layer']} {p['ratio']:.3f}" for p in rc_per_layer))

    # ============================================================ PART 2
    # ---- B=16 bootstrap of a* on eval-256 (12 fresh + the e069 4)
    log(f"PART 2: battery B — {B_X} fresh sequences (prompt seed "
        f"{SEED_PROMPT_X}, sampling seed {SEED_SAMPLE_X}; family = 202/7 "
        f"+100/+10)")
    gen_px = torch.Generator().manual_seed(SEED_PROMPT_X)
    ix12 = torch.randint(len(corp.val) - PROMPT_TOK - 1, (B_X,),
                         generator=gen_px)
    prompts12 = [corp.val[i:i + PROMPT_TOK] for i in ix12]
    log(f"  prompt4 prefix: {corp.decode(prompts12[0])[:32]!r}")
    gen_x = torch.Generator().manual_seed(SEED_SAMPLE_X)
    idx12, _ = generate_batch(net, prompts12, gen_x)
    idx16 = torch.cat([idx, idx12], 0)
    log(f"  battery B: {idx16.shape[0]} sequences = e069's 4 + {B_X} fresh")

    ctx256_16 = idx16[:, T_TOTAL - EVAL256:-1]
    tgt16 = idx16[:, -1]
    clean256_16 = manual_logits(net, ctx256_16, None, None, None, chunk=CHUNK_256)
    log("B=16 eval-256 fine onset fit (the registered B=16 readout)")
    d256_16, ce256_16 = sweep_block(net, ctx256_16, tgt16, clean256_16,
                                    CHUNK_256, pos_offset=0)
    dist256 = bootstrap_a_star_dist(d256_16["dce_age"])
    ds256 = dist_stats(dist256)
    log(f"  a* bootstrap: frac@6 {ds256['frac_eq_6']:.3f} | frac 12-24 "
        f"{ds256['frac_in_12_24']:.3f} | frac<=8 {ds256['frac_le_8']:.3f} | "
        f"frac>24 {ds256['frac_gt_24']:.3f} | median {ds256['median']:.0f}")

    # SECONDARY (unregistered contrast): eval-512 fit at B=16 — is the a*
    # fragility window-specific or does the 512-frame CI tighten too?
    log("SECONDARY: B=16 eval-512 fine onset fit (unregistered contrast)")
    ctx512_16 = idx16[:, :-1]
    clean512_16 = manual_logits(net, ctx512_16, None, None, None, chunk=CHUNK_512)
    d512_16, ce512_16 = sweep_block(net, ctx512_16, tgt16, clean512_16,
                                    CHUNK_512, pos_offset=0)
    dist512 = bootstrap_a_star_dist(d512_16["dce_age"])
    ds512 = dist_stats(dist512)
    log(f"  a* bootstrap: frac@6 {ds512['frac_eq_6']:.3f} | frac 12-24 "
        f"{ds512['frac_in_12_24']:.3f} | median {ds512['median']:.0f}")

    # ---- REGISTERED decision (frozen numbers)
    vn_in_band = bool(VN_LO <= vn_ratio <= VN_HI)
    ci16 = [float(np.nanpercentile(dist256, 2.5)),
            float(np.nanpercentile(dist256, 97.5))]
    straddle = bool(ci16[0] <= STRADDLE_LO and ci16[1] >= STRADDLE_HI)
    wide = bool((ci16[1] - ci16[0]) >= WIDE_BAR)
    tight = bool(ci16[0] >= TIGHT_LO and ci16[1] <= TIGHT_HI)
    clauses = dict(
        vnorm_ratio_outside_band=not vn_in_band,
        b16_ci_straddling=straddle, b16_ci_wide=wide,
        b16_ci_tight_12_20=tight)
    if not vn_in_band:
        clause = "H-VALUE-SIDE"
        verdict = (f"H-VALUE-SIDE: V-norm ratio at ages 4-17 = {vn_ratio:.3f} "
                   f"is OUTSIDE [{VN_LO},{VN_HI}] — the load genuinely grew in "
                   f"the value pathway")
    elif straddle or wide:
        clause = "H-THRESHOLD"
        verdict = (f"H-THRESHOLD: V-norms stable ({vn_ratio:.3f}, inside "
                   f"[{VN_LO},{VN_HI}]) AND the B=16 a* CI "
                   f"[{ci16[0]:.0f},{ci16[1]:.0f}] is still "
                   f"{'straddling' if straddle else 'wide'} — instrument "
                   f"fragility, not a load property")
    elif tight:
        clause = "NEITHER-SIMPLE-STORY"
        verdict = (f"NEITHER simple story: V-norms stable ({vn_ratio:.3f}) AND "
                   f"the B=16 CI [{ci16[0]:.0f},{ci16[1]:.0f}] is tight around "
                   f"12-20 — a real, stable, attention-invariant, "
                   f"value-norm-invariant liveness shift (residual-scale/LN "
                   f"story to register next)")
    else:
        clause = "INDETERMINATE"
        verdict = (f"V-norms stable ({vn_ratio:.3f}) and the B=16 CI "
                   f"[{ci16[0]:.0f},{ci16[1]:.0f}] matches neither the "
                   f"wide/straddling nor the tight-around-12-20 clause — "
                   f"reported honestly, no forcing")
    log(f"REGISTERED DECISION: {verdict}")
    log(f"  clauses: {clauses}")

    # ---------------------------------------------------------------- metrics
    def sweep_summary(d, ce, role, wpe):
        return dict(
            role=role, wpe=wpe, a_star=d["a_star"],
            a_star_naive=d["a_star_naive"], a_star_robust=d["a_star_robust"],
            a_star_ci=d["a_star_ci"], per_seq_a_star=list(d["per_seq_a_star"]),
            live_frac=d["live_frac"],
            ages1_5=[float(a) for a in d["mean_age_curve"][:5]],
            ages6_17=[float(a) for a in d["mean_age_curve"][5:17]],
            mean_curve=d["mean_age_curve"].tolist(),
            ci_lo=d["ci"][0].tolist(), ci_hi=d["ci"][1].tolist(),
            clean_ce_per_seq=ce.tolist(), clean_ce_mean=float(np.mean(ce)),
        )

    metrics = dict(
        experiment="e072_value_vs_threshold",
        purpose="T044's registered value-side vs threshold discriminator on "
                "the frozen e053c ctx-512 net, e069/e070 protocol verbatim: "
                "(1) V-vector norms at ages 4-17 (value-projection outputs, "
                "per layer) eval-256 vs eval-512 on the SAME sequences + the "
                "post-attention aggregated residual contribution of those "
                "keys; (2) B=16 bootstrap of a* on eval-256 (battery = "
                "e069's 4 + 12 fresh family-extended sequences). FROZEN: "
                "V-norm ratio outside [0.9,1.1] => H-value-side; stable AND "
                "B=16 CI wide/straddling => H-threshold; stable AND CI tight "
                "around 12-20 => NEITHER simple story (report honestly).",
        started=started, wall_s=elapsed(), threads=THREADS, cpu_only=True,
        net=dict(ckpt=str(CKPT), arch=dict(n_layer=4, n_head=4, n_embd=128,
                                           block_size=T_TOTAL, vocab=65),
                 params=n_params, steps_in_ckpt=steps_in_ckpt,
                 val_ce=val_ce, val_ce_e053c=E053C_VAL_CE),
        seeds=dict(
            corpus=1337,
            battery_A=dict(prompts=SEED_PROMPT, sampling=SEED_SAMPLE,
                           note="e069/e070 verbatim: 8-draw seed 202, first "
                                "B=4; seed-7 B=4 row-order sampling stream"),
            battery_B_extra12=dict(
                prompts=SEED_PROMPT_X, sampling=SEED_SAMPLE_X,
                note="202/7 family extended deterministically: +100 on the "
                     "prompt seed (202->302), +10 on the sampling seed "
                     "(7->17); SAME draw rules (val split, 64-token prompts, "
                     "randint high len(val)-65, temp 0.8 top-k 40, fixed "
                     "anchor, free-run 64->512); 12-draw + B=12 row-order "
                     "stream; battery B = battery A's 4 + these 12"),
            bootstrap=0),
        protocol=dict(B=B, B16=int(idx16.shape[0]), prompt_tokens=PROMPT_TOK,
                      t_total=T_TOTAL, temp=TEMP, topk=TOPK, fixed_anchor=True,
                      threshold=THRESH, a_star_window=A_STAR_K, boot_n=BOOT_N,
                      chunks=dict(w512=CHUNK_512, w256=CHUNK_256),
                      eval_window="e069/e070 verbatim: same net "
                                  "(e053c_ctx512.pt), same seeds 202/7, same "
                                  "B=4 sequences, same generation/sweep/onset "
                                  "machinery; eval-256 = last-256-positions "
                                  "re-indexed (wpe 0..254)",
                      vnorm="L2 norm of the concatenated C=128 "
                            "value-projection output (pre head-split norm "
                            "over H*d), per layer per key; headline = "
                            "per-seq mean over layers x ages 4-17, ratio of "
                            "means over sequences, paired bootstrap CI",
                      resid_contrib="per layer, ||c_proj(sum_k p_final[k] "
                                    "* v[k])|| over the ages-4-17 keys — "
                                    "the keys' share of the final query's "
                                    "residual write"),
        gates=gates,
        replica_sweeps=dict(
            eval512=sweep_summary(d512, ce512,
                                  "e069 sweep A replica (baseline)",
                                  "native 0..510"),
            eval256=sweep_summary(d256, ce256,
                                  "e069 sweep B replica (the D1 frame)",
                                  "re-indexed 0..254"),
        ),
        value_side=dict(
            ages=[A_LO, A_HI],
            frames=dict(eval512="ctx=idx[:, :-1], wpe 0..510 (ages 4-17 -> "
                                "wpe 494..507)",
                        eval256="ctx=idx[:, 256:511], wpe re-indexed 0..254 "
                                "(ages 4-17 -> wpe 238..251)"),
            headline=dict(
                mean_vnorm_eval512=float(full512.mean()),
                mean_vnorm_eval256=float(full256.mean()),
                per_seq_eval512=full512.mean((1, 2)).tolist(),
                per_seq_eval256=full256.mean((1, 2)).tolist(),
                ratio=vn_ratio, ratio_ci=vn_ci,
                ci_excludes_1=bool(not (vn_ci[0] <= 1.0 <= vn_ci[1])),
                in_band=vn_in_band),
            per_head_mean=dict(ratio=ph_ratio, ratio_ci=ph_ci,
                               note="secondary aggregate: mean of PER-HEAD "
                                    "norms (catches head redistribution the "
                                    "concatenated norm can hide)"),
            per_layer=per_layer, per_age=per_age,
            per_layerhead_ratio=lh_ratio,
            age_curve_1_20=dict(ages=ages20, ratio_of_means=cur_ratio.tolist(),
                                per_seq_quartiles=[rq_lo.tolist(),
                                                   rq_hi.tolist()],
                                mean_eval512=c512.mean((0, 1)).tolist(),
                                mean_eval256=c256.mean((0, 1)).tolist()),
            resid_contrib=dict(
                post_cproj=dict(mean_eval512=float(post512.mean()),
                                mean_eval256=float(post256.mean()),
                                ratio=rc_ratio, ratio_ci=rc_ci,
                                per_layer=rc_per_layer),
                pre_cproj=dict(ratio=rc_pre_ratio, ratio_ci=rc_pre_ci)),
        ),
        bootstrap_b16=dict(
            eval256=dict(
                role="REGISTERED B=16 readout (T044 discriminator 2)",
                summary=sweep_summary(d256_16, ce256_16,
                                      "B=16 eval-256 fit", "re-indexed 0..254"),
                a_star=d256_16["a_star"],
                a_star_ci=ci16,
                ci_from="percentiles of the 1000-resample a* distribution "
                        "(same rng stream as bootstrap_stat)",
                dist=ds256,
                edge=dict(frac_at_6=ds256["frac_eq_6"],
                          frac_in_12_24=ds256["frac_in_12_24"],
                          note="the 5-pt robust rule's edge: how often the "
                               "resampled a* lands AT 6 (the eval-512 "
                               "family value) vs inside 12-24 (the eval-256 "
                               "family neighborhood)")),
            eval512=dict(
                role="SECONDARY (unregistered contrast): B=16 eval-512 fit — "
                     "is the a* fragility window-specific?",
                summary=sweep_summary(d512_16, ce512_16,
                                      "B=16 eval-512 fit", "native 0..510"),
                a_star=d512_16["a_star"],
                a_star_ci=[float(np.nanpercentile(dist512, 2.5)),
                           float(np.nanpercentile(dist512, 97.5))],
                dist=ds512),
            b4_reference=dict(
                eval256=dict(a_star=d256["a_star"], ci=d256["a_star_ci"],
                             per_seq=list(d256["per_seq_a_star"])),
                eval512=dict(a_star=d512["a_star"], ci=d512["a_star_ci"],
                             per_seq=list(d512["per_seq_a_star"]))),
        ),
        registered_decision=dict(
            frozen_rules=dict(
                h_value_side=f"V-norm ratio (ages 4-17) outside "
                             f"[{VN_LO},{VN_HI}]",
                h_threshold=f"V-norms stable AND B=16 a* CI wide "
                            f"(width >= {WIDE_BAR:g}) or straddling "
                            f"(lo <= {STRADDLE_LO:g} AND hi >= "
                            f"{STRADDLE_HI:g})",
                neither=f"V-norms stable AND B=16 CI tight inside "
                        f"[{TIGHT_LO:g},{TIGHT_HI:g}]"),
            vnorm_ratio=vn_ratio, vnorm_ratio_ci=vn_ci,
            vnorm_in_band=vn_in_band,
            b16_a_star=d256_16["a_star"], b16_ci=ci16,
            b16_ci_width=ci16[1] - ci16[0],
            b16_straddling=straddle, b16_wide=wide, b16_tight_12_20=tight,
            frac_at_6=ds256["frac_eq_6"], frac_in_12_24=ds256["frac_in_12_24"],
            clauses_fired=clauses, clause=clause, verdict=verdict),
    )
    save_json(out_dir / "metrics.json", metrics)
    log("metrics.json written")
    plot(out_dir / "value_vs_threshold.png", metrics, dist256, dist512,
         cur_ratio, rq_lo, rq_hi)
    log(f"plot written; total wall {elapsed():.0f}s")


# ---------------------------------------------------------------------- plot


def plot(path: Path, M: dict, dist256: np.ndarray, dist512: np.ndarray,
         cur_ratio: np.ndarray, rq_lo: np.ndarray, rq_hi: np.ndarray):
    vs = M["value_side"]
    dec = M["registered_decision"]
    b16 = M["bootstrap_b16"]
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    # ---- panel 1: V-norm ratio vs age (the registered readout)
    ages20 = np.asarray(vs["age_curve_1_20"]["ages"])
    ax1.fill_between(ages20, rq_lo, rq_hi, alpha=0.25, color="tab:blue",
                     label="per-seq quartiles (B=4)")
    ax1.plot(ages20, cur_ratio, lw=1.4, color="tab:blue",
             label="ratio of means (all layers)")
    ax1.axhspan(VN_LO, VN_HI, color="tab:green", alpha=0.10)
    ax1.axhline(1.0, color="k", lw=0.8)
    for y in (VN_LO, VN_HI):
        ax1.axhline(y, color="tab:green", ls="--", lw=0.9)
    ax1.axvspan(A_LO, A_HI, color="tab:red", alpha=0.08)
    ax1.text((A_LO + A_HI) / 2, ax1.get_ylim()[0] + 0.002,
             "registered band 4-17", ha="center", fontsize=8, color="tab:red")
    hl = vs["headline"]
    ax1.plot([np.mean([A_LO, A_HI])], [hl["ratio"]], "o", ms=8, color="tab:red",
             zorder=5)
    ax1.annotate(f"headline r={hl['ratio']:.3f}\nCI [{hl['ratio_ci'][0]:.3f},"
                 f"{hl['ratio_ci'][1]:.3f}]",
                 (np.mean([A_LO, A_HI]), hl["ratio"]), textcoords="offset points",
                 xytext=(10, -28), fontsize=8.5)
    ax1.set_xlabel("cache age (tokens)")
    ax1.set_ylabel("V-vector norm ratio (eval-256 / eval-512)")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_title("E072-1 — value-projection output norms, same tokens both "
                  "windows\n" + dec["verdict"], fontsize=9.5)

    # ---- panel 2: per-layer ratios (V-norm + resid contribution)
    L = len(vs["per_layer"])
    x = np.arange(L)
    r_v = [p["ratio"] for p in vs["per_layer"]]
    e_v = [[p["ratio"] - p["ci"][0] for p in vs["per_layer"]],
           [p["ci"][1] - p["ratio"] for p in vs["per_layer"]]]
    rc = vs["resid_contrib"]["post_cproj"]["per_layer"]
    r_c = [p["ratio"] for p in rc]
    e_c = [[p["ratio"] - p["ci"][0] for p in rc],
           [p["ci"][1] - p["ratio"] for p in rc]]
    ax2.bar(x - 0.19, r_v, 0.38, color="tab:blue", alpha=0.85,
            label="V-vector norm (pre-attention)")
    ax2.errorbar(x - 0.19, r_v, yerr=e_v, fmt="none", ecolor="k", lw=0.8,
                 capsize=2)
    ax2.bar(x + 0.19, r_c, 0.38, color="tab:orange", alpha=0.85,
            label="resid contribution ||c_proj(sum p*v)||")
    ax2.errorbar(x + 0.19, r_c, yerr=e_c, fmt="none", ecolor="k", lw=0.8,
                 capsize=2)
    ax2.axhspan(VN_LO, VN_HI, color="tab:green", alpha=0.10)
    ax2.axhline(1.0, color="k", lw=0.8)
    ax2.set_xticks(x, [f"layer {i}" for i in range(L)])
    ax2.set_ylabel("ratio (eval-256 / eval-512), ages 4-17")
    ax2.legend(fontsize=8)
    ax2.set_title("per-layer value-pathway ratios (ages 4-17, mean over "
                  "B=4 seqs, bootstrap CI)", fontsize=9.5)

    # ---- panel 3: per-layer x head V-norm ratio heatmap
    R = np.asarray(vs["per_layerhead_ratio"])
    im = ax3.imshow(R, cmap="RdBu_r", vmin=0.75, vmax=1.25, aspect="auto")
    for li in range(R.shape[0]):
        for hi in range(R.shape[1]):
            v = R[li, hi]
            ax3.text(hi, li, f"{v:.2f}", ha="center", va="center", fontsize=8,
                     color="white" if abs(v - 1.0) > 0.22 else "black",
                     fontweight="bold" if not (VN_LO <= v <= VN_HI) else "normal")
    ax3.set_xticks(range(R.shape[1]), [f"h{h}" for h in range(R.shape[1])])
    ax3.set_yticks(range(R.shape[0]), [f"layer {l}" for l in range(R.shape[0])])
    ax3.set_title("V-norm ratio per layer x head (ages 4-17; bold = outside "
                  f"[{VN_LO},{VN_HI}])", fontsize=9.5)
    fig.colorbar(im, ax=ax3, fraction=0.046)

    # ---- panel 4: B=16 bootstrap a* distribution on eval-256
    d = dist256[~np.isnan(dist256)]
    clip = np.minimum(d, 40)
    ax4.hist(clip, bins=np.arange(0.5, 41.5, 1.0), color="tab:blue",
             alpha=0.8, edgecolor="white")
    ax4.axvline(6, color="tab:red", lw=1.6, ls="--")
    ax4.text(6.3, ax4.get_ylim()[1] * 0.92, "a* = 6 (eval-512 family)",
             fontsize=8, color="tab:red")
    ax4.axvspan(12, 24, color="tab:orange", alpha=0.12)
    ax4.text(18, ax4.get_ylim()[1] * 0.80, "12-24 (eval-256 family)",
             fontsize=8, ha="center", color="darkorange")
    s256 = b16["eval256"]
    ax4.axvline(s256["a_star"], color="k", lw=1.4)
    for q in s256["a_star_ci"]:
        ax4.axvline(q, color="k", lw=0.9, ls=":")
    ax4.set_xlabel("bootstrapped a* (1000 resamples of B=16, eval-256)")
    ax4.set_ylabel("resamples")
    ax4.text(0.98, 0.60,
             f"B=16 point a* = {s256['a_star']}\nCI [{s256['a_star_ci'][0]:.0f},"
             f"{s256['a_star_ci'][1]:.0f}] (width "
             f"{s256['a_star_ci'][1] - s256['a_star_ci'][0]:.0f})\n"
             f"frac@6 = {dec['frac_at_6']:.3f}\nfrac 12-24 = "
             f"{dec['frac_in_12_24']:.3f}\nfrac<=8 = "
             f"{s256['dist']['frac_le_8']:.3f}\n"
             f"(B=4 ref: a* 18 CI [7,30])",
             transform=ax4.transAxes, fontsize=8, ha="right", va="top",
             family="monospace")
    ax4.set_title("E072-2 — the 5-pt robust rule's edge under B=16 "
                  "(eval-256)\nstraddling: "
                  f"{dec['b16_straddling']}, wide: {dec['b16_wide']}, "
                  f"tight 12-20: {dec['b16_tight_12_20']}", fontsize=9.5)

    # ---- panel 5: mean dCE curves (the haze crossing the rule)
    r4_512 = M["replica_sweeps"]["eval512"]
    r4_256 = M["replica_sweeps"]["eval256"]
    s16_256 = s256["summary"]
    zoom = 60
    a4 = np.arange(1, len(r4_512["mean_curve"]) + 1)
    ax5.plot(a4[:zoom], r4_512["mean_curve"][:zoom], lw=1.3, color="tab:red",
             label=f"B=4 eval-512 (a*={r4_512['a_star']})")
    ax5.plot(a4[:zoom], r4_256["mean_curve"][:zoom], lw=1.3, color="tab:blue",
             label=f"B=4 eval-256 (a*={r4_256['a_star']})")
    a16 = np.arange(1, len(s16_256["mean_curve"]) + 1)
    ax5.fill_between(a16[:zoom], np.asarray(s16_256["ci_lo"])[:zoom],
                     np.asarray(s16_256["ci_hi"])[:zoom], alpha=0.18,
                     color="tab:cyan")
    ax5.plot(a16[:zoom], s16_256["mean_curve"][:zoom], lw=1.5,
             color="tab:cyan",
             label=f"B=16 eval-256 (a*={s16_256['a_star']})")
    ax5.axhline(THRESH, color="gray", ls=":", lw=1.1)
    ax5.text(zoom, THRESH, f" thresh {THRESH}", fontsize=8, va="bottom",
             ha="right", color="gray")
    ax5.axhline(0, color="k", lw=0.6)
    ax5.set_xlabel("cache age (tokens)")
    ax5.set_ylabel("V-zero dCE (nats, mean over seqs)")
    ax5.legend(fontsize=8)
    ax5.set_title("the dCE haze vs the 0.01-nat 5-pt rule — same net, same "
                  "rule, three fits", fontsize=9.5)

    # ---- panel 6: SECONDARY — B=16 eval-512 contrast
    d5 = dist512[~np.isnan(dist512)]
    ax6.hist(np.minimum(d5, 40), bins=np.arange(0.5, 41.5, 1.0),
             color="tab:red", alpha=0.8, edgecolor="white")
    s512 = b16["eval512"]
    ax6.axvline(s512["a_star"], color="k", lw=1.4)
    for q in s512["a_star_ci"]:
        ax6.axvline(q, color="k", lw=0.9, ls=":")
    ax6.axvspan(4, 8, color="tab:green", alpha=0.10)
    ax6.text(6, ax6.get_ylim()[1] * 0.92, "e053c/e069\nwindow [4,8]",
             fontsize=8, ha="center", color="tab:green")
    ax6.set_xlabel("bootstrapped a* (1000 resamples of B=16, eval-512)")
    ax6.set_ylabel("resamples")
    ax6.text(0.98, 0.75,
             f"B=16 point a* = {s512['a_star']}\nCI "
             f"[{s512['a_star_ci'][0]:.0f},{s512['a_star_ci'][1]:.0f}]\n"
             f"frac@6 = {s512['dist']['frac_eq_6']:.3f}\n"
             f"(B=4 ref: a* 6 CI [4,8])",
             transform=ax6.transAxes, fontsize=8, ha="right", va="top",
             family="monospace")
    ax6.set_title("SECONDARY — same B=16 battery on eval-512 (unregistered "
                  "contrast): fragility window-specific?", fontsize=9.5)

    fig.suptitle("E072 — T044 value-side vs threshold | clause: "
                 f"{dec['clause']} | V-norm r={dec['vnorm_ratio']:.3f} "
                 f"[{dec['vnorm_ratio_ci'][0]:.3f},{dec['vnorm_ratio_ci'][1]:.3f}]"
                 f" | B=16 a*={dec['b16_a_star']} CI "
                 f"[{dec['b16_ci'][0]:.0f},{dec['b16_ci'][1]:.0f}]",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
