"""E070 — T039-amendment's REGISTERED attention-mass discriminator (INTERPRETER PASS).

Trigger: scratch/interpretation_t039amendment.md §4 (the interpreter-pass
discriminator that supersedes the card's original one) + THINKING.md
"T039-amendment" (E069: spike is circuit, onset statistic is
eval-window-sensitive). E069 found, on the SAME net/sequences: a*(eval-512)
= 6 [4,8] vs a*(eval-256, re-indexed) = 18 [7,30]; shoulder (ages 4-5)
COLLAPSED (0.91/0.32 -> 0.06/0.13 nats) while a window-start spike RESURRECTED
(+0.699 nats at age 255 / wpe row 0; dead +0.007 at the same token in the
512-frame). Two of the candidate explanations make attention-level claims:
H-wpe-domain (C1: row-specific positional reshape) vs H-instrument (C2:
statistic-only) vs H-redistribute (C3: proportional survivor re-scaling).

THIS RUN (eval-only, CPU, minutes): rebuild e069's exact eval-window protocol
(same checkpoint runs/checkpoints/e053c_ctx512.pt, same seeds 202/7, same B=4
sequences, same generation/sweep/onset machinery VERBATIM) and measure, per
layer x head, the FINAL QUERY's attention-received mass under eval-512
(ctx = idx[:, :-1], wpe 0..510) vs eval-256 (ctx = idx[:, 256:511], wpe
re-indexed 0..254) on the SAME sequences, in buckets:
  ages 1-20 (recent), age 4, age 5 (the collapsed shoulder),
  window-start token (seq position 256: age 255 in BOTH frames — wpe row 256
  in the 512-frame, wpe row 0 in the 256-frame — the F2 conjunction cell),
plus K-drop vs V-zero lesion dCE at age 4 / age 5 / window-start under both
windows (all-layer whole-position lesions, e069 instrument).

REGISTERED NUMBERS (frozen before running, §4 of the interpretation doc):
  mass ratio (eval-256 / eval-512), total mass summed over all 16 layer x
  head, mean over B=4 sequences, ratio of means:
  - window-start ratio >= 2.0 AND age-4 ratio <= 0.75  => H-wpe-domain
    (positional reshape; C1's extra sign checks: ages 1-3 within +/-20%,
    window-start K-drop >> V-zero if anchor/sink mass);
  - ALL of {ages1-20, age4, age5, window-start} in [0.9, 1.1] => H-instrument
    (the dCE reshape is a statistic, not an attention change);
  - ALL of the same set > 1.1 (uniform inflation ~ 1/(1 - far mass))
    => H-redistribute (contributory);
  - anything else => MIXED, reported honestly.

SECONDARY (free, same run): a* fit on the NATIVE positions-0-255 slice —
ctx = idx[:, :255] (native wpe 0..254), target = position 255's token, which
was generated with exactly that history (in-distribution at generation).
If a*(native-256) ~ 6 with a dead window start, the D1 effect is specific to
truncating a LONG window (code band / composition), not window length per se.

Run:     python lab/e070_attention_mass.py
Outputs: runs/e070/metrics.json + runs/e070/attention_mass.png
Envelope: NO training, NO new automations; CPU-only, minutes. No
NOTES/THINKING/QUEUE/STATE edits; no commit (the dispatcher owns those).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # task spec: CPU-only (pre-torch)
import torch

# this torch build reports is_available()==True even with an empty
# CUDA_VISIBLE_DEVICES; force CPU so common.DEVICE=="cpu" (e053b/e069 pattern)
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
SEED_PROMPT, SEED_SAMPLE = 202, 7         # e053b/e053c/e069 exact seeds
N_PROMPTS = 8
B = 4                                     # e053b/e069 uniform B
BOOT_N = 1000
THREADS = 12
torch.set_num_threads(THREADS)

EVAL256 = 256
P256 = EVAL256 - 1                        # 255 positions in the eval-256 window
P512 = T_TOTAL - 1                        # 511 positions in the eval-512 window
THRESH = 0.01                             # dead-weight threshold (nats)
A_STAR_K = 5                              # sustained window (e053b verbatim)
CHUNK_512 = 64                            # e069 measured chunks
CHUNK_256 = 128

WS_AGE = 255                              # window-start token's age (both frames)
WS_SEQ_POS = T_TOTAL - EVAL256            # 256: seq position of the shared token

# ---- REGISTERED decision numbers (FROZEN, scratch §4) ----------------------
R_WS_BAR = 2.0                            # window-start ratio >=
R_AGE4_BAR = 0.75                         # age-4 ratio <=
INSTR_LO, INSTR_HI = 0.9, 1.1             # all-ratios-in band
REDIST_BAR = 1.1                          # uniform-ratio floor
DECISION_SET = ["ages1_20", "age4", "age5", "window_start"]

# ---- e069 stored readouts (protocol-identity instrument gates) -------------
# winstart_dce = age-255 entry of e069's stored vzero_dce_mean curves (the
# window-start token in EACH frame's own coordinates: eval-512 age 255 wears
# wpe row 256 = +0.0069; eval-256 age 255 wears wpe row 0 = +0.6986; the
# eval-512 curve's LAST entry, -0.0655, is age 511 = the true sequence start,
# a different object — see scratch/interpretation_t039amendment.md F2)
E069_REF = {
    "eval512": {"a_star": 6, "ci": [4.0, 8.0], "clean_ce": 0.4528,
                "ages1_5": [1.742, 4.467, 2.319, 0.914, 0.321],
                "winstart_dce": 0.0069},
    "eval256": {"a_star": 18, "ci": [7.0, 30.0], "clean_ce": 0.4587,
                "ages1_5": [2.715, 4.888, 2.349, 0.064, 0.131],
                "winstart_dce": 0.6986},
}
CKPT = REPO / "runs" / "checkpoints" / "e053c_ctx512.pt"
E053C_VAL_CE = 1.5226792494455974
T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# ------------------------------------------------- machinery (VERBATIM e069)

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
    draws from the shared seed-7 CPU generator in row order per step
    (e053b-identical stream math). Returns idx (B, T) and the final
    (position-T_TOTAL-1) logits (B, V). [CPU-only e069 verbatim]"""
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


# --------------------------------------------- NEW: final-query attention probe

@torch.no_grad()
def attn_probe(net: TinyGPT, idxs: torch.Tensor):
    """Full manual forward that ALSO records, per layer x head, the attention
    probabilities of the FINAL position's query over all T keys (causal
    softmax row; the query attends to keys 0..T-1 including itself).
    Returns probs (N, L, H, T) and the final-position logits (N, vocab).
    Same op order as _manual_chunk (G4 checks them equal)."""
    N, T = idxs.shape
    pos = torch.arange(T)
    x = net.wte(idxs) + net.wpe(pos).unsqueeze(0)
    H = net.cfg.n_head
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    L = len(net.h)
    probs = torch.empty(N, L, H, T)
    for li, blk in enumerate(net.h):
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
        probs[:, li] = p[:, :, -1, :]                  # final query's row
        y = (p @ v).transpose(1, 2).reshape(N, T, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
    return probs, net.lm_head(net.ln_f(x[:, -1, :]))


def age_keys(Pw: int, a_lo: int, a_hi: int) -> torch.Tensor:
    """Context key positions for ages a_lo..a_hi (age of key p = Pw - p)."""
    return torch.arange(Pw - a_hi, Pw - a_lo + 1)


def bucket_mass(probs: torch.Tensor, keys: torch.Tensor) -> np.ndarray:
    """probs (B, L, H, T) -> (B, L, H) summed mass on the given key set."""
    return probs.index_select(-1, keys).sum(-1).numpy().astype(np.float64)


def ratio_block(m256: np.ndarray, m512: np.ndarray):
    """m256/m512: (B, ...) bucket masses under the two frames. Returns
    per-frame mean over sequences, ratio of means, bootstrap CI, per-seq
    ratios (on the total summed over trailing axes)."""
    t256 = m256.reshape(m256.shape[0], -1).sum(1)
    t512 = m512.reshape(m512.shape[0], -1).sum(1)
    mean256, mean512 = float(t256.mean()), float(t512.mean())
    ratio = mean256 / max(mean512, 1e-30)
    rng = np.random.default_rng(0)
    S = len(t256)
    vals = []
    for _ in range(BOOT_N):
        sel = rng.integers(0, S, S)
        vals.append(t256[sel].mean() / max(t512[sel].mean(), 1e-30))
    lo, hi = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    per_seq = (t256 / np.maximum(t512, 1e-30)).tolist()
    per_layer_ratio = (m256.mean(0).sum(-1) / np.maximum(m512.mean(0).sum(-1), 1e-30)).tolist()
    return dict(
        mass_eval512=mean512, mass_eval256=mean256, ratio=ratio,
        ratio_ci=[lo, hi], per_seq_ratio=per_seq,
        per_layer_eval512=m512.mean(0).sum(-1).tolist(),
        per_layer_eval256=m256.mean(0).sum(-1).tolist(),
        per_layer_ratio=per_layer_ratio,
        per_layerhead_eval512=m512.mean(0).tolist(),
        per_layerhead_eval256=m256.mean(0).tolist(),
        per_layerhead_ratio=(m256.mean(0) / np.maximum(m512.mean(0), 1e-30)).tolist(),
    )


@torch.no_grad()
def probe_dce(net, ctx, tgt, clean_logits, key_pos: int, ch: int,
              pos_offset: int = 0):
    """K-drop vs V-zero dCE (nats, per sequence) at ONE context position.
    All-layer whole-position lesions — the e069/e053b instrument verbatim
    (kdrop masks the key for every query except the diagonal, every layer,
    every head; vzero zeroes the value vector at every layer, every head)."""
    Bs, Pw = ctx.shape
    rows = torch.arange(Bs)
    p_clean = torch.softmax(clean_logits.float(), -1)
    ce_clean = -torch.log(p_clean[rows, tgt].clamp_min(1e-12))
    out = {}
    for kind in ("vzero", "kdrop"):
        mask = torch.zeros(Bs, Pw, dtype=torch.bool)
        mask[:, key_pos] = True
        les = manual_logits(net, ctx,
                            mask if kind == "vzero" else None,
                            mask if kind == "kdrop" else None,
                            None, chunk=ch, pos_offset=pos_offset)
        lp = torch.log_softmax(les.float(), -1)
        dce = (-ce_clean - lp[rows, tgt])
        out[kind] = dce.numpy()
    return out


def agg_dce(per_seq: np.ndarray):
    per_seq = np.asarray(per_seq, float)
    lo, hi = boot_ci(per_seq)
    return dict(mean=float(per_seq.mean()), ci=[float(lo), float(hi)],
                per_seq=per_seq.tolist())


# ---------------------------------------------------------------------- main


def main():
    assert torch.cuda.device_count() == 0, "CPU-only violated: CUDA visible"
    assert common.DEVICE == "cpu", f"common.DEVICE={common.DEVICE} != cpu"
    out_dir = run_dir("e070")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gates = dict(cpu_only=True, threads=THREADS, smoke=False)

    # ---- battery: rebuild corpus + net EXACTLY as e053c/e069 did
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

    # ---- sequences: e069 protocol verbatim (seeds 202/7, B=4)
    gen_p = torch.Generator().manual_seed(SEED_PROMPT)
    ix = torch.randint(len(corp.val) - PROMPT_TOK - 1, (N_PROMPTS,),
                       generator=gen_p)
    prompts = [corp.val[i:i + PROMPT_TOK] for i in ix]
    log(f"{N_PROMPTS} prompts (seed {SEED_PROMPT}); using first B={B}; "
        f"prompt0 prefix: {corp.decode(prompts[0])[:32]!r}")
    gen = torch.Generator().manual_seed(SEED_SAMPLE)     # per-cell seed-7 re-init
    idx, final_logits = generate_batch(net, prompts[:B], gen)
    log(f"generated {B} seqs 64->{T_TOTAL} (fixed anchor, CPU)")

    # G0b: batched incremental-KV final logits vs full recompute
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
    ctx_nat = idx[:, :P256]                              # seq 0..254, native wpe 0..254
    tgt_nat = idx[:, P256]                               # seq 255 (generated w/ that history)
    assert torch.equal(ctx256[:, 0], idx[:, WS_SEQ_POS])
    assert torch.equal(ctx256[:, -1], idx[:, T_TOTAL - 2])
    assert torch.equal(tgt, idx[:, -1])

    log("sweep 1/3: eval-512 fine onset fit (e069 sweep A replica)")
    d512, ce512 = sweep_block(net, ctx512, tgt, final_logits, CHUNK_512)
    log("sweep 2/3: eval-256 re-indexed fine onset fit (e069 sweep B replica)")
    clean256 = manual_logits(net, ctx256, None, None, None, chunk=CHUNK_256)
    d256, ce256 = sweep_block(net, ctx256, tgt, clean256, CHUNK_256,
                              pos_offset=0)
    log("sweep 3/3 (SECONDARY): native positions-0-255 slice onset fit")
    clean_nat = manual_logits(net, ctx_nat, None, None, None, chunk=CHUNK_256)
    dnat, ce_nat = sweep_block(net, ctx_nat, tgt_nat, clean_nat, CHUNK_256,
                               pos_offset=0)

    # G3: instrument / protocol-identity gates vs e069's stored readouts
    for key, d, ce in [("eval512", d512, ce512), ("eval256", d256, ce256)]:
        ref = E069_REF[key]
        ages15 = d["mean_age_curve"][:5]
        dev15 = float(np.max(np.abs(ages15 - np.asarray(ref["ages1_5"]))))
        ws_dev = abs(float(d["mean_age_curve"][WS_AGE - 1]) - ref["winstart_dce"])
        a_match = d["a_star"] == ref["a_star"]
        ce_dev = abs(float(ce.mean()) - ref["clean_ce"])
        ok = bool(a_match and dev15 < 0.05 and ws_dev < 0.05 and ce_dev < 0.02)
        gates[f"G3_{key}_vs_e069"] = dict(
            a_star=d["a_star"], a_star_ref=ref["a_star"], a_star_match=a_match,
            ages1_5=[float(a) for a in ages15], ages1_5_ref=ref["ages1_5"],
            max_dev_ages1_5=dev15, winstart_dce=float(d["mean_age_curve"][WS_AGE - 1]),
            winstart_dce_ref=ref["winstart_dce"], winstart_dev=ws_dev,
            clean_ce=float(ce.mean()), clean_ce_ref=ref["clean_ce"],
            clean_ce_dev=ce_dev, ok=ok)
        log(f"G3 {key}: a* {d['a_star']} (ref {ref['a_star']}), ages1-5 maxdev "
            f"{dev15:.4f}, window-start {d['mean_age_curve'][WS_AGE - 1]:.3f} "
            f"(ref {ref['winstart_dce']:.3f}), CE dev {ce_dev:.4f} "
            f"-> {'PASS' if ok else 'FAIL'}")

    # ---- attention-received mass of the final query, per layer x head
    log("attention probe: final-query per-layer/head mass, eval-512 frame")
    probs512, lg512p = attn_probe(net, ctx512)
    log("attention probe: eval-256 re-indexed frame")
    probs256, lg256p = attn_probe(net, ctx256)
    probs_nat, lgnatp = attn_probe(net, ctx_nat)

    # G4: probe forward logits == manual recompute (same op order)
    with torch.no_grad():
        ref512 = manual_logits(net, ctx512, None, None, None, chunk=CHUNK_512)
        ref256 = manual_logits(net, ctx256, None, None, None, chunk=CHUNK_256)
    g4 = float(max((lg512p - ref512).abs().max(), (lg256p - ref256).abs().max()))
    gates["G4_probe_vs_manual"] = dict(max_logit_dev=g4, ok=bool(g4 < 1e-5))
    rs512 = float(probs512.sum(-1).sub(1.0).abs().max())
    rs256 = float(probs256.sum(-1).sub(1.0).abs().max())
    gates["G5_prob_rows_sum1"] = dict(max_dev=max(rs512, rs256),
                                      ok=bool(max(rs512, rs256) < 1e-5))
    log(f"G4 probe-vs-manual logits dev {g4:.2e}; G5 prob-row dev "
        f"{max(rs512, rs256):.2e} -> "
        f"{'PASS' if g4 < 1e-5 and max(rs512, rs256) < 1e-5 else 'FAIL'}")

    # buckets: shared ages across frames (both windows end at the same final
    # query; ages align to the SAME underlying tokens)
    ws512, ws256 = P512 - WS_AGE, P256 - WS_AGE          # 256 and 0
    bucket_defs = {
        "ages1_20": (1, 20), "ages1_3": (1, 3), "age4": (4, 4), "age5": (5, 5),
        "ages6_17": (6, 17), "ages21_255": (21, 255),
    }
    buckets = {}
    for name, (a_lo, a_hi) in bucket_defs.items():
        m512 = bucket_mass(probs512, age_keys(P512, a_lo, a_hi))
        m256 = bucket_mass(probs256, age_keys(P256, a_lo, a_hi))
        blk = ratio_block(m256, m512)
        blk["ages"] = [a_lo, a_hi]
        blk["keys_eval512"] = [int(P512 - a_hi), int(P512 - a_lo)]
        blk["keys_eval256"] = [int(P256 - a_hi), int(P256 - a_lo)]
        buckets[name] = blk
    # window-start token: seq position 256 — SAME token, both frames
    m512 = bucket_mass(probs512, torch.tensor([ws512]))
    m256 = bucket_mass(probs256, torch.tensor([ws256]))
    buckets["window_start"] = ratio_block(m256, m512)
    buckets["window_start"].update(
        ages=[WS_AGE, WS_AGE], keys_eval512=[ws512], keys_eval256=[ws256],
        seq_position=WS_SEQ_POS,
        wpe_eval512=T_TOTAL - EVAL256, wpe_eval256=0,
        note="same underlying token (seq position 256): wpe row 256 in the "
             "512-frame vs wpe row 0 in the re-indexed 256-frame (F2 cell)")
    # native-frame reference mass at ITS window start (seq position 0) —
    # a DIFFERENT token; reported for the secondary only, no ratio
    nat_ws_mass = bucket_mass(probs_nat, torch.tensor([0]))
    # eval-512-only far half (ages 256..511 = seq 0..255): mass the softmax
    # renormalization must re-home when the window is truncated
    far512 = bucket_mass(probs512, torch.arange(0, ws512))
    far_total = float(far512.reshape(B, -1).sum(1).mean())
    far_per_head = far512.mean(0)                        # (L, H)
    uniform_ratio = float(np.mean(1.0 / np.maximum(1.0 - far_per_head, 1e-6)))
    log(f"far-half mass (eval-512, ages 256-511, all 16 l*h): {far_total:.3f} "
        f"of 16 -> pure-renorm uniform survivor ratio would be "
        f"~{uniform_ratio:.3f}")

    for name in DECISION_SET + ["ages1_3", "ages6_17", "ages21_255"]:
        bkt = buckets[name]
        log(f"bucket {name:13s} mass 512 {bkt['mass_eval512']:.6f} -> "
            f"256 {bkt['mass_eval256']:.6f} | ratio {bkt['ratio']:.3f} "
            f"CI [{bkt['ratio_ci'][0]:.3f},{bkt['ratio_ci'][1]:.3f}]")

    # age-aligned total-mass curves (for the plot/record): ages 1..255
    cur512 = probs512.sum(1).sum(1).numpy()              # (B, P512) by key pos
    cur256 = probs256.sum(1).sum(1).numpy()              # (B, P256)
    age_mass512 = cur512[:, ::-1]                        # (B, 511) age-ascending
    age_mass256 = cur256[:, ::-1]                        # (B, 255)
    ratio_curve = (age_mass256.mean(0) / np.maximum(age_mass512[:, :P256].mean(0),
                                                    1e-30))
    per_seq_ratio_curve = age_mass256 / np.maximum(age_mass512[:, :P256], 1e-30)
    rq_lo = np.percentile(per_seq_ratio_curve, 25, axis=0)
    rq_hi = np.percentile(per_seq_ratio_curve, 75, axis=0)

    # ---- K-drop vs V-zero probes at age 4 / age 5 / window-start
    probes = {}
    frame_tbl = [
        ("eval512", ctx512, tgt, final_logits, CHUNK_512, 0,
         {"age4": P512 - 4, "age5": P512 - 5, "window_start": ws512}),
        ("eval256", ctx256, tgt, clean256, CHUNK_256, 0,
         {"age4": P256 - 4, "age5": P256 - 5, "window_start": ws256}),
        ("native256", ctx_nat, tgt_nat, clean_nat, CHUNK_256, 0,
         {"age4": P256 - 4, "age5": P256 - 5, "window_start": 0}),
    ]
    for fname, fctx, ftgt, fclean, fch, foff, pmap in frame_tbl:
        probes[fname] = {}
        for pname, key_pos in pmap.items():
            pr = probe_dce(net, fctx, ftgt, fclean, key_pos, fch, foff)
            blk = {k: agg_dce(v) for k, v in pr.items()}
            blk["k_minus_v"] = agg_dce(pr["kdrop"] - pr["vzero"])
            blk["key_pos"] = int(key_pos)
            probes[fname][pname] = blk
            log(f"probe {fname:9s} {pname:13s} V-zero {blk['vzero']['mean']:+.3f} "
                f"CI [{blk['vzero']['ci'][0]:+.3f},{blk['vzero']['ci'][1]:+.3f}] | "
                f"K-drop {blk['kdrop']['mean']:+.3f} CI "
                f"[{blk['kdrop']['ci'][0]:+.3f},{blk['kdrop']['ci'][1]:+.3f}] | "
                f"K-V {blk['k_minus_v']['mean']:+.3f}")

    # G6: probe V-zero must equal the full-sweep dCE at the same positions
    g6_dev = 0.0
    for fname, d, Pw in [("eval512", d512, P512), ("eval256", d256, P256),
                         ("native256", dnat, P256)]:
        for pname in ("age4", "age5", "window_start"):
            key_pos = probes[fname][pname]["key_pos"]
            sweep_col = d["dce_age"][:, Pw - key_pos - 1]  # age Pw-key_pos
            g6_dev = max(g6_dev, float(np.abs(
                np.asarray(probes[fname][pname]["vzero"]["per_seq"]) - sweep_col).max()))
    gates["G6_probe_vs_sweep"] = dict(max_dce_dev=g6_dev, ok=bool(g6_dev < 1e-5))
    log(f"G6 probe V-zero vs full-sweep dCE: max dev {g6_dev:.2e} "
        f"-> {'PASS' if g6_dev < 1e-5 else 'FAIL'}")

    # ---- REGISTERED decision (frozen numbers)
    prim = {k: buckets[k]["ratio"] for k in DECISION_SET}
    fires_wpe = prim["window_start"] >= R_WS_BAR and prim["age4"] <= R_AGE4_BAR
    fires_instr = all(INSTR_LO <= v <= INSTR_HI for v in prim.values())
    fires_redist = all(v > REDIST_BAR for v in prim.values())
    clauses = {
        "window_start_ge_2.0": bool(prim["window_start"] >= R_WS_BAR),
        "age4_le_0.75": bool(prim["age4"] <= R_AGE4_BAR),
        "all_in_[0.9,1.1]": bool(fires_instr),
        "all_gt_1.1": bool(fires_redist),
    }
    r13 = buckets["ages1_3"]["ratio"]
    c1_extra = dict(ages1_3_ratio=r13,
                    ages1_3_within_pm20pct=bool(0.8 <= r13 <= 1.2))
    if fires_wpe and not fires_instr and not fires_redist:
        verdict = ("H-wpe-domain (positional reshape): window-start mass ratio "
                   f">= 2.0 ({prim['window_start']:.2f}) AND age-4 ratio <= 0.75 "
                   f"({prim['age4']:.2f}) — frozen rule fires")
    elif fires_instr:
        verdict = ("H-instrument (tail): all registered ratios in [0.9,1.1] — "
                   "the dCE reshape is a statistic, not an attention change")
    elif fires_redist:
        verdict = ("H-redistribute (contributory): all registered ratios "
                   "uniformly > 1.1 — proportional survivor re-scaling")
    else:
        verdict = ("MIXED — the pattern straddles the frozen branches: " +
                   "; ".join(f"{k}={v:.2f}" for k, v in prim.items()))
    log(f"REGISTERED DECISION: {verdict}")
    log(f"  C1 extra sign checks: ages1-3 ratio {r13:.3f} (within +-20%: "
        f"{c1_extra['ages1_3_within_pm20pct']}); window-start K-V "
        f"{probes['eval256']['window_start']['k_minus_v']['mean']:+.3f} nats")

    # ---- SECONDARY verdict: native positions-0-255 slice
    nat_ws = float(dnat["mean_age_curve"][WS_AGE - 1])
    a_nat = dnat["a_star"]
    nat_dead = nat_ws < THRESH
    nat_in_win = a_nat is not None and 4 <= a_nat <= 8
    if nat_in_win and nat_dead:
        nat_verdict = (f"a*(native-256 slice) = {a_nat} (~6, in [4,8]) with a "
                       f"DEAD window start ({nat_ws:+.3f} nats) — the D1 effect "
                       "is specific to truncating a LONG window (code band / "
                       "composition), not window length per se")
    else:
        nat_verdict = (f"a*(native-256 slice) = {a_nat}, window-start "
                       f"V-zero dCE {nat_ws:+.3f} nats "
                       f"({'dead' if nat_dead else 'LIVE'}, threshold "
                       f"{THRESH}) — reported as-is")
    log(f"SECONDARY (native slice): {nat_verdict}")

    # ---------------------------------------------------------------- metrics
    def sweep_summary(d, ce, role, wpe):
        return dict(
            role=role, wpe=wpe, a_star=d["a_star"],
            a_star_naive=d["a_star_naive"], a_star_robust=d["a_star_robust"],
            a_star_ci=d["a_star_ci"], per_seq_a_star=d["per_seq_a_star"],
            live_frac=d["live_frac"],
            ages1_5=[float(a) for a in d["mean_age_curve"][:5]],
            ages6_17=[float(a) for a in d["mean_age_curve"][5:17]],
            window_start_dce=float(d["mean_age_curve"][WS_AGE - 1]),
            clean_ce_per_seq=ce.tolist(), clean_ce_mean=float(np.mean(ce)),
        )

    metrics = dict(
        experiment="e070_attention_mass",
        purpose="T039-amendment INTERPRETER-PASS registered discriminator: "
                "final-query attention-received mass per layer x head under "
                "eval-256 (re-indexed) vs eval-512 (native) on the SAME "
                "sequences + K-drop vs V-zero at age4/age5/window-start. "
                "FROZEN: window-start ratio >= 2.0 AND age-4 ratio <= 0.75 "
                "=> H-wpe-domain; all of {ages1-20, age4, age5, window-start} "
                "in [0.9,1.1] => H-instrument; all > 1.1 => H-redistribute; "
                "else MIXED. Secondary: a* on the native positions-0-255 "
                "slice (in-distribution at generation).",
        started=started, wall_s=elapsed(), threads=THREADS, cpu_only=True,
        net=dict(ckpt=str(CKPT), arch=dict(n_layer=4, n_head=4, n_embd=128,
                                           block_size=T_TOTAL, vocab=65),
                 params=n_params, steps_in_ckpt=steps_in_ckpt,
                 val_ce=val_ce, val_ce_e053c=E053C_VAL_CE),
        seeds=dict(corpus=1337, prompts=SEED_PROMPT, sampling=SEED_SAMPLE,
                   bootstrap=0),
        protocol=dict(B=B, prompt_tokens=PROMPT_TOK, t_total=T_TOTAL, temp=TEMP,
                      topk=TOPK, fixed_anchor=True, threshold=THRESH,
                      a_star_window=A_STAR_K, boot_n=BOOT_N,
                      chunks=dict(w512=CHUNK_512, w256=CHUNK_256),
                      eval_window="e069 verbatim: same net (e053c_ctx512.pt), "
                                  "same seeds 202/7, same B=4 sequences, same "
                                  "generation/sweep/onset machinery; eval-256 = "
                                  "last-256-positions re-indexed (wpe 0..254)",
                      mass_aggregation="per layer x head softmax row of the "
                                       "final query; bucket mass summed over "
                                       "the key set; DECISION ratios use the "
                                       "total summed over all 16 layer x head, "
                                       "mean over B=4 sequences, ratio of means",
                      lesion_instrument="all-layer whole-position K-drop / "
                                        "V-zero (e069 manual-forward masks)"),
        gates=gates,
        frames=dict(
            eval512=dict(ctx="idx[:, :-1] (seq positions 0..510)",
                         wpe="native 0..510", Pw=P512),
            eval256=dict(ctx="idx[:, 256:511] (seq positions 256..510)",
                         wpe="re-indexed 0..254", Pw=P256),
            native256=dict(ctx="idx[:, :255] (seq positions 0..254)",
                           wpe="native 0..254 (in-distribution at generation)",
                           Pw=P256, target="idx[:, 255]"),
        ),
        sweeps=dict(
            eval512=sweep_summary(d512, ce512, "e069 sweep A replica (baseline)",
                                  "native 0..510"),
            eval256=sweep_summary(d256, ce256,
                                  "e069 sweep B replica (the D1 surprise frame)",
                                  "re-indexed 0..254"),
            native256_slice=sweep_summary(
                dnat, ce_nat, "SECONDARY: native positions-0-255 slice",
                "native 0..254"),
        ),
        attention_mass=dict(
            far_half_eval512=dict(
                ages=[256, 511], total_mass_all16lh=far_total,
                per_head_mean=far_per_head.tolist(),
                pure_renorm_uniform_survivor_ratio=uniform_ratio,
                note="eval-512-only bucket (ages 256..511 = seq 0..255): the "
                     "mass softmax truncation removes; if H-redistribute's "
                     "plain form held, every survivor ratio would equal "
                     "~1/(1-far_mass) per head"),
            buckets=buckets,
            ratio_curve=dict(
                ages=list(range(1, P256 + 1)),
                ratio_of_means=ratio_curve.tolist(),
                per_seq_quartiles=[rq_lo.tolist(), rq_hi.tolist()],
                note="total mass (all 16 layer x head) per age, eval-256 / "
                     "eval-512, age-aligned (same underlying tokens)"),
            native_window_start_mass_all16lh=float(nat_ws_mass.reshape(B, -1).sum(1).mean()),
        ),
        lesion_probes=dict(
            eval512=probes["eval512"], eval256=probes["eval256"],
            native256=probes["native256"],
            reading="window-start K-drop >> V-zero => anchor/sink mass (the "
                    "key is needed for routing); ~equal => value read "
                    "(scratch §4 C1 sign check; all-layer whole-position "
                    "lesions)"),
        registered_decision=dict(
            frozen_rules=dict(
                h_wpe_domain=f"window-start ratio >= {R_WS_BAR} AND age-4 "
                             f"ratio <= {R_AGE4_BAR}",
                h_instrument=f"all of {DECISION_SET} in [{INSTR_LO},{INSTR_HI}]",
                h_redistribute=f"all of {DECISION_SET} > {REDIST_BAR}"),
            ratio_set=DECISION_SET,
            ratios=prim,
            ratio_cis={k: buckets[k]["ratio_ci"] for k in DECISION_SET},
            clauses_fired=clauses,
            c1_extra_sign_checks=c1_extra,
            pure_renorm_prediction=uniform_ratio,
            verdict=verdict),
        secondary_native_slice=dict(
            registered="a*(native-256 slice) ~ 6 with a dead window start => "
                       "the D1 effect is specific to truncating a LONG window "
                       "(code band / composition), not window length per se "
                       "[scratch §4 SECONDARY]",
            a_star=a_nat, a_star_ci=dnat["a_star_ci"],
            per_seq_a_star=dnat["per_seq_a_star"],
            window_start_vzero_dce=nat_ws,
            window_start_kdrop_dce=probes["native256"]["window_start"]["kdrop"]["mean"],
            window_start_dead=bool(nat_dead),
            a_star_in_48=bool(nat_in_win),
            ages1_5=[float(a) for a in dnat["mean_age_curve"][:5]],
            clean_ce_mean=float(np.mean(ce_nat)),
            vs_eval256=dict(a_star=d256["a_star"],
                            window_start_vzero_dce=float(
                                d256["mean_age_curve"][WS_AGE - 1])),
            verdict=nat_verdict),
    )
    save_json(out_dir / "metrics.json", metrics)
    log("metrics.json written")
    plot(out_dir / "attention_mass.png", metrics, buckets, probes,
         ratio_curve, rq_lo, rq_hi, far_total, uniform_ratio)
    log(f"plot written; total wall {elapsed():.0f}s")


# ---------------------------------------------------------------------- plot


def plot(path: Path, M: dict, buckets: dict, probes: dict, ratio_curve,
         rq_lo, rq_hi, far_total: float, uniform_ratio: float):
    L = H = 4
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]
    dec = M["registered_decision"]

    # ---- panel 1: ratio vs age (the registered readout)
    ages = np.arange(1, P256 + 1)
    ax1.fill_between(ages, rq_lo, rq_hi, alpha=0.25, color="tab:blue",
                     label="per-seq quartiles (B=4)")
    ax1.plot(ages, ratio_curve, lw=1.3, color="tab:blue",
             label="ratio of means (all 16 layer x head)")
    for y, c, lbl in [(0.75, "tab:purple", "0.75 (age-4 bar)"),
                      (1.0, "k", "1.0"), (1.1, "tab:orange", "1.1"),
                      (2.0, "tab:red", "2.0 (window-start bar)")]:
        ax1.axhline(y, color=c, ls="--", lw=0.9)
        ax1.text(P256 * 0.99, y, f" {lbl}", fontsize=7, va="bottom", ha="right",
                 color=c)
    ax1.axhline(uniform_ratio, color="tab:green", ls=":", lw=1.2)
    ax1.text(2, uniform_ratio, f" pure-renorm {uniform_ratio:.2f}",
             fontsize=7, color="tab:green", va="bottom")
    for a, nm in [(4, "age 4"), (5, "age 5"), (WS_AGE, "window-start\n(age 255)")]:
        r = buckets["age4" if a == 4 else "age5" if a == 5 else "window_start"]["ratio"]
        ax1.plot([a], [r], "o", ms=7, color="tab:red", zorder=5)
        ax1.annotate(f"{nm}\nr={r:.2f}", (a, r), textcoords="offset points",
                     xytext=(6, 8), fontsize=7.5)
    ax1.set_xscale("log")
    ax1.set_xlim(1, P256)
    ax1.set_xlabel("cache age (tokens, log scale)")
    ax1.set_ylabel("attention mass ratio (eval-256 / eval-512)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_title("E070 — final-query attention mass, eval-256 vs eval-512 "
                  "(same net, same seqs)\n" + dec["verdict"], fontsize=9.5)

    # ---- panel 2: bucket masses + ratios (log scale)
    order = ["ages1_3", "age4", "age5", "ages6_17", "ages21_255", "window_start"]
    labels = ["ages 1-3", "age 4", "age 5", "ages 6-17", "ages 21-255",
              "window-start\n(age 255)"]
    m512 = [max(buckets[k]["mass_eval512"], 1e-9) for k in order]
    m256 = [max(buckets[k]["mass_eval256"], 1e-9) for k in order]
    x = np.arange(len(order))
    ax2.bar(x - 0.19, m512, 0.38, color="tab:red", alpha=0.85,
            label="eval-512 (native)")
    ax2.bar(x + 0.19, m256, 0.38, color="tab:blue", alpha=0.85,
            label="eval-256 (re-indexed)")
    for i, k in enumerate(order):
        r = buckets[k]["ratio"]
        ax2.text(x[i], max(m512[i], m256[i]) * 1.35, f"r={r:.2f}",
                 ha="center", fontsize=8, color="dimgray")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-6, 20)
    ax2.set_xticks(x, labels, fontsize=8)
    ax2.set_ylabel("attention-received mass (summed, all 16 layer x head)")
    ax2.legend(fontsize=8)
    ax2.set_title("bucket masses of the final query (sum over 16 layer x head, "
                  f"mean over B=4)\neval-512 far-half mass (ages 256-511) = "
                  f"{far_total:.3f} of 16", fontsize=9.5)

    # ---- panels 3/4: per-layer x head ratio heatmaps
    for ax, key, ttl in [(ax3, "window_start", "window-start mass ratio "
                                              "(seq pos 256: wpe 256 -> row 0)"),
                         (ax4, "age4", "age-4 mass ratio (the collapsed "
                                       "shoulder)")]:
        R = np.asarray(buckets[key]["per_layerhead_ratio"])
        im = ax.imshow(R, cmap="RdBu_r", vmin=0.0, vmax=3.0, aspect="auto")
        for li in range(L):
            for hi in range(H):
                v = R[li, hi]
                ax.text(hi, li, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(v - 1.5) > 0.9 else "black",
                        fontweight="bold" if (v >= 2.0 or v <= 0.75) else "normal")
        ax.set_xticks(range(H), [f"h{h}" for h in range(H)])
        ax.set_yticks(range(L), [f"layer {l}" for l in range(L)])
        ax.set_title(f"{ttl}\ntotal ratio {buckets[key]['ratio']:.2f} "
                     f"CI [{buckets[key]['ratio_ci'][0]:.2f},"
                     f"{buckets[key]['ratio_ci'][1]:.2f}]", fontsize=9.5)
        fig.colorbar(im, ax=ax, fraction=0.046)

    # ---- panel 5: K-drop vs V-zero probes
    groups = [("age 4", "age4"), ("age 5", "age5"),
              ("window-start", "window_start")]
    w = 0.2
    xs = np.arange(len(groups))
    for i, (frame, col) in enumerate([("eval512", "tab:red"),
                                      ("eval256", "tab:blue")]):
        for j, kind in enumerate(["vzero", "kdrop"]):
            means, lo_e, hi_e = [], [], []
            for _, key in groups:
                a = probes[frame][key][kind]
                means.append(a["mean"])
                lo_e.append(a["mean"] - a["ci"][0])
                hi_e.append(a["ci"][1] - a["mean"])
            ax5.bar(xs + (i - 0.5) * 0.4 + (j - 0.5) * 0.19, means, w,
                    color=col, alpha=0.55 + 0.35 * j,
                    label=f"{frame} {'V-zero' if kind == 'vzero' else 'K-drop'}")
            ax5.errorbar(xs + (i - 0.5) * 0.4 + (j - 0.5) * 0.19, means,
                         yerr=[lo_e, hi_e], fmt="none", ecolor="k", lw=0.8,
                         capsize=2)
    ax5.axhline(0, color="k", lw=0.6)
    ax5.set_xticks(xs, [g[0] for g in groups])
    ax5.set_ylabel("lesion dCE (nats)")
    ax5.legend(fontsize=7.5, ncol=2)
    ax5.set_title("K-drop vs V-zero at the probe positions (all-layer "
                  "whole-position lesions, e069 instrument)\nwindow-start "
                  f"(eval-256): K-V = "
                  f"{probes['eval256']['window_start']['k_minus_v']['mean']:+.3f} "
                  "nats", fontsize=9.5)

    # ---- panel 6: secondary — native slice vs re-indexed eval-256
    for key, col, lbl in [
            ("eval256", "tab:blue", f"eval-256 re-indexed (a*={M['sweeps']['eval256']['a_star']})"),
            ("native256_slice", "tab:green",
             f"native 0-255 slice (a*={M['sweeps']['native256_slice']['a_star']})")]:
        s = M["sweeps"][key]
        ag = np.arange(1, 18)
        curve = np.asarray(s["ages1_5"] + s["ages6_17"])   # ages 1..17
        ax6.plot(ag, curve, "o-", lw=1.3, color=col, label=lbl)
        if s["a_star"]:
            ax6.axvline(s["a_star"], color=col, ls="--", lw=1.1)
    sec = M["secondary_native_slice"]
    ax6.axhline(THRESH, color="gray", ls=":", lw=1.0)
    ax6.axhline(0, color="k", lw=0.6)
    ax6.set_xlabel("cache age (tokens)")
    ax6.set_ylabel("V-zero dCE (nats)")
    ax6.legend(fontsize=8)
    ws_r = M["sweeps"]["eval256"]["window_start_dce"]
    ax6.text(0.03, 0.05,
             f"window-start V-zero dCE: native "
             f"{sec['window_start_vzero_dce']:+.3f} "
             f"({'dead' if sec['window_start_dead'] else 'LIVE'}) vs "
             f"re-indexed {ws_r:+.3f} (LIVE)\n" + sec["verdict"],
             transform=ax6.transAxes, fontsize=7.5, va="bottom", color="dimgray")
    ax6.set_title("SECONDARY — a* on the native positions-0-255 slice "
                  "(in-distribution at generation)", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
