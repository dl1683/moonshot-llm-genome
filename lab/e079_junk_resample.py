"""E079 — T047 claim-C REGISTERED fix: B=16 resample of the junk source-split.

Trigger (THINKING.md T047 claim C, 2026-09-26 ~12:05Z): e073's source split
(generated-old junk 0.085-0.372 across the e053 family vs prompt junk
0.000-0.063; e074's shuffled-prompt control then fired H-SOURCE 4/4) is
SOLID-WITH-FLAGS: (i) the -0.01-nat rule carves a tail off a broad haze
(gen-band mean dCE -0.0013; 39-48% of entries negative — threshold-artifact
risk), (ii) per-seq concentration heavy (one sequence carried 55-64% of
late-band junk at B=4). Registered fix: B=16 resample of the split on >=2
nets — kills the threshold-artifact and per-seq flags together if the
fractions hold.

Part 1 (measured, e053c ctx-512 net, CPU): regenerate B=16 sequences =
e069's exact battery A (seeds 202/7, B=4) + 12 fresh from the SAME family
extended deterministically exactly as e072 documented it (prompt seed
202->302, sampling seed 7->17, identical draw rules). Full 511-position
final-step V-zero sweep (e069/e072/e074 machinery verbatim), recompute the
source split (junk = V-zero dCE <= -0.01 nats; PROMPT band positions 1..63
/ ages 448..510, n=63/seq, position 0 excluded as in e073/e074; GENERATED
band positions 64..510 restricted to ages >= the sweep's own a*), headline
= mean over sequences of the per-seq junk fraction with bootstrap CI over
the sequence axis.

Part 2 (propagated, no rerun): pull the stored final_profiles of the e053
family cells (runs/e053/metrics.json: small_0.84M n_seq=8, mid_2.7M n_seq=2,
large_10M n_seq=2 — ctx-256 nets, prompt = oldest 64 -> ages 192..254,
generated -> ages a*..191, e073's age-level convention) and compute their
B=16-equivalent junk-fraction uncertainty by propagating the stored
per-age dCE bootstrap CIs: per age, sd = (hi-lo)/(2*1.96) scaled by
sqrt(n_seq/16) (sd of an age-mean shrinks as 1/sqrt(B)); Monte Carlo the
threshold count. Correlation-free worst case also reported (box bounds:
fraction of ages whose CI_hi <= -0.01 vs CI_lo <= -0.01). FLAGGED honestly:
normal approximation; cross-age independence within a sequence is not
recoverable from stored profiles; the age-level fraction is a different
statistic from part 1's per-seq-then-mean fraction (cross-net comparison is
contrast-level, not statistic-identical).

REGISTERED BARS (frozen, T047/task spec, before any measurement):
  - BAR 1: on e053c at B=16, generated-old junk >= 2x prompt-band junk
    (point ratio; prompt junk 0 counts as satisfied when gen junk > 0)
    AND the paired-bootstrap CI of (gen - prompt) excludes 0.
  - BAR 2: the same contrast holds in the propagated B=16-equivalent CIs
    of >= 1 of the mid_2.7M / large_10M cells.
  - CONCENTRATION FLAG: if one sequence still carries > 40% of late-band
    junk at B=16, report it — that is the flag that survives.

Run:     python lab/e079_junk_resample.py
Outputs: runs/e079/metrics.json + runs/e079/junk_resample.png
Envelope: NO training, NO new automations; CPU-only, single step, minutes.
No NOTES/THINKING/QUEUE/STATE edits; no commit (the dispatcher owns those).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # task spec: CPU-only (pre-torch)
import torch

# this torch build reports is_available()==True even with an empty
# CUDA_VISIBLE_DEVICES; force CPU so common.DEVICE=="cpu" (e053b..e074)
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

import json  # noqa: E402
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
SEED_PROMPT, SEED_SAMPLE = 202, 7         # e053b/e053c/e069/e070/e072/e074 seeds
N_PROMPTS = 8
B = 4                                     # battery A uniform B
BOOT_N = 1000
THREADS = 12
torch.set_num_threads(THREADS)

# battery B (B=16): e069's exact 4 + 12 fresh from the extended family
# (e072's documented convention, reused verbatim)
B_X = 12
SEED_PROMPT_X = 302                       # 202-family + 100
SEED_SAMPLE_X = 17                        # 7-family + 10
B16 = B + B_X                             # 16

P512 = T_TOTAL - 1                        # 511 sweep positions (native frame)
THRESH = 0.01                             # dead-weight / junk threshold (nats)
A_STAR_K = 5                              # sustained window (e053b verbatim)
CHUNK_512 = 64                            # e069/e070/e072/e074 measured chunk

# ---- bands (e073/e074 instrument) --------------------------------------------
PROMPT_POS = (1, 63)                      # positions 1..63 (pos 0 = seq start)
GEN_POS = (64, 510)                       # generated context entries
BASE_A_STAR = 6                           # e053c/e069/e072 eval-512 onset

# ---- REGISTERED bars (FROZEN, T047 / task spec) ------------------------------
RATIO_BAR = 2.0                           # gen-old junk >= 2x prompt junk
CONC_BAR = 0.40                           # one-seq share of late-band junk

# ---- multi-net propagation constants -----------------------------------------
MC_N = 20000                              # Monte Carlo draws per band
SEED_MC = 79                              # e079's own: propagation rng
B_TARGET = 16                             # the B=16-equivalent scale
Z95 = 1.959963985                        # percentile-CI -> sd factor
E053_METRICS = REPO / "runs" / "e053" / "metrics.json"
MULTINET_CELLS = ["small_0.84M", "mid_2.7M", "large_10M"]
MULTINET_BAR_CELLS = ["mid_2.7M", "large_10M"]   # BAR 2's registered pool

# ---- e069/e073/e072 stored readouts (protocol-identity gates) ----------------
E069_REF = {"eval512": {"a_star": 6, "ci": [4.0, 8.0], "clean_ce": 0.4528,
                        "ages1_5": [1.742, 4.467, 2.319, 0.914, 0.321],
                        "per_seq": [8, 4, 6, 3], "winstart_dce": 0.0069}}
# e073's stored reanalysis of the e053c cell (runs/e073_junk_split_reanalysis.json)
E073_REF = {"gen_old_junk": 0.09954751131221719,
            "gen_old_n": 442,
            "prompt_junk": 0.06349206349206349,
            "prompt_n": 63,
            "mean_dce_gen_old": -0.001330780132454546,
            "mean_dce_prompt": 0.00648569551459144}
# e072's stored B=16 eval-512 secondary (same battery, same sweep; identity gate)
E072_B16_REF = {"a_star": 6, "ci": [4.0, 14.0],
                "per_seq": [8, 4, 6, 3, 11, 14, 3, 31, 18, 1, 1, 1, 7, 6, 2, 3],
                "clean_ce": 1.2227551937}
# e073's stored family numbers (NOTES.md E073: 0.101/0.085/0.367 + prompt
# 0.000/0.032/0.000) — the multi-net reconstruction gate
E073_FAMILY_REF = {"small_0.84M": {"gen": 0.101, "prompt": 0.000},
                   "mid_2.7M": {"gen": 0.085, "prompt": 0.032},
                   "large_10M": {"gen": 0.367, "prompt": 0.000}}
CKPT = REPO / "runs" / "checkpoints" / "e053c_ctx512.pt"
E053C_VAL_CE = 1.5226792494455974         # the net being interrogated
T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# ------------------------------------------------- machinery (VERBATIM e069..e074)

@torch.no_grad()
def _manual_chunk(net: TinyGPT, idxs, vzero, kdrop, layers, pos_offset=0):
    """Exact manual forward, returns LAST-position logits (N, vocab)."""
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
    (position-T_TOTAL-1) logits (B, V). [CPU-only, e069..e074 verbatim]"""
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
    """Final-step fine V-zero sweep for all B sequences at once. Returns dce
    (B, Pw) indexed by POSITION; age of pos p = Pw - p, so the age-ascending
    view is dce[:, ::-1], plus the per-sequence clean CE. [e053c math]"""
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


def bootstrap_stat(X_by_seq: np.ndarray, stat, n: int = BOOT_N, seed: int = 0):
    """Bootstrap any per-sequence stat. X_by_seq: (S, ...) -> (lo, hi)."""
    S = X_by_seq.shape[0]
    if S < 2:
        v = stat(X_by_seq)
        return (float(v), float(v))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sel = rng.integers(0, S, S)
        vals.append(stat(X_by_seq[sel]))
    vals = np.asarray(vals, float)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def analyze_window(sweep_pos: np.ndarray):
    """sweep_pos: (B, Pw) V-zero dCE by position. Returns the derived block."""
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
    """Full V-zero sweep -> analysis, plus bookkeeping."""
    dce, ce_clean = fine_sweep(net, ctx, tgt, clean_logits, ch, pos_offset)
    nan_n = int(np.isnan(dce).sum())
    if nan_n:
        dce = np.nan_to_num(dce, nan=0.0)
    d = analyze_window(dce)
    log(f"  a*={d['a_star']} CI [{d['a_star_ci'][0]:.0f},{d['a_star_ci'][1]:.0f}] "
        f"naive={d['a_star_naive']} robust={d['a_star_robust']} "
        f"live_frac={d['live_frac']:.3f} | clean CE {ce_clean.mean():.4f} "
        f"(nan->0: {nan_n})")
    return d, ce_clean


# ------------------------------------- the junk census (e073/e074 instrument)

def junk_band(dce_age: np.ndarray, a_lo: int, a_hi: int):
    """Junk census of the age band [a_lo, a_hi] (inclusive; ages ascending).
    Junk = dCE <= -THRESH (lesion-helpful = negative utility)."""
    band = dce_age[:, a_lo - 1:a_hi]
    n = band.shape[1]
    per_seq_frac = (band <= -THRESH).mean(1)
    per_seq_count = (band <= -THRESH).sum(1)
    lo, hi = bootstrap_stat(per_seq_frac[:, None], lambda m: float(m.mean()))
    return dict(
        ages=[a_lo, a_hi], n_per_seq=int(n),
        junk_frac_mean=float(per_seq_frac.mean()),
        junk_frac_mean_ci=[float(lo), float(hi)],
        junk_frac_max=float(per_seq_frac.max()),
        junk_frac_pooled=float((band <= -THRESH).mean()),
        junk_counts_per_seq=[int(c) for c in per_seq_count],
        mean_dce=float(band.mean()),
        frac_dce_negative=float((band < 0).mean()),
    )


def gen_position_range(a_star: int):
    """Generated band position range under the beyond-onset restriction.
    Context positions 64..510 are generated; age of position p = 511 - p;
    ages >= a_star -> positions 64..(511 - a_star)."""
    p_hi = P512 - int(a_star)
    return GEN_POS[0], p_hi


def gen_order_halves(dce_age: np.ndarray, a_star: int):
    """Early vs late generated halves at the midpoint of the restricted
    position range (generation order == position order; early = older)."""
    p_lo, p_hi = gen_position_range(a_star)
    mid = (p_lo + p_hi) // 2
    out = {}
    for name, (lo, hi) in [("early_generated", (p_lo, mid)),
                           ("late_generated", (mid + 1, p_hi))]:
        a_band = (P512 - hi, P512 - lo)
        out[name] = junk_band(dce_age, a_band[0], a_band[1])
        out[name]["positions"] = [lo, hi]
    return out


def concentration(band: dict, rows: list | None = None, tag: str = ""):
    """Per-seq concentration of junk in a band census: max single-sequence
    share of the pooled junk count (over the given rows; default all).
    [T047's flag: one sequence > 40% of late-band junk]"""
    counts = (band["junk_counts_per_seq"] if rows is None
              else [band["junk_counts_per_seq"][i] for i in rows])
    tot = int(sum(counts))
    share = (max(counts) / tot) if tot > 0 else float("nan")
    return dict(tag=tag, counts=counts, pooled_total=tot,
                max_share=float(share),
                max_seq=int(np.argmax(counts)),
                flag_heavy=bool(share > CONC_BAR))


# ------------------------------------------------------ PART 2 (propagation)

def propagate_cell(cell: dict, rng: np.random.Generator):
    """Stored e053 final_profile -> age-level junk split + B=16-equivalent
    propagated CIs. Convention (e073 reanalysis, reproduced + gated):
    ctx-256 window, Pw=255, age of position p = 255 - p; prompt = oldest 64
    -> positions 1..63 = ages 192..254 (position 0 = sequence-start anchor,
    age 255, excluded); generated = positions 64..254 = ages 1..191,
    restricted to ages >= the cell's own a*. Junk (age level) = mean dCE
    (over the cell's sequences) <= -0.01.
    Propagation: per age, sd = (ci_hi - ci_lo)/(2*1.96) from the stored
    per-age bootstrap CI; B=16-equivalent sd scales it by sqrt(n_seq/16);
    MC the threshold count per band. Box bounds = correlation-free worst
    case from the stored CIs directly."""
    fp = cell["final_profile"]
    age = np.asarray(fp["age"], float)
    mu = np.asarray(fp["vzero_dce_mean"], float)
    ci = np.asarray(fp["vzero_ci"], float)
    n_seq = int(cell["protocol"]["n_seq"])
    a_star = int(cell["derived"]["a_star"])
    gen = (age >= a_star) & (age <= 191)
    pr = (age >= 192) & (age <= 254)
    sd_own = np.maximum((ci[1] - ci[0]) / (2.0 * Z95), 1e-6)
    sd_16 = sd_own * math.sqrt(n_seq / B_TARGET)

    def mc_junk(mask, sd):
        draws = rng.normal(mu[mask][None, :], sd[mask][None, :],
                           size=(MC_N, int(mask.sum())))
        return (draws <= -THRESH).mean(1)

    def box(mask):
        lo = float((ci[1][mask] <= -THRESH).mean())   # junk in EVERY draw
        hi = float((ci[0][mask] <= -THRESH).mean())   # junk in SOME draw
        return lo, hi

    jg16, jp16 = mc_junk(gen, sd_16), mc_junk(pr, sd_16)
    jg_own, jp_own = mc_junk(gen, sd_own), mc_junk(pr, sd_own)
    diff16 = jg16 - jp16
    g_lo, g_hi = box(gen)
    p_lo, p_hi = box(pr)
    gen_pt, pr_pt = float((mu[gen] <= -THRESH).mean()), float((mu[pr] <= -THRESH).mean())
    ratio = (gen_pt / pr_pt) if pr_pt > 0 else float("inf")
    # registered contrast reading: point ratio >= 2 (prompt 0 + gen > 0
    # passes) AND the CI of the difference excludes equality (0)
    ratio_ok = bool(ratio >= RATIO_BAR) and gen_pt > 0
    ratio_json = round(ratio, 2) if math.isfinite(ratio) else "inf"
    diff_ci = [float(np.percentile(diff16, 2.5)), float(np.percentile(diff16, 97.5))]
    diff_excl0 = bool(diff_ci[0] > 0)
    box_diff = [g_lo - p_hi, g_hi - p_lo]
    return dict(
        n_seq=n_seq, a_star=a_star, arch=cell["arch"], ckpt=cell["ckpt"],
        bands=dict(generated=dict(ages=[a_star, 191], n=int(gen.sum())),
                   prompt=dict(ages=[192, 254], n=int(pr.sum())),
                   excluded="age 255 = position 0 = sequence-start anchor "
                            "(e073 convention)"),
        point=dict(gen_junk=gen_pt, prompt_junk=pr_pt, ratio=ratio_json,
                   mean_dce_gen=float(mu[gen].mean()),
                   mean_dce_prompt=float(mu[pr].mean())),
        propagated_b16=dict(
            method=f"per-age normal sd from stored percentile CI, scaled "
                   f"sqrt(n_seq/{B_TARGET}); {MC_N} MC draws; seed {SEED_MC}",
            gen_junk_ci=[float(np.percentile(jg16, 2.5)),
                         float(np.percentile(jg16, 97.5))],
            prompt_junk_ci=[float(np.percentile(jp16, 2.5)),
                            float(np.percentile(jp16, 97.5))],
            diff_ci=diff_ci, diff_excludes_0=diff_excl0,
            p_diff_gt0=float((diff16 > 0).mean()),
            p_contrast_holds=float((jg16 >= RATIO_BAR * jp16).mean()),
            note="p_contrast_holds counts prompt_junk=0 draws as held "
                 "whenever gen_junk>0 (2x0 = 0); assumption flags: per-age "
                 "normality; cross-age independence within a sequence is "
                 "not recoverable from stored profiles"),
        propagated_own_B=dict(
            gen_junk_ci=[float(np.percentile(jg_own, 2.5)),
                         float(np.percentile(jg_own, 97.5))],
            prompt_junk_ci=[float(np.percentile(jp_own, 2.5)),
                            float(np.percentile(jp_own, 97.5))],
            note="unscaled (the cell's own n_seq) — transparency only"),
        box_bounds=dict(
            gen_junk=[g_lo, g_hi], prompt_junk=[p_lo, p_hi],
            diff=[box_diff[0], box_diff[1]],
            note="correlation-free worst case from the stored CIs alone: lo "
                 "= fraction of ages junk under EVERY resample-consistent "
                 "mean (CI_hi <= -0.01), hi = under SOME (CI_lo <= -0.01)"),
        contrast=dict(ratio_bar=RATIO_BAR, ratio_ok=ratio_ok,
                      diff_ci_excludes_0=diff_excl0,
                      fires=bool(ratio_ok and diff_excl0),
                      box_diff_excludes_0=bool(box_diff[0] > 0)),
    )


# ---------------------------------------------------------------------- main


def main():
    assert torch.cuda.device_count() == 0, "CPU-only violated: CUDA visible"
    assert common.DEVICE == "cpu", f"common.DEVICE={common.DEVICE} != cpu"
    out_dir = run_dir("e079")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gates = dict(cpu_only=True, threads=THREADS, smoke=False)

    # ---- net: rebuild corpus + e053c EXACTLY as e069..e074 did
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

    # ---- battery A: e069/e072/e074 protocol verbatim (seeds 202/7, B=4)
    gen_p = torch.Generator().manual_seed(SEED_PROMPT)
    ix = torch.randint(len(corp.val) - PROMPT_TOK - 1, (N_PROMPTS,),
                       generator=gen_p)
    prompts = [corp.val[i:i + PROMPT_TOK] for i in ix]
    log(f"battery A: {N_PROMPTS} prompts (seed {SEED_PROMPT}); using first "
        f"B={B}; prompt0 prefix: {corp.decode(prompts[0])[:32]!r}")
    gen_a = torch.Generator().manual_seed(SEED_SAMPLE)   # seed-7 B=4 stream
    idx4, final_logits4 = generate_batch(net, prompts[:B], gen_a)
    log(f"battery A: generated {B} seqs 64->{T_TOTAL} (fixed anchor, CPU)")

    # G0b: incremental-KV final logits vs full recompute (prob dev)
    with torch.no_grad():
        full4 = manual_logits(net, idx4[:, :-1], None, None, None, chunk=B)
    dev_b = float((torch.softmax(final_logits4.float(), -1)
                   - torch.softmax(full4.float(), -1)).abs().max())
    gates["G0b_kvs_vs_full"] = dict(max_prob_dev=dev_b, ok=bool(dev_b < 1e-4))
    log(f"G0b incremental-KV vs full recompute: max prob dev {dev_b:.2e} "
        f"-> {'PASS' if dev_b < 1e-4 else 'FAIL'}")

    # ---- battery B: 12 fresh (seeds 302/17 — e072's documented extension)
    log(f"battery B: {B_X} fresh sequences (prompt seed {SEED_PROMPT_X}, "
        f"sampling seed {SEED_SAMPLE_X}; family = 202/7 +100/+10, e072 "
        f"convention verbatim)")
    gen_px = torch.Generator().manual_seed(SEED_PROMPT_X)
    ix12 = torch.randint(len(corp.val) - PROMPT_TOK - 1, (B_X,),
                         generator=gen_px)
    prompts12 = [corp.val[i:i + PROMPT_TOK] for i in ix12]
    log(f"  fresh prompt0 prefix: {corp.decode(prompts12[0])[:32]!r}")
    gen_x = torch.Generator().manual_seed(SEED_SAMPLE_X)
    idx12, _ = generate_batch(net, prompts12, gen_x)
    idx16 = torch.cat([idx4, idx12], 0)
    assert idx16.shape == (B16, T_TOTAL)
    log(f"battery assembled: {B16} sequences = battery A's {B} + {B_X} fresh")

    # ---- THE MEASUREMENT: B=16 full 511-position V-zero sweep
    log(f"PART 1: B={B16} eval-512 fine V-zero sweep (the registered resample)")
    tgt16 = idx16[:, -1]
    ctx512_16 = idx16[:, :-1]
    clean16 = manual_logits(net, ctx512_16, None, None, None, chunk=CHUNK_512)
    d16, ce16 = sweep_block(net, ctx512_16, tgt16, clean16, CHUNK_512)
    ce_a, ce_f = float(ce16[:B].mean()), float(ce16[B:].mean())
    log(f"clean CE: battery A {ce_a:.4f} | fresh 12 {ce_f:.4f} "
        f"(e072 flagged the fresh battery as harder — honesty note)")

    # G3: battery-A rows vs e069's stored eval-512 readouts (rows 0..3 of the
    # B=16 sweep ARE the battery-A sweep: the sweep is row-independent)
    ref = E069_REF["eval512"]
    dce_a = d16["dce_age"][:B]
    mean_a = dce_a.mean(0)
    ages15 = mean_a[:5]
    dev15 = float(np.max(np.abs(ages15 - np.asarray(ref["ages1_5"]))))
    ws_dev = abs(float(mean_a[255]) - ref["winstart_dce"])
    a_match = d16["per_seq_a_star"][:B] == ref["per_seq"]
    ce_dev = abs(ce_a - ref["clean_ce"])
    ok3 = bool(a_match and dev15 < 0.05 and ws_dev < 0.05 and ce_dev < 0.02)
    gates["G3_batteryA_vs_e069"] = dict(
        per_seq_a_star=list(d16["per_seq_a_star"][:B]), per_seq_ref=ref["per_seq"],
        per_seq_match=bool(a_match), ages1_5=[float(a) for a in ages15],
        max_dev_ages1_5=dev15, winstart_dev=ws_dev,
        clean_ce=ce_a, clean_ce_dev=ce_dev, ok=ok3)
    log(f"G3 battery A vs e069: per-seq {list(d16['per_seq_a_star'][:B])} "
        f"(ref {ref['per_seq']}), ages1-5 maxdev {dev15:.4f}, CE dev "
        f"{ce_dev:.4f} -> {'PASS' if ok3 else 'FAIL'}")

    # G4: battery-A junk instrument vs e073's stored reanalysis
    b4_prompt = junk_band(dce_a, P512 - PROMPT_POS[1], P512 - PROMPT_POS[0])
    pa4_lo, pa4_hi = gen_position_range(BASE_A_STAR)
    b4_gen = junk_band(dce_a, P512 - pa4_hi, P512 - pa4_lo)
    g4a = abs(b4_gen["junk_frac_mean"] - E073_REF["gen_old_junk"])
    g4b = abs(b4_prompt["junk_frac_max"] - E073_REF["prompt_junk"])
    g4c = abs(b4_gen["mean_dce"] - E073_REF["mean_dce_gen_old"])
    g4d = abs(b4_prompt["mean_dce"] - E073_REF["mean_dce_prompt"])
    gates["G4_junk_instrument_vs_e073"] = dict(
        gen_old_junk=b4_gen["junk_frac_mean"], gen_old_dev=g4a,
        prompt_junk_max=b4_prompt["junk_frac_max"], prompt_dev=g4b,
        mean_dce_gen_old_dev=g4c, mean_dce_prompt_dev=g4d,
        tol=dict(junk_frac=1e-6, mean_dce=2e-3),
        ok=bool(g4a < 1e-6 and g4b < 1e-6 and g4c < 2e-3 and g4d < 2e-3))
    log(f"G4 battery-A junk vs e073 reanalysis: gen_old dev {g4a:.2e}, "
        f"prompt(max) dev {g4b:.2e}, mean-dce devs {g4c:.2e}/{g4d:.2e} "
        f"-> {'PASS' if gates['G4_junk_instrument_vs_e073']['ok'] else 'FAIL'}")

    # G5: the whole B=16 sweep vs e072's stored secondary (same battery)
    ok5 = bool(d16["a_star"] == E072_B16_REF["a_star"]
               and list(d16["per_seq_a_star"]) == list(E072_B16_REF["per_seq"])
               and abs(float(ce16.mean()) - E072_B16_REF["clean_ce"]) < 0.02)
    gates["G5_b16_vs_e072"] = dict(
        a_star=d16["a_star"], a_star_ref=E072_B16_REF["a_star"],
        per_seq=list(d16["per_seq_a_star"]), per_seq_ref=E072_B16_REF["per_seq"],
        per_seq_match=bool(list(d16["per_seq_a_star"]) == list(E072_B16_REF["per_seq"])),
        clean_ce=float(ce16.mean()), clean_ce_ref=E072_B16_REF["clean_ce"],
        clean_ce_dev=abs(float(ce16.mean()) - E072_B16_REF["clean_ce"]),
        a_star_ci=d16["a_star_ci"], a_star_ci_ref=E072_B16_REF["ci"], ok=ok5)
    log(f"G5 B=16 vs e072's stored secondary: a* {d16['a_star']} "
        f"(ref {E072_B16_REF['a_star']}), per-seq match "
        f"{gates['G5_b16_vs_e072']['per_seq_match']}, CE dev "
        f"{gates['G5_b16_vs_e072']['clean_ce_dev']:.4f} "
        f"-> {'PASS' if ok5 else 'FAIL'}")

    # ---- the B=16 source split (THE registered readout)
    prompt16 = junk_band(d16["dce_age"], P512 - PROMPT_POS[1],
                         P512 - PROMPT_POS[0])              # ages 448..510
    a16 = d16["a_star"] if d16["a_star"] is not None else BASE_A_STAR
    g16_lo, g16_hi = gen_position_range(a16)
    gen16 = junk_band(d16["dce_age"], P512 - g16_hi, P512 - g16_lo)
    f16_lo, f16_hi = gen_position_range(BASE_A_STAR)
    gen16_frozen = junk_band(d16["dce_age"], P512 - f16_hi, P512 - f16_lo)
    pos0 = d16["dce_age"][:, P512 - 1]
    log(f"B=16 junk: generated-old (ages {gen16['ages'][0]}..{gen16['ages'][1]}, "
        f"n={gen16['n_per_seq']}/seq) {gen16['junk_frac_mean']:.4f} CI "
        f"[{gen16['junk_frac_mean_ci'][0]:.4f},{gen16['junk_frac_mean_ci'][1]:.4f}] "
        f"| prompt (n={prompt16['n_per_seq']}/seq) "
        f"{prompt16['junk_frac_mean']:.4f} CI "
        f"[{prompt16['junk_frac_mean_ci'][0]:.4f},"
        f"{prompt16['junk_frac_mean_ci'][1]:.4f}] "
        f"| counts gen {gen16['junk_counts_per_seq']} "
        f"prompt {prompt16['junk_counts_per_seq']} | pos0 dCE {pos0.mean():+.4f}")

    halves16 = gen_order_halves(d16["dce_age"], a16)
    conc_late = concentration(halves16["late_generated"], tag="late_half_B16")
    conc_gen = concentration(gen16, tag="generated_band_B16")
    conc_late_a = concentration(halves16["late_generated"], rows=list(range(B)),
                                tag="late_half_batteryA_rows")
    log(f"B=16 secondary (gen order): early(pos {halves16['early_generated']['positions']}) "
        f"junk {halves16['early_generated']['junk_frac_mean']:.4f} vs late"
        f"(pos {halves16['late_generated']['positions']}) "
        f"{halves16['late_generated']['junk_frac_mean']:.4f}")
    log(f"CONCENTRATION: late-band max single-seq share "
        f"{conc_late['max_share']:.3f} (seq {conc_late['max_seq']}, "
        f"{conc_late['counts'][conc_late['max_seq']]}/{conc_late['pooled_total']}) "
        f"| generated-band share {conc_gen['max_share']:.3f} | battery-A-rows "
        f"late share {conc_late_a['max_share']:.3f} | >{CONC_BAR:.0%} flag: "
        f"late {conc_late['flag_heavy']}, gen {conc_gen['flag_heavy']}")

    # BAR 1: the band contrast at B=16 (paired bootstrap over sequences)
    pg, pp = gen16["junk_frac_mean"], prompt16["junk_frac_mean"]
    ratio16 = pg / pp if pp > 0 else float("inf")
    ratio16_json = round(ratio16, 2) if math.isfinite(ratio16) else "inf"
    ratio16_str = f"{ratio16:.2f}" if math.isfinite(ratio16) else "inf"
    per_seq_fracs = np.stack([
        (d16["dce_age"][:, P512 - g16_hi:P512 - g16_lo + 1] <= -THRESH).mean(1),
        (d16["dce_age"][:, P512 - PROMPT_POS[1]:P512 - PROMPT_POS[0] + 1]
         <= -THRESH).mean(1)], 1)                          # (B16, 2)
    d_lo, d_hi = bootstrap_stat(per_seq_fracs, lambda m: float(m[:, 0].mean() - m[:, 1].mean()))
    diff16 = pg - pp
    ratio_ok = bool(ratio16 >= RATIO_BAR and pg > 0)
    diff_excl = bool(d_lo > 0)
    bar1 = bool(ratio_ok and diff_excl)
    # supplementary: bootstrap distribution of the contrast-holding event
    rng = np.random.default_rng(79)
    sel = rng.integers(0, B16, size=(BOOT_N, B16))
    boot_gen = per_seq_fracs[sel, 0].mean(1)
    boot_prm = per_seq_fracs[sel, 1].mean(1)
    p_holds16 = float((boot_gen >= RATIO_BAR * boot_prm).mean())
    p_diff_pos16 = float((boot_gen > boot_prm).mean())
    log(f"BAR 1 (e053c B=16): gen-old {pg:.4f} vs prompt {pp:.4f} | ratio "
        f"{ratio16:.2f} (bar >= {RATIO_BAR:g}) | diff {diff16:+.4f} CI "
        f"[{d_lo:+.4f},{d_hi:+.4f}] excludes 0: {diff_excl} | P(diff>0) "
        f"{p_diff_pos16:.3f} P(2x holds) {p_holds16:.3f} -> "
        f"{'FIRES' if bar1 else 'DOES NOT FIRE'}")

    # ============================================================ PART 2
    log("PART 2: multi-net propagation from runs/e053/metrics.json "
        "(stored final_profiles, no rerun)")
    with open(E053_METRICS) as f:
        e053 = json.load(f)
    rng_mc = np.random.default_rng(SEED_MC)
    multinet = {}
    g6 = {}
    for cell_name in MULTINET_CELLS:
        cell = e053["cells"][cell_name]
        prop = propagate_cell(cell, rng_mc)
        r = E073_FAMILY_REF[cell_name]
        dev_g = abs(prop["point"]["gen_junk"] - r["gen"])
        dev_p = abs(prop["point"]["prompt_junk"] - r["prompt"])
        g6[cell_name] = dict(gen_recon=prop["point"]["gen_junk"],
                             gen_ref=r["gen"], gen_dev=dev_g,
                             prompt_recon=prop["point"]["prompt_junk"],
                             prompt_ref=r["prompt"], prompt_dev=dev_p,
                             tol=0.006,
                             ok=bool(dev_g < 0.006 and dev_p < 0.006))
        multinet[cell_name] = prop
        p16 = prop["propagated_b16"]
        log(f"  {cell_name} (n_seq {prop['n_seq']}, a* {prop['a_star']}): gen "
            f"{prop['point']['gen_junk']:.4f} CI {p16['gen_junk_ci']} vs "
            f"prompt {prop['point']['prompt_junk']:.4f} CI "
            f"{p16['prompt_junk_ci']} | diff CI {p16['diff_ci']} excl0 "
            f"{p16['diff_excludes_0']} | ratio {prop['point']['ratio']} "
            f"| contrast fires {prop['contrast']['fires']} (e073 recon devs "
            f"g {dev_g:.4f} p {dev_p:.4f})")
    g6["note"] = ("e073 stored 3-dp numbers (NOTES.md E073); 10M recon 70/188 "
                  "vs stored 69/188=0.367 — one borderline age (mean dCE "
                  "-0.0107); tol 0.006 documents it; small/mid exact")
    gates["G6_multinet_reconstruction_vs_e073"] = g6
    bar2_cells = [c for c in MULTINET_BAR_CELLS if multinet[c]["contrast"]["fires"]]
    bar2 = len(bar2_cells) >= 1
    log(f"BAR 2 (propagated, >=1 of {MULTINET_BAR_CELLS}): fired by "
        f"{bar2_cells} -> {'FIRES' if bar2 else 'DOES NOT FIRE'}")

    # ---- REGISTERED decision
    conc_flag = bool(conc_late["flag_heavy"] or conc_gen["flag_heavy"])
    clauses = dict(
        bar1_e053c_b16=dict(ratio=round(ratio16, 2) if math.isfinite(ratio16) else "inf",
                            ratio_ge_2=ratio_ok, diff_ci=[d_lo, d_hi],
                            diff_ci_excludes_0=diff_excl, fires=bar1),
        bar2_propagated=dict(pool=MULTINET_BAR_CELLS, fired_by=bar2_cells,
                             fires=bar2),
        concentration_flag=dict(late_band=conc_late, generated_band=conc_gen,
                                bar=CONC_BAR, flagged=conc_flag))
    if bar1 and bar2:
        verdict = (f"BOTH BARS FIRE: the source split holds at B=16 on e053c "
                   f"(gen-old {pg:.4f} vs prompt {pp:.4f}, diff CI "
                   f"[{d_lo:+.4f},{d_hi:+.4f}]) and in the propagated "
                   f"B=16-equivalent CIs of {bar2_cells} — the fractions are "
                   f"not a B=4 threshold artifact.")
    elif bar1 or bar2:
        which = "BAR 1 (e053c B=16)" if bar1 else f"BAR 2 ({bar2_cells})"
        verdict = (f"PARTIAL: {which} fires but the other bar does not — "
                   f"report honestly; the split's status is mixed at B=16.")
    else:
        verdict = ("NEITHER BAR FIRES: the B=16 resample does not support "
                   "the source split at the registered strength.")
    if conc_flag:
        verdict += (f" The per-seq concentration flag SURVIVES: one sequence "
                    f"carries {conc_late['max_share']:.0%} of late-band junk "
                    f"(>{CONC_BAR:.0%} bar).")
    else:
        verdict += (f" Per-seq concentration dilutes at B=16 (late-band max "
                    f"share {conc_late['max_share']:.0%} <= {CONC_BAR:.0%}).")
    log(f"REGISTERED DECISION: {verdict}")

    # ---------------------------------------------------------------- metrics
    def frame_summary(d, ce, role):
        return dict(
            role=role, wpe="native 0..510", a_star=d["a_star"],
            a_star_naive=d["a_star_naive"], a_star_robust=d["a_star_robust"],
            a_star_ci=d["a_star_ci"], per_seq_a_star=list(d["per_seq_a_star"]),
            live_frac=d["live_frac"],
            ages1_5=[float(a) for a in d["mean_age_curve"][:5]],
            mean_curve=d["mean_age_curve"].tolist(),
            ci_lo=d["ci"][0].tolist(), ci_hi=d["ci"][1].tolist(),
            clean_ce_per_seq=ce.tolist(), clean_ce_mean=float(np.mean(ce)),
            clean_ce_batteryA=ce_a, clean_ce_fresh12=ce_f)

    metrics = dict(
        experiment="e079_junk_resample",
        purpose="T047 claim-C registered fix: B=16 resample of the junk "
                "source-split. Part 1 (measured, e053c): regenerate B=16 "
                "sequences (battery = e069's exact 4 [seeds 202/7] + 12 "
                "fresh [seeds 302/17, the e072-documented family extension "
                "+100/+10]); full 511-position V-zero sweep; source split "
                "(junk = dCE <= -0.01 nats; prompt band positions 1..63; "
                "generated band beyond the sweep's own a*) with B=16 "
                "bootstrap CIs. Part 2 (propagated): the stored e053 "
                "final_profiles (0.84M/2.7M/10M) — B=16-equivalent "
                "junk-fraction CIs propagated from the per-age dCE CIs "
                "(sd scaled sqrt(n_seq/16), MC threshold count; box bounds "
                "as the correlation-free worst case). REGISTERED BARS: band "
                "contrast (gen-old >= 2x prompt, CI excluding equality) at "
                "B=16 on e053c AND in propagated CIs of >=1 of "
                "2.7M/10M; concentration flag if one seq > 40% of "
                "late-band junk.",
        started=started, wall_s=elapsed(), threads=THREADS, cpu_only=True,
        net=dict(ckpt=str(CKPT), arch=dict(n_layer=4, n_head=4, n_embd=128,
                                           block_size=T_TOTAL, vocab=65),
                 params=n_params, steps_in_ckpt=steps_in_ckpt,
                 val_ce=val_ce, val_ce_e053c=E053C_VAL_CE),
        seeds=dict(
            corpus=1337,
            battery_A=dict(prompts=SEED_PROMPT, sampling=SEED_SAMPLE,
                           note="e069/e072/e074 verbatim: 8-draw seed 202, "
                                "first B=4; seed-7 B=4 row-order stream"),
            battery_B_extra12=dict(
                prompts=SEED_PROMPT_X, sampling=SEED_SAMPLE_X,
                note="202/7 family extended deterministically exactly as "
                     "e072 documented: +100 prompt seed (202->302), +10 "
                     "sampling seed (7->17); SAME draw rules (val split, "
                     "64-token prompts, randint high len(val)-65, temp 0.8 "
                     "top-k 40, fixed anchor, free-run 64->512); 12-draw + "
                     "B=12 row-order stream; battery = A's 4 + these 12"),
            bootstrap=0, mc_propagation=SEED_MC),
        protocol=dict(B16=B16, prompt_tokens=PROMPT_TOK, t_total=T_TOTAL,
                      temp=TEMP, topk=TOPK, fixed_anchor=True, threshold=THRESH,
                      junk_rule="V-zero dCE <= -0.01 nats (lesion-helpful = "
                                "negative utility)",
                      a_star_window=A_STAR_K, boot_n=BOOT_N, chunk=CHUNK_512,
                      mc_n=MC_N, b_target=B_TARGET,
                      bands=dict(
                          prompt=dict(positions=list(PROMPT_POS),
                                      ages=[P512 - PROMPT_POS[1],
                                            P512 - PROMPT_POS[0]],
                                      n_per_seq=63,
                                      note="position 0 (sequence start, age "
                                           "511) excluded as in e073/e074; "
                                           "measured separately as pos0_dce"),
                          generated=dict(
                              positions="64..510 restricted to ages >= the "
                                        "sweep's own a* (B=16 a* == 6 -> "
                                        "positions 64..505, n=442)",
                              statistic="mean over sequences of the per-seq "
                                        "junk fraction (e073 generated-band "
                                        "convention); max/pooled/counts too"),
                          multinet="e073 age-level convention: fraction of "
                                   "AGES in band whose sequence-mean dCE <= "
                                   "-0.01 (a different statistic from the "
                                   "per-seq-then-mean convention — "
                                   "cross-net comparison is contrast-level)"),
                      honest_flags=["fresh-12 battery is HARDER than battery "
                                    "A (clean CE %.3f vs %.3f; e072 flagged "
                                    "the same) — junk fractions on the B=16 "
                                    "battery mix both difficulties"
                                    % (ce_f, ce_a),
                                    "propagation assumes per-age normality "
                                    "and cross-age independence; within-seq "
                                    "age correlation is not recoverable from "
                                    "stored profiles (box bounds reported as "
                                    "the correlation-free case)",
                                    "mid/10M cells stored at n_seq=2: their "
                                    "per-age percentile CIs are [min,max] of "
                                    "2 sequences; the sqrt(2/16) scaling is "
                                    "the registered B=16-equivalent mapping"]),
        gates=gates,
        e073_reference=E073_REF, e072_b16_reference=E072_B16_REF,
        b4_batteryA=dict(prompt_band=b4_prompt, generated_band=b4_gen,
                         note="battery-A rows of the B=16 sweep = the e069/"
                              "e073 instrument replica (gated G3/G4)"),
        b16=dict(
            frame=frame_summary(d16, ce16, "THE REGISTERED B=16 RESAMPLE"),
            prompt_band=prompt16, generated_band=gen16,
            generated_band_frozen_astar6=gen16_frozen,
            gen_order_halves=halves16,
            pos0_dce=dict(mean=float(pos0.mean()), per_seq=pos0.tolist())),
        concentration=dict(late_band=conc_late, generated_band=conc_gen,
                           batteryA_rows_late=conc_late_a, bar=CONC_BAR,
                           flagged=conc_flag),
        multinet_propagation=multinet,
        registered_decision=dict(
            frozen_bars=dict(
                bar1=f"e053c B=16: gen-old junk >= {RATIO_BAR:g}x prompt junk "
                     f"(point; prompt 0 + gen>0 passes) AND paired-bootstrap "
                     f"CI of (gen - prompt) excludes 0",
                bar2=f"same contrast in the propagated B=16-equivalent CIs of "
                     f">=1 of {MULTINET_BAR_CELLS}",
                concentration=f"one sequence > {CONC_BAR:.0%} of late-band "
                              f"junk -> the flag that survives"),
            bar1=dict(gen_old_junk=pg, prompt_junk=pp, ratio=ratio16_json,
                      ratio_ok=ratio_ok, diff=diff16,
                      diff_ci=[d_lo, d_hi], diff_ci_excludes_0=diff_excl,
                      p_diff_gt0=p_diff_pos16, p_contrast_holds=p_holds16,
                      fires=bar1),
            bar2=dict(pool=MULTINET_BAR_CELLS, fired_by=bar2_cells, fires=bar2),
            concentration_flag=clauses["concentration_flag"],
            clauses=clauses, verdict=verdict),
    )
    save_json(out_dir / "metrics.json", metrics)
    log("metrics.json written")

    plot(out_dir / "junk_resample.png", metrics)
    log(f"plot written; total wall {elapsed():.0f}s")


# ---------------------------------------------------------------------- plot


def plot(path: Path, M: dict):
    dec = M["registered_decision"]
    b16 = M["b16"]
    conc = M["concentration"]
    multi = M["multinet_propagation"]
    r16 = dec["bar1"]["ratio"]
    ratio16_str = f"{r16:.2f}" if isinstance(r16, (int, float)) else str(r16)
    ages = np.arange(1, T_TOTAL)
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    # ---- panel 1: B=16 mean dCE curve with bands + onset
    fr = b16["frame"]
    ax1.fill_between(ages, np.asarray(fr["ci_lo"]), np.asarray(fr["ci_hi"]),
                     alpha=0.15, color="tab:blue")
    ax1.plot(ages, fr["mean_curve"], lw=1.2, color="tab:blue",
             label=f"B=16 mean (a*={fr['a_star']})")
    ax1.axhline(0, color="k", lw=0.6)
    ax1.axhline(THRESH, color="gray", ls=":", lw=0.9)
    ax1.axhline(-THRESH, color="gray", ls=":", lw=0.9)
    ax1.axvline(448, color="tab:purple", ls="--", lw=1.0)
    ax1.text(451, 0.30, "prompt band (ages 448-511)", fontsize=8,
             color="tab:purple", rotation=90, va="top")
    ax1.set_xlim(0, T_TOTAL)
    lo = float(np.nanpercentile(np.asarray(fr["mean_curve"])[10:], 1)) - 0.03
    ax1.set_ylim(lo, 0.35)
    ax1.set_xlabel("cache age (tokens)")
    ax1.set_ylabel("V-zero dCE (nats, mean over B=16 seqs)")
    ax1.legend(fontsize=9)
    ax1.set_title("E079-1 — the B=16 resample sweep: junk rule dCE <= -0.01 "
                  "carves the tail off a broad haze", fontsize=10)

    # ---- panel 2: THE registered readout — junk by band, B=4 replica vs B=16
    b4p, b4g = M["b4_batteryA"]["prompt_band"], M["b4_batteryA"]["generated_band"]
    p16, g16 = b16["prompt_band"], b16["generated_band"]
    labels = ["B=4 prompt\n(63/seq)", "B=4 gen-old\n(442/seq)",
              "B=16 prompt\n(63/seq)", "B=16 gen-old\n(442/seq)"]
    vals = [b4p["junk_frac_mean"], b4g["junk_frac_mean"],
            p16["junk_frac_mean"], g16["junk_frac_mean"]]
    cis = [b4p["junk_frac_mean_ci"], b4g["junk_frac_mean_ci"],
           p16["junk_frac_mean_ci"], g16["junk_frac_mean_ci"]]
    cols = ["tab:blue", "tab:cyan", "tab:red", "tab:orange"]
    x = np.arange(4)
    ax2.bar(x, vals, 0.62, color=cols, alpha=0.85, edgecolor="k", lw=0.5)
    ax2.errorbar(x, vals, yerr=[[v - c[0] for v, c in zip(vals, cis)],
                                [c[1] - v for v, c in zip(vals, cis)]],
                 fmt="none", ecolor="k", lw=1.0, capsize=3)
    for xi, v in zip(x, vals):
        ax2.text(xi, v + 0.006, f"{v:.3f}", ha="center", fontsize=9)
    b1 = dec["bar1"]
    ax2.set_xticks(x, labels, fontsize=8)
    ax2.set_ylabel("junk fraction (dCE <= -0.01 nats)")
    ax2.set_ylim(0, max(0.12, max(vals) * 1.3))
    ax2.set_title("E079-2 — THE REGISTERED BAR 1: band contrast at B=16\n"
                  f"ratio {ratio16_str} (bar >= {RATIO_BAR:g}) | diff CI "
                  f"[{b1['diff_ci'][0]:+.4f},{b1['diff_ci'][1]:+.4f}] excl 0: "
                  f"{b1['diff_ci_excludes_0']} -> "
                  f"{'FIRES' if b1['fires'] else 'NO FIRE'}", fontsize=10)

    # ---- panel 3: per-seq concentration (the surviving flag)
    cl = conc["late_band"]
    counts = cl["counts"]
    x3 = np.arange(len(counts))
    cols3 = ["tab:red" if i == cl["max_seq"] else "tab:gray" for i in x3]
    ax3.bar(x3, counts, 0.7, color=cols3, alpha=0.85, edgecolor="k", lw=0.4)
    ax3.axhline(cl["pooled_total"] / len(counts), color="tab:blue", ls="--",
                lw=1.0, label=f"uniform share ({1/len(counts):.1%})")
    ax3.set_xticks(x3, [f"A{i}" if i < B else f"F{i - B}" for i in x3],
                   fontsize=7)
    ax3.set_xlabel("sequence (A0-A3 = battery A / e069; F0-F11 = fresh 302/17)")
    ax3.set_ylabel("late-band junk entries per sequence")
    ax3.legend(fontsize=8)
    ax3.set_title("E079-3 — per-seq CONCENTRATION (T047's flag)\n"
                  f"late-band max share {cl['max_share']:.0%} (seq "
                  f"{cl['max_seq']}, {counts[cl['max_seq']]}/{cl['pooled_total']})"
                  f" | gen-band {conc['generated_band']['max_share']:.0%} | "
                  f">{CONC_BAR:.0%} flag: "
                  f"{'SURVIVES' if conc['flagged'] else 'diluted'}", fontsize=10)

    # ---- panel 4: multi-net propagated junk fractions
    names = ["e053c\nB=16\n(measured)"] + [c.replace("_", "\n") for c in MULTINET_CELLS]
    gm = [g16["junk_frac_mean"]] + [multi[c]["point"]["gen_junk"] for c in MULTINET_CELLS]
    pm = [p16["junk_frac_mean"]] + [multi[c]["point"]["prompt_junk"] for c in MULTINET_CELLS]
    gc = [g16["junk_frac_mean_ci"]] + [multi[c]["propagated_b16"]["gen_junk_ci"]
                                       for c in MULTINET_CELLS]
    pc = [p16["junk_frac_mean_ci"]] + [multi[c]["propagated_b16"]["prompt_junk_ci"]
                                       for c in MULTINET_CELLS]
    x4 = np.arange(len(names))
    w = 0.36
    ax4.bar(x4 - w / 2, gm, w, color="tab:orange", alpha=0.9,
            label="generated-old junk")
    ax4.errorbar(x4 - w / 2, gm,
                 yerr=[[v - c[0] for v, c in zip(gm, gc)],
                       [c[1] - v for v, c in zip(gm, gc)]],
                 fmt="none", ecolor="k", lw=1.0, capsize=3)
    ax4.bar(x4 + w / 2, pm, w, color="tab:purple", alpha=0.9,
            label="prompt-band junk")
    ax4.errorbar(x4 + w / 2, pm,
                 yerr=[[v - c[0] for v, c in zip(pm, pc)],
                       [c[1] - v for v, c in zip(pm, pc)]],
                 fmt="none", ecolor="k", lw=1.0, capsize=3)
    ax4.set_xticks(x4, names, fontsize=8)
    ax4.set_ylabel("junk fraction")
    ax4.legend(fontsize=9)
    ax4.set_title("E079-4 — multi-net picture: measured B=16 (e053c) + "
                  "PROPAGATED B=16-equivalent CIs\n(e053 stored per-age dCE "
                  "CIs, sd x sqrt(n_seq/16); 2.7M/10M are BAR 2's pool)", fontsize=10)

    # ---- panel 5: propagated/gen-band mean dCE per net (the haze)
    for i, c in enumerate(MULTINET_CELLS):
        ax5.scatter([i], multi[c]["point"]["mean_dce_gen"], color="tab:orange",
                    zorder=5, s=60)
        ax5.text(i, multi[c]["point"]["mean_dce_gen"] + 0.0004,
                 f"{multi[c]['point']['mean_dce_gen']:+.4f}", ha="center",
                 fontsize=8, color="tab:orange")
        ax5.scatter([i], multi[c]["point"]["mean_dce_prompt"], color="tab:purple",
                    zorder=5, s=60, marker="s")
        ax5.text(i, multi[c]["point"]["mean_dce_prompt"] - 0.0009,
                 f"{multi[c]['point']['mean_dce_prompt']:+.4f}", ha="center",
                 fontsize=8, color="tab:purple")
    ax5.scatter([3], g16["mean_dce"], color="tab:red", zorder=5, s=60, marker="^")
    ax5.text(3, g16["mean_dce"] + 0.0004, f"{g16['mean_dce']:+.4f}",
             ha="center", fontsize=8, color="tab:red")
    ax5.scatter([3], p16["mean_dce"], color="tab:blue", zorder=5, s=60,
                marker="v")
    ax5.text(3, p16["mean_dce"] - 0.0009, f"{p16['mean_dce']:+.4f}",
             ha="center", fontsize=8, color="tab:blue")
    ax5.axhline(0, color="k", lw=0.7)
    ax5.axhline(-THRESH, color="gray", ls=":", lw=1.0)
    ax5.set_xticks(range(4), [c.replace("_", " ") for c in MULTINET_CELLS]
                   + ["e053c B=16"], fontsize=8)
    ax5.set_ylabel("band mean dCE (nats)")
    ax5.set_title("E079-5 — the T047 haze honesty check: band MEAN dCE "
                  "(circles/squares = age-level cells, triangles = per-seq "
                  "B=16)\nonly 10M and e053c have genuinely negative "
                  "gen-band means; the rule still splits the tail", fontsize=10)

    # ---- panel 6: verdict text
    ax6.axis("off")
    b2 = dec["bar2"]
    lines = [
        "REGISTERED (frozen before measurement):",
        f"  BAR 1: e053c B=16 gen-old >= {RATIO_BAR:g}x prompt, diff CI excl 0",
        f"  BAR 2: same in propagated CIs of >=1 of {', '.join(MULTINET_BAR_CELLS)}",
        f"  FLAG: one seq > {CONC_BAR:.0%} of late-band junk",
        "",
        f"B=16 e053c: gen-old {b1['gen_old_junk']:.4f} CI "
        f"{g16['junk_frac_mean_ci']} | prompt {b1['prompt_junk']:.4f} CI "
        f"{p16['junk_frac_mean_ci']}",
        f"  ratio {ratio16_str} | diff CI "
        f"[{b1['diff_ci'][0]:+.4f},{b1['diff_ci'][1]:+.4f}] | "
        f"P(diff>0) {b1['p_diff_gt0']:.3f}",
    ]
    for c in MULTINET_CELLS:
        m = multi[c]
        lines.append(
            f"{c} (n_seq {m['n_seq']}, a* {m['a_star']}): gen "
            f"{m['point']['gen_junk']:.3f} {m['propagated_b16']['gen_junk_ci']} "
            f"| prompt {m['point']['prompt_junk']:.3f} "
            f"{m['propagated_b16']['prompt_junk_ci']}"
            f"{' << BAR-2 cell' if c in MULTINET_BAR_CELLS else ''}")
    lines += [
        "",
        f"BAR 1 {'FIRES' if b1['fires'] else 'NO'} | BAR 2 "
        f"{'FIRES via ' + ', '.join(b2['fired_by']) if b2['fires'] else 'NO'} | "
        f"concentration flag "
        f"{'SURVIVES' if conc['flagged'] else 'diluted'} "
        f"(late {cl['max_share']:.0%})",
        "",
        "VERDICT:",
    ] + [f"  {w}" for w in _wrap(dec["verdict"], 92)]
    ax6.text(0.02, 0.97, "E079 — T047 claim-C fix: B=16 junk resample",
             fontsize=13, weight="bold", va="top")
    for i, t in enumerate(lines):
        ax6.text(0.02, 0.935 - i * 0.0365, t, fontsize=8.2, va="top",
                 family="monospace")

    fig.suptitle("E079 — B=16 junk source-split resample | BAR1 "
                 f"{'FIRES' if dec['bar1']['fires'] else 'no'} | BAR2 "
                 f"{'FIRES' if dec['bar2']['fires'] else 'no'} | conc flag "
                 f"{'SURVIVES' if dec['concentration_flag']['flagged'] else 'diluted'}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _wrap(text: str, width: int):
    import textwrap
    return textwrap.wrap(text, width=width)


if __name__ == "__main__":
    main()
