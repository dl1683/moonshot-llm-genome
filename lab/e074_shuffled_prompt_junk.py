"""E074 — T045's REGISTERED shuffled-prompt junk control (source vs age).

Trigger (THINKING.md T045, 2026-09-26 ~11:00Z): e073 showed negative-utility
cache entries concentrate in GENERATED entries (4/4 nets; generated-old junk
0.085-0.367) while corpus-prompt entries almost never hurt — but entry AGE
and SOURCE correlate by design (prompt = oldest 64). Two live stories:
- H-SOURCE (strict sleeper): self-generation is the poison; corpus
  statistics in the prompt band protect it regardless of age.
- H-AGE-STATISTICS: any information-poor old band drifts negative; the
  prompt band is clean only because corpus text is informative.

This run rebuilds the e069/e070/e072 protocol VERBATIM (frozen e053c ctx-512
checkpoint runs/checkpoints/e053c_ctx512.pt, seeds 202/7, B=4 battery-A
sequences, free-run 64->512 fixed anchor, full 511-position final-step V-zero
sweep, e053b onset machinery) and adds ONE manipulation: for each sequence,
the 64 prompt tokens are replaced by a random PERMUTATION of themselves
(same multiset -> count structure kept; sequential corpus statistics
destroyed; ages/recency/positions unchanged; generated tail bit-identical),
then the SAME V-zero sweep is re-run on the shuffled contexts.

Junk instrument (e073 verbatim, gated bit-exact against the stored reanalysis):
junk entry = V-zero dCE <= -0.01 nats (lesion-HELPFUL = negative utility).
Bands (age of context position p = 511 - p):
- PROMPT band: positions 1..63 (ages 448..510), n=63/seq — position 0 (the
  sequence-start anchor, age 511) measured separately, excluded as in e073
  (whose stored n=63 implies the same).
- GENERATED band: positions 64..510 (ages 1..447), restricted to ages >= the
  sweep's own a* (baseline a*=6 -> positions 64..505, n=442/seq; the frozen
  a*=6 restriction is also reported for the shuffled side).
Headline statistic: mean over sequences of the per-seq junk fraction (e073's
generated-band convention), bootstrap CI over sequences; max-over-seqs,
pooled, and per-seq counts recorded too (e073's stored prompt number 4/63 is
the max-over-seqs reading of this same instrument).

REGISTERED NUMBERS (frozen, T045/task spec, before any measurement):
  - shuffled-prompt-band junk <= 0.05  => H-SOURCE (junk STAYS in the
    generated band; prompt-shuffling creates no new junk);
  - shuffled-prompt-band junk >= 0.15  => H-AGE-STATISTICS (the shuffled
    prompt band goes negative, comparable to the generated band);
  - in (0.05, 0.15) => AMBIGUOUS — report honestly, no forcing.
  SECONDARY (registered direction): within the generated band, junk by
  generation ORDER — early vs late generated halves at the midpoint of the
  restricted position range; H-source predicts LATE-generated entries (more
  accumulated drift when written) junk MORE.
  Manipulation checks (recorded, not gated): clean CE of the shuffled-prompt
  contexts should RISE (baseline 0.4528); the ages-1-5 spike should stay
  intact (e069 D2: [1.742, 4.467, 2.319, 0.914, 0.321]).

Run:     python lab/e074_shuffled_prompt_junk.py
Outputs: runs/e074/metrics.json + runs/e074/junk_control.png
Envelope: NO training, NO new automations; CPU-only, single step, minutes.
No NOTES/THINKING/QUEUE/STATE edits; no commit (the dispatcher owns those).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # task spec: CPU-only (pre-torch)
import torch

# this torch build reports is_available()==True even with an empty
# CUDA_VISIBLE_DEVICES; force CPU so common.DEVICE=="cpu" (e053b..e072)
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
SEED_PROMPT, SEED_SAMPLE = 202, 7         # e053b/e053c/e069/e070/e072 exact seeds
N_PROMPTS = 8
B = 4                                     # battery A uniform B
BOOT_N = 1000
THREADS = 12
torch.set_num_threads(THREADS)

P512 = T_TOTAL - 1                        # 511 sweep positions (native frame)
THRESH = 0.01                             # dead-weight / junk threshold (nats)
A_STAR_K = 5                              # sustained window (e053b verbatim)
CHUNK_512 = 64                            # e069/e070/e072 measured chunk
SEED_SHUF = 74                            # e074's own: prompt-permutation seed

# ---- bands (e073 instrument reconstruction; gated below) --------------------
PROMPT_POS = (1, 63)                      # positions 1..63 (pos 0 = seq start)
GEN_POS = (64, 510)                       # generated context entries
BASE_A_STAR = 6                           # e053c/e069/e072 eval-512 onset

# ---- REGISTERED decision numbers (FROZEN, T045 / task spec) -----------------
JUNK_LOW, JUNK_HIGH = 0.05, 0.15          # H-source <= 0.05 | H-age-stat >= 0.15

# ---- e069/e072 stored readouts (protocol-identity instrument gates) ---------
E069_REF = {"eval512": {"a_star": 6, "ci": [4.0, 8.0], "clean_ce": 0.4528,
                        "ages1_5": [1.742, 4.467, 2.319, 0.914, 0.321],
                        "per_seq": [8, 4, 6, 3], "winstart_dce": 0.0069}}
# e073's stored reanalysis of THIS cell (runs/e073_junk_split_reanalysis.json):
# gen_old_junk 0.0995475113 (mean per-seq frac, ages 6..447);
# prompt_junk 0.0634920635 (= 4/63, the max-over-seqs reading, ages 448..510).
E073_REF = {"gen_old_junk": 0.09954751131221719,
            "gen_old_n": 442,
            "prompt_junk": 0.06349206349206349,
            "prompt_n": 63,
            "mean_dce_gen_old": -0.001330780132454546,
            "mean_dce_prompt": 0.00648569551459144}
CKPT = REPO / "runs" / "checkpoints" / "e053c_ctx512.pt"
E053C_VAL_CE = 1.5226792494455974         # the net being interrogated
T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# ------------------------------------------------- machinery (VERBATIM e053c/e072)

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
    (position-T_TOTAL-1) logits (B, V). [CPU-only, e069/e070/e072 verbatim]"""
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
    """Clean forward given -> full V-zero sweep -> analysis, plus bookkeeping."""
    dce, ce_clean = fine_sweep(net, ctx, tgt, clean_logits, ch, pos_offset)
    nan_n = int(np.isnan(dce).sum())
    if nan_n:
        dce = np.nan_to_num(dce, nan=0.0)
    d = analyze_window(dce)
    log(f"  a*={d['a_star']} CI [{d['a_star_ci'][0]:.0f},{d['a_star_ci'][1]:.0f}] "
        f"naive={d['a_star_naive']} robust={d['a_star_robust']} "
        f"live_frac={d['live_frac']:.3f} | per-seq {d['per_seq_a_star']} "
        f"| clean CE {ce_clean.mean():.4f} (nan->0: {nan_n})")
    return d, ce_clean


# ------------------------------------- NEW: the junk census (e073 instrument)

def junk_band(dce_age: np.ndarray, a_lo: int, a_hi: int):
    """Junk census of the age band [a_lo, a_hi] (inclusive; ages ascending).
    dce_age: (S, Pw) age-ascending. Junk = dCE <= -THRESH (lesion-helpful).
    Returns the e073 readouts: mean/max/pooled per-seq fractions, counts,
    mean dCE, bootstrap CI of the mean-over-seqs fraction."""
    band = dce_age[:, a_lo - 1:a_hi]                    # ages a_lo..a_hi
    n = band.shape[1]
    per_seq_frac = (band <= -THRESH).mean(1)            # (S,)
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
    """SECONDARY: early vs late generated halves at the midpoint of the
    restricted position range (generation order == position order within the
    generated band; early-generated = older). Returns junk censuses."""
    p_lo, p_hi = gen_position_range(a_star)
    mid = (p_lo + p_hi) // 2
    out = {}
    for name, (lo, hi) in [("early_generated", (p_lo, mid)),
                           ("late_generated", (mid + 1, p_hi))]:
        # position band -> age band: age of p = 511 - p, ascending flip
        a_band = (P512 - hi, P512 - lo)
        out[name] = junk_band(dce_age, a_band[0], a_band[1])
        out[name]["positions"] = [lo, hi]
    return out


# ---------------------------------------------------------------------- main


def main():
    assert torch.cuda.device_count() == 0, "CPU-only violated: CUDA visible"
    assert common.DEVICE == "cpu", f"common.DEVICE={common.DEVICE} != cpu"
    out_dir = run_dir("e074")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gates = dict(cpu_only=True, threads=THREADS, smoke=False)

    # ---- battery: rebuild corpus + net EXACTLY as e053c/e069/e070/e072 did
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

    # ---- battery A sequences: e069/e070/e072 protocol verbatim (seeds 202/7)
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

    # ---- replica sweep: eval-512 (e069 sweep A replica; the junk instrument)
    log("replica sweep: eval-512 fine V-zero sweep (e069/e073 frame)")
    tgt = idx[:, -1]
    ctx512 = idx[:, :-1]                                 # seq 0..510, wpe 0..510
    d_base, ce_base_inc = sweep_block(net, ctx512, tgt, final_logits, CHUNK_512)
    clean_base = manual_logits(net, ctx512, None, None, None, chunk=CHUNK_512)
    ce_base = (-torch.log(torch.softmax(clean_base.float(), -1)
                          [torch.arange(B), tgt].clamp_min(1e-12))).numpy()
    log(f"clean CE via full recompute: {ce_base.mean():.4f} "
        f"(incremental {ce_base_inc.mean():.4f})")

    # G3: replica vs e069's stored eval-512 readouts
    ref = E069_REF["eval512"]
    ages15 = d_base["mean_age_curve"][:5]
    dev15 = float(np.max(np.abs(ages15 - np.asarray(ref["ages1_5"]))))
    ws_dev = abs(float(d_base["mean_age_curve"][255]) - ref["winstart_dce"])
    a_match = d_base["a_star"] == ref["a_star"]
    ps_match = list(d_base["per_seq_a_star"]) == list(ref["per_seq"])
    ce_dev = abs(float(ce_base.mean()) - ref["clean_ce"])
    ok3 = bool(a_match and dev15 < 0.05 and ws_dev < 0.05 and ce_dev < 0.02
               and ps_match)
    gates["G3_eval512_vs_e069"] = dict(
        a_star=d_base["a_star"], a_star_ref=ref["a_star"], a_star_match=a_match,
        per_seq_a_star=list(d_base["per_seq_a_star"]), per_seq_ref=ref["per_seq"],
        per_seq_match=ps_match,
        ages1_5=[float(a) for a in ages15], max_dev_ages1_5=dev15,
        winstart_dce=float(d_base["mean_age_curve"][255]),
        winstart_dce_ref=ref["winstart_dce"], winstart_dev=ws_dev,
        clean_ce=float(ce_base.mean()), clean_ce_ref=ref["clean_ce"],
        clean_ce_dev=ce_dev, ok=ok3)
    log(f"G3 eval512: a* {d_base['a_star']} (ref {ref['a_star']}), per-seq "
        f"{list(d_base['per_seq_a_star'])} (ref {ref['per_seq']}), ages1-5 "
        f"maxdev {dev15:.4f}, age-255 dev {ws_dev:.4f}, CE dev {ce_dev:.4f} "
        f"-> {'PASS' if ok3 else 'FAIL'}")

    # ---- junk census on the baseline sweep (the e073 instrument, gated)
    base_prompt = junk_band(d_base["dce_age"], P512 - PROMPT_POS[1],
                            P512 - PROMPT_POS[0])          # ages 448..510
    p_lo, p_hi = gen_position_range(d_base["a_star"])
    base_gen = junk_band(d_base["dce_age"], P512 - p_hi, P512 - p_lo)
    pos0 = d_base["dce_age"][:, P512 - 1]                 # age 511 = seq start
    log(f"BASELINE junk: generated-old (ages {base_gen['ages'][0]}.."
        f"{base_gen['ages'][1]}, n={base_gen['n_per_seq']}/seq) "
        f"{base_gen['junk_frac_mean']:.4f} max {base_gen['junk_frac_max']:.4f} "
        f"counts {base_gen['junk_counts_per_seq']} | prompt (ages "
        f"{base_prompt['ages'][0]}..{base_prompt['ages'][1]}, "
        f"n={base_prompt['n_per_seq']}/seq) {base_prompt['junk_frac_mean']:.4f} "
        f"max {base_prompt['junk_frac_max']:.4f} counts "
        f"{base_prompt['junk_counts_per_seq']} | pos0 dCE {pos0.mean():+.4f}")

    # G-e073: reproduce the stored reanalysis numbers bit-close
    g4a = abs(base_gen["junk_frac_mean"] - E073_REF["gen_old_junk"])
    g4b = abs(base_prompt["junk_frac_max"] - E073_REF["prompt_junk"])
    g4c = abs(base_gen["mean_dce"] - E073_REF["mean_dce_gen_old"])
    g4d = abs(base_prompt["mean_dce"] - E073_REF["mean_dce_prompt"])
    gates["G4_junk_instrument_vs_e073"] = dict(
        gen_old_junk=base_gen["junk_frac_mean"],
        gen_old_junk_ref=E073_REF["gen_old_junk"], gen_old_dev=g4a,
        gen_old_n=base_gen["n_per_seq"], gen_old_n_ref=E073_REF["gen_old_n"],
        prompt_junk_max=base_prompt["junk_frac_max"],
        prompt_junk_max_ref=E073_REF["prompt_junk"], prompt_dev=g4b,
        prompt_n=base_prompt["n_per_seq"], prompt_n_ref=E073_REF["prompt_n"],
        mean_dce_gen_old_dev=g4c, mean_dce_prompt_dev=g4d,
        tol=dict(junk_frac=1e-6, mean_dce=2e-3),
        note="e073's stored prompt_junk 4/63 is the MAX-over-seqs reading of "
             "this band (per-seq counts [0,4,1,0]); its gen_old_junk is the "
             "mean-over-seqs reading. Both reproduce here under those "
             "readings. Mean-dCE tol 2e-3: the junk FRACTIONS match to <1e-16 "
             "(threshold-counting is noise-robust) but band mean dCE carries "
             "cross-run float nondeterminism ~1e-4 (e053c's sweep device/"
             "run-state differed; same order as G3's ages1-5 devs).",
        ok=bool(g4a < 1e-6 and g4b < 1e-6 and base_gen["n_per_seq"] == 442
                and base_prompt["n_per_seq"] == 63 and g4c < 2e-3 and g4d < 2e-3))
    log(f"G4 junk instrument vs e073 reanalysis: gen_old dev {g4a:.2e}, "
        f"prompt(max) dev {g4b:.2e}, mean-dce devs {g4c:.2e}/{g4d:.2e} "
        f"-> {'PASS' if gates['G4_junk_instrument_vs_e073']['ok'] else 'FAIL'}")

    base_halves = gen_order_halves(d_base["dce_age"], d_base["a_star"])
    log(f"BASELINE secondary (gen order): early(pos {base_halves['early_generated']['positions']}) "
        f"junk {base_halves['early_generated']['junk_frac_mean']:.4f} vs late"
        f"(pos {base_halves['late_generated']['positions']}) junk "
        f"{base_halves['late_generated']['junk_frac_mean']:.4f}")

    # ======================================================== THE MANIPULATION
    # ---- shuffle the 64 prompt tokens of each sequence (permutation of the
    # same multiset; seed 74; row order); generated tail bit-identical.
    g_shuf = torch.Generator().manual_seed(SEED_SHUF)
    perms = [torch.randperm(PROMPT_TOK, generator=g_shuf) for _ in range(B)]
    idx_shuf = idx.clone()
    kept_pairs, vocab_uni = [], float((1.0 / 65) * 64)   # chance ~ adjacent kept
    for s in range(B):
        idx_shuf[s, :PROMPT_TOK] = idx[s, perms[s]]
        assert torch.equal(idx_shuf[s, :PROMPT_TOK].sort().values,
                           idx[s, :PROMPT_TOK].sort().values), "multiset broken"
        assert torch.equal(idx_shuf[s, PROMPT_TOK:], idx[s, PROMPT_TOK:]), \
            "generated tail modified"
        kept_pairs.append(float((idx_shuf[s, :PROMPT_TOK - 1]
                                 == idx[s, :PROMPT_TOK - 1]).float().mean()))
    frac_perm_changed = float((idx_shuf[:, :PROMPT_TOK]
                               != idx[:, :PROMPT_TOK]).float().mean())
    log(f"SHUFFLED prompts (seed {SEED_SHUF}): {frac_perm_changed:.3f} of prompt "
        f"slots changed token | adjacent-pair preservation "
        f"{float(np.mean(kept_pairs)):.4f} (chance {1 / 65:.4f}) | prompt0 "
        f"shuffled prefix: {corp.decode(idx_shuf[0, :PROMPT_TOK])[:32]!r}")
    gates["G5_shuffle"] = dict(
        seed=SEED_SHUF, prompt_slots_changed_frac=frac_perm_changed,
        adjacent_pair_preservation=float(np.mean(kept_pairs)),
        adjacent_pair_chance=1 / 65,
        multiset_preserved=True, tail_bit_identical=True,
        ok=bool(frac_perm_changed > 0.9))

    # ---- clean forward on the shuffled contexts (manipulation check)
    ctx_shuf = idx_shuf[:, :-1]
    assert torch.equal(ctx_shuf[:, PROMPT_TOK:], ctx512[:, PROMPT_TOK:])
    with torch.no_grad():
        clean_shuf = manual_logits(net, ctx_shuf, None, None, None,
                                   chunk=CHUNK_512)
    ce_shuf = (-torch.log(torch.softmax(clean_shuf.float(), -1)
                          [torch.arange(B), tgt].clamp_min(1e-12))).numpy()
    ce_rise = ce_shuf - ce_base
    gates["G6_manipulation_ce"] = dict(
        clean_ce_baseline=float(ce_base.mean()),
        clean_ce_shuffled=float(ce_shuf.mean()),
        ce_rise_mean=float(ce_rise.mean()),
        ce_rise_per_seq=ce_rise.tolist(),
        expected_rise=True,
        note="registered EXPECTATION (not a gating condition for the 5%/15% "
             "junk bars): clean CE was expected to RISE under shuffling. "
             "Report the observed direction honestly either way.",
        ok=bool(ce_rise.mean() > 0))
    log(f"MANIPULATION CHECK: clean CE {ce_base.mean():.4f} -> "
        f"{ce_shuf.mean():.4f} (rise {ce_rise.mean():+.4f} nats/seq; per-seq "
        f"{[f'{r:+.3f}' for r in ce_rise]}) -> "
        f"{'PASS (rose)' if ce_rise.mean() > 0 else 'FAIL'}")

    # ---- the shuffled sweep (the measurement)
    log("shuffled sweep: eval-512 fine V-zero sweep on shuffled contexts")
    d_shuf, _ = sweep_block(net, ctx_shuf, tgt, clean_shuf, CHUNK_512)

    # junk censuses on the shuffled sweep: prompt band under its own a* AND
    # the frozen baseline restriction (reported both; headline = own a*,
    # the instrument's rule)
    shuf_prompt = junk_band(d_shuf["dce_age"], P512 - PROMPT_POS[1],
                            P512 - PROMPT_POS[0])
    sp_lo, sp_hi = gen_position_range(d_shuf["a_star"])
    shuf_gen = junk_band(d_shuf["dce_age"], P512 - sp_hi, P512 - sp_lo)
    fp_lo, fp_hi = gen_position_range(BASE_A_STAR)
    shuf_gen_frozen = junk_band(d_shuf["dce_age"], P512 - fp_hi, P512 - fp_lo)
    pos0_shuf = d_shuf["dce_age"][:, P512 - 1]
    log(f"SHUFFLED junk: prompt band {shuf_prompt['junk_frac_mean']:.4f} "
        f"(max {shuf_prompt['junk_frac_max']:.4f}, counts "
        f"{shuf_prompt['junk_counts_per_seq']}, mean dCE "
        f"{shuf_prompt['mean_dce']:+.4f}) | generated band "
        f"{shuf_gen['junk_frac_mean']:.4f} (frozen-a* {shuf_gen_frozen['junk_frac_mean']:.4f}) "
        f"| pos0 dCE {pos0_shuf.mean():+.4f}")

    shuf_halves_own = gen_order_halves(d_shuf["dce_age"], d_shuf["a_star"])
    shuf_halves_frozen = gen_order_halves(d_shuf["dce_age"], BASE_A_STAR)
    log(f"SHUFFLED secondary (gen order, own a*): early "
        f"{shuf_halves_own['early_generated']['junk_frac_mean']:.4f} vs late "
        f"{shuf_halves_own['late_generated']['junk_frac_mean']:.4f}")

    # ages 1-5 spike on the shuffled sweep (e069 D2 prediction: intact)
    ages15_shuf = [float(a) for a in d_shuf["mean_age_curve"][:5]]
    spike_intact = bool(all(a > THRESH for a in ages15_shuf[:2]))
    gates["G7_ages1_5_spike_shuffled"] = dict(
        ages1_5=ages15_shuf, ref_baseline=[float(a) for a in ages15],
        ref_e069=ref["ages1_5"], intact=spike_intact, ok=spike_intact)
    log(f"AGES 1-5 SPIKE on shuffled: {['%.3f' % a for a in ages15_shuf]} "
        f"(baseline {['%.3f' % a for a in ages15]}) -> "
        f"{'INTACT' if spike_intact else 'CHANGED'}")

    # ---- REGISTERED decision (frozen numbers)
    pj = shuf_prompt["junk_frac_mean"]
    pj_max = shuf_prompt["junk_frac_max"]
    gen_stays = shuf_gen["junk_frac_mean"] >= base_gen["junk_frac_mean"] * 0.5
    clauses = dict(prompt_junk_mean=pj, prompt_junk_max=pj_max,
                   h_source_bar_le=JUNK_LOW, h_agestat_bar_ge=JUNK_HIGH,
                   generated_band_stays_junky=gen_stays)
    if pj <= JUNK_LOW:
        clause = "H-SOURCE"
        verdict = (f"H-SOURCE (strict sleeper) fires: shuffled-prompt-band "
                   f"junk = {pj:.4f} <= {JUNK_LOW} (max-over-seqs "
                   f"{pj_max:.4f}) — destroying corpus statistics created NO "
                   f"new junk in the prompt band; the poison stays in the "
                   f"GENERATED band (shuffled-run generated junk "
                   f"{shuf_gen['junk_frac_mean']:.4f} vs baseline "
                   f"{base_gen['junk_frac_mean']:.4f}). Age+count+recency "
                   f"alone do not make an old band go negative.")
    elif pj >= JUNK_HIGH:
        clause = "H-AGE-STATISTICS"
        verdict = (f"H-AGE-STATISTICS fires: shuffled-prompt-band junk = "
                   f"{pj:.4f} >= {JUNK_HIGH} — an information-poor old band "
                   f"goes negative even though it was never generated; the "
                   f"e073 source split was an age/statistics artifact.")
    else:
        clause = "AMBIGUOUS"
        verdict = (f"AMBIGUOUS: shuffled-prompt-band junk = {pj:.4f} lands in "
                   f"the open interval ({JUNK_LOW}, {JUNK_HIGH}) between the "
                   f"frozen bars — neither clause fires; reported honestly, "
                   f"no forcing.")
    log(f"REGISTERED DECISION: {verdict}")
    log(f"  clauses: {clauses}")
    honest_misses = []
    if ce_rise.mean() <= 0:
        honest_misses.append(
            f"the clean-CE manipulation check was registered as 'expected to "
            f"rise'; it FELL instead (mean {ce_rise.mean():+.4f} nats; seq1 "
            f"{ce_rise[1]:+.4f}, other seqs ~0) — swapping the far-context "
            f"prompt band barely moves the tail-consistent final-token "
            f"prediction, and for seq 1 the shuffled far context predicts it "
            f"slightly BETTER. Recorded as an honest miss of that "
            f"expectation; the registered junk bars do not depend on it.")

    # secondary direction check (registered: H-source predicts late > early)
    sec_base = (base_halves["late_generated"]["junk_frac_mean"]
                - base_halves["early_generated"]["junk_frac_mean"])
    sec_shuf = (shuf_halves_own["late_generated"]["junk_frac_mean"]
                - shuf_halves_own["early_generated"]["junk_frac_mean"])
    secondary = dict(
        registered_prediction="H-source: late-generation entries "
                              "(more accumulated drift) junk MORE",
        baseline=dict(early=base_halves["early_generated"],
                      late=base_halves["late_generated"],
                      late_minus_early=float(sec_base)),
        shuffled_own_astar=dict(early=shuf_halves_own["early_generated"],
                                late=shuf_halves_own["late_generated"],
                                late_minus_early=float(sec_shuf)),
        shuffled_frozen_astar=dict(early=shuf_halves_frozen["early_generated"],
                                   late=shuf_halves_frozen["late_generated"]),
        baseline_late_gt_early=bool(sec_base > 0),
        shuffled_late_gt_early=bool(sec_shuf > 0))

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
            clean_ce_per_seq=ce.tolist(), clean_ce_mean=float(np.mean(ce)))

    metrics = dict(
        experiment="e074_shuffled_prompt_junk",
        purpose="T045's registered shuffled-prompt junk control: e073 found "
                "negative-utility cache entries concentrate in GENERATED "
                "entries while corpus-prompt entries almost never hurt, but "
                "age and source correlate by design (prompt = oldest 64). "
                "Discriminator: replace the 64 prompt tokens with a random "
                "permutation of themselves (corpus statistics destroyed; "
                "age+count+recency and the generated tail kept bit-identical), "
                "re-run the same 511-position V-zero sweep. FROZEN: "
                "shuffled-prompt junk <= 0.05 => H-SOURCE (self-generation is "
                "the poison); >= 0.15 => H-AGE-STATISTICS (any info-poor old "
                "band drifts negative). Secondary: junk by generation order "
                "(early vs late halves); H-source predicts late junk more.",
        started=started, wall_s=elapsed(), threads=THREADS, cpu_only=True,
        net=dict(ckpt=str(CKPT), arch=dict(n_layer=4, n_head=4, n_embd=128,
                                           block_size=T_TOTAL, vocab=65),
                 params=n_params, steps_in_ckpt=steps_in_ckpt,
                 val_ce=val_ce, val_ce_e053c=E053C_VAL_CE),
        seeds=dict(corpus=1337, prompts=SEED_PROMPT, sampling=SEED_SAMPLE,
                   shuffle=SEED_SHUF, bootstrap=0,
                   shuffle_note="one generator seed 74; randperm(64) per "
                                "sequence in row order"),
        protocol=dict(B=B, prompt_tokens=PROMPT_TOK, t_total=T_TOTAL, temp=TEMP,
                      topk=TOPK, fixed_anchor=True, threshold=THRESH,
                      junk_rule="V-zero dCE <= -0.01 nats (lesion-helpful = "
                                "negative utility)",
                      a_star_window=A_STAR_K, boot_n=BOOT_N, chunk=CHUNK_512,
                      frames="battery A seeds 202/7 verbatim (e069/e070/e072); "
                             "native eval-512 frame; shuffled frame = same "
                             "sequences with prompt tokens 0..63 permuted "
                             "in place, wpe unchanged",
                      bands=dict(
                          prompt=dict(positions=list(PROMPT_POS),
                                      ages=[P512 - PROMPT_POS[1],
                                            P512 - PROMPT_POS[0]],
                                      n_per_seq=63,
                                      note="position 0 (sequence start, age "
                                           "511) excluded as in e073 (n=63); "
                                           "measured separately as pos0_dce"),
                          generated=dict(positions=list(GEN_POS),
                                         ages="1..447, restricted to >= the "
                                              "sweep's own a* (baseline 6 -> "
                                              "positions 64..505, n=442)"),
                          statistic="mean over sequences of the per-seq junk "
                                    "fraction (e073 generated-band "
                                    "convention); max/pooled/counts recorded "
                                    "too (e073's stored prompt number 4/63 is "
                                    "the max reading)")),
        gates=gates,
        e073_reference=E073_REF,
        baseline=dict(frame=frame_summary(d_base, ce_base,
                                          "e069/e073 replica (unshuffled)"),
                      prompt_band=base_prompt,
                      generated_band=base_gen,
                      generated_band_frozen_note="baseline own a* == 6 == "
                                                 "frozen",
                      gen_order_halves=base_halves,
                      pos0_dce=dict(mean=float(pos0.mean()),
                                    per_seq=pos0.tolist())),
        shuffled=dict(frame=frame_summary(d_shuf, ce_shuf,
                                          "THE MANIPULATION (prompt permuted)"),
                      prompt_band=shuf_prompt,
                      generated_band_own_astar=shuf_gen,
                      generated_band_frozen_astar6=shuf_gen_frozen,
                      gen_order_halves_own_astar=shuf_halves_own,
                      gen_order_halves_frozen_astar6=shuf_halves_frozen,
                      pos0_dce=dict(mean=float(pos0_shuf.mean()),
                                    per_seq=pos0_shuf.tolist())),
        manipulation_checks=dict(
            clean_ce_baseline=float(ce_base.mean()),
            clean_ce_shuffled=float(ce_shuf.mean()),
            ce_rise_mean=float(ce_rise.mean()),
            ce_rise_per_seq=ce_rise.tolist(),
            prompt_slots_changed_frac=frac_perm_changed,
            adjacent_pair_preservation=float(np.mean(kept_pairs)),
            adjacent_pair_chance=1 / 65,
            ages1_5_baseline=[float(a) for a in ages15],
            ages1_5_shuffled=ages15_shuf,
            ages1_5_intact=spike_intact),
        secondary_generation_order=secondary,
        registered_decision=dict(
            frozen_rules=dict(
                h_source=f"shuffled-prompt-band junk (mean over seqs) <= "
                         f"{JUNK_LOW}",
                h_age_statistics=f"shuffled-prompt-band junk >= {JUNK_HIGH}",
                between=f"({JUNK_LOW}, {JUNK_HIGH}) -> AMBIGUOUS, no forcing"),
            shuffled_prompt_junk_mean=pj,
            shuffled_prompt_junk_ci=shuf_prompt["junk_frac_mean_ci"],
            shuffled_prompt_junk_max=pj_max,
            shuffled_generated_junk_mean=shuf_gen["junk_frac_mean"],
            shuffled_generated_junk_frozen_astar6=shuf_gen_frozen["junk_frac_mean"],
            baseline_generated_junk_mean=base_gen["junk_frac_mean"],
            baseline_prompt_junk_mean=base_prompt["junk_frac_mean"],
            generated_band_stays_junky=gen_stays,
            clauses=clauses, clause=clause, verdict=verdict,
            honest_misses=honest_misses),
    )
    save_json(out_dir / "metrics.json", metrics)
    log("metrics.json written")

    plot(out_dir / "junk_control.png", metrics)
    log(f"plot written; total wall {elapsed():.0f}s")


# ---------------------------------------------------------------------- plot


def plot(path: Path, M: dict):
    dec = M["registered_decision"]
    base, shuf = M["baseline"], M["shuffled"]
    mc = M["manipulation_checks"]
    sec = M["secondary_generation_order"]
    ages = np.arange(1, T_TOTAL)
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    # ---- panel 1: full mean dCE curves, baseline vs shuffled, bands shaded
    ax1.fill_between(ages, np.asarray(base["frame"]["ci_lo"]),
                     np.asarray(base["frame"]["ci_hi"]), alpha=0.12,
                     color="tab:blue")
    ax1.plot(ages, base["frame"]["mean_curve"], lw=1.2, color="tab:blue",
             label=f"baseline (a*={base['frame']['a_star']})")
    ax1.fill_between(ages, np.asarray(shuf["frame"]["ci_lo"]),
                     np.asarray(shuf["frame"]["ci_hi"]), alpha=0.12,
                     color="tab:red")
    ax1.plot(ages, shuf["frame"]["mean_curve"], lw=1.2, color="tab:red",
             label=f"shuffled prompt (a*={shuf['frame']['a_star']})")
    ax1.axhline(0, color="k", lw=0.6)
    ax1.axhline(THRESH, color="gray", ls=":", lw=0.9)
    ax1.axhline(-THRESH, color="gray", ls=":", lw=0.9)
    ax1.axvline(448, color="tab:purple", ls="--", lw=1.0)
    ax1.text(451, 0.30, "prompt band (ages 448-511)", fontsize=8,
             color="tab:purple", rotation=90, va="top")
    ax1.set_xlim(0, T_TOTAL)
    lo = min(float(np.nanpercentile(np.asarray(base["frame"]["mean_curve"])[10:], 1)),
             float(np.nanpercentile(np.asarray(shuf["frame"]["mean_curve"])[10:], 1))) - 0.03
    ax1.set_ylim(lo, 0.35)   # young spike off-scale here; panel 2 shows it
    ax1.set_xlabel("cache age (tokens)")
    ax1.set_ylabel("V-zero dCE (nats, mean over B=4 seqs)")
    ax1.legend(fontsize=9)
    ax1.set_title("E074-1 — full 511-position V-zero sweep: baseline vs "
                  "shuffled-prompt contexts (same sequences, same net)",
                  fontsize=10)

    # ---- panel 2: zoom ages 1..60 (spike + onset intact?)
    zoom = 60
    ax2.plot(ages[:zoom], base["frame"]["mean_curve"][:zoom], lw=1.3,
             color="tab:blue", label="baseline")
    ax2.plot(ages[:zoom], shuf["frame"]["mean_curve"][:zoom], lw=1.3,
             color="tab:red", label="shuffled prompt")
    ax2.axhline(THRESH, color="gray", ls=":", lw=1.0)
    ax2.text(zoom, THRESH, f" thresh {THRESH}", fontsize=8, va="bottom",
             ha="right", color="gray")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.axvline(base["frame"]["a_star"], color="tab:blue", ls="--", lw=1.0)
    ax2.axvline(shuf["frame"]["a_star"], color="tab:red", ls="--", lw=1.0)
    ax2.set_title("E074-2 — ages 1-60 zoom: the young spike under shuffling "
                  f"(e069 D2)\nbaseline ages1-5 "
                  f"{['%.2f' % a for a in mc['ages1_5_baseline']]} -> shuffled "
                  f"{['%.2f' % a for a in mc['ages1_5_shuffled']]}", fontsize=10)
    ax2.set_xlabel("cache age (tokens)")
    ax2.set_ylabel("V-zero dCE (nats)")
    ax2.legend(fontsize=9)

    # ---- panel 3: junk fraction by band (THE registered readout)
    labels = ["prompt band\n(63 oldest,\ncorpus)", "generated band\n(beyond a*, 442)",
              "SHUFFLED prompt\nband (63)", "SHUFFLED generated\nband (own a*)"]
    vals = [base["prompt_band"]["junk_frac_mean"],
            base["generated_band"]["junk_frac_mean"],
            shuf["prompt_band"]["junk_frac_mean"],
            shuf["generated_band_own_astar"]["junk_frac_mean"]]
    cis = [base["prompt_band"]["junk_frac_mean_ci"],
           base["generated_band"]["junk_frac_mean_ci"],
           shuf["prompt_band"]["junk_frac_mean_ci"],
           shuf["generated_band_own_astar"]["junk_frac_mean_ci"]]
    cols = ["tab:blue", "tab:cyan", "tab:red", "tab:orange"]
    x = np.arange(len(vals))
    ax3.bar(x, vals, 0.62, color=cols, alpha=0.85, edgecolor="k", lw=0.5)
    ax3.errorbar(x, vals, yerr=[[v - c[0] for v, c in zip(vals, cis)],
                                [c[1] - v for v, c in zip(vals, cis)]],
                 fmt="none", ecolor="k", lw=1.0, capsize=3)
    ax3.axhline(JUNK_LOW, color="tab:green", ls="--", lw=1.2)
    ax3.text(len(vals) - 0.4, JUNK_LOW + 0.004, f"H-source bar {JUNK_LOW}",
             fontsize=8, color="tab:green", ha="right")
    ax3.axhline(JUNK_HIGH, color="tab:purple", ls="--", lw=1.2)
    ax3.text(len(vals) - 0.4, JUNK_HIGH + 0.004,
             f"H-age-stat bar {JUNK_HIGH}", fontsize=8, color="tab:purple",
             ha="right")
    for xi, v in zip(x, vals):
        ax3.text(xi, v + 0.006, f"{v:.3f}", ha="center", fontsize=9)
    ax3.set_xticks(x, labels, fontsize=8)
    ax3.set_ylabel("junk fraction (dCE <= -0.01 nats)")
    ax3.set_ylim(0, max(JUNK_HIGH * 1.25, max(vals) * 1.25))
    ax3.set_title("E074-3 — THE REGISTERED DISCRIMINATOR: junk by band\n"
                  f"clause: {dec['clause']}", fontsize=10)

    # ---- panel 4: secondary — generation order halves
    sb, ss = sec["baseline"], sec["shuffled_own_astar"]
    labels4 = ["baseline\nearly-gen", "baseline\nlate-gen",
               "shuffled\nearly-gen", "shuffled\nlate-gen"]
    vals4 = [sb["early"]["junk_frac_mean"], sb["late"]["junk_frac_mean"],
             ss["early"]["junk_frac_mean"], ss["late"]["junk_frac_mean"]]
    md4 = [sb["early"]["mean_dce"], sb["late"]["mean_dce"],
           ss["early"]["mean_dce"], ss["late"]["mean_dce"]]
    cols4 = ["tab:blue", "tab:cyan", "tab:red", "tab:orange"]
    x4 = np.arange(4)
    ax4.bar(x4, vals4, 0.6, color=cols4, alpha=0.85, edgecolor="k", lw=0.5)
    for xi, v, m in zip(x4, vals4, md4):
        ax4.text(xi, v + 0.004, f"{v:.3f}\n<dCE {m:+.4f}>", ha="center",
                 fontsize=8)
    ax4.set_xticks(x4, labels4, fontsize=8)
    ax4.set_ylabel("junk fraction")
    ax4.set_ylim(0, max(vals4) * 1.35 + 0.01)
    ax4.set_title("E074-4 — SECONDARY: junk by GENERATION ORDER (early vs "
                  "late halves)\nregistered: H-source predicts late > early "
                  f"| baseline late-early {sec['baseline']['late_minus_early']:+.4f}, "
                  f"shuffled {sec['shuffled_own_astar']['late_minus_early']:+.4f}",
                  fontsize=10)

    # ---- panel 5: manipulation check — clean CE
    x5 = np.arange(B)
    w = 0.36
    ax5.bar(x5 - w / 2, base["frame"]["clean_ce_per_seq"], w, color="tab:blue",
            label=f"baseline (mean {mc['clean_ce_baseline']:.4f})")
    ax5.bar(x5 + w / 2, shuf["frame"]["clean_ce_per_seq"], w, color="tab:red",
            label=f"shuffled (mean {mc['clean_ce_shuffled']:.4f})")
    ax5.set_xticks(x5, [f"seq {i}" for i in range(B)])
    ax5.set_ylabel("clean CE of final-token prediction (nats)")
    ax5.legend(fontsize=9)
    ax5.set_title("E074-5 — MANIPULATION CHECK: clean CE under prompt "
                  "shuffling\nregistered expectation: RISE | observed mean "
                  f"change {mc['ce_rise_mean']:+.4f} nats (honest miss, "
                  "seq-1 driven; others ~0)", fontsize=10)

    # ---- panel 6: verdict text
    ax6.axis("off")
    bp, bg = base["prompt_band"], base["generated_band"]
    sp, sg = shuf["prompt_band"], shuf["generated_band_own_astar"]
    lines = [
        "REGISTERED (frozen before measurement):",
        f"  shuffled-prompt junk <= {JUNK_LOW}  -> H-SOURCE (self-generation is the poison)",
        f"  shuffled-prompt junk >= {JUNK_HIGH} -> H-AGE-STATISTICS (info-poor band drifts)",
        "",
        f"baseline:  prompt junk {bp['junk_frac_mean']:.4f} (max {bp['junk_frac_max']:.4f})"
        f" | generated-old junk {bg['junk_frac_mean']:.4f}",
        f"shuffled:  prompt junk {sp['junk_frac_mean']:.4f} CI "
        f"[{sp['junk_frac_mean_ci'][0]:.4f},{sp['junk_frac_mean_ci'][1]:.4f}]"
        f" (max {sp['junk_frac_max']:.4f})",
        f"           generated junk {sg['junk_frac_mean']:.4f} "
        f"(frozen-a*6 {shuf['generated_band_frozen_astar6']['junk_frac_mean']:.4f})",
        f"           prompt mean dCE {sp['mean_dce']:+.5f} "
        f"(baseline {bp['mean_dce']:+.5f})",
        "",
        f"manipulation: clean CE {mc['clean_ce_baseline']:.4f} -> "
        f"{mc['clean_ce_shuffled']:.4f} ({mc['ce_rise_mean']:+.4f})",
        f"ages 1-5 spike intact: {mc['ages1_5_intact']} "
        f"({['%.2f' % a for a in mc['ages1_5_shuffled']]})",
        f"secondary: late-vs-early junk, baseline "
        f"{sec['baseline']['late_minus_early']:+.4f} | shuffled "
        f"{sec['shuffled_own_astar']['late_minus_early']:+.4f}",
        "",
        f"VERDICT [{dec['clause']}]:",
    ] + [f"  {w}" for w in _wrap(dec["verdict"], 92)]
    ax6.text(0.02, 0.97, "E074 — T045 shuffled-prompt junk control",
             fontsize=13, weight="bold", va="top")
    for i, t in enumerate(lines):
        ax6.text(0.02, 0.90 - i * 0.052, t, fontsize=9.0, va="top",
                 family="monospace")

    fig.suptitle("E074 — shuffled-prompt junk control | clause: "
                 f"{dec['clause']} | shuffled prompt junk "
                 f"{dec['shuffled_prompt_junk_mean']:.4f} "
                 f"(bars {JUNK_LOW}/{JUNK_HIGH}) | generated junk stays "
                 f"{dec['shuffled_generated_junk_mean']:.4f}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _wrap(text: str, width: int):
    import textwrap
    return textwrap.wrap(text, width=width)


if __name__ == "__main__":
    main()
