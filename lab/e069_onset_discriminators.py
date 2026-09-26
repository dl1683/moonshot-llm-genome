"""E069 — T039's TWO registered discriminators on the EXISTING e053c net.

Trigger (THINKING.md T039, 2026-09-26T10:05Z): e053c resolved the onset
ABSOLUTE (a*(512)=6 CI [4,8] vs a*(256)=7) but left two live hypotheses
for WHY the horizon is ~6 tokens:

- H1 CIRCUIT HORIZON: the decision circuit reads a fixed number of recent
  positions (induction-head/local integration window) — window-independent
  by construction.
- H2 STATISTICAL HORIZON: char-level local redundancy (n-gram
  predictiveness) has a fixed effective length ~5-6; the spike tracks the
  corpus statistics, not the circuit.

T039 registered two eval-only discriminators; this run executes BOTH on
the frozen e053c ctx-512 checkpoint (runs/checkpoints/e053c_ctx512.pt,
873,472 params, 4L/4H/128d/wpe-512, val CE 1.5227). CPU-ONLY.

D1 EVAL-WINDOW TRUNCATION (registered): re-run the fine-onset fit (same
seeds 202/7, same prompt-draw modulus, same onset rule, B=4) with eval
context truncated to 256 — feed only the last 256 positions of each
512-position sequence (wpe re-indexed 0..255, primary; an UNREGISTERED
robustness variant keeps absolute wpe 256..510). REGISTERED: a*(eval-256)
in [4,8] confirms within-net window-independence (H1+H2 both pass this);
outside -> the horizon is eval-window-sensitive (surprise — write it
down, don't force it).

D2 SHUFFLED-CHAR SPIKE (registered): on the FULL 512 window, run the
V-zero sweep on prompts whose OLDER half (positions before the last ~8,
i.e. context positions 0..502 = ages >= 9) has characters permuted
(destroys n-gram statistics; keeps recency structure + the fresh tail of
ages 1..8 and the target intact). REGISTERED: spike magnitude at ages
1-2 retains >= 30% of normal => H1 circuit-horizon; collapses to <= 10%
=> H2 statistical-horizon. Ages 1-5 magnitudes normal-vs-shuffled
reported.

Battery/instrument sanity FIRST: rebuild the corpus exactly as e053c did
(CharCorpus(data/input.txt), seed 1337), G1 val CE must be ~1.5227
(+/-0.02), G2 params must be 873,472, G0b batched incremental-KV final
logits vs full recompute (prob dev < 1e-4), G3 the CPU 512-window fit
must land a* in the registered [4,8] neighborhood (e053c: 6, CI [4,8]).
Machinery (manual forward, batched generation with the shared seed-7 CPU
generator in row order, fine sweep, onset rule, bootstrap) is VERBATIM
e053b/e053c, device-parameterization removed (CPU-only).

Run:     python lab/e069_onset_discriminators.py
Outputs: runs/e069/metrics.json + runs/e069/discriminators.png
Envelope: NO training, NO new automations; CPU-only, batched, minutes.
No NOTES/THINKING/QUEUE/STATE edits; no commit (the dispatcher owns those).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # task spec: CPU-only (pre-torch)
import torch

# this torch build reports is_available()==True even with an empty
# CUDA_VISIBLE_DEVICES; force CPU so common.DEVICE=="cpu" (e053b pattern)
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
P = T_TOTAL - 1                           # 511 sweep positions / max age
TEMP, TOPK = 0.8, 40
SEED_PROMPT, SEED_SAMPLE = 202, 7         # e053b/e053c exact seeds
SEED_SHUF = 209                           # e069's own (char permutations)
THRESH = 0.01                             # dead-weight threshold (nats)
A_STAR_K = 5                              # sustained window (e053b verbatim)
N_PROMPTS = 8                             # same draw size as e053b/e053c
B = 4                                     # e053b's uniform B (task spec)
BOOT_N = 1000
THREADS = 12
torch.set_num_threads(THREADS)

EVAL256 = 256                             # D1 truncation
P256 = EVAL256 - 1                        # 255 positions in the eval-256 window
TAIL_KEEP = 8                             # D2 fresh tail (ages 1..8 intact)
CHUNK_512 = 64                            # measured ~60s / full 512 sweep (CPU)
CHUNK_256 = 128                           # measured ~13s / full 256 sweep (CPU)

CKPT = REPO / "runs" / "checkpoints" / "e053c_ctx512.pt"
E053C_VAL_CE = 1.5226792494455974         # the run being interrogated
E053C_ASTAR = dict(a_star=6, ci=[4.0, 8.0])
D1_WINDOW = [4.0, 8.0]                    # T039's registered eval-256 window
RET_H1, RET_H2 = 0.30, 0.10               # T039's registered retention cuts
UNIFORM_CE = math.log(65)
T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# ------------------------------------------------- machinery (VERBATIM e053b/c)

@torch.no_grad()
def _manual_chunk(net: TinyGPT, idxs, vzero, kdrop, layers, pos_offset=0):
    """Exact manual forward, returns LAST-position logits (N, vocab).
    pos_offset lets the same tokens be fed at shifted wpe indices (e069)."""
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
    (position-T_TOTAL-1) logits (B, V). [CPU-only e069]"""
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


# ---------------------------------------------------------------------- main


def main():
    assert torch.cuda.device_count() == 0, "CPU-only violated: CUDA visible"
    assert common.DEVICE == "cpu", f"common.DEVICE={common.DEVICE} != cpu"
    out_dir = run_dir("e069")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gates = dict(cpu_only=True, threads=THREADS, smoke=False)

    # ---- battery: rebuild corpus + net EXACTLY as e053c did
    corp = CharCorpus(REPO / "data" / "input.txt")       # seed 1337 (e053c)
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

    # ---- sequences: e053b/e053c protocol verbatim (seeds 202/7, B=4)
    gen_p = torch.Generator().manual_seed(SEED_PROMPT)
    ix = torch.randint(len(corp.val) - PROMPT_TOK - 1, (N_PROMPTS,),
                       generator=gen_p)
    prompts = [corp.val[i:i + PROMPT_TOK] for i in ix]
    log(f"{N_PROMPTS} prompts (seed {SEED_PROMPT}); using first B={B}; "
        f"prompt0 prefix: {corp.decode(prompts[0])[:32]!r}")
    gen = torch.Generator().manual_seed(SEED_SAMPLE)     # per-cell seed-7 re-init
    idx, final_logits = generate_batch(net, prompts[:B], gen)
    log(f"generated {B} seqs 64->{T_TOTAL} (fixed anchor, CPU)")

    # G0b: batched incremental-KV final logits vs full recompute (prob dev)
    with torch.no_grad():
        full = manual_logits(net, idx[:, :-1], None, None, None, chunk=B)
    dev_b = float((torch.softmax(final_logits.float(), -1)
                   - torch.softmax(full.float(), -1)).abs().max())
    gates["G0b_kvs_vs_full"] = dict(max_prob_dev=dev_b, ok=bool(dev_b < 1e-4))
    log(f"G0b incremental-KV vs full recompute: max prob dev {dev_b:.2e} "
        f"-> {'PASS' if dev_b < 1e-4 else 'FAIL'}")

    # ---- sweep A: FULL-512 fit (the normal baseline + instrument check)
    log("sweep A: eval-512 fine onset fit (normal baseline)")
    d512, ce512 = sweep_block(net, idx[:, :-1], idx[:, -1], final_logits,
                              CHUNK_512)
    ov_inst = not (d512["a_star_ci"][1] < D1_WINDOW[0]
                   or d512["a_star_ci"][0] > D1_WINDOW[1])
    pt_inst = d512["a_star"] is not None and D1_WINDOW[0] <= d512["a_star"] <= D1_WINDOW[1]
    gates["G3_instrument"] = dict(
        a_star_512_cpu=d512["a_star"], ci=d512["a_star_ci"],
        e053c_reported=E053C_ASTAR, registered_window=D1_WINDOW,
        point_in_window=bool(pt_inst), ci_overlaps_window=bool(ov_inst),
        ok=bool(ov_inst))
    log(f"G3 instrument: a*(512, CPU) = {d512['a_star']} CI "
        f"[{d512['a_star_ci'][0]:.0f},{d512['a_star_ci'][1]:.0f}] vs e053c 6 "
        f"[4,8] -> {'PASS' if ov_inst else 'FAIL'}")

    # ---- sweep B: DISCRIMINATOR 1 — eval window truncated to 256
    win = idx[:, -EVAL256:]                    # last 256 positions (256..511)
    ctx256, tgt256 = win[:, :-1], win[:, -1]
    assert torch.equal(tgt256, idx[:, -1])     # same final target token
    log(f"D1 sweep B: eval-256 window (feed last {EVAL256} positions, "
        f"wpe re-indexed 0..{P256 - 1})")
    clean256 = manual_logits(net, ctx256, None, None, None, chunk=CHUNK_256)
    d256, ce256 = sweep_block(net, ctx256, tgt256, clean256, CHUNK_256,
                             pos_offset=0)
    # UNREGISTERED robustness: same tokens, ABSOLUTE wpe indices 256..510
    log("D1 sweep B': eval-256 robustness (absolute wpe 256..510)")
    clean256a = manual_logits(net, ctx256, None, None, None, chunk=CHUNK_256,
                              pos_offset=T_TOTAL - EVAL256)
    d256a, ce256a = sweep_block(net, ctx256, tgt256, clean256a, CHUNK_256,
                                pos_offset=T_TOTAL - EVAL256)

    in_win = d256["a_star"] is not None and D1_WINDOW[0] <= d256["a_star"] <= D1_WINDOW[1]
    ci_ov = not (d256["a_star_ci"][1] < D1_WINDOW[0]
                 or d256["a_star_ci"][0] > D1_WINDOW[1])
    if in_win:
        v1 = (f"WINDOW-INDEPENDENT: a*(eval-256)={d256['a_star']} is inside the "
              f"registered [4,8] — H1+H2 both pass; the ~6-token horizon does "
              f"not depend on the eval window within this net")
    else:
        v1 = (f"EVAL-WINDOW-SENSITIVE (surprise): a*(eval-256)={d256['a_star']} "
              f"is OUTSIDE the registered [4,8] — recorded as-is, not forced")
    log(f"D1 DECISION: {v1}")

    # ---- sweep C: DISCRIMINATOR 2 — shuffled-char contexts, FULL 512 window
    n_old = P - TAIL_KEEP                    # context positions 0..502 shuffled
    rng = np.random.default_rng(SEED_SHUF)
    shuf = idx[:, :-1].clone()
    for b in range(B):
        perm = torch.from_numpy(rng.permutation(n_old).astype(np.int64))
        old = idx[b, :-1][:n_old].clone()
        shuf[b, :n_old] = old[perm]          # permute chars among older slots
    n_changed = int((shuf != idx[:, :-1]).sum())
    log(f"D2 sweep C: shuffled-char contexts on FULL 512 window (positions "
        f"0..{n_old - 1} shuffled = ages >= {TAIL_KEEP + 1}; ages 1..{TAIL_KEEP} "
        f"+ target intact; {n_changed}/{B * P} slots moved; seed {SEED_SHUF})")
    clean_sh = manual_logits(net, shuf, None, None, None, chunk=CHUNK_512)
    dsh, ce_sh = sweep_block(net, shuf, idx[:, -1], clean_sh, CHUNK_512)
    log(f"manipulation check: clean CE normal {ce512.mean():.3f} -> shuffled "
        f"{ce_sh.mean():.3f} (uniform {UNIFORM_CE:.3f})")

    ages5 = list(range(1, 6))
    m_norm = d512["mean_age_curve"][[a - 1 for a in ages5]]
    m_shuf = dsh["mean_age_curve"][[a - 1 for a in ages5]]
    ret_age = (m_shuf / m_norm).tolist()
    r12 = float(m_shuf[:2].sum() / m_norm[:2].sum())
    # paired bootstrap of the ages-1-2 retention over sequences
    S = B
    rb = np.random.default_rng(0)
    per_seq_r12 = []
    for s in range(S):
        nn = d512["dce_age"][s, :2].sum()
        per_seq_r12.append(float(dsh["dce_age"][s, :2].sum() / nn) if nn > 0
                           else float("nan"))
    vals = []
    for _ in range(BOOT_N):
        sel = rb.integers(0, S, S)
        nn = d512["dce_age"][sel, :2].mean(0).sum()
        vals.append(dsh["dce_age"][sel, :2].mean(0).sum() / nn if nn > 0
                    else np.nan)
    r12_lo, r12_hi = (float(np.nanpercentile(vals, 2.5)),
                      float(np.nanpercentile(vals, 97.5)))
    if r12 >= RET_H1:
        v2 = (f"H1 CIRCUIT-HORIZON: ages-1-2 spike retains {r12 * 100:.0f}% of "
              f"normal (>= 30%) under shuffled-char contexts")
    elif r12 <= RET_H2:
        v2 = (f"H2 STATISTICAL-HORIZON: ages-1-2 spike collapses to {r12 * 100:.0f}% "
              f"of normal (<= 10%) under shuffled-char contexts")
    else:
        v2 = (f"INTERMEDIATE: ages-1-2 retention {r12 * 100:.0f}% (between 10% "
              f"and 30%) — neither registered branch fires cleanly")
    log(f"D2 DECISION: {v2} (per-age retention 1..5: "
        + ", ".join(f"{r:.2f}" for r in ret_age) + ")")

    # ---------------------------------------------------------------- metrics
    def curve_block(d, ce, extra=None):
        blk = dict(
            a_star=d["a_star"], a_star_naive=d["a_star_naive"],
            a_star_robust=d["a_star_robust"], a_star_ci=d["a_star_ci"],
            a_star_frac=d["a_star_frac"], live_frac=d["live_frac"],
            live_frac_ci=d["live_frac_ci"], per_seq_a_star=d["per_seq_a_star"],
            age=list(range(1, len(d["mean_age_curve"]) + 1)),
            vzero_dce_mean=d["mean_age_curve"].tolist(),
            vzero_ci=[d["ci"][0].tolist(), d["ci"][1].tolist()],
            clean_ce_per_seq=ce.tolist(), clean_ce_mean=float(np.mean(ce)),
        )
        if extra:
            blk.update(extra)
        return blk

    metrics = dict(
        experiment="e069_onset_discriminators",
        purpose="T039's two registered eval-only discriminators on the frozen "
                "e053c ctx-512 net: (D1) eval-window truncation to 256 — "
                "a*(eval-256) in [4,8] => within-net window-independence; "
                "(D2) shuffled-char spike — ages-1-2 retention >=30% => H1 "
                "circuit-horizon, <=10% => H2 statistical-horizon.",
        started=started, wall_s=elapsed(), threads=THREADS, cpu_only=True,
        net=dict(ckpt=str(CKPT), arch=dict(n_layer=4, n_head=4, n_embd=128,
                                           block_size=T_TOTAL, vocab=65),
                 params=n_params, steps_in_ckpt=steps_in_ckpt,
                 val_ce=val_ce, val_ce_e053c=E053C_VAL_CE),
        seeds=dict(corpus=1337, prompts=SEED_PROMPT, sampling=SEED_SAMPLE,
                   shuffle=SEED_SHUF),
        protocol=dict(B=B, prompt_tokens=PROMPT_TOK, t_total=T_TOTAL, temp=TEMP,
                      topk=TOPK, fixed_anchor=True, threshold=THRESH,
                      a_star_window=A_STAR_K, boot_n=BOOT_N,
                      chunks=dict(w512=CHUNK_512, w256=CHUNK_256),
                      sampling="CPU tensors + shared seed-7 CPU generator, "
                               "per-row in row order (e053b-identical)"),
        gates=gates,
        reference_e053c=E053C_ASTAR,
        discriminator1_eval_window=dict(
            registered=("a*(eval-256) in [4,8] confirms within-net "
                        "window-independence (H1+H2 both pass); outside -> "
                        "eval-window-sensitive (surprise — write it, don't "
                        "force it) [T039]"),
            window=D1_WINDOW,
            eval512=curve_block(d512, ce512,
                                dict(role="normal baseline + instrument check",
                                     wpe="0..510 (native)")),
            eval256=curve_block(d256, ce256,
                                dict(role="PRIMARY registered readout",
                                     wpe="re-indexed 0..254 (last 256 "
                                         "positions fed as a fresh window)",
                                     note="same generated sequences, same "
                                          "seeds, same onset rule; clean "
                                          "logits recomputed within the "
                                          "truncated window")),
            eval256_abspos=curve_block(d256a, ce256a,
                                       dict(role="UNREGISTERED robustness",
                                            wpe="absolute 256..510 (cache-"
                                                "truncation semantics)")),
            point_in_window=bool(in_win), ci_overlaps_window=bool(ci_ov),
            verdict=v1,
        ),
        discriminator2_shuffled_char=dict(
            registered=("ages-1-2 spike retention >= 30% of normal => H1 "
                        "circuit-horizon; collapses to <= 10% => H2 "
                        "statistical-horizon [T039]"),
            retention_thresholds=dict(h1_ge=RET_H1, h2_le=RET_H2),
            manipulation=dict(
                shuffled_positions=f"0..{n_old - 1} (ages >= {TAIL_KEEP + 1})",
                fresh_tail=f"context positions {n_old}..{P - 1} (ages "
                           f"1..{TAIL_KEEP}) + target intact",
                tail_keep=TAIL_KEEP, seed=SEED_SHUF,
                slots_moved=n_changed, slots_total=B * P,
                clean_ce_normal=float(ce512.mean()),
                clean_ce_shuffled=float(ce_sh.mean()),
                uniform_ce=UNIFORM_CE),
            ages=ages5,
            normal_nats=m_norm.tolist(),
            shuffled_nats=m_shuf.tolist(),
            retention_per_age=ret_age,
            retention_ages12=r12, retention_ages12_ci=[r12_lo, r12_hi],
            per_seq_retention_ages12=per_seq_r12,
            eval512_full=curve_block(d512, ce512, dict(role="normal baseline")),
            shuffled_full=curve_block(dsh, ce_sh,
                                      dict(role="shuffled-char sweep, FULL "
                                                "512 window, target intact")),
            verdict=v2,
        ),
    )
    save_json(out_dir / "metrics.json", metrics)
    log("metrics.json written")
    plot(out_dir / "discriminators.png", metrics)
    log(f"plot written; total wall {elapsed():.0f}s")


# ---------------------------------------------------------------------- plot


def plot(path: Path, M: dict):
    d1 = M["discriminator1_eval_window"]
    d2 = M["discriminator2_shuffled_char"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.8))

    # ---- panel 1: onset fit at eval-256 vs 512
    for key, col, lbl in [("eval512", "tab:red", "eval-512 (native window)"),
                          ("eval256", "tab:blue",
                           f"eval-256 (wpe 0..{P256 - 1})")]:
        c = d1[key]
        age = np.asarray(c["age"])
        m = np.asarray(c["vzero_dce_mean"])
        lo, hi = np.asarray(c["vzero_ci"][0]), np.asarray(c["vzero_ci"][1])
        ax1.fill_between(age, lo, hi, alpha=0.2, color=col)
        ax1.plot(age, m, lw=1.4, color=col,
                 label=f"{lbl}: a*={c['a_star']} CI "
                       f"[{c['a_star_ci'][0]:.0f},{c['a_star_ci'][1]:.0f}]")
        if c["a_star"]:
            ax1.axvline(c["a_star"], color=col, ls="--", lw=1.3)
    ax1.axhline(0, color="k", lw=0.6)
    ax1.axhline(THRESH, color="gray", ls=":", lw=1.0)
    ax1.axvspan(D1_WINDOW[0], D1_WINDOW[1], color="tab:green", alpha=0.08)
    ax1.text((D1_WINDOW[0] + D1_WINDOW[1]) / 2, 1.0, "registered [4,8]",
             ha="center", fontsize=8, color="tab:green")
    ax1.set_xlim(0, T_TOTAL)
    m512 = np.asarray(d1["eval512"]["vzero_dce_mean"])
    ax1.set_ylim(min(-0.1, np.nanpercentile(m512, 2) - 0.05),
                 np.nanpercentile(np.asarray(d1["eval512"]["vzero_ci"][1]), 98))
    ax1.set_xlabel("cache age (tokens)")
    ax1.set_ylabel("V-zero dCE (nats)")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.set_title("D1 — onset fit: eval context truncated to 256 vs 512 "
                  "(same net, same seqs, same seeds)\n"
                  f"{d1['verdict']}", fontsize=10)
    axi = ax1.inset_axes([0.30, 0.42, 0.28, 0.30])
    for key, col in [("eval512", "tab:red"), ("eval256", "tab:blue")]:
        c = d1[key]
        axi.plot(np.asarray(c["age"]), np.asarray(c["vzero_dce_mean"]),
                 lw=1.2, color=col)
        if c["a_star"]:
            axi.axvline(c["a_star"], color=col, ls="--", lw=1.0)
    axi.axhline(THRESH, color="gray", ls=":", lw=0.8)
    axi.set_xlim(0, 40)
    axi.set_title("zoom ages 0..40", fontsize=8)

    # ---- panel 2: spike magnitudes normal vs shuffled (ages 1..5)
    ages = np.asarray(d2["ages"])
    norm = np.asarray(d2["normal_nats"])
    shuf = np.asarray(d2["shuffled_nats"])
    x = np.arange(len(ages))
    ax2.bar(x - 0.19, norm, 0.38, color="tab:red", alpha=0.85,
            label="normal (eval-512)")
    ax2.bar(x + 0.19, shuf, 0.38, color="tab:purple", alpha=0.85,
            label="shuffled-char (older half)")
    for i in range(len(ages)):
        ax2.text(x[i] - 0.19, norm[i], f"{norm[i]:.3f}", ha="center",
                 va="bottom", fontsize=8)
        ax2.text(x[i] + 0.19, shuf[i], f"{shuf[i]:.3f}", ha="center",
                 va="bottom", fontsize=8)
        ax2.text(x[i], max(norm[i], shuf[i]) + 0.09 * max(norm.max(), 0.01),
                 f"{d2['retention_per_age'][i] * 100:.0f}%", ha="center",
                 fontsize=8, color="dimgray")
    ax2.set_xticks(x, [f"age {a}" for a in ages])
    ax2.set_ylabel("V-zero dCE spike magnitude (nats, mean over B=4 seqs)")
    ax2.legend(fontsize=9)
    ax2.set_title("D2 — young spike: normal vs shuffled-char contexts "
                  "(FULL 512 window, ages 1..8 + target intact)\n"
                  f"ages-1-2 retention {d2['retention_ages12'] * 100:.0f}% "
                  f"(CI {d2['retention_ages12_ci'][0] * 100:.0f}-"
                  f"{d2['retention_ages12_ci'][1] * 100:.0f}%) — "
                  f"{d2['verdict']}", fontsize=10)
    mani = d2["manipulation"]
    ax2.text(0.02, 0.97,
             f"manipulation: clean CE {mani['clean_ce_normal']:.3f} (normal) "
             f"-> {mani['clean_ce_shuffled']:.3f} (shuffled); "
             f"uniform {mani['uniform_ce']:.2f}",
             transform=ax2.transAxes, fontsize=8, va="top", color="dimgray")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
