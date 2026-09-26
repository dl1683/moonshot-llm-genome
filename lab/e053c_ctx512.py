"""E053c — Phase-2 CTX-512 ONSET DECIDER (the registered discriminator).

Trigger (THINKING.md T031 + e053b verdicts.abs_vs_proportional): the
cache-timeline paper's onset a* was measured only at ctx-256 (wpe=256 is
a hard limit in every existing ckpt), so ABSOLUTE (a* is a fixed token
count) vs PROPORTIONAL (a* is a fixed fraction of the window) could not
be separated. e053b REGISTERED the decider:

    "Phase-2 ctx-512 cell (same arch family retrained with block_size=512):
     a*(ctx512) ~ a*(ctx256) in tokens -> ABSOLUTE;
     a*(ctx512) ~ frac x 512 -> PROPORTIONAL."

This run executes it:
- TRAIN exactly ONE net: the 0.84M-class cfg (4L/4H/128d, e005s small
  family) with block_size=512 -> 873,472 params (<=1M envelope). Same
  corpus (seed 1337), set_seed(42), batch 32 at ctx-512 (== e005s's
  batch 64 at ctx-256 in tokens/step: matched token exposure), steps=4000
  schedule target, 180s hard cap, ckpt-resumable. gpu_ok() gate +
  thermal cooldown BEFORE launch; temp recorded after.
- MEASURE with e053b's fine-onset protocol VERBATIM (same expressions,
  same seeds): 8 candidate prompts drawn with seed 202 from val (same
  modulus as e053b -> identical prompt positions), B=4, free-run 64->512
  fixed-anchor, temp 0.8 top-k 40, per-row draws from the shared seed-7
  CPU generator, then the full 511-position V-zero lesion sweep at the
  final step; a* = youngest cache age with mean dCE < 0.01 nats (naive +
  5-pt-sustained robust), bootstrap CI over sequences (1000x).
  Heavy forward passes run on the idle GPU (task-authorized, brief,
  envelope-checked); SAMPLING stays on CPU tensors with the CPU
  generator so the stream math is e053b-identical. Gates prove device
  equivalence: G0b (batched incremental-KV final logits vs full
  recompute, prob dev < 1e-4) and G0-dev (CPU vs GPU dCE on a sweep
  subset incl. the young spike, dev < 1e-4). If either fails -> redo the
  whole measurement on CPU (honest fallback).
- DECIDE: compare a*(512) to e053b's small_0.84M cell (a*=7, CI [5,13]).
  ABSOLUTE window [5,13]; PROPORTIONAL window = [5,13]x(511/255) =
  [10.0, 26.1]. Verdict by CI overlap with honest ambiguity labeling.

Registered predictions (before any measurement):
- PA (absolute): a*(512) in [5,13] (CI overlaps the absolute window and
  the point is nearer 7 than 14).
- PP (proportional): a*(512) in [10,26] (nearer 14).
- Anomaly out of both windows -> the 512 window changes the regime;
  report as such (super-proportional / window-dependent onset).

Run:    python lab/e053c_ctx512.py
Smoke:  E053C_SMOKE=1 python lab/e053c_ctx512.py   (CPU-only, tiny, own ckpt)
Outputs: runs/e053c/metrics.json + runs/e053c/e053c_ctx512.png
Envelope: ONE training <=1M params, batch 32, <=180s; gpu_ok() before
launch; no NOTES/THINKING/QUEUE/STATE edits; no commit (dispatcher owns).
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, str(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import json  # noqa: E402
import math  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import common  # noqa: E402
from common import (  # noqa: E402
    MAX_MODEL_PARAMS_DEFAULT,
    REPO,
    Cfg,
    CharCorpus,
    TinyGPT,
    cooldown,
    estimate_loss,
    gpu_ok,
    gpu_status,
    run_dir,
    save_json,
    set_seed,
    train_model,
)

# ------------------------------------------------------------------ constants
SMOKE = os.environ.get("E053C_SMOKE") == "1"

PROMPT_TOK = 64
T_TOTAL = 128 if SMOKE else 512            # ctx-512 window (the decider)
P = T_TOTAL - 1                            # sweep positions / max age
TEMP, TOPK = 0.8, 40
SEED_PROMPT, SEED_SAMPLE = 202, 7          # e053b's exact seeds
THRESH = 0.01                              # dead-weight threshold (nats)
A_STAR_K = 5                               # sustained window (e053b verbatim)
N_PROMPTS = 8                              # same draw size as e053b
B = 2 if SMOKE else 4                      # e053b's uniform B
BOOT_N = 200 if SMOKE else 1000
THREADS = 12
torch.set_num_threads(THREADS)

# ---- training (ONE net; envelope: <=1M params, batch 32, <=180s)
SEED_TRAIN = 42                            # e005s's training seed
TRAIN_STEPS = 40 if SMOKE else 4000        # e005s schedule target
TRAIN_BATCH = 4 if SMOKE else 32           # 32x512 == 64x256 tokens/step
TRAIN_CAP_S = 30.0 if SMOKE else 180.0
CK = REPO / "runs" / "checkpoints"
CKPT = CK / ("e053c_smoke.pt" if SMOKE else "e053c_ctx512.pt")

CHUNK_ROWS = 32 if SMOKE else 128          # sweep rows per forward chunk
E053B_METRICS = REPO / "runs" / "e053b" / "metrics.json"
REF_CELL = "small_0.84M"                   # the matched ctx-256 comparator
REF_T = 255                                # e053b ages 1..255
T0 = time.time()
BUDGET_S = 240.0 if SMOKE else 1500.0


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# ------------------------------------------------- machinery (VERBATIM e053b)

@torch.no_grad()
def _manual_chunk(net: TinyGPT, idxs, vzero, kdrop, layers):
    """Exact manual forward, returns LAST-position logits (N, vocab)."""
    N, T = idxs.shape
    pos = torch.arange(T, device=idxs.device)
    x = net.wte(idxs) + net.wpe(pos).unsqueeze(0)
    H = net.cfg.n_head
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idxs.device), diagonal=1)
    nondiag = ~torch.eye(T, dtype=torch.bool, device=idxs.device)
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
def manual_logits(net, idxs, vzero=None, kdrop=None, layers=None, chunk=64):
    outs = []
    for i in range(0, idxs.shape[0], chunk):
        outs.append(_manual_chunk(net, idxs[i:i + chunk],
                                  vzero[i:i + chunk] if vzero is not None else None,
                                  kdrop[i:i + chunk] if kdrop is not None else None,
                                  layers))
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


def boot_ci(X, n: int = 1000, seed: int = 0):
    """Bootstrap CI over the sequence axis. X: (n_seq, ...) -> (lo, hi)."""
    X = np.asarray(X, float)
    if X.shape[0] < 2:
        return X[0], X[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, X.shape[0], size=(n, X.shape[0]))
    means = X[idx].mean(1)
    return np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)


def onset_age(dce_by_age: np.ndarray, k: int = A_STAR_K, thresh: float = THRESH):
    """dce_by_age: ages ascending (1=youngest ... P=oldest/sink).
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
    x = net.wte(idx) + net.wpe(torch.arange(T, device=idx.device))
    H = net.cfg.n_head
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idx.device), diagonal=1)
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
    x = net.wte(toks) + net.wpe(torch.full((B,), pos, device=toks.device))
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
def generate_batch(net: TinyGPT, prompts, gen: torch.Generator, dev: str):
    """Free-run 64->T_TOTAL for B sequences at once, fixed anchor. Forwards on
    `dev`; SAMPLING on CPU logits from the shared seed-7 CPU generator in row
    order per step (e053b-identical stream math). Returns idx (B, T) CPU,
    final (position-T-1) logits CPU copy, and the dev copy for gates."""
    B = len(prompts)
    idx = torch.stack(prompts)                                    # CPU
    logits_dev, kv = prefill_batch(net, idx.to(dev))
    logits_cpu = logits_dev.detach().float().cpu()
    for t in range(PROMPT_TOK, T_TOTAL):
        toks = torch.zeros(B, dtype=torch.long)
        for j in range(B):
            tok, _ = sample_and_ce(logits_cpu[j], gen)
            toks[j] = tok
        idx = torch.cat([idx, toks[:, None]], 1)
        if t < T_TOTAL - 1:
            logits_dev = decode_step_batch(net, toks.to(dev), t, kv)
            logits_cpu = logits_dev.detach().float().cpu()
    return idx, logits_cpu, logits_dev.detach()


@torch.no_grad()
def fine_sweep(net: TinyGPT, idx_cpu: torch.Tensor, final_logits_cpu: torch.Tensor,
               ch: int, dev: str):
    """Final-step fine V-zero sweep for all B sequences at once.
    Returns dce (B, P) indexed by POSITION (0..P-1); age of pos p =
    T_TOTAL-1-p, so the age-ascending view is dce[:, ::-1]. [e053b math]"""
    B = idx_cpu.shape[0]
    ctx, tgt = idx_cpu[:, :-1], idx_cpu[:, -1]                   # (B, P), (B,)
    p_clean = torch.softmax(final_logits_cpu.float(), -1)        # CPU, as e053b
    ce_clean = -torch.log(p_clean[torch.arange(B), tgt].clamp_min(1e-12))
    idxs = ctx.to(dev).repeat_interleave(P, dim=0)               # (B*P, P)
    vz = torch.eye(P, dtype=torch.bool, device=dev).repeat(B, 1)
    les = manual_logits(net, idxs, vz, None, None, chunk=ch)
    lp = torch.log_softmax(les.float(), -1)
    rows = torch.arange(B * P, device=dev)
    dce = (-ce_clean.repeat_interleave(P).to(dev)
           - lp[rows, tgt.to(dev).repeat_interleave(P)])
    return dce.view(B, P).cpu().numpy()


@torch.no_grad()
def sweep_subset(net: TinyGPT, ctx_cpu: torch.Tensor, tgt: int, ce0: float,
                 positions: list, ch: int, dev: str):
    """dCE for single-position V-zero lesions on `positions` of one context."""
    n = len(positions)
    Pl = ctx_cpu.shape[0]
    idxs = ctx_cpu[None].repeat(n, 1).to(dev)
    vz = torch.zeros(n, Pl, dtype=torch.bool, device=dev)
    for r, p in enumerate(positions):
        vz[r, p] = True
    les = manual_logits(net, idxs, vz, None, None, chunk=ch)
    lp = torch.log_softmax(les.float(), -1)
    rows = torch.arange(n, device=dev)
    tg = torch.full((n,), int(tgt), dtype=torch.long, device=dev)
    return (-ce0 - lp[rows, tg]).cpu().numpy()


# ------------------------------------------------------------------ analysis


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


def analyze_cell(sweep_pos: np.ndarray):
    """sweep_pos: (S, P) V-zero dCE by position. Returns the derived block."""
    dce_age = sweep_pos[:, ::-1]                       # (S, P) young->old
    mean_age = dce_age.mean(0)
    ages = np.arange(1, T_TOTAL)                       # 1..P
    a_star, naive, robust = a_star_rule(mean_age)
    a_lo, a_hi = bootstrap_stat(dce_age, lambda m: a_star_rule(m.mean(0))[0])
    live = mean_age >= THRESH
    live_frac = float(live.mean())
    lf_lo, lf_hi = bootstrap_stat(dce_age, lambda m: float((m.mean(0) >= THRESH).mean()))
    frac = (a_star / P) if a_star is not None else None
    if a_star is not None:
        step_live = ages < a_star                      # a*-implied step function
        match = float((step_live == live).mean())
        junk_old = float((dce_age[:, a_star - 1:] <= -THRESH).mean())
    else:
        match = junk_old = None
    return dict(
        a_star=a_star, a_star_naive=naive, a_star_robust=robust,
        a_star_ci=[a_lo, a_hi], a_star_frac=frac,
        live_frac=live_frac, live_frac_ci=[lf_lo, lf_hi],
        identity_gap=(frac - live_frac) if frac is not None else None,
        one_statistic=bool(frac is not None and abs(frac - live_frac) <= 0.02),
        step_match_frac=match,
        junk_frac_all=float((dce_age <= -THRESH).mean()),
        junk_frac_beyond_ast=junk_old,
        monotone_frac=float(np.mean(np.diff(mean_age) < 0)),
        per_seq_a_star=[a_star_rule(dce_age[s])[0] for s in range(dce_age.shape[0])],
        mean_age_curve=mean_age, ci=boot_ci(dce_age, BOOT_N),
        dce_age=dce_age,
    )


def load_ref():
    """e053b's small_0.84M fine cell — the registered ctx-256 comparator."""
    if not E053B_METRICS.exists():
        return None
    m = json.loads(E053B_METRICS.read_text())
    c = m["cells"].get(REF_CELL)
    if c is None:
        return None
    d = c["derived"]
    return dict(
        cell=REF_CELL, a_star=d["a_star"], a_star_ci=d["a_star_ci"],
        a_star_frac=d["a_star_frac"], live_frac=d["live_frac"],
        val_ce=c["val_ce"], B=c["protocol"]["B"],
        per_seq_a_star=d["per_seq_a_star"],
        age=c["age"], vzero_dce_mean=c["vzero_dce_mean"], vzero_ci=c["vzero_ci"],
        source=str(E053B_METRICS),
    )


# ---------------------------------------------------------------------- main


def main():
    out_dir = run_dir("e053c")
    fname = "metrics_smoke.json" if SMOKE else "metrics.json"
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ---- envelope: ONE training, gpu_ok gate + cooldown BEFORE launch
    g0 = gpu_status()
    log(f"gpu at start: util {g0['util']:.0f}% temp {g0['temp']:.0f}C "
        f"mem {g0['mem_used']:.0f}/{g0['mem_total']:.0f}MB power {g0['power']:.0f}W")
    if not SMOKE:
        cooldown(60.0)                       # pre-launch cooldown (task spec)
    gpu_launch_ok = gpu_ok()
    train_dev = "cpu" if (SMOKE or not torch.cuda.is_available() or not gpu_launch_ok) \
        else "cuda"
    common.DEVICE = train_dev                # batches/eval follow the choice
    log(f"train device: {train_dev} (gpu_ok={gpu_launch_ok}, smoke={SMOKE})")

    corp = CharCorpus(REPO / "data" / "input.txt")     # seed 1337 (e005s)
    assert corp.vocab_size == 65

    # ---- TRAIN: 0.84M-class cfg @ block_size=512, seed 42, batch 32, 180s cap
    set_seed(SEED_TRAIN)
    cfg = Cfg(vocab=corp.vocab_size, n_layer=4, n_head=4, n_embd=128,
              block_size=T_TOTAL)
    net = TinyGPT(cfg)
    n_params = net.num_params()
    assert n_params <= MAX_MODEL_PARAMS_DEFAULT, f"envelope: {n_params} params"
    log(f"cfg {cfg_dict_str(cfg)} | {n_params:,} params (<=1M) | "
        f"batch {TRAIN_BATCH} x ctx {T_TOTAL} = {TRAIN_BATCH * T_TOTAL} tok/step "
        f"(e005s small: 64x256 — matched)")
    net = net.to(train_dev)
    hist = train_model(net, corp, steps=TRAIN_STEPS, lr=1e-3,
                       batch_size=TRAIN_BATCH, max_seconds=TRAIN_CAP_S,
                       eval_every=25 if SMOKE else 250, ckpt=CKPT)
    steps_done = int(hist[-1]["step"]) if hist else 0
    g1 = gpu_status()
    log(f"trained to step {steps_done}/{TRAIN_STEPS} "
        f"(cap {TRAIN_CAP_S:.0f}s) | gpu after: {g1['temp']:.0f}C {g1['power']:.0f}W")
    val_ce = estimate_loss(net, corp, "val", n_batches=12)
    log(f"val CE (ctx-{T_TOTAL}): {val_ce:.4f} "
        f"[e053b small ctx-256: 1.5392; e005s anchor: 1.5581]")
    net.eval()

    # ---- measurement device (GPU for the heavy sweeps; task-authorized)
    if not SMOKE:
        cooldown(45.0)
    meas_ok = (SMOKE is False) and torch.cuda.is_available() and gpu_ok()
    meas_dev = "cuda" if meas_ok else "cpu"
    log(f"measure device: {meas_dev}")

    # ---- prompts: seed-202 draw, SAME modulus as e053b -> identical positions
    gen_p = torch.Generator().manual_seed(SEED_PROMPT)
    ix = torch.randint(len(corp.val) - PROMPT_TOK - 1, (N_PROMPTS,), generator=gen_p)
    prompts = [corp.val[i:i + PROMPT_TOK] for i in ix]
    log(f"{N_PROMPTS} prompts (seed {SEED_PROMPT}); using first B={B}; "
        f"prompt0 prefix: {corp.decode(prompts[0])[:32]!r}")

    gen = torch.Generator().manual_seed(SEED_SAMPLE)   # e053b's per-cell seed
    idx, final_logits_cpu, final_logits_dev = generate_batch(net, prompts[:B], gen,
                                                             meas_dev)
    log(f"generated {B} seqs 64->{T_TOTAL} (fixed anchor)")

    gates = dict(envelope=dict(params=n_params, batch=TRAIN_BATCH,
                               cap_s=TRAIN_CAP_S, steps_target=TRAIN_STEPS,
                               max_params=MAX_MODEL_PARAMS_DEFAULT,
                               gpu_at_start=g0, gpu_launch_ok=gpu_launch_ok,
                               gpu_after_train=g1, train_dev=train_dev,
                               meas_dev=meas_dev),
                 G1_val_ce=dict(val_ce=val_ce, ref_e053b_small=1.5392,
                                ref_e005s_anchor=1.5581,
                                plausible=bool(val_ce < 1.70)))

    # G0b: batched incremental-KV final logits vs full recompute (prob dev)
    with torch.no_grad():
        full = manual_logits(net, idx[:, :-1].to(meas_dev), None, None, None,
                             chunk=max(1, B))
    p_inc = torch.softmax(final_logits_dev.float(), -1)
    p_full = torch.softmax(full.float(), -1)
    dev_b = float((p_inc - p_full).abs().max())
    gates["G0b_kvs_vs_full"] = dict(max_prob_dev=dev_b, ok=bool(dev_b < 1e-4))
    log(f"G0b incremental-KV vs full recompute: max prob dev {dev_b:.2e} "
        f"-> {'PASS' if dev_b < 1e-4 else 'FAIL'}")

    # G0-dev: CPU vs measure-device dCE on a subset (incl. the young spike)
    ctx0, tgt0 = idx[0, :-1], int(idx[0, -1])
    p_clean = torch.softmax(final_logits_cpu[0].float(), -1)
    ce0 = float(-math.log(max(p_clean[tgt0].item(), 1e-12)))
    positions = sorted(set(list(range(0, P, max(1, P // 8)))[:8]
                           + [P - 1 - i for i in range(8)]))
    d_dev = sweep_subset(net, ctx0, tgt0, ce0, positions, 8, meas_dev)
    if meas_dev != "cpu":
        import copy
        net_c = copy.deepcopy(net).to("cpu")
        net_c.eval()
        d_cpu = sweep_subset(net_c, ctx0, tgt0, ce0, positions, 8, "cpu")
        del net_c
    else:
        d_cpu = d_dev
    dev_g = float(np.abs(d_dev - d_cpu).max())
    gates["G0_dev_equivalence"] = dict(positions=positions, max_dce_dev=dev_g,
                                       ok=bool(dev_g < 1e-4))
    log(f"G0-dev CPU vs {meas_dev} dCE (subset n={len(positions)}): "
        f"max dev {dev_g:.2e} -> {'PASS' if dev_g < 1e-4 else 'FAIL'}")

    if not (dev_b < 1e-4 and dev_g < 1e-4) and meas_dev != "cpu":
        log("gate FAIL on GPU -> honest fallback: redoing measurement on CPU")
        meas_dev = "cpu"
        net = net.to("cpu")
        gen = torch.Generator().manual_seed(SEED_SAMPLE)
        idx, final_logits_cpu, final_logits_dev = generate_batch(net, prompts[:B],
                                                                 gen, "cpu")
        gates["fallback_cpu"] = True

    # ---- the fine sweep (the measurement)
    t_sw = time.time()
    sweep = fine_sweep(net, idx, final_logits_cpu, CHUNK_ROWS, meas_dev)
    nan_n = int(np.isnan(sweep).sum())
    if nan_n:
        sweep = np.nan_to_num(sweep, nan=0.0)
    log(f"sweep done: {sweep.shape} on {meas_dev} in {time.time() - t_sw:.1f}s "
        f"(nan->0: {nan_n})")

    d = analyze_cell(sweep)
    log(f"a*(ctx-{T_TOTAL}) = {d['a_star']} CI [{d['a_star_ci'][0]:.0f}, "
        f"{d['a_star_ci'][1]:.0f}] naive={d['a_star_naive']} "
        f"robust={d['a_star_robust']} | live_frac={d['live_frac']:.3f} "
        f"| per-seq {d['per_seq_a_star']}")

    # ---- THE REGISTERED DECISION (absolute vs proportional)
    ref = load_ref()
    if ref is None:
        ref = dict(cell=REF_CELL, a_star=7, a_star_ci=[5.0, 13.0],
                   a_star_frac=7 / 255, live_frac=0.051, val_ce=1.5392, B=4,
                   per_seq_a_star=[5, 13, 6, 3], age=None, vzero_dce_mean=None,
                   vzero_ci=None, source="hardcoded fallback (e053b missing)")
        log("WARNING: e053b metrics not found; using hardcoded reference")
    pred_abs = float(ref["a_star"])                                  # 7 tokens
    pred_prop = float(ref["a_star"]) / REF_T * P                     # 7/255*511
    win_abs = [float(ref["a_star_ci"][0]), float(ref["a_star_ci"][1])]
    win_prop = [win_abs[0] / REF_T * P, win_abs[1] / REF_T * P]
    ci512 = [float(d["a_star_ci"][0]), float(d["a_star_ci"][1])]
    a512 = d["a_star"]

    def overlaps(c1, c2):
        return not (c1[1] < c2[0] or c2[1] < c1[0])

    if a512 is None:
        verdict, favored, strength = ("NO ONSET (no age below threshold — window "
                                      "fully live)", None, "anomaly")
        ov_abs = ov_prop = None
    else:
        ov_abs, ov_prop = overlaps(ci512, win_abs), overlaps(ci512, win_prop)
        d_abs, d_prop = abs(a512 - pred_abs), abs(a512 - pred_prop)
        point = "absolute" if d_abs < d_prop else (
            "proportional" if d_prop < d_abs else "tie")
        if ov_abs and not ov_prop:
            verdict, favored, strength = (f"ABSOLUTE: a*(512)={a512} CI "
                                          f"[{ci512[0]:.0f},{ci512[1]:.0f}] "
                                          f"overlaps absolute window {win_abs} "
                                          f"and excludes proportional {win_prop}",
                                          "absolute", "clear")
        elif ov_prop and not ov_abs:
            verdict, favored, strength = (f"PROPORTIONAL: a*(512)={a512} CI "
                                          f"[{ci512[0]:.0f},{ci512[1]:.0f}] "
                                          f"overlaps proportional window {win_prop} "
                                          f"and excludes absolute {win_abs}",
                                          "proportional", "clear")
        elif ov_abs and ov_prop:
            verdict, favored, strength = (f"AMBIGUOUS: a*(512)={a512} CI "
                                          f"[{ci512[0]:.0f},{ci512[1]:.0f}] "
                                          f"overlaps BOTH windows (abs {win_abs}, "
                                          f"prop {win_prop}); point estimate "
                                          f"favors {point}", point, "ambiguous")
        else:
            verdict, favored, strength = (f"NEITHER: a*(512)={a512} CI "
                                          f"[{ci512[0]:.0f},{ci512[1]:.0f}] "
                                          f"excludes BOTH windows — onset is "
                                          f"window-dependent (super-proportional)",
                                          "neither", "anomaly")
    log(f"DECISION: {verdict}")

    decision = dict(
        reference_cell=ref["cell"], a_star_256=ref["a_star"],
        a_star_256_ci=ref["a_star_ci"], a_star_512=a512, a_star_512_ci=ci512,
        predictions=dict(absolute_tokens=pred_abs, proportional_tokens=pred_prop,
                         absolute_window=win_abs, proportional_window=win_prop,
                         note="proportional = a*(256)/255 x 511 (frac of window)"),
        distances=dict(to_absolute=None if a512 is None else abs(a512 - pred_abs),
                       to_proportional=None if a512 is None else abs(a512 - pred_prop)),
        ci_overlaps=dict(absolute=ov_abs, proportional=ov_prop),
        ratio_512_over_256=(a512 / ref["a_star"]) if a512 else None,
        frac_512=d["a_star_frac"], frac_256=ref["a_star_frac"],
        verdict=verdict, favored=favored, strength=strength,
        caveats=[
            "B=4 sequences; e053b's small-cell CI [5,13] was itself wide "
            "(per-seq a* 3-13) — treat single-cell a* as noisy",
            "e053b PB3 failed (a* NOT stable across cells at ctx-256: "
            "7/25/35/182/32); this decider compares the MATCHED arch pair "
            "(4L/4H/128d, matched tokens/step, same corpus+seeds), the "
            "cleanest available comparison",
            "the two registered windows overlap ([10,13]); an a* landing "
            "there cannot decide — reported as AMBIGUOUS, not forced",
        ],
    )

    metrics = dict(
        experiment="e053c_ctx512", phase=2,
        purpose="registered decider (e053b/T031): a*(ctx512) ~ a*(ctx256) tokens "
                "-> ABSOLUTE onset; ~ frac x 512 -> PROPORTIONAL. One 0.84M-class "
                "net retrained at block_size=512; e053b fine-onset protocol "
                "verbatim (same seeds, same prompt draw, same onset rule).",
        started=started, wall_s=elapsed(), smoke=SMOKE, threads=THREADS,
        seeds=dict(train=SEED_TRAIN, corpus=1337, prompts=SEED_PROMPT,
                   sampling=SEED_SAMPLE),
        training=dict(ckpt=str(CKPT), steps_done=steps_done,
                      steps_target=TRAIN_STEPS, batch=TRAIN_BATCH,
                      cap_s=TRAIN_CAP_S, device=train_dev,
                      tokens_seen=steps_done * TRAIN_BATCH * T_TOTAL,
                      e005s_small_reference="4000 x 64 x 256 = 65.5M tokens "
                                            "(assumed complete)",
                      history=hist),
        protocol=dict(B=B, prompt_tokens=PROMPT_TOK, t_total=T_TOTAL, temp=TEMP,
                      topk=TOPK, fixed_anchor=True, threshold=THRESH,
                      a_star_window=A_STAR_K, boot_n=BOOT_N,
                      sweep_chunk_rows=CHUNK_ROWS, measure_dev=meas_dev,
                      sampling="CPU tensors + shared seed-7 CPU generator, "
                               "per-row in row order (e053b-identical)"),
        gates=gates,
        cell=dict(
            name="small_0.84M_ctx512", ckpt=str(CKPT),
            arch=dict(n_layer=4, n_head=4, n_embd=128, block_size=T_TOTAL,
                      vocab=65),
            params=n_params, val_ce=val_ce, steps_trained=steps_done,
            age=list(range(1, T_TOTAL)),
            vzero_dce_mean=d["mean_age_curve"].tolist(),
            vzero_ci=[d["ci"][0].tolist(), d["ci"][1].tolist()],
            per_seq_sweeps_by_age=d["dce_age"].tolist(),
            derived={k: v for k, v in d.items()
                     if k not in ("mean_age_curve", "ci", "dce_age")},
        ),
        reference_ctx256=dict(
            cell=ref["cell"], a_star=ref["a_star"], a_star_ci=ref["a_star_ci"],
            a_star_frac=ref["a_star_frac"], live_frac=ref["live_frac"],
            val_ce=ref["val_ce"], B=ref["B"], per_seq_a_star=ref["per_seq_a_star"],
            age=ref["age"], vzero_dce_mean=ref["vzero_dce_mean"],
            vzero_ci=ref["vzero_ci"], source=ref["source"],
        ),
        decision=decision,
    )
    save_json(out_dir / fname, metrics)
    log(f"{fname} written")

    plot(out_dir / ("e053c_smoke.png" if SMOKE else "e053c_ctx512.png"),
         metrics, d, ref, decision)
    log(f"plot written; total wall {elapsed():.0f}s")


def cfg_dict_str(cfg):
    return (f"{cfg.n_layer}L/{cfg.n_head}H/{cfg.n_embd}d/wpe-{cfg.block_size}")


# ---------------------------------------------------------------------- plot


def plot(path: Path, M: dict, d: dict, ref: dict, decision: dict):
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(3, 2, hspace=0.38, wspace=0.25)

    # (0,0) ctx-256 reference curve (e053b small cell)
    ax = fig.add_subplot(gs[0, 0])
    m = np.asarray([0.0])
    hi = np.asarray([0.1])
    if ref.get("vzero_dce_mean"):
        age = np.asarray(ref["age"])
        m = np.asarray(ref["vzero_dce_mean"])
        lo, hi = np.asarray(ref["vzero_ci"][0]), np.asarray(ref["vzero_ci"][1])
        ax.fill_between(age, lo, hi, alpha=0.25, color="tab:blue")
        ax.plot(age, m, lw=1.4, color="tab:blue")
        ax.axvline(ref["a_star"], color="tab:blue", ls="--", lw=1.2)
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(THRESH, color="gray", ls=":", lw=0.9)
    ax.set_xlim(0, REF_T + 1)
    ax.set_ylim(min(-0.1, float(np.nanpercentile(m, 2)) - 0.05),
                max(0.1, float(np.nanpercentile(hi, 98))))
    ax.set_title(f"ctx-256 reference ({ref['cell']}, e053b)\n"
                 f"a*={ref['a_star']} CI {ref['a_star_ci']} "
                 f"| val CE {ref['val_ce']:.4f}", fontsize=10)
    ax.set_xlabel("cache age (tokens)")
    ax.set_ylabel("V-zero dCE (nats)")

    # (0,1) ctx-512 curve + both predictions
    ax = fig.add_subplot(gs[0, 1])
    age = np.asarray(M["cell"]["age"])
    m = np.asarray(M["cell"]["vzero_dce_mean"])
    lo, hi = np.asarray(M["cell"]["vzero_ci"][0]), np.asarray(M["cell"]["vzero_ci"][1])
    ax.fill_between(age, lo, hi, alpha=0.25, color="tab:red")
    ax.plot(age, m, lw=1.4, color="tab:red")
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(THRESH, color="gray", ls=":", lw=0.9)
    ax.axvline(decision["predictions"]["absolute_tokens"], color="tab:blue",
               ls=":", lw=1.4)
    ax.axvline(decision["predictions"]["proportional_tokens"], color="gray",
               ls="-.", lw=1.4)
    if d["a_star"]:
        ax.axvline(d["a_star"], color="tab:red", ls="--", lw=1.4)
    ax.set_xlim(0, T_TOTAL)
    ax.set_ylim(min(-0.1, np.nanpercentile(m, 2) - 0.05),
                np.nanpercentile(hi, 98))
    ax.set_title(f"ctx-512 (this run, {M['cell']['name']})\n"
                 f"a*={d['a_star']} CI [{d['a_star_ci'][0]:.0f},"
                 f"{d['a_star_ci'][1]:.0f}] | val CE {M['cell']['val_ce']:.4f} "
                 f"| B={M['protocol']['B']}", fontsize=10)
    ax.set_xlabel(f"cache age (tokens); blue dotted = ABSOLUTE pred "
                  f"{decision['predictions']['absolute_tokens']:.0f}, gray dash-dot"
                  f" = PROPORTIONAL pred {decision['predictions']['proportional_tokens']:.0f}")

    # (1,0) overlay by fraction of window
    ax = fig.add_subplot(gs[0 + 1, 0])
    if ref.get("vzero_dce_mean"):
        ax.plot(np.asarray(ref["age"]) / REF_T, ref["vzero_dce_mean"], lw=1.3,
                color="tab:blue", label=f"ctx-256 (a*/T={ref['a_star_frac']:.3f})")
        ax.axvline(ref["a_star_frac"], color="tab:blue", ls="--", lw=1.0)
    ax.plot(age / P, m, lw=1.3, color="tab:red",
            label=f"ctx-512 (a*/T={d['a_star_frac']:.3f})" if d["a_star_frac"]
            else "ctx-512")
    if d["a_star_frac"]:
        ax.axvline(d["a_star_frac"], color="tab:red", ls="--", lw=1.0)
    ax.axhline(THRESH, color="gray", ls=":", lw=0.9)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("cache age / window (fraction)")
    ax.set_ylabel("V-zero dCE (nats)")
    ax.legend(fontsize=9)
    ax.set_title("overlay by fraction of window — PROPORTIONAL would superpose; "
                 "ABSOLUTE would shift left", fontsize=10)

    # (1,1) per-sequence a* spread
    ax = fig.add_subplot(gs[1, 1])
    for i, (lbl, vals, col) in enumerate([
            ("ctx-256", ref["per_seq_a_star"], "tab:blue"),
            ("ctx-512", d["per_seq_a_star"], "tab:red")]):
        ax.scatter([i] * len(vals), vals, s=40, color=col, alpha=0.75,
                   edgecolors="k", linewidths=0.4, zorder=3)
    ax.scatter([0], [ref["a_star"]], marker="D", s=90, color="tab:blue", zorder=4)
    if d["a_star"]:
        ax.scatter([1], [d["a_star"]], marker="D", s=90, color="tab:red", zorder=4)
    ax.errorbar([1], [d["a_star"]],
                yerr=[[max(0, d["a_star"] - d["a_star_ci"][0])],
                      [max(0, d["a_star_ci"][1] - d["a_star"])]],
                color="tab:red", capsize=4, lw=1.2)
    ax.set_xticks([0, 1], ["ctx-256 (e053b)", "ctx-512 (this)"])
    ax.set_ylabel("onset a* (tokens)")
    ax.set_title("per-sequence onset spread (dots=seqs, diamond=cell)", fontsize=10)

    # (2,0) training history
    ax = fig.add_subplot(gs[2, 0])
    h = M["training"]["history"]
    if h:
        ax.plot([x["step"] for x in h], [x["train_loss"] for x in h], alpha=0.7,
                label="train")
        ax.plot([x["step"] for x in h], [x["val_loss"] for x in h], alpha=0.7,
                label="val")
    ax.axhline(1.5581, color="gray", ls=":", lw=1.0)
    ax.text(0.02, 1.5581 + 0.01, "e005s small anchor 1.5581 (ctx-256)",
            fontsize=8, transform=ax.get_yaxis_transform())
    ax.set_xlabel("step")
    ax.set_ylabel("CE (nats/char)")
    ax.set_title(f"training: {M['cell']['params']:,} params, batch "
                 f"{M['training']['batch']}, cap {M['training']['cap_s']:.0f}s, "
                 f"{M['training']['device']}", fontsize=10)
    ax.legend(fontsize=9)

    # (2,1) verdict text
    ax = fig.add_subplot(gs[2, 1])
    ax.axis("off")
    p = decision["predictions"]
    lines = [
        f"a*(ctx256) = {decision['a_star_256']}  CI {decision['a_star_256_ci']}",
        f"a*(ctx512) = {decision['a_star_512']}  CI "
        f"[{decision['a_star_512_ci'][0]:.0f},{decision['a_star_512_ci'][1]:.0f}]",
        f"ABSOLUTE predicts {p['absolute_tokens']:.0f} tokens "
        f"(window {p['absolute_window']})",
        f"PROPORTIONAL predicts {p['proportional_tokens']:.1f} tokens "
        f"(window [{p['proportional_window'][0]:.1f},"
        f"{p['proportional_window'][1]:.1f}])",
        "",
        f"VERDICT [{decision['strength']}]: {decision['favored']}",
        decision["verdict"],
        "",
        f"ratio a*(512)/a*(256) = "
        f"{decision['ratio_512_over_256']:.2f}" if decision["ratio_512_over_256"]
        else "ratio n/a",
        f"frac: {decision['frac_256']:.4f} (256) vs "
        f"{decision['frac_512']:.4f} (512)",
        "caveat: B=4, wide CIs; e053b PB3 showed a* varies across",
        "cells — this is the MATCHED-arch pair comparison.",
    ]
    ax.text(0.02, 0.95, "E053c — ctx-512 onset decider (registered at e053b)",
            fontsize=13, weight="bold", va="top")
    for i, t in enumerate(lines):
        ax.text(0.02, 0.82 - i * 0.058, t, fontsize=9.5, va="top",
                family="monospace")

    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
