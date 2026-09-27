"""E097 — T051's registered SINK-ASYMMETRY hedge: position-stratified subset
removal at fixed k inside the anchor band (does WHICH-entry irrelevance
survive stratifying by position, or do sink-adjacent old entries carry
disproportionate mass?). CPU-only.

The hedge (scratch/massaction_key_lit.md novelty verdict, frozen in T051):
e089's variance collapse said WHICH entries doesn't matter — but its subsets
were UNIFORM over ages 97-447 (band positions 64..414). Before claiming full
which-irrelevance for the mass-action law, stratify by position: if
sink-adjacent (old) entries carry disproportionate mass, the claim needs its
boundary (the KV-eviction literature's position priors echo).

DESIGN (registered here BEFORE compute; bars frozen from the tasking).
Battery: the e053c ctx-512 net (873,472 params, val CE 1.5227), the seed-202
8-draw prompts, B=8, the seed-7 control free run (the e075..e089 A-none
convention; G3 gated against e080's stored none arm). Anchor band: positions
64..414 = final-frame ages 97..447 (self-generated), verbatim e089.

STRATA (position thirds of the 351-position band; age of position p = 511-p):
  old = 64..180   (ages 331..447 — nearest the prompt/sink side),
  mid = 181..297  (ages 214..330),
  new = 298..414  (ages 97..213 — nearest the generation head).

ARMS (10 draws each, 80 draws total) at k=128 (PRIMARY) and k=64 (SECONDARY,
runtime permitting — it was):
  k=128 stratum arms — a pure within-stratum draw is combinatorially
  impossible (117 stratum positions < 128), so the REGISTERED
  operationalization is MAXIMAL WITHIN-STRATUM CONCENTRATION at fixed k: the
  subset = ALL 117 stratum positions + 11 draws uniform from the 234-position
  complement (91% concentration; the 10 draws vary only the 11 fills). The
  contrast: uniform k=128 draws land ~43 of 128 entries in any given third.
  k=64 stratum arms — PURE within-stratum: 64 draws uniform inside the
  stratum (64 <= 117, fully random; the clean within-stratum test).
  k=128 / k=64 uniform controls — the e089 replication arm VERBATIM (fresh
  RNG stream): rng.choice(band, k, replace=False), rejected and redrawn while
  min(cols) > 350 (the full-tail-exposure floor; rejections tallied). All
  stratum arms satisfy min <= 297 by construction (asserted, no rejection
  needed).
RNG (documented): numpy default_rng(seed=97), consumed arm-major (k128: old,
mid, new, unif; then k64: old, mid, new, unif), draw-minor. Run assignment
(documented): run = (arm_index*10 + draw) % 8 — every run hosts exactly 10
draws and >= 1 draw of every arm.

INSTRUMENT: the e089 rig VERBATIM, zero adaptations — per-entry age-97-crossing
V-zero (entry p zeroed immediately before the decode processing position
p+97), T_int = min(S)+97 teacher-forced, matched-stream continuation with
shared seed (SEED_CONT=971000 + min(S)), clean-net tail judgment 448..511.
cost = clean-judge tail CE(removal stream) - clean-judge tail CE(control).
Edge case (e089 verbatim, tallied): p=414 crosses at 511 > last decode 510 —
unremovable-by-construction (exposure 0, stays live, k_eff = k-1). NOTE the
k=128 'new' arm contains 414 deterministically (k_eff=127 for all its draws).
EXPOSURE CONFOUND (honest, registered): older strata necessarily cross
earlier (old-128: T_int=161 vs new-128: T_int~300+), so position strata
mechanically couple with removal timing — an old entry can only matter
through its longer causal leash; that coupling is intrinsic to the asymmetry
question, reported as texture (per-arm mean min/exposure), not controlled
away.

REGISTERED BARS (frozen from the tasking; evaluated on arm MEANS over 10
draws, cluster bootstrap CIs over the 8 runs; ratio = stratum mean / uniform
mean at the same k; priority order indeterminate -> asymmetry -> survives):
  0. uniform(k=128) mean <= 0 -> INDETERMINATE (no ratio defined).
  1. POSITION ASYMMETRY: any k=128 stratum with ratio >= 2.0 AND its ratio
     cluster-CI excluding 1.0 (lower bound > 1) => the sink-adjacent side
     carries disproportionate mass; the mass-action claim gets its boundary.
     (Ratio bootstrap guard, registered: resamples where the uniform mean
     <= 0.05 nats are discarded as denominator-unstable; validity count
     reported.)
  2. WHICH-IRRELEVANCE SURVIVES: uniform(k=128) mean > 0 AND ALL THREE
     strata with 0.75 <= ratio <= 1.25 AND stratum mean-CI overlapping the
     uniform mean-CI => the hedge closes; mass-action stands un-hedged.
  3. Anything else -> MIXED/TEXTURE (per-arm table, honest).
  SECONDARY (report-only, same clauses at k=64 if uniform(k=64) mean > 0;
  e089's k=64 level was +0.204 [0.069, 0.400] — smaller signal, so k=128 is
  the primary). Descriptive riders throughout: per-draw cost vs subset mean
  position, first stream divergence, k_eff, e089 k=128 reference (+0.683).

Gates: G1 val CE vs e053c 1.5227 +-0.02; G2 params 873472; G3 control
battery identity vs e080's stored none arm (clean-judge per-seq, hard bar
1e-4, bitwise flag; e088/e089's scoped G3); G4 dynamic-instrument identity
per group (prefix identity through the teacher-forced token; fired removal
columns exactly 0; non-removed and control columns live; matched seed across
arms; each crossing fired at most once).

Run:     python lab/e097_sink_asymmetry.py
Outputs: runs/e097/metrics.json + runs/e097/sink_asymmetry.png
Envelope: NO training, NO new automations; CPU-only (CUDA_VISIBLE_DEVICES=-1),
8 threads (T050), single step, target ~10-20 min.
No NOTES/THINKING/QUEUE/STATE edits; no commit (the dispatcher owns those).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # task spec: CPU-only (pre-torch)
import torch

# this torch build reports is_available()==True even with an empty
# CUDA_VISIBLE_DEVICES; force CPU so common.DEVICE=="cpu" (e053b..e089)
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
SEED_PROMPT, SEED_SAMPLE = 202, 7         # e053b/e053c/e069..e089 seeds
SEED_SUB = 97                             # e097: stratified-subset rng (fresh)
SEED_CONT = 971000                        # e097: continuation seeds (+ min)
B = 8
N_PROMPTS = 8
BOOT_N = 1000
THREADS = 8                               # T050: 12 spin-thrashes the box
torch.set_num_threads(THREADS)

T_TAIL_START = T_TOTAL - 64               # 448
KEY_T = (T_TOTAL - 64, T_TOTAL - 1)       # judged tail window (448, 511)
G = T_TOTAL - PROMPT_TOK                  # 448 generation steps

# ---- the anchor band (final-frame ages; age of position p = 511 - p) ---------
ANCHOR_POS = (64, 414)                    # ages 97..447 (self-generated)
P1_MAX = 350                              # full-tail-exposure: min+98 <= 448
BAND = np.arange(ANCHOR_POS[0], ANCHOR_POS[1] + 1)   # 351 positions

# ---- position strata (thirds of the 351-position band) -----------------------
STRATA = [                                # (name, lo, hi) — 117 positions each
    ("old", 64, 180),                     # ages 331..447, sink/prompt side
    ("mid", 181, 297),                    # ages 214..330
    ("new", 298, 414),                    # ages 97..213, generation-head side
]
K_PRIMARY, K_SECONDARY = 128, 64
N_DRAWS = 10                              # independent draws per arm
FILL_128 = K_PRIMARY - 117                # 11 uniform complement fills

# arms in rng/run-assignment order: k-major (primary first), strata then unif
ARMS = ([(K_PRIMARY, s[0], s[1], s[2]) for s in STRATA]
        + [(K_PRIMARY, "unif", None, None)]
        + [(K_SECONDARY, s[0], s[1], s[2]) for s in STRATA]
        + [(K_SECONDARY, "unif", None, None)])
ARM_KEYS = [f"k{k}_{name}" for (k, name, _lo, _hi) in ARMS]

# ---- REGISTERED decision numbers (frozen, docstring verbatim) ----------------
SURV_LO, SURV_HI = 0.75, 1.25             # +-25% of the uniform control
ASYM_X = 2.0                              # stratum >= 2x uniform
RATIO_DEN_FLOOR = 0.05                    # bootstrap denominator-stability guard

# ---- reference numbers (protocol-identity gates) -----------------------------
CKPT = REPO / "runs" / "checkpoints" / "e053c_ctx512.pt"
E053C_VAL_CE = 1.5226792494455974         # the net being interrogated
E080_METRICS = REPO / "runs" / "e080" / "metrics.json"
E089_METRICS = REPO / "runs" / "e089" / "metrics.json"
T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# ------------------------------------------------- machinery (VERBATIM e089)

@torch.no_grad()
def manual_all_logits(net: TinyGPT, idxs):
    """Clean forward returning logits at ALL positions (N, T, vocab) — the
    clean-judge instrument (scores a stream without any pruning)."""
    N, T = idxs.shape
    pos = torch.arange(T)
    x = net.wte(idxs) + net.wpe(pos).unsqueeze(0)
    H = net.cfg.n_head
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
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
        y = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(N, T, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
    return net.lm_head(net.ln_f(x))


def sample_and_ce(logits_clean: torch.Tensor, gen: torch.Generator):
    """Sample (temp 0.8, top-k 40) from filtered probs; CE from FULL softmax.
    [VERBATIM e053b — CPU tensors + CPU generator, stream-identical]"""
    lg = logits_clean / TEMP
    v, _ = torch.topk(lg, TOPK)
    lg_f = lg.masked_fill(lg < v[-1], float("-inf"))
    tok = int(torch.multinomial(torch.softmax(lg_f, -1), 1, generator=gen))
    p_full = torch.softmax(logits_clean, -1)
    return tok, float(-math.log(max(p_full[tok].item(), 1e-12)))


@torch.no_grad()
def prefill_batch(net: TinyGPT, idx: torch.Tensor):
    """Batched prefill, (B, Tp) -> last-position logits (B, V) + KV cache.
    [VERBATIM e075/e080/e085/e088/e089]"""
    Bb, T = idx.shape
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
        q = q.view(Bb, T, H, d).transpose(1, 2)
        k = k.view(Bb, T, H, d).transpose(1, 2)
        v = v.view(Bb, T, H, d).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
        att = att.masked_fill(causal, float("-inf"))
        y = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(Bb, T, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
        kv.append((k, v))
    return net.lm_head(net.ln_f(x[:, -1, :])), kv


@torch.no_grad()
def decode_step_batch(net: TinyGPT, toks: torch.Tensor, pos: int, kv: list):
    """Batched incremental decode: (B,) tokens at position pos -> (B, V).
    [VERBATIM e075/e085/e088/e089]"""
    Bb = toks.shape[0]
    x = net.wte(toks) + net.wpe(torch.full((Bb,), pos))
    H = net.cfg.n_head
    for li, blk in enumerate(net.h):
        xh = blk.ln1(x)
        qkv = blk.attn.c_attn(xh)
        C = qkv.shape[-1] // 3
        d = C // H
        q, k, v = qkv.split(C, dim=1)
        q = q.view(Bb, 1, H, d).transpose(1, 2)
        k = k.view(Bb, 1, H, d).transpose(1, 2)
        v = v.view(Bb, 1, H, d).transpose(1, 2)
        kp, vp = kv[li]
        k = torch.cat([kp, k], dim=2)
        v = torch.cat([vp, v], dim=2)
        kv[li] = (k, v)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))   # (B,H,1,t+1)
        probs = torch.softmax(att, -1)
        y = (probs @ v).transpose(1, 2).reshape(Bb, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
    return net.lm_head(net.ln_f(x))


@torch.no_grad()
def generate_control(net: TinyGPT, prompts, gen: torch.Generator):
    """Free-run 64->512 for B sequences (e075/e080/e085/e088/e089 A-none
    VERBATIM stream math)."""
    Bb = len(prompts)
    idx = torch.stack(prompts)
    logits, kv = prefill_batch(net, idx)
    for g in range(G):
        t = PROMPT_TOK + g
        toks = torch.zeros(Bb, dtype=torch.long)
        for j in range(Bb):
            tok, _ce = sample_and_ce(logits[j], gen)
            toks[j] = tok
        idx = torch.cat([idx, toks[:, None]], 1)
        if t < T_TOTAL - 1:
            logits = decode_step_batch(net, toks, t, kv)
    return dict(idx=idx, kv=kv)


@torch.no_grad()
def run_continuation(net: TinyGPT, prefix: torch.Tensor, forced_tok: torch.Tensor,
                     forced_pos: int, seed: int, remove=None):
    """Crossing-time progressive entry-removal continuation (and its matched
    control when remove=None). [VERBATIM e089:]

    remove = list of (row, col): every listed entry p is V-zeroed immediately
    BEFORE the decode step that processes position p+97 (e085's age-97-crossing
    timing applied to sets). The earliest entry's crossing is the forced decode
    itself (T_int = min+97), so the token at forced_pos is teacher-forced and
    the first affected sample is forced_pos+1. Entries whose crossing falls at
    position 511 (p = 414 — beyond the last decode at 510) are NEVER zeroed
    inside the window: unremovable-by-construction (exposure 0), returned
    unfired for honest tallying. Free-run from forced_pos+1 with the shared
    row-order generator — control and removal arms share the seed, so streams
    are row-by-row matched until sampled divergence.

    Returns (idx, kv, fired) where fired = set of (row, col) actually zeroed.
    """
    R = prefix.shape[0]
    gen = torch.Generator().manual_seed(seed)
    logits, kv = prefill_batch(net, prefix)
    fired: set = set()
    cross: dict = {}                       # decode position -> [(row, col)]
    if remove is not None:
        for (r, c) in remove:
            cross.setdefault(c + 97, []).append((r, c))

        def fire(t: int) -> None:
            for (r, c) in cross.get(t, ()):
                for (_k, v) in kv:
                    v[r, :, c, :] = 0.0
                fired.add((r, c))

        fire(forced_pos)                   # the min entry's crossing (T_int)
    idx = torch.cat([prefix, forced_tok[:, None]], 1)
    logits = decode_step_batch(net, forced_tok, forced_pos, kv)
    n_free = T_TOTAL - 1 - forced_pos
    for s in range(n_free):
        pos = forced_pos + 1 + s
        new = torch.zeros(R, dtype=torch.long)
        for j in range(R):
            tok, _ce = sample_and_ce(logits[j], gen)
            new[j] = tok
        idx = torch.cat([idx, new[:, None]], 1)
        if pos < T_TOTAL - 1:
            if remove is not None:
                fire(pos)                  # zero BEFORE the decode at pos
            logits = decode_step_batch(net, new, pos, kv)
    return idx, kv, fired


def judge_windows(all_lg: torch.Tensor, idx: torch.Tensor, windows):
    """Clean-net CE of target windows [(lo, hi), ...] (queries lo-1..hi-1).
    Returns dict window -> (R,) mean CE per row. [VERBATIM e085/e088/e089]"""
    out = {}
    for (lo, hi) in windows:
        lg = all_lg[:, lo - 1:hi, :]
        tgt = idx[:, lo:hi + 1]
        lp = torch.log_softmax(lg.float(), -1)
        out[(lo, hi)] = -lp.gather(2, tgt[:, :, None]).squeeze(2).mean(1).numpy()
    return out


# ------------------------------------------------------------- statistics

def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    def ranks(v):
        v = np.asarray(v, float)
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), float)
        r[order] = np.arange(1, len(v) + 1, dtype=float)
        return r
    return pearson(ranks(x), ranks(y))


def cluster_boot(draws, fn, n: int = BOOT_N, seed: int = 0):
    """Cluster bootstrap over the 8 runs: resample run ids with replacement,
    rebuild the (multiply-counted) draw list, recompute fn (None if the
    resample can't support it). CI = 2.5/97.5 percentiles of valid draws.
    [VERBATIM e088/e089's cluster_boot]"""
    runs = sorted({p["run"] for p in draws})
    by_run = {r: [i for i, p in enumerate(draws) if p["run"] == r] for r in runs}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sel = []
        for r in rng.integers(0, len(runs), len(runs)):
            sel.extend(by_run[runs[r]])
        v = fn([draws[i] for i in sel])
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return [float("nan"), float("nan")], 0
    return [float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5))], len(vals)


def _fmt_ci(ci):
    return f"[{ci[0]:+.3f},{ci[1]:+.3f}]"


def _ci_overlap(a, b):
    """Two [lo, hi] intervals intersect."""
    return bool(a[0] <= b[1] and b[0] <= a[1])


def _wrap(text: str, width: int):
    import textwrap
    return textwrap.wrap(text, width=width)


# ---------------------------------------------------------------------- main


def main():
    assert torch.cuda.device_count() == 0, "CPU-only violated: CUDA visible"
    assert common.DEVICE == "cpu", f"common.DEVICE={common.DEVICE} != cpu"
    out_dir = run_dir("e097")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gates = dict(cpu_only=True, threads=THREADS, smoke=False)

    # ---- battery: rebuild corpus + net EXACTLY as e053c/e069..e089 did
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
    gates["G2_params"] = dict(params=n_params, expected=873472,
                              ok=bool(n_params == 873472))
    val_ce = estimate_loss(net, corp, "val", n_batches=12)
    gates["G1_val_ce"] = dict(val_ce=val_ce, ref=E053C_VAL_CE, tol=0.02,
                              ok=bool(abs(val_ce - E053C_VAL_CE) <= 0.02))
    log(f"e053c net loaded ({n_params:,} params) | val CE {val_ce:.4f} vs "
        f"{E053C_VAL_CE:.4f} -> G1 {'PASS' if gates['G1_val_ce']['ok'] else 'FAIL'}")

    # ---- battery: the seed-202 8-draw, ALL 8
    gen_p = torch.Generator().manual_seed(SEED_PROMPT)
    ix = torch.randint(len(corp.val) - PROMPT_TOK - 1, (N_PROMPTS,),
                       generator=gen_p)
    prompts8 = [corp.val[i:i + PROMPT_TOK] for i in ix]
    log(f"battery: {N_PROMPTS} prompts (seed {SEED_PROMPT}); prompt0 prefix: "
        f"{corp.decode(prompts8[0])[:32]!r}")

    # ---- control free run (seed-7 A-none convention)
    log("control battery: seed-7 free run (e075/e080/e085/e088/e089 A-none)")
    gen7 = torch.Generator().manual_seed(SEED_SAMPLE)
    run1 = generate_control(net, prompts8, gen7)
    idx1 = run1["idx"]
    log(f"  control run done ({T_TOTAL} positions x {B} rows)")

    # ---- G3 (scope: control battery) vs e080's stored A-none arm
    cj1 = judge_windows(manual_all_logits(net, idx1), idx1, [KEY_T])[KEY_T]
    g3 = dict(ref_file=str(E080_METRICS), ok=False,
              note="e097 has no static arm; G3 scope = control-battery "
                   "identity (clean-judge per-seq vs e080 stored none arm) "
                   "[e088/e089's scoped G3 verbatim]")
    if E080_METRICS.exists():
        import json as _json
        with open(E080_METRICS) as f:
            e080 = _json.load(f)
        ref_cj = np.asarray(e080["arms"]["none"]["clean_judge_tail_ce"]["per_seq"])
        dev_cj = float(np.abs(cj1 - ref_cj).max())
        bit = bool(np.array_equal(cj1, ref_cj))
        g3.update(clean_judge_max_dev=dev_cj, clean_judge_bitwise=bit,
                  ok=bool(dev_cj < 1e-4))
        log(f"G3 vs e080 A-none: clean-judge dev {dev_cj:.2e} (bitwise {bit})"
            f" -> {'PASS' if g3['ok'] else 'FAIL'}")
    else:
        g3["note"] += " | runs/e080/metrics.json not found; gate skipped"
        log("G3: e080 metrics missing — skipped")
    gates["G3_protocol_identity_vs_e080"] = g3

    # ---- e089 references (the uniform levels this experiment re-tests)
    e089_ref = {}
    if E089_METRICS.exists():
        import json as _json
        with open(E089_METRICS) as f:
            e089 = _json.load(f)
        for k in (K_PRIMARY, K_SECONDARY):
            pk = e089["summary"]["per_k"][str(k)]
            e089_ref[str(k)] = dict(mean=pk["mean"], ci=pk["ci"])
        log(f"e089 references: k=128 {e089_ref['128']['mean']:+.3f} "
            f"{_fmt_ci(e089_ref['128']['ci'])} | k=64 "
            f"{e089_ref['64']['mean']:+.3f} {_fmt_ci(e089_ref['64']['ci'])}")

    # ================================================== stratified sampling
    log(f"stratified sampling: rng seed {SEED_SUB}, {N_DRAWS} draws per arm "
        f"in {ARM_KEYS} (k=128 strata = 117/117 + {FILL_128} fills; k=64 "
        f"strata pure within-stratum; unif = e089 rule, reject min > {P1_MAX})")
    rng = np.random.default_rng(SEED_SUB)
    draws = []
    n_reject = 0
    for ai, (k, sname, lo, hi) in enumerate(ARMS):
        for d in range(N_DRAWS):
            if sname == "unif":
                # e089 replication arm VERBATIM (fresh stream)
                n_try = 0
                while True:
                    cols = np.sort(rng.choice(BAND, size=k, replace=False))
                    n_try += 1
                    if int(cols[0]) <= P1_MAX:
                        break
                    n_reject += 1
            elif k == K_PRIMARY:
                # maximal within-stratum concentration at fixed k=128
                strat = np.arange(lo, hi + 1)
                comp = np.setdiff1d(BAND, strat)
                extras = rng.choice(comp, size=k - len(strat), replace=False)
                cols = np.sort(np.concatenate([strat, extras]))
            else:
                # pure within-stratum at k=64
                cols = np.sort(rng.choice(np.arange(lo, hi + 1), size=k,
                                          replace=False))
            run = (ai * N_DRAWS + d) % B
            draws.append(dict(
                arm=f"k{k}_{sname}", k=k, stratum=sname, d=d, run=int(run),
                cols=[int(c) for c in cols],
                min_p=int(cols[0]), max_p=int(cols[-1]),
                mean_p=float(cols.mean()),
                t_int=int(cols[0]) + 97,
                exposure=int(T_TOTAL - 1 - (int(cols[0]) + 98) + 1),
                seed=SEED_CONT + int(cols[0]),
                n_tries=(n_try if sname == "unif" else 1),
            ))
    assert len(draws) == len(ARMS) * N_DRAWS == 80
    assert all(dr["exposure"] >= 64 for dr in draws), "tail exposure < 64"
    per_run_counts = np.bincount([dr["run"] for dr in draws], minlength=B)
    log(f"  {len(draws)} draws; uniform-arm rejections {n_reject}; draws/run "
        f"{per_run_counts.tolist()}")
    for key in ARM_KEYS:
        dd = [dr for dr in draws if dr["arm"] == key]
        log(f"  {key:>9}: min(S) in [{min(d['min_p'] for d in dd)}.."
            f"{max(d['min_p'] for d in dd)}], mean pos "
            f"{np.mean([d['mean_p'] for d in dd]):.0f}, p=414 in "
            f"{sum(414 in d['cols'] for d in dd)}/{N_DRAWS} draws")

    # ================================================== the matched continuations
    log("dynamic arm: matched control/removal continuations per group "
        "(crossing-time progressive V-zero, clean-judge tail 448..511)")
    groups: dict = {}
    for dr in draws:
        groups.setdefault(dr["min_p"], []).append(dr)

    g4_all = dict(ident_pre=True, fired_zero=True, unfired_live=True,
                  col_live=True, seed_matched=True, n_groups=0,
                  n_fired=0, n_unfired=0)
    for gi, (gkey, grp) in enumerate(sorted(groups.items())):
        grp = sorted(grp, key=lambda x: (x["arm"], x["d"]))
        rows = [dr["run"] for dr in grp]
        forced_pos = gkey + 97
        prefix = idx1[rows, :forced_pos]
        forced = idx1[rows, forced_pos]
        seed = SEED_CONT + gkey
        rm = [(j, p) for j, dr in enumerate(grp) for p in dr["cols"]]
        idx_ctl, _kv_ctl, _f0 = run_continuation(net, prefix, forced,
                                                 forced_pos, seed, remove=None)
        idx_rm, kv_rm, fired = run_continuation(net, prefix, forced,
                                                forced_pos, seed, remove=rm)
        J_ctl = judge_windows(manual_all_logits(net, idx_ctl), idx_ctl,
                              [KEY_T])[KEY_T]
        J_rm = judge_windows(manual_all_logits(net, idx_rm), idx_rm,
                             [KEY_T])[KEY_T]
        # ---- G4 identity checks for this group
        ident_pre = bool(torch.equal(idx_rm[:, :forced_pos + 1],
                                     idx_ctl[:, :forced_pos + 1]))
        col_zero, col_live, unfired_live = True, True, True
        for (r, c) in rm:
            for (_k, v) in kv_rm:
                col = v[r, :, c, :]
                if (r, c) in fired:
                    col_zero &= bool(col.abs().max().item() == 0.0)
                else:
                    unfired_live &= bool(col.abs().max().item() > 0.0)
        for (_k, v) in kv_rm:
            vals = v.abs().amax(dim=(1, 3))          # (R, n_cols) liveness
            n_kv = vals.shape[1]                     # 511 (col 511 never
            for j, dr in enumerate(grp):             # written: no decode at
                mask = np.ones(n_kv, dtype=bool)     # 511 in this window)
                mask[dr["cols"]] = False
                col_live &= bool(vals[j, torch.from_numpy(mask)].min().item() > 0.0)
        g4_all["ident_pre"] &= ident_pre
        g4_all["fired_zero"] &= col_zero
        g4_all["unfired_live"] &= unfired_live
        g4_all["col_live"] &= col_live
        g4_all["n_groups"] += 1
        g4_all["n_fired"] += len(fired)
        g4_all["n_unfired"] += len(rm) - len(fired)
        assert len(fired) == len(set(fired)), "a crossing fired twice"
        # ---- per-draw readouts
        for j, dr in enumerate(grp):
            dr["tail_control"] = float(J_ctl[j])
            dr["tail_removal"] = float(J_rm[j])
            dr["cost"] = float(J_rm[j] - J_ctl[j])
            dr["k_eff"] = sum(1 for (r, c) in rm
                              if r == j and (r, c) in fired)
            eq = torch.eq(idx_rm[j], idx_ctl[j])
            dr["stream_identical"] = bool(eq.all().item())
            nz = (~eq).nonzero().flatten()
            dr["first_div"] = int(nz[0].item()) if len(nz) else None
        if (gi + 1) % 6 == 0 or gi + 1 == len(groups):
            log(f"  groups {gi + 1}/{len(groups)} done ({elapsed():.0f}s)")
    g4_all["ok"] = bool(g4_all["ident_pre"] and g4_all["fired_zero"]
                        and g4_all["unfired_live"] and g4_all["col_live"])
    gates["G4_dynamic_instrument"] = g4_all
    log(f"G4 dynamic instrument: {g4_all['n_groups']} groups, prefix identity "
        f"{g4_all['ident_pre']}, fired cols zero {g4_all['fired_zero']} "
        f"({g4_all['n_fired']} fired), unremovable-in-window cols live "
        f"{g4_all['unfired_live']} ({g4_all['n_unfired']}), other cols live "
        f"{g4_all['col_live']} -> "
        f"{'PASS' if g4_all['ok'] else 'FAIL'}")

    # ================================================== per-arm table + bars
    arm_stats = {}
    for key in ARM_KEYS:
        dd = [dr for dr in draws if dr["arm"] == key]
        costs = [dr["cost"] for dr in dd]
        ci, _ = cluster_boot(dd, lambda gg: (float(np.mean([p["cost"] for p in gg]))
                                             if len(gg) >= 2 else None))
        arm_stats[key] = dict(
            n=len(dd), mean=float(np.mean(costs)),
            sd=float(np.std(costs, ddof=1)),
            ci=ci, draws=costs,
            mean_min=float(np.mean([d["min_p"] for d in dd])),
            mean_exposure=float(np.mean([d["exposure"] for d in dd])),
            mean_pos=float(np.mean([d["mean_p"] for d in dd])),
            k_eff_mean=float(np.mean([d["k_eff"] for d in dd])),
            stream_identical=sum(d["stream_identical"] for d in dd),
        )
        if abs(arm_stats[key]["mean"]) > 1e-9:
            arm_stats[key]["cv"] = arm_stats[key]["sd"] / abs(arm_stats[key]["mean"])
        else:
            arm_stats[key]["cv"] = float("nan")

    def _ratio_fn(strat_key, unif_key):
        def fn(gg):
            den = np.mean([p["cost"] for p in gg if p["arm"] == unif_key])
            num = np.mean([p["cost"] for p in gg if p["arm"] == strat_key])
            if len([p for p in gg if p["arm"] == unif_key]) < 2:
                return None
            if den <= RATIO_DEN_FLOOR:      # registered denominator guard
                return None
            return float(num / den)
        return fn

    ratios = {}   # k -> {stratum: {val, ci, valid}}
    for k in (K_PRIMARY, K_SECONDARY):
        ratios[k] = {}
        for (sname, _lo, _hi) in STRATA:
            ci, nvalid = cluster_boot(
                draws, _ratio_fn(f"k{k}_{sname}", f"k{k}_unif"))
            m_s = arm_stats[f"k{k}_{sname}"]["mean"]
            m_u = arm_stats[f"k{k}_unif"]["mean"]
            ratios[k][sname] = dict(
                val=float(m_s / m_u) if abs(m_u) > 1e-9 else float("nan"),
                ci=ci, n_valid_boot=nvalid)

    # descriptive texture: cost vs subset mean position
    sp_all = spearman([dr["cost"] for dr in draws],
                      [dr["mean_p"] for dr in draws])
    dd128 = [dr for dr in draws if dr["k"] == K_PRIMARY]
    sp_128 = spearman([dr["cost"] for dr in dd128],
                      [dr["mean_p"] for dr in dd128])

    # ---- REGISTERED bars (frozen; evaluated in priority order)
    def _eval_bars(k):
        m_u = arm_stats[f"k{k}_unif"]["mean"]
        ci_u = arm_stats[f"k{k}_unif"]["ci"]
        if not (m_u > 0):
            return "INDETERMINATE", (f"uniform(k={k}) mean {m_u:+.4f} <= 0 — "
                                     f"no ratio is defined"), {}
        ev = {}
        asym, asym_who = [], []
        for (sname, _lo, _hi) in STRATA:
            key = f"k{k}_{sname}"
            r = ratios[k][sname]
            ci_s = arm_stats[key]["ci"]
            m_s = arm_stats[key]["mean"]
            fires = (r["val"] >= ASYM_X and r["ci"][0] > 1.0
                     and r["n_valid_boot"] > 0)
            within = (SURV_LO <= r["val"] <= SURV_HI)
            ov = _ci_overlap(ci_s, ci_u)
            ev[sname] = dict(ratio=r["val"], ratio_ci=r["ci"],
                             ratio_valid_boot=r["n_valid_boot"],
                             asym_fires=bool(fires), within_25=bool(within),
                             ci_overlap=bool(ov))
            if fires:
                asym.append(sname)
                asym_who.append(
                    f"{sname}: {arm_stats[key]['mean']:+.3f} = "
                    f"{r['val']:.2f}x uniform {m_u:+.3f}, ratio CI "
                    f"{_fmt_ci(r['ci'])} excludes 1")
        if asym:
            return ("POSITION ASYMMETRY",
                    f"k={k}: " + "; ".join(asym_who) +
                    f" — the sink-adjacent side carries disproportionate "
                    f"mass; the which-irrelevance claim gets its boundary "
                    f"(echo of the KV-eviction position priors).", ev)
        if all(ev[s]["within_25"] and ev[s]["ci_overlap"] for s in ev):
            return ("WHICH-IRRELEVANCE SURVIVES",
                    f"k={k}: all three strata within +-25% of the uniform "
                    f"control with overlapping CIs — "
                    + "; ".join(
                        f"{s}: {arm_stats[f'k{k}_{s}']['mean']:+.3f} = "
                        f"{ev[s]['ratio']:.2f}x, CI "
                        f"{_fmt_ci(arm_stats[f'k{k}_{s}']['ci'])} vs uniform "
                        f"CI {_fmt_ci(ci_u)}" for s in ev)
                    + f". The hedge closes; the mass-action claim stands "
                      f"un-hedged under position stratification.", ev)
        viol = [f"{s}: {arm_stats[f'k{k}_{s}']['mean']:+.3f} = "
                f"{ev[s]['ratio']:.2f}x (want 0.75..1.25), CI "
                f"{_fmt_ci(arm_stats[f'k{k}_{s}']['ci'])}, overlap "
                f"{ev[s]['ci_overlap']}" for s in ev
                if not (ev[s]["within_25"] and ev[s]["ci_overlap"])]
        return ("MIXED/TEXTURE",
                f"k={k}: no asymmetry bar fired (no stratum >= {ASYM_X}x "
                f"with CI excluding 1) but not all strata sit within +-25% "
                f"with overlapping CIs: " + "; ".join(viol)
                + f". Uniform control {m_u:+.3f} {_fmt_ci(ci_u)}.", ev)

    clause128, verdict128, ev128 = _eval_bars(K_PRIMARY)
    clause64, verdict64, ev64 = _eval_bars(K_SECONDARY)
    log(f"table: " + " | ".join(
        f"{key} {arm_stats[key]['mean']:+.3f}" for key in ARM_KEYS))
    log(f"ratios k=128: " + " | ".join(
        f"{s} {ratios[128][s]['val']:.2f} {_fmt_ci(ratios[128][s]['ci'])}"
        for s, _lo, _hi in STRATA))
    log(f"ratios k=64:  " + " | ".join(
        f"{s} {ratios[64][s]['val']:.2f} {_fmt_ci(ratios[64][s]['ci'])}"
        for s, _lo, _hi in STRATA))
    log(f"texture: Spearman(cost, mean pos) all {sp_all:+.3f} | k=128 only "
        f"{sp_128:+.3f}")
    log(f"PRIMARY DECISION (k=128) [{clause128}]: {verdict128}")
    log(f"SECONDARY (k=64) [{clause64}]: {verdict64}")

    # ---------------------------------------------------------------- metrics
    metrics = dict(
        experiment="e097_sink_asymmetry",
        purpose="T051's registered sink-asymmetry hedge (the massaction_key_lit "
                "novelty verdict): position-stratified subset removal at fixed "
                "k inside the anchor band — oldest/middle/newest thirds vs the "
                "uniform e089 replication — with the e089 dynamic rig VERBATIM "
                "(crossing-time progressive V-zero, matched-stream "
                "continuation, clean-net final-64 tail judgment). Tests "
                "whether the e089 variance-collapse which-irrelevance survives "
                "position stratification, or sink-adjacent old entries carry "
                "disproportionate mass.",
        started=started, wall_s=elapsed(), threads=THREADS, cpu_only=True,
        net=dict(ckpt=str(CKPT), arch=dict(n_layer=4, n_head=4, n_embd=128,
                                           block_size=T_TOTAL, vocab=65),
                 params=n_params, val_ce=val_ce, val_ce_e053c=E053C_VAL_CE),
        seeds=dict(corpus=1337, prompts=SEED_PROMPT, sampling=SEED_SAMPLE,
                   subset_selection=SEED_SUB, continuation=SEED_CONT,
                   bootstrap=0),
        protocol=dict(
            B=B, prompt_tokens=PROMPT_TOK, t_total=T_TOTAL, temp=TEMP,
            topk=TOPK, tail_window=list(KEY_T), boot_n=BOOT_N,
            threads=THREADS,
            anchor_band=list(ANCHOR_POS), band_n=int(len(BAND)),
            strata={s: [lo, hi] for (s, lo, hi) in STRATA},
            ks=dict(primary=K_PRIMARY, secondary=K_SECONDARY),
            n_draws_per_arm=N_DRAWS, arm_order=ARM_KEYS,
            sampling_rule=(
                "rng default_rng(97), arm-major (k128 old/mid/new/unif, then "
                "k64 old/mid/new/unif) draw-minor. k=128 strata: ALL 117 "
                "stratum positions + 11 uniform complement fills (maximal "
                "concentration at fixed k=128 — pure within-stratum is "
                "combinatorially impossible, 117 < 128; REGISTERED "
                "operationalization). k=64 strata: 64 pure within-stratum "
                "draws. uniform: e089 VERBATIM — choice(band, k, "
                "replace=False), reject & redraw while min > 350 "
                f"(rejections {n_reject})."),
            run_rule="run = (arm_index*10 + draw) % 8 — 10 draws per run "
                     "overall, >= 1 draw of every arm per run",
            timing=dict(
                schedule="per-entry age-97-crossing V-zero (e089 VERBATIM, "
                         "no adaptations)",
                t_int="min(S) + 97 (the earliest entry's crossing = the "
                      "forced decode)",
                exposure="414 - min(S) (>= 64 everywhere; stratum arms by "
                         "construction, uniform arms by the min<=350 floor)",
                unremovable="p = 414 crosses at 511, beyond the last decode "
                            "(510): never zeroed, stays live, k_eff = k-1 "
                            "(the k=128 'new' arm contains 414 "
                            "deterministically)",
                confound="position strata mechanically couple with removal "
                         "timing (older strata cross earlier — longer causal "
                         "leash): intrinsic to the asymmetry question, "
                         "reported as per-arm exposure texture, not "
                         "controlled away"),
            arms="2 per group, SAME continuation seed (matched streams): "
                 "control / crossing-time progressive removal of the subset",
            outcome="clean-judge tail CE (448..511) of removal stream minus "
                    "matched control stream; arm MEAN over 10 draws",
            bars=dict(
                indeterminate="uniform(k=128) mean <= 0",
                asymmetry=f"any k=128 stratum ratio >= {ASYM_X}x uniform "
                          f"with ratio cluster-CI excluding 1 (lower > 1; "
                          f"bootstrap resamples with uniform mean <= "
                          f"{RATIO_DEN_FLOOR} nats discarded as "
                          f"denominator-unstable)",
                survives=f"ALL three k=128 strata: ratio in "
                         f"[{SURV_LO}, {SURV_HI}] (+-25%) AND stratum mean-CI "
                         f"overlapping the uniform mean-CI",
                order="indeterminate -> asymmetry -> survives -> mixed/texture",
                secondary="same clauses at k=64, report-only"),
            registered_numbers=dict(surv_lo=SURV_LO, surv_hi=SURV_HI,
                                    asym_x=ASYM_X,
                                    ratio_den_floor=RATIO_DEN_FLOOR),
        ),
        gates=gates,
        control_run=dict(clean_judge_tail_ce=cj1.tolist()),
        e089_reference=dict(file=str(E089_METRICS), per_k=e089_ref,
                            note="e089's uniform-subset levels (seed 89) — "
                                 "the k=128 uniform control here is a fresh "
                                 "replication (seed 97)"),
        summary=dict(
            n_draws=len(draws), n_groups=len(groups),
            uniform_rejections=n_reject,
            per_arm={key: {kk: vv for kk, vv in st.items() if kk != "draws"}
                     for key, st in arm_stats.items()},
            per_arm_draws={key: st["draws"] for key, st in arm_stats.items()},
            ratios={f"k{k}": {s: ratios[k][s] for s in ratios[k]}
                    for k in (K_PRIMARY, K_SECONDARY)},
            texture=dict(spearman_cost_meanpos_all=sp_all,
                         spearman_cost_meanpos_k128=sp_128,
                         stream_identical_by_arm={
                             key: arm_stats[key]["stream_identical"]
                             for key in ARM_KEYS}),
        ),
        draws=[{k2: v for k2, v in dr.items() if k2 != "cols"} | dict(
            n_cols=len(dr["cols"])) for dr in draws],
        subsets={f"{dr['arm']}_d{dr['d']}": dr["cols"] for dr in draws},
        registered_decision=dict(
            primary_k=K_PRIMARY, clause=clause128, verdict=verdict128,
            evidence=ev128,
            secondary=dict(k=K_SECONDARY, clause=clause64, verdict=verdict64,
                           evidence=ev64)),
    )
    save_json(out_dir / "metrics.json", metrics)
    log("metrics.json written")

    plot(out_dir / "sink_asymmetry.png", metrics)
    log(f"plot written; total wall {elapsed():.0f}s")


# ---------------------------------------------------------------------- plot


def plot(path: Path, M: dict):
    dec = M["registered_decision"]
    S = M["summary"]
    per_arm = S["per_arm"]
    per_draws = S["per_arm_draws"]

    fig, axes = plt.subplots(2, 2, figsize=(20, 11))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]
    strat_names = [s for (s, _lo, _hi) in STRATA]
    strat_lab = ["old\n(64..180, sink side)", "mid\n(181..297)",
                 "new\n(298..414, head side)"]
    strat_col = dict(old="tab:red", mid="tab:olive", new="tab:blue")

    for ax, k, ttl, tag in ((ax1, K_PRIMARY, "PRIMARY", "E097-1"),
                            (ax2, K_SECONDARY, "SECONDARY", "E097-2")):
        mu = per_arm[f"k{k}_unif"]["mean"]
        cu = per_arm[f"k{k}_unif"]["ci"]
        xs = np.arange(3)
        for x, s in zip(xs, strat_names):
            st = per_arm[f"k{k}_{s}"]
            ax.bar(x, st["mean"], width=0.55, color=strat_col[s], alpha=0.75,
                   zorder=2)
            ax.errorbar(x, st["mean"],
                        yerr=[[max(0.0, st["mean"] - st["ci"][0])],
                              [max(0.0, st["ci"][1] - st["mean"])]],
                        fmt="k_", capsize=6, lw=1.6, ms=12, zorder=4)
            # per-draw dots, jittered
            ys = per_draws[f"k{k}_{s}"]
            ax.scatter(np.full(len(ys), x) + np.linspace(-0.18, 0.18, len(ys)),
                       ys, s=26, color="k", alpha=0.45, zorder=3)
            r = S["ratios"][f"k{k}"][s]["val"]
            ax.text(x, st["mean"], f"  {r:.2f}x", fontsize=10, va="bottom",
                    ha="left", weight="bold", zorder=5)
        # the uniform band overlaid
        ax.axhspan(cu[0], cu[1], color="tab:gray", alpha=0.30, zorder=1,
                   label=f"uniform control band {mu:+.3f} "
                         f"{_fmt_ci(cu)}")
        ax.axhline(mu, color="tab:gray", lw=1.8, ls="-", zorder=1)
        ax.axhline(SURV_LO * mu, color="tab:green", ls=":", lw=1.5,
                   label=f"survives band {SURV_LO}..{SURV_HI}x uniform")
        ax.axhline(SURV_HI * mu, color="tab:green", ls=":", lw=1.5)
        ax.axhline(ASYM_X * mu, color="tab:red", ls="--", lw=1.5,
                   label=f"asymmetry bar {ASYM_X}x uniform")
        ax.axhline(0, color="k", lw=0.6)
        # uniform arm as its own bar, offset
        ax.bar(3.35, mu, width=0.5, color="tab:gray", alpha=0.9, zorder=2,
               hatch="//")
        ax.errorbar(3.35, mu, yerr=[[max(0.0, mu - cu[0])],
                                    [max(0.0, cu[1] - mu)]],
                    fmt="k_", capsize=6, lw=1.6, ms=12, zorder=4)
        ax.set_xticks(list(xs) + [3.35])
        ax.set_xticklabels(strat_lab + ["uniform\n(e089 replication)"],
                           fontsize=9)
        ax.set_ylabel("clean-judge tail cost (nats)")
        e089m = M["e089_reference"]["per_k"][str(k)]["mean"]
        ax.set_title(f"{tag} — {ttl} k={k}: per-stratum cost bars with the "
                     f"uniform band overlaid (e089 k={k} level "
                     f"{e089m:+.3f})", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")

    # ---- panel 3: per-draw cost vs subset mean position
    cols = {s: strat_col[s] for s in strat_names}
    cols["unif"] = "tab:gray"
    for k, mk, al in ((K_PRIMARY, "o", 0.9), (K_SECONDARY, "s", 0.6)):
        for s in strat_names + ["unif"]:
            dd = [dr for dr in M["draws"]
                  if dr["k"] == k and dr["stratum"] == s]
            ax3.scatter([d["mean_p"] for d in dd], [d["cost"] for d in dd],
                        s=34, color=cols[s], marker=mk, alpha=al,
                        label=f"k={k} {s}")
    ax3.axvline(64 + 117 / 2, color="k", ls=":", lw=0.8)
    ax3.axvline(181 + 117 / 2, color="k", ls=":", lw=0.8)
    ax3.axvline(298 + 117 / 2, color="k", ls=":", lw=0.8)
    ax3.text(122, ax3.get_ylim()[1] * 0.95, "old third", fontsize=8, ha="center")
    ax3.text(239, ax3.get_ylim()[1] * 0.95, "mid third", fontsize=8, ha="center")
    ax3.text(356, ax3.get_ylim()[1] * 0.95, "new third", fontsize=8, ha="center")
    ax3.axhline(0, color="k", lw=0.6)
    ax3.set_xlabel("subset mean position (old=sink side, new=head side)")
    ax3.set_ylabel("per-draw cost (nats)")
    ax3.set_title("E097-3 — per-draw cost vs subset mean position | Spearman "
                  f"all {S['texture']['spearman_cost_meanpos_all']:+.3f}, "
                  f"k=128 {S['texture']['spearman_cost_meanpos_k128']:+.3f} "
                  "(descriptive; exposure texture: older strata cross earlier)",
                  fontsize=9.5)
    ax3.legend(fontsize=7, ncol=2)

    # ---- panel 4: the registered decision
    ax4.axis("off")
    lines = [
        "REGISTERED (frozen in T051/the tasking):",
        f"  ASYMMETRY: any k=128 stratum >= {ASYM_X}x uniform, ratio CI excl. 1",
        f"  SURVIVES: all k=128 strata in [{SURV_LO}, {SURV_HI}]x uniform, "
        f"CIs overlap",
        "  indeterminate if uniform(k=128) <= 0; else mixed/texture",
        f"  secondary: same at k={K_SECONDARY} (report-only)",
        "",
        f"STRATUM TABLE (mean cost, cluster CI, ratio vs uniform):",
    ]
    for k in (K_PRIMARY, K_SECONDARY):
        mu = per_arm[f"k{k}_unif"]["mean"]
        lines.append(f"  k={k}: uniform {mu:+.3f} "
                     f"{_fmt_ci(per_arm[f'k{k}_unif']['ci'])}")
        for s in strat_names:
            st = per_arm[f"k{k}_{s}"]
            r = S["ratios"][f"k{k}"][s]
            lines.append(f"    {s:>4}: {st['mean']:+.3f} "
                         f"{_fmt_ci(st['ci'])} | {r['val']:.2f}x "
                         f"{_fmt_ci(r['ci'])}")
    lines += [
        "",
        f"PRIMARY DECISION (k={K_PRIMARY}) [{dec['clause']}]:",
    ] + [f"  {wd}" for wd in _wrap(dec["verdict"], 92)]
    lines += [
        f"SECONDARY (k={K_SECONDARY}) [{dec['secondary']['clause']}]:",
    ] + [f"  {wd}" for wd in _wrap(dec["secondary"]["verdict"], 92)]
    ax4.text(0.02, 0.97, "E097 — T051 sink-asymmetry hedge: position "
                         "stratification of the anchor mass",
             fontsize=13, weight="bold", va="top")
    for i, tx in enumerate(lines):
        ax4.text(0.02, 0.93 - i * 0.0285, tx, fontsize=8.2, va="top",
                 family="monospace")

    fig.suptitle("E097 — sink-position asymmetry inside the anchor band at "
                 f"fixed k | primary clause: {dec['clause']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
