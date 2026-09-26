"""E089 — T051's registered MASS-RESPONSE curve: removal cost vs NUMBER of
anchor entries removed (P3's capstone measurement — the anchor's dose-response
law). CPU-only.

T051's E088 CLOSE left the anchor as MASS-ACTION: no critical entry (e085
singles near-null, mean +0.0006), no critical pair (e088 sub-additive, median
ratio 0.464, CI [0.024, 0.500]), yet whole-band removal costs +0.26 (e075
A-self-prune). The registered next measurement (T051, frozen): the
mass-response curve.

DESIGN (registered here BEFORE compute; bars frozen from T051/the tasking).
Battery: the e053c ctx-512 net (873,472 params, val CE 1.5227), the seed-202
8-draw prompts, B=8, the seed-7 control free run (e075/e080/e085/e088 A-none
convention; G3 gated against e080's stored none arm). Anchor band: positions
64..414 = final-frame ages 97..447 (self-generated), verbatim.

DRAWS: k in {2, 4, 8, 16, 32, 64, 128, 200}, 10 independent draws per k
(80 draws). RNG (documented): numpy default_rng(seed=89), consumed in
(k-major, draw-minor) order; each draw = rng.choice(band_64_414, size=k,
replace=False); REJECTED and redrawn (same rng stream) while min(cols) > 350
(the e088 P1_MAX full-tail-exposure floor: first affected sample min+98 must
stay <= 448 so the judged tail 448..511 is fully free-sampled after removal;
rejection tallied). Run assignment (documented): run = (k_index*10 + draw)
% 8 — every run hosts exactly 10 draws overall and >= 1 draw of every k.

INSTRUMENT (the e085/e088 dynamic-removal rig: matched-stream continuation +
clean-net final-64 tail judgment) with ONE registered, documented adaptation.
e088 zeroed all removal columns in one shot at T_int = min+97 — legal only
because its pair distance d <= 96 kept both columns inside the prefix. For
subsets of size k spread over the whole band, entries at positions >= min+97
do NOT exist in the KV cache at T_int, so single-shot removal is infeasible.
Adaptation (e085's per-entry timing VERBATIM, applied to sets): each entry p
is V-zeroed immediately BEFORE the decode step that processes position p+97
("removal before the decode that processes the token at position p+97" —
e085's age-97-crossing timing; e075's progressive prune schedule at K=1
continuous instead of K=32). The earliest entry's crossing IS the forced
decode: T_int = min(S)+97, the unaffected token at T_int is teacher-forced,
first affected sample = min+98, exposure = 414 - min >= 64. Consequence:
the k=200 arm is the direct fine-schedule analogue of e075's A-self-prune
(the +0.26 reference). Edge case (honest, tallied): entry p=414 crosses at
position 511, beyond the last decode (510) — it can never be zeroed inside
the window, stays live, and contributes nothing (exposure 0 by e085's
formula); draws containing it simply carry k-1 effective removals.
Matched control: same prefix, same continuation seed (SEED_CONT=891000 +
min(S), the e088 convention), same decode loop, no zeroing — streams match
the removal arm until sampled divergence.

READOUT per draw: cost = clean-judge tail CE (448..511, clean net, no
pruning) of the removal stream minus the matched control stream.

REGISTERED BARS (frozen in T051/the tasking; evaluated on per-k MEANS over
the 10 draws, cluster bootstrap CIs over the 8 runs reported alongside):
  1. mean(200) <= 0 -> INDETERMINATE (instrument failure; no ratio defined).
  2. MASS-ACTION (threshold): mean(32) < 0.25 * mean(200) AND
     mean(128) > 3 * mean(64). Near-flat at small k, breaking sharply
     upward toward the e075 whole-band level as k approaches the band size.
  3. LINEAR/diffuse: |mean(64)/mean(200) - 0.32| <= 0.10 (the documented
     operationalization of "cost(k=64) ~ 32% of cost(k=200)"; 64/200 = 0.32)
     AND no sharp break (mean(128) <= 3 * mean(64)). Cost proportional to k
     throughout — diffuse independent contributions (the null view).
  4. Anything else -> honest texture (per-k means with cluster CIs).
  Clauses 2 and 3 are mutually exclusive by construction (the break clause).

SECONDARY (registered): variance across draws per k — SD and CV; mass-action
predicts variance COLLAPSES once k is past the break (the threshold depends
on HOW MANY entries, not WHICH). Texture riders: per-draw cost vs min(S)
(exposure confound visibility) and first stream-divergence position.

Gates: G1 val CE vs e053c 1.5227 +-0.02; G2 params 873472; G3 control
battery identity vs e080's stored none arm (clean-judge per-seq, hard bar
1e-4, bitwise flag; e088's scoped G3 — e089 likewise has no static arm);
G4 dynamic-instrument identity per group (prefix identity through the
teacher-forced token; fired removal columns exactly 0; non-removed and
control columns live; matched seed across arms; each crossing fired at most
once).

Run:     python lab/e089_mass_response.py
Outputs: runs/e089/metrics.json + runs/e089/mass_response.png
Envelope: NO training, NO new automations; CPU-only (CUDA_VISIBLE_DEVICES=-1),
8 threads (T050), single step, target ~10-20 min.
No NOTES/THINKING/QUEUE/STATE edits; no commit (the dispatcher owns those).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # task spec: CPU-only (pre-torch)
import torch

# this torch build reports is_available()==True even with an empty
# CUDA_VISIBLE_DEVICES; force CPU so common.DEVICE=="cpu" (e053b..e088)
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
SEED_PROMPT, SEED_SAMPLE = 202, 7         # e053b/e053c/e069..e088 seeds
SEED_SUB = 89                             # e089: subset-sampling rng
SEED_CONT = 891000                        # e089: continuation seeds (+ min)
B = 8
N_PROMPTS = 8
BOOT_N = 1000
THREADS = 8                               # T050: 12 spin-thrashes the box
torch.set_num_threads(THREADS)

P512 = T_TOTAL - 1                        # 511
TAIL = 64                                 # the e075/e080/e085/e088 tail window
KEY_T = (T_TOTAL - TAIL, T_TOTAL - 1)     # judged tail window (448, 511)
G = T_TOTAL - PROMPT_TOK                  # 448 generation steps

# ---- the anchor band (final-frame ages; age of position p = 511 - p) ---------
ANCHOR_POS = (64, 414)                    # ages 97..447 (self-generated)
P1_MAX = 350                              # full-tail-exposure: min+98 <= 448
KS = [2, 4, 8, 16, 32, 64, 128, 200]      # registered subset sizes
N_DRAWS = 10                              # independent draws per k
BAND = np.arange(ANCHOR_POS[0], ANCHOR_POS[1] + 1)   # 351 positions
LAST_DECODE = T_TOTAL - 2                 # 510: last decode step position

# ---- REGISTERED decision numbers (frozen, docstring verbatim) ----------------
LIN_TOL = 0.10                            # |ratio - 0.32| window ("~ 32%")
MASS_LOW = 0.25                           # mean(32) < 25% of mean(200)
MASS_BREAK = 3.0                          # mean(128) > 3 x mean(64)
LIN_FRAC = 0.32                           # 64/200

# ---- reference numbers (protocol-identity gates) -----------------------------
CKPT = REPO / "runs" / "checkpoints" / "e053c_ctx512.pt"
E053C_VAL_CE = 1.5226792494455974         # the net being interrogated
E080_METRICS = REPO / "runs" / "e080" / "metrics.json"
E075_METRICS = REPO / "runs" / "e075" / "metrics.json"
T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# ------------------------------------------------- machinery (VERBATIM e088)

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
    [VERBATIM e075/e080/e085/e088]"""
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
    [VERBATIM e075/e085/e088]"""
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
    """Free-run 64->512 for B sequences (e075/e080/e085/e088 A-none VERBATIM
    stream math)."""
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
    control when remove=None). [e088's rig with ONE registered adaptation,
    documented in the module docstring:]

    remove = list of (row, col): every listed entry p is V-zeroed immediately
    BEFORE the decode step that processes position p+97 (e085's age-97-crossing
    timing VERBATIM, applied to sets). The earliest entry's crossing is the
    forced decode itself (T_int = min+97; e075/e085/e088 'top of the event'
    timing), so the token at forced_pos is teacher-forced and the first
    affected sample is forced_pos+1. Entries whose crossing falls at position
    511 (p = 414 — beyond the last decode at 510) are NEVER zeroed inside the
    window: they are unremovable-by-construction (exposure 0) and are returned
    unfired for honest tallying. Free-run from forced_pos+1 with the shared
    row-order generator (e053b stream math) — control and removal arms share
    the seed, so streams are row-by-row matched until sampled divergence.

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
    Returns dict window -> (R,) mean CE per row. [VERBATIM e085/e088]"""
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
    [VERBATIM e088's cluster_boot]"""
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


def _wrap(text: str, width: int):
    import textwrap
    return textwrap.wrap(text, width=width)


# ---------------------------------------------------------------------- main


def main():
    assert torch.cuda.device_count() == 0, "CPU-only violated: CUDA visible"
    assert common.DEVICE == "cpu", f"common.DEVICE={common.DEVICE} != cpu"
    out_dir = run_dir("e089")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gates = dict(cpu_only=True, threads=THREADS, smoke=False)

    # ---- battery: rebuild corpus + net EXACTLY as e053c/e069..e088 did
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
    log("control battery: seed-7 free run (e075/e080/e085/e088 A-none)")
    gen7 = torch.Generator().manual_seed(SEED_SAMPLE)
    run1 = generate_control(net, prompts8, gen7)
    idx1 = run1["idx"]
    log(f"  control run done ({T_TOTAL} positions x {B} rows)")

    # ---- G3 (scope: control battery) vs e080's stored A-none arm
    cj1 = judge_windows(manual_all_logits(net, idx1), idx1, [KEY_T])[KEY_T]
    g3 = dict(ref_file=str(E080_METRICS), ok=False,
              note="e089 has no static arm; G3 scope = control-battery "
                   "identity (clean-judge per-seq vs e080 stored none arm) "
                   "[e088's scoped G3 verbatim]")
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

    # ---- e075 whole-band reference (the +0.26 the curve should approach)
    e075_ref = None
    if E075_METRICS.exists():
        import json as _json
        with open(E075_METRICS) as f:
            e075 = _json.load(f)
        r1 = e075["registered_decision"]["r1_tail_clean_ce"]
        e075_ref = float(r1["self"] - r1["none"])
        log(f"e075 reference loaded: whole-band (K=32 schedule) self-minus-none "
            f"= {e075_ref:+.4f} nats")

    # ================================================== subset sampling (seed 89)
    log(f"subset sampling: rng seed {SEED_SUB}, {N_DRAWS} draws per k in "
        f"{KS} from band {ANCHOR_POS} (reject while min > {P1_MAX})")
    rng = np.random.default_rng(SEED_SUB)
    draws = []
    n_reject = 0
    for ki, k in enumerate(KS):
        for d in range(N_DRAWS):
            n_try = 0
            while True:
                cols = np.sort(rng.choice(BAND, size=k, replace=False))
                n_try += 1
                if int(cols[0]) <= P1_MAX:
                    break
                n_reject += 1
            run = (ki * N_DRAWS + d) % B
            draws.append(dict(
                k=k, d=d, run=int(run),
                cols=[int(c) for c in cols],
                min_p=int(cols[0]), max_p=int(cols[-1]),
                mean_p=float(cols.mean()),
                t_int=int(cols[0]) + 97,
                exposure=int(T_TOTAL - 1 - (int(cols[0]) + 98) + 1),
                seed=SEED_CONT + int(cols[0]),
                n_tries=n_try,
            ))
    assert len(draws) == len(KS) * N_DRAWS == 80
    assert all(dr["exposure"] >= 64 for dr in draws)
    per_run_counts = np.bincount([dr["run"] for dr in draws], minlength=B)
    log(f"  {len(draws)} draws; rejections {n_reject}; draws/run "
        f"{per_run_counts.tolist()}; min(S) range "
        f"{min(dr['min_p'] for dr in draws)}..{max(dr['min_p'] for dr in draws)}")
    for k in KS:
        ms = [dr["min_p"] for dr in draws if dr["k"] == k]
        log(f"  k={k:>3}: min(S) in [{min(ms)}..{max(ms)}], mean "
            f"{np.mean(ms):.0f}; contains p=414 in "
            f"{sum(414 in dr['cols'] for dr in draws if dr['k'] == k)}/"
            f"{N_DRAWS} draws")

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
        grp = sorted(grp, key=lambda x: (x["k"], x["d"]))
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
            eq = torch.eq(idx_rm[j], idx_ctl[j])
            dr["stream_identical"] = bool(eq.all().item())
            nz = (~eq).nonzero().flatten()
            dr["first_div"] = int(nz[0].item()) if len(nz) else None
            dr["n_fired_unremovable"] = sum(
                1 for (r, c) in rm if r == j and (r, c) not in fired)
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

    # ================================================== per-k curve + bars
    means, sds, cvs, cis, per_k_draws = {}, {}, {}, {}, {}
    for k in KS:
        costs = [dr["cost"] for dr in draws if dr["k"] == k]
        per_k_draws[k] = costs
        means[k] = float(np.mean(costs))
        sds[k] = float(np.std(costs, ddof=1))
        cvs[k] = float(sds[k] / abs(means[k])) if abs(means[k]) > 1e-9 else float("nan")
        cis[k], _ = cluster_boot(
            [dr for dr in draws if dr["k"] == k],
            lambda gg: (float(np.mean([p["cost"] for p in gg]))
                        if len(gg) >= 2 else None))
    m32, m64, m128, m200 = means[32], means[64], means[128], means[200]

    def _ratio_fn(num_k, den_k):
        def fn(gg):
            den = np.mean([p["cost"] for p in gg if p["k"] == den_k])
            num = np.mean([p["cost"] for p in gg if p["k"] == num_k])
            if len([p for p in gg if p["k"] == den_k]) < 2 or abs(den) < 1e-9:
                return None
            return float(num / den)
        return fn

    r32_200 = m32 / m200 if abs(m200) > 1e-9 else float("nan")
    r64_200 = m64 / m200 if abs(m200) > 1e-9 else float("nan")
    r128_64 = m128 / m64 if abs(m64) > 1e-9 else float("nan")
    ci_r32, _ = cluster_boot(draws, _ratio_fn(32, 200))
    ci_r64, _ = cluster_boot(draws, _ratio_fn(64, 200))
    ci_r128, _ = cluster_boot(draws, _ratio_fn(128, 64))

    # variance-collapse secondary: SD(k=128)/SD(k=8), SD(k=200)/SD(k=8)
    sd_ratio_128_8 = sds[128] / sds[8] if sds[8] > 1e-12 else float("nan")
    sd_ratio_200_8 = sds[200] / sds[8] if sds[8] > 1e-12 else float("nan")
    # exposure texture (descriptive, NOT a registered bar)
    sp_min = spearman([dr["cost"] for dr in draws],
                      [dr["min_p"] for dr in draws])

    # ---- REGISTERED bars (frozen; evaluated in order)
    if not (m200 > 0):
        clause = "INDETERMINATE (mean k=200 cost <= 0)"
        verdict = (f"mean cost at k=200 is {m200:+.4f} nats — the whole-mass "
                   f"anchor of the curve failed to separate from 0, so no "
                   f"ratio bar is defined. Honest texture: per-k means "
                   + ", ".join(f"{k}:{means[k]:+.3f}" for k in KS) + ".")
    else:
        mass_break = r128_64 > MASS_BREAK
        mass_low = r32_200 < MASS_LOW
        lin_frac = abs(r64_200 - LIN_FRAC) <= LIN_TOL
        if mass_low and mass_break:
            clause = "MASS-ACTION (threshold-shaped rise)"
            verdict = (f"Both threshold clauses fire: cost(k=32) = "
                       f"{r32_200:.3f} of cost(k=200) (< {MASS_LOW}) AND "
                       f"cost(k=128) = {r128_64:.2f}x cost(k=64) "
                       f"(> {MASS_BREAK}x). The curve is near-flat at small k "
                       f"and breaks sharply upward between k=64 and k=128 "
                       f"toward the k=200 level {m200:+.3f} (e075 whole-band "
                       f"reference {e075_ref:+.3f} on its K=32 schedule). "
                       f"Removal cost is a function of MASS, not of any "
                       f"identifiable entries — P3's capstone: the anchor's "
                       f"dose-response law is threshold-shaped. "
                       f"Variance collapse: SD(128)/SD(8) = "
                       f"{sd_ratio_128_8:.2f}, SD(200)/SD(8) = "
                       f"{sd_ratio_200_8:.2f}.")
        elif lin_frac and not mass_break:
            clause = "LINEAR/DIFFUSE (cost proportional to k)"
            verdict = (f"The linear clauses fire: cost(k=64)/cost(k=200) = "
                       f"{r64_200:.3f} (within {LIN_TOL} of {LIN_FRAC}) with "
                       f"no sharp break (cost(128)/cost(64) = {r128_64:.2f} "
                       f"<= {MASS_BREAK}). Removal cost grows in proportion to "
                       f"the number removed — diffuse independent "
                       f"contributions, the null view; mass-action is "
                       f"refuted at this resolution.")
        else:
            clause = "TEXTURE (no registered shape fires cleanly)"
            verdict = (f"Ratios: cost(32)/cost(200) = {r32_200:.3f} "
                       f"(mass-action wants < {MASS_LOW}); cost(128)/cost(64) "
                       f"= {r128_64:.2f} (wants > {MASS_BREAK}); "
                       f"cost(64)/cost(200) = {r64_200:.3f} (linear wants "
                       f"{LIN_FRAC}+-{LIN_TOL}). Neither registered shape "
                       f"fires; the honest curve is the per-k means with "
                       f"cluster CIs — "
                       + "; ".join(f"k={k}: {means[k]:+.3f} "
                                   f"{_fmt_ci(cis[k])}" for k in KS)
                       + f". k=200 level {m200:+.3f} vs e075 whole-band "
                         f"{e075_ref:+.3f} (K=32 schedule).")
    log("curve: " + " | ".join(f"k={k} {means[k]:+.3f}" for k in KS))
    log(f"ratios: 32/200 {r32_200:+.3f} CI {_fmt_ci(ci_r32)} | 64/200 "
        f"{r64_200:+.3f} CI {_fmt_ci(ci_r64)} | 128/64 {r128_64:+.2f} CI "
        f"{_fmt_ci(ci_r128)}")
    log(f"secondary: SD/8={sds[8]:.3f} SD/64={sds[64]:.3f} SD/128={sds[128]:.3f} "
        f"SD/200={sds[200]:.3f} | Spearman(cost, min) {sp_min:+.3f}")
    log(f"REGISTERED DECISION [{clause}]: {verdict}")

    # ---------------------------------------------------------------- metrics
    metrics = dict(
        experiment="e089_mass_response",
        purpose="T051's registered mass-response curve (P3 capstone): dynamic "
                "removal cost vs NUMBER of anchor-band entries removed, "
                "random subsets k in {2..200}, 10 draws each, matched-stream "
                "continuations with clean-net final-64 tail judgment (the "
                "e085/e088 instrument; per-entry age-97-crossing V-zero "
                "schedule — e085's timing applied to sets, the registered "
                "adaptation documented in the file docstring). Threshold rise "
                "=> mass-action; proportional rise => diffuse linear.",
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
            ks=KS, n_draws_per_k=N_DRAWS,
            subset_rule="rng default_rng(89), k-major draw-minor order; "
                        "rng.choice(band, size=k, replace=False); reject & "
                        "redraw while min(cols) > 350 (full-tail exposure; "
                        "rejections tallied)",
            run_rule="run = (k_index*10 + draw) % 8 — 10 draws per run "
                     "overall, >= 1 draw of every k per run",
            timing=dict(
                schedule="per-entry age-97-crossing V-zero: entry p zeroed "
                         "immediately before the decode processing position "
                         "p+97 (e085 timing VERBATIM applied to sets; the "
                         "single-shot e088 variant is infeasible for k > 97 "
                         "spread entries because later columns do not exist "
                         "at T_int)",
                t_int="min(S) + 97 (the earliest entry's crossing = the "
                      "forced decode)",
                first_affected_sample="min(S) + 98",
                exposure="414 - min(S) (>= 64 by the min <= 350 floor)",
                unremovable="p = 414 crosses at 511, beyond the last decode "
                            "(510): never zeroed, stays live, exposure 0 — "
                            "tallied per draw",
                e075_relation="k=200 arm = e075's A-self-prune at K=1 "
                              "continuous schedule (e075 used K=32 events)"),
            arms="2 per group, SAME continuation seed (matched streams): "
                 "control / crossing-time progressive removal of the subset",
            outcome="clean-judge tail CE (448..511) of removal stream minus "
                    "matched control stream; per-k MEAN over 10 draws",
            bars=dict(
                mass_action=f"mean(32) < {MASS_LOW}*mean(200) AND mean(128) > "
                            f"{MASS_BREAK}*mean(64)",
                linear=f"|mean(64)/mean(200) - {LIN_FRAC}| <= {LIN_TOL} AND "
                       f"mean(128) <= {MASS_BREAK}*mean(64)",
                indeterminate="mean(200) <= 0",
                order="indeterminate -> mass-action -> linear -> honest "
                      "texture",
                secondary="per-k SD/CV (mass-action predicts variance "
                          "collapses past the break); per-draw cost vs "
                          "min(S); first divergence"),
            registered_numbers=dict(mass_low=MASS_LOW, mass_break=MASS_BREAK,
                                    lin_frac=LIN_FRAC, lin_tol=LIN_TOL),
        ),
        gates=gates,
        control_run=dict(clean_judge_tail_ce=cj1.tolist()),
        e075_reference=dict(file=str(E075_METRICS),
                            self_minus_none=e075_ref,
                            note="e075 A-self-prune whole-band cost, K=32 "
                                 "event schedule; e089 k=200 is the K=1 "
                                 "continuous analogue"),
        summary=dict(
            n_draws=len(draws), n_groups=len(groups),
            per_k={str(k): dict(n=N_DRAWS, mean=means[k], sd=sds[k], cv=cvs[k],
                                ci=cis[k], draws=per_k_draws[k]) for k in KS},
            ratios=dict(
                r32_over_200=dict(val=r32_200, ci=ci_r32,
                                  bar=f"< {MASS_LOW} (mass-action)"),
                r64_over_200=dict(val=r64_200, ci=ci_r64,
                                  bar=f"~ {LIN_FRAC}+-{LIN_TOL} (linear)"),
                r128_over_64=dict(val=r128_64, ci=ci_r128,
                                  bar=f"> {MASS_BREAK} (mass-action break)")),
            variance=dict(sd={str(k): sds[k] for k in KS},
                          cv={str(k): cvs[k] for k in KS},
                          sd_128_over_8=sd_ratio_128_8,
                          sd_200_over_8=sd_ratio_200_8),
            texture=dict(spearman_cost_min=sp_min,
                         stream_identical_by_k={
                             str(k): sum(dr["stream_identical"] for dr in draws
                                         if dr["k"] == k) for k in KS},
                         unremovable_draws=sum(1 for dr in draws
                                               if dr["n_fired_unremovable"])),
        ),
        draws=[{k2: v for k2, v in dr.items() if k2 != "cols"} | dict(
            n_cols=len(dr["cols"])) for dr in draws],
        subsets={f"k{dr['k']}_d{dr['d']}": dr["cols"] for dr in draws},
        registered_decision=dict(clause=clause, verdict=verdict,
                                 numbers=dict(
                                     means={str(k): means[k] for k in KS},
                                     r32_200=r32_200, r64_200=r64_200,
                                     r128_64=r128_64,
                                     ci_r32_200=ci_r32, ci_r64_200=ci_r64,
                                     ci_r128_64=ci_r128,
                                     mass_low=MASS_LOW,
                                     mass_break=MASS_BREAK,
                                     lin_frac=LIN_FRAC, lin_tol=LIN_TOL,
                                     e075_ref=e075_ref)),
    )
    save_json(out_dir / "metrics.json", metrics)
    log("metrics.json written")

    plot(out_dir / "mass_response.png", metrics)
    log(f"plot written; total wall {elapsed():.0f}s")


# ---------------------------------------------------------------------- plot


def plot(path: Path, M: dict):
    dec = M["registered_decision"]
    S = M["summary"]
    n = dec["numbers"]
    ks = KS
    means = [n["means"][str(k)] for k in ks]
    cis = [S["per_k"][str(k)]["ci"] for k in ks]
    c200 = means[-1]
    e075 = n["e075_ref"]
    run_cols = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, axes = plt.subplots(2, 3, figsize=(24, 11))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    # ---- panel 1: THE curve with the two prediction shapes overlaid
    xk = np.array(ks, float)
    jit = (np.arange(N_DRAWS) - (N_DRAWS - 1) / 2) * 0.06
    draws_by_k = {str(k): S["per_k"][str(k)]["draws"] for k in ks}
    for k in ks:
        ys = draws_by_k[str(k)]
        xs = np.full(N_DRAWS, math.log2(k)) + jit
        run_of = [((ks.index(k)) * N_DRAWS + d) % B
                  for d in range(N_DRAWS)]
        ax1.scatter(xs, ys, s=30, color=run_cols[run_of], alpha=0.65, zorder=3)
    err = [[max(0.0, m - ci[0]) for m, ci in zip(means, cis)],
           [max(0.0, ci[1] - m) for m, ci in zip(means, cis)]]
    ax1.errorbar(np.log2(xk), means, yerr=err, fmt="o-", color="k", lw=2.2,
                 ms=9, capsize=5, zorder=4, label="measured mean (cluster CI)")
    kfine = np.logspace(np.log10(2), np.log10(200), 200)
    ax1.plot(np.log2(kfine), c200 * kfine / 200.0, "--", color="tab:blue",
             lw=1.8, label="LINEAR prediction: c(200)*k/200 (diffuse)")
    ax1.plot(np.log2(kfine), c200 * kfine ** 3 / (100.0 ** 3 + kfine ** 3),
             "--", color="tab:red", lw=1.8,
             label="MASS-ACTION illustration: Hill k$^3$/(100$^3$+k$^3$)")
    if e075 is not None:
        ax1.axhline(e075, color="tab:green", ls="-.", lw=1.4,
                    label=f"e075 whole-band {e075:+.3f} (K=32 schedule)")
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_xticks(np.log2(xk))
    ax1.set_xticklabels([str(k) for k in ks])
    ax1.set_xlabel("k (anchor entries removed, log2 scale)")
    ax1.set_ylabel("clean-judge tail cost (nats)")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.set_title("E089-1 — THE mass-response curve: removal cost vs k "
                  f"(colored dots = 10 draws/run shades; k=200 = {c200:+.3f})",
                  fontsize=10)

    # ---- panel 2: the three registered ratios vs their thresholds
    rr = S["ratios"]
    labs = ["cost(32)/cost(200)\nmass: < 0.25", "cost(64)/cost(200)\nlin: 0.32",
            "cost(128)/cost(64)\nmass: > 3"]
    vals = [rr["r32_over_200"]["val"], rr["r64_over_200"]["val"],
            rr["r128_over_64"]["val"]]
    cites = [rr["r32_over_200"]["ci"], rr["r64_over_200"]["ci"],
             rr["r128_over_64"]["ci"]]
    xpos = np.arange(3)
    ax2.bar(xpos, vals, width=0.55, color=["tab:red", "tab:blue",
                                           "tab:red"], alpha=0.75)
    for x, v, ci in zip(xpos, vals, cites):
        ax2.errorbar(x, v, yerr=[[max(0, v - ci[0])], [max(0, ci[1] - v)]],
                     fmt="k_", capsize=6, lw=1.6, ms=12)
    for x, thr, lab in ((0, MASS_LOW, f"< {MASS_LOW}"), (1, LIN_FRAC, "0.32"),
                        (2, MASS_BREAK, f"> {MASS_BREAK}")):
        ax2.axhline(thr, color="gray", ls=":", lw=1.2)
        ax2.text(x, thr, " " + lab, fontsize=9, va="bottom", ha="center")
    ax2.axhline(1.0, color="k", lw=0.7)
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(labs, fontsize=9)
    ax2.set_ylabel("ratio (per-k means)")
    ax2.set_title("E089-2 — REGISTERED ratios with cluster-bootstrap CIs",
                  fontsize=10)

    # ---- panel 3: variance across draws (the collapse test)
    sds_ = [S["per_k"][str(k)]["sd"] for k in ks]
    cvs_ = [S["per_k"][str(k)]["cv"] for k in ks]
    ax3.plot(np.log2(xk), sds_, "o-", color="tab:purple", lw=2,
             label="SD across draws")
    ax3.set_xticks(np.log2(xk))
    ax3.set_xticklabels([str(k) for k in ks])
    ax3.set_xlabel("k (log2 scale)")
    ax3.set_ylabel("SD (nats)", color="tab:purple")
    ax3b = ax3.twinx()
    ax3b.plot(np.log2(xk), cvs_, "s--", color="tab:orange", lw=1.8, ms=6,
              label="CV = SD/|mean|")
    ax3b.set_ylabel("CV", color="tab:orange")
    ax3.set_title("E089-3 — variance across draws: mass-action predicts "
                  "COLLAPSE past the break (SD128/SD8 = "
                  f"{S['variance']['sd_128_over_8']:.2f}, SD200/SD8 = "
                  f"{S['variance']['sd_200_over_8']:.2f})", fontsize=9.5)
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3b.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, fontsize=9)

    # ---- panel 4: per-draw cost vs min(S) (exposure texture)
    all_dr = M["draws"]
    kcols = plt.cm.viridis(np.linspace(0, 0.9, len(ks)))
    for ki, k in enumerate(ks):
        dd = [dr for dr in all_dr if dr["k"] == k]
        ax4.scatter([dr["min_p"] for dr in dd], [dr["cost"] for dr in dd],
                    s=34, color=kcols[ki], alpha=0.8, label=f"k={k}")
    ax4.axhline(0, color="k", lw=0.6)
    ax4.set_xlabel("min(S) — the subset's earliest entry (exposure = 414-min)")
    ax4.set_ylabel("per-draw cost (nats)")
    ax4.set_title("E089-4 — exposure texture: cost vs min(S) | Spearman "
                  f"{S['texture']['spearman_cost_min']:+.3f} (descriptive)",
                  fontsize=10)
    ax4.legend(fontsize=8, ncol=2)

    # ---- panel 5: first divergence vs k (stream texture)
    for ki, k in enumerate(ks):
        dd = [dr for dr in all_dr if dr["k"] == k]
        fd = [dr["first_div"] if dr["first_div"] is not None else 512
              for dr in dd]
        idn = [dr["stream_identical"] for dr in dd]
        ax5.scatter(np.full(len(dd), math.log2(k)) + jit[:len(dd)], fd, s=34,
                    color=kcols[ki], alpha=0.8,
                    marker=("x" if all(idn) else "o"))
    ax5.axhline(KEY_T[0], color="tab:green", ls="-.", lw=1.2,
                label="tail start 448")
    ax5.axhline(512, color="gray", ls=":", lw=1.2)
    ax5.text(math.log2(2), 512.5, "512 = stream never diverged (x)", fontsize=8)
    ax5.set_xticks(np.log2(xk))
    ax5.set_xticklabels([str(k) for k in ks])
    ax5.set_ylim(150, 516)
    ax5.set_xlabel("k (log2 scale)")
    ax5.set_ylabel("first position removal stream != control")
    ax5.set_title("E089-5 — stream divergence vs k", fontsize=10)
    ax5.legend(fontsize=9)

    # ---- panel 6: verdict text
    ax6.axis("off")
    lines = [
        "REGISTERED (frozen in T051):",
        f"  mass-action: c(32) < {MASS_LOW}*c(200) AND c(128) > "
        f"{MASS_BREAK}*c(64)",
        f"  linear: |c(64)/c(200) - {LIN_FRAC}| <= {LIN_TOL} AND no break",
        "  c(200) <= 0 -> INDETERMINATE; else honest texture",
        "",
        "CURVE (per-k means, cluster CI):",
    ] + [
        f"  k={k:>3}: {n['means'][str(k)]:+.3f} "
        f"{_fmt_ci(S['per_k'][str(k)]['ci'])}"
        for k in ks
    ] + [
        "",
        f"RATIOS: 32/200 {r32_200_lab(n)} | 64/200 "
        f"{n['r64_200']:+.3f} {_fmt_ci(n['ci_r64_200'])} | 128/64 "
        f"{n['r128_64']:+.2f} {_fmt_ci(n['ci_r128_64'])}",
        f"  k=200 level {c200:+.3f} vs e075 whole-band "
        f"{e075 if e075 is not None else float('nan'):+.3f} (K=32 sched)",
        f"  variance: SD8 {S['per_k']['8']['sd']:.3f} -> SD128 "
        f"{S['per_k']['128']['sd']:.3f} -> SD200 "
        f"{S['per_k']['200']['sd']:.3f}",
        "",
        f"DECISION [{dec['clause']}]:",
    ] + [f"  {wd}" for wd in _wrap(dec["verdict"], 96)]
    ax6.text(0.02, 0.97, "E089 — T051 mass-response curve (P3 capstone)",
             fontsize=13, weight="bold", va="top")
    for i, tx in enumerate(lines):
        ax6.text(0.02, 0.93 - i * 0.030, tx, fontsize=8.4, va="top",
                 family="monospace")

    fig.suptitle("E089 — mass-response curve: anchor removal cost vs number "
                  f"removed | clause: {dec['clause']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def r32_200_lab(n):
    return f"{n['r32_200']:+.3f} {_fmt_ci(n['ci_r32_200'])}"


if __name__ == "__main__":
    main()
