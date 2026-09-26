"""E088 — T051's registered PAIR-LEVEL anchor probe (the view-a/view-b
discriminator). CPU-only.

T051 (E085) left the anchor undescriptionable at entry level: readership +0.089
(bar 0.35), every property |r| <= 0.10 on the dynamic arm, static/dynamic r =
-0.004. Two surviving views: (a) the anchor is SEQUENCE-level — a basin
property of the whole run trajectory, no single entry load-bearing (consistent
with e085's near-null single-entry dynamic costs: median 0.0, only 18% of
anchors > 0.10 nats); or (b) the anchor is entry-level but in an unmeasured
property family — e.g. entry-PAIR interactions.

DESIGN (registered here BEFORE compute; bars frozen from T051 verbatim).
Battery: the e053c ctx-512 net, the seed-202 8-draw prompts, B=8, the seed-7
control free run (e075/e080/e085 A-none convention). Instrument: the e085
dynamic single-entry removal rig VERBATIM (run_continuation with post-prefill
KV-column zeroing, teacher-forced unaffected token, matched-stream
free-run, clean-net judgment of the final-64 tail) extended to a 4-arm
factorial per pair: control / remove-entry-1 / remove-entry-2 / remove-BOTH,
all four arms sharing the SAME continuation seed (matched streams row-by-row;
the pair arm and each single arm diverge only through the extra zeroed
column's effect on the sampled stream).

SAMPLING (rng seed 88, 40 pairs, all WITHIN-RUN, from the anchor band
positions 64..414 = final-frame ages 97..447, with p1 <= 350 so the judged
tail 448..511 is fully free-sampled after removal; 5 pairs per run x 8 runs):
  - 16 ADJACENT pairs  (2/run): d = p2-p1 ~ U{1,2,3}, p1 ~ U[64, 350-d];
  - 8  SAME-DECADE     (1/run): decade k ~ U{17..43} (ages 170..439), two
    distinct ages in the decade with |delta age| in [2,9] -> d in [2,9];
  - 16 CROSS-DECADE    (2/run): d ~ U[40,96] (>= 4 decades apart), p1 ~
    U[64, min(350, 414-d)].
  Duplicate (run, p1, p2) triples rejected and redrawn.

TIMING (registered adaptation, documented): all four arms intervene at
T_int = p1 + 97 — the EARLIER entry's age-crossing (e075/e085 timing verbatim
for entry-1; entry-2 is removed earlier than its own crossing, and its single
arm uses the SAME T_int so the factorial is internally matched at one fixed
intervention time; both cache columns exist at T_int because d <= 96).
First affected sample = p1+98; exposure (free samples) = 414 - p1 >= 64.

READOUT per pair (clean-judge tail CE of arm stream minus control stream):
  c_s1, c_s2, c_pair; RATIO = c_pair / (c_s1 + c_s2), defined only when
  s1+s2 > DEN_FLOOR = 0.10 nats (registered stability floor; e085's singles
  median was 0.0, so a large fraction of pairs is expected to fall below it —
  tallied honestly, not discarded silently).

REGISTERED BARS (frozen in T051 / the tasking, evaluated in this order):
  1. n_qualifying < 8  -> INDETERMINATE (honest texture; the floor ate the
     readout) — no bar may fire on a median of a handful of ratios.
  2. median ratio >= 1.5 AND cluster-bootstrap CI (8 runs, 1000 reps) of the
     median EXCLUDES 1.0 (ci_low > 1.0) -> INTERACTION STRUCTURE (view b:
     the anchor lives in entry-pair interactions).
  3. median ratio within [0.8, 1.2] -> ADDITIVE -> SEQUENCE-LEVEL (view a:
     the anchor is a basin property; P3 proceeds to run-level descriptors —
     run-PCA distance to attractor, drift entropy). CI reported alongside;
     contradiction noted in the verdict text.
  4. anything else -> honest texture (report the distribution as measured).

SECONDARY (registered): does interaction strength decay with pair distance?
Spearman(ratio, d) over qualifying pairs (+ cluster CI) and per-class medians
(adjacent / same-decade / cross-decade). View-b localization predicts a
NEGATIVE slope (near entries interact; far entries additive).

REGISTERED TEXTURE TALLIES (companions, given e085's near-null singles):
  - pure-synergy pairs: s1+s2 <= 0.10 AND c_pair >= 0.25 (both singles free,
    the pair costly — view-b-supportive, ratio undefined);
  - floor pairs: s1+s2 <= 0.10 and pair below synergy threshold (null/null);
  - negative denominators (a single removal HELPED the tail);
  - aggregate ratio sum(c_pair)/sum(s1+s2) over ALL pairs;
  - stream flags: pair stream identical to s1 / s2 streams (if the pair
    stream equals a single's, c_pair == that single's cost mechanically).

Gates: G1 val CE vs e053c 1.5227 +-0.02; G2 params 873472; G3 (scope: the
control battery — r1 tail CE + clean-judge per-seq identity vs e080's stored
none arm, hard bar 1e-4, bitwise flag; the static-sweep/a* leg of e085's G3
is NOT APPLICABLE here — e088 has no static arm; documented tractability
cut); G4 dynamic-instrument identity per group (prefix identity through the
teacher-forced token; removed columns exactly 0; non-removed and control
columns live; matched per-group seeds across the four arms).

Run:     python lab/e088_pair_anchor.py
Outputs: runs/e088/metrics.json + runs/e088/pair_anchor.png
Envelope: NO training, NO new automations; CPU-only (CUDA_VISIBLE_DEVICES=-1),
8 threads (T050: 12 spin-thrashes this box), single step, target ~10-20 min.
No NOTES/THINKING/QUEUE/STATE edits; no commit (the dispatcher owns those).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # task spec: CPU-only (pre-torch)
import torch

# this torch build reports is_available()==True even with an empty
# CUDA_VISIBLE_DEVICES; force CPU so common.DEVICE=="cpu" (e053b..e085)
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
SEED_PROMPT, SEED_SAMPLE = 202, 7         # e053b/e053c/e069..e085 seeds
SEED_PAIR = 88                            # e088: pair-sampling rng
SEED_CONT = 881000                        # e088: continuation seeds (+ p1)
B = 8
N_PROMPTS = 8
BOOT_N = 1000
THREADS = 8                               # T050: 12 spin-thrashes the box
torch.set_num_threads(THREADS)

P512 = T_TOTAL - 1                        # 511
TAIL = 64                                 # the e075/e080/e085 tail window
KEY_T = (T_TOTAL - TAIL, T_TOTAL - 1)     # judged tail window (448, 511)
G = T_TOTAL - PROMPT_TOK                  # 448 generation steps

# ---- the anchor band (final-frame ages; age of position p = 511 - p) ---------
ANCHOR_POS = (64, 414)                    # ages 97..447 (self-generated)
P1_MAX = 350                              # full-tail-exposure: p1+98 <= 448
PER_RUN = dict(adjacent=2, same_decade=1, cross_decade=2)   # 5 x 8 = 40
DIST_ADJ = (1, 3)                         # inclusive
DIST_CROSS = (40, 96)                     # inclusive
DEC_K = (17, 43)                          # inclusive age decades (170..439)

# ---- REGISTERED decision numbers (frozen, docstring verbatim) ----------------
DEN_FLOOR = 0.10                          # ratio defined iff s1+s2 > this
SYN_PAIR_MIN = 0.25                       # pure-synergy pair-cost threshold
MIN_QUAL = 8                              # indeterminacy guard
BAR_INTERACT = 1.5                        # median >= AND ci_low > 1.0
BAR_ADD = (0.8, 1.2)                      # median within => ADDITIVE

# ---- reference numbers (protocol-identity gates) -----------------------------
CKPT = REPO / "runs" / "checkpoints" / "e053c_ctx512.pt"
E053C_VAL_CE = 1.5226792494455974         # the net being interrogated
E080_METRICS = REPO / "runs" / "e080" / "metrics.json"
T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# ------------------------------------------------- machinery (VERBATIM e085)
# (e085's _manual_chunk/manual_logits static-sweep instrument is NOT used
#  here — e088 has no static arm; documentented in the G3 gate note.)

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
    [VERBATIM e075/e080/e085]"""
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
    [e075/e085 verbatim; e088 never collects attention mass]"""
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
    """Free-run 64->512 for B sequences (e075/e080/e085 A-none VERBATIM stream
    math; e085's mass collection dropped — it never touched the logits path)."""
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
    """Entry-removal continuation (and its matched control when remove=None).
    prefix = control tokens through position forced_pos-1; the token at
    forced_pos is teacher-forced (it precedes the first affected sample by
    construction); removal (list of (row, col)) is applied to the KV cache
    AFTER prefill and BEFORE the decode of the forced token — exactly e075's
    'top of the event iteration' timing — so the first AFFECTED sample is
    position forced_pos+1. Free-run from there with the row-order shared
    generator (e053b stream math). [VERBATIM e085; remove may hold >1 col/row]
    Returns the full 512-token idx and the final kv."""
    R = prefix.shape[0]
    gen = torch.Generator().manual_seed(seed)
    logits, kv = prefill_batch(net, prefix)
    if remove is not None:
        for (r, c) in remove:
            for (_k, v) in kv:
                v[r, :, c, :] = 0.0
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
            logits = decode_step_batch(net, new, pos, kv)
    return idx, kv


def judge_windows(all_lg: torch.Tensor, idx: torch.Tensor, windows):
    """Clean-net CE of target windows [(lo, hi), ...] (queries lo-1..hi-1).
    Returns dict window -> (R,) mean CE per row. [VERBATIM e085]"""
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


def cluster_boot(pairs, fn, n: int = BOOT_N, seed: int = 0):
    """Cluster bootstrap over the 8 runs: resample run ids with replacement,
    rebuild the (multiply-counted) pair list, recompute fn (None if the
    resample can't support it). CI = 2.5/97.5 percentiles of valid draws."""
    runs = sorted({p["run"] for p in pairs})
    by_run = {r: [i for i, p in enumerate(pairs) if p["run"] == r] for r in runs}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sel = []
        for r in rng.integers(0, len(runs), len(runs)):
            sel.extend(by_run[runs[r]])
        v = fn([pairs[i] for i in sel])
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return [float("nan"), float("nan")], 0
    return [float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5))], len(vals)


def _fmt_ci(ci):
    return f"[{ci[0]:+.3f},{ci[1]:+.3f}]"


# ---------------------------------------------------------------------- main


def main():
    assert torch.cuda.device_count() == 0, "CPU-only violated: CUDA visible"
    assert common.DEVICE == "cpu", f"common.DEVICE={common.DEVICE} != cpu"
    out_dir = run_dir("e088")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gates = dict(cpu_only=True, threads=THREADS, smoke=False)

    # ---- battery: rebuild corpus + net EXACTLY as e053c/e069..e085 did
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
    log("control battery: seed-7 free run (e075/e080/e085 A-none convention)")
    gen7 = torch.Generator().manual_seed(SEED_SAMPLE)
    run1 = generate_control(net, prompts8, gen7)
    idx1 = run1["idx"]
    log(f"  control run done ({T_TOTAL} positions x {B} rows)")

    # ---- G3 (scope: control battery) vs e080's stored A-none arm
    cj1 = judge_windows(manual_all_logits(net, idx1), idx1, [KEY_T])[KEY_T]
    g3 = dict(ref_file=str(E080_METRICS), ok=False,
              note="e088 has no static arm; G3 scope = control-battery "
                   "identity (r1 online tail CE + clean-judge per-seq vs "
                   "e080 stored none arm); the sweep/a* leg of e085's G3 is "
                   "not applicable")
    ce_tail = np.zeros(B, float)
    _lg, _kv = prefill_batch(net, idx1[:, :PROMPT_TOK])
    # r1 per-seq online tail CE recomputed on the control stream:
    with torch.no_grad():
        allg = manual_all_logits(net, idx1[:, :-1])
        lp = torch.log_softmax(allg.float(), -1)
        tgt = idx1[:, 1:]
        ce_tokens = -lp.gather(2, tgt[:, :, None]).squeeze(2)
        ce_tail = ce_tokens[:, -TAIL:].mean(1).numpy()
    if E080_METRICS.exists():
        import json as _json
        with open(E080_METRICS) as f:
            e080 = _json.load(f)
        ref_cj = np.asarray(e080["arms"]["none"]["clean_judge_tail_ce"]["per_seq"])
        dev_cj = float(np.abs(cj1 - ref_cj).max())
        bit = bool(np.array_equal(cj1, ref_cj))
        dev_r1 = float(np.abs(ce_tail - ref_cj).max())
        g3.update(clean_judge_max_dev=dev_cj, clean_judge_bitwise=bit,
                  r1_proxy_max_dev=dev_r1,
                  ok=bool(dev_cj < 1e-4))
        log(f"G3 vs e080 A-none: clean-judge dev {dev_cj:.2e} (bitwise {bit})"
            f" -> {'PASS' if g3['ok'] else 'FAIL'}")
    else:
        g3["note"] += " | runs/e080/metrics.json not found; gate skipped"
        log("G3: e080 metrics missing — skipped")
    gates["G3_protocol_identity_vs_e080"] = g3

    # ================================================== pair sampling (seed 88)
    log("pair sampling: rng seed 88, 40 within-run pairs from the anchor "
        "band (p1 <= 350 for full-tail exposure)")
    rng = np.random.default_rng(SEED_PAIR)
    seen = set()
    pairs = []

    def add(b, p1, p2, cls):
        if (b, p1, p2) in seen:
            return False
        seen.add((b, p1, p2))
        pairs.append(dict(run=b, p1=int(p1), p2=int(p2), cls=cls,
                          d=int(p2 - p1), age1=P512 - int(p2), age2=P512 - int(p1),
                          exposure=int(T_TOTAL - 1 - (p1 + 98) + 1)))
        return True

    for b in range(B):
        made = dict(adjacent=0, same_decade=0, cross_decade=0)
        for cls, want in PER_RUN.items():
            while made[cls] < want:
                if cls == "adjacent":
                    d = int(rng.integers(DIST_ADJ[0], DIST_ADJ[1] + 1))
                    p1 = int(rng.integers(ANCHOR_POS[0], P1_MAX - d + 1))
                    ok = add(b, p1, p1 + d, cls)
                elif cls == "same_decade":
                    k = int(rng.integers(DEC_K[0], DEC_K[1] + 1))
                    offs = rng.choice(10, size=2, replace=False)
                    if abs(int(offs[0]) - int(offs[1])) < 2:
                        continue                      # resample: need d >= 2
                    ages = sorted(int(10 * k + o) for o in offs)
                    p1, p2 = P512 - ages[1], P512 - ages[0]
                    assert ANCHOR_POS[0] <= p1 and p2 <= ANCHOR_POS[1]
                    ok = add(b, p1, p2, cls)
                else:
                    d = int(rng.integers(DIST_CROSS[0], DIST_CROSS[1] + 1))
                    p1 = int(rng.integers(ANCHOR_POS[0],
                                          min(P1_MAX, ANCHOR_POS[1] - d) + 1))
                    ok = add(b, p1, p1 + d, cls)
                if ok:
                    made[cls] += 1
    assert len(pairs) == sum(PER_RUN.values()) * B == 40
    for cls in PER_RUN:
        ds = [p["d"] for p in pairs if p["cls"] == cls]
        log(f"  stratum {cls}: n={len(ds)} d in [{min(ds)},{max(ds)}]")
    log(f"  p1 range {min(p['p1'] for p in pairs)}.."
        f"{max(p['p1'] for p in pairs)}; exposure range "
        f"{min(p['exposure'] for p in pairs)}.."
        f"{max(p['exposure'] for p in pairs)}")

    # ================================================== the 4-arm factorial
    log("dynamic arm: 4 matched continuations per pair-group (control / s1 / "
        "s2 / pair) at T_int = p1+97, clean-judge tail (448..511)")
    groups = {}
    for pr in pairs:
        groups.setdefault(pr["p1"], []).append(pr)

    g4_all = dict(ident_pre=True, col_zero=True, col_live=True, n_groups=0)
    for gi, (p1, grp) in enumerate(sorted(groups.items())):
        rows = [pr["run"] for pr in grp]
        forced_pos = p1 + 97
        prefix = idx1[rows, :forced_pos]
        forced = idx1[rows, forced_pos]
        seed = SEED_CONT + p1
        rm_s1 = [(j, p1) for j in range(len(grp))]
        rm_s2 = [(j, pr["p2"]) for j, pr in enumerate(grp)]
        rm_pair = rm_s1 + rm_s2
        idx_ctl, kv_ctl = run_continuation(net, prefix, forced, forced_pos,
                                           seed, remove=None)
        idx_s1, kv_s1 = run_continuation(net, prefix, forced, forced_pos,
                                         seed, remove=rm_s1)
        idx_s2, kv_s2 = run_continuation(net, prefix, forced, forced_pos,
                                         seed, remove=rm_s2)
        idx_pr, kv_pr = run_continuation(net, prefix, forced, forced_pos,
                                         seed, remove=rm_pair)
        streams = dict(control=idx_ctl, s1=idx_s1, s2=idx_s2, pair=idx_pr)
        kvs = dict(control=kv_ctl, s1=kv_s1, s2=kv_s2, pair=kv_pr)
        rmm = dict(s1=set(rm_s1), s2=set(rm_s2), pair=set(rm_pair))
        J = {}
        for nm, ix_ in streams.items():
            J[nm] = judge_windows(manual_all_logits(net, ix_), ix_, [KEY_T])[KEY_T]
        # ---- G4 identity checks for this group
        ident_pre = all(bool(torch.equal(ix_[:, :forced_pos + 1],
                                         idx_ctl[:, :forced_pos + 1]))
                        for nm, ix_ in streams.items() if nm != "control")
        col_zero, col_live = True, True
        for nm in ("s1", "s2", "pair"):
            for (r, c) in rmm[nm]:
                for (_k, v) in kvs[nm]:
                    col_zero &= bool(v[r, :, c, :].abs().max().item() == 0.0)
            for (r, c) in (rm_s1 + rm_s2):
                if (r, c) not in rmm[nm]:
                    for (_k, v) in kvs[nm]:
                        col_live &= bool(v[r, :, c, :].abs().max().item() > 0.0)
        for (r, c) in (rm_s1 + rm_s2):
            for (_k, v) in kv_ctl:
                col_live &= bool(v[r, :, c, :].abs().max().item() > 0.0)
        g4_all["ident_pre"] &= ident_pre
        g4_all["col_zero"] &= col_zero
        g4_all["col_live"] &= col_live
        g4_all["n_groups"] += 1
        # ---- per-pair readouts
        for j, pr in enumerate(grp):
            pr["tail_control"] = float(J["control"][j])
            pr["tail_s1"] = float(J["s1"][j])
            pr["tail_s2"] = float(J["s2"][j])
            pr["tail_pair"] = float(J["pair"][j])
            pr["c_s1"] = float(J["s1"][j] - J["control"][j])
            pr["c_s2"] = float(J["s2"][j] - J["control"][j])
            pr["c_pair"] = float(J["pair"][j] - J["control"][j])
            pr["div_s1"] = bool(not torch.equal(idx_s1[j], idx_ctl[j]))
            pr["div_s2"] = bool(not torch.equal(idx_s2[j], idx_ctl[j]))
            pr["div_pair"] = bool(not torch.equal(idx_pr[j], idx_ctl[j]))
            pr["pair_eq_s1"] = bool(torch.equal(idx_pr[j], idx_s1[j]))
            pr["pair_eq_s2"] = bool(torch.equal(idx_pr[j], idx_s2[j]))
        if (gi + 1) % 8 == 0 or gi + 1 == len(groups):
            log(f"  pair groups {gi + 1}/{len(groups)} done "
                f"({elapsed():.0f}s)")
    g4_all["ok"] = bool(g4_all["ident_pre"] and g4_all["col_zero"]
                        and g4_all["col_live"])
    gates["G4_dynamic_instrument"] = g4_all
    log(f"G4 dynamic instrument: {g4_all['n_groups']} groups, prefix identity "
        f"{g4_all['ident_pre']}, removed cols zero {g4_all['col_zero']}, "
        f"other/control cols live {g4_all['col_live']} -> "
        f"{'PASS' if g4_all['ok'] else 'FAIL'}")

    # ================================================== ratio readout + bars
    for pr in pairs:
        pr["denom"] = pr["c_s1"] + pr["c_s2"]
        pr["ratio"] = (pr["c_pair"] / pr["denom"]
                       if pr["denom"] > DEN_FLOOR else None)
        pr["qualifies"] = bool(pr["denom"] > DEN_FLOOR)
        pr["synergy"] = bool(pr["denom"] <= DEN_FLOOR
                             and pr["c_pair"] >= SYN_PAIR_MIN)
    qual = [p for p in pairs if p["qualifies"]]
    ratios = np.array([p["ratio"] for p in qual], float)
    n_qual = len(qual)
    log(f"readout: {n_qual}/40 pairs qualify (s1+s2 > {DEN_FLOOR}); "
        f"synergy (den<=floor, pair>={SYN_PAIR_MIN}): "
        f"{sum(p['synergy'] for p in pairs)}")

    med = float(np.median(ratios)) if n_qual else float("nan")
    if n_qual:
        ci_med, n_valid = cluster_boot(
            qual, lambda gg: (float(np.median([p["ratio"] for p in gg]))
                              if len(gg) >= 3 else None))
    else:
        ci_med, n_valid = [float("nan")] * 2, 0

    # secondary: ratio vs distance
    if n_qual >= 4:
        sp = spearman([p["ratio"] for p in qual], [p["d"] for p in qual])
        pe = pearson([p["ratio"] for p in qual], [p["d"] for p in qual])
        ci_sp, _ = cluster_boot(
            qual, lambda gg: (spearman([p["ratio"] for p in gg],
                                       [p["d"] for p in gg])
                              if len(gg) >= 4 else None))
    else:
        sp = pe = float("nan")
        ci_sp = [float("nan"), float("nan")]
    class_med = {}
    for cls in ("adjacent", "same_decade", "cross_decade"):
        cq = [p for p in qual if p["cls"] == cls]
        if cq:
            cm = float(np.median([p["ratio"] for p in cq]))
            ci_c, _ = cluster_boot(
                cq, lambda gg: (float(np.median([p["ratio"] for p in gg]))
                                if len(gg) >= 3 else None))
        else:
            cm, ci_c = float("nan"), [float("nan"), float("nan")]
        class_med[cls] = dict(n=len(cq), median=cm, ci=ci_c)

    # tallies + aggregate
    n_syn = sum(1 for p in pairs if p["synergy"])
    n_floor = sum(1 for p in pairs
                  if not p["qualifies"] and not p["synergy"]
                  and p["denom"] > 0)
    n_den_neg = sum(1 for p in pairs if p["denom"] <= 0)
    n_fac_single = sum(1 for p in pairs if p["c_s1"] < 0 or p["c_s2"] < 0)
    tot_pair = float(sum(p["c_pair"] for p in pairs))
    tot_sing = float(sum(p["denom"] for p in pairs))
    agg_ratio = (tot_pair / tot_sing) if abs(tot_sing) > 1e-9 else float("nan")
    n_pair_eq_s1 = sum(1 for p in pairs if p["pair_eq_s1"])
    n_pair_eq_s2 = sum(1 for p in pairs if p["pair_eq_s2"])
    n_div_any = sum(1 for p in pairs if p["div_pair"])
    n_div_s1 = sum(1 for p in pairs if p["div_s1"])
    n_div_s2 = sum(1 for p in pairs if p["div_s2"])

    # ---- REGISTERED bars (frozen; evaluated in order)
    if n_qual < MIN_QUAL:
        clause = "INDETERMINATE (too few qualifying pairs)"
        verdict = (f"Only {n_qual}/40 pairs clear the registered denominator "
                   f"floor (s1+s2 > {DEN_FLOOR} nats) — consistent with "
                   f"e085's near-null single-entry costs (median 0.0). The "
                   f"median ratio is too unstable to fire any registered "
                   f"bar. Texture: {n_syn} pure-synergy pairs "
                   f"(singles free, pair cost >= {SYN_PAIR_MIN}), "
                   f"{n_floor} floor-null pairs, {n_den_neg} negative "
                   f"denominators, aggregate ratio {agg_ratio:+.2f}.")
    elif med >= BAR_INTERACT and ci_med[0] > 1.0:
        clause = "INTERACTION STRUCTURE (view b)"
        verdict = (f"Super-additivity fires: median pair/single ratio {med:.3f} "
                   f">= {BAR_INTERACT} with cluster CI {_fmt_ci(ci_med)} "
                   f"excluding 1.0 (n={n_qual} qualifying). The anchor lives "
                   f"in entry-PAIR interactions (T051 view b) — removal of "
                   f"anchor pairs costs MORE than the sum of their singles; "
                   f"P3's description leg pivots to interaction descriptors.")
    elif BAR_ADD[0] <= med <= BAR_ADD[1]:
        clause = "ADDITIVE -> SEQUENCE-LEVEL (view a)"
        verdict = (f"Additivity fires: median pair/single ratio {med:.3f} "
                   f"within [{BAR_ADD[0]}, {BAR_ADD[1]}] "
                   f"(cluster CI {_fmt_ci(ci_med)}, n={n_qual}). The anchor "
                   f"is SEQUENCE-LEVEL — a basin property of the run "
                   f"trajectory, not of entries or pairs (T051 view a). P3 "
                   f"proceeds to run-level descriptors (run-PCA distance to "
                   f"attractor, drift entropy).")
    else:
        clause = "TEXTURE (no registered bar fires cleanly)"
        verdict = (f"Median pair/single ratio {med:.3f} (cluster CI "
                   f"{_fmt_ci(ci_med)}, n={n_qual} qualifying) lands outside "
                   f"both registered windows ([0.8,1.2] additive; >= {BAR_INTERACT} "
                   f"with CI excluding 1.0 for interaction). Honest texture: "
                   f"see the distribution — quartiles "
                   f"{np.percentile(ratios, [25, 75]).round(3).tolist()} if "
                   f"defined; synergy tally {n_syn}, floor-null {n_floor}, "
                   f"negative-denominator {n_den_neg}, aggregate "
                   f"{agg_ratio:+.2f}.")
    log(f"READOUT: median ratio {med:.3f} CI {_fmt_ci(ci_med)} | "
        f"Spearman(ratio, d) {sp:+.3f} CI {_fmt_ci(ci_sp)} | "
        f"qual {n_qual}, synergy {n_syn}, floor {n_floor}, den-neg {n_den_neg}")
    log(f"REGISTERED DECISION [{clause}]: {verdict}")

    # ---------------------------------------------------------------- metrics
    metrics = dict(
        experiment="e088_pair_anchor",
        purpose="T051's registered pair-level anchor discriminator: removal "
                "cost of anchor entry PAIRS vs the SUM of singles (4-arm "
                "matched-stream factorial — control / s1 / s2 / pair — the "
                "e085 dynamic-removal instrument verbatim, extended to "
                "two-column removal at a shared intervention time). "
                "Super-additivity => the anchor lives in entry-pair "
                "interactions (view b); additivity => sequence-level basin "
                "property (view a, P3 proceeds to run-level descriptors).",
        started=started, wall_s=elapsed(), threads=THREADS, cpu_only=True,
        net=dict(ckpt=str(CKPT), arch=dict(n_layer=4, n_head=4, n_embd=128,
                                           block_size=T_TOTAL, vocab=65),
                 params=n_params, val_ce=val_ce, val_ce_e053c=E053C_VAL_CE),
        seeds=dict(corpus=1337, prompts=SEED_PROMPT, sampling=SEED_SAMPLE,
                   pair_selection=SEED_PAIR, continuation=SEED_CONT,
                   bootstrap=0),
        protocol=dict(
            B=B, prompt_tokens=PROMPT_TOK, t_total=T_TOTAL, temp=TEMP,
            topk=TOPK, tail_window=list(KEY_T), boot_n=BOOT_N,
            threads=THREADS,
            anchor_band=list(ANCHOR_POS), p1_max=P1_MAX,
            strata=dict(per_run=PER_RUN, dist_adjacent=list(DIST_ADJ),
                        decades=list(DEC_K), dist_cross=list(DIST_CROSS),
                        rule="all pairs within-run; p1 ~ U[64, min(350, "
                             "414-d)]; same-decade = two ages >= 2 apart in "
                             "one decade k ~ U{17..43}; duplicate triples "
                             "redrawn"),
            timing=dict(t_int="p1 + 97 (the EARLIER entry's age-crossing; "
                              "e075/e085 verbatim for entry-1; entry-2 "
                              "removed earlier than its own crossing — its "
                              "single arm shares the SAME t_int so the "
                              "factorial is matched at one intervention time)",
                        first_affected_sample="p1 + 98",
                        exposure="414 - p1 (>= 64 by the p1 <= 350 floor)"),
            arms="4 per group, SAME continuation seed (matched streams): "
                 "control / zero col p1 / zero col p2 / zero both",
            outcome="clean-judge tail CE (448..511) of arm stream minus the "
                    "matched control stream; RATIO = c_pair / (c_s1 + c_s2) "
                    "iff s1+s2 > 0.10 nats (registered floor)",
            bars=dict(interaction="median ratio >= 1.5 AND cluster CI of the "
                                  "median excludes 1.0",
                      additive="median ratio within [0.8, 1.2]",
                      indeterminate=f"n_qualifying < {MIN_QUAL}",
                      order="indeterminate -> interaction -> additive -> "
                            "honest texture",
                      secondary="Spearman(ratio, distance) + per-class "
                                "medians; view-b localization predicts a "
                                "negative slope",
                      synergy=f"s1+s2 <= {DEN_FLOOR} AND c_pair >= "
                              f"{SYN_PAIR_MIN} (registered texture tally)"),
            registered_numbers=dict(den_floor=DEN_FLOOR,
                                    syn_pair_min=SYN_PAIR_MIN,
                                    min_qual=MIN_QUAL,
                                    bar_interact=BAR_INTERACT,
                                    bar_add=list(BAR_ADD))),
        gates=gates,
        control_run=dict(clean_judge_tail_ce=cj1.tolist()),
        summary=dict(
            n_pairs=len(pairs), n_groups=len(groups), n_qualifying=n_qual,
            median_ratio=med, median_ratio_ci=ci_med,
            ratio_quartiles=(np.percentile(ratios, [25, 50, 75]).tolist()
                             if n_qual else None),
            spearman_ratio_d=sp, spearman_ratio_d_ci=ci_sp,
            pearson_ratio_d=pe,
            class_medians=class_med,
            tallies=dict(pure_synergy=n_syn, floor_null=n_floor,
                         negative_denominator=n_den_neg,
                         facilitation_single=n_fac_single,
                         pair_eq_s1=n_pair_eq_s1, pair_eq_s2=n_pair_eq_s2,
                         diverged_pair=n_div_any, diverged_s1=n_div_s1,
                         diverged_s2=n_div_s2),
            aggregate=dict(total_pair_cost=tot_pair, total_single_cost=tot_sing,
                           aggregate_ratio=agg_ratio),
            mean_costs=dict(c_s1=float(np.mean([p["c_s1"] for p in pairs])),
                            c_s2=float(np.mean([p["c_s2"] for p in pairs])),
                            c_pair=float(np.mean([p["c_pair"] for p in pairs])),
                            c_s1_qual=float(np.mean([p["c_s1"] for p in qual]))
                            if qual else None,
                            c_s2_qual=float(np.mean([p["c_s2"] for p in qual]))
                            if qual else None)),
        pairs=pairs,
        registered_decision=dict(clause=clause, verdict=verdict,
                                 numbers=dict(n_qualifying=n_qual,
                                              median_ratio=med,
                                              median_ratio_ci=ci_med,
                                              bar_interact=BAR_INTERACT,
                                              bar_add=list(BAR_ADD),
                                              spearman_ratio_d=sp,
                                              spearman_ci=ci_sp,
                                              pure_synergy=n_syn,
                                              aggregate_ratio=agg_ratio)),
    )
    save_json(out_dir / "metrics.json", metrics)
    log("metrics.json written")

    plot(out_dir / "pair_anchor.png", metrics)
    log(f"plot written; total wall {elapsed():.0f}s")


# ---------------------------------------------------------------------- plot


def _wrap(text: str, width: int):
    import textwrap
    return textwrap.wrap(text, width=width)


def plot(path: Path, M: dict):
    dec = M["registered_decision"]
    S = M["summary"]
    pairs = M["pairs"]
    qual = [p for p in pairs if p["qualifies"]]
    cls_col = dict(adjacent="tab:red", same_decade="tab:purple",
                   cross_decade="tab:blue")

    fig, axes = plt.subplots(2, 3, figsize=(24, 11))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    # ---- panel 1: THE factorial scatter — pair cost vs sum of singles
    xs_all = [p["denom"] for p in pairs]
    ys_all = [p["c_pair"] for p in pairs]
    lim = max(0.6, float(np.nanmax(np.abs(xs_all + ys_all))) * 1.08)
    for p in pairs:
        ax1.scatter(p["denom"], p["c_pair"], s=46,
                    color=cls_col[p["cls"]], alpha=0.75,
                    edgecolors=("k" if p["qualifies"] else "none"), lw=0.8)
    xx = np.linspace(-lim, lim, 50)
    for mult, lb, st in ((1.0, "y = x (exact additivity)", "-"),
                         (1.5, "y = 1.5x (interaction bar)", "--"),
                         (0.8, "y = 0.8x (additive band)", ":")):
        ax1.plot(xx, mult * xx, st, color=("k" if mult == 1 else "tab:green"
                                           if mult == 1.5 else "gray"),
                 lw=1.1, label=lb)
    ax1.plot(xx, 1.2 * xx, ":", color="gray", lw=1.1, label="y = 1.2x")
    ax1.axvline(DEN_FLOOR, color="tab:orange", lw=1.0, ls="-.",
                label=f"denominator floor {DEN_FLOOR}")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.axvline(0, color="k", lw=0.5)
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_xlabel("sum of single-removal costs  c(p1) + c(p2)  (nats)")
    ax1.set_ylabel("pair-removal cost  c(p1, p2)  (nats)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_title("E088-1 — THE pair factorial: pair cost vs sum of singles "
                  "(clean-judge tail; black edge = qualifying)", fontsize=10)

    # ---- panel 2: ratio distribution
    if qual:
        rs = [p["ratio"] for p in qual]
        ax2.hist(rs, bins=min(16, max(6, len(rs) // 2)), color="tab:gray",
                 alpha=0.8, edgecolor="k")
        ax2.axvline(np.median(rs), color="k", lw=2.0,
                    label=f"median {np.median(rs):.3f} CI "
                          f"{_fmt_ci(S['median_ratio_ci'])}")
    ax2.axvline(1.0, color="k", lw=1.0, ls="-", label="1.0 (additive)")
    ax2.axvline(BAR_ADD[0], color="tab:green", lw=1.2, ls=":")
    ax2.axvline(BAR_ADD[1], color="tab:green", lw=1.2, ls=":",
                label=f"additive band [{BAR_ADD[0]},{BAR_ADD[1]}]")
    ax2.axvline(BAR_INTERACT, color="tab:red", lw=1.4, ls="--",
                label=f"interaction bar {BAR_INTERACT}")
    ax2.set_xlabel("super-additivity ratio  c(p1,p2) / (c(p1)+c(p2))")
    ax2.set_ylabel(f"pairs (n={len(qual)} qualifying of {len(pairs)})")
    ax2.legend(fontsize=8)
    ax2.set_title("E088-2 — ratio distribution + REGISTERED bars", fontsize=10)

    # ---- panel 3: ratio vs pair distance
    if qual:
        for p in qual:
            ax3.scatter(p["d"], p["ratio"], s=52, color=cls_col[p["cls"]],
                        alpha=0.8)
        for cls, col in cls_col.items():
            cm = S["class_medians"][cls]["median"]
            if np.isfinite(cm):
                dd = np.mean([p["d"] for p in qual if p["cls"] == cls])
                ax3.scatter(dd, cm, marker="D", s=130, color=col,
                            edgecolors="k", zorder=5,
                            label=f"{cls} median {cm:.2f}")
        ax3.axhline(1.0, color="k", lw=0.9)
        ax3.axhline(BAR_INTERACT, color="tab:red", lw=1.0, ls="--")
        ax3.set_xlabel("pair distance d = p2 - p1 (positions)")
        ax3.set_ylabel("ratio")
        ax3.set_title("E088-3 — interaction vs distance: Spearman "
                      f"{S['spearman_ratio_d']:+.3f} CI "
                      f"{_fmt_ci(S['spearman_ratio_d_ci'])} "
                      f"(view-b localization predicts negative)", fontsize=9.5)
        ax3.legend(fontsize=8)

    # ---- panel 4: cost components per pair
    for k, col, off in (("c_s1", "tab:red", -0.22), ("c_s2", "tab:purple", 0.0),
                        ("c_pair", "tab:cyan", 0.22)):
        vals = [p[k] for p in pairs]
        ax4.scatter(np.arange(len(pairs)) + off, vals, s=16, color=col,
                    alpha=0.7, label=k)
        ax4.axhline(np.mean(vals), color=col, lw=1.0, ls=":")
    ax4.axhline(0, color="k", lw=0.8)
    ax4.set_xlabel("pair index (0..39)")
    ax4.set_ylabel("clean-judge tail cost (nats)")
    mc = S["mean_costs"]
    ax4.set_title("E088-4 — cost components: mean c_s1 "
                  f"{mc['c_s1']:+.4f} | c_s2 {mc['c_s2']:+.4f} | c_pair "
                  f"{mc['c_pair']:+.4f}", fontsize=10)
    ax4.legend(fontsize=8, loc="upper left")

    # ---- panel 5: tallies
    t = S["tallies"]
    cats = ["qualifying\n(den > floor)", "floor-null\n(both ~free)",
            f"pure synergy\n(pair >= {SYN_PAIR_MIN})", "negative\ndenominator"]
    vals = [S["n_qualifying"], t["floor_null"], t["pure_synergy"],
            t["negative_denominator"]]
    bars = ax5.bar(cats, vals, color=["tab:gray", "tab:green", "tab:red",
                                      "tab:orange"], alpha=0.85)
    ax5.bar_label(bars, fontsize=10)
    ag = S["aggregate"]
    ax5.set_ylabel("pairs")
    ax5.set_title("E088-5 — tallies | aggregate ratio "
                  f"sum(c_pair)/sum(singles) = {ag['aggregate_ratio']:+.2f} "
                  f"({ag['total_pair_cost']:+.2f}/{ag['total_single_cost']:+.2f} nats)",
                  fontsize=9.5)

    # ---- panel 6: verdict text
    ax6.axis("off")
    n = dec["numbers"]
    lines = [
        "REGISTERED (frozen in T051):",
        f"  median ratio >= {BAR_INTERACT} AND cluster CI excl 1.0 => "
        f"INTERACTION (view b)",
        f"  median ratio in [{BAR_ADD[0]},{BAR_ADD[1]}] => ADDITIVE => "
        f"SEQUENCE-LEVEL (view a)",
        f"  n_qualifying < {MIN_QUAL} => INDETERMINATE; else honest texture",
        "",
        f"RESULT: n_qual {n['n_qualifying']}/40 | median ratio "
        f"{n['median_ratio']:.3f} CI {_fmt_ci(n['median_ratio_ci'])}",
        f"  Spearman(ratio, d) {n['spearman_ratio_d']:+.3f} "
        f"CI {_fmt_ci(n['spearman_ci'])}",
        "  class medians: " + ", ".join(
            f"{c} {S['class_medians'][c]['median']:+.2f}"
            f"(n={S['class_medians'][c]['n']})"
            for c in ("adjacent", "same_decade", "cross_decade")),
        f"  synergy {t['pure_synergy']} | floor-null {t['floor_null']} | "
        f"den-neg {t['negative_denominator']}",
        f"  aggregate ratio {n['aggregate_ratio']:+.2f} | pair stream == s1 "
        f"in {t['pair_eq_s1']}, == s2 in {t['pair_eq_s2']}",
        "",
        f"DECISION [{dec['clause']}]:",
    ] + [f"  {wd}" for wd in _wrap(dec["verdict"], 96)]
    ax6.text(0.02, 0.97, "E088 — T051 pair-level anchor discriminator",
             fontsize=13, weight="bold", va="top")
    for i, tx in enumerate(lines):
        ax6.text(0.02, 0.93 - i * 0.034, tx, fontsize=8.6, va="top",
                 family="monospace")

    fig.suptitle("E088 — pair-level anchor probe: pair removal cost vs sum of "
                 f"singles | clause: {dec['clause']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
