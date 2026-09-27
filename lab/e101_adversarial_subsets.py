"""E101 — the ADVERSARIAL falsification of the mass-action law (Rule-11
registered; scratch/next_wave_programs.md item-adjacent + T051's e097
amendment). CPU-only.

THE KILL-ATTEMPT: the law (e089, confirmed in e097) says removal cost is a
function of HOW MANY entries, not WHICH. Adversarial subset selection should
fail to beat the mass ladder if the law is iron. T051's E097 amendment
(refined law: mass dominates, recency modulates 2-3x, sink-side cheapest)
gives the selection priors their best shot: the recency gradient and the
readership (importance) prior.

DESIGN (registered here BEFORE compute; bars frozen from the tasking).
Battery: the e053c ctx-512 net (873,472 params, val CE 1.5227), the seed-202
8-draw prompts, B=8, the seed-7 control free run (the e075..e097 A-none
convention; G3 gated against e080's stored none arm) run WITH e085's
attention collection (generate_collect — logits math untouched; the stream
must remain bitwise identical, G3 checks). Anchor band: positions 64..414 =
final-frame ages 97..447 (self-generated), verbatim e089/e097.

ARMS at k in {64, 128}:
  (a) RANDOM uniform — the control band: 10 draws per k, fresh rng
      default_rng(seed=101), the e089 rule VERBATIM (choice from the band,
      replace=False, rejected & redrawn while min > 350; rejections
      tallied). Run rule: run = (arm_index*10 + draw) % 8.
  (b) GREEDY-RECENCY — the k NEWEST REMOVABLE entries: positions
      413-k+1..413 (k=64 -> 350..413, exposure exactly 64; k=128 ->
      286..413). Position 414 (the single truly-newest entry) is
      unremovable-by-construction (crosses at 511, beyond the last decode
      510 — e089's documented edge case; it contributes nothing in ANY arm,
      so 'newest removable' = 413 downward is the honest operationalization
      and maximizes k_eff: 64/64 and 128/128 fire). The subset is
      DETERMINISTIC -> the arm's draws are the 8 run-streams (n=8, one draw
      per run; documented — no draw randomness exists to sample).
  (c) TOP-READERSHIP — per run, the k band positions with the highest
      accumulated attention received over the control free run (e085/e070's
      instrument: per-decode-step attention probs, mean over heads, summed
      over the 4 layers, accumulated over all decode steps 64..510). This is
      the eviction literature's importance heuristic given GOD-MODE access
      to the full clean-run statistics (oracle-grade, maximally favorable
      to selection — the right adversarial direction for a kill attempt).
      Deterministic per (run, k) -> 8 draws (one per run). Contingency
      (registered): if a run's k=64 top-k had min > 350 (only k=64 can,
      arithmetically), the arm keeps its verbatim top-k with the reduced
      exposure documented (assert only min <= 413); k=128 cannot (the top
      128 of 351 must include positions <= 350).
  (d) ADVERSARIAL-GREEDY (REPORT-ONLY, k=16, one run): iteratively remove
      the single most-damaging entry per step — a small greedy oracle.
      Pool (documented): the run's top-16 readership entries UNION the 16
      newest removable (398..413) UNION 8 fresh rng(seed=1010) band entries;
      position 414 excluded (unremovable); pool-min forced <= 350 by
      extending downward if needed. ANCHORED greedy (batchability
      restriction, documented): every candidate set contains the pool-min
      m0 (all sets share min = m0 -> one batched group, one control);
      entries below m0 are unreachable — the anchor makes the oracle
      conservative, never generous. 15 greedy steps; at each step all pool
      rows re-evaluated batched (rows whose candidate was absorbed
      re-measure the current set — internal replication), the winner is the
      argmax-cost row. Run choice (adversarial direction): the run with the
      MAX mean cost in the random k=64 arm (the most damage-prone stream).
      Report-only — no registered bar consumes it.

INSTRUMENT: the e089/e097 rig VERBATIM, zero adaptations — per-entry
age-97-crossing V-zero (entry p zeroed immediately before the decode
processing position p+97), T_int = min(S)+97 teacher-forced, matched-stream
continuation with shared row-order generator (SEED_CONT=101000 + min(S)),
clean-net final-64 tail judgment 448..511. cost = clean-judge tail CE of the
removal stream minus the matched control stream.

REGISTERED BARS (frozen from the tasking; cluster bootstrap CIs over the 8
runs; ratio denominator guard verbatim e097: bootstrap resamples with
denominator mean <= 0.05 nats discarded):
  BAR-A — SELECTION BEATS MASS (the law falls): any adversarial arm
     (recency or readership; the greedy oracle is report-only) at k=64 with
     mean cost EXCEEDING the random band's k=128 cost, CI-backed — the
     ratio arm/random(k=128) > 1.0 AND its cluster-CI lower bound > 1.0.
     A point-estimate exceedance without CI support does NOT fire the bar
     (it feeds MIXED).
  BAR-B — THE LAW STANDS with the recency rider (mass first-order,
     selection second-order): (i) EVERY adversarial arm mean at k=64 below
     the random k=128 mean (point estimates; CI overlap reported) AND every
     k=128 adversarial arm below the random k=128 mean — selection at half
     the mass buys less than doubling the mass does; AND (ii) greedy-recency
     within 1.5x of random at the SAME k at BOTH k=64 and k=128 (point
     estimates). [The k=128 'k+1' reference for the k=128 arms is e089's
     stored k=200 random level (+2.153, same battery, seed-89 rng) —
     descriptive cross-reference only, since e101 runs no k=200 arm.]
  Anything else -> MIXED/TEXTURE (per-arm table, honest).
  Indeterminate guard: random(k=128) mean <= 0.05 -> ratio bars undefined.
SECONDARY/texture: per-arm ratios vs same-k random with CIs (the e097-style
table); k_eff; exposure; first stream divergence; readership-recency
overlap of the selection priors; greedy path vs the e089 k=16 random level.

Gates: G1 val CE vs e053c 1.5227 +-0.02; G2 params 873472; G3 control
battery identity vs e080's stored none arm (clean-judge per-seq, hard bar
1e-4, bitwise flag; e088/e089/e097's scoped G3); G4 dynamic-instrument
identity per group (prefix identity through the teacher-forced token; fired
removal columns exactly 0; non-removed and control columns live; matched
seed across arms; each crossing fired at most once) + the greedy group's
own checks.

Run:     python lab/e101_adversarial_subsets.py
Outputs: runs/e101/metrics.json + runs/e101/adversarial_subsets.png
Envelope: NO training, NO new automations; CPU-only (CUDA_VISIBLE_DEVICES=-1),
8 threads (T050), single step, target ~10-20 min.
No NOTES/THINKING/QUEUE/STATE edits; no commit (the dispatcher owns those).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # task spec: CPU-only (pre-torch)
import torch

# this torch build reports is_available()==True even with an empty
# CUDA_VISIBLE_DEVICES; force CPU so common.DEVICE=="cpu" (e053b..e097)
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
SEED_PROMPT, SEED_SAMPLE = 202, 7         # e053b/e053c/e069..e097 seeds
SEED_SUB = 101                            # e101: main-arm subset rng (fresh)
SEED_GREEDY = 1010                        # e101: greedy pool's 8 random entries
SEED_CONT = 101000                        # e101: continuation seeds (+ min)
B = 8
N_PROMPTS = 8
BOOT_N = 1000
THREADS = 8                               # T050: 12 spin-thrashes the box
torch.set_num_threads(THREADS)

KEY_T = (T_TOTAL - 64, T_TOTAL - 1)       # judged tail window (448, 511)
G = T_TOTAL - PROMPT_TOK                  # 448 generation steps

# ---- the anchor band (final-frame ages; age of position p = 511 - p) ---------
ANCHOR_POS = (64, 414)                    # ages 97..447 (self-generated)
P1_MAX = 350                              # full-tail-exposure: min+98 <= 448
BAND = np.arange(ANCHOR_POS[0], ANCHOR_POS[1] + 1)   # 351 positions
LAST_REMOVABLE = 413                      # 414 crosses at 511 > last decode 510
GREEDY_K = 16                             # report-only greedy oracle size

# ---- arms (rng consumption + bookkeeping order) ------------------------------
# (key, k, kind, n_draws) — kinds: rand (10 draws, e089 rule) / recency /
# reader (deterministic per run: n=8, one draw per run)
ARM_SPECS = [
    ("k64_rand", 64, "rand", 10),
    ("k128_rand", 128, "rand", 10),
    ("k64_recency", 64, "recency", 8),
    ("k128_recency", 128, "recency", 8),
    ("k64_reader", 64, "reader", 8),
    ("k128_reader", 128, "reader", 8),
]
ARM_KEYS = [s[0] for s in ARM_SPECS]
ADV_KEYS = ["k64_recency", "k64_reader", "k128_recency", "k128_reader"]

# ---- REGISTERED decision numbers (frozen, docstring verbatim) ----------------
RIDER_X = 1.5                             # recency within 1.5x random same-k
RATIO_DEN_FLOOR = 0.05                    # e097's denominator-stability guard
GREEDY_POOL_TOP = 16                      # readership + newest block sizes
GREEDY_POOL_RAND = 8

# ---- reference numbers (protocol-identity gates) -----------------------------
CKPT = REPO / "runs" / "checkpoints" / "e053c_ctx512.pt"
E053C_VAL_CE = 1.5226792494455974         # the net being interrogated
E080_METRICS = REPO / "runs" / "e080" / "metrics.json"
E089_METRICS = REPO / "runs" / "e089" / "metrics.json"
E097_METRICS = REPO / "runs" / "e097" / "metrics.json"
T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# ------------------------------------------------- machinery (VERBATIM e097)

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
    [VERBATIM e075/e080/e085/e088/e089/e097]"""
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
def decode_step_batch(net: TinyGPT, toks: torch.Tensor, pos: int, kv: list,
                      collect_att: bool = False):
    """Batched incremental decode: (B,) tokens at position pos -> (B, V);
    with collect_att=True also returns the mean-over-heads attention mass
    over the cache (B, t+1), summed over layers by the caller (the e070/e085
    instrument accumulated). The logits math is untouched by collection.
    [e085's decode_step_batch verbatim]"""
    Bb = toks.shape[0]
    x = net.wte(toks) + net.wpe(torch.full((Bb,), pos))
    H = net.cfg.n_head
    m = None
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
        if collect_att:
            pm = probs[:, :, 0, :].mean(1)                      # (B, t+1)
            m = pm if m is None else m + pm
        y = (probs @ v).transpose(1, 2).reshape(Bb, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
    logits = net.lm_head(net.ln_f(x))
    return (logits, m) if collect_att else logits


@torch.no_grad()
def generate_collect(net: TinyGPT, prompts, gen: torch.Generator):
    """Free-run 64->512 for B sequences (e075..e097 A-none VERBATIM stream
    math) while accumulating per-position attention-received mass — e085's
    generate_collect verbatim: mass_all (B, T) = mean over heads, summed
    over the 4 layers, accumulated over all decode steps. This is the
    control run AND the readership instrument in one pass."""
    Bb = len(prompts)
    idx = torch.stack(prompts)
    logits, kv = prefill_batch(net, idx)
    mass_all = np.zeros((Bb, T_TOTAL), float)
    for g in range(G):
        t = PROMPT_TOK + g
        toks = torch.zeros(Bb, dtype=torch.long)
        for j in range(Bb):
            tok, _ce = sample_and_ce(logits[j], gen)
            toks[j] = tok
        idx = torch.cat([idx, toks[:, None]], 1)
        if t < T_TOTAL - 1:
            logits, m = decode_step_batch(net, toks, t, kv, collect_att=True)
            mass_all[:, :t + 1] += m.numpy()                     # (B, t+1)
    return dict(idx=idx, kv=kv, mass_all=mass_all)


@torch.no_grad()
def run_continuation(net: TinyGPT, prefix: torch.Tensor, forced_tok: torch.Tensor,
                     forced_pos: int, seed: int, remove=None):
    """Crossing-time progressive entry-removal continuation (and its matched
    control when remove=None). [VERBATIM e089/e097:]

    remove = list of (row, col): every listed entry p is V-zeroed immediately
    BEFORE the decode step that processes position p+97 (e085's age-97-crossing
    timing applied to sets). The earliest entry's crossing is the forced decode
    itself (T_int = min+97), so the token at forced_pos is teacher-forced and
    the first affected sample is forced_pos+1. Entries whose crossing falls at
    position 511 (p = 414) are NEVER zeroed inside the window:
    unremovable-by-construction (exposure 0), returned unfired for honest
    tallying. Free-run from forced_pos+1 with the shared row-order generator —
    control and removal arms share the seed, so streams are row-by-row matched
    until sampled divergence.

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
    Returns dict window -> (R,) mean CE per row. [VERBATIM e085/e097]"""
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
    [VERBATIM e088/e089/e097's cluster_boot]"""
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
    out_dir = run_dir("e101")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gates = dict(cpu_only=True, threads=THREADS, smoke=False)

    # ---- battery: rebuild corpus + net EXACTLY as e053c/e069..e097 did
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

    # ---- control free run + READERSHIP collection (e085's instrument)
    log("control battery: seed-7 free run with attention collection "
        "(e085 generate_collect — stream math untouched)")
    gen7 = torch.Generator().manual_seed(SEED_SAMPLE)
    run1 = generate_collect(net, prompts8, gen7)
    idx1 = run1["idx"]
    mass_all = run1["mass_all"]
    log(f"  control run done ({T_TOTAL} positions x {B} rows); readership "
        f"mass accumulated (mean over heads, summed over 4 layers)")

    # ---- G3 (scope: control battery) vs e080's stored A-none arm
    cj1 = judge_windows(manual_all_logits(net, idx1), idx1, [KEY_T])[KEY_T]
    g3 = dict(ref_file=str(E080_METRICS), ok=False,
              note="e101 has no static arm; G3 scope = control-battery "
                   "identity (clean-judge per-seq vs e080 stored none arm) "
                   "[e088/e089/e097's scoped G3 verbatim] — this ALSO gates "
                   "that collect_att did not perturb the stream")
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

    # ---- cross-experiment references (same battery: net/prompts/seed-7 run)
    e089_ref, e097_ref = {}, {}
    import json as _json
    if E089_METRICS.exists():
        with open(E089_METRICS) as f:
            e089 = _json.load(f)
        for k in (16, 64, 128, 200):
            pk = e089["summary"]["per_k"][str(k)]
            e089_ref[str(k)] = dict(mean=pk["mean"], ci=pk["ci"])
        log("e089 references (uniform, seed 89): " + " | ".join(
            f"k={k} {e089_ref[str(k)]['mean']:+.3f}" for k in (16, 64, 128, 200)))
    if E097_METRICS.exists():
        with open(E097_METRICS) as f:
            e097 = _json.load(f)
        for key in ("k64_unif", "k128_unif", "k64_new", "k128_new"):
            st97 = e097["summary"]["per_arm"][key]
            e097_ref[key] = dict(mean=st97["mean"], ci=st97["ci"])
        log("e097 references (seed 97): " + " | ".join(
            f"{key} {e097_ref[key]['mean']:+.3f}" for key in e097_ref))

    # ================================================== readership ranking
    band_lo, band_hi = ANCHOR_POS
    reader_top = {}                        # k -> list of 8 arrays (per run)
    for k in (64, 128):
        tops = []
        for r in range(B):
            scores = mass_all[r, band_lo:band_hi + 1]
            order = np.argsort(-scores, kind="mergesort")
            tops.append(np.sort(band_lo + order[:k]))
        reader_top[k] = tops
    for k in (64, 128):
        mins = [int(t[0]) for t in reader_top[k]]
        ov_rec = [len(np.intersect1d(reader_top[k][r],
                                     np.arange(LAST_REMOVABLE - k + 1,
                                               LAST_REMOVABLE + 1)))
                  for r in range(B)]
        spr = [spearman(mass_all[r, band_lo:band_hi + 1],
                        np.arange(band_lo, band_hi + 1)) for r in range(B)]
        log(f"readership top-{k}: min(S) in [{min(mins)}..{max(mins)}]; "
            f"overlap with recency block {np.mean(ov_rec):.1f}/{k}; "
            f"Spearman(mass, pos) per run mean {np.nanmean(spr):+.3f}")
    texture_reader = dict(
        min_top64=[int(t[0]) for t in reader_top[64]],
        min_top128=[int(t[0]) for t in reader_top[128]],
        overlap_with_recency_block={str(k): [len(np.intersect1d(
            reader_top[k][r],
            np.arange(LAST_REMOVABLE - k + 1, LAST_REMOVABLE + 1)))
            for r in range(B)] for k in (64, 128)},
        spearman_mass_pos_per_run={str(k): [spearman(
            mass_all[r, band_lo:band_hi + 1],
            np.arange(band_lo, band_hi + 1)) for r in range(B)]
            for k in (64, 128)},
    )

    # ================================================== subset construction
    log(f"subset construction: rng seed {SEED_SUB} for the random arms "
        f"(e089 rule, reject min > {P1_MAX}); recency = newest removable "
        f"block; reader = per-run attention top-k (deterministic -> 8 draws)")
    rng = np.random.default_rng(SEED_SUB)
    draws = []
    n_reject = 0
    for ai, (key, k, kind, n) in enumerate(ARM_SPECS):
        for d in range(n):
            if kind == "rand":
                n_try = 0
                while True:
                    cols = np.sort(rng.choice(BAND, size=k, replace=False))
                    n_try += 1
                    if int(cols[0]) <= P1_MAX:
                        break
                    n_reject += 1
                run = (ai * 10 + d) % B
            elif kind == "recency":
                cols = np.arange(LAST_REMOVABLE - k + 1, LAST_REMOVABLE + 1)
                run = d
                n_try = 1
            else:                           # reader
                cols = reader_top[k][d].copy()
                run = d
                n_try = 1
            draws.append(dict(
                arm=key, k=k, kind=kind, d=d, run=int(run),
                cols=[int(c) for c in cols],
                min_p=int(cols[0]), max_p=int(cols[-1]),
                mean_p=float(cols.mean()),
                t_int=int(cols[0]) + 97,
                exposure=int(T_TOTAL - 1 - (int(cols[0]) + 98) + 1),
                seed=SEED_CONT + int(cols[0]),
                n_tries=n_try,
            ))
    assert len(draws) == sum(s[3] for s in ARM_SPECS) == 52
    for dr in draws:
        if dr["kind"] in ("rand", "recency"):
            assert dr["exposure"] >= 64, f"arm {dr['arm']} exposure < 64"
        else:
            assert dr["min_p"] <= LAST_REMOVABLE   # verbatim top-k contingency
    per_run_counts = np.bincount([dr["run"] for dr in draws], minlength=B)
    log(f"  {len(draws)} draws; random-arm rejections {n_reject}; draws/run "
        f"{per_run_counts.tolist()}")
    for key in ARM_KEYS:
        dd = [dr for dr in draws if dr["arm"] == key]
        log(f"  {key:>12}: n={len(dd)}, min(S) in "
            f"[{min(d['min_p'] for d in dd)}..{max(d['min_p'] for d in dd)}], "
            f"mean pos {np.mean([d['mean_p'] for d in dd]):.0f}, p=414 in "
            f"{sum(414 in d['cols'] for d in dd)}/{len(dd)} draws")

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

    # ================================================== greedy oracle (report-only)
    greedy = _greedy_oracle(net, idx1, mass_all, draws, gates)

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

    def _ratio_fn(num_key, den_key):
        def fn(gg):
            den = np.mean([p["cost"] for p in gg if p["arm"] == den_key])
            num = np.mean([p["cost"] for p in gg if p["arm"] == num_key])
            if len([p for p in gg if p["arm"] == den_key]) < 2:
                return None
            if len([p for p in gg if p["arm"] == num_key]) < 2:
                return None
            if den <= RATIO_DEN_FLOOR:      # registered denominator guard
                return None
            return float(num / den)
        return fn

    # same-k ratios (adversarial arm / random at the same k) — e097 style
    ratios_same_k = {}
    for key in ADV_KEYS:
        k = 64 if key.startswith("k64") else 128
        ci, nvalid = cluster_boot(draws, _ratio_fn(key, f"k{k}_rand"))
        m_a = arm_stats[key]["mean"]
        m_r = arm_stats[f"k{k}_rand"]["mean"]
        ratios_same_k[key] = dict(
            vs=f"k{k}_rand",
            val=float(m_a / m_r) if abs(m_r) > 1e-9 else float("nan"),
            ci=ci, n_valid_boot=nvalid)

    # kill ratios: adversarial arm at k=64 / random at k=128
    kill_ratios = {}
    for key in ("k64_recency", "k64_reader"):
        ci, nvalid = cluster_boot(draws, _ratio_fn(key, "k128_rand"))
        m_a = arm_stats[key]["mean"]
        m_r = arm_stats["k128_rand"]["mean"]
        kill_ratios[key] = dict(
            vs="k128_rand",
            val=float(m_a / m_r) if abs(m_r) > 1e-9 else float("nan"),
            ci=ci, n_valid_boot=nvalid)

    # descriptive texture: cost vs subset mean position (all arms)
    sp_all = spearman([dr["cost"] for dr in draws],
                      [dr["mean_p"] for dr in draws])

    # ---- REGISTERED bars (frozen; evaluated in priority order)
    m_r128 = arm_stats["k128_rand"]["mean"]
    ci_r128 = arm_stats["k128_rand"]["ci"]
    m_r64 = arm_stats["k64_rand"]["mean"]
    if not (m_r128 > RATIO_DEN_FLOOR):
        clause = "INDETERMINATE"
        verdict = (f"random(k=128) mean {m_r128:+.4f} <= {RATIO_DEN_FLOOR} — "
                   f"the band the kill bar needs did not separate from 0; no "
                   f"ratio is defined. Honest texture: "
                   + "; ".join(f"{key} {arm_stats[key]['mean']:+.3f}"
                               for key in ARM_KEYS) + ".")
        ev = {}
    else:
        ev = dict(band=dict(random_k64_mean=m_r64,
                            random_k128_mean=m_r128, random_k128_ci=ci_r128),
                  kill={}, law_stands=dict())
        # BAR-A: selection beats mass (CI-backed)
        killers, point_only = [], []
        for key in ("k64_recency", "k64_reader"):
            kr = kill_ratios[key]
            fires = bool(kr["val"] > 1.0 and kr["ci"][0] > 1.0
                         and kr["n_valid_boot"] > 0)
            ev["kill"][key] = dict(ratio=kr["val"], ci=kr["ci"],
                                   n_valid_boot=kr["n_valid_boot"],
                                   fires=fires,
                                   point_exceeds=bool(
                                       arm_stats[key]["mean"] > m_r128))
            if fires:
                killers.append(
                    f"{key}: {arm_stats[key]['mean']:+.3f} vs random k=128 "
                    f"{m_r128:+.3f} (ratio {kr['val']:.2f}, CI "
                    f"{_fmt_ci(kr['ci'])} excludes 1)")
            elif arm_stats[key]["mean"] > m_r128:
                point_only.append(
                    f"{key}: {arm_stats[key]['mean']:+.3f} > {m_r128:+.3f} "
                    f"point-wise but ratio CI {_fmt_ci(kr['ci'])} does not "
                    f"exclude 1")
        # BAR-B: the law stands with the recency rider
        all_below = all(arm_stats[key]["mean"] < m_r128 for key in ADV_KEYS)
        rec64_ratio = ratios_same_k["k64_recency"]["val"]
        rec128_ratio = ratios_same_k["k128_recency"]["val"]
        rider_ok = bool(np.isfinite(rec64_ratio) and np.isfinite(rec128_ratio)
                        and rec64_ratio <= RIDER_X and rec128_ratio <= RIDER_X)
        ev["law_stands"] = dict(
            all_adv_below_random128=dict(
                checks={key: dict(arm_mean=arm_stats[key]["mean"],
                                  below=bool(arm_stats[key]["mean"] < m_r128))
                        for key in ADV_KEYS},
                ok=bool(all_below)),
            recency_rider=dict(ratio_k64=rec64_ratio, ratio_k128=rec128_ratio,
                               within=f"<= {RIDER_X}x at both k", ok=rider_ok),
            k128_cross_ref=dict(
                note="descriptive: k=128 adversarial arms vs e089's stored "
                     "k=200 random level (the k+1 rung of the mass ladder)",
                e089_k200=e089_ref.get("200", {}).get("mean"),
                arm_means={key: arm_stats[key]["mean"]
                           for key in ("k128_recency", "k128_reader")}))
        if killers:
            clause = "SELECTION BEATS MASS (the law falls)"
            verdict = (f"BAR-A FIRES: " + "; ".join(killers) +
                       f" — an adversarially SELECTED {64}-entry subset "
                       f"out-damages DOUBLE the randomly-selected mass "
                       f"(random k=128 {m_r128:+.3f} {_fmt_ci(ci_r128)}). "
                       f"WHICH entries are removed matters after all; the "
                       f"mass-action law falls (mass was the "
                       f"first-order story only).")
        elif all_below and rider_ok:
            clause = "THE LAW STANDS (with the recency rider)"
            verdict = (f"BAR-B FIRES: every adversarial arm at k=64 stays "
                       f"below the random k=128 band {m_r128:+.3f} "
                       f"{_fmt_ci(ci_r128)} (recency {arm_stats['k64_recency']['mean']:+.3f} "
                       f"= {rec64_ratio:.2f}x random-at-64, readership "
                       f"{arm_stats['k64_reader']['mean']:+.3f} = "
                       f"{ratios_same_k['k64_reader']['val']:.2f}x) AND "
                       f"greedy-recency sits within {RIDER_X}x of "
                       f"random-at-same-k at both k (k=64: {rec64_ratio:.2f}x, "
                       f"k=128: {rec128_ratio:.2f}x). Selection modulates "
                       f"second-order; mass is first-order — selection at "
                       f"half the mass buys less than doubling the mass.")
        else:
            why = []
            if not all_below:
                viol = [f"{key} {arm_stats[key]['mean']:+.3f}"
                        for key in ADV_KEYS
                        if arm_stats[key]["mean"] >= m_r128]
                why.append("arms at/above the random k=128 band: "
                           + ", ".join(viol) + (f"; {len(point_only)} more "
                                                f"point-only exceedance(s)"
                                                if point_only else ""))
            elif point_only:
                why.append("; ".join(point_only))
            if not rider_ok:
                why.append(f"greedy-recency exceeds {RIDER_X}x "
                           f"random-at-same-k (k=64: {rec64_ratio:.2f}x, "
                           f"k=128: {rec128_ratio:.2f}x) — the e097 "
                           f"recency modulation is stronger than the rider "
                           f"clause allows")
            clause = "MIXED/TEXTURE"
            verdict = (f"No kill (no CI-backed exceedance of the random "
                       f"k=128 band {m_r128:+.3f} {_fmt_ci(ci_r128)}), but "
                       f"the law-stands clauses do not both fire either: "
                       + "; ".join(why)
                       + f". Honest reading with the arm table; the greedy "
                         f"oracle (report-only): k=16 cost "
                         f"{greedy['final_cost']:+.3f} on run "
                         f"{greedy['run']} (e089 random k=16 "
                         f"{e089_ref.get('16', {}).get('mean', float('nan')):+.3f}, "
                         f"e101 random k=64 {m_r64:+.3f}).")
    log("table: " + " | ".join(
        f"{key} {arm_stats[key]['mean']:+.3f}" for key in ARM_KEYS))
    log("same-k ratios: " + " | ".join(
        f"{key} {ratios_same_k[key]['val']:.2f} "
        f"{_fmt_ci(ratios_same_k[key]['ci'])}" for key in ADV_KEYS))
    log("kill ratios (arm64/random128): " + " | ".join(
        f"{key} {kill_ratios[key]['val']:.2f} "
        f"{_fmt_ci(kill_ratios[key]['ci'])}" for key in kill_ratios))
    log(f"greedy oracle (report-only): run {greedy['run']}, k=16 final cost "
        f"{greedy['final_cost']:+.3f} (path: "
        + ", ".join(f"{s}:{c:+.2f}" for s, c in greedy["path"]) + ")")
    log(f"texture: Spearman(cost, mean pos) all arms {sp_all:+.3f}")
    log(f"REGISTERED DECISION [{clause}]: {verdict}")

    # ---------------------------------------------------------------- metrics
    metrics = dict(
        experiment="e101_adversarial_subsets",
        purpose="Rule-11 registered adversarial falsification of the "
                "mass-action law (T051's e097 amendment): recency / readership "
                "/ greedy-oracle subset selection vs the random control band "
                "at k in {64, 128}, on the e089/e097 dynamic instrument "
                "VERBATIM (crossing-time progressive V-zero, matched-stream "
                "continuation, clean-net final-64 tail judgment). Kill bar: "
                "any adversarial k=64 arm CI-backed above the random k=128 "
                "band => selection beats mass. Law-stands bar: all arms below "
                "the random k=128 band AND greedy-recency within 1.5x "
                "random-at-same-k => mass first-order, selection "
                "second-order.",
        started=started, wall_s=elapsed(), threads=THREADS, cpu_only=True,
        net=dict(ckpt=str(CKPT), arch=dict(n_layer=4, n_head=4, n_embd=128,
                                           block_size=T_TOTAL, vocab=65),
                 params=n_params, val_ce=val_ce, val_ce_e053c=E053C_VAL_CE),
        seeds=dict(corpus=1337, prompts=SEED_PROMPT, sampling=SEED_SAMPLE,
                   subset_selection=SEED_SUB, greedy_pool=SEED_GREEDY,
                   continuation=SEED_CONT, bootstrap=0),
        protocol=dict(
            B=B, prompt_tokens=PROMPT_TOK, t_total=T_TOTAL, temp=TEMP,
            topk=TOPK, tail_window=list(KEY_T), boot_n=BOOT_N,
            threads=THREADS,
            anchor_band=list(ANCHOR_POS), band_n=int(len(BAND)),
            ks=[64, 128], greedy_k=GREEDY_K, arm_order=ARM_KEYS,
            arms=dict(
                random="10 draws per k; rng default_rng(101); e089 rule "
                       "VERBATIM (choice(band, k, replace=False), reject & "
                       "redraw while min > 350; rejections tallied); run = "
                       "(arm_index*10 + draw) % 8",
                greedy_recency="the k NEWEST REMOVABLE entries = positions "
                               "413-k+1..413 (414 unremovable-by-construction "
                               "in every arm — e089's edge case; k_eff = k); "
                               "deterministic -> 8 draws = the 8 run-streams",
                top_readership="per run, the k band positions with highest "
                               "accumulated attention received over the "
                               "control free run (e085/e070 instrument: mean "
                               "over heads, summed over layers, accumulated "
                               "over all decode steps — oracle-grade full-run "
                               "prior, maximally favorable to selection); "
                               "deterministic per run -> 8 draws",
                adversarial_greedy="REPORT-ONLY k=16 oracle: pool = top-16 "
                                   "readership U 16 newest U 8 rng(1010) "
                                   "random band entries; ANCHORED at pool-min "
                                   "(all sets share min -> one batched group; "
                                   "entries below pool-min unreachable — "
                                   "conservative); 15 greedy steps of "
                                   "batched argmax marginal damage; run = the "
                                   "random-k64-max run (adversarial "
                                   "direction)"),
            timing=dict(
                schedule="per-entry age-97-crossing V-zero (e089/e097 "
                         "VERBATIM, no adaptations)",
                t_int="min(S) + 97 (the earliest entry's crossing = the "
                      "forced decode)",
                exposure="414 - min(S) (>= 64 for random/recency arms by "
                         "construction; reader arms keep their verbatim "
                         "top-k with min documented)",
                unremovable="p = 414 crosses at 511, beyond the last decode "
                            "(510): never zeroed, stays live, k_eff = k-1 "
                            "(can only occur in random arms here)"),
            arms_instrument="2 per group, SAME continuation seed (matched "
                            "streams): control / crossing-time progressive "
                            "removal of the subset",
            outcome="clean-judge tail CE (448..511) of removal stream minus "
                    "matched control stream; arm MEAN over draws",
            bars=dict(
                kill_bar_A=f"any adversarial arm at k=64: mean > random "
                           f"k=128 mean AND ratio cluster-CI lower > 1.0 "
                           f"(resamples with denominator <= "
                           f"{RATIO_DEN_FLOOR} discarded) => SELECTION "
                           f"BEATS MASS, the law falls; point-only "
                           f"exceedance feeds MIXED",
                law_stands_bar_B=f"EVERY adversarial arm mean < random k=128 "
                                 f"mean AND greedy-recency ratio <= "
                                 f"{RIDER_X}x random-at-same-k at BOTH k=64 "
                                 f"and k=128 => THE LAW STANDS with the "
                                 f"recency rider (mass first-order, selection "
                                 f"second-order)",
                order="indeterminate (band <= 0.05) -> kill -> law-stands -> "
                      "mixed/texture",
                greedy="the k=16 adversarial-greedy oracle is REPORT-ONLY (no "
                       "registered bar); k=128 'k+1' cross-reference = e089's "
                       "stored k=200 random level (same battery, seed-89)"),
            registered_numbers=dict(rider_x=RIDER_X,
                                    ratio_den_floor=RATIO_DEN_FLOOR),
        ),
        gates=gates,
        control_run=dict(clean_judge_tail_ce=cj1.tolist()),
        references=dict(
            e089=dict(file=str(E089_METRICS), per_k=e089_ref,
                      note="e089's uniform-subset curve (seed 89) — same "
                           "battery (net/prompts/seed-7 control run); the "
                           "k=16 and k=200 rungs bracket e101's arms"),
            e097=dict(file=str(E097_METRICS), per_arm=e097_ref,
                      note="e097's stratified thirds (seed 97): the recency "
                           "gradient (new 2.2-2.8x) this experiment tries to "
                           "weaponize into a full selection arm")),
        summary=dict(
            n_draws=len(draws), n_groups=len(groups),
            uniform_rejections=n_reject,
            per_arm={key: {kk: vv for kk, vv in st.items() if kk != "draws"}
                     for key, st in arm_stats.items()},
            per_arm_draws={key: st["draws"] for key, st in arm_stats.items()},
            ratios_same_k=ratios_same_k,
            kill_ratios=kill_ratios,
            greedy=greedy,
            readership_texture=texture_reader,
            texture=dict(spearman_cost_meanpos_all=sp_all,
                         stream_identical_by_arm={
                             key: arm_stats[key]["stream_identical"]
                             for key in ARM_KEYS}),
        ),
        draws=[{k2: v for k2, v in dr.items() if k2 != "cols"} | dict(
            n_cols=len(dr["cols"])) for dr in draws],
        subsets={f"{dr['arm']}_d{dr['d']}": dr["cols"] for dr in draws},
        registered_decision=dict(clause=clause, verdict=verdict, evidence=ev),
    )
    save_json(out_dir / "metrics.json", metrics)
    log("metrics.json written")

    plot(out_dir / "adversarial_subsets.png", metrics)
    log(f"plot written; total wall {elapsed():.0f}s")


# --------------------------------------------------------- greedy oracle

def _greedy_oracle(net, idx1, mass_all, draws, gates):
    """REPORT-ONLY k=16 anchored greedy oracle (see module docstring, arm d).
    Batched: all pool rows share min = pool-min m0 -> one group, one control
    continuation with R rows (row-order generator matching, the e089/e097
    convention); each greedy step re-runs the removal arm over all R rows
    (rows whose candidate was already absorbed re-measure the current set —
    internal replication), the winner is the argmax-cost row."""
    # the adversarial run: max mean cost among the random k=64 arm's draws
    r64 = [dr for dr in draws if dr["arm"] == "k64_rand"]
    run_means = {r: float(np.mean([d["cost"] for d in r64 if d["run"] == r]))
                 for r in range(B)}
    r0 = int(max(run_means, key=run_means.get))
    log(f"greedy oracle: run {r0} (random-k64 per-run means "
        + ", ".join(f"{r}:{m:+.2f}" for r, m in run_means.items()) + ")")

    rng = np.random.default_rng(SEED_GREEDY)
    pool = set(int(p) for p in mass_all[r0, ANCHOR_POS[0]:ANCHOR_POS[1] + 1]
               .argsort()[::-1][:GREEDY_POOL_TOP] + ANCHOR_POS[0]) \
        | set(range(LAST_REMOVABLE - GREEDY_POOL_TOP + 1, LAST_REMOVABLE + 1)) \
        | set(int(p) for p in rng.choice(BAND, size=GREEDY_POOL_RAND,
                                         replace=False))
    pool.discard(414)                      # unremovable-by-construction
    while min(pool) > P1_MAX:              # registered pool-min guard
        pool.add(min(pool) - 1)
    m0 = min(pool)
    cands = sorted(pool - {m0})
    R = len(cands)
    log(f"greedy pool: {R} candidates + anchor m0={m0} (top-{GREEDY_POOL_TOP} "
        f"readership U newest {GREEDY_POOL_TOP} U {GREEDY_POOL_RAND} random)")

    forced_pos = m0 + 97
    rows = [r0] * R
    prefix = idx1[rows, :forced_pos]
    forced = idx1[rows, forced_pos]
    seed = SEED_CONT + m0
    idx_ctl, _kv_ctl, _f0 = run_continuation(net, prefix, forced,
                                             forced_pos, seed, remove=None)
    J_ctl = judge_windows(manual_all_logits(net, idx_ctl), idx_ctl,
                          [KEY_T])[KEY_T]
    S = [m0]
    path = []                              # (|S|, winning cost)
    picks = []
    g4g = dict(run=r0, n_rows=R, anchor=m0, ident_pre=True, fired_zero=True,
               col_live=True, n_steps=0)
    last_kv, last_idx_rm, last_best, last_fired = None, None, None, None
    for step in range(GREEDY_K - 1):
        rm = [(j, c) for j in range(R) for c in sorted(set(S + [cands[j]]))]
        idx_rm, kv_rm, fired = run_continuation(net, prefix, forced,
                                                forced_pos, seed, remove=rm)
        J_rm = judge_windows(manual_all_logits(net, idx_rm), idx_rm,
                             [KEY_T])[KEY_T]
        costs = J_rm - J_ctl
        # only rows whose candidate is not yet absorbed may win (absorbed
        # rows re-measure cost(S) — internal replication, never winners)
        eligible = [j for j in range(R) if cands[j] not in S]
        best = max(eligible, key=lambda j: costs[j])
        S.append(cands[best])
        picks.append(dict(step=step + 1, picked=cands[best],
                          cost=float(costs[best]),
                          margin=float(costs[best] - np.sort(costs)[-2])))
        path.append((len(S), float(costs[best])))
        last_kv, last_idx_rm = kv_rm, idx_rm
        last_best, last_fired = best, fired
        g4g["n_steps"] += 1
        log(f"  greedy step {step + 1}/{GREEDY_K - 1}: picked {cands[best]} "
            f"(set size {len(S)}, cost {costs[best]:+.3f}, margin over "
            f"runner-up {costs[best] - np.sort(costs)[-2]:+.3f})")
        # NOTE: rows stay FIXED across steps (batchability; see docstring)
    # ---- greedy G4 checks (final step's tensors). The final S only
    # materializes as the WINNING row's removal set (S_prev U its candidate)
    # — row 0 removed S_prev U {cands[0]} instead, so the gate must validate
    # the winning row `last_best`, whose columns are exactly the final S.
    ident = bool(torch.equal(last_idx_rm[:, :forced_pos + 1],
                             idx_ctl[:, :forced_pos + 1]))
    cz = all(bool(last_kv[li][1][last_best, :, c, :].abs().max().item() == 0.0)
             for li in range(len(last_kv)) for c in S)
    cl = True
    for li in range(len(last_kv)):
        vals = last_kv[li][1][last_best].abs().amax(dim=(0, 2))
        mask = np.ones(vals.shape[0], dtype=bool)
        mask[S] = False
        cl &= bool(vals[torch.from_numpy(mask)].min().item() > 0.0)
    g4g.update(ident_pre=ident, fired_zero=cz, col_live=cl,
               ok=bool(ident and cz and cl),
               final_subset=sorted(S), final_cost=path[-1][1],
               winning_row=int(last_best),
               n_fired_final_step=len({(r, c) for (r, c) in last_fired
                                       if c in S}))
    gates["G4g_greedy_oracle"] = g4g
    log(f"greedy G4g: prefix identity {ident}, final fired cols zero {cz}, "
        f"other cols live {cl} -> {'PASS' if g4g['ok'] else 'FAIL'}")
    return dict(
        run=r0, run_means_random_k64=run_means, pool=sorted(pool),
        anchor=m0, n_candidates=R, picks=picks, path=path,
        final_subset=sorted(S), final_cost=path[-1][1], gates=g4g,
        note="report-only; anchored at pool-min (batchability restriction — "
             "conservative: a free greedy could also drop the anchor); run = "
             "the random-k64-max run (the adversarial direction)")


# ---------------------------------------------------------------------- plot


def plot(path: Path, M: dict):
    dec = M["registered_decision"]
    S = M["summary"]
    per_arm = S["per_arm"]
    per_draws = S["per_arm_draws"]
    greedy = S["greedy"]
    rsk = S["ratios_same_k"]
    kr = S["kill_ratios"]
    e089 = M["references"]["e089"]["per_k"]

    arm_col = dict(k64_rand="tab:gray", k128_rand="tab:gray",
                   k64_recency="tab:blue", k128_recency="tab:blue",
                   k64_reader="tab:orange", k128_reader="tab:orange")

    fig, axes = plt.subplots(2, 3, figsize=(24, 11))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    # ---- panel 1: THE cost-vs-k plot with the random band
    ks = [64, 128]
    band_means = [per_arm[f"k{k}_rand"]["mean"] for k in ks]
    band_cis = [per_arm[f"k{k}_rand"]["ci"] for k in ks]
    xk = np.log2(np.array(ks, float))
    ax1.fill_between(xk, [c[0] for c in band_cis], [c[1] for c in band_cis],
                     color="tab:gray", alpha=0.30, zorder=1,
                     label="RANDOM control band (mean CI, seed 101)")
    ax1.plot(xk, band_means, "o-", color="tab:gray", lw=2.4, ms=10,
             zorder=4, label="RANDOM mean")
    for k in ks:
        ys = per_draws[f"k{k}_rand"]
        ax1.scatter(np.full(len(ys), math.log2(k)) + np.linspace(-0.05, 0.05,
                     len(ys)), ys, s=26, color="k", alpha=0.35, zorder=3)
    for arm, mk in (("recency", "D"), ("reader", "^")):
        for k in ks:
            st = per_arm[f"k{k}_{arm}"]
            ys = per_draws[f"k{k}_{arm}"]
            ax1.scatter(np.full(len(ys), math.log2(k)) - 0.09,
                        ys, s=26, color=arm_col[f"k{k}_{arm}"], alpha=0.45,
                        zorder=3, marker=mk)
            ax1.errorbar([math.log2(k) + 0.09], [st["mean"]],
                         yerr=[[max(0.0, st["mean"] - st["ci"][0])],
                               [max(0.0, st["ci"][1] - st["mean"])]],
                         fmt=mk, color=arm_col[f"k{k}_{arm}"], ms=13,
                         capsize=6, lw=2, zorder=5,
                         label=f"{arm.upper()} mean (n={st['n']})"
                         if k == 64 else None)
    # the kill bar: random k=128 level
    ax1.axhline(band_means[1], color="tab:red", ls="--", lw=2.0, zorder=2,
                label=f"KILL bar: random k=128 = {band_means[1]:+.3f}")
    # report-only greedy oracle at k=16
    ax1.scatter([math.log2(GREEDY_K)], [greedy["final_cost"]], marker="*",
                color="tab:red", s=420, zorder=6,
                label=f"GREEDY oracle k=16 "
                f"{greedy['final_cost']:+.3f} (report-only)")
    # e089 cross-references (same battery, seed 89)
    if "16" in e089:
        ax1.scatter([math.log2(16) + 0.12], [e089["16"]["mean"]], marker="x",
                    color="tab:green", s=90, lw=2.5, zorder=5,
                    label=f"e089 random k=16 {e089['16']['mean']:+.3f} "
                          f"(seed 89)")
    if "200" in e089:
        ax1.scatter([math.log2(200)], [e089["200"]["mean"]], marker="x",
                    color="tab:green", s=110, lw=2.5, zorder=5,
                    label=f"e089 random k=200 {e089['200']['mean']:+.3f} "
                          f"(the k+1 rung for k=128)")
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_xticks([math.log2(GREEDY_K)] + list(xk) + [math.log2(200)])
    ax1.set_xticklabels(["16*", "64", "128", "200 (e089)"])
    ax1.set_xlabel("k (anchor entries removed, log2 scale; 16* = greedy "
                   "oracle, report-only)")
    ax1.set_ylabel("clean-judge tail cost (nats)")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.set_title("E101-1 — THE adversarial kill-attempt: cost-vs-k per arm "
                  "vs the RANDOM band (dots = draws)", fontsize=10)

    # ---- panel 2: ratios vs random (same-k + kill) with CIs
    labs, vals, cites, cols, hats = [], [], [], [], []
    for key in ("k64_recency", "k64_reader", "k128_recency", "k128_reader"):
        labs.append(f"{key.split('_', 1)[1]}\nvs random\nsame k")
        vals.append(rsk[key]["val"])
        cites.append(rsk[key]["ci"])
        cols.append(arm_col[key])
        hats.append("")
    for key in ("k64_recency", "k64_reader"):
        labs.append(f"{key.split('_', 1)[1]}\nvs random\nk=128 (KILL)")
        vals.append(kr[key]["val"])
        cites.append(kr[key]["ci"])
        cols.append(arm_col[key])
        hats.append("//")
    xpos = np.arange(len(vals))
    for x, v, ci, c, h in zip(xpos, vals, cites, cols, hats):
        ax2.bar(x, v, width=0.6, color=c, alpha=0.75, hatch=h or None)
        if np.isfinite(v) and np.isfinite(ci[0]):
            ax2.errorbar(x, v, yerr=[[max(0.0, v - ci[0])],
                                     [max(0.0, ci[1] - v)]],
                         fmt="k_", capsize=6, lw=1.6, ms=12)
    ax2.axhline(1.0, color="k", lw=1.0, label="parity with random")
    ax2.axhline(RIDER_X, color="tab:green", ls=":", lw=1.8,
                label=f"recency rider {RIDER_X}x (law-stands clause)")
    ax2.text(len(vals) - 0.4, 1.0, " 1.0", fontsize=9, va="bottom")
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(labs, fontsize=8)
    ax2.set_ylabel("cost ratio (cluster CI)")
    ax2.set_title("E101-2 — adversarial/random ratios: hatched = the KILL "
                  "ratio (arm at k=64 vs random at k=128; >1 with CI "
                  "excluding 1 kills the law)", fontsize=9.5)
    ax2.legend(fontsize=9)

    # ---- panel 3: per-draw cost vs subset mean position
    for key in ARM_KEYS:
        dd = [dr for dr in M["draws"] if dr["arm"] == key]
        mk = "o" if key.startswith("k64") else "s"
        ax3.scatter([d["mean_p"] for d in dd], [d["cost"] for d in dd],
                    s=40, color=arm_col[key], marker=mk, alpha=0.8,
                    label=key)
    ax3.axhline(band_means[1], color="tab:red", ls="--", lw=1.6,
                label=f"random k=128 {band_means[1]:+.3f}")
    ax3.axvline(ANCHOR_POS[0] + 117 / 2, color="k", ls=":", lw=0.8)
    ax3.axvline(181 + 117 / 2, color="k", ls=":", lw=0.8)
    ax3.axvline(298 + 117 / 2, color="k", ls=":", lw=0.8)
    ax3.text(122, ax3.get_ylim()[1] * 0.93, "old", fontsize=8, ha="center")
    ax3.text(239, ax3.get_ylim()[1] * 0.93, "mid", fontsize=8, ha="center")
    ax3.text(356, ax3.get_ylim()[1] * 0.93, "new", fontsize=8, ha="center")
    ax3.axhline(0, color="k", lw=0.6)
    ax3.set_xlabel("subset mean position (old=sink side, new=head side)")
    ax3.set_ylabel("per-draw cost (nats)")
    ax3.set_title("E101-3 — selection-gradient texture: per-draw cost vs "
                  f"subset mean position | Spearman "
                  f"{S['texture']['spearman_cost_meanpos_all']:+.3f}",
                  fontsize=10)
    ax3.legend(fontsize=8, ncol=2)

    # ---- panel 4: the greedy oracle path
    steps = [s for s, _c in greedy["path"]]
    gcosts = [c for _s, c in greedy["path"]]
    ax4.plot(steps, gcosts, "o-", color="tab:red", lw=2.2, ms=8,
             label="greedy oracle cost(|S|) — report-only")
    margins = [p["margin"] for p in greedy["picks"]]
    ax4b = ax4.twinx()
    ax4b.bar([s for s, _c in greedy["path"]], margins, width=0.5,
             color="tab:purple", alpha=0.35,
             label="per-step margin over runner-up")
    ax4b.set_ylabel("margin (nats)", color="tab:purple")
    if "16" in e089:
        ax4.axhline(e089["16"]["mean"], color="tab:green", ls="-.", lw=1.6,
                    label=f"e089 random k=16 {e089['16']['mean']:+.3f}")
    ax4.axhspan(per_arm["k64_rand"]["ci"][0], per_arm["k64_rand"]["ci"][1],
                color="tab:gray", alpha=0.25,
                label=f"e101 random k=64 band {band_means[0]:+.3f}")
    ax4.axhline(band_means[1], color="tab:red", ls="--", lw=1.6,
                label=f"random k=128 {band_means[1]:+.3f}")
    ax4.axhline(0, color="k", lw=0.6)
    ax4.set_xlabel("greedy set size |S| (anchor + picks)")
    ax4.set_ylabel("clean-judge tail cost (nats)")
    ax4.set_title(f"E101-4 — adversarial-greedy oracle (k=16, run "
                  f"{greedy['run']}, anchored at {greedy['anchor']}): does "
                  f"16 adversarial entries reach mass-64 damage?",
                  fontsize=9.5)
    h1, l1 = ax4.get_legend_handles_labels()
    h2, l2 = ax4b.get_legend_handles_labels()
    ax4.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    # ---- panel 5: readership prior texture (what selection had to work with)
    rt = S["readership_texture"]
    for r in range(B):
        ax5.scatter([r], [rt["min_top64"][r]], color="tab:orange", s=40,
                    label="top-64 min(S)" if r == 0 else None)
        ax5.scatter([r], [rt["min_top128"][r]], color="tab:orange", s=80,
                    marker="s", alpha=0.6,
                    label="top-128 min(S)" if r == 0 else None)
    ax5.axhline(ANCHOR_POS[0], color="gray", lw=0.6,
                label=f"band floor {ANCHOR_POS[0]}")
    ax5.set_xlabel("run")
    ax5.set_ylabel("min(S) of the readership top-k")
    ax5.set_xticks(range(B))
    ov64 = np.mean(rt["overlap_with_recency_block"]["64"])
    ov128 = np.mean(rt["overlap_with_recency_block"]["128"])
    spr = np.nanmean(rt["spearman_mass_pos_per_run"]["64"])
    ax5.set_ylim(56, 100)                  # data lives at 64..76; the 350
    # exposure floor is >4x above every observed minimum — note, don't plot
    ax5.set_title("E101-5 — the readership prior: top-k minima per run "
                  "(ALL at the band floor 64..76, far below the 350 "
                  f"exposure floor); overlap with the recency block "
                  f"{ov64:.0f}/64, {ov128:.0f}/128; Spearman(mass,pos) "
                  f"{spr:+.2f} (prior agreement texture)", fontsize=9)
    ax5.legend(fontsize=9)

    # ---- panel 6: the registered decision
    ax6.axis("off")
    lines = [
        "REGISTERED (frozen in the tasking):",
        f"  KILL (BAR-A): any adversarial k=64 arm > random k=128 mean, "
        f"ratio CI excl. 1",
        f"  LAW STANDS (BAR-B): all arms < random k=128 AND recency <= "
        f"{RIDER_X}x random same-k (both k)",
        "  greedy oracle k=16: REPORT-ONLY; else MIXED/TEXTURE",
        "",
        "ARM TABLE (mean cost, cluster CI, ratio vs random same-k):",
    ]
    for k in ks:
        rk = per_arm[f"k{k}_rand"]
        lines.append(f"  k={k}: RANDOM {rk['mean']:+.3f} "
                     f"{_fmt_ci(rk['ci'])} (n={rk['n']})")
        for arm in ("recency", "reader"):
            key = f"k{k}_{arm}"
            st = per_arm[key]
            lines.append(f"    {arm:>7}: {st['mean']:+.3f} "
                         f"{_fmt_ci(st['ci'])} (n={st['n']}) | "
                         f"{rsk[key]['val']:.2f}x same-k, "
                         f"{kr[key]['val']:.2f}x k128"
                         if key in kr else
                         f"    {arm:>7}: {st['mean']:+.3f} "
                         f"{_fmt_ci(st['ci'])} (n={st['n']}) | "
                         f"{rsk[key]['val']:.2f}x same-k")
    lines += [
        "",
        f"GREEDY oracle (report-only): k=16 cost {greedy['final_cost']:+.3f} "
        f"(e089 random k=16 {e089.get('16', {}).get('mean', float('nan')):+.3f}; "
        f"e101 random k=64 {band_means[0]:+.3f})",
        f"  cross-ref: e089 random k=200 "
        f"{e089.get('200', {}).get('mean', float('nan')):+.3f} (the k+1 rung "
        f"for the k=128 arms)",
        "",
        f"DECISION [{dec['clause']}]:",
    ] + [f"  {wd}" for wd in _wrap(dec["verdict"], 96)]
    ax6.text(0.02, 0.97, "E101 — adversarial falsification of the "
                         "mass-action law (Rule-11)",
             fontsize=13, weight="bold", va="top")
    for i, tx in enumerate(lines):
        ax6.text(0.02, 0.93 - i * 0.0285, tx, fontsize=8.2, va="top",
                 family="monospace")

    fig.suptitle("E101 — adversarial subsets vs the mass ladder | clause: "
                 f"{dec['clause']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
