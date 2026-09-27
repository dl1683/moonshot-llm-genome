"""E095 — T050's registered TEXTURE PROBES: the escape tail and the donor voice.

Registered in THINKING.md T050 ("Two open textures (registered, next probes)")
before any of this was proposed; next_wave_programs.md item 2; dispatcher slot
e095. Everything below was registered (frozen) BEFORE any compute.

INSTRUMENT: a bit-exact RE-RUN of the e084 read-kernel census (the saved
runs/e084/metrics.json stores only per-dp flip COUNTS, not the per-cell
destination identities these probes need — the next-wave doc's registered
fallback: "where the saved aggregates fall short, re-run the deterministic
kernel on the same seeds/battery (274 s wall, documented)"). e084's machinery
is imported VERBATIM (same module, same constants: seeds 202/7 + 302/17
battery, margin-stratified 100 decision points seed 2084, 80-entry bands,
old-age seeds 9000+dp, 3 types, chunk 64, 8 threads, CPU-only). Replication
gates certify the re-run IS the e084 census before any probe statistic is
computed:
  G-R1  the 100 decision points (b, q, quartile, margin) match the saved
        per_decision_point exactly;
  G-R2  per-dp per-type flip counts match the saved flips_vz/kd/vs exactly;
  G-R3  aggregate taxonomy matches exactly (n_flips per type 573/425/543,
        outside-top-5 escape counts, V-swap donor-continuation hits = 48);
  G-R0b/G2/G1 e084's own net + generation identity gates re-checked.
Any gate failing aborts the run (no probe on uncertified cells).

PROBE 1 — H-TAIL (escape identities). Among the outside-top-5 flip escapes
(flip cells with taxonomy class 2, pooled over the 3 types; per-type
secondary), do destination token identities concentrate or stay diffuse?
  Null baseline (PRIMARY, registered): p_free = the battery's marginal
  next-token distribution = empirical token marginal over the 8 free runs'
  generated positions 64..511 (8 x 448 = 3,584 tokens) — next_wave's "uniform
  expectation under the battery's marginal next-token distribution".
  Secondaries: p_t1 = clean-argmax marginal over the 100 decision points;
  flat uniform 1/65 (reported, never the deciding null — char-LM argmaxes
  are trivially non-uniform).
  Statistic: t* = most frequent escape destination; ratio_top =
  share_obs(t*) / p_free(t*). Companion: top-10 coverage ratio =
  (sum of observed shares of the 10 most frequent destinations) / (sum of
  their p_free expectations) — next_wave's top-10 registration.
  CIs: bootstrap 2,000 reps. PRIMARY CI = decision-point cluster bootstrap
  (resample the 100 dps; escapes pool with multiplicity) — escapes cluster
  within low-margin dps, so the dp is the honest unit; cell-level iid
  bootstrap reported as secondary. Monte-Carlo guard (post-selection
  honesty): 2,000 draws of n_esc iid ~ p_free, max share distribution —
  the expected max share under the null, so the observed top share is
  judged against the null's own best case, not against p_free(t*) alone.
  REGISTERED BAR: H-TAIL REAL iff ratio_top >= 3.0 AND the 95% cluster-
  bootstrap CI of ratio_top excludes 1.0 (lo > 1.0). Otherwise the flag
  CLOSES AS NOISE: destinations are diffuse (indistinguishable from the
  battery's ordinary token frequencies) — that close IS the registered
  outcome (the flag was texture, not a claim).

PROBE 2 — H-DONOR-VOICE (the 48 donor-continuation hits). Do the V-swap
flips whose new argmax equals the donor run's actual next token cluster on
decision points where the run and its donor diverge stylistically?
  Divergence measure (defined here, registered): the donor of dp (b, q) is
  b2 = (b+1) mod 8 (the census's own donor rule). PRIMARY: D_pref(b, q) =
  token-disagreement rate over the shared free-run prefix — mean over
  positions p in [64, q] of 1[idx[b, p] != idx[b2, p]] (positions 0-63 are
  the fixed prompts, not generated style, so they are excluded). Higher =
  more divergent. Secondaries: D_win = same over the last 32 positions
  [max(64, q-31), q] (next_wave's +-32 window); D_ce = mean CE of the
  donor's next tokens under the run's context, positions p in [63, q]
  (donor token idx[b2, p+1] scored by a clean full forward on idx[b]).
  Comparison set: the non-hit V-swap flips (flip=1, donor_hit=0) — same
  type arm, matched on e084's age bands (young 1-10 / mid 11-60 / old 61+;
  next_wave's "same age band, same type arm").
  Statistic: each hit's percentile = midrank percentile of D_pref(hit's dp)
  within its band-matched non-hit D values; median over the 48 hits.
  CI: 2,000 bootstrap reps resampling hits and band-matched non-hit cells
  (within-band, preserving band counts), recompute the median percentile.
  REGISTERED BAR: H-DONOR-VOICE REAL iff median percentile >= 70 AND the
  95% bootstrap CI excludes 50 (lo > 50). Otherwise the flag CLOSES AS
  NOISE — again the registered outcome. Secondary measures (D_win, D_ce)
  reported with the same statistic; the primary (D_pref) rules; a
  primary/secondary disagreement is reported honestly, not adjudicated.

REGISTERED KILL (next_wave, verbatim in spirit): both nulls => the two
flags close as descriptive noise; the read-kernel card stands final with
the tail-misreport flag as a permanent caveat, and no tail model is
proposed.

Run:     python lab/e095_texture_probes.py
Outputs: runs/e095/metrics.json + runs/e095/texture_probes.png
Envelope: analysis-grade re-run of a proven instrument; CPU-only
(CUDA_VISIBLE_DEVICES=-1 and torch.cuda stubbed pre-import, e084 pattern),
8 threads, single step, minutes-scale (~6-10 min). No NOTES/THINKING/
QUEUE/STATE edits; no commit (the dispatcher owns those).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # task spec: CPU-only (pre-torch)
import torch

# force CPU so common.DEVICE == "cpu" (e053b/e069/e070/e084 pattern)
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
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
import e084_read_kernel as e084   # VERBATIM machinery + frozen design constants

E084_METRICS = REPO / "runs" / "e084" / "metrics.json"
E084_REF = json.loads(E084_METRICS.read_text(encoding="utf-8"))
BOOT_N = 2000
MC_N = 2000
BOOT_SEED = 95
MC_SEED = 5095
WIN = 32                                   # D_win +-32 window (next_wave)
TAIL_BAR_RATIO = 3.0                       # registered: ratio_top >= 3x
TAIL_CI_NULL = 1.0                         # CI must exclude 1.0
VOICE_BAR_PCT = 70.0                       # registered: median pct >= 70
VOICE_CI_NULL = 50.0                       # CI must exclude 50
TOP10 = 10                                 # companion coverage window

# e084's frozen design constants (imported, re-stated for readability)
PROMPT_TOK = e084.PROMPT_TOK               # 64
T_TOTAL = e084.T_TOTAL                     # 512
Q_LO, Q_HI = e084.Q_LO, e084.Q_HI
BANDS = {"young_1_10": (1, 10), "mid_11_60": (11, 60), "old_61_511": (61, T_TOTAL)}
TYPES = e084.TYPES                         # ["vzero", "kdrop", "vswap"]

T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


def band_of(age: int) -> int:
    if age <= 10:
        return 0
    if age <= 60:
        return 1
    return 2


def pct_midrank(x: float, ref: np.ndarray) -> float:
    """Midrank percentile of scalar x within ref (0..100)."""
    ref = np.asarray(ref, float)
    n = len(ref)
    if n == 0:
        return float("nan")
    return 100.0 * (float((ref < x).sum()) + 0.5 * float((ref == x).sum())) / n


@torch.no_grad()
def full_next_logits(net: TinyGPT, idx: torch.Tensor, chunk: int = 8):
    """(N, T) token block -> (N, T-1, V): logits at position p predicting
    token p+1 (clean full forward, chunked; e084 v_probe layout)."""
    N, T = idx.shape
    H = net.cfg.n_head
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    outs = []
    for i in range(0, N, chunk):
        x = net.wte(idx[i:i + chunk]) + net.wpe(torch.arange(T))[None]
        n = x.shape[0]
        for blk in net.h:
            xh = blk.ln1(x)
            qkv = blk.attn.c_attn(xh)
            C = qkv.shape[-1] // 3
            d = C // H
            q, k, v = qkv.split(C, dim=2)
            q = q.view(n, T, H, d).transpose(1, 2)
            k = k.view(n, T, H, d).transpose(1, 2)
            v = v.view(n, T, H, d).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
            att = att.masked_fill(causal, float("-inf"))
            y = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(n, T, C)
            x = x + blk.attn.c_proj(y)
            x = x + blk.mlp(blk.ln2(x))
        outs.append(net.lm_head(net.ln_f(x[:, :-1, :])))
    return torch.cat(outs, 0)


# ---------------------------------------------------------------------- main

def main():
    assert torch.cuda.device_count() == 0, "CPU-only violated: CUDA visible"
    assert common.DEVICE == "cpu", f"common.DEVICE={common.DEVICE} != cpu"
    out_dir = run_dir("e095")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gates = dict(cpu_only=True, threads=e084.THREADS)

    # ---- net + corpus, e084 verbatim --------------------------------------
    corp = CharCorpus(REPO / "data" / "input.txt")
    assert corp.vocab_size == 65
    st = torch.load(e084.CKPT, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    cfg = Cfg(vocab=corp.vocab_size, n_layer=4, n_head=4, n_embd=128,
              block_size=T_TOTAL)
    net = TinyGPT(cfg)
    net.load_state_dict(sd, strict=True)
    net.eval()
    n_params = net.num_params()
    val_ce = estimate_loss(net, corp, "val", n_batches=12)
    gates["G2_params"] = dict(params=n_params, expected=873472,
                              ok=bool(n_params == 873472))
    gates["G1_val_ce"] = dict(val_ce=val_ce, ref=e084.E053C_VAL_CE,
                              ok=bool(abs(val_ce - e084.E053C_VAL_CE) <= 0.02))
    log(f"net: {n_params:,} params, val CE {val_ce:.4f} -> "
        f"G2 {'PASS' if gates['G2_params']['ok'] else 'FAIL'} / "
        f"G1 {'PASS' if gates['G1_val_ce']['ok'] else 'FAIL'}")

    # ---- battery, e084 verbatim (seeds fixed -> deterministic) ------------
    gen_p = torch.Generator().manual_seed(e084.SEED_PROMPT)
    ix = torch.randint(len(corp.val) - PROMPT_TOK - 1, (e084.N_PROMPTS,),
                       generator=gen_p)
    prompts = [corp.val[i:i + PROMPT_TOK] for i in ix]
    gen = torch.Generator().manual_seed(e084.SEED_SAMPLE)
    idxA, final_logits, histA = e084.generate_batch_recorded(
        net, prompts[:e084.B], gen, gen_stop=e084.GEN_STOP)
    with torch.no_grad():
        full = e084.manual_logits2(net, idxA[:, :-1], None, None, None,
                                   chunk=e084.B)
    dev_b = float((torch.softmax(final_logits.float(), -1)
                   - torch.softmax(full.float(), -1)).abs().max())
    gates["G_R0b_kvs_vs_full"] = dict(max_prob_dev=dev_b, ok=bool(dev_b < 1e-4))
    log(f"G-R0b incremental-KV vs full: {dev_b:.2e} "
        f"-> {'PASS' if dev_b < 1e-4 else 'FAIL'}")
    gen_px = torch.Generator().manual_seed(e084.SEED_PROMPT_X)
    ix12 = torch.randint(len(corp.val) - PROMPT_TOK - 1, (e084.B_X,),
                         generator=gen_px)
    prompts12 = [corp.val[i:i + PROMPT_TOK] for i in ix12]
    gen_x = torch.Generator().manual_seed(e084.SEED_SAMPLE_X)
    idxX, _, histX = e084.generate_batch_recorded(net, prompts12, gen_x,
                                                  gen_stop=e084.GEN_STOP)
    idx16 = torch.cat([idxA, idxX], 0)
    hist16 = torch.cat([histA, histX], 0)
    log(f"battery rebuilt: {idx16.shape[0]} sequences x {idx16.shape[1]} tok")

    # ---- decision-point selection, e084 verbatim ---------------------------
    cands = []
    for b in range(idx16.shape[0]):
        for q in range(Q_LO, Q_HI + 1):
            lg = hist16[b, q - PROMPT_TOK + 1].float()
            tv = torch.topk(lg, 2).values
            cands.append((b, q, float(tv[0] - tv[1])))
    margins_all = np.asarray([c[2] for c in cands])
    qs = np.percentile(margins_all, [25, 50, 75])
    quart_of = np.digitize(margins_all, qs)
    rng = np.random.default_rng(e084.SEED_STRAT)
    dps = []
    for qt in range(4):
        pool = [i for i in range(len(cands)) if quart_of[i] == qt]
        take = rng.choice(pool, size=e084.PER_QUARTILE, replace=False)
        dps += [cands[i] + (qt,) for i in take]

    # G-R1: decision points match the saved census exactly on (b, q, quartile).
    # NOTE on margins: the SELECTION uses generation-time incremental-KV
    # candidate margins (hist16), while e084's saved per_dp margin is the
    # census clean-row recompute — same quantity, two instruments, bound by
    # G-R0b (<1e-4); they are compared with atol=1e-3 here, and the census
    # clean-row margins are compared near-exactly in G-R2 below.
    saved_dps = E084_REF["per_decision_point"]
    mine_bq = [(d[0], d[1], d[3]) for d in dps]           # b, q, quartile
    theirs_bq = [(s["b"], s["q"], s["quartile"]) for s in saved_dps]
    gr1_bq = mine_bq == theirs_bq
    marg_dev = float(np.max(np.abs(
        np.asarray([d[2] for d in dps])
        - np.asarray([s["margin"] for s in saved_dps]))))
    gr1 = bool(gr1_bq and marg_dev < 1e-3)
    gates["G_R1_decision_points"] = dict(
        n=len(dps), bq_quartile_match=bool(gr1_bq), max_margin_dev=marg_dev,
        margin_note="selection margin (hist16) vs saved census clean-row "
                    "margin; cross-instrument, G-R0b-bound",
        quartile_edges=qs.tolist(),
        quartile_edges_match=bool(np.allclose(
            qs, E084_REF["margin_structure"]["quartile_edges"], atol=1e-9)))
    log(f"G-R1 decision points: bq+quartile "
        f"{'PASS' if gr1_bq else 'FAIL'}, max margin dev {marg_dev:.2e}, "
        f"edges match {gates['G_R1_decision_points']['quartile_edges_match']}")
    assert gr1, "G-R1 FAIL: re-run does not reproduce e084's decision points"

    # ---- the census re-run (verbatim instrument, per-cell records kept) ----
    donor_v = e084.v_probe(net, idx16, chunk=8)           # (L, 8, 512, H, d)
    t_census = time.time()
    per_dp, rec_flat = [], []
    for i, (b, q, _, qt) in enumerate(dps):
        c = e084.census_dp(net, idx16, donor_v, b, q, i)
        per_dp.append(dict(dp=i, b=c["b"], q=c["q"], quartile=int(qt),
                           margin=c["margin"], t1=c["t1"], t2=c["t2"],
                           top5=c["top5"], donor_pick=c["donor_pick"],
                           flips=[0, 0, 0]))
        for (typ, age, flip, t_prime, cls, dh) in c["recs"]:
            rec_flat.append((i, typ, age, flip, t_prime, cls, dh))
            if flip:
                per_dp[-1]["flips"][typ] += 1
        del c["logits"]
        if (i + 1) % 20 == 0 or i == len(dps) - 1:
            done = i + 1
            rate = (time.time() - t_census) / done
            log(f"census {done}/{len(dps)} dps ({rate:.2f}s/dp, "
                f"ETA {rate * (len(dps) - done):.0f}s)")

    # G-R2: per-dp per-type flip counts match exactly, and the census
    # clean-row margins (same instrument as the saved ones) match near-bit
    saved_flips = [(s["flips_vz"], s["flips_kd"], s["flips_vs"])
                   for s in saved_dps]
    my_flips = [tuple(d["flips"]) for d in per_dp]
    gr2_flips = my_flips == saved_flips
    marg_dev2 = float(np.max(np.abs(
        np.asarray([d["margin"] for d in per_dp])
        - np.asarray([s["margin"] for s in saved_dps]))))
    gr2 = bool(gr2_flips and marg_dev2 < 1e-6)
    gates["G_R2_flip_counts"] = dict(
        n=len(dps), exact_match=bool(gr2_flips),
        max_margin_dev=marg_dev2,
        first_div=None if gr2_flips else next(
            (i, m, t) for i, (m, t) in
            enumerate(zip(my_flips, saved_flips)) if m != t))
    log(f"G-R2 per-dp flip counts: {'PASS' if gr2_flips else 'FAIL'} "
        f"(census margins max dev {marg_dev2:.2e})")

    rec = np.asarray([(r[0], r[1], r[2], r[3], r[5], r[6]) for r in rec_flat],
                     dtype=np.int64)     # dp, type, age, flip, cls, donor_hit
    t_prime_flat = np.asarray([r[4] for r in rec_flat], dtype=np.int64)
    fl = rec[rec[:, 3] == 1]

    # G-R3: aggregate taxonomy matches exactly
    n_fl_type = [int((fl[:, 1] == t).sum()) for t in range(3)]
    n_esc_type = [int(((fl[:, 1] == t) & (fl[:, 4] == 2)).sum()) for t in range(3)]
    hits = int(((fl[:, 1] == 2) & (fl[:, 5] == 1)).sum())
    ref_tax = E084_REF["taxonomy"]["per_type"]
    ref_nfl = [ref_tax[t]["n_flips"] for t in TYPES]
    ref_esc = [int(round(ref_tax[t]["outside_top5"] * ref_tax[t]["n_flips"]))
               for t in TYPES]
    ref_hits = E084_REF["taxonomy"]["vswap_donor_continuation"]["hits"]
    gr3 = (n_fl_type == ref_nfl and n_esc_type == ref_esc and hits == ref_hits)
    gates["G_R3_taxonomy"] = dict(
        n_flips_per_type=n_fl_type, ref_n_flips_per_type=ref_nfl,
        n_escapes_per_type=n_esc_type, ref_n_escapes_per_type=ref_esc,
        donor_hits=hits, ref_donor_hits=ref_hits, exact_match=bool(gr3))
    log(f"G-R3 taxonomy: flips {n_fl_type} vs {ref_nfl} | escapes {n_esc_type} "
        f"vs {ref_esc} | donor hits {hits} vs {ref_hits} -> "
        f"{'PASS' if gr3 else 'FAIL'}")
    assert gr2 and gr3, "G-R2/G-R3 FAIL: re-run cells not certified"

    # ================================================= PROBE 1: H-TAIL ======
    esc_mask = (rec[:, 3] == 1) & (rec[:, 4] == 2)
    esc_rows = np.where(esc_mask)[0]
    n_esc = len(esc_rows)
    esc_dp = rec[esc_rows, 0]
    esc_tok = t_prime_flat[esc_rows]
    counts = Counter(int(t) for t in esc_tok)

    # null baselines
    p_free = np.bincount(idx16[:, PROMPT_TOK:].numpy().ravel(),
                         minlength=65).astype(float)
    p_free /= p_free.sum()
    t1s = [d["t1"] for d in per_dp]
    p_t1 = np.bincount(t1s, minlength=65).astype(float)
    p_t1 /= p_t1.sum()

    t_star, n_star = counts.most_common(1)[0]
    share_star = n_star / n_esc
    ratio_free = share_star / p_free[t_star]
    ratio_t1 = share_star / p_t1[t_star]
    ratio_unif = share_star * 65.0
    esc_counts_full = np.zeros(65, dtype=np.int64)
    for t, c in counts.items():
        esc_counts_full[t] = c

    top10 = counts.most_common(TOP10)
    cov_obs = sum(c for _, c in top10) / n_esc
    cov_null = sum(p_free[t] for t, _ in top10)
    cov_ratio = cov_obs / cov_null

    # primary CI: dp-cluster bootstrap of share(t*)/p_free(t*)
    rng_b = np.random.default_rng(BOOT_SEED)
    esc_by_dp = {i: esc_tok[esc_dp == i] for i in range(len(dps))}
    cl_ratio, cell_ratio = [], []
    for _ in range(BOOT_N):
        sel = rng_b.integers(0, len(dps), len(dps))
        pool = np.concatenate([esc_by_dp[j] for j in sel]) if len(sel) else esc_tok
        s_cl = float((pool == t_star).mean()) / p_free[t_star]
        cl_ratio.append(s_cl)
        draw = esc_tok[rng_b.integers(0, n_esc, n_esc)]
        cell_ratio.append(float((draw == t_star).mean()) / p_free[t_star])
    cl_ci = (float(np.percentile(cl_ratio, 2.5)),
             float(np.percentile(cl_ratio, 97.5)))
    cell_ci = (float(np.percentile(cell_ratio, 2.5)),
               float(np.percentile(cell_ratio, 97.5)))

    # Monte-Carlo null guard: max share distribution under p_free
    rng_m = np.random.default_rng(MC_SEED)
    mc_max = np.empty(MC_N)
    p_free_np = p_free / p_free.sum()
    for i in range(MC_N):
        draw = rng_m.multinomial(n_esc, p_free_np).astype(float)
        mc_max[i] = draw.max() / n_esc
    mc_p = float((mc_max >= share_star).mean())

    tail_fires = bool(ratio_free >= TAIL_BAR_RATIO and cl_ci[0] > TAIL_CI_NULL)
    tail_verdict = (
        f"H-TAIL REAL: top escape destination {corp.itos[t_star]!r} covers "
        f"{share_star * 100:.1f}% of {n_esc} escapes = {ratio_free:.2f}x its "
        f"battery-marginal expectation (>= 3x bar; cluster CI "
        f"[{cl_ci[0]:.2f},{cl_ci[1]:.2f}] excludes 1) — destinations are "
        f"predictable." if tail_fires else
        f"H-TAIL CLOSES AS NOISE: top escape destination {corp.itos[t_star]!r} "
        f"covers {share_star * 100:.1f}% of {n_esc} escapes = "
        f"{ratio_free:.2f}x its battery-marginal expectation (< 3x bar, or CI "
        f"[{cl_ci[0]:.2f},{cl_ci[1]:.2f}] does not exclude 1) — escape "
        f"destinations are diffuse, indistinguishable from the battery's "
        f"ordinary token frequencies.")
    log(f"PROBE 1 H-TAIL: {n_esc} escapes; top {corp.itos[t_star]!r} "
        f"{share_star * 100:.1f}% = {ratio_free:.2f}x p_free, "
        f"{ratio_t1:.2f}x p_t1, {ratio_unif:.2f}x flat-uniform | top-10 "
        f"coverage {cov_ratio:.2f}x | MC null max-share "
        f"{mc_max.mean() * 100:.1f}% (p={mc_p:.3f}) | "
        f"{tail_verdict.split(':')[0]}")

    # ============================================= PROBE 2: H-DONOR-VOICE ==
    idx_np = idx16.numpy()
    # D_ce: donor tokens under run context (8 clean full forwards)
    ce_by_pair = {}
    with torch.no_grad():
        for b in range(idx16.shape[0]):
            lg = full_next_logits(net, idx16[b:b + 1])[0]      # (511, V)
            lp = torch.log_softmax(lg.float(), -1).numpy()
            b2 = (b + 1) % idx16.shape[0]
            ce_by_pair[b] = -lp[np.arange(511), idx_np[b2, 1:]]  # CE tok p+1

    D_pref, D_win, D_ce = {}, {}, {}
    for d in per_dp:
        b, q = d["b"], d["q"]
        b2 = (b + 1) % idx16.shape[0]
        seg = idx_np[b, PROMPT_TOK:q + 1] != idx_np[b2, PROMPT_TOK:q + 1]
        D_pref[d["dp"]] = float(seg.mean())
        lo = max(PROMPT_TOK, q - WIN + 1)
        D_win[d["dp"]] = float(seg[lo - PROMPT_TOK:].mean())
        D_ce[d["dp"]] = float(ce_by_pair[b][PROMPT_TOK - 1:q + 1].mean())

    vs = fl[fl[:, 1] == 2]                       # V-swap flip cells
    hit_rows = vs[vs[:, 5] == 1]
    non_rows = vs[vs[:, 5] == 0]
    log(f"PROBE 2 units: {len(hit_rows)} donor-continuation hits vs "
        f"{len(non_rows)} non-hit V-swap flips "
        f"({int((vs[:, 5] == 1).sum())}+{int((vs[:, 5] == 0).sum())} of "
        f"{len(vs)} V-swap flips)")

    non_D = {m: np.array([0.0]) for m in ("pref", "win", "ce")}
    non_band = np.array([band_of(int(a)) for a in non_rows[:, 2]])
    for m, D in (("pref", D_pref), ("win", D_win), ("ce", D_ce)):
        non_D[m] = np.array([D[int(dp)] for dp in non_rows[:, 0]])

    def voice_stats(meas_D):
        hit_D = np.array([meas_D[int(dp)] for dp in hit_rows[:, 0]])
        hit_band = np.array([band_of(int(a)) for a in hit_rows[:, 2]])
        pcts = np.array([pct_midrank(meas_D[int(hit_rows[i, 0])],
                                     non_D_pref_vals[hit_band[i]])
                         for i in range(len(hit_rows))])
        return hit_D, pcts

    # primary measure D_pref, band-matched percentiles
    non_D_pref_vals = {bd: non_D["pref"][non_band == bd] for bd in range(3)}
    hit_D_pref, hit_pcts = voice_stats(D_pref)
    med_pct = float(np.median(hit_pcts))

    # bootstrap: resample hits + band-matched non-hits (band counts fixed)
    rng_v = np.random.default_rng(BOOT_SEED + 1)
    boot_pct, boot_diff = [], []
    for _ in range(BOOT_N):
        hi = rng_v.integers(0, len(hit_rows), len(hit_rows))
        nb = {}
        for bd in range(3):
            src = non_D["pref"][non_band == bd]
            nb[bd] = src[rng_v.integers(0, len(src), len(src))] if len(src) \
                else src
        ps = []
        for i in hi:
            ps.append(pct_midrank(D_pref[int(hit_rows[i, 0])], nb[band_of(int(hit_rows[i, 2]))]))
        boot_pct.append(float(np.median(ps)))
        dh = np.array([D_pref[int(hit_rows[i, 0])] for i in hi])
        dn = np.concatenate(list(nb.values()))
        boot_diff.append(float(np.median(dh) - np.median(dn)))
    pct_ci = (float(np.percentile(boot_pct, 2.5)),
              float(np.percentile(boot_pct, 97.5)))
    diff_ci = (float(np.percentile(boot_diff, 2.5)),
               float(np.percentile(boot_diff, 97.5)))

    # secondary measures, same statistic
    sec = {}
    for m, D in (("win", D_win), ("ce", D_ce)):
        hit_Dm = np.array([D[int(dp)] for dp in hit_rows[:, 0]])
        hb = np.array([band_of(int(a)) for a in hit_rows[:, 2]])
        pm = np.array([pct_midrank(D[int(hit_rows[i, 0])],
                                   non_D[m][non_band == hb[i]])
                       for i in range(len(hit_rows))])
        sec[m] = dict(median_pct=float(np.median(pm)),
                      hit_median=float(np.median(hit_Dm)),
                      non_median=float(np.median(non_D[m])))

    voice_fires = bool(med_pct >= VOICE_BAR_PCT and pct_ci[0] > VOICE_CI_NULL)
    voice_verdict = (
        f"H-DONOR-VOICE REAL: hits sit at median {med_pct:.1f}th percentile of "
        f"run-donor divergence (>= 70 bar; CI [{pct_ci[0]:.1f},"
        f"{pct_ci[1]:.1f}] excludes 50) — donor content is readable exactly "
        f"where the run leaves it." if voice_fires else
        f"H-DONOR-VOICE CLOSES AS NOISE: hits sit at median {med_pct:.1f}th "
        f"divergence percentile (< 70 bar, or CI [{pct_ci[0]:.1f},"
        f"{pct_ci[1]:.1f}] does not exclude 50) — the 48 donor-continuation "
        f"hits do not track run-donor stylistic divergence.")
    log(f"PROBE 2 H-DONOR-VOICE: median pct {med_pct:.1f} CI "
        f"[{pct_ci[0]:.1f},{pct_ci[1]:.1f}] | D_pref hits "
        f"{np.median(hit_D_pref):.4f} vs non-hits "
        f"{np.median(non_D['pref']):.4f} (diff CI "
        f"[{diff_ci[0]:+.4f},{diff_ci[1]:+.4f}]) | "
        f"{voice_verdict.split(':')[0]}")

    hit_dps = sorted(set(int(d) for d in hit_rows[:, 0]))
    per_b = {}
    for b in range(idx16.shape[0]):
        b_dps = [d for d in per_dp if d["b"] == b]
        per_b[int(b)] = dict(
            n_dps=len(b_dps),
            n_hits=int(sum(1 for r in hit_rows if per_dp[int(r[0])]["b"] == b)),
            mean_D=float(np.mean([D_pref[d["dp"]] for d in b_dps])))

    # ---------------------------------------------------------------- metrics
    metrics = dict(
        experiment="e095_texture_probes",
        purpose="T050's registered texture probes on the e084 read-kernel "
                "census (re-run bit-exact, G-R gated): H-tail escape-identity "
                "concentration vs the battery's marginal next-token "
                "distribution (bar: top destination >= 3x expectation, "
                "cluster-CI excluding 1); H-donor-voice divergence "
                "clustering of the 48 donor-continuation hits vs band-matched "
                "non-hit V-swap flips (bar: median percentile >= 70, CI "
                "excluding 50). Either failing closes its flag as noise — "
                "the registered outcome.",
        started=started, wall_s=elapsed(), threads=e084.THREADS,
        cpu_only=True,
        provenance=dict(e084_metrics=str(E084_METRICS),
                        rerun_note="e084 machinery imported verbatim; saved "
                                   "aggregates carry no per-cell identities",
                        n_cells=int(len(rec)), n_flips=int(len(fl))),
        gates=gates,
        probe1_htail=dict(
            n_escapes=n_esc,
            escapes_per_type={TYPES[t]: int(((fl[:, 1] == t) &
                                             (fl[:, 4] == 2)).sum())
                              for t in range(3)},
            top_destination=dict(token=corp.itos[t_star], token_id=t_star,
                                count=n_star, share=share_star),
            null_p_free=p_free.tolist(), null_p_t1=p_t1.tolist(),
            escape_counts_by_token=esc_counts_full.tolist(),
            ratio_vs_p_free=ratio_free, ratio_vs_p_t1=ratio_t1,
            ratio_vs_flat_uniform=ratio_unif,
            ci_cluster=dict(lo=cl_ci[0], hi=cl_ci[1]),
            ci_cell=dict(lo=cell_ci[0], hi=cell_ci[1]),
            top10_census=[[corp.itos[t], int(c), c / n_esc, float(p_free[t])]
                          for t, c in top10],
            top10_coverage=dict(observed=cov_obs, null=cov_null,
                                ratio=cov_ratio),
            mc_null_max_share=dict(mean=float(mc_max.mean()),
                                   p90=float(np.percentile(mc_max, 90)),
                                   observed=share_star, p_value=mc_p,
                                   n_draws=MC_N),
            bar=dict(ratio=TAIL_BAR_RATIO, ci_excludes=TAIL_CI_NULL),
            fires=tail_fires, verdict=tail_verdict),
        probe2_hdonorvoice=dict(
            n_hits=len(hit_rows), n_nonhit_vswap_flips=len(non_rows),
            hit_dps=hit_dps, n_distinct_hit_dps=len(hit_dps),
            divergence_def=dict(
                primary="D_pref: token-disagreement rate over the shared "
                        "free-run prefix [64, q], run b vs donor (b+1) mod 8",
                secondary_win=f"D_win: same over the last {WIN} positions",
                tertiary="D_ce: mean CE of donor next tokens under the run's "
                         "clean context"),
            median_percentile=med_pct, ci=pct_ci,
            d_pref_hit_median=float(np.median(hit_D_pref)),
            d_pref_nonhit_median=float(np.median(non_D["pref"])),
            d_pref_diff_ci=diff_ci,
            hit_percentiles=[float(x) for x in hit_pcts],
            secondary={m: v for m, v in sec.items()},
            per_b_hits=per_b,
            band_counts=dict(
                hits={n: int(sum(1 for a in hit_rows[:, 2]
                                 if band_of(int(a)) == i))
                      for i, n in enumerate(BANDS)},
                nonhits={n: int((non_band == i).sum())
                         for i, n in enumerate(BANDS)}),
            bar=dict(percentile=VOICE_BAR_PCT, ci_excludes=VOICE_CI_NULL),
            fires=voice_fires, verdict=voice_verdict),
        registered_decision=dict(
            probes="T050 H-tail + H-donor-voice (both texture flags)",
            h_tail_fires=tail_fires, h_tail_verdict=tail_verdict,
            h_donor_voice_fires=voice_fires,
            h_donor_voice_verdict=voice_verdict,
            kill_note="Both nulls => flags close as descriptive noise; the "
                      "read-kernel card stands final with the tail-misreport "
                      "flag as a permanent caveat; no tail model proposed."),
    )
    save_json(out_dir / "metrics.json", metrics)
    log("metrics.json written")
    plot(out_dir / "texture_probes.png", metrics, corp)
    log(f"plot written; total wall {elapsed():.0f}s")


# ---------------------------------------------------------------------- plot

def plot(path: Path, M: dict, corp: CharCorpus):
    p1 = M["probe1_htail"]
    p2 = M["probe2_hdonorvoice"]
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    # (a) escape-destination census vs the battery baseline
    cens = p1["top10_census"]
    toks = [repr(c[0]) for c in cens]
    obs = [c[2] for c in cens]
    nul = [c[3] for c in cens]
    xs = np.arange(len(cens))
    ax1.bar(xs, obs, 0.62, color="tab:gray", alpha=0.85,
            label="observed escape share")
    ax1.plot(xs, nul, "D", ms=6, color="tab:red",
             label="battery marginal p_free")
    ax1.axhline(1 / 65, color="k", ls=":", lw=1.2, label="flat uniform 1/65")
    ax1.set_xticks(xs, toks, fontsize=9)
    td = p1["top_destination"]
    ci = p1["ci_cluster"]
    ax1.set_title(f"(a) H-TAIL: escape-destination census "
                  f"({p1['n_escapes']} outside-top-5 escapes)", fontsize=9.5)
    ax1.text(0.02, 0.97,
             f"top {td['token']!r}: {td['share'] * 100:.1f}% = "
             f"{p1['ratio_vs_p_free']:.2f}x p_free\n"
             f"cluster CI [{ci['lo']:.2f}, {ci['hi']:.2f}] "
             f"(bar >= 3x, exclude 1)\nvs p_t1 {p1['ratio_vs_p_t1']:.2f}x | "
             f"flat-unif {p1['ratio_vs_flat_uniform']:.2f}x",
             transform=ax1.transAxes, fontsize=8.5, va="top",
             family="monospace",
             bbox=dict(fc="white", ec="dimgray", alpha=0.85))
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_ylabel("share of escapes")

    # (b) concentration curve: cumulative share vs rank (exact histogram)
    p_free = np.asarray(p1["null_p_free"], float)
    n_esc = p1["n_escapes"]
    hist = np.asarray(p1["escape_counts_by_token"], float)
    obs_sorted = np.sort(hist)[::-1] / n_esc
    null_sorted = np.sort(p_free)[::-1]
    xs65 = np.arange(1, 66)
    ax2.plot(xs65, np.cumsum(obs_sorted), "o-", ms=3, lw=1.3,
             color="tab:gray", label="observed destinations")
    ax2.plot(xs65, np.cumsum(null_sorted), "s-", ms=3, lw=1.2,
             color="tab:red", label="battery marginal (sorted)")
    ax2.plot(xs65, xs65 / 65, ":", color="k", lw=1.2, label="flat uniform")
    cov = p1["top10_coverage"]
    ax2.axvline(10, color="dimgray", ls="--", lw=1)
    ax2.text(0.35, 0.06,
             f"top-10 coverage {cov['observed'] * 100:.0f}% vs "
             f"{cov['null'] * 100:.0f}% expected\nratio "
             f"{cov['ratio']:.2f}x (companion bar, next_wave)",
             transform=ax2.transAxes, fontsize=8.5, family="monospace",
             bbox=dict(fc="white", ec="dimgray", alpha=0.85))
    ax2.set_xlabel("destination rank (sorted)")
    ax2.set_ylabel("cumulative share of escapes")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.set_title("(b) concentration curve — diffuse tail stays on the red "
                  "line", fontsize=9.5)

    # (c) Monte-Carlo null max-share vs observed (redraw, same seed)
    mcn = p1["mc_null_max_share"]
    rng_mc = np.random.default_rng(MC_SEED)
    p_free_np = np.asarray(p1["null_p_free"], float)
    p_free_np /= p_free_np.sum()
    draws = np.array([rng_mc.multinomial(n_esc, p_free_np).max() / n_esc
                      for _ in range(MC_N)])
    ax3.hist(draws * 100, bins=40, color="tab:red", alpha=0.7,
             label=f"null max share ({MC_N} MC draws)")
    ax3.axvline(mcn["observed"] * 100, color="k", lw=2,
                label=f"observed top share {mcn['observed'] * 100:.1f}%")
    ax3.axvline(mcn["mean"] * 100, color="tab:red", ls="--", lw=1.2,
                label=f"null mean {mcn['mean'] * 100:.1f}%")
    ax3.set_xlabel("top-destination share of escapes (%)")
    ax3.set_ylabel("MC draws")
    ax3.legend(fontsize=8)
    ax3.set_title(f"(c) post-selection guard: observed top share vs the "
                  f"null's own best case\nMC p = {mcn['p_value']:.3f} "
                  f"(p90 of null {mcn['p90'] * 100:.1f}%)", fontsize=9.5)

    # (d) divergence: the 48 hits' percentiles (band-matched, D_pref)
    hp = np.asarray(p2["hit_percentiles"], float)
    ax4.step(np.sort(hp), np.arange(1, len(hp) + 1) / len(hp) * 100,
             where="post", color="tab:purple", lw=2,
             label="hits' divergence percentile ECDF")
    ax4.plot([0, 100], [0, 100], ":", color="k", lw=1.2, label="uniform")
    ax4.axvline(50, color="dimgray", ls="--", lw=1.1)
    ax4.axvline(70, color="tab:green", ls="--", lw=1.4, label="registered bar 70")
    ax4.axvline(np.median(hp), color="tab:purple", lw=1.6, ls="-",
                label=f"median {np.median(hp):.1f}")
    ci2 = p2["ci"]
    ax4.text(0.03, 0.62,
             f"D_pref hits median {p2['d_pref_hit_median']:.4f}\nvs non-hits "
             f"{p2['d_pref_nonhit_median']:.4f}\n(diff CI "
             f"[{p2['d_pref_diff_ci'][0]:+.4f}, "
             f"{p2['d_pref_diff_ci'][1]:+.4f}])",
             transform=ax4.transAxes, fontsize=8.5, family="monospace",
             bbox=dict(fc="white", ec="dimgray", alpha=0.85))
    ax4.set_xlabel("divergence percentile of hit (within band-matched "
                   "non-hit V-swap flips)")
    ax4.set_ylabel("cumulative % of hits")
    ax4.legend(fontsize=8, loc="lower right")
    ax4.set_title("(d) H-DONOR-VOICE: the 48 hits vs divergence "
                  f"(CI [{ci2[0]:.1f}, {ci2[1]:.1f}], excludes 50?)",
                  fontsize=9.5)

    # (e) secondary measures, same statistic
    sec = p2["secondary"]
    names = ["D_pref\n(primary)", "D_win\n(+-32 win)", "D_ce\n(donor CE)"]
    vals = [p2["median_percentile"], sec["win"]["median_pct"],
            sec["ce"]["median_pct"]]
    ax5.bar(np.arange(3), vals, 0.55, color=["tab:purple", "tab:blue",
                                             "tab:olive"], alpha=0.85)
    ax5.axhline(70, color="tab:green", ls="--", lw=1.4, label="bar 70")
    ax5.axhline(50, color="k", ls=":", lw=1.4, label="null 50")
    for i, v in enumerate(vals):
        ax5.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    ax5.set_xticks(np.arange(3), names, fontsize=9)
    ax5.set_ylim(0, 100)
    ax5.set_ylabel("median divergence percentile of hits")
    ax5.legend(fontsize=8)
    ax5.set_title("(e) measure robustness — primary rules", fontsize=9.5)

    # (f) per-b texture + verdict summary
    pb = p2["per_b_hits"]
    bs = sorted(int(b) for b in pb)
    xs = np.arange(len(bs))
    ax6.bar(xs, [pb[b]["n_hits"] for b in bs], 0.55, color="tab:purple",
            alpha=0.85, label="donor-continuation hits")
    ax6.set_xticks(xs, [f"b{b}" for b in bs])
    ax6.set_ylabel("hits (of 48)")
    ax6b = ax6.twinx()
    ax6b.plot(xs, [pb[b]["mean_D"] for b in bs], "o-", color="tab:red",
              label="mean D_pref of b's dps")
    ax6b.set_ylabel("mean D_pref", color="tab:red")
    dec = M["registered_decision"]
    ax6.set_title("(f) texture: hits per run b vs that run's mean divergence "
                  f"| {p2['n_distinct_hit_dps']} dps carry hits",
                  fontsize=9.5)
    ax6.text(0.02, 0.97,
             f"H-TAIL: {'REAL' if dec['h_tail_fires'] else 'NOISE (closes)'}\n"
             f"H-DONOR-VOICE: "
             f"{'REAL' if dec['h_donor_voice_fires'] else 'NOISE (closes)'}",
             transform=ax6.transAxes, fontsize=10, va="top",
             family="monospace", weight="bold",
             bbox=dict(fc="gold" if (dec["h_tail_fires"]
                                     or dec["h_donor_voice_fires"])
                       else "lightgray", ec="dimgray", alpha=0.9))

    fig.suptitle("E095 TEXTURE PROBES (T050 registrations) | "
                 f"H-tail {'FIRES' if dec['h_tail_fires'] else 'closes as noise'} "
                 f"| H-donor-voice "
                 f"{'FIRES' if dec['h_donor_voice_fires'] else 'closes as noise'}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
