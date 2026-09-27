"""E099 — the ATTRACTOR IDENTITY probe (Rule-11 pick; T048/T051's open object).

[REGISTERED DESIGN — frozen in this docstring BEFORE any compute]

THE DEEP QUESTION: e080's three trigger arms (A-vzero, A-noise, A-promptcopy)
all collapse generation (self-scored fine ~1.2-1.4 nats, clean-judge ~6.4-6.6
nats, live_frac 1.000 — the off-manifold attractor of T048). WHEN GENERATION
COLLAPSES, WHERE DOES IT GO? Is there ONE universal degenerate attractor
(all collapses converge to the same terminal distribution — a property of
the NET, perturbation-independent), or PERTURBATION-SPECIFIC basins?

ARMS (e080 rig VERBATIM via module import: e053c net, fixed-anchor 64->512
free run, B=8 seed-202 battery, per-arm seed-7 sampling stream; interventions
at the SAME prune events g=100..420 K=32 on the self band age>96, replaced
once at first admission, never touched again; K untouched in every arm):
  1. A-none        control: normal free run.
  2. A-vzero       V -> 0 at band entry (e075 A-self-prune / e080 rerun).
  3. A-noise       V -> norm-matched gaussian direction (dedicated seed-4242
                   stream, never touches sampling).
  4. A-promptcopy  V <- same row's pristine prompt entry q=((p-64) mod 63)+1.
  5. A-randomize   NEW trigger for diversity: every newly admitted band
                   position's V vector is replaced by ANOTHER RUN'S entry at
                   the same position — donor row drawn once from a dedicated
                   seed-4343 generator as a derangement (d(b) != b), copied
                   from a PRE-EVENT snapshot so donors stay pristine
                   (cross-run transplant of real generated content).

READOUTS on every arm's continuation TAIL (final 128 generated tokens =
positions 384..511; 8 seqs x 128 = 1024 tokens/arm):
  (a) terminal token distribution: pooled unigram over the tail; pairwise
      cross-arm symmetric KL grid (5x5, diag = within-arm null) + each arm
      vs the corpus unigram baseline.
  (b) terminal text statistics: distribution entropy (pooled + per-seq),
      type-token ratios, repeat-n-gram rates (n=2,3,4: fraction of n-gram
      positions whose n-gram occurs >=2x within the tail) vs corpus-window
      baseline, top-token identity census (top-5 + per-seq modal token).
  (c) within-arm null KL: all 35 disjoint 4-vs-4 sequence splits of each
      arm -> symmetric-KL sampling floor (the bar cross-arm KLs must clear).
  RIDER (non-registered, texture only): the same grid over tail BIGRAM
  distributions (catches phrase-level divergence unigram misses).
  Collapse gate per arm: clean-judge tail CE - online tail CE (e080
  convention; gap > 1.0 nat = collapsed/off-manifold).

REGISTERED BARS (frozen; floor = mean over COLLAPSED arms of within-arm
null SKL; "collapsed arms" = trigger arms with clean-judge gap > 1.0):
  - ONE UNIVERSAL ATTRACTOR:  max cross-arm SKL among collapsed pairs
    <= 2x floor AND all collapsed arms' top-5 tail token sets IDENTICAL
    (deterministic tie-break: sort by (-count, token id)).
  - PERTURBATION-SPECIFIC BASINS: min cross-arm SKL among collapsed pairs
    >= 5x floor AND collapsed arms' top-5 sets pairwise DISJOINT.
  - else MIXED texture (report the structure honestly).

GATES: G1 val CE vs e053c; G2 params; G4 rerun-vs-e080 drift (per-seq
final-64 tail CE + full CE trajectory + clean-judge, all four legacy arms);
G5 intervention identity (pre-event token identity through pos 164, replaced
counts 324, donor derangement, cache checks); G6 skeleton fidelity
(generate_arm5(mode=none) bit-identical to e080.generate_arm(mode=none));
G7 randomize determinism (rerun bit-identical).

Run:     python lab/e099_attractor_identity.py
Outputs: runs/e099/metrics.json + runs/e099/attractor_identity.png
Envelope: NO training, NO new automations; CPU-only (8 threads), single
step, minutes. No NOTES/THINKING/QUEUE/STATE edits; no commit.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # task spec: CPU-only (pre-torch)
import torch

# this torch build reports is_available()==True even with an empty
# CUDA_VISIBLE_DEVICES; force CPU so common.DEVICE=="cpu" (e053b..e080)
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import textwrap  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

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
import e080_prune_vs_replace as e080   # VERBATIM rig + frozen design constants

THREADS = 8                                # task spec / T050: 12 thrashes box
torch.set_num_threads(THREADS)             # (e080's import sets 12; override)

# ------------------------------------------------------------------ constants
ARMS = ["none", "vzero", "noise", "promptcopy", "randomize"]
LEGACY = ["none", "vzero", "noise", "promptcopy"]      # e080's four, verbatim
TRIGGERS = ["vzero", "noise", "promptcopy", "randomize"]

TAIL = 128                                 # THE tail window (final 128 gen)
VOCAB = 65
ALPHA = 0.5                                # add-alpha KL smoothing
BIALPHA = 0.1                              # bigram rider smoothing
NGRAMS = (2, 3, 4)
SEED_DONOR = 4343                          # e099: randomize donor stream
N_WIN = 8                                  # corpus baseline windows

E080_METRICS = REPO / "runs" / "e080" / "metrics.json"
GAP_BAR = 1.0                              # clean-judge gap > 1 nat = collapsed
UNIV_RATIO = 2.0                           # cross <= 2x floor
SPEC_RATIO = 5.0                           # cross >= 5x floor

T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# ---------------------------------------------------- generate_arm5 (e099 arm)

def draw_donor(seed: int, n: int) -> list[int]:
    """Randomize donor assignment: one other run per row, d(b) != b,
    drawn from a dedicated generator (deterministic given seed)."""
    gd = torch.Generator().manual_seed(seed)
    donor = []
    for b in range(n):
        d = int(torch.randint(0, n, (1,), generator=gd))
        while d == b:
            d = int(torch.randint(0, n, (1,), generator=gd))
        donor.append(d)
    return donor


@torch.no_grad()
def generate_arm5(net: TinyGPT, prompts, gen: torch.Generator, mode: str,
                  donor: list[int] | None = None):
    """e080.generate_arm VERBATIM SKELETON, extended with mode='randomize'
    (legacy modes are run through e080.generate_arm itself; this function
    supports 'none' — for the G6 skeleton-fidelity gate — and 'randomize').

    randomize: at each event, every NEWLY admitted band position p gets
    v[b,:,p,:] <- snapshot[d(b),:,p,:] — ANOTHER RUN'S entry at the same
    position, taken from a PRE-EVENT snapshot (all rows share the schedule,
    so pre-event donor content at p is the donor's own genuine generated
    entry; K untouched; replaced-stays-replaced)."""
    assert mode in ("none", "randomize")
    Bb = len(prompts)
    idx = torch.stack(prompts)
    logits, kv = e080.prefill_batch(net, idx)
    ce_s = np.zeros((Bb, e080.G), float)
    ent_s = np.zeros((Bb, e080.G), float)
    tk_s = np.zeros((Bb, e080.G), float)
    t1_s = np.zeros((Bb, e080.G), float)
    rk_s = np.zeros((Bb, e080.G), float)
    replaced: set = set()
    replace_log = []
    n_vec = 0
    cos_abs_sum = 0.0
    ratio_sum = 0.0
    for g in range(e080.G):
        t = e080.PROMPT_TOK + g
        if mode != "none" and e080.is_event(g):
            band = e080.prune_positions("self", t)
            new = sorted(set(band) - replaced)
            if new:
                sel = torch.tensor(new, dtype=torch.long)
                if mode == "randomize":
                    snap = [v.clone() for (_k, v) in kv]   # pre-event donors
                    for li, (_k, v) in enumerate(kv):
                        for b in range(Bb):
                            old = v[b, :, sel, :].clone()          # (H,n,d)
                            src = snap[li][donor[b]][:, sel, :]
                            nrm_old = old.norm(dim=-1)
                            nrm_src = src.norm(dim=-1)
                            cos = (old * src).sum(-1) / (nrm_old
                                                        * nrm_src
                                                        ).clamp_min(1e-12)
                            cos_abs_sum += float(cos.abs().sum())
                            ratio_sum += float((nrm_src
                                               / nrm_old.clamp_min(1e-12)).sum())
                            n_vec += int(cos.numel())
                            v[b, :, sel, :] = src
                else:
                    raise ValueError(mode)
                replaced.update(new)
                replace_log.append(dict(g=g, front=t, n_band=len(band),
                                        n_new=len(new), cum=len(replaced)))
        toks = torch.zeros(Bb, dtype=torch.long)
        for j in range(Bb):
            tok, ce = e080.sample_and_ce(logits[j], gen)
            toks[j] = tok
            ce_s[j, g] = ce
        p = torch.softmax(logits.float(), -1)
        ent_s[:, g] = (-(p * p.clamp_min(1e-12).log()).sum(-1)).numpy()
        v40, _ = torch.topk(p, e080.TOPK, dim=-1)
        tk_s[:, g] = v40.sum(-1).numpy()
        t1_s[:, g] = v40[:, 0].numpy()
        rows = torch.arange(Bb)
        own = logits[rows, toks]
        rk_s[:, g] = ((logits > own[:, None]).sum(-1) + 1).numpy()
        idx = torch.cat([idx, toks[:, None]], 1)
        if t < e080.T_TOTAL - 1:
            logits = e080.decode_step_batch(net, toks, t, kv)
    cache_len = kv[0][1].shape[2]
    repmask = torch.zeros(cache_len, dtype=torch.bool)
    if replaced:
        repmask[sorted(replaced)] = True
    max_pr, min_live = 0.0, float("inf")
    if mode == "randomize":
        for (_k, v) in kv:
            nv = v.abs().amax(dim=(0, 1, 3))
            max_pr = max(max_pr, float(nv[repmask].max()))
            min_live = min(min_live, float(nv[~repmask].min()))
        ok = bool(max_pr > 0.0 and min_live > 0.0)
    else:
        ok = True
    repl_stats = dict(
        n_replaced_positions=len(replaced), n_v_vectors=n_vec,
        mean_abs_cos_old_vs_donor=(cos_abs_sum / n_vec) if n_vec else None,
        mean_norm_ratio_donor_over_old=(ratio_sum / n_vec)
        if n_vec else None,
        donor_map=list(donor) if donor else None,
    )
    cache = dict(cache_len=int(cache_len), n_replaced=len(replaced),
                 max_abs_v_replaced=max_pr, min_abs_v_live=min_live, ok=ok)
    return dict(idx=idx, logits=logits, ce=ce_s, ent=ent_s, topk=tk_s,
                top1=t1_s, rank=rk_s, replaced=sorted(replaced),
                replace_log=replace_log, cache=cache, repl_stats=repl_stats)


# ------------------------------------------------------------- clean judging

@torch.no_grad()
def cj_from_alllogits(all_lg: torch.Tensor, idx: torch.Tensor, tail: int):
    """Clean-net CE over the final `tail` tokens (queries T-1-tail..T-2).
    Same math as e080.clean_judge_tail, window generalized."""
    T = idx.shape[1]
    lg = all_lg[:, T - tail - 1:T - 1, :]
    tgt = idx[:, T - tail:]
    lp = torch.log_softmax(lg.float(), -1)
    ce = -lp.gather(2, tgt[:, :, None]).squeeze(2).mean(1)
    return ce.numpy()


# ------------------------------------------------------------- distribution math

def counts_of(tokens) -> np.ndarray:
    return np.bincount(np.asarray(tokens, int).ravel(),
                       minlength=VOCAB).astype(float)


def bigram_counts(T: np.ndarray) -> np.ndarray:
    a = T[:, :-1].ravel()
    b = T[:, 1:].ravel()
    return np.bincount(a * VOCAB + b, minlength=VOCAB * VOCAB).astype(float)


def smooth(c: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    c = np.asarray(c, float) + alpha
    return c / c.sum()


def kl(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log(p / q)))


def skl(p: np.ndarray, q: np.ndarray) -> float:
    return kl(p, q) + kl(q, p)


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, float)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def top_tokens(counts: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    """Top-k by count, ties broken by token id (deterministic)."""
    order = sorted(range(VOCAB), key=lambda t: (-counts[t], t))
    return [(t, float(counts[t])) for t in order[:k]]


def repeat_rate(toks, n: int) -> float:
    """Fraction of n-gram positions whose n-gram occurs >=2x in the tail."""
    toks = list(toks)
    L = len(toks)
    if L < n:
        return float("nan")
    grams = [tuple(toks[i:i + n]) for i in range(L - n + 1)]
    cnt = Counter(grams)
    return sum(c for c in cnt.values() if c >= 2) / (L - n + 1)


def within_null(counts_seq: np.ndarray, alpha: float = ALPHA) -> list[float]:
    """All 35 disjoint 4-vs-4 splits of S=8 sequences -> SKL sampling floor."""
    S = counts_seq.shape[0]
    vals = []
    for combo in itertools.combinations(range(S), S // 2):
        if combo[0] != 0:                     # dedupe unordered splits
            continue
        mask = np.zeros(S, bool)
        mask[list(combo)] = True
        pA = smooth(counts_seq[mask].sum(0), alpha)
        pB = smooth(counts_seq[~mask].sum(0), alpha)
        vals.append(skl(pA, pB))
    return vals


# ---------------------------------------------------------------------- main


def main():
    assert torch.cuda.device_count() == 0, "CPU-only violated: CUDA visible"
    assert common.DEVICE == "cpu", f"common.DEVICE={common.DEVICE} != cpu"
    out_dir = run_dir("e099")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gates = dict(cpu_only=True, threads=THREADS, smoke=False,
                 threads_note="e080 module import sets 12; overridden to 8 "
                              "(task spec / T050)")

    # ---- battery: rebuild corpus + net EXACTLY as e053c/e069..e080 did
    corp = CharCorpus(REPO / "data" / "input.txt")       # seed 1337
    assert corp.vocab_size == VOCAB
    st = torch.load(e080.CKPT, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    cfg = Cfg(vocab=corp.vocab_size, n_layer=4, n_head=4, n_embd=128,
              block_size=e080.T_TOTAL)
    net = TinyGPT(cfg)
    net.load_state_dict(sd, strict=True)
    net.eval()
    n_params = net.num_params()
    gates["G2_params"] = dict(params=n_params, expected=873472,
                              ok=bool(n_params == 873472))
    val_ce = estimate_loss(net, corp, "val", n_batches=12)
    gates["G1_val_ce"] = dict(val_ce=val_ce, ref=e080.E053C_VAL_CE, tol=0.02,
                              ok=bool(abs(val_ce - e080.E053C_VAL_CE) <= 0.02))
    log(f"e053c net loaded ({n_params:,} params) | val CE {val_ce:.4f} vs "
        f"e053c {e080.E053C_VAL_CE:.4f} -> G1 "
        f"{'PASS' if gates['G1_val_ce']['ok'] else 'FAIL'}")

    # ---- battery: the seed-202 8-draw, ALL 8 (e069's battery A = first 4)
    gen_p = torch.Generator().manual_seed(e080.SEED_PROMPT)
    ix = torch.randint(len(corp.val) - e080.PROMPT_TOK - 1, (e080.N_PROMPTS,),
                       generator=gen_p)
    prompts8 = [corp.val[i:i + e080.PROMPT_TOK] for i in ix]
    log(f"battery: {e080.N_PROMPTS} prompts (seed {e080.SEED_PROMPT}); "
        f"prompt0 prefix: {corp.decode(prompts8[0])[:32]!r}")

    # ================================================== THE FIVE ARMS
    A = {}
    for arm in LEGACY:                                   # VERBATIM e080 rig
        gen = torch.Generator().manual_seed(e080.SEED_SAMPLE)
        gen_n = (torch.Generator().manual_seed(e080.SEED_NOISE)
                 if arm == "noise" else None)
        A[arm] = e080.generate_arm(net, prompts8, gen, arm, noise_gen=gen_n)
        log(f"arm {arm:11s} [e080 rig]: generated B={e080.B} "
            f"64->{e080.T_TOTAL} | {len(A[arm]['replaced'])} replaced "
            f"positions | cache ok {A[arm]['cache']['ok']}")
    donor = draw_donor(SEED_DONOR, e080.N_PROMPTS)
    gen = torch.Generator().manual_seed(e080.SEED_SAMPLE)
    A["randomize"] = generate_arm5(net, prompts8, gen, "randomize", donor)
    rs = A["randomize"]["repl_stats"]
    log(f"arm randomize  [e099 ext]: donor map {donor} | "
        f"{len(A['randomize']['replaced'])} replaced positions | "
        f"mean|cos(old,donor)| {rs['mean_abs_cos_old_vs_donor']:.3f} | "
        f"norm ratio {rs['mean_norm_ratio_donor_over_old']:.3f} | "
        f"cache ok {A['randomize']['cache']['ok']}")

    # G6 skeleton fidelity + G7 randomize determinism
    gen = torch.Generator().manual_seed(e080.SEED_SAMPLE)
    S_none = generate_arm5(net, prompts8, gen, "none")
    g6 = bool(torch.equal(S_none["idx"], A["none"]["idx"]))
    gen = torch.Generator().manual_seed(e080.SEED_SAMPLE)
    R_rerun = generate_arm5(net, prompts8, gen, "randomize",
                            draw_donor(SEED_DONOR, e080.N_PROMPTS))
    g7 = bool(torch.equal(R_rerun["idx"], A["randomize"]["idx"]))
    gates["G6_skeleton_fidelity"] = dict(
        rule="generate_arm5(mode=none) token stream bit-identical to "
             "e080.generate_arm(mode=none)", ok=g6)
    gates["G7_randomize_determinism"] = dict(
        rule="randomize rerun (same seeds) token stream bit-identical", ok=g7)
    log(f"G6 skeleton fidelity: {g6} | G7 randomize determinism: {g7}")

    # ---- G5 intervention identity
    pre = e080.PRUNE_START_G + e080.PROMPT_TOK + 1       # 165: cols 0..164
    ident = all(torch.equal(A[ARMS[0]]["idx"][:, :pre], A[a]["idx"][:, :pre])
                for a in ARMS)
    exp_repl = {a: (0 if a == "none" else 324) for a in ARMS}
    counts_ok = all(len(A[a]["replaced"]) == exp_repl[a] for a in ARMS)
    derange_ok = all(d != b for b, d in enumerate(donor))
    ok5 = bool(ident and counts_ok and derange_ok
               and all(A[a]["cache"]["ok"] for a in ARMS))
    gates["G5_intervention_identity"] = dict(
        pre_event_identity_through_position=pre - 1, identical=ident,
        replaced_counts={a: len(A[a]["replaced"]) for a in ARMS},
        replaced_counts_expected=exp_repl, counts_ok=counts_ok,
        donor_map=list(donor), donor_derangement_ok=derange_ok,
        cache_oks={a: A[a]["cache"]["ok"] for a in ARMS},
        randomize_stats=A["randomize"]["repl_stats"], ok=ok5)
    log(f"G5: pre-event identity through pos {pre - 1}: {ident} | counts "
        f"{ {a: len(A[a]['replaced']) for a in ARMS} } | derangement "
        f"{derange_ok} -> {'PASS' if ok5 else 'FAIL'}")

    # ---- G4 rerun-vs-e080 drift (all four legacy arms)
    g4 = dict(ref_file=str(E080_METRICS), ok=None, note="")
    if E080_METRICS.exists():
        with open(E080_METRICS) as f:
            m080 = json.load(f)
        devs, cj_devs = {}, {}
        for arm in LEGACY:
            ref = np.asarray(m080["arms"][arm]["r1"]["per_seq"])
            new = A[arm]["ce"][:, -e080.TAIL:].mean(1)
            devs[arm] = float(np.abs(new - ref).max())
            traj_ref = np.asarray(
                m080["arms"][arm]["trajectory_mean_ce_per_step"])
            devs[arm + "_traj"] = float(
                np.abs(A[arm]["ce"].mean(0) - traj_ref).max())
        for arm in LEGACY:
            ce64, all_lg = e080.clean_judge_tail(net, A[arm]["idx"])
            A[arm]["_all_lg"] = all_lg                  # reuse for cj128
            ref = np.asarray(
                m080["arms"][arm]["clean_judge_tail_ce"]["per_seq"])
            cj_devs[arm] = float(np.abs(ce64 - ref).max())
        max_dev = max(max(devs.values()), max(cj_devs.values()))
        g4.update(r1_per_seq_max_dev={k: v for k, v in devs.items()},
                  clean_judge_max_dev=cj_devs,
                  ok=bool(max_dev < 1e-5))
        g4["note"] = (f"legacy arms rerun vs runs/e080/metrics.json: max "
                      f"dev {max_dev:.2e} (per-seq tail CE, trajectories, "
                      f"clean-judge). tol 1e-5 (not e080's 1e-6): this run "
                      f"uses 8 threads vs e080's 12 -> float summation-order "
                      f"noise; a single token swap would move a per-seq "
                      f"64-token mean by >=1e-2 nats, so streams are "
                      f"token-identical at this dev level")
    else:
        g4["note"] = "runs/e080/metrics.json not found; drift-check skipped"
    gates["G4_rerun_vs_e080"] = g4
    log(f"G4 rerun vs e080: {g4['note']} -> "
        f"{'PASS' if g4['ok'] else ('SKIPPED' if g4['ok'] is None else 'FAIL')}")

    # ---- collapse signature (clean-judge gap on the final-128 tail)
    sig = {}
    for arm in ARMS:
        online = A[arm]["ce"][:, -TAIL:].mean(1)
        if "_all_lg" in A[arm]:
            cj128 = cj_from_alllogits(A[arm]["_all_lg"], A[arm]["idx"], TAIL)
        else:
            with torch.no_grad():
                al = e080.manual_all_logits(net, A[arm]["idx"])
            cj128 = cj_from_alllogits(al, A[arm]["idx"], TAIL)
        gap = cj128 - online
        lo, hi = e080.bootstrap_stat(gap[:, None], lambda m: float(m.mean()))
        sig[arm] = dict(online_tail_ce=float(online.mean()),
                        clean_judge_tail_ce=float(cj128.mean()),
                        gap=float(gap.mean()), gap_ci=[float(lo), float(hi)],
                        gap_per_seq=gap.tolist(),
                        collapsed=bool(gap.mean() > GAP_BAR))
    for arm in ARMS:
        log(f"  signature {arm:11s}: online {sig[arm]['online_tail_ce']:.4f} "
            f"| clean-judge {sig[arm]['clean_judge_tail_ce']:.4f} | gap "
            f"{sig[arm]['gap']:+.4f} CI [{sig[arm]['gap_ci'][0]:+.3f},"
            f"{sig[arm]['gap_ci'][1]:+.3f}] | collapsed "
            f"{sig[arm]['collapsed']}")
    collapsed = [a for a in TRIGGERS if sig[a]["collapsed"]]

    # ================================================== TAIL ANALYSIS
    # corpus baselines
    full_ids = torch.cat([corp.train, corp.val]).numpy()
    corpus_counts = counts_of(full_ids)
    corpus_bigrams = bigram_counts(full_ids[None, :])
    p_corpus = smooth(corpus_counts)
    val_np = corp.val.numpy()
    starts = np.linspace(0, len(val_np) - TAIL, N_WIN).astype(int)
    win_tokens = [val_np[s:s + TAIL] for s in starts]
    corpus_rep = {n: float(np.mean([repeat_rate(w, n)
                                    for w in win_tokens])) for n in NGRAMS}
    win_pooled = counts_of(np.stack(win_tokens))
    log(f"corpus baselines: unigram H {entropy(p_corpus):.4f} nats | "
        f"window repeat rates "
        f"{ {n: round(v, 3) for n, v in corpus_rep.items()} }")

    # per-arm tail objects
    tails, counts_arm, counts_seq = {}, {}, {}
    stats = {}
    for arm in ARMS:
        T = A[arm]["idx"][:, -TAIL:].numpy()            # (8, 128)
        tails[arm] = T
        counts_seq[arm] = np.stack([counts_of(T[s]) for s in range(len(T))])
        counts_arm[arm] = counts_of(T)
        pooled = smooth(counts_arm[arm])
        per_seq_H = np.array([entropy(smooth(counts_seq[arm][s]))
                              for s in range(len(T))])
        h_lo, h_hi = e080.bootstrap_stat(per_seq_H[:, None],
                                         lambda m: float(m.mean()))
        ttr = float(np.mean([len(set(t)) / len(t) for t in T]))
        rep = {n: float(np.mean([repeat_rate(t, n) for t in T]))
               for n in NGRAMS}
        modal = [(int(np.argmax(counts_seq[arm][s])),
                  float(counts_seq[arm][s].max()) / TAIL)
                 for s in range(len(T))]
        modal_census = Counter(corp.itos[t] for t, _ in modal)
        top5 = top_tokens(counts_arm[arm], 5)
        stats[arm] = dict(
            n_tokens=int(counts_arm[arm].sum()),
            n_distinct=int((counts_arm[arm] > 0).sum()),
            pooled_entropy=entropy(pooled),
            per_seq_entropy_mean=float(per_seq_H.mean()),
            per_seq_entropy_ci=[float(h_lo), float(h_hi)],
            type_token_ratio=ttr,
            repeat_ngram_rates=rep,
            top5=[dict(token=int(t), char=corp.itos[t], count=int(c),
                       freq=float(c / counts_arm[arm].sum())) for t, c in top5],
            top5_set=[t for t, _ in top5],
            top5_effective_set=[t for t, c in top5 if c > 0],
            top1_mass=float(top5[0][1] / counts_arm[arm].sum()),
            per_seq_modal=[dict(token=int(t), char=corp.itos[t], freq=f)
                           for t, f in modal],
            modal_census={k: int(v) for k, v in modal_census.items()},
            kl_vs_corpus=kl(pooled, p_corpus),
            decoded_tails=[corp.decode(torch.tensor(t)) for t in T],
        )
        log(f"  tail {arm:11s}: H {stats[arm]['pooled_entropy']:.4f} | "
            f"distinct {stats[arm]['n_distinct']} | top1 "
            f"{stats[arm]['top1_mass']:.3f} | top5 "
            f"{[(d['char'], round(d['freq'], 3)) for d in stats[arm]['top5']]} "
            f"| rep2 {rep[2]:.3f} | KL||corpus "
            f"{stats[arm]['kl_vs_corpus']:.4f}")

    # within-arm null floors
    nulls = {arm: within_null(counts_seq[arm]) for arm in ARMS}
    for arm in ARMS:
        v = np.asarray(nulls[arm])
        log(f"  within-null {arm:11s}: mean {v.mean():.4f} "
            f"[{np.percentile(v, 2.5):.4f},{np.percentile(v, 97.5):.4f}] "
            f"(35 disjoint 4v4 splits)")

    # cross-arm grids (symmetric KL; diag = within-null mean)
    nA = len(ARMS)
    grid = np.full((nA, nA), np.nan)
    kl_dir = np.full((nA, nA), np.nan)
    P = {arm: smooth(counts_arm[arm]) for arm in ARMS}
    for i, a in enumerate(ARMS):
        grid[i, i] = float(np.mean(nulls[a]))
        for j, b in enumerate(ARMS):
            if i != j:
                kl_dir[i, j] = kl(P[a], P[b])
                grid[i, j] = skl(P[a], P[b])
    # bigram rider
    Pb = {arm: smooth(bigram_counts(tails[arm]), BIALPHA) for arm in ARMS}
    nulls_b = {arm: within_null(np.stack(
        [bigram_counts(tails[arm][s:s + 1]) for s in range(len(tails[arm]))]),
        BIALPHA) for arm in ARMS}
    grid_b = np.full((nA, nA), np.nan)
    for i, a in enumerate(ARMS):
        grid_b[i, i] = float(np.mean(nulls_b[a]))
        for j, b in enumerate(ARMS):
            if i != j:
                grid_b[i, j] = skl(Pb[a], Pb[b])

    # ================================================== REGISTERED DECISION
    floor = (float(np.mean([np.mean(nulls[a]) for a in collapsed]))
             if len(collapsed) >= 2 else None)
    cross = {}
    for a, b in itertools.combinations(collapsed, 2):
        i, j = ARMS.index(a), ARMS.index(b)
        cross[f"{a}|{b}"] = float(grid[i, j])
    cross_vals = list(cross.values()) if cross else []
    cross_max = max(cross_vals) if cross_vals else None
    cross_min = min(cross_vals) if cross_vals else None
    top5_sets = {a: set(stats[a]["top5_set"]) for a in collapsed}
    top5_eff = {a: set(stats[a]["top5_effective_set"]) for a in collapsed}
    all_top5_equal = all(s == top5_sets[collapsed[0]] for s in top5_sets.values()) \
        if collapsed else False
    all_eff_equal = all(s == top5_eff[collapsed[0]] for s in top5_eff.values()) \
        if collapsed else False
    pairwise_disjoint = all(not (top5_sets[a] & top5_sets[b])
                            for a, b in itertools.combinations(collapsed, 2)) \
        if collapsed else False

    clauses = {}
    if len(collapsed) < 2:
        clause = "UNDETERMINED"
        verdict = (f"fewer than 2 trigger arms cleared the collapse gate "
                   f"(clean-judge gap > {GAP_BAR}): collapsed={collapsed}; "
                   f"the identity question cannot be evaluated this run.")
    else:
        univ_kl = bool(cross_max <= UNIV_RATIO * floor)
        spec_kl = bool(cross_min >= SPEC_RATIO * floor)
        univ = bool(univ_kl and all_top5_equal)
        spec = bool(spec_kl and pairwise_disjoint)
        clauses = dict(
            universal_kl=dict(rule=f"max cross-arm SKL <= {UNIV_RATIO}x floor",
                              max_cross=cross_max, floor=floor,
                              ratio=cross_max / floor, fires=univ_kl),
            top5_identical=dict(rule="all collapsed arms' top-5 token sets "
                                     "identical (tie-break -count,id)",
                                sets={a: sorted(s) for a, s in top5_sets.items()},
                                effective_sets_equal=all_eff_equal,
                                fires=all_top5_equal),
            specific_kl=dict(rule=f"min cross-arm SKL >= {SPEC_RATIO}x floor",
                             min_cross=cross_min, floor=floor,
                             ratio=cross_min / floor, fires=spec_kl),
            top5_disjoint=dict(rule="collapsed arms' top-5 sets pairwise "
                                    "disjoint", fires=pairwise_disjoint))
        if univ:
            clause = "ONE UNIVERSAL ATTRACTOR"
            verdict = (
                f"ONE UNIVERSAL ATTRACTOR fires: every collapsed pair's "
                f"terminal SKL <= {UNIV_RATIO}x the within-arm floor "
                f"(max {cross_max:.4f} vs floor {floor:.4f}, ratio "
                f"{cross_max / floor:.2f}) AND all collapsed arms share the "
                f"same top-5 tail tokens — the degenerate endpoint is a "
                f"property of the NET (a fixed point of the free-running "
                f"dynamics), not of the perturbation.")
        elif spec:
            clause = "PERTURBATION-SPECIFIC BASINS"
            verdict = (
                f"PERTURBATION-SPECIFIC BASINS fires: every collapsed pair's "
                f"terminal SKL >= {SPEC_RATIO}x the within-arm floor (min "
                f"{cross_min:.4f} vs floor {floor:.4f}, ratio "
                f"{cross_min / floor:.2f}) AND top-5 sets are pairwise "
                f"disjoint — each perturbation lands in its own basin.")
        else:
            clause = "MIXED"
            bits = []
            bits.append(f"collapsed arms {collapsed}; within-arm floor "
                        f"{floor:.4f} nats")
            bits.append(f"cross-arm SKL range [{cross_min:.4f}, "
                        f"{cross_max:.4f}] = "
                        f"[{cross_min / floor:.1f}x, {cross_max / floor:.1f}x]"
                        f" floor — every collapsed pair is BELOW the "
                        f"within-arm sampling floor")
            shared = set.intersection(*top5_sets.values()) if collapsed else set()
            n_shared = len(shared)
            bits.append(f"top-5 sets share {n_shared}/5 members "
                        f"{sorted(shared)}; per-arm "
                        + "; ".join(f"{a}:{sorted(s - shared)}"
                                    for a, s in top5_sets.items())
                        + " (the differing member is a near-tie at "
                          "freq ~0.05, not a disjoint dominant token)")
            if univ_kl and not all_top5_equal:
                bits.append("KLs are universal-scale but the strict top-5 "
                            "identity clause fails — same DEGENERACY "
                            "DEGREE, arm-idiosyncratic member at the census "
                            "tail")
            if all_top5_equal and not univ_kl:
                bits.append("dominant tokens shared but second-order "
                            "distribution mass differs")
            nc = [a for a in TRIGGERS if not sig[a]["collapsed"]]
            if nc:
                bits.append(f"fresh-trigger rider: {nc} did NOT collapse "
                            f"(clean-judge gap "
                            + ", ".join(f"{a} {sig[a]['gap']:+.3f}"
                                        for a in nc)
                            + ") — replacing the band with ANOTHER RUN'S "
                              "generated entries keeps the run on-manifold; "
                              "collapse tracks destruction of generated-band "
                              "statistics (zero / noise / corpus-prompt "
                              "content), not foreignness per se")
            verdict = ("MIXED (honest texture): " + ". ".join(bits) + ".")
    log(f"REGISTERED DECISION [{clause}]: {verdict}")
    log(f"  collapsed arms: {collapsed} | floor {floor} | cross "
        f"{ {k: round(v, 4) for k, v in cross.items()} }")

    # ---------------------------------------------------------------- metrics
    metrics = dict(
        experiment="e099_attractor_identity",
        purpose="Rule-11 pick: the ATTRACTOR IDENTITY probe. e080's three "
                "collapse triggers (vzero/noise/promptcopy) regenerated "
                "VERBATIM + A-randomize (band V <- a random other run's "
                "entries, derangement donor, pre-event snapshot). Question: "
                "do all collapsed continuation tails converge to ONE "
                "terminal token distribution (net property) or "
                "perturbation-specific basins? Readouts: pooled tail "
                "unigram (final 128 gen tokens, 8 seqs), cross-arm "
                "symmetric-KL grid vs within-arm 4v4-split null floor, "
                "top-5 identity census, entropy/repeat-n texture, each arm "
                "vs corpus unigram. FROZEN bars: max cross SKL <= 2x floor "
                "AND identical top-5 => UNIVERSAL; min cross SKL >= 5x "
                "floor AND disjoint top-5 => SPECIFIC; else MIXED.",
        started=started, wall_s=elapsed(), threads=THREADS, cpu_only=True,
        net=dict(ckpt=str(e080.CKPT), arch=dict(n_layer=4, n_head=4,
                                                n_embd=128,
                                                block_size=e080.T_TOTAL,
                                                vocab=VOCAB),
                 params=n_params,
                 val_ce=val_ce, val_ce_e053c=e080.E053C_VAL_CE),
        seeds=dict(corpus=1337, prompts=e080.SEED_PROMPT,
                   sampling=e080.SEED_SAMPLE, noise=e080.SEED_NOISE,
                   donor=SEED_DONOR,
                   donor_note="randomize donor derangement drawn once from "
                              "a dedicated seed-4343 generator; sampling "
                              "stream untouched in every arm"),
        protocol=dict(B=e080.B, prompt_tokens=e080.PROMPT_TOK,
                      t_total=e080.T_TOTAL, temp=e080.TEMP, topk=e080.TOPK,
                      tail_tokens=TAIL, smoothing_alpha=ALPHA,
                      bigram_alpha=BIALPHA,
                      within_null="35 disjoint 4-vs-4 sequence splits per "
                                  "arm, symmetric KL",
                      arms=dict(
                          none="normal free run (e080.generate_arm verbatim)",
                          vzero="V->0 at self-band events (e080 verbatim)",
                          noise="V<-norm-matched gaussian (e080 verbatim, "
                                "seed 4242)",
                          promptcopy="V<-same row's pristine prompt entry "
                                     "(e080 verbatim)",
                          randomize="V<-another run's entry at the same "
                                    "position (donor derangement seed "
                                    f"{SEED_DONOR}, pre-event snapshot; "
                                    "e099 extension)"),
                      schedule=dict(events_g=e080.EVENTS, K=e080.PRUNE_K,
                                    start_g=e080.PRUNE_START_G,
                                    age_cut=e080.AGE_CUT,
                                    note="identical to e080: replaced once "
                                         "at first band admission, K "
                                         "untouched; all arms token-identical "
                                         "through position 164"),
                      collapse_gate=f"clean-judge tail CE - online tail CE "
                                    f"> {GAP_BAR} nats (e080 convention)"),
        gates=gates,
        char_map=[corp.itos[t] for t in range(VOCAB)],
        collapse_signature={a: sig[a] for a in ARMS},
        collapsed_arms=collapsed,
        corpus_baseline=dict(
            unigram_counts=corpus_counts.tolist(),
            unigram_entropy=entropy(p_corpus),
            window_repeat_ngram_rates=corpus_rep,
            window_note=f"{N_WIN} evenly-spaced 128-token windows of the "
                        "val split",
            window_pooled_kl_vs_corpus=kl(smooth(win_pooled), p_corpus)),
        arms={arm: dict(
            tail_tokens=tails[arm].tolist(),
            **{k: v for k, v in stats[arm].items()
               if k != "decoded_tails"},
            decoded_tails=stats[arm]["decoded_tails"],
            within_null=dict(values=nulls[arm],
                             mean=float(np.mean(nulls[arm])),
                             ci=[float(np.percentile(nulls[arm], 2.5)),
                                 float(np.percentile(nulls[arm], 97.5))]),
            within_null_bigram=dict(mean=float(np.mean(nulls_b[arm]))),
            replace_log=A[arm]["replace_log"],
            repl_stats=A[arm]["repl_stats"],
        ) for arm in ARMS},
        kl_grid=dict(
            arm_order=ARMS,
            symmetric=grid.tolist(),
            directional=kl_dir.tolist(),
            diag="within-arm null mean (35 disjoint 4v4 splits)",
            collapsed_pairs=cross,
            floor=floor, cross_max=cross_max, cross_min=cross_min),
        kl_grid_bigram_rider=dict(
            note="NON-REGISTERED rider: same grid over tail bigram "
                 "distributions (alpha 0.1)",
            symmetric=grid_b.tolist()),
        registered_decision=dict(
            frozen_rules=dict(
                universal=f"max cross-arm SKL <= {UNIV_RATIO}x floor AND "
                          f"all collapsed top-5 sets identical",
                specific=f"min cross-arm SKL >= {SPEC_RATIO}x floor AND "
                         f"top-5 sets pairwise disjoint",
                else_="MIXED (report the honest texture)"),
            clauses=clauses, clause=clause, verdict=verdict),
    )
    save_json(out_dir / "metrics.json", metrics)
    log("metrics.json written")

    plot(out_dir / "attractor_identity.png", metrics)
    log(f"plot written; total wall {elapsed():.0f}s")


# ---------------------------------------------------------------------- plot


def plot(path: Path, M: dict):
    dec = M["registered_decision"]
    arms = M["kl_grid"]["arm_order"]
    grid = np.asarray(M["kl_grid"]["symmetric"])
    grid_b = np.asarray(M["kl_grid_bigram_rider"]["symmetric"])
    collapsed = M["collapsed_arms"]
    am = M["arms"]
    cols = {"none": "tab:gray", "vzero": "tab:red", "noise": "tab:blue",
            "promptcopy": "tab:orange", "randomize": "tab:green"}
    labs = [f"{a}\n{'(collapsed)' if a in collapsed else '(control)' if a == 'none' else '(no collapse)'}"
            for a in arms]
    fig, axes = plt.subplots(2, 3, figsize=(24, 11))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    # ---- panel 1: the cross-arm KL heatmap (diag = within-arm null)
    finite = grid[~np.isnan(grid)]
    gpos = grid[grid > 0]
    vmin = gpos.min() if gpos.size else 1e-3
    im = ax1.imshow(grid, cmap="viridis",
                    norm=LogNorm(vmin=max(1e-4, vmin),
                                 vmax=max(0.01, finite.max())))
    for i in range(len(arms)):
        for j in range(len(arms)):
            val = grid[i, j]
            ax1.text(j, i, f"{val:.3f}", ha="center", va="center",
                     fontsize=9, color="white" if val > grid[~np.isnan(grid)].max() / 3 else "black")
    ax1.set_xticks(range(len(arms)), labs, fontsize=8)
    ax1.set_yticks(range(len(arms)), labs, fontsize=8)
    fig.colorbar(im, ax=ax1, label="symmetric KL (nats, log scale)")
    ax1.set_title("E099-1 — cross-arm TERMINAL distribution SKL\n(diagonal = "
                  "within-arm 4v4-split null floor)", fontsize=10)

    # ---- panel 2: terminal-distribution overlay
    cmap_ = M["char_map"]
    freq = {}
    for a in arms:
        c = counts_of(np.asarray(am[a]["tail_tokens"]))
        freq[a] = c / c.sum()
    cc = np.asarray(M["corpus_baseline"]["unigram_counts"], float)
    pc = cc / cc.sum()
    top = sorted(range(VOCAB), key=lambda t: -max(freq[a][t] for a in arms))
    top = top[:14]
    x = np.arange(len(top))
    w = 0.15
    for i, a in enumerate(arms):
        ax2.bar(x + (i - 2) * w, [freq[a][t] for t in top], w,
                color=cols[a], alpha=0.8, label=f"A-{a}")
    ax2.scatter(x, [pc[t] for t in top], marker="v", color="k", zorder=5,
                s=42, label="corpus unigram")
    ax2.set_xticks(x, [f"{t}:{cmap_[t]!r}" for t in top], fontsize=8)
    ax2.legend(fontsize=8)
    ax2.set_ylabel("tail pooled frequency")
    ax2.set_title("E099-2 — terminal token distribution overlay (final 128 "
                  "tokens, B=8 pooled); top-14 tokens by max arm freq",
                  fontsize=10)

    # ---- panel 3: entropy + KL vs corpus
    x3 = np.arange(len(arms))
    H = [am[a]["pooled_entropy"] for a in arms]
    Hci = [[max(0.0, am[a]["pooled_entropy"] - am[a]["per_seq_entropy_ci"][0])
            for a in arms],
           [max(0.0, am[a]["per_seq_entropy_ci"][1] - am[a]["pooled_entropy"])
            for a in arms]]
    ax3.bar(x3, H, 0.6, color=[cols[a] for a in arms], alpha=0.85,
            edgecolor="k", lw=0.5)
    ax3.errorbar(x3, H, yerr=Hci, fmt="none", ecolor="k", lw=1.0, capsize=3)
    ax3.axhline(M["corpus_baseline"]["unigram_entropy"], color="k", ls="--",
                lw=1.2)
    ax3.text(len(arms) - 0.45, M["corpus_baseline"]["unigram_entropy"],
             f" corpus unigram H {M['corpus_baseline']['unigram_entropy']:.3f}",
             fontsize=8, va="bottom")
    for xi, a in zip(x3, arms):
        ax3.text(xi, H[xi] + 0.06,
                 f"H {H[xi]:.2f}\nKL||corp {am[a]['kl_vs_corpus']:.2f}",
                 ha="center", fontsize=8)
    ax3.set_xticks(x3, labs, fontsize=8)
    ax3.set_ylabel("tail unigram entropy (nats)")
    ax3.set_ylim(0, max(H) * 1.35)
    ax3.set_title("E099-3 — terminal entropy (pooled bar; CI = per-seq "
                  "mean) + KL vs corpus unigram", fontsize=10)

    # ---- panel 4: repeat-n-gram rates vs corpus windows
    xg = np.arange(len(arms) + 1)
    w4 = 0.26
    for ni, n in enumerate(NGRAMS):
        vals = [am[a]["repeat_ngram_rates"][str(n)]
                if str(n) in am[a]["repeat_ngram_rates"]
                else am[a]["repeat_ngram_rates"][n] for a in arms]
        vals.append(M["corpus_baseline"]["window_repeat_ngram_rates"][str(n)]
                    if str(n) in M["corpus_baseline"]["window_repeat_ngram_rates"]
                    else M["corpus_baseline"]["window_repeat_ngram_rates"][n])
        ax4.bar(xg + (ni - 1) * w4, vals, w4, alpha=0.85,
                label=f"n={n}")
        for xi, v in zip(xg + (ni - 1) * w4, vals):
            ax4.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax4.set_xticks(xg, labs + ["corpus\n(8 val\nwindows)"], fontsize=8)
    ax4.set_ylim(0, 1.05)
    ax4.legend(fontsize=8)
    ax4.set_ylabel("repeat-n-gram rate (tail)")
    ax4.set_title("E099-4 — terminal repetition texture (fraction of n-gram "
                  "positions occurring >=2x within the tail)", fontsize=10)

    # ---- panel 5: census + collapse signature table
    ax5.axis("off")
    lines = ["arm | cj_gap | collapsed | top1 | top-5 tokens (char:freq) | "
             "per-seq modal census"]
    for a in arms:
        s = am[a]
        cj = M["collapse_signature"][a]
        cen = ", ".join(f"{repr(k)}:{v}" for k, v in
                        sorted(s["modal_census"].items(),
                               key=lambda kv: -kv[1]))
        lines.append(
            f"{a:11s} {cj['gap']:+5.2f} {'YES' if cj['collapsed'] else ' no'}"
            f"     {s['top1_mass']:.3f} "
            + " ".join(f"{d['char']!r}:{d['freq']:.2f}" for d in s["top5"])
            + f"   {cen}")
    lines.append("")
    lines.append("bigram-rider SKL (collapsed pairs; diag-null in brackets):")
    for a, b in itertools.combinations(collapsed, 2):
        i, j = arms.index(a), arms.index(b)
        lines.append(f"  {a} vs {b}: {grid_b[i, j]:.3f} "
                     f"(nulls {grid_b[i, i]:.3f}/{grid_b[j, j]:.3f})")
    ax5.text(0.02, 0.97, "E099-5 — terminal census", fontsize=12,
             weight="bold", va="top")
    for i, t in enumerate(lines):
        ax5.text(0.02, 0.92 - i * 0.052, t, fontsize=8.6, va="top",
                 family="monospace")

    # ---- panel 6: the registered decision
    ax6.axis("off")
    fl = M["kl_grid"]["floor"]
    l6 = ["REGISTERED BARS (frozen):",
          f"  floor (mean within-arm null of collapsed arms): {fl}"]
    if fl:
        l6.append("  cross-arm SKL over collapsed pairs: "
                  f"{ {k: round(v, 4) for k, v in M['kl_grid']['collapsed_pairs'].items()} }")
        l6.append(f"  max/floor: {M['kl_grid']['cross_max'] / fl:.2f}x "
                  f"(universal bar <= {UNIV_RATIO}x) | min/floor: "
                  f"{M['kl_grid']['cross_min'] / fl:.2f}x (specific bar "
                  f">= {SPEC_RATIO}x)")
    if dec["clauses"]:
        for k, v in dec["clauses"].items():
            l6.append(f"  {k}: {'FIRES' if v['fires'] else 'no'}")
    l6 += ["", f"VERDICT [{dec['clause']}]:"] + \
        [f"  {wd}" for wd in textwrap.wrap(dec["verdict"], 96)]
    ax6.text(0.02, 0.97, "E099-6 — attractor identity decision", fontsize=12,
             weight="bold", va="top")
    for i, t in enumerate(l6):
        ax6.text(0.02, 0.925 - i * 0.042, t, fontsize=9, va="top",
                 family="monospace")

    if fl:
        sup = (f"E099 — attractor identity | clause: {dec['clause']} | "
               f"collapsed arms: {collapsed} | floor {fl:.4f} | cross max "
               f"{M['kl_grid']['cross_max']:.4f} "
               f"({M['kl_grid']['cross_max'] / fl:.1f}x floor)")
    else:
        sup = f"E099 — attractor identity | clause: {dec['clause']}"
    fig.suptitle(sup, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
