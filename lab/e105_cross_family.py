"""E105 — the CROSS-FAMILY anchor test (T055's registration, the family
boundary's sharpest form).

[REGISTERED DESIGN — frozen in this docstring BEFORE any compute]

THE QUESTION (T055): e099's A-randomize rider proved the anchor is
NET-FAMILY-SPECIFIC — sibling-run entries (same trained net, different seed-7
stream) keep the run healthy (gap +0.027) while corpus text collapses
(promptcopy +5.33). WHERE is the family boundary? Is it "the same trained
weights" (the anchor reads the generator's IDENTITY) or "any net-generated
text" (the anchor reads generic generated-text shape)?

DESIGN: the e099 five-arm structure VERBATIM (e080 rig: e053c net, fixed-anchor
64->512 free run, B=8 seed-202 battery, per-arm seed-7 matched sampling
stream; interventions at the same events g=100..420 K=32 on the self band
age>96, replaced once at first admission, K untouched) + ONE new arm:

  1. none          e080.generate_arm verbatim (control).
  2. vzero         e080 verbatim (known collapse).
  3. noise         e080 verbatim, seed 4242 (known collapse).
  4. promptcopy    e080 verbatim — CORPUS-PROMPT control (replicates the
                   collapse, e099 gap +5.33).
  5. randomize     e099.generate_arm5 VERBATIM (seed-4343 derangement,
                   pre-event snapshot donors = sibling battery rows of the
                   same trained net) — SIBLING-RUN control (replicates the
                   healthy +0.027).
  6. crossfamily   NEW: same skeleton/schedule; every newly admitted band
                   position p's V vectors are spliced from the e021_task
                   COPY-TASK net's own free run on its own corpus — a
                   DIFFERENTLY TRAINED net (6L/6H/192d, block 256, vocab 65,
                   step 1980, copy acc 0.9996; corpus data/e021_task.txt
                   "ID: XXXXX\\n<filler>\\nCOPY: XXXXX\\n").
                   - donor runs: TWO windows of B=8 rows each, free run
                     64->256 (192 native generation steps) on 64-token val
                     prompts of e021_task.txt; e080 sampling math (temp 0.8,
                     top-k 40, per-row shared generator). Dedicated seeds:
                     window1 prompts 5601 / sampling 5701; window2 prompts
                     5602 / sampling 5702 — never touching any arm stream.
                   - position map (the block-256 donor cannot host positions
                     >=256 natively; NO wpe aliasing, NO out-of-distribution
                     donor compute — every spliced vector is a genuine native
                     donor computation): p <= 255 <- donor1[:, :, p, :]
                     (position-matched); p in [256,387] <- donor2[:, :, p-192,
                     :] (donor-native positions 64..195, position-shifted
                     overflow; 324 band positions total, 384 native donor
                     positions available, zero reuse).
                   - layer map: recipient layer li (0..3) <- donor layer li;
                     head map: recipient head h (0..3) <- donor head h
                     (head_dim 32 on both sides; donor L4-5/H4-5 unused).
                   - row map: derangement d(b) != b from dedicated seed 4646.
                   - donor content FROZEN before the recipient run (pristine
                     by construction; no snapshot machinery needed).

READOUTS (e099 convention): final-128 tail (positions 384..511, 8 seqs x 128
= 1024 tokens/arm): online tail CE, clean-judge tail CE (full clean forward,
manual_all_logits), GAP = clean-judge - online (bootstrap CI); pooled tail
unigram; within-arm 4-vs-4-split null SKL (35 splits); 6x6 cross-arm
symmetric-KL grid; top-5 census; KL vs corpus unigram. Riders (texture):
donor pooled tail unigram (final 128 tokens of each donor row, both windows)
vs each arm and both corpora.

REGISTERED BARS (frozen; e099-calibrated cluster rule — verified to classify
all five e099 arms correctly on the published grid, where healthy<->collapsed
SKLs sit at 1.6-3.0x the collapsed floor and intra-cluster SKLs at 0.2-0.7x):
  clusters: healthy = {none, randomize}; collapsed = trigger arms with
  gap > 1.0 (expect vzero/noise/promptcopy). floor_c = mean within-arm null
  of collapsed arms; floor_h = mean within-arm null of healthy arms;
  d_c(x) = mean SKL(x, collapsed); d_h(x) = mean SKL(x, healthy);
  joins_collapsed(x) := d_c(x) <= 2*floor_c AND d_h(x) > d_c(x);
  joins_healthy(x)   := d_h(x) <= 2*floor_h AND d_c(x) > d_h(x).

  FAMILY = TRAINED WEIGHTS (sharpest boundary) fires iff
      gap(crossfamily) > 1.0 AND joins_collapsed(crossfamily)
      => the anchor reads the generator's identity: the family is the trained
         weights themselves.
  FAMILY = GENERIC STATS (weaker, broader) fires iff
      gap(crossfamily) < 0.3 AND joins_healthy(crossfamily)
      => the anchor reads any competent char-LM's generated-text shape.
  else MIXED => report the texture (incl. the 0.3 <= gap <= 1.0 gray zone
      and gap/cluster disagreement).

VALIDATION CONTROLS (required for the run to count): randomize healthy
(gap < 0.3, joins healthy cluster) AND promptcopy collapsed (gap > 1.0,
joins collapsed cluster).

GATES: G1 e053c val CE (tol 0.02); G2 params 873,472; G3 donor-net identity
(e021_task arch 6L/6H/192d/blk256/vocab65, ckpt step 1980, val CE on its own
corpus vs e063b's 1.4318 ref, soft tol 0.15; donor-run determinism
bit-identical); G4 drift vs runs/e099/metrics.json for the five legacy arms
(online/clean-judge/gap_per_seq on the 128 tail, max dev < 1e-5, same
8-thread setting); G5 intervention identity (all six arms token-identical
through position 164; replaced counts 0/324 per schedule; derangements ok;
cache checks ok); G6 crossfamily full-pipeline determinism (donors + arm
rerun bit-identical).

Run:     python lab/e105_cross_family.py
Outputs: runs/e105/metrics.json + runs/e105/cross_family.png
Envelope: NO training, NO new automations; CPU-only (CUDA masked pre-torch,
8 threads), single step, minutes. No NOTES/THINKING/QUEUE/STATE edits; no
commit (the dispatcher owns those).
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
from matplotlib.patches import Rectangle  # noqa: E402

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
import e080_prune_vs_replace as e080   # the VERBATIM rig (constants + arms)
import e099_attractor_identity as e099  # the VERBATIM 5-arm extension

THREADS = 8                                # task spec / T050: 12 thrashes box
torch.set_num_threads(THREADS)             # (e080 sets 12; e099 resets 8)

# ------------------------------------------------------------------ constants
# cluster-friendly order: healthy refs, the probe, then the collapsed trio
ARMS = ["none", "randomize", "crossfamily", "vzero", "noise", "promptcopy"]
LEGACY = ["none", "vzero", "noise", "promptcopy"]      # e080's four, verbatim
TRIGGERS = ["vzero", "noise", "promptcopy", "randomize", "crossfamily"]
PROBE = "crossfamily"
HEALTHY_REFS = ["none", "randomize"]

TAIL = 128                                 # e099's tail window (final 128 gen)
VOCAB = 65
ALPHA = 0.5                                # e099 smoothing
NGRAMS = (2, 3, 4)
N_WIN = 8                                  # corpus baseline windows

# the cross-family donor net (T055: "the e063b copy-task net" = e021_task)
DONOR_CKPT = REPO / "runs" / "checkpoints" / "e021_task.train.pt"
DONOR_CORPUS = REPO / "data" / "e021_task.txt"
DONOR_BLOCK = 256                          # e021 arch: 6L/6H/192d, block 256
DONOR_GEN = DONOR_BLOCK - e080.PROMPT_TOK  # 192 native donor gen steps
DONOR_POS_SPLIT = 255                      # p <= 255 <- window1; else window2
DONOR_SHIFT = 192                          # window2 src pos = p - 192
SEED_DONOR_MAP = 4646                      # crossfamily row derangement
DONOR_WIN = {                              # dedicated donor streams
    1: dict(prompt=5601, sample=5701),
    2: dict(prompt=5602, sample=5702),
}

E099_METRICS = REPO / "runs" / "e099" / "metrics.json"
E063B_VAL_CE_TASK = 1.4318429072697958     # e063b E021_REF val_ce_task
E099_GAP_RANDOMIZE = 0.027                 # e099's healthy rider (+0.0275)
E099_GAP_PROMPTCOPY = 5.325                # e099's collapsed corpus control

# registered bars (frozen; see docstring)
GAP_BAR = 1.0                              # gap > 1 nat = collapsed (e099)
GAP_HEALTHY = 0.3                          # gap < 0.3 = healthy bar
CLUSTER_RATIO = 2.0                        # d(cluster) <= 2x floor

T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# ------------------------------------------------------------ donor machinery

@torch.no_grad()
def donor_run(net21: TinyGPT, corp21: CharCorpus, prompt_seed: int,
              sample_seed: int):
    """The e021_task net's OWN free run on ITS OWN corpus: 8 rows, 64-token
    val prompts, 192 native generation steps (64->256, block-256 net), e080
    sampling math verbatim (temp 0.8 / top-k 40 / per-row shared generator).
    Returns idx (8,256), per-step CE, and the per-layer V caches
    (each (8, 6, 256, 32)) — the cross-family anchor content."""
    Bb = e080.N_PROMPTS
    gen_p = torch.Generator().manual_seed(prompt_seed)
    ix = torch.randint(len(corp21.val) - e080.PROMPT_TOK - 1, (Bb,),
                       generator=gen_p)
    prompts = [corp21.val[i:i + e080.PROMPT_TOK] for i in ix]
    idx = torch.stack(prompts)
    gen = torch.Generator().manual_seed(sample_seed)
    logits, kv = e080.prefill_batch(net21, idx)
    ce_s = np.zeros((Bb, DONOR_GEN), float)
    for g in range(DONOR_GEN):
        t = e080.PROMPT_TOK + g
        toks = torch.zeros(Bb, dtype=torch.long)
        for j in range(Bb):
            tok, ce = e080.sample_and_ce(logits[j], gen)
            toks[j] = tok
            ce_s[j, g] = ce
        idx = torch.cat([idx, toks[:, None]], 1)
        if t < DONOR_BLOCK:
            # decode through position 255 (wpe[255] valid on a block-256
            # net) so the V cache covers ALL generated positions 64..255
            logits = e080.decode_step_batch(net21, toks, t, kv)
    V = [v.clone() for (_k, v) in kv]          # list of (8, 6, 256, 32)
    assert V[0].shape == (e080.N_PROMPTS, 6, DONOR_BLOCK, 32), \
        f"donor V cache {tuple(V[0].shape)} != (8, 6, {DONOR_BLOCK}, 32)"
    return dict(idx=idx, ce=ce_s, V=V, prompts=prompts,
                online_tail_ce=float(ce_s[:, -TAIL:].mean()))


def donor_source(p: int) -> tuple[int, int]:
    """Recipient band position p -> (window, donor-native cache position).
    p <= 255: window 1, position-matched. p in [256,387]: window 2 at
    p-192 (donor-native 64..195). Every source position is inside the
    donor's own generated band (64..255) — no prompt positions, no reuse."""
    if p <= DONOR_POS_SPLIT:
        return 1, p
    return 2, p - DONOR_SHIFT


@torch.no_grad()
def generate_arm6(net: TinyGPT, prompts, gen: torch.Generator,
                  donors: dict[int, dict], donor: list[int]):
    """e099.generate_arm5's VERBATIM SKELETON with mode='crossfamily': at
    each event, every NEWLY admitted band position p gets
    v[b,:,p,:] <- donors[w].V[li][d(b), 0:4, ps, :] (recipient layer li <-
    donor layer li, recipient head h <- donor head h, d=32 both sides; K
    untouched; replaced-stays-replaced; donor content frozen pre-run)."""
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
    dm = torch.tensor(donor, dtype=torch.long)             # (8,)
    for g in range(e080.G):
        t = e080.PROMPT_TOK + g
        if e080.is_event(g):
            band = e080.prune_positions("self", t)
            new = sorted(set(band) - replaced)
            if new:
                groups = {1: ([], []), 2: ([], [])}        # w -> (rec, src)
                for p in new:
                    w, ps = donor_source(p)
                    groups[w][0].append(p)
                    groups[w][1].append(ps)
                for w, (rec, src) in groups.items():
                    if not rec:
                        continue
                    rec_sel = torch.tensor(rec, dtype=torch.long)
                    src_sel = torch.tensor(src, dtype=torch.long)
                    for li, (_k, v) in enumerate(kv):
                        old = v[:, :, rec_sel, :].clone()              # (8,4,n,32)
                        src = donors[w]["V"][li][dm][:, 0:4, src_sel, :]
                        nrm_old = old.norm(dim=-1)
                        nrm_src = src.norm(dim=-1)
                        cos = (old * src).sum(-1) / (nrm_old
                                                    * nrm_src
                                                    ).clamp_min(1e-12)
                        cos_abs_sum += float(cos.abs().sum())
                        ratio_sum += float((nrm_src
                                           / nrm_old.clamp_min(1e-12)).sum())
                        n_vec += int(cos.numel())
                        v[:, :, rec_sel, :] = src
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
    for (_k, v) in kv:
        nv = v.abs().amax(dim=(0, 1, 3))
        max_pr = max(max_pr, float(nv[repmask].max()))
        min_live = min(min_live, float(nv[~repmask].min()))
    ok = bool(max_pr > 0.0 and min_live > 0.0)
    repl_stats = dict(
        n_replaced_positions=len(replaced), n_v_vectors=n_vec,
        mean_abs_cos_old_vs_donor=(cos_abs_sum / n_vec) if n_vec else None,
        mean_norm_ratio_donor_over_old=(ratio_sum / n_vec)
        if n_vec else None,
        donor_map=list(donor),
        position_map=("p<=255 <- donor1[:, :, p, :] (position-matched); "
                      "p in [256,387] <- donor2[:, :, p-192, :] "
                      "(donor-native 64..195, no reuse)"),
        layer_head_map=("recipient layer li (0..3) <- donor layer li; "
                        "recipient head h (0..3) <- donor head h; d=32"),
    )
    cache = dict(cache_len=int(cache_len), n_replaced=len(replaced),
                 max_abs_v_replaced=max_pr, min_abs_v_live=min_live, ok=ok)
    return dict(idx=idx, logits=logits, ce=ce_s, ent=ent_s, topk=tk_s,
                top1=t1_s, rank=rk_s, replaced=sorted(replaced),
                replace_log=replace_log, cache=cache, repl_stats=repl_stats)


# ------------------------------------------------- distribution math (e099)
counts_of = e099.counts_of
smooth = e099.smooth
kl = e099.kl
skl = e099.skl
entropy = e099.entropy
top_tokens = e099.top_tokens
repeat_rate = e099.repeat_rate
within_null = e099.within_null


def pcoa_2d(D: np.ndarray) -> np.ndarray:
    """Classical MDS (2 dims) of a dissimilarity matrix (pure numpy)."""
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    order = np.argsort(-w)[:2]
    w, V = w[order], V[:, order]
    w = np.clip(w, 1e-12, None)
    return V * np.sqrt(w)


# ---------------------------------------------------------------------- main


def main():
    assert torch.cuda.device_count() == 0, "CPU-only violated: CUDA visible"
    assert common.DEVICE == "cpu", f"common.DEVICE={common.DEVICE} != cpu"
    out_dir = run_dir("e105")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gates = dict(cpu_only=True, threads=THREADS, smoke=False,
                 threads_note="e080 module import sets 12; overridden to 8 "
                              "(task spec / T050)")

    # ---- recipient battery: e053c net EXACTLY as e080/e099 did
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
    log(f"e053c recipient net loaded ({n_params:,} params) | val CE "
        f"{val_ce:.4f} vs e053c {e080.E053C_VAL_CE:.4f} -> G1 "
        f"{'PASS' if gates['G1_val_ce']['ok'] else 'FAIL'}")

    gen_p = torch.Generator().manual_seed(e080.SEED_PROMPT)
    ix = torch.randint(len(corp.val) - e080.PROMPT_TOK - 1, (e080.N_PROMPTS,),
                       generator=gen_p)
    prompts8 = [corp.val[i:i + e080.PROMPT_TOK] for i in ix]
    log(f"battery: {e080.N_PROMPTS} prompts (seed {e080.SEED_PROMPT}); "
        f"prompt0 prefix: {corp.decode(prompts8[0])[:32]!r}")

    # ================================================== G3: the DONOR net
    corp21 = CharCorpus(DONOR_CORPUS)                    # seed 1337
    st21 = torch.load(DONOR_CKPT, map_location="cpu", weights_only=False)
    sd21 = st21["model"] if "model" in st21 else st21
    cfg21 = Cfg(vocab=corp21.vocab_size, n_layer=6, n_head=6, n_embd=192,
                block_size=DONOR_BLOCK)
    net21 = TinyGPT(cfg21)
    net21.load_state_dict(sd21, strict=True)
    net21.eval()
    val_ce21 = estimate_loss(net21, corp21, "val", n_batches=12)
    head_dim_ok = (cfg21.n_embd // cfg21.n_head == cfg.n_embd // cfg.n_head
                   == 32)
    g3 = dict(ckpt=str(DONOR_CKPT), step=int(st21.get("step", -1)),
              arch=dict(n_layer=6, n_head=6, n_embd=192,
                        block_size=DONOR_BLOCK, vocab=corp21.vocab_size),
              donor_vocab_ok=bool(corp21.vocab_size == VOCAB),
              head_dim_32_both=bool(head_dim_ok),
              val_ce_own_corpus=val_ce21, ref_e063b=E063B_VAL_CE_TASK,
              tol=0.15,
              val_ce_ok=bool(abs(val_ce21 - E063B_VAL_CE_TASK) <= 0.15))
    log(f"donor net e021_task loaded (step {g3['step']}, "
        f"{net21.num_params():,} params, vocab {corp21.vocab_size}) | val CE "
        f"on own corpus {val_ce21:.4f} vs e063b {E063B_VAL_CE_TASK:.4f} "
        f"(soft gate {'PASS' if g3['val_ce_ok'] else 'FAIL'}) | head_dim 32 "
        f"both sides: {head_dim_ok}")

    # donor runs: two windows, frozen before ANY recipient arm
    donors = {w: donor_run(net21, corp21, DONOR_WIN[w]["prompt"],
                           DONOR_WIN[w]["sample"]) for w in (1, 2)}
    for w in (1, 2):
        d = donors[w]
        vtail = d["V"][0][:, :, e080.PROMPT_TOK:, :]     # layer0 gen band
        log(f"donor window {w}: 8 rows 64->{DONOR_BLOCK} | online tail CE "
            f"{d['online_tail_ce']:.4f} | donor row0 tail: "
            f"{corp21.decode(d['idx'][0, -48:])[:48]!r}")
    # donor determinism (window 1 rerun, bit-identical)
    d1r = donor_run(net21, corp21, DONOR_WIN[1]["prompt"], DONOR_WIN[1]["sample"])
    det_don = bool(torch.equal(d1r["idx"], donors[1]["idx"])
                   and all(torch.equal(a, b) for a, b in zip(d1r["V"],
                                                             donors[1]["V"])))
    g3["donor_run_determinism"] = dict(
        rule="window-1 rerun (same seeds) tokens + V caches bit-identical",
        ok=det_don)
    g3["ok"] = bool(g3["donor_vocab_ok"] and head_dim_ok and det_don
                    and g3["val_ce_ok"])
    gates["G3_donor_net"] = g3
    log(f"G3 donor identity: vocab/head_dim/determinism/val-CE "
        f"{g3['donor_vocab_ok']}/{head_dim_ok}/{det_don}/{g3['val_ce_ok']} "
        f"-> {'PASS' if g3['ok'] else 'FAIL'}")

    # ================================================== THE SIX ARMS
    A = {}
    for arm in LEGACY:                                   # VERBATIM e080 rig
        gen = torch.Generator().manual_seed(e080.SEED_SAMPLE)
        gen_n = (torch.Generator().manual_seed(e080.SEED_NOISE)
                 if arm == "noise" else None)
        A[arm] = e080.generate_arm(net, prompts8, gen, arm, noise_gen=gen_n)
        log(f"arm {arm:11s} [e080 rig]: B={e080.B} 64->{e080.T_TOTAL} | "
            f"{len(A[arm]['replaced'])} replaced | cache ok "
            f"{A[arm]['cache']['ok']}")
    donor_rz = e099.draw_donor(e099.SEED_DONOR, e080.N_PROMPTS)   # 4343
    gen = torch.Generator().manual_seed(e080.SEED_SAMPLE)
    A["randomize"] = e099.generate_arm5(net, prompts8, gen, "randomize",
                                        donor_rz)
    rs = A["randomize"]["repl_stats"]
    log(f"arm randomize  [e099 rig]: donor map {donor_rz} | "
        f"{len(A['randomize']['replaced'])} replaced | mean|cos(old,donor)| "
        f"{rs['mean_abs_cos_old_vs_donor']:.3f} | norm ratio "
        f"{rs['mean_norm_ratio_donor_over_old']:.3f} | cache ok "
        f"{A['randomize']['cache']['ok']}")
    donor_cf = e099.draw_donor(SEED_DONOR_MAP, e080.N_PROMPTS)    # 4646
    gen = torch.Generator().manual_seed(e080.SEED_SAMPLE)
    A[PROBE] = generate_arm6(net, prompts8, gen, donors, donor_cf)
    rs = A[PROBE]["repl_stats"]
    log(f"arm crossfamily[e105 ext]: donor map {donor_cf} | "
        f"{len(A[PROBE]['replaced'])} replaced | mean|cos(old,donor)| "
        f"{rs['mean_abs_cos_old_vs_donor']:.3f} | norm ratio "
        f"{rs['mean_norm_ratio_donor_over_old']:.3f} | cache ok "
        f"{A[PROBE]['cache']['ok']}")

    # G6: crossfamily full-pipeline determinism (fresh donors + fresh arm)
    donors_rerun = {w: donor_run(net21, corp21, DONOR_WIN[w]["prompt"],
                                 DONOR_WIN[w]["sample"]) for w in (1, 2)}
    gen = torch.Generator().manual_seed(e080.SEED_SAMPLE)
    rerun = generate_arm6(net, prompts8, gen, donors_rerun,
                          e099.draw_donor(SEED_DONOR_MAP, e080.N_PROMPTS))
    g6 = bool(torch.equal(rerun["idx"], A[PROBE]["idx"]))
    gates["G6_crossfamily_determinism"] = dict(
        rule="crossfamily arm (donors regenerated + seed-7 stream) rerun "
             "token stream bit-identical", ok=g6)
    log(f"G6 crossfamily determinism: {g6}")

    # ---- G5 intervention identity
    pre = e080.PRUNE_START_G + e080.PROMPT_TOK + 1       # 165: cols 0..164
    ident = all(torch.equal(A[ARMS[0]]["idx"][:, :pre], A[a]["idx"][:, :pre])
                for a in ARMS)
    exp_repl = {a: (0 if a == "none" else 324) for a in ARMS}
    counts_ok = all(len(A[a]["replaced"]) == exp_repl[a] for a in ARMS)
    derange_ok = (all(d != b for b, d in enumerate(donor_rz))
                  and all(d != b for b, d in enumerate(donor_cf)))
    donor_pos_ok = all(64 <= donor_source(p)[1] <= 255
                       and donor_source(p)[0] in (1, 2)
                       for p in range(e080.GEN_FIRST, 388))
    n_w1 = sum(1 for p in range(64, 388) if donor_source(p)[0] == 1)
    n_w2 = sum(1 for p in range(64, 388) if donor_source(p)[0] == 2)
    ok5 = bool(ident and counts_ok and derange_ok and donor_pos_ok
               and all(A[a]["cache"]["ok"] for a in ARMS))
    gates["G5_intervention_identity"] = dict(
        pre_event_identity_through_position=pre - 1, identical=ident,
        replaced_counts={a: len(A[a]["replaced"]) for a in ARMS},
        replaced_counts_expected=exp_repl, counts_ok=counts_ok,
        donor_map_randomize=list(donor_rz),
        donor_map_crossfamily=list(donor_cf),
        donor_derangements_ok=derange_ok,
        donor_position_map=dict(
            rule="p<=255 <- window1 pos p; p in [256,387] <- window2 pos "
                 "p-192; all source positions within donor-native generated "
                 "band 64..255; zero reuse",
            n_window1=n_w1, n_window2=n_w2, all_sources_native=donor_pos_ok),
        cache_oks={a: A[a]["cache"]["ok"] for a in ARMS},
        crossfamily_stats=A[PROBE]["repl_stats"], ok=ok5)
    log(f"G5: pre-event identity through pos {pre - 1}: {ident} | counts "
        f"{ {a: len(A[a]['replaced']) for a in ARMS} } | derangements "
        f"{derange_ok} | donor sources native {donor_pos_ok} "
        f"(w1 {n_w1} / w2 {n_w2}) -> {'PASS' if ok5 else 'FAIL'}")

    # ---- G4 drift vs e099 (the five legacy arms, 8 threads both sides)
    g4 = dict(ref_file=str(E099_METRICS), ok=None, note="")
    if E099_METRICS.exists():
        with open(E099_METRICS) as f:
            m099 = json.load(f)
        devs = {}
        for arm in LEGACY + ["randomize"]:
            ref = m099["collapse_signature"][arm]
            devs[arm + "_online"] = abs(float(A[arm]["ce"][:, -TAIL:].mean())
                                        - ref["online_tail_ce"])
            al = e080.manual_all_logits(net, A[arm]["idx"])
            cj = float(e099.cj_from_alllogits(al, A[arm]["idx"], TAIL).mean())
            devs[arm + "_cj"] = abs(cj - ref["clean_judge_tail_ce"])
            gap_ps = (e099.cj_from_alllogits(al, A[arm]["idx"], TAIL)
                      - A[arm]["ce"][:, -TAIL:].mean(1))
            devs[arm + "_gap_ps"] = float(
                np.abs(gap_ps - np.asarray(ref["gap_per_seq"])).max())
        max_dev = max(devs.values())
        g4.update(per_quantity_max_dev={k: v for k, v in devs.items()},
                  ok=bool(max_dev < 1e-5))
        g4["note"] = (f"five legacy arms rerun vs runs/e099/metrics.json: "
                      f"max dev {max_dev:.2e} (online tail CE, clean-judge, "
                      f"per-seq gap; same 8-thread setting -> expect ~1e-7)")
    else:
        g4["note"] = "runs/e099/metrics.json not found; drift-check skipped"
    gates["G4_rerun_vs_e099"] = g4
    log(f"G4 rerun vs e099: {g4['note']} -> "
        f"{'PASS' if g4['ok'] else ('SKIPPED' if g4['ok'] is None else 'FAIL')}")

    # ---- collapse signature (clean-judge gap on the final-128 tail)
    sig = {}
    for arm in ARMS:
        online = A[arm]["ce"][:, -TAIL:].mean(1)
        with torch.no_grad():
            al = e080.manual_all_logits(net, A[arm]["idx"])
        cj128 = e099.cj_from_alllogits(al, A[arm]["idx"], TAIL)
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
    full_ids = torch.cat([corp.train, corp.val]).numpy()
    p_corpus = smooth(counts_of(full_ids))
    full21 = torch.cat([corp21.train, corp21.val]).numpy()
    p_corpus21 = smooth(counts_of(full21))
    val_np = corp.val.numpy()
    starts = np.linspace(0, len(val_np) - TAIL, N_WIN).astype(int)
    win_tokens = [val_np[s:s + TAIL] for s in starts]
    corpus_rep = {n: float(np.mean([repeat_rate(w, n)
                                    for w in win_tokens])) for n in NGRAMS}
    log(f"corpus baselines: shakespeare unigram H {entropy(p_corpus):.4f} | "
        f"e021_task unigram H {entropy(p_corpus21):.4f} | window repeat "
        f"rates { {n: round(v, 3) for n, v in corpus_rep.items()} }")

    tails, counts_arm, counts_seq, stats = {}, {}, {}, {}
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
            top1_mass=float(top5[0][1] / counts_arm[arm].sum()),
            per_seq_modal=[dict(token=int(t), char=corp.itos[t], freq=f)
                           for t, f in modal],
            modal_census={k: int(v) for k, v in modal_census.items()},
            kl_vs_corpus=kl(pooled, p_corpus),
            kl_vs_donor_corpus=kl(pooled, p_corpus21),
            decoded_tails=[corp.decode(torch.tensor(t)) for t in T],
        )
        log(f"  tail {arm:11s}: H {stats[arm]['pooled_entropy']:.4f} | "
            f"distinct {stats[arm]['n_distinct']} | top1 "
            f"{stats[arm]['top1_mass']:.3f} | top5 "
            f"{[(d['char'], round(d['freq'], 3)) for d in stats[arm]['top5']]} "
            f"| rep2 {rep[2]:.3f} | KL||corpus "
            f"{stats[arm]['kl_vs_corpus']:.4f}")

    # donor rider: donor pooled tail (final 128 of each donor row, 2 windows)
    donor_tails = np.concatenate(
        [donors[w]["idx"][:, -TAIL:].numpy() for w in (1, 2)], 0)  # (16,128)
    donor_counts = counts_of(donor_tails)
    p_donor = smooth(donor_counts)
    donor_rep = {n: float(np.mean([repeat_rate(t, n)
                                   for t in donor_tails])) for n in NGRAMS}
    donor_top5 = top_tokens(donor_counts, 5)
    skl_donor = {arm: skl(p_donor, smooth(counts_arm[arm])) for arm in ARMS}
    log(f"donor rider: pooled donor tail H {entropy(p_donor):.4f} | top5 "
        f"{[(corp21.itos[t], round(float(c) / donor_counts.sum(), 3)) for t, c in donor_top5]} "
        f"| KL||e21corpus {kl(p_donor, p_corpus21):.4f} | KL||shakes "
        f"{kl(p_donor, p_corpus):.4f} | SKL vs arms "
        f"{ {a: round(v, 3) for a, v in skl_donor.items()} }")

    # within-arm null floors + cross-arm SKL grid
    nulls = {arm: within_null(counts_seq[arm]) for arm in ARMS}
    for arm in ARMS:
        v = np.asarray(nulls[arm])
        log(f"  within-null {arm:11s}: mean {v.mean():.4f} "
            f"[{np.percentile(v, 2.5):.4f},{np.percentile(v, 97.5):.4f}] "
            f"(35 disjoint 4v4 splits)")
    nA = len(ARMS)
    grid = np.full((nA, nA), np.nan)
    P = {arm: smooth(counts_arm[arm]) for arm in ARMS}
    for i, a in enumerate(ARMS):
        grid[i, i] = float(np.mean(nulls[a]))
        for j, b in enumerate(ARMS):
            if i != j:
                grid[i, j] = skl(P[a], P[b])

    # ================================================== REGISTERED DECISION
    floor_c = (float(np.mean([np.mean(nulls[a]) for a in collapsed]))
               if len(collapsed) >= 2 else
               float(np.mean([np.mean(nulls[a]) for a in ARMS])))
    floor_h = float(np.mean([np.mean(nulls[a]) for a in HEALTHY_REFS]))
    dc = {a: float(np.mean([grid[ARMS.index(a), ARMS.index(c)]
                            for c in collapsed])) for a in ARMS}
    dh = {a: float(np.mean([grid[ARMS.index(a), ARMS.index(h)]
                            for h in HEALTHY_REFS])) for a in ARMS}
    joins_c = {a: bool(dc[a] <= CLUSTER_RATIO * floor_c and dh[a] > dc[a])
               for a in ARMS}
    joins_h = {a: bool(dh[a] <= CLUSTER_RATIO * floor_h and dc[a] > dh[a])
               for a in ARMS}
    log(f"clusters: floor_c {floor_c:.4f} floor_h {floor_h:.4f} | collapsed "
        f"{collapsed}")
    for a in ARMS:
        log(f"  {a:11s}: d_c {dc[a]:.3f} ({dc[a] / floor_c:.1f}x floor_c) | "
            f"d_h {dh[a]:.3f} ({dh[a] / floor_h:.1f}x floor_h) | joins "
            f"collapsed {joins_c[a]} / healthy {joins_h[a]}")

    controls = dict(
        randomize_healthy=bool(sig["randomize"]["gap"] < GAP_HEALTHY
                               and joins_h["randomize"]),
        promptcopy_collapsed=bool(sig["promptcopy"]["gap"] > GAP_BAR
                                  and joins_c["promptcopy"]),
        randomize_gap_ref=E099_GAP_RANDOMIZE,
        promptcopy_gap_ref=E099_GAP_PROMPTCOPY)
    controls_ok = bool(controls["randomize_healthy"]
                       and controls["promptcopy_collapsed"])

    gap_x = sig[PROBE]["gap"]
    fam_weights = bool(gap_x > GAP_BAR and joins_c[PROBE])
    fam_generic = bool(gap_x < GAP_HEALTHY and joins_h[PROBE])
    clauses = dict(
        gap_probe=dict(rule=f"crossfamily clean-judge gap > {GAP_BAR} nat",
                       gap=gap_x, fires=bool(gap_x > GAP_BAR)),
        joins_collapsed_cluster=dict(
            rule=f"d_c <= {CLUSTER_RATIO}x floor_c ({CLUSTER_RATIO * floor_c:.3f})"
                 f" AND d_h > d_c", d_c=dc[PROBE], d_h=dh[PROBE],
            fires=joins_c[PROBE]),
        gap_healthy=dict(rule=f"crossfamily clean-judge gap < {GAP_HEALTHY}",
                         gap=gap_x, fires=bool(gap_x < GAP_HEALTHY)),
        joins_healthy_cluster=dict(
            rule=f"d_h <= {CLUSTER_RATIO}x floor_h ({CLUSTER_RATIO * floor_h:.3f})"
                 f" AND d_c > d_h", d_h=dh[PROBE], d_c=dc[PROBE],
            fires=joins_h[PROBE]),
        family_trained_weights=dict(
            rule=f"gap > {GAP_BAR} AND joins collapsed cluster",
            fires=fam_weights),
        family_generic_stats=dict(
            rule=f"gap < {GAP_HEALTHY} AND joins healthy cluster",
            fires=fam_generic))
    if fam_weights:
        clause = "FAMILY = TRAINED WEIGHTS"
        verdict = (
            f"FAMILY = TRAINED WEIGHTS fires: the cross-family arm COLLAPSED "
            f"(clean-judge gap {gap_x:+.3f} > {GAP_BAR}) and its terminal "
            f"distribution joined the collapsed cluster (d_c {dc[PROBE]:.3f} "
            f"= {dc[PROBE] / floor_c:.1f}x floor_c, below the "
            f"{CLUSTER_RATIO}x bar; d_h {dh[PROBE]:.3f} above d_c) — a "
            f"differently-TRAINED net's generated entries cannot anchor the "
            f"run where the same trained net's sibling entries do. The "
            f"anchor reads the generator's IDENTITY: the family boundary is "
            f"the trained weights themselves (the sharpest form).")
    elif fam_generic:
        clause = "FAMILY = GENERIC STATS"
        verdict = (
            f"FAMILY = GENERIC STATS fires: the cross-family arm stayed "
            f"HEALTHY (clean-judge gap {gap_x:+.3f} < {GAP_HEALTHY}) and its "
            f"terminal distribution joined the none/randomize cluster (d_h "
            f"{dh[PROBE]:.3f} = {dh[PROBE] / floor_h:.1f}x floor_h, below "
            f"the {CLUSTER_RATIO}x bar; d_c {dc[PROBE]:.3f} above d_h) — the "
            f"copy-task net's generated entries anchor the run as well as "
            f"the same net's own. The anchor reads generic competent-"
            f"char-LM generated-text shape, NOT the generator's identity "
            f"(the weaker, broader claim).")
    else:
        clause = "MIXED"
        bits = [f"crossfamily gap {gap_x:+.3f} (gray zone "
                f"[{GAP_HEALTHY},{GAP_BAR}])" if GAP_HEALTHY <= gap_x <= GAP_BAR
                else f"crossfamily gap {gap_x:+.3f}"]
        bits.append(f"d_c {dc[PROBE]:.3f} ({dc[PROBE] / floor_c:.1f}x "
                    f"floor_c) vs d_h {dh[PROBE]:.3f} "
                    f"({dh[PROBE] / floor_h:.1f}x floor_h)")
        if gap_x > GAP_BAR and not joins_c[PROBE]:
            bits.append("gap collapses but the terminal distribution does "
                        "NOT join the collapsed cluster — off-manifold by "
                        "the judge, yet a distinct basin")
        if gap_x < GAP_HEALTHY and not joins_h[PROBE]:
            bits.append("gap is healthy but the terminal distribution sits "
                        "apart from the none/randomize cluster — healthy by "
                        "the judge, distinct by distribution")
        bits.append(f"SKL(crossfamily, donor pooled) "
                    f"{skl_donor[PROBE]:.3f} (does the tail drift toward the "
                    f"donor family's own statistics?)")
        verdict = ("MIXED (honest texture): " + "; ".join(bits) + ".")
    log(f"REGISTERED DECISION [{clause}]: {verdict}")
    log(f"  controls: randomize healthy "
        f"{controls['randomize_healthy']} (gap "
        f"{sig['randomize']['gap']:+.3f}) | promptcopy collapsed "
        f"{controls['promptcopy_collapsed']} (gap "
        f"{sig['promptcopy']['gap']:+.3f}) | controls_ok {controls_ok}")

    # ---------------------------------------------------------------- metrics
    metrics = dict(
        experiment="e105_cross_family",
        purpose="T055's registered CROSS-FAMILY anchor test: WHERE is the "
                "net-family boundary? e099's five-arm structure verbatim "
                "(none/vzero/noise/promptcopy/randomize; matched seed-7 "
                "streams) + crossfamily: the e021_task copy-task net's OWN "
                "free-run V entries (its own corpus, two native block-256 "
                "windows; p<=255 position-matched, p in [256,387] <- "
                "window2 p-192; recipient L/H 0..3 <- donor L/H 0..3, d=32) "
                "spliced into the anchor band the way randomize did. "
                "Readouts: clean-judge gap (final-128 tail) + terminal "
                "unigram SKL cluster map. FROZEN bars: gap > 1 AND joins "
                "collapsed cluster => FAMILY = TRAINED WEIGHTS; gap < 0.3 "
                "AND joins healthy cluster => FAMILY = GENERIC STATS; else "
                "MIXED.",
        started=started, wall_s=elapsed(), threads=THREADS, cpu_only=True,
        recipient_net=dict(ckpt=str(e080.CKPT),
                           arch=dict(n_layer=4, n_head=4, n_embd=128,
                                     block_size=e080.T_TOTAL, vocab=VOCAB),
                           params=n_params, val_ce=val_ce,
                           val_ce_e053c=e080.E053C_VAL_CE),
        donor_net=gates["G3_donor_net"],
        seeds=dict(corpus=1337, prompts=e080.SEED_PROMPT,
                   sampling=e080.SEED_SAMPLE, noise=e080.SEED_NOISE,
                   donor_map_randomize=e099.SEED_DONOR,
                   donor_map_crossfamily=SEED_DONOR_MAP,
                   donor_windows=DONOR_WIN,
                   donor_note="donor windows use DEDICATED prompt/sample "
                              "generators (never touching arm streams); "
                              "donor content frozen before all arms"),
        protocol=dict(B=e080.B, prompt_tokens=e080.PROMPT_TOK,
                      t_total=e080.T_TOTAL, temp=e080.TEMP, topk=e080.TOPK,
                      tail_tokens=TAIL, smoothing_alpha=ALPHA,
                      within_null="35 disjoint 4-vs-4 sequence splits per "
                                  "arm, symmetric KL",
                      arms=dict(
                          none="normal free run (e080.generate_arm verbatim)",
                          randomize="V<-sibling battery row's entry at the "
                                    "same position (e099.generate_arm5 "
                                    "verbatim, seed-4343 derangement, "
                                    "pre-event snapshot) — same trained net",
                          crossfamily="V<-e021_task copy-task net's own "
                                      "free-run entries on its own corpus "
                                      "(e105 extension; see purpose)",
                          vzero="V->0 at self-band events (e080 verbatim)",
                          noise="V<-norm-matched gaussian (e080 verbatim, "
                                "seed 4242)",
                          promptcopy="V<-same row's pristine prompt entry "
                                     "(e080 verbatim) — corpus-text control"),
                      schedule=dict(events_g=e080.EVENTS, K=e080.PRUNE_K,
                                    start_g=e080.PRUNE_START_G,
                                    age_cut=e080.AGE_CUT,
                                    note="identical to e080/e099: replaced "
                                         "once at first band admission, K "
                                         "untouched; all six arms "
                                         "token-identical through position "
                                         "164; 324 replaced positions"),
                      collapse_gate=f"clean-judge tail CE - online tail CE "
                                    f"> {GAP_BAR} nats (e099 convention)",
                      cluster_rule=dict(
                          healthy_refs=HEALTHY_REFS,
                          collapsed_refs="trigger arms with gap > 1.0",
                          floor_c=floor_c, floor_h=floor_h,
                          joins_collapsed=f"d_c <= {CLUSTER_RATIO}x floor_c "
                                          f"AND d_h > d_c",
                          joins_healthy=f"d_h <= {CLUSTER_RATIO}x floor_h "
                                        f"AND d_c > d_h",
                          calibration="verified on e099's published grid: "
                                      "classifies all five e099 arms "
                                      "correctly")),
        gates=gates,
        char_map=[corp.itos[t] for t in range(VOCAB)],
        collapse_signature={a: sig[a] for a in ARMS},
        collapsed_arms=collapsed,
        corpus_baseline=dict(
            unigram_entropy=entropy(p_corpus),
            window_repeat_ngram_rates=corpus_rep,
            donor_corpus_unigram_entropy=entropy(p_corpus21),
            kl_donor_pool_vs_shakespeare=kl(p_donor, p_corpus),
            kl_donor_pool_vs_e21corpus=kl(p_donor, p_corpus21)),
        donor_rider=dict(
            note="NON-REGISTERED rider: the donor net's own generated-text "
                 "statistics (pooled final-128 tails of both donor windows, "
                 "16x128 tokens) vs the recipient arms",
            pooled_entropy=entropy(p_donor),
            repeat_ngram_rates=donor_rep,
            top5=[dict(token=int(t), char=corp21.itos[t], count=int(c),
                       freq=float(c) / donor_counts.sum())
                  for t, c in donor_top5],
            skl_vs_arms=skl_donor,
            donor_tail_samples=[corp21.decode(donors[w]["idx"][s, -64:])
                                for w in (1, 2) for s in (0, 1)]),
        arms={arm: dict(
            tail_tokens=tails[arm].tolist(),
            **{k: v for k, v in stats[arm].items()
               if k != "decoded_tails"},
            decoded_tails=stats[arm]["decoded_tails"],
            within_null=dict(values=nulls[arm],
                             mean=float(np.mean(nulls[arm])),
                             ci=[float(np.percentile(nulls[arm], 2.5)),
                                 float(np.percentile(nulls[arm], 97.5))]),
            replace_log=A[arm]["replace_log"],
            repl_stats=A[arm]["repl_stats"],
            d_c=dc[arm], d_h=dh[arm],
            joins_collapsed_cluster=joins_c[arm],
            joins_healthy_cluster=joins_h[arm],
        ) for arm in ARMS},
        kl_grid=dict(arm_order=ARMS, symmetric=grid.tolist(),
                     diag="within-arm null mean (35 disjoint 4v4 splits)",
                     floors=dict(collapsed=floor_c, healthy=floor_h)),
        cluster_map=dict(
            d_c=dc, d_h=dh, joins_collapsed=joins_c, joins_healthy=joins_h),
        registered_decision=dict(
            frozen_rules=dict(
                family_trained_weights=f"crossfamily gap > {GAP_BAR} AND "
                                       f"d_c <= {CLUSTER_RATIO}x floor_c AND "
                                       f"d_h > d_c",
                family_generic_stats=f"crossfamily gap < {GAP_HEALTHY} AND "
                                     f"d_h <= {CLUSTER_RATIO}x floor_h AND "
                                     f"d_c > d_h",
                else_="MIXED (report the honest texture)"),
            controls=controls, controls_ok=controls_ok,
            clauses=clauses, clause=clause, verdict=verdict),
    )
    save_json(out_dir / "metrics.json", metrics)
    log("metrics.json written")

    plot(out_dir / "cross_family.png", metrics)
    log(f"plot written; total wall {elapsed():.0f}s")


# ---------------------------------------------------------------------- plot


def plot(path: Path, M: dict):
    arms = M["kl_grid"]["arm_order"]
    grid = np.asarray(M["kl_grid"]["symmetric"])
    sig = M["collapse_signature"]
    am = M["arms"]
    cm = M["cluster_map"]
    dec = M["registered_decision"]
    donor = M["donor_rider"]
    collapsed = set(M["collapsed_arms"])
    healthy_refs = set(["none", "randomize"])
    cols = {a: ("tab:gray" if a == "none" else
                "tab:green" if a == "randomize" else
                "tab:purple" if a == "crossfamily" else
                "tab:red" if a == "vzero" else
                "tab:blue" if a == "noise" else "tab:orange")
            for a in arms}
    labs = {a: f"{a}\n(gap {sig[a]['gap']:+.2f})" for a in arms}
    fig, axes = plt.subplots(2, 3, figsize=(24, 11))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    # ---- panel 1: the clean-judge gaps + registered bars
    x = np.arange(len(arms))
    gaps = [sig[a]["gap"] for a in arms]
    yerr = [[max(0.0, g - sig[a]["gap_ci"][0])
             for g, a in zip(gaps, arms)],
            [max(0.0, sig[a]["gap_ci"][1] - g)
             for g, a in zip(gaps, arms)]]
    ax1.bar(x, gaps, 0.62, color=[cols[a] for a in arms], alpha=0.85,
            edgecolor="k", lw=0.5)
    ax1.errorbar(x, gaps, yerr=yerr, fmt="none", ecolor="k", lw=1.0,
                 capsize=3)
    ax1.axhline(GAP_HEALTHY, color="tab:green", ls="--", lw=1.4)
    ax1.text(len(arms) - 0.4, GAP_HEALTHY + 0.06,
             f"healthy bar gap < {GAP_HEALTHY}", fontsize=9,
             color="tab:green", ha="right")
    ax1.axhline(GAP_BAR, color="tab:red", ls="--", lw=1.4)
    ax1.text(len(arms) - 0.4, GAP_BAR + 0.06,
             f"collapse bar gap > {GAP_BAR}", fontsize=9, color="tab:red",
             ha="right")
    for xi, a in zip(x, arms):
        ax1.text(xi, max(gaps[xi], 0) + 0.28,
                 f"{gaps[xi]:+.3f}", ha="center", fontsize=9)
    ax1.set_xticks(x, [labs[a] for a in arms], fontsize=8.5)
    ax1.set_ylabel("clean-judge tail CE - online tail CE (nats, final 128)")
    ax1.set_ylim(min(0, min(gaps)) - 0.3, max(gaps) * 1.18 + 0.3)
    ax1.set_title("E105-1 — the arms' clean-judge gaps (matched seed-7 "
                  "streams); gray zone shaded", fontsize=10)
    ax1.axhspan(GAP_HEALTHY, GAP_BAR, color="gray", alpha=0.10)

    # ---- panel 2: terminal-KL cluster map (SKL heatmap, diag = null)
    finite = grid[~np.isnan(grid)]
    gpos = grid[grid > 0]
    vmin = gpos.min() if gpos.size else 1e-3
    im = ax2.imshow(grid, cmap="viridis",
                    norm=LogNorm(vmin=max(1e-4, vmin),
                                 vmax=max(0.01, finite.max())))
    for i in range(len(arms)):
        for j in range(len(arms)):
            v = grid[i, j]
            ax2.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.5,
                     color="white" if v > finite.max() / 3 else "black")
    # cluster boxes: healthy (0,1), probe (2), collapsed (3,4,5)
    ax2.add_patch(Rectangle((-0.5, -0.5), 2, 2, fill=False,
                            ec="tab:green", lw=2.5))
    ax2.add_patch(Rectangle((2.5, 2.5), 3, 3, fill=False,
                            ec="tab:red", lw=2.5))
    ax2.add_patch(Rectangle((1.5, 1.5), 1, 1, fill=False,
                            ec="tab:purple", lw=2.5, ls="--"))
    ax2.set_xticks(range(len(arms)), arms, fontsize=8, rotation=30)
    ax2.set_yticks(range(len(arms)), arms, fontsize=8)
    fig.colorbar(im, ax=ax2, label="symmetric KL (nats, log scale)")
    ax2.set_title("E105-2 — TERMINAL-KL cluster map: cross-arm tail SKL\n"
                  "(diag = within-arm 4v4 null; green box healthy refs, red "
                  "box collapsed refs, dashed = the probe)", fontsize=10)

    # ---- panel 3: MDS embedding of the SKL geometry + donor point
    D = grid.copy()
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    pts = pcoa_2d(D)
    for i, a in enumerate(arms):
        ax3.scatter(*pts[i], s=210, color=cols[a], edgecolor="k", zorder=4,
                    marker=("D" if a == "crossfamily" else "o"))
        ax3.annotate(a, pts[i], textcoords="offset points",
                     xytext=(9, -4), fontsize=9)
    # donor point: SKL to arms via rider (place by triangulation onto the
    # arms' coordinates using the same PCoA machinery on the augmented matrix)
    aug = np.zeros((len(arms) + 1, len(arms) + 1))
    aug[:len(arms), :len(arms)] = D
    for i, a in enumerate(arms):
        v = donor["skl_vs_arms"][a]
        aug[i, -1] = aug[-1, i] = v
    pts2 = pcoa_2d(aug)
    ax3.scatter(*pts2[-1], s=210, color="black", marker="*", zorder=4,
                edgecolor="k")
    ax3.annotate("donor text\n(e021_task)", pts2[-1], textcoords="offset "
                 "points", xytext=(8, 6), fontsize=9)
    ax3.set_title("E105-3 — terminal-distribution geometry (classical MDS "
                  "of the SKL grid)", fontsize=10)
    ax3.set_xticks([])
    ax3.set_yticks([])

    # ---- panel 4: d_c vs d_h per arm (cluster-membership view)
    x4 = np.arange(len(arms))
    dcv = [cm["d_c"][a] for a in arms]
    dhv = [cm["d_h"][a] for a in arms]
    w = 0.38
    ax4.bar(x4 - w / 2, dcv, w, color=[cols[a] for a in arms], alpha=0.9,
            edgecolor="k", lw=0.5, label="d_c (mean SKL to collapsed refs)")
    ax4.bar(x4 + w / 2, dhv, w, color=[cols[a] for a in arms], alpha=0.35,
            edgecolor="k", lw=0.5, hatch="//",
            label="d_h (mean SKL to healthy refs)")
    fl = M["kl_grid"]["floors"]
    ax4.axhline(CLUSTER_RATIO * fl["collapsed"], color="tab:red", ls="--",
                lw=1.2)
    ax4.text(len(arms) - 0.4, CLUSTER_RATIO * fl["collapsed"] + 0.02,
             f"2x floor_c ({CLUSTER_RATIO * fl['collapsed']:.3f})",
             fontsize=8, color="tab:red", ha="right")
    ax4.axhline(CLUSTER_RATIO * fl["healthy"], color="tab:green", ls="--",
                lw=1.2)
    ax4.text(len(arms) - 0.4, CLUSTER_RATIO * fl["healthy"] + 0.02,
             f"2x floor_h ({CLUSTER_RATIO * fl['healthy']:.3f})",
             fontsize=8, color="tab:green", ha="right")
    for xi, a in zip(x4, arms):
        ax4.text(xi, max(dcv[xi], dhv[xi]) + 0.03,
                 f"c:{'IN' if cm['joins_collapsed'][a] else '--'} "
                 f"h:{'IN' if cm['joins_healthy'][a] else '--'}",
                 ha="center", fontsize=7.5)
    ax4.set_xticks(x4, arms, fontsize=8, rotation=30)
    ax4.set_ylabel("mean terminal SKL (nats)")
    ax4.legend(fontsize=8, loc="upper left")
    ax4.set_ylim(0, max(dcv + dhv) * 1.3)
    ax4.set_title("E105-4 — cluster membership distances (probe joins "
                  "whichever bar it clears with d(other) > d(cluster))",
                  fontsize=10)

    # ---- panel 5: census + control table
    ax5.axis("off")
    lines = ["arm | cj_gap | cluster | d_c/floor_c | d_h/floor_h | top-5 "
             "tokens (char:freq) | KL||corpus"]
    for a in arms:
        s = am[a]
        cl = ("collapsed" if cm["joins_collapsed"][a] else
              "healthy" if cm["joins_healthy"][a] else "apart")
        lines.append(
            f"{a:11s} {sig[a]['gap']:+6.2f} {cl:9s} "
            f"{cm['d_c'][a] / fl['collapsed']:9.2f}x "
            f"{cm['d_h'][a] / fl['healthy']:9.2f}x "
            + " ".join(f"{d['char']!r}:{d['freq']:.2f}" for d in s["top5"])
            + f"  {s['kl_vs_corpus']:.3f}")
    lines.append("")
    lines.append(f"controls: randomize healthy "
                 f"[{dec['controls']['randomize_healthy']}] gap "
                 f"{sig['randomize']['gap']:+.3f} (e099 ref "
                 f"{E099_GAP_RANDOMIZE:+.3f})")
    lines.append(f"          promptcopy collapsed "
                 f"[{dec['controls']['promptcopy_collapsed']}] gap "
                 f"{sig['promptcopy']['gap']:+.3f} (e099 ref "
                 f"{E099_GAP_PROMPTCOPY:+.3f})")
    lines.append(f"donor rider: SKL(donor pooled tail, crossfamily tail) = "
                 f"{donor['skl_vs_arms']['crossfamily']:.3f} | donor top5 "
                 + " ".join(f"{d['char']!r}:{d['freq']:.2f}"
                            for d in donor["top5"]))
    ax5.text(0.02, 0.97, "E105-5 — terminal census + controls", fontsize=12,
             weight="bold", va="top")
    for i, t in enumerate(lines):
        ax5.text(0.02, 0.93 - i * 0.05, t, fontsize=8.6, va="top",
                 family="monospace")

    # ---- panel 6: the registered decision
    ax6.axis("off")
    l6 = ["REGISTERED BARS (frozen):",
          f"  FAMILY = TRAINED WEIGHTS: gap(probe) > {GAP_BAR} AND joins "
          f"collapsed cluster",
          f"  FAMILY = GENERIC STATS:  gap(probe) < {GAP_HEALTHY} AND joins "
          f"healthy cluster",
          f"  else MIXED (texture; gray zone [{GAP_HEALTHY},{GAP_BAR}])", "",
          "clauses:"]
    for k, v in dec["clauses"].items():
        l6.append(f"  {k}: {'FIRES' if v['fires'] else 'no'}")
    l6.append(f"  controls_ok: {dec['controls_ok']}")
    l6 += ["", f"VERDICT [{dec['clause']}]:"] + \
        [f"  {wd}" for wd in textwrap.wrap(dec["verdict"], 100)]
    ax6.text(0.02, 0.97, "E105-6 — where is the family boundary?",
             fontsize=12, weight="bold", va="top")
    for i, t in enumerate(l6):
        ax6.text(0.02, 0.94 - i * 0.036, t, fontsize=8.8, va="top",
                 family="monospace")

    fig.suptitle(f"E105 — cross-family anchor test (T055) | {dec['clause']} "
                 f"| probe gap {sig['crossfamily']['gap']:+.3f} | d_c "
                 f"{cm['d_c']['crossfamily']:.3f} "
                 f"({cm['d_c']['crossfamily'] / fl['collapsed']:.1f}x "
                 f"floor_c) | d_h {cm['d_h']['crossfamily']:.3f} "
                 f"({cm['d_h']['crossfamily'] / fl['healthy']:.1f}x floor_h)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
