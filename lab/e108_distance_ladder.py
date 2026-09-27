"""E108 — the DISTANCE-LADDER anchor test (W002's bilinear-unification design).

[REGISTERED DESIGN — frozen in this docstring BEFORE any compute]

THE QUESTION (W002 RECON OUTCOME): the anchor family test (T055/T057) holds
the NET fixed and varies the TEXT; the crossmatch (T046/e062) holds the text
fixed and varies the NET (pre-graft stream-cosine, partial r -0.976 with
graft damage). Are these two projections of ONE bilinear compatibility
form? The discriminating observation: anchor health should track CROSSMATCH
DISTANCE monotonically. A two-point ladder (sibling 1.0 healthy / foreign
collapsed) cannot discriminate one quantity from two — the design needs a
MIDDLE rung at known, measured distance.

ARMS (content sources for the anchor band, e105 splice procedure VERBATIM;
recipient e053c_ctx512, g=100..420 K=32 self-band events, K untouched,
replaced once at first admission, matched seed-7 sampling stream):
  1. none          e080.generate_arm verbatim (control).
  2. randomize     e099.generate_arm5 VERBATIM (seed-4343 derangement,
                   pre-event sibling snapshots) — SIBLING-RUN control,
                   replicates e099's healthy +0.027.
  3. middle        NEW: generated text from e040_ref — the e040 lineage's
                   REFERENCE net (fresh init set_seed(4304), 4000 steps,
                   batch 32, block 256, data/input.txt; 4L/4H/128d, vocab
                   65): same arch, same corpus, DIFFERENTLY-SEEDED and
                   differently-trained — the closest available analog of
                   W002's B43 (same family, different seed). Donor
                   windows/prompt+sample seeds 5811/5911, 5812/5912
                   (dedicated, never touching arm streams); row derangement
                   seed 4747; position map and layer/head maps exactly
                   e105's crossfamily procedure
                   (p <= 255 <- window1[:, :, p, :] position-matched;
                   p in [256,387] <- window2[:, :, p-192, :] donor-native
                   64..195, zero reuse; recipient L/H 0..3 <- donor L/H
                   0..3, d=32 — here an IDENTITY map, donor has exactly
                   4L/4H).
                   MIDDLE-DONOR CHOICE RULE (pre-registered, frozen): among
                   the same-arch same-corpus candidates {e005s_small,
                   e040_w, e040_ref}, the donor whose 2-batch stream-cos vs
                   the recipient is CLOSEST TO 0.5 (the W002 middle); all
                   candidates' cosines are measured in-script and recorded
                   (G3c). Pre-compute recon: all candidates sit at the
                   stream-cos FLOOR (~0.02-0.04) — the W002 card's ~0.53
                   was e029's dW-alignment ladder, NOT stream-cos; e062's
                   measured B43 stream-cos is 0.03-0.04 (scale-B). The
                   intended intermediate rung does not exist in the pool;
                   the rule's winner is the middle donor and the ladder's
                   x-axis is MEASURED, not assumed. (Dispatch recon note:
                   the task text named e005s_small as its fallback
                   candidate BEFORE measuring — e005s_small shares training
                   seed 42 with the recipient's recipe (not different-seed)
                   and its measured cos 0.0233 loses the frozen rule to
                   e040_ref's 0.0326; e040_ref is also the truer B43
                   analog. Recorded in G3c.)
  4. crossfamily   e105's arm VERBATIM (e021_task copy-net donor, windows
                   5601/5701 + 5602/5702, derangement 4646) — FOREIGN
                   control, replicates e105's collapsed +5.09.

THE MEASURED X-AXIS (e062 method, 2 probe batches, common block 256, RNG
stream seed 1337): stream-cosine of each donor net vs the recipient at
depths d0..d4 (token-normalized residual streams); headline cos_all =
mean over depths, cos_gi = mean(d2,d3) as in e062. Sibling donor = the
recipient itself -> 1.0 by construction (verified). e021_task (192d) cannot
yield a stream-cos across stream dims -> recorded undefined; the secondary
measured axes cover all three anchors: (a) splice-time V-geometry
mean|cos(old, donor)| (repl_stats, every donor arm), (b) vocab-space mean
JS of predictive distributions on the shared shakespeare probe (every
donor net, incl. e021_task).

REGISTERED BARS (frozen; e099/e105 gap convention, final-128 tail):
  MONOTONE      fires iff gap(middle) < 0.3 AND gap(crossfamily) > 1.0
                (one quantity organizes both axes — the unification has
                legs; read together with the measured-axis texture).
  SHARP FAMILY  fires iff gap(middle) > 1.0 (middle collapsed like the
                foreign net — the text-axis family is sharper than the
                net-axis instrument; metaphor, not mechanism).
  else          PARTIAL/AMBIGUOUS -> report the texture honestly
                (0.3 <= gap <= 1.0 gray zone; cluster disagreement; and
                the floor-position caveat: if MONOTONE fires while the
                middle sits at floor stream-cos, the ladder distinguishes
                "one quantity with a step near 1.0" from "two quantities"
                only through the V-cos/JS orderings — report both).
  VALIDATION CONTROLS (required for the run to count): randomize healthy
  (gap < 0.3; e099 ref +0.0275) AND crossfamily collapsed (gap > 1.0;
  e105 ref +5.094).

GATES: G1 e053c val CE (tol 0.02); G2 params 873,472; G3a middle-donor
identity (e040_ref 4L/4H/128d/blk256/vocab65, params 840,704, val CE vs
e062 cached base_ce 1.5441 soft tol 0.05, donor-run determinism); G3b
copy-donor identity (e021_task arch 6L/6H/192d/blk256, step 1980, val CE
vs e063b 1.4318 tol 0.15, donor determinism); G3c middle-choice rule
(argmin |stream-cos - 0.5| over measured candidates == e040_ref);
G4 drift vs runs/e099/metrics.json (none, randomize) and
runs/e105/metrics.json (crossfamily): online/clean-judge/per-seq gap on the
final-128 tail, max dev < 1e-5 (same 8-thread setting); G5 intervention
identity (all four arms token-identical through position 164; replaced
counts 0/324/324/324; derangements ok; donor sources native; caches ok);
G6 middle full-pipeline determinism (donors regenerated + rerun
bit-identical); G7 ladder-axis sanity (recipient self stream-cos == 1.0
within 1e-6; probe rerun bit-identical; e021_task stream-cos recorded
undefined with reason).

Run:     python lab/e108_distance_ladder.py
Outputs: runs/e108/metrics.json + runs/e108/distance_ladder.png
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
import e099_attractor_identity as e099  # the VERBATIM randomize arm
import e105_cross_family as e105        # the VERBATIM crossfamily arm

THREADS = 8                                # task spec / T050: 12 thrashes box
torch.set_num_threads(THREADS)             # (e080 sets 12; e099/e105 reset 8)

# ------------------------------------------------------------------ constants
ARMS = ["none", "randomize", "middle", "crossfamily"]
PROBE = "middle"
HEALTHY_REFS = ["none", "randomize"]
TRIGGERS = ["randomize", "middle", "crossfamily"]

TAIL = 128                                 # e099/e105 tail window (final 128)
VOCAB = 65
ALPHA = 0.5                                # e099 smoothing
NGRAMS = (2, 3, 4)
N_WIN = 8                                  # corpus baseline windows

# ---- the MIDDLE donor: e040_ref (frozen choice rule's winner; see G3c)
MID_NAME = "e040_ref"
MID_CKPT = REPO / "runs" / "checkpoints" / "e040_ref.pt"
MID_BLOCK = 256                            # 4L/4H/128d, block 256 (e040 REF)
MID_ARCH = dict(n_layer=4, n_head=4, n_embd=128)
MID_PARAMS = 840_704
MID_WIN = {                                # dedicated donor streams (e108)
    1: dict(prompt=5811, sample=5911),
    2: dict(prompt=5812, sample=5912),
}
SEED_DONOR_MID = 4747                      # middle row derangement
ALT_CANDIDATES = ["e005s_small", "e040_w"]  # choice-rule alternates (same arch)

# ---- the FOREIGN donor: e105's copy-net, VERBATIM
DONOR_CKPT = e105.DONOR_CKPT               # runs/checkpoints/e021_task.train.pt
DONOR_CORPUS = e105.DONOR_CORPUS           # data/e021_task.txt
DONOR_BLOCK = e105.DONOR_BLOCK             # 256
E063B_VAL_CE_TASK = e105.E063B_VAL_CE_TASK

# ---- ladder axis references (published numbers, for context panels)
E099_METRICS = REPO / "runs" / "e099" / "metrics.json"
E105_METRICS = REPO / "runs" / "e105" / "metrics.json"
E099_GAP_RANDOMIZE = 0.027                 # e099's healthy sibling rider
E105_GAP_CROSSFAMILY = 5.094               # e105's collapsed foreign arm
E062_BASE_CE = {"e040_ref": 1.5441,        # e062 cached base_ce (15 batches)
                "e005s_small": 1.5504,
                "e040_w": 1.5366}
E062_B43_COS = dict(gi=0.0402, all=0.0315)     # e062 scale-B e001 <- e028_b43
E062_BDO_COS = dict(gi=0.8155, all=0.7426)     # e062 scale-B e001 <- e041_bdo

# registered bars (frozen; see docstring)
GAP_BAR = 1.0                              # gap > 1 nat = collapsed (e099)
GAP_HEALTHY = 0.3                          # gap < 0.3 = healthy bar
CLUSTER_RATIO = 2.0                        # d(cluster) <= 2x floor (texture)

T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# ------------------------------------------------------------ donor machinery

@torch.no_grad()
def donor_run_mid(net_m: TinyGPT, corp_m: CharCorpus, prompt_seed: int,
                  sample_seed: int):
    """e105.donor_run VERBATIM SKELETON for the middle donor: the middle
    net's OWN free run on ITS OWN corpus (data/input.txt): 8 rows, 64-token
    val prompts, 192 native generation steps (64->256), e080 sampling math.
    Returns idx (8,256), per-step CE, per-layer V caches (each
    (8, 4, 256, 32)) — the middle anchor content."""
    Bb = e080.N_PROMPTS
    gen_p = torch.Generator().manual_seed(prompt_seed)
    ix = torch.randint(len(corp_m.val) - e080.PROMPT_TOK - 1, (Bb,),
                       generator=gen_p)
    prompts = [corp_m.val[i:i + e080.PROMPT_TOK] for i in ix]
    idx = torch.stack(prompts)
    gen = torch.Generator().manual_seed(sample_seed)
    logits, kv = e080.prefill_batch(net_m, idx)
    gen_steps = DONOR_BLOCK - e080.PROMPT_TOK              # 192
    ce_s = np.zeros((Bb, gen_steps), float)
    for g in range(gen_steps):
        t = e080.PROMPT_TOK + g
        toks = torch.zeros(Bb, dtype=torch.long)
        for j in range(Bb):
            tok, ce = e080.sample_and_ce(logits[j], gen)
            toks[j] = tok
            ce_s[j, g] = ce
        idx = torch.cat([idx, toks[:, None]], 1)
        if t < DONOR_BLOCK:
            logits = e080.decode_step_batch(net_m, toks, t, kv)
    V = [v.clone() for (_k, v) in kv]
    assert V[0].shape == (e080.N_PROMPTS, net_m.cfg.n_head, DONOR_BLOCK,
                          net_m.cfg.n_embd // net_m.cfg.n_head), \
        f"middle donor V cache {tuple(V[0].shape)}"
    return dict(idx=idx, ce=ce_s, V=V, prompts=prompts,
                online_tail_ce=float(ce_s[:, -TAIL:].mean()))


# the splice arm: e105.generate_arm6 is donor-shape generic (slices donor
# heads 0..3; e040_ref HAS exactly 4) — reuse it VERBATIM.
generate_arm_donor = e105.generate_arm6
donor_source = e105.donor_source               # position map, identical rule


# ------------------------------------------- ladder x-axis (e062 2-batch probe)

@torch.no_grad()
def probe_windows(corpus: CharCorpus, block: int = 256, n_batches: int = 2,
                  rows: int = 16):
    """e062's probe batch draw, one fixed common block so shapes match across
    nets (recipient blk512 >= 256; donors blk256): RNG stream seed =
    corpus.seed, exactly e062.probe_streams' index rule."""
    gen = torch.Generator().manual_seed(corpus.seed)
    ix = torch.randint(len(corpus.val) - block - 1, (rows * n_batches,),
                       generator=gen)
    return [torch.stack([corpus.val[i:i + block]
                         for i in ix[b * rows:(b + 1) * rows]])
            for b in range(n_batches)]


@torch.no_grad()
def probe_streams_fixed(model: TinyGPT, xs, block: int = 256):
    """e062.probe_streams VERBATIM math (token-normalized residual streams,
    depths d0..dL) on the fixed common windows."""
    acc = None
    for x in xs:
        pos = torch.arange(block)
        s = model.wte(x) + model.wpe(pos)
        outs = [s]
        for blk in model.h:
            s = blk(s)
            outs.append(s)
        nrm = [t / t.norm(dim=-1, keepdim=True).clamp_min(1e-8) for t in outs]
        flat = [t.reshape(-1, t.shape[-1]) for t in nrm]
        acc = flat if acc is None else [torch.cat([a, f])
                                        for a, f in zip(acc, flat)]
    return acc


@torch.no_grad()
def probe_logits_fixed(model: TinyGPT, xs):
    """Forward logits on the SAME fixed windows (the vocab-space axis)."""
    outs = []
    for x in xs:
        T = x.shape[1]
        pos = torch.arange(T)
        h = model.wte(x) + model.wpe(pos)
        for blk in model.h:
            h = blk(h)
        outs.append(model.lm_head(model.ln_f(h)))
    return torch.cat(outs, 0)                       # (N, block, vocab)


def stream_cos_pair(sa: list, sb: list):
    """e062 pair rule: per-depth mean token-cosine; cos_gi = mean(d2,d3);
    cos_all = mean over depths."""
    nd = min(len(sa), len(sb))
    per = [float((sa[d] * sb[d]).sum(-1).mean()) for d in range(nd)]
    return dict(per_depth=per, cos_gi=float(np.mean(per[2:4])),
                cos_all=float(np.mean(per)), n_depths=nd)


def js_pair(la: torch.Tensor, lb: torch.Tensor):
    """Mean over probe tokens of JS(p_a, p_b) (nats, base-e) — the vocab-space
    secondary axis, computable for EVERY donor (incl. 192d nets)."""
    pa = torch.softmax(la.float(), -1)
    pb = torch.softmax(lb.float(), -1)
    m = 0.5 * (pa + pb)
    kl_am = (pa * (pa.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())
             ).sum(-1)
    kl_bm = (pb * (pb.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())
             ).sum(-1)
    return float((0.5 * kl_am + 0.5 * kl_bm).mean())


def nll_on(lg: torch.Tensor, xs):
    """Mean NLL of a net's predictions on the probe windows (competence
    context for the JS axis). logits (N, block, V) aligned with cat(xs)."""
    ys = torch.cat(list(xs), 0)
    lp = torch.log_softmax(lg.float(), -1)
    # predict token t+1 from position t: score positions 0..T-2 vs targets
    # 1..T-1 (next-token convention, same as the lab's CE instruments)
    nll = -lp[torch.arange(ys.shape[0])[:, None],
              torch.arange(ys.shape[1] - 1)[None, :], ys[:, 1:]]
    return float(nll.mean())


# ------------------------------------------------- distribution math (e099)
counts_of = e099.counts_of
smooth = e099.smooth
kl = e099.kl
skl = e099.skl
entropy = e099.entropy
top_tokens = e099.top_tokens
repeat_rate = e099.repeat_rate
within_null = e099.within_null


# ---------------------------------------------------------------------- main


def main():
    assert torch.cuda.device_count() == 0, "CPU-only violated: CUDA visible"
    assert common.DEVICE == "cpu", f"common.DEVICE={common.DEVICE} != cpu"
    out_dir = run_dir("e108")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gates = dict(cpu_only=True, threads=THREADS, smoke=False,
                 threads_note="e080 module import sets 12; overridden to 8 "
                              "(task spec / T050)")

    # ---- recipient battery: e053c net EXACTLY as e080/e099/e105 did
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

    # ================================================== G7 + the measured X-AXIS
    xs_probe = probe_windows(corp, block=256, n_batches=2)
    rec_streams = probe_streams_fixed(net, xs_probe, 256)
    rec_logits = probe_logits_fixed(net, xs_probe)
    # probe determinism: rerun bit-identical
    det_probe = all(torch.equal(a, b) for a, b in
                    zip(rec_streams,
                        probe_streams_fixed(net, xs_probe, 256)))
    self_cos = stream_cos_pair(rec_streams, rec_streams)

    def load_net(name, ckpt, arch, block):
        stx = torch.load(ckpt, map_location="cpu", weights_only=False)
        sdx = stx["model"] if isinstance(stx, dict) and "model" in stx else stx
        cfgx = Cfg(vocab=corp.vocab_size, block_size=block, **arch)
        netx = TinyGPT(cfgx)
        netx.load_state_dict(sdx, strict=True)
        netx.eval()
        return netx, stx.get("step", None)

    # middle candidates: measure stream-cos of each vs the recipient
    cand = {}
    for name in [MID_NAME] + ALT_CANDIDATES:
        ck = REPO / "runs" / "checkpoints" / f"{name}.pt"
        netc, stepc = load_net(name, ck, MID_ARCH, MID_BLOCK)
        sc = stream_cos_pair(rec_streams, probe_streams_fixed(netc, xs_probe,
                                                              256))
        lg = probe_logits_fixed(netc, xs_probe)
        cand[name] = dict(stream_cos=sc, js_vs_recipient=js_pair(rec_logits,
                                                                 lg),
                          probe_nll_shakespeare=nll_on(lg, xs_probe),
                          params=netc.num_params(), step=stepc,
                          val_ce=estimate_loss(netc, corp, "val",
                                               n_batches=12))
        log(f"candidate {name:12s}: stream-cos_all "
            f"{sc['cos_all']:.4f} (gi {sc['cos_gi']:.4f}) | JS vs recipient "
            f"{cand[name]['js_vs_recipient']:.4f} | val CE "
            f"{cand[name]['val_ce']:.4f}")
    chosen = min(cand, key=lambda n: abs(cand[n]["stream_cos"]["cos_all"]
                                         - 0.5))
    g3c = dict(rule="argmin |stream-cos_all - 0.5| over same-arch "
                    "same-corpus candidates {e040_ref, e005s_small, e040_w}",
               candidates={n: dict(cos_all=cand[n]["stream_cos"]["cos_all"],
                                   cos_gi=cand[n]["stream_cos"]["cos_gi"],
                                   js=cand[n]["js_vs_recipient"],
                                   val_ce=cand[n]["val_ce"])
                           for n in cand},
               chosen=chosen, expected=MID_NAME,
               ok=bool(chosen == MID_NAME),
               note="pre-compute recon: ALL candidates sit at the stream-cos "
                    "floor (~0.02-0.04, cf. e062 scale-B B43 0.0315) — the "
                    "W002 card's ~0.53 middle was e029's dW-alignment "
                    "ladder, not stream-cos; the intended intermediate rung "
                    "does not exist in the checkpoint pool. Dispatch "
                    "caveat: the task text named e005s_small as fallback "
                    "before measuring; e005s_small shares training seed 42 "
                    "with the recipient's recipe (not different-seed) and "
                    "loses the frozen rule (0.0233 vs e040_ref 0.0326); "
                    "e040_ref (init seed 4304) is the truer B43 analog")
    gates["G3c_middle_choice"] = g3c
    log(f"G3c middle choice: {chosen} (rule argmin|cos-0.5|) -> "
        f"{'PASS' if g3c['ok'] else 'FAIL'} | all candidates at floor "
        f"({ {n: round(cand[n]['stream_cos']['cos_all'], 4) for n in cand} })")

    # ================================================== G3a: the MIDDLE donor
    net_m, step_m = load_net(MID_NAME, MID_CKPT, MID_ARCH, MID_BLOCK)
    val_ce_m = cand[MID_NAME]["val_ce"]
    head_dim_ok = (net_m.cfg.n_embd // net_m.cfg.n_head
                   == cfg.n_embd // cfg.n_head == 32)
    g3a = dict(ckpt=str(MID_CKPT), step=step_m,
               arch=dict(n_layer=4, n_head=4, n_embd=128,
                         block_size=MID_BLOCK, vocab=corp.vocab_size),
               params=net_m.num_params(), params_expected=MID_PARAMS,
               params_ok=bool(net_m.num_params() == MID_PARAMS),
               donor_vocab_ok=bool(corp.vocab_size == VOCAB),
               head_dim_32_both=bool(head_dim_ok),
               layers_heads_identity=bool(net_m.cfg.n_layer == 4
                                          and net_m.cfg.n_head == 4),
               val_ce_own_corpus=val_ce_m, ref_e062_cached=E062_BASE_CE[MID_NAME],
               tol=0.05,
               val_ce_ok=bool(abs(val_ce_m - E062_BASE_CE[MID_NAME]) <= 0.05),
               training_note="e040 REF: fresh init set_seed(4304), 4000 "
                             "steps, batch 32, block 256, data/input.txt "
                             "(e040_graft_evolution.py); the recipient is "
                             "the e053c retrain (seed 42 recipe, ctx-512) — "
                             "same-arch, same-corpus, DIFFERENT SEED, "
                             "differently-trained (the B43 analog)")
    donors_m = {w: donor_run_mid(net_m, corp, MID_WIN[w]["prompt"],
                                 MID_WIN[w]["sample"]) for w in (1, 2)}
    for w in (1, 2):
        d = donors_m[w]
        log(f"middle donor window {w}: 8 rows 64->{DONOR_BLOCK} | online "
            f"tail CE {d['online_tail_ce']:.4f} | row0 tail: "
            f"{corp.decode(d['idx'][0, -48:])[:48]!r}")
    d1r = donor_run_mid(net_m, corp, MID_WIN[1]["prompt"], MID_WIN[1]["sample"])
    det_m = bool(torch.equal(d1r["idx"], donors_m[1]["idx"])
                 and all(torch.equal(a, b) for a, b in zip(d1r["V"],
                                                           donors_m[1]["V"])))
    g3a["donor_run_determinism"] = dict(
        rule="window-1 rerun (same seeds) tokens + V caches bit-identical",
        ok=det_m)
    g3a["ok"] = bool(g3a["params_ok"] and g3a["donor_vocab_ok"]
                     and head_dim_ok and det_m and g3a["val_ce_ok"]
                     and g3a["layers_heads_identity"])
    gates["G3a_middle_donor"] = g3a
    log(f"G3a middle donor: params/vocab/head_dim/identity/determinism/val-CE "
        f"{g3a['params_ok']}/{g3a['donor_vocab_ok']}/{head_dim_ok}/"
        f"{g3a['layers_heads_identity']}/{det_m}/{g3a['val_ce_ok']} -> "
        f"{'PASS' if g3a['ok'] else 'FAIL'}")

    # ================================================== G3b: the FOREIGN donor
    corp21 = CharCorpus(DONOR_CORPUS)                    # seed 1337
    st21 = torch.load(DONOR_CKPT, map_location="cpu", weights_only=False)
    sd21 = st21["model"] if isinstance(st21, dict) and "model" in st21 else st21
    cfg21 = Cfg(vocab=corp21.vocab_size, n_layer=6, n_head=6, n_embd=192,
                block_size=DONOR_BLOCK)
    net21 = TinyGPT(cfg21)
    net21.load_state_dict(sd21, strict=True)
    net21.eval()
    val_ce21 = estimate_loss(net21, corp21, "val", n_batches=12)
    g3b = dict(ckpt=str(DONOR_CKPT), step=int(st21.get("step", -1)),
               arch=dict(n_layer=6, n_head=6, n_embd=192,
                         block_size=DONOR_BLOCK, vocab=corp21.vocab_size),
               donor_vocab_ok=bool(corp21.vocab_size == VOCAB),
               val_ce_own_corpus=val_ce21, ref_e063b=E063B_VAL_CE_TASK,
               tol=0.15,
               val_ce_ok=bool(abs(val_ce21 - E063B_VAL_CE_TASK) <= 0.15))
    donors_f = {w: e105.donor_run(net21, corp21, e105.DONOR_WIN[w]["prompt"],
                                  e105.DONOR_WIN[w]["sample"]) for w in (1, 2)}
    d1f = e105.donor_run(net21, corp21, e105.DONOR_WIN[1]["prompt"],
                         e105.DONOR_WIN[1]["sample"])
    det_f = bool(torch.equal(d1f["idx"], donors_f[1]["idx"])
                 and all(torch.equal(a, b) for a, b in zip(d1f["V"],
                                                           donors_f[1]["V"])))
    g3b["donor_run_determinism"] = dict(rule="window-1 rerun bit-identical",
                                        ok=det_f)
    # e021 stream-cos across dims: undefined (vocab-space JS still measured)
    lg21 = probe_logits_fixed(net21, xs_probe)
    js21 = js_pair(rec_logits, lg21)
    g3b["stream_cos"] = dict(defined=False,
                             reason="n_embd 192 != 128: e062 stream-cos is "
                                    "undefined across stream dims",
                             js_vs_recipient_on_shakespeare_probe=js21,
                             probe_nll_shakespeare=nll_on(lg21, xs_probe),
                             note="off-its-corpus probe (trained on "
                                  "e021_task.txt): JS bundles task mismatch "
                                  "with geometry mismatch")
    g3b["ok"] = bool(g3b["donor_vocab_ok"] and det_f and g3b["val_ce_ok"])
    gates["G3b_foreign_donor"] = g3b
    log(f"G3b foreign donor (e021_task, step {g3b['step']}): vocab/determinism"
        f"/val-CE {g3b['donor_vocab_ok']}/{det_f}/{g3b['val_ce_ok']} -> "
        f"{'PASS' if g3b['ok'] else 'FAIL'} | stream-cos UNDEFINED (192d) | "
        f"JS vs recipient (shakespeare probe) {js21:.4f}")

    gates["G7_ladder_axis"] = dict(
        self_stream_cos=self_cos, self_ok=bool(abs(self_cos["cos_all"] - 1.0)
                                               < 1e-6),
        probe_determinism=det_probe,
        ok=bool(abs(self_cos["cos_all"] - 1.0) < 1e-6 and det_probe))
    log(f"G7 ladder axis: recipient self cos_all "
        f"{self_cos['cos_all']:.6f} | probe determinism {det_probe} -> "
        f"{'PASS' if gates['G7_ladder_axis']['ok'] else 'FAIL'}")

    # ================================================== THE FOUR ARMS
    A = {}
    gen = torch.Generator().manual_seed(e080.SEED_SAMPLE)          # none
    A["none"] = e080.generate_arm(net, prompts8, gen, "none")
    log(f"arm none        [e080 rig]: B={e080.B} 64->{e080.T_TOTAL} | "
        f"{len(A['none']['replaced'])} replaced")
    donor_rz = e099.draw_donor(e099.SEED_DONOR, e080.N_PROMPTS)    # 4343
    gen = torch.Generator().manual_seed(e080.SEED_SAMPLE)          # randomize
    A["randomize"] = e099.generate_arm5(net, prompts8, gen, "randomize",
                                        donor_rz)
    rs = A["randomize"]["repl_stats"]
    log(f"arm randomize  [e099 rig]: donor map {donor_rz} | "
        f"{len(A['randomize']['replaced'])} replaced | mean|cos(old,donor)| "
        f"{rs['mean_abs_cos_old_vs_donor']:.3f} | norm ratio "
        f"{rs['mean_norm_ratio_donor_over_old']:.3f}")
    donor_mid = e099.draw_donor(SEED_DONOR_MID, e080.N_PROMPTS)    # middle
    gen = torch.Generator().manual_seed(e080.SEED_SAMPLE)
    A[PROBE] = generate_arm_donor(net, prompts8, gen, donors_m, donor_mid)
    rs = A[PROBE]["repl_stats"]
    log(f"arm middle     [e108 ext]: donor map {donor_mid} | "
        f"{len(A[PROBE]['replaced'])} replaced | mean|cos(old,donor)| "
        f"{rs['mean_abs_cos_old_vs_donor']:.3f} | norm ratio "
        f"{rs['mean_norm_ratio_donor_over_old']:.3f}")
    donor_cf = e099.draw_donor(e105.SEED_DONOR_MAP, e080.N_PROMPTS)  # 4646
    gen = torch.Generator().manual_seed(e080.SEED_SAMPLE)          # crossfam
    A["crossfamily"] = generate_arm_donor(net, prompts8, gen, donors_f,
                                          donor_cf)
    rs = A["crossfamily"]["repl_stats"]
    log(f"arm crossfamily[e105 rig]: donor map {donor_cf} | "
        f"{len(A['crossfamily']['replaced'])} replaced | mean|cos(old,donor)| "
        f"{rs['mean_abs_cos_old_vs_donor']:.3f} | norm ratio "
        f"{rs['mean_norm_ratio_donor_over_old']:.3f}")

    # G6: middle full-pipeline determinism (fresh donors + fresh arm)
    donors_m2 = {w: donor_run_mid(net_m, corp, MID_WIN[w]["prompt"],
                                  MID_WIN[w]["sample"]) for w in (1, 2)}
    gen = torch.Generator().manual_seed(e080.SEED_SAMPLE)
    rerun = generate_arm_donor(net, prompts8, gen, donors_m2,
                               e099.draw_donor(SEED_DONOR_MID,
                                               e080.N_PROMPTS))
    g6 = bool(torch.equal(rerun["idx"], A[PROBE]["idx"]))
    gates["G6_middle_determinism"] = dict(
        rule="middle arm (donors regenerated + seed-7 stream) rerun token "
             "stream bit-identical", ok=g6)
    log(f"G6 middle determinism: {g6}")

    # ---- G5 intervention identity
    pre = e080.PRUNE_START_G + e080.PROMPT_TOK + 1       # 165: cols 0..164
    ident = all(torch.equal(A[ARMS[0]]["idx"][:, :pre], A[a]["idx"][:, :pre])
                for a in ARMS)
    exp_repl = {a: (0 if a == "none" else 324) for a in ARMS}
    counts_ok = all(len(A[a]["replaced"]) == exp_repl[a] for a in ARMS)
    derange_ok = (all(d != b for b, d in enumerate(donor_rz))
                  and all(d != b for b, d in enumerate(donor_mid))
                  and all(d != b for b, d in enumerate(donor_cf)))
    donor_pos_ok = all(64 <= donor_source(p)[1] <= 255
                       and donor_source(p)[0] in (1, 2)
                       for p in range(e080.GEN_FIRST, 388))
    cache_oks = {a: A[a]["cache"].get("ok", True) for a in ARMS}
    ok5 = bool(ident and counts_ok and derange_ok and donor_pos_ok
               and all(cache_oks.values()))
    gates["G5_intervention_identity"] = dict(
        pre_event_identity_through_position=pre - 1, identical=ident,
        replaced_counts={a: len(A[a]["replaced"]) for a in ARMS},
        replaced_counts_expected=exp_repl, counts_ok=counts_ok,
        donor_map_randomize=list(donor_rz),
        donor_map_middle=list(donor_mid),
        donor_map_crossfamily=list(donor_cf),
        donor_derangements_ok=derange_ok,
        donor_position_map=dict(
            rule="p<=255 <- window1 pos p; p in [256,387] <- window2 pos "
                 "p-192; all sources within donor-native generated band "
                 "64..255; zero reuse (e105 rule, both donor arms)",
            all_sources_native=donor_pos_ok),
        cache_oks=cache_oks, ok=ok5)
    log(f"G5: pre-event identity through pos {pre - 1}: {ident} | counts "
        f"{ {a: len(A[a]['replaced']) for a in ARMS} } | derangements "
        f"{derange_ok} | donor sources native {donor_pos_ok} -> "
        f"{'PASS' if ok5 else 'FAIL'}")

    # ---- G4 drift vs e099 (none, randomize) and e105 (crossfamily)
    g4 = dict(refs=[str(E099_METRICS), str(E105_METRICS)], ok=None, note="")
    devs = {}
    ref_data = {}
    for path, arms_ref in ((E099_METRICS, ["none", "randomize"]),
                           (E105_METRICS, ["crossfamily"])):
        if path.exists():
            with open(path) as f:
                mref = json.load(f)
            ref_data.update({a: mref["collapse_signature"][a]
                             for a in arms_ref})
        else:
            g4["note"] += f"{path.name} missing; "
    if ref_data:
        for arm, ref in ref_data.items():
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
        g4["note"] += (f"legacy arms rerun vs published metrics (final-128 "
                       f"tail, online/clean-judge/per-seq gap; same 8-thread "
                       f"setting): max dev {max_dev:.2e}")
    gates["G4_rerun_vs_published"] = g4
    log(f"G4 drift vs published: {g4['note']} -> "
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

    # ================================================== TAIL ANALYSIS (texture)
    full_ids = torch.cat([corp.train, corp.val]).numpy()
    p_corpus = smooth(counts_of(full_ids))
    val_np = corp.val.numpy()
    starts = np.linspace(0, len(val_np) - TAIL, N_WIN).astype(int)
    win_tokens = [val_np[s:s + TAIL] for s in starts]
    corpus_rep = {n: float(np.mean([repeat_rate(w, n)
                                    for w in win_tokens])) for n in NGRAMS}
    log(f"corpus baselines: shakespeare unigram H {entropy(p_corpus):.4f} | "
        f"window repeat rates "
        f"{ {n: round(v, 3) for n, v in corpus_rep.items()} }")

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
            decoded_tails=[corp.decode(torch.tensor(t)) for t in T],
        )
        log(f"  tail {arm:11s}: H {stats[arm]['pooled_entropy']:.4f} | "
            f"distinct {stats[arm]['n_distinct']} | top1 "
            f"{stats[arm]['top1_mass']:.3f} | top5 "
            f"{[(d['char'], round(d['freq'], 3)) for d in stats[arm]['top5']]} "
            f"| rep2 {rep[2]:.3f} | KL||corpus "
            f"{stats[arm]['kl_vs_corpus']:.4f}")

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
    floor_c = (float(np.mean([np.mean(nulls[a]) for a in collapsed]))
               if len(collapsed) >= 2 else
               float(np.mean([np.mean(nulls[a]) for a in ARMS])))
    floor_h = float(np.mean([np.mean(nulls[a]) for a in HEALTHY_REFS]))
    dc = {a: float(np.mean([grid[ARMS.index(a), ARMS.index(c)]
                            for c in collapsed])) if collapsed else float("nan")
          for a in ARMS}
    dh = {a: float(np.mean([grid[ARMS.index(a), ARMS.index(h)]
                            for h in HEALTHY_REFS])) for a in ARMS}
    joins_c = {a: bool(dc[a] <= CLUSTER_RATIO * floor_c and dh[a] > dc[a])
               for a in ARMS}
    joins_h = {a: bool(dh[a] <= CLUSTER_RATIO * floor_h and dc[a] > dh[a])
               for a in ARMS}
    log(f"clusters (texture): floor_c {floor_c:.4f} floor_h {floor_h:.4f} | "
        f"collapsed {collapsed}")
    for a in ARMS:
        log(f"  {a:11s}: d_c {dc[a]:.3f} ({dc[a] / floor_c:.1f}x floor_c) | "
            f"d_h {dh[a]:.3f} ({dh[a] / floor_h:.1f}x floor_h) | joins "
            f"collapsed {joins_c[a]} / healthy {joins_h[a]}")

    # ================================================== THE MEASURED LADDER
    v_cos = {a: A[a]["repl_stats"].get("mean_abs_cos_old_vs_donor")
             for a in ARMS}
    js_axis = dict(
        randomize=0.0,                                  # donor IS the recipient
        middle=cand[MID_NAME]["js_vs_recipient"],
        crossfamily=js21)
    ladder = dict(
        rungs=[
            dict(anchor="sibling (randomize)", arm="randomize", donor_net=
                 "e053c_ctx512 itself (sibling battery rows)",
                 stream_cos=self_cos["cos_all"], stream_cos_gi=self_cos["cos_gi"],
                 v_cos=v_cos["randomize"], js=js_axis["randomize"],
                 gap=sig["randomize"]["gap"]),
            dict(anchor="middle", arm=PROBE, donor_net=MID_NAME,
                 stream_cos=cand[MID_NAME]["stream_cos"]["cos_all"],
                 stream_cos_gi=cand[MID_NAME]["stream_cos"]["cos_gi"],
                 v_cos=v_cos[PROBE], js=js_axis[PROBE],
                 gap=sig[PROBE]["gap"]),
            dict(anchor="foreign (crossfamily)", arm="crossfamily",
                 donor_net="e021_task (copy net)",
                 stream_cos=None, stream_cos_gi=None,
                 v_cos=v_cos["crossfamily"], js=js_axis["crossfamily"],
                 gap=sig["crossfamily"]["gap"]),
        ],
        axis_notes=dict(
            stream_cos="e062 2-batch probe (common block 256, seed-1337 "
                       "stream), cos_all = mean over depths d0..d4; sibling "
                       "= 1.0 by identity; foreign UNDEFINED (192d stream)",
            v_cos="mean|cos(old V, donor V)| over all spliced vectors at "
                  "event time (e099/e105 repl_stats instrument)",
            js="mean JS of next-token predictive distributions on the shared "
               "shakespeare probe; foreign's JS bundles task mismatch"),
        monotonicity={})
    for axis, key in (("stream_cos", "stream_cos"), ("v_cos", "v_cos"),
                      ("js", "js")):
        rungs = [r for r in ladder["rungs"] if r[key] is not None]
        order = sorted(rungs, key=lambda r: -r[key])
        gaps_ok = all(order[i]["gap"] <= order[i + 1]["gap"] + 1e-9
                      for i in range(len(order) - 1))
        cos_ok = all(order[i][key] >= order[i + 1][key] - 1e-12
                     for i in range(len(order) - 1))
        ladder["monotonicity"][axis] = dict(
            rungs=[(r[key], r["gap"], r["anchor"]) for r in order],
            n_rungs=len(order), gaps_nondecreasing_as_distance_grows=gaps_ok,
            note=("foreign rung absent (undefined)" if len(rungs) < 3
                  else "all three rungs measured"))
        log(f"ladder axis {axis:10s}: rungs "
            f"{[(round(r[key], 4) if r[key] is not None else None, round(r['gap'], 3)) for r in ladder['rungs']]} "
            f"| gap monotone in distance: {gaps_ok}")

    # ================================================== REGISTERED DECISION
    gap_m = sig[PROBE]["gap"]
    gap_f = sig["crossfamily"]["gap"]
    gap_r = sig["randomize"]["gap"]
    controls = dict(
        randomize_healthy=bool(gap_r < GAP_HEALTHY),
        crossfamily_collapsed=bool(gap_f > GAP_BAR),
        randomize_gap_ref=E099_GAP_RANDOMIZE,
        crossfamily_gap_ref=E105_GAP_CROSSFAMILY)
    controls_ok = bool(controls["randomize_healthy"]
                       and controls["crossfamily_collapsed"])
    monotone = bool(gap_m < GAP_HEALTHY and controls["crossfamily_collapsed"])
    sharp = bool(gap_m > GAP_BAR)
    clauses = dict(
        middle_gap_healthy=dict(rule=f"gap(middle) < {GAP_HEALTHY}",
                                gap=gap_m, fires=bool(gap_m < GAP_HEALTHY)),
        foreign_collapsed=dict(rule=f"gap(crossfamily) > {GAP_BAR}",
                               gap=gap_f,
                               fires=controls["crossfamily_collapsed"]),
        middle_gap_collapsed=dict(rule=f"gap(middle) > {GAP_BAR}", gap=gap_m,
                                  fires=bool(gap_m > GAP_BAR)),
        randomize_control_healthy=dict(rule=f"gap(randomize) < {GAP_HEALTHY}",
                                       gap=gap_r,
                                       fires=controls["randomize_healthy"]),
        monotone=dict(rule=f"gap(middle) < {GAP_HEALTHY} AND "
                           f"gap(crossfamily) > {GAP_BAR}", fires=monotone),
        sharp_family=dict(rule=f"gap(middle) > {GAP_BAR} (middle collapsed "
                               f"like foreign)", fires=sharp))
    mono = ladder["monotonicity"]
    if monotone:
        clause = "MONOTONE"
        verdict = (
            f"MONOTONE fires: the middle arm stayed HEALTHY (clean-judge gap "
            f"{gap_m:+.3f} < {GAP_HEALTHY}) while the foreign net collapsed "
            f"({gap_f:+.3f} > {GAP_BAR}) — anchor health organizes by donor "
            f"distance. MEASURED-AXIS CAVEAT (honest): the middle donor sits "
            f"at stream-cos {cand[MID_NAME]['stream_cos']['cos_all']:.3f} = "
            f"the FLOOR (e062's own B43 value is 0.0315), not the ~0.53 the "
            f"W002 card assumed — the intermediate rung never existed. What "
            f"the ladder actually shows: two floor-distance nets "
            f"({MID_NAME} healthy / e021_task collapsed) split by the "
            f"measured V-cos "
            f"({v_cos[PROBE]:.3f} vs {v_cos['crossfamily']:.3f}) and JS "
            f"({js_axis[PROBE]:.3f} vs {js_axis['crossfamily']:.3f}) axes "
            f"(monotone there: "
            f"{mono['v_cos']['gaps_nondecreasing_as_distance_grows']}/"
            f"{mono['js']['gaps_nondecreasing_as_distance_grows']}) — read "
            f"the unification through the axes that separate the rungs, and "
            f"as a step near cos 1.0 on the stream axis, not a graded law.")
    elif sharp:
        clause = "SHARP FAMILY"
        verdict = (
            f"SHARP FAMILY fires: the middle arm COLLAPSED like the foreign "
            f"net (gap {gap_m:+.3f} > {GAP_BAR} vs foreign {gap_f:+.3f}) "
            f"where the same-net sibling stayed healthy ({gap_r:+.3f}) — the "
            f"text-axis family boundary is SHARPER than the net-axis "
            f"instrument: a same-arch same-corpus differently-trained net "
            f"(stream-cos "
            f"{cand[MID_NAME]['stream_cos']['cos_all']:.3f}, at the "
            f"cross-family floor) cannot anchor the run. The bilinear "
            f"unification is a metaphor at this resolution: the anchor does "
            f"not grade donor distance, it demands the trained weights "
            f"themselves (T057's sharpest form, now with the middle rung "
            f"measured).")
    else:
        clause = "PARTIAL/AMBIGUOUS"
        bits = [f"middle gap {gap_m:+.3f} (gray zone "
                f"[{GAP_HEALTHY},{GAP_BAR}])"
                if GAP_HEALTHY <= gap_m <= GAP_BAR
                else f"middle gap {gap_m:+.3f}"]
        bits.append(f"d_c {dc[PROBE]:.3f} ({dc[PROBE] / floor_c:.1f}x "
                    f"floor_c) vs d_h {dh[PROBE]:.3f} "
                    f"({dh[PROBE] / floor_h:.1f}x floor_h); joins collapsed "
                    f"{joins_c[PROBE]} / healthy {joins_h[PROBE]}")
        if not controls_ok:
            bits.append(f"CONTROLS: randomize healthy "
                        f"{controls['randomize_healthy']} (gap {gap_r:+.3f}), "
                        f"crossfamily collapsed "
                        f"{controls['crossfamily_collapsed']} (gap "
                        f"{gap_f:+.3f}) — treat verdicts as invalid")
        bits.append(f"measured axes: stream-cos(middle) "
                    f"{cand[MID_NAME]['stream_cos']['cos_all']:.3f} (floor), "
                    f"V-cos sib/mid/for "
                    f"{v_cos['randomize']:.3f}/{v_cos[PROBE]:.3f}/"
                    f"{v_cos['crossfamily']:.3f}, JS "
                    f"{js_axis['randomize']:.3f}/{js_axis[PROBE]:.3f}/"
                    f"{js_axis['crossfamily']:.3f}")
        verdict = ("PARTIAL/AMBIGUOUS (honest texture): " + "; ".join(bits)
                   + ".")
    log(f"REGISTERED DECISION [{clause}]: {verdict}")
    log(f"  controls: randomize healthy {controls['randomize_healthy']} "
        f"(gap {gap_r:+.3f}, e099 ref {E099_GAP_RANDOMIZE:+.3f}) | "
        f"crossfamily collapsed {controls['crossfamily_collapsed']} (gap "
        f"{gap_f:+.3f}, e105 ref {E105_GAP_CROSSFAMILY:+.3f}) | controls_ok "
        f"{controls_ok}")

    # ---------------------------------------------------------------- metrics
    metrics = dict(
        experiment="e108_distance_ladder",
        purpose="W002's registered DISTANCE-LADDER anchor test: does anchor "
                "health track CROSSMATCH DISTANCE monotonically? e099/e105 "
                "rig verbatim (recipient e053c_ctx512, matched seed-7 "
                "streams, g=100..420 K=32 self-band splice, clean-judge "
                "final-128 gap) with four arms: none / randomize (sibling "
                "control) / middle (e040_ref — closest same-arch "
                "same-corpus differently-trained net; free-run V entries, "
                "e105 splice procedure) / crossfamily (e021_task copy net, "
                "e105 verbatim). The ladder's x-axis is MEASURED: e062 "
                "2-batch stream-cos (common block 256) for every "
                "dims-compatible net + splice-time V-cos + vocab-space JS "
                "for all three anchors. FROZEN bars: gap(middle)<0.3 AND "
                "gap(crossfamily)>1 => MONOTONE; gap(middle)>1 => SHARP "
                "FAMILY; else PARTIAL/AMBIGUOUS.",
        started=started, wall_s=elapsed(), threads=THREADS, cpu_only=True,
        recipient_net=dict(ckpt=str(e080.CKPT),
                           arch=dict(n_layer=4, n_head=4, n_embd=128,
                                     block_size=e080.T_TOTAL, vocab=VOCAB),
                           params=n_params, val_ce=val_ce,
                           val_ce_e053c=e080.E053C_VAL_CE),
        middle_donor=g3a,
        foreign_donor=g3b,
        middle_choice=g3c,
        ladder_axes=dict(
            stream_cos=dict(method="e062.probe_streams, 2 batches x 16 rows "
                                   "x 256 tokens, RNG seed 1337, common "
                                   "block so shapes match; cos per depth, "
                                   "cos_gi=mean(d2,d3), cos_all=mean(d0..d4)",
                            recipient_self=self_cos,
                            candidates={n: cand[n]["stream_cos"] for n in cand},
                            e062_scaleB_refs=dict(
                                b43_same_family_diff_init=E062_B43_COS,
                                bdo_same_init_diff_data_order=E062_BDO_COS)),
            js=dict(method="mean JS of next-token predictive distributions "
                           "on the shared shakespeare probe windows",
                    per_net=dict(recipient_nll=nll_on(rec_logits, xs_probe),
                                 **{n: dict(js_vs_recipient=cand[n]
                                            ["js_vs_recipient"],
                                            probe_nll=cand[n]
                                            ["probe_nll_shakespeare"])
                                    for n in cand},
                                 e021_task=dict(
                                     js_vs_recipient=js21,
                                     probe_nll=nll_on(lg21, xs_probe)))),
            v_cos=dict(method="mean|cos(old V, donor V)| over spliced "
                              "vectors at event time (repl_stats)",
                       per_arm=v_cos),
            ladder=ladder),
        seeds=dict(corpus=1337, prompts=e080.SEED_PROMPT,
                   sampling=e080.SEED_SAMPLE,
                   donor_map_randomize=e099.SEED_DONOR,
                   donor_map_middle=SEED_DONOR_MID,
                   donor_map_crossfamily=e105.SEED_DONOR_MAP,
                   middle_donor_windows=MID_WIN,
                   foreign_donor_windows=e105.DONOR_WIN,
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
                          middle="V<-e040_ref's own free-run entries on "
                                 "data/input.txt (e105 splice procedure; "
                                 "windows 5811/5911+5812/5912, derangement "
                                 "4747; L/H identity map, d=32)",
                          crossfamily="V<-e021_task copy-task net's own "
                                      "free-run entries on its own corpus "
                                      "(e105 VERBATIM: windows "
                                      "5601/5701+5602/5702, derangement "
                                      "4646)"),
                      schedule=dict(events_g=e080.EVENTS, K=e080.PRUNE_K,
                                    start_g=e080.PRUNE_START_G,
                                    age_cut=e080.AGE_CUT,
                                    note="identical to e080/e099/e105: all "
                                         "four arms token-identical through "
                                         "position 164; 324 replaced "
                                         "positions in each trigger arm"),
                      collapse_gate=f"clean-judge tail CE - online tail CE "
                                    f"> {GAP_BAR} nats (e099 convention)"),
        gates=gates,
        char_map=[corp.itos[t] for t in range(VOCAB)],
        collapse_signature={a: sig[a] for a in ARMS},
        collapsed_arms=collapsed,
        corpus_baseline=dict(
            unigram_entropy=entropy(p_corpus),
            window_repeat_ngram_rates=corpus_rep),
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
        cluster_map=dict(d_c=dc, d_h=dh, joins_collapsed=joins_c,
                         joins_healthy=joins_h,
                         note="TEXTURE ONLY (4 arms; e105's calibrated rule "
                              "assumed >=2 collapsed refs)"),
        donor_rider=dict(
            note="middle + foreign donors' own generated-tail samples "
                 "(competence context)",
            middle=dict(online_tail_ce={w: donors_m[w]["online_tail_ce"]
                                        for w in (1, 2)},
                        tail_samples=[corp.decode(donors_m[w]["idx"][s, -64:])
                                      for w in (1, 2) for s in (0, 1)]),
            foreign=dict(online_tail_ce={w: donors_f[w]["online_tail_ce"]
                                         for w in (1, 2)},
                         tail_samples=[corp21.decode(donors_f[w]["idx"][s,
                                                                      -64:])
                                       for w in (1, 2) for s in (0, 1)])),
        registered_decision=dict(
            frozen_rules=dict(
                monotone=f"gap(middle) < {GAP_HEALTHY} AND "
                         f"gap(crossfamily) > {GAP_BAR}",
                sharp_family=f"gap(middle) > {GAP_BAR}",
                else_="PARTIAL/AMBIGUOUS (report the honest texture)"),
            controls=controls, controls_ok=controls_ok,
            clauses=clauses, clause=clause, verdict=verdict),
    )
    save_json(out_dir / "metrics.json", metrics)
    log("metrics.json written")

    plot(out_dir / "distance_ladder.png", metrics)
    log(f"plot written; total wall {elapsed():.0f}s")


# ---------------------------------------------------------------------- plot


def plot(path: Path, M: dict):
    arms = M["kl_grid"]["arm_order"]
    grid = np.asarray(M["kl_grid"]["symmetric"])
    sig = M["collapse_signature"]
    cm = M["cluster_map"]
    dec = M["registered_decision"]
    lad = M["ladder_axes"]["ladder"]
    rungs = lad["rungs"]
    mono = lad["monotonicity"]
    cols = {a: ("tab:gray" if a == "none" else
                "tab:green" if a == "randomize" else
                "tab:purple" if a == "middle" else "tab:red")
            for a in arms}
    rcols = {"sibling (randomize)": "tab:green", "middle": "tab:purple",
             "foreign (crossfamily)": "tab:red"}
    fig, axes = plt.subplots(2, 3, figsize=(24, 11))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    # ---- panel 1: THE LADDER — gap vs measured stream-cos
    xs = [r["stream_cos"] for r in rungs if r["stream_cos"] is not None]
    ys = [r["gap"] for r in rungs if r["stream_cos"] is not None]
    lab = [r["anchor"] for r in rungs if r["stream_cos"] is not None]
    ax1.plot(xs, ys, "-", color="k", lw=1.0, alpha=0.5, zorder=2)
    for r in rungs:
        x = r["stream_cos"]
        y = r["gap"]
        if x is None:
            ax1.scatter([-0.03], [y], s=230, facecolor="none",
                        edgecolor=rcols[r["anchor"]], lw=2.5, zorder=4)
            ax1.annotate(f"{r['anchor']}\n(stream-cos UNDEFINED, 192d;\n"
                         f"V-cos {r['v_cos']:.3f}, JS {r['js']:.2f})",
                         (-0.03, y), textcoords="offset points",
                         xytext=(10, -6), fontsize=8.5)
        else:
            ax1.scatter([x], [y], s=230, color=rcols[r["anchor"]],
                        edgecolor="k", zorder=4,
                        marker=("D" if r["anchor"] == "middle" else "o"))
            ax1.annotate(f"{r['anchor']}\ncos {x:.3f}, gap {y:+.3f}",
                         (x, y), textcoords="offset points",
                         xytext=(10, -6), fontsize=8.5)
    ax1.axhline(GAP_HEALTHY, color="tab:green", ls="--", lw=1.4)
    ax1.axhline(GAP_BAR, color="tab:red", ls="--", lw=1.4)
    ax1.text(0.98, GAP_HEALTHY + 0.06, f"healthy bar < {GAP_HEALTHY}",
             fontsize=9, color="tab:green", ha="right")
    ax1.text(0.98, GAP_BAR + 0.06, f"collapse bar > {GAP_BAR}", fontsize=9,
             color="tab:red", ha="right")
    ax1.axhspan(GAP_HEALTHY, GAP_BAR, color="gray", alpha=0.10)
    ax1.set_xlim(-0.12, 1.08)
    ax1.set_xlabel("donor stream-cos vs recipient (e062 2-batch probe, "
                   "cos_all)")
    ax1.set_ylabel("clean-judge tail gap (nats, final 128)")
    ax1.set_title("E108-1 — THE DISTANCE LADDER: anchor gap vs measured "
                  "crossmatch distance\n(sibling = 1.0 by identity; all "
                  "available middle candidates sit at the ~0.02-0.04 floor; "
                  "foreign rung undefined)", fontsize=10)

    # ---- panel 2: arm gaps + registered bars
    x = np.arange(len(arms))
    gaps = [sig[a]["gap"] for a in arms]
    yerr = [[max(0.0, g - sig[a]["gap_ci"][0]) for g, a in zip(gaps, arms)],
            [max(0.0, sig[a]["gap_ci"][1] - g)
             for g, a in zip(gaps, arms)]]
    ax2.bar(x, gaps, 0.62, color=[cols[a] for a in arms], alpha=0.85,
            edgecolor="k", lw=0.5)
    ax2.errorbar(x, gaps, yerr=yerr, fmt="none", ecolor="k", lw=1.0,
                 capsize=3)
    ax2.axhline(GAP_HEALTHY, color="tab:green", ls="--", lw=1.4)
    ax2.axhline(GAP_BAR, color="tab:red", ls="--", lw=1.4)
    for xi, a in zip(x, arms):
        ax2.text(xi, max(gaps[xi], 0) + 0.25, f"{gaps[xi]:+.3f}",
                 ha="center", fontsize=9)
    ax2.set_xticks(x, [f"{a}\n(gap {sig[a]['gap']:+.2f})" for a in arms],
                   fontsize=8.5)
    ax2.set_ylabel("clean-judge tail CE - online tail CE (nats)")
    ax2.set_ylim(min(0, min(gaps)) - 0.3, max(gaps) * 1.18 + 0.3)
    ax2.set_title("E108-2 — the four arms' gaps (matched seed-7 streams)",
                  fontsize=10)

    # ---- panel 3: secondary measured axes — gap vs V-cos (all 3 anchors)
    for r in rungs:
        ax3.scatter([r["v_cos"]], [r["gap"]], s=230,
                    color=rcols[r["anchor"]], edgecolor="k", zorder=4,
                    marker=("D" if r["anchor"] == "middle" else "o"))
        ax3.annotate(f"{r['anchor']}\nV-cos {r['v_cos']:.3f}, JS "
                     f"{r['js']:.2f}", (r["v_cos"], r["gap"]),
                     textcoords="offset points", xytext=(8, -6), fontsize=8.5)
    order = sorted(rungs, key=lambda r: -r["v_cos"])
    ax3.plot([r["v_cos"] for r in order], [r["gap"] for r in order], "-",
             color="k", lw=1.0, alpha=0.5, zorder=2)
    ax3.axhline(GAP_HEALTHY, color="tab:green", ls="--", lw=1.4)
    ax3.axhline(GAP_BAR, color="tab:red", ls="--", lw=1.4)
    ax3.set_xlabel("splice-time V-geometry mean|cos(old, donor)| "
                   "(all three anchors measured)")
    ax3.set_ylabel("clean-judge tail gap (nats)")
    ax3.set_title("E108-3 — the ladder on the V-geometry axis (JS annotated; "
                  f"monotone: "
                  f"{mono['v_cos']['gaps_nondecreasing_as_distance_grows']})",
                  fontsize=10)

    # ---- panel 4: terminal-KL cluster map (texture)
    finite = grid[~np.isnan(grid)]
    gpos = grid[grid > 0]
    vmin = gpos.min() if gpos.size else 1e-3
    im = ax4.imshow(grid, cmap="viridis",
                    norm=LogNorm(vmin=max(1e-4, vmin),
                                 vmax=max(0.01, finite.max())))
    for i in range(len(arms)):
        for j in range(len(arms)):
            v = grid[i, j]
            ax4.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.5,
                     color="white" if v > finite.max() / 3 else "black")
    ax4.add_patch(Rectangle((-0.5, -0.5), 2, 2, fill=False,
                            ec="tab:green", lw=2.5))
    ax4.add_patch(Rectangle((1.5, 1.5), 1, 1, fill=False,
                            ec="tab:purple", lw=2.5, ls="--"))
    ax4.set_xticks(range(len(arms)), arms, fontsize=8, rotation=30)
    ax4.set_yticks(range(len(arms)), arms, fontsize=8)
    fig.colorbar(im, ax=ax4, label="symmetric KL (nats, log scale)")
    ax4.set_title("E108-4 — terminal-KL cluster map (texture; diag = "
                  "within-arm null)", fontsize=10)

    # ---- panel 5: the ladder table + middle-choice measurement
    ax5.axis("off")
    lines = ["THE MEASURED LADDER (x-axis measured, not assumed):",
             "anchor            donor net                  stream-cos  "
             "V-cos    JS      gap     verdict",
             "-" * 100]
    for r in rungs:
        sc = f"{r['stream_cos']:.3f}" if r["stream_cos"] is not None \
            else "  N/A "
        cl = ("healthy" if r["gap"] < GAP_HEALTHY else
              "COLLAPSED" if r["gap"] > GAP_BAR else "gray")
        dn = r["donor_net"][:24]
        lines.append(f"{r['anchor']:17s} {dn:26s} {sc:9s}  "
                     f"{r['v_cos']:.3f}  {r['js']:6.3f}  "
                     f"{r['gap']:+6.3f}  {cl}")
    lines.append("")
    lines.append("middle-choice rule (argmin |cos-0.5| over same-arch "
                 "same-corpus candidates):")
    for n, c in M["middle_choice"]["candidates"].items():
        lines.append(f"  {n:14s} cos_all {c['cos_all']:.4f}  cos_gi "
                     f"{c['cos_gi']:.4f}  JS {c['js']:.3f}  val CE "
                     f"{c['val_ce']:.4f}")
    lines.append(f"  chosen: {M['middle_choice']['chosen']} | "
                 f"e062 refs: B43 (same-family diff-init) cos_all "
                 f"{E062_B43_COS['all']:.4f}, BDO (same-init) "
                 f"{E062_BDO_COS['all']:.4f}")
    lines.append("  NOTE: the W002 ~0.53 middle was e029's dW-alignment "
                 "number, not stream-cos;")
    lines.append("        every available candidate sits at the floor — "
                 "the intermediate rung does not exist.")
    lines.append("")
    lines.append(f"monotonicity: stream-cos axis "
                 f"{mono['stream_cos']['gaps_nondecreasing_as_distance_grows']} "
                 f"({mono['stream_cos']['n_rungs']} rungs) | V-cos axis "
                 f"{mono['v_cos']['gaps_nondecreasing_as_distance_grows']} "
                 f"(3 rungs) | JS axis "
                 f"{mono['js']['gaps_nondecreasing_as_distance_grows']} "
                 f"(3 rungs)")
    lines.append("")
    lines.append(f"controls: randomize healthy "
                 f"[{dec['controls']['randomize_healthy']}] gap "
                 f"{sig['randomize']['gap']:+.3f} (e099 ref "
                 f"{E099_GAP_RANDOMIZE:+.3f})")
    lines.append(f"          crossfamily collapsed "
                 f"[{dec['controls']['crossfamily_collapsed']}] gap "
                 f"{sig['crossfamily']['gap']:+.3f} (e105 ref "
                 f"{E105_GAP_CROSSFAMILY:+.3f})")
    ax5.text(0.02, 0.97, "E108-5 — ladder table + the measured x-axis",
             fontsize=12, weight="bold", va="top")
    for i, t in enumerate(lines):
        ax5.text(0.02, 0.935 - i * 0.032, t, fontsize=8.3, va="top",
                 family="monospace")

    # ---- panel 6: the registered decision
    ax6.axis("off")
    l6 = ["REGISTERED BARS (frozen):",
          f"  MONOTONE:     gap(middle) < {GAP_HEALTHY} AND "
          f"gap(crossfamily) > {GAP_BAR}",
          f"  SHARP FAMILY: gap(middle) > {GAP_BAR}",
          f"  else PARTIAL/AMBIGUOUS (honest texture)", "", "clauses:"]
    for k, v in dec["clauses"].items():
        l6.append(f"  {k}: {'FIRES' if v['fires'] else 'no'}")
    l6.append(f"  controls_ok: {dec['controls_ok']}")
    l6 += ["", f"VERDICT [{dec['clause']}]:"] + \
        [f"  {wd}" for wd in textwrap.wrap(dec["verdict"], 96)]
    ax6.text(0.02, 0.97, "E108-6 — does anchor health track crossmatch "
                         "distance?", fontsize=12, weight="bold", va="top")
    for i, t in enumerate(l6):
        ax6.text(0.02, 0.94 - i * 0.034, t, fontsize=8.7, va="top",
                 family="monospace")

    fig.suptitle(f"E108 — the distance-ladder anchor test (W002) | "
                 f"{dec['clause']} | middle gap "
                 f"{sig['middle']['gap']:+.3f} at stream-cos "
                 f"{rungs[1]['stream_cos']:.3f} (floor) | foreign gap "
                 f"{sig['crossfamily']['gap']:+.3f} | sibling gap "
                 f"{sig['randomize']['gap']:+.3f}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
