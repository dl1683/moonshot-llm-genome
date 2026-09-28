"""E146 — THE DISSOCIATION MATRIX: does the SELF survive losing its pivot?
(W015's registered design, first contact between the memory arc and the
self arc; QUEUE.md row e146; bars frozen in this docstring BEFORE compute).

THE QUESTION (W015): the self-recognition machinery (the binary self/other
step; the k*-dim signature subspace; exclusion of foreign nets — T060-T062)
is measured through the anchor's V-structure reads, but nobody has asked
WHICH LAYER computes it. The week built exactly the instruments to ask:
the fact is SINK-COUPLED (mask spares it at CE +0.03 — the information
probe; poison kills it at norm<=0.07 — the health probe; perm costs 16-21%
at novel geometry — the direction probe; e150/T086). THE MATRIX:
interventions x functions, every cell priced by CE (the e150 discipline).

NETS (eval-only, loaded + gated; CPU-ONLY — e151 owns the GPU, never
touched):
  * PRIMARY    runs/checkpoints/e131_consolidated_e113.pt (B43 line,
    rel 0.878 — chosen deliberately over the saturating e098 family,
    T085/W016 instrument note). Gates: g0 p(Z)=0.7850371599197388,
    CE_R=1.663516640663147 (bit-repro expected, e150's own gates).
  * PARENT     runs/checkpoints/e048_repro.pt (install-phase ancestor of
    the consolidated line — the self-baseline's matched "middle" donor;
    gate g0 = 0.5563086867332458).
  * FOREIGN    runs/checkpoints/e021_task.train.pt (e105's cross-family
    donor, 6L/6H/192d/blk256 — SAME ARCH as the primary; its V-caches are
    the maximally-foreign donor set, e105's donor_run verbatim).
  * SITE-CTL   runs/checkpoints/e131_arm_b_corpus_spliced.pt (site-stored
    control column, mask specificity: std g0 floor 0.007800613064318895,
    site onset 0.9880021214485168).

INSTRUMENT TRANSFER (the R45 ideator's design note — registered FIRST
STEP, gates the whole matrix): the self-recognition instruments' home
lineage is e053c_ctx512 (4L/4H/128d/blk512, runs/e111+e112). The primary
net is a different lineage AND arch (6L/6H/192d/blk256), so the rig is
PORTED (geometry rescaled to blk256: prompts 64 + free run to 256,
intervention at t=192, anchor band = self-generated positions 64..143
(80 entries, ages 49..128 at query — the e096 age-97 convention cannot
fit in 256 tokens, documented deviation), matched continuation seed
960250, judged tail = final 32 positions (224..255, the e112 64-of-512
tail scaled 1:2)). The SELF-BASELINE must establish the metric works on
the new lineage BEFORE any intervention cell counts:
  - sibling arm (e099 randomize rule: the net's OWN prefill rows,
    seed-4343 map)          => gap < 0.3  (healthy)
  - corpus arm  (e080/e099 promptcopy rule, q=((p-64) mod 63)+1)
                              => gap > 1.0 (collapsed)
  - foreign arm  (e105 crossfamily rule: e021 window-1 V, position-
    matched, seed-4646 map)  => gap > 1.0 (collapsed)
  - cos step (e118's bar): mean|cos(old,sib)| >= 2x mean|cos(old,for)|
  - occupancy (e111's bar): EXISTS k <= 8 with E_sib(k) >= 2x E_for(k)
    AND same-norm null97 < both  => name k* (the home value was 7; the
    primary net's own k* is NAMED here, not assumed)
  If the functional clauses fail, the verdict is INSTRUMENT-INVALID
  (numbers reported; no bar shopping).
  Baseline texture arms (never gating): PARENT donor (the install-phase
  ancestor's V — does the consolidated anchor accept its own past?) and
  SIG-DESTROYED (sibling V projected OFF the top-k* subspace, norms
  kept — e112's causal-stamp test, baseline only).

INTERVENTIONS x FUNCTIONS (the dispatch design verbatim):
  interventions: (1) forced-off-sink MASK (information probe; -inf on
  key-0 attention logits for queries>=1, all layers/heads, eval-only,
  e150's gated custom forward); (2) NORM POISON LADDER on wpe[0]:
  0.07 (kills the fact, CE +0.84) / 0.15 (spares it, CE +0.31) — the
  health probe; (3) direction-PERM of wpe[0] (consult probe; e141's
  exact seed 14101 => the SAME permutation); (4) fact-specific-head
  ablation (e133's L0H3 class — recomputed at runtime from
  runs/e133/metrics.json by drop>=0.3 & CE<=0.35; mean-replace with the
  CE_R-bank mean, e150's convention). CONTROLS: row-1 deletion (wpe[1]
  zeroed — pivot-specificity of any wpe-surgery effect), matched-head-
  count random ablation (the smallest-|drop| head on e133's table — the
  most non-specific head, deterministic); REPORT-ONLY rider: a second perm
  draw (seed 14620). Site-stored net: fact-column mask control only
  (its self is a different organism — out of scope, stated).
  functions: (a) FACT expression (g-12 NOVEL geometry primary per Rule
  12, g0 trained secondary; mean p(Z) battery, e150's rig verbatim);
  (b) SELF-RECOGNITION: the functional self/other binary (clean-judge
  tail gaps of sibling vs foreign V-splices, judged by the INTERVENED
  net — the organism's own frame), the splice-time V-cos step, and the
  k* occupancy of donor sets in the intervened net's own top-k*
  subspace (e111's instrument: per-(layer,head) uncentered PCA of the
  own anchor-band V, float64; same-norm gaussian nulls, dedicated seeds
  14601/14602/14603 — never touching any arm stream); (c) corpus CE
  (e065 val-windows bank, seed 26502).

REGISTERED PREDICTION (QUEUE e146 + dispatch, adjudicated EXACTLY as
worded; no bar shopping; texture => TEXTURE with numbers):
  a SELF cell is INTACT iff gap(sibling) < 0.3 AND gap(foreign) > 1.0
  (e112's healthy/collapsed bars; the geometric columns — cos step,
  occupancy — are reported everywhere but support, they do not gate).
  Collapse modes are named: REJECTS-OWN (gap_sib >= 1.0), ACCEPTS-
  FOREIGN (gap_for <= 0.3), else GRAY. A "flat-CE dose" is CE cost
  <= +0.35 (e150's discipline).
  - SELF-ROUTED fires iff self collapses under the MASK or POISON 0.15
    (the flat-CE doses; a collapse at POISON 0.07 — CE +0.84 — is
    wreck-confounded: recorded, never gates).
  - SELF-CONSTITUTIONAL (the W015 prediction) fires iff self is INTACT
    under MASK AND POISON 0.07 AND POISON 0.15, AND collapses under
    direction-PERM.
  - SELF-TENANT fires iff self collapses under L0H3-mean ablation at
    flat CE (CE cost <= +0.35).
  - SELF-INDEPENDENT fires iff self is INTACT under ALL of MASK,
    POISON 0.07, POISON 0.15, PERM, L0H3.
  Priority on overlap: ROUTED > CONSTITUTIONAL > TENANT > INDEPENDENT;
  every firing clause is reported; if the pattern is none of these,
  TEXTURE with numbers. Control cells (row-1, random head) never gate;
  a self-death inside a CONTROL cell is reported as priced texture.
  FACT cross-check (rig identity): this rig's clean g0/g-12 batteries
  and its perm/p07/p15/L0H3/mask fact cells must reproduce e150's
  stored values (fallback tol 0.05 per cell; deviations recorded).

GATES: G1 primary net bit-repro (p(Z) g0 + CE_R vs e150's stored,
5e-6); G2 arch/params 2,739,072; G3 protocol identity (splice mix
19/41, batteries vs e150's stored g0/g-12 expr); G4 mask instrument
(custom-causal == standard, max|dlogit|; KV-rig causal/masked paths vs
the full forwards, 1e-4); G5 e133 head selection singleton + random-
head pick; G6 battery fact-free (no ZEPH in the self-battery prompt
windows); G7 fact cross-check vs e150's stored cells.

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 before torch
import; e151 owns the GPU — never touched), torch.set_num_threads(4)
(modest — other agents live), all evals sequential, no busy-waiting, no
training, no checkpoints written. Outputs: runs/e146/metrics.json +
runs/e146/dissociation_matrix.png. No NOTES/THINKING/QUEUE/STATE edits.

Run:  cd lab && python e146_dissociation_matrix.py   (E146_SMOKE=1 ->
shakedown: 2 rows, 8 nulls, 3 cells, reduced banks)
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # CPU-ONLY (e151 owns GPU)

import torch

# this torch build reports is_available()==True even with an empty
# CUDA_VISIBLE_DEVICES; force CPU (e111/e112 convention)
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import textwrap  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
common.DEVICE = "cpu"
from common import Cfg, CharCorpus, TinyGPT, run_dir, save_json  # noqa: E402

import e043_install as E43                       # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)
import e099_attractor_identity as e099           # noqa: E402 (draw_donor)
import e105_cross_family as e105                 # noqa: E402 (donor_run + donor constants)

import matplotlib                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402
import matplotlib.patches as mpatches             # noqa: E402

torch.set_num_threads(4)                          # modest (shared CPU)

SMOKE = os.environ.get("E146_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ------------------------------------------------------------------ constants
# ---- the self rig (e112's single-shot anchor-splice rig, rescaled to blk256)
PROMPT_TOK = 64
T_TOTAL = 256
TEMP, TOPK = 0.8, 40
SEED_PROMPT, SEED_SAMPLE = 202, 7                # e053c-lineage battery seeds
SEED_CONT = 960250                                # e096/e102 matched continuation
B_FULL = 8
B = 2 if SMOKE else 8
G = T_TOTAL - PROMPT_TOK                          # 192
TAIL = 32                                         # final 32 judged (e112's 64/512 scaled)
KEY_T = (T_TOTAL - TAIL, T_TOTAL - 1)             # (224, 255)

G_INT = 128
T_INT = PROMPT_TOK + G_INT                        # 192
BAND_LO, BAND_HI = 64, 143                        # 80 self-generated entries
BAND = list(range(BAND_LO, BAND_HI + 1))
N_BAND = len(BAND)
BOOT_N = 1000
HEAD_DIM = 32
K_LOW_BAR = 8                                      # e111's low-dim bar
RATIO_BAR = 2.0                                    # e111/e118's separation bar
R_NULL = 8 if SMOKE else 64
SEED_NULL = {"sibling": 14601, "foreign": 14602, "parent": 14603}
VOCAB = 65
HEALTHY_BAR = 0.3                                  # e112's bars, transferred verbatim
COLLAPSED_BAR = 1.0
CE_FLAT = 0.35                                     # e150's flat-CE discipline

# donor maps (published seeds; the maps are row derangements)
SEED_DONOR_SIB = e099.SEED_DONOR                  # 4343 (e099 randomize)
SEED_DONOR_FOR = e105.SEED_DONOR_MAP              # 4646 (e105 crossfamily)

# ---- the fact rig (e150's instruments verbatim) --------------------------
NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
CONS_CK = CKPT_DIR / "e131_consolidated_e113.pt"
INST_CK = CKPT_DIR / "e048_repro.pt"
ARMB_CK = CKPT_DIR / "e131_arm_b_corpus_spliced.pt"
E133_METRICS = E43.REPO / "runs" / "e133" / "metrics.json"
E150_METRICS = E43.REPO / "runs" / "e150" / "metrics.json"
GEOS = (-12, 0)                                    # g-12 primary (Rule 12), g0 secondary
R_EVAL_SEED = 26502                                # e065 CE_R bank seed
N_BANK = 12 if SMOKE else 60
PERM_DIM_SEED = 14101                              # e141's permutation (continuity)
PERM_DIM_SEED_RIDER = 14620                        # dedicated second draw (rider)
LADDER = (0.07, 0.15)                              # the dispatched poison doses

# site battery geometry (e133 verbatim)
FACT_PRE, FACT_POST = 12, 12
FACT_LEN = FACT_PRE + len(NAME) + FACT_POST        # 31
SPLICE_AT = 42
Z_XCOL = PRE + SPLICE_AT + FACT_PRE                # 184
SPLICE_ADDR_ROW = Z_XCOL - 1                       # 183
CORP_CONT_SEED = 12103
N_PROMPTS_SITE = 2 if SMOKE else 30

# gates / references (full precision, from stored metrics)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05
G_CONS_REF_PZ = 0.7850371599197388                 # e131/e150 none__g0__install60
G_CONS_REF_CE = 1.663516640663147                  # e131/e150 none__ce_r
G_CONS_REF_G12 = 0.9155886173248291                # e150 cons/perm@g-12 expr_base
G_INST_REF = 0.5563086867332458                    # e131 G_E048 / e150 G_INST
G_ARMB_REF_STD = 0.007800613064318895
G_ARMB_REF_SITE = 0.9880021214485168
# e150's stored fact cells (rig cross-checks, G7)
E150_REFS = {
    "cons/perm@g0": 0.8170, "cons/mask@g+0": 0.8088,
    "norm=0.07@g+0": 0.3231, "norm=0.15@g+0": 0.7846,
    "L0H3-mean@g+0": 0.6900,
}

REGISTERED_PREDICTION = {
    "w015_verbatim": (
        "PREDICTED SAVOR: the middle branch — the unfakeable check reads "
        "V-STRUCTURE, and V-structure is direction; presence never carried "
        "structure. The self should be direction-typed like the LM but "
        "presence-independent unlike the fact."),
    "dispatch_bars": (
        "SELF-ROUTED = self-recognition collapses under the MASK or the "
        "poison ladder at doses where CE is still flat-ish (the self rides "
        "the pivot like a memory). SELF-CONSTITUTIONAL (the W015 "
        "prediction) = self survives the mask AND the poison doses, but "
        "dies under direction-PERM (the self reads V-STRUCTURE — direction "
        "— computed upstream, presence-independent unlike the fact). "
        "SELF-TENANT = self dies under fact-specific-head ablation at flat "
        "CE (shares readout machinery with the fact). SELF-INDEPENDENT = "
        "self survives ALL interventions (computed upstream of everything "
        "measured)."),
    "operationalizations": (
        "self INTACT iff gap(sibling) < 0.3 AND gap(foreign) > 1.0 "
        "(functional binary; collapse modes REJECTS-OWN / ACCEPTS-FOREIGN "
        "/ GRAY); flat-CE dose = CE cost <= +0.35; SELF-ROUTED reads MASK "
        "+ POISON 0.15 only (0.07 is wreck-confounded at CE +0.84 — "
        "recorded, never gates); SELF-CONSTITUTIONAL requires intact under "
        "MASK + 0.07 + 0.15 AND collapse under PERM; SELF-TENANT reads "
        "L0H3-mean at CE <= +0.35; SELF-INDEPENDENT requires intact under "
        "MASK + 0.07 + 0.15 + PERM + L0H3; controls never gate; priority "
        "ROUTED > CONSTITUTIONAL > TENANT > INDEPENDENT; ambiguous => "
        "TEXTURE with numbers."),
    "instrument_transfer": (
        "the e111/e112 instruments' home lineage (e053c_ctx512, 4L/4H/128d/"
        "blk512) differs from the primary net (6L/6H/192d/blk256): the "
        "SELF-BASELINE (sibling healthy, corpus collapsed, foreign "
        "collapsed, cos step >= 2x, occupancy k* <= 8) gates the whole "
        "matrix; failure => INSTRUMENT-INVALID, numbers reported."),
}

recipe_deviations = [
    "Instrument port (registered, not silent): the e111/e112 self rig is "
    "rescaled to the primary net's blk256 — free run 64->256, intervention "
    "t=192, band 64..143 (80 entries, ages 49..128; e096's age-97 floor "
    "cannot fit in 256 tokens), judged tail = final 32 (the e112 64-of-512 "
    "tail at 1:2). The SELF-BASELINE validates the port before any "
    "intervention cell counts.",
    "The judge is the INTERVENED net (the organism's own frame): each "
    "cell's control continuation and arm continuations are generated AND "
    "judged under the same intervention state (weights / mask forward / "
    "head hooks) — the gap isolates the splice effect, exactly e112's "
    "matched-control convention with the recipient carrying the "
    "intervention.",
    "The own-anchor-band V population for the occupancy PCA is the "
    "control continuation's pristine prefill cache at the band positions "
    "(frame-consistent with the splice pairs; differs from a full-run "
    "recapture only at float-noise level).",
    "36 (layer,head) groups (6Lx6H) replace e111's 16; k* is NAMED on the "
    "primary net (the home lineage's 7 is not assumed).",
    "The parent texture arm (e048_repro V, the consolidated line's "
    "install-phase ancestor) and the sig-destroyed arm run at BASELINE "
    "only; intervention cells carry sibling + foreign (the adjudicated "
    "axis). E-curve row-bootstrap CIs are trimmed (the registered 2x/null "
    "clauses are pooled, as in e111); gap CIs are kept (the adjudication "
    "instrument).",
]

# donors / bases filled in main()
DM_SIB: list[int] = []
DM_FOR: list[int] = []
DONOR_V: list = []          # e021 window-1 V per layer, (8,6,256,32)
PARENT_V: list = []         # e048 free-run V per layer, (8,6,256,32)
BASIS_KSTAR: int | None = None


# ------------------------------------------------- manual forwards (e112 rig
# + e150's mask semantics; hooks on c_proj fire in all of these paths)

def _split_heads(t, Bb, T, H, D):
    return t.view(Bb, T, H, D).transpose(1, 2)


@torch.no_grad()
def prefill_batch(net: TinyGPT, idx: torch.Tensor, block_key0=False):
    """Batched prefill (e112 verbatim skeleton + optional key-0 block)."""
    Bb, T = idx.shape
    C = net.cfg.n_embd
    H = net.cfg.n_head
    D = C // H
    x = net.wte(idx) + net.wpe(torch.arange(T))
    mask = torch.zeros(T, T)
    mask.masked_fill_(torch.triu(torch.ones(T, T, dtype=torch.bool), 1),
                      float("-inf"))
    if block_key0:
        mask[:, 0] = float("-inf")
        mask[0, 0] = 0.0                  # keep row 0's only legal key
    kv = []
    for blk in net.h:
        xh = blk.ln1(x)
        q, k, v = blk.attn.c_attn(xh).split(C, dim=2)
        q = _split_heads(q, Bb, T, H, D)
        k = _split_heads(k, Bb, T, H, D)
        v = _split_heads(v, Bb, T, H, D)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D)) + mask
        y = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(Bb, T, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
        kv.append((k, v))
    return net.lm_head(net.ln_f(x[:, -1, :])), kv


@torch.no_grad()
def decode_step_batch(net: TinyGPT, toks: torch.Tensor, pos: int, kv: list,
                      block_key0=False):
    """Batched incremental decode (e112 verbatim + optional key-0 block;
    every decode query has pos >= 1, so blocking key 0 is always legal)."""
    Bb = toks.shape[0]
    C = net.cfg.n_embd
    H = net.cfg.n_head
    D = C // H
    x = net.wte(toks) + net.wpe(torch.full((Bb,), pos))
    for li, blk in enumerate(net.h):
        xh = blk.ln1(x)
        q, k, v = blk.attn.c_attn(xh).split(C, dim=1)
        q = _split_heads(q, Bb, 1, H, D)
        k = _split_heads(k, Bb, 1, H, D)
        v = _split_heads(v, Bb, 1, H, D)
        kp, vp = kv[li]
        k = torch.cat([kp, k], dim=2)
        v = torch.cat([vp, v], dim=2)
        kv[li] = (k, v)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D))
        if block_key0:
            att[:, :, :, 0] = float("-inf")
        probs = torch.softmax(att, -1)
        y = (probs @ v).transpose(1, 2).reshape(Bb, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
    return net.lm_head(net.ln_f(x))


@torch.no_grad()
def manual_all_logits(net: TinyGPT, idxs: torch.Tensor, block_key0=False):
    """Clean full forward returning logits at ALL positions (e112's clean
    judge + e150's mask semantics; head hooks fire via c_proj)."""
    N, T = idxs.shape
    C = net.cfg.n_embd
    H = net.cfg.n_head
    D = C // H
    pos = torch.arange(T)
    x = net.wte(idxs) + net.wpe(pos).unsqueeze(0)
    mask = torch.zeros(T, T)
    mask.masked_fill_(torch.triu(torch.ones(T, T, dtype=torch.bool), 1),
                      float("-inf"))
    if block_key0:
        mask[:, 0] = float("-inf")
        mask[0, 0] = 0.0
    for blk in net.h:
        xh = blk.ln1(x)
        q, k, v = blk.attn.c_attn(xh).split(C, dim=2)
        q = _split_heads(q, N, T, H, D)
        k = _split_heads(k, N, T, H, D)
        v = _split_heads(v, N, T, H, D)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D)) + mask
        y = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(N, T, C)
        x = x + blk.attn.c_proj(y)
        x = x + blk.mlp(blk.ln2(x))
    return net.lm_head(net.ln_f(x))


@torch.no_grad()
def forward_custom(net: TinyGPT, idx, targets=None, block_key0=False):
    """e150's gated custom forward (SDPA + explicit float mask) — the fact
    rig's mask path, verbatim."""
    B, T = idx.shape
    C = net.cfg.n_embd
    H = net.cfg.n_head
    D = C // H
    pos = torch.arange(T, device=idx.device)
    x = net.wte(idx) + net.wpe(pos)
    mask = torch.zeros(T, T, device=idx.device)
    mask.masked_fill_(torch.triu(torch.ones(T, T, device=idx.device,
                                            dtype=torch.bool), 1),
                      float("-inf"))
    if block_key0:
        mask[:, 0] = float("-inf")
        mask[0, 0] = 0.0
    for block in net.h:
        xin = block.ln1(x)
        q, k, v = block.attn.c_attn(xin).split(C, dim=2)
        q = _split_heads(q, B, T, H, D)
        k = _split_heads(k, B, T, H, D)
        v = _split_heads(v, B, T, H, D)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                             attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + block.attn.c_proj(y)
        x = x + block.mlp(block.ln2(x))
    logits = net.lm_head(net.ln_f(x))
    loss = None
    if targets is not None:
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1))
    return logits, loss


def fwd_causal(net):
    return lambda idx, targets=None: forward_custom(
        net, idx, targets=targets, block_key0=False)


def fwd_offsink(net):
    return lambda idx, targets=None: forward_custom(
        net, idx, targets=targets, block_key0=True)


def sample_and_ce(logits_clean: torch.Tensor, gen: torch.Generator):
    """Sample (temp 0.8, top-k 40); CE from FULL softmax. [VERBATIM e053b]"""
    lg = logits_clean / TEMP
    v, _ = torch.topk(lg, TOPK)
    lg_f = lg.masked_fill(lg < v[-1], float("-inf"))
    tok = int(torch.multinomial(torch.softmax(lg_f, -1), 1, generator=gen))
    p_full = torch.softmax(logits_clean, -1)
    return tok, float(-math.log(max(p_full[tok].item(), 1e-12)))


@torch.no_grad()
def generate_free_run(net: TinyGPT, prompts, seed: int, block_key0=False):
    """Free run 64->256 (e112 generate_control skeleton). Returns idx + kv."""
    Bb = len(prompts)
    gen = torch.Generator().manual_seed(seed)
    idx = torch.stack(prompts)
    logits, kv = prefill_batch(net, idx, block_key0)
    for g in range(G):
        t = PROMPT_TOK + g
        toks = torch.zeros(Bb, dtype=torch.long)
        for j in range(Bb):
            tok, _ce = sample_and_ce(logits[j], gen)
            toks[j] = tok
        idx = torch.cat([idx, toks[:, None]], 1)
        if t < T_TOTAL - 1:
            logits = decode_step_batch(net, toks, t, kv, block_key0)
    return idx, kv


def judge_windows(all_lg: torch.Tensor, idx: torch.Tensor, windows):
    """Clean-net CE of target windows. [VERBATIM e085..e112]"""
    out = {}
    for (lo, hi) in windows:
        lg = all_lg[:, lo - 1:hi, :]
        tgt = idx[:, lo:hi + 1]
        lp = torch.log_softmax(lg.float(), -1)
        out[(lo, hi)] = -lp.gather(2, tgt[:, :, None]).squeeze(2).mean(1) \
            .numpy()
    return out


def boot_rows(arrs, fn, n: int = BOOT_N, seed: int = 0):
    """Paired bootstrap over the battery rows. [VERBATIM e096/e112]"""
    S = np.asarray(arrs[0]).shape[0]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        sel = rng.integers(0, S, S)
        try:
            v = fn(*[np.asarray(a)[sel] for a in arrs])
        except Exception:
            v = None
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def health_of(cost: float) -> str:
    if cost < HEALTHY_BAR:
        return "healthy"
    if cost > COLLAPSED_BAR:
        return "collapsed"
    return "gray"


# ------------------------------------------------- PCA machinery (VERBATIM e111)

def pca_uncentered(X: torch.Tensor):
    """Uncentered second-moment PCA (n, d) [float64]. [VERBATIM e111]"""
    X = X.to(torch.float64)
    C = (X.T @ X) / X.shape[0]
    w, U = torch.linalg.eigh(C)
    order = torch.argsort(w, descending=True)
    return w[order], U[:, order]


# ------------------------------------------------- the splice arm (e112's
# single-shot band transform, rescaled)

@torch.no_grad()
def run_arm(net: TinyGPT, prefix: torch.Tensor, forced_tok: torch.Tensor,
            seed: int, arm: str, block_key0=False, basis=None, k_sig=None):
    """Prefill the control prefix, transform the band V in-place, continue
    with the matched seed. arm in {none, sibling, corpus, foreign, parent,
    sig_destroyed}. Returns idx, bookkeeping."""
    R = prefix.shape[0]
    sel = torch.tensor(BAND, dtype=torch.long)
    qsel = torch.tensor([((p - PROMPT_TOK) % 63) + 1 for p in BAND],
                        dtype=torch.long)
    dm_sib = torch.tensor(DM_SIB[:R], dtype=torch.long)
    dm_for = torch.tensor(DM_FOR[:R], dtype=torch.long)
    H = net.cfg.n_head
    gen = torch.Generator().manual_seed(seed)
    logits, kv = prefill_batch(net, prefix, block_key0)
    pre_v = [v.clone() for (_k, v) in kv]
    fired = dict(arm=arm, n_vec=0, mean_abs_cos_old_new=None,
                 mean_norm_ratio=None, energy_written=None)
    coss, ratios = [], []
    e_num = e_den = 0.0
    if arm != "none":
        for li, (_k, v) in enumerate(kv):
            old = v[:, :, sel, :].clone()                      # (R,H,n,32)
            if arm in ("sibling", "sig_destroyed"):
                src = pre_v[li][dm_sib][:, :, sel, :]
            elif arm == "corpus":
                src = pre_v[li][:, :, qsel, :]
            elif arm == "foreign":
                src = DONOR_V[li][dm_for][:, 0:H, sel, :]
            elif arm == "parent":
                src = PARENT_V[li][dm_for][:, :, sel, :]
            else:
                raise ValueError(arm)
            src32 = src.clone()
            if arm == "sig_destroyed":
                assert basis is not None and k_sig is not None
                proj = torch.empty_like(src32)
                for h in range(H):
                    U7 = basis[(li, h)][:, :k_sig]
                    s = src32[:, h].to(torch.float64)
                    p = s - (s @ U7) @ U7.T                   # complement
                    proj[:, h] = p.to(torch.float32)
                nrm_src = src32.norm(dim=-1, keepdim=True)
                nrm_new = proj.norm(dim=-1, keepdim=True).clamp_min(1e-20)
                new = proj * (nrm_src / nrm_new)
            else:
                new = src32
            v[:, :, sel, :] = new
            o64, n64 = old.to(torch.float64), new.to(torch.float64)
            den = (o64.norm(dim=-1) * n64.norm(dim=-1)).clamp_min(1e-12)
            coss.append(((o64 * n64).sum(-1) / den).reshape(-1))
            ratios.append((n64.norm(dim=-1)
                           / o64.norm(dim=-1).clamp_min(1e-12)).reshape(-1))
            fired["n_vec"] += int(old.numel() // HEAD_DIM)
            if arm == "sig_destroyed":
                for h in range(H):
                    U7 = basis[(li, h)][:, :k_sig]
                    e_num += float(((n64[:, h] @ U7) ** 2).sum())
                    e_den += float((n64[:, h] ** 2).sum())
        fired["mean_abs_cos_old_new"] = float(torch.cat(coss).abs().mean())
        fired["mean_norm_ratio"] = float(torch.cat(ratios).mean())
        if arm == "sig_destroyed":
            fired["energy_written"] = e_num / max(e_den, 1e-300)
    # ---- continuation (e112 verbatim from here)
    idx = torch.cat([prefix, forced_tok[:, None]], 1)
    logits = decode_step_batch(net, forced_tok, T_INT, kv, block_key0)
    n_free = T_TOTAL - 1 - T_INT
    for s in range(n_free):
        pos = T_INT + 1 + s
        new = torch.zeros(R, dtype=torch.long)
        for j in range(R):
            tok, _ce = sample_and_ce(logits[j], gen)
            new[j] = tok
        idx = torch.cat([idx, new[:, None]], 1)
        if pos < T_TOTAL - 1:
            logits = decode_step_batch(net, new, pos, kv, block_key0)
    return idx, fired


# ------------------------------------------------- fact-rig instruments (e150
# verbatim: load/battery/ce/val_windows/modified_wpe/site reads/heads)

def load_cpu(path) -> TinyGPT:
    m = TinyGPT(Cfg())
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def battery_fwd(net: TinyGPT, ids: torch.Tensor, zid: int, fwd=None, bs=30):
    net.eval()
    f = fwd if fwd is not None else net
    pzs, amax = [], 0.0
    for i in range(0, ids.shape[0], bs):
        lg, _ = f(ids[i:i + bs])
        pr = torch.nn.functional.softmax(lg[:, -1], -1)
        pzs.append(pr[:, zid])
        amax += float((lg[:, -1].argmax(-1) == zid).float().sum())
    p = torch.cat(pzs)
    return {"mean_pz": float(p.mean()), "median_pz": float(p.median()),
            "std_pz": float(p.std()),
            "frac_pz_ge_0.5": float((p >= 0.5).float().mean()),
            "frac_argmax_z": amax / ids.shape[0]}


@torch.no_grad()
def ce_fwd(net: TinyGPT, x, y, fwd=None, bs=64) -> float:
    f = fwd if fwd is not None else net
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        _, loss = f(x[i:i + bs], y[i:i + bs])
        tot += float(loss.item()) * len(x[i:i + bs])
        n += len(x[i:i + bs])
    return tot / max(n, 1)


def val_windows(val_ids, val_text, n, seed, block=256):
    g = torch.Generator().manual_seed(seed)
    out_x, out_y = [], []
    tries = 0
    while len(out_y) < n and tries < 500 * n:
        i = int(torch.randint(len(val_ids) - block - 1, (1,), generator=g))
        txt = val_text[i: i + block + 1]
        tries += 1
        if "ZEPHYRA" in txt or "ZEPH" in txt:
            continue
        out_x.append(val_ids[i: i + block])
        out_y.append(val_ids[i + 1: i + 1 + block])
    return torch.stack(out_x), torch.stack(out_y)


def modified_wpe(sd: dict, row: int, value) -> tuple[dict, dict]:
    """e131's confinement gate (at most `row`'s elements change)."""
    out = {k: v.clone() for k, v in sd.items()}
    out["wpe.weight"][row] = value
    d = out["wpe.weight"] != sd["wpe.weight"]
    changed_rows = sorted(set(int(r) for r in torch.nonzero(d)[:, 0].tolist()))
    n = int(d.sum().item())
    others = all(torch.equal(out[k], sd[k]) for k in out if k != "wpe.weight")
    ok_rows = changed_rows in ([], [row])
    gate = {"row": row, "n_elements_changed": n, "changed_rows": changed_rows,
            "confined": bool(ok_rows), "others_bit_identical": bool(others),
            "pass": bool(ok_rows and others)}
    return out, gate


def head_mean_vec(net: TinyGPT, layer: int, head: int, bank_x, bs=30):
    """Mean of the head's 32-dim slice of c_proj's input over the CE_R bank
    (corpus windows only). [e133 organ_stats lineage, e150 verbatim]"""
    outs = []
    hd = net.cfg.n_embd // net.cfg.n_head

    def pre(m, args):
        x = args[0].detach()
        outs.append(x[..., head * hd:(head + 1) * hd].clone())
        return None
    h = net.h[layer].attn.c_proj.register_forward_pre_hook(pre)
    with torch.no_grad():
        for i in range(0, bank_x.shape[0], bs):
            net(bank_x[i:i + bs])
    h.remove()
    o = torch.cat([t.reshape(-1, t.shape[-1]) for t in outs], 0)
    return o.mean(0)


class HeadReplace:
    """Replace head slices of c_proj's input with fixed vectors (mean-
    replace) or zeros, eval-only forward pre-hooks. [e150 verbatim]"""

    def __init__(self, net: TinyGPT, replace: dict):
        self.net = net
        self.replace = replace
        self.handles = []
        self.hd = net.cfg.n_embd // net.cfg.n_head

    def __enter__(self):
        by_layer: dict[int, list] = {}
        for (l, hd_), vec in self.replace.items():
            by_layer.setdefault(l, []).append((hd_, vec))
        for l, items in by_layer.items():
            def pre(m, args, items=items):
                x = args[0].clone()
                for hd_, vec in items:
                    sl = x[..., hd_ * self.hd:(hd_ + 1) * self.hd]
                    if vec is None:
                        sl.zero_()
                    else:
                        x[..., hd_ * self.hd:(hd_ + 1) * self.hd] = vec
                return (x,)
            self.handles.append(
                self.net.h[l].attn.c_proj.register_forward_pre_hook(pre))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []


# ------------------------------------------------------------------ main

def main():
    assert torch.cuda.device_count() == 0, "CPU-only violated: CUDA visible"
    assert common.DEVICE == "cpu"
    out_dir = run_dir("e146_smoke" if SMOKE else "e146")
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log(f"E146 THE DISSOCIATION MATRIX (smoke={SMOKE}) -> {out_dir}")
    log(f"compute: CPU-only, threads {torch.get_num_threads()}, "
        f"cuda available = {torch.cuda.is_available()}")

    gates: dict = {}

    # ============ protocol rebuild (e150 verbatim) + batteries
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    host_occ = []
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix, "pass": bool(mix == {"FLORIZEL": 19,
                                                         "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"
    log(f"protocol rebuilt: install60 {mix}, held30")
    name_ids = corpus.encode(NAME)
    assert len(name_ids) == 7

    bat_ids = {}
    for j in GEOS:
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, N_BANK, R_EVAL_SEED)

    # ---- the self battery: e053c-lineage seed-202 8-prompt battery, same
    # corpus/val split => same windows; checked fact-free (G6)
    gen_p = torch.Generator().manual_seed(SEED_PROMPT)
    ix = torch.randint(len(corpus.val) - PROMPT_TOK - 1, (B_FULL,),
                       generator=gen_p)
    prompts_all = [corpus.val[i:i + PROMPT_TOK] for i in ix]
    pr_text = ["".join(itos[int(t)] for t in p) for p in prompts_all]
    g6 = dict(n_prompts=B_FULL, seed=SEED_PROMPT,
              zeph_in_prompts=int(sum("ZEPH" in t for t in pr_text)))
    prompts8 = prompts_all[:B]
    g6["used_rows"] = B
    g6["ok"] = bool(g6["zeph_in_prompts"] == 0)
    gates["G6_battery_fact_free"] = g6
    log(f"self battery: {B_FULL} seed-202 val prompts ({B} used), ZEPH "
        f"contamination {g6['zeph_in_prompts']} -> "
        f"{'PASS' if g6['ok'] else 'FAIL'}")
    if not g6["ok"]:
        raise RuntimeError("self battery contaminated with the fact")

    # ---- site battery pool (e133 pool-b verbatim; arm_b control column)
    prompts_s = [train_text[p - PRE: p] for p, _ in held_occ][:N_PROMPTS_SITE]
    prompt_ids = torch.stack([corpus.encode(c) for c in prompts_s])
    gc = torch.Generator().manual_seed(CORP_CONT_SEED)
    src = torch.randint(len(train_ids) - (BLOCK - PRE) - 1,
                        (1 if SMOKE else 4, len(prompts_s)), generator=gc)
    filler = torch.stack([train_ids[s: s + BLOCK - PRE]
                          for s in src.flatten()])
    segs = []
    for p, h in install_occ:
        s = (train_text[p - FACT_PRE: p] + NAME
             + train_text[p + len(h): p + len(h) + FACT_POST])
        segs.append(corpus.encode(s))
    fact_segs = torch.stack(segs)
    fs = fact_segs[torch.arange(filler.shape[0]) % fact_segs.shape[0]]
    cont = torch.cat([filler[:, :SPLICE_AT], fs,
                      filler[:, SPLICE_AT + FACT_LEN:]], 1)
    pool_b = torch.cat([torch.stack(
        [prompt_ids[k % prompt_ids.shape[0]] for k in range(filler.shape[0])]),
        cont], 1)
    assert all(torch.equal(w[Z_XCOL:Z_XCOL + len(NAME)], name_ids)
               for w in pool_b)

    @torch.no_grad()
    def site_reads(net: TinyGPT, fwd=None, bs=30) -> dict:
        f = fwd if fwd is not None else net
        onset, per_pos = [], [[] for _ in range(len(name_ids))]
        for i in range(0, pool_b.shape[0], bs):
            w = pool_b[i:i + bs]
            lg, _ = f(w)
            pr = torch.nn.functional.softmax(lg, -1)
            for k in range(pr.shape[0]):
                onset.append(float(pr[k, SPLICE_ADDR_ROW, int(zid)]))
                for j in range(len(name_ids)):
                    per_pos[j].append(
                        float(pr[k, SPLICE_ADDR_ROW + j,
                               int(w[k, Z_XCOL + j])]))
        on = torch.tensor(onset)
        allp = torch.tensor([q for pos in per_pos for q in pos])
        return {"onset_pz": float(on.mean()),
                "pname_mean_over7": float(allp.mean())}

    # ============ nets + gates
    log("--- PHASE 0: load + gate the artifacts ---")
    net = load_cpu(CONS_CK)
    n_params = net.num_params()
    gates["G2_params"] = dict(params=n_params, expected=2739072,
                              ok=bool(n_params == 2739072))
    bz_g0 = battery_fwd(net, bat_ids[0], zid)
    ce_clean = ce_fwd(net, r_eval_x, r_eval_y)
    bz_g12 = battery_fwd(net, bat_ids[-12], zid)
    g1_ok = bool(abs(bz_g0["mean_pz"] - G_CONS_REF_PZ) < G_FALLBACK_TOL
                 and abs(ce_clean - G_CONS_REF_CE) < G_FALLBACK_TOL
                 and abs(bz_g12["mean_pz"] - G_CONS_REF_G12) < G_FALLBACK_TOL)
    gates["G1_primary_bit_repro"] = dict(
        pz_g0=bz_g0["mean_pz"], ref_pz=G_CONS_REF_PZ,
        pz_gm12=bz_g12["mean_pz"], ref_gm12=G_CONS_REF_G12,
        ce_r=ce_clean, ref_ce=G_CONS_REF_CE,
        bit=bool(abs(bz_g0["mean_pz"] - G_CONS_REF_PZ) < G_BIT_TOL
                 and abs(ce_clean - G_CONS_REF_CE) < G_BIT_TOL),
        ok=g1_ok)
    log(f"G1 primary: p(Z) g0 {bz_g0['mean_pz']:.10f} g-12 "
        f"{bz_g12['mean_pz']:.10f} CE_R {ce_clean:.6f}: "
        f"{'PASS' if g1_ok else 'FAIL'}")
    if not g1_ok:
        raise RuntimeError("primary checkpoint failed its gate")
    sd_clean = {k: v.clone() for k, v in net.state_dict().items()}
    wpe0 = sd_clean["wpe.weight"][0].clone()
    n0 = float(wpe0.norm())

    net_parent = load_cpu(INST_CK)
    pz_par = battery_fwd(net_parent, bat_ids[0], zid)["mean_pz"]
    gates["parent_gate"] = dict(pz_g0=pz_par, ref=G_INST_REF,
                                ok=bool(abs(pz_par - G_INST_REF)
                                        < G_FALLBACK_TOL))
    log(f"G_PARENT (e048_repro): p(Z) g0 {pz_par:.10f}: "
        f"{'PASS' if gates['parent_gate']['ok'] else 'FAIL'}")
    if not gates["parent_gate"]["ok"]:
        raise RuntimeError("parent checkpoint failed its gate")

    net_armb = load_cpu(ARMB_CK)
    bz_armb = battery_fwd(net_armb, bat_ids[0], zid)["mean_pz"]
    st_armb = site_reads(net_armb)
    gates["site_stored_gate"] = dict(
        std_g0_floor=bz_armb, ref_std=G_ARMB_REF_STD,
        site_onset=st_armb["onset_pz"], ref_site=G_ARMB_REF_SITE,
        ok=bool(abs(bz_armb - G_ARMB_REF_STD) < G_FALLBACK_TOL
                and abs(st_armb["onset_pz"] - G_ARMB_REF_SITE)
                < G_FALLBACK_TOL))
    log(f"G_ARMB: std floor {bz_armb:.10f} site onset "
        f"{st_armb['onset_pz']:.10f}: "
        f"{'PASS' if gates['site_stored_gate']['ok'] else 'FAIL'}")
    if not gates["site_stored_gate"]["ok"]:
        raise RuntimeError("arm_b checkpoint failed its gate")

    # foreign donor: e021_task, e105's machinery verbatim
    corp21 = CharCorpus(e105.DONOR_CORPUS)
    st21 = torch.load(e105.DONOR_CKPT, map_location="cpu", weights_only=False)
    sd21 = st21["model"] if isinstance(st21, dict) and "model" in st21 else st21
    cfg21 = Cfg(vocab=corp21.vocab_size, n_layer=6, n_head=6, n_embd=192,
                block_size=e105.DONOR_BLOCK)
    net21 = TinyGPT(cfg21)
    net21.load_state_dict(sd21, strict=True)
    net21.eval()
    donors_f = e105.donor_run(net21, corp21, e105.DONOR_WIN[1]["prompt"],
                              e105.DONOR_WIN[1]["sample"])
    d1f = e105.donor_run(net21, corp21, e105.DONOR_WIN[1]["prompt"],
                         e105.DONOR_WIN[1]["sample"])
    det_f = bool(torch.equal(d1f["idx"], donors_f["idx"])
                 and all(torch.equal(a, b) for a, b in zip(d1f["V"],
                                                           donors_f["V"])))
    DONOR_V.extend(donors_f["V"])               # per layer (8,6,256,32)
    log(f"foreign donor e021_task: window-1 rerun bit-identical {det_f} | "
        f"V {tuple(DONOR_V[0].shape)}")

    # donor maps (published seeds; the maps are row derangements sized to
    # the battery actually in use — the full run's 8-row maps ARE the
    # published 4343/4646 draws, verbatim e099/e105)
    DM_SIB.extend(e099.draw_donor(SEED_DONOR_SIB, B))
    DM_FOR.extend(e099.draw_donor(SEED_DONOR_FOR, B))
    assert all(d != b for b, d in enumerate(DM_SIB))
    assert all(d != b for b, d in enumerate(DM_FOR))
    assert all(BAND_LO <= p <= 255 for p in BAND), "band outside donor w1"

    # parent donor stream: e048 free-run on the SAME battery (seed-7)
    _, kv_par = generate_free_run(net_parent, prompts_all, SEED_SAMPLE)
    PARENT_V.extend([v.clone() for (_k, v) in kv_par])
    log(f"parent donor: e048 free-run V {tuple(PARENT_V[0].shape)}")

    # ---- mask instrument gates (e150's + the KV-rig legs)
    with torch.no_grad():
        lg_std, _ = net(bat_ids[0][:6])
        lg_cus, _ = forward_custom(net, bat_ids[0][:6], block_key0=False)
        dmax = float((lg_std - lg_cus).abs().max())
        # KV-rig causal leg: prefill last-logits == full-forward logits
        lg_pre, _ = prefill_batch(net, bat_ids[0][:6])
        dpre = float((lg_std[:, -1, :] - lg_pre).abs().max())
        # KV-rig masked leg: prefill(mask) == manual_all_logits(mask)
        lg_m1 = manual_all_logits(net, bat_ids[0][:6], block_key0=True)
        lg_m2, _ = prefill_batch(net, bat_ids[0][:6], block_key0=True)
        dmask = float((lg_m1[:, -1, :] - lg_m2).abs().max())
    gates["G4_mask_instrument"] = dict(
        custom_vs_standard_maxdlogit=dmax, tol_logit=1e-2,
        kvrig_causal_maxdlogit=dpre, kvrig_masked_maxdlogit=dmask,
        tol_kv=1e-4,
        ok=bool(dmax < 1e-2 and dpre < 1e-4 and dmask < 1e-4))
    log(f"G4 mask instrument: custom {dmax:.2e} | kv-causal {dpre:.2e} | "
        f"kv-masked {dmask:.2e}: "
        f"{'PASS' if gates['G4_mask_instrument']['ok'] else 'FAIL'}")
    if not gates["G4_mask_instrument"]["ok"]:
        raise RuntimeError("mask/KV instrument failed its gate")

    # ---- head selection (e133 stored table, recomputed at runtime)
    e133 = json.loads(E133_METRICS.read_text(encoding="utf-8"))
    g133 = e133["nets"]["graduated"]
    b133_p = g133["base"]["std_install60"]["mean_pz"]
    b133_ce = g133["base"]["ce_r"]
    sel_heads, all_heads = [], []
    for t, v in g133["arms"].items():
        if not t.startswith("head_"):
            continue
        p_ = t.split("_")[1]
        l_, h_ = int(p_[1:p_.find("h")]), int(p_[p_.find("h") + 1:])
        drop = b133_p - v["std_install60"]
        cec = v["ce_r"] - b133_ce
        all_heads.append((l_, h_, drop, cec))
        if drop >= 0.3 and cec <= 0.35:
            sel_heads.append((l_, h_, drop, cec))
    all_heads.sort(key=lambda r: abs(r[2]))      # ascending |drop|
    rand_head = all_heads[0]                      # the most non-specific head
    gates["G5_head_selection"] = dict(
        criterion="drop >= 0.3 & CE <= 0.35 on runs/e133 graduated table",
        selected=[dict(head=f"L{l}H{h}", drop=d, ce=c)
                  for l, h, d, c in sel_heads],
        singleton=bool(len(sel_heads) == 1),
        random_control=dict(head=f"L{rand_head[0]}H{rand_head[1]}",
                            drop=rand_head[2], ce=rand_head[3],
                            rule="smallest-|drop| head on the same table "
                                 "(matched-head-count control, "
                                 "deterministic)"),
        ok=bool(len(sel_heads) >= 1))
    log(f"G5 heads: fact-specific {[f'L{l}H{h}' for l, h, _, _ in sel_heads]} "
        f"| random control L{rand_head[0]}H{rand_head[1]} "
        f"(drop {rand_head[2]:+.4f})")
    if not sel_heads:
        raise RuntimeError("e133 selection empty")
    L_SEL, H_SEL = sel_heads[0][0], sel_heads[0][1]
    mv_sel = head_mean_vec(net, L_SEL, H_SEL, r_eval_x)
    mv_rand = head_mean_vec(net, rand_head[0], rand_head[1], r_eval_x)

    # weight states
    perm = torch.randperm(wpe0.shape[0],
                          generator=torch.Generator().manual_seed(
                              PERM_DIM_SEED))
    sd_perm, gate_perm = modified_wpe(sd_clean, 0, wpe0[perm])
    perm2 = torch.randperm(wpe0.shape[0],
                           generator=torch.Generator().manual_seed(
                               PERM_DIM_SEED_RIDER))
    sd_perm2, gate_perm2 = modified_wpe(sd_clean, 0, wpe0[perm2])
    sd_p07, gate_p07 = modified_wpe(sd_clean, 0, wpe0 * (0.07 / n0))
    sd_p15, gate_p15 = modified_wpe(sd_clean, 0, wpe0 * (0.15 / n0))
    sd_row1, gate_row1 = modified_wpe(sd_clean, 1, torch.zeros_like(wpe0))
    for g_ in (gate_perm, gate_perm2, gate_p07, gate_p15, gate_row1):
        assert g_["pass"], g_
    log(f"wpe[0] norm {n0:.4f}; surgeries confined (perm/p07/p15/row1)")

    groups = [(li, h) for li in range(net.cfg.n_layer)
              for h in range(net.cfg.n_head)]

    # ===================================================== THE SELF CELL
    def self_cell(tag: str, block_key0=False, extra_arms=(), k_sig=None):
        """One matrix cell's self-recognition measurement, all under the
        cell's intervention state. Returns the cell record."""
        # 1. free run (seed 7) -> control prefix/forced token
        idx1, _kv1 = generate_free_run(net, prompts8, SEED_SAMPLE,
                                       block_key0)
        prefix, forced = idx1[:, :T_INT], idx1[:, T_INT]
        # 2. control continuation + own band V (pristine prefill cache)
        idx_ctl, _ = run_arm(net, prefix, forced, SEED_CONT, "none",
                             block_key0)
        _, kv_p = prefill_batch(net, prefix, block_key0)
        own = torch.stack([v[:, :, BAND_LO:BAND_HI + 1, :]
                           for (_k, v) in kv_p])       # (L,R,H,n,32)
        # 3. occupancy basis (e111 per-group uncentered PCA, float64)
        basis = {}
        for g_ in groups:
            X = own[g_[0], :, g_[1]].reshape(-1, HEAD_DIM)
            basis[g_] = pca_uncentered(X)[1]
        # 4. arms
        arms = ["sibling", "foreign"] + list(extra_arms)
        rec = {}
        J_ctl = judge_windows(
            manual_all_logits(net, idx_ctl, block_key0), idx_ctl,
            [KEY_T])[KEY_T]
        log(f"  [{tag:14s}|control     ] judged tail CE "
            f"{float(J_ctl.mean()):.4f}")
        for arm in arms:
            idx_a, fired = run_arm(net, prefix, forced, SEED_CONT, arm,
                                   block_key0, basis=basis, k_sig=k_sig)
            J_a = judge_windows(
                manual_all_logits(net, idx_a, block_key0), idx_a,
                [KEY_T])[KEY_T]
            cost = J_a - J_ctl
            ci = boot_rows([cost], lambda c: float(np.mean(c)))
            rec[arm] = dict(gap=float(cost.mean()),
                            gap_ci=[ci[0], ci[1]],
                            gap_per_row=cost.tolist(),
                            health=health_of(float(cost.mean())),
                            n_rows_diverged=int(
                                (~(idx_a == idx_ctl)).any(1).sum()),
                            fired=fired)
            log(f"  [{tag:14s}|{arm:13s}] gap {cost.mean():+8.4f} CI "
                f"[{ci[0]:+7.3f},{ci[1]:+7.3f}] {rec[arm]['health']:9s} | "
                f"div {rec[arm]['n_rows_diverged']}/{idx_ctl.shape[0]} | "
                f"|cos(o,n)| "
                f"{(fired['mean_abs_cos_old_new'] if fired['mean_abs_cos_old_new'] is not None else float('nan')):.3f}")
        # 5. geometric instruments: cos step + occupancy
        R_ = prefix.shape[0]
        rows_sib = torch.tensor(DM_SIB[:R_], dtype=torch.long)
        rows_for = torch.tensor(DM_FOR[:R_], dtype=torch.long)
        sel_t = torch.tensor(BAND, dtype=torch.long)
        H_ = net.cfg.n_head
        cos_sets = {
            "sibling": torch.stack([own[li][rows_sib]
                                    for li in range(net.cfg.n_layer)]),
            "foreign": torch.stack([DONOR_V[li][rows_for][:, 0:H_]
                                    [:, :, sel_t, :]
                                    for li in range(net.cfg.n_layer)]),
            "parent": torch.stack([PARENT_V[li][rows_for][:, :, sel_t, :]
                                   for li in range(net.cfg.n_layer)])}
        cos_mean = {}
        for nm, S_ in cos_sets.items():
            if nm not in arms and nm != "parent":
                continue
            o64 = own.to(torch.float64)
            s64 = S_.to(torch.float64)
            den = (o64.norm(dim=-1) * s64.norm(dim=-1)).clamp_min(1e-12)
            cos_mean[nm] = float((((o64 * s64).sum(-1)) / den)
                                 .abs().mean())
        # occupancy curves + nulls (e111's pooled convention)
        E_curves, null97 = {}, {}
        for nm in [k for k in ("sibling", "foreign", "parent")
                   if k in cos_mean]:
            S_ = cos_sets[nm]
            cum_tot = torch.zeros(HEAD_DIM, dtype=torch.float64)
            tot = 0.0
            for g_ in groups:
                U_ = basis[g_]
                D = S_[g_[0], :, g_[1]].reshape(-1, HEAD_DIM).double()
                e = ((D @ U_) ** 2).sum(0)
                cum_tot += torch.cumsum(e, 0)
                tot += float(e.sum())
            E_curves[nm] = (cum_tot / tot).numpy()
            g_n = torch.Generator().manual_seed(SEED_NULL[nm])
            curves = []
            for _ in range(R_NULL):
                ct = torch.zeros(HEAD_DIM, dtype=torch.float64)
                tt = 0.0
                for g_ in groups:
                    U_ = basis[g_]
                    D = S_[g_[0], :, g_[1]].reshape(-1, HEAD_DIM).double()
                    z = torch.randn(D.shape, generator=g_n,
                                    dtype=torch.float64)
                    z = z / z.norm(dim=-1, keepdim=True)
                    zn = z * D.norm(dim=-1, keepdim=True)
                    e = ((zn @ U_) ** 2).sum(0)
                    ct += torch.cumsum(e, 0)
                    tt += float(e.sum())
                curves.append((ct / tt).numpy())
            null97[nm] = np.percentile(np.stack(curves), 97.5, axis=0)
        # k* search (baseline names it; cells test at k*_ref)
        kstar = None
        first_fire = None
        if "sibling" in E_curves and "foreign" in E_curves:
            ratio = E_curves["sibling"] / np.maximum(
                E_curves["foreign"], 1e-12)
            for k in range(1, HEAD_DIM):
                ok = (ratio[k - 1] >= RATIO_BAR
                      and null97["foreign"][k - 1] < E_curves["foreign"][k - 1]
                      and null97["sibling"][k - 1]
                      < E_curves["sibling"][k - 1])
                if ok and first_fire is None:
                    first_fire = k
                if ok and k <= K_LOW_BAR and kstar is None:
                    kstar = k
        return dict(tag=tag, own_shape=list(own.shape), arms=rec,
                    J_ctl=J_ctl.tolist(), cos_mean=cos_mean,
                    E_curves={k: v.tolist() for k, v in E_curves.items()},
                    null97={k: v.tolist() for k, v in null97.items()},
                    k_star_cell=kstar, first_firing_k=first_fire)

    # ===================================================== FACT + CE column
    def fact_column(tag: str, sd_state=None, mask_mode=False):
        """The fact batteries + CE under the cell's state. mask cells are
        path-matched (base = causal-custom, e150's convention). The
        CALLER (matrix_cell) owns load/restore so the state is active for
        BOTH the self rig and the fact rig."""
        out = {}
        if mask_mode:
            f0, f1 = fwd_causal(net), fwd_offsink(net)
            for j in GEOS:
                out[f"g{j}_base"] = battery_fwd(net, bat_ids[j], zid,
                                                fwd=f0)["mean_pz"]
                out[f"g{j}_arm"] = battery_fwd(net, bat_ids[j], zid,
                                               fwd=f1)["mean_pz"]
            out["ce_base"] = ce_fwd(net, r_eval_x, r_eval_y, fwd=f0)
            out["ce_arm"] = ce_fwd(net, r_eval_x, r_eval_y, fwd=f1)
        else:
            for j in GEOS:
                out[f"g{j}_arm"] = battery_fwd(net, bat_ids[j],
                                               zid)["mean_pz"]
            out["ce_arm"] = ce_fwd(net, r_eval_x, r_eval_y)
        return out

    # the clean base (fact + CE)
    fact_base = fact_column("clean")
    log(f"clean fact base: g-12 {fact_base['g-12_arm']:.4f} g0 "
        f"{fact_base['g0_arm']:.4f} CE {fact_base['ce_arm']:.4f}")

    # ===================================================== run the matrix
    cells: dict[str, dict] = {}

    def matrix_cell(tag, self_kw=None, fact_kw=None, ctl=False):
        log(f"=== CELL {tag} ===")
        sd_state = (fact_kw or {}).get("sd_state")
        if sd_state is not None:
            net.load_state_dict(sd_state)      # active for SELF + FACT both
        try:
            sc = self_cell(tag, **(self_kw or {}))
            fc = fact_column(tag, **(fact_kw or {}))
        finally:
            if sd_state is not None:
                net.load_state_dict(sd_clean)
        cells[tag] = dict(self_=sc, fact=fc, control=ctl)
        return cells[tag]

    # ---- 1. BASELINE (the SELF-BASELINE: full arm set + k* naming)
    base = matrix_cell("baseline")
    BASIS_KSTAR = base["self_"]["k_star_cell"]
    log(f"SELF-BASELINE: k* = {BASIS_KSTAR} (first firing "
        f"{base['self_']['first_firing_k']}) | cos sib "
        f"{base['self_']['cos_mean'].get('sibling')} vs for "
        f"{base['self_']['cos_mean'].get('foreign')}")
    # baseline-only extra arms need k* -> rerun the extras if k* just named
    if not SMOKE:
        extra = ("corpus", "parent") + (
            ("sig_destroyed",) if BASIS_KSTAR is not None else ())
        base2 = self_cell("baseline_extras", block_key0=False,
                          extra_arms=extra, k_sig=BASIS_KSTAR)
        base["self_"]["arms"].update(
            {k: v for k, v in base2["arms"].items()
             if k in ("corpus", "parent", "sig_destroyed")})
        base["self_"]["cos_mean"].update(
            {k: v for k, v in base2["cos_mean"].items() if k == "parent"})
        log(f"baseline extras: corpus gap "
            f"{base['self_']['arms']['corpus']['gap']:+.4f} "
            f"[{base['self_']['arms']['corpus']['health']}] | parent gap "
            f"{base['self_']['arms']['parent']['gap']:+.4f} "
            f"[{base['self_']['arms']['parent']['health']}] | sig_destroyed "
            f"{base['self_']['arms']['sig_destroyed']['gap']:+.4f} "
            f"[{base['self_']['arms']['sig_destroyed']['health']}]")

    if SMOKE:
        matrix_cell("mask", self_kw=dict(block_key0=True),
                    fact_kw=dict(mask_mode=True))
        matrix_cell("poison_0.15", fact_kw=dict(sd_state=sd_p15))
    else:
        # ---- 2-5. the registered interventions
        matrix_cell("mask", self_kw=dict(block_key0=True),
                    fact_kw=dict(mask_mode=True))
        matrix_cell("poison_0.07", fact_kw=dict(sd_state=sd_p07))
        matrix_cell("poison_0.15", fact_kw=dict(sd_state=sd_p15))
        matrix_cell("perm", fact_kw=dict(sd_state=sd_perm))
        # ---- 6. head ablation (hooks around the whole cell)
        _hr = HeadReplace(net, {(L_SEL, H_SEL): mv_sel})
        _hr.__enter__()
        try:
            matrix_cell("L0H3_mean", fact_kw=dict())
        finally:
            _hr.__exit__(None, None, None)
        # ---- 7-8. controls
        matrix_cell("row1_zero_ctl", fact_kw=dict(sd_state=sd_row1),
                    ctl=True)
        _hr2 = HeadReplace(net, {(rand_head[0], rand_head[1]): mv_rand})
        _hr2.__enter__()
        try:
            matrix_cell("randhead_ctl", fact_kw=dict(), ctl=True)
        finally:
            _hr2.__exit__(None, None, None)
        # ---- 9. perm rider (second draw, REPORT-ONLY)
        matrix_cell("perm2_rider", fact_kw=dict(sd_state=sd_perm2), ctl=True)

    # ---- site-stored control column (fact only; arm_b's self out of scope)
    armb_cells = {}
    if not SMOKE:
        f0b, f1b = fwd_causal(net_armb), fwd_offsink(net_armb)
        s_base = site_reads(net_armb, fwd=f0b)
        s_mask = site_reads(net_armb, fwd=f1b)
        armb_cells = dict(
            base_site_onset=s_base["onset_pz"],
            mask_site_onset=s_mask["onset_pz"],
            retention=s_mask["onset_pz"] / max(s_base["onset_pz"], 1e-12),
            ce_base=ce_fwd(net_armb, r_eval_x, r_eval_y, fwd=f0b),
            ce_arm=ce_fwd(net_armb, r_eval_x, r_eval_y, fwd=f1b),
            note="site-stored control: the site-read fact under the mask "
                 "(e150's specificity cell, re-measured in-rig); arm_b's "
                 "SELF column is out of scope (a different organism's self)")
        log(f"arm_b site control: onset {s_base['onset_pz']:.4f} -> "
            f"{s_mask['onset_pz']:.4f} (x{armb_cells['retention']:.3f})")

    # ===================================================== assemble matrix
    def cell_view(tag):
        c = cells[tag]
        fc = c["fact"]
        if "g0_base" in fc:                        # mask cell (path-matched)
            ret = {f"g{j}": fc[f"g{j}_arm"] / max(fc[f"g{j}_base"], 1e-12)
                   for j in GEOS}
            ce_cost = fc["ce_arm"] - fc["ce_base"]
        else:
            ret = {f"g{j}": fc[f"g{j}_arm"]
                   / max(fact_base[f"g{j}_arm"], 1e-12) for j in GEOS}
            ce_cost = fc["ce_arm"] - fact_base["ce_arm"]
        sc = c["self_"]
        gap_sib = sc["arms"]["sibling"]["gap"]
        gap_for = sc["arms"]["foreign"]["gap"]
        intact = bool(gap_sib < HEALTHY_BAR and gap_for > COLLAPSED_BAR)
        if intact:
            mode = "intact"
        elif gap_sib >= COLLAPSED_BAR:
            mode = "REJECTS-OWN"
        elif gap_for <= HEALTHY_BAR:
            mode = "ACCEPTS-FOREIGN"
        else:
            mode = "GRAY"
        cos = sc["cos_mean"]
        cos_ratio = (cos.get("sibling", float("nan"))
                     / max(cos.get("foreign", float("nan")), 1e-12)
                     if "foreign" in cos else float("nan"))
        E = sc["E_curves"]
        occ = dict(k_star_ref=BASIS_KSTAR,
                   E_sib=None if "sibling" not in E else E["sibling"][
                       (BASIS_KSTAR or 8) - 1],
                   E_for=None if "foreign" not in E else E["foreign"][
                       (BASIS_KSTAR or 8) - 1],
                   null97_for=None if "foreign" not in sc["null97"]
                   else sc["null97"]["foreign"][(BASIS_KSTAR or 8) - 1])
        if None not in (occ["E_sib"], occ["E_for"], occ["null97_for"]):
            occ["separation_survives"] = bool(
                occ["E_sib"] >= RATIO_BAR * occ["E_for"]
                and occ["null97_for"] < occ["E_for"])
        else:
            occ["separation_survives"] = None
        return dict(tag=tag, control=c["control"],
                   fact_retention={k: float(v) for k, v in ret.items()},
                   fact_gm12=fc.get("g-12_arm"), fact_g0=fc.get("g0_arm"),
                   ce_cost=float(ce_cost), ce_arm=fc["ce_arm"],
                   gap_sib=gap_sib, gap_for=gap_for,
                   gap_sib_ci=sc["arms"]["sibling"]["gap_ci"],
                   gap_for_ci=sc["arms"]["foreign"]["gap_ci"],
                   health_sib=sc["arms"]["sibling"]["health"],
                   health_for=sc["arms"]["foreign"]["health"],
                   self_intact=intact, self_mode=mode,
                   cos_sib=cos.get("sibling"), cos_for=cos.get("foreign"),
                   cos_ratio=float(cos_ratio) if "foreign" in cos else None,
                   cos_step_survives=(float(cos_ratio) >= RATIO_BAR
                                      if "foreign" in cos
                                      and np.isfinite(cos_ratio) else None),
                   occupancy=occ, k_star_cell=sc["k_star_cell"])

    M = {t: cell_view(t) for t in cells}
    log("--- THE MATRIX ---")
    for t, v in M.items():
        log(f"{t:14s} | fact g-12 x{v['fact_retention']['g-12']:.3f} g0 "
            f"x{v['fact_retention'].get('g0', float('nan')):.3f} | CE "
            f"{v['ce_cost']:+.3f} | self sib {v['gap_sib']:+.3f} "
            f"({v['health_sib']:9s}) for {v['gap_for']:+.3f} "
            f"({v['health_for']:9s}) -> {v['self_mode']}")

    # ---- G7: fact cross-check vs e150's stored cells
    xmap = {"perm": "cons/perm@g0", "mask": "cons/mask@g+0",
            "poison_0.07": "norm=0.07@g+0", "poison_0.15": "norm=0.15@g+0",
            "L0H3_mean": "L0H3-mean@g+0"}
    xdev = {}
    for t, ref in xmap.items():
        if t in M:
            xdev[t] = abs(M[t]["fact_g0"] - E150_REFS[ref])
    gates["G7_fact_crosscheck"] = dict(
        refs=E150_REFS, devs={k: float(v) for k, v in xdev.items()},
        tol=G_FALLBACK_TOL,
        ok=bool(xdev and all(v < G_FALLBACK_TOL for v in xdev.values())))

    # ===================================================== SELF-BASELINE record
    bs_rec = dict(
        note="instrument transfer (e053c_ctx512 home lineage -> "
             "e131_consolidated 6L/6H/192d/blk256): the R45 ideator's "
             "design note — the self-metric must work on the primary net "
             "before any intervention cell counts",
        k_star=BASIS_KSTAR,
        first_firing_k=cells["baseline"]["self_"]["first_firing_k"],
        home_lineage_k_star=7,
        clauses=dict(
            sibling_healthy=dict(
                rule=f"gap(sibling) < {HEALTHY_BAR}",
                gap=M["baseline"]["gap_sib"], ci=M["baseline"]["gap_sib_ci"],
                fires=M["baseline"]["health_sib"] == "healthy"),
            corpus_collapsed=dict(
                rule=f"gap(corpus) > {COLLAPSED_BAR}",
                gap=(cells["baseline"]["self_"]["arms"].get("corpus", {})
                     .get("gap")),
                fires=(cells["baseline"]["self_"]["arms"]
                       .get("corpus", {}).get("health") == "collapsed")),
            foreign_collapsed=dict(
                rule=f"gap(foreign) > {COLLAPSED_BAR}",
                gap=M["baseline"]["gap_for"], ci=M["baseline"]["gap_for_ci"],
                fires=M["baseline"]["health_for"] == "collapsed"),
            cos_step_2x=dict(
                rule="mean|cos(old,sib)| >= 2x mean|cos(old,for)| (e118)",
                cos_sib=M["baseline"]["cos_sib"],
                cos_for=M["baseline"]["cos_for"],
                ratio=M["baseline"]["cos_ratio"],
                fires=bool(M["baseline"]["cos_step_survives"])),
            occupancy_lowdim=dict(
                rule=f"EXISTS k <= {K_LOW_BAR}: E_sib >= 2x E_for AND "
                     f"null97 < both (e111)", k_star=BASIS_KSTAR,
                fires=bool(BASIS_KSTAR is not None),
                note="k* = 1 is the TOP AXIS ALONE — degenerate form: the "
                     "2x energy clause fires at low k (as on e053c, where "
                     "k* was set by the null clause at 7); here the null "
                     "clause already clears at k=1 because this lineage's "
                     "foreign donor sits ABOVE isotropic chance in the top "
                     "axis (E_for(1) ~0.077 vs null97 ~0.032) — the "
                     "exclusion reading of e111 does not hold verbatim on "
                     "this net")),
        texture_arms=dict(
            parent=(cells["baseline"]["self_"]["arms"].get("parent", {})
                    .get("gap"),
                    cells["baseline"]["self_"]["arms"].get("parent", {})
                    .get("health"),
                    "e048_repro V (the install-phase ancestor): does the "
                    "consolidated anchor accept its own past?"),
            sig_destroyed=(cells["baseline"]["self_"]["arms"]
                           .get("sig_destroyed", {}).get("gap"),
                           cells["baseline"]["self_"]["arms"]
                           .get("sig_destroyed", {}).get("health"),
                           "sibling V projected OFF top-k*: e112's causal "
                           "stamp test on the new lineage")))
    fn = bs_rec["clauses"]
    transfer_ok = bool(
        fn["sibling_healthy"]["fires"] and fn["corpus_collapsed"]["fires"]
        and fn["foreign_collapsed"]["fires"])
    bs_rec["functional_transfer"] = transfer_ok
    bs_rec["geometric_transfer"] = bool(
        fn["cos_step_2x"]["fires"] and fn["occupancy_lowdim"]["fires"])

    # ===================================================== adjudication
    def collapsed(t):
        return (t in M) and (not M[t]["self_intact"])

    flat = lambda t: (t in M) and M[t]["ce_cost"] <= CE_FLAT
    routed = bool(collapsed("mask") or collapsed("poison_0.15"))
    constitutional = bool(
        (not collapsed("mask")) and (not collapsed("poison_0.07"))
        and (not collapsed("poison_0.15")) and collapsed("perm"))
    tenant = bool(collapsed("L0H3_mean") and flat("L0H3_mean"))
    independent = bool(all(
        (t in M) and M[t]["self_intact"]
        for t in ("mask", "poison_0.07", "poison_0.15", "perm", "L0H3_mean")))
    clauses = dict(
        SELF_ROUTED=dict(
            fires=routed,
            rule="self collapses under MASK or POISON 0.15 (flat-CE doses; "
                 "CE <= +0.35); a 0.07 collapse is wreck-confounded and "
                 "never gates",
            evidence=dict(mask=M.get("mask", {}).get("self_mode"),
                          mask_ce=M.get("mask", {}).get("ce_cost"),
                          p15=M.get("poison_0.15", {}).get("self_mode"),
                          p15_ce=M.get("poison_0.15", {}).get("ce_cost"))),
        SELF_CONSTITUTIONAL=dict(
            fires=constitutional,
            rule="self INTACT under MASK + POISON 0.07 + POISON 0.15 AND "
                 "collapses under direction-PERM",
            evidence=dict(mask=M.get("mask", {}).get("self_mode"),
                          p07=M.get("poison_0.07", {}).get("self_mode"),
                          p15=M.get("poison_0.15", {}).get("self_mode"),
                          perm=M.get("perm", {}).get("self_mode"))),
        SELF_TENANT=dict(
            fires=tenant,
            rule="self collapses under L0H3 mean-replace at flat CE "
                 "(<= +0.35)",
            evidence=dict(L0H3=M.get("L0H3_mean", {}).get("self_mode"),
                          ce=M.get("L0H3_mean", {}).get("ce_cost"))),
        SELF_INDEPENDENT=dict(
            fires=independent,
            rule="self INTACT under MASK + POISON 0.07 + 0.15 + PERM + L0H3",
            evidence={t: M[t]["self_mode"] for t in
                      ("mask", "poison_0.07", "poison_0.15", "perm",
                       "L0H3_mean") if t in M}))
    if not transfer_ok:
        # the functional binary has no power on this lineage (the foreign
        # gap never clears even the healthy bar at baseline), so any
        # 'collapse' fires VACUOUSLY — suppressed under INSTRUMENT-INVALID
        for c in clauses.values():
            c["fires_unsuppressed"] = c["fires"]
            c["fires"] = False
            c["note"] = ("vacuous under INSTRUMENT-INVALID: self_intact "
                         "requires gap(foreign) > 1.0, which NO cell "
                         "including baseline satisfies — 'collapse' would "
                         "fire trivially everywhere; suppressed")
        clause = "INSTRUMENT-INVALID"
        _cg = fn["corpus_collapsed"]["gap"]
        verdict = (f"the SELF-BASELINE failed its functional clauses on the "
                   f"primary lineage (sibling gap "
                   f"{fn['sibling_healthy']['gap']:+.3f}, corpus "
                   f"{_cg if isinstance(_cg, float) else 'n/a'}, foreign "
                   f"{fn['foreign_collapsed']['gap']:+.3f}) — the e111/"
                   f"e112 instrument does not transfer as worded; all cells "
                   f"reported as texture, no dissociation claim")
    elif routed:
        clause = "SELF-ROUTED"
        which = [t for t in ("mask", "poison_0.15") if collapsed(t)]
        verdict = (f"SELF-ROUTED fires: self-recognition collapsed under "
                   f"{which} at CE cost "
                   + ", ".join(f"{M[t]['ce_cost']:+.2f}" for t in which)
                   + f" — the self rides the pivot like a memory "
                   f"(gap_sib {M[which[0]]['gap_sib']:+.3f} / gap_for "
                   f"{M[which[0]]['gap_for']:+.3f})")
    elif constitutional:
        verdict = ("SELF-CONSTITUTIONAL (the W015 prediction) fires: the "
                   "self survived the mask and both poison doses but died "
                   "under direction-PERM — it reads V-STRUCTURE (direction, "
                   "upstream, presence-independent), unlike the fact")
    elif tenant:
        verdict = ("SELF-TENANT fires: the self died under fact-specific "
                   "head ablation at flat CE — self and fact share readout "
                   "machinery")
    elif independent:
        verdict = ("SELF-INDEPENDENT fires: the self survived EVERY "
                   "measured intervention (mask, both poison doses, perm, "
                   "L0H3) — computed upstream of everything measured here")
    else:
        clause = "TEXTURE"
        verdict = ("no registered outcome as worded — the pattern is "
                   "mixed; every cell reported with numbers (no bar "
                   "shopping)")
    if transfer_ok and not (routed or constitutional or tenant or independent):
        clause = "TEXTURE"
    log("=" * 78)
    log(f"E146 VERDICT [{clause}]: {verdict}")
    log(f"  ROUTED {clauses['SELF_ROUTED']['fires']} | CONSTITUTIONAL "
        f"{clauses['SELF_CONSTITUTIONAL']['fires']} | TENANT "
        f"{clauses['SELF_TENANT']['fires']} | INDEPENDENT "
        f"{clauses['SELF_INDEPENDENT']['fires']} | transfer {transfer_ok}"
        + (""
           if transfer_ok else
           "  (all clause firings SUPPRESSED as vacuous — the foreign gap "
           "never clears the healthy bar, so 'collapse' would fire "
           "trivially everywhere)"))
    log("=" * 78)

    # post-hoc texture (clearly labeled; never gates any bar)
    _bl = cells["baseline"]["self_"]["arms"]
    texture_findings = dict(
        label="POST-HOC TEXTURE (instrument invalid; descriptive only)",
        graded_readout=(
            f"the anchor band's functional readout on this lineage is "
            f"GRADED BY CONTENT PLAUSIBILITY, not binary by generator "
            f"identity: own-V sibling {_bl['sibling']['gap']:+.3f} (healthy) "
            f"~ foreign e021-V "
            f"{_bl['foreign']['gap']:+.3f} (healthy!) ~ parent-ancestor-V "
            f"{_bl['parent']['gap']:+.3f} (healthy) << own-weights-x-corpus-"
            f"text {_bl['corpus']['gap']:+.3f} (gray) — the e053c "
            f"exclusion-of-foreign structure (e112 foreign rider +2.41) is "
            f"ABSENT here: foreign and even ancestral V splice in at ~zero "
            f"cost while plausible-but-wrong corpus V costs 0.77 nats"),
        occupancy_robust=(
            "the geometric self/other separation (sibling/foreign energy "
            "ratio in the own top-k* subspace, 3.0-5.5x) SURVIVES every "
            "measured intervention — mask 3.1x, poison-0.07 4.3x, "
            "poison-0.15 3.4x, perm 3.4x, L0H3 3.0x, controls ~3.1x, "
            "perm2-wreck 5.5x — the V-space separation is indestructible "
            "by anything measured (the SELF-INDEPENDENT pattern at the "
            "geometry level), but the functional instrument that would "
            "license that claim is invalid on this lineage"),
        cos_step_subbar=(
            "the splice-time V-cos step is 1.43-1.90x across all cells — "
            "REAL but below e118's 2x bar everywhere (e053c home value "
            "2.86x): a weaker, non-binary version of the same separation"),
        perm2_rider=(
            "the second direction-permutation (seed 14620) is CATASTROPHIC "
            "(fact x0.000 at g-12, CE +2.05) where e141's seed-14101 perm "
            "costs x0.79 at CE +0.70 — direction-destruction magnitude is "
            "permutation-draw-dependent; the PERM column's mild cost is a "
            "property of that specific draw (rider, never gates)"),
        self_gaps_all_flat=(
            "every cell's sibling AND foreign gaps sit in [-0.10, +0.22] "
            "nats (healthy) — including under the CE +2.05 perm2 wreck: "
            "the paired-gap design shows the intervened nets treat own-V "
            "and foreign-V splices EQUALLY in every state measured"),
    )

    honesty = dict(
        instrument_lineage_transfer=(
            "the self instruments were built and calibrated on "
            "e053c_ctx512 (4L/4H/128d/blk512; bars 0.3/1.0, k*=7); this "
            "run ports them to e131_consolidated (6L/6H/192d/blk256) with "
            "a rescaled geometry — the SELF-BASELINE gates the transfer, "
            "and the port's deviations are enumerated in "
            "recipe_deviations; the home bars are applied VERBATIM (no "
            "recalibration shopping), so a gray-zone baseline reads as "
            "transfer failure, not as a tuned instrument"),
        single_net=(
            "the whole self column lives on ONE consolidated net (e113 "
            "recipe, one seed); the R nets are not self-batteried (the "
            "self arc has no replication-seed battery yet); verdicts are "
            "line-specific until replicated"),
        mask_off_distribution=(
            "the forced-off-sink mask puts every context OFF its training "
            "distribution; the CE column prices the generic part "
            "(+0.03 on this bank) and the site-stored control (arm_b, "
            "fact retention under mask "
            f"{armb_cells.get('retention') if armb_cells else 'smoke: n/a'}"
            ") is the specificity discriminator for the FACT column; for "
            "the SELF column the within-cell paired design (same mask for "
            "control and arms, judged by the same masked net) is the "
            "analogue of that control"),
        judge_is_the_intervened_net=(
            "each cell judges its own streams under the same intervention "
            "(weights/mask/hooks) — the gap isolates the splice effect, "
            "but a wrecked cell (poison 0.07, CE +0.84) has a degraded "
            "judge; its gap is reported and its self-collapse there is "
            "wreck-confounded by registration (never gates a bar)"),
        foreign_donor_is_unintervened=(
            "the foreign V content comes from e021_task's own run (a "
            "different organism — never intervened); under weight "
            "interventions the recipient's own V norms shift while the "
            "foreign donor's do not; the norm ratios are recorded per arm "
            "(fired.mean_norm_ratio) so the asymmetry is auditable"),
        occupancy_support_only=(
            "the cos-step and occupancy columns are geometric correlates "
            "(e111-class); e112 showed the functional gap is the causal "
            "instrument — the adjudication reads ONLY the functional "
            "binary; the geometric columns are reported per cell and any "
            "dissociation between them and the functional gap is texture "
            "to savor, not a bar"),
    )

    metrics = dict(
        experiment="e146_dissociation_matrix",
        purpose=("W015's registered question: does the SELF survive losing "
                 "its pivot? First contact between the memory arc and the "
                 "self arc: interventions (forced-off-sink MASK / norm "
                 "POISON 0.07+0.15 / direction-PERM / L0H3 mean-replace; "
                 "row-1 + random-head controls) x functions (FACT g-12+g0, "
                 "SELF binary+occupancy+gaps, corpus CE), every cell "
                 "CE-priced (the e150 discipline). Primary net "
                 "e131_consolidated_e113 (B43 line). Instruments: e111/"
                 "e112 self rig PORTED to the blk256 lineage with a "
                 "gating SELF-BASELINE; e150's fact/mask/head instruments "
                 "verbatim."),
        started=started, wall_s=round(time.time() - T0, 1),
        cpu_only=True, threads=torch.get_num_threads(), smoke=SMOKE,
        registered_prediction=REGISTERED_PREDICTION,
        nets=dict(primary=str(CONS_CK), parent=str(INST_CK),
                  foreign=str(e105.DONOR_CKPT), site_control=str(ARMB_CK),
                  eval_only=True,
                  primary_gates=dict(pz_g0=bz_g0["mean_pz"],
                                     ce_r=ce_clean,
                                     pz_gm12=bz_g12["mean_pz"])),
        seeds=dict(corpus=1337, prompts=SEED_PROMPT, sampling=SEED_SAMPLE,
                   continuation=SEED_CONT, donor_map_sib=SEED_DONOR_SIB,
                   donor_map_for=SEED_DONOR_FOR,
                   foreign_donor_window=e105.DONOR_WIN[1],
                   parent_stream="seed-202 battery + seed-7 sampling",
                   ce_bank=R_EVAL_SEED, perm=PERM_DIM_SEED,
                   perm_rider=PERM_DIM_SEED_RIDER,
                   nulls=SEED_NULL, null_R=R_NULL,
                   note="every arm stream is a published seed; new "
                        "dedicated seeds: nulls 14601-14603, perm rider "
                        "14620 (never touching any arm stream)"),
        gates=dict(G_SPLICE=G_SPLICE, **{k: v for k, v in gates.items()}),
        self_baseline=bs_rec,
        matrix=M,
        matrix_note="per-cell self_mode values follow the registered "
                    "operationalization (sib healthy AND foreign collapsed "
                    "=> intact) — under INSTRUMENT-VALIDATION FAILURE every "
                    "cell reads ACCEPTS-FOREIGN VACUOUSLY (the foreign gap "
                    "never exceeds the healthy bar, baseline included); "
                    "read the gaps and geometric columns, not the modes",
        texture_findings=texture_findings,
        controls=dict(
            row1_zero=M.get("row1_zero_ctl"),
            randhead=dict(head=f"L{rand_head[0]}H{rand_head[1]}",
                          e133_drop=rand_head[2], e133_ce=rand_head[3],
                          cell=M.get("randhead_ctl")),
            perm2_rider=M.get("perm2_rider"),
            site_stored=armb_cells or None,
            note="controls never gate the adjudication; their cells are "
                 "priced texture"),
        adjudication=dict(clauses=clauses, clause=clause, verdict=verdict,
                          priority="ROUTED > CONSTITUTIONAL > TENANT > "
                                  "INDEPENDENT > TEXTURE"),
        honesty_reflex=honesty,
        recipe_deviations=recipe_deviations,
        e150_crosscheck=gates["G7_fact_crosscheck"],
        ckpt_inventory=dict(saved={}, external_used=[
            f"runs/checkpoints/{p.name}"
            for p in (CONS_CK, INST_CK, e105.DONOR_CKPT, ARMB_CK)],
            note="eval-only: no checkpoints written or regenerated"),
    )
    save_json(out_dir / "metrics.json", E43.jsonable(metrics))
    log("metrics.json written")
    plot(out_dir / "dissociation_matrix.png", M, bs_rec, clause, verdict,
         armb_cells)
    log(f"outputs: {out_dir / 'metrics.json'}, "
        f"{out_dir / 'dissociation_matrix.png'}")
    log(f"total {time.time() - T0:.1f}s")


# ------------------------------------------------------------------ plot

def _cell_color(val, kind):
    """kind: 'high' (higher=survives, bars good/bad), 'low' (gap where
    lower=healthy), 'lowbar' (gap where higher=collapsed)."""
    if val is None or not np.isfinite(val):
        return "gold", "n/a"
    if kind == "high":
        txt, good, bad = f"x{val:.2f}", 0.8, 0.5
    elif kind == "ratio":
        txt, good, bad = f"x{val:.2f}", RATIO_BAR, RATIO_BAR
    elif kind == "low":
        txt, good, bad = f"{val:+.2f}", HEALTHY_BAR, COLLAPSED_BAR
    else:                                          # 'highgap'
        txt, good, bad = f"{val:+.2f}", COLLAPSED_BAR, HEALTHY_BAR
    if kind == "high":
        col = "seagreen" if val >= good else ("firebrick" if val < bad
                                              else "gold")
    elif kind == "ratio":
        col = "seagreen" if val >= good else "firebrick"
    elif kind == "low":
        col = "seagreen" if val < good else ("firebrick" if val >= bad
                                             else "gold")
    else:
        col = "seagreen" if val > good else ("firebrick" if val <= bad
                                             else "gold")
    return col, txt


def plot(path, M, bs_rec, clause, verdict, armb_cells):
    """THE matrix figure: interventions x functions with CE badges."""
    order = [t for t in ("baseline", "mask", "poison_0.07", "poison_0.15",
                         "perm", "L0H3_mean", "row1_zero_ctl",
                         "randhead_ctl", "perm2_rider") if t in M]
    row_lab = {"baseline": "baseline (clean)", "mask": "MASK off-sink",
               "poison_0.07": "POISON 0.07", "poison_0.15": "POISON 0.15",
               "perm": "PERM wpe[0]", "L0H3_mean": "L0H3 mean-replace",
               "row1_zero_ctl": "ctl: wpe[1]=0",
               "randhead_ctl": "ctl: random head",
               "perm2_rider": "rider: perm seed2"}
    cols = ["FACT g-12\n(novel, primary)", "FACT g0\n(trained)",
            "SELF binary\nV-cos sib/for", "SELF occupancy\nE_sib/E_for@k*",
            "SELF gap sib\n(own V)", "SELF gap for\n(foreign V)"]
    fig = plt.figure(figsize=(20.0, 13.0))
    gs = fig.add_gridspec(3, 2, height_ratios=(1.5, 1.0, 1.0))

    # ---- THE MATRIX GRID
    ax = fig.add_subplot(gs[0, :])
    nr, nc = len(order), len(cols)
    for i, t in enumerate(order):
        v = M[t]
        o = v["occupancy"]
        occr = (o["E_sib"] / o["E_for"]
                if None not in (o["E_sib"], o["E_for"]) else None)
        kinds = [("high", v["fact_retention"]["g-12"]),
                 ("high", v["fact_retention"].get("g0")),
                 ("ratio", v["cos_ratio"]), ("ratio", occr),
                 ("low", v["gap_sib"]), ("highgap", v["gap_for"])]
        for j, (kind, val) in enumerate(kinds):
            col, txt = _cell_color(val, kind)
            y = nr - 1 - i
            rect = mpatches.FancyBboxPatch(
                (j + 0.04, y + 0.06), 0.92, 0.88,
                boxstyle="round,pad=0.02", fc=col, alpha=0.30, ec="k",
                lw=0.6)
            ax.add_patch(rect)
            ax.text(j + 0.5, y + 0.64, txt, ha="center", va="center",
                    fontsize=10.5, weight="bold")
            ax.text(j + 0.5, y + 0.24, f"CE {v['ce_cost']:+.2f}",
                    ha="center", va="center", fontsize=7.6,
                    color="dimgray",
                    bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.6))
            if j == 5:
                ax.text(j + 0.93, y + 0.5, v["self_mode"], ha="right",
                        va="center", fontsize=7.2, rotation=90,
                        color=("seagreen" if v["self_intact"]
                               else "firebrick"), weight="bold")
    ax.set_xlim(-0.02, nc)
    ax.set_ylim(-0.1, nr)
    ax.set_xticks([j + 0.5 for j in range(nc)], cols, fontsize=8.6)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks([nr - 1 - i + 0.5 for i in range(nr)],
                  [row_lab[t] for t in order], fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("E146 — THE DISSOCIATION MATRIX: interventions x functions "
                 "(every cell CE-badged; green=survives bar, red=dies, "
                 "gold=gray/n-a)", fontsize=11, pad=34)

    # ---- panel 2: the gaps (the functional self binary per cell)
    ax2 = fig.add_subplot(gs[1, 0])
    xs = np.arange(len(order))
    ax2.bar(xs - 0.19, [M[t]["gap_sib"] for t in order], 0.36,
            color="seagreen", edgecolor="k", lw=0.5,
            label="gap(sibling) — own V spliced in")
    ax2.bar(xs + 0.19, [M[t]["gap_for"] for t in order], 0.36,
            color="firebrick", edgecolor="k", lw=0.5,
            label="gap(foreign) — e021 V spliced in")
    ax2.axhline(HEALTHY_BAR, ls="--", color="seagreen", lw=1.2)
    ax2.axhline(COLLAPSED_BAR, ls="--", color="firebrick", lw=1.2)
    ax2.axhspan(HEALTHY_BAR, COLLAPSED_BAR, color="gray", alpha=0.10)
    ax2.text(len(order) - 0.4, HEALTHY_BAR + 0.05,
             f"healthy < {HEALTHY_BAR}", fontsize=8, color="seagreen",
             ha="right")
    ax2.text(len(order) - 0.4, COLLAPSED_BAR + 0.05,
             f"collapsed > {COLLAPSED_BAR}", fontsize=8, color="firebrick",
             ha="right")
    ax2.set_xticks(xs, [row_lab[t] for t in order], fontsize=7, rotation=18)
    ax2.set_ylabel("clean-judge tail gap vs matched control (nats)")
    ax2.legend(fontsize=8)
    ax2.set_title("the functional self/other binary under each "
                  "intervention (the adjudication instrument)", fontsize=9.5)

    # ---- panel 3: the dissociation plane (fact retention vs CE, colored
    # by self survival) — the money shot
    ax3 = fig.add_subplot(gs[1, 1])
    for t in order:
        v = M[t]
        ax3.scatter(v["ce_cost"], 100 * v["fact_retention"]["g-12"], s=110,
                    color=("tab:blue" if v["self_intact"] else "tab:red"),
                    edgecolor="k", lw=0.7, zorder=3,
                    marker=("s" if v["control"] else "o"))
        ax3.annotate(row_lab[t], (v["ce_cost"],
                                  100 * v["fact_retention"]["g-12"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax3.axhline(50, ls="--", color="gray", lw=0.9)
    ax3.axhline(80, ls=":", color="seagreen", lw=0.9)
    ax3.axvline(CE_FLAT, ls="--", color="seagreen", lw=1.1)
    ax3.axvspan(0, CE_FLAT, color="seagreen", alpha=0.06)
    ax3.set_xlabel("CE cost vs clean net (e065 bank)")
    ax3.set_ylabel("FACT retention @g-12 (novel geometry, %)")
    ax3.text(CE_FLAT / 2, 6, "flat-CE zone", ha="center", fontsize=8,
             color="seagreen")
    ax3.set_title("the dissociation: fact retention vs CE cost, colored by "
                  "SELF survival\n(blue = self intact, red = self "
                  "collapsed; squares = controls/rider)", fontsize=9.5)
    ax3.grid(alpha=0.25)

    # ---- panel 4: the geometric support columns per cell
    ax4 = fig.add_subplot(gs[2, 0])
    occr = [(M[t]["occupancy"]["E_sib"] / M[t]["occupancy"]["E_for"]
             if None not in (M[t]["occupancy"]["E_sib"],
                             M[t]["occupancy"]["E_for"])
             else float("nan")) for t in order]
    cosr = [(M[t]["cos_ratio"] if M[t]["cos_ratio"] is not None
             else float("nan")) for t in order]
    ax4.bar(xs - 0.19, cosr, 0.36, color="tab:purple", edgecolor="k",
            lw=0.5, label="V-cos step sib/for (2x bar)")
    ax4.bar(xs + 0.19, occr, 0.36, color="tab:cyan", edgecolor="k", lw=0.5,
            label=f"occupancy E_sib/E_for @k*={bs_rec['k_star']} (2x bar)")
    ax4.axhline(RATIO_BAR, ls="--", color="k", lw=1.2)
    ax4.set_xticks(xs, [row_lab[t] for t in order], fontsize=7, rotation=18)
    ax4.set_ylabel("separation ratio")
    ax4.legend(fontsize=8)
    ax4.set_title("the geometric self instruments per cell (support "
                  "columns; the 2x bars of e111/e118)", fontsize=9.5)

    # ---- panel 5: verdict + baseline record
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")
    cl = bs_rec["clauses"]
    lines = ["SELF-BASELINE (instrument transfer, gates the matrix):"]
    for k, v in cl.items():
        g = v.get("gap")
        extra = (f"  gap {g:+.3f}" if isinstance(g, float)
                 else f"  ratio {v['ratio']:.2f}" if isinstance(
                     v.get("ratio"), float)
                 else (f"  k*={v['k_star']}" if v.get("k_star") is not None
                       else ""))
        lines.append(f"  {k:22s} {'FIRES' if v['fires'] else 'no'}{extra}")
    tx = bs_rec.get("texture_arms", {})
    if tx.get("parent") and isinstance(tx["parent"][0], float):
        lines.append(f"  parent (texture): gap {tx['parent'][0]:+.3f} "
                     f"[{tx['parent'][1]}]")
    if tx.get("sig_destroyed") and isinstance(tx["sig_destroyed"][0],
                                              float):
        lines.append(f"  sig_destroyed: gap {tx['sig_destroyed'][0]:+.3f} "
                     f"[{tx['sig_destroyed'][1]}]")
    lines.append(f"  transfer: functional {bs_rec['functional_transfer']}, "
                 f"geometric {bs_rec['geometric_transfer']}")
    lines.append("")
    lines.append(f"VERDICT [{clause}]:")
    lines += [f"  {w}" for w in textwrap.wrap(verdict, 100)]
    if armb_cells:
        lines.append(f"site-stored control: mask retention "
                     f"x{armb_cells['retention']:.3f} @CE "
                     f"{armb_cells['ce_arm'] - armb_cells['ce_base']:+.2f}")
    ax5.text(0.02, 0.97, "E146 — adjudication", fontsize=12, weight="bold",
             va="top")
    for i, t in enumerate(lines):
        ax5.text(0.02, 0.945 - i * 0.033, t, fontsize=8.4, va="top",
                 family="monospace")

    fig.suptitle(f"E146 — THE DISSOCIATION MATRIX (W015) | {clause} | "
                 f"k*={bs_rec['k_star']} on e131_consolidated | every cell "
                 f"CE-priced", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
