"""E226 — FQ5: THE GMAIL/IPHONE INTERIOR (eval-only; CPU; 2026-10-04).

THE QUESTION (scratch/fq5_design.md, frozen at dispatch): stop asking
WHAT names the third sorting dimension — ask WHERE the organism
differentiates the two near-token-identical, opposite-fate relations
during the wash; let the measurement decide whether REAL-AND-UNNAMED is
the object's correct state.

The anchor pair (e216's committed table, the record this cell hangs on):
Gmail HOLDS (hr 0.905/0.931 on washes 1/2; resid +0.429/+0.428) while
iPhone DIES (hr 0.184/0.342; resid -0.206/-0.065) — near token-identical
(both ~0/1M source frequency, frag 0.167 vs 0.143, cue 8 vs 7 tokens),
same family (product), same battery, and e220's best token model WIDENS
the gap. The instrument moves from the FEATURE side to the MECHANISM
side: what does the wash's first-order gradient DO to each relation's
own support direction?

THE INSTRUMENT (all eval-only on the THREE-wash 124M archive + CPU
batch-1 forwards/backwards, threads 4):
  (1) THE SUPPORT DIRECTION PER PROBE: at t=0, each of the 54 probes'
      s_i = grad p(answer|context) over ALL 124,439,808 parameters
      (batch-1, eval, dropout 0, fp32, L2-normalized); the 54x54
      pairwise geometry (the Gmail/iPhone cos; within-family spreads).
  (2) THE WASH'S TREATMENT: g(w,s) = grad of the batch-CE on draw #(s+1)
      of wash w's frozen window stream, at the archived weights W_{w,s}
      (matched point: the state's OWN next batch — the actual next
      gradient of each trajectory; s=0 -> pristine + draw #1, which is
      also the hot-set source). Alignment A(w,s,i) = cos(g(w,s), s_i(0)).
  (3) THE ROTATION READ: at the settled +80 states, R(w,i) =
      cos(s_i(w,80), s_i(0)) and the g11 hot-set overlap
      H(w,i) = |top2000|s_i(w,80)|  n  top2000|g(w,0)|| / 2000
      (k=2000 primary; ladder 2k/20k/200k co-reported on 124.4M coords).
  (4) THE ANCHOR PLOT: Gmail and iPhone's three curves against the
      within-family (product, 5 non-anchor probes) noise bands.

GATES (Rule 12): G_ENV (threads 4, CPU-only, psutil load checks), G_SIZE
(inherited 124M reason), G_CORPUS (rebuild == e182's record), G_BATT
(the four batteries rebuilt VERBATIM by module import reproduce the
committed t=0 records; the 54 == e216's committed assignment), G_STATES
(inventory + per-state re-probe dp <= 0.005 vs committed), G_DRAWS (the
reproduced draw streams certified BIT-EXACTLY against the archived
generator states at step 80), G_SUPPORT (per-probe directional FD gate
dp>0 at eps 0.02 L2, e204's convention, 0.05 co-read; determinism
re-verify), G_WASHGRAD (per-state directional FD gate dCE<0 at eps 0.02;
anchors' +80 supports FD co-gated).

BARS VERBATIM (scratch/fq5_design.md, frozen by the dispatch BEFORE any
compute; adjudicate against exactly this; no bar shopping):
  SUPPORT-DIFFERENTIATES: "fires if Gmail and iPhone separate on the
  alignment or rotation curves (>= 2 sigma of the within-family spread
  at any state) — the wash treats the two relations' supports
  differently; the interior mapped; the dimension's seat located even if
  its name stays open."
  GEOMETRY-IDENTICAL: "fires if the curves sit within the family noise
  at every read — the differentiation is below the first-order floor;
  REAL-AND-UNNAMED confirmed as the correct state; the honest bound
  recorded."

Envelope: CPU-only, torch threads 4 (the dispatch envelope; overrides
e182c's module-level 8), tiny bursts, load checks per phase. No GPU. No
NOTES/THINKING/QUEUE/STATE edits (dispatch). Smoke via E226_SMOKE=1
(subset probes/states; nothing adjudicated).
"""

from __future__ import annotations

import copy
import json
import math
import os
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                          # noqa: E402
import torch.nn.functional as F                       # noqa: E402

import common                                          # noqa: E402
from common import now_iso, run_dir, save_json          # noqa: E402

import e182c_forgetting_control as e1                  # noqa: E402 — phase-1 machinery, VERBATIM
import e182c2_template as e2                           # noqa: E402 — the template battery, VERBATIM

# e182c sets torch threads 8 at module level; the dispatch envelope for
# THIS cell is threads <= 4 (shared box, e225 co-running) — reset AFTER
# the imports, before any compute.
THREADS = 4
torch.set_num_threads(THREADS)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import textwrap                                        # noqa: E402

try:
    import psutil                                     # noqa: E402
except ImportError:
    psutil = None

SMOKE = os.environ.get("E226_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e226_smoke" if SMOKE else "e226"

T0 = time.time()
_log_t0 = time.time()
def log(m: str) -> None:
    print(f"[{time.time() - _log_t0:7.1f}s] {m}", flush=True)

# ------------------------------------------------------------ the archive
# Three washes, all the SAME frozen recipe (corpus, SEQ 512, BATCH 8,
# AdamW 5e-5), differing ONLY in the window-draw seed; all state weights
# archived fp32. w1 = the CPU replay of e182's own wash; w2/w3 = fresh
# draws (computed on GPU fp32 in their own cells; the states are what is
# read here, on CPU).
WASHES = ("w1", "w2", "w3")
W_SEED = {"w1": 18202, "w2": 20261002, "w3": 21703}
W_LABEL = {"w1": "wash 1 (e182c replay, seed 18202)",
           "w2": "wash 2 (e182c2 fresh, seed 20261002)",
           "w3": "wash 3 (e217 fresh, seed 21703)"}
CK = common.REPO / "runs" / "checkpoints"
W_ARCH = {
    "w1": {2: CK / "e182c_s2.pt", 10: CK / "e182c_s10.pt",
           50: CK / "e182c_s50.pt", 80: CK / "e182c_s80.pt"},
    "w2": {10: CK / "e182c2_fresh_s10.pt", 50: CK / "e182c2_fresh_s50.pt",
           80: CK / "e182c2_fresh_s80.pt"},
    "w3": {10: CK / "e217_fresh_s10.pt", 50: CK / "e217_fresh_s50.pt",
           80: CK / "e217_fresh_s80.pt"},
}
W_LATEST = {"w1": CK / "e182c_replay_latest.pt",
            "w2": CK / "e182c2_fresh_latest.pt",
            "w3": CK / "e217_fresh_latest.pt"}
W_STATES = {"w1": (0, 2, 10, 50, 80), "w2": (0, 10, 50, 80),
            "w3": (0, 10, 50, 80)} if not SMOKE else {
    "w1": (0, 2, 10), "w2": (0, 10), "w3": (0, 10)}
ROT_STATE = 80 if not SMOKE else 10    # the settled state's rotation read

# committed records (the re-probe convention, e214's discipline)
E182C_M = common.REPO / "runs" / "e182c" / "metrics.json"
E182C2_M = common.REPO / "runs" / "e182c2" / "metrics.json"
E182C2_J = common.REPO / "runs" / "e182c2" / "journal_p2.json"
E217_M = common.REPO / "runs" / "e217" / "metrics.json"
E216_M = common.REPO / "runs" / "e216" / "metrics.json"
E219_M = common.REPO / "runs" / "e219" / "metrics.json"
E220_M = common.REPO / "runs" / "e220" / "metrics.json"

# ------------------------------------------------------------ instruments
K_HOT = 2000                       # g11's primary k (ladder co-reported)
K_LADDER = (2000, 20000, 200000)
FD_EPS = (0.05,) if SMOKE else (0.02, 0.05)   # e204's FD sizes (L2 units)
Z_BAR = 2.0                        # the frozen 2-sigma bar
TOL_T0_DP = 0.010                  # e214's TOL_PROBE_DP (t=0 vs records)
TOL_STATE_DP = 0.005               # e214's TOL_STATE_DP (loaded states)
RAM_FLOOR_GB = 16.0                # below this the fp16 cache is unsafe

ANCHOR_G = "The email service made by Google->Gmail"
ANCHOR_I = "The phone made by Apple->iPhone"
BAND_FAMILY = "product"            # the anchors' own family6 label

REGISTERED_PREDICTION = {
    "bars_verbatim": {
        "SUPPORT-DIFFERENTIATES": "fires if Gmail and iPhone separate on "
        "the alignment or rotation curves (>= 2 sigma of the "
        "within-family spread at any state) — the wash treats the two "
        "relations' supports differently; the interior mapped; the "
        "dimension's seat located even if its name stays open.",
        "GEOMETRY-IDENTICAL": "fires if the curves sit within the family "
        "noise at every read — the differentiation is below the "
        "first-order floor; REAL-AND-UNNAMED confirmed as the correct "
        "state; the honest bound recorded.",
    },
    "operationalizations": (
        "probes = e216's committed 54-probe assignment (fact 20 / ctrl 12 "
        "/ near 3 / tmpl 19; families = e216's family6 labels, inherited "
        "by joining on the fact string); support s_i = grad of p(ans "
        "first token | the probe's VERBATIM 2-shot prompt) w.r.t. ALL "
        "124,439,808 params, batch-1 forward, eval, dropout 0, CPU fp32, "
        "L2-normalized; cached fp16 with fp64-accumulated chunked dots "
        "(cache rounding ~5e-4 on cos, disclosed; determinism re-verified "
        "by recomputing probe 0); wash gradient g(w,s) = grad of the "
        "wash's OWN batch-CE (mean CE over 8x512 tokens, e182's VERBATIM "
        "loss) on draw #(s+1) of wash w's frozen window stream, at the "
        "archived weights W_{w,s} (MATCHED POINT: the state's own next "
        "batch, the actual next gradient of the trajectory; s=0 = "
        "pristine weights + draw #1, also the hot-set source); draws "
        "reproduced from the archived seeds and certified BIT-EXACTLY "
        "against the archived generator states at step 80 (G_DRAWS); "
        "alignment A(w,s,i) = cos(g(w,s), s_i(0)) per wash per state per "
        "probe (w1 states 0/2/10/50/80; w2/w3 0/10/50/80); rotation "
        "R(w,i) = cos(s_i(w,80), s_i(0)) at the settled +80 states; hot "
        "overlap H(w,i) = |top2000|s_i(w,80)| n top2000|g(w,0)|| / 2000 "
        "(g11's convention; ladder 2k/20k/200k co-reported; random "
        "baseline k/P = 1.6e-5); family band = the product family's 5 "
        "non-anchor probes (Xbox, Chrome, iPad, iTunes, PlayStation); "
        "sigma(read) = std ddof=1 of the band at that read; Z(read) = "
        "|read_Gmail - read_iPhone| / sigma; REGISTERED READS = A at "
        "every (wash, state) [13 reads] + R per wash [3] + H per wash "
        "[3] = 19 reads; SUPPORT-DIFFERENTIATES := any registered read "
        "Z >= 2.0; GEOMETRY-IDENTICAL := all registered reads Z < 2.0; "
        "the two bars are complements (no GRADED tier); multiplicity "
        "disclosed in honesty (P(any Z>=2 | pure noise) ~= 0.59 for 19 "
        "Gaussian reads) and the replication count (how many washes "
        "fire) co-reported, never adjudicated; founder-anchor family "
        "band co-reported, never adjudicated; FD gates: per-probe "
        "p(theta0 + eps*s_hat) > p0 at eps 0.02 (e204's convention; 0.05 "
        "co-read), per-state CE(W + eps*g_hat) < CE(W) at eps 0.02, "
        "anchors' +80 supports FD co-gated; gated on G_ENV/G_SIZE/"
        "G_CORPUS/G_BATT/G_STATES/G_DRAWS/G_SUPPORT/G_WASHGRAD"
    ),
    "registration": "bars frozen VERBATIM from scratch/fq5_design.md "
        "(the ripened design note, frozen by the dispatch brief BEFORE "
        "any compute; the dispatch brief is the registration); adjudicate "
        "against exactly this; no bar shopping",
}

deviations: list[str] = [
    "The two bars are EXHAUSTIVE complements (any-Z>=2 vs all-Z<2): no "
    "GRADED tier exists for this cell — registered as such, not a trim.",
    "The support cache is fp16 (13.4 GB) not fp32 (26.9 GB): the box is "
    "shared (e225 co-running; 63 GB total). Cache rounding moves cos by "
    "~5e-4 — two orders below any plausible family sigma; disclosed in "
    "honesty; determinism re-verified against a fresh recompute.",
    "w2/w3 weights were computed on GPU fp32 in their own cells (their "
    "archived states are the record); every gradient HERE is CPU fp32 on "
    "those archived states — the origin is disclosed, the read is "
    "state-faithful.",
    "The +80 rotation/FD reads use each wash's archived +80 state only "
    "(the design's 'settled +80 states'); intermediate-state supports "
    "are not computed (cost), and this is registered, not trimmed.",
    "The wash forward is eval-mode: GPT-2 has no batchnorm and every "
    "dropout module was already zeroed by e182c's load_organism (the "
    "VERBATIM organism loader), so train-mode and eval-mode forwards are "
    "identical here — the gradient is the wash's own loss gradient "
    "either way.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch).",
    "Smoke mode (E226_SMOKE=1): product family + one probe per other "
    "battery (11 probes), states w1 {0,2,10} / w2,w3 {0,10}, rotation at "
    "+10, FD eps {0.05}; nothing adjudicated or gated (SMOKE stamp).",
]

trims: list[str] = []


# ------------------------------------------------------------ envelope

def cpu_load_check(tag: str) -> dict:
    """Owner envelope: CPU-only cell; log load + RAM before each burst."""
    if psutil is None:
        return {"tag": tag, "psutil": "absent"}
    rec = {"tag": tag,
           "cpu_percent": psutil.cpu_percent(interval=0.5),
           "ram_avail_gb": round(psutil.virtual_memory().available / 2**30, 1),
           "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 1)}
    log(f"  [load] {tag}: cpu {rec['cpu_percent']}% "
        f"ram_avail {rec['ram_avail_gb']} GB")
    return rec


# ------------------------------------------------------------ gradient core

def flat_grad(net) -> torch.Tensor:
    """Concatenate all parameter grads (fixed parameter order) fp32."""
    parts = [p.grad.reshape(-1) for p in net.parameters()
             if p.grad is not None]
    return torch.cat(parts)


def add_flat(net, base_sds, vec: torch.Tensor, eps: float):
    """W <- base + eps * vec (in place on net; exact restore by re-copying
    base). e204's load_flat convention, vectorized."""
    ofs = 0
    with torch.no_grad():
        for p, b in zip(net.parameters(), base_sds):
            n = p.numel()
            p.copy_(b.reshape(p.shape)
                    + (eps * vec[ofs:ofs + n]).reshape(p.shape).to(
                        b.dtype))
            ofs += n


def restore_flat(net, base_sds):
    with torch.no_grad():
        for p, b in zip(net.parameters(), base_sds):
            p.copy_(b)


def probe_support(net, probe: dict) -> tuple[torch.Tensor, float]:
    """s = grad p(ans | prompt) over all params, L2-normalized fp32;
    returns (direction, p). Batch-1, eval, deterministic CPU path."""
    net.eval()
    net.zero_grad(set_to_none=True)
    logits = net(input_ids=probe["ids"]).logits
    pvec = F.softmax(logits[0, -1], dim=-1)
    p0 = float(pvec[probe["ans_id"]].item())
    pvec[probe["ans_id"]].backward()
    g = flat_grad(net)
    nrm = float(g.norm().item())
    if not (nrm > 0 and math.isfinite(nrm)):
        raise RuntimeError(f"degenerate support for {probe['fact']}")
    net.zero_grad(set_to_none=True)
    return g / nrm, p0


def wash_batch(train_ids, off: torch.Tensor):
    """e182's VERBATIM batch construction from window offsets."""
    seq = e1.SEQ
    x = torch.stack([train_ids[o: o + seq] for o in off])
    y = torch.stack([train_ids[o + 1: o + 1 + seq] for o in off])
    return x, y


def wash_grad(net, x, y) -> tuple[torch.Tensor, float]:
    """g = grad of the wash's OWN loss (mean CE over the batch) at the
    CURRENT weights; returns (normalized direction, batch CE)."""
    net.eval()
    net.zero_grad(set_to_none=True)
    logits = net(input_ids=x).logits
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                         y.reshape(-1))
    ce.backward()
    g = flat_grad(net)
    nrm = float(g.norm().item())
    if not (nrm > 0 and math.isfinite(nrm)):
        raise RuntimeError("degenerate wash gradient")
    net.zero_grad(set_to_none=True)
    return g / nrm, float(ce.item())


@torch.no_grad()
def wash_ce(net, x, y) -> float:
    net.eval()
    logits = net(input_ids=x).logits
    return float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 y.reshape(-1)).item())


@torch.no_grad()
def probe_p(net, probe: dict) -> float:
    lg = net(input_ids=probe["ids"]).logits[0, -1]
    return float(F.softmax(lg, -1)[probe["ans_id"]].item())


def topk_idx(v: torch.Tensor, k: int) -> set:
    """g11's VERBATIM convention: indices of the k largest |coords|."""
    return set(torch.topk(v.abs(), k).indices.tolist())


def overlap_frac(a: set, b: set, k: int) -> float:
    return len(a & b) / k


# ------------------------------------------------------------ cache math

CHUNK = 8_388_608          # 2^23 coords: 16 MB fp16 / 32 MB fp32 chunks


def cache_row_norms(cache: torch.Tensor) -> torch.Tensor:
    """fp64 norms of the (fp16) cache rows — the cos denominators."""
    out = torch.zeros(cache.shape[0], dtype=torch.float64)
    for s in range(0, cache.shape[1], CHUNK):
        sl = slice(s, min(s + CHUNK, cache.shape[1]))
        out += cache[:, sl].to(torch.float32).pow(2).sum(1).to(torch.float64)
    return out.sqrt()


def dot_vec_rows(vec: torch.Tensor, cache: torch.Tensor,
                 rows: list[int]) -> list[float]:
    """dot(vec, cache[r]) fp32-chunked, fp64-accumulated (raw dots).
    Column-slice FIRST (a view), then row-select — never copies whole
    rows."""
    out = [0.0] * len(rows)
    rws = torch.tensor(rows, dtype=torch.long)
    for s in range(0, cache.shape[1], CHUNK):
        sl = slice(s, min(s + CHUNK, cache.shape[1]))
        C = cache[:, sl].index_select(0, rws).to(torch.float32)
        v = vec[sl]
        d = (C @ v).to(torch.float64).tolist()
        out = [a + b for a, b in zip(out, d)]
    return out


def gram_cache(cache: torch.Tensor) -> torch.Tensor:
    """full Gram (raw dots) in one cache pass, fp64-accumulated."""
    n = cache.shape[0]
    G = torch.zeros(n, n, dtype=torch.float64)
    for s in range(0, cache.shape[1], CHUNK):
        sl = slice(s, min(s + CHUNK, cache.shape[1]))
        C = cache[:, sl].to(torch.float32)
        G += (C @ C.T).to(torch.float64)
    return G


# ------------------------------------------------------------ plot

def make_plot(rd, align, rot_flat, hot_flat, bands, verdict, zmax):
    """THE ANCHOR FIGURE: three wash panels (Gmail vs iPhone alignment
    curves, the product-family 2-sigma band shaded) + the rotation/hot
    panel. The anchor pair against its family noise."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    washes = ("w1", "w2", "w3")
    titles = {"w1": "WASH 1 (e182c replay)", "w2": "WASH 2 (e182c2 fresh)",
              "w3": "WASH 3 (e217 fresh)"}
    for ax, w in zip([axes[0][0], axes[0][1], axes[1][0]], washes):
        sts = sorted(int(s) for s in align[w])
        g = [align[w][str(s)][ANCHOR_G] for s in sts]
        i = [align[w][str(s)][ANCHOR_I] for s in sts]
        lo = [bands[w][str(s)]["align_mu"] - 2 * bands[w][str(s)]["align_sd"]
              for s in sts]
        hi = [bands[w][str(s)]["align_mu"] + 2 * bands[w][str(s)]["align_sd"]
              for s in sts]
        ax.fill_between(sts, lo, hi, color="0.82", zorder=0,
                        label="product family ±2σ (n=5, excl. anchors)")
        ax.plot(sts, g, "o-", color="#1a6faf", lw=2, ms=6, label="Gmail (HOLDS)")
        ax.plot(sts, i, "s--", color="#c0392b", lw=2, ms=6,
                label="iPhone (DIES)")
        ax.axhline(0.0, color="0.55", lw=0.7, ls=":")
        ax.set_title(titles[w] + " — wash-gradient vs support alignment",
                     fontsize=10)
        ax.set_xlabel("wash step (archived state)")
        ax.set_ylabel("cos(g_wash(w,s), s_probe(0))")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    ax = axes[1][1]
    xs = list(range(3))
    gw = [rot_flat[w][ANCHOR_G] for w in washes]
    iw = [rot_flat[w][ANCHOR_I] for w in washes]
    rlo = [bands[w]["rot_mu"] - 2 * bands[w]["rot_sd"] for w in washes]
    rhi = [bands[w]["rot_mu"] + 2 * bands[w]["rot_sd"] for w in washes]
    ax.fill_between(xs, rlo, rhi, color="0.82", zorder=0,
                    label="family ±2σ (rotation)")
    ax.plot(xs, gw, "o-", color="#1a6faf", lw=2, ms=7, label="Gmail R(cos)")
    ax.plot(xs, iw, "s--", color="#c0392b", lw=2, ms=7, label="iPhone R(cos)")
    ax2 = ax.twinx()
    hg = [hot_flat[w][ANCHOR_G] for w in washes]
    hiw = [hot_flat[w][ANCHOR_I] for w in washes]
    hlo = [bands[w]["hot_mu"] - 2 * bands[w]["hot_sd"] for w in washes]
    hhi = [bands[w]["hot_mu"] + 2 * bands[w]["hot_sd"] for w in washes]
    ax2.fill_between(xs, hlo, hhi, color="#ffe9b8", zorder=0, alpha=0.85)
    ax2.plot(xs, hg, "o:", color="#0d5c3f", lw=1.6, ms=6,
             label="Gmail hot-overlap H")
    ax2.plot(xs, hiw, "s:", color="#8e44ad", lw=1.6, ms=6,
             label="iPhone hot-overlap H")
    ax2.set_ylabel("H = top2000 overlap frac (dotted)", fontsize=8)
    ax2.tick_params(labelsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"w{k+1} @ +{ROT_STATE}" for k in xs], fontsize=8)
    ax.set_ylabel("R = cos(s(settled), s(0))")
    ax.set_title(f"THE ROTATION READ at +{ROT_STATE} (bands shaded)",
                 fontsize=10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left")
    ax.grid(alpha=0.25)
    wrap = textwrap.fill(verdict, 96)
    fig.suptitle("E226 — THE GMAIL/IPHONE INTERIOR (FQ5): where does the "
                 "wash differentiate the anchor pair?\n" + wrap,
                 fontsize=11.5)
    fig.text(0.5, 0.012,
             f"max |Z| = {zmax:.2f} (bar 2σ) — band = product family n=5 "
             f"non-anchor probes, ±2σ ddof=1 per read",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.025, 1, 0.93))
    png = rd / f"{NAME}_anchor.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


# ------------------------------------------------------------ main

def main():
    rd = run_dir(NAME)
    jp = rd / "journal.json"
    log(f"E226 — THE GMAIL/IPHONE INTERIOR (smoke={SMOKE}) -> {rd}")

    metrics = {
        "experiment": "e226_interior",
        "phase": "eval-only on the THREE-wash 124M archive (CPU fp32 "
                 "batch-1 gradient geometry; supports / wash-gradient "
                 "alignment / rotation)",
        "date": now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED_PREDICTION["registration"],
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("FQ5: WHERE does the organism differentiate Gmail "
                     "(holds) and iPhone (dies) during the wash — the "
                     "support-gradient interior — and does the first-order "
                     "instrument see the differentiation at all?"),
        "builds_on": [
            "scratch/fq5_design.md (the ripened design, frozen at dispatch)",
            "T197 / e221 (the hunt closed at REAL-AND-UNNAMED; WHAT "
            "retired — the question moved to WHERE)",
            "T195 / e220 + T193 / e219 + T190-T191 / e216-e218 (the anchor "
            "pair at its sharpest: token-silent, not-entrenchment, "
            "beyond-height residual)",
            "T196 / g11 (the rotated-support x shared-hot-set geometry; "
            "the top-k overlap convention this cell ports to 124M)",
            "T149 / e182c + T183 / e182c2 + T192 / e217 (the three-wash "
            "archive + the batteries, VERBATIM by module import)",
            "e204 (the fact-gradient support + FD gate convention)",
        ],
        "whats_new": [
            "the per-probe SUPPORT DIRECTION at 124M: grad p(answer|context) "
            "over all 124.4M params, all 54 probes, t=0 pairwise Gram",
            "the wash gradient at every archived state decomposed against "
            "EVERY probe's support (the matched-point alignment curves, "
            "n=3 washes — the wash's treatment read per relation)",
            "the per-probe ROTATION read at +80 on three washes (cos to "
            "own t=0 support + g11's hot-set overlap at 124M)",
            "the anchor adjudication: Gmail vs iPhone against the product "
            "family's 2-sigma bands — the interior version of the "
            "WHAT-hunt, bars frozen before compute",
        ],
        "smoke": SMOKE,
    }

    def write_metrics(status):
        metrics["status"] = status
        metrics["timing"] = {"total_s": round(time.time() - T0, 1)}
        save_json(rd / "metrics.json", metrics)

    journal: dict = {}
    if jp.exists():
        try:
            journal = json.loads(jp.read_text(encoding="utf-8"))
            log(f"journal restored: {list(journal)}")
        except Exception as e:                              # noqa: BLE001
            log(f"journal unreadable ({e}); starting fresh")
            journal = {}

    def save_journal():
        jp.write_text(json.dumps(journal, indent=1, default=float),
                      encoding="utf-8")

    load_checks: list[dict] = [cpu_load_check("launch")]
    metrics["load_checks"] = load_checks

    # ------------------------------------------------ P0 the committed records
    paths = [E182C_M, E182C2_M, E182C2_J, E217_M, E216_M, E219_M, E220_M]
    for p in paths:
        if not p.exists():
            log(f"FATAL: committed record missing: {p}")
            return 1
    p1m = json.loads(E182C_M.read_text(encoding="utf-8"))
    c2m = json.loads(E182C2_M.read_text(encoding="utf-8"))
    j2 = json.loads(E182C2_J.read_text(encoding="utf-8"))
    u3m = json.loads(E217_M.read_text(encoding="utf-8"))
    e216 = json.loads(E216_M.read_text(encoding="utf-8"))
    w1_rec = {s["step"]: s for s in p1m["states"]}          # fact/ctrl/near
    w1_tmpl_rec = {s["step"]: s for s in c2m["part1_template"]["states"]}
    w2_rec = {s["step"]: s for s in j2["states"]}
    w3_rec = {s["step"]: s for s in u3m["wash3_states"]}
    for w, sts in (("w1", W_STATES["w1"]), ("w2", W_STATES["w2"]),
                   ("w3", W_STATES["w3"])):
        for s_ in sts:
            if s_ == 0:
                continue
            assert s_ in W_ARCH[w] and W_ARCH[w][s_].exists(), \
                f"archive gap: {w} +{s_}"
            recs = {"w1": (w1_rec, w1_tmpl_rec), "w2": (w2_rec, None),
                    "w3": (w3_rec, None)}[w]
            assert s_ in recs[0] and (w != "w1" or s_ in w1_tmpl_rec)
    e216_tab = e216["residual"]["table"]
    assert len(e216_tab) == 54
    e216_rows = {r["fact"]: r for r in e216_tab}
    assert ANCHOR_G in e216_rows and ANCHOR_I in e216_rows
    log(f"records: e216 54-probe table ok; anchor resid_w1 "
        f"{e216_rows[ANCHOR_G]['resid_w1']:+.3f} / "
        f"{e216_rows[ANCHOR_I]['resid_w1']:+.3f}")

    # the anchors' fates on all THREE washes (committed, load-only)
    def _probes(rec, step, batt):
        return {f: v["p"] for f, v in rec[step][batt]["probes"].items()}
    anchor_fates = {}
    for tag, fkey in (("Gmail", ANCHOR_G), ("iPhone", ANCHOR_I)):
        anchor_fates[tag] = {}
        for w, recs in (("w1", w1_rec), ("w2", w2_rec), ("w3", w3_rec)):
            p0c, s80 = _probes(recs, 0, "ctrl"), _probes(recs, 80, "ctrl")
            anchor_fates[tag][w] = {"p0": p0c[fkey], "p80": s80[fkey],
                                    "hr": s80[fkey] / p0c[fkey]}
    metrics["anchor_fates_committed"] = anchor_fates
    log("anchor fates (committed): " + " | ".join(
        f"{t}: " + ", ".join(f"{w} hr {v['hr']:.3f}"
                             for w, v in anchor_fates[t].items())
        for t in anchor_fates))

    # ------------------------------------------------ P1 organism + envelope
    tok, net0, org_meta = e1.load_organism()
    metrics["organism"] = {**org_meta, "torch_threads": THREADS}
    G_SIZE = {"params": org_meta["params"], "ceiling": e1.SIZE_CEILING,
              "reason": e1.SIZE_REASON,
              "pass": bool(org_meta["params"] <= e1.SIZE_CEILING)}
    assert G_SIZE["pass"]
    G_ENV = {"device": "cpu", "threads": THREADS, "gpu_used": False,
             "pass": True,
             "note": "eval-only; batch-1 probe backwards + 8x512 batch "
                     "backwards; load checks per phase"}
    metrics["gates"] = {"G_SIZE": G_SIZE, "G_ENV": G_ENV}
    metrics["size_gate"] = G_SIZE
    write_metrics("PARTIAL: records read; organism loaded")

    # param order contract (every gradient shares ONE flattening order)
    P_NAMES = [n for n, _ in net0.named_parameters()]
    N_PARAMS = sum(p.numel() for p in net0.parameters())
    assert N_PARAMS == org_meta["params"]
    log(f"param order frozen: {len(P_NAMES)} tensors, {N_PARAMS:,} coords")

    # ------------------------------------------------ P2 corpus (G_CORPUS)
    text = (common.REPO / "data" / "input.txt").read_text(encoding="utf-8")
    e182 = json.loads(e1.E182_METRICS.read_text(encoding="utf-8"))
    e_corp = e182["corpus"]
    e_str = e182["gates"]["G_STR"]
    e_banned = e_str["banned"]
    banned = sorted({s.lower() for rel in e1.POOLS
                     for s, _ in e1.POOLS[rel]}
                    | {a.lower() for rel in e1.POOLS
                       for _, a in e1.POOLS[rel]}
                    | set(e1.BANNED_EXTRA))
    assert banned == e_banned, "banned list diverged from e182's record"
    cand, _dropped = e1.build_candidates(tok)
    for r, b in zip(cand, e1.probe_battery(net0, cand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(cand)
    battery = [r for r in cand if r["kept"]]
    answer_ids = {r["fact"]: r["ans_id"] for r in battery}
    train_ids, _bank_xy, filtered, G_STR, _G_TOK, corpus_stats = \
        e1.build_wash_corpus(tok, text, banned, answer_ids)
    G_CORPUS = {
        "banned_list_identical": True,
        "tokens_after": [corpus_stats["tokens_after"],
                         e_corp["tokens_after"]],
        "train_tokens": [corpus_stats["train_tokens"],
                         e_corp["train_tokens"]],
        "format": "[rebuilt, e182_recorded]",
        "note": "the corpus is INHERITED FROZEN — it feeds the battery "
                "contamination scans and the wash-batch reproduction "
                "(the SAME train_ids every wash drew its windows from); "
                "no wash is run here",
    }
    G_CORPUS["pass"] = bool(
        G_STR["lines_total"] == e_str["lines_total"]
        and G_STR["lines_dropped"] == e_str["lines_dropped"]
        and corpus_stats["chars_after"] == e_corp["chars_after"]
        and corpus_stats["tokens_after"] == e_corp["tokens_after"]
        and corpus_stats["train_tokens"] == e_corp["train_tokens"])
    log(f"G_CORPUS: {'PASS' if G_CORPUS['pass'] else 'FAIL'} "
        f"({corpus_stats['tokens_after']} tokens)")
    assert G_CORPUS["pass"] or SMOKE
    metrics["gates"]["G_CORPUS"] = G_CORPUS
    write_metrics("PARTIAL: corpus certified")

    # --------------------------------- P3 the four batteries, VERBATIM (G_BATT)
    ccand, _cd = e1.build_control_candidates(
        tok, filtered.lower(), train_ids, e_banned,
        e1.CTRL_POOLS, e1.CTRL_TMPL)
    for r, b in zip(ccand, e1.probe_battery(net0, ccand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(ccand)
    cbattery = [r for r in ccand if r["kept"]]

    ncand, _nd = e1.build_control_candidates(
        tok, filtered.lower(), train_ids, e_banned,
        {"near": e1.NEAR_POOL}, {"near": e1.NEAR_TMPL})
    for r, b in zip(ncand, e1.probe_battery(net0, ncand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(ncand)
    nbattery = [r for r in ncand if r["kept"]]

    tcand, _td = e2.build_tmpl_candidates(tok, filtered.lower(), train_ids,
                                          e_banned)
    for r, b in zip(tcand, e1.probe_battery(net0, tcand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(tcand)
    tbattery = [r for r in tcand if r["kept"]]

    bats = {"fact": battery, "ctrl": cbattery,
            "near": nbattery, "tmpl": tbattery}
    G_BATT = {}
    for b, bl in bats.items():
        mine = [r["fact"] for r in bl]
        mine_p = {r["fact"]: r["p"] for r in bl}
        ref1 = _probes(w1_rec[0], 0, b) if b != "tmpl" \
            else _probes(w1_tmpl_rec[0], 0, "tmpl")
        ref2 = _probes(w2_rec[0], 0, b)
        ref3 = _probes(w3_rec[0], 0, b)
        G_BATT[b] = {
            "n": len(bl),
            "order_equal_e216":
                mine == [r["fact"] for r in e216_tab if r["battery"] == b],
            "set_equal_committed": bool(set(mine) == set(ref1)
                                        == set(ref2) == set(ref3)),
            "max_dp_w1": max(abs(mine_p[f] - v) for f, v in ref1.items()),
            "max_dp_w2": max(abs(mine_p[f] - v) for f, v in ref2.items()),
            "max_dp_w3": max(abs(mine_p[f] - v) for f, v in ref3.items()),
        }
    G_BATT["tol_per_probe_dp"] = TOL_T0_DP
    G_BATT["pass"] = bool(
        all(G_BATT[b]["set_equal_committed"]
            and max(G_BATT[b]["max_dp_w1"], G_BATT[b]["max_dp_w2"],
                    G_BATT[b]["max_dp_w3"]) <= TOL_T0_DP
            for b in bats)) if not SMOKE else True
    G_BATT["note"] = ("batteries = the phase-1/phase-2 pools VERBATIM "
                      "(module import); t=0 must reproduce all THREE "
                      "committed records; the e216 order check is "
                      "informational (facts are joined by string, "
                      "order-independent)")
    metrics["gates"]["G_BATT"] = G_BATT
    log("G_BATT: " + ("PASS" if G_BATT["pass"] else "FAIL") + " | "
        + " | ".join(f"{b}: n {G_BATT[b]['n']} dp "
                     f"{max(G_BATT[b]['max_dp_w1'], G_BATT[b]['max_dp_w2'], G_BATT[b]['max_dp_w3']):.2e} "
                     f"ord {G_BATT[b]['order_equal_e216']}" for b in bats))
    if not G_BATT["pass"]:
        write_metrics("PARTIAL: G_BATT FAILED — halted before any gradient "
                      "compute")
        return 1

    # THE 54 (e216's committed assignment, order = battery construction)
    probes54: list[dict] = []
    for b in ("fact", "ctrl", "near", "tmpl"):
        for r in bats[b]:
            row = e216_rows[r["fact"]]
            probes54.append({**r, "battery": b, "family6": row["family6"],
                             "p0_e216": row["p0"],
                             "resid_w1_e216": row["resid_w1"],
                             "resid_w2_e216": row["resid_w2"]})
    assert len(probes54) == 54
    if SMOKE:
        keep = {ANCHOR_G, ANCHOR_I,
                "The gaming console made by Microsoft->Xbox",
                "The web browser made by Google->Chrome",
                "The tablet made by Apple->iPad",
                "The music store made by Apple->iTunes",
                "The console made by Sony->PlayStation",
                "The social network founded by Mark Zuckerberg->Facebook",
                "France->Paris", "Massachusetts->Boston",
                "Boston->Massachusetts"}
        probes54 = [r for r in probes54 if r["fact"] in keep]
        log(f"SMOKE: probe subset n={len(probes54)}")
    n_p = len(probes54)
    P_IDX = {r["fact"]: i for i, r in enumerate(probes54)}
    metrics["probes"] = [{"i": i, "fact": r["fact"], "battery": r["battery"],
                          "family6": r["family6"], "p0": r["p"]}
                         for i, r in enumerate(probes54)]
    band_idx = [i for i, r in enumerate(probes54)
                if r["family6"] == BAND_FAMILY
                and r["fact"] not in (ANCHOR_G, ANCHOR_I)]
    founder_idx = [i for i, r in enumerate(probes54)
                   if r["family6"] == "founder-anchor"]
    log(f"probes: {n_p} (band n={len(band_idx)} product non-anchor; "
        f"founder n={len(founder_idx)})")
    write_metrics("PARTIAL: batteries rebuilt + certified")

    # --------------------------------------------- P4 the draw streams (G_DRAWS)
    load_checks.append(cpu_load_check("draws"))
    hi = train_ids.shape[0] - e1.SEQ - 1
    G_DRAWS = {}
    batch_off = {}                     # (wash, state) -> offsets draw #(s+1)
    for w in WASHES:
        gen = torch.Generator().manual_seed(W_SEED[w])
        offs = []
        for step in range(1, 82):      # through draw #81 (state 80's next)
            offs.append(torch.randint(hi, (e1.BATCH,), generator=gen))
            if step == 80:
                archived = torch.load(W_LATEST[w], map_location=CPU,
                                      weights_only=False)["gen"]
                G_DRAWS[w] = {
                    "seed": W_SEED[w],
                    "gen_state_identical_after_80":
                        bool(torch.equal(gen.get_state(), archived)),
                }
        for s_ in W_STATES[w]:
            if s_ > 0:
                batch_off[(w, s_)] = offs[s_]      # draw #(s+1), 0-indexed s
    G_DRAWS["pass"] = bool(all(v["gen_state_identical_after_80"]
                               for k, v in G_DRAWS.items()
                               if isinstance(v, dict) and "seed" in v)) \
        if not SMOKE else True
    G_DRAWS["note"] = ("each wash's window-draw stream reproduced from "
                       "its archived seed and certified BIT-EXACTLY "
                       "against the generator state archived at step 80 "
                       "in its own *_latest.pt — the batches used here "
                       "are the states' OWN next batches (draw #(s+1))")
    metrics["gates"]["G_DRAWS"] = G_DRAWS
    log("G_DRAWS: " + ("PASS" if G_DRAWS["pass"] else "FAIL")
        + " | " + ", ".join(f"{w}: "
                            f"{G_DRAWS[w]['gen_state_identical_after_80']}"
                            for w in WASHES))
    assert G_DRAWS["pass"] or SMOKE

    def inv(p: Path):
        return {"path": str(p), "exists": p.exists(),
                "size_bytes": p.stat().st_size if p.exists() else None,
                "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(
                    p.stat().st_mtime)) if p.exists() else None}
    metrics["inventory"] = {
        "states": {w: {str(s): inv(W_ARCH[w][s])
                       for s in W_STATES[w] if s > 0} for w in WASHES},
        "latest": {w: inv(W_LATEST[w]) for w in WASHES},
        "note": "three washes: w1 CPU-fp32 replay of e182's own wash; "
                "w2/w3 fresh draws (GPU fp32 in their cells); all state "
                "weights read here on CPU fp32",
    }
    write_metrics("PARTIAL: draw streams certified")

    # --------------------------------------------- P5 THE t=0 SUPPORTS (cache)
    load_checks.append(cpu_load_check("supports t=0"))
    if psutil is not None and \
            psutil.virtual_memory().available / 2**30 < RAM_FLOOR_GB:
        log(f"FATAL: RAM available below {RAM_FLOOR_GB} GB floor — cache "
            "unsafe on the shared box")
        write_metrics("PARTIAL: RAM floor hit at cache alloc — HALT")
        return 1
    cache = torch.empty((n_p, N_PARAMS), dtype=torch.float16)
    fd_records = []
    netFD = copy.deepcopy(net0)
    base_sds = [p.detach().clone() for p in netFD.parameters()]
    assert [n for n, _ in netFD.named_parameters()] == P_NAMES
    for i, pr in enumerate(probes54):
        s_i, p0 = probe_support(net0, pr)
        cache[i] = s_i.to(torch.float16)
        # FD gate: p(theta0 + eps*s_hat) must RISE (e204's convention)
        fd = {}
        for eps in FD_EPS:
            add_flat(netFD, base_sds, s_i, eps)
            fd[str(eps)] = probe_p(netFD, pr) - p0
            restore_flat(netFD, base_sds)
        fd_records.append({"fact": pr["fact"], "p0": p0, "fd_dp": fd,
                           "fd_pass": bool(fd[str(FD_EPS[0])] > 0)})
        log(f"  s[{i:2d}] {pr['fact'][:44]:<44} p0 {p0:.4f} "
            f"fd {fd[str(FD_EPS[0])]:+.2e}")
    del netFD, base_sds
    journal["supports_t0"] = fd_records
    save_journal()
    row_norms = cache_row_norms(cache).tolist()
    G_SUPPORT = {
        "fd_eps": list(FD_EPS),
        "rule": "p(theta0 + eps*s_hat) - p0 > 0 at the primary eps "
                "(e204's directional gate, ported to 124M)",
        "per_probe_fd": {r["fact"]: r["fd_dp"] for r in fd_records},
        "n_fail": sum(1 for r in fd_records if not r["fd_pass"]),
    }
    # determinism re-verify: recompute probe 0 fresh, cos vs cached row
    s_rep, _ = probe_support(net0, probes54[0])
    d_rep = dot_vec_rows(s_rep, cache, [0])[0] / row_norms[0]
    G_SUPPORT["determinism_selfcos_probe0"] = d_rep
    G_SUPPORT["pass"] = bool(G_SUPPORT["n_fail"] == 0 and d_rep > 0.999) \
        if not SMOKE else True
    metrics["gates"]["G_SUPPORT"] = G_SUPPORT
    log(f"G_SUPPORT: {'PASS' if G_SUPPORT['pass'] else 'FAIL'} "
        f"(fd fails {G_SUPPORT['n_fail']}; probe-0 recompute self-cos "
        f"{d_rep:.6f})")
    assert G_SUPPORT["pass"] or SMOKE
    write_metrics("PARTIAL: t=0 supports cached + FD-gated")

    # the pairwise geometry (the 54x54 Gram + the anchor read)
    G = gram_cache(cache).tolist()
    cos_ij = [[G[i][j] / (row_norms[i] * row_norms[j]) for j in range(n_p)]
              for i in range(n_p)]
    gi, ii = P_IDX[ANCHOR_G], P_IDX[ANCHOR_I]
    prod_all = [i for i, r in enumerate(probes54)
                if r["family6"] == BAND_FAMILY]
    fam_pair_cos = sorted(cos_ij[a][b] for x, a in enumerate(prod_all)
                          for b in prod_all[x + 1:])
    mu_f = sum(fam_pair_cos) / len(fam_pair_cos)
    sd_f = (sum((x - mu_f) ** 2 for x in fam_pair_cos)
            / (len(fam_pair_cos) - 1)) ** 0.5
    fam_summary = {}
    for fam in sorted({r["family6"] for r in probes54}):
        idxs = [i for i, r in enumerate(probes54) if r["family6"] == fam]
        pcs = sorted(cos_ij[a][b] for x, a in enumerate(idxs)
                     for b in idxs[x + 1:])
        if pcs:
            fam_summary[fam] = {"n": len(idxs),
                                "within_mean_cos": sum(pcs) / len(pcs),
                                "within_min": pcs[0], "within_max": pcs[-1]}
    metrics["support_geometry_t0"] = {
        "gmail_iphone_cos": cos_ij[gi][ii],
        "product_family_pairwise": {"n_pairs": len(fam_pair_cos),
                                    "mean": mu_f, "sd": sd_f,
                                    "min": fam_pair_cos[0],
                                    "max": fam_pair_cos[-1],
                                    "gmail_iphone_z_within_family":
                                        (cos_ij[gi][ii] - mu_f) / sd_f},
        "family_within_summary": fam_summary,
        "cache": "fp16 rows of L2-normalized fp32 grads; dots "
                 "fp32-chunked, fp64-accumulated; cos denominators = "
                 "fp64 row norms (cache rounding ~5e-4, disclosed)",
    }
    log(f"GEOMETRY t=0: cos(Gmail, iPhone) = {cos_ij[gi][ii]:+.4f} "
        f"[product family pairwise mean {mu_f:+.4f} sd {sd_f:.4f}]")
    write_metrics("PARTIAL: t=0 pairwise geometry done")

    # --------------------------------------------- P6 the state loop
    align: dict = journal.get("align", {})
    rot: dict = journal.get("rotation", {})
    state_gates: dict = journal.get("state_gates", {})
    hot_journal: dict = journal.get("hot_sets", {})
    hot_sets: dict[str, dict[int, set]] = {
        w: {int(k): set(v) for k, v in hot_journal[w].items()}
        for w in hot_journal}

    for w in WASHES:
        for s_ in W_STATES[w]:
            key = f"{w}:{s_}"
            rot_ready = (s_ != ROT_STATE) or (w in rot)
            if key in state_gates and key in align and rot_ready:
                continue
            load_checks.append(cpu_load_check(f"{w}s{s_}"))
            # load the state (s_=0 -> pristine net0)
            if s_ > 0:
                sd = torch.load(W_ARCH[w][s_], map_location=CPU,
                                weights_only=False)["model"]
                net = copy.deepcopy(net0)
                net.load_state_dict(sd)
                del sd
            else:
                net = copy.deepcopy(net0)
            # (a) re-probe the four batteries vs committed (G_STATES)
            dps = {}
            for b, bl in bats.items():
                pp = {r["fact"]: probe_p(net, r) for r in bl}
                if w == "w1":
                    rec = (w1_rec[s_] if b != "tmpl"
                           else w1_tmpl_rec[s_])
                    ref = _probes(rec, s_, b if b != "tmpl" else "tmpl")
                else:
                    ref = _probes({"w2": w2_rec, "w3": w3_rec}[w], s_, b)
                dps[b] = max(abs(pp[f] - v) for f, v in ref.items())
            # (b) the wash's own next-batch gradient + FD gate
            x, y = wash_batch(train_ids, batch_off[(w, s_)])
            g_hat, ce0 = wash_grad(net, x, y)
            base_sds = [p.detach().clone() for p in net.parameters()]
            add_flat(net, base_sds, g_hat, FD_EPS[0])
            ce_fd = wash_ce(net, x, y)
            restore_flat(net, base_sds)
            del base_sds
            gfd_pass = ce_fd < ce0
            # (c) alignment dots vs the t=0 supports
            dots = dot_vec_rows(g_hat, cache, list(range(n_p)))
            cosr = {probes54[i]["fact"]: dots[i] / row_norms[i]
                    for i in range(n_p)}
            align.setdefault(w, {})[str(s_)] = cosr
            state_gates[key] = {
                "reprobe_max_dp": dps, "batch_ce": ce0,
                "fd_dce": ce_fd - ce0, "fd_pass": bool(gfd_pass),
                "g_convention": "direction L2-normalized; matched point "
                                "= the state's own next batch (draw #(s+1))",
            }
            if s_ == 0 and w not in hot_sets:
                hot_sets[w] = {k: topk_idx(g_hat, k) for k in K_LADDER}
            journal["align"] = align
            journal["state_gates"] = state_gates
            journal["hot_sets"] = {w: {str(k): sorted(v)
                                       for k, v in hh.items()}
                                   for w, hh in hot_sets.items()}
            save_journal()
            log(f"  {w} +{s_:2d}: CE {ce0:.4f} fd {ce_fd - ce0:+.2e} "
                f"cosG {cosr[ANCHOR_G]:+.4f} cosI {cosr[ANCHOR_I]:+.4f} "
                f"dp {max(dps.values()):.2e}")
            del g_hat
            write_metrics(f"PARTIAL: state loop through {w} +{s_}")

            # (d) the rotation read at the settled state
            if s_ == ROT_STATE and w not in rot:
                rot_rec = {}
                for i, pr in enumerate(probes54):
                    s_i, p_s = probe_support(net, pr)
                    d_self = dot_vec_rows(s_i, cache, [i])[0] / row_norms[i]
                    tk = {k: topk_idx(s_i, k) for k in K_LADDER}
                    ovl = {str(k): overlap_frac(tk[k], hot_sets[w][k], k)
                           for k in K_LADDER}
                    fd_dp = None
                    if pr["fact"] in (ANCHOR_G, ANCHOR_I) and not SMOKE:
                        base2 = [q.detach().clone()
                                 for q in net.parameters()]
                        add_flat(net, base2, s_i, FD_EPS[0])
                        fd_dp = probe_p(net, pr) - p_s
                        restore_flat(net, base2)
                        del base2
                    rot_rec[pr["fact"]] = {
                        "self_cos": d_self, "hot_overlap": ovl,
                        "p_at_state": p_s, "anchor_fd_dp": fd_dp}
                rot[w] = rot_rec
                journal["rotation"] = rot
                save_journal()
                log(f"  {w} +{s_}: ROTATION read done "
                    f"(Gmail {rot_rec[ANCHOR_G]['self_cos']:.4f} "
                    f"iPhone {rot_rec[ANCHOR_I]['self_cos']:.4f})")
                write_metrics(f"PARTIAL: rotation read through {w}")
            del net

    # G_STATES part B: re-probe dps across every loaded state
    all_dp = [max(v["reprobe_max_dp"].values())
              for v in state_gates.values()]
    G_STATES = {
        "tol_reprobe_dp": TOL_STATE_DP,
        "per_state": state_gates,
        "max_dp": max(all_dp) if all_dp else None,
        "pass": bool(all_dp and max(all_dp) <= TOL_STATE_DP)
        if not SMOKE else True,
        "note": "the archived states reproduce the committed per-probe "
                "records when re-probed through the VERBATIM batteries "
                "(e214's re-probe convention)",
    }
    metrics["gates"]["G_STATES"] = G_STATES
    metrics["gates"]["G_WASHGRAD"] = {
        "rule": "CE(W + eps*g_hat) < CE(W) at eps 0.02 — every state's "
                "wash gradient is a descent direction of its own batch "
                "loss; anchors' +80 supports FD co-gated (recorded in "
                "rotation records)",
        "per_state_fd": {k: v["fd_dce"] for k, v in state_gates.items()},
        "anchor_rot_fd": {w: {a: rot[w][a]["anchor_fd_dp"]
                              for a in (ANCHOR_G, ANCHOR_I)
                              if a in rot.get(w, {})} for w in WASHES},
        "pass": bool(all(v["fd_pass"] for v in state_gates.values()))
        if not SMOKE else True,
    }
    log(f"G_STATES: {'PASS' if G_STATES['pass'] else 'FAIL'} "
        f"(max re-probe dp {G_STATES['max_dp']:.2e}) | G_WASHGRAD: "
        f"{'PASS' if metrics['gates']['G_WASHGRAD']['pass'] else 'FAIL'}")

    # --------------------------------------------- P7 the curves + adjudication
    def band(vals_idx, val_of):
        xs = [val_of(i) for i in vals_idx]
        return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0)

    bands_out = {}
    z_reads = []
    for w in WASHES:
        bands_out[w] = {}
        for s_ in W_STATES[w]:
            kk = str(s_)
            mu, sd = band(band_idx,
                          lambda i: align[w][kk][probes54[i]["fact"]])
            bands_out[w][kk] = {"align_mu": mu, "align_sd": sd}
            zr = (abs(align[w][kk][ANCHOR_G] - align[w][kk][ANCHOR_I]) / sd
                  if sd > 0 else float("inf"))
            z_reads.append({"read": f"align {w} +{s_}", "z": zr,
                            "gmail": align[w][kk][ANCHOR_G],
                            "iphone": align[w][kk][ANCHOR_I],
                            "sigma": sd})
        rmu, rsd = band(band_idx,
                        lambda i: rot[w][probes54[i]["fact"]]["self_cos"])
        hmu, hsd = band(
            band_idx,
            lambda i: rot[w][probes54[i]["fact"]]["hot_overlap"][str(K_HOT)])
        bands_out[w].update({"rot_mu": rmu, "rot_sd": rsd,
                             "hot_mu": hmu, "hot_sd": hsd})
        z_reads.append({
            "read": f"rotation {w} +{ROT_STATE}",
            "z": abs(rot[w][ANCHOR_G]["self_cos"]
                     - rot[w][ANCHOR_I]["self_cos"]) / rsd
            if rsd > 0 else float("inf"),
            "gmail": rot[w][ANCHOR_G]["self_cos"],
            "iphone": rot[w][ANCHOR_I]["self_cos"], "sigma": rsd})
        z_reads.append({
            "read": f"hot-overlap {w} +{ROT_STATE}",
            "z": abs(rot[w][ANCHOR_G]["hot_overlap"][str(K_HOT)]
                     - rot[w][ANCHOR_I]["hot_overlap"][str(K_HOT)]) / hsd
            if hsd > 0 else float("inf"),
            "gmail": rot[w][ANCHOR_G]["hot_overlap"][str(K_HOT)],
            "iphone": rot[w][ANCHOR_I]["hot_overlap"][str(K_HOT)],
            "sigma": hsd})
        # founder-anchor co-report band (never adjudicated)
        fmu, fsd = band(founder_idx,
                       lambda i: rot[w][probes54[i]["fact"]]["self_cos"])
        bands_out[w]["founder_rot_mu"] = fmu
        bands_out[w]["founder_rot_sd"] = fsd

    n_firing = sum(1 for z in z_reads if z["z"] >= Z_BAR)
    max_z = max(z["z"] for z in z_reads)
    fired = [z for z in z_reads if z["z"] >= Z_BAR]
    verdict = ("SUPPORT-DIFFERENTIATES" if n_firing > 0
               else "GEOMETRY-IDENTICAL") \
        + (" (SMOKE — not adjudicated)" if SMOKE else "")

    metrics["curves"] = {
        "alignment": {w: dict(align[w]) for w in WASHES},
        "rotation": rot,
        "bands": bands_out,
        "z_reads": z_reads,
        "z_bar": Z_BAR,
        "n_reads": len(z_reads),
        "n_firing": n_firing,
        "hot_ladder_anchors": {
            w: {a: rot[w][a]["hot_overlap"] for a in (ANCHOR_G, ANCHOR_I)}
            for w in WASHES if w in rot},
    }
    metrics["adjudication"] = {
        "verdict": verdict,
        "bars_verbatim": REGISTERED_PREDICTION["bars_verbatim"],
        "fired_reads": fired,
        "max_z": max_z,
        "replication_note": (
            f"{n_firing} of {len(z_reads)} registered reads at Z >= "
            f"{Z_BAR}; firing reads: "
            f"{[z['read'] for z in fired]}"
            if fired else
            f"all {len(z_reads)} registered reads within the family "
            f"noise (max Z {max_z:.2f} < {Z_BAR})"),
        "gated_on": "G_ENV/G_SIZE/G_CORPUS/G_BATT/G_STATES/G_DRAWS/"
                    "G_SUPPORT/G_WASHGRAD",
        "all_gates_pass": bool(all(
            v.get("pass", True) for v in metrics["gates"].values()
            if isinstance(v, dict))),
    }

    metrics["honesty_reflex"] = {
        "n_washes": 3,
        "batch_noise": "supports are SINGLE-CONTEXT batch-1 gradients "
                       "(the probe's own 2-shot prompt; the design's "
                       "instrument); the wash gradient is a single 8x512 "
                       "batch (the state's own next batch — matched point)",
        "first_order_floor": "the instruments are first-order (gradients "
                             "at a point); g11's lesson licenses NO "
                             "extrapolation to realized-step damage — a "
                             "GEOMETRY-IDENTICAL verdict bounds the "
                             "differentiation to below the FIRST-ORDER "
                             "floor, not to nothing (the nonlinear "
                             "interaction stays open, by design)",
        "multiplicity": f"{len(z_reads)} registered reads; under a pure-"
                        "noise Gaussian null P(any Z>=2) ~= 0.59 — the "
                        "frozen bar fires on any read as written; the "
                        "replication count + per-read table carry the "
                        "robustness read (no bar shopping)",
        "cache_precision": "fp16 support cache (13.4 GB; shared box) — "
                           "cos rounding ~5e-4, two orders below plausible "
                           "family sigmas; determinism re-verified "
                           "(probe-0 recompute self-cos > 0.999)",
        "p0_difference": "Gmail p0 0.746 vs iPhone 0.636 — the anchors "
                         "differ in baseline strength; the instruments "
                         "here are direction-only (normalized), so level "
                         "differences enter only through the geometry "
                         "itself, disclosed",
        "states_provenance": "w2/w3 weights were computed on GPU fp32 in "
                             "their own cells; every gradient here is CPU "
                             "fp32 on the archived states",
    }

    # the plot + final write
    rot_flat = {w: {f: rot[w][f]["self_cos"] for f in rot[w]}
                for w in WASHES}
    hot_flat = {w: {f: rot[w][f]["hot_overlap"][str(K_HOT)]
                    for f in rot[w]} for w in WASHES}
    png = make_plot(rd, align, rot_flat, hot_flat, bands_out,
                    f"{verdict} — {n_firing}/{len(z_reads)} reads at "
                    f"Z>={Z_BAR:.0f}; max Z {max_z:.2f}", max_z)
    metrics["plot_outputs"] = [str(png)]
    metrics["compute"] = {
        "wall_s": round(time.time() - T0, 1),
        "device": "CPU only",
        "threads": THREADS,
        "backwards": f"{n_p} probe supports t=0 + {n_p}x{len(WASHES)} "
                     f"rotation + {len(state_gates)} wash-batch grads",
        "load_checks": len(load_checks),
    }
    metrics["trims"] = trims
    metrics["deviations"] = deviations
    status = ("SMOKE DONE" if SMOKE else
              ("DONE" if metrics["adjudication"]["all_gates_pass"]
               else "DONE (gate failures disclosed — see gates)"))
    write_metrics(status)
    log(f"VERDICT: {verdict} (max Z {max_z:.2f}, {n_firing}/"
        f"{len(z_reads)} reads firing)")
    log(f"outputs: {rd / 'metrics.json'}, {png}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
