"""E125a — THE INVERTED KNIFE: the site-stored fact's own kill set (e125
arm-1, R47 ideator top pick; QUEUE.md row e125a, DISPATCHED ~11:45Z; bars are
the QUEUE row VERBATIM, written before compute).

WHY (T090's flip-cell): e160 found the head-set {L1H0,L0H0} (N2) kills the
SINK-COUPLED fact at CE +0.25 (both modes) while the SITE-STORED fact
SURVIVES the same coordinates (<= 10.6% drop, e160's site control column) —
the knife is TYPE-SELECTIVE. The claim that would flip T090: does a
SYMMETRIC site-stored-killing head-set exist at flat CE AT ALL? If NO, the
two memory types differ in REMOVABILITY itself — asymmetric surgery, the
strongest form of the split-custody claim and arguably the deepest
unlearning fact available.

REGISTERED PREDICTION (QUEUE.md e125a row, VERBATIM — no bar shopping):
  "Bars: SITE-KNIFE-EXISTS = some head-set drops the site-fact >=60% at
  CE <= +0.35; NO-SITE-KNIFE = best <30% at CE <= +0.35 across the
  escalation (the types differ in REMOVABILITY — asymmetric surgery, the
  strongest form of the split-custody claim)"

OPERATIONALIZATIONS (fixed before compute):
  * PRIMARY NET = e131_arm_b_corpus_spliced.pt (the site-stored fact at
    address row 183; e133's pool-b battery = ZEPHYRA at x-col 184, onset
    read at row 183 — which is also arm_b's TRAINING geometry, e131 Phase A
    verbatim). Gates: std install-60 floor 0.007800613064318895, full-battery
    site onset 0.9880021214485168 (e131/e160 stored), CE_R 1.682660698890686.
  * SPLIT-WINDOW HYGIENE (the honesty fix for select-then-escalate on one
    net): the e133 site battery's 30 held-prompts are split in two —
    prompts[0:15] build the SELECTION battery (pool_sel, filler seed 12511)
    and prompts[15:30] build the ADJUDICATION battery (pool_bar, filler seed
    12512), 4 corpus fillers x 15 prompts each (60 windows per battery).
    The census ranks heads on pool_sel ONLY; the registered bar read is
    pool_bar ONLY (windows never touched during selection). The full
    30-prompt battery (pool_full, seed 12103 = e133/e160 verbatim) is used
    ONLY for stored-value continuity gates, never for selection or bars.
  * (A) CENSUS: per-head ZERO ablation on arm_b, 36 heads — site-expression
    drop (pool_sel) + CE_R cost per head (e133 conventions). CORRECTION to
    the dispatch, recorded: arm_b's zero-mode census ALREADY EXISTS in
    runs/e133/metrics.json (nets.site_locked.arms.head_*, 36 heads, full
    battery); what e133 never ran is the mean-mode bracket, the split-window
    selection, and the ESCALATION. e125a re-derives the census live on
    pool_sel AND audits it against e133's stored full-battery table (36
    zero-mode head cells, tol 5e-4). Top-8 census heads also run in MEAN
    mode (bracket). Ladder = heads ranked by drop within the CE<=+0.35
    class (e160's convention): t1..t5 at runtime; e133's stored table says
    to expect L1H2, L0H5, L0H1, L1H4, L1H1.
  * (B) ESCALATION on arm_b, both modes (zero; mean-replace = CE_R-bank
    mean, per-net bank, no fact contexts — e150/e160 convention):
      - own-census singles: t1..t4 alone;
      - B2={t1+next1}, B3={t1+top2}, B4={t1+top3};
      - M2/M3/M4 = top-k WITHOUT t1 (k matched);
      - e160's sets re-pointed: singles L0H3/L1H0/L0H0/L1H3; E2/E3/E4
        (L0H3-ladder), N2/N3/N4 (no-L0H3 ladder);
      - ADDRESS-HEAD candidates (the dispatch's "L3H5-class" site-readers,
        e133's geo criterion site_mass>=0.25 -> L3H4/L4H3/L4H5, plus the
        named L3H5): singles L3H5/L4H5, A2={L4H5,L4H3}, A4={L4H5,L4H3,
        L3H4,L3H5};
      - COMBINED: X1={t1,t2}+{L1H0,L0H0} (both knives' cores),
        X2={t1,t2}+A2 (census core + reader core);
      - random-scatter controls: 3 draws of 4 heads (seed 12503, all 36
        coords), NOT bar-eligible.
    Per cell: site onset at pool_bar (PRIMARY bar read) + pool_sel
    (selection-window companion, reported), pname-over-7 span read,
    std install-60 g0 (the site net's old-fact floor, Rule-12 companion),
    CE_R (e065 bank seed 26502, 60 windows), NLL3 (3 windows seed 31337).
  * (C) CROSS-CHECK (report-only, out of the bars): the e160 family sets
    (singles + E/N ladders) plus B3/A4/S:t1 on e143_near.pt at ITS site
    (name x-cols 6..12, onset read row 5 — e143's own-geometry battery,
    install-60 windows, base onset 0.9986116290092468).
  * CONTROL NET: e048_repro.pt (install-phase; no site-stored fact) — the
    site battery reads its floor (~0) and the largest sets run as floor
    controls (drops off a ~0 base are meaningless; raw onsets reported).
  * ADJUDICATION PRECEDENCE (pre-registered): SITE-KNIFE-EXISTS if any
    bar-eligible cell has pool_bar drop >= 60% at CE <= +0.35; else
    NO-SITE-KNIFE if the best flat-CE pool_bar drop across ALL bar-eligible
    cells is < 30% (wreck-priced kills, if any, reported alongside — the
    bar's letter concerns flat-CE surgery); else TEXTURE with numbers.
    Non-gating companions (pool_sel drop, pname drop, either-window union)
    are RECORDED and reported, never gating.
  * INSTRUMENT CONTINUITY gates: e160's E4/zero and N4/zero site cells
    re-run on pool_full must match e160's stored onsets (tol 5e-6); the
    36-head zero census on pool_full must match e133's stored site_onset
    table (tol 5e-4); e048's install battery (bit, 5e-6); near-net base
    onset vs e143 stored (tol 1e-3, GPU-trained checkpoint on CPU).
    Smoke mode: gates informational only (reduced batteries).

NETS (on disk, gated; eval-only — no training; CPU-ONLY, e158-class runs
share the box — threads <= 4, staggered, no busy-waiting):
  * SITE-STORED  runs/checkpoints/e131_arm_b_corpus_spliced.pt  (primary)
  * REPLICATION  runs/checkpoints/e143_near.pt                  (cross-check)
  * CONTROL      runs/checkpoints/e048_repro.pt                 (floor)

INSTRUMENT PROVENANCE: battery/eval/surgery instruments are e160's
(lab/e160_headset_escalation.py) verbatim — which are e150's (battery_cell
e068/e113/e120, ce_fixed_cpu, val_windows e065 seed 26502, load_cpu/evl_load)
with e133's c_proj pre-hook head conventions (e001/e038 lesion lineage); the
pool-b site battery is e133's construction verbatim; the near battery is
e143's pool_near construction verbatim. Protocol rebuild: corpus seed 1337,
SPLICE_RNG host shuffle, install-60 / held-30 split, mix gate.

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 before torch import),
torch.set_num_threads(4), sequential evals, staggers between phases, no
training. Outputs: runs/e125a/{metrics.json, inverted_knife.png}. No
checkpoints written (eval-only).

Run:  cd lab && python e125a_inverted_knife.py    (E125A_SMOKE=1 -> shakedown)
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # CPU-ONLY (GPU user-occupied)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                           # noqa: E402

torch.set_num_threads(4)                              # LOW (shared CPU)

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
common.DEVICE = "cpu"
from common import Cfg, CharCorpus, TinyGPT, run_dir, save_json   # noqa: E402
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E125A_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
ARMB_CK = CKPT_DIR / "e131_arm_b_corpus_spliced.pt"   # primary (site-stored @183)
NEAR_CK = CKPT_DIR / "e143_near.pt"                   # replication (@5-13)
INST_CK = CKPT_DIR / "e048_repro.pt"                  # floor control
E133_METRICS = E43.REPO / "runs" / "e133" / "metrics.json"   # arm_b census audit + cons census
E160_METRICS = E43.REPO / "runs" / "e160" / "metrics.json"   # continuity refs + sink-coupled plane

# ---- arm_b site geometry (e133 pool-b verbatim) ------------------------------
FACT_PRE, FACT_POST = 12, 12
FACT_LEN = FACT_PRE + len(NAME) + FACT_POST      # 31
SPLICE_AT = 42
Z_XCOL = PRE + SPLICE_AT + FACT_PRE             # 184
SPLICE_ADDR_ROW = Z_XCOL - 1                    # 183

# ---- e143_near geometry (e143 pool_near verbatim) -----------------------------
NEAR_PRE = 6                                     # name x-cols 6..12
NEAR_ADDR_ROW = NEAR_PRE - 1                     # 5: wpe row predicting 'Z'
NEAR_Z_XCOL = NEAR_PRE                           # 6
NEAR_CONT = BLOCK - NEAR_PRE - len(NAME)         # 243

# seeds (fixed before compute)
R_EVAL_SEED = 26502               # e065 CE_R bank seed (verbatim, e150/e160)
SKILL_SEED = 31337                # base-skill probe windows (e160 verbatim)
CORP_CONT_SEED = 12103            # e133 pool_full filler seed (verbatim)
SEL_SEED = 12511                  # pool_sel filler draws (prompts 0-14)
BAR_SEED = 12512                  # pool_bar filler draws (prompts 15-29)
RAND_SEED = 12503                 # random-head scatter draws
N_PROMPTS_FULL = 6 if SMOKE else 30
HALF = (2 if SMOKE else 15)       # prompts per split battery
N_DRAWS = (1 if SMOKE else 4)     # fillers per prompt in the batteries
N_INSTALL = 6 if SMOKE else 60    # install prompts in smoke

# gates / references (full precision, from stored metrics)
G_BIT_TOL = 5e-6                  # bit-exact reproduction gate (CPU->CPU)
G_CENSUS_TOL = 5e-4               # census audit vs e133 stored (float noise)
G_FALLBACK_TOL = 0.05             # e113 G_REPRO convention
G_ARMB_REF_STD = 0.007800613064318895       # e133 std battery floor (site net)
G_ARMB_REF_SITE = 0.9880021214485168        # e131/e160 site onset (pool_full)
G_ARMB_REF_CE = 1.682660698890686           # e160 site net CE_R
G_E160_E4Z_REF = 0.8830115795135498         # e160 E4/zero site onset (pool_full)
G_E160_N4Z_REF = 0.6837095022201538         # e160 N4/zero site onset (pool_full)
G_INST_REF = 0.5563086867332458             # e131 G_E048 / e160 G_INST
G_NEAR_REF_ONSET = 0.9986116290092468       # e143 own_geometry_read.near
G_NEAR_REF_PNAME = 0.9992892146110535       # e143 pname over 7
G_NEAR_REF_CE = 1.683059811592102           # e143 near net CE_R (CPU eval)
G_NEAR_TOL = 1e-3                           # GPU-trained ckpt on CPU
G_CONS_BASE = 0.7850371599197388            # e133 graduated base (std battery)

# registered bars (numeric)
KILL_DROP = 60.0                  # site-drop % bar (>= 60% = kill)
NO_KNIFE_DROP = 30.0              # NO-SITE-KNIFE bar (< 30% best flat-CE)
CE_FLAT = 0.35                    # flat-CE bar (<= +0.35)

REGISTERED_PREDICTION = {
    "queue_row_verbatim": (
        "re-point e160's escalation grid at e131_arm_b (site-stored) + "
        "e143_near (replication column): singles->E4 + site-reader candidates "
        "(L3H5-class) + random scatter. Bars: SITE-KNIFE-EXISTS = some "
        "head-set drops the site-fact >=60% at CE <= +0.35; NO-SITE-KNIFE = "
        "best <30% at CE <= +0.35 across the escalation (the types differ in "
        "REMOVABILITY — asymmetric surgery, the strongest form of the "
        "split-custody claim)"),
    "bar_read": (
        "PRIMARY bar read = pool_bar site onset (held-prompts 15-29, filler "
        "seed 12512 — windows NEVER used for head selection; the census "
        "selects on pool_sel, prompts 0-14, seed 12511). Companions recorded "
        "per cell and NON-GATING: pool_sel drop (selection window), "
        "pname-span drop, either-window union. pool_full (e133/e160's exact "
        "battery) is used only for stored-value continuity gates."),
    "operationalizations": (
        "census: 36 heads x ZERO mode on pool_sel (drop + CE per head, e133 "
        "conventions) audited vs e133's stored site_locked table on pool_full "
        "(tol 5e-4); top-8 census heads also in MEAN mode. Ladder t1..t5 = "
        "drop-ranked within CE<=+0.35 class (e160 convention; e133's stored "
        "table expects L1H2,L0H5,L0H1,L1H4,L1H1). Escalation sets: census "
        "singles t1-t4; B2/B3/B4 (t1-ladder); M2/M3/M4 (no-t1, k-matched); "
        "e160's singles L0H3/L1H0/L0H0/L1H3 + E2/E3/E4 + N2/N3/N4; address "
        "candidates S:L3H5, S:L4H5, A2={L4H5,L4H3}, A4={L4H5,L4H3,L3H4,L3H5}; "
        "combined X1={t1,t2,L1H0,L0H0}, X2={t1,t2,L4H5,L4H3}; random-4 "
        "scatter x3 (seed 12503, control). Modes zero + mean-replace "
        "(per-net CE_R-bank mean). Per cell: site onset pool_bar (bar) + "
        "pool_sel + pname span + std install g0 floor + CE_R + NLL3."),
    "no_bar_shopping": "No bar shopping. Ambiguous => say TEXTURE with "
                       "numbers.",
}

recipe_deviations: list[str] = [
    "DISPATCH CORRECTION (recorded, not shopped): arm_b's zero-mode head "
    "census already exists in runs/e133/metrics.json (nets.site_locked.arms, "
    "36 heads, full 30-prompt battery) — the dispatch's 'census was never "
    "run' note is wrong in letter. e125a still runs its own census (needed "
    "for the split-window selection and the mean-mode bracket) and audits it "
    "against e133's stored table; the ESCALATION is what never existed.",
    "SPLIT-WINDOW SELECTION/ADJUDICATION (honesty): the dispatch says "
    "'selection on own-census then escalation on same net — split windows if "
    "possible'. Implemented: census/selection on pool_sel (prompts 0-14), "
    "bars on pool_bar (prompts 15-29) — disjoint prompts AND disjoint filler "
    "draws. Cost: pool_bar has 60 windows (vs e160's 120), so the bar read "
    "is noisier; the pool_sel companion column prices that noise per cell.",
    "The dispatch's 'singles->E4' family and its 'L3H5-class site-reader "
    "candidates' are kept literal: e133's geo criterion names L3H4/L4H3/L4H5 "
    "(site_mass>=0.25); L3H5 itself sits at site_mass 0.199 (5th) — A4 "
    "carries all four (the three criterion heads + the dispatch's named "
    "representative); L3H2 (site_mass 0.214, between L3H4 and L3H5) is in "
    "the census table and reported, not promoted to a set.",
    "ADDED cells beyond the dispatched grid (densifiers, same bars): the "
    "census singles t1-t4 and M2/M3/M4 (the no-t1 matched ladder — e160's "
    "N-ladder analogue, required for the superadditivity read), and the two "
    "combined sets X1/X2 (the direct two-knife superposition the flip-cell "
    "implies). All dispatched families run.",
    "e048 control column reads the SITE battery's floor (base ~0): drop "
    "percentages off a ~0 base are meaningless there, so RAW onsets are "
    "reported (no drop, no bar eligibility).",
    "Near-net base gates use tol 1e-3 (GPU-trained checkpoint re-evaluated "
    "on CPU; e143's own GPU-repro tolerances were 0.05 — this is tighter "
    "and reported with exact diffs).",
]

trims: list[str] = []


# ------------------------------------------------------------------ instruments
# (e160 verbatim: load_cpu/evl_load/battery_fwd/ce_fwd/val_windows/
#  head_mean_vec/HeadReplace; site reads generalized over geometry)

def evl_load(sd: dict) -> TinyGPT:
    m = TinyGPT(Cfg())
    m.load_state_dict(sd)
    m.eval()
    return m


def load_cpu(path) -> TinyGPT:
    m = TinyGPT(Cfg())
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def battery_fwd(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> dict:
    """e068/e113/e120 battery on CPU: p(Z) at the last position."""
    net.eval()
    pzs, amax = [], 0.0
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs.append(pr[:, zid])
        amax += float((lg[:, -1].argmax(-1) == zid).float().sum())
    p = torch.cat(pzs)
    return {"mean_pz": float(p.mean()), "median_pz": float(p.median()),
            "std_pz": float(p.std()),
            "frac_pz_ge_0.5": float((p >= 0.5).float().mean()),
            "frac_argmax_z": amax / ids.shape[0]}


@torch.no_grad()
def ce_fwd(net: TinyGPT, x, y, bs=64) -> float:
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        _, loss = net(x[i:i + bs], y[i:i + bs])
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


def head_mean_vec(net: TinyGPT, layer: int, head: int, bank_x, bs=30):
    """Mean of the head's 32-dim slice of c_proj's input over the CE_R bank
    (corpus windows only — no fact contexts). e133 organ_stats / e150 / e160
    convention."""
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
    """Replace head slices of c_proj's input with fixed vectors (mean-replace)
    or zeros (e133's zero arm), eval-only forward pre-hooks."""

    def __init__(self, net: TinyGPT, replace: dict):
        # replace: {(layer, head): vector-or-None}; None -> zero
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


@torch.no_grad()
def site_reads(net: TinyGPT, pool: torch.Tensor, z_xcol: int, addr_row: int,
               name_ids: torch.Tensor, zid: int, bs=30) -> dict:
    """e133 battery_site logic over an arbitrary site geometry: onset p(Z) at
    addr_row + mean p(true name char) over the 7 name positions."""
    onset, per_pos = [], [[] for _ in range(len(name_ids))]
    for i in range(0, pool.shape[0], bs):
        w = pool[i:i + bs]
        lg, _ = net(w)
        pr = F.softmax(lg, -1)
        for k in range(pr.shape[0]):
            onset.append(float(pr[k, addr_row, int(zid)]))
            for j in range(len(name_ids)):
                per_pos[j].append(
                    float(pr[k, addr_row + j,
                           int(w[k, z_xcol + j])]))
    on = torch.tensor(onset)
    allp = torch.tensor([q for pos in per_pos for q in pos])
    return {"onset_pz": float(on.mean()),
            "onset_frac_ge_0.5": float((on >= 0.5).float().mean()),
            "pname_mean_over7": float(allp.mean())}


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e125a_smoke" if SMOKE else "e125a")
    log(f"E125A THE INVERTED KNIFE (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-only, torch threads {torch.get_num_threads()}, "
        f"cuda available = {torch.cuda.is_available()}")

    # ---------------- protocol rebuild (e141/e150/e160 verbatim)
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
    inst_ctx = install_occ[:N_INSTALL]
    bat_g0 = torch.stack([corpus.encode(train_text[p - PRE: p])
                          for p, _ in inst_ctx])
    bat_gm12 = torch.stack([corpus.encode(train_text[p - PRE + 12: p])
                            for p, _ in inst_ctx])   # e160 bat_ids[(-12, ...)]
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    skill_x, skill_y = val_windows(val_ids, val_text, 3, SKILL_SEED)
    skill_xy = (skill_x, skill_y)
    log(f"CE_R bank {tuple(r_eval_x.shape)} (seed {R_EVAL_SEED}); "
        f"base-skill probe {tuple(skill_x.shape)} (seed {SKILL_SEED})")

    # ---------------- site batteries (e133 pool-b construction; split windows)
    def splice_pool(prompt_list, n_draws, seed):
        pid = torch.stack([corpus.encode(c) for c in prompt_list])
        gc_ = torch.Generator().manual_seed(seed)
        src = torch.randint(len(train_ids) - (BLOCK - PRE) - 1,
                            (n_draws, len(prompt_list)), generator=gc_)
        filler = torch.stack([train_ids[s: s + BLOCK - PRE]
                              for s in src.flatten()])
        segs = []
        for p, h in inst_ctx:
            s = (train_text[p - FACT_PRE: p] + NAME
                 + train_text[p + len(h): p + len(h) + FACT_POST])
            assert len(s) == FACT_LEN
            segs.append(corpus.encode(s))
        fact_segs = torch.stack(segs)
        fs = fact_segs[torch.arange(filler.shape[0]) % fact_segs.shape[0]]
        cont = torch.cat([filler[:, :SPLICE_AT], fs,
                          filler[:, SPLICE_AT + FACT_LEN:]], 1)
        pool = torch.cat([torch.stack(
            [pid[k % pid.shape[0]] for k in range(filler.shape[0])]),
            cont], 1)
        assert all(torch.equal(w[Z_XCOL:Z_XCOL + len(NAME)], name_ids)
                   for w in pool)
        return pool

    prompts_full = [train_text[p - PRE: p] for p, _ in held_occ][:N_PROMPTS_FULL]
    prompts_sel = prompts_full[:HALF]
    prompts_bar = prompts_full[HALF:HALF + HALF]
    pool_full = splice_pool(prompts_full, N_DRAWS, CORP_CONT_SEED)
    pool_sel = splice_pool(prompts_sel, N_DRAWS, SEL_SEED)
    pool_bar = splice_pool(prompts_bar, N_DRAWS, BAR_SEED)
    log(f"site batteries: full {tuple(pool_full.shape)} (seed "
        f"{CORP_CONT_SEED}, gating only) | sel {tuple(pool_sel.shape)} "
        f"(seed {SEL_SEED}, selection) | bar {tuple(pool_bar.shape)} (seed "
        f"{BAR_SEED}, bars) — ZEPHYRA at x-col {Z_XCOL} (address row "
        f"{SPLICE_ADDR_ROW})")

    # ---------------- near battery (e143 pool_near construction verbatim)
    near_wins = []
    for p, h in inst_ctx:
        pre = train_ids[p - NEAR_PRE: p]
        post = train_ids[p + len(h): p + len(h) + NEAR_CONT]
        w = torch.cat([pre, name_ids, post])
        if len(w) != BLOCK:
            raise RuntimeError(f"NEAR window len {len(w)} != {BLOCK}")
        near_wins.append(w)
    pool_near = torch.stack(near_wins)
    assert all(torch.equal(w[NEAR_Z_XCOL:NEAR_Z_XCOL + len(NAME)], name_ids)
               for w in pool_near)
    log(f"near battery: {tuple(pool_near.shape)}, ZEPHYRA at x-col "
        f"{NEAR_Z_XCOL} (address row {NEAR_ADDR_ROW})")

    # ---------------- stored references for audit / plot
    e133 = json.loads(E133_METRICS.read_text(encoding="utf-8"))
    e160 = json.loads(E160_METRICS.read_text(encoding="utf-8"))
    sl133 = e133["nets"]["site_locked"]
    cons133 = e133["nets"]["graduated"]

    def parse_head(tag):
        p_ = tag.split("_")[1]
        return (int(p_[1:p_.find("h")]), int(p_[p_.find("h") + 1:]))

    e133_census = {parse_head(t): v["site_onset"]                # zero, full
                   for t, v in sl133["arms"].items() if t.startswith("head_")}
    sl_base_onset = sl133["base"]["site"]["onset_pz"]
    sl_base_ce = sl133["base"]["ce_r"]
    e133_rank = sorted(
        [(parse_head(t), sl_base_onset - v["site_onset"],
          v["ce_r"] - sl_base_ce)
         for t, v in sl133["arms"].items() if t.startswith("head_")],
        key=lambda r: -r[1])
    e133_rank_flat = [r for r in e133_rank if r[2] <= CE_FLAT][:5]
    cons_base = cons133["base"]["std_install60"]["mean_pz"]   # == G_CONS_BASE
    cons_drops = {parse_head(t): cons_base - v["std_install60"]
                  for t, v in cons133["arms"].items()
                  if t.startswith("head_")}

    def hx(l, h):
        return f"L{l}H{h}"

    # ---------------- nets + gates
    log("--- PHASE 0: load + gate the three artifacts ---")
    gates: dict = {}

    net_armb = load_cpu(ARMB_CK)
    bz_armb = battery_fwd(net_armb, bat_g0, zid)["mean_pz"]
    st_full = site_reads(net_armb, pool_full, Z_XCOL, SPLICE_ADDR_ROW,
                         name_ids, zid)
    ce_armb = ce_fwd(net_armb, *r_eval_xy)
    gates["arm_b"] = {
        "std_g0_floor": bz_armb, "ref_std": G_ARMB_REF_STD,
        "site_onset_full": st_full["onset_pz"], "ref_site": G_ARMB_REF_SITE,
        "ce_r": ce_armb, "ref_ce": G_ARMB_REF_CE,
        "pass": bool(abs(bz_armb - G_ARMB_REF_STD) < G_FALLBACK_TOL
                     and abs(st_full["onset_pz"] - G_ARMB_REF_SITE)
                     < G_FALLBACK_TOL
                     and abs(ce_armb - G_ARMB_REF_CE) < G_FALLBACK_TOL),
        "site_onset_bit": bool(abs(st_full["onset_pz"] - G_ARMB_REF_SITE)
                               < G_BIT_TOL)}
    log(f"G_ARMB: std floor {bz_armb:.10f} | site onset "
        f"{st_full['onset_pz']:.10f} (ref {G_ARMB_REF_SITE:.10f}) | CE "
        f"{ce_armb:.6f}: "
        f"{'PASS' if gates['arm_b']['pass'] else 'FAIL'}"
        + ("" if SMOKE else
           f" [bit {'OK' if gates['arm_b']['site_onset_bit'] else 'DRIFT'}]"))
    if not gates["arm_b"]["pass"]:
        raise RuntimeError("arm_b checkpoint failed its gate")

    net_near = load_cpu(NEAR_CK)
    st_near = site_reads(net_near, pool_near, NEAR_Z_XCOL, NEAR_ADDR_ROW,
                         name_ids, zid)
    ce_near = ce_fwd(net_near, *r_eval_xy)
    gates["near"] = {
        "onset": st_near["onset_pz"], "ref_onset": G_NEAR_REF_ONSET,
        "pname": st_near["pname_mean_over7"], "ref_pname": G_NEAR_REF_PNAME,
        "ce_r": ce_near, "ref_ce": G_NEAR_REF_CE, "tol": G_NEAR_TOL,
        "pass": bool(abs(st_near["onset_pz"] - G_NEAR_REF_ONSET) < G_NEAR_TOL
                     and abs(ce_near - G_NEAR_REF_CE) < G_NEAR_TOL)}
    log(f"G_NEAR: onset {st_near['onset_pz']:.10f} (ref "
        f"{G_NEAR_REF_ONSET:.10f}, diff "
        f"{st_near['onset_pz'] - G_NEAR_REF_ONSET:+.2e}) | CE {ce_near:.6f}: "
        f"{'PASS' if gates['near']['pass'] else 'FAIL'}")
    if not gates["near"]["pass"] and not SMOKE:
        raise RuntimeError("e143_near checkpoint failed its gate")

    net_inst = load_cpu(INST_CK)
    bz_inst = battery_fwd(net_inst, bat_g0, zid)["mean_pz"]
    gates["install"] = {"battery_pz": bz_inst, "ref": G_INST_REF,
                        "pass": bool(abs(bz_inst - G_INST_REF) < G_BIT_TOL)}
    log(f"G_INST: p(Z) {bz_inst:.10f}: "
        f"{'PASS' if gates['install']['pass'] else 'FAIL'}")
    if not gates["install"]["pass"] and not SMOKE:
        raise RuntimeError("e048_repro checkpoint failed its gate")

    # ---------------- e160 instrument continuity (E4/N4 zero on pool_full)
    log("--- PHASE 0b: e160 site-column instrument continuity (pool_full) ---")
    E160_E4 = [(0, 3), (1, 0), (0, 0), (1, 3)]
    E160_N4 = [(1, 0), (0, 0), (1, 3), (0, 1)]
    with HeadReplace(net_armb, {c: None for c in E160_E4}):
        e4z = site_reads(net_armb, pool_full, Z_XCOL, SPLICE_ADDR_ROW,
                         name_ids, zid)["onset_pz"]
    with HeadReplace(net_armb, {c: None for c in E160_N4}):
        n4z = site_reads(net_armb, pool_full, Z_XCOL, SPLICE_ADDR_ROW,
                         name_ids, zid)["onset_pz"]
    gates["e160_continuity"] = {
        "E4_zero_onset": e4z, "E4_ref": G_E160_E4Z_REF,
        "N4_zero_onset": n4z, "N4_ref": G_E160_N4Z_REF, "tol": G_BIT_TOL,
        "pass": bool(abs(e4z - G_E160_E4Z_REF) < G_BIT_TOL
                     and abs(n4z - G_E160_N4Z_REF) < G_BIT_TOL)}
    log(f"G_E160X: E4/zero {e4z:.10f} (ref {G_E160_E4Z_REF:.10f}) | N4/zero "
        f"{n4z:.10f} (ref {G_E160_N4Z_REF:.10f}): "
        f"{'PASS' if gates['e160_continuity']['pass'] else 'FAIL'}")
    if not gates["e160_continuity"]["pass"] and not SMOKE:
        raise RuntimeError("head-hook instrument drifted from e160's anchors")
    time.sleep(2.0)                                   # stagger (shared CPU)

    # ---------------- PHASE 1: bases
    log("--- PHASE 1: bases (no intervention) ---")
    st_sel = site_reads(net_armb, pool_sel, Z_XCOL, SPLICE_ADDR_ROW,
                        name_ids, zid)
    st_bar = site_reads(net_armb, pool_bar, Z_XCOL, SPLICE_ADDR_ROW,
                        name_ids, zid)
    gm12_armb = battery_fwd(net_armb, bat_gm12, zid)["mean_pz"]
    base_armb = {
        "site_full": st_full["onset_pz"], "site_sel": st_sel["onset_pz"],
        "site_bar": st_bar["onset_pz"],
        "site_sel_detail": st_sel, "site_bar_detail": st_bar,
        "std_g0": bz_armb, "std_gm12": gm12_armb,
        "ce_r": ce_armb, "nll3": ce_fwd(net_armb, *skill_xy)}
    log(f"armB base: site full {base_armb['site_full']:.6f} | sel "
        f"{base_armb['site_sel']:.6f} | bar {base_armb['site_bar']:.6f} "
        f"(pname {st_bar['pname_mean_over7']:.4f}) | std g0 floor "
        f"{bz_armb:.6f} | g-12 {gm12_armb:.6f} | CE {ce_armb:.6f}")

    base_near = {"site": st_near["onset_pz"], "site_detail": st_near,
                 "ce_r": ce_near, "nll3": ce_fwd(net_near, *skill_xy)}
    log(f"near base: onset {base_near['site']:.6f} (pname "
        f"{st_near['pname_mean_over7']:.4f}) CE {ce_near:.6f}")

    st_inst = site_reads(net_inst, pool_bar, Z_XCOL, SPLICE_ADDR_ROW,
                         name_ids, zid)
    base_inst = {"site_bar": st_inst["onset_pz"],
                 "ce_r": ce_fwd(net_inst, *r_eval_xy),
                 "nll3": ce_fwd(net_inst, *skill_xy)}
    log(f"e048 control base: site-bar floor {base_inst['site_bar']:.2e} "
        f"CE {base_inst['ce_r']:.6f}")

    # ---------------- PHASE 2: the census (selection on pool_sel; audit on
    # pool_full vs e133's stored table)
    log("--- PHASE 2: arm_b's own head census (zero; selection=pool_sel) ---")
    all_coords = [(l, h) for l in range(6) for h in range(6)]
    CENSUS: list[dict] = []
    audit_diffs = []
    census_coords = all_coords if not SMOKE else all_coords[:8]
    for (l, h) in census_coords:
        with HeadReplace(net_armb, {(l, h): None}):
            on_sel = site_reads(net_armb, pool_sel, Z_XCOL, SPLICE_ADDR_ROW,
                                name_ids, zid)["onset_pz"]
            on_full = site_reads(net_armb, pool_full, Z_XCOL, SPLICE_ADDR_ROW,
                                 name_ids, zid)["onset_pz"]
            ce_arm = ce_fwd(net_armb, *r_eval_xy)
        ref = e133_census.get((l, h))
        if ref is not None and not SMOKE:
            audit_diffs.append(abs(on_full - ref))
        CENSUS.append({
            "head": hx(l, h), "l": l, "h": h,
            "onset_sel": on_sel,
            "drop_sel": base_armb["site_sel"] - on_sel,
            "onset_full": on_full, "ref_e133_full": ref,
            "ce_cost": ce_arm - base_armb["ce_r"]})
    max_audit = max(audit_diffs) if audit_diffs else float("nan")
    gates["census_audit"] = {
        "n_audited": len(audit_diffs), "max_abs_diff_vs_e133": max_audit,
        "tol": G_CENSUS_TOL, "pass": bool(max_audit < G_CENSUS_TOL)}
    log(f"census audit vs e133 stored (pool_full, zero): {len(audit_diffs)} "
        f"heads, max |diff| {max_audit:.2e} "
        f"({'PASS' if gates['census_audit']['pass'] else 'FAIL'})")
    if not gates["census_audit"]["pass"] and not SMOKE:
        raise RuntimeError("census drifted from e133's stored site table")

    ce_class = sorted([c for c in CENSUS if c["ce_cost"] <= CE_FLAT],
                      key=lambda c: -c["drop_sel"])
    top = ce_class[:5]
    log("census CE<=+0.35 class, top by drop (pool_sel): "
        + ", ".join(f"{c['head']} (drop {c['drop_sel']:.4f}, CE "
                    f"{c['ce_cost']:+.4f})" for c in top))
    log("e133 stored ranking (full battery, CE<=+0.35): "
        + ", ".join(f"{hx(*r[0])} ({r[1]:.4f}@{r[2]:+.4f})"
                    for r in e133_rank_flat))
    ladder = [(c["l"], c["h"]) for c in top]
    while len(ladder) < 5:                    # smoke guard: pad harmlessly
        ladder.append(all_coords[len(ladder)])
    (t1, t2, t3, t4, t5) = ladder
    time.sleep(2.0)                                   # stagger

    # mean-mode bracket on the top-8 census heads
    mvcache_armb: dict = {}

    def mv_armb(l, h):
        if (l, h) not in mvcache_armb:
            mvcache_armb[(l, h)] = head_mean_vec(net_armb, l, h, r_eval_x)
        return mvcache_armb[(l, h)]

    log("--- PHASE 2b: census mean-mode bracket (top-8) ---")
    for c in sorted(CENSUS, key=lambda c: -c["drop_sel"])[:8]:
        l, h = c["l"], c["h"]
        with HeadReplace(net_armb, {(l, h): mv_armb(l, h)}):
            on_sel = site_reads(net_armb, pool_sel, Z_XCOL, SPLICE_ADDR_ROW,
                                name_ids, zid)["onset_pz"]
            ce_arm = ce_fwd(net_armb, *r_eval_xy)
        c["onset_sel_mean"] = on_sel
        c["drop_sel_mean"] = base_armb["site_sel"] - on_sel
        c["ce_cost_mean"] = ce_arm - base_armb["ce_r"]
        log(f"  {c['head']}: mean drop {c['drop_sel_mean']:.4f} @ CE "
            f"{c['ce_cost_mean']:+.4f} (zero: {c['drop_sel']:.4f} @ "
            f"{c['ce_cost']:+.4f})")

    # ---------------- PHASE 3: set definitions
    singles_census = [("S:" + hx(*t1), [t1]), ("S:" + hx(*t2), [t2]),
                      ("S:" + hx(*t3), [t3]), ("S:" + hx(*t4), [t4])]
    blad = [("B2:" + hx(*t1) + "+top1", [t1, t2]),
            ("B3:" + hx(*t1) + "+top2", [t1, t2, t3]),
            ("B4:" + hx(*t1) + "+top3", [t1, t2, t3, t4])]
    msets = [("M2:top2-no" + hx(*t1), [t2, t3]),
             ("M3:top3-no" + hx(*t1), [t2, t3, t4]),
             ("M4:top4-no" + hx(*t1), [t2, t3, t4, t5])]
    singles_e160 = [("S:L0H3", [(0, 3)]), ("S:L1H0", [(1, 0)]),
                    ("S:L0H0", [(0, 0)]), ("S:L1H3", [(1, 3)])]
    esets = [("E2:L0H3+top1", [(0, 3), (1, 0)]),
             ("E3:L0H3+top2", [(0, 3), (1, 0), (0, 0)]),
             ("E4:L0H3+top3", [(0, 3), (1, 0), (0, 0), (1, 3)])]
    nsets = [("N2:top2-noL0H3", [(1, 0), (0, 0)]),
             ("N3:top3-noL0H3", [(1, 0), (0, 0), (1, 3)]),
             ("N4:top4-noL0H3", [(1, 0), (0, 0), (1, 3), (0, 1)])]
    asets = [("S:L3H5", [(3, 5)]), ("S:L4H5", [(4, 5)]),
             ("A2:site-mass-top2", [(4, 5), (4, 3)]),
             ("A4:L3H5-class", [(4, 5), (4, 3), (3, 4), (3, 5)])]
    xsets = [("X1:B2+N2core", [t1, t2, (1, 0), (0, 0)]),
             ("X2:B2+A2", [t1, t2, (4, 5), (4, 3)])]
    arm_b_sets = (singles_census + blad + msets + singles_e160 + esets
                  + nsets + asets + xsets)

    rr = random.Random(RAND_SEED)
    rand_sets = [(f"R{d + 1}:random4", sorted(rr.sample(all_coords, 4)))
                 for d in range(3 if not SMOKE else 1)]
    log("random scatter draws: "
        + "; ".join(f"{t}=[{','.join(hx(*c) for c in s)}]"
                    for t, s in rand_sets))

    near_sets = (singles_e160 + esets + nsets
                 + [("B3:" + hx(*t1) + "+top2", [t1, t2, t3]),
                    ("S:" + hx(*t1), [t1]),
                    ("A4:L3H5-class", [(4, 5), (4, 3), (3, 4), (3, 5)])])
    inst_sets = [("B4:" + hx(*t1) + "+top3", [t1, t2, t3, t4]),
                 ("E4:L0H3+top3", [(0, 3), (1, 0), (0, 0), (1, 3)]),
                 ("A4:L3H5-class", [(4, 5), (4, 3), (3, 4), (3, 5)]),
                 ("X1:B2+N2core", [t1, t2, (1, 0), (0, 0)])]

    if SMOKE:
        arm_b_sets = [singles_census[0], blad[0], esets[0], asets[2]]
        near_sets = near_sets[:3]
        inst_sets = inst_sets[:1]

    # ---------------- PHASE 4: the cell machine (bar read = pool_bar)
    CELLS: list[dict] = []

    def log_cell(c):
        sel = c.get("drop_sel_pct")
        sel_txt = f"{sel:6.1f}%" if sel is not None else "   n/a"
        nll = c.get("nll3_delta")
        nll_txt = f"{nll:+.4f}" if nll is not None else "  n/a"
        log(f"  [{c['net'][:6]:6s} | {c['set']:17s} | {c['mode']:4s}] "
            f"bar {c['expr_bar_base']:.4f}->{c['expr_bar']:.4f} "
            f"(drop {c['drop_bar_pct']:6.1f}%) sel-drop "
            f"{sel_txt} | CE {c['ce_cost']:+.4f} | NLL3 {nll_txt}"
            + ("  [ctl]" if not c["bar_eligible"] else ""))

    def run_arm_b_cells(sets, modes, bar_eligible):
        for tag, heads in sets:
            for mode in modes:
                replace = {(l, h): (None if mode == "zero" else mv_armb(l, h))
                           for (l, h) in heads}
                with HeadReplace(net_armb, replace):
                    sr_bar = site_reads(net_armb, pool_bar, Z_XCOL,
                                        SPLICE_ADDR_ROW, name_ids, zid)
                    sr_sel = site_reads(net_armb, pool_sel, Z_XCOL,
                                        SPLICE_ADDR_ROW, name_ids, zid)
                    ce_arm = ce_fwd(net_armb, *r_eval_xy)
                    nll_arm = ce_fwd(net_armb, *skill_xy)
                    std_g0 = battery_fwd(net_armb, bat_g0, zid)["mean_pz"]
                c = {
                    "net": "arm_b", "set": tag,
                    "heads": [hx(l, h) for (l, h) in heads],
                    "n_heads": len(heads), "mode": mode,
                    "bar_eligible": bool(bar_eligible),
                    "expr_bar_base": base_armb["site_bar"],
                    "expr_bar": sr_bar["onset_pz"],
                    "drop_bar_pct": 100.0 * (1.0 - sr_bar["onset_pz"]
                                             / max(base_armb["site_bar"],
                                                   1e-12)),
                    "expr_sel_base": base_armb["site_sel"],
                    "expr_sel": sr_sel["onset_pz"],
                    "drop_sel_pct": 100.0 * (1.0 - sr_sel["onset_pz"]
                                             / max(base_armb["site_sel"],
                                                   1e-12)),
                    "site_detail_bar": sr_bar,
                    "pname_drop_pct": 100.0 * (1.0 - sr_bar["pname_mean_over7"]
                                               / max(base_armb["site_bar_detail"]
                                                     ["pname_mean_over7"], 1e-12)),
                    "std_g0_floor": std_g0,
                    "ce_base": base_armb["ce_r"], "ce_arm": ce_arm,
                    "ce_cost": ce_arm - base_armb["ce_r"],
                    "nll3_base": base_armb["nll3"], "nll3_arm": nll_arm,
                    "nll3_delta": nll_arm - base_armb["nll3"],
                    "kills60_bar": bool(100.0 * (1.0 - sr_bar["onset_pz"]
                                                 / max(base_armb["site_bar"],
                                                       1e-12)) >= KILL_DROP),
                }
                CELLS.append(c)
                log_cell(c)

    log("--- PHASE 4: the escalation grid on ARM_B (bar read = pool_bar) ---")
    run_arm_b_cells(arm_b_sets, ("zero", "mean"), bar_eligible=True)
    time.sleep(3.0)                                   # stagger (shared CPU)
    log("--- PHASE 4b: random scatter (control) ---")
    run_arm_b_cells(rand_sets, ("zero", "mean"), bar_eligible=False)
    time.sleep(3.0)

    # ---------------- PHASE 5: near cross-check (report-only)
    log("--- PHASE 5: e143_near cross-check at ITS site (report-only) ---")
    mvcache_near: dict = {}

    def mv_near(l, h):
        if (l, h) not in mvcache_near:
            mvcache_near[(l, h)] = head_mean_vec(net_near, l, h, r_eval_x)
        return mvcache_near[(l, h)]

    for tag, heads in near_sets:
        for mode in ("zero", "mean"):
            replace = {(l, h): (None if mode == "zero" else mv_near(l, h))
                       for (l, h) in heads}
            with HeadReplace(net_near, replace):
                sr = site_reads(net_near, pool_near, NEAR_Z_XCOL,
                                NEAR_ADDR_ROW, name_ids, zid)
                ce_arm = ce_fwd(net_near, *r_eval_xy)
                nll_arm = ce_fwd(net_near, *skill_xy)
            c = {
                "net": "near", "set": tag,
                "heads": [hx(l, h) for (l, h) in heads],
                "n_heads": len(heads), "mode": mode, "bar_eligible": False,
                "expr_bar_base": base_near["site"], "expr_bar": sr["onset_pz"],
                "drop_bar_pct": 100.0 * (1.0 - sr["onset_pz"]
                                         / max(base_near["site"], 1e-12)),
                "site_detail_bar": sr,
                "pname_drop_pct": 100.0 * (
                    1.0 - sr["pname_mean_over7"]
                    / max(base_near["site_detail"]["pname_mean_over7"], 1e-12)),
                "ce_base": base_near["ce_r"], "ce_arm": ce_arm,
                "ce_cost": ce_arm - base_near["ce_r"],
                "nll3_base": base_near["nll3"], "nll3_arm": nll_arm,
                "nll3_delta": nll_arm - base_near["nll3"],
            }
            CELLS.append(c)
            log_cell(c)
    time.sleep(2.0)

    # ---------------- PHASE 6: e048 floor control (report-only)
    log("--- PHASE 6: e048 site-floor control (report-only) ---")
    for tag, heads in inst_sets:
        for mode in ("zero", "mean"):
            replace = {(l, h): (None if mode == "zero" else head_mean_vec(
                net_inst, l, h, r_eval_x)) for (l, h) in heads}
            with HeadReplace(net_inst, replace):
                sr = site_reads(net_inst, pool_bar, Z_XCOL, SPLICE_ADDR_ROW,
                                name_ids, zid)
                ce_arm = ce_fwd(net_inst, *r_eval_xy)
            c = {
                "net": "install", "set": tag,
                "heads": [hx(l, h) for (l, h) in heads],
                "n_heads": len(heads), "mode": mode, "bar_eligible": False,
                "expr_bar_base": base_inst["site_bar"],
                "expr_bar": sr["onset_pz"],
                "note": "floor control: raw onset reported; drop% off a ~0 "
                        "base is meaningless",
                "ce_base": base_inst["ce_r"], "ce_arm": ce_arm,
                "ce_cost": ce_arm - base_inst["ce_r"],
            }
            CELLS.append(c)
            log(f"  [e048  | {tag:17s} | {mode:4s}] site onset "
                f"{sr['onset_pz']:.2e} | CE {c['ce_cost']:+.4f}  [ctl]")

    # ---------------- adjudication (registered — no bar shopping)
    elig = [c for c in CELLS if c["bar_eligible"]]
    flat = [c for c in elig if c["ce_cost"] <= CE_FLAT]
    best_flat = (max(flat, key=lambda c: c["drop_bar_pct"]) if flat else None)
    site_knife = [c for c in flat if c["drop_bar_pct"] >= KILL_DROP]
    wrecks = [c for c in elig
              if c["drop_bar_pct"] >= KILL_DROP and c["ce_cost"] > CE_FLAT]
    best_flat_sel = (max(flat, key=lambda c: c["drop_sel_pct"])
                     if flat else None)

    def cs(c):
        return (f"{c['set']}/{c['mode']} bar-drop {c['drop_bar_pct']:.1f}% "
                f"(sel-window {c['drop_sel_pct']:.1f}%) at CE "
                f"{c['ce_cost']:+.3f}")

    if site_knife:
        overall = ("SITE-KNIFE-EXISTS (a symmetric site-stored-killing "
                   "head-set exists at flat CE — the split-custody claim "
                   "WEAKENS to 'differently attackable': "
                   + "; ".join(cs(c) for c in site_knife) + ")")
    elif best_flat is not None and best_flat["drop_bar_pct"] < NO_KNIFE_DROP:
        overall = ("NO-SITE-KNIFE (best flat-CE site-drop across the whole "
                   f"escalation is {best_flat['drop_bar_pct']:.1f}% < "
                   f"{NO_KNIFE_DROP:.0f}% — the types differ in REMOVABILITY; "
                   "asymmetric surgery; T090's type-selectivity inverts into "
                   "an asymmetry of EXISTENCE"
                   + (f"; {len(wrecks)} wreck-priced kill(s): "
                      + "; ".join(cs(c) for c in wrecks) if wrecks else "")
                   + ")")
    else:
        overall = ("TEXTURE (partial site-drops without a flat-CE kill: best "
                   f"flat-CE cell {cs(best_flat) if best_flat else 'n/a'}; "
                   f"{len(wrecks)} wreck-priced kill(s))"
                   + ("; " + "; ".join(cs(c) for c in wrecks) if wrecks
                      else ""))

    adjudication = {
        "bars_verbatim": REGISTERED_PREDICTION["queue_row_verbatim"],
        "bar_read": REGISTERED_PREDICTION["bar_read"],
        "n_cells_total": len(CELLS), "n_bar_eligible": len(elig),
        "SITE_KNIFE_EXISTS": {
            "fires": bool(site_knife),
            "cells": [{"set": c["set"], "mode": c["mode"], "heads": c["heads"],
                       "drop_bar_pct": c["drop_bar_pct"],
                       "drop_sel_pct": c["drop_sel_pct"],
                       "ce_cost": c["ce_cost"]} for c in site_knife]},
        "NO_SITE_KNIFE": {
            "fires": bool(best_flat is not None
                          and best_flat["drop_bar_pct"] < NO_KNIFE_DROP
                          and not site_knife),
            "best_flat_cell": (None if best_flat is None else
                               {"set": best_flat["set"],
                                "mode": best_flat["mode"],
                                "drop_bar_pct": best_flat["drop_bar_pct"],
                                "ce_cost": best_flat["ce_cost"]}),
            "wreck_priced_kills": [{"set": c["set"], "mode": c["mode"],
                                    "drop_bar_pct": c["drop_bar_pct"],
                                    "ce_cost": c["ce_cost"]}
                                   for c in wrecks]},
        "secondary_reads_REPORT_ONLY": {
            "best_flat_cell_sel_window": (None if best_flat_sel is None else
                                          {"set": best_flat_sel["set"],
                                           "mode": best_flat_sel["mode"],
                                           "drop_sel_pct":
                                               best_flat_sel["drop_sel_pct"],
                                           "ce_cost": best_flat_sel["ce_cost"]}),
            "note": "selection-window (pool_sel) and pname-span drops are "
                    "recorded per cell; they cannot gate (registered before "
                    "compute)."},
        "cross_check_summary": near_summary(CELLS),
        "random_scatter_summary": scatter_summary(CELLS),
        "verdict": overall,
    }
    log("=" * 78)
    log(f"E125A VERDICT: {overall}")
    log(f"  SITE-KNIFE-EXISTS {bool(site_knife)} | NO-SITE-KNIFE "
        f"{adjudication['NO_SITE_KNIFE']['fires']}")
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e125a_inverted_knife",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": ("QUEUE.md e125a row dispatched ~11:45Z (R47 ideator "
                         "top pick, T090's flip-cell). Docstring + bars "
                         "written before compute."),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("does a symmetric site-stored-killing head-set exist at "
                     "flat CE (>=60% site-fact drop at CE <= +0.35), or is "
                     "the site-stored fact's best flat-CE drop < 30% across "
                     "the whole escalation — the types differ in REMOVABILITY "
                     "itself (asymmetric surgery)?"),
        "nets": {
            "primary_site_stored": f"runs/checkpoints/{ARMB_CK.name} "
                                   f"(site fact @183; gated)",
            "replication_near": f"runs/checkpoints/{NEAR_CK.name} "
                                f"(site fact @5-13; cross-check, report-only)",
            "control_install": f"runs/checkpoints/{INST_CK.name} "
                               f"(site floor control, report-only)",
            "eval_only": True},
        "census": {
            "selection_battery": "pool_sel (held prompts 0-14, seed "
                                 f"{SEL_SEED})",
            "audit_battery": "pool_full (e133's exact battery, seed "
                             f"{CORP_CONT_SEED}) vs e133 stored, tol "
                             f"{G_CENSUS_TOL}",
            "ladder_ce_class_top5": [{"head": c["head"],
                                      "drop_sel": c["drop_sel"],
                                      "ce_cost": c["ce_cost"]} for c in top],
            "e133_stored_ranking_flat": [
                {"head": hx(*r[0]), "drop": r[1], "ce_cost": r[2]}
                for r in e133_rank_flat],
            "table": sorted(CENSUS, key=lambda c: -c["drop_sel"]),
            "consolidated_census_comparison": {
                "note": "the consolidated (sink-coupled) net's zero-mode "
                        "fact-drops from e133 stored arms (std install-60 "
                        "battery), for the two-censuses story",
                "cons_base": G_CONS_BASE,
                "top5": sorted(
                    [{"head": hx(l, h), "cons_drop": d}
                     for (l, h), d in cons_drops.items()],
                    key=lambda x: -x["cons_drop"])[:5]}},
        "bases": {"arm_b": base_armb, "near": base_near,
                  "install_control": base_inst},
        "gates": {"G_SPLICE": G_SPLICE, "arm_b": gates["arm_b"],
                  "near": gates["near"], "install": gates["install"],
                  "e160_continuity": gates["e160_continuity"],
                  "census_audit": gates["census_audit"]},
        "grid": CELLS,
        "adjudication": adjudication,
        "honesty_reflex": {
            "selection_then_escalation": "the census selects the B-ladder on "
                "pool_sel and the bars read pool_bar — DISJOINT prompts and "
                "filler draws, so selection cannot contaminate the "
                "adjudication window; the price is a 60-window bar battery "
                "(noisier), priced per cell by the sel-window companion "
                "column (a big sel-vs-bar gap = selection-window inflation, "
                "reported not hidden).",
            "census_already_existed": "e133's stored site_locked table "
                "ALREADY contains arm_b's 36-head zero-mode census (the "
                "dispatch's 'never run' note is corrected in deviations); "
                "e125a's audit gate (tol 5e-4) makes the re-derivation "
                "redundant-safe rather than novel — the ESCALATION is the "
                "new measurement, and its candidate families (e160 sets, "
                "L3H5-class address heads, combined sets) were registered "
                "independently of the census outcome.",
            "mode_choice": "zero is e133's convention; mean-replace is an "
                "intervention (per-net CE_R-bank means, no fact contexts). "
                "The two modes bracket the surgical range and DISAGREE "
                "sharply on the consolidated net's L0H3 (12.1% vs 58.6%) — "
                "any knife claim must name its mode; a kill in only one mode "
                "is weaker than one that survives both.",
            "single_lineage": "arm_b and e143_near are two nets of ONE "
                "lineage (e113 recipe chain; near = e143's locked re-teach "
                "at 5-13). The near column is a SITE-GEOMETRY replication, "
                "not a family replication — the removability map is "
                "lineage-specific until a second-family replication runs.",
            "geometry_of_the_read": "the site battery's onset row (183) IS "
                "arm_b's trained geometry — unlike e160 (where the bar read "
                "was the install battery and g-12 was the Rule-12 novel "
                "geometry), the site fact is geometry-BOUND (e151: D-all "
                "0.098), so there is no second site geometry to read; the "
                "std install g0/g-12 floors are recorded per cell as the "
                "old-fact companions instead.",
            "ce_bank_is_position_blind": "CE_R prices general LM damage on "
                "fact-free corpus windows; NLL3 is a second disjoint-window "
                "read — neither prices site-geometry damage specifically "
                "(e150's note; the point AND the confound).",
            "kill_semantics": "a site 'kill' = onset drop >= 60% vs the "
                "same-net base on pool_bar; onset_frac_ge_0.5 and the "
                "pname-span drop are recorded to check the drop is a real "
                "readout collapse, not a mean-shift artifact.",
        },
        "trims": trims,
        "recipe_deviations": recipe_deviations,
        "ckpt_inventory": {
            "saved": {},
            "external_used": [f"runs/checkpoints/{p.name}"
                              for p in (ARMB_CK, NEAR_CK, INST_CK)],
            "note": "eval-only: no checkpoints written or regenerated"},
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": 2739072, "device": "cpu",
                   "torch_threads": torch.get_num_threads(), "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "inverted_knife.png", CELLS, adjudication, e160, CENSUS,
         cons_drops, top)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'inverted_knife.png'}")
    log(f"total {time.time() - T0:.1f}s")


# ------------------------------------------------------------------ summaries

def near_summary(cells):
    ns = [c for c in cells if c["net"] == "near"]
    if not ns:
        return {"note": "no near cells (smoke)"}
    flat_ = [c for c in ns if c["ce_cost"] <= CE_FLAT]
    best = (max(flat_, key=lambda c: c["drop_bar_pct"]) if flat_ else
            max(ns, key=lambda c: c["drop_bar_pct"]))
    return {"n_cells": len(ns),
            "best_flat_drop_pct": best["drop_bar_pct"],
            "best_cell": {"set": best["set"], "mode": best["mode"],
                          "drop_pct": best["drop_bar_pct"],
                          "ce_cost": best["ce_cost"]},
            "any_kill60_flat": any(c["drop_bar_pct"] >= KILL_DROP
                                   and c["ce_cost"] <= CE_FLAT for c in ns),
            "note": "report-only: do arm_b's/e160's sets kill a SECOND "
                    "site-stored fact at a different site (5-13)?"}


def scatter_summary(cells):
    rs = [c for c in cells if c["set"].startswith("R")]
    if not rs:
        return {"note": "no random cells (smoke)"}
    return {"n_cells": len(rs),
            "drop_bar_min": min(c["drop_bar_pct"] for c in rs),
            "drop_bar_max": max(c["drop_bar_pct"] for c in rs),
            "ce_min": min(c["ce_cost"] for c in rs),
            "ce_max": max(c["ce_cost"] for c in rs)}


# ------------------------------------------------------------------ plot

def plot(path, cells, adj, e160, census, cons_drops, top):
    """THE TWO KNIVES side by side: e160's sink-coupled plane vs e125a's
    site-stored plane, plus the two nets' censuses and the near cross-check."""
    fig = plt.figure(figsize=(16.0, 12.0))
    gs = fig.add_gridspec(2, 2)
    msty = {"zero": ("o", "tab:red"), "mean": ("^", "tab:blue")}

    def plane(ax, pts, rand, title, ylabel=True, xlabel=True):
        xmax = max(1.0, max([c["ce_cost"] for c in pts + rand] or [0.8])
                   + 0.25)
        ax.axvspan(0, CE_FLAT, color="seagreen", alpha=0.07)
        ax.fill_between([-0.05, CE_FLAT], KILL_DROP, 118, color="seagreen",
                        alpha=0.18)
        ax.axhline(KILL_DROP, ls="--", color="gray", lw=1.1)
        ax.axvline(CE_FLAT, ls="--", color="seagreen", lw=1.2)
        ax.text(CE_FLAT * 0.5, 112, "flat-CE region (kill >=60% @ CE<=+0.35)",
                ha="center", fontsize=8, color="seagreen", weight="bold")
        for c in pts:
            m, col = msty[c["mode"]]
            ax.scatter(c["ce_cost"], c["drop"], s=64, color=col,
                       edgecolor="k", lw=0.6, zorder=4, marker=m)
            ax.annotate(c["tag"], (c["ce_cost"], c["drop"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=6.2)
        if rand:
            ax.scatter([c["ce_cost"] for c in rand],
                       [c["drop"] for c in rand], s=44, marker="x",
                       color="dimgray", lw=1.4, zorder=3,
                       label="random-4 (control)")
        ax.scatter([], [], s=64, marker="o", color="tab:red", edgecolor="k",
                   label="zero")
        ax.scatter([], [], s=64, marker="^", color="tab:blue", edgecolor="k",
                   label="mean-replace")
        ax.set_xlim(-0.06, xmax)
        ax.set_ylim(-12, 120)
        if xlabel:
            ax.set_xlabel("CE cost vs same-net base (nats, e065 bank)")
        if ylabel:
            ax.set_ylabel("fact drop (% of base expression)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(alpha=0.25)

    # ---- panel 1: the sink-coupled knife (e160 stored plane)
    ax = fig.add_subplot(gs[0, 0])
    e160p = [{"tag": f"{c['set']}({c['mode'][0]})",
              "ce_cost": c["ce_cost"], "drop": c["drop_g0_pct"],
              "mode": c["mode"]}
             for c in e160["grid"] if c["bar_eligible"]]
    e160r = [{"ce_cost": c["ce_cost"], "drop": c["drop_g0_pct"]}
             for c in e160["grid"] if c["set"].startswith("R")]
    plane(ax, e160p, e160r,
          "KNIFE 1 — SINK-COUPLED fact (e160 stored, consolidated net)\n"
          "FLAT-CE-FACT-KILL fired: N2 {L1H0,L0H0} kills at CE +0.25 both "
          "modes")

    # ---- panel 2: the inverted knife (e125a site-stored plane)
    ax = fig.add_subplot(gs[0, 1])
    pts = [{"tag": f"{c['set']}({c['mode'][0]})", "ce_cost": c["ce_cost"],
            "drop": c["drop_bar_pct"], "mode": c["mode"]}
           for c in cells if c["net"] == "arm_b" and c["bar_eligible"]]
    rand = [{"ce_cost": c["ce_cost"], "drop": c["drop_bar_pct"]}
            for c in cells if c["net"] == "arm_b"
            and c["set"].startswith("R")]
    short = adj["verdict"].split(" (")[0]
    plane(ax, pts, rand,
          f"KNIFE 2 (INVERTED) — SITE-STORED fact @183 (arm_b, this run)\n"
          f"{short} — bar read on pool_bar (disjoint from selection)")

    # ---- panel 3: the two nets' own censuses
    ax = fig.add_subplot(gs[1, 0])
    arm_sorted = sorted(census, key=lambda c: -c["drop_sel"])[:12]
    heads = [c["head"] for c in arm_sorted]
    x = np.arange(len(heads))
    ax.bar(x - 0.2, [100 * c["drop_sel"] / 0.9880 for c in arm_sorted],
           width=0.4, color="purple", alpha=0.85,
           label="arm_b site-fact drop (this census, pool_sel, zero)")
    ax.bar(x + 0.2, [100 * cons_drops.get((c["l"], c["h"]), 0.0)
                     / G_CONS_BASE for c in arm_sorted], width=0.4,
           color="tab:orange", alpha=0.85,
           label="consolidated (sink-coupled) drop, same head (e133)")
    ax.set_xticks(x)
    ax.set_xticklabels(heads, rotation=45, fontsize=7)
    ax.set_ylabel("fact drop (% of own net's base)")
    ax.set_title("TWO CENSUSES, ONE SET OF COORDINATES: arm_b's own fact-heads "
                 "vs the consolidated net's\n(top-12 by arm_b's census — "
                 "type-selectivity read directly off the paired bars)",
                 fontsize=9)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25, axis="y")

    # ---- panel 4: near cross-check plane
    ax = fig.add_subplot(gs[1, 1])
    pts = [{"tag": f"{c['set']}({c['mode'][0]})", "ce_cost": c["ce_cost"],
            "drop": c["drop_bar_pct"], "mode": c["mode"]}
           for c in cells if c["net"] == "near"]
    plane(ax, pts, [],
          "CROSS-CHECK — second site-stored fact @5-13 (e143_near)\n"
          "do the same sets kill a site-stored fact at a DIFFERENT site?")

    fig.suptitle("E125a — THE INVERTED KNIFE: does the site-stored fact have "
                 "its own flat-CE kill set?", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
