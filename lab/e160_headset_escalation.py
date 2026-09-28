"""E160 — THE HEAD-SET ESCALATION (R46 critic's closing line; QUEUE.md row
e160, DISPATCHED 10:45Z; bars are the QUEUE row VERBATIM, written before
compute).

WHY (R46 CRITIC, closing line): "the lab is one head-ablation away from
either the paper's best figure or its retraction." e150's L0H3-zero came
within 1.4 points of a flat-CE fact-kill (58.6% drop at CE +0.21, single
head, one head-set cell); e150's top-3-mean rider cleared 93% drop but at
CE +0.71 (wreck-priced). The day's noun architecture (sink-coupled memory
type, the removable->irremovable reframe) rests on the claim that NO
flat-CE fact-kill exists. This experiment escalates the head-set until it
clears the bar or the frontier is honestly mapped.

REGISTERED PREDICTION (QUEUE.md e160 row, VERBATIM — no bar shopping):
  "Bars: FLAT-CE-FACT-KILL = some head-set >=60% fact drop at CE <= +0.35
  (the surgical surface EXISTS — paper's best figure; the removable->
  irremovable sentence inverts to 'attackable by head coordinates');
  NEAR-MISS-CONFIRMED = best set 40-60% (the frontier stays open);
  WRECK-ONLY = every >=60% cell costs CE >= +0.70 (the reframe holds; noun
  architecture survives bounded)"

OPERATIONALIZATIONS (fixed before compute):
  * head selection recomputed at runtime from runs/e133/metrics.json
    (nets.graduated arms) exactly as e150 did: drop = base_pz -
    arm.std_install60, ce_cost = arm.ce_r - base.ce_r. The fact-specific
    class (drop >= 0.3 & CE <= +0.35) is EXPECTED to be the singleton
    {L0H3} (e150's finding); the "full fact-specific set" cell therefore
    degenerates to the singleton — stated, not shopped (e150 lineage).
  * escalation ladder = heads ranked by drop within the CE <= +0.35
    class: L0H3, L1H0, L0H0, L1H3, L0H1 (stored e133 numbers). Sets:
    singles (each of the top-4 alone), E2 = {L0H3 + next-1} (an added
    intermediate dose step), E3 = {L0H3 + top2} (== e150's top-3 rider,
    the continuity anchor), E4 = {L0H3 + top3}; k-matched no-L0H3 sets
    N2/N3/N4 (top-k without L0H3, same counts); scatter control = 3
    random same-count draws (4 heads, matched to E4, seed 16003, drawn
    from all 36 heads).
  * modes: zero (e133's head-lesion convention) AND mean-replace (the
    head's 32-dim c_proj-input slice -> its CE_R-bank mean, e150's
    convention). Every escalation cell runs in BOTH modes.
  * reads per cell: fact expression at g0 AND g-12 (Rule 12; e119/e141
    battery ctx = train_text[p-PRE-j:p], install-60), corpus CE_R (e065
    val-windows bank, seed 26502, 60 windows), and a base-skill probe =
    3 held-out corpus windows' NLL (seed 31337, ZEPHYRA-free — prices
    collateral beyond the CE bank).
  * BAR READ (pre-registered): the PRIMARY bar read is g0 (trained
    geometry — the read every e133/e150 drop table uses, and the
    geometry of the 58.6% near-miss). The g-12 numbers are Rule-12
    robustness, recorded per cell, and a SECONDARY union read (fire if
    EITHER geometry clears) is recorded but CANNOT gate the headline bar
    (registered now, before compute, so geometry shopping is impossible).
  * PLANE (bar-eligible cells) = escalation cells on the PRIMARY net
    (consolidated). The random scatter, the install-net column (same
    sets on e048_repro) and the site-stored column (same sets on
    e131_arm_b, site battery read) are CONTROLS — recorded with full
    numbers, flagged out of the bars.
  * adjudication precedence (pre-registered): FLAT-CE-FACT-KILL if any
    bar-eligible cell has drop >= 60% at CE <= +0.35 (g0 read); else if
    kills exist and ALL cost CE >= +0.70 -> WRECK-ONLY; else if kills
    exist strictly inside (+0.35, +0.70) -> AMBIGUOUS-GAP (texture with
    numbers, the e150 precedent); NEAR-MISS-CONFIRMED is reported
    alongside whenever the best flat-CE cell lands 40-60% (it can
    co-fire with WRECK-ONLY, as in e150).

NETS (on disk, gated; eval-only — no training, no fine-tunes; CPU-ONLY,
e152 owns the GPU, e153 also on CPU — threads <= 4, staggered, no
busy-waiting):
  * CONSOLIDATED  runs/checkpoints/e131_consolidated_e113.pt (primary;
    gate bit-exact vs e151's stored cells: g0 p(Z) = 0.7850371599197388,
    CE_R = 1.663516640663147; g-12 cross-check vs e150's stored base
    0.9155886173248291; L0H3-zero@g0 instrument continuity vs e150's
    stored 0.32514816522598267).
  * INSTALL       runs/checkpoints/e048_repro.pt (control column,
    report-only; gate g0 = 0.5563086867332458).
  * SITE-STORED   runs/checkpoints/e131_arm_b_corpus_spliced.pt (site
    comparison, report-only; gates std g0 floor = 0.007800613064318895,
    site onset = 0.9880021214485168; battery = e133's pool-b site
    construction, Z at x-col 184, onset read at row 183).

INSTRUMENT PROVENANCE: battery/eval/surgery instruments are e150's
(lab/e150_flatce_route.py) verbatim — which are themselves e141's
(battery_cell e068/e113/e120, ce_fixed_cpu, val_windows e065 seed 26502,
load_cpu/evl_load) with e133's c_proj pre-hook head conventions
(e001/e038 lesion lineage). Protocol rebuild: corpus seed 1337, SPLICE_RNG
host shuffle, install-60 / held-30 split, mix gate.

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 before torch import),
torch.set_num_threads(4), all evals sequential, tiny staggers between
phases, no training. Outputs: runs/e160/{metrics.json,
headset_escalation.png}. No checkpoints written (eval-only).

Run:  cd lab && python e160_headset_escalation.py    (E160_SMOKE=1 -> shakedown)
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # CPU-ONLY (e152 owns the GPU)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                           # noqa: E402

torch.set_num_threads(4)                              # modest (shared CPU)

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
common.DEVICE = "cpu"
from common import Cfg, CharCorpus, TinyGPT, run_dir, save_json   # noqa: E402
import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E160_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
CONS_CK = CKPT_DIR / "e131_consolidated_e113.pt"   # primary
INST_CK = CKPT_DIR / "e048_repro.pt"               # install control (report-only)
ARMB_CK = CKPT_DIR / "e131_arm_b_corpus_spliced.pt"  # site-stored comparison
E133_METRICS = E43.REPO / "runs" / "e133" / "metrics.json"   # head-selection source table

GEOS = (0, -12)                   # Rule 12: trained g0 + novel g-12

# seeds (fixed before compute)
R_EVAL_SEED = 26502               # e065 CE_R bank seed (verbatim, e150)
SKILL_SEED = 31337                # base-skill probe windows (held-out from the CE bank)
RAND_SEED = 16003                 # random-head scatter draws
CORP_CONT_SEED = 12103            # e133 site-battery corpus-filler seed (verbatim)

# site battery geometry (e133 verbatim)
FACT_PRE, FACT_POST = 12, 12
FACT_LEN = FACT_PRE + len(NAME) + FACT_POST      # 31
SPLICE_AT = 42
Z_XCOL = PRE + SPLICE_AT + FACT_PRE             # 184
SPLICE_ADDR_ROW = Z_XCOL - 1                    # 183
N_PROMPTS = 8 if SMOKE else 30

# gates / references (full precision, from stored metrics)
G_BIT_TOL = 5e-6                  # bit-exact reproduction gate
G_FALLBACK_TOL = 0.05             # e113 G_REPRO convention
G_CONS_REF_PZ = 0.7850371599197388          # e151 G_CONS (bit-exact anchor)
G_CONS_REF_CE = 1.663516640663147           # e151 G_CONS
G_CONS_REF_G12 = 0.9155886173248291         # e150 probe-3 g-12 base
G_L0H3Z_REF = 0.32514816522598267           # e150 L0H3-zero@g0 (instrument anchor)
G_INST_REF = 0.5563086867332458             # e131 G_E048 / e150 G_INST
G_ARMB_REF_STD = 0.007800613064318895       # e133 std battery floor (site net)
G_ARMB_REF_SITE = 0.9880021214485168        # e133 site onset

# registered bars (numeric)
KILL_DROP = 60.0                  # fact-drop % bar (>= 60% = kill)
NEAR_LO, NEAR_HI = 40.0, 60.0     # NEAR-MISS-CONFIRMED band (40-60%)
CE_FLAT = 0.35                    # FLAT-CE-FACT-KILL CE bar (<= +0.35)
CE_WRECK = 0.70                   # WRECK-ONLY CE bar (>= +0.70)

REGISTERED_PREDICTION = {
    "queue_row_verbatim": (
        "L0H3-zero was single-head 58.6% @ CE +0.21 (1.4pts under bar). "
        "Escalate: L0H3 + top-2 and top-3 fact-specific heads (e133's list, "
        "CE<=0.35 class), zero and mean-replace, singly and jointly, graded; "
        "co-measure CE + base skills per cell. Bars: FLAT-CE-FACT-KILL = some "
        "head-set >=60% fact drop at CE <= +0.35 (the surgical surface EXISTS "
        "- paper's best figure; the removable->irremovable sentence inverts "
        "to 'attackable by head coordinates'); NEAR-MISS-CONFIRMED = best set "
        "40-60% (the frontier stays open); WRECK-ONLY = every >=60% cell "
        "costs CE >= +0.70 (the reframe holds; noun architecture survives "
        "bounded)"),
    "bar_read": (
        "PRIMARY bar read = g0 (trained geometry; the e133/e150 drop-table "
        "convention and the geometry of the 58.6% near-miss). g-12 recorded "
        "per cell (Rule 12) + a SECONDARY union read (either geometry) that "
        "is REPORTED but cannot gate the headline bar — registered before "
        "compute, so geometry shopping is impossible."),
    "operationalizations": (
        "head sets recomputed at runtime from runs/e133/metrics.json "
        "(drop >= 0.3 & CE <= 0.35 -> expected singleton {L0H3}; escalation "
        "ladder ranked by drop within the CE<=0.35 class: L0H3, L1H0, L0H0, "
        "L1H3, L0H1). Sets: singles top-4; E2={L0H3+1}, E3={L0H3+top2} "
        "(== e150 top-3 rider), E4={L0H3+top3}; N2/N3/N4 top-k without L0H3 "
        "(k matched); random-4 scatter x3 (seed 16003); modes zero + "
        "mean-replace (CE_R-bank mean); per cell: expr g0 + g-12 install-60, "
        "CE_R (e065 seed 26502), base-skill NLL (3 windows seed 31337). "
        "PLANE = consolidated escalation cells; random/install/site-stored "
        "are controls (full numbers, out of the bars)."),
    "no_bar_shopping": "No bar shopping. Ambiguous => say AMBIGUOUS with "
                       "numbers.",
}

recipe_deviations: list[str] = [
    "E2 = {L0H3 + next-1 head} is an ADDED intermediate dose step between "
    "the dispatched {L0H3} and {L0H3+top2} — it only densifies the "
    "dose-response map (more cells, same bars); the dispatched sets all run.",
    "The dispatched 'full fact-specific set' (drop >= 0.3 & CE <= 0.35 on "
    "e133's stored table) recomputes to the SINGLETON {L0H3} (e150's "
    "finding) — the full-set cell therefore coincides with the singleton "
    "cell; recorded as such, never shopped.",
    "Singles of the three companion heads (L1H0, L0H0, L1H3) run in both "
    "modes so the escalation's additivity can be priced against its parts "
    "(e133's organ-level additivity already FAILED: 0.785 joint vs 1.98 "
    "sum).",
    "Base-skill probe = 3 corpus windows at seed 31337 (disjoint draw from "
    "the CE_R bank's seed 26502); NLL reported per cell and as delta vs the "
    "same net's base — the 'collateral beyond CE' column.",
    "Random scatter draws are matched in COUNT to the largest escalation "
    "set (4 heads) rather than to every size — the null needs one matched "
    "count to bracket the plane; 3 draws x 2 modes.",
    "The install column skips the singles (dose-response is a primary-net "
    "question); it runs the 7 escalation/no-L0H3 sets x 2 modes.",
]

trims: list[str] = []


# ------------------------------------------------------------------ instruments
# (e150 verbatim: load_cpu/evl_load/battery_fwd/ce_fwd/val_windows/
#  head_mean_vec/HeadReplace; e133 verbatim: site battery construction)

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
    (corpus windows only — no fact contexts). e133 organ_stats / e150
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


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e160_smoke" if SMOKE else "e160")
    log(f"E160 THE HEAD-SET ESCALATION (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-only, torch threads {torch.get_num_threads()}, "
        f"cuda available = {torch.cuda.is_available()}")

    # ---------------- protocol rebuild (e141/e150 verbatim)
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

    # batteries: ctx = train_text[p-PRE-j : p] (e119/e141/e150 construction)
    bat_ids = {}
    for j in GEOS:
        for tag, occ in (("install60", install_occ), ("held30", held_occ)):
            cs = [train_text[p - PRE - j: p] for p, _ in occ]
            bat_ids[(j, tag)] = torch.stack([corpus.encode(c) for c in cs])
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    skill_x, skill_y = val_windows(val_ids, val_text, 3, SKILL_SEED)
    skill_xy = (skill_x, skill_y)
    log(f"CE_R bank {tuple(r_eval_x.shape)} (seed {R_EVAL_SEED}); "
        f"base-skill probe {tuple(skill_x.shape)} (seed {SKILL_SEED})")

    # ---------------- site battery pool (e133 pool-b construction verbatim)
    prompts = [train_text[p - PRE: p] for p, _ in held_occ][:N_PROMPTS]
    prompt_ids = torch.stack([corpus.encode(c) for c in prompts])
    gc_ = torch.Generator().manual_seed(CORP_CONT_SEED)
    src = torch.randint(len(train_ids) - (BLOCK - PRE) - 1,
                        (4 if not SMOKE else 1, len(prompts)), generator=gc_)
    filler = torch.stack([train_ids[s: s + BLOCK - PRE] for s in src.flatten()])
    segs = []
    for p, h in install_occ:
        s = (train_text[p - FACT_PRE: p] + NAME
             + train_text[p + len(h): p + len(h) + FACT_POST])
        assert len(s) == FACT_LEN
        segs.append(corpus.encode(s))
    fact_segs = torch.stack(segs)
    fs = fact_segs[torch.arange(filler.shape[0]) % fact_segs.shape[0]]
    cont = torch.cat([filler[:, :SPLICE_AT], fs,
                      filler[:, SPLICE_AT + FACT_LEN:]], 1)
    pool_b = torch.cat([torch.stack(
        [prompt_ids[k % prompt_ids.shape[0]] for k in range(filler.shape[0])]),
        cont], 1)
    name_ids = corpus.encode(NAME)
    assert all(torch.equal(w[Z_XCOL:Z_XCOL + len(NAME)], name_ids)
               for w in pool_b)
    log(f"site battery: {tuple(pool_b.shape)}, ZEPHYRA at x-col {Z_XCOL} "
        f"(address row {SPLICE_ADDR_ROW})")

    @torch.no_grad()
    def site_reads(net: TinyGPT, bs=30) -> dict:
        """e133 battery_site logic: onset p(Z) at row 183 + mean p(true name
        char) over the 7 name positions."""
        onset, per_pos = [], [[] for _ in range(len(name_ids))]
        for i in range(0, pool_b.shape[0], bs):
            w = pool_b[i:i + bs]
            lg, _ = net(w)
            pr = F.softmax(lg, -1)
            for k in range(pr.shape[0]):
                onset.append(float(pr[k, SPLICE_ADDR_ROW, int(zid)]))
                for j in range(len(name_ids)):
                    per_pos[j].append(
                        float(pr[k, SPLICE_ADDR_ROW + j,
                               int(w[k, Z_XCOL + j])]))
        on = torch.tensor(onset)
        allp = torch.tensor([q for pos in per_pos for q in pos])
        return {"onset_pz": float(on.mean()),
                "onset_frac_ge_0.5": float((on >= 0.5).float().mean()),
                "pname_mean_over7": float(allp.mean())}

    # ---------------- head selection from e133's stored table (auditable)
    e133 = json.loads(E133_METRICS.read_text(encoding="utf-8"))
    g133 = e133["nets"]["graduated"]
    b133_p = g133["base"]["std_install60"]["mean_pz"]
    b133_ce = g133["base"]["ce_r"]
    all_heads = []      # [(layer, head, drop, ce_cost)]
    for t, v in g133["arms"].items():
        if not t.startswith("head_"):
            continue
        p_ = t.split("_")[1]
        l_, h_ = int(p_[1:p_.find("h")]), int(p_[p_.find("h") + 1:])
        all_heads.append((l_, h_, b133_p - v["std_install60"],
                          v["ce_r"] - b133_ce))
    ce_class = sorted([hh for hh in all_heads if hh[3] <= CE_FLAT],
                      key=lambda x: -x[2])
    fact_specific = [hh for hh in ce_class if hh[2] >= 0.3]
    top = ce_class[:5]
    log("e133 CE<=0.35 class, top by drop: "
        + ", ".join(f"L{l}H{h} (drop {d:.4f}, CE {c:+.4f})"
                    for l, h, d, c in top))
    log(f"e133 fact-specific set (drop>=0.3 & CE<=0.35): "
        + (", ".join(f"L{l}H{h}" for l, h, _, _ in fact_specific) or "EMPTY"))
    if len(top) < 5:
        raise RuntimeError("e133 CE-class ranking unexpectedly short")

    def hx(l, h):
        return f"L{l}H{h}"

    (t1, t2, t3, t4, t5) = [(l, h) for l, h, _, _ in top]
    singles = [("S:" + hx(*t1), [t1]), ("S:" + hx(*t2), [t2]),
               ("S:" + hx(*t3), [t3]), ("S:" + hx(*t4), [t4])]
    escal = [("E2:L0H3+top1", [t1, t2]),
             ("E3:L0H3+top2", [t1, t2, t3]),
             ("E4:L0H3+top3", [t1, t2, t3, t4])]
    nol0 = [("N2:top2-noL0H3", [t2, t3]),
            ("N3:top3-noL0H3", [t2, t3, t4]),
            ("N4:top4-noL0H3", [t2, t3, t4, t5])]
    # the dispatched "full fact-specific set" — recomputed; degenerates to the
    # singleton (stated in deviations, cell coincides with S:t1)
    full_fs_tag = "FULL-FS:" + "+".join(hx(l, h) for l, h, _, _ in fact_specific)

    # random scatter: 3 draws of 4 heads (matched to E4), seeded
    rr = random.Random(RAND_SEED)
    all_coords = [(l, h) for l in range(6) for h in range(6)]
    rand_sets = []
    for d in range(3):
        rand_sets.append((f"R{d + 1}:random4", sorted(rr.sample(all_coords, 4))))
    log("random scatter draws: "
        + "; ".join(f"{t}=[{','.join(hx(*c) for c in s)}]" for t, s in rand_sets))

    cons_sets = (singles + escal + nol0) if not SMOKE else \
        ([singles[0]] + [escal[1]])
    cons_rand = rand_sets[:1] if SMOKE else rand_sets
    ctl_sets = ([("E1:" + hx(*t1), [t1])] + escal + nol0) if not SMOKE else \
        [("E1:" + hx(*t1), [t1]), escal[1]]

    # ---------------- nets + gates
    log("--- PHASE 0: load + gate the three artifacts ---")
    gates: dict = {}

    net_cons = load_cpu(CONS_CK)
    bz_cons = battery_fwd(net_cons, bat_ids[(0, "install60")], zid)
    ce_cons = ce_fwd(net_cons, *r_eval_xy)
    gates["consolidated"] = {
        "battery_pz": bz_cons["mean_pz"], "ref_pz": G_CONS_REF_PZ,
        "ce_r": ce_cons, "ref_ce": G_CONS_REF_CE,
        "pass": bool(abs(bz_cons["mean_pz"] - G_CONS_REF_PZ) < G_FALLBACK_TOL
                     and abs(ce_cons - G_CONS_REF_CE) < G_FALLBACK_TOL),
        "bit_reproducible": bool(abs(bz_cons["mean_pz"] - G_CONS_REF_PZ)
                                 < G_BIT_TOL
                                 and abs(ce_cons - G_CONS_REF_CE) < G_BIT_TOL)}
    bz_cons12 = battery_fwd(net_cons, bat_ids[(-12, "install60")], zid)
    gates["consolidated_g12"] = {
        "battery_pz": bz_cons12["mean_pz"], "ref": G_CONS_REF_G12,
        "tol": G_BIT_TOL,
        "pass": bool(abs(bz_cons12["mean_pz"] - G_CONS_REF_G12) < G_BIT_TOL)}
    log(f"G_CONS: p(Z) {bz_cons['mean_pz']:.10f} CE_R {ce_cons:.6f} "
        f"| g-12 {bz_cons12['mean_pz']:.10f}: "
        f"{'PASS' if (gates['consolidated']['pass']
                      and gates['consolidated_g12']['pass']) else 'FAIL'}")
    if not (gates["consolidated"]["pass"]
            and gates["consolidated_g12"]["pass"]):
        raise RuntimeError("consolidated checkpoint failed its gate")

    net_inst = load_cpu(INST_CK)
    bz_inst = battery_fwd(net_inst, bat_ids[(0, "install60")], zid)
    gates["install"] = {"battery_pz": bz_inst["mean_pz"], "ref": G_INST_REF,
                        "pass": bool(abs(bz_inst["mean_pz"] - G_INST_REF)
                                     < G_FALLBACK_TOL),
                        "bit_reproducible": bool(abs(bz_inst["mean_pz"]
                                                     - G_INST_REF) < G_BIT_TOL)}
    log(f"G_INST: p(Z) {bz_inst['mean_pz']:.10f}: "
        f"{'PASS' if gates['install']['pass'] else 'FAIL'}")
    if not gates["install"]["pass"]:
        raise RuntimeError("e048_repro checkpoint failed its gate")

    net_armb = load_cpu(ARMB_CK)
    bz_armb = battery_fwd(net_armb, bat_ids[(0, "install60")], zid)["mean_pz"]
    st_armb = site_reads(net_armb)
    gates["site_stored"] = {
        "std_g0_floor": bz_armb, "ref_std": G_ARMB_REF_STD,
        "site_onset": st_armb["onset_pz"], "ref_site": G_ARMB_REF_SITE,
        "pass": bool(abs(bz_armb - G_ARMB_REF_STD) < G_FALLBACK_TOL
                     and abs(st_armb["onset_pz"] - G_ARMB_REF_SITE)
                     < G_FALLBACK_TOL)}
    log(f"G_ARMB: std floor {bz_armb:.10f} site onset "
        f"{st_armb['onset_pz']:.10f}: "
        f"{'PASS' if gates['site_stored']['pass'] else 'FAIL'}")
    if not gates["site_stored"]["pass"]:
        raise RuntimeError("arm_b checkpoint failed its gate")

    # ---------------- the cell machine
    CELLS: list[dict] = []

    def log_cell(c):
        log(f"  [{c['net'][:4]} | {c['set']:18s} | {c['mode']:4s}] "
            f"g0 {c['expr_g0_base']:.4f}->{c['expr_g0']:.4f} "
            f"(drop {c['drop_g0_pct']:5.1f}%) "
            f"g-12 {c['expr_gm12_base']:.4f}->{c['expr_gm12']:.4f} "
            f"(drop {c['drop_gm12_pct']:5.1f}%) "
            f"| CE {c['ce_cost']:+.4f} | NLL3 {c['nll3_delta']:+.4f}"
            + ("  [ctl]" if not c["bar_eligible"] else ""))

    def run_cells(net, net_tag, sets, modes, base, bar_eligible,
                  fact_reader="battery"):
        """base: dict with expr_g0, expr_gm12, ce_r, nll3 (and site onset if
        the site reader). fact_reader 'battery' -> install-60 batteries at
        both geometries; 'site' -> site onset (row 183) + std battery g0."""
        mvcache: dict = {}

        def mv(l, h):
            if (l, h) not in mvcache:
                mvcache[(l, h)] = head_mean_vec(net, l, h, r_eval_x)
            return mvcache[(l, h)]

        for tag, heads in sets:
            for mode in modes:
                replace = {(l, h): (None if mode == "zero" else mv(l, h))
                           for (l, h) in heads}
                with HeadReplace(net, replace):
                    if fact_reader == "battery":
                        e_g0 = battery_fwd(net, bat_ids[(0, "install60")],
                                           zid)["mean_pz"]
                        e_12 = battery_fwd(net, bat_ids[(-12, "install60")],
                                           zid)["mean_pz"]
                    else:
                        sr = site_reads(net)
                        e_g0 = sr["onset_pz"]
                        e_12 = battery_fwd(net, bat_ids[(-12, "install60")],
                                           zid)["mean_pz"]
                        e_g0_detail = sr
                    ce_arm = ce_fwd(net, *r_eval_xy)
                    nll_arm = ce_fwd(net, *skill_xy)
                c = {
                    "net": net_tag, "set": tag,
                    "heads": [hx(l, h) for (l, h) in heads],
                    "n_heads": len(heads), "mode": mode,
                    "bar_eligible": bool(bar_eligible),
                    "expr_g0_base": base["expr_g0"], "expr_g0": e_g0,
                    "drop_g0_pct": 100.0 * (1.0 - e_g0
                                            / max(base["expr_g0"], 1e-12)),
                    "expr_gm12_base": base["expr_gm12"], "expr_gm12": e_12,
                    "drop_gm12_pct": 100.0 * (1.0 - e_12
                                              / max(base["expr_gm12"], 1e-12)),
                    "ce_base": base["ce_r"], "ce_arm": ce_arm,
                    "ce_cost": ce_arm - base["ce_r"],
                    "nll3_base": base["nll3"], "nll3_arm": nll_arm,
                    "nll3_delta": nll_arm - base["nll3"],
                    "kills60_g0": bool(100.0 * (1.0 - e_g0
                                                / max(base["expr_g0"], 1e-12))
                                       >= KILL_DROP),
                    "kills60_gm12": bool(100.0 * (1.0 - e_12
                                                  / max(base["expr_gm12"],
                                                        1e-12)) >= KILL_DROP),
                }
                if fact_reader != "battery":
                    c["site_detail_arm"] = e_g0_detail
                    c["site_detail_base"] = base.get("site_detail")
                CELLS.append(c)
                log_cell(c)

    # ---------------- PHASE 1: bases (no intervention)
    log("--- PHASE 1: bases ---")
    base_cons = {"expr_g0": bz_cons["mean_pz"], "expr_gm12": bz_cons12["mean_pz"],
                 "ce_r": ce_cons, "nll3": ce_fwd(net_cons, *skill_xy)}
    base_cons["held30_g0"] = battery_fwd(net_cons, bat_ids[(0, "held30")], zid)
    base_cons["battery_detail_g0"] = bz_cons
    log(f"cons base: g0 {base_cons['expr_g0']:.6f} g-12 "
        f"{base_cons['expr_gm12']:.6f} CE {base_cons['ce_r']:.6f} "
        f"NLL3 {base_cons['nll3']:.6f}")

    base_inst = {"expr_g0": bz_inst["mean_pz"],
                 "expr_gm12": battery_fwd(net_inst, bat_ids[(-12, "install60")],
                                          zid)["mean_pz"],
                 "ce_r": ce_fwd(net_inst, *r_eval_xy),
                 "nll3": ce_fwd(net_inst, *skill_xy)}
    log(f"inst base: g0 {base_inst['expr_g0']:.6f} g-12 "
        f"{base_inst['expr_gm12']:.6f} CE {base_inst['ce_r']:.6f} "
        f"NLL3 {base_inst['nll3']:.6f}")

    base_armb = {"expr_g0": st_armb["onset_pz"], "site_detail": st_armb,
                 "expr_gm12": battery_fwd(net_armb,
                                          bat_ids[(-12, "install60")],
                                          zid)["mean_pz"],
                 "ce_r": ce_fwd(net_armb, *r_eval_xy),
                 "nll3": ce_fwd(net_armb, *skill_xy)}
    log(f"armB base: site onset {base_armb['expr_g0']:.6f} "
        f"pname {st_armb['pname_mean_over7']:.6f} CE "
        f"{base_armb['ce_r']:.6f} NLL3 {base_armb['nll3']:.6f}")

    # ---------------- PHASE 2: instrument continuity (L0H3-zero == e150)
    log("--- PHASE 2: L0H3-zero@g0 instrument continuity vs e150 ---")
    t1l, t1h = t1
    with HeadReplace(net_cons, {(t1l, t1h): None}):
        l0h3z = battery_fwd(net_cons, bat_ids[(0, "install60")], zid)["mean_pz"]
    gates["l0h3_zero_continuity"] = {
        "value": l0h3z, "ref": G_L0H3Z_REF, "tol": G_BIT_TOL,
        "e133_zero_arm": 0.32514744997024536,
        "pass": bool(abs(l0h3z - G_L0H3Z_REF) < G_BIT_TOL)}
    log(f"L0H3-zero@g0 {l0h3z:.10f} vs e150 {G_L0H3Z_REF:.10f}: "
        f"{'PASS' if gates['l0h3_zero_continuity']['pass'] else 'DRIFT'}")
    if not gates["l0h3_zero_continuity"]["pass"]:
        raise RuntimeError("head-hook instrument drifted from e150's anchor")

    # ---------------- PHASE 3: the escalation grid (primary net)
    log("--- PHASE 3: escalation grid on CONSOLIDATED (bar-eligible) ---")
    run_cells(net_cons, "consolidated", cons_sets, ("zero", "mean"),
              base_cons, bar_eligible=True)
    time.sleep(3.0)                                   # stagger (shared CPU)

    log("--- PHASE 3b: random scatter (control) ---")
    run_cells(net_cons, "consolidated", cons_rand, ("zero", "mean"),
              base_cons, bar_eligible=False)
    time.sleep(3.0)

    # ---------------- PHASE 4: control columns (report-only)
    log("--- PHASE 4a: install-net control column (report-only) ---")
    run_cells(net_inst, "install", ctl_sets, ("zero", "mean"),
              base_inst, bar_eligible=False)
    time.sleep(3.0)

    log("--- PHASE 4b: site-stored control column (report-only) ---")
    run_cells(net_armb, "site_stored", ctl_sets, ("zero", "mean"),
              base_armb, bar_eligible=False, fact_reader="site")

    # ---------------- adjudication (registered — no bar shopping)
    elig = [c for c in CELLS if c["bar_eligible"]]
    flat_cells = [c for c in elig if c["ce_cost"] <= CE_FLAT]
    best_flat = (max(flat_cells, key=lambda c: c["drop_g0_pct"])
                 if flat_cells else None)
    flatce = [c for c in flat_cells if c["drop_g0_pct"] >= KILL_DROP]
    kills = [c for c in elig if c["drop_g0_pct"] >= KILL_DROP]
    wreck = [c for c in kills if c["ce_cost"] >= CE_WRECK]
    gap = [c for c in kills if CE_FLAT < c["ce_cost"] < CE_WRECK]
    near_miss = bool(best_flat is not None
                     and NEAR_LO <= best_flat["drop_g0_pct"] < NEAR_HI)
    # secondary union read (reported, never gating — registered pre-compute)
    flatce_union = [c for c in flat_cells
                    if (c["drop_g0_pct"] >= KILL_DROP
                        or c["drop_gm12_pct"] >= KILL_DROP)]
    best_flat_union = (max(flat_cells,
                           key=lambda c: max(c["drop_g0_pct"],
                                             c["drop_gm12_pct"]))
                       if flat_cells else None)

    def cs(c):
        return (f"{c['set']}/{c['mode']} drop {c['drop_g0_pct']:.1f}% "
                f"(g-12 {c['drop_gm12_pct']:.1f}%) at CE {c['ce_cost']:+.3f}")

    near_txt = (f" | best flat-CE cell: {cs(best_flat)}" if best_flat else "")
    if flatce:
        overall = ("FLAT-CE-FACT-KILL (the surgical surface EXISTS — "
                   "attackable by head coordinates: "
                   + "; ".join(cs(c) for c in flatce) + ")")
    elif kills and len(kills) == len(wreck):
        overall = ("WRECK-ONLY (every >=60% cell costs CE >= +0.70 — the "
                   "reframe holds; the noun architecture survives bounded)"
                   + near_txt)
    elif gap:
        overall = ("AMBIGUOUS-GAP (kills exist strictly inside CE "
                   f"(+{CE_FLAT}, +{CE_WRECK}): "
                   + "; ".join(cs(c) for c in gap) + ")" + near_txt)
    elif not kills:
        overall = ("NO-KILLS (no escalation cell dropped the fact >= 60% at "
                   "any CE — the frontier plateaus below the kill bar)"
                   + near_txt)
    else:
        overall = "ADJUDICATION-ERROR (unreachable)"

    adjudication = {
        "bars_verbatim": REGISTERED_PREDICTION["queue_row_verbatim"],
        "bar_read": REGISTERED_PREDICTION["bar_read"],
        "n_cells_total": len(CELLS),
        "n_bar_eligible": len(elig),
        "FLAT_CE_FACT_KILL": {
            "fires": bool(flatce),
            "cells": [{"set": c["set"], "mode": c["mode"],
                       "heads": c["heads"],
                       "drop_g0_pct": c["drop_g0_pct"],
                       "drop_gm12_pct": c["drop_gm12_pct"],
                       "ce_cost": c["ce_cost"]} for c in flatce]},
        "NEAR_MISS_CONFIRMED": {
            "fires": near_miss,
            "definition": f"best flat-CE cell lands {NEAR_LO:.0f}-"
                          f"{NEAR_HI:.0f}% drop (frontier stays open)",
            "best_flat_cell": (None if best_flat is None else
                               {"set": best_flat["set"],
                                "mode": best_flat["mode"],
                                "drop_g0_pct": best_flat["drop_g0_pct"],
                                "ce_cost": best_flat["ce_cost"]})},
        "WRECK_ONLY": {
            "fires": bool(kills and len(kills) == len(wreck)),
            "kill_cells": [{"set": c["set"], "mode": c["mode"],
                            "drop_g0_pct": c["drop_g0_pct"],
                            "ce_cost": c["ce_cost"]} for c in kills]},
        "secondary_union_read_REPORT_ONLY": {
            "flatce_union_cells": [{"set": c["set"], "mode": c["mode"],
                                    "drop_g0_pct": c["drop_g0_pct"],
                                    "drop_gm12_pct": c["drop_gm12_pct"],
                                    "ce_cost": c["ce_cost"]}
                                   for c in flatce_union],
            "best_union_cell": (None if best_flat_union is None else
                                {"set": best_flat_union["set"],
                                 "mode": best_flat_union["mode"],
                                 "drop_g0_pct": best_flat_union["drop_g0_pct"],
                                 "drop_gm12_pct":
                                     best_flat_union["drop_gm12_pct"],
                                 "ce_cost": best_flat_union["ce_cost"]}),
            "note": "fire-if-either-geometry read, REPORTED but not gating "
                    "(registered before compute)"},
        "dose_response": dose_table(elig),
        "random_scatter_summary": scatter_summary(CELLS),
        "control_columns": control_summary(CELLS),
        "verdict": overall,
    }
    log("=" * 78)
    log(f"E160 VERDICT: {overall}")
    log(f"  FLAT-CE-FACT-KILL {bool(flatce)} | NEAR-MISS-CONFIRMED "
        f"{near_miss} | WRECK-ONLY {bool(kills and len(kills) == len(wreck))}")
    if flatce_union and not flatce:
        log(f"  [secondary union read, non-gating] would fire on: "
            + "; ".join(cs(c) for c in flatce_union))
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e160_headset_escalation",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": ("R46 critic's closing line; QUEUE.md e160 row "
                         "dispatched 10:45Z. Docstring + bars written before "
                         "compute."),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("does a graded escalation of fact-specific head-sets "
                     "find a flat-CE fact-kill (>=60% fact drop at CE "
                     "<= +0.35) on the consolidated net, or does every "
                     "kill stay wreck-priced (>= +0.70)?"),
        "nets": {
            "consolidated": f"runs/checkpoints/{CONS_CK.name} (primary; "
                            f"gated bit-exact vs e151 stored cells)",
            "install_control": f"runs/checkpoints/{INST_CK.name} (report-only)",
            "site_stored_control": f"runs/checkpoints/{ARMB_CK.name} "
                                   f"(report-only)",
            "eval_only": True,
        },
        "selection": {
            "source": "runs/e133/metrics.json nets.graduated (recomputed at "
                      "runtime, auditable)",
            "ce_class_top5": [{"head": hx(l, h), "drop": d, "ce_cost": c}
                              for l, h, d, c in top],
            "fact_specific_set": [{"head": hx(l, h), "drop": d, "ce_cost": c}
                                  for l, h, d, c in fact_specific],
            "full_fs_tag": full_fs_tag,
            "degenerate_note": (
                "the dispatched full fact-specific set (drop>=0.3 & "
                "CE<=0.35) recomputes to the singleton "
                f"{full_fs_tag} — the full-set cell coincides with "
                f"S:{hx(*t1)} (stated, e150 lineage)"
                if len(fact_specific) == 1 else
                f"recomputes to {len(fact_specific)} heads: {full_fs_tag}"),
            "random_draws": [{"tag": t, "heads": [hx(*c) for c in s]}
                             for t, s in rand_sets],
        },
        "bases": {
            "consolidated": {k: v for k, v in base_cons.items()},
            "install": base_inst,
            "site_stored": base_armb,
        },
        "gates": {"G_SPLICE": G_SPLICE, "consolidated": gates["consolidated"],
                  "consolidated_g12": gates["consolidated_g12"],
                  "install": gates["install"],
                  "site_stored": gates["site_stored"],
                  "l0h3_zero_continuity": gates["l0h3_zero_continuity"]},
        "grid": CELLS,
        "adjudication": adjudication,
        "honesty_reflex": {
            "additivity_assumption": "the escalation ladder assumes head "
                "effects can be summed by set union; e133's joint-vs-sum "
                "already FAILED at organ level (0.785 joint vs 1.98 sum) — "
                "the dose-response column exists to expose the same failure "
                "mode here (a set can kill less than its strongest member, "
                "as L0H3+L1H0 may if L1H0's route partially restores the "
                "read).",
            "joint_interactions": "mean-replace of multiple heads replaces "
                "each head's slice with its ALL-HEADS-INTACT bank mean — the "
                "replacement vectors are not recomputed under the joint "
                "ablation; zero mode has no such coupling but both modes "
                "ablate coordinates, not circuits: downstream heads may "
                "re-route and the measured drop is the NET effect.",
            "mean_replace_mode_choice": "mean-replace is itself an "
                "intervention (the bank mean over corpus windows, no fact "
                "contexts); zero is e133's convention. The two modes bracket "
                "the surgical range and DISAGREE sharply on L0H3 (e150: "
                "12.1% vs 58.6% drop) — any flat-CE kill claim must name "
                "its mode, and a kill in only one mode is a weaker claim "
                "than one that survives both.",
            "single_lineage": "the primary net is ONE consolidated lineage "
                "(e113 recipe, seed 1337 chain); the install and site-stored "
                "controls change the memory phase, not the seed — the "
                "frontier map is line-specific until e157's second-lineage "
                "conversion replication lands.",
            "ce_bank_is_position_blind": "CE_R prices general LM damage on "
                "fact-free corpus windows; the base-skill NLL3 column is a "
                "second, disjoint-window read of the same — neither prices "
                "fact-geometry damage specifically (that dissociation is "
                "the point AND the confound, e150's note verbatim).",
            "bar_geometry": "the primary bar read is g0; the union "
                "(either-geometry) read is recorded and non-gating — if it "
                "differs from the primary, the difference is REPORTED, not "
                "shopped.",
        },
        "trims": trims,
        "recipe_deviations": recipe_deviations,
        "ckpt_inventory": {
            "saved": {},
            "external_used": [f"runs/checkpoints/{p.name}"
                              for p in (CONS_CK, INST_CK, ARMB_CK)],
            "note": "eval-only: no checkpoints written or regenerated",
        },
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": 2739072,
                   "device": "cpu", "torch_threads": torch.get_num_threads(),
                   "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "headset_escalation.png", CELLS, adjudication, base_cons)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'headset_escalation.png'}")
    log(f"total {time.time() - T0:.1f}s")


# ------------------------------------------------------------------ summaries

def dose_table(elig):
    """drop% by set size and mode (g0 primary read; g-12 shown too), with the
    sum-of-singles column that prices additivity per escalation set."""
    singles = {(c["mode"], c["heads"][0]): c["drop_g0_pct"]
               for c in elig if c["n_heads"] == 1}
    out = []
    for mode in ("zero", "mean"):
        for c in sorted([c for c in elig if c["mode"] == mode],
                        key=lambda c: c["n_heads"]):
            ssum = (sum(singles.get((mode, hh), 0.0) for hh in c["heads"])
                    if c["n_heads"] > 1 else None)
            out.append({"mode": mode, "set": c["set"],
                        "n_heads": c["n_heads"],
                        "has_L0H3": "L0H3" in c["heads"],
                        "drop_g0_pct": c["drop_g0_pct"],
                        "drop_gm12_pct": c["drop_gm12_pct"],
                        "ce_cost": c["ce_cost"],
                        "sum_single_drops": ssum})
    return out


def scatter_summary(cells):
    rs = [c for c in cells if c["set"].startswith("R")]
    if not rs:
        return {"note": "no random cells (smoke)"}
    drops = [c["drop_g0_pct"] for c in rs]
    ces = [c["ce_cost"] for c in rs]
    return {"n_cells": len(rs),
            "drop_g0_pct_min": min(drops), "drop_g0_pct_max": max(drops),
            "ce_cost_min": min(ces), "ce_cost_max": max(ces),
            "any_kill": any(c["kills60_g0"] for c in rs)}


def control_summary(cells_all):
    inst = [c for c in cells_all if c["net"] == "install"]
    site = [c for c in cells_all if c["net"] == "site_stored"]
    def best(col):
        if not col:
            return None
        b = max(col, key=lambda c: c["drop_g0_pct"])
        return {"set": b["set"], "mode": b["mode"],
                "drop_g0_pct": b["drop_g0_pct"], "ce_cost": b["ce_cost"]}
    return {"install_best_drop": best(inst),
            "site_stored_best_drop": best(site),
            "note": "report-only: does the same head-set kill the "
                    "install-phase fact or the site-stored fact? A "
                    "consolidated-only kill is phase-specific surgery; a "
                    "universal kill is a shared readout circuit."}


# ------------------------------------------------------------------ plot

def plot(path, cells, adj, base_cons):
    """THE fact-drop vs CE-cost plane + dose-response + collateral + controls."""
    fig = plt.figure(figsize=(16.0, 12.5))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.25, 1.0))

    elig = [c for c in cells if c["bar_eligible"]]
    rand = [c for c in cells if c["set"].startswith("R")]
    inst = [c for c in cells if c["net"] == "install"]
    site = [c for c in cells if c["net"] == "site_stored"]
    xmax = max(1.0, max((c["ce_cost"] for c in cells), default=0.8) + 0.25)

    # ---- main panel: fact-drop vs CE cost (g0 primary read)
    ax = fig.add_subplot(gs[0, :])
    ax.axvspan(0, CE_FLAT, color="seagreen", alpha=0.07)
    ax.fill_between([-0.05, CE_FLAT], KILL_DROP, 118, color="seagreen",
                    alpha=0.18)
    ax.fill_between([CE_WRECK, xmax], KILL_DROP, 118, color="firebrick",
                    alpha=0.10)
    ax.axhline(KILL_DROP, ls="--", color="gray", lw=1.1)
    ax.axvline(CE_FLAT, ls="--", color="seagreen", lw=1.2)
    ax.axvline(CE_WRECK, ls="--", color="firebrick", lw=1.2)
    ax.text(CE_FLAT * 0.5, 113, "FLAT-CE region\n(drop>=60% @ CE<=+0.35)",
            ha="center", fontsize=8.5, color="seagreen", weight="bold")
    ax.text((CE_WRECK + xmax) / 2, 113, "WRECK-priced kills (CE>=+0.70)",
            ha="center", fontsize=8.5, color="firebrick")
    ax.text((CE_FLAT + CE_WRECK) / 2, -6, "gap (+0.35,+0.70)", ha="center",
            fontsize=7.5, color="darkgoldenrod")
    msty = {"zero": ("o", "tab:red"), "mean": ("^", "tab:blue")}
    for c in elig:
        m, col = msty[c["mode"]]
        ax.scatter(c["ce_cost"], c["drop_g0_pct"], s=70, color=col,
                   edgecolor="k", lw=0.6, zorder=4, marker=m)
        ax.annotate(f"{c['set']}({c['mode'][0]})",
                    (c["ce_cost"], c["drop_g0_pct"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=6.6)
        ax.plot([c["ce_cost"]] * 2, [c["drop_g0_pct"], c["drop_gm12_pct"]],
                ls=":", color="dimgray", lw=0.7, zorder=2)
        ax.scatter(c["ce_cost"], c["drop_gm12_pct"], s=18, color=col,
                   edgecolor="none", alpha=0.45, marker=m, zorder=3)
    if rand:
        ax.scatter([c["ce_cost"] for c in rand],
                   [c["drop_g0_pct"] for c in rand], s=46, marker="x",
                   color="dimgray", lw=1.4, zorder=3,
                   label="random-4 (control)")
    if inst:
        ax.scatter([c["ce_cost"] for c in inst],
                   [c["drop_g0_pct"] for c in inst], s=40, marker="s",
                   facecolor="none", edgecolor="darkorange", lw=1.2,
                   zorder=3, label="install net (control)")
    if site:
        ax.scatter([c["ce_cost"] for c in site],
                   [c["drop_g0_pct"] for c in site], s=40, marker="s",
                   facecolor="none", edgecolor="purple", lw=1.2, zorder=3,
                   label="site-stored net (control)")
    ax.scatter([], [], s=70, marker="o", color="tab:red", edgecolor="k",
               label="zero mode (g0 read)")
    ax.scatter([], [], s=70, marker="^", color="tab:blue", edgecolor="k",
               label="mean mode (g0 read)")
    ax.scatter([], [], s=18, color="gray", alpha=0.5,
               label="g-12 (Rule-12, dotted stem)")
    ax.set_xlabel("CE cost vs same-net baseline (nats, e065 bank seed 26502)")
    ax.set_ylabel("fact drop (% of base expression, g0 read)")
    ax.set_xlim(-0.06, xmax)
    ax.set_ylim(-10, 120)
    short = adj["verdict"].split(" (")[0]
    ax.set_title(f"E160 — THE HEAD-SET ESCALATION: {short}\n"
                 f"FLAT-CE-FACT-KILL {adj['FLAT_CE_FACT_KILL']['fires']} | "
                 f"NEAR-MISS {adj['NEAR_MISS_CONFIRMED']['fires']} | "
                 f"WRECK-ONLY {adj['WRECK_ONLY']['fires']} — dotted stems "
                 f"connect each cell's g0 (big) and g-12 (small) drops",
                 fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.25)

    # ---- dose-response: drop% vs n_heads
    ax = fig.add_subplot(gs[1, 0])
    for mode, col in (("zero", "tab:red"), ("mean", "tab:blue")):
        es = sorted([c for c in elig if c["mode"] == mode
                     and c["set"].startswith(("S:L0H3", "E"))],
                    key=lambda c: c["n_heads"])
        if es:
            ax.plot([c["n_heads"] for c in es],
                    [c["drop_g0_pct"] for c in es], "o-", color=col,
                    label=f"L0H3-escalation {mode} (g0)")
            ax.plot([c["n_heads"] for c in es],
                    [c["drop_gm12_pct"] for c in es], "o--", color=col,
                    alpha=0.45, label=f"L0H3-escalation {mode} (g-12)")
        ns = sorted([c for c in elig if c["mode"] == mode
                     and c["set"].startswith("N")],
                    key=lambda c: c["n_heads"])
        if ns:
            ax.plot([c["n_heads"] for c in ns],
                    [c["drop_g0_pct"] for c in ns], "s-", color=col,
                    alpha=0.55, mfc="none",
                    label=f"no-L0H3 {mode} (g0)")
    rd_ = [c for c in rand if c["mode"] == "zero"]
    if rd_:
        lo = min(c["drop_g0_pct"] for c in rd_)
        hi = max(c["drop_g0_pct"] for c in rd_)
        ax.plot([4, 4], [lo, hi], color="dimgray", lw=6, alpha=0.35)
        ax.annotate("random-4\nrange", (4, (lo + hi) / 2), fontsize=7,
                    textcoords="offset points", xytext=(8, 0),
                    color="dimgray")
    ax.axhline(KILL_DROP, ls="--", color="gray", lw=1.0, label="60% kill bar")
    ax.axhline(NEAR_LO, ls=":", color="seagreen", lw=0.9,
               label="near-miss band 40-60%")
    ax.set_xlabel("head-set size (n heads in the set)")
    ax.set_ylabel("fact drop (% g0)")
    ax.set_title("DOSE-RESPONSE: drop vs set size (singles -> E4; "
                 "no-L0H3 matched; random-4 null)", fontsize=9.5)
    ax.legend(fontsize=6.6, loc="best")
    ax.grid(alpha=0.25)

    # ---- collateral: NLL3 delta vs CE cost
    ax = fig.add_subplot(gs[1, 1])
    for c in elig:
        m, col = msty[c["mode"]]
        ax.scatter(c["ce_cost"], c["nll3_delta"], s=55, color=col,
                   edgecolor="k", lw=0.5, marker=m, zorder=4)
        ax.annotate(f"{c['set']}", (c["ce_cost"], c["nll3_delta"]),
                    textcoords="offset points", xytext=(4, 3), fontsize=6.0)
    if rand:
        ax.scatter([c["ce_cost"] for c in rand],
                   [c["nll3_delta"] for c in rand], marker="x", s=40,
                   color="dimgray", lw=1.3, label="random-4 (control)")
    ax.axvline(CE_FLAT, ls="--", color="seagreen", lw=1.1)
    ax.axvline(CE_WRECK, ls="--", color="firebrick", lw=1.1)
    ax.set_xlabel("CE cost (nats, e065 bank)")
    ax.set_ylabel("base-skill NLL3 delta (3 held-out windows, seed 31337)")
    ax.set_title("COLLATERAL BEYOND CE: NLL3 delta vs CE cost "
                 "(escalation cells; the two columns must agree in sign)",
                 fontsize=9.5)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
