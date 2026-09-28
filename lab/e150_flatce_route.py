"""E150 — THE FLAT-CE ROUTE TEST (R45 critic attack 2, the ultimatum;
REGISTERED — bars are QUEUE.md row e150 verbatim, written before compute).

WHY (Review 45 CRITIC, attack 2, verbatim): "the lab owns NO row-0-plane
intervention that kills the fact without wrecking the LM — every killing
cell sits at CE +0.70 to +4.44. 'Routed through row-0 presence' and 'dies
whenever the net dies' are observationally equivalent except the perm
spare, which is itself indistinguishable from 'the fact never consults
row-0's direction.' Cures named, cheap: perm@novel-geometry,
forced-off-sink mask, L0H3-class head ablation (the only flat-CE
fact-kill candidate, 0.46 drop at 0.21 CE)." This experiment runs all
three cures plus the two amendment controls (W014's never-consults null,
T081/R45-attack-3's unmeasured norm threshold). It alone decides whether
'routed' is a memory property or a wreck artifact.

NETS (on disk, gated; eval-only — no training, no fine-tunes; CPU-ONLY,
e147 owns the GPU):
  * CONSOLIDATED  runs/checkpoints/e131_consolidated_e113.pt (primary;
    gates g0 battery p(Z) = 0.7850371599197388, CE_R = 1.663516640663147).
  * R@150 / R@300 runs/checkpoints/e119_r_jittered_s150.pt / _s300.pt
    (the novel-geometry cells; gates g0 = 0.5597274303436279 /
    0.7753651738166809, R150 g-12 = 0.8130165338516235).
  * INSTALL       runs/checkpoints/e048_repro.pt (W013's forced-off-sink
    differential control; gate g0 = 0.5563086867332458).
  * SITE-STORED   runs/checkpoints/e131_arm_b_corpus_spliced.pt (the
    mask's specificity control — its fact is site-read, not routed; gates
    std g0 floor = 0.007800613064318895, site onset = 0.9880021214485168).

REGISTERED PREDICTION (QUEUE.md e150 row, VERBATIM — no bar shopping):
  "Bars: FLAT-CE-ROUTE = any intervention kills fact >=60% at CE cost
  <=+0.35; ALL-KILLS-WRECK = every fact-kill costs CE >=+0.70 (T081/T082
  hard-bounded — 'routed' becomes a wreck statement); DIRECTION-CONSULTED
  = fact-at-position-0 scramble kills (tenant reading dies);
  PRESENCE-AT-NOVEL = perm@g-12 spares >=80% of g-12 level"

THE FIVE PROBES (dispatch verbatim):
  (1) PERM AT BOTH GEOMETRIES — direction-permutation of wpe[0] (e141's
      arm) at g-12 AND g0: if the fact survives perm@g-12 at ~its g-12
      level, presence-only holds at novel geometry.
  (2) FORCED-OFF-SINK — additive attention mask blocking keys at
      position 0 (all layers/heads) at eval: the sink's ATTENTIONAL role
      removed while wpe[0] remains in the stream. The intervention that
      might kill the route at flat-ish CE.
  (3) FACT-SPECIFIC HEAD ABLATION — from e133's tables the fact-specific
      head set (expression-drop >= 0.3, CE <= 0.35); ablate singly and
      jointly (mean-replace head outputs). The only registered flat-CE
      fact-kill candidate.
  (4) FACT-AT-POSITION-0 — contexts truncated so the fact's onset sits
      AT position 0, then direction-scramble wpe[0]: W014's
      never-consults control.
  (5) NORM LADDER — wpe[0] rescaled to norms {0.07, 0.15, 0.25, 0.35}
      (direction kept): locate the threshold in (0.066, 0.382).

OPERATIONALIZATIONS (fixed before compute):
  * kill(60%) = expr_arm <= 0.40 x expr_base; the lab's die bar (0.5x,
    e131 convention) is reported alongside every cell; retention =
    expr_arm / expr_base. CE cost = CE_R(arm) - CE_R(base of the SAME
    net, no intervention), e065 val-windows bank (seed 26502, 60
    windows) — a CE column on EVERY cell.
  * PLANE (bar-eligible cells) = interventions on the ROUTED-fact nets
    (consolidated + R@150/R@300). The install-mask cell (W013's
    differential) and the site-stored-mask cell are specificity
    CONTROLS — recorded with full numbers, flagged out of the bars. The
    top-3 joint head ablation and the R150 norm ladder are REPORT-ONLY
    riders. No bar shopping: ambiguous => say so with numbers.
  * perm = seeded 192-dim permutation of wpe[0] (norm-preserving,
    presence-preserving, direction-destroying), PERM_DIM_SEED = 14101 =
    e141's exact seed => the SAME permutation as e141's perm@t0 arm
    (continuity by construction).
  * forced-off-sink mask = additive -inf on the attention logits at key
    column 0 for queries 1..T-1, ALL layers, ALL heads, eval only,
    implemented in a custom forward (common.TinyGPT replicated; SDPA
    with an explicit float mask). Query row 0 keeps its self-attention
    (a fully-masked row would be undefined). Instrument gate: the custom
    forward with a causal-only mask must reproduce the standard forward
    (max |dp(Z)| over the battery < 1e-3, max |dlogit| < 1e-2 — actuals
    recorded); each mask cell's base is the SAME custom forward with
    causal-only mask, so the delta is the pure mask effect.
  * head set recomputed at runtime from runs/e133/metrics.json
    (graduated arms) by the dispatch criterion drop >= 0.3 & CE <= 0.35
    — expected singleton {L0H3} (drop 0.4599 at CE +0.2077 stored);
    'jointly' then degenerates to the singleton (stated, not shopped).
    REPORT-ONLY rider: joint mean-replace of the top-3 heads by the same
    table ({L0H3, L1H0, L0H0} stored: 0.4599/0.1494/0.1258).
    mean-replace = the head's 32-dim slice of c_proj's input replaced by
    its mean over the CE_R bank (corpus windows, no fact contexts);
    zero arm = e133's convention (comparability).
  * fact-at-position-0: fact segment [NAME + 12 post-host chars + corpus
    continuation] (130 tokens, Z at column 0); expression = mean
    p(true next name char) over reads 0..5 (within-name continuation,
    e133's pname convention minus the onset read which cannot exist at
    col 0). Control cell = the e133 segment geometry [12 pre-host chars
    + NAME + 12 post-host chars + continuation], name at cols 12..18,
    reads 11..17 (onset + 6 within-name). DIRECTION-CONSULTED reads the
    col-0 cell under perm (die bar 0.5x; the 60% number also reported).
  * ladder = wpe[0] x (target_norm / |wpe[0]|), targets {0.07, 0.15,
    0.25, 0.35} + the 0.0 removal anchor, consolidated @g0 + @g-12;
    R@150 @g-12 ladder = rider. Threshold bracket = (largest dying norm,
    smallest surviving norm), survive >= 0.8x / die <= 0.5x.
  * PRESENCE-AT-NOVEL reads BOTH R nets at g-12 (fires iff both retain
    >= 80% under perm); consolidated @g-12 is the supporting cell.

INSTRUMENT PROVENANCE: battery/eval/surgery instruments are e141's
(lab/e141_sinkkey_mechanism.py) VERBATIM — battery_cell (e068/e113/e120),
ce_fixed_cpu, val_windows (e065 seed 26502), modified_wpe (e131
confinement gate), load_cpu/evl_load — and the protocol rebuild (corpus
seed 1337, SPLICE_RNG host shuffle, install-60 / held-30 split, mix
gate), with batteries built ctx = train_text[p-PRE-j : p] exactly as
e119/e141 built the g0/g-12 cells. The site battery (for the site-stored
control net) is e133's pool-b construction verbatim (corpus filler seed
12103, splice-at-42, Z at x-col 184, onset read at row 183). Head hooks
follow e133's c_proj pre-hook conventions (e001/e038 lesion lineage).

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 before torch import;
e147 owns the GPU — never touched), torch.set_num_threads(4) (modest —
other CPU agents live), all evals sequential, no busy-waiting, no
training. Outputs: runs/e150/{metrics.json, flatce_route.png}. No
checkpoints written (eval-only).

Run:  cd lab && python e150_flatce_route.py     (E150_SMOKE=1 -> shakedown)
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # CPU-ONLY (e147 owns the GPU)

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

SMOKE = os.environ.get("E150_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
CONS_CK = CKPT_DIR / "e131_consolidated_e113.pt"   # consolidated (primary)
R150_CK = CKPT_DIR / "e119_r_jittered_s150.pt"     # road-R @150 (novel geom)
R300_CK = CKPT_DIR / "e119_r_jittered_s300.pt"     # road-R @300 (registered default)
INST_CK = CKPT_DIR / "e048_repro.pt"               # install-phase (mask differential control)
ARMB_CK = CKPT_DIR / "e131_arm_b_corpus_spliced.pt"  # site-stored (mask specificity control)
E133_METRICS = E43.REPO / "runs" / "e133" / "metrics.json"   # head-selection source table

GEOS = (-12, 0)                   # g0 (trained) + g-12 (e119's novel geometry)
R_NETS = ("R150",) if SMOKE else ("R150", "R300")

# seeds (fixed before compute)
R_EVAL_SEED = 26502               # e065 CE_R bank seed (verbatim)
PERM_DIM_SEED = 14101             # e141's PERM_DIM_SEED -> the SAME permutation

# site battery geometry (e133 verbatim)
FACT_PRE, FACT_POST = 12, 12
FACT_LEN = FACT_PRE + len(NAME) + FACT_POST      # 31
SPLICE_AT = 42
Z_XCOL = PRE + SPLICE_AT + FACT_PRE             # 184
SPLICE_ADDR_ROW = Z_XCOL - 1                    # 183
CORP_CONT_SEED = 12103
N_PROMPTS = 8 if SMOKE else 30

# ladder norms (registered) + removal anchor
LADDER = (0.07, 0.15, 0.25, 0.35)
ANCHOR_ZERO = 0.0

# gates / references (full precision, from stored metrics)
G_BIT_TOL = 5e-6                  # loaded-checkpoint reproduction gate
G_FALLBACK_TOL = 0.05             # e113 G_REPRO convention
G_CONS_REF_PZ = 0.7850371599197388          # e131 none__g+0__install60
G_CONS_REF_CE = 1.663516640663147           # e131 none__ce_r
G_INST_REF = 0.5563086867332458             # e131 G_E048 / e119 line
G_R150_REF_PZ = 0.5597274303436279          # e119 battery_table r_jittered g0
G_R150_REF_G12 = 0.8130165338516235         # e119 battery_table r_jittered g-12
G_R300_REF_PZ = 0.7753651738166809          # e119 calibration dial_default p_r
G_ARMB_REF_STD = 0.007800613064318895       # e133 std battery floor (site net)
G_ARMB_REF_SITE = 0.9880021214485168        # e133 site onset

# registered bars (numeric)
KILL_RET = 0.40                   # kill = fact-drop >= 60% (FLAT-CE-ROUTE)
DIE_RET = 0.50                    # lab die bar (0.5x, e131 convention) — reported
CE_FLAT = 0.35                    # FLAT-CE-ROUTE CE bar (<= +0.35)
CE_WRECK = 0.70                   # ALL-KILLS-WRECK CE bar (>= +0.70)
SPARE_BAR = 0.80                  # PRESENCE-AT-NOVEL (>= 80% retained)

REGISTERED_PREDICTION = {
    "queue_row_verbatim": (
        "On consolidated + R nets: (1) perm@g-12 AND perm@g0 "
        "(direction-permutation — fact survives => presence-only holds at both "
        "geometries); (2) FORCED-OFF-SINK attention mask (mask attention to "
        "position 0 at eval — removes the sink's attentional role at flat-ish "
        "CE); (3) L0H3-class fact-specific head ablation (e133's flat-CE "
        "fact-kill candidate, alone and as a set); (4) FACT-AT-POSITION-0 "
        "scramble (W014's never-consults control); (5) NORM LADDER on wpe[0] "
        "(0.07/0.15/0.25/0.35 — the threshold in (0.066,0.382)). Bars: "
        "FLAT-CE-ROUTE = any intervention kills fact >=60% at CE cost <=+0.35; "
        "ALL-KILLS-WRECK = every fact-kill costs CE >=+0.70 (T081/T082 "
        "hard-bounded — 'routed' becomes a wreck statement); "
        "DIRECTION-CONSULTED = fact-at-position-0 scramble kills (tenant "
        "reading dies); PRESENCE-AT-NOVEL = perm@g-12 spares >=80% of g-12 "
        "level"),
    "operationalizations": (
        "kill(60%) = expr <= 0.40 x base (die bar 0.5x reported alongside); "
        "CE cost on the e065 bank (seed 26502); PLANE = bar-eligible cells on "
        "the routed-fact nets (consolidated + R150/R300) — install-mask "
        "(differential) and site-stored-mask (specificity) are controls, "
        "top-3 joint heads and R150 ladder are report-only riders; perm seed "
        "14101 = e141's permutation; mask = -inf on key column 0 for queries "
        ">= 1, all layers/heads, eval-only custom forward gated against the "
        "standard forward; head set recomputed from runs/e133/metrics.json by "
        "drop >= 0.3 & CE <= 0.35 (expected singleton L0H3 — 'jointly' "
        "degenerates, stated); mean-replace = head slice -> its CE_R-bank "
        "mean; fact-at-position-0 = [NAME + 12 post-host + continuation], "
        "expression = mean p(next name char) over reads 0..5, control cell = "
        "e133 segment geometry (name at cols 12..18); ladder = norm rescale "
        "direction-kept + 0.0 anchor; PRESENCE-AT-NOVEL fires iff BOTH R nets "
        "retain >= 80% at g-12 under perm."),
    "no_bar_shopping": "No bar shopping. Ambiguous => say AMBIGUOUS with "
                       "numbers.",
}

recipe_deviations: list[str] = [
    "Eval-only battery: every net is a LOADED, gated artifact (e131/e119/"
    "e048/e133-lineage checkpoints); nothing regenerated, nothing trained — "
    "the dispatch's eval-only constraint.",
    "e133's head-selection criterion (drop >= 0.3 & CE <= 0.35) yields a "
    "SINGLETON {L0H3} on the stored graduated table — the dispatched "
    "'singly and jointly' therefore collapses to the singleton; the top-3 "
    "joint ({L0H3, L1H0, L0H0} by the same table) runs as a REPORT-ONLY "
    "rider so a 'joint' cell still exists. Selection recomputed at runtime "
    "from runs/e133/metrics.json (auditable), not hand-copied.",
    "Forced-off-sink mask: query row 0 keeps its self-attention (blocking a "
    "query's ONLY legal key would leave an undefined softmax row); the "
    "battery reads p(Z) at the LAST position, so rows 1..T-1 — every read "
    "that matters — lose key 0 entirely.",
    "Each mask cell's base is the custom forward with a CAUSAL-ONLY mask "
    "(not the standard SDPA path), so recorded deltas isolate the mask "
    "effect from any kernel-path numerics; the custom-causal vs standard "
    "equivalence gate is recorded once per net.",
    "Fact-at-position-0 cells cannot carry an onset read (it would sit at "
    "position -1); expression = the 6 within-name continuation reads, "
    "matched in the col-12 control cell (reads 12..17) so the col-0 vs "
    "col-12 comparison is read-matched.",
    "Site-stored control net (e131 arm b): its primary battery is the "
    "e133 183-geometry site pool (onset p(Z) at row 183); its std g0 floor "
    "(0.0078) is gate-checked only.",
]

trims: list[str] = []


# ------------------------------------------------------------------ instruments
# (e141 verbatim: load_cpu/evl_load/battery_cell/ce_fixed_cpu/val_windows/
#  modified_wpe; e133 verbatim: site battery construction, c_proj pre-hook
#  head conventions; provenance comments inline)

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
def battery_fwd(net: TinyGPT, ids: torch.Tensor, zid: int, fwd=None,
                bs=30) -> dict:
    """e068/e113/e120 battery on CPU: p(Z) at the last position. `fwd`
    defaults to the standard forward; pass forward_custom for mask cells."""
    net.eval()
    f = fwd if fwd is not None else net
    pzs, amax = [], 0.0
    for i in range(0, ids.shape[0], bs):
        lg, _ = f(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
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
    """e131's confinement gate, generalized to row-value surgery: at most
    `row`'s elements change, every other tensor bit-identical."""
    out = {k: v.clone() for k, v in sd.items()}
    out["wpe.weight"][row] = value
    d = out["wpe.weight"] != sd["wpe.weight"]
    changed_rows = sorted(set(int(r) for r in torch.nonzero(d)[:, 0].tolist()))
    n = int(d.sum().item())
    others = all(torch.equal(out[k], sd[k]) for k in out if k != "wpe.weight")
    ok_rows = changed_rows in ([], [row])
    gate = {"row": row, "n_elements_changed": n,
            "changed_rows": changed_rows,
            "identity": bool(n == 0),
            "confined": bool(ok_rows),
            "others_bit_identical": bool(others),
            "pass": bool(ok_rows and others)}
    return out, gate


# ---- the forced-off-sink custom forward (probe 2's instrument) ----

@torch.no_grad()
def forward_custom(net: TinyGPT, idx, targets=None, block_key0=False):
    """common.TinyGPT.forward replicated with an explicit additive attention
    mask. block_key0=True: attention TO key position 0 blocked for queries
    1..T-1, ALL layers, ALL heads (row 0 keeps its self-attention). Eval
    only — never trained through."""
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
        mask[0, 0] = 0.0                 # keep row 0's only legal key
    for block in net.h:
        xin = block.ln1(x)
        q, k, v = block.attn.c_attn(xin).split(C, dim=2)
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + block.attn.c_proj(y)
        x = x + block.mlp(block.ln2(x))
    logits = net.lm_head(net.ln_f(x))
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.reshape(-1))
    return logits, loss


def fwd_causal(net):
    return lambda idx, targets=None: forward_custom(
        net, idx, targets=targets, block_key0=False)


def fwd_offsink(net):
    return lambda idx, targets=None: forward_custom(
        net, idx, targets=targets, block_key0=True)


@torch.no_grad()
def sink_mass_at_read(net: TinyGPT, ids: torch.Tensor, pos: int,
                      bs=30) -> dict:
    """Manipulation check for the mask: mean attention mass on key 0 at the
    read position, per layer (mean over heads/batch; e133's manual-softmax
    measurement lineage). Post-mask this is exactly 0 by construction; this
    measures the pre-mask mass the mask removes."""
    cfg = net.cfg
    H, D = cfg.n_head, cfg.n_embd // cfg.n_head
    per_layer = {l: [] for l in range(cfg.n_layer)}
    for i in range(0, ids.shape[0], bs):
        x = ids[i:i + bs]
        B, Tt = x.shape
        emb = net.wte(x) + net.wpe(torch.arange(Tt))
        causal = torch.triu(torch.ones(Tt, Tt, dtype=torch.bool), 1)
        for l, block in enumerate(net.h):
            xin = block.ln1(emb)
            q, k, v = block.attn.c_attn(xin).split(cfg.n_embd, dim=2)
            q = q.view(B, Tt, H, D).transpose(1, 2)
            k = k.view(B, Tt, H, D).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) / (D ** 0.5)
            att = F.softmax(att.masked_fill(causal, float("-inf")), dim=-1)
            per_layer[l].append(float(att[:, :, pos, 0].mean()))
            y = (att @ v.view(B, Tt, H, D).transpose(1, 2))
            emb = emb + block.attn.c_proj(
                y.transpose(1, 2).contiguous().view(B, Tt, cfg.n_embd))
            emb = emb + block.mlp(block.ln2(emb))
    return {f"L{l}": float(np.mean(v)) for l, v in per_layer.items()}


# ---- head mean-replace (probe 3; e133 c_proj pre-hook conventions) ----

def head_mean_vec(net: TinyGPT, layer: int, head: int, bank_x, bs=30):
    """Mean of the head's 32-dim slice of c_proj's input over the CE_R bank
    (corpus windows only — no fact contexts). e133 organ_stats lineage."""
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
    rd = run_dir("e150_smoke" if SMOKE else "e150")
    log(f"E150 THE FLAT-CE ROUTE TEST (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-only, torch threads {torch.get_num_threads()}, "
        f"cuda available = {torch.cuda.is_available()}")

    # ---------------- protocol rebuild (e141 verbatim)
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

    # batteries: ctx = train_text[p-PRE-j : p] (e119/e141 construction)
    bat_ids = {}
    for j in GEOS:
        for tag, occ in (("install60", install_occ), ("held30", held_occ)):
            cs = [train_text[p - PRE - j: p] for p, _ in occ]
            bat_ids[(j, tag)] = torch.stack([corpus.encode(c) for c in cs])
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

    # ---------------- probe-4 windows: the fact AT position 0
    # segment = [NAME + 12 post-host chars] (col-0) or e133's
    # [12 pre-host + NAME + 12 post-host] (col-12 control); + corpus continuation
    fact_col0, fact_col12 = [], []
    for p, h in install_occ:
        post0 = p + len(h)
        seg0 = NAME + train_text[post0: post0 + FACT_POST]      # 7 + 12
        cont0 = train_ids[post0 + FACT_POST:
                          post0 + FACT_POST + (PRE - len(NAME) - FACT_POST)]
        w0 = torch.cat([corpus.encode(seg0), cont0])
        seg12 = (train_text[p - FACT_PRE: p] + NAME
                 + train_text[post0: post0 + FACT_POST])        # 12 + 7 + 12
        cont12 = train_ids[post0 + FACT_POST:
                           post0 + FACT_POST + (PRE - FACT_LEN)]
        w12 = torch.cat([corpus.encode(seg12), cont12])
        assert w0.shape[0] == PRE and w12.shape[0] == PRE
        assert corpus.decode(w0[:7]) == NAME
        assert corpus.decode(w12[FACT_PRE:FACT_PRE + 7]) == NAME
        fact_col0.append(w0)
        fact_col12.append(w12)
    fact_col0 = torch.stack(fact_col0)
    fact_col12 = torch.stack(fact_col12)
    log(f"probe-4 windows: col0 name at 0..6, col12 name at "
        f"{FACT_PRE}..{FACT_PRE + 6}; {fact_col0.shape[0]} windows each")

    @torch.no_grad()
    def fact_pos_reads(net: TinyGPT, win: torch.Tensor, name_col: int,
                       fwd=None, bs=30) -> dict:
        """Within-name continuation reads (e133's pname convention): mean
        p(true next name char) over the 6 reads inside the name, + onset
        p(Z) read when it exists (name_col >= 1)."""
        f = fwd if fwd is not None else net
        onset, within = [], []
        for i in range(0, win.shape[0], bs):
            lg, _ = f(win[i:i + bs])
            pr = F.softmax(lg, -1)
            for k in range(pr.shape[0]):
                if name_col >= 1:
                    onset.append(float(pr[k, name_col - 1, int(zid)]))
                for j in range(6):     # reads name_col..name_col+5 -> chars 1..6
                    r = name_col + j
                    within.append(float(pr[k, r, int(win[i + k, r + 1])]))
        on = torch.tensor(onset) if onset else None
        wt = torch.tensor(within)
        return {"onset_pz": (float(on.mean()) if on is not None else None),
                "within6_mean": float(wt.mean()),
                "within6_frac_ge_0.5": float((wt >= 0.5).float().mean()),
                "n_reads": int(wt.numel())}

    # ---------------- site battery pool (e133 pool-b construction verbatim)
    prompts = [train_text[p - PRE: p] for p, _ in held_occ][:N_PROMPTS]
    prompt_ids = torch.stack([corpus.encode(c) for c in prompts])
    gc = torch.Generator().manual_seed(CORP_CONT_SEED)
    src = torch.randint(len(train_ids) - (BLOCK - PRE) - 1,
                        (4 if not SMOKE else 1, len(prompts)), generator=gc)
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
    assert all(torch.equal(w[Z_XCOL:Z_XCOL + len(NAME)], name_ids)
               for w in pool_b)
    log(f"site battery: {tuple(pool_b.shape)}, ZEPHYRA at x-col {Z_XCOL} "
        f"(address row {SPLICE_ADDR_ROW})")

    @torch.no_grad()
    def site_reads(net: TinyGPT, fwd=None, bs=30) -> dict:
        """e133 battery_site logic: onset p(Z) at row 183 + mean p(true name
        char) over the 7 name positions."""
        f = fwd if fwd is not None else net
        onset, per_pos = [], [[] for _ in range(len(name_ids))]
        for i in range(0, pool_b.shape[0], bs):
            w = pool_b[i:i + bs]
            lg, _ = f(w)
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
    sel_heads = []      # [(layer, head, drop, ce_cost)]
    for t, v in g133["arms"].items():
        if not t.startswith("head_"):
            continue
        drop = b133_p - v["std_install60"]
        ce_cost = v["ce_r"] - b133_ce
        if drop >= 0.3 and ce_cost <= 0.35:
            p_ = t.split("_")[1]
            l_, h_ = int(p_[1:p_.find("h")]), int(p_[p_.find("h") + 1:])
            sel_heads.append((l_, h_, drop, ce_cost))
    top3 = sorted(((b133_p - v["std_install60"], v["ce_r"] - b133_ce, t)
                   for t, v in g133["arms"].items() if t.startswith("head_")),
                  reverse=True)[:3]
    top3_heads = []
    for drop, cec, t in top3:
        p_ = t.split("_")[1]
        top3_heads.append((int(p_[1:p_.find("h")]), int(p_[p_.find("h") + 1:]),
                           drop, cec))
    log(f"e133 head selection (drop>=0.3 & CE<=0.35): "
        + (", ".join(f"L{l}H{h} (drop {d:.4f}, CE {c:+.4f})"
                     for l, h, d, c in sel_heads) or "EMPTY"))
    log(f"e133 top-3 rider set: "
        + ", ".join(f"L{l}H{h} ({d:.4f}/{c:+.4f})"
                    for l, h, d, c in top3_heads))
    if not sel_heads:
        raise RuntimeError("e133 selection empty — dispatch premise broken")

    # ---------------- nets + gates
    log("--- PHASE 0: load + gate the five artifacts ---")
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
    log(f"G_CONS: p(Z) {bz_cons['mean_pz']:.10f} CE_R {ce_cons:.6f}: "
        f"{'PASS' if gates['consolidated']['pass'] else 'FAIL'}")
    if not gates["consolidated"]["pass"]:
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

    r_nets: dict = {}
    for tag, path, ref_g0, ref_g12 in (
            ("R150", R150_CK, G_R150_REF_PZ, G_R150_REF_G12),
            ("R300", R300_CK, G_R300_REF_PZ, None)):
        if tag not in R_NETS:
            continue
        net_r = load_cpu(path)
        pz_g0 = battery_fwd(net_r, bat_ids[(0, "install60")], zid)["mean_pz"]
        g = {"battery_pz_g0": pz_g0, "ref_g0": ref_g0,
             "pass": bool(abs(pz_g0 - ref_g0) < G_FALLBACK_TOL)}
        if ref_g12 is not None:
            pz_g12 = battery_fwd(net_r, bat_ids[(-12, "install60")],
                                 zid)["mean_pz"]
            g.update({"battery_pz_g12": pz_g12, "ref_g12": ref_g12})
            g["pass"] = bool(g["pass"] and abs(pz_g12 - ref_g12)
                             < G_FALLBACK_TOL)
        r_nets[tag] = net_r
        gates[tag] = g
        log(f"G_{tag}: {json.dumps({k: v for k, v in g.items() if k != 'pass'})} "
            f": {'PASS' if g['pass'] else 'FAIL'}")
        if not g["pass"]:
            raise RuntimeError(f"{tag} checkpoint failed its gate")

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

    # ---------------- mask-instrument validation (custom-causal == standard)
    log("--- mask instrument validation: custom-causal vs standard ---")
    mask_gate = {}
    with torch.no_grad():
        lg_std, _ = net_cons(bat_ids[(0, "install60")][:6])
        lg_cus, _ = forward_custom(net_cons, bat_ids[(0, "install60")][:6],
                                   block_key0=False)
        dmax = float((lg_std - lg_cus).abs().max())
    pz_cus = battery_fwd(net_cons, bat_ids[(0, "install60")], zid,
                         fwd=fwd_causal(net_cons))["mean_pz"]
    dpz = abs(pz_cus - bz_cons["mean_pz"])
    mask_gate = {"max_abs_logit_diff_batch6": dmax,
                 "abs_dpz_install60_g0": dpz,
                 "tol_logit": 1e-2, "tol_pz": 1e-3,
                 "pass": bool(dmax < 1e-2 and dpz < 1e-3)}
    log(f"MASK-GATE: max|dlogit| {dmax:.3e}, |dp(Z)| {dpz:.3e}: "
        f"{'PASS' if mask_gate['pass'] else 'FAIL'}")
    if not mask_gate["pass"]:
        raise RuntimeError("custom forward does not reproduce the standard one")

    sd_cons = {k: v.clone() for k, v in net_cons.state_dict().items()}
    wpe0_cons = sd_cons["wpe.weight"][0].clone()
    perm = torch.randperm(wpe0_cons.shape[0],
                          generator=torch.Generator().manual_seed(PERM_DIM_SEED))
    perm_row = wpe0_cons[perm]
    assert abs(float(perm_row.norm()) - float(wpe0_cons.norm())) < 1e-5
    log(f"wpe[0]: norm {float(wpe0_cons.norm()):.4f}; perm seed "
        f"{PERM_DIM_SEED} (= e141's arm; norm preserved)")

    # ---------------- the fact-vs-CE plane (every intervention lands here)
    PLANE: list[dict] = []

    def cell(probe, tag, net_tag, geo, expr_base, expr_arm, ce_base, ce_arm,
             rider=False, extra=None):
        ret = expr_arm / max(expr_base, 1e-12)
        c = {"probe": probe, "tag": tag, "net": net_tag, "geo": geo,
             "expr_base": expr_base, "expr_arm": expr_arm,
             "retention": ret, "fact_drop_pct": 100.0 * (1.0 - ret),
             "ce_base": ce_base, "ce_arm": ce_arm, "ce_cost": ce_arm - ce_base,
             "kills60": bool(ret <= KILL_RET), "dies50": bool(ret <= DIE_RET),
             "in_plane": bool(not rider)}
        if extra:
            c.update(extra)
        PLANE.append(c)
        log(f"  [{probe} | {tag:26s}] expr {expr_base:.4f} -> {expr_arm:.4f} "
            f"(x{ret:.3f}, drop {100 * (1 - ret):5.1f}%) | CE {ce_base:.4f} "
            f"-> {ce_arm:.4f} (cost {ce_arm - ce_base:+.4f})"
            + ("  [rider]" if rider else ""))
        return c

    # reusable eval twin (no RNG in eval)
    ev = evl_load(sd_cons)

    def cons_eval(sd, label, geos=(0,), held=False, extra_bats=None,
                  fwd=None, hooks=None):
        """Load a state into the eval twin and run the requested batteries."""
        if sd is not None:
            ev.load_state_dict(sd)
        out: dict = {}
        f = fwd
        if hooks is not None:
            hooks.__enter__()
        try:
            for j in geos:
                out[f"g{j}_install60"] = battery_fwd(
                    ev, bat_ids[(j, "install60")], zid, fwd=f)
            if held:
                out["g0_held30"] = battery_fwd(
                    ev, bat_ids[(0, "held30")], zid, fwd=f)
            if extra_bats:
                for k, (bids, fn) in extra_bats.items():
                    out[k] = fn(ev, bids, fwd=f)
            out["ce_r"] = ce_fwd(ev, *r_eval_xy, fwd=f)
        finally:
            if hooks is not None:
                hooks.__exit__(None, None, None)
        log(f"  [{label:26s}] " + " ".join(
            f"{k} {(v['mean_pz'] if isinstance(v, dict) and 'mean_pz' in v else v):.4f}"
            for k, v in out.items()
            if isinstance(v, (dict, float)) and not isinstance(v, list)))
        return out

    # =====================================================================
    # PROBE 1 — PERM AT BOTH GEOMETRIES (the missing g-12 cell)
    # =====================================================================
    log("--- PROBE 1: direction-perm of wpe[0] at g-12 AND g0 ---")
    probe1: dict = {"cells": {}}

    # consolidated: none + perm, both geometries
    base_cons = cons_eval(None, "cons none (g0+g-12)", geos=GEOS, held=True)
    sd_perm, gate_perm = modified_wpe(sd_cons, 0, perm_row)
    assert gate_perm["pass"], gate_perm
    perm_cons = cons_eval(sd_perm, "cons perm@g0+g-12", geos=GEOS, held=True)
    probe1["cells"]["cons__perm__g0"] = cell(
        "P1", "cons/perm@g0", "consolidated", 0,
        base_cons["g0_install60"]["mean_pz"],
        perm_cons["g0_install60"]["mean_pz"], base_cons["ce_r"],
        perm_cons["ce_r"],
        extra={"gate": gate_perm, "held30_base": base_cons["g0_held30"]["mean_pz"],
               "held30_arm": perm_cons["g0_held30"]["mean_pz"]})
    probe1["cells"]["cons__perm__gm12"] = cell(
        "P1", "cons/perm@g-12", "consolidated", -12,
        base_cons["g-12_install60"]["mean_pz"],
        perm_cons["g-12_install60"]["mean_pz"], base_cons["ce_r"],
        perm_cons["ce_r"], extra={"gate": gate_perm})

    # R nets: none + perm at g-12 (the registered novel-geometry cells)
    probe1["r_nets"] = {}
    for tag in R_NETS:
        net_r = r_nets[tag]
        sd_r = {k: v.clone() for k, v in net_r.state_dict().items()}
        wpe0_r = sd_r["wpe.weight"][0].clone()
        perm_r = wpe0_r[perm]
        net_r.load_state_dict(sd_r)
        n_g12 = battery_fwd(net_r, bat_ids[(-12, "install60")], zid)
        n_ce = ce_fwd(net_r, *r_eval_xy)
        sd_rp, gate_rp = modified_wpe(sd_r, 0, perm_r)
        assert gate_rp["pass"], gate_rp
        net_r.load_state_dict(sd_rp)
        p_g12 = battery_fwd(net_r, bat_ids[(-12, "install60")], zid)
        p_ce = ce_fwd(net_r, *r_eval_xy)
        c = cell("P1", f"{tag}/perm@g-12", tag, -12,
                 n_g12["mean_pz"], p_g12["mean_pz"], n_ce, p_ce,
                 extra={"gate": gate_rp})
        probe1["cells"][f"{tag}__perm__gm12"] = c
        probe1["r_nets"][tag] = {"none_g12": n_g12, "perm_g12": p_g12,
                                 "none_ce": n_ce, "perm_ce": p_ce,
                                 "spares_80": bool(
                                     c["retention"] >= SPARE_BAR)}
        net_r.load_state_dict(sd_r)     # restore

    presence_at_novel = all(
        probe1["cells"][f"{t}__perm__gm12"]["retention"] >= SPARE_BAR
        for t in R_NETS)
    rets = {t: probe1["cells"][f"{t}__perm__gm12"]["retention"]
            for t in R_NETS}
    probe1["verdict"] = {
        "PRESENCE_AT_NOVEL": bool(presence_at_novel),
        "bar": f"perm@g-12 spares >= {SPARE_BAR:.0%} of g-12 level on BOTH "
               f"R nets {tuple(R_NETS)}",
        "retentions": rets,
        "supporting_cons_g12": probe1["cells"]["cons__perm__gm12"]["retention"],
        "verdict": ("PRESENCE-AT-NOVEL (presence-only holds at novel geometry)"
                    if presence_at_novel else
                    "PRESENCE-AT-NOVEL FAILS the 80% bar (retentions "
                    + ", ".join(f"{t} {v:.3f}" for t, v in rets.items())
                    + f", cons {probe1['cells']['cons__perm__gm12']['retention']:.3f}"
                    + " — perm@g-12 costs the fact a consistent 16-21%: NOT a "
                    "kill, but the fact MILDLY consults row-0's direction at "
                    "novel geometry, unlike the +4% spare at trained g0)")}

    # =====================================================================
    # PROBE 2 — FORCED-OFF-SINK (attention mask blocking keys at position 0)
    # =====================================================================
    log("--- PROBE 2: forced-off-sink attention mask (all layers/heads) ---")
    # manipulation check: the pre-mask sink mass the mask removes
    sm_cons = sink_mass_at_read(net_cons, bat_ids[(0, "install60")], 129)
    sm_r150 = sink_mass_at_read(r_nets["R150"], bat_ids[(-12, "install60")],
                                117)
    log(f"pre-mask sink mass @read (cons g0 / R150 g-12): "
        f"{ {k: round(v, 3) for k, v in sm_cons.items()} } / "
        f"{ {k: round(v, 3) for k, v in sm_r150.items()} }")
    probe2: dict = {"cells": {}, "sink_mass_premask": {
        "cons_g0_read129": sm_cons, "R150_gm12_read117": sm_r150,
        "note": "mean attention mass on key 0 at the read position, per "
                "layer (mean over heads/batch); post-mask this is exactly "
                "0 by construction (the -inf column)"}}

    def mask_cell(net, net_tag, geo, bats, base_lbl):
        f0 = fwd_causal(net)
        f1 = fwd_offsink(net)
        b_outs, m_outs = {}, {}
        for key, bids in bats:
            b_outs[key] = battery_fwd(net, bids, zid, fwd=f0)
            m_outs[key] = battery_fwd(net, bids, zid, fwd=f1)
        b_ce = ce_fwd(net, *r_eval_xy, fwd=f0)
        m_ce = ce_fwd(net, *r_eval_xy, fwd=f1)
        log(f"  [{base_lbl:26s}] causal-custom CE {b_ce:.4f} -> masked "
            f"{m_ce:.4f}")
        return b_outs, m_outs, b_ce, m_ce

    # consolidated @g0 + @g-12 (PLANE cells)
    b2, m2, bce2, mce2 = mask_cell(
        net_cons, "consolidated", 0,
        [(f"g{j}", bat_ids[(j, "install60")]) for j in GEOS]
        + [("held", bat_ids[(0, "held30")])], "cons mask base")
    for j in GEOS:
        probe2["cells"][f"cons__mask__g{j}"] = cell(
            "P2", f"cons/mask@g{j:+d}", "consolidated", j,
            b2[f"g{j}"]["mean_pz"], m2[f"g{j}"]["mean_pz"], bce2, mce2,
            extra={"held30_base": b2["held"]["mean_pz"],
                   "held30_arm": m2["held"]["mean_pz"],
                   "note": "base = causal-only CUSTOM forward (path-matched)"})
    probe2["cons_detail"] = {"base": b2, "masked": m2,
                             "ce_base": bce2, "ce_masked": mce2}

    # R nets @g-12 (PLANE cells)
    for tag in R_NETS:
        net_r = r_nets[tag]
        b2r, m2r, bce2r, mce2r = mask_cell(
            net_r, tag, -12, [("g-12", bat_ids[(-12, "install60")])],
            f"{tag} mask base")
        probe2["cells"][f"{tag}__mask__gm12"] = cell(
            "P2", f"{tag}/mask@g-12", tag, -12,
            b2r["g-12"]["mean_pz"], m2r["g-12"]["mean_pz"], bce2r, mce2r)
        probe2.setdefault("r_detail", {})[tag] = {
            "base": b2r, "masked": m2r, "ce_base": bce2r, "ce_masked": mce2r}

    # CONTROLS (flagged out of the bars):
    # (a) install net @g0 — W013's differential signature
    bi, mi, bcei, mcei = mask_cell(
        net_inst, "install", 0, [("g0", bat_ids[(0, "install60")])],
        "install mask base")
    d_cons = 1.0 - probe2["cells"]["cons__mask__g0"]["retention"]
    d_inst = 1.0 - mi["g0"]["mean_pz"] / max(bi["g0"]["mean_pz"], 1e-12)
    probe2["cells"]["install__mask__g0__CONTROL"] = cell(
        "P2", "inst/mask@g0 [ctl]", "install", 0,
        bi["g0"]["mean_pz"], mi["g0"]["mean_pz"], bcei, mcei, rider=True,
        extra={"role": "W013 differential control (not bar-eligible)"})
    probe2["w013_differential"] = {
        "fact_damage_cons": d_cons, "fact_damage_install": d_inst,
        "differential": d_cons - d_inst,
        "signature": bool(d_cons > d_inst),
        "note": "W013's registered savor: consolidated >> install damaged "
                "under the same mask is the policy signature"}

    # (b) site-stored net @site — specificity control
    bs_, ms_, bces_, mces_ = None, None, None, None
    f0b, f1b = fwd_causal(net_armb), fwd_offsink(net_armb)
    bs_ = site_reads(net_armb, fwd=f0b)
    ms_ = site_reads(net_armb, fwd=f1b)
    bces_ = ce_fwd(net_armb, *r_eval_xy, fwd=f0b)
    mces_ = ce_fwd(net_armb, *r_eval_xy, fwd=f1b)
    probe2["cells"]["armb__mask__site__CONTROL"] = cell(
        "P2", "armB/mask@site [ctl]", "site_stored", 183,
        bs_["onset_pz"], ms_["onset_pz"], bces_, mces_, rider=True,
        extra={"role": "specificity control: site-stored fact should NOT "
                       "die under the mask (its read is not row-0-routed)",
               "base_detail": bs_, "arm_detail": ms_})
    probe2["specificity"] = {
        "site_stored_retention_under_mask":
            probe2["cells"]["armb__mask__site__CONTROL"]["retention"],
        "site_stored_survives":
            bool(probe2["cells"]["armb__mask__site__CONTROL"]["retention"]
                 >= SPARE_BAR)}

    # =====================================================================
    # PROBE 3 — FACT-SPECIFIC HEAD ABLATION (the flat-CE fact-kill candidate)
    # =====================================================================
    log("--- PROBE 3: fact-specific head ablation (mean-replace + zero) ---")
    probe3: dict = {"selected": [{"head": f"L{l}H{h}", "layer": l, "head": h,
                                  "e133_drop": d, "e133_ce_cost": c}
                                 for l, h, d, c in sel_heads],
                    "cells": {}}

    sel = sel_heads[0]                       # the singleton (asserted above)
    l_sel, h_sel = sel[0], sel[1]
    mv = head_mean_vec(net_cons, l_sel, h_sel, r_eval_x)

    # mean-replace @g0 (dispatch primary) + @g-12
    for j in GEOS:
        with HeadReplace(net_cons, {(l_sel, h_sel): mv}) as _:
            o = battery_fwd(net_cons, bat_ids[(j, "install60")], zid)
            oce = ce_fwd(net_cons, *r_eval_xy)
        bj = base_cons[f"g{j}_install60"]["mean_pz"]
        probe3["cells"][f"L{l_sel}H{h_sel}__mean__g{j}"] = cell(
            "P3", f"L{l_sel}H{h_sel}-mean@g{j:+d}", "consolidated", j,
            bj, o["mean_pz"], base_cons["ce_r"], oce,
            extra={"mode": "mean-replace (CE_R-bank mean)"})

    # zero arm @g0 (e133 comparability)
    with HeadReplace(net_cons, {(l_sel, h_sel): None}) as _:
        o = battery_fwd(net_cons, bat_ids[(0, "install60")], zid)
        oce = ce_fwd(net_cons, *r_eval_xy)
    probe3["cells"][f"L{l_sel}H{h_sel}__zero__g0"] = cell(
        "P3", f"L{l_sel}H{h_sel}-zero@g0", "consolidated", 0,
        base_cons["g0_install60"]["mean_pz"], o["mean_pz"],
        base_cons["ce_r"], oce,
        extra={"mode": "zero (e133 convention)",
               "e133_reference": {"drop": sel[2], "ce_cost": sel[3]}})

    # REPORT-ONLY rider: top-3 joint mean-replace @g0
    mv3 = {(l, h): head_mean_vec(net_cons, l, h, r_eval_x)
           for l, h, _, _ in top3_heads}
    with HeadReplace(net_cons, mv3) as _:
        o = battery_fwd(net_cons, bat_ids[(0, "install60")], zid)
        oce = ce_fwd(net_cons, *r_eval_xy)
    probe3["cells"]["top3__mean__g0__RIDER"] = cell(
        "P3", "top3-mean@g0 [rider]", "consolidated", 0,
        base_cons["g0_install60"]["mean_pz"], o["mean_pz"],
        base_cons["ce_r"], oce, rider=True,
        extra={"role": "REPORT-ONLY post-hoc rider (selection set is a "
                       "singleton; this restores a 'joint' cell)",
               "heads": [f"L{l}H{h}" for l, h, _, _ in top3_heads]})
    probe3["singleton_note"] = (
        f"the dispatch criterion (drop>=0.3 & CE<=0.35) selects exactly one "
        f"head on e133's stored table: L{l_sel}H{h_sel} (drop {sel[2]:.4f} at "
        f"CE {sel[3]:+.4f}); 'singly and jointly' collapses to the singleton; "
        f"the top-3 joint is a labeled rider")

    # =====================================================================
    # PROBE 4 — FACT-AT-POSITION-0 (W014's never-consults control)
    # =====================================================================
    log("--- PROBE 4: fact-at-position-0 scramble ---")
    probe4: dict = {"cells": {}}
    ev.load_state_dict(sd_cons)
    for col_tag, win, name_col in (("col0", fact_col0, 0),
                                   ("col12", fact_col12, FACT_PRE)):
        base_r = fact_pos_reads(ev, win, name_col)
        ev.load_state_dict(sd_perm)
        perm_r = fact_pos_reads(ev, win, name_col)
        ev.load_state_dict(sd_cons)
        probe4["cells"][f"{col_tag}__none"] = base_r
        probe4["cells"][f"{col_tag}__perm"] = perm_r
        probe4["cells"][f"{col_tag}__perm__cell"] = cell(
            "P4", f"perm@fact-{col_tag}", "consolidated", 0,
            base_r["within6_mean"], perm_r["within6_mean"],
            base_cons["ce_r"], perm_cons["ce_r"],
            extra={"metric": "within6_mean (6 within-name continuation "
                             "reads; col-0 has no onset read by construction)",
                   "onset_base": base_r["onset_pz"],
                   "onset_perm": perm_r["onset_pz"],
                   "ce_note": "CE columns are the net-level perm state from "
                              "probe 1 (same surgery)"})
    c0 = probe4["cells"]["col0__perm__cell"]
    c12 = probe4["cells"]["col12__perm__cell"]
    direction_consulted = bool(c0["retention"] <= DIE_RET)
    probe4["verdict"] = {
        "DIRECTION_CONSULTED": direction_consulted,
        "bar": "col-0 scramble retention <= 0.5 x col-0 base (die bar); "
               "60% convention also reported",
        "col0_retention": c0["retention"],
        "col0_kills60": c0["kills60"],
        "col12_retention": c12["retention"],
        "verdict": ("DIRECTION-CONSULTED (the route consults row-0's "
                    "direction when the fact lives there — W014's tenant "
                    "reading dies)" if direction_consulted else
                    "NOT-CONSULTED (col-0 scramble spares the fact — W014's "
                    "tenant reading survives its control)"),
        "texture_note_col12_control": (
            f"the col-12 CONTROL cell (same windows, name at cols 12..18, "
            f"reads 12..17) {'DIES' if c12['retention'] <= DIE_RET else 'is spared'} "
            f"under the same perm (x{c12['retention']:.3f}) while the "
            f"standard 130-token battery's read at position 129 is spared "
            f"(probe 1: x{probe1['cells']['cons__perm__g0']['retention']:.3f}) "
            f"— the route's direction-insensitivity is READ-HORIZON-"
            f"dependent: short-horizon within-name reads (key 0 is 1-of-~15 "
            f"keys) consult row-0's direction; the long-context onset read "
            f"(1-of-130) does not. Texture, not a registered bar.")}

    # =====================================================================
    # PROBE 5 — NORM LADDER (the threshold in (0.066, 0.382))
    # =====================================================================
    log("--- PROBE 5: norm ladder on wpe[0] (direction kept) ---")
    probe5: dict = {"ladder": [], "rider_r150": [], "cells": {}}
    n0 = float(wpe0_cons.norm())
    targets = ([0.07, 0.35] if SMOKE else list(LADDER)) + [ANCHOR_ZERO]
    for tgt in sorted(targets):
        if tgt == 0.0:
            row = torch.zeros_like(wpe0_cons)
        else:
            row = wpe0_cons * (tgt / n0)
        sd_t, gate_t = modified_wpe(sd_cons, 0, row)
        assert gate_t["pass"], gate_t
        o = cons_eval(sd_t, f"ladder norm={tgt:.2f}", geos=GEOS, held=(tgt > 0))
        ent = {"target_norm": tgt, "actual_norm": float(row.norm()),
               "gate": gate_t,
               "g0_install60": o["g0_install60"]["mean_pz"],
               "gm12_install60": o["g-12_install60"]["mean_pz"],
               "ce_r": o["ce_r"]}
        if tgt > 0:
            ent["g0_held30"] = o["g0_held30"]["mean_pz"]
        probe5["ladder"].append(ent)
        for j, key in ((0, "g0_install60"), (-12, "gm12_install60")):
            probe5["cells"][f"norm{tgt:g}__g{j}"] = cell(
                "P5", f"norm={tgt:g}@g{j:+d}", "consolidated", j,
                base_cons[f"g{j}_install60"]["mean_pz"], ent[key],
                base_cons["ce_r"], ent["ce_r"], extra={"target_norm": tgt})
    # rider: R150 @g-12 ladder
    net_r = r_nets["R150"]
    sd_r = {k: v.clone() for k, v in net_r.state_dict().items()}
    wpe0_r = sd_r["wpe.weight"][0].clone()
    n0r = float(wpe0_r.norm())
    base_r150_g12 = battery_fwd(net_r, bat_ids[(-12, "install60")], zid)
    base_r150_ce = ce_fwd(net_r, *r_eval_xy)
    for tgt in ([0.07, 0.35] if SMOKE else LADDER):
        row = wpe0_r * (tgt / n0r)
        sd_t, gate_t = modified_wpe(sd_r, 0, row)
        net_r.load_state_dict(sd_t)
        o = battery_fwd(net_r, bat_ids[(-12, "install60")], zid)
        oce = ce_fwd(net_r, *r_eval_xy)
        probe5["rider_r150"].append(
            {"target_norm": tgt, "actual_norm": float(row.norm()),
             "gm12_install60": o["mean_pz"], "ce_r": oce})
        cell("P5", f"R150 norm={tgt:g}@g-12 [rider]", "R150", -12,
             base_r150_g12["mean_pz"], o["mean_pz"], base_r150_ce, oce,
             rider=True, extra={"target_norm": tgt})
    net_r.load_state_dict(sd_r)

    # ladder threshold bracket (consolidated @g0, registered reading)
    surv = [e["target_norm"] for e in probe5["ladder"]
            if e["target_norm"] > 0
            and e["g0_install60"] >= SPARE_BAR * base_cons["g0_install60"]["mean_pz"]]
    dead = [e["target_norm"] for e in probe5["ladder"]
            if e["g0_install60"] <= DIE_RET * base_cons["g0_install60"]["mean_pz"]]
    probe5["threshold_bracket_g0"] = {
        "smallest_surviving_norm": min(surv) if surv else None,
        "largest_dying_norm": max(dead) if dead else None,
        "bracket": f"({max(dead) if dead else 0.0}, "
                   f"{min(surv) if surv else float('inf')})",
        "e141_anchors": {"mean_replace_0.066": "killed (expr 0.0008)",
                         "halfnorm_0.382": "survived (expr 0.782)"}}

    # =====================================================================
    # adjudication (registered — no bar shopping)
    # =====================================================================
    plane = [c for c in PLANE if c["in_plane"]]
    kills = [c for c in plane if c["kills60"]]
    flatce = [c for c in kills if c["ce_cost"] <= CE_FLAT]
    wreck = [c for c in kills if c["ce_cost"] >= CE_WRECK]
    gap = [c for c in kills if CE_FLAT < c["ce_cost"] < CE_WRECK]
    near = [c for c in plane if c["dies50"] and not c["kills60"]
            and c["ce_cost"] <= CE_FLAT]
    all_kills_wreck = bool(len(kills) > 0 and len(kills) == len(wreck))
    flat_ce_route = bool(len(flatce) > 0)

    near_txt = (" | near-miss: " + "; ".join(
        f"{c['tag']} drop {c['fact_drop_pct']:.1f}% at CE {c['ce_cost']:+.2f}"
        for c in near)) if near else ""
    if flat_ce_route:
        overall = ("FLAT-CE-ROUTE (the route is real and isolable: "
                   + "; ".join(f"{c['tag']} drop {c['fact_drop_pct']:.0f}% "
                               f"at CE {c['ce_cost']:+.2f}" for c in flatce)
                   + ")")
    elif all_kills_wreck:
        overall = ("ALL-KILLS-WRECK (every fact-kill costs CE >= +0.70 — "
                   "'routed' is hard-bounded to a wreck statement; T081/"
                   "T082's frame dies)" + near_txt)
    elif len(kills) == 0:
        overall = ("NO-KILLS (no plane cell dropped the fact >= 60% — "
                   "unexpected; texture with numbers)" + near_txt)
    else:
        overall = ("AMBIGUOUS-GAP (fact-kills exist at CE strictly between "
                   f"+{CE_FLAT} and +{CE_WRECK}: "
                   + "; ".join(f"{c['tag']} drop {c['fact_drop_pct']:.0f}% "
                               f"at CE {c['ce_cost']:+.2f}" for c in gap)
                   + ") — neither registered extreme; texture with numbers")

    adjudication = {
        "bars_verbatim": REGISTERED_PREDICTION["queue_row_verbatim"],
        "n_plane_cells": len(plane),
        "n_killing_cells": len(kills),
        "FLAT_CE_ROUTE": {
            "fires": flat_ce_route,
            "cells": [{"tag": c["tag"], "drop_pct": c["fact_drop_pct"],
                       "ce_cost": c["ce_cost"]} for c in flatce]},
        "ALL_KILLS_WRECK": {
            "fires": all_kills_wreck,
            "kill_cells": [{"tag": c["tag"], "drop_pct": c["fact_drop_pct"],
                            "ce_cost": c["ce_cost"]} for c in kills]},
        "near_misses_report_only": {
            "definition": "die-bar cells (drop >= 50%) at CE <= +0.35 that "
                          "miss the registered 60% kill bar — margins "
                          "reported, never gated (no bar shopping)",
            "cells": [{"tag": c["tag"], "drop_pct": c["fact_drop_pct"],
                       "ce_cost": c["ce_cost"]} for c in near]},
        "DIRECTION_CONSULTED": {"fires": direction_consulted,
                                "col0_retention": c0["retention"]},
        "PRESENCE_AT_NOVEL": {"fires": bool(presence_at_novel),
                              "retentions": probe1["verdict"]["retentions"]},
        "verdict": overall,
        "headline": (
            f"kills {len(kills)}/{len(plane)} plane cells; flat-CE kills "
            f"{len(flatce)}; wreck-priced kills {len(wreck)}; gap kills "
            f"{len(gap)} | mask cells: "
            + ", ".join(f"{c['tag']} x{c['retention']:.2f}@CE{c['ce_cost']:+.2f}"
                        for c in PLANE if c["probe"] == "P2" and c["in_plane"])
            + f" | head L{l_sel}H{h_sel}-mean: "
            f"x{probe3['cells'][f'L{l_sel}H{h_sel}__mean__g0']['retention']:.2f}"
            f"@CE{probe3['cells'][f'L{l_sel}H{h_sel}__mean__g0']['ce_cost']:+.2f}"
            + f" | ladder threshold (g0): "
              f"{probe5['threshold_bracket_g0']['bracket']}"),
    }
    log("=" * 78)
    log(f"E150 VERDICT: {overall}")
    log(f"  FLAT-CE-ROUTE {flat_ce_route} | ALL-KILLS-WRECK {all_kills_wreck} "
        f"| DIRECTION-CONSULTED {direction_consulted} "
        f"| PRESENCE-AT-NOVEL {presence_at_novel}")
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e150_flatce_route",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": ("R45 critic attack 2 (the flat-CE ultimatum) + "
                         "amendments T081 (probe power) / T082 (hard bound) / "
                         "W014 (never-consults null). Docstring + bars "
                         "written before compute."),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("is 'the consolidated fact is ROUTED through row-0 "
                     "presence' a memory property (a flat-CE fact-kill "
                     "exists) or a wreck artifact (every kill wrecks the "
                     "LM)?"),
        "nets": {
            "consolidated": f"runs/checkpoints/{CONS_CK.name} (loaded, gated)",
            "R150": f"runs/checkpoints/{R150_CK.name} (loaded, gated)",
            "R300": f"runs/checkpoints/{R300_CK.name} (loaded, gated)",
            "install_control": f"runs/checkpoints/{INST_CK.name} (loaded, gated)",
            "site_stored_control": f"runs/checkpoints/{ARMB_CK.name} (loaded, gated)",
            "eval_only": True,
        },
        "gates": {"G_SPLICE": G_SPLICE, "consolidated": gates["consolidated"],
                  "install": gates["install"],
                  "r_arms": {t: gates[t] for t in R_NETS},
                  "site_stored": gates["site_stored"],
                  "mask_instrument": mask_gate},
        "probe1_perm_both_geometries": probe1,
        "probe2_forced_off_sink": probe2,
        "probe3_head_ablation": probe3,
        "probe4_fact_at_position0": probe4,
        "probe5_norm_ladder": probe5,
        "plane": PLANE,
        "adjudication": adjudication,
        "honesty_reflex": {
            "mask_off_distribution": "the forced-off-sink mask puts every "
                "context OFF its training distribution (the sink was present "
                "at train time in every window); fact loss under the mask "
                "conflates route-interruption with generic off-distribution "
                "damage — the CE column prices the generic part, and the "
                "site-stored control (a fact that does NOT route) is the "
                "discriminator: if arm_b's fact survives the mask at "
                "similar CE, the damage is route-specific; if BOTH die, the "
                "mask is just wreckage and reads as ALL-KILLS-WRECK.",
            "mask_implementation_risk": "the mask keeps row 0's self-"
                "attention (a fully-masked softmax row is undefined) and "
                "blocks key 0 only for queries >= 1 — every read the "
                "batteries score (last-position / onset-row reads) is such "
                "a query; the custom forward was gated against the standard "
                "forward (max |dp(Z)| recorded) and every mask cell's base "
                "is the path-matched causal-only custom forward.",
            "single_net_caveats": "probe 3/4/5 and the consolidated cells of "
                "probes 1/2 run on ONE consolidated net (e113 recipe, one "
                "seed); the R nets add two more lineages at g-12 only; "
                "verdicts are line-specific until e145's replication seeds "
                "land.",
            "singleton_head_set": "the dispatch's fact-specific head class "
                "is a singleton (L0H3) on e133's stored table — the "
                "'jointly' cell degenerates by the registered criterion; "
                "the top-3 joint rider is post-hoc and never gates a bar.",
            "probe4_base_power": "the col-0 cell reads only 6 within-name "
                "continuation reads with NO preceding context — if the "
                "col-0 base expression is near floor, DIRECTION-CONSULTED "
                "has no power and must be read as uninformative, not as "
                "evidence for the tenant reading; the base level is "
                "reported in probe4.cells.col0__none.",
            "ce_bank_is_position_blind": "CE_R windows are 256-token corpus "
                "windows that never contain the fact; the CE column prices "
                "general LM damage, not fact-geometry damage — a cell could "
                "in principle be flat on CE_R while disrupting the fact's "
                "read geometry specifically (that is the dissociation being "
                "sought, and also its confound).",
        },
        "trims": trims,
        "recipe_deviations": recipe_deviations,
        "ckpt_inventory": {
            "saved": {},
            "external_used": [f"runs/checkpoints/{p.name}"
                              for p in (CONS_CK, R150_CK, R300_CK, INST_CK,
                                        ARMB_CK)],
            "note": "eval-only: no checkpoints written or regenerated",
        },
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": 2739072,
                   "device": "cpu", "torch_threads": torch.get_num_threads(),
                   "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "flatce_route.png", PLANE, probe1, probe2, probe3, probe4,
         probe5, adjudication, base_cons)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'flatce_route.png'}")
    log(f"total {time.time() - T0:.1f}s")


# ------------------------------------------------------------------ plot

def plot(path, plane, probe1, probe2, probe3, probe4, probe5, adjudication,
         base_cons):
    """The killer visualization: the fact-vs-CE plane with every
    intervention plotted, the flat-CE region shaded."""
    fig = plt.figure(figsize=(16.0, 12.5))
    gs = fig.add_gridspec(3, 2, height_ratios=(1.25, 1.0, 1.0))

    # ---- main panel: fact retention vs CE cost, every intervention
    ax = fig.add_subplot(gs[0, :])
    cols = {"P1": "tab:purple", "P2": "tab:red", "P3": "tab:green",
            "P4": "tab:orange", "P5": "tab:blue"}
    xmax = max(3.2, max((c["ce_cost"] for c in plane), default=2.0) + 0.3)
    # shaded regions
    ax.axhspan(0, 100 * KILL_RET, xmin=0, xmax=CE_FLAT / xmax,
               color="seagreen", alpha=0.16)
    ax.axhspan(0, 100 * KILL_RET, xmin=CE_WRECK / xmax, xmax=1,
               color="firebrick", alpha=0.12)
    ax.axvspan(CE_FLAT, CE_WRECK, color="gold", alpha=0.10)
    ax.axhline(100 * KILL_RET, ls="--", color="gray", lw=1.0)
    ax.axhline(100 * SPARE_BAR, ls=":", color="seagreen", lw=1.0)
    ax.axvline(CE_FLAT, ls="--", color="seagreen", lw=1.2)
    ax.axvline(CE_WRECK, ls="--", color="firebrick", lw=1.2)
    ax.text(CE_FLAT / 2 - 0.02, 4, "FLAT-CE-ROUTE region\n(kill >=60% at "
            "CE<=+0.35)", ha="center", fontsize=8, color="seagreen")
    ax.text((CE_WRECK + xmax) / 2, 4, "WRECK-priced kills\n(CE>=+0.70)",
            ha="center", fontsize=8, color="firebrick")
    ax.text((CE_FLAT + CE_WRECK) / 2, 97, "CE gap (+0.35,+0.70)", ha="center",
            fontsize=7.5, color="darkgoldenrod")
    seen = set()
    for c in plane:
        lab = f"probe {c['probe']}" + (" (rider/ctl)" if not c["in_plane"]
                                       else "")
        ax.scatter(c["ce_cost"], 100 * c["retention"], s=64,
                   color=cols[c["probe"]], edgecolor="k", lw=0.6, zorder=3,
                   marker=("o" if c["in_plane"] else "s"),
                   label=lab if lab not in seen else None)
        seen.add(lab)
        ax.annotate(c["tag"], (c["ce_cost"], 100 * c["retention"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=6.4)
    ax.set_xlabel("CE cost vs same-net baseline (nats, e065 bank seed 26502)")
    ax.set_ylabel("fact retention (expr_arm / expr_base, %)")
    ax.set_xlim(-0.08, xmax)
    ax.set_ylim(-4, 112)
    a = adjudication
    ax.set_title("E150 — THE FLAT-CE ROUTE TEST: every intervention in the "
                 "fact-vs-CE plane\n"
                 f"kills {a['n_killing_cells']}/{a['n_plane_cells']} plane "
                 f"cells | FLAT-CE-ROUTE {a['FLAT_CE_ROUTE']['fires']} | "
                 f"ALL-KILLS-WRECK {a['ALL_KILLS_WRECK']['fires']} | "
                 f"DIR-CONSULTED {a['DIRECTION_CONSULTED']['fires']} | "
                 f"PRESENCE-AT-NOVEL {a['PRESENCE_AT_NOVEL']['fires']}",
                 fontsize=10)
    ax.legend(fontsize=7.5, loc="center right")
    ax.grid(alpha=0.25)

    # ---- ladder
    ax = fig.add_subplot(gs[1, 0])
    lad = [e for e in probe5["ladder"]]
    ns = [e["actual_norm"] for e in lad]
    ax.plot(ns, [e["g0_install60"] for e in lad], "o-", color="crimson",
            label="expr @g0")
    ax.plot(ns, [e["gm12_install60"] for e in lad], "s-", color="darkorange",
            label="expr @g-12")
    b0 = base_cons["g0_install60"]["mean_pz"]
    b12 = base_cons["g-12_install60"]["mean_pz"]
    ax.axhline(b0, ls=":", color="crimson", lw=0.9, label=f"base g0 {b0:.3f}")
    ax.axhline(b0 * SPARE_BAR, ls="--", color="seagreen", lw=0.9,
               label="survive bar (0.8x)")
    ax.axhline(b0 * DIE_RET, ls="--", color="gray", lw=0.9,
               label="die bar (0.5x)")
    ax.axvline(0.066, ls="-.", color="dimgray", lw=0.9)
    ax.axvline(0.382, ls="-.", color="dimgray", lw=0.9)
    ax.text(0.072, 0.05, "e141 mean-replace 0.066 (killed)", fontsize=6.5,
            rotation=90, va="bottom")
    ax.text(0.352, 0.05, "e141 halfnorm 0.382 (survived)", fontsize=6.5,
            rotation=90, va="bottom")
    ax2 = ax.twinx()
    ax2.plot(ns, [e["ce_r"] for e in lad], "D--", color="steelblue", ms=5,
             alpha=0.85, label="CE_R (right)")
    ax2.set_ylabel("CE_R", color="steelblue")
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax.set_xlabel("|wpe[0]| (direction kept)")
    ax.set_ylabel("expression p(Z)")
    ax.set_title(f"PROBE 5: norm ladder — threshold bracket "
                 f"{probe5['threshold_bracket_g0']['bracket']}", fontsize=9.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.4, loc="center right")

    # ---- perm at both geometries
    ax = fig.add_subplot(gs[1, 1])
    rows = []
    for t in probe1["r_nets"]:
        rows.append((f"{t}@g-12", probe1["r_nets"][t]["none_g12"]["mean_pz"],
                     probe1["r_nets"][t]["perm_g12"]["mean_pz"]))
    rows.append(("cons@g-12", base_cons["g-12_install60"]["mean_pz"],
                 probe1["cells"]["cons__perm__gm12"]["expr_arm"]))
    rows.append(("cons@g0", base_cons["g0_install60"]["mean_pz"],
                 probe1["cells"]["cons__perm__g0"]["expr_arm"]))
    xs = np.arange(len(rows))
    ax.bar(xs - 0.19, [r[1] for r in rows], 0.36, color="steelblue",
           edgecolor="k", lw=0.5, label="none")
    ax.bar(xs + 0.19, [r[2] for r in rows], 0.36, color="tab:purple",
           edgecolor="k", lw=0.5, label="perm wpe[0]")
    for i, r in enumerate(rows):
        ax.plot([i - 0.45, i + 0.45], [SPARE_BAR * r[1]] * 2, ls="--",
                color="seagreen", lw=1.0)
        ax.text(i + 0.19, r[2] + 0.02, f"x{r[2] / max(r[1], 1e-12):.3f}",
                ha="center", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("expression p(Z)")
    ax.set_title(f"PROBE 1: perm at both geometries — PRESENCE-AT-NOVEL "
                 f"{probe1['verdict']['PRESENCE_AT_NOVEL']}", fontsize=9.5)
    ax.legend(fontsize=7.5)

    # ---- mask cells
    ax = fig.add_subplot(gs[2, 0])
    mc = [c for c in plane if c["probe"] == "P2"]
    xs = np.arange(len(mc))
    ax.bar(xs, [100 * c["retention"] for c in mc], 0.55,
           color=["tab:red" if c["in_plane"] else "lightgray" for c in mc],
           edgecolor="k", lw=0.5)
    for i, c in enumerate(mc):
        ax.text(i, 100 * c["retention"] + 1.5,
                f"x{c['retention']:.2f}\nCE{c['ce_cost']:+.2f}", ha="center",
                fontsize=6.8)
    ax.axhline(100 * SPARE_BAR, ls=":", color="seagreen", lw=1.0)
    ax.axhline(100 * KILL_RET, ls="--", color="gray", lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([c["tag"].replace("/", "\n") for c in mc], fontsize=6.8)
    ax.set_ylabel("fact retention under mask (%)")
    ax.set_ylim(0, 112)
    ax.set_title("PROBE 2: forced-off-sink (grey = controls: install "
                 "differential, site-stored specificity)", fontsize=9.5)

    # ---- head + probe-4 cells
    ax = fig.add_subplot(gs[2, 1])
    pc3 = [(c["tag"], 100 * c["retention"], c["ce_cost"])
           for c in plane if c["probe"] == "P3"]
    pc4 = [(c["tag"], 100 * c["retention"], c["ce_cost"])
           for c in plane if c["probe"] == "P4"]
    rows = pc3 + pc4
    xs = np.arange(len(rows))
    ax.bar(xs, [r[1] for r in rows], 0.55,
           color=(["tab:green"] * len(pc3)) + (["tab:orange"] * len(pc4)),
           edgecolor="k", lw=0.5)
    for i, r in enumerate(rows):
        ax.text(i, r[1] + 1.5, f"x{r[1] / 100:.2f}\nCE{r[2]:+.2f}",
                ha="center", fontsize=6.8)
    ax.axhline(100 * SPARE_BAR, ls=":", color="seagreen", lw=1.0)
    ax.axhline(100 * KILL_RET, ls="--", color="gray", lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0].replace("-", "\n") for r in rows], fontsize=6.8)
    ax.set_ylabel("fact retention (%)")
    ax.set_ylim(0, 112)
    ax.set_title("PROBE 3 (green): head ablation | PROBE 4 (orange): "
                 "fact-at-position-0 scramble", fontsize=9.5)

    short = adjudication["verdict"].split(" (")[0]
    nm = adjudication["near_misses_report_only"]["cells"]
    nm_txt = (f" | near-miss {nm[0]['tag']} {nm[0]['drop_pct']:.1f}% @ CE "
              f"{nm[0]['ce_cost']:+.2f}") if nm else ""
    fig.suptitle(f"E150 — {short}: kills {adjudication['n_killing_cells']}/"
                 f"{adjudication['n_plane_cells']} plane cells, all CE "
                 f">= +0.70{nm_txt}", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
