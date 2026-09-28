"""E159 — COUPLED-OR-ORGANISM (R46 critic attack 2 — the two cheap
reconcilers; REGISTERED — bars are QUEUE.md row e159 verbatim, written
before compute).

WHY (the day's unresolved contradiction, e150/T086 + R46 critic attack
2): masking ALL attention to position 0 is BENIGN (consolidated fact
spared, x1.03 at CE +0.03) while partially shrinking wpe[0]'s norm to
0.07 is CATASTROPHIC (fact dies, x0.41 @g0 / x0.16 @g-12 at CE +0.84;
0.15 spares at CE +0.31). If the kill were read-mediated, the mask —
which removes EVERY attentional read of row 0 — should kill at least as
hard as the poison. It does not. Only a query-side/global-softmax story
obviously reconciles them (the shrunken k0 redistributes every query's
attention over every other key) — which would make "sink-COUPLED
memory" a misnomer for organism-death. The two cells below decide.

QUEUE.md ROW e159 (VERBATIM): "(a) MASK+LADDER JOINT: norm 0.07 UNDER
the mask — heals (poisoning carried by attention reads; information
sneaks back through the health door) or still kills (query-side
global-softmax collapse; the reframe hardens). (b) SITE-STORED LADDER:
norm ladder on e131_arm_b — dies at the same bracket (sink-poisoning is
organism death; 'COUPLED' is a misnomer) or survives (coupling is
real)".

REGISTERED PREDICTION (verbatim from the dispatch; no shopping):
  - MASK-HEALS: if norm 0.07 UNDER the mask spares the fact (retention
    >= 0.7 at CE <= +0.35) — the poisoning is carried by attention
    reads of row 0; "information flow" sneaks back in through the
    health door; the coupling is read-mediated.
  - POISON-PERSISTS: if the joint cell still kills (retention <= 0.3)
    — poisoning is query-side/global-softmax (the shrunken k0
    redistributes every query's attention); the reframe hardens;
    "COUPLED" is organism-language and the noun needs a further bound.
  - SITE-SPARED: if arm_b's fact survives its ladder at the same
    brackets — sink-poisoning is memory-coupled for the consolidated
    type only (COUPLED survives as a type distinction).
  - SITE-DIES-TOO: if arm_b dies at the same brackets — sink-poisoning
    is organism death, full stop; "COUPLED" is a misnomer; the type
    names need revisiting.
  - No bar shopping; texture => TEXTURE with numbers.

THE TWO PROBES:
  (a) MASK+LADDER JOINT (consolidated net, geometries g0 AND g-12 —
      both must agree for a verdict, pre-registered): set wpe[0] norm
      to 0.07 (direction kept) AND apply the forced-off-sink mask
      simultaneously; measure fact expression + CE. Compare to
      poison-alone (dies @ +0.84) and mask-alone (spares @ +0.03) —
      both rebuilt here and GATED against e151's stored cells.
      Riders (labeled, never gating): joint at 0.15 and the 0.0
      removal anchor under the mask. Manipulation check (texture):
      pre-mask attention mass on key 0, at the read position AND mean
      over queries 1..T-1, healthy vs poisoned state — the quantity
      the query-side story turns on.
  (b) SITE-STORED LADDER (arm_b net at ITS site geometry, Z at x-col
      184, onset read row 183): the norm ladder {0.07, 0.15} + the 0.5
      midpoint (dispatch-allowed) + the 0.0 anchor, on ITS own wpe[0]
      direction. Brackets compared with the consolidated net's
      (0.07, 0.15) [e150 stored, read from runs/e150/metrics.json at
      runtime].

OPERATIONALIZATIONS (fixed before compute):
  * retention = expr_arm / expr_base with expr_base = the SAME net's
    UNMODIFIED expression at that geometry (the healthy consolidated
    base for every probe-A cell; arm_b's base site onset for probe B);
    CE cost = CE_R(arm) - CE_R(same net, unmodified), e065 val-windows
    bank (seed 26502, 60 windows) — a CE column on EVERY cell.
  * probe-A verdict requires BOTH geometries to fire the same branch:
    MASK-HEALS iff retention >= 0.70 & CE <= +0.35 at BOTH g0 and
    g-12; POISON-PERSISTS iff retention <= 0.30 at BOTH; anything else
    (including split geometries) => MIXED-A, texture with numbers.
  * probe-B verdict: SITE-SPARED iff retention >= 0.8x (e150's survive
    convention) at BOTH 0.07 and 0.15; SITE-DIES-TOO iff retention <=
    0.5x (e150's die convention) at 0.07 — dying where the
    consolidated net dies; else MIXED-B, texture with numbers.
  * bracket = (largest dying norm, smallest surviving norm) with
    survive >= 0.8x / die <= 0.5x, e150's convention, per net.

INSTRUMENT PROVENANCE: everything is e150's (lab/e150_flatce_route.py)
VERBATIM — battery_cell/ce (e068/e113/e120 lineage), ce bank
(val_windows, e065 seed 26502), modified_wpe (e131 confinement gate),
the forced-off-sink custom forward (gated against the standard forward;
query row 0 keeps its self-attention), the protocol rebuild (corpus
seed 1337, SPLICE_RNG host shuffle, install-60 split, mix gate), and
the e133 site battery pool-b construction (corpus filler seed 12103,
splice-at-42, Z at x-col 184) for the arm_b net. e150's ladder and
e151's stored joint-ingredient cells are read at runtime from
runs/{e150,e151}/metrics.json (auditable) and gate this run.

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 before torch
import; e152 owns the GPU — never touched), torch.set_num_threads(4),
all evals sequential, no training, minutes. Outputs:
runs/e159/{metrics.json, coupled_organism.png}. No checkpoints
written (eval-only).

Run:  cd lab && python e159_coupled_organism.py     (E159_SMOKE=1 -> shakedown)
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

SMOKE = os.environ.get("E159_SMOKE") == "1"
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
ARMB_CK = CKPT_DIR / "e131_arm_b_corpus_spliced.pt"  # site-stored (its ladder never ran)
E150_METRICS = E43.REPO / "runs" / "e150" / "metrics.json"   # consolidated ladder + bracket
E151_METRICS = E43.REPO / "runs" / "e151" / "metrics.json"   # stored joint-ingredient cells (gates)

GEOS = (0, -12)                   # trained + novel geometry; both must agree

R_EVAL_SEED = 26502               # e065 CE_R bank seed (verbatim)

# site battery geometry (e133 verbatim)
FACT_PRE, FACT_POST = 12, 12
FACT_LEN = FACT_PRE + len(NAME) + FACT_POST      # 31
SPLICE_AT = 42
Z_XCOL = PRE + SPLICE_AT + FACT_PRE             # 184
SPLICE_ADDR_ROW = Z_XCOL - 1                    # 183
CORP_CONT_SEED = 12103
N_PROMPTS = 8 if SMOKE else 30

# probe-A cells (registered): 0.07 = the poison bracket; riders labeled
JOINT_NORM = 0.07
RIDER_JOINT_NORMS = () if SMOKE else (0.15, 0.0)
# probe-B ladder (registered): same brackets + 0.5 midpoint + 0.0 anchor
ARMB_LADDER = (0.07, 0.5) if SMOKE else (0.07, 0.15, 0.5)
ARMB_ANCHOR_ZERO = 0.0

# gates / references (full precision, from stored metrics)
G_BIT_TOL = 5e-6                  # same-instrument reproduction gate
G_CROSS_TOL = 1e-4                # cross-run (e151's own runs differ ~1e-7)
G_CONS_REF_PZ = 0.7850371599197388          # e131/e150/e151 none g0 install60
G_CONS_REF_CE = 1.663516640663147           # e131/e150/e151 none ce_r
G_CONS_REF_GM12 = 0.9155886173248291        # e150 base g-12 install60
G_E151_MASK_G0 = 0.808849573135376          # e151 before.mask g+0 mask_pz
G_E151_MASK_GM12 = 0.9242185950279236       # e151 before.mask g-12 mask_pz
G_E151_MASK_CE = 1.6974624395370483         # e151 before.mask ce_mask
G_E151_P07_G0 = 0.3231039047241211          # e151 before.ladder 0.07 g+0
G_E151_P07_GM12 = 0.14433333277702332       # e151 before.ladder 0.07 g-12
G_E151_P07_CE = 2.506317377090454           # e151 before.ladder 0.07 ce_r
G_E151_P15_G0 = 0.7846040725708008          # e151 before.ladder 0.15 g+0
G_E151_P15_CE = 1.9703083038330078          # e151 before.ladder 0.15 ce_r
G_ARMB_REF_STD = 0.007800613064318895       # e133/e150 std battery floor (site net)
G_ARMB_REF_SITE = 0.9880021214485168        # e133/e150 site onset
G_ARMB_REF_CE = 1.682660698890686           # e150 armb causal-custom ce_base

# registered bars (numeric)
HEAL_RET = 0.70                   # MASK-HEALS retention bar
HEAL_CE = 0.35                    # MASK-HEALS CE bar
PERSIST_RET = 0.30                # POISON-PERSISTS retention bar
SPARE_RET = 0.80                  # survive convention (e150)
DIE_RET = 0.50                    # die convention (e150)

REGISTERED_PREDICTION = {
    "queue_row_verbatim": (
        "(a) MASK+LADDER JOINT: norm 0.07 UNDER the mask — heals "
        "(poisoning carried by attention reads; information sneaks back "
        "through the health door) or still kills (query-side global-softmax "
        "collapse; the reframe hardens). (b) SITE-STORED LADDER: norm ladder "
        "on e131_arm_b — dies at the same bracket (sink-poisoning is "
        "organism death; 'COUPLED' is a misnomer) or survives (coupling is "
        "real)"),
    "bars_verbatim": {
        "MASK-HEALS": "if norm 0.07 UNDER the mask spares the fact "
                      "(retention >= 0.7 at CE <= +0.35) — the poisoning is "
                      "carried by attention reads of row 0; 'information "
                      "flow' sneaks back in through the health door; the "
                      "coupling is read-mediated.",
        "POISON-PERSISTS": "if the joint cell still kills (retention <= 0.3) "
                           "— poisoning is query-side/global-softmax (the "
                           "shrunken k0 redistributes every query's "
                           "attention); the reframe hardens; 'COUPLED' is "
                           "organism-language and the noun needs a further "
                           "bound.",
        "SITE-SPARED": "if arm_b's fact survives its ladder at the same "
                       "brackets — sink-poisoning is memory-coupled for the "
                       "consolidated type only (COUPLED survives as a type "
                       "distinction).",
        "SITE-DIES-TOO": "if arm_b dies at the same brackets — sink-poisoning "
                         "is organism death, full stop; 'COUPLED' is a "
                         "misnomer; the type names need revisiting.",
    },
    "operationalizations": (
        "retention = expr_arm / expr_base(healthy same net, same geometry); "
        "CE cost vs same-net unmodified CE_R (e065 bank seed 26502); probe A "
        "verdict requires BOTH g0 and g-12 to fire the same branch (heal "
        ">=0.70 & CE<=+0.35 / persist <=0.30), else MIXED-A; probe B: spared "
        ">= 0.8x at BOTH 0.07 and 0.15, dies-too <= 0.5x at 0.07, else "
        "MIXED-B; bracket = (largest dying, smallest surviving) with "
        "survive 0.8x / die 0.5x (e150 convention); poison-alone and "
        "mask-alone rebuilt and gated against e151's stored cells "
        "(cross-run tol 1e-4); joint riders (0.15, 0.0) and the "
        "all-queries sink-mass check are labeled texture, never gating."),
    "no_bar_shopping": "No bar shopping. Texture => TEXTURE with numbers.",
}

recipe_deviations: list[str] = [
    "Eval-only battery: both nets are LOADED, gated artifacts (e131 "
    "consolidated + e131 arm_b); nothing regenerated, nothing trained.",
    "Every probe-A cell's base is the healthy consolidated net (the same "
    "base poison-alone and mask-alone were measured against in e150/e151); "
    "the mask path (custom forward, causal-only) is bit-identical to the "
    "standard forward (gated, max|dp(Z)| recorded), so path-matching is "
    "immaterial here — stated, recorded.",
    "The joint cell is a doubly-off-distribution state (the net never "
    "trained under mask nor poison); its retention is read RELATIVE to the "
    "two single-intervention cells, not as an absolute health statement.",
    "Probe-A verdict requires both geometries (g0 AND g-12) to fire the "
    "same registered branch; split geometries => MIXED-A with numbers "
    "(pre-registered before compute).",
    "arm_b's ladder includes the 0.0 removal anchor (e150's ladder "
    "convention) and the dispatch-allowed 0.5 midpoint; the bars adjudicate "
    "on the registered brackets 0.07/0.15 only.",
    "e150's consolidated ladder and e151's stored joint-ingredient cells "
    "are read at runtime from runs/{e150,e151}/metrics.json (auditable), "
    "not hand-copied.",
]

trims: list[str] = []


# ------------------------------------------------------------------ instruments
# (e150 verbatim: load_cpu/evl_load/battery_fwd/ce_fwd/val_windows/
#  modified_wpe/forward_custom/fwd_causal/fwd_offsink/sink_mass; e133
#  verbatim: site battery construction + site_reads; provenance inline)

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
def sink_mass_profile(net: TinyGPT, ids: torch.Tensor, bs=30) -> dict:
    """e150's sink_mass_at_read extended: pre-mask attention mass on key 0
    per layer — at the READ position (T-1) AND mean over queries 1..T-1
    (the query-side/global-softmax texture quantity). Mean over
    heads/batch; manual-softmax measurement (e133 lineage)."""
    cfg = net.cfg
    H, D = cfg.n_head, cfg.n_embd // cfg.n_head
    per_layer = {l: {"read": [], "allq": []} for l in range(cfg.n_layer)}
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
            per_layer[l]["read"].append(float(att[:, :, Tt - 1, 0].mean()))
            per_layer[l]["allq"].append(
                float(att[:, :, 1:, 0].mean()))
            y = (att @ v.view(B, Tt, H, D).transpose(1, 2))
            emb = emb + block.attn.c_proj(
                y.transpose(1, 2).contiguous().view(B, Tt, cfg.n_embd))
            emb = emb + block.mlp(block.ln2(emb))
    return {f"L{l}": {"read_pos": float(np.mean(v["read"])),
                      "mean_queries_ge1": float(np.mean(v["allq"]))}
            for l, v in per_layer.items()}


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e159_smoke" if SMOKE else "e159")
    log(f"E159 COUPLED-OR-ORGANISM (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-only, torch threads {torch.get_num_threads()}, "
        f"cuda available = {torch.cuda.is_available()}")

    # ---------------- protocol rebuild (e150 verbatim)
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
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

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

    # ---------------- e150/e151 stored cells (read at runtime, auditable)
    e150 = json.loads(E150_METRICS.read_text(encoding="utf-8"))
    e151 = json.loads(E151_METRICS.read_text(encoding="utf-8"))
    e150_ladder = e150["probe5_norm_ladder"]["ladder"]
    e150_bracket = e150["probe5_norm_ladder"]["threshold_bracket_g0"]
    e151_mask = e151["before"]["mask"]
    e151_ladder = {e["target_norm"]: e for e in e151["before"]["ladder"]}
    log(f"e150 stored ladder bracket (g0): {e150_bracket['bracket']}; "
        f"e151 stored mask/ladder cells loaded for gating")

    # ---------------- nets + gates
    log("--- PHASE 0: load + gate the two artifacts ---")
    gates: dict = {}

    net_cons = load_cpu(CONS_CK)
    bz_cons = {j: battery_fwd(net_cons, bat_ids[j], zid) for j in GEOS}
    ce_cons = ce_fwd(net_cons, *r_eval_xy)
    gates["consolidated"] = {
        "battery_pz_g0": bz_cons[0]["mean_pz"], "ref_g0": G_CONS_REF_PZ,
        "battery_pz_gm12": bz_cons[-12]["mean_pz"], "ref_gm12": G_CONS_REF_GM12,
        "ce_r": ce_cons, "ref_ce": G_CONS_REF_CE,
        "pass": bool(abs(bz_cons[0]["mean_pz"] - G_CONS_REF_PZ) < G_BIT_TOL
                     and abs(bz_cons[-12]["mean_pz"] - G_CONS_REF_GM12)
                     < G_BIT_TOL
                     and abs(ce_cons - G_CONS_REF_CE) < G_BIT_TOL)}
    log(f"G_CONS: p(Z) g0 {bz_cons[0]['mean_pz']:.10f} g-12 "
        f"{bz_cons[-12]['mean_pz']:.10f} CE_R {ce_cons:.6f}: "
        f"{'PASS' if gates['consolidated']['pass'] else 'FAIL'}")
    if not gates["consolidated"]["pass"]:
        raise RuntimeError("consolidated checkpoint failed its gate")

    # mask-instrument validation (custom-causal == standard)
    with torch.no_grad():
        lg_std, _ = net_cons(bat_ids[0][:6])
        lg_cus, _ = forward_custom(net_cons, bat_ids[0][:6], block_key0=False)
        dmax = float((lg_std - lg_cus).abs().max())
    pz_cus = battery_fwd(net_cons, bat_ids[0], zid,
                         fwd=fwd_causal(net_cons))["mean_pz"]
    dpz = abs(pz_cus - bz_cons[0]["mean_pz"])
    mask_gate = {"max_abs_logit_diff_batch6": dmax,
                 "abs_dpz_install60_g0": dpz,
                 "tol_logit": 1e-2, "tol_pz": 1e-3,
                 "pass": bool(dmax < 1e-2 and dpz < 1e-3)}
    gates["mask_instrument"] = mask_gate
    log(f"MASK-GATE: max|dlogit| {dmax:.3e}, |dp(Z)| {dpz:.3e}: "
        f"{'PASS' if mask_gate['pass'] else 'FAIL'}")
    if not mask_gate["pass"]:
        raise RuntimeError("custom forward does not reproduce the standard one")

    sd_cons = {k: v.clone() for k, v in net_cons.state_dict().items()}
    wpe0_cons = sd_cons["wpe.weight"][0].clone()
    n0_cons = float(wpe0_cons.norm())
    log(f"consolidated wpe[0] norm {n0_cons:.4f}")

    ev = evl_load(sd_cons)          # reusable eval twin (no RNG in eval)

    CELLS: list[dict] = []

    def cell(probe, tag, net_tag, geo, expr_base, expr_arm, ce_base, ce_arm,
             rider=False, extra=None):
        ret = expr_arm / max(expr_base, 1e-12)
        c = {"probe": probe, "tag": tag, "net": net_tag, "geo": geo,
             "expr_base": expr_base, "expr_arm": expr_arm,
             "retention": ret, "fact_drop_pct": 100.0 * (1.0 - ret),
             "ce_base": ce_base, "ce_arm": ce_arm,
             "ce_cost": ce_arm - ce_base,
             "rider": bool(rider)}
        if extra:
            c.update(extra)
        CELLS.append(c)
        log(f"  [{probe} | {tag:28s}] expr {expr_base:.4f} -> {expr_arm:.4f} "
            f"(x{ret:.3f}, drop {100 * (1 - ret):5.1f}%) | CE {ce_base:.4f} "
            f"-> {ce_arm:.4f} (cost {ce_arm - ce_base:+.4f})"
            + ("  [rider]" if rider else ""))
        return c

    @torch.no_grad()
    def cons_state_eval(sd, fwd=None, label=""):
        """Load a state into the eval twin; battery at both geos + CE."""
        ev.load_state_dict(sd)
        f = fwd
        out = {f"g{j}": battery_fwd(ev, bat_ids[j], zid, fwd=f)["mean_pz"]
               for j in GEOS}
        out["ce_r"] = ce_fwd(ev, *r_eval_xy, fwd=f)
        log(f"  [{label:28s}] " + " ".join(
            f"{k} {v:.4f}" for k, v in out.items()))
        return out

    # =====================================================================
    # PROBE A — MASK+LADDER JOINT (the reconciliation cell)
    # =====================================================================
    log("--- PROBE A: mask + ladder joint on the consolidated net ---")
    probeA: dict = {"cells": {}}

    # (0) healthy base
    base_out = cons_state_eval(sd_cons, label="cons none (healthy)")
    assert abs(base_out["g0"] - G_CONS_REF_PZ) < G_BIT_TOL

    # (1) mask-alone (gate vs e151 stored)
    m_out = cons_state_eval(sd_cons, fwd=fwd_offsink(ev), label="mask alone")
    g_e151_mask = {
        "g0": {"value": m_out["g0"], "ref": G_E151_MASK_G0,
               "pass": bool(abs(m_out["g0"] - G_E151_MASK_G0) < G_CROSS_TOL)},
        "gm12": {"value": m_out["g-12"], "ref": G_E151_MASK_GM12,
                 "pass": bool(abs(m_out["g-12"] - G_E151_MASK_GM12)
                              < G_CROSS_TOL)},
        "ce": {"value": m_out["ce_r"], "ref": G_E151_MASK_CE,
               "pass": bool(abs(m_out["ce_r"] - G_E151_MASK_CE)
                            < G_CROSS_TOL)}}
    gates["e151_mask_cells"] = g_e151_mask
    log(f"G_E151_MASK: g0 {m_out['g0']:.10f} g-12 {m_out['g-12']:.10f} "
        f"CE {m_out['ce_r']:.6f}: "
        f"{'PASS' if all(v['pass'] for v in g_e151_mask.values()) else 'FAIL'}")
    if not all(v["pass"] for v in g_e151_mask.values()):
        raise RuntimeError("mask-alone does not reproduce e151's stored cells")

    # (2) poison-alone 0.07 (gate vs e151 stored)
    sd_p07, gate_p07 = modified_wpe(sd_cons, 0,
                                    wpe0_cons * (JOINT_NORM / n0_cons))
    assert gate_p07["pass"], gate_p07
    p07_out = cons_state_eval(sd_p07, label=f"poison {JOINT_NORM} alone")
    g_e151_p07 = {
        "g0": {"value": p07_out["g0"], "ref": G_E151_P07_G0,
               "pass": bool(abs(p07_out["g0"] - G_E151_P07_G0) < G_CROSS_TOL)},
        "gm12": {"value": p07_out["g-12"], "ref": G_E151_P07_GM12,
                 "pass": bool(abs(p07_out["g-12"] - G_E151_P07_GM12)
                              < G_CROSS_TOL)},
        "ce": {"value": p07_out["ce_r"], "ref": G_E151_P07_CE,
               "pass": bool(abs(p07_out["ce_r"] - G_E151_P07_CE)
                            < G_CROSS_TOL)}}
    gates["e151_poison07_cells"] = g_e151_p07
    log(f"G_E151_P07: g0 {p07_out['g0']:.10f} g-12 {p07_out['g-12']:.10f} "
        f"CE {p07_out['ce_r']:.6f}: "
        f"{'PASS' if all(v['pass'] for v in g_e151_p07.values()) else 'FAIL'}")
    if not all(v["pass"] for v in g_e151_p07.values()):
        raise RuntimeError("poison-alone does not reproduce e151's stored cells")

    # (3) THE JOINT CELL: norm 0.07 UNDER the mask
    j_out = cons_state_eval(sd_p07, fwd=fwd_offsink(ev),
                            label=f"JOINT {JOINT_NORM} + mask")

    for j in GEOS:
        probeA["cells"][f"mask__g{j}"] = cell(
            "A", f"mask@g{j:+d}", "consolidated", j,
            base_out[f"g{j}"], m_out[f"g{j}"], base_out["ce_r"],
            m_out["ce_r"],
            extra={"note": "forced-off-sink mask alone (e150 P2 / e151 "
                           "before.mask cell, rebuilt + gated)"})
        probeA["cells"][f"poison07__g{j}"] = cell(
            "A", f"poison{JOINT_NORM}@g{j:+d}", "consolidated", j,
            base_out[f"g{j}"], p07_out[f"g{j}"], base_out["ce_r"],
            p07_out["ce_r"],
            extra={"target_norm": JOINT_NORM, "gate": gate_p07,
                   "note": "norm ladder 0.07 alone (e150 P5 / e151 "
                           "before.ladder cell, rebuilt + gated)"})
        probeA["cells"][f"joint07__g{j}"] = cell(
            "A", f"JOINT {JOINT_NORM}+mask@g{j:+d}", "consolidated", j,
            base_out[f"g{j}"], j_out[f"g{j}"], base_out["ce_r"],
            j_out["ce_r"],
            extra={"target_norm": JOINT_NORM, "gate": gate_p07,
                   "note": "THE reconciliation cell: wpe[0] norm 0.07 AND "
                           "forced-off-sink mask, simultaneously"})

    # (4) riders: joint at 0.15 and the 0.0 removal anchor under the mask
    for tgt in RIDER_JOINT_NORMS:
        if tgt == 0.0:
            row = torch.zeros_like(wpe0_cons)
        else:
            row = wpe0_cons * (tgt / n0_cons)
        sd_t, gate_t = modified_wpe(sd_cons, 0, row)
        assert gate_t["pass"], gate_t
        o = cons_state_eval(sd_t, fwd=fwd_offsink(ev),
                            label=f"joint {tgt:g} + mask [rider]")
        for j in GEOS:
            probeA["cells"][f"joint{tgt:g}__g{j}__RIDER"] = cell(
                "A", f"joint{tgt:g}+mask@g{j:+d} [rider]", "consolidated",
                j, base_out[f"g{j}"], o[f"g{j}"], base_out["ce_r"],
                o["ce_r"], rider=True,
                extra={"target_norm": tgt, "gate": gate_t,
                       "role": "REPORT-ONLY rider: the dose dimension of "
                               "the joint cell; never gates a bar"})

    # (5) manipulation check: pre-mask sink mass, healthy vs poisoned
    ev.load_state_dict(sd_cons)
    sm_healthy = sink_mass_profile(ev, bat_ids[0])
    ev.load_state_dict(sd_p07)
    sm_poison = sink_mass_profile(ev, bat_ids[0])
    ev.load_state_dict(sd_cons)
    probeA["sink_mass_premask"] = {
        "healthy": sm_healthy, "poison07": sm_poison,
        "note": ("pre-mask attention mass on key 0 per layer (mean over "
                 "heads/batch, install60@g0): read_pos = at the battery's "
                 "last-position read; mean_queries_ge1 = mean over ALL "
                 "query positions 1..T-1 — the quantity the query-side/"
                 "global-softmax story turns on. Post-mask both are exactly "
                 "0 by construction. TEXTURE, never gating.")}
    tot_h = sum(v["mean_queries_ge1"] for v in sm_healthy.values())
    tot_p = sum(v["mean_queries_ge1"] for v in sm_poison.values())
    log(f"  sink mass (mean over queries>=1, all layers): healthy "
        f"{tot_h:.4f} -> poisoned {tot_p:.4f}")

    # probe-A adjudication (registered — both geometries must agree)
    def branch(c):
        if c["retention"] >= HEAL_RET and c["ce_cost"] <= HEAL_CE:
            return "MASK-HEALS"
        if c["retention"] <= PERSIST_RET:
            return "POISON-PERSISTS"
        return "GAP"

    per_geo = {j: branch(probeA["cells"][f"joint07__g{j}"])
               for j in GEOS}
    rets = {j: probeA["cells"][f"joint07__g{j}"]["retention"] for j in GEOS}
    ces = {j: probeA["cells"][f"joint07__g{j}"]["ce_cost"] for j in GEOS}
    if all(b == "MASK-HEALS" for b in per_geo.values()):
        a_out, a_txt = "MASK-HEALS", (
            "the poisoning is carried by attention reads of row 0; "
            "'information flow' sneaks back in through the health door; "
            "the coupling is read-mediated")
    elif all(b == "POISON-PERSISTS" for b in per_geo.values()):
        a_out, a_txt = "POISON-PERSISTS", (
            "poisoning is query-side/global-softmax (the shrunken k0 "
            "redistributes every query's attention); the reframe hardens; "
            "'COUPLED' is organism-language and the noun needs a further "
            "bound")
    else:
        a_out = "MIXED-A"
        a_txt = ("the joint cell lands in the gap or splits across "
                 "geometries: " + "; ".join(
                     f"g{j}: x{rets[j]:.3f}@CE{ces[j]:+.3f} [{per_geo[j]}]"
                     for j in GEOS) + " — texture with numbers")
    probeA["verdict"] = {
        "branches_per_geometry": {f"g{j}": per_geo[j] for j in GEOS},
        "joint_retentions": {f"g{j}": rets[j] for j in GEOS},
        "joint_ce_costs": {f"g{j}": ces[j] for j in GEOS},
        "bars": {"heal": f"retention >= {HEAL_RET} at CE <= +{HEAL_CE} at "
                         f"BOTH {GEOS}",
                 "persist": f"retention <= {PERSIST_RET} at BOTH {GEOS}"},
        "anchor_cells": {"mask_alone": {
            f"g{j}": probeA["cells"][f"mask__g{j}"]["retention"]
            for j in GEOS},
            "poison07_alone": {
                f"g{j}": probeA["cells"][f"poison07__g{j}"]["retention"]
                for j in GEOS}},
        "outcome": a_out, "verdict": a_txt}
    log(f"PROBE A VERDICT: {a_out} ({a_txt})")

    # =====================================================================
    # PROBE B — SITE-STORED LADDER (arm_b at ITS site geometry)
    # =====================================================================
    log("--- PROBE B: norm ladder on the site-stored net (arm_b) ---")
    net_armb = load_cpu(ARMB_CK)
    bz_armb_std = battery_fwd(net_armb, bat_ids[0], zid)["mean_pz"]
    st_armb = site_reads(net_armb)
    ce_armb = ce_fwd(net_armb, *r_eval_xy)
    gates["site_stored"] = {
        "std_g0_floor": bz_armb_std, "ref_std": G_ARMB_REF_STD,
        "site_onset": st_armb["onset_pz"], "ref_site": G_ARMB_REF_SITE,
        "ce_r": ce_armb, "ref_ce": G_ARMB_REF_CE,
        "pass": False}
    # site-onset tolerance: bit-tight on the full pool (deterministic
    # construction, same seeds as e150); loose in smoke (reduced pool)
    tol_site = G_BIT_TOL if not SMOKE else 0.05
    gates["site_stored"]["site_tol"] = tol_site
    gates["site_stored"]["pass"] = bool(
        abs(bz_armb_std - G_ARMB_REF_STD) < G_BIT_TOL
        and abs(st_armb["onset_pz"] - G_ARMB_REF_SITE) < tol_site
        and abs(ce_armb - G_ARMB_REF_CE) < G_CROSS_TOL)
    log(f"G_ARMB: std floor {bz_armb_std:.10f} site onset "
        f"{st_armb['onset_pz']:.10f} CE_R {ce_armb:.6f}: "
        f"{'PASS' if gates['site_stored']['pass'] else 'FAIL'} "
        f"(site tol {tol_site:g})")
    if not gates["site_stored"]["pass"]:
        raise RuntimeError("arm_b checkpoint failed its gate")

    sd_armb = {k: v.clone() for k, v in net_armb.state_dict().items()}
    wpe0_armb = sd_armb["wpe.weight"][0].clone()
    n0_armb = float(wpe0_armb.norm())
    evb = evl_load(sd_armb)
    log(f"arm_b wpe[0] norm {n0_armb:.4f} (consolidated {n0_cons:.4f})")

    probeB: dict = {"cells": {}, "ladder": [], "base": {
        "site_onset": st_armb["onset_pz"],
        "pname_mean_over7": st_armb["pname_mean_over7"],
        "std_g0_floor": bz_armb_std, "ce_r": ce_armb,
        "wpe0_norm": n0_armb}}
    for tgt in sorted(list(ARMB_LADDER) + [ARMB_ANCHOR_ZERO]):
        if tgt == 0.0:
            row = torch.zeros_like(wpe0_armb)
        else:
            row = wpe0_armb * (tgt / n0_armb)
        sd_t, gate_t = modified_wpe(sd_armb, 0, row)
        assert gate_t["pass"], gate_t
        evb.load_state_dict(sd_t)
        sr = site_reads(evb)
        oce = ce_fwd(evb, *r_eval_xy)
        ent = {"target_norm": tgt, "actual_norm": float(row.norm()),
               "gate": gate_t, "site_onset": sr["onset_pz"],
               "pname_mean_over7": sr["pname_mean_over7"], "ce_r": oce}
        probeB["ladder"].append(ent)
        probeB["cells"][f"armb_norm{tgt:g}"] = cell(
            "B", f"armB/norm={tgt:g}@site", "site_stored", SPLICE_ADDR_ROW,
            st_armb["onset_pz"], sr["onset_pz"], ce_armb, oce,
            rider=bool(tgt not in (0.07, 0.15)),
            extra={"target_norm": tgt,
                   "pname_mean_over7": sr["pname_mean_over7"],
                   "metric": "site onset p(Z) at row 183 (e133 battery)",
                   "role": (None if tgt in (0.07, 0.15) else
                            ("0.5 midpoint (dispatch-allowed), report-only"
                             if tgt == 0.5 else
                             "0.0 removal anchor (e150 ladder convention), "
                             "report-only"))})
    evb.load_state_dict(sd_armb)

    # probe-B adjudication (registered — the same brackets as consolidated)
    lad = {e["target_norm"]: e for e in probeB["ladder"]}
    r07 = lad[0.07]["site_onset"] / st_armb["onset_pz"]
    r15 = (lad[0.15]["site_onset"] / st_armb["onset_pz"]
           if 0.15 in lad else None)
    if r07 >= SPARE_RET and (r15 is None or r15 >= SPARE_RET):
        b_out, b_txt = "SITE-SPARED", (
            "sink-poisoning is memory-coupled for the consolidated type "
            "only (COUPLED survives as a type distinction)")
    elif r07 <= DIE_RET:
        b_out, b_txt = "SITE-DIES-TOO", (
            "sink-poisoning is organism death, full stop; 'COUPLED' is a "
            "misnomer; the type names need revisiting")
    else:
        b_out = "MIXED-B"
        b_txt = (f"arm_b's 0.07 cell lands between the bars (x{r07:.3f}; "
                 f"survive >= {SPARE_RET}, die <= {DIE_RET}) — texture "
                 f"with numbers")
    surv = [e["target_norm"] for e in probeB["ladder"]
            if e["site_onset"] >= SPARE_RET * st_armb["onset_pz"]]
    dead = [e["target_norm"] for e in probeB["ladder"]
            if e["site_onset"] <= DIE_RET * st_armb["onset_pz"]]
    probeB["verdict"] = {
        "retention_0.07": r07, "retention_0.15": r15,
        "bars": {"spared": f"retention >= {SPARE_RET} at BOTH 0.07 and 0.15",
                 "dies_too": f"retention <= {DIE_RET} at 0.07"},
        "threshold_bracket_site": {
            "smallest_surviving_norm": min(surv) if surv else None,
            "largest_dying_norm": max(dead) if dead else None,
            "bracket": f"({max(dead) if dead else 0.0}, "
                       f"{min(surv) if surv else float('inf')})"},
        "consolidated_bracket_g0_stored": e150_bracket["bracket"],
        "outcome": b_out, "verdict": b_txt}
    log(f"PROBE B VERDICT: {b_out} ({b_txt}); site bracket "
        f"({max(dead) if dead else 0.0}, {min(surv) if surv else 'inf'}) "
        f"vs consolidated {e150_bracket['bracket']}")

    # =====================================================================
    # combined adjudication — what happens to the 'sink-coupled' noun
    # =====================================================================
    noun = {
        ("MASK-HEALS", "SITE-SPARED"): (
            "read-mediated coupling + type distinction: the consolidated "
            "fact's poisoning rides attention READS of row 0 and arm_b's "
            "fact does not need sink health — 'sink-COUPLED' is accurate "
            "as a type noun (coupled = read-coupled), and organism-death "
            "language stays out of it."),
        ("MASK-HEALS", "SITE-DIES-TOO"): (
            "read-mediated healing for the consolidated fact, but the "
            "poison kills BOTH nets' facts at 0.07 — the shrunken row 0 "
            "damages the organism generally while the consolidated fact's "
            "own dependence is read-mediated: the noun splits into "
            "READ-COUPLED (consolidated) vs organism-collateral (everyone) "
            "— 'COUPLED' survives only with that qualifier."),
        ("MASK-HEALS", "MIXED-B"): (
            "read-mediated coupling on probe A; probe B texture — the noun "
            "keeps 'sink-coupled' for the consolidated type with probe B's "
            "numbers quoted verbatim."),
        ("POISON-PERSISTS", "SITE-SPARED"): (
            "the reframe hardens AND the type distinction survives: the "
            "consolidated fact dies of query-side/global-softmax collapse "
            "while arm_b's fact survives the same poison — 'sink-coupled' "
            "is memory-coupled (the consolidated memory NEEDS the sink "
            "organism-intact), which is exactly T086's reading; the noun "
            "survives with 'coupled = health-coupled, not read-coupled'."),
        ("POISON-PERSISTS", "SITE-DIES-TOO"): (
            "'COUPLED' IS A MISNOMER — organism death, full stop: the "
            "joint cell still kills (query-side collapse) and arm_b's fact "
            "dies at the same bracket; sink-poisoning kills every memory "
            "in every net tested; the type names need revisiting "
            "(sink-HEALTH-DEPENDENT, both types — the type distinction "
            "must stand on its e139/e143 cells, not on poison sparing)."),
        ("POISON-PERSISTS", "MIXED-B"): (
            "the reframe hardens on probe A; probe B texture — noun "
            "revisiting deferred with probe B's numbers quoted verbatim."),
    }
    key = (a_out, b_out)
    joint_txt = "; ".join(f"g{j} x{rets[j]:.3f}@CE{ces[j]:+.3f}"
                          for j in GEOS)
    armb_txt = (f"arm_b 0.07 x{r07:.3f}"
                + (f", 0.15 x{r15:.3f}" if r15 is not None else ""))
    mixed_fallback = ("at least one probe landed MIXED — no clean noun "
                      "fold; quote the numbers: joint [" + joint_txt
                      + "]; " + armb_txt)
    combined = {
        "probe_a_outcome": a_out, "probe_b_outcome": b_out,
        "noun_verdict": noun.get(key, mixed_fallback),
    }
    log("=" * 78)
    log(f"E159 COMBINED: A={a_out} B={b_out} -> {combined['noun_verdict']}")
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "e159_coupled_organism",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registration": ("R46 critic attack 2 (the mask/ladder "
                         "contradiction) — the two cheap reconcilers. "
                         "Docstring + bars written before compute."),
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("is the consolidated fact's sink-poisoning carried by "
                     "attention READS of row 0 (mask heals; coupling is "
                     "read-mediated) or query-side/global-softmax (joint "
                     "still kills; organism death)? and is the poison "
                     "memory-coupled (arm_b survives its ladder) or "
                     "organism-wide (arm_b dies too)?"),
        "nets": {
            "consolidated": f"runs/checkpoints/{CONS_CK.name} (loaded, gated "
                            f"vs e131/e150/e151 + e151 stored cells)",
            "site_stored": f"runs/checkpoints/{ARMB_CK.name} (loaded, gated; "
                           f"its ladder had never been run)",
            "eval_only": True,
        },
        "gates": {"G_SPLICE": G_SPLICE, "consolidated": gates["consolidated"],
                  "mask_instrument": mask_gate,
                  "e151_mask_cells": g_e151_mask,
                  "e151_poison07_cells": g_e151_p07,
                  "site_stored": gates["site_stored"]},
        "probe_a_mask_ladder_joint": probeA,
        "probe_b_site_stored_ladder": probeB,
        "cells": CELLS,
        "combined": combined,
        "honesty_reflex": {
            "mask_off_distribution": "the joint cell stacks TWO "
                "off-distribution manipulations (mask + poison), each "
                "individually off the training distribution; its CE column "
                "prices only the generic LM damage — the heal/persist "
                "reading is anchored by the two single-intervention cells "
                "(rebuilt here and gated bit-tight against e151's stored "
                "values), not by absolute health claims.",
            "retention_base_choice": "every retention is vs the SAME net's "
                "unmodified expression at the same geometry — the same "
                "convention e150's plane used; the custom-causal base is "
                "bit-identical to standard (mask gate), so path-matching "
                "cannot move a number.",
            "probe_b_expression_metric": "arm_b's fact is read at row 183 "
                "(onset p(Z)) — a different battery from the consolidated "
                "install60 read; retentions compare each net to ITS OWN "
                "base (never cross-net raw levels); the bracket comparison "
                "is the registered like-for-like.",
            "single_net_caveats": "one consolidated net + one arm_b net "
                "(e131 lineage, single recipe); the R150 rider in e150 "
                "agreed with the consolidated ladder at 0.07/0.15, but "
                "verdicts are lineage-specific until replication seeds "
                "land.",
            "ce_bank_is_position_blind": "CE_R windows are 256-token corpus "
                "windows that never contain the fact; the CE column prices "
                "general LM damage, not fact-geometry damage (e150's "
                "caveat, inherited).",
            "sink_mass_is_texture": "the all-queries sink-mass check is a "
                "manipulation/texture read-out for the query-side story; "
                "it gates nothing.",
        },
        "trims": trims,
        "recipe_deviations": recipe_deviations,
        "ckpt_inventory": {
            "saved": {},
            "external_used": [f"runs/checkpoints/{p.name}"
                              for p in (CONS_CK, ARMB_CK)],
            "metrics_read": [str(E150_METRICS.relative_to(E43.REPO)),
                             str(E151_METRICS.relative_to(E43.REPO))],
            "note": "eval-only: no checkpoints written or regenerated",
        },
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": 2739072,
                   "device": "cpu", "torch_threads": torch.get_num_threads(),
                   "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "coupled_organism.png", probeA, probeB, combined, e150_ladder,
         e150_bracket, base_out, st_armb, ce_armb)
    log(f"outputs: {rd / 'metrics.json'}, {rd / 'coupled_organism.png'}")
    log(f"total {time.time() - T0:.1f}s")


# ------------------------------------------------------------------ plot

def plot(path, probeA, probeB, combined, e150_ladder, e150_bracket,
         base_out, st_armb, ce_armb):
    """The reconciliation panel: mask/poison/joint cells (both geometries)
    + the fact-vs-CE plane + both nets' ladders."""
    fig = plt.figure(figsize=(16.0, 11.0))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.15, 1.0))

    # ---- panel 1 (top, spans): the reconciliation quartet, retention
    ax = fig.add_subplot(gs[0, :])
    series = [
        ("base (healthy)", {j: 1.0 for j in GEOS}, None, "steelblue"),
        ("mask alone", {j: probeA["cells"][f"mask__g{j}"]["retention"]
                        for j in GEOS},
         {j: probeA["cells"][f"mask__g{j}"]["ce_cost"] for j in GEOS},
         "seagreen"),
        ("poison 0.07 alone", {j: probeA["cells"][f"poison07__g{j}"]
                               ["retention"] for j in GEOS},
         {j: probeA["cells"][f"poison07__g{j}"]["ce_cost"] for j in GEOS},
         "firebrick"),
        ("JOINT 0.07 + mask", {j: probeA["cells"][f"joint07__g{j}"]
                               ["retention"] for j in GEOS},
         {j: probeA["cells"][f"joint07__g{j}"]["ce_cost"] for j in GEOS},
         "darkviolet"),
    ]
    riders = []
    for tgt in (0.15, 0.0):
        keys = {j: f"joint{tgt:g}__g{j}__RIDER" for j in GEOS}
        if all(k in probeA["cells"] for k in keys.values()):
            riders.append((f"joint {tgt:g} + mask [rider]",
                           {j: probeA["cells"][keys[j]]["retention"]
                            for j in GEOS},
                           {j: probeA["cells"][keys[j]]["ce_cost"]
                            for j in GEOS}, "lightgray"))
    all_series = series + riders
    ng, ns = len(GEOS), len(all_series)
    w = 0.8 / ns
    for si, (lab, rets, ces, col) in enumerate(all_series):
        xs = [g - 0.4 + w * (si + 0.5) for g in range(ng)]
        vs = [100 * rets[j] for j in GEOS]
        ax.bar(xs, vs, w * 0.92, color=col, edgecolor="k", lw=0.6,
               label=lab)
        for gi, x in enumerate(xs):
            j = GEOS[gi]
            txt = f"x{rets[j]:.2f}"
            if ces is not None:
                txt += f"\nCE{ces[j]:+.2f}"
            ax.text(x, vs[gi] + 2.0, txt, ha="center", fontsize=6.8,
                    color=("dimgray" if col == "lightgray" else "k"))
    ax.axhline(100 * HEAL_RET, ls="--", color="seagreen", lw=1.2)
    ax.axhline(100 * PERSIST_RET, ls="--", color="firebrick", lw=1.2)
    ax.axhline(100 * SPARE_RET, ls=":", color="seagreen", lw=0.9)
    ax.axhline(100 * DIE_RET, ls=":", color="gray", lw=0.9)
    ax.text(ng - 0.42, 100 * HEAL_RET + 1.5, "heal bar 0.70", fontsize=7,
            color="seagreen")
    ax.text(ng - 0.42, 100 * PERSIST_RET + 1.5, "persist bar 0.30",
            fontsize=7, color="firebrick")
    ax.set_xticks(range(ng))
    ax.set_xticklabels([f"g{j:+d}  (base expr {base_out[f'g{j}']:.3f})"
                        for j in GEOS], fontsize=9)
    ax.set_ylabel("fact retention vs healthy same-net base (%)")
    ax.set_ylim(0, 118)
    ax.set_title("PROBE A — the reconciliation quartet: mask alone spares, "
                 "poison 0.07 alone kills, JOINT (0.07 UNDER the mask) "
                 "decides the mechanism", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(alpha=0.25, axis="y")

    # ---- panel 2 (bottom-left): fact-vs-CE plane with the two regions
    ax = fig.add_subplot(gs[1, 0])
    xmax = 1.6
    ax.axhspan(100 * HEAL_RET, 112, xmin=0, xmax=HEAL_CE / xmax,
               color="seagreen", alpha=0.15)
    ax.axhspan(-4, 100 * PERSIST_RET, color="firebrick", alpha=0.10)
    ax.axvline(HEAL_CE, ls="--", color="seagreen", lw=1.1)
    ax.axhline(100 * HEAL_RET, ls="--", color="seagreen", lw=1.1)
    ax.axhline(100 * PERSIST_RET, ls="--", color="firebrick", lw=1.1)
    ax.text(0.02, 104, "MASK-HEALS region\n(ret>=0.70 & CE<=+0.35)",
            fontsize=7.5, color="seagreen")
    ax.text(xmax * 0.55, 4, "POISON-PERSISTS region (ret<=0.30)",
            fontsize=7.5, color="firebrick")
    pts = []
    for j in GEOS:
        pts.append(("mask", probeA["cells"][f"mask__g{j}"], "o",
                    "seagreen", f"mask@g{j:+d}"))
        pts.append(("poison07", probeA["cells"][f"poison07__g{j}"], "o",
                    "firebrick", f"poison.07@g{j:+d}"))
        pts.append(("joint07", probeA["cells"][f"joint07__g{j}"], "*",
                    "darkviolet", f"JOINT@g{j:+d}"))
    for tgt in (0.15, 0.0):
        for j in GEOS:
            k = f"joint{tgt:g}__g{j}__RIDER"
            if k in probeA["cells"]:
                pts.append(("rider", probeA["cells"][k], "s", "lightgray",
                            f"joint{tgt:g}@g{j:+d}"))
    for e in probeB["ladder"]:
        if e["target_norm"] == 0.0:
            continue
        ret = e["site_onset"] / st_armb["onset_pz"]
        pts.append(("armb", {"ce_cost": e["ce_r"] - ce_armb, "retention":
                    ret}, "D", "tab:blue",
                    f"armB n={e['target_norm']:g}"))
    seen = set()
    for kind, c, mk, col, lab in pts:
        ax.scatter(c["ce_cost"], 100 * c["retention"], s=(
            260 if kind == "joint07" else 55), marker=mk, color=col,
            edgecolor="k", lw=0.6, zorder=3,
            label=(kind if kind not in seen else None))
        seen.add(kind)
        ax.annotate(lab, (c["ce_cost"], 100 * c["retention"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=6.4)
    ax.set_xlabel("CE cost vs same-net baseline (nats, e065 bank seed 26502)")
    ax.set_ylabel("fact retention (%)")
    ax.set_xlim(-0.08, xmax)
    ax.set_ylim(-4, 112)
    ax.set_title("the two cells in the fact-vs-CE plane (violet star = the "
                 "joint cell; blue = arm_b's ladder)", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="center right")
    ax.grid(alpha=0.25)

    # ---- panel 3 (bottom-right): both nets' ladders (retention + CE)
    ax = fig.add_subplot(gs[1, 1])
    cons_base_g0 = G_CONS_REF_PZ
    cons_base_g12 = G_CONS_REF_GM12
    ns_c = [e["actual_norm"] for e in e150_ladder]
    ax.plot(ns_c, [100 * e["g0_install60"] / cons_base_g0
                   for e in e150_ladder], "o-", color="crimson", lw=1.4,
            label="consolidated @g0 (e150 stored)")
    ax.plot(ns_c, [100 * e["gm12_install60"] / cons_base_g12
                   for e in e150_ladder], "s-", color="darkorange", lw=1.4,
            label="consolidated @g-12 (e150 stored)")
    ns_b = [e["actual_norm"] for e in probeB["ladder"]]
    ax.plot(ns_b, [100 * e["site_onset"] / st_armb["onset_pz"]
                   for e in probeB["ladder"]], "D-", color="tab:blue",
            lw=2.0, ms=7, label="arm_b @site-183 (E159, new)")
    ax.axhline(100 * SPARE_RET, ls=":", color="seagreen", lw=1.0)
    ax.axhline(100 * DIE_RET, ls=":", color="gray", lw=1.0)
    ax.axhspan(100 * DIE_RET, 100 * SPARE_RET, color="gold", alpha=0.08)
    ax.axvspan(0.07, 0.15, color="dimgray", alpha=0.12)
    ax.text(0.11, 8, "consolidated bracket\n(0.07, 0.15)", ha="center",
            fontsize=7, color="dimgray")
    ax.text(0.012, 100 * SPARE_RET + 2, "survive 0.8x", fontsize=7,
            color="seagreen")
    ax.text(0.012, 100 * DIE_RET + 2, "die 0.5x", fontsize=7, color="gray")
    for e in probeB["ladder"]:
        if e["target_norm"] == 0.0:
            continue
        ax.annotate(f"x{e['site_onset'] / st_armb['onset_pz']:.2f}",
                    (e["actual_norm"],
                     100 * e["site_onset"] / st_armb["onset_pz"]),
                    textcoords="offset points", xytext=(4, -11),
                    fontsize=7, color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(ns_c, [e["ce_r"] - G_CONS_REF_CE for e in e150_ladder],
             "v--", color="lightsteelblue", ms=4, alpha=0.9,
             label="CE cost cons (right)")
    ax2.plot(ns_b, [e["ce_r"] - ce_armb for e in probeB["ladder"]],
             "^--", color="lightskyblue", ms=5, alpha=0.9,
             label="CE cost arm_b (right)")
    ax2.set_ylabel("CE cost (nats)", color="steelblue")
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax.set_xlabel("|wpe[0]| (direction kept; each net rescales ITS OWN row)")
    ax.set_ylabel("fact retention (%)")
    ax.set_ylim(-4, 112)
    ax.set_title(f"PROBE B — the site-stored ladder vs the consolidated "
                 f"bracket {e150_bracket['bracket']}", fontsize=9.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.8, loc="center right")

    a = combined["probe_a_outcome"]
    b = combined["probe_b_outcome"]
    fig.suptitle(f"E159 — COUPLED-OR-ORGANISM: A={a} + B={b} — "
                 f"{combined['noun_verdict']}", fontsize=11.0)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
