"""E044 — SCAR TISSUE: does erasure leave re-learning cheaper? (REGISTERED)

The edit-asymmetry law's first refinement (T015 card consequence): after
two-factor erasure of JULIET (e042: D2 row-zero + head L3H5 content-triggered
zero), is RE-learning her cheaper than learning a fresh name — i.e. did
erasure leave a SCAR (residual usage-ability in the body) or was removal
complete?

ARMS (same exposure protocol as e043's best guarded exposure route = Dmix:
16 name windows + 48 anchors (16 paired originals + 32 random corpus windows),
AdamW lr 1e-3 betas (0.9,0.95) wd 0.1 clip 1.0, cosine total 1000 (house
warmup 100), token-weighted union CE; capped at 400 steps, dense evals):
  (a)  RE-INSTALL: the ERASED net (D2 rows + L3H5 patch INSTALLED) re-exposed
       to JULIET contexts = e043's own 60 install host windows with JULIET
       spliced (SPLICE_RNG 24301 — only the name differs from e043's set).
  (b)  FRESH-INSTALL CONTROL: the SAME erased net (patch installed) exposed
       to ZEPHYRA contexts (e043's exact install set, matched volume).
  (c)  REFERENCE: fresh ZEPHYRA install on the UNERASED B net, same protocol
       (rerun for dense steps-to-bar; e043's Dmix quoted as cross-check).
  (a2) [added control] D2-only re-install (no patch): isolates the head
       lesion's share of re-learning cost (the e023-era scar).
  (b2) [added control] fresh ZEPHYRA on the erased net with Z rows pre-
       zeroed: isolates the from-zero address-regrowth cost (the row confound
       that could make (a) look slow even with full scar).

READOUTS: steps-to-bar (arm battery NLL <= 1.0 AND acc >= 0.8; primary =
spliced install battery under the arm's deployed patch state; arm (a) also
the natural ctx-120 JULIET battery, patch-on AND patch-off), corpus CE cost
at bar, and the SCAR probes: row-norm regrowth (|wte_J|, |lm_J| at every
eval; cos(regrown, original-e001 rows) live), the body-route check (e042-style
36-head + 12-block atlas at JULIET positions after re-exposure, including the
D2-CONDITIONED residual-carrier question: does L3H5 re-become the carrier or
does a new route form?), and a re-erasability probe (re-apply D2+patch to the
re-trained net: is the surgical coordinate intact after re-learning?).

REGISTERED PREDICTIONS (before running):
  P1 (complete removal): steps-to-bar(a) within 25% of steps-to-bar(b)
     (0.75 <= ratio <= 1.25) AND row regrowth ~0 toward the original
     direction (cos <= 0.3 on BOTH wte and lm J rows at end).
  P2 (scar): steps-to-bar(a) <= 0.75 x steps-to-bar(b) (>= 25% cheaper/
     faster) AND/OR rows regrow toward the original (cos > 0.3 on either
     row) — usage-ability partially survived erasure.
  P3 (law refined): whichever way, incumbents stay intact — max incumbent
     battery dNLL at (a)'s bar step < +0.10 (non-J names, vs the ERASED
     net's own step-0 baseline; JOHN is J-class, already hit ~1.5 nats by D2
     itself per T011, reported separately; also reported vs unerased B).

DEVIATIONS / SPEC-GAP FILLS (none of the tasked cells dropped):
  1. No e042 two-factor-erased checkpoint exists in runs/checkpoints/
     (checked — only e023_d2_zero_both.pt). The most complete erasure
     available is RECONSTRUCTED: D2 rows re-derived and asserted bit-identical
     to the saved checkpoint, + L3H5 content-triggered patch (full rule:
     fires where consumed context ends with J..JULIE), verified against
     e042's deployed D2+1h cell (gate G6: JULIET 8.715/0.0013, dCE +0.00083)
     and against e042's text-based mask (bit-identical fires).
  2. The head factor stays INSTALLED during re-exposure training and primary
     evals: hooks do not edit weights, so dropping the patch would silently
     restore the pre-lesion head — an unregistered edit. Patch-off readouts
     are recorded as secondary for arm (a). The patch cannot fire at ZEPHYRA
     name queries (rule is JULIET-prefix-triggered), so arm (b)'s handicap is
     confined to natural J-prefix positions in anchors/pre-contexts (fires
     counted and reported; same deployed net for both arms).
  3. JULIET re-exposure contexts = e043's spliced host windows (not natural
     JULIET occurrences) for exact (a)-vs-(b) pairing (identical hosts,
     anchors, draws); the natural-occurrence JULIET battery (e023 ctx-120
     construction, 125 occs) is the secondary "real memory" readout. JULIET
     windows use post-length 120 (256-130-6) vs ZEPHYRA's 119 so ALL windows
     are exactly 256 tokens; name length 6 vs 7 gives 96 vs 112 masked
     targets/step out of ~12.3k tokens (negligible volume asymmetry).
  4. Reference (c) rerun with dense evals (e043's first eval point was s25 —
     too coarse for steps-to-bar). Bar here is [NLL <= 1.0, acc >= 0.8] per
     tasking (e043's Bar-I2 used acc >= 0.90; noted, not silently swapped).
  5. Two extra control arms (a2, b2) beyond the 3 tasked — registered above;
     they decompose the ratio into head-lesion cost and row-regrowth cost.
  6. All five arms share generator seed 24401: identical window/anchor draws
     per step (paired design; only net state and spliced name differ).
  7. Atlases run patch-off (the question is weight-space routing; patch +
     atlas hooks would zero L3H5 by construction in every atlas cell).
  8. No NOTES/THINKING/QUEUE/STATE edits; no git commit (operator instruction).

Run: python lab/e044_scar_tissue.py   (requires runs/checkpoints/{e001.pt,
e023_d2_zero_both.pt}; cross-references runs/{e023,e042,e043}/metrics.json)
E044_SMOKE=1 runs a reduced shakedown (separate log lines, no new dirs).
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import re
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, cfg_dict,
                    cosine_lr, estimate_loss, run_dir, save_json, set_seed)

SMOKE = os.environ.get("E044_SMOKE") == "1"

E001_CKPT = REPO / "runs" / "checkpoints" / "e001.pt"
D2_CKPT = REPO / "runs" / "checkpoints" / "e023_d2_zero_both.pt"
CK = REPO / "runs" / "checkpoints"

SEED = 24400
SPLICE_RNG = 24301            # e043's frozen install-set construction
GEN_EXP = 24401               # shared by ALL arms (paired draws)
HOSTS = ["FLORIZEL", "ELIZABETH"]
PRE = 130
BLOCK = 256
CTX = 120
N_BLOCKS = 60 if SMOKE else 400
CE_SEED = 202

LR = 1e-3
NAME_BS, CORP_BS, MIX_RANDOM = 16, 48, 32
PATCH_LAYER, PATCH_HEAD = 3, 5          # e042's two-factor head: L3H5

BAR_NLL, BAR_ACC = 1.0, 0.8             # tasking bar (see deviation 4)
GUARD_CE = 0.10

BATTERY = ["JULIET", "JOHN", "ROMEO", "GLOUCESTER", "MENENIUS",
           "CORIOLANUS", "ISABELLA", "LUCIO", "PETRUCHIO", "PROSPERO"]
NON_J_INCUMBENTS = ["ROMEO", "GLOUCESTER", "MENENIUS", "CORIOLANUS",
                    "ISABELLA", "LUCIO", "PETRUCHIO"]
NL, NH = 6, 6

# e023/e042/e043 registered references (gates)
E001_VAL_CE = 1.622391
E023_BASE_JULIET = (0.4099, 0.8827)
E023_D2_JULIET = (7.3788, 0.1360)
E023_D2_DCE = 0.000822
E042_D2P_JULIET = (8.715, 0.0013)       # D2+1h, full rule
E042_D2P_DCE = 0.00083
E043_DMX_S25 = {"nll": 0.197, "acc": 0.879, "dce": 0.3097}

if SMOKE:
    EXPOSE_STEPS = 6
    EVAL_STEPS = [1, 2, 4, 6]
    ARMS = ["a", "b"]
else:
    EXPOSE_STEPS = 400
    EVAL_STEPS = [1, 2, 3, 4, 6, 8, 12, 16, 25, 35, 50, 75, 100, 150, 200, 300, 400]
    ARMS = ["a", "a2", "b", "b2", "c"]


# ------------------------------------------------------------------ helpers

def find_occ(text: str, word: str) -> list[int]:
    pat = re.compile(r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])")
    return [m.start() for m in pat.finditer(text)]


def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den > 0 else float("nan")


def fixed_blocks(src, block, n, seed):
    gen = torch.Generator().manual_seed(seed)
    ix = torch.randint(len(src) - block - 1, (n,), generator=gen)
    x = torch.stack([src[i: i + block] for i in ix])
    y = torch.stack([src[i + 1: i + 1 + block] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


def build_name_bat(ids, text, stoi, w, cap):
    """e023 ctx-CTX battery construction (verbatim)."""
    all_occ = find_occ(text, w)
    occs = all_occ[:cap] if cap else all_occ
    keep = [p for p in occs if p >= CTX]
    wid = torch.tensor([stoi[c] for c in w], dtype=torch.long)
    seqs = [torch.cat([ids[p - CTX: p], wid]) for p in keep]
    return {"seq": torch.stack(seqs) if seqs else None, "L": len(w),
            "n": len(keep), "n_all": len(all_occ)}


def prefix_mask_text(text: str, T: int, ks) -> list[bool]:
    """e042's text-based content rule, verbatim (gate cross-check)."""
    m = [False] * T
    for j in range(T):
        for k in ks:
            if j + 1 >= k and text[j + 1 - k: j + 1] == "JULIET"[:k]:
                m[j] = True
                break
    return m


def juliet_fires(x: torch.Tensor, prefs) -> torch.Tensor:
    """(B,T) token ids -> (B,T) bool; fires at query j iff consumed context
    x[..j] (inclusive) ends with a JULIET proper prefix J..JULIE (full rule).
    Id-based twin of e042's prefix_mask."""
    B, T = x.shape
    m = torch.zeros(B, T, dtype=torch.bool, device=x.device)
    for k, p in enumerate(prefs, start=1):
        if T < k:
            break
        w = x.unfold(1, k, 1)
        mk = (w == p.to(x.device)).all(-1)
        m[:, k - 1:] |= mk
    return m


@contextmanager
def pos_lesion(model, specs):
    """e042/e043 position-resolved lesion hook, verbatim."""
    state = {"mask": None}
    handles = []

    def mk_head_pre(head, hd):
        def pre(module, args):
            ms = state["mask"]
            if ms is None:
                return None
            keep = (~ms).to(args[0].dtype).unsqueeze(-1)
            x = args[0].clone()
            x[..., head * hd: (head + 1) * hd] = x[..., head * hd: (head + 1) * hd] * keep
            return (x,)
        return pre

    def mk_sub_fwd():
        def fwd(module, args, out):
            ms = state["mask"]
            if ms is None:
                return None
            return out * (~ms).to(out.dtype).unsqueeze(-1)
        return fwd

    for kind, layer, head in specs:
        block = model.h[layer]
        if kind == "head":
            hd = model.cfg.n_embd // model.cfg.n_head
            handles.append(block.attn.c_proj.register_forward_pre_hook(mk_head_pre(head, hd)))
        elif kind in ("attn", "mlp"):
            mod = block.attn if kind == "attn" else block.mlp
            handles.append(mod.register_forward_hook(mk_sub_fwd()))
        else:
            raise ValueError(kind)
    try:
        yield state
    finally:
        for h in handles:
            h.remove()


def install_patch(net: TinyGPT):
    """Persistent content-triggered L3H5 zero (e042's deployed head factor).
    Returns (state, handle); state['mask'] = (B,T) bool or None (no-op)."""
    state = {"mask": None, "fires": 0}
    hd = net.cfg.n_embd // net.cfg.n_head
    lo, hi = PATCH_HEAD * hd, (PATCH_HEAD + 1) * hd

    def pre(module, args):
        ms = state["mask"]
        if ms is None:
            return None
        state["fires"] += int(ms.sum().item())
        keep = (~ms).to(args[0].dtype).unsqueeze(-1)
        x = args[0].clone()
        x[..., lo:hi] = x[..., lo:hi] * keep
        return (x,)

    h = net.h[PATCH_LAYER].attn.c_proj.register_forward_pre_hook(pre)
    return state, h


@torch.no_grad()
def eval_bat44(net, seq, L, lo, patch_state=None, chunk=128):
    """Battery NLL/acc (e043 eval_seq metric path) + optional patch-on."""
    net.eval()
    x, y = seq[:, :-1], seq[:, 1:]
    nlls, accs = [], []
    try:
        for i in range(0, len(x), chunk):
            xc, yc = x[i: i + chunk].to(DEVICE), y[i: i + chunk].to(DEVICE)
            if patch_state is not None:
                patch_state["mask"] = juliet_fires(xc, PREFS)
            logits, _ = net(xc)
            lg = logits[:, lo: lo + L, :]
            tg = yc[:, lo: lo + L]
            nll = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1),
                                  reduction="none").view(-1, L)
            nlls.append(nll)
            accs.append((lg.argmax(-1) == tg).float())
    finally:
        if patch_state is not None:
            patch_state["mask"] = None
    nll_m, acc_m = torch.cat(nlls), torch.cat(accs)
    return {"n": int(nll_m.shape[0]), "nll": float(nll_m.mean().item()),
            "acc": float(acc_m.mean().item()),
            "per_pos_acc": [float(v) for v in acc_m.mean(0).tolist()],
            "per_pos_nll": [float(v) for v in nll_m.mean(0).tolist()]}


@torch.no_grad()
def ce44(net, x, y, patch_state=None, bs=64):
    net.eval()
    try:
        tot, n = 0.0, 0
        for i in range(0, len(x), bs):
            xb, yb = x[i: i + bs], y[i: i + bs]
            if patch_state is not None:
                patch_state["mask"] = juliet_fires(xb, PREFS)
            _, loss = net(xb, yb)
            tot += float(loss.item()) * len(xb)
            n += len(xb)
    finally:
        if patch_state is not None:
            patch_state["mask"] = None
    return tot / max(n, 1)


def row_probe(net, tid, orig=None):
    wte, lm = net.wte.weight[tid], net.lm_head.weight[tid]
    out = {"wte_norm": float(wte.norm().item()), "lm_norm": float(lm.norm().item())}
    if orig is not None:
        out["cos_wte"] = float(F.cosine_similarity(wte, orig[0], dim=0).item())
        out["cos_lm"] = float(F.cosine_similarity(lm, orig[1], dim=0).item())
    return out


def run_atlas44(net, seq, L, lo):
    """e042-style 36-head + 12-block atlas at the uniform name-position slice.
    Runs patch-off (deviation 7)."""
    b0 = eval_bat44(net, seq, L, lo)
    heads = torch.zeros(NL, NH)
    blocks = {"attn": [0.0] * NL, "mlp": [0.0] * NL}
    for l in range(NL):
        for h in range(NH):
            with pos_lesion(net, [("head", l, h)]) as st:
                m_ = torch.zeros(len(seq), seq.shape[1] - 1, dtype=torch.bool)
                m_[:, lo: lo + L] = True
                st["mask"] = m_.to(DEVICE)
                r = eval_bat44(net, seq, L, lo)
            heads[l, h] = r["nll"] - b0["nll"]
        for kind in ("attn", "mlp"):
            with pos_lesion(net, [(kind, l, None)]) as st:
                m_ = torch.zeros(len(seq), seq.shape[1] - 1, dtype=torch.bool)
                m_[:, lo: lo + L] = True
                st["mask"] = m_.to(DEVICE)
                r = eval_bat44(net, seq, L, lo)
            blocks[kind][l] = r["nll"] - b0["nll"]
    return {"base": {"nll": b0["nll"], "acc": b0["acc"]},
            "heads_dce": heads, "blocks_dce": blocks}


def top_heads(heads, k=3):
    flat = heads.flatten().tolist()
    idx = sorted(range(len(flat)), key=lambda i: -flat[i])[:k]
    return [(i // NH, i % NH) for i in idx]


def exposure44(net, inst_x, inst_mask, anchor, train_ids, *, steps, total, gen,
               patch_state=None, eval_at=(), on_eval=None):
    """e043's exposure (Dmix route) verbatim + patch-mask handling."""
    opt = torch.optim.AdamW(net.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    net.train()
    n_inst, n_anc = inst_x.shape[0], anchor.shape[0]
    for step in range(1, steps + 1):
        f = cosine_lr(step - 1, total)
        for g in opt.param_groups:
            g["lr"] = LR * f
        ix = torch.randint(n_inst, (NAME_BS,), generator=gen)
        aj = torch.randint(n_anc, (CORP_BS - MIX_RANDOM,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (MIX_RANDOM,), generator=gen)
        corp = torch.cat([anchor[aj],
                          torch.stack([train_ids[s: s + BLOCK] for s in rj]).to(DEVICE)], 0)
        nw = inst_x[ix]
        x = torch.cat([nw[:, :-1], corp[:, :-1]], 0)
        y = torch.cat([nw[:, 1:], corp[:, 1:]], 0)
        m = torch.zeros(NAME_BS + CORP_BS, x.shape[1], dtype=torch.bool, device=DEVICE)
        m[:NAME_BS] = inst_mask[ix]
        if patch_state is not None:
            patch_state["mask"] = juliet_fires(x, PREFS)
        logits, _ = net(x)
        if patch_state is not None:
            patch_state["mask"] = None
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                              reduction="none").view(x.shape[0], x.shape[1])
        nm = nll[:NAME_BS][m[:NAME_BS]]
        cm = nll[NAME_BS:]
        loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step in eval_at:
            net.eval()
            on_eval(net, step)
            net.train()
    net.eval()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, bool) or x is None or isinstance(x, (int, str)):
        return x
    if isinstance(x, float):
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        if math.isnan(x):
            return "nan"
        return x
    if isinstance(x, torch.Tensor):
        return jsonable(x.tolist())
    return str(x)


PREFS = None   # set in main (needs stoi)


# ------------------------------------------------------------------ main

def main():
    T0 = time.time()
    stamp = lambda: f"[{time.time() - T0:7.1f}s]"
    log = lambda m: print(f"{stamp()} {m}", flush=True)
    set_seed(SEED)
    rd = run_dir("e044")
    probes = rd / "probes.txt"
    probes.write_text(f"E044 SCAR TISSUE — probes  (smoke={SMOKE}, seed {SEED})\n",
                      encoding="utf-8")

    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    assert corpus.vocab_size == 65
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)
    stoi, itos = corpus.stoi, corpus.itos
    global PREFS
    PREFS = [torch.tensor([stoi[c] for c in "JULIET"[:k]]) for k in range(1, 6)]
    jid, zid = stoi["J"], stoi["Z"]
    qid = stoi.get("Q")
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)

    def load(path):
        m = TinyGPT(cfg).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        m.eval()
        return m

    B = load(E001_CKPT)
    B_sd = {k: v.clone() for k, v in B.state_dict().items()}

    # ---------------------------------------------------------------- G0/G1
    g_est = estimate_loss(B, corpus, "val", n_batches=20)
    G0 = {"val_ce": g_est, "ref": E001_VAL_CE,
          "pass": bool(abs(g_est - E001_VAL_CE) <= 0.03)}
    log(f"G0 val CE {g_est:.4f} vs {E001_VAL_CE} -> {G0['pass']}")

    # ---------------------------------------------------------------- batteries
    log("P0: batteries (e023 ctx-120 cap 125; e043 install-set construction)")
    bat_nat = build_name_bat(train_ids, train_text, stoi, "JULIET", cap=125)
    assert bat_nat["n"] == 125
    inc_bats = {"JULIET": bat_nat}
    for w in BATTERY:
        if w == "JULIET":
            continue
        ids, txt = (val_ids, val_text) if w == "PROSPERO" else (train_ids, train_text)
        inc_bats[w] = build_name_bat(ids, txt, stoi, w, cap=125)

    # e043 install-set construction (frozen)
    host_occ = []
    for host in HOSTS:
        for p in find_occ(train_text, host):
            if p >= 280 and p + len(host) + 119 <= len(train_ids):
                host_occ.append((p, host))
    rng = random.Random(SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    assert len(install_occ) == 60 and len(held_occ) == 30
    anchor_full = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                               for p, _ in install_occ]).to(DEVICE)

    def build_win(p, host, name_ids, L):
        post = BLOCK - PRE - L                      # deviation 3: exact 256
        return torch.cat([train_ids[p - PRE: p], name_ids,
                          train_ids[p + len(host): p + len(host) + post]])

    wins = {}
    for nm in ("JULIET", "ZEPHYRA"):
        nid = torch.tensor([stoi[c] for c in nm], dtype=torch.long)
        L = len(nm)
        wi = torch.stack([build_win(p, h, nid, L) for p, h in install_occ])
        wh = torch.stack([build_win(p, h, nid, L) for p, h in held_occ])
        mk = torch.zeros(60, BLOCK - 1, dtype=torch.bool)
        mk[:, PRE - 1: PRE - 1 + L] = True
        wins[nm] = {"L": L, "inst_x": wi.to(DEVICE), "inst_mask": mk.to(DEVICE),
                    "bat_i": {"seq": wi[:, :PRE + L], "L": L},
                    "bat_h": {"seq": wh[:, :PRE + L], "L": L}}
    log(f"install 60 / held 30 windows (hosts FLORIZEL/ELIZABETH, rng {SPLICE_RNG}); "
        f"JULIET post {BLOCK - PRE - 6}, ZEPHYRA post {BLOCK - PRE - 7}")

    vx, vy = fixed_blocks(val_ids, BLOCK, N_BLOCKS, CE_SEED)
    gx, gy = fixed_blocks(val_ids, BLOCK, 400, CE_SEED)   # e023 convention for gate CEs

    # ---------------------------------------------------------------- erased net + gates
    log("G2/G5/G6: erased-net reconstruction and patch verification")
    er_sd = {k: v.clone() for k, v in B_sd.items()}
    er_sd["wte.weight"][jid] = 0.0
    er_sd["lm_head.weight"][jid] = 0.0
    n_diff, confined = 0, True
    for key in ("wte.weight", "lm_head.weight"):
        d = er_sd[key] != B_sd[key]
        nd = int(d.sum().item())
        n_diff += nd
        rows = torch.nonzero(d)[:, 0].unique()
        if not (len(rows) == 1 and int(rows[0]) == jid and nd == cfg.n_embd):
            confined = False
    others = all(torch.equal(er_sd[k], B_sd[k])
                 for k in er_sd if k not in ("wte.weight", "lm_head.weight"))
    G2 = {"n_elements_changed": n_diff, "expected": 2 * cfg.n_embd,
          "confined_to_J_rows": confined, "others_bit_identical": bool(others),
          "pass": bool(n_diff == 2 * cfg.n_embd and confined and others)}
    if D2_CKPT.exists():
        saved = torch.load(D2_CKPT, map_location=DEVICE, weights_only=True)
        G2["matches_e023_saved_dict"] = all(torch.equal(saved[k], er_sd[k]) for k in er_sd)
        G2["pass"] = G2["pass"] and G2["matches_e023_saved_dict"]
    assert G2["pass"], f"G2 FAILED: {G2}"
    log(f"G2 D2 rows local ({n_diff} elems, saved-match "
        f"{G2.get('matches_e023_saved_dict')}): {G2['pass']}")

    erased = copy.deepcopy(B)
    erased.load_state_dict(er_sd)

    r_d2 = eval_bat44(erased, bat_nat["seq"], 6, CTX - 1)
    ce_base_B = ce44(B, gx, gy)
    ce_d2_nopatch = ce44(erased, gx, gy)
    G5 = {"nll": r_d2["nll"], "acc": r_d2["acc"], "ref": E023_D2_JULIET,
          "dce": ce_d2_nopatch - ce_base_B, "dce_ref": E023_D2_DCE,
          "pass": bool(abs(r_d2["nll"] - E023_D2_JULIET[0]) <= 0.002
                       and abs(r_d2["acc"] - E023_D2_JULIET[1]) <= 0.002
                       and abs((ce_d2_nopatch - ce_base_B) - E023_D2_DCE) <= 0.001)}
    assert G5["pass"], f"G5 FAILED: {G5}"
    log(f"G5 D2 reproduction: JULIET {r_d2['nll']:.3f}/{r_d2['acc']:.3f} "
        f"dCE {ce_d2_nopatch - ce_base_B:+.5f} -> {G5['pass']}")

    # patch machinery: bit-identical fires vs e042's text rule, then the D2+1h cell
    seq_x = bat_nat["seq"][:, :-1]
    my_fires = juliet_fires(seq_x, PREFS)
    txt_fires = torch.tensor([prefix_mask_text(train_text[p - CTX: p] + "JULIET",
                                               seq_x.shape[1], range(1, 6))
                              for p in [q for q in find_occ(train_text, "JULIET")[:125] if q >= CTX]],
                             dtype=torch.bool)
    fires_match = bool(torch.equal(my_fires.cpu(), txt_fires))
    pst, ph = install_patch(erased)
    r_d2p = eval_bat44(erased, bat_nat["seq"], 6, CTX - 1, patch_state=pst)
    ce_d2p = ce44(erased, gx, gy, patch_state=pst)
    G6 = {"fires_bit_identical_to_e042_text_rule": fires_match,
          "juliet_nll": r_d2p["nll"], "juliet_acc": r_d2p["acc"],
          "ref": E042_D2P_JULIET, "dce_content": ce_d2p - ce_base_B,
          "dce_ref": E042_D2P_DCE,
          "pass": bool(fires_match
                       and abs(r_d2p["nll"] - E042_D2P_JULIET[0]) <= 0.02
                       and abs(r_d2p["acc"] - E042_D2P_JULIET[1]) <= 0.005
                       and abs((ce_d2p - ce_base_B) - E042_D2P_DCE) <= 0.0005)}
    assert G6["pass"], f"G6 FAILED: {G6}"
    log(f"G6 two-factor reconstruction: fires bit-identical {fires_match}; "
        f"JULIET {r_d2p['nll']:.3f}/{r_d2p['acc']:.4f} "
        f"dCE {ce_d2p - ce_base_B:+.5f} -> {G6['pass']}")
    ph.remove()   # arm a gets its own patch on its own net copy

    # step-0 baselines
    base_ro = {"B_juliet_nat": eval_bat44(B, bat_nat["seq"], 6, CTX - 1),
               "B_ce": ce_base_B}
    log(f"baselines: B JULIET {base_ro['B_juliet_nat']['nll']:.3f}/"
        f"{base_ro['B_juliet_nat']['acc']:.3f} | D2 {r_d2['nll']:.3f}/{r_d2['acc']:.3f} | "
        f"D2+patch {r_d2p['nll']:.3f}/{r_d2p['acc']:.4f} | CE {ce_base_B:.5f}")

    # ---------------------------------------------------------------- arms
    orig_j_rows = (B_sd["wte.weight"][jid].clone(), B_sd["lm_head.weight"][jid].clone())
    orig_z_rows = (B_sd["wte.weight"][zid].clone(), B_sd["lm_head.weight"][zid].clone())
    orig_q_rows = (B_sd["wte.weight"][qid].clone(), B_sd["lm_head.weight"][qid].clone()) if qid is not None else None

    ARM_DEF = {
        "a":  {"name": "JULIET", "patch": True,  "zero_z": False, "base": "erased"},
        "a2": {"name": "JULIET", "patch": False, "zero_z": False, "base": "erased"},
        "b":  {"name": "ZEPHYRA", "patch": True,  "zero_z": False, "base": "erased"},
        "b2": {"name": "ZEPHYRA", "patch": True,  "zero_z": True,  "base": "erased"},
        "c":  {"name": "ZEPHYRA", "patch": False, "zero_z": False, "base": "B"},
    }
    results = {}
    final_nets = {}

    for arm in ARMS:
        d = ARM_DEF[arm]
        nm, L = d["name"], len(d["name"])
        net = copy.deepcopy(erased if d["base"] == "erased" else B)
        if d["zero_z"]:
            with torch.no_grad():
                net.wte.weight[zid] = 0.0
                net.lm_head.weight[zid] = 0.0
        pst = None
        if d["patch"]:
            pst, ph_ = install_patch(net)
        is_j = nm == "JULIET"
        tid = jid if is_j else zid
        orig = orig_j_rows if is_j else orig_z_rows
        w = wins[nm]
        traj = []
        inc_list = [x for x in BATTERY if x != nm]

        def probe(net_, step, arm=arm, nm=nm, L=L, is_j=is_j, pst=pst,
                  tid=tid, orig=orig, w=w, inc_list=inc_list, traj=traj):
            rec = {"step": step, "arm": arm,
                   "spliced": eval_bat44(net_, w["bat_i"]["seq"], L, PRE - 1, pst),
                   "held": eval_bat44(net_, w["bat_h"]["seq"], L, PRE - 1, pst)}
            if is_j:
                rec["nat_patchoff"] = eval_bat44(net_, bat_nat["seq"], 6, CTX - 1, None)
                if pst is not None:
                    rec["nat_patchon"] = eval_bat44(net_, bat_nat["seq"], 6, CTX - 1, pst)
                    rec["spliced_patchoff"] = eval_bat44(net_, w["bat_i"]["seq"], L, PRE - 1, None)
            rec["incumbents"] = {x: {"nll": (r := eval_bat44(net_, inc_bats[x]["seq"],
                                                            inc_bats[x]["L"], CTX - 1, pst))["nll"],
                                     "acc": r["acc"]} for x in inc_list}
            rec["ce"] = ce44(net_, vx, vy, pst)
            rec["rows"] = row_probe(net_, tid, orig)
            traj.append(rec)
            sp = rec["spliced"]
            nat = rec.get("nat_patchon") or rec.get("nat_patchoff")
            log(f"  [{arm}] s{step:3d} spliced {sp['nll']:6.3f}/{sp['acc']:.3f}"
                + (f" nat {nat['nll']:6.3f}/{nat['acc']:.3f}" if nat else "")
                + f" ce {rec['ce']:.4f} |wte| {rec['rows']['wte_norm']:.3f}"
                + (f" cos {rec['rows'].get('cos_wte', float('nan')):+.3f}" if is_j else ""))
            return rec

        log(f"arm {arm}: {nm} on {'erased' if d['base'] == 'erased' else 'B'}"
            f"{' + patch' if d['patch'] else ''}{' + Zrows0' if d['zero_z'] else ''}, "
            f"{EXPOSE_STEPS} steps")
        probe(net, 0)
        gen = torch.Generator().manual_seed(GEN_EXP)
        exposure44(net, w["inst_x"], w["inst_mask"], anchor_full, train_ids,
                   steps=EXPOSE_STEPS, total=1000, gen=gen, patch_state=pst,
                   eval_at=set(EVAL_STEPS), on_eval=probe)
        torch.save({"model": net.state_dict(),
                    "meta": {"arm": arm, "name": nm, "steps": EXPOSE_STEPS,
                             "seed": SEED, "gen": GEN_EXP}},
                   CK / f"e044_{arm}_reinstall.pt" if is_j else CK / f"e044_{arm}_{nm.lower()}.pt")
        results[arm] = {"def": d, "traj": traj,
                        "patch_fires_total": pst["fires"] if pst else 0}
        final_nets[arm] = net
        if pst is not None:
            ph_.remove()

    # steps-to-bar + CE at bar
    def bar_scan(traj, key):
        for rec in traj:
            r = rec[key]
            if r["nll"] <= BAR_NLL and r["acc"] >= BAR_ACC:
                return rec["step"], r, rec
        return None, None, None

    bars = {}
    for arm in ARMS:
        s_sp, r_sp, rec_sp = bar_scan(results[arm]["traj"], "spliced")
        ent = {"spliced": {"step": s_sp, "nll": r_sp["nll"] if r_sp else None,
                           "acc": r_sp["acc"] if r_sp else None,
                           "dce_at_bar": (rec_sp["ce"] - results[arm]["traj"][0]["ce"])
                           if rec_sp else None}}
        for key in ("nat_patchon", "nat_patchoff", "spliced_patchoff"):
            if key in results[arm]["traj"][0] or (results[arm]["traj"] and key in results[arm]["traj"][-1]):
                s_, r_, rec_ = bar_scan(results[arm]["traj"], key)
                ent[key] = {"step": s_, "nll": r_["nll"] if r_ else None,
                            "acc": r_["acc"] if r_ else None,
                            "dce_at_bar": (rec_["ce"] - results[arm]["traj"][0]["ce"]) if rec_ else None}
        bars[arm] = ent
        log(f"arm {arm}: steps-to-bar (spliced) {s_sp}"
            + (f" at dCE {ent['spliced']['dce_at_bar']:+.4f}" if rec_sp else "")
            + "".join(f" | {k} {bars[arm][k]['step']}" for k in ("nat_patchon", "nat_patchoff")
                      if k in ent))

    # ---------------------------------------------------------------- scar probes
    log("scar probes: rows + routes + re-erasability")
    a_final = results["a"]["traj"][-1]
    scar_rows = {
        "orig_j_norms": {"wte": float(orig_j_rows[0].norm()), "lm": float(orig_j_rows[1].norm())},
        "orig_z_norms": {"wte": float(orig_z_rows[0].norm()), "lm": float(orig_z_rows[1].norm())},
        "a_final": a_final["rows"],
        "a2_final": results["a2"]["traj"][-1]["rows"] if "a2" in results else None,
        "b_final": results["b"]["traj"][-1]["rows"],
        "b2_final": results["b2"]["traj"][-1]["rows"] if "b2" in results else None,
        "c_final": results["c"]["traj"][-1]["rows"] if "c" in results else None,
        "a_row_trajectory": [(r["step"], r["rows"]) for r in results["a"]["traj"]],
        "q_row_drift_final": row_probe(final_nets["a"], qid) if qid is not None else None,
    }

    # atlases (patch-off)
    atlas = {}
    d2ref = copy.deepcopy(erased)
    atlas["d2_orig_d2cond"] = run_atlas44(d2ref, bat_nat["seq"], 6, CTX - 1)
    dh = atlas["d2_orig_d2cond"]["heads_dce"]
    G7a = {"top1": f"L{int(dh.argmax())//NH}H{int(dh.argmax())%NH}",
           "top1_dce": float(dh.max()),
           "pass": bool((int(dh.argmax())//NH, int(dh.argmax())%NH) == (PATCH_LAYER, PATCH_HEAD)
                        and 0.7 <= float(dh.max()) <= 1.05)}
    log(f"G7a recomputed D2-conditioned atlas top1 {G7a['top1']} "
        f"{G7a['top1_dce']:+.3f} -> {G7a['pass']}")

    # e042 stored atlas cross-check
    e042m = json.load(open(REPO / "runs" / "e042" / "metrics.json"))
    stored = torch.tensor(e042m["d2_atlas"]["heads_dce"])
    G7b = {"spearman_vs_e042_stored": spearman(dh.flatten().tolist(), stored.flatten().tolist()),
           "pass": True}
    G7b["pass"] = bool(G7b["spearman_vs_e042_stored"] >= 0.95)
    log(f"G7b vs e042 stored D2 atlas: spearman {G7b['spearman_vs_e042_stored']:.4f} "
        f"-> {G7b['pass']}")

    anet = final_nets["a"]
    saved_rows = (anet.wte.weight[jid].clone(), anet.lm_head.weight[jid].clone())
    with torch.no_grad():
        anet.wte.weight[jid] = 0.0
        anet.lm_head.weight[jid] = 0.0
    atlas["a_final_d2cond"] = run_atlas44(anet, bat_nat["seq"], 6, CTX - 1)
    r_reerase = eval_bat44(anet, bat_nat["seq"], 6, CTX - 1, patch_state=None)
    pst2, ph2 = install_patch(anet)
    ce_reerase = ce44(anet, vx, vy, patch_state=pst2)
    r_reerase_p = eval_bat44(anet, bat_nat["seq"], 6, CTX - 1, patch_state=pst2)
    ph2.remove()
    with torch.no_grad():
        anet.wte.weight[jid] = saved_rows[0]
        anet.lm_head.weight[jid] = saved_rows[1]
    reerase = {"a_final": {"d2_rows_only": {"nll": r_reerase["nll"], "acc": r_reerase["acc"]},
                           "d2_plus_patch": {"nll": r_reerase_p["nll"], "acc": r_reerase_p["acc"]},
                           "dce": ce_reerase - results["a"]["traj"][0]["ce"],
                           "vs_e042": E042_D2P_JULIET}}
    log(f"re-erasability (a-final): D2+patch -> JULIET {r_reerase_p['nll']:.3f}/"
        f"{r_reerase_p['acc']:.4f} at dCE {ce_reerase - results['a']['traj'][0]['ce']:+.5f}")

    atlas["base_B_nat"] = run_atlas44(B, bat_nat["seq"], 6, CTX - 1)
    bh = atlas["base_B_nat"]["heads_dce"]
    G7c = {"top1": f"L{int(bh.argmax())//NH}H{int(bh.argmax())%NH}",
           "pass": bool((int(bh.argmax())//NH, int(bh.argmax())%NH) == (0, 3))}
    log(f"G7c base natural JULIET atlas top1 {G7c['top1']} -> {G7c['pass']}")

    atlas["a_final_nat"] = run_atlas44(anet, bat_nat["seq"], 6, CTX - 1)
    if "b" in final_nets:
        atlas["b_final_zephyra"] = run_atlas44(final_nets["b"],
                                               wins["ZEPHYRA"]["bat_i"]["seq"], 7, PRE - 1)

    ah = atlas["a_final_d2cond"]["heads_dce"]
    route = {
        "d2cond": {"orig_top1": G7a["top1"], "orig_top1_dce": G7a["top1_dce"],
                   "a_final_top1": f"L{int(ah.argmax())//NH}H{int(ah.argmax())%NH}",
                   "a_final_top1_dce": float(ah.max()),
                   "L3H5_dce_a_final": float(ah[PATCH_LAYER, PATCH_HEAD]),
                   "spearman_orig_vs_a_final": spearman(dh.flatten().tolist(), ah.flatten().tolist()),
                   "l3h5_rebecomes_carrier": bool((int(ah.argmax())//NH, int(ah.argmax())%NH)
                                                  == (PATCH_LAYER, PATCH_HEAD)),
                   "orig_top3": [f"L{l}H{h}" for l, h in top_heads(dh, 3)],
                   "a_final_top3": [f"L{l}H{h}" for l, h in top_heads(ah, 3)]},
        "nat": {"base_top1": G7c["top1"], "base_top1_dce": float(bh.max()),
                "a_final_top1": f"L{int(atlas['a_final_nat']['heads_dce'].argmax())//NH}"
                                f"H{int(atlas['a_final_nat']['heads_dce'].argmax())%NH}",
                "spearman_base_vs_a_final": spearman(
                    bh.flatten().tolist(),
                    atlas["a_final_nat"]["heads_dce"].flatten().tolist())},
    }
    if "b_final_zephyra" in atlas:
        zh = atlas["b_final_zephyra"]["heads_dce"]
        tb = max([("attn", i) for i in range(NL)] + [("mlp", i) for i in range(NL)],
                 key=lambda s: atlas["b_final_zephyra"]["blocks_dce"][s[0]][s[1]])
        route["zephyra_b"] = {"top1": f"L{int(zh.argmax())//NH}H{int(zh.argmax())%NH}",
                              "top1_dce": float(zh.max()),
                              "top_block": f"L{tb[1]}-{tb[0]}",
                              "top_block_dce": atlas["b_final_zephyra"]["blocks_dce"][tb[0]][tb[1]],
                              "no_dominant_head": bool(float(zh.max()) < 0.5 * max(
                                  atlas["b_final_zephyra"]["blocks_dce"][tb[0]][tb[1]], 1e-9))}
    log(f"route: D2-cond orig {route['d2cond']['orig_top1']} {route['d2cond']['orig_top1_dce']:+.2f} "
        f"-> a-final {route['d2cond']['a_final_top1']} {route['d2cond']['a_final_top1_dce']:+.2f} "
        f"(L3H5 {route['d2cond']['L3H5_dce_a_final']:+.2f}, "
        f"rho {route['d2cond']['spearman_orig_vs_a_final']:.2f}); "
        f"nat base {route['nat']['base_top1']} -> a-final {route['nat']['a_final_top1']}")

    # ---------------------------------------------------------------- verdicts
    sa = bars["a"]["spliced"]["step"]
    sb = bars["b"]["spliced"]["step"]
    ratio = (sa / sb) if (sa is not None and sb is not None) else None
    cos_final = max(abs(a_final["rows"].get("cos_wte", 0.0)),
                    abs(a_final["rows"].get("cos_lm", 0.0)))
    sa2 = bars.get("a2", {}).get("spliced", {}).get("step") if "a2" in bars else None
    sb2 = bars.get("b2", {}).get("spliced", {}).get("step") if "b2" in bars else None

    # P3: incumbent collateral at (a)'s bar step vs erased-net step-0
    step0_a = results["a"]["traj"][0]
    bar_rec_a = (next((r for r in results["a"]["traj"] if r["step"] == sa), None)
                 if sa is not None else None)
    inc_at_bar = {}
    if bar_rec_a:
        for w_ in NON_J_INCUMBENTS:
            inc_at_bar[w_] = bar_rec_a["incumbents"][w_]["nll"] - step0_a["incumbents"][w_]["nll"]
        inc_at_bar["JOHN_(J-class, excl)"] = (bar_rec_a["incumbents"]["JOHN"]["nll"]
                                              - step0_a["incumbents"]["JOHN"]["nll"])
    inc_max = max((v for k, v in inc_at_bar.items() if not k.startswith("JOHN")),
                  default=None)
    p1 = {"ratio": ratio, "within_25pct": bool(ratio is not None and 0.75 <= ratio <= 1.25),
          "cos_max_final": cos_final, "cos_le_0.3": bool(cos_final <= 0.3),
          "confirmed": bool(ratio is not None and 0.75 <= ratio <= 1.25 and cos_final <= 0.3)}
    p2 = {"ratio": ratio, "steps_cheaper_25pct": bool(ratio is not None and ratio <= 0.75),
          "cos_max_final": cos_final, "rows_regrow_cos_gt_0.3": bool(cos_final > 0.3),
          "confirmed": bool((ratio is not None and ratio <= 0.75) or cos_final > 0.3)}
    p3 = {"incumbent_dnll_at_bar": inc_at_bar, "incumbent_abs_max_nonJ": inc_max,
          "bar": 0.10, "confirmed": bool(inc_max is not None and inc_max < 0.10)}
    decomp = {"head_lesion_cost_ratio": (sa / sa2) if (sa is not None and sa2 is not None) else None,
              "row_regrow_cost_ratio_b_over_b2": (sb2 / sb) if (sb2 is not None and sb is not None) else None,
              "steps": {"a": sa, "a2": sa2, "b": sb, "b2": sb2}}
    log("=" * 70)
    log(f"P1 (complete removal) confirmed: {p1['confirmed']} "
        f"(ratio {ratio}, cos {cos_final:+.3f})")
    log(f"P2 (scar) confirmed: {p2['confirmed']} "
        f"(cheaper {p2['steps_cheaper_25pct']}, rows {p2['rows_regrow_cos_gt_0.3']})")
    log(f"P3 (incumbents intact) confirmed: {p3['confirmed']} (max non-J dNLL {inc_max})")
    log(f"decomposition: {decomp}")

    with probes.open("a", encoding="utf-8") as f:
        f.write(f"\nVERDICTS: P1 {p1['confirmed']} | P2 {p2['confirmed']} | "
                f"P3 {p3['confirmed']}\nratio(a/b) {ratio} | cos_max {cos_final:+.3f} | "
                f"steps a/a2/b/b2 = {sa}/{sa2}/{sb}/{sb2}\n")

    # ---------------------------------------------------------------- outputs
    metrics = {
        "experiment": "e044_scar_tissue",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED, "smoke": SMOKE,
        "deviations": [l.strip() for l in __doc__.split("DEVIATIONS")[1].split("Run:")[0]
                       .splitlines() if l.strip()],
        "bars": {"nll": BAR_NLL, "acc": BAR_ACC, "guard_ce": GUARD_CE},
        "gates": {"G0": G0, "G2": G2, "G5": G5, "G6": G6,
                  "G7a_d2cond_l3h5": G7a, "G7b_vs_e042_stored": G7b, "G7c_base_l0h3": G7c},
        "baselines": {"B_juliet_nat": base_ro["B_juliet_nat"],
                      "D2_juliet": r_d2, "D2patch_juliet": {"nll": r_d2p["nll"], "acc": r_d2p["acc"]},
                      "ce_B": ce_base_B, "ce_D2": ce_d2_nopatch, "ce_D2patch": ce_d2p,
                      "e043_dmix_s25_crosscheck": E043_DMX_S25},
        "arms": {arm: {"def": results[arm]["def"],
                       "patch_fires_total": results[arm]["patch_fires_total"],
                       "traj": results[arm]["traj"]} for arm in ARMS},
        "steps_to_bar": bars,
        "scar_rows": scar_rows,
        "route_check": route,
        "re_erasability": reerase,
        "verdicts": {"P1": p1, "P2": p2, "P3": p3, "decomposition": decomp},
        "atlases": {k: {"heads_dce": v["heads_dce"], "blocks_dce": v["blocks_dce"],
                        "base": v["base"]} for k, v in atlas.items()},
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": cfg_dict(cfg),
    }
    save_json(rd / "metrics.json", jsonable(metrics))

    # ---------------------------------------------------------------- plots
    if not SMOKE:
        fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
        ax = axes[0]
        style = {"a": ("crimson", "o"), "a2": ("salmon", "v"), "b": ("royalblue", "s"),
                 "b2": ("cornflowerblue", "P"), "c": ("seagreen", "D")}
        for arm in ARMS:
            col, mk = style[arm]
            tr = results[arm]["traj"]
            ax.plot([r["step"] for r in tr], [r["spliced"]["nll"] for r in tr],
                    marker=mk, ms=3.5, color=col, label=f"{arm} ({results[arm]['def']['name']})")
            for key, ls in (("nat_patchon", ":"), ("nat_patchoff", "--")):
                if any(key in r for r in tr):
                    ax.plot([r["step"] for r in tr if key in r],
                            [r[key]["nll"] for r in tr if key in r],
                            ls, color=col, lw=1.2, alpha=0.85,
                            label=f"{arm} nat {'patch-on' if key.endswith('on') else 'patch-off'}")
        ax.axhline(BAR_NLL, color="purple", ls=":", lw=1, label="bar NLL 1.0")
        for arm in ("a", "b"):
            s_ = bars[arm]["spliced"]["step"]
            if s_ is not None:
                ax.axvline(s_, color=style[arm][0], lw=0.7, alpha=0.5)
        ax.set_xlabel("exposure step"); ax.set_ylabel("name NLL (arm battery)")
        ax.set_title("steps-to-bar: re-install (a) vs fresh (b) vs reference (c)")
        ax.legend(fontsize=6.5)
        ax = axes[1]
        for arm in ARMS:
            col, mk = style[arm]
            tr = results[arm]["traj"]
            ax.plot([r["step"] for r in tr], [r["ce"] - tr[0]["ce"] for r in tr],
                    marker=mk, ms=3.5, color=col, label=arm)
        ax.axhline(GUARD_CE, color="gray", ls="--", lw=1, label="+0.10 guard")
        ax.set_xlabel("exposure step"); ax.set_ylabel("dCE val (vs arm step-0)")
        ax.set_title("corpus CE cost trajectories"); ax.legend(fontsize=7)
        ax = axes[2]
        for arm in ARMS:
            col, mk = style[arm]
            tr = results[arm]["traj"]
            ax.plot([r["step"] for r in tr], [r["rows"]["wte_norm"] for r in tr],
                    marker=mk, ms=3.5, color=col, label=f"{arm} wte")
            ax.plot([r["step"] for r in tr], [r["rows"]["lm_norm"] for r in tr],
                    marker=mk, ms=2.5, color=col, alpha=0.45, ls="--")
        ax.axhline(scar_rows["orig_j_norms"]["wte"], color="crimson", ls=":", lw=1,
                   label="orig |wte_J|")
        ax.axhline(scar_rows["orig_j_norms"]["lm"], color="crimson", ls="-.", lw=1,
                   label="orig |lm_J|")
        ax.set_xlabel("exposure step"); ax.set_ylabel("target-row norm (solid wte / dashed lm)")
        ax.set_title("row regrowth (J for a/a2, Z for b/b2/c)"); ax.legend(fontsize=6.5)
        fig.suptitle("E044 — scar tissue: is re-learning cheaper than fresh learning?")
        fig.tight_layout()
        fig.savefig(rd / "scar_trajectories.png", dpi=130)
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(17, 9))

        def heat(ax, M, title, cmap="inferno"):
            im = ax.imshow(M.tolist(), cmap=cmap, aspect="auto")
            for l in range(NL):
                for h in range(NH):
                    ax.text(h, l, f"{M[l, h]:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if M[l, h] < 0.55 * float(M.max()) else "black")
            ax.set_xticks(range(NH)); ax.set_xticklabels([f"H{h}" for h in range(NH)], fontsize=7)
            ax.set_yticks(range(NL)); ax.set_yticklabels([f"L{l}" for l in range(NL)], fontsize=7)
            ax.set_title(title, fontsize=9)
            return im

        heat(axes[0, 0], dh, f"D2-conditioned residual atlas — original D2 net\n"
             f"top1 {route['d2cond']['orig_top1']} {route['d2cond']['orig_top1_dce']:+.2f}")
        heat(axes[0, 1], ah, f"D2-conditioned residual atlas — after re-install (a)\n"
             f"top1 {route['d2cond']['a_final_top1']} {route['d2cond']['a_final_top1_dce']:+.2f} "
             f"(L3H5 {route['d2cond']['L3H5_dce_a_final']:+.2f})")
        heat(axes[0, 2], atlas["a_final_nat"]["heads_dce"],
             f"natural JULIET atlas — after re-install (a)\n"
             f"top1 {route['nat']['a_final_top1']}, rho vs base "
             f"{route['nat']['spearman_base_vs_a_final']:.2f}")
        heat(axes[1, 0], bh, f"natural JULIET atlas — base B\n"
             f"top1 {route['nat']['base_top1']} {route['nat']['base_top1_dce']:+.2f}")
        if "b_final_zephyra" in atlas:
            heat(axes[1, 1], atlas["b_final_zephyra"]["heads_dce"],
                 f"ZEPHYRA atlas — fresh install (b)\n"
                 f"top1 {route['zephyra_b']['top1']} {route['zephyra_b']['top1_dce']:+.2f}, "
                 f"top block {route['zephyra_b']['top_block']}")
        ax = axes[1, 2]
        names = [k for k in inc_at_bar if not k.startswith("JOHN")]
        vals = [inc_at_bar[k] for k in names]
        ax.barh(range(len(names)), vals, 0.6, color="steelblue")
        ax.axvline(0.10, color="crimson", ls="--", lw=1, label="P3 bar +0.10")
        ax.axvline(0.0, color="k", lw=0.5)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("dNLL at (a)'s bar step (vs erased step-0)")
        ax.set_title(f"P3 incumbents (max {inc_max})"); ax.legend(fontsize=7)
        fig.suptitle("E044 route check — did L3H5 re-become the residual carrier?")
        fig.tight_layout()
        fig.savefig(rd / "route_atlas.png", dpi=130)
        plt.close(fig)

    log(f"outputs: {rd}")
    log(f"total {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
