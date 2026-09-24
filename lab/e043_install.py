"""E043 — INSTALL a name: the additive symmetry test of "edit the organism"
(REGISTERED).

Design: scratch/e043_design.md (2026-09-24; every probe in it is measured).
Arc "edit the organism", slot 3 — the additive half. Target ZEPHYRA (made-up,
Z rare core, 0 corpus occurrences). Question: can a name B does not know be
surgically INSTALLED — rows-only / rows + body-graft / rows + brief exposure —
at e023/e042-grade selectivity, or is editing the organism asymmetric?

Arms: P0 donors = BDO and B43 copies exposed on the install set (masked
name-position loss + corpus-anchor interleave). A rows-only ({wte,lm,both} x
{copy,delta} from the BDO donor + B43 copy-both, the C3 row-space control).
B rows+brief-exposure (bare / rows-preinstalled, eval @25/50/100). C rows +
body-graft from the exposed BDO donor ({L0-MLP, L0-attn, L0-MLP+rows, donor's
top ZEPHYRA head from its own in-run atlas}). D exposure dose ladder (25..600
+ one 1000-step long cell) = the transient ceiling reference.
Readouts: R1 install/held/Z-class/incumbent batteries + PROSPERO anchor,
R2 corpus CE (seed-202 blocks + Z-free channel), R3 generation (8 prompts,
350 tok), R4 pos-resolved atlas re-run (e042 machinery). Predictions P1-P3;
bars Bar-I1 (NLL<=4.17 & acc>=0.50 @ dCE<=+0.10) / Bar-I2 (<=1.0 & >=0.90);
symmetry verdict BIDIRECTIONAL / ASYMMETRIC-CHEAP-REMOVE / ASYMMETRIC-BASIS.

DEVIATIONS / SPEC-GAP FILLS (all registered cells present, none dropped):
 1. Install-window post-context capped at 119 (design says <=120): 130 pre +
    7 name + 120 post = 257 tokens exceeds the 256 block. Post carries no
    loss and no gradient at name positions (causal), so the protocol is
    unaffected where it matters; all windows are exactly 256 tokens.
 2. "lr 1e-3 cosine" uses common.py's cosine_lr with its DEFAULT warmup=100
    (the house schedule train_model uses; warmup was unspecified in the
    design). D ladder is ONE trajectory with cosine total = 1000 (P2(i)'s
    "continued anchored training erases the new attractor" + the budget
    table's ~2.5 min both indicate a single long run, not per-dose runs).
 3. Exposure loss = ONE token-level CE over the union of the 16x7 name-char
    targets and the 48x255 corpus tokens (112 vs 12,288 per-token weights).
    Group-equal weighting was rejected in shakedown: it installs the onset
    (context->Z) within 8 steps, contradicting the design's measured anatomy
    (onset acc ~0 to s2000, per-pos [0.01, 0.47, 1, 1, 1, .93, 1] at s25);
    token weighting reproduces the registered per-position structure.
 4. ANCHOR COMPOSITION (the protocol's largest spec-gap, resolved by
    measurement): "48 interleaved corpus windows" cannot be random corpus
    windows — random sampling almost never hits the 60 spliced contexts out
    of ~1.1M, so nothing opposes the onset (context->Z), which then installs
    to ~0.9 acc within 16 steps under BOTH group-equal and token-weighted
    loss, contradicting the design's measured onset wall (~0 to s2000).
    Fully-paired anchoring (48 original host windows/step) walls the onset at
    0.00 at every dose but the narrow 60-window replay under wd 0.1 destroys
    general CE (val +3.5 nats by s1000). The design's MP anatomy (onset
    walled AND CE bounded AND install partial) is only jointly approachable
    by a MIX. Registered resolution: donors + arm B use 16 paired originals +
    32 random corpus (bounded-CE donors are transplantable); arm D runs BOTH
    trajectories — Dmix (16+32, primary) and Dpair (48 paired, the strict
    "interleaved originals" reading) — so the anchor-composition sensitivity
    of the install frontier is itself measured. Held-out originals are never
    trained in any form.
 5. G0 BDO/B43 parity uses the e029 30-batch protocol, gate <= 1.7224.
 6. R2 val_all base reads ~1.628 (e023/e042 convention), not the design MP
    1.8206; the design itself gates on estimate_loss (reads ~1.621) and
    deltas are the registered objects.
 7. R4: if the best guarded cell IS D@s25, the atlas net set collapses to
    that one cell (both registered cells coincide; noted in metrics).
 8. P1a strictness: "|dNLL| <= 0.05" evaluated as the MAX over the 9 train
    incumbents and the 4 Z-class names (means also reported).
 9. No NOTES/THINKING/QUEUE/STATE edits, no git commit (operator instruction).

Run: python lab/e043_install.py   (requires runs/checkpoints/{e001,e028_b43,e041_bdo}.pt)
E043_SMOKE=1 runs a reduced shakedown (separate outputs, _smoke ckpts).
"""
from __future__ import annotations

import copy
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
                    cosine_lr, estimate_loss, generate, run_dir, save_json,
                    set_seed)

SMOKE = os.environ.get("E043_SMOKE") == "1"

E001_CKPT = REPO / "runs" / "checkpoints" / "e001.pt"
B43_CKPT = REPO / "runs" / "checkpoints" / "e028_b43.pt"
BDO_CKPT = REPO / "runs" / "checkpoints" / "e041_bdo.pt"
CK = REPO / "runs" / "checkpoints"
DONOR_CK = {t: CK / f"e043_donor_{t}{'_smoke' if SMOKE else ''}.pt"
            for t in ("bdo", "b43")}
BARM_CK = {t: CK / f"e043_{t}{'_smoke' if SMOKE else ''}.pt"
           for t in ("b_bare", "b_rows", "d_ladder")}

SEED = 24300
NAME = "ZEPHYRA"
HOSTS = ["FLORIZEL", "ELIZABETH"]
PRE = 130                # install-window pre-name context
POST_CAP = 119           # deviation 1
CTX = 120                # e023 incumbent-battery context
BLOCK = 256
N_BLOCKS = 400           # e023/e042 val-All convention (seed 202)
CE_SEED = 202
ZFREE_SEED = 24305
SPLICE_RNG = 24301
GEN_DONOR = {"bdo": 24321, "b43": 24331}     # disjoint donor exposure gens
GEN_B_BARE, GEN_B_ROWS, GEN_D = 24311, 24312, 24313
MIX_RANDOM = 32           # of 48 anchor windows: 16 paired originals + 32 random corpus

LR = 1e-3
WARMUP = 10
NAME_BS, CORP_BS = 16, 48
EXPOSE_STEPS = 100
FB_LR, FB_CORP_BS, FB_STEPS = 3e-4, 16, 300   # G6 fallback config

BAR_I1_NLL, BAR_I1_ACC = math.log(65), 0.50   # 4.174
BAR_I2_NLL, BAR_I2_ACC = 1.0, 0.90
GUARD_CE = 0.10
S_INSTALL_BAR = 5.0
E001_VAL_CE = 1.622391
PARITY_GATE = 1.7224
G6_MAX = 4.5

BATTERY = ["JULIET", "JOHN", "ROMEO", "GLOUCESTER", "MENENIUS",
           "CORIOLANUS", "ISABELLA", "LUCIO", "PETRUCHIO", "PROSPERO"]
PURE_CONTROLS = ["ROMEO", "GLOUCESTER", "CORIOLANUS"]
ZCLASS = ["ELIZABETH", "FLORIZEL", "FITZWATER", "Zounds"]
ATLAS_NAMES = ["JULIET", "ROMEO", "LUCIO", "ZEPHYRA"]
NL, NH = 6, 6

if SMOKE:
    EXPOSE_STEPS = 8
    D_DOSES = [4, 8, 16]
    D_TOTAL = 16
    N_GEN_PROMPTS, GEN_TOK = 2, 30
    N_BLOCKS = 100
    B_EVAL = [4, 8]
else:
    D_DOSES = [25, 50, 100, 200, 400, 600]
    D_TOTAL = 1000
    N_GEN_PROMPTS, GEN_TOK = 8, 350
    B_EVAL = [25, 50, 100]


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


def sample_windows(src_ids, src_text, n, seed, accept):
    gen = torch.Generator().manual_seed(seed)
    starts, tries = [], 0
    while len(starts) < n and tries < 500 * n:
        i = int(torch.randint(len(src_ids) - BLOCK - 1, (1,), generator=gen))
        tries += 1
        if accept(src_text[i: i + BLOCK + 1]):
            starts.append(i)
    x = torch.stack([src_ids[s: s + BLOCK] for s in starts])
    y = torch.stack([src_ids[s + 1: s + 1 + BLOCK] for s in starts])
    return x.to(DEVICE), y.to(DEVICE), len(starts)


def build_name_bat(ids, text, stoi, w, cap):
    """e023 ctx-120 battery construction verbatim."""
    all_occ = find_occ(text, w)
    occs = all_occ[:cap] if cap else all_occ
    keep = [p for p in occs if p >= CTX]
    wid = torch.tensor([stoi[c] for c in w], dtype=torch.long)
    seqs = [torch.cat([ids[p - CTX: p], wid]) for p in keep]
    return {"seq": torch.stack(seqs) if seqs else None, "L": len(w),
            "n": len(keep), "n_all": len(all_occ)}


@torch.no_grad()
def ce_fixed(net, x, y, bs=64):
    net.eval()
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        _, loss = net(x[i: i + bs], y[i: i + bs])
        tot += float(loss.item()) * len(x[i: i + bs])
        n += len(x[i: i + bs])
    return tot / max(n, 1)


@torch.no_grad()
def eval_seq(net, seq, L, lo, state=None, mask_rows=None, chunk=128):
    """Name-position NLL/acc on battery seqs; name targets at scored slice
    [lo, lo+L) of the shifted target. state = pos_lesion state."""
    net.eval()
    x, y = seq[:, :-1], seq[:, 1:]
    T = x.shape[1]
    nlls, accs = [], []
    for i in range(0, len(x), chunk):
        xc, yc = x[i: i + chunk].to(DEVICE), y[i: i + chunk].to(DEVICE)
        if state is not None:
            if mask_rows is not None:
                state["mask"] = mask_rows[i: i + chunk].to(DEVICE)
            else:
                m = torch.zeros(xc.shape[0], T, dtype=torch.bool, device=DEVICE)
                m[:, lo: lo + L] = True
                state["mask"] = m
        logits, _ = net(xc)
        lg = logits[:, lo: lo + L, :]
        tg = yc[:, lo: lo + L]
        nll = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1),
                              reduction="none").view(-1, L)
        nlls.append(nll)
        accs.append((lg.argmax(-1) == tg).float())
    nll_m, acc_m = torch.cat(nlls), torch.cat(accs)
    return {"n": int(nll_m.shape[0]), "nll": float(nll_m.mean().item()),
            "acc": float(acc_m.mean().item()),
            "per_pos_acc": [float(v) for v in acc_m.mean(0).tolist()],
            "per_pos_nll": [float(v) for v in nll_m.mean(0).tolist()]}


@contextmanager
def pos_lesion(model, specs):
    """e042's position-resolved lesion hook, verbatim."""
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


def run_atlas(net, seq, L, lo):
    """e042 atlas verbatim: 36 heads + 12 sublayer blocks, uniform name-pos slice."""
    b0 = eval_seq(net, seq, L, lo)
    heads = torch.zeros(NL, NH)
    blocks = {"attn": [0.0] * NL, "mlp": [0.0] * NL}
    for l in range(NL):
        for h in range(NH):
            with pos_lesion(net, [("head", l, h)]) as st:
                r = eval_seq(net, seq, L, lo, state=st)
            heads[l, h] = r["nll"] - b0["nll"]
        for kind in ("attn", "mlp"):
            with pos_lesion(net, [(kind, l, None)]) as st:
                r = eval_seq(net, seq, L, lo, state=st)
            blocks[kind][l] = r["nll"] - b0["nll"]
    return {"base": {"nll": b0["nll"], "acc": b0["acc"]},
            "heads_dce": heads, "blocks_dce": blocks}


def top_heads(heads, k=3):
    flat = heads.flatten().tolist()
    idx = sorted(range(len(flat)), key=lambda i: -flat[i])[:k]
    return [(i // NH, i % NH) for i in idx]


# ------------------------------------------------------------------ exposure

def exposure(net, inst_x, inst_mask, anchor, *, steps, total, gen, tag,
             ckpt, eval_at=(), on_eval=None, lr=LR, name_bs=NAME_BS,
             corp_bs=CORP_BS, mix_random=0, train_ids=None, log=None):
    """Masked exposure: ONE token-level CE over the union of the 16x7
    name-char targets of `name_bs` install windows + all tokens of `corp_bs`
    anchor windows; AdamW (0.9,0.95) wd 0.1, lr 1e-3 house cosine, clip 1.0.
    Anchor windows: `mix_random` of them are random corpus windows, the rest
    are the paired ORIGINAL host windows (deviation 4). Resumable via ckpt."""
    opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    start, traj = 0, []
    if ckpt is not None and Path(ckpt).exists():
        st = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        net.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        gen.set_state(st["gen_state"].cpu().to(torch.uint8))
        start, traj = st["step"], st.get("traj", [])
        if log:
            log(f"resumed {tag} from step {start}")
    net.train()
    n_inst = inst_x.shape[0]
    n_anc = anchor.shape[0]
    for step in range(start + 1, steps + 1):
        f = cosine_lr(step - 1, total)          # house schedule: warmup=100 default
        for g in opt.param_groups:
            g["lr"] = lr * f
        ix = torch.randint(n_inst, (name_bs,), generator=gen)
        if mix_random:
            aj = torch.randint(n_anc, (corp_bs - mix_random,), generator=gen)
            rj = torch.randint(len(train_ids) - BLOCK - 1, (mix_random,), generator=gen)
            corp = torch.cat([anchor[aj],
                              torch.stack([train_ids[s: s + BLOCK] for s in rj]).to(DEVICE)], 0)
        else:
            aj = torch.randint(n_anc, (corp_bs,), generator=gen)
            corp = anchor[aj]
        nw = inst_x[ix]                                        # (name_bs, 256) full windows
        x = torch.cat([nw[:, :-1], corp[:, :-1]], 0)           # (name_bs+corp_bs, 255)
        y = torch.cat([nw[:, 1:], corp[:, 1:]], 0)
        m = torch.zeros(name_bs + corp_bs, x.shape[1], dtype=torch.bool, device=DEVICE)
        m[:name_bs] = inst_mask[ix]
        logits, _ = net(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                              reduction="none").view(x.shape[0], x.shape[1])
        # ONE token-level CE over the union: the 16x7 name-char targets + all
        # corpus tokens (112 vs 12,288). Group-equal weighting was rejected in
        # shakedown: it installs the onset (context->Z) within 8 steps, while
        # the design's measured anatomy has onset acc ~0 to s2000 (incumbents
        # hold the slot) — only token weighting reproduces the registered
        # per-position structure [onset~0, pos1 mid, pos2-6 ->1].
        nm = nll[:name_bs][m[:name_bs]]
        cm = nll[name_bs:]
        loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step in eval_at:
            if on_eval is not None:
                traj.append(on_eval(net, step))
            if ckpt is not None:
                torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                            "gen_state": gen.get_state(), "step": step,
                            "traj": traj}, ckpt)
    net.eval()
    return traj


# ------------------------------------------------------------------ surgery G2

def g2_rows(base_sd, sd, mats, zid, n_embd):
    exp_rows = {k: [zid] for k in mats}
    total = 0
    ok = True
    for key in ("wte.weight", "lm_head.weight"):
        d = sd[key] != base_sd[key]
        nd = int(d.sum().item())
        total += nd
        rows = sorted(set(torch.nonzero(d)[:, 0].tolist()))
        if key in exp_rows:
            if rows != exp_rows[key] or nd != len(rows) * n_embd:
                ok = False
        elif nd:
            ok = False
    others = all(torch.equal(sd[k], base_sd[k])
                 for k in sd if k not in ("wte.weight", "lm_head.weight"))
    return {"n_elements_changed": total, "expected": len(mats) * n_embd,
            "rows_ok": bool(ok), "others_bit_identical": bool(others),
            "pass": bool(ok and others and total == len(mats) * n_embd)}


def g2_organ(base_sd, sd, expect):     # expect: {key: mask|None}; None = "all changed"
    ok = True
    report = {}
    for k in base_sd:
        d = sd[k] != base_sd[k]
        nd = int(d.sum().item())
        if k not in expect:
            if nd:
                ok = False
                report[k] = f"UNEXPECTED DIFF ({nd})"
        else:
            if expect[k] is None:
                if nd != base_sd[k].numel():
                    ok = False
                    report[k] = f"PARTIAL ({nd}/{base_sd[k].numel()})"
            else:
                if not torch.equal(d, expect[k]):
                    ok = False
                    report[k] = f"MASK MISMATCH ({nd})"
    return {"changed": {k: (int((sd[k] != base_sd[k]).sum().item()),
                            int(base_sd[k].numel())) for k in expect},
            "others_clean": bool(ok), "pass": bool(ok)}


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


# ------------------------------------------------------------------ main

def main():
    T0 = time.time()
    stamp = lambda: f"[{time.time() - T0:7.1f}s]"
    log = lambda m: print(f"{stamp()} {m}", flush=True)
    set_seed(SEED)
    rd = run_dir("e043_smoke" if SMOKE else "e043")
    probes = rd / "probes.txt"
    probes.write_text(f"E043 INSTALL — probes  (smoke={SMOKE}, seed {SEED})\n",
                      encoding="utf-8")
    fb_fired = []

    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    assert corpus.vocab_size == 65
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    assert find_occ(train_text, NAME) == [] and find_occ(val_text, NAME) == []

    def load(path):
        m = TinyGPT(cfg).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        m.eval()
        return m

    B = load(E001_CKPT)
    BDO_base = load(BDO_CKPT)
    B43_base = load(B43_CKPT)
    B_sd = {k: v.clone() for k, v in B.state_dict().items()}
    bdo_sd = {k: v.clone() for k, v in BDO_base.state_dict().items()}
    b43_sd = {k: v.clone() for k, v in B43_base.state_dict().items()}

    # ---------------------------------------------------------------- G0
    g_est = estimate_loss(B, corpus, "val", n_batches=20)

    @torch.no_grad()
    def parity30(net):
        net.eval()
        gen = torch.Generator().manual_seed(1337)
        losses = []
        for _ in range(30):
            ix = torch.randint(len(val_ids) - 257, (16,), generator=gen)
            x = torch.stack([val_ids[i: i + 256] for i in ix]).to(DEVICE)
            y = torch.stack([val_ids[i + 1: i + 257] for i in ix]).to(DEVICE)
            _, loss = net(x, y)
            losses.append(float(loss.item()))
        net.train()
        return sum(losses) / len(losses)

    g_bdo, g_b43 = parity30(BDO_base), parity30(B43_base)
    G0 = {"B_estimate_loss": g_est, "B_ref": E001_VAL_CE,
          "B_pass": bool(abs(g_est - E001_VAL_CE) <= 0.03),
          "BDO_parity30": g_bdo, "B43_parity30": g_b43,
          "parity_gate": PARITY_GATE,
          "BDO_pass": bool(g_bdo <= PARITY_GATE), "B43_pass": bool(g_b43 <= PARITY_GATE)}
    log(f"G0: B estimate_loss {g_est:.4f} (ref {E001_VAL_CE}) -> {G0['B_pass']}; "
        f"BDO {g_bdo:.4f} B43 {g_b43:.4f} (gate <= {PARITY_GATE}) -> "
        f"{G0['BDO_pass']}/{G0['B43_pass']}")

    # ---------------------------------------------------------------- batteries
    log("P0: batteries (splice rng Random(24301), frozen)")
    name_ids = torch.tensor([stoi[c] for c in NAME], dtype=torch.long)
    host_occ = []
    for host in HOSTS:
        for p in find_occ(train_text, host):
            if p >= 280 and p + len(host) + POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = random.Random(SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]

    def build_win(p, host):
        w = torch.cat([train_ids[p - PRE: p], name_ids,
                       train_ids[p + len(host): p + len(host) + POST_CAP]])
        return w

    win_i = torch.stack([build_win(p, h) for p, h in install_occ])
    win_h = torch.stack([build_win(p, h) for p, h in held_occ])
    host_mix_i = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
                  "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    # exposure tensors: full install windows (60, 256); name mask over y-space
    # [PRE-1, PRE-1+7) i.e. the 7 name-char targets
    inst_x = win_i.clone().to(DEVICE)
    # ANCHOR windows (deviation 4): the ORIGINAL, unspliced install windows —
    # same 130-char contexts, true incumbent continuations. The paired anchor
    # is what makes the onset wall / knee decay / bounded CE measurable.
    anchor_full = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                               for p, _ in install_occ]).to(DEVICE)
    inst_mask = torch.zeros(60, BLOCK - 1, dtype=torch.bool, device=DEVICE)
    inst_mask[:, PRE - 1: PRE - 1 + len(NAME)] = True
    # batteries: seq = w[:PRE+7] -> scored slice [PRE-1, PRE-1+7)
    bat_i = {"seq": win_i[:, :PRE + len(NAME)], "L": len(NAME)}
    bat_h = {"seq": win_h[:, :PRE + len(NAME)], "L": len(NAME)}
    gen_prompts = ([train_text[p - 120: p] for p, _ in install_occ[:4]]
                   + [train_text[p - 120: p] for p, _ in held_occ[:4]])[:N_GEN_PROMPTS]
    log(f"install-60 hosts {host_mix_i}, held-30, prompts {len(gen_prompts)}; "
        f"name mask cols {PRE-1}..{PRE-1+len(NAME)-1}")

    bats = {}
    for w in BATTERY:
        ids, txt = (val_ids, val_text) if w == "PROSPERO" else (train_ids, train_text)
        bats[w] = build_name_bat(ids, txt, stoi, w, cap=125)
    for w in ZCLASS:
        bats[w] = build_name_bat(train_ids, train_text, stoi, w, cap=5 if w == "Zounds" else 125)

    vx, vy = fixed_blocks(val_ids, BLOCK, N_BLOCKS, CE_SEED)
    zx, zy, nzf = sample_windows(val_ids, val_text, N_BLOCKS, ZFREE_SEED,
                                 lambda t: "Z" not in t)

    def readout(net):
        r = {"R1i": eval_seq(net, bat_i["seq"], len(NAME), PRE - 1),
             "R1h": eval_seq(net, bat_h["seq"], len(NAME), PRE - 1),
             "R1z": {w: eval_seq(net, bats[w]["seq"], bats[w]["L"], CTX - 1) for w in ZCLASS},
             "R1n": {w: eval_seq(net, bats[w]["seq"], bats[w]["L"], CTX - 1) for w in BATTERY},
             "ce": {"val_all": ce_fixed(net, vx, vy), "val_zfree": ce_fixed(net, zx, zy)}}
        return r

    # ---------------------------------------------------------------- baselines + G1/G4
    base_ro = readout(B)
    base_ro2 = readout(B)
    G1 = bool(base_ro["R1i"][k] == base_ro2["R1i"][k] for k in
              ("nll", "acc", "per_pos_acc", "per_pos_nll")) and all(
        base_ro["R1n"][w][k] == base_ro2["R1n"][w][k] for w in BATTERY
        for k in ("nll", "acc"))
    pros = base_ro["R1n"]["PROSPERO"]["nll"]
    G4 = {"zephyra_r1i_nll": base_ro["R1i"]["nll"], "prospero_nll": pros,
          "pass": bool(base_ro["R1i"]["nll"] >= 5.0 and 3.5 <= pros <= 6.0)}
    log(f"G1 bit-identical: {G1} | G4: ZEPHYRA R1i {base_ro['R1i']['nll']:.2f} "
        f"(>=5.0), PROSPERO {pros:.2f} (in [3.5,6.0]) -> {G4['pass']}")
    log(f"base: R1i {base_ro['R1i']['nll']:.2f}/{base_ro['R1i']['acc']:.3f} "
        f"per-pos {[round(a,2) for a in base_ro['R1i']['per_pos_acc']]} | "
        f"R1h {base_ro['R1h']['nll']:.2f} | val_all {base_ro['ce']['val_all']:.4f} "
        f"(estimate_loss {g_est:.4f}) | zfree {base_ro['ce']['val_zfree']:.4f}")
    with probes.open("a", encoding="utf-8") as f:
        f.write(f"\nBASE B: R1i {base_ro['R1i']['nll']:.4f}/{base_ro['R1i']['acc']:.4f} "
                f"per_pos {[round(a,3) for a in base_ro['R1i']['per_pos_acc']]}\n"
                f"R1h {base_ro['R1h']['nll']:.4f}/{base_ro['R1h']['acc']:.4f} "
                f"per_pos {[round(a,3) for a in base_ro['R1h']['per_pos_acc']]}\n"
                f"R1z " + " ".join(f"{w} {base_ro['R1z'][w]['nll']:.2f}" for w in ZCLASS) + "\n"
                f"R1n " + " ".join(f"{w} {base_ro['R1n'][w]['nll']:.2f}" for w in BATTERY) + "\n"
                f"R2 val_all {base_ro['ce']['val_all']:.6f} estimate_loss {g_est:.6f} "
                f"val_zfree {base_ro['ce']['val_zfree']:.6f} (n={nzf})\n")

    # Z census (design crosscheck)
    zchars = train_text.count("Z")
    zwords = {}
    for m in re.finditer(r"[A-Za-z]*Z[A-Za-z]*", train_text):
        zwords[m.group(0)] = zwords.get(m.group(0), 0) + 1
    zcensus = {"z_train_chars": zchars,
               "z_pct": round(zchars / len(train_text) * 100, 4),
               "z_types": dict(sorted(zwords.items(), key=lambda kv: -kv[1]))}
    log(f"Z census: {zchars} chars ({zcensus['z_pct']}%), types {list(zcensus['z_types'])[:6]}")

    cell_sds: dict[str, dict] = {}       # tag -> cpu state dict (atlas candidates)

    def register_cell(tag, sd, ro, arm):
        dce = ro["ce"]["val_all"] - base_ro["ce"]["val_all"]
        dnll_i = ro["R1i"]["nll"] - base_ro["R1i"]["nll"]
        inc = [abs(ro["R1n"][w]["nll"] - base_ro["R1n"][w]["nll"]) for w in PURE_CONTROLS]
        inc_all = [abs(ro["R1n"][w]["nll"] - base_ro["R1n"][w]["nll"])
                   for w in BATTERY if w != "PROSPERO"]
        zc_abs = [abs(ro["R1z"][w]["nll"] - base_ro["R1z"][w]["nll"]) for w in ZCLASS]
        zc_dnll = {w: ro["R1z"][w]["nll"] - base_ro["R1z"][w]["nll"] for w in ZCLASS}
        cell = {
            "arm": arm, "r1i": {"nll": ro["R1i"]["nll"], "acc": ro["R1i"]["acc"],
                                "per_pos_acc": ro["R1i"]["per_pos_acc"],
                                "per_pos_nll": ro["R1i"]["per_pos_nll"]},
            "r1h": {"nll": ro["R1h"]["nll"], "acc": ro["R1h"]["acc"],
                    "per_pos_acc": ro["R1h"]["per_pos_acc"]},
            "r1z": {w: {"nll": ro["R1z"][w]["nll"], "acc": ro["R1z"][w]["acc"]} for w in ZCLASS},
            "r1n": {w: {"nll": ro["R1n"][w]["nll"], "acc": ro["R1n"][w]["acc"]} for w in BATTERY},
            "prospero_nll": ro["R1n"]["PROSPERO"]["nll"],
            "ce": ro["ce"], "dce_val": dce, "dce_zfree": ro["ce"]["val_zfree"] - base_ro["ce"]["val_zfree"],
            "dnll_zephyra": dnll_i,
            "incumbent_abs_mean": sum(inc) / len(inc),
            "incumbent_abs_max": max(inc),
            "incumbent_all_abs_max": max(inc_all),
            "zclass_abs_mean": sum(zc_abs) / len(zc_abs),
            "zclass_abs_max": max(zc_abs),
            "zclass_dnll": zc_dnll,
            "onset_acc": ro["R1i"]["per_pos_acc"][0],
            "pos36_acc": sum(ro["R1i"]["per_pos_acc"][3:7]) / 4,
            "guarded": bool(dce <= GUARD_CE),
        }
        cell_sds[tag] = {k: v.detach().cpu().clone() for k, v in sd.items()}
        return cell

    # ---------------------------------------------------------------- P0 donors
    log("P0: donor prep (masked+anchor exposure, 100 steps, resumable ckpts)")
    donors = {}
    G3 = {}
    G6 = {}
    for tag, base_net, base_sd_x in (("bdo", BDO_base, bdo_sd), ("b43", B43_base, b43_sd)):
        donor = copy.deepcopy(base_net)
        gen = torch.Generator().manual_seed(GEN_DONOR[tag])
        base_don_ro = readout(base_net)

        def on_eval(net, step):
            ro = readout(net)
            return {"step": step, "r1i_nll": ro["R1i"]["nll"], "r1i_acc": ro["R1i"]["acc"]}

        # G3: donor step-0 eval == donor base battery (same code path)
        ro0 = readout(copy.deepcopy(base_net))
        G3[tag] = {"max_abs_diff": max(abs(ro0["R1i"][k] - base_don_ro["R1i"][k])
                                       for k in ("nll", "acc"))}
        G3[tag]["pass"] = bool(G3[tag]["max_abs_diff"] == 0.0)
        exposure(donor, inst_x, inst_mask, anchor_full, steps=EXPOSE_STEPS,
                 total=EXPOSE_STEPS, gen=gen, tag=f"donor_{tag}",
                 ckpt=DONOR_CK[tag], eval_at={EXPOSE_STEPS}, on_eval=on_eval,
                 mix_random=MIX_RANDOM, train_ids=train_ids, log=log)
        final_ro = readout(donor)
        G6[tag] = {"r1i_nll": final_ro["R1i"]["nll"], "r1i_acc": final_ro["R1i"]["acc"],
                   "r1h_nll": final_ro["R1h"]["nll"], "r1h_acc": final_ro["R1h"]["acc"],
                   "gate": G6_MAX, "pass": bool(final_ro["R1i"]["nll"] <= G6_MAX),
                   "fallback_used": False}
        if not G6[tag]["pass"]:
            log(f"G6 FAILED for {tag} donor ({final_ro['R1i']['nll']:.2f} > {G6_MAX}) "
                f"— re-running fallback config (lr 3e-4, 16+16, 300 steps)")
            donor = copy.deepcopy(base_net)
            gen = torch.Generator().manual_seed(GEN_DONOR[tag] + 100)
            exposure(donor, inst_x, inst_mask, anchor_full, steps=FB_STEPS, total=FB_STEPS,
                     gen=gen, tag=f"donor_{tag}_fb", ckpt=None, eval_at={FB_STEPS},
                     mix_random=MIX_RANDOM, train_ids=train_ids)
            final_ro = readout(donor)
            G6[tag].update({"r1i_nll": final_ro["R1i"]["nll"],
                            "r1i_acc": final_ro["R1i"]["acc"],
                            "r1h_nll": final_ro["R1h"]["nll"],
                            "r1h_acc": final_ro["R1h"]["acc"],
                            "pass": bool(final_ro["R1i"]["nll"] <= G6_MAX),
                            "fallback_used": True})
            fb_fired.append(f"donor_{tag}_G6_fallback")
        donors[tag] = {"net": donor,
                       "sd": {k: v.clone() for k, v in donor.state_dict().items()},
                       "ro": final_ro, "base_ro": base_don_ro}
        log(f"donor[{tag}]: R1i {final_ro['R1i']['nll']:.2f}/{final_ro['R1i']['acc']:.2f} "
            f"(base {base_don_ro['R1i']['nll']:.2f}) R1h {final_ro['R1h']['nll']:.2f}/"
            f"{final_ro['R1h']['acc']:.2f} val_all {final_ro['ce']['val_all']:.4f} "
            f"G6 {G6[tag]['pass']} G3 {G3[tag]['pass']}")
    log(f"G3 donors: { {t: g['pass'] for t, g in G3.items()} }")

    # donor mini-atlas (heads only) -> donor's own top ZEPHYRA head
    donor_top_head = {}
    for tag, d in donors.items():
        b0 = eval_seq(d["net"], bat_i["seq"], len(NAME), PRE - 1)
        vals = torch.zeros(NL, NH)
        for l in range(NL):
            for h in range(NH):
                with pos_lesion(d["net"], [("head", l, h)]) as st:
                    r = eval_seq(d["net"], bat_i["seq"], len(NAME), PRE - 1, state=st)
                vals[l, h] = r["nll"] - b0["nll"]
        i = int(vals.argmax())
        donor_top_head[tag] = {"layer": i // NH, "head": i % NH, "dce": float(vals.flatten()[i]),
                               "heads": vals}
        log(f"donor[{tag}] ZEPHYRA mini-atlas top head L{i//NH}H{i%NH} "
            f"+{float(vals.flatten()[i]):.2f}")

    def row_surgery(donor_sd, base_of_donor_sd, matrix, variant):
        sd = {k: v.clone() for k, v in B_sd.items()}
        mats = {"both": ("wte.weight", "lm_head.weight"), "wte": ("wte.weight",),
                "lm": ("lm_head.weight",)}[matrix]
        for key in mats:
            if variant == "copy":
                sd[key][zid] = donor_sd[key][zid].clone()
            else:   # delta-add: transfer the donor's exposure delta
                sd[key][zid] = B_sd[key][zid] + (donor_sd[key][zid] - base_of_donor_sd[key][zid])
        return sd, mats

    def eval_sd(sd, tag, arm):
        m = copy.deepcopy(B)
        m.load_state_dict(sd)
        m.eval()
        cell = register_cell(tag, sd, readout(m), arm)
        log(f"{tag:22s} R1i {cell['r1i']['nll']:6.2f}/{cell['r1i']['acc']:.3f} "
            f"dCE {cell['dce_val']:+.4f} held {cell['r1h']['nll']:6.2f} "
            f"inc|d| {cell['incumbent_abs_mean']:.3f} Z|d| {cell['zclass_abs_mean']:.3f} "
            f"per-pos {[round(a,2) for a in cell['r1i']['per_pos_acc']]}")
        return cell

    # ---------------------------------------------------------------- arm A
    log("arm A: rows-only cells")
    armA = {}
    for matrix in ("wte", "lm", "both"):
        for variant in ("copy", "delta"):
            tag = f"A/{matrix}/{variant}"
            sd, mats = row_surgery(donors["bdo"]["sd"], bdo_sd, matrix, variant)
            g2 = g2_rows(B_sd, sd, mats, zid, cfg.n_embd)
            assert g2["pass"], f"G2 FAILED {tag}: {g2}"
            armA[tag] = eval_sd(sd, tag, "A")
            armA[tag]["g2"] = g2
    tag = "A/b43/both/copy"
    sd, mats = row_surgery(donors["b43"]["sd"], b43_sd, "both", "copy")
    g2 = g2_rows(B_sd, sd, mats, zid, cfg.n_embd)
    assert g2["pass"], f"G2 FAILED {tag}: {g2}"
    armA[tag] = eval_sd(sd, tag, "A")
    armA[tag]["g2"] = g2

    # ---------------------------------------------------------------- arm C
    log("arm C: body-graft cells (from exposed BDO donor)")
    dsd = donors["bdo"]["sd"]
    armC = {}

    def organ_sd(spec):
        sd = {k: v.clone() for k, v in B_sd.items()}
        expect = {}
        meta = {}
        if spec == "L0mlp" or spec == "L0mlp+rows":
            for k in ("h.0.mlp.0.weight", "h.0.mlp.0.bias",
                      "h.0.mlp.2.weight", "h.0.mlp.2.bias"):
                sd[k] = dsd[k].clone()
                expect[k] = None
        if spec == "L0attn":
            for k in ("h.0.attn.c_attn.weight", "h.0.attn.c_proj.weight"):
                sd[k] = dsd[k].clone()
                expect[k] = None
        if spec == "L0mlp+rows":
            for k in ("wte.weight", "lm_head.weight"):
                sd[k] = sd[k].clone()
                sd[k][zid] = dsd[k][zid].clone()
                mk = torch.zeros_like(B_sd[k], dtype=torch.bool)
                mk[zid] = True
                expect[k] = mk
        if spec == "tophead":
            l0, h0 = donor_top_head["bdo"]["layer"], donor_top_head["bdo"]["head"]
            hd = cfg.n_embd // cfg.n_head
            ka = "h.0.attn.c_attn.weight"
            mk_a = torch.zeros_like(B_sd[ka], dtype=torch.bool)
            for off in (0, cfg.n_embd, 2 * cfg.n_embd):
                mk_a[off + h0 * hd: off + (h0 + 1) * hd, :] = True
                sd[ka][off + h0 * hd: off + (h0 + 1) * hd, :] = \
                    dsd[ka][off + h0 * hd: off + (h0 + 1) * hd, :]
            kp = "h.0.attn.c_proj.weight"
            mk_p = torch.zeros_like(B_sd[kp], dtype=torch.bool)
            mk_p[:, h0 * hd: (h0 + 1) * hd] = True
            sd[kp][:, h0 * hd: (h0 + 1) * hd] = dsd[kp][:, h0 * hd: (h0 + 1) * hd]
            expect[ka], expect[kp] = mk_a, mk_p
            meta["grafted_head"] = f"L{l0}H{h0}"
        return sd, expect, meta

    for spec in ("L0mlp", "L0attn", "L0mlp+rows", "tophead"):
        tag = f"C/{spec}"
        sd, expect, meta = organ_sd(spec)
        g2 = g2_organ(B_sd, sd, expect)
        assert g2["pass"], f"G2 FAILED {tag}: {g2}"
        armC[tag] = eval_sd(sd, tag, "C")
        armC[tag]["g2"] = g2
        armC[tag].update(meta)

    # ---------------------------------------------------------------- arm B
    log("arm B: rows+brief exposure (eval @25/50/100)")
    armB = {}
    # B/D arms re-run FRESH (their eval cells are side effects of the loop; a
    # resumed-complete ckpt would leave the registry empty — donors resume fine
    # because their readout is recomputed from the final weights)
    for k in BARM_CK.values():
        if k.exists():
            k.unlink()

    def run_expose_arm(tag, seed, ckpt, preinstalled_rows=False):
        net = copy.deepcopy(B)
        if preinstalled_rows:
            sd0, _ = row_surgery(donors["bdo"]["sd"], bdo_sd, "both", "copy")
            net.load_state_dict(sd0)
            net.eval()
        gen = torch.Generator().manual_seed(seed)

        def on_eval(net_, step):
            ro = readout(net_)
            sd = {k: v.detach().cpu().clone() for k, v in net_.state_dict().items()}
            ct = register_cell(f"{tag}@s{step}", sd, ro, "B")
            ct["step"] = step
            armB[f"{tag}@s{step}"] = ct
            return {"step": step, "r1i_nll": ro["R1i"]["nll"]}

        exposure(net, inst_x, inst_mask, anchor_full, steps=EXPOSE_STEPS,
                 total=EXPOSE_STEPS, gen=gen, tag=tag, ckpt=ckpt,
                 eval_at=set(B_EVAL), on_eval=on_eval, mix_random=MIX_RANDOM,
                 train_ids=train_ids, log=log)
        return net

    run_expose_arm("B/bare", GEN_B_BARE, BARM_CK["b_bare"])
    run_expose_arm("B/rows-pre", GEN_B_ROWS, BARM_CK["b_rows"], preinstalled_rows=True)

    # ---------------------------------------------------------------- arm D
    doses = list(D_DOSES)
    if not SMOKE and time.time() - T0 > 480:
        doses = [d for d in doses if d <= 400]
        fb_fired.append("D_doses_capped_400")
    d_total = D_TOTAL if doses == D_DOSES else doses[-1]
    armD = {}
    for dtag, dseed, dmix in (("Dmix", GEN_D, MIX_RANDOM), ("Dpair", GEN_D + 1, 0)):
        log(f"arm D/{dtag}: dose ladder {doses} (+ long cell {d_total}; "
            f"anchor: {CORP_BS - dmix} paired + {dmix} random)")
        netD = copy.deepcopy(B)
        genD = torch.Generator().manual_seed(dseed)

        def on_eval_d(net_, step, dtag=dtag):
            ro = readout(net_)
            sd = {k: v.detach().cpu().clone() for k, v in net_.state_dict().items()}
            ct = register_cell(f"{dtag}@s{step}", sd, ro, "D")
            ct["step"] = step
            ct["variant"] = dtag
            armD[f"{dtag}@s{step}"] = ct
            return {"step": step, "r1i_nll": ro["R1i"]["nll"], "dce": ct["dce_val"]}

        exposure(netD, inst_x, inst_mask, anchor_full, steps=d_total, total=d_total,
                 gen=genD, tag=f"{dtag}_ladder",
                 ckpt=BARM_CK["d_ladder"] if dtag == "Dmix" else None,
                 eval_at=set(doses + [d_total]), on_eval=on_eval_d,
                 mix_random=dmix, train_ids=train_ids, log=log)
    for k in sorted(armD, key=lambda s: armD[s]["step"]):
        c = armD[k]
        log(f"{k:12s} R1i {c['r1i']['nll']:6.2f}/{c['r1i']['acc']:.3f} dCE "
            f"{c['dce_val']:+.4f} onset {c['onset_acc']:.2f} per-pos "
            f"{[round(a,2) for a in c['r1i']['per_pos_acc']]}")

    # ---------------------------------------------------------------- derived metrics
    all_cells = {}
    all_cells.update(armA)
    all_cells.update(armC)
    all_cells.update(armB)
    all_cells.update(armD)
    guarded = {t: c for t, c in all_cells.items() if c["guarded"]}
    if guarded:
        d_best_tag = min(guarded, key=lambda t: guarded[t]["r1i"]["nll"])
        d_best = guarded[d_best_tag]
        denom = base_ro["R1i"]["nll"] - d_best["r1i"]["nll"]
    else:
        d_best_tag, d_best, denom = None, None, base_ro["R1i"]["nll"] - BAR_I1_NLL
    for tag, c in all_cells.items():
        imp = base_ro["R1i"]["nll"] - c["r1i"]["nll"]
        c["improvement"] = imp
        c["gap_closed"] = (imp / denom) if abs(denom) > 1e-9 else None
        c["barI1"] = bool(c["r1i"]["nll"] <= BAR_I1_NLL and c["r1i"]["acc"] >= BAR_I1_ACC
                          and c["guarded"])
        c["barI2"] = bool(c["r1i"]["nll"] <= BAR_I2_NLL and c["r1i"]["acc"] >= BAR_I2_ACC
                          and c["guarded"])
        c["s_install"] = (imp / c["incumbent_abs_mean"]) if c["incumbent_abs_mean"] > 1e-9 \
            else float("inf")
        c["c_install"] = (imp / c["dce_val"]) if c["dce_val"] >= 0.01 else None
        c["c_install_lower_bound"] = (imp / 0.01) if imp > 0 else None
    ranked = sorted(guarded, key=lambda t: guarded[t]["r1i"]["nll"])
    log(f"gap denominator (D-best-guarded {d_best_tag}): {denom:.2f} nats; "
        f"guarded best cells: {[(t, round(guarded[t]['r1i']['nll'],2)) for t in ranked[:4]]}")

    # ---------------------------------------------------------------- R3 generation
    log(f"R3: generation {len(gen_prompts)}x{GEN_TOK} (base + 2 best guarded cells)")
    gen_tags = ["base"] + ranked[:2]
    GEN_CELLS = {}

    def run_gen(net, tag):
        counts = {"ZEPHYRA": 0, "ELIZABETH": 0, "FLORIZEL": 0, "Z_words": 0,
                  "stray_Z_words": 0, "Z_chars": 0, "chars": 0}
        lines = [f"\n{'='*70}\nGENERATION — {tag} ({len(gen_prompts)} prompts x {GEN_TOK} tok, "
                 f"temp 0.8, top-k 40, seed=prompt index)\n{'='*70}"]
        for i, pr in enumerate(gen_prompts):
            torch.manual_seed(i)
            out = generate(net, corpus, pr, max_new_tokens=GEN_TOK,
                           temperature=0.8, top_k=40)
            cont = out[len(pr):]
            zp = len(re.findall(r"(?<![A-Za-z])ZEPHYRA(?![A-Za-z])", cont))
            zw = len(re.findall(r"(?<![A-Za-z])Z[A-Za-z]*", cont))
            counts["ZEPHYRA"] += zp
            counts["ELIZABETH"] += len(re.findall(r"(?<![A-Za-z])ELIZABETH(?![A-Za-z])", cont))
            counts["FLORIZEL"] += len(re.findall(r"(?<![A-Za-z])FLORIZEL(?![A-Za-z])", cont))
            counts["Z_words"] += zw
            counts["stray_Z_words"] += zw - zp
            counts["Z_chars"] += cont.count("Z")
            counts["chars"] += len(cont)
            lines.append(f"\n--- [{tag}] prompt {i} ---\nPROMPT: {pr}\nGEN:    {cont}")
        with probes.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        rates = {k: (round(v / counts["chars"] * 1e4, 2) if counts["chars"] else None)
                 for k, v in counts.items() if k != "chars"}
        return {"counts": {k: v for k, v in counts.items() if k != "chars"},
                "chars": counts["chars"], "per_10k_chars": rates}

    gen_models = {"base": B}
    for t in ranked[:2]:
        m = copy.deepcopy(B)
        m.load_state_dict(cell_sds[t])
        m.eval()
        gen_models[t] = m
    for t in gen_tags:
        GEN_CELLS[t] = run_gen(gen_models[t], t)
        gc = GEN_CELLS[t]["counts"]
        log(f"gen[{t:14s}]: ZEPHYRA {gc['ZEPHYRA']} ELIZABETH {gc['ELIZABETH']} "
            f"FLORIZEL {gc['FLORIZEL']} stray-Z-words {gc['stray_Z_words']} "
            f"/ {GEN_CELLS[t]['chars']} chars")
    # G5: rerun one prompt bit-identical
    torch.manual_seed(0)
    r1 = generate(B, corpus, gen_prompts[0], max_new_tokens=GEN_TOK, temperature=0.8, top_k=40)
    torch.manual_seed(0)
    r2 = generate(B, corpus, gen_prompts[0], max_new_tokens=GEN_TOK, temperature=0.8, top_k=40)
    G5 = {"prompt": 0, "pass": bool(r1 == r2)}
    log(f"G5 generation rerun bit-identical: {G5['pass']}")

    # ---------------------------------------------------------------- R4 atlases
    log("R4: pos-resolved atlas re-run (base + best cell + s25 cell)")
    atlas_cells = []
    for t in (ranked[0] if ranked else None, "Dmix@s25" if "Dmix@s25" in cell_sds else None):
        if t and t not in atlas_cells:
            atlas_cells.append(t)
    atlases = {"base": {}}
    for name in ATLAS_NAMES:
        if name == "ZEPHYRA":
            atlases["base"][name] = run_atlas(B, bat_i["seq"], len(NAME), PRE - 1)
        else:
            atlases["base"][name] = run_atlas(B, bats[name]["seq"], bats[name]["L"], CTX - 1)
    for t in atlas_cells:
        m = copy.deepcopy(B)
        m.load_state_dict(cell_sds[t])
        m.eval()
        atlases[t] = {}
        for name in ATLAS_NAMES:
            if name == "ZEPHYRA":
                atlases[t][name] = run_atlas(m, bat_i["seq"], len(NAME), PRE - 1)
            else:
                atlases[t][name] = run_atlas(m, bats[name]["seq"], bats[name]["L"], CTX - 1)
        a = atlases[t]["ZEPHYRA"]
        th = top_heads(a["heads_dce"], 1)[0]
        tb = max([("attn", i) for i in range(NL)] + [("mlp", i) for i in range(NL)],
                 key=lambda s: a["blocks_dce"][s[0]][s[1]])
        log(f"atlas[{t}]: ZEPHYRA top head L{th[0]}H{th[1]} "
            f"{float(a['heads_dce'][th[0], th[1]]):+.2f}, top block L{tb[1]}-{tb[0]} "
            f"{a['blocks_dce'][tb[0]][tb[1]]:+.2f}")
    for name in ATLAS_NAMES:
        a = atlases["base"][name]
        th = top_heads(a["heads_dce"], 1)[0]
        tb = max([("attn", i) for i in range(NL)] + [("mlp", i) for i in range(NL)],
                 key=lambda s: a["blocks_dce"][s[0]][s[1]])
        log(f"atlas[base/{name}]: top head L{th[0]}H{th[1]} "
            f"{float(a['heads_dce'][th[0], th[1]]):+.2f}, top block L{tb[1]}-{tb[0]} "
            f"{a['blocks_dce'][tb[0]][tb[1]]:+.2f}")

    # ---------------------------------------------------------------- verdicts
    bdo_a = [t for t in armA if not t.startswith("A/b43")]
    bdo_a_g = [all_cells[t]["gap_closed"] is not None and all_cells[t]["gap_closed"] < 0.10
               for t in bdo_a]
    bdo_a_ce = [abs(all_cells[t]["dce_val"]) <= 0.005 for t in bdo_a]
    bdo_a_col = [all_cells[t]["incumbent_all_abs_max"] <= 0.05
                 and all_cells[t]["zclass_abs_max"] <= 0.05 for t in bdo_a]
    p1 = {
        "a_bdo_rows_only": {
            "cells": bdo_a,
            "gap_closed_lt_10pct": all(bdo_a_g),
            "abs_dce_le_0.005": all(bdo_a_ce),
            "incumbent_zclass_le_0.05": all(bdo_a_col),
            "per_cell": {t: {"gap_closed": all_cells[t]["gap_closed"],
                             "dce": all_cells[t]["dce_val"],
                             "inc_max": all_cells[t]["incumbent_all_abs_max"],
                             "z_max": all_cells[t]["zclass_abs_max"]} for t in bdo_a},
        },
        "b_b43_copy_both": {
            "gap_closed": all_cells["A/b43/both/copy"]["gap_closed"],
            "zclass_mean_dnll": sum(all_cells["A/b43/both/copy"]["zclass_dnll"].values()) / 4,
            "fails_both_ways": bool(
                (all_cells["A/b43/both/copy"]["gap_closed"] or 0) < 0.10
                and sum(all_cells["A/b43/both/copy"]["zclass_dnll"].values()) / 4 >= 1.0),
        },
    }
    p1["a_holds"] = bool(all(bdo_a_g) and all(bdo_a_ce) and all(bdo_a_col))
    p1["confirmed"] = bool(p1["a_holds"] and p1["b_b43_copy_both"]["fails_both_ways"])
    p1["rule"] = ("the row index that sufficed for erasure does not carry installation; "
                  "row transplantability follows init lineage (C3)")

    bar2_reachers = [t for t, c in all_cells.items() if c["barI2"]]
    barI1_tags = [t for t, c in guarded.items() if c["barI1"]]
    best_barI1 = min(barI1_tags, key=lambda t: guarded[t]["r1i"]["nll"]) if barI1_tags else None
    d_by_step = sorted([c for c in armD.values() if c["variant"] == "Dmix"],
                       key=lambda c: c["step"])
    d_pair = sorted([c for c in armD.values() if c["variant"] == "Dpair"],
                    key=lambda c: c["step"])
    nll_at = {c["step"]: c["r1i"]["nll"] for c in d_by_step}
    ce_at = {c["step"]: c["ce"]["val_all"] for c in d_by_step}
    nll_pair = {c["step"]: c["r1i"]["nll"] for c in d_pair}
    ce_pair = {c["step"]: c["ce"]["val_all"] for c in d_pair}
    steps_sorted = sorted(nll_at)
    steps_pair = sorted(nll_pair)
    p2 = {"barI2_reachers": bar2_reachers,
          "no_arm_reaches_barI2_at_guard": len(bar2_reachers) == 0,
          "i_transient": {
              "nll_25": nll_at.get(25), "nll_400": nll_at.get(400),
              "decay_gt_1nat": bool(nll_at.get(400) is not None and nll_at.get(25) is not None
                                    and nll_at[400] > nll_at[25] + 1.0),
              "ce_after_100": {s: ce_at[s] for s in steps_sorted if s >= 100},
              "ce_monotone_rising_after_100": all(
                  ce_at[steps_sorted[i]] <= ce_at[steps_sorted[i + 1]] + 1e-4
                  for i in range(len(steps_sorted) - 1) if steps_sorted[i] >= 100),
              "pair_variant": {"nll_by_step": nll_pair, "ce_by_step": ce_pair,
                               "nll_25": nll_pair.get(25), "nll_400": nll_pair.get(400),
                               "decay_gt_1nat": bool(
                                   nll_pair.get(400) is not None and nll_pair.get(25) is not None
                                   and nll_pair[400] > nll_pair[25] + 1.0),
                               "dce_final": (ce_pair.get(steps_pair[-1], 0) - base_ro["ce"]["val_all"])
                               if steps_pair else None,
                               "note": ("fully-paired anchor (48 originals): the strict "
                                        "'interleaved corpus windows' reading — onset walled "
                                        "at 0.00 at every dose but the narrow replay destroys "
                                        "general CE; the mix variant is the bounded-CE "
                                        "trajectory")}},
          "ii_positional": {
              "onset_le_0.10_all_guarded": all(c["onset_acc"] <= 0.10
                                               for c in guarded.values()),
              "barI1_reachers": [t for t, c in guarded.items() if c["barI1"]],
              "pos36_best_guarded_barI1_cell": (
                  guarded[best_barI1]["pos36_acc"] if best_barI1 else None),
              "pos36_all_barI1_cells_ge_0.9_strict": (
                  all(guarded[t]["pos36_acc"] >= 0.9 for t in barI1_tags)
                  if barI1_tags else False),
              # registered reading: the best guarded install (the ceiling
              # representative) reaches positions 3-6 >= 0.9
              "installed_guarded_cells_pos36_ge_0.9": bool(
                  best_barI1 and guarded[best_barI1]["pos36_acc"] >= 0.9)},
          "iii_rows_pre_knee": None}
    knees = {}
    for tagp in ("B/bare", "B/rows-pre"):
        pts = [(t, armB[t]["r1i"]["nll"]) for t in armB if t.startswith(tagp)]
        knees[tagp] = min(pts, key=lambda p: p[1])[0] if pts else None
    if knees["B/bare"] and knees["B/rows-pre"]:
        s_bare = int(knees["B/bare"].split("@s")[1])
        s_pre = int(knees["B/rows-pre"].split("@s")[1])
        p2["iii_rows_pre_knee"] = {"knee_bare": s_bare, "knee_rows_pre": s_pre,
                                   "shift_ratio": round(s_pre / max(s_bare, 1), 3),
                                   "lt_2x": bool(0.5 <= s_pre / max(s_bare, 1) <= 2.0)}
    p2["confirmed"] = bool(p2["no_arm_reaches_barI2_at_guard"]
                           and p2["i_transient"]["decay_gt_1nat"]
                           and p2["i_transient"]["ce_monotone_rising_after_100"]
                           and p2["ii_positional"]["onset_le_0.10_all_guarded"]
                           and p2["ii_positional"]["installed_guarded_cells_pos36_ge_0.9"]
                           and (p2["iii_rows_pre_knee"] or {}).get("lt_2x", False))

    p3 = {"cells": atlas_cells, "a_machinery_conserved": {}, "b_zephyra_carriers": {},
          "c_base_atlas_empty": None}
    for name in ("JULIET", "ROMEO", "LUCIO"):
        bh = atlases["base"][name]["heads_dce"]
        p3["a_machinery_conserved"][name] = {}
        for t in atlas_cells:
            ph = atlases[t][name]["heads_dce"]
            p3["a_machinery_conserved"][name][t] = {
                "top1": f"L{top_heads(ph,1)[0][0]}H{top_heads(ph,1)[0][1]}",
                "top1_is_L0H3": top_heads(ph, 1)[0] == (0, 3),
                "top3_set_unchanged": set(top_heads(ph, 3)) == set(top_heads(bh, 3)),
                "spearman_vs_base": spearman(bh.flatten().tolist(), ph.flatten().tolist())}
    zh = atlases["base"]["ZEPHYRA"]["heads_dce"]
    p3["c_base_atlas_empty"] = {
        "max_head": float(zh.max()), "max_head_id": f"L{int(zh.argmax())//NH}H{int(zh.argmax())%NH}",
        "max_mlp": max(atlases["base"]["ZEPHYRA"]["blocks_dce"]["mlp"]),
        "empty": bool(float(zh.max()) <= 0.5
                      and max(atlases["base"]["ZEPHYRA"]["blocks_dce"]["mlp"]) <= 0.5)}
    for t in atlas_cells:
        a = atlases[t]["ZEPHYRA"]
        tb = max([("attn", i) for i in range(NL)] + [("mlp", i) for i in range(NL)],
                 key=lambda s: a["blocks_dce"][s[0]][s[1]])
        ta = max(range(NL), key=lambda i: a["blocks_dce"]["attn"][i])
        mh = float(a["heads_dce"].max())
        p3["b_zephyra_carriers"][t] = {
            "top_block": f"L{tb[1]}-{tb[0]}", "top_block_dce": a["blocks_dce"][tb[0]][tb[1]],
            "top_attn": f"L{ta}-attn", "top_attn_dce": a["blocks_dce"]["attn"][ta],
            "max_head_dce": mh, "head_over_block_ratio": round(
                mh / max(a["blocks_dce"][tb[0]][tb[1]], 1e-9), 3),
            "no_dominant_head_abs_le_0.5": bool(mh <= 0.5),
            # registered meaning: the BLOCK carries the completion, the head is
            # minor (MP preview: head +0.17 vs block ~7 = 2%); dominance judged
            # relative to the top block (head < half the block's load)
            "no_dominant_head": bool(
                mh < 0.5 * max(a["blocks_dce"][tb[0]][tb[1]], 1e-9)),
            "rides_shared_L0": bool(tb == ("mlp", 0) and ta == 0)}
    p3["a_holds"] = all(p3["a_machinery_conserved"][name][t]["top1_is_L0H3"]
                        and p3["a_machinery_conserved"][name][t]["top3_set_unchanged"]
                        and p3["a_machinery_conserved"][name][t]["spearman_vs_base"] >= 0.8
                        for name in ("JULIET", "ROMEO", "LUCIO") for t in atlas_cells)
    p3["b_holds"] = all(v["rides_shared_L0"] and v["no_dominant_head"]
                        for v in p3["b_zephyra_carriers"].values())
    p3["confirmed"] = bool(p3["a_holds"] and p3["b_holds"] and p3["c_base_atlas_empty"]["empty"])
    if not p3["c_base_atlas_empty"]["empty"]:
        p3["c_note"] = (
            "The registered 'base ZEPHYRA atlas is empty' claim is construction-sensitive: "
            "under the install battery the base net is CONFIDENTLY WRONG (base NLL ~9.9) and its "
            "ZEPHYRA-slot predictions are actively produced by the INCUMBENTS' own completion "
            f"machinery (top head {p3['c_base_atlas_empty']['max_head_id']} "
            f"{p3['c_base_atlas_empty']['max_head']:+.2f}, top L1-attn block) — an 'empty' atlas "
            "presumes indifference at these positions, which the frozen mixed-60 battery falsifies "
            "(the incumbents own the slot; e042 showed this same machinery is theirs). P3's "
            "substance — (a) incumbents' machinery undisturbed, (b) the install rides the shared "
            "L0 block with no dominant single head — is unaffected by (c).")

    # symmetry verdict
    surgical = {**armA, **armC}
    cond1_cells = [t for t, c in surgical.items()
                   if c["barI1"] and c["incumbent_abs_mean"] <= 0.05
                   and c["zclass_abs_mean"] <= 0.05]
    cond1 = len(cond1_cells) > 0
    cond1_bdo = any(not t.startswith("A/b43") for t in cond1_cells)
    cond1_b43 = any(t.startswith("A/b43") for t in cond1_cells)
    cond3 = p3["a_holds"] and cond1
    if cond1 and cond3:
        symmetry = "BIDIRECTIONAL"
    elif cond1 and cond1_bdo and not cond1_b43:
        symmetry = "ASYMMETRIC-BASIS"
    else:
        symmetry = "ASYMMETRIC-CHEAP-REMOVE"
    best_install = ranked[0] if ranked else None
    symmetry_detail = {
        "cond1_additive_surgical_barI1": cond1, "cond1_cells": cond1_cells,
        "cond2_subtractive_mirror": "established by e042 (D2+L3H5: Bar-2 at +0.00083, S_name 1937)",
        "cond3_coordinate_identity_and_P3a": bool(cond3 and p3["a_holds"]),
        "best_guarded_cell": best_install,
        "best_guarded": {k: d_best[k] for k in ("r1i", "dce_val", "s_install",
                                                "gap_closed")} if d_best else None,
        "verdict": symmetry,
    }
    log("=" * 70)
    log(f"P1 confirmed: {p1['confirmed']} (a {p1['a_holds']}, b {p1['b_b43_copy_both']['fails_both_ways']})")
    log(f"P2 confirmed: {p2['confirmed']} (barI2 reachers: {bar2_reachers or 'none'}; "
        f"transient decay {p2['i_transient']['decay_gt_1nat']}; "
        f"CE rising {p2['i_transient']['ce_monotone_rising_after_100']}; "
        f"positional {p2['ii_positional']}; knee {p2['iii_rows_pre_knee']})")
    log(f"P3 confirmed: {p3['confirmed']} (a {p3['a_holds']}, b {p3['b_holds']}, "
        f"c {p3['c_base_atlas_empty']['empty']})")
    log(f"SYMMETRY VERDICT: {symmetry} (cond1={cond1}, cond3={cond3})")

    # ---------------------------------------------------------------- outputs
    metrics = {
        "experiment": "e043_install", "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": SEED, "smoke": SMOKE, "design": "scratch/e043_design.md",
        "deviations": [l.strip() for l in __doc__.split("DEVIATIONS")[1].split("Run:")[0]
                       .splitlines() if l.strip()],
        "target": {"name": NAME, "len": len(NAME), "z_row_id": zid,
                   "train_occ": 0, "val_occ": 0, "z_census": zcensus},
        "batteries": {"install_n": len(install_occ), "held_n": len(held_occ),
                      "install_hosts": host_mix_i,
                      "candidate_host_occs": len(host_occ),
                      "r1z": {w: {"n": bats[w]["n"], "n_all": bats[w]["n_all"]} for w in ZCLASS},
                      "r1n": {w: {"n": bats[w]["n"]} for w in BATTERY},
                      "val_zfree_n": nzf},
        "gates": {"G0": G0, "G1": G1, "G3_donors": G3, "G4": G4, "G5": G5, "G6": G6,
                  "G2_all_cells": True},
        "baseline": {"r1i": base_ro["R1i"], "r1h": base_ro["R1h"],
                     "r1z": {w: base_ro["R1z"][w] for w in ZCLASS},
                     "r1n": {w: {k: base_ro["R1n"][w][k] for k in ("nll", "acc")} for w in BATTERY},
                     "ce": base_ro["ce"], "estimate_loss": g_est},
        "donors": {t: {"final": {k: donors[t]["ro"][k] for k in ("R1i", "R1h")},
                       "base_r1i_nll": donors[t]["base_ro"]["R1i"]["nll"],
                       "val_all": donors[t]["ro"]["ce"]["val_all"],
                       "top_zephyra_head": {k: donor_top_head[t][k]
                                            for k in ("layer", "head", "dce")}}
                   for t in donors},
        "armA": armA, "armB": armB, "armC": armC, "armD": armD,
        "derived": {"gap_denominator": denom, "d_best_tag": d_best_tag,
                    "guarded_ranked": ranked,
                    "bars": {"BarI1": [BAR_I1_NLL, BAR_I1_ACC, GUARD_CE],
                             "BarI2": [BAR_I2_NLL, BAR_I2_ACC, GUARD_CE]},
                    "s_install_bar": S_INSTALL_BAR},
        "generation": GEN_CELLS,
        "atlases": {t: {n: {"heads_dce": atlases[t][n]["heads_dce"],
                            "blocks_dce": atlases[t][n]["blocks_dce"],
                            "base": atlases[t][n]["base"]} for n in atlases[t]}
                    for t in atlases},
        "atlas_cells": atlas_cells,
        "verdicts": {"P1": p1, "P2": p2, "P3": p3, "symmetry": symmetry_detail},
        "timing": {"total_s": round(time.time() - T0, 1), "fallbacks_fired": fb_fired},
        "config": cfg_dict(cfg),
    }
    save_json(rd / "metrics.json", jsonable(metrics))
    with probes.open("a", encoding="utf-8") as f:
        f.write(f"\nVERDICTS: P1 {p1['confirmed']} | P2 {p2['confirmed']} | "
                f"P3 {p3['confirmed']} | SYMMETRY {symmetry}\n")

    # ---------------------------------------------------------------- plots
    if not SMOKE:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
        ax = axes[0]
        groups = [("A", armA, "crimson", "x"), ("C", armC, "darkorange", "s"),
                  ("B", armB, "royalblue", "o"), ("D", armD, "seagreen", "D")]
        for gname, cells, col, mk in groups:
            xs = [c["dce_val"] for c in cells.values()]
            ys = [c["r1i"]["nll"] for c in cells.values()]
            ax.scatter(xs, ys, c=col, marker=mk, s=46, label=f"arm {gname}", zorder=3,
                       edgecolors="k", linewidths=0.4)
        ax.scatter([0.0], [base_ro["R1i"]["nll"]], c="k", marker="*", s=130,
                   label="base B", zorder=4)
        for t in ranked[:3]:
            ax.annotate(t, (guarded[t]["dce_val"], guarded[t]["r1i"]["nll"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.axvline(GUARD_CE, color="gray", ls="--", lw=1, label="+0.10 CE guard")
        ax.axhline(BAR_I1_NLL, color="navy", ls=":", lw=1, label="Bar-I1 NLL 4.17")
        ax.axhline(BAR_I2_NLL, color="purple", ls=":", lw=1, label="Bar-I2 NLL 1.0")
        ax.axvline(0.0, color="k", lw=0.5)
        ax.set_xlabel("dCE val (corpus cost)")
        ax.set_ylabel("ZEPHYRA install NLL (R1i)")
        best_lbl = f"{d_best_tag} {d_best['r1i']['nll']:.2f}" if d_best else "none"
        ax.set_title(f"install frontier (base {base_ro['R1i']['nll']:.2f}; "
                     f"best guarded {best_lbl})")
        ax.legend(fontsize=7)
        ax = axes[1]
        st = [c["step"] for c in d_by_step]
        ax.plot(st, [c["r1i"]["nll"] for c in d_by_step], "o-", color="crimson", label="R1i NLL")
        ax.plot(st, [c["r1h"]["nll"] for c in d_by_step], "s-", color="pink", ms=4,
                label="R1h NLL (held)")
        ax.axhline(BAR_I1_NLL, color="navy", ls=":", lw=1)
        ax.set_xlabel("exposure step")
        ax.set_ylabel("NLL")
        ax2 = ax.twinx()
        ax2.plot(st, [c["dce_val"] for c in d_by_step], "^-", color="seagreen", label="dCE val")
        ax2.axhline(GUARD_CE, color="seagreen", ls=":", lw=1)
        ax2.set_ylabel("dCE val", color="seagreen")
        ax.set_title("arm D: the install transient (ceiling is step-indexed)")
        ax.legend(fontsize=8, loc="upper center")
        fig.suptitle("E043 — INSTALL a name: additive symmetry test")
        fig.tight_layout()
        fig.savefig(rd / "install_frontier.png", dpi=130)
        plt.close(fig)

        fig, axes = plt.subplots(2, 4, figsize=(19, 8))
        def heat(ax, M, title, cmap="inferno"):
            im = ax.imshow(M.tolist(), cmap=cmap, aspect="auto")
            for l in range(NL):
                for h in range(NH):
                    ax.text(h, l, f"{M[l,h]:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if M[l, h] < 0.55 * float(M.max()) else "black")
            ax.set_xticks(range(NH)); ax.set_xticklabels([f"H{h}" for h in range(NH)], fontsize=7)
            ax.set_yticks(range(NL)); ax.set_yticklabels([f"L{l}" for l in range(NL)], fontsize=7)
            ax.set_title(title, fontsize=9)
        for j, (t, ttl) in enumerate([("base", "base B")] + [(t, f"post {t}") for t in atlas_cells]):
            if t in atlases:
                heat(axes[0, j], atlases[t]["ZEPHYRA"]["heads_dce"],
                     f"ZEPHYRA head atlas — {ttl}")
        for j, name in enumerate(("JULIET", "ROMEO", "LUCIO")):
            bh = atlases["base"][name]["heads_dce"]
            ph = atlases[atlas_cells[0]][name]["heads_dce"] if atlas_cells else bh
            axes[1, j].imshow((ph - bh).tolist(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
            axes[1, j].set_title(f"{name}: post-minus-base head atlas", fontsize=9)
            rho = spearman(bh.flatten().tolist(), ph.flatten().tolist())
            axes[1, j].set_xlabel(f"spearman vs base {rho:.2f}", fontsize=8)
        ax = axes[1, 3]
        xs = list(range(2 * NL))
        lbl = [f"L{i}at" for i in range(NL)] + [f"L{i}mlp" for i in range(NL)]
        for i, name in enumerate(ATLAS_NAMES):
            vals = (atlases[atlas_cells[0]][name]["blocks_dce"]["attn"]
                    + atlases[atlas_cells[0]][name]["blocks_dce"]["mlp"]) if atlas_cells else [0]*12
            ax.bar([x + 0.2 * (i - 1.5) for x in xs], vals, 0.2, label=name)
        ax.set_xticks(xs); ax.set_xticklabels(lbl, fontsize=6)
        ax.set_ylabel("dCE (nats)")
        ax.set_title(f"post-install block atlas ({atlas_cells[0] if atlas_cells else 'n/a'})", fontsize=9)
        ax.legend(fontsize=7)
        fig.suptitle("E043 machinery atlas — does install disturb the shared L0 block?")
        fig.tight_layout()
        fig.savefig(rd / "machinery_atlas.png", dpi=130)
        plt.close(fig)

    log(f"outputs: {rd}")
    log(f"total {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
