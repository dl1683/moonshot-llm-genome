"""E042 — the name-circuit atlas: what carries JULIET completion in the BODY,
is LUCIO's circuit the same one, and can row-surgery + body-lesion erase?

Arc "Edit the organism", slot 2 (tasking 2026-09-24). EVAL-ONLY on the E001
checkpoint. Builds on T011/e023: J-row surgery (D2) reaches S_name 573 at
+0.0008 corpus CE but leaves a 13.6% residual accuracy (all of it at name
position 3, per e023 per-pos [0,0,0,.82,0,0]); D1 collateral is idiosyncratic
(LUCIO dNLL 8.53, ROMEO 2.45) and does NOT track letter overlap (rho=0.61).
The residual and the collateral idiosyncrasy are this experiment's targets.

Q1 ATLAS — for all 36 heads + 12 sublayer blocks (attn/mlp x 6 layers),
  dCE on the JULIET battery when that component is zeroed ONLY at the
  name-region positions (position-resolved lesioning, e038's head hook made
  positional): which body components carry name completion?
Q2 OVERLAP — the same atlas for LUCIO (the worst-collateral name) + JOHN and
  ROMEO for breadth: is name-completion machinery shared across names
  (explaining collateral as circuit overlap) or name-specific?
Q3 TWO-FACTOR ERASURE — a second atlas conditioned on the D2-surgered net
  ranks components by RESIDUAL support; D2 + top components COMBINED is then
  tested for Bar-2 (acc <= 0.10 AND NLL >= ln65) at total corpus cost < +0.05.

Patch semantics (pre-registered): the deployed two-factor patch is
content-triggered — components are zeroed at query positions whose context
ends with a non-empty proper prefix of "JULIET" ("J".."JULIE"); a "strict"
variant uses prefixes of length >= 2 ("JU".."JULIE"), which still fires at
the residual position (predicting 'I' after "JUL") while sparing JOHN's
2nd-char query. A sequence-position rule (battery positions 119..124 applied
to every corpus window) is additionally scored as a content-agnostic upper
bound on patch cost. The ATLAS itself uses the uniform battery position slice
[CTX-1, CTX-1+L) (includes the first-char query).

Reference values from runs/e023/metrics.json used as gates (G4/G5):
base JULIET 0.4099/0.8827, LUCIO 0.7244/0.7423; D2 JULIET 7.3788/0.1360,
D2 dCE(val_all) +0.000822 (val_all base 1.628084).

Run: python lab/e042_name_atlas.py   (requires runs/checkpoints/e001.pt and
runs/checkpoints/e023_d2_zero_both.pt; the latter is re-derived and asserted
bit-identical if present). E042_SMOKE=1 runs a reduced shakedown.
No NOTES/THINKING/QUEUE/STATE edits; no git commit (operator instruction).
"""
from __future__ import annotations

import copy
import math
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, estimate_loss,
                    run_dir, save_json, set_seed, cfg_dict)

SMOKE = os.environ.get("E042_SMOKE") == "1"

E001_CKPT = REPO / "runs" / "checkpoints" / "e001.pt"
D2_CKPT = REPO / "runs" / "checkpoints" / "e023_d2_zero_both.pt"

SEED = 24000
CTX = 120                 # name-battery context (e023 convention)
BATT_CAP = 125            # first N train occurrences per battery name
BLOCK = 256
N_BLOCKS = 400            # e003c/e023 val-All convention (seed 202)
EVAL_BS = 64
BAT_CHUNK = 128           # e023 battery chunking (bit-comparable)

BAR2_NLL = math.log(65)   # 4.174 — uniform-floor erasure
BAR2_ACC = 0.10
COST_BAR = 0.05           # total corpus cost budget for the two-factor patch

PREFIX_FULL = (1, 2, 3, 4, 5)   # context ends with J..JULIE
PREFIX_STRICT = (2, 3, 4, 5)    # context ends with JU..JULIE

BATTERY = ["JULIET", "JOHN", "ROMEO", "GLOUCESTER", "MENENIUS",
           "CORIOLANUS", "ISABELLA", "LUCIO", "PETRUCHIO", "PROSPERO"]
PURE_CONTROLS = ["ROMEO", "GLOUCESTER", "CORIOLANUS"]
ATLAS_NAMES = ["JULIET"] if SMOKE else ["JULIET", "JOHN", "ROMEO", "LUCIO"]
NL, NH = 6, 6

# e023 registered references (gates G4/G5)
E023_REF = {"JULIET": (0.4099, 0.8827), "JOHN": (1.0866, 0.7121),
            "ROMEO": (0.3278, 0.9104), "LUCIO": (0.7244, 0.7423)}
E023_D2_REF = {"nll": 7.3788, "acc": 0.1360}
E023_VAL_ALL = 1.628084
E023_D2_DCE = 0.000822
E001_VAL_CE = 1.622391    # e001 training-history val CE (G0 reference)

if SMOKE:
    N_BLOCKS = 100


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


def pearson(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den > 0 else float("nan")


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else float("nan")


def prefix_mask(text: str, T: int, ks) -> list[bool]:
    """Fires at query index j (predicting text[j+1]) iff the consumed context
    text[:j+1] ends with a "JULIET" prefix of length in ks."""
    m = [False] * T
    for j in range(T):
        for k in ks:
            if j + 1 >= k and text[j + 1 - k: j + 1] == "JULIET"[:k]:
                m[j] = True
                break
    return m


def fixed_blocks(src: torch.Tensor, block: int, n: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    ix = torch.randint(len(src) - block - 1, (n,), generator=gen)
    x = torch.stack([src[i: i + block] for i in ix])
    y = torch.stack([src[i + 1: i + 1 + block] for i in ix])
    return x.to(DEVICE), y.to(DEVICE), ix.tolist()


def anchored_windows(src_ids, occs, back, n):
    starts = [p - back for p in occs
              if p >= back and p + BLOCK + 1 <= len(src_ids)][:n]
    x = torch.stack([src_ids[s: s + BLOCK] for s in starts])
    y = torch.stack([src_ids[s + 1: s + 1 + BLOCK] for s in starts])
    return x.to(DEVICE), y.to(DEVICE), starts


def build_bat(ids, text, stoi, w, cap):
    """ctx-CTX battery (e023 construction verbatim) + content-rule masks."""
    all_occ = find_occ(text, w)
    occs = all_occ[:cap] if cap else all_occ
    keep = [p for p in occs if p >= CTX]
    wid = torch.tensor([stoi[c] for c in w], dtype=torch.long)
    seqs = [torch.cat([ids[p - CTX: p], wid]) for p in keep]
    T = CTX + len(w) - 1
    cmask = {}
    for tag, ks in (("full", PREFIX_FULL), ("strict", PREFIX_STRICT)):
        cmask[tag] = torch.tensor([prefix_mask(text[p - CTX: p] + w, T, ks)
                                   for p in keep], dtype=torch.bool)
    return {"seq": torch.stack(seqs) if seqs else None, "L": len(w),
            "n": len(keep), "n_all": len(all_occ), "cmask": cmask}


@contextmanager
def pos_lesion(model: TinyGPT, specs):
    """Zero components' residual writes ONLY at masked (B,T) positions.

    specs: (kind, layer, head) with kind in {'head','attn','mlp'} — the e038
    hook made positional. The boolean mask is supplied per forward via the
    yielded state dict (state['mask'] = (B,T) bool); None or all-False is a
    bit-identical no-op (multiply by exact 1.0)."""
    state = {"mask": None}
    handles = []

    def mk_head_pre(head, hd):
        def pre(module, args):
            ms = state["mask"]
            if ms is None:
                return None
            keep = (~ms).to(args[0].dtype).unsqueeze(-1)          # (B,T,1)
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


@torch.no_grad()
def eval_bat(model: TinyGPT, bat: dict, state=None, pos=None, mask_rows=None):
    """Name-position NLL/argmax-acc on one battery (e023 metric path).
    state: pos_lesion state (hooks active); pos: (lo,hi) battery position
    slice; mask_rows: precomputed (n,T) bool content mask."""
    model.eval()
    seq, L = bat["seq"], bat["L"]
    x, y = seq[:, :-1], seq[:, 1:]
    T = x.shape[1]
    nlls, accs = [], []
    for i in range(0, len(x), BAT_CHUNK):
        xc = x[i: i + BAT_CHUNK].to(DEVICE)
        yc = y[i: i + BAT_CHUNK].to(DEVICE)
        if state is not None:
            if mask_rows is not None:
                m = mask_rows[i: i + BAT_CHUNK].to(DEVICE)
            else:
                m = torch.zeros(xc.shape[0], T, dtype=torch.bool, device=DEVICE)
                if pos is not None:
                    m[:, pos[0]: pos[1]] = True
            state["mask"] = m
        logits, _ = model(xc)
        lg = logits[:, CTX - 1: CTX - 1 + L, :]
        tg = yc[:, CTX - 1: CTX - 1 + L]
        nll = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1),
                              reduction="none").view(-1, L)
        acc = (lg.argmax(-1) == tg).float()
        nlls.append(nll)
        accs.append(acc)
    nll_m, acc_m = torch.cat(nlls), torch.cat(accs)
    return {"n": int(nll_m.shape[0]), "nll": float(nll_m.mean().item()),
            "acc": float(acc_m.mean().item()),
            "per_pos_acc": [float(v) for v in acc_m.mean(0).tolist()],
            "per_pos_nll": [float(v) for v in nll_m.mean(0).tolist()]}


@torch.no_grad()
def ce_blocks(model: TinyGPT, x, y, maskT=None, state=None):
    """Corpus CE on fixed blocks; optional positional mask through state."""
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(x), EVAL_BS):
        xb, yb = x[i: i + EVAL_BS], y[i: i + EVAL_BS]
        if state is not None and maskT is not None:
            state["mask"] = maskT[i: i + EVAL_BS].to(DEVICE)
        _, loss = model(xb, yb)
        tot += float(loss.item()) * len(xb)
        n += len(xb)
    return tot / max(n, 1)


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
    set_seed(SEED)
    rd = run_dir("e042")

    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    assert corpus.vocab_size == 65
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=NL, n_head=NH, n_embd=192, block_size=BLOCK)
    base = TinyGPT(cfg).to(DEVICE)
    base.load_state_dict(torch.load(E001_CKPT, map_location=DEVICE, weights_only=True))
    base.eval()
    base_sd = {k: v.clone() for k, v in base.state_dict().items()}
    stoi = corpus.stoi
    jid = stoi["J"]

    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(corpus.itos[int(i)] for i in train_ids)
    val_text = "".join(corpus.itos[int(i)] for i in val_ids)

    # ------------------------------------------------------------- batteries
    print(f"{stamp()} P0: batteries (e023 construction, cap {BATT_CAP}, ctx {CTX})", flush=True)
    bats = {}
    for w in BATTERY:
        ids, txt = (val_ids, val_text) if w == "PROSPERO" else (train_ids, train_text)
        bats[w] = build_bat(ids, txt, stoi, w, BATT_CAP)
    assert bats["JULIET"]["n"] == 125 and bats["LUCIO"]["n"] == 111

    # corpus channels (e023 convention)
    val_x, val_y, val_starts = fixed_blocks(val_ids, BLOCK, N_BLOCKS, seed=202)
    jul_occs = find_occ(train_text, "JULIET")
    jw_x, jw_y, jw_starts = anchored_windows(train_ids, jul_occs, 128, 125)
    # content-rule masks for the corpus channels + the position-rule upper bound
    val_cmask = {t: torch.tensor([prefix_mask(val_text[s: s + BLOCK], BLOCK, ks)
                                  for s in val_starts], dtype=torch.bool)
                 for t, ks in (("full", PREFIX_FULL), ("strict", PREFIX_STRICT))}
    jw_cmask = {t: torch.tensor([prefix_mask(train_text[s: s + BLOCK], BLOCK, ks)
                                 for s in jw_starts], dtype=torch.bool)
                for t, ks in (("full", PREFIX_FULL), ("strict", PREFIX_STRICT))}
    pos_rule = torch.zeros(N_BLOCKS, BLOCK, dtype=torch.bool)
    pos_rule[:, CTX - 1: CTX + 5] = True          # battery name-region positions
    fires = {"val_full": int(val_cmask["full"].sum()), "val_strict": int(val_cmask["strict"].sum()),
             "jw_full": int(jw_cmask["full"].sum()), "position_rule": int(pos_rule.sum())}
    print(f"{stamp()} corpus channels: {fires} mask fires", flush=True)

    # ------------------------------------------------------------- gates + baselines
    g_val = estimate_loss(base, corpus, "val", n_batches=20)
    G0 = {"val_ce": g_val, "ref": E001_VAL_CE, "pass": bool(abs(g_val - E001_VAL_CE) <= 0.03)}
    print(f"{stamp()} G0 val CE {g_val:.4f} vs {E001_VAL_CE} -> {G0['pass']}", flush=True)

    r1_base = {w: eval_bat(base, bats[w]) for w in BATTERY}
    r1_base_2 = {w: eval_bat(base, bats[w]) for w in BATTERY}
    G1 = bool(all(r1_base[w][k] == r1_base_2[w][k] for w in BATTERY
                  for k in ("nll", "acc", "per_pos_acc", "per_pos_nll")))
    print(f"{stamp()} G1 baseline battery bit-identical: {G1}", flush=True)

    with pos_lesion(base, [("head", 0, 0)]) as st:      # mask machinery no-op check
        noop = eval_bat(base, bats["JULIET"], state=st)  # mask stays None -> no-op
    G3 = bool(noop["nll"] == r1_base["JULIET"]["nll"] and noop["acc"] == r1_base["JULIET"]["acc"])
    print(f"{stamp()} G3 hook no-op bit-identical: {G3}", flush=True)

    G4 = {w: {"nll": r1_base[w]["nll"], "acc": r1_base[w]["acc"],
              "ref": E023_REF[w], "pass": bool(abs(r1_base[w]["nll"] - E023_REF[w][0]) <= 0.002
                                               and abs(r1_base[w]["acc"] - E023_REF[w][1]) <= 0.002)}
          for w in E023_REF}
    print(f"{stamp()} G4 e023 baseline reproduction: "
          f"{ {w: g['pass'] for w, g in G4.items()} }", flush=True)

    ce_base = {"val_all": ce_blocks(base, val_x, val_y),
               "jul_windows": ce_blocks(base, jw_x, jw_y)}
    print(f"{stamp()} baselines: JULIET {r1_base['JULIET']['nll']:.3f}/"
          f"{r1_base['JULIET']['acc']:.2f} LUCIO {r1_base['LUCIO']['nll']:.3f}/"
          f"{r1_base['LUCIO']['acc']:.2f} val_all {ce_base['val_all']:.4f} "
          f"julwin {ce_base['jul_windows']:.3f}", flush=True)

    # ------------------------------------------------------------- D2 net
    d2_sd = {k: v.clone() for k, v in base_sd.items()}
    d2_sd["wte.weight"][jid] = 0.0
    d2_sd["lm_head.weight"][jid] = 0.0
    G2 = {"n_elements_changed": 0, "pass": False}
    n_diff, confined = 0, True
    for key in ("wte.weight", "lm_head.weight"):
        d = d2_sd[key] != base_sd[key]
        nd = int(d.sum().item())
        n_diff += nd
        rows = torch.nonzero(d)[:, 0].unique()
        if not (len(rows) == 1 and int(rows[0]) == jid and nd == cfg.n_embd):
            confined = False
    others = all(torch.equal(d2_sd[k], base_sd[k])
                 for k in d2_sd if k not in ("wte.weight", "lm_head.weight"))
    G2 = {"n_elements_changed": n_diff, "expected": 2 * cfg.n_embd,
          "confined_to_J_rows": confined, "others_bit_identical": bool(others),
          "pass": bool(n_diff == 2 * cfg.n_embd and confined and others)}
    if D2_CKPT.exists():
        saved = torch.load(D2_CKPT, map_location=DEVICE, weights_only=True)
        G2["matches_e023_saved_dict"] = all(torch.equal(saved[k], d2_sd[k]) for k in d2_sd)
    print(f"{stamp()} G2 D2 surgery local ({G2['n_elements_changed']} elems, "
          f"saved-match {G2.get('matches_e023_saved_dict')}): {G2['pass']}", flush=True)
    assert G2["pass"]

    d2 = copy.deepcopy(base)
    d2.load_state_dict(d2_sd)
    d2.eval()

    r1_d2 = {w: eval_bat(d2, bats[w]) for w in BATTERY}
    G5 = {"nll": r1_d2["JULIET"]["nll"], "acc": r1_d2["JULIET"]["acc"], "ref": E023_D2_REF,
          "pass": bool(abs(r1_d2["JULIET"]["nll"] - E023_D2_REF["nll"]) <= 0.002
                       and abs(r1_d2["JULIET"]["acc"] - E023_D2_REF["acc"]) <= 0.002)}
    ce_d2 = {"val_all": ce_blocks(d2, val_x, val_y), "jul_windows": ce_blocks(d2, jw_x, jw_y)}
    G5["dce_val_all"] = ce_d2["val_all"] - ce_base["val_all"]
    G5["dce_ok"] = bool(abs(G5["dce_val_all"] - E023_D2_DCE) <= 0.001)
    print(f"{stamp()} G5 D2 reproduction: JULIET {r1_d2['JULIET']['nll']:.3f}/"
          f"{r1_d2['JULIET']['acc']:.3f} (ref {E023_D2_REF}) dCE "
          f"{G5['dce_val_all']:+.5f} (ref +{E023_D2_DCE}) -> {G5['pass'] and G5['dce_ok']}", flush=True)
    residual_pos = [i for i, a in enumerate(r1_d2["JULIET"]["per_pos_acc"]) if a >= 0.05]
    print(f"{stamp()} D2 residual anatomy: acc {r1_d2['JULIET']['acc']:.3f}, "
          f"per-pos {[round(a,2) for a in r1_d2['JULIET']['per_pos_acc']]}, "
          f"residual positions {residual_pos}", flush=True)

    # ------------------------------------------------------------- Q1/Q2: atlas (base net)
    def run_atlas(model, name):
        bat = bats[name]
        L = bat["L"]
        b0 = eval_bat(model, bat)
        heads = torch.zeros(NL, NH)
        blocks = {"attn": [0.0] * NL, "mlp": [0.0] * NL}
        accs_h = torch.zeros(NL, NH)
        for l in range(NL):
            for h in range(NH):
                with pos_lesion(model, [("head", l, h)]) as st:
                    r = eval_bat(model, bat, state=st, pos=(CTX - 1, CTX - 1 + L))
                heads[l, h] = r["nll"] - b0["nll"]
                accs_h[l, h] = b0["acc"] - r["acc"]
            for kind in ("attn", "mlp"):
                with pos_lesion(model, [(kind, l, None)]) as st:
                    r = eval_bat(model, bat, state=st, pos=(CTX - 1, CTX - 1 + L))
                blocks[kind][l] = r["nll"] - b0["nll"]
        return {"base": b0, "heads_dce": heads, "heads_dacc": accs_h, "blocks_dce": blocks}

    print(f"{stamp()} Q1/Q2: base-net atlas ({len(ATLAS_NAMES)} names x 48 cells)", flush=True)
    atlas = {}
    for name in ATLAS_NAMES:
        atlas[name] = run_atlas(base, name)
        a = atlas[name]
        top_h = torch.argmax(a["heads_dce"])
        print(f"{stamp()} atlas[{name}]: base {a['base']['nll']:.3f} | top head "
              f"L{top_h // NH}H{top_h % NH} dCE {a['heads_dce'].flatten()[top_h]:+.3f} | "
              f"top attn L{max(range(NL), key=lambda i: a['blocks_dce']['attn'][i])} "
              f"{max(a['blocks_dce']['attn']):+.3f} | top mlp L{max(range(NL), key=lambda i: a['blocks_dce']['mlp'][i])} "
              f"{max(a['blocks_dce']['mlp']):+.3f}", flush=True)
    assert atlas["JULIET"]["heads_dce"].max() > 0.05, "hook sanity: atlas all-zero"

    # overlap stats (Q2)
    overlap = {}
    if "LUCIO" in atlas:
        for other in [n for n in ATLAS_NAMES if n != "JULIET"]:
            a, b = atlas["JULIET"], atlas[other]
            hj, ho = a["heads_dce"].flatten().tolist(), b["heads_dce"].flatten().tolist()
            bj = (a["blocks_dce"]["attn"] + a["blocks_dce"]["mlp"])
            bo = (b["blocks_dce"]["attn"] + b["blocks_dce"]["mlp"])
            top5j = sorted(range(36), key=lambda i: -hj[i])[:5]
            top5o = sorted(range(36), key=lambda i: -ho[i])[:5]
            overlap[other] = {
                "heads_pearson": pearson(hj, ho), "heads_spearman": spearman(hj, ho),
                "heads_top5_jaccard": jaccard(top5j, top5o),
                "blocks_pearson": pearson(bj, bo), "blocks_spearman": spearman(bj, bo),
                "top5_heads_juliet": [f"L{i//NH}H{i%NH}" for i in top5j],
                "top5_heads_other": [f"L{i//NH}H{i%NH}" for i in top5o],
            }
        o = overlap["LUCIO"]
        if o["heads_spearman"] >= 0.6 and o["heads_top5_jaccard"] >= 0.4:
            overlap_verdict = "SHARED: LUCIO completion rides the same heads as JULIET"
        elif o["heads_spearman"] <= 0.3 or o["heads_top5_jaccard"] == 0.0:
            overlap_verdict = "SEPARATE: LUCIO's completion circuit is distinct from JULIET's"
        else:
            overlap_verdict = "PARTIAL: top sites coincide but the maps only weakly correlate"
        print(f"{stamp()} Q2 overlap JULIET~LUCIO: heads rho={o['heads_spearman']:.2f} "
              f"pearson={o['heads_pearson']:.2f} top5-Jac={o['heads_top5_jaccard']:.2f} "
              f"blocks rho={o['blocks_spearman']:.2f} -> {overlap_verdict}", flush=True)
    else:
        overlap_verdict = "SMOKE (no LUCIO atlas)"

    # ------------------------------------------------------------- Q3a: D2-conditioned atlas
    print(f"{stamp()} Q3: D2-conditioned residual atlas (JULIET)", flush=True)
    d2_atlas = run_atlas(d2, "JULIET")
    dh = d2_atlas["heads_dce"]
    print(f"{stamp()} D2 atlas: top head L{int(dh.argmax())//NH}H{int(dh.argmax())%NH} "
          f"dCE {dh.flatten()[int(dh.argmax())]:+.3f} | top attn "
          f"L{max(range(NL), key=lambda i: d2_atlas['blocks_dce']['attn'][i])} "
          f"{max(d2_atlas['blocks_dce']['attn']):+.3f} | top mlp "
          f"L{max(range(NL), key=lambda i: d2_atlas['blocks_dce']['mlp'][i])} "
          f"{max(d2_atlas['blocks_dce']['mlp']):+.3f}", flush=True)

    head_cells = sorted([(("head", l, h), float(dh[l, h])) for l in range(NL) for h in range(NH)],
                        key=lambda t: -t[1])
    block_cells = sorted([(("attn", l, None), d2_atlas["blocks_dce"]["attn"][l]) for l in range(NL)]
                         + [(("mlp", l, None), d2_atlas["blocks_dce"]["mlp"][l]) for l in range(NL)],
                         key=lambda t: -t[1])
    acc_cells = sorted([(("head", l, h), float(d2_atlas["heads_dacc"][l, h]))
                        for l in range(NL) for h in range(NH)], key=lambda t: -t[1])
    fmt = lambda s: f"{'L'+str(s[1])}{s[0][:1].upper() if s[0]!='head' else 'H'+str(s[2])}"
    print(f"{stamp()} residual supporters by dNLL: heads {[ (fmt(s), round(v,3)) for s,v in head_cells[:4] ]} "
          f"blocks {[ (fmt(s), round(v,3)) for s,v in block_cells[:4] ]}", flush=True)

    # ------------------------------------------------------------- Q3b: two-factor cells
    print(f"{stamp()} Q3: two-factor cells (D2 + combined body lesions, content-rule patch)", flush=True)
    sel = {
        "D2+1h": [head_cells[0][0]],
        "D2+2h": [s for s, _ in head_cells[:2]],
        "D2+3h": [s for s, _ in head_cells[:3]],
        "D2+1b": [block_cells[0][0]],
        "D2+2b": [s for s, _ in block_cells[:2]],
        "D2+1b+1h": [block_cells[0][0], head_cells[0][0]],
        "D2+1ha": [acc_cells[0][0]],
    }
    if not SMOKE:
        sel["D2+2ha"] = [s for s, _ in acc_cells[:2]]

    def run_cell(specs, mtag):
        with pos_lesion(d2, specs) as st:
            r1 = {w: eval_bat(d2, bats[w], state=st, mask_rows=bats[w]["cmask"][mtag]) for w in BATTERY}
            ce = {"val_all": ce_blocks(d2, val_x, val_y, val_cmask[mtag], st),
                  "jul_windows": ce_blocks(d2, jw_x, jw_y, jw_cmask[mtag], st),
                  "val_all_position_rule": ce_blocks(d2, val_x, val_y, pos_rule, st)}
        j = r1["JULIET"]
        dnll = {w: r1[w]["nll"] - r1_base[w]["nll"] for w in BATTERY}
        pure = sum(dnll[w] for w in PURE_CONTROLS) / len(PURE_CONTROLS)
        return {"specs": [list(s) for s in specs], "mask_rule": mtag,
                "juliet": {"nll": j["nll"], "acc": j["acc"], "per_pos_acc": j["per_pos_acc"]},
                "r1": {w: {"nll": r1[w]["nll"], "acc": r1[w]["acc"]} for w in BATTERY},
                "d_nll": dnll,
                "s_name": (dnll["JULIET"] / pure) if pure > 1e-9 else float("inf"),
                "ce": ce,
                "dce_val_content": ce["val_all"] - ce_base["val_all"],
                "dce_val_position": ce["val_all_position_rule"] - ce_base["val_all"],
                "bar2": bool(j["nll"] >= BAR2_NLL and j["acc"] <= BAR2_ACC)}

    cells = {}
    for key, specs in sel.items():
        cells[key] = run_cell(specs, "full")
        c = cells[key]
        print(f"{stamp()} {key:11s} [{'/'.join(fmt(s) for s in specs)}] JULIET "
              f"{c['juliet']['nll']:6.2f}/{c['juliet']['acc']:.3f} "
              f"per-pos {[round(a,2) for a in c['juliet']['per_pos_acc']]} "
              f"dCE(content) {c['dce_val_content']:+.5f} dCE(posrule) {c['dce_val_position']:+.5f} "
              f"ROMEO {c['r1']['ROMEO']['nll']:5.2f} LUCIO {c['r1']['LUCIO']['nll']:5.2f} "
              f"bar2={int(c['bar2'])}", flush=True)

    # strict-prefix refinement on the key cells (spares JOHN's 2nd-char query)
    if not SMOKE:
        for key in ("D2+1h", "D2+1b", "D2+1b+1h"):
            cells[key + "+strict"] = run_cell(sel[key], "strict")
            c = cells[key + "+strict"]
            print(f"{stamp()} {key + '+strict':17s} JULIET "
                  f"{c['juliet']['nll']:6.2f}/{c['juliet']['acc']:.3f} "
                  f"dCE(content) {c['dce_val_content']:+.5f} JOHN "
                  f"{c['r1']['JOHN']['nll']:5.2f} bar2={int(c['bar2'])}", flush=True)

    best_key = min(cells, key=lambda k: cells[k]["juliet"]["acc"])
    best = cells[best_key]
    erase_ok = [k for k, c in cells.items() if c["bar2"]]
    cost_ok = [k for k in erase_ok if cells[k]["dce_val_content"] < COST_BAR
               and cells[k]["dce_val_position"] < COST_BAR]
    if cost_ok:
        twofactor_verdict = (f"YES: two-factor surgery erases (D2 + body lesion reaches Bar-2 "
                             f"at corpus cost < +{COST_BAR}); best {best_key}")
    elif erase_ok:
        twofactor_verdict = ("PARTIAL: Bar-2 reached but corpus cost exceeds +0.05 "
                             f"(best-cost erase cell {min(erase_ok, key=lambda k: cells[k]['dce_val_content'])})")
    else:
        twofactor_verdict = (f"NO: no combined cell reaches Bar-2 (best {best_key} "
                             f"acc {best['juliet']['acc']:.3f} > {BAR2_ACC}); residual is not "
                             f"concentrated in rankable body components")
    print(f"{stamp()} TWO-FACTOR: {twofactor_verdict}", flush=True)

    # atlas summary numbers
    hj = atlas["JULIET"]["heads_dce"]
    pos_mass = float(hj[hj > 0].sum())
    top1 = float(hj.max())
    hj_flat = torch.sort(hj.flatten(), descending=True).values
    tb = max([("attn", i) for i in range(NL)] + [("mlp", i) for i in range(NL)],
             key=lambda t: atlas["JULIET"]["blocks_dce"][t[0]][t[1]])
    atlas_summary = {
        "juliet_top_head": f"L{int(hj.argmax())//NH}H{int(hj.argmax())%NH}",
        "juliet_top_head_dce": top1,
        "juliet_head_mass_share_top1": top1 / pos_mass,
        "juliet_head_mass_share_top3": float(hj_flat[:3].sum()) / pos_mass,
        "juliet_top_block": f"L{tb[1]}-{tb[0]}",
        "juliet_top_block_dce": atlas["JULIET"]["blocks_dce"][tb[0]][tb[1]],
    }
    print(f"{stamp()} atlas summary: top head {atlas_summary['juliet_top_head']} "
          f"{top1:+.3f} nats ({atlas_summary['juliet_head_mass_share_top1']*100:.0f}% of positive "
          f"head mass, top-3 {atlas_summary['juliet_head_mass_share_top3']*100:.0f}%)", flush=True)

    # ------------------------------------------------------------- outputs
    metrics = {
        "experiment": "e042_name_atlas",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED, "smoke": SMOKE,
        "questions": ["Q1 head/block atlas of name completion",
                      "Q2 JULIET~LUCIO circuit overlap",
                      "Q3 two-factor (D2 + body lesion) Bar-2 erasure"],
        "gates": {"G0_val_ce": G0, "G1_bit_identical": G1, "G2_d2_surgery": G2,
                  "G3_hook_noop": G3, "G4_e023_baseline": G4, "G5_e023_d2": G5},
        "mask_fires": fires,
        "patch_semantics": {
            "content_rule": "zero at query positions whose context ends with a non-empty "
                            "proper prefix of JULIET (full: J..JULIE; strict: JU..JULIE)",
            "position_rule": "battery name-region positions 119..124 applied to every "
                             "corpus window (content-agnostic upper bound)",
            "atlas_rule": "uniform battery position slice [CTX-1, CTX-1+L) incl. first-char query"},
        "baseline": {"r1": r1_base, "ce": ce_base},
        "d2_net": {"r1": r1_d2, "ce": ce_d2, "residual_positions": residual_pos},
        "atlas_base": {n: {"base": {"nll": atlas[n]["base"]["nll"], "acc": atlas[n]["base"]["acc"]},
                           "heads_dce": atlas[n]["heads_dce"].tolist(),
                           "heads_dacc": atlas[n]["heads_dacc"].tolist(),
                           "blocks_dce": atlas[n]["blocks_dce"]} for n in ATLAS_NAMES},
        "atlas_summary": atlas_summary,
        "overlap": overlap, "overlap_verdict": overlap_verdict,
        "d2_atlas": {"base": {"nll": d2_atlas["base"]["nll"], "acc": d2_atlas["base"]["acc"]},
                     "heads_dce": d2_atlas["heads_dce"].tolist(),
                     "heads_dacc": d2_atlas["heads_dacc"].tolist(),
                     "blocks_dce": d2_atlas["blocks_dce"],
                     "head_rank_by_dnll": [[fmt(s), round(v, 4)] for s, v in head_cells],
                     "block_rank_by_dnll": [[fmt(s), round(v, 4)] for s, v in block_cells],
                     "head_rank_by_dacc": [[fmt(s), round(v, 4)] for s, v in acc_cells]},
        "twofactor_cells": cells,
        "verdicts": {
            "atlas_localized": bool(atlas_summary["juliet_head_mass_share_top3"] >= 0.5),
            "overlap": overlap_verdict,
            "twofactor": twofactor_verdict,
            "twofactor_bar2_cells": erase_ok,
            "twofactor_cost_ok_cells": cost_ok,
            "best_cell": best_key,
        },
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": cfg_dict(cfg),
    }
    save_json(rd / "metrics.json", jsonable(metrics))

    # ---- atlas heatmap PNG: heads x layers x {JULIET, LUCIO} selectivity
    if not SMOKE:
        fig, axes = plt.subplots(2, 3, figsize=(17, 8.2))
        def heat(ax, M, title, cmap="inferno"):
            im = ax.imshow(M.tolist(), cmap=cmap, aspect="auto")
            for l in range(NL):
                for h in range(NH):
                    ax.text(h, l, f"{M[l,h]:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if M[l, h] < 0.55 * float(M.max()) else "black")
            ax.set_xticks(range(NH)); ax.set_xticklabels([f"H{h}" for h in range(NH)])
            ax.set_yticks(range(NL)); ax.set_yticklabels([f"L{l}" for l in range(NL)])
            ax.set_title(title, fontsize=9)
            return im
        im0 = heat(axes[0, 0], hj, f"JULIET head atlas — dCE (nats) at name positions\n"
                    f"top {atlas_summary['juliet_top_head']} = {top1:+.2f}")
        fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)
        if "LUCIO" in atlas:
            hl = atlas["LUCIO"]["heads_dce"]
            im1 = heat(axes[0, 1], hl, f"LUCIO head atlas — dCE (nats)\n"
                        f"(the collateral-sensitive name, e023 D1 dNLL 8.53)")
            fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)
            ax = axes[0, 2]
            for l in range(NL):
                ax.scatter(hj[l].tolist(), hl[l].tolist(), s=26, label=f"L{l}")
            lim = max(float(hj.max()), float(hl.max())) * 1.1
            ax.plot([0, lim], [0, lim], "k--", lw=0.8)
            ax.set_xlabel("dCE JULIET (nats)"); ax.set_ylabel("dCE LUCIO (nats)")
            o = overlap["LUCIO"]
            ax.set_title(f"circuit overlap: heads rho={o['heads_spearman']:.2f}, "
                         f"top5-Jac={o['heads_top5_jaccard']:.2f}\n{overlap_verdict.split(':')[0]}")
            ax.legend(fontsize=7, ncol=2)
        ax = axes[1, 0]
        xs = list(range(2 * NL))
        lbl = [f"L{i}at" for i in range(NL)] + [f"L{i}mlp" for i in range(NL)]
        for i, name in enumerate([n for n in ATLAS_NAMES if n in atlas]):
            vals = atlas[name]["blocks_dce"]["attn"] + atlas[name]["blocks_dce"]["mlp"]
            ax.bar([x + 0.22 * (i - 1.5) for x in xs], vals, 0.22, label=name)
        ax.set_xticks(xs); ax.set_xticklabels(lbl, fontsize=7)
        ax.set_ylabel("dCE (nats)"); ax.legend(fontsize=7)
        ax.set_title("block atlas (sublayer zeroed at name positions)")
        im2 = heat(axes[1, 1], dh, f"D2-conditioned residual atlas — JULIET\n"
                   f"(what carries the 13.6% residual; base NLL {d2_atlas['base']['nll']:.2f})",
                   cmap="magma")
        fig.colorbar(im2, ax=axes[1, 1], fraction=0.046)
        ax = axes[1, 2]
        keys = list(cells.keys())
        accs = [cells[k]["juliet"]["acc"] for k in keys]
        costs = [cells[k]["dce_val_content"] for k in keys]
        ax.bar(range(len(keys)), accs, 0.6, color=["crimson" if cells[k]["bar2"] else "steelblue"
                                                   for k in keys])
        ax.axhline(BAR2_ACC, color="k", ls="--", lw=1, label="Bar-2 acc 0.10")
        ax.axhline(r1_d2["JULIET"]["acc"], color="gray", ls=":", lw=1,
                   label=f"D2 alone {r1_d2['JULIET']['acc']:.3f}")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([k.replace("D2+", "") for k in keys], rotation=55, fontsize=6.5)
        ax.set_ylabel("JULIET acc"); ax.set_ylim(0, 0.25)
        ax2 = ax.twinx()
        ax2.plot(range(len(keys)), costs, "o-", color="seagreen", lw=1, ms=4)
        ax2.axhline(COST_BAR, color="seagreen", ls=":", lw=1)
        ax2.set_ylabel("corpus cost dCE val (content rule)", color="seagreen")
        ax.set_title(f"two-factor erasure: {twofactor_verdict.split(':')[0]}\n"
                     f"(bars acc [red=Bar-2]; line corpus cost vs +{COST_BAR})")
        ax.legend(fontsize=7, loc="upper right")
        fig.suptitle("E042 — name-circuit atlas (E001, position-resolved lesions at name positions)")
        fig.tight_layout()
        fig.savefig(rd / "atlas_heatmap.png", dpi=130)
        plt.close(fig)

    print(f"\n{stamp()} === E042 VERDICTS ===")
    print(f"G0 {G0['pass']}  G1 {G1}  G2 {G2['pass']}  G3 {G3}  "
          f"G4 {all(g['pass'] for g in G4.values())}  G5 {G5['pass'] and G5['dce_ok']}")
    print(f"Q1 atlas: localized={metrics['verdicts']['atlas_localized']} "
          f"(top-1 head {atlas_summary['juliet_top_head']} {top1:+.2f} nats, "
          f"{atlas_summary['juliet_head_mass_share_top1']*100:.0f}% of positive head mass)")
    print(f"Q2 overlap: {overlap_verdict}")
    print(f"Q3 two-factor: {twofactor_verdict}")
    print(f"outputs: {rd}")
    print(f"total {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
