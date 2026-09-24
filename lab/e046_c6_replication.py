"""E046 — C6 replication on second/third nets: two-factor erasure with
IN-RUN head discovery + the R5 missing observations (J-census prefix-leak,
uniform-floor name battery).

Review 5 (INTERPRETER, accepted): "T013's 'one body head' is seed-fragile —
e043's memo measured L3H5 = -0.02 in B43; only the L0-MLP BLOCK is invariant.
C6 degrades to 'rows + a seed-specific head found by in-run atlas' until
replicated (e046 registered: B43/BDO D2-analog + J-census + uniform-floor
battery). Prefix-leak (JULIUS/Jove/John-class continuations touched by the
content rule) unmeasured in e042."

Hosts (no training, eval-only):
  B43 = runs/checkpoints/e028_b43.pt — DIFFERENT INIT (seed 43), same regime
        as B (val 1.5696, e028). Memo (healthy JULIET atlas): L0H5 #1 +1.18,
        L0H3 +0.01, L3H5 -0.02.
  BDO = runs/checkpoints/e041_bdo.pt — SAME INIT as B (seed 42), different
        data order (corpus seed 7777; val 1.595, e041). Memo: L1H0 #1 +1.11,
        L0H3 +0.26.

Protocol per net:
  1. In-run atlas FIRST: D2-analog (zero JULIET's rare-letter rows = the 'J'
     rows, wte + lm_head, 384 params — same corpus, J is the rare core in all
     nets) then the e042 D2-CONDITIONED position-resolved head atlas (36
     heads + 12 blocks) on the JULIET train battery -> THAT net's own top
     residual-carrier head. (The recipe instrument is the D2-conditioned
     atlas, e042 Q3a — the memo's healthy-net numbers are cited for
     comparison, not used for discovery.)
  2. Two-factor cells (4 total = 2 nets x 2 recipes):
       OWN      = D2 + own-top head, content rule (context ends with a
                  non-empty proper prefix of JULIET: J..JULIE; e042's
                  deployed patch semantics, pos_lesion verbatim);
       B-L3H5   = D2 + B's L3H5 under the same rule (the transplanted
                  recipe; expected NO on B43 per the memo's -0.02).
  3. Full honesty battery: train-derived name battery (e023 construction),
     UNIFORM-FLOOR constructed name battery (R5: not just train-derived),
     J-census (12 e023 words, uncapped), incumbent names, corpus CE channels
     (val-All seed 202 x400, JULIET-anchored train windows, position-rule
     upper bound).
  4. Verdicts: (a) replication with in-run discovery = Bar-2 (JULIET NLL >=
     ln65 AND acc <= 0.10) on BOTH batteries at corpus cost <= +0.05;
     (b) does B's L3H5 recipe transfer (Bar-2 on that net); (c) content-rule
     leak = Julius-class EXTRA dNLL over D2-alone > 0.5 nats flags it
     (D2 itself costs J-words +1.4..+7.0 nats BY DESIGN — e023 — so the leak
     readout is the patch's marginal effect, not the row-zero's).

DEVIATIONS / SPEC-GAP FILLS:
  1. "Uniform-floor battery" construction was named by R5 but not specified:
     each battery name is spliced into the 63 registered-anchor (PROSPERO)
     VAL occurrence contexts (120 chars, e023's PROSPERO battery that read
     4.456 ~ ln65 = zero-knowledge on B). Same 63 contexts for every name ->
     context-independent completion readout. In-run gate: PROSPERO NLL in
     [3.5, 5.5] per net.
  2. The leak flag word is "Julius" (3 train occurrences = 18 predictions;
     uppercase "JULIUS" has 0 occurrences) — the memo's "JULIUS" =
     Julius-class. "Juliet" (48 occ; shares all 5 prefixes) is the maximal-
     leak word and is reported alongside; Jove/JOHN/John/Jack/Juno/etc.
     (share only the 'J' onset) complete the census.
  3. Only the FULL prefix rule (J..JULIE) is run — e042's deployed recipe;
     its strict variant (JU..JULIE) changed nothing material on B and is not
     re-registered here.
  4. If a net's own-top head == L3H5 the OWN and B-L3H5 cells coincide; they
     are then scored once and noted (did not occur: B43/BDO own-tops
     differ — see metrics).
  5. Healthy-net atlas not re-run (memo already measured it on both hosts);
     D2-conditioned atlas is the discovery instrument.
  6. No NOTES/THINKING/QUEUE/STATE edits; no git commit (operator
     instruction).
  7. SUPPLEMENTARY cells (not in the registered 4-cell set, added after the
     smoke showed the single-head recipe failing on both hosts): D2 + that
     net's own top BLOCK (attn/mlp sublayer) under the same content rule —
     distinguishes "residual distributed across heads" from "residual needs
     the block" (on B, e042's D2+1b block cell also reached Bar-2). They are
     tagged "-supp" everywhere and never replace the registered readouts.

Run: python lab/e046_c6_replication.py   (requires runs/checkpoints/{e028_b43,e041_bdo}.pt;
runs/e042/metrics.json loaded opportunistically for B's reference row).
E046_SMOKE=1 runs a reduced shakedown. Budget: eval-only, < 12 min.
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

SMOKE = os.environ.get("E046_SMOKE") == "1"

B43_CKPT = REPO / "runs" / "checkpoints" / "e028_b43.pt"
BDO_CKPT = REPO / "runs" / "checkpoints" / "e041_bdo.pt"
E042_METRICS = REPO / "runs" / "e042" / "metrics.json"

SEED = 24600
CTX = 120                 # name-battery context (e023/e042 convention)
BATT_CAP = 125            # first N train occurrences per battery name
BLOCK = 256
N_BLOCKS = 100 if SMOKE else 400   # e023/e042 val-All convention (seed 202)
EVAL_BS = 64
BAT_CHUNK = 128

BAR2_NLL = math.log(65)   # 4.174 — uniform-floor erasure
BAR2_ACC = 0.10
COST_BAR = 0.05           # two-factor corpus cost budget (e042)
LEAK_BAR = 0.5            # Julius-class extra dNLL over D2 (R5 flag)
ANCHOR_LO, ANCHOR_HI = 3.5, 5.5     # uniform-floor anchor gate around ln65

B_HEAD = (3, 5)           # B's recipe: L3H5 (e042 D2-atlas top, +0.8786)

PREFIX_FULL = (1, 2, 3, 4, 5)   # context ends with J..JULIE
BATTERY = ["JULIET", "JOHN", "ROMEO", "GLOUCESTER", "MENENIUS",
           "CORIOLANUS", "ISABELLA", "LUCIO", "PETRUCHIO", "PROSPERO"]
PURE_CONTROLS = ["ROMEO", "GLOUCESTER", "CORIOLANUS"]
CENSUS = ["JULIET", "Juliet", "JOHN", "John", "Jove", "Jack", "Jesu",
          "Justice", "Jupiter", "Join", "Juno", "Julius"]
LEAK_WORDS = ["Julius", "Juliet", "Jove", "JOHN", "John", "Jesu", "Juno"]
NL, NH = 6, 6

# e042 references (net B) — hard checkpoint row for the report
E042_REF = {"d2_atlas_top_head": "L3H5", "d2_atlas_top_dce": 0.8786,
            "best_cell": "D2+1h", "juliet_nll": 8.715, "juliet_acc": 0.0013,
            "dce_val_content": 0.00083, "dce_val_position": 0.00227,
            "d2_per_pos_acc": [0.0, 0.0, 0.0, 0.82, 0.0, 0.0],
            "memo_healthy_atlas": {"B": "L0H3 #1 +1.52 / L3H5 #2 +0.58",
                                   "B43": "L0H5 #1 +1.18 / L0H3 +0.01 / L3H5 -0.02",
                                   "BDO": "L1H0 #1 +1.11 / L0H3 +0.26"}}


# ------------------------------------------------------------------ helpers

def find_occ(text: str, word: str) -> list[int]:
    pat = re.compile(r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])")
    return [m.start() for m in pat.finditer(text)]


def prefix_mask(text: str, T: int, ks) -> list[bool]:
    """Fires at query index j (predicting text[j+1]) iff text[:j+1] ends with
    a "JULIET" prefix of length in ks. (e042 verbatim.)"""
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


def build_bat_train(ids, text, stoi, w, cap):
    """e023/e042 train-derived battery: ctx-CTX local context + word."""
    all_occ = find_occ(text, w)
    occs = all_occ[:cap] if cap else all_occ
    keep = [p for p in occs if p >= CTX]
    wid = torch.tensor([stoi[c] for c in w], dtype=torch.long)
    seqs = [torch.cat([ids[p - CTX: p], wid]) for p in keep]
    T = CTX + len(w) - 1
    cmask = torch.tensor([prefix_mask(text[p - CTX: p] + w, T, PREFIX_FULL)
                          for p in keep], dtype=torch.bool) if seqs else None
    return {"seq": torch.stack(seqs) if seqs else None, "L": len(w),
            "n": len(keep), "n_all": len(all_occ), "cmask": {"full": cmask}}


def build_bat_uniform(val_ids, val_text, anchor_occs, stoi, w):
    """Uniform-floor battery: w spliced into the PROSPERO val contexts."""
    keep = [p for p in anchor_occs if p >= CTX]
    wid = torch.tensor([stoi[c] for c in w], dtype=torch.long)
    seqs = [torch.cat([val_ids[p - CTX: p], wid]) for p in keep]
    T = CTX + len(w) - 1
    cmask = torch.tensor([prefix_mask(val_text[p - CTX: p] + w, T, PREFIX_FULL)
                          for p in keep], dtype=torch.bool) if seqs else None
    return {"seq": torch.stack(seqs) if seqs else None, "L": len(w),
            "n": len(keep), "n_all": len(anchor_occs), "cmask": {"full": cmask}}


@contextmanager
def pos_lesion(model: TinyGPT, specs):
    """Zero components' residual writes ONLY at masked (B,T) positions.
    (e042 verbatim; state['mask'] supplied per forward; None = bit-identical
    no-op.)"""
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
    """Name-position NLL/argmax-acc on one battery (e023 metric path, e042
    verbatim)."""
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
    rd = run_dir("e046")

    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    assert corpus.vocab_size == 65
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=NL, n_head=NH, n_embd=192, block_size=BLOCK)
    stoi = corpus.stoi
    jid = stoi["J"]

    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(corpus.itos[int(i)] for i in train_ids)
    val_text = "".join(corpus.itos[int(i)] for i in val_ids)

    # ------------------------------------------------------------- batteries
    print(f"{stamp()} P0: batteries (train-derived, uniform-floor, J-census)", flush=True)
    train_bats, unif_bats, cens_bats = {}, {}, {}
    anchor_occs = find_occ(val_text, "PROSPERO")
    for w in BATTERY:
        ids, txt = (val_ids, val_text) if w == "PROSPERO" else (train_ids, train_text)
        train_bats[w] = build_bat_train(ids, txt, stoi, w, BATT_CAP)
        unif_bats[w] = build_bat_uniform(val_ids, val_text, anchor_occs, stoi, w)
    for w in CENSUS:
        cens_bats[w] = build_bat_train(train_ids, train_text, stoi, w, cap=None)
    assert train_bats["JULIET"]["n"] == 125 and unif_bats["PROSPERO"]["n"] == 63

    # corpus channels (e023/e042 convention) + content-rule masks
    val_x, val_y, val_starts = fixed_blocks(val_ids, BLOCK, N_BLOCKS, seed=202)
    jul_occs = find_occ(train_text, "JULIET")
    jw_x, jw_y, jw_starts = anchored_windows(train_ids, jul_occs, 128, 125)
    val_cmask = torch.tensor([prefix_mask(val_text[s: s + BLOCK], BLOCK, PREFIX_FULL)
                              for s in val_starts], dtype=torch.bool)
    jw_cmask = torch.tensor([prefix_mask(train_text[s: s + BLOCK], BLOCK, PREFIX_FULL)
                             for s in jw_starts], dtype=torch.bool)
    pos_rule = torch.zeros(N_BLOCKS, BLOCK, dtype=torch.bool)
    pos_rule[:, CTX - 1: CTX + 5] = True
    fires = {"val_full": int(val_cmask.sum()), "jw_full": int(jw_cmask.sum()),
             "position_rule": int(pos_rule.sum()),
             "juliet_train_bat": int(train_bats["JULIET"]["cmask"]["full"].sum()),
             "juliet_unif_bat": int(unif_bats["JULIET"]["cmask"]["full"].sum()),
             "census_Julius": int(cens_bats["Julius"]["cmask"]["full"].sum())}
    print(f"{stamp()} mask fires: {fires}", flush=True)

    # B reference row from e042 (opportunistic, for the report only)
    if E042_METRICS.exists():
        import json
        e042 = json.loads(E042_METRICS.read_text(encoding="utf-8"))
        E042_REF["d2_atlas_top_head"] = e042["d2_atlas"]["head_rank_by_dnll"][0][0]
        E042_REF["d2_atlas_top_dce"] = e042["d2_atlas"]["head_rank_by_dnll"][0][1]
        c = e042["twofactor_cells"]["D2+1h"]
        E042_REF.update({"juliet_nll": c["juliet"]["nll"], "juliet_acc": c["juliet"]["acc"],
                         "dce_val_content": c["dce_val_content"],
                         "dce_val_position": c["dce_val_position"],
                         "d2_per_pos_acc": e042["d2_net"]["r1"]["JULIET"]["per_pos_acc"]})
    print(f"{stamp()} B reference (e042): top {E042_REF['d2_atlas_top_head']} "
          f"+{E042_REF['d2_atlas_top_dce']}, D2+top -> acc {E042_REF['juliet_acc']} "
          f"at dCE +{E042_REF['dce_val_content']}", flush=True)

    # ------------------------------------------------------------- per-net work
    def eval_all(model, state=None, bats=True, census=True, ce=True):
        out = {}
        if bats:
            out["train_bat"] = {w: eval_bat(model, train_bats[w], state=state,
                                            mask_rows=train_bats[w]["cmask"]["full"])
                                for w in BATTERY}
            out["unif_bat"] = {w: eval_bat(model, unif_bats[w], state=state,
                                           mask_rows=unif_bats[w]["cmask"]["full"])
                               for w in BATTERY}
        if census:
            out["census"] = {w: eval_bat(model, cens_bats[w], state=state,
                                         mask_rows=cens_bats[w]["cmask"]["full"])
                             for w in CENSUS if cens_bats[w]["n"] > 0}
        if ce:
            out["ce"] = {"val_all": ce_blocks(model, val_x, val_y,
                                              val_cmask if state is not None else None, state),
                         "jul_windows": ce_blocks(model, jw_x, jw_y,
                                                  jw_cmask if state is not None else None, state),
                         "val_all_position_rule": ce_blocks(model, val_x, val_y,
                                                            pos_rule if state is not None else None, state)}
        return out

    def run_atlas_d2(model, name):
        """e042 Q3a instrument: D2-conditioned atlas on the JULIET battery."""
        bat = train_bats[name]
        L = bat["L"]
        b0 = eval_bat(model, bat)
        heads = torch.zeros(NL, NH)
        blocks = {"attn": [0.0] * NL, "mlp": [0.0] * NL}
        for l in range(NL):
            for h in range(NH):
                with pos_lesion(model, [("head", l, h)]) as st:
                    r = eval_bat(model, bat, state=st, pos=(CTX - 1, CTX - 1 + L))
                heads[l, h] = r["nll"] - b0["nll"]
            for kind in ("attn", "mlp"):
                with pos_lesion(model, [(kind, l, None)]) as st:
                    r = eval_bat(model, bat, state=st, pos=(CTX - 1, CTX - 1 + L))
                blocks[kind][l] = r["nll"] - b0["nll"]
        return {"base": b0, "heads_dce": heads, "blocks_dce": blocks}

    nets = {}
    for net_name, ckpt in (("B43", B43_CKPT), ("BDO", BDO_CKPT)):
        print(f"\n{stamp()} ===== host {net_name} ({ckpt.name}) =====", flush=True)
        net = TinyGPT(cfg).to(DEVICE)
        net.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        net.eval()
        net_sd = {k: v.clone() for k, v in net.state_dict().items()}

        # gates + baselines
        g_val = estimate_loss(net, corpus, "val", n_batches=20)
        base = eval_all(net)
        base_2 = eval_all(net, bats=True, census=False, ce=False)
        G1 = bool(all(base["train_bat"][w][k] == base_2["train_bat"][w][k]
                      and base["unif_bat"][w][k] == base_2["unif_bat"][w][k]
                      for w in BATTERY for k in ("nll", "acc")))
        with pos_lesion(net, [("head", 0, 0)]) as st:
            noop = eval_bat(net, train_bats["JULIET"], state=st)  # mask None -> no-op
        G3 = bool(noop["nll"] == base["train_bat"]["JULIET"]["nll"]
                  and noop["acc"] == base["train_bat"]["JULIET"]["acc"])
        G4 = bool(base["train_bat"]["JULIET"]["nll"] <= 1.0
                  and base["train_bat"]["JULIET"]["acc"] >= 0.5)
        G5 = bool(ANCHOR_LO <= base["unif_bat"]["PROSPERO"]["nll"] <= ANCHOR_HI)
        gates = {"G0_val_ce": {"val": g_val, "ref": {"B43": 1.5696, "BDO": 1.595}[net_name],
                               "pass": bool(g_val < 2.0)},
                 "G1_bit_identical": G1, "G3_hook_noop": G3,
                 "G4_juliet_memorized": G4,
                 "G5_uniform_floor_anchor": {
                     "prospero_unif_nll": base["unif_bat"]["PROSPERO"]["nll"],
                     "gate": [ANCHOR_LO, ANCHOR_HI], "ln65": BAR2_NLL, "pass": G5}}
        print(f"{stamp()} G0 val {g_val:.4f}  G1 {G1}  G3 {G3}  G4 {G4}  "
              f"G5 anchor {base['unif_bat']['PROSPERO']['nll']:.3f} -> {G5}", flush=True)
        print(f"{stamp()} base: JULIET train {base['train_bat']['JULIET']['nll']:.3f}/"
              f"{base['train_bat']['JULIET']['acc']:.2f}  uniform "
              f"{base['unif_bat']['JULIET']['nll']:.3f}/{base['unif_bat']['JULIET']['acc']:.2f}  "
              f"val_all {base['ce']['val_all']:.4f}", flush=True)
        assert G4 and G5 and G3, "host gates failed"

        # D2-analog surgery + gate
        d2_sd = {k: v.clone() for k, v in net_sd.items()}
        d2_sd["wte.weight"][jid] = 0.0
        d2_sd["lm_head.weight"][jid] = 0.0
        n_diff, confined = 0, True
        for key in ("wte.weight", "lm_head.weight"):
            d = d2_sd[key] != net_sd[key]
            nd = int(d.sum().item())
            n_diff += nd
            rows = torch.nonzero(d)[:, 0].unique()
            if not (len(rows) == 1 and int(rows[0]) == jid and nd == cfg.n_embd):
                confined = False
        others = all(torch.equal(d2_sd[k], net_sd[k])
                     for k in d2_sd if k not in ("wte.weight", "lm_head.weight"))
        G2 = {"n_elements_changed": n_diff, "expected": 2 * cfg.n_embd,
              "confined_to_J_rows": confined, "others_bit_identical": bool(others),
              "pass": bool(n_diff == 2 * cfg.n_embd and confined and others)}
        assert G2["pass"], G2
        d2net = copy.deepcopy(net)
        d2net.load_state_dict(d2_sd)
        d2net.eval()

        d2 = eval_all(d2net)
        d2["per_pos_acc_juliet"] = d2["train_bat"]["JULIET"]["per_pos_acc"]
        d2["residual_positions"] = [i for i, a in enumerate(d2["per_pos_acc_juliet"]) if a >= 0.05]
        print(f"{stamp()} D2-analog: JULIET train {d2['train_bat']['JULIET']['nll']:.3f}/"
              f"{d2['train_bat']['JULIET']['acc']:.3f} uniform "
              f"{d2['unif_bat']['JULIET']['nll']:.3f}/{d2['unif_bat']['JULIET']['acc']:.3f} "
              f"per-pos {[round(a, 2) for a in d2['per_pos_acc_juliet']]} "
              f"dCE {d2['ce']['val_all'] - base['ce']['val_all']:+.5f}", flush=True)

        # in-run D2-conditioned atlas
        atlas = run_atlas_d2(d2net, "JULIET")
        dh = atlas["heads_dce"]
        head_rank = sorted([(f"L{l}H{h}", float(dh[l, h])) for l in range(NL) for h in range(NH)],
                           key=lambda t: -t[1])
        own_top_name = head_rank[0][0]
        own_top = (int(own_top_name[1]), int(own_top_name[3]))
        block_rank = sorted([(f"L{l}{k[:1].upper()}", v) for k in ("attn", "mlp")
                             for l, v in enumerate(atlas["blocks_dce"][k])], key=lambda t: -t[1])
        print(f"{stamp()} D2-atlas (in-run): top head {own_top_name} "
              f"+{head_rank[0][1]:.3f} | top-4 {[(s, round(v, 3)) for s, v in head_rank[:4]]} | "
              f"B's L3H5 here: {dict(head_rank)['L3H5']:+.3f} | top block {block_rank[0]}", flush=True)

        # two-factor cells
        def score_cell(specs, tag, label):
            with pos_lesion(d2net, specs) as st:
                r = eval_all(d2net, state=st)
            cell = {
                "head": label, "mask_rule": "full (J..JULIE)",
                "juliet_train": {"nll": r["train_bat"]["JULIET"]["nll"],
                                 "acc": r["train_bat"]["JULIET"]["acc"],
                                 "per_pos_acc": r["train_bat"]["JULIET"]["per_pos_acc"]},
                "juliet_unif": {"nll": r["unif_bat"]["JULIET"]["nll"],
                                "acc": r["unif_bat"]["JULIET"]["acc"],
                                "per_pos_acc": r["unif_bat"]["JULIET"]["per_pos_acc"]},
                "train_bat": {w: {"nll": r["train_bat"][w]["nll"], "acc": r["train_bat"][w]["acc"]}
                              for w in BATTERY},
                "unif_bat": {w: {"nll": r["unif_bat"][w]["nll"], "acc": r["unif_bat"][w]["acc"]}
                             for w in BATTERY},
                "d_nll_train_vs_d2": {w: r["train_bat"][w]["nll"] - d2["train_bat"][w]["nll"]
                                      for w in BATTERY},
                "d_nll_unif_vs_d2": {w: r["unif_bat"][w]["nll"] - d2["unif_bat"][w]["nll"]
                                     for w in BATTERY},
                "census_extra_vs_d2": {w: r["census"][w]["nll"] - d2["census"][w]["nll"]
                                       for w in r["census"]},
                "ce": r["ce"],
                "dce_val_content": r["ce"]["val_all"] - base["ce"]["val_all"],
                "dce_julwin": r["ce"]["jul_windows"] - base["ce"]["jul_windows"],
                "dce_val_position": r["ce"]["val_all_position_rule"] - base["ce"]["val_all"],
                "bar2_train": bool(r["train_bat"]["JULIET"]["nll"] >= BAR2_NLL
                                   and r["train_bat"]["JULIET"]["acc"] <= BAR2_ACC),
                "bar2_unif": bool(r["unif_bat"]["JULIET"]["nll"] >= BAR2_NLL
                                  and r["unif_bat"]["JULIET"]["acc"] <= BAR2_ACC),
                "cost_ok": bool(r["ce"]["val_all"] - base["ce"]["val_all"] <= COST_BAR
                                and r["ce"]["val_all_position_rule"] - base["ce"]["val_all"] <= COST_BAR),
            }
            cell["julius_extra"] = cell["census_extra_vs_d2"].get("Julius")
            cell["leak_flag"] = bool(cell["julius_extra"] is not None
                                     and cell["julius_extra"] > LEAK_BAR)
            print(f"{stamp()} cell {tag:14s} [{label}] JULIET train "
                  f"{cell['juliet_train']['nll']:6.2f}/{cell['juliet_train']['acc']:.4f} "
                  f"unif {cell['juliet_unif']['nll']:6.2f}/{cell['juliet_unif']['acc']:.4f} "
                  f"dCE {cell['dce_val_content']:+.5f} (posrule {cell['dce_val_position']:+.5f}) "
                  f"Julius extra {cell['julius_extra']:+.3f} leak={int(cell['leak_flag'])} "
                  f"bar2 {int(cell['bar2_train'])}/{int(cell['bar2_unif'])}", flush=True)
            return cell

        cells = {}
        recipes = [("own", own_top)] if own_top == B_HEAD else [("own", own_top), ("b_L3H5", B_HEAD)]
        for tag, head in recipes:
            cells[tag] = score_cell([("head", head[0], head[1])], tag, f"L{head[0]}H{head[1]}")
        # supplementary (deviation 7): own-top BLOCK under the same rule
        tb = block_rank[0][0]                     # e.g. "L1A" or "L1M"
        tb_kind = "attn" if tb.endswith("A") else "mlp"
        cells["own-block-supp"] = score_cell([(tb_kind, int(tb[1]), None)],
                                             "own-block-supp", tb)
        if own_top == B_HEAD:
            cells["b_L3H5"] = dict(cells["own"], note="cells coincide: own-top == L3H5")
            print(f"{stamp()} NOTE: own-top == B's L3H5 on {net_name}; cells coincide", flush=True)
        best = min(cells, key=lambda k: cells[k]["juliet_train"]["acc"])

        nets[net_name] = {
            "ckpt": str(ckpt), "gates": gates, "base": base, "d2": d2, "G2": G2,
            "d2_atlas": {"base": {"nll": atlas["base"]["nll"], "acc": atlas["base"]["acc"]},
                         "heads_dce": dh, "blocks_dce": atlas["blocks_dce"],
                         "head_rank_by_dnll": [[s, round(v, 4)] for s, v in head_rank],
                         "block_rank_by_dnll": [[s, round(v, 4)] for s, v in block_rank],
                         "own_top": own_top_name, "own_top_dce": head_rank[0][1],
                         "l3h5_dce": dict(head_rank)["L3H5"]},
            "cells": cells,
            "best_cell": {"tag": best, "head": cells[best]["head"],
                          "juliet_acc_train": cells[best]["juliet_train"]["acc"],
                          "juliet_acc_unif": cells[best]["juliet_unif"]["acc"],
                          "bar2_train": cells[best]["bar2_train"],
                          "bar2_unif": cells[best]["bar2_unif"],
                          "cost_ok": cells[best]["cost_ok"]},
        }

    # ------------------------------------------------------------- verdicts
    va, vb, vc = {}, {}, {}
    for net_name, N in nets.items():
        own, bl = N["cells"]["own"], N["cells"]["b_L3H5"]
        va[net_name] = {
            "own_head": own["head"], "differs_from_B_L3H5": own["head"] != "L3H5",
            "bar2_train": own["bar2_train"], "bar2_unif": own["bar2_unif"],
            "cost_ok": own["cost_ok"], "dce_val_content": own["dce_val_content"],
            "juliet_acc_train": own["juliet_train"]["acc"],
            "juliet_acc_unif": own["juliet_unif"]["acc"],
            "replicates": bool(own["bar2_train"] and own["cost_ok"]),
            "replicates_full": bool(own["bar2_train"] and own["bar2_unif"] and own["cost_ok"]),
        }
        va[net_name]["verdict"] = (
            "FULL (train+uniform Bar-2 at cost <= +0.05)" if va[net_name]["replicates_full"]
            else "TRAIN-ONLY (Bar-2 on train battery, uniform battery retains knowledge)"
            if va[net_name]["replicates"] and not own["bar2_unif"]
            else "FAIL (no Bar-2 with in-run head)")
        va[net_name]["best_cell"] = N["best_cell"]
        vb[net_name] = {
            "bar2_train": bl["bar2_train"], "bar2_unif": bl["bar2_unif"],
            "juliet_acc_train": bl["juliet_train"]["acc"],
            "l3h5_dce_in_this_net": N["d2_atlas"]["l3h5_dce"],
            "transfers": bool(bl["bar2_train"] and bl["cost_ok"]),
            "expected": "NO (memo: healthy B43 atlas L3H5 -0.02)" if net_name == "B43"
                        else "open (same-init as B; memo silent on BDO L3H5)"}
        vb[net_name]["verdict"] = ("YES (B's recipe reaches Bar-2)" if vb[net_name]["transfers"]
                                   else "NO (B's recipe does not erase here)")
        vc[net_name] = {
            "own": {"julius_extra": own["julius_extra"], "leak": own["leak_flag"],
                    "all_extras": {w: own["census_extra_vs_d2"].get(w) for w in LEAK_WORDS}},
            "b_L3H5": {"julius_extra": bl["julius_extra"], "leak": bl["leak_flag"]},
        }
    vc["flag_bar_nats"] = LEAK_BAR
    vc["leaky"] = bool(any(c["own"]["leak"] for c in (vc["B43"], vc["BDO"])))
    vc["verdict"] = ("LEAKY: Julius-class extra damage > 0.5 nats under the deployed recipe"
                     if vc["leaky"] else
                     "CLEAN: content rule does not leak onto Julius-class continuations "
                     "(extra dNLL <= 0.5 nats over D2-alone)")
    a_all = "REPLICATES" if all(v["replicates"] for v in va.values()) else "PARTIAL/FAIL"
    b_all = "TRANSFERS" if all(v["transfers"] for v in vb.values()) else \
            ("DOES NOT TRANSFER (as expected)" if not any(v["transfers"] for v in vb.values())
             else "MIXED")

    metrics = {
        "experiment": "e046_c6_replication", "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED, "smoke": SMOKE,
        "hosts": {"B43": f"different-init (seed 43), same regime; memo healthy atlas "
                         f"{E042_REF['memo_healthy_atlas']['B43']}",
                  "BDO": f"same-init as B (seed 42), different data order (corpus seed 7777); "
                         f"memo healthy atlas {E042_REF['memo_healthy_atlas']['BDO']}"},
        "b_reference_e042": E042_REF,
        "mask_fires": fires,
        "bars": {"bar2_nll": BAR2_NLL, "bar2_acc": BAR2_ACC, "cost": COST_BAR,
                 "leak_extra_nats": LEAK_BAR},
        "nets": nets,
        "verdicts": {"a_replication": {"per_net": va,
                                       "overall": a_all,
                                       "n_nets_replicated": sum(1 for v in va.values() if v["replicates"])},
                     "b_transfer": {"per_net": vb, "overall": b_all},
                     "c_leak": vc},
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": cfg_dict(cfg),
    }
    save_json(rd / "metrics.json", jsonable(metrics))

    # ------------------------------------------------------------- PNG
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.6))

    def heat(ax, M, title, mark_own=None):
        im = ax.imshow(M.tolist(), cmap="magma", aspect="auto")
        for l in range(NL):
            for h in range(NH):
                ax.text(h, l, f"{M[l, h]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if M[l, h] < 0.55 * float(M.max()) else "black")
        ax.set_xticks(range(NH)); ax.set_xticklabels([f"H{h}" for h in range(NH)])
        ax.set_yticks(range(NL)); ax.set_yticklabels([f"L{l}" for l in range(NL)])
        if mark_own is not None:
            ax.add_patch(plt.Rectangle((mark_own[1] - 0.5, mark_own[0] - 0.5), 1, 1,
                                       fill=False, edgecolor="lime", lw=2.2))
        ax.add_patch(plt.Rectangle((B_HEAD[1] - 0.5, B_HEAD[0] - 0.5), 1, 1,
                                   fill=False, edgecolor="cyan", lw=2.0, ls="--"))
        ax.set_title(title, fontsize=9)
        return im

    own_heads = {}
    for i, net_name in enumerate(("B43", "BDO")):
        N = nets[net_name]
        oth = N["d2_atlas"]["own_top"]
        own_heads[net_name] = (int(oth[1]), int(oth[3]))
        im = heat(axes[0, i], N["d2_atlas"]["heads_dce"],
                  f"{net_name} — D2-conditioned residual atlas (dNLL)\n"
                  f"green = own top {oth} (+{N['d2_atlas']['own_top_dce']:.2f}); "
                  f"cyan dashed = B's L3H5 ({N['d2_atlas']['l3h5_dce']:+.2f})",
                  mark_own=own_heads[net_name])
        fig.colorbar(im, ax=axes[0, i], fraction=0.046)

    ax = axes[0, 2]
    wid = 0.12
    leak_cols = {"B43-own": "crimson", "B43-L3H5": "lightcoral",
                 "BDO-own": "steelblue", "BDO-L3H5": "lightsteelblue"}
    for ci, (net_name, tag) in enumerate([("B43", "own"), ("B43", "b_L3H5"),
                                          ("BDO", "own"), ("BDO", "b_L3H5")]):
        cell = nets[net_name]["cells"][tag]
        xs = [LEAK_WORDS.index(w) + wid * (ci - 1.5) for w in LEAK_WORDS]
        ys = [cell["census_extra_vs_d2"].get(w, 0.0) for w in LEAK_WORDS]
        ax.bar(xs, ys, wid, label=f"{net_name}-{tag}({cell['head']})",
               color=leak_cols[f"{net_name}-{'own' if tag=='own' else 'L3H5'}"])
    ax.axhline(LEAK_BAR, color="k", ls="--", lw=1, label=f"leak bar {LEAK_BAR} nats")
    ax.set_xticks(range(len(LEAK_WORDS))); ax.set_xticklabels(LEAK_WORDS, fontsize=7)
    ax.set_ylabel("census dNLL extra over D2-alone (nats)")
    ax.legend(fontsize=6.5, ncol=2)
    ax.set_title(f"J-census prefix-leak (R5 missing observation)\n{vc['verdict'].split(':')[0]}")

    ax = axes[1, 0]
    cellsel = [("B43", "own"), ("B43", "b_L3H5"), ("B43", "own-block-supp"),
               ("BDO", "own"), ("BDO", "b_L3H5"), ("BDO", "own-block-supp")]
    labels, accs_t, accs_u = [], [], []
    for net_name, tag in cellsel:
        cell = nets[net_name]["cells"][tag]
        star = "*" if tag == "own-block-supp" else ""
        labels.append(f"{net_name}\n{tag.replace('-supp','')}{star}\n{cell['head']}")
        accs_t.append(cell["juliet_train"]["acc"])
        accs_u.append(cell["juliet_unif"]["acc"])
    x = range(len(labels))
    ax.bar([i - 0.18 for i in x], accs_t, 0.36, color="navy", label="train battery")
    ax.bar([i + 0.18 for i in x], accs_u, 0.36, color="darkorange", label="uniform-floor battery")
    ax.axhline(BAR2_ACC, color="k", ls="--", lw=1, label="Bar-2 acc 0.10")
    ax.axhline(nets["B43"]["d2"]["train_bat"]["JULIET"]["acc"], color="gray", ls=":", lw=1,
               label=f"D2 alone (B43) {nets['B43']['d2']['train_bat']['JULIET']['acc']:.3f}")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("JULIET acc"); ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"(a) replication / (b) transfer — {a_all}; transfer: {b_all}\n"
                 f"(* = supplementary block cell; B ref acc 0.0013)")

    ax = axes[1, 1]
    for ci, (net_name, tag) in enumerate(cellsel):
        cell = nets[net_name]["cells"][tag]
        ax.bar(ci - 0.25, cell["dce_val_content"], 0.24, color="seagreen", label="val (content rule)" if ci == 0 else None)
        ax.bar(ci, cell["dce_julwin"], 0.24, color="olivedrab", label="JULIET windows" if ci == 0 else None)
        ax.bar(ci + 0.25, cell["dce_val_position"], 0.24, color="gray", label="val (position rule)" if ci == 0 else None)
    ax.axhline(COST_BAR, color="k", ls="--", lw=1, label=f"cost bar +{COST_BAR}")
    ax.set_xticks(range(len(cellsel)))
    ax.set_xticklabels([f"{n}-{t}\n{nets[n]['cells'][t]['head']}" for n, t in cellsel], fontsize=6.5)
    ax.set_ylabel("corpus dCE vs base (nats)")
    ax.legend(fontsize=7)
    ax.set_title("two-factor corpus cost")

    ax = axes[1, 2]
    x = range(6)
    w_ = 0.27
    ax.bar([i - w_ for i in x], nets["B43"]["d2"]["per_pos_acc_juliet"], w_, color="crimson", label="B43 D2")
    ax.bar([i for i in x], nets["BDO"]["d2"]["per_pos_acc_juliet"], w_, color="steelblue", label="BDO D2")
    ax.bar([i + w_ for i in x], E042_REF["d2_per_pos_acc"], w_, color="gray", label="B D2 (e042 ref)")
    ax.set_xticks(list(x)); ax.set_xticklabels([f"JULIE"[i] if i < 5 else "T" for i in x])
    ax.set_xlabel("name position (predicting this char)")
    ax.set_ylabel("per-position acc under D2 alone")
    ax.legend(fontsize=7)
    ax.set_title("D2 residual anatomy across nets\n(is the residual still the pos-3 '?UL->I' transition?)")

    fig.suptitle(f"E046 — C6 replication (two-factor erasure, in-run head discovery) | "
                 f"(a) {a_all} | (b) {b_all} | (c) {vc['verdict'].split(':')[0]}")
    fig.tight_layout()
    fig.savefig(rd / "e046_replication.png", dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------- report
    print(f"\n{stamp()} === E046 VERDICTS ===")
    print(f"(a) REPLICATION (rows + in-run head): {a_all}")
    for net_name, v in va.items():
        print(f"    {net_name}: own head {v['own_head']} "
              f"(differs from B's L3H5: {v['differs_from_B_L3H5']}) -> {v['verdict']} "
              f"[train acc {v['juliet_acc_train']:.4f}, uniform acc {v['juliet_acc_unif']:.4f}, "
              f"dCE {v['dce_val_content']:+.5f}]")
        bc = v["best_cell"]
        print(f"      best cell overall: {bc['tag']} [{bc['head']}] acc {bc['juliet_acc_train']:.4f} "
              f"(bar2 {int(bc['bar2_train'])}/{int(bc['bar2_unif'])}, cost_ok {int(bc['cost_ok'])})")
    print(f"(b) B's L3H5 TRANSPLANT: {b_all}")
    for net_name, v in vb.items():
        print(f"    {net_name}: {v['verdict']} [acc {v['juliet_acc_train']:.4f}; "
              f"L3H5 dNLL in this net's D2-atlas {v['l3h5_dce_in_this_net']:+.3f}] "
              f"(expected: {v['expected']})")
    print(f"(c) CONTENT-RULE LEAK: {vc['verdict']}")
    for net_name in ("B43", "BDO"):
        for tag in ("own", "b_L3H5", "own-block-supp"):
            cell = nets[net_name]["cells"][tag]
            print(f"    {net_name}/{tag}: Julius extra {cell['julius_extra']:+.3f} nats "
                  f"(leak {int(cell['leak_flag'])}); extras "
                  f"{ {w: round(cell['census_extra_vs_d2'].get(w, 0.0), 3) for w in LEAK_WORDS} }")
    print(f"outputs: {rd}")
    print(f"total {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
