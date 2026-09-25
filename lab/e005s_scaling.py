"""E005s — the scaling capstone: do the day-one laws survive 0.7M and ~8M?

Two NEW nets on the SAME Shakespeare corpus, same seed 42, same trainer
(steps=4000, lr=1e-3, batch=64, cosine; 252s wall cap, ckpt-resumable):
  SMALL: 4 layers / 4 heads / 128 dim   -> 840,704 params  ("0.7M-class";
         the dims as specified verify to 0.84M, ~20% over the ~0.7M label)
  LARGE: 8 layers / 8 heads / 320 dim   -> 9,977,600 params ("8M-class";
         the dims as specified verify to 9.98M, ~25% over the ~8M label)
  Reference B = runs/checkpoints/e001.pt (2,739,072 params) — all readout
  numbers loaded from runs/, never re-measured.

FROZEN READOUTS per net (T014 re-scope: L1 tests the GATE'S EXISTENCE,
never depth invariance; card v3 = T017):
  L1 gate-existence  — e018 causal census protocol verbatim (1536 positions,
      counterfactual pairing seeds 18/19/20, patch d = 0..L-1).
  L2 front-loading   — attn/mlp block zero-ablation damage per layer
      (estimate_loss n_batches=30, e001 protocol).
  L4 16-token sufficiency — e013c protocol: far-value = CE(trunc-16) -
      CE(full-256), mean over 2000 held-out positions (gen seed 15).
  L6 row-surgery     — e023's D2 cell only: zero the 'J' rows of
      wte + lm_head (the JULIET address; corpus is shared so JULIET/J is
      each net's own corpus analysis too), battery CTX=120 cap=125,
      pure controls ROMEO/GLOUCESTER/CORIOLANUS, letter control JOHN,
      12-word J-census, corpus cost on 400 fixed val blocks (seed 202).

REGISTERED BEFORE ANY RUN (frozen):
  P1 (qualitative gate exists at both scales). gate_exists(net) :=
       suffix_monotone_flip_fraction >= 0.60
       AND causal mode <= L-2 (a clear mode below the last layer, modal
           share >= 0.15 of flippers)
       AND flip curve non-decreasing within noise (min consecutive
           diff >= -0.03)
       AND flip_curve[-1] >= 0.5 (the census has power).
     P1 holds iff gate_exists(SMALL) and gate_exists(LARGE).
     NO depth-invariance claim is made or tested.
  P2 (front-loading holds): attn-L0 damage >= 2 x attn-last-layer damage,
     both new nets.
  P3 (far-value stays < 0.05 nats): mean far-value (trunc-16 minus
     full-256) < 0.05 at both scales.
  P4 (row surgery >= 3x the naive-selectivity bar): S_name >= 3 x 1.23
     = 3.69 (1.23 = day-one naive-ascent selectivity at the forgetting
     bar, e003c) at both scales. Secondary shape readout (not gated):
     class-exact = max|dNLL| over non-J battery names <= 0.05 AND every
     J-census word dNLL >= 1.0; cheap = dCE(val-All) <= 0.01 nats.

Registered fallback: if LARGE's 252s cap ends below step 1000, training may
extend to a 400s wall total (implemented as a budget carried across ckpt
resumes). Everything else runs as registered.

Run: python lab/e005s_scaling.py   (B checkpoint + runs/ references required)
Outputs: runs/e005s/{metrics.json, scaling_readouts.png}. smoke: false.
"""
from __future__ import annotations

import copy
import json
import math
import re
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, cfg_dict,
                    estimate_loss, lesion_loss, run_dir, save_json, set_seed,
                    train_model)

T0 = time.time()
SEED = 42
CKPT_DIR = REPO / "runs" / "checkpoints"
E001_CKPT = CKPT_DIR / "e001.pt"
N_POS, BATCH = 1536, 64          # e018 census convention
N_POS_L4 = 2000                  # e013c convention
TRUNC = 16
CTX, BATT_CAP = 120, 125         # e023 battery convention
N_VAL_BLOCKS, VAL_SEED = 400, 202

NETS = {
    "small": {"n_layer": 4, "n_head": 4, "n_embd": 128},
    "large": {"n_layer": 8, "n_head": 8, "n_embd": 320},
}
BATTERY = ["JULIET", "JOHN", "ROMEO", "GLOUCESTER", "MENENIUS",
           "CORIOLANUS", "ISABELLA", "LUCIO", "PETRUCHIO"]
PURE_CONTROLS = ["ROMEO", "GLOUCESTER", "CORIOLANUS"]
CENSUS = ["JULIET", "Juliet", "JOHN", "John", "Jove", "Jack", "Jesu",
          "Justice", "Jupiter", "Join", "Juno", "Julius"]
NAIVE_SEL = 1.23                 # day-one naive-ascent selectivity (e003c)
P4_BAR = 3 * NAIVE_SEL           # 3.69


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ------------------------------------------------------------------ L1 (e018)

@torch.no_grad()
def snapshots_full(model, xs, last_only=False):
    snaps, handles = [], []

    def pre(module, args):
        snaps.append(args[0].detach() if not last_only else args[0].detach()[:, -1, :])
        return None

    handles.append(model.h[0].register_forward_pre_hook(pre))
    for block in model.h:
        def h(module, args, out):
            snaps.append(out.detach() if not last_only else out.detach()[:, -1, :])
        handles.append(block.register_forward_hook(h))
    model(xs)
    for h_ in handles:
        h_.remove()
    return snaps


@torch.no_grad()
def readout_top1(model, v):
    return model.lm_head(model.ln_f(v)).argmax(-1)


@torch.no_grad()
def causal_census(model, corpus):
    """e018 protocol verbatim, parametric in depth; returns the gate readout."""
    L = model.cfg.n_layer
    set_seed(18)
    gen_o = torch.Generator().manual_seed(18)
    gen_p = torch.Generator().manual_seed(19)
    gen_m = torch.Generator().manual_seed(20)
    hi = len(corpus.val) - model.cfg.block_size - 2
    ix_o = torch.randint(hi, (N_POS,), generator=gen_o)
    ix_p = torch.randint(hi, (N_POS,), generator=gen_p)
    xs_o = torch.stack([corpus.val[i: i + model.cfg.block_size] for i in ix_o]).to(DEVICE)
    xs_p = torch.stack([corpus.val[i: i + model.cfg.block_size] for i in ix_p]).to(DEVICE)

    top1_o = torch.cat([readout_top1(model, snapshots_full(model, xs_o[b: b + BATCH], last_only=True)[-1])
                        for b in range(0, N_POS, BATCH)]).cpu()
    top1_p = torch.cat([readout_top1(model, snapshots_full(model, xs_p[b: b + BATCH], last_only=True)[-1])
                        for b in range(0, N_POS, BATCH)]).cpu()
    cf = torch.zeros(N_POS, dtype=torch.long)
    for i in range(N_POS):
        valid = torch.nonzero((top1_p != top1_o[i]) & (ix_p != ix_o[i]), as_tuple=False).squeeze(1)
        cf[i] = valid[int(torch.randint(len(valid), (1,), generator=gen_m))]
    xs_c = xs_p[cf.to(DEVICE)]

    causal_depths = torch.full((N_POS,), -1, dtype=torch.long)
    flips = torch.zeros(N_POS, L, dtype=torch.bool)
    for b0 in range(0, N_POS, BATCH):
        xb, cb = xs_o[b0: b0 + BATCH], xs_c[b0: b0 + BATCH]
        A = snapshots_full(model, xb)                   # L+1 x (B,T,C)
        C = snapshots_full(model, cb, last_only=True)   # L+1 x (B, C)
        final = readout_top1(model, A[-1][:, -1, :]).cpu()
        cf_final = readout_top1(model, C[-1]).cpu()
        for d in range(L):
            patched = A[d].clone()
            patched[:, -1, :] = C[d]
            x = patched
            for block in model.h[d:]:
                x = block(x)
            pt1 = readout_top1(model, x[:, -1, :]).cpu()
            flips[b0: b0 + BATCH, d] = pt1 == cf_final
        del A, C
    for d in range(L):
        sel = (causal_depths < 0) & flips[:, d]
        causal_depths[sel] = d

    fl = causal_depths >= 0
    hist = torch.bincount(causal_depths[fl], minlength=L).float()
    mode = int(hist.argmax())
    modal_share = float(hist[mode] / max(1, int(fl.sum())))
    flip_curve = flips.float().mean(0)
    no_flip = float((~fl).float().mean())
    suf = torch.zeros(N_POS, dtype=torch.bool)
    for i in range(N_POS):
        f = flips[i]
        if f.any():
            d0 = int(f.nonzero()[0])
            suf[i] = bool(f[d0:].all())
    suffix_frac = float(suf[fl].float().mean())
    fc = flip_curve.tolist()
    min_diff = min(b - a for a, b in zip(fc, fc[1:])) if len(fc) > 1 else 0.0
    gate = {
        "flip_curve_per_depth": fc,
        "causal_hist_d0_dLm1": hist.tolist(),
        "causal_mode": mode, "modal_share": modal_share,
        "no_single_depth_flip_fraction": no_flip,
        "suffix_monotone_flip_fraction": suffix_frac,
        "min_consecutive_flip_diff": min_diff,
        "n_flippers": int(fl.sum()),
    }
    gate["gate_exists"] = bool(
        suffix_frac >= 0.60 and mode <= L - 2 and modal_share >= 0.15
        and min_diff >= -0.03 and fc[-1] >= 0.5)
    return gate


# ------------------------------------------------------------------ L4 (e013c)

@torch.no_grad()
def last_ce(model, xs, ys):
    out = torch.zeros(xs.shape[0])
    for b0 in range(0, xs.shape[0], BATCH):
        logits, _ = model(xs[b0: b0 + BATCH])
        out[b0: b0 + BATCH] = F.cross_entropy(
            logits[:, -1], ys[b0: b0 + BATCH], reduction="none").cpu()
    return out


@torch.no_grad()
def far_value(model, corpus):
    gen = torch.Generator().manual_seed(15)
    ix = torch.randint(len(corpus.val) - model.cfg.block_size - 2,
                       (N_POS_L4,), generator=gen)
    xs_full = torch.stack([corpus.val[i: i + model.cfg.block_size] for i in ix]).to(DEVICE)
    ys = corpus.val[ix + model.cfg.block_size].to(DEVICE)
    xs_trunc = xs_full[:, -TRUNC:].contiguous()
    fv = last_ce(model, xs_trunc, ys) - last_ce(model, xs_full, ys)
    return {"far_value_mean": float(fv.mean()),
            "far_value_median": float(fv.median()),
            "frac_ge_015": float((fv >= 0.15).float().mean()),
            "p99": float(fv.quantile(0.99)),
            "n_positions": N_POS_L4}


# ------------------------------------------------------------------ L6 (e023 D2)

def find_occ(text: str, word: str) -> list[int]:
    pat = re.compile(r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])")
    return [m.start() for m in pat.finditer(text)]


def build_bats(ids, text, stoi, words, cap=BATT_CAP):
    bats = {}
    for w in words:
        occs = find_occ(text, w)[:cap]
        keep = [p for p in occs if p >= CTX]
        wid = torch.tensor([stoi[c] for c in w], dtype=torch.long)
        seqs = [torch.cat([ids[p - CTX: p], wid]) for p in keep]
        bats[w] = {"seq": torch.stack(seqs) if seqs else None, "n": len(keep)}
    return bats


@torch.no_grad()
def eval_bats(model, bats):
    model.eval()
    V = model.cfg.vocab
    out = {}
    for w, b in bats.items():
        if b["seq"] is None or b["n"] == 0:
            out[w] = {"n": 0, "nll": None, "acc": None}
            continue
        L = len(w)
        seq = b["seq"].to(DEVICE)
        x, y = seq[:, :-1], seq[:, 1:]
        nlls, accs = [], []
        for i in range(0, len(seq), 128):
            logits, _ = model(x[i: i + 128])
            lg = logits[:, CTX - 1: CTX - 1 + L, :]
            tg = y[i: i + 128][:, CTX - 1: CTX - 1 + L]
            nll = F.cross_entropy(lg.reshape(-1, V), tg.reshape(-1),
                                  reduction="none").view(-1, L)
            nlls.append(nll)
            accs.append((lg.argmax(-1) == tg).float())
        nll_m, acc_m = torch.cat(nlls), torch.cat(accs)
        out[w] = {"n": int(nll_m.shape[0]), "nll": float(nll_m.mean()),
                  "acc": float(acc_m.mean())}
    model.train()
    return out


def fixed_blocks(src, block, n, seed):
    gen = torch.Generator().manual_seed(seed)
    ix = torch.randint(len(src) - block - 1, (n,), generator=gen)
    x = torch.stack([src[i: i + block] for i in ix])
    y = torch.stack([src[i + 1: i + 1 + block] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def ce_fixed(model, x, y):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(x), 64):
        _, loss = model(x[i: i + 64], y[i: i + 64])
        tot += float(loss.item()) * len(x[i: i + 64])
        n += len(x[i: i + 64])
    model.train()
    return tot / max(n, 1)


def row_surgery(model, corpus):
    """e023 D2/zero/both: zero the 'J' rows of wte + lm_head; verify confinement."""
    jid = corpus.stoi["J"]
    d = model.cfg.n_embd
    base_sd = {k: v.clone() for k, v in model.state_dict().items()}
    sd = {k: v.clone() for k, v in base_sd.items()}
    for key in ("wte.weight", "lm_head.weight"):
        sd[key][jid] = 0.0
    m = copy.deepcopy(model)
    m.load_state_dict(sd)
    rows_changed, total, confined = {}, 0, True
    for key in ("wte.weight", "lm_head.weight"):
        diff = sd[key] != base_sd[key]
        nd = int(diff.sum().item())
        total += nd
        rows = sorted(set(torch.nonzero(diff)[:, 0].tolist()))
        rows_changed[key] = rows
        if nd != len(rows) * d:
            confined = False
    others = all(torch.equal(sd[k], base_sd[k])
                 for k in sd if k not in ("wte.weight", "lm_head.weight"))
    g2 = {"rows_changed": rows_changed, "n_elements_changed": total,
          "expected_elements": 2 * d, "confined_to_full_rows": confined,
          "others_bit_identical": bool(others),
          "pass": bool(total == 2 * d and confined and others
                       and all(r == [jid] for r in rows_changed.values()))}
    return m, g2


def l6_readout(model, corpus, train_text):
    jid = corpus.stoi["J"]
    bats = build_bats(corpus.train, train_text, corpus.stoi, BATTERY)
    cens = build_bats(corpus.train, train_text, corpus.stoi, CENSUS, cap=None)
    x_val, y_val = fixed_blocks(corpus.val, 256, N_VAL_BLOCKS, VAL_SEED)
    r1_base = eval_bats(model, bats)
    cens_base = eval_bats(model, cens)
    ce_base = ce_fixed(model, x_val, y_val)
    surg, g2 = row_surgery(model, corpus)
    r1 = eval_bats(surg, bats)
    cens_s = eval_bats(surg, cens)
    ce = ce_fixed(surg, x_val, y_val)
    assert g2["pass"], f"G2 confinement failed: {g2}"

    dnll = {w: r1[w]["nll"] - r1_base[w]["nll"] for w in BATTERY if r1_base[w]["n"]}
    pure = [dnll[w] for w in PURE_CONTROLS]
    mpure = sum(pure) / len(pure)
    dj = dnll["JULIET"]
    s_name = dj / mpure if mpure > 1e-9 else float("inf")
    s_letter = dj / dnll["JOHN"] if abs(dnll["JOHN"]) > 1e-9 else float("inf")
    dce = ce - ce_base
    cens_d = {w: (cens_s[w]["nll"] - cens_base[w]["nll"])
              for w in CENSUS if cens_base[w]["n"]}
    nonj = [abs(dnll[w]) for w in BATTERY
            if w not in ("JULIET", "JOHN") and w in dnll]
    return {
        "g2_confinement": g2,
        "baseline": {w: {"nll": r1_base[w]["nll"], "acc": r1_base[w]["acc"],
                         "n": r1_base[w]["n"]} for w in BATTERY},
        "surgery": {w: {"nll": r1[w]["nll"], "acc": r1[w]["acc"]} for w in BATTERY},
        "d_nll": dnll, "d_ce_val_all": dce,
        "s_name": s_name, "s_letter": s_letter,
        "census_d_nll": cens_d,
        "class_exact": {"max_abs_nonj_d_nll": max(nonj),
                        "min_census_d_nll": min(cens_d.values())},
        "cheap": dce <= 0.01,
        # inf (pure controls undamaged at 1e-9) counts as passing only with
        # Bar-1-level real damage (e023 convention: inf is the extreme of
        # selectivity, not an artifact)
        "p4_s_name_ge_bar": bool(
            (s_name >= P4_BAR) if not math.isinf(s_name) else (dj >= 2.0)),
    }


# ------------------------------------------------------------------ B reference

def load_b_ref():
    ref = {}
    try:
        d = json.loads((REPO / "runs" / "e012d" / "metrics.json").read_text(encoding="utf-8"))
        b = d["nets"]["B"]
        ref["l1"] = {"flip_curve_per_depth": b["flip_curve_per_depth"],
                     "causal_mode": b["causal_mode"],
                     "suffix_monotone_flip_fraction": b["suffix_monotone_flip_fraction"],
                     "no_single_depth_flip_fraction": b["no_single_depth_flip_fraction"],
                     "causal_hist_d0_d5": b["causal_hist_d0_d5"],
                     "n_layer": 6}
    except Exception as e:
        ref["l1"] = {"error": str(e)}
    try:
        d = json.loads((REPO / "runs" / "e001" / "metrics.json").read_text(encoding="utf-8"))
        ref["l2"] = {"attn_block_damage": d["attn_block_damage"],
                     "mlp_block_damage": d["mlp_block_damage"],
                     "baseline_val_loss": d["baseline_val_loss"],
                     "params": d["params"], "n_layer": 6}
    except Exception as e:
        ref["l2"] = {"error": str(e)}
    try:
        d = json.loads((REPO / "runs" / "e013c" / "metrics.json").read_text(encoding="utf-8"))
        ref["l4"] = {"far_value_mean": d["far_value_mean"],
                     "frac_ge_015": d["frac_tail_ge_015"]}
    except Exception as e:
        ref["l4"] = {"error": str(e)}
    try:
        d = json.loads((REPO / "runs" / "e023" / "metrics.json").read_text(encoding="utf-8"))
        c = d["arm_ab"]["D2/zero/both"]
        ref["l6"] = {"juliet_base_nll": d["baseline"]["r1"]["JULIET"]["nll"],
                     "juliet_base_acc": d["baseline"]["r1"]["JULIET"]["acc"],
                     "juliet_surg_nll": c["r1"]["JULIET"]["nll"],
                     "juliet_surg_acc": c["r1"]["JULIET"]["acc"],
                     "d_ce_val_all": c["d_ce"]["val_all"],
                     "s_name": c["s_name"], "s_letter": c["s_letter"],
                     "n_elements": c["g2"]["n_elements_changed"]}
    except Exception as e:
        ref["l6"] = {"error": str(e)}
    return ref


# ------------------------------------------------------------------ main

def train_net(tag: str, cfg_kw: dict, corpus, cap_s: float, total_s: float | None = None):
    """252s cap; LARGE may extend to total_s=400 across ckpt resumes."""
    ckpt_final = CKPT_DIR / f"e005s_{tag}.pt"
    ckpt_train = CKPT_DIR / f"e005s_{tag}.train.pt"
    set_seed(SEED)
    model = TinyGPT(Cfg(vocab=corpus.vocab_size, block_size=256, **cfg_kw)).to(DEVICE)
    n_params = model.num_params()
    if ckpt_final.exists():
        model.load_state_dict(torch.load(ckpt_final, map_location=DEVICE, weights_only=True))
        log(f"{tag}: checkpoint found ({n_params:,} params), skipping training")
        return model, n_params, None
    prior = 0.0
    if ckpt_train.exists():
        st = torch.load(ckpt_train, map_location="cpu", weights_only=False)
        hist_prev = st.get("history") or []
        # elapsed_s is per-call wall time; a single prior segment => its total
        prior = float(hist_prev[-1]["elapsed_s"]) if hist_prev else 0.0
    budget = cap_s if (total_s is None or prior <= 0.0) else max(60.0, total_s - prior)
    log(f"{tag}: training {n_params:,} params (budget {budget:.0f}s, seed {SEED}, "
        f"prior segment {prior:.0f}s)")
    hist = train_model(model, corpus, steps=4000, lr=1e-3, batch_size=64,
                       max_seconds=budget, ckpt=ckpt_train)
    steps_done = hist[-1]["step"]
    torch.save(model.state_dict(), ckpt_final)
    log(f"{tag}: trained to step {steps_done} (budget {budget:.0f}s), "
        f"val {hist[-1]['val_loss']:.4f}")
    return model, n_params, {"steps": steps_done, "budget_s": budget,
                             "prior_segment_s": prior,
                             "final_val_loss": hist[-1]["val_loss"]}


def main():
    rd = run_dir("e005s")
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    assert corpus.vocab_size == 65
    train_text = "".join(corpus.itos[int(i)] for i in corpus.train)

    results = {}
    train_meta = {}
    params = {}
    for tag, cfg_kw in NETS.items():
        cap, total = (252.0, None) if tag == "small" else (252.0, 400.0)
        model, n_params, tmeta = train_net(tag, cfg_kw, corpus, cap, total)
        model.eval()
        train_meta[tag] = tmeta
        params[tag] = n_params
        cfg = cfg_dict(model.cfg)

        log(f"{tag}: L1 causal census ({N_POS} positions)")
        l1 = causal_census(model, corpus)
        log(f"{tag}: L1 flip curve {[f'{v:.3f}' for v in l1['flip_curve_per_depth']]} "
            f"mode {l1['causal_mode']} suffix-mono {l1['suffix_monotone_flip_fraction']:.0%} "
            f"gate={l1['gate_exists']}")

        log(f"{tag}: L2 lesion map")
        base_val = estimate_loss(model, corpus, "val", n_batches=30)
        attn_d, mlp_d = [], []
        for i in range(model.cfg.n_layer):
            attn_d.append(lesion_loss(model, corpus, "attn", i, n_batches=30) - base_val)
            mlp_d.append(lesion_loss(model, corpus, "mlp", i, n_batches=30) - base_val)
        l2 = {"baseline_val_loss": base_val,
              "attn_block_damage": attn_d, "mlp_block_damage": mlp_d,
              "attn_l0_over_last": (attn_d[0] / attn_d[-1]) if attn_d[-1] > 1e-9 else float("inf")}
        log(f"{tag}: L2 attn {[f'{v:+.2f}' for v in attn_d]} | L0/last "
            f"{l2['attn_l0_over_last']:.1f}x")

        log(f"{tag}: L4 far-value tail")
        l4 = far_value(model, corpus)
        log(f"{tag}: L4 mean far-value {l4['far_value_mean']:+.4f}")

        log(f"{tag}: L6 D2 row surgery")
        l6 = l6_readout(model, corpus, train_text)
        sn_str = "inf" if math.isinf(l6["s_name"]) else f"{l6['s_name']:.1f}"
        log(f"{tag}: L6 JULIET {l6['baseline']['JULIET']['nll']:.3f}->"
            f"{l6['surgery']['JULIET']['nll']:.3f} dCE {l6['d_ce_val_all']:+.5f} "
            f"S_name {sn_str}")

        results[tag] = {"params": n_params, "config": cfg, "train": tmeta,
                        "l1_gate": l1, "l2_front_loading": l2,
                        "l4_far_value": l4, "l6_row_surgery": l6}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    b_ref = load_b_ref()
    b_params = b_ref.get("l2", {}).get("params")

    # ---------------- registered verdicts ----------------
    def p2_ok(r):
        a = r["l2_front_loading"]
        return bool(a["attn_l0_over_last"] >= 2.0)

    verdicts = {
        "P1_gate_exists_small": results["small"]["l1_gate"]["gate_exists"],
        "P1_gate_exists_large": results["large"]["l1_gate"]["gate_exists"],
        "P1_holds": bool(results["small"]["l1_gate"]["gate_exists"]
                         and results["large"]["l1_gate"]["gate_exists"]),
        "P2_front_loading_small": p2_ok(results["small"]),
        "P2_front_loading_large": p2_ok(results["large"]),
        "P2_holds": bool(p2_ok(results["small"]) and p2_ok(results["large"])),
        "P3_far_value_small": bool(results["small"]["l4_far_value"]["far_value_mean"] < 0.05),
        "P3_far_value_large": bool(results["large"]["l4_far_value"]["far_value_mean"] < 0.05),
        "P3_holds": bool(results["small"]["l4_far_value"]["far_value_mean"] < 0.05
                         and results["large"]["l4_far_value"]["far_value_mean"] < 0.05),
        "P4_sname_bar": P4_BAR, "P4_naive_reference": NAIVE_SEL,
        "P4_row_surgery_small": results["small"]["l6_row_surgery"]["p4_s_name_ge_bar"],
        "P4_row_surgery_large": results["large"]["l6_row_surgery"]["p4_s_name_ge_bar"],
        "P4_holds": bool(results["small"]["l6_row_surgery"]["p4_s_name_ge_bar"]
                         and results["large"]["l6_row_surgery"]["p4_s_name_ge_bar"]),
    }

    metrics = {
        "experiment": "e005s_scaling", "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": SEED, "smoke": False,
        "registered": "P1-P4 in lab/e005s_scaling.py header (written before the run)",
        "params_verified": {"small": params["small"], "large": params["large"],
                            "B_reference": b_params,
                            "note": ("dims as specified verify to 0.84M / 9.98M, ~20-25% "
                                     "over the ~0.7M / ~8M labels; dims were the frozen spec")},
        "training": {"protocol": "train_model steps=4000 lr=1e-3 batch=64 cosine, seed 42",
                     "small": train_meta["small"], "large": train_meta["large"]},
        "nets": results,
        "B_reference_loaded": b_ref,
        "verdicts": verdicts,
        "timing_s": round(time.time() - T0, 1),
    }

    # ---------------- composite figure: 4 readouts x 3 scales ----------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = {"small": "seagreen", "large": "purple", "B": "steelblue"}

    ax = axes[0, 0]
    series = {"B": b_ref["l1"], **{t: results[t]["l1_gate"] for t in NETS}}
    for tag, s in series.items():
        fc = s["flip_curve_per_depth"]
        x = [i / (len(fc) - 1) for i in range(len(fc))]
        ax.plot(x, fc, "o-", color=colors[tag],
                label=f"{tag} (L={len(fc)}, mode {s['causal_mode']}, "
                      f"suf-mono {s['suffix_monotone_flip_fraction']:.0%})")
    ax.set_xlabel("relative depth d/(L-1)")
    ax.set_ylabel("P(flip | patch at d)")
    ax.set_title("L1 causal gate — flip curves at three scales")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    l2b = b_ref["l2"]
    for tag, s, L in [("B", l2b, 6), ("small", results["small"]["l2_front_loading"], 4),
                      ("large", results["large"]["l2_front_loading"], 8)]:
        x = [i / (L - 1) for i in range(L)]
        ax.plot(x, s["attn_block_damage"], "o-", color=colors[tag], label=f"{tag} attn")
        ax.plot(x, s["mlp_block_damage"], "s--", color=colors[tag], alpha=0.5, label=f"{tag} mlp")
    ax.set_xlabel("relative layer i/(L-1)")
    ax.set_ylabel("damage: d val CE (nats)")
    ax.set_title("L2 zero-ablation damage profiles (front-loading)")
    ax.legend(fontsize=7, ncol=3)

    ax = axes[1, 0]
    tags = ["B", "small", "large"]
    fv_means = [b_ref["l4"]["far_value_mean"], results["small"]["l4_far_value"]["far_value_mean"],
                results["large"]["l4_far_value"]["far_value_mean"]]
    bars = ax.bar(tags, fv_means, color=[colors[t] for t in tags])
    ax.axhline(0.05, color="crimson", ls="--", lw=1, label="P3 bar +0.05")
    for b, v in zip(bars, fv_means):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.4f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("mean far-value = CE(16) - CE(256) (nats)")
    ax.set_title("L4 16-token sufficiency at three scales")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    sn = [b_ref["l6"]["s_name"], results["small"]["l6_row_surgery"]["s_name"],
          results["large"]["l6_row_surgery"]["s_name"]]
    dj = [b_ref["l6"]["juliet_surg_nll"] - b_ref["l6"]["juliet_base_nll"],
          results["small"]["l6_row_surgery"]["d_nll"]["JULIET"],
          results["large"]["l6_row_surgery"]["d_nll"]["JULIET"]]
    dc = [b_ref["l6"]["d_ce_val_all"], results["small"]["l6_row_surgery"]["d_ce_val_all"],
          results["large"]["l6_row_surgery"]["d_ce_val_all"]]
    w = 0.35
    xs = range(len(tags))
    ax.bar([i - w / 2 for i in xs], dj, w, color="navy", label="dNLL JULIET (nats)")
    ax.bar([i + w / 2 for i in xs], dc, w, color="gray", label="dCE corpus (nats)")
    ax.set_yscale("symlog", linthresh=0.01)
    for i, (s, d) in enumerate(zip(sn, dj)):
        ax.text(i, d, f"S_name={'inf' if math.isinf(s) else f'{s:.0f}'}",
                ha="center", va="bottom", fontsize=9, color="crimson")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(list(xs)); ax.set_xticklabels(tags)
    ax.set_title("L6 D2 row surgery (J rows, both matrices)")
    ax.legend(fontsize=8)

    fig.suptitle("E005s scaling capstone — four frozen readouts across three scales "
                 f"(P1 {verdicts['P1_holds']} / P2 {verdicts['P2_holds']} / "
                 f"P3 {verdicts['P3_holds']} / P4 {verdicts['P4_holds']})")
    fig.tight_layout()
    fig.savefig(rd / "scaling_readouts.png", dpi=140)
    plt.close(fig)

    save_json(rd / "metrics.json", metrics)

    log("=== REGISTERED VERDICTS ===")
    for k in ("P1_holds", "P2_holds", "P3_holds", "P4_holds"):
        log(f"{k}: {verdicts[k]}")
    log(f"params: small {params['small']:,} | large {params['large']:,} | B {b_params:,}")
    log(f"outputs: {rd}")
    log(f"total {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
