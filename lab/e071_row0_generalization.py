"""E071 — T042-registered row-0 generalization discriminator (eval-only).

Question (T042, two explanations for e067's row-0 dominance):
  H-window-key   : the install protocol made position 0 a high-precision
                   window-recognition key for INSTALL-type windows ->
                   row-0 damage hits install-60 AND held-30, spares uniform
                   contexts, spares the base net.
  H-generic-start: window-start anchoring is a generic circuit feature of
                   this net family -> row-0 damage hits EVERYWHERE
                   (including the base net and uniform contexts).

REGISTERED DESIGN (frozen in THINKING.md T042 before compute):
  row-0 intervention, BOTH arms (row <- mean of all 256 rows [e066b A3];
  row <- 0 [e066b zero arm]) x batteries {install-60 (the trained windows),
  held-30 (host_occ[60:90] of the same protocol rebuild — never trained in
  any form), uniform-random ~60 corpus windows (no host-name/install
  structure)} x nets {installed e048_repro.pt, base e001.pt}.
  Readout: mean battery p(Z) drop per (battery, net, arm).
  H-window-key: big drop on install-60 AND held-30, small on uniform,
                small everywhere on base net.
  H-generic-start: big drop everywhere including base net and uniform.
  Report honestly if MIXED.
  Secondary (registered): row-129 perturbation x held-30 on the installed
  net — does the address row generalize beyond the trained windows?

OPERATIONALIZATION of "big"/"small" (T042 fixed no numbers; set here before
compute, anchored on e067's census scale: row-0 drop 0.555, secondary top
rows 0.03-0.04, micro-carpet ~0.002, null band 0.0):
  BIG   : drop >= 0.10 absolute p(Z) points
  SMALL : drop <  0.02
  FLOOR-LIMITED: a cell whose base p(Z) < 0.05 cannot show a big drop by
  construction; its "small" is counted as uninformative (auto-pass), never
  as confirmatory, and is flagged in metrics/plot.

FLOOR-FREE ADD-ON (secondary, NOT registered, honesty aid): per cell the
mean KL(base || perturbed) over the battery's full next-char distributions,
so the uniform / base-net legs (p(Z) floor ~0) are not silently
uninterpretable in the registered absolute-drop readout.

Protocol machinery reused verbatim from e067_address_census (itself the
exact e055/e066/e066b rebuild): corpus seed 1337, E43.SPLICE_RNG shuffle,
install/held split host_occ[:60]/[60:90], 130-char pre-name contexts,
in-place wpe-row <- mean / zero arms, CPU only. Uniform battery: e048's
UNIF_SEED draw, rejection-filtered so no window touches a host name
(the registered "no host-name/install structure" reading).

Outputs: runs/e071/{metrics.json, row0_generalization.png}.
Run: python lab/e071_row0_generalization.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")          # CPU experiment

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import torch.nn.functional as F                             # noqa: E402

torch.set_num_threads(min(16, os.cpu_count() or 8))

import common                                               # noqa: E402
common.DEVICE = "cpu"
from common import Cfg, CharCorpus, TinyGPT, run_dir, save_json   # noqa: E402
import e043_install as E43                                  # noqa: E402
import matplotlib                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                             # noqa: E402

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
HOSTS = ["FLORIZEL", "ELIZABETH"]
INSTALLED_CK = "e048_repro.pt"
BASE_CK = "e001.pt"
ROW0, ROW129 = 0, 129
CONTROL_ROW = 60             # fed, mid-window, micro-carpet-level in e067's
                             # census -> interpretive control for the KL arm
UNIF_SEED = 24801            # e048's uniform-battery seed
N_UNIF = 60
NULL_ROWS = (160, 200, 240)  # never fed for 130-char contexts -> drop must be 0

BIG_BAR = 0.10               # >= this => "big" drop
SMALL_BAR = 0.02             # <  this => "small" drop
FLOOR_PZ = 0.05              # base p(Z) below this => cell is floor-limited


def load(path: Path):
    m = TinyGPT(Cfg())
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def battery(m, ids, zid, bs=30):
    """e067's battery, extended: per-window p(Z) + full next-char probs."""
    pzs, prs = [], []
    for i in range(0, ids.shape[0], bs):
        lg, _ = m(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs += [float(pr[k, zid]) for k in range(pr.shape[0])]
        prs.append(pr.numpy())
    return np.array(pzs, dtype=np.float64), np.concatenate(prs)


def kl_rows(p, q, eps=1e-12):
    """KL(p || q) per row for probability matrices [N, V]."""
    P, Q = p + eps, q + eps
    P, Q = P / P.sum(1, keepdims=True), Q / Q.sum(1, keepdims=True)
    return np.sum(P * np.log(P / Q), axis=1)


def main():
    rd = run_dir("e071")
    log("E071 row-0 generalization discriminator (T042 registered)")

    # ---- exact e055/e066/e066b/e067 protocol rebuild (verbatim)
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi = corpus.stoi
    train_ids = corpus.train
    train_text = "".join(corpus.itos[int(i)] for i in train_ids)
    zid = stoi["Z"]

    host_occ = []
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + 119 <= len(train_ids):
                host_occ.append((p, host))
    import random as _random
    rng = _random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    assert mix == {"FLORIZEL": 19, "ELIZABETH": 41}, f"splice drift {mix}"

    ctx_i = [train_text[p - PRE: p] for p, _ in install_occ]     # trained 60
    ctx_h = [train_text[p - PRE: p] for p, _ in held_occ]        # held-out 30

    # ---- uniform battery: e048's seed, rejection-filtered against hosts
    ug = torch.Generator().manual_seed(UNIF_SEED)
    uq = torch.randint(PRE, len(train_ids) - 8, (6000,), generator=ug)
    ctx_u, rejected, seen = [], 0, set()
    for q in uq.tolist():
        if q in seen:
            continue
        seen.add(q)
        span = train_text[q - PRE: q + 8]
        if any(h in span for h in HOSTS):
            rejected += 1
            continue
        ctx_u.append(train_text[q - PRE: q])
        if len(ctx_u) == N_UNIF:
            break
    assert len(ctx_u) == N_UNIF
    log(f"protocol rebuilt: install60 {mix}, held30, uniform60 "
        f"(host-overlap rejections {rejected}/{len(seen)})")

    ctx_ids = {
        "install60": torch.stack([corpus.encode(c) for c in ctx_i]),
        "held30": torch.stack([corpus.encode(c) for c in ctx_h]),
        "uniform60": torch.stack([corpus.encode(c) for c in ctx_u]),
    }

    ARMS = ("mean", "zero")

    def arm_perturb(net, w_orig, row, arm):
        w = net.wpe.weight.data
        w.copy_(w_orig)
        if arm == "mean":
            w[row] = w_orig.mean(0)
        elif arm == "zero":
            w[row] = 0.0
        else:
            raise ValueError(arm)

    cells = {}          # (net_tag, bat) -> dict
    row129 = {}         # installed net only: (bat, arm) -> dict
    control = {}        # (net_tag, bat, arm) -> dict; interpretive only
    null_check = {}

    for net_tag, ck in [("installed", INSTALLED_CK), ("base", BASE_CK)]:
        net = load(E43.REPO / "runs" / "checkpoints" / ck)
        w_orig = net.wpe.weight.data.clone()
        log(f"[{net_tag}] {ck}: params {net.num_params():,}, "
            f"row0 norm {float(w_orig[0].norm()):.3f}, "
            f"row129 norm {float(w_orig[129].norm()):.3f}")

        for bat, ids in ctx_ids.items():
            pz0, pr0 = battery(net, ids, zid)
            cell = {"base_pz": float(pz0.mean()),
                    "base_pz_per_ctx": pz0.tolist(),
                    "floor_limited": bool(pz0.mean() < FLOOR_PZ),
                    "arms": {}}
            for arm in ARMS:
                arm_perturb(net, w_orig, ROW0, arm)
                pz1, pr1 = battery(net, ids, zid)
                cell["arms"][arm] = {
                    "pert_pz": float(pz1.mean()),
                    "pert_pz_per_ctx": pz1.tolist(),
                    "drop": float(pz0.mean() - pz1.mean()),
                    "rel_drop": float((pz0.mean() - pz1.mean())
                                      / max(pz0.mean(), 1e-12)),
                    "kl_base_pert": float(kl_rows(pr0, pr1).mean()),
                }
            net.wpe.weight.data.copy_(w_orig)

            # registered secondary: row-129 x held-30 on the installed net
            # (install-60 and uniform-60 also recorded as within-run context)
            if net_tag == "installed":
                for arm in ARMS:
                    arm_perturb(net, w_orig, ROW129, arm)
                    pz1, pr1 = battery(net, ids, zid)
                    row129[(bat, arm)] = {
                        "pert_pz": float(pz1.mean()),
                        "drop": float(pz0.mean() - pz1.mean()),
                        "kl_base_pert": float(kl_rows(pr0, pr1).mean()),
                    }
                net.wpe.weight.data.copy_(w_orig)

            # interpretive control (NOT registered): a fed but census-light
            # row, same arms — makes the floor-free KL lens quantitative
            # (is row-0's distribution-level effect special, or any row's?)
            for arm in ARMS:
                arm_perturb(net, w_orig, CONTROL_ROW, arm)
                pz1, pr1 = battery(net, ids, zid)
                control[(net_tag, bat, arm)] = {
                    "pert_pz": float(pz1.mean()),
                    "drop": float(pz0.mean() - pz1.mean()),
                    "kl_control_pert": float(kl_rows(pr0, pr1).mean()),
                }
            net.wpe.weight.data.copy_(w_orig)

            cells[(net_tag, bat)] = cell
            fl = " FLOOR-LIMITED" if cell["floor_limited"] else ""
            a = cell["arms"]
            log(f"[{net_tag}] {bat}: base p(Z) {cell['base_pz']:.4f}{fl} | "
                + " | ".join(f"row0-{arm}: pZ {a[arm]['pert_pz']:.4f} "
                             f"drop {a[arm]['drop']:.4f} "
                             f"KL {a[arm]['kl_base_pert']:.3f}"
                             for arm in ARMS))
            if net_tag == "installed":
                log(f"[{net_tag}] {bat}: row129 "
                    + " | ".join(f"{arm}: drop {row129[(bat, arm)]['drop']:.4f}"
                                 for arm in ARMS))

        # sanity null: never-fed rows must give exactly-zero effect
        if net_tag == "installed":
            pz0, _ = battery(net, ctx_ids["install60"], zid)
            for r in NULL_ROWS:
                arm_perturb(net, w_orig, r, "mean")
                pz1, _ = battery(net, ctx_ids["install60"], zid)
                null_check[r] = float(pz0.mean() - pz1.mean())
            net.wpe.weight.data.copy_(w_orig)
            assert max(abs(v) for v in null_check.values()) < 1e-9, null_check
            log(f"null check (never-fed rows, mean arm): {null_check}")

    # ---- verdict (registered mapping; floors auto-pass smallness but are
    #      never counted as confirmatory)
    def big(net_tag, bat, arm):
        return cells[(net_tag, bat)]["arms"][arm]["drop"] >= BIG_BAR

    def small_or_floor(net_tag, bat, arm):
        c = cells[(net_tag, bat)]
        return (c["arms"][arm]["drop"] < SMALL_BAR) or c["floor_limited"]

    bts = ("install60", "held30", "uniform60")
    hwk_pos = all(big("installed", b, a) for b in ("install60", "held30")
                  for a in ARMS)
    hwk_neg = (all(small_or_floor("installed", "uniform60", a) for a in ARMS)
               and all(small_or_floor("base", b, a) for b in bts for a in ARMS))
    h_window_key = bool(hwk_pos and hwk_neg)
    h_generic_start = bool(all(big(n, b, a) for n in ("installed", "base")
                               for b in bts for a in ARMS))
    verdict = ("H-GENERIC-START" if h_generic_start else
               "H-WINDOW-KEY" if h_window_key else "MIXED")
    floor_cells = [f"{n}/{b}" for (n, b), c in cells.items() if c["floor_limited"]]
    log(f"VERDICT: {verdict} (window-key pos {hwk_pos} neg {hwk_neg}; "
        f"generic-start all-big {h_generic_start}; floor-limited cells "
        f"{floor_cells})")

    held = cells[("installed", "held30")]
    sec129 = ("row-129 on held-30: drop mean-arm "
              f"{row129[('held30', 'mean')]['drop']:.3f} / zero-arm "
              f"{row129[('held30', 'zero')]['drop']:.3f} "
              f"(install-60: {row129[('install60', 'mean')]['drop']:.3f} / "
              f"{row129[('install60', 'zero')]['drop']:.3f}; held base p(Z) "
              f"{held['base_pz']:.3f})")
    log(sec129)

    # floor-free diagnostic (interpretive): row-0 KL vs control-row KL
    kl_ratio = {f"{n}__{b}__{a}":
                cells[(n, b)]["arms"][a]["kl_base_pert"]
                / max(control[(n, b, a)]["kl_control_pert"], 1e-12)
                for n in ("installed", "base") for b in bts for a in ARMS}
    for k, v in kl_ratio.items():
        log(f"KL ratio row0/control(row {CONTROL_ROW}) {k}: {v:.2f}")

    out = {
        "experiment": "e071_row0_generalization",
        "registered": {
            "design": ("row-0 intervention x {install-60, held-30, "
                       "uniform-60} x {e048_repro installed, e001 base}; "
                       "arms row<-mean(all 256) and row<-0; readout mean "
                       "battery p(Z) drop per cell"),
            "h_window_key": "big on install-60 AND held-30, small on "
                            "uniform, small everywhere on base net",
            "h_generic_start": "big drop everywhere including base and "
                               "uniform",
            "secondary": "row-129 x held-30 on installed net",
            "bars_operationalized_here_not_in_T042": {
                "big_bar": BIG_BAR, "small_bar": SMALL_BAR,
                "floor_pz": FLOOR_PZ,
                "note": "bars anchored on e067 census scale (row0 0.555, "
                        "secondary rows 0.03-0.04, micro-carpet 0.002); "
                        "floor-limited cells auto-pass smallness, flagged, "
                        "never confirmatory"},
        },
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "n_held": len(held_occ),
                     "uniform": {"seed": UNIF_SEED, "n": N_UNIF,
                                 "host_overlap_rejections": rejected,
                                 "draws_seen": len(seen)}},
        "cells": {f"{n}__{b}": cells[(n, b)] for (n, b) in cells},
        "row129_installed": {f"{b}__{a}": row129[(b, a)]
                             for (b, a) in row129},
        "control_row": {"row": CONTROL_ROW,
                        "note": "interpretive control for the KL lens; "
                                "NOT part of the registered design",
                        "cells": {f"{n}__{b}__{a}": control[(n, b, a)]
                                  for (n, b, a) in control}},
        "kl_ratio_row0_over_control": kl_ratio,
        "null_check_neverfed_rows_mean_arm": null_check,
        "verdict": {"verdict": verdict,
                    "h_window_key": h_window_key,
                    "h_generic_start": h_generic_start,
                    "window_key_pos_legs": hwk_pos,
                    "window_key_neg_legs": hwk_neg,
                    "floor_limited_cells": floor_cells,
                    "floor_caveat": ("uniform and all base-net legs are "
                                     "floor-limited (base p(Z) < 0.05): "
                                     "'small' there is uninformative, not "
                                     "confirmatory"),
                    "secondary_row129_held30": sec129},
        "references": {"e067_row0_install60_drop_mean_arm": 0.55495,
                       "e067_row0_install60_drop_zero_arm": 0.54553,
                       "e067_row129_install60_drop_mean_arm": 0.24047,
                       "e067_row129_install60_drop_zero_arm": 0.34151,
                       "e048_held30_base_pz_installed": 0.42871,
                       "e048_uniform60_base_pz_installed": 0.0},
        "elapsed_s": round(time.time() - T0, 1),
        "cpu_threads": torch.get_num_threads(),
    }
    save_json(rd / "metrics.json", out)

    # ---- figure: grouped bars, drop per battery x net, both arms
    nets = ("installed", "base")
    colors = {("installed", "mean"): "crimson", ("installed", "zero"): "salmon",
              ("base", "mean"): "dimgray", ("base", "zero"): "silver"}
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))

    ax = axes[0]
    xs = np.arange(len(bts))
    bw = 0.19
    for k, (n, a) in enumerate([("installed", "mean"), ("installed", "zero"),
                                ("base", "mean"), ("base", "zero")]):
        vals = [cells[(n, b)]["arms"][a]["drop"] for b in bts]
        bars = ax.bar(xs + (k - 1.5) * bw, vals, bw,
                      color=colors[(n, a)], label=f"{n} net, {a} arm",
                      edgecolor="k", linewidth=0.4)
        for x, v in zip(xs + (k - 1.5) * bw, vals):
            ax.text(x, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.5,
                    rotation=90, va="bottom")
    ax.axhline(BIG_BAR, color="crimson", ls="--", lw=1.1,
               label=f"BIG bar {BIG_BAR:.2f}")
    ax.axhline(SMALL_BAR, color="gray", ls=":", lw=1.1,
               label=f"SMALL bar {SMALL_BAR:.2f}")
    labels = []
    for b in bts:
        fl = ("*" if cells[("installed", b)]["floor_limited"] else "") + \
             ("*" if cells[("base", b)]["floor_limited"] else "")
        labels.append(b + fl)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("drop in mean battery p(Z) from row-0 perturbation")
    ax.set_title("PRIMARY (registered): row-0 p(Z) drop per battery x net\n"
                 "* = floor-limited cell (base p(Z) < "
                 f"{FLOOR_PZ:.2f}; smallness uninformative)")
    ax.legend(fontsize=7)
    ax.set_ylim(0, max(0.62, max(cells[(n, b)]["arms"][a]["drop"]
                                 for n in nets for b in bts for a in ARMS)
                       * 1.18))

    ax = axes[1]
    for k, n in enumerate(nets):
        vals = [cells[(n, b)]["base_pz"] for b in bts]
        ax.bar(xs + (k - 0.5) * 0.35, np.clip(vals, 1e-9, None), 0.35,
               color="crimson" if n == "installed" else "dimgray",
               label=f"{n} net", edgecolor="k", linewidth=0.4)
        for x, v in zip(xs + (k - 0.5) * 0.35, vals):
            ax.text(x, v * 1.2 if v > 0 else 2e-7,
                    f"{v:.2e}" if v < 0.01 else f"{v:.3f}",
                    ha="center", fontsize=7, rotation=90, va="bottom")
    ax.axhline(FLOOR_PZ, color="gray", ls="--", lw=1,
               label=f"FLOOR {FLOOR_PZ:.2f}")
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 2)
    ax.set_xticks(xs)
    ax.set_xticklabels(bts)
    ax.set_ylabel("base battery p(Z) (log)")
    ax.set_title("floors: unperturbed p(Z) per cell (why absolute drops\n"
                 "are bounded on uniform / base legs)")
    ax.legend(fontsize=8)

    ax = axes[2]
    for k, (n, a) in enumerate([("installed", "mean"), ("installed", "zero"),
                                ("base", "mean"), ("base", "zero")]):
        vals = [cells[(n, b)]["arms"][a]["kl_base_pert"] for b in bts]
        ax.bar(xs + (k - 1.5) * bw, np.clip(vals, 1e-9, None), bw,
               color=colors[(n, a)], label=f"{n} net, {a} arm",
               edgecolor="k", linewidth=0.4)
        for x, v in zip(xs + (k - 1.5) * bw, vals):
            ax.text(x, max(v, 1e-5) * 1.25, f"{v:.3f}", ha="center",
                    fontsize=6.5, rotation=90, va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1e2)
    ax.set_xticks(xs)
    ax.set_xticklabels(bts)
    ax.set_ylabel("mean KL(base || row-0-perturbed), next-char (log)")
    ax.set_title("SECONDARY (NOT registered, floor-free): full-distribution\n"
                 "row-0 effect — the honest uniform/base-leg lens")
    ax.legend(fontsize=7)

    fig.suptitle(f"e071 T042 row-0 generalization discriminator — "
                 f"VERDICT: {verdict}\n{sec129}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(rd / "row0_generalization.png", dpi=140)
    log(f"done -> {rd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
