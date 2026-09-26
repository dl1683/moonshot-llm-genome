"""E066 — CLOSE THE LOOP (P1 COORDINATE ramp, step 1; registered in
scratch/post_paper_programs.md P1 "first 3" #1 and T037).

Registered question: is the e056b mean-donor relay direction at d5 the
same object as the wpe-130 address row, or two distinct "addresses"?

Registered verdict bar (frozen pre-run):
  |cos(relay_d5, wpe[130])| >= 0.4  -> LOOP CLOSED (position row feeds
                                       the state direction)
  |cos| ~ 0 (<= 0.15, null sd 1/sqrt(192) ~ 0.072) -> TWO OBJECTS.
  In between (0.15-0.4) -> PARTIAL: report, no framing change.

Instruments:
  - exact e055 protocol rebuild (corpus 1337, splice RNG, install-60,
    battery ranking, primary4+replication2 donors) on e048_repro.pt;
  - relay_d = mean donor decision-state (pos 129) per depth 0..6;
  - full wpe sweep per depth (max |cos|, its row, rank of 129/130);
  - delta relay (relay - shuffled-family mean at d5) as the direction-
    purified variant;
  - controls: wte sweep at d5, adjacent-row correlations (wpe 128-131),
    null sd, wpe-vs-wpe norm context.

Zero training. CPU only. Outputs: runs/e066/{metrics.json, wpe_cos_sweep.png}.
Run: python lab/e066_loop_closer.py
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
import matplotlib.pyplot as plt                             # noqa: E402

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
HOSTS = ["FLORIZEL", "ELIZABETH"]
SHUF_SEED = 25501
N_SHUF = 4
NULL_SD = 1.0 / np.sqrt(192.0)


def load(path: Path):
    m = TinyGPT(Cfg())
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def states_forward(net, toks):
    toks = toks.unsqueeze(0)
    T = toks.shape[1]
    x = net.wte(toks) + net.wpe(torch.arange(T))
    xs = [x]
    for block in net.h:
        x = block(x)
        xs.append(x)
    return xs


def cos(a, b):
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def main():
    rd = run_dir("e066")
    log("E066 close-the-loop: relay direction vs wpe address row")

    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi = corpus.stoi
    train_ids = corpus.train
    train_text = "".join(corpus.itos[int(i)] for i in train_ids)
    zid = stoi["Z"]

    CK = E43.REPO / "runs" / "checkpoints"
    net = load(CK / "e048_repro.pt")
    log(f"installed net loaded; params {net.num_params():,}")

    # ---- exact e055 protocol rebuild
    host_occ = []
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + 119 <= len(train_ids):
                host_occ.append((p, host))
    import random as _random
    rng = _random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, _held = host_occ[:60], host_occ[60:90]
    ctx130_i = [train_text[p - PRE: p] for p, _ in install_occ]

    @torch.no_grad()
    def battery_pz(m, ctxs, bs=30):
        rows = []
        for i in range(0, len(ctxs), bs):
            ids = torch.stack([corpus.encode(c) for c in ctxs[i:i + bs]])
            lg, _ = m(ids)
            pr = F.softmax(lg[:, -1], -1)
            rows += [float(pr[k, zid]) for k in range(len(ctxs[i:i + bs]))]
        return rows

    bat = battery_pz(net, ctx130_i)
    order = sorted(range(60), key=lambda i: -bat[i])
    primary4 = list(dict.fromkeys(order[:2]
                    + [i for i in order[2:] if install_occ[i][1] == "FLORIZEL"][:2]))[:4]
    replication2 = [i for i in order if i not in primary4][:2]
    donor_idx = primary4 + replication2
    log(f"donors {donor_idx} (battery p_z "
        + " ".join(f"{bat[i]:.2f}" for i in donor_idx) + ")")

    # ---- donor states + relay per depth
    dstates = {d: [] for d in range(7)}
    for i in donor_idx:
        xs = states_forward(net, corpus.encode(ctx130_i[i]))
        for d in range(7):
            dstates[d].append(xs[d][0, -1].clone())
    relay = {d: torch.stack(dstates[d]).mean(0) for d in range(7)}

    # ---- shuffled-family states (M5 family) for the delta relay at d5
    _s = _random.Random(SHUF_SEED)
    shuf_ctxs = []
    while len(shuf_ctxs) < N_SHUF:
        q = _s.randrange(PRE + 1, len(train_ids) - 1)
        shuf_ctxs.append(train_text[q - PRE: q])
    sstates = {d: [] for d in range(7)}
    for c in shuf_ctxs:
        xs = states_forward(net, corpus.encode(c))
        for d in range(7):
            sstates[d].append(xs[d][0, -1].clone())
    shuf_mean = {d: torch.stack(sstates[d]).mean(0) for d in range(7)}

    # ---- wpe sweep per depth
    wpe = net.wpe.weight.detach()          # [256, 192]
    sweep = {}
    for d in range(7):
        cs = F.cosine_similarity(wpe, relay[d].unsqueeze(0)).squeeze(0)
        ac = cs.abs()
        top = int(ac.argmax())
        sweep[d] = {
            "max_abs_cos": float(ac[top]), "max_row": top,
            "cos_129": float(cs[129]), "cos_130": float(cs[130]),
            "abs_cos_129": float(ac[129]), "abs_cos_130": float(ac[130]),
            "rank_130": int((ac > ac[130]).sum()) + 1,
            "rank_129": int((ac > ac[129]).sum()) + 1,
        }
        log(f"d{d}: max|cos| {float(ac[top]):.3f} @ row {top}; "
            f"row129 {float(cs[129]):+.3f} row130 {float(cs[130]):+.3f} "
            f"(rank130 {sweep[d]['rank_130']})")

    # ---- delta relay (direction purified against generic last-pos state)
    delta5 = relay[5] - shuf_mean[5]
    dcs = F.cosine_similarity(wpe, delta5.unsqueeze(0)).squeeze(0)
    dac = dcs.abs()
    dtop = int(dac.argmax())

    # ---- controls
    wte = net.wte.weight.detach()          # [65, 192]
    wte_cs = F.cosine_similarity(wte, relay[5].unsqueeze(0)).squeeze(0)
    wte_top = int(wte_cs.abs().argmax())
    adj = {"wpe129_130": cos(wpe[129], wpe[130]),
           "wpe128_129": cos(wpe[128], wpe[129]),
           "wpe130_131": cos(wpe[130], wpe[131]),
           "wpe130_mean_abs_cos_all_rows": float(wpe @ wpe[130] /
                                                 (wpe.norm(dim=1) * wpe[130].norm())).item()
           if False else float((F.cosine_similarity(wpe, wpe[130].unsqueeze(0))
                                .abs().mean()))}

    # ---- registered verdict (primary: raw relay d5 vs wpe[130])
    c130 = sweep[5]["abs_cos_130"]
    if c130 >= 0.4:
        verdict = "LOOP-CLOSED"
    elif c130 <= 0.15:
        verdict = "TWO-OBJECTS"
    else:
        verdict = "PARTIAL"
    c130_delta = float(dac[130])
    log(f"REGISTERED METRIC |cos(relay_d5, wpe130)| = {c130:.3f} -> {verdict}")
    log(f"delta-variant |cos(relay_d5 - shuf, wpe130)| = {c130_delta:.3f} "
        f"(max {float(dac[dtop]):.3f} @ row {dtop})")
    log(f"controls: wte max|cos| {float(wte_cs.abs()[wte_top]):.3f} @ row "
        f"'{corpus.itos[wte_top]}' | wpe129~130 cos {adj['wpe129_130']:.3f} "
        f"| mean |wpe row~wpe130| {adj['wpe130_mean_abs_cos_all_rows']:.3f} "
        f"| null sd {NULL_SD:.3f}")

    metrics = {
        "experiment": "e066_loop_closer",
        "registered_metric": "abs_cos(relay_d5, wpe[130])",
        "registered_bars": {"closed": 0.4, "two_objects_max": 0.15},
        "result": {"abs_cos_130": c130, "verdict": verdict,
                   "cos_130_signed": sweep[5]["cos_130"],
                   "delta_variant_abs_cos_130": c130_delta,
                   "delta_variant_max": {"abs_cos": float(dac[dtop]), "row": dtop}},
        "wpe_sweep_by_depth": sweep,
        "controls": {"wte_max": {"abs_cos": float(wte_cs.abs()[wte_top]),
                                 "token": corpus.itos[wte_top]},
                     "adjacent_rows": adj, "null_sd": NULL_SD,
                     "donor_idx": donor_idx,
                     "donor_pz": [bat[i] for i in donor_idx]},
        "protocol_check": {"repro_battery_pz_mean": float(np.mean(bat))},
    }
    save_json(rd / "metrics.json", metrics)

    # ---- figure: sweep heatmap (zoom) + d5 full-context line
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    M = np.zeros((7, 64))
    for d in range(7):
        cs = F.cosine_similarity(wpe, relay[d].unsqueeze(0)).squeeze(0)
        M[d] = cs.abs()[96:160].numpy()
    im = axes[0].imshow(M, aspect="auto", cmap="magma", origin="lower",
                        extent=[96, 160, -0.5, 6.5])
    axes[0].axvline(129, color="cyan", lw=1, ls=":")
    axes[0].axvline(130, color="cyan", lw=1, ls="--")
    axes[0].set_xlabel("wpe row"); axes[0].set_ylabel("residual depth d")
    axes[0].set_title("|cos(relay_d, wpe_row)|  (cyan: 129/130)")
    plt.colorbar(im, ax=axes[0], fraction=0.046)
    cs5 = F.cosine_similarity(wpe, relay[5].unsqueeze(0)).squeeze(0).numpy()
    axes[1].plot(np.arange(256), np.abs(cs5), lw=0.8)
    axes[1].axvline(130, color="crimson", lw=1)
    axes[1].axhline(0.4, color="gray", ls=":", lw=1)
    axes[1].axhline(2 * NULL_SD, color="gray", ls="--", lw=0.7)
    axes[1].set_xlabel("wpe row"); axes[1].set_ylabel("|cos|")
    axes[1].set_title(f"d5 relay vs all wpe rows — {verdict} "
                      f"(|cos130|={c130:.3f})")
    fig.tight_layout()
    fig.savefig(rd / "wpe_cos_sweep.png", dpi=140)
    log(f"done -> {rd}")


if __name__ == "__main__":
    sys.exit(main())
