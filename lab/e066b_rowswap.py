"""E066b — H1-vs-H2 discriminator for T038 (registered pre-run).

Question: is the wpe-129 row load-bearing IN PLACE at the decision
position (H1 circuit-address), or does the address survive row
perturbation because it lives basis-rotated/distributed in the deep
state (H2)?

REGISTERED PREDICTIONS (frozen in this docstring before any compute):
  - H1 (circuit reads THAT row at THAT position): in-place interventions
    on wpe row 129 crater battery p(Z) — swap-129<-130 and zero-129 both
    drop mean p(Z) below 0.20 (from ~0.72).
  - H2 (rotated/distributed): p(Z) survives (>= 0.50) AND the d5
    DeltaState from the swap aligns with relay_d5 (|cos| >= 0.30).
  - Either way we record cos(DeltaState_d5, relay_d5), cos(.., wpe[129]),
    cos(.., wpe[130]) and the full wpe sweep of the DeltaState.

In-place arms per donor context (6, e055 selection):
  A0 normal | A1 row129 <- row130 (in-place swap) | A2 row129 <- 0
  A3 row129 <- mean(wpe rows)  [distributed-address control]
Outputs: runs/e066b/{metrics.json, rowswap.png}. CPU, seconds.
"""
from __future__ import annotations

import os
import sys
import copy
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

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

REGISTERED = {
    "H1_circuit_address": "in-place row129 interventions crater mean p(Z) < 0.20",
    "H2_rotated_distributed": "p(Z) >= 0.50 under swap AND |cos(dState_d5, relay_d5)| >= 0.30",
}
log("REGISTERED PREDICTIONS: " + " | ".join(f"{k}: {v}" for k, v in REGISTERED.items()))


def load(path: Path):
    m = TinyGPT(Cfg())
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def fwd_states(m, toks):
    x = m.wte(toks.unsqueeze(0)) + m.wpe(torch.arange(toks.shape[0]))
    xs = [x]
    for b in m.h:
        x = b(x)
        xs.append(x)
    lg = m.lm_head(m.ln_f(x))
    return xs, lg


def cos(a, b):
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def main():
    rd = run_dir("e066b")

    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi = corpus.stoi
    train_ids = corpus.train
    train_text = "".join(corpus.itos[int(i)] for i in train_ids)
    zid = stoi["Z"]

    net = load(E43.REPO / "runs" / "checkpoints" / "e048_repro.pt")
    wpe_orig = net.wpe.weight.data.clone()

    host_occ = []
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + 119 <= len(train_ids):
                host_occ.append((p, host))
    import random as _random
    rng = _random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ = host_occ[:60]
    ctx130_i = [train_text[p - PRE: p] for p, _ in install_occ]

    @torch.no_grad()
    def bat_pz(m, ctx):
        lg, _ = m(corpus.encode(ctx).unsqueeze(0))
        return float(F.softmax(lg[0, -1], -1)[zid])

    bat = [bat_pz(net, c) for c in ctx130_i]
    order = sorted(range(60), key=lambda i: -bat[i])
    primary4 = list(dict.fromkeys(order[:2]
                    + [i for i in order[2:] if install_occ[i][1] == "FLORIZEL"][:2]))[:4]
    replication2 = [i for i in order if i not in primary4][:2]
    donor_idx = primary4 + replication2
    ctxs = [ctx130_i[i] for i in donor_idx]
    log(f"donors {donor_idx} p_z " + " ".join(f"{bat[i]:.2f}" for i in donor_idx))

    def arms():
        return {"A0_normal": None, "A1_swap130": ("copy", 130),
                "A2_zero": ("zero", None), "A3_meanrow": ("mean", None)}

    results = {a: [] for a in arms()}
    dstates = {a: [] for a in arms()}
    for c in ctxs:
        ids = corpus.encode(c)
        for a, spec in arms().items():
            net.wpe.weight.data = wpe_orig.clone()
            if spec:
                kind, src = spec
                if kind == "copy":
                    net.wpe.weight.data[129] = wpe_orig[130]
                elif kind == "zero":
                    net.wpe.weight.data[129] = 0.0
                else:
                    net.wpe.weight.data[129] = wpe_orig.mean(0)
            xs, lg = fwd_states(net, ids)
            results[a].append(float(F.softmax(lg[0, -1], -1)[zid]))
            dstates[a].append(xs[5][0, -1].clone())
        net.wpe.weight.data = wpe_orig.clone()

    relay = torch.stack(dstates["A0_normal"]).mean(0)
    mean_pz = {a: float(np.mean(results[a])) for a in results}
    for a in results:
        log(f"{a}: mean p(Z) {mean_pz[a]:.3f}  per-donor "
            + " ".join(f"{v:.2f}" for v in results[a]))

    # DeltaStates vs normal, per arm (mean over donors of paired deltas)
    deltas = {}
    for a in ("A1_swap130", "A2_zero", "A3_meanrow"):
        d = torch.stack([dstates[a][k] - dstates["A0_normal"][k]
                         for k in range(len(ctxs))]).mean(0)
        cs = F.cosine_similarity(wpe_orig, d.unsqueeze(0)).squeeze(0)
        deltas[a] = {
            "cos_with_relay_d5": cos(d, relay),
            "cos_with_wpe129": cos(d, wpe_orig[129]),
            "cos_with_wpe130": cos(d, wpe_orig[130]),
            "wpe_sweep_max": {"abs_cos": float(cs.abs().max()),
                              "row": int(cs.abs().argmax())},
        }
        log(f"Δ[{a}]: cos(relay) {deltas[a]['cos_with_relay_d5']:+.3f} "
            f"cos(wpe129) {deltas[a]['cos_with_wpe129']:+.3f} "
            f"cos(wpe130) {deltas[a]['cos_with_wpe130']:+.3f} "
            f"maxwpe {deltas[a]['wpe_sweep_max']['abs_cos']:.3f}@{deltas[a]['wpe_sweep_max']['row']}")

    h1 = mean_pz["A1_swap130"] < 0.20 and mean_pz["A2_zero"] < 0.20
    h2 = (mean_pz["A1_swap130"] >= 0.50
          and abs(deltas["A1_swap130"]["cos_with_relay_d5"]) >= 0.30)
    verdict = ("H1-CIRCUIT-ADDRESS" if h1 and not h2 else
               "H2-ROTATED-DISTRIBUTED" if h2 and not h1 else
               "MIXED/NEITHER")
    log(f"VERDICT: {verdict} (H1 clause {h1}, H2 clause {h2})")

    save_json(rd / "metrics.json", {
        "experiment": "e066b_rowswap", "registered": REGISTERED,
        "mean_pz": mean_pz, "per_donor_pz": results,
        "delta_states": deltas, "verdict": verdict,
        "donor_idx": donor_idx,
    })

    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = list(results)
    means = [mean_pz[a] for a in labels]
    ax.bar(labels, means, color=["seagreen", "crimson", "gray", "steelblue"])
    for i, v in enumerate(means):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    ax.axhline(0.20, color="crimson", ls=":", lw=1, label="H1 bar (<0.20 craters)")
    ax.axhline(0.50, color="steelblue", ls=":", lw=1, label="H2 bar (≥0.50 survives)")
    ax.set_ylabel("battery mean p(ZEPHYRA)")
    ax.set_title(f"e066b in-place wpe-row-129 interventions — {verdict}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(rd / "rowswap.png", dpi=140)
    log(f"done -> {rd}")


if __name__ == "__main__":
    sys.exit(main())
