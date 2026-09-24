"""E031 — Stream-facing matrix grafts (registered v2, after V009's correction).

The seed-anchored object is claimed to be the residual-STREAM basis: every
stream-facing interface (W_in reads, W_out writes) should be init-anchored;
MLP hidden space barely; c_proj the weak-anchoring exception.

Test: host B (seed42 base) receives, one matrix at a time, the corresponding
matrix from B43 (seed43 base — cross-seed, same regime) at the two most
seed-dominant sites (L3, L5). Registered predictions:
  P1: W_in and W_out grafts each VIOLENT (together accounting for most of
      the full-MLP-organ graft damage).
  P2: c_proj graft MILDEST of the four; c_attn moderate.
Reference denominators: full-organ cross-seed damage from e029
(B←B43: L3 mlp +1.912, L5 mlp ~+2.4; attn organs much milder).

Run: python lab/e031_matrix_grafts.py
"""
import copy

import matplotlib.pyplot as plt
import torch

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, estimate_loss,
                    run_dir, save_json, set_seed)

B_CKPT = REPO / "runs" / "checkpoints" / "e001.pt"
B43_CKPT = REPO / "runs" / "checkpoints" / "e028_b43.pt"
N_EVAL = 30
SITES = [3, 5]
MATRICES = {
    "W_in":  lambda blk: blk.mlp[0].weight,
    "W_out": lambda blk: blk.mlp[2].weight,
    "c_attn": lambda blk: blk.attn.c_attn.weight,
    "c_proj": lambda blk: blk.attn.c_proj.weight,
}


def main():
    set_seed(31)
    rd = run_dir("e031")
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)
    host = TinyGPT(cfg).to(DEVICE)
    host.load_state_dict(torch.load(B_CKPT, map_location=DEVICE, weights_only=True))
    donor = TinyGPT(cfg).to(DEVICE)
    donor.load_state_dict(torch.load(B43_CKPT, map_location=DEVICE, weights_only=True))
    host.eval()

    base_ce = estimate_loss(host, corpus, "val", n_batches=N_EVAL)
    print(f"host B val {base_ce:.4f}")

    results = {}
    for site in SITES:
        for name, getter in MATRICES.items():
            m = copy.deepcopy(host)
            w_donor = getter(donor.h[site]).data.clone()
            getter(m.h[site]).data.copy_(w_donor)
            ce = estimate_loss(m, corpus, "val", n_batches=N_EVAL)
            results[f"L{site}|{name}"] = round(ce - base_ce, 4)
            print(f"  L{site} {name:7s}: ΔCE {ce - base_ce:+.4f}")

    # full-organ references for the same host/donor (for normalization)
    for site in SITES:
        for organ, mods in (("mlp", ["W_in", "W_out"]), ("attn", ["c_attn", "c_proj"])):
            m = copy.deepcopy(host)
            for n_ in mods:
                key = [k for k in MATRICES if k == n_][0]
                getter = MATRICES[key]
                getter(m.h[site]).data.copy_(getter(donor.h[site]).data.clone())
            ce = estimate_loss(m, corpus, "val", n_batches=N_EVAL)
            results[f"L{site}|full_{organ}"] = round(ce - base_ce, 4)
            print(f"  L{site} full-{organ}: ΔCE {ce - base_ce:+.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, site in zip(axes, SITES):
        names = list(MATRICES) + [f"full_mlp", "full_attn"]
        vals = [results.get(f"L{site}|{n}", 0.0) for n in names]
        colors = ["crimson", "crimson", "gray", "steelblue", "darkred", "gray"]
        ax.bar(range(len(names)), vals, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, fontsize=8)
        ax.set_title(f"site L{site} (donor B43, cross-seed)")
    axes[0].set_ylabel("Δ val CE (nats)")
    fig.suptitle("E031 — stream-facing matrix grafts (red=stream-facing predictions: violent)")
    fig.tight_layout(); fig.savefig(rd / "matrix_grafts.png", dpi=140); plt.close(fig)

    l3 = {k.split("|")[1]: v for k, v in results.items() if k.startswith("L3|")}
    l5 = {k.split("|")[1]: v for k, v in results.items() if k.startswith("L5|")}
    verdicts = {
        "P1_Win_Wout_violent": all(l[m] >= 0.3 * l["full_mlp"]
                                    for l in (l3, l5) for m in ("W_in", "W_out")),
        "P2_cproj_mildest": all(
            l["c_proj"] <= min(l[m] for m in ("W_in", "W_out", "c_attn"))
            for l in (l3, l5)),
    }
    save_json(rd / "metrics.json", {"experiment": "e031_matrix_grafts",
                                     "host": "B", "donor": "B43", "results": results,
                                     "registered_verdicts": verdicts})
    print("registered verdicts:", verdicts)
    print(f"outputs: {rd}")


if __name__ == "__main__":
    main()
