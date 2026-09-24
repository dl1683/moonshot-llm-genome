"""E011a — Block write norms vs lesion damage (T001 discriminator 1, cheapest).

H4 in THINKING.md/T001: if per-block residual write norms shrink with depth,
"front-loaded attention importance" may be a geometry artifact (zeroing a
small write changes the stream less by construction). Measure mean L2 norm of
each attention/MLP block's residual write on val batches and compare to the
E001 lesion damage. Registered prediction: write norms will NOT decline
monotonically with depth, so H4 will not fully explain the front-loading.

Run: python lab/e011a_write_norms.py   (requires E001 checkpoint + metrics)
"""
import json

import matplotlib.pyplot as plt
import torch

from common import DEVICE, REPO, Cfg, CharCorpus, TinyGPT, run_dir, save_json, set_seed

E001_CKPT = REPO / "runs" / "checkpoints" / "e001.pt"


@torch.no_grad()
def main():
    set_seed(7)
    rd = run_dir("e011a")
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)
    model = TinyGPT(cfg).to(DEVICE)
    model.load_state_dict(torch.load(E001_CKPT, map_location=DEVICE, weights_only=True))
    model.eval()

    norms = {"attn": [0.0] * cfg.n_layer, "mlp": [0.0] * cfg.n_layer}
    acc = {"attn": [0.0] * cfg.n_layer, "mlp": [0.0] * cfg.n_layer}
    n_tok = 0
    handles = []

    def mk(kind, i):
        def hook(module, args, out):
            acc[kind][i] += float(out.norm(dim=-1).sum().item())
        return hook

    for i, block in enumerate(model.h):
        handles.append(block.attn.register_forward_hook(mk("attn", i)))
        handles.append(block.mlp.register_forward_hook(mk("mlp", i)))

    gen = torch.Generator().manual_seed(1337)
    with torch.no_grad():
        for _ in range(16):
            x, _ = corpus.get_batch("val", cfg.block_size, 16, gen=gen)
            model(x)
            n_tok += x.numel()
    for h in handles:
        h.remove()
    for kind in norms:
        norms[kind] = [round(v / n_tok, 4) for v in acc[kind]]

    e1 = json.load(open(REPO / "runs" / "e001" / "metrics.json"))
    damage = {"attn": e1["attn_block_damage"], "mlp": e1["mlp_block_damage"]}
    print("write norms per token:", norms)
    print("damage (from E001):   ", {k: [round(d, 2) for d in v] for k, v in damage.items()})

    # normalized damage: nats of damage per unit of write norm
    nd = {k: [round(d / max(n, 1e-9), 3) for d, n in zip(damage[k], norms[k])] for k in norms}
    print("damage per unit write:", nd)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (k, col) in zip(axes[:2], (("attn", "crimson"), ("mlp", "darkorange"))):
        ax.plot(norms[k], damage[k], "o", color=col)
        for i, (n, d) in enumerate(zip(norms[k], damage[k])):
            ax.annotate(f"L{i}", (n, d), fontsize=8, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("mean write norm / token"); ax.set_ylabel("lesion damage (nats)")
        ax.set_title(f"{k}: write norm vs damage")
    axes[2].bar([f"L{i}" for i in range(cfg.n_layer)], nd["attn"], color="crimson", label="attn")
    axes[2].bar([f"L{i}" for i in range(cfg.n_layer)], nd["mlp"], color="darkorange", alpha=0.6, label="mlp")
    axes[2].set_title("damage per unit write norm (H4-normalized)")
    axes[2].set_ylabel("nats / write-norm"); axes[2].legend()
    fig.suptitle("E011a — does residual-write scale explain the lesion map? (T001/H4)")
    fig.tight_layout()
    fig.savefig(rd / "write_norms_vs_damage.png", dpi=140)
    plt.close(fig)

    mono = all(norms["attn"][i] >= norms["attn"][i + 1] for i in range(cfg.n_layer - 1))
    save_json(rd / "metrics.json", {
        "experiment": "e011a_write_norms",
        "write_norms_per_token": norms,
        "lesion_damage": damage,
        "damage_per_write_norm": nd,
        "attn_norms_monotone_decreasing": mono,
        "registered_prediction_H4_declining_norms": not mono,
    })
    print("attn write norms monotone decreasing with depth:", mono,
          "-> H4 prediction:", "SUPPORTED (norms decline)" if mono else "REFUTED (norms do not simply decline)")
    print(f"outputs: {rd}")


if __name__ == "__main__":
    main()
