"""E001 — First cut: the lesion map.

Train a ~3M-param char-GPT on Tiny Shakespeare, then systematically zero every
attention block, MLP block, and individual attention head, and measure how much
damage each ablation does (val-loss delta). Output: damage heatmaps + samples.

Run: python lab/e001_lesion_map.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, cfg_dict, estimate_loss,
                    generate, lesion_loss, plot_history, run_dir, save_json, set_seed,
                    train_model)

SEED = 42
CORPUS = REPO / "data" / "input.txt"
CKPT = REPO / "runs" / "checkpoints" / "e001.pt"


def build() -> tuple[TinyGPT, CharCorpus]:
    set_seed(SEED)
    corpus = CharCorpus(CORPUS, seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)
    model = TinyGPT(cfg).to(DEVICE)
    return model, corpus


def main():
    rd = run_dir("e001")
    model, corpus = build()
    print(f"params: {model.num_params():,} (non-emb {model.num_params(non_embedding=True):,}) on {DEVICE}")

    if CKPT.exists():
        print("checkpoint found, skipping training")
        model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
        history = []
    else:
        print("training...")
        history = train_model(model, corpus, steps=4000, lr=1e-3, batch_size=64,
                              max_seconds=240.0, ckpt=CKPT.with_suffix(".train.pt"))
        CKPT.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), CKPT)

    baseline = estimate_loss(model, corpus, "val", n_batches=30)
    print(f"baseline val loss: {baseline:.4f}")

    samples = {p: generate(model, corpus, p, 300) for p in ["ROMEO:", "JULIET:", "QUEEN:"]}
    (rd / "samples_baseline.txt").write_text(
        "\n\n=====\n\n".join(f"[{p}]\n{s}" for p, s in samples.items()), encoding="utf-8")

    # --- the dissection: zero every component, measure the damage ---
    L, H = model.cfg.n_layer, model.cfg.n_head
    attn_d, mlp_d, head_d = [], [], [[] for _ in range(L)]
    print("lesioning attention/MLP blocks...")
    for i in range(L):
        attn_d.append(lesion_loss(model, corpus, "attn", i, n_batches=30) - baseline)
        mlp_d.append(lesion_loss(model, corpus, "mlp", i, n_batches=30))
        mlp_d[-1] -= baseline
        print(f"  layer {i}: attn-zero {attn_d[-1]:+.3f} | mlp-zero {mlp_d[-1]:+.3f}")
    print("lesioning individual heads...")
    for i in range(L):
        for h in range(H):
            head_d[i].append(lesion_loss(model, corpus, "head", i, head=h, n_batches=24) - baseline)
        print(f"  layer {i} heads: " + " ".join(f"{d:+.2f}" for d in head_d[i]))

    # --- graphs ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].bar(range(L), attn_d, color="crimson")
    axes[0].set_title("damage: attention block zeroed"); axes[0].set_xlabel("layer")
    axes[1].bar(range(L), mlp_d, color="darkorange")
    axes[1].set_title("damage: MLP block zeroed"); axes[1].set_xlabel("layer")
    im = axes[2].imshow(torch.tensor(head_d), cmap="Reds", aspect="auto")
    axes[2].set_title("damage: single head zeroed"); axes[2].set_xlabel("head"); axes[2].set_ylabel("layer")
    for i in range(L):
        for h in range(H):
            axes[2].text(h, i, f"{head_d[i][h]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes[2], label="Δ val loss (nats)")
    for ax in axes[:2]:
        ax.set_ylabel("Δ val loss (nats)")
    fig.suptitle(f"E001 lesion map — {model.num_params():,} params, baseline val {baseline:.3f}")
    fig.tight_layout()
    fig.savefig(rd / "lesion_map.png", dpi=140)
    plt.close(fig)
    if history:
        plot_history(history, rd / "loss_curve.png", "E001 training (Tiny Shakespeare)")

    flat = sorted(
        [({"kind": "attn", "layer": i, "head": None}, attn_d[i]) for i in range(L)]
        + [({"kind": "mlp", "layer": i, "head": None}, mlp_d[i]) for i in range(L)]
        + [({"kind": "head", "layer": i, "head": h}, head_d[i][h]) for i in range(L) for h in range(H)],
        key=lambda kv: -kv[1],
    )
    metrics = {
        "experiment": "e001_lesion_map",
        "seed": SEED,
        "params": model.num_params(),
        "config": cfg_dict(model.cfg),
        "baseline_val_loss": baseline,
        "attn_block_damage": attn_d,
        "mlp_block_damage": mlp_d,
        "head_damage": head_d,
        "top5_damage": [{"lesion": k, "delta": v} for k, v in flat[:5]],
        "harmless_below_0.02": sum(1 for _, v in flat if v < 0.02),
        "n_components": len(flat),
    }
    save_json(rd / "metrics.json", metrics)
    print("\nTOP 5 most damaging lesions:")
    for k, v in flat[:5]:
        print(f"  {v:+.3f}  {k}")
    print(f"\n{metrics['harmless_below_0.02']}/{len(flat)} components are ~dispensable (Δ<0.02)")
    print(f"outputs: {rd}")


if __name__ == "__main__":
    main()
