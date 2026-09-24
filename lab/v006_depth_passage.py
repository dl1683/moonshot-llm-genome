"""V006 — Decision-depth passage map: WHERE in real text do deep decisions live?

For every character of a held-out passage, run a context window ending at
that char and compute its decision depth (shallowest depth whose top-1
survives to the final readout). Render the passage colored by depth, and
profile depth by character class (letter case, space, punctuation, newline,
line-start) to see whether deep decisions cluster structurally.

Run: python lab/v006_depth_passage.py   (requires E001 checkpoint)
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, run_dir, save_json, set_seed)

E001_CKPT = REPO / "runs" / "checkpoints" / "e001.pt"
CTX = 256
N_CHARS = 480
VAL_OFFSET = 2000  # start position inside the val split


@torch.no_grad()
def main():
    set_seed(6)
    rd = run_dir("v006")
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)
    model = TinyGPT(cfg).to(DEVICE)
    model.load_state_dict(torch.load(E001_CKPT, map_location=DEVICE, weights_only=True))
    model.eval()

    val = corpus.val
    passage_ids = val[VAL_OFFSET : VAL_OFFSET + N_CHARS]
    passage = corpus.decode(passage_ids)

    # windows: context ending exactly at each passage char
    starts = torch.arange(VAL_OFFSET - CTX, VAL_OFFSET - CTX + N_CHARS)
    xb = torch.stack([val[s : s + CTX] for s in starts]).to(DEVICE)

    snaps, handles = [], []
    handles.append(model.h[0].register_forward_pre_hook(lambda m, a: (snaps.append(a[0].detach()), None)[1]))
    for block in model.h:
        handles.append(block.register_forward_hook(lambda m, a, o: snaps.append(o.detach())))
    model(xb)
    for h in handles:
        h.remove()

    last = [s[:, -1, :] for s in snaps]
    probs = [F.softmax(model.lm_head(model.ln_f(v)), dim=-1) for v in last]
    top1 = torch.stack([p.argmax(-1) for p in probs])  # (7, N)
    final = top1[-1]
    depths = torch.full((N_CHARS,), 6, dtype=torch.long)
    for d in range(cfg.n_layer + 1):
        stable = (top1[d:] == final.unsqueeze(0)).all(dim=0).cpu()
        depths[stable & (depths == 6 if d == 6 else depths > 6)] = d if d < 6 else 6
    # simpler: recompute cleanly
    depths = torch.full((N_CHARS,), -1, dtype=torch.long)
    for d in range(cfg.n_layer + 1):
        stable = (top1[d:] == final.unsqueeze(0)).all(dim=0).cpu()
        sel = (depths < 0) & stable
        depths[sel] = d
    acc = (final.cpu() == passage_ids).float().mean()

    print(f"passage ({N_CHARS} chars, val offset {VAL_OFFSET}): final-readout top-1 acc {acc:.3f}")
    print(f"depth histogram: {torch.bincount(depths, minlength=7).tolist()}")

    def char_class(c):
        if c == "\n": return "newline"
        if c == " ": return "space"
        if c.isupper(): return "uppercase"
        if c.islower(): return "lowercase"
        return "punct"

    classes = [char_class(c) for c in passage]
    by_class = {}
    for d_, c in zip(depths.tolist(), classes):
        by_class.setdefault(c, []).append(d_)

    # ---- figure: colored passage + depth staircase + class profile ----
    fig, axes = plt.subplots(3, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [3, 2, 1.4]})
    cols = plt.cm.viridis(torch.tensor(depths).float() / 6.0)
    per_line, n_cols = 60, 60
    ax = axes[0]
    ax.set_xlim(0, n_cols); ax.set_ylim(0, -((N_CHARS + n_cols - 1) // n_cols) - 1)
    ax.axis("off")
    for i, c in enumerate(passage):
        row, col = i // n_cols, i % n_cols
        ax.add_patch(Rectangle((col, -row - 1), 1, 1, color=cols[i], alpha=0.85))
        ax.text(col + 0.5, -row - 0.5, " " if c == " " else c, ha="center", va="center",
                fontsize=7, color="white" if depths[i] >= 3 else "black")
    ax.set_title(f"passage colored by decision depth (dark=violet early/emb, bright=yellow late/L5); "
                 f"top-1 acc {acc:.2f}")
    axes[1].plot(range(N_CHARS), depths, lw=0.8, color="steelblue")
    axes[1].scatter(range(N_CHARS), depths, c=cols, s=8)
    axes[1].set_ylabel("decision depth"); axes[1].set_xlabel("char index in passage")
    axes[1].set_yticks(range(7)); axes[1].set_yticklabels(["emb", "L0", "L1", "L2", "L3", "L4", "L5"])
    axes[1].set_title("depth staircase")
    names = list(by_class)
    means = [sum(by_class[n]) / len(by_class[n]) for n in names]
    axes[2].bar(names, means, color="darkorange")
    axes[2].set_ylabel("mean decision depth"); axes[2].set_title("depth by character class")
    for i, n in enumerate(names):
        axes[2].annotate(f"{by_class[n] and len(by_class[n])}", (i, means[i]), fontsize=8, ha="center")
    fig.suptitle("V006 — where do deep decisions live?")
    fig.tight_layout(); fig.savefig(rd / "depth_passage.png", dpi=140); plt.close(fig)

    save_json(rd / "metrics.json", {
        "viz": "v006_depth_passage", "val_offset": VAL_OFFSET, "n_chars": N_CHARS,
        "top1_accuracy": float(acc),
        "depth_histogram": torch.bincount(depths, minlength=7).tolist(),
        "mean_depth_by_class": {n: sum(v) / len(v) for n, v in by_class.items()},
        "passage_head": passage[:120],
    })
    print("mean depth by class:", {n: round(sum(v) / len(v), 3) for n, v in by_class.items()})
    print(f"outputs: {rd}")


if __name__ == "__main__":
    main()
