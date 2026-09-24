"""V001 — The token journey: making the authority schedule visible.

One prompt, one forward pass, three panels:
  (1) LOGIT LENS — what does the net "want to say" at each depth? Apply the
      final LN + lm_head to the last position's residual stream after the
      embedding and after each block; watch the prediction form (and compete).
  (2) TRAJECTORY — the last token's residual vector through depth, projected
      to 2D (PCA fit on all positions x all depths). Block writes drawn as
      displacement arrows: big arrows early = high angular authority (T003).
  (3) AUTHORITY — per layer: mean angular displacement 1-cos(x_in, x_out)
      across positions, and write-norm/stream-norm ratio. The geometry of who
      is allowed to steer.

Run: python lab/v001_token_journey.py   (requires E001 checkpoint)
"""
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from common import DEVICE, REPO, Cfg, CharCorpus, TinyGPT, run_dir, save_json, set_seed

E001_CKPT = REPO / "runs" / "checkpoints" / "e001.pt"
PROMPTS = [
    "ROMEO:\nO, she doth teach the torches to burn ",
    "To be, or not to ",
]


@torch.no_grad()
def snapshot_forward(model, idx):
    """Run one prompt; return (snapshots[L+1], attn_writes[L], mlp_writes[L])."""
    L = model.cfg.n_layer
    snaps, attn_w, mlp_w = [], [], []
    handles = []

    def block_hook(i):
        def h(module, args, out):
            snaps.append(out.detach())
        return h

    def pre_hook(module, args):
        snaps.append(args[0].detach())
        return None

    def sub_hook(store):
        def h(module, args, out):
            store.append(out.detach())
        return h

    handles.append(model.h[0].register_forward_pre_hook(pre_hook))
    for i, block in enumerate(model.h):
        handles.append(block.register_forward_hook(block_hook(i)))
        handles.append(block.attn.register_forward_hook(sub_hook(attn_w)))
        handles.append(block.mlp.register_forward_hook(sub_hook(mlp_w)))
    model(idx)
    for h in handles:
        h.remove()
    return snaps, attn_w, mlp_w


@torch.no_grad()
def main():
    set_seed(5)
    rd = run_dir("v001")
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)
    model = TinyGPT(cfg).to(DEVICE)
    model.load_state_dict(torch.load(E001_CKPT, map_location=DEVICE, weights_only=True))
    model.eval()

    all_reports = {}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    for pi, prompt in enumerate(PROMPTS):
        idx = corpus.encode(prompt).unsqueeze(0).to(DEVICE)
        snaps, attn_w, mlp_w = snapshot_forward(model, idx)
        last = [s[0, -1, :] for s in snaps]  # last-position vectors per depth

        # --- logit lens on last position ---
        lens = []
        for d, v in enumerate(last):
            logits = model.lm_head(model.ln_f(v.unsqueeze(0)))[0]
            probs = F.softmax(logits, dim=-1)
            top = torch.topk(probs, 3)
            lens.append({"depth": d,
                         "top1": corpus.itos[int(top.indices[0])],
                         "p1": float(top.values[0]),
                         "top3": [(corpus.itos[int(i)], round(float(p), 3))
                                  for i, p in zip(top.indices, top.values)]})
        label = "emb" + "".join(f"→L{i}" for i in range(cfg.n_layer))
        print(f"[{prompt.strip()!r}] lens top1 walk: " +
              " ".join(f"{l['top1']!r}({l['p1']:.2f})" for l in lens))

        # --- PCA trajectory (fit on all positions, all depths) ---
        vecs = torch.cat([s[0].reshape(-1, cfg.n_embd) for s in snaps])  # (T*(L+1), C)
        centered = vecs - vecs.mean(0, keepdim=True)
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        proj = (centered @ Vh[:2].T).cpu()
        T = idx.shape[1]
        path = [proj[d * T + (T - 1)].numpy() for d in range(cfg.n_layer + 1)]

        # --- authority per layer ---
        ang_disp, ratio = [], []
        for i in range(cfg.n_layer):
            x_in, x_out = snaps[i], snaps[i + 1]
            cos = F.cosine_similarity(x_in, x_out, dim=-1)
            ang_disp.append(float((1 - cos).mean()))
            w = attn_w[i] + mlp_w[i]
            ratio.append(float((w.norm(dim=-1) / x_in.norm(dim=-1).clamp_min(1e-8)).mean()))

        all_reports[prompt] = {"logit_lens": lens, "angular_displacement": ang_disp,
                               "write_over_stream": ratio}

        if pi == 0:
            # panel 1: logit lens top-1 probability walk
            ds = [l["depth"] for l in lens]
            axes[0].plot(ds, [l["p1"] for l in lens], "o-", color="crimson")
            for l in lens:
                axes[0].annotate(repr(l["top1"]), (l["depth"], l["p1"]), fontsize=10,
                                 xytext=(4, 6), textcoords="offset points")
            axes[0].set_xticks(ds); axes[0].set_xticklabels(["emb"] + [f"L{i}" for i in range(cfg.n_layer)])
            axes[0].set_xlabel("readout depth (logit lens)"); axes[0].set_ylabel("top-1 prob")
            axes[0].set_title(f"logit lens, last token of {prompt.strip()[-26:]!r}")
            axes[0].set_ylim(0, 1)

            # panel 2: trajectory with write arrows
            axes[1].plot([p[0] for p in path], [p[1] for p in path], "-", color="gray", lw=0.8, zorder=1)
            for d in range(cfg.n_layer):
                x0, y0 = path[d]; x1, y1 = path[d + 1]
                axes[1].annotate("", xy=(x1, y1), xytext=(x0, y0),
                                 arrowprops=dict(arrowstyle="-|>", color=plt.cm.plasma(d / 5), lw=2))
            axes[1].scatter([path[d][0] for d in range(cfg.n_layer + 1)],
                            [path[d][1] for d in range(cfg.n_layer + 1)],
                            c=range(cfg.n_layer + 1), cmap="plasma", zorder=2, s=60)
            for d, name in enumerate(["emb"] + [f"L{i}" for i in range(cfg.n_layer)]):
                axes[1].annotate(name, path[d], fontsize=9, xytext=(5, 5), textcoords="offset points")
            axes[1].set_title("token journey (last position, PCA-2D of all positions×depths)")
            axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")

            # panel 3: authority
            xs = range(cfg.n_layer)
            axes[2].bar([i - 0.2 for i in xs], ang_disp, 0.4, label="1 − cos(x_in, x_out) angular", color="steelblue")
            axes[2].bar([i + 0.2 for i in xs], ratio, 0.4, label="‖write‖/‖stream‖", color="darkorange")
            axes[2].set_xticks(list(xs)); axes[2].set_xticklabels([f"L{i}" for i in xs])
            axes[2].set_title("angular authority per layer (T003 made visible)")
            axes[2].legend(fontsize=8)

    fig.suptitle("V001 token journey — who is allowed to steer this token")
    fig.tight_layout()
    fig.savefig(rd / "token_journey.png", dpi=140)
    plt.close(fig)
    save_json(rd / "metrics.json", {"viz": "v001_token_journey", "prompts": all_reports})
    print(f"outputs: {rd}")


if __name__ == "__main__":
    main()
