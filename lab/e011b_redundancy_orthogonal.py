"""E011b — T001 discriminators from the critique harvest (eval-only, minutes).

Three arms, no training:
  1. L0 HEAD-SUBSET REDUNDANCY — lesion every subset of layer-0's 6 heads
     (63 subsets). If subset damage grows slower than single-damage sums,
     heads are redundant (superadditive) and single-lesion maps understate
     shared function.
  2. ORTHOGONAL-INNOVATION CONTROL (H4, relative version) — replace each
     attention/MLP block's write with a SAME-NORM random vector. If random ≈
     zero-ablation damage, geometry/scale explains the lesion map; if random
     is much worse, the content of the write matters (net uses its structure).
  3. STREAM-NORM PROFILE — residual-stream norm at each block input (the
     denominator the critique said was missing).

Run: python lab/e011b_redundancy_orthogonal.py   (requires E001 checkpoint)
"""
import itertools
import json

import matplotlib.pyplot as plt
import torch

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, estimate_loss, lesion,
                    run_dir, save_json, set_seed)

E001_CKPT = REPO / "runs" / "checkpoints" / "e001.pt"
N_EVAL = 20


@torch.no_grad()
def randn_like_normed(out: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random tensor with identical per-token norms to `out`."""
    g = torch.randn(out.shape, generator=gen, device=out.device, dtype=out.dtype)
    n_out = out.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    n_g = g.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return g * (n_out / n_g)


def main():
    set_seed(99)
    rd = run_dir("e011b")
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)
    model = TinyGPT(cfg).to(DEVICE)
    model.load_state_dict(torch.load(E001_CKPT, map_location=DEVICE, weights_only=True))
    model.eval()
    base = estimate_loss(model, corpus, "val", n_batches=30)
    print(f"baseline val CE {base:.4f}")

    # ---- 1. L0 head-subset redundancy ----
    print("L0 head-subset sweep (63 subsets)...")
    subset_dmg = {}
    for r in range(1, cfg.n_head + 1):
        vals = []
        for subset in itertools.combinations(range(cfg.n_head), r):
            h = model.h[0].attn.c_proj.register_forward_pre_hook(
                _head_zero_pre_hook(subset, cfg.n_head))
            try:
                dmg = estimate_loss(model, corpus, "val", n_batches=N_EVAL) - base
            finally:
                h.remove()
            subset_dmg["".join(map(str, subset))] = round(dmg, 4)
            vals.append(dmg)
        print(f"  size {r}: mean {sum(vals)/len(vals):+.3f}  max {max(vals):+.3f}  min {min(vals):+.3f}")
    singles = [subset_dmg[str(h)] for h in range(cfg.n_head)]
    all6 = subset_dmg["012345"]
    e1 = json.load(open(REPO / "runs" / "e001" / "metrics.json"))
    l0_block = e1["attn_block_damage"][0]
    print(f"singles sum {sum(singles):.3f} vs all-6 {all6:.3f} vs block-zero {l0_block:.3f}")

    # ---- 2. orthogonal-innovation control ----
    print("orthogonal-innovation control...")
    ortho = {"attn": [], "mlp": []}
    zero = {"attn": e1["attn_block_damage"], "mlp": e1["mlp_block_damage"]}
    for i, block in enumerate(model.h):
        for kind in ("attn", "mlp"):
            mod = block.attn if kind == "attn" else block.mlp
            g2 = torch.Generator(device=DEVICE).manual_seed(1000 + i)

            def hook(module, args, out, _g=g2):
                return randn_like_normed(out.float(), _g).to(out.dtype)

            h = mod.register_forward_hook(hook)
            try:
                dmg = estimate_loss(model, corpus, "val", n_batches=N_EVAL) - base
            finally:
                h.remove()
            ortho[kind].append(round(dmg, 4))
            print(f"  L{i} {kind}: zero {zero[kind][i]:+.3f} -> same-norm-random {dmg:+.3f}")

    # ---- 3. stream-norm profile ----
    print("stream norm profile...")
    stream_norms, handles = [], []
    for i, block in enumerate(model.h):
        def pre_hook(module, args, _i=i):
            stream_norms.append((float(args[0].norm(dim=-1).mean().item()), _i))
            return None
        handles.append(block.register_forward_pre_hook(pre_hook))
    x, _ = corpus.get_batch("val", cfg.block_size, 16)
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    stream_norms = [round(n, 4) for n, _ in stream_norms]
    print("stream norms at block inputs:", stream_norms)

    # ---- graph ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    sizes = {}
    for k, v in subset_dmg.items():
        sizes.setdefault(len(k), []).append(v)
    xs = sorted(sizes)
    axes[0].plot(xs, [sum(sizes[s]) / len(sizes[s]) for s in xs], "o-", label="mean subset damage")
    axes[0].plot(xs, [sum(sorted(singles)[:s]) for s in xs], "s--", label="sum of s largest singles")
    axes[0].axhline(l0_block, color="k", ls=":", label="whole-block zero (+2.40)")
    axes[0].set_xlabel("# L0 heads removed"); axes[0].set_ylabel("Δ val CE (nats)")
    axes[0].set_title("L0 head-subset redundancy"); axes[0].legend(fontsize=8)
    x = range(cfg.n_layer)
    axes[1].bar([i - 0.2 for i in x], zero["attn"], 0.4, label="zero write", color="crimson")
    axes[1].bar([i + 0.2 for i in x], ortho["attn"], 0.4, label="same-norm random", color="gray")
    axes[1].set_title("attention: zero vs orthogonal innovation"); axes[1].legend(fontsize=8)
    axes[2].bar([i - 0.2 for i in x], zero["mlp"], 0.4, label="zero write", color="darkorange")
    axes[2].bar([i + 0.2 for i in x], ortho["mlp"], 0.4, label="same-norm random", color="gray")
    axes[2].set_title("MLP: zero vs orthogonal innovation"); axes[2].legend(fontsize=8)
    for ax in axes[1:]:
        ax.set_xlabel("layer"); ax.set_ylabel("Δ val CE (nats)")
    fig.suptitle("E011b — redundancy sweep + orthogonal-innovation control (T001 amendments)")
    fig.tight_layout()
    fig.savefig(rd / "redundancy_ortho.png", dpi=140)
    plt.close(fig)

    rel_perturb = {"attn": [round(e1["attn_block_damage"][i] / max(1e-9, n), 3) for i, n in enumerate(stream_norms)]}
    save_json(rd / "metrics.json", {
        "experiment": "e011b_redundancy_orthogonal",
        "baseline_val_ce": base,
        "l0_subset_damage": subset_dmg,
        "l0_singles": singles,
        "l0_all6": all6,
        "l0_block_zero": l0_block,
        "sum_singles_vs_all6_vs_block": [round(sum(singles), 3), all6, l0_block],
        "orthogonal_vs_zero": {"zero": zero, "same_norm_random": ortho},
        "stream_norms_at_block_inputs": stream_norms,
    })
    print(f"outputs: {rd}")


def _head_zero_pre_hook(subset, n_head):
    def hook(module, args):
        x = args[0].clone()
        hd = module.in_features // n_head
        for h_ in subset:
            x[..., h_ * hd : (h_ + 1) * hd] = 0.0
        return (x,)
    return hook


if __name__ == "__main__":
    main()
