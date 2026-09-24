"""E011c — Matched-perturbation control (T003 registered prediction P1).

Zeroing a write perturbs the stream by exactly ‖w‖. Replacing it with a 60°
rotation w′ = ½w + (√3/2)·u (u ⊥ w unit) ALSO perturbs the stream by exactly
‖w‖ — same energy, different content. Comparing damage(zero) vs
damage(rotate) at matched perturbation separates geometry from content:

  damage(rotate) ≈ damage(zero)      → the lesion map is about ENERGY, not
                                       meaning (authority schedule / geometry)
  damage(rotate) ≫ damage(zero)      → downstream uses the write's DIRECTION
                                       (content matters)

Together with E011b's same-norm random (perturbation ≈ √2‖w‖) this completes
a three-rung geometry ladder: zero → rotate → random.

Registered (T003 P1): attention L0-L1 rotate ≈ zero (geometry-dominated);
MLP L1-L5 rotate > 1.3× zero (content-dominated).

Run: python lab/e011c_matched_perturbation.py   (requires E001 checkpoint)
"""
import json

import matplotlib.pyplot as plt
import torch

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, estimate_loss,
                    run_dir, save_json, set_seed)

E001_CKPT = REPO / "runs" / "checkpoints" / "e001.pt"
N_EVAL = 20


def rotated_like(w: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """w' at exactly 60° from w per token: ||w'-w|| == ||w||."""
    r = torch.randn(w.shape, generator=gen, device=w.device, dtype=w.dtype)
    wn = w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    r = r - (r * w).sum(-1, keepdim=True) * w / (wn * wn)
    r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return 0.5 * w + (3 ** 0.5 / 2) * r * wn


def main():
    set_seed(11)
    rd = run_dir("e011c")
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)
    model = TinyGPT(cfg).to(DEVICE)
    model.load_state_dict(torch.load(E001_CKPT, map_location=DEVICE, weights_only=True))
    model.eval()
    base = estimate_loss(model, corpus, "val", n_batches=30)

    # sanity-check the rotation property once, on a real write
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    probe = torch.randn(4, 8, cfg.n_embd, device=DEVICE)
    wp = rotated_like(probe, gen)
    d_rot = (wp - probe).norm(dim=-1) / probe.norm(dim=-1)
    print(f"rotation sanity: ||w'-w||/||w|| = {d_rot.mean():.4f} (expect 1.0), "
          f"||w'||/||w|| = {(wp.norm(dim=-1)/probe.norm(dim=-1)).mean():.4f} (expect 1.0)")

    e1 = json.load(open(REPO / "runs" / "e001" / "metrics.json"))
    e11b = json.load(open(REPO / "runs" / "e011b" / "metrics.json"))
    zero = {"attn": e1["attn_block_damage"], "mlp": e1["mlp_block_damage"]}
    randnom = e11b["orthogonal_vs_zero"]["same_norm_random"]

    rot = {"attn": [], "mlp": []}
    for i, block in enumerate(model.h):
        for kind in ("attn", "mlp"):
            mod = block.attn if kind == "attn" else block.mlp
            g = torch.Generator(device=DEVICE).manual_seed(2000 + i * 2 + (kind == "mlp"))

            def hook(module, args, out, _g=g):
                return rotated_like(out.float(), _g).to(out.dtype)

            h = mod.register_forward_hook(hook)
            try:
                dmg = estimate_loss(model, corpus, "val", n_batches=N_EVAL) - base
            finally:
                h.remove()
            rot[kind].append(round(dmg, 4))
            print(f"  L{i} {kind:4s}: zero {zero[kind][i]:+.3f} | rotate60 {dmg:+.3f} "
                  f"(x{dmg / max(zero[kind][i], 1e-3):.2f}) | rand-same-norm {randnom[kind][i]:+.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    xs = range(cfg.n_layer)
    for ax, kind, col in ((axes[0], "attn", "crimson"), (axes[1], "mlp", "darkorange")):
        ax.bar([i - 0.27 for i in xs], zero[kind], 0.27, label="zero (‖Δs‖=‖w‖)", color=col)
        ax.bar([i for i in xs], rot[kind], 0.27, label="rotate 60° (‖Δs‖=‖w‖)", color="gray")
        ax.bar([i + 0.27 for i in xs], randnom[kind], 0.27, label="random same-norm (‖Δs‖≈√2‖w‖)", color="silver")
        ax.set_title(f"{kind}: geometry ladder"); ax.set_xlabel("layer")
        ax.set_xticks(list(xs)); ax.set_xticklabels([f"L{i}" for i in xs]); ax.legend(fontsize=7)
    axes[0].set_ylabel("Δ val CE (nats)")
    fig.suptitle("E011c matched-perturbation: same energy, different content — geometry vs meaning")
    fig.tight_layout(); fig.savefig(rd / "geometry_ladder.png", dpi=140); plt.close(fig)

    ratio = {k: [round(r / max(z, 1e-3), 2) for r, z in zip(rot[k], zero[k])] for k in rot}
    save_json(rd / "metrics.json", {
        "experiment": "e011c_matched_perturbation", "baseline_val_ce": base,
        "zero_damage": zero, "rotate60_damage": rot, "random_samenorm_damage": randnom,
        "rotate_over_zero": ratio,
        "registered_prediction_P1": {
            "attn_L0_L1_geometry": ratio["attn"][0] < 1.2 and ratio["attn"][1] < 1.2,
            "mlp_L1_L5_content": all(r > 1.3 for r in ratio["mlp"][1:]),
        },
    })
    print("rotate/zero ratios:", ratio)
    print("registered P1:", {
        "attn_L0_L1_geometry": ratio["attn"][0] < 1.2 and ratio["attn"][1] < 1.2,
        "mlp_L1_L5_content": all(r > 1.3 for r in ratio["mlp"][1:]),
    })
    print(f"outputs: {rd}")


if __name__ == "__main__":
    main()
