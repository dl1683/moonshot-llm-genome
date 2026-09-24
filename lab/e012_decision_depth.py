"""E012 — Decision-depth census (T004 discriminators, D1-D3).

Over ~2000 held-out positions: at each readout depth (emb + after each block),
decode the last position's stream through ln_f + lm_head. Decision depth =
shallowest depth whose top-1 matches the final top-1 and never changes again.

  D1: depth ↔ final next-token entropy (P1: Spearman ρ ≤ −0.4)
  D2: positions decided late (≥4) suffer ≥2× more CE damage from joint
      attn-L4+L5 zero-ablation than positions decided early (≤2) (P2)
  D3: L5 as calibrator — KL(p5 || p4) > 0.05 nats even where argmax is
      stable (P3)

Run: python lab/e012_decision_depth.py   (requires E001 checkpoint)
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, run_dir, save_json, set_seed)

E001_CKPT = REPO / "runs" / "checkpoints" / "e001.pt"
N_POS = 2000
BATCH = 64


def ranks(x):
    return torch.argsort(torch.argsort(x)).float()


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ra = (ra - ra.mean()) / ra.std().clamp_min(1e-8)
    rb = (rb - rb.mean()) / rb.std().clamp_min(1e-8)
    return float((ra * rb).mean())


@torch.no_grad()
def batched_snapshots(model, xs):
    """xs: (B,T) → list of (B, C) last-position stream vectors per depth 0..L."""
    snaps, handles = [], []

    def pre(module, args):
        snaps.append(args[0].detach())
        return None

    handles.append(model.h[0].register_forward_pre_hook(pre))
    for block in model.h:
        def h(module, args, out):
            snaps.append(out.detach())
        handles.append(block.register_forward_hook(h))
    model(xs)
    for h_ in handles:
        h_.remove()
    return [s[:, -1, :] for s in snaps]


@torch.no_grad()
def main():
    set_seed(12)
    rd = run_dir("e012")
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)
    model = TinyGPT(cfg).to(DEVICE)
    model.load_state_dict(torch.load(E001_CKPT, map_location=DEVICE, weights_only=True))
    model.eval()

    gen = torch.Generator().manual_seed(12)
    ix = torch.randint(len(corpus.val) - cfg.block_size - 2, (N_POS,), generator=gen)
    xs = torch.stack([corpus.val[i : i + cfg.block_size] for i in ix]).to(DEVICE)
    ys = corpus.val[ix + cfg.block_size].to(DEVICE)

    depths = torch.full((N_POS,), -1, dtype=torch.long)
    entrop = torch.zeros(N_POS)
    top1_final = torch.zeros(N_POS, dtype=torch.long)
    kl_cal = torch.zeros(N_POS)  # KL(final(L5-readout) || L4-readout): L5's calibration role

    for b0 in range(0, N_POS, BATCH):
        xb, yb = xs[b0 : b0 + BATCH], ys[b0 : b0 + BATCH]
        last = batched_snapshots(model, xb)  # 7 × (B, C); 0=emb ... 6=after L5 (final)
        probs = [F.softmax(model.lm_head(model.ln_f(v)), dim=-1) for v in last]
        top1 = torch.stack([p.argmax(-1) for p in probs])          # (7, B)
        final = top1[-1]
        for d in range(cfg.n_layer + 1):  # shallowest depth whose top-1 survives to the end
            stable_d = (top1[d:] == final.unsqueeze(0)).all(dim=0)
            sel = (depths[b0:b0+BATCH] < 0) & stable_d.cpu()
            depths[b0:b0+BATCH][sel] = d
        p6 = probs[-1]
        entrop[b0:b0+BATCH] = -(p6 * (p6 + 1e-12).log()).sum(-1)
        top1_final[b0:b0+BATCH] = final
        kl_cal[b0:b0+BATCH] = (probs[6] * (probs[6] / (probs[5] + 1e-12)).log()).sum(-1)

    decided = depths >= 0
    print(f"decided (stable) positions: {int(decided.sum())}/{N_POS}")
    d = depths[decided].float()
    e = entrop[decided]

    rho = spearman(d, e)
    print(f"Spearman(depth, entropy) = {rho:.4f}   [P1 predicts <= -0.4]")

    # ablation: joint attn-L4+L5 zero; per-position CE at the last token only
    dce = torch.zeros(N_POS)
    base_ce = torch.zeros(N_POS)
    with torch.no_grad():
        for b0 in range(0, N_POS, BATCH):
            xb, yb = xs[b0 : b0 + BATCH], ys[b0 : b0 + BATCH]
            logits_b, _ = model(xb)  # logits are already ln_f+lm_head outputs
            base_ce[b0:b0+BATCH] = F.cross_entropy(logits_b[:, -1], yb, reduction="none")
    handles = [model.h[i].attn.register_forward_hook(
        lambda m, a, o: torch.zeros_like(o)) for i in (4, 5)]
    with torch.no_grad():
        for b0 in range(0, N_POS, BATCH):
            xb, yb = xs[b0 : b0 + BATCH], ys[b0 : b0 + BATCH]
            logits_a, _ = model(xb)  # hooks active now (ablated)
            dce[b0:b0+BATCH] = F.cross_entropy(logits_a[:, -1], yb, reduction="none")
    for h_ in handles:
        h_.remove()
    delta = (dce - base_ce)[decided]
    dd = depths[decided]
    early, late = delta[dd <= 2], delta[dd >= 4]
    ratio_late_over_early = float(late.mean() / early.mean().clamp_min(1e-6))
    print(f"ablation ΔCE: early(≤2) {early.mean():.4f}  late(≥4) {late.mean():.4f}  "
          f"ratio {ratio_late_over_early:.2f}   [P2 predicts >= 2]")

    kl_stable = kl_cal[decided]
    print(f"KL(p_L5readout || p_L4readout) mean {kl_stable.mean():.4f} nats, median {kl_stable.median():.4f}"
          f"   [P3 predicts > 0.05]")

    # ---- graphs ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    dcounts = torch.bincount(depths[decided], minlength=cfg.n_layer + 1).float()
    axes[0].bar(range(cfg.n_layer + 1), dcounts.cpu(), color="steelblue")
    axes[0].set_xticks(range(cfg.n_layer + 1))
    axes[0].set_xticklabels(["emb", "L0", "L1", "L2", "L3", "L4", "L5"])
    axes[0].set_title(f"decision depth (unstable: {int((~decided).sum())})")
    axes[0].set_xlabel("depth"); axes[0].set_ylabel("positions")
    for dd_ in range(cfg.n_layer + 1):
        m = depths[decided] == dd_
        if m.sum() > 5:
            axes[1].scatter(torch.full((int(m.sum()),), dd_), e[m].cpu(), s=4, alpha=0.25, color="crimson")
    import statistics
    means = [float(e[depths[decided] == k].mean()) if (depths[decided] == k).sum() > 5 else float("nan")
             for k in range(cfg.n_layer + 1)]
    axes[1].plot(range(cfg.n_layer + 1), means, "ko-", label="mean entropy")
    axes[1].set_xticks(range(cfg.n_layer + 1))
    axes[1].set_xticklabels(["emb", "L0", "L1", "L2", "L3", "L4", "L5"])
    axes[1].set_title(f"depth vs final entropy (Spearman {rho:.3f})")
    axes[1].set_xlabel("decision depth"); axes[1].set_ylabel("final next-token entropy (nats)"); axes[1].legend()
    axes[2].boxplot([early.cpu(), late.cpu()], labels=["early (≤2)", "late (≥4)"], showfliers=False)
    axes[2].set_title(f"ΔCE under attn-L4+L5 zero (ratio {ratio_late_over_early:.2f})")
    axes[2].set_ylabel("Δ CE (nats)")
    fig.suptitle("E012 decision-depth census")
    fig.tight_layout(); fig.savefig(rd / "decision_depth.png", dpi=140); plt.close(fig)

    save_json(rd / "metrics.json", {
        "experiment": "e012_decision_depth", "n_positions": N_POS,
        "decided_fraction": float(decided.float().mean()),
        "depth_histogram": dcounts.tolist(),
        "spearman_depth_entropy": rho,
        "ablation_dce_early": float(early.mean()), "ablation_dce_late": float(late.mean()),
        "late_over_early": ratio_late_over_early,
        "kl_L5_vs_L4_mean": float(kl_stable.mean()),
        "registered_predictions": {
            "P1_rho_le_-0.4": rho <= -0.4,
            "P2_late_ge_2x_early": ratio_late_over_early >= 2.0,
            "P3_kl_gt_0.05": float(kl_stable.mean()) > 0.05,
        },
    })
    print("registered:", {
        "P1_rho_le_-0.4": rho <= -0.4,
        "P2_late_ge_2x_early": ratio_late_over_early >= 2.0,
        "P3_kl_gt_0.05": float(kl_stable.mean()) > 0.05,
    })
    print(f"outputs: {rd}")


if __name__ == "__main__":
    main()
