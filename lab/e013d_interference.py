"""E013d — Interference audit (T007 P1+P2): why does far context HURT 28% of positions?

Two discriminators:
  (1) DIVERGENT-CONTINUATION PROXIMITY (P1): for each position, search the
      far context (>=16 chars back) for the longest suffix of the local
      window that re-appears followed by a DIFFERENT character. H1
      (interference) predicts losers have systematically longer divergent
      matches than neutral positions.
  (2) SHUFFLED-FAR COLLAPSE (P2): recompute CE with the far 240 chars
      char-shuffled (local 16 intact). Structure-driven tails (H1+G1)
      collapse toward 0; a surviving tail is length/noise-driven (H2).

Run: python lab/e013d_interference.py   (requires E001 checkpoint)
"""
import random

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, run_dir, save_json, set_seed)

E001_CKPT = REPO / "runs" / "checkpoints" / "e001.pt"
N_POS = 2000
BATCH = 64
TRUNC = 16


@torch.no_grad()
def last_ce(model, xs, ys):
    out = torch.zeros(xs.shape[0])
    for b0 in range(0, xs.shape[0], BATCH):
        xb, yb = xs[b0:b0+BATCH], ys[b0:b0+BATCH]
        logits, _ = model(xb)
        out[b0:b0+BATCH] = F.cross_entropy(logits[:, -1], yb, reduction="none").cpu()
    return out


def longest_divergent_match(window: str, target: str, k_max=12, k_min=4) -> int:
    """Longest suffix of the local context re-appearing in the far region
    followed by a different char than the target."""
    local = window[-TRUNC:]
    far = window[:-TRUNC]
    for m in range(min(k_max, len(local)), k_min - 1, -1):
        sfx = local[-m:]
        start = 0
        while True:
            j = far.find(sfx, start)
            if j < 0:
                break
            nxt = far[j + m] if j + m < len(far) else window[len(window) - TRUNC] if False else ""
            if nxt and nxt != target:
                return m
            start = j + 1
    return 0


def main():
    set_seed(15)
    rd = run_dir("e013d")
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)
    model = TinyGPT(cfg).to(DEVICE)
    model.load_state_dict(torch.load(E001_CKPT, map_location=DEVICE, weights_only=True))
    model.eval()

    gen = torch.Generator().manual_seed(15)  # identical positions to E013c
    ix = torch.randint(len(corpus.val) - cfg.block_size - 2, (N_POS,), generator=gen)
    xs_full = torch.stack([corpus.val[i : i + cfg.block_size] for i in ix]).to(DEVICE)
    ys = corpus.val[ix + cfg.block_size].to(DEVICE)
    xs_trunc = xs_full[:, -TRUNC:].contiguous()

    rng = random.Random(15)
    xs_shuf = xs_full.clone()
    far_len = cfg.block_size - TRUNC
    for r in range(N_POS):
        far = xs_shuf[r, :far_len].tolist()
        rng.shuffle(far)
        xs_shuf[r, :far_len] = torch.tensor(far)

    ce_f = last_ce(model, xs_full, ys)
    ce_t = last_ce(model, xs_trunc, ys)
    ce_s = last_ce(model, xs_shuf, ys)
    fv = ce_t - ce_f
    fv_shuf = ce_t - ce_s

    losers = fv <= -0.15
    gainers = fv >= 0.15
    neutral = (fv.abs() < 0.05)

    print(f"populations: losers {int(losers.sum())} | neutral {int(neutral.sum())} | gainers {int(gainers.sum())}")

    # ---- P1: divergent-continuation proximity ----
    dmatch = torch.zeros(N_POS)
    for p in range(N_POS):
        w = corpus.decode(corpus.val[int(ix[p]) : int(ix[p]) + cfg.block_size])
        t = corpus.itos[int(ys[p])]
        dmatch[p] = longest_divergent_match(w, t)
    for name, mask in (("losers", losers), ("neutral", neutral), ("gainers", gainers)):
        seg = dmatch[mask]
        print(f"  {name}: mean divergent match {seg.mean():.2f} | frac >=6: {(seg >= 6).float().mean()*100:.0f}% | frac 0: {(seg == 0).float().mean()*100:.0f}%")
    pooled_sd = dmatch.std()
    effect = (dmatch[losers].mean() - dmatch[neutral].mean()) / pooled_sd
    print(f"P1 (losers > neutral by >= 0.5 sd): effect {effect:+.2f} -> "
          f"{'CONFIRMED' if effect >= 0.5 else 'REFUTED'}")

    # ---- P2: shuffled-far collapse ----
    def dec_stats(v):
        top = v[v.argsort(descending=True)[: N_POS // 10]].mean()
        bot = v[v.argsort()[: N_POS // 10]].mean()
        return float(bot), float(top)
    bot_f, top_f = dec_stats(fv)
    bot_s, top_s = dec_stats(fv_shuf)
    print(f"P2 real tails:  bottom {bot_f:+.2f} top {top_f:+.2f}")
    print(f"P2 shuffled tails: bottom {bot_s:+.2f} top {top_s:+.2f}")
    both_collapse = (bot_s >= -0.5) and (top_s <= 1.0)
    print(f"P2 (both tails collapse: bottom >= -0.5 AND top <= +1.0): "
          f"{'CONFIRMED' if both_collapse else 'REFUTED'}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    groups = [dmatch[losers].tolist(), dmatch[neutral].tolist(), dmatch[gainers].tolist()]
    axes[0].boxplot(groups, tick_labels=["losers", "neutral", "gainers"], showfliers=False)
    axes[0].set_ylabel("longest divergent-continuation match (chars)")
    axes[0].set_title(f"P1 interference signature (effect {effect:+.2f}σ)")
    axes[1].hist(fv.tolist(), bins=50, alpha=0.55, label="real far", color="steelblue")
    axes[1].hist(fv_shuf.tolist(), bins=50, alpha=0.55, label="shuffled far", color="gray")
    axes[1].set_xlabel("far-value = CE(16) − CE(256)"); axes[1].legend()
    axes[1].set_title("P2: shuffled-far collapse test")
    fig.suptitle("E013d — is the hurt population interference from near-repeat phrases?")
    fig.tight_layout(); fig.savefig(rd / "interference.png", dpi=140); plt.close(fig)

    save_json(rd / "metrics.json", {
        "experiment": "e013d_interference",
        "p1_effect_sd": float(effect),
        "mean_divergent_match": {"losers": float(dmatch[losers].mean()),
                                  "neutral": float(dmatch[neutral].mean()),
                                  "gainers": float(dmatch[gainers].mean())},
        "p2_tails": {"real": {"bottom": bot_f, "top": top_f},
                      "shuffled": {"bottom": bot_s, "top": top_s}},
        "registered": {"P1_interference": bool(effect >= 0.5),
                       "P2_structure_driven": bool(both_collapse)},
    })
    print(f"outputs: {rd}")


if __name__ == "__main__":
    main()
