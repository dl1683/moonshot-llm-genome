"""E013c — Far-value tail: who actually uses far context?

E013 found 16-token sufficiency ON AVERAGE (far context worth ≈0 nats). This
maps the distribution: per held-out position, far-value = CE(trunc-16) −
CE(full-256). Registered predictions (T005 closure block):
  P1: ≥5% of positions have far-value ≥ +0.15 nats (a real tail exists).
  P2: top-decile mean far-value ≥ +0.3 nats.
  P3: far-value correlates positively with local difficulty (trunc CE):
      harder positions benefit more from far context (Spearman ρ ≥ 0.2).

Run: python lab/e013c_far_value_tail.py   (requires E001 checkpoint)
"""
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


def ranks(x):
    return torch.argsort(torch.argsort(x)).float()


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ra = (ra - ra.mean()) / ra.std()
    rb = (rb - rb.mean()) / rb.std()
    return float((ra * rb).mean())


def main():
    set_seed(15)
    rd = run_dir("e013c")
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)
    model = TinyGPT(cfg).to(DEVICE)
    model.load_state_dict(torch.load(E001_CKPT, map_location=DEVICE, weights_only=True))
    model.eval()

    gen = torch.Generator().manual_seed(15)
    ix = torch.randint(len(corpus.val) - cfg.block_size - 2, (N_POS,), generator=gen)
    xs_full = torch.stack([corpus.val[i : i + cfg.block_size] for i in ix]).to(DEVICE)
    ys = corpus.val[ix + cfg.block_size].to(DEVICE)
    xs_trunc = xs_full[:, -TRUNC:].contiguous()

    ce_f = last_ce(model, xs_full, ys)
    ce_t = last_ce(model, xs_trunc, ys)
    far_value = ce_t - ce_f  # positive = far context helps

    frac_tail = float((far_value >= 0.15).float().mean())
    top_dec = far_value[far_value.argsort(descending=True)[: N_POS // 10]].mean()
    bottom_dec = far_value[far_value.argsort()[: N_POS // 10]].mean()
    rho = spearman(far_value, ce_t)
    frac_hurt = float((far_value <= -0.15).float().mean())

    print(f"far-value: mean {far_value.mean():+.4f} | median {far_value.median():+.4f} "
          f"| p90 {far_value.quantile(0.9):+.3f} | p99 {far_value.quantile(0.99):+.3f}")
    print(f"P1 (>=5% positions >= +0.15): {frac_tail*100:.1f}%  -> {'CONFIRMED' if frac_tail >= 0.05 else 'REFUTED'}")
    print(f"P2 (top-decile mean >= +0.3): {float(top_dec):+.3f} -> {'CONFIRMED' if top_dec >= 0.3 else 'REFUTED'}")
    print(f"P3 (rho(far-value, trunc-CE) >= 0.2): {rho:.3f} -> {'CONFIRMED' if rho >= 0.2 else 'REFUTED'}")
    print(f"positions HURT by far context (<= -0.15): {frac_hurt*100:.1f}% | "
          f"bottom-decile mean {float(bottom_dec):+.3f}")

    # what are the top far-value positions? show a few contexts
    top_idx = far_value.argsort(descending=True)[:8]
    ctx_samples = []
    for i in top_idx.tolist():
        ctx = corpus.decode(corpus.val[int(ix[i]) - 24 : int(ix[i]) + 1])
        target = corpus.itos[int(ys[i])]
        ctx_samples.append({"context_tail": ctx.replace("\n", "\\n"), "next_char": target,
                            "far_value": round(float(far_value[i]), 3),
                            "trunc_ce": round(float(ce_t[i]), 3)})
    for s in ctx_samples[:5]:
        print("  top:", s)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    axes[0].hist(far_value.tolist(), bins=60, color="steelblue")
    axes[0].axvline(0.15, color="crimson", ls="--", label="+0.15 tail")
    axes[0].axvline(-0.15, color="darkorange", ls="--", label="−0.15 hurt")
    axes[0].set_xlabel("far-value = CE(16 ctx) − CE(256 ctx)"); axes[0].legend()
    axes[0].set_title(f"far-value distribution (mean {far_value.mean():+.3f})")
    axes[1].scatter(ce_t.tolist(), far_value.tolist(), s=4, alpha=0.25, color="gray")
    axes[1].set_xlabel("local difficulty: CE with 16-token context")
    axes[1].set_ylabel("far-value"); axes[1].set_title(f"harder positions benefit more? (ρ={rho:.2f})")
    fig.suptitle("E013c — who actually uses far context?")
    fig.tight_layout(); fig.savefig(rd / "far_value_tail.png", dpi=140); plt.close(fig)

    save_json(rd / "metrics.json", {
        "experiment": "e013c_far_value_tail", "n_positions": N_POS,
        "far_value_mean": float(far_value.mean()), "far_value_median": float(far_value.median()),
        "frac_tail_ge_015": frac_tail, "frac_hurt_le_-015": frac_hurt,
        "top_decile_mean": float(top_dec), "bottom_decile_mean": float(bottom_dec),
        "spearman_farvalue_vs_truncce": rho,
        "top_contexts": ctx_samples,
        "registered": {"P1_tail_exists": frac_tail >= 0.05,
                       "P2_top_decile_ge_0.3": bool(top_dec >= 0.3),
                       "P3_rho_ge_0.2": rho >= 0.2},
    })
    print(f"outputs: {rd}")


if __name__ == "__main__":
    main()
