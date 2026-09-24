"""E013 — Context-truncation test: does L5's calibration depend on far context?

T005 final form says L5 is a diffuse re-globalizer (census-confirmed shape).
The causal question: is L5's distribution reshaping (KL(L5-readout ‖
L4-readout) ≈ 1 nat) actually FUELED by far context?

Per held-out position, run the same window at full length (256) and
truncated (last 16). Measure KL(final ‖ L4-readout) and the L5 argmax-flip
rate under both. Registered prediction: KL shrinks ≥50% under truncation.
If KL is unchanged, L5's calibration is locally derived and
"re-globalization" is epiphenomenal attention shape.

Run: python lab/e013_truncation.py   (requires E001 checkpoint)
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, run_dir, save_json, set_seed)
from e012_decision_depth import batched_snapshots

E001_CKPT = REPO / "runs" / "checkpoints" / "e001.pt"
N_POS = 300
BATCH = 64
TRUNC = 16


@torch.no_grad()
def kl_and_flips(model, xs, ys):
    kl = torch.zeros(xs.shape[0])
    ce = torch.zeros(xs.shape[0])
    flips = torch.zeros(xs.shape[0])
    for b0 in range(0, xs.shape[0], BATCH):
        xb, yb = xs[b0:b0+BATCH], ys[b0:b0+BATCH]
        last = batched_snapshots(model, xb)
        p4 = F.softmax(model.lm_head(model.ln_f(last[5])), dim=-1)  # L4 readout
        p6 = F.softmax(model.lm_head(model.ln_f(last[6])), dim=-1)  # final/L5 readout
        kl[b0:b0+BATCH] = (p6 * (p6 / (p4 + 1e-12)).log()).sum(-1).cpu()
        idx = torch.arange(xb.shape[0])
        ce[b0:b0+BATCH] = -(p6[idx, yb] + 1e-12).log().cpu()  # CE from probs directly
        flips[b0:b0+BATCH] = (p4.argmax(-1) != p6.argmax(-1)).float().cpu()
    return kl, ce, flips


def main():
    set_seed(14)
    rd = run_dir("e013")
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)
    model = TinyGPT(cfg).to(DEVICE)
    model.load_state_dict(torch.load(E001_CKPT, map_location=DEVICE, weights_only=True))
    model.eval()

    gen = torch.Generator().manual_seed(14)
    ix = torch.randint(len(corpus.val) - cfg.block_size - 2, (N_POS,), generator=gen)
    xs_full = torch.stack([corpus.val[i : i + cfg.block_size] for i in ix]).to(DEVICE)
    ys = corpus.val[ix + cfg.block_size].to(DEVICE)
    xs_trunc = xs_full[:, -TRUNC:].contiguous()

    kl_f, ce_f, fl_f = kl_and_flips(model, xs_full, ys)
    kl_t, ce_t, fl_t = kl_and_flips(model, xs_trunc, ys)

    shrink = 1.0 - (kl_t.mean() / kl_f.mean())
    print(f"KL(L5||L4 readout): full-256 {kl_f.mean():.4f} -> trunc-16 {kl_t.mean():.4f} "
          f"(shrink {shrink*100:.1f}%  [registered: >= 50%])")
    print(f"argmax-flip rate L4->L5: full {fl_f.mean()*100:.1f}%  trunc {fl_t.mean()*100:.1f}%")
    print(f"CE: full {ce_f.mean():.4f}  trunc {ce_t.mean():.4f} (far context worth "
          f"{ce_t.mean()-ce_f.mean():+.3f} nats)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(kl_f.tolist(), bins=40, alpha=0.6, label=f"full-256 (mean {kl_f.mean():.2f})", color="steelblue")
    axes[0].hist(kl_t.tolist(), bins=40, alpha=0.6, label=f"trunc-16 (mean {kl_t.mean():.2f})", color="crimson")
    axes[0].set_xlabel("KL(final ‖ L4 readout), nats"); axes[0].legend()
    axes[0].set_title("L5's calibration reshaping vs context length")
    axes[1].bar(["full-256", "trunc-16"], [ce_f.mean(), ce_t.mean()], color=["steelblue", "crimson"])
    axes[1].set_ylabel("CE (nats)"); axes[1].set_title("how much far context is worth at all")
    fig.suptitle("E013 — is L5's re-globalization causally fed by far context?")
    fig.tight_layout(); fig.savefig(rd / "truncation.png", dpi=140); plt.close(fig)

    save_json(rd / "metrics.json", {
        "experiment": "e013_truncation", "n_positions": N_POS, "trunc_len": TRUNC,
        "kl_full": float(kl_f.mean()), "kl_trunc": float(kl_t.mean()),
        "kl_shrink_fraction": float(shrink),
        "flip_rate_full": float(fl_f.mean()), "flip_rate_trunc": float(fl_t.mean()),
        "ce_full": float(ce_f.mean()), "ce_trunc": float(ce_t.mean()),
        "registered": {"kl_shrinks_ge_50pct": bool(shrink >= 0.5)},
    })
    print("registered:", {"kl_shrinks_ge_50pct": bool(shrink >= 0.5)})
    print(f"outputs: {rd}")


if __name__ == "__main__":
    main()
