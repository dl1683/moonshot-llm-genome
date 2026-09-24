"""V002 — The attention atlas: what the L5 calibrator actually looks at.

E012 found layer-5 attention is a "calibrator" — KL(L5||L4 readout) ~1 nat
despite +0.03 ablation cost. This viz makes the mechanism visible: one
96-token Romeo prompt (padded with its true preceding corpus text so heads
have history), re-implemented attention to extract, for every one of the
36 heads, the last-token attention row — attention FROM the final token TO
every previous token.

Two figures:
  (1) ATLAS — 6x6 grid (layer x head), each cell the last-token attention
      distribution over position (log color); annotated with entropy.
  (2) DISTANCE — the money plot: per-head entropy bars L4 vs L5, mean
      attention-mass-by-distance curves (log bins) for L0-L5, and coarse
      distance-bin mass (local vs mid vs far vs sink).

Run: python lab/v002_attention_atlas.py   (requires E001 checkpoint)
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from common import DEVICE, REPO, Cfg, CharCorpus, TinyGPT, run_dir, save_json, set_seed

E001_CKPT = REPO / "runs" / "checkpoints" / "e001.pt"
PROMPT = "ROMEO:\nO, she doth teach the torches to burn "
CONTEXT_LEN = 96  # pad with true preceding corpus text so attention has history


def build_context(corpus: CharCorpus) -> str:
    """Prompt plus the text that genuinely precedes it in the corpus."""
    text = (REPO / "data" / "input.txt").read_text(encoding="utf-8")
    i = text.find(PROMPT)
    if i >= CONTEXT_LEN - len(PROMPT):
        return text[i - (CONTEXT_LEN - len(PROMPT)) : i] + PROMPT
    # fallback: prepend start-of-val text (continuity is worse but history exists)
    pad = corpus.decode(corpus.val[: CONTEXT_LEN - len(PROMPT)])
    return pad + PROMPT


@torch.no_grad()
def capture_ln1_inputs(model, idx):
    """One forward pass; return list of each block's residual input (pre-ln1)."""
    ln1_in, handles = [], []

    def pre(module, args):
        ln1_in.append(args[0].detach())
        return None

    for block in model.h:
        handles.append(block.ln1.register_forward_pre_hook(pre))
    model(idx)
    for h in handles:
        h.remove()
    return ln1_in


@torch.no_grad()
def last_token_attention(model, ln1_in):
    """Re-implement attention to expose weights the SDPA kernel hides.

    Returns attn of shape (L, n_head, T): softmax(q k^T / sqrt(d)) row for
    the LAST query position, per head, per layer. Also returns the max abs
    difference vs the model's own c_proj output at the last position (a
    sanity check that the re-implementation is faithful).
    """
    cfg = model.cfg
    L, T = cfg.n_layer, ln1_in[0].shape[1]
    out = torch.zeros(L, cfg.n_head, T)
    max_diff = 0.0
    for i, block in enumerate(model.h):
        x = block.ln1(ln1_in[i])                       # (1, T, C)
        B, T_, C = x.shape
        causal = torch.triu(torch.ones(T_, T_, dtype=torch.bool, device=x.device), diagonal=1)
        q, k, v = block.attn.c_attn(x).split(C, dim=2)
        q = q.view(B, T_, cfg.n_head, -1).transpose(1, 2)   # (1, nH, T, hd)
        k = k.view(B, T_, cfg.n_head, -1).transpose(1, 2)
        v = v.view(B, T_, cfg.n_head, -1).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        att = att.masked_fill(causal, float("-inf")).softmax(dim=-1)
        out[i] = att[0, :, -1, :].cpu()                # last query row
        # sanity: rebuild last-position output and compare with the real block
        y = (att @ v).transpose(1, 2).reshape(B, T_, C)
        rebuilt = block.attn.c_proj(y)[0, -1]
        real = block.attn(block.ln1(ln1_in[i]))[0, -1]
        max_diff = max(max_diff, float((rebuilt - real).abs().max()))
    return out, max_diff


@torch.no_grad()
def main():
    set_seed(5)
    rd = run_dir("v002")
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)
    model = TinyGPT(cfg).to(DEVICE)
    model.load_state_dict(torch.load(E001_CKPT, map_location=DEVICE, weights_only=True))
    model.eval()

    ctx = build_context(corpus)
    idx = corpus.encode(ctx).unsqueeze(0).to(DEVICE)
    T = idx.shape[1]
    chars = list(ctx)
    print(f"context ({T} tokens), tail: {ctx[-44:]!r}")

    ln1_in = capture_ln1_inputs(model, idx)
    attn, max_diff = last_token_attention(model, ln1_in)   # (L, nH, T)
    print(f"re-implementation sanity: max |rebuilt - model| at last pos = {max_diff:.2e}")
    L, nH = cfg.n_layer, cfg.n_head

    # --- per-head stats ---
    eps = 1e-12
    ent = -(attn.clamp_min(eps) * attn.clamp_min(eps).log()).sum(-1)      # (L, nH) nats
    dist = T - 1 - torch.arange(T)                                        # distance from end
    bins = {"1-3": (1, 3), "4-16": (4, 16), "17-64": (17, 64), "65+": (65, T - 1)}
    self_mass = attn[:, :, -1]                                            # d = 0
    bin_mass = {name: ((dist >= lo) & (dist <= hi)).float() @ attn.mT for name, (lo, hi) in bins.items()}
    sink_mass = attn[:, :, 0]                                             # mass on token 0

    # surprisal (bits) of the context actually attended to, per head
    freq = torch.bincount(corpus.train, minlength=cfg.vocab).float()
    freq = (freq + 1) / (freq.sum() + cfg.vocab)
    surprisal = -torch.log2(freq[idx[0].cpu()])                           # (T,)
    attended_surprisal = attn @ surprisal                                  # (L, nH)

    # log-spaced distance bins for the curve
    edges = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
    edges = sorted({e for e in edges if 1 <= e < T} | {T})
    curve = {}
    for li in range(L):
        m = attn[li].mean(0)                                              # (T,) mean over heads
        dens, centers = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (dist >= a) & (dist < b)
            dens.append(float(m[sel].sum() / max(1, int(sel.sum()))))
            centers.append(math.sqrt(a * max(a + 1, b - 1)))
        curve[li] = (centers, dens)

    per_head_entropy = {f"L{li}": {f"h{hi}": round(float(ent[li, hi]), 3) for hi in range(nH)} for li in range(L)}
    per_layer_bins = {
        f"L{li}": {"d0_self": round(float(self_mass[li].mean()), 4),
                   **{f"d{name}": round(float(bin_mass[name][li].mean()), 4) for name in bins}}
        for li in range(L)
    }
    metrics = {
        "viz": "v002_attention_atlas",
        "prompt_tail": PROMPT.strip()[-26:],
        "context_len": T,
        "sanity_max_abs_diff_vs_model": max_diff,
        "per_head_entropy_nats": per_head_entropy,
        "mean_entropy_by_layer": {f"L{li}": round(float(ent[li].mean()), 3) for li in range(L)},
        "per_layer_distance_bins": per_layer_bins,
        "per_layer_logbin_density": {f"L{li}": [round(d, 6) for d in curve[li][1]] for li in range(L)},
        "logbin_edges": edges,
        "per_head_sink_mass_pos0": {f"L{li}": {f"h{hi}": round(float(sink_mass[li, hi]), 4) for hi in range(nH)} for li in range(L)},
        "per_head_attended_surprisal_bits": {f"L{li}": {f"h{hi}": round(float(attended_surprisal[li, hi]), 3) for hi in range(nH)} for li in range(L)},
        "top_positions_L4_L5": {
            f"L{li}.h{hi}": [{"pos": int(p), "char": chars[int(p)], "w": round(float(attn[li, hi, p]), 3)}
                             for p in attn[li, hi].topk(3).indices]
            for li in (4, 5) for hi in range(nH)
        },
    }

    # ---------------- Figure 1: the atlas ----------------
    A = attn.numpy()
    fig, axes = plt.subplots(L, nH, figsize=(16, 9), sharex=True)
    im = None
    for li in range(L):
        for hi in range(nH):
            ax = axes[li, hi]
            im = ax.imshow(A[li, hi][None, :], aspect="auto", cmap="magma",
                           norm=LogNorm(vmin=max(A.min(), 1e-5), vmax=A.max()))
            ax.set_yticks([])
            ax.text(0.01, 0.72, f"H={ent[li, hi]:.2f}", transform=ax.transAxes,
                    fontsize=7.5, color="white",
                    bbox=dict(fc="black", alpha=0.55, ec="none", pad=1))
            if li == 0:
                ax.set_title(f"head {hi}", fontsize=9)
            if hi == 0:
                ax.set_ylabel(f"L{li}", fontsize=10, fontweight="bold")
            if li == L - 1:
                ticks = list(range(0, T, 4))
                ax.set_xticks(ticks)
                ax.set_xticklabels([chars[t] for t in ticks], fontsize=5, fontname="monospace")
            else:
                ax.set_xticks([])
    fig.colorbar(im, ax=axes, shrink=0.7, label="attention (log scale)")
    fig.suptitle(f"V002 attention atlas — last-token attention over context (query = final char of "
                 f"{PROMPT.strip()[-26:]!r}); H = entropy (nats)", fontsize=12)
    fig.savefig(rd / "attention_atlas.png", dpi=140)
    plt.close(fig)

    # ---------------- Figure 2: the money plot ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # (a) per-head entropy, L4 vs L5
    ax = axes[0]
    xs = np.arange(nH)
    ax.bar(xs - 0.2, ent[4].numpy(), 0.4, label=f"L4 (mean {ent[4].mean():.2f})", color="steelblue")
    ax.bar(xs + 0.2, ent[5].numpy(), 0.4, label=f"L5 (mean {ent[5].mean():.2f})", color="crimson")
    ax.axhline(math.log(T), ls="--", c="gray", lw=0.8, label=f"uniform = {math.log(T):.2f}")
    ax.set_xticks(xs); ax.set_xlabel("head"); ax.set_ylabel("entropy (nats)")
    ax.set_title("last-token attention entropy: L4 vs L5\n(uniform = log 96 = 4.56)")
    ax.legend(fontsize=8)

    # (b) mean attention-mass density vs distance from end, per layer
    ax = axes[1]
    for li in range(L):
        c, d = curve[li]
        ax.plot(c, d, "o-", color=plt.cm.plasma(li / 5), lw=1.6, label=f"L{li}", ms=3.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("distance from last token (log bins)")
    ax.set_ylabel("attention mass per distance")
    ax.set_title("how far back each layer reaches\n(mean over heads, log-log)")
    ax.legend(fontsize=8, ncol=2)

    # (c) coarse distance bins per layer (stacked, includes self)
    ax = axes[2]
    segs = [("self (d=0)", self_mass.mean(1), "lightgray"),
            ("d 1-3 (local)", bin_mass["1-3"].mean(1), "seagreen"),
            ("d 4-16 (mid)", bin_mass["4-16"].mean(1), "goldenrod"),
            ("d 17-64 (far)", bin_mass["17-64"].mean(1), "darkorange"),
            ("d 65+ (ancient)", bin_mass["65+"].mean(1), "purple")]
    bottom = np.zeros(L)
    for name, vals, color in segs:
        ax.bar(range(L), vals.numpy(), 0.6, bottom=bottom, label=name, color=color)
        bottom += vals.numpy()
    ax.set_xticks(range(L)); ax.set_xticklabels([f"L{i}" for i in range(L)])
    ax.set_ylabel("fraction of attention mass")
    ax.set_title("local vs global: distance profile by layer")
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.0, 0.5))

    fig.suptitle("V002 — what the L5 calibrator looks at (vs L4): entropy, reach, locality")
    fig.tight_layout()
    fig.savefig(rd / "attention_distance.png", dpi=140)
    plt.close(fig)

    save_json(rd / "metrics.json", metrics)

    # ---------------- verdict ----------------
    e4, e5 = float(ent[4].mean()), float(ent[5].mean())
    loc4, loc5 = float(bin_mass["1-3"][4].mean()), float(bin_mass["1-3"][5].mean())
    far4, far5 = float(bin_mass["65+"][4].mean()), float(bin_mass["65+"][5].mean())
    s4, s5 = float(sink_mass[4].mean()), float(sink_mass[5].mean())
    u4, u5 = float(attended_surprisal[4].mean()), float(attended_surprisal[5].mean())
    print("\nVERDICT (L5 vs L4, last-token attention):")
    print(f"  entropy      L4 {e4:.2f} -> L5 {e5:.2f} nats ({e5 - e4:+.2f}; uniform=4.56)")
    print(f"  local d1-3   L4 {loc4:.3f} -> L5 {loc5:.3f}")
    print(f"  ancient d65+ L4 {far4:.3f} -> L5 {far5:.3f}")
    print(f"  sink (pos 0) L4 {s4:.3f} -> L5 {s5:.3f}")
    print(f"  attended-token surprisal L4 {u4:.2f} -> L5 {u5:.2f} bits (rarity of what is read)")
    if e5 > e4 + 0.15:
        diff = "MORE DIFFUSE"
    elif e5 < e4 - 0.15:
        diff = "MORE CONCENTRATED"
    else:
        diff = "similarly spread"
    positional = "positional/sink-driven" if abs(s5 - s4) > 0.08 else "no strong sink shift"
    rare = "reads RARER tokens" if u5 > u4 + 0.15 else ("reads COMMONER tokens" if u5 < u4 - 0.15 else "similar token rarity")
    print(f"  => L5 attention is {diff}, {positional}, {rare}")
    print(f"outputs: {rd}")


if __name__ == "__main__":
    main()
