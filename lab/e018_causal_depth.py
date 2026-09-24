"""E018 — Causal depth by activation patching (T004 instrument validation).

Review 4's load-bearing-uncertainty finding: C1's depth-census invariance
could replicate by construction — every net shares the same ln_f+lm_head lens
bias, and e012's lens "depth 6" is definitionally the L5 argmax flip. The
causal upgrade: for ~1500 held-out positions, each context is paired with a
COUNTERFACTUAL context (a different random context whose final-token top-1
prediction differs from the original's). At each depth d (0..5 = the
last-position stream entering block L0..L5, i.e. snapshots emb-out/L0-out/
.../L4-out), that stream vector is replaced by the counterfactual's and the
remaining blocks run. A flip = the final top-1 becomes the counterfactual's
answer.

CAUSAL DEPTH = the shallowest d from which patching switches the final
decision — the point of no return. Patching at d replaces the information
flowing INTO block d; if the final answer becomes the counterfactual's, the
decision was still open at d.

REGISTERED (Review 4):
  (a) INSTRUMENT VALIDATED iff per-position Spearman(causal, lens) >= 0.6
      AND causal/lens distributions share the same dominant mode.
  (b) causal systematically SHALLOWER than the lens => C1's "deep decision"
      language overstates — late L5 argmax flips are recalibration, not
      decision (the lens watched argmax wobble).
  (c) causal >> lens => the lens under-detects decision depth.
  Also: fraction of positions where NO single-depth patch flips the decision
  (distributed decisions needing 2+ simultaneous patches).

Run: python lab/e018_causal_depth.py   (requires E001 checkpoint; eval-only)
"""
import json

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from common import DEVICE, REPO, Cfg, CharCorpus, TinyGPT, run_dir, save_json, set_seed

E001_CKPT = REPO / "runs" / "checkpoints" / "e001.pt"
E012_METRICS = REPO / "runs" / "e012" / "metrics.json"
N_POS = 1536          # ~1500 held-out positions
BATCH = 64
SEED = 18
DEPTHS = 6            # patch points d = 0..5 (streams entering L0..L5)


@torch.no_grad()
def snapshots_full(model, xs, last_only=False):
    """xs: (B,T) -> list of activations per depth 0..L (0 = emb, k = after L_{k-1}).
    last_only: keep only the last position's vector (counterfactual streams)."""
    snaps, handles = [], []

    def pre(module, args):
        snaps.append(args[0].detach() if not last_only else args[0].detach()[:, -1, :])
        return None

    handles.append(model.h[0].register_forward_pre_hook(pre))
    for block in model.h:
        def h(module, args, out):
            snaps.append(out.detach() if not last_only else out.detach()[:, -1, :])
        handles.append(block.register_forward_hook(h))
    model(xs)
    for h_ in handles:
        h_.remove()
    return snaps


@torch.no_grad()
def readout_top1(model, v):
    """v: (B, C) stream vectors -> top-1 through ln_f + lm_head."""
    return model.lm_head(model.ln_f(v)).argmax(-1)


def avg_ranks(x):
    """1-based average ranks with tie correction."""
    n = x.numel()
    order = torch.argsort(x)
    sx = x[order]
    r = torch.empty(n, dtype=torch.float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        r[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    out = torch.empty(n, dtype=torch.float)
    out[order] = r
    return out


def spearman(a, b):
    ra, rb = avg_ranks(a), avg_ranks(b)
    ra = (ra - ra.mean()) / ra.std().clamp_min(1e-8)
    rb = (rb - rb.mean()) / rb.std().clamp_min(1e-8)
    return float((ra * rb).mean())


@torch.no_grad()
def main():
    set_seed(SEED)
    rd = run_dir("e018")
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)
    model = TinyGPT(cfg).to(DEVICE)
    model.load_state_dict(torch.load(E001_CKPT, map_location=DEVICE, weights_only=True))
    model.eval()

    # ---- sample originals + counterfactual pool from held-out val ----
    gen_o = torch.Generator().manual_seed(SEED)         # original positions
    gen_p = torch.Generator().manual_seed(SEED + 1)     # counterfactual pool
    gen_m = torch.Generator().manual_seed(SEED + 2)     # pairing
    hi = len(corpus.val) - cfg.block_size - 2
    ix_o = torch.randint(hi, (N_POS,), generator=gen_o)
    ix_p = torch.randint(hi, (N_POS,), generator=gen_p)
    xs_o = torch.stack([corpus.val[i : i + cfg.block_size] for i in ix_o]).to(DEVICE)
    xs_p = torch.stack([corpus.val[i : i + cfg.block_size] for i in ix_p]).to(DEVICE)

    top1_o = torch.cat([
        readout_top1(model, snapshots_full(model, xs_o[b : b + BATCH], last_only=True)[-1])
        for b in range(0, N_POS, BATCH)
    ]).cpu()
    top1_p = torch.cat([
        readout_top1(model, snapshots_full(model, xs_p[b : b + BATCH], last_only=True)[-1])
        for b in range(0, N_POS, BATCH)
    ]).cpu()

    # pair each original with a pool context whose final top-1 DIFFERS
    cf = torch.zeros(N_POS, dtype=torch.long)
    for i in range(N_POS):
        valid = torch.nonzero((top1_p != top1_o[i]) & (ix_p != ix_o[i]), as_tuple=False).squeeze(1)
        cf[i] = valid[int(torch.randint(len(valid), (1,), generator=gen_m))]
    xs_c = xs_p[cf.to(DEVICE)]
    same_last_tok = (xs_o[:, -1].cpu() == xs_c[:, -1].cpu())  # d=0 patch is a no-op for these
    print(f"pairs: {N_POS}; counterfactual shares the same final char: "
          f"{int(same_last_tok.sum())} ({float(same_last_tok.float().mean()):.1%})")

    # ---- main pass: lens depth + patching at d = 0..5 ----
    lens_depths = torch.full((N_POS,), -1, dtype=torch.long)
    causal_depths = torch.full((N_POS,), -1, dtype=torch.long)
    flips = torch.zeros(N_POS, DEPTHS, dtype=torch.bool)   # top-1 became the CF's answer
    third = torch.zeros(N_POS, DEPTHS, dtype=torch.bool)   # top-1 became neither
    lens_top1 = torch.zeros(N_POS, DEPTHS + 1, dtype=torch.long)

    for b0 in range(0, N_POS, BATCH):
        xb, cb = xs_o[b0 : b0 + BATCH], xs_c[b0 : b0 + BATCH]
        A = snapshots_full(model, xb)                # 7 x (B,T,C): 0=emb ... 6=after L5
        C = snapshots_full(model, cb, last_only=True)  # 7 x (B, C)
        t1 = torch.stack([readout_top1(model, a[:, -1, :]) for a in A]).cpu()  # (7, B)
        lens_top1[b0 : b0 + BATCH] = t1.T
        final = t1[-1]
        for d in range(DEPTHS + 1):  # lens depth: shallowest stable top-1 (e012 protocol)
            stable = (t1[d:] == final.unsqueeze(0)).all(dim=0)
            sel = (lens_depths[b0 : b0 + BATCH] < 0) & stable.cpu()
            lens_depths[b0 : b0 + BATCH][sel] = d
        cf_final = readout_top1(model, C[-1]).cpu()  # CF answer on the CF context itself
        for d in range(DEPTHS):  # patch: original stream at depth d, CF last-position vector
            patched = A[d].clone()
            patched[:, -1, :] = C[d]
            x = patched
            for block in model.h[d:]:
                x = block(x)
            pt1 = readout_top1(model, x[:, -1, :]).cpu()
            flips[b0 : b0 + BATCH, d] = pt1 == cf_final
            third[b0 : b0 + BATCH, d] = (pt1 != cf_final) & (pt1 != final)
        del A, C
    for d in range(DEPTHS):
        sel = (causal_depths < 0) & flips[:, d]
        causal_depths[sel] = d

    # ---- distributions ----
    fl = causal_depths >= 0
    causal_hist = torch.bincount(causal_depths[fl], minlength=DEPTHS).float()
    lens_hist = torch.bincount(lens_depths, minlength=DEPTHS + 1).float()
    causal_mode = int(causal_hist.argmax())
    lens_mode = int(lens_hist.argmax())
    flip_curve = flips.float().mean(0)  # P(flip | patch at d)

    rho = spearman(causal_depths[fl].float(), lens_depths[fl].float())
    causal_cens = causal_depths.clone()
    causal_cens[~fl] = DEPTHS  # censored: no single-depth patch flips
    rho_cens = spearman(causal_cens.float(), lens_depths.float())

    mean_c = float(causal_depths[fl].float().mean())
    mean_l_same = float(lens_depths[fl].float().mean())
    mean_l_all = float(lens_depths.float().mean())
    med_c = float(causal_depths[fl].float().median())
    med_l = float(lens_depths.float().median())
    no_flip = float((~fl).float().mean())
    # closure offset: a flip at d means the decision was still open AT d; the
    # lens says "closed by L".  closure_causal = d + 1 puts them on one scale.
    closure_c = mean_c + 1.0

    # monotone "point of no return": flip set is a contiguous suffix d*..5
    suf = torch.zeros(N_POS, dtype=torch.bool)
    for i in range(N_POS):
        f = flips[i]
        if f.any():
            d0 = int(f.nonzero()[0])
            suf[i] = bool(f[d0:].all())
    suffix_frac = float(suf[fl].float().mean())
    third_rate = float(third.float().mean())

    # ---- registered verdict ----
    validated = (rho >= 0.6) and (causal_mode == lens_mode)
    if validated:
        verdict = "a"
    elif closure_c < mean_l_same - 0.25 or causal_mode < lens_mode:
        verdict = "b"
    elif closure_c > mean_l_same + 0.25:
        verdict = "c"
    else:
        verdict = "ambiguous"

    e012_hist = None
    if E012_METRICS.exists():
        e012_hist = json.loads(E012_METRICS.read_text(encoding="utf-8")).get("depth_histogram")

    print(f"\nflip curve P(flip|patch at d): {[f'{v:.3f}' for v in flip_curve.tolist()]}")
    print(f"no single-depth flip (distributed): {no_flip:.1%}")
    print(f"causal hist (d=0..5): {causal_hist.tolist()}   mode {causal_mode}")
    print(f"lens   hist (d=0..6): {lens_hist.tolist()}   mode {lens_mode}")
    print(f"Spearman(causal, lens) = {rho:.4f} (flippers) / {rho_cens:.4f} (censored)")
    print(f"mean causal {mean_c:.3f} (closure d+1: {closure_c:.3f}) vs lens {mean_l_same:.3f} "
          f"(same positions) / {mean_l_all:.3f} (all)")
    print(f"VERDICT: {verdict}   [a=validated, b=lens overstates depth, c=lens under-detects]")

    # ---- overlay histogram ----
    labels = ["emb", "after L0", "after L1", "after L2", "after L3", "after L4", "after L5", "no\nflip"]
    lx = list(range(DEPTHS + 2))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
    w = 0.4
    axes[0].bar([v - w / 2 for v in lx[: DEPTHS + 1]], (lens_hist / N_POS).tolist(), w,
                color="steelblue", label="lens depth (e012 protocol, same positions)")
    axes[0].bar([v + w / 2 for v in lx[:DEPTHS]], (causal_hist / N_POS).tolist(), w,
                color="crimson", label="causal depth (patching, flippers)")
    axes[0].bar([lx[DEPTHS + 1] + w / 2], [no_flip], w, color="crimson",
                alpha=0.45, hatch="//", label="causal: no single-depth flip")
    axes[0].set_xticks(lx); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("fraction of positions")
    axes[0].set_title(f"decision depth: causal vs lens  (modes {causal_mode} vs {lens_mode}; "
                      f"Spearman {rho:.3f})")
    axes[0].legend(fontsize=8)
    dd = torch.arange(DEPTHS)
    axes[1].plot(dd.tolist(), flip_curve.tolist(), "o-", color="crimson", label="P(flip to CF answer)")
    axes[1].plot(dd.tolist(), third.float().mean(0).tolist(), "s--", color="gray",
                 label="P(third token)")
    axes[1].set_xticks(dd.tolist())
    axes[1].set_xticklabels(["emb", "after L0", "after L1", "after L2", "after L3", "after L4"])
    axes[1].set_xlabel("patch depth (stream entering block d)")
    axes[1].set_ylabel("fraction")
    axes[1].set_title(f"patch flip curve (suffix-monotone: {suffix_frac:.0%} of flippers)")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"E018 causal depth — verdict {verdict} "
                 f"(a=validated / b=lens overstates / c=lens under-detects)")
    fig.tight_layout()
    fig.savefig(rd / "causal_vs_lens.png", dpi=140); plt.close(fig)

    save_json(rd / "metrics.json", {
        "experiment": "e018_causal_depth", "n_pairs": N_POS, "seed": SEED,
        "pairing": "counterfactual final top-1 differs from original's",
        "frac_cf_same_final_char": float(same_last_tok.float().mean()),
        "flip_curve_per_depth": flip_curve.tolist(),
        "third_token_rate_per_depth": third.float().mean(0).tolist(),
        "causal_depth_histogram_d0_d5": causal_hist.tolist(),
        "lens_depth_histogram_d0_d6_same_positions": lens_hist.tolist(),
        "e012_reference_histogram": e012_hist,
        "causal_mode": causal_mode, "lens_mode": lens_mode,
        "spearman_causal_lens_flippers": rho,
        "spearman_causal_lens_censored_none_as_6": rho_cens,
        "mean_causal_flippers": mean_c, "mean_causal_closure_d_plus_1": closure_c,
        "mean_lens_same_positions": mean_l_same, "mean_lens_all": mean_l_all,
        "median_causal": med_c, "median_lens": med_l,
        "no_single_depth_flip_fraction": no_flip,
        "suffix_monotone_flip_fraction": suffix_frac,
        "registered_verdicts": {
            "a_instrument_validated_rho_ge_0.6_and_same_mode": validated,
            "b_causal_shallower_lens_overstates": verdict == "b",
            "c_causal_deeper_lens_underdetects": verdict == "c",
            "verdict": verdict,
        },
    })
    print(f"outputs: {rd}")


if __name__ == "__main__":
    main()
