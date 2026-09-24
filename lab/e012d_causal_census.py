"""E012d — Causal-depth census across the 4-net matrix (T012's evidence debt).

T012 (e018): the lens (argmax-stability) is per-position UNCORRELATED with
causal depth (Spearman -0.009) on net B; causal mode = depth 3. All of C1's
cross-net depth evidence (the e012/e012b/e012c 4-net census invariance
0.82-0.85) was lens-based and is demoted to "argmax-stability profile".
Open debt: is CAUSAL depth the cross-net invariant the lens census falsely
claimed?

This run repeats the e018 protocol EXACTLY (same 1536 held-out positions,
same sampling seeds 18/19/20, counterfactual pairing per net, patch points
d=0..5) on the other three nets of the 2x2 matrix:
  B43 = seed-43 baseline   (runs/checkpoints/e028_b43.pt, no hooks)
  R   = seed-42 renorm     (runs/checkpoints/e014b.pt, register_renorm hooks at eval)
  R43 = seed-43 renorm     (runs/checkpoints/e029_r43.pt, register_renorm hooks at eval)
B itself is recomputed (determinism check against runs/e018/metrics.json)
because e018 did not persist per-position depths and the T012 scoping check
needs them.

REGISTERED (before running):
  - C1 RESTORED at H on causal footing iff cross-net Pearson correlation of
    the 7-bin causal histograms (d0..d5 + no-flip, fraction of positions;
    e012c's method applied to the causal instrument) is >= 0.8 on BOTH axes:
    seed axis  = min corr(B,B43), corr(R,R43)
    regime axis = min corr(B,R), corr(B43,R43)
  - Clustering or low correlation -> C1 stays scoped single-net (causal
    depth known only for B).
  - T012 scoping check (B only): within the lens-depth-6 subset the lens is
    constant, so Spearman(causal, lens) is undefined there; the well-posed
    form is Spearman(causal, 1[lens=6]) plus conditional causal-depth
    means/histograms: does the lens's "decided at L5" mass sit on causally
    deeper positions, or is "uncorrelated" exactly where the lens mass is?

Run: python lab/e012d_causal_census.py   (eval-only; ~minutes on GPU)
"""
import json
import time

import torch
import matplotlib.pyplot as plt

from common import DEVICE, REPO, Cfg, CharCorpus, TinyGPT, run_dir, save_json, set_seed
from e014b_stream_renorm import register_renorm

E018_METRICS = REPO / "runs" / "e018" / "metrics.json"
CKPTS = {  # name -> (ckpt path, renorm hooks at eval)
    "B":   (REPO / "runs" / "checkpoints" / "e001.pt",    False),
    "B43": (REPO / "runs" / "checkpoints" / "e028_b43.pt", False),
    "R":   (REPO / "runs" / "checkpoints" / "e014b.pt",    True),
    "R43": (REPO / "runs" / "checkpoints" / "e029_r43.pt", True),
}
N_POS = 1536          # identical to e018
BATCH = 64
SEED = 18             # e018's sampling seed (originals / pool / pairing)
DEPTHS = 6            # patch points d = 0..5 (streams entering L0..L5)
CENSOR_BIN = 6        # causal "no single-depth flip" bin (lens histogram also has 7 bins d0..d6)


# ------------------------------------------------ e018 machinery, verbatim
@torch.no_grad()
def snapshots_full(model, xs, last_only=False):
    """xs: (B,T) -> list of activations per depth 0..L (0 = emb, k = after L_{k-1})."""
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
    return model.lm_head(model.ln_f(v)).argmax(-1)


def avg_ranks(x):
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


def pearson(a, b):
    a = torch.as_tensor(a, dtype=torch.float)
    b = torch.as_tensor(b, dtype=torch.float)
    a = (a - a.mean()) / a.std().clamp_min(1e-8)
    b = (b - b.mean()) / b.std().clamp_min(1e-8)
    return float((a * b).mean())


# ------------------------------------------------ renorm liveness (e029 §3)
@torch.no_grad()
def renorm_liveness(model, corpus, c=5.6):
    probes, handles = [], []
    for block in model.h:
        def mk():
            def pre(m, args):
                probes.append(float(args[0].norm(dim=-1).mean()))
                return None
            return pre
        handles.append(block.register_forward_pre_hook(mk()))  # AFTER renorm hooks
    x, _ = corpus.get_batch("val", model.cfg.block_size, 16,
                            gen=torch.Generator().manual_seed(7))
    model(x)
    for h in handles:
        h.remove()
    if not all(abs(n - c) <= 1e-3 for n in probes):
        raise SystemExit(f"RENORM LIVENESS ASSERT FAILED: block-input norms {probes} != {c}")


# ------------------------------------------------ per-net census (e018 main pass, refactored)
@torch.no_grad()
def census(model, xs_o, xs_p, name=""):
    t0 = time.time()
    top1_o = torch.cat([
        readout_top1(model, snapshots_full(model, xs_o[b : b + BATCH], last_only=True)[-1])
        for b in range(0, N_POS, BATCH)
    ]).cpu()
    top1_p = torch.cat([
        readout_top1(model, snapshots_full(model, xs_p[b : b + BATCH], last_only=True)[-1])
        for b in range(0, N_POS, BATCH)
    ]).cpu()

    # pair each original with a pool context whose final top-1 DIFFERS (e018)
    gen_m = torch.Generator().manual_seed(SEED + 2)
    cf = torch.zeros(N_POS, dtype=torch.long)
    for i in range(N_POS):
        valid = torch.nonzero((top1_p != top1_o[i]) & (IX_P_CPU != IX_O_CPU), as_tuple=False).squeeze(1)
        cf[i] = valid[int(torch.randint(len(valid), (1,), generator=gen_m))]
    xs_c = xs_p[cf.to(DEVICE)]
    same_last_tok = (xs_o[:, -1].cpu() == xs_c[:, -1].cpu())

    lens_depths = torch.full((N_POS,), -1, dtype=torch.long)
    causal_depths = torch.full((N_POS,), -1, dtype=torch.long)
    flips = torch.zeros(N_POS, DEPTHS, dtype=torch.bool)
    third = torch.zeros(N_POS, DEPTHS, dtype=torch.bool)
    snap_consistent = True  # snapshot readout == full forward readout

    for b0 in range(0, N_POS, BATCH):
        xb, cb = xs_o[b0 : b0 + BATCH], xs_c[b0 : b0 + BATCH]
        A = snapshots_full(model, xb)
        C = snapshots_full(model, cb, last_only=True)
        t1 = torch.stack([readout_top1(model, a[:, -1, :]) for a in A]).cpu()
        final = t1[-1]
        if b0 == 0:  # plumbing sanity: full forward must match the snapshot readout
            fwd = model(xb)[0][:, -1, :].argmax(-1).cpu()
            snap_consistent = bool((fwd == final).all())
        for d in range(DEPTHS + 1):
            stable = (t1[d:] == final.unsqueeze(0)).all(dim=0)
            sel = (lens_depths[b0 : b0 + BATCH] < 0) & stable.cpu()
            lens_depths[b0 : b0 + BATCH][sel] = d
        cf_final = readout_top1(model, C[-1]).cpu()
        for d in range(DEPTHS):
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

    fl = causal_depths >= 0
    causal_cens = causal_depths.clone()
    causal_cens[~fl] = CENSOR_BIN
    hist6 = torch.bincount(causal_depths[fl], minlength=DEPTHS).float()
    hist7 = torch.bincount(causal_cens, minlength=DEPTHS + 1).float()
    lens_hist = torch.bincount(lens_depths, minlength=DEPTHS + 1).float()
    suf = torch.zeros(N_POS, dtype=torch.bool)
    for i in range(N_POS):
        f = flips[i]
        if f.any():
            d0 = int(f.nonzero()[0])
            suf[i] = bool(f[d0:].all())
    out = {
        "causal_hist_d0_d5": hist6.tolist(),
        "causal_hist_7bin_d0_d5_noflip": hist7.tolist(),
        "lens_hist_d0_d6_same_positions": lens_hist.tolist(),
        "causal_mode": int(hist6.argmax()),
        "lens_mode": int(lens_hist.argmax()),
        "no_single_depth_flip_fraction": float((~fl).float().mean()),
        "suffix_monotone_flip_fraction": float(suf[fl].float().mean()),
        "flip_curve_per_depth": flips.float().mean(0).tolist(),
        "third_token_rate_per_depth": third.float().mean(0).tolist(),
        "mean_causal_flippers": float(causal_depths[fl].float().mean()),
        "mean_lens_all": float(lens_depths.float().mean()),
        "frac_cf_same_final_char": float(same_last_tok.float().mean()),
        "snapshot_consistency_ok": snap_consistent,
        "_per_position": {"causal_cens": causal_cens, "lens": lens_depths,
                          "causal_raw": causal_depths, "fl": fl},
    }
    print(f"[{name}] causal hist {hist6.tolist()} mode {out['causal_mode']} "
          f"no-flip {out['no_single_depth_flip_fraction']:.1%} "
          f"(sanity {'OK' if snap_consistent else 'FAIL'}; {time.time()-t0:.0f}s)")
    return out


def main():
    set_seed(SEED)
    rd = run_dir("e012d")
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)

    # identical position sampling to e018 (generators, not global RNG)
    global IX_O_CPU, IX_P_CPU
    gen_o = torch.Generator().manual_seed(SEED)
    gen_p = torch.Generator().manual_seed(SEED + 1)
    hi = len(corpus.val) - cfg.block_size - 2
    IX_O_CPU = torch.randint(hi, (N_POS,), generator=gen_o)
    IX_P_CPU = torch.randint(hi, (N_POS,), generator=gen_p)
    xs_o = torch.stack([corpus.val[i : i + cfg.block_size] for i in IX_O_CPU]).to(DEVICE)
    xs_p = torch.stack([corpus.val[i : i + cfg.block_size] for i in IX_P_CPU]).to(DEVICE)

    e018 = json.loads(E018_METRICS.read_text(encoding="utf-8"))
    results = {}
    for name, (ckpt, hooked) in CKPTS.items():
        model = TinyGPT(cfg).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        model.eval()
        if hooked:
            register_renorm(model)
            renorm_liveness(model, corpus)
        results[name] = census(model, xs_o, xs_p, name=name)
        results[name]["ckpt"] = ckpt.name
        results[name]["renorm_hooks_at_eval"] = hooked
        del model

    # ---- B determinism check vs e018 ----
    b18 = e018["causal_depth_histogram_d0_d5"]
    b_now = results["B"]["causal_hist_d0_d5"]
    det_ok = all(abs(a - c) <= 2 for a, c in zip(b18, b_now))  # <=2 counts tolerance
    rho_e018 = e018["spearman_causal_lens_flippers"]
    pp = results["B"]["_per_position"]
    rho_repro = spearman(pp["causal_raw"][pp["fl"]].float(), pp["lens"][pp["fl"]].float())
    print(f"B vs e018: hist {b_now} vs {b18} (det {'OK' if det_ok else 'DRIFT'}); "
          f"spearman {rho_repro:+.4f} vs {rho_e018:+.4f}")

    # ---- cross-net correlations (e012c method on the causal instrument) ----
    order = ["B", "B43", "R", "R43"]
    h7 = {n: torch.tensor(results[n]["causal_hist_7bin_d0_d5_noflip"]) / N_POS for n in order}
    corr = {}
    for a in order:
        for b in order:
            if a == b:  # diagonal = even/odd split-half reliability
                even = results[a]["_per_position"]["causal_cens"][0::2]
                odd = results[a]["_per_position"]["causal_cens"][1::2]
                he = torch.bincount(even, minlength=DEPTHS + 1).float() / even.numel()
                ho = torch.bincount(odd, minlength=DEPTHS + 1).float() / odd.numel()
                corr[f"{a}|{b}"] = pearson(he, ho)
            else:
                corr[f"{a}|{b}"] = pearson(h7[a], h7[b])
    cross_seed = [corr["B|B43"], corr["R|R43"]]
    cross_regime = [corr["B|R"], corr["B43|R43"]]
    both_axes = [corr["B|R43"], corr["B43|R"]]
    seed_min, regime_min = min(cross_seed), min(cross_regime)
    restored = seed_min >= 0.8 and regime_min >= 0.8
    clustering_delta = float(sum(both_axes) / 2 - sum(cross_seed) / 2)
    verdict = "C1_RESTORED_H_causal" if restored else "C1_scoped_single_net"

    print("\n4x4 causal-hist Pearson (7-bin):")
    for a in order:
        print("  " + "  ".join(f"{corr[f'{a}|{b}']:+.3f}" for b in order) + f"   ({a})")
    print(f"seed axis {cross_seed} (min {seed_min:.3f}) | regime axis {cross_regime} "
          f"(min {regime_min:.3f}) | both-axes-diff {both_axes}")
    print(f"VERDICT: {verdict}")

    # ---- T012 scoping check (B only): causal depth where the lens mass is ----
    lens6 = pp["lens"] == 6
    n6 = int(lens6.sum())
    rho_all = spearman(pp["causal_cens"].float(), pp["lens"].float())
    ind = lens6.float()
    rho_ind = spearman(pp["causal_cens"].float(), ind)  # well-posed within-subset form
    mean_c_in = float(pp["causal_raw"][lens6 & pp["fl"]].float().mean())
    mean_c_out = float(pp["causal_raw"][(~lens6) & pp["fl"]].float().mean())
    med_c_in = float(pp["causal_raw"][lens6 & pp["fl"]].float().median())
    med_c_out = float(pp["causal_raw"][(~lens6) & pp["fl"]].float().median())
    noflip_in = float((~pp["fl"])[lens6].float().mean())
    noflip_out = float((~pp["fl"])[~lens6].float().mean())
    cond_hist_in = torch.bincount(pp["causal_cens"][lens6], minlength=DEPTHS + 1).tolist()
    cond_hist_out = torch.bincount(pp["causal_cens"][~lens6], minlength=DEPTHS + 1).tolist()
    print(f"\nB scoping: lens=6 subset n={n6} ({n6/N_POS:.1%}); "
          f"Spearman(causal, 1[lens=6]) = {rho_ind:+.4f} (all-position rho {rho_all:+.4f})")
    print(f"  mean causal depth: lens=6 {mean_c_in:.3f} (med {med_c_in}) vs lens<6 "
          f"{mean_c_out:.3f} (med {med_c_out}); no-flip {noflip_in:.1%} vs {noflip_out:.1%}")
    print(f"  causal hist | lens=6:  {cond_hist_in}")
    print(f"  causal hist | lens<6:  {cond_hist_out}")

    # ---- plot: 4-histogram overlay + flip curves + B conditional ----
    labels = ["emb", "after L0", "after L1", "after L2", "after L3", "after L4", "no\nflip"]
    colors = {"B": "crimson", "B43": "darkorange", "R": "steelblue", "R43": "teal"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    x = torch.arange(DEPTHS + 1)
    w = 0.2
    for k, n in enumerate(order):
        axes[0].bar((x + (k - 1.5) * w).tolist(), h7[n].tolist(), w,
                    color=colors[n], label=f"{n} (mode {results[n]['causal_mode']})")
        axes[1].plot(range(DEPTHS), results[n]["flip_curve_per_depth"], "o-",
                     color=colors[n], label=n)
    axes[0].set_xticks(x.tolist()); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("fraction of positions"); axes[0].legend(fontsize=8)
    axes[0].set_title(f"causal depth, 4-net overlay (seed {seed_min:.2f} / regime {regime_min:.2f})")
    axes[1].set_xticks(range(DEPTHS))
    axes[1].set_xticklabels(labels[:DEPTHS])
    axes[1].set_xlabel("patch depth"); axes[1].set_ylabel("P(flip to CF answer)")
    axes[1].set_title("patch flip curves"); axes[1].legend(fontsize=8)
    x2 = torch.arange(DEPTHS + 1)
    axes[2].bar((x2 - 0.2).tolist(), (torch.tensor(cond_hist_in) / n6).tolist(), 0.4,
                color="purple", label=f"lens=6 subset (n={n6})")
    axes[2].bar((x2 + 0.2).tolist(), (torch.tensor(cond_hist_out) / (N_POS - n6)).tolist(), 0.4,
                color="gray", label="lens<6")
    axes[2].set_xticks(x2.tolist()); axes[2].set_xticklabels(labels)
    axes[2].set_ylabel("fraction of subset"); axes[2].legend(fontsize=8)
    axes[2].set_title(f"B: causal depth where the lens mass is\n"
                      f"Spearman(causal,1[lens=6])={rho_ind:+.3f}; means {mean_c_in:.2f} vs {mean_c_out:.2f}")
    fig.suptitle(f"E012d causal census — {verdict} "
                 f"(seed-axis min {seed_min:.2f}, regime-axis min {regime_min:.2f}; bar 0.80)")
    fig.tight_layout()
    fig.savefig(rd / "causal_census_4net.png", dpi=140); plt.close(fig)

    per_net = {n: {k: v for k, v in results[n].items() if k != "_per_position"} for n in order}
    save_json(rd / "metrics.json", {
        "experiment": "e012d_causal_census", "n_pairs": N_POS, "seed": SEED,
        "protocol": "identical to e018: same positions (seeds 18/19/20), per-net counterfactual pairing, patch d=0..5",
        "nets": per_net,
        "b_determinism_vs_e018": {"e018_hist": b18, "recomputed_hist": b_now,
                                  "ok_le2counts": det_ok,
                                  "spearman_repro": rho_repro, "spearman_e018": rho_e018},
        "hist_correlation_4x4_pearson_7bin": corr,
        "cross_seed_same_regime": {"B|B43": corr["B|B43"], "R|R43": corr["R|R43"], "min": seed_min},
        "cross_regime_same_seed": {"B|R": corr["B|R"], "B43|R43": corr["B43|R43"], "min": regime_min},
        "both_axes_different": {"B|R43": corr["B|R43"], "B43|R": corr["B43|R"]},
        "verdict_rule": "C1 restored at H iff seed-axis min >= 0.8 AND regime-axis min >= 0.8",
        "verdict": verdict,
        "t012_scoping_B": {
            "lens6_n": n6, "lens6_frac": n6 / N_POS,
            "note": "within-subset Spearman(causal, lens) undefined (lens constant); well-posed form = Spearman(causal, 1[lens=6])",
            "spearman_causal_vs_lens6_indicator": rho_ind,
            "spearman_causal_vs_lens_all_positions": rho_all,
            "mean_causal_lens6": mean_c_in, "mean_causal_lens_lt6": mean_c_out,
            "median_causal_lens6": med_c_in, "median_causal_lens_lt6": med_c_out,
            "noflip_frac_lens6": noflip_in, "noflip_frac_lens_lt6": noflip_out,
            "causal_hist7_given_lens6": cond_hist_in,
            "causal_hist7_given_lens_lt6": cond_hist_out,
        },
    })
    print(f"outputs: {rd}")


if __name__ == "__main__":
    main()
