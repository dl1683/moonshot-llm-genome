"""E030 — replication-debt eval slot: three debts, one run (ALL EVAL-ONLY).

Task 1 (e012c) — de-confound T008 claim 1 ("function is staged; stages are
    the organism", currently H only via the B-vs-R seed-42 pair):
    decision-depth census (2000 positions, seed-12 sampling IDENTICAL to
    E012) on B43 (seed-43 baseline, runs/checkpoints/e028_b43.pt, no hooks)
    and R43 (seed-43 renorm, runs/checkpoints/e029_r43.pt, register_renorm
    hooks active). 4-net histogram bin-correlation table vs E012's B
    (seed 42, stored metrics) and E012b's R (seed 42, stored metrics).
    KEY QUESTION: do depth histograms match across SEEDS as well as across
    regimes? Registered verdict rule: claim 1 is init-independent (stays/
    is restored at H) iff min(corr(B,B43), corr(R,R43)) >= 0.8 — the same
    bar E012b applied to the regime axis — AND corr(B43,R43) >= 0.8
    (regime-axis result replicates in seed 43). If histograms cluster by
    seed (cross-seed corrs materially below cross-regime), claim 1 is
    init-bound and stays M.

Task 2 (e014b.1) — replicate claim 2 (anatomy plasticity) in a second
    renorm seed: full lesion map (attn+mlp zero per layer, 30 batches,
    exact e014b protocol) + write/stream profile on R43, hooks active.
    Val CE reported FIRST (parity gate <= 1.7224 = e001 val + 0.10).
    Registered: keystone dissolution replicated iff R43 mlp-L0 damage
    <= 0.5 nats (B +4.08, R +0.10); late-heavy MLP flip replicated iff
    mean(mlp[L3..L5]) > mean(mlp[L0..L2]) (B is front-heavy 1.50 vs 0.55);
    attn front-load replicated iff attn-L0 - attn-L5 >= 1.0 (B 2.37,
    R 2.77). Claim-2 replication = parity AND all three.

Task 3 (e011c-ci) — error bars on E011c's rotate/zero ratios for the
    three flagged cells (attn-L0 x1.38, mlp-L1 x2.95, mlp-L5 x0.24):
    rerun each cell with fresh eval-batch seeds AND fresh rotation seeds
    (zero and rotate PAIRED on the same batches per variant), mean +/- sd
    of the ratio. Beyond noise iff |mean - 1| > 2*sd (ratio = 1 means
    rotate ~= zero = pure perturbation energy).
    Deviation from the brief (a superset, noted here): 5 noise variants
    instead of 3 (same cost class, better sd), plus one exact-protocol
    anchor replication (batch seed 1337, E011c's original rotation seeds).

No NOTES/THINKING/QUEUE/STATE edits; no git commit (per instructions).

Run: python lab/e030_debt_slot.py
"""
from __future__ import annotations

import json
import statistics
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (DEVICE, REPO, Cfg, CharCorpus, TinyGPT, estimate_loss,
                    run_dir, save_json, set_seed)
from e011c_matched_perturbation import rotated_like
from e012_decision_depth import N_POS, spearman
from e012b_renorm_census import depth_census
from e014b_stream_renorm import (lesion_profile, register_renorm,
                                 write_stream_profile)

CKPTS = {
    "B":   REPO / "runs" / "checkpoints" / "e001.pt",      # seed 42, base
    "R":   REPO / "runs" / "checkpoints" / "e014b.pt",     # seed 42, renorm
    "B43": REPO / "runs" / "checkpoints" / "e028_b43.pt",  # seed 43, base
    "R43": REPO / "runs" / "checkpoints" / "e029_r43.pt",  # seed 43, renorm
}
REF_VALS = {"B": 1.622391394774119, "R": 1.6099902311960856,
            "B43": 1.5695625305175782, "R43": 1.559572164217631}
RENORM = {"B": False, "R": True, "B43": False, "R43": True}
PARITY_GATE = 1.7224
N_EVAL = 30
NAMES = ["B", "R", "B43", "R43"]
CLASSES = ("newline", "space", "uppercase", "lowercase", "punct")
CLASS_ID = {"\n": 0, " ": 1}
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def class_id(ch: str) -> int:
    if ch in CLASS_ID:
        return CLASS_ID[ch]
    if ch.isupper():
        return 2
    if ch.islower():
        return 3
    return 4


@torch.no_grad()
def eval_ce(model, corpus, gen_seed: int, n_batches: int = 20) -> float:
    """estimate_loss replica with a configurable batch-generator seed."""
    model.eval()
    cfg = model.cfg
    gen = torch.Generator().manual_seed(gen_seed)
    src = corpus.val
    losses = []
    for _ in range(n_batches):
        ix = torch.randint(len(src) - cfg.block_size - 1, (16,), generator=gen)
        x = torch.stack([src[i:i + cfg.block_size] for i in ix]).to(DEVICE)
        y = torch.stack([src[i + 1:i + 1 + cfg.block_size] for i in ix]).to(DEVICE)
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    return sum(losses) / len(losses)


def hist_corr(a: list, b: list) -> float:
    """Bin-wise z-scored correlation (exact e012b formula)."""
    ta, tb = torch.tensor(a, dtype=torch.float), torch.tensor(b, dtype=torch.float)
    za = (ta - ta.mean()) / ta.std().clamp_min(1e-8)
    zb = (tb - tb.mean()) / tb.std().clamp_min(1e-8)
    return float((za * zb).mean())


@torch.no_grad()
def renorm_liveness(model, corpus) -> list[float]:
    probes, handles = [], []
    for block in model.h:
        def mk():
            def pre(m, args):
                probes.append(float(args[0].norm(dim=-1).mean()))
                return None
            return pre
        handles.append(block.register_forward_pre_hook(mk()))
    x, _ = corpus.get_batch("val", model.cfg.block_size, 16,
                            gen=torch.Generator().manual_seed(7))
    model(x)
    for h in handles:
        h.remove()
    return probes


def main():
    set_seed(30)
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192, block_size=256)
    for name, path in CKPTS.items():
        if not path.exists():
            raise SystemExit(f"missing checkpoint {name}: {path}")

    e1 = json.loads((REPO / "runs" / "e001" / "metrics.json").read_text(encoding="utf-8"))
    e12 = json.loads((REPO / "runs" / "e012" / "metrics.json").read_text(encoding="utf-8"))
    e12b = json.loads((REPO / "runs" / "e012b" / "metrics.json").read_text(encoding="utf-8"))
    e14b = json.loads((REPO / "runs" / "e014b" / "metrics.json").read_text(encoding="utf-8"))
    e11c = json.loads((REPO / "runs" / "e011c" / "metrics.json").read_text(encoding="utf-8"))

    # ---- load all four nets; val CE first (doubles as e014b.1 parity check)
    nets, vals = {}, {}
    for name in NAMES:
        m = TinyGPT(cfg).to(DEVICE)
        m.load_state_dict(torch.load(CKPTS[name], map_location=DEVICE, weights_only=True))
        m.eval()
        if RENORM[name]:
            register_renorm(m)  # stays active for every R/R43 eval below
        nets[name] = m
        vals[name] = estimate_loss(m, corpus, "val", n_batches=N_EVAL)
        ok = abs(vals[name] - REF_VALS[name]) <= 0.02
        log(f"{name:3s} (seed {'42' if '43' not in name else '43'} "
            f"{'renorm' if RENORM[name] else 'base  '}) val {vals[name]:.4f} "
            f"[ref {REF_VALS[name]:.4f}] {'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit(f"{name} val far from reference — wrong checkpoint or hooks")
    for name in ("R", "R43"):
        probes = renorm_liveness(nets[name], corpus)
        if not all(abs(p - 5.6) <= 1e-3 for p in probes):
            raise SystemExit(f"renorm liveness FAILED for {name}: {probes}")
    log("renorm liveness R / R43: block-input norms pinned to 5.6")

    # seed-12 position sampling identical to E012 (same fresh-generator draw)
    ix_gen = torch.Generator().manual_seed(12)
    ix = torch.randint(len(corpus.val) - cfg.block_size - 2, (N_POS,), generator=ix_gen)
    ys_cls = torch.tensor([class_id(corpus.itos[int(t)]) for t in corpus.val[ix + cfg.block_size]])

    # =====================================================================
    # TASK 1 — e012c: 4-net decision-depth histogram table
    # =====================================================================
    log("TASK 1 e012c: depth censuses on B43 (no hooks) + R43 (hooks)")
    rd1 = run_dir("e012c")
    hists = {"B": e12["depth_histogram"], "R": e12b["depth_hist_renorm"]}
    cens = {"B": {"spearman_depth_entropy": e12["spearman_depth_entropy"]},
            "R": {"spearman_depth_entropy": e12b["spearman_depth_entropy_renorm"]}}
    for name in ("B43", "R43"):
        depths, entrop = depth_census(nets[name], corpus)  # renorm hooks active for R43
        dec = depths >= 0
        hists[name] = torch.bincount(depths[dec], minlength=7).tolist()
        cens[name] = {
            "spearman_depth_entropy": spearman(depths[dec].float(), entrop[dec]),
            "l5_finalization": int(hists[name][6]),
            "class_mean_depth": {CLASSES[k]: round(float(depths[dec][ys_cls[dec] == k].float().mean()), 3)
                                 for k in range(5) if (ys_cls[dec] == k).any()},
        }
        log(f"  {name}: hist {hists[name]}  spearman(d,H) "
            f"{cens[name]['spearman_depth_entropy']:+.3f}  classes "
            f"{cens[name]['class_mean_depth']}")

    corr = {f"{a}|{b}": hist_corr(hists[a], hists[b]) for a in NAMES for b in NAMES}
    if abs(corr["B|R"] - 0.8218) > 0.01:
        raise SystemExit(f"self-check failed: corr(B,R) {corr['B|R']:.4f} != e012b 0.8218")
    cross_seed = {"B|B43": corr["B|B43"], "R|R43": corr["R|R43"]}          # same regime, diff seed
    cross_regime = {"B|R": corr["B|R"], "B43|R43": corr["B43|R43"]}        # same seed, diff regime
    diff_both = {"B|R43": corr["B|R43"], "B43|R": corr["B43|R"]}
    seed_bar_pass = min(cross_seed.values()) >= 0.8
    regime_bar_pass = corr["B43|R43"] >= 0.8
    claim1_init_independent = bool(seed_bar_pass and regime_bar_pass)
    ms_seed = statistics.mean(cross_seed.values())
    ms_regime = statistics.mean(cross_regime.values())
    clusters_by_seed = ms_regime - ms_seed
    log("  4x4 corr table: " + " | ".join(
        f"{k} {v:.3f}" for k, v in corr.items() if k.split("|")[0] < k.split("|")[1]))
    log(f"  cross-seed (same regime)   {cross_seed}  mean {ms_seed:.3f}")
    log(f"  cross-regime (same seed)   { {k: round(v, 3) for k, v in cross_regime.items()} }  mean {ms_regime:.3f}")
    log(f"  both-axes-different        { {k: round(v, 3) for k, v in diff_both.items()} }")
    log(f"  VERDICT e012c: seed-axis bar (>=0.8) {'PASS' if seed_bar_pass else 'FAIL'}; "
        f"regime-axis replication in seed43 {'PASS' if regime_bar_pass else 'FAIL'} -> "
        f"claim 1 {'INIT-INDEPENDENT (H stands)' if claim1_init_independent else 'INIT-BOUND / stays M'}; "
        f"seed-clustering delta {clusters_by_seed:+.3f}")

    save_json(rd1 / "metrics.json", {
        "experiment": "e012c_seed_deconfound", "n_positions": N_POS, "sampling": "seed-12, identical to e012",
        "nets": {n: {"seed": 42 if "43" not in n else 43, "regime": "renorm" if RENORM[n] else "base",
                     "ckpt": CKPTS[n].name, "val_ce": vals[n],
                     "depth_histogram": hists[n], **cens[n]} for n in NAMES},
        "hist_correlation_4x4": corr,
        "cross_seed_same_regime": cross_seed, "cross_regime_same_seed": cross_regime,
        "both_axes_different": diff_both,
        "mean_cross_seed": ms_seed, "mean_cross_regime": ms_regime,
        "seed_clustering_delta": clusters_by_seed,
        "registered_verdict": {
            "rule": "claim 1 init-independent iff min(corr(B,B43), corr(R,R43)) >= 0.8 "
                    "AND corr(B43,R43) >= 0.8 (e012b bar applied to both axes)",
            "seed_axis_pass": bool(seed_bar_pass), "regime_axis_pass_seed43": bool(regime_bar_pass),
            "claim1_init_independent": claim1_init_independent,
        },
    })
    fig, ax = plt.subplots(figsize=(9, 4.4))
    cols = {"B": "crimson", "B43": "darkorange", "R": "steelblue", "R43": "seagreen"}
    bins = range(7)
    for off, n in zip((-0.3, -0.1, 0.1, 0.3), NAMES):
        ax.bar([i + off for i in bins], hists[n], 0.2, label=f"{n} "
                f"(s{42 if '43' not in n else 43} {'renorm' if RENORM[n] else 'base'}, val {vals[n]:.2f})",
                color=cols[n])
    ax.set_xticks(list(bins)); ax.set_xticklabels(["emb", "L0", "L1", "L2", "L3", "L4", "L5"])
    ax.set_ylabel("positions"); ax.legend(fontsize=8)
    ax.set_title(f"E012c — decision-depth histograms x seed x regime "
                 f"(cross-seed {ms_seed:.2f} vs cross-regime {ms_regime:.2f})")
    fig.tight_layout(); fig.savefig(rd1 / "depth_hist_4nets.png", dpi=140); plt.close(fig)

    # =====================================================================
    # TASK 2 — e014b.1: replicate claim 2 on R43 (hooks active)
    # =====================================================================
    log("TASK 2 e014b.1: R43 lesion map + write/stream profile (30 batches)")
    rd2 = run_dir("e014b1")
    r43_val = vals["R43"]
    parity = bool(r43_val <= PARITY_GATE)
    log(f"  R43 val {r43_val:.4f} — parity gate <= {PARITY_GATE}: {'PASS' if parity else 'FAIL'}")
    attn43_ce, mlp43_ce = lesion_profile(nets["R43"], corpus, n_eval=N_EVAL)
    a43 = [round(v - r43_val, 4) for v in attn43_ce]
    m43 = [round(v - r43_val, 4) for v in mlp43_ce]
    prof43 = write_stream_profile(nets["R43"], corpus)
    log(f"  R43 attn damage {a43}")
    log(f"  R43 mlp  damage {m43}")
    log(f"  R43 write profile attn {prof43['attn']} mlp {prof43['mlp']} (in pinned 5.6)")

    b_attn, b_mlp = e1["attn_block_damage"], e1["mlp_block_damage"]
    r_attn, r_mlp = e14b["renorm_damage"]["attn"], e14b["renorm_damage"]["mlp"]
    keystone = m43[0]
    keystone_replicated = bool(keystone <= 0.5)
    late_heavy = bool(statistics.mean(m43[3:]) > statistics.mean(m43[:3]))
    front_load = bool(a43[0] - a43[5] >= 1.0)
    mlp_trend = spearman(torch.arange(6).float(), torch.tensor(m43).float())
    claim2_replicated = bool(parity and keystone_replicated and late_heavy and front_load)
    log(f"  keystone dissolution: R43 mlp-L0 {keystone:+.3f} [B {b_mlp[0]:+.2f}, R {r_mlp[0]:+.2f}] "
        f"<= 0.5: {'REPLICATED' if keystone_replicated else 'NOT replicated'}")
    log(f"  late-heavy MLP flip: mean L3-5 {statistics.mean(m43[3:]):.3f} vs L0-2 "
        f"{statistics.mean(m43[:3]):.3f} (spearman(layer, mlp dmg) {mlp_trend:+.2f}): "
        f"{'REPLICATED' if late_heavy else 'NOT replicated'}")
    log(f"  attn front-load: spread {a43[0] - a43[5]:.2f} [B {b_attn[0]-b_attn[5]:.2f}, "
        f"R {r_attn[0]-r_attn[5]:.2f}] >= 1.0: {'REPLICATED' if front_load else 'NOT replicated'}")
    log(f"  VERDICT e014b.1: claim 2 (plasticity) {'REPLICATED in seed 43' if claim2_replicated else 'NOT fully replicated'}")

    save_json(rd2 / "metrics.json", {
        "experiment": "e014b1_renorm_seed43_replication", "ckpt": CKPTS["R43"].name,
        "r43_val_ce": r43_val, "parity_gate": PARITY_GATE, "parity_pass": parity,
        "r43_damage": {"attn": a43, "mlp": m43},
        "r43_write_profile": prof43,
        "references": {"B_attn": b_attn, "B_mlp": b_mlp, "R_attn": r_attn, "R_mlp": r_mlp,
                       "B_val": REF_VALS["B"], "R_val": REF_VALS["R"], "B43_val": REF_VALS["B43"]},
        "registered_verdict": {
            "keystone_dissolution_le_0.5": keystone_replicated, "r43_mlp_L0": keystone,
            "late_heavy_mlp_flip": late_heavy, "mlp_layer_trend_spearman": mlp_trend,
            "attn_frontload_ge_1.0": front_load, "attn_spread": a43[0] - a43[5],
            "claim2_replicated": claim2_replicated,
        },
    })
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    xs = range(6)
    for ax, (bd, rd, nd), t in ((axes[0], (b_attn, r_attn, a43), "attention"),
                                (axes[1], (b_mlp, r_mlp, m43), "MLP")):
        ax.plot(xs, bd, "o-", color="crimson", label="B s42 (e001)")
        ax.plot(xs, rd, "s-", color="steelblue", label=f"R s42 (e014b, val {REF_VALS['R']:.2f})")
        ax.plot(xs, nd, "^-", color="seagreen", label=f"R43 s43 (val {r43_val:.2f})")
        ax.set_title(f"{t} zero-ablation damage"); ax.set_xlabel("layer")
        ax.set_xticks(list(xs)); ax.set_xticklabels([f"L{i}" for i in xs]); ax.legend(fontsize=8)
    axes[0].set_ylabel("Δ val CE (nats)")
    fig.suptitle("E014b.1 — claim 2 replication in a second renorm seed: "
                 f"{'REPLICATED' if claim2_replicated else 'NOT fully replicated'}")
    fig.tight_layout(); fig.savefig(rd2 / "r43_lesion_replication.png", dpi=140); plt.close(fig)

    # =====================================================================
    # TASK 3 — e011c_ci: bootstrap-style CIs on rotate/zero ratios
    # =====================================================================
    log("TASK 3 e011c_ci: fresh-seed reruns of attn-L0, mlp-L1, mlp-L5")
    rd3 = run_dir("e011c_ci")
    base = nets["B"]  # no hooks
    cells = [("attn", 0), ("mlp", 1), ("mlp", 5)]
    variants = [(1401, 7101), (1402, 7102), (1403, 7103), (1404, 7104), (1405, 7105)]
    base_ce = {bs: eval_ce(base, corpus, bs) for bs, _ in variants}
    base_ce[1337] = eval_ce(base, corpus, 1337)  # anchor protocol

    ci_results = {}
    for kind, layer in cells:
        mod = base.h[layer].attn if kind == "attn" else base.h[layer].mlp
        per_variant = []
        for bs, rs in variants:
            def zero_hook(module, args, out):
                return torch.zeros_like(out)
            h = mod.register_forward_hook(zero_hook)
            zce = eval_ce(base, corpus, bs)
            h.remove()
            g = torch.Generator(device=DEVICE).manual_seed(rs)
            def rot_hook(module, args, out, _g=g):
                return rotated_like(out.float(), _g).to(out.dtype)
            h = mod.register_forward_hook(rot_hook)
            rce = eval_ce(base, corpus, bs)
            h.remove()
            z, r = zce - base_ce[bs], rce - base_ce[bs]
            per_variant.append({"batch_seed": bs, "rot_seed": rs, "zero": round(z, 4),
                                "rotate": round(r, 4), "ratio": round(r / max(z, 1e-3), 3)})
        # exact-protocol anchor: E011c's own rotation seed + batch stream, e001 zero ref
        g0 = torch.Generator(device=DEVICE).manual_seed(2000 + layer * 2 + (kind == "mlp"))
        def rot0(module, args, out, _g=g0):
            return rotated_like(out.float(), _g).to(out.dtype)
        h = mod.register_forward_hook(rot0)
        r0 = eval_ce(base, corpus, 1337) - base_ce[1337]
        h.remove()
        anchor_ratio = r0 / e11c["zero_damage"][kind][layer]
        ratios = [v["ratio"] for v in per_variant]
        mu, sd = statistics.mean(ratios), statistics.stdev(ratios)
        beyond = bool(mu - 2 * sd > 1 or mu + 2 * sd < 1)
        tag = ("direction-sensitive (>1)" if mu - 2 * sd > 1
               else "energy-carrier (<1)" if mu + 2 * sd < 1 else "within noise of 1")
        ci_results[f"{kind}-L{layer}"] = {
            "variants": per_variant, "ratio_mean": round(mu, 3), "ratio_sd": round(sd, 3),
            "anchor_original_protocol": {"rotate": round(r0, 4), "ratio_vs_e001_zero": round(anchor_ratio, 3),
                                         "e011c_reported_ratio": e11c["rotate_over_zero"][kind][layer]},
            "beyond_noise_2sd": beyond, "reading": tag,
        }
        log(f"  {kind}-L{layer}: ratio {mu:.3f} +/- {sd:.3f} (anchor {anchor_ratio:.2f}, "
            f"e011c {e11c['rotate_over_zero'][kind][layer]:.2f}) -> {tag}")

    survivors = {k: v["beyond_noise_2sd"] for k, v in ci_results.items()}
    log(f"  VERDICT e011c-ci: beyond-noise cells: "
        f"{[k for k, v in survivors.items() if v] or 'NONE'}")
    save_json(rd3 / "metrics.json", {
        "experiment": "e011c_ci_bootstrap", "model": "B (e001, seed 42)",
        "protocol": "5 variants x (fresh batch seed, fresh rotation seed); zero and rotate "
                    "paired on identical batches; 20 batches each; beyond-noise iff "
                    "|mean-1| > 2sd",
        "cells": ci_results,
        "survive_ci": survivors,
        "e011c_reported_ratios": {f"{k}-L{l}": e11c["rotate_over_zero"][k][l] for k, l in cells},
    })
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    xs = np.arange(len(cells))
    mus = [ci_results[f"{k}-L{l}"]["ratio_mean"] for k, l in cells]
    sds = [ci_results[f"{k}-L{l}"]["ratio_sd"] for k, l in cells]
    anchors = [ci_results[f"{k}-L{l}"]["anchor_original_protocol"]["ratio_vs_e001_zero"] for k, l in cells]
    ax.bar(xs, mus, 0.45, yerr=sds, capsize=5, color="steelblue", alpha=0.85,
           label="mean +/- sd over 5 fresh-seed variants")
    ax.scatter(xs, anchors, marker="D", color="crimson", zorder=5, label="original-protocol anchor")
    ax.axhline(1.0, color="k", lw=1, ls="--", label="ratio = 1 (rotate == zero: pure energy)")
    for i, (k, l) in enumerate(cells):
        v = ci_results[f"{k}-L{l}"]
        ax.text(i, mus[i] + sds[i] + 0.08, v["reading"], ha="center", fontsize=8)
    ax.set_xticks(xs, [f"{k}-L{l}" for k, l in cells])
    ax.set_ylabel("rotate60 damage / zero damage")
    ax.set_title("E011c-ci — which rotate/zero ratios survive error bars?")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(rd3 / "ratio_ci.png", dpi=140); plt.close(fig)

    # =====================================================================
    # composite figure
    # =====================================================================
    rd0 = run_dir("e030")
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5))
    ax = axes[0][0]
    for off, n in zip((-0.3, -0.1, 0.1, 0.3), NAMES):
        ax.bar([i + off for i in bins], hists[n], 0.2, color=cols[n],
               label=f"{n} (s{42 if '43' not in n else 43} {'renorm' if RENORM[n] else 'base'})")
    ax.set_xticks(list(bins)); ax.set_xticklabels(["emb", "L0", "L1", "L2", "L3", "L4", "L5"])
    ax.set_ylabel("positions"); ax.legend(fontsize=8)
    ax.set_title(f"E012c depth histograms: cross-seed {ms_seed:.2f} vs cross-regime {ms_regime:.2f} "
                 f"-> claim 1 {'init-independent (H)' if claim1_init_independent else 'init-bound (M)'}")
    ax = axes[0][1]
    mat = np.array([[corr[f'{a}|{b}'] for b in NAMES] for a in NAMES])
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=11)
    ax.set_xticks(range(4), [f"{n}\ns{42 if '43' not in n else 43} {'R' if RENORM[n] else 'B'}" for n in NAMES], fontsize=8)
    ax.set_yticks(range(4), NAMES)
    ax.set_title("4-net histogram bin-correlation")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax = axes[1][0]
    lx = list(range(6))
    for bd, rd, nd, t in ((b_attn, r_attn, a43, "attn"), (b_mlp, r_mlp, m43, "mlp")):
        style = "-" if t == "attn" else "--"
        ax.plot(lx, bd, style + "o", color="crimson", label=f"B {t}")
        ax.plot(lx, rd, style + "s", color="steelblue", label=f"R {t}")
        ax.plot(lx, nd, style + "^", color="seagreen", label=f"R43 {t}")
    ax.set_xticks(lx); ax.set_xticklabels([f"L{i}" for i in lx])
    ax.set_xlabel("layer"); ax.set_ylabel("Δ val CE (nats)")
    ax.set_title(f"E014b.1 claim-2 replication on R43 (val {r43_val:.2f}): "
                 f"{'REPLICATED' if claim2_replicated else 'NOT fully replicated'} "
                 f"(keystone {keystone:+.2f}, late-heavy {late_heavy}, front-load {front_load})")
    ax.legend(fontsize=7, ncol=3)
    ax = axes[1][1]
    cx = np.arange(len(cells))
    ax.bar(cx, mus, 0.45, yerr=sds, capsize=5, color="steelblue", alpha=0.85,
           label="mean +/- sd (5 fresh-seed variants)")
    ax.scatter(cx, anchors, marker="D", color="crimson", zorder=5, label="original-protocol anchor")
    ax.axhline(1.0, color="k", lw=1, ls="--")
    for i, (k, l) in enumerate(cells):
        ax.text(i, mus[i] + sds[i] + 0.08, ci_results[f"{k}-L{l}"]["reading"], ha="center", fontsize=8)
    ax.set_xticks(cx, [f"{k}-L{l}" for k, l in cells])
    ax.set_ylabel("rotate60 / zero damage")
    ax.set_title(f"E011c-ci: beyond-noise cells = {[k for k, v in survivors.items() if v] or 'NONE'}")
    ax.legend(fontsize=8)
    fig.suptitle("E030 replication-debt slot: claim-1 de-confound | claim-2 seed replication | e011c error bars",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(rd0 / "composite_debt_slot.png", dpi=140)
    plt.close(fig)

    log("=" * 78)
    log(f"TASK1 e012c   : cross-seed { {k: round(v,3) for k,v in cross_seed.items()} } | "
        f"cross-regime { {k: round(v,3) for k,v in cross_regime.items()} } | "
        f"claim1 init-independent = {claim1_init_independent}")
    log(f"TASK2 e014b.1 : R43 val {r43_val:.4f} parity {parity} | keystone {keystone:+.3f} "
        f"| late-heavy {late_heavy} | front-load {front_load} | claim2 replicated = {claim2_replicated}")
    log(f"TASK3 e011c-ci: " + "; ".join(f"{k} {v['ratio_mean']:.2f}±{v['ratio_sd']:.2f} "
                                        f"({'SURVIVES' if v['beyond_noise_2sd'] else 'noise'})"
                                        for k, v in ci_results.items()))
    log(f"outputs: {rd1}  {rd2}  {rd3}  {rd0 / 'composite_debt_slot.png'}  "
        f"(total {time.time() - T0:.1f}s)")


if __name__ == "__main__":
    main()
