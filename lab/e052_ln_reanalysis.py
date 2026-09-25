"""E052 — INTERPRETER decisive reanalysis: LN-calibration vs basis-geometry.

Adversarial-audit follow-up on the e040 lineage (verdict P2-FROZEN). The audit
found (i) D correlates ~0.8 with own-organ load A across members, (ii) the
reverse-graft is worse for the winner (host-specific damage), and (iii) the
e031 alternative — damage tracks host-side LayerNorm calibration rather than
organ basis geometry — was never excluded. This experiment excludes it or
confirms it, CPU-only, zero GPU, no new training.

Members (all finals in runs/checkpoints/, 0.84M params, 4L/4H/128d):
  REF=e040_ref (donor), plus 10 hosts: w, m1..m3, g1a..g1c, g2a..g2c.
  Inits rebuilt bitwise exactly as e040 did (seeded construction + sigma 0.005
  eps on ".weight" keys), so dW-alignment is reproducible and cross-checkable
  against runs/e040/metrics.json.

Per host member (15 fixed val batches, e028/e040 protocol, batch gen seed
= corpus.seed — the SAME first 15 of e040's 30 batches):
  (a) graft damage D at L2/L3 (member <- REF mlp organ)
  (b) own-ablation A at L2/L3 (zero own mlp organ)
  (c) LN-distance to REF: L2 norm over the graft-site block's ln1/ln2
      weight+bias vectors (member vs REF)
  (d) dW-alignment to REF at graft sites: cos(trained-own_init flattened)
  (e) W-space distance: 1 - mean column-cosine of W_in (mlp.0) and W_out
      (mlp.2) vs REF at each graft site.

Regressions (n=10 hosts; REF excluded — it is the zero-distance origin):
  Pearson r + Spearman rho of D against each candidate; standardized 2-predictor
  OLS D ~ z(LN-dist) + z(A) with incremental R^2.

REGISTERED VERDICT (from the audit, frozen before running):
  r(D, LN-dist) >= r(D, A) >= 0.6        -> WRONG-TRAIT (assay selected LN
                                            calibration, not the basis; FROZEN
                                            collapses)
  W-space / alignment distance best      -> basis-geometry survives (weaker)
  nothing exceeds 0.5                    -> member-specific noise (single-
                                            lineage artifact)

Run: python lab/e052_ln_reanalysis.py   (CPU-only; CUDA masked before torch)
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # zero-GPU reanalysis

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as sstats

import common

common.DEVICE = "cpu"                              # estimate_loss/get_batch CPU
from common import (REPO, Cfg, CharCorpus, TinyGPT, lesion, run_dir,
                    save_json, set_seed)

# ---------------------------------------------------------------- protocol
CORPUS_PATH = REPO / "data" / "input.txt"
CKPT_DIR = REPO / "runs" / "checkpoints"
E040_METRICS = REPO / "runs" / "e040" / "metrics.json"
SITES = (2, 3)
N_EVAL = 15                                        # audit spec: 15 fixed batches
SIGMA_MUT = 0.005
SEED_W, SEED_REF = 42, 4304

MEMBERS = ["e040_w", "e040_m1", "e040_m2", "e040_m3",
           "e040_g1a", "e040_g1b", "e040_g1c",
           "e040_g2a", "e040_g2b", "e040_g2c"]

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- surgery (e040 verbatim, CPU)
def mlp_keys(i: int) -> list[str]:
    return [f"h.{i}.mlp.0.weight", f"h.{i}.mlp.0.bias",
            f"h.{i}.mlp.2.weight", f"h.{i}.mlp.2.bias"]


def ln_keys(i: int) -> list[str]:
    return [f"h.{i}.ln1.weight", f"h.{i}.ln1.bias",
            f"h.{i}.ln2.weight", f"h.{i}.ln2.bias"]


def snapshot(model) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def transplant(model, donor_sd: dict, keys: list[str]) -> None:
    sd = model.state_dict()
    for k in keys:
        assert sd[k].shape == donor_sd[k].shape, k
        sd[k] = donor_sd[k].detach().clone().to(sd[k].dtype)
    model.load_state_dict(sd, strict=True)


def assert_identical(model, snap: dict, name: str) -> None:
    now = model.state_dict()
    bad = [k for k in snap if not torch.equal(now[k], snap[k])]
    if bad:
        raise SystemExit(f"HOST {name} NOT BITWISE-RESTORED after a cell: {bad[:5]}")


@torch.no_grad()
def per_batch_losses_cpu(model, corpus, n_batches: int = N_EVAL) -> list[float]:
    """e028/e040 fixed-batch protocol: same RNG stream => first 15 of e040's 30."""
    model.eval()
    cfg = model.cfg
    gen = torch.Generator().manual_seed(corpus.seed)
    src = corpus.val
    losses = []
    for _ in range(n_batches):
        ix = torch.randint(len(src) - cfg.block_size - 1, (16,), generator=gen)
        x = torch.stack([src[i: i + cfg.block_size] for i in ix])
        y = torch.stack([src[i + 1: i + 1 + cfg.block_size] for i in ix])
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return losses


def mean(xs) -> float:
    return float(sum(xs) / len(xs))


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12))


def dw_vec(sd: dict, init: dict, keys: list[str]) -> torch.Tensor:
    return torch.cat([(sd[k].float() - init[k].float()).reshape(-1) for k in keys])


def vec(sd: dict, keys: list[str]) -> torch.Tensor:
    return torch.cat([sd[k].float().reshape(-1) for k in keys])


def mean_col_cos(A: torch.Tensor, B: torch.Tensor) -> float:
    """Mean column-cosine between two (R, C) matrices (column = shared basis dir)."""
    an = A / A.norm(dim=0, keepdim=True).clamp_min(1e-12)
    bn = B / B.norm(dim=0, keepdim=True).clamp_min(1e-12)
    return float((an * bn).sum(dim=0).mean())


# ---------------------------------------------------------------- init lineage (e040 verbatim)
def make_init(cfg, seed: int) -> dict:
    set_seed(seed)
    return snapshot(TinyGPT(cfg))                  # CPU construction: bitwise-stable


def mutated_init(base_sd: dict, seed: int, sigma: float) -> dict:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in base_sd.items():
        if k.endswith(".weight"):
            out[k] = v + torch.randn(v.shape, generator=g) * sigma
        else:
            out[k] = v.clone()
    return out


def build_inits(cfg) -> dict[str, dict]:
    ref = make_init(cfg, SEED_REF)
    w = make_init(cfg, SEED_W)
    m1 = mutated_init(w, 4021, SIGMA_MUT)
    m2 = mutated_init(w, 4022, SIGMA_MUT)
    m3 = mutated_init(w, 4023, SIGMA_MUT)
    g1a = mutated_init(m1, 4031, SIGMA_MUT)
    g1b = mutated_init(m1, 4032, SIGMA_MUT)
    g1c = mutated_init(m2, 4033, SIGMA_MUT)
    g2a = mutated_init(g1c, 4041, SIGMA_MUT)
    g2b = mutated_init(g1c, 4042, SIGMA_MUT)
    g2c = mutated_init(g1a, 4043, SIGMA_MUT)
    return {"e040_ref": ref, "e040_w": w, "e040_m1": m1, "e040_m2": m2,
            "e040_m3": m3, "e040_g1a": g1a, "e040_g1b": g1b, "e040_g1c": g1c,
            "e040_g2a": g2a, "e040_g2b": g2b, "e040_g2c": g2c}


# ---------------------------------------------------------------- stats helpers
def pearson(x, y):
    r, p = sstats.pearsonr(x, y)
    return float(r), float(p)


def spearman(x, y):
    r, p = sstats.spearmanr(x, y)
    return float(r), float(p)


def zscore(v):
    v = np.asarray(v, dtype=np.float64)
    return (v - v.mean()) / (v.std(ddof=0) if v.std(ddof=0) > 0 else 1.0)


def two_predictor_ols(y, x1, x2):
    """Standardized OLS y ~ z(x1) + z(x2): betas, R^2, incremental R^2 each way."""
    y = zscore(y)
    Z = np.column_stack([zscore(x1), zscore(x2)])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    r2_full = float(1 - ((y - Z @ beta) ** 2).sum() / (y ** 2).sum())
    out = {"beta_ln": float(beta[0]), "beta_a": float(beta[1]), "r2_full": r2_full}
    for tag, j in (("ln", 0), ("a", 1)):
        b1, *_ = np.linalg.lstsq(Z[:, [j]], y, rcond=None)
        r2_s = float(1 - ((y - Z[:, [j]] @ b1) ** 2).sum() / (y ** 2).sum())
        out[f"r2_{tag}_alone"] = r2_s
        out[f"delta_r2_adding_{tag}"] = r2_full - r2_s
    return out


# ---------------------------------------------------------------- main
def main():
    rd = run_dir("e052")
    log("E052 LN-calibration vs basis-geometry reanalysis of the e040 lineage "
        "(CPU-only, zero-GPU)")
    corpus = CharCorpus(CORPUS_PATH, seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=4, n_head=4, n_embd=128,
              block_size=256)
    net = TinyGPT(cfg)
    n_params = net.num_params()
    assert n_params == 840_704, n_params
    log(f"cfg e005s SMALL verbatim: 4L/4H/128d — {n_params} params; "
        f"N_EVAL={N_EVAL} fixed val batches (= first {N_EVAL} of e040's 30)")

    inits = build_inits(cfg)
    trained = {name: torch.load(CKPT_DIR / f"{name}.pt", map_location="cpu",
                                weights_only=True)
               for name in ["e040_ref"] + MEMBERS}
    REF_SD = trained["e040_ref"]
    REF_DW = {s: dw_vec(REF_SD, inits["e040_ref"], mlp_keys(s)) for s in SITES}
    log("inits rebuilt (seeded construction + sigma-0.005 eps) and 11 finals loaded")

    # ---- REF self-consistency (instrument gate, e040 G1 analogue) -----------
    net.load_state_dict(REF_SD)
    snap = snapshot(net)
    base_ref = per_batch_losses_cpu(net, corpus)
    transplant(net, REF_SD, mlp_keys(SITES[0]))
    self_losses = per_batch_losses_cpu(net, corpus)
    net.load_state_dict(snap)
    assert_identical(net, snap, "e040_ref")
    ref_self_graft_exact = bool(self_losses == base_ref)
    ref_ablate = {}
    for s in SITES:
        with lesion(net, "mlp", s):
            abl = per_batch_losses_cpu(net, corpus)
        ref_ablate[s] = mean([a - b for a, b in zip(abl, base_ref)])
    assert_identical(net, snap, "e040_ref")
    ref_base_ce = mean(base_ref)
    log(f"REF: base CE {ref_base_ce:.4f} (e040: 1.5350) | self-graft dCE exactly "
        f"0.0: {ref_self_graft_exact} | own-ablate "
        + " ".join(f"L{s} {ref_ablate[s]:+.4f}" for s in SITES))

    # ---- per-member assays -------------------------------------------------
    rows = []
    for name in MEMBERS:
        sd = trained[name]
        net.load_state_dict(sd)
        snap = snapshot(net)
        base = per_batch_losses_cpu(net, corpus)
        rec = {"name": name, "base_ce": mean(base)}

        rec["graft"], rec["ablate"] = {}, {}
        for site in SITES:
            transplant(net, REF_SD, mlp_keys(site))     # member <- REF organ
            losses = per_batch_losses_cpu(net, corpus)
            net.load_state_dict(snap)
            assert_identical(net, snap, name)
            diffs = [a - b for a, b in zip(losses, base)]
            rec["graft"][site] = {"dce": mean(diffs), "per_batch": diffs}
            with lesion(net, "mlp", site):              # zero own organ
                abl = per_batch_losses_cpu(net, corpus)
            adiff = [a - b for a, b in zip(abl, base)]
            assert_identical(net, snap, name)
            rec["ablate"][site] = mean(adiff)

        # (c) LN-distance to REF at graft-site blocks
        rec["ln_dist"] = {s: float((vec(sd, ln_keys(s)) - vec(REF_SD, ln_keys(s)))
                                   .norm()) for s in SITES}
        # (d) dW-alignment to REF at graft organs
        rec["dw_cos"] = {s: cos(dw_vec(sd, inits[name], mlp_keys(s)), REF_DW[s])
                         for s in SITES}
        # (e) W-space distance: 1 - mean column-cos vs REF (W_in, W_out per site)
        rec["col_cos"] = {s: {"w_in": mean_col_cos(sd[f"h.{s}.mlp.0.weight"],
                                                   REF_SD[f"h.{s}.mlp.0.weight"]),
                              "w_out": mean_col_cos(sd[f"h.{s}.mlp.2.weight"],
                                                    REF_SD[f"h.{s}.mlp.2.weight"])}
                          for s in SITES}

        rec["D"] = mean([rec["graft"][s]["dce"] for s in SITES])
        rec["A"] = mean([rec["ablate"][s] for s in SITES])
        rec["LN_dist"] = mean([rec["ln_dist"][s] for s in SITES])
        rec["align_dist"] = 1.0 - mean([rec["dw_cos"][s] for s in SITES])
        rec["wspace_dist"] = 1.0 - mean([rec["col_cos"][s][m]
                                         for s in SITES for m in ("w_in", "w_out")])
        rows.append(rec)
        log(f"  {name:9s} D {rec['D']:+.4f} | A {rec['A']:+.4f} | "
            f"LN-dist {rec['LN_dist']:.4f} | align {rec['align_dist']:+.4f} | "
            f"W-space {rec['wspace_dist']:.4f} | base CE {rec['base_ce']:.4f}")

    # ---- reproduction cross-check vs runs/e040/metrics.json ----------------
    e040 = json.loads(E040_METRICS.read_text(encoding="utf-8"))
    e040_by_name = {m["name"]: m for g in e040["members"].values() for m in g}
    repro = {"graft_per_batch_max_abs_diff": 0.0, "ablate_per_batch_max_abs_diff": 0.0,
             "dw_cos_max_abs_diff": 0.0, "n_members_checked": 0,
             "my_D": [], "e040_D": [], "my_A": [], "e040_A": []}
    for rec in rows:
        m = e040_by_name[rec["name"]]
        for s in SITES:
            mine = rec["graft"][s]["per_batch"]
            theirs = m["graft"][str(s)]["per_batch"][:N_EVAL]
            repro["graft_per_batch_max_abs_diff"] = max(
                repro["graft_per_batch_max_abs_diff"],
                max(abs(a - b) for a, b in zip(mine, theirs)))
            repro["dw_cos_max_abs_diff"] = max(
                repro["dw_cos_max_abs_diff"], abs(rec["dw_cos"][s] - m["dw_cos"][str(s)]))
        repro["my_D"].append(rec["D"])
        repro["e040_D"].append(m["D"])
        repro["my_A"].append(rec["A"])
        repro["e040_A"].append(mean([m["A"][str(s)] for s in SITES]))
        repro["n_members_checked"] += 1
    r_repro_d, _ = pearson(repro["my_D"], repro["e040_D"])
    repro["r_myD_vs_e040D"] = r_repro_d
    repro["max_abs_D_offset_15v30_batches"] = max(abs(a - b) for a, b in
                                                  zip(repro["my_D"], repro["e040_D"]))
    log(f"reproduction vs e040 (30-batch): graft per-batch max |diff| "
        f"{repro['graft_per_batch_max_abs_diff']:.2e} | dw_cos max |diff| "
        f"{repro['dw_cos_max_abs_diff']:.2e} | r(D_15, D_30) {r_repro_d:.4f}")

    # ---- regressions --------------------------------------------------------
    D = np.array([r["D"] for r in rows])
    preds = {"ln_dist": np.array([r["LN_dist"] for r in rows]),
             "A": np.array([r["A"] for r in rows]),
             "wspace_dist": np.array([r["wspace_dist"] for r in rows]),
             "align_dist": np.array([r["align_dist"] for r in rows])}
    table = {}
    for k, v in preds.items():
        pr, pp = pearson(D, v)
        sr, sp = spearman(D, v)
        table[k] = {"pearson_r": pr, "pearson_p": pp,
                    "spearman_rho": sr, "spearman_p": sp}
    ols = two_predictor_ols(D, preds["ln_dist"], preds["A"])
    r_lnA, p_lnA = pearson(preds["ln_dist"], preds["A"])
    log("correlations with D (n=10 hosts): " + " | ".join(
        f"{k}: r={v['pearson_r']:+.3f} (p={v['pearson_p']:.3f}) "
        f"rho={v['spearman_rho']:+.3f}" for k, v in table.items()))
    log(f"collinearity r(LN-dist, A) = {r_lnA:+.3f} | OLS z-scores: "
        f"beta_LN {ols['beta_ln']:+.3f}, beta_A {ols['beta_a']:+.3f}, "
        f"R2 {ols['r2_full']:.3f}")

    # ---- registered verdict --------------------------------------------------
    r_ln = table["ln_dist"]["pearson_r"]
    r_a = table["A"]["pearson_r"]
    basis_best = max(table["wspace_dist"]["pearson_r"], table["align_dist"]["pearson_r"])
    all_max = max(abs(v["pearson_r"]) for v in table.values())
    c1 = bool(r_ln >= r_a >= 0.6)
    c2 = bool(basis_best > max(r_ln, r_a) and basis_best >= 0.5)
    c3 = bool(all_max < 0.5)
    if c1:
        verdict = ("WRONG-TRAIT — damage tracks host LN calibration (r(D,LN-dist) "
                   f"{r_ln:+.3f} >= r(D,A) {r_a:+.3f} >= 0.6); the e040 assay "
                   "selected LN calibration, not organ basis geometry; FROZEN "
                   "collapses")
    elif c2:
        verdict = ("BASIS-GEOMETRY (weaker survival) — W-space/alignment distance "
                   f"correlates best (max r {basis_best:+.3f}); the geometry "
                   "reading survives, weakened by LN/organ-load correlation")
    elif c3:
        verdict = ("NOISE — no candidate reaches |r| 0.5 (max "
                   f"{all_max:.3f}); the e040 trickle is member-specific noise, "
                   "a single-lineage artifact")
    else:
        extra = ""
        if r_a >= 0.6 and r_a > r_ln:
            extra = (" (note: A alone >= 0.6 and beats LN-dist — damage tracks "
                     "own-organ load, not LN calibration)")
        verdict = ("MIXED/INCONCLUSIVE — registered bars not met: r(D,LN-dist) "
                   f"{r_ln:+.3f}, r(D,A) {r_a:+.3f}, best basis-geometry "
                   f"{basis_best:+.3f}{extra}")
    log("=" * 78)
    log(f"VERDICT: {verdict}")

    # ---- outputs -------------------------------------------------------------
    def strip(rec):
        out = dict(rec)
        for s in SITES:
            out["graft"][s] = {"dce": rec["graft"][s]["dce"],
                               "per_batch": rec["graft"][s]["per_batch"]}
        return out

    metrics = {
        "experiment": "e052_ln_reanalysis",
        "date": common.now_iso(),
        "status": "complete",
        "device": "cpu-only (CUDA_VISIBLE_DEVICES masked; zero-GPU reanalysis)",
        "config": common.cfg_dict(cfg),
        "params": 840_704,
        "protocol": {"sites": list(SITES), "n_eval_batches": N_EVAL,
                     "batch_note": ("first 15 of e040's fixed 30 val batches "
                                    "(same RNG stream, seed=corpus.seed)"),
                     "predictors": {
                         "ln_dist": "mean over sites of L2(member-REF) over "
                                    "[ln1.w, ln1.b, ln2.w, ln2.b] at graft blocks",
                         "A": "mean over sites of own-mlp-ablation dCE",
                         "wspace_dist": "mean over sites/matrices of 1 - mean "
                                        "column-cosine (W_in=mlp.0, W_out=mlp.2)",
                         "align_dist": "1 - mean over sites of cos(dW_member, "
                                       "dW_REF) at graft organs"},
                     "n_hosts": len(rows),
                     "ref_excluded_reason": "zero-distance origin (donor)"},
        "ref": {"name": "e040_ref", "base_ce": ref_base_ce,
                "e040_base_ce": e040["ref"]["val_ce"],
                "self_graft_dce_exactly_zero": ref_self_graft_exact,
                "own_ablate": {str(s): ref_ablate[s] for s in SITES}},
        "instrument_gates": {
            "ref_self_transplant_zero": ref_self_graft_exact,
            "hosts_bitwise_restored": True,
            "reproduction_vs_e040": {k: v for k, v in repro.items()
                                     if not isinstance(v, list)},
            "per_member_D_15batch": repro["my_D"],
            "per_member_D_e040_30batch": repro["e040_D"],
            "per_member_A_15batch": repro["my_A"],
            "per_member_A_e040": repro["e040_A"]},
        "members": [strip(r) for r in rows],
        "correlations_with_D": table,
        "collinearity_ln_vs_A": {"pearson_r": r_lnA, "pearson_p": float(p_lnA)},
        "two_predictor_ols_D_on_ln_and_A": ols,
        "per_site_secondary": {
            str(s): {"pearson_Dsite_ln": pearson([r["graft"][s]["dce"] for r in rows],
                                                 [r["ln_dist"][s] for r in rows]),
                     "pearson_Dsite_A": pearson([r["graft"][s]["dce"] for r in rows],
                                                [r["ablate"][s] for r in rows]),
                     "pearson_Dsite_wspace": pearson(
                         [r["graft"][s]["dce"] for r in rows],
                         [1 - mean([r["col_cos"][s][m] for m in ("w_in", "w_out")])
                          for r in rows]),
                     "pearson_Dsite_align": pearson(
                         [r["graft"][s]["dce"] for r in rows],
                         [1 - r["dw_cos"][s] for r in rows])}
            for s in SITES},
        "registered_verdict_rules": {
            "wrong_trait": "r(D,LN-dist) >= r(D,A) >= 0.6",
            "basis_geometry_weaker": "wspace/align best and >= 0.5",
            "noise": "nothing exceeds 0.5"},
        "conditions": {"r_D_ln": r_ln, "r_D_A": r_a,
                       "r_D_wspace": table["wspace_dist"]["pearson_r"],
                       "r_D_align": table["align_dist"]["pearson_r"],
                       "basis_best": basis_best, "all_max_abs_r": all_max,
                       "c1_wrong_trait": c1, "c2_basis_geometry": c2,
                       "c3_noise": c3},
        "verdict": verdict,
        "timing_s": round(time.time() - T0, 1),
    }
    save_json(rd / "metrics.json", metrics)

    # ---- scatter figure ------------------------------------------------------
    names = [r["name"].replace("e040_", "") for r in rows]
    panels = [("LN-dist to REF (graft blocks)", preds["ln_dist"], table["ln_dist"]),
              ("own-ablation load A", preds["A"], table["A"]),
              ("W-space distance (1 - col-cos)", preds["wspace_dist"],
               table["wspace_dist"]),
              ("dW-alignment distance (1 - cos)", preds["align_dist"],
               table["align_dist"])]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    axes = axes.ravel()
    for ax, (lab, v, st) in zip(axes, panels):
        ax.scatter(v, D, color="#2c6fbb", s=46, zorder=3)
        for x, y, nm in zip(v, D, names):
            ax.annotate(nm, (x, y), fontsize=7, alpha=0.75,
                        xytext=(3, 3), textcoords="offset points")
        if np.std(v) > 0:
            b1, b0 = np.polyfit(v, D, 1)
            xs = np.linspace(v.min(), v.max(), 50)
            ax.plot(xs, b0 + b1 * xs, color="#c0392b", lw=1.5, ls="--", zorder=2)
        ax.set_xlabel(lab)
        ax.set_ylabel("D = graft damage (member <- REF mlp L2/L3, 15 batches)")
        ax.set_title(f"r = {st['pearson_r']:+.3f} (p={st['pearson_p']:.3f}) | "
                     f"rho = {st['spearman_rho']:+.3f}")
        ax.grid(alpha=0.25)
    ax = axes[4]
    ax.scatter(repro["my_D"], repro["e040_D"], color="#2c6fbb", s=46, zorder=3)
    for x, y, nm in zip(repro["my_D"], repro["e040_D"], names):
        ax.annotate(nm, (x, y), fontsize=7, alpha=0.75,
                    xytext=(3, 3), textcoords="offset points")
    lo, hi = min(repro["my_D"] + repro["e040_D"]), max(repro["my_D"] + repro["e040_D"])
    ax.plot([lo, hi], [lo, hi], color="#888", lw=1, ls=":")
    ax.set_xlabel("D reproduced here (15 batches)")
    ax.set_ylabel("D from runs/e040/metrics.json (30 batches)")
    ax.set_title(f"reproduction r = {r_repro_d:+.4f}")
    ax.grid(alpha=0.25)
    ax = axes[5]
    ax.scatter(preds["ln_dist"], preds["A"], color="#7b3fa0", s=46, zorder=3)
    for x, y, nm in zip(preds["ln_dist"], preds["A"], names):
        ax.annotate(nm, (x, y), fontsize=7, alpha=0.75,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("LN-dist to REF")
    ax.set_ylabel("own-ablation load A")
    ax.set_title(f"collinearity of the two host-side candidates "
                 f"(r = {r_lnA:+.3f})")
    ax.grid(alpha=0.25)
    fig.suptitle("E052 — does e040 graft damage track host LN calibration or organ "
                 "basis geometry? (10 hosts; REF excluded as donor origin)\n"
                 f"VERDICT: {verdict.split(' — ')[0]}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(rd / "ln_vs_geometry_scatter.png", dpi=140)
    plt.close(fig)
    log(f"outputs: {rd} (metrics.json, ln_vs_geometry_scatter.png)")
    return metrics


if __name__ == "__main__":
    main()
