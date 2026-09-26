"""E059 — WINNER DIFFERENCING: what did e040's selection actually change?

Lineage chain (T024 -> T026 -> T027): e040 selected hosts on R = graft/own-ablation
(member <- REF mlp organs at L2/L3) for two generations; graft damage trickled
-5.6% with ZERO donor-alignment drift; T026 (e052) showed cross-member damage
tracks OWN-ORGAN LOAD A (r = +0.807) and killed the LN-confound reading at its
registered bar; T027 (e050) showed directed mutation doesn't help either. OPEN
QUESTION: which parameters actually MOVED in the selected winners (and their
children) relative to their UNSELECTED siblings and the wildtype — i.e., is
there a winner signature at all, and does it explain the damage trickle?

Groups (from runs/e040/metrics.json selection_events; ranking index = R_mean):
  WINNERS (selected parents + their children):
      g1a, g1c (selected at event 2) + g2a, g2b, g2c (children of g1c/g1a)
  UNSELECTED: w (wildtype), m3 (rejected at event 1), g1b (rejected at event 2)
  Per-event secondary contrasts:
      E1: winners {m1, m2} vs unselected {w, m3}
      E2: winners {g1a, g1c} vs unselected {g1b}

Audit families (per member, vs own init and vs REF):
  LN gains (ln1/ln2/ln_f weight+bias means), W_in/W_out row norms, effective
  ranks (erank = exp(entropy of s^2 spectrum)), organ-band ratios
  (R_site = graft dCE / ablate dCE), plus distance-to-REF forms of each.

REGISTERED VERDICT (frozen in this header BEFORE running; CPU-only, zero GPU):
  H-load     — winners reduced organ reliance: primary contrast (WINNERS vs
               UNSELECTED) on A has bootstrap 95% CI excluding 0 with winners
               LOWER (A fell), r(D, A) >= 0.6 AND the strongest |r| in the
               correlation table, and E1/E2 contrasts agree in sign.
  H-LN       — winners recalibrated LN statistics toward REF: LN-dist(graft
               sites) primary contrast CI excludes 0 with winners CLOSER to REF
               (smaller), r(D, LN-dist) > r(D, A), events agree in sign.
  H-interface— winners changed interface geometry: some W_in/W_out row-scale or
               effective-rank axis (incl. distance-to-REF forms) has primary
               contrast CI excluding 0, |r(D, axis)| >= |r(D, A)|, events agree.
  H-nothing  — none of the above fires, OR firing axes disagree in sign across
               E1/E2 (inconsistent => single-lineage artifact).
  Multiple firing with consistent events -> declare the largest standardized
  primary-contrast |effect| and log a MIXED note honestly.
  (Correlations are shift-invariant, so r over all 10 members IS the
  Delta-vs-wildtype correlation; the A-residualized D column implements T026's
  registered "regress D on A, read the residual" prescription.)

Run: python lab/e059_winner_diff.py   (CPU-only; CUDA masked before torch)
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # zero-GPU analysis

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as sstats

import common

common.DEVICE = "cpu"                              # eval strictly on CPU
from common import (REPO, Cfg, CharCorpus, TinyGPT, lesion, run_dir,
                    save_json, set_seed)

# ---------------------------------------------------------------- protocol
CORPUS_PATH = REPO / "data" / "input.txt"
CKPT_DIR = REPO / "runs" / "checkpoints"
E052_METRICS = REPO / "runs" / "e052" / "metrics.json"
SITES = (2, 3)
N_EVAL = 15                                        # e052 fixed-batch protocol
SIGMA_MUT = 0.005
SEED_W, SEED_REF = 42, 4304
N_BOOT = 10000

MEMBERS = ["e040_w", "e040_m1", "e040_m2", "e040_m3",
           "e040_g1a", "e040_g1b", "e040_g1c",
           "e040_g2a", "e040_g2b", "e040_g2c"]
WINNERS = ["e040_g1a", "e040_g1c", "e040_g2a", "e040_g2b", "e040_g2c"]
UNSELECTED = ["e040_w", "e040_m3", "e040_g1b"]
EVENT1 = (["e040_m1", "e040_m2"], ["e040_w", "e040_m3"])
EVENT2 = (["e040_g1a", "e040_g1c"], ["e040_g1b"])

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- surgery (e052 verbatim, CPU)
def mlp_keys(i: int) -> list[str]:
    return [f"h.{i}.mlp.0.weight", f"h.{i}.mlp.0.bias",
            f"h.{i}.mlp.2.weight", f"h.{i}.mlp.2.bias"]


def ln_keys(i: int) -> list[str]:
    return [f"h.{i}.ln1.weight", f"h.{i}.ln1.bias",
            f"h.{i}.ln2.weight", f"h.{i}.ln2.bias"]


def all_ln_keys(cfg) -> list[str]:
    ks = [k for i in range(cfg.n_layer) for k in ln_keys(i)]
    ks += ["ln_f.weight", "ln_f.bias"]
    return ks


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
    """e028/e040/e052 fixed-batch protocol: same RNG stream as e040's first 15."""
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


def erank(M: torch.Tensor) -> float:
    """Effective rank = exp(Shannon entropy of the normalized s^2 spectrum)."""
    s = torch.linalg.svdvals(M.float())
    p = s * s
    p = p / p.sum()
    return float(torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum()))


def profile_shape_dist(rows_m: torch.Tensor, rows_ref: torch.Tensor) -> float:
    """L2 distance between mean-normalized row-norm profiles (shape, not scale)."""
    a = rows_m / rows_m.mean().clamp_min(1e-12)
    b = rows_ref / rows_ref.mean().clamp_min(1e-12)
    return float((a - b).norm())


def boot_contrast(win_vals, un_vals, n: int = N_BOOT, seed: int = 59):
    """Bootstrap 95% CI of mean(winners) - mean(unselected), resampling members."""
    rng = np.random.default_rng(seed)
    w = np.asarray(win_vals, dtype=np.float64)
    u = np.asarray(un_vals, dtype=np.float64)
    wi = rng.integers(0, len(w), size=(n, len(w)))
    ui = rng.integers(0, len(u), size=(n, len(u)))
    diffs = w[wi].mean(axis=1) - u[ui].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    point = float(w.mean() - u.mean())
    pooled_sd = float(np.std(np.concatenate([w, u]), ddof=1))
    std_eff = point / pooled_sd if pooled_sd > 0 else 0.0
    return {"diff": point, "ci95": [float(lo), float(hi)],
            "pooled_sd": pooled_sd,
            "std_effect": float(std_eff), "ci_excludes_0": bool(lo > 0 or hi < 0),
            "n_win": len(w), "n_un": len(u)}


# ---------------------------------------------------------------- family audit (state-dict only)
def family_axes(sd: dict, ref_sd: dict, cfg) -> dict:
    """All registered parameter axes for one (trained or init) state dict."""
    ax: dict = {}
    # LN gains per block + final
    for i in range(cfg.n_layer):
        for ln in ("ln1", "ln2"):
            w = sd[f"h.{i}.{ln}.weight"].float()
            b = sd[f"h.{i}.{ln}.bias"].float()
            ax[f"L{i}_{ln}_gain_mean"] = float(w.mean())
            ax[f"L{i}_{ln}_gain_std"] = float(w.std())
            ax[f"L{i}_{ln}_bias_l2"] = float(b.norm())
    ax["lnf_gain_mean"] = float(sd["ln_f.weight"].float().mean())
    # per-matrix row norms + eranks, every block (W_in=mlp.0, W_out=mlp.2,
    # c_attn / c_proj secondary)
    for i in range(cfg.n_layer):
        w_in = sd[f"h.{i}.mlp.0.weight"].float()      # (4d, d)
        w_out = sd[f"h.{i}.mlp.2.weight"].float()     # (d, 4d)
        r_in = w_in.norm(dim=1)
        r_out = w_out.norm(dim=1)
        ax[f"L{i}_win_rowmean"] = float(r_in.mean())
        ax[f"L{i}_win_rowstd"] = float(r_in.std())
        ax[f"L{i}_wout_rowmean"] = float(r_out.mean())
        ax[f"L{i}_wout_rowstd"] = float(r_out.std())
        ax[f"L{i}_win_erank"] = erank(w_in)
        ax[f"L{i}_wout_erank"] = erank(w_out)
        ax[f"L{i}_cattn_erank"] = erank(sd[f"h.{i}.attn.c_attn.weight"].float())
        ax[f"L{i}_cproj_erank"] = erank(sd[f"h.{i}.attn.c_proj.weight"].float())
        # distance-to-REF forms (only meaningful for trained sds; for inits they
        # are computed too but unused)
        ri_ref = ref_sd[f"h.{i}.mlp.0.weight"].float().norm(dim=1)
        ro_ref = ref_sd[f"h.{i}.mlp.2.weight"].float().norm(dim=1)
        ax[f"L{i}_win_shape_dist"] = profile_shape_dist(r_in, ri_ref)
        ax[f"L{i}_wout_shape_dist"] = profile_shape_dist(r_out, ro_ref)
        ax[f"L{i}_win_erank_dist"] = abs(ax[f"L{i}_win_erank"]
                                          - erank(ref_sd[f"h.{i}.mlp.0.weight"]))
        ax[f"L{i}_wout_erank_dist"] = abs(ax[f"L{i}_wout_erank"]
                                           - erank(ref_sd[f"h.{i}.mlp.2.weight"]))
        ax[f"L{i}_ln_dist"] = float((vec(sd, ln_keys(i)) - vec(ref_sd, ln_keys(i))).norm())
    ax["ln_dist_all"] = float((vec(sd, all_ln_keys(cfg)) - vec(ref_sd, all_ln_keys(cfg))).norm())
    # graft-site aggregates (the registered hypothesis axes)
    ax["ln_dist_sites"] = mean([ax[f"L{s}_ln_dist"] for s in SITES])
    ax["win_rowmean_sites"] = mean([ax[f"L{s}_win_rowmean"] for s in SITES])
    ax["wout_rowmean_sites"] = mean([ax[f"L{s}_wout_rowmean"] for s in SITES])
    ax["win_erank_sites"] = mean([ax[f"L{s}_win_erank"] for s in SITES])
    ax["wout_erank_sites"] = mean([ax[f"L{s}_wout_erank"] for s in SITES])
    ax["win_shape_dist_sites"] = mean([ax[f"L{s}_win_shape_dist"] for s in SITES])
    ax["wout_shape_dist_sites"] = mean([ax[f"L{s}_wout_shape_dist"] for s in SITES])
    ax["win_erank_dist_sites"] = mean([ax[f"L{i}_win_erank_dist"] for i in range(cfg.n_layer)])
    ax["wout_erank_dist_sites"] = mean([ax[f"L{i}_wout_erank_dist"] for i in range(cfg.n_layer)])
    # NOTE: erank_dist aggregated over ALL blocks (whole-organ interface), while
    # row/erank levels aggregate over graft SITES (the violent interface).
    return ax


# ---------------------------------------------------------------- main
def main():
    rd = run_dir("e059")
    log("E059 winner differencing — what moved in e040's winners? (CPU-only)")
    corpus = CharCorpus(CORPUS_PATH, seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=4, n_head=4, n_embd=128,
              block_size=256)
    net = TinyGPT(cfg)
    assert net.num_params() == 840_704, net.num_params()
    log(f"cfg e040 verbatim: 4L/4H/128d — {net.num_params()} params; "
        f"N_EVAL={N_EVAL} fixed val batches (e052 protocol)")

    inits = build_inits(cfg)
    trained = {name: torch.load(CKPT_DIR / f"{name}.pt", map_location="cpu",
                                weights_only=True)
               for name in ["e040_ref"] + MEMBERS}
    REF_SD = trained["e040_ref"]
    REF_DW = {s: dw_vec(REF_SD, inits["e040_ref"], mlp_keys(s)) for s in SITES}
    log("inits rebuilt (seeded construction + sigma-0.005 eps) and 11 finals loaded")

    ref_ax = family_axes(REF_SD, REF_SD, cfg)       # self-distance = 0 by construction
    ref_ax = {k: v for k, v in ref_ax.items() if not k.endswith(
        ("_dist", "_dist_sites", "_dist_all"))}

    # ---- REF self-consistency (instrument gate, e040 G1/e052 analogue) ---------
    net.load_state_dict(REF_SD)
    snap = snapshot(net)
    base_ref = per_batch_losses_cpu(net, corpus)
    transplant(net, REF_SD, mlp_keys(SITES[0]))
    self_losses = per_batch_losses_cpu(net, corpus)
    net.load_state_dict(snap)
    assert_identical(net, snap, "e040_ref")
    ref_self_graft_exact = bool(self_losses == base_ref)
    log(f"REF: base CE {mean(base_ref):.4f} (e052: 1.5441) | self-graft dCE "
        f"exactly 0.0: {ref_self_graft_exact}")

    # ---- per-member: forward assays + family axes -----------------------------
    ablate_w: dict[int, float] | None = None       # e040 organ-band guard base
    rows = []
    for name in MEMBERS:
        sd = trained[name]
        net.load_state_dict(sd)
        snap = snapshot(net)
        base = per_batch_losses_cpu(net, corpus)
        rec: dict = {"name": name, "base_ce": mean(base)}

        rec["graft"], rec["ablate"] = {}, {}
        for site in SITES:
            transplant(net, REF_SD, mlp_keys(site))
            losses = per_batch_losses_cpu(net, corpus)
            net.load_state_dict(snap)
            assert_identical(net, snap, name)
            rec["graft"][site] = mean([a - b for a, b in zip(losses, base)])
            with lesion(net, "mlp", site):
                abl = per_batch_losses_cpu(net, corpus)
            assert_identical(net, snap, name)
            rec["ablate"][site] = mean([a - b for a, b in zip(abl, base)])

        rec["D"] = mean([rec["graft"][s] for s in SITES])
        rec["A"] = mean([rec["ablate"][s] for s in SITES])
        rec["R_sites"] = {s: rec["graft"][s] / rec["ablate"][s] for s in SITES}
        rec["R_mean"] = mean([rec["R_sites"][s] for s in SITES])
        # e040 organ-band guard verbatim: A within [0.5, 2.0] x wildtype's A/site
        if name == "e040_w":
            ablate_w = {s: rec["ablate"][s] for s in SITES}
        rec["band_pass"] = all(0.5 * ablate_w[s] <= rec["ablate"][s] <= 2.0 * ablate_w[s]
                               for s in SITES)

        ax = family_axes(sd, REF_SD, cfg)
        init_ax = family_axes(inits[name], REF_SD, cfg)
        rec["axes"] = ax
        rec["init_axes"] = init_ax
        rec["delta_from_init"] = {
            k: ax[k] - init_ax[k] for k in
            ("ln_dist_sites", "ln_dist_all", "win_rowmean_sites",
             "wout_rowmean_sites", "win_erank_sites", "wout_erank_sites")}
        # e052 geometry axes (replication context, not registered firing axes)
        rec["dw_cos"] = {s: cos(dw_vec(sd, inits[name], mlp_keys(s)), REF_DW[s])
                         for s in SITES}
        rec["align_dist"] = 1.0 - mean([rec["dw_cos"][s] for s in SITES])
        rows.append(rec)
        log(f"  {name:9s} D {rec['D']:+.4f} | A {rec['A']:+.4f} | R {rec['R_mean']:.4f} | "
            f"LN {ax['ln_dist_sites']:.4f} | W_in row {ax['win_rowmean_sites']:.4f} | "
            f"W_out row {ax['wout_rowmean_sites']:.4f} | erank {ax['win_erank_sites']:.2f}/"
            f"{ax['wout_erank_sites']:.2f} | CE {rec['base_ce']:.4f}")

    by = {r["name"]: r for r in rows}

    # ---- reproduction gate vs runs/e052/metrics.json ---------------------------
    e052 = json.loads(E052_METRICS.read_text(encoding="utf-8"))
    e052_by = {m["name"]: m for m in e052["members"]}
    repro = {"max_abs_D_diff": 0.0, "max_abs_A_diff": 0.0, "n_checked": 0}
    for r in rows:
        m = e052_by[r["name"]]
        repro["max_abs_D_diff"] = max(repro["max_abs_D_diff"], abs(r["D"] - m["D"]))
        repro["max_abs_A_diff"] = max(repro["max_abs_A_diff"], abs(r["A"] - m["A"]))
        repro["n_checked"] += 1
    log(f"reproduction vs e052 (same 15-batch protocol): max|dD| "
        f"{repro['max_abs_D_diff']:.2e}, max|dA| {repro['max_abs_A_diff']:.2e} "
        f"over {repro['n_checked']} members")
    if max(repro["max_abs_D_diff"], repro["max_abs_A_diff"]) > 1e-6:
        raise SystemExit("REPRODUCTION GATE FAILED vs runs/e052/metrics.json")

    # ---- correlation table (n=10) ----------------------------------------------
    AXES = ["A", "R_mean", "ln_dist_sites", "ln_dist_all",
            "win_rowmean_sites", "wout_rowmean_sites",
            "win_erank_sites", "wout_erank_sites",
            "win_shape_dist_sites", "wout_shape_dist_sites",
            "win_erank_dist_sites", "wout_erank_dist_sites",
            "align_dist"]
    AXIS_LABEL = {
        "A": "own-organ load A", "R_mean": "organ-band ratio R",
        "ln_dist_sites": "LN-dist to REF (graft sites)",
        "ln_dist_all": "LN-dist to REF (whole net)",
        "win_rowmean_sites": "W_in row-norm mean (sites)",
        "wout_rowmean_sites": "W_out row-norm mean (sites)",
        "win_erank_sites": "erank W_in (sites)",
        "wout_erank_sites": "erank W_out (sites)",
        "win_shape_dist_sites": "W_in row-profile shape dist to REF",
        "wout_shape_dist_sites": "W_out row-profile shape dist to REF",
        "win_erank_dist_sites": "|erank W_in - REF| (all blocks)",
        "wout_erank_dist_sites": "|erank W_out - REF| (all blocks)",
        "align_dist": "dW-alignment dist (e052 axis)"}
    D = np.array([r["D"] for r in rows])
    A = np.array([r["A"] for r in rows])
    b1, b0 = np.polyfit(A, D, 1)                    # T026 prescription: residual on A
    Dres = D - (b0 + b1 * A)
    cors = {}
    for k in AXES:
        v = np.array([r["axes"][k] if k in r["axes"] else r[k] for r in rows])
        pr, pp = pearson(D, v)
        sr, sp = spearman(D, v)
        rr, rp = pearson(Dres, v)                   # A-residualized damage
        cors[k] = {"r_D": pr, "p_D": pp, "rho_D": sr, "rho_p_D": sp,
                   "r_Dres_on_A": rr, "p_Dres": rp}
    # winners+unselected-only subtable (n=8: selection-relevant members)
    sub = [r for r in rows if r["name"] in WINNERS + UNSELECTED]
    Ds = np.array([r["D"] for r in sub])
    As_ = np.array([r["A"] for r in sub])
    sb1, sb0 = np.polyfit(As_, Ds, 1)
    Dres_s = Ds - (sb0 + sb1 * As_)
    cors_n8 = {}
    for k in AXES:
        v = np.array([r["axes"][k] if k in r["axes"] else r[k] for r in sub])
        pr, pp = pearson(Ds, v)
        rr, rp = pearson(Dres_s, v)
        cors_n8[k] = {"r_D": pr, "p_D": pp, "r_Dres_on_A": rr, "p_Dres": rp}
    log("correlations with D (n=10): " + " | ".join(
        f"{k}: r={v['r_D']:+.3f}" for k, v in cors.items()))

    # ---- winner-vs-unselected contrasts ----------------------------------------
    def vals(names, key):
        out = []
        for nm in names:
            r = by[nm]
            out.append(r["axes"][key] if key in r["axes"] else r[key])
        return out

    contrasts = {}
    for k in AXES + ["D"]:
        primary = boot_contrast(vals(WINNERS, k), vals(UNSELECTED, k), seed=59)
        e1w, e1u = EVENT1
        e2w, e2u = EVENT2
        e1 = float(np.mean(vals(e1w, k)) - np.mean(vals(e1u, k)))
        e2 = float(np.mean(vals(e2w, k)) - np.mean(vals(e2u, k)))
        signs = np.sign([primary["diff"], e1, e2])
        consistent = bool(np.all(signs == signs[0]) and signs[0] != 0)
        contrasts[k] = {"primary": primary, "event1_diff": e1, "event2_diff": e2,
                        "events_consistent": consistent,
                        "fires": bool(primary["ci_excludes_0"] and consistent)}
        log(f"  contrast {k:24s} prim {primary['diff']:+.4f} "
            f"[{primary['ci95'][0]:+.4f},{primary['ci95'][1]:+.4f}] "
            f"{'*' if primary['ci_excludes_0'] else ' '} | E1 {e1:+.4f} E2 {e2:+.4f} "
            f"{'consistent' if consistent else 'INCONSISTENT'}")

    # delta-from-init audit contrasts (what training changed, winners vs unselected)
    delta_keys = list(rows[0]["delta_from_init"].keys())
    delta_contrasts = {}
    for k in delta_keys:
        delta_contrasts[k] = boot_contrast(
            [by[n]["delta_from_init"][k] for n in WINNERS],
            [by[n]["delta_from_init"][k] for n in UNSELECTED], seed=59)

    # ---- registered verdict -----------------------------------------------------
    r_DA = cors["A"]["r_D"]
    max_axis = max(cors, key=lambda k: abs(cors[k]["r_D"]))
    r_max = cors[max_axis]["r_D"]
    load_axes = ["A"]
    ln_axes = ["ln_dist_sites", "ln_dist_all"]
    iface_axes = ["win_rowmean_sites", "wout_rowmean_sites",
                  "win_erank_sites", "wout_erank_sites",
                  "win_shape_dist_sites", "wout_shape_dist_sites",
                  "win_erank_dist_sites", "wout_erank_dist_sites"]

    c_load = (contrasts["A"]["fires"] and contrasts["A"]["primary"]["diff"] < 0
              and r_DA >= 0.6 and abs(r_DA) >= abs(r_max))
    c_ln = any(contrasts[k]["fires"] and contrasts[k]["primary"]["diff"] < 0
               and cors[k]["r_D"] > r_DA for k in ln_axes)
    c_iface = any(contrasts[k]["fires"] and abs(cors[k]["r_D"]) >= abs(r_DA)
                  for k in iface_axes)
    fired = {"H-load": c_load, "H-LN": c_ln, "H-interface": c_iface}
    if not any(fired.values()):
        verdict = ("H-nothing — no consistent winner signature: no registered "
                   "family separates winners from unselected siblings with a "
                   "sign-consistent, CI-excluding-0 contrast at the registered "
                   "correlation bars; the e040 trickle is a single-lineage "
                   "artifact at parameter level")
    else:
        winners_h = [h for h, ok in fired.items() if ok]
        # largest standardized primary effect among the firing axes
        cand = []
        if c_load:
            cand.append((abs(contrasts["A"]["primary"]["std_effect"]), "H-load", "A"))
        if c_ln:
            cand += [(abs(contrasts[k]["primary"]["std_effect"]), "H-LN", k)
                     for k in ln_axes
                     if contrasts[k]["fires"] and contrasts[k]["primary"]["diff"] < 0
                     and cors[k]["r_D"] > r_DA]
        if c_iface:
            cand += [(abs(contrasts[k]["primary"]["std_effect"]), "H-interface", k)
                     for k in iface_axes
                     if contrasts[k]["fires"] and abs(cors[k]["r_D"]) >= abs(r_DA)]
        cand.sort(reverse=True)
        _, best_h, best_ax = cand[0]
        mixed = " (MIXED note: " + ", ".join(winners_h) + " all fired; declared by " \
                "largest standardized contrast)" if len(winners_h) > 1 else ""
        detail = {k: {"diff": contrasts[k]["primary"]["diff"],
                      "ci95": contrasts[k]["primary"]["ci95"],
                      "r_D": cors[k]["r_D"]} for _, h, k in cand if h == best_h}
        verdict = (f"{best_h} — winner signature on {best_ax}: winner-vs-unselected "
                   f"contrast {contrasts[best_ax]['primary']['diff']:+.4f} "
                   f"CI {contrasts[best_ax]['primary']['ci95']}, "
                   f"r(D, axis) {cors[best_ax]['r_D']:+.3f} vs r(D, A) {r_DA:+.3f}; "
                   f"event contrasts agree{mixed}. Firing axes detail: {detail}")
    log("=" * 78)
    log(f"VERDICT: {verdict}")

    # ---- outputs -----------------------------------------------------------------
    def clean(r):
        out = {"name": r["name"], "base_ce": r["base_ce"],
               "graft_dce": {str(s): r["graft"][s] for s in SITES},
               "ablate_dce": {str(s): r["ablate"][s] for s in SITES},
               "D": r["D"], "A": r["A"],
               "R_sites": {str(s): r["R_sites"][s] for s in SITES},
               "R_mean": r["R_mean"], "band_pass": r["band_pass"],
               "axes": r["axes"], "delta_from_init": r["delta_from_init"],
               "align_dist": r["align_dist"]}
        return out

    metrics = {
        "experiment": "e059_winner_diff",
        "date": common.now_iso(),
        "status": "complete",
        "device": "cpu-only (CUDA_VISIBLE_DEVICES masked; zero-GPU analysis)",
        "config": common.cfg_dict(cfg),
        "params": 840_704,
        "question": ("what did e040's selection actually change in the winners "
                     "vs unselected siblings and the wildtype; does dD track "
                     "dA, dLN-calibration, d(W_in row scales), or d(effective rank)?"),
        "registered_verdict_rules": {
            "H-load": "A contrast CI excludes 0, winners LOWER, r(D,A)>=0.6 and "
                      "strongest |r|, E1/E2 sign-consistent",
            "H-LN": "LN-dist contrast CI excludes 0, winners CLOSER to REF, "
                    "r(D,LN) > r(D,A), events consistent",
            "H-interface": "some W_in/W_out row-scale or erank axis (incl. "
                           "dist-to-REF forms): CI excludes 0, |r| >= |r(D,A)|, "
                           "events consistent",
            "H-nothing": "none fire, or firing axes inconsistent across events",
            "multi-fire": "declare largest standardized primary-contrast effect; "
                          "log MIXED note"},
        "groups": {"winners": WINNERS, "unselected": UNSELECTED,
                   "event1": {"winners": EVENT1[0], "unselected": EVENT1[1]},
                   "event2": {"winners": EVENT2[0], "unselected": EVENT2[1]},
                   "selection_note": ("e040 ranked by R_mean = D/A (organ-"
                                      "devaluation throttled); g1b had the lowest "
                                      "D but highest R and was rejected")},
        "protocol": {"sites": list(SITES), "n_eval_batches": N_EVAL,
                     "n_boot": N_BOOT,
                     "axes": AXIS_LABEL,
                     "band_semantics": ("e040 verbatim organ-band guard: A_site "
                                        "within [0.5, 2.0] x wildtype A_site"),
                     "correlation_note": ("r over all 10 members is shift-"
                                          "invariant == Delta-vs-wildtype r; "
                                          "Dres column = D residualized on A "
                                          "(T026 registered prescription)")},
        "ref": {"name": "e040_ref", "base_ce": mean(base_ref),
                "self_graft_dce_exactly_zero": ref_self_graft_exact,
                "family_axes_self": ref_ax},
        "instrument_gates": {
            "ref_self_transplant_zero": ref_self_graft_exact,
            "hosts_bitwise_restored": True,
            "reproduction_vs_e052": repro,
            "all_band_pass": all(r["band_pass"] for r in rows)},
        "members": [clean(r) for r in rows],
        "d_on_a_ols": {"slope": float(b1), "intercept": float(b0)},
        "correlations_with_D_n10": cors,
        "correlations_with_D_n8_selection_only": cors_n8,
        "winner_vs_unselected_contrasts": contrasts,
        "delta_from_init_contrasts": delta_contrasts,
        "verdict_conditions": {"fired": fired, "r_D_A": r_DA,
                               "strongest_axis": max_axis,
                               "strongest_r": r_max},
        "verdict": verdict,
        "timing_s": round(time.time() - T0, 1),
    }
    save_json(rd / "metrics.json", metrics)

    # ---- differencing figure ------------------------------------------------------
    C_WIN, C_UN, C_E0 = "#2c6fbb", "#c0392b", "#17a2b8"
    groups_color = {nm: (C_WIN if nm in WINNERS else
                         C_UN if nm in UNSELECTED else C_E0) for nm in MEMBERS}
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # (1) forest: standardized winner-unselected contrasts
    ax = axes[0, 0]
    order = AXES
    ys = np.arange(len(order))[::-1]
    for y, k in zip(ys, order):
        c = contrasts[k]["primary"]
        sd_ = c["pooled_sd"] if c["pooled_sd"] > 0 else 1.0
        xerr = [[c["std_effect"] - c["ci95"][0] / sd_],
                [c["ci95"][1] / sd_ - c["std_effect"]]]
        ax.errorbar(c["std_effect"], y, xerr=xerr, fmt="none",
                    ecolor="#8e44ad", capsize=3, zorder=2)
        ax.scatter(c["std_effect"], y,
                   s=90 if c["ci_excludes_0"] else 40,
                   marker="*" if c["ci_excludes_0"] else "o",
                   color="#8e44ad" if c["ci_excludes_0"] else "#555",
                   zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([AXIS_LABEL.get(k, k) for k in order], fontsize=8)
    ax.axvline(0, color="#888", lw=1, ls=":")
    ax.set_xlabel("standardized contrast winners - unselected (pooled SD)")
    ax.set_title("What moved in winners? (* = bootstrap 95% CI excludes 0)")
    ax.grid(alpha=0.25, axis="x")

    # (2-5) key scatters
    panels = [("own-organ load A", "A", False), ("LN-dist to REF (graft sites)",
              "ln_dist_sites", False), ("W_in row-norm mean (graft sites)",
              "win_rowmean_sites", False), ("erank W_in (graft sites)",
              "win_erank_sites", False)]
    for ax, (lab, k, _) in zip(axes.ravel()[1:5], panels):
        v = np.array([r["axes"][k] if k in r["axes"] else r[k] for r in rows])
        for r, x_, y_ in zip(rows, v, D):
            ax.scatter(x_, y_, color=groups_color[r["name"]], s=52, zorder=3)
            ax.annotate(r["name"].replace("e040_", ""), (x_, y_), fontsize=7,
                        alpha=0.8, xytext=(3, 3), textcoords="offset points")
        if np.std(v) > 0:
            bb1, bb0 = np.polyfit(v, D, 1)
            xs = np.linspace(v.min(), v.max(), 50)
            ax.plot(xs, bb0 + bb1 * xs, color="#c0392b", lw=1.2, ls="--", zorder=2)
        rr, pp = pearson(D, v)
        ax.set_xlabel(lab)
        ax.set_ylabel("D = graft damage (member <- REF mlp L2/L3)")
        ax.set_title(f"r = {rr:+.3f} (p={pp:.3f})")
        ax.grid(alpha=0.25)

    # (6) correlation bar chart
    ax = axes[1, 2]
    keys = AXES
    rs = [cors[k]["r_D"] for k in keys]
    fam = {"A": "#c0392b", "R_mean": "#e67e22",
           "ln_dist_sites": "#27ae60", "ln_dist_all": "#27ae60",
           "align_dist": "#7f8c8d"}
    colors = [fam.get(k, "#2c6fbb") for k in keys]
    ypos = np.arange(len(keys))[::-1]
    ax.barh(ypos, rs, color=colors, alpha=0.85)
    ax.axvline(cors["A"]["r_D"], color="#c0392b", lw=1, ls=":", label="r(D, A)")
    ax.set_yticks(ypos)
    ax.set_yticklabels([AXIS_LABEL.get(k, k) for k in keys], fontsize=7)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("Pearson r with D (n=10)")
    ax.set_title("Correlation table (red=load, green=LN, blue=interface)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.25, axis="x")

    fig.suptitle("E059 — winner differencing: e040's selected winners vs unselected "
                 "siblings and wildtype (10 hosts, CPU)\n"
                 f"VERDICT: {verdict.split(' — ')[0]}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(rd / "winner_diff.png", dpi=140)
    plt.close(fig)
    log(f"outputs: {rd} (metrics.json, winner_diff.png)")
    return metrics


if __name__ == "__main__":
    main()
