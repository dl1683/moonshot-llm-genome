"""E076 — COSINE MECHANISM DISCRIMINATOR (T046) + T047 CLAIM-D CRITIC FIXES.

Registered in THINKING.md T046 (discriminator) and T047 (claim-D fixes, folded
here per the critic's own note). Pure reanalysis of runs/e062 (204 cached
host<->donor pairs with D, A, per-depth cosines) plus checkpoint norm/dW
extraction. NO new grafts, NO training, CPU-only.

PART 1 — mechanism discriminator (T046):
  The e062 winner P2 (pre-graft stream-cosine at graft-input depths,
  partial r(D|A) = -0.976 over 204 pairs) admits two explanations:
    H-basis-alignment : cosine measures donor write DIRECTION relative to the
                        host's init-anchored stream basis; damage = geometry
                        misalignment (e029/e040 ladder, made measurable).
    H-magnitude-proxy : cosine proxies write-mass mismatch — bigger donor
                        writes both hurt more and align worse.
  Discriminator: partial the P2 cosine on donor write-norms (W_in/W_out
  row-norm means at graft sites — e062's checkpoint axes) + host stream-norms
  (mean token ||x|| at depths d2/d3 on the same 2-batch probe), with
  functional sensitivities: donor MLP write norm on its OWN stream and IN
  SITU on the host's exact graft-input (ln2(stream); the true post-graft
  write mass).
  REGISTERED BAR: |partial r| >= 0.8 survives -> H-basis-alignment;
                  collapse (< 0.5, e062's winning bar) -> H-magnitude-proxy;
                  else intermediate, reported honestly.

PART 2 — critic fix 1 (T047 claim D): out-of-sample threshold.
  50/50 split of the 204 pairs stratified by host lineage (18 strata = the
  18 host nets); label = D > median(D_TRAIN) (no test leakage); Youden
  threshold fit on the train half only; report out-of-sample AUC + J on the
  test half. Primary = one seeded split (seed 76); sensitivity = 200 random
  stratified resplits (median + 2.5/97.5 pct). Plus leave-one-lineage-out
  partial r (18 folds; A-partial and full-norm partial) and, because 11/12
  donors are seed-42 kin, leave-one-DONOR-out (12 folds) as sensitivity.
  REGISTERED BAR: out-of-sample AUC >= 0.85 -> instrument survives.

PART 3 — critic fix 2 (T047 claim D): dW-alignment comparison.
  e059's axis — cosine of dW (= trained - init) over graft-site MLP keys,
  there measured vs the fixed REF donor (align_dist, r = +0.543 at n=10) —
  generalized to pairs: dWcos(host, donor) = mean over sites {2,3} of
  cos(dW_host, dW_donor). Inits rebuilt EXACTLY (e040 lineage via e059's
  build_inits; e005s_small = seed-42 init, asserted bitwise-equal to e040_w's
  init; e050 directed-mutation chain per runs/e050 selection events:
  m1/m2/m3 = init42 + eps seeds 5021/22/23 sigma 0.005 on MLP W_in/W_out
  only; g1a/g1b from m3's init seeds 5031/32, g1c from m2's init seed 5033)
  and GATED against e059's recorded align_dist (10 members) and e050's
  recorded dw_cos (6 members) at 1e-6.
  REGISTERED BAR: |partial r(P2 | A)| - |partial r(dWcos | A)| >= 0.1 ->
  instrument-novel (the cheap cosine beats e059's dW instrument).

Run: python lab/e076_cosine_mechanism.py   (CPU-only; CUDA masked before torch)
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # strict CPU-only

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as sstats

import common

common.DEVICE = "cpu"
from common import (REPO, Cfg, CharCorpus, TinyGPT, run_dir, save_json,
                    set_seed)

# ---------------------------------------------------------------- protocol
CORPUS_PATH = REPO / "data" / "input.txt"
CKPT_DIR = REPO / "runs" / "checkpoints"
E062_METRICS = REPO / "runs" / "e062" / "metrics.json"
E059_METRICS = REPO / "runs" / "e059" / "metrics.json"
E050_METRICS = REPO / "runs" / "e050" / "metrics.json"
SITES = (2, 3)
N_BOOT_CLUSTER = 2000
N_RESPLIT = 200
SEED = 76
SIGMA_MUT = 0.005
SEED_W, SEED_REF = 42, 4304

POOL_A = ["e040_ref", "e040_w", "e005s_small",
          "e040_m1", "e040_m2", "e040_m3",
          "e040_g1a", "e040_g1b", "e040_g1c",
          "e040_g2a", "e040_g2b", "e040_g2c"]
HOSTS_EXTRA = ["e050_m1", "e050_m2", "e050_m3",
               "e050_g1a", "e050_g1b", "e050_g1c"]   # hosts only (same arch)
HOSTS_ALL = POOL_A + HOSTS_EXTRA                     # 18 lineages

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def mean(xs) -> float:
    return float(sum(xs) / len(xs))


# ---------------------------------------------------------------- model IO
def mlp_keys(i: int) -> list[str]:
    return [f"h.{i}.mlp.0.weight", f"h.{i}.mlp.0.bias",
            f"h.{i}.mlp.2.weight", f"h.{i}.mlp.2.bias"]


def load_sd(name: str) -> dict:
    obj = torch.load(CKPT_DIR / f"{name}.pt", map_location="cpu",
                     weights_only=False)
    return obj["model"] if "model" in obj else obj


@torch.no_grad()
def probe_full(model, corpus, n_batches: int = 2):
    """e062's shared probe, verbatim batch stream, PLUS what e076 needs:
    (a) token-unit-normalized flats per depth (e062 gate),
    (b) mean token stream-norm per depth (host stream-norm covariates),
    (c) EXACT mlp inputs at SITES (block.ln2 of the post-attention stream) —
        the true input the grafted donor organ reads in situ.
    Block math replicates Block.forward op-for-op (bitwise-identical CPU path).
    """
    model.eval()
    cfg = model.cfg
    gen = torch.Generator().manual_seed(corpus.seed)
    src = corpus.val
    ix = torch.randint(len(src) - cfg.block_size - 1, (16 * n_batches,),
                       generator=gen)
    flat = None
    norms = [[] for _ in range(cfg.n_layer + 1)]
    mlpin = {s: [] for s in SITES}
    for b in range(n_batches):
        x = torch.stack([src[i: i + cfg.block_size]
                         for i in ix[b * 16: (b + 1) * 16]])
        pos = torch.arange(cfg.block_size)
        s = model.wte(x) + model.wpe(pos)
        outs = [s]
        for i, block in enumerate(model.h):
            s2 = s + block.attn(block.ln1(s))
            mi = block.ln2(s2)
            if i in SITES:
                mlpin[i].append(mi.reshape(-1, mi.shape[-1]))
            s = s2 + block.mlp(mi)
            outs.append(s)
        new_flat = []
        for k, t in enumerate(outs):
            norms[k].append(float(t.norm(dim=-1).mean()))
            nrm = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            new_flat.append(nrm.reshape(-1, t.shape[-1]))
        if flat is None:
            flat = new_flat
        else:
            flat = [torch.cat([a, f], dim=0) for a, f in zip(flat, new_flat)]
    model.train()
    return flat, [mean(v) for v in norms], {s: torch.cat(v, 0) for s, v in mlpin.items()}


@torch.no_grad()
def functional_write_norm(mlpin_site: torch.Tensor, sd: dict, site: int) -> float:
    """||donor mlp|| on given (host or donor own) exact graft-input tokens."""
    h = sd[f"h.{site}.mlp.0.weight"].float()
    b0 = sd[f"h.{site}.mlp.0.bias"].float()
    w = sd[f"h.{site}.mlp.2.weight"].float()
    b2 = sd[f"h.{site}.mlp.2.bias"].float()
    hid = torch.nn.functional.gelu(mlpin_site @ h.T + b0)
    out = hid @ w.T + b2
    return float(out.norm(dim=-1).mean())


# ---------------------------------------------------------------- init lineage
# e059/e050-verbatim init reconstruction (bitwise gates below prove it).
def make_init(cfg, seed: int) -> dict:
    set_seed(seed)
    return {k: v.detach().clone() for k, v in TinyGPT(cfg).state_dict().items()}


def mutated_init(base_sd: dict, seed: int, sigma: float) -> dict:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in base_sd.items():
        if k.endswith(".weight"):
            out[k] = v + torch.randn(v.shape, generator=g) * sigma
        else:
            out[k] = v.clone()
    return out


def build_e040_inits(cfg) -> dict[str, dict]:
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


DIRECTED_KEYS = tuple(f"h.{i}.mlp.{j}.weight" for i in range(4) for j in (0, 2))
DIRECTED_SET = frozenset(DIRECTED_KEYS)


def directed_mutant_init(base_sd: dict, seed: int, sigma: float) -> dict:
    """e050 verbatim: eps lands ONLY on stream-facing MLP W_in/W_out."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in base_sd.items():
        if k in DIRECTED_SET:
            out[k] = v + torch.randn(v.shape, generator=g) * sigma
        else:
            out[k] = v.clone()
    return out


def build_all_inits(cfg) -> dict[str, dict]:
    inits = build_e040_inits(cfg)
    inits["e005s_small"] = make_init(cfg, SEED_W)     # e005s: set_seed(42) init
    init42 = make_init(cfg, SEED_W)
    assert torch.equal(init42["h.0.mlp.0.weight"],
                       inits["e040_w"]["h.0.mlp.0.weight"])
    # e050 chain (selection events from runs/e050/metrics.json):
    # parents m3 (top) and m2 (second); g1a/g1b from m3, g1c from m2.
    m1 = directed_mutant_init(init42, 5021, SIGMA_MUT)
    m2 = directed_mutant_init(init42, 5022, SIGMA_MUT)
    m3 = directed_mutant_init(init42, 5023, SIGMA_MUT)
    inits["e050_m1"], inits["e050_m2"], inits["e050_m3"] = m1, m2, m3
    inits["e050_g1a"] = directed_mutant_init(m3, 5031, SIGMA_MUT)
    inits["e050_g1b"] = directed_mutant_init(m3, 5032, SIGMA_MUT)
    inits["e050_g1c"] = directed_mutant_init(m2, 5033, SIGMA_MUT)
    return inits


def dw_vec(sd: dict, init: dict, keys: list[str]) -> torch.Tensor:
    return torch.cat([(sd[k].float() - init[k].float()).reshape(-1) for k in keys])


def tcos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12))


# ---------------------------------------------------------------- stats
def pearson(x, y):
    r, p = sstats.pearsonr(x, y)
    return float(r), float(p)


def ols_resid(y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return y - Z @ beta


def design(cols) -> np.ndarray:
    cols = [np.asarray(c, dtype=float) for c in cols]
    return np.column_stack([np.ones(len(cols[0]))] + cols)


def partial_r(y, x, covars) -> float:
    """Partial correlation of y, x given covariates (list; may be empty)."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    if len(covars) == 0:
        return float(np.corrcoef(y, x)[0, 1])
    Z = design(covars)
    ry, rx = ols_resid(y, Z), ols_resid(x, Z)
    if np.std(ry) == 0 or np.std(rx) == 0:
        return float("nan")
    return float(np.corrcoef(ry, rx)[0, 1])


def fisher_ci(r: float, n: int, k_cov: int) -> tuple[float, float, float]:
    if n - k_cov - 2 <= 0:
        return float("nan"), float("nan"), float("nan")
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - k_cov - 2)
    lo, hi = np.tanh(z - 1.959964 * se), np.tanh(z + 1.959964 * se)
    p = float(2 * sstats.norm.sf(abs(z) * np.sqrt(n - k_cov - 2)))
    return float(lo), float(hi), p


def cluster_boot(y, x, covars, clusters, n: int = N_BOOT_CLUSTER,
                 seed: int = 76) -> list[float]:
    y, x, cl = (np.asarray(v) for v in (y, x, clusters))
    uniq = np.unique(cl)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([np.where(cl == c)[0] for c in pick])
        if len(np.unique(cl[idx])) < 4:
            continue
        r = partial_r(y[idx], x[idx], [c[idx] for c in covars])
        if np.isfinite(r):
            out.append(r)
    return out


def boot_ci(vals) -> list[float]:
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] \
        if vals else [float("nan")] * 2


def auc_roc(score, label) -> float:
    """AUC for a COMPATIBILITY score (higher = compatible) vs label 1 =
    damaging; e062 implementation verbatim (P(damaging scores below
    compatible))."""
    score, label = np.asarray(score, float), np.asarray(label, int)
    order = np.argsort(-score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n1, n0 = int(label.sum()), int((1 - label).sum())
    return float((ranks[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def youden(score, label) -> tuple[float, float]:
    """Best threshold (max Youden J) for a compatibility score; e062 verbatim."""
    score, label = np.asarray(score, float), np.asarray(label, int)
    best_t, best_j = float("nan"), -1.0
    for t in np.unique(score):
        pred_ok = score >= t
        n1, n0 = max(label.sum(), 1), max((1 - label).sum(), 1)
        tpr = np.sum(pred_ok & (label == 0)) / n0
        fpr = np.sum(pred_ok & (label == 1)) / n1
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = float(j), float(t)
    return best_t, best_j


def apply_J(score, label, thr) -> float:
    """Youden J of a FIXED threshold on new data (out-of-sample J)."""
    score, label = np.asarray(score, float), np.asarray(label, int)
    pred_ok = score >= thr
    n1, n0 = max(label.sum(), 1), max((1 - label).sum(), 1)
    tpr = np.sum(pred_ok & (label == 0)) / n0
    fpr = np.sum(pred_ok & (label == 1)) / n1
    return float(tpr - fpr)


# ---------------------------------------------------------------- main
def main():
    rd = run_dir("e076")
    log("E076 — cosine mechanism discriminator + crossmatch critic fixes "
        "(CPU-only, reanalysis of e062's 204 cached pairs)")
    set_seed(SEED)
    corpus = CharCorpus(CORPUS_PATH, seed=1337)
    cfg = Cfg(vocab=corpus.vocab_size, n_layer=4, n_head=4, n_embd=128,
              block_size=256)
    net = TinyGPT(cfg)
    assert net.num_params() == 840_704

    # ---- cached pair table (e062 metrics, verbatim) --------------------------
    e062 = json.loads(E062_METRICS.read_text(encoding="utf-8"))
    rows = e062["pairs"]
    assert len(rows) == 204
    D = np.array([r["D"] for r in rows])
    A = np.array([r["A_host"] for r in rows])
    P2 = np.array([r["cos_graftinput"] for r in rows])
    P1 = np.array([r["P1_wout_rowmean_sites"] for r in rows])
    hcl = np.array([r["host"] for r in rows])
    dcl = np.array([r["donor"] for r in rows])
    d_wout = np.array([r["donor_wout_rowmean_sites"] for r in rows])
    d_win = np.array([r["donor_win_rowmean_sites"] for r in rows])
    n = len(rows)
    log(f"loaded {n} cached pairs | D [{D.min():+.3f},{D.max():+.3f}] | "
        f"reproducing e062 partial r(D,P2|A)={partial_r(D, P2, [A]):+.4f} "
        f"(recorded {e062['correlations_scaleA']['P2_gi']['partial_r_given_A']:+.4f})")

    # ---- recompute streams / norms / functional writes from checkpoints ------
    sds = {name: load_sd(name) for name in HOSTS_ALL}
    flats, hsnorm, mlpin_host, dwnorm_own = {}, {}, {}, {}
    for name in HOSTS_ALL:
        net.load_state_dict(sds[name])
        fl, nrm, mpi = probe_full(net, corpus)
        flats[name], hsnorm[name], mlpin_host[name] = fl, nrm, mpi
        dwnorm_own[name] = mean([functional_write_norm(mpi[s], sds[name], s)
                                 for s in SITES])
    # gate: cosines bitwise-reproduce e062's
    max_dcos = 0.0
    for r in rows:
        a = flats[r["host"]][2]
        b = flats[r["donor"]][2]
        c2 = float((a * b).sum(-1).mean())
        a3 = flats[r["host"]][3]
        b3 = flats[r["donor"]][3]
        c3 = float((a3 * b3).sum(-1).mean())
        max_dcos = max(max_dcos, abs(mean([c2, c3]) - r["cos_graftinput"]))
    # gate: donor row-norm axes reproduce e062's
    max_dax = 0.0
    for name in POOL_A:
        wout = mean([float(sds[name][f"h.{s}.mlp.2.weight"].float()
                           .norm(dim=1).mean()) for s in SITES])
        win = mean([float(sds[name][f"h.{s}.mlp.0.weight"].float()
                          .norm(dim=1).mean()) for s in SITES])
        for r in rows:
            if r["donor"] == name:
                max_dax = max(max_dax, abs(wout - r["donor_wout_rowmean_sites"]),
                              abs(win - r["donor_win_rowmean_sites"]))
    # pair-level covariates
    hsn = np.array([mean([hsnorm[r["host"]][2], hsnorm[r["host"]][3]])
                    for r in rows])               # host stream-norm, graft-input
    hsn2 = np.array([hsnorm[r["host"]][2] for r in rows])
    hsn3 = np.array([hsnorm[r["host"]][3] for r in rows])
    dwr_insitu = np.array([mean([functional_write_norm(
        mlpin_host[r["host"]][s], sds[r["donor"]], s) for s in SITES])
        for r in rows])                            # donor write mass IN SITU
    dwo = np.array([dwnorm_own[r["donor"]] for r in rows])
    gates = {
        "pairs_from_e062_cache": n,
        "e062_partial_reproduced":
            abs(partial_r(D, P2, [A])
                - e062["correlations_scaleA"]["P2_gi"]["partial_r_given_A"]) <= 1e-9,
        "stream_cosine_max_abs_diff": max_dcos,
        "donor_axes_max_abs_diff": max_dax,
    }
    log(f"gates: cos max|diff| {max_dcos:.2e} | donor axes max|diff| "
        f"{max_dax:.2e}")
    if max_dcos > 1e-6 or max_dax > 1e-6:
        raise SystemExit("GATE FAILED: recomputed streams/axes != e062")

    # ================= PART 1 — mechanism discriminator (T046) ================
    log("PART 1 — partial P2 on donor write-norms + host stream-norms")
    cov_donor = [d_wout, d_win]
    cov_host = [hsn2, hsn3]
    full_cov = [A] + cov_donor + cov_host

    waterfall = []
    steps = [("raw r(D, P2)", []),
             ("+ A", [A]),
             ("+ donor W_out rowmean", [A, d_wout]),
             ("+ donor W_in rowmean", [A, d_wout, d_win]),
             ("+ host stream-norm d2", [A, d_wout, d_win, hsn2]),
             ("+ host stream-norm d3 (FULL)", full_cov)]
    for lab, cov in steps:
        r = partial_r(D, P2, cov)
        k = len(cov)
        lo, hi, p = fisher_ci(r, n, k)
        waterfall.append({"step": lab, "n_covariates": k, "partial_r": r,
                          "ci95_fisher": [lo, hi], "p": p})
        log(f"  {lab:30s} partial r = {r:+.4f} "
            f"[{lo:+.4f},{hi:+.4f}] p={p:.1e}")

    r_full = waterfall[-1]["partial_r"]
    boot_h = cluster_boot(D, P2, full_cov, hcl, seed=76)
    boot_d = cluster_boot(D, P2, full_cov, dcl, seed=77)
    ci_full_h, ci_full_d = boot_ci(boot_h), boot_ci(boot_d)
    lo, hi, p = fisher_ci(r_full, n, len(full_cov))
    log(f"  FULL partial r = {r_full:+.4f} fisher [{lo:+.4f},{hi:+.4f}] | "
        f"hostCl {ci_full_h} | donorCl {ci_full_d}")

    # sensitivities
    r_nonly = partial_r(D, P2, cov_donor + cov_host)          # norms, no A
    r_func = partial_r(D, P2, full_cov + [dwo, dwr_insitu])   # + functional
    r_func_only = partial_r(D, P2, [A, dwo, dwr_insitu, hsn2, hsn3])
    r_insitu_only = partial_r(D, P2, [A, dwr_insitu, hsn])
    # what does each magnitude family ALONE remove?
    r_donor_only = partial_r(D, P2, [A] + cov_donor)
    r_host_only = partial_r(D, P2, [A] + cov_host)
    sens = {
        "norms_only_no_A": r_nonly,
        "plus_functional_own_and_insitu": r_func,
        "functional_replacing_rowmeans": r_func_only,
        "insitu_write_only": r_insitu_only,
        "donor_norms_only_given_A": r_donor_only,
        "host_norms_only_given_A": r_host_only,
        "corr_D_insitu_write": pearson(D, dwr_insitu)[0],
        "corr_D_host_streamnorm": pearson(D, hsn)[0],
        "corr_P2_insitu_write": pearson(P2, dwr_insitu)[0],
        "corr_P2_host_streamnorm": pearson(P2, hsn)[0],
    }
    log("  sensitivities: " + " | ".join(f"{k}={v:+.3f}"
                                         for k, v in sens.items()))

    bar1 = abs(r_full)
    if bar1 >= 0.8:
        v1 = ("H-BASIS-ALIGNMENT SURVIVES: |partial r| after partialing on "
              f"donor write-norms + host stream-norms = {bar1:.3f} >= 0.8 — "
              "the cosine carries direction information beyond write mass")
    elif bar1 < 0.5:
        v1 = ("H-MAGNITUDE-PROXY WINS: partial r collapses to "
              f"{bar1:.3f} < 0.5 — the cosine was proxying write-mass mismatch")
    else:
        v1 = (f"INTERMEDIATE: partial r survives at {bar1:.3f} (0.5-0.8 band) "
              "— neither registered mechanism cleanly wins")
    log(f"  PART 1 VERDICT: {v1}")

    # ============ PART 2 — out-of-sample threshold + LOLO (critic fix 1) =====
    log("PART 2 — split-sample threshold (stratified by host lineage) + LOLO")

    def stratified_split(rng):
        tr, te = [], []
        odd_turn = 0
        for h in [x for x in HOSTS_ALL]:
            idx = np.where(hcl == h)[0]
            perm = rng.permutation(idx)
            k = len(idx) // 2
            if len(idx) % 2 == 1:
                take = k + (1 if odd_turn % 2 == 0 else 0)
                odd_turn += 1
            else:
                take = k
            tr.extend(perm[:take].tolist())
            te.extend(perm[take:].tolist())
        return np.array(tr), np.array(te)

    def oos_eval(tr, te):
        dmed = float(np.median(D[tr]))
        lab_tr = (D[tr] > dmed).astype(int)
        lab_te = (D[te] > dmed).astype(int)
        thr, j_in = youden(P2[tr], lab_tr)
        auc_te = auc_roc(P2[te], lab_te)
        j_te = apply_J(P2[te], lab_te, thr)
        return {"threshold": thr, "j_train": j_in, "auc_test": auc_te,
                "j_test": j_te, "n_train": len(tr), "n_test": len(te),
                "n_pos_test": int(lab_te.sum()),
                "label_median": dmed}

    rng = np.random.default_rng(SEED)
    tr0, te0 = stratified_split(rng)
    primary = oos_eval(tr0, te0)
    log(f"  primary split: n {primary['n_train']}/{primary['n_test']} | "
        f"train Youden thr {primary['threshold']:.4f} (J_train "
        f"{primary['j_train']:.3f}) | TEST AUC {primary['auc_test']:.3f} "
        f"J {primary['j_test']:.3f}")

    resplits = []
    for _ in range(N_RESPLIT):
        tr, te = stratified_split(rng)
        resplits.append(oos_eval(tr, te))
    auc_oos = np.array([r["auc_test"] for r in resplits])
    j_oos = np.array([r["j_test"] for r in resplits])
    thr_oos = np.array([r["threshold"] for r in resplits])
    oos_summary = {
        "n_resplits": N_RESPLIT,
        "auc_test_median": float(np.median(auc_oos)),
        "auc_test_p2.5_p97.5": [float(np.percentile(auc_oos, 2.5)),
                                float(np.percentile(auc_oos, 97.5))],
        "auc_test_min": float(auc_oos.min()),
        "j_test_median": float(np.median(j_oos)),
        "j_test_p2.5_p97.5": [float(np.percentile(j_oos, 2.5)),
                              float(np.percentile(j_oos, 97.5))],
        "threshold_median": float(np.median(thr_oos)),
        "threshold_p2.5_p97.5": [float(np.percentile(thr_oos, 2.5)),
                                 float(np.percentile(thr_oos, 97.5))],
        "frac_auc_ge_0.85": float((auc_oos >= 0.85).mean()),
    }
    log(f"  {N_RESPLIT} resplits: AUC median {oos_summary['auc_test_median']:.3f} "
        f"[{oos_summary['auc_test_p2.5_p97.5'][0]:.3f},"
        f"{oos_summary['auc_test_p2.5_p97.5'][1]:.3f}] | "
        f"J median {oos_summary['j_test_median']:.3f} | "
        f"{oos_summary['frac_auc_ge_0.85']*100:.0f}% of splits >= 0.85")

    # in-sample reference (e062 label def) for context
    lab_all = (D > np.median(D)).astype(int)
    auc_in = auc_roc(P2, lab_all)
    thr_in, j_in_all = youden(P2, lab_all)

    # leave-one-lineage-out (host) partials + leave-one-donor-out
    lolo = []
    for h in HOSTS_ALL:
        m = hcl != h
        lolo.append({"left_out_host": h, "n_pairs": int(m.sum()),
                     "partial_r_given_A": partial_r(D[m], P2[m], [A[m]]),
                     "partial_r_full": partial_r(D[m], P2[m],
                                                 [c[m] for c in full_cov])})
    rl_a = np.array([x["partial_r_given_A"] for x in lolo])
    rl_f = np.array([x["partial_r_full"] for x in lolo])
    lodo = []
    for d in POOL_A:
        m = dcl != d
        lodo.append({"left_out_donor": d, "n_pairs": int(m.sum()),
                     "partial_r_given_A": partial_r(D[m], P2[m], [A[m]]),
                     "partial_r_full": partial_r(D[m], P2[m],
                                                 [c[m] for c in full_cov])})
    rd_a = np.array([x["partial_r_given_A"] for x in lodo])
    rd_f = np.array([x["partial_r_full"] for x in lodo])
    log(f"  LOLO (18 host folds): A-partial r in "
        f"[{rl_a.min():+.4f},{rl_a.max():+.4f}] | full-norm partial r in "
        f"[{rl_f.min():+.4f},{rl_f.max():+.4f}]")
    log(f"  LODO (12 donor folds): A-partial r in "
        f"[{rd_a.min():+.4f},{rd_a.max():+.4f}] | full-norm partial r in "
        f"[{rd_f.min():+.4f},{rd_f.max():+.4f}]")

    surv = (primary["auc_test"] >= 0.85
            and oos_summary["auc_test_p2.5_p97.5"][0] >= 0.85)
    surv_primary_only = primary["auc_test"] >= 0.85
    v2 = (f"out-of-sample AUC = {primary['auc_test']:.3f} (primary split), "
          f"median {oos_summary['auc_test_median']:.3f} over {N_RESPLIT} "
          f"resplits [2.5th pct {oos_summary['auc_test_p2.5_p97.5'][0]:.3f}]; "
          f"out-of-sample J = {primary['j_test']:.3f} (median "
          f"{oos_summary['j_test_median']:.3f}). "
          + ("INSTRUMENT SURVIVES the registered 0.85 bar (primary split "
             + ("AND resplit 2.5th pct)" if surv else
                "but resplit 2.5th pct dips below — borderline)")
             if surv_primary_only else
             "INSTRUMENT FAILS the registered 0.85 out-of-sample bar"))
    log(f"  PART 2 VERDICT: {v2}")

    # ============ PART 3 — dW-alignment comparison (critic fix 2) ============
    log("PART 3 — dW-alignment predictor (e052/e059 method, pair-generalized)")
    inits = build_all_inits(cfg)
    dws = {name: {s: dw_vec(sds[name], inits[name], mlp_keys(s))
                  for s in SITES} for name in HOSTS_ALL}

    # init gates vs e059 align_dist (10 members) and e050 dw_cos (6 members)
    e059 = json.loads(E059_METRICS.read_text(encoding="utf-8"))
    e059_by = {m["name"]: m for m in e059["members"]}
    g059 = 0.0
    for nm, m in e059_by.items():
        mine = 1.0 - mean([tcos(dws[nm][s], dws["e040_ref"][s]) for s in SITES])
        g059 = max(g059, abs(mine - m["align_dist"]))
    e050 = json.loads(E050_METRICS.read_text(encoding="utf-8"))
    g050 = 0.0
    for gen_list in e050["members"].values():
        for m in gen_list:
            nm = m["name"]
            nm = "e040_w" if nm == "e050_w" else nm   # e050_w = e040_w reused
            if nm in dws:
                for s in SITES:
                    g050 = max(g050, abs(tcos(dws[nm][s], dws["e040_ref"][s])
                                         - m["dw_cos"][str(s)]))
    gates["e059_align_dist_max_abs_diff"] = g059
    gates["e050_dw_cos_max_abs_diff"] = g050
    log(f"  init gates: e059 align_dist max|diff| {g059:.2e} | e050 dw_cos "
        f"max|diff| {g050:.2e}")
    if g059 > 1e-6 or g050 > 1e-6:
        raise SystemExit("GATE FAILED: rebuilt inits != e059/e050 instruments")

    dWc = np.array([mean([tcos(dws[r["host"]][s], dws[r["donor"]][s])
                          for s in SITES]) for r in rows])
    r_dw, p_dw = pearson(D, dWc)
    pr_dw = partial_r(D, dWc, [A])
    lo_dw, hi_dw, pp_dw = fisher_ci(pr_dw, n, 1)
    dw_boot_h = cluster_boot(D, dWc, [A], hcl, seed=78)
    dw_boot_d = cluster_boot(D, dWc, [A], dcl, seed=79)
    auc_dw = auc_roc(dWc, lab_all)
    thr_dw, j_dw = youden(dWc, lab_all)
    # norm-partialed variant (same full covariate set as part 1)
    pr_dw_full = partial_r(D, dWc, full_cov)
    log(f"  dWcos: r(D)={r_dw:+.3f} (p={p_dw:.1e}) | partial r(D|A)="
        f"{pr_dw:+.3f} fisher [{lo_dw:+.3f},{hi_dw:+.3f}] | hostCl "
        f"{boot_ci(dw_boot_h)} donorCl {boot_ci(dw_boot_d)} | AUC {auc_dw:.3f} "
        f"(J {j_dw:.3f}) | partial-full {pr_dw_full:+.3f}")

    pr_p2 = partial_r(D, P2, [A])
    margin = abs(pr_p2) - abs(pr_dw)
    v3 = (f"|partial r| P2 {abs(pr_p2):.3f} vs dW {abs(pr_dw):.3f}: margin "
          f"{margin:+.3f} " + ("-> INSTRUMENT-NOVEL (>= 0.1 bar met)"
                               if margin >= 0.1 else
                               ("-> NOT novel at the 0.1 bar"
                                if margin >= 0 else
                                "-> dW BEATS the cosine")))
    log(f"  PART 3 VERDICT: {v3}")

    verdict = {
        "part1_mechanism": v1,
        "part1_partial_full": r_full,
        "part1_bar": "|partial r| >= 0.8 alignment; < 0.5 magnitude",
        "part2_oos": v2,
        "part2_primary_auc_test": primary["auc_test"],
        "part2_bar": "out-of-sample AUC >= 0.85",
        "part3_dw": v3,
        "part3_margin": margin,
        "part3_bar": "|partial r(P2)| - |partial r(dW)| >= 0.1",
    }

    # ---------------------------------------------------------------- outputs
    metrics = {
        "experiment": "e076_cosine_mechanism",
        "date": common.now_iso(),
        "status": "complete",
        "device": "cpu-only (CUDA_VISIBLE_DEVICES masked; zero-GPU analysis)",
        "question": ("T046: does the e062 cosine crossmatch measure basis "
                     "ALIGNMENT or write MAGNITUDE? T047 claim-D fixes: does "
                     "the 0.919 AUC survive an out-of-sample threshold, and "
                     "does the cheap cosine beat e059's dW-alignment axis?"),
        "registered_bars": {
            "part1": "|partial r(D, P2 | A + donor write-norms + host "
                     "stream-norms)| >= 0.8 -> H-basis-alignment; < 0.5 -> "
                     "H-magnitude-proxy (collapse)",
            "part2": "out-of-sample AUC >= 0.85 (Youden threshold fit on "
                     "stratified train half) -> instrument survives",
            "part3": "|partial r(D, P2 | A)| - |partial r(D, dWcos | A)| "
                     ">= 0.1 -> instrument-novel",
        },
        "data": "runs/e062 metrics.json pair table (204 pairs, cached D/A/"
                "cosines) + checkpoints; streams/norms/dW recomputed and "
                "gated against e062/e059/e050 recordings",
        "gates": gates,
        "covariates": {
            "A": "host own-organ ablation dCE (e062 cache)",
            "donor_wout_rowmean_sites": "donor W_out row-norm mean, sites "
                                        "(checkpoint, e062 axis)",
            "donor_win_rowmean_sites": "donor W_in row-norm mean, sites "
                                       "(checkpoint, e062 axis)",
            "host_streamnorm_d2_d3": "host mean token ||stream|| at graft-"
                                     "input depths (2-batch probe)",
            "donor_write_own": "donor MLP write norm on OWN exact graft input",
            "donor_write_insitu": "donor MLP write norm on HOST's exact graft "
                                  "input (true post-graft write mass)",
        },
        "part1": {
            "waterfall": waterfall,
            "partial_r_full": r_full,
            "partial_r_full_ci95_fisher": [lo, hi],
            "partial_r_full_ci95_hostcluster": ci_full_h,
            "partial_r_full_ci95_donorcluster": ci_full_d,
            "sensitivities": sens,
            "verdict": v1,
        },
        "part2": {
            "split": "50/50 stratified by host lineage (18 strata); label = "
                     "D > median(D_train); Youden on train only",
            "primary_split": primary,
            "resplits": oos_summary,
            "insample_reference": {"auc": auc_in, "youden_threshold": thr_in,
                                   "youden_J": j_in_all,
                                   "label_rule": e062["label_definition"]["rule"]},
            "leave_one_host_lineage_out": lolo,
            "lolo_summary": {
                "partial_r_given_A_min_max": [float(rl_a.min()), float(rl_a.max())],
                "partial_r_full_min_max": [float(rl_f.min()), float(rl_f.max())],
                "n_folds": len(lolo)},
            "leave_one_donor_out": lodo,
            "lodo_summary": {
                "partial_r_given_A_min_max": [float(rd_a.min()), float(rd_a.max())],
                "partial_r_full_min_max": [float(rd_f.min()), float(rd_f.max())],
                "n_folds": len(lodo)},
            "verdict": v2,
        },
        "part3": {
            "predictor": "dWcos(host, donor) = mean_{s in 2,3} cos(dW_host_s, "
                         "dW_donor_s); dW = trained - rebuilt init over "
                         "mlp_keys(s) (e052/e059 verbatim)",
            "r_D": r_dw, "p_r": p_dw,
            "partial_r_given_A": pr_dw,
            "partial_ci95_fisher": [lo_dw, hi_dw],
            "partial_ci95_hostcluster": boot_ci(dw_boot_h),
            "partial_ci95_donorcluster": boot_ci(dw_boot_d),
            "partial_r_full_covariates": pr_dw_full,
            "auc": auc_dw, "youden_threshold": thr_dw, "youden_J": j_dw,
            "e059_recorded_reference": {
                "align_dist_r_D_n10":
                    e059["correlations_with_D_n10"]["align_dist"]["r_D"],
                "note": "e059: r(D, 1 - cos(dW_member, dW_REF)) over 10 "
                        "members, fixed REF donor"},
            "comparison": {
                "partial_r_P2_given_A": pr_p2,
                "partial_r_dW_given_A": pr_dw,
                "abs_margin": margin,
                "auc_P2": auc_in, "auc_dW": auc_dw,
                "auc_P1_reference": e062["roc"]["P1"]["auc"],
                "partial_r_P1_reference":
                    e062["correlations_scaleA"]["P1"]["partial_r_given_A"]},
            "verdict": v3,
        },
        "verdicts": verdict,
        "timing_s": round(time.time() - T0, 1),
    }
    save_json(rd / "metrics.json", metrics)

    # ---------------------------------------------------------------- figure
    fig, axs = plt.subplots(2, 3, figsize=(17, 10))

    # (a) cumulative partial-r waterfall
    ax = axs[0, 0]
    labs = [w["step"].replace(" (FULL)", "\n(FULL)") for w in waterfall]
    vals = [w["partial_r"] for w in waterfall]
    ax.bar(range(len(vals)), vals, color="#27ae60")
    ax.axhline(0.8, color="#e67e22", lw=1.2, ls="--", label="|r| = 0.8 bar")
    ax.axhline(-0.8, color="#e67e22", lw=1.2, ls="--")
    ax.axhline(-0.5, color="#888", lw=0.9, ls=":", label="|r| = 0.5 (e062 bar)")
    for i, v_ in enumerate(vals):
        ax.text(i, v_ - 0.06, f"{v_:+.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labs, fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("signed r (D, P2 | covariates so far)")
    ax.set_title("PART 1: norm-partialing waterfall (T046 discriminator)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, axis="y")

    # (b) residual-residual scatter after FULL partialing
    ax = axs[0, 1]
    Z = design(full_cov)
    Dres, Pres = ols_resid(D, Z), ols_resid(P2, Z)
    ax.scatter(Pres, Dres, s=20, color="#2c6fbb", alpha=0.65, zorder=3)
    b1, b0 = np.polyfit(Pres, Dres, 1)
    xs = np.linspace(Pres.min(), Pres.max(), 50)
    ax.plot(xs, b0 + b1 * xs, color="#e67e22", lw=1.4, ls="--", zorder=2)
    ax.set_xlabel("P2 residual (cosine | A + donor write-norms + host stream-norms)")
    ax.set_ylabel("D residual (same partialing)")
    ax.set_title(f"discriminator scatter: r = {r_full:+.3f} "
                 f"{'(>= 0.8: ALIGNMENT)' if abs(r_full) >= 0.8 else '(< 0.8)'}")
    ax.grid(alpha=0.25)

    # (c) D vs raw cosine (context)
    ax = axs[0, 2]
    ax.scatter(P2, D, s=20, color="#7d3c98", alpha=0.65, zorder=3)
    b1, b0 = np.polyfit(P2, D, 1)
    xs = np.linspace(P2.min(), P2.max(), 50)
    ax.plot(xs, b0 + b1 * xs, color="#e67e22", lw=1.4, ls="--", zorder=2)
    ax.axvline(thr_in, color="#c0392b", lw=1, ls=":",
               label=f"e062 in-sample thr {thr_in:.4f}")
    ax.axvline(primary["threshold"], color="#2471a3", lw=1, ls="--",
               label=f"e076 OOS-fit thr {primary['threshold']:.4f}")
    r_raw, p_raw = pearson(P2, D)
    ax.set_xlabel("P2: stream-cosine at graft-input depths")
    ax.set_ylabel("D = graft damage")
    ax.set_title(f"raw instrument: r = {r_raw:+.3f} (n=204)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # (d) out-of-sample AUC distribution
    ax = axs[1, 0]
    ax.hist(auc_oos, bins=24, color="#27ae60", alpha=0.8)
    ax.axvline(auc_in, color="#7d3c98", lw=1.4, label=f"in-sample AUC {auc_in:.3f}")
    ax.axvline(primary["auc_test"], color="#c0392b", lw=1.4,
               label=f"primary-split OOS {primary['auc_test']:.3f}")
    ax.axvline(0.85, color="k", lw=1.2, ls="--", label="0.85 survival bar")
    ax.set_xlabel(f"out-of-sample AUC over {N_RESPLIT} stratified resplits")
    ax.set_ylabel("splits")
    ax.set_title(f"PART 2: OOS AUC (J median {np.median(j_oos):.3f})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # (e) leave-one-lineage-out
    ax = axs[1, 1]
    ypos = np.arange(len(lolo))
    ax.scatter(rl_a, ypos, marker="o", s=34, color="#7d3c98",
               label="partial r(D,P2 | A)")
    ax.scatter(rl_f, ypos, marker="x", s=42, color="#e67e22",
               label="full-norm partial r")
    ax.axvline(pr_p2, color="#7d3c98", lw=1, ls=":")
    ax.axvline(r_full, color="#e67e22", lw=1, ls=":")
    ax.set_yticks(ypos)
    ax.set_yticklabels([x["left_out_host"] for x in lolo], fontsize=6)
    ax.set_xlabel("partial r with one host lineage left out")
    ax.set_title("PART 2: leave-one-lineage-out (18 folds)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.25, axis="x")

    # (f) dW comparison
    ax = axs[1, 2]
    ax.scatter(dWc, D, s=20, color="#c0392b", alpha=0.65, zorder=3)
    b1, b0 = np.polyfit(dWc, D, 1)
    xs = np.linspace(dWc.min(), dWc.max(), 50)
    ax.plot(xs, b0 + b1 * xs, color="#e67e22", lw=1.4, ls="--", zorder=2)
    ax.set_xlabel("dWcos(host, donor) — e052/e059 dW-alignment, pair-level")
    ax.set_ylabel("D")
    ax.set_title(f"PART 3: dW axis, r = {r_dw:+.3f} | partial (D|A) "
                 f"{pr_dw:+.3f} vs P2 {pr_p2:+.3f}")
    ax.grid(alpha=0.25)
    txt = (f"AUC: P2 {auc_in:.3f} | dW {auc_dw:.3f}\n"
           f"|partial| margin {margin:+.3f} "
           f"({'novel' if margin >= 0.1 else 'not novel at 0.1 bar'})")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=8,
            va="top", ha="left",
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    fig.suptitle("E076 — cosine crossmatch: mechanism discriminator (T046) + "
                 "critic fixes (T047 claim D) — 204 cached pairs, CPU\n"
                 f"P1 {v1.split(':')[0]} | P2 {v2.split(';')[0]} | "
                 f"P3 {v3.split(':')[-1].strip()}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(rd / "cosine_mechanism.png", dpi=140)
    plt.close(fig)
    log(f"outputs: {rd} (metrics.json, cosine_mechanism.png)")
    log("=" * 78)
    for k in ("part1_mechanism", "part2_oos", "part3_dw"):
        log(f"VERDICT {k}: {verdict[k]}")
    return metrics


if __name__ == "__main__":
    main()
