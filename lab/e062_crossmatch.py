"""E062 — PRE-GRAFT COMPATIBILITY (CROSSMATCH) PREDICTOR TEST.

Registered design (R34 decision 2 + T040): transplant medicine needs a cheap
pre-graft crossmatch. Two zero-training candidates, tested IN THIS ORDER
(first to clear the bar wins the "cheap crossmatch" title):

  P1  W_out row-norm mean at graft sites (HOST trait; e059's A-independent
      interface signal: partial r(D|A-resid) = -0.654, p=.040, n=10 hosts
      with frozen donor e040_ref).
  P2  Pre-graft stream-cosine (original e062 idea): cosine between host and
      donor residual-stream activations on a small shared probe batch,
      per depth; aggregate = mean cosine at the graft-input depths.

Dependent variable D (e040/e052-family measure VERBATIM — the one e059
audited): mean over graft sites {L2, L3} of dCE(host <- donor mlp-Lsite),
15 fixed val batches (e052 stream), i.e. transplant donor's full mlp organ
at site s, eval, restore, dCE vs host base, then average the two sites.
Covariate A (organ-reliance, e063's method): host's OWN-organ ablation dCE
(zero own mlp at site s), mean over the same sites/batches.

Pair inventory (all same-arch checkpoints available; e033 excluded — broken
without its eval-time equalizer hook; e053c_ctx512 excluded — block_size 512):
  Scale A (0.84M, 4L/4H/128d): 12-net donor pool
      {e040_ref, e040_w, e005s_small, m1, m2, m3, g1a, g1b, g1c, g2a, g2b, g2c}
      x 18 hosts (pool + 6 e050 directed-mutation nets as HOSTS ONLY),
      host != donor  ->  204 measured pairs. Includes the 10 recorded
      e059 pairs (member <- e040_ref) as a bitwise reproduction gate, and
      e058's recorded scaleB cells as a B-cohort gate.
  Scale B (2.7M, 6L/6H/192d): {e001, e028_b43, e041_bdo} full square minus
      self -> 6 pairs (secondary replication panel; no bars claimed at n=6).
  Recorded-but-not-reused pairs (noted for honesty): e050's own D table used
      30 fixed batches (different instrument length); e031 grafted single
      matrices not organs. Both excluded from fits.

REGISTERED BARS (frozen before running): a predictor wins only if
  |partial r(D, P | A)| >= 0.5  AND  host-cluster bootstrap 95% CI of the
  partial r excludes 0  AND  n_pairs >= 20.
P1 is tested first; if P1 clears, P1 is the cheap crossmatch; else if P2
clears, P2 wins; else NEITHER clears -> the cheap-predictor hope is dead and
P2 (program) pivots to engineered/trained tolerance.

Deliverables: decision rule (threshold on the winner, Youden J) with
ROC/AUC on the available pairs for label "damaging graft" = D > median(D).

Run: python lab/e062_crossmatch.py   (CPU-only; CUDA masked before torch)
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
from common import (REPO, Cfg, CharCorpus, TinyGPT, lesion, run_dir,
                    save_json, set_seed)

# ---------------------------------------------------------------- protocol
CORPUS_PATH = REPO / "data" / "input.txt"
CKPT_DIR = REPO / "runs" / "checkpoints"
E059_METRICS = REPO / "runs" / "e059" / "metrics.json"
E058_METRICS = REPO / "runs" / "e058" / "metrics.json"
SITES = (2, 3)                                     # e059 graft sites verbatim
N_EVAL = 15                                        # e052 fixed-batch protocol
N_BOOT_CLUSTER = 2000
SEED = 62

POOL_A = ["e040_ref", "e040_w", "e005s_small",
          "e040_m1", "e040_m2", "e040_m3",
          "e040_g1a", "e040_g1b", "e040_g1c",
          "e040_g2a", "e040_g2b", "e040_g2c"]
HOSTS_EXTRA = ["e050_m1", "e050_m2", "e050_m3",
               "e050_g1a", "e050_g1b", "e050_g1c"]   # hosts only (same arch)
POOL_B = ["e001", "e028_b43", "e041_bdo"]          # 2.7M secondary panel

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- surgery (e059 verbatim, CPU)
def mlp_keys(i: int) -> list[str]:
    return [f"h.{i}.mlp.0.weight", f"h.{i}.mlp.0.bias",
            f"h.{i}.mlp.2.weight", f"h.{i}.mlp.2.bias"]


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


def load_sd(name: str) -> dict:
    obj = torch.load(CKPT_DIR / f"{name}.pt", map_location="cpu",
                     weights_only=False)
    return obj["model"] if "model" in obj else obj


@torch.no_grad()
def per_batch_losses_cpu(model, corpus, n_batches: int = N_EVAL) -> list[float]:
    """e028/e040/e052/e059 fixed-batch protocol: RNG stream seeded 1337."""
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


@torch.no_grad()
def probe_streams(model, corpus, n_batches: int = 2) -> list[torch.Tensor]:
    """Token-normalized residual streams on a shared probe batch.

    Depths: d=0 embedding output (wte+wpe); d=i stream after block i-1.
    The stream entering block s is xs[s]; the organ grafted at site s reads
    (approximately) that stream, so graft-input depths for SITES=(2,3) are
    d=2 and d=3.
    """
    model.eval()
    cfg = model.cfg
    gen = torch.Generator().manual_seed(corpus.seed)   # same fixed stream
    src = corpus.val
    ix = torch.randint(len(src) - cfg.block_size - 1, (16 * n_batches,),
                       generator=gen)
    acc: list[torch.Tensor] | None = None
    for b in range(n_batches):
        x = torch.stack([src[i: i + cfg.block_size]
                         for i in ix[b * 16: (b + 1) * 16]])
        pos = torch.arange(cfg.block_size)
        s = model.wte(x) + model.wpe(pos)
        outs = [s]
        for block in model.h:
            s = block(s)
            outs.append(s)
        nrm = [t / t.norm(dim=-1, keepdim=True).clamp_min(1e-8) for t in outs]
        flat = [t.reshape(-1, t.shape[-1]) for t in nrm]
        if acc is None:
            acc = flat
        else:
            acc = [torch.cat([a, f], dim=0) for a, f in zip(acc, flat)]
    model.train()
    return acc


def mean(xs) -> float:
    return float(sum(xs) / len(xs))


# ---------------------------------------------------------------- stats helpers
def pearson(x, y):
    r, p = sstats.pearsonr(x, y)
    return float(r), float(p)


def fisher_ci(r: float, n: int, k_cov: int = 1) -> tuple[float, float, float]:
    """95% Fisher-z CI for a (partial, k_cov=1) correlation, n observations."""
    if n - k_cov - 2 <= 0:
        return float("nan"), float("nan"), float("nan")
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - k_cov - 2)
    lo, hi = np.tanh(z - 1.959964 * se), np.tanh(z + 1.959964 * se)
    p = float(2 * sstats.norm.sf(abs(z) * np.sqrt(n - k_cov - 2)))
    return float(lo), float(hi), p


def ols_resid(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones_like(z), z])
    beta, *_ = np.linalg.lstsq(X, x, rcond=None)
    return x - X @ beta


def partial_r(x, y, z) -> float:
    """Proper partial correlation of x,y given z (both residualized on z)."""
    x, y, z = (np.asarray(v, dtype=float) for v in (x, y, z))
    if (x.shape == z.shape and np.allclose(x, z)) or \
       (y.shape == z.shape and np.allclose(y, z)):
        return 0.0                              # r(., z | z) == 0 by definition
    rx, ry = ols_resid(x, z), ols_resid(y, z)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def cluster_boot_partial(d, p, a, clusters, n: int = N_BOOT_CLUSTER,
                         seed: int = 62) -> list[float]:
    """Bootstrap CI resampling CLUSTERS (hosts, and donors as sensitivity)."""
    d, p, a, cl = (np.asarray(v) for v in (d, p, a, clusters))
    uniq = np.unique(cl)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([np.where(cl == c)[0] for c in pick])
        if len(np.unique(cl[idx])) < 4:
            continue
        r = partial_r(d[idx], p[idx], a[idx])
        if np.isfinite(r):
            out.append(r)
    return out


def auc_roc(score, label) -> tuple[float, np.ndarray, np.ndarray]:
    """AUC + ROC curve for a COMPATIBILITY score vs label 1 = damaging."""
    score, label = np.asarray(score, float), np.asarray(label, int)
    order = np.argsort(-score)                      # high score -> low D
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # tie-aware ranks
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
    auc = float((ranks[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    ths = np.unique(np.concatenate([[-np.inf], score, [np.inf]]))
    fpr, tpr = [], []
    for t in ths:
        pred = score >= t                            # score>=t -> predict OK
        fpr.append(float(np.sum(pred & (label == 1)) / max(n1, 1)))
        tpr.append(float(np.sum(pred & (label == 0)) / max(n0, 1)))
    return auc, np.array(fpr), np.array(tpr)


def youden(score, label) -> tuple[float, float]:
    """Best threshold (max Youden J) for a compatibility score."""
    score, label = np.asarray(score, float), np.asarray(label, int)
    best_t, best_j = float("nan"), -1.0
    for t in np.unique(score):
        pred_ok = score >= t
        n1, n0 = max(label.sum(), 1), max((1 - label).sum(), 1)
        tpr = np.sum(pred_ok & (label == 0)) / n0   # correctly waved through
        fpr = np.sum(pred_ok & (label == 1)) / n1   # damaging waved through
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = float(j), float(t)
    return best_t, best_j


# ---------------------------------------------------------------- main
def main():
    rd = run_dir("e062")
    cache_path = rd / "cache.json"
    log("E062 crossmatch — pre-graft compatibility predictors (CPU-only)")
    set_seed(SEED)
    corpus = CharCorpus(CORPUS_PATH, seed=1337)

    cfg_a = Cfg(vocab=corpus.vocab_size, n_layer=4, n_head=4, n_embd=128,
                block_size=256)
    cfg_b = Cfg(vocab=corpus.vocab_size, n_layer=6, n_head=6, n_embd=192,
                block_size=256)
    net_a = TinyGPT(cfg_a)
    assert net_a.num_params() == 840_704
    net_b = TinyGPT(cfg_b)

    hosts_all = POOL_A + HOSTS_EXTRA                  # 18 scale-A hosts
    sds = {n: load_sd(n) for n in set(hosts_all + POOL_B)}

    cache = json.loads(cache_path.read_text(encoding="utf-8")) \
        if cache_path.exists() else {"base": {}, "ablate": {}, "graft": {}}
    dirty = False

    # ---- per-host instrument pass: base, own-ablation A, interface axes ----
    host_rec: dict[str, dict] = {}
    streams: dict[str, list[torch.Tensor]] = {}
    for name in hosts_all:
        net_a.load_state_dict(sds[name])
        snap = snapshot(net_a)
        if name not in cache["base"]:
            cache["base"][name] = per_batch_losses_cpu(net_a, corpus)
            dirty = True
        base = cache["base"][name]
        rec = {"name": name, "base_ce": mean(base)}
        if name not in cache["ablate"]:
            cache["ablate"][name] = {}
            for s in SITES:
                with lesion(net_a, "mlp", s):
                    abl = per_batch_losses_cpu(net_a, corpus)
                cache["ablate"][name][str(s)] = mean(
                    [a - b for a, b in zip(abl, base)])
                dirty = True
        assert_identical(net_a, snap, name)
        rec["ablate"] = {s: cache["ablate"][name][str(s)] for s in SITES}
        rec["A"] = mean([rec["ablate"][s] for s in SITES])
        # interface axes (e059 definitions verbatim)
        rowm = []
        for s in SITES:
            w_out = sds[name][f"h.{s}.mlp.2.weight"].float()
            w_in = sds[name][f"h.{s}.mlp.0.weight"].float()
            rowm.append(float(w_out.norm(dim=1).mean()))
            rec[f"win_rowmean_L{s}"] = float(w_in.norm(dim=1).mean())
        rec["P1_wout_rowmean_sites"] = mean(rowm)
        host_rec[name] = rec
        streams[name] = probe_streams(net_a, corpus)
        log(f"  host {name:13s} CE {rec['base_ce']:.4f} | A {rec['A']:+.4f} | "
            f"P1 {rec['P1_wout_rowmean_sites']:.4f}")

    # donor-side axes (secondary columns)
    for name in POOL_A:
        rowm = [float(sds[name][f"h.{s}.mlp.0.weight"].float().norm(dim=1).mean())
                for s in SITES]
        rowo = [float(sds[name][f"h.{s}.mlp.2.weight"].float().norm(dim=1).mean())
                for s in SITES]
        host_rec[name]["donor_win_rowmean_sites"] = mean(rowm)
        host_rec[name]["donor_wout_rowmean_sites"] = mean(rowo)

    # ---- instrument gate: REF self-graft exactly zero ------------------------
    net_a.load_state_dict(sds["e040_ref"])
    snap = snapshot(net_a)
    base_ref = cache["base"]["e040_ref"]
    transplant(net_a, sds["e040_ref"], mlp_keys(SITES[0]))
    self_losses = per_batch_losses_cpu(net_a, corpus)
    net_a.load_state_dict(snap)
    ref_self_zero = bool(self_losses == base_ref)
    log(f"gate: REF self-graft dCE exactly 0.0 -> {ref_self_zero}")

    # ---- pair matrix (graft evals), cached -----------------------------------
    pairs = [(h, d) for h in hosts_all for d in POOL_A if h != d]
    log(f"measuring {len(pairs)} scale-A host<->donor pairs "
        f"({len(SITES)} transplant evals each) ...")
    for i, (h, d) in enumerate(pairs):
        key = f"{h}|{d}"
        if key in cache["graft"]:
            continue
        net_a.load_state_dict(sds[h])
        snap = snapshot(net_a)
        base = cache["base"][h]
        cell = {}
        for s in SITES:
            transplant(net_a, sds[d], mlp_keys(s))
            losses = per_batch_losses_cpu(net_a, corpus)
            net_a.load_state_dict(snap)
            assert_identical(net_a, snap, h)
            cell[str(s)] = mean([a - b for a, b in zip(losses, base)])
        cache["graft"][key] = cell
        dirty = True
        if (i + 1) % 24 == 0:
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
            log(f"  {i + 1}/{len(pairs)} pairs done ({h} <- {d} last)")
    if dirty:
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    log("scale-A pair matrix complete")

    # ---- assemble scale-A pair table -----------------------------------------
    def cosdepth(h, d, depth) -> float:
        a, b = streams[h][depth], streams[d][depth]
        return float((a * b).sum(-1).mean())

    rows = []
    for h, d in pairs:
        cell = cache["graft"][f"{h}|{d}"]
        D = mean([cell[str(s)] for s in SITES])
        cos = {f"cos_d{k}": cosdepth(h, d, k) for k in range(5)}
        rows.append({
            "host": h, "donor": d,
            "D": D, "D_sites": {str(s): cell[str(s)] for s in SITES},
            "A_host": host_rec[h]["A"],
            "P1_wout_rowmean_sites": host_rec[h]["P1_wout_rowmean_sites"],
            **cos,
            "cos_graftinput": mean([cos["cos_d2"], cos["cos_d3"]]),
            "cos_all": mean([cos[f"cos_d{k}"] for k in range(5)]),
            "donor_win_rowmean_sites": host_rec[d]["donor_win_rowmean_sites"],
            "donor_wout_rowmean_sites": host_rec[d]["donor_wout_rowmean_sites"],
        })
    n_pairs = len(rows)
    Dv = np.array([r["D"] for r in rows])
    Av = np.array([r["A_host"] for r in rows])
    P1 = np.array([r["P1_wout_rowmean_sites"] for r in rows])
    P2gi = np.array([r["cos_graftinput"] for r in rows])
    P2all = np.array([r["cos_all"] for r in rows])
    hcl = np.array([r["host"] for r in rows])
    dcl = np.array([r["donor"] for r in rows])

    # ---- scale-B secondary panel (n=6, no bars claimed) ----------------------
    net_b_state: dict[str, dict] = {}
    b_streams: dict[str, list[torch.Tensor]] = {}
    for name in POOL_B:
        net_b.load_state_dict(sds[name])
        net_b_state[name] = {"base": per_batch_losses_cpu(net_b, corpus)}
        snap = snapshot(net_b)
        abl = {}
        for s in SITES:
            with lesion(net_b, "mlp", s):
                a = per_batch_losses_cpu(net_b, corpus)
            abl[str(s)] = mean([x - y for x, y in zip(a, net_b_state[name]["base"])])
        assert_identical(net_b, snap, name)
        net_b_state[name]["ablate"] = abl
        net_b_state[name]["A"] = mean([abl[str(s)] for s in SITES])
        net_b_state[name]["P1"] = mean(
            [float(sds[name][f"h.{s}.mlp.2.weight"].float().norm(dim=1).mean())
             for s in SITES])
        b_streams[name] = probe_streams(net_b, corpus)
    b_rows = []
    for h in POOL_B:
        for d in POOL_B:
            if h == d:
                continue
            net_b.load_state_dict(sds[h])
            snap = snapshot(net_b)
            cell = {}
            for s in SITES:
                transplant(net_b, sds[d], mlp_keys(s))
                losses = per_batch_losses_cpu(net_b, corpus)
                net_b.load_state_dict(snap)
                assert_identical(net_b, snap, h)
                cell[str(s)] = mean([a - b for a, b in
                                     zip(losses, net_b_state[h]["base"])])
            cosd = {f"cos_d{k}": float((b_streams[h][k] * b_streams[d][k])
                                       .sum(-1).mean()) for k in range(7)}
            b_rows.append({
                "host": h, "donor": d,
                "D": mean([cell[str(s)] for s in SITES]),
                "D_sites": {str(s): cell[str(s)] for s in SITES},
                "A_host": net_b_state[h]["A"],
                "P1_wout_rowmean_sites": net_b_state[h]["P1"],
                **cosd,
                "cos_graftinput": mean([cosd["cos_d2"], cosd["cos_d3"]]),
                "cos_all": mean(list(cosd.values())),
            })
    log(f"scale-B panel: {len(b_rows)} pairs | D "
        + " ".join(f"{r['host']}<-{r['donor'].split('_')[-1]}:{r['D']:+.3f}"
                   for r in b_rows))

    # ---- reproduction gates ---------------------------------------------------
    e059 = json.loads(E059_METRICS.read_text(encoding="utf-8"))
    e059_by = {m["name"]: m for m in e059["members"]}
    repro = {"max_abs_D_diff": 0.0, "max_abs_A_diff": 0.0,
             "max_abs_P1_diff": 0.0, "n_checked": 0}
    for r in rows:
        if r["donor"] == "e040_ref" and r["host"] in e059_by:
            m = e059_by[r["host"]]
            repro["max_abs_D_diff"] = max(repro["max_abs_D_diff"],
                                          abs(r["D"] - m["D"]))
            repro["max_abs_A_diff"] = max(repro["max_abs_A_diff"],
                                          abs(r["A_host"] - m["A"]))
            repro["max_abs_P1_diff"] = max(repro["max_abs_P1_diff"],
                                           abs(r["P1_wout_rowmean_sites"]
                                               - m["axes"]["wout_rowmean_sites"]))
            repro["n_checked"] += 1
    e058 = json.loads(E058_METRICS.read_text(encoding="utf-8"))
    sb = e058["scaleB"]["graft"]
    bmap = {"e001": "B", "e028_b43": "B43"}
    repro_b = {"max_abs_Dsite_diff": 0.0, "n_checked": 0}
    for r in b_rows:
        h, d = bmap.get(r["host"]), bmap.get(r["donor"])
        if h and d and f"{h}<-{d}|L2" in sb:
            for s in SITES:
                repro_b["max_abs_Dsite_diff"] = max(
                    repro_b["max_abs_Dsite_diff"],
                    abs(r["D_sites"][str(s)] - sb[f"{h}<-{d}|L{s}"]))
            repro_b["n_checked"] += 1
    log(f"gates: vs e059 max|dD| {repro['max_abs_D_diff']:.2e}, max|dA| "
        f"{repro['max_abs_A_diff']:.2e}, max|dP1| {repro['max_abs_P1_diff']:.2e} "
        f"({repro['n_checked']} pairs) | vs e058 scaleB max|dD_site| "
        f"{repro_b['max_abs_Dsite_diff']:.2e} ({repro_b['n_checked']} pairs)")
    gates = {
        "ref_self_transplant_zero": ref_self_zero,
        "hosts_bitwise_restored": True,
        "reproduction_vs_e059": repro,
        "reproduction_vs_e058_scaleB": repro_b,
        "repro_ok": bool(ref_self_zero
                         and repro["max_abs_D_diff"] <= 1e-6
                         and repro["max_abs_A_diff"] <= 1e-6
                         and repro_b["max_abs_Dsite_diff"] <= 1e-6),
    }
    if not gates["repro_ok"]:
        raise SystemExit("REPRODUCTION GATE FAILED (e059/e058 instruments)")

    # ---- correlation table (scale A, pooled pairs) ----------------------------
    table = {}

    def entry(key, v, label):
        r, p = pearson(Dv, v)
        pr = partial_r(Dv, v, Av)
        lo, hi, pp = fisher_ci(pr, n_pairs)
        hc = cluster_boot_partial(Dv, v, Av, hcl)
        dc = cluster_boot_partial(Dv, v, Av, dcl)
        hc_ci = [float(np.percentile(hc, 2.5)), float(np.percentile(hc, 97.5))] \
            if hc else [float("nan")] * 2
        dc_ci = [float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))] \
            if dc else [float("nan")] * 2
        table[key] = {"label": label, "r_D": r, "p_r": p,
                      "partial_r_given_A": pr, "partial_ci95_fisher": [lo, hi],
                      "partial_p": pp,
                      "partial_ci95_hostcluster": hc_ci,
                      "partial_ci95_donorcluster": dc_ci}
        log(f"  {label:34s} r(D)={r:+.3f} (p={p:.1e}) | "
            f"partial r(D|A)={pr:+.3f} fisher [{lo:+.3f},{hi:+.3f}] "
            f"p={pp:.1e} | hostCl [{hc_ci[0]:+.3f},{hc_ci[1]:+.3f}] "
            f"donorCl [{dc_ci[0]:+.3f},{dc_ci[1]:+.3f}]")

    entry("A", Av, "own-organ load A (covariate/reference)")
    entry("P1", P1, "P1: host W_out row-norm mean (graft sites)")
    entry("P2_gi", P2gi, "P2: stream-cosine, graft-input depths (d2,d3)")
    entry("P2_all", P2all, "P2: stream-cosine, all depths")
    for k in range(5):
        entry(f"cos_d{k}", np.array([r[f"cos_d{k}"] for r in rows]),
              f"stream-cosine depth d{k}"
              f"{' (graft input)' if k in SITES else ''}")
    entry("donor_win", np.array([r["donor_win_rowmean_sites"] for r in rows]),
          "secondary: donor W_in row-norm mean (graft sites)")

    # host-level fit for P1 (host trait: mean D over donors per host)
    hosts_present = sorted(set(hcl))
    Dbar = np.array([Dv[hcl == h].mean() for h in hosts_present])
    P1h = np.array([host_rec[h]["P1_wout_rowmean_sites"] for h in hosts_present])
    Ah = np.array([host_rec[h]["A"] for h in hosts_present])
    r_h, p_h = pearson(Dbar, P1h)
    pr_h = partial_r(Dbar, P1h, Ah)
    lo_h, hi_h, pp_h = fisher_ci(pr_h, len(hosts_present))
    host_level = {"n_hosts": len(hosts_present), "r_Dbar_P1": r_h, "p": p_h,
                  "partial_r_given_A": pr_h,
                  "partial_ci95_fisher": [lo_h, hi_h], "partial_p": pp_h}
    log(f"  host-level (n={len(hosts_present)}): r(Dbar, P1)={r_h:+.3f} "
        f"(p={p_h:.3f}) | partial given A={pr_h:+.3f} "
        f"[{lo_h:+.3f},{hi_h:+.3f}]")

    # ---- decision rule + ROC/AUC ----------------------------------------------
    label = (Dv > np.median(Dv)).astype(int)          # 1 = damaging graft
    lab_def = {"rule": "D > median(D) over all scale-A pairs",
               "median_D": float(np.median(Dv)), "n_pos": int(label.sum()),
               "n_neg": int((1 - label).sum())}
    rocs = {}
    for key, score, orient in (
            ("P1", P1, "higher = compatible"),
            ("P2_gi", P2gi, "higher = compatible"),
            ("A", -Av, "lower A = compatible")):
        auc, fpr, tpr = auc_roc(score, label)
        thr, j = youden(score, label)
        rocs[key] = {"auc": auc, "orientation": orient,
                     "youden_threshold": thr, "youden_J": j,
                     "fpr": fpr.tolist(), "tpr": tpr.tolist()}
        log(f"  ROC {key:5s} AUC={auc:.3f} ({orient}) | Youden thr {thr:.4f} "
            f"J={j:.3f}")

    # ---- registered verdict (order: P1 first, then P2) ------------------------
    def clears(key):
        t = table[key]
        return (abs(t["partial_r_given_A"]) >= 0.5
                and t["partial_ci95_hostcluster"][0] * t["partial_ci95_hostcluster"][1] > 0
                and n_pairs >= 20)

    p1_ok, p2_ok = clears("P1"), clears("P2_gi")
    if p1_ok:
        winner = "P1"
        verdict = (f"P1 WINS the cheap crossmatch: host W_out row-norm mean at "
                   f"graft sites has partial r(D|A) "
                   f"{table['P1']['partial_r_given_A']:+.3f} "
                   f"(host-cluster CI {table['P1']['partial_ci95_hostcluster']}) "
                   f"over {n_pairs} pairs; decision rule: graft only if P1 "
                   f">= {rocs['P1']['youden_threshold']:.4f} "
                   f"(AUC {rocs['P1']['auc']:.3f}, Youden J "
                   f"{rocs['P1']['youden_J']:.3f} vs label {lab_def['rule']})")
    elif p2_ok:
        winner = "P2"
        verdict = (f"P2 WINS the cheap crossmatch: pre-graft stream-cosine "
                   f"(graft-input depths) partial r(D|A) "
                   f"{table['P2_gi']['partial_r_given_A']:+.3f} "
                   f"(host-cluster CI {table['P2_gi']['partial_ci95_hostcluster']}) "
                   f"over {n_pairs} pairs; decision rule: graft only if "
                   f"cos_graftinput >= {rocs['P2_gi']['youden_threshold']:.4f} "
                   f"(AUC {rocs['P2_gi']['auc']:.3f}, Youden J "
                   f"{rocs['P2_gi']['youden_J']:.3f})")
    else:
        winner = "none"
        verdict = (f"NEITHER predictor clears the registered bar "
                   f"(|partial r(D|A)| >= 0.5 with host-cluster CI excluding 0 "
                   f"over >= 20 pairs): P1 partial "
                   f"{table['P1']['partial_r_given_A']:+.3f} "
                   f"CI {table['P1']['partial_ci95_hostcluster']}; P2 partial "
                   f"{table['P2_gi']['partial_r_given_A']:+.3f} CI "
                   f"{table['P2_gi']['partial_ci95_hostcluster']} "
                   f"(n={n_pairs} pairs). The cheap-predictor hope is dead at "
                   f"this instrument; P2 (program) pivots to engineered/"
                   f"trained tolerance.")
    log("=" * 78)
    log(f"VERDICT: {verdict}")

    # ---- scale-B secondary correlations (report only) --------------------------
    bD = np.array([r["D"] for r in b_rows])
    bA = np.array([r["A_host"] for r in b_rows])
    bP1 = np.array([r["P1_wout_rowmean_sites"] for r in b_rows])
    bP2 = np.array([r["cos_graftinput"] for r in b_rows])
    b_stats = {"n_pairs": len(b_rows),
               "note": "2.7M cohort, n=6 directional pairs — no bars claimed"}
    for nm, v in (("A", bA), ("P1", bP1), ("P2_gi", bP2)):
        b_stats[f"r_D_{nm}"] = pearson(bD, v)[0]
        b_stats[f"partial_r_given_A_{nm}"] = partial_r(bD, v, bA)
    log("scale-B secondary: " + " | ".join(
        f"{k}={v:+.3f}" for k, v in b_stats.items()
        if isinstance(v, float)))

    # ---- outputs ----------------------------------------------------------------
    metrics = {
        "experiment": "e062_crossmatch",
        "date": common.now_iso(),
        "status": "complete",
        "device": "cpu-only (CUDA_VISIBLE_DEVICES masked; zero-GPU analysis)",
        "question": ("is there a cheap pre-graft crossmatch: does host W_out "
                     "row-norm mean at graft sites (P1, e059's interface "
                     "signal) or pre-graft stream-cosine (P2) predict graft "
                     "damage D across host x donor pairs, partial on A?"),
        "registered_design": {
            "D": "mean over graft sites {2,3} of dCE(host<-donor mlp-Lsite), "
                 "15 fixed e052 batches (e059-verbatim)",
            "A": "host own-organ ablation dCE, same sites/batches (e063 method)",
            "P1": "host W_out row-norm mean at graft sites (e059 axis)",
            "P2": "mean token-cosine of host/donor residual streams on shared "
                  "2-batch probe, per depth; graft-input aggregate = d2,d3",
            "test_order": ["P1", "P2"],
            "bars": "winner: |partial r(D|A)| >= 0.5 AND host-cluster "
                    "bootstrap 95% CI excludes 0 AND n_pairs >= 20",
            "outcome_if_neither": "cheap predictor dead; P2 program pivots to "
                                  "trained tolerance"},
        "pair_inventory": {
            "scale_A": {"donor_pool": POOL_A, "hosts": hosts_all,
                        "hosts_note": "12-net pool full square + 6 e050 "
                                      "directed-mutation nets as hosts only",
                        "n_pairs": n_pairs},
            "scale_B": {"nets": POOL_B, "n_pairs": len(b_rows)},
            "excluded": ["e033 (equalizer hook required)", 
                         "e053c_ctx512 (block_size 512)",
                         "e050 recorded D (30-batch instrument, differs)",
                         "e031 (single-matrix grafts, not organs)"]},
        "gates": gates,
        "label_definition": lab_def,
        "hosts": [{"name": h, **{k: v for k, v in host_rec[h].items()
                                 if k != "name"}} for h in hosts_all],
        "pairs": rows,
        "pairs_scaleB": b_rows,
        "correlations_scaleA": table,
        "host_level_P1": host_level,
        "roc": rocs,
        "scale_B_secondary": b_stats,
        "winner": winner,
        "verdict": verdict,
        "timing_s": round(time.time() - T0, 1),
    }
    save_json(rd / "metrics.json", metrics)

    # ---- figure ------------------------------------------------------------------
    fig, axs = plt.subplots(2, 3, figsize=(17, 10))
    C_REF, C_OTH, C_HOST = "#c0392b", "#2c6fbb", "#222222"

    def scat(ax, x, y, lab, refmask=None):
        if refmask is None:
            ax.scatter(x, y, s=22, color=C_OTH, alpha=0.65, zorder=3)
        else:
            ax.scatter(x[~refmask], y[~refmask], s=20, color=C_OTH, alpha=0.6,
                       zorder=3, label="donor != e040_ref")
            ax.scatter(x[refmask], y[refmask], s=30, color=C_REF, alpha=0.8,
                       zorder=4, label="donor = e040_ref (e059 column)")
        if np.std(x) > 0:
            b1, b0 = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, b0 + b1 * xs, color="#e67e22", lw=1.4, ls="--",
                    zorder=2)
        r, p = pearson(x, y)
        ax.set_xlabel(lab)
        ax.set_ylabel("D = graft damage (mean dCE, sites L2/L3, 15 batches)")
        ax.set_title(f"r = {r:+.3f} (p={p:.1e}), n={len(x)} pairs")
        ax.grid(alpha=0.25)
        if refmask is not None:
            ax.legend(fontsize=7)

    scat(axs[0, 0], P1, Dv, "P1: host W_out row-norm mean (graft sites)",
         refmask=np.array([r["donor"] == "e040_ref" for r in rows]))
    # host-level means overlay
    axs[0, 0].scatter(P1h, Dbar, marker="X", s=90, color=C_HOST, zorder=5,
                      label=f"host means (n={len(hosts_present)})")
    axs[0, 0].legend(fontsize=7)
    scat(axs[0, 1], P2gi, Dv,
         "P2: stream-cosine at graft-input depths (d2,d3)")
    scat(axs[0, 2], Av, Dv, "own-organ load A (reference predictor)")

    # per-depth cosine correlations
    ax = axs[1, 0]
    ks = list(range(5))
    rs_, los_, his_ = [], [], []
    for k in ks:
        t = table[f"cos_d{k}"]
        lo, hi, _ = fisher_ci(t["r_D"], n_pairs, k_cov=0)
        rs_.append(t["r_D"])
        los_.append(t["r_D"] - lo)
        his_.append(hi - t["r_D"])
    cols = ["#8e44ad" if k in SITES else "#95a5a6" for k in ks]
    ax.bar(ks, rs_, color=cols)
    ax.errorbar(ks, rs_, yerr=[los_, his_], fmt="none", ecolor="#333",
                capsize=3)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(ks)
    ax.set_xticklabels([f"d{k}{'*' if k in SITES else ''}" for k in ks])
    ax.set_xlabel("residual-stream depth (* = graft input)")
    ax.set_ylabel("r(D, cos_dk)")
    ax.set_title("Per-depth stream-cosine vs damage (95% Fisher CI)")
    ax.grid(alpha=0.25, axis="y")

    # ROC curves
    ax = axs[1, 1]
    for key, colr in (("P1", "#2c6fbb"), ("P2_gi", "#27ae60"), ("A", "#c0392b")):
        ax.plot(rocs[key]["fpr"], rocs[key]["tpr"], color=colr, lw=1.6,
                label=f"{key} (AUC {rocs[key]['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="#888", lw=0.8, ls=":")
    ax.set_xlabel("FPR: damaging grafts waved through")
    ax.set_ylabel("TPR: compatible grafts waved through")
    ax.set_title(f"ROC, label = {lab_def['rule']}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # partial-r summary
    ax = axs[1, 2]
    keys = ["A", "P1", "P2_gi", "P2_all", "donor_win"]
    labs = ["A (reference)", "P1 W_out rowmean", "P2 cosine (d2,d3)",
            "P2 cosine (all)", "donor W_in rowmean"]
    prs = [table[k]["partial_r_given_A"] for k in keys]
    ci = [table[k]["partial_ci95_hostcluster"] for k in keys]
    yerr = [[p - c[0] for p, c in zip(prs, ci)], [c[1] - p for p, c in zip(prs, ci)]]
    ypos = np.arange(len(keys))[::-1]
    ax.barh(ypos, prs, color=["#c0392b", "#2c6fbb", "#27ae60", "#7dcea0",
                              "#95a5a6"])
    ax.errorbar(prs, ypos, xerr=yerr, fmt="none", ecolor="#111", capsize=3)
    ax.axvline(0, color="#333", lw=0.8)
    ax.axvline(0.5, color="#e67e22", lw=1, ls="--", label="bar |r|=0.5")
    ax.axvline(-0.5, color="#e67e22", lw=1, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labs, fontsize=8)
    ax.set_xlabel("partial r(D | A) — whiskers: host-cluster bootstrap 95% CI")
    ax.set_title("Crossmatch candidates (pooled pairs)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, axis="x")

    fig.suptitle("E062 — pre-graft crossmatch: can damage D be predicted "
                 f"before the graft? ({n_pairs} scale-A pairs + 6 scale-B, CPU)\n"
                 f"WINNER: {winner}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(rd / "crossmatch.png", dpi=140)
    plt.close(fig)
    log(f"outputs: {rd} (metrics.json, crossmatch.png)")
    return metrics


if __name__ == "__main__":
    main()
