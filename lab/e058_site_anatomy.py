"""E058 — site anatomy of the geometry-damage coupling: is it L2, or the
pre-causal-gate stream-writer, or noise? (CPU-only, zero-GPU, no training.)

Extends T026's per-site nuance: on the e040 0.84M lineage, graft damage
correlates r=+0.747 with dW-alignment distance at site L2 but ~0 at L3 —
why does ONE site carry basis-structure and its neighbor not? Three live
hypotheses:

  H-role  — structure lives at the site just BEFORE the causal gate (the
            last stream-writer before the point-of-no-return), at any scale.
  H-depth — it is L2 specifically (a small-net artifact: L2 is pre-gate on
            the 4L net only by coincidence of depth indexing).
  H-none  — the e052 L2 signal was single-lineage noise and replicates
            nowhere.

Causal-gate anchors (published, frozen):
  0.84M small 4L/4H/128d  causal_mode=3 (e005s l1_gate hist [222,65,329,520])
  2.7M  B     6L/6H/192d  causal_mode=3 (e018; e012d reproduces)
  2.7M  B43   6L/6H/192d  causal_mode=4 (e012d)  <- the ONLY cell that
  10M   large 8L/8H/320d  causal_mode=2 (e005s)     discriminates H-role
                                                     vs H-depth: B43's pre-gate
  site is L3, not L2. B's own gate (3, pre-gate L2) cannot discriminate.

Design:
  Scale A (0.84M, e040 family): 10 hosts, donor e040_ref, ALL MLP sites
    L0-L3. Per host/site: graft dce (15 fixed batches, e028/e040/e052
    protocol), own-ablation dce A, dW-alignment distance 1-cos(dW_host,
    dW_REF), W-space distance 1-mean col-cos (W_in, W_out). Per site:
    r(D_s, predictor_s) over the 10 hosts. L2/L3 cells must reproduce
    runs/e052/metrics.json per-batch (bitwise gate).
  Scale B (2.7M): B=e001 (seed 42) vs B43=e028_b43 (seed 43), BOTH
    directions x ALL 6 MLP sites = 12 grafts. Per graft: dce, alignment
    distance 1-cos(dW_host, dW_donor), W-space distance, own-ablation A,
    interference ratio R = D/A. dW cross-check vs runs/e029 dw_alignment
    (B<->B43 mlp L0/L3/L5, bitwise); B<-B43 graft per-batch cross-check vs
    runs/e028 'within|L{s}|mlp' first 15 (GPU-eval'd there: 2e-3 tol).

REGISTERED VERDICT (frozen before running):
  replicated_at_depth := pooled-over-12-grafts partial r(D, geometry | A)
                         >= +0.5 for geometry in {align-dist, wspace-dist}
                         OR per-direction cross-site Spearman(D_s, geom_s)
                         >= +0.6 in both directions.
  H-role  := replicated AND the per-direction R-ladder peak sits at
             gate(host)-1 for BOTH hosts (B: L2, B43: L3).
  H-depth := replicated AND the R-ladder peaks at L2 in BOTH directions
             (i.e. also on the B43 host whose pre-gate is L3).
  H-none  := not replicated (and scale-A L2 selectivity fails to extend).
  Anything else -> MIXED, with the B43-direction cell called out.

Registered predictions (before running):
  P1: scale-A r_align(L2) ~ 0.75 (same data as e052); L0 and L1 < 0.4.
  P2: scale-B pooled partial r(D, wspace | A) >= 0.5 — cross-seed damage
      still tracks basis geometry at depth.
  P3: R-ladder peaks at pre-gate sites (B: L2, B43: L3).

Run: python lab/e058_site_anatomy.py   (CPU-only; CUDA masked before torch)
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # zero-GPU anatomy

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as sstats

import common

common.DEVICE = "cpu"                              # everything on CPU
from common import (REPO, Cfg, CharCorpus, TinyGPT, lesion, run_dir,
                    save_json, set_seed)

# ---------------------------------------------------------------- protocol
CORPUS_PATH = REPO / "data" / "input.txt"
CKPT_DIR = REPO / "runs" / "checkpoints"
E052_METRICS = REPO / "runs" / "e052" / "metrics.json"
E028_METRICS = REPO / "runs" / "e028" / "metrics.json"
E029_METRICS = REPO / "runs" / "e029" / "metrics.json"
E005S_METRICS = REPO / "runs" / "e005s" / "metrics.json"

N_EVAL = 15                                        # e052 protocol batches
SIGMA_MUT = 0.005
SEED_W, SEED_REF = 42, 4304
MEMBERS = ["e040_w", "e040_m1", "e040_m2", "e040_m3",
           "e040_g1a", "e040_g1b", "e040_g1c",
           "e040_g2a", "e040_g2b", "e040_g2c"]
SITES_A = (0, 1, 2, 3)                             # all MLP sites, 4L net

CFG_B = dict(n_layer=6, n_head=6, n_embd=192)      # 2.7M anatomy (B / B43)
SITES_B = (0, 1, 2, 3, 4, 5)

# causal-gate anchors (published; frozen in the header)
GATE = {"small_0.84M": {"n_layer": 4, "causal_mode": 3, "source": "e005s l1_gate"},
        "B_2.7M": {"n_layer": 6, "causal_mode": 3, "source": "e018 + e012d"},
        "B43_2.7M": {"n_layer": 6, "causal_mode": 4, "source": "e012d"},
        "large_10M": {"n_layer": 8, "causal_mode": 2, "source": "e005s l1_gate"}}
PREGATE = {k: v["causal_mode"] - 1 for k, v in GATE.items()}

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- surgery (e052/e028 verbatim, CPU)
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


@torch.no_grad()
def per_batch_losses_cpu(model, corpus, n_batches: int = N_EVAL) -> list[float]:
    """e028/e040/e052 fixed-batch protocol (same RNG stream, seed=corpus.seed)."""
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


def mean_col_cos(A: torch.Tensor, B: torch.Tensor) -> float:
    an = A / A.norm(dim=0, keepdim=True).clamp_min(1e-12)
    bn = B / B.norm(dim=0, keepdim=True).clamp_min(1e-12)
    return float((an * bn).sum(dim=0).mean())


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


# ---------------------------------------------------------------- stats
def pearson(x, y):
    r, p = sstats.pearsonr(x, y)
    return float(r), float(p)


def spearman(x, y):
    r, p = sstats.spearmanr(x, y)
    return float(r), float(p)


def z(v):
    v = np.asarray(v, dtype=np.float64)
    sd = v.std(ddof=0)
    return (v - v.mean()) / (sd if sd > 0 else 1.0)


def partial_r(y, x1, x2):
    """Partial correlation of y and x1 controlling x2 (residualize both)."""
    rx = z(x2) @ z(x1) / len(y)
    ry = z(x2) @ z(y) / len(y)
    den = np.sqrt(max(1e-12, (1 - rx ** 2) * (1 - ry ** 2)))
    rp = (z(y) @ z(x1) / len(y) - rx * ry) / den
    return float(rp)


# ---------------------------------------------------------------- main
def main():
    rd = run_dir("e058")
    log("E058 site anatomy: which MLP site carries basis-structure, and does "
        "it replicate at 2.7M? (CPU-only, zero-GPU)")
    corpus = CharCorpus(CORPUS_PATH, seed=1337)

    metrics = {
        "experiment": "e058_site_anatomy",
        "date": common.now_iso(),
        "status": "complete",
        "device": "cpu-only (CUDA_VISIBLE_DEVICES masked; zero-GPU)",
        "causal_gate_anchors": GATE,
        "pregate_sites": PREGATE,
        "protocol": {"n_eval_batches": N_EVAL,
                     "batch_note": "first 15 of e028/e040's fixed 30 val batches",
                     "scaleA": {"sites": list(SITES_A), "hosts": MEMBERS,
                                "donor": "e040_ref"},
                     "scaleB": {"sites": list(SITES_B),
                                "hosts": ["e001(B, seed42)", "e028_b43(seed43)"],
                                "grafts": "both directions x 6 sites = 12"}},
    }

    # ================================================================ SCALE A
    cfg_a = Cfg(vocab=corpus.vocab_size, n_layer=4, n_head=4, n_embd=128,
                block_size=256)
    net_a = TinyGPT(cfg_a)
    assert net_a.num_params() == 840_704, net_a.num_params()
    inits_a = build_inits(cfg_a)
    trained_a = {name: torch.load(CKPT_DIR / f"{name}.pt", map_location="cpu",
                                  weights_only=True)
                 for name in ["e040_ref"] + MEMBERS}
    REF_SD = trained_a["e040_ref"]
    REF_DW = {s: dw_vec(REF_SD, inits_a["e040_ref"], mlp_keys(s)) for s in SITES_A}
    log(f"scale A: 0.84M cfg, {len(inits_a)} inits rebuilt, 11 finals loaded")

    # instrument gates: REF self-transplant exact zero
    net_a.load_state_dict(REF_SD)
    snap = snapshot(net_a)
    base_ref = per_batch_losses_cpu(net_a, corpus)
    transplant(net_a, REF_SD, mlp_keys(2))
    self_losses = per_batch_losses_cpu(net_a, corpus)
    net_a.load_state_dict(snap)
    assert_identical(net_a, snap, "e040_ref")
    ref_self_graft_exact = bool(self_losses == base_ref)
    log(f"REF self-graft L2 exactly 0.0: {ref_self_graft_exact} | base CE "
        f"{mean(base_ref):.4f}")

    rows_a = []
    for name in MEMBERS:
        sd = trained_a[name]
        net_a.load_state_dict(sd)
        snap = snapshot(net_a)
        base = per_batch_losses_cpu(net_a, corpus)
        rec = {"name": name, "base_ce": mean(base)}
        rec["graft"], rec["ablate"] = {}, {}
        for site in SITES_A:
            transplant(net_a, REF_SD, mlp_keys(site))     # member <- REF organ
            losses = per_batch_losses_cpu(net_a, corpus)
            net_a.load_state_dict(snap)
            assert_identical(net_a, snap, name)
            diffs = [a - b for a, b in zip(losses, base)]
            rec["graft"][site] = {"dce": mean(diffs), "per_batch": diffs}
            with lesion(net_a, "mlp", site):              # zero own organ
                abl = per_batch_losses_cpu(net_a, corpus)
            assert_identical(net_a, snap, name)
            rec["ablate"][site] = mean([a - b for a, b in zip(abl, base)])
        rec["align_dist"] = {s: 1.0 - cos(dw_vec(sd, inits_a[name], mlp_keys(s)),
                                          REF_DW[s]) for s in SITES_A}
        rec["wspace_dist"] = {s: 1.0 - mean(
            [mean_col_cos(sd[f"h.{s}.mlp.0.weight"], REF_SD[f"h.{s}.mlp.0.weight"]),
             mean_col_cos(sd[f"h.{s}.mlp.2.weight"], REF_SD[f"h.{s}.mlp.2.weight"])])
            for s in SITES_A}
        rows_a.append(rec)
        log(f"  A {name:9s} D " + " ".join(f"L{s} {rec['graft'][s]['dce']:+.3f}"
                                           for s in SITES_A)
            + " | align " + " ".join(f"L{s} {rec['align_dist'][s]:.3f}"
                                     for s in SITES_A))

    # reproduction gate vs runs/e052 (L2/L3 cells are the SAME measurements)
    e052 = json.loads(E052_METRICS.read_text(encoding="utf-8"))
    e052_by = {m["name"]: m for m in e052["members"]}
    gate_a = {"graft_L2L3_per_batch_max_abs_diff": 0.0, "dw_cos_max_abs_diff": 0.0}
    for rec in rows_a:
        m = e052_by[rec["name"]]
        for s in (2, 3):
            mine = rec["graft"][s]["per_batch"]
            theirs = m["graft"][str(s)]["per_batch"]
            gate_a["graft_L2L3_per_batch_max_abs_diff"] = max(
                gate_a["graft_L2L3_per_batch_max_abs_diff"],
                max(abs(a - b) for a, b in zip(mine, theirs)))
            mine_dw = 1.0 - rec["align_dist"][s]
            gate_a["dw_cos_max_abs_diff"] = max(
                gate_a["dw_cos_max_abs_diff"], abs(mine_dw - m["dw_cos"][str(s)]))
    log(f"gate A vs e052: graft per-batch max |diff| "
        f"{gate_a['graft_L2L3_per_batch_max_abs_diff']:.2e} | dw_cos max |diff| "
        f"{gate_a['dw_cos_max_abs_diff']:.2e}")

    per_site_a = {}
    for s in SITES_A:
        D = [r["graft"][s]["dce"] for r in rows_a]
        per_site_a[s] = {
            "r_align": pearson(D, [r["align_dist"][s] for r in rows_a]),
            "r_wspace": pearson(D, [r["wspace_dist"][s] for r in rows_a]),
            "r_ablate": pearson(D, [r["ablate"][s] for r in rows_a]),
            "rho_align": spearman(D, [r["align_dist"][s] for r in rows_a]),
            "rho_wspace": spearman(D, [r["wspace_dist"][s] for r in rows_a]),
        }
    log("scale A per-site r(D_s, predictor_s), n=10 hosts:")
    for s in SITES_A:
        t = per_site_a[s]
        log(f"  L{s}: align r={t['r_align'][0]:+.3f} (p={t['r_align'][1]:.3f}) | "
            f"wspace r={t['r_wspace'][0]:+.3f} (p={t['r_wspace'][1]:.3f}) | "
            f"A r={t['r_ablate'][0]:+.3f} | align rho={t['rho_align'][0]:+.3f}")

    # ================================================================ SCALE B
    cfg_b = Cfg(vocab=corpus.vocab_size, **CFG_B)
    net_b = TinyGPT(cfg_b)
    n_params_b = net_b.num_params()
    assert n_params_b == 2_739_072, n_params_b
    init_b = make_init(cfg_b, 42)                   # e001 build(): set_seed(42) then TinyGPT
    init_b43 = make_init(cfg_b, 43)                 # e028: set_seed(43) then TinyGPT
    sds_b = {"B": torch.load(CKPT_DIR / "e001.pt", map_location="cpu",
                             weights_only=True),
             "B43": torch.load(CKPT_DIR / "e028_b43.pt", map_location="cpu",
                               weights_only=True)}
    inits_b = {"B": init_b, "B43": init_b43}
    log(f"scale B: 2.7M cfg ({n_params_b} params), B + B43 loaded, inits rebuilt")

    scale_b = {"base_ce": {}, "graft": {}, "ablate": {}, "align_dist": {},
               "wspace_dist": {}}
    for host in ("B", "B43"):
        donor = "B43" if host == "B" else "B"
        net_b.load_state_dict(sds_b[host])
        snap = snapshot(net_b)
        base = per_batch_losses_cpu(net_b, corpus)
        scale_b["base_ce"][host] = mean(base)
        scale_b["ablate"][host] = {}
        for site in SITES_B:
            transplant(net_b, sds_b[donor], mlp_keys(site))
            losses = per_batch_losses_cpu(net_b, corpus)
            net_b.load_state_dict(snap)
            assert_identical(net_b, snap, f"{host}@L{site}")
            scale_b["graft"][f"{host}<-{donor}|L{site}"] = {
                "dce": mean([a - b for a, b in zip(losses, base)]),
                "per_batch": [a - b for a, b in zip(losses, base)]}
            with lesion(net_b, "mlp", site):
                abl = per_batch_losses_cpu(net_b, corpus)
            assert_identical(net_b, snap, f"{host}@L{site}")
            scale_b["ablate"][host][site] = mean([a - b for a, b in zip(abl, base)])
        log(f"  B host {host}: base CE {scale_b['base_ce'][host]:.4f} | graft dce "
            + " ".join(f"L{s} {scale_b['graft'][f'{host}<-{donor}|L{s}']['dce']:+.3f}"
                       for s in SITES_B))

    for host in ("B", "B43"):
        donor = "B43" if host == "B" else "B"
        dw_host = {s: dw_vec(sds_b[host], inits_b[host], mlp_keys(s))
                   for s in SITES_B}
        dw_donor = {s: dw_vec(sds_b[donor], inits_b[donor], mlp_keys(s))
                    for s in SITES_B}
        scale_b["align_dist"][host] = {
            s: 1.0 - cos(dw_host[s], dw_donor[s]) for s in SITES_B}
        scale_b["wspace_dist"][host] = {
            s: 1.0 - mean([
                mean_col_cos(sds_b[host][f"h.{s}.mlp.0.weight"],
                             sds_b[donor][f"h.{s}.mlp.0.weight"]),
                mean_col_cos(sds_b[host][f"h.{s}.mlp.2.weight"],
                             sds_b[donor][f"h.{s}.mlp.2.weight"])])
            for s in SITES_B}

    # gate B1: dW reconstruction vs runs/e029 (bitwise: same init convention)
    e029 = json.loads(E029_METRICS.read_text(encoding="utf-8"))
    ref029 = e029["dw_alignment"]["per_pair"]["B<->B43"]["organs"]
    gate_b = {"dw_cos_vs_e029": {}, "max_abs_diff": 0.0}
    for s in (0, 3, 5):
        mine = 1.0 - scale_b["align_dist"]["B"][s]
        theirs = ref029[f"L{s}|mlp"]
        gate_b["dw_cos_vs_e029"][f"L{s}"] = {"mine": mine, "e029": theirs,
                                             "abs_diff": abs(mine - theirs)}
        gate_b["max_abs_diff"] = max(gate_b["max_abs_diff"], abs(mine - theirs))
    # gate B2: B<-B43 graft per-batch vs runs/e028 'within' cells (GPU eval there)
    e028 = json.loads(E028_METRICS.read_text(encoding="utf-8"))
    gate_b["graft_vs_e028"] = {}
    for s in (0, 2, 3, 5):
        mine = scale_b["graft"][f"B<-B43|L{s}"]["per_batch"]
        theirs = e028["cells"][f"within|L{s}|mlp"]["per_batch_dce"][:N_EVAL]
        d = max(abs(a - b) for a, b in zip(mine, theirs))
        gate_b["graft_vs_e028"][f"L{s}"] = d
    log(f"gate B vs e029 dw: max |diff| {gate_b['max_abs_diff']:.2e} | vs e028 "
        f"graft per-batch max |diff| "
        f"{max(gate_b['graft_vs_e028'].values()):.2e} (GPU-eval tolerance 2e-3)")

    # ---- pooled correlations over the 12 grafts ----
    keys12 = [(h, s) for h in ("B", "B43") for s in SITES_B]
    D12 = [scale_b["graft"][f"{h}<-{'B43' if h == 'B' else 'B'}|L{s}"]["dce"]
           for h, s in keys12]
    a12 = [scale_b["align_dist"][h][s] for h, s in keys12]
    w12 = [scale_b["wspace_dist"][h][s] for h, s in keys12]
    A12 = [scale_b["ablate"][h][s] for h, s in keys12]
    pooled = {"r_align": pearson(D12, a12), "r_wspace": pearson(D12, w12),
              "r_ablate": pearson(D12, A12),
              "partial_r_align_given_A": partial_r(D12, a12, A12),
              "partial_r_wspace_given_A": partial_r(D12, w12, A12)}
    log("scale B pooled over 12 grafts: "
        + " | ".join(f"{k}={v:+.3f}" for k, v in pooled.items()
                     if not isinstance(v, tuple))
        + f" | align r={pooled['r_align'][0]:+.3f} wspace r={pooled['r_wspace'][0]:+.3f}"
        + f" A r={pooled['r_ablate'][0]:+.3f}")

    # ---- per-direction cross-site coupling (n=6 sites) ----
    dir_stats = {}
    for h in ("B", "B43"):
        don = "B43" if h == "B" else "B"
        Ds = [scale_b["graft"][f"{h}<-{don}|L{s}"]["dce"] for s in SITES_B]
        As = [scale_b["ablate"][h][s] for s in SITES_B]
        als = [scale_b["align_dist"][h][s] for s in SITES_B]
        ws = [scale_b["wspace_dist"][h][s] for s in SITES_B]
        Rs = [d / a if abs(a) > 1e-9 else float("inf") for d, a in zip(Ds, As)]
        dir_stats[h] = {
            "D": Ds, "A": As, "R": Rs, "align": als, "wspace": ws,
            "rho_D_align": spearman(Ds, als), "rho_D_wspace": spearman(Ds, ws),
            "rho_D_A": spearman(Ds, As),
            "R_ladder_peak_site": int(np.argmax(Rs)),
            "D_ladder_peak_site": int(np.argmax(Ds))}
        log(f"  B dir {h}<-{don}: R ladder "
            + " ".join(f"L{s} {Rs[i]:.2f}" for i, s in enumerate(SITES_B))
            + f" | peak R at L{dir_stats[h]['R_ladder_peak_site']} | "
              f"rho(D,align) {dir_stats[h]['rho_D_align'][0]:+.2f} "
              f"rho(D,wspace) {dir_stats[h]['rho_D_wspace'][0]:+.2f} "
              f"rho(D,A) {dir_stats[h]['rho_D_A'][0]:+.2f}")

    # ================================================================ VERDICT
    r_align_a = {s: per_site_a[s]["r_align"][0] for s in SITES_A}
    l2_lead = min(r_align_a[2] - r_align_a[s] for s in SITES_A if s != 2)
    scaleA_l2_selective = bool(r_align_a[2] >= 0.6 and l2_lead >= 0.15)

    geom_partial_best = max(pooled["partial_r_align_given_A"],
                            pooled["partial_r_wspace_given_A"])
    dir_coupling_ok = all(
        max(dir_stats[h]["rho_D_align"][0], dir_stats[h]["rho_D_wspace"][0]) >= 0.6
        for h in ("B", "B43"))
    replicated = bool(geom_partial_best >= 0.5 or dir_coupling_ok)

    peak_B = dir_stats["B"]["R_ladder_peak_site"]
    peak_B43 = dir_stats["B43"]["R_ladder_peak_site"]
    conds = {
        "scaleA_l2_selective": scaleA_l2_selective,
        "r_align_A": r_align_a,
        "replicated_at_depth": replicated,
        "pooled_partial_best": geom_partial_best,
        "dir_coupling_ok_both_dirs": dir_coupling_ok,
        "R_peak_B": peak_B, "R_peak_B43": peak_B43,
        "pregate_B": PREGATE["B_2.7M"], "pregate_B43": PREGATE["B43_2.7M"],
    }
    if replicated and peak_B == PREGATE["B_2.7M"] and peak_B43 == PREGATE["B43_2.7M"]:
        verdict = ("H-ROLE — structure lives at the pre-causal-gate stream-writer "
                   f"site across scales: R-ladder peaks at gate-1 on both 2.7M "
                   f"hosts (B: L{peak_B}, B43: L{peak_B43}; gates L3/L4), and the "
                   "geometry-damage relation replicates at depth")
    elif replicated and peak_B == 2 and peak_B43 == 2:
        verdict = ("H-DEPTH — the coupling is L2 specifically at both scales "
                   f"(R peaks at L2 even on B43, whose pre-gate site is "
                   f"L{PREGATE['B43_2.7M']}) — a depth-indexing artifact")
    elif not replicated:
        verdict = ("H-NONE — no geometry-damage relation at 2.7M (pooled partial "
                   f"r {geom_partial_best:+.3f} < 0.5, no cross-site coupling); "
                   "the e052 L2 signal does not replicate at depth — "
                   "single-lineage noise")
    else:
        verdict = ("MIXED — replication signal present but the site pattern "
                   f"matches neither H-role (peaks L{peak_B}/L{peak_B43} vs "
                   f"pre-gate L{PREGATE['B_2.7M']}/L{PREGATE['B43_2.7M']}) nor "
                   f"H-depth cleanly; B43-direction (pre-gate L3) is the "
                   "discriminating cell")
    log("=" * 78)
    log(f"VERDICT: {verdict}")
    log(f"conditions: {json.dumps(conds, default=float)}")

    # ================================================================ outputs
    metrics.update({
        "params": {"scaleA": 840_704, "scaleB": n_params_b},
        "config": {"scaleA": common.cfg_dict(cfg_a), "scaleB": common.cfg_dict(cfg_b)},
        "instrument_gates": {
            "ref_self_transplant_zero": ref_self_graft_exact,
            "hosts_bitwise_restored": True,
            "scaleA_vs_e052": gate_a,
            "scaleB_vs_e029_dw": gate_b["dw_cos_vs_e029"],
            "scaleB_vs_e028_graft_per_batch_max_abs_diff": gate_b["graft_vs_e028"],
        },
        "scaleA_members": [{"name": r["name"], "base_ce": r["base_ce"],
                            "graft": {str(s): r["graft"][s] for s in SITES_A},
                            "ablate": {str(s): r["ablate"][s] for s in SITES_A},
                            "align_dist": {str(s): r["align_dist"][s]
                                           for s in SITES_A},
                            "wspace_dist": {str(s): r["wspace_dist"][s]
                                            for s in SITES_A}} for r in rows_a],
        "scaleA_per_site": {str(s): per_site_a[s] for s in SITES_A},
        "scaleB": {"base_ce": scale_b["base_ce"],
                   "graft": {k: v["dce"] for k, v in scale_b["graft"].items()},
                   "graft_per_batch": {k: v["per_batch"]
                                       for k, v in scale_b["graft"].items()},
                   "ablate": {h: {str(s): v for s, v in scale_b["ablate"][h].items()}
                              for h in ("B", "B43")},
                   "align_dist": {h: {str(s): v for s, v in d.items()}
                                  for h, d in scale_b["align_dist"].items()},
                   "wspace_dist": {h: {str(s): v for s, v in d.items()}
                                   for h, d in scale_b["wspace_dist"].items()},
                   "R": {h: {str(s): dir_stats[h]["R"][i]
                             for i, s in enumerate(SITES_B)}
                         for h in ("B", "B43")}},
        "scaleB_pooled_12grafts": pooled,
        "scaleB_per_direction": {h: {k: (v if not isinstance(v, list) else
                                         [float(x) for x in v])
                                     for k, v in dir_stats[h].items()}
                                 for h in ("B", "B43")},
        "registered_verdict_rules": {
            "replicated_at_depth": "pooled partial r(D, geom|A) >= 0.5 over 12 "
                                   "grafts, or per-direction cross-site "
                                   "Spearman(D, geom) >= 0.6 both directions",
            "H-role": "replicated AND R-ladder peaks at gate(host)-1 both hosts",
            "H-depth": "replicated AND R peaks at L2 in both directions",
            "H-none": "not replicated"},
        "conditions": conds,
        "verdict": verdict,
        "timing_s": round(time.time() - T0, 1),
    })
    save_json(rd / "metrics.json", metrics)

    # ---- site x metric heatmap + correlation panel --------------------------
    fig = plt.figure(figsize=(17, 10))

    ax1 = fig.add_subplot(2, 3, 1)
    pred_names = ["dW-align dist", "W-space dist", "own-ablate A"]
    matA = np.array([[per_site_a[s]["r_align"][0] for s in SITES_A],
                     [per_site_a[s]["r_wspace"][0] for s in SITES_A],
                     [per_site_a[s]["r_ablate"][0] for s in SITES_A]])
    im = ax1.imshow(matA, vmin=-1, vmax=1, cmap="RdBu_r")
    ax1.set_xticks(range(4), [f"L{s}\n[pre-gate]" if s == PREGATE["small_0.84M"]
                              else f"L{s}" for s in SITES_A])
    ax1.set_yticks(range(3), pred_names)
    for i in range(3):
        for j in range(4):
            ax1.text(j, i, f"{matA[i, j]:+.2f}", ha="center", va="center",
                     fontsize=10, color="k")
    ax1.add_patch(plt.Rectangle((2 - 0.5, -0.5), 1, 3, fill=False,
                                edgecolor="#1a9641", lw=3))
    ax1.set_title("0.84M e040 family: per-site r(D_s, pred_s), n=10 hosts\n"
                  "green box = pre-gate site (L2; causal gate L3)", fontsize=10)
    fig.colorbar(im, ax=ax1, fraction=0.046)

    ax2 = fig.add_subplot(2, 3, 4)
    rows_b, row_tags, preg_cols = [], [], []
    for h in ("B", "B43"):
        don = "B43" if h == "B" else "B"
        rows_b.append([dir_stats[h]["D"][i] for i in range(6)])
        row_tags.append(f"D {h}<-{don}")
        preg_cols.append(PREGATE[f"{h}_2.7M"])
        rows_b.append([dir_stats[h]["R"][i] for i in range(6)])
        row_tags.append(f"R=D/A {h}")
        preg_cols.append(PREGATE[f"{h}_2.7M"])
        rows_b.append([dir_stats[h]["align"][i] for i in range(6)])
        row_tags.append(f"align-dist {h}")
        preg_cols.append(PREGATE[f"{h}_2.7M"])
        rows_b.append([dir_stats[h]["wspace"][i] for i in range(6)])
        row_tags.append(f"W-space {h}")
        preg_cols.append(PREGATE[f"{h}_2.7M"])
    rows_b.append([dir_stats["B"]["A"][i] for i in range(6)])
    row_tags.append("A (B)"); preg_cols.append(-1)
    Zb = np.array([[(v - np.min(r)) / (np.ptp(r) + 1e-12) for v in r] for r in rows_b])
    ax2.imshow(Zb, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax2.set_xticks(range(6), [f"L{s}" for s in SITES_B])
    ax2.set_yticks(range(len(row_tags)), row_tags, fontsize=8)
    for i, r in enumerate(rows_b):
        for j in range(6):
            ax2.text(j, i, f"{r[j]:.2f}", ha="center", va="center", fontsize=7,
                     color="w" if Zb[i, j] < 0.6 else "k")
    for i, pc in enumerate(preg_cols):
        if pc >= 0:
            ax2.add_patch(plt.Rectangle((pc - 0.5, i - 0.5), 1, 1, fill=False,
                                        edgecolor="#ff7043", lw=2.5))
    ax2.set_title("2.7M B vs B43 (12 grafts): per-site profiles (row-normalized)\n"
                  "orange box = that host's pre-gate site (B: L2, B43: L3)",
                  fontsize=10)

    ax3 = fig.add_subplot(2, 3, 2)
    bars = ["r(D,align)", "r(D,W-space)", "r(D,A)",
            "partial r(D,align|A)", "partial r(D,W|A)"]
    vals = [pooled["r_align"][0], pooled["r_wspace"][0], pooled["r_ablate"][0],
            pooled["partial_r_align_given_A"], pooled["partial_r_wspace_given_A"]]
    colors = ["#2c6fbb" if v >= 0 else "#c0392b" for v in vals]
    ax3.barh(range(len(vals)), vals, color=colors)
    ax3.axvline(0.5, color="#1a9641", ls="--", lw=1.5, label="replication bar 0.5")
    ax3.set_yticks(range(len(bars)), bars, fontsize=9)
    ax3.set_xlim(-1, 1)
    ax3.set_title("2.7M pooled over 12 grafts (both directions, 6 sites)",
                  fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.25, axis="x")

    ax4 = fig.add_subplot(2, 3, 3)
    xs = np.arange(6)
    ax4.plot(xs, dir_stats["B"]["R"], "o-", label="B host (gate L3, pre L2)",
             color="#2c6fbb")
    ax4.plot(xs, dir_stats["B43"]["R"], "s-", label="B43 host (gate L4, pre L3)",
             color="#c0392b")
    ax4.axhline(1.1, color="#888", ls=":", lw=1, label="active-interference band")
    for x, tag in ((PREGATE["B_2.7M"], "B pre-gate"), (PREGATE["B43_2.7M"], "B43 pre-gate")):
        ax4.axvline(x, color="#1a9641" if x == PREGATE["B_2.7M"] else "#7b3fa0",
                    ls="--", lw=1.5)
    ax4.set_xticks(xs, [f"L{s}" for s in SITES_B])
    ax4.set_ylabel("R = graft damage / own-ablation damage")
    ax4.set_title("2.7M interference-ratio ladder — the H-role vs H-depth cell",
                  fontsize=10)
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.25)

    ax5 = fig.add_subplot(2, 3, 5)
    for r in rows_a:
        ax5.scatter(r["align_dist"][2], r["graft"][2]["dce"], color="#2c6fbb",
                    s=30, zorder=3)
        ax5.annotate(r["name"].replace("e040_", ""), (r["align_dist"][2],
                                                      r["graft"][2]["dce"]),
                     fontsize=6, alpha=0.8, xytext=(2, 2),
                     textcoords="offset points")
    ax5.set_xlabel("L2 dW-alignment distance to REF")
    ax5.set_ylabel("L2 graft damage D (15 batches)")
    rr = per_site_a[2]["r_align"]
    ax5.set_title(f"0.84M L2 cell (e052 extension): r={rr[0]:+.3f} "
                  f"(p={rr[1]:.3f})", fontsize=10)
    ax5.grid(alpha=0.25)

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    lines = [
        "CAUSAL-GATE ANCHORS (published):",
        "  0.84M small 4L: mode L3 (e005s)  -> pre-gate L2",
        "  2.7M  B     6L: mode L3 (e018/e012d) -> pre-gate L2",
        "  2.7M  B43   6L: mode L4 (e012d)  -> pre-gate L3  <- discriminator",
        "  10M   large 8L: mode L2 (e005s, bimodal 2/3) -> pre-gate L1",
        "",
        "REGISTERED RULES:",
        "  replicated := pooled partial r>=0.5 or both-direction rho>=0.6",
        "  H-role  := replicated AND R-peaks at gate-1 on both hosts",
        "  H-depth := replicated AND R-peaks at L2 on both hosts",
        "  H-none  := not replicated",
        "",
        f"VERDICT: {verdict}",
    ]
    ax6.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=9, family="monospace",
             wrap=True)

    fig.suptitle("E058 — site anatomy of the geometry-damage coupling: L2, "
                 "pre-gate site, or noise?", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(rd / "site_anatomy_heatmap.png", dpi=140)
    plt.close(fig)
    log(f"outputs: {rd} (metrics.json, site_anatomy_heatmap.png)")
    return metrics


if __name__ == "__main__":
    main()
