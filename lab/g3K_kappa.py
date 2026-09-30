"""G3K — THE KAPPA CELL (W022's named measurement; spec scratch/g3K_design.md).

THE QUESTION (design verbatim): kappa = (isotropic kill rung) / (wash kill
rung) at matched MEAN per-coordinate RMS (total L2 / sqrt(N) equal to the
wash's own 1x displacement; the wash's RMS may be unevenly distributed — that
unevenness is itself a reading). Read kappa TWICE under ONE convention:
kappa_store (the store's g0 ruler on the g3/g3R organism) and kappa_host (the
host fact's battery ruler on the g3 lineage's own host-fact organism).

ORGANISMS (both provenance-gated against committed metrics BEFORE any ladder
compute — Rule 12 / W021's meta-law):
  * STORE  runs/checkpoints/g3_gen.pt — the g3/g3R organism (e098_base_s4305
    host, bit-identical; Hopfield store carries the fact; root g0 must
    reproduce 0.8885950 within 1e-6; store-off g0 <= 0.27 — g3R's gates).
    Wash direction SOURCE: the committed full-wash trajectory snapshot at its
    own t* = +1 — runs/checkpoints/g3_gen_s1.pt minus the root (g3's wash
    seed 10902; NOT recomputed: loaded from the committed artifact; the s1
    cell must reproduce its committed battery read 5.889e-05 and displacement
    Dstore 0.13184139 / D_all 0.92580074).
  * HOST   runs/checkpoints/e157_f2_consolidated.pt — the g3 lineage's OWN
    host-fact control (S-DISC, recorded inside runs/g3/metrics.json:
    s_disc_control, verified_vs_file true). g3's host is pristine BY
    CONSTRUCTION (G_STOREOFF g0 0.0012 — there is no host-carried fact on
    g3_gen to read), so "the host fact's battery" in the g3 lineage means the
    discriminative ZEPHYRA fact installed in HOST WEIGHTS on the SAME e098
    s4305 host family, same corpus/battery protocol, same wash recipe+seed
    10902, same t* = +1. Its ruler (install60 offset-0 battery ids130) must
    reproduce the committed root read 0.5784125924110413, the s1 wash
    displacement 0.916419706836259 and the s1 battery read 0.0040224288
    (all bit-exact-tolerance 1e-6) — the ruler is the committed one, not an
    improvisation. Wash direction SOURCE: e157_f2_neutral_s1.pt minus the
    root (committed artifact, seed 10902).

ARMS (eval-only — NO training, NO GPU claim; CPU only): direction {wash (the
stored full-wash unit direction above), isotropic (fresh full-parameter
Gaussian, 3 seeds: store 11401-3 / host 11411-13 — no collisions with any
registered block)} x rung {1, 2, 4, 8, 16, 32, 64} x matched MEAN
per-coordinate RMS (perturbation total L2 = rung x ||wash 1x displacement||,
so mean per-coordinate RMS = rung x ||D||/sqrt(N) in BOTH arms — the wash's
per-coordinate RMS happens to be near-uniform (~1e-3 everywhere: the first
AdamW step), which the concentration co-read quantifies rather than assumes).
Isotropic directions are drawn once per seed and rescaled across rungs
(g3/g3R's ladder methodology).

READS at every rung: STORE g0 (install60 offset-0 battery mean p(Z), the
fact's expression through the store's full 8-pattern retrieval; kill g0 < 0.5,
g3R's SURVIVE bar per the frozen design) AND HOST battery p(Z) (same battery
instrument on the host-fact organism; kill p(Z) < 0.5). CE_R co-read (e185's
collateral currency) separates fact-death from organism-wreck. Kill rung =
first rung below bar, interpolated in log2 between adjacent rungs; kill at
rung 1 is reported as threshold <= 1x (bounded above, not interpolated);
no kill by 64x is flagged off-grid-high. kappa per iso seed = iso threshold /
wash threshold; the adjudicated kappa is the MEAN over the 3 seeds with
min/max co-reported (any seed straddling a bar edge is flagged, not hidden).

WHAT EACH ARM GUARANTEES (stated before compute; W021's meta-law):
  * the WASH arm guarantees exact displacement rung x ||D|| along the measured
    wash direction — it guarantees NOTHING about killing (the kill is the
    measurement);
  * the ISOTROPIC arm guarantees an even spread — each coordinate receives
    the same RMS in expectation — it does NOT guarantee sparing at any rung;
    the upper rungs could genuinely fail (via organism wreck; the CE_R
    co-read names it). That is why this cell is honest.

REGISTERED BARS (frozen — VERBATIM from scratch/g3K_design.md; no shopping):
  KAPPA-SPLIT: "fires if kappa_host <= 4 and kappa_store >= 8 at matched mean
      per-coordinate RMS — directional immunity is purchasable; the store's
      wide cone is a design property."
  NO-BASIN-UNIVERSAL: "fires if both kappas <= 4 — e185's no-basin
      generalizes; the critic's 24-32x was convention-bound; the
      concentration co-read names where the wash's energy lives."
  MIXED: "any other pattern — graded; the pair reported verbatim."
Off-grid kappas enter the bars conservatively: lower bounds can satisfy
">= 8", never "<= 4"; an off-grid kappa_host forces MIXED with the bound
stated.

CO-READS: (i) wash RMS concentration profile — top-decile coordinate share of
the wash L2 (vs the iso draws' measured same statistic); (ii) store-coords'
share of wash RMS vs of iso RMS, plus per-group (store/host) per-coordinate
RMS at 1x (the convention-disagreement reading); (iii) wash-direction drift —
cos(snapshot d01, cumulative d0T) and cos(snapshot d01, terminal segment d1T)
at T = 300 for both organisms (the direction is ONE trajectory snapshot; the
tilt ladder already licensed ~45-degree tilt robustness).

HONESTY BLOCK (pre-registered): organ n=1 (construction seed 43050; g3R's
scope carried — the split's n=3 was over wash/noise draws); ruler asymmetry —
g0 (store attractor readout through behavior) vs battery p(Z) on a
discriminative host fact have different dynamics, so the kappa PAIR is the
finding, never a single number; single wash-snapshot direction (drift
co-read reported); the host ruler is read on the lineage's second organism
(the discriminative twin), not on g3_gen itself — same family, same battery,
same wash, but n=1 per organism.

Outputs: runs/g3K/{metrics.json, kappa.png}. No NOTES/THINKING/QUEUE/STATE
edits (the coordinator folds); single commit + push.

Run:  cd lab && python g3K_kappa.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # visible but NEVER claimed

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                           # noqa: E402

torch.set_num_threads(8)                              # e143/e151/e157/g3 convention

import common                                          # noqa: E402
from common import Cfg, TinyGPT, gpu_status, run_dir, save_json  # noqa: E402

import e043_install as E43                             # noqa: E402 (REPO, jsonable)

import g3R_seed_replicates as R                        # noqa: E402 (rebuild_protocol, load_root)
from g3_generative_store import (                      # noqa: E402
    CKPT_DIR, evl_load, battery_cell, battery_pz, ce_fixed_cpu, sd_disp,
    store_keys, STOREOFF_BAR,
)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ---- the frozen cell constants ---------------------------------------------------
RUNGS = (1, 2, 4, 8, 16, 32, 64)
KILL_BAR = 0.50                    # both rulers, verbatim from the frozen design
CE_COLLATERAL = 3.0                # e185's collateral vocabulary (CO-READ only)
ISO_SEEDS = {"store": (11401, 11402, 11403), "host": (11411, 11412, 11413)}
G_TOL = 1e-6                       # bit-exact-tolerance for committed reads

# committed references (runs/g3/metrics.json, runs/g3R/metrics.json,
# runs/e157/metrics.json — the provenance gates below reproduce these)
REF = {
    "store": {
        "root": "g3_gen.pt", "wash_s1": "g3_gen_s1.pt",
        "root_g0": 0.8885950446128845,
        "wash_s1_g0": 5.8887377235805616e-05,
        "Dstore_s1": 0.13184139341968207,
        "D_all_s1": 0.9258007407188416,
        "drift_cks": {"s10": "g3_gen_s10.pt", "s300": "g3_gen_s300.pt"},
    },
    "host": {
        "root": "e157_f2_consolidated.pt", "wash_s1": "e157_f2_neutral_s1.pt",
        "root_g0": 0.5784125924110413,      # runs/g3 metrics s_disc_control.trace.g0[0]
        "wash_s1_g0": 0.0040224287658929825,  # e157 B_wash traj step 1
        "D_all_s1": 0.916419706836259,      # runs/g3 metrics s_disc_control.displacement[1]
        "drift_cks": {"s4": "e157_f2_neutral_s4.pt", "s300": "e157_f2_neutral.pt"},
    },
}

F2_CFG = Cfg(vocab=65, n_layer=4, n_head=4, n_embd=128, block_size=512)

REGISTERED_BARS = {
    "KAPPA-SPLIT": "fires if kappa_host <= 4 and kappa_store >= 8 at matched "
        "mean per-coordinate RMS — directional immunity is purchasable; the "
        "store's wide cone is a design property.",
    "NO-BASIN-UNIVERSAL": "fires if both kappas <= 4 — e185's no-basin "
        "generalizes; the critic's 24-32x was convention-bound; the "
        "concentration co-read names where the wash's energy lives.",
    "MIXED": "any other pattern — graded; the pair reported verbatim.",
}

INSTRUMENT_GUARANTEES = {
    "wash_arm": "guarantees exact displacement rung x ||D|| along the measured "
        "full-wash direction (mean per-coordinate RMS = rung x ||D||/sqrt(N)); "
        "guarantees NOTHING about killing — the kill is the measurement.",
    "iso_arm": "guarantees an even spread (each coordinate receives the same "
        "RMS in expectation = rung x ||D||/sqrt(N)); does NOT guarantee "
        "sparing at any rung — upper rungs could genuinely fail (organism "
        "wreck; the CE_R co-read names it).",
    "convention": "matched MEAN per-coordinate RMS over ALL parameters: "
        "perturbation total L2 = rung x ||wash-1x displacement|| in both arms "
        "and both organisms; rungs are multiples of each organism's OWN wash "
        "1x displacement (store 0.9258 over 890,880 coords; host 0.9164 over "
        "873,472 coords — mean RMS at 1x is 9.81e-4 on both, the AdamW step).",
}

deviations: list[str] = [
    "The host-fact ruler is read on the g3 lineage's own host-fact organism "
    "(e157_f2_consolidated — the S-DISC control recorded inside runs/g3/"
    "metrics.json), NOT on g3_gen itself: g3's host is pristine by "
    "construction (G_STOREOFF 0.0012), so no host-carried fact exists on "
    "g3_gen to read. Same e098 s4305 host family, same corpus/battery "
    "protocol, same wash recipe+seed 10902, same t* = +1; both rulers' "
    "provenance gates reproduce committed values (bit-exact to 1e-6).",
    "The wash direction is the committed t*=+1 trajectory SNAPSHOT (single "
    "snapshot, g3R's convention; loaded from the committed s1 checkpoints, "
    "not recomputed); the drift co-read (cos to the cumulative and terminal "
    "directions at +300) is reported for both organisms.",
    "The isotropic arm is a full-parameter-vector Gaussian (NOT the "
    "store-subspace Gaussian of g3/g3R's iso legs): the matched-MEAN-"
    "per-coordinate-RMS convention spans all coordinates. Numerically the "
    "store subspace's expected exposure is nearly unchanged (r x 0.1295 vs "
    "the critic's store-subspace ladder at level x 0.1318), but the host "
    "coordinates are now also perturbed — the honest full-organism form of "
    "the critic's extension.",
    "Kill bars: BOTH rulers at 0.5 verbatim from the frozen design (g3R's "
    "SURVIVE bar for the store; e185's bar family for the host). The store's "
    "SHUT bar 0.27 is NOT used here; curves are reported in full.",
    "Kill at rung 1 (the wash itself, both organisms' committed s1 reads are "
    "far below bar) is reported as threshold <= 1x — bounded above, not "
    "interpolated; kappa is then numerically the iso threshold.",
    "CPU only; no GPU claimed (another agent holds it for training); no "
    "training of any kind; no checkpoints written.",
]


# ------------------------------------------------------------------ instruments

def load_ck(name: str) -> dict:
    st = torch.load(CKPT_DIR / name, map_location="cpu", weights_only=False)
    return {k: v.clone() for k, v in st["model"].items()}


def top_decile_share(v: torch.Tensor) -> float:
    """L2 share of the 10% largest-|coordinate| entries (RMS concentration)."""
    n = v.numel()
    k = max(1, int(round(0.1 * n)))
    a = v.abs()
    thr = torch.kthvalue(a, n - k + 1).values
    top = a[a >= thr]
    return float(top.norm() / v.norm())


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def read_store(sd: dict, net, pr) -> dict:
    net.load_state_dict(sd)
    bz = battery_cell(net, pr["ids130"], pr["zid"])
    ce = ce_fixed_cpu(net, *pr["r_eval_xy"])
    return {"g0": bz["mean_pz"], "frac_argmax_z": bz["frac_argmax_z"],
            "ce_r": ce}


def read_host(sd: dict, net, pr) -> dict:
    net.load_state_dict(sd)
    bz = battery_cell(net, pr["ids130"], pr["zid"])
    ce = ce_fixed_cpu(net, *pr["r_eval_xy"])
    return {"pz": bz["mean_pz"], "frac_argmax_z": bz["frac_argmax_z"],
            "ce_r": ce}


# ------------------------------------------------------------------ kill/kappa

def kill_analysis(rows: list[dict], read_key: str) -> dict:
    """rows: [{'rung': 0, read...}, {'rung': r in RUNGS, ...}]; rung 0 = root.
    Kill = first rung with read < KILL_BAR; threshold interpolated in log2
    between the last above-bar rung and the kill rung (root anchors rung 1
    kills as a <=1x bound)."""
    ladder = [r for r in rows if r["rung"] > 0]
    ki = next((i for i, r in enumerate(ladder)
               if ladder[i][read_key] < KILL_BAR), None)
    out = {"curve": [{ "rung": r["rung"], read_key: r[read_key],
                       "ce_r": r["ce_r"]} for r in rows],
           "bar": KILL_BAR}
    if ki is None:
        out.update(kill_rung=None, threshold=None, off_grid_high=True,
                   bounded_above=False,
                   bound=f"no rung through {RUNGS[-1]}x below bar")
        return out
    kr = ladder[ki][read_key]
    rung_k = ladder[ki]["rung"]
    out.update(kill_rung=rung_k, off_grid_high=False)
    if ki == 0:                      # rung 1 already below bar
        out.update(threshold=1.0, bounded_above=True,
                   note="first rung (1x) already below bar — true threshold "
                        "<= 1x (anchored at the root read "
                        f"{rows[0][read_key]:.4f}; not interpolated)")
        return out
    prev = ladder[ki - 1]
    lr_prev, lr_kill = np.log2(prev["rung"]), np.log2(rung_k)
    vp, vk = prev[read_key], kr
    t = (vp - KILL_BAR) / max(vp - vk, 1e-12)
    thr = float(2.0 ** (lr_prev + t * (lr_kill - lr_prev)))
    out.update(threshold=thr, bounded_above=False,
               interp_from=prev["rung"], interp_to=rung_k)
    return out


def kappa_from(iso_analyses: list[dict], wash_an: dict) -> dict:
    """kappa per seed = iso threshold / wash threshold; mean adjudicated,
    min/max + straddle flags co-reported; off-grid handled conservatively."""
    wash_thr = wash_an["threshold"]
    per_seed, bounds = [], []
    for a in iso_analyses:
        if a["off_grid_high"] or a["threshold"] is None:
            per_seed.append({"kappa": None,
                             "lower_bound": RUNGS[-1] / wash_thr,
                             "seed_iso": a})
            bounds.append(RUNGS[-1] / wash_thr)
        else:
            per_seed.append({"kappa": a["threshold"] / wash_thr,
                             "seed_iso": a})
    finite = [p["kappa"] for p in per_seed if p["kappa"] is not None]
    if len(finite) == len(per_seed):
        mean_k = float(np.mean(finite))
        val, lo, hi = mean_k, float(np.min(finite)), float(np.max(finite))
        off_grid = False
    elif finite:
        mean_k = float(np.mean(finite))
        val, lo, hi = None, float(np.min(finite)), max(RUNGS[-1] / wash_thr,
                                                       float(np.max(finite)))
        off_grid = True                      # at least one seed off-grid-high
    else:
        mean_k = val = None
        lo, hi = RUNGS[-1] / wash_thr, None
        off_grid = True
    straddle8 = bool(finite and (np.max(finite) >= 8 > np.min(finite)))
    straddle4 = bool(finite and (np.max(finite) >= 4 > np.min(finite)))
    return {"per_seed": per_seed, "kappa_mean": mean_k, "kappa_value": val,
            "kappa_min": lo, "kappa_max": hi, "off_grid_high": off_grid,
            "any_seed_off_grid": off_grid, "bar_straddle_8": straddle8,
            "bar_straddle_4": straddle4,
            "wash_threshold": wash_thr,
            "wash_bounded_above": wash_an.get("bounded_above", False)}


# ------------------------------------------------------------------ the cell

def run_organism(tag: str, sd_root: dict, sd_s1: dict, skeys: list,
                 iso_seeds: tuple[int, ...], pr, reader, net_builder,
                 drift_cks: dict, read_key: str) -> dict:
    delta = {k: (sd_s1[k].float() - sd_root[k].float()) for k in sd_root}
    Dw = float(sum(float(delta[k].norm() ** 2) for k in delta) ** 0.5)
    n_params = int(sum(sd_root[k].numel() for k in sd_root))
    net = net_builder()

    # ---- concentration co-reads (wash vs iso draws) ----
    host_keys = [k for k in sd_root if k not in set(skeys)] if skeys else None
    w_flat = flat_delta(sd_root, delta)
    conc = {"n_params": n_params, "wash_L2_1x": Dw,
            "wash_mean_rms_1x": Dw / n_params ** 0.5,
            "wash_top_decile_share": top_decile_share(w_flat)}
    if skeys:
        w_store = flat_delta(sd_root, delta, keys=skeys)
        w_host = flat_delta(sd_root, delta, keys=host_keys)
        conc.update({
            "wash_store_share_of_L2": float(w_store.norm() / w_flat.norm()),
            "wash_store_per_coord_rms":
                float(w_store.norm() / w_store.numel() ** 0.5),
            "wash_host_per_coord_rms":
                float(w_host.norm() / w_host.numel() ** 0.5),
            "store_n_params": int(w_store.numel()),
            "host_n_params": int(w_host.numel())})
    iso_conc = []
    for s in iso_seeds:
        g = iso_draw(sd_root, s)
        gn = float(sum(float(g[k].norm() ** 2) for k in g) ** 0.5)
        sc = Dw / gn                       # scale the draw to the rung-1 L2
        gs1 = {k: sc * g[k] for k in g}    # (shares are scale-invariant; the
        gf = flat_delta(sd_root, gs1)      #  per-coord RMS stats are not)
        row = {"seed": s, "iso_top_decile_share": top_decile_share(gf)}
        if skeys:
            gsk = flat_delta(sd_root, gs1, keys=skeys)
            ghk = flat_delta(sd_root, gs1, keys=host_keys)
            row.update({
                "iso_store_share_of_L2": float(gsk.norm() / gf.norm()),
                "iso_store_per_coord_rms":
                    float(gsk.norm() / gsk.numel() ** 0.5),
                "iso_host_per_coord_rms":
                    float(ghk.norm() / ghk.numel() ** 0.5)})
        iso_conc.append(row)
    conc["iso_draws"] = iso_conc
    if iso_conc:
        for kk in iso_conc[0]:
            if kk.startswith("iso_"):
                conc[f"{kk}_mean"] = float(np.mean([r[kk] for r in iso_conc]))
    log(f"[{tag}] convention: L2(1x)={Dw:.4f} over {n_params} coords -> mean "
        f"RMS {conc['wash_mean_rms_1x']:.3e}; wash top-decile share "
        f"{conc['wash_top_decile_share']:.3f} (iso mean "
        f"{conc.get('iso_top_decile_share_mean', float('nan')):.3f})")
    if skeys:
        log(f"[{tag}] store-share of wash L2 {conc['wash_store_share_of_L2']:.3f} "
            f"vs iso {conc['iso_store_share_of_L2_mean']:.3f}; store per-coord "
            f"RMS wash {conc['wash_store_per_coord_rms']:.3e} vs iso "
            f"{conc['iso_store_per_coord_rms_mean']:.3e}")

    # ---- rung 0 (root) ----
    rows = [{"rung": 0, **reader(sd_root, net, pr)}]
    iso_rows = {s: [{"rung": 0, **reader(sd_root, net, pr)}]
                for s in iso_seeds}

    # ---- wash leg ----
    for r in RUNGS:
        sd = {k: sd_root[k] + r * delta[k] for k in sd_root}
        rows.append({"rung": r, **reader(sd, net, pr)})
        log(f"[{tag}] wash rung {r:2d}x: {read_key} {rows[-1][read_key]:.4f} "
            f"CE_R {rows[-1]['ce_r']:.3f}")
    wash_an = kill_analysis(rows, read_key=read_key)

    # ---- isotropic legs ----
    iso_ans = []
    for s in iso_seeds:
        g = iso_draw(sd_root, s)
        gn = float(sum(float(g[k].norm() ** 2) for k in g) ** 0.5)
        for r in RUNGS:
            sd = {k: sd_root[k] + (r * Dw / gn) * g[k] for k in sd_root}
            iso_rows[s].append({"rung": r, **reader(sd, net, pr)})
        a = kill_analysis(iso_rows[s], read_key=read_key)
        a["draw_seed"] = s
        a["draw_L2"] = gn
        iso_ans.append(a)
        log(f"[{tag}] iso seed {s}: kill_rung {a['kill_rung']} thr "
            f"{a['threshold']}")

    # ---- drift co-read ----
    drift = {}
    d01 = flat_delta(sd_root, delta)
    for nm, ck in drift_cks.items():
        sd_t = load_ck(ck)
        d0t = flat_delta(sd_root, {k: (sd_t[k].float() - sd_root[k].float())
                                   for k in sd_root})
        drift[f"cos_d01_d0_{nm}"] = cos(d01, d0t)
        drift[f"L2_d0_{nm}"] = float(d0t.norm())
    sd_t = load_ck(drift_cks["s300"])
    d1t = flat_delta(sd_root, {k: (sd_t[k].float() - sd_s1[k].float())
                               for k in sd_root})
    drift["cos_d01_d1_s300"] = cos(d01, d1t)

    del net
    return {"rows_wash": rows, "rows_iso": iso_rows, "wash": wash_an,
            "iso": iso_ans, "kappa": kappa_from(iso_ans, wash_an),
            "concentration": conc, "drift": drift}


def flat_delta(sd_root: dict, delta: dict, keys=None) -> torch.Tensor:
    ks = sorted(delta.keys()) if keys is None else sorted(keys)
    return torch.cat([delta[k].float().reshape(-1) for k in ks])


def iso_draw(sd_root: dict, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {k: torch.randn(sd_root[k].shape, generator=g) for k in sd_root}


# ------------------------------------------------------------------ plot

def plot(path: Path, M: dict):
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10))
    rkey = {"store": "g0", "host": "pz"}
    rlabel = {"store": "g0 (store ruler)", "host": "p(Z) (host ruler)"}
    cols = ["royalblue", "seagreen", "darkorange"]

    for j, tag in enumerate(("store", "host")):
        ax = axes[0, j]
        org = M["organisms"][tag]
        rk = rkey[tag]
        rows = org["rows_wash"]
        xs = [r["rung"] for r in rows if r["rung"] > 0]
        ys = [r[rk] for r in rows if r["rung"] > 0]
        ax.plot(xs, ys, "o-", color="crimson", lw=2.4, ms=7,
                label="wash direction (committed s1 snapshot)")
        for i, (s, rr) in enumerate(org["rows_iso"].items()):
            ax.plot([r["rung"] for r in rr if r["rung"] > 0],
                    [r[rk] for r in rr if r["rung"] > 0], "^--", color=cols[i],
                    lw=1.5, ms=6, alpha=0.9, label=f"isotropic seed {s}")
        ax.axhline(KILL_BAR, color="gray", ls=":", lw=1.4,
                   label=f"kill bar {KILL_BAR}")
        kap = org["kappa"]
        if org["kill"]["threshold"] is not None:
            ax.axvline(org["kill"]["threshold"], color="crimson", ls=":",
                       lw=1.2, alpha=0.7)
        if kap.get("kappa_mean") is not None:
            for a in org["iso"]:
                if a["threshold"] is not None:
                    ax.axvline(a["threshold"], color="gray", ls=":", lw=0.7,
                               alpha=0.5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(RUNGS))
        ax.set_xticklabels([f"{r}x" for r in RUNGS])
        ax.set_xlabel("rung (multiples of own wash-1x displacement; matched "
                      "mean per-coordinate RMS)")
        ax.set_ylabel(rlabel[tag])
        ktxt = (f"kappa = {kap['kappa_mean']:.1f} "
                f"[{kap['kappa_min']:.1f}, {kap['kappa_max']:.1f}]"
                if kap.get("kappa_mean") is not None
                else f"kappa > {kap['kappa_min']:.0f} (off-grid-high)")
        ax.set_title(f"{tag.upper()} organism — kill rungs: wash "
                     f"{org['kill']['kill_rung']}x, iso "
                     f"{[a['kill_rung'] for a in org['iso']]} | {ktxt}",
                     fontsize=9)
        ax.legend(fontsize=7, loc="lower left")
        # CE_R co-read on twin axis (organism-wreck vs fact-death)
        ax2 = ax.twinx()
        ax2.plot(xs, [r["ce_r"] for r in rows if r["rung"] > 0], ":", 
                 color="dimgray", lw=1.0, alpha=0.8)
        ax2.axhline(CE_COLLATERAL, color="dimgray", ls=":", lw=0.8)
        ax2.set_ylabel("CE_R (dotted; wash leg)", fontsize=8, color="dimgray")
        ax2.tick_params(labelsize=7, colors="dimgray")

    # (1,0) concentration co-reads
    ax = axes[1, 0]
    st, hs = M["organisms"]["store"]["concentration"], \
        M["organisms"]["host"]["concentration"]
    groups, wvals, ivals = [], [], []

    def add(name, wv, iv_lo, iv_hi):
        groups.append(name); wvals.append(wv); ivals.append((iv_lo, iv_hi))

    add("store org:\nwash top-decile\nL2 share vs iso",
        st["wash_top_decile_share"], st["iso_top_decile_share_mean"],
        st["iso_top_decile_share_mean"])
    add("host org:\nwash top-decile\nL2 share vs iso",
        hs["wash_top_decile_share"], hs["iso_top_decile_share_mean"],
        hs["iso_top_decile_share_mean"])
    add("store org:\nstore-coord share\nof L2 (wash vs iso)",
        st["wash_store_share_of_L2"], st["iso_store_share_of_L2_mean"],
        st["iso_store_share_of_L2_mean"])
    x = np.arange(len(groups))
    ax.bar(x - 0.18, wvals, width=0.36, color="crimson", alpha=0.85,
           label="wash direction")
    ax.bar(x + 0.18, [i[0] for i in ivals], width=0.36, color="dimgray",
           alpha=0.6, label="isotropic draws (mean)")
    for xi, v in zip(x - 0.18, wvals):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=7)
    for xi, v in zip(x + 0.18, [i[0] for i in ivals]):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=7)
    ax.set_ylabel("share of displacement L2")
    rms = (f"per-coordinate RMS at 1x — store org: store coords "
           f"wash {st['wash_store_per_coord_rms']:.2e} vs iso "
           f"{st['iso_store_per_coord_rms_mean']:.2e}; host coords wash "
           f"{st['wash_host_per_coord_rms']:.2e} vs iso "
           f"{st['iso_host_per_coord_rms_mean']:.2e}\n"
           f"host org (all coords): wash {hs['wash_mean_rms_1x']:.2e} vs iso "
           f"same by construction")
    ax.set_title("CONCENTRATION CO-READ — where the wash's energy lives\n"
                 + rms, fontsize=8)
    ax.legend(fontsize=7)

    # (1,1) verdict
    ax = axes[1, 1]
    ax.axis("off")
    v = M["adjudication"]
    lines = [f"VERDICT: {v['verdict']}", ""]
    for tag in ("store", "host"):
        k = M["organisms"][tag]["kappa"]
        w = M["organisms"][tag]["kill"]
        lines.append(
            f"kappa_{tag} = "
            + (f"{k['kappa_value']:.1f} (mean of seeds "
               f"{[None if p['kappa'] is None else round(p['kappa'], 1) for p in k['per_seed']]})"
               if k.get("kappa_value") is not None
               else f"> {k['kappa_min']:.0f} (off-grid-high)")
            + f" | wash kill {w['kill_rung']}x"
            + (f" (thr <= {w['threshold']:g}x, bounded)"
               if w.get("bounded_above") else
               f" (thr {w['threshold']:.2g}x)" if w["threshold"] else " (none)"))
        lines.append(f"  iso kill rungs: "
                     f"{[a['kill_rung'] for a in M['organisms'][tag]['iso']]}")
    lines += ["", "BARS (frozen):"]
    for bk, bv in REGISTERED_BARS.items():
        lines.append(f"  {bk}: {bv}")
    lines += ["", v["clause"], "",
              "HONESTY: organ n=1 each; ruler asymmetry (store attractor "
              "readout vs host behavioral battery); single wash-snapshot "
              "direction (drift co-read in metrics); host ruler on the "
              "lineage's discriminative twin organism."]
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, fontsize=7.5,
            va="top", wrap=True, bbox=dict(fc="whitesmoke", ec="gray"))

    fig.suptitle("G3K THE KAPPA CELL — both kappas side by side at matched "
                 "mean per-coordinate RMS (eval-only, CPU; organisms: "
                 "g3_gen.pt + e157_f2_consolidated.pt — the g3 lineage's "
                 "store and host-fact rulers)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("g3K")
    common.DEVICE = "cpu"                       # eval-only; NO GPU claim
    log(f"G3K THE KAPPA CELL -> {rd}; gpu (observed, not claimed): "
        f"{gpu_status()}")

    pr = R.rebuild_protocol()
    log(f"protocol rebuilt (g3 verbatim via g3R; gates {pr['gates']})")

    # ---- instrument check 1+2: organisms load; provenance gates (asserted)
    sd_store, skeys, _, store_gates = R.load_root(pr)      # asserts its gates
    log("STORE organism gates PASS (g3R's load_root)")
    hs = REF["host"]
    sd_host = load_ck(hs["root"])
    assert not any(k.startswith("store.") for k in sd_host), \
        "host organism carries store keys — wrong checkpoint"
    assert sum(v.numel() for v in sd_host.values()) == 873_472, \
        "host organism param count mismatch"
    host_net = TinyGPT(F2_CFG); host_net.load_state_dict(sd_host)
    g0h = battery_cell(host_net, pr["ids130"], pr["zid"])["mean_pz"]
    sd_host_s1 = load_ck(hs["wash_s1"])
    Dh = sd_disp(sd_host_s1, sd_host)
    host_net.load_state_dict(sd_host_s1)
    g0h1 = battery_cell(host_net, pr["ids130"], pr["zid"])["mean_pz"]
    host_gates = {
        "root_ckpt": f"runs/checkpoints/{hs['root']}",
        "wash_s1_ckpt": f"runs/checkpoints/{hs['wash_s1']}",
        "provenance": "runs/g3/metrics.json s_disc_control (verified_vs_file "
                      "true) + runs/e157/metrics.json stages.B_wash",
        "root_g0": g0h, "ref_root_g0": hs["root_g0"],
        "root_g0_reproduces": bool(abs(g0h - hs["root_g0"]) < G_TOL),
        "wash_s1_g0": g0h1, "ref_wash_s1_g0": hs["wash_s1_g0"],
        "wash_s1_g0_reproduces": bool(abs(g0h1 - hs["wash_s1_g0"]) < G_TOL),
        "D_all_s1": Dh, "ref_D_all_s1": hs["D_all_s1"],
        "D_all_s1_reproduces": bool(abs(Dh - hs["D_all_s1"]) < G_TOL),
        "n_params": 873_472,
    }
    host_gates["pass"] = bool(host_gates["root_g0_reproduces"]
                              and host_gates["wash_s1_g0_reproduces"]
                              and host_gates["D_all_s1_reproduces"])
    log(f"HOST ruler gates: root {g0h:.10f}, wash s1 {g0h1:.6e}, D {Dh:.6f} "
        f"-> {'PASS' if host_gates['pass'] else 'FAIL'}")
    assert host_gates["pass"], f"host ruler provenance gate FAILED: {host_gates}"
    del host_net

    # ---- instrument check 3: arm guarantees (stated; echoed into metrics)
    log(f"instrument guarantees (pre-registered): wash: "
        f"{INSTRUMENT_GUARANTEES['wash_arm']}")
    log(f"iso: {INSTRUMENT_GUARANTEES['iso_arm']}")

    # ---- the cell
    store_net = evl_load("gen", sd_store)
    out_store = run_organism(
        "store", sd_store, load_ck(REF["store"]["wash_s1"]), skeys,
        ISO_SEEDS["store"], pr, read_store, lambda: store_net,
        REF["store"]["drift_cks"], read_key="g0")
    out_store["provenance_gates"] = store_gates
    host_net2 = TinyGPT(F2_CFG)
    out_host = run_organism(
        "host", sd_host, sd_host_s1, [],
        ISO_SEEDS["host"], pr, read_host, lambda: host_net2,
        REF["host"]["drift_cks"], read_key="pz")
    out_host["provenance_gates"] = host_gates

    # ---- adjudication (bars verbatim; conservative off-grid handling)
    for tag_, o_ in (("store", out_store), ("host", out_host)):
        assert o_["wash"]["threshold"] is not None, (
            f"{tag_}: wash leg did not kill within the grid — kappa "
            "undefined (the pre-compute gates bound this away: both "
            "committed s1 reads are far below bar); report as a bound")

    ks, kh = (out_store["kappa"], out_host["kappa"])
    ks_val = ks["kappa_value"] if ks["kappa_value"] is not None else np.inf
    kh_val = kh["kappa_value"] if kh["kappa_value"] is not None else np.inf
    ks_lo = ks["kappa_min"] if ks["kappa_min"] is not None else np.inf
    kh_lo = kh["kappa_min"] if kh["kappa_min"] is not None else np.inf
    kh_hi = kh["kappa_max"] if kh["kappa_max"] is not None else np.inf
    pair = (f"kappa_store = "
            + (f"{ks_val:.1f}" if np.isfinite(ks_val) else f"> {ks_lo:.0f}")
            + f" (iso kill rungs {[a['kill_rung'] for a in out_store['iso']]} /"
              f" wash {out_store['wash']['kill_rung']}x)"
            + f"; kappa_host = "
            + (f"{kh_val:.1f}" if np.isfinite(kh_val) else f"> {kh_lo:.0f}")
            + f" (iso kill rungs {[a['kill_rung'] for a in out_host['iso']]} /"
              f" wash {out_host['wash']['kill_rung']}x)")
    if kh_val <= 4 and ks_val >= 8:
        verdict = "KAPPA-SPLIT"
        clause = (REGISTERED_BARS[verdict] + " — FIRED. " + pair
                  + ". At matched mean per-coordinate RMS the store survives "
                    "roughly an order of magnitude more isotropic "
                    "displacement than the wash direction while the host "
                    "fact does not — directional immunity is purchasable "
                    "by attractor readout.")
    elif kh_val <= 4 and ks_val <= 4:
        verdict = "NO-BASIN-UNIVERSAL"
        clause = (REGISTERED_BARS[verdict] + " — FIRED. " + pair
                  + ". Both rulers die to isotropic displacement within 4x "
                    "of their own wash displacement; the critic's 24-32x was "
                    "convention-bound (see the concentration co-read for "
                    "where the wash's energy lives).")
    else:
        verdict = "MIXED"
        clause = (REGISTERED_BARS[verdict] + " — the pair verbatim: " + pair
                  + ". Graded; no architectural claim beyond the pair.")
    log("=" * 78)
    log(f"G3K VERDICT: {verdict}")
    log(f"  {clause}")
    log("=" * 78)

    metrics = {
        "experiment": "g3K_kappa",
        "date": common.now_iso(),
        "purpose": ("THE KAPPA CELL (W022's named measurement): both kappas "
                    "side by side at matched mean per-coordinate RMS — "
                    "kappa_store (g3/g3R organism, g0 ruler) and kappa_host "
                    "(the g3 lineage's own host-fact battery, S-DISC) — one "
                    "convention, one wash snapshot family (committed s1 "
                    "trajectory snapshots, seed 10902, t*=+1 on both)"),
        "design_spec": "scratch/g3K_design.md (committed 2026-09-29, frozen)",
        "registered_bars": dict(REGISTERED_BARS),
        "kill_bars": {"store_g0": KILL_BAR, "host_pz": KILL_BAR,
                      "note": "both 0.5 verbatim from the frozen design; "
                              "CE_COLLATERAL 3.0 is a co-read only"},
        "instrument_guarantees": INSTRUMENT_GUARANTEES,
        "convention": {
            "rungs": list(RUNGS),
            "match": "perturbation total L2 = rung x ||own wash 1x||; mean "
                     "per-coordinate RMS = rung x ||D||/sqrt(N) in both arms",
            "store": {"D_wash_1x": out_store["concentration"]["wash_L2_1x"],
                      "n_params": out_store["concentration"]["n_params"],
                      "mean_rms_1x":
                          out_store["concentration"]["wash_mean_rms_1x"]},
            "host": {"D_wash_1x": out_host["concentration"]["wash_L2_1x"],
                     "n_params": out_host["concentration"]["n_params"],
                     "mean_rms_1x":
                         out_host["concentration"]["wash_mean_rms_1x"]},
        },
        "organisms": {
            "store": {"root": "runs/checkpoints/g3_gen.pt",
                      "wash_direction_source":
                          "runs/checkpoints/g3_gen_s1.pt (committed t*=+1 "
                          "snapshot, wash seed 10902) minus root — FULL "
                          "parameter vector; loaded, not recomputed",
                      "iso_seeds": list(ISO_SEEDS["store"]),
                      "ruler": "g0 = install60 offset-0 battery mean p(Z) "
                               "(the fact through the store's 8-pattern "
                               "retrieval; g3/g3R's committed instrument)",
                      "kill": out_store["wash"],
                      "iso": out_store["iso"],
                      "rows_wash": out_store["rows_wash"],
                      "rows_iso": out_store["rows_iso"],
                      "kappa": out_store["kappa"],
                      "concentration": out_store["concentration"],
                      "drift": out_store["drift"],
                      "provenance_gates": store_gates},
            "host": {"root": "runs/checkpoints/e157_f2_consolidated.pt",
                     "wash_direction_source":
                         "runs/checkpoints/e157_f2_neutral_s1.pt (committed "
                         "t*=+1 snapshot, wash seed 10902) minus root — FULL "
                         "parameter vector; loaded, not recomputed",
                     "iso_seeds": list(ISO_SEEDS["host"]),
                     "ruler": "battery p(Z) = install60 offset-0 mean p(Z) "
                              "on the discriminative host fact (the g3 "
                              "lineage's S-DISC ruler; e185's bar family)",
                     "kill": out_host["wash"],
                     "iso": out_host["iso"],
                     "rows_wash": out_host["rows_wash"],
                     "rows_iso": out_host["rows_iso"],
                     "kappa": out_host["kappa"],
                     "concentration": out_host["concentration"],
                     "drift": out_host["drift"],
                     "provenance_gates": host_gates},
        },
        "adjudication": {"verdict": verdict, "clause": clause,
                         "pair": pair, "bars_verbatim": REGISTERED_BARS,
                         "no_bar_shopping": True},
        "honesty": {
            "organ_n": 1,
            "ruler_asymmetry": "g0 (store attractor readout through "
                "behavior) vs battery p(Z) on a discriminative host fact — "
                "different dynamics; the kappa PAIR is the finding, never a "
                "single number; both kill defs stated on every curve",
            "single_wash_snapshot": "the wash direction is ONE t*=+1 "
                "trajectory snapshot per organism (g3R's convention); the "
                "wash drifts — drift co-reads reported "
                f"(store {out_store['drift']}, host {out_host['drift']})",
            "host_ruler_scope": "the host-fact ruler is read on the "
                "lineage's SECOND organism (the discriminative twin), not on "
                "g3_gen itself — same family/battery/wash; n=1 per organism",
            "iso_seeds": "3 seeds per organism; mean adjudicated, min/max + "
                "straddle flags co-reported; no seed dropped",
        },
        "gates": {"protocol": pr["gates"], "gpu_observed_at_start":
                  gpu_status(), "gpu_claimed": False, "device": "cpu"},
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "threads": torch.get_num_threads(),
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))
    log("metrics.json written")
    plot(rd / "kappa.png", metrics)
    log(f"plot written -> {rd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
