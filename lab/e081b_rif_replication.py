"""E081b — T049-registered RIF replication cell on e048_repro (2026-09-25).

Eval-only, CPU-only, NO training, NO weight edits (verbatim e081 machinery:
same session layout, same arms/readouts, same batteries — imported, not
rewritten). This is the registered discriminator from THINKING.md T049:

  REGISTERED (frozen in T049):
    - H-pure  predicts the null replicates with a FLAT sham
      (per-window |delta| 95th pct < 0.02)
    - H-coarse predicts the sham moves again (noise floor > 0.04)
    - null replicates AND flat sham  -> reads-are-pure confirmed at n=2,
      T049 closes clean
    - EL->FL scramble texture (dose: +0.094 [0.011, 0.178], report-only)
      gets one look: if it replicates with CI excluding 0, FLAG IT LOUDLY.

UPGRADES over e081 (all registered in T049 / documented here):
  (a) TIGHT SHAM: sham arms at B=96 resamples (3x e081's 32; each resample
      redraws the 112-token sham segment) -> per-window paired half-split
      deltas (48 vs 48, e081's first-half/second-half placebo convention);
      the 95th percentile of |per-window delta| is the instrument-noise
      floor. Robustness: 200 random disjoint half-splits (median + 80% CI).
      Comparability: e081-style 16-vs-16 placebo split on the first 32
      resamples (dose reference: FL 0.0441 / EL 0.0206 — the failure that
      motivated T049).
  (b) scramble control kept, at B=32 (e081 ran it at B=16; the replication
      question needs the tighter elicitation noise — deviation documented,
      non-gating leg as before).

ARMS (all on e048_repro only; e048_dose is e081's cell, loaded read-only
from runs/e081/metrics.json for comparison):
  base FL/EL/Z @129 (instrument Z gate vs published 0.5563087304433186,
  |drift| < 5e-3); read-FL -> probe-EL (B=32); read-EL -> probe-FL (B=32);
  TIGHT sham -> EL / FL (B=96); read-scrFL -> probe-EL, read-scrEL ->
  probe-FL (B=32, anagram controls LFEORZIL / ZIBLETHEA).

VERDICT RULES (registered): fired direction = supp mean >= 0.05 with
per-window t-CI excluding 0 (e081 rule, verbatim); null replicates iff no
direction fires on repro (dose fired none); sham flat iff BOTH probes'
canonical-split |delta| p95 < 0.02; sham coarse iff ANY probe's > 0.04;
texture replicates iff scr_EL_to_FL mean > 0 with t-CI excluding 0 (boot CI
reported alongside; dose had both barely excluding 0).

Outputs: runs/e081b/{metrics.json, rif_replication.png}.
Run: python lab/e081b_rif_replication.py
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                    # CPU experiment

import random as _random                                    # noqa: E402
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402

torch.set_num_threads(min(16, os.cpu_count() or 8))

import common                                               # noqa: E402
common.DEVICE = "cpu"
from common import run_dir, save_json                       # noqa: E402
import e043_install as E43                                  # noqa: E402
import e081_rif_probe as E81                                # noqa: E402
from e081_rif_probe import (arm_stats, build_pools, delta_cell, load,   # noqa: E402
                            rebuild_protocol, run_cell)
import matplotlib                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                             # noqa: E402

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ---- registered constants (e081 verbatim + T049 tight-sham additions) ------
CKPT_NAME, CKPT_FILE = "e048_repro", "e048_repro.pt"
Z_GATE_REPRO = E81.Z_GATE[CKPT_NAME]            # 0.5563087304433186 (e068)
B_RESAMPLE = E81.B_RESAMPLE                     # 32 (primary arms, as e081)
B_SCRAMBLE = E81.B_RESAMPLE                     # 32 (upgrade: e081 used 16)
B_TIGHT = 3 * E81.B_RESAMPLE                    # 96 (T049: 3x sham windows)
FLAT_BAR = 0.02          # T049: H-pure flat sham, per-window |delta| p95
COARSE_BAR = 0.04        # T049: H-coarse sham moves again
SUPP_BAR = E81.SUPP_BAR                        # 0.05
SHAM_NOISE_BAR = E81.SHAM_NOISE_BAR            # 0.01 (e081 placebo gate)
CI_Z = E81.CI_Z
N_SPLITS = 200           # random disjoint half-splits (robustness)

REGISTERED = {
    "question": ("T049 discriminator: does the e081 RIF null replicate on "
                 "e048_repro, and is the sham FLAT (H-pure) or does it move "
                 "again (H-coarse)?"),
    "t049_frozen": {
        "H_pure": "null replicates with flat sham (per-window |delta| "
                  "95th pct < 0.02)",
        "H_coarse": "sham moves again (noise floor > 0.04)",
        "close_clean": "null replicates AND flat sham -> reads-are-pure "
                       "confirmed at n=2, T049 closes clean",
        "texture": "EL->FL scramble texture (dose +0.094 [0.011,0.178]): "
                   "one look; replicates with CI excluding 0 -> FLAG LOUDLY",
    },
    "machinery": "verbatim e081_rif_probe (imported): session layout "
                 "[prefix 112]['\\n\\n'][ctx 130][name], readout rows "
                 "129/243, batteries FL=29/EL=61/Z=60, onset p + joint "
                 "readouts, clustered bootstrap CIs, logit deltas",
    "checkpoint": CKPT_NAME,
    "arms": ["base FL/EL/Z @129 (instrument gate)",
             f"read-FL -> probe-EL (B={B_RESAMPLE})",
             f"read-EL -> probe-FL (B={B_RESAMPLE})",
             f"TIGHT sham -> probe-EL (B={B_TIGHT}, 3x)",
             f"TIGHT sham -> probe-FL (B={B_TIGHT}, 3x)",
             f"read-scrFL -> probe-EL (B={B_SCRAMBLE})",
             f"read-scrEL -> probe-FL (B={B_SCRAMBLE})"],
    "noise_floor_statistic": ("tight sham split 48-vs-48 (first-half/"
                              "second-half, e081 placebo convention); "
                              "per-window paired delta; floor = 95th pct "
                              "of |delta|; robustness = 200 random disjoint "
                              "half-splits"),
    "deviations_from_e081": [
        f"(a) sham arms at B={B_TIGHT} (3x) — the T049 tight sham",
        f"(b) scramble legs at B={B_SCRAMBLE} (e081: 16) — tighter "
        "elicitation noise for the CI-excluding-0 replication check",
        "repro checkpoint only (dose cell is e081's, loaded read-only)",
        f"independent frozen seed base 8150 (e081 used 8100)",
    ],
    "bars": {"flat": FLAT_BAR, "coarse": COARSE_BAR, "supp": SUPP_BAR,
             "sham_noise_mean": SHAM_NOISE_BAR, "ci_z": CI_Z},
}
log("REGISTERED: " + " | ".join(f"{k}: {v}" for k, v in REGISTERED.items()))


def tight_sham_floor(P, n_splits=N_SPLITS, seed=99):
    """P: (B, n_win) onset probs of a tight-sham arm. Canonical split =
    first half vs second half (e081 placebo convention); robustness =
    random disjoint half-splits."""
    B, n = P.shape
    h = B // 2
    d_can = P[:h].mean(axis=0) - P[h:2 * h].mean(axis=0)
    p95_can = float(np.percentile(np.abs(d_can), 95))
    rng = np.random.default_rng(seed)
    p95s, means = [], []
    for _ in range(n_splits):
        idx = rng.permutation(B)
        d = P[idx[:h]].mean(axis=0) - P[idx[h:2 * h]].mean(axis=0)
        p95s.append(float(np.percentile(np.abs(d), 95)))
        means.append(float(d.mean()))
    return {
        "n_resamples": int(B), "n_windows": int(n),
        "canonical_split": {"per_window_delta": d_can.tolist(),
                            "abs_p95": p95_can,
                            "mean": float(d_can.mean())},
        "random_half_splits": {
            "n": n_splits,
            "abs_p95_median": float(np.median(p95s)),
            "abs_p95_pct10_90": [float(np.percentile(p95s, 10)),
                                 float(np.percentile(p95s, 90))],
            "abs_mean_median": float(np.median(np.abs(means))),
            "abs_mean_max": float(np.max(np.abs(means)))},
    }


def load_dose_ref():
    """e081's dose cell, read-only, for side-by-side reporting."""
    p = E43.REPO / "runs" / "e081" / "metrics.json"
    if not p.exists():
        log("WARNING: runs/e081/metrics.json not found — no dose reference")
        return None
    m = json.loads(p.read_text(encoding="utf-8"))
    ck = m["verdicts"].get("e048_dose", {})
    dd = m["deltas_suppression"].get("e048_dose", {})
    cells = m.get("cells", {}).get("e048_dose", {})
    legs = ("cross_FL_to_EL", "cross_EL_to_FL", "scr_FL_to_EL", "scr_EL_to_FL")
    return {
        "overall": m.get("overall"),
        "placebo_e081_gate": ck.get("gate_placebo"),
        "e081_gate_pass": ck.get("gate_pass"),
        "legs": {k: {"mean": dd[k]["mean"], "ci95": dd[k]["ci95"],
                     "boot_ci95": dd[k]["delta_bootstrap_ci95"]}
                 for k in legs if k in dd},
        "arm_onset_means": {k: v["onset"]["mean"] for k, v in cells.items()},
    }


# ------------------------------------------------------------------ main
def main():
    rd = run_dir("e081b")
    log("E081b (T049 discriminator): RIF null replication on e048_repro "
        "with a tight sham — eval-only, no weight edits")

    corpus, train_text, install_occ, probes, mix, held_mix = rebuild_protocol()
    pools, pool_flags = build_pools(train_text, install_occ, probes)
    log(f"protocol rebuilt (verbatim e081): install60 {mix}, held30 "
        f"{held_mix}; probes FL={len(probes['FL'])} EL={len(probes['EL'])} "
        f"Z={len(probes['Z'])}")

    net = load(E43.REPO / "runs" / "checkpoints" / CKPT_FILE)
    w_sig = (net.wpe.weight.data.clone(), net.wte.weight.data.clone())
    log(f"net {CKPT_NAME}: params {net.num_params():,}")

    CELLS = [  # (read, probe, n_res) — order fixes the frozen RNG seeds
        ("FL", "EL", B_RESAMPLE), ("EL", "FL", B_RESAMPLE),       # primary
        ("sham", "EL", B_TIGHT), ("sham", "FL", B_TIGHT),         # TIGHT
        ("scrFL", "EL", B_SCRAMBLE), ("scrEL", "FL", B_SCRAMBLE),  # control
    ]
    res = {}
    # canonical baselines first (instrument gates; identical across resamples)
    for probe_key in ("FL", "EL", "Z"):
        rng = _random.Random(8150 + 900 + {"FL": 3, "EL": 1, "Z": 7}[probe_key])
        c = run_cell(net, corpus, train_text, "sham", probe_key,
                     probes[probe_key], None, rng, shifted=False)
        res[f"base__probe{probe_key}"] = c
        a = arm_stats(c)
        log(f"[{CKPT_NAME}] baseline probe-{probe_key} @129: "
            f"p(first) {a['onset']['mean']:.4f} "
            f"[{a['onset']['ci95'][0]:.4f},{a['onset']['ci95'][1]:.4f}] "
            f"joint {a['joint_mean']:.2e}")
    gz = res["base__probeZ"]["P"].mean()
    drift = abs(gz - Z_GATE_REPRO)
    gate = {"pz_canonical": float(gz), "pz_published": Z_GATE_REPRO,
            "drift": float(drift), "pass": bool(drift < 5e-3)}
    assert gate["pass"], (f"{CKPT_NAME} Z gate drift {drift:.2e} "
                          f"(protocol identity failed)")
    log(f"[{CKPT_NAME}] instrument gate: p(Z)@129 {gz:.7f} vs published "
        f"{Z_GATE_REPRO:.7f} (|drift| {drift:.2e}, OK)")

    # shifted arms (independent frozen seed base 8150)
    for ci, (read, probe, n_res) in enumerate(CELLS):
        rng = _random.Random(8150 + ci * 100)
        pool_key = read[3:] if read.startswith("scr") else read
        c = run_cell(net, corpus, train_text, read, probe,
                     probes[probe], pools.get((pool_key, probe)), rng,
                     shifted=True, n_res=n_res)
        res[f"read{read}__probe{probe}"] = c
        a = arm_stats(c)
        log(f"[{CKPT_NAME}] read-{read:5s} -> probe-{probe} @243: "
            f"p(first) {a['onset']['mean']:.4f} "
            f"[{a['onset']['ci95'][0]:.4f},{a['onset']['ci95'][1]:.4f}] "
            f"joint {a['joint_mean']:.2e} ({a['n_sessions']} sessions)")

    # no-weight-edit check
    assert torch.equal(net.wpe.weight.data, w_sig[0]) and \
        torch.equal(net.wte.weight.data, w_sig[1]), "weights changed!"
    log(f"[{CKPT_NAME}] no-weight-edit check passed")

    # ---- deltas (verbatim e081 statistic, tight sham as the null arm)
    d = {
        "cross_FL_to_EL": delta_cell(res["readsham__probeEL"],
                                     res["readFL__probeEL"]),
        "cross_EL_to_FL": delta_cell(res["readsham__probeFL"],
                                     res["readEL__probeFL"]),
        "scr_FL_to_EL": delta_cell(res["readsham__probeEL"],
                                   res["readscrFL__probeEL"]),
        "scr_EL_to_FL": delta_cell(res["readsham__probeFL"],
                                   res["readscrEL__probeFL"]),
    }
    for k, s in d.items():
        log(f"[{CKPT_NAME}] {k}: {s['mean']:+.4f} CI "
            f"[{s['ci95'][0]:+.4f},{s['ci95'][1]:+.4f}] boot "
            f"[{s['delta_bootstrap_ci95'][0]:+.4f},"
            f"{s['delta_bootstrap_ci95'][1]:+.4f}] "
            f"({s['frac_windows_suppressed']:.0%} windows suppressed)")

    # ---- TIGHT SHAM noise floor (the registered discriminator)
    floors = {pk: tight_sham_floor(res[f"readsham__probe{pk}"]["P"])
              for pk in ("EL", "FL")}
    placebo = {pk: floors[pk]["canonical_split"]["mean"] for pk in floors}
    placebo_e081 = {}
    for pk in ("EL", "FL"):
        P = res[f"readsham__probe{pk}"]["P"][:32]     # e081's B=32 arm, 16v16
        placebo_e081[pk] = float((P[:16].mean(axis=0)
                                  - P[16:32].mean(axis=0)).mean())
    p95_can = {pk: floors[pk]["canonical_split"]["abs_p95"] for pk in floors}
    p95_med = {pk: floors[pk]["random_half_splits"]["abs_p95_median"]
               for pk in floors}
    sham_flat = all(v < FLAT_BAR for v in p95_can.values())
    sham_coarse = any(v > COARSE_BAR for v in p95_can.values())
    gate_pass_tight = all(abs(v) < SHAM_NOISE_BAR for v in placebo.values())
    for pk in floors:
        f = floors[pk]
        log(f"[{CKPT_NAME}] TIGHT SHAM probe-{pk}: floor |d|p95 "
            f"{p95_can[pk]:.4f} (200-split median {p95_med[pk]:.4f}, "
            f"80% CI [{f['random_half_splits']['abs_p95_pct10_90'][0]:.4f},"
            f"{f['random_half_splits']['abs_p95_pct10_90'][1]:.4f}]) | "
            f"tight 48v48 mean placebo {placebo[pk]:+.4f} | e081-style 16v16 "
            f"{placebo_e081[pk]:+.4f}")

    # ---- verdicts (registered rules)
    fired = [k for k in ("cross_FL_to_EL", "cross_EL_to_FL")
             if d[k]["mean"] >= SUPP_BAR and d[k]["ci95"][0] > 0]
    null_replicates = not fired                      # dose fired none (NULL)
    tex = d["scr_EL_to_FL"]
    texture_replicates = bool(tex["mean"] > 0 and tex["ci95"][0] > 0)
    h_pure = bool(null_replicates and sham_flat and not sham_coarse)
    h_coarse = bool(sham_coarse)
    if h_pure:
        overall = ("T049 CLOSES CLEAN: the null replicates on e048_repro "
                   "AND the tight sham is FLAT — reads-are-pure confirmed "
                   "at n=2 (H-pure takes the verdict; H-coarse rejected)")
    elif h_coarse:
        overall = ("T049: H-COARSE — the tight sham still moves (floor > "
                   f"{COARSE_BAR}); instrument artifact dominates, purity "
                   "cannot be certified at this resolution")
    else:
        overall = ("T049: INTERMEDIATE — null "
                   f"{'replicates' if null_replicates else 'DOES NOT replicate'}"
                   f", sham floor {p95_can} lands between the flat "
                   f"({FLAT_BAR}) and coarse ({COARSE_BAR}) bars; honest "
                   "report, no clean close")
    if texture_replicates:
        overall += (" *** FLAG (loudly): the EL->FL scramble texture "
                    "REPLICATES with CI excluding 0 — an asymmetry worth "
                    "its own card ***")
    log(f"[{CKPT_NAME}] null_replicates={null_replicates} "
        f"sham_flat={sham_flat} sham_coarse={sham_coarse} "
        f"h_pure={h_pure} h_coarse={h_coarse} "
        f"texture_replicates={texture_replicates}")
    log(f"T049 VERDICT: {overall}")

    dose_ref = load_dose_ref()

    # ---- metrics
    def cells_json(res_):
        return {k: {**arm_stats(c), "row": c["row"], "sess_len": c["sess_len"],
                    "P_flat": c["P"].flatten().tolist()}
                for k, c in res_.items()}

    out = {
        "experiment": "e081b_rif_replication",
        "t049_source": "THINKING.md T049 (2026-09-26 ~14:05Z)",
        "registered": REGISTERED,
        "protocol": {
            "machinery": "e081_rif_probe imported verbatim",
            "corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
            "install_mix": mix, "held_mix": held_mix,
            "probe_battery_sizes": {k: len(v) for k, v in probes.items()},
            "prefix_len": E81.PREFIX, "separator": E81.SEP,
            "readout_rows": {"shifted": E81.ROW_SHIFT,
                             "canonical": E81.ROW_CANON},
            "resamples": {"primary": B_RESAMPLE, "tight_sham": B_TIGHT,
                          "scramble": B_SCRAMBLE,
                          "bootstrap_draws": E81.N_BOOT,
                          "half_split_draws": N_SPLITS},
            "pool_flags": pool_flags,
        },
        "gates_instrument": {"z_gate": gate,
                             "no_weight_edit": True},
        "cells": cells_json(res),
        "deltas": {k: {kk: vv for kk, vv in v.items()
                       if kk != "per_window_delta"} for k, v in d.items()},
        "noise_floor": {
            "statistic": "95th pct of |per-window paired half-split delta| "
                         "on the tight sham (48 vs 48)",
            "floors": floors,
            "p95_canonical": p95_can, "p95_split_median": p95_med,
            "placebo_mean_tight_48v48": placebo,
            "placebo_mean_e081_style_16v16": placebo_e081,
            "dose_reference_placebo_e081_style":
                (dose_ref or {}).get("placebo_e081_gate"),
            "bars": {"flat": FLAT_BAR, "coarse": COARSE_BAR},
            "sham_flat": bool(sham_flat), "sham_coarse": bool(sham_coarse),
        },
        "verdicts": {
            "fired_directions": fired,
            "e081_placebo_gate_at_tight_resolution": bool(gate_pass_tight),
            "null_replicates": bool(null_replicates),
            "sham_flat": bool(sham_flat), "sham_coarse": bool(sham_coarse),
            "H_pure_supported": h_pure, "H_coarse_supported": h_coarse,
            "scramble_texture_EL_to_FL": {
                "mean": tex["mean"], "ci95": tex["ci95"],
                "boot_ci95": tex["delta_bootstrap_ci95"],
                "replicates_ci_excluding_0": texture_replicates,
                "loud_flag": ("*** EL->FL scramble texture REPLICATES with "
                              "CI excluding 0 — an asymmetry worth its own "
                              "card ***" if texture_replicates else None)},
            "overall": overall,
        },
        "dose_reference_e081": dose_ref,
        "elapsed_s": round(time.time() - T0, 1),
        "cpu_threads": torch.get_num_threads(),
    }
    save_json(rd / "metrics.json", out)

    # ---- figure
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10))
    dr = (dose_ref or {}).get("arm_onset_means", {})

    # (0,0) primary arms: read-other vs tight sham vs canonical base
    ax = axes[0, 0]
    xs = np.arange(2)
    bw = 0.2
    series = [
        ("read-OTHER name", ["readFL__probeEL", "readEL__probeFL"], "crimson"),
        (f"TIGHT sham (B={B_TIGHT})", ["readsham__probeEL", "readsham__probeFL"],
         "steelblue"),
        ("canonical base @129", ["base__probeEL", "base__probeFL"], "gray"),
    ]
    for j, (lbl, keys, col) in enumerate(series):
        vals = [arm_stats(res[k])["onset"]["mean"] for k in keys]
        errs = [CI_Z * arm_stats(res[k])["onset"]["sem"] for k in keys]
        ax.bar(xs + (j - 1) * bw, vals, bw * 0.92, yerr=errs, capsize=3,
               color=col, edgecolor="k", linewidth=0.4, label=lbl)
        for x, v in zip(xs + (j - 1) * bw, vals):
            ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=7,
                    rotation=90, va="bottom")
    for j, (keys, col, mk) in enumerate((
            (["readFL__probeEL", "readEL__probeFL"], "crimson", "o"),
            (["readsham__probeEL", "readsham__probeFL"], "steelblue", "s"))):
        vals = [dr.get(k) for k in keys]
        if all(v is not None for v in vals):
            ax.plot(xs + (j - 1) * bw, vals, mk, mfc="none", mec=col, ms=9,
                    mew=1.6, label=f"e081 dose ({'read' if j == 0 else 'sham'})")
    ax.set_xticks(xs)
    ax.set_xticklabels(["probe ELIZABETH\n(61 windows)",
                        "probe FLORIZEL\n(29 windows)"])
    ax.set_ylabel("onset p(first name char)")
    s1, s2 = d["cross_FL_to_EL"], d["cross_EL_to_FL"]
    ax.set_title(f"PRIMARY on e048_repro (dose = open markers): supp FL->EL "
                 f"{s1['mean']:+.4f} CI[{s1['ci95'][0]:+.4f},"
                 f"{s1['ci95'][1]:+.4f}] | supp EL->FL {s2['mean']:+.4f} "
                 f"CI[{s2['ci95'][0]:+.4f},{s2['ci95'][1]:+.4f}] | fired: "
                 f"{fired if fired else 'none — null replicates' if null_replicates else fired}",
                 fontsize=8.5)
    ax.legend(fontsize=7, ncols=2)
    ax.set_ylim(0, 1.02)

    # (0,1) TIGHT SHAM noise floor
    ax = axes[0, 1]
    for pk, col in (("EL", "steelblue"), ("FL", "seagreen")):
        dw = np.abs(np.array(floors[pk]["canonical_split"]["per_window_delta"]))
        ax.hist(dw, bins=14, alpha=0.55, color=col, edgecolor="k",
                linewidth=0.3,
                label=f"probe {pk}: |d| p95 {p95_can[pk]:.4f} "
                      f"(split-median {p95_med[pk]:.4f}), n={len(dw)}")
    ax.axvline(FLAT_BAR, color="seagreen", ls="--", lw=1.6,
               label=f"H-pure flat bar {FLAT_BAR}")
    ax.axvline(COARSE_BAR, color="red", ls="--", lw=1.6,
               label=f"H-coarse bar {COARSE_BAR}")
    ax.set_xlabel("|per-window paired delta|  (tight sham 48 vs 48, B=96)")
    ax.set_ylabel("probe windows")
    dose_pl = (dose_ref or {}).get("placebo_e081_gate") or {}
    ax.set_title(f"TIGHT SHAM noise floor (e048_repro): "
                 f"{'FLAT' if sham_flat else 'NOT flat'}"
                 f"{' / COARSE' if sham_coarse else ''} | tight 48v48 mean "
                 f"placebo { {k: round(v, 4) for k, v in placebo.items()} } | "
                 f"e081-style 16v16 { {k: round(v, 4) for k, v in placebo_e081.items()} } "
                 f"(dose e081: { {k: round(v, 4) for k, v in dose_pl.items()} if dose_pl else 'n/a'} )",
                 fontsize=8)
    ax.legend(fontsize=7.5)

    # (1,0) all legs with CIs, repro bars vs dose ghosts
    ax = axes[1, 0]
    legs = ["cross_FL_to_EL", "cross_EL_to_FL", "scr_FL_to_EL", "scr_EL_to_FL"]
    lbls = ["FL->EL\n(CROSS a)", "EL->FL\n(CROSS c)",
            "scrFL->EL\n(anagram)", "scrEL->FL\n(anagram TEXTURE)"]
    xs = np.arange(len(legs))
    vals = [d[k]["mean"] for k in legs]
    errs = [CI_Z * d[k]["sem"] for k in legs]
    ax.bar(xs, vals, 0.5, yerr=errs, capsize=3, color="steelblue",
           edgecolor="k", linewidth=0.4, label="e048_repro (this cell)")
    dl = (dose_ref or {}).get("legs", {})
    for k, x in zip(legs, xs):
        if k in dl:
            ax.hlines(dl[k]["mean"], x - 0.32, x + 0.32, color="crimson",
                      lw=2.0)
            lo, hi = dl[k]["ci95"]
            ax.vlines(x, lo, hi, color="crimson", lw=1.2, ls=":")
    ax.hlines(0, -0.4, len(legs) - 0.6, color="k", lw=1)
    ax.axhline(SUPP_BAR, color="seagreen", ls="--", lw=1.2,
               label=f"RIF bar +{SUPP_BAR}")
    ax.axhline(-SUPP_BAR, color="purple", ls=":", lw=1.2,
               label=f"facilitation -{SUPP_BAR}")
    ax.plot([], [], color="crimson", lw=2, ls=":",
            label="e081 dose mean:tCI (ghost)")
    ax.set_xticks(xs)
    ax.set_xticklabels(lbls, fontsize=8)
    ax.set_ylabel("delta: p(sham) - p(read)")
    ax.set_title("All legs, e048_repro bars + e081 dose ghosts — positive "
                 "= the read SUPPRESSES the probe", fontsize=9.5)
    ax.legend(fontsize=8)

    # (1,1) scramble texture focus
    ax = axes[1, 1]
    dw = np.array(d["scr_EL_to_FL"]["per_window_delta"])
    ax.hist(dw, bins=12, alpha=0.6, color="darkorange", edgecolor="k",
            linewidth=0.3,
            label=f"repro per-window: mean {dw.mean():+.4f}, "
                  f"{int((dw > 0).sum())}/{len(dw)} windows supp")
    ax.axvline(0, color="k", lw=1)
    if "scr_EL_to_FL" in dl:
        dd_ = dl["scr_EL_to_FL"]
        ax.axvline(dd_["mean"], color="crimson", lw=2,
                   label=f"dose mean {dd_['mean']:+.4f} "
                         f"CI[{dd_['ci95'][0]:+.4f},{dd_['ci95'][1]:+.4f}]")
    ax.set_xlabel("per-window delta: p(sham) - p(read-scrEL) on probe FL")
    ax.set_ylabel("probe windows")
    flag = ("*** REPLICATES: CI excludes 0 — FLAG IT LOUDLY ***"
            if texture_replicates else
            "does NOT replicate (CI includes 0)")
    ax.set_title(f"SCRAMBLE TEXTURE (read ZIBLETHEA -> probe FLORIZEL)\n"
                 f"repro {tex['mean']:+.4f} CI[{tex['ci95'][0]:+.4f},"
                 f"{tex['ci95'][1]:+.4f}] boot "
                 f"[{tex['delta_bootstrap_ci95'][0]:+.4f},"
                 f"{tex['delta_bootstrap_ci95'][1]:+.4f}] — {flag}",
                 fontsize=9,
                 color="darkred" if texture_replicates else "k")
    ax.legend(fontsize=7.5)

    fig.suptitle(f"e081b (T049 discriminator) — {overall}\n"
                 f"e048_repro cell, verbatim e081 machinery, tight sham "
                 f"B={B_TIGHT}; eval-only, no weight edits", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(rd / "rif_replication.png", dpi=140)
    log(f"done -> {rd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
