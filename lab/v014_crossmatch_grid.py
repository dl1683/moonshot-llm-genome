"""V014 — THE CROSSMATCH GRID: P2 IMMUNOLOGY's standing visualization (CPU-only).

The visual form of THINKING.md T046 (E062): the pre-graft crossmatch.
Every number is lifted verbatim from registered run metrics
(runs/e062/metrics.json — nothing recomputed, GPU untouched):

  MAIN   hosts x donors matrix, cell color = graft damage D (viridis);
         the DECISION RULE is drawn on top: cells passing
         cos_graftinput >= 0.4459 (Youden, in-sample) get a green frame
         -> visual claim: high-cosine region ~ low-damage region.
         Kinship strips color the axes by lineage family (e040 w/m/g
         descendants vs fresh seed vs e050 directed-mutation hosts).
  SIDE   the instrument in one glance: cos vs D scatter with the 0.4459
         threshold line + ROC inset (AUC 0.919).
  NOTES  kinship legend + the honesty footnotes (in-sample threshold,
         partial r, donor-redundancy caveat).

Outputs: runs/v014/crossmatch_grid.png, runs/v014/metrics.json.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-ONLY by lab rule (GPU busy)

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch, Rectangle

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
SRC = RUNS / "e062"
OUT = RUNS / "v014"
OUT.mkdir(parents=True, exist_ok=True)

T0 = time.time()
M = json.loads((SRC / "metrics.json").read_text(encoding="utf-8"))
PAIRS = M["pairs"]                          # 204 scale-A host<-donor pairs
ROC = M["roc"]["P2_gi"]                     # AUC 0.919 instrument
CORR = M["correlations_scaleA"]["P2_gi"]
MED_D = M["label_definition"]["median_D"]   # label = D > median(D)
THR = ROC["youden_threshold"]               # 0.4459490... (in-sample)
CI = CORR["partial_ci95_hostcluster"]

# lab palette conventions (e062/v010)
C_KIN, C_FRESH, C_50 = "#2c6fbb", "#e67e22", "#95a5a6"
C_RULE, C_REF, C_M, C_G = "#2ecc71", "#c0392b", "#8e44ad", "#27ae60"
C_GRAY = "#888888"

# ---------------------------------------------------------------- kinship
# Lineage family from checkpoint names (T046 / CLAIM-D caveat: 11/12 donors
# are e040 seed-42 kin; the bootstrap CI is therefore anti-conservative).
POOL = ["e040_ref", "e040_w", "e040_m1", "e040_m2", "e040_m3",
        "e040_g1a", "e040_g1b", "e040_g1c", "e040_g2a", "e040_g2b",
        "e040_g2c", "e005s_small"]
HOSTS = POOL + ["e050_m1", "e050_m2", "e050_m3",
                "e050_g1a", "e050_g1b", "e050_g1c"]
FAMILY = {}
for n in POOL + HOSTS:
    if n.startswith("e040_ref"):
        f = "ancestor (e040_ref)"
    elif n.startswith("e040_w"):
        f = "w lineage"
    elif n.startswith("e040_m"):
        f = "m lineage"
    elif n.startswith("e040_g"):
        f = "g lineage"
    elif n.startswith("e005"):
        f = "fresh seed (e005)"
    else:
        f = "e050 directed-mutation (hosts only)"
    FAMILY[n] = f
FCOL = {"ancestor (e040_ref)": C_REF, "w lineage": "#9b59b6",
        "m lineage": C_KIN, "g lineage": C_G, "fresh seed (e005)": C_FRESH,
        "e050 directed-mutation (hosts only)": C_50}

# ---------------------------------------------------------------- grid data
# Aggregate per (host, donor): mean D / mean cos over however many
# measurements the pair has; annotate the cell with "xn" if n > 1.
cell = defaultdict(list)
for p in PAIRS:
    cell[(p["host"], p["donor"])].append(p)
NH, ND = len(HOSTS), len(POOL)
D = np.full((NH, ND), np.nan)
CG = np.full((NH, ND), np.nan)
CNT = np.zeros((NH, ND), dtype=int)
for (h, d), rows in cell.items():
    i, j = HOSTS.index(h), POOL.index(d)
    D[i, j] = float(np.mean([r["D"] for r in rows]))
    CG[i, j] = float(np.mean([r["cos_graftinput"] for r in rows]))
    CNT[i, j] = len(rows)
for i, h in enumerate(HOSTS):               # self-transplants not grafted
    if h in POOL:
        j = POOL.index(h)
        assert np.isnan(D[i, j]), (h, "self cell unexpectedly measured")

# decision rule applied cellwise (threshold verbatim from e062 ROC)
PASS = (CG >= THR) & ~np.isnan(D)          # NaN self-cells never "pass"
FAIL = ~PASS & ~np.isnan(D)

# pair-level confusion at the threshold (compatible = D <= median)
valid = ~np.isnan(D)
n_pass = int(PASS.sum())
n_fail = int(FAIL.sum())
compat = (D <= MED_D) & valid
tp = int((PASS & compat).sum())   # compatible grafts waved through
fn = int((FAIL & compat).sum())
fp = int((PASS & ~compat).sum())  # damaging grafts waved through
tn = int((FAIL & ~compat).sum())
assert tp + fn + fp + tn == 204, (tp, fn, fp, tn)
tpr_thr = tp / max(tp + fn, 1)
fpr_thr = fp / max(fp + tn, 1)
mean_D_pass = float(np.nanmean(D[PASS]))
mean_D_fail = float(np.nanmean(D[FAIL]))

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(16.5, 10.5))
outer = GridSpec(1, 2, figure=fig, width_ratios=[1.62, 1.0],
                 wspace=0.22, left=0.055, right=0.985, top=0.885, bottom=0.10)

# left block: kinship strips + main matrix -------------------------------
gleft = GridSpecFromSubplotSpec(
    2, 2, subplot_spec=outer[0, 0],
    height_ratios=[0.035, 1], width_ratios=[0.028, 1],
    hspace=0.04, wspace=0.035)
ax_top = fig.add_subplot(gleft[0, 1])       # donor family strip
ax_left = fig.add_subplot(gleft[1, 0])      # host family strip
ax = fig.add_subplot(gleft[1, 1])           # MAIN PANEL

masked = np.ma.masked_invalid(D)
cmap = plt.get_cmap("viridis").copy()
cmap.set_bad("#ffffff")
im = ax.imshow(masked, cmap=cmap, aspect="auto",
               vmin=float(np.nanmin(D)), vmax=float(np.nanmax(D)))

# self / unmeasured cells: hatched gray
for (i, j) in zip(*np.where(np.isnan(D))):
    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="#e8e8e8",
                           edgecolor="#bbbbbb", lw=0.4, hatch="///",
                           zorder=2))

# cell text: D value (nats); "xn" count annotation if a pair was measured
# more than once (all n=1 in e062 — kept honest by construction)
vmin, vmax = float(np.nanmin(D)), float(np.nanmax(D))
for (i, j) in zip(*np.where(~np.isnan(D))):
    v = (D[i, j] - vmin) / (vmax - vmin)
    ax.text(j, i + (0.16 if CNT[i, j] > 1 else 0), f"{D[i, j]:.1f}",
            ha="center", va="center", fontsize=6.2,
            color="white" if v < 0.55 else "black", zorder=4)
    if CNT[i, j] > 1:
        ax.text(j, i - 0.27, f"x{CNT[i, j]}", ha="center", va="center",
                fontsize=5.2, color=C_RULE, zorder=4, fontweight="bold")

# DECISION RULE overlay: green frame on graft-if cells (cos >= 0.4459)
for (i, j) in zip(*np.where(PASS)):
    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                           edgecolor=C_RULE, lw=1.9, zorder=5))

ax.set_xticks(range(ND))
ax.set_xticklabels(POOL, rotation=90, fontsize=7.5)
ax.set_yticks(range(NH))
ax.set_yticklabels(HOSTS, fontsize=7.5)
for lbl in ax.get_xticklabels():
    lbl.set_color(FCOL[FAMILY[lbl.get_text()]])
for lbl in ax.get_yticklabels():
    lbl.set_color(FCOL[FAMILY[lbl.get_text()]])
ax.set_xlabel("donor (graft MLP organs L2/L3)", fontsize=9)
ax.set_ylabel("host", fontsize=9)
ax.set_title(f"graft damage D — green frame = PASS: cos_graftinput "
             f">= {THR:.4f}; gray hatch = self (excluded)", fontsize=10)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)

cb = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.015)
cb.set_label("D — graft damage (mean dCE, host<-donor MLP L2/L3, 15 batches)",
             fontsize=8)
cb.ax.tick_params(labelsize=7)

# kinship strips (lineage family of each axis entry) — positioned from the
# matrix's post-colorbar bbox so the strips stay column/row-aligned
pos = ax.get_position()
ax_top.set_position([pos.x0, pos.y1 + 0.047, pos.width, 0.017])
ax_top.set_xlim(-.5, ND - .5); ax_top.set_ylim(0, 1)
ax_top.axis("off")
for j, d in enumerate(POOL):
    ax_top.add_patch(Rectangle((j - .5, 0), 1, 1, facecolor=FCOL[FAMILY[d]],
                               edgecolor="white", lw=0.6))
ax_left.set_position([pos.x0 - 0.064, pos.y0, 0.012, pos.height])
ax_left.set_ylim(NH - .5, -.5); ax_left.set_xlim(0, 1)
ax_left.axis("off")
for i, h in enumerate(HOSTS):
    ax_left.add_patch(Rectangle((0, i - .5), 1, 1, facecolor=FCOL[FAMILY[h]],
                                edgecolor="white", lw=0.6))

# right block: instrument scatter (top) + kinship/notes (bottom) -----------
gright = GridSpecFromSubplotSpec(
    2, 1, subplot_spec=outer[0, 1], height_ratios=[1.45, 1.0], hspace=0.28)
axs = fig.add_subplot(gright[0, 0])

cos_all = np.array([p["cos_graftinput"] for p in PAIRS])
d_all = np.array([p["D"] for p in PAIRS])
fresh = np.array([p["donor"] == "e005s_small" for p in PAIRS])
axs.scatter(cos_all[~fresh], d_all[~fresh], s=24, color=C_KIN, alpha=0.65,
            zorder=3, label="donor = e040 lineage (seed-42 kin, 11/12)")
axs.scatter(cos_all[fresh], d_all[fresh], s=34, color=C_FRESH, alpha=0.9,
            zorder=4, marker="D", label="donor = e005s_small (fresh seed)")
axs.axvline(THR, color=C_RULE, lw=1.8, ls="--", zorder=2,
            label=f"graft-if threshold {THR:.4f} (in-sample)")
axs.axhline(MED_D, color=C_GRAY, lw=1.0, ls=":", zorder=1,
            label=f"label boundary: D = median = {MED_D:.3f}")
axs.set_xlabel("P2: pre-graft stream-cosine at graft-input depths (d2,d3)",
               fontsize=9)
axs.set_ylabel("D = graft damage (mean dCE, sites L2/L3, 15 batches)",
               fontsize=9)
axs.set_title(f"cos vs D — raw r = {CORR['r_D']:+.3f}; "
              f"partial r(D|A) = {CORR['partial_r_given_A']:+.3f} "
              f"[{CI[0]:+.3f}, {CI[1]:+.3f}]", fontsize=10)
axs.grid(alpha=0.25)
axs.legend(fontsize=7, loc="lower left", framealpha=0.9)

# ROC inset (fpr/tpr arrays verbatim from e062) — upper-right corner (empty
# region: no pairs there)
axi = axs.inset_axes([0.595, 0.665, 0.38, 0.31])
axi.plot(ROC["fpr"], ROC["tpr"], color=C_G, lw=1.6,
         label=f"P2 (AUC {ROC['auc']:.3f})")
axi.plot([0, 1], [0, 1], color="#888888", lw=0.8, ls=":")
axi.scatter([fpr_thr], [tpr_thr], s=28, color=C_RULE, zorder=5,
            marker="*", label=f"rule @ {THR:.4f}")
axi.set_xlabel("FPR: damaging grafts waved through", fontsize=6.5)
axi.set_ylabel("TPR: compatible grafts waved through", fontsize=6.5)
axi.set_title(f"ROC (label = D > median)  J = {ROC['youden_J']:.3f}",
              fontsize=7)
axi.tick_params(labelsize=6)
axi.legend(fontsize=6, loc="lower right")
axi.grid(alpha=0.25)

# notes / kinship legend
axn = fig.add_subplot(gright[1, 0])
axn.axis("off")
order = ["ancestor (e040_ref)", "w lineage", "m lineage", "g lineage",
         "fresh seed (e005)", "e050 directed-mutation (hosts only)"]
handles = [Patch(facecolor=FCOL[f], label=f) for f in order]
handles += [Patch(facecolor="none", edgecolor=C_RULE, lw=1.9,
                  label=f"PASS: cos >= {THR:.4f} (n={n_pass}/204 cells)"),
            Patch(facecolor="#e8e8e8", edgecolor="#bbbbbb", hatch="///",
                  label="self-transplant (excluded, n=12)")]
axn.legend(handles=handles, fontsize=7.5, loc="upper left", frameon=True,
           title="lineage family (kinship structure)", title_fontsize=8.5)
notes = (
    f"honesty: n = 204 pairs / 18 hosts x 12 donors (self-grafts excluded);\n"
    f"threshold {THR:.4f} = Youden cut fitted IN-SAMPLE on these same pairs.\n"
    f"partial r(D|A) = {CORR['partial_r_given_A']:+.3f} "
    f"[{CI[0]:+.3f}, {CI[1]:+.3f}] host-cluster bootstrap; A itself\n"
    f"screens nothing (r {M['correlations_scaleA']['A']['r_D']:+.3f}).\n"
    f"CAVEAT (kinship): 11/12 donors share the e040 seed-42 lineage;\n"
    f"donor redundancy is NOT absorbed by the host-cluster bootstrap\n"
    f"(CI anti-conservative). \"Similar nets graft well\" is near-\n"
    f"tautological as mechanism — the defensible claim is the INSTRUMENT.\n"
    f"at threshold: mean D = {mean_D_pass:.2f} (pass) vs {mean_D_fail:.2f} "
    f"(fail); TPR {tpr_thr:.3f} / FPR {fpr_thr:.3f}.\n"
    f"scale-B transfer hint: partial r "
    f"{M['scale_B_secondary']['partial_r_given_A_P2_gi']:+.3f} "
    f"(n=6, no bars claimed)."
)
axn.text(0.01, 0.16, notes, fontsize=7.0, va="bottom", ha="left",
         family="monospace", color="#222222")

fig.suptitle("V014 — THE CROSSMATCH GRID: pre-graft stream-cosine screens "
             "graft damage before a single organ is transplanted "
             "(E062 / T046)\n"
             f"rule: graft only if cos_graftinput >= {THR:.4f}  ->  "
             f"AUC {ROC['auc']:.3f}, Youden J {ROC['youden_J']:.3f}; "
             f"partial r(D|A) = {CORR['partial_r_given_A']:+.3f} "
             f"[{CI[0]:+.3f}, {CI[1]:+.3f}] — high-cosine region ~ "
             f"low-damage region", fontsize=12, y=0.975, va="top")

fig.savefig(OUT / "crossmatch_grid.png", dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- metrics
fam_counts = defaultdict(int)
for n in POOL:
    fam_counts[FAMILY[n]] += 1
out = {
    "viz": "v014_crossmatch_grid",
    "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "device": "cpu-only (CUDA_VISIBLE_DEVICES masked; zero-GPU analysis)",
    "source": {"run": "e062", "thinking": "T046", "n_pairs": len(PAIRS),
               "n_hosts": NH, "n_donors": ND},
    "decision_rule": {
        "predictor": "P2: mean token-cosine of host/donor residual streams "
                     "at graft-input depths (d2,d3), 2-batch probe",
        "threshold": THR, "in_sample": True,
        "auc": ROC["auc"], "youden_J": ROC["youden_J"],
        "n_pass_cells": n_pass, "n_fail_cells": n_fail,
    },
    "correlation": {
        "raw_r_D": CORR["r_D"],
        "partial_r_given_A": CORR["partial_r_given_A"],
        "ci95_hostcluster": CI,
    },
    "label_definition": M["label_definition"],
    "grid": {"cells_measured": int((~np.isnan(D)).sum()),
             "cells_self_excluded": int(np.isnan(D).sum()),
             "aggregation": "mean over measurements per (host,donor)",
             "max_measurements_per_cell": int(CNT.max()),
             "count_annotations_shown": bool((CNT > 1).any())},
    "at_threshold": {
        "mean_D_pass": mean_D_pass, "mean_D_fail": mean_D_fail,
        "confusion": {"pass_compatible": tp, "fail_compatible": fn,
                      "pass_damaging": fp, "fail_damaging": tn},
        "tpr": tpr_thr, "fpr": fpr_thr},
    "kinship": {"family_of": FAMILY, "donor_family_counts": dict(fam_counts),
                "caveat": "11/12 donors are e040 seed-42 kin; host-cluster "
                          "bootstrap does not absorb donor redundancy — CI "
                          "anti-conservative (REVIEWS CLAIM D)"},
    "outputs": {"figure": "runs/v014/crossmatch_grid.png",
                "metrics": "runs/v014/metrics.json"},
    "timing_s": round(time.time() - T0, 1),
}
(OUT / "metrics.json").write_text(json.dumps(out, indent=2) + "\n",
                                  encoding="utf-8")
print(f"v014 done: {OUT / 'crossmatch_grid.png'}  "
      f"(pass {n_pass}/{len(PAIRS)}, tp/fp {tp}/{fp}, "
      f"{out['timing_s']}s)")
