"""V015 — THE READ KERNEL: P-3 READ-POLICY's standing visualization (CPU-only).

The visual form of THINKING.md T050 (E084): the first direct read-policy
measurement — kernel = shadow. Every number is lifted verbatim from the
registered run (runs/e084/metrics.json; the dCE shadow curve is the stored
e053c profile copied inside it — nothing recomputed, GPU untouched):

  (a) MAIN   flip-rate(age) with bootstrap CIs OVERLAID on the stored
             dCE-load(age) curve (twin axes: log flip-rate left, linear
             nats right) — the kernel-equals-shadow visual. Annotated:
             r = 0.918 [0.867, 0.940] overall, young-band (1-10) r = 0.961,
             and the honest tail flag — Spearman = -0.13, shaded misreport
             zone where the shadow reads ~0 while the rule still opens.
             Inset: the 64 correlation points themselves (line + flat cloud).
  (b) SPARSE-OPEN  per-decision opened-coordinate histogram (median 3-4 of
             80) — the rule opens a handful, not a distribution.
  (c) TAXONOMY    flip destinations per intervention type: runner-up ~51% /
             top-5 ~22% / outside-top-5 ~26% — long-tail vulnerability.
  (d) CONTENT-FOLLOWING  V-swap flips land on the donor's actual next token
             8.8% vs 2% chance — real but partial content read.
  NOTES     honesty footnotes: n cells, verdict, tail flag, scale.

Outputs: runs/v015/read_kernel.png, runs/v015/metrics.json.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-ONLY by lab rule (GPU busy)

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "runs" / "e084"
OUT = REPO / "runs" / "v015"
OUT.mkdir(parents=True, exist_ok=True)

T0 = time.time()
M = json.loads((SRC / "metrics.json").read_text(encoding="utf-8"))

KC = M["kernel_curve"]
CORR = M["correlations"]
R_PRIM = CORR["primary_vzero_vs_dce"]["pearson"]           # 0.9176
R_CI = CORR["primary_vzero_vs_dce"]["ci"]                  # [0.867, 0.940]
RHO = CORR["primary_vzero_vs_dce"]["spearman"]             # -0.133
R_YOUNG = CORR["vzero_ages1_10_only"]["pearson"]           # 0.9615
BANDS = M["band_rates"]
TAX = M["taxonomy"]
OPENED = M["opened_coordinates"]
DONOR = TAX["vswap_donor_continuation"]
KILL = M["battery"]["kill_bar"]                            # 0.001

X_EX = np.array(KC["x_ages_exact"], dtype=float)           # ages 1..60
X_BIN = np.array(KC["x_bin_mids"], dtype=float)            # 4 old bins
DCE = np.array(KC["dce_stored"], dtype=float)              # 64 pts
VZ = np.array(KC["rates"]["vzero"], dtype=float)           # 64 pts
POOL = np.array(KC["rates"]["pooled"], dtype=float)
CI_LO = np.array(KC["ci_vzero"]["lo"], dtype=float)        # 60 (ages 1-60)
CI_HI = np.array(KC["ci_vzero"]["hi"], dtype=float)
CI_FLOOR = 5e-4           # log-axis floor; bounds of exactly 0 clipped (note)

# lab palette (e084 figure family: tab:red/green/purple; v014 navy conventions)
C_KERN, C_POOL, C_SHADOW = "#d62728", "#5d8fbb", "#08306b"
C_KD, C_VS = "#2ca02c", "#9467bd"
C_TAIL, C_YOUNG, C_GRAY = "#f9ddaf", "#deebf7", "#888888"
C_OUT5 = "#c0392b"

n_cells = M["protocol"]["cells"]                            # 24,000
n_flips = M["battery"]["n_flips"]                           # 1,541

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(16.5, 10.5))
outer = GridSpec(2, 2, figure=fig, width_ratios=[1.6, 1.0],
                 height_ratios=[1.32, 1.0], wspace=0.21, hspace=0.26,
                 left=0.052, right=0.985, top=0.895, bottom=0.065)

# ================================================================ (a) MAIN
ax = fig.add_subplot(outer[0, 0])
ax.set_xscale("log")
ax.set_yscale("log")

# honest misreport zone: past the young band the shadow reads ~0 while the
# rule still opens (Spearman flag) — shaded before curves so it sits behind
ax.axvspan(10, 560, color=C_TAIL, alpha=0.45, zorder=0)
ax.axvspan(0.9, 10, color=C_YOUNG, alpha=0.55, zorder=0)
ax.axvline(10, color="#c8a165", ls="--", lw=1.0, zorder=1)

# the rule: V-zero argmax flip rate (primary registered readout) + boot CI
ax.fill_between(X_EX, np.clip(CI_LO, CI_FLOOR, None), CI_HI,
                color=C_KERN, alpha=0.16, lw=0, zorder=2,
                label="V-zero flip rate, 95% bootstrap CI")
ax.plot(X_EX, VZ[:60], "o-", ms=3.2, lw=1.7, color=C_KERN, zorder=5,
        label="V-zero flip rate — THE KERNEL (the rule)")
ax.plot(X_BIN, VZ[60:], "s", ms=8, mfc="none", mew=1.7, color=C_KERN,
        zorder=5, label="old ages, pooled bins (at bin mids)")
ax.plot(X_EX, POOL[:60], ls=(0, (4, 2)), lw=1.0, color=C_POOL,
        zorder=4, label="pooled (3 types)")
ax.plot(X_BIN, POOL[60:], "s", ms=6, mfc="none", mew=1.2, color=C_POOL,
        zorder=4)
ax.axhline(KILL, color="k", ls=":", lw=1.1, zorder=3)

# the shadow: stored e053c dCE-load on a linear twin axis
axs = ax.twinx()
axs.plot(X_EX, DCE[:60], "-o", ms=2.6, lw=1.7, color=C_SHADOW, zorder=4,
         label="stored dCE-load(age) — THE SHADOW (e053c, nats)")
axs.plot(X_BIN, DCE[60:], "s", ms=6, mfc="none", mew=1.5, color=C_SHADOW,
         zorder=4)
axs.axhline(0.0, color=C_SHADOW, ls=":", lw=0.7, alpha=0.6, zorder=1)
ax.set_zorder(axs.get_zorder() + 1)   # keep the kernel on top
ax.patch.set_visible(False)

ax.set_xlim(0.9, 560)
ax.set_ylim(4e-4, 1.6)
axs.set_ylim(-0.25, 5.2)
ax.set_xticks([1, 2, 3, 5, 10, 20, 60, 100, 200, 511])
ax.set_yticks([1e-3, 1e-2, 1e-1, 1])
ax.set_xlabel("cache entry age (tokens; log scale)", fontsize=9)
ax.set_ylabel("argmax flip rate (log scale) — the rule", fontsize=9,
              color=C_KERN)
axs.set_ylabel("stored V-zero dCE-load (nats) — the CE shadow", fontsize=9,
               color=C_SHADOW)
ax.tick_params(axis="y", colors=C_KERN)
axs.tick_params(axis="y", colors=C_SHADOW)
ax.grid(alpha=0.22, which="both")

# annotations: verdict (top), young band + peak (left), tail flag (mid),
# kill bar (bottom) — positioned to clear the curves, legend and inset
ax.text(1.02, 1.13, f"KERNEL = SHADOW:  r = {R_PRIM:.3f} "
        f"[{R_CI[0]:.3f}, {R_CI[1]:.3f}]  (Pearson, n = 64 age points)",
        fontsize=9.5, color="black", fontweight="bold", va="top")
ax.text(9.4, 0.70, f"young band 1-10:\nr = {R_YOUNG:.3f}\n"
        f"peak @ age 2:\n82% flip | 4.47 nats", fontsize=7.5,
        color=C_SHADOW, ha="right", va="top", fontweight="bold")
ax.text(380, 1.45e-3, f"kill bar {KILL:.1%}", fontsize=7, color="k",
        ha="right", va="bottom")
ax.text(11.5, 0.36, "TAIL — shadow misreport zone (age > 10):\n"
        "shadow dCE ~ 0 (±0.07 nats) while the rule\n"
        f"still opens {BANDS['mid_11_60']['pooled']:.1%} (mid) / "
        f"{BANDS['old_61_511']['pooled']:.1%} (old) of cells\n"
        f"Spearman(rank) = {RHO:.2f} — rank agreement fails",
        fontsize=6.8, color="#7a5b1e", va="top")

# inset: the 64 correlation points themselves — tight line (young) + flat
# cloud at dCE~0 (tail): the r = 0.918 structure in one glance
axi = ax.inset_axes([0.685, 0.62, 0.30, 0.335])
axi.scatter(DCE[10:60], VZ[10:60], s=13, color=C_GRAY, alpha=0.75, zorder=3,
            label="mid (11-60)")
axi.scatter(DCE[60:], VZ[60:], s=30, facecolor="none", edgecolor=C_SHADOW,
            lw=1.1, zorder=4, label="old bins")
axi.scatter(DCE[:10], VZ[:10], s=22, color=C_KERN, zorder=5,
            label="young (1-10)")
axi.set_xlabel("stored dCE (nats)", fontsize=6.5)
axi.set_ylabel("V-zero flip rate", fontsize=6.5)
axi.set_title(f"the 64 points: r = {R_PRIM:.3f}", fontsize=7)
axi.tick_params(labelsize=6)
axi.legend(fontsize=5.6, loc="upper left", framealpha=0.92,
           handletextpad=0.3, borderpad=0.4)
axi.grid(alpha=0.25)

# main-panel legend (composed: kernel + shadow live on twin axes)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = axs.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=6.9, loc="upper left",
          bbox_to_anchor=(0.365, 0.99), framealpha=0.94)
ax.set_title("(a) THE READ KERNEL vs ITS SHADOW — flip-rate(age) over stored "
             "dCE-load(age)\n"
             f"the argmax rule opens what the CE curves measure  "
             f"(bands: young {BANDS['young_1_10']['pooled']:.0%} / mid "
             f"{BANDS['mid_11_60']['pooled']:.1%} / old "
             f"{BANDS['old_61_511']['pooled']:.1%} pooled)", fontsize=10,
             loc="left")

# ====================================================== (b) SPARSE-OPEN
axb = fig.add_subplot(outer[1, 0])
KMAX = 16            # show 0..15; residual tail annotated (max observed noted)
ks = np.arange(KMAX)
hists = {t: np.array(OPENED["per_type_hist"][t]) for t in
         ("vzero", "kdrop", "vswap")}
cols = {"vzero": C_KERN, "kdrop": C_KD, "vswap": C_VS}
meds = OPENED["per_type_median"]
tail_txt = []
for t, h in hists.items():
    tail_n = int(h[KMAX:].sum())
    tail_txt.append(f"{t}: {tail_n} dps >15, max {int(np.nonzero(h)[0].max())}")
    axb.bar(ks + {"vzero": -0.28, "kdrop": 0.0, "vswap": 0.28}[t], h[:KMAX],
            width=0.26, color=cols[t], alpha=0.85,
            label=f"{t} (median {meds[t]:g})")
    axb.axvline(meds[t] + {"vzero": -0.28, "kdrop": 0.0, "vswap": 0.28}[t],
                color=cols[t], ls="--", lw=1.0, alpha=0.8)
le5 = {t: int(h[:6].sum()) for t, h in hists.items()}
axb.text(0.985, 0.96, "of 80 cache entries per decision point\n"
         f"any-type union median: {OPENED['any_type_median']:g} opened\n"
         f"{le5['vzero']}/100 V-zero decisions open <= 5 coordinates\n"
         f"not shown: {', '.join(tail_txt)}",
         transform=axb.transAxes, fontsize=7.0, ha="right", va="top",
         bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.9,
                   pad=3.5))
axb.set_xticks(ks)
axb.set_xlabel("opened cache coordinates per decision point "
               "(entries whose single intervention flipped the argmax)",
               fontsize=9)
axb.set_ylabel("decision points (of 100)", fontsize=9)
axb.legend(fontsize=7.5, title="intervention type", title_fontsize=8)
axb.grid(alpha=0.25, axis="y")
axb.set_title("(b) SPARSE OPEN — the rule opens a handful of coordinates per "
              "decision, not a distribution (median 3-4 of 80)", fontsize=10,
              loc="left")

# ====================================================== (c) TAXONOMY
gright = GridSpecFromSubplotSpec(
    3, 1, subplot_spec=outer[:, 1], height_ratios=[1.22, 0.78, 1.06],
    hspace=0.52)
axc = fig.add_subplot(gright[0, 0])
types = ["vzero", "kdrop", "vswap"]
tlabel = {"vzero": "V-zero", "kdrop": "K-drop", "vswap": "V-swap"}
segs = [("runner_up", "runner-up (rank 2)", "#08306b"),
        ("top5", "ranks 3-5 (top-5)", "#95a5a6"),
        ("outside_top5", "ESCAPES the top-5", C_OUT5)]
bottom = np.zeros(3)
for key, lab, col in segs:
    vals = np.array([TAX["per_type"][t][key] * 100 for t in types])
    axc.bar(np.arange(3), vals, 0.55, bottom=bottom, color=col, alpha=0.92,
            label=lab, edgecolor="white", lw=0.6)
    for i, v in enumerate(vals):
        axc.text(i, bottom[i] + v / 2, f"{v:.1f}%", ha="center", va="center",
                 fontsize=8, color="white", fontweight="bold")
    bottom += vals
axc.set_xticks(np.arange(3))
axc.set_xticklabels([f"{tlabel[t]}\nn = {TAX['per_type'][t]['n_flips']} flips"
                     for t in types], fontsize=8.5)
axc.set_ylim(0, 100)
axc.set_ylabel("% of argmax flips", fontsize=9)
axc.legend(fontsize=7.2, loc="upper right", framealpha=0.94)
axc.grid(alpha=0.25, axis="y")
axc.set_title("(c) FLIP TAXONOMY — the runner-up wins only ~half;\n"
              "~1 in 4 flips escapes the top-5 entirely", fontsize=9.5,
              loc="left")

# ============================================== (d) CONTENT-FOLLOWING
axd = fig.add_subplot(gright[1, 0])
hit = DONOR["hit_rate"] * 100
chance = DONOR["chance_all_vswap_cells"] * 100
axd.barh([1, 0], [hit, chance], height=0.52,
         color=[C_VS, "#b0b0b0"], alpha=0.92)
axd.text(hit + 0.25, 1, f"{hit:.1f}%  ({DONOR['hits']}/"
         f"{DONOR['n_vswap_flips']} V-swap flips)", va="center", fontsize=8.5,
         color=C_VS, fontweight="bold")
axd.text(chance + 0.25, 0, f"{chance:.1f}%  (uniform-chance baseline)",
         va="center", fontsize=8.5, color="#707070")
axd.annotate(f"{hit / chance:.1f}x chance", xy=(hit, 0.72),
             xytext=(hit * 0.52, 0.63), fontsize=9.5, color="black",
             fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="black", lw=1.1))
axd.set_yticks([1, 0])
axd.set_yticklabels(["new argmax = donor's\nactual next token", "chance"],
                    fontsize=8)
axd.set_xlim(0, 14.0)
axd.set_xlabel("% of V-swap flips", fontsize=9)
axd.grid(alpha=0.25, axis="x")
axd.set_title("(d) CONTENT-FOLLOWING — a real but partial content read",
              fontsize=9.5, loc="left")

# ====================================================== notes / footnotes
axn = fig.add_subplot(gright[2, 0])
axn.axis("off")
MS = M["margin_structure"]
notes = (
    f"honesty: n = {n_cells:,} intervention cells (metrics.json verbatim:\n"
    f"  100 margin-stratified decision points x 80 entries x 3 types;\n"
    f"  {n_flips:,} argmax flips battery-wide. T050/NOTES say 48,000 —\n"
    f"  the run file records {n_cells:,}; this figure follows the run file).\n"
    f"VERDICT: battery flip rate {M['battery']['flip_rate_battery_wide']:.1%}"
    f" >= {KILL:.1%} kill bar; primary r = {R_PRIM:.3f}\n"
    f"  [{R_CI[0]:.3f}, {R_CI[1]:.3f}] >= 0.8 registered bar -> "
    f"KERNEL = SHADOW — the\n  dissociation branch did NOT fire.\n"
    f"TAIL FLAG (shaded): Spearman {RHO:.2f} — the shadow nails the young\n"
    f"  spike ({R_YOUNG:.3f} on ages 1-10) but does NOT rank-order the\n"
    f"  rule's residual tail opening; below the registered dissociation\n"
    f"  bar, but shadow claims about plateau fine structure carry it.\n"
    f"fine print: old ages pooled in 4 bins (plotted at bin mids, no stored\n"
    f"  CI); CI bounds of exactly 0 clipped to 5e-4 (log axis); flips sit\n"
    f"  at low margin (median {MS['flip_cell_margin_median']:.2f} vs "
    f"{MS['all_cell_margin_median']:.2f} all-cell logits); content-\n"
    f"  following is T050's weakest-novelty arm (interchange-intervention\n"
    f"  lineage, alignment map fixed).\n"
    f"scale: 0.84M/4L char-LM (4L/4H/128d, ctx 512, vocab 65;\n"
    f"  {M['net']['params']:,} params; e053c ckpt @ "
    f"{M['net']['steps_in_ckpt']:,} steps, val CE\n"
    f"  {M['net']['val_ce']:.3f}) — CPU-only; every number lifted verbatim\n"
    f"  from runs/e084/metrics.json (T050); zero new model evaluations."
)
axn.text(0.005, 0.995, notes, fontsize=6.8, va="top", ha="left",
         family="monospace", color="#222222")

fig.suptitle("V015 — THE READ KERNEL: the rule opens what the shadow "
             "measures (E084 / T050)\n"
             f"V-zero argmax-flip rate vs stored e053c dCE-load: r = "
             f"{R_PRIM:.3f} [{R_CI[0]:.3f}, {R_CI[1]:.3f}] -> KERNEL = "
             f"SHADOW;  young band r = {R_YOUNG:.3f};  tail rank agreement "
             f"fails (Spearman {RHO:.2f}, shaded)", fontsize=12, y=0.982,
             va="top")

fig.savefig(OUT / "read_kernel.png", dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- metrics
out = {
    "viz": "v015_read_kernel",
    "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "device": "cpu-only (CUDA_VISIBLE_DEVICES masked; zero-GPU analysis)",
    "source": {"run": "e084", "thinking": "T050",
               "metrics": "runs/e084/metrics.json",
               "n_cells_run_file": n_cells,
               "note": "T050/NOTES say 48,000 cells; run file records "
                       f"{n_cells:,} (100 dps x 80 x 3) — figure follows "
                       "the run file"},
    "verdict": {"r_primary": R_PRIM, "ci95": R_CI,
                "bar": 0.8, "fires": "KERNEL_EQUALS_SHADOW",
                "battery_flip_rate": M["battery"]["flip_rate_battery_wide"],
                "n_flips": n_flips},
    "main_panel": {"spearman_tail_flag": RHO,
                   "young_1_10_r": R_YOUNG,
                   "bands_pooled": {k: BANDS[k]["pooled"] for k in BANDS},
                   "ci_floor_log_axis": CI_FLOOR,
                   "old_bins_at_mids": X_BIN.tolist()},
    "panels": {
        "sparse_open": {"per_type_median": meds,
                        "any_type_median": OPENED["any_type_median"],
                        "le5_fraction": {t: le5[t] / 100 for t in le5}},
        "taxonomy": TAX["per_type"],
        "content_following": {"hit_rate": DONOR["hit_rate"],
                              "chance": DONOR["chance_all_vswap_cells"],
                              "hits": DONOR["hits"],
                              "n_vswap_flips": DONOR["n_vswap_flips"]},
    },
    "outputs": {"figure": "runs/v015/read_kernel.png",
                "metrics": "runs/v015/metrics.json"},
    "timing_s": round(time.time() - T0, 1),
}
(OUT / "metrics.json").write_text(json.dumps(out, indent=2) + "\n",
                                  encoding="utf-8")
print(f"v015 done: {OUT / 'read_kernel.png'}  "
      f"(r={R_PRIM:.3f}, rho={RHO:.2f}, young r={R_YOUNG:.3f}, "
      f"{out['timing_s']}s)")
