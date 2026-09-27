"""V016 — THE UNLEARNING MECHANISM TABLE (CPU-only).

The visual form of THINKING.md T052 (E065 + E091 close-out):
"four removals, four causal signatures."  Every number is lifted
verbatim from registered run metrics (runs/e065/metrics.json +
runs/e091/metrics.json — nothing recomputed, GPU untouched):

  MAIN   5 arms (no-removal / RMU / surgery / ascent / retain-only)
         x 4 readouts (R1 transplant-rescue, R2 linear probe,
         R3 relearn steps, R4 collateral dCE_R); each cell prints
         its value with a per-row hue encoding on that row's own
         scale (slim gradient strips carry the scales + bars).
  SIDE   the three-readout dissociation: probe vs transplant-rescue
         vs elicitation, parallel-coordinates strip — three
         instruments, three different verdicts on the same arms.
  NOTE   the E091 gate verdict as a callout on the RMU column:
         reverse-transplant fails ~45x under the 0.30 bar; the
         RMU net's own d5 state still carries 0.362 into an
         intact net (the store survives, the gate is sealed).

Outputs: runs/v016/unlearning_signatures.png, runs/v016/metrics.json.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-ONLY by lab rule (GPU busy)

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
M65 = json.loads((RUNS / "e065" / "metrics.json").read_text(encoding="utf-8"))
M91 = json.loads((RUNS / "e091" / "metrics.json").read_text(encoding="utf-8"))
OUT = RUNS / "v016"
OUT.mkdir(parents=True, exist_ok=True)

T0 = time.time()

# ---------------------------------------------------------------- data
# Everything below is pulled from the metrics dicts — no literals that
# could drift from the registered runs.
ARMS = ["no_removal", "rmu", "surgery", "ascent", "retain_only"]
ARM_LABEL = {
    "no_removal": "no-removal\n(intact control)",
    "rmu": "RMU a2x d4+5\nSEALS THE\nREADOUT GATE",
    "surgery": "surgery row-reset\nROW GONE,\nTRACE ALIVE",
    "ascent": "projected ascent\nDESTROYS\nTHE NET",
    "retain_only": "retain-only FT\nHIDES — GATE OPEN\n(obfuscator)",
}
ARM_SHORT = {"no_removal": "NONE", "rmu": "RMU", "surgery": "SURG",
             "ascent": "ASCENT", "retain_only": "RETAIN"}
ARM_COLOR = {"no_removal": "#34495e", "rmu": "#8e44ad", "surgery": "#c0392b",
             "ascent": "#e67e22", "retain_only": "#2980b9"}

V = M65["verdicts"]
R1 = {a: V[a]["R1"]["max_mean_tf_d5"] for a in ARMS}          # rescue
R1_DEPTHS = {a: V[a]["R1"]["rescuable_depths"] for a in ARMS}
R2 = {a: V[a]["R2"]["probe_d45"] for a in ARMS}                # probe d4-5
R2_D6 = {a: M65["r2"]["per_arm"][a]["probe_acc_by_depth"]["6"] for a in ARMS}
R3 = {a: (V[a]["R3"] or {}).get("steps_to_bar") for a in ARMS}  # relearn
R3_COS = {a: (V[a]["R3"] or {}).get("cos_wte_final") for a in ARMS}
R3_COSCLASS = {a: (V[a]["R3"] or {}).get("cos_class") for a in ARMS}
R4 = {a: V[a]["R4"]["dce_r"] for a in ARMS}                    # collateral
R4_BAND = {a: V[a]["R4"]["band"] for a in ARMS}
R4_ENTDRIFT = {a: V[a]["R4"]["entropy_drift"] for a in ARMS}
BATT = {a: M65["arms_battery"][a]["battery"]["p_z_mean"] for a in ARMS}
CE_R0 = M65["r4"]["no_removal"]["ce_r"]                        # 1.6668 ref

BAR = 0.30                                    # rescuable bar (e055/e065 verbatim)
RMU_XCELL = M91["sweeps"]["net0__rmu"]["max_mean_tf_d5"]       # reverse sweep
UNDERR = BAR / RMU_XCELL                                       # 44.48x exact
UNDERR_TXT = math.ceil(UNDERR)                                 # "~45x" (T052 wording)
YCELL_D5 = M91["secondary_battery_swap"]["cells"]["rmu__to__no_removal"][5]
ONSET_CHARS = M91["sites"]["rmu_own_harvest_chars"]             # 22400
ONSET_BANK = len(M91["sites"]["rmu_own_bank"])                  # 0

CFG = M65["config"]                                            # 6L/6H/192d/256
PARAMS = CFG["params"]                                         # 2,739,072
SCALE_STR = (f"{PARAMS / 1e6:.2f}M/{CFG['n_layer']}L char-LM "
             f"({CFG['n_layer']}L/{CFG['n_head']}H/{CFG['n_embd']}d, "
             f"ctx {CFG['block_size']})")


def f_r1(v: float) -> str:
    return f"{v:.3f}" if v >= 0.01 else f"{v:.1e}"


def f_batt(v: float) -> str:
    return f"{v:.2e}"


# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(17.0, 11.0))
outer = GridSpec(2, 2, figure=fig, width_ratios=[1.72, 1.0],
                 hspace=0.20, wspace=0.16,
                 left=0.032, right=0.985, top=0.865, bottom=0.028,
                 height_ratios=[1.0, 0.44])

gleft = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[:, 0],
                                height_ratios=[1.0, 0.12], hspace=0.06)
axm = fig.add_subplot(gleft[0, 0])

ROWS = ["R1  TRANSPLANT-RESCUE", "R2  LINEAR PROBE",
        "R3  RELEARN", "R4  COLLATERAL"]
ROW_SUB = ["max site-mean TF, d<=5 band (21 sites)",
           "acc d4-5 (chance 0.50); @d6 in cell",
           "steps to bar NLL<=1.0 & acc>=0.80 (cap 300)",
           f"dCE_R nats vs no-removal ({CE_R0:.3f})"]

# per-row hue encodings (each row on its OWN scale) -------------------------
r1_norm = LogNorm(min(R1.values()), 0.55)
r1_cmap = plt.get_cmap("Greens")
r2_norm = Normalize(0.48, 0.58)
r2_cmap = plt.get_cmap("Purples")
r3_norm = Normalize(0, 80)
r3_cmap = plt.get_cmap("Blues")
r4_norm = TwoSlopeNorm(vcenter=0.0, vmin=-0.25, vmax=0.25)
r4_cmap = plt.get_cmap("RdYlGn")

NORMS = [r1_norm, r2_norm, r3_norm, r4_norm]
CMAPS = [r1_cmap, r2_cmap, r3_cmap, r4_cmap]


def cellval(row: int, a: str) -> float | None:
    return [R1, R2, R3, R4][row][a]


def fmt(row: int, a: str) -> str:
    v = cellval(row, a)
    if v is None:
        return "—"
    if row == 0:
        return f_r1(v)
    if row == 1:
        return f"{v:.3f}"
    if row == 2:
        return f"{int(v)}"
    return f"{v:+.3f}"


def tag(row: int, a: str) -> str:
    if row == 0:
        d = R1_DEPTHS[a]
        return ("rescuable @d" + ",".join(map(str, d))) if d else "fails everywhere"
    if row == 1:
        if a == "surgery":
            return f"SURVIVES —\ndeep probe {R2_D6[a]:.3f} @d6"
        return "above chance" if R2[a] > 0.51 else "chance — probe dead"
    if row == 2:
        if a == "no_removal":
            return "(never removed)"
        if a == "ascent":
            return "SLOW — R4-disqualified"
        c = R3_COSCLASS[a] or ""
        short = "groove" if "groove" in c else ("invalid*" if "invalid" in c else "")
        return f"fast; cos_wte {R3_COS[a]:.3f}\n({short})" if short else "fast"
    # row 3
    if a == "no_removal":
        return "reference net"
    if a == "ascent":
        return f"BAND FAILS\n(entropy drift {R4_ENTDRIFT[a]:+.2f})"
    if a == "rmu":
        return f"~0 band; entropy drift\n{R4_ENTDRIFT[a]:+.3f} flagged"
    return "~0 band"


axm.set_xlim(-1.58, 5.88)
axm.set_ylim(4.62, -2.55)                    # inverted: row 0 on top
axm.set_xticks([])
axm.set_yticks([])
for s in axm.spines.values():
    s.set_visible(False)

# column headers ------------------------------------------------------------
for j, a in enumerate(ARMS):
    hot = a == "rmu"
    axm.add_patch(Rectangle((j - 0.5, -1.98), 1, 1.05, facecolor="#f4f4f4",
                            edgecolor=ARM_COLOR[a], lw=2.2 if hot else 1.1,
                            zorder=1))
    axm.text(j, -1.52, ARM_LABEL[a], ha="center", va="center", fontsize=8.0,
             color=ARM_COLOR[a], fontweight="bold", zorder=3, linespacing=1.30)

# row labels ----------------------------------------------------------------
for i in range(4):
    axm.text(-0.60, i - 0.10, ROWS[i], ha="right", va="center", fontsize=9.2,
             fontweight="bold", color="#1a1a1a")
    axm.text(-0.60, i + 0.24, ROW_SUB[i], ha="right", va="center", fontsize=6.6,
             color="#555555", style="italic")

# cells ---------------------------------------------------------------------
for i in range(4):
    for j, a in enumerate(ARMS):
        v = cellval(i, a)
        if v is None:                        # no-removal R3: not applicable
            axm.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#ececec",
                                    edgecolor="#bbbbbb", lw=0.6, hatch="///",
                                    zorder=2))
            axm.text(j, i - 0.08, "—", ha="center", va="center", fontsize=15,
                     color="#777777", zorder=4)
            axm.text(j, i + 0.27, tag(i, a), ha="center", va="center",
                     fontsize=5.9, color="#666666", zorder=4)
            continue
        if i == 3 and v > r4_norm.vmax:      # ascent collateral: clip + flag
            frac = 1.0
        else:
            frac = NORMS[i](v)
        fc = CMAPS[i](frac)
        hatch = "///" if (i == 3 and v > 1.0) else None
        axm.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=fc,
                                edgecolor="#999999", lw=0.7, hatch=hatch,
                                zorder=2))
        big_color = "white" if (frac > 0.62 or (i == 3 and v > 1.0)) else "#111111"
        if i == 3 and v > 1.0:
            big_color = "white"
        axm.text(j, i - 0.10, fmt(i, a), ha="center", va="center", fontsize=15,
                 fontweight="bold", color=big_color, zorder=4)
        tcol = "#f2f2f2" if (frac > 0.62 or (i == 3 and v > 1.0)) else "#333333"
        axm.text(j, i + 0.27, tag(i, a), ha="center", va="center", fontsize=5.9,
                 color=tcol, zorder=4)

# slim per-row scale strips (right edge) ------------------------------------
STRIP_X0, STRIP_W = 4.72, 0.80
grad = np.linspace(0, 1, 256).reshape(1, -1)
STRIPS = [
    dict(norm=r1_norm, cmap=r1_cmap, lo="1e-9", hi="0.55",
         marks=[(BAR, f"bar {BAR:.2f}")],
         mpos=lambda val: (np.log10(val) - np.log10(min(R1.values())))
         / (np.log10(0.55) - np.log10(min(R1.values())))),
    dict(norm=r2_norm, cmap=r2_cmap, lo="0.48", hi="0.58",
         marks=[(0.50, "chance .50")],
         mpos=lambda val: (val - 0.48) / 0.10),
    dict(norm=r3_norm, cmap=r3_cmap, lo="0", hi="80",
         marks=[(50, "fast<=50")], mpos=lambda val: val / 80),
    dict(norm=r4_norm, cmap=r4_cmap, lo="-.25", hi="+.25",
         marks=[(0.0, "0")], mpos=lambda val: (val + 0.25) / 0.50),
]
for i, st in enumerate(STRIPS):
    yc = i
    axm.imshow(grad, extent=(STRIP_X0, STRIP_X0 + STRIP_W, yc + 0.075, yc - 0.075),
               aspect="auto", cmap=st["cmap"],
               norm=Normalize(0, 1), zorder=3, interpolation="bilinear")
    axm.text(STRIP_X0 - 0.04, yc, st["lo"], ha="right", va="center", fontsize=5.6,
             color="#555555")
    axm.text(STRIP_X0 + STRIP_W + 0.04, yc, st["hi"], ha="left", va="center",
             fontsize=5.6, color="#555555")
    for val, lab in st["marks"]:
        xm = STRIP_X0 + STRIP_W * st["mpos"](val)
        axm.plot([xm, xm], [yc - 0.115, yc + 0.115], color="#111111", lw=1.1,
                 zorder=4)
        axm.text(xm, yc + 0.20, lab, ha="center", va="bottom", fontsize=5.0,
                 color="#111111", zorder=4)
axm.text(STRIP_X0 + STRIP_W / 2, -0.90, "row scales\n(hue per row)",
         ha="center", va="top", fontsize=6.0, color="#555555",
         style="italic", linespacing=1.3)

# E091 gate-verdict callout on the RMU column -------------------------------
callout = (
    "E091 GATE VERDICT — H-READOUT-GATE FIRES\n"
    f"reverse-transplant (good no-removal states INTO the RMU net) fails "
    f"~{UNDERR_TXT}x under the {BAR:.2f} bar at every depth\n"
    f"(max site-mean TF d<=5 = {RMU_XCELL:.4f}, shuffled flat) — yet the RMU "
    f"net's own d5 state still carries a\n"
    f"half-strength trace: {YCELL_D5:.3f} when transplanted into an intact net "
    f"(e091 Y-cell). The loss did NOT\n"
    "evacuate the store — it sealed the gate. And the RMU net's free-run "
    f"onset geometry died too ({ONSET_BANK}/{ONSET_CHARS:,} chars)."
)
axm.annotate(callout, xy=(1.0, 3.54), xytext=(1.05, 4.58),
             ha="center", va="bottom", fontsize=6.6, linespacing=1.42,
             color="#4a235a",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="#f7eefc",
                       edgecolor=ARM_COLOR["rmu"], lw=1.8),
             arrowprops=dict(arrowstyle="-|>", color=ARM_COLOR["rmu"], lw=2.0,
                             shrinkA=2, shrinkB=1),
             zorder=6)
axm.set_title("MAIN — the mechanism matrix: every removal read four ways "
              "(numbers verbatim from e065)",
              fontsize=10.5, pad=8)

# ---------------------------------------------------------------- side panel
axp = fig.add_subplot(outer[0, 1])

probe_vals = [R2_D6[a] for a in ARMS]
resc_vals = [R1[a] for a in ARMS]
batt_vals = [BATT[a] for a in ARMS]


def norm01(vals, log=False):
    v = np.log10(vals) if log else np.array(vals, dtype=float)
    return (v - v.min()) / (v.max() - v.min())


AXES = [
    dict(name="PROBE\nacc @d6", vals=probe_vals, y=norm01(probe_vals),
         fmt=lambda v: f"{v:.3f}", marks=[(0.50, "chance")],
         mval=lambda t: (t - min(probe_vals)) / (max(probe_vals) - min(probe_vals))),
    dict(name="TRANSPLANT-RESCUE\nmax TF d<=5", vals=resc_vals,
         y=norm01(resc_vals, log=True), fmt=f_r1,
         marks=[(BAR, "bar .30")],
         mval=lambda t: (np.log10(t) - np.log10(min(resc_vals)))
         / (np.log10(max(resc_vals)) - np.log10(min(resc_vals)))),
    dict(name="ELICITATION p(Z)\nbattery (log)", vals=batt_vals,
         y=norm01(batt_vals, log=True), fmt=f_batt, marks=[], mval=None),
]
XS = [0, 1, 2]

for k, ax_ in enumerate(AXES):
    axp.plot([XS[k], XS[k]], [0.0, 1.0], color="#999999", lw=1.2, zorder=1)
    for val, lab in ax_["marks"]:
        ym = ax_["mval"](val)
        axp.plot([XS[k] - 0.05, XS[k] + 0.05], [ym, ym], color="#c0392b",
                 lw=1.3, zorder=2)
        axp.text(XS[k] + 0.07, ym, lab, fontsize=6.0, color="#c0392b",
                 va="center", ha="left")
    axp.text(XS[k], 1.07, ax_["name"], ha="center", va="bottom", fontsize=7.6,
             fontweight="bold", color="#1a1a1a", linespacing=1.3)

for idx, a in enumerate(ARMS):
    ys = [AXES[k]["y"][idx] for k in range(3)]
    axp.plot(XS, ys, color=ARM_COLOR[a], lw=2.4 if a in ("surgery", "retain_only",
                                                         "rmu") else 1.8,
             alpha=0.95, zorder=3,
             marker="o", ms=5.5, mec="white", mew=0.8,
             label=ARM_SHORT[a])

# orderings + values beneath the panel (three instruments, three rankings)
def order_line(vals, fmt):
    order = sorted(range(len(ARMS)), key=lambda i: -vals[i])
    return " > ".join(f"{ARM_SHORT[ARMS[i]]} {fmt(vals[i])}" for i in order)


ORDER_LINES = [
    "probe @d6 : " + order_line(probe_vals, lambda v: f"{v:.3f}"),
    "rescue    : " + order_line(resc_vals, f_r1),
    "p(Z) log  : " + order_line(batt_vals, f_batt),
]
for r, line in enumerate(ORDER_LINES):
    axp.text(1.0, -0.125 - 0.115 * r, line, ha="center", va="top",
             fontsize=5.5, family="monospace", color="#333333", zorder=4)

axp.set_xlim(-0.52, 2.60)
axp.set_ylim(-0.62, 1.34)
axp.set_xticks([])
axp.set_yticks([])
for s in axp.spines.values():
    s.set_visible(False)
axp.legend(fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.42),
           frameon=False, ncol=5, columnspacing=1.3, handletextpad=0.4,
           handlelength=1.6)
axp.set_title("SIDE — the three-readout dissociation: probe crowns surgery; "
              "rescue crowns retain-only;\nelicitation crowns only the intact "
              "net — RMU: probe 0.492 yet rescue 0.007 on the SAME net",
              fontsize=9.0, pad=10)

# ---------------------------------------------------------------- footnotes
axn = fig.add_subplot(outer[1, 1])
axn.axis("off")
notes = (
    "honesty footnotes\n"
    f"[0] scale: {SCALE_STR}; params {PARAMS:,} verbatim from the e065/e091\n"
    "    configs — the brief's '0.84M' is the e005s small cfg, a different\n"
    "    net. Toy-scale causal signatures, not production-scale claims.\n"
    "[1] n = 1 net per condition (e048_repro install; the R3 protocol itself\n"
    "    flags n=1) — single-seed evidence. e091 replicated the RMU and\n"
    "    retain cells bit-close (gates G_RMU_REPLICA / G_R1_REPRO pass).\n"
    "[2] R2 ran PROBE-ONLY: free-run ZEPHYRA rate 0.0/kchar in ALL five\n"
    "    arms (2,800 chars each), so the behavioral axis uses the prompted\n"
    "    elicitation battery p(Z), log scale (e091 free-run onset harvest:\n"
    "    RMU 0/22,400 chars — expression geometry dead).\n"
    "[3] scales: R1 = max site-mean transplant TF over d<=5 (rescuable:\n"
    "    bar 0.30 AND shuffled <= 0.05 AND site-bootstrap CI excludes 0);\n"
    "    R2 = linear probe acc, chance 0.50; R3 = steps to relearn bar\n"
    "    NLL<=1.0 & acc>=0.80, cap 300, fast<=50; R4 = dCE_R vs the\n"
    "    no-removal net. Every number lifted verbatim from\n"
    "    runs/e065/metrics.json + runs/e091/metrics.json (CPU-only build).\n"
    "[4] rescue and elicitation share rank order but NOT verdicts:\n"
    "    retain-only is rescuable (0.345) yet behaviorally dead (0.0022,\n"
    "    ~256x below control) — the Orgad-instrument caveat. *surgery R3\n"
    "    cos 'invalid': <25% norm regrowth under the e044 guard."
)
axn.text(0.010, 0.99, notes, fontsize=6.0, va="top", ha="left",
         family="monospace", color="#222222", linespacing=1.34,
         bbox=dict(boxstyle="round,pad=0.45", facecolor="#fbfbfb",
                   edgecolor="#bbbbbb", lw=0.8))

# ---------------------------------------------------------------- title
fig.suptitle(
    "V016 — THE UNLEARNING MECHANISM TABLE: four removals, four causal "
    "signatures (E065 + E091 / T052)\n"
    "RMU seals the readout gate (trace survives, reception sealed, "
    "expression dead); surgery removes the row but not the probe trace; "
    "ascent destroys; retain-only hides with the gate open",
    fontsize=12.5, y=0.982, va="top")

fig.savefig(OUT / "unlearning_signatures.png", dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- metrics
out = {
    "viz": "v016_unlearning_signatures",
    "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "device": "cpu-only (CUDA_VISIBLE_DEVICES masked; zero-GPU analysis)",
    "source": {"e065": "runs/e065/metrics.json",
               "e091": "runs/e091/metrics.json",
               "thinking": "T052"},
    "matrix": {a: {"R1_rescue_max_tf_d5": R1[a],
                   "R1_rescuable_depths": R1_DEPTHS[a],
                   "R2_probe_d45": R2[a], "R2_probe_d6": R2_D6[a],
                   "R3_steps_to_bar": R3[a], "R3_cos_wte_final": R3_COS[a],
                   "R4_dce_r": R4[a], "R4_band": R4_BAND[a],
                   "battery_pz": BATT[a]} for a in ARMS},
    "e091_callout": {
        "hypothesis_fired": M91["verdict"]["fired"],
        "reverse_bar": BAR,
        "reverse_max_tf_d5": RMU_XCELL,
        "under_bar_ratio": UNDERR,
        "under_bar_rounded": UNDERR_TXT,
        "ycell_d5_rmu_into_intact": YCELL_D5,
        "onset_geometry": f"{ONSET_BANK}/{ONSET_CHARS} chars",
    },
    "scale": {
        "params": PARAMS,
        "desc": SCALE_STR,
        "note": ("brief said '0.84M' but both e065 and e091 configs read "
                 "2,739,072 (6L/6H/192d) — the config number is what the "
                 "figure prints; the 0.84M family (840,704) is the e005s "
                 "small cfg, a different net"),
    },
    "footnotes": [f"scale: {SCALE_STR}, params {PARAMS:,} verbatim "
                  "(brief said 0.84M; configs read 2,739,072)",
                  "n=1 per condition (R3 protocol-flagged; e091 replica "
                  "gates pass)",
                  "R2 probe-only: free-run Z-rate 0.0/kchar all arms; "
                  "behavioral axis = battery p(Z) log",
                  "row scales: R1 log 1e-9..0.55 bar 0.30; R2 0.48..0.58 "
                  "chance 0.50; R3 0..80 fast<=50; R4 +/-.25 vcenter 0 "
                  "(ascent +11.40 clipped, hatched)",
                  "rescue vs elicitation: same rank order, different "
                  "verdicts (retain-only 0.345 rescuable vs 0.0022 dead)"],
    "outputs": {"figure": "runs/v016/unlearning_signatures.png",
                "metrics": "runs/v016/metrics.json"},
    "timing_s": round(time.time() - T0, 1),
}
(OUT / "metrics.json").write_text(json.dumps(out, indent=2) + "\n",
                                  encoding="utf-8")
print(f"v016 done: {OUT / 'unlearning_signatures.png'}  "
      f"(under-bar {UNDERR:.1f}x, ycell {YCELL_D5:.3f}, "
      f"{out['timing_s']}s)")
