"""V012 — SELF-PORTRAIT v2: the anatomy after 48 hours (CPU-ONLY, no torch).

Updates the v010 poster with everything the day-two runs added. One figure,
six panels; every number is lifted verbatim from registered run metrics
(nothing recomputed, GPU untouched, no torch import):

  A · C2  one function, three anatomies      e001 / e014b / e035 lesion maps
          + STAMPS: scale ladder (e005s), write-equalizer (e033),
            replication sweep (e047)
  B · C1  the causal gate — RE-ANCHORED      e012d / e018 causal census
          (T012/T014) + depth-budget slides  (replaces v010's lens
          depth-histogram "invariance", which was an instrument artifact)
  C · C4  the locality funnel                e013a / e021 + e005s erosion
  D · C3  the anchoring ladder               e041 (unchanged; e040 registered)
  E · C4' the retrieval threshold (NEW)      e049 refrain-density curve
  F · C7  the edit law: four faculties (NEW) e042 / e043 / e044 / e046 /
          e048 / e005s  (T015 / T018 / T019)

Outputs: runs/v012/portrait_v2.png, runs/v012/panel_data.json.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-ONLY by lab rule (GPU busy)

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as pe_mod
from matplotlib.patches import FancyBboxPatch, Patch

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
OUT = RUNS / "v012"
OUT.mkdir(parents=True, exist_ok=True)

# palette (lab convention: B blue, R orange, task green, COPY red emphasis)
C_B, C_R, C_T, C_COPY = "#4c72b0", "#dd8452", "#55a868", "#c44e52"
C_FAR, C_LOC, C_ID = "#8172b3", "#55a868", "#c44e52"
C_B43, C_R43 = "#9db8d6", "#eec49a"  # lighter seed-43 variants
PE = [pe_mod.withStroke(linewidth=2.2, foreground="white")]
FLAG_BX = dict(boxstyle="round,pad=0.32", fc="#fdf3f3", ec=C_COPY, lw=1.0)


def load(name: str) -> dict:
    return json.loads((RUNS / name / "metrics.json").read_text(encoding="utf-8"))


e001, e013a, e014b, e021 = load("e001"), load("e013a"), load("e014b"), load("e021")
e035, e041 = load("e035"), load("e041")
e005s, e033, e049 = load("e005s"), load("e033"), load("e049")
e012d, e018 = load("e012d"), load("e018")
e042, e043, e044 = load("e042"), load("e043"), load("e044")
e046, e047, e048 = load("e046"), load("e047"), load("e048")

# ---------------------------------------------------------------- panel A data
# C2 — lesion maps of the three anatomies (zero-ablation damage, val CE nats)
A_NETS = [
    ("B  (Shakespeare, e001)", C_B, e001["attn_block_damage"],
     e001["mlp_block_damage"], e001["baseline_val_loss"]),
    ("R  (stream-renorm, e014b)", C_R, e014b["renorm_damage"]["attn"],
     e014b["renorm_damage"]["mlp"], e014b["renorm_val"]),
    ("Task  (ID→COPY, e035)", C_T, e035["q1_lesion"]["attn_block_damage"],
     e035["q1_lesion"]["mlp_block_damage"],
     e035["q1_lesion"]["baseline_val_ce_task_corpus"]),
]
# scale stamps (e005s): attention front-loading multiplier attn-L0/last
FL_SMALL = e005s["nets"]["small"]["l2_front_loading"]["attn_l0_over_last"]
FL_LARGE = e005s["nets"]["large"]["l2_front_loading"]["attn_l0_over_last"]
FL_B = (e005s["B_reference_loaded"]["l2"]["attn_block_damage"][0]
        / e005s["B_reference_loaded"]["l2"]["attn_block_damage"][-1])
MLP0_10M = e005s["nets"]["large"]["l2_front_loading"]["mlp_block_damage"][0]
MLP_REST_10M = max(e005s["nets"]["large"]["l2_front_loading"]["mlp_block_damage"][1:])
# e033 write-equalizer (0.84M, single net)
EQ_BASE, EQ_EQ = e033["baseline_val"], e033["equalized_val"]
EQ_MLP_B, EQ_MLP_E = e033["baseline_mlp"], e033["equalized_mlp"]
EQ_KL = e033["kl_l5_l4_equalized"]

# ---------------------------------------------------------------- panel B data
# C1 RE-ANCHORED — causal census (e012d; instrument verdict e018)
B_NETS = [
    ("B", C_B, e012d["nets"]["B"]["causal_hist_d0_d5"], e012d["nets"]["B"]["causal_mode"]),
    ("B43", C_B43, e012d["nets"]["B43"]["causal_hist_d0_d5"], e012d["nets"]["B43"]["causal_mode"]),
    ("R", C_R, e012d["nets"]["R"]["causal_hist_d0_d5"], e012d["nets"]["R"]["causal_mode"]),
    ("R43", C_R43, e012d["nets"]["R43"]["causal_hist_d0_d5"], e012d["nets"]["R43"]["causal_mode"]),
]
B_HIST_CORR = e012d["hist_correlation_4x4_pearson_7bin"]
B_XSEED = e012d["cross_seed_same_regime"]           # 0.735 / 0.799
B_XREGIME = e012d["cross_regime_same_seed"]         # 0.753 / 0.500
B_CEIL = min(v for k, v in B_HIST_CORR.items() if k.split("|")[0] == k.split("|")[1])
B_CEIL_MAX = max(v for k, v in B_HIST_CORR.items() if k.split("|")[0] == k.split("|")[1])
B_SPEARMAN = e018["spearman_causal_lens_flippers"]
B_FLIPCURVE = e018["flip_curve_per_depth"]
B_SUFFMON = {
    "4L/0.84M": e005s["nets"]["small"]["l1_gate"]["suffix_monotone_flip_fraction"],
    "6L/2.7M": e018["suffix_monotone_flip_fraction"],
    "8L/10M": e005s["nets"]["large"]["l1_gate"]["suffix_monotone_flip_fraction"],
}
# depth-budget slide (e005s): relative commitment depth = mode/(n_layer-1)
B_SCALE = [
    ("4L · 0.84M\n(4000 steps)",
     e005s["nets"]["small"]["l1_gate"]["causal_mode"] / 3,
     e005s["nets"]["small"]["l1_gate"]["no_single_depth_flip_fraction"],
     e005s["nets"]["small"]["l1_gate"]["modal_share"], False),
    ("6L · 2.7M (B)\n(2226 steps)",
     e005s["B_reference_loaded"]["l1"]["causal_mode"] / 5,
     e005s["B_reference_loaded"]["l1"]["no_single_depth_flip_fraction"], None, True),
    ("8L · 10M\n(1086 steps)",
     e005s["nets"]["large"]["l1_gate"]["causal_mode"] / 7,
     e005s["nets"]["large"]["l1_gate"]["no_single_depth_flip_fraction"], None, True),
]

# ---------------------------------------------------------------- panel C data
C_FAR_M = e013a["layer_far_mass"]
C_LOC_M = e013a["layer_local_mass"]
C_ID_TASK = e021["attention_census"]["task_layer_id_mass"]
C_ID_CTRL = e021["attention_census"]["control_layer_id_mass"]
C_HEAD = e021["attention_census"]["best_head"]
# scale erosion of the 16-token margin (e005s l4 far-value)
FV = {
    "0.84M": e005s["nets"]["small"]["l4_far_value"]["far_value_mean"],
    "2.7M": e005s["B_reference_loaded"]["l4"]["far_value_mean"],
    "10M": e005s["nets"]["large"]["l4_far_value"]["far_value_mean"],
}
FV_P99 = (e005s["nets"]["small"]["l4_far_value"]["p99"],
          e005s["nets"]["large"]["l4_far_value"]["p99"])

# ---------------------------------------------------------------- panel D data
LAD = e041["ladder"]
D_LADDER = [
    ("B vs B\nsame init · same order", LAD["same_init_same_order"]["value"], "#08306b"),
    ("B vs BDO\nsame init · diff order", LAD["same_init_diff_order"]["value"], "#4c72b0"),
    (f"B vs R, B43 vs R43\nsame init · diff regime\n(pooled; B↔R alone "
     f"{LAD['same_init_diff_regime']['value']:.3f})",
     LAD["same_init_diff_regime"]["pooled_same_init"], "#dd8452"),
    ("B vs B43, …\ndiff init · (FLOOR)", LAD["diff_init"]["value"], "#b0b0b0"),
]
D_ORGAN = e041["dw_ceiling"]["per_organ"]

# ---------------------------------------------------------------- panel E data
# T021 — the retrieval threshold (e049): far-value at refrain prefix (k=1..3)
E_LEVELS = [0, 5, 20, 60]
E_FV = [e049["readouts"][f"s27_p{l}"]["shared" if l == 0 else "own"]["fv_prefix_k123"]
        for l in E_LEVELS]  # p0 has no own events -> shared probe (flagged)
E_ACC = [e049["readouts"][f"s27_p{l}"]["shared" if l == 0 else "own"]["acc_prefix_k123"]
         for l in E_LEVELS]
E_EVTS = [e049["corpora"][str(l)]["events"] for l in E_LEVELS]
E_EVTS_VAL = [e049["corpora"][str(l)]["events_val"] for l in E_LEVELS]
E_VAL = [e049["training"][f"s27_p{l}"]["val_loss"] for l in E_LEVELS]
E_HEADS = [e049["readouts"][f"s27_p{l}"]["attention"]["best_head"] for l in E_LEVELS]
E_HEAD_RATIO = [h["refrain_mass"] / h["control_mass"] for h in E_HEADS]
E_NONREFRAIN = [e049["readouts"][f"s27_p{l}"]["non_refrain"]["far_value_mean"]
                for l in E_LEVELS]
E_RATIOS = e049["verdicts"]["P2_adjacent_ratios"]
E_P0_FV = e049["verdicts"]["fv_prefix_baseline_p0_shared"]
E_10M = e049["readouts_10m"]["own"]

# ---------------------------------------------------------------- panel F data
# C7 — the edit law's four faculties (T015/T018/T019)
F_DMIX = e043["armD"]["Dmix@s400"]
F_GRAFT_BEST = max(e043["verdicts"]["P1"]["a_bdo_rows_only"]["per_cell"][c]["gap_closed"]
                   for c in e043["verdicts"]["P1"]["a_bdo_rows_only"]["cells"])
F_DIFFINIT = e043["verdicts"]["P1"]["b_b43_copy_both"]
F_ATLAS_RHO = e043["verdicts"]["P3"]["a_machinery_conserved"]["JULIET"]["Dmix@s400"]["spearman_vs_base"]
F_E046 = e046["verdicts"]["a_replication"]["per_net"]
F_SNAME_ROWS = e005s["B_reference_loaded"]["l6"]["s_name"]
F_SNAME_PATCH = e042["twofactor_cells"]["D2+1h"]["s_name"]
F_SNAME_SMALL = e005s["nets"]["small"]["l6_row_surgery"]["s_name"]
F_SNAME_LARGE = e005s["nets"]["large"]["l6_row_surgery"]["s_name"]
F_HB = e048["verdicts"]["H_battery_overfit"]
F_OA = e048["verdicts"]["onset_audit"]
F_P1E = e048["verdicts"]["P1_teacher_forcing_bound"]
F_P3D = e048["verdicts"]["P3_dose_quantity"]
F_V2 = e044["verdicts"]["P2"]
F_RE = e044["re_erasability"]["a_final"]
F_RC = e044["route_check"]
F_SCAR = e044["scar_rows"]["a_final"]

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(17.2, 16.8))
gs = fig.add_gridspec(3, 2, hspace=0.40, wspace=0.185,
                      left=0.048, right=0.977, top=0.912, bottom=0.068)
LAYERS = np.arange(6)


def panel_title(ax, claim, headline, sub, foot=None):
    ax.set_title(f"{claim} — {headline}", loc="left", fontsize=13.0,
                 fontweight="bold", pad=30)
    ax.text(1.0, 1.004, sub, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.0, style="italic", color="#444444")
    if foot:
        ax.annotate(foot, xy=(1.0, 1.004), xycoords="axes fraction",
                    xytext=(0, 11.5), textcoords="offset points",
                    ha="right", va="bottom", fontsize=7.1, style="italic",
                    color="#8b4444")


# ================================================================ panel A (C2)
gsA = gs[0, 0].subgridspec(1, 2, wspace=0.28)
axA1 = fig.add_subplot(gsA[0, 0])
axA2 = fig.add_subplot(gsA[0, 1], sharey=axA1)
w = 0.24
for ax, kind, dmg_idx in ((axA1, "attention blocks", 2), (axA2, "MLP blocks", 3)):
    for i, (name, col, attn, mlp, _v) in enumerate(A_NETS):
        dmg = (attn if dmg_idx == 2 else mlp)
        xs = LAYERS + (i - 1) * w
        ax.bar(xs, dmg, width=w, color=col, alpha=0.92, label=name)
        for x, v in zip(xs, dmg):
            if v >= 0.9:
                ax.text(x, v + 0.07, f"{v:.1f}", ha="center", va="bottom",
                        fontsize=6.7, color=col, path_effects=PE)
    ax.set_xticks(LAYERS, [f"L{l}" for l in LAYERS], fontsize=9.5)
    ax.set_xlabel(kind, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
axA1.set_ylabel("zero-ablation damage  (Δval CE, nats)", fontsize=9.5)
axA1.set_ylim(0, 7.3)
panel_title(axA1, "A · C2", "One function, three anatomies — now scale-stamped",
            "organs move; early attention stays critical",
            foot="cross-corpus nats; task net is a different function (e035)")
axA1.annotate("MLP-0 keystone:\nB +4.08, task +4.90", xy=(0.24, 4.97), xytext=(0.5, 7.08),
              ha="left", va="top", fontsize=8.6, fontweight="bold", color="#333333",
              arrowprops=dict(arrowstyle="->", lw=1.3, color="#333333"))
axA1.annotate("dissolves under renorm (+0.10)", xy=(0.0, 0.10), xytext=(0.45, 3.05),
              fontsize=8.2, color=C_R,
              arrowprops=dict(arrowstyle="->", lw=1.1, color=C_R))
axA1.annotate("L5 attention ≈ dead weight\nin every anatomy (≤ +0.03)", xy=(5.0, 0.06),
              xytext=(2.3, 1.0), fontsize=8.2, color="#333333",
              arrowprops=dict(arrowstyle="->", lw=1.1, color="#333333"))
axA2.annotate("MLP profile INVERTS\nto late-heavy in R", xy=(5.24, 0.72), xytext=(2.35, 3.3),
              ha="left", va="center", fontsize=8.2, color=C_R,
              arrowprops=dict(arrowstyle="->", lw=1.1, color=C_R))
axA2.text(0.985, 0.995,
          f"DAY-2 STAMPS\n"
          f"• front-loading INTENSIFIES with scale (e005s): attn-L0/last\n"
          f"  {FL_SMALL:.0f}× (0.84M) → {FL_B:.0f}× (2.7M) → {FL_LARGE:.0f}× (10M); at 10M the\n"
          f"  MLP-0 keystone sharpens ({MLP0_10M:.2f}; rest ≤{MLP_REST_10M:.2f})\n"
          f"• write-equalizer (e033, 1 net): all MLP write norms pinned\n"
          f"  equal → val {EQ_EQ:.4f} BEATS baseline {EQ_BASE:.4f}; damage\n"
          f"  flattens, calibrator untouched (KL {EQ_KL:.2f}) — the schedule\n"
          f"  is decorative (energy carrier still replicates 5/5, e047)",
          transform=axA2.transAxes, ha="right", va="top", fontsize=7.0,
          linespacing=1.35,
          bbox=dict(boxstyle="round,pad=0.45", fc="#fbfbfb", ec="#bbbbbb"))
handles = [Patch(fc=col, alpha=0.92, label=f"{name} · val {v:.3f}")
           for name, col, _a, _m, v in A_NETS]
axA1.legend(handles=handles, loc="upper right", fontsize=7.2, framealpha=0.95,
            title="anatomy (6L/6H/192 char-GPT, 2.7M params)", title_fontsize=7.2)

# ================================================================ panel B (C1)
gsB = gs[0, 1].subgridspec(1, 2, wspace=0.24, width_ratios=[1.35, 1.0])
axB1 = fig.add_subplot(gsB[0, 0])
xb = np.arange(6)
B_HIST_FRAC = {}
for name, col, hist, mode in B_NETS:
    f = np.array(hist) / sum(hist)
    B_HIST_FRAC[name] = f.tolist()
    axB1.plot(xb, f * 100, "-o" if name in ("B", "R") else "--s", color=col,
              lw=2.2 if name in ("B", "R") else 1.6, ms=5.5, alpha=0.95,
              label=f"{name}  (mode d{mode})")
    axB1.plot(mode, f[mode] * 100, "o", ms=11, mfc="none", mec=col, mew=2.2, zorder=5)
axB1.annotate("the mode SLIDES with seed + regime:\nB 3 → B43 4 → R 4 → R43 5\n"
              "(renorm pushes mass deeper: d5 share 250→423)",
              xy=(5, B_HIST_FRAC["R43"][5] * 100 - 1.2), xytext=(1.0, 6.5),
              ha="left", va="bottom", fontsize=8.0, color="#333333",
              arrowprops=dict(arrowstyle="->", lw=1.2, color="#333333"))
axB1.text(0.03, 0.70,
          f"NON-INVARIANCE (e012d): hist corr\n"
          f"seed axis {B_XSEED['B|B43']:.2f}/{B_XSEED['R|R43']:.2f} ·\n"
          f"regime axis {B_XREGIME['B|R']:.2f}/{B_XREGIME['B43|R43']:.2f} —\n"
          f"all below the 0.8 bar, vs instrument\n"
          f"ceiling {B_CEIL:.2f}–{B_CEIL_MAX:.2f}\n"
          f"flip curves suffix-monotone\n"
          f"{min(B_SUFFMON.values())*100:.0f}–{max(B_SUFFMON.values())*100:.0f}% "
          f"(B {B_FLIPCURVE[0]:.2f}→{B_FLIPCURVE[-1]:.2f})",
          transform=axB1.transAxes, fontsize=7.2, va="top",
          bbox=dict(boxstyle="round,pad=0.4", fc="#fdf3f3", ec=C_COPY, lw=0.9))
axB1.set_xticks(xb, [f"d{d}" for d in xb], fontsize=9.0)
axB1.set_xlabel("counterfactual patch depth (causal decision point)", fontsize=9.0)
axB1.set_ylabel("% of flippers decided at depth", fontsize=9.5)
axB1.set_ylim(0, 38)
axB1.legend(loc="upper right", fontsize=7.4, framealpha=0.95)
axB1.spines[["top", "right"]].set_visible(False)

axB2 = fig.add_subplot(gsB[0, 1])
rel = [r for _l, r, _n, _m, _g in B_SCALE]
nosf = [n for _l, _r, n, _m, _g in B_SCALE]
bars = axB2.bar(np.arange(3), rel, color=["#777777", C_B, "#08306b"], width=0.55,
                alpha=0.9)
for i, (lbl, r, n, m, strict_ok) in enumerate(B_SCALE):
    axB2.text(i, r + 0.035, f"{r:.2f}", ha="center", fontsize=10.5,
              fontweight="bold", path_effects=PE)
    tag = "" if strict_ok else "\n✗ strict gate"
    axB2.text(i, -0.085, f"no-single-flip\n{n*100:.0f}%{tag}", ha="center",
              va="top", fontsize=7.4, color="#8b4444" if not strict_ok else "#555555")
axB2.axhline(0, color="k", lw=0.9)
axB2.set_xticks(np.arange(3), [l for l, _r, _n, _m, _g in B_SCALE], fontsize=8.2)
axB2.set_ylabel("relative commitment depth  mode/(L−1)", fontsize=9.0)
axB2.set_ylim(-0.33, 1.25)
axB2.set_xlim(-0.55, 2.55)
axB2.spines[["top", "right"]].set_visible(False)
axB2.text(0.985, 0.99,
          "depth budget slides the gate (e005s)\n"
          "4L has no mid: 46% at the FINAL block\n"
          "distributed collapse 26% → 15% → 7%\n"
          "R8: steps confound (4000/2226/1086)",
          transform=axB2.transAxes, ha="right", va="top", fontsize=7.0,
          bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#bbbbbb"))
panel_title(axB1, "B · C1", "The causal gate: universal, depth NOT invariant",
            "re-anchored T012/T014: lens histograms were an argmax-stability artifact",
            foot=f"ρ(causal, lens) = {B_SPEARMAN:+.3f} (e018) — lens demoted; task-COPY L4 mode "
                 f"survives (L4-H1 causal lesion, e021)")

# ================================================================ panel C (C4)
axC = fig.add_subplot(gs[1, 0])
lc = np.arange(6)
axC.fill_between(lc, C_FAR_M, color=C_FAR, alpha=0.28, zorder=1)
axC.plot(lc, C_FAR_M, "-o", color=C_FAR, lw=2.2, ms=6,
         label="far-mass (d>16) · Shakespeare")
axC.fill_between(lc, C_LOC_M, color=C_LOC, alpha=0.28, zorder=1)
axC.plot(lc, C_LOC_M, "-s", color=C_LOC, lw=2.2, ms=6,
         label="local-mass (d≤3) · Shakespeare")
axC.plot(lc, C_ID_TASK, "--D", color=C_ID, lw=2.6, ms=7.5,
         label="ID-mass · task-net at COPY")
axC.plot(lc, C_ID_CTRL, ":", color=C_ID, lw=1.4, alpha=0.8,
         label="ID-mass · control")
axC.annotate("the funnel: near-uniform at L0, tightest\nat L2, re-broadens at L5 (far is idle\n"
             "grazing — KL −6.9% under truncation)",
             xy=(0.06, 0.82), xytext=(0.95, 0.99), ha="left", va="top",
             fontsize=8.2, color="#333333",
             arrowprops=dict(arrowstyle="->", lw=1.2, color="#333333"))
axC.annotate(f"retrieval head L4-H1:\n{C_HEAD['id_mass']*100:.1f}% of its mass on the\n"
             f"5-char ID nonce (control {C_HEAD['control_same_head']*100:.0f}%)",
             xy=(4, C_ID_TASK[4] + 0.03), xytext=(5.15, 0.80), ha="right", va="top",
             fontsize=8.4, fontweight="bold", color=C_ID,
             arrowprops=dict(arrowstyle="->", lw=1.4, color=C_ID,
                             connectionstyle="arc3,rad=-0.18"))
axC.text(0.30, 0.035,
         f"SCALE EROSION of the 16-token margin (e005s far-value mean):\n"
         f"{FV['2.7M']:+.3f} (2.7M) → {FV['0.84M']:+.3f} (0.84M) → "
         f"{FV['10M']:+.3f} (10M) · p99 {FV_P99[0]:.2f}→{FV_P99[1]:.2f}  "
         f"[R8: steps confound]\n"
         f"the no-retrieval law was one point on a curve — its shape is now panel E →",
         transform=axC.transAxes, fontsize=7.6, ha="left", va="bottom",
         bbox=dict(boxstyle="round,pad=0.42", fc="#f7f4fb", ec=C_FAR, lw=1.0))
axC.set_xticks(lc, [f"L{l}" for l in lc], fontsize=9.5)
axC.set_ylabel("share of attention mass", fontsize=9.5)
axC.set_ylim(0, 1.0)
axC.set_xlim(-0.25, 5.25)
axC.legend(loc="upper left", bbox_to_anchor=(0.02, 0.80), fontsize=7.2,
           framealpha=0.95)
axC.spines[["top", "right"]].set_visible(False)
panel_title(axC, "C · C4", "The locality funnel",
            "no far retrieval on natural text; the task grows one at L4")

# ================================================================ panel D (C3)
axD = fig.add_subplot(gs[1, 1])
vals = [v for _l, v, _c in D_LADDER]
cols = [c for _l, _v, c in D_LADDER]
axD.bar(np.arange(4), vals, color=cols, alpha=0.93, width=0.60)
for i, v in enumerate(vals):
    axD.text(i, v + 0.03, f"{v:.3f}" + ("  = CEILING" if i == 1 else ""),
             ha="center", va="bottom", fontsize=11.5, fontweight="bold",
             path_effects=PE)
axD.axhline(0.0, color="k", lw=0.9)
axD.set_xticks(np.arange(4), [l for l, _v, _c in D_LADDER], fontsize=8.4)
axD.set_ylabel("cos(ΔW_A, ΔW_B)   organ motion", fontsize=9.5)
axD.set_ylim(-0.07, 1.16)
axD.set_xlim(-0.55, 3.55)
axD.spines[["top", "right"]].set_visible(False)
panel_title(axD, "D · C3", "The anchoring ladder",
            "the seed-anchored object is the residual-stream basis",
            foot="unchanged since e041 (no newer causal C3 data); the e040 lineage "
                 "test — is anchoring evolvable? — is registered (T022)")
axD.text(0.985, 0.86,
         "PARTIAL ANCHORING\n"
         "batch order erases ~half; regime all but 0.15;\n"
         "different inits are orthogonal (0.000)\n\n"
         "per-organ ceiling (same init, diff order):\n"
         f"  L0-attn {D_ORGAN['L0|attn']['cos_dW_B_vs_BDO']:.2f}   "
         f"L0-mlp {D_ORGAN['L0|mlp']['cos_dW_B_vs_BDO']:.2f}   (early organs order-robust)\n"
         f"  L5-attn {D_ORGAN['L5|attn']['cos_dW_B_vs_BDO']:.2f}   "
         f"L5-mlp {D_ORGAN['L5|mlp']['cos_dW_B_vs_BDO']:.2f}   (depth erodes)",
         transform=axD.transAxes, ha="right", va="top", fontsize=8.0,
         bbox=dict(boxstyle="round,pad=0.45", fc="#f5f5f5", ec="#bbbbbb"))

# ================================================================ panel E (C4')
gsE = gs[2, 0].subgridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.50)
axE = fig.add_subplot(gsE[0, 0])
xE = np.arange(4)
axE.axvspan(-0.45, 1.45, color="#e8c34e", alpha=0.14, zorder=0)
bar_cols = [C_COPY if v < 0 else C_FAR for v in E_FV]
bars = axE.bar(xE, E_FV, width=0.5, color=bar_cols, alpha=0.9, zorder=2)
bars[0].set_hatch("//")
bars[0].set_alpha(0.55)
for x, v in zip(xE, E_FV):
    axE.text(x, v + (0.09 if v >= 0 else -0.09), f"{v:+.2f}", ha="center",
             va="bottom" if v >= 0 else "top", fontsize=10.5, fontweight="bold",
             path_effects=PE)
axE.axhline(0, color="k", lw=1.1, zorder=1)
axE.set_xticks(xE, [f"p{l}\n({ev} events)" for l, ev in zip(E_LEVELS, E_EVTS)],
               fontsize=8.5)
axE.set_ylabel("far-value at refrain prefix  (nats, k=1–3)", fontsize=9.5)
axE.set_ylim(-3.0, 1.6)
axE.set_xlim(-0.55, 3.55)
axE.spines[["top", "right"]].set_visible(False)
axE.annotate("THE FLIP: at p=0 far context HURTS\n(−2.24, T007 interference); ~200 events\n"
             "(≈20 exposures/pair) cancel ~1 nat —\nretrieval is net-positive already at p5",
             xy=(0.5, -1.75), xytext=(1.62, -2.62), ha="left", fontsize=8.2,
             color="#333333",
             arrowprops=dict(arrowstyle="->", lw=1.3, color="#a08020"),
             bbox=dict(boxstyle="round,pad=0.4", fc="#fdf9ec", ec="#c9a227"))
axE.text(0.99, 0.50,
         f"GRADED, not sharp (P2 refuted): +{E_FV[1]:.2f} → +{E_FV[2]:.2f} → "
         f"+{E_FV[3]:.2f}\n"
         f"adjacent ratios {E_RATIOS['5->20']:.2f}× then {E_RATIOS['20->60']:.2f}×\n"
         f"COMPARTMENTALIZED (P3): non-refrain far-value\n"
         f"{E_NONREFRAIN[1]:+.2f}/{E_NONREFRAIN[2]:+.2f}/{E_NONREFRAIN[3]:+.2f} "
         f"at p5/p20/p60 — zero leak",
         transform=axE.transAxes, ha="right", va="top", fontsize=7.6,
         bbox=dict(boxstyle="round,pad=0.42", fc="#f5f5f5", ec="#bbbbbb"))
axE2 = axE.twinx()
axE2.plot(xE, E_ACC, "--D", color=C_T, lw=2.2, ms=7, label="completion acc (k=1–3)",
          zorder=4)
for x, a in zip(xE, E_ACC):
    axE2.text(x + 0.07, a + 0.015, f"{a*100:.0f}%", fontsize=8.0, color=C_T,
              path_effects=PE)
axE2.set_ylim(0, 1.12)
axE2.set_ylabel("completion accuracy (right)", fontsize=9.0, color=C_T)
axE2.tick_params(axis="y", colors=C_T)
axE2.spines[["top"]].set_visible(False)
axE2.legend(loc="upper left", fontsize=7.6, framealpha=0.95)
panel_title(axE, "E · C4′  NEW", "The retrieval threshold",
            "T021 (e049): refrain density vs far-value — the flip and the head",
            foot="p0 bar hatched = shared-probe view (0 own events) on an off-parity net\n"
                 f"(val {E_VAL[0]:.2f} vs {E_VAL[1]:.2f}) — interference MAGNITUDE confounded, sign real;\n"
                 "'≤5%' rests on one n=20-events cell (shared-probe view: 5–20%); 10M arm 1.87×\n"
                 "more sensitive at p5 but only 491 steps (weak support)")
axH = fig.add_subplot(gsE[1, 0])
axH.axis("off")
axH.text(0.0, 0.92, "best head\n(refrain/control mass)", fontsize=7.6,
         ha="left", va="top", color="#333333", style="italic")
head_txt = [
    (f"{E_HEADS[0]['layer']}H{E_HEADS[0]['head']} · {E_HEAD_RATIO[0]:.2f}×", "no head", "#999999"),
    (f"{E_HEADS[1]['layer']}H{E_HEADS[1]['head']} · {E_HEAD_RATIO[1]:.2f}×", "still no head", "#999999"),
    (f"{E_HEADS[2]['layer']}H{E_HEADS[2]['head']} · {E_HEAD_RATIO[2]:.2f}×", "HEAD APPEARS", C_ID),
    (f"{E_HEADS[3]['layer']}H{E_HEADS[3]['head']} · {E_HEAD_RATIO[3]:.2f}×", "strongest", C_ID),
]
for i, (t1, t2, col) in enumerate(head_txt):
    axH.text(0.235 + i * 0.19, 0.97, t1, fontsize=8.6, ha="center", va="top",
             fontweight="bold", color=col,
             bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec=col, lw=1.1))
    axH.text(0.235 + i * 0.19, 0.46, t2, fontsize=7.2, ha="center", va="top",
             color=col)
axH.text(0.995, 0.0,
         "behavior grades; anatomy is lumpy — the head forms DISCRETELY,\n"
         "always in the last attention block (L5 at 6 layers; L7 in the 8-layer 10M)",
         fontsize=7.4, ha="right", va="bottom", color="#333333", style="italic")

# ================================================================ panel F (C7)
axF = fig.add_subplot(gs[2, 1])
axF.axis("off")
BOXES = [
    (0.012, 0.515, 0.487, 0.465, "#eef2f8", C_B,
     f"1 · ADDRESS — rows (concentrated, surgical)",
     f"• removal: {384} params, class-exact, corpus +0.0008 CE\n"
     f"• generalizes: rows alone → 13–17% acc band on EVERY\n  net tested (e046: B43 15.7%, BDO 13.9%)\n"
     f"• S_name {F_SNAME_ROWS:.0f} (rows) → {F_SNAME_PATCH:.0f} (+L3H5 patch, e042)\n"
     f"• scale-invariant shape (e005s): S_name {F_SNAME_SMALL:.0f} (0.84M) /\n"
     f"  {F_SNAME_LARGE:.0f} (10M) at +0.0005 CE\n"
     f"• complete erasure achieved ONCE; the second factor\n  is net-specific (e046 demotion)"),
    (0.501, 0.515, 0.487, 0.465, "#fdf6ee", C_R,
     "2 · ABILITY — body (distributed, train-only)",
     f"• NO surgical install: best row graft closes\n  {F_GRAFT_BEST*100:.1f}% of the gap; diff-init rows install\n"
     f"  nothing while damaging +{F_DIFFINIT['zclass_mean_dnll']:.2f} (e043)\n"
     f"• anchored exposure installs cheap + selective:\n"
     f"  Dmix@s400 NLL {F_DMIX['r1i']['nll']:.3f} / acc {F_DMIX['r1i']['acc']:.3f} at\n"
     f"  ΔCE +{F_DMIX['dce_val']:.4f}, S_install {F_DMIX['s_install']:.0f}\n"
     f"• rides incumbents' shared L0 machine — L0H3 stays\n  top-1 (ρ {F_ATLAS_RHO:.2f}); no private circuit grows"),
    (0.012, 0.02, 0.487, 0.465, "#eef7f0", C_T,
     "3 · EXPRESSION — free generation (a third thing)",
     f"• {F_DMIX['r1i']['acc']:.3f} battery acc yet ZEPHYRA ×0 in 2,800 generated\n  chars (e043)\n"
     f"• ×0 across seeds, T ∈ 0.7/1.0/1.3, doses s400–1600\n  (e048; battery holds 0.92–0.96 throughout)\n"
     f"• prior is geometry-bound: p(Z) {F_OA['std_p_z_battery_ctx130']:.3f} battery →\n"
     f"  {F_OA['std_p_z_gen_prompts']:.3f} free prompt → {F_OA['std_p_z_uniform']:.0e} uniform\n"
     f"• uniform-floor {F_HB['uniform_acc_std']:.3f} vs {F_HB['host_acc_std']:.3f} host —\n"
     f"  battery acc ≠ usable knowledge"),
    (0.501, 0.02, 0.487, 0.465, "#fdf0f0", C_COPY,
     "4 · HISTORY — scar (re-learned ≠ original)",
     f"• re-learn {F_V2['ratio']:.2f}× SLOWER, yet the zeroed row re-grows\n"
     f"  along its ORIGINAL direction (cos {F_SCAR['cos_wte']:.2f};\n"
     f"  {F_V2['cos_meta']['regrowth_frac']['wte_frac']*100:.0f}% norm regrowth); fresh name 0.28\n"
     f"• new route corr {F_RC['d2cond']['spearman_orig_vs_a_final']:.2f}; L3H5 flips\n"
     f"  carrier→anti-carrier ({F_RC['d2cond']['orig_top1_dce']:+.2f} → "
     f"{F_RC['d2cond']['L3H5_dce_a_final']:+.2f})\n"
     f"• re-erasibility degraded: {F_RE['d2_plus_patch']['acc']*100:.1f}% acc survives\n"
     f"  re-erase vs 0.13% originally (ΔCE +{F_RE['dce']:.3f})"),
]
for x, y, wbox, hbox, fc, ec, title, body in BOXES:
    axF.add_patch(FancyBboxPatch((x, y), wbox, hbox,
                                 boxstyle="round,pad=0.008",
                                 fc=fc, ec=ec, lw=1.3, transform=axF.transAxes,
                                 zorder=1))
    axF.text(x + 0.016, y + hbox - 0.035, title, fontsize=10.0,
             fontweight="bold", color=ec, transform=axF.transAxes, va="top")
    axF.text(x + 0.016, y + hbox - 0.105, body, fontsize=7.5,
             transform=axF.transAxes, va="top", linespacing=1.42, zorder=2)
axF.text(0.501 + 0.487 - 0.014, 0.02 + 0.014,
         "[ n=1 — single net, R7: numbers await replication ]",
         fontsize=7.6, fontweight="bold", color=C_COPY, ha="right", va="bottom",
         transform=axF.transAxes, bbox=FLAG_BX, zorder=3)
axF.text(0.501 + 0.016, 0.515 + 0.030,
         "[ protocol-fragile: onset wall was anchor-manufactured (T015) ]",
         fontsize=7.2, color="#8b4444", ha="left", va="bottom",
         transform=axF.transAxes, zorder=3)
axF.text(0.012 + 0.016, 0.02 + 0.028,
         "[ R8: dose legs bit-identical — instrument suspect;\n   dose-response OPEN, not refuted ]",
         fontsize=7.2, color="#8b4444", ha="left", va="bottom",
         transform=axF.transAxes, zorder=3, linespacing=1.3)
axF.set_title("F · C7  NEW — The edit law: four separable faculties",
              loc="left", fontsize=13.0, fontweight="bold", pad=30,
              color="#333333")
axF.text(1.0, 1.004,
         "T015 / T018 / T019 — address is concentrated; ability is trained;\n"
         "expression needs free-generation exposure; erasure leaves history",
         transform=axF.transAxes, ha="right", va="bottom", fontsize=8.0,
         style="italic", color="#444444")

# ---------------------------------------------------------------- poster chrome
fig.suptitle("The 2.7M char-transformer: anatomy after 48 hours",
             fontsize=20, fontweight="bold", y=0.972)
fig.text(0.5, 0.936,
         "portrait v2 (mechanism card T010→T023) · 6L/6H/192 char-GPT reference + "
         "0.84M/10M scale arms · every number from registered runs\n"
         "(e001 e013a e014b e018 e021 e012d e033 e035 e005s e041 e042 e043 e044 "
         "e046 e047 e048 e049) · panels E and F are new; A–D refreshed",
         ha="center", fontsize=9.5, color="#444444")
fig.text(0.5, 0.022,
         "C1 a mid-stack causal gate exists in every net, at a depth its history "
         "chooses · C2 anatomy plastic, front-loading defended (11→72→116×), write-norm "
         "schedule decorative · C3 partial init-anchoring\n"
         "C4 retrieval task-elicited, threshold low + graded + compartmentalized · "
         "C7 editing needs address, ability and expression — and erasure leaves history "
         "· n=1 and confound flags boxed/footnoted on every panel",
         ha="center", fontsize=9.6, style="italic", color="#666666")

fig.savefig(OUT / "portrait_v2.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- panel_data.json
panel_data = {
    "experiment": "v012_portrait_v2",
    "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
    "device": "cpu",
    "poster": str((OUT / "portrait_v2.png").relative_to(REPO)),
    "panels": {
        "A_C2_anatomies_stamped": {
            "nets": {name.split("  ")[0]: {"val_ce": v, "attn_damage": a,
                                           "mlp_damage": m}
                     for name, _c, a, m, v in A_NETS},
            "scale_front_loading_attn_l0_over_last": {
                "0.84M_4L": FL_SMALL, "2.7M_6L": FL_B, "10M_8L": FL_LARGE},
            "mlp0_keystone_10M": MLP0_10M,
            "mlp_rest_max_10M": MLP_REST_10M,
            "e033_equalizer": {
                "baseline_val": EQ_BASE, "equalized_val": EQ_EQ,
                "baseline_mlp": EQ_MLP_B, "equalized_mlp": EQ_MLP_E,
                "kl_l5_l4_equalized": EQ_KL,
                "flags": ["single net, 4L, <=1M envelope"],
            },
            "e047_replication": {"mlp5_energy_carrier": "5/5",
                                 "rotate_over_zero": "0.25-0.34",
                                 "l5_calibrator": "5/5, KL 0.91-1.08"},
            "sources": ["runs/e001/metrics.json", "runs/e014b/metrics.json",
                        "runs/e035/metrics.json", "runs/e005s/metrics.json",
                        "runs/e033/metrics.json", "runs/e047/metrics.json"],
        },
        "B_C1_causal_gate_reanchored": {
            "replaces": "v010 lens depth-histogram invariance claim "
                        "(T012/T014: instrument artifact)",
            "causal_hist_frac_d0_d5": B_HIST_FRAC,
            "modes": {n: m for n, _c, _h, m in B_NETS},
            "spearman_causal_vs_lens": B_SPEARMAN,
            "flip_curve_B": B_FLIPCURVE,
            "cross_net_hist_corr": {"seed_axis": B_XSEED, "regime_axis": B_XREGIME,
                                    "instrument_ceiling": [B_CEIL, B_CEIL_MAX]},
            "depth_budget_slide": [
                {"net": l, "relative_mode": r, "no_single_flip": n,
                 "modal_share": m, "strict_gate_pass": g}
                for l, r, n, m, g in B_SCALE],
            "suffix_monotone": B_SUFFMON,
            "flags": ["R8 steps confound: 4000/2226/1086 steps anti-correlate "
                      "with scale; LARGE stopped mid-cosine",
                      "4L failed the strict registered gate criterion "
                      "(mode at final block)"],
            "sources": ["runs/e012d/metrics.json", "runs/e018/metrics.json",
                        "runs/e005s/metrics.json"],
        },
        "C_C4_locality_funnel_plus_erosion": {
            "far_mass": C_FAR_M, "local_mass": C_LOC_M,
            "task_id_mass": C_ID_TASK, "control_id_mass": C_ID_CTRL,
            "best_head": C_HEAD,
            "scale_far_value_mean": FV, "p99_small_large": FV_P99,
            "sources": ["runs/e013a/metrics.json", "runs/e021/metrics.json",
                        "runs/e005s/metrics.json"],
        },
        "D_C3_anchoring_ladder": {
            "ladder": [{"step": lbl.replace("\n", " "), "cos_dw": v}
                       for lbl, v, _c in D_LADDER],
            "fraction_regime_over_ceiling": e041["fraction_regime_over_ceiling"],
            "verdict": e041["verdict"],
            "note": "unchanged since e041; e040 lineage test registered (T022)",
            "sources": ["runs/e041/metrics.json", "runs/e029/metrics.json"],
        },
        "E_retrieval_threshold_NEW": {
            "levels_pct": E_LEVELS,
            "far_value_prefix_k123": dict(zip(map(str, E_LEVELS), E_FV)),
            "acc_prefix_k123": dict(zip(map(str, E_LEVELS), E_ACC)),
            "events": dict(zip(map(str, E_LEVELS), E_EVTS)),
            "val_losses": dict(zip(map(str, E_LEVELS), E_VAL)),
            "adjacent_ratios": E_RATIOS,
            "non_refrain_far_value_means": dict(zip(map(str, E_LEVELS),
                                                    E_NONREFRAIN)),
            "best_heads": [
                {"level": l, "layer": h["layer"], "head": h["head"],
                 "refrain_mass": h["refrain_mass"],
                 "control_mass": h["control_mass"],
                 "ratio": r}
                for l, h, r in zip(E_LEVELS, E_HEADS, E_HEAD_RATIO)],
            "verdicts": e049["verdicts"],
            "flags": [
                "p0 = shared-probe view (0 own events), net off-parity "
                "(val 2.82): interference magnitude confounded, sign real",
                "'<=5%' rests on one n=20-events cell; shared-probe view 5-20%",
                "10M arm wall-matched only 491 steps (P4 weak-support)",
                "corpora 620KB ~75 epochs"],
            "sources": ["runs/e049/metrics.json"],
        },
        "F_edit_law_NEW": {
            "address": {
                "params": 384, "s_name_rows": F_SNAME_ROWS,
                "s_name_rows_plus_patch": F_SNAME_PATCH,
                "s_name_0.84M": F_SNAME_SMALL, "s_name_10M": F_SNAME_LARGE,
                "e046_rows_alone_acc_band": [
                    F_E046["B43"]["juliet_acc_train"],
                    F_E046["BDO"]["juliet_acc_train"]],
                "note": "general damager; complete erasure once, second factor "
                        "net-specific"},
            "ability": {
                "best_graft_gap_closed": F_GRAFT_BEST,
                "diff_init": F_DIFFINIT,
                "best_install": {"cell": "Dmix@s400",
                                 "nll": F_DMIX["r1i"]["nll"],
                                 "acc": F_DMIX["r1i"]["acc"],
                                 "dce": F_DMIX["dce_val"],
                                 "s_install": F_DMIX["s_install"]},
                "incumbent_atlas_spearman": F_ATLAS_RHO,
                "flags": ["protocol-fragile: onset wall anchor-manufactured"]},
            "expression": {
                "zephyra_in_2800_chars": 0,
                "zero_across": F_P1E["counts"],
                "p_z_battery": F_OA["std_p_z_battery_ctx130"],
                "p_z_free_prompt": F_OA["std_p_z_gen_prompts"],
                "p_z_uniform": F_OA["std_p_z_uniform"],
                "uniform_floor_acc": F_HB["uniform_acc_std"],
                "host_acc": F_HB["host_acc_std"],
                "flags": ["R8: dose legs bit-identical — instrument suspect; "
                          "dose-response open"]},
            "history": {
                "relearn_slower_ratio": F_V2["ratio"],
                "row_regrow_cos_wte": F_SCAR["cos_wte"],
                "wte_regrowth_frac": F_V2["cos_meta"]["regrowth_frac"]["wte_frac"],
                "fresh_name_cos": e044["scar_rows"]["b2_final"]["cos_wte"],
                "route_corr_orig_vs_relearned":
                    F_RC["d2cond"]["spearman_orig_vs_a_final"],
                "l3h5_orig_dce": F_RC["d2cond"]["orig_top1_dce"],
                "l3h5_relearned_dce": F_RC["d2cond"]["L3H5_dce_a_final"],
                "reerase_acc": F_RE["d2_plus_patch"]["acc"],
                "reerase_dce": F_RE["dce"],
                "flags": ["n=1 — single net, awaits replication (R7)"]},
            "thinking_refs": ["T015", "T018", "T019"],
            "sources": ["runs/e042/metrics.json", "runs/e043/metrics.json",
                        "runs/e044/metrics.json", "runs/e046/metrics.json",
                        "runs/e048/metrics.json", "runs/e005s/metrics.json"],
        },
    },
}
(OUT / "panel_data.json").write_text(
    json.dumps(panel_data, indent=2, default=float), encoding="utf-8")
print(f"[v012] poster   -> {OUT / 'portrait_v2.png'}")
print(f"[v012] data     -> {OUT / 'panel_data.json'}")
print(f"[v012] gate     -> modes "
      + ", ".join(f"{n} d{m}" for n, _c, _h, m in B_NETS)
      + f"; rel depth {B_SCALE[0][1]:.2f}/{B_SCALE[1][1]:.2f}/{B_SCALE[2][1]:.2f}")
print(f"[v012] threshold-> fv " + " / ".join(f"{v:+.2f}" for v in E_FV)
      + f"; head ratios " + "/".join(f"{r:.1f}x" for r in E_HEAD_RATIO))
