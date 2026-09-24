"""V010 — SELF-PORTRAIT: the lab's first poster (CPU-ONLY, no model touched).

The visual form of THINKING.md T010 (mechanism card v1). One figure, four
panels, one per high-confidence claim; every number is lifted verbatim from
registered run metrics (nothing recomputed, GPU untouched):

  A · C2  one function, three anatomies     e001 / e014b / e035 lesion maps
  B · C1  decision depth: invariant +       e012 / e012b / e035 / e021
           the L4 exception
  C · C4  the locality funnel               e013a (+ e021 ID-mass overlay)
  D · C3  the anchoring ladder              e041 (ladder incl. e029 pools)

Outputs: runs/v010/self_portrait.png, runs/v010/panel_data.json.
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
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
OUT = RUNS / "v010"
OUT.mkdir(parents=True, exist_ok=True)

# palette (lab convention: B blue, R orange, task green, COPY red emphasis)
C_B, C_R, C_T, C_COPY = "#4c72b0", "#dd8452", "#55a868", "#c44e52"
C_FAR, C_LOC, C_ID = "#8172b3", "#55a868", "#c44e52"
PE = [pe_mod.withStroke(linewidth=2.2, foreground="white")]


def load(name: str) -> dict:
    return json.loads((RUNS / name / "metrics.json").read_text(encoding="utf-8"))


e001, e012, e012b = load("e001"), load("e012"), load("e012b")
e013a, e014b, e021 = load("e013a"), load("e014b"), load("e021")
e035, e041 = load("e035"), load("e041")

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
# renorm rebuilt write allocation: attn write / pinned stream norm c=5.6
RENORM_WRITE_C = [w / e014b["c_renorm"] for w in e014b["renorm_profile"]["attn"]]

# ---------------------------------------------------------------- panel B data
# C1 — decision-depth histograms (bins: emb, L0..L5; trailing 'unstable' bin
# in the 8-entry arrays is 0 everywhere and is dropped)
DEPTH_BINS = ["emb", "L0", "L1", "L2", "L3", "L4", "L5"]


def frac7(counts: list[float]) -> list[float]:
    c = counts[:7]
    assert len(c) == 7 and sum(counts[7:]) == 0.0, counts
    return [x / sum(c) for x in c]


B_HISTS = [
    ("Shakespeare  (e012, n=2000)", C_B, "--o", frac7(e012["depth_histogram"])),
    ("Renorm  (e012b, n=2000)", C_R, "--s", frac7(e012b["depth_hist_renorm"])),
    ("Task filler  (e035, n=1000)", C_T, "--^",
     frac7(e035["q2_filler_depth"]["task_filler_counts"])),
    ("Task COPY  (e021, n=1000)", C_COPY, "-D",
     frac7(e021["depth_census"]["task_counts"])),
]
B_F_SH = frac7(e012["depth_histogram"])
B_L4_COPY = frac7(e021["depth_census"]["task_counts"])[5]
B_L5 = {name: f[6] for name, _c, _m, f in B_HISTS}
B_L4_NAT = [frac7(e012["depth_histogram"])[5], frac7(e012b["depth_hist_renorm"])[5],
            frac7(e035["q2_filler_depth"]["task_filler_counts"])[5]]

# ---------------------------------------------------------------- panel C data
# C4 — locality funnel on Shakespeare (e013a) + task-net ID-mass overlay (e021)
C_FAR_M = e013a["layer_far_mass"]
C_LOC_M = e013a["layer_local_mass"]
C_ID_TASK = e021["attention_census"]["task_layer_id_mass"]
C_ID_CTRL = e021["attention_census"]["control_layer_id_mass"]
C_HEAD = e021["attention_census"]["best_head"]

# ---------------------------------------------------------------- panel D data
# C3 — alignment ladder (e041): cos(ΔW_A, ΔW_B), ΔW = W_trained − W_init
LAD = e041["ladder"]
D_LADDER = [
    ("B vs B\nsame init · same order", LAD["same_init_same_order"]["value"], "#08306b"),
    ("B vs BDO\nsame init · diff order\n(CEILING)", LAD["same_init_diff_order"]["value"], "#4c72b0"),
    ("B vs R\nsame init · diff regime", LAD["same_init_diff_regime"]["pooled_same_init"], "#dd8452"),
    ("B vs B43, …\ndiff init · (FLOOR)", LAD["diff_init"]["value"], "#b0b0b0"),
]
D_ORGAN = e041["dw_ceiling"]["per_organ"]

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(16.0, 11.6))
gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.20,
                      left=0.055, right=0.975, top=0.852, bottom=0.100)
LAYERS = np.arange(6)


def panel_title(ax, claim, headline, sub):
    """Bold claim title well above the axes; small italic subtitle just above."""
    ax.set_title(f"{claim} — {headline}", loc="left", fontsize=13.5,
                 fontweight="bold", pad=30)
    ax.text(1.0, 1.004, sub, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.2, style="italic", color="#444444")


# ================================================================ panel A (C2)
gsA = gs[0, 0].subgridspec(1, 2, wspace=0.30)
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
                        fontsize=7.3, color=col, path_effects=PE)
    ax.set_xticks(LAYERS, [f"L{l}" for l in LAYERS], fontsize=9.5)
    ax.set_xlabel(kind, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
axA1.set_ylabel("zero-ablation damage  (Δval CE, nats)", fontsize=9.5)
axA1.set_ylim(0, 7.3)
panel_title(axA1, "A · C2", "One function, three anatomies",
            "organs move; early attention stays critical")
axA1.annotate("MLP-0 keystone:\nB +4.08, task +4.90", xy=(0.24, 4.97), xytext=(0.5, 7.08),
              ha="left", va="top", fontsize=9, fontweight="bold", color="#333333",
              arrowprops=dict(arrowstyle="->", lw=1.4, color="#333333"))
axA1.annotate("dissolves under renorm (+0.10)", xy=(0.0, 0.10), xytext=(0.5, 3.15),
              fontsize=8.6, color=C_R,
              arrowprops=dict(arrowstyle="->", lw=1.2, color=C_R))
axA1.annotate("L5 attention ≈ dead weight\nin every anatomy (≤ +0.03)", xy=(5.0, 0.06),
              xytext=(2.35, 1.15), fontsize=8.6, color="#333333",
              arrowprops=dict(arrowstyle="->", lw=1.2, color="#333333"))
axA2.annotate("MLP profile INVERTS\nto late-heavy in R", xy=(5.24, 0.72), xytext=(2.2, 2.9),
              ha="left", va="center", fontsize=8.6, color=C_R,
              arrowprops=dict(arrowstyle="->", lw=1.2, color=C_R))
axA2.text(0.985, 0.985,
          "renorm rebuilt a 9.3× declining write/c\nschedule — its rank order = damage\n"
          "rank order (ρ = 1.0)",
          transform=axA2.transAxes, ha="right", va="top", fontsize=7.8,
          bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#bbbbbb"))
handles = [Patch(fc=col, alpha=0.92, label=f"{name} · val {v:.3f}")
           for name, col, _a, _m, v in A_NETS]
axA1.legend(handles=handles, loc="upper right", fontsize=7.4, framealpha=0.95,
            title="anatomy (6L/6H/192 char-GPT, 2.7M params)", title_fontsize=7.4)

# ================================================================ panel B (C1)
axB = fig.add_subplot(gs[0, 1])
xb = np.arange(7)
axB.axvspan(4.55, 5.45, color=C_COPY, alpha=0.10, zorder=0)
for name, col, style, f in B_HISTS:
    emph = name.startswith("Task COPY")
    axB.plot(xb, np.array(f) * 100, style, color=col, lw=2.6 if emph else 1.8,
             ms=8 if emph else 5.5, label=name, zorder=4 if emph else 3,
             alpha=1.0 if emph else 0.85)
axB.text(5, B_L4_COPY * 100 + 2.2, f"{B_L4_COPY*100:.1f}% at L4", ha="center",
         fontsize=9, fontweight="bold", color=C_COPY, path_effects=PE)
axB.annotate("task COPY: 88.3% decided at L4\nretrieval completes BEFORE calibration\n"
             f"(natural profiles at L4: {min(B_L4_NAT)*100:.0f}–{max(B_L4_NAT)*100:.0f}%)",
             xy=(4.9, B_L4_COPY * 100 - 3), xytext=(0.55, 96), ha="left", va="top",
             fontsize=8.6, color=C_COPY,
             arrowprops=dict(arrowstyle="->", lw=1.3, color=C_COPY))
axB.annotate("the invariant: 54–62% finalize at L5\nin EVERY natural profile\n"
             "(renorm reshuffles mid-stack, corr 0.82)",
             xy=(5.95, B_L5["Shakespeare  (e012, n=2000)"] * 100 - 3.5),
             xytext=(3.15, 43), ha="left", va="center", fontsize=8.6,
             color="#333333",
             arrowprops=dict(arrowstyle="->", lw=1.3, color="#333333"))
axB.set_xticks(xb, DEPTH_BINS, fontsize=9.5)
axB.set_ylabel("% of positions decided at depth", fontsize=9.5)
axB.set_ylim(0, 100)
axB.set_xlim(-0.4, 6.4)
axB.legend(loc="upper left", bbox_to_anchor=(0.02, 0.66), fontsize=7.4,
           framealpha=0.95)
axB.spines[["top", "right"]].set_visible(False)
panel_title(axB, "B · C1", "Decision depth: the invariant and the exception",
            "one net, two stage profiles, selected per-position")

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
axC.annotate("the funnel: near-uniform at L0, tightest\n"
             "at L2, re-broadens at L5 (far is idle\n"
             "grazing — KL −6.9% under truncation)",
             xy=(0.06, 0.82), xytext=(0.95, 0.99), ha="left", va="top",
             fontsize=8.4, color="#333333",
             arrowprops=dict(arrowstyle="->", lw=1.2, color="#333333"))
axC.annotate(f"retrieval head L4-H1:\n{C_HEAD['id_mass']*100:.1f}% of its mass on the\n"
             f"5-char ID nonce (control {C_HEAD['control_same_head']*100:.0f}%)",
             xy=(4, C_ID_TASK[4] + 0.03), xytext=(5.15, 0.99), ha="right", va="top",
             fontsize=8.8, fontweight="bold", color=C_ID,
             arrowprops=dict(arrowstyle="->", lw=1.5, color=C_ID,
                             connectionstyle="arc3,rad=-0.18"))
axC.set_xticks(lc, [f"L{l}" for l in lc], fontsize=9.5)
axC.set_ylabel("share of attention mass", fontsize=9.5)
axC.set_ylim(0, 1.0)
axC.set_xlim(-0.25, 5.25)
axC.legend(loc="upper left", bbox_to_anchor=(0.235, 0.80), fontsize=7.2,
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
    axD.text(i, v + 0.03, f"{v:.3f}", ha="center", va="bottom", fontsize=12,
             fontweight="bold", path_effects=PE)
axD.axhline(0.0, color="k", lw=0.9)
axD.set_xticks(np.arange(4), [l for l, _v, _c in D_LADDER], fontsize=8.6)
axD.set_ylabel("cos(ΔW_A, ΔW_B)   organ motion", fontsize=9.5)
axD.set_ylim(-0.07, 1.16)
axD.set_xlim(-0.55, 3.55)
axD.spines[["top", "right"]].set_visible(False)
panel_title(axD, "D · C3", "The anchoring ladder",
            "the seed-anchored object is the residual-stream basis")
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

# ---------------------------------------------------------------- poster chrome
fig.suptitle("The 2.7M char-transformer: a first anatomy — day one",
             fontsize=19, fontweight="bold", y=0.975)
fig.text(0.5, 0.925,
         "mechanism card T010 · 6L/6H/192 char-GPT · 6 trained nets, ~5 GPU-hours · "
         "every number from registered runs (e001 e012 e012b e013a e014b e021 e035 e041)",
         ha="center", fontsize=10.5, color="#444444")
fig.text(0.5, 0.018,
         "C1 stages are the organism · C2 anatomy is plastic, damage tracks write "
         "allocation · C3 partial init-anchoring of the stream basis · C4 retrieval is "
         "task-elicited, not architectural",
         ha="center", fontsize=10, style="italic", color="#666666")

fig.savefig(OUT / "self_portrait.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- panel_data.json
panel_data = {
    "experiment": "v010_self_portrait",
    "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
    "device": "cpu",
    "poster": str((OUT / "self_portrait.png").relative_to(REPO)),
    "panels": {
        "A_C2_one_function_three_anatomies": {
            "x": "layer L0..L5; zero-ablation damage = Δval CE (nats)",
            "nets": {name.split("  ")[0]: {
                "label": name, "val_ce": v, "attn_damage": a, "mlp_damage": m}
                for name, _c, a, m, v in A_NETS},
            "mlp0_keystone": {"B": 4.079, "R": 0.097, "task": 4.896},
            "attn_L5_max": max(a[5] for _n, _c, a, _m, _v in A_NETS),
            "renorm_write_over_c": RENORM_WRITE_C,
            "write_damage_rank_order_rho": 1.0,
            "sources": ["runs/e001/metrics.json", "runs/e014b/metrics.json",
                        "runs/e035/metrics.json"],
        },
        "B_C1_decision_depth": {
            "bins": DEPTH_BINS,
            "histograms": {name: f for name, _c, _m, f in B_HISTS},
            "frac_at_L4": {name: f[5] for name, _c, _m, f in B_HISTS},
            "frac_at_L5": B_L5,
            "copy_L4_mode": B_L4_COPY,
            "shakespeare_L4": B_F_SH[5],
            "note": "bin convention [emb, L0..L5] per e012/e021 batched_snapshots "
                    "(index 5 = L4 readout). T009/E021 notes cite Shakespeare '8.0% "
                    "at L4' — that is the L3 bin (index 4); at L4 Shakespeare is "
                    "20.95%. Poster annotations use the histograms directly.",
            "renorm_vs_baseline_hist_corr": e012b["depth_hist_correlation"],
            "js_filler_vs_shakespeare": e035["q2_filler_depth"]["js_filler_vs_shakespeare"],
            "js_filler_vs_copy": e035["q2_filler_depth"]["js_filler_vs_copy"],
            "sources": ["runs/e012/metrics.json", "runs/e012b/metrics.json",
                        "runs/e035/metrics.json", "runs/e021/metrics.json"],
        },
        "C_C4_locality_funnel": {
            "far_mass": C_FAR_M, "local_mass": C_LOC_M,
            "task_id_mass": C_ID_TASK, "control_id_mass": C_ID_CTRL,
            "best_head": C_HEAD,
            "frac_prompts_L5_abandons_local": e013a["frac_prompts_L5_abandons_local"],
            "sources": ["runs/e013a/metrics.json", "runs/e021/metrics.json"],
        },
        "D_C3_anchoring_ladder": {
            "ladder": [
                {"step": lbl.replace("\n", " "), "cos_dw": v}
                for lbl, v, _c in D_LADDER],
            "same_init_diff_regime_B_vs_R_specific":
                LAD["same_init_diff_regime"]["value"],
            "per_organ_ceiling": {k: x["cos_dW_B_vs_BDO"]
                                  for k, x in D_ORGAN.items()},
            "fraction_regime_over_ceiling": e041["fraction_regime_over_ceiling"],
            "verdict": e041["verdict"],
            "sources": ["runs/e041/metrics.json", "runs/e029/metrics.json"],
        },
    },
}
(OUT / "panel_data.json").write_text(json.dumps(panel_data, indent=2, default=float),
                                     encoding="utf-8")
print(f"[v010] poster   -> {OUT / 'self_portrait.png'}")
print(f"[v010] data     -> {OUT / 'panel_data.json'}")
print(f"[v010] ladder   -> " + " -> ".join(f"{v:.3f}" for _l, v, _c in D_LADDER))
print(f"[v010] L4 modes -> " + ", ".join(f"{n.split('  ')[0]} {f[5]*100:.1f}%"
                                         for n, _c, _m, f in B_HISTS))
