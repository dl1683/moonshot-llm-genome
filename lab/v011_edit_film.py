# -*- coding: utf-8 -*-
"""
V011 — THE EDIT FILM (law L7, card v3)  [visualization, CPU-only, no torch]

A six-frame still "film strip" of the edit law, every number real, harvested
from saved metrics/probes:

  frame 1  BASELINE   e023  : JULIET known (acc .883 / NLL .41) + real sample
  frame 2  SURGERY    e023  : 384-param address burn (S_name 573, +0.0008 nats)
  frame 3  ERASURE    e042  : rows + L3H5 -> acc 0.0013  [n=1, net-specific]
  frame 4  INSTALL    e043+ : 0.09 NLL / 97% at +0.05; e048: 0 ZEPHYRA spoken,
                              p(Z) 0.556 -> 0.0898 -> 1.7e-6
  frame 5  SCAR       e044  : groove cos 0.760 vs 0.278; L3H5 flips anti-carrier
                              [n=1]
  frame 6  THE LAW    L7    : ADDRESS / ABILITY / EXPRESSION / HISTORY

Output: runs/v011/edit_film.png
Pure matplotlib (Agg). Run:  python lab/v011_edit_film.py
"""

import math
import os
import textwrap

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU-only lab policy
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = os.path.join("runs", "v011", "edit_film.png")

# ----------------------------------------------------------------------------
# palette (dark film-strip look)
BG_FIG = "#0a0b0e"    # theatre black
BG_FRAME = "#161a21"  # frame stock
BG_INSET = "#1e242e"  # chart inset
FG = "#e8e6e0"        # near-white text
MUTE = "#9aa3b2"      # muted text
DIM = "#b8bfcb"       # footer text
GREEN = "#7fd18a"     # known / good
RED = "#e06c6c"       # surgery / erasure
PALE_RED = "#e08a8a"
DARK_RED = "#a83a3a"
AMBER = "#e8b34b"     # install
PURPLE = "#b48cf2"    # expression
TEAL = "#5bc8c8"      # scar / history
GREY = "#55607a"      # control bars

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": FG,
    "axes.edgecolor": MUTE,
})

fig = plt.figure(figsize=(33, 12.6), facecolor=BG_FIG)

# --- film sprocket strips ----------------------------------------------------
NH = 46
for y in (0.977, 0.008):
    for i in range(NH):
        cx = (i + 0.5) / NH
        fig.patches.append(FancyBboxPatch(
            (cx - 0.0055, y), 0.011, 0.0155,
            boxstyle="round,pad=0.001,rounding_size=0.004",
            transform=fig.transFigure, facecolor="#2a2f3a",
            edgecolor="#3a4150", linewidth=0.6))

# --- title --------------------------------------------------------------------
fig.text(0.5, 0.948, "THE EDIT LAW",
         ha="center", va="center", fontsize=27, fontweight="bold", color=FG)
fig.text(0.5, 0.9305,
         "removal is surgical  ·  installation is plastic  ·  expression is "
         "separate  ·  history persists",
         ha="center", va="center", fontsize=16.5, color="#c9c4b8")
fig.text(0.5, 0.9165,
         "Law L7 — 2.7M-param char transformer — frames from real runs "
         "e023 / e042 / e043 / e044 / e048 — Neural Dissection Lab, "
         "2026-09-24/25",
         ha="center", va="center", fontsize=11, color=MUTE, style="italic")

# --- frame grid ----------------------------------------------------------------
M, G = 0.008, 0.006
FW = (1 - 2 * M - 5 * G) / 6
TOP, BOT, H = 0.903, 0.038, 0.847


def frame(i, accent):
    ax = fig.add_axes([M + i * (FW + G), BOT, FW, H])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(BG_FRAME)
    ax.patch.set_linewidth(1.4)
    ax.patch.set_edgecolor("#39404d")
    ax.add_patch(Rectangle((0, 0.962), 1, 0.038, facecolor=accent,
                           edgecolor="none", alpha=0.92))
    return ax


def inset(ax, x, y, w, h):
    axi = ax.inset_axes([x, y, w, h])
    axi.set_facecolor(BG_INSET)
    axi.patch.set_alpha(0.9)
    for s in axi.spines.values():
        s.set_color("#3a4150")
        s.set_linewidth(0.8)
    axi.tick_params(colors=MUTE, labelsize=8.5, length=2.5)
    return axi


def num(ax, x, y, s, color=FG, size=15, weight="bold", ha="left"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
            ha=ha, va="center", transform=ax.transAxes)


def cap(ax, x, y, s, size=10.2, color=MUTE, ha="left", weight="normal"):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="center",
            transform=ax.transAxes, fontweight=weight)


def snippet(ax, x, y, w, h, lines, accent, fsize=9.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.008,rounding_size=0.012",
                                transform=ax.transAxes,
                                facecolor="#10131a", edgecolor=accent,
                                linewidth=1.0, alpha=0.95))
    body = "\n".join(lines)
    ax.text(x + 0.025, y + h - 0.05, body, fontsize=fsize,
            family="monospace", color="#d6d2c8", va="top", ha="left",
            transform=ax.transAxes, linespacing=1.45)


# =============================================================================
# FRAME 1 — BASELINE
# =============================================================================
ax = frame(0, GREEN)
num(ax, 0.04, 0.935, "1 · BASELINE", GREEN, 19)
cap(ax, 0.04, 0.895, "JULIET is known — and speaks unprompted", 12.5, FG,
    weight="bold")
num(ax, 0.04, 0.835, "acc 0.883 · NLL 0.41", GREEN, 21)
cap(ax, 0.04, 0.787, "125 train-name contexts · rest-of-name 0.023 nats/char",
    10)
cap(ax, 0.04, 0.759, "logit(J | _JUL) = 9.99 — the address is hot", 10)

axi = inset(ax, 0.06, 0.565, 0.88, 0.175)
pp = [0.33, 0.97, 1.0, 1.0, 1.0, 1.0]
axi.bar(range(6), pp, color=GREEN, width=0.62)
axi.set_ylim(0, 1.16)
axi.set_xticks(range(6))
axi.set_xticklabels(list("JULIET"), fontsize=10, color=FG)
axi.set_yticks([0, 0.5, 1.0])
axi.set_yticklabels(["0", ".5", "1"])
for i, v in enumerate(pp):
    axi.text(i, v + 0.035, f"{v:.2f}", ha="center", fontsize=8.6, color=FG)
cap(ax, 0.06, 0.522, "per-position completion accuracy (e023, R1 battery)",
    9.6, MUTE)

snippet(ax, 0.05, 0.185, 0.90, 0.30, [
    "PROMPT: ...Grace go with you, Benedicite!",
    "",
    "GEN:    JULIET:",
    "   O, all the sight are to encounter,",
    "   The very bar word of hour that bled's sons,",
    "   Discoursed more upon the fair sons,",
], GREEN)
cap(ax, 0.05, 0.145, "real free generation (e023 probes, base p5) —",
    9.6, MUTE)
cap(ax, 0.05, 0.118, "the name surfaces on its own", 9.6, MUTE)

# =============================================================================
# FRAME 2 — SURGERY
# =============================================================================
ax = frame(1, RED)
num(ax, 0.04, 0.935, "2 · SURGERY", RED, 19)
cap(ax, 0.04, 0.895, "the 384-parameter address burn", 12.5, FG, weight="bold")
num(ax, 0.04, 0.835, "384 params · +0.0008 nats", RED, 21)
cap(ax, 0.04, 0.787, "zeroed: wte row ‘J’ + lm_head row ‘J’  (2 × 192)", 10)
cap(ax, 0.04, 0.759, "corpus cost · S_name 573× · ascent control 4,900× dearer",
    10)

axi = inset(ax, 0.06, 0.545, 0.88, 0.185)
before = [0.33, 0.97, 1.0, 1.0, 1.0, 1.0]
after = [0.0, 0.0, 0.0, 0.82, 0.0, 0.0]
w = 0.34
axi.bar([i - w / 2 for i in range(6)], before, width=w, color=GREEN,
        label="before")
axi.bar([i + w / 2 for i in range(6)], after, width=w, color=RED,
        label="after burn")
axi.set_ylim(0, 1.38)
axi.set_xticks(range(6))
axi.set_xticklabels(list("JULIET"), fontsize=10, color=FG)
axi.set_yticks([0, 0.5, 1.0])
axi.set_yticklabels(["0", ".5", "1"])
axi.legend(loc="upper right", fontsize=8.5, framealpha=0.25, labelcolor=FG)
cap(ax, 0.06, 0.502, "acc 0.883 → 0.136 (13.6%) — collateral class-exact:",
    9.6, MUTE)
cap(ax, 0.06, 0.475, "ROMEO +0.037, GLOUCESTER −0.0007 nats", 9.6, MUTE)

snippet(ax, 0.05, 0.185, 0.90, 0.26, [
    "GEN (before): JULIET: O, all the sight...",
    "GEN (after):  day things are to it,",
    "   As bites his thanks in senses; and by my true",
    "   And love his designs that slew...",
], RED)
cap(ax, 0.05, 0.145, "same prompt, burned net (e023 probes, D2/zero/both p0)",
    9.3, MUTE)
cap(ax, 0.05, 0.118, "JULIET in generation: 2 → 0 per 4,200 chars", 9.3, MUTE)

# =============================================================================
# FRAME 3 — COMPLETE ERASURE
# =============================================================================
ax = frame(2, DARK_RED)
num(ax, 0.04, 0.935, "3 · COMPLETE ERASURE", PALE_RED, 19)
cap(ax, 0.04, 0.895, "rows + one head — achieved once", 12.5, FG,
    weight="bold")
num(ax, 0.04, 0.835, "acc 0.0013 · NLL 8.72", PALE_RED, 21)
cap(ax, 0.04, 0.787, "JULIET after D2 rows + L3H5 patch (e042, D2+1h)", 10)
cap(ax, 0.04, 0.759, "content CE +0.0008 · per-pos acc [0, 0, 0, .008, 0, 0]",
    10)

axi = inset(ax, 0.10, 0.525, 0.80, 0.21)
vals = [88.3, 13.6, 0.13]
labs = ["baseline", "rows only", "rows + L3H5"]
cols = [GREEN, RED, "#7a2424"]
axi.bar(range(3), vals, color=cols, width=0.58)
axi.set_ylim(0, 106)
axi.set_xticks(range(3))
axi.set_xticklabels(labs, fontsize=9.5, color=FG)
axi.set_yticks([0, 50, 100])
axi.set_yticklabels(["0%", "50%", "100%"])
for i, v in enumerate(vals):
    axi.text(i, v + 2.5, f"{v}%", ha="center", fontsize=10, color=FG,
             fontweight="bold")
cap(ax, 0.10, 0.462, "the surgical ladder: address first, then the one head",
    9.6, MUTE)

ax.add_patch(FancyBboxPatch((0.05, 0.16), 0.90, 0.26,
                            boxstyle="round,pad=0.008,rounding_size=0.012",
                            transform=ax.transAxes, facecolor="#1c1216",
                            edgecolor=DARK_RED, linewidth=1.0))
cap(ax, 0.085, 0.365, "⚠  n = 1 — flagged, single net (B)", 11.2, PALE_RED,
    weight="bold")
cap(ax, 0.085, 0.320, "The ADDRESS half (rows → 13–17%) replicates on every "
    "net tested.", 9.6, FG)
cap(ax, 0.085, 0.290, "The COMPLETION half — which head to patch — is "
    "net-specific:", 9.6, FG)
cap(ax, 0.085, 0.260, "B43's L4H4 and BDO's L3H1 both fail the bar (e046).",
    9.6, FG)
cap(ax, 0.085, 0.222, "The residual's structure is a sample, not a law.",
    9.6, MUTE, weight="bold")

# =============================================================================
# FRAME 4 — INSTALL (with EXPRESSION sub-panel)
# =============================================================================
ax = frame(3, AMBER)
num(ax, 0.04, 0.935, "4 · INSTALL", AMBER, 19)
cap(ax, 0.04, 0.895, "exposure installs cheaply — and stays silent", 12.5,
    FG, weight="bold")
num(ax, 0.04, 0.835, "0.09 NLL · 97% acc", AMBER, 21)
cap(ax, 0.04, 0.787, "ZEPHYRA at ΔCE +0.05 · 400 exposure steps "
    "(e043 Dmix@s400)", 10)
cap(ax, 0.04, 0.759, "was NLL 6.67 / acc 9.8% · S_install ≈ 145–289", 10)

axi = inset(ax, 0.06, 0.565, 0.40, 0.175)
axi.bar([0, 1], [6.67, 0.09], color=[GREY, AMBER], width=0.55)
axi.set_ylim(0, 7.6)
axi.set_xticks([0, 1])
axi.set_xticklabels(["before", "installed"], fontsize=9, color=FG)
axi.set_yticks([0, 3, 6])
for i, v in enumerate([6.67, 0.09]):
    axi.text(i, v + 0.22, f"{v}", ha="center", fontsize=9.5, color=FG)
cap(ax, 0.06, 0.522, "battery NLL", 9.3, MUTE)

axi = inset(ax, 0.52, 0.565, 0.42, 0.175)
pz = [0.556, 0.0898, 1.7e-6]
hgt = [-math.log10(max(v, 1e-7)) for v in pz]  # taller bar = smaller p
axi.bar(range(3), hgt, color=[PURPLE, PURPLE, "#4a3d66"], width=0.55)
axi.set_ylim(0, 7.0)
axi.set_xticks(range(3))
axi.set_xticklabels(["battery\nctx", "free\nprompt", "uniform\nctx"],
                    fontsize=8.2, color=FG)
axi.set_yticks([])
for i, t in enumerate(["0.556", "0.0898", "1.7e-6"]):
    axi.text(i, hgt[i] + 0.22, t, ha="center", fontsize=9.3, color=PURPLE,
             fontweight="bold")
cap(ax, 0.52, 0.522, "p(Z) — log scale ↓ (e048)", 9.3, MUTE)

snippet(ax, 0.05, 0.185, 0.90, 0.26, [
    "GEN (installed net):  ELIZABETH:",
    "   George to his hope; and the regal is morn;",
    "   As my mother did set to hear the flower slain?",
], PURPLE)
cap(ax, 0.05, 0.145, "EXPRESSION: 0 × ZEPHYRA in 2,800 chars — all doses, "
    "temps,", 9.3, PURPLE, weight="bold")
cap(ax, 0.05, 0.118, "seeding, greedy — battery holds 92–96% the whole time",
    9.3, MUTE)

# =============================================================================
# FRAME 5 — SCAR
# =============================================================================
ax = frame(4, TEAL)
num(ax, 0.04, 0.935, "5 · SCAR", TEAL, 19)
cap(ax, 0.04, 0.895, "re-learning regrows the old groove", 12.5, FG,
    weight="bold")
num(ax, 0.04, 0.835, "cos 0.760 vs 0.278", TEAL, 21)
cap(ax, 0.04, 0.787, "re-learned J-row vs original · fresh-name control 0.278",
    10)
cap(ax, 0.04, 0.759, "58% norm regrowth · 2.08× slower to re-learn", 10)

axi = inset(ax, 0.06, 0.565, 0.40, 0.175)
axi.bar([0, 1], [0.760, 0.278], color=[TEAL, GREY], width=0.55)
axi.set_ylim(0, 0.92)
axi.set_xticks([0, 1])
axi.set_xticklabels(["re-learned\nJULIET", "fresh\nname"], fontsize=8.6,
                    color=FG)
for i, v in enumerate([0.760, 0.278]):
    axi.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=9.5, color=FG)
cap(ax, 0.06, 0.522, "wte ‘J’ direction (e044)", 9.3, MUTE)

axi = inset(ax, 0.52, 0.565, 0.42, 0.175)
axi.bar([0, 1], [0.88, -2.03], color=[GREEN, RED], width=0.55)
axi.axhline(0, color=MUTE, linewidth=0.8)
axi.set_ylim(-2.75, 1.45)
axi.set_xticks([0, 1])
axi.set_xticklabels(["L3H5\nbefore", "L3H5\nafter relearn"], fontsize=8.6,
                    color=FG)
axi.set_yticks([-2, 0])
axi.text(0, 1.00, "+0.88", ha="center", fontsize=9.5, color=GREEN)
axi.text(1, -2.42, "−2.03", ha="center", fontsize=9.5, color=RED)
cap(ax, 0.52, 0.522, "L3H5: carrier → anti-carrier (nats)", 9.3, MUTE)

ax.add_patch(FancyBboxPatch((0.05, 0.16), 0.90, 0.26,
                            boxstyle="round,pad=0.008,rounding_size=0.012",
                            transform=ax.transAxes, facecolor="#0f1a1a",
                            edgecolor=TEAL, linewidth=1.0))
cap(ax, 0.085, 0.365, "new route ≠ old route — and it resists the old key",
    10.6, TEAL, weight="bold")
cap(ax, 0.085, 0.320, "route atlas ρ 0.21 · same surgical key re-applied:",
    9.6, FG)
cap(ax, 0.085, 0.290, "0.13% → 44.5% of the memory survives (+0.021 CE)",
    9.6, FG)
cap(ax, 0.085, 0.260, "the second memory won't fit the first's key", 9.6, FG)
cap(ax, 0.085, 0.222, "⚠  n = 1 — history clause awaits replication (R7)",
    9.6, AMBER)

# =============================================================================
# FRAME 6 — THE LAW
# =============================================================================
ax = frame(5, "#9aa5b8")
num(ax, 0.04, 0.935, "6 · THE LAW", "#c9d2e0", 19)
cap(ax, 0.04, 0.895, "four separable faculties (L7)", 12.5, FG, weight="bold")

boxes = [
    ("ADDRESS", GREEN,
     "Concentrated in token rows. Burn 384 of 2.7M params and the name "
     "drops to 13.6% for +0.0008 corpus nats. Surgical, robust, "
     "replicates on every net."),
    ("ABILITY", AMBER,
     "Distributed in the body. Cannot be written — best graft closes ~6% "
     "of the gap — but exposure trains it cheaply: 0.09 NLL / 97% at "
     "+0.05 nats."),
    ("EXPRESSION", PURPLE,
     "A third faculty. Teacher-forced install never speaks: 0 ZEPHYRA at "
     "any dose or temperature; p(Z) collapses 0.556 → 1.7e-6 out of its "
     "geometry."),
    ("HISTORY", TEAL,
     "The attractor survives erasure. The address regrows in the old "
     "groove (cos 0.76) and the new route resists the old key "
     "(0.13% → 44.5%).  [n=1]"),
]
y = 0.785
for name, col, txt in boxes:
    hh = 0.150
    ax.add_patch(FancyBboxPatch((0.055, y - hh), 0.89, hh,
                                boxstyle="round,pad=0.006,"
                                "rounding_size=0.012",
                                transform=ax.transAxes,
                                facecolor="#11141b", edgecolor=col,
                                linewidth=1.1))
    ax.text(0.085, y - hh / 2, name, fontsize=13, fontweight="bold",
            color=col, va="center", transform=ax.transAxes)
    wrapped = "\n".join(textwrap.wrap(txt, width=48))
    ax.text(0.325, y - hh / 2, wrapped, fontsize=8.8, color=FG, va="center",
            transform=ax.transAxes, linespacing=1.4)
    y -= hh + 0.021

cap(ax, 0.5, 0.038, "sources: runs/e023 · e042 · e043 · e044 · e048 — "
    "lab/v011_edit_film.py", 8.6, DIM, ha="center")

# --- footer --------------------------------------------------------------------
fig.text(0.5, 0.0265,
         "Frames 3 & 5 carry the lab’s honesty flags (n=1: net-specific "
         "completion / unreplicated history clause). Laws are ensemble "
         "properties; mechanisms are samples.",
         ha="center", fontsize=9.5, color=MUTE, style="italic")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=115, facecolor=BG_FIG)
print("wrote", OUT)
