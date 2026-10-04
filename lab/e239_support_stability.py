"""E239 — FQ6: THE CROSS-WASH SUPPORT-STABILITY READ (eval-only; CPU; 2026-10-04).

THE QUESTION (the ideator's Rank 1; R63's named W037-direct-test): are the
54 supports stable AS VECTORS at matched deep states across independent
washes? W037's defensible form ("behavior-INDEXED objects stable; the u0
sign-ray family a cloud") vs its deflation ("stable as READS only"). THE
THIRD BRANCH IS THE PRIZE: if supports are set-level stable but
vector-cloudy, the fact is POPULATION-stable — the shape/height law at the
support level.

THE CELL (dispatch brief, frozen): (1) re-derive all 54 supports at
{+50, +80} under wash-1 AND wash-2 (the committed checkpoint states
e182c_s{50,80} + e182c2_fresh_s{50,80}; e226's machinery module-imported
VERBATIM — the 54 probes, fp64 dots, the committed states; the t=0 bank
as the anchor); (2) THE READS: (a) cross-wash vector stability
cos(s@w1(t), s@w2(t)) per probe at each state — the distribution vs the
u0 cloud (e231's committed cross-draw cos: J1 0.3184 / J2 0.3339 /
J3 0.3539 / J4 0.3703, mean 0.3441) and the random null; (b) within-wash
rotation cos(s@w(t), s@0(t)) (e226 measured it at +80 for all 54 — here
at BOTH states, both washes, and G_REPRO-certified against e226's
records); (c) SET-level stability: the rank correlation (Spearman rho) of
the 54 probes' rotation amounts across washes (does the POPULATION rotate
coherently?); (3) the anchors (Gmail/iPhone) called out against the
product-family band.

REGISTERED BARS (frozen BEFORE compute, VERBATIM from the dispatch; the
dispatch brief is the registration; adjudicate against exactly this; no
bar shopping):
  STABLE-AS-VECTORS — "median cross-wash cos >= 0.7 at both states, far
  above the u0 cloud — W037 earns physics: behavior-indexed directions
  are stable where corpus-anchored rays cloud"
  CLOUD-LEVEL — "median cross-wash cos ~ 0.3-0.4 (the u0 cloud's range)
  — W037 deflates to reads-only; examine the third branch: if the
  SET-level rotation ranks correlate cross-wash (rho >= 0.6), the fact
  is POPULATION-stable — the law at the support level (name it)"
  MIXED — "between — the distributions verbatim (per battery), no
  narrative inflation"

FROZEN OPERATIONALIZATION (registered with the bars, before any compute):
  * states = {+50, +80}; washes = {w1 (e182c replay, seed 18202), w2
    (e182c2 fresh, seed 20261002)}; median over ALL 54 probes of
    cos(s_i@w1(t), s_i@w2(t)).
  * STABLE-AS-VECTORS := median(+50) >= 0.7 AND median(+80) >= 0.7.
  * CLOUD-LEVEL := median(+50) in [0.30, 0.40] AND median(+80) in
    [0.30, 0.40] (the registered range, verbatim).
  * MIXED := neither of the above (everything between, including split
    states).
  * THE THIRD BRANCH (declared only under CLOUD-LEVEL, per the bar text;
    co-reported always): Spearman rho over the 54 probes of the rotation
    AMOUNTS across washes at a state, rotation amount = angle
    acos(cos(s@w(t), s@0)) — rank-identical under any monotone
    reparametrization (cos / 1-cos / angle give the same rho; no choice
    to shop); POPULATION-stable := rho >= 0.6 at either state;
    permutation null (20,000 shuffles, seed 239) co-reported.
  * u0 cloud reference = e231's committed cross-draw cos values (above);
    random null = cos of two random unit vectors in R^124,439,808,
    N(0, 1/sqrt(P)) analytic (sd 2.84e-4) + 16 empirical draws.
  * anchors (Gmail/iPhone) co-reported against the product family's 5
    non-anchor probes (mean +/- 2 sd), e226's convention — CALLED OUT,
    never adjudicated (the bars are population-level).

CHECKS (registered): the states' provenance (every battery's p re-probed
on every LOADED state vs the committed records, tol 0.005 — e228's
re-probe convention); batch-1 support noise carried as the floor
(e226's determinism self-cos ~1.0 within-session; probe-0 recompute
re-verified here on the t=0 bank AND the first state bank); G_REPRO —
the recomputed t=0 Gram reproduces e226's committed support_geometry_t0
and the +80 rotations reproduce e226's committed rotation records (all
54 x 2 washes, tol 5e-3); the e208/T215 distinctions restated where
relevant (e208's margins are u0-sign-ray objects — the corpus-anchored
class W037 says clouds; T215's family-clause negative: the family's
other members' supports do NOT share the anchor's direction — the band
is a noise band of per-probe objects, not a shared-direction family);
nothing guaranteed.

MACHINERY: lab/e226_interior.py MODULE-IMPORTED VERBATIM (probe_support,
probe_p, dot_vec_rows, cache_row_norms, gram_cache, CACHE_SCALE/CHUNK,
W_ARCH state paths, the battery-rebuild blocks copied from its main(),
and through it e182c + e182c2's VERBATIM organism/corpus/battery
builders). Supports cached as fp16 (2^14 scale) bank files in the
system TEMP dir (the repo sits under OneDrive; 5 banks x 13.4 GB do not
belong in the synced tree); every dot fp64-accumulated in chunks (fp32
products of fp16 inputs are exact; e226's disclosed ~5e-4 cache floor
carries). Banks deleted on DONE (disclosed; the dot records + G_REPRO
certify the reads; resume-safe via journal until then).

Envelope: CPU-only, torch threads 4 (two other agents live — load
checks per phase, RAM floor 12 GB before bank passes). No GPU. No
NOTES/THINKING/QUEUE/STATE edits (dispatch). Smoke via E239_SMOKE=1
(subset probes, +80 only; nothing adjudicated).
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import statistics as st
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402

import common                                        # noqa: E402
from common import now_iso, run_dir, save_json       # noqa: E402

import e226_interior as e226                         # noqa: E402 — the support machinery, VERBATIM
import e182c_forgetting_control as e1                # noqa: E402 — phase-1 machinery, VERBATIM
import e182c2_template as e2                         # noqa: E402 — the template battery, VERBATIM

# e226 resets torch threads to 4 at module level (the shared-box envelope)
THREADS = 4
torch.set_num_threads(THREADS)

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import textwrap                                      # noqa: E402

try:
    import psutil                                    # noqa: E402
except ImportError:
    psutil = None

SMOKE = os.environ.get("E239_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e239_smoke" if SMOKE else "e239"

T0 = time.time()
_log_t0 = time.time()
def log(m: str) -> None:
    print(f"[{time.time() - _log_t0:7.1f}s] {m}", flush=True)

# ------------------------------------------------------------ the cell
STATES = (80,) if SMOKE else (50, 80)
WASHES = ("w1", "w2")
W_LABEL = {"w1": "wash 1 (e182c replay, seed 18202)",
           "w2": "wash 2 (e182c2 fresh, seed 20261002)"}
W_ARCH = {"w1": {s: e226.W_ARCH["w1"][s] for s in STATES},
          "w2": {s: e226.W_ARCH["w2"][s] for s in STATES}}
BANK_TAGS = ("t0",) + tuple(f"{w}s{s}" for w in WASHES for s in STATES)

# the committed records (re-probe + G_REPRO anchors)
E226_M = common.REPO / "runs" / "e226" / "metrics.json"
E231_M = common.REPO / "runs" / "e231" / "metrics.json"

# u0 cloud (e231's committed cross-draw cos; T213's "~0.35")
def _load_u0_cloud() -> dict:
    m = json.loads(E231_M.read_text(encoding="utf-8"))
    vals = [float(m["cells"][c]["second_stream_coreads"]
                  ["cos_u0_10902_vs_10914"])
            for c in ("J1", "J2", "J3", "J4")]
    return {"per_cell": vals, "mean": sum(vals) / len(vals),
            "min": min(vals), "max": max(vals)}

# ------------------------------------------------------------ frozen bars
REGISTERED_PREDICTION = {
    "bars_verbatim": {
        "STABLE-AS-VECTORS": "median cross-wash cos >= 0.7 at both "
        "states, far above the u0 cloud — W037 earns physics: "
        "behavior-indexed directions are stable where corpus-anchored "
        "rays cloud",
        "CLOUD-LEVEL": "median cross-wash cos ~ 0.3-0.4 (the u0 cloud's "
        "range) — W037 deflates to reads-only; examine the third branch: "
        "if the SET-level rotation ranks correlate cross-wash "
        "(rho >= 0.6), the fact is POPULATION-stable — the law at the "
        "support level (name it)",
        "MIXED": "between — the distributions verbatim (per battery), no "
        "narrative inflation",
    },
    "operationalization": (
        "states {+50,+80}; washes {w1 (e182c replay), w2 (e182c2 fresh)}; "
        "median over all 54 probes of cos(s_i@w1(t), s_i@w2(t)); "
        "STABLE-AS-VECTORS := median(+50) >= 0.7 AND median(+80) >= 0.7; "
        "CLOUD-LEVEL := median(+50) in [0.30,0.40] AND median(+80) in "
        "[0.30,0.40] (the registered range verbatim); MIXED := neither "
        "(incl. split states); THIRD BRANCH declared only under "
        "CLOUD-LEVEL, co-reported always: Spearman rho of the 54 "
        "probes' rotation amounts (angle acos(cos(s@w(t), s@0)) — rank-"
        "identical under any monotone reparametrization) across washes "
        "at a state; POPULATION-stable := rho >= 0.6 at either state; "
        "permutation null 20k shuffles seed 239; u0 cloud = e231's "
        "committed cross-draw cos {0.3184,0.3339,0.3539,0.3703} mean "
        "0.3441; random null N(0,1/sqrt(P)) sd 2.84e-4 analytic + 16 "
        "empirical draws; anchors co-reported vs the product family's 5 "
        "non-anchor probes (mean +/- 2sd), never adjudicated"
    ),
    "registration": "bars frozen VERBATIM from the dispatch brief (the "
        "ideator's FQ6 spec as dispatched by the coordinator, R63's "
        "named W037-direct-test) BEFORE any compute; the dispatch brief "
        "is the registration; adjudicate against exactly this; no bar "
        "shopping",
}

deviations: list[str] = [
    "The support banks live in the system TEMP dir, not the repo (the "
    "repo tree sits under OneDrive sync; 5 banks x 13.4 GB do not belong "
    "in the synced tree) and are DELETED on DONE (the journal + full dot "
    "records + G_REPRO certification remain; resume-safe until DONE).",
    "All cross-bank dots are computed from the fp16 banks (both sides "
    "rounded), unlike e226's rotation dots (fresh fp32 vector vs fp16 "
    "cache row) — one extra fp16 rounding per dot, within e226's "
    "disclosed ~5e-4 cache floor; G_REPRO quantifies the net drift "
    "against e226's committed values.",
    "The t=0 bank is RECOMPUTED (deterministic same-session pipeline as "
    "e226); it is not bit-certified against e226's in-RAM cache (which "
    "was never persisted) — G_REPRO-A anchors it to e226's committed "
    "t=0 geometry (Gram-derived reads, tol 5e-3).",
    "No wash-gradient work (no G_DRAWS): the reads are support-only at "
    "the committed states; the draw streams are not consumed. The FD "
    "gates of e226 are not re-run: E239 uses supports as DIRECTIONS for "
    "cos reads only (never as step directions); e226 already FD-gated "
    "all 54 t=0 supports and both anchors' +80 supports.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch).",
    "Smoke mode (E239_SMOKE=1): 11 probes (product family + founder "
    "anchor + one per other battery), state +80 only; nothing "
    "adjudicated or gated (SMOKE stamp).",
]

trims: list[str] = []

# ------------------------------------------------------------ envelope

def cpu_load_check(tag: str) -> dict:
    if psutil is None:
        return {"tag": tag, "psutil": "absent"}
    rec = {"tag": tag,
           "cpu_percent": psutil.cpu_percent(interval=0.5),
           "ram_avail_gb": round(psutil.virtual_memory().available / 2**30, 1),
           "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 1)}
    log(f"  [load] {tag}: cpu {rec['cpu_percent']}% "
        f"ram_avail {rec['ram_avail_gb']} GB")
    return rec


def ram_floor_ok(floor_gb: float, why: str) -> bool:
    if psutil is None:
        return True
    avail = psutil.virtual_memory().available / 2**30
    if avail < floor_gb:
        log(f"FATAL: RAM avail {avail:.1f} GB < floor {floor_gb} GB ({why})")
        return False
    return True


# ------------------------------------------------------------ bank math
# fp16 banks on disk (np.memmap), torch views over them; every dot
# fp64-accumulated over column chunks. fp32 products of two fp16 inputs
# are EXACT (11-bit mantissas -> 22-bit products fit fp32's 24), so the
# e226 exactness class carries (disclosed floor ~5e-4 from the fp16
# rounding itself).

BANK_DIR = Path(tempfile.gettempdir()) / ("e239_banks_smoke" if SMOKE
                                          else "e239_banks")


def bank_path(tag: str) -> Path:
    return BANK_DIR / f"bank_{tag}.fp16"


def open_bank_rw(tag: str, n_p: int, n_params: int) -> np.memmap:
    return np.memmap(bank_path(tag), dtype=np.float16, mode="w+",
                     shape=(n_p, n_params))


def open_bank_r(tag: str, n_p: int, n_params: int) -> np.memmap:
    # mode "r+" keeps the numpy array writeable (torch.from_numpy warns
    # on read-only arrays); we never write through these views
    return np.memmap(bank_path(tag), dtype=np.float16, mode="r+",
                     shape=(n_p, n_params))


def bank_view(mm: np.memmap) -> torch.Tensor:
    return torch.from_numpy(mm)


def diag_dots(A: torch.Tensor, B: torch.Tensor) -> list[float]:
    """Row-wise dot(A[i], B[i]) — fp32 products, fp64 accumulation, one
    streaming pass over column chunks (e226's cache_row_norms pattern)."""
    n = A.shape[0]
    out = torch.zeros(n, dtype=torch.float64)
    for s in range(0, A.shape[1], e226.CHUNK):
        sl = slice(s, min(s + e226.CHUNK, A.shape[1]))
        a = A[:, sl].to(torch.float32)
        b = B[:, sl].to(torch.float32)
        out += (a * b).sum(1, dtype=torch.float64)
    return out.tolist()


# ------------------------------------------------------------ stats

def rank_avg(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    rx, ry = rank_avg(x), rank_avg(y)
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    return float((rx * ry).mean())


def perm_null_rho(x, y, n_perm: int = 20_000, seed: int = 239) -> dict:
    """Spearman rho with a permutation null (shuffle y's ranks)."""
    rx, ry = rank_avg(x), rank_avg(y)
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    obs = float((rx * ry).mean())
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for t in range(n_perm):
        null[t] = rx.dot(rng.permutation(ry)) / len(ry)
    return {"obs": obs, "null_mean": float(null.mean()),
            "null_sd": float(null.std(ddof=1)),
            "p_ge_obs": float((np.sum(null >= obs) + 1) / (n_perm + 1)),
            "n_perm": n_perm, "seed": seed}


# ------------------------------------------------------------ plots

BAT_COLOR = {"fact": "#1a6faf", "ctrl": "#c0392b",
             "near": "#0d5c3f", "tmpl": "#8e44ad"}


def make_plot_crosswash(rd, crosswash, probes, u0c, verdict, medians):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    for ax, t in zip((axes[0][0], axes[0][1]), sorted(crosswash)):
        vals = crosswash[t]
        for bi, b in enumerate(("fact", "ctrl", "near", "tmpl")):
            ys = [v for v, r in zip(vals, probes) if r["battery"] == b]
            xs = bi + np.linspace(-0.18, 0.18, len(ys)) if len(ys) > 1 \
                else np.array([bi])
            ax.scatter(xs, ys, s=26, alpha=0.75, color=BAT_COLOR[b],
                       label=f"{b} (n={len(ys)})")
        ax.axhspan(u0c["min"], u0c["max"], color="#f2c14e", alpha=0.35,
                   zorder=0,
                   label=f"u0 cloud [{u0c['min']:.3f},{u0c['max']:.3f}] "
                         f"(e231)")
        ax.axhline(0.7, color="#0d5c3f", lw=1.4, ls="--",
                   label="STABLE bar 0.7")
        ax.axhline(0.0, color="0.55", lw=0.8, ls=":")
        ax.axhline(st.median(vals), color="k", lw=1.2, ls="-.",
                   label=f"median {st.median(vals):.4f}")
        ax.set_xticks(range(4))
        ax.set_xticklabels(("fact", "ctrl", "near", "tmpl"))
        ax.set_title(f"CROSS-WASH cos(s@w1(+{t}), s@w2(+{t})) — all "
                     f"{len(vals)} probes", fontsize=10)
        ax.set_ylabel("cos")
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(alpha=0.25)
    ax = axes[1][0]
    ax.hist([crosswash[t] for t in sorted(crosswash)], bins=36,
            range=(-0.2, 1.0), label=[f"+{t}" for t in sorted(crosswash)],
            color=["#7fb3d8", "#1a6faf"][:len(crosswash)], alpha=0.85)
    ax.axvspan(u0c["min"], u0c["max"], color="#f2c14e", alpha=0.35,
               label="u0 cloud (e231)")
    ax.axvline(0.7, color="#0d5c3f", lw=1.6, ls="--", label="STABLE bar")
    ax.axvline(0.0, color="0.55", lw=0.8, ls=":")
    for t in sorted(crosswash):
        ax.axvline(st.median(crosswash[t]), color="k", lw=1.0, ls="-.")
    ax.set_title("the cross-wash distributions vs the u0 cloud "
                 "(dot-dash = medians)", fontsize=10)
    ax.set_xlabel("cos")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax = axes[1][1]
    band_idx = [i for i, r in enumerate(probes)
                if r["family6"] == e226.BAND_FAMILY
                and r["fact"] not in (e226.ANCHOR_G, e226.ANCHOR_I)]
    gi = next(i for i, r in enumerate(probes) if r["fact"] == e226.ANCHOR_G)
    ii = next(i for i, r in enumerate(probes) if r["fact"] == e226.ANCHOR_I)
    for k, t in enumerate(sorted(crosswash)):
        band = [crosswash[t][j] for j in band_idx]
        mu, sd = st.mean(band), st.stdev(band)
        ax.errorbar([k - 0.12], [mu], yerr=[[2 * sd], [2 * sd]],
                    fmt="o", color="0.45", capsize=4, ms=7,
                    label="product band ±2σ" if k == 0 else None)
        ax.plot([k + 0.08], [crosswash[t][gi]], "*", color="#1a6faf",
                ms=17, label="Gmail" if k == 0 else None)
        ax.plot([k + 0.22], [crosswash[t][ii]], "*", color="#c0392b",
                ms=17, label="iPhone" if k == 0 else None)
    ax.axhspan(u0c["min"], u0c["max"], color="#f2c14e", alpha=0.25,
               zorder=0)
    ax.axhline(0.7, color="#0d5c3f", lw=1.4, ls="--")
    ax.set_xticks(range(len(sorted(crosswash))))
    ax.set_xticklabels([f"+{t}" for t in sorted(crosswash)])
    ax.set_title("THE ANCHORS vs the family band (called out; not "
                 "adjudicated)", fontsize=10)
    ax.set_ylabel("cross-wash cos")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    wrap = textwrap.fill(verdict, 96)
    fig.suptitle("E239 — FQ6: THE CROSS-WASH SUPPORT-STABILITY READ\n"
                 + wrap, fontsize=11.5)
    fig.text(0.5, 0.012, "medians: " + " | ".join(
        f"+{t}: {medians[t]:.4f}" for t in sorted(medians))
        + f" — u0 cloud mean {u0c['mean']:.4f} — random null "
        f"0 ± 2.8e-4", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.025, 1, 0.92))
    png = rd / f"{NAME}_crosswash.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


def make_plot_setlevel(rd, rot_angles, crosswash, probes, rhos, verdict):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))
    for ax, t in zip((axes[0], axes[1]), sorted(rot_angles)):
        x1 = [math.degrees(a) for a in rot_angles[t]["w1"]]
        x2 = [math.degrees(a) for a in rot_angles[t]["w2"]]
        lim = max(max(x1), max(x2)) * 1.08
        ax.plot([0, lim], [0, lim], ":", color="0.55", lw=1,
                label="y=x (same rotation)")
        for b in ("fact", "ctrl", "near", "tmpl"):
            xs = [v for v, r in zip(x1, probes) if r["battery"] == b]
            ys = [v for v, r in zip(x2, probes) if r["battery"] == b]
            ax.scatter(xs, ys, s=30, alpha=0.8, color=BAT_COLOR[b],
                       label=b)
        for fact, cl in ((e226.ANCHOR_G, "#1a6faf"),
                         (e226.ANCHOR_I, "#c0392b")):
            j = next(i for i, r in enumerate(probes) if r["fact"] == fact)
            ax.scatter([x1[j]], [x2[j]], marker="*", s=230, color=cl,
                       edgecolors="k", zorder=5,
                       label=fact.split("->")[-1])
        rr = rhos[t]
        ax.set_title(f"SET-LEVEL at +{t}: rotation angle w1 vs w2\n"
                     f"Spearman rho = {rr['obs']:+.3f} "
                     f"(null {rr['null_mean']:+.3f}±{rr['null_sd']:.3f}, "
                     f"p = {rr['p_ge_obs']:.4f})", fontsize=10)
        ax.set_xlabel("angle w1 (deg)")
        ax.set_ylabel("angle w2 (deg)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    ax = axes[2]
    for k, t in enumerate(sorted(crosswash)):
        th_b = [math.degrees(math.acos(max(-1.0, min(1.0, c))))
                for c in crosswash[t]]
        th_sum = [math.degrees(a1 + a2) for a1, a2 in
                  zip(rot_angles[t]["w1"], rot_angles[t]["w2"])]
        ax.scatter(th_sum, th_b, s=28, alpha=0.8,
                   color=["#7fb3d8", "#1a6faf"][k % 2], label=f"+{t}")
    allsum = [a1 + a2 for t in rot_angles
              for a1, a2 in zip(rot_angles[t]["w1"], rot_angles[t]["w2"])]
    lim = max(max(math.degrees(a) for a in allsum), 1.0) * 1.08
    ax.plot([0, lim], [0, lim], ":", color="0.55", lw=1,
            label="independence (θ1+θ2)")
    ax.set_xlabel("θ1+θ2 (deg) — both washes' rotations from t=0")
    ax.set_ylabel("θ_between (deg) — angle across washes")
    ax.set_title("the coherence envelope: θ_between vs θ1+θ2\n"
                 "(near the diagonal = independent rotations; near 0 = "
                 "coherent)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    wrap = textwrap.fill(verdict, 110)
    fig.suptitle("E239 — THE SET-LEVEL READ: does the POPULATION rotate "
                 "coherently across independent washes?\n" + wrap,
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    png = rd / f"{NAME}_setlevel.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


# ------------------------------------------------------------ main

def main():
    rd = run_dir(NAME)
    jp = rd / "journal.json"
    log(f"E239 — THE CROSS-WASH SUPPORT-STABILITY READ (smoke={SMOKE}) "
        f"-> {rd}")
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    log(f"bank dir: {BANK_DIR} (TEMP; repo is under OneDrive — banks "
        f"stay out of the synced tree)")

    metrics = {
        "experiment": "e239_support_stability",
        "phase": "eval-only on the two-wash 124M archive (CPU fp32 "
                 "batch-1 support geometry; cross-wash / rotation / "
                 "set-level reads)",
        "date": now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED_PREDICTION["registration"],
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("FQ6 / W037's direct test: are the 54 supports "
                     "stable AS VECTORS at matched deep states across "
                     "independent washes — or stable only as reads / as "
                     "a population?"),
        "builds_on": [
            "scratch/review_ideator_2026-10-04.md (FQ6, the Rank 1)",
            "W037 (R63 downgrade: 'behavior-INDEXED objects stable; the "
            "u0 sign-ray family a cloud' — THIS CELL is its direct test)",
            "T213 / e231 (the u0 cloud: cross-draw cos 0.318-0.370, the "
            "comparison object)",
            "T215 / e233 (the family-clause negative: supports do not "
            "share family directions — the band is a noise band)",
            "T149 / e182c + T183 / e182c2 (the two committed wash "
            "archives, read-only)",
            "e226 (THE machinery, module-imported VERBATIM: the 54 "
            "probes, fp64 chunked dots, the committed states, the t=0 "
            "bank convention; its +80 rotation records are the G_REPRO "
            "anchor)",
            "e208 (the edge-multiple census — u0-sign-ray margins, the "
            "corpus-anchored class this read contrasts)",
        ],
        "whats_new": [
            "the CROSS-WASH vector read: cos(s@w1(t), s@w2(t)) for all "
            "54 probes at {+50,+80} — the stability of supports AS "
            "VECTORS across independent washes (never measured; e226 "
            "only rotated each wash against its own t=0)",
            "the +50 supports and their rotations (never computed — "
            "e226's rotation read was +80 only)",
            "the SET-LEVEL read: Spearman rank correlation of the 54 "
            "probes' rotation amounts across washes (the "
            "POPULATION-stability branch) + permutation null",
            "the coherence envelope: theta_between vs theta1+theta2 per "
            "probe (independent vs coherent rotation, co-reported)",
            "G_REPRO: the t=0 Gram and the +80 rotations reproduce "
            "e226's committed records (the bank pipeline certified "
            "against the prior cell's published numbers)",
        ],
        "smoke": SMOKE,
    }

    def write_metrics(status):
        metrics["status"] = status
        metrics["timing"] = {"total_s": round(time.time() - T0, 1)}
        save_json(rd / "metrics.json", metrics)

    journal: dict = {}
    if jp.exists():
        try:
            journal = json.loads(jp.read_text(encoding="utf-8"))
            log(f"journal restored: {list(journal)}")
        except Exception as e:                             # noqa: BLE001
            log(f"journal unreadable ({e}); starting fresh")
            journal = {}

    def save_journal():
        jp.write_text(json.dumps(journal, indent=1, default=float),
                      encoding="utf-8")

    load_checks: list[dict] = [cpu_load_check("launch")]
    metrics["load_checks"] = load_checks

    u0_cloud = _load_u0_cloud()
    metrics["u0_cloud_reference"] = {
        **u0_cloud,
        "source": "runs/e231/metrics.json cells J1-J4 "
                  "second_stream_coreads.cos_u0_10902_vs_10914 "
                  "(T213's committed cloud)",
    }
    log(f"u0 cloud: mean {u0_cloud['mean']:.4f} "
        f"range [{u0_cloud['min']:.4f}, {u0_cloud['max']:.4f}]")

    # ------------------------------------------------ P0 the committed records
    for p in (E226_M, E231_M, e226.E182C_M, e226.E182C2_M, e226.E182C2_J,
              e226.E216_M):
        if not p.exists():
            log(f"FATAL: committed record missing: {p}")
            return 1
    e226m = json.loads(E226_M.read_text(encoding="utf-8"))
    p1m = json.loads(e226.E182C_M.read_text(encoding="utf-8"))
    c2m = json.loads(e226.E182C2_M.read_text(encoding="utf-8"))
    j2 = json.loads(e226.E182C2_J.read_text(encoding="utf-8"))
    e216 = json.loads(e226.E216_M.read_text(encoding="utf-8"))
    w1_rec = {s["step"]: s for s in p1m["states"]}
    w1_tmpl_rec = {s["step"]: s for s in c2m["part1_template"]["states"]}
    w2_rec = {s["step"]: s for s in j2["states"]}
    for s_ in STATES:
        assert s_ in w1_rec and s_ in w1_tmpl_rec and s_ in w2_rec, \
            f"committed record gap at +{s_}"
        assert W_ARCH["w1"][s_].exists() and W_ARCH["w2"][s_].exists(), \
            f"checkpoint gap at +{s_}"
    # e226's committed probe order (the row contract for the banks)
    e226_probes = e226m["probes"]
    e226_order = [r["fact"] for r in e226_probes]
    assert len(e226_order) == 54 and len(set(e226_order)) == 54
    # e226's committed +80 rotation records (G_REPRO-B anchor) — dicts
    # keyed BY fact, self_cos inside
    e226_rot80 = {w: {f: v["self_cos"]
                      for f, v in e226m["curves"]["rotation"][w].items()
                      if isinstance(v, dict) and "self_cos" in v}
                  for w in WASHES}
    e226_geo = e226m["support_geometry_t0"]
    log(f"records: e226 probe order (54) + +80 rotation records "
        f"({sum(len(v) for v in e226_rot80.values())} reads) + t=0 "
        f"geometry loaded")

    # ------------------------------------------------ P1 organism + envelope
    tok, net0, org_meta = e1.load_organism()
    metrics["organism"] = {**org_meta, "torch_threads": THREADS}
    G_SIZE = {"params": org_meta["params"], "ceiling": e1.SIZE_CEILING,
              "reason": e1.SIZE_REASON,
              "pass": bool(org_meta["params"] <= e1.SIZE_CEILING)}
    assert G_SIZE["pass"]
    G_ENV = {"device": "cpu", "threads": THREADS, "gpu_used": False,
             "pass": True,
             "note": "eval-only; 54 batch-1 probe backwards per bank "
                     "(t0 + 4 state banks); load checks per phase; two "
                     "other agents live"}
    metrics["gates"] = {"G_SIZE": G_SIZE, "G_ENV": G_ENV}
    metrics["size_gate"] = G_SIZE
    write_metrics("PARTIAL: records read; organism loaded")

    P_NAMES = [n for n, _ in net0.named_parameters()]
    N_PARAMS = sum(p.numel() for p in net0.parameters())
    assert N_PARAMS == org_meta["params"]
    log(f"param order frozen: {len(P_NAMES)} tensors, {N_PARAMS:,} coords")

    # ------------------------------------------------ P2 corpus (G_CORPUS)
    text = (common.REPO / "data" / "input.txt").read_text(encoding="utf-8")
    e182 = json.loads(e1.E182_METRICS.read_text(encoding="utf-8"))
    e_corp = e182["corpus"]
    e_str = e182["gates"]["G_STR"]
    e_banned = e_str["banned"]
    banned = sorted({s.lower() for rel in e1.POOLS
                     for s, _ in e1.POOLS[rel]}
                    | {a.lower() for rel in e1.POOLS
                       for _, a in e1.POOLS[rel]}
                    | set(e1.BANNED_EXTRA))
    assert banned == e_banned, "banned list diverged from e182's record"
    cand, _dropped = e1.build_candidates(tok)
    for r, b in zip(cand, e1.probe_battery(net0, cand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(cand)
    battery = [r for r in cand if r["kept"]]
    answer_ids = {r["fact"]: r["ans_id"] for r in battery}
    train_ids, _bank_xy, filtered, G_STR, _G_TOK, corpus_stats = \
        e1.build_wash_corpus(tok, text, banned, answer_ids)
    G_CORPUS = {
        "banned_list_identical": True,
        "tokens_after": [corpus_stats["tokens_after"],
                         e_corp["tokens_after"]],
        "train_tokens": [corpus_stats["train_tokens"],
                         e_corp["train_tokens"]],
        "format": "[rebuilt, e182_recorded]",
        "note": "the corpus is INHERITED FROZEN (e226's convention); it "
                "feeds the battery contamination scans only — no wash "
                "is run here",
    }
    G_CORPUS["pass"] = bool(
        G_STR["lines_total"] == e_str["lines_total"]
        and G_STR["lines_dropped"] == e_str["lines_dropped"]
        and corpus_stats["chars_after"] == e_corp["chars_after"]
        and corpus_stats["tokens_after"] == e_corp["tokens_after"]
        and corpus_stats["train_tokens"] == e_corp["train_tokens"])
    log(f"G_CORPUS: {'PASS' if G_CORPUS['pass'] else 'FAIL'} "
        f"({corpus_stats['tokens_after']} tokens)")
    assert G_CORPUS["pass"] or SMOKE
    metrics["gates"]["G_CORPUS"] = G_CORPUS
    write_metrics("PARTIAL: corpus certified")

    # --------------------------------- P3 the four batteries, VERBATIM (G_BATT)
    ccand, _cd = e1.build_control_candidates(
        tok, filtered.lower(), train_ids, e_banned,
        e1.CTRL_POOLS, e1.CTRL_TMPL)
    for r, b in zip(ccand, e1.probe_battery(net0, ccand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(ccand)
    cbattery = [r for r in ccand if r["kept"]]

    ncand, _nd = e1.build_control_candidates(
        tok, filtered.lower(), train_ids, e_banned,
        {"near": e1.NEAR_POOL}, {"near": e1.NEAR_TMPL})
    for r, b in zip(ncand, e1.probe_battery(net0, ncand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(ncand)
    nbattery = [r for r in ncand if r["kept"]]

    tcand, _td = e2.build_tmpl_candidates(tok, filtered.lower(), train_ids,
                                          e_banned)
    for r, b in zip(tcand, e1.probe_battery(net0, tcand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(tcand)
    tbattery = [r for r in tcand if r["kept"]]

    bats = {"fact": battery, "ctrl": cbattery,
            "near": nbattery, "tmpl": tbattery}

    def _probes(state_rec, batt):
        return {f: v["p"] for f, v in state_rec[batt]["probes"].items()}

    e216_tab = e216["residual"]["table"]
    G_BATT = {}
    for b, bl in bats.items():
        mine = [r["fact"] for r in bl]
        mine_p = {r["fact"]: r["p"] for r in bl}
        ref1 = _probes(w1_rec[0], b) if b != "tmpl" \
            else _probes(w1_tmpl_rec[0], "tmpl")
        ref2 = _probes(w2_rec[0], b)
        G_BATT[b] = {
            "n": len(bl),
            "set_equal_committed": bool(set(mine) == set(ref1)
                                        == set(ref2)),
            "max_dp_w1": max(abs(mine_p[f] - v) for f, v in ref1.items()),
            "max_dp_w2": max(abs(mine_p[f] - v) for f, v in ref2.items()),
        }
    G_BATT["tol_per_probe_dp"] = e226.TOL_T0_DP
    G_BATT["pass"] = bool(
        all(G_BATT[b]["set_equal_committed"]
            and max(G_BATT[b]["max_dp_w1"], G_BATT[b]["max_dp_w2"])
            <= e226.TOL_T0_DP for b in bats)) if not SMOKE else True
    metrics["gates"]["G_BATT"] = G_BATT
    log("G_BATT: " + ("PASS" if G_BATT["pass"] else "FAIL") + " | "
        + " | ".join(f"{b}: n {G_BATT[b]['n']} dp "
                     f"{max(G_BATT[b]['max_dp_w1'], G_BATT[b]['max_dp_w2']):.2e}"
                     for b in bats))
    if not G_BATT["pass"]:
        write_metrics("PARTIAL: G_BATT FAILED — halted before any "
                      "gradient compute")
        return 1

    # THE 54 (e226's committed order — the row contract for the banks)
    probes54: list[dict] = []
    for b in ("fact", "ctrl", "near", "tmpl"):
        for r in bats[b]:
            row = {rr["fact"]: rr for rr in e216_tab}[r["fact"]]
            probes54.append({**r, "battery": b, "family6": row["family6"]})
    assert len(probes54) == 54
    order_ok = [r["fact"] for r in probes54] == e226_order
    G_ORDER = {
        "rule": "bank row i == e226's committed probe i (the row "
                "contract that makes G_REPRO row-aligned)",
        "equal_e226": order_ok,
        "pass": order_ok,
    }
    metrics["gates"]["G_ORDER"] = G_ORDER
    assert G_ORDER["pass"] or SMOKE
    if SMOKE:
        keep = {e226.ANCHOR_G, e226.ANCHOR_I,
                "The gaming console made by Microsoft->Xbox",
                "The web browser made by Google->Chrome",
                "The tablet made by Apple->iPad",
                "The music store made by Apple->iTunes",
                "The game console made by Sony->PlayStation",
                "The social network founded by Mark Zuckerberg->Facebook",
                "France->Paris", "Massachusetts->Boston",
                "Boston->Massachusetts"}
        probes54 = [r for r in probes54 if r["fact"] in keep]
        log(f"SMOKE: probe subset n={len(probes54)}")
    n_p = len(probes54)
    metrics["probes"] = [{"i": i, "fact": r["fact"], "battery": r["battery"],
                          "family6": r["family6"], "p0": r["p"]}
                         for i, r in enumerate(probes54)]
    write_metrics("PARTIAL: batteries rebuilt + order certified")

    # ------------------------------------------------ P5 THE t=0 BANK
    load_checks.append(cpu_load_check("bank t0"))
    if not ram_floor_ok(12.0, "bank passes (chunk transients)"):
        write_metrics("PARTIAL: RAM floor hit — HALT")
        return 1

    def compute_bank(tag: str, net) -> dict:
        """54 batch-1 supports -> fp16 bank file; returns per-probe p."""
        mm = open_bank_rw(tag, n_p, N_PARAMS)
        p_at = {}
        for i, pr in enumerate(probes54):
            s_i, p0 = e226.probe_support(net, pr)
            mm[i, :] = (s_i * e226.CACHE_SCALE).to(torch.float16).numpy()
            p_at[pr["fact"]] = p0
            if i % 9 == 0:
                mm.flush()
                log(f"  [{tag}] s[{i:2d}/{n_p}] {pr['fact'][:44]:<44} "
                    f"p {p0:.4f}")
        mm.flush()
        del mm
        gc.collect()
        return p_at

    if journal.get("banks", {}).get("t0", False) and \
            bank_path("t0").exists():
        log("bank t0: on disk (journal) — skipping recompute")
        p_t0 = journal["p_at_state"]["t0"]
    else:
        p_t0 = compute_bank("t0", net0)
        journal.setdefault("banks", {})["t0"] = True
        journal.setdefault("p_at_state", {})["t0"] = p_t0
        save_journal()
        write_metrics("PARTIAL: t=0 bank computed")

    # G_SUPPORT: the determinism floor (e226's convention — probe-0
    # recompute, fresh fp32 vs its cached fp16 row)
    t0_view = bank_view(open_bank_r("t0", n_p, N_PARAMS))
    s_rep, _ = e226.probe_support(net0, probes54[0])
    n0 = e226.cache_row_norms(t0_view[0:1]).tolist()[0]
    det_t0 = e226.dot_vec_rows(s_rep, t0_view, [0])[0] / n0
    G_SUPPORT = {
        "rule": "probe-0 recompute self-cos > 0.999 (e226's determinism "
                "floor convention; batch-1 supports are deterministic "
                "same-session — the floor carries as the instrument "
                "floor on every cos)",
        "determinism_selfcos_probe0_t0": det_t0,
        "e226_floor_reference": abs(
            1.0 - e226m["gates"]["G_SUPPORT"]["determinism_selfcos_probe0"]),
        "pass": bool(det_t0 > 0.999) if not SMOKE else True,
    }
    metrics["gates"]["G_SUPPORT"] = G_SUPPORT
    log(f"G_SUPPORT: {'PASS' if G_SUPPORT['pass'] else 'FAIL'} "
        f"(probe-0 t0 self-cos {det_t0:.8f})")

    # the t=0 Gram + geometry (G_REPRO part A vs e226's committed t=0)
    if "geo_t0" not in journal:
        G = e226.gram_cache(t0_view).tolist()
        rn = e226.cache_row_norms(t0_view).tolist()
        journal["norms"] = {"t0": rn}
        cos_ij = [[G[i][j] / (rn[i] * rn[j]) for j in range(n_p)]
                  for i in range(n_p)]
        gi = next(i for i, r in enumerate(probes54)
                  if r["fact"] == e226.ANCHOR_G)
        ii = next(i for i, r in enumerate(probes54)
                  if r["fact"] == e226.ANCHOR_I)
        prod_all = [i for i, r in enumerate(probes54)
                    if r["family6"] == e226.BAND_FAMILY]
        fam_pair_cos = sorted(cos_ij[a][b] for x, a in enumerate(prod_all)
                              for b in prod_all[x + 1:])
        fam_summary = {}
        for fam in sorted({r["family6"] for r in probes54}):
            idxs = [i for i, r in enumerate(probes54)
                    if r["family6"] == fam]
            pcs = sorted(cos_ij[a][b] for x, a in enumerate(idxs)
                         for b in idxs[x + 1:])
            if pcs:
                fam_summary[fam] = {"n": len(idxs),
                                    "within_mean_cos": sum(pcs) / len(pcs)}
        journal["geo_t0"] = {
            "gmail_iphone_cos": cos_ij[gi][ii],
            "product_pairwise_mean": (sum(fam_pair_cos)
                                      / len(fam_pair_cos)),
            "product_pairwise_sd": st.stdev(fam_pair_cos),
            "family_within_summary": fam_summary,
        }
        save_journal()
    metrics["support_geometry_t0_recomputed"] = journal["geo_t0"]
    d_geo = {
        "gmail_iphone_cos": abs(journal["geo_t0"]["gmail_iphone_cos"]
                                - e226_geo["gmail_iphone_cos"]),
        "product_pairwise_mean": abs(
            journal["geo_t0"]["product_pairwise_mean"]
            - e226_geo["product_family_pairwise"]["mean"]),
        "product_pairwise_sd": abs(
            journal["geo_t0"]["product_pairwise_sd"]
            - e226_geo["product_family_pairwise"]["sd"]),
    }
    log(f"t=0 geometry vs e226: d(cosGI) {d_geo['gmail_iphone_cos']:.2e} "
        f"d(fam mean) {d_geo['product_pairwise_mean']:.2e}")
    write_metrics("PARTIAL: t=0 bank + geometry + determinism")

    # ------------------------------------------------ P6 the state banks
    state_gates: dict = journal.get("state_gates", {})
    for w in WASHES:
        for s_ in STATES:
            tag = f"{w}s{s_}"
            if journal.get("banks", {}).get(tag, False) and \
                    bank_path(tag).exists():
                log(f"bank {tag}: on disk (journal) — skipping")
                continue
            load_checks.append(cpu_load_check(f"bank {tag}"))
            if not ram_floor_ok(12.0, f"bank {tag}"):
                write_metrics("PARTIAL: RAM floor hit — HALT (resumable)")
                return 1
            sd = torch.load(W_ARCH[w][s_], map_location=CPU,
                            weights_only=False)["model"]
            net = copy.deepcopy(net0)
            net.load_state_dict(sd)
            del sd
            # (a) re-probe vs committed (e228's convention, tol 0.005)
            dps = {}
            for b, bl in bats.items():
                pp = {r["fact"]: e226.probe_p(net, r) for r in bl}
                rec = (w1_rec[s_] if b != "tmpl" else w1_tmpl_rec[s_]) \
                    if w == "w1" else w2_rec[s_]
                ref = _probes(rec, b)
                dps[b] = max(abs(pp[f] - v) for f, v in ref.items())
            # (b) the bank
            p_at = compute_bank(tag, net)
            # (c) determinism co-check on the first state bank's probe 0
            det_state = None
            if tag == ("w1s50" if not SMOKE else "w1s80"):
                mm_v = bank_view(open_bank_r(tag, n_p, N_PARAMS))
                s_rep2, _ = e226.probe_support(net, probes54[0])
                nr = e226.cache_row_norms(mm_v[0:1]).tolist()[0]
                det_state = e226.dot_vec_rows(s_rep2, mm_v, [0])[0] / nr
                G_SUPPORT["determinism_selfcos_probe0_state"] = det_state
                del mm_v
            state_gates[tag] = {
                "reprobe_max_dp": dps,
                "checkpoint": str(W_ARCH[w][s_]),
                "p_at_state": p_at,
                "det_state": det_state,
            }
            journal.setdefault("banks", {})[tag] = True
            journal["state_gates"] = state_gates
            save_journal()
            log(f"  {tag}: bank done | re-probe dp "
                f"{max(dps.values()):.2e}"
                + (f" | det {det_state:.8f}" if det_state else ""))
            del net
            write_metrics(f"PARTIAL: bank {tag} done")
    G_STATES = {
        "tol_reprobe_dp": e226.TOL_STATE_DP,
        "per_state": {k: {"reprobe_max_dp": v["reprobe_max_dp"],
                          "checkpoint": v["checkpoint"]}
                      for k, v in state_gates.items()},
        "max_dp": max(max(v["reprobe_max_dp"].values())
                      for v in state_gates.values()) if state_gates
        else None,
        "pass": bool(state_gates and all(
            max(v["reprobe_max_dp"].values()) <= e226.TOL_STATE_DP
            for v in state_gates.values())) if not SMOKE else True,
        "note": "the archived states reproduce the committed per-probe "
                "records when re-probed through the VERBATIM batteries "
                "(e228's re-probe convention)",
    }
    metrics["gates"]["G_STATES"] = G_STATES
    log(f"G_STATES: {'PASS' if G_STATES['pass'] else 'FAIL'} "
        f"(max re-probe dp {G_STATES['max_dp']:.2e})")
    write_metrics("PARTIAL: all banks done")

    # ------------------------------------------------ P7 the dots
    load_checks.append(cpu_load_check("dots"))
    views = {tag: bank_view(open_bank_r(tag, n_p, N_PARAMS))
             for tag in BANK_TAGS}
    for tag in BANK_TAGS:
        if tag not in journal.get("norms", {}):
            journal["norms"][tag] = e226.cache_row_norms(
                views[tag]).tolist()
            save_journal()
    norms = journal["norms"]

    rot: dict = journal.get("rotation", {})
    if not rot:
        for s_ in STATES:
            rot[str(s_)] = {}
            for w in WASHES:
                d = diag_dots(views[f"{w}s{s_}"], views["t0"])
                rot[str(s_)][w] = [d[i] / (norms[f"{w}s{s_}"][i]
                                           * norms["t0"][i])
                                   for i in range(n_p)]
        journal["rotation"] = rot
        save_journal()
    crosswash: dict = journal.get("crosswash", {})
    if not crosswash:
        for s_ in STATES:
            d = diag_dots(views[f"w1s{s_}"], views[f"w2s{s_}"])
            crosswash[str(s_)] = [d[i] / (norms[f"w1s{s_}"][i]
                                          * norms[f"w2s{s_}"][i])
                                  for i in range(n_p)]
        journal["crosswash"] = crosswash
        save_journal()
    write_metrics("PARTIAL: all dots done")

    # G_REPRO part B: the +80 rotations vs e226's committed records
    repro_b: dict = {}
    if not SMOKE and "80" in rot:
        for w in WASHES:
            devs = [abs(rot["80"][w][i] - e226_rot80[w][r["fact"]])
                    for i, r in enumerate(probes54)]
            repro_b[w] = {"max_abs_dev": max(devs),
                          "mean_abs_dev": st.mean(devs)}
        repro_b["tol"] = 0.005
        repro_b["pass"] = bool(all(repro_b[w]["max_abs_dev"] <= 0.005
                                   for w in WASHES))
    G_REPRO = {
        "rule": "the recomputed banks reproduce e226's committed "
                "numbers: (A) t=0 Gram-derived geometry; (B) +80 "
                "rotation self-cos, all 54 x both washes; tol 5e-3 "
                "(generous vs the ~5e-4 disclosed fp16 floor — it is "
                "provenance, not adjudication)",
        "part_A_t0_geometry": {
            "devs": d_geo,
            "tol": 0.005,
            "pass": bool(all(v <= 0.005 for v in d_geo.values())),
        },
        "part_B_rot80": repro_b,
        "pass": bool(all(v <= 0.005 for v in d_geo.values())
                     and (SMOKE or repro_b.get("pass", False))),
    }
    metrics["gates"]["G_REPRO"] = G_REPRO
    log(f"G_REPRO: {'PASS' if G_REPRO['pass'] else 'FAIL'} "
        f"(t0 dGeo max {max(d_geo.values()):.2e}"
        + (f"; rot80 max dev "
           f"{max(repro_b[w]['max_abs_dev'] for w in WASHES):.2e}"
           if repro_b else "") + ")")

    # ------------------------------------------------ P8 the reads
    cw_summary = {}
    for s_ in STATES:
        vals = crosswash[str(s_)]
        svals = sorted(vals)
        per_bat = {b: st.median([v for v, r in zip(vals, probes54)
                                 if r["battery"] == b])
                   for b in ("fact", "ctrl", "near", "tmpl")}
        cw_summary[str(s_)] = {
            "median": st.median(vals), "mean": st.mean(vals),
            "min": min(vals), "max": max(vals),
            "q25": svals[n_p // 4], "q75": svals[(3 * n_p) // 4],
            "n_below_u0_max": sum(1 for v in vals if v < u0_cloud["max"]),
            "n_above_stable_bar": sum(1 for v in vals if v >= 0.7),
            "per_battery_median": per_bat,
        }
        log(f"CROSS-WASH +{s_}: median {st.median(vals):.4f} "
            f"[{min(vals):+.4f}, {max(vals):+.4f}] | per-battery "
            + " ".join(f"{b}:{per_bat[b]:.3f}" for b in per_bat))

    rot_summary = {}
    for s_ in STATES:
        for w in WASHES:
            vals = rot[str(s_)][w]
            rot_summary[f"{w}@+{s_}"] = {
                "median": st.median(vals), "mean": st.mean(vals),
                "min": min(vals), "max": max(vals),
            }
        log(f"ROTATION +{s_}: w1 median "
            f"{rot_summary[f'w1@+{s_}']['median']:.4f} | w2 median "
            f"{rot_summary[f'w2@+{s_}']['median']:.4f}")

    # set-level: rotation amounts (angles) + Spearman + permutation
    rot_angles: dict = {}
    rhos: dict = {}
    for s_ in STATES:
        rot_angles[str(s_)] = {
            w: [math.acos(max(-1.0, min(1.0, c)))
                for c in rot[str(s_)][w]] for w in WASHES}
        rho = perm_null_rho(rot_angles[str(s_)]["w1"],
                            rot_angles[str(s_)]["w2"])
        rhos[str(s_)] = rho
        log(f"SET-LEVEL +{s_}: Spearman rho {rho['obs']:+.4f} "
            f"(null {rho['null_mean']:+.4f}±{rho['null_sd']:.4f}, "
            f"p {rho['p_ge_obs']:.4f})")

    # the coherence envelope (co-read: independent vs coherent rotation)
    coherence = {}
    for s_ in STATES:
        th_b = [math.acos(max(-1.0, min(1.0, c)))
                for c in crosswash[str(s_)]]
        th_s = [a1 + a2 for a1, a2 in zip(rot_angles[str(s_)]["w1"],
                                          rot_angles[str(s_)]["w2"])]
        ratios = sorted(b / a if a > 0 else 0.0
                        for b, a in zip(th_b, th_s))
        coherence[str(s_)] = {
            "median_ratio_thetaB_over_theta1plus2": st.median(ratios),
            "mean_ratio": st.mean(ratios),
            "q25_ratio": ratios[n_p // 4],
            "q75_ratio": ratios[(3 * n_p) // 4],
            "note": "ratio ~1 = independent rotations (theta_between "
                    "fills the theta1+theta2 envelope); ~0 = the two "
                    "washes rotate supports the SAME way",
        }
        log(f"COHERENCE +{s_}: median theta_B/(theta1+theta2) "
            f"{st.median(ratios):.4f}")

    # the random null (analytic + empirical)
    g = torch.Generator().manual_seed(2390)
    emp = []
    for _ in range(16):
        a = torch.randn(N_PARAMS, generator=g)
        b = torch.randn(N_PARAMS, generator=g)
        emp.append(float((a.dot(b) / (a.norm() * b.norm())).item()))
    random_null = {
        "analytic_sd": 1.0 / math.sqrt(N_PARAMS),
        "empirical_16": {"mean": st.mean(emp),
                         "min": min(emp), "max": max(emp)},
    }

    # the anchors, called out (never adjudicated)
    anchors = {}
    band_idx = [i for i, r in enumerate(probes54)
                if r["family6"] == e226.BAND_FAMILY
                and r["fact"] not in (e226.ANCHOR_G, e226.ANCHOR_I)]
    for name, fact in (("Gmail", e226.ANCHOR_G),
                       ("iPhone", e226.ANCHOR_I)):
        j = next(i for i, r in enumerate(probes54) if r["fact"] == fact)
        anchors[name] = {
            "crosswash": {str(s_): crosswash[str(s_)][j] for s_ in STATES},
            "rotation": {str(s_): {w: rot[str(s_)][w][j]
                                   for w in WASHES} for s_ in STATES},
            "crosswash_z_vs_band": {
                str(s_): (crosswash[str(s_)][j]
                          - st.mean([crosswash[str(s_)][k]
                                     for k in band_idx]))
                / st.stdev([crosswash[str(s_)][k] for k in band_idx])
                for s_ in STATES},
        }

    # ------------------------------------------------ P9 adjudication
    med = {str(s_): cw_summary[str(s_)]["median"] for s_ in STATES}
    stable = all(med[str(s_)] >= 0.7 for s_ in STATES)
    cloud = all(0.30 <= med[str(s_)] <= 0.40 for s_ in STATES)
    verdict = ("STABLE-AS-VECTORS" if stable else
               "CLOUD-LEVEL" if cloud else "MIXED")
    third_branch = {
        "declared_under": "CLOUD-LEVEL only (per the registered bar "
                          "text); co-reported always",
        "rho_bar": 0.6,
        "rho": {str(s_): rhos[str(s_)]["obs"] for s_ in STATES},
        "rho_p_perm": {str(s_): rhos[str(s_)]["p_ge_obs"] for s_ in STATES},
        "population_stable_if_cloud": any(
            rhos[str(s_)]["obs"] >= 0.6 for s_ in STATES),
    }
    if SMOKE:
        verdict += " (SMOKE — not adjudicated)"

    metrics["reads"] = {
        "crosswash": {
            "values": {str(s_): crosswash[str(s_)] for s_ in STATES},
            "summary": cw_summary,
            "medians": med,
        },
        "rotation": {
            "values": {str(s_): rot[str(s_)] for s_ in STATES},
            "summary": rot_summary,
        },
        "set_level": {
            "rho_detail": rhos,
            "third_branch": third_branch,
        },
        "coherence_envelope": coherence,
        "random_null": random_null,
        "anchors_called_out": anchors,
        "probes_facts": [r["fact"] for r in probes54],
    }
    metrics["adjudication"] = {
        "verdict": verdict,
        "bars_verbatim": REGISTERED_PREDICTION["bars_verbatim"],
        "rule_applied": {
            "STABLE-AS-VECTORS": "median(+50)>=0.7 AND median(+80)>=0.7",
            "CLOUD-LEVEL": "median(+50) in [0.30,0.40] AND median(+80) "
                           "in [0.30,0.40]",
            "MIXED": "neither",
            "medians": med,
        },
        "gated_on": "G_ENV/G_SIZE/G_CORPUS/G_BATT/G_ORDER/G_STATES/"
                    "G_SUPPORT/G_REPRO",
        "all_gates_pass": bool(all(
            v.get("pass", True) for v in metrics["gates"].values()
            if isinstance(v, dict))),
    }
    write_metrics("PARTIAL: reads computed; plots next")

    # ------------------------------------------------ P10 plots
    png1 = make_plot_crosswash(
        rd, crosswash,
        [{"battery": r["battery"], "family6": r["family6"],
          "fact": r["fact"]} for r in probes54],
        u0_cloud,
        f"{verdict} — medians " +
        ", ".join(f"+{t}: {med[str(t)]:.4f}" for t in STATES)
        + f" | u0 cloud mean {u0_cloud['mean']:.4f}", med)
    png2 = make_plot_setlevel(
        rd, rot_angles, crosswash,
        [{"battery": r["battery"], "fact": r["fact"]} for r in probes54],
        rhos, verdict)
    metrics["plot_outputs"] = [str(png1), str(png2)]

    metrics["honesty_reflex"] = {
        "the_a_priori_bound": "e226's committed +80 rotations (medians "
                              "~0.93) ARITHMETICALLY bound the "
                              "cross-wash angle by theta_between <= "
                              "theta1+theta2 (cos >= cos(theta1+theta2) "
                              "~= 0.73 at e226's angles) — a STABLE "
                              "firing must be read against that "
                              "envelope: the informative content is the "
                              "coherence ratio (independent vs same-way "
                              "rotation), co-reported per state",
        "batch_noise": "supports are SINGLE-CONTEXT batch-1 gradients "
                       "(deterministic same-session; e226's self-cos "
                       "~1.0); the instrument floor on any cos is the "
                       "fp16 bank rounding (~5e-4, e226's disclosed "
                       "floor) — every distribution spread is orders "
                       "above it",
        "t0_recompute": "the t=0 bank is RECOMPUTED, not e226's "
                        "in-RAM cache; G_REPRO-A anchors the geometry "
                        "to e226's committed t=0 records",
        "e208_distinction": "e208's memory-vs-noise margins are "
                            "u0-SIGN-RAY objects (the corpus-anchored "
                            "class W037 clouds; e231's 0.318-0.370) — "
                            "this cell reads the per-context gradient "
                            "class; the two stabilities must not be "
                            "conflated",
        "t215_distinction": "the family-clause negative: the family's "
                            "other members' supports do NOT share the "
                            "anchor's direction (e226 t=0 within-family "
                            "cos ~0.004) — the product band is a noise "
                            "band of per-probe objects; a set-level rho "
                            "firing is therefore about ROTATION AMOUNTS, "
                            "not shared directions",
        "nothing_guaranteed": "the +50 cross-wash read has no prior "
                              "instrument anywhere; MIXED is a real "
                              "outcome (the registered text carries it)",
        "provenance": "w1 = CPU-fp32 replay of e182's wash; w2 weights "
                      "computed GPU fp32 in e182c2's cell; every "
                      "gradient here is CPU fp32 on the archived "
                      "states; every loaded state re-probe-certified "
                      "(tol 0.005)",
    }
    metrics["compute"] = {
        "wall_s": round(time.time() - T0, 1),
        "device": "CPU only", "threads": THREADS,
        "backwards": f"{n_p} probe supports x {len(BANK_TAGS)} banks "
                     f"(t0 + 4 states)",
        "load_checks": len(load_checks),
    }
    metrics["trims"] = trims
    metrics["deviations"] = deviations
    status = ("SMOKE DONE" if SMOKE else
              ("DONE" if metrics["adjudication"]["all_gates_pass"]
               else "DONE (gate failures disclosed — see gates)"))
    write_metrics(status)
    log(f"VERDICT: {verdict} — medians " +
        ", ".join(f"+{s_}: {med[str(s_)]:.4f}" for s_ in STATES)
        + " | rho " +
        ", ".join(f"+{s_}: {rhos[str(s_)]['obs']:+.3f}" for s_ in STATES))

    # banks deleted on DONE (disclosed; resume-safe until then)
    if status.startswith("DONE") and not status.startswith("SMOKE"):
        del views, t0_view
        gc.collect()
        for tag in BANK_TAGS:
            try:
                bank_path(tag).unlink(missing_ok=True)
            except OSError as e:                              # noqa: BLE001
                log(f"bank cleanup skip {tag}: {e}")
        try:
            BANK_DIR.rmdir()
        except OSError:                                       # noqa: BLE001
            pass
        journal["banks_deleted_on_done"] = True
        save_journal()
        log("banks deleted on DONE (disclosed in deviations)")

    log(f"outputs: {rd / 'metrics.json'}, {png1}, {png2}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
