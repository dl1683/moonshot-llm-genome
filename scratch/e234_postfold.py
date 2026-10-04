"""E234 POST-FOLD (desk-only; zero wash compute):
(1) relabel the first-session replay_cert entries '2'/'10'/'50' — they were
    computed as |W_80(live) - W_s(archived)| (wash DISPLACEMENT, the
    final-net-vs-mid-checkpoint bug fixed in lab commit 2dca1c7); the '80'
    entries are the true end-state replay drifts and stand;
(2) fold the post-run DESK READS into metrics (raw-alignment join (a),
    the anchors' raw-exposure separation, battery/family context);
(3) regenerate e234_anchor.png with a neutral caption (the original
    caption overclaimed 'the dying anchor rides the wind' — true in w1
    only; plot regenerated VERBATIM from the frozen journal).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lab"))
import numpy as np                                       # noqa: E402
from scipy.stats import spearmanr                        # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RD = REPO / "runs" / "e234"

j = json.loads((RD / "journal.json").read_text(encoding="utf-8"))
m = json.loads((RD / "metrics.json").read_text(encoding="utf-8"))

# ---------------- (1) cert relabel
NOTE = ("MISLABEL RELABEL (post-fold): computed as |W_80(live replay) - "
        "W_s(archived)| — wash DISPLACEMENT from s to +80, NOT replay "
        "drift (the final-net-vs-mid-checkpoint comparison bug, fixed in "
        "lab/e234 commit 2dca1c7: drift is now measured inline at each "
        "checkpoint's own step). The true end-state replay drift is the "
        "'80' entry; intermediate checkpoints are certified by G_TRAJ's "
        "probe-level gate (max dp 9.41e-03) + the bit-exact draw states.")
for w in ("w1", "w2", "w3"):
    cert = j[w]["replay_cert"]
    disp = {k: cert.pop(k) for k in ("2", "10", "50") if k in cert}
    cert["displacement_mislabeled_as_drift"] = {**disp, "_note": NOTE}

# ---------------- (2) desk reads
rows = j["join_rows"]
G = "The email service made by Google->Gmail"
I = "The phone made by Apple->iPhone"
desk = {
    "raw_alignment_join_a": {
        "definition": "Spearman(cumulative RAW alignment sum_t gnorm_t*"
                      "|cos(g_t, s_i)| = full_cum, belief decline) — the "
                      "UNPROJECTED control for join (a): bounds the "
                      "'projection ate the signal' alternative",
        "pooled": float(spearmanr([r["full_cum"] for r in rows],
                                  [r["belief_decl"] for r in rows]).statistic),
        "per_wash": {w: float(spearmanr(
            [r["full_cum"] for r in rows if r["wash"] == w],
            [r["belief_decl"] for r in rows if r["wash"] == w]).statistic)
            for w in ("w1", "w2", "w3")},
        "read": "the raw join is as flat as the projected join (0.05 vs "
                "-0.01 pooled) — NEITHER is NOT a projection artifact",
    },
    "anchors_raw_exposure": {
        r["wash"] + " " + ("iPhone" if r["fact"] == I else "Gmail"): {
            "full_cum": round(r["full_cum"], 2),
            "fric_cum": round(r["fric_cum"], 2),
            "wind_cum": round(r["wind_cum"], 4),
            "belief_decl": round(r["belief_decl"], 3),
            "margin_decl": round(r["margin_decl"], 3)}
        for r in rows if r["fact"] in (G, I)},
    "anchors_read": ("iPhone's RAW cumulative exposure is 8-10x Gmail's on "
                     "every wash (1.5-2.5 vs 0.2-0.3) — e226's seat, "
                     "expressed cumulatively and cross-wash-stable — but "
                     "it does not generalize: the battery between the "
                     "anchors is uncorrelated (raw rho 0.05). The seat is "
                     "an anchors-scale tale of extremes, not a battery "
                     "law. Gmail's margin GROWS through the wash "
                     "(margin_decl -0.40..-0.52): the holder's commitment "
                     "thickens."),
    "battery_belief_decl_means": {
        b: float(np.mean([r["belief_decl"] for r in rows
                          if r["battery"] == b]))
        for b in ("fact", "ctrl", "near", "tmpl")},
    "battery_read": ("the nearrel family dies HARDEST (mean belief decline "
                     "0.71) while carrying the LOWEST median wind-"
                     "alignment (0.029 vs product 0.042 — prediction (a) "
                     "inverted, ratio 0.70): the dying family is the "
                     "least wind-reached; its deaths are not span-directed"),
    "family_wind_cum_medians": {
        f: float(np.median([r["wind_cum"] for r in rows
                            if r["family6"] == f]))
        for f in sorted({r["family6"] for r in rows})},
}
m["desk_reads"] = desk

m["deviations"].append(
    "POST-FOLD RELABELS (desk-only, zero wash compute): (1) the first "
    "session's replay_cert '2'/'10'/'50' entries were wash displacement "
    "(final net vs mid-run checkpoints — the comparison bug fixed in "
    "commit 2dca1c7), relabeled in journal+metrics; the '80' entries are "
    "the true end-state drifts (w1 0.0 bit-exact; w2 1.5e-3; w3 1.5e-3 "
    "vs GPU-origin archives — the disclosed TEXTURE tier); (2) the "
    "anchor figure's caption was regenerated neutral (the original "
    "overclaimed 'the dying anchor rides the wind', which the w1 panel "
    "alone supports); (3) desk_reads folded (the raw-alignment control "
    "join, the anchors' raw-exposure separation, the battery context).")

# ---------------- (3) anchor plot regeneration (caption fix)
import torch  # noqa: E402  (matplotlib import chain in e226 module)
import e234_wind_friction as e234                           # noqa: E402
P_IDX = {p["fact"]: i for i, p in enumerate(m["probes"])}
dec = {}
for w in ("w1", "w2", "w3"):
    gnorms = np.array(j[w]["gnorms"], dtype=np.float64)
    SCALE2 = float(e234.CACHE_SCALE ** 2)
    Gs = np.array(j[f"gram_scaled_{w}"], dtype=np.float64)
    Ds = np.array(j[f"dmat_scaled_{w}"], dtype=np.float64)
    dec[w] = e234.decompose(Gs, gnorms, Ds,
                            [float(e234.CACHE_SCALE)] * len(m["probes"]),
                            e234.HALF)
probes54 = m["probes"]

import matplotlib                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
band = [i for i, r in enumerate(probes54)
        if r["family6"] == "product"
        and r["fact"] not in (e234.ANCHOR_G, e234.ANCHOR_I)]
for ax, w in zip(axes[:3], ("w1", "w2", "w3")):
    cw = dec[w]["cos_wind_A"]
    xs = list(range(e234.HALF + 1, e234.HALF + 1 + cw.shape[0]))
    gi, ii = P_IDX[e234.ANCHOR_G], P_IDX[e234.ANCHOR_I]
    bm = np.mean(cw[:, band], axis=1)
    bs = np.std(cw[:, band], axis=1, ddof=1)
    ax.fill_between(xs, bm - 2 * bs, bm + 2 * bs, color="0.82",
                    label="product family ±2σ (n=5)")
    ax.plot(xs, cw[:, gi], "o-", color="#1a6faf", ms=4, label="Gmail")
    ax.plot(xs, cw[:, ii], "s--", color="#c0392b", ms=4, label="iPhone")
    ax.set_xlabel("wash step (decomposed half)")
    ax.set_ylabel("|cos(P_span g_t, s_probe)|")
    ax.set_title(f"{w}: the anchors' wind-alignment per step", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.25)
fig.suptitle("E234 — THE ANCHOR OVERLAY (span-projected): iPhone rides "
             "the wind in W1 ONLY; in w2/w3 both anchors sit in the "
             "family band — e226's raw-gradient seat does not live in "
             "the 7%-coverage span; the raw-exposure separation "
             "(8-10x, all washes) is in metrics.desk_reads", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(RD / "e234_anchor.png", dpi=130)
plt.close(fig)

(RD / "journal.json").write_text(
    json.dumps(j, indent=1, default=float), encoding="utf-8")
m["status"] = ("DONE (post-fold relabels + neutral anchor caption + desk "
               "reads folded — see deviations[-1]; no recompute)")
(RD / "metrics.json").write_text(
    json.dumps(m, indent=2, default=float), encoding="utf-8")
print("postfold done:", m["status"])
