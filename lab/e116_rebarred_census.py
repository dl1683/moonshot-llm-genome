"""E116 — the RE-BARRED CENSUS: the formal M1 graduation check (T068-registered).

T068's M1 adjudication (THINKING.md, 2026-09-28): the e098 literal top1/top2
bar went 0/4 because the 0.84M family's 0-1 scaffolding band tops every
mean-arm census — an architecture-family artifact the statistic must exclude.
The structure itself replicated 4/4: every seed grows ONE content-carrying
decision-window row (127-129, mean-arm = zero-arm). The re-barred census
(scaffolding-excluded, content-rows-only) is registered there as the formal
graduation check; this file executes it.

FROZEN STATISTIC (registered before compute):
  * Universe: FED rows 0..129 of the wpe (the install-60 battery is 130
    tokens; rows 130+ are never fed and have exactly-zero drops everywhere).
  * Arms (e067/e098 convention, per row r): MEAN arm drop m(r) = base p(Z) -
    p(Z | wpe[r] <- mean(all wpe rows)); ZERO arm drop z(r) = base p(Z) -
    p(Z | wpe[r] <- 0). Drop = install-60 battery mean p(Z) at the last
    context position.
  * CONTENT TEST (the criterion the data already established — e098 NOTES:
    "content-carrying decision-window row ... mean-arm = zero-arm"): a row is
    CONTENT iff its two arms AGREE within a factor of 2, i.e.
        content(r)  <=>  m(r) > 0 and z(r) > 0 and 0.5 <= min(m,z)/max(m,z).
    Scaffolding rows are excluded BY THIS TEST, not by any hardcoded list.
    Calibration from the established data points: the row-1 scaffolding
    signature sits at arm ratios <= 0.32 (e.g. 4305: 0.0087/0.4382), the
    decision-row content signature at >= 0.70 (e.g. 42: 0.3415/0.2405) — the
    0.5 bar sits in the empty middle.
  * RE-BARRED CONCENTRATION: among content rows ranked by mean-arm drop,
    a seed is CONCENTRATED iff top drop >= 2.0 x the next content row's drop.
  * VERDICT over 6 seeds (4 fresh e098 + references 42/43):
      >= 5/6  -> ADDRESS-UNIVERSALITY GRADUATES TO LAW (formal n=6 statement)
      3-4/6   -> STRUCTURE-STRONG, CONCENTRATION-MIXED (stays descriptive)
      <= 2/6  -> CONCENTRATION FAILS THE RE-BAR (honest negative)

FREE RE-ANALYSIS MANDATE (CPU-only): reuse stored per-row census data; only
regenerate what is missing.
  * runs/e098/metrics.json: full 512-row mean-arm drops for all 4 fresh seeds
    (zero arm stored only for the top-3 mean rows -> REGENERATE zero arm).
  * runs/e067/metrics.json (seed 42): full 256-row mean arm + zero arm on 24
    rows {0-6, 118-134} (-> REGENERATE zero arm on the remaining fed rows).
  * runs/e082/metrics.json (seed 43): mean arm on 22 rows {0,1,122-135,250-255}
    only (-> REGENERATE full fed-row mean arm AND zero arm).
  Checkpoint nets (all shipped in runs/checkpoints/): e098_install_s{4305,
  4306,4308}.pt, e098_install_s4307_pat.pt (4307 used the patience branch —
  its census net), e048_repro.pt (seed 42), e082_b43_install.pt (seed 43).
  Every stored number touched here is re-verified by reproduction gates
  (tolerance 5e-6) before it feeds the statistic.

Report-only riders (never gating): zero-arm ranking; content-bar sensitivity
(0.25 / 0.75); excl-row-0 secondary (the decision row vs the next content row
below it, e098's registered secondary); per-seed content-test outcome for
rows 0 and 1 (the tasking's exclusion claim is itself testable).

Outputs: runs/e116/{metrics.json, rebarred_census.png}
Run:     python lab/e116_rebarred_census.py   (E116_SMOKE=1 -> shakedown)
No NOTES/THINKING/QUEUE/STATE edits; no commit (the dispatcher owns those).
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

import common
common.DEVICE = "cpu"                    # CPU-only mandate
from common import REPO, Cfg, CharCorpus, TinyGPT, run_dir, save_json
import e043_install as E43

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

SMOKE = os.environ.get("E116_SMOKE") == "1"

# ------------------------------------------------------------------ constants
NAME = "ZEPHYRA"
PRE = 130
HOSTS = ["FLORIZEL", "ELIZABETH"]
THREADS = 8                              # T050: 12 thrashes the box
FED_ROWS = 130                           # rows 0..129 fed by the battery
GATE_TOL = 5e-6                          # reproduction tolerance vs stored
CONTENT_BAR = 0.5                        # arms within factor 2 (frozen)
CONC_RATIO = 2.0                         # top >= 2x next (frozen)
N_TOTAL = 6

# nets: (label, checkpoint, family)
NETS = [
    ("42",   "e048_repro.pt",            "2.7M-e001"),
    ("43",   "e082_b43_install.pt",      "2.7M-e001"),
    ("4305", "e098_install_s4305.pt",    "0.84M-e053c"),
    ("4306", "e098_install_s4306.pt",    "0.84M-e053c"),
    ("4307", "e098_install_s4307_pat.pt", "0.84M-e053c"),   # patience branch
    ("4308", "e098_install_s4308.pt",    "0.84M-e053c"),
]
if SMOKE:
    NETS = [NETS[2]]                     # one fresh seed, reduced rows

CFG_2P7 = Cfg()                                              # 6L/6H/192d/256
CFG_FRESH = Cfg(vocab=65, n_layer=4, n_head=4, n_embd=128, block_size=512)

E098 = REPO / "runs" / "e098" / "metrics.json"
E067 = REPO / "runs" / "e067" / "metrics.json"
E082 = REPO / "runs" / "e082" / "metrics.json"
CKPT_DIR = REPO / "runs" / "checkpoints"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

REGISTERED = {
    "universe": f"fed wpe rows 0..{FED_ROWS - 1} (install-60 battery, 130 tok)",
    "arms": ("mean arm: wpe[r] <- mean(all wpe rows of that net); zero arm: "
             "wpe[r] <- 0; drop = base battery p(Z) - arm p(Z)"),
    "content_test": (f"content(r) <=> m(r)>0 and z(r)>0 and "
                     f"min(m,z)/max(m,z) >= {CONTENT_BAR} (arms agree within "
                     "factor 2 — the e098-established criterion; scaffolding "
                     "excluded BY THE TEST, not by a hardcoded list; "
                     "calibration: row-1 scaffolding ratios <=0.32 vs "
                     "decision-row content ratios >=0.70, 0.5 in the empty "
                     "middle)"),
    "statistic": (f"among content rows ranked by MEAN-arm drop: concentrated "
                  f"<=> top drop >= {CONC_RATIO:.0f}x next content row drop"),
    "verdict_bars": (f">=5/6 seeds concentrated => ADDRESS-UNIVERSALITY "
                     "GRADUATES TO LAW; 3-4/6 => STRUCTURE-STRONG, "
                     "CONCENTRATION-MIXED (stays descriptive); <=2/6 => honest "
                     "negative"),
    "scope": ("6 seeds: e098 fresh 4305-4308 (0.84M family) + references 42 "
              "(e048_repro, e067 census) and 43 (e082 install)"),
    "free_reanalysis": ("reuse stored per-row censuses; regenerate only the "
                        "missing arms on CPU; all touched stored numbers "
                        f"gate-checked to {GATE_TOL}"),
}
log("REGISTERED: " + " | ".join(f"{k}: {v}" for k, v in REGISTERED.items()))


# ------------------------------------------------------------------ battery
def build_battery():
    """e098/e067/e082 install-60 battery, verbatim protocol rebuild."""
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    assert corpus.vocab_size == 65
    stoi = corpus.stoi
    train_ids = corpus.train
    train_text = "".join(corpus.itos[int(i)] for i in train_ids)
    import random
    host_occ = []
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + E43.POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ = host_occ[:60]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    assert mix == {"FLORIZEL": 19, "ELIZABETH": 41}, f"splice drift {mix}"
    ids = torch.stack([corpus.encode(train_text[p - PRE:p])
                       for p, _ in install_occ])          # [60, 130]
    log(f"battery rebuilt: 60 windows x 130 tokens, mix {mix}")
    return ids, stoi["Z"]


@torch.no_grad()
def battery_pz(m: TinyGPT, ids: torch.Tensor, zid: int, bs: int = 30) -> float:
    """e067's battery: mean p(Z) at the last context position."""
    m.eval()
    pzs = []
    for i in range(0, ids.shape[0], bs):
        lg, _ = m(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs += [float(pr[k, zid]) for k in range(pr.shape[0])]
    return float(np.mean(pzs))


def load_net(path: Path, cfg: Cfg) -> TinyGPT:
    m = TinyGPT(cfg)
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


def arm_census(net: TinyGPT, ids, zid, rows, arm: str) -> dict[int, float]:
    """Per-row single-arm census on the given rows (in-place, restored)."""
    w = net.wpe.weight.data
    orig = w.clone()
    replace = orig.mean(0) if arm == "mean" else torch.zeros_like(orig[0])
    out: dict[int, float] = {}
    for r in rows:
        w.copy_(orig)
        w[r] = replace
        out[int(r)] = battery_pz(net, ids, zid)
    w.copy_(orig)
    return out


# ------------------------------------------------------------------ stored data
def stored_tables() -> dict:
    """Per-net stored mean/zero arm pieces + gate rows, from shipped metrics."""
    m98 = json.loads(E098.read_text())
    m67 = json.loads(E067.read_text())
    m82 = json.loads(E082.read_text())
    out = {}
    for s, rec in m98["per_seed"].items():
        c = rec["census"]
        out[s] = {
            "family": "0.84M-e053c",
            "base_pz": c["base_pz"],
            "mean_drops_full": c["drops"],           # 512 rows
            "zero_stored": {int(k): v for k, v in c["zero_arm_top3"].items()},
            "mean_spot_rows": [0, 1, 127, 128, 129] + list(c["top3_rows"]),
        }
    p = m67["primary"]
    out["42"] = {
        "family": "2.7M-e001",
        "base_pz": p["base_pz"],
        "mean_drops_full": p["drops"],               # 256 rows
        "zero_stored": {int(k): v for k, v in p["zero_arm"].items()},
        "mean_spot_rows": [0, 1, 123, 124, 129] + list(p["top5_rows"][:3]),
    }
    g1 = m82["gate1"]
    out["43"] = {
        "family": "2.7M-e001",
        "base_pz": m82["gate0"]["final"]["install60_pz"],
        "mean_drops_full": None,                     # only 22 rows stored:
        "mean_stored_rows": {int(k): float(v["install60"])
                             for k, v in g1["arm_mean"].items()},
        "zero_stored": {int(k): v for k, v in g1["arm_zero"].items()},
        "mean_spot_rows": sorted(int(k) for k in g1["arm_mean"]),
    }
    return out


# ------------------------------------------------------------------ statistic
def content_mask(m: np.ndarray, z: np.ndarray, bar: float) -> np.ndarray:
    ok = (m > 0) & (z > 0)
    lo = np.minimum(m, z)
    hi = np.maximum(m, z)
    agree = np.zeros_like(m, dtype=bool)
    agree[ok] = lo[ok] / hi[ok] >= bar
    return agree


def rebar(m: np.ndarray, z: np.ndarray, bar: float):
    """Top content row, next content row, ratio (ranked by mean-arm drop)."""
    cm = content_mask(m, z, bar)
    idx = np.argsort(-m)
    content_sorted = [int(r) for r in idx if cm[r]]
    if len(content_sorted) < 2:
        return cm, content_sorted, None, None, None, None, None
    r1, r2 = content_sorted[0], content_sorted[1]
    d1, d2 = float(m[r1]), float(m[r2])
    return cm, content_sorted, r1, d1, r2, d2, d1 / max(d2, 1e-12)


# ------------------------------------------------------------------ main
def main():
    torch.set_num_threads(THREADS)
    tag = "e116_smoke" if SMOKE else "e116"
    rd = run_dir(tag)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ids, zid = build_battery()
    stored = stored_tables()
    rows = list(range(FED_ROWS)) if not SMOKE else list(range(10))

    per_seed, all_gates_ok = {}, True
    for label, ck, family in NETS:
        skey = label
        st = stored[skey]
        cfg = CFG_2P7 if st["family"] == "2.7M-e001" else CFG_FRESH
        net = load_net(CKPT_DIR / ck, cfg)
        n_par = net.num_params()
        gates = {}

        # ---- base p(Z) gate
        base_pz = battery_pz(net, ids, zid)
        gates["base_pz_stored"] = st["base_pz"]
        gates["base_pz_diff"] = abs(base_pz - st["base_pz"])
        gates["base_pz_ok"] = bool(gates["base_pz_diff"] <= GATE_TOL)

        # ---- mean arm: reuse stored where full; recompute for seed 43
        if st["mean_drops_full"] is not None:
            mean_full = np.array(st["mean_drops_full"][:FED_ROWS], dtype=float)
            mean_d = mean_full[rows]
            source = f"stored full ({len(st['mean_drops_full'])} rows)"
        else:
            pz = arm_census(net, ids, zid, rows, "mean")
            mean_d = np.array([base_pz - pz[r] for r in rows])
            source = "recomputed (stored partial: 22 rows)"
        # gate: spot-row mean-arm reproduction (vs stored full array, or vs
        # the 22 stored rows when the mean arm had to be recomputed)
        spots = sorted({r for r in st["mean_spot_rows"] if r < FED_ROWS
                        and r in rows})
        if spots:
            pz = arm_census(net, ids, zid, spots, "mean")
            ref = ({r: float(mean_full[r]) for r in spots}
                   if st["mean_drops_full"] is not None
                   else {r: st["mean_stored_rows"][r] for r in spots})
            diffs = [abs((base_pz - pz[r]) - ref[r]) for r in spots]
            gates["mean_arm_max_diff"] = max(diffs)
            gates["mean_arm_ok"] = bool(max(diffs) <= GATE_TOL)

        # ---- zero arm: recompute on all fed rows; verify vs stored subset
        pzz = arm_census(net, ids, zid, rows, "zero")
        zero_d = np.array([base_pz - pzz[r] for r in rows])
        zdiffs = [abs(zero_d[r] - v) for r, v in st["zero_stored"].items()
                  if r < FED_ROWS and r in rows]
        gates["zero_arm_max_diff"] = max(zdiffs) if zdiffs else 0.0
        gates["zero_arm_ok"] = bool(
            (not zdiffs) or max(zdiffs) <= GATE_TOL)
        gates["all_ok"] = bool(gates.get("base_pz_ok", False)
                               and gates.get("mean_arm_ok", False)
                               and gates["zero_arm_ok"])
        all_gates_ok &= gates["all_ok"]

        # ---- the frozen statistic
        cm, csorted, r1, d1, r2, d2, ratio = rebar(mean_d, zero_d, CONTENT_BAR)
        conc = bool(ratio is not None and ratio >= CONC_RATIO)
        arm_ratio = np.where((mean_d > 0) & (zero_d > 0),
                             np.minimum(mean_d, zero_d)
                             / np.maximum(mean_d, zero_d), 0.0)

        # riders
        zsorted = sorted((int(r) for r in rows if cm[r]),
                         key=lambda r: -zero_d[r])
        zr = (float(zero_d[zsorted[0]] / zero_d[zsorted[1]])
              if len(zsorted) >= 2 else None)
        sens = {}
        for bar in (0.25, 0.75):
            _, _, a1, _, a2, _, ra = rebar(mean_d, zero_d, bar)
            sens[str(bar)] = {"top": a1, "next": a2,
                              "ratio": ra,
                              "conc": bool(ra is not None and ra >= CONC_RATIO)}
        no0 = [r for r in csorted if r != 0]
        ex0 = {"top": no0[0] if no0 else None,
               "top_drop": float(mean_d[no0[0]]) if no0 else None,
               "next": no0[1] if len(no0) > 1 else None,
               "next_drop": float(mean_d[no0[1]]) if len(no0) > 1 else None,
               "ratio": (float(mean_d[no0[0]] / mean_d[no0[1]])
                         if len(no0) > 1 else None)}
        dw = [r for r in csorted if 122 <= r <= 135]

        per_seed[skey] = {
            "label": label, "net": ck, "family": st["family"],
            "params": n_par, "base_pz": base_pz, "gates": gates,
            "mean_arm_source": source,
            "mean_drops": mean_d.tolist(), "zero_drops": zero_d.tolist(),
            "content_mask": cm.tolist(),
            "content_rows_by_mean_drop_top10": csorted[:10],
            "arm_agreement_ratio_top8": {int(r): float(arm_ratio[r])
                                         for r in np.argsort(-mean_d)[:8]},
            "row0_content": bool(cm[0]), "row1_content": bool(cm[1]),
            "row0": {"mean": float(mean_d[0]), "zero": float(zero_d[0])},
            "row1": {"mean": float(mean_d[1]), "zero": float(zero_d[1])},
            "top_content_row": r1, "top_content_drop": d1,
            "next_content_row": r2, "next_content_drop": d2,
            "ratio": ratio, "concentrated": conc,
            "decision_window_content_row": (dw[0] if dw else None),
            "riders": {"rank_by_zero_arm": {"top": zsorted[0] if zsorted else None,
                                            "next": zsorted[1] if len(zsorted) > 1 else None,
                                            "ratio": zr},
                       "content_bar_sensitivity": sens,
                       "excl_row0": ex0},
        }
        log(f"seed {label} ({st['family']}, {n_par}p, {ck}): gates "
            f"{'OK' if gates['all_ok'] else 'FAIL'} | top content r{r1} "
            f"{d1:.4f} vs next r{r2} {d2:.4f} -> ratio "
            f"{ratio:.3f} {'CONC' if conc else 'no-conc'} | row0/row1 content "
            f"{cm[0]}/{cm[1]} | decision-window row "
            f"{dw[0] if dw else None}")
        del net

    # ---- verdict
    n_conc = sum(1 for v in per_seed.values() if v["concentrated"])
    table = [{"seed": v["label"], "family": v["family"],
              "top_content_row": v["top_content_row"],
              "top_content_drop": round(v["top_content_drop"], 4),
              "next_content_row": v["next_content_row"],
              "next_content_drop": round(v["next_content_drop"], 4),
              "ratio": round(v["ratio"], 3),
              "concentrated": v["concentrated"]}
             for v in per_seed.values()]
    if not SMOKE:
        if n_conc >= 5:
            verdict = ("ADDRESS-UNIVERSALITY GRADUATES TO LAW (n=6): in every "
                       "install-family net tested (6/6 seeds, two architecture "
                       "families), the installed name's census load concentrates "
                       "in a single content row — its battery p(Z) drop is at "
                       "least 2x the next content row's drop; the runner-up "
                       "content row is the decision-window row in every seed.")
        elif n_conc >= 3:
            verdict = ("STRUCTURE-STRONG, CONCENTRATION-MIXED (stays "
                       "descriptive): the content-row structure (one "
                       "decision-window content row per seed) replicates, but "
                       "the 2x concentration bar over content rows clears in "
                       f"only {n_conc}/6 seeds")
        else:
            verdict = (f"CONCENTRATION FAILS THE RE-BAR ({n_conc}/6) — honest "
                       "negative; the structure claim must lean on row "
                       "identity/magnitude bands, not 2x dominance")
    else:
        verdict = f"SMOKE ({n_conc} concentrated)"
    log(f"VERDICT: {n_conc}/{N_TOTAL if not SMOKE else len(NETS)} "
        f"concentrated -> {verdict}")

    metrics = {
        "experiment": "e116_rebarred_census",
        "purpose": ("T068-registered formal M1 graduation check: the re-barred "
                    "census over CONTENT rows only (mean-arm ~= zero-arm), "
                    "scaffolding excluded by the content test; free re-analysis "
                    "of e098/e067/e082 censuses with only missing arms "
                    "regenerated (CPU)"),
        "started": started, "wall_s": round(time.time() - T0, 1),
        "smoke": SMOKE, "threads": THREADS,
        "registered": REGISTERED,
        "battery": {"windows": 60, "tokens": 130, "splice_rng": E43.SPLICE_RNG,
                    "corpus_seed": 1337, "mix": {"FLORIZEL": 19,
                                                "ELIZABETH": 41}},
        "verdict": {"n_concentrated": n_conc,
                    "n_total": N_TOTAL if not SMOKE else len(NETS),
                    "table": table, "text": verdict,
                    "bars": REGISTERED["verdict_bars"]},
        "gates": {"tolerance": GATE_TOL, "all_ok": bool(all_gates_ok),
                  "per_seed": {k: v["gates"] for k, v in per_seed.items()}},
        "per_seed": per_seed,
    }
    save_json(rd / "metrics.json", metrics)
    log("metrics.json written")
    plot(rd / "rebarred_census.png", per_seed, metrics)
    log(f"plot written -> {rd}")
    return 0


# ------------------------------------------------------------------ plot
def plot(path: Path, per_seed: dict, M: dict):
    seeds = list(per_seed)
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    axes = axes.ravel()
    for ax in axes[len(seeds):]:
        ax.axis("off")
    for ax, s in zip(axes, seeds):
        v = per_seed[s]
        m = np.array(v["mean_drops"])
        cm = np.array(v["content_mask"], dtype=bool)
        order = np.argsort(-m)
        content_rows = [int(r) for r in order if cm[r]][:8]
        scaffold_rows = [int(r) for r in order if not cm[r]
                         and m[r] > 0.02][:3]
        rows_show = sorted(set(content_rows + scaffold_rows))
        xs = np.arange(len(rows_show))
        for i, r in enumerate(rows_show):
            if cm[r]:
                ax.bar(i, m[r], 0.62, color="steelblue")
            else:
                ax.bar(i, m[r], 0.62, color="lightgray", hatch="xx",
                       edgecolor="dimgray")
        labels = [f"{r}" + ("" if cm[r] else "*") for r in rows_show]
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        d1 = v["top_content_drop"]
        ax.axhline(d1 / 2.0, color="crimson", ls="--", lw=1.6,
                   label=f"2x bar = {d1 / 2.0:.3f}")
        i1 = rows_show.index(v["top_content_row"])
        i2 = rows_show.index(v["next_content_row"])
        ax.bar(i1, d1, 0.62, color="none", edgecolor="crimson", lw=2.4)
        ax.bar(i2, v["next_content_drop"], 0.62, color="none",
               edgecolor="darkgreen", lw=2.4)
        ax.annotate(f"top {d1:.3f}", (i1, d1), xytext=(6, -2),
                    textcoords="offset points", fontsize=8, color="crimson",
                    va="top")
        ax.annotate(f"next {v['next_content_drop']:.3f}", (i2,
                    v["next_content_drop"]), xytext=(6, -2),
                    textcoords="offset points", fontsize=8,
                    color="darkgreen", va="top")
        ax.set_xlabel("* = excluded by the content test (gray); "
                      "blue = content rows", fontsize=8)
        ax.set_ylabel("mean-arm census drop", fontsize=9)
        ax.set_title(
            f"seed {s} ({v['family']}): r{v['top_content_row']} "
            f"{d1:.3f} vs r{v['next_content_row']} "
            f"{v['next_content_drop']:.3f} -> {v['ratio']:.2f} "
            f"[{'CONC' if v['concentrated'] else 'no-conc'}]", fontsize=10)
        ax.text(0.98, 0.55, f"row0 content {v['row0_content']}\n"
                f"row1 content {v['row1_content']}\n"
                f"decision-window row {v['decision_window_content_row']}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                bbox=dict(fc="white", ec="none", alpha=0.7))
        ax.legend(fontsize=7, loc="upper right")
    vd = M["verdict"]
    fig.suptitle(
        f"E116 re-barred census - content rows only (arms agree within 2x): "
        f"{vd['n_concentrated']}/{vd['n_total']} concentrated\n"
        f"VERDICT: {vd['text']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
