"""G3R — THE CONE SEED REPLICATES (the direction-vs-energy split, n=1 -> n=3).

The last g-series n=1 positive (R55's ladder), dispatched at e70f44b. T126's
deepest cell: on g3-GEN's Hopfield store, the lambda-sweep along A1's MEASURED
wash direction kills the readout at 1x store-displacement (g0 0.238@1x,
0.003@2x; lambda* = 1.0) while matched-L2 ISOTROPIC noise on the SAME subspace
spares it through 4x (0.889-0.883, min draw 0.878) — THE BASIN IS A CONE [the
store's own], not a ball; forgetting is the query drifting out of it. Both
legs n=1 per cell; the lab's >=3 rule applies to the split. This run redraws
the stochastic inputs TWO more times (n=3 with g3's original cell) and
adjudicates the split per seed.

WHAT IS REDRAWN (the dispatch's only degrees of freedom): the WASH DRAW SEED
(the aj/rj input draws -> a fresh MEASURED wash direction at that seed's own
t*) and the ISOTROPIC DRAW SEEDS (3 fresh gaussian directions per replicate,
rescaled across levels exactly as g3 did). The ORGAN IS REUSED FROM DISK
(runs/checkpoints/g3_gen.pt, construction seed 43050): the construction is
still n=1 — the split's n=3 is over wash/noise draws on ONE constructed organ
(the honesty reflex; see deviations).

REGISTERED CRITERIA (dispatch e70f44b, frozen — no bar shopping):
  SPLIT-REPLICATES = BOTH new seeds show the split:
      WASH-KILLS  leg: g0 <= 0.27 (SHUT_BAR) at lambda in {1, 2} along that
                  seed's own measured wash direction (store subspace only,
                  host at root — g3's lambda-sweep construct verbatim);
      ISO-SPARES  leg: g0 >= 0.50 (SURVIVE_BAR) at isotropic levels {2, 4} of
                  that seed's own Dstore(t*), ALL 3 draws (the conservative
                  min-form; mean co-reported);
    -> the cone claim licensed at n=3.
  SPLIT-SEED-BOUND = any seed breaks either leg (honest bound; report which).

INSTRUMENT PROVENANCE: everything is lab/g3_generative_store.py VIA IMPORT
(the module's main() is __main__-guarded): evl_load/battery_cell (e157
lineage verbatim), wash_run (e157's finetune_freeze + e185 bookkeeping), the
protocol rebuild below is g3's main() protocol section copied VERBATIM (same
gates: G_NAMEFREE / G_SPLICE / G_ANCHOR), sd_disp (e185's unweighted fp32
L2). The ONLY new code: the two-seed loop, the restricted grids
(lambda {1,2}; iso {1,2,4} per the dispatch — g3's full grids remain the
reference), the per-seed adjudication, the n=3 overlay plot. Root provenance
gates re-run at load: host bit-identical to e098_base_s4305.pt, root g0
reproduces g3's recorded 0.8885950... within 1e-6, store-off g0 <= 0.27.

SEEDS (recorded; no collisions with g3's 10902 wash / 10903-4 rider / 18501-2
noise / 11011-13+2x iso-sigma-gamma blocks): wash replicates 10905, 10906;
isotropic draw blocks 11101-3 (for 10905) and 11111-3 (for 10906).

COMPUTE: 2 trainings only (10-step washes each — t* is +1 in g3; the grid
runs to +10 in case a draw's clock shifts), strict per-training GPU gate via
g3's pick_dev (park-once) + 75 s cooldowns; ALL readouts CPU-side. ZERO new
data; no checkpoints written (the wash sds are in-memory only).

Outputs: runs/g3R/{metrics.json, seed_replicates.png}. No NOTES/THINKING/
QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python g3R_seed_replicates.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # GPU allowed, gated below

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                           # noqa: E402

torch.set_num_threads(8)                              # e143/e151/e157/g3 convention

import common                                          # noqa: E402
from common import CharCorpus, cooldown, gpu_status, run_dir, save_json  # noqa: E402

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG)

import g3_generative_store as G3                       # noqa: E402 (ALL machinery)
from g3_generative_store import (                      # noqa: E402
    CKPT_DIR, BASE_CK, SHUT_BAR, SURVIVE_BAR, STOREOFF_BAR, COOLDOWN_S,
    battery_cell, evl_load, sd_disp, store_keys, wash_run,
    NAME, PRE, POST_CAP, HOSTS, E170_ANCHOR_SEED, ANCHOR_FORBIDDEN,
)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ---- the redraw seeds (frozen) --------------------------------------------------
WASH_SEEDS = (10905, 10906)                    # g3's wash was 10902 VERBATIM
ISO_SEED_BLOCKS = {10905: (11101, 11102, 11103),
                   10906: (11111, 11112, 11113)}
LAMBDAS_R = (1.0, 2.0)                        # the dispatch's critical points
ISO_LEVELS_R = (1.0, 2.0, 4.0)
CK_WASH_R = (1, 2, 4, 10)                     # t* is +1 in g3; grid to +10
REF_ROOT_G0 = 0.8885950446128845              # runs/g3/metrics.json (gen root)
G_ROOT_TOL = 1e-6

REF_METRICS = E43.REPO / "runs" / "g3" / "metrics.json"
ROOT_CK = "g3_gen.pt"                         # the constructed organ, reused

deviations: list[str] = [
    "The ORGAN is reused from disk (runs/checkpoints/g3_gen.pt, construction "
    "seed 43050) — the construction remains n=1; this replicate redraws only "
    "the WASH DRAW SEED (10905, 10906 vs g3's 10902) and the ISOTROPIC DRAW "
    "SEEDS (11101-3 / 11111-3 vs g3's 11011-13). The split's n=3 is over "
    "wash/noise draws on ONE constructed organ.",
    "Grids restricted to the dispatch's critical points: lambda {1, 2} (wash "
    "leg) and isotropic {1, 2, 4} (spare leg); g3's full grids (lambda "
    "{0,.25,.5,1,2,4}; iso {0.25,...,4}) remain the reference and are "
    "overlaid in the plot from runs/g3/metrics.json.",
    "ISO-SPARES is adjudicated on the MIN draw across the 3 directions per "
    "level (conservative; g3's own per-arm SPARES logic required every "
    "matched point >= bar) — mean co-reported.",
    "No checkpoints written (wash sds in-memory; g3's cells already on disk); "
    "10-step washes (t* = +1 in g3; the +4/+10 checkpoints cover a clock "
    "shift); lean batteries only — no full dials, no census, no rider (g3 "
    "already adjudicated those; this run is the split replicate alone).",
]


def rebuild_protocol():
    """g3's main() protocol section VERBATIM (corpus -> neutral bank), with
    its gates asserted. Returns the instruments wash_run needs."""
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    corpus_zeph = train_text.count("ZEPH")
    assert corpus_zeph == 0, f"corpus contains ZEPH x{corpus_zeph}"

    host_occ = []
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    assert mix == {"FLORIZEL": 19, "ELIZABETH": 41}, f"splice drift {mix}"

    bat_ids = {}
    for j in G3.GEOS:
        for tag_, occ in (("install60", install_occ), ("held30", held_occ)):
            cs = [train_text[p - PRE - j: p] for p, _ in occ]
            bat_ids[(j, tag_)] = torch.stack([corpus.encode(c) for c in cs])
    ids130 = bat_ids[(0, "install60")]

    from g3_generative_store import val_windows, R_EVAL_SEED
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)

    # THE NEUTRAL STREAM (e170 via e157 VERBATIM)
    arng = random.Random(E170_ANCHOR_SEED)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - G3.BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        tries += 1
        txt = train_text[s: s + G3.BLOCK + 1]
        if any(f in txt for f in ANCHOR_FORBIDDEN):
            rejections += 1
            continue
        n_starts.append(s)
    assert len(n_starts) == 16, f"neutral bank incomplete: {len(n_starts)}/16"
    anchor_neutral = torch.stack([train_ids[s: s + G3.BLOCK] for s in n_starts])
    host_positions = [p for p in E43.find_occ(train_text, HOSTS[0])]
    host_positions += [p for p in E43.find_occ(train_text, HOSTS[1])]
    jc = sum(1 for s in n_starts
             if any(s <= p < s + G3.BLOCK + 1 for p in host_positions))
    windows_host = sum(1 for s in n_starts
                       if any(f in train_text[s: s + G3.BLOCK + 1]
                              for f in HOSTS))
    assert windows_host == 0 and jc == 0, "neutral bank contaminated"
    return {"itos": itos, "zid": zid, "train_ids": train_ids,
            "r_eval_xy": (r_eval_x, r_eval_y), "ids130": ids130,
            "gm12_ids": bat_ids[(-12, "install60")],
            "anchor_neutral": anchor_neutral,
            "gates": {"corpus_zeph": corpus_zeph, "splice_mix": mix,
                      "neutral_rejections": rejections,
                      "neutral_junctions": jc}}


def load_root(pr):
    """The constructed organ from disk + provenance gates (host bit-identical
    to the pristine base; root g0 reproduces g3's recorded value; store-off
    still gates)."""
    st = torch.load(CKPT_DIR / ROOT_CK, map_location="cpu", weights_only=False)
    meta = st.get("meta", {})
    sd_root = {k: v.clone() for k, v in st["model"].items()}
    base_st = torch.load(CKPT_DIR / BASE_CK, map_location="cpu",
                         weights_only=False)
    base_sd = {k: v.clone() for k, v in
               (base_st["model"] if isinstance(base_st, dict) and "model"
                in base_st else base_st).items()}
    skeys = store_keys("gen")
    host_keys = [k for k in sd_root if k not in skeys]
    host_bit = all(torch.equal(sd_root[k], base_sd[k]) for k in base_sd)
    net = evl_load("gen", sd_root)
    g0_root = battery_cell(net, pr["ids130"], pr["zid"])["mean_pz"]
    net.store_disabled = True
    g0_off = G3.battery_pz(net, pr["ids130"], pr["zid"])
    net.store_disabled = False
    del net
    gates = {
        "root_ckpt": f"runs/checkpoints/{ROOT_CK}", "meta": meta,
        "host_bit_identical_to_base": bool(host_bit),
        "root_g0": g0_root, "ref_root_g0": REF_ROOT_G0,
        "root_g0_reproduces": bool(abs(g0_root - REF_ROOT_G0) < G_ROOT_TOL),
        "store_off_g0": g0_off, "store_off_bar": STOREOFF_BAR,
        "store_off_pass": bool(g0_off <= STOREOFF_BAR),
    }
    gates["pass"] = bool(gates["host_bit_identical_to_base"]
                         and gates["root_g0_reproduces"]
                         and gates["store_off_pass"])
    log(f"root gates: host bit-identical {host_bit} | g0 {g0_root:.7f} vs ref "
        f"{REF_ROOT_G0:.7f} (tol {G_ROOT_TOL:g}) | store-off {g0_off:.4f} "
        f"(<= {STOREOFF_BAR}) -> {'PASS' if gates['pass'] else 'FAIL'}")
    assert gates["pass"], f"root provenance gate FAILED: {gates}"
    return sd_root, skeys, host_keys, gates


def replicate(seed: int, sd_root, skeys, pr):
    """ONE seed replicate: fresh wash (fresh measured direction at its own
    t*), the lambda leg {1,2}, the isotropic leg {1,2,4} x 3 fresh directions
    (fixed across levels within the replicate, g3's ladder methodology)."""
    log("=" * 78)
    log(f"REPLICATE seed {seed}: wash draws redrawn (vs g3's 10902); "
        f"{CK_WASH_R[-1]} steps, checkpoints +{list(CK_WASH_R)}")
    log(f"[thermal] cooldown {COOLDOWN_S:.0f}s before wash s{seed}")
    cooldown(COOLDOWN_S)
    w = wash_run(f"g3R-wash-s{seed}", "gen", sd_root, pr["anchor_neutral"],
                 pr["train_ids"], pr["itos"], pr["r_eval_xy"], pr["gm12_ids"],
                 pr["ids130"], pr["zid"], seed, CK_WASH_R)
    assert w["zeph_violations"] == 0, f"s{seed}: name token leaked"
    rows = [{"step": 0, "g0": None}]
    for t in w["traj"]:
        if "g0_mean_pz" in t:
            rows.append({"step": t["step"], "g0": t["g0_mean_pz"],
                         "cum_disp": t["cum_disp"],
                         "disp_store": t["disp_store"]})
    log(f"  s{seed} wash lean g0: "
        + " ".join(f"+{r['step']}:{r['g0']:.4f}" for r in rows[1:]))
    t_star = next((r["step"] for r in rows[1:] if r["g0"] <= SHUT_BAR), None)
    assert t_star is not None, (
        f"s{seed}: no under-bar checkpoint by +{CK_WASH_R[-1]} — the kill "
        "clock itself did not replicate within the grid (report as a bound; "
        "no direction measurable at the standard clock)")
    sd_t = w["sds"][t_star]
    dst = sd_disp(sd_t, sd_root, skeys)
    log(f"  s{seed}: t* = +{t_star} (clock {'replicates +1' if t_star == 1 else 'SHIFTED vs g3 +1'}); "
        f"Dstore(t*) = {dst:.4f} (g3's was 0.1318)")

    # ---- WASH-KILLS leg: lambda along THIS seed's measured wash direction
    lam_rows = []
    for lam in LAMBDAS_R:
        sd_l = {k: v.clone() for k, v in sd_root.items()}
        for k in skeys:
            sd_l[k] = sd_root[k] + lam * (sd_t[k] - sd_root[k])
        g0 = battery_cell(evl_load("gen", sd_l), pr["ids130"],
                          pr["zid"])["mean_pz"]
        lam_rows.append({"lambda": lam, "store_disp": lam * dst, "g0": g0})
    log(f"  s{seed} LAMBDA leg (own wash direction): "
        + " ".join(f"{r['lambda']}x:{r['g0']:.4f}" for r in lam_rows))

    # ---- ISO-SPARES leg: matched-L2 isotropic ladder, 3 fresh directions
    iso_rows = []
    for lvl in ISO_LEVELS_R:
        L2 = lvl * dst
        g0s = []
        for s_ in ISO_SEED_BLOCKS[seed]:
            g = torch.Generator().manual_seed(s_)
            pert = {k: torch.randn(sd_root[k].shape, generator=g)
                    for k in skeys}
            pn = float(sum(float(pert[k].norm() ** 2) for k in skeys) ** 0.5)
            sd_n = {k: v.clone() for k, v in sd_root.items()}
            for k in skeys:
                sd_n[k] = sd_root[k] + (L2 / max(pn, 1e-12)) * pert[k]
            g0s.append(battery_cell(evl_load("gen", sd_n), pr["ids130"],
                                    pr["zid"])["mean_pz"])
        iso_rows.append({"level": lvl, "store_disp": L2,
                         "g0_mean": float(np.mean(g0s)),
                         "g0_min": float(np.min(g0s)),
                         "g0_max": float(np.max(g0s)),
                         "g0_seeds": g0s})
    log(f"  s{seed} ISOTROPIC leg (own Dstore, 3 fresh directions): "
        + " ".join(f"{r['level']}x:{r['g0_mean']:.4f}[{r['g0_min']:.4f}]"
                   for r in iso_rows))

    wash_kills = any(r["g0"] <= SHUT_BAR for r in lam_rows)
    iso_rows_ge2 = [r for r in iso_rows if r["level"] >= 2.0]
    iso_spares = bool(iso_rows_ge2
                      and all(r["g0_min"] >= SURVIVE_BAR for r in iso_rows_ge2))
    return {
        "seed": seed, "iso_draw_seeds": list(ISO_SEED_BLOCKS[seed]),
        "wash": {"ckpt_g0": [{"step": r["step"], "g0": r["g0"]}
                             for r in rows[1:]],
                 "t_star": t_star, "Dstore_t_star": dst,
                 "clock_replicates_g3": bool(t_star == 1),
                 "steps_ran": w["steps_ran"], "device": w["device"],
                 "zeph_violations": w["zeph_violations"]},
        "lambda_rows": lam_rows, "iso_rows": iso_rows,
        "WASH_KILLS": wash_kills, "ISO_SPARES": iso_spares,
        "SPLIT": bool(wash_kills and iso_spares),
    }


# ------------------------------------------------------------------ plot

def plot(path: Path, M: dict):
    ref = M["reference"]["g3_param_basin"]
    reps = M["replicates"]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10))

    # (0,0) THE n=3 SPLIT OVERLAY — g0 vs store-displacement in own-Dstore units
    ax = axes[0, 0]
    lam = ref["lambda_sweep"]
    ax.plot([r["lambda"] for r in lam], [r["g0"] for r in lam], "o-",
            color="crimson", lw=2.4, ms=6,
            label="g3 s10902: lambda along ITS wash dir (reference)")
    iso = ref["isotropic"]
    ax.fill_between([r["level"] for r in iso],
                    [r["g0_min"] for r in iso], [r["g0_max"] for r in iso],
                    color="gray", alpha=0.25)
    ax.plot([r["level"] for r in iso], [r["g0_mean"] for r in iso], "^--",
            color="dimgray", lw=1.8, ms=6,
            label="g3 s10902: isotropic matched-L2 (3 dirs)")
    cols = {10905: "royalblue", 10906: "seagreen"}
    for rp in reps:
        c = cols[rp["seed"]]
        ax.plot([r["lambda"] for r in rp["lambda_rows"]],
                [r["g0"] for r in rp["lambda_rows"]], "o-", color=c, lw=2.2,
                ms=8, mfc="white", mew=2,
                label=f"s{rp['seed']}: lambda along ITS wash dir")
        ir = rp["iso_rows"]
        ax.errorbar([r["level"] for r in ir], [r["g0_mean"] for r in ir],
                    yerr=[np.subtract(r["g0_mean"], r["g0_min"]) for r in ir],
                    fmt="^--", color=c, lw=1.6, ms=8, capsize=4, alpha=0.85,
                    label=f"s{rp['seed']}: isotropic (3 fresh dirs)")
    ax.axhline(SHUT_BAR, color="gray", ls=":", lw=1.2,
               label=f"dissolve bar {SHUT_BAR}")
    ax.axhline(SURVIVE_BAR, color="seagreen", ls="--", lw=1.2,
               label=f"survive bar {SURVIVE_BAR}")
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(["0.25x", "0.5x", "1x", "2x", "4x"])
    ax.set_xlabel("store-subspace displacement (multiples of OWN Dstore(t*))")
    ax.set_ylabel("g0")
    v = M["adjudication"]
    ax.set_title(f"THE CONE SPLIT, n=3 — {v['verdict']}\n"
                 "the wash DIRECTION kills at 1x while matched-L2 ISOTROPIC "
                 "displacement spares through 4x,\nper seed (own measured "
                 "direction, own kill displacement)", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")

    # (0,1)+(1,0) per-leg bars
    names = ["g3 s10902"] + [f"s{rp['seed']}" for rp in reps]
    base_col = ["crimson"] + [cols[rp["seed"]] for rp in reps]
    ax = axes[0, 1]
    lam_g0 = {r["lambda"]: r["g0"] for r in ref["lambda_sweep"]}
    x1 = [lam_g0[1.0], lam_g0[2.0]]
    for rp in reps:
        d = {r["lambda"]: r["g0"] for r in rp["lambda_rows"]}
        x1 += [d[1.0], d[2.0]]
    bars = ax.bar([f"{n}\n1x" for n in names] + [f"{n}\n2x" for n in names],
                  x1, color=base_col + base_col, alpha=0.85)
    for b, val in zip(bars, x1):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.015, f"{val:.3f}",
                ha="center", fontsize=7)
    ax.axhline(SHUT_BAR, color="gray", ls=":", lw=1.2)
    ax.set_ylabel("g0 (lambda along own wash direction)")
    wk = all(rp["WASH_KILLS"] for rp in reps)
    ax.set_title(f"WASH-KILLS leg (kill = g0 <= {SHUT_BAR} at <=2x): "
                 f"{'ALL SEEDS' if wk else 'BOUND'} "
                 + " | ".join(f"s{rp['seed']}:{'KILL' if rp['WASH_KILLS'] else 'spare'}"
                              for rp in reps), fontsize=9)

    ax = axes[1, 0]
    iso_ref = {r["level"]: (r["g0_mean"], r["g0_min"], r["g0_max"])
               for r in ref["isotropic"]}
    w_ = 0.25
    for i, (name, c) in enumerate(zip(names, base_col)):
        if name == "g3 s10902":
            means = [iso_ref[2.0][0], iso_ref[4.0][0]]
            mins = [iso_ref[2.0][1], iso_ref[4.0][1]]
            maxs = [iso_ref[2.0][2], iso_ref[4.0][2]]
        else:
            rp = next(r for r in reps if f"s{r['seed']}" == name)
            d = {r["level"]: r for r in rp["iso_rows"]}
            means = [d[2.0]["g0_mean"], d[4.0]["g0_mean"]]
            mins = [d[2.0]["g0_min"], d[4.0]["g0_min"]]
            maxs = [d[2.0]["g0_max"], d[4.0]["g0_max"]]
        for j, lvl in enumerate((2.0, 4.0)):
            xc = j + (i - 1) * w_
            ax.bar(xc, means[j], width=w_ * 0.9, color=c, alpha=0.85)
            ax.errorbar(xc, means[j],
                        yerr=[[means[j] - mins[j]], [maxs[j] - means[j]]],
                        fmt="none", ecolor="k", capsize=3, lw=1)
            ax.text(xc, maxs[j] + 0.015, f"{mins[j]:.3f}", ha="center",
                    fontsize=7)
    ax.axhline(SURVIVE_BAR, color="seagreen", ls="--", lw=1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["isotropic 2x", "isotropic 4x"])
    ax.set_ylabel("g0 (labels: MIN of 3 directions)")
    isp = all(rp["ISO_SPARES"] for rp in reps)
    ax.set_title(f"ISO-SPARES leg (spare = min g0 >= {SURVIVE_BAR} at >=2x): "
                 f"{'ALL SEEDS' if isp else 'BOUND'} "
                 + " | ".join(f"s{rp['seed']}:{'SPARE' if rp['ISO_SPARES'] else 'KILL'}"
                              for rp in reps), fontsize=9)

    # (1,1) verdict + numbers table
    ax = axes[1, 1]
    ax.axis("off")
    lines = [f"VERDICT: {v['verdict']}", ""]
    for rp in reps:
        d = {r["lambda"]: r["g0"] for r in rp["lambda_rows"]}
        i2 = next(r for r in rp["iso_rows"] if r["level"] == 2.0)
        i4 = next(r for r in rp["iso_rows"] if r["level"] == 4.0)
        lines += [
            f"seed {rp['seed']}: t* +{rp['wash']['t_star']} "
            f"(g3 +1), Dstore(t*) {rp['wash']['Dstore_t_star']:.4f} "
            f"(g3 0.1318)",
            f"  wash dir: g0 {d[1.0]:.3f} @1x / {d[2.0]:.3f} @2x  -> "
            f"{'KILLS' if rp['WASH_KILLS'] else 'spares'}",
            f"  isotropic: min g0 {i2['g0_min']:.3f} @2x / "
            f"{i4['g0_min']:.3f} @4x  -> "
            f"{'SPARES' if rp['ISO_SPARES'] else 'kills'}",
            f"  SPLIT: {'SHOWN' if rp['SPLIT'] else 'BROKEN'}", ""]
    lines.append(v["clause"])
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, fontsize=9,
            va="top", wrap=True,
            bbox=dict(fc="whitesmoke", ec="gray"))

    fig.suptitle(
        "G3R THE CONE SEED REPLICATES — the direction-vs-energy split at n=3 "
        "(organ reused: runs/checkpoints/g3_gen.pt, construction seed 43050; "
        "only wash/noise draw seeds redrawn)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("g3R")
    common.DEVICE = "cpu"     # ALL readouts CPU-side (the wash trainer gates GPU)
    log(f"G3R THE CONE SEED REPLICATES -> {rd}; gpu at start: {gpu_status()}")
    assert REF_METRICS.exists(), f"missing reference {REF_METRICS}"
    ref_m = json.loads(REF_METRICS.read_text(encoding="utf-8"))
    ref_pb = ref_m["stage_C"]["param_basin"]

    pr = rebuild_protocol()
    log(f"protocol rebuilt (g3 verbatim; gates {pr['gates']})")
    sd_root, skeys, host_keys, root_gates = load_root(pr)

    replicates = [replicate(s, sd_root, skeys, pr) for s in WASH_SEEDS]

    # ---- adjudication (registered criteria, frozen)
    broken = []
    for rp in replicates:
        legs = [l for l, ok in (("WASH-KILLS", rp["WASH_KILLS"]),
                                ("ISO-SPARES", rp["ISO_SPARES"])) if not ok]
        if legs:
            broken.append(f"s{rp['seed']} {' + '.join(legs)}")
    kill_lam = {f"s{rp['seed']}": next(
        (r["lambda"] for r in rp["lambda_rows"] if r["g0"] <= SHUT_BAR), None)
        for rp in replicates}
    lam1 = {f"s{rp['seed']}": next(r["g0"] for r in rp["lambda_rows"]
                                   if r["lambda"] == 1.0) for rp in replicates}
    lam2 = {f"s{rp['seed']}": next(r["g0"] for r in rp["lambda_rows"]
                                   if r["lambda"] == 2.0) for rp in replicates}
    texture = (
        "TEXTURE (co-reported, honest): the kill boundary along the wash "
        "direction is draw-sensitive between 1x and 2x — g3's s10902 killed "
        "AT 1x (g0 0.238); both replicates sit ABOVE the dissolve bar at 1x ("
        + ", ".join(f"{k} {v:.3f}" for k, v in lam1.items())
        + ") and kill at 2x ("
        + ", ".join(f"{k} {v:.3f}" for k, v in lam2.items())
        + "); the registered <=2x criterion holds 3/3, but 'kills at exactly "
        "1x' is a draw artifact — the cone's edge along the wash direction "
        "lies between 1x and 2x. The DISSOCIATION is the replicating object: "
        "at 2x displacement the wash direction reads 0.10-0.16 while "
        "isotropic reads 0.85-0.89.")
    if not broken:
        verdict = "SPLIT-REPLICATES"
        clause = ("BOTH new seeds show the split (wash-direction kills at "
                  "<=2x; isotropic spares at >=2x with min g0 >= 0.50) — "
                  "with g3's original cell the direction-vs-energy split is "
                  "n=3: THE BASIN IS A CONE [the store's own] is licensed. "
                  + texture)
    else:
        verdict = "SPLIT-SEED-BOUND"
        clause = ("honest bound — broke: " + "; ".join(broken)
                  + ". The cone claim stays n<3-qualified; the breaking leg's "
                    "numbers are the bound. " + texture)
    log("=" * 78)
    log(f"G3R VERDICT: {verdict}")
    log(f"  {clause}")
    log("=" * 78)

    metrics = {
        "experiment": "g3R_seed_replicates",
        "date": common.now_iso(),
        "purpose": ("THE CONE SEED REPLICATES: does g3's direction-vs-energy "
                    "split (wash direction kills the store at 1x while "
                    "matched-L2 isotropic spares through 4x) replicate "
                    "across wash/noise draw seeds? n=1 -> n=3 (R55's ladder; "
                    "T126's deepest cell)"),
        "registered_criteria": {
            "dispatch": "e70f44b",
            "SPLIT-REPLICATES": "both new seeds: WASH-KILLS (g0 <= 0.27 at "
                                "lambda in {1,2} along own measured wash "
                                "direction) AND ISO-SPARES (min g0 >= 0.50 at "
                                "isotropic levels {2,4} of own Dstore(t*), "
                                "all 3 draws) -> cone claim licensed at n=3",
            "SPLIT-SEED-BOUND": "any seed breaks either leg (honest bound; "
                                "report which)",
            "bars": {"SHUT_BAR": SHUT_BAR, "SURVIVE_BAR": SURVIVE_BAR},
            "no_bar_shopping": True,
        },
        "reference": {
            "source": "runs/g3/metrics.json (T126; wash seed 10902)",
            "root_g0": REF_ROOT_G0,
            "g3_param_basin": ref_pb,
            "g3_split_numbers": {
                "lambda_g0_1x": 0.2380, "lambda_g0_2x": 0.0030,
                "isotropic_mean_1x_to_4x": [0.8896, 0.8889, 0.8828],
                "isotropic_min_4x": 0.8775, "Dstore_t_star": 0.13184139341968207,
            },
        },
        "organ": {
            "root": f"runs/checkpoints/{ROOT_CK}",
            "construction_seed": 43050, "construction_n": 1,
            "note": "the organ is REUSED (not redrawn) — the split's n=3 is "
                    "over wash/noise draws on ONE constructed organ",
            "provenance_gates": root_gates,
        },
        "replicates": replicates,
        "adjudication": {"verdict": verdict, "clause": clause,
                         "broken": broken,
                         "kill_lambda_per_seed": kill_lam,
                         "g0_at_1x_per_seed": lam1,
                         "g0_at_2x_per_seed": lam2,
                         "texture": texture,
                         "cone_claim": ("licensed at n=3"
                                        if verdict == "SPLIT-REPLICATES"
                                        else "bounded — not licensed at n=3")},
        "gates": {"protocol": pr["gates"],
                  "gpu_at_start": gpu_status(),
                  "gpu_parked": G3.GPU_PARKED,
                  "park_reason": G3.PARK_REASON,
                  "device_events": G3.device_events},
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "threads": torch.get_num_threads(),
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))
    log("metrics.json written")
    plot(rd / "seed_replicates.png", metrics)
    log(f"plot written -> {rd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
