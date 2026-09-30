"""[SUPERSEDED — DO NOT RUN (R58-audit quarantine, 2026-09-30): this is the concurrent-session draft with the WRONG HOST RULER (e185 2.7M lineage-1 battery instead of the g3-lineage S-DISC). The canonical cell is lab/g3K_kappa.py (runs/g3K/metrics.json, commit 294dc6e). Retained as the double-session era's record; nothing here is a result."""]
"""G3K — THE KAPPA CELL (W022's fork; spec scratch/g3K_design.md, bars frozen).

kappa = (isotropic kill rung) / (wash kill rung) at MATCHED MEAN PER-COORDINATE
RMS (total L2 over sqrt(N) equal), read on two rulers side by side:
  kappa_store — the store's g0 ruler (g3-GEN organism, install-60 p(Z) at ctx
                offset 0, kill def g0 < 0.5, g3R's bar);
  kappa_host  — the host fact's battery ruler (e185's organism, install-60
                p(Z) at offset -12 = "the fact", kill def p(Z) < 0.5).
Eval-only, CPU, no training, no checkpoints written.

BUILDS ON: g3R (rebuild_protocol / instrument imports; lambda-sweep + matched
isotropic machinery, generalized here from store-subspace to ALL parameters),
g3 (organism runs/checkpoints/g3_gen.pt and its stored full-wash checkpoints
g3_gen_s{1,10,300}.pt = the wash direction, no new wash training), e185/e187
(host organism e131_consolidated_e113 + stored real-wash checkpoints
e185_ctrl_s{2,10}.pt), R56 critic (scratch/r56_critic.md). NEW: both kappas
on one convention (all-parameter matched mean RMS), interpolated kill rungs,
concentration co-reads.

DEVIATIONS (pre-run, recorded):
 1. PROVENANCE FINDING (spec's own pre-dispatch instrument check): the g3
    organism's host is bit-identical to the pristine base and carries NO fact
    (store-off g0 = 0.001), so "the host fact's battery ruler ON the g3
    organism" is not literally instantiable. kappa_host is therefore read on
    the e185 host-fact organism (e131 root, its stored real-wash checkpoint as
    the direction) under the IDENTICAL convention and grid; kappa_store on the
    g3 organism. The pair is cross-organism, NOT one organism. The bars are
    applied verbatim to the pair, with this caveat on the verdict.
 2. Grid extended below the spec's {1..64} with {1/64..1/2} rungs (1/64-1/16 added AFTER a first pass showed the store wash off-grid-low at 1/8; bars unaffected) (units of
    the wash checkpoint's own total displacement D(t*)): the wash checkpoint
    is already dead at rung 1, so a kill rung <= 1 needs sub-1 resolution.
 3. Wash direction = full ALL-parameter delta (theta_t* - theta_root) at the
    first checkpoint under the bar (g3: +1; e185 ctrl: +2); the terminal
    (g3 +300, e185 +10) and mid (g3 +10) directions are co-reads.
Run:  python lab/g3K_kappa_cell.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                     # noqa: E402
import torch                                            # noqa: E402

import common                                           # noqa: E402
common.DEVICE = "cpu"
import g3R_seed_replicates as G3R                       # noqa: E402
import g3_generative_store as G3                        # noqa: E402
import e043_install as E43                              # noqa: E402
from common import Cfg, TinyGPT, run_dir, save_json     # noqa: E402

torch.set_num_threads(4)

import matplotlib                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

CK = E43.REPO / "runs" / "checkpoints"
RUNGS = [1/64, 1/32, 1/16, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
BAR = 0.5
ISO_SEEDS = {"store": (11211, 11212, 11213), "host": (11201, 11202, 11203)}
REF = {"g3_root_g0": 0.8885950446128845, "g3_s1_g0": 5.8887377235805616e-05,
       "e185_root_gm12": 0.9155886173248291, "e185_s2_gm12": 0.027077054604887962}
BARS = {
    "KAPPA-SPLIT": "kappa_host <= 4 AND kappa_store >= 8 at matched mean per-coordinate RMS",
    "NO-BASIN-UNIVERSAL": "both kappas <= 4",
    "MIXED": "any other pattern; the pair reported verbatim",
}
M: dict = {"experiment": "g3K_kappa_cell", "date": common.now_iso(),
           "bars": BARS, "rungs": RUNGS, "kill_def": f"p(Z) < {BAR}",
           "stages": {}}
RD = run_dir("g3K")


def flush(stage):
    M["stages"][stage] = True
    save_json(RD / "metrics.json", E43.jsonable(M))


def load_sd(path):
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    return {k: v.clone() for k, v in sd.items()}


def kill_rung(rungs, vals, bar=BAR):
    """First rung below bar, interpolated in log2 (linear in value) with the
    previous rung. Returns (rung, flag)."""
    for i, v in enumerate(vals):
        if v < bar:
            if i == 0:
                return rungs[0], "OFF-GRID-LOW (<= %g)" % rungs[0]
            v0, v1 = vals[i - 1], v
            l0, l1 = np.log2(rungs[i - 1]), np.log2(rungs[i])
            f = (v0 - bar) / max(v0 - v1, 1e-12)
            return float(2 ** (l0 + f * (l1 - l0))), "ok"
    return float(rungs[-1]), "OFF-GRID-HIGH (> %g)" % rungs[-1]


def perturb(sd_root, keys, direction, l2, rms_only=None):
    """sd_root + direction scaled to total L2 `l2` (direction is a dict)."""
    n = float(sum(float(direction[k].float().norm() ** 2) for k in keys) ** 0.5)
    sd = {k: v.clone() for k, v in sd_root.items()}
    for k in keys:
        sd[k] = sd_root[k] + (l2 / max(n, 1e-12)) * direction[k].to(sd_root[k].dtype)
    return sd


def concentration(delta, keys, store_keys=None):
    flat = torch.cat([delta[k].flatten().float() for k in keys])
    e = flat ** 2
    N = flat.numel()
    srt = torch.sort(e, descending=True).values
    top = float(srt[: max(N // 10, 1)].sum() / e.sum())
    out = {"N": N, "L2": float(e.sum().sqrt()), "mean_RMS": float((e.sum() / N).sqrt()),
           "top_decile_energy_share": top}
    if store_keys:
        se = float(sum(float((delta[k].float() ** 2).sum()) for k in store_keys))
        sn = sum(delta[k].numel() for k in store_keys)
        out.update({"store_coord_frac": sn / N, "store_energy_share_wash": se / float(e.sum()),
                    "store_RMS_wash": (se / sn) ** 0.5,
                    "store_energy_share_iso": sn / N,
                    "store_RMS_iso": float(e.sum() / N) ** 0.5,
                    "store_RMS_ratio_wash_over_iso": ((se / sn) ** 0.5) / float((e.sum() / N).sqrt())})
    return out


def kappa_cell(name, mk, read, sd_root, sd_wash, keys, seeds, store_keys=None):
    """One direction on one organism: wash curve + 3-seed isotropic curves at
    matched total L2 over ALL params (=> equal mean per-coordinate RMS)."""
    delta = {k: (sd_wash[k].float() - sd_root[k].float()) for k in keys}
    D = float(sum(float((delta[k] ** 2).sum()) for k in keys) ** 0.5)
    conc = concentration(delta, keys, store_keys)
    wash = []
    for r in RUNGS:
        sd = perturb(sd_root, keys, delta, r * D)
        wash.append(read(mk(sd)))
    iso = []
    for r in RUNGS:
        vs = []
        for s_ in seeds:
            g = torch.Generator().manual_seed(s_)
            pert = {k: torch.randn(sd_root[k].shape, generator=g) for k in keys}
            vs.append(read(mk(perturb(sd_root, keys, pert, r * D))))
        iso.append(vs)
    iso_mean = [float(np.mean(v)) for v in iso]
    wk, wf = kill_rung(RUNGS, wash)
    ik, ifl = kill_rung(RUNGS, iso_mean)
    per_seed = []
    for j in range(len(seeds)):
        k_, f_ = kill_rung(RUNGS, [v[j] for v in iso])
        per_seed.append({"seed": seeds[j], "kill_rung": k_, "flag": f_,
                         "kappa": k_ / wk})
    kap = ik / wk
    log(f"{name}: D={D:.4f} wash kill {wk:.3f} [{wf}] | iso kill {ik:.2f} [{ifl}] "
        f"| kappa={kap:.2f}{'+' if 'HIGH' in ifl else ''}")
    log(f"  wash  : " + " ".join(f"{v:.3f}" for v in wash))
    log(f"  iso   : " + " ".join(f"{v:.3f}" for v in iso_mean))
    return {"direction": name, "D_total_L2": D, "concentration": conc,
            "wash_curve": wash, "iso_curves_per_seed": iso,
            "iso_curve_mean": iso_mean,
            "wash_kill_rung": wk, "wash_flag": wf,
            "iso_kill_rung": ik, "iso_flag": ifl,
            "kappa": kap, "kappa_is_lower_bound": "HIGH" in ifl,
            "kappa_is_upper_bound_wash_offgrid_low": "LOW" in wf,
            "per_seed": per_seed}


def cos_delta(a, b, keys):
    num = sum(float((a[k].float() * b[k].float()).sum()) for k in keys)
    da = sum(float((a[k].float() ** 2).sum()) for k in keys) ** 0.5
    db = sum(float((b[k].float() ** 2).sum()) for k in keys) ** 0.5
    return num / (da * db)


def main():
    log("G3K THE KAPPA CELL — eval-only CPU, threads 4")
    pr = G3R.rebuild_protocol()
    M["protocol_gates"] = pr["gates"]
    zid = pr["zid"]

    # ------------------------------------------------ STORE ruler (g3 organism)
    sd_root, skeys, host_keys, root_gates = G3R.load_root(pr)
    M["store_root_gates"] = root_gates
    mk_s = lambda sd: G3.evl_load("gen", sd)
    read_s = lambda net: G3.battery_cell(net, pr["ids130"], zid)["mean_pz"]
    net0 = mk_s(sd_root)
    pkeys = [k for k, _ in net0.named_parameters()]
    del net0
    M["store_n_params"] = int(sum(sd_root[k].numel() for k in pkeys))
    dirs = {}
    for tag, ck in (("t*(+1)", "g3_gen_s1.pt"), ("mid(+10)", "g3_gen_s10.pt"),
                    ("terminal(+300)", "g3_gen_s300.pt")):
        dirs[tag] = load_sd(CK / ck)
    g0_s1 = read_s(mk_s(dirs["t*(+1)"]))
    gate_s1 = abs(g0_s1 - REF["g3_s1_g0"]) < 1e-3
    M["store_wash_gate"] = {"s1_g0": g0_s1, "ref": REF["g3_s1_g0"], "pass": bool(gate_s1)}
    log(f"store wash ckpt gate: g0(+1) = {g0_s1:.3e} vs {REF['g3_s1_g0']:.3e} -> {gate_s1}")
    assert gate_s1
    store = {}
    skp = [k for k in skeys if k in pkeys]
    for tag, sd_w in dirs.items():
        store[tag] = kappa_cell(f"STORE[{tag}]", mk_s, read_s, sd_root, sd_w, pkeys,
                                ISO_SEEDS["store"], store_keys=skp)
    dl = {t: {k: sd_w[k].float() - sd_root[k].float() for k in pkeys}
          for t, sd_w in dirs.items()}
    M["store_direction_cosines"] = {
        "t*_vs_mid": cos_delta(dl["t*(+1)"], dl["mid(+10)"], pkeys),
        "t*_vs_terminal": cos_delta(dl["t*(+1)"], dl["terminal(+300)"], pkeys)}
    M["store"] = store
    flush("store")

    # ------------------------------------------------ HOST ruler (e185 organism)
    root_h = load_sd(CK / "e131_consolidated_e113.pt")
    mk_h = lambda sd: (lambda m: (m.load_state_dict(sd), m.eval(), m)[2])(TinyGPT(Cfg()))
    read_h = lambda net: G3.battery_cell(net, pr["gm12_ids"], zid)["mean_pz"]
    gr = read_h(mk_h(root_h))
    dirs_h = {"t*(+2)": load_sd(CK / "e185_ctrl_s2.pt"),
              "terminal(+10)": load_sd(CK / "e185_ctrl_s10.pt")}
    g2 = read_h(mk_h(dirs_h["t*(+2)"]))
    M["host_gates"] = {"root_gm12": gr, "ref_root": REF["e185_root_gm12"],
                       "s2_gm12": g2, "ref_s2": REF["e185_s2_gm12"],
                       "pass": bool(abs(gr - REF["e185_root_gm12"]) < 1e-4
                                    and abs(g2 - REF["e185_s2_gm12"]) < 1e-3)}
    log(f"host gates: root g-12 {gr:.6f} (ref {REF['e185_root_gm12']:.6f}); "
        f"+2 {g2:.5f} (ref {REF['e185_s2_gm12']:.5f}) -> {M['host_gates']['pass']}")
    assert M["host_gates"]["pass"], "host provenance gate failed"
    hkeys = [k for k, _ in mk_h(root_h).named_parameters()]
    host = {}
    for tag, sd_w in dirs_h.items():
        host[tag] = kappa_cell(f"HOST[{tag}]", mk_h, read_h, root_h, sd_w, hkeys,
                               ISO_SEEDS["host"])
    M["host"] = host
    flush("host")

    # ------------------------------------------------ adjudication (frozen bars)
    ks = store["t*(+1)"]["kappa"]
    kh = host["t*(+2)"]["kappa"]
    ks_lb = store["t*(+1)"]["kappa_is_lower_bound"]
    kh_lb = host["t*(+2)"]["kappa_is_lower_bound"]
    if kh <= 4 and ks >= 8:
        verdict = "KAPPA-SPLIT"
    elif kh <= 4 and ks <= 4:
        verdict = "NO-BASIN-UNIVERSAL"
    else:
        verdict = "MIXED"
    M["adjudication"] = {
        "verdict": verdict, "kappa_store": ks, "kappa_store_lower_bound": ks_lb,
        "kappa_host": kh, "kappa_host_lower_bound": kh_lb,
        "bar_text": BARS[verdict],
        "co_read_kappas": {"store": {t: v["kappa"] for t, v in store.items()},
                           "host": {t: v["kappa"] for t, v in host.items()}},
        "caveat": ("cross-organism pair (deviation 1): kappa_host on the e185 host-"
                   "fact organism, kappa_store on the g3 store organism; "
                   "organ n=1, wash-direction n=1 per organism, iso n=3 draws"),
    }
    log(f"VERDICT {verdict}: kappa_store={ks:.2f}{'+' if ks_lb else ''} "
        f"kappa_host={kh:.2f}{'+' if kh_lb else ''}")
    plot(RD / "kappa_cell.png")
    flush("adjudication")
    return 0


def plot(path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (nm, grp) in zip(axes[0], (("STORE (g3 organism, g0 ruler)", M["store"]),
                                       ("HOST (e185 organism, g-12 ruler)", M["host"]))):
        for i, (tag, c) in enumerate(grp.items()):
            col = ["crimson", "darkorange", "purple"][i]
            ax.plot(RUNGS, c["wash_curve"], "o-", color=col, label=f"wash {tag}")
            ax.plot(RUNGS, c["iso_curve_mean"], "^--", color=col, alpha=0.6,
                    label=f"iso {tag} (kappa {c['kappa']:.1f})")
        ax.axhline(BAR, color="gray", ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("rung (multiples of wash ckpt total L2, matched mean RMS)")
        ax.set_ylabel("p(Z)")
        ax.set_title(nm, fontsize=9)
        ax.legend(fontsize=7)
    ax = axes[1, 0]
    labs, vals = [], []
    for g, nm in ((M["store"], "store"), (M["host"], "host")):
        for t, c in g.items():
            labs.append(f"{nm}\n{t}")
            vals.append(c["kappa"])
    ax.bar(labs, vals, color=["crimson"] * 3 + ["steelblue"] * 2)
    for x, v in zip(labs, vals):
        ax.text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(4, color="k", ls=":")
    ax.axhline(8, color="k", ls="--")
    ax.set_ylabel("kappa = iso kill rung / wash kill rung")
    ax = axes[1, 1]
    ax.axis("off")
    a = M["adjudication"]
    ax.text(0.02, 0.98, f"VERDICT: {a['verdict']}\nkappa_store={a['kappa_store']:.2f}\n"
            f"kappa_host={a['kappa_host']:.2f}\n{a['bar_text']}\n\n{a['caveat']}",
            va="top", fontsize=9, wrap=True, transform=ax.transAxes)
    fig.suptitle("G3K the kappa cell", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
