"""E078 — P1-COORDINATE rebinding surgery, install #2 (4x-dose): does the
single-row address replicate? (T047 claim-B fix, registered.)

CONTEXT. e068 ran the PAIR/ROW0/ROW129/NO-COPY rebinding battery on
e048_repro.pt (install #1) and landed MIXED under the original T042 bars:
row129-only ~= pair-copy >> row0-only ~= no-copy. T047 audited the claim:
"Necessary" overshoots, "Sufficient" is partial — reworded claim is
"~70% rebind, partial necessity — the single most load-bearing portable
row". REGISTERED FIX (T047): rerun the exact e068 battery on e048_dose.pt
(the 4x-dose Dmix install, the only other full-strength install net — same
install-60 windows, base battery p(Z) = 0.5013645 per e067's replication).
Eval-only, zero training, n=2 installs.

REGISTERED QUESTION (frozen before compute): does the reworded pattern
  ROW129-ONLY ~= PAIR-COPY  >>  ROW0-ONLY ~= NO-COPY
replicate on the second (4x-dose) install?

REGISTERED DESIGN (frozen — VERBATIM e068 battery, checkpoint swap only):
  - Surgery arms (eval-only; wpe.weight edited in place then restored):
      PAIR-COPY  : wpe rows (k, k+129) := original rows (0, 129)
      ROW0-ONLY  : wpe row  k        := original row 0    (row 129 left in place)
      ROW129-ONLY: wpe row  k+129    := original row 129  (row 0 left in place)
      NO-COPY    : no wpe edit (the T032 knife-edge replication — should collapse)
    for k in {10, 20}.
  - Battery: install-60 contexts LEFT-SHIFTED by k (content moves WITH the
    anchors; k real preceding corpus chars fill 0..k-1 — the exact T032/e068
    probe construction). Readout: mean p(Z) at the new last position k+129.
    Held-30 run as the generalization leg in a second pass.
  - REVERSE-CONTEXT control (report-only): unshifted battery with the
    pair-copied rows.
  - REWORDED BARS (T047, operationalized; primary battery = install-60
    shifted, per k):
      (1) ROW129 ~= PAIR : min(pair,row129) >= 0.80 * max(pair,row129)
      (2) >> (gap)       : max(row0,nocopy) <= 0.50 * min(pair,row129)
      (3) ROW0 ~= NOCOPY : min(row0,nocopy) >= 0.80 * max(row0,nocopy)
    REPLICATE = all three fire for every k. "~70% rebind" reported as the
    descriptive ratio row129_shifted / unshifted install-60 ref (e068 gave
    0.714 / 0.663; band 0.45-0.95 flagged, non-gating). e068's original
    T042 bars (portable >= 0.20, fail < 0.05) kept as secondary flags.
  - Honesty note (unchanged from e068): destination row k+129 in {139, 149}
    lies in the never-trained band (rows 130-255) — the transplant is into
    untrained rows by construction.

Protocol machinery reused verbatim from e067/e068 (the exact e055/e066/
e066b rebuild): corpus seed 1337, E43.SPLICE_RNG shuffle, install-60 =
host_occ[:60] / held-30 = host_occ[60:90], 130-char pre-name contexts,
in-place wpe surgery + restore, CPU only.

Outputs: runs/e078/{metrics.json, dose_rebinding.png}.
Run: python lab/e078_dose_rebinding.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                    # CPU experiment

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import torch.nn.functional as F                             # noqa: E402

torch.set_num_threads(min(16, os.cpu_count() or 8))

import common                                               # noqa: E402
common.DEVICE = "cpu"
from common import Cfg, CharCorpus, TinyGPT, run_dir, save_json   # noqa: E402
import e043_install as E43                                  # noqa: E402
import matplotlib                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                             # noqa: E402

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
HOSTS = ["FLORIZEL", "ELIZABETH"]
INSTALLED_CK = "e048_dose.pt"     # 4x-dose Dmix install (install #2)
KS = (10, 20)                     # shifted window starts (registered)
ARMS = ("pair", "row0", "row129", "nocopy")
APPROX_RATIO = 0.80               # "X ~= Y" bar (min >= ratio * max)
GAP_RATIO = 0.50                  # ">>" bar (bottom max <= ratio * top min)
REBIND_LO, REBIND_HI = 0.45, 0.95  # descriptive "~70%-class" band (non-gating)
PORTABLE_BAR = 0.20               # e068 original secondary bars
FAIL_BAR = 0.05
DOSE_BASE_E067 = 0.5013645460208257   # e067 replication: dose install-60 base p(Z)

# e068 published values (e048_repro.pt, install #1) — full precision from
# runs/e068/metrics.json, embedded for the side-by-side.
E068_PUBLISHED = {
    "checkpoint": "e048_repro.pt",
    "refs_unshifted": {"install60": 0.5563087304433186,
                       "held30": 0.4287093544534097},
    "results": {
        "k10__pair__install60": 0.3842690971990426,
        "k10__pair__held30": 0.2656126926187426,
        "k10__row0__install60": 0.14121925948808592,
        "k10__row0__held30": 0.09863122039435741,
        "k10__row129__install60": 0.39708501727630696,
        "k10__row129__held30": 0.2732948899269104,
        "k10__nocopy__install60": 0.14510029503144323,
        "k10__nocopy__held30": 0.10124642937541163,
        "k20__pair__install60": 0.35334603004157544,
        "k20__pair__held30": 0.24205978246948992,
        "k20__row0__install60": 0.12456215241691097,
        "k20__row0__held30": 0.09461780549463583,
        "k20__row129__install60": 0.36888256917397183,
        "k20__row129__held30": 0.2519498047457697,
        "k20__nocopy__install60": 0.1298698772637484,
        "k20__nocopy__held30": 0.09957850583208104,
    },
    "reverse_context": {
        "k10__install60": 0.5527750087281068, "k10__held30": 0.42585871272409953,
        "k20__install60": 0.549319047977527, "k20__held30": 0.42701778329598405,
    },
}

REGISTERED = {
    "question": ("does ROW129-ONLY ~= PAIR-COPY >> ROW0-ONLY ~= NO-COPY "
                 "replicate on the second (4x-dose) install?"),
    "claim_reworded_T047": ("~70% rebind, partial necessity — the single most "
                            "load-bearing portable row"),
    "pair_copy": "wpe rows (k, k+129) := original rows (0, 129), k in {10, 20}",
    "row0_only": "wpe row k := original row 0 (129 left in place)",
    "row129_only": "wpe row k+129 := original row 129 (0 left in place)",
    "nocopy": "no wpe edit (T032 knife-edge replication)",
    "battery": ("install-60 contexts left-shifted by k (content moves WITH "
                "the anchors; k real preceding corpus chars fill 0..k-1); "
                "held-30 as the generalization battery, second pass"),
    "readout": "mean p(Z) at the new last position (k+129)",
    "reverse_context": ("report-only: unshifted battery with pair-copied "
                        "rows — does the pair ALONE do anything at the old "
                        "geometry?"),
    "bars": {
        "row129_approx_pair": f"min(pair,row129) >= {APPROX_RATIO} * "
                              f"max(pair,row129) (install-60 shifted, per k)",
        "gap": f"max(row0,nocopy) <= {GAP_RATIO} * min(pair,row129)",
        "row0_approx_nocopy": f"min(row0,nocopy) >= {APPROX_RATIO} * "
                              f"max(row0,nocopy)",
        "rebind_descriptive": (f"row129_shifted / unshifted install-60 ref in "
                               f"[{REBIND_LO}, {REBIND_HI}] (~70%-class, "
                               f"non-gating)"),
    },
    "replicate": "all three pattern bars fire for every k (install-60 shifted)",
    "mixed": "fires at some k but not all = honest PARTIAL; none = NOT REPLICATED",
}
log("REGISTERED: " + " | ".join(f"{k}: {v}" for k, v in REGISTERED.items()))


def load(path: Path):
    m = TinyGPT(Cfg())
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def battery(m, ids, zid, bs=30):
    """e067/e068/e071's battery, extended with an argmax-Z flag per window."""
    pzs, amax = [], []
    for i in range(0, ids.shape[0], bs):
        lg, _ = m(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs += [float(pr[b, zid]) for b in range(pr.shape[0])]
        amax += [bool(int(pr[b].argmax()) == zid) for b in range(pr.shape[0])]
    return np.array(pzs, dtype=np.float64), np.array(amax, dtype=bool)


def cell(pz, amax):
    return {
        "mean_pz": float(pz.mean()),
        "median_pz": float(np.median(pz)),
        "std_pz": float(pz.std()),
        "frac_pz_ge_0.5": float((pz >= 0.5).mean()),
        "frac_argmax_z": float(amax.mean()),
        "pz_per_ctx": pz.tolist(),
    }


def surgery(net, w_orig, k, arm):
    """In-place wpe rebinding; caller restores from w_orig afterwards."""
    w = net.wpe.weight.data
    w.copy_(w_orig)
    if arm == "pair":
        w[k] = w_orig[0]
        w[k + 129] = w_orig[129]
    elif arm == "row0":
        w[k] = w_orig[0]
    elif arm == "row129":
        w[k + 129] = w_orig[129]
    elif arm == "nocopy":
        pass
    else:
        raise ValueError(arm)


def main():
    rd = run_dir("e078")
    log("E078 rebinding surgery on install #2 (4x-dose): single-row "
        "address replication (T047 fix)")

    # ---- exact e055/e066/e066b/e067/e068 protocol rebuild (verbatim)
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi = corpus.stoi
    train_ids = corpus.train
    train_text = "".join(corpus.itos[int(i)] for i in train_ids)
    zid = stoi["Z"]

    host_occ = []
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + 119 <= len(train_ids):
                host_occ.append((p, host))
    import random as _random
    rng = _random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    assert mix == {"FLORIZEL": 19, "ELIZABETH": 41}, f"splice drift {mix}"

    ctx_unshifted = {
        "install60": [train_text[p - PRE: p] for p, _ in install_occ],
        "held30": [train_text[p - PRE: p] for p, _ in held_occ],
    }
    # ---- left-shifted batteries: window content that sat at 0..129 now sits
    #      at k..k+129 (positions 0..k-1 = the k real preceding corpus chars;
    #      p >= 280 guarantees the backward extension exists). Readout at the
    #      new last position k+129 still predicts the name-start train_text[p].
    ctx_shifted = {}
    for k in KS:
        for tag, occ in [("install60", install_occ), ("held30", held_occ)]:
            cs = [train_text[p - PRE - k: p] for p, _ in occ]
            for c_new, c_old in zip(cs, ctx_unshifted[tag]):
                assert len(c_new) == PRE + k and c_new[k:] == c_old
            ctx_shifted[(k, tag)] = cs
    ids_unshifted = {t: torch.stack([corpus.encode(c) for c in cs])
                     for t, cs in ctx_unshifted.items()}
    ids_shifted = {kk: torch.stack([corpus.encode(c) for c in cs])
                   for kk, cs in ctx_shifted.items()}
    log(f"protocol rebuilt: install60 {mix}, held30; shifted batteries for "
        f"k={list(KS)} (content verified at k..k+129)")

    net = load(E43.REPO / "runs" / "checkpoints" / INSTALLED_CK)
    w_orig = net.wpe.weight.data.clone()
    log(f"installed net {INSTALLED_CK}: params {net.num_params():,}, "
        f"row0 norm {float(w_orig[0].norm()):.3f}, "
        f"row129 norm {float(w_orig[129].norm()):.3f}")

    # ---- unshifted references (old geometry, no surgery)
    ref = {}
    for tag in ids_unshifted:
        pz, am = battery(net, ids_unshifted[tag], zid)
        ref[tag] = cell(pz, am)
        log(f"[ref] unshifted {tag}: base p(Z) {ref[tag]['mean_pz']:.5f} "
            f"(argmax-Z frac {ref[tag]['frac_argmax_z']:.2f})")
    drift = abs(ref["install60"]["mean_pz"] - DOSE_BASE_E067)
    assert drift < 5e-3, (f"install60 ref {ref['install60']['mean_pz']:.6f} "
                          f"drifts from e067 dose replication "
                          f"{DOSE_BASE_E067:.6f} (wrong checkpoint or "
                          f"protocol drift?)")
    log(f"protocol check: install60 ref vs e067 dose replication "
        f"{DOSE_BASE_E067:.5f} -> |drift| {drift:.2e} (OK); e068/repro ref "
        f"was {E068_PUBLISHED['refs_unshifted']['install60']:.5f}")

    # ---- primary arms: surgery x shifted batteries
    results = {}                       # (k, arm, bat) -> cell
    for k in KS:
        for arm in ARMS:
            surgery(net, w_orig, k, arm)
            for tag in ("install60", "held30"):
                pz, am = battery(net, ids_shifted[(k, tag)], zid)
                results[(k, arm, tag)] = cell(pz, am)
            net.wpe.weight.data.copy_(w_orig)
            r60 = results[(k, arm, "install60")]
            r30 = results[(k, arm, "held30")]
            log(f"[k={k:2d}] {arm:8s}: install60-shifted p(Z) "
                f"{r60['mean_pz']:.4f} (argmax-Z {r60['frac_argmax_z']:.2f}) "
                f"| held30-shifted p(Z) {r30['mean_pz']:.4f} "
                f"(argmax-Z {r30['frac_argmax_z']:.2f})")

    # ---- reverse-context control (report-only): pair surgery, UNSHIFTED
    reverse = {}                       # (k, bat) -> cell
    for k in KS:
        surgery(net, w_orig, k, "pair")
        for tag in ids_unshifted:
            pz, am = battery(net, ids_unshifted[tag], zid)
            reverse[(k, tag)] = cell(pz, am)
        net.wpe.weight.data.copy_(w_orig)
        log(f"[reverse k={k:2d}] pair-copy, unshifted: install60 p(Z) "
            f"{reverse[(k, 'install60')]['mean_pz']:.4f} "
            f"(ref {ref['install60']['mean_pz']:.4f}) | held30 p(Z) "
            f"{reverse[(k, 'held30')]['mean_pz']:.4f} "
            f"(ref {ref['held30']['mean_pz']:.4f})")

    # ---- restore checks: wpe bitwise restored; unshifted base reproduces
    assert torch.equal(net.wpe.weight.data, w_orig), "wpe not restored"
    pz_chk, _ = battery(net, ids_unshifted["install60"], zid)
    assert np.allclose(pz_chk, ref["install60"]["pz_per_ctx"], atol=0, rtol=0)
    log("restore checks passed: wpe bitwise restored, base battery reproduced")

    # ---- T047 reworded bars (primary battery: install-60 shifted, per k)
    pat, pat_held = {}, {}
    rebind = {}
    for k in KS:
        g = {a: results[(k, a, "install60")]["mean_pz"] for a in ARMS}
        b1 = min(g["pair"], g["row129"]) >= APPROX_RATIO * max(g["pair"], g["row129"])
        b2 = max(g["row0"], g["nocopy"]) <= GAP_RATIO * min(g["pair"], g["row129"])
        b3 = min(g["row0"], g["nocopy"]) >= APPROX_RATIO * max(g["row0"], g["nocopy"])
        pat[k] = {"row129_approx_pair": bool(b1), "gap": bool(b2),
                  "row0_approx_nocopy": bool(b3), "fired": bool(b1 and b2 and b3)}
        gh = {a: results[(k, a, "held30")]["mean_pz"] for a in ARMS}
        h1 = min(gh["pair"], gh["row129"]) >= APPROX_RATIO * max(gh["pair"], gh["row129"])
        h2 = max(gh["row0"], gh["nocopy"]) <= GAP_RATIO * min(gh["pair"], gh["row129"])
        h3 = min(gh["row0"], gh["nocopy"]) >= APPROX_RATIO * max(gh["row0"], gh["nocopy"])
        pat_held[k] = {"row129_approx_pair": bool(h1), "gap": bool(h2),
                       "row0_approx_nocopy": bool(h3),
                       "fired": bool(h1 and h2 and h3)}
        rebind[k] = {
            "row129_over_ref": g["row129"] / ref["install60"]["mean_pz"],
            "pair_over_ref": g["pair"] / ref["install60"]["mean_pz"],
            "row129_70pct_class": bool(REBIND_LO <=
                                       g["row129"] / ref["install60"]["mean_pz"]
                                       <= REBIND_HI),
        }
    n_fired = sum(pat[k]["fired"] for k in KS)
    if n_fired == len(KS):
        verdict = ("REPLICATED — ROW129-ONLY ~= PAIR-COPY >> ROW0-ONLY ~= "
                   "NO-COPY on install #2 (4x-dose): the '~70% rebind, "
                   "partial necessity' single-row pattern holds at n=2 "
                   "installs")
    elif n_fired == 0:
        verdict = ("NOT REPLICATED — the reworded single-row pattern fails "
                   "on the dose install (write the honest divergence)")
    else:
        verdict = (f"PARTIAL — pattern fires at {n_fired}/{len(KS)} k values "
                   f"(honest mixed report)")
    rebind_str = ", ".join(
        f"{k}: ({rebind[k]['row129_over_ref']:.3f}, "
        f"{rebind[k]['pair_over_ref']:.3f})" for k in KS)
    log(f"BARS: pattern {pat} | held30 legs {pat_held} | rebind ratios "
        f"(row129/ref, pair/ref) {{{rebind_str}}}")
    log(f"VERDICT: {verdict}")

    # ---- e068 original T042 bars kept as secondary flags (continuity)
    portable = {k: results[(k, "pair", "install60")]["mean_pz"] >= PORTABLE_BAR
                for k in KS}
    conj_fail = {k: all(results[(k, a, "install60")]["mean_pz"] < FAIL_BAR
                        for a in ("row0", "row129", "nocopy")) for k in KS}
    log(f"[secondary, e068 original bars] portable {portable} | "
        f"conjunction-fail {conj_fail}")

    # ---- side-by-side vs e068 (install #1, e048_repro)
    side_by_side = {}
    for k in KS:
        for arm in ARMS:
            for bat in ("install60", "held30"):
                kk = f"k{k}__{arm}__{bat}"
                side_by_side[kk] = {
                    "e068_repro": E068_PUBLISHED["results"][kk],
                    "e078_dose": results[(k, arm, bat)]["mean_pz"],
                    "delta": (results[(k, arm, bat)]["mean_pz"]
                              - E068_PUBLISHED["results"][kk]),
                }
    for kk, v in side_by_side.items():
        log(f"[side-by-side {kk}] e068 {v['e068_repro']:.4f} vs e078 "
            f"{v['e078_dose']:.4f} (delta {v['delta']:+.4f})")

    out = {
        "experiment": "e078_dose_rebinding",
        "registered": REGISTERED,
        "protocol": {
            "corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
            "install_mix": mix, "n_held": len(held_occ),
            "ks": list(KS), "pre": PRE,
            "shift_construction": ("left-extension by k real preceding "
                                   "corpus chars: content at 0..129 moves to "
                                   "k..k+129; readout at k+129 predicts the "
                                   "name-start; verified c_new[k:] == c_old"),
            "destination_rows": {str(k): [k, k + 129] for k in KS},
            "honesty_note": ("destination row k+129 (139/149) is in the "
                             "never-trained band rows 130-255 — the "
                             "transplant lands in untrained rows by "
                             "construction"),
        },
        "checkpoint": INSTALLED_CK,
        "checkpoint_note": ("e048_dose.pt = 4x-dose Dmix install, same "
                            "install-60 windows as e048_repro (the only "
                            "other full-strength install net per e067's "
                            "survey); install-60 base p(Z) e067 replication "
                            f"{DOSE_BASE_E067:.7f}, this run "
                            f"{ref['install60']['mean_pz']:.7f}"),
        "references_unshifted": ref,
        "e068_published_side_by_side": {
            "e068_verdict": ("MIXED — PORTABLE PAIR fires but "
                             "conjunction-unit NOT confirmed"),
            **E068_PUBLISHED,
            "comparison": side_by_side,
        },
        "prior_numbers": {"e066b_A0_install60": 0.7149,
                          "e067_dose_base_pz": DOSE_BASE_E067,
                          "e067_row0_drop_mean_arm": 0.55495,
                          "e067_row129_drop_mean_arm": 0.24047,
                          "t032_onechar_shift": "129/131-context + left-pad "
                                                "collapse 0.556 -> ~0.12"},
        "results": {f"k{k}__{arm}__{bat}": results[(k, arm, bat)]
                    for (k, arm, bat) in results},
        "reverse_context_report_only": {f"k{k}__{bat}": reverse[(k, bat)]
                                        for (k, bat) in reverse},
        "bars": {
            "approx_ratio": APPROX_RATIO, "gap_ratio": GAP_RATIO,
            "rebind_band": [REBIND_LO, REBIND_HI],
            "pattern_install60": {str(k): pat[k] for k in KS},
            "pattern_held30": {str(k): pat_held[k] for k in KS},
            "rebind_ratios": {str(k): rebind[k] for k in KS},
            "n_k_fired": n_fired,
            "verdict": verdict,
            "e068_original_bars": {
                "portable_bar": PORTABLE_BAR, "fail_bar": FAIL_BAR,
                "portable_pair_fired": {str(k): bool(portable[k]) for k in KS},
                "conjunction_unit_fired": {str(k): bool(conj_fail[k])
                                           for k in KS},
            },
        },
        "elapsed_s": round(time.time() - T0, 1),
        "cpu_threads": torch.get_num_threads(),
    }
    save_json(rd / "metrics.json", out)

    # ---- figure
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    arm_lbl = {"pair": "PAIR-COPY\n(k,k+129)<-(0,129)", "row0": "ROW0-ONLY\n(k<-0)",
               "row129": "ROW129-ONLY\n(k+129<-129)", "nocopy": "NO-COPY\n(T032 knife-edge)"}
    for ax, bat, ttl in [(axes[0, 0], "install60",
                          "PRIMARY: install-60 SHIFTED by k (4x-dose install #2)"),
                         (axes[0, 1], "held30",
                          "GENERALIZATION: held-30 SHIFTED by k (never trained)")]:
        xs = np.arange(len(ARMS))
        bw = 0.36
        for j, k in enumerate(KS):
            vals = [results[(k, a, bat)]["mean_pz"] for a in ARMS]
            bars = ax.bar(xs + (j - 0.5) * bw, vals, bw,
                          color="crimson" if k == 10 else "steelblue",
                          label=f"k={k} (rows {k},{k + 129})", edgecolor="k",
                          linewidth=0.4)
            for x, v in zip(xs + (j - 0.5) * bw, vals):
                ax.text(x, v + 0.008, f"{v:.3f}", ha="center", fontsize=7.5,
                        rotation=90, va="bottom")
        ax.axhline(PORTABLE_BAR, color="seagreen", ls="--", lw=1.2,
                   label=f"PORTABLE bar {PORTABLE_BAR}")
        ax.axhline(FAIL_BAR, color="gray", ls=":", lw=1.2,
                   label=f"FAIL bar {FAIL_BAR}")
        refv = ref[bat]["mean_pz"]
        ax.axhline(refv, color="k", ls="-.", lw=0.9,
                   label=f"unshifted ref {refv:.3f}")
        ax.set_xticks(xs)
        ax.set_xticklabels([arm_lbl[a] for a in ARMS], fontsize=7.5)
        ax.set_ylabel("mean p(Z) at new last position (k+129)")
        top = max([results[(k, a, bat)]["mean_pz"] for k in KS for a in ARMS]
                  + [refv])
        ax.set_ylim(0, max(0.8, top * 1.25))
        ax.set_title(ttl, fontsize=10)
        ax.legend(fontsize=7)

    # side-by-side: e068 (install #1, repro) vs e078 (install #2, dose)
    ax = axes[1, 0]
    xs = np.arange(len(ARMS))
    bw = 0.19
    for j, (k, col) in enumerate([(10, "crimson"), (20, "steelblue")]):
        v78 = [results[(k, a, "install60")]["mean_pz"] for a in ARMS]
        v68 = [E068_PUBLISHED["results"][f"k{k}__{a}__install60"]
               for a in ARMS]
        ax.bar(xs + (j - 0.5) * 2 * bw, v68, bw, color=col, alpha=0.35,
               edgecolor=col, linewidth=0.8, hatch="//",
               label=f"k={k} e068 repro (install #1)")
        ax.bar(xs + (j - 0.5) * 2 * bw + bw, v78, bw, color=col,
               edgecolor="k", linewidth=0.4,
               label=f"k={k} e078 dose (install #2)")
        for x, v in list(zip(xs + (j - 0.5) * 2 * bw, v68)) + \
                     list(zip(xs + (j - 0.5) * 2 * bw + bw, v78)):
            ax.text(x, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.5,
                    rotation=90, va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels([arm_lbl[a] for a in ARMS], fontsize=7.5)
    ax.set_ylabel("mean p(Z), install-60 SHIFTED")
    ax.set_ylim(0, 0.8)
    ax.set_title("SIDE-BY-SIDE install #1 (e068/e048_repro, hatched) vs "
                 "install #2 (e078/e048_dose, solid)", fontsize=10)
    ax.legend(fontsize=6.5, ncols=2)

    # reverse-context control
    ax = axes[1, 1]
    xs = np.arange(len(KS))
    for j, bat in enumerate(("install60", "held30")):
        vals = [reverse[(k, bat)]["mean_pz"] for k in KS]
        ax.bar(xs + (j - 0.5) * 0.35, vals, 0.35,
               color="crimson" if bat == "install60" else "steelblue",
               label=f"{bat}, pair surgery", edgecolor="k", linewidth=0.4)
        for x, v in zip(xs + (j - 0.5) * 0.35, vals):
            ax.text(x, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)
        ax.axhline(ref[bat]["mean_pz"], color="crimson" if bat == "install60"
                   else "steelblue", ls="-.", lw=1.0,
                   label=f"{bat} unshifted ref {ref[bat]['mean_pz']:.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"k={k}\n(rows {k},{k + 129} overwritten)" for k in KS],
                       fontsize=8)
    ax.set_ylabel("mean p(Z) at last position (129)")
    ax.set_ylim(0, max(0.85, max(v for (k, b), c in reverse.items()
                                 for v in [c["mean_pz"]]) * 1.25))
    ax.set_title("REVERSE-CONTEXT control (report-only): pair ALONE at the "
                 "old geometry", fontsize=10)
    ax.legend(fontsize=7)

    rebind_title = ", ".join(
        f"{k}: {rebind[k]['row129_over_ref']:.3f}" for k in KS)
    fig.suptitle(f"e078 rebinding on install #2 (e048_dose, 4x) — VERDICT: "
                 f"{verdict}\npattern install60 {pat} | held30 legs "
                 f"{pat_held} | rebind row129/ref {rebind_title}",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(rd / "dose_rebinding.png", dpi=140)
    log(f"done -> {rd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
