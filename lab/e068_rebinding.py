"""E068 — P1-COORDINATE rebinding surgery: is the address portable as a ROW PAIR?
(T042 registered, ungated by the e071 outcome block.)

REGISTERED QUESTION (frozen in THINKING.md T042 / e071 OUTCOME before compute):
  is the address portable as a ROW PAIR? Registered prediction: pair-copy to
  a shifted window restores partial expression at the new geometry;
  single-row copies fail — the conjunction is the unit.

REGISTERED DESIGN (frozen):
  - Surgery arms (eval-only; wpe.weight edited in place then restored):
      PAIR-COPY  : wpe rows (k, k+129) := original rows (0, 129)
      ROW0-ONLY  : wpe row  k        := original row 0    (row 129 left in place)
      ROW129-ONLY: wpe row  k+129    := original row 129  (row 0 left in place)
      NO-COPY    : no wpe edit (the T032 knife-edge replication — should collapse)
    for k in {10, 20}.
  - Battery: install-60 contexts LEFT-SHIFTED by k — the 130-char window that
    sat at positions 0..129 now sits at k..k+129 (the context chars move WITH
    the anchors; positions 0..k-1 are filled with the k real corpus chars
    immediately preceding, i.e. left-padding with content fixed, the exact
    T032 probe construction). Readout: mean p(Z) at the new last position
    (k+129), which predicts the name-start exactly as the unshifted battery
    does. Held-30 (same protocol rebuild, never trained) run as the cleaner
    generalization battery in a second pass.
  - REVERSE-CONTEXT control (report-only): unshifted battery with the
    pair-copied rows — does the pair ALONE (without moving content) do
    anything at the old geometry?
  - REGISTERED BARS: pair-copy >= 0.20 mean p(Z) at the new geometry
    (install-60 shifted battery, per k) = PORTABLE PAIR; single-row arms AND
    no-copy-shift all < 0.05 = conjunction-unit confirmed; anything else =
    write the honest MIXED outcome, don't force.
  - Honesty note (pre-registered here): destination row k+129 in {139, 149}
    lies in the never-trained band (rows 130..255 were never fed for the
    130-char install contexts) — the transplant is into untrained rows by
    construction; k >= 10 keeps both destinations well inside the 256-row
    table and clear of the {0, 129} source slots.

Protocol machinery reused verbatim from e067_address_census /
e071_row0_generalization (the exact e055/e066/e066b rebuild): corpus seed
1337, E43.SPLICE_RNG shuffle, install-60 = host_occ[:60] / held-30 =
host_occ[60:90], 130-char pre-name contexts, installed net e048_repro.pt,
in-place wpe surgery + restore, CPU only.

Outputs: runs/e068/{metrics.json, rebinding.png}.
Run: python lab/e068_rebinding.py
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
INSTALLED_CK = "e048_repro.pt"
KS = (10, 20)                   # shifted window starts (registered)
ARMS = ("pair", "row0", "row129", "nocopy")
PORTABLE_BAR = 0.20             # pair-copy >= this at new geometry
FAIL_BAR = 0.05                 # single-row / no-copy arms < this

REGISTERED = {
    "question": "is the address portable as a ROW PAIR?",
    "prediction": ("pair-copy to a shifted window (k, k+129) restores partial "
                   "expression at the new geometry; single-row copies fail — "
                   "the conjunction is the unit"),
    "pair_copy": "wpe rows (k, k+129) := original rows (0, 129), k in {10, 20}",
    "row0_only": "wpe row k := original row 0 (129 left in place)",
    "row129_only": "wpe row k+129 := original row 129 (0 left in place)",
    "nocopy": "no wpe edit (T032 knife-edge replication)",
    "battery": ("install-60 contexts left-shifted by k (content moves WITH "
                "the anchors; k real preceding corpus chars fill 0..k-1); "
                "held-30 as the cleaner generalization battery, second pass"),
    "readout": "mean p(Z) at the new last position (k+129)",
    "reverse_context": ("report-only: unshifted battery with pair-copied "
                        "rows — does the pair ALONE do anything at the old "
                        "geometry?"),
    "bars": {"portable_pair": f"pair-copy mean p(Z) >= {PORTABLE_BAR} "
                              "(install-60 shifted, per k)",
             "conjunction_unit": f"single-row arms AND no-copy-shift all "
                                 f"< {FAIL_BAR} (install-60 shifted)"},
    "mixed": "anything else = write the honest MIXED outcome, don't force",
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
    """e067/e071's battery, extended with an argmax-Z flag per window."""
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
    rd = run_dir("e068")
    log("E068 rebinding surgery: is the address portable as a ROW PAIR?")

    # ---- exact e055/e066/e066b/e067/e071 protocol rebuild (verbatim)
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
        log(f"[ref] unshifted {tag}: base p(Z) {ref[tag]['mean_pz']:.4f} "
            f"(argmax-Z frac {ref[tag]['frac_argmax_z']:.2f})")
    log("protocol check: install60 ref vs e066b A0 0.7149 / "
        "held30 ref vs e071 0.42871")

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

    # ---- registered bars (primary battery: install-60 shifted, per k)
    portable = {k: results[(k, "pair", "install60")]["mean_pz"] >= PORTABLE_BAR
                for k in KS}
    conj_fail = {k: all(results[(k, a, "install60")]["mean_pz"] < FAIL_BAR
                        for a in ("row0", "row129", "nocopy")) for k in KS}
    portable_held = {k: results[(k, "pair", "held30")]["mean_pz"] >= PORTABLE_BAR
                     for k in KS}
    conj_fail_held = {k: all(results[(k, a, "held30")]["mean_pz"] < FAIL_BAR
                             for a in ("row0", "row129", "nocopy"))
                      for k in KS}
    all_portable = all(portable.values())
    all_conj = all(conj_fail.values())
    if all_portable and all_conj:
        verdict = "PORTABLE PAIR — conjunction-unit CONFIRMED"
    elif all_portable:
        verdict = "MIXED — PORTABLE PAIR fires but conjunction-unit NOT " \
                  "confirmed (a single row or no-copy arm escapes the bar)"
    elif all_conj:
        verdict = "MIXED — pair does NOT reach the portable bar; single-row " \
                  "and no-copy arms all fail (pair adds nothing detectable)"
    else:
        verdict = "MIXED — honest report: bars cross inconsistently across k"
    log(f"BARS: portable {portable} | conjunction-fail {conj_fail} | "
        f"held30 legs portable {portable_held} conj-fail {conj_fail_held}")
    log(f"VERDICT: {verdict}")

    out = {
        "experiment": "e068_rebinding",
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
        "references_unshifted": ref,
        "prior_numbers": {"e066b_A0_install60": 0.7149,
                          "e071_held30_installed": 0.42871,
                          "e067_row0_drop_mean_arm": 0.55495,
                          "e067_row129_drop_mean_arm": 0.24047,
                          "t032_onechar_shift": "129/131-context + left-pad "
                                                "collapse 0.556 -> ~0.12"},
        "results": {f"k{k}__{arm}__{bat}": results[(k, arm, bat)]
                    for (k, arm, bat) in results},
        "reverse_context_report_only": {f"k{k}__{bat}": reverse[(k, bat)]
                                        for (k, bat) in reverse},
        "bars": {
            "portable_bar": PORTABLE_BAR, "fail_bar": FAIL_BAR,
            "portable_pair_fired": {str(k): bool(portable[k]) for k in KS},
            "conjunction_unit_fired": {str(k): bool(conj_fail[k]) for k in KS},
            "generalization_leg_held30": {
                "portable_pair_fired": {str(k): bool(portable_held[k])
                                        for k in KS},
                "conjunction_unit_fired": {str(k): bool(conj_fail_held[k])
                                           for k in KS}},
            "verdict": verdict,
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
                          "PRIMARY: install-60 SHIFTED by k (trained windows)"),
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

    ax = axes[1, 0]
    for j, k in enumerate(KS):
        for arm, sty in [("pair", "-o"), ("nocopy", ":s"), ("row0", "--^"),
                         ("row129", "--v")]:
            pz = np.sort(results[(k, arm, "install60")]["pz_per_ctx"])[::-1]
            ax.plot(np.arange(1, len(pz) + 1), pz, sty, ms=3, lw=1.1,
                    color=("crimson" if k == 10 else "steelblue")
                    if arm == "pair" else "gray",
                    alpha=1.0 if arm == "pair" else 0.75,
                    label=f"k={k} {arm}")
    pz0 = np.sort(ref["install60"]["pz_per_ctx"])[::-1]
    ax.plot(np.arange(1, 61), pz0, "-k", lw=1.4, label="unshifted ref")
    ax.axhline(PORTABLE_BAR, color="seagreen", ls="--", lw=1)
    ax.axhline(FAIL_BAR, color="gray", ls=":", lw=1)
    ax.set_xlabel("install-60 window rank (sorted)")
    ax.set_ylabel("p(Z) per window")
    ax.set_title("per-window distribution, install-60 shifted: broad rescue "
                 "or a few windows?", fontsize=10)
    ax.legend(fontsize=6.5, ncols=2)

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

    fig.suptitle(f"e068 P1-COORDINATE rebinding surgery — VERDICT: {verdict}\n"
                 f"portable {portable} | conjunction-fail {conj_fail} | "
                 f"held30 legs portable {portable_held} conj-fail "
                 f"{conj_fail_held}", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(rd / "rebinding.png", dpi=140)
    log(f"done -> {rd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
