"""E067 — P1-COORDINATE address census (registered design, T038 follow-up).

Question: what is the FULL single-row weight structure of the installed-name
address? e066b showed row-129 in-place zero: battery p(Z) 0.715 -> 0.272 —
the address is a distributed conjunction with wpe-row concentration. This
census maps every wpe row's causal weight at the decision position.

REGISTERED DESIGN (frozen before compute):
  - For EVERY wpe row r in 0..255: in-place intervention row r <- mean of
    all 256 rows (the e066b A3 arm; preserves norm statistics better than
    zeroing), measure mean battery p(Z) over the 60 install contexts
    (exact e066/e066b protocol rebuild: corpus seed 1337, E43.SPLICE_RNG
    shuffle, install-60, battery ranking).
  - Drop(r) = base_pZ - pZ(r). Sparse verdict: <=5 rows carry >=80% of the
    total POSITIVE drop mass -> SPARSE; else DENSE/clustered, and we report
    the narrowest contiguous window containing row 129 that carries 80%.
  - Confirmation: the ZERO arm additionally run for all rows within +/-5 of
    the top-5 effected rows.
  - Secondary (e066b method): for the top-5 rows, the d5 DeltaState (mean
    over the 6 e055 donors, paired) vs relay_d5 (mean A0 donor state);
    question: do the high-weight rows all feed the same relay content?
  - Replication: same census on the 2.7M install-family net e048_dose.pt
    (4x-dose Dmix install; same install-60 windows). CHECKPOINT SURVEY
    (pre-run): every e043/e048-family checkpoint is 2,739,072 params (2.7M)
    — the e066 primary e048_repro IS 2.7M; e058's 2.7M nets (e001.pt,
    e028_b43.pt) carry NO install; e048_direct* are direct-trained
    controls, not install family. e048_dose is the only other
    full-strength install net. Gate: if its base battery p(Z) < 0.20 the
    replication is uninformative -> note and skip.
  - Sanity null: contexts are 130 chars, so positions 130..255 are never
    fed to the net; their drops form the null band.
  - No training. CPU only. torch.set_num_threads(min(16, cpu_count)).

Outputs: runs/e067/{metrics.json, census.png}.
Run: python lab/e067_address_census.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")          # CPU experiment

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
PRIMARY_CK = "e048_repro.pt"
REPLICA_CK = "e048_dose.pt"
REPLICA_GATE = 0.20          # skip replication if base p(Z) below this
SPARSE_MAX_ROWS = 5
SPARSE_MASS_BAR = 0.80


def load(path: Path):
    m = TinyGPT(Cfg())
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def fwd_states(m, toks):
    """e066b's instrument: per-depth residual states + logits."""
    x = m.wte(toks.unsqueeze(0)) + m.wpe(torch.arange(toks.shape[0]))
    xs = [x]
    for b in m.h:
        x = b(x)
        xs.append(x)
    lg = m.lm_head(m.ln_f(x))
    return xs, lg


def cos(a, b):
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def main():
    rd = run_dir("e067")
    log("E067 address census: per-wpe-row causal weight on installed p(Z)")

    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi = corpus.stoi
    train_ids = corpus.train
    train_text = "".join(corpus.itos[int(i)] for i in train_ids)
    zid = stoi["Z"]

    # ---- exact e055/e066/e066b protocol rebuild (verbatim)
    host_occ = []
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + 119 <= len(train_ids):
                host_occ.append((p, host))
    import random as _random
    rng = _random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, _held = host_occ[:60], host_occ[60:90]
    ctx130_i = [train_text[p - PRE: p] for p, _ in install_occ]

    ctx_ids = torch.stack([corpus.encode(c) for c in ctx130_i])   # [60,130]

    @torch.no_grad()
    def battery(m, bs=30):
        rows = []
        for i in range(0, len(ctx130_i), bs):
            lg, _ = m(ctx_ids[i:i + bs])
            pr = F.softmax(lg[:, -1], -1)
            rows += [float(pr[k, zid]) for k in range(pr.shape[0])]
        return rows

    # donor selection (e066b rule) — used for the d5 secondary only
    net0 = load(E43.REPO / "runs" / "checkpoints" / PRIMARY_CK)
    base_bat = battery(net0)
    order = sorted(range(60), key=lambda i: -base_bat[i])
    primary4 = list(dict.fromkeys(order[:2]
                    + [i for i in order[2:] if install_occ[i][1] == "FLORIZEL"][:2]))[:4]
    replication2 = [i for i in order if i not in primary4][:2]
    donor_idx = primary4 + replication2
    log(f"donors {donor_idx} p_z "
        + " ".join(f"{base_bat[i]:.2f}" for i in donor_idx))

    def census(tag: str, ck_name: str, full: bool):
        net = net0 if ck_name == PRIMARY_CK else load(
            E43.REPO / "runs" / "checkpoints" / ck_name)
        wpe_orig = net.wpe.weight.data.clone()
        mean_row = wpe_orig.mean(0)
        w = net.wpe.weight.data

        base = battery(net)
        base_pz = float(np.mean(base))
        log(f"[{tag}] {ck_name}: base battery p(Z) {base_pz:.4f} "
            f"(params {net.num_params():,})")
        norm_ctx = {
            "row129_norm": float(wpe_orig[129].norm()),
            "row130_norm": float(wpe_orig[130].norm()),
            "mean_row_norm": float(mean_row.norm()),
            "mean_of_row_norms": float(wpe_orig.norm(dim=1).mean()),
        }

        pz_rows = []
        for r in range(256):
            w.copy_(wpe_orig)
            w[r] = mean_row
            pz_rows.append(float(np.mean(battery(net))))
        w.copy_(wpe_orig)
        drops = [base_pz - p for p in pz_rows]

        # null band: positions 130..255 are never fed (130-char contexts)
        null_max = max(drops[130:])
        null_mean = float(np.mean(drops[130:]))
        pos = np.array([d if d > 0 else 0.0 for d in drops])
        total_mass = float(pos.sum())
        top_idx = sorted(range(256), key=lambda r: -pos[r])[:SPARSE_MAX_ROWS]
        top_mass = float(sum(pos[r] for r in top_idx))
        frac = top_mass / total_mass if total_mass > 0 else 0.0
        verdict = ("SPARSE" if frac >= SPARSE_MASS_BAR else "DENSE/CLUSTERED")

        # narrowest contiguous window containing row 129 carrying >=80% mass
        target = SPARSE_MASS_BAR * total_mass
        best_win = None
        for a in range(0, 130):
            csum = 0.0
            for b in range(a, 130):
                csum += pos[b]
                if csum >= target:
                    if best_win is None or (b - a + 1) < (best_win[1] - best_win[0] + 1):
                        best_win = (a, b)
                    break
        win = {"span": list(best_win), "width": best_win[1] - best_win[0] + 1,
               "mass_in_window": float(pos[best_win[0]:best_win[1] + 1].sum())
               } if best_win else None

        # sorted cumulative mass curve
        cum = np.cumsum(np.sort(pos)[::-1]) / total_mass
        k80 = int((cum < SPARSE_MASS_BAR).sum()) + 1

        log(f"[{tag}] top-5 rows {top_idx} drops "
            + " ".join(f"{pos[r]:.3f}" for r in top_idx)
            + f" | top5 mass frac {frac:.3f} -> {verdict}")
        log(f"[{tag}] null band rows130-255: max {null_max:.4f} mean {null_mean:.4f}"
            + (f" | 80% window {win['span']} width {win['width']}" if win else "")
            + f" | rows-for-80% {k80}")

        # ---- zero-arm confirmation within +/-5 of the top-5 rows
        confirm_rows = sorted({q for r in top_idx for q in
                               range(max(0, r - 5), min(256, r + 6))})
        zero_pz = {}
        for r in confirm_rows:
            w.copy_(wpe_orig)
            w[r] = 0.0
            zero_pz[r] = base_pz - float(np.mean(battery(net)))
        w.copy_(wpe_orig)
        zc_top = {r: zero_pz[r] for r in top_idx}
        log(f"[{tag}] zero-arm on {len(confirm_rows)} rows: top5 "
            + " ".join(f"{r}:{zero_pz[r]:.3f}" for r in top_idx))

        # ---- secondary: d5 DeltaState vs relay_d5 (e066b method)
        secondary = None
        if full:
            a0 = []
            for i in donor_idx:
                xs, _ = fwd_states(net, ctx_ids[i])
                a0.append(xs[5][0, -1].clone())
            relay = torch.stack(a0).mean(0)
            secondary = {"relay_norm": float(relay.norm()), "rows": {}}
            delta_vecs = {}
            for r in top_idx:
                ds = []
                for i in donor_idx:
                    w.copy_(wpe_orig)
                    w[r] = mean_row
                    xs, _ = fwd_states(net, ctx_ids[i])
                    ds.append(xs[5][0, -1].clone())
                w.copy_(wpe_orig)
                dvec = torch.stack([ds[k] - a0[k] for k in range(len(donor_idx))]).mean(0)
                delta_vecs[r] = dvec
                secondary["rows"][r] = {
                    "cos_delta_relay_d5": cos(dvec, relay),
                    "cos_delta_wpe_row": cos(dvec, wpe_orig[r]),
                    "delta_norm": float(dvec.norm()),
                }
            keys = list(delta_vecs)
            pairs = [abs(cos(delta_vecs[i], delta_vecs[j]))
                     for n, i in enumerate(keys) for j in keys[n + 1:]]
            secondary["mean_abs_pairwise_cos_deltas"] = float(np.mean(pairs))
            for r in top_idx:
                s = secondary["rows"][r]
                log(f"[{tag}] top row {r}: cos(d5_delta, relay_d5) "
                    f"{s['cos_delta_relay_d5']:+.3f} cos(d5_delta, wpe[{r}]) "
                    f"{s['cos_delta_wpe_row']:+.3f}")
            log(f"[{tag}] mean |pairwise cos| between top-5 d5 deltas "
                f"{secondary['mean_abs_pairwise_cos_deltas']:.3f}")

        return {
            "checkpoint": ck_name, "params": net.num_params(),
            "base_pz": base_pz, "pz_rows": pz_rows, "drops": drops,
            "norm_context": norm_ctx,
            "null_band_130_255": {"max": null_max, "mean": null_mean},
            "total_positive_mass": total_mass,
            "top5_rows": top_idx, "top5_drops": [float(pos[r]) for r in top_idx],
            "top5_mass_fraction": frac, "verdict": verdict,
            "rows_for_80pct_mass": k80,
            "window_129_80pct": win,
            "zero_arm": {str(r): v for r, v in zero_pz.items()},
            "zero_arm_top5": {str(r): v for r, v in zc_top.items()},
            "secondary_d5": secondary,
        }

    # ---- primary: the e066 installed net (2.7M install-family)
    prim = census("prim", PRIMARY_CK, full=True)
    log(f"protocol check: base {prim['base_pz']:.4f} vs e066b A0 0.7149")

    # ---- replication: the 4x-dose install net (2.7M install-family)
    replica = None
    gate_net = load(E43.REPO / "runs" / "checkpoints" / REPLICA_CK)
    gate_pz = float(np.mean(battery(gate_net)))
    if gate_pz >= REPLICA_GATE:
        replica = census("repl", REPLICA_CK, full=True)
    else:
        log(f"replication SKIPPED: {REPLICA_CK} base p(Z) {gate_pz:.3f} "
            f"< gate {REPLICA_GATE}")
    repl_note = {
        "base_pz": gate_pz, "ran": replica is not None,
        "checkpoint_survey": (
            "all e043/e048 checkpoints are 2,739,072 params (2.7M); "
            "e048_repro (e066 primary) is itself 2.7M; e058's 2.7M nets "
            "e001.pt/e028_b43.pt carry NO install (base nets); "
            "e048_direct400/800 are direct-trained controls, not install "
            "family; e048_dose.pt (4x-dose Dmix install, same install-60 "
            "windows) is the only other full-strength install net"),
    }

    save_json(rd / "metrics.json", {
        "experiment": "e067_address_census",
        "registered": {
            "arm": "wpe row r <- mean(all 256 rows), in place, every r in 0..255",
            "sparse_rule": f"<={SPARSE_MAX_ROWS} rows carry >={SPARSE_MASS_BAR:.2f} "
                           "of total positive drop mass -> SPARSE else DENSE",
            "zero_arm": "rows within +/-5 of top-5",
            "replica_gate": REPLICA_GATE,
        },
        "donor_idx": donor_idx,
        "donor_pz": [base_bat[i] for i in donor_idx],
        "primary": prim, "replication_note": repl_note, "replication": replica,
    })

    # ---- figure: per-row drop curves + cumulative mass, both nets
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, (res, tag) in enumerate([(prim, "e048_repro (install)")]
                                     + ([(replica, "e048_dose (4x install)")]
                                        if replica else [])):
        d = np.array(res["drops"])
        ax = axes[row, 0]
        ax.plot(np.arange(256), d, lw=0.9, color="steelblue")
        ax.axvline(129, color="cyan", lw=1.2, ls=":")
        ax.axvline(130, color="cyan", lw=1.2, ls="--")
        ax.axhspan(-res["null_band_130_255"]["max"],
                   res["null_band_130_255"]["max"], color="gray", alpha=0.18,
                   label="null band (unused rows 130-255)")
        ax.plot(res["top5_rows"], [d[r] for r in res["top5_rows"]], "o",
                color="crimson", ms=5, label="top-5 rows")
        zr = sorted(int(k) for k in res["zero_arm"])
        ax.plot(zr, [res["zero_arm"][str(r)] for r in zr], "x", color="k",
                ms=4, lw=0.8, label="zero arm (±5 of top-5)")
        ax.set_xlabel("wpe row"); ax.set_ylabel("drop in battery p(Z)")
        ax.set_title(f"{tag}: base p(Z) {res['base_pz']:.3f} — "
                     f"{res['verdict']} (top5 mass {res['top5_mass_fraction']:.2f})")
        ax.legend(fontsize=7)
        pos = np.clip(d, 0, None)
        cum = np.cumsum(np.sort(pos)[::-1]) / max(pos.sum(), 1e-9)
        ax2 = axes[row, 1]
        ax2.plot(np.arange(1, 257), cum, lw=1.2, color="seagreen")
        ax2.axvline(SPARSE_MAX_ROWS, color="crimson", ls=":", lw=1)
        ax2.axhline(SPARSE_MASS_BAR, color="gray", ls="--", lw=1)
        ax2.set_xlabel("top-k rows (sorted by drop)")
        ax2.set_ylabel("cumulative positive drop-mass fraction")
        ax2.set_title(f"{tag}: rows needed for 80% mass = "
                      f"{res['rows_for_80pct_mass']}"
                      + (f"; 80% window {res['window_129_80pct']['span']} "
                         f"width {res['window_129_80pct']['width']}"
                         if res["window_129_80pct"] else ""))
        ax2.annotate(f"k80={res['rows_for_80pct_mass']}",
                     xy=(res["rows_for_80pct_mass"], SPARSE_MASS_BAR),
                     xytext=(res["rows_for_80pct_mass"] + 25, 0.62),
                     arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=9)
    if not replica:
        axes[1, 0].axis("off"); axes[1, 1].axis("off")
        axes[1, 0].text(0.5, 0.5, f"replication skipped (gate p(Z) "
                        f"{repl_note['base_pz']:.3f} < {REPLICA_GATE})",
                        ha="center", va="center", transform=axes[1, 0].transAxes)
    fig.suptitle("e067 P1-COORDINATE address census — per-wpe-row causal "
                 f"weight on installed p(Z) | primary verdict: "
                 f"{prim['verdict']}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(rd / "census.png", dpi=140)
    log(f"done -> {rd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
