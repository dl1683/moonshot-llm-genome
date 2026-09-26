"""E087 — T049 two-rig RIF adjudication (registered 2026-09-25, run CPU-only).

THE QUESTION (verbatim from THINKING.md T049, CONFLICT REGISTERED block):
two rigs disagree on the EL-to-FL cell — e081b's rig fires (+0.083, CIs
excluding 0, anagram-dead on e048_repro) while e081's own cell was sub-bar
(+0.046, CI crossing 0) with its decomposition attributing the modulations
to bigram priming and Z-char induction. T049's verdict is UNRESOLVED pending
this adjudication.

REGISTERED META-BAR (frozen in T049): the EL->FL real-name effect exceeds
its anagram control by >= 0.05 with CI excluding 0 under BOTH rigs on BOTH
nets -> RIF-real (fact-level reading writes); anything less -> 'string-level
induction only' (the e081 reading).

DESIGN (T049-registered): both rigs' EL->FL legs on BOTH nets
(e048_repro + e048_dose) at B=96 resamples, with the anagram control
(ZIBLETHEA) AND a Z-bearing-string control on every leg.
  Arms per (rig, net) leg, probe battery always FL (29 windows):
    real    : read ELIZABETH   -> probe FLORIZEL
    sham    : read a name-free 112-token corpus segment (the null arm)
    anagram : read ZIBLETHEA   (e081's EL anagram: same unigrams as
              ELIZABETH, no "EL"/"FL" bigrams, Z-initial)
    zctrl   : read ZABMOTHIC   (NEW Z-bearing control: 9 chars, Z-initial
              like the anagram, NOT an anagram of ELIZABETH — letters
              {Z,A,B,M,O,T,H,I,C} vs {E,L,I,Z,A,B,E,T,H}: E,E,L swapped
              for M,O,C — no F, no L at all, no "EL"/"FL" bigrams; same
              length so the prefix geometry is identical, ctx_len 47)
  Registered statistic per leg: the four deltas are p(sham)-p(read) per
  window (e081's delta_cell verbatim); the META-BAR statistic is
    gap_w = P_anagram.mean_w - P_real.mean_w   (per window),
  i.e. delta_real - delta_anagram with the sham cancelling EXACTLY per
  window (the gap is sham-free: immune to sham-stream differences, the
  main rig confound). Fires iff mean gap >= 0.05 AND the per-window 95%
  t-CI excludes 0 AND the clustered bootstrap CI (segments AND windows
  resampled, 2000 draws — e081's deviation-8 convention) excludes 0
  (strict: both CIs; the t-only and boot-only variants reported beside).
  Report-only: gap(real, zctrl) and gap(anagram, zctrl) — does the
  anagram behave like any Z-bearing string (e081's reading) or like sham?

THE TWO RIGS, MADE PRECISE (documented deviation from T049's shorthand
"different rig"): e081b imports e081's machinery verbatim, so once B is
matched at 96 the rigs differ ONLY in their frozen RNG streams (which
elicitation occurrences and which sham segments get drawn) — which is
exactly the adjudication question: does the cell depend on the resample
stream? Seed streams (order fixes the draws; elicitation_prefix is called
once per resample in index order in both rigs):
  RIG A = e081_rig_probe,  base 8100 + net offset (dose +0, repro +5000):
    readEL seed 8100+200+off  (e081 cell index 2; B=32 there -> first 32
                               draws of our B=96 replicate e081's cell)
    sham   8100+300+off       (e081 index 3, B=32 -> first 32 replicate)
    scrEL  8100+1300+off      (e081 index 13, B=16 -> first 16 replicate)
    scrZC  8100+1400+off      (NEW index 14, clean stream)
  RIG B = e081b_rif_replication, base 8150 + net offset (repro +0 per
  e081b; dose +5000 NEW — e081b never ran dose; mirrors e081's per-net
  offset convention so no stream collides):
    readEL 8150+100+off       (e081b index 1, B=32 -> first 32 replicate)
    sham   8150+300+off       (e081b index 3, B=96 -> FULL replicate on
                               repro; this IS e081b's tight sham)
    scrEL  8150+500+off       (e081b index 5, B=32 -> first 32 replicate)
    scrZC  8150+600+off       (NEW index 6, clean stream)

DEVIATIONS (registered before compute):
  1. Compute envelope: e087 needs 16 arms x 96 resamples = 1536 session
     batches; e081/e081b pace (~2.3 s/batch) projects ~60 min, over the
     30-min single-step rule. FIX: run_cell_batched() below stacks EIGHT
     resamples (232 sessions) per forward kernel batch. Protocol-identical:
     the elicitation_prefix RNG discipline is called once per resample in
     index order with the SAME seed (streams replicate the originals
     exactly); only the kernel batch shape differs. HONESTY CHECK (in-run,
     before any leg): one 6-resample cell run through e081's verbatim
     run_cell and through run_cell_batched with the same throwaway seed
     must agree within 1e-6 on every onset probability (asserted).
  2. Z-control arm is NEW (T049 says "a Z-bearing-string control" without
     fixing the string): ZABMOTHIC chosen and frozen here — see above for
     the constraint set (Z-initial as the anagram, length-matched, not an
     anagram, no name bigrams, minimal FLORIZEL letter overlap).
  3. Rig B on dose uses a new +5000 offset (documented above).
  4. Only the EL->FL direction is run (the T049 conflict cell; FL->EL and
     Z-probe legs are out of scope — the meta-bar is EL->FL-specific).
  5. Canonical baselines FL/EL/Z @129 run per net for the instrument
     p(Z)@129 gate (|drift| < 5e-3 vs published) and the shift-cost
     honesty number; they consume no RNG (prefix-free single pass).

Outputs: runs/e087/{metrics.json, rif_adjudication.png}.
Run: python lab/e087_rif_adjudication.py
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                    # CPU experiment

import random as _random                                    # noqa: E402
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import torch.nn.functional as F                              # noqa: E402

torch.set_num_threads(min(16, os.cpu_count() or 8))

import common                                               # noqa: E402
common.DEVICE = "cpu"
from common import run_dir, save_json                       # noqa: E402
import e043_install as E43                                  # noqa: E402
import e081_rif_probe as E81                                # noqa: E402
from e081_rif_probe import (arm_stats, build_pools,         # noqa: E402
                            delta_cell, elicitation_prefix, load,
                            rebuild_protocol, run_cell, stats)
import matplotlib                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ---- registered constants --------------------------------------------------
B = 96                       # resamples per arm (T049: B=96 on every leg)
CI_Z = E81.CI_Z              # 1.96
N_BOOT = E81.N_BOOT          # 2000 clustered bootstrap draws
GAP_BAR = 0.05               # T049 frozen meta-bar (absolute, onset scale)
ZCONTROL = "ZABMOTHIC"       # deviation 2: the Z-bearing-string control
E81.SCRAMBLES["ZC"] = ZCONTROL   # extends e081's scramble table in-process
CKPTS = {"e048_dose": "e048_dose.pt", "e048_repro": "e048_repro.pt"}
NET_ORDER = ("e048_dose", "e048_repro")
ARMS = ("EL", "sham", "scrEL", "scrZC")            # probe is always FL
ARM_LABELS = {"EL": "real ELIZABETH", "sham": "sham",
              "scrEL": "anagram ZIBLETHEA", "scrZC": f"zctrl {ZCONTROL}"}
RIGS = {
    "rigA_e081": {"base": 8100,
                  "net_off": {"e048_dose": 0, "e048_repro": 5000},
                  "idx": {"EL": 2, "sham": 3, "scrEL": 13, "scrZC": 14}},
    "rigB_e081b": {"base": 8150,
                   "net_off": {"e048_dose": 5000, "e048_repro": 0},
                   "idx": {"EL": 1, "sham": 3, "scrEL": 5, "scrZC": 6}},
}
RIG_ORDER = ("rigA_e081", "rigB_e081b")

REGISTERED = {
    "question": ("T049 CONFLICT REGISTERED: e081b's rig fires on the EL->FL "
                 "cell (+0.083, CIs excluding 0) while e081's own cell was "
                 "sub-bar (+0.046, CI crossing 0) with character-level "
                 "decomposition — adjudicate at B=96 with anagram + "
                 "Z-bearing controls on every leg."),
    "meta_bar": ("the EL->FL real-name effect exceeds its anagram control by "
                 ">= 0.05 with CI excluding 0 (per-window t-CI AND clustered "
                 "bootstrap CI) under BOTH rigs on BOTH nets -> RIF-real "
                 "(fact-level reading writes); anything less -> 'string-"
                 "level induction only' (the e081 reading)"),
    "statistic": ("gap_w = P_anagram.mean_w - P_real.mean_w = "
                  "(delta_real - delta_anagram) per window; the sham cancels "
                  "exactly, so the meta-bar statistic is sham-free"),
    "arms": {a: ARM_LABELS[a] for a in ARMS},
    "zcontrol_definition": ("ZABMOTHIC: 9 chars, Z-initial (as ZIBLETHEA), "
                            "NOT an anagram of ELIZABETH (E,E,L -> M,O,C), "
                            "no F/no L, no 'EL'/'FL' bigrams, length-matched "
                            "so prefix geometry is identical (ctx_len 47)"),
    "resamples": B,
    "rigs": {r: {"seed_base": RIGS[r]["base"],
                 "net_offsets": RIGS[r]["net_off"],
                 "cell_indices": RIGS[r]["idx"],
                 "source": ("e081_rif_probe CELLS order" if r == "rigA_e081"
                            else "e081b_rif_replication CELLS order; "
                                 "dose offset +5000 new (deviation 3)")}
             for r in RIG_ORDER},
    "stream_provenance": ("first 32 (EL, rigA/B) / 16 (scrEL, rigA) / all 96 "
                          "(sham, rigB repro) draws replicate the original "
                          "cells' RNG streams exactly"),
    "deviations": [
        "1. compute envelope: 8 resamples (232 sessions) per kernel batch "
        "(run_cell_batched, below); in-run equivalence assert vs verbatim "
        "run_cell within 1e-6 on a 6-resample smoke cell (same throwaway "
        "seed both paths)",
        "2. Z-control string fixed here (ZABMOTHIC) — constraint set in "
        "module docstring",
        "3. rigB on dose: new +5000 stream offset",
        "4. EL->FL only (the T049 conflict cell); FL->EL and Z-probe legs "
        "out of scope",
        "5. canonical baselines per net for the p(Z)@129 instrument gate "
        "and shift-cost honesty; no RNG consumed",
    ],
}
log("REGISTERED: " + " | ".join(f"{k}: {v}" for k, v in REGISTERED.items()))


# ------------------------------------------------------------------ machinery
@torch.no_grad()
def run_cell_batched(net, corpus, train_text, read, probe_occ, pool, rng,
                     n_res=B, bs=232):
    """Deviation-1 vectorized run_cell: identical sessions and RNG
    discipline (elicitation_prefix called once per resample, in index
    order, from the same seed), 8 resamples stacked per kernel batch.
    Returns the same dict shape as e081's run_cell (shifted=True)."""
    name = E81.NAME1                                   # probe: FLORIZEL
    name_ids = [corpus.stoi[c] for c in name]
    row = E81.ROW_SHIFT
    ctx_ids = [corpus.encode(train_text[p - E81.PRE: p]) for p, _ in probe_occ]
    sep_ids = corpus.encode(E81.SEP)
    n_win = len(probe_occ)
    pres = [elicitation_prefix(train_text, corpus, read, pool, rng)
            for _ in range(n_res)]                     # same draw order
    sess = [torch.cat([pre, sep_ids, c, corpus.encode(name)])
            for pre in pres for c in ctx_ids]
    L = sess[0].shape[0]
    assert L <= net.cfg.block_size, L
    P = np.zeros((n_res, n_win), dtype=np.float64)
    J = np.zeros((n_res, n_win), dtype=np.float64)
    tgt = torch.tensor(name_ids).view(1, -1, 1)
    idx = np.arange(n_res * n_win)
    r_idx, w_idx = idx // n_win, idx % n_win
    for i in range(0, len(sess), bs):
        x = torch.stack(sess[i: i + bs])
        lg, _ = net(x)
        pr = F.softmax(lg[:, row: row + len(name), :], -1)
        pnm = torch.gather(pr, 2, tgt.expand(pr.shape[0], -1, 1)).squeeze(-1)
        k = slice(i, min(i + bs, len(sess)))
        P[r_idx[k], w_idx[k]] = pnm[:, 0].numpy()
        J[r_idx[k], w_idx[k]] = pnm.prod(-1).numpy()
    return {"P": P, "J": J, "sess_len": int(L), "row": row,
            "read": read, "probe": "FL"}


@torch.no_grad()
def equivalence_check(net, corpus, train_text, probe_occ, pool):
    """Deviation-1 honesty assert: verbatim run_cell vs run_cell_batched on
    the same throwaway seed (777001 — NOT one of the frozen leg seeds)."""
    r1 = _random.Random(777001)
    c1 = run_cell(net, corpus, train_text, "EL", "FL", probe_occ, pool, r1,
                  shifted=True, n_res=6)
    r2 = _random.Random(777001)
    c2 = run_cell_batched(net, corpus, train_text, "EL", probe_occ, pool, r2,
                          n_res=6)
    md = float(np.max(np.abs(c1["P"] - c2["P"])))
    log(f"equivalence check (verbatim vs batched, 6 resamples): "
        f"max |dP| {md:.2e}")
    assert md < 1e-6, f"batched runner diverges from verbatim run_cell: {md}"
    return md


def gap_cell(cell_sup, cell_ref, seed):
    """META-BAR statistic: per-window paired gap between two READ arms.
    gap_w = P_sup.mean_w - P_ref.mean_w  (positive = the 'sup' read
    suppresses the probe more than the 'ref' read). Sham-free by
    construction. Clustered bootstrap (segments AND windows, e081's
    deviation-8 convention) + per-window t-CI."""
    A, Bc = cell_sup["P"], cell_ref["P"]
    g = A.mean(axis=0) - Bc.mean(axis=0)
    s = stats(g)
    s["frac_windows_gap_positive"] = float((g > 0).mean())
    s["per_window_gap"] = g.tolist()
    Ba, Bb, n = A.shape[0], Bc.shape[0], A.shape[1]
    brng = np.random.default_rng(seed)
    ha, hb = max(1, Ba // 2), max(1, Bb // 2)
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        a_ = brng.integers(0, Ba, ha)
        b_ = brng.integers(0, Bb, hb)
        w_ = brng.integers(0, n, n)
        draws[i] = A[np.ix_(a_, w_)].mean() - Bc[np.ix_(b_, w_)].mean()
    s["gap_bootstrap_ci95"] = [float(np.percentile(draws, 2.5)),
                               float(np.percentile(draws, 97.5))]
    return s


def load_hist():
    """Prior cells (read-only) for side-by-side provenance."""
    out = {}
    p = E43.REPO / "runs" / "e081" / "metrics.json"
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        out["e081"] = {ck: {k: {"mean": d[k]["mean"], "ci95": d[k]["ci95"],
                                "boot_ci95": d[k]["delta_bootstrap_ci95"]}
                            for k in ("cross_EL_to_FL", "scr_EL_to_FL")
                            if k in d}
                       for ck, d in m["deltas_suppression"].items()}
    p = E43.REPO / "runs" / "e081b" / "metrics.json"
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        d = m["deltas"]
        out["e081b"] = {"e048_repro": {k: {"mean": d[k]["mean"],
                                           "ci95": d[k]["ci95"],
                                           "boot_ci95":
                                               d[k]["delta_bootstrap_ci95"]}
                                     for k in ("cross_EL_to_FL",
                                               "scr_EL_to_FL") if k in d}}
    return out


# ------------------------------------------------------------------ main
def main():
    rd = run_dir("e087")
    log("E087 (T049 adjudication): two rigs x two nets x {real, sham, "
        "anagram, Z-control} at B=96 — eval-only, no weight edits")

    corpus, train_text, install_occ, probes, mix, held_mix = rebuild_protocol()
    pools, pool_flags = build_pools(train_text, install_occ, probes)
    pool = pools[("EL", "FL")]          # elicitation pool for every read arm
    log(f"protocol rebuilt (verbatim e081): install60 {mix}, held30 "
        f"{held_mix}; FL probe battery {len(probes['FL'])} windows; "
        f"EL elicitation pool {len(pool)}")

    results, gates = {}, {}
    for ck_name in NET_ORDER:
        net = load(E43.REPO / "runs" / "checkpoints" / CKPTS[ck_name])
        w_sig = (net.wpe.weight.data.clone(), net.wte.weight.data.clone())
        log(f"net {ck_name}: params {net.num_params():,}")

        # deviation-1 honesty check (throwaway seed; frozen streams intact)
        eq = equivalence_check(net, corpus, train_text, probes["FL"], pool)

        # canonical baselines @129: instrument Z gate + shift-cost honesty
        base = {}
        for pk in ("FL", "EL", "Z"):
            rng = _random.Random(8100 + 900 + {"FL": 3, "EL": 1, "Z": 7}[pk])
            c = run_cell(net, corpus, train_text, "sham", pk, probes[pk],
                         None, rng, shifted=False)
            base[pk] = c
            a = arm_stats(c)
            log(f"[{ck_name}] baseline probe-{pk} @129: p(first) "
                f"{a['onset']['mean']:.4f} "
                f"[{a['onset']['ci95'][0]:.4f},{a['onset']['ci95'][1]:.4f}]")
        gz = base["Z"]["P"].mean()
        drift = abs(gz - E81.Z_GATE[ck_name])
        gates[ck_name] = {"pz_canonical": float(gz),
                          "pz_published": E81.Z_GATE[ck_name],
                          "drift": float(drift), "pass": bool(drift < 5e-3),
                          "equivalence_max_dP": eq}
        assert gates[ck_name]["pass"], (f"{ck_name} Z gate drift {drift:.2e} "
                                        f"(protocol identity failed)")
        log(f"[{ck_name}] instrument gate: p(Z)@129 {gz:.7f} vs published "
            f"{E81.Z_GATE[ck_name]:.7f} (|drift| {drift:.2e}, OK)")

        # the 8 arms: 2 rigs x 4 read strings, B=96 each, frozen streams
        cells = {}
        for rig in RIG_ORDER:
            spec = RIGS[rig]
            off = spec["net_off"][ck_name]
            for arm in ARMS:
                seed = spec["base"] + spec["idx"][arm] * 100 + off
                rng = _random.Random(seed)
                c = run_cell_batched(net, corpus, train_text, arm,
                                     probes["FL"], pool, rng, n_res=B)
                cells[(rig, arm)] = c
                a = arm_stats(c)
                log(f"[{ck_name}/{rig}] read-{arm:5s} -> probe-FL @243: "
                    f"p(first) {a['onset']['mean']:.4f} "
                    f"[{a['onset']['ci95'][0]:.4f},{a['onset']['ci95'][1]:.4f}] "
                    f"joint {a['joint_mean']:.2e} (seed {seed}, "
                    f"{a['n_sessions']} sessions)")
        assert torch.equal(net.wpe.weight.data, w_sig[0]) and \
            torch.equal(net.wte.weight.data, w_sig[1]), "weights changed!"
        log(f"[{ck_name}] no-weight-edit check passed")
        results[ck_name] = {"base": base, "cells": cells}

    # ---- deltas, gaps, per-leg verdicts
    legs, leg_order = {}, []
    for ck_name in NET_ORDER:
        for rig in RIG_ORDER:
            lid = f"{rig}/{ck_name}"
            leg_order.append(lid)
            cells = results[ck_name]["cells"]
            sham_c, real_c = cells[(rig, "sham")], cells[(rig, "EL")]
            ana_c, zc_c = cells[(rig, "scrEL")], cells[(rig, "scrZC")]
            d = {"real": delta_cell(sham_c, real_c),
                 "anagram": delta_cell(sham_c, ana_c),
                 "zctrl": delta_cell(sham_c, zc_c)}
            gseed = 6160 + leg_order.index(lid) * 10
            g = {"meta_real_minus_anagram": gap_cell(ana_c, real_c, gseed),
                 "real_minus_zctrl": gap_cell(zc_c, real_c, gseed + 1),
                 "anagram_minus_zctrl": gap_cell(zc_c, ana_c, gseed + 2)}
            mg = g["meta_real_minus_anagram"]
            fire_t = bool(mg["mean"] >= GAP_BAR and mg["ci95"][0] > 0)
            fire_b = bool(mg["mean"] >= GAP_BAR
                          and mg["gap_bootstrap_ci95"][0] > 0)
            # sham placebo split (48 vs 48) — noise-floor honesty
            Ps = sham_c["P"]
            h = Ps.shape[0] // 2
            placebo = float((Ps[:h].mean(axis=0) - Ps[h:2 * h].mean(axis=0))
                            .mean())
            shift = {pk: float(results[ck_name]["base"][pk]["P"].mean()
                               - sham_c["P"].mean()) for pk in ("FL", "Z")}
            legs[lid] = {
                "net": ck_name, "rig": rig,
                "seeds": {arm: RIGS[rig]["base"]
                          + RIGS[rig]["idx"][arm] * 100
                          + RIGS[rig]["net_off"][ck_name] for arm in ARMS},
                "deltas": d, "gaps": g,
                "meta_gap_fire_t": fire_t, "meta_gap_fire_boot": fire_b,
                "meta_gap_fire": bool(fire_t and fire_b),
                "sham_placebo_48v48": placebo,
                "shift_cost_base_minus_sham243": shift,
            }
            log(f"[{lid}] real {d['real']['mean']:+.4f} boot "
                f"{[round(x, 4) for x in d['real']['delta_bootstrap_ci95']]} | "
                f"anagram {d['anagram']['mean']:+.4f} boot "
                f"{[round(x, 4) for x in d['anagram']['delta_bootstrap_ci95']]} | "
                f"zctrl {d['zctrl']['mean']:+.4f} boot "
                f"{[round(x, 4) for x in d['zctrl']['delta_bootstrap_ci95']]}")
            log(f"[{lid}] META GAP (real - anagram): {mg['mean']:+.4f} "
                f"t-CI [{mg['ci95'][0]:+.4f},{mg['ci95'][1]:+.4f}] boot "
                f"[{mg['gap_bootstrap_ci95'][0]:+.4f},"
                f"{mg['gap_bootstrap_ci95'][1]:+.4f}] "
                f"({mg['frac_windows_gap_positive']:.0%} windows +) -> "
                f"fire_t {fire_t} fire_boot {fire_b} | sham placebo 48v48 "
                f"{placebo:+.4f}")

    # ---- META-VERDICT (T049 frozen rule)
    firing = [lid for lid in leg_order if legs[lid]["meta_gap_fire"]]
    if len(firing) == len(leg_order):
        meta = ("RIF-REAL (fact-level reading writes): the EL->FL real-name "
                "effect exceeds its anagram control by >= 0.05 with CIs "
                "excluding 0 under BOTH rigs on BOTH nets — the T049 frozen "
                "meta-bar is met on all four legs")
    else:
        meta = ("STRING-LEVEL INDUCTION ONLY (the e081 reading): the T049 "
                f"meta-bar fails on {len(leg_order) - len(firing)} of "
                f"{len(leg_order)} legs (firing: {firing if firing else 'none'})")
    log(f"META-VERDICT: {meta}")

    # ---- metrics.json
    def cells_json(cells):
        return {f"{rig}__read{arm}": {**arm_stats(c), "row": c["row"],
                                      "sess_len": c["sess_len"],
                                      "P_flat": c["P"].flatten().tolist()}
                for (rig, arm), c in cells.items()}

    out = {
        "experiment": "e087_rif_adjudication",
        "t049_source": "THINKING.md T049, CONFLICT REGISTERED block",
        "registered": REGISTERED,
        "protocol": {
            "machinery": "e081_rif_probe imported verbatim (elicitation_"
                         "prefix, run_cell baselines, delta_cell, batteries); "
                         "run_cell_batched per deviation 1",
            "corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
            "install_mix": mix, "held_mix": held_mix,
            "fl_probe_battery": len(probes["FL"]),
            "el_elicitation_pool": len(pool),
            "prefix_len": E81.PREFIX, "separator": E81.SEP,
            "readout_row": E81.ROW_SHIFT,
            "resamples": B, "bootstrap_draws": N_BOOT,
            "arms": {a: ARM_LABELS[a] for a in ARMS},
            "zcontrol": ZCONTROL,
            "pool_flags": pool_flags,
        },
        "gates_instrument": gates,
        "legs": {lid: {k: ({kk: {vvk: vvv for vvk, vvv in vv.items()
                                 if vvk != "per_window_delta"}
                            for kk, vv in v.items()} if k == "deltas" else
                          ({kk: {vvk: vvv for vvk, vvv in vv.items()
                                 if vvk != "per_window_gap"}
                            for kk, vv in v.items()} if k == "gaps" else v))
                  for k, v in legs[lid].items()}
                 for lid in leg_order},
        "cells": {ck: {"base": {pk: {**arm_stats(c),
                                      "P_flat": c["P"].flatten().tolist()}
                                for pk, c in r["base"].items()},
                       "shifted": cells_json(r["cells"])}
                  for ck, r in results.items()},
        "meta_verdict": {"rule": REGISTERED["meta_bar"], "firing_legs": firing,
                         "n_firing": len(firing), "n_legs": len(leg_order),
                         "verdict": meta},
        "historical_reference": load_hist(),
        "elapsed_s": round(time.time() - T0, 1),
        "cpu_threads": torch.get_num_threads(),
    }
    save_json(rd / "metrics.json", out)

    # ---- figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    short = {lid: lid.replace("rigA_e081", "rigA").replace("rigB_e081b",
                                                           "rigB")
             .replace("e048_", "") for lid in leg_order}

    # (0,0) the table: real / anagram / Z-control deltas vs sham, boot CIs
    ax = axes[0, 0]
    xs = np.arange(len(leg_order))
    bw = 0.25
    cols = {"real": "crimson", "anagram": "steelblue", "zctrl": "darkorange"}
    for j, key in enumerate(("real", "anagram", "zctrl")):
        vals = [legs[lid]["deltas"][key]["mean"] for lid in leg_order]
        lo = [legs[lid]["deltas"][key]["mean"]
              - legs[lid]["deltas"][key]["delta_bootstrap_ci95"][0]
              for lid in leg_order]
        hi = [legs[lid]["deltas"][key]["delta_bootstrap_ci95"][1]
              - legs[lid]["deltas"][key]["mean"] for lid in leg_order]
        ax.bar(xs + (j - 1) * bw, vals, bw * 0.92, yerr=[lo, hi], capsize=3,
               color=cols[key], edgecolor="k", linewidth=0.4,
               label=ARM_LABELS[{"real": "EL", "anagram": "scrEL",
                                 "zctrl": "scrZC"}[key]])
        for x, v in zip(xs + (j - 1) * bw, vals):
            ax.text(x, v + 0.004, f"{v:+.3f}", ha="center", fontsize=6.4,
                    rotation=90, va="bottom")
    ax.axhline(0, color="k", lw=1)
    ax.axhline(E81.SUPP_BAR, color="seagreen", ls="--", lw=1.3,
               label=f"e081 supp bar +{E81.SUPP_BAR}")
    ax.set_xticks(xs)
    ax.set_xticklabels([short[lid] for lid in leg_order], fontsize=8.5)
    ax.set_ylabel("delta: p(sham) - p(read)  [positive = read suppresses FL]")
    ax.set_title("Per-leg deltas vs sham (EL->FL, B=96, clustered bootstrap "
                 "CIs)", fontsize=10)
    ax.legend(fontsize=7.5, ncols=2)

    # (0,1) THE META-BAR: gap(real - anagram) per leg
    ax = axes[0, 1]
    vals = [legs[lid]["gaps"]["meta_real_minus_anagram"]["mean"]
            for lid in leg_order]
    lo = [legs[lid]["gaps"]["meta_real_minus_anagram"]["mean"]
          - legs[lid]["gaps"]["meta_real_minus_anagram"]["gap_bootstrap_ci95"][0]
          for lid in leg_order]
    hi = [legs[lid]["gaps"]["meta_real_minus_anagram"]["gap_bootstrap_ci95"][1]
          - legs[lid]["gaps"]["meta_real_minus_anagram"]["mean"]
          for lid in leg_order]
    cols_ = ["seagreen" if legs[lid]["meta_gap_fire"] else "crimson"
             for lid in leg_order]
    ax.bar(xs, vals, 0.5, yerr=[lo, hi], capsize=4, color=cols_,
           edgecolor="k", linewidth=0.5)
    for x, lid in zip(xs, leg_order):
        mg = legs[lid]["gaps"]["meta_real_minus_anagram"]
        ax.plot([x - 0.3, x + 0.3], [mg["ci95"][0], mg["ci95"][0]],
                color="k", lw=1.6, ls=":")
        ax.text(x, vals[list(xs).index(x)] + 0.006,
                f"{mg['mean']:+.3f}\nt[{mg['ci95'][0]:+.3f},"
                f"{mg['ci95'][1]:+.3f}]", ha="center", fontsize=7)
    ax.axhline(GAP_BAR, color="seagreen", ls="--", lw=1.6,
               label=f"T049 meta-bar {GAP_BAR} (dotted whisker = t-CI lower)")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([short[lid] for lid in leg_order], fontsize=8.5)
    ax.set_ylabel("gap: delta(real ELIZABETH) - delta(anagram ZIBLETHEA)")
    ax.set_title(f"META-BAR PANEL — positive = the real name suppresses FL "
                 f"more than its anagram\ngreen = leg fires (>= 0.05, both "
                 f"CIs exclude 0); red = fails; firing "
                 f"{len(firing)}/{len(leg_order)}", fontsize=10)
    ax.legend(fontsize=8)

    # (1,0) per-window gap texture + report-only control gaps
    ax = axes[1, 0]
    for i, lid in enumerate(leg_order):
        g = np.array(legs[lid]["gaps"]["meta_real_minus_anagram"]
                     ["per_window_gap"])
        ax.scatter(np.full(len(g), i) + np.random.uniform(-0.16, 0.16,
                                                          len(g)),
                   g, s=14, color="crimson" if "rigA" in lid else "steelblue",
                   alpha=0.65, edgecolor="k", linewidth=0.3)
        ax.hlines(g.mean(), i - 0.28, i + 0.28, color="k", lw=1.6)
    ax.axhline(0, color="k", lw=1)
    ax.axhline(GAP_BAR, color="seagreen", ls="--", lw=1.3)
    ax.set_xticks(range(len(leg_order)))
    ax.set_xticklabels([short[lid] for lid in leg_order], fontsize=8.5)
    ax.set_ylabel("per-window gap: P(anagram) - P(real)")
    ax.set_title("Per-window meta-gap texture (29 FL windows per leg; black "
                 "bar = mean)", fontsize=10)
    rep = legs[leg_order[-1]]["gaps"]
    ax.set_xlabel(f"report-only control gaps (last leg): real-vs-zctrl "
                  f"{rep['real_minus_zctrl']['mean']:+.4f} | "
                  f"anagram-vs-zctrl "
                  f"{rep['anagram_minus_zctrl']['mean']:+.4f}", fontsize=8)

    # (1,1) what actually moved: onset means per arm + honesty
    ax = axes[1, 1]
    for j, (arm_key, arm_read) in enumerate((("EL", "readEL"),
                                             ("sham", "readsham"),
                                             ("scrEL", "readscrEL"),
                                             ("scrZC", "readscrZC"))):
        vals = [arm_stats(results[legs[lid]["net"]]["cells"][
            (legs[lid]["rig"], arm_key)])["onset"]["mean"] for lid in leg_order]
        errs = [CI_Z * arm_stats(results[legs[lid]["net"]]["cells"][
            (legs[lid]["rig"], arm_key)])["onset"]["sem"]
            for lid in leg_order]
        ax.bar(xs + (j - 1.5) * 0.19, vals, 0.18, yerr=errs, capsize=2,
               color={"readEL": "crimson", "readsham": "gray",
                      "readscrEL": "steelblue",
                      "readscrZC": "darkorange"}[arm_read],
               edgecolor="k", linewidth=0.4,
               label={"readEL": "read ELIZABETH (real)",
                      "readsham": "sham",
                      "readscrEL": "read ZIBLETHEA (anagram)",
                      "readscrZC": f"read {ZCONTROL} (Z-ctrl)"}[arm_read])
    ax.set_xticks(xs)
    ax.set_xticklabels([short[lid] for lid in leg_order], fontsize=8.5)
    ax.set_ylabel("onset p(first FLORIZEL char) @ row 243")
    pl = {short[lid]: round(legs[lid]["sham_placebo_48v48"], 4)
          for lid in leg_order}
    ax.set_title(f"Arm onset means (what actually moved)\nhonesty: sham "
                 f"placebo 48v48 per leg {pl}; shift cost FL base@129 - "
                 f"sham@243 "
                 f"{ {short[lid]: round(legs[lid]['shift_cost_base_minus_sham243']['FL'], 3) for lid in leg_order} }",
                 fontsize=8.5)
    ax.legend(fontsize=7, ncols=2)
    ax.set_ylim(0, 1.02)

    fig.suptitle(f"e087 T049 two-rig RIF adjudication — META-VERDICT: "
                 f"{'RIF-REAL' if len(firing) == len(leg_order) else 'STRING-LEVEL INDUCTION ONLY'} "
                 f"({len(firing)}/{len(leg_order)} legs fire)\n"
                 f"both rigs x both nets x {{real, sham, anagram, Z-control}} "
                 f"at B=96; eval-only, no weight edits", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(rd / "rif_adjudication.png", dpi=140)
    log(f"done -> {rd}")

    # ---- stdout table (the report)
    print("\n==== E087 PER-RIG PER-NET TABLE (EL->FL, B=96, probe FL n=29) ====")
    for lid in leg_order:
        L = legs[lid]
        print(f"\n--- {lid} (seeds {L['seeds']}) ---")
        for key in ("real", "anagram", "zctrl"):
            s = L["deltas"][key]
            print(f"  delta {key:8s} [{ARM_LABELS[{'real': 'EL', 'anagram': 'scrEL', 'zctrl': 'scrZC'}[key]]:20s}] "
                  f"{s['mean']:+.4f}  t[{s['ci95'][0]:+.4f},{s['ci95'][1]:+.4f}]  "
                  f"boot[{s['delta_bootstrap_ci95'][0]:+.4f},{s['delta_bootstrap_ci95'][1]:+.4f}]  "
                  f"({s['frac_windows_suppressed']:.0%} windows supp)")
        mg = L["gaps"]["meta_real_minus_anagram"]
        print(f"  META GAP real-anagram {mg['mean']:+.4f}  "
              f"t[{mg['ci95'][0]:+.4f},{mg['ci95'][1]:+.4f}]  "
              f"boot[{mg['gap_bootstrap_ci95'][0]:+.4f},{mg['gap_bootstrap_ci95'][1]:+.4f}]  "
              f"({mg['frac_windows_gap_positive']:.0%} windows +)  "
              f"FIRE {L['meta_gap_fire']} (t {L['meta_gap_fire_t']}, "
              f"boot {L['meta_gap_fire_boot']})  sham placebo48v48 "
              f"{L['sham_placebo_48v48']:+.4f}")
    print(f"\nMETA-VERDICT ({len(firing)}/{len(leg_order)} legs fire): {meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
