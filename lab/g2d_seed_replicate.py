"""G2D — THE SEED REPLICATE (g2c's named last debt; T130 / R55's noun rule).

CONTEXT (T130): the onset-monitor organ MAINTAINS-IN-RHYTHM (cycle-median
0.615, duty 69%, the waveform mapped) — but at n=1: single seed (10902),
single root. R55's rule: "rhythm" is not a noun until seeds replicate.
One more run at two fresh wash-draw seeds, and the noun's fate is decided.

THE DELTA (exactly one number changes per cell): the wash-draw seed.
g2c's CELL-G2 rerun with seed 10903 and seed 10904 (e184's seed-replicate
convention: FREEZE_SEED + 1, + 2) — the SAME g2_root.pt, the SAME onset-only
monitor (this file imports g2b_onset_monitor, whose import patches
G2.G2Net = G2BNet, so the organ is reused, not retyped), the SAME protocol
(cue pool bit-gated against the e113 rebuild), the SAME dense measurement
grid (step 1 + every 2 steps + step 25 — g2c's grid verbatim). run_cell's
single Generator is seeded by this one number and draws every wash batch
(aj, rj) and every event replay batch (ix, aj, rj); the monitor, the
refractory, the rotation and the replay arithmetic are deterministic —
so each seed is a fresh realization of the SAME organ, which is exactly
what "seed-robust" must mean here.

Seed 10902's leg is NOT rerun: g2c's stored dense trace (fidelity-gated to
g2b's own realization — F_SCHED/F_TRACE/F_SD all PASS) is the reference
leg of the n=3 overlay, re-verified here only by recomputing its
cycle-median from the stored samples (F_REF).

REGISTERED BARS (dispatch g2d, VERBATIM; frozen — no shopping):
  RHYTHM-REPLICATES: "both new seeds show self-timed events (>=5) in/near
      the 20-45 band AND cycle-median >= 0.5 (the noun licensed — the
      self-maintaining rhythm is seed-robust)".
  RHYTHM-SEED-BOUND: "any seed fails either clause (the rhythm was a seed
      lottery — honest bound)".
Operationalization (frozen here, BEFORE the run; g2c's conventions):
  - EVENTS clause    = n_events >= 5 AND 100% of event spacings in the
                       registered band [20, 45] (g2b/g2c's own band);
  - CYCLE-MEDIAN clause = median of the ruler over all dense in-cycle
                       samples of the COMPLETE inter-event intervals
                       [e_k, e_{k+1}) >= 0.5 (g2c's cycle-median, same
                       grid, same pooling; pre-rhythm death and partial
                       tail co-reported, not adjudicated);
  - RHYTHM-REPLICATES fires iff BOTH new seeds pass BOTH clauses.
    Anything else = RHYTHM-SEED-BOUND. The dispatch's "in/near" is
    honored as a CO-REPORTED texture, never as a widened bar: if a seed
    passes the counts but shows an out-of-band spacing (or a median
    within noise of 0.5), that texture is recorded prominently under
    near_texture — the two registered outcomes stay exhaustive.
  - Seed 10902's clauses are read from g2c's stored adjudication (events
    10/10 in band, cycle-median 0.615) — embedded, not adjudicated again.

REGISTERED PREDICTION (before running): the closed loop (deterministic
monitor -> refractory -> replay resurrection) should dominate the wash-draw
lottery: predict 7-12 self-timed events per seed, spacings refractory-
bounded 24-40, cycle-median 0.5-0.7, duty 55-75%. THE OPEN RISK: g2c's
per-cycle troughs ranged 0.20-0.47 — the trough depth rides wash-draw luck,
so a seed whose dips deepen could land duty < 50% and cycle-median just
under the bar (TROUGH-BOUND's texture, one seed's worth). Genuinely open;
both outcomes pre-registered. DISCRIMINATING OBSERVATION: the per-cycle
trough distribution of the new seeds vs g2c's 0.20-0.47 — if the new
troughs deepen, the near-miss texture names the failure mode; if they hold,
the noun is licensed with room to spare.

CONSTRAINTS (dispatch): CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 before any g2/g2b
import), torch threads 4 (LOW — set by g2b's import), 20 s stagger between
the two trainings, g2's machinery REUSED (root loaded + bit-gated, cue pool
verified, run_cell unmodified), NO new checkpoint files (the measurement
states live in memory; the reusable artifacts remain g2_root/g2b's).

Outputs: runs/g2d/{metrics.json, seed_replicate.png}. No NOTES/THINKING/
QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python g2d_seed_replicate.py    (G2D_SMOKE=1 shakedown)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"      # CPU-ONLY (before g2/g2b import)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                # noqa: E402
import torch                                      # noqa: E402

import g2_rehearsal_organ as G2                   # noqa: E402 — the organ
import g2b_onset_monitor as G2B                   # noqa: E402 — REUSE: sets
                                                  # LOW threads (4) AND patches
                                                  # G2.G2Net = G2BNet (onset-only)
import common                                     # noqa: E402
from common import run_dir, save_json             # noqa: E402

assert torch.get_num_threads() == 4, "g2b's import must set LOW threads (4)"
assert G2.G2Net is G2B.G2BNet, "g2b's onset-only monitor patch must be live"

import matplotlib                                  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
import matplotlib.gridspec as gridspec             # noqa: E402

SMOKE = os.environ.get("G2D_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ---- frozen constants (inherited; restated for the record) ----------------------
FREEZE_SEED = G2.FREEZE_SEED                      # 10902 — the locked lineage
SEEDS: tuple[int, ...] = (10903, 10904)           # e184's replicate convention
RULER_GEO = 0                                      # g2_root.pt meta (frozen)
RULER_KEY = {-12: "gm12", 0: "g0", 12: "gp12"}[RULER_GEO]
SHUT_BAR, MAINTAIN_BAR = G2.SHUT_BAR, G2.MAINTAIN_BAR   # 0.27 / 0.50
SPACING_BAND = G2.SPACING_BAND                     # (20, 45)
EVENTS_MIN = 5                                     # the dispatch's >=5
G2D_CAP_S = 900.0                                  # g2c's cap (dense evals add
                                                  # wall time; train ~100 s)
N_STEPS = 300 if not SMOKE else 36
PHASE_EVERY = 2                                    # g2c's dense stride
# g2c's grid VERBATIM (step 1 + every 2 + step 25): identical pooling/convention
MEAS_GRID: tuple[int, ...] = tuple(sorted(
    set((1, 25) + tuple(range(PHASE_EVERY, N_STEPS + 1, PHASE_EVERY)))))
PHASE_BINS = 8                                     # g2c's phase-bin count
STAGGER_S = 20.0 if not SMOKE else 2.0            # the dispatch's CPU stagger
G2C_NAME = "g2c_smoke" if SMOKE else "g2c"
G2C_METRICS = G2.E43.REPO / "runs" / G2C_NAME / "metrics.json"

REGISTERED_BARS = {
    "rhythm_replicates": "RHYTHM-REPLICATES fires if both new seeds "
                         "(10903 AND 10904) show self-timed events (>=5) "
                         "in/near the 20-45 band AND cycle-median >= 0.5 "
                         "— the noun licensed (the self-maintaining rhythm "
                         "is seed-robust)",
    "rhythm_seed_bound": "RHYTHM-SEED-BOUND fires if any seed fails either "
                         "clause — the rhythm was a seed lottery (honest "
                         "bound)",
    "source": "dispatch g2d (T130 / R55's noun rule), VERBATIM; "
              "operationalized in this docstring before the run; no "
              "shopping — 'in/near' is co-reported texture, never a "
              "widened bar.",
}

deviations: list[str] = [
    "THE DELTA: the wash-draw seed only — g2c's CELL-G2 at seeds "
    "10903/10904 (e184's replicate convention). Root, cue pool, monitor "
    "(g2b's G2BNet via module import), protocol, replay/wash arithmetic, "
    "refractory and the dense measurement grid are g2c VERBATIM; run_cell "
    "is reused unmodified — the seed is the one changed argument.",
    "SEED 10902 NOT RERUN: g2c's stored dense trace (fidelity-gated to "
    "g2b's realization: F_SCHED/F_TRACE/F_SD all PASS) is the reference "
    "leg of the n=3; F_REF here only recomputes its cycle-median from the "
    "stored samples (a load-integrity check, not a re-measurement).",
    "CONTRAST EMBEDDED: g2b's stored CELL-BASE (ruler 0.0241 <= 0.27 at "
    "+50, CPU, seed 10902) is the wash-law contrast; its seed-generality "
    "is established by e184 (ALL-DISSOLVE, n=3 seeds 10902/10903/10904) — "
    "the un-gated organ dies at every seed tested.",
    "NO NEW CHECKPOINT FILES: the 152-per-cell measurement state_dicts "
    "live in memory and are discarded (deliverables are metrics + PNG; "
    "the reusable artifacts remain g2_root.pt and g2b's cells).",
    "TRAIN CAP: G2D_CAP_S=900 s (g2c's bound, inherited; train-only is "
    "~100 s/cell — the cap has never bound).",
    "Stagger: 20 s sleep between the two trainings (the dispatch's CPU "
    "stagger). Cooldowns skipped (CPU-only; g2b/g2c's convention).",
    "n=3 REALIZATIONS, ONE ROOT, ONE LINEAGE: this replicates the WASH-DRAW "
    "seed (the cell's RNG stream), not the install/root seed — the noun "
    "licensed is 'the organ's rhythm is wash-draw-robust on the locked "
    "root'; root-seed generality remains g-series-open (honesty).",
    "Smoke mode trims: 36-step cells, grid to 36, g2c reference from "
    "runs/g2c_smoke — nothing adjudicated (0 complete cycles).",
]


# ------------------------------------------------------------------ analysis
def phase_analysis(traj: dict, ev_steps: list[int]):
    """g2c's phase conventions VERBATIM, as a function: complete inter-event
    intervals pooled; cycle-median/mean/duty over all dense in-cycle
    samples; 8 phase bins; per-cycle stats."""
    dense = sorted(traj)

    def seg(a: int, b):
        ss = [s for s in dense if s >= a and (b is None or s < b)]
        return ss, [traj[s][RULER_KEY] for s in ss]

    cycles, pooled_v, pooled_ph = [], [], []
    for k in range(len(ev_steps) - 1):
        a, b = ev_steps[k], ev_steps[k + 1]
        ss, vs = seg(a, b)
        if not vs:
            continue
        phs = [(s - a) / (b - a) for s in ss]
        pooled_v += vs
        pooled_ph += phs
        cycles.append({
            "k": k + 1, "event_step": a, "next_event": b, "spacing": b - a,
            "n_samples": len(ss), "first_post_event": vs[0],
            "peak": max(vs), "trough": min(vs),
            "median": float(np.median(vs)), "mean": float(np.mean(vs)),
            "duty_ge_0.5": float(np.mean([v >= MAINTAIN_BAR for v in vs])),
            "last_before_next_event": vs[-1]})
    pre_ss, pre_v = seg(0, ev_steps[0] if ev_steps else N_STEPS + 1)
    tail_ss, tail_v = (seg(ev_steps[-1] + 1, None) if ev_steps else ([], []))
    binned = []
    for ib in range(PHASE_BINS):
        sel = [v for ph, v in zip(pooled_ph, pooled_v)
               if ib / PHASE_BINS <= ph < (ib + 1) / PHASE_BINS]
        binned.append({
            "center": round((ib + 0.5) / PHASE_BINS, 4), "n": len(sel),
            "mean": float(np.mean(sel)) if sel else None,
            "median": float(np.median(sel)) if sel else None})
    return {
        "n_complete_cycles": len(cycles), "n_pooled_samples": len(pooled_v),
        "cycle_median": (float(np.median(pooled_v)) if pooled_v else None),
        "cycle_mean": (float(np.mean(pooled_v)) if pooled_v else None),
        "duty_ge_0.5": (float(np.mean([v >= MAINTAIN_BAR for v in pooled_v]))
                        if pooled_v else None),
        "peak_max": max(pooled_v) if pooled_v else None,
        "trough_min": min(pooled_v) if pooled_v else None,
        "per_cycle": cycles,
        "per_cycle_peak_range": ([min(c["peak"] for c in cycles),
                                  max(c["peak"] for c in cycles)]
                                 if cycles else None),
        "per_cycle_median_range": ([min(c["median"] for c in cycles),
                                    max(c["median"] for c in cycles)]
                                   if cycles else None),
        "per_cycle_trough_range": ([min(c["trough"] for c in cycles),
                                    max(c["trough"] for c in cycles)]
                                   if cycles else None),
        "frac_cycles_median_ge_0.5": (
            float(np.mean([c["median"] >= MAINTAIN_BAR for c in cycles]))
            if cycles else None),
        "phase_bin_centers": [b["center"] for b in binned],
        "phase_bin_medians": [b["median"] for b in binned],
        "phase_bin_means": [b["mean"] for b in binned],
        "pre_rhythm": {"steps": [pre_ss[0], pre_ss[-1]] if pre_ss else None,
                       "min": min(pre_v) if pre_v else None},
        "tail_partial": {"n_samples": len(tail_ss),
                         "min": min(tail_v) if tail_v else None,
                         "max": max(tail_v) if tail_v else None,
                         "ruler_at_final_step": traj[N_STEPS][RULER_KEY]
                         if N_STEPS in traj else None},
    }


def clauses(n_events: int, spac: list[int], ph: dict) -> dict:
    """The two registered clauses, evaluated g2c-convention strictly."""
    band_ok = bool(spac) and all(SPACING_BAND[0] <= s <= SPACING_BAND[1]
                                 for s in spac)
    frac_band = (float(np.mean([SPACING_BAND[0] <= s <= SPACING_BAND[1]
                                for s in spac])) if spac else None)
    events_clause = bool(n_events >= EVENTS_MIN and band_ok)
    cm = ph["cycle_median"]
    median_clause = bool(cm is not None and cm >= MAINTAIN_BAR)
    near = []
    if n_events >= EVENTS_MIN and not band_ok:
        near.append(f"{sum(1 for s in spac if not SPACING_BAND[0] <= s <= SPACING_BAND[1])} "
                    f"of {len(spac)} spacings outside [20, 45] "
                    f"(max {max(spac) if spac else None})")
    if cm is not None and MAINTAIN_BAR - 0.05 <= cm < MAINTAIN_BAR:
        near.append(f"cycle-median {cm:.3f} within 0.05 of the bar")
    return {"n_events": n_events, "event_spacings": spac,
            "frac_spacing_in_20_45": frac_band,
            "band_stays_20_45": band_ok, "events_clause": events_clause,
            "cycle_median": cm, "cycle_mean": ph["cycle_mean"],
            "duty_ge_0.5": ph["duty_ge_0.5"],
            "median_clause": median_clause,
            "near_texture": near, "passes_both": bool(events_clause
                                                      and median_clause)}


# ------------------------------------------------------------------ main
def main():
    rd = run_dir("g2d_smoke" if SMOKE else "g2d")
    common.DEVICE = "cpu"
    G2.TRAIN_CAP_S = G2D_CAP_S          # read by G2.run_cell at call time
    log(f"G2D THE SEED REPLICATE (g2c's named last debt; CPU-only, threads "
        f"{torch.get_num_threads()}, cuda avail {torch.cuda.is_available()}) "
        f"-> {rd}")
    log(f"new wash-draw seeds {SEEDS} (root/monitor/protocol/grid: g2c "
        f"VERBATIM); grid {len(MEAS_GRID)} reads/cell")

    # ---- g2c's stored references (seed 10902's leg of the n=3) --------------
    g2cm = json.loads(G2C_METRICS.read_text(encoding="utf-8"))
    g2c_cell = g2cm["cell"]
    g2c_traj = {r["step"]: r for r in g2c_cell["traj_dense"]}
    g2c_events = list(g2c_cell["events"])
    g2c_spac = list(g2c_cell["event_spacings"])
    g2c_ph_stored = g2cm["adjudication"]["bars"]["cycle_median"]
    g2c_duty_stored = g2cm["adjudication"]["bars"]["duty_ge_0.5"]
    g2b_base50 = g2cm["g2b_references"]["contrast_base_ruler_50"]
    g2b_root_cells = g2cm["root"]["gates"]["G_ROOT"]["g2b_stored"]
    g2b_root_mon = g2b_root_cells["root_monitor_onset"]
    contrast = {"form": "g2b's stored CELL-BASE (CPU, seed 10902; embedded; "
                        "seed-generality per e184 ALL-DISSOLVE n=3)",
                "base_ruler_50": g2b_base50, "bar": SHUT_BAR,
                "pass": (None if g2b_base50 is None
                         else bool(g2b_base50 <= SHUT_BAR))}
    log(f"g2c references: events {g2c_events} ({len(g2c_events)}), stored "
        f"cycle-median {g2c_ph_stored}, duty {g2c_duty_stored}; "
        f"embedded contrast base@+50 {g2b_base50} vs {SHUT_BAR}: "
        + ("PASS" if contrast["pass"] else "N/A"))

    # seed 10902's clauses, read from g2c's stored adjudication + F_REF
    ph_ref = phase_analysis(g2c_traj, g2c_events)
    both_none = ph_ref["cycle_median"] is None and g2c_ph_stored is None
    F_REF = {"form": "g2c's stored dense trace re-pooled (load integrity)",
             "cycle_median_recomputed": ph_ref["cycle_median"],
             "g2c_stored": g2c_ph_stored,
             "abs_diff": (None if both_none else
                          abs(ph_ref["cycle_median"] - g2c_ph_stored)),
             "n_dense_rows": len(g2c_traj)}
    F_REF["pass"] = bool(both_none or F_REF["abs_diff"] < 1e-9)
    if both_none:
        F_REF["note"] = "smoke: 0 complete cycles on both sides — vacuous"
    log(f"F_REF (seed 10902 leg): recomputed cycle-median "
        f"{ph_ref['cycle_median']} vs stored {g2c_ph_stored}: "
        f"{'PASS' if F_REF['pass'] else 'FAIL'}")
    assert F_REF["pass"], "g2c's stored dense trace failed to re-pool"
    ref_clauses = clauses(len(g2c_events), g2c_spac, ph_ref)
    ref_clauses["source"] = ("g2c's stored realization (fidelity-gated to "
                             "g2b's own); embedded reference leg, not "
                             "re-adjudicated")

    # ---- protocol + root (REUSE — g2b/g2c's own builders and bit-gates) -----
    P = G2B.rebuild_protocol()
    ck = torch.load(G2.CKPT_DIR / ("smoke_g2_root.pt" if SMOKE
                                   else "g2_root.pt"),
                    map_location="cpu", weights_only=False)
    root_sd = G2.organ_body_sd(ck["model"])
    cue_pool = ck["model"]["cue_pool"].long()
    cue_mask = ck["model"]["cue_mask"]

    G_POOL = {"form": "bit",
              "pool_bit_identical": bool(torch.equal(cue_pool,
                                                     P["jit_pool_x"])),
              "mask_bit_identical": bool(torch.equal(cue_mask,
                                                     P["jit_pool_mask"]))}
    G_POOL["pass"] = bool(G_POOL["pool_bit_identical"]
                          and G_POOL["mask_bit_identical"])
    assert G_POOL["pass"], f"G_POOL FAILED: {G_POOL}"

    net_root = G2B.G2BNet(G2.evl_load(root_sd), cue_pool, cue_mask)
    root_mon = net_root.monitor(0, dev=CPU)
    re_cells = {j: G2.battery_cell(G2.evl_load(root_sd), P["bat_ids"][j],
                                   P["zid"])["mean_pz"] for j in G2.GEOS}
    root_diffs = {"gm12": re_cells[-12] - g2b_root_cells["g-12"],
                  "g0": re_cells[0] - g2b_root_cells["g+0"],
                  "gp12": re_cells[12] - g2b_root_cells["g+12"],
                  "root_monitor_onset": root_mon - g2b_root_mon}
    G_ROOT = {"form": "reuse-verify vs g2b's own reloaded dials (threads 4)",
              "cells_reloaded": {f"g{j:+d}": re_cells[j] for j in G2.GEOS},
              "root_monitor_onset": root_mon,
              "g2b_stored": {**g2b_root_cells},
              "diffs": root_diffs,
              "max_abs_diff": max(abs(v) for v in root_diffs.values()),
              "bit_tol": G2.G_BIT_TOL, "tol": G2.G_FALLBACK_TOL}
    G_ROOT["bit"] = bool(G_ROOT["max_abs_diff"] < G2.G_BIT_TOL)
    G_ROOT["pass"] = bool(G_ROOT["max_abs_diff"] < G2.G_FALLBACK_TOL)
    log(f"G_POOL bit: PASS | G_ROOT (vs g2b's reloaded): max|diff| "
        f"{G_ROOT['max_abs_diff']:.2e}: "
        + ("PASS" if G_ROOT["pass"] else "FAIL")
        + (" (bit)" if G_ROOT["bit"] else "")
        + f" | root onset monitor {root_mon:.4f}")
    if not SMOKE:
        assert G_ROOT["pass"], "G_ROOT FAILED — environment drifted since g2b"

    # ---- THE TWO CELLS (the delta: the seed argument of run_cell) -----------
    per_seed: dict = {}
    for si, seed in enumerate(SEEDS):
        log("=" * 78)
        if si:
            log(f"stagger {STAGGER_S:.0f}s (CPU-only dispatch)")
            time.sleep(STAGGER_S)
        log(f"CELL G2 SEED {seed} (g2c's CELL-G2 verbatim — onset-only "
            f"monitor, {N_STEPS} steps, dense grid {len(MEAS_GRID)} reads)")
        cell = G2.run_cell(
            "g2", root_sd, cue_pool, cue_mask, P["anchor_neutral"],
            P["train_ids"], P["itos"], P["r_eval_xy"], P["bat_ids"],
            P["zid"], seed, MEAS_GRID)
        ev_steps = [e["step"] for e in cell["event_log"]]
        traj = {r["step"]: r for r in cell["traj"]}
        ph = phase_analysis(traj, ev_steps)
        cl = clauses(cell["n_events"], cell["event_spacings"], ph)
        gates = {
            "G_NAMEFREE": {"cell_zeph_violations": cell["zeph_violations"],
                           "pass": bool(cell["zeph_violations"] == 0)},
            "G_REPLAY": {"n_events": cell["n_events"],
                         "checks": cell["replay_checks"],
                         "pass": bool(cell["n_events"] == 0 or
                                      (cell["replay_checks"]["mask7_ok"]
                                       == cell["n_events"]
                                       and cell["replay_checks"]
                                       ["anchors_8_8_ok"]
                                       == cell["n_events"]))},
            "G_STEP": {"steps_ran": cell["steps_ran"],
                       "expected": N_STEPS, "batch": 32,
                       "pass": bool(cell["steps_ran"] == N_STEPS)},
        }
        per_seed[str(seed)] = {
            "cell": {"mode": "g2", "seed": seed,
                     "steps_ran": cell["steps_ran"],
                     "n_events": cell["n_events"],
                     "n_checks": cell["n_checks"],
                     "realized_r": cell["realized_r"],
                     "event_spacings": cell["event_spacings"],
                     "events": ev_steps,
                     "monitor_trace": cell["monitor_trace"],
                     "event_log": cell["event_log"],
                     "traj_dense": cell["traj"],
                     "devices": {"initial": cell["initial_device"],
                                 "final": cell["final_device"]},
                     "replay_checks": cell["replay_checks"]},
            "phase": ph, "clauses": cl, "gates": gates,
        }
        del cell["sds"]                # measurement states are not artifacts
        log(f"SEED {seed}: {cell['n_events']} events {ev_steps}, spacings "
            f"{cell['event_spacings']}; cycle-median "
            f"{ph['cycle_median'] if ph['cycle_median'] is not None else 'NA'}, "
            f"duty {ph['duty_ge_0.5'] if ph['duty_ge_0.5'] is not None else 'NA'}, "
            f"troughs {ph['per_cycle_trough_range']}; clauses: events "
            f"{cl['events_clause']}, median {cl['median_clause']}")

    hard = {"G_POOL": G_POOL["pass"], "G_ROOT": G_ROOT["pass"],
            "CONTRAST": contrast["pass"], "F_REF": F_REF["pass"]}
    for s in per_seed:
        for g in per_seed[s]["gates"]:
            hard[f"{g}@{s}"] = per_seed[s]["gates"][g]["pass"]
    bad = [k for k, v in hard.items() if not v]
    if bad and not SMOKE:
        raise RuntimeError(f"gate(s) FAILED: {bad}")

    # schedules must differ across seeds (else the seed did not bite)
    scheds = {str(FREEZE_SEED): g2c_events} | {
        str(s): per_seed[str(s)]["cell"]["events"] for s in SEEDS}
    SCHEDS_DIFFER = {"schedules": scheds,
                     "all_distinct": bool(len({tuple(v) for v in
                                               scheds.values()})
                                          == len(scheds))}
    log(f"SCHEDULES across seeds all distinct: "
        f"{SCHEDS_DIFFER['all_distinct']} ({scheds})")

    # =====================================================================
    # ADJUDICATION (registered bars; no shopping)
    # =====================================================================
    new_pass = {s: per_seed[str(s)]["clauses"]["passes_both"] for s in SEEDS}
    RHYTHM_REPLICATES = bool(all(new_pass.values())
                             and contrast["pass"])
    RHYTHM_SEED_BOUND = not RHYTHM_REPLICATES

    def per_seed_str(seed: int) -> str:
        cl = per_seed[str(seed)]["clauses"]
        ph = per_seed[str(seed)]["phase"]
        return (f"seed {seed}: {cl['n_events']} events (spacings "
                f"{cl['event_spacings']}, "
                f"{cl['frac_spacing_in_20_45']:.0%} in 20-45), CYCLE-MEDIAN "
                f"{cl['cycle_median'] if cl['cycle_median'] is None else round(cl['cycle_median'], 3)}, "
                f"mean {cl['cycle_mean'] if cl['cycle_mean'] is None else round(cl['cycle_mean'], 3)}, "
                f"duty {cl['duty_ge_0.5'] if cl['duty_ge_0.5'] is None else round(cl['duty_ge_0.5'], 2)}, "
                f"per-cycle peaks {ph['per_cycle_peak_range']}, troughs "
                f"{ph['per_cycle_trough_range']}, medians "
                f"{ph['per_cycle_median_range']}"
                + (f" — NEAR-TEXTURE: {'; '.join(cl['near_texture'])}"
                   if cl["near_texture"] else "")
                + f" -> clauses {'PASS' if cl['passes_both'] else 'FAIL'}")

    if SMOKE:
        verdict = "SMOKE (nothing adjudicated)"
        clause = "smoke trim — no complete cycles."
    elif RHYTHM_REPLICATES:
        verdict = ("RHYTHM-REPLICATES (the noun licensed — the "
                   "self-maintaining rhythm is seed-robust)")
        clause = (f"{per_seed_str(SEEDS[0])}; {per_seed_str(SEEDS[1])}; "
                  f"reference seed {FREEZE_SEED} (g2c): {ref_clauses['n_events']} "
                  f"events, cycle-median "
                  f"{ref_clauses['cycle_median']:.3f}. BOTH new seeds pass "
                  f"both clauses — the self-timed sawtooth (event count, "
                  f"spacing band, cycle-median, duty) reproduces under "
                  f"fresh wash-draw streams on the same root: per R55's "
                  f"rule, 'rhythm' is now a licensed noun — the organ's "
                  f"closed loop, not a seed lottery.")
    else:
        failed = [s for s in SEEDS if not new_pass[s]]
        verdict = ("RHYTHM-SEED-BOUND (the rhythm was a seed lottery — "
                   "honest bound)")
        clause = (f"{per_seed_str(SEEDS[0])}; {per_seed_str(SEEDS[1])}; "
                  f"reference seed {FREEZE_SEED} (g2c): cycle-median "
                  f"{ref_clauses['cycle_median']:.3f}. SEED(S) {failed} "
                  f"failed a registered clause — the maintain-in-rhythm "
                  f"finding does not generalize across wash-draw seeds; "
                  f"'rhythm' stays a one-seed description (honest bound, "
                  f"recorded per R55's rule; near-texture above).")
    log("=" * 78)
    log(f"G2D VERDICT: {verdict}")
    log(f"  {clause}")
    log("=" * 78)

    # =====================================================================
    # PLOT — A: the n=3 dense ruler overlay; B: cycle shapes aligned by
    # phase (per-seed binned medians); C: per-cycle medians + clause numbers
    # =====================================================================
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.25, 1.0])
    axA = fig.add_subplot(gs[0, :])
    seed_colors = {str(FREEZE_SEED): "#1f77b4", str(SEEDS[0]): "#d62728",
                   str(SEEDS[1]): "#ff7f0e"}
    seed_styles = {str(FREEZE_SEED): ("-", 1.6, "g2c stored (seed 10902)"),
                   str(SEEDS[0]): ("-", 1.4, f"g2d seed {SEEDS[0]}"),
                   str(SEEDS[1]): ("-", 1.4, f"g2d seed {SEEDS[1]}")}
    traces = {str(FREEZE_SEED): (g2c_traj, g2c_events)}
    for s in SEEDS:
        traces[str(s)] = ({r["step"]: r for r in per_seed[str(s)]["cell"]
                           ["traj_dense"]},
                          per_seed[str(s)]["cell"]["events"])
    for skey, (tr, evs) in traces.items():
        ls, lw, lab = seed_styles[skey]
        xs = sorted(tr)
        axA.plot(xs, [tr[x][RULER_KEY] for x in xs], ls, lw=lw,
                 color=seed_colors[skey], zorder=3, label=lab)
        for e in evs:
            axA.axvline(e, color=seed_colors[skey], ls="-", lw=0.7,
                        alpha=0.30, zorder=1)
    axA.axhline(MAINTAIN_BAR, color="k", ls="--", lw=0.8, alpha=0.6)
    axA.axhline(SHUT_BAR, color="k", ls=":", lw=0.8, alpha=0.6)
    axA.text(N_STEPS + 2, MAINTAIN_BAR, " maintain 0.5", va="bottom",
             fontsize=8)
    axA.text(N_STEPS + 2, SHUT_BAR, " die 0.27", va="bottom", fontsize=8)
    axA.axhline(g2cm["adjudication"]["ruler"]["root_value"], color="#2ca02c",
                lw=0.8, alpha=0.5)
    axA.set_xlabel("wash step")
    axA.set_ylabel(f"ruler g{RULER_GEO:+d} mean p(Z)")
    axA.set_title(f"G2D THE SEED REPLICATE — n=3 waveform overlay "
                  f"(vlines = self-timed events, color-matched; verdict: "
                  f"{verdict.split(' (')[0]})")
    axA.legend(loc="center right", fontsize=8)
    axA.set_xlim(0, N_STEPS + 14)

    axB = fig.add_subplot(gs[1, 0])
    for skey, (tr, evs) in traces.items():
        dense = sorted(tr)
        for k in range(len(evs) - 1):
            a, b = evs[k], evs[k + 1]
            ss = [x for x in dense if a <= x < b]
            axB.plot([(x - a) / (b - a) for x in ss],
                     [tr[x][RULER_KEY] for x in ss], "-", color="#cccccc",
                     lw=0.7, alpha=0.8, zorder=2)
        ph = (ph_ref if skey == str(FREEZE_SEED)
              else per_seed[skey]["phase"])
        meds = ph["phase_bin_medians"]
        if any(m is not None for m in meds):
            axB.plot(ph["phase_bin_centers"], meds, "o-",
                     color=seed_colors[skey], lw=2.0, ms=4, zorder=4,
                     label=f"seed {skey} binned MEDIAN "
                           f"(cyc-med {ph['cycle_median']:.3f})")
    axB.axhline(MAINTAIN_BAR, color="k", ls="--", lw=0.8)
    axB.set_xlabel("cycle phase (0 = event step, post-replay; 1 = next event)")
    axB.set_ylabel(f"ruler g{RULER_GEO:+d}")
    axB.set_title("THE CYCLE'S SHAPE per seed (gray = individual cycles)")
    axB.legend(fontsize=7, loc="upper right")

    axC = fig.add_subplot(gs[1, 1])
    plotted_c = False
    for skey in (str(FREEZE_SEED),) + tuple(str(s) for s in SEEDS):
        ph = (ph_ref if skey == str(FREEZE_SEED)
              else per_seed[skey]["phase"])
        cl = (ref_clauses if skey == str(FREEZE_SEED)
              else per_seed[skey]["clauses"])
        if not ph["per_cycle"]:
            continue
        plotted_c = True
        ks = [c["k"] for c in ph["per_cycle"]]
        axC.plot(ks, [c["median"] for c in ph["per_cycle"]], "o-",
                 color=seed_colors[skey], lw=1.3, ms=5,
                 label=f"seed {skey} "
                       f"({'PASS' if cl['passes_both'] else 'FAIL'})")
        axC.plot(ks, [c["trough"] for c in ph["per_cycle"]], "v--",
                 color=seed_colors[skey], lw=0.7, ms=4, alpha=0.6)
    if not plotted_c:
        axC.text(0.5, 0.5, "no complete cycles", ha="center", va="center")
    axC.axhline(MAINTAIN_BAR, color="k", ls="--", lw=0.8)
    axC.axhline(SHUT_BAR, color="k", ls=":", lw=0.8)
    axC.set_xlabel("cycle (event k -> event k+1)")
    axC.set_ylabel(f"ruler g{RULER_GEO:+d} (o = median, v = trough)")
    axC.set_title("per-cycle medians and troughs — the three seeds")
    axC.legend(fontsize=8, loc="center right")

    fig.tight_layout()
    png = rd / "seed_replicate.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    log(f"[plot] {png}")

    # =====================================================================
    # metrics.json
    # =====================================================================
    metrics = {
        "experiment": "g2d",
        "date": common.now_iso(),
        "purpose": "THE SEED REPLICATE — g2c's named last debt (T130): the "
                   "onset-monitor organ's self-maintaining rhythm re-run at "
                   "two additional wash-draw seeds (10903, 10904) so the "
                   "noun's fate is decided per R55's rule; g2c's cell "
                   "machinery verbatim, only the seed argument of run_cell "
                   "changes; g2c's stored dense trace is seed 10902's leg.",
        "delta_vs_g2c": {
            "g2c": "one realization: seed 10902, dense grid, cycle-median "
                   "0.615 / duty 69% (n=1 seed, n=1 root)",
            "g2d": "seeds 10903 + 10904 (e184's replicate convention); "
                   "identical root, monitor, protocol, measurement grid, "
                   "and adjudication conventions",
            "the_seed": "run_cell's single Generator draws every wash batch "
                        "(aj, rj) and event replay batch (ix, aj, rj); the "
                        "monitor/refractory/rotation are deterministic — "
                        "each seed is a fresh realization of the SAME organ",
        },
        "smoke": SMOKE,
        "threads": torch.get_num_threads(),
        "cpu_only": True,
        "cfg": g2cm["cfg"],
        "compute": {"cpu_only": True, "cuda_visible_devices": "-1",
                    "threads": torch.get_num_threads(),
                    "stagger_s": STAGGER_S, "train_cap_s": G2D_CAP_S,
                    "meas_grid_n": len(MEAS_GRID),
                    "meas_grid_every": PHASE_EVERY},
        "g2c_references": {"metrics": f"runs/{G2C_NAME}/metrics.json",
                           "events": g2c_events,
                           "cycle_median_stored": g2c_ph_stored,
                           "duty_stored": g2c_duty_stored,
                           "verdict_then": g2cm["adjudication"]["verdict"],
                           "contrast_base_ruler_50": g2b_base50},
        "root": {"source": "runs/checkpoints/"
                 + ("smoke_g2_root.pt" if SMOKE else "g2_root.pt")
                 + " (REUSED — not rebuilt; g2b/g2c's own root)",
                "gates": {"G_POOL": G_POOL, "G_ROOT": G_ROOT}},
        "reference_seed": {
            "seed": FREEZE_SEED,
            "source": "g2c's stored dense realization (fidelity-gated to "
                      "g2b's: F_SCHED/F_TRACE/F_SD PASS); embedded leg",
            "F_REF": F_REF,
            "clauses": ref_clauses,
            "phase": ph_ref,
            "events": g2c_events,
            "event_log": g2c_cell["event_log"]},
        "per_seed": per_seed,
        "schedules_differ": SCHEDS_DIFFER,
        "gates": {"CONTRAST": contrast,
                  **{f"{g}@{s}": per_seed[str(s)]["gates"][g]
                     for s in SEEDS for g in per_seed[str(s)]["gates"]}},
        "registered_bars": REGISTERED_BARS,
        "registered_prediction": {
            "pre_run": "the closed loop (deterministic monitor -> "
                       "refractory -> replay resurrection) should dominate "
                       "the wash-draw lottery: 7-12 events/seed, spacings "
                       "24-40, cycle-median 0.5-0.7, duty 55-75%. OPEN "
                       "RISK: g2c's per-cycle troughs ranged 0.20-0.47 — a "
                       "seed whose dips deepen could land duty < 50% and "
                       "median just under the bar (both outcomes "
                       "pre-registered; no shopping).",
            "discriminating_observation": "the new seeds' per-cycle trough "
                                          "distributions vs g2c's 0.20-0.47 "
                                          "— deepening troughs name the "
                                          "failure mode; holding troughs "
                                          "license the noun with room.",
        },
        "adjudication": {
            "ruler": {"geo": RULER_GEO, "key": RULER_KEY,
                      "root_value": g2cm["adjudication"]["ruler"]
                                    ["root_value"],
                      "die_bar": SHUT_BAR, "maintain_bar": MAINTAIN_BAR},
            "clauses": {
                "events_clause": "n_events >= 5 AND 100% of spacings in "
                                 "[20, 45] (g2b/g2c's registered band)",
                "median_clause": "cycle-median over dense in-cycle samples "
                                 "of complete inter-event intervals >= 0.5 "
                                 "(g2c's convention, same grid)",
                "per_seed_passes_both": {**{str(FREEZE_SEED):
                                            ref_clauses["passes_both"]},
                                         **new_pass}},
            "bars": {"RHYTHM_REPLICATES": RHYTHM_REPLICATES,
                     "RHYTHM_SEED_BOUND": RHYTHM_SEED_BOUND,
                     "near_texture": {str(s): per_seed[str(s)]["clauses"]
                                      ["near_texture"] for s in SEEDS}},
            "verdict": verdict, "clause": clause,
        },
        "honesty": [
            "WHAT WAS REPLICATED: the WASH-DRAW seed (the cell's RNG "
            "stream) at n=3 realizations on ONE root (g2_root.pt, the "
            "locked lineage) — the licensed noun is 'the organ's rhythm "
            "is wash-draw-robust on this root'; root-seed/install-seed "
            "generality remains open (the g-series' standing single-root "
            "bound, same as g2b/g2c's own honesty clause).",
            "SEED 10902's LEG IS STORED, NOT RERUN: g2c's dense trace was "
            "fidelity-gated to g2b's realization; F_REF re-pools it "
            f"(max|diff| {F_REF['abs_diff']}) as a load check only.",
            "THE CONTRAST IS EMBEDDED: g2b's CELL-BASE (seed 10902) + "
            "e184's ALL-DISSOLVE n=3 for the wash law's seed-generality; "
            "no un-gated cell was rerun here.",
            "NO BAR WIDENING: the dispatch's 'in/near' is co-reported "
            "texture (out-of-band spacings, medians within 0.05 of the "
            "bar); the two registered outcomes stay exhaustive.",
            "n=2 NEW SEEDS, ONE PER VERDICT CLAUSE PAIR: a 2-of-2 pass "
            "licenses the noun at the lab's established replicate standard "
            "(e184/e187's n=2 new seeds + n=1 reference); it does not "
            "measure the pass RATE — recorded as the bound it is.",
        ],
        "checkpoints_saved": {},
        "deviations": deviations,
        "timing_s": round(time.time() - T0, 1),
    }
    save_json(rd / "metrics.json", G2.E43.jsonable(metrics))
    log(f"[done] metrics -> {rd / 'metrics.json'} "
        f"({metrics['timing_s']:.0f}s total); verdict: {verdict}")


if __name__ == "__main__":
    main()
