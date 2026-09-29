"""G2C — THE PHASE-OFFSET REPLICATE (g2b's owed cell; T128's registered follow-up).

CONTEXT (T128): g2b's onset-only monitor organ fired 10 self-timed events
(spacings 24-36, 100% in the 20-45 band) but the frozen +300 checkpoint
sampled a sawtooth TROUGH (0.305 vs the 0.5 bar; the peaks reach 0.62-0.72;
every event resurrects from near-zero) — "+300 rides cycle phase ... a
phase-offset replicate is the owed cell". The registered follow-up:
checkpoints at several PHASE OFFSETS relative to the event cycle, so the
verdict does not ride one phase.

THE DELTA (measurement-only — the organ is UNTOUCHED): g2c re-runs g2b's
CELL-G2 (same g2_root.pt, same onset-only monitor via g2b's own G2BNet —
this file imports g2b_onset_monitor, whose import patches G2.G2Net, so the
monitor is reused, not retyped — same wash/replay economy, same seed 10902)
with the ruler measured on a DENSE grid (every 2 steps + step 1 = 151 reads
instead of g2b's 9+9). run_cell's measurements are no-RNG no-grad reads of
a CPU eval twin; the training stream draws only from the seeded generator,
so the grid cannot touch the dynamics — the rerun reproduces g2b's exact
realization and the dense trace IS g2b's trajectory phase-resolved. Three
gates prove it before anything adjudicates:
  F_SCHED — the event schedule equals g2b's stored log (24,56,...,284);
  F_TRACE — the ruler at g2b's stored checkpoint steps matches (bit 5e-6 /
            fallback 0.05, g2's own tolerances) + monitor values/fired flags;
  F_SD    — the final state_dict bit-diffs vs runs/checkpoints/g2b_g2.pt
            (and +50 vs g2b_g2_s50.pt).
If the realization ever diverged (environment drift), the gates SAY SO and
the bars adjudicate on the rerun's own dense trace (a same-seed twin), with
the divergence recorded — never silently.

REGISTERED BARS (dispatch g2c, VERBATIM; frozen — no shopping):
  MAINTAINS-IN-RHYTHM: "fires if the ruler's CYCLE-median (or the mean over
      phase-offset checkpoints) >= 0.5 AND the event band stays 20-45 — the
      organ vindicated as a self-maintaining memory in oscillation".
  TROUGH-BOUND: "fires if even the phase-median stays < 0.5 (the sawtooth's
      duty cycle is too low — the organ maintains peaks but not
      expression-in-general)".
Operationalization (frozen here, BEFORE the run):
  - cycles = the COMPLETE inter-event intervals [e_k, e_{k+1}) of the
    self-timed schedule (the pre-rhythm death before e_1 and the partial
    tail after the last event are co-reported, not adjudicated);
  - CYCLE-median = median of the ruler over all dense in-cycle samples;
    "mean over phase-offset checkpoints" = the mean over the same samples
    (the 8 phase-bin means and their average are co-reported — identical
    by uniform construction);
  - duty cycle = fraction of in-cycle dense samples >= 0.5;
  - "the event band stays 20-45" = 100% of the rerun's event spacings in
    [20, 45] (g2b: 9/9).
  MAINTAINS-IN-RHYTHM = (cycle_median >= 0.5 OR cycle_mean >= 0.5) AND band.
  TROUGH-BOUND        = (cycle_median < 0.5 AND cycle_mean < 0.5).
  A band break with the ruler clause passing adjudicates the ruler clause
  and flags the band (recorded; no third bar invented).

CONSTRAINTS (dispatch): CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 before any g2
import), torch threads 4 (LOW — inherited from g2b's import), g2b's
checkpoints/machinery REUSED: the root is loaded and bit-gated (G_POOL,
G_ROOT vs g2b's own reloaded dials), g2b's stored CELL-BASE contrast is
embedded (0.0241 <= 0.27 at +50 — passed in g2b, not rerun), and NO new
checkpoint files are written (the measurement states live in memory only).

Outputs: runs/g2c/{metrics.json, phase_replicate.png}. No NOTES/THINKING/
QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python g2c_phase_replicate.py    (G2C_SMOKE=1 shakedown)
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
import g2b_onset_monitor as G2B                   # noqa: E402 — REUSE: this
                                                  # import sets LOW threads (4)
                                                  # AND patches G2.G2Net = G2BNet
import common                                     # noqa: E402
from common import run_dir, save_json             # noqa: E402

assert torch.get_num_threads() == 4, "g2b's import must set LOW threads (4)"
assert G2.G2Net is G2B.G2BNet, "g2b's onset-only monitor patch must be live"

import matplotlib                                  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
import matplotlib.gridspec as gridspec             # noqa: E402

SMOKE = os.environ.get("G2C_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ---- frozen constants (inherited; restated for the record) ----------------------
FREEZE_SEED = G2.FREEZE_SEED                      # 10902 — the locked lineage
RULER_GEO = 0                                      # g2_root.pt meta (frozen)
RULER_KEY = {-12: "gm12", 0: "g0", 12: "gp12"}[RULER_GEO]
SHUT_BAR, MAINTAIN_BAR = G2.SHUT_BAR, G2.MAINTAIN_BAR   # 0.27 / 0.50
SPACING_BAND = G2.SPACING_BAND                     # (20, 45)
G2C_CAP_S = 900.0                                  # dense evals add wall time
                                                  # (train-only ~100 s; g2b's
                                                  # own 600 s cap kept headroom)
N_STEPS = 300 if not SMOKE else 36
PHASE_EVERY = 2                                    # the dense grid's stride
# every 2 steps + g2b's two odd checkpoint steps (1, 25) so F_TRACE covers
# EVERY stored g2b point (the rest of g2b's grid is even and subsumed)
MEAS_GRID: tuple[int, ...] = tuple(sorted(
    set((1, 25) + tuple(range(PHASE_EVERY, N_STEPS + 1, PHASE_EVERY)))))
PHASE_BINS = 8                                     # the phase-offset grid
G2B_NAME = "g2b_smoke" if SMOKE else "g2b"
G2B_METRICS = G2.E43.REPO / "runs" / G2B_NAME / "metrics.json"
G2B_CK_FINAL = G2.CKPT_DIR / ("smoke_g2b_g2.pt" if SMOKE else "g2b_g2.pt")
G2B_CK_S50 = G2.CKPT_DIR / ("smoke_g2b_g2_s50.pt" if SMOKE
                            else "g2b_g2_s50.pt")

REGISTERED_BARS = {
    "maintains_in_rhythm": "MAINTAINS-IN-RHYTHM fires if the ruler's "
                           "CYCLE-median (or the mean over phase-offset "
                           "checkpoints) >= 0.5 AND the event band stays "
                           "20-45 — the organ vindicated as a self-maintaining "
                           "memory in oscillation",
    "trough_bound": "TROUGH-BOUND fires if even the phase-median stays < 0.5 "
                    "(the sawtooth's duty cycle is too low — the organ "
                    "maintains peaks but not expression-in-general)",
    "source": "dispatch g2c (T128's registered follow-up), VERBATIM; "
              "operationalized in this docstring before the run; no shopping.",
}

deviations: list[str] = [
    "THE DELTA: measurement grid only — g2b's CELL-G2 rerun with the ruler "
    "read every 2 steps (+ step 1; 151 reads vs g2b's 9+9). The organ, "
    "root, monitor (g2b's G2BNet via module import), seed, and wash/replay "
    "arithmetic are g2b VERBATIM; run_cell is reused unmodified.",
    "WHY A RERUN, NOT CONTINUATIONS: g2b saved bodies only at +50 and +300; "
    "mid-cycle states were never on disk, so the phase grid cannot be "
    "evaluated from checkpoints alone. Short continuations from +50 would "
    "fork a NEW self-timed schedule (the monitor re-fires on its own clock), "
    "which samples schedule-space, not phase. The deterministic rerun with "
    "dense passive reads is the only mechanism that fills the cells BETWEEN "
    "g2b's saved points on g2b's OWN realization — and F_SCHED/F_TRACE/F_SD "
    "verify exactly that.",
    "NO BASE RERUN: g2b's stored CELL-BASE contrast (ruler 0.0241 <= 0.27 at "
    "+50, CPU, same seed) is embedded as the contrast gate; it passed in g2b "
    "and nothing in g2c touches the wash law.",
    "NO NEW CHECKPOINT FILES: the 151 measurement state_dicts live in memory "
    "and are discarded after the +50 bit-check (deliverables are metrics + "
    "PNG; the reusable artifacts remain g2b's).",
    "TRAIN CAP: G2C_CAP_S=900 s (g2b's 600 s bound + the dense evals' wall "
    "time; train-only is unchanged at ~100 s — the cap has never bound).",
    "Stagger: none needed (a single training; the dispatch's between-trainings "
    "stagger is vacuous here). Cooldowns skipped (CPU-only, g2b's convention).",
    "Single seed (10902), single lineage, n=1 realization — this replicate "
    "samples PHASE, not seed-space; a seed replicate remains owed (honesty).",
    "Smoke mode trims: 36-step cell, grid every 2 steps to 36, fidelity vs "
    "runs/g2b_smoke — nothing adjudicated (0 complete cycles).",
]


def sd_max_diff(a: dict, b: dict) -> tuple[float, bool]:
    """Max |diff| over float tensors; bitwise flag for integer tensors."""
    assert set(a) == set(b), f"sd key mismatch: {sorted(set(a) ^ set(b))}"
    mx, ints_ok = 0.0, True
    for k in a:
        ta, tb = a[k], b[k]
        if ta.is_floating_point():
            mx = max(mx, float((ta.float() - tb.float()).abs().max().item()))
        else:
            ints_ok = ints_ok and bool(torch.equal(ta, tb))
    return mx, ints_ok


# ------------------------------------------------------------------ main
def main():
    rd = run_dir("g2c_smoke" if SMOKE else "g2c")
    common.DEVICE = "cpu"
    G2.TRAIN_CAP_S = G2C_CAP_S          # read by G2.run_cell at call time
    log(f"G2C THE PHASE-OFFSET REPLICATE (g2b's owed cell; CPU-only, threads "
        f"{torch.get_num_threads()}, cuda avail {torch.cuda.is_available()}) "
        f"-> {rd}")
    log(f"measurement grid: {len(MEAS_GRID)} ruler reads (step 1 + every "
        f"{PHASE_EVERY} steps to +{N_STEPS}) over a {N_STEPS}-step cell")

    # ---- g2b's stored references (the realization being phase-resolved) ----
    g2bm = json.loads(G2B_METRICS.read_text(encoding="utf-8"))
    g2b_g2 = g2bm["cells"]["g2"]
    g2b_traj = {r["step"]: r for r in g2b_g2["traj"]}
    g2b_events = [e["step"] for e in g2b_g2["event_log"]]
    g2b_mon = {t["step"]: t for t in g2b_g2["monitor_trace"]}
    g2b_ruler = {r["step"]: r[RULER_KEY] for r in g2b_g2["traj"]}
    g2b_root_cells = g2bm["root"]["gates"]["G_ROOT2"]["cells_reloaded"]
    g2b_root_mon = g2bm["organ"]["root_monitor_onset"]
    g2b_base50 = g2bm["adjudication"]["bars"]["contrast_gate"]["base_ruler_50"]
    contrast = {"form": "g2b's stored CELL-BASE (CPU, seed 10902; embedded)",
                "base_ruler_50": g2b_base50, "bar": SHUT_BAR,
                "pass": (None if g2b_base50 is None
                         else bool(g2b_base50 <= SHUT_BAR))}
    if g2b_base50 is None:      # smoke-only (g2b's smoke cells end at +36)
        contrast["note"] = ("g2b smoke stored no +50 read — contrast "
                            "not applicable in smoke")
    log(f"g2b references loaded: {len(g2b_traj)} traj rows, events "
        f"{g2b_events}, {len(g2b_mon)} monitor checks; embedded contrast "
        f"base@+50 {g2b_base50} vs {SHUT_BAR}: " +
        ("PASS" if contrast["pass"] else
         ("N/A (smoke)" if contrast["pass"] is None else "FAIL")))

    # ---- protocol + root (REUSE — g2b's own builders and bit-gates) --------
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
              "g2b_stored": {**g2b_root_cells,
                             "root_monitor_onset": g2b_root_mon},
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

    # ---- THE CELL: g2b's CELL-G2 rerun, dense passive reads -----------------
    log("=" * 78)
    log(f"CELL G2 RERUN (g2b's CELL-G2 verbatim — onset-only monitor, seed "
        f"{FREEZE_SEED}, {N_STEPS} steps): dense grid, {len(MEAS_GRID)} reads")
    cell = G2.run_cell(
        "g2", root_sd, cue_pool, cue_mask, P["anchor_neutral"],
        P["train_ids"], P["itos"], P["r_eval_xy"], P["bat_ids"],
        P["zid"], FREEZE_SEED, MEAS_GRID)
    ev_steps = [e["step"] for e in cell["event_log"]]
    spac = cell["event_spacings"]
    traj = {r["step"]: r for r in cell["traj"]}
    dense = sorted(traj)

    # ---- FIDELITY GATES (the honest-reflex core) ----------------------------
    F_SCHED = {"g2b_events": g2b_events, "rerun_events": ev_steps,
               "match": bool(ev_steps == g2b_events)}
    keys = ("gm12", "g0", "gp12", "ce_r")
    common_steps = sorted(set(traj) & set(g2b_traj))
    tdiff = {k: max((abs(traj[s][k] - g2b_traj[s][k])
                     for s in common_steps), default=0.0) for k in keys}
    mon_steps = sorted(set(g2b_mon) & {t["step"] for t in
                                       cell["monitor_trace"]})
    mon_by_step = {t["step"]: t for t in cell["monitor_trace"]}
    mon_diff = max((abs(mon_by_step[s]["monitor"] - g2b_mon[s]["monitor"])
                    for s in mon_steps), default=0.0)
    mon_fired_match = all(mon_by_step[s]["fired"] == g2b_mon[s]["fired"]
                          for s in mon_steps)
    F_TRACE = {"form": "ruler at g2b's stored steps + monitor values",
               "n_steps_compared": len(common_steps),
               "max_abs_diff": max(tdiff.values()), "per_key": tdiff,
               "monitor_n_checks_compared": len(mon_steps),
               "monitor_max_abs_diff": mon_diff,
               "monitor_fired_flags_match": bool(mon_fired_match),
               "bit_tol": G2.G_BIT_TOL, "tol": G2.G_FALLBACK_TOL}
    F_TRACE["bit"] = bool(F_TRACE["max_abs_diff"] < G2.G_BIT_TOL
                          and mon_diff < G2.G_BIT_TOL and mon_fired_match)
    F_TRACE["pass"] = bool(F_TRACE["max_abs_diff"] < G2.G_FALLBACK_TOL
                           and mon_diff < G2.G_FALLBACK_TOL
                           and mon_fired_match)

    f_sd_final, f_s50 = None, None
    if G2B_CK_FINAL.exists():
        a = torch.load(G2B_CK_FINAL, map_location="cpu",
                       weights_only=False)["model"]
        mx, ints_ok = sd_max_diff(a, cell["final_sd"])
        f_sd_final = {"vs": "runs/checkpoints/g2b_g2.pt (final, +"
                           f"{N_STEPS})",
                      "max_abs_diff": mx, "ints_bit_identical": ints_ok,
                      "bit_tol": G2.G_BIT_TOL}
        f_sd_final["bit"] = bool(mx < G2.G_BIT_TOL and ints_ok)
        f_sd_final["pass"] = bool(mx < G2.G_FALLBACK_TOL and ints_ok)
    if (not SMOKE) and G2B_CK_S50.exists() and 50 in cell["sds"]:
        b = torch.load(G2B_CK_S50, map_location="cpu",
                       weights_only=False)["model"]
        mx, ints_ok = sd_max_diff(b, cell["sds"][50])
        f_s50 = {"vs": "runs/checkpoints/g2b_g2_s50.pt (body @ +50)",
                 "max_abs_diff": mx, "ints_bit_identical": ints_ok,
                 "bit_tol": G2.G_BIT_TOL}
        f_s50["bit"] = bool(mx < G2.G_BIT_TOL and ints_ok)
        f_s50["pass"] = bool(mx < G2.G_FALLBACK_TOL and ints_ok)
    F_SD = {"final": f_sd_final, "s50": f_s50}
    F_SD["pass"] = bool(f_sd_final and f_sd_final["pass"]
                        and (f_s50 is None or f_s50["pass"]))
    same_realization = bool(F_SCHED["match"] and F_TRACE["pass"]
                            and F_SD["pass"])
    log("=" * 78)
    log(f"F_SCHED (event schedule vs g2b): "
        f"{'MATCH' if F_SCHED['match'] else 'DIVERGED'} "
        f"({len(ev_steps)} events)")
    log(f"F_TRACE (ruler@stored steps, {len(common_steps)} pts): max|diff| "
        f"{F_TRACE['max_abs_diff']:.2e}; monitor max|diff| {mon_diff:.2e}, "
        f"fired flags {'match' if mon_fired_match else 'DIVERGE'}: "
        f"{'PASS' if F_TRACE['pass'] else 'FAIL'}"
        + (" (bit)" if F_TRACE["bit"] else ""))
    if f_sd_final:
        log(f"F_SD final sd vs g2b_g2.pt: max|diff| "
            f"{f_sd_final['max_abs_diff']:.2e}: "
            f"{'PASS' if f_sd_final['pass'] else 'FAIL'}"
            + (" (bit)" if f_sd_final["bit"] else ""))
    if f_s50:
        log(f"F_SD +50 sd vs g2b_g2_s50.pt: max|diff| "
            f"{f_s50['max_abs_diff']:.2e}: "
            f"{'PASS' if f_s50['pass'] else 'FAIL'}"
            + (" (bit)" if f_s50["bit"] else ""))
    if same_realization:
        log("SAME-REALIZATION: YES — the dense trace IS g2b's trajectory")
    else:
        log("SAME-REALIZATION: NO — same-seed twin; divergence recorded")
    del cell["sds"]        # measurement states are not artifacts

    G_NAMEFREE = {"cell_zeph_violations": cell["zeph_violations"],
                  "pass": bool(cell["zeph_violations"] == 0)}
    G_REPLAY = {"n_events": cell["n_events"],
                "checks": cell["replay_checks"],
                "pass": bool(cell["n_events"] == 0 or
                             (cell["replay_checks"]["mask7_ok"]
                              == cell["n_events"]
                              and cell["replay_checks"]["anchors_8_8_ok"]
                              == cell["n_events"]))}
    G_STEP = {"steps_ran": cell["steps_ran"], "expected": N_STEPS,
              "batch": 32, "pass": bool(cell["steps_ran"] == N_STEPS)}
    hard = {"G_POOL": G_POOL["pass"], "G_ROOT": G_ROOT["pass"],
            "G_NAMEFREE": G_NAMEFREE["pass"], "G_REPLAY": G_REPLAY["pass"],
            "G_STEP": G_STEP["pass"], "CONTRAST": contrast["pass"]}
    bad = [k for k, v in hard.items() if not v]
    if bad and not SMOKE:
        raise RuntimeError(f"gate(s) FAILED: {bad}")

    # =====================================================================
    # PHASE ANALYSIS — the full cycle shape (complete cycles adjudicated)
    # =====================================================================
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
            "n_samples": len(ss),
            "first_post_event": vs[0],
            "peak": max(vs), "trough": min(vs),
            "median": float(np.median(vs)), "mean": float(np.mean(vs)),
            "duty_ge_0.5": float(np.mean([v >= MAINTAIN_BAR for v in vs])),
            "last_before_next_event": vs[-1]})
    pre_ss, pre_v = seg(0, ev_steps[0] if ev_steps else N_STEPS + 1)
    tail_ss, tail_v = seg(ev_steps[-1] + 1 if ev_steps else 0, None) \
        if ev_steps else ([], [])
    tail_first = traj[ev_steps[-1]][RULER_KEY] if ev_steps else None

    binned = []
    for ib in range(PHASE_BINS):
        sel = [v for ph, v in zip(pooled_ph, pooled_v)
               if ib / PHASE_BINS <= ph < (ib + 1) / PHASE_BINS]
        binned.append({
            "phase_bin": [round(ib / PHASE_BINS, 4),
                          round((ib + 1) / PHASE_BINS, 4)],
            "center": round((ib + 0.5) / PHASE_BINS, 4), "n": len(sel),
            "mean": float(np.mean(sel)) if sel else None,
            "median": float(np.median(sel)) if sel else None,
            "duty_ge_0.5": (float(np.mean([v >= MAINTAIN_BAR for v in sel]))
                            if sel else None)})
    bin_means = [b["mean"] for b in binned if b["mean"] is not None]
    bin_medians = [b["median"] for b in binned if b["median"] is not None]
    mean_over_phase_offsets = float(np.mean(bin_means)) if bin_means else None
    cycle_median = float(np.median(pooled_v)) if pooled_v else None
    cycle_mean = float(np.mean(pooled_v)) if pooled_v else None
    duty = (float(np.mean([v >= MAINTAIN_BAR for v in pooled_v]))
            if pooled_v else None)
    last_bin_ge = max((b["center"] for b in binned
                       if b["median"] is not None
                       and b["median"] >= MAINTAIN_BAR), default=None)
    phase = {
        "definition": "cycles = complete inter-event intervals [e_k, "
                      "e_{k+1}); phase = (step - e_k)/spacing; ruler "
                      f"{RULER_KEY}; dense samples every {PHASE_EVERY} steps",
        "n_complete_cycles": len(cycles),
        "n_pooled_samples": len(pooled_v),
        "cycle_median": cycle_median,
        "cycle_mean": cycle_mean,
        "mean_over_phase_offset_checkpoints": mean_over_phase_offsets,
        "duty_ge_0.5": duty,
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
        "phase_bins": binned,
        "expression_ge_0.5_through_phase": last_bin_ge,
        "pre_rhythm": {"steps": [pre_ss[0], pre_ss[-1]] if pre_ss else None,
                       "min": min(pre_v) if pre_v else None,
                       "max": max(pre_v) if pre_v else None,
                       "note": "the initial death before the first event — "
                               "co-reported, not adjudicated"},
        "tail_partial": {"from_event": ev_steps[-1] if ev_steps else None,
                         "first_post_event": tail_first,
                         "n_samples": len(tail_ss),
                         "min": min(tail_v) if tail_v else None,
                         "max": max(tail_v) if tail_v else None,
                         "ruler_at_final_step": traj[N_STEPS][RULER_KEY]
                         if N_STEPS in traj else None,
                         "note": "the partial cycle after the last event — "
                                 "co-reported, not adjudicated"},
    }
    log("PHASE: " + (f"cycle-median {cycle_median:.4f}, cycle-mean "
                     f"{cycle_mean:.4f}, duty {duty:.0%}, peaks "
                     f"{phase['per_cycle_peak_range']}, troughs "
                     f"{phase['per_cycle_trough_range']}, expression >= 0.5 "
                     f"through phase {last_bin_ge}"
                     if cycle_median is not None else
                     "no complete cycles (smoke)"))

    # =====================================================================
    # ADJUDICATION (registered bars; no shopping)
    # =====================================================================
    band_ok = bool(spac) and all(SPACING_BAND[0] <= s <= SPACING_BAND[1]
                                 for s in spac)
    frac_band = (float(np.mean([SPACING_BAND[0] <= s <= SPACING_BAND[1]
                                for s in spac])) if spac else None)
    ruler_clause = bool(cycle_median is not None
                        and (cycle_median >= MAINTAIN_BAR
                             or cycle_mean >= MAINTAIN_BAR))
    MAINTAINS_IN_RHYTHM = bool(contrast["pass"] and band_ok and ruler_clause)
    TROUGH_BOUND = bool(cycle_median is not None
                        and cycle_median < MAINTAIN_BAR
                        and cycle_mean < MAINTAIN_BAR)

    def shape_str() -> str:
        return (f"the full cycle shape over {len(cycles)} complete cycles "
                f"({len(pooled_v)} dense samples): CYCLE-MEDIAN "
                f"{cycle_median:.3f}, cycle-MEAN {cycle_mean:.3f} (mean over "
                f"the {PHASE_BINS} phase-offset checkpoints "
                f"{mean_over_phase_offsets:.3f}), DUTY {duty:.0%} of in-cycle "
                f"steps >= 0.5, per-cycle peaks "
                f"{phase['per_cycle_peak_range'][0]:.2f}-"
                f"{phase['per_cycle_peak_range'][1]:.2f} / medians "
                f"{phase['per_cycle_median_range'][0]:.2f}-"
                f"{phase['per_cycle_median_range'][1]:.2f} / troughs "
                f"{phase['per_cycle_trough_range'][0]:.2f}-"
                f"{phase['per_cycle_trough_range'][1]:.2f}; the binned-"
                f"median curve holds >= 0.5 through phase "
                f"{last_bin_ge:.2f} of the cycle")

    if SMOKE or cycle_median is None:
        verdict = "SMOKE (nothing adjudicated)"
        clause = "smoke trim — no complete cycles."
    elif MAINTAINS_IN_RHYTHM:
        verdict = ("MAINTAINS-IN-RHYTHM (the organ vindicated as a "
                   "self-maintaining memory in oscillation)")
        clause = (f"{shape_str()}; the event band stayed 20-45 "
                  f"({frac_band:.0%} of spacings {spac}). g2b's frozen +300 "
                  f"endpoint ({g2b_ruler.get(300, float('nan')):.3f}, "
                  f"reproduced {'bit-exact' if F_TRACE['bit'] else 'within tol'} "
                  f"here) sampled the trough-side of cycle 10's partial tail — "
                  f"phase luck, exactly as T128 suspected; the CYCLE-median, "
                  f"not one frozen phase, is the honest read, and it clears "
                  f"the 0.5 bar. The resurrection economy maintains "
                  f"expression-in-general, in rhythm, self-timed.")
    elif TROUGH_BOUND:
        verdict = ("TROUGH-BOUND (the organ maintains peaks but not "
                   "expression-in-general)")
        clause = (f"{shape_str()}; even the phase-median stays < 0.5 and the "
                  f"spacings {'stayed in' if band_ok else 'BROKE'} the 20-45 "
                  f"band ({frac_band:.0%}). The sawtooth's duty cycle is too "
                  f"low: every event resurrects the ruler to "
                  f"{phase['per_cycle_peak_range'][1]:.2f} but the wash "
                  f"drags it under the bar within the first "
                  f"{last_bin_ge:.2f} of each cycle — the organ is a "
                  f"peak-maintaining oscillator, not a standing memory. "
                  f"g2b's MAINTAIN-FAILED verdict stands beyond one phase.")
    else:
        verdict = "RHYTHM-BAND-BROKE (ruler clause recorded; no third bar)"
        clause = (f"{shape_str()}; the ruler clause "
                  f"{'PASSES' if ruler_clause else 'fails'} but the event "
                  f"band broke ({frac_band:.0%} in 20-45, spacings {spac}) — "
                  f"not one of the two registered bars; recorded, no bar "
                  f"invented.")
    log("=" * 78)
    log(f"G2C VERDICT: {verdict}")
    log(f"  {clause}")
    log(f"  fidelity: same_realization={same_realization} "
        f"(F_SCHED {F_SCHED['match']}, F_TRACE max|diff| "
        f"{F_TRACE['max_abs_diff']:.2e}, F_SD "
        f"{f_sd_final['max_abs_diff'] if f_sd_final else None})")
    log("=" * 78)

    # =====================================================================
    # PLOT — A: the dense phase-resolved ruler; B: cycles aligned by phase;
    # C: per-cycle peak/median/trough
    # =====================================================================
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.25, 1.0])
    axA = fig.add_subplot(gs[0, :])
    xs = dense
    ys = [traj[s][RULER_KEY] for s in xs]
    axA.plot(xs, ys, "-", color="#d62728", lw=1.5, zorder=3,
             label=f"G2C dense ruler (every {PHASE_EVERY} steps)")
    gxs = sorted(g2b_traj)
    axA.plot(gxs, [g2b_traj[s][RULER_KEY] for s in gxs], "o", ms=6,
             mfc="none", mec="#1f77b4", mew=1.4, zorder=4,
             label="g2b's stored checkpoints (fidelity overlay)")
    if ev_steps:
        axA.axvspan(0, ev_steps[0], color="#f0f0f0", zorder=0)
        axA.text(ev_steps[0] / 2, 0.04, "pre-rhythm\n(the death)",
                 ha="center", fontsize=8, color="#888888")
    for s in ev_steps:
        axA.axvline(s, color="#d62728", ls="-", lw=0.8, alpha=0.35,
                    zorder=1)
    axA.axhline(MAINTAIN_BAR, color="k", ls="--", lw=0.8, alpha=0.6)
    axA.axhline(SHUT_BAR, color="k", ls=":", lw=0.8, alpha=0.6)
    axA.text(N_STEPS + 2, MAINTAIN_BAR, " maintain 0.5", va="bottom",
             fontsize=8)
    axA.text(N_STEPS + 2, SHUT_BAR, " die 0.27", va="bottom", fontsize=8)
    axA.axhline(g2bm["root"]["cells_stored"]["g0"], color="#2ca02c", lw=0.8,
                alpha=0.5)
    if 300 in traj:
        axA.annotate(f"+300 = {traj[300][RULER_KEY]:.3f}\n(g2b's frozen "
                     f"endpoint)", xy=(300, traj[300][RULER_KEY]),
                     xytext=(236, 0.06), fontsize=8,
                     arrowprops=dict(arrowstyle="->", lw=0.8, color="#555"))
    axA.set_xlabel("wash step")
    axA.set_ylabel(f"ruler g{RULER_GEO:+d} mean p(Z)")
    axA.set_title(f"G2C THE PHASE-OFFSET REPLICATE — g2b's cell re-measured "
                  f"densely (vlines = self-timed events; "
                  f"{'same realization, gates PASS' if same_realization else 'same-seed twin — see fidelity gates'}); "
                  f"verdict: {verdict}")
    axA.legend(loc="center right", fontsize=8)
    axA.set_xlim(0, N_STEPS + 14)

    axB = fig.add_subplot(gs[1, 0])
    if cycles:
        for c in cycles:
            a, b = c["event_step"], c["next_event"]
            ss = [s for s in dense if a <= s < b]
            axB.plot([(s - a) / (b - a) for s in ss],
                     [traj[s][RULER_KEY] for s in ss], "-", color="#bbbbbb",
                     lw=0.9, alpha=0.9, zorder=2)
        ctr = [b_["center"] for b_ in binned]
        axB.plot(ctr, bin_medians, "o-", color="k", lw=2.2, ms=5, zorder=4,
                 label="phase-binned MEDIAN")
        axB.plot(ctr, bin_means, "s--", color="#d62728", lw=1.3, ms=4,
                 zorder=3, label="phase-binned mean")
        axB.axhline(MAINTAIN_BAR, color="k", ls="--", lw=0.8)
        axB.set_xlabel("cycle phase (0 = event step, post-replay; "
                       "1 = next event)")
        axB.set_ylabel(f"ruler g{RULER_GEO:+d}")
        axB.set_title(f"THE CYCLE'S SHAPE — {len(cycles)} complete cycles "
                      f"aligned (gray) | duty {duty:.0%} | expression >= 0.5 "
                      f"through phase {last_bin_ge:.2f}")
        axB.legend(fontsize=8, loc="upper right")
        axB.text(0.02, 0.02,
                 f"CYCLE-MEDIAN {cycle_median:.3f} | mean {cycle_mean:.3f} "
                 f"(bar 0.5)", transform=axB.transAxes, fontsize=9,
                 color="#333333")
    else:
        axB.text(0.5, 0.5, "no complete cycles", ha="center", va="center")
        axB.set_title("cycle-aligned")

    axC = fig.add_subplot(gs[1, 1])
    if cycles:
        ks = [c["k"] for c in cycles]
        axC.plot(ks, [c["peak"] for c in cycles], "^-", color="#d62728",
                 lw=1.0, ms=6, label="peak")
        axC.plot(ks, [c["median"] for c in cycles], "o-", color="k",
                 lw=1.4, ms=5, label="median")
        axC.plot(ks, [c["trough"] for c in cycles], "v-", color="#1f77b4",
                 lw=1.0, ms=6, label="trough")
        axC.axhline(MAINTAIN_BAR, color="k", ls="--", lw=0.8)
        axC.axhline(SHUT_BAR, color="k", ls=":", lw=0.8)
        axC.set_xlabel("cycle (event k -> event k+1)")
        axC.set_ylabel(f"ruler g{RULER_GEO:+d}")
        axC.set_title("per-cycle peak / median / trough "
                      f"({phase['frac_cycles_median_ge_0.5']:.0%} of cycles "
                      f"median >= 0.5)")
        axC.legend(fontsize=8, loc="center right")
    else:
        axC.text(0.5, 0.5, "no complete cycles", ha="center", va="center")
        axC.set_title("per-cycle stats")

    fig.tight_layout()
    png = rd / "phase_replicate.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    log(f"[plot] {png}")

    # =====================================================================
    # metrics.json
    # =====================================================================
    metrics = {
        "experiment": "g2c",
        "date": common.now_iso(),
        "purpose": "THE PHASE-OFFSET REPLICATE — g2b's owed cell (T128): the "
                   "onset-monitor organ's self-timed sawtooth re-measured on "
                   "a dense grid (every 2 steps) so the maintain verdict "
                   "rides the CYCLE, not one frozen phase. g2b's CELL-G2 "
                   "rerun verbatim (same root/monitor/seed); measurement-only "
                   "delta; fidelity-gated against g2b's stored trace, event "
                   "log, and checkpoints.",
        "delta_vs_g2b": {
            "g2b": "9 checkpoint reads + 9 post-event reads of the ruler "
                   "over the 300-step cell; +300 sampled a sawtooth trough "
                   "(0.305)",
            "g2c": f"{len(MEAS_GRID)} dense reads (step 1 + every "
                   f"{PHASE_EVERY} steps) of the SAME cell; cycle-median / "
                   "duty / phase-binned curve adjudicated",
            "organ": "VERBATIM g2b (onset-only monitor via g2b module "
                     "import; G2.G2Net patched by g2b_onset_monitor itself)",
            "training_stream": "bit-untouched by the grid: run_cell's "
                               "evals are no-RNG no-grad CPU-twin reads; "
                               "verified by F_SCHED/F_TRACE/F_SD rather "
                               "than assumed",
        },
        "smoke": SMOKE,
        "threads": torch.get_num_threads(),
        "cpu_only": True,
        "cfg": g2bm["cfg"],
        "compute": {"cpu_only": True, "cuda_visible_devices": "-1",
                    "threads": torch.get_num_threads(), "stagger_s": 0.0,
                    "train_cap_s": G2C_CAP_S,
                    "meas_grid_n": len(MEAS_GRID),
                    "meas_grid_every": PHASE_EVERY},
        "g2b_references": {
            "metrics": f"runs/{G2B_NAME}/metrics.json",
            "events": g2b_events,
            "n_events": len(g2b_events),
            "ruler_at_300": g2b_ruler.get(300),
            "verdict_then": g2bm["adjudication"]["verdict"],
            "contrast_base_ruler_50": g2b_base50,
        },
        "root": {"source": "runs/checkpoints/"
                 + ("smoke_g2_root.pt" if SMOKE else "g2_root.pt")
                 + " (REUSED — not rebuilt; g2b's own root)",
                "gates": {"G_POOL": G_POOL, "G_ROOT": G_ROOT}},
        "cell": {"mode": "g2", "steps_ran": cell["steps_ran"],
                 "n_events": cell["n_events"],
                 "n_checks": cell["n_checks"],
                 "realized_r": cell["realized_r"],
                 "event_spacings": spac,
                 "events": ev_steps,
                 "monitor_trace": cell["monitor_trace"],
                 "event_log": cell["event_log"],
                 "traj_dense": cell["traj"],
                 "devices": {"initial": cell["initial_device"],
                             "final": cell["final_device"]},
                 "replay_checks": cell["replay_checks"]},
        "fidelity": {"F_SCHED": F_SCHED, "F_TRACE": F_TRACE, "F_SD": F_SD,
                     "same_realization": same_realization,
                     "note": "same_realization=True means the dense trace "
                             "IS g2b's trajectory (schedule + stored "
                             "checkpoints + state_dicts all match); False "
                             "means a same-seed twin — divergence recorded, "
                             "bars adjudicated on the rerun's own trace."},
        "phase": phase,
        "gates": {"CONTRAST": contrast, "G_NAMEFREE": G_NAMEFREE,
                  "G_REPLAY": G_REPLAY, "G_STEP": G_STEP},
        "registered_bars": REGISTERED_BARS,
        "registered_prediction": {
            "pre_run": "from g2b's sparse grid the sawtooth jumps to "
                       "0.62-0.72 post-event and decays through ~0.36-0.44 "
                       "at +24; the cycle-median hinges on the unknown "
                       "trough-side between +24 and the next event — a "
                       "genuinely open call between MAINTAINS-IN-RHYTHM and "
                       "TROUGH-BOUND (no shopping: both outcomes "
                       "pre-registered).",
            "discriminating_observation": "the dense trough just before "
                                          "each event and the phase-binned "
                                          "median curve's crossing of 0.5 — "
                                          "the duty cycle.",
        },
        "adjudication": {
            "ruler": {"geo": RULER_GEO, "key": RULER_KEY,
                      "root_value": g2bm["root"]["cells_stored"]["g0"],
                      "die_bar": SHUT_BAR, "maintain_bar": MAINTAIN_BAR},
            "bars": {"MAINTAINS_IN_RHYTHM": MAINTAINS_IN_RHYTHM,
                     "TROUGH_BOUND": TROUGH_BOUND,
                     "cycle_median": cycle_median,
                     "cycle_mean": cycle_mean,
                     "mean_over_phase_offset_checkpoints":
                         mean_over_phase_offsets,
                     "duty_ge_0.5": duty,
                     "n_complete_cycles": len(cycles),
                     "n_events": cell["n_events"],
                     "event_spacings": spac,
                     "frac_spacing_in_20_45": frac_band,
                     "band_stays_20_45": band_ok,
                     "ruler_clause_ge_0.5": ruler_clause,
                     "g2b_ruler_300_coreport": g2b_ruler.get(300)},
            "verdict": verdict, "clause": clause,
        },
        "honesty": [
            "PHASE-RECONSTRUCTION FIDELITY: the dense trace is honest only "
            "if it reproduces g2b's realization — F_SCHED (event schedule), "
            "F_TRACE (ruler at g2b's stored steps + monitor values/flags), "
            "F_SD (bit-diff vs g2b_g2.pt and g2b_g2_s50.pt) are reported as "
            f"gates; this run: same_realization={same_realization}.",
            "SINGLE ROOT (g2_root.pt), SINGLE SEED (10902), n=1: this "
            "replicate samples PHASE, not seed-space; the organ's rhythm is "
            "one lineage's point estimate and a seed replicate remains owed.",
            "THE GRID CANNOT STEER: measurements are no-RNG no-grad reads "
            "of a CPU twin; the training stream draws only from the seeded "
            "generator — verified by the fidelity gates, not assumed.",
            "COMPLETE CYCLES ONLY are adjudicated (the pre-rhythm death and "
            "the partial tail after the last event are co-reported); with "
            f"every-{PHASE_EVERY}-step sampling the duty cycle is "
            "step-weighted and unbiased.",
            "g2b's +300 endpoint (a trough) is co-reported everywhere — the "
            "point of g2c is that ONE frozen phase must not carry a "
            "verdict.",
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
