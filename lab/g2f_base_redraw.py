"""G2F — THE BASE-SEED REDRAW (T132's registered next rung; the stranger
rung of the root-lottery bracket).

CONTEXT (T130-T132): g2d licensed RHYTHM-REPLICATES on the LOCKED root (3
wash seeds, one waveform, medians 0.587/0.602/0.615); g2e redrew the INSTALL
gen seed (4305->4306) on the SAME base and hit ROOT-DRAW-BOUND — the timing
replicated (11 self-timed events, 100% of spacings in the 20-45 band) but
the amplitude did not (cycle-median 0.388 on the frozen ruler) AND the
fresh root missed the 0.7 strength gate (0.684; the root recipe is itself a
lottery 0.591/0.684/0.711). T132's decomposition: THE ORGAN CONTRIBUTES THE
CLOCK; THE ROOT CONTRIBUTES THE FLOOR — the redraw dissociates organ-lottery
from root-lottery at the BASE level.

THE DELTA (exactly one knob vs the LOCKED root's recipe): the BASE
checkpoint — runs/checkpoints/e098_base_s4305.pt -> e098_base_s4306.pt (a
stranger base: a fully fresh base-training draw from e098's ladder; with
g2e, the two rungs bracket the root recipe: g2e = install-lottery on a
sibling base, g2f = base-lottery). Everything else VERBATIM from the locked
root's recipe: e043-Dmix install with gen seed 4305 (the locked root's OWN
gen seed; house cosine at total=300, 300 steps), e113 jitter consolidation
(G2.consolidate, seed 10901), cue pool write-once (built from the same
install_occ — corpus-derived, bit-identical to the locked root's; verified
G_POOL), wash seed 10902 (the g2c reference convention). runs/checkpoints/
e098_install_s4306.pt is NOT used — that is e098's own different install
recipe; the g2 install recipe runs FRESH on the new base.

MECHANISM NOTE (the dispatch's override clause): the base path is NOT
hardcoded inside G2.build_root — it receives the loaded net via the
`net_base` parameter (the `base_ck` string is a label only). g2f passes
load_f2(runs/checkpoints/e098_base_s4306.pt); NO monkeypatch and NO edit to
the g2/g2b files. The organ is REUSED, not retyped: importing
g2b_onset_monitor patches G2.G2Net = G2BNet (onset-only monitor) and sets
LOW threads (4), and G2.run_cell / G2.build_root run unmodified.

THE READOUT (g2c/g2d/g2e conventions VERBATIM): CELL-G2 on the fresh root
at the dense grid (step 1 + every 2 steps + step 25 = 152 reads), phase
analysis over the COMPLETE inter-event intervals (cycle-median / duty / 8
phase bins), the frozen no-shopping ruler rule (among {-12,0,+12} battery
geos at construction pick max root mean_pz; all three co-reported; the g0
value named explicitly), and a CELL-BASE contrast (gate hard-disabled,
sparse grid, same wash seed) re-establishing the intervention contrast on
the fresh base — co-reported, not a registered bar.

REGISTERED BARS (dispatch g2f, VERBATIM; frozen — no shopping):
  ORGAN-REPLICATES: "fires if the fresh root clears the strength gate (ruler
      >= 0.7 at construction) AND the organ sustains the rhythm on it (>=5
      self-timed events with 100% of spacings in the 20-45 band;
      cycle-median >= 0.5) — the architecture claim licensed at n=2 roots,
      base level bracketed."
  ORGAN-ROOT-BOUND: "fires if the fresh root passes the gate but the rhythm
      fails (or vice versa) — the organ's success was root-specific (honest
      bound; the claim scoped)."
  Plus T132's pre-registered DISSOCIATION READING (report-only, both
  outcomes informative): "IF the redraw root clears the gate (>=0.7), the
  T132 decomposition PREDICTS amplitude recovery (cycle-median >= 0.5); a
  strong root with amplitude failure REFUTES the decomposition." State
  which happened.
Operationalization (frozen here, BEFORE the run; g2d/g2e's conventions):
  - GATE clause    = the fresh root's ruler (argmax over {-12,0,+12}
                     battery geos at construction) >= ROOT_BAR 0.7; all
                     three geos + held30 co-reported, g0 named explicitly;
  - EVENTS clause  = n_events >= 5 AND 100% of spacings in [20, 45];
  - CYCLE-MEDIAN clause = median of the ruler over all dense in-cycle
                     samples of the COMPLETE inter-event intervals >= 0.5;
  - RHYTHM clause  = EVENTS AND CYCLE-MEDIAN;
  - ORGAN-REPLICATES = GATE AND RHYTHM; ORGAN-ROOT-BOUND = GATE XOR RHYTHM;
  - residual (inherited VERBATIM from g2e's pre-registration, the
    exhaustive remainder): gate fails AND rhythm fails -> ROOT-DRAW-BOUND
    (the recipe's draw produced neither a strong root nor a rhythm; the
    claim stays n=1-root; a re-draw owed). No third bar invented.

ENVELOPE (dispatch, hard): gpu_ok() double-poll before every training
launch (G2.pick_dev's own quick double-poll 5 s apart; park-once — which
falls back to CPU at the FIRST failed double-poll, sooner than the
dispatch's 10-min patience; recorded), cooldown(120 s) between trainings,
each training's optimizer segment <= 180 s (GPU trainings; wall time is
dominated by the mandated CPU evals — the 180 s rule cannot hold CPU-only
at 4 threads, g2b's recorded note), evals on CPU (run_cell's CPU eval twin
for every ruler/CE read; the organ's own monitor read follows the wrapper's
device, g2's machinery unmodified), NO concurrent GPU jobs (double-poll +
mid-run guard every 25 steps with CPU migration). If the GPU is
user-occupied, everything parks to CPU and the deviation is documented.
Pre-dispatch instrument check (Rule 12): the ruler battery's three
geometries must SPAN the readout position before any compute — asserted in
code (G_SPAN).

Outputs: runs/g2f/{metrics.json, base_redraw.png}. No NOTES/THINKING/
QUEUE/STATE edits; single commit + push.

Run:  cd lab && python g2f_base_redraw.py    (G2F_SMOKE=1 shakedown)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

SMOKE = os.environ.get("G2F_SMOKE") == "1"
if SMOKE:
    os.environ["G2_SMOKE"] = "1"              # align G2's own smoke trims
os.environ["CUDA_VISIBLE_DEVICES"] = "0"      # GPU-first (this rung's envelope;
                                              # set before ANY torch import)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                # noqa: E402
import torch                                      # noqa: E402

import g2_rehearsal_organ as G2                   # noqa: E402 — the organ
import g2b_onset_monitor as G2B                   # noqa: E402 — REUSE: sets
                                                  # LOW threads (4), patches
                                                  # G2.G2Net = G2BNet, and
                                                  # clobbers CUDA_VISIBLE_DEVICES
                                                  # to "-1" (g2b's CPU-only era)
# undo g2b's CPU-only clobber: CUDA was already detected at common's import
# (GPU visible), so the availability cache survives the round trip; the env
# is restored for cleanliness before any training launch (verified by probe).
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import common                                     # noqa: E402
from common import run_dir, save_json             # noqa: E402

assert torch.get_num_threads() == 4, "g2b's import must set LOW threads (4)"
assert G2.G2Net is G2B.G2BNet, "g2b's onset-only monitor patch must be live"

import matplotlib                                  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
import matplotlib.gridspec as gridspec             # noqa: E402

CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ---- frozen constants (inherited; restated for the record) ----------------------
FREEZE_SEED = G2.FREEZE_SEED                      # 10902 — HELD (the g2c
                                                  # reference convention)
INST_GEN = 4305                                   # HELD at the locked root's
                                                  # OWN install gen seed — the
                                                  # delta is the BASE, not the
                                                  # install draw
BASE_CK = "e098_base_s4306.pt"                    # THE KNOB (was s4305)
LOCKED_BASE_CK = "e098_base_s4305.pt"             # provenance comparison only
INST_STEPS = 300                                  # g2's r1 exposure steps
CONS_SEED = G2.CONS_SEED                          # 10901 — recipe VERBATIM
ROOT_TAG = "g2f_base4306_g4305"
ROOT_BAR = G2.ROOT_BAR                            # 0.7 (frozen; never lowered)
SHUT_BAR, MAINTAIN_BAR = G2.SHUT_BAR, G2.MAINTAIN_BAR   # 0.27 / 0.50
SPACING_BAND = G2.SPACING_BAND                    # (20, 45)
EVENTS_MIN = 5                                     # the dispatch's >=5
G2F_CAP_S = 900.0                                  # g2d/g2e's wall bound (the
                                                  # wall is CPU-eval-dominated)
COOLDOWN_GPU_S = 120.0                             # the dispatch's between-
                                                  # training cooldown (GPU)
STAGGER_CPU_S = 20.0                               # CPU-fallback stagger
N_STEPS = 300 if not SMOKE else 36
PHASE_EVERY = 2                                    # g2c/g2d/g2e's dense stride
MEAS_GRID: tuple[int, ...] = tuple(sorted(        # g2c's grid VERBATIM
    set((1, 25) + tuple(range(PHASE_EVERY, N_STEPS + 1, PHASE_EVERY)))))
BASE_GRID: tuple[int, ...] = ((1, 2, 4, 10, 25, 50, 100, 200, 300)
                              if not SMOKE else (1, 2, 4, 36))  # g2b's sparse
PHASE_BINS = 8                                     # g2c's phase-bin count
SMOKE_TAG = "_smoke" if SMOKE else ""
G2C_METRICS = G2.E43.REPO / "runs" / f"g2c{SMOKE_TAG}" / "metrics.json"
G2E_METRICS = G2.E43.REPO / "runs" / f"g2e{SMOKE_TAG}" / "metrics.json"
G2B_METRICS = G2.E43.REPO / "runs" / f"g2b{SMOKE_TAG}" / "metrics.json"
G2_METRICS = G2.E43.REPO / "runs" / f"g2{SMOKE_TAG}" / "metrics.json"

REGISTERED_BARS = {
    "organ_replicates": "ORGAN-REPLICATES fires if the fresh root clears the "
                        "strength gate (ruler >= 0.7 at construction) AND the "
                        "organ sustains the rhythm on it (>=5 self-timed "
                        "events with 100% of spacings in the 20-45 band; "
                        "cycle-median >= 0.5) — the architecture claim "
                        "licensed at n=2 roots, base level bracketed",
    "organ_root_bound": "ORGAN-ROOT-BOUND fires if the fresh root passes the "
                        "gate but the rhythm fails (or vice versa) — the "
                        "organ's success was root-specific (honest bound; "
                        "the claim scoped)",
    "residual_root_draw_bound": "residual inherited VERBATIM from g2e's "
                                "pre-registration (the exhaustive remainder): "
                                "gate fails AND rhythm fails -> ROOT-DRAW-BOUND "
                                "(the recipe's draw produced neither; claim "
                                "stays n=1-root; a re-draw owed; no third bar "
                                "invented)",
    "dissociation_reading": "T132's pre-registered DISSOCIATION READING "
                            "(report-only, both outcomes informative): IF the "
                            "redraw root clears the gate (>=0.7), the T132 "
                            "decomposition PREDICTS amplitude recovery "
                            "(cycle-median >= 0.5); a strong root with "
                            "amplitude failure REFUTES the decomposition.",
    "source": "dispatch g2f (T132's registered next rung), VERBATIM; "
              "operationalized in this docstring before the run; no shopping.",
}

deviations: list[str] = [
    "THE DELTA: the BASE checkpoint only — e098_base_s4305.pt -> "
    "e098_base_s4306.pt (a stranger base, a fully fresh base-training draw "
    "from e098's ladder). The install recipe runs FRESH on the new base "
    "(e043-Dmix, gen seed 4305 HELD at the locked root's own, house cosine "
    "total=300, 300 steps); runs/checkpoints/e098_install_s4306.pt is NOT "
    "used (that is e098's own different install recipe). Consolidation "
    "(seed 10901), cue pool, wash seed (10902), cells (G2.run_cell "
    "unmodified, onset-only monitor via g2b's patch), dense grid and "
    "adjudication conventions are g2c/g2d/g2e VERBATIM.",
    "ROOT SCOPE (the honest-reflex point): g2f renews the BASE INIT — the "
    "install windows, cue pool (write-once, corpus-derived, bit-identical: "
    "G_POOL) and the consolidation stream (seed 10901) are still shared with "
    "the locked root; g2f's root is a STRANGER at the base level where g2e's "
    "was a sibling (same base, fresh install trajectory). The two rungs "
    "bracket the root recipe; neither measures the draw pass rate (n=1 each).",
    "GPU-FIRST ENVELOPE (this rung's dispatch; a change from g2b-g2e's "
    "CPU-only era): trainings may use the GPU via G2.pick_dev's own "
    "gpu_ok() double-poll (5 s apart) with park-once fallback and the 25-step "
    "mid-run guard. g2b's import clobbers CUDA_VISIBLE_DEVICES to '-1' — "
    "undone immediately after the import (CUDA was already detected at "
    "common's import with the GPU visible; availability survives, verified "
    "by probe and re-checked at runtime). Park-once falls back at the FIRST "
    "failed double-poll — sooner than the dispatch's 10-min patience "
    "(strictly safer; recorded).",
    "COOLDOWN(120 s) between trainings (the dispatch's band): "
    "G2.COOLDOWN_S=120 covers install->consolidation inside build_root; the "
    "cells pre-cooldown explicitly. If parked to CPU, the stagger trims to "
    "20 s (g2b/g2c/g2d/g2e's CPU convention) and the trim is recorded.",
    "TRAIN CAP: G2F_CAP_S=900 s wall (g2d/g2e's inherited bound). The "
    "dispatch's 180 s single-run rule is GPU-era and applies to the "
    "optimizer segment — on GPU each training's optimizer segment is a small "
    "fraction of wall (g2's own GPU cells); the WALL is dominated by the "
    "mandated CPU evals (152 dense reads on the g2 cell), which the envelope "
    "itself puts on CPU. Per-training walls + eval counts recorded.",
    "EVALS ON CPU: every ruler/CE readout runs on run_cell's CPU eval twin "
    "(g2's machinery); the organ's own monitor read (the sensor, 8 cue "
    "windows, no-grad) follows the wrapper's device — g2's original GPU-era "
    "behavior, machinery unmodified. Event timings therefore carry GPU float "
    "texture when the GPU is live (co-reported; bars live at "
    "order-of-magnitude separations).",
    "DEVICE TEXTURE vs THE SIBLING RUNG: g2e was CPU-only end-to-end; g2f's "
    "trainings run GPU-first — the g2e-vs-g2f comparison carries device "
    "texture on top of the base-seed delta (recorded; the frozen bars do not "
    "sit near float-precision boundaries).",
    "CELL-BASE RERUN on the fresh root (sparse g2b grid, seed 10902): the "
    "intervention contrast re-established on the NEW base — the honest-reflex "
    "'does intervening change behavior' check. Co-reported, NOT a registered "
    "bar; e184's ALL-DISSOLVE (n=3 wash seeds) + g4's two-root dissolve are "
    "the embedded cross-root context.",
    "NO +300 ANATOMY DIAL: g2f's registered bars are gate + rhythm (+ the "
    "report-only dissociation); the anatomy-at-+300 claim remains g2's, on "
    "the locked root.",
    "NO NEW CHECKPOINT FILES: the fresh root and the measurement states live "
    "in memory and are discarded (deliverables are metrics + PNG; the fresh "
    "root is deterministically rebuildable from the recorded recipe: base "
    "s4306 + install gen 4305 + cons 10901). The commit adds lab/ + runs/g2f "
    "(+ runs/g2f_smoke) only.",
    "RULE 12 PRE-DISPATCH INSTRUMENT CHECK (G_SPAN, asserted in code before "
    "any training): the ruler battery's three geometries {-12, 0, +12} span "
    "the readout position — window lengths {PRE-12, PRE, PRE+12} = "
    "{118, 130, 142}, every geo's window TERMINATES at the same char (the "
    "onset-preceding position), and the dense grid's stride (2) is finer "
    "than the registered minimum spacing (20) so every complete "
    "inter-event interval contains dense samples.",
    "phase_analysis/clauses are g2d/g2e VERBATIM with the one mechanical "
    "change they already made: the ruler key is a parameter (frozen at "
    "runtime from the fresh root's construction argmax, never shopped).",
    "Smoke mode trims: 8-step install/consolidation, 36-step cells, grid to "
    "36, references from runs/*_smoke, cooldowns to 2 s — nothing adjudicated.",
]


# ------------------------------------------------------------------ analysis
# PROVENANCE: lab/g2e_root_replicate.py VERBATIM (itself g2d's, itself
# g2c's) — the one mechanical change: ruler_key is a parameter.

def phase_analysis(traj: dict, ev_steps: list[int], ruler_key: str = "g0"):
    """g2c's phase conventions VERBATIM: complete inter-event intervals
    pooled; cycle-median/mean/duty over all dense in-cycle samples; 8 phase
    bins; per-cycle stats."""
    dense = sorted(traj)

    def seg(a: int, b):
        ss = [s for s in dense if s >= a and (b is None or s < b)]
        return ss, [traj[s][ruler_key] for s in ss]

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
                         "ruler_at_final_step": traj[N_STEPS][ruler_key]
                         if N_STEPS in traj else None},
    }


def clauses(n_events: int, spac: list[int], ph: dict) -> dict:
    """The two registered rhythm clauses, evaluated g2c/g2d-convention
    strictly (near-misses are co-reported texture, never a widened bar)."""
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
    rd = run_dir("g2f_smoke" if SMOKE else "g2f")
    common.DEVICE = "cpu"                # ALL readouts CPU-side; trainers
                                         # own their devices (pick_dev)
    G2.TRAIN_CAP_S = G2F_CAP_S           # read by G2.run_cell/consolidate
    G2.COOLDOWN_S = 0.0 if SMOKE else COOLDOWN_GPU_S   # install->cons gap
    log(f"G2F THE BASE-SEED REDRAW (T132's rung; GPU-first, threads "
        f"{torch.get_num_threads()}, cuda avail "
        f"{torch.cuda.is_available()}) -> {rd}")
    log(f"fresh root: {BASE_CK} (was {LOCKED_BASE_CK}) + e043-Dmix install "
        f"{INST_STEPS if not SMOKE else 8} steps (gen {INST_GEN} HELD) + "
        f"e113 consolidation (seed {CONS_SEED}); wash seed HELD at "
        f"{FREEZE_SEED}; dense grid {len(MEAS_GRID)} reads")

    # ---- the stored legs (references; never delete runs/) -------------------
    g2cm = json.loads(G2C_METRICS.read_text(encoding="utf-8"))
    g2c_cell = g2cm["cell"]
    g2c_traj = {r["step"]: r for r in g2c_cell["traj_dense"]}
    g2c_events = list(g2c_cell["events"])
    g2c_spac = list(g2c_cell["event_spacings"])
    g2c_ph_stored = g2cm["adjudication"]["bars"]["cycle_median"]
    g2c_duty_stored = g2cm["adjudication"]["bars"]["duty_ge_0.5"]
    g2c_root_ruler = g2cm["adjudication"]["ruler"]["root_value"]
    g2b_base50_locked = g2cm["g2b_references"]["contrast_base_ruler_50"]
    g2m = json.loads(G2_METRICS.read_text(encoding="utf-8"))
    g2_root_cells = g2m["root"]["cells"]            # the locked root's dials
    g2bm = json.loads(G2B_METRICS.read_text(encoding="utf-8"))
    locked_root_mon = g2bm["organ"]["root_monitor_onset"]
    # the sibling rung (g2e: install-lottery on the SAME base)
    g2em = json.loads(G2E_METRICS.read_text(encoding="utf-8"))
    g2e_cell = g2em["cells"]["g2"]
    g2e_traj = {r["step"]: r for r in g2e_cell["traj"]}
    g2e_events = list(g2e_cell["events"])
    g2e_spac = list(g2e_cell["event_spacings"])
    g2e_ph_stored = g2em["phase_fresh_root"]["cycle_median"]
    g2e_ruler_key = g2em["adjudication"]["gate"]["ruler_key"]
    g2e_gate = g2em["adjudication"]["gate"]
    g2e_root_mon = g2em["fresh_root"]["root_monitor_onset"]
    def _f3(v):
        return "NA" if v is None else f"{v:.3f}"

    log(f"stored legs: LOCKED root (g2c) {len(g2c_events)} events, cm "
        f"{_f3(g2c_ph_stored)}, ruler {g2c_root_ruler:.4f}; SIBLING root "
        f"(g2e, gen 4306 on {LOCKED_BASE_CK}) {len(g2e_events)} events, cm "
        f"{_f3(g2e_ph_stored)}, ruler {g2e_gate['ruler']:.4f} "
        f"(gate FAIL at 0.7); locked-root contrast base@+50 "
        f"{g2b_base50_locked}")

    # F_REF: g2c's stored dense trace re-pools to its stored cycle-median
    ph_ref = phase_analysis(g2c_traj, g2c_events, "g0")
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
    log(f"F_REF (locked-root leg): recomputed cycle-median "
        f"{ph_ref['cycle_median']} vs stored {g2c_ph_stored}: "
        f"{'PASS' if F_REF['pass'] else 'FAIL'}")
    assert F_REF["pass"], "g2c's stored dense trace failed to re-pool"
    ref_clauses = clauses(len(g2c_events), g2c_spac, ph_ref)
    ref_clauses["source"] = ("g2c's stored realization (locked root, seed "
                             "10902); embedded reference leg, not "
                             "re-adjudicated")

    # F_SIB: g2e's stored dense trace re-pools to its stored cycle-median
    ph_sib = phase_analysis(g2e_traj, g2e_events, g2e_ruler_key)
    both_none_s = ph_sib["cycle_median"] is None and g2e_ph_stored is None
    F_SIB = {"form": "g2e's stored dense trace re-pooled (sibling-rung load "
                     "integrity)",
             "cycle_median_recomputed": ph_sib["cycle_median"],
             "g2e_stored": g2e_ph_stored,
             "abs_diff": (None if both_none_s else
                          abs(ph_sib["cycle_median"] - g2e_ph_stored)),
             "n_dense_rows": len(g2e_traj)}
    F_SIB["pass"] = bool(both_none_s or F_SIB["abs_diff"] < 1e-9)
    if both_none_s:
        F_SIB["note"] = "smoke: 0 complete cycles on both sides — vacuous"
    log(f"F_SIB (sibling-rung leg): recomputed cycle-median "
        f"{ph_sib['cycle_median']} vs stored {g2e_ph_stored}: "
        f"{'PASS' if F_SIB['pass'] else 'FAIL'}")
    assert F_SIB["pass"], "g2e's stored dense trace failed to re-pool"
    sib_clauses = clauses(len(g2e_events), g2e_spac, ph_sib)
    sib_clauses["source"] = ("g2e's stored realization (install-redraw root, "
                             "seed 10902); embedded leg, not re-adjudicated")

    # ---- protocol (REUSE — g2b/g2c/g2d/g2e's own builder and bit-gates) ----
    P = G2B.rebuild_protocol()
    corpus, train_ids, itos, zid = P["corpus"], P["train_ids"], P["itos"], P["zid"]
    install_occ = P["install_occ"]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix, "pass": bool(mix == {"FLORIZEL": 19,
                                                         "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"

    # the install windows + original-host anchor bank (g2.main's arithmetic
    # VERBATIM — g2b's rebuild_protocol does not carry them)
    name_ids = corpus.encode(G2.NAME)
    L = len(G2.NAME)
    win_i = torch.stack([torch.cat([train_ids[p - G2.PRE: p], name_ids,
                                    train_ids[p + len(h):
                                              p + len(h) + G2.POST_CAP]])
                         for p, h in install_occ])
    inst_mask = torch.zeros(60, G2.BLOCK - 1, dtype=torch.bool)
    inst_mask[:, G2.PRE - 1: G2.PRE - 1 + L] = True
    anchor_full = torch.stack([train_ids[p - G2.PRE: p - G2.PRE + G2.BLOCK]
                               for p, _ in install_occ])   # 60 originals
    ids130 = P["bat_ids"][0]

    # ---- G_POOL: the organ cue pool == the locked root's -------------------
    # (the pool is built from install_occ — corpus-derived, independent of
    # the base init — so it must be BIT-IDENTICAL to the locked root's
    # on-disk buffers; verified against them)
    ck_locked = torch.load(G2.CKPT_DIR / ("smoke_g2_root.pt" if SMOKE
                                          else "g2_root.pt"),
                           map_location="cpu", weights_only=False)
    G_POOL = {"form": "bit vs the LOCKED root's on-disk cue pool "
                      "(g2_root.pt registered buffers)",
              "pool_bit_identical": bool(torch.equal(
                  ck_locked["model"]["cue_pool"].long(), P["jit_pool_x"])),
              "mask_bit_identical": bool(torch.equal(
                  ck_locked["model"]["cue_mask"], P["jit_pool_mask"])),
              "shape": list(P["jit_pool_x"].shape)}
    del ck_locked
    G_POOL["pass"] = bool(G_POOL["pool_bit_identical"]
                          and G_POOL["mask_bit_identical"]
                          and P["jit_pool_x"].shape[0] == 300
                          and int(P["jit_pool_mask"][0].sum()) == L)
    assert G_POOL["pass"], f"G_POOL FAILED: {G_POOL}"
    log("G_POOL: cue pool bit-identical to the locked root's (corpus-derived, "
        "base-independent): PASS")

    # ---- G_SPAN: Rule 12 pre-dispatch instrument check (BEFORE compute) ----
    geo_lens = sorted({int(P["bat_ids"][j].shape[1]) for j in G2.GEOS})
    last_common = all(torch.equal(P["bat_ids"][j][:, -1],
                                  P["bat_ids"][0][:, -1]) for j in G2.GEOS)
    G_SPAN = {
        "form": "Rule 12: the ruler battery's three geometries span the "
                "readout position before compute (asserted)",
        "geos": list(G2.GEOS),
        "battery_shapes": {f"g{j:+d}": list(P["bat_ids"][j].shape)
                           for j in G2.GEOS},
        "window_lengths": geo_lens,
        "spans_pre_plus_minus_12": bool(
            geo_lens == [G2.PRE + j for j in sorted(G2.GEOS)]),
        "common_readout_last_token": bool(last_common),
        "brackets_offset_zero": bool(min(G2.GEOS) < 0 < max(G2.GEOS)),
        "grid_stride": PHASE_EVERY,
        "grid_finer_than_min_spacing": bool(PHASE_EVERY <= SPACING_BAND[0]),
        "grid_covers": {"step1": 1 in MEAS_GRID, "step25": 25 in MEAS_GRID,
                        "max_is_N": max(MEAS_GRID) == N_STEPS,
                        "n_reads": len(MEAS_GRID)},
    }
    G_SPAN["pass"] = bool(G_SPAN["spans_pre_plus_minus_12"]
                          and G_SPAN["common_readout_last_token"]
                          and G_SPAN["brackets_offset_zero"]
                          and G_SPAN["grid_finer_than_min_spacing"]
                          and G_SPAN["grid_covers"]["step1"]
                          and G_SPAN["grid_covers"]["step25"]
                          and G_SPAN["grid_covers"]["max_is_N"])
    assert G_SPAN["pass"], f"G_SPAN FAILED (Rule 12): {G_SPAN}"
    log(f"G_SPAN (Rule 12): battery window lengths {geo_lens} span "
        f"[PRE-12, PRE, PRE+12] around the common onset-preceding readout "
        f"token; grid stride {PHASE_EVERY} <= min spacing "
        f"{SPACING_BAND[0]}: PASS")

    # ---- G_BASE: the knob itself — identity, hash, and a REAL fresh init ---
    def ck_info(p: Path) -> dict:
        st = p.stat()
        return {"path": f"runs/checkpoints/{p.name}", "bytes": st.st_size,
                "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime(st.st_mtime)),
                "sha256_16": hashlib.sha256(p.read_bytes()).hexdigest()[:16]}

    b_new = G2.load_f2(G2.CKPT_DIR / BASE_CK)
    b_old = G2.load_f2(G2.CKPT_DIR / LOCKED_BASE_CK)
    sd_new, sd_old = b_new.state_dict(), b_old.state_dict()
    max_diff = max(float((sd_new[k].float() - sd_old[k].float()).abs().max())
                   for k in sd_new)
    n_par = sum(v.numel() for v in sd_new.values())
    G_BASE = {"form": "the delta is real: the two base checkpoints are "
                      "distinct inits of the same family-2 cfg",
              "new_base": ck_info(G2.CKPT_DIR / BASE_CK),
              "locked_base": ck_info(G2.CKPT_DIR / LOCKED_BASE_CK),
              "params": n_par,
              "max_abs_param_diff": max_diff,
              "distinct_inits": bool(max_diff > 1e-3),
              "cfg": {k: v for k, v in G2.F2_CFG.__dict__.items()}}
    del b_new, b_old, sd_new, sd_old
    assert G_BASE["distinct_inits"], \
        f"G_BASE FAILED: bases look identical (max|diff| {max_diff:.2e})"
    log(f"G_BASE: {BASE_CK} sha256:{G_BASE['new_base']['sha256_16']} vs "
        f"{LOCKED_BASE_CK} sha256:{G_BASE['locked_base']['sha256_16']}; "
        f"max|param diff| {max_diff:.3f} over {n_par:,} params — distinct "
        f"inits: PASS")

    # =====================================================================
    # THE FRESH ROOT — one draw, no ladder (the gate adjudicates, it does
    # not re-roll)
    # =====================================================================
    log("=" * 78)
    inst_steps = INST_STEPS if not SMOKE else 8
    log(f"FRESH ROOT {ROOT_TAG}: {BASE_CK} (THE KNOB) + e043-Dmix install "
        f"{inst_steps} steps (gen {INST_GEN}, HELD at the locked root's own) "
        f"+ e113 jitter consolidation (seed {CONS_SEED}) — the g2 r1 recipe "
        f"verbatim, one knob changed")
    net_base = G2.load_f2(G2.CKPT_DIR / BASE_CK)
    rec = G2.build_root(ROOT_TAG, BASE_CK, INST_GEN, inst_steps, net_base,
                        win_i, inst_mask, anchor_full, train_ids,
                        P["jit_pool_x"], P["jit_pool_mask"], ids130,
                        P["r_eval_xy"], zid)
    del net_base
    root_sd = rec["root_sd"]

    # ---- THE GATE (frozen argmax ruler >= 0.7 at construction) -------------
    root_net = G2.evl_load(root_sd)
    geos = {j: G2.battery_cell(root_net, P["bat_ids"][j], zid)["mean_pz"]
            for j in G2.GEOS}
    held = {j: G2.battery_cell(root_net, P["held_bat"][j], zid)["mean_pz"]
            for j in G2.GEOS}
    ce_r_root = G2.ce_fixed_cpu(root_net, *P["r_eval_xy"])
    del root_net
    rgeo = max(G2.GEOS, key=lambda j: geos[j])
    ruler_key = {-12: "gm12", 0: "g0", 12: "gp12"}[rgeo]
    gate = {
        "form": "g2's frozen rule: ruler = argmax over {-12,0,+12} battery "
                "geos at construction; ROOT_BAR 0.7 never lowered",
        "geos": {f"g{j:+d}": geos[j] for j in G2.GEOS},
        "held30": {f"g{j:+d}": held[j] for j in G2.GEOS},
        "ce_r": ce_r_root,
        "ruler_geo": rgeo, "ruler_key": ruler_key, "ruler": geos[rgeo],
        "g0": geos[0], "root_bar": ROOT_BAR,
        "locked_root_geos": {k: g2_root_cells[k]
                             for k in ("gm12", "g0", "gp12", "ce_r")},
        "sibling_rung_geos": g2e_gate["geos"],
        "geo_matches_locked": bool(rgeo == 0),
        "passes": bool(geos[rgeo] >= ROOT_BAR),
    }
    log(f"FRESH ROOT geos " + " ".join(f"g{j:+d} {geos[j]:.4f}" for j in G2.GEOS)
        + f" | held30 g0 {held[0]:.4f} | CE_R {ce_r_root:.4f}")
    log(f"THE GATE: ruler g{rgeo:+d} = {geos[rgeo]:.4f} vs {ROOT_BAR} "
        f"(locked root's was g0 {g2_root_cells['g0']:.4f}; sibling g2e's was "
        f"g{g2e_gate['ruler_geo']:+d} {g2e_gate['ruler']:.4f}) -> "
        f"{'PASS' if gate['passes'] else 'FAIL'} "
        f"[g0 named per the dispatch letter: g0 {geos[0]:.4f}]")

    # ---- G_ROOTF: organ-registration round-trip self-consistency ----------
    wrap = G2B.G2BNet(G2.evl_load(root_sd), P["jit_pool_x"], P["jit_pool_mask"])
    root_mon = wrap.monitor(0, dev=CPU)
    rt_sd = G2.organ_body_sd(wrap.state_dict())
    re2 = {j: G2.battery_cell(G2.evl_load(rt_sd), P["bat_ids"][j],
                              zid)["mean_pz"] for j in G2.GEOS}
    del wrap
    G_ROOTF = {"form": "fresh root sd through the organ registration "
                       "(G2Net wrap -> body-strip -> reload), re-measured",
               "cells_roundtrip": {f"g{j:+d}": re2[j] for j in G2.GEOS},
               "cells_construction": {f"g{j:+d}": geos[j] for j in G2.GEOS},
               "max_abs_diff": max(abs(re2[j] - geos[j]) for j in G2.GEOS),
               "bit_tol": G2.G_BIT_TOL, "tol": G2.G_FALLBACK_TOL}
    G_ROOTF["bit"] = bool(G_ROOTF["max_abs_diff"] < G2.G_BIT_TOL)
    G_ROOTF["pass"] = bool(G_ROOTF["max_abs_diff"] < G2.G_FALLBACK_TOL)
    log(f"G_ROOTF (organ-registration round-trip): max|diff| "
        f"{G_ROOTF['max_abs_diff']:.2e}: "
        + ("PASS" if G_ROOTF["pass"] else "FAIL")
        + (" (bit)" if G_ROOTF["bit"] else "")
        + f" | fresh root onset monitor {root_mon:.4f} "
          f"(locked root's {locked_root_mon:.4f}; sibling g2e's "
          f"{g2e_root_mon:.4f}) — gate starts "
          f"{'CLOSED' if root_mon >= G2.THETA_OPEN else 'OPEN'}")

    # =====================================================================
    # THE CELLS (base first — the intervention contrast on the fresh root)
    # =====================================================================
    cells: dict = {}
    for i, (mode, grid) in enumerate((("base", BASE_GRID), ("g2", MEAS_GRID))):
        log("=" * 78)
        if not SMOKE:
            if not G2.GPU_PARKED:
                G2.cooldown(COOLDOWN_GPU_S)     # the dispatch's between-
                                                # training cooldown (GPU live)
            else:
                log(f"stagger {STAGGER_CPU_S:.0f}s (PARKED to CPU — the "
                    f"dispatch's fallback convention)")
                time.sleep(STAGGER_CPU_S)
        else:
            time.sleep(2.0)
        desc = {"base": "gate hard-disabled — the intervention contrast ON "
                        "THE FRESH BASE (co-reported; not a registered bar)",
                "g2": "gate live, onset-only monitor — THE RHYTHM LEG "
                      "(dense grid, g2c/g2d/g2e readout)"}[mode]
        log(f"CELL {mode.upper()} (fresh root): {desc} — {N_STEPS} steps, "
            f"seed {FREEZE_SEED}, {len(grid)} reads")
        t_cell = time.time()
        cells[mode] = G2.run_cell(
            mode, root_sd, P["jit_pool_x"], P["jit_pool_mask"],
            P["anchor_neutral"], train_ids, itos, P["r_eval_xy"],
            P["bat_ids"], zid, FREEZE_SEED, grid)
        cells[mode]["dials_note"] = desc
        cells[mode]["wall_s"] = round(time.time() - t_cell, 1)
        cells[mode]["n_evals"] = len(cells[mode]["traj"])
        del cells[mode]["sds"]            # measurement states are not
        del cells[mode]["final_sd"]       # artifacts (no ckpt files)

    # ---- gates -------------------------------------------------------------
    G_NAMEFREE = {"cell_zeph_violations": {m: cells[m]["zeph_violations"]
                                           for m in cells},
                  "pass": bool(all(cells[m]["zeph_violations"] == 0
                                   for m in cells))}
    G_REPLAY = {m: {"n_events": cells[m]["n_events"],
                    "checks": cells[m]["replay_checks"],
                    "pass": bool(cells[m]["n_events"] == 0 or
                                 (cells[m]["replay_checks"]["mask7_ok"]
                                  == cells[m]["n_events"]
                                  and cells[m]["replay_checks"]
                                  ["anchors_8_8_ok"]
                                  == cells[m]["n_events"]))}
                for m in cells}
    G_STEP = {m: {"steps_ran": cells[m]["steps_ran"], "expected": N_STEPS,
                  "batch": 32,
                  "pass": bool(cells[m]["steps_ran"] == N_STEPS)}
              for m in cells}
    hard = {"G_POOL": G_POOL["pass"], "G_ROOTF": G_ROOTF["pass"],
            "G_SPAN": G_SPAN["pass"], "G_BASE": G_BASE["distinct_inits"],
            "G_NAMEFREE": G_NAMEFREE["pass"],
            "G_REPLAY": all(v["pass"] for v in G_REPLAY.values()),
            "G_STEP": all(v["pass"] for v in G_STEP.values()),
            "F_REF": F_REF["pass"], "F_SIB": F_SIB["pass"]}
    log("G_NAMEFREE: " + ("PASS" if G_NAMEFREE["pass"] else "FAIL")
        + " | G_REPLAY: " + " ".join(f"{m}={G_REPLAY[m]['pass']}"
                                     for m in G_REPLAY)
        + " | G_STEP: " + " ".join(f"{m}={G_STEP[m]['steps_ran']}"
                                   for m in G_STEP))
    bad = [k for k, v in hard.items() if not v]
    if bad and not SMOKE:
        raise RuntimeError(f"gate(s) FAILED: {bad}")

    # ---- the intervention contrast (co-reported) ---------------------------
    base_ruler = {r["step"]: r[ruler_key] for r in cells["base"]["traj"]}
    contrast = {
        "form": "CELL-BASE on the FRESH BASE-REDRAW ROOT (sparse grid, seed "
                "10902) — the honest-reflex intervention check",
        "base_ruler_50": base_ruler.get(50),
        "bar": SHUT_BAR,
        "locked_root_base_50": g2b_base50_locked,
        "sibling_root_base_50": g2em["adjudication"]["contrast_coreport"]
        ["base_ruler_50"],
        "embedded_context": "e184 ALL-DISSOLVE (n=3 wash seeds, locked root); "
                            "g4's two-root dissolve (T127)",
        "pass": (None if base_ruler.get(50) is None
                 else bool(base_ruler.get(50) <= SHUT_BAR)),
        "structural_void_on_fresh_root": bool(
            base_ruler.get(50) is not None and base_ruler.get(50) > SHUT_BAR),
    }
    log(f"CONTRAST (fresh base): base ruler@+50 {base_ruler.get(50)} vs "
        f"{SHUT_BAR} (locked root's {g2b_base50_locked}; sibling g2e's "
        f"{contrast['sibling_root_base_50']}): "
        + ("the wash kills the un-gated organ here too"
           if contrast["pass"] else
           ("NOT APPLICABLE (smoke)" if contrast["pass"] is None
            else "STRUCTURAL VOID — the wash did not kill by +50")))

    # ---- the rhythm leg (g2c/g2d/g2e readout on the fresh root) ------------
    ev_steps = [e["step"] for e in cells["g2"]["event_log"]]
    traj = {r["step"]: r for r in cells["g2"]["traj"]}
    ph = phase_analysis(traj, ev_steps, ruler_key)
    cl = clauses(cells["g2"]["n_events"], cells["g2"]["event_spacings"], ph)
    log(f"RHYTHM LEG: {cells['g2']['n_events']} events {ev_steps}, spacings "
        f"{cells['g2']['event_spacings']}; cycle-median "
        f"{ph['cycle_median'] if ph['cycle_median'] is not None else 'NA'}, "
        f"duty {ph['duty_ge_0.5'] if ph['duty_ge_0.5'] is not None else 'NA'}, "
        f"troughs {ph['per_cycle_trough_range']}; clauses: events "
        f"{cl['events_clause']}, median {cl['median_clause']}")

    # geo texture co-report (g2's 'all three geos co-reported everywhere'
    # rule): the same cycle pooling on EACH geo key — the adjudicated key is
    # the frozen construction-argmax ruler; the others are non-adjudicated
    # texture (a geo-shifted root can pass on one key and fail on another;
    # recorded, never shopped)
    geo_texture = {}
    for j in G2.GEOS:
        kk = {-12: "gm12", 0: "g0", 12: "gp12"}[j]
        pj = phase_analysis(traj, ev_steps, kk)
        geo_texture[f"g{j:+d}"] = {
            "cycle_median": pj["cycle_median"],
            "duty_ge_0.5": pj["duty_ge_0.5"],
            "adjudicated": bool(kk == ruler_key),
            "locked_root_ruler_geo": bool(j == 0)}
    log("GEO TEXTURE (same pooling per key; adjudicated key = frozen "
        "construction argmax): "
        + " | ".join(f"{k} cyc-med "
                    f"{(None if v['cycle_median'] is None else round(v['cycle_median'], 3))}"
                    f" duty {(None if v['duty_ge_0.5'] is None else round(v['duty_ge_0.5'], 2))}"
                    + (" [ADJUDICATED]" if v["adjudicated"] else "")
                    for k, v in geo_texture.items()))

    # the base change must bite: the three roots' schedules differ (same wash
    # seed — a different organ read -> different timings)
    SCHEDS = {"locked_root_g2c_10902": g2c_events,
              "sibling_root_g2e_10902": g2e_events,
              "fresh_root_g2f_10902": ev_steps}
    SCHEDS_DIFFER = {"schedules": SCHEDS,
                     "fresh_vs_locked": bool(tuple(g2c_events) != tuple(ev_steps)),
                     "fresh_vs_sibling": bool(tuple(g2e_events) != tuple(ev_steps))}
    log(f"SCHEDULES: fresh vs locked differ {SCHEDS_DIFFER['fresh_vs_locked']}; "
        f"fresh vs sibling differ {SCHEDS_DIFFER['fresh_vs_sibling']} "
        f"(same wash seed throughout)")

    # =====================================================================
    # ADJUDICATION (registered bars + the dissociation reading; no shopping)
    # =====================================================================
    gate_pass = bool(gate["passes"])    # the frozen argmax rule decides; g0
                                        # is co-reported above
    rhythm_pass = bool(cl["passes_both"])
    ORGAN_REPLICATES = bool(gate_pass and rhythm_pass)
    ORGAN_ROOT_BOUND = bool(gate_pass != rhythm_pass)
    ROOT_DRAW_BOUND = bool((not gate_pass) and (not rhythm_pass))

    def gate_str() -> str:
        return (f"fresh root ruler g{rgeo:+d} {geos[rgeo]:.4f} "
                f"({'>=' if gate['passes'] else '<'} {ROOT_BAR}; geos "
                + " ".join(f"g{j:+d} {geos[j]:.4f}" for j in G2.GEOS)
                + f"; held30 g0 {held[0]:.4f}, CE_R {ce_r_root:.4f}; the "
                f"locked root's ruler was g0 {g2_root_cells['g0']:.4f}; the "
                f"sibling g2e root's was g{g2e_gate['ruler_geo']:+d} "
                f"{g2e_gate['ruler']:.4f})")

    def rhythm_str() -> str:
        return (f"organ on the fresh root: {cl['n_events']} events (spacings "
                f"{cl['event_spacings']}, "
                f"{0 if cl['frac_spacing_in_20_45'] is None else round(cl['frac_spacing_in_20_45'], 2)} "
                f"in 20-45), CYCLE-MEDIAN "
                f"{None if cl['cycle_median'] is None else round(cl['cycle_median'], 3)}, "
                f"duty {None if cl['duty_ge_0.5'] is None else round(cl['duty_ge_0.5'], 2)}, "
                f"per-cycle peaks {ph['per_cycle_peak_range']}, troughs "
                f"{ph['per_cycle_trough_range']}"
                + (f" — NEAR-TEXTURE: {'; '.join(cl['near_texture'])}"
                   if cl["near_texture"] else "")
                + f" -> rhythm clauses {'PASS' if rhythm_pass else 'FAIL'}")

    # T132's pre-registered dissociation reading (report-only)
    if gate_pass:
        if rhythm_pass:
            diss_outcome = "prediction_confirmed"
            diss_clause = (f"the redraw root CLEARED the gate (ruler "
                           f"{geos[rgeo]:.4f} >= 0.7) and the amplitude "
                           f"RECOVERED (cycle-median "
                           f"{cl['cycle_median']:.3f} >= 0.5) — T132's "
                           f"decomposition PREDICTED this: the organ "
                           f"contributes the clock (timing held), the root "
                           f"contributes the floor (a strong base-redraw "
                           f"root restores it). The decomposition stands.")
        else:
            diss_outcome = "decomposition_refuted"
            diss_clause = (f"the redraw root CLEARED the gate (ruler "
                           f"{geos[rgeo]:.4f} >= 0.7) but the amplitude "
                           f"FAILED (cycle-median "
                           f"{cl['cycle_median']:.3f} < 0.5) — a STRONG root "
                           f"with amplitude failure REFUTES the T132 "
                           f"decomposition as stated: the floor is not the "
                           f"root-strength dial alone (the organ-lottery / "
                           f"root-lottery split does not reduce to the gate).")
    else:
        diss_outcome = "untested_gate_failed"
        diss_clause = (f"the redraw root MISSED the gate (ruler "
                       f"{geos[rgeo]:.4f} < 0.7) — the dissociation "
                       f"prediction is conditional on a strong root and is "
                       f"UNTESTED at this draw (both outcomes informative, "
                       f"as registered): the base-level draw renews root "
                       f"weakness — the recipe's lottery now reads "
                       f"{g2_root_cells['g0']:.3f} (locked) / "
                       f"{g2e_gate['ruler']:.3f} (g2e install-draw) / "
                       f"{geos[rgeo]:.3f} (g2f base-draw) against the 0.7 "
                       f"bar, with e157's ported 0.591 before them. The "
                       f"timing clauses are co-reported regardless.")
    log("DISSOCIATION READING (report-only): " + diss_clause)

    if SMOKE:
        verdict = "SMOKE (nothing adjudicated)"
        clause = "smoke trim — 8-step root draw, 36-step cells."
    elif ORGAN_REPLICATES:
        verdict = ("ORGAN-REPLICATES (the architecture claim licensed at "
                   "n=2 roots, base level bracketed)")
        clause = (f"{gate_str()} — the strength gate CLEARED; and "
                  f"{rhythm_str()}; the locked root's stored leg (g2c, same "
                  f"wash seed {FREEZE_SEED}): {ref_clauses['n_events']} "
                  f"events, cycle-median {ref_clauses['cycle_median']:.3f}, "
                  f"duty {ph_ref['duty_ge_0.5']:.0%}; the sibling g2e leg: "
                  f"{sib_clauses['n_events']} events, cycle-median "
                  f"{sib_clauses['cycle_median']:.3f}. The self-maintaining "
                  f"rhythm sustains on a STRANGER BASE (a fresh base-init "
                  f"draw): with g2e (install-lottery, sibling) this rung "
                  f"brackets the root recipe — 'the organ works' is an "
                  f"ARCHITECTURE claim with the base level bracketed."
                  + (" [fresh-root contrast flag: see structural_void]"
                     if contrast["structural_void_on_fresh_root"] else ""))
    elif ORGAN_ROOT_BOUND:
        failed = ("the rhythm" if gate_pass else "the strength gate")
        verdict = ("ORGAN-ROOT-BOUND (the organ's success was root-specific "
                   "— honest bound; the claim scoped)")
        clause = (f"{gate_str()}; {rhythm_str()}. Exactly one leg passed "
                  f"({failed} failed) — on the locked root the organ "
                  f"maintained in rhythm (cycle-median "
                  f"{ref_clauses['cycle_median']:.3f}); the base-redraw "
                  f"draw does not reproduce both conditions, so the claim "
                  f"is scoped to the locked root's draw (recorded per the "
                  f"dispatch; no bar shopping).")
    else:
        verdict = ("ROOT-DRAW-BOUND (pre-registered residual inherited from "
                   "g2e: neither the gate nor the rhythm — the recipe's "
                   "draw lottery)")
        clause = (f"{gate_str()}; {rhythm_str()}. NEITHER dispatch bar's "
                  f"condition holds — the base-redraw draw produced neither "
                  f"a strong root nor a sustained rhythm (g2e's install "
                  f"redraw hit the same residual); the architecture claim "
                  f"stays n=1-root and the recipe's root lottery is now "
                  f"bracketed at BOTH levels (install-draw and base-draw), "
                  f"each failing once at n=1.")
    log("=" * 78)
    log(f"G2F VERDICT: {verdict}")
    log(f"  {clause}")
    log(f"  dissociation ({diss_outcome}): {diss_clause}")
    log(f"  contrast (fresh base): base@+50 {contrast['base_ruler_50']} vs "
        f"{SHUT_BAR}; pre-rhythm min on the g2 cell "
        f"{ph['pre_rhythm']['min']}")
    log("=" * 78)

    # =====================================================================
    # PLOT — A: the three-root dense waveform overlay + CELL-BASE contrast;
    # B: cycle shapes aligned by phase; C: per-cycle medians + troughs
    # =====================================================================
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.25, 1.0])
    axA = fig.add_subplot(gs[0, :])
    col_locked, col_sib, col_fresh, col_base = ("#1f77b4", "#ff7f0e",
                                                "#d62728", "#777777")
    xs_b = sorted(base_ruler)
    axA.plot(xs_b, [base_ruler[x] for x in xs_b], "-o", ms=3, lw=1.4,
             color=col_base, zorder=2,
             label=f"CELL-BASE on the fresh base (gate off; @+50 "
                   f"{base_ruler.get(50) if base_ruler.get(50) is None else round(base_ruler.get(50), 3)})")
    legs = [(g2c_traj, g2c_events, "g0", col_locked, "-",
             f"LOCKED root (g2c stored, seed 10902; ruler "
             f"{g2c_root_ruler:.3f})"),
            (g2e_traj, g2e_events, g2e_ruler_key, col_sib, "-",
             f"SIBLING root (g2e: install-draw, gen 4306 on s4305; ruler "
             f"{g2e_gate['ruler']:.3f})"),
            (traj, ev_steps, ruler_key, col_fresh, "-",
             f"FRESH root (g2f: BASE-draw s4306, gen {INST_GEN}; ruler "
             f"{geos[rgeo]:.3f})")]
    for tr, evs, rk, col, ls, lab in legs:
        xs = sorted(tr)
        axA.plot(xs, [tr[x][rk] for x in xs], ls, lw=1.5, color=col,
                 zorder=3, label=lab)
        for e in evs:
            axA.axvline(e, color=col, ls="-", lw=0.7, alpha=0.30, zorder=1)
    axA.axhline(MAINTAIN_BAR, color="k", ls="--", lw=0.8, alpha=0.6)
    axA.axhline(SHUT_BAR, color="k", ls=":", lw=0.8, alpha=0.6)
    axA.text(N_STEPS + 2, MAINTAIN_BAR, " maintain 0.5", va="bottom",
             fontsize=8)
    axA.text(N_STEPS + 2, SHUT_BAR, " die 0.27", va="bottom", fontsize=8)
    axA.axhline(g2c_root_ruler, color=col_locked, lw=0.8, alpha=0.4, ls="--")
    axA.axhline(geos[rgeo], color=col_fresh, lw=0.8, alpha=0.4, ls="--")
    if ev_steps:
        axA.axvspan(0, ev_steps[0], color="#f0f0f0", zorder=0)
    axA.set_xlabel("wash step")
    axA.set_ylabel("ruler mean p(Z) (each root's frozen geo)")
    axA.set_title(f"G2F THE BASE-SEED REDRAW — three-root waveform overlay "
                  f"(vlines = self-timed events, color-matched; verdict: "
                  f"{verdict.split(' (')[0]})")
    axA.legend(loc="center right", fontsize=8)
    axA.set_xlim(0, N_STEPS + 14)

    axB = fig.add_subplot(gs[1, 0])
    dense = sorted(traj)
    for k in range(len(ev_steps) - 1):
        a, b = ev_steps[k], ev_steps[k + 1]
        ss = [s for s in dense if a <= s < b]
        axB.plot([(s - a) / (b - a) for s in ss],
                 [traj[s][ruler_key] for s in ss], "-", color="#cccccc",
                 lw=0.7, alpha=0.8, zorder=2)

    def fm3(v):
        return "NA" if v is None else f"{v:.3f}"

    for (pp, rk, col, lab) in ((ph_ref, "g0", col_locked,
                                f"locked root binned MEDIAN (cyc-med "
                                f"{fm3(ph_ref['cycle_median'])})"),
                               (ph_sib, g2e_ruler_key, col_sib,
                                f"sibling g2e binned MEDIAN (cyc-med "
                                f"{fm3(ph_sib['cycle_median'])})"),
                               (ph, ruler_key, col_fresh,
                                f"fresh root binned MEDIAN (cyc-med "
                                f"{fm3(ph['cycle_median'])})")):
        meds = pp["phase_bin_medians"]
        if any(m is not None for m in meds):
            axB.plot(pp["phase_bin_centers"], meds, "o-", color=col,
                     lw=2.0, ms=4, zorder=4, label=lab)
    axB.axhline(MAINTAIN_BAR, color="k", ls="--", lw=0.8)
    axB.set_xlabel("cycle phase (0 = event step, post-replay; 1 = next event)")
    axB.set_ylabel("ruler mean p(Z)")
    axB.set_title("THE CYCLE'S SHAPE per root (gray = fresh root's "
                  "individual cycles)")
    axB.legend(fontsize=7, loc="upper right")

    axC = fig.add_subplot(gs[1, 1])
    plotted = False
    for (pp, cc, col, lab) in ((ph_ref, ref_clauses, col_locked,
                                f"locked root "
                                f"({'PASS' if ref_clauses['passes_both'] else 'FAIL'})"),
                               (ph_sib, sib_clauses, col_sib,
                                f"sibling g2e "
                                f"({'PASS' if sib_clauses['passes_both'] else 'FAIL'}; "
                                f"gate FAIL)"),
                               (ph, cl, col_fresh,
                                f"fresh root (rhythm "
                                f"{'PASS' if cl['passes_both'] else 'FAIL'}; "
                                f"gate {'PASS' if gate['passes'] else 'FAIL'})")):
        if not pp["per_cycle"]:
            continue
        plotted = True
        ks = [c["k"] for c in pp["per_cycle"]]
        axC.plot(ks, [c["median"] for c in pp["per_cycle"]], "o-",
                 color=col, lw=1.3, ms=5, label=lab)
        axC.plot(ks, [c["trough"] for c in pp["per_cycle"]], "v--",
                 color=col, lw=0.7, ms=4, alpha=0.6)
    if not plotted:
        axC.text(0.5, 0.5, "no complete cycles", ha="center", va="center")
    axC.axhline(MAINTAIN_BAR, color="k", ls="--", lw=0.8)
    axC.axhline(SHUT_BAR, color="k", ls=":", lw=0.8)
    axC.set_xlabel("cycle (event k -> event k+1)")
    axC.set_ylabel("ruler (o = median, v = trough)")
    axC.set_title("per-cycle medians and troughs — the three roots")
    axC.legend(fontsize=8, loc="center right")

    fig.tight_layout()
    png = rd / "base_redraw.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    log(f"[plot] {png}")

    # =====================================================================
    # metrics.json
    # =====================================================================
    def strip_cell(m: str) -> dict:
        c = cells[m]
        return {"mode": m, "seed": FREEZE_SEED,
                "steps_ran": c["steps_ran"], "n_events": c["n_events"],
                "n_checks": c["n_checks"], "realized_r": c["realized_r"],
                "event_spacings": c["event_spacings"],
                "events": [e["step"] for e in c["event_log"]],
                "monitor_trace": c["monitor_trace"],
                "event_log": c["event_log"],
                "traj": c["traj"],
                "devices": {"initial": c["initial_device"],
                            "final": c["final_device"]},
                "wall_s": c["wall_s"], "n_evals": c["n_evals"],
                "replay_checks": c["replay_checks"],
                "note": c["dials_note"]}

    metrics = {
        "experiment": "g2f",
        "date": common.now_iso(),
        "purpose": "THE BASE-SEED REDRAW — T132's registered next rung: a "
                   "STRANGER base (e098_base_s4306.pt, a fully fresh "
                   "base-training draw) under the locked root's own recipe "
                   "(e043-Dmix install gen 4305 HELD, 300 steps; e113 "
                   "consolidation seed 10901; cue pool write-once) must "
                   "clear the 0.7 root-strength gate AND sustain the "
                   "onset-monitor organ's rhythm (g2c/g2d/g2e readout, wash "
                   "seed 10902 held) — dissociating organ-lottery from "
                   "root-lottery at the BASE level; with g2e the two rungs "
                   "bracket the root recipe.",
        "delta_vs_g2e": {
            "g2e": "install-lottery: gen 4305->4306 on the SAME base "
                   "(sibling root) — timing replicated, amplitude and the "
                   "gate did not (ROOT-DRAW-BOUND residual)",
            "g2f": "base-lottery: the BASE checkpoint s4305->s4306 with the "
                   "install gen seed HELD at 4305 (stranger root) — the "
                   "dissociation rung T132 registered",
            "the_knob": "runs/checkpoints/e098_base_s4306.pt replaces "
                        "e098_base_s4305.pt (see provenance.base for hashes "
                        "and the distinct-init check); everything else "
                        "verbatim from the locked root's recipe; "
                        "e098_install_s4306.pt NOT used (e098's own "
                        "different install recipe — the g2 install runs "
                        "fresh on the new base)",
        },
        "smoke": SMOKE,
        "threads": torch.get_num_threads(),
        "cpu_only": False,
        "cfg": g2cm["cfg"],
        "compute": {"gpu_first": True, "parked": G2.GPU_PARKED,
                    "park_reason": G2.PARK_REASON,
                    "device_events": G2.device_events,
                    "cuda_visible_devices": os.environ.get(
                        "CUDA_VISIBLE_DEVICES"),
                    "threads": torch.get_num_threads(),
                    "cooldown_gpu_s": COOLDOWN_GPU_S,
                    "stagger_cpu_s": STAGGER_CPU_S,
                    "train_cap_s_wall": G2F_CAP_S,
                    "train_cap_note": "the 180-s rule is the GPU-era "
                                      "optimizer-segment bound; wall is "
                                      "CPU-eval-dominated (152 dense reads)",
                    "meas_grid_n": len(MEAS_GRID),
                    "meas_grid_every": PHASE_EVERY,
                    "evals": "CPU (run_cell's eval twin); the organ's own "
                             "monitor read follows the wrapper device"},
        "provenance": {
            "base": G_BASE,
            "install_gen_seed": INST_GEN,
            "install_steps": inst_steps,
            "install_house_cosine_total": inst_steps,
            "consolidation_seed": CONS_SEED,
            "wash_seed": FREEZE_SEED,
            "cue_pool": "write-once from install_occ (corpus-derived, "
                        "base-independent; G_POOL bit-identical to the "
                        "locked root's)",
            "root_tag": ROOT_TAG,
            "rebuild_recipe": f"load_f2(runs/checkpoints/{BASE_CK}) + "
                              f"G2.build_root(tag, base, gen {INST_GEN}, "
                              f"{inst_steps} steps) — deterministic",
        },
        "fresh_root": {
            "recipe": f"{BASE_CK} (THE KNOB) + e043-Dmix install "
                      f"({inst_steps} steps, gen {INST_GEN} HELD) + e113 "
                      f"jitter consolidation (300 steps, seed {CONS_SEED}) — "
                      f"g2's r1 VERBATIM, one base knob changed",
            "install_traj": rec["install_traj"],
            "install_wall_s": rec["install_wall_s"],
            "install_device": rec["install_device"],
            "cons_traj": rec["cons_traj"],
            "cons_device": rec["cons_device"],
            "root_monitor_onset": root_mon,
            "locked_root_monitor_onset": locked_root_mon,
            "sibling_root_monitor_onset": g2e_root_mon,
            "gates": {"G_SPLICE": G_SPLICE, "G_POOL": G_POOL,
                      "G_ROOTF": G_ROOTF, "G_SPAN": G_SPAN},
        },
        "root_lottery_series": {
            "e157_ported_root": 0.5911,
            "g2_locked_r1": g2_root_cells["g0"],
            "g2e_install_redraw": g2e_gate["ruler"],
            "g2f_base_redraw": geos[rgeo],
            "bar": ROOT_BAR,
            "note": "the root recipe's draw lottery across levels: locked "
                    "install / fresh install (sibling base) / fresh base "
                    "(stranger) — each n=1",
        },
        "locked_root_reference": {
            "source": "runs/g2/metrics.json (root.cells) + runs/g2c/"
                      "metrics.json (dense trace) + runs/g2b/metrics.json "
                      "(root onset monitor)",
            "cells": {k: g2_root_cells[k]
                      for k in ("gm12", "g0", "gp12", "ce_r")},
            "ruler_value": g2c_root_ruler,
            "g2c_cycle_median_stored": g2c_ph_stored,
            "g2c_duty_stored": g2c_duty_stored,
            "contrast_base_ruler_50": g2b_base50_locked,
        },
        "sibling_root_reference": {
            "source": "runs/g2e/metrics.json (the install-lottery rung)",
            "gate": {k: g2e_gate[k] for k in ("geos", "ruler_geo", "ruler",
                                              "passes")},
            "ruler_key": g2e_ruler_key,
            "events": g2e_events,
            "cycle_median_stored": g2e_ph_stored,
            "root_monitor_onset": g2e_root_mon,
            "contrast_base_ruler_50": contrast["sibling_root_base_50"],
        },
        "cells": {m: strip_cell(m) for m in cells},
        "phase_fresh_root": ph,
        "clauses_fresh_root": cl,
        "reference_legs": {
            "locked_g2c": {"seed": FREEZE_SEED,
                           "root": "locked (g2_root.pt via g2c)",
                           "F_REF": F_REF, "clauses": ref_clauses,
                           "phase": ph_ref, "events": g2c_events,
                           "source": ("g2c's stored dense realization; "
                                      "embedded leg, not re-adjudicated")},
            "sibling_g2e": {"seed": FREEZE_SEED,
                            "root": "install-redraw (g2e, gen 4306 on "
                                    "s4305)",
                            "F_SIB": F_SIB, "clauses": sib_clauses,
                            "phase": ph_sib, "events": g2e_events,
                            "source": ("g2e's stored dense realization; "
                                       "embedded leg, not re-adjudicated")},
        },
        "schedules_differ": SCHEDS_DIFFER,
        "gates": {"G_NAMEFREE": G_NAMEFREE, "G_REPLAY": G_REPLAY,
                  "G_STEP": G_STEP, "CONTRAST": contrast, "hard": hard},
        "registered_bars": REGISTERED_BARS,
        "registered_prediction": {
            "pre_run": "T132's decomposition (organ = clock, root = floor) "
                       "predicts: IF the base-redraw root clears the gate "
                       "(ruler >= 0.7) THEN amplitude recovers (cycle-median "
                       ">= 0.5); a strong root with amplitude failure "
                       "REFUTES the decomposition. A gate failure leaves the "
                       "prediction untested at this draw (both outcomes "
                       "informative — the root lottery extends to base "
                       "redraws). No shopping.",
            "discriminating_observation": "the fresh root's construction "
                                          "onset monitor vs the locked "
                                          "root's 0.609 and the sibling's "
                                          "0.596 (a root starting below "
                                          "THETA_OPEN would open the gate "
                                          "immediately — texture), and the "
                                          "fresh cell's per-cycle troughs vs "
                                          "g2d's 0.08-0.60 / g2e's "
                                          "0.019-0.280 — shallow troughs "
                                          "license with room; deep troughs "
                                          "extend the amplitude lottery "
                                          "while the median decides.",
        },
        "adjudication": {
            "gate": gate,
            "gate_pass": gate_pass, "rhythm_pass": rhythm_pass,
            "clauses": {
                "gate_clause": "fresh-root ruler (argmax of the three "
                               "battery geos, g2's frozen rule) >= 0.7 at "
                               "construction; g0 co-reported per the "
                               "dispatch's letter",
                "events_clause": "n_events >= 5 AND 100% of spacings in "
                                 "[20, 45] (g2b/g2c/g2d/g2e's registered "
                                 "band)",
                "median_clause": "cycle-median over dense in-cycle samples "
                                 "of complete inter-event intervals >= 0.5 "
                                 "(g2c's convention, same grid)",
            },
            "bars": {"ORGAN_REPLICATES": ORGAN_REPLICATES,
                     "ORGAN_ROOT_BOUND": ORGAN_ROOT_BOUND,
                     "ROOT_DRAW_BOUND_residual": ROOT_DRAW_BOUND,
                     "near_texture": cl["near_texture"]},
            "dissociation_reading": {"outcome": diss_outcome,
                                     "report_only": True,
                                     "registered": REGISTERED_BARS[
                                         "dissociation_reading"],
                                     "clause": diss_clause},
            "geo_texture_cycle_medians": geo_texture,
            "contrast_coreport": contrast,
            "pre_rhythm_min": ph["pre_rhythm"]["min"],
            "verdict": verdict, "clause": clause,
        },
        "honesty": [
            "WHAT WAS REPLICATED: the BASE-SEED draw at n=1 (the locked "
            "root's base s4305 vs one stranger draw s4306, install gen seed "
            "4305 held) — with g2e (install-lottery, same base) the two "
            "rungs BRACKET the root recipe; neither rung measures the draw "
            "pass rate, and neither re-establishes wash-draw robustness on "
            "any fresh root (g2d's n=3 was on the locked root; every fresh "
            "root here is a single wash-seed point estimate).",
            "SIBLINGS VS STRANGERS: the locked root and g2e share the base "
            "init (siblings — g2e renewed the install batch trajectory "
            "only); g2f renews the BASE INIT — a stranger at the base "
            "level. Install windows, cue pool (corpus-derived, "
            "bit-identical) and the consolidation stream (seed 10901) are "
            "shared across all three roots.",
            "BASE-DRAW n=1: a single stranger base; the base-level pass "
            "rate is unmeasured (as the install-level one is at n=1 after "
            "g2e). The ROOT-DRAW-BOUND residual at BOTH levels is one "
            "observation each — consistent with a coin-flip-ish lottery, "
            "not a measured rate.",
            "DEVICE TEXTURE: this rung's trainings are GPU-first (the "
            "dispatch's envelope) where g2b/g2c/g2d/g2e were CPU-only — "
            "the fresh root's weights and the organ's monitor reads (hence "
            "event timings) carry GPU float texture; every ruler/CE "
            "readout ran on the CPU eval twin. The locked root was itself "
            "cuda-install + cpu-consolidation, so the lineage has never "
            "been device-uniform; every bar sits at order-of-magnitude "
            "separations.",
            "THE INTERVENTION CHECK RAN ON THE FRESH BASE: CELL-BASE (gate "
            "off, same seed) — behavior with vs without the organ is the "
            "honest-reflex test; its +50 value is co-reported (not a "
            "registered bar).",
            "THE REFERENCE LEGS ARE STORED, NOT RERUN: g2c's and g2e's "
            "dense traces were fidelity-gated in their own runs; F_REF and "
            "F_SIB re-pool them as load checks only.",
            "NO BAR WIDENING: near-misses are co-reported texture; the two "
            "registered outcomes stay exhaustive over their domain, with "
            "the ROOT-DRAW-BOUND residual (inherited VERBATIM from g2e's "
            "pre-registration) for the both-fail corner.",
            "GEO TEXTURE IS CO-REPORTED, NEVER ADJUDICATED: the "
            "cycle-median is pooled per geo key and all three are recorded; "
            "the adjudicated key is the frozen construction-argmax ruler "
            "(g2e's lesson: a geo-shifted root can pass on one key and "
            "fail on another — the frozen rule decides).",
            "n=2 ROOTS would license the architecture claim ONLY via "
            "ORGAN-REPLICATES (gate AND rhythm); the lab's replicate "
            "standard for this claim remains what it was — this rung adds "
            "the base-level bracket, not a rate.",
        ],
        "checkpoints_saved": {},
        "deviations": deviations,
        "timing_s": round(time.time() - T0, 1),
    }
    save_json(rd / "metrics.json", G2.E43.jsonable(metrics))
    log(f"[done] metrics -> {rd / 'metrics.json'} "
        f"({metrics['timing_s']:.0f}s total); verdict: {verdict}; "
        f"dissociation: {diss_outcome}")


if __name__ == "__main__":
    main()
