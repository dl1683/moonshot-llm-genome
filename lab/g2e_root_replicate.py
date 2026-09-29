"""G2E — THE ROOT/LINEAGE REPLICATE (the architecture claim's last ladder
rung; T131's named debt).

CONTEXT (T131): the self-maintaining rhythm is wash-draw-robust on the
LOCKED root (g2d: 3 seeds, one waveform, medians 0.587/0.602/0.615) — but
the root itself is n=1 (g2's family-2 root, a single install+consolidation
draw). For "the organ works" as an ARCHITECTURE claim, a FRESH root (a new
install/consolidation draw at the same recipe) must also sustain the rhythm.

THE DELTA (exactly one knob of the root recipe changes): the install's GEN
SEED — 4305 -> 4306 (e184's +1 replicate convention) — on the SAME family-2
base (runs/checkpoints/e098_base_s4305.pt, fixed init on disk), with the g2
recipe VERBATIM otherwise: e043-Dmix install (E43.exposure, 16 paired + 32
random, house cosine at total=300, 300 steps) + e113 jitter consolidation
(G2.consolidate = e157's finetune_replay port, 300 steps, seed 10901). The
install windows, cue pool (write-once, built from the same install_occ) and
consolidation stream are therefore SHARED with the locked root; the fresh
draw is the install's batch-composition trajectory. The WASH-DRAW SEED is
held at 10902 for both cells, so the delta vs the g2c reference leg is the
ROOT, not the wash stream. The organ is REUSED, not retyped: this file
imports g2b_onset_monitor, whose import patches G2.G2Net = G2BNet (onset-only
monitor), and G2.run_cell / G2.build_root / the whole cell machinery run
unmodified.

THE READOUT (g2c/g2d conventions VERBATIM): CELL-G2 on the fresh root at the
dense grid (step 1 + every 2 steps + step 25 = 152 reads), phase analysis
over the COMPLETE inter-event intervals (cycle-median / duty / 8 phase bins),
and the two g2d clauses. A CELL-BASE (gate hard-disabled, sparse grid, same
seed) re-establishes the intervention contrast ON THE FRESH ROOT (the wash
must kill the un-gated organ there too — co-reported, not a registered bar);
the g2c stored dense trace (locked root, seed 10902) is the reference leg of
the n=2-root overlay, re-pooled only as a load check (F_REF).

REGISTERED BARS (dispatch g2e, VERBATIM; frozen — no shopping):
  ORGAN-REPLICATES: "fires if the fresh root clears the strength gate (ruler
      g0 >= 0.7 at construction) AND the organ sustains the rhythm on it
      (>=5 self-timed events in/near 20-45 band; cycle-median >= 0.5) — the
      architecture claim licensed at n=2 roots".
  ORGAN-ROOT-BOUND: "fires if the fresh root passes the gate but the rhythm
      fails (or vice versa) — the organ's success was root-specific (honest
      bound; the architecture claim scoped)".
Operationalization (frozen here, BEFORE the run; g2d's conventions):
  - GATE clause    = the fresh root's ruler (g2's frozen no-shopping rule:
                     among {-12, 0, +12} battery geos at construction, the
                     geo with maximum root mean_pz; the locked root's was
                     g0) >= ROOT_BAR 0.7; all three geos co-reported, the
                     g0 value named explicitly per the dispatch's letter;
  - EVENTS clause  = n_events >= 5 AND 100% of event spacings in the
                     registered band [20, 45] (g2b/g2c/g2d's own band);
  - CYCLE-MEDIAN clause = median of the ruler over all dense in-cycle
                     samples of the COMPLETE inter-event intervals
                     [e_k, e_{k+1}) >= 0.5 (g2c's cycle-median, same grid,
                     same pooling; pre-rhythm death and partial tail
                     co-reported, not adjudicated);
  - RHYTHM clause  = EVENTS AND CYCLE-MEDIAN; the dispatch's "in/near" is
                     honored as CO-REPORTED texture, never a widened bar
                     (g2d's rule);
  - ORGAN-REPLICATES = GATE AND RHYTHM; ORGAN-ROOT-BOUND = GATE XOR RHYTHM.
  - Residual contingency (pre-registered, the exhaustive remainder): if the
                     fresh root fails the gate AND the rhythm fails, neither
                     dispatch bar's condition holds — recorded as
                     ROOT-DRAW-BOUND (the recipe's draw produced neither a
                     strong root nor a rhythm; the architecture claim stays
                     n=1-root; a re-draw is owed). No third bar invented.

REGISTERED PREDICTION (before running): the GATE is the open risk — the
recipe's r1 draw landed 0.7106 (barely over 0.7; e157's earlier ported root
0.591), so root-strength has real draw variance: predict the fresh draw
lands 0.62-0.78 (coin-flip zone). If it clears, predict the organ finds the
same rhythm as on the locked root (8-12 events, spacing 24-40, cycle-median
0.5-0.7, duty 55-75%) — g2d's argument that the closed loop (deterministic
monitor -> refractory -> replay resurrection) dominates draw luck should
transfer from wash-draws to root-draws. DISCRIMINATING OBSERVATION: the
fresh root's construction-time onset monitor vs the locked root's 0.609 (a
root starting below THETA_OPEN would open the gate immediately — texture),
and the fresh cell's per-cycle troughs vs g2d's 0.08-0.60 range — shallow
troughs license the claim with room; deep troughs extend the amplitude
lottery to roots while the median decides.

CONSTRAINTS (dispatch): CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 before any
g2/g2b import), LOW threads (4 — set by g2b's import), stagger between the
four trainings (install, consolidation, base cell, g2 cell). The organ
machinery is REUSED; only the root changes.

Outputs: runs/g2e/{metrics.json, root_replicate.png}. No NOTES/THINKING/
QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python g2e_root_replicate.py    (G2E_SMOKE=1 shakedown)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

SMOKE = os.environ.get("G2E_SMOKE") == "1"
if SMOKE:
    os.environ["G2_SMOKE"] = "1"              # align G2's own smoke trims
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY (before g2/g2b import)
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

CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ---- frozen constants (inherited; restated for the record) ----------------------
FREEZE_SEED = G2.FREEZE_SEED                      # 10902 — HELD (the delta is
                                                  # the root, not the wash draw)
FRESH_GEN = 4306                                  # e184's +1 convention on g2's
                                                  # r1 install gen (4305)
BASE_CK = "e098_base_s4305.pt"                    # same family-2 base (fixed init)
INST_STEPS = 300                                  # g2's r1 exposure steps VERBATIM
CONS_SEED = G2.CONS_SEED                          # 10901 — recipe VERBATIM
ROOT_TAG = "g2e_s4305_g4306"
ROOT_BAR = G2.ROOT_BAR                            # 0.7 (frozen; never lowered)
SHUT_BAR, MAINTAIN_BAR = G2.SHUT_BAR, G2.MAINTAIN_BAR   # 0.27 / 0.50
SPACING_BAND = G2.SPACING_BAND                    # (20, 45)
EVENTS_MIN = 5                                     # the dispatch's >=5
G2E_CAP_S = 900.0                                  # g2d's bound (train-only is
                                                  # ~100-200 s/cell at 4 threads)
N_STEPS = 300 if not SMOKE else 36
PHASE_EVERY = 2                                    # g2c/g2d's dense stride
MEAS_GRID: tuple[int, ...] = tuple(sorted(        # g2c's grid VERBATIM
    set((1, 25) + tuple(range(PHASE_EVERY, N_STEPS + 1, PHASE_EVERY)))))
BASE_GRID: tuple[int, ...] = ((1, 2, 4, 10, 25, 50, 100, 200, 300)
                              if not SMOKE else (1, 2, 4, 36))  # g2b's sparse
PHASE_BINS = 8                                     # g2c's phase-bin count
STAGGER_S = 20.0 if not SMOKE else 2.0            # the dispatch's CPU stagger
G2C_NAME = "g2c_smoke" if SMOKE else "g2c"
G2C_METRICS = G2.E43.REPO / "runs" / G2C_NAME / "metrics.json"
G2B_METRICS = G2.E43.REPO / "runs" / ("g2b_smoke" if SMOKE else "g2b") \
    / "metrics.json"
G2_METRICS = G2.E43.REPO / "runs" / ("g2_smoke" if SMOKE else "g2") \
    / "metrics.json"

REGISTERED_BARS = {
    "organ_replicates": "ORGAN-REPLICATES fires if the fresh root clears the "
                        "strength gate (ruler g0 >= 0.7 at construction) AND "
                        "the organ sustains the rhythm on it (>=5 self-timed "
                        "events in/near 20-45 band; cycle-median >= 0.5) — "
                        "the architecture claim licensed at n=2 roots",
    "organ_root_bound": "ORGAN-ROOT-BOUND fires if the fresh root passes the "
                        "gate but the rhythm fails (or vice versa) — the "
                        "organ's success was root-specific (honest bound; the "
                        "architecture claim scoped)",
    "residual_root_draw_bound": "pre-registered remainder: gate fails AND "
                                "rhythm fails -> ROOT-DRAW-BOUND (the recipe's "
                                "draw produced neither; claim stays n=1-root; "
                                "a re-draw owed; no third bar invented)",
    "source": "dispatch g2e (T131), VERBATIM; operationalized in this "
              "docstring before the run; no shopping — 'in/near' is "
              "co-reported texture, never a widened bar (g2d's rule).",
}

deviations: list[str] = [
    "THE DELTA: the root's install GEN SEED only — 4305 -> 4306 (e184's +1 "
    "convention) on the SAME base (e098_base_s4305.pt), same e043-Dmix "
    "install (300 steps), same e113 consolidation (300 steps, seed 10901). "
    "The organ (g2b's G2BNet via module import), wash seed (10902), cells "
    "(G2.run_cell unmodified), dense grid and adjudication conventions are "
    "g2c/g2d VERBATIM.",
    "ROOT SCOPE (the honest-reflex point): the fresh draw renews the INSTALL "
    "batch-composition trajectory only — base init, install windows, cue pool "
    "(write-once from the same install_occ) and the consolidation stream are "
    "SHARED with the locked root; the two roots are siblings, not strangers. "
    "A base-seed redraw (e098_base_s4306.pt exists on disk) is a bigger "
    "redraw the dispatch's letter ('a new gen seed') did not ask for — "
    "recorded as the scope; base-init generality remains open.",
    "WASH SEED HELD AT 10902: the delta vs the g2c reference leg is the root, "
    "not the wash draw. Wash-draw robustness is g2d's n=3 ON THE LOCKED ROOT "
    "— it is NOT re-established on the fresh root (single wash seed there; "
    "honesty clause in metrics).",
    "CPU-ONLY END-TO-END (install + consolidation + both cells): the locked "
    "root was cuda-install + cpu-consolidation — a device texture difference "
    "between the two roots; every bar lives at order-of-magnitude separations "
    "(g2's own GPU-float deviation note, in reverse).",
    "CELL-BASE RERUN on the fresh root (sparse g2b grid, seed 10902): the "
    "intervention contrast re-established on the NEW root — the honest-reflex "
    "'does intervening change behavior' check. Co-reported, NOT a registered "
    "bar; e184's ALL-DISSOLVE (n=3 wash seeds) + g4's two-root dissolve are "
    "the embedded cross-root context.",
    "NO +300 ANATOMY DIAL: g2e's registered bars are gate + rhythm; the "
    "anatomy-at-+300 claim remains g2's, on the locked root.",
    "NO NEW CHECKPOINT FILES: the fresh root and the measurement states live "
    "in memory and are discarded (deliverables are metrics + PNG; the fresh "
    "root is deterministically rebuildable from the recorded recipe: base + "
    "gen 4306 + cons 10901). The commit adds lab/ + runs/g2e only.",
    "TRAIN CAP: G2E_CAP_S=900 s (g2d's bound, inherited; train-only is "
    "~100-200 s per training at the mandated 4 CPU threads — the lab's 180-s "
    "GPU-era single-run rule cannot hold; g2b's recorded convention).",
    "Cooldowns trimmed to 0 s (CPU-only; g2b/g2c/g2d's convention); 20 s "
    "stagger between the four trainings (the dispatch's CPU stagger).",
    "RULER FREEZE on the fresh root follows g2's frozen rule (argmax of the "
    "three battery geos at construction); the locked root's ruler was g0 and "
    "the dispatch's letter names g0 — if the fresh argmax differed, both the "
    "argmax ruler and g0 are reported and the frozen rule decides (no "
    "shopping).",
    "phase_analysis/clauses are g2d's VERBATIM with one mechanical change: "
    "the ruler key is a parameter (g2e's ruler is frozen at runtime from the "
    "fresh root, not hardcoded from a checkpoint meta).",
    "Smoke mode trims: 8-step install/consolidation, 36-step cells, grid to "
    "36, references from runs/g2c_smoke — nothing adjudicated.",
]


# ------------------------------------------------------------------ analysis
# PROVENANCE: lab/g2d_seed_replicate.py VERBATIM (the one mechanical change:
# ruler_key is a parameter — see deviations).

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
    strictly ('in/near' is co-reported texture, never a widened bar)."""
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
    rd = run_dir("g2e_smoke" if SMOKE else "g2e")
    common.DEVICE = "cpu"
    G2.TRAIN_CAP_S = G2E_CAP_S          # read by G2.run_cell/G2.consolidate
    G2.COOLDOWN_S = 0.0                 # CPU-only (g2b/g2c/g2d convention)
    log(f"G2E THE ROOT/LINEAGE REPLICATE (the architecture claim's last rung; "
        f"CPU-only, threads {torch.get_num_threads()}, cuda avail "
        f"{torch.cuda.is_available()}) -> {rd}")
    log(f"fresh root: {BASE_CK} + e043-Dmix install "
        f"{INST_STEPS if not SMOKE else 8} steps (gen {FRESH_GEN}, was 4305) + "
        f"e113 consolidation (seed {CONS_SEED}); wash seed HELD at "
        f"{FREEZE_SEED}; dense grid {len(MEAS_GRID)} reads")

    # ---- the locked root's stored legs (references; never delete runs/) ----
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
    log(f"locked-root references: g2c events {g2c_events} ({len(g2c_events)}), "
        f"cycle-median {g2c_ph_stored}, duty {g2c_duty_stored}, ruler "
        f"{g2c_root_ruler:.4f}, root onset monitor {locked_root_mon:.4f}; "
        f"locked-root contrast base@+50 {g2b_base50_locked} vs {SHUT_BAR}")

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
    ref_clauses["source"] = ("g2c's stored realization (fidelity-gated to "
                             "g2b's own); embedded reference leg, not "
                             "re-adjudicated")

    # ---- protocol (REUSE — g2b/g2c/g2d's own builder and bit-gates) --------
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

    # ---- G_POOL: the fresh root's organ cue pool == the locked root's -----
    # (both pools are write-once buffers built from the same install_occ, so
    # they must be BIT-IDENTICAL — the delta between the roots is purely the
    # body weights; verified against the locked root's on-disk buffers)
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

    # =====================================================================
    # THE FRESH ROOT — one draw, no ladder (the dispatch's letter; the gate
    # adjudicates, it does not re-roll)
    # =====================================================================
    log("=" * 78)
    inst_steps = INST_STEPS if not SMOKE else 8
    log(f"FRESH ROOT {ROOT_TAG}: {BASE_CK} + e043-Dmix install {inst_steps} "
        f"steps (gen {FRESH_GEN}) + e113 jitter consolidation (seed "
        f"{CONS_SEED}) — the g2 r1 recipe verbatim, one knob changed")
    net_base = G2.load_f2(G2.CKPT_DIR / BASE_CK)
    rec = G2.build_root(ROOT_TAG, BASE_CK, FRESH_GEN, inst_steps, net_base,
                        win_i, inst_mask, anchor_full, train_ids,
                        P["jit_pool_x"], P["jit_pool_mask"], ids130,
                        P["r_eval_xy"], zid)
    del net_base
    root_sd = rec["root_sd"]

    # ---- THE GATE (ruler g0 >= 0.7 at construction; frozen no-shopping rule)
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
        "geo_matches_locked": bool(rgeo == 0),
        "passes": bool(geos[rgeo] >= ROOT_BAR),
    }
    log(f"FRESH ROOT geos " + " ".join(f"g{j:+d} {geos[j]:.4f}" for j in G2.GEOS)
        + f" | held30 g0 {held[0]:.4f} | CE_R {ce_r_root:.4f}")
    log(f"THE GATE: ruler g{rgeo:+d} = {geos[rgeo]:.4f} vs {ROOT_BAR} "
        f"(locked root's was g0 {g2_root_cells['g0']:.4f}) -> "
        f"{'PASS' if gate['passes'] else 'FAIL'} "
        f"[locked-root ruler key g0 named per the dispatch letter: "
        f"g0 {geos[0]:.4f}]")

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
          f"(locked root's {locked_root_mon:.4f}) — gate starts "
        f"{'CLOSED' if root_mon >= G2.THETA_OPEN else 'OPEN'}")

    # =====================================================================
    # THE CELLS (base first — the intervention contrast on the fresh root)
    # =====================================================================
    cells: dict = {}
    for i, (mode, grid) in enumerate((("base", BASE_GRID), ("g2", MEAS_GRID))):
        log("=" * 78)
        if i:
            log(f"stagger {STAGGER_S:.0f}s (CPU-only dispatch)")
            time.sleep(STAGGER_S)
        desc = {"base": "gate hard-disabled — the intervention contrast ON "
                        "THE FRESH ROOT (co-reported; not a registered bar)",
                "g2": "gate live, onset-only monitor — THE RHYTHM LEG "
                      "(dense grid, g2c/g2d readout)"}[mode]
        log(f"CELL {mode.upper()} (fresh root): {desc} — {N_STEPS} steps, "
            f"seed {FREEZE_SEED}, {len(grid)} reads")
        cells[mode] = G2.run_cell(
            mode, root_sd, P["jit_pool_x"], P["jit_pool_mask"],
            P["anchor_neutral"], train_ids, itos, P["r_eval_xy"],
            P["bat_ids"], zid, FREEZE_SEED, grid)
        cells[mode]["dials_note"] = desc
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
            "G_NAMEFREE": G_NAMEFREE["pass"],
            "G_REPLAY": all(v["pass"] for v in G_REPLAY.values()),
            "G_STEP": all(v["pass"] for v in G_STEP.values()),
            "F_REF": F_REF["pass"]}
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
        "form": "CELL-BASE on the FRESH ROOT (sparse grid, seed 10902) — "
                "the honest-reflex intervention check",
        "base_ruler_50": base_ruler.get(50),
        "bar": SHUT_BAR,
        "locked_root_base_50": g2b_base50_locked,
        "embedded_context": "e184 ALL-DISSOLVE (n=3 wash seeds, locked root); "
                            "g4's two-root dissolve (T127)",
        "pass": (None if base_ruler.get(50) is None
                 else bool(base_ruler.get(50) <= SHUT_BAR)),
        "structural_void_on_fresh_root": bool(
            base_ruler.get(50) is not None and base_ruler.get(50) > SHUT_BAR),
    }
    log(f"CONTRAST (fresh root): base ruler@+50 {base_ruler.get(50)} vs "
        f"{SHUT_BAR} (locked root's {g2b_base50_locked}): "
        + ("the wash kills the un-gated organ here too"
           if contrast["pass"] else
           ("NOT APPLICABLE (smoke)" if contrast["pass"] is None
            else "STRUCTURAL VOID — the wash did not kill by +50")))

    # ---- the rhythm leg (g2c/g2d readout on the fresh root) ----------------
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
            "cycle_median": pj["cycle_median"], "duty_ge_0.5": pj["duty_ge_0.5"],
            "adjudicated": bool(kk == ruler_key),
            "locked_root_ruler_geo": bool(j == 0)}
    log("GEO TEXTURE (same pooling per key; adjudicated key = frozen "
        "construction argmax): "
        + " | ".join(f"{k} cyc-med "
                    f"{(None if v['cycle_median'] is None else round(v['cycle_median'], 3))}"
                    f" duty {(None if v['duty_ge_0.5'] is None else round(v['duty_ge_0.5'], 2))}"
                    + (" [ADJUDICATED]" if v["adjudicated"] else "")
                    for k, v in geo_texture.items()))

    # the root change must bite: the fresh schedule differs from the locked
    # root's (same wash seed — a different organ read -> different timings)
    SCHEDS = {"locked_root_g2c_10902": g2c_events,
              "fresh_root_g2e_10902": ev_steps}
    SCHEDS_DIFFER = {"schedules": SCHEDS,
                     "distinct": bool(tuple(g2c_events) != tuple(ev_steps))}
    log(f"SCHEDULES differ across roots (same wash seed): "
        f"{SCHEDS_DIFFER['distinct']}")

    # =====================================================================
    # ADJUDICATION (registered bars; no shopping)
    # =====================================================================
    gate_pass = bool(gate["passes"])    # the frozen rule decides; the
                                        # dispatch's g0 is co-reported above
    rhythm_pass = bool(cl["passes_both"])
    ORGAN_REPLICATES = bool(gate_pass and rhythm_pass)
    ORGAN_ROOT_BOUND = bool(gate_pass != rhythm_pass)
    ROOT_DRAW_BOUND = bool((not gate_pass) and (not rhythm_pass))

    def gate_str() -> str:
        return (f"fresh root ruler g{rgeo:+d} {geos[rgeo]:.4f} "
                f"({'>=' if gate['passes'] else '<'} {ROOT_BAR}; geos "
                + " ".join(f"g{j:+d} {geos[j]:.4f}" for j in G2.GEOS)
                + f"; held30 g0 {held[0]:.4f}, CE_R {ce_r_root:.4f}; the "
                f"locked root's ruler was g0 {g2_root_cells['g0']:.4f})")

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

    if SMOKE:
        verdict = "SMOKE (nothing adjudicated)"
        clause = "smoke trim — 8-step root draw, 36-step cells."
    elif ORGAN_REPLICATES:
        verdict = ("ORGAN-REPLICATES (the architecture claim licensed at "
                   "n=2 roots)")
        clause = (f"{gate_str()} — the strength gate CLEARED; and "
                  f"{rhythm_str()}; the locked root's stored leg (g2c, same "
                  f"wash seed {FREEZE_SEED}): {ref_clauses['n_events']} "
                  f"events, cycle-median {ref_clauses['cycle_median']:.3f}, "
                  f"duty {ph_ref['duty_ge_0.5']:.0%}. The self-maintaining "
                  f"rhythm sustains on a FRESH install+consolidation draw of "
                  f"the same recipe: per T131's ladder, 'the organ works' is "
                  f"now an ARCHITECTURE claim at the lab's n=2-root standard "
                  f"— not one root's luck."
                  + (" [fresh-root contrast flag: see structural_void]"
                     if contrast["structural_void_on_fresh_root"] else ""))
    elif ORGAN_ROOT_BOUND:
        failed = ("the rhythm" if gate_pass else "the strength gate")
        verdict = ("ORGAN-ROOT-BOUND (the organ's success was root-specific "
                   "— honest bound; the architecture claim scoped)")
        clause = (f"{gate_str()}; {rhythm_str()}. Exactly one leg passed "
                  f"({failed} failed) — on the locked root the organ "
                  f"maintained in rhythm (cycle-median "
                  f"{ref_clauses['cycle_median']:.3f}); the fresh draw does "
                  f"not reproduce both conditions, so the claim is scoped to "
                  f"the locked root's draw (recorded per the dispatch; no "
                  f"bar shopping).")
    else:
        verdict = ("ROOT-DRAW-BOUND (pre-registered residual: neither the "
                   "gate nor the rhythm — the recipe's draw lottery)")
        clause = (f"{gate_str()}; {rhythm_str()}. NEITHER dispatch bar's "
                  f"condition holds — the fresh draw produced neither a "
                  f"strong root nor a sustained rhythm; the architecture "
                  f"claim stays n=1-root and a re-draw is owed (recorded as "
                  f"the pre-registered residual, not a third bar).")
    log("=" * 78)
    log(f"G2E VERDICT: {verdict}")
    log(f"  {clause}")
    log(f"  contrast (fresh root): base@+50 {contrast['base_ruler_50']} vs "
        f"{SHUT_BAR}; pre-rhythm min on the g2 cell "
        f"{ph['pre_rhythm']['min']}")
    log("=" * 78)

    # =====================================================================
    # PLOT — A: the n=2-root dense waveform overlay; B: cycle shapes aligned
    # by phase; C: per-cycle medians + clause numbers
    # =====================================================================
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.25, 1.0])
    axA = fig.add_subplot(gs[0, :])
    col_locked, col_fresh = "#1f77b4", "#d62728"
    legs = [(g2c_traj, g2c_events, "g0", col_locked, "-",
             f"LOCKED root (g2c stored, seed 10902; ruler "
             f"{g2c_root_ruler:.3f})"),
            (traj, ev_steps, ruler_key, col_fresh, "-",
             f"FRESH root (g2e, gen {FRESH_GEN}, seed 10902; ruler "
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
    axA.set_title(f"G2E THE ROOT/LINEAGE REPLICATE — n=2-root waveform "
                  f"overlay (vlines = self-timed events, color-matched; "
                  f"verdict: {verdict.split(' (')[0]})")
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
    axC.set_title("per-cycle medians and troughs — the two roots")
    axC.legend(fontsize=8, loc="center right")

    fig.tight_layout()
    png = rd / "root_replicate.png"
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
                "replay_checks": c["replay_checks"],
                "note": c["dials_note"]}

    metrics = {
        "experiment": "g2e",
        "date": common.now_iso(),
        "purpose": "THE ROOT/LINEAGE REPLICATE — the architecture claim's "
                   "last ladder rung (T131): a FRESH root (a new "
                   "install+consolidation draw of the g2 recipe, gen 4305 -> "
                   "4306 on the same e098 base) must clear the 0.7 "
                   "root-strength gate at construction AND sustain the "
                   "onset-monitor organ's self-maintaining rhythm (g2c/g2d "
                   "readout, wash seed 10902 held) for 'the organ works' to "
                   "be licensed as an architecture claim at n=2 roots.",
        "delta_vs_g2d": {
            "g2d": "wash-draw seeds 10903/10904 on the LOCKED root — the "
                   "rhythm is wash-draw-robust at n=3 seeds, one root",
            "g2e": "a fresh ROOT draw (install gen 4306, same base/recipe) "
                   "with the wash seed HELD at 10902 — the rhythm's "
                   "root-generality, the architecture claim's last rung",
            "the_knob": "G2.build_root's gen_seed only: 4305 -> 4306 "
                        "(e184's +1 convention); base init, install windows, "
                        "cue pool and the consolidation stream (seed 10901) "
                        "are shared with the locked root (siblings, not "
                        "strangers — see honesty)",
        },
        "smoke": SMOKE,
        "threads": torch.get_num_threads(),
        "cpu_only": True,
        "cfg": g2cm["cfg"],
        "compute": {"cpu_only": True, "cuda_visible_devices": "-1",
                    "threads": torch.get_num_threads(),
                    "stagger_s": STAGGER_S, "train_cap_s": G2E_CAP_S,
                    "meas_grid_n": len(MEAS_GRID),
                    "meas_grid_every": PHASE_EVERY,
                    "cooldowns": "0 s (CPU-only; g2b/g2c/g2d convention)"},
        "fresh_root": {
            "recipe": f"{BASE_CK} + e043-Dmix install ({inst_steps} steps, "
                      f"gen {FRESH_GEN}) + e113 jitter consolidation (300 "
                      f"steps, seed {CONS_SEED}) — g2's r1 VERBATIM, one gen "
                      f"knob changed",
            "install_traj": rec["install_traj"],
            "install_wall_s": rec["install_wall_s"],
            "install_device": rec["install_device"],
            "cons_traj": rec["cons_traj"],
            "cons_device": rec["cons_device"],
            "root_monitor_onset": root_mon,
            "locked_root_monitor_onset": locked_root_mon,
            "gates": {"G_SPLICE": G_SPLICE, "G_POOL": G_POOL,
                      "G_ROOTF": G_ROOTF},
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
        "cells": {m: strip_cell(m) for m in cells},
        "phase_fresh_root": ph,
        "clauses_fresh_root": cl,
        "reference_leg": {
            "seed": FREEZE_SEED, "root": "locked (g2_root.pt via g2c)",
            "F_REF": F_REF, "clauses": ref_clauses, "phase": ph_ref,
            "events": g2c_events, "event_log": g2c_cell["event_log"],
            "source": ("g2c's stored dense realization (fidelity-gated to "
                       "g2b's own); embedded leg, not re-adjudicated"),
        },
        "schedules_differ": SCHEDS_DIFFER,
        "gates": {"G_NAMEFREE": G_NAMEFREE, "G_REPLAY": G_REPLAY,
                  "G_STEP": G_STEP, "CONTRAST": contrast, "hard": hard},
        "registered_bars": REGISTERED_BARS,
        "registered_prediction": {
            "pre_run": "the GATE is the open risk (r1 landed 0.7106, barely "
                       "over 0.7; e157's ported root 0.591) — predict the "
                       "fresh draw lands 0.62-0.78; if it clears, predict "
                       "8-12 events, spacing 24-40, cycle-median 0.5-0.7, "
                       "duty 55-75% (g2d's closed-loop dominance argument "
                       "transfers from wash-draws to root-draws). Both "
                       "outcomes pre-registered; no shopping.",
            "discriminating_observation": "the fresh root's construction "
                                          "onset monitor vs the locked "
                                          "root's 0.609 (a root starting "
                                          "below theta opens the gate "
                                          "immediately — texture), and the "
                                          "fresh cell's per-cycle troughs vs "
                                          "g2d's 0.08-0.60 — shallow "
                                          "troughs license with room; deep "
                                          "troughs extend the amplitude "
                                          "lottery to roots while the "
                                          "median decides.",
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
                                 "[20, 45] (g2b/g2c/g2d's registered band)",
                "median_clause": "cycle-median over dense in-cycle samples "
                                 "of complete inter-event intervals >= 0.5 "
                                 "(g2c's convention, same grid)",
            },
            "bars": {"ORGAN_REPLICATES": ORGAN_REPLICATES,
                     "ORGAN_ROOT_BOUND": ORGAN_ROOT_BOUND,
                     "ROOT_DRAW_BOUND_residual": ROOT_DRAW_BOUND,
                     "near_texture": cl["near_texture"]},
            "geo_texture_cycle_medians": geo_texture,
            "contrast_coreport": contrast,
            "pre_rhythm_min": ph["pre_rhythm"]["min"],
            "verdict": verdict, "clause": clause,
        },
        "honesty": [
            "WHAT WAS REPLICATED: the ROOT DRAW at n=2 (the locked root + "
            "one fresh install+consolidation draw), wash seed 10902 held "
            "fixed — the licensed claim is 'the organ's rhythm generalizes "
            "across root draws of this recipe'; it does NOT re-establish "
            "wash-draw robustness on the fresh root (g2d's n=3 was on the "
            "locked root; the fresh root's rhythm is a single wash-seed "
            "point estimate).",
            "SIBLING ROOTS, NOT STRANGERS: the fresh draw renews the install "
            "batch trajectory (gen 4306) only — base init, install windows, "
            "cue pool and the consolidation stream (seed 10901) are shared; "
            "a base-seed redraw (e098_base_s4306.pt exists) and a "
            "consolidation-seed redraw remain open rungs.",
            "DEVICE TEXTURE: the fresh root is CPU-only end-to-end; the "
            "locked root was cuda-install + cpu-consolidation — the two "
            "roots differ in float texture as well as draw; every bar sits "
            "at order-of-magnitude separations.",
            "THE INTERVENTION CHECK RAN ON THE FRESH ROOT: CELL-BASE (gate "
            "off, same seed) — behavior with vs without the organ is the "
            "honest-reflex test; its +50 value is co-reported (not a "
            "registered bar).",
            "THE REFERENCE LEG IS STORED, NOT RERUN: g2c's dense trace was "
            "fidelity-gated to g2b's realization; F_REF re-pools it "
            f"(max|diff| {F_REF['abs_diff']}) as a load check only.",
            "NO BAR WIDENING: the dispatch's 'in/near' is co-reported "
            "texture; the two registered outcomes stay exhaustive over "
            "their domain, with the pre-registered ROOT-DRAW-BOUND residual "
            "for the both-fail corner.",
            "GEO TEXTURE IS CO-REPORTED, NEVER ADJUDICATED: the cycle-median "
            "is pooled per geo key and all three are recorded; the "
            "adjudicated key is the frozen construction-argmax ruler. A "
            "geo-shifted root (this run: argmax g+12 at construction while "
            "g0 reads higher under wash) can pass on one key and fail on "
            "another — the frozen rule decides, the others are texture.",
            "n=2 ROOTS is the lab's replicate standard for this claim "
            "(e184/e187-style n=2 new draws + n=1 reference); it does not "
            "measure the root-draw pass rate — recorded as the bound it is.",
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
