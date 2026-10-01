"""G2G2 — THE SEED LADDER (the R60-critic's exact replication protocol for
the +0.07 autonomy number; scratch/r60_critic.md ATTACK 3, the licensed
protocol; QUEUE: "the g2g seed ladder (CPU-deterministic, the +0.03x3 bar)").

Builds on: g2g (the parent cell — the machinery, the +0.0758/+0.072 at 1x,
the frozen ruler + pooling this file reuses VERBATIM), g2c (the dense grid
+ the CPU-lane convention), g2d (the wash-seed replicate convention +
e184's seed lineage), T152 (the seed ladder LICENSED), the R60 critic's
ATTACK 3 (the protocol below), the 2x float-fragility lesson (g2g's
cross_run: one near-threshold GPU check moved a pooled median 0.24->0.17).
WHAT IS NEW: the +0.07 has never met a second seed — this cell runs the
critic's head-to-head seed ladder (>=3 FRESH wash seeds x {organ,
count-matched fixed} at 1x ONLY, CPU-DETERMINISTIC) plus ONE paired-batch
control arm (the fixed schedule replaying the organ's own realized event
batches — the count-vs-batch parity co-read the critic named).

THE PROTOCOL (the critic's, frozen — scratch/r60_critic.md ATTACK 3):
  ">=3 fresh wash seeds x {organ, matched-count fixed} on the SAME locked
  root, 1x threat only (the run-stable leg), CPU-deterministic (retires
  the float class), same frozen ruler; co-report per-seed deltas + sign
  consistency; ONE seed additionally run with a paired-batch control (the
  fixed arm replays the organ's realized event batches) to close the
  count-vs-batch seam."

THE BAR (frozen, VERBATIM — the dispatch's letter, the critic's own
threshold):
  AUTONOMY-REPLICATES: "fires if the organ beats the count-matched fixed
      schedule with the SAME SIGN in 3/3 seeds AND the in-cell median
      delta >= +0.03 of cycle-median — the autonomy number licensed at
      n>=3; 'worth' returns to the paper's clause with the seed scope.
      Any other outcome: the +0.07 stays a single-run reading and the
      clause carries the honesty."
  (critic's original wording, same bar: "AUTONOMY-REPLICATES = same sign
  in >=3/3 seeds with median delta >= +0.03 (the borrowed spread, now
  measured in-cell); AUTONOMY-ANECDOTE = mixed signs. Until then, the
  abstract may not carry +0.07 as 'worth' — only 'won by +0.07 in the one
  root tested'.")

OPERATIONALIZATION (frozen here, BEFORE the run; every sub-boolean reported):
  - the 3 fresh wash seeds = 10909, 10910, 10911 — the NEXT FREE DRAWS of
    the locked 109xx lineage (10902 the locked wash seed = g2c/g2g's
    parent; 10903/10904 g2d's wash seeds; 10905/10906 e152r's straddle +
    g3R/g4R's installs; 10907/10908 g1bR's wash seeds — no prior use).
  - per seed: O_s = G2.run_cell('g2', ..., seed=s, MEAS_GRID) — the organ
    VERBATIM (refractory 24, onset monitor, lr 1x, the g2c dense grid);
    k_s = max(1, floor(300 / n_organ_events)); F_s = run_cell('sched',
    SCHED_K=k_s, seed=s) — count-matched per seed (the parent cell's own
    convention, VERBATIM).
  - a seed is ADJUDICATED iff n_organ >= 1 AND |n_fixed - n_organ| <= 1
    (g2g's count-parity convention VERBATIM); dropped seeds are recorded,
    never adjudicated.
  - delta_s = organ cycle-median - fixed cycle-median; cycle-median =
    g2d's pooling VERBATIM (phase_analysis copied from g2g): median of the
    ruler (g0, the frozen root rule) over all dense in-cycle samples of
    the COMPLETE inter-event intervals; pre-rhythm + partial tail
    co-reported, never adjudicated.
  - AUTONOMY-REPLICATES = (n_adjudicated == 3) AND (delta_s > 0 in 3/3
    seeds) AND (median of the 3 deltas >= +0.03).
  - AUTONOMY-ANECDOTE = NOT AUTONOMY-REPLICATES (the exhaustive pair; the
    critic's "mixed signs" plus every other failure mode — parity drop,
    median short — each as its own sub-boolean; no bar shopping).
  - the +0.03 threshold IS g2d's borrowed cross-seed spread (organ
    cycle-medians 0.587-0.615 across 3 wash seeds, ~0.03) — this cell
    measures the spread IN-CELL (organ spread + delta spread co-reported).
  - PAIRED-BATCH ARM (seed 10909 — the FIRST seed in the frozen ladder
    order, chosen before any seed ran): the fixed schedule (SAME k_10909,
    SAME firing steps as F_10909) whose event batches are the organ
    O_10909's REALIZED event draws (the exact ix/aj/rj index tuples the
    organ's generator produced at its events, recorded pass-through by a
    torch.randint recorder — zero arithmetic change), event i of the
    schedule replaying the organ's event-i draws. If the schedule fires
    one MORE time than the organ realized (n_fixed = n_organ + 1), the
    shortfall firing draws fresh from the generator ONCE (counted +
    recorded). CO-READ ONLY — no bar lives on the paired arm (the critic's
    "co-report"). Named residual: the WASH stream remains
    schedule-dependent in every arm (no arm pairs wash draws — disclosed).
  - DEVICE CO-READ (no bar, NOT a ladder seed): seed 10902 (the parent
    seed) organ+fixed pair re-run on CPU — the parent +0.0758/+0.072 was
    GPU; this leg reads the device class's move on the SAME seed and
    soft-x-checks the organ arm against g2c's stored realization
    (n_shared_event_steps + cycle-median diff; g2g's G_XCHECK form).

REGISTERED PREDICTION (before running): g2d's organ cross-seed spread was
0.587-0.615 at 10902-10904 and the parent fixed read 0.543 — predict each
fresh organ fires 9-12 events (spacings >= 24), cycle-median 0.55-0.65;
deltas POSITIVE in 3/3 with median in [+0.03, +0.09] — AUTONOMY-REPLICATES
fires. PAIRED: delta_organ-paired within +-0.03 of delta_organ-fixed (the
+0.07 is timing, not draw luck). REF: seed-10902-CPU organ reproduces
g2c's schedule >= 9/10 steps and the CPU parent delta lands within +-0.03
of the GPU +0.0758/+0.072. RISKS (pre-registered): a seed whose dips
deepen could fire fewer/later events (k mismatch -> parity drop ->
ANECDOTE by protocol, recorded as the bound it is); a seed could fire 0
events (F arm skipped + recorded); CPU-vs-GPU float class could move the
parent delta (that is what the ref leg measures).

ENVELOPE (the dispatch): CPU-ONLY, DETERMINISTIC (CUDA_VISIBLE_DEVICES=-1
before any import; torch threads 4 — the g2b/g2c/g2d CPU-lane convention;
fp32 end-to-end; G_DET = two identical full-length wash-only trainings
bit-compared, doubling as the train-only timing receipt); runs sequential
(cap 600 s/arm, g2b's bound — g2c's own dense CPU cell was 188.7 s wall
including its 152 light evals; train-only is measured and recorded); no
GPU -> no cooldowns; resumable (completed arms adopted bit-trusted from a
prior interrupted metrics.json; every arm is deterministic, a re-run is
identical by construction).

Outputs: runs/g2g2/{metrics.json (PROGRESSIVE — rewritten after every
arm), seed_ladder_head_to_head.png, paired_batch_control.png}.
No NOTES/THINKING/QUEUE/STATE edits; commits + push per the dispatch.

Run:  python lab/g2g2_seed_ladder.py    (G2G2_SMOKE=1 shakedown)
"""
from __future__ import annotations

import json
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # CPU-DETERMINISTIC lane —
                                                  # set BEFORE any torch/g2
                                                  # import (the 2x lesson)

import sys                                        # noqa: E402
from pathlib import Path                          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                # noqa: E402
import torch                                      # noqa: E402
import torch.nn.functional as F                   # noqa: E402

import g2_rehearsal_organ as G2                   # noqa: E402 — the organ
import common                                     # noqa: E402
from common import run_dir, save_json             # noqa: E402

torch.set_num_threads(4)                          # AFTER the G2 import
                                                  # (which sets 8) — the
                                                  # g2b/g2c/g2d CPU lane

import matplotlib                                  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
import matplotlib.gridspec as gridspec             # noqa: E402

SMOKE = os.environ.get("G2G2_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ---- frozen constants (inherited; restated for the record) --------------------
FREEZE_SEED = G2.FREEZE_SEED                      # 10902 — the locked lineage
SEEDS: tuple[int, ...] = (10909,) if SMOKE else (10909, 10910, 10911)
PAIRED_SEED = 10909                               # FIRST in ladder order —
                                                  # frozen BEFORE any seed ran
REF_SEED = FREEZE_SEED                            # 10902 device co-read (no
                                                  # bar, not a ladder seed)
LR_MULT = 1.0                                     # 1x THREAT ONLY (the
                                                  # run-stable leg; the 2x
                                                  # leg is not re-tested
                                                  # by design)
RULER_GEO = 0                                     # g2_root.pt meta (frozen)
RULER_KEY = {-12: "gm12", 0: "g0", 12: "gp12"}[RULER_GEO]
SHUT_BAR, MAINTAIN_BAR = G2.SHUT_BAR, G2.MAINTAIN_BAR   # 0.27 / 0.50
SPACING_BAND = G2.SPACING_BAND                    # (20, 45)
N_STEPS = 36 if SMOKE else 300
PHASE_EVERY = 2                                   # g2c's dense stride
MEAS_GRID: tuple[int, ...] = tuple(sorted(        # g2c's grid VERBATIM
    set((1, 25) + tuple(range(PHASE_EVERY, N_STEPS + 1, PHASE_EVERY)))))
if SMOKE:
    MEAS_GRID = tuple(sorted(set((1, 2, 4, 36))))
G2G2_CAP_S = 600.0                                # g2b's bound (see
                                                  # docstring envelope)
BAR_DELTA = 0.03                                  # the frozen bar threshold
                                                  # (g2d's borrowed spread)
G2_METRICS = G2.E43.REPO / "runs" / "g2" / "metrics.json"
G2C_METRICS = G2.E43.REPO / "runs" / ("g2c_smoke" if SMOKE else "g2c") \
    / "metrics.json"
G2B_METRICS = G2.E43.REPO / "runs" / ("g2b_smoke" if SMOKE else "g2b") \
    / "metrics.json"
G2G_METRICS = G2.E43.REPO / "runs" / ("g2g_smoke" if SMOKE else "g2g") \
    / "metrics.json"

REGISTERED_BARS = {
    "autonomy_replicates": "AUTONOMY-REPLICATES: fires if the organ beats "
        "the count-matched fixed schedule with the SAME SIGN in 3/3 seeds "
        "AND the in-cell median delta >= +0.03 of cycle-median — the "
        "autonomy number licensed at n>=3; 'worth' returns to the paper's "
        "clause with the seed scope. Any other outcome: the +0.07 stays a "
        "single-run reading and the clause carries the honesty.",
    "autonomy_anecdote": "AUTONOMY-ANECDOTE: every other outcome (mixed "
        "signs, median delta < +0.03, or a seed dropped by count parity) — "
        "the +0.07 stays a single-run reading and the clause carries the "
        "honesty.",
    "critic_source": "scratch/r60_critic.md ATTACK 3 (the licensed exact "
        "replication): 'registered bar: AUTONOMY-REPLICATES = same sign in "
        ">=3/3 seeds with median delta >= +0.03 (the borrowed spread, now "
        "measured in-cell); AUTONOMY-ANECDOTE = mixed signs. Until then, "
        "the abstract may not carry +0.07 as \"worth\" — only \"won by "
        "+0.07 in the one root tested\".' Frozen — no bar shopping.",
    "paired_arm": "the paired-batch arm and the 10902 device co-read are "
        "CO-READS (the critic's 'co-report'); NO bar lives on either.",
}

deviations: list[str] = [
    "CPU-DETERMINISTIC LANE (the dispatch + the 2x float-fragility lesson): "
    "CUDA_VISIBLE_DEVICES=-1 before any import, torch threads 4 (the "
    "g2b/g2c/g2d CPU-lane convention, reset AFTER the G2 import which sets "
    "8), fp32 end-to-end. G_DET proves determinism ON THIS WORKLOAD: two "
    "identical full-length (300-step) wash-only trainings on the root body "
    "bit-compared (state_dicts torch.equal) — and doubles as the "
    "train-only timing receipt for the envelope (recorded in metrics).",
    "G2.pick_dev is patched to ALWAYS return CPU and G2.GPU_PARKED=True "
    "(so run_cell's device records read 'cpu', never a stale 'cuda'); "
    "G2.MIDRUN_POLL_EVERY is left untouched — the migration branch is "
    "unreachable on CPU (dev.type != 'cuda').",
    "THE ORGAN VERBATIM: G2.run_cell('g2', ...) unmodified (refractory 24, "
    "onset-only monitor via the G2GNet patch = g2b's arithmetic VERBATIM, "
    "lr 1e-3 = G2.FT_LR untouched, THETA_OPEN 0.5, CADENCE 4 — asserted at "
    "entry). THE FIXED ARM: G2.run_cell('sched', ...) with exactly one "
    "dial set (G2.SCHED_K = k_s, count-matched per seed; restored to 32 "
    "after) — the parent cell's own convention.",
    "THE PAIRED ARM: the loop is G2.run_cell's 'sched' branch copied "
    "VERBATIM with ONE change — the event-draw triple (ix, aj, rj) is the "
    "organ's RECORDED i-th event draw (injected; no generator consumption "
    "at firings); a shortfall firing (n_fixed = n_organ + 1) draws fresh "
    "ONCE (counted + recorded). The organ's draws are captured by a "
    "pass-through torch.randint recorder (zero arithmetic change; parse "
    "gate-checked by G_DRAWS: the per-step draw pattern must match the "
    "realized schedule exactly, else the paired arm does not run).",
    "THE 10902 DEVICE CO-READ (O_ref/F_ref): seed 10902 organ+fixed on CPU "
    "— the parent +0.0758/+0.072 was GPU; this leg reads the device "
    "class's move on the SAME seed and soft-x-checks the organ arm against "
    "g2c's stored CPU realization (n_shared_event_steps; never a gate). "
    "NOT a ladder seed; no bar.",
    "sds NOT RETAINED: run_cell builds dense state_dicts for its light "
    "evals; g2g deleted them post-arm (no rider there) — this cell has no "
    "rider either, so the paired-arm copy simply never stores them (the "
    "measurement rows are identical either way).",
    "TRAIN CAP 600 s/arm (g2b's precedent; run_cell's cap counts the "
    "interleaved dense CPU light-evals — g2c's own dense cell was 188.7 s "
    "wall; train-only is measured by G_DET and recorded). No cooldowns "
    "(no GPU); arms run sequential per the dispatch.",
    "RESUME: a prior interrupted run's metrics.json is loaded at start and "
    "its COMPLETED arms (records with a full cell block) are adopted "
    "bit-trusted and not re-run; the determinism protocol makes re-runs "
    "identical by construction. Adjudication is recomputed from the "
    "adopted arms at the end.",
    "Smoke mode trims: 36-step cells, grid {1,2,4,36}, one seed (10909) "
    "with organ+fixed+paired, 36-step G_DET, no ref pair — nothing "
    "adjudicated.",
]

# ---------------------------------------------------------------- the delta
# g2b's G2BNet.monitor VERBATIM (the onset-only fix) on g2's G2Net — copied
# from g2g (which copied g2b; the CPU lane imports g2b's arithmetic but we
# keep g2g's standalone form so this file owns it).


class G2GNet(G2.G2Net):
    """g2's organ with g2b's ONSET-ONLY monitor (the current organ form);
    every arithmetic step is g2b's monitor VERBATIM."""

    @torch.no_grad()
    def monitor(self, c: int, dev=None) -> float:
        dev = dev or next(self.body.parameters()).device
        pool = self.cue_pool.to(dev).long()
        mask = self.cue_mask.to(dev)
        blocks = {j: G2.JITTERS.index(j) for j in G2.MON_OFFSETS}
        idx = []
        for k in range(G2.K_MON):
            b = blocks[G2.MON_OFFSETS[k % len(G2.MON_OFFSETS)]] * 60
            idx.append(b + ((c * G2.K_MON + k) % 60))
        idx = torch.tensor(idx, device=dev)
        w = pool[idx]
        x, y = w[:, :-1], w[:, 1:]
        logits, _ = self.body(x)
        pr = F.softmax(logits, -1)
        ptrue = pr.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        m = mask[idx]
        onset = m & (m.cumsum(1) == 1)     # THE FIX (g2b's): 1 of the 7 name
                                           # positions — p(Z | ctx)
        return float(ptrue[onset].mean())


G2.G2Net = G2GNet                 # run_cell builds the organ via this module
                                  # global (g2b/g2g's own patch)

# CPU-only device policy: never CUDA, recorded per arm.
G2.GPU_PARKED, G2.PARK_REASON = True, "g2g2 CPU-deterministic protocol"
G2.pick_dev = lambda tag: CPU

# ---------------------------------------------------------------- protocol
# g2b's rebuild_protocol VERBATIM (copied via g2g — pure CPU construction
# arithmetic; importing g2b directly would be equivalent but this keeps the
# file standalone in g2g's lineage).


def rebuild_protocol() -> dict:
    corpus = common.CharCorpus(G2.E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    corpus_zeph = train_text.count("ZEPH")
    assert corpus_zeph == 0, f"corpus contains ZEPH x{corpus_zeph}"

    host_occ = []
    for host in G2.HOSTS:
        for p in G2.E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + G2.POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    import random as _random
    rng = _random.Random(G2.E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    name_ids = corpus.encode(G2.NAME)
    L = len(G2.NAME)

    def offset_pool(j: int):
        wins = []
        for p, h in install_occ:
            pre = train_ids[p - G2.PRE - j: p]
            post = train_ids[p + len(h): p + len(h) + G2.POST_CAP - j]
            w = torch.cat([pre, name_ids, post])
            assert len(w) == G2.BLOCK
            wins.append(w)
        px = torch.stack(wins)
        pm = torch.zeros(len(wins), G2.BLOCK - 1, dtype=torch.bool)
        pm[:, G2.PRE - 1 + j: G2.PRE - 1 + j + L] = True
        return px, pm

    jit_pools = {j: offset_pool(j) for j in G2.JITTERS}
    jit_pool_x = torch.cat([jit_pools[j][0] for j in G2.JITTERS])
    jit_pool_mask = torch.cat([jit_pools[j][1] for j in G2.JITTERS])
    pool_183_x, _ = offset_pool(G2.RETEACH_J)

    bat_ids, held_bat = {}, {}
    for j in G2.GEOS:
        cs = [train_text[p - G2.PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
        hs = [train_text[p - G2.PRE - j: p] for p, _ in held_occ]
        held_bat[j] = torch.stack([corpus.encode(c) for c in hs])
    r_eval_x, r_eval_y = G2.val_windows(val_ids, val_text, 60, G2.R_EVAL_SEED)

    arng = _random.Random(G2.E170_ANCHOR_SEED)
    hi_start = len(train_ids) - G2.BLOCK - 2
    n_starts = []
    while len(n_starts) < 16:
        s = arng.randrange(hi_start)
        if any(f in train_text[s: s + G2.BLOCK + 1]
               for f in G2.ANCHOR_FORBIDDEN):
            continue
        n_starts.append(s)
    anchor_neutral = torch.stack([train_ids[s: s + G2.BLOCK]
                                  for s in n_starts])
    return {"corpus": corpus, "itos": itos, "zid": zid,
            "train_ids": train_ids, "install_occ": install_occ,
            "jit_pool_x": jit_pool_x, "jit_pool_mask": jit_pool_mask,
            "pool_183_x": pool_183_x, "bat_ids": bat_ids,
            "held_bat": held_bat, "r_eval_xy": (r_eval_x, r_eval_y),
            "anchor_neutral": anchor_neutral, "n_starts": n_starts}


# ---------------------------------------------------------------- analysis
def phase_analysis(traj: dict, ev_steps: list[int], n_steps: int) -> dict:
    """g2d's phase_analysis VERBATIM (copied from g2g): complete inter-event
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
    pre_ss, pre_v = seg(0, ev_steps[0] if ev_steps else n_steps + 1)
    tail_ss, tail_v = (seg(ev_steps[-1] + 1, None) if ev_steps else ([], []))
    binned = []
    for ib in range(8):
        sel = [v for ph, v in zip(pooled_ph, pooled_v)
               if ib / 8 <= ph < (ib + 1) / 8]
        binned.append({"center": round((ib + 0.5) / 8, 4), "n": len(sel),
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
                         "ruler_at_final_step": traj[n_steps][RULER_KEY]
                         if n_steps in traj else None},
    }


# ---------------------------------------------------------------- the draw
# recorder (pass-through; zero arithmetic change) + the parse gate

class DrawRecorder:
    """Records every torch.randint call made inside its context (args +
    values). run_cell's only randint calls are the loop's batch draws, in
    step order — the parse below reconstructs per-event draw triples."""

    def __init__(self):
        self.calls: list[dict] = []

    def __enter__(self):
        self._orig = torch.randint
        torch.randint = self._patched            # type: ignore[assignment]
        return self

    def _patched(self, *a, **k):
        out = self._orig(*a, **k)
        self.calls.append({"high": int(a[0]), "size": list(a[1]),
                           "vals": out.tolist()})
        return out

    def __exit__(self, *exc):
        torch.randint = self._orig
        return False


def parse_event_draws(calls: list[dict], ev_steps: list[int],
                      steps_ran: int) -> dict:
    """Walk run_cell's realized schedule; consume the recorded draws in
    order (event steps: ix(16,high=300)+aj(8,high=16)+rj(8,high=rj_high);
    wash steps: aj(16,high=16)+rj(16,high=rj_high)). Returns the per-event
    draw triples + the G_DRAWS gate."""
    n_jit, n_anc = 300, 16                       # run_cell's own constants
    rj_high = len(G2_P["train_ids"]) - G2.BLOCK - 1
    ev_set = set(ev_steps)
    out, i, ok, why = [], 0, True, []
    for step in range(1, steps_ran + 1):
        if step in ev_set:
            trip = calls[i:i + 3]
            if len(trip) == 3 and trip[0]["high"] == n_jit \
                    and trip[0]["size"] == [G2.RP_NAME_BS] \
                    and trip[1]["high"] == n_anc \
                    and trip[1]["size"] == [G2.RP_ANCH_BS // 2] \
                    and trip[2]["high"] == rj_high \
                    and trip[2]["size"] == [G2.RP_ANCH_BS // 2]:
                out.append({"step": step, "ix": trip[0]["vals"],
                            "aj": trip[1]["vals"], "rj": trip[2]["vals"]})
                i += 3
            else:
                ok, _ = False, why.append(
                    f"step {step}: event draw pattern mismatch")
                break
        else:
            pair = calls[i:i + 2]
            if len(pair) == 2 and pair[0]["high"] == n_anc \
                    and pair[0]["size"] == [G2.WASH_ANCH_BS] \
                    and pair[1]["high"] == rj_high \
                    and pair[1]["size"] == [G2.WASH_RAND_BS]:
                i += 2
            else:
                ok, _ = False, why.append(
                    f"step {step}: wash draw pattern mismatch")
                break
    return {"draws": out, "n_calls": len(calls), "consumed": i,
            "n_events_parsed": len(out), "n_events_expected": len(ev_steps),
            "all_calls_consumed": bool(i == len(calls)),
            "pass": bool(ok and i == len(calls)
                         and len(out) == len(ev_steps)),
            "why": why}


# ---------------------------------------------------------------- paired arm
def run_cell_paired(root_body_sd: dict, cue_pool: torch.Tensor,
                    cue_mask: torch.Tensor, anchor_neutral, train_ids, itos,
                    r_eval_xy, bat_ids, zid, seed: int,
                    ckpt_steps: tuple[int, ...], sched_k: int,
                    event_draws: list[dict]) -> dict:
    """G2.run_cell's 'sched' branch VERBATIM with exactly ONE change (frozen
    in the docstring BEFORE the run): at each firing the event batch triple
    (ix, aj, rj) is the ORGAN'S RECORDED i-th event draw (injected; NO
    generator consumption at firings); a shortfall firing (schedule fires
    more often than the organ realized events) draws fresh ONCE (counted).
    Wash branch, optimizer, clip, dense light evals, bookkeeping, and post-
    processing are byte-faithful to run_cell('sched')."""
    tag = "CELL-PAIRED"
    dev = CPU
    cap = G2.TRAIN_CAP_S
    ckpt_set = set(ckpt_steps)
    n_steps = ckpt_steps[-1]
    gen = torch.Generator().manual_seed(seed)
    n_anc = anchor_neutral.shape[0]
    n_jit = cue_pool.shape[0]

    net = G2.TinyGPT(G2.F2_CFG)                  # organ-less, exactly like
    net.load_state_dict(root_body_sd)            # run_cell's sched branch
    net = net.to(dev)
    body = net
    pool_x, pool_m = cue_pool, cue_mask
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=G2.FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    evl = G2.TinyGPT(G2.F2_CFG)                  # CPU eval twin (body only)
    evl.load_state_dict(root_body_sd)
    evl.eval()

    traj: list[dict] = []
    monitor_trace: list[dict] = []
    event_log: list[dict] = []
    last_event = 0
    n_events = n_checks = zeph_checks = 0
    replay_checks = {"events": 0, "mask7_ok": 0, "anchors_8_8_ok": 0,
                     "echo_j0_ok": 0, "injected_draws": 0,
                     "fallback_draws": 0}
    t_start = time.time()
    step = 0

    def light_eval(step_, was_event, why):
        sd_cpu = {k: v.detach().cpu().clone()
                  for k, v in body.state_dict().items()}
        evl.load_state_dict(sd_cpu)
        evl.eval()
        cells = {j: G2.battery_cell(evl, bat_ids[j], zid) for j in G2.GEOS}
        ce_r = G2.ce_fixed_cpu(evl, *r_eval_xy)
        row = {"step": step_, "why": why,
               "gm12": cells[-12]["mean_pz"], "g0": cells[0]["mean_pz"],
               "gp12": cells[12]["mean_pz"],
               "frac_argmax_z": cells[0]["frac_argmax_z"],
               "ce_r": ce_r, "monitor": None,
               "n_events_so_far": n_events, "was_event": bool(was_event),
               "elapsed_s": round(time.time() - t_start, 1)}
        traj.append(row)
        log(f"  [{tag}] {why} +{step_:4d} ruler-cells "
            f"g-12 {row['gm12']:.4f} g0 {row['g0']:.4f} g+12 "
            f"{row['gp12']:.4f} CE_R {ce_r:.4f} (events {n_events})")

    for step in range(1, n_steps + 1):
        kind = "event" if (step % sched_k == 0) else "wash"   # the sched
                                                              # predicate
                                                              # VERBATIM
        if kind == "event":
            # ---- THE ONE CHANGE: the organ's recorded i-th draw -----------
            if n_events < len(event_draws):
                d = event_draws[n_events]
                ix = torch.tensor(d["ix"], dtype=torch.long)
                aj = torch.tensor(d["aj"], dtype=torch.long)
                rj = torch.tensor(d["rj"], dtype=torch.long)
                replay_checks["injected_draws"] += 1
            else:
                ix = torch.randint(n_jit, (G2.RP_NAME_BS,), generator=gen)
                aj = torch.randint(n_anc, (G2.RP_ANCH_BS // 2,),
                                   generator=gen)
                rj = torch.randint(len(train_ids) - G2.BLOCK - 1,
                                   (G2.RP_ANCH_BS // 2,), generator=gen)
                replay_checks["fallback_draws"] += 1
            for s in rj:                     # name-free verify (random half)
                txt = "".join(itos[int(c)] for c in
                              train_ids[s: s + 64]) + \
                      "".join(itos[int(c)] for c in
                              train_ids[s + 192: s + G2.BLOCK])
                if "ZEPH" in txt:
                    zeph_checks += 1
            replay_checks["events"] += 1
            replay_checks["anchors_8_8_ok"] += int(
                aj.numel() == G2.RP_ANCH_BS // 2
                and rj.numel() == G2.RP_ANCH_BS // 2)
            per_win = pool_m[ix].sum(-1)
            replay_checks["mask7_ok"] += int(bool(
                (per_win == len(G2.NAME)).all().item()))
            loss = G2.replay_loss(body, pool_x, pool_m, ix, anchor_neutral,
                                  aj, train_ids, rj, dev)
            last_event = step
            n_events += 1
            event_log.append({"n": n_events, "step": step,
                              "mode": "sched_paired",
                              "pre_event_monitor": None})
            log(f"  [{tag}] EVENT {n_events} @s{step} "
                f"(injected draw {n_events})")
        else:
            aj = torch.randint(n_anc, (G2.WASH_ANCH_BS,), generator=gen)
            rj = torch.randint(len(train_ids) - G2.BLOCK - 1,
                               (G2.WASH_RAND_BS,), generator=gen)
            for s in rj:
                txt = "".join(itos[int(c)] for c in
                              train_ids[s: s + 64]) + \
                      "".join(itos[int(c)] for c in
                              train_ids[s + 192: s + G2.BLOCK])
                if "ZEPH" in txt:
                    zeph_checks += 1
            loss = G2.wash_loss(body, anchor_neutral, aj, train_ids, rj, dev)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step in ckpt_set:
            light_eval(step, kind == "event", "ckpt")
        if (time.time() - t_start) > cap:
            log(f"  [{tag}] time cap {cap:.0f}s at s{step}")
            break
    net.eval()
    del net

    for ev in event_log:                     # run_cell's post bookkeeping
        post = next((t for t in monitor_trace
                     if t["step"] >= ev["step"] + G2.REFRACTORY), None)
        ev["post_refractory_monitor"] = post["monitor"] if post else None
        ev["next_check_fired"] = post["fired"] if post else None
        plus = next((r for r in traj
                     if r["step"] == min(ev["step"] + G2.REFRACTORY,
                                         n_steps)), None)
        ev["ruler_cells_at_plus_refractory"] = (
            {k: plus[k] for k in ("gm12", "g0", "gp12")} if plus else None)
        nxt = next((r for r in traj if r["why"] == "ckpt"
                    and r["step"] >= ev["step"]), None)
        ev["ruler_cells_at_next_ckpt"] = (
            {k: nxt[k] for k in ("gm12", "g0", "gp12")} if nxt else None)
    spacings = [b["step"] - a["step"] for a, b in zip(event_log, event_log[1:])]
    return {"mode": "sched_paired", "final_sd": None, "traj": traj,
            "monitor_trace": monitor_trace, "event_log": event_log,
            "event_spacings": spacings, "n_events": n_events,
            "n_checks": n_checks, "realized_r": n_events / max(step, 1),
            "steps_ran": step, "seed": seed,
            "zeph_violations": zeph_checks, "replay_checks": replay_checks,
            "initial_device": "cpu", "final_device": "cpu",
            "time_cap_s": cap}


# ---------------------------------------------------------------- state
ARMS: dict = {}                 # name -> record (progressive)
ARM_ORDER: list[str] = []
PLAN: list[str] = []
WRITE_LOG: list[str] = []
G2_P: dict = {}                 # the protocol (set in main; parse gate uses)
RD = run_dir("g2g2_smoke" if SMOKE else "g2g2")


def write_metrics(state: dict, note: str) -> None:
    state["arms"] = {k: ARMS[k] for k in ARM_ORDER}
    state["arms_pending"] = [a for a in PLAN if a not in ARM_ORDER]
    state["progressive_writes"] = WRITE_LOG + [f"{common.now_iso()} {note}"]
    save_json(RD / "metrics.json", G2.E43.jsonable(state))
    WRITE_LOG.append(f"{common.now_iso()} {note}")
    log(f"[metrics] PROGRESSIVE write ({note}) -> {RD / 'metrics.json'}")


# ---------------------------------------------------------------- one arm
def arm_common(cell: dict, mode: str, seed: int, sched_k: int | None,
               refr: int) -> dict:
    """The shared record: event table, dense trace, phase pooling, gates."""
    ev_steps = [e["step"] for e in cell["event_log"]]
    traj = {r["step"]: r for r in cell["traj"]}
    ph = phase_analysis(traj, ev_steps, N_STEPS) if ev_steps else None
    gates = {
        "G_NAMEFREE": {"zeph_violations": cell["zeph_violations"],
                       "pass": bool(cell["zeph_violations"] == 0)},
        "G_REPLAY": {"n_events": cell["n_events"],
                     "checks": cell["replay_checks"],
                     "pass": bool(cell["n_events"] == 0 or
                                  (cell["replay_checks"]["mask7_ok"]
                                   == cell["n_events"]
                                   and cell["replay_checks"]["anchors_8_8_ok"]
                                   == cell["n_events"]))},
        "G_STEP": {"steps_ran": cell["steps_ran"], "expected": N_STEPS,
                   "batch": 32, "pass": bool(cell["steps_ran"] == N_STEPS)},
    }
    spac = cell["event_spacings"]
    return {
        "kind": mode, "lr_mult": LR_MULT, "lr": G2.FT_LR,
        "refractory": refr, "sched_k": sched_k, "seed": seed,
        "cell": {"mode": cell["mode"], "steps_ran": cell["steps_ran"],
                 "n_events": cell["n_events"], "n_checks": cell["n_checks"],
                 "realized_r": cell["realized_r"],
                 "event_spacings": spac, "events": ev_steps,
                 "monitor_trace": cell["monitor_trace"],
                 "event_log": cell["event_log"], "traj": cell["traj"],
                 "devices": {"initial": cell["initial_device"],
                             "final": cell["final_device"]},
                 "replay_checks": cell["replay_checks"]},
        "phase": ph,
        "event_rate": cell["n_events"] / max(cell["steps_ran"], 1),
        "spacing_min_median_max": ([min(spac), float(np.median(spac)),
                                    max(spac)] if spac else None),
        "frac_spacings_at_refractory_floor": (
            float(np.mean([s == refr for s in spac])) if spac else None),
        "frac_spacings_in_20_45": (
            float(np.mean([SPACING_BAND[0] <= s <= SPACING_BAND[1]
                           for s in spac])) if spac else None),
        "gates": gates,
    }


def run_organ_arm(name: str, seed: int, state: dict, root_sd, cue_pool,
                  cue_mask, P: dict, crosscheck=None) -> None:
    ARM_ORDER.append(name)
    log("=" * 78)
    log(f"ARM {name}: the organ VERBATIM at 1x ({G2.FT_LR:g}), refractory "
        f"{G2.REFRACTORY}, seed {seed} — {N_STEPS} steps, dense grid "
        f"{len(MEAS_GRID)} reads, CPU")
    t_arm = time.time()
    with DrawRecorder() as dr:
        cell = G2.run_cell("g2", root_sd, cue_pool, cue_mask,
                           P["anchor_neutral"], P["train_ids"], P["itos"],
                           P["r_eval_xy"], P["bat_ids"], P["zid"], seed,
                           MEAS_GRID)
    wall = time.time() - t_arm
    draws = parse_event_draws(dr.calls, [e["step"] for e in
                                         cell["event_log"]],
                              cell["steps_ran"])
    rec = arm_common(cell, "organ", seed, None, G2.REFRACTORY)
    rec["wall_s"] = round(wall, 1)
    rec["draw_parse_G_DRAWS"] = {k: draws[k] for k in
                                 ("n_calls", "consumed",
                                  "n_events_parsed", "n_events_expected",
                                  "all_calls_consumed", "pass", "why")}
    rec["event_draws"] = draws["draws"]      # the paired arm's source
    if crosscheck is not None:
        rec["G_XCHECK_vs_g2c"] = crosscheck(rec)
    ARMS[name] = rec
    ok = "PASS" if draws["pass"] else "FAIL"
    cm = rec["phase"]["cycle_median"] if rec["phase"] else None
    log(f"ARM {name} DONE in {wall:.0f}s: {cell['n_events']} events "
        f"{[e['step'] for e in cell['event_log']]}; G_DRAWS {ok}; "
        + (f"cycle-median {cm:.4f}" if cm is not None
           else "cycle-median NA (no complete cycles)"))
    for g in rec["gates"]:
        if not rec["gates"][g]["pass"]:
            log(f"  GATE {g} FAILED on arm {name}: {rec['gates'][g]}")
    write_metrics(state, f"after arm {name}")


def run_fixed_arm(name: str, seed: int, k: int, state: dict, root_sd,
                  cue_pool, cue_mask, P: dict) -> None:
    ARM_ORDER.append(name)
    log("=" * 78)
    log(f"ARM {name}: fixed schedule k={k} (count-matched), lr 1x, seed "
        f"{seed} — {N_STEPS} steps, CPU")
    G2.SCHED_K = k
    t_arm = time.time()
    cell = G2.run_cell("sched", root_sd, cue_pool, cue_mask,
                       P["anchor_neutral"], P["train_ids"], P["itos"],
                       P["r_eval_xy"], P["bat_ids"], P["zid"], seed,
                       MEAS_GRID)
    wall = time.time() - t_arm
    G2.SCHED_K = 32                          # restore
    rec = arm_common(cell, "fixed", seed, k, 24)
    rec["wall_s"] = round(wall, 1)
    ARMS[name] = rec
    cm = rec["phase"]["cycle_median"] if rec["phase"] else None
    log(f"ARM {name} DONE in {wall:.0f}s: {cell['n_events']} events "
        f"{[e['step'] for e in cell['event_log']]}; "
        + (f"cycle-median {cm:.4f}" if cm is not None
           else "cycle-median NA (no complete cycles)"))
    for g in rec["gates"]:
        if not rec["gates"][g]["pass"]:
            log(f"  GATE {g} FAILED on arm {name}: {rec['gates'][g]}")
    write_metrics(state, f"after arm {name}")


def run_paired_arm(name: str, seed: int, k: int, state: dict, root_sd,
                   cue_pool, cue_mask, P: dict, event_draws: list[dict]) \
        -> None:
    ARM_ORDER.append(name)
    log("=" * 78)
    log(f"ARM {name}: PAIRED-BATCH fixed schedule k={k} (the organ's own "
        f"event batches), lr 1x, seed {seed} — {N_STEPS} steps, CPU")
    t_arm = time.time()
    cell = run_cell_paired(root_sd, cue_pool, cue_mask, P["anchor_neutral"],
                           P["train_ids"], P["itos"], P["r_eval_xy"],
                           P["bat_ids"], P["zid"], seed, MEAS_GRID, k,
                           event_draws)
    wall = time.time() - t_arm
    rec = arm_common(cell, "paired", seed, k, 24)
    rec["wall_s"] = round(wall, 1)
    ARMS[name] = rec
    cm = rec["phase"]["cycle_median"] if rec["phase"] else None
    log(f"ARM {name} DONE in {wall:.0f}s: {cell['n_events']} events "
        f"{[e['step'] for e in cell['event_log']]} "
        f"({cell['replay_checks']['injected_draws']} injected / "
        f"{cell['replay_checks']['fallback_draws']} fallback draws); "
        + (f"cycle-median {cm:.4f}" if cm is not None
           else "cycle-median NA (no complete cycles)"))
    for g in rec["gates"]:
        if not rec["gates"][g]["pass"]:
            log(f"  GATE {g} FAILED on arm {name}: {rec['gates'][g]}")
    write_metrics(state, f"after arm {name}")


# ---------------------------------------------------------------- adjudicate
def adjudicate(state: dict) -> dict:
    per_seed: dict = {}
    deltas: list[float] = []
    for s in SEEDS:
        o_n, f_n = f"O_{s}", f"F_{s}"
        if o_n not in ARMS or f_n not in ARMS:
            continue
        n_org = ARMS[o_n]["cell"]["n_events"]
        k = ARMS[f_n]["sched_k"]
        n_fix = ARMS[f_n]["cell"]["n_events"]
        parity = bool(n_org >= 1 and abs(n_fix - n_org) <= 1)
        oc = (ARMS[o_n]["phase"]["cycle_median"]
              if ARMS[o_n]["phase"] else None)
        fc = (ARMS[f_n]["phase"]["cycle_median"]
              if ARMS[f_n]["phase"] else None)
        delta = oc - fc if (oc is not None and fc is not None) else None
        per_seed[str(s)] = {
            "n_organ": n_org, "n_fixed": n_fix, "k": k,
            "count_parity_ok": parity,
            "organ_events": ARMS[o_n]["cell"]["events"],
            "fixed_events": ARMS[f_n]["cell"]["events"],
            "organ_cycle_median": oc, "fixed_cycle_median": fc,
            "delta_organ_minus_fixed": delta,
            "adjudicated": bool(parity and delta is not None),
            "delta_positive": (bool(delta > 0)
                               if delta is not None else None),
        }
        if parity and delta is not None:
            deltas.append(delta)

    n_adj = len(deltas)
    n_pos = sum(1 for d in deltas if d > 0)
    med = float(np.median(deltas)) if deltas else None
    replicates = bool(n_adj == len(SEEDS) == 3 and n_pos == 3
                      and med is not None and med >= BAR_DELTA)
    anecdote = not replicates

    org_cms = [per_seed[str(s)]["organ_cycle_median"] for s in SEEDS
               if str(s) in per_seed
               and per_seed[str(s)]["organ_cycle_median"] is not None]

    paired = None
    p_n, o_n, f_n = f"P_{PAIRED_SEED}", f"O_{PAIRED_SEED}", f"F_{PAIRED_SEED}"
    if p_n in ARMS and o_n in ARMS and f_n in ARMS:
        pc = (ARMS[p_n]["phase"]["cycle_median"]
              if ARMS[p_n]["phase"] else None)
        oc = (ARMS[o_n]["phase"]["cycle_median"]
              if ARMS[o_n]["phase"] else None)
        fc = (ARMS[f_n]["phase"]["cycle_median"]
              if ARMS[f_n]["phase"] else None)
        paired = {
            "seed": PAIRED_SEED, "k": ARMS[p_n]["sched_k"],
            "n_paired": ARMS[p_n]["cell"]["n_events"],
            "paired_events": ARMS[p_n]["cell"]["events"],
            "injected_draws": ARMS[p_n]["cell"]["replay_checks"]
                ["injected_draws"],
            "fallback_draws": ARMS[p_n]["cell"]["replay_checks"]
                ["fallback_draws"],
            "paired_cycle_median": pc,
            "delta_organ_minus_paired": (oc - pc if oc is not None
                                         and pc is not None else None),
            "delta_paired_minus_fixed": (pc - fc if pc is not None
                                         and fc is not None else None),
            "note": "CO-READ, no bar: closes the replay-batch seam (the "
                    "fixed schedule replaying the organ's realized event "
                    "batches); the WASH stream remains schedule-dependent "
                    "in every arm (the named residual).",
        }

    ref = None
    if not SMOKE and f"O_{REF_SEED}" in ARMS and f"F_{REF_SEED}" in ARMS:
        oc = (ARMS[f"O_{REF_SEED}"]["phase"]["cycle_median"]
              if ARMS[f"O_{REF_SEED}"]["phase"] else None)
        fc = (ARMS[f"F_{REF_SEED}"]["phase"]["cycle_median"]
              if ARMS[f"F_{REF_SEED}"]["phase"] else None)
        ref = {
            "seed": REF_SEED,
            "note": "DEVICE CO-READ, no bar, NOT a ladder seed: the parent "
                    "+0.0758/+0.072 was GPU; this pair reads the CPU "
                    "device class on the SAME seed.",
            "organ_cycle_median_cpu": oc,
            "fixed_cycle_median_cpu": fc,
            "delta_organ_minus_fixed_cpu": (oc - fc if oc is not None
                                            and fc is not None else None),
            "parent_gpu": state["parent_reference"]["g2g_1x"],
        }

    return {
        "AUTONOMY_REPLICATES": replicates,
        "AUTONOMY_ANECDOTE": anecdote,
        "bar_threshold": BAR_DELTA,
        "n_adjudicated": n_adj, "n_delta_positive": n_pos,
        "median_delta": med,
        "delta_min_max": ([min(deltas), max(deltas)] if deltas else None),
        "in_cell_organ_spread": ([min(org_cms), max(org_cms)]
                                 if org_cms else None),
        "in_cell_delta_spread": ([min(deltas), max(deltas)]
                                 if deltas else None),
        "sub_booleans": {
            "all_3_seeds_adjudicated": bool(n_adj == 3),
            "same_sign_3_of_3": bool(n_adj == 3 and n_pos == 3),
            "median_ge_0.03": bool(med is not None and med >= BAR_DELTA),
        },
        "per_seed": per_seed,
        "paired_cread": paired,
        "ref_cread_cpu": ref,
    }


# ---------------------------------------------------------------- plots
def plot_all(state: dict) -> None:
    b = state["adjudication"]["bars"]

    # ---- 1. per-seed head-to-heads + the median delta ----------------------
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.25, 1.0])
    for pi, s in enumerate(SEEDS):
        ax = fig.add_subplot(gs[0 if pi < 2 else 1, pi % 2])
        for name, col, lab in ((f"O_{s}", "#d62728", "organ (self-timed)"),
                               (f"F_{s}", "#7f7f7f",
                                f"fixed k={ARMS[f'F_{s}'].get('sched_k')}"
                                if f"F_{s}" in ARMS else "fixed")):
            if name not in ARMS or "cell" not in ARMS[name]:
                continue
            tr = {r["step"]: r[RULER_KEY] for r in ARMS[name]["cell"]["traj"]}
            xs_ = sorted(tr)
            ax.plot(xs_, [tr[x] for x in xs_], "-", lw=1.4, color=col,
                    label=lab)
            for e in ARMS[name]["cell"]["events"]:
                ax.axvline(e, color=col, lw=0.7, alpha=0.3,
                           ls="-" if name.startswith("O") else "--")
        if f"P_{s}" in ARMS and "cell" in ARMS[f"P_{s}"]:
            tr = {r["step"]: r[RULER_KEY]
                  for r in ARMS[f"P_{s}"]["cell"]["traj"]}
            xs_ = sorted(tr)
            ax.plot(xs_, [tr[x] for x in xs_], "-", lw=1.4, color="#1f77b4",
                    alpha=0.85, label=f"paired-batches k="
                    f"{ARMS[f'P_{s}']['sched_k']}")
        ax.axhline(MAINTAIN_BAR, color="k", ls="--", lw=0.7)
        ax.axhline(SHUT_BAR, color="k", ls=":", lw=0.7)
        psd = b["per_seed"].get(str(s), {})
        d = psd.get("delta_organ_minus_fixed")
        ax.set_title(f"seed {s} — n {psd.get('n_organ')} vs "
                     f"{psd.get('n_fixed')}@k{psd.get('k')}, delta "
                     + (f"{d:+.3f}" if d is not None else "NA"))
        ax.set_xlabel("wash step")
        ax.set_ylabel(f"ruler {RULER_KEY} (dense grid)")
        ax.legend(fontsize=7, loc="center right")
    axD = fig.add_subplot(gs[1, 1])
    ds = [b["per_seed"][str(s)]["delta_organ_minus_fixed"]
          for s in SEEDS if str(s) in b["per_seed"]
          and b["per_seed"][str(s)]["delta_organ_minus_fixed"] is not None]
    if ds:
        xs = np.arange(len(ds))
        axD.bar(xs, ds, width=0.55, color="#d62728",
                label="organ - fixed (per seed)")
        pd_ = (b["paired_cread"] or {}).get("delta_organ_minus_paired")
        if pd_ is not None:
            axD.bar([len(ds)], [pd_], width=0.55, color="#1f77b4",
                    label=f"organ - PAIRED ({PAIRED_SEED})")
        rd = (b["ref_cread_cpu"] or {}).get("delta_organ_minus_fixed_cpu")
        if rd is not None:
            axD.bar([len(ds) + (1 if pd_ is not None else 0)], [rd],
                    width=0.55, color="#2ca02c",
                    label=f"10902 CPU co-read")
        md = b["median_delta"]
        if md is not None:
            axD.axhline(md, color="#d62728", ls="-", lw=1.6,
                        label=f"median delta {md:+.3f}")
        axD.axhline(BAR_DELTA, color="k", ls="--", lw=1.2,
                    label=f"BAR +{BAR_DELTA}")
        axD.axhline(0.0758, color="#7f7f7f", ls=":", lw=1.2,
                    label="parent +0.0758 (GPU, seed 10902)")
        axD.axhline(0, color="k", lw=0.6)
        axD.set_xticks(range(len(ds) + (1 if pd_ is not None else 0)
                             + (1 if rd is not None else 0)))
        axD.set_xticklabels([str(s) for s in SEEDS[:len(ds)]]
                            + (["paired"] if pd_ is not None else [])
                            + (["10902cpu"] if rd is not None else []),
                            fontsize=8)
    axD.set_ylabel("delta of cycle-median (organ - comparator)")
    axD.set_title("THE BAR — same sign 3/3 AND median >= +0.03 "
                  f"(fired: {b['AUTONOMY_REPLICATES']})")
    if ds:
        axD.legend(fontsize=7, loc="best")
    else:
        axD.text(0.5, 0.5, "no adjudicable deltas", ha="center",
                 va="center", transform=axD.transAxes)
    fig.suptitle("G2G2 THE SEED LADDER — organ vs count-matched fixed at "
                 f"1x, CPU-deterministic, fresh wash seeds {list(SEEDS)}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(RD / "seed_ladder_head_to_head.png", dpi=130)
    plt.close(fig)
    log(f"[plot] {RD / 'seed_ladder_head_to_head.png'}")

    # ---- 2. the paired-batch control ---------------------------------------
    fig = plt.figure(figsize=(14, 5.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1.0])
    axA = fig.add_subplot(gs[0, 0])
    for name, col, lab in ((f"O_{PAIRED_SEED}", "#d62728", "organ"),
                           (f"F_{PAIRED_SEED}", "#7f7f7f", "fixed (fresh "
                            "draws)"),
                           (f"P_{PAIRED_SEED}", "#1f77b4", "fixed (the "
                            "organ's OWN event batches)")):
        if name not in ARMS or "cell" not in ARMS[name]:
            continue
        tr = {r["step"]: r[RULER_KEY] for r in ARMS[name]["cell"]["traj"]}
        xs_ = sorted(tr)
        axA.plot(xs_, [tr[x] for x in xs_], "-", lw=1.4, color=col,
                 label=lab)
        for e in ARMS[name]["cell"]["events"]:
            axA.axvline(e, color=col, lw=0.7, alpha=0.3)
    axA.axhline(MAINTAIN_BAR, color="k", ls="--", lw=0.7)
    axA.set_xlabel("wash step")
    axA.set_ylabel(f"ruler {RULER_KEY}")
    axA.set_title(f"THE COUNT-VS-BATCH CO-READ — seed {PAIRED_SEED} "
                  "(same firing steps; only the replay batches differ)")
    axA.legend(fontsize=8, loc="center right")
    axB = fig.add_subplot(gs[0, 1])
    labels, vals, cols = [], [], []
    for name, col, lab in ((f"O_{PAIRED_SEED}", "#d62728", "organ"),
                           (f"F_{PAIRED_SEED}", "#7f7f7f", "fixed"),
                           (f"P_{PAIRED_SEED}", "#1f77b4", "paired")):
        if name in ARMS and ARMS[name].get("phase") \
                and ARMS[name]["phase"]["cycle_median"] is not None:
            labels.append(lab)
            vals.append(ARMS[name]["phase"]["cycle_median"])
            cols.append(col)
    if vals:
        axB.bar(np.arange(len(vals)), vals, width=0.55, color=cols)
        for i, v in enumerate(vals):
            axB.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    else:
        axB.text(0.5, 0.5, "no complete cycles", ha="center", va="center",
                 transform=axB.transAxes)
    axB.axhline(MAINTAIN_BAR, color="k", ls="--", lw=0.7)
    axB.set_xticks(range(len(labels)))
    axB.set_xticklabels(labels, fontsize=9)
    axB.set_ylabel(f"cycle-median {RULER_KEY}")
    pc = (state["adjudication"]["bars"]["paired_cread"] or {})
    axB.set_title("cycle-medians — paired vs fixed = the batch-draw "
                  "effect on the comparator "
                  + (f"({pc.get('delta_paired_minus_fixed'):+.3f})"
                     if pc.get("delta_paired_minus_fixed") is not None
                     else ""))
    fig.tight_layout()
    fig.savefig(RD / "paired_batch_control.png", dpi=130)
    plt.close(fig)
    log(f"[plot] {RD / 'paired_batch_control.png'}")


# ---------------------------------------------------------------- det probe
def det_probe(root_sd: dict, P: dict, n_steps: int) -> dict:
    """G_DET: two identical wash-only full-length trainings on the root
    body (run_cell's wash-branch arithmetic VERBATIM: aj(16)+rj(16) per
    step, AdamW(0.9,0.95) wd 0.1 lr G2.FT_LR, clip 1.0, generator seeded
    10909). Bit-compare the final state_dicts. Doubles as the train-only
    timing receipt (no evals interleaved)."""

    def one():
        net = G2.evl_load(root_sd)
        net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=G2.FT_LR,
                                betas=(0.9, 0.95), weight_decay=0.1)
        gen = torch.Generator().manual_seed(PAIRED_SEED)
        t0 = time.time()
        last = None
        for _ in range(n_steps):
            aj = torch.randint(16, (G2.WASH_ANCH_BS,), generator=gen)
            rj = torch.randint(len(P["train_ids"]) - G2.BLOCK - 1,
                               (G2.WASH_RAND_BS,), generator=gen)
            loss = G2.wash_loss(net, P["anchor_neutral"], aj,
                                P["train_ids"], rj, CPU)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            last = float(loss.detach())
        wall = time.time() - t0
        sd = {k: v.detach().clone() for k, v in net.state_dict().items()}
        del net
        return sd, wall, last

    sd1, w1, l1 = one()
    sd2, w2, l2 = one()
    bit = all(torch.equal(sd1[k], sd2[k]) for k in sd1)
    return {"pass": bool(bit), "bit_identical": bool(bit),
            "form": "two identical full-length wash-only trainings (same "
                    "generator seed, AdamW, clip 1.0), final state_dicts "
                    "torch.equal-compared; wall times are TRAIN-ONLY "
                    "(no evals) — the envelope's receipt",
            "steps_each": n_steps, "seed": PAIRED_SEED,
            "wall_s_each": [round(w1, 1), round(w2, 1)],
            "final_loss_each": [l1, l2]}


# ---------------------------------------------------------------- main
def main():
    global G2_P
    G2.TRAIN_CAP_S = G2G2_CAP_S          # read by run_cell at call time
    log(f"G2G2 THE SEED LADDER (the R60-critic's exact replication for the "
        f"+0.07; smoke={SMOKE}) -> {RD}")
    cuda = torch.cuda.is_available()
    log(f"compute: CPU-DETERMINISTIC lane (CUDA_VISIBLE_DEVICES=-1 before "
        f"any import), threads {torch.get_num_threads()}, fp32; "
        f"cuda.is_available()={cuda} (must be False)")
    assert not cuda, "CPU-DETERMINISM ASSERTION FAILED — CUDA visible"
    assert (G2.FT_LR, G2.REFRACTORY, G2.THETA_OPEN, G2.CADENCE,
            G2.SCHED_K) == (1e-3, 24, 0.5, 4, 32), \
        "organ-verbatim guard failed — G2 dials drifted"

    g2m = json.loads(G2_METRICS.read_text(encoding="utf-8"))
    g2cm = json.loads(G2C_METRICS.read_text(encoding="utf-8"))
    g2b_org = json.loads(G2B_METRICS.read_text(encoding="utf-8"))["organ"]
    g2gm = json.loads(G2G_METRICS.read_text(encoding="utf-8"))
    parent_1x = ((g2gm.get("adjudication") or {}).get("bars", {})
                 .get("head_to_head_pairs", {}).get("1.0"))
    cross = g2gm.get("cross_run") or {}
    parent_reference = {
        "form": "runs/g2g/metrics.json (the parent cell): the +0.07 at 1x, "
                "GPU lane, seed 10902, ONE root, count-matched fixed k=30",
        "g2g_1x": parent_1x,
        "g2g_cross_run_1x_medians": (cross.get("run1_run2_cycle_medians")
                                     or {}).get("L1"),
        "g2g_cross_run_fixed_1x_medians": (cross.get("run1_run2_cycle_medians")
                                           or {}).get("F1"),
        "g2d_borrowed_spread": {"organ_cycle_medians": [0.5870, 0.6016,
                                                        0.6149],
                                "spread": 0.0279,
                                "note": "g2d's cross-seed spread (~0.03) — "
                                        "the bar's borrowed threshold, now "
                                        "measured in-cell"},
    }

    # ---- resume: adopt completed arms from a prior interrupted run --------
    prior: dict = {}
    if (RD / "metrics.json").exists():
        try:
            prior = json.loads((RD / "metrics.json").read_text(
                encoding="utf-8"))
        except Exception as exc:
            log(f"[resume] prior metrics unreadable ({exc!r}) — fresh start")
    for name, rec in (prior.get("arms") or {}).items():
        if isinstance(rec, dict) and isinstance(rec.get("cell"), dict) \
                and rec.get("gates"):
            ARMS[name] = rec
            ARM_ORDER.append(name)
    if ARM_ORDER:
        log(f"[resume] adopted {len(ARMS)} completed arms: {ARM_ORDER}")
        WRITE_LOG[:] = list(prior.get("progressive_writes") or [])

    # ---- protocol + root (REUSE — g2b/g2c/g2d/g2g's own builders + gates)
    P = rebuild_protocol()
    G2_P = P
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

    net_root = G2GNet(G2.evl_load(root_sd), cue_pool, cue_mask)
    root_mon = net_root.monitor(0, dev=CPU)
    re_cells = {j: G2.battery_cell(G2.evl_load(root_sd), P["bat_ids"][j],
                                   P["zid"])["mean_pz"] for j in G2.GEOS}
    re_ce = G2.ce_fixed_cpu(G2.evl_load(root_sd), *P["r_eval_xy"])
    diffs = {"gm12": re_cells[-12] - g2m["root"]["cells"]["gm12"],
             "g0": re_cells[0] - g2m["root"]["cells"]["g0"],
             "gp12": re_cells[12] - g2m["root"]["cells"]["gp12"],
             "ce_r": re_ce - g2m["root"]["cells"]["ce_r"],
             "root_monitor_onset": root_mon - g2b_org["root_monitor_onset"]}
    G_ROOT = {"form": "reuse-verify vs g2's stored dials (CPU 4 threads)",
              "cells_reloaded": {f"g{j:+d}": re_cells[j] for j in G2.GEOS},
              "root_monitor_onset": root_mon,
              "g2_stored": {k: g2m["root"]["cells"][k]
                            for k in ("gm12", "g0", "gp12", "ce_r")},
              "diffs": diffs, "max_abs_diff": max(abs(v) for v in
                                                  diffs.values()),
              "bit_tol": G2.G_BIT_TOL, "tol": G2.G_FALLBACK_TOL}
    G_ROOT["bit"] = bool(G_ROOT["max_abs_diff"] < G2.G_BIT_TOL)
    G_ROOT["pass"] = bool(G_ROOT["max_abs_diff"] < G2.G_FALLBACK_TOL)
    log(f"G_POOL bit: PASS | G_ROOT (vs g2's stored): max|diff| "
        f"{G_ROOT['max_abs_diff']:.2e}: "
        + ("PASS" if G_ROOT["pass"] else "FAIL")
        + (" (bit)" if G_ROOT["bit"] else "")
        + f" | root onset monitor {root_mon:.4f}")
    if not SMOKE:
        assert G_ROOT["pass"], "G_ROOT FAILED — environment drifted since g2"
    del net_root

    state: dict = {
        "experiment": "g2g2", "date": common.now_iso(),
        "purpose": "THE SEED LADDER (the R60-critic's exact replication "
                   "protocol for the +0.07 autonomy number): >=3 FRESH wash "
                   "seeds x {organ, count-matched fixed schedule} at 1x "
                   "threat ONLY, CPU-DETERMINISTIC end-to-end, same frozen "
                   "ruler + pooling as g2g's registered comparison — plus "
                   "ONE paired-batch control arm (the fixed schedule "
                   "replaying the organ's own realized event batches; the "
                   "count-vs-batch parity co-read) and a 10902 CPU device "
                   "co-read of the parent number.",
        "builds_on": ["g2g (the parent cell: +0.0758/+0.072 at 1x, the "
                      "machinery, the frozen ruler + pooling)",
                      "g2c (the dense grid + CPU-lane convention)",
                      "g2d (the wash-seed replicate convention + the "
                      "borrowed ~0.03 spread)",
                      "T152 (the seed ladder licensed)",
                      "R60-critic ATTACK 3 (the protocol + the bar)",
                      "g2g's cross_run (the 2x float-fragility lesson -> "
                      "CPU-deterministic)"],
        "what_is_new": "the +0.07 has never met a second seed — the "
                       "critic's head-to-head seed ladder + the paired-"
                       "batch control close the two disclosed seams "
                       "(single wash seed; count-vs-batch)",
        "smoke": SMOKE, "threads": torch.get_num_threads(),
        "cpu_only": True, "cuda_visible_devices": "-1",
        "cfg": g2m["cfg"],
        "organ": {**g2m["organ"], "monitor_channel":
                  "onset-only (1 of 7; g2b's, reproduced verbatim)",
                  "root_monitor_onset": root_mon},
        "compute": {"lane": "CPU-DETERMINISTIC (fp32 end-to-end; the 2x "
                            "float-fragility lesson)",
                    "threads": torch.get_num_threads(),
                    "cap_s": G2G2_CAP_S, "grid_n": len(MEAS_GRID),
                    "grid": "g2c VERBATIM (1 + every-2 + 25)",
                    "sequential": True, "cooldowns": "none (no GPU)"},
        "root": {"source": "runs/checkpoints/"
                 + ("smoke_g2_root.pt" if SMOKE else "g2_root.pt")
                 + " (REUSED — not rebuilt; the SAME locked root as g2g)",
                 "gates": {"G_POOL": G_POOL, "G_ROOT": G_ROOT}},
        "seeds": {"ladder": list(SEEDS), "paired": PAIRED_SEED,
                  "ref_device_cread": (None if SMOKE else REF_SEED),
                  "provenance": "10909/10910/10911 = the NEXT FREE DRAWS "
                                "of the locked 109xx lineage (10902 the "
                                "locked wash seed; 10903/10904 g2d; "
                                "10905/10906 e152r/g3R/g4R; 10907/10908 "
                                "g1bR — no prior use)"},
        "registered_bars": REGISTERED_BARS,
        "operationalization": "this file's docstring (frozen before the "
                              "run; no bar shopping)",
        "registered_prediction": {
            "ladder": "each fresh organ fires 9-12 events (spacings >= 24), "
                      "cycle-median 0.55-0.65; deltas POSITIVE 3/3 with "
                      "median in [+0.03, +0.09] — AUTONOMY-REPLICATES "
                      "fires.",
            "paired": "delta_organ-paired within +-0.03 of "
                      "delta_organ-fixed (the +0.07 is timing, not draw "
                      "luck).",
            "ref": "seed-10902-CPU organ reproduces g2c's schedule >= 9/10 "
                   "steps; the CPU parent delta lands within +-0.03 of the "
                   "GPU +0.0758/+0.072.",
            "risks": "a seed firing fewer/later events (k mismatch -> "
                     "parity drop -> ANECDOTE by protocol); a 0-event seed "
                     "(F arm skipped, recorded); CPU-vs-GPU float class "
                     "moving the parent delta (what the ref leg measures).",
        },
        "parent_reference": parent_reference,
        "adjudication": None,
        "honesty": [
            "n=1 ROOT STILL — the ladder varies WASH seeds only "
            "(10909/10910/10911); root-seed/install-seed generality remains "
            "open (the g-series' standing single-root bound, g2d's own "
            "clause).",
            "THE 2x LEG NOT RE-TESTED BY DESIGN — the critic's protocol is "
            "1x-only (the run-stable leg); g2g's 2x comparison of two "
            "failing oscillators stays retired either way.",
            "CPU fp32 END-TO-DETERMINISTIC (the float-fragility class "
            "retired by construction): env -1 before any import, threads 4 "
            "fixed, per-arm devices recorded cpu, G_DET bit-equality on "
            "two identical full-length trainings.",
            "COUNT-MATCHED, NOT BATCH-MATCHED (the parent's own disclosure) "
            "— the paired arm closes the REPLAY-batch seam for seed "
            f"{PAIRED_SEED}; the WASH stream remains schedule-dependent in "
            "every arm (no arm pairs wash draws — the named residual).",
            "THE +0.03 BAR IS g2d's BORROWED CROSS-SEED SPREAD (~0.03), now "
            "measured in-cell (organ spread + delta spread co-reported in "
            "the adjudication).",
            "THE PARENT +0.0758/+0.072 WAS GPU — the 10902 CPU co-read "
            "measures the device class's move on the same seed; a CO-READ, "
            "never a gate.",
            "The paired arm's injected draws are the organ's REALIZED event "
            "batches (bit-identical index tuples, G_DRAWS-gate-parsed); a "
            "shortfall firing draws fresh ONCE (counted + recorded).",
        ],
        "gates": None, "deviations": deviations, "timing_s": None,
    }
    write_metrics(state, "scaffold (gates G_POOL/G_ROOT"
                         + ("" if SMOKE else " asserted") + ")")

    # ---- G_DET determinism probe (skipped if a prior run recorded a pass)
    prior_gates = prior.get("gates") if isinstance(prior.get("gates"),
                                                   dict) else {}
    if prior_gates.get("G_DET") is True \
            and isinstance(prior.get("det_probe"), dict):
        state["det_probe"] = prior["det_probe"]
        log(f"[resume] G_DET adopted from prior run: "
            f"{prior['det_probe']}")
    else:
        log("G_DET probe: two identical full-length trainings, bit-compared")
        det = det_probe(root_sd, P, N_STEPS)
        det["train_only_wall_s_receipt"] = (
            f"train-only {det['wall_s_each']} s x2 at "
            f"{torch.get_num_threads()} threads (no evals) — vs the "
            f"per-arm cap {G2G2_CAP_S:.0f}s and the 180s single-run rule; "
            f"the dense light-evals are the rest of the arm wall "
            f"(g2c's own dense cell: 188.7s total)")
        state["det_probe"] = det
        write_metrics(state, "after G_DET probe")
        log(f"G_DET: {'PASS' if det['pass'] else 'FAIL'} "
            f"(bit {det['bit_identical']}, walls {det['wall_s_each']}s)")
        assert det["pass"], "G_DET FAILED — CPU determinism violated"

    # ---- the arms (priority: the 3 fresh organ legs -> their fixed legs ->
    #      the paired control -> the 10902 device co-read) ------------------
    g2c_events = list(g2cm["cell"]["events"]) if not SMOKE else []
    g2c_cm = (g2cm["adjudication"]["bars"].get("cycle_median")
              if not SMOKE else None)

    def xcheck_ref(rec: dict) -> dict:
        evs = rec["cell"]["events"]
        common = sorted(set(evs) & set(g2c_events)) if g2c_events else []
        return {"form": "soft (g2c: CPU 4 threads, same code path; this "
                        "cell: CPU 4 threads — the shared-thread xcheck is "
                        "expected to be exact); never a gate",
                "g2c_stored_events": g2c_events,
                "g2c_stored_cycle_median": g2c_cm,
                "n_shared_event_steps": len(common),
                "this_run_events": evs,
                "cycle_median_diff": (rec["phase"]["cycle_median"] - g2c_cm
                                      if rec["phase"] and g2c_cm
                                      is not None else None)}

    PLAN[:] = [f"O_{s}" for s in SEEDS] + [f"F_{s}" for s in SEEDS] \
        + [f"P_{PAIRED_SEED}"]
    if not SMOKE:
        PLAN[:] += [f"O_{REF_SEED}", f"F_{REF_SEED}"]

    for s in SEEDS:
        if f"O_{s}" not in ARMS:
            run_organ_arm(f"O_{s}", s, state, root_sd, cue_pool, cue_mask,
                          P, crosscheck=(xcheck_ref if s == REF_SEED
                                         and not SMOKE else None))
        else:
            log(f"[resume] O_{s} adopted — skip")
    # provisional adjudication as soon as the ladder pair exists
    for s in SEEDS:
        if f"F_{s}" in ARMS:
            continue
        n_org = ARMS[f"O_{s}"]["cell"]["n_events"]
        if n_org < 1:
            log(f"F_{s} SKIPPED: organ fired 0 events (no count to match) "
                f"— recorded")
            ARMS[f"F_{s}"] = {"kind": "fixed", "seed": s, "lr_mult":
                              LR_MULT, "skipped": "organ fired 0 events",
                              "gates": {"G_STEP": {"pass": False}}}
            ARM_ORDER.append(f"F_{s}")
            write_metrics(state, f"F_{s} skipped (no organ events)")
            continue
        k = max(1, N_STEPS // n_org)
        run_fixed_arm(f"F_{s}", s, k, state, root_sd, cue_pool, cue_mask, P)
        state["adjudication"] = {"provisional": True,
                                 "bars": adjudicate(state)}
        write_metrics(state, f"ladder pair {s} complete (provisional bars)")

    if f"P_{PAIRED_SEED}" not in ARMS:
        o_rec = ARMS[f"O_{PAIRED_SEED}"]
        if not o_rec.get("draw_parse_G_DRAWS", {}).get("pass"):
            log(f"P_{PAIRED_SEED} SKIPPED: G_DRAWS parse failed — recorded")
            ARMS[f"P_{PAIRED_SEED}"] = {"kind": "paired",
                                        "seed": PAIRED_SEED,
                                        "skipped": "G_DRAWS parse failed"}
            ARM_ORDER.append(f"P_{PAIRED_SEED}")
        else:
            k = ARMS[f"F_{PAIRED_SEED}"]["sched_k"]
            run_paired_arm(f"P_{PAIRED_SEED}", PAIRED_SEED, k, state,
                           root_sd, cue_pool, cue_mask, P,
                           o_rec["event_draws"])
    else:
        log(f"[resume] P_{PAIRED_SEED} adopted — skip")

    if not SMOKE:
        if f"O_{REF_SEED}" in ARMS:
            log(f"[resume] O_{REF_SEED} adopted — skip")
        else:
            run_organ_arm(f"O_{REF_SEED}", REF_SEED, state, root_sd,
                          cue_pool, cue_mask, P, crosscheck=xcheck_ref)
        if f"F_{REF_SEED}" in ARMS:
            log(f"[resume] F_{REF_SEED} adopted — skip")
        else:
            n_org = ARMS[f"O_{REF_SEED}"]["cell"]["n_events"]
            if n_org < 1:
                ARMS[f"F_{REF_SEED}"] = {"kind": "fixed", "seed": REF_SEED,
                                         "skipped": "organ fired 0 events",
                                         "gates": {"G_STEP": {"pass": False}}}
                ARM_ORDER.append(f"F_{REF_SEED}")
                write_metrics(state, f"F_{REF_SEED} skipped (0 events)")
            else:
                k = max(1, N_STEPS // n_org)
                run_fixed_arm(f"F_{REF_SEED}", REF_SEED, k, state, root_sd,
                              cue_pool, cue_mask, P)

    # ---- final adjudication + plots ---------------------------------------
    state["adjudication"] = {"provisional": False, "bars": adjudicate(state)}
    b = state["adjudication"]["bars"]
    if SMOKE:
        verdict = "SMOKE (nothing adjudicated)"
        clause = "smoke trim — machinery shakedown only."
    else:
        verdict = ("AUTONOMY-REPLICATES" if b["AUTONOMY_REPLICATES"]
                   else "AUTONOMY-ANECDOTE")
        dsum = {str(s): (None if b["per_seed"][str(s)]
                         ["delta_organ_minus_fixed"] is None else
                         round(b["per_seed"][str(s)]
                               ["delta_organ_minus_fixed"], 4))
                for s in SEEDS if str(s) in b["per_seed"]}
        clause = (f"deltas {dsum} "
                  f"({b['n_delta_positive']}/{b['n_adjudicated']} positive, "
                  f"median {b['median_delta']:+.4f} vs the +0.03 bar); "
                  f"in-cell organ spread "
                  f"{b['in_cell_organ_spread']}")
        pc = b["paired_cread"]
        if pc:
            clause += (f"; paired co-read organ-paired "
                       f"{pc['delta_organ_minus_paired']:+.3f}, "
                       f"paired-fixed "
                       f"{pc['delta_paired_minus_fixed']:+.3f}")
        rc = b["ref_cread_cpu"]
        if rc and rc["delta_organ_minus_fixed_cpu"] is not None:
            clause += (f"; 10902 CPU device co-read delta "
                       f"{rc['delta_organ_minus_fixed_cpu']:+.3f} (parent "
                       f"GPU +0.0758/+0.072)")
    state["adjudication"]["verdict"] = verdict
    state["adjudication"]["clause"] = clause
    hard = {"G_POOL": G_POOL["pass"], "G_ROOT": G_ROOT["pass"],
            "G_DET": state["det_probe"]["pass"]}
    for n, rec in ARMS.items():
        if isinstance(rec.get("gates"), dict):
            for g, gv in rec["gates"].items():
                if isinstance(gv, dict):
                    hard[f"{g}@{n}"] = bool(gv["pass"])
    state["gates"] = hard
    log("=" * 78)
    log(f"G2G2 VERDICT: {verdict}")
    log(f"  {clause}")
    log(f"  bars: AUTONOMY_REPLICATES={b['AUTONOMY_REPLICATES']} "
        f"AUTONOMY_ANECDOTE={b['AUTONOMY_ANECDOTE']} "
        f"sub={b['sub_booleans']}")
    log(f"  gates: {'ALL PASS' if all(hard.values()) else [k for k, v in hard.items() if not v]}")
    log("=" * 78)

    try:
        plot_all(state)
    except Exception as exc:                      # plots never kill metrics
        log(f"[plot] FAILED ({exc!r}) — metrics stand")
        state["deviations"] = deviations + [f"PLOT FAILURE: {exc!r}"]
    state["timing_s"] = round(time.time() - T0, 1)
    save_json(RD / "metrics.json", G2.E43.jsonable(state))
    log(f"[done] metrics -> {RD / 'metrics.json'} "
        f"({state['timing_s']:.0f}s total); verdict: {verdict}")


if __name__ == "__main__":
    main()
