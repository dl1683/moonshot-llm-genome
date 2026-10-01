"""G2G — THE RHYTHM'S CONTROLS (R56-critic-forced; bars frozen in
scratch/g2g_design.md, VERBATIM below; QUEUE row g2g).

Builds on: R56's attack 2a/2b (construction disclosures: constant wash
intensity; the fixed-arm co-read 0.693 vs 0.587-0.615), T131 + its R56
amendment (the rhythm noun and its debts), T136 (timing 3/3 roots;
amplitude = ruler-geo alignment), W023 (the replay-as-mini-shock mechanism
candidate, transferred to monitor slopes per its own amendment), W026 (the
rhythm as managed bleed — zero reads around any replay event owed), g2b's
onset-only monitor (the organ's current form), g2c's dense grid and cycle
median, g2d's phase conventions (copied VERBATIM below). WHAT IS NEW: every
prior g2 run held wash intensity constant — the "self-timed" claim has never
met a threat ladder, a lowered refractory, or a registered fixed-period
head-to-head (the design's own sentence).

THE CELL (locked root, organ verbatim, wash seed 10902 family):
  (a) THREAT LADDER: wash-lr in {0.5x, 1x, 2x, 4x} x the organ verbatim;
      dense readout per rung (the g2c convention: step 1 + every 2 steps +
      step 25); event intervals + monitor decay slopes. CONFOUND NAMED
      UP FRONT (the design's own wording): the threat dial IS the wash lr —
      threat and clock speed are one variable at this instrument; the
      discriminator is the ORGAN's rate response (does the interval track
      the wash's decay speed?), not the fact's.
  (b) REFRACTORY CONTROL: refractory {8, 24} at 1x (band-widening:
      spacings <20 become possible at 8). The refractory-24 leg IS the
      ladder's 1x arm — one execution, identical config (g2d's no-duplicate
      convention); the organ's 1x realization is re-run (not embedded from
      g2c) because the ladder must be measured by one code path on one
      device, and the rider needs this run's dense state_dicts (g2c stored
      traces only).
  (c) FIXED-PERIOD HEAD-TO-HEAD: the fixed replay schedule (e179's
      finetune_rate family, run_cell mode 'sched') at matched event count
      vs the organ at the same threat level; both cycle-medians on the
      same frozen ruler (g0). Levels: 1x (the design's registered cell)
      + 2x + 4x (the SELF-TIMED-WINS bar demands >= 2 threat levels;
      0.5x fixed is skipped unless its organ rung fires events — kept out
      of the default envelope).
  RIDER (W023/W026, cheap): the monitor trace around replay events —
      post-event decay-slope change (the mini-shock signature) at every
      event, both refractory settings, via post-hoc no-RNG monitor reads on
      the dense state_dicts with a FIXED window set per event.

REGISTERED BARS (scratch/g2g_design.md, VERBATIM; frozen — no shopping):
  SELF-TIMED-THERMOSTAT: "fires if event rate is monotone in wash-lr with
      >=2 distinct rates across the ladder — the organ responds to threat;
      the rhythm is not fixed-period."
  FIXED-PERIOD-ARTIFACT: "fires if event rate is constant across the
      ladder — the rhythm is a threshold+cooldown oscillator at one threat
      level; 'self-timed' retires to 'event-driven at constant threat'."
  SELF-TIMED-WINS: "fires if the organ's cycle-median exceeds the
      matched-count fixed schedule's by >= 0.05 at >=2 threat levels."
  FIXED-MATCHES-OR-WINS: "fires if the fixed schedule is within noise or
      better — the organ's claim narrows to autonomy (zero scheduling
      signal, zero parameters), performance equal; reported honestly."
  REFRACTORY-REAL: "fires if at refractory 8 >=1 event spacing lands
      < 20 — the band claim was refractory-bound (construction artifact
      disclosed); if all spacings still >= 20, the band reflects the
      wash's own decay clock."

OPERATIONALIZATION (frozen here, BEFORE the run; every sub-boolean reported):
  - event rate of a rung = n_events / steps_ran (steps_ran = 300).
  - SELF-TIMED-THERMOSTAT = rates at lr {0.5,1,2,4}x sorted by lr are
    monotone (non-decreasing OR non-increasing, >= is allowed) AND
    >= 2 distinct values (distinct at 1e-9); direction co-reported.
  - FIXED-PERIOD-ARTIFACT = all four rates identical (1 distinct value).
    (The two ladder bars are exhaustive and mutually exclusive.)
  - matched k for a level = floor(300 / n_organ_events); n_fixed =
    floor(300 / k); the level is adjudicated only if n_organ_events >= 1
    AND |n_fixed - n_organ_events| <= 1 (else dropped + recorded).
  - cycle-median = g2d's pooling VERBATIM: median of the ruler (g0) over
    all dense in-cycle samples of the COMPLETE inter-event intervals,
    same grid, same convention; pre-rhythm death and partial tail
    co-reported, never adjudicated.
  - SELF-TIMED-WINS = #levels >= 2 AND #{levels with organ_cycle_median -
    fixed_cycle_median >= 0.05} >= 2.
  - FIXED-MATCHES-OR-WINS = SELF-TIMED-WINS does not fire (the pair is
    exhaustive; per-level deltas + g2d's cross-seed spread 0.587-0.615
    [~0.03] co-reported as the empirical noise reference).
  - REFRACTORY-REAL = at the refractory-8 arm (lr 1x): min event spacing
    < 20. Spacings at refractory 8 are possible in {8, 12, 16, 20, ...}.

REGISTERED PREDICTION (before running): the wash lr scales the monitor's
decay (e180's rate law, t* ~ lr^-1.1), so the interval should track the
wash's decay speed: predict event rate monotone INCREASING in lr — roughly
0.5x: 3-8 events, 1x: ~10 (g2c/g2d's own realization), 2x: 11-13, 4x:
12-13 (the refractory ceiling ~300/24 = 12.5); median spacing compresses
toward the 24 floor; the monitor's post-event decay slope steepens ~
linearly in lr. OPEN RISKS (both outcomes pre-registered): 4x replay may
overshoot (ruler dies, GATE-HYPER texture) — the rate bar does not depend
on the ruler; 0.5x may fire < 5 events. REFRACTORY-REAL: predict it does
NOT fire — the post-event onset read re-enters above theta quickly (g2c's
waveform: cycle start 0.6-0.74), so even at refractory 8 the gate waits
for the wash to push the monitor back under; a partial resurrection would
give a spacing of 8-16 and fire the bar (genuinely open). HEAD-TO-HEAD:
predict FIXED-MATCHES-OR-WINS (the critic's co-read: fixed 0.693 vs organ
0.587-0.615 at 1x); the organ's only chance is at high threat (firing
earlier under stronger drift) — 2x/4x are the discriminators. RIDER
(W023): the post-event decay ACCELERATES through the cycle (|late-cycle
slope| > |early-cycle slope| — the replay's optimizer re-mismatch
protecting the read initially, then wearing off); the monitor jumps AT
the event (re-teach), no immediate post-replay dip. DISCRIMINATING
OBSERVATION for the ladder: monotone rates that ALL sit at the refractory
ceiling (every spacing = 24) would be saturation, not a thermostat —
frac(spacings at floor) is co-reported with every rate.

DEVICES (the dispatch's envelope): GPU lane, gpu_ok() double-poll with
bounded PAUSE-AND-WAIT before each arm (the neighbor project's INT8 jobs
may appear — pause and wait, NEVER migrate; C12-5c policy); 120 s cooldown
between arms (overlapped with the arm's CPU-side rider analysis);
per-training cap 600 s (g2b's precedent — run_cell's cap counts the
interleaved CPU light-evals; train-only is far under the lab's 180 s
single-run rule at this 0.87M net on GPU); mid-run migration disabled
(arms stay device-pure; recorded as THE deviation that honors
pause-and-wait — an in-arm migration would device-confound the rung).

Outputs: runs/g2g/{metrics.json (PROGRESSIVE — rewritten after every
arm), rate_vs_threat.png, head_to_head.png, mini_shock_traces.png}.
No NOTES/THINKING/QUEUE/STATE edits; commits + push per the dispatch.

Run:  python lab/g2g_rhythm_controls.py    (G2G_SMOKE=1 shakedown)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                # noqa: E402
import torch                                      # noqa: E402
import torch.nn.functional as F                   # noqa: E402

import g2_rehearsal_organ as G2                   # noqa: E402 — the organ
import common                                     # noqa: E402
from common import cooldown, gpu_ok, gpu_status, run_dir, save_json  # noqa: E402

torch.set_num_threads(8)                          # g2's GPU-lane convention
                                                  # (g2b/g2d's 4 was CPU-only)

import matplotlib                                  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
import matplotlib.gridspec as gridspec             # noqa: E402

SMOKE = os.environ.get("G2G_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

# ---- frozen constants (inherited; restated for the record) --------------------
FREEZE_SEED = G2.FREEZE_SEED                      # 10902 — the locked lineage
LR_MULTS: tuple[float, ...] = (0.5, 2.0, 4.0) if SMOKE else (0.5, 1.0, 2.0, 4.0)
FIXED_LEVELS: tuple[float, ...] = (2.0,) if SMOKE else (1.0, 2.0, 4.0)
RULER_GEO = 0                                      # g2_root.pt meta (frozen)
RULER_KEY = {-12: "gm12", 0: "g0", 12: "gp12"}[RULER_GEO]
SHUT_BAR, MAINTAIN_BAR = G2.SHUT_BAR, G2.MAINTAIN_BAR   # 0.27 / 0.50
SPACING_BAND = G2.SPACING_BAND                     # (20, 45)
N_STEPS = 36 if SMOKE else 300
PHASE_EVERY = 2                                    # g2c's dense stride
MEAS_GRID: tuple[int, ...] = tuple(sorted(         # g2c's grid VERBATIM
    set((1, 25) + tuple(range(PHASE_EVERY, N_STEPS + 1, PHASE_EVERY)))))
if SMOKE:
    MEAS_GRID = tuple(sorted(set((1, 2, 4, 36))))
G2G_CAP_S = 600.0                                  # see docstring (g2b's bound)
COOLDOWN_S = 0.0 if SMOKE else 120.0               # the dispatch's between-arm
                                                  # cooldown (overlapped with
                                                  # the CPU rider analysis)
WAIT_MAX_S = 1200.0                                # pause-and-wait bound/arm
G2_METRICS = G2.E43.REPO / "runs" / "g2" / "metrics.json"
G2C_METRICS = G2.E43.REPO / "runs" / ("g2c_smoke" if SMOKE else "g2c") \
    / "metrics.json"
G2B_METRICS = G2.E43.REPO / "runs" / ("g2b_smoke" if SMOKE else "g2b") \
    / "metrics.json"

REGISTERED_BARS = {
    "self_timed_thermostat": "SELF-TIMED-THERMOSTAT: fires if event rate is "
        "monotone in wash-lr with >=2 distinct rates across the ladder — the "
        "organ responds to threat; the rhythm is not fixed-period.",
    "fixed_period_artifact": "FIXED-PERIOD-ARTIFACT: fires if event rate is "
        "constant across the ladder — the rhythm is a threshold+cooldown "
        "oscillator at one threat level; 'self-timed' retires to 'event-"
        "driven at constant threat'.",
    "self_timed_wins": "SELF-TIMED-WINS: fires if the organ's cycle-median "
        "exceeds the matched-count fixed schedule's by >= 0.05 at >=2 threat "
        "levels.",
    "fixed_matches_or_wins": "FIXED-MATCHES-OR-WINS: fires if the fixed "
        "schedule is within noise or better — the organ's claim narrows to "
        "autonomy (zero scheduling signal, zero parameters), performance "
        "equal; reported honestly.",
    "refractory_real": "REFRACTORY-REAL: fires if at refractory 8 >=1 event "
        "spacing lands < 20 — the band claim was refractory-bound "
        "(construction artifact disclosed); if all spacings still >= 20, the "
        "band reflects the wash's own decay clock.",
    "source": "scratch/g2g_design.md ## Registered bars (frozen), VERBATIM; "
              "operationalized in this file's docstring BEFORE the run; no "
              "bar shopping.",
}

deviations: list[str] = [
    "MONITOR: g2b's onset-only monitor (the organ's current form since g2c) "
    "is REPRODUCED, not imported — importing g2b would force "
    "CUDA_VISIBLE_DEVICES=-1 and kill the GPU lane; the arithmetic is g2b's "
    "G2BNet.monitor VERBATIM (onset = mask & (mask.cumsum(1) == 1)), applied "
    "by the same module-global patch g2b used.",
    "THE THREE PATCHED DIALS: per arm exactly one of G2.FT_LR (threat), "
    "G2.REFRACTORY (band-widening), G2.SCHED_K (the fixed schedule's "
    "period) is set and restored — run_cell, the event/wash batch "
    "arithmetic, the monitor, the cadence and THETA_OPEN are untouched. "
    "NOTE: the optimizer is ONE AdamW per cell, so the threat dial scales "
    "the replay steps too — 'the organ verbatim at wash-lr m' means the "
    "whole cell at lr m (the design's named confound: threat IS clock "
    "speed at this instrument).",
    "REFRACTORY-24 LEG = THE 1x LADDER RUNG: identical config (lr 1x, seed "
    "10902, organ verbatim) — executed once, read twice (g2d's no-duplicate "
    "convention). The design's '~7 arms' counted it twice; 8 arms execute: "
    "4 ladder + 1 refractory-8 + 3 fixed.",
    "1x RERUN (not g2c-embedded): the ladder must be measured by one code "
    "path on one device, and the rider needs this run's dense state_dicts "
    "(g2c stored traces only). g2c's stored 1x realization (10 events, "
    "cycle-median 0.615, CPU 4 threads) is the soft cross-check G_XCHECK.",
    "DEVICE POLICY (C12-5c + the dispatch): pause-and-wait, never migrate. "
    "run_cell's mid-run contention branch would MIGRATE (device-confounding "
    "the rung), so MIDRUN_POLL_EVERY is disabled here and replaced by a "
    "strict pre-arm gpu_ok() double-poll with bounded wait; if the neighbor "
    "job appears mid-arm (~2 min window) the arm completes device-pure and "
    "the NEXT arm waits. Devices are recorded per arm.",
    "TRAIN CAP: G2G_CAP_S=600 s (g2b's precedent; run_cell's cap counts the "
    "interleaved CPU light-evals — train-only is far under the lab's 180 s "
    "single-run rule for this 0.87M net on GPU).",
    "FIXED ARM'S RNG: matched event COUNT, not matched batches (the design's "
    "own honesty clause) — the schedule's firing pattern diverges the RNG "
    "stream from the organ's after the first differing step. Batch "
    "COMPOSITION parity (16 cue + 8 neutral + 8 random, batch 32) is "
    "identical by construction and gate-checked; draw identity is NOT "
    "claimed (co-reported).",
    "THE RIDER IS DESCRIPTIVE: no bar lives on the mini-shock traces (W023 "
    "is a wonder card); slopes are ordinary least squares on 2-step-spaced "
    "post-hoc monitor reads with a FIXED window set per event (the check "
    "counter c = event_step // CADENCE), so the trace is pure weight "
    "dynamics, no rotation noise. RIDER REPAIR (run 2): run 1's pre-event "
    "window inherited its start parity from prev_event+1 — spacing-24 "
    "cycles started odd against an even grid and got EMPTY pre-windows "
    "(L4 all-None pre/jump); the window is now even-aligned counting down "
    "from e-2. No bar touched; arms re-run deterministically (GPU) — "
    "run 2 doubles as the free reproducibility check of run 1's bars.",
    "Cooldown 120 s BETWEEN arms, overlapped with the arm's CPU-side rider "
    "analysis (>= 120 s GPU-idle gap between trainings either way).",
    "Single root (the locked g2_root.pt), ONE wash seed (10902) per rung — "
    "n=1 per cell; the replicate ladder (seeds) is licensed only if "
    "SELF-TIMED-THERMOSTAT fires (the design's own clause).",
    "Smoke mode trims: 36-step cells, grid {1,2,4,36}, lr mults "
    "{0.5, 2, 4} (+L1 first), one fixed level, no cooldown — nothing "
    "adjudicated.",
]

# ---------------------------------------------------------------- the delta
# g2b's G2BNet.monitor VERBATIM (the onset-only fix), on g2's G2Net — copied,
# not imported (importing g2b would force the CPU lane; see deviations).


class G2GNet(G2.G2Net):
    """g2's organ with g2b's ONSET-ONLY monitor (the current organ form);
    every arithmetic step is g2b's monitor VERBATIM (same window rotation,
    same softmax read; the one changed line is flagged)."""

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


G2.G2Net = G2GNet                 # g2.run_cell builds the organ via this
                                  # module global — the whole cell machinery
                                  # is reused unchanged (g2b's own patch).

# ---------------------------------------------------------------- device policy
# The dispatch: pause-and-wait, never migrate. run_cell's mid-run branch
# migrates on contention, so it is disabled (deviation #5) and replaced by
# this strict pre-arm gate. GPU_PARKED stays False unless CUDA is absent or
# the bounded wait dies (then everything runs CPU, recorded per arm).

G2.MIDRUN_POLL_EVERY = 10 ** 9    # the migration branch is unreachable
WAIT_LOG: list[dict] = []


def pick_dev_wait(tag: str) -> torch.device:
    """gpu_ok() double-poll 5 s apart; on failure WAIT (bounded), never
    park-then-migrate. Falls back to CPU only if CUDA is absent or the
    bounded wait expires (recorded)."""
    if not torch.cuda.is_available():
        G2.GPU_PARKED, G2.PARK_REASON = True, "no CUDA"
        return CPU
    t_deadline = time.time() + WAIT_MAX_S
    while True:
        s1 = gpu_status()
        if gpu_ok():
            time.sleep(5)
            if gpu_ok():
                s2 = gpu_status()
                G2.GPU_PARKED, G2.PARK_REASON = False, None
                log(f"[gpu] '{tag}' GPU after double-poll (util "
                    f"{s2['util']:.0f}% temp {s2['temp']:.0f}C mem "
                    f"{s2['mem_used']:.0f}/{s2['mem_total']:.0f}MB)")
                return torch.device("cuda")
        if time.time() >= t_deadline:
            G2.GPU_PARKED = True
            G2.PARK_REASON = f"bounded wait expired: {s1}"
            WAIT_LOG.append({"tag": tag, "event": "wait-expired->CPU",
                             "status": s1})
            log(f"[gpu] '{tag}' bounded wait expired ({s1}) -> CPU for this "
                f"and remaining arms (recorded per arm)")
            return CPU
        WAIT_LOG.append({"tag": tag, "event": "pause-and-wait", "status": s1})
        log(f"[gpu] '{tag}' BUSY ({s1}) — pause-and-wait (never migrate); "
            f"re-poll in 60 s")
        time.sleep(60)


G2.pick_dev = pick_dev_wait       # run_cell calls this per training

# ---------------------------------------------------------------- protocol
# g2b's rebuild_protocol VERBATIM (copied, not imported — the GPU lane; the
# function itself is pure CPU construction arithmetic).


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
    """g2d's phase_analysis VERBATIM (parameterized n_steps): complete
    inter-event intervals pooled; cycle-median/mean/duty over all dense
    in-cycle samples; 8 phase bins; per-cycle stats."""
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


def fit_slope(steps: list[int], vals: list[float]):
    if len(steps) >= 2:
        return float(np.polyfit(np.array(steps, float),
                                np.array(vals, float), 1)[0])
    return None


def rider_traces(sds: dict, event_steps: list[int], refr: int,
                 cue_pool: torch.Tensor, cue_mask: torch.Tensor,
                 n_steps: int) -> dict:
    """THE RIDER (W023/W026): post-hoc, no-RNG monitor reads on the dense
    state_dicts, FIXED window set per event (c = event_step // CADENCE), so
    the trace is pure weight dynamics. Pre window: (prev_event, event),
    last <=24 steps. Post window: [event, min(event+refr, next_event-1)].
    Slopes are OLS per step. Descriptive — no bar lives here."""
    if not event_steps or not sds:
        return {"per_event": [], "n_events": len(event_steps),
                "median_pre_slope": None, "median_post_slope_full": None,
                "median_post_slope_early": None,
                "median_post_slope_late": None,
                "median_jump_at_event": None}
    net = G2GNet(G2.evl_load(sds[sorted(sds)[0]]), cue_pool, cue_mask)
    per = []
    for i, e in enumerate(event_steps):
        c = e // G2.CADENCE
        prev_e = event_steps[i - 1] if i > 0 else 0
        next_e = event_steps[i + 1] if i + 1 < len(event_steps) \
            else n_steps + 1
        # even-aligned, counting DOWN from e-2 (the grid holds evens; e is a
        # multiple of CADENCE=4): the last <=24 pre-event steps above prev_e
        pre_s = sorted(s for s in range(e - 2, e - 26, -2)
                       if s in sds and s > prev_e)
        post_s = [s for s in range(e, min(e + refr, next_e - 1, n_steps) + 1,
                                   2) if s in sds]
        steps_all = sorted(set(pre_s + post_s))
        vals = {}
        for s in steps_all:
            net.body.load_state_dict(sds[s])
            vals[s] = net.monitor(c, dev=CPU)
        pre_v = [vals[s] for s in pre_s]
        post_v = [vals[s] for s in post_s]
        mid = e + refr / 2.0
        early_s = [s for s in post_s if s <= mid]
        late_s = [s for s in post_s if s > mid]
        v_e = vals.get(e)
        v_em2 = vals.get(e - 2)
        per.append({
            "n": i + 1, "step": e, "c": c,
            "pre_steps": pre_s, "pre_vals": [round(v, 4) for v in pre_v],
            "post_steps": post_s, "post_vals": [round(v, 4) for v in post_v],
            "pre_slope": fit_slope(pre_s, pre_v),
            "post_slope_full": fit_slope(post_s, post_v),
            "post_slope_early": fit_slope(early_s, [vals[s] for s in early_s]),
            "post_slope_late": fit_slope(late_s, [vals[s] for s in late_s]),
            "val_at_event": v_e,
            "jump_at_event": (v_e - v_em2 if v_e is not None
                              and v_em2 is not None else None),
            "next_event": next_e if next_e <= n_steps else None,
        })
    med = lambda k: (float(np.median([p[k] for p in per if p[k] is not None]))
                     if any(p[k] is not None for p in per) else None)
    return {"per_event": per, "n_events": len(per),
            "median_pre_slope": med("pre_slope"),
            "median_post_slope_full": med("post_slope_full"),
            "median_post_slope_early": med("post_slope_early"),
            "median_post_slope_late": med("post_slope_late"),
            "median_jump_at_event": med("jump_at_event"),
            "frac_late_steeper_than_early": (
                float(np.mean([
                    abs(p["post_slope_late"]) > abs(p["post_slope_early"])
                    for p in per
                    if p["post_slope_late"] is not None
                    and p["post_slope_early"] is not None]))
                if any(p["post_slope_late"] is not None
                       and p["post_slope_early"] is not None for p in per)
                else None)}


# ---------------------------------------------------------------- state
ARMS: dict = {}                 # name -> record (progressive)
ARM_ORDER: list[str] = []
PLAN: list[str] = []            # the full arm plan (arms_pending = PLAN - ran)
WRITE_LOG: list[str] = []
RD = run_dir("g2g_smoke" if SMOKE else "g2g")


def arm_rate(rec: dict) -> float:
    return rec["cell"]["n_events"] / max(rec["cell"]["steps_ran"], 1)


def write_metrics(state: dict, note: str) -> None:
    state["arms"] = {k: ARMS[k] for k in ARM_ORDER}
    state["arms_pending"] = [a for a in PLAN if a not in ARM_ORDER]
    state["progressive_writes"] = WRITE_LOG + [f"{common.now_iso()} {note}"]
    save_json(RD / "metrics.json", G2.E43.jsonable(state))
    WRITE_LOG.append(f"{common.now_iso()} {note}")
    log(f"[metrics] PROGRESSIVE write ({note}) -> {RD / 'metrics.json'}")


# ---------------------------------------------------------------- one arm
def run_arm(name: str, mode: str, lr_mult: float, refr: int,
            state: dict, root_sd, cue_pool, cue_mask, P: dict,
            sched_k: int | None = None, crosscheck: dict | None = None) \
        -> None:
    ARM_ORDER.append(name)
    log("=" * 78)
    if ARM_ORDER.index(name) > 0 and COOLDOWN_S > 0:
        log(f"cooldown {COOLDOWN_S:.0f}s (the dispatch's between-arm gap; "
            f"never migrate)")
        cooldown(COOLDOWN_S)
    desc = {"organ": f"the organ VERBATIM at wash-lr {lr_mult}x "
                     f"({1e-3 * lr_mult:g}), refractory {refr}",
            "fixed": f"fixed replay schedule k={sched_k} "
                     f"(matched count), lr {lr_mult}x"}[mode]
    log(f"ARM {name}: {desc} — {N_STEPS} steps, seed {FREEZE_SEED}, dense "
        f"grid {len(MEAS_GRID)} reads")
    G2.FT_LR = 1e-3 * lr_mult
    G2.REFRACTORY = refr
    if sched_k is not None:
        G2.SCHED_K = sched_k
    t_arm = time.time()
    cell = G2.run_cell(
        "g2" if mode == "organ" else "sched",
        root_sd, cue_pool, cue_mask, P["anchor_neutral"],
        P["train_ids"], P["itos"], P["r_eval_xy"], P["bat_ids"],
        P["zid"], FREEZE_SEED, MEAS_GRID)
    train_wall = time.time() - t_arm
    G2.FT_LR, G2.REFRACTORY, G2.SCHED_K = 1e-3, 24, 32   # restore

    ev_steps = [e["step"] for e in cell["event_log"]]
    traj = {r["step"]: r for r in cell["traj"]}
    ph = phase_analysis(traj, ev_steps, N_STEPS) if ev_steps else None
    rid = (rider_traces(cell["sds"], ev_steps, refr, cue_pool, cue_mask,
                        N_STEPS) if mode == "organ" else None)
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
    rec = {
        "kind": mode, "lr_mult": lr_mult, "lr": 1e-3 * lr_mult,
        "refractory": refr, "sched_k": sched_k, "seed": FREEZE_SEED,
        "cell": {"mode": "g2" if mode == "organ" else "sched",
                 "steps_ran": cell["steps_ran"], "n_events":
                     cell["n_events"], "n_checks": cell["n_checks"],
                 "realized_r": cell["realized_r"],
                 "event_spacings": spac, "events": ev_steps,
                 "monitor_trace": cell["monitor_trace"],
                 "event_log": cell["event_log"], "traj": cell["traj"],
                 "devices": {"initial": cell["initial_device"],
                             "final": cell["final_device"]},
                 "replay_checks": cell["replay_checks"]},
        "phase": ph,
        "rider": rid,
        "event_rate": arm_rate({"cell": cell}),
        "spacing_min_median_max": ([min(spac), float(np.median(spac)),
                                    max(spac)] if spac else None),
        "frac_spacings_at_refractory_floor": (
            float(np.mean([s == refr for s in spac])) if spac else None),
        "frac_spacings_in_20_45": (
            float(np.mean([SPACING_BAND[0] <= s <= SPACING_BAND[1]
                           for s in spac])) if spac else None),
        "gates": gates,
        "wall_s": round(train_wall, 1),
    }
    if crosscheck is not None:
        rec["G_XCHECK_vs_g2c"] = crosscheck(rec)
    del cell["sds"]                       # measurement states, not artifacts
    ARMS[name] = rec
    for g in gates:
        if not gates[g]["pass"]:
            log(f"  GATE {g} FAILED on arm {name}: {gates[g]}")
    log(f"ARM {name} DONE in {train_wall:.0f}s: {cell['n_events']} events "
        f"{ev_steps}; rate {rec['event_rate']:.4f}; spacings {spac}"
        + (f"; cycle-median "
           f"{ph['cycle_median'] if ph['cycle_median'] is not None else 'NA'}"
           if ph else "")
        + (f"; monitor post-slope "
           f"{rid['median_post_slope_full'] if rid and rid['median_post_slope_full'] is not None else 'NA'}"
           if rid else ""))
    write_metrics(state, f"after arm {name}")
    return None


# ---------------------------------------------------------------- adjudicate
def adjudicate(state: dict) -> dict:
    ladder_names = [f"L{m:g}" for m in LR_MULTS]
    rates = {n: ARMS[n]["event_rate"] for n in ladder_names if n in ARMS}
    vals = [rates[n] for n in ladder_names if n in rates]
    mono_up = all(a <= b + 1e-9 for a, b in zip(vals, vals[1:]))
    mono_dn = all(a >= b - 1e-9 for a, b in zip(vals, vals[1:]))
    n_distinct = len({round(v, 9) for v in vals})
    thermostat = bool(len(vals) == len(LR_MULTS) and (mono_up or mono_dn)
                      and n_distinct >= 2)
    artifact = bool(len(vals) == len(LR_MULTS) and n_distinct == 1)

    pairs, deltas = {}, {}
    for m in FIXED_LEVELS:
        o_n, f_n = f"L{m:g}", f"F{m:g}"
        if o_n not in ARMS or f_n not in ARMS:
            continue
        n_org = ARMS[o_n]["cell"]["n_events"]
        k = ARMS[f_n]["sched_k"]
        n_fix = ARMS[f_n]["cell"]["n_events"]
        ok = n_org >= 1 and abs(n_fix - n_org) <= 1
        oc = (ARMS[o_n]["phase"]["cycle_median"]
              if ARMS[o_n]["phase"] else None)
        fc = (ARMS[f_n]["phase"]["cycle_median"]
              if ARMS[f_n]["phase"] else None)
        pairs[m] = {"n_organ": n_org, "n_fixed": n_fix, "k": k,
                    "count_parity_ok": bool(ok),
                    "organ_cycle_median": oc, "fixed_cycle_median": fc,
                    "delta_organ_minus_fixed": (oc - fc
                                                if oc is not None
                                                and fc is not None
                                                else None)}
        if ok and oc is not None and fc is not None:
            deltas[m] = oc - fc
    wins = bool(len(deltas) >= 2
                and sum(1 for d in deltas.values() if d >= 0.05) >= 2)
    matches = not wins

    r8 = ARMS.get("R8")
    r24 = ARMS.get("L1")          # the refractory-24 leg IS the 1x rung
    refr_real = bool(r8 is not None and r8["cell"]["event_spacings"]
                     and min(r8["cell"]["event_spacings"]) < 20)

    bars = {
        "SELF_TIMED_THERMOSTAT": thermostat,
        "FIXED_PERIOD_ARTIFACT": artifact,
        "SELF_TIMED_WINS": wins,
        "FIXED_MATCHES_OR_WINS": matches,
        "REFRACTORY_REAL": refr_real,
        "ladder_rates": {str(k): rates[k] for k in rates},
        "ladder_monotone_up": bool(mono_up), "ladder_monotone_dn":
            bool(mono_dn),
        "ladder_n_distinct_rates": n_distinct,
        "ladder_direction": ("increasing" if mono_up and not mono_dn else
                             "decreasing" if mono_dn and not mono_up else
                             "flat" if n_distinct == 1 else "non-monotone"),
        "head_to_head_pairs": {str(m): pairs[m] for m in pairs},
        "n_head_to_head_levels": len(deltas),
        "refractory_min_spacing_r8": (min(r8["cell"]["event_spacings"])
                                      if r8 and r8["cell"]["event_spacings"]
                                      else None),
        "refractory_min_spacing_r24": (min(r24["cell"]["event_spacings"])
                                       if r24 and r24["cell"]["event_spacings"]
                                       else None),
    }
    return bars


# ---------------------------------------------------------------- plots
def plot_all(state: dict) -> None:
    ladder_names = [f"L{m:g}" for m in LR_MULTS
                    if f"L{m:g}" in ARMS]
    lrs = [ARMS[n]["lr_mult"] for n in ladder_names]

    # ---- 1. rate vs threat -------------------------------------------------
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1.0])
    axA = fig.add_subplot(gs[0, :])
    if ladder_names:
        axA.plot(lrs, [ARMS[n]["event_rate"] for n in ladder_names], "o-",
                 lw=2, ms=8, color="#d62728", zorder=3,
                 label="organ event rate (n/300)")
        ceil = 1.0 / 24
        axA.axhline(ceil, color="k", ls=":", lw=1.0)
        axA.text(4.05, ceil, " refractory ceiling 1/24", va="bottom",
                 fontsize=8)
        for n in ladder_names:
            axA.annotate(f"{ARMS[n]['cell']['n_events']} ev",
                         (ARMS[n]["lr_mult"], ARMS[n]["event_rate"]),
                         textcoords="offset points", xytext=(6, 6),
                         fontsize=8)
        axA.set_xscale("log")
        axA.set_xticks(lrs)
        axA.set_xticklabels([f"{m:g}x" for m in lrs])
    else:
        axA.text(0.5, 0.5, "no ladder arms", ha="center", va="center")
    axA.set_xlabel("wash-lr multiplier (THREAT = lr = clock speed — named)")
    axA.set_ylabel("event rate (events/step)")
    axA.set_title("THE THERMOSTAT QUESTION — event rate vs threat "
                  f"(verdict: {'MONOTONE' if state['adjudication']['bars']['ladder_n_distinct_rates'] >= 2 and (state['adjudication']['bars']['ladder_monotone_up'] or state['adjudication']['bars']['ladder_monotone_dn']) else 'FLAT/NON-MONOTONE'})")
    axA.legend(fontsize=8, loc="center right")

    axB = fig.add_subplot(gs[1, 0])
    xs, ys = [], []
    for n in ladder_names:
        r = ARMS[n]["rider"]
        if r and r["median_post_slope_full"] is not None:
            xs.append(ARMS[n]["lr_mult"])
            ys.append(r["median_post_slope_full"])
    if xs:
        axB.plot(xs, ys, "s-", lw=1.8, ms=7, color="#1f77b4")
        if min(xs) > 0:
            axB.set_xscale("log")
        axB.set_xticks(lrs)
        axB.set_xticklabels([f"{m:g}x" for m in lrs])
    axB.set_xlabel("wash-lr multiplier")
    axB.set_ylabel("median post-event monitor decay slope (/step)")
    axB.set_title("the organ's RATE response — monitor decay vs threat "
                  "(the discriminator: does the interval track this?)")
    axB.axhline(0, color="k", lw=0.6)

    axC = fig.add_subplot(gs[1, 1])
    for i, n in enumerate(ladder_names):
        spac = ARMS[n]["cell"]["event_spacings"]
        axC.scatter([ARMS[n]["lr_mult"]] * len(spac), spac, s=26, alpha=0.75,
                    color=plt.cm.viridis(i / max(1, len(ladder_names) - 1)),
                    zorder=3)
        if spac:
            axC.hlines(float(np.median(spac)), ARMS[n]["lr_mult"] * 0.92,
                       ARMS[n]["lr_mult"] * 1.08, color="k", lw=1.6,
                       zorder=4)
    axC.axhline(20, color="#d62728", ls="--", lw=0.8)
    axC.axhline(45, color="#d62728", ls="--", lw=0.8)
    axC.axhline(24, color="k", ls=":", lw=0.8)
    axC.text(4.1, 24, " r24 floor", va="bottom", fontsize=7)
    if ladder_names and min(lrs) > 0:
        axC.set_xscale("log")
        axC.set_xticks(lrs)
        axC.set_xticklabels([f"{m:g}x" for m in lrs])
    axC.set_ylim(0, 60)
    axC.set_xlabel("wash-lr multiplier")
    axC.set_ylabel("event spacing (steps; black = median)")
    axC.set_title("spacings per rung (dashed = the registered 20-45 band)")
    fig.tight_layout()
    fig.savefig(RD / "rate_vs_threat.png", dpi=130)
    plt.close(fig)
    log(f"[plot] {RD / 'rate_vs_threat.png'}")

    # ---- 2. head-to-head ---------------------------------------------------
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1.0])
    axA = fig.add_subplot(gs[0, :])
    lvls = [m for m in FIXED_LEVELS
            if f"L{m:g}" in ARMS and f"F{m:g}" in ARMS
            and ARMS[f"L{m:g}"].get("phase")
            and ARMS[f"F{m:g}"].get("phase")
            and ARMS[f"L{m:g}"]["phase"]["cycle_median"] is not None
            and ARMS[f"F{m:g}"]["phase"]["cycle_median"] is not None]
    if lvls:
        xs = np.arange(len(lvls))
        axA.bar(xs - 0.18, [ARMS[f"L{m:g}"]["phase"]["cycle_median"]
                            for m in lvls], width=0.36, color="#d62728",
                label="organ (self-timed)")
        axA.bar(xs + 0.18, [ARMS[f"F{m:g}"]["phase"]["cycle_median"]
                            for m in lvls], width=0.36, color="#7f7f7f",
                label="fixed schedule (matched count)")
        for i, m in enumerate(lvls):
            d = (ARMS[f"L{m:g}"]["phase"]["cycle_median"]
                 - ARMS[f"F{m:g}"]["phase"]["cycle_median"])
            axA.text(i, max(ARMS[f"L{m:g}"]["phase"]["cycle_median"],
                            ARMS[f"F{m:g}"]["phase"]["cycle_median"]) + 0.02,
                     f"delta {d:+.3f}", ha="center", fontsize=9)
        axA.set_xticks(xs)
        axA.set_xticklabels([f"{m:g}x" for m in lvls])
    axA.axhline(MAINTAIN_BAR, color="k", ls="--", lw=0.8)
    axA.set_xlabel("threat level (wash-lr mult)")
    axA.set_ylabel(f"cycle-median ruler {RULER_KEY} (g2d's pooling)")
    axA.set_title("THE HEAD-TO-HEAD — organ vs matched-count fixed schedule "
                  f"(SELF-TIMED-WINS needs organ > fixed + 0.05 at >=2 "
                  f"levels; fired: {state['adjudication']['bars']['SELF_TIMED_WINS']})")
    axA.legend(fontsize=9, loc="lower left")

    axB = fig.add_subplot(gs[1, 0])
    if "L1" in ARMS:
        tr = {r["step"]: r[RULER_KEY] for r in ARMS["L1"]["cell"]["traj"]}
        xs_ = sorted(tr)
        axB.plot(xs_, [tr[s] for s in xs_], "-", lw=1.5, color="#d62728",
                 label="organ 1x")
        for e in ARMS["L1"]["cell"]["events"]:
            axB.axvline(e, color="#d62728", lw=0.8, alpha=0.35)
    if "F1" in ARMS:
        tr = {r["step"]: r[RULER_KEY] for r in ARMS["F1"]["cell"]["traj"]}
        xs_ = sorted(tr)
        axB.plot(xs_, [tr[s] for s in xs_], "-", lw=1.5, color="#7f7f7f",
                 label=f"fixed k={ARMS['F1']['sched_k']}")
        for e in ARMS["F1"]["cell"]["events"]:
            axB.axvline(e, color="#7f7f7f", lw=0.8, alpha=0.35, ls="--")
    axB.axhline(MAINTAIN_BAR, color="k", ls="--", lw=0.7)
    axB.axhline(SHUT_BAR, color="k", ls=":", lw=0.7)
    axB.set_xlabel("wash step")
    axB.set_ylabel(f"ruler {RULER_KEY}")
    axB.set_title("1x legs overlaid (vlines = replay events)")
    axB.legend(fontsize=8, loc="center right")

    axCv = fig.add_subplot(gs[1, 1])
    for name, col in (("L1", "#d62728"), ("F1", "#7f7f7f")):
        if name in ARMS and ARMS[name]["phase"]["per_cycle"]:
            cs = ARMS[name]["phase"]["per_cycle"]
            axCv.plot([c["k"] for c in cs], [c["median"] for c in cs],
                      "o-", color=col, lw=1.3, ms=5,
                      label={"L1": "organ 1x", "F1": "fixed 1x"}[name])
            axCv.plot([c["k"] for c in cs], [c["trough"] for c in cs],
                      "v--", color=col, lw=0.7, ms=4, alpha=0.6)
    axCv.axhline(MAINTAIN_BAR, color="k", ls="--", lw=0.7)
    axCv.set_xlabel("cycle")
    axCv.set_ylabel(f"{RULER_KEY} median (o) / trough (v)")
    axCv.set_title("per-cycle medians and troughs — the 1x pair")
    axCv.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RD / "head_to_head.png", dpi=130)
    plt.close(fig)
    log(f"[plot] {RD / 'head_to_head.png'}")

    # ---- 3. mini-shock traces ----------------------------------------------
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.1, 1.0])
    for pi, (name, ttl) in enumerate(
            (("L1", "refractory 24 (the 1x rung)"),
             ("R8", "refractory 8"))):
        ax = fig.add_subplot(gs[0, pi])
        rec = ARMS.get(name)
        if rec and rec["rider"]["per_event"]:
            for p in rec["rider"]["per_event"]:
                ss = [s - p["step"] for s in p["pre_steps"]] \
                    + [s - p["step"] for s in p["post_steps"]]
                vv = p["pre_vals"] + p["post_vals"]
                ax.plot(ss, vv, "-", lw=0.9, alpha=0.8,
                        color="#d62728" if p["n"] % 2 else "#1f77b4")
            ax.axvline(0, color="k", lw=1.0)
            ax.axhline(G2.THETA_OPEN, color="k", ls="--", lw=0.8)
        else:
            ax.text(0.5, 0.5, "no events", ha="center", va="center")
        ax.set_xlabel("steps since the replay event (0 = post-replay state)")
        ax.set_ylabel("onset monitor p(Z|ctx), fixed window set")
        ax.set_title(f"THE MINI-SHOCK RIDER — {ttl}")
    axC = fig.add_subplot(gs[1, :])
    labels, pres, posts, lates = [], [], [], []
    for name in ("L1", "R8"):
        rec = ARMS.get(name)
        if not rec or not rec["rider"]["per_event"]:
            continue
        for p in rec["rider"]["per_event"]:
            if p["pre_slope"] is not None and \
                    p["post_slope_full"] is not None:
                labels.append(f"{name} e{p['n']}")
                pres.append(p["pre_slope"])
                posts.append(p["post_slope_full"])
                lates.append(p["post_slope_late"] if
                             p["post_slope_late"] is not None else 0.0)
    if labels:
        xs = np.arange(len(labels))
        w = 0.8 / 3
        if any(v is not None for v in pres):
            axC.bar(xs - w, [v if v is not None else 0.0 for v in pres],
                    width=w, label="pre-event slope", color="#ff7f0e")
        if any(v is not None for v in posts):
            axC.bar(xs, [v if v is not None else 0.0 for v in posts],
                    width=w, label="post slope (full)", color="#1f77b4")
        if any(v is not None for v in lates):
            axC.bar(xs + w, [v if v is not None else 0.0 for v in lates],
                    width=w, label="post slope (late half)", color="#2ca02c")
        axC.set_xticks(xs)
        axC.set_xticklabels(labels, rotation=90, fontsize=7)
        axC.axhline(0, color="k", lw=0.6)
        axC.legend(fontsize=8)
    axC.set_ylabel("monitor slope per step (OLS)")
    axC.set_title("post-event decay-slope CHANGE per event (W023's "
                  "mini-shock: late-half steeper than early = the replay's "
                  "protective mismatch wearing off)")
    fig.tight_layout()
    fig.savefig(RD / "mini_shock_traces.png", dpi=130)
    plt.close(fig)
    log(f"[plot] {RD / 'mini_shock_traces.png'}")


# ---------------------------------------------------------------- main
def main():
    G2.TRAIN_CAP_S = G2G_CAP_S          # read by G2.run_cell at call time
    log(f"G2G THE RHYTHM'S CONTROLS (R56-critic-forced; smoke={SMOKE}) "
        f"-> {RD}")
    log(f"compute: GPU lane (pause-and-wait, never migrate), cooldown "
        f"{COOLDOWN_S:.0f}s between arms, cap {G2G_CAP_S:.0f}s/arm; gpu "
        f"now: {gpu_status()}; threads {torch.get_num_threads()}")

    g2m = json.loads(G2_METRICS.read_text(encoding="utf-8"))
    g2_root_cells = g2m["root"]["cells"]
    g2cm = json.loads(G2C_METRICS.read_text(encoding="utf-8"))
    g2b_org = json.loads(G2B_METRICS.read_text(encoding="utf-8"))["organ"]

    # ---- protocol + root (REUSE — g2b/g2c/g2d's own builders + bit-gates)
    P = rebuild_protocol()
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
    diffs = {"gm12": re_cells[-12] - g2_root_cells["gm12"],
             "g0": re_cells[0] - g2_root_cells["g0"],
             "gp12": re_cells[12] - g2_root_cells["gp12"],
             "ce_r": re_ce - g2_root_cells["ce_r"],
             "root_monitor_onset": root_mon - g2b_org["root_monitor_onset"]}
    G_ROOT = {"form": "reuse-verify vs g2's stored dials (8 threads)",
              "cells_reloaded": {f"g{j:+d}": re_cells[j] for j in G2.GEOS},
              "root_monitor_onset": root_mon,
              "g2_stored": {k: g2_root_cells[k]
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
        "experiment": "g2g", "date": common.now_iso(),
        "purpose": "THE RHYTHM'S CONTROLS (R56-critic-forced): the threat "
                   "ladder (does the organ's event rate respond to wash "
                   "threat?), the refractory control (was the 20-45 band a "
                   "construction artifact?), the registered fixed-period "
                   "head-to-head at matched event count, and W023's "
                   "mini-shock monitor-slope rider.",
        "builds_on": ["R56 attack 2a/2b (constant-threat + fixed-arm "
                      "co-read 0.693 vs 0.587-0.615)",
                      "T131 + R56 amendment (the rhythm noun + debts)",
                      "T136 (timing 3/3 roots)", "W023/W026 (mini-shock; "
                      "managed bleed — zero reads around replay events)",
                      "g2b onset-only monitor", "g2c dense grid + "
                      "cycle-median", "g2d phase conventions + honesty "
                      "clauses"],
        "what_is_new": "every prior g2 run held wash intensity constant — "
                       "the self-timed claim meets its first threat "
                       "ladder, lowered refractory, and registered "
                       "fixed-period head-to-head",
        "smoke": SMOKE, "threads": torch.get_num_threads(),
        "cfg": g2m["cfg"],
        "organ": {**g2m["organ"], "monitor_channel":
                  "onset-only (1 of 7; g2b's, reproduced verbatim)",
                  "root_monitor_onset": root_mon},
        "compute": {"lane": "GPU (pause-and-wait, never migrate; "
                            "C12-5c + dispatch)",
                    "cooldown_s": COOLDOWN_S, "cap_s": G2G_CAP_S,
                    "wait_bound_s": WAIT_MAX_S, "grid_n": len(MEAS_GRID),
                    "grid": "g2c VERBATIM (1 + every-2 + 25)"},
        "root": {"source": "runs/checkpoints/"
                 + ("smoke_g2_root.pt" if SMOKE else "g2_root.pt")
                 + " (REUSED — not rebuilt)",
                 "gates": {"G_POOL": G_POOL, "G_ROOT": G_ROOT}},
        "confound_registered": "the threat dial IS the wash lr — threat and "
                               "clock speed are one variable at this "
                               "instrument (the design's own naming); the "
                               "reading is the organ's RATE response, never "
                               "'threat caused'",
        "registered_bars": REGISTERED_BARS,
        "operationalization": "this file's docstring (frozen before the "
                              "run; no bar shopping)",
        "registered_prediction": {
            "ladder": "rate monotone INCREASING in lr (0.5x: 3-8, 1x: ~10, "
                      "2x: 11-13, 4x: 12-13 at the ceiling); median spacing "
                      "compresses toward 24; monitor post-decay slope "
                      "steepens ~linearly. RISKS: 4x replay overshoot "
                      "(GATE-HYPER texture); 0.5x may fire <5.",
            "refractory": "REFRACTORY-REAL does NOT fire (post-event onset "
                          "re-enters above theta quickly; the band is the "
                          "wash's own clock) — a partial resurrection "
                          "would give spacing 8-16 and fire it (open).",
            "head_to_head": "FIXED-MATCHES-OR-WINS (the critic's co-read "
                            "at 1x: fixed 0.693 vs organ 0.587-0.615); the "
                            "organ's only chance is high threat (2x/4x).",
            "rider_w023": "post-event decay ACCELERATES through the cycle "
                          "(|late slope| > |early slope|); the monitor "
                          "jumps AT the event; no immediate post-replay "
                          "dip.",
            "discriminating_observation": "monotone rates that ALL sit at "
                          "the refractory ceiling (every spacing = 24) are "
                          "saturation, not a thermostat — "
                          "frac(spacings at floor) co-reported per rung.",
        },
        "adjudication": None,
        "honesty": [
            "n=1 ROOT (the locked g2_root.pt), ONE wash seed (10902) per "
            "rung — first cell per the design; replicate seeds licensed "
            "only if SELF-TIMED-THERMOSTAT fires.",
            "THE NAMED CONFOUND: threat = lr = clock speed — one dial, "
            "two readings; the discriminator is the organ's rate response "
            "(interval vs monitor decay speed), never 'threat caused'.",
            "FIXED-ARM BATCH PARITY: composition identical by construction "
            "(16 cue + 8 neutral + 8 random, batch 32; gate-checked); the "
            "draws themselves differ (the firing pattern diverges the RNG "
            "stream) — matched COUNT, not matched batches (the design's "
            "own clause), co-reported.",
            "GPU float nondeterminism: g2's own convention (bars at "
            "order-of-magnitude separations; G_ROOT reports the bit flag "
            "and the 0.05 fallback). L1 vs g2c's stored 1x realization "
            "(CPU 4 threads) is the soft G_XCHECK, never a gate.",
            "THE RIDER IS DESCRIPTIVE (no bar); slopes are OLS on fixed-"
            "window-set post-hoc reads.",
        ],
        "gates": None, "wait_log": WAIT_LOG, "deviations": deviations,
        "timing_s": None,
    }
    save_json(RD / "metrics.json", G2.E43.jsonable(state))
    WRITE_LOG.append(f"{common.now_iso()} scaffold (gates G_POOL/G_ROOT)")

    # ---- the arms (priority order: ladder -> refractory -> fixed) ----------
    g2c_events = list(g2cm["cell"]["events"]) if not SMOKE else []
    g2c_cm = g2cm["adjudication"]["bars"]["cycle_median"] if not SMOKE \
        else None

    def xcheck(cell_like: dict) -> dict:
        evs = cell_like["cell"]["events"]
        common_ev = sorted(set(evs) & set(g2c_events)) if g2c_events else []
        return {"form": "soft (device differs: GPU here vs g2c's CPU 4 "
                        "threads); never a gate",
                "g2c_stored_events": g2c_events,
                "g2c_stored_cycle_median": g2c_cm,
                "n_shared_event_steps": len(common_ev),
                "this_run_events": evs}

    queue: list[tuple] = [("L1", "organ", 1.0, 24, None, None)]
    queue += [(f"L{m:g}", "organ", m, 24, None, None)
              for m in LR_MULTS if m != 1.0]
    queue += [("R8", "organ", 1.0, 8, None, None)]
    PLAN[:] = [q[0] for q in queue] + [f"F{m:g}" for m in FIXED_LEVELS]

    for i, (name, mode, m, refr, _k, _x) in enumerate(queue):
        run_arm(name, mode, m, refr, state, root_sd, cue_pool, cue_mask, P,
                crosscheck=xcheck if name == "L1" else None)
        # provisional adjudication as soon as the ladder exists
        if len([n for n in ARM_ORDER if n.startswith("L")]) == len(LR_MULTS):
            state["adjudication"] = {"provisional": True,
                                     "bars": adjudicate(state)}
            write_metrics(state, "ladder complete (provisional bars)")

    # fixed arms at matched count (need the organ rungs' event counts)
    for m in FIXED_LEVELS:
        o_n = f"L{m:g}"
        if o_n not in ARMS:
            continue
        n_org = ARMS[o_n]["cell"]["n_events"]
        if n_org < 1:
            log(f"F{m:g} SKIPPED: organ rung fired 0 events (no count to "
                f"match) — recorded")
            ARMS[f"F{m:g}"] = {"kind": "fixed", "lr_mult": m,
                               "skipped": "organ fired 0 events"}
            ARM_ORDER.append(f"F{m:g}")
            write_metrics(state, f"F{m:g} skipped (no organ events)")
            continue
        k = max(1, int(N_STEPS / n_org))
        run_arm(f"F{m:g}", "fixed", m, 24, state, root_sd, cue_pool,
                cue_mask, P, sched_k=k)

    # ---- final adjudication + plots ---------------------------------------
    state["adjudication"] = {"provisional": False,
                             "bars": adjudicate(state)}
    b = state["adjudication"]["bars"]
    if SMOKE:
        verdict = "SMOKE (nothing adjudicated)"
        clause = "smoke trim — machinery shakedown only."
    else:
        verdict = ("SELF-TIMED-THERMOSTAT" if b["SELF_TIMED_THERMOSTAT"]
                   else "FIXED-PERIOD-ARTIFACT" if b["FIXED_PERIOD_ARTIFACT"]
                   else "NEITHER-LADDER-BAR (non-monotone with >=2 rates)")
        clause = (f"ladder rates "
                  f"{ {k: round(v, 4) for k, v in b['ladder_rates'].items()} } "
                  f"({b['ladder_direction']}, {b['ladder_n_distinct_rates']} "
                  f"distinct); head-to-head at {b['n_head_to_head_levels']} "
                  f"levels: "
                  + "; ".join(f"{m}x organ-fixed "
                              f"{p['delta_organ_minus_fixed']:+.3f}"
                              f" (n {p['n_organ']} vs {p['n_fixed']}@k"
                              f"{p['k']})"
                              for m, p in
                              b["head_to_head_pairs"].items())
                  + f"; refractory-8 min spacing "
                  f"{b['refractory_min_spacing_r8']} (r24 leg min "
                  f"{b['refractory_min_spacing_r24']}).")
    state["adjudication"]["verdict"] = verdict
    state["adjudication"]["clause"] = clause
    hard = {"G_POOL": G_POOL["pass"], "G_ROOT": G_ROOT["pass"]}
    for n, rec in ARMS.items():
        if isinstance(rec.get("gates"), dict):
            for g, gv in rec["gates"].items():
                hard[f"{g}@{n}"] = bool(gv["pass"])
    state["gates"] = hard
    log("=" * 78)
    log(f"G2G VERDICT: {verdict}")
    log(f"  {clause}")
    log("  bars: " + " ".join(
        f"{k}={b[k]}" for k in ("SELF_TIMED_THERMOSTAT",
                                "FIXED_PERIOD_ARTIFACT", "SELF_TIMED_WINS",
                                "FIXED_MATCHES_OR_WINS", "REFRACTORY_REAL")))
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
