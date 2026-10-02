"""G1BS5 — THE FORMATION CURVE (T165's named cut; the coordinator's eighth-
dispatch lineage, 2026-10-02). CONSOLIDATION-ONLY — NO install, NO wall arms.

==================== THE QUESTION (T165's named cut) =========================
g1bS4's record (runs/g1bS4/metrics.json, committed 2026-10-02): matching
e113's total movement made the channel WEAKER (root g-12 0.2523 at 0.30 rms
vs 0.6498 at 0.12 rms — g1bS3) — the 10M formation landscape is
NON-MONOTONIC in consolidation movement, with the optimum (if it exists)
sharp between 0.12 and 0.30 rms. THE CELL (the cheapest possible form of
the question): from the SAME loaded base+install checkpoint (g1bS2's PASSED
artifacts — never retrained), run THREE consolidations at 4e-4 (the
licensed G1BS3 rate, unchanged) with total movements {0.15, 0.20, 0.25} rms
(steps 375/500/625 at 4e-4 — the arithmetic below), read ONLY the root gate
(g-12 vs the 0.78 express bar; the co-reads g0/CE_R/held30 verbatim from
the prior takes' root conventions). NO install. NO wall arms. NO wash.

THE CURVE: root g-12 (the registered displaced-geometry battery channel)
vs total movement at 4e-4, joined with the committed points:
  0.12 rms (s300, g1bS3) -> 0.6498      (committed; runs/g1bS3/metrics.json)
  0.30 rms (s750, g1bS4) -> 0.2523      (committed; runs/g1bS4/metrics.json)
  the 1e-3 point (0.30 rms at RATE 1e-3, g1bS2) -> 0.0010 kept SEPARATE as
  the stability casualty (its failure was stability, not dose).

=============== THE REGISTERED BARS (frozen verbatim; no shopping) ===========
  SHARP-OPTIMUM: "fires if the curve peaks between 0.12 and 0.30 (some
      interior movement reading >= 0.78 or >= g1bS3's 0.6498 + 0.05) — the
      formation optimum located; a fifth take at the peak movement would
      finally adjudicate the wall."
  MONOTONE-DECLINE: "fires if the curve falls monotonically from 0.12 —
      g1bS3's 0.12 WAS the optimum; more consolidation movement only
      dissolves the channel; the e113 form's 10M ceiling is the finding,
      honestly accepted."
  GRADED: "any partial — the curve verbatim."

OPERATIONALIZATION (registered here BEFORE compute — the frozen words made
computable; nothing re-tuned after seeing data):
  curve_abscissas := (0.12, 0.15, 0.20, 0.25, 0.30) rms at 4e-4
  curve_readings  := (0.6498 committed, m15, m20, m25, 0.2523 committed)
  interior        := the three NEW readings {m15, m20, m25}
  SHARP-OPTIMUM fires iff max(interior) >= 0.6998 (= 0.6498 + 0.05; the
      ">= 0.78" disjunct is subsumed — 0.78 > 0.6998, either fires it).
      The peak movement = argmax over the five curve readings (reported).
  MONOTONE-DECLINE fires iff every later reading <= earlier reading + 0.02
      across the ordered abscissas (0.02 = the measured SAME-RECIPE replay
      fuzz floor, g1bS4's committed overlap check: mean |dg0| 0.0218 — a
      "monotone decline" must not be broken by fuzz-scale upticks; the
      max-fuzz 0.111 is co-reported as the caveat riding the verdict).
  Priority: SHARP-OPTIMUM -> MONOTONE-DECLINE -> GRADED (the first two are
  mutually exclusive by construction: 0.6998 > 0.6498 + 0.02).
  PREDICTION: NONE registered — T165's cut is open by design (the interior
  could peak, fall, or partial); the only mechanical expectation is the
  REPLAY PROPERTY (below), checked at runtime as texture, never a gate.

THE MOVEMENT ARITHMETIC (documented BEFORE compute; three currencies):
  ADAM-CLOCK (T139 — one AdamW step moves parameters ~lr in per-coordinate
  RMS regardless of N; the LICENSED currency of the lineage): steps =
  movement / lr at the licensed rate 4e-4 (the G1BS3 width-scaled license,
  unchanged):
      0.15 rms / 4e-4 = 375 steps     0.20 rms / 4e-4 = 500 steps
      0.25 rms / 4e-4 = 625 steps
  e113 verbatim total (the 2.74M reference): 300 x 1e-3 = 0.30 rms.
  WIDTH-PROPORTIONAL (the standing cure that fixed BASE_LR, INSTALL_LR and
  CONS_LR): 4e-4 = 1e-3 x 128/320 (the house recipe was minted at n_embd
  128; this host is 320).
  RAW L2 (per step ~ lr*sqrt(P); co-reported for honesty, NOT the licensed
  currency): 4e-4*sqrt(9.98e6) = 1.264 raw/step; totals {474, 632, 790} raw
  vs e113's 300 x 1e-3*sqrt(2.74e6) = 1.655 raw total.
  THE JITTER EXPOSURE WINDOW is part of the treatment's identity: seed
  10901 verbatim, so steps 1..375 of every run REPLAY g1bS3/g1bS4's
  committed trajectories (same seed + same draw order + same net0 + same
  lr) up to device float fuzz — the overlap is cross-checked at runtime
  (texture only). The registered g-12 channel is read ONLY at each run's
  endpoint (the prior takes' root-dial convention) — the interior g-12
  readings are genuinely new, not previewable from committed data.

==================== THE OWNER ENVELOPE (STATE.json compute_directive) ======
The tightest constraint (2026-10-02, permanent): the lab is the LOWEST
compute priority. (1) GPU util AND temperature checked BEFORE every launch
(nvidia-smi); launch only when util <= 20% AND temp <= 70C (double-poll,
5 s apart; plus mem <= 60% as resident-neighbor caution); (2) SHORT bursts
<= 90 s GPU (TRAIN_CAP_GPU = 90 — the consolidations are split at the cap
with full-state resume ckpts); (3) cooldown >= 180 s between bursts
(COOLDOWN_S = 180; the CPU root dials add further natural cooling on top);
(4) NO back-to-back; (5) when in doubt WAIT (the gate parks in 30 s polls;
after GPU_WAIT_MAX it STOPS honestly — partial metrics + resume ckpts hold
the record; NEVER a silent CPU hop). Outside load mid-burst => PAUSE-AND-
WAIT (the g1bW policy), never migrate. torch threads 4 (shared machine).

==================== LINEAGE (the standing headers, abridged) ===============
G1BS4 — the movement-matched dose (T161): 750 x 4e-4 = 0.30 rms; the
channel WEAKER (0.2523) — the dose question inverted (T165).
G1BS3 — the width-scaled e113 license (T159): CONS_LR 1e-3 -> 4e-4; the
channel formed to 0.6498 (650x g1bS2's 0.0010), CE_R cured (~1.70 vs the
2.97@25 blowout); G-ROOT near-miss -> TEXTURE with arms-for-the-record.
G1BS2 — the val-min-anchored base (T148's cure as a pattern): base PASS,
install PASS, the verbatim 1e-3 consolidation killed the channel in
formation (root 0.0010; the edge-of-stability casualty, kept separate).
Builds on: g1/g1b/g1bR (the commit-and-project wall), e113 (the jitter
consolidation form), e176N (the wash — NOT run here), T159/T161/T165.
What is NEW: the formation-vs-movement CURVE at 10M — the first dose
sweep of the consolidation itself (prior takes were single points on this
axis); consolidation-only (no wall claims possible).

CHECKS (Rule 12): the loaded checkpoints' identity gates re-derived ON THE
LOAD (G-BASE cosine-complete + G-BASE-QUAL legs; G-INST step + the
post-install reuse check vs g1bS2's committed record |d gm12| <= 0.02);
the movement arithmetic registered above BEFORE compute; the committed
curve endpoints cross-checked at runtime against runs/g1bS3/metrics.json +
runs/g1bS4/metrics.json (warn-only); the replay property cross-checked
(run B vs run A prefix; run C vs run B prefix; all vs g1bS4's committed
traj — texture, never a gate). NOTHING guaranteed — the openness is the
point.

Outputs: runs/g1bS5/{metrics.json, formation_curve.png}; checkpoints
runs/checkpoints/g1bS5_*.pt (all prior takes' artifacts untouched; the
base+install are g1bS2's, LOADED read-only). No NOTES/THINKING/QUEUE/STATE
edits (the coordinator folds). Progressive metrics writes after every
phase (the outage lesson); commit + push per phase.

Run:  cd lab && python g1bS5_formation_curve.py    (G1BS5_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import os
import random
import sys
import time
from pathlib import Path

SMOKE = os.environ.get("G1BS5_SMOKE") == "1"
if SMOKE:
    os.environ["G1_SMOKE"] = "1"     # trims G1.ROWS_OLD only (machinery smoke)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
import torch.nn.functional as F                       # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT, cooldown, gpu_ok,   # noqa: E402
                    gpu_status, run_dir, save_json, set_seed)
import e043_install as E43                             # noqa: E402 (REPO,
                                                      # find_occ, SPLICE_RNG,
                                                      # jsonable)
import g1_anchored_ball as G1                          # noqa: E402 — g1's
                                                      # machinery (instruments)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

CPU = torch.device("cpu")
torch.set_num_threads(4)     # shared machine (g1bW's adopted trim; g1's
                             # import resets to 8 — reset after import)

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)
G1.log = log                                          # unify the timeline

# ======================================================================
# THE CONFIG (the same 10M host; consolidation-only; the owner envelope)
# ======================================================================
G1BS_CFG = Cfg(vocab=65, n_layer=8, n_head=8, n_embd=320, block_size=256)
G1BS_PARAMS = 9_977_600          # e005s LARGE's verified 10M-class config
HOST_SEED = 1337                 # the "1337 family" (host init + base corpus)
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
SRC_BASE_CK, SRC_INSTALL_CK = "g1bS2_base.pt", "g1bS2_install.pt"
                                 # the lineage license (takes 3-5): g1bS2's
                                 # PASSED base + movement-matched install,
                                 # LOADED VERBATIM (never retrained;
                                 # read-only sources)
BASE_STEPS = 1200 if not SMOKE else 24
INSTALL_STEPS = 1000 if not SMOKE else 8
CONS_LR = 4e-4                    # the G1BS3 licensed rate, UNCHANGED (the
                                  # width-scaled e113 license: 1e-3 x 128/320)

# ---- THE CELL: three consolidations at 4e-4, movements {0.15, 0.20, 0.25} --
DOSES_FULL = ((0.15, 375, "m015"), (0.20, 500, "m020"), (0.25, 625, "m025"))
DOSES_SMOKE = ((0.15, 6, "m015"), (0.20, 8, "m020"), (0.25, 10, "m025"))
DOSES = DOSES_FULL if not SMOKE else DOSES_SMOKE
if not SMOKE:                               # the registered arithmetic,
    for _mv, _st, _tg in DOSES:             # asserted BEFORE compute
        assert abs(_mv / CONS_LR - _st) < 1e-9, \
            f"dose arithmetic broken: {_tg}"
CONS_MOVEMENT = {                           # the dispatch-documented math
    "licensed_currency": "ADAM-CLOCK rms (T139): steps = movement / lr",
    "lr": CONS_LR,
    "doses": [{"tag": tg, "movement_rms": mv, "steps": st,
               "check": f"{mv} / {CONS_LR} = {mv / CONS_LR:.0f}"}
              for mv, st, tg in DOSES],
    "width_proportional": "4e-4 = 1e-3 x 128/320 (n_embd 128 -> 320)",
    "raw_per_step": CONS_LR * math.sqrt(G1BS_PARAMS),
    "raw_totals": [st * CONS_LR * math.sqrt(G1BS_PARAMS)
                   for _, st, _ in DOSES],
    "e113_verbatim_reference": {"steps": 300, "lr": 1e-3,
                                "movement_rms": 300 * 1e-3,
                                "raw_total": 300 * 1e-3
                                * math.sqrt(2_739_072)},
}

# ---- THE OWNER ENVELOPE (tighter than the lab's standing guards) ----------
OWNER_UTIL_C = 20.0              # launch only when util <= 20%
OWNER_TEMP_C = 70.0              # AND temp <= 70C (double-poll)
OWNER_MEM_FRAC = 0.60            # resident-neighbor caution (mem <= 60%)
COOLDOWN_S = 180.0               # >= 180 s between bursts (the dispatch)
TRAIN_CAP_GPU = 90.0             # <= 90 s GPU bursts (the dispatch)
GPU_WAIT_MAX = float(os.environ.get("G1BS5_GPU_WAIT_MAX", "7200"))
                                 # when in doubt WAIT; after 2 h of no
                                 # windows STOP honestly (resumable) —
                                 # never a silent CPU hop

# ---- the frozen curve priors (committed endpoints; cited, cross-checked) --
G1BS3_POINT = {                  # runs/g1bS3/metrics.json (committed)
    "movement_rms": 0.12, "steps": 300, "rate": CONS_LR,
    "gm12": 0.6498003602027893, "g0": 0.7257831692695618,
    "gp12": 0.7768217921255019,
    "held30_gm12": 0.3919232188959304, "held30_g0": 0.6312348842654087,
    "ce_r": 1.6988168954849243,
    "site_read_onset": 0.6186004284004006, "site_read_span": 0.9395311979450378,
    "source": "runs/g1bS3/metrics.json traces.C[0] (root row)",
}
G1BS4_POINT = {                  # runs/g1bS4/metrics.json (committed; the
                                 # recovery runner's record — print precision)
    "movement_rms": 0.30, "steps": 750, "rate": CONS_LR,
    "gm12": 0.2523, "g0": 0.6204, "gp12": 0.286,
    "held30_gm12": 0.0602, "held30_g0": 0.438,
    "ce_r": 1.6372, "site_read_onset": 0.1614, "site_read_span": 0.8786,
    "source": ("runs/g1bS4/metrics.json traces.C[0] (root row; assembled by "
               "the recovery runner at print precision — disclosed)"),
}
G1BS2_CASUALTY = {               # kept SEPARATE (stability casualty, 1e-3)
    "movement_rms": 0.30, "steps": 300, "rate": 1e-3,
    "gm12": 0.0010485216043889523, "g0": 0.658349871635437,
    "ce_r": 2.578374147415161, "held30_gm12": 0.0011116008362766586,
    "source": ("runs/g1bS2/metrics.json traces.C[0] — the 1e-3 RATE casualty "
               "(CE_R 1.5401 -> 2.97 by s25, the edge-of-stability blowout); "
               "NOT on the 4e-4 curve: its failure was stability, not dose"),
}

REGISTERED = {
    "bars_verbatim": {
        "SHARP-OPTIMUM": ("fires if the curve peaks between 0.12 and 0.30 "
                          "(some interior movement reading >= 0.78 or >= "
                          "g1bS3's 0.6498 + 0.05) — the formation optimum "
                          "located; a fifth take at the peak movement would "
                          "finally adjudicate the wall."),
        "MONOTONE-DECLINE": ("fires if the curve falls monotonically from "
                             "0.12 — g1bS3's 0.12 WAS the optimum; more "
                             "consolidation movement only dissolves the "
                             "channel; the e113 form's 10M ceiling is the "
                             "finding, honestly accepted."),
        "GRADED": "any partial — the curve verbatim.",
    },
    "operationalization": {
        "registered_before_compute": True,
        "curve_abscissas_rms": [0.12, 0.15, 0.20, 0.25, 0.30],
        "committed_readings": {"0.12": G1BS3_POINT["gm12"],
                               "0.30": G1BS4_POINT["gm12"]},
        "interior": "the three NEW readings at {0.15, 0.20, 0.25}",
        "SHARP-OPTIMUM_fires_if": "max(interior) >= 0.6998 (= 0.6498 + 0.05; "
                                  "the >= 0.78 disjunct is subsumed: "
                                  "0.78 > 0.6998)",
        "MONOTONE-DECLINE_fires_if": ("every later reading <= earlier + 0.02 "
                                      "across the ordered abscissas (0.02 = "
                                      "the measured same-recipe replay fuzz "
                                      "floor, g1bS4's committed overlap: mean "
                                      "|dg0| 0.0218; max 0.111 co-reported "
                                      "as the caveat)"),
        "priority": "SHARP-OPTIMUM -> MONOTONE-DECLINE -> GRADED (the first "
                    "two mutually exclusive by construction)",
        "prediction": "NONE registered — T165's cut is open by design; the "
                      "only mechanical expectation is the replay property "
                      "(checked at runtime, texture only)",
    },
    "movement_arithmetic": CONS_MOVEMENT,
    "root_gate": {
        "bar": G1.EXPRESS_BAR,
        "reading": ("per-dose root g-12 vs the 0.78 express bar — a READ, "
                    "not a hard stop: there are no arms to stop in this "
                    "cell; the curve bars are the adjudication"),
        "co_reads": ("g0 / gp12 / held30 / CE_R / site read / A129 — the "
                     "prior takes' root-dial conventions verbatim (measure() "
                     "full dial, CPU)"),
    },
    "sources_loaded": {
        "base": ("runs/checkpoints/g1bS2_base.pt — g1bS2's PASSED base "
                 "(val-min-anchored 1200-step cosine, lr 4e-4, seed 1337); "
                 "G-BASE-QUAL PASS on record at runs/g1bS2/metrics.json, "
                 "re-derived ON THE LOAD by this run (takes 3-5's standing "
                 "license: never retrained)"),
        "install": ("runs/checkpoints/g1bS2_install.pt — the "
                    "movement-matched s1000 @ 4e-4 install (seed 42); "
                    "post-install state REUSABLE (g1bS2's failure was IN the "
                    "consolidation, strictly downstream; re-verified by "
                    "takes 3-4 and re-checked here)"),
    },
    "owner_envelope": {
        "launch_gate": f"util <= {OWNER_UTIL_C:.0f}% AND temp <= "
                       f"{OWNER_TEMP_C:.0f}C (double-poll 5 s) AND mem <= "
                       f"{OWNER_MEM_FRAC:.0%}",
        "burst_cap_s": TRAIN_CAP_GPU, "cooldown_s": COOLDOWN_S,
        "no_cpu_hop": ("the consolidations NEVER run on CPU (a 10M CPU "
                       "consolidation is not a viable burst; if no CUDA or "
                       "no window: STOP honestly, partial metrics + resume "
                       "ckpts hold the record)"),
    },
    "checks_rule12": [
        "the loaded checkpoints' identity gates (G-BASE cosine-complete + "
        "G-BASE-QUAL legs re-derived on the load; G-INST step + the "
        "post-install reuse check |d gm12| <= 0.02 vs g1bS2's record)",
        "the movement arithmetic registered BEFORE compute (asserted at "
        "import: mv/lr == steps for every dose)",
        "the committed endpoints cross-checked at runtime against the "
        "committed metrics files (warn-only)",
        "the replay property cross-checked (B vs A prefix; C vs B prefix; "
        "all vs g1bS4's committed traj — texture, never a gate)",
    ],
}

deviations: list[str] = [
    "CONSOLIDATION-ONLY CELL (the dispatch's license): no install training "
    "(g1bS2's install is LOADED read-only), no wall arms, no wash — the "
    "cheapest possible form of T165's question; the wall's adjudication is "
    "explicitly OUT of scope (no wall claims possible from this cell).",
    "THE OWNER ENVELOPE (STATE.json compute_directive 2026-10-02, permanent) "
    "replaces the lineage's standing GPU policy for this run — TIGHTER on "
    "every axis: launch gate util <= 20% AND temp <= 70C (double-poll; the "
    "lineage's gpu_ok() was util <= 85% / temp <= 80C), burst cap 90 s (was "
    "180 s), cooldown 180 s (was 120 s), no CPU fallback for the "
    "consolidations (pause-and-wait / stop-honestly instead). Training "
    "arithmetic, streams and seeds UNTOUCHED (g1bS4_consolidate's verbatim "
    "e113 arithmetic, re-capped).",
    "THREE SEPARATE consolidations (the dispatch's letter) at the SAME seed "
    "10901 from the SAME theta_install: runs B and C REPLAY run A's prefix "
    "(and all replay g1bS4's committed s1..625 traj) up to device float "
    "fuzz — the overlap deltas are reported per pair as free "
    "reproducibility/fuzz-floor reads (g1bS4's overlap-check pattern); the "
    "alternative single-run-with-snapshots was NOT taken (three complete "
    "runs, each its own resumable artifact, is the dispatch's form).",
    "The registered g-12 channel is read ONLY at each run's endpoint (the "
    "prior takes' root-dial convention — full measure() dial, CPU ~4-5 min "
    "each); the in-run every-25-step evals stay on the g0/CE_R light "
    "channels (the consolidation driver's own convention, verbatim).",
    "torch threads 4 (shared machine; g1bW's adopted trim; g1's import "
    "resets to 8, reset after import).",
    "Smoke mode trims: base 24 steps, install 8 steps (g1bS2's smoke ckpts), "
    "doses 6/8/10 steps, lean dials, no cooldowns, gate waits capped at 30 s "
    "— nothing adjudicated.",
]

device_events: list[dict] = []
trims: list[str] = []

_progressive = {"n": 0, "phases": []}


class OwnerWindowShut(Exception):
    """No owner-envelope GPU window opened within GPU_WAIT_MAX — the run
    stops honestly (partial metrics + resume ckpts hold the record)."""


def write_partial(rd: Path, phase: str, payload: dict) -> None:
    """The outage lesson, mechanized: metrics.json exists from the first
    phase onward and is rewritten after every phase/dose. Superseded by the
    final full write (partial=false). Bookkeeping only — never allowed to
    kill compute."""
    _progressive["n"] += 1
    _progressive["phases"].append(phase)
    try:
        out = {
            "experiment": "g1bS5_formation_curve",
            "date": common.now_iso(),
            "partial": True,
            "phase": phase,
            "progressive_writes": _progressive["n"],
            "phases": list(_progressive["phases"]),
            "device_events": device_events,
        }
        out.update(E43.jsonable(payload))
        save_json(rd / "metrics.json", out)
        log(f"[partial] metrics.json updated (phase '{phase}', write "
            f"#{_progressive['n']})")
    except Exception as e:         # bookkeeping must never kill compute
        log(f"[partial] WRITE FAILED at '{phase}' ({e}) — continuing")


# ------------------------------------------------------------------ device
# THE OWNER ENVELOPE gate: util AND temp AND mem, double-poll, park-and-wait.

def owner_gpu_ok() -> bool:
    s = gpu_status()
    return bool(s["util"] <= OWNER_UTIL_C and s["temp"] <= OWNER_TEMP_C
                and (s["mem_total"] == 0
                     or s["mem_used"] <= OWNER_MEM_FRAC * s["mem_total"]))


def wait_gpu_owner(tag: str, max_wait: float | None = None) -> torch.device:
    """THE OWNER ENVELOPE gate (STATE.json compute_directive — the tightest
    constraint): launch only when util <= 20% AND temp <= 70C, double-poll
    5 s apart (+ mem <= 60% resident-neighbor caution). Parks in 30 s polls
    (when in doubt WAIT); after max_wait STOPS honestly via OwnerWindowShut
    — never migrates to CPU."""
    mw = 30.0 if SMOKE else (GPU_WAIT_MAX if max_wait is None else max_wait)
    if not torch.cuda.is_available():
        raise OwnerWindowShut(
            f"'{tag}': no CUDA device — the consolidation-only cell has no "
            f"CPU form under the owner envelope (stop honestly; resume "
            f"ckpts hold the record)")
    t0, waited = time.time(), 0.0
    while (time.time() - t0) <= mw:
        if owner_gpu_ok():
            time.sleep(5)                      # the double-poll
            if owner_gpu_ok():
                s = gpu_status()
                device_events.append(
                    {"tag": tag, "event": "LAUNCH WINDOW OPEN",
                     "waited_s": round(waited, 1), "status": s})
                log(f"[gpu] '{tag}' OWNER window open (util {s['util']:.0f}% "
                    f"temp {s['temp']:.0f}C mem {s['mem_used']:.0f}/"
                    f"{s['mem_total']:.0f}MB"
                    + (f"; waited {waited:.0f}s" if waited > 0 else "") + ")")
                return torch.device("cuda")
        else:
            s = gpu_status()
            log(f"[gpu] '{tag}' window shut (util {s['util']:.0f}% temp "
                f"{s['temp']:.0f}C mem {s['mem_used']:.0f}MB) — WAITING "
                f"(owner envelope: util<={OWNER_UTIL_C:.0f}% "
                f"temp<={OWNER_TEMP_C:.0f}C)")
        time.sleep(30)
        waited = time.time() - t0
    device_events.append({"tag": tag, "event": "OWNER-WINDOW TIMEOUT — STOP",
                          "waited_s": round(waited, 1),
                          "status": gpu_status()})
    raise OwnerWindowShut(
        f"'{tag}': no owner window in {mw:.0f}s — stopping honestly "
        f"(resume ckpts + partial metrics hold the record; re-launch "
        f"continues)")


def midrun_pause_wait(tag: str) -> None:
    """g1bW policy: outside load/heat => PAUSE, never migrate."""
    while True:
        s = gpu_status()
        if s["mem_total"] == 0 or (s["mem_used"] <= 0.85 * s["mem_total"]
                                   and s["temp"] <= 75.0):
            log(f"  [{tag}] contention cleared ({s}); resuming")
            return
        log(f"  [{tag}] PAUSED for outside load/heat ({s})")
        time.sleep(30)


def coherence_stats(text: str) -> dict:
    """The g4 coherence dial (house stat; registered BEFORE use — g1bS4's
    verbatim helper, reused for the loaded-base re-verification)."""
    n = max(1, len(text))
    words = [w for w in text.split() if w]
    runs = max((sum(1 for _ in g) for _, g in itertools.groupby(text)),
               default=0)
    return {
        "len": len(text),
        "lower_space_fraction": sum(1 for c in text
                                    if c.islower() or c == " ") / n,
        "mean_word_len": (sum(len(w) for w in words) / len(words))
                         if words else None,
        "max_char_run": runs,
        "distinct_chars": len(set(text)),
    }


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("g1bS5_smoke" if SMOKE else "g1bS5")
    log(f"G1BS5 THE FORMATION CURVE (consolidation-only dose sweep: "
        f"{[f'{mv} rms/s{st}' for mv, st, _ in DOSES]} @ lr {CONS_LR} from "
        f"the SAME loaded base+install; owner envelope: bursts <= "
        f"{TRAIN_CAP_GPU:.0f}s, cooldown {COOLDOWN_S:.0f}s, gate util<= "
        f"{OWNER_UTIL_C:.0f}% temp<={OWNER_TEMP_C:.0f}C; smoke={SMOKE}) "
        f"-> {rd}")
    write_partial(rd, "start", {
        "design": ("T165's named cut (the dispatch, 2026-10-02): the "
                   "formation-vs-movement curve at 10M — consolidation-only, "
                   "no install, no wall arms; bars frozen verbatim; no bar "
                   "shopping"),
        "question": ("where does the displaced-geometry g-12 channel's "
                     "formation strength peak as a function of total "
                     "consolidation movement at 4e-4 on the 10M host — "
                     "inside (SHARP-OPTIMUM), at the 0.12 endpoint "
                     "(MONOTONE-DECLINE), or partial (GRADED)?"),
        "registered": REGISTERED, "deviations": deviations, "smoke": SMOKE,
        "priors": {"g1bs3_0.12": G1BS3_POINT, "g1bs4_0.30": G1BS4_POINT,
                   "g1bs2_casualty_1e-3": G1BS2_CASUALTY},
        "device_events": device_events})
    set_seed(HOST_SEED)      # host init + base corpus (the 1337 family)

    # ---- patch g1's machinery to the 10M family (g1b's pattern) ---------
    G1.G1_CFG = G1BS_CFG
    G1.G1_PARAMS = G1BS_PARAMS

    # ---------------- protocol rebuild (g1bS4's main VERBATIM) -------------
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    corpus_zeph = train_text.count("ZEPH")
    G_NAMEFREE = {"corpus_zeph_count": corpus_zeph,
                  "pass": bool(corpus_zeph == 0)}
    assert G_NAMEFREE["pass"], f"corpus contains ZEPH x{corpus_zeph}"

    host_occ = []
    for host in G1.HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + G1.POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix, "pass": bool(mix == {"FLORIZEL": 19,
                                                         "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"
    log(f"protocol rebuilt: install60 {mix}, held30 (SPLICE_RNG {E43.SPLICE_RNG})")
    name_ids = corpus.encode(G1.NAME)

    # measurement pool: e152's locked j=54 windows (instrument only)
    wins = []
    for p, h in install_occ:
        pre = train_ids[p - G1.PRE - G1.RETEACH_J: p]
        post = train_ids[p + len(h): p + len(h) + G1.SITE_CONT]
        if len(pre) != G1.PRE + G1.RETEACH_J or len(post) != G1.SITE_CONT:
            raise RuntimeError(f"pool window short at p={p}")
        w = torch.cat([pre, name_ids, post])
        if len(w) != G1.BLOCK:
            raise RuntimeError(f"pool window len {len(w)} != {G1.BLOCK}")
        wins.append(w)
    pool_x = torch.stack(wins)
    G_POOL = {"shape": list(pool_x.shape),
              "name_xcols": [G1.SITE_Z_XCOL, G1.SITE_Z_XCOL + len(G1.NAME) - 1],
              "all_windows_name_in_place": bool(
                  all(torch.equal(w[G1.SITE_Z_XCOL: G1.SITE_Z_XCOL + len(G1.NAME)],
                                  name_ids) for w in pool_x)),
              "note": "MEASUREMENT instrument only (e152's locked j=54 pool); "
                      "NO window from this pool enters any training"}
    G_POOL["pass"] = G_POOL["all_windows_name_in_place"]
    assert G_POOL["pass"], f"pool gate FAILED: {G_POOL}"

    # install windows (e043 build_win verbatim) + original-host anchor bank
    def build_win(p, host):
        return torch.cat([train_ids[p - G1.PRE: p], name_ids,
                          train_ids[p + len(host): p + len(host) + G1.POST_CAP]])

    _ = torch.stack([build_win(p, h) for p, h in install_occ])  # identity
    anchor_full = torch.stack([train_ids[p - G1.PRE: p - G1.PRE + G1.BLOCK]
                               for p, _ in install_occ])

    # jittered install pool (e113's construction VERBATIM) for consolidation
    jit_x, jit_mask = {}, {}
    for j in G1.JITTERS:
        jwins = []
        for p, h in install_occ:
            pre = train_ids[p - G1.PRE - j: p]
            post = train_ids[p + len(h): p + len(h) + G1.POST_CAP - j]
            w = torch.cat([pre, name_ids, post])
            if len(w) != G1.BLOCK:
                raise RuntimeError(f"jit window len {len(w)} != {G1.BLOCK} at {j}")
            jwins.append(w)
        jit_x[j] = torch.stack(jwins)
        m = torch.zeros(len(jwins), G1.BLOCK - 1, dtype=torch.bool)
        m[:, G1.PRE - 1 + j: G1.PRE - 1 + j + len(G1.NAME)] = True
        jit_mask[j] = m
    pool_a_x = torch.cat([jit_x[j] for j in G1.JITTERS])
    pool_a_mask = torch.cat([jit_mask[j] for j in G1.JITTERS])
    cons_anchor = anchor_full[:16]      # e113: first-16-install original bank

    # ---------------- batteries (e119/e176n verbatim) --------------------
    bat_ids, held_ids = {}, {}
    for j in G1.GEOS:
        cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
        hs = [train_text[p - G1.PRE - j: p] for p, _ in held_occ]
        held_ids[j] = torch.stack([corpus.encode(c) for c in hs])
    r_eval_x, r_eval_y = G1.val_windows(val_ids, val_text, 60, G1.R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    g0_ids = bat_ids[0]

    gates_surg: dict = {}

    def measure(sd: dict, tag: str, lean: bool = False) -> dict:
        """g1b's measure() VERBATIM dial (the prior takes' root convention)
        on evl_load. CPU-only, offline (no cap)."""
        net = G1.evl_load(sd)
        sd_local = {k: v.detach().clone() for k, v in net.state_dict().items()}
        out: dict = {"tag": tag}
        out["base"] = {j: G1.battery_cell(net, bat_ids[j], zid) for j in G1.GEOS}
        out["base_held"] = {j: G1.battery_cell(net, held_ids[j], zid)
                            for j in G1.GEOS}
        out["ce_r"] = G1.ce_fixed_cpu(net, *r_eval_xy)
        log(f"[{tag}] base: " + " ".join(f"g{j:+d} {out['base'][j]['mean_pz']:.4f}"
                                         for j in G1.GEOS)
            + " | held30: " + " ".join(
                f"g{j:+d} {out['base_held'][j]['mean_pz']:.4f}" for j in G1.GEOS)
            + f" | CE_R {out['ce_r']:.4f}")
        out["site_read"] = G1.read_fact_at(net, pool_x, name_ids, zid,
                                           G1.SITE_ADDR_ROW, G1.SITE_Z_XCOL)
        log(f"[{tag}] site read @183: onset "
            f"{out['site_read']['pz_onset_mean']:.4f} span "
            f"{out['site_read']['pname_mean_over7']:.4f}")
        if not lean and not SMOKE:
            out["census_old"] = G1.row_census(net, G1.ROWS_OLD,
                                              lambda n: G1.battery_pz(
                                                  n, bat_ids[0], zid))
            co = out["census_old"]["rows"]
            out["old_band"] = {
                "base_pz": out["census_old"]["base_readout"],
                "row0_strength": co["0"]["strength"],
                "A129": co["129"]["strength"],
                "band121_129_max": max(co[str(r)]["strength"]
                                       for r in range(121, 130)
                                       if str(r) in co)}
            log(f"[{tag}] old band: row0 S "
                f"{out['old_band']['row0_strength']:+.4f} | A(129) "
                f"{out['old_band']['A129']:+.4f}")
            DELS = {"d_all": G1.D_ALL, "d183": (G1.SITE_ADDR_ROW,)}
            out["del_table"] = {}
            for dl, rows_ in DELS.items():
                sd_d, gate = G1.deleted_wpe(sd_local, rows_)
                gates_surg[f"{tag}__{dl}"] = gate
                if not gate["pass"]:
                    raise RuntimeError(f"deletion gate FAILED {tag}/{dl}: "
                                       f"{gate}")
                net.load_state_dict(sd_d)
                cell = {"g0": G1.battery_cell(net, bat_ids[0], zid)["mean_pz"]}
                if dl == "d183":
                    cell["gm12"] = G1.battery_cell(net, bat_ids[-12],
                                                   zid)["mean_pz"]
                out["del_table"][dl] = cell
            net.load_state_dict(sd_local)
            log(f"[{tag}] deletions g0: " + " | ".join(
                f"{dl} {out['del_table'][dl]['g0']:.3f}" for dl in DELS))
        else:
            w = net.wpe.weight.data
            orig = w.clone()
            mean_row = orig.mean(0)
            bp = G1.battery_pz(net, bat_ids[0], zid)
            w[129] = mean_row
            m129 = bp - G1.battery_pz(net, bat_ids[0], zid)
            w.copy_(orig)
            w[129] = 0.0
            z129 = bp - G1.battery_pz(net, bat_ids[0], zid)
            w.copy_(orig)
            assert torch.equal(w, orig), "lean A129 failed to restore wpe"
            out["A129_quick"] = float(min(m129, z129))
        del net
        return out

    def flat_cells(m: dict) -> dict:
        c = {"gm12": m["base"][-12]["mean_pz"],
             "g0": m["base"][0]["mean_pz"],
             "gp12": m["base"][12]["mean_pz"],
             "held30_gm12": m["base_held"][-12]["mean_pz"],
             "held30_g0": m["base_held"][0]["mean_pz"],
             "ce_r": m["ce_r"],
             "site_read_onset": m["site_read"]["pz_onset_mean"],
             "site_read_span": m["site_read"]["pname_mean_over7"]}
        if "old_band" in m:
            c["A129"] = m["old_band"]["A129"]
            c["row0_strength"] = m["old_band"]["row0_strength"]
            c["dall_g0"] = m["del_table"]["d_all"]["g0"]
            c["d183_g0"] = m["del_table"]["d183"]["g0"]
            c["d183_gm12"] = m["del_table"]["d183"]["gm12"]
        else:
            c["A129"] = m["A129_quick"]
        return c

    # =====================================================================
    # G-CONFIG: the 10M band + battery geometry (Rule 12; no R-ladder here)
    # =====================================================================
    torch.manual_seed(HOST_SEED)
    host_net = TinyGPT(G1BS_CFG)
    n_params = host_net.num_params()
    G_CONFIG = {
        "params": n_params,
        "band": [8.5e6, 11.5e6],
        "vocab": G1BS_CFG.vocab, "block_size": G1BS_CFG.block_size,
        "n_layer": G1BS_CFG.n_layer, "n_head": G1BS_CFG.n_head,
        "n_embd": G1BS_CFG.n_embd,
        "battery_geometry": {
            "vocab_unchanged": bool(G1BS_CFG.vocab == corpus.vocab_size == 65),
            "block_unchanged": bool(G1BS_CFG.block_size == G1.BLOCK == 256),
            "positions_read": {"site_rows": [G1.SITE_ADDR_ROW,
                                             G1.SITE_ADDR_ROW + 6],
                               "wpe_band": [121, 129],
                               "inside_block": True},
            "tokenizer": "same data/input.txt char stoi (65 chars)",
        },
        "note": "no R-ladder in this cell (no wall arms) — the config gate "
                "carries the 10M band + the battery geometry only",
    }
    G_CONFIG["pass"] = bool(
        8.5e6 <= n_params <= 11.5e6
        and G_CONFIG["battery_geometry"]["vocab_unchanged"]
        and G_CONFIG["battery_geometry"]["block_unchanged"])
    assert G_CONFIG["pass"], f"G-CONFIG FAILED: {G_CONFIG}"
    log(f"G-CONFIG: {n_params:,} params in [8.5M, 11.5M]; vocab 65, block "
        f"256, battery geometry unchanged: PASS")
    del host_net
    write_partial(rd, "G-CONFIG", {"gates_partial": {"G_CONFIG": G_CONFIG},
                                   "config": {"n_layer": 8, "n_head": 8,
                                              "n_embd": 320,
                                              "block_size": 256,
                                              "params": n_params,
                                              "smoke": SMOKE}})

    # =====================================================================
    # PHASE 0a-host — THE BASE: g1bS2's PASSED base, LOADED VERBATIM (the
    # lineage license (takes 3-5): never retrain; identity gates ON THE LOAD)
    # =====================================================================
    log("=" * 78)
    log(f"HOST BASE: LOADED VERBATIM from {SRC_BASE_CK} (g1bS2's PASSED "
        f"base — {n_params:,} params, {BASE_STEPS}-step val-min-anchored "
        f"cosine, lr 4e-4, seed {HOST_SEED}); G-BASE-QUAL re-derived ON THE "
        f"LOAD (a corrupted load cannot slip through); never retrained")
    base_ck = CKPT_DIR / ("smoke_" + SRC_BASE_CK if SMOKE else SRC_BASE_CK)
    base_sample_prompt = val_text[:64]      # the fixed g4 prompt (recorded)
    if not base_ck.exists():
        raise SystemExit(
            f"G1BS5 source base missing: {base_ck} (the licensed cell LOADS "
            f"g1bS2's PASSED base; it cannot be retrained)")
    bstate = torch.load(base_ck, map_location="cpu", weights_only=False)
    hist: list[dict] = list(bstate.get("history", []))
    if not hist:
        raise SystemExit(f"loaded base carries no history: {base_ck}")
    bstep = int(bstate.get("step", hist[-1]["step"]))
    theta_base = {k: v.detach().clone() for k, v in bstate["model"].items()}
    del bstate
    # g1bS2's own committed record (read-only provenance cross-check)
    g1bs2_rec = {}
    _rec = E43.REPO / "runs" / ("g1bS2_smoke" if SMOKE else "g1bS2") / \
        "metrics.json"
    if _rec.exists():
        try:
            _r = json.loads(_rec.read_text(encoding="utf-8"))
            g1bs2_rec = {
                "G_BASE_QUAL": {k: v for k, v in
                                _r.get("gates", {}).get("G_BASE_QUAL",
                                                        {}).items()
                                if not isinstance(v, (dict, list))},
                "inst_cells": _r.get("provenance", {}).get("install", {})
                               .get("post_install_cells", {}),
                "G_ROOT": _r.get("gates", {}).get("G_ROOT", {}),
            }
        except Exception as e:                                     # noqa: BLE001
            log(f"[base] g1bS2 record unreadable ({e}) — continuing without "
                f"the cross-check (recorded)")
    G_BASE = {"steps": bstep, "final_val_loss": hist[-1]["val_loss"],
              "loaded_from": f"runs/checkpoints/{SRC_BASE_CK}",
              "pass": bool(bstep == BASE_STEPS)}
    assert G_BASE["pass"], f"G-BASE FAILED (cosine incomplete): {G_BASE}"
    log(f"G-BASE: loaded {bstep}/{BASE_STEPS} steps, final val "
        f"{hist[-1]['val_loss']:.4f} — cosine COMPLETE (g1bS2's own PASS on "
        f"record at runs/g1bS2/metrics.json): PASS")
    CKPT_INVENTORY[f"src_{SRC_BASE_CK[:-3]}"] = {
        "path": f"runs/checkpoints/{SRC_BASE_CK}",
        "desc": ("g1bS2's PASSED 10M base (val-min-anchored 1200-step "
                 "cosine, lr 4e-4, seed 1337) — LOADED read-only by g1bS5 "
                 "(the license: never retrained)"),
        "step": bstep, "final_val_loss": hist[-1]["val_loss"]}
    base_cells = flat_cells(measure(theta_base, "g1bS5_base", lean=True))
    log(f"base cells: g-12 {base_cells['gm12']:.4f} CE_R "
        f"{base_cells['ce_r']:.4f}")

    # ---- G-BASE-QUAL (the registered legs re-derived ON THE LOADED STATE)
    eval_vals = [e["val_loss"] for e in hist]
    first_eval_val, final_val = eval_vals[0], eval_vals[-1]
    mono_ok = all(eval_vals[i + 1] <= eval_vals[i] + 0.02
                  for i in range(len(eval_vals) - 1))
    g2 = bool(mono_ok and final_val <= first_eval_val - 0.15)
    g3 = bool(final_val <= 1.70)
    common.DEVICE = "cpu"            # deterministic CPU sample (house tool)
    bnet = TinyGPT(G1BS_CFG)
    bnet.load_state_dict(theta_base)
    bnet.to(common.DEVICE)
    base_sample = common.generate(bnet, corpus, base_sample_prompt,
                                  max_new_tokens=300)
    del bnet
    base_gen = base_sample[len(base_sample_prompt):]
    bstats = coherence_stats(base_gen)
    mwl = bstats["mean_word_len"]
    g4 = bool(bstats["lower_space_fraction"] >= 0.70 and mwl is not None
              and 2.0 <= mwl <= 8.0 and bstats["max_char_run"] <= 10
              and bstats["distinct_chars"] >= 18)
    G_BASE_QUAL = {
        "cosine_complete": G_BASE["pass"],
        "eval_vals": eval_vals,
        "val_monotone_decreasing_noise0.02": mono_ok,
        "first_eval_val": first_eval_val, "final_val": final_val,
        "g2_val_decreasing": g2,
        "g3_final_val<=1.70": g3,
        "sample_prompt": base_sample_prompt,
        "sample_generated": base_gen,
        "sample_stats": bstats,
        "g4_coherence_sample": g4,
        "pass": bool(G_BASE["pass"] and g2 and g3 and g4),
        "enforced": bool(not SMOKE),
        "verification_of": ("the LOADED g1bS2 base (its own G-BASE-QUAL "
                            "PASS is on record at runs/g1bS2/metrics.json; "
                            "these legs re-derive on the carried history + "
                            "a fresh coherence sample so a corrupted load "
                            "cannot slip through)"),
        "g1bs2_committed_record": g1bs2_rec.get("G_BASE_QUAL", {}),
    }
    log(f"G-BASE-QUAL (on the loaded state): cosine {G_BASE['pass']} | val "
        f"decreasing {g2} ({first_eval_val:.4f} -> {final_val:.4f}) | "
        f"final<=1.70 {g3} | coherence {g4}: "
        f"{'PASS' if G_BASE_QUAL['pass'] else 'FAIL'}"
        + ("" if not SMOKE else " (smoke: NOT enforced)"))
    write_partial(rd, "G-BASE-QUAL", {"G_BASE": G_BASE,
                                      "base_cells": base_cells,
                                      "G_BASE_QUAL": G_BASE_QUAL})
    if not G_BASE_QUAL["pass"] and not SMOKE:
        raise SystemExit(
            "G-BASE-QUAL FAILED on the loaded base (a corrupted load or a "
            "record mismatch — the dispatch bar: NO consolidation; report "
            "and stop, no dial search)")

    # =====================================================================
    # PHASE 0a — INSTALL: g1bS2's movement-matched install, LOADED VERBATIM
    # (the standing license: the post-install state is REUSABLE)
    # =====================================================================
    log("=" * 78)
    log(f"INSTALL: LOADED VERBATIM from {SRC_INSTALL_CK} (g1bS2's "
        f"movement-matched e043-Dmix install: 16 install + 16 paired + 32 "
        f"random = 64 windows/step, dose s{INSTALL_STEPS}, house cosine, lr "
        f"4e-4 width-scaled, seed {G1.INSTALL_SEED}); post-install cells "
        f"re-measured THIS run and cross-checked against g1bS2's record")
    inst_ck = CKPT_DIR / ("smoke_" + SRC_INSTALL_CK if SMOKE
                          else SRC_INSTALL_CK)
    if not inst_ck.exists():
        raise SystemExit(
            f"G1BS5 source install missing: {inst_ck} (the licensed cell "
            f"LOADS g1bS2's install checkpoint — the REUSABLE post-install "
            f"state)")
    istate = torch.load(inst_ck, map_location="cpu", weights_only=False)
    istep = int(istate.get("step", 0))
    theta_install = {k: v.detach().clone()
                     for k, v in istate["model"].items()}
    del istate
    G_INST = {"steps": istep,
              "loaded_from": f"runs/checkpoints/{SRC_INSTALL_CK}",
              "recipe": (f"e043 Dmix s{INSTALL_STEPS} @ lr 4e-4 "
                         f"(movement-matched, RECOVERY-3), seed "
                         f"{G1.INSTALL_SEED} — g1bS2's run"),
              "pass": bool(istep == INSTALL_STEPS)}
    assert G_INST["pass"], f"INSTALL incomplete: {G_INST}"
    inst_cells = flat_cells(measure(theta_install, "post_install", lean=True))
    prev_inst = g1bs2_rec.get("inst_cells") or {}
    d_gm12 = (abs(inst_cells["gm12"] - prev_inst["gm12"])
              if prev_inst.get("gm12") is not None else None)
    reuse_ok = bool(SMOKE or (d_gm12 is not None and d_gm12 <= 0.02))
    G_INST["reuse_check"] = {
        "g1bs2_committed": prev_inst,
        "this_run_remeasured": inst_cells,
        "abs_d_gm12": d_gm12, "tol": 0.02,
        "reuse_ok": reuse_ok,
        "rationale": ("the post-install state is the licensed source of "
                      "every consolidation in this sweep; g1bS2's G-ROOT "
                      "failure was caused by its lr-1e-3 consolidation "
                      "itself, strictly downstream of this state; smoke "
                      "skips the check (machinery only)"),
    }
    assert reuse_ok, f"post-install drift vs g1bS2 record: {G_INST['reuse_check']}"
    pv = prev_inst.get("gm12")
    log(f"post-install (re-measured): g-12 {inst_cells['gm12']:.4f} g0 "
        f"{inst_cells['g0']:.4f} CE_R {inst_cells['ce_r']:.4f} "
        + (f"(g1bS2 record g-12 {pv:.4f}; |d| {d_gm12:.2e} <= 0.02): "
           f"REUSABLE" if pv is not None else "(no g1bS2 record read)"))
    write_partial(rd, "G-INST", {"G_INST": G_INST, "inst_cells": inst_cells})
    CKPT_INVENTORY[f"src_{SRC_INSTALL_CK[:-3]}"] = {
        "path": f"runs/checkpoints/{SRC_INSTALL_CK}",
        "desc": ("g1bS2's movement-matched install (e043 Dmix s1000 @ 4e-4, "
                 "seed 42) — LOADED read-only by g1bS5; the licensed "
                 "consolidation source (the SAME start for all three doses)"),
        "step": istep}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =====================================================================
    # THE COMMITTED-ENDPOINT CROSS-CHECK (warn-only; the frozen curve priors
    # vs the committed files — Rule 12 provenance)
    # =====================================================================
    priors_check = {"g1bs3": None, "g1bs4": None, "g1bs2": None}
    for run_name, point, gm_key in (("g1bS3", G1BS3_POINT, None),
                                     ("g1bS4", G1BS4_POINT, None),
                                     ("g1bS2", G1BS2_CASUALTY, None)):
        _p = E43.REPO / "runs" / run_name / "metrics.json"
        try:
            _d = json.loads(_p.read_text(encoding="utf-8"))
            _row = _d["traces"]["C"][0]
            _d_gm = abs(_row["gm12"] - point["gm12"])
            _d_ce = abs(_row["ce_r"] - point["ce_r"])
            priors_check[run_name.replace("g1bS", "g1bs")] = {
                "file": f"runs/{run_name}/metrics.json",
                "file_gm12": _row["gm12"], "frozen_gm12": point["gm12"],
                "abs_d_gm12": _d_gm, "abs_d_ce_r": _d_ce,
                "match": bool(_d_gm <= 1e-9 and _d_ce <= 5e-3),
            }
            log(f"[priors] {run_name}: committed gm12 {_row['gm12']:.4f} vs "
                f"frozen {point['gm12']:.4f} (|d| {_d_gm:.1e}) — "
                f"{'match' if _d_gm <= 1e-9 else 'MISMATCH (warn)'}")
        except Exception as e:                                     # noqa: BLE001
            priors_check[run_name.replace("g1bS", "g1bs")] = {"error": str(e)}
            log(f"[priors] {run_name} unreadable ({e}) — continuing "
                f"(frozen priors stand; recorded)")

    # =====================================================================
    # THE THREE CONSOLIDATIONS (e113 jitter convention, licensed lr 4e-4,
    # seed 10901 — VERBATIM arithmetic; the ONLY delta per run is the DOSE)
    # =====================================================================
    roots: dict[str, dict] = {}          # tag -> {cells, root_read, cons}
    g1bs4_traj = None
    _s4 = E43.REPO / "runs" / "g1bS4" / "metrics.json"
    try:
        _s4r = json.loads(_s4.read_text(encoding="utf-8"))
        g1bs4_traj = {t["step"]: t
                      for t in _s4r["provenance"]["consolidation"]["traj"]}
    except Exception as e:                                         # noqa: BLE001
        log(f"[replay] g1bS4 committed traj unreadable ({e}) — the vs-g1bS4 "
            f"overlap check is skipped (recorded)")

    for di, (mv, steps, tg) in enumerate(DOSES):
        log("=" * 78)
        if di > 0 and not SMOKE:
            log(f"[owner envelope] cooldown {COOLDOWN_S:.0f}s before dose "
                f"{tg} (no back-to-back bursts)")
            cooldown(COOLDOWN_S)
        log(f"DOSE {tg}: {steps} steps @ lr {CONS_LR} = {mv:.2f} rms total "
            f"movement ({mv} / {CONS_LR} = {mv / CONS_LR:.0f} — the "
            f"registered arithmetic), jitters {G1.JITTERS}, batch 16 "
            f"install + 16 anchor, name-masked union CE, AdamW (0.9,0.95) "
            f"wd 0.1 clip 1.0, seed {G1.CONS_SEED} — from the SAME loaded "
            f"post-install state")

        def _chunk_partial(summary: dict, _tg=tg) -> None:
            write_partial(rd, f"cons-{_tg}-chunk{summary['chunk']}",
                          {"doses_partial": {_tg: {
                              "consolidation_partial": summary}}})

        try:
            cons = g1bS5_consolidate(
                G1.evl_load(theta_install), pool_a_x, pool_a_mask,
                cons_anchor, train_ids, g0_ids, r_eval_xy, zid, tg,
                steps=steps, lr=CONS_LR,
                resume_ck=(CKPT_DIR /
                           (f"smoke_g1bS5_cons_{tg}_resume.pt" if SMOKE
                            else f"g1bS5_cons_{tg}_resume.pt")),
                on_chunk=_chunk_partial)
        except OwnerWindowShut as e:
            write_partial(rd, f"owner-window-shut-{tg}",
                          {"stopped_at": tg, "reason": str(e),
                           "device_events": device_events})
            log(f"OWNER WINDOW SHUT at dose {tg}: {e} — stopping honestly "
                f"(partial metrics + resume ckpts hold the record)")
            raise SystemExit(3)
        theta_root = cons["sd"]
        if cons["steps_ran"] < steps:
            trims.append(f"consolidate {tg} capped at step "
                         f"{cons['steps_ran']} of {steps} (documented, not "
                         f"silent — resume ckpt exhausted; re-launch "
                         f"continues it)")

        # ---- replay cross-checks (texture, never a gate) -----------------
        replay = {"vs_g1bs4": None, "vs_prev_run": None}
        mine = {t["step"]: t for t in cons["traj"]}
        if g1bs4_traj is not None and not SMOKE:
            rows = []
            for s, e in sorted(mine.items()):
                if s in g1bs4_traj:
                    rows.append({"step": s,
                                 "g0_this": e["install60_g0_pz"],
                                 "g0_g1bs4": g1bs4_traj[s]["install60_g0_pz"],
                                 "ce_this": e["ce_r"],
                                 "ce_g1bs4": g1bs4_traj[s]["ce_r"]})
            replay["vs_g1bs4"] = {
                "n_overlap": len(rows),
                "mean_d_g0": (sum(abs(r["g0_this"] - r["g0_g1bs4"])
                                  for r in rows) / len(rows)) if rows else None,
                "max_d_g0": max((abs(r["g0_this"] - r["g0_g1bs4"])
                                 for r in rows), default=None),
                "reading": ("same seed 10901 + same draw order + same net0 "
                            "+ same lr => this run replays g1bS4's committed "
                            "750-step traj prefix up to device float fuzz "
                            "(g1bS4's own overlap measured mean |dg0| "
                            "0.0218 / max 0.111 — the curve's fuzz floor)"),
            }
        prev_tags = [t for _, _, t in DOSES[:di]]
        if prev_tags and not SMOKE:
            prev_traj = {t["step"]: t
                         for t in roots[prev_tags[-1]]["cons"]["traj"]}
            rows = []
            for s, e in sorted(mine.items()):
                if s in prev_traj:
                    rows.append({"step": s,
                                 "g0_this": e["install60_g0_pz"],
                                 "g0_prev": prev_traj[s]["install60_g0_pz"],
                                 "ce_this": e["ce_r"],
                                 "ce_prev": prev_traj[s]["ce_r"]})
            replay["vs_prev_run"] = {
                "prev_tag": prev_tags[-1],
                "n_overlap": len(rows),
                "mean_d_g0": (sum(abs(r["g0_this"] - r["g0_prev"])
                                  for r in rows) / len(rows)) if rows else None,
                "max_d_g0": max((abs(r["g0_this"] - r["g0_prev"])
                                 for r in rows), default=None),
                "reading": ("run-to-run prefix replay (the same-recipe "
                            "fuzz floor measured INSIDE this cell)"),
            }
        if not SMOKE:
            _r46, _rp = replay["vs_g1bs4"], replay["vs_prev_run"]
            log(f"[replay {tg}] vs g1bS4: "
                + (f"n {_r46['n_overlap']} mean|dg0| {_r46['mean_d_g0']:.2e} "
                   f"max {_r46['max_d_g0']:.2e}" if _r46 and
                   (_r46.get("n_overlap") or 0) > 0 else "no overlap rows")
                + (f" | vs {replay['vs_prev_run']['prev_tag']}: "
                   f"mean|dg0| {_rp['mean_d_g0']:.2e} "
                   f"max {_rp['max_d_g0']:.2e}" if _rp and
                   (_rp.get("n_overlap") or 0) > 0 else ""))

        # ---- the ROOT DIAL (the prior takes' convention: full measure) ---
        root_m = measure(theta_root, f"g1bS5_{tg}", lean=SMOKE)
        root_cells = flat_cells(root_m)
        root_read = {"bar": G1.EXPRESS_BAR, "gm12": root_cells["gm12"],
                     "pass": bool(root_cells["gm12"] >= G1.EXPRESS_BAR),
                     "note": ("a READ, not a hard stop — no arms to stop in "
                              "this cell; the curve bars adjudicate")}
        log(f"ROOT {tg}: g-12 {root_cells['gm12']:.4f} (bar >= "
            f"{G1.EXPRESS_BAR}) g0 {root_cells['g0']:.4f} CE_R "
            f"{root_cells['ce_r']:.4f} held30_gm12 "
            f"{root_cells['held30_gm12']:.4f} | "
            f"{'PASS' if root_read['pass'] else 'below bar'}")
        roots[tg] = {"cells": root_cells, "root_read": root_read,
                     "cons": cons, "replay": replay}
        write_partial(rd, f"root-{tg}", {
            "doses_partial": {tg: {
                "movement_rms": mv, "steps": steps, "lr": CONS_LR,
                "consolidation": {
                    "steps_ran": cons["steps_ran"], "seed": cons["seed"],
                    "device": cons["device"], "n_chunks": cons["n_chunks"],
                    "chunk_table": cons["chunk_table"], "traj": cons["traj"]},
                "replay": replay,
                "root_read": root_read, "root_cells": root_cells}}})
        save_ckpt(f"g1bS5_root_{tg}", theta_root,
                  {"desc": f"g1bS2's PASSED base+install (LOADED read-only) "
                           f"+ e113 jitter consolidation s{cons['steps_ran']} "
                           f"@ lr {CONS_LR} (seed {G1.CONS_SEED}) = movement "
                           f"{mv} rms — the {tg} point of the formation curve",
                   "params": n_params, "movement_rms": mv,
                   "steps": cons["steps_ran"], "lr": CONS_LR,
                   "cons_seed": G1.CONS_SEED,
                   "root_gm12": root_cells["gm12"],
                   "base": f"runs/checkpoints/{SRC_BASE_CK}.pt",
                   "install": f"runs/checkpoints/{SRC_INSTALL_CK}.pt"})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =====================================================================
    # THE CURVE + THE ADJUDICATION (frozen bars; operationalization
    # registered BEFORE compute in REGISTERED.operationalization)
    # =====================================================================
    tags = [t for _, _, t in DOSES]
    curve = {
        "x_movement_rms": [0.12, 0.15, 0.20, 0.25, 0.30],
        "x_steps": [300, DOSES[0][1], DOSES[1][1], DOSES[2][1], 750],
        "gm12": [G1BS3_POINT["gm12"], roots["m015"]["cells"]["gm12"],
                 roots["m020"]["cells"]["gm12"], roots["m025"]["cells"]["gm12"],
                 G1BS4_POINT["gm12"]],
        "g0": [G1BS3_POINT["g0"], roots["m015"]["cells"]["g0"],
               roots["m020"]["cells"]["g0"], roots["m025"]["cells"]["g0"],
               G1BS4_POINT["g0"]],
        "held30_gm12": [G1BS3_POINT["held30_gm12"],
                        roots["m015"]["cells"]["held30_gm12"],
                        roots["m020"]["cells"]["held30_gm12"],
                        roots["m025"]["cells"]["held30_gm12"],
                        G1BS4_POINT["held30_gm12"]],
        "ce_r": [G1BS3_POINT["ce_r"], roots["m015"]["cells"]["ce_r"],
                 roots["m020"]["cells"]["ce_r"], roots["m025"]["cells"]["ce_r"],
                 G1BS4_POINT["ce_r"]],
        "committed": [True, False, False, False, True],
        "casualty_point_separate": G1BS2_CASUALTY,
        "fuzz_floor": {"mean_d_g0": 0.0218, "max_d_g0": 0.111,
                       "source": ("g1bS4's committed replay overlap "
                                  "(same recipe, same seed); the in-cell "
                                  "run-to-run overlaps are in "
                                  "doses.*.replay")},
    }
    interior = curve["gm12"][1:4]
    sharp_threshold = G1BS3_POINT["gm12"] + 0.05          # 0.6998...
    tol = 0.02
    sharp = bool(max(interior) >= sharp_threshold)
    mono_pairs = [(curve["gm12"][i], curve["gm12"][i + 1])
                  for i in range(len(curve["gm12"]) - 1)]
    monotone = bool(all(later <= earlier + tol
                        for earlier, later in mono_pairs))
    peak_i = max(range(len(curve["gm12"])), key=lambda i: curve["gm12"][i])
    if sharp:
        fired = "SHARP-OPTIMUM"
        clause = (f"the curve PEAKS in the interior: max interior reading "
                  f"{max(interior):.4f} at "
                  f"{curve['x_movement_rms'][1 + interior.index(max(interior))]:.2f}"
                  f" rms (s{curve['x_steps'][1 + interior.index(max(interior))]}"
                  f") >= the registered threshold {sharp_threshold:.4f} "
                  f"(= g1bS3's 0.6498 + 0.05"
                  + ("; also >= the 0.78 express bar" if max(interior) >= 0.78
                     else "") + f"); the overall peak sits at "
                  f"{curve['x_movement_rms'][peak_i]:.2f} rms (s"
                  f"{curve['x_steps'][peak_i]}, reading "
                  f"{curve['gm12'][peak_i]:.4f}) — THE FORMATION OPTIMUM "
                  f"LOCATED between 0.12 and 0.30 rms at 4e-4; a fifth take "
                  f"at the peak movement would finally adjudicate the wall. "
                  f"Curve verbatim: "
                  + " -> ".join(f"{x:.2f}:{v:.4f}" for x, v in
                                zip(curve["x_movement_rms"], curve["gm12"]))
                  + f". Fuzz caveat: the same-recipe replay floor is mean "
                  f"|dg0| 0.0218 / max 0.111 — the margin over the "
                  f"threshold ({max(interior) - sharp_threshold:+.4f}) "
                  f"{'EXCEEDS' if max(interior) - sharp_threshold > 0.111 else 'is WITHIN'} "
                  f"the max-fuzz scale (co-reported honestly).")
    elif monotone:
        fired = "MONOTONE-DECLINE"
        clause = (f"the curve falls monotonically from 0.12 (within the "
                  f"registered +{tol} fuzz band): "
                  + " -> ".join(f"{x:.2f}:{v:.4f}" for x, v in
                                zip(curve["x_movement_rms"], curve["gm12"]))
                  + f" — g1bS3's 0.12 rms WAS the optimum; more "
                  f"consolidation movement only dissolves the displaced-"
                  f"geometry channel; the e113 form's 10M ceiling IS the "
                  f"finding, honestly accepted (the 1e-3 casualty stays "
                  f"separate: a stability failure, not a dose point).")
    else:
        fired = "GRADED"
        ups = [f"{curve['x_movement_rms'][i]:.2f}->{curve['x_movement_rms'][i + 1]:.2f} "
               f"({curve['gm12'][i]:.4f}->{curve['gm12'][i + 1]:.4f})"
               for i, (a, b) in enumerate(mono_pairs) if b > a + tol]
        ups_txt = ", ".join(ups) if ups else "none (within band, but not cleanly)"
        clause = (f"a partial — neither frozen bar fires cleanly: no interior "
                  f"reading >= {sharp_threshold:.4f} (max interior "
                  f"{max(interior):.4f}) AND the curve is not monotone "
                  f"within +{tol} (rises at: {ups_txt}). The curve verbatim: "
                  + " -> ".join(f"{x:.2f}:{v:.4f}" for x, v in
                                zip(curve["x_movement_rms"], curve["gm12"]))
                  + ". The 1e-3 casualty stays separate.")
    log("=" * 78)
    log(f"THE FORMATION CURVE @ 4e-4: "
        + " -> ".join(f"{x:.2f} rms (s{s}): {v:.4f}"
                      for x, s, v in zip(curve["x_movement_rms"],
                                         curve["x_steps"], curve["gm12"])))
    log(f"CASUALTY (separate, rate 1e-3): 0.30 rms (s300): "
        f"{G1BS2_CASUALTY['gm12']:.4f}")
    log(f"G1BS5 VERDICT: {fired}")
    log(f"  {clause}")
    log("=" * 78)
    write_partial(rd, "curve+adjudication", {
        "curve": curve, "adjudication_partial": {
            "fired": fired, "clause": clause,
            "sharp_threshold": sharp_threshold, "tol": tol,
            "interior": interior, "peak_i": peak_i}})

    # =====================================================================
    # OUTPUTS
    # =====================================================================
    metrics = {
        "experiment": "g1bS5_formation_curve",
        "date": common.now_iso(),
        "partial": False,
        "progressive_writes": _progressive["n"],
        "phases": list(_progressive["phases"]),
        "design": ("T165's named cut (the dispatch, 2026-10-02): the "
                   "formation-vs-movement curve at 10M — consolidation-only "
                   "(no install, no wall arms, no wash); bars frozen "
                   "verbatim; no bar shopping"),
        "question": ("where does the displaced-geometry g-12 channel's "
                     "formation strength peak as a function of total "
                     "consolidation movement at 4e-4 on the 10M host?"),
        "registered": REGISTERED,
        "provenance": {
            "host": {"config": {"n_layer": 8, "n_head": 8, "n_embd": 320,
                                "block_size": 256, "vocab": 65},
                     "params": n_params, "band": "[8.5M, 11.5M]",
                     "config_provenance": "e005s LARGE's 10M-class config "
                                          "(the lineage's standing host)",
                     "host_seed": HOST_SEED, "corpus_seed": 1337,
                     "base": ("g1bS2's PASSED base LOADED VERBATIM from "
                              f"runs/checkpoints/{SRC_BASE_CK} (never "
                              "retrained; val-min-anchored 1200-step cosine, "
                              "lr 4e-4; G-BASE-QUAL re-derived on the load)"),
                     "final_val_loss": G_BASE["final_val_loss"]},
            "install": {"steps": INSTALL_STEPS, "seed": G1.INSTALL_SEED,
                        "source": ("g1bS2's movement-matched install LOADED "
                                   f"VERBATIM from runs/checkpoints/"
                                   f"{SRC_INSTALL_CK} (post-install state "
                                   "REUSABLE; cells re-measured and "
                                   "cross-checked by this run)"),
                        "post_install_cells": inst_cells},
            "consolidations": {
                tg: {"movement_rms": mv, "steps": steps, "lr": CONS_LR,
                     "seed": G1.CONS_SEED,
                     "recipe": ("e113 jitter convention VERBATIM (jitters "
                                "{-8..+8}, batch 16 install + 16 anchor, "
                                "name-masked union CE, AdamW (0.9,0.95) "
                                "wd 0.1 const lr clip 1.0, seed 10901) at "
                                "the G1BS3-licensed rate 4e-4 — the dose "
                                "is the ONLY delta across the three runs "
                                "and vs g1bS3/g1bS4's single points"),
                     "steps_ran": roots[tg]["cons"]["steps_ran"],
                     "device": roots[tg]["cons"]["device"],
                     "n_chunks": roots[tg]["cons"]["n_chunks"],
                     "chunk_table": roots[tg]["cons"]["chunk_table"],
                     "traj": roots[tg]["cons"]["traj"],
                     "replay": roots[tg]["replay"],
                     "root_read": roots[tg]["root_read"],
                     "root_cells": roots[tg]["cells"]}
                for mv, steps, tg in DOSES},
            "owner_envelope": {
                "launch_gate": REGISTERED["owner_envelope"]["launch_gate"],
                "burst_cap_s": TRAIN_CAP_GPU, "cooldown_s": COOLDOWN_S,
                "events": device_events},
            "priors_crosscheck": priors_check,
            "seeds": {"host_init_base_corpus": HOST_SEED,
                      "install": G1.INSTALL_SEED,
                      "consolidation": G1.CONS_SEED,
                      "protocol_corpus": 1337},
        },
        "curve": curve,
        "adjudication": {
            "bars_verbatim": REGISTERED["bars_verbatim"],
            "operationalization": REGISTERED["operationalization"],
            "sharp_threshold": sharp_threshold,
            "monotone_tol": tol,
            "interior_readings": interior,
            "peak": {"movement_rms": curve["x_movement_rms"][peak_i],
                     "steps": curve["x_steps"][peak_i],
                     "gm12": curve["gm12"][peak_i]},
            "SHARP_OPTIMUM_fired": sharp,
            "MONOTONE_DECLINE_fired": monotone,
            "fired": fired,
            "clause": clause,
            "casualty_separate": ("g1bS2's 1e-3 point (0.30 rms at RATE "
                                 "1e-3 -> 0.0010) is NOT on the 4e-4 curve: "
                                 "its failure was stability (the CE_R "
                                 "blowout), not dose"),
            "no_bar_shopping": ("the three doses and the bars were frozen "
                                "in the dispatch and registered in the "
                                "script BEFORE any compute; the committed "
                                "endpoints are prior takes' records, "
                                "cross-checked (warn-only) not re-tuned"),
        },
        "gates": {"G_CONFIG": G_CONFIG, "G_BASE": G_BASE,
                  "G_BASE_QUAL": G_BASE_QUAL, "G_INST": G_INST,
                  "G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                  "G_POOL": G_POOL, "G_SURG": gates_surg,
                  "pass": bool(G_CONFIG["pass"] and G_BASE["pass"]
                               and G_BASE_QUAL["pass"] and G_INST["pass"])},
        "honesty_reflex": {
            "n1_scope": ("n=1 host, one consolidation seed (10901), one "
                         "fact — the curve is ONE draw of the dose landscape "
                         "per point; the run-to-run replay overlaps bound "
                         "the same-recipe float fuzz, not the seed variance "
                         "(a different seed could shift the whole curve); "
                         "the interior points are single readings, honestly "
                         "stamped"),
            "consolidation_only": ("NO WALL CLAIMS: no arms, no wash, no "
                                   "commit-and-project ran in this cell; "
                                   "the SHARP-OPTIMUM clause's 'a fifth take "
                                   "at the peak movement would finally "
                                   "adjudicate the wall' names the FOLLOW-ON, "
                                   "not a result — the wall's scale question "
                                   "remains exactly as T165 left it"),
            "committed_endpoint_texture": ("the 0.12 and 0.30 endpoints are "
                                           "PRIOR TAKES' records (different "
                                           "processes, GPU consolidations "
                                           "with their own device histories; "
                                           "g1bS4's root row was assembled by "
                                           "its recovery runner at print "
                                           "precision — disclosed); the "
                                           "curve joins five readings whose "
                                           "fuzz floor is the measured "
                                           "same-recipe replay scale "
                                           "(mean |dg0| 0.0218 / max 0.111)"),
            "channel_note": ("the registered channel is g-12 (the "
                             "displaced-geometry battery); the g0 site "
                             "channel is a CO-READ (its in-run trajectory "
                             "is previewable from g1bS4's committed traj by "
                             "the replay property — the overlap check "
                             "verifies the machinery, never a gate)"),
            "openness": ("nothing was guaranteed — T165's cut was open by "
                         "design; the registered prediction was NONE and "
                         "the verdict above is the curve's, not a thesis's"),
        },
        "trims": trims, "deviations": deviations,
        "device_events": device_events,
        "ckpt_inventory": CKPT_INVENTORY,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 8, "n_head": 8, "n_embd": 320,
                   "block_size": 256, "params": n_params,
                   "smoke": SMOKE},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "formation_curve.png", curve, roots, fired, clause,
         sharp_threshold, tol, G1BS2_CASUALTY)

    log(f"outputs: {rd / 'metrics.json'}, {rd / 'formation_curve.png'}, "
        f"{len(CKPT_INVENTORY)} checkpoints in runs/checkpoints/g1bS5_*.pt")
    log(f"VERDICT: {fired}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ driver

def g1bS5_consolidate(net0, pool_x, pool_mask, anchor, train_ids,
                      f_eval_ids, r_eval_xy, zid, tag,
                      steps: int, lr: float = CONS_LR,
                      resume_ck: Path | None = None,
                      on_chunk=None) -> dict:
    """e113's finetune_arm VERBATIM arithmetic (g1/g1bS3/g1bS4's
    consolidate): draw order ix/aj/rj, batch 16 install + 16 anchor,
    name-masked union CE, AdamW (0.9,0.95) wd 0.1 const lr clip 1.0, seed
    10901, eval every 25 steps — with the OWNER-ENVELOPE adaptations ONLY
    (device policy; no training-semantics change): the launch gate is
    util<=20% AND temp<=70C double-polled (wait_gpu_owner), the burst cap is
    90 s (TRAIN_CAP_GPU), the between-chunk cooldown is 180 s, outside load
    mid-burst => PAUSE-AND-WAIT (never migrate), no CPU fallback. CHUNK-
    RESUMABLE (the lineage's standing resilience mandate): full-state
    resume ckpt ({model, opt, gen_state, step, traj}) every 125 steps and
    at every chunk boundary — bit-identical to uninterrupted execution
    because the generator + optimizer state carry the whole stream."""
    n_pool, n_anc = pool_x.shape[0], anchor.shape[0]
    RESUME_EVERY = 125 if not SMOKE else 4
    chunk_table, devices = [], []
    n_chunks = 0
    step, traj, net, opt, gen = 0, [], None, None, None
    active_s = 0.0
    # edge: a prior process saved the FINAL state but died before returning
    if resume_ck is not None and Path(resume_ck).exists():
        _pre = torch.load(resume_ck, map_location="cpu", weights_only=False)
        if int(_pre.get("step", 0)) >= steps:
            log(f"  [{tag}] resume ckpt already COMPLETE at s{_pre['step']} — "
                f"returning the saved final state (no steps re-run)")
            return {"sd": {k: v.detach().clone() for k, v
                           in _pre["model"].items()},
                    "traj": _pre.get("traj", []), "steps_ran": steps,
                    "device": "resumed (final state)",
                    "n_chunks": _pre.get("n_chunks", 0),
                    "chunk_table": [{"chunk": 0, "device": "resume",
                                     "seconds": None, "steps_done": steps,
                                     "capped": False,
                                     "note": "returned from the saved final "
                                             "state (prior process died "
                                             "post-save)"}],
                    "seed": G1.CONS_SEED, "lr": lr}
    while step < steps:
        n_chunks += 1
        dev = wait_gpu_owner(f"{tag}-chunk{n_chunks}")   # OWNER envelope
        cap = TRAIN_CAP_GPU                             # <= 90 s, always
        if net is None:
            net = copy.deepcopy(net0).to(dev)
            net.train()
            opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95),
                                    weight_decay=0.1)
            gen = torch.Generator().manual_seed(G1.CONS_SEED)
            if resume_ck is not None and Path(resume_ck).exists():
                state = torch.load(resume_ck, map_location="cpu",
                                   weights_only=False)
                net.load_state_dict(state["model"])
                net.to(dev)
                net.train()
                opt.load_state_dict(state["opt"])
                gen.set_state(state["gen_state"])
                step, traj = state["step"], state.get("traj", [])
                active_s = state.get("active_s", 0.0)
                log(f"  [{tag}] RESUMED from {Path(resume_ck).name} at step "
                    f"{step}/{steps} ({len(traj)} eval points carried)")
        else:
            _pd = str(next(net.parameters()).device)
            if _pd != str(dev):
                _osd = opt.state_dict()
                net = net.to(dev)
                opt = torch.optim.AdamW(net.parameters(), lr=lr,
                                        betas=(0.9, 0.95), weight_decay=0.1)
                opt.load_state_dict(_osd)
                log(f"  [{tag}] chunk {n_chunks}: device moved {_pd} -> "
                    f"{dev} (optimizer state recast)")
            net.train()
        devices.append(str(dev))
        evl = copy.deepcopy(net0).to(dev)      # eval twin rides the device
        t_start = time.time()
        chunk_capped = False
        for step in range(step + 1, steps + 1):
            ix = torch.randint(n_pool, (16,), generator=gen)
            aj = torch.randint(n_anc, (8,), generator=gen)
            rj = torch.randint(len(train_ids) - G1.BLOCK - 1, (8,),
                               generator=gen)
            nw = pool_x[ix].to(dev)
            anc = torch.cat([anchor[aj],
                             torch.stack([train_ids[s: s + G1.BLOCK]
                                          for s in rj])], 0).to(dev)
            x = torch.cat([nw[:, :-1], anc[:, :-1]], 0)
            y = torch.cat([nw[:, 1:], anc[:, 1:]], 0)
            m = torch.zeros(32, x.shape[1], dtype=torch.bool, device=dev)
            m[:16] = pool_mask[ix].to(dev)
            logits, _ = net(x)
            nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                  y.reshape(-1), reduction="none"
                                  ).view(x.shape[0], x.shape[1])
            nm = nll[:16][m[:16]]
            cm = nll[16:]
            loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if step % 25 == 0 or step == steps or \
                    (time.time() - t_start) > cap:
                evl.load_state_dict(net.state_dict())
                evl.eval()
                bz = G1.battery_cell(evl, f_eval_ids.to(dev), zid)
                ce_r = G1.ce_fixed_cpu(evl, r_eval_xy[0].to(dev),
                                       r_eval_xy[1].to(dev))
                traj.append({"step": step, "install60_g0_pz": bz["mean_pz"],
                             "ce_r": ce_r,
                             "elapsed_s": round(active_s + time.time()
                                                - t_start, 1)})
                log(f"  [{tag}] s{step:4d} g0 {bz['mean_pz']:.4f} CE_R "
                    f"{ce_r:.4f} ({traj[-1]['elapsed_s']:.0f}s)")
            if resume_ck is not None and \
                    (step % RESUME_EVERY == 0 or step == steps):
                torch.save({"model": {k: v.detach().cpu()
                                      for k, v in net.state_dict().items()},
                            "opt": opt.state_dict(),
                            "gen_state": gen.get_state(),
                            "step": step, "traj": traj,
                            "active_s": active_s + time.time() - t_start,
                            "lr": lr, "n_chunks": n_chunks},
                           resume_ck)
            if (time.time() - t_start) > cap:
                log(f"  [{tag}] chunk {n_chunks}: owner-envelope burst cap "
                    f"{cap:.0f}s at s{step} — resume ckpt saved; next chunk "
                    f"continues after the cooldown")
                chunk_capped = True
                break
            if dev.type == "cuda" and step % G1.MIDRUN_POLL_EVERY == 0:
                s = gpu_status()
                if s["mem_total"] > 0 and (s["mem_used"] > 0.85 * s["mem_total"]
                                           or s["temp"] > 80):
                    device_events.append({"tag": tag, "step": step,
                                          "event": "MID-RUN PAUSE (no "
                                                   "migration)", "status": s})
                    t_p = time.time()
                    midrun_pause_wait(tag)
                    t_start += time.time() - t_p     # paused time is not cap
        chunk_secs = round(time.time() - t_start, 1)
        active_s += chunk_secs
        chunk_table.append({"chunk": n_chunks, "device": str(dev),
                            "seconds": chunk_secs,
                            "steps_done": step, "capped": chunk_capped})
        if on_chunk is not None:
            on_chunk({"chunk": n_chunks, "device": str(dev),
                      "seconds": chunk_secs, "steps_done": step,
                      "capped": chunk_capped, "steps_total": steps,
                      "traj": traj})
        if step >= steps:
            break
        if n_chunks >= 24:                # pathological-cap guard
            trims.append(f"{tag}: stopped at chunk {n_chunks} (cap-loop "
                         f"guard) at step {step} of {steps}")
            break                          # (documented, not silent)
        if not SMOKE:
            cooldown(COOLDOWN_S)          # >= 180 s between bursts (owner)
        del evl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    net.eval()
    sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"sd": sd_cpu, "traj": traj, "steps_ran": step,
            "device": devices[0] if devices else "n/a", "devices": devices,
            "n_chunks": n_chunks, "chunk_table": chunk_table,
            "seed": G1.CONS_SEED, "lr": lr}


CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "g1bS5", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name}")


# ------------------------------------------------------------------ plot
# THE FORMATION CURVE figure: root g-12 vs total movement (the headline),
# the stability/holding/g0 co-reads, the in-run trajectories, the verdict.

def plot(path, curve, roots, fired, clause, sharp_threshold, tol,
         casualty):
    import textwrap
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.5))
    xs = curve["x_movement_rms"]
    gm = curve["gm12"]
    committed = curve["committed"]
    fuzz_mean, fuzz_max = (curve["fuzz_floor"]["mean_d_g0"],
                           curve["fuzz_floor"]["max_d_g0"])

    def split(series):
        cx, cy, nx, ny = [], [], [], []
        for x, v, c in zip(xs, series, committed):
            (cx.append(x), cy.append(v)) if c else (nx.append(x), ny.append(v))
        return cx, cy, nx, ny

    # (0,0) THE HEADLINE: the formation curve
    ax = axes[0, 0]
    ax.plot(xs, gm, "-", lw=1.4, color="gray", alpha=0.6, zorder=1)
    cx, cy, nx, ny = split(gm)
    ax.plot(cx, cy, "o", ms=11, mfc="white", mec="black", mew=2, zorder=3,
            label="committed (g1bS3 s300 / g1bS4 s750)")
    ax.errorbar(nx, ny, yerr=[[min(fuzz_mean, v) for v in ny],
                              [min(fuzz_mean, 1 - v) for v in ny]],
                fmt="s", ms=10, color="seagreen", capsize=4, zorder=4,
                label="NEW (this cell: s375/s500/s625 @ 4e-4)")
    for x, v in zip(nx, ny):
        ax.annotate(f"{v:.4f}", (x, v), textcoords="offset points",
                    xytext=(0, 10), fontsize=8, ha="center", weight="bold")
    for x, v in zip(cx, cy):
        ax.annotate(f"{v:.4f}", (x, v), textcoords="offset points",
                    xytext=(0, -16), fontsize=8, ha="center")
    ax.plot([casualty["movement_rms"]], [casualty["gm12"]], "x", ms=12,
            mew=3, color="crimson", zorder=4,
            label="1e-3 casualty (g1bS2, rate 1e-3 — SEPARATE)")
    ax.axhline(G1.EXPRESS_BAR, ls="--", lw=1.8, color="black", alpha=0.9,
               label=f"express bar {G1.EXPRESS_BAR}")
    ax.axhline(sharp_threshold, ls="-.", lw=1.4, color="darkorange",
               alpha=0.9, label=f"SHARP-OPTIMUM threshold {sharp_threshold:.4f} "
                                "(0.6498+0.05)")
    ax.axhline(G1BS3_POINT["gm12"], ls=":", lw=1.1, color="gray",
               alpha=0.8, label=f"g1bS3's 0.12 reading {G1BS3_POINT['gm12']:.4f}")
    ax.annotate("the stability casualty:\n0.30 rms at RATE 1e-3 (CE blowout)\n"
                "kept OFF the 4e-4 curve",
                (casualty["movement_rms"], casualty["gm12"]),
                textcoords="offset points", xytext=(10, 22), fontsize=7.5,
                color="crimson")
    peak_i = max(range(len(gm)), key=lambda i: gm[i])
    ax.annotate(f"peak {xs[peak_i]:.2f} rms", (xs[peak_i], gm[peak_i]),
                textcoords="offset points", xytext=(0, 22), fontsize=9,
                ha="center", weight="bold",
                arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.set_xlabel("total consolidation movement (rms, Adam-clock @ 4e-4)")
    ax.set_ylabel("root g-12 (displaced-geometry battery)")
    ax.set_ylim(-0.05, 1.08)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("THE FORMATION CURVE at 10M — root g-12 vs movement",
                 fontsize=10)

    # (0,1) CE_R (the stability axis)
    ax = axes[0, 1]
    ce = curve["ce_r"]
    ax.plot(xs, ce, "-", lw=1.4, color="gray", alpha=0.6, zorder=1)
    cx, cy, nx, ny = split(ce)
    ax.plot(cx, cy, "o", ms=10, mfc="white", mec="black", mew=2, zorder=3,
            label="committed")
    ax.plot(nx, ny, "s", ms=9, color="seagreen", zorder=4, label="NEW")
    ax.plot([casualty["movement_rms"]], [casualty["ce_r"]], "x", ms=12,
            mew=3, color="crimson", zorder=4, label="1e-3 casualty (SEPARATE)")
    ax.axhspan(1.5, 1.8, color="seagreen", alpha=0.10)
    ax.annotate("healthy band ~1.5-1.8 (g1bS3/4 settled range)",
                (0.125, 1.76), fontsize=7.5, color="seagreen")
    ax.set_xlabel("total consolidation movement (rms @ 4e-4)")
    ax.set_ylabel("root CE_R (60-window val bank)")
    ax.legend(fontsize=7.5)
    ax.set_title("THE STABILITY AXIS — CE_R vs movement", fontsize=10)

    # (0,2) held30_gm12 (generality of the displaced-geometry channel)
    ax = axes[0, 2]
    hd = curve["held30_gm12"]
    ax.plot(xs, hd, "-", lw=1.4, color="gray", alpha=0.6, zorder=1)
    cx, cy, nx, ny = split(hd)
    ax.plot(cx, cy, "o", ms=10, mfc="white", mec="black", mew=2, zorder=3,
            label="committed")
    ax.plot(nx, ny, "s", ms=9, color="royalblue", zorder=4, label="NEW")
    ax.set_xlabel("total consolidation movement (rms @ 4e-4)")
    ax.set_ylabel("held30 g-12 (untrained hosts)")
    ax.legend(fontsize=7.5)
    ax.set_title("GENERALITY — held30 g-12 vs movement", fontsize=10)

    # (1,0) g0 (the jitter-0 site channel — a CO-READ, never the ruler)
    ax = axes[1, 0]
    g0s = curve["g0"]
    ax.plot(xs, g0s, "-", lw=1.4, color="gray", alpha=0.6, zorder=1)
    cx, cy, nx, ny = split(g0s)
    ax.plot(cx, cy, "o", ms=10, mfc="white", mec="black", mew=2, zorder=3,
            label="committed")
    ax.plot(nx, ny, "s", ms=9, color="darkorange", zorder=4, label="NEW")
    ax.plot([casualty["movement_rms"]], [casualty["g0"]], "x", ms=12,
            mew=3, color="crimson", zorder=4, label="1e-3 casualty (SEPARATE)")
    ax.set_xlabel("total consolidation movement (rms @ 4e-4)")
    ax.set_ylabel("root g0 (jitter-0 site channel — CO-READ)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7.5)
    ax.set_title("THE SITE CHANNEL (co-read) — g0 vs movement", fontsize=10)

    # (1,1) the in-run trajectories (the replay property made visible)
    ax = axes[1, 1]
    cols = {"m015": "seagreen", "m020": "royalblue", "m025": "darkorange"}
    for tg, col in cols.items():
        tj = roots[tg]["cons"]["traj"]
        ax.plot([t["step"] for t in tj], [t["install60_g0_pz"] for t in tj],
                "-", lw=1.5, color=col, alpha=0.85,
                label=f"{tg} (s{roots[tg]['cons']['steps_ran']}) in-run g0")
    try:
        _s4 = E43.REPO / "runs" / "g1bS4" / "metrics.json"
        _t4 = json.loads(_s4.read_text(encoding="utf-8"))
        _tr = _t4["provenance"]["consolidation"]["traj"]
        ax.plot([t["step"] for t in _tr],
                [t["install60_g0_pz"] for t in _tr], "k--", lw=1.0,
                alpha=0.45, label="g1bS4 committed s750 traj (replayed)")
        for st, lb in ((300, "g1bS3 end"), (750, "g1bS4 end")):
            ax.axvline(st, ls=":", lw=1.0, color="gray", alpha=0.6)
            ax.annotate(lb, (st, 0.06), fontsize=7, rotation=90,
                        color="gray", ha="right")
    except Exception:                                             # noqa: BLE001
        pass
    for tg in cols:
        ax.axvline(roots[tg]["cons"]["steps_ran"], ls=":", lw=1.2,
                   color=cols[tg], alpha=0.7)
    ax.set_xlabel("consolidation step (seed 10901 — shared prefix)")
    ax.set_ylabel("in-run g0 (site channel, every 25 steps)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("THE IN-RUN TRAJECTORIES (the replay property)", fontsize=10)

    # (1,2) THE VERDICT PANEL
    ax = axes[1, 2]
    ax.axis("off")
    y = 0.97
    ax.text(0.02, y, "G1BS5 — THE FORMATION CURVE (consolidation-only: "
            "s375/s500/s625 @ 4e-4)", fontsize=11, va="top",
            family="monospace", weight="bold")
    y -= 0.052
    ax.text(0.02, y, f"host {G1BS_PARAMS:,} params | lr {CONS_LR} | seed "
            f"{G1.CONS_SEED} | from the SAME loaded base+install | n=1 "
            f"host/seed/fact", fontsize=7.2, va="top", family="monospace")
    y -= 0.034
    ax.text(0.02, y, f"  curve (movement rms -> g-12): "
            + " -> ".join(f"{x:.2f}:{v:.4f}" for x, v in zip(xs, gm)),
            fontsize=6.8, va="top", family="monospace")
    y -= 0.028
    ax.text(0.02, y, f"  casualty (SEPARATE, rate 1e-3): 0.30 -> "
            f"{casualty['gm12']:.4f} | CE_R {casualty['ce_r']:.2f}",
            fontsize=6.8, va="top", family="monospace", color="crimson")
    y -= 0.028
    interior = gm[1:4]
    ax.text(0.02, y, f"  interior max {max(interior):.4f} vs threshold "
            f"{sharp_threshold:.4f} | monotone tol +{tol} (fuzz floor mean "
            f"{fuzz_mean}/max {fuzz_max})", fontsize=6.8, va="top",
            family="monospace")
    y -= 0.032
    for tg in ("m015", "m020", "m025"):
        c = roots[tg]["cells"]
        ax.text(0.02, y, f"  {tg}: g-12 {c['gm12']:.4f} g0 {c['g0']:.4f} "
                f"CE_R {c['ce_r']:.4f} held30 {c['held30_gm12']:.4f} "
                f"(express {'PASS' if c['gm12'] >= G1.EXPRESS_BAR else 'below'})",
                fontsize=6.6, va="top", family="monospace",
                color=cols[tg])
        y -= 0.028
    y -= 0.012
    ax.text(0.02, y, f"VERDICT: {fired}", fontsize=9.5, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.042
    for wd in textwrap.wrap(clause, width=88, break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=6.6, va="top",
                family="monospace")
        y -= 0.026

    fig.suptitle("G1BS5 — THE FORMATION CURVE at 10M: root g-12 vs total "
                 "consolidation movement at 4e-4 (consolidation-only; the "
                 f"committed endpoints joined) -> {fired}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
