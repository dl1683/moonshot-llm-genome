"""G1BS7 — THE REDRAWN INTERIOR DOSE (T167's honesty-ledger debt; the
coordinator's tenth-dispatch lineage, 2026-10-02). CONSOLIDATION-ONLY —
NO install, NO wall arms. The formation curve's fine-ordering leg.

==================== THE QUESTION (T167's owed redraw) ======================
g1bS5's curve (runs/g1bS5/metrics.json, T167): the 10M formation optimum
peaks at 0.20 rms (root g-12 0.7677 at s500 @ 4e-4), SHARP-OPTIMUM fired —
but the honesty ledger carried the critic's caveat: the curve is ONE draw
per point (seed 10901), and the fine ordering (0.20's 0.7677 vs 0.25's
0.7431) sits WITHIN the same-recipe replay fuzz (mean |dg0| 0.0218 / max
0.111). T167 named the debt: "a redrawn interior dose still owed for it."
THE REGISTERED QUESTION (the dispatch): the fine ordering needed a redraw;
the PEAK HEIGHT is what the wall rode on — g1bS6's WALL-FADES verdict ran
its arms on g1bS5_root_m020.pt, i.e. on ONE draw of the 0.7677 peak. Is
the peak's height draw-robust?

THE CELL (the cheapest possible form of the question — one fresh-jitter
consolidation at the peak dose): the SAME loaded base+install (g1bS2's
PASSED artifacts — never retrained), the consolidation at 4e-4 to 0.20 rms
total movement (500 steps — the registered Adam-clock arithmetic) with a
FRESH JITTER SEED: the original draw was 10901; THIS run uses 10903
(registered in the dispatch BEFORE compute). Everything else VERBATIM from
g1bS5's m020 run (same recipe, same pools, same lr, same steps, same
readout conventions). Read the root gate (g-12 vs the 0.78 express bar — a
READ, not a stop; the co-reads g0/CE_R/held30/site read/A129 verbatim from
the prior takes' root-dial conventions).

THE COMPARISON: the fresh draw's root g-12 vs the original draw's
0.7676599621772766 (runs/g1bS5/metrics.json, cross-checked at runtime) —
is the peak's height draw-robust?

=============== THE REGISTERED BARS (frozen verbatim; no shopping) ==========
  PEAK-ROBUST: "fires if the fresh-draw root lands within +-0.05 of
      0.7677 — the peak height is draw-robust; the curve's shape licensed
      at the peak; the wall's root (g1bS6) stands."
  PEAK-LOTTERY: "fires if the fresh draw lands > 0.10 from the original —
      the formation peak is itself a draw lottery (the g2e lesson at the
      consolidation level); the curve is one biography; reported honestly."
  GRADED: "any partial — the value verbatim."

OPERATIONALIZATION (registered here BEFORE compute — the frozen words made
computable; nothing re-tuned after seeing data):
  orig_gm12 := 0.7676599621772766   (frozen from runs/g1bS5/metrics.json
                                      provenance.consolidations.m020.root_
                                      read.gm12; cross-checked warn-only)
  d         := |fresh_gm12 - orig_gm12|
  PEAK-ROBUST fires iff d <= 0.05.
  PEAK-LOTTERY fires iff d > 0.10.
  GRADED fires otherwise (0.05 < d <= 0.10) — "any partial".
  Priority: PEAK-ROBUST -> PEAK-LOTTERY -> GRADED (the first two are
  mutually exclusive by construction: d <= 0.05 and d > 0.10 cannot both
  hold; the middle band is GRADED's).
  CO-REPORTED TEXTURE (never bars): (a) the fresh root vs the 0.78 express
  bar; (b) the fresh root vs the SHARP-OPTIMUM threshold 0.6998
  (= g1bS3's 0.6498 + 0.05 — would the redraw ALONE still have fired
  g1bS5's SHARP-OPTIMUM?); (c) the fresh root vs the 0.25 reading 0.7431
  (the fine ordering under a redraw); (d) the same-recipe replay fuzz floor
  (mean |dg0| 0.0218 / max 0.111 — a DIFFERENT currency from the seed
  redraw: fuzz bounds same-seed device nondeterminism, the redraw measures
  seed variance; both honestly stamped).
  PREDICTION: NONE registered — the redraw is open by design (the peak
  could be robust, a lottery, or partial); the only mechanical expectation
  is that the fresh seed GENUINELY diverges from the original stream
  (checked at runtime: the s25 in-run read must leave float-fuzz distance
  AND the final root state must NOT be bit-identical to g1bS5's saved
  m020 root — a real gate: a bit-identical "redraw" means the seed change
  failed and NOTHING may be adjudicated from it).

THE MOVEMENT ARITHMETIC (documented BEFORE compute; verbatim from g1bS5 —
the licensed currency): steps = movement / lr at the licensed rate 4e-4
(the G1BS3 width-scaled license, unchanged): 0.20 rms / 4e-4 = 500 steps.
WIDTH-PROPORTIONAL: 4e-4 = 1e-3 x 128/320 (the house recipe was minted at
n_embd 128; this host is 320). RAW L2 (co-reported for honesty, NOT the
licensed currency): 4e-4*sqrt(9.98e6) = 1.264 raw/step; 500 steps = 632
raw total. THE JITTER SEED IS THE TREATMENT: 10903 (fresh) vs 10901 (the
original draw) — the ONLY delta vs g1bS5's m020 run; steps 1..500 do NOT
replay the original trajectory (they redraw the jitter/batch stream over
the SAME pools and the SAME net0).

==================== THE OWNER ENVELOPE (STATE.json compute_directive) =====
The tightest constraint (2026-10-02, permanent): the lab is the LOWEST
compute priority. (1) GPU util AND temperature checked BEFORE every launch
(nvidia-smi); launch only when util <= 20% AND temp <= 70C (double-poll,
5 s apart; plus mem <= 60% as resident-neighbor caution) — every poll
appended to runs/_envelope_log.jsonl (the R61-critic audit trail; g1bS6's
adopted trim); (2) SHORT bursts <= 90 s GPU (TRAIN_CAP_GPU = 90 — the
consolidation splits at the cap with full-state resume ckpts); (3) cooldown
>= 180 s between bursts; (4) NO back-to-back; (5) when in doubt WAIT (the
gate parks in 30 s polls; after GPU_WAIT_MAX it STOPS honestly — partial
metrics + resume ckpts hold the record; NEVER a silent CPU hop). Outside
load mid-burst => PAUSE-AND-WAIT (the g1bW policy), never migrate.
torch threads 4 (shared machine).

==================== LINEAGE (the standing headers, abridged) ===============
G1BS6 — the wall arms on the peak root (T170): WALL-FADES at 10M — every
rung breached at +1 (the first-step blindness structural); the direction
survives (1000x separation); RAN ON THE 0.7677 ROOT — this cell asks
whether that root's HEIGHT is draw-robust.
G1BS5 — the formation curve (T167): SHARP-OPTIMUM; peak 0.7677 @ 0.20 rms;
the honesty ledger: n=1 draw, the fine ordering within fuzz, the redraw
owed (THIS cell).
G1BS4 — the movement-matched dose (T161): 0.30 rms -> 0.2523 (the
inversion that demanded the curve).
G1BS3 — the width-scaled e113 license (T159): 4e-4; 0.12 rms -> 0.6498.
G1BS2 — the val-min-anchored base (T148's cure as a pattern): base PASS,
install PASS, the 1e-3 casualty (kept separate).
Builds on: g1/g1b/g1bR (the commit-and-project wall), e113 (the jitter
consolidation form), g2e (the root replicate lesson — the amplitude
lottery), T159/T161/T165/T167/T170.
What is NEW: the first REDRAW of any consolidation point in the lineage —
the formation curve's peak re-drawn under a fresh jitter seed; a direct
measure of the consolidation-level draw lottery (the g2e lesson's analog
at the consolidation axis); consolidation-only (no wall claims possible).

CHECKS (Rule 12): the loaded checkpoints' identity gates re-derived ON THE
LOAD (G-BASE cosine-complete + G-BASE-QUAL legs; G-INST step + the
post-install reuse check vs g1bS2's committed record |d gm12| <= 0.02);
the movement arithmetic + the fresh-seed registration asserted BEFORE
compute (mv/lr == steps; 10903 != 10901); the redraw's GENUINE divergence
(the s25 float-fuzz distance + the final root NOT bit-identical to
g1bS5_root_m020.pt — a real gate); the original draw's committed numbers
cross-checked at runtime against runs/g1bS5/metrics.json (warn-only).
NOTHING guaranteed — the openness is the point.

Outputs: runs/g1bS7/{metrics.json, redraw.png}; checkpoint
runs/checkpoints/g1bS7_root_m020f.pt (all prior takes' artifacts untouched;
the base+install are g1bS2's and the original m020 root is g1bS5's, LOADED
read-only). No NOTES/THINKING/QUEUE/STATE edits (the coordinator folds).
Progressive metrics writes after every phase (the outage lesson); commit +
push per phase.

Run:  cd lab && python g1bS7_redraw.py    (G1BS7_SMOKE=1 shakedown)
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

SMOKE = os.environ.get("G1BS7_SMOKE") == "1"
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

# ======================================================================
# THE CELL: ONE fresh-jitter consolidation at the 0.20 rms peak dose
# ======================================================================
ORIG_SEED = 10901                 # g1bS5's m020 draw (the compared original)
FRESH_SEED = 10903                # THE TREATMENT: the fresh jitter seed,
                                 # registered in the dispatch BEFORE compute
assert FRESH_SEED != ORIG_SEED, "the redraw must not reuse the original seed"
G1.CONS_SEED = FRESH_SEED         # THE ONLY DELTA vs g1bS5's m020 run: every
                                 # consumer of the consolidation seed (the
                                 # driver's generator, the provenance stamps)
                                 # now carries 10903; the driver itself stays
                                 # VERBATIM (its manual_seed(G1.CONS_SEED)
                                 # line untouched)

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)
G1.log = log                                          # unify the timeline

G1BS_CFG = Cfg(vocab=65, n_layer=8, n_head=8, n_embd=320, block_size=256)
G1BS_PARAMS = 9_977_600          # e005s LARGE's verified 10M-class config
HOST_SEED = 1337                 # the "1337 family" (host init + base corpus)
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
SRC_BASE_CK, SRC_INSTALL_CK = "g1bS2_base.pt", "g1bS2_install.pt"
BASE_STEPS = 1200 if not SMOKE else 24
INSTALL_STEPS = 1000 if not SMOKE else 8
CONS_LR = 4e-4                    # the G1BS3 licensed rate, UNCHANGED

MV, STEPS, TAG = 0.20, (500 if not SMOKE else 8), "m020f"
if not SMOKE:                     # the registered arithmetic + seed,
    assert abs(MV / CONS_LR - STEPS) < 1e-9, "dose arithmetic broken: redraw"
CONS_MOVEMENT = {
    "licensed_currency": "ADAM-CLOCK rms (T139): steps = movement / lr",
    "lr": CONS_LR,
    "dose": {"tag": TAG, "movement_rms": MV, "steps": STEPS,
             "check": f"{MV} / {CONS_LR} = {MV / CONS_LR:.0f}"},
    "seed_registration": {"original_draw": ORIG_SEED, "fresh_draw": FRESH_SEED,
                          "note": "the jitter seed is THE TREATMENT — the "
                                  "only delta vs g1bS5's m020 run"},
    "width_proportional": "4e-4 = 1e-3 x 128/320 (n_embd 128 -> 320)",
    "raw_per_step": CONS_LR * math.sqrt(G1BS_PARAMS),
    "raw_total": STEPS * CONS_LR * math.sqrt(G1BS_PARAMS),
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
GPU_WAIT_MAX = float(os.environ.get("G1BS7_GPU_WAIT_MAX", "7200"))
                                 # when in doubt WAIT; after 2 h of no
                                 # windows STOP honestly (resumable) —
                                 # never a silent CPU hop

# ---- the frozen priors (the original draw + the committed curve) ----------
G1BS5_M020 = {                   # runs/g1bS5/metrics.json (committed)
    "movement_rms": 0.20, "steps": 500, "rate": CONS_LR, "seed": ORIG_SEED,
    "gm12": 0.7676599621772766, "g0": 0.7367021441459656,
    "gp12": 0.8544988036155701,
    "held30_gm12": 0.5573378801345825, "held30_g0": 0.6574546694755544,
    "ce_r": 1.6968417167663574,
    "site_read_onset": 0.6970594525337219, "site_read_span": 0.9553064108894897,
    "A129": -0.07295363429002466, "row0_strength": 0.5237537777129889,
    "source": ("runs/g1bS5/metrics.json "
               "provenance.consolidations.m020.root_cells (the original "
               "draw — the comparison anchor)"),
}
G1BS5_CURVE = {                  # the committed curve (frozen for the plot)
    "x_movement_rms": [0.12, 0.15, 0.20, 0.25, 0.30],
    "x_steps": [300, 375, 500, 625, 750],
    "gm12": [0.6498003602027893, 0.7284520268440247, 0.7676599621772766,
             0.7430918216705322, 0.2523],
    "committed": [True, False, True, False, True],
    "source": ("runs/g1bS5/metrics.json curve (the m015/m020/m025 points "
               "are g1bS5's fresh cells; 0.12 = g1bS3, 0.30 = g1bS4)"),
}
FUZZ_FLOOR = {                   # the SAME-RECIPE (same-seed) replay fuzz —
    "mean_d_g0": 0.0218, "max_d_g0": 0.111,   # a DIFFERENT currency from the
    "source": ("g1bS4's committed replay overlap (same recipe, same seed "
               "10901); bounds device float fuzz, NOT seed variance — the "
               "redraw measures the seed axis; both stamped honestly"),
}
SHARP_THRESHOLD = 0.6498003602027893 + 0.05    # g1bS5's registered 0.6998

REGISTERED = {
    "bars_verbatim": {
        "PEAK-ROBUST": ("fires if the fresh-draw root lands within +-0.05 "
                        "of 0.7677 — the peak height is draw-robust; the "
                        "curve's shape licensed at the peak; the wall's "
                        "root (g1bS6) stands."),
        "PEAK-LOTTERY": ("fires if the fresh draw lands > 0.10 from the "
                         "original — the formation peak is itself a draw "
                         "lottery (the g2e lesson at the consolidation "
                         "level); the curve is one biography; reported "
                         "honestly."),
        "GRADED": "any partial — the value verbatim.",
    },
    "operationalization": {
        "registered_before_compute": True,
        "orig_gm12": G1BS5_M020["gm12"],
        "d_definition": "d := |fresh_gm12 - orig_gm12|",
        "PEAK-ROBUST_fires_if": "d <= 0.05",
        "PEAK-LOTTERY_fires_if": "d > 0.10",
        "GRADED_fires_if": "otherwise (0.05 < d <= 0.10) — any partial",
        "priority": ("PEAK-ROBUST -> PEAK-LOTTERY -> GRADED (the first two "
                     "mutually exclusive by construction; the middle band "
                     "is GRADED's)"),
        "texture_co_reported_never_bars": [
            "fresh vs the 0.78 express bar (the root gate READ)",
            "fresh vs the SHARP-OPTIMUM threshold 0.6998 (would the redraw "
            "ALONE still have fired g1bS5's SHARP-OPTIMUM?)",
            "fresh vs the 0.25 reading 0.7431 (the fine ordering redrawn)",
            "the same-recipe replay fuzz floor (mean |dg0| 0.0218 / max "
            "0.111 — same-seed device fuzz; a different currency from the "
            "seed redraw)",
        ],
        "prediction": ("NONE registered — the redraw is open by design; the "
                       "only mechanical expectation is GENUINE divergence "
                       "from the original stream (a real runtime gate: "
                       "s25 float-fuzz distance + the final root NOT "
                       "bit-identical to g1bS5_root_m020.pt)"),
    },
    "movement_arithmetic": CONS_MOVEMENT,
    "root_gate": {
        "bar": G1.EXPRESS_BAR,
        "reading": ("the fresh root g-12 vs the 0.78 express bar — a READ, "
                    "not a hard stop: there are no arms to stop in this "
                    "cell; the redraw bars are the adjudication"),
        "co_reads": ("g0 / gp12 / held30 / CE_R / site read / A129 — the "
                     "prior takes' root-dial conventions verbatim (measure() "
                     "full dial, CPU)"),
    },
    "sources_loaded": {
        "base": ("runs/checkpoints/g1bS2_base.pt — g1bS2's PASSED base "
                 "(val-min-anchored 1200-step cosine, lr 4e-4, seed 1337); "
                 "G-BASE-QUAL PASS on record at runs/g1bS2/metrics.json, "
                 "re-derived ON THE LOAD by this run (takes 3-7's standing "
                 "license: never retrained)"),
        "install": ("runs/checkpoints/g1bS2_install.pt — the "
                    "movement-matched s1000 @ 4e-4 install (seed 42); "
                    "post-install state REUSABLE (g1bS2's failure was IN the "
                    "consolidation, strictly downstream; re-verified by "
                    "takes 3-6 and re-checked here)"),
        "original_root": ("runs/checkpoints/g1bS5_root_m020.pt — g1bS5's "
                          "saved m020 root (seed 10901), LOADED READ-ONLY "
                          "for the bit-divergence check (the redraw must "
                          "NOT bit-replay it); never a training source"),
    },
    "owner_envelope": {
        "launch_gate": f"util <= {OWNER_UTIL_C:.0f}% AND temp <= "
                       f"{OWNER_TEMP_C:.0f}C (double-poll 5 s) AND mem <= "
                       f"{OWNER_MEM_FRAC:.0%}",
        "burst_cap_s": TRAIN_CAP_GPU, "cooldown_s": COOLDOWN_S,
        "poll_audit": ("every owner-gate poll appended to "
                       "runs/_envelope_log.jsonl (the R61-critic trail; "
                       "g1bS6's adopted trim)"),
        "no_cpu_hop": ("the consolidation NEVER runs on CPU (a 10M CPU "
                       "consolidation is not a viable burst; if no CUDA or "
                       "no window: STOP honestly, partial metrics + resume "
                       "ckpt hold the record)"),
    },
    "checks_rule12": [
        "the loaded checkpoints' identity gates (G-BASE cosine-complete + "
        "G-BASE-QUAL legs re-derived on the load; G-INST step + the "
        "post-install reuse check |d gm12| <= 0.02 vs g1bS2's record)",
        "the movement arithmetic + the fresh-seed registration asserted "
        "BEFORE compute (0.20/4e-4 == 500; 10903 != 10901)",
        "THE REDREW CHECK (a real gate): the s25 in-run read must leave "
        "float-fuzz distance from the original draw's s25 AND the final "
        "root state must NOT be bit-identical to g1bS5_root_m020.pt",
        "the original draw's committed numbers cross-checked at runtime "
        "against runs/g1bS5/metrics.json (warn-only)",
        "the in-run trajectory deltas vs the original m020 traj — the "
        "DRAW-SENSITIVITY read (texture, never a gate)",
    ],
}

deviations: list[str] = [
    "CONSOLIDATION-ONLY REDRAW CELL (the dispatch's license): no install "
    "training (g1bS2's install is LOADED read-only), no wall arms, no wash "
    "— the cheapest possible form of T167's owed redraw; no wall claims "
    "possible from this cell (g1bS6's WALL-FADES verdict is CONTEXT, "
    "not re-adjudicated here).",
    "THE FRESH SEED IS THE TREATMENT: consolidation seed 10903 (registered) "
    "vs the original draw's 10901 — the ONLY delta vs g1bS5's m020 run; "
    "implemented by overriding G1.CONS_SEED before any compute so the "
    "verbatim driver (manual_seed(G1.CONS_SEED)) carries the fresh seed "
    "with its arithmetic untouched.",
    "THE OWNER ENVELOPE (STATE.json compute_directive 2026-10-02, permanent) "
    "verbatim from g1bS5/g1bS6: launch gate util <= 20% AND temp <= 70C "
    "(double-poll; mem <= 60%), burst cap 90 s, cooldown 180 s, no CPU "
    "fallback, plus g1bS6's adopted poll-audit trim (every owner-gate poll "
    "appended to runs/_envelope_log.jsonl).",
    "The registered g-12 channel is read ONLY at the endpoint (the prior "
    "takes' root-dial convention — full measure() dial, CPU ~4-5 min); the "
    "in-run every-25-step evals stay on the g0/CE_R light channels (the "
    "consolidation driver's own convention, verbatim).",
    "g1bS5's committed m020 ROOT checkpoint is LOADED (read-only) purely "
    "for the bit-divergence check — the honest machinery gate that the "
    "redraw actually redrew; it never enters training or the readout.",
    "torch threads 4 (shared machine; g1bW's adopted trim; g1's import "
    "resets to 8, reset after import).",
    "Smoke mode trims: base 24 steps, install 8 steps (g1bS2's smoke ckpts), "
    "dose 8 steps, lean dials, no cooldowns, gate waits capped at 30 s, the "
    "bit-divergence gate vs the FULL m020 root SKIPPED (a smoke redraw of 8 "
    "steps trivially differs from a 500-step root; the divergence check "
    "falls to the s25/s8 float-fuzz distance read) — nothing adjudicated.",
]

device_events: list[dict] = []
trims: list[str] = []

_progressive = {"n": 0, "phases": []}


class OwnerWindowShut(Exception):
    """No owner-envelope GPU window opened within GPU_WAIT_MAX — the run
    stops honestly (partial metrics + resume ckpt hold the record)."""


def write_partial(rd: Path, phase: str, payload: dict) -> None:
    """The outage lesson, mechanized: metrics.json exists from the first
    phase onward and is rewritten after every phase. Superseded by the
    final full write (partial=false). Bookkeeping only — never allowed to
    kill compute."""
    _progressive["n"] += 1
    _progressive["phases"].append(phase)
    try:
        out = {
            "experiment": "g1bS7_redraw",
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
# Every poll is appended to runs/_envelope_log.jsonl (the R61-critic audit
# trail — common's _log_envelope_poll, g1bS6's adopted trim).

def owner_gpu_ok(tag: str = "owner") -> bool:
    s = gpu_status()
    ok = bool(s["util"] <= OWNER_UTIL_C and s["temp"] <= OWNER_TEMP_C
              and (s["mem_total"] == 0
                   or s["mem_used"] <= OWNER_MEM_FRAC * s["mem_total"]))
    try:
        common._log_envelope_poll(f"g1bS7:{tag}", s["util"], s["temp"], ok)
    except Exception:                                             # noqa: BLE001
        pass
    return ok


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
            f"CPU form under the owner envelope (stop honestly; the resume "
            f"ckpt holds the record)")
    t0, waited = time.time(), 0.0
    while (time.time() - t0) <= mw:
        if owner_gpu_ok(tag):
            time.sleep(5)                      # the double-poll
            if owner_gpu_ok(tag):
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
        f"(resume ckpt + partial metrics hold the record; re-launch "
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
    """The g4 coherence dial (house stat; registered BEFORE use — g1bS4/S5's
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
    rd = run_dir("g1bS7_smoke" if SMOKE else "g1bS7")
    log(f"G1BS7 THE REDRAWN INTERIOR DOSE (one fresh-jitter consolidation: "
        f"{MV} rms / {STEPS} steps @ lr {CONS_LR}, seed {FRESH_SEED} (the "
        f"original draw's seed was {ORIG_SEED}) from the SAME loaded "
        f"base+install; owner envelope: bursts <= {TRAIN_CAP_GPU:.0f}s, "
        f"cooldown {COOLDOWN_S:.0f}s, gate util<={OWNER_UTIL_C:.0f}% "
        f"temp<={OWNER_TEMP_C:.0f}C, every poll audited; smoke={SMOKE}) "
        f"-> {rd}")
    write_partial(rd, "start", {
        "design": ("T167's honesty-ledger debt (the dispatch, 2026-10-02): "
                   "the formation curve's fine-ordering leg — one fresh-"
                   "jitter consolidation at the 0.20 rms peak; bars frozen "
                   "verbatim; no bar shopping"),
        "question": ("is the formation peak's HEIGHT draw-robust? — the "
                     "fresh draw's root g-12 vs the original draw's "
                     f"{G1BS5_M020['gm12']:.4f} (g1bS5's m020, seed "
                     f"{ORIG_SEED}); the fresh seed {FRESH_SEED} is the "
                     "ONLY delta"),
        "registered": REGISTERED, "deviations": deviations, "smoke": SMOKE,
        "priors": {"g1bs5_m020_original": G1BS5_M020,
                   "g1bs5_curve": G1BS5_CURVE, "fuzz_floor": FUZZ_FLOOR},
        "device_events": device_events})
    set_seed(HOST_SEED)      # host init + base corpus (the 1337 family)

    # ---- patch g1's machinery to the 10M family (g1b's pattern) ---------
    G1.G1_CFG = G1BS_CFG
    G1.G1_PARAMS = G1BS_PARAMS

    # ---------------- protocol rebuild (g1bS5's main VERBATIM) -------------
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
    # lineage license (takes 3-7): never retrain; identity gates ON THE LOAD)
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
            f"G1BS7 source base missing: {base_ck} (the licensed cell LOADS "
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
                 "cosine, lr 4e-4, seed 1337) — LOADED read-only by g1bS7 "
                 "(the license: never retrained)"),
        "step": bstep, "final_val_loss": hist[-1]["val_loss"]}
    base_cells = flat_cells(measure(theta_base, "g1bS7_base", lean=True))
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
            f"G1BS7 source install missing: {inst_ck} (the licensed cell "
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
                      "this redraw; g1bS2's G-ROOT failure was caused by "
                      "its lr-1e-3 consolidation itself, strictly "
                      "downstream of this state; smoke skips the check "
                      "(machinery only)"),
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
                 "seed 42) — LOADED read-only by g1bS7; the licensed "
                 "redraw source (the SAME start as the original draw)"),
        "step": istep}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =====================================================================
    # THE ORIGINAL-DRAW CROSS-CHECK (warn-only; the frozen comparison anchor
    # vs the committed file — Rule 12 provenance)
    # =====================================================================
    priors_check = {"g1bs5_m020": None}
    orig_traj = None
    _s5 = E43.REPO / "runs" / "g1bS5" / "metrics.json"
    try:
        _d = json.loads(_s5.read_text(encoding="utf-8"))
        _row = _d["provenance"]["consolidations"]["m020"]
        _d_gm = abs(_row["root_cells"]["gm12"] - G1BS5_M020["gm12"])
        _d_ce = abs(_row["root_cells"]["ce_r"] - G1BS5_M020["ce_r"])
        _curve_ok = all(
            abs(a - b) <= 1e-9 for a, b in zip(
                _d["curve"]["gm12"], G1BS5_CURVE["gm12"]))
        priors_check["g1bs5_m020"] = {
            "file": "runs/g1bS5/metrics.json",
            "file_gm12": _row["root_cells"]["gm12"],
            "frozen_gm12": G1BS5_M020["gm12"],
            "abs_d_gm12": _d_gm, "abs_d_ce_r": _d_ce,
            "curve_gm12_match": bool(_curve_ok),
            "match": bool(_d_gm <= 1e-9 and _d_ce <= 5e-3),
        }
        orig_traj = list(_row["traj"])
        log(f"[priors] g1bS5 m020: committed gm12 "
            f"{_row['root_cells']['gm12']:.4f} vs frozen "
            f"{G1BS5_M020['gm12']:.4f} (|d| {_d_gm:.1e}); curve match "
            f"{_curve_ok} — {'match' if _d_gm <= 1e-9 else 'MISMATCH (warn)'}")
    except Exception as e:                                         # noqa: BLE001
        priors_check["g1bs5_m020"] = {"error": str(e)}
        log(f"[priors] g1bS5 m020 unreadable ({e}) — continuing (the frozen "
            f"anchor stands; the side-by-side traj leg is skipped; recorded)")

    # =====================================================================
    # THE ONE CONSOLIDATION — THE REDRAW (e113 jitter convention, licensed
    # lr 4e-4, FRESH SEED 10903 — the registered delta; the dose VERBATIM:
    # 0.20 rms = 500 steps, the SAME loaded post-install state)
    # =====================================================================
    log("=" * 78)
    log(f"THE REDRAW: {STEPS} steps @ lr {CONS_LR} = {MV:.2f} rms total "
        f"movement ({MV} / {CONS_LR} = {MV / CONS_LR:.0f} — the registered "
        f"arithmetic), jitters {G1.JITTERS}, batch 16 install + 16 anchor, "
        f"name-masked union CE, AdamW (0.9,0.95) wd 0.1 clip 1.0, seed "
        f"{G1.CONS_SEED} (FRESH — the original draw's was {ORIG_SEED}; the "
        f"seed is THE TREATMENT, everything else VERBATIM vs g1bS5's m020) "
        f"— from the SAME loaded post-install state")

    def _chunk_partial(summary: dict) -> None:
        write_partial(rd, f"cons-{TAG}-chunk{summary['chunk']}",
                      {"redraw_partial": {
                          "consolidation_partial": summary}})

    try:
        cons = g1bS7_consolidate(
            G1.evl_load(theta_install), pool_a_x, pool_a_mask,
            cons_anchor, train_ids, g0_ids, r_eval_xy, zid, TAG,
            steps=STEPS, lr=CONS_LR,
            resume_ck=(CKPT_DIR /
                       (f"smoke_g1bS7_cons_{TAG}_resume.pt" if SMOKE
                        else f"g1bS7_cons_{TAG}_resume.pt")),
            on_chunk=_chunk_partial)
    except OwnerWindowShut as e:
        write_partial(rd, f"owner-window-shut-{TAG}",
                      {"stopped_at": TAG, "reason": str(e),
                       "device_events": device_events})
        log(f"OWNER WINDOW SHUT at the redraw: {e} — stopping honestly "
            f"(partial metrics + the resume ckpt hold the record)")
        raise SystemExit(3)
    theta_root = cons["sd"]
    if cons["steps_ran"] < STEPS:
        trims.append(f"consolidate {TAG} capped at step "
                     f"{cons['steps_ran']} of {STEPS} (documented, not "
                     f"silent — resume ckpt exhausted; re-launch "
                     f"continues it)")

    # ---- THE REDREW CHECKS (Rule 12: the redraw must genuinely redraw) ---
    # (1) the s25 float-fuzz distance vs the original draw's own s25 read
    fresh_traj = {t["step"]: t for t in cons["traj"]}
    first_eval_step = 25 if not SMOKE else STEPS
    redrew_stream = None
    if orig_traj is not None:
        orig_traj_d = {t["step"]: t for t in orig_traj}
        if first_eval_step in fresh_traj and first_eval_step in orig_traj_d:
            dg0_first = abs(fresh_traj[first_eval_step]["install60_g0_pz"]
                            - orig_traj_d[first_eval_step]["install60_g0_pz"])
            redrew_stream = {
                "step": first_eval_step,
                "fresh_g0": fresh_traj[first_eval_step]["install60_g0_pz"],
                "orig_g0": orig_traj_d[first_eval_step]["install60_g0_pz"],
                "abs_d_g0": dg0_first,
                "float_fuzz_scale": FUZZ_FLOOR["mean_d_g0"],
                "verdict": ("GENUINELY DIVERGED" if dg0_first >
                            10 * FUZZ_FLOOR["mean_d_g0"] else
                            "SUSPICIOUSLY CLOSE (float-fuzz scale — the "
                            "seed change may not have taken; co-reported, "
                            "the bit check decides)"),
            }
            log(f"[redrew] stream check @s{first_eval_step}: fresh "
                f"{fresh_traj[first_eval_step]['install60_g0_pz']:.4f} vs "
                f"orig {orig_traj_d[first_eval_step]['install60_g0_pz']:.4f} "
                f"(|d| {dg0_first:.4f} vs fuzz mean "
                f"{FUZZ_FLOOR['mean_d_g0']}) — {redrew_stream['verdict']}")
    # (2) the draw-sensitivity read: in-run |dg0| fresh-vs-orig per shared
    #     eval step (texture, never a gate — the seed axis made visible)
    draw_sensitivity = None
    if orig_traj is not None and not SMOKE:
        orig_traj_d = {t["step"]: t for t in orig_traj}
        rows = []
        for s, e in sorted(fresh_traj.items()):
            if s in orig_traj_d:
                rows.append({"step": s,
                             "g0_fresh": e["install60_g0_pz"],
                             "g0_orig": orig_traj_d[s]["install60_g0_pz"],
                             "ce_fresh": e["ce_r"],
                             "ce_orig": orig_traj_d[s]["ce_r"]})
        draw_sensitivity = {
            "n_shared": len(rows),
            "mean_d_g0": (sum(abs(r["g0_fresh"] - r["g0_orig"])
                              for r in rows) / len(rows)) if rows else None,
            "max_d_g0": max((abs(r["g0_fresh"] - r["g0_orig"])
                             for r in rows), default=None),
            "reading": ("the SEED axis: same recipe/pools/net0/lr/dose, "
                        "different jitter seed — the in-run g0 deltas vs "
                        "the original draw; compare against the same-seed "
                        "replay fuzz floor (mean "
                        f"{FUZZ_FLOOR['mean_d_g0']} / max "
                        f"{FUZZ_FLOOR['max_d_g0']}) — a different currency, "
                        "both stamped"),
        }
        if rows:
            log(f"[draw] fresh-vs-orig over {len(rows)} shared evals: "
                f"mean|dg0| {draw_sensitivity['mean_d_g0']:.4f} max "
                f"{draw_sensitivity['max_d_g0']:.4f} (the seed axis; "
                f"same-seed replay fuzz floor is mean "
                f"{FUZZ_FLOOR['mean_d_g0']} / max "
                f"{FUZZ_FLOOR['max_d_g0']})")
    # (3) THE BIT-DIVERGENCE GATE vs g1bS5's saved m020 root (full run only)
    redrew_bits = None
    if not SMOKE:
        orig_root_ck = CKPT_DIR / "g1bS5_root_m020.pt"
        if not orig_root_ck.exists():
            raise SystemExit(
                f"THE REDREW CHECK cannot run: {orig_root_ck} missing (the "
                f"bit-divergence gate vs the original draw's saved root is "
                f"REQUIRED before any adjudication)")
        _or = torch.load(orig_root_ck, map_location="cpu",
                         weights_only=False)
        _osd = _or["model"]
        del _or
        assert set(_osd.keys()) == set(theta_root.keys()), \
            "root state keys differ from g1bS5's m020 — architecture drift"
        max_abs, n_diff, n_tot = 0.0, 0, 0
        for k in sorted(_osd.keys()):
            a, b = _osd[k].float(), theta_root[k].float()
            d = (a - b).abs()
            max_abs = max(max_abs, float(d.max()))
            n_diff += int((d > 0).sum())
            n_tot += int(d.numel())
        redrew_bits = {
            "compared_to": "runs/checkpoints/g1bS5_root_m020.pt",
            "max_abs_diff": max_abs,
            "frac_elements_differing": n_diff / max(n_tot, 1),
            "gate": "the final root state must NOT be bit-identical "
                    "(max_abs_diff > 1e-6): a bit-identical 'redraw' means "
                    "the seed change failed",
            "pass": bool(max_abs > 1e-6),
        }
        _pct = 100 * redrew_bits["frac_elements_differing"]
        log(f"[redrew] bit check vs g1bS5_root_m020.pt: max|diff| "
            f"{max_abs:.4f}, {_pct:.1f}% of elements differ — "
            f"{'PASS (genuinely redrawn)' if redrew_bits['pass'] else 'FAIL'}")
        assert redrew_bits["pass"], ("THE REDREW CHECK FAILED: the fresh "
                                     "seed produced a bit-identical root — "
                                     "the seed change did not take; NOTHING "
                                     "may be adjudicated")
        write_partial(rd, "redrew-checks", {
            "redrew_stream": redrew_stream, "redrew_bits": redrew_bits,
            "draw_sensitivity": draw_sensitivity})

    # ---- the ROOT DIAL (the prior takes' convention: full measure) -------
    root_m = measure(theta_root, f"g1bS7_{TAG}", lean=SMOKE)
    root_cells = flat_cells(root_m)
    root_read = {"bar": G1.EXPRESS_BAR, "gm12": root_cells["gm12"],
                 "pass": bool(root_cells["gm12"] >= G1.EXPRESS_BAR),
                 "note": ("a READ, not a hard stop — no arms to stop in "
                          "this cell; the redraw bars adjudicate")}
    log(f"ROOT {TAG} (fresh draw): g-12 {root_cells['gm12']:.4f} (bar >= "
        f"{G1.EXPRESS_BAR}) g0 {root_cells['g0']:.4f} CE_R "
        f"{root_cells['ce_r']:.4f} held30_gm12 "
        f"{root_cells['held30_gm12']:.4f} | "
        f"{'PASS' if root_read['pass'] else 'below bar'}")
    write_partial(rd, f"root-{TAG}", {
        "redraw": {
            "movement_rms": MV, "steps": STEPS, "lr": CONS_LR,
            "seed": cons["seed"],
            "consolidation": {
                "steps_ran": cons["steps_ran"], "seed": cons["seed"],
                "device": cons["device"], "n_chunks": cons["n_chunks"],
                "chunk_table": cons["chunk_table"], "traj": cons["traj"]},
            "redrew_stream": redrew_stream, "redrew_bits": redrew_bits,
            "draw_sensitivity": draw_sensitivity,
            "root_read": root_read, "root_cells": root_cells}})
    save_ckpt(f"g1bS7_root_{TAG}", theta_root,
              {"desc": "g1bS2's PASSED base+install (LOADED read-only) + "
                       f"e113 jitter consolidation s{cons['steps_ran']} @ lr "
                       f"{CONS_LR} (seed {FRESH_SEED}, FRESH — the original "
                       f"draw's was {ORIG_SEED}) = movement {MV} rms — the "
                       "REDRAWN peak point (g1bS7, the fine-ordering leg)",
               "params": n_params, "movement_rms": MV,
               "steps": cons["steps_ran"], "lr": CONS_LR,
               "cons_seed": FRESH_SEED,
               "root_gm12": root_cells["gm12"],
               "base": f"runs/checkpoints/{SRC_BASE_CK}",
               "install": f"runs/checkpoints/{SRC_INSTALL_CK}"})
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =====================================================================
    # THE ADJUDICATION (frozen bars; operationalization registered BEFORE
    # compute in REGISTERED.operationalization)
    # =====================================================================
    fresh_gm12 = root_cells["gm12"]
    orig_gm12 = G1BS5_M020["gm12"]
    d = abs(fresh_gm12 - orig_gm12)
    signed_d = fresh_gm12 - orig_gm12
    if d <= 0.05:
        fired = "PEAK-ROBUST"
        clause = (f"the fresh draw's root lands within +-0.05 of the "
                  f"original: fresh {fresh_gm12:.4f} vs original "
                  f"{orig_gm12:.4f} (|d| {d:.4f} <= 0.05) — the peak HEIGHT "
                  f"is draw-robust; the curve's shape licensed at the peak; "
                  f"the wall's root (g1bS6's WALL-FADES on the 0.7677 "
                  f"root) stands."
                  + (f" NOTE: the fresh draw's read sits "
                     f"{'ABOVE' if signed_d > 0 else 'BELOW'} the original "
                     f"by {abs(signed_d):.4f} — inside the robust band."
                     if d > FUZZ_FLOOR["max_d_g0"] else
                     " (the delta is itself within the same-seed replay "
                     "max-fuzz 0.111 — even the fuzz floor alone could not "
                     "have produced a bar flip; co-reported honestly).")
                  + f" Co-reads: express bar {G1.EXPRESS_BAR} "
                  f"({'PASS' if root_read['pass'] else 'below'}), SHARP-"
                  f"OPTIMUM threshold {SHARP_THRESHOLD:.4f} "
                  f"({'still fires on the redraw' if fresh_gm12 >= SHARP_THRESHOLD else 'NOT met on the redraw'}), "
                  f"the 0.25 reading 0.7431 "
                  f"({'the peak still outranks 0.25' if fresh_gm12 >= 0.7430918216705322 else 'the redraw FALLS BELOW the 0.25 reading — the fine ordering flips'}). "
                  f"Honesty: n=1 redraw (two draws total at the peak); the "
                  f"seed-axis in-run |dg0| texture is co-reported.")
    elif d > 0.10:
        fired = "PEAK-LOTTERY"
        clause = (f"the fresh draw lands > 0.10 from the original: fresh "
                  f"{fresh_gm12:.4f} vs original {orig_gm12:.4f} (|d| "
                  f"{d:.4f} > 0.10) — the formation peak is itself a draw "
                  f"lottery (the g2e lesson at the consolidation level: "
                  f"one-seed readings of a formation optimum do not "
                  f"transfer); the curve is one biography; reported "
                  f"honestly. Co-reads: express bar "
                  f"{G1.EXPRESS_BAR} "
                  f"({'PASS' if root_read['pass'] else 'below'}), "
                  f"SHARP-OPTIMUM threshold {SHARP_THRESHOLD:.4f} "
                  f"({'still met on the redraw' if fresh_gm12 >= SHARP_THRESHOLD else 'NOT met on the redraw — even the interior-optimum finding weakens'}). "
                  f"Honesty: n=1 redraw; no bar shopping — the value "
                  f"verbatim.")
    else:
        fired = "GRADED"
        clause = (f"a partial — the fresh draw lands in the middle band: "
                  f"fresh {fresh_gm12:.4f} vs original {orig_gm12:.4f} "
                  f"(|d| {d:.4f}; 0.05 < |d| <= 0.10) — neither robust "
                  f"(<= 0.05) nor lottery (> 0.10); the value verbatim. "
                  f"Co-reads: express bar {G1.EXPRESS_BAR} "
                  f"({'PASS' if root_read['pass'] else 'below'}), "
                  f"SHARP-OPTIMUM threshold {SHARP_THRESHOLD:.4f} "
                  f"({'still fires on the redraw' if fresh_gm12 >= SHARP_THRESHOLD else 'NOT met on the redraw'}), "
                  f"the 0.25 reading 0.7431 "
                  f"({'the peak still outranks 0.25' if fresh_gm12 >= 0.7430918216705322 else 'the redraw FALLS BELOW the 0.25 reading — the fine ordering flips'}). "
                  f"Honesty: n=1 redraw (two draws total at the peak); a "
                  f"third draw would be owed before any fine-ordering "
                  f"claim.")
    texture = {
        "express_bar": {"bar": G1.EXPRESS_BAR, "fresh": fresh_gm12,
                        "read": "PASS" if root_read["pass"] else "below"},
        "sharp_threshold": {"threshold": SHARP_THRESHOLD, "fresh": fresh_gm12,
                            "would_refire": bool(fresh_gm12 >=
                                                 SHARP_THRESHOLD)},
        "fine_ordering": {"reading_0p25": 0.7430918216705322,
                          "fresh": fresh_gm12,
                          "peak_still_outranks_0p25": bool(
                              fresh_gm12 >= 0.7430918216705322)},
        "fuzz_context": {"same_seed_replay": FUZZ_FLOOR,
                         "note": ("the redraw's d is on the SEED axis; the "
                                  "fuzz floor bounds the same-seed device "
                                  "axis — different currencies, both "
                                  "stamped")},
    }
    log("=" * 78)
    log(f"THE REDRAW COMPARISON: fresh {fresh_gm12:.4f} vs original "
        f"{orig_gm12:.4f} (|d| {d:.4f})")
    log(f"G1BS7 VERDICT: {fired}")
    log(f"  {clause}")
    log("=" * 78)
    write_partial(rd, "adjudication", {
        "adjudication_partial": {
            "fired": fired, "clause": clause, "d": d,
            "fresh_gm12": fresh_gm12, "orig_gm12": orig_gm12,
            "texture": texture}})

    # =====================================================================
    # OUTPUTS
    # =====================================================================
    metrics = {
        "experiment": "g1bS7_redraw",
        "date": common.now_iso(),
        "partial": False,
        "progressive_writes": _progressive["n"],
        "phases": list(_progressive["phases"]),
        "design": ("T167's honesty-ledger debt (the dispatch, 2026-10-02): "
                   "the formation curve's fine-ordering leg — one fresh-"
                   "jitter consolidation at the 0.20 rms peak "
                   "(consolidation-only, no install, no wall arms, no wash); "
                   "bars frozen verbatim; no bar shopping"),
        "question": ("is the formation peak's height draw-robust? — the "
                     "fresh draw's (seed 10903) root g-12 vs the original "
                     "draw's (seed 10901) "
                     f"{orig_gm12:.4f}"),
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
                              "retrained; val-min-anchored 1200-step "
                              "cosine, lr 4e-4; G-BASE-QUAL re-derived on "
                              "the load)"),
                     "final_val_loss": G_BASE["final_val_loss"]},
            "install": {"steps": INSTALL_STEPS, "seed": G1.INSTALL_SEED,
                        "source": ("g1bS2's movement-matched install LOADED "
                                   f"VERBATIM from runs/checkpoints/"
                                   f"{SRC_INSTALL_CK} (post-install state "
                                   "REUSABLE; cells re-measured and "
                                   "cross-checked by this run)"),
                        "post_install_cells": inst_cells},
            "redraw": {
                "movement_rms": MV, "steps": STEPS, "lr": CONS_LR,
                "seed": cons["seed"],
                "recipe": ("e113 jitter convention VERBATIM (jitters "
                            "{-8..+8}, batch 16 install + 16 anchor, "
                            "name-masked union CE, AdamW (0.9,0.95) wd 0.1 "
                            "const lr clip 1.0) at the G1BS3-licensed rate "
                            "4e-4 — the SEED (10903, fresh) is the ONLY "
                            "delta vs g1bS5's m020 run (10901); the dose "
                            "0.20 rms / 500 steps is VERBATIM the peak "
                            "point"),
                "steps_ran": cons["steps_ran"],
                "device": cons["device"],
                "n_chunks": cons["n_chunks"],
                "chunk_table": cons["chunk_table"],
                "traj": cons["traj"],
                "redrew_stream": redrew_stream,
                "redrew_bits": redrew_bits,
                "draw_sensitivity": draw_sensitivity,
                "root_read": root_read,
                "root_cells": root_cells},
            "original_draw": G1BS5_M020,
            "committed_curve": G1BS5_CURVE,
            "fuzz_floor": FUZZ_FLOOR,
            "owner_envelope": {
                "launch_gate": REGISTERED["owner_envelope"]["launch_gate"],
                "burst_cap_s": TRAIN_CAP_GPU, "cooldown_s": COOLDOWN_S,
                "poll_audit": REGISTERED["owner_envelope"]["poll_audit"],
                "events": device_events},
            "priors_crosscheck": priors_check,
            "seeds": {"host_init_base_corpus": HOST_SEED,
                      "install": G1.INSTALL_SEED,
                      "consolidation": FRESH_SEED,
                      "original_draw_consolidation": ORIG_SEED,
                      "protocol_corpus": 1337},
        },
        "comparison": {
            "fresh_gm12": fresh_gm12, "orig_gm12": orig_gm12,
            "abs_d": d, "signed_d": signed_d,
            "robust_band": [orig_gm12 - 0.05, orig_gm12 + 0.05],
            "lottery_edge": [orig_gm12 - 0.10, orig_gm12 + 0.10],
        },
        "adjudication": {
            "bars_verbatim": REGISTERED["bars_verbatim"],
            "operationalization": REGISTERED["operationalization"],
            "d": d, "fresh_gm12": fresh_gm12, "orig_gm12": orig_gm12,
            "PEAK_ROBUST_fired": bool(d <= 0.05),
            "PEAK_LOTTERY_fired": bool(d > 0.10),
            "fired": fired,
            "clause": clause,
            "texture": texture,
            "no_bar_shopping": ("the fresh seed (10903), the dose (0.20 "
                                "rms / 500 steps) and the bars were frozen "
                                "in the dispatch and registered in the "
                                "script BEFORE any compute; the original "
                                "draw's numbers are g1bS5's committed "
                                "record, cross-checked (warn-only) not "
                                "re-tuned"),
        },
        "gates": {"G_CONFIG": G_CONFIG, "G_BASE": G_BASE,
                  "G_BASE_QUAL": G_BASE_QUAL, "G_INST": G_INST,
                  "G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                  "G_POOL": G_POOL, "G_SURG": gates_surg,
                  "G_REDREW": {"stream": redrew_stream,
                               "bits": redrew_bits,
                               "pass": bool((redrew_bits or {}).get(
                                   "pass", redrew_stream is not None))},
                  "pass": bool(G_CONFIG["pass"] and G_BASE["pass"]
                               and G_BASE_QUAL["pass"] and G_INST["pass"])},
        "honesty_reflex": {
            "n1_draw": ("n=1 REDRAW (two draws total at the peak dose: "
                        "10901's and 10903's; one host, one fact) — the "
                        "|d| reading is a TWO-POINT variance glimpse, not "
                        "a variance estimate; the seed-axis in-run |dg0| "
                        "texture and the same-seed replay fuzz floor are "
                        "DIFFERENT currencies, both stamped; no third draw "
                        "was taken (none registered)"),
            "consolidation_only": ("NO WALL CLAIMS and NO CURVE REVISION: "
                                   "this cell does not re-adjudicate "
                                   "g1bS5's SHARP-OPTIMUM (its frozen "
                                   "operationalization lives on the "
                                   "original draw's interior readings) nor "
                                   "g1bS6's WALL-FADES; the redraw's "
                                   "relation to the 0.78 bar / 0.6998 "
                                   "threshold / 0.7431 reading is TEXTURE, "
                                   "co-reported never adjudicated"),
            "original_anchor": ("the comparison anchor is g1bS5's "
                                "COMMITTED m020 record (cross-checked "
                                "warn-only at runtime); g1bS4-assembled "
                                "print-precision texture does not touch "
                                "this leg (the m020 row is g1bS5's own "
                                "full-precision run)"),
            "openness": ("nothing was guaranteed — the redraw was open by "
                         "design; the registered prediction was NONE and "
                         "the verdict above is the comparison's, not a "
                         "thesis's"),
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

    plot(rd / "redraw.png", fresh_gm12, orig_gm12, root_cells, cons,
         orig_traj, draw_sensitivity, fired, clause, d, texture)

    log(f"outputs: {rd / 'metrics.json'}, {rd / 'redraw.png'}, "
        f"root ckpt in runs/checkpoints/g1bS7_root_{TAG}.pt")
    log(f"VERDICT: {fired}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ driver

def g1bS7_consolidate(net0, pool_x, pool_mask, anchor, train_ids,
                      f_eval_ids, r_eval_xy, zid, tag,
                      steps: int, lr: float = CONS_LR,
                      resume_ck: Path | None = None,
                      on_chunk=None) -> dict:
    """g1bS5_consolidate VERBATIM (itself e113's finetune_arm VERBATIM
    arithmetic — draw order ix/aj/rj, batch 16 install + 16 anchor,
    name-masked union CE, AdamW (0.9,0.95) wd 0.1 const lr clip 1.0, eval
    every 25 steps) with the OWNER-ENVELOPE adaptations ONLY (device
    policy; no training-semantics change): the launch gate is util<=20% AND
    temp<=70C double-poll (wait_gpu_owner), the burst cap is 90 s
    (TRAIN_CAP_GPU), the between-chunk cooldown is 180 s, outside load
    mid-burst => PAUSE-AND-WAIT (never migrate), no CPU fallback. THE SEED
    rides G1.CONS_SEED (= 10903, THE TREATMENT, set at import). CHUNK-
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
    torch.save({"model": sd, "meta": {"experiment": "g1bS7", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name}")


# ------------------------------------------------------------------ plot
# THE REDRAW figure: the two draws side by side on the committed curve
# (the headline), the in-run divergence, the co-reads, the verdict.

def plot(path, fresh_gm12, orig_gm12, root_cells, cons, orig_traj,
         draw_sensitivity, fired, clause, d, texture):
    import textwrap
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.5))
    xs = G1BS5_CURVE["x_movement_rms"]
    gm = list(G1BS5_CURVE["gm12"])
    committed = G1BS5_CURVE["committed"]

    # (0,0) THE HEADLINE: the two draws side by side on the curve
    ax = axes[0, 0]
    ax.plot(xs, gm, "-", lw=1.4, color="gray", alpha=0.6, zorder=1)
    cx = [x for x, c in zip(xs, committed) if c]
    cy = [v for v, c in zip(gm, committed) if c]
    nx = [x for x, c in zip(xs, committed) if not c]
    ny = [v for v, c in zip(gm, committed) if not c]
    ax.plot(cx, cy, "o", ms=11, mfc="white", mec="black", mew=2, zorder=3,
            label="committed endpoints (g1bS3 s300 / g1bS4 s750)")
    ax.plot(nx, ny, "s", ms=10, color="seagreen", zorder=4,
            label="g1bS5 interior (seed 10901)")
    # THE PEAK-ROBUST band around the original draw's height
    ax.axhspan(orig_gm12 - 0.05, orig_gm12 + 0.05, color="royalblue",
               alpha=0.12, zorder=0,
               label=f"PEAK-ROBUST band {orig_gm12:.4f} +- 0.05")
    ax.axhline(orig_gm12 - 0.10, ls="--", lw=1.2, color="crimson",
               alpha=0.7, label=f"PEAK-LOTTERY edges {orig_gm12:.4f} +- 0.10")
    ax.axhline(orig_gm12 + 0.10, ls="--", lw=1.2, color="crimson", alpha=0.7)
    ax.axhline(orig_gm12, ls=":", lw=1.1, color="gray", alpha=0.9)
    # the two draws at x=0.20, side by side
    ax.plot([0.20], [orig_gm12], "s", ms=13, mfc="seagreen", mec="black",
            mew=1.5, zorder=5, label=f"original draw (seed 10901): {orig_gm12:.4f}")
    ax.plot([0.20], [fresh_gm12], "D", ms=13, mfc="crimson", mec="black",
            mew=1.5, zorder=6, label=f"FRESH draw (seed 10903): {fresh_gm12:.4f}")
    ax.annotate(f"|d| {d:.4f}\n{fired}", (0.20, (orig_gm12 + fresh_gm12) / 2),
                textcoords="offset points", xytext=(46, 0), fontsize=9,
                ha="left", weight="bold",
                arrowprops=dict(arrowstyle="-", lw=1.0))
    ax.annotate(f"{orig_gm12:.4f}", (0.20, orig_gm12),
                textcoords="offset points", xytext=(-4, -16), fontsize=8,
                ha="right", weight="bold")
    ax.annotate(f"{fresh_gm12:.4f}", (0.20, fresh_gm12),
                textcoords="offset points", xytext=(-4, 10), fontsize=8,
                ha="right", weight="bold", color="crimson")
    ax.axhline(G1.EXPRESS_BAR, ls="--", lw=1.8, color="black", alpha=0.9,
               label=f"express bar {G1.EXPRESS_BAR}")
    ax.axhline(SHARP_THRESHOLD, ls="-.", lw=1.2, color="darkorange",
               alpha=0.9, label=f"SHARP-OPTIMUM threshold {SHARP_THRESHOLD:.4f}")
    ax.set_xlabel("total consolidation movement (rms, Adam-clock @ 4e-4)")
    ax.set_ylabel("root g-12 (displaced-geometry battery)")
    ax.set_ylim(-0.05, 1.08)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("THE REDRAWN PEAK — the two draws side by side on the "
                 "formation curve", fontsize=10)

    # (0,1) the in-run trajectories: the two draws' streams
    ax = axes[0, 1]
    tj_f = cons["traj"]
    ax.plot([t["step"] for t in tj_f], [t["install60_g0_pz"] for t in tj_f],
            "-", lw=1.6, color="crimson", alpha=0.9,
            label=f"fresh draw seed 10903 (s{cons['steps_ran']}) in-run g0")
    if orig_traj is not None:
        ax.plot([t["step"] for t in orig_traj],
                [t["install60_g0_pz"] for t in orig_traj], "--", lw=1.6,
                color="seagreen", alpha=0.9,
                label="original draw seed 10901 (g1bS5 committed) in-run g0")
    ax.axvline(cons["steps_ran"], ls=":", lw=1.2, color="crimson", alpha=0.7)
    ax.set_xlabel("consolidation step")
    ax.set_ylabel("in-run g0 (site channel, every 25 steps)")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("THE TWO STREAMS — the redraw is not a replay", fontsize=10)

    # (0,2) CE_R stability of both draws
    ax = axes[0, 2]
    ax.plot([t["step"] for t in tj_f], [t["ce_r"] for t in tj_f], "-",
            lw=1.6, color="crimson", alpha=0.9, label="fresh draw CE_R")
    if orig_traj is not None:
        ax.plot([t["step"] for t in orig_traj], [t["ce_r"] for t in orig_traj],
                "--", lw=1.6, color="seagreen", alpha=0.9,
                label="original draw CE_R")
    ax.axhspan(1.5, 1.8, color="seagreen", alpha=0.10)
    ax.annotate("healthy band ~1.5-1.8", (10, 1.76), fontsize=7.5,
                color="seagreen")
    ax.set_xlabel("consolidation step")
    ax.set_ylabel("CE_R (60-window val bank)")
    ax.legend(fontsize=7.5)
    ax.set_title("THE STABILITY AXIS — both draws", fontsize=10)

    # (1,0) the draw-sensitivity: |dg0| fresh vs orig per eval step
    ax = axes[1, 0]
    if draw_sensitivity is not None and (draw_sensitivity.get("n_shared")
                                         or 0) > 0:
        _tr = cons["traj"]
        _od = {t["step"]: t for t in (orig_traj or [])}
        ss = [t["step"] for t in _tr if t["step"] in _od]
        dd = [abs(t["install60_g0_pz"] - _od[t["step"]]["install60_g0_pz"])
              for t in _tr if t["step"] in _od]
        ax.plot(ss, dd, "-", lw=1.5, color="crimson", alpha=0.9,
                label="|dg0| fresh vs orig (the SEED axis)")
    ax.axhline(FUZZ_FLOOR["mean_d_g0"], ls="--", lw=1.2, color="gray",
               label=f"same-seed replay fuzz mean {FUZZ_FLOOR['mean_d_g0']}")
    ax.axhline(FUZZ_FLOOR["max_d_g0"], ls=":", lw=1.2, color="gray",
               label=f"same-seed replay fuzz max {FUZZ_FLOOR['max_d_g0']}")
    ax.set_xlabel("consolidation step")
    ax.set_ylabel("|d g0| between the two draws")
    ax.legend(fontsize=7.5)
    ax.set_title("THE SEED AXIS — draw sensitivity in-run (texture)",
                 fontsize=10)

    # (1,1) the root co-reads: fresh vs original
    ax = axes[1, 1]
    keys = ["gm12", "g0", "gp12", "held30_gm12"]
    labels = ["g-12", "g0", "g+12", "held30 g-12"]
    xq = np.arange(len(keys))
    orig_vals = [orig_gm12, G1BS5_M020["g0"], G1BS5_M020["gp12"],
                 G1BS5_M020["held30_gm12"]]
    fresh_vals = [fresh_gm12, root_cells["g0"], root_cells["gp12"],
                  root_cells["held30_gm12"]]
    ax.bar(xq - 0.18, orig_vals, 0.34, color="seagreen", alpha=0.85,
           label="original draw (10901)")
    ax.bar(xq + 0.18, fresh_vals, 0.34, color="crimson", alpha=0.85,
           label="fresh draw (10903)")
    for i, (o, f) in enumerate(zip(orig_vals, fresh_vals)):
        ax.annotate(f"{o:.3f}", (i - 0.18, o), textcoords="offset points",
                    xytext=(0, 4), fontsize=7, ha="center")
        ax.annotate(f"{f:.3f}", (i + 0.18, f), textcoords="offset points",
                    xytext=(0, 4), fontsize=7, ha="center", color="crimson")
    ax.axhline(G1.EXPRESS_BAR, ls="--", lw=1.4, color="black", alpha=0.8,
               label=f"express bar {G1.EXPRESS_BAR}")
    ax.set_xticks(xq, labels)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7.5)
    ax.set_title(f"THE ROOT CELLS — both draws "
                 f"(CE_R {G1BS5_M020['ce_r']:.4f} -> {root_cells['ce_r']:.4f})",
                 fontsize=10)

    # (1,2) THE VERDICT PANEL
    ax = axes[1, 2]
    ax.axis("off")
    y = 0.97
    ax.text(0.02, y, "G1BS7 — THE REDRAWN INTERIOR DOSE (fresh seed 10903 "
            "@ the 0.20 rms peak)", fontsize=11, va="top",
            family="monospace", weight="bold")
    y -= 0.052
    ax.text(0.02, y, f"host {G1BS_PARAMS:,} params | lr {CONS_LR} | dose "
            f"{MV} rms = s{STEPS} | the seed is THE TREATMENT (10901 -> "
            f"10903) | n=1 redraw", fontsize=7.2, va="top",
            family="monospace")
    y -= 0.034
    ax.text(0.02, y, f"  original draw (10901): g-12 {orig_gm12:.4f}   "
            f"fresh draw (10903): g-12 {fresh_gm12:.4f}   |d| {d:.4f}",
            fontsize=7.4, va="top", family="monospace")
    y -= 0.028
    ax.text(0.02, y, f"  bands: ROBUST <= 0.05 | LOTTERY > 0.10 | the "
            f"middle is GRADED | same-seed fuzz floor mean "
            f"{FUZZ_FLOOR['mean_d_g0']}/max {FUZZ_FLOOR['max_d_g0']} "
            f"(a different currency)", fontsize=6.8, va="top",
            family="monospace")
    y -= 0.032
    for k, v in (("express bar 0.78",
                  texture["express_bar"]["read"]),
                 ("SHARP-OPTIMUM threshold 0.6998 on the redraw",
                  "would re-fire" if texture["sharp_threshold"]
                  ["would_refire"] else "NOT met"),
                 ("the fine ordering vs 0.25's 0.7431",
                  "peak still on top" if texture["fine_ordering"]
                  ["peak_still_outranks_0p25"] else "FLIPS below 0.25")):
        ax.text(0.02, y, f"  texture: {k}: {v}", fontsize=6.8, va="top",
                family="monospace")
        y -= 0.028
    y -= 0.012
    ax.text(0.02, y, f"VERDICT: {fired}", fontsize=9.5, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.042
    for wd in textwrap.wrap(clause, width=88, break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=6.6, va="top",
                family="monospace")
        y -= 0.026

    fig.suptitle("G1BS7 — THE REDRAWN INTERIOR DOSE: the fresh draw (seed "
                 "10903) vs the original (seed 10901) at the 0.20 rms peak "
                 f"of the formation curve -> {fired}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
