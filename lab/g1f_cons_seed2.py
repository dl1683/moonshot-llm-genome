"""G1F-CONS2 — THE SECOND CONS SEED (g1e's named open rung: the
+1-crush variance; dispatched 2026-10-02, Fleet 2's GPU leg).

CONTEXT (T188): g1e closed the wall's last stream axis with the family's
first maintain-bar breach — the fresh cons draw's W1 read 0.2719 AT +1
(the truncation of the first projected step landing harder) then
RECOVERED and held flat 0.52-0.63 (the flat phase surviving every axis,
at 0.61-0.74x its own root). T188's map: wash n=3 HOLDS; root n=2 HOLDS;
cons n=1-of-2 BOUND at the +1 transient; base observed-unadjudicated;
and the named open rung: "a second cons seed to split the +1-crush
variance from n=1 noise." THE +1 LEDGER going in: the wash/root family
reads 0.82-0.96 at +1 (g1b 0.9452; g1bR 0.9623/0.9099; g1c 0.8214);
g1d's base axis 0.4820 (observed-unadjudicated); g1e's cons draw 0.2719
(the only adjudicated breach). Was 0.27 a draw or the cons axis's
texture?

THE CELL (the g1e machinery VERBATIM with cons seed 10912 -> 10913 —
the next free neighbor; the ONLY delta from g1e; vs the locked root,
10901 -> 10913 with base/install/wash all HELD, two draws deep):
  phase 0 (the second fresh cons draw):
    base    runs/checkpoints/e001.pt — LOADED FIXED (2,739,072 params,
            gated bit-exact + fact-free).
    install runs/checkpoints/e048_repro.pt — THE LOCKED DRAW'S OWN
            ARTIFACT (gen 24313), LOADED, NEVER RETRAINED.
    cons    e113 jitter replay VERBATIM at seed 10913 — THE CONS-DRAW
            KNOB (genealogy: 10901 locked, 10902 wash/e119-L, 10903-4
            e184/e185c, 10905-6 g3R/e152r, 10907-8 g1bR, 10909-11 g2g2,
            10912 g1e's first redraw (the +1 breach), 10913 THIS DRAW —
            unused anywhere before this cell).
  the arms (VERBATIM; wash seed 10902 HELD so the delta vs BOTH g1e's
  leg and g1b's reference leg is THE CONS DRAW, never the wash):
    C    uncommitted neutral wash 300 (the per-root kill clock)
    W1   commit(R=0.7 RAW L2 — the 2.74M convention) then the
         seed-identical neutral wash; step-1 weights equal to C's.

THE ROOT-STRENGTH GATE + every hard gate: g1e's form VERBATIM (the
coin-flip caveat on G-ROOT registered; G-CONS hard with the g1d
precedent — failure => TEXTURE, the arms run anyway, the record texture
the finding).

THE FROZEN BARS (the g1f dispatch letter, VERBATIM; operationalized
BEFORE any compute — no shopping):
  CRUSH-WAS-A-DRAW: "fires if the second cons draw's +1 reads in-family
      (>= 0.5) and W1 maintains — the g1e breach was a draw; the cons
      axis joins the protection grid as HOLDS at n=2; the +1 crush
      variance is ordinary draw noise."
  CRUSH-IS-TEXTURE: "fires if the second cons draw also breaches at +1
      (< 0.50) — the cons axis carries a systematically deeper crush;
      the grid's cons cell stays BOUND; the texture named."
  GRADED: "any partial — the tables verbatim (the +1 read co-reported
      with the flat phase either way)."
  OPERATIONALIZATION (frozen here, g1e's committed machinery): W1
      MAINTAINS = g1b's maintain bar VERBATIM (g-12 >= 0.50 at EVERY
      ckpt {1,2,4,10,50,100,200,300} — the +1 ckpt INCLUDED, so the
      in-family clause and the maintain clause are ONE bar) AND C dies
      by +50 (g-12@+50 <= 0.27) => CRUSH-WAS-A-DRAW; C dies AND the +1
      read itself < 0.50 => CRUSH-IS-TEXTURE; any other partial (a +1
      in-family read with a LATER maintain-bar breach; C not dying — a
      wash-scale anomaly) => GRADED. STRICT CO-REPORT (never primary):
      the every-flat-phase-ckpt {10,50,100,200,300} 0.9/0.9156xroot form
      (g1c's convention); the 0.9xroot flat-phase reading, the +2 dip
      and FLAT-AT-PIN (|g(+300)-g(+50)| <= 0.05) co-reported as texture.
      Hard-gate failure (outside the root-gate caveat) => TEXTURE; the
      record completes.
  REGISTERED PREDICTION (before compute; NOTHING GUARANTEED — the
  openness is the point): if the +1 crush is ordinary draw noise (the
  single-draw lottery T188 named), this draw's +1 lands in the family
  band (>= 0.5; the family 0.82-0.96) and W1 maintains flat-scaled-to-
  its-own-root => CRUSH-WAS-A-DRAW. FALSIFIER: a second +1 breach =>
  CRUSH-IS-TEXTURE — n=2-of-3 cons draws breaching at +1 is no longer
  draw-shaped against a wash/root family at 0.82-0.96. THE OPEN BITS:
  the +1 crush depth at n=2; whether the flat phase survives again (it
  has in EVERY expressed draw).

GATES (g1e's form VERBATIM + the extended cons-draw provenance):
  G-BASE  G-INST  G-CONSDRAW (seed 10913 registered free + divergence
  from BOTH the locked e131 root (the gate) and g1e's 10912 root
  (co-reported texture — the sibling distance))  G-CONS (hard, final
  root g-12 >= 0.78)  G-ROOT (0.7 ruler; caveat-eligible)  G-BITROOT
  G-INPUTS  G-STEP1  G-CTRL  G-PIN  + the construction gates G_SPLICE /
  G_NAMEFREE / G_POOL / G_INSTMASK / G_ANCHOR.

THE OWNER ENVELOPE (STATE.json compute_directive 2026-10-02 + this
dispatch's letter): launch gate util <= 20% AND temp <= 70C
(double-poll 5 s; mem <= 60%); SHORT bursts <= 60 s (TRAIN_CAP_GPU=60 —
tighter than g1e's 90; the chunked full-state resume makes the cap a
device-policy knob only, no training semantics); cooldown >= 180 s; NO
back-to-back; when in doubt WAIT (after GPU_WAIT_MAX=2 h of no windows:
park to CPU, cap 1800 s/training, loudly recorded — never a silent
hop); every poll appended to runs/_envelope_log.jsonl. torch threads 4
(shared machine).

COMPUTE ENVELOPE: 2,739,072 params (inside the <= 100M free tier; the
stated reason is the lineage's own: CONTINUITY on the e131 line — the
reference scale the claim was minted on). 3 trainings (cons 300 / C 300
/ W1 300 — base and install are LOADED artifacts), in owner-envelope
bursts.

Outputs: runs/g1f/{metrics.json (PROGRESSIVE), cons_seed2.png};
checkpoints runs/checkpoints/g1f_*.pt. No NOTES/THINKING/QUEUE/STATE
edits (the coordinator folds). Commit + push per phase.

Run:  cd lab && python g1f_cons_seed2.py    (G1F_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

SMOKE = os.environ.get("G1F_SMOKE") == "1"
if SMOKE:
    os.environ["G1B_SMOKE"] = "1"     # cascades: g1b sets G1_SMOKE before
                                      # g1_anchored_ball's first import

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
import torch.nn.functional as F                       # noqa: E402

import common                                          # noqa: E402
from common import (CharCorpus, cooldown, gpu_status,  # noqa: E402
                    run_dir, save_json, set_seed)
import e043_install as E43                             # noqa: E402 (REPO,
                                                      # find_occ, SPLICE_RNG,
                                                      # jsonable)
import g1b_continuity as GB                            # noqa: E402 — MUST be
                                                      # imported BEFORE G1
                                                      # (the 2.74M patch +
                                                      # smoke cascade)
import g1_anchored_ball as G1                          # noqa: E402 — the
                                                      # machinery (patched to
                                                      # 2.74M by g1b's import)

torch.set_num_threads(4)           # shared machine (g1bS8's adopted trim;
                                   # g1/g1b's import resets to 8)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)
G1.log = log                                          # unify the timeline

# ======================================================================
# THE CONFIG (the whole cons-draw delta + the envelope)
# ======================================================================
BASE_CK = "e001.pt"               # the 2.74M corpus base (LOADED FIXED)
INST_CK = "e048_repro.pt"         # the LOCKED draw's install artifact at gen
                                 # 24313 (LOADED, never retrained)
LOCKED_ROOT_CK = GB.ROOT_CK       # e131_consolidated_e113.pt
INST_GEN = 24313                  # HELD (recorded provenance of the loaded
                                 # artifact — the locked draw's own)
CONS_SEED = 10913                 # THE CONS-DRAW KNOB (10912 -> 10913:
                                 # g1e's next free neighbor — the SECOND
                                 # cons redraw; see the genealogy in the
                                 # docstring)
WASH_SEED = 10902                 # HELD (g2e: the delta vs g1b's leg is the
                                 # CONS DRAW, not the wash)
CONS_STEPS = 300 if not SMOKE else 8
R_CLAIM = 0.7                     # RAW L2 — the 2.74M convention (verbatim)
ROOT_BAR = 0.7                    # g2's root-strength bar (the ruler gate)
G1B_ROOT_GM12 = 0.9155886769294739   # the locked root's g-12 (frozen ref)
COMMITTED_LOCKED_INST_G0 = 0.5563086867332458
                                 # the locked artifact's same-instrument g0,
                                 # measured in-run by BOTH g1c and g1d
REUSE_CHECK_TOL = 0.02            # g1bS7's loaded-artifact reuse-check bar
FLAT_PHASE_CK = (10, 50, 100, 200, 300)   # g1bS8's post-transient set
USED_FAMILY_SEEDS = {10901: "locked cons (e109a/e113/g1b/g1c/g1d)",
                     10902: "wash (g1b) / e119-L / e143 NEAR+FAR / e152",
                     10903: "e184/e185c replicate / g1bS7 fresh jitter (10M)",
                     10904: "e184/e185c replicate",
                     10905: "g3R wash replicate / e152r reseed",
                     10906: "g3R wash replicate / e152r reseed",
                     10907: "g1bR wash band", 10908: "g1bR wash band",
                     10909: "g2g2 ladder", 10910: "g2g2 ladder",
                     10911: "g2g2 ladder",
                     10912: "g1e cons redraw (the first +1 breach, T188)"}
assert CONS_SEED not in USED_FAMILY_SEEDS, \
    f"cons seed {CONS_SEED} collides with the family genealogy"

# ---- THE OWNER ENVELOPE (tighter than the lab's standing guards) ----------
OWNER_UTIL_C = 20.0              # launch only when util <= 20%
OWNER_TEMP_C = 70.0              # AND temp <= 70C (double-poll)
OWNER_MEM_FRAC = 0.60            # resident-neighbor caution (mem <= 60%)
COOLDOWN_S = 180.0               # >= 180 s between bursts (the dispatch)
TRAIN_CAP_GPU = 60.0             # <= 60 s GPU bursts (the g1f dispatch
                                 # letter; g1e ran 90 — TIGHTER here; the
                                 # chunked full-state resume makes the cap
                                 # device-policy only, no training semantics)
TRAIN_CAP_CPU = 1800.0           # the parked-CPU cap (g2e's convention)
GPU_WAIT_MAX = float(os.environ.get("G1F_GPU_WAIT_MAX", "7200"))
                                 # when in doubt WAIT; after 2 h of no
                                 # windows: park to CPU (loudly recorded;
                                 # 2.74M is CPU-viable — never a silent hop)

REF_METRICS = E43.REPO / "runs" / "g1b" / "metrics.json"   # the n=1-cons ref
REFB_METRICS = E43.REPO / "runs" / "g1bR" / "metrics.json"  # the wash band
REFC_METRICS = E43.REPO / "runs" / "g1c_root" / "metrics.json"  # the root
                                 # axis sibling (same cons seed 10901 —
                                 # co-plotted context)
REFD_METRICS = E43.REPO / "runs" / "g1d" / "metrics.json"   # the base axis
                                 # (observed-unadjudicated — context)
REFE_METRICS = E43.REPO / "runs" / "g1e" / "metrics.json"   # the FIRST cons
                                 # redraw (seed 10912 — the +1 breach this
                                 # cell adjudicates; T188's open rung)

G1F_PREDICTION = {
    "bars_verbatim": {
        "CRUSH-WAS-A-DRAW": ("fires if the second cons draw's +1 reads "
                             "in-family (>= 0.5) and W1 maintains — the "
                             "g1e breach was a draw; the cons axis joins "
                             "the protection grid as HOLDS at n=2; the +1 "
                             "crush variance is ordinary draw noise."),
        "CRUSH-IS-TEXTURE": ("fires if the second cons draw also breaches "
                             "at +1 (< 0.50) — the cons axis carries a "
                             "systematically deeper crush; the grid's cons "
                             "cell stays BOUND; the texture named."),
        "GRADED": ("any partial — the tables verbatim (the +1 read "
                   "co-reported with the flat phase either way)."),
    },
    "operationalization": (
        "frozen BEFORE compute from the dispatch letter (the g1f bars "
        "VERBATIM; g1e's committed bar machinery carried over): W1 "
        "MAINTAINS = g1b's maintain bar VERBATIM (g-12 >= 0.50 at EVERY "
        "ckpt {1,2,4,10,50,100,200,300} — the +1 ckpt INCLUDED, so the "
        "in-family clause and the maintain clause are ONE bar) + C dies "
        "by +50 (g-12@+50 <= 0.27); THE +1 READ (the crush read) is "
        "named in every verdict; STRICT CO-REPORT (never primary) = the "
        "every-flat-ckpt {10,50,100,200,300} 0.9/0.9156xroot form (g1c's "
        "convention); co-reported texture: the 0.9xroot flat-phase "
        "reading, the +2 dip, FLAT-AT-PIN (|g(+300)-g(+50)| <= 0.05). No "
        "bar widened after seeing data."),
    "predicted": ("NOTHING GUARANTEED — the openness is the point (the "
                  "dispatch letter). The +1 ledger going in: the "
                  "wash/root family 0.82-0.96 (g1b 0.9452; g1bR "
                  "0.9623/0.9099; g1c 0.8214); g1d's base axis 0.4820 "
                  "(observed-unadjudicated); g1e's cons draw 0.2719 (the "
                  "family's only adjudicated breach — then RECOVERED "
                  "flat 0.52-0.63; T188: the flat phase survived every "
                  "axis, the transient is the last lottery). IF the +1 "
                  "crush is ordinary draw noise, this draw's +1 lands "
                  "in-family (>= 0.5) and W1 maintains flat-scaled-to-"
                  "its-own-root -> CRUSH-WAS-A-DRAW; the cons axis joins "
                  "the grid as HOLDS at n=2 redraws."),
    "falsifier": ("a SECOND +1 breach (< 0.50 at +1) -> CRUSH-IS-TEXTURE "
                  "(n=2-of-3 cons draws breaching at +1 is no longer "
                  "draw-shaped against a family at 0.82-0.96); a +1 "
                  "in-family read with a LATER maintain-bar breach (the "
                  "partial zone) -> GRADED, the tables verbatim; G-CONS "
                  "failure (< 0.78 final root) -> TEXTURE with the arms "
                  "run anyway (the g1d precedent); a ruler under 0.7 -> "
                  "the registered coin-flip deviation, stamped on the "
                  "verdict, never an abort (g1bS6/g1d precedent)"),
}

deviations: list[str] = [
    "THE CONS-DRAW DELTA (the whole difference from g1e): the e113 "
    "jitter-replay consolidation runs at seed 10913 (10912's next free "
    "neighbor — the SECOND cons redraw of the locked 10901; genealogy in "
    "the config: 10901 locked, 10902 wash/e119-L, 10903-10904 e184/e185c, "
    "10905-10906 g3R/e152r, 10907-10908 g1bR, 10909-10911 g2g2, 10912 "
    "g1e's first redraw (the +1 breach), 10913 unused anywhere before "
    "this cell). The base (e001.pt, LOADED FIXED), the install "
    "(e048_repro.pt — the LOCKED DRAW'S OWN artifact at gen 24313, LOADED "
    "and never retrained; g1bS7's loaded-artifact convention) and the "
    "wash (10902 HELD) are ALL g1e-verbatim — the only delta vs g1e is "
    "THE 300-STEP CONS STREAM (and vs the locked root: the same "
    "isolation, two draws deep).",
    "THE SCOPE (T188's named open rung): this cell exists to split the "
    "+1-CRUSH VARIANCE from n=1 noise — g1e's W1 read 0.2719 at +1 (the "
    "family's first maintain-bar breach; the wash/root family reads "
    "0.82-0.96 at +1: g1b 0.9452, g1bR 0.9623/0.9099, g1c 0.8214; g1d's "
    "base axis 0.4820 observed-unadjudicated) then recovered flat "
    "0.52-0.63. ONE new draw decides between CRUSH-WAS-A-DRAW and "
    "CRUSH-IS-TEXTURE per the frozen bars — nothing else is re-run.",
    "THE BAR OPERATIONALIZATION (the dispatch letter's three bars, frozen "
    "BEFORE compute; g1e's committed machinery): W1 MAINTAINS = g1b's "
    "maintain bar VERBATIM (>= 0.50 at EVERY ckpt — the +1 ckpt included: "
    "the in-family clause and the maintain clause are ONE bar) + C dies "
    "by +50 => CRUSH-WAS-A-DRAW; C dies AND the +1 read itself < 0.50 => "
    "CRUSH-IS-TEXTURE; any other partial => GRADED (the +1 read "
    "co-reported with the flat phase either way). The strict "
    "0.9/0.9156xroot form stays a CO-REPORT, never primary (the "
    "reference family itself straddles it). No bar widened after seeing "
    "data.",
    "G-CONS IS HARD, THE g1d PRECEDENT (registered BEFORE compute): if "
    "the second fresh cons draw fails to express (final root g-12 < 0.78) "
    "the verdict is TEXTURE — the cons-level lottery owning the "
    "expression channel — BUT THE ARMS RUN ANYWAY and the record texture "
    "(the wall guarding whatever expressed) is itself the finding. G-ROOT "
    "(the 0.7 ruler) stays the ONE caveat-eligible gate: a ruler under "
    "0.7 is a REGISTERED DEVIATION (the coin-flip clause; g1bS6/g1d "
    "precedent), stamped on the verdict, never an abort.",
    "THE OWNER ENVELOPE (STATE.json compute_directive 2026-10-02 + this "
    "dispatch's letter) — TIGHTER THAN g1e on the burst cap: "
    "TRAIN_CAP_GPU 60 s (g1e ran 90; the letter says <= 60 s; g1e's own "
    "chunks finished in 17-22 s so the cap is slack either way); launch "
    "gate util <= 20% AND temp <= 70C double-poll (+ mem <= 60%), "
    "cooldown >= 180 s, NO back-to-back, every poll appended to "
    "runs/_envelope_log.jsonl (the R61-critic trail); after GPU_WAIT_MAX "
    "with no window: PARK TO CPU (cap 1800 s/training, 4 threads) loudly "
    "recorded — at 2.74M the cell is CPU-viable; never a silent hop.",
    "CHUNK-RESUMABLE TRAININGS (g1bS8's mechanical addition, carried from "
    "g1e VERBATIM; device policy only, no training semantics): "
    "consolidation/wash run in counted bursts with full-state resume "
    "ckpts ({model, opt, gen_state, step, traj, ...}) at every burst "
    "boundary — bit-identical to an uninterrupted run (the generator + "
    "optimizer state carry the whole stream). The consolidation's "
    "arithmetic is G1.consolidate (e113) VERBATIM; the wash's is "
    "G1.g1_wash (e185/e176N) VERBATIM.",
    "ROOT DIAL FULL: the second fresh root is a NEW object — the full "
    "dial (census, deletions, site read) is measured and recorded (g1c's "
    "convention); the base and the install are LEAN dials (loaded "
    "artifacts; their provenance is the load gates + the committed "
    "records).",
    "torch threads 4 (shared machine; g1bS8's trim; g1/g1b's import "
    "resets to 8 — reset after import).",
    "Smoke mode trims: 8-step cons, 4-step washes (ckpts {1,2,4} via g1's "
    "smoke cascade), lean dials, no cooldowns, gate waits capped at 30 s "
    "— nothing adjudicated.",
]

device_events: list[dict] = []
trims: list[str] = []
_progressive = {"n": 0, "phases": []}
CKPT_INVENTORY: dict = {}


# ------------------------------------------------------------------ envelope
class OwnerWindowShut(Exception):
    """No owner-envelope GPU window within GPU_WAIT_MAX — the remaining
    trainings park to CPU (loudly recorded; 2.74M is CPU-viable)."""


def owner_gpu_ok(tag: str = "owner") -> bool:
    s = gpu_status()
    ok = bool(s["util"] <= OWNER_UTIL_C and s["temp"] <= OWNER_TEMP_C
              and (s["mem_total"] == 0
                   or s["mem_used"] <= OWNER_MEM_FRAC * s["mem_total"]))
    try:                            # g1bS7's adopted poll-audit trim
        common._log_envelope_poll(f"g1f:{tag}", s["util"], s["temp"], ok)
    except Exception:
        pass
    return ok


def wait_gpu_owner(tag: str, max_wait: float | None = None) -> torch.device:
    """THE OWNER ENVELOPE gate: launch only when util <= 20% AND temp
    <= 70C, double-poll 5 s apart (+ mem <= 60% caution). Parks in 30 s
    polls (when in doubt WAIT); after max_wait raises OwnerWindowShut —
    the caller parks to CPU with the deviation recorded."""
    mw = 30.0 if SMOKE else (GPU_WAIT_MAX if max_wait is None else max_wait)
    if not torch.cuda.is_available():
        raise OwnerWindowShut(f"'{tag}': no CUDA device")
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
    device_events.append({"tag": tag, "event": "OWNER-WINDOW TIMEOUT — CPU "
                          "PARK", "waited_s": round(waited, 1),
                          "status": gpu_status()})
    raise OwnerWindowShut(
        f"'{tag}': no owner window in {mw:.0f}s — PARKING TO CPU (recorded; "
        f"cap {TRAIN_CAP_CPU:.0f}s per training)")


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


def next_dev(tag: str, parked: bool) -> torch.device:
    """The owner-envelope device pick: park-once — after the first window
    shut, every remaining training runs CPU (recorded)."""
    if parked:
        return CPU
    try:
        return wait_gpu_owner(tag)
    except OwnerWindowShut as e:
        log(f"[gpu] {e}")
        trims.append(f"{tag}: owner window shut -> CPU park (recorded)")
        return CPU


def flat_params_cpu(net) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).cpu()
                      for p in net.parameters()]).clone()


def sd_md5(sd: dict) -> str:
    """Provenance hash of a state dict (g2f's 'hashes recorded' form)."""
    h = hashlib.md5()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------ partial
def write_partial(rd: Path, phase: str, payload: dict) -> None:
    """PROGRESSIVE metrics (the outage lesson): metrics.json after every
    phase; bookkeeping must never kill compute."""
    _progressive["n"] += 1
    _progressive["phases"].append(phase)
    try:
        out = {
            "experiment": "g1f_cons_seed2", "date": common.now_iso(),
            "phase": phase, "write_n": _progressive["n"],
            "phases": list(_progressive["phases"]),
            "device_events": device_events,
        }
        out.update(E43.jsonable(payload))
        save_json(rd / "metrics.json", out)
        log(f"[partial] metrics.json updated (phase '{phase}', write "
            f"#{_progressive['n']})")
    except Exception as e:
        log(f"[partial] WRITE FAILED at '{phase}' ({e}) — continuing")


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}" if not name.startswith("smoke_") else name
    path = GB.CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "g1f", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name}")


# ------------------------------------------------------------------ trainers
# CHUNKED DRIVERS: the consolidation/wash arithmetic is G1.consolidate /
# G1.g1_wash VERBATIM (draw order, batch composition, losses, optimizer,
# clip); the chunking/cooldown/polling is the owner-envelope device policy
# (g1bS8's registered mechanical addition — no training semantics changed).

def chunked_consolidate(tag, net0, pool_a_x, pool_a_mask, cons_anchor,
                        train_ids, g0_ids, r_eval_xy, zid,
                        resume_ck: Path) -> dict:
    """G1.consolidate (e113's finetune_arm) VERBATIM arithmetic: batch 32
    = 16 jittered-pool install windows (name-masked) + 16 anchors (8
    paired originals + 8 random); token-level union CE; AdamW (0.9,0.95)
    wd 0.1 const lr 1e-3; clip 1.0; s300, seed 10913 (THE cons-draw knob)
    — in owner-envelope bursts with full-state resume."""
    n_steps = CONS_STEPS
    n_pool, n_anc = pool_a_x.shape[0], cons_anchor.shape[0]
    state = {"step": 0, "traj": []}
    parked = False
    if resume_ck.exists():
        state = torch.load(resume_ck, map_location="cpu", weights_only=False)
        log(f"  [{tag}] RESUMED from {resume_ck.name} at step "
            f"{state['step']}/{n_steps}")
    # edge: a prior process already completed this training — return its
    # saved final state (bit-identical; no steps re-run)
    if int(state.get("step", 0)) >= n_steps:
        log(f"  [{tag}] resume ckpt already COMPLETE at s{state['step']} — "
            f"returning the saved final state")
        return {"sd": state["model"], "traj": state.get("traj", []),
                "steps_ran": n_steps, "seed": CONS_SEED,
                "device": "resumed (final state)", "devices": ["resumed"],
                "n_chunks": state.get("n_chunks", 0),
                "chunk_table": state.get("chunk_table", []),
                "theta0_norm": None}
    net, opt, gen, evl = None, None, None, None
    theta0 = None
    n_chunks, chunk_table, devices = 0, [], []
    step = state["step"]
    while step < n_steps:
        n_chunks += 1
        dev = next_dev(f"{tag}-chunk{n_chunks}", parked)
        parked = parked or dev.type == "cpu"
        cap = TRAIN_CAP_GPU if dev.type == "cuda" else TRAIN_CAP_CPU
        if net is None:
            net = copy.deepcopy(net0).to(dev)
            net.train()
            opt = torch.optim.AdamW(net.parameters(), lr=G1.FT_LR,
                                    betas=(0.9, 0.95), weight_decay=0.1)
            gen = torch.Generator().manual_seed(CONS_SEED)
            if resume_ck.exists():
                net.load_state_dict(state["model"])
                net.to(dev)
                net.train()
                opt.load_state_dict(state["opt"])
                gen.set_state(state["gen_state"])
                step = state["step"]
            theta0 = flat_params_cpu(net)
            evl = copy.deepcopy(net0).to(CPU)
        else:
            _osd = opt.state_dict()
            net = net.to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=G1.FT_LR,
                                    betas=(0.9, 0.95), weight_decay=0.1)
            opt.load_state_dict(_osd)
            net.train()
        devices.append(str(dev))
        t_start = time.time()
        chunk_capped = False
        for step in range(step + 1, n_steps + 1):
            ix = torch.randint(n_pool, (16,), generator=gen)
            aj = torch.randint(n_anc, (8,), generator=gen)
            rj = torch.randint(len(train_ids) - G1.BLOCK - 1, (8,),
                               generator=gen)
            nw = pool_a_x[ix].to(dev)
            anc = torch.cat([cons_anchor[aj],
                             torch.stack([train_ids[s: s + G1.BLOCK]
                                          for s in rj])], 0).to(dev)
            x = torch.cat([nw[:, :-1], anc[:, :-1]], 0)
            y = torch.cat([nw[:, 1:], anc[:, 1:]], 0)
            m = torch.zeros(32, x.shape[1], dtype=torch.bool, device=dev)
            m[:16] = pool_a_mask[ix].to(dev)
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
            if step % 25 == 0 or step == n_steps or SMOKE:
                sd_cpu = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
                evl.load_state_dict(sd_cpu)
                evl.eval()
                bz = G1.battery_cell(evl, g0_ids, zid)
                ce_r = G1.ce_fixed_cpu(evl, *r_eval_xy)
                state["traj"].append({"step": step, "g0_pz": bz["mean_pz"],
                                      "ce_r": ce_r,
                                      "elapsed_s": round(time.time() - t_start, 1)})
                log(f"  [{tag}] s{step:4d} g0 {bz['mean_pz']:.4f} CE_R "
                    f"{ce_r:.4f}")
            if (time.time() - t_start) > cap:
                log(f"  [{tag}] chunk {n_chunks}: burst cap {cap:.0f}s at "
                    f"s{step} — resume ckpt saved")
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
                    t_start += time.time() - t_p
        chunk_table.append({"chunk": n_chunks, "device": str(dev),
                            "seconds": round(time.time() - t_start, 1),
                            "steps_done": step, "capped": chunk_capped})
        torch.save({"model": {k: v.detach().cpu()
                              for k, v in net.state_dict().items()},
                    "opt": opt.state_dict(), "gen_state": gen.get_state(),
                    "step": step, "traj": state["traj"],
                    "n_chunks": n_chunks, "chunk_table": chunk_table},
                   resume_ck)
        if step >= n_steps:
            break
        if n_chunks >= 24:
            trims.append(f"{tag}: stopped at chunk {n_chunks} (cap-loop "
                         f"guard) at step {step} of {n_steps}")
            break
        if not SMOKE:
            log(f"[owner envelope] cooldown {COOLDOWN_S:.0f}s before "
                f"{tag} chunk {n_chunks + 1}")
            cooldown(COOLDOWN_S)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    net.eval()
    sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    return {"sd": sd_cpu, "traj": state["traj"], "steps_ran": step,
            "seed": CONS_SEED, "device": devices[0] if devices else "n/a",
            "devices": devices, "n_chunks": n_chunks,
            "chunk_table": chunk_table,
            "theta0_norm": float(torch.norm(theta0)) if theta0 is not None
            else None}


def chunked_wash(tag, net0, anchor_neutral, train_ids, itos, r_eval_xy,
                 gm12_ids, g0_ids, zid, resume_ck: Path,
                 lr: float = G1.FT_LR, seed: int = WASH_SEED) -> dict:
    """G1.g1_wash VERBATIM arithmetic (e185's noise_wash = e176n's
    finetune_freeze: per step aj(16) neutral + rj(16) random; full-token
    CE; AdamW (0.9,0.95) wd 0.1 const lr; clip 1.0; the wall projects at
    every forward on committed arms; displacement bookkeeping; CPU ARMED
    eval twin's light g-12/g0/CE_R at the ckpt steps; per-step md5 x-hashes)
    — in owner-envelope bursts with full-state resume."""
    ckpt_steps = G1.CK_WASH
    ckpt_set = set(ckpt_steps)
    n_steps = ckpt_steps[-1]
    n_anc = anchor_neutral.shape[0]
    step = 0
    traj, sds, x_hashes, deltas = [], {}, {}, {}
    zeph_checks = 0
    net, opt, gen, evl = None, None, None, None
    theta0, prev = None, None
    wall_R = getattr(net0, "R", None)
    n_chunks, chunk_table, devices = 0, [], []
    active_s = 0.0
    parked = False
    # edge: a prior process saved the FINAL state but died before returning
    if resume_ck.exists():
        _pre = torch.load(resume_ck, map_location="cpu", weights_only=False)
        if int(_pre.get("step", 0)) >= n_steps:
            log(f"  [{tag}] resume ckpt already COMPLETE at s{_pre['step']}")
            return {"sds": _pre["sds"], "traj": _pre.get("traj", []),
                    "steps_ran": n_steps, "seed": seed, "lr": lr,
                    "target_mode": "true", "zeph_violations": _pre.get("zeph", 0),
                    "x_hashes": _pre.get("x_hashes", {}),
                    "deltas": _pre.get("deltas", {}),
                    "wall_R": _pre.get("wall_R", wall_R),
                    "theta0_norm": float(torch.norm(flat_params_cpu(net0))),
                    "device": "resumed (final state)",
                    "n_chunks": _pre.get("n_chunks", 0),
                    "chunk_table": _pre.get("chunk_table", [])}
    while step < n_steps:
        n_chunks += 1
        dev = next_dev(f"{tag}-chunk{n_chunks}", parked)
        parked = parked or dev.type == "cpu"
        cap = TRAIN_CAP_GPU if dev.type == "cuda" else TRAIN_CAP_CPU
        if net is None:
            net = copy.deepcopy(net0).to(dev)
            net.train()
            opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95),
                                    weight_decay=0.1)
            gen = torch.Generator().manual_seed(seed)
            theta0 = flat_params_cpu(net)          # displacement origin (CPU)
            prev = theta0.clone()
            if resume_ck.exists():
                state = torch.load(resume_ck, map_location="cpu",
                                   weights_only=False)
                net.load_state_dict(state["model"])
                net.to(dev)
                net.train()
                opt.load_state_dict(state["opt"])
                gen.set_state(state["gen_state"])
                step = state["step"]
                traj, sds = state.get("traj", []), state.get("sds", {})
                deltas = state.get("deltas", {})
                x_hashes = state.get("x_hashes", {})
                zeph_checks = state.get("zeph", 0)
                active_s = state.get("active_s", 0.0)
                prev = flat_params_cpu(net)
                log(f"  [{tag}] RESUMED at step {step}/{n_steps} "
                    f"({len(traj)} traj rows)")
            evl = copy.deepcopy(net0).to(CPU)      # CPU eval twin (ARMED)
        else:
            _osd = opt.state_dict()
            net = net.to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95),
                                    weight_decay=0.1)
            opt.load_state_dict(_osd)
            net.train()
            prev = flat_params_cpu(net)
        devices.append(str(dev))
        t_start = time.time()
        chunk_capped = False
        for step in range(step + 1, n_steps + 1):
            aj = torch.randint(n_anc, (G1.ANCH_BS,), generator=gen)
            rj = torch.randint(len(train_ids) - G1.BLOCK - 1, (G1.RAND_BS,),
                               generator=gen)
            anc = anchor_neutral[aj]
            rnd = torch.stack([train_ids[s: s + G1.BLOCK] for s in rj])
            for w in rnd:                    # name-free VERIFY (hard-fail)
                txt = "".join(itos[int(c)] for c in w[:64]) + \
                      "".join(itos[int(c)] for c in w[192:])
                if "ZEPH" in txt:
                    zeph_checks += 1
            x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
            y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
            x_hashes[step] = hashlib.md5(
                x.contiguous().numpy().tobytes()).hexdigest()
            xd, yd = x.to(dev), y.to(dev)
            logits, _ = net(xd)                    # <- the wall projects here
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   yd.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            cur = flat_params_cpu(net)
            cum_disp = float(torch.norm(cur - theta0))
            inc_disp = float(torch.norm(cur - prev))
            prev = cur
            if step in ckpt_set:
                deltas[step] = cur - theta0
            row = {"step": step, "ce_batch": float(loss.item()),
                   "cum_disp": cum_disp, "step_disp": inc_disp,
                   "d_proj": min(cum_disp, wall_R) if wall_R else None,
                   "elapsed_s": round(active_s + time.time() - t_start, 1)}
            if step in ckpt_set:
                sd_cpu = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
                sds[step] = sd_cpu
                evl.load_state_dict(sd_cpu)
                evl.eval()
                gz = G1.battery_cell(evl, gm12_ids, zid)
                gz0 = G1.battery_cell(evl, g0_ids, zid)
                ce_r = G1.ce_fixed_cpu(evl, *r_eval_xy)
                row.update({"g_m12_mean_pz": gz["mean_pz"],
                            "g0_mean_pz": gz0["mean_pz"],
                            "frac_argmax_z": gz["frac_argmax_z"],
                            "ce_r": ce_r})
                log(f"  [{tag}] CKPT +{step:4d} g-12 {gz['mean_pz']:.4f} "
                    f"g0 {gz0['mean_pz']:.4f} CE_R {ce_r:.4f} |d| "
                    f"{cum_disp:.4f} (CE {float(loss.item()):.4f})")
            traj.append(row)
            if step % 50 == 0 and step not in ckpt_set:
                log(f"  [{tag}] s{step:4d} CE {float(loss.item()):.4f} "
                    f"|d| {cum_disp:.4f} ({row['elapsed_s']:.0f}s)")
            if (time.time() - t_start) > cap:
                log(f"  [{tag}] chunk {n_chunks}: burst cap {cap:.0f}s at "
                    f"s{step} — resume ckpt saved")
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
                    t_start += time.time() - t_p
        chunk_secs = round(time.time() - t_start, 1)
        active_s += chunk_secs
        chunk_table.append({"chunk": n_chunks, "device": str(dev),
                            "seconds": chunk_secs, "steps_done": step,
                            "capped": chunk_capped})
        torch.save({"model": {k: v.detach().cpu()
                              for k, v in net.state_dict().items()},
                    "opt": opt.state_dict(), "gen_state": gen.get_state(),
                    "step": step, "traj": traj, "sds": sds,
                    "deltas": deltas, "x_hashes": x_hashes,
                    "zeph": zeph_checks, "active_s": active_s,
                    "wall_R": wall_R, "n_chunks": n_chunks,
                    "chunk_table": chunk_table}, resume_ck)
        if step >= n_steps:
            break
        if n_chunks >= 24:
            trims.append(f"{tag}: stopped at chunk {n_chunks} (cap-loop "
                         f"guard) at step {step} of {n_steps}")
            break
        if not SMOKE:
            log(f"[owner envelope] cooldown {COOLDOWN_S:.0f}s before "
                f"{tag} chunk {n_chunks + 1}")
            cooldown(COOLDOWN_S)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    net.eval()
    return {"sds": sds, "traj": traj, "steps_ran": step, "seed": seed,
            "lr": lr, "target_mode": "true", "zeph_violations": zeph_checks,
            "x_hashes": x_hashes, "deltas": deltas, "wall_R": wall_R,
            "theta0_norm": float(torch.norm(theta0)),
            "device": devices[0] if devices else "n/a", "devices": devices,
            "n_chunks": n_chunks, "chunk_table": chunk_table}


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("g1f_smoke" if SMOKE else "g1f")
    log(f"G1F-CONS2 THE SECOND CONS SEED — the +1-crush variance (T188's "
        f"open rung) (smoke={SMOKE}) -> {rd}")
    log(f"second fresh cons draw: e001 base LOADED + e048_repro install "
        f"LOADED (gen {INST_GEN} HELD, never retrained) + e113 cons "
        f"s{CONS_STEPS} at seed {CONS_SEED} (10901 locked -> 10912 g1e "
        f"-> {CONS_SEED} THIS) — THE ONLY STOCHASTIC DELTA; wash seed "
        f"{WASH_SEED} HELD; R={R_CLAIM} raw L2; owner envelope: bursts <= "
        f"{TRAIN_CAP_GPU:.0f}s, cooldown {COOLDOWN_S:.0f}s, gate util <= "
        f"{OWNER_UTIL_C:.0f}% temp <= {OWNER_TEMP_C:.0f}C")
    set_seed(CONS_SEED)             # global init only; every RNG is its own
    write_partial(rd, "start", {
        "design": ("the g1e machinery VERBATIM with the SECOND fresh "
                   "CONS SEED (the e113 consolidation seed 10912 -> 10913 "
                   "— g1e's next free neighbor; everything else held — "
                   "the e001 base LOADED, the locked install artifact "
                   "e048_repro LOADED (gen 24313, never retrained), the "
                   "wash 10902 HELD): T188's named open rung — the "
                   "+1-CRUSH VARIANCE; is g1e's 0.2719 at +1 (the "
                   "family's only breach; the wash/root family reads "
                   "0.82-0.96) a draw or the cons axis's texture?"),
        "registered": G1F_PREDICTION, "deviations": deviations,
        "smoke": SMOKE, "device_events": device_events})

    # ---------------- reference legs (the committed record) ---------------
    ref = json.loads(REF_METRICS.read_text(encoding="utf-8"))
    ref_wall = ref["adjudication"]["wall"]
    REF = {
        "source": "runs/g1b/metrics.json", "seed": G1.FREEZE_SEED,
        "verdict": ref["adjudication"]["verdict"],
        "W1_g_m12": ref_wall["W1"]["g_m12"], "W1_min": ref_wall["W1"]["min_gm12"],
        "C_g_m12": ref_wall["C"]["g_m12"], "C_min": ref_wall["C"]["min_gm12"],
        "W1_flat_delta": ref["adjudication"]["flat_delta"],
        "traces": ref["traces"], "disp_table": ref["displacement"]["table"],
        "root_gm12": ref["traces"]["W1"][0]["gm12"],
    }
    refb = json.loads(REFB_METRICS.read_text(encoding="utf-8"))
    REF["g1bR_band"] = {
        s: {"min": refb["adjudication"]["wall"][f"W1_{s}"]["min_gm12"],
            "g300": refb["adjudication"]["wall"][f"W1_{s}"]["g_m12"]["300"],
            "at_1": refb["adjudication"]["wall"][f"W1_{s}"]["g_m12"]["1"]}
        for s in (10907, 10908)}
    refc = json.loads(REFC_METRICS.read_text(encoding="utf-8"))
    REF["g1c_root_axis"] = {
        "source": "runs/g1c_root/metrics.json",
        "verdict": refc["adjudication"]["verdict"],
        "root_gm12": refc["root_build"]["root_cells"]["gm12"],
        "ruler": refc["gates"]["G_ROOT"]["ruler"],
        "W1_trace": refc["traces"]["W1"], "C_trace": refc["traces"]["C"],
        "W1_min": refc["adjudication"]["wall"]["W1"]["min_gm12"],
        "C_min": refc["adjudication"]["wall"]["C"]["min_gm12"],
    }
    refd = json.loads(REFD_METRICS.read_text(encoding="utf-8"))
    REF["g1d_base_axis"] = {
        "source": "runs/g1d/metrics.json",
        "verdict": refd["adjudication"]["verdict"],
        "root_gm12": refd["root_build"]["root_cells"]["gm12"],
        "ruler": refd["gates"]["G_ROOT"]["ruler"],
        "W1_trace": refd["traces"]["W1"], "C_trace": refd["traces"]["C"],
        "W1_min": refd["adjudication"]["wall"]["W1"]["min_gm12"],
        "C_min": refd["adjudication"]["wall"]["C"]["min_gm12"],
    }
    refe = json.loads(REFE_METRICS.read_text(encoding="utf-8"))
    REF["g1e_cons_axis"] = {
        "source": "runs/g1e/metrics.json",
        "verdict": refe["adjudication"]["verdict"],
        "cons_seed": refe["config"]["cons_seed"],
        "root_gm12": refe["root_build"]["root_cells"]["gm12"],
        "ruler": refe["gates"]["G_ROOT"]["ruler"],
        "W1_trace": refe["traces"]["W1"], "C_trace": refe["traces"]["C"],
        "W1_min": refe["adjudication"]["wall"]["W1"]["min_gm12"],
        "W1_at_1": refe["adjudication"]["wall"]["W1"]["g_m12"]["1"],
        "C_min": refe["adjudication"]["wall"]["C"]["min_gm12"],
    }
    log(f"reference loaded: g1b (locked root cons 10901, wash "
        f"{G1.FREEZE_SEED}) W1 min {REF['W1_min']:.4f} +300 "
        f"{REF['W1_g_m12']['300']:.4f}, C dead at +2 "
        f"({REF['C_g_m12']['2']:.4f}); g1bR wash band mins "
        + "/".join(f"{v['min']:.4f}" for v in REF["g1bR_band"].values())
        + " (+1s " + "/".join(f"{v['at_1']:.4f}" for v in
                              REF["g1bR_band"].values()) + ")"
        + f"; g1c (fresh ROOT, same cons 10901) W1 min "
        f"{REF['g1c_root_axis']['W1_min']:.4f}; g1d (fresh BASE) W1 min "
        f"{REF['g1d_base_axis']['W1_min']:.4f} [TEXTURE]; g1e (FIRST cons "
        f"redraw 10912) W1 min {REF['g1e_cons_axis']['W1_min']:.4f}, +1 "
        f"{REF['g1e_cons_axis']['W1_at_1']:.4f} [THE BREACH THIS CELL "
        f"ADJUDICATES])")

    # ---------------- protocol rebuild (g1bR/g1c/g1d's main VERBATIM) ------
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

    # install windows + original-host anchor bank (g1's phase-0 construction)
    def build_win(p, host):
        return torch.cat([train_ids[p - G1.PRE: p], name_ids,
                          train_ids[p + len(host):
                                    p + len(host) + G1.POST_CAP]])

    win_i = torch.stack([build_win(p, h) for p, h in install_occ])
    inst_x = win_i.clone()                                  # (60, 256)
    anchor_full = torch.stack([train_ids[p - G1.PRE: p - G1.PRE + G1.BLOCK]
                               for p, _ in install_occ])    # (60, 256)
    inst_mask = torch.zeros(60, G1.BLOCK - 1, dtype=torch.bool)
    inst_mask[:, G1.PRE - 1: G1.PRE - 1 + len(G1.NAME)] = True
    G_INSTMASK = {"name_positions": int(inst_mask.sum()),
                  "expected": 60 * len(G1.NAME),
                  "pass": bool(int(inst_mask.sum()) == 60 * len(G1.NAME))}
    assert G_INSTMASK["pass"], f"install mask gate FAILED: {G_INSTMASK}"

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
    pool_a_x = torch.cat([jit_x[j] for j in G1.JITTERS])          # (300, 256)
    pool_a_mask = torch.cat([jit_mask[j] for j in G1.JITTERS])
    cons_anchor = anchor_full[:16]      # e113: first-16-install original bank

    # ---------------- the neutral stream (e170's construction VERBATIM) ----
    arng = random.Random(G1.E170_ANCHOR_SEED)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - G1.BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        txt = train_text[s: s + G1.BLOCK + 1]
        if any(f in txt for f in G1.ANCHOR_FORBIDDEN):
            rejections += 1
            continue
        n_starts.append(s)
    if len(n_starts) != 16:
        raise RuntimeError(f"neutral bank incomplete: {len(n_starts)}/16")
    anchor_neutral = torch.stack([train_ids[s: s + G1.BLOCK] for s in n_starts])
    G_ANCHOR = {"neutral_bank": {"seed": G1.E170_ANCHOR_SEED,
                                 "n_windows": 16, "starts": n_starts,
                                 "note": "e170's construction VERBATIM via "
                                         "g1b (= e176n arm A's stream; FIXED "
                                         "content, shared by ALL arms here)"},
                "budget": list(anchor_neutral.shape)}
    G_ANCHOR["pass"] = bool(anchor_neutral.shape == (16, G1.BLOCK))
    assert G_ANCHOR["pass"], f"anchor gate FAILED: {G_ANCHOR}"
    log(f"G_ANCHOR: neutral bank 16x{G1.BLOCK} (seed {G1.E170_ANCHOR_SEED}): PASS")

    # ---------------- batteries (e119/e176n verbatim) ----------------------
    bat_ids, held_ids = {}, {}
    for j in G1.GEOS:
        cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
        hs = [train_text[p - G1.PRE - j: p] for p, _ in held_occ]
        held_ids[j] = torch.stack([corpus.encode(c) for c in hs])
    r_eval_x, r_eval_y = G1.val_windows(val_ids, val_text, 60, G1.R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    gm12_ids = bat_ids[-12]
    g0_ids = bat_ids[0]

    gates_surg: dict = {}

    def measure(sd: dict, tag: str, lean: bool = False) -> dict:
        """g1b's measure() VERBATIM dial (e176n's = the e131 dial set) on
        evl_load (settle+disarm; g1's PIVOT)."""
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
    # PHASE 0a — THE LOADED BASE (e001) + THE LOADED INSTALL (e048_repro):
    # both locked artifacts, gated on the load (nothing retrained)
    # =====================================================================
    log("=" * 78)
    log("PHASE 0a — LOADED ARTIFACTS: e001 base + e048_repro install "
        f"(gen {INST_GEN} HELD — never retrained; g1bS7's convention)")
    base_net = G1.load_g1(GB.CKPT_DIR / BASE_CK)
    assert base_net.num_params() == GB.G1B_PARAMS, \
        f"base param count {base_net.num_params()} != {GB.G1B_PARAMS}"
    base_sd = {k: v.detach().clone() for k, v in base_net.state_dict().items()}
    raw = torch.load(GB.CKPT_DIR / BASE_CK, map_location="cpu",
                     weights_only=False)
    raw_sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    mdb = max(float((base_sd[k].float() - raw_sd[k].float()).abs().max())
              for k in raw_sd)
    base_cells = flat_cells(measure(base_sd, "g1f_base", lean=True))
    G_BASE = {"checkpoint": f"runs/checkpoints/{BASE_CK}",
              "n_tensors": len(raw_sd), "max_abs_diff_vs_file": mdb,
              "fact_free_gm12": base_cells["gm12"], "ce_r": base_cells["ce_r"],
              "fact_free": bool(base_cells["gm12"] <= 0.05),
              "params": GB.G1B_PARAMS,
              "pass": bool(mdb == 0.0 and base_cells["gm12"] <= 0.05)}
    assert G_BASE["pass"], f"G-BASE FAILED: {G_BASE}"
    log(f"G-BASE: {BASE_CK} bit-exact ({GB.G1B_PARAMS} params), fact-free "
        f"(g-12 {base_cells['gm12']:.4f}, CE_R {base_cells['ce_r']:.4f}): PASS")
    del base_net, raw, raw_sd

    # ---- the locked install artifact (the cons draw's starting state) -----
    inst_raw = torch.load(GB.CKPT_DIR / INST_CK, map_location="cpu",
                          weights_only=False)
    sd_install = {k: v.detach().clone() for k, v in
                  (inst_raw["model"] if isinstance(inst_raw, dict)
                   and "model" in inst_raw else inst_raw).items()}
    inst_net = G1.load_g1(GB.CKPT_DIR / INST_CK)
    assert inst_net.num_params() == GB.G1B_PARAMS, \
        f"install param count {inst_net.num_params()} != {GB.G1B_PARAMS}"
    md_load = max(float((sd_install[k].float()
                         - inst_net.state_dict()[k].float()).abs().max())
                  for k in sd_install)
    locked_inst_g0 = G1.battery_cell(inst_net, g0_ids, zid)["mean_pz"]
    del inst_net
    G_INST = {
        "form": (f"the LOCKED install artifact LOADED (never retrained — "
                 f"g1bS7's loaded-artifact convention): loads into the "
                 f"{GB.G1B_PARAMS} cfg clean; the same-instrument install-60 "
                 f"g0 reproduces the committed in-run reading (|d| <= "
                 f"{REUSE_CHECK_TOL}, the reuse-check convention; committed "
                 f"= {COMMITTED_LOCKED_INST_G0:.10f}, measured on THIS "
                 f"artifact by BOTH g1c and g1d)"),
        "artifact": f"runs/checkpoints/{INST_CK}", "gen_seed": INST_GEN,
        "held": True, "sd_md5": sd_md5(sd_install),
        "load_max_abs_diff": md_load,
        "install60_g0": locked_inst_g0,
        "committed_g0": COMMITTED_LOCKED_INST_G0,
        "reuse_check_d": abs(locked_inst_g0 - COMMITTED_LOCKED_INST_G0),
        "reuse_check_pass": bool(abs(locked_inst_g0
                                     - COMMITTED_LOCKED_INST_G0)
                                 <= REUSE_CHECK_TOL),
        "params": GB.G1B_PARAMS,
        "pass": bool(md_load == 0.0
                     and abs(locked_inst_g0 - COMMITTED_LOCKED_INST_G0)
                     <= REUSE_CHECK_TOL),
    }
    log(f"G-INST: {INST_CK} loaded bit-clean (gen {INST_GEN} HELD; md5 "
        f"{G_INST['sd_md5'][:10]}); install-60 g0 {locked_inst_g0:.4f} vs "
        f"the committed {COMMITTED_LOCKED_INST_G0:.4f} (|d| "
        f"{G_INST['reuse_check_d']:.4f} <= {REUSE_CHECK_TOL}): "
        f"{'PASS' if G_INST['pass'] else 'FAIL'}")
    del inst_raw
    if not G_INST["pass"] and not SMOKE:
        log("G-INST FAILED — the record completes; verdict will be TEXTURE "
            "(gate failure), nothing adjudicated")

    write_partial(rd, "artifacts_loaded", {
        "base_cells": base_cells,
        "install": {"artifact": f"runs/checkpoints/{INST_CK}",
                    "gen_seed": INST_GEN, "held": True,
                    "sd_md5": G_INST["sd_md5"],
                    "install60_g0": locked_inst_g0},
        "gates": {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                  "G_POOL": G_POOL, "G_INSTMASK": G_INSTMASK,
                  "G_ANCHOR": G_ANCHOR, "G_BASE": G_BASE, "G_INST": G_INST},
        "reference": {"source": REF["source"], "verdict": REF["verdict"],
                      "W1_min": REF["W1_min"], "root_gm12": REF["root_gm12"],
                      "g1bR_band": REF["g1bR_band"],
                      "g1c_root_axis": {k: v for k, v in
                                        REF["g1c_root_axis"].items()
                                        if k not in ("W1_trace", "C_trace")},
                      "g1d_base_axis": {k: v for k, v in
                                        REF["g1d_base_axis"].items()
                                        if k not in ("W1_trace", "C_trace")},
                      "g1e_cons_axis": {k: v for k, v in
                                        REF["g1e_cons_axis"].items()
                                        if k not in ("W1_trace", "C_trace")}},
        "device_events": device_events})

    # =====================================================================
    # PHASE 0b — THE FRESH CONS DRAW (the whole knob: seed 10901 -> 10912)
    # =====================================================================
    if not SMOKE:
        log(f"[owner envelope] cooldown {COOLDOWN_S:.0f}s before consolidate")
        cooldown(COOLDOWN_S)
    log(f"PHASE 0b — FRESH CONS: e113 VERBATIM (jitters {list(G1.JITTERS)}, "
        f"s{CONS_STEPS}, seed {CONS_SEED} (was 10901) — THE ONLY STOCHASTIC "
        f"DELTA, lr 1e-3 constant) on the LOADED locked install")
    cons = chunked_consolidate(
        "consolidate", G1.evl_load(sd_install), pool_a_x, pool_a_mask,
        cons_anchor, train_ids, g0_ids, r_eval_xy, zid,
        GB.CKPT_DIR / ("smoke_g1f_cons_resume.pt" if SMOKE
                       else "g1f_cons_resume.pt"))
    theta0 = cons["sd"]
    cons_min_g0 = min((t["g0_pz"] for t in cons["traj"]), default=None)
    cons_traj_texture = {
        "note": ("co-reported TEXTURE (g1c's amendment): the in-loop "
                 "install-60 g0 is a noisy transient read (step-to-step "
                 "swings)"),
        "min_g0_traj": cons_min_g0, "final_g0_traj":
            (cons["traj"][-1]["g0_pz"] if cons["traj"] else None),
        "traj_rows": len(cons["traj"])}

    # ---- the fresh root's dial + G-CONSDRAW + G-CONS + G-ROOT -------------
    log("=" * 78)
    root = measure(theta0, "g1f_root", lean=SMOKE)
    root_cells = flat_cells(root)
    # G-CONS (hard; the g1d precedent — failure => TEXTURE, arms run anyway)
    G_CONS = {
        "form": ("the fact survives the fresh consolidation, read on the "
                 "FINAL root (g-12 >= 0.78, g1b's own express bar; g1c's "
                 "amended form — the in-loop min demoted to co-reported "
                 "texture); HARD gate: failure => TEXTURE with the arms run "
                 "anyway (the g1d precedent — the record texture the "
                 "finding)"),
        "final_root_gm12": root_cells["gm12"], "bar": G1.EXPRESS_BAR,
        "traj_texture": cons_traj_texture,
        "pass": bool(root_cells["gm12"] >= G1.EXPRESS_BAR)}
    log(f"G-CONS: final root g-12 {root_cells['gm12']:.4f} >= "
        f"{G1.EXPRESS_BAR:.2f}: {'PASS' if G_CONS['pass'] else 'FAIL'} "
        f"(in-loop min g0 {cons_min_g0} co-reported as texture)")
    if not G_CONS["pass"] and not SMOKE:
        log("G-CONS FAILED — the record completes; verdict will be TEXTURE "
            "(the g1d precedent: the arms run anyway; the record texture "
            "the finding)")
    save_ckpt("g1f_root", theta0,
              {"desc": (f"SECOND-FRESH-CONS ROOT: e001 base LOADED + "
                        f"e048_repro install LOADED (gen {INST_GEN} HELD) "
                        f"+ e113 consolidation s{CONS_STEPS} (seed "
                        f"{CONS_SEED}: 10901 locked -> 10912 g1e -> "
                        f"{CONS_SEED} this — the second cons redraw of the "
                        f"e131 line)"),
               "install_seed": INST_GEN, "install_artifact":
                   f"runs/checkpoints/{INST_CK}",
               "cons_seed": CONS_SEED, "cons_steps": CONS_STEPS,
               "base": f"runs/checkpoints/{BASE_CK}",
               "locked_root": f"runs/checkpoints/{LOCKED_ROOT_CK}"})

    # G-CONSDRAW: the fresh root GENUINELY diverged from the locked root
    # AND from g1e's first-redraw root (the sibling distance, co-reported)
    locked_root = torch.load(GB.CKPT_DIR / LOCKED_ROOT_CK, map_location="cpu",
                             weights_only=False)
    locked_root_sd = locked_root["model"] if isinstance(locked_root, dict) \
        and "model" in locked_root else locked_root
    mdr = max(float((theta0[k].float() - locked_root_sd[k].float()).abs().max())
              for k in locked_root_sd if k in theta0)
    l2r = float(np.sqrt(sum(float(((theta0[k].float()
                                    - locked_root_sd[k].float()) ** 2).sum())
                            for k in locked_root_sd if k in theta0)))
    g1e_root_ck = GB.CKPT_DIR / "g1e_root.pt"
    l2e = None
    if g1e_root_ck.exists():      # the sibling distance (co-reported texture)
        g1e_root = torch.load(g1e_root_ck, map_location="cpu",
                              weights_only=False)
        g1e_sd = g1e_root["model"] if isinstance(g1e_root, dict) \
            and "model" in g1e_root else g1e_root
        l2e = float(np.sqrt(sum(float(((theta0[k].float()
                                        - g1e_sd[k].float()) ** 2).sum())
                                for k in g1e_sd if k in theta0)))
        del g1e_root, g1e_sd
    l2e_s = "n/a" if l2e is None else f"{l2e:.3f}"
    G_CONSDRAW = {
        "form": (f"the cons-seed redraw is REAL: seed {CONS_SEED} != 10901 "
                 f"AND != 10912 (asserted at registration; 10912's next "
                 f"free neighbor) AND the fresh root GENUINELY diverged "
                 f"from the locked e131 root (max|diff| > 0; L2 + md5 "
                 f"recorded — a bit-identical 'redraw' means the seed "
                 f"change failed; g1bS7's real gate); the distance to "
                 f"g1e's 10912 root co-reported (the sibling texture)"),
        "seed_registered": CONS_SEED, "locked_seed": 10901,
        "g1e_seed": 10912,
        "seed_genealogy": USED_FAMILY_SEEDS,
        "vs": f"runs/checkpoints/{LOCKED_ROOT_CK}",
        "vs_g1e": "runs/checkpoints/g1e_root.pt",
        "max_abs_diff": mdr, "l2_distance": l2r, "l2_vs_g1e_root": l2e,
        "sd_md5": sd_md5(theta0),
        "pass": bool(CONS_SEED != 10901 and CONS_SEED != 10912 and mdr > 0.0)}
    log(f"G-CONSDRAW: second fresh cons root vs the locked e131 root: "
        f"max|diff| {mdr:.3e}, L2 {l2r:.3f} (must be > 0); vs g1e's 10912 "
        f"root L2 {l2e_s} (sibling texture): "
        f"{'PASS' if G_CONSDRAW['pass'] else 'FAIL'}")
    del locked_root, locked_root_sd

    geos_root = {j: root_cells[k] for j, k in
                 ((-12, "gm12"), (0, "g0"), (12, "gp12"))}
    rgeo = max(geos_root, key=lambda j: geos_root[j])
    ruler_key = {-12: "gm12", 0: "g0", 12: "gp12"}[rgeo]
    G_ROOT0 = {
        "form": ("g2's frozen rule: ruler = argmax over {-12,0,+12} battery "
                 "geos at construction; ROOT_BAR 0.7 never lowered"),
        "bar": ROOT_BAR, "geos": {f"g{j:+d}": geos_root[j] for j in G1.GEOS},
        "ruler_geo": rgeo, "ruler_key": ruler_key, "ruler": geos_root[rgeo],
        "g0": geos_root[0], "gm12": geos_root[-12],
        "held30_gm12": root_cells["held30_gm12"], "ce_r": root_cells["ce_r"],
        "locked_root_gm12": G1B_ROOT_GM12,
        "express_bar_coread": G1.EXPRESS_BAR,
        "express_pass_coread": bool(root_cells["gm12"] >= G1.EXPRESS_BAR),
        "coin_flip_zone": ("g2e's lottery 0.591-0.711 + g2f's stranger "
                           "miss 0.6005 + g1d's half-expression 0.5234 "
                           "(registered)"),
        "passes": bool(geos_root[rgeo] >= ROOT_BAR),
    }
    root_gate_deviation = None
    if not G_ROOT0["passes"]:
        root_gate_deviation = {
            "what": ("the fresh-cons root's ruler landed UNDER the 0.7 bar — "
                     "the registered coin-flip caveat (the dispatch letter + "
                     "g2e's lottery + g2f's stranger miss + g1d's "
                     "half-expression): the arms run ANYWAY; the g1bS6/g1d "
                     "precedent — the record texture (the wall guarding "
                     "whatever expressed) is itself the finding; stamped on "
                     "the verdict"),
            "ruler": geos_root[rgeo], "bar": ROOT_BAR}
        log(f"G-ROOT: ruler g{rgeo:+d} = {geos_root[rgeo]:.4f} < {ROOT_BAR} "
            f"— REGISTERED DEVIATION (coin-flip zone; arms run anyway): "
            + " ".join(f"g{j:+d} {geos_root[j]:.4f}" for j in G1.GEOS))
    else:
        log(f"G-ROOT: ruler g{rgeo:+d} = {geos_root[rgeo]:.4f} >= {ROOT_BAR} "
            f"(geos " + " ".join(f"g{j:+d} {geos_root[j]:.4f}"
                                 for j in G1.GEOS)
            + f"; g-12 (the adjudicated channel) {root_cells['gm12']:.4f}; "
            f"locked root's {G1B_ROOT_GM12:.4f}): PASS")

    write_partial(rd, "root_built", {
        "root_build": {
            "recipe": (f"e001 LOADED + e048_repro LOADED (gen {INST_GEN} "
                       f"HELD, never retrained) + e113 s{CONS_STEPS} (seed "
                       f"{CONS_SEED}: 10901 locked -> 10912 g1e -> "
                       f"{CONS_SEED} THIS — the second cons redraw, the "
                       f"only stochastic delta vs BOTH)"),
            "cons_traj": cons["traj"], "cons_devices": cons["devices"],
            "cons_seed": CONS_SEED, "cons_steps": CONS_STEPS,
            "cons_chunk_table": cons["chunk_table"]},
        "root_cells": root_cells, "gates": {
            "G_CONS": G_CONS, "G_CONSDRAW": G_CONSDRAW, "G_ROOT": G_ROOT0,
            "root_gate_deviation": root_gate_deviation},
        "ckpt_inventory": CKPT_INVENTORY, "device_events": device_events})

    # the co-reported bars, registered from THIS root BEFORE any arm trains
    bar_flat = 0.9 * root_cells["gm12"]
    bar_strict = 0.9 * root_cells["gm12"] / G1B_ROOT_GM12
    log(f"THE BARS (frozen from this root, before any arm trains): PRIMARY "
        f"maintain {G1.MAINTAIN_BAR} at EVERY ckpt | strict co-report "
        f"0.9/0.9156 x root = {bar_strict:.4f} at every {FLAT_PHASE_CK} "
        f"ckpt | flat-phase texture bar 0.9 x root = {bar_flat:.4f}")

    # =====================================================================
    # THE ARMS — C (uncommitted) + W1 (commit R=0.7); the ONLY delta vs the
    # g1b reference leg is THE CONS DRAW (wash seed 10902 HELD)
    # =====================================================================
    ARM_SPECS = [
        ("C", None,
         "CONTROL — uncommitted neutral wash (e176N arm A form at this "
         "root): the per-root kill clock"),
        ("W1", R_CLAIM,
         f"WALL R={R_CLAIM} — commit({R_CLAIM}) then the seed-identical "
         f"neutral wash; step-1 weights equal to C's (the wall first acts "
         f"at forward 2)"),
    ]
    arms: dict = {}
    batteries_all: dict = {}
    G_BITROOT = {}
    for tag, R, desc in ARM_SPECS:
        log("=" * 78)
        if not SMOKE:
            log(f"[owner envelope] cooldown {COOLDOWN_S:.0f}s before {tag}")
            cooldown(COOLDOWN_S)
        log(f"ARM {tag} — {desc}")
        net0 = G1.evl_load(theta0) if R is None else G1.CommittedGPT(GB.G1B_CFG)
        if R is not None:
            net0.load_state_dict(theta0)
            net0.commit(R)
            body, _ = G1.split_anchored_sd(net0.state_dict())
            md = max(float((body[k].float() - theta0[k].float()).abs().max())
                     for k in theta0)
            anch_ok = all(torch.equal(net0._anchor(n), p.detach())
                          for n, p in net0.named_parameters())
            G_BITROOT[tag] = {"max_abs_diff": md,
                              "anchors_bit_equal": bool(anch_ok),
                              "n_anchor_tensors": net0._n_anchor_tensors,
                              "pass": bool(md == 0.0 and anch_ok)}
            assert G_BITROOT[tag]["pass"], f"{tag}: wall root != theta0"
            log(f"G_BITROOT[{tag}]: max|diff| {md:.1e}, anchors bit-equal: "
                f"PASS")
        arm = chunked_wash(
            tag, net0, anchor_neutral, train_ids, itos, r_eval_xy,
            gm12_ids, g0_ids, zid,
            GB.CKPT_DIR / ((f"smoke_g1f_{tag}_resume.pt" if SMOKE
                            else f"g1f_{tag}_resume.pt")),
            seed=WASH_SEED)
        G_DRAWFREE = {"zeph_violations": arm["zeph_violations"],
                      "pass": bool(arm["zeph_violations"] == 0)}
        assert G_DRAWFREE["pass"], f"{tag}: name token leaked into a window"
        gates_surg[f"G_DRAWFREE_{tag}"] = G_DRAWFREE
        arms[tag] = arm

        for s in sorted(arm["sds"]):
            if s == max(arm["sds"]):
                save_ckpt(f"g1f_{tag}_s{s}", arm["sds"][s],
                          {"desc": (f"g1f second-fresh-cons root + {s}-step "
                                    f"true-target neutral wash (R={R}, input "
                                    f"seed {WASH_SEED}, lr {G1.FT_LR})"),
                           "steps": int(s), "R": R, "input_seed": WASH_SEED,
                           "lr": G1.FT_LR, "target_mode": "true",
                           "base": "runs/checkpoints/g1f_root.pt"})

        # full dials: W1 at g1b's FULL_DIAL_WASH {2,50,300}; C light-only
        full_at = ([s for s in G1.FULL_DIAL_WASH if s in arm["sds"]]
                   if R is not None else [])
        batteries = {}
        for s in full_at:
            log(f"{tag} +{s} full dial")
            batteries[str(s)] = measure(arm["sds"][s], f"{tag}{s}", lean=SMOKE)
        batteries_all[tag] = batteries
        write_partial(rd, f"arm_{tag}", {
            "arms": {tag: {"desc": desc, "R": R, "wash_seed": WASH_SEED,
                           "ckpt_steps": list(G1.CK_WASH),
                           "steps_ran": arm["steps_ran"],
                           "devices": arm.get("devices"),
                           "n_chunks": arm.get("n_chunks"),
                           "chunk_table": arm.get("chunk_table"),
                           "traj": arm["traj"],
                           "missing_checkpoints": [s for s in G1.CK_WASH
                                                   if s not in arm["sds"]]}},
            "gates": {"G_BITROOT": G_BITROOT,
                      f"G_DRAWFREE_{tag}": G_DRAWFREE},
            "ckpt_inventory": CKPT_INVENTORY,
            "device_events": device_events})

    # =====================================================================
    # CONSTRUCTION-FIDELITY GATES (g1bR's form)
    # =====================================================================
    G_INPUTS = {"within_run": {
        "steps_compared": len(set(arms["C"]["x_hashes"])
                              & set(arms["W1"]["x_hashes"])),
        "identical": bool(len(set(arms["C"]["x_hashes"])
                              & set(arms["W1"]["x_hashes"])) > 0
                          and all(arms["C"]["x_hashes"][s]
                                  == arms["W1"]["x_hashes"][s]
                                  for s in (set(arms["C"]["x_hashes"])
                                            & set(arms["W1"]["x_hashes"])))),
        "note": "C and W1 share the seed-10902 stream: bit-identical "
                "per-step inputs (md5) through +300 — the commit is the "
                "only delta"},
    }
    G_INPUTS["pass"] = G_INPUTS["within_run"]["identical"]
    assert G_INPUTS["pass"], f"input stream gate FAILED: {G_INPUTS}"
    log(f"G-INPUTS: per-step inputs bit-identical C==W1 (md5, "
        f"{G_INPUTS['within_run']['steps_compared']} steps): PASS")

    body_w, _ = G1.split_anchored_sd(arms["W1"]["sds"][1])
    sd_c1 = arms["C"]["sds"][1]
    md1 = max(float((body_w[k].float() - sd_c1[k].float()).abs().max())
              for k in sd_c1)
    G_STEP1 = {"max_abs_diff": md1, "tol": 1e-4,
               "pass": bool(md1 <= 1e-4)}
    assert G_STEP1["pass"], f"G-STEP1 FAILED (wall acted at forward 1?): {G_STEP1}"
    log(f"G-STEP1: W1 step-1 body == C step-1 body (max|diff| {md1:.1e} <= "
        f"1e-4): PASS — the wall is inert until forward 2")

    # =====================================================================
    # DISPLACEMENT TABLES (e185's currency, measured)
    # =====================================================================
    def disp_rows(tag):
        return [{"step": t["step"], "ce_batch": t["ce_batch"],
                 "cum_disp": t["cum_disp"], "step_disp": t["step_disp"],
                 "d_proj": t["d_proj"],
                 **({"g_m12_light": t["g_m12_mean_pz"],
                     "ce_r_light": t["ce_r"]}
                    if "g_m12_mean_pz" in t else {})}
                for t in arms[tag]["traj"]]

    disp_table = {tag: disp_rows(tag) for tag in arms}

    # =====================================================================
    # ADJUDICATION (the frozen bars; no shopping)
    # =====================================================================
    def light_gm12(tag):
        return {t["step"]: t["g_m12_mean_pz"] for t in arms[tag]["traj"]
                if "g_m12_mean_pz" in t}

    def arm_verdict(tag, cks):
        g = light_gm12(tag)
        vals = [g[s] for s in cks if s in g]
        maintains = bool(vals and all(v >= G1.MAINTAIN_BAR for v in vals))
        dies_by_50 = bool(g.get(50, 1.0) <= G1.SHUT_BAR)
        first_under = next((s for s in cks if g.get(s, 1.0) <= G1.SHUT_BAR),
                           None)
        return {"g_m12": g, "min_gm12": min(vals) if vals else None,
                "argmin_step": (min(g, key=lambda s: g[s]) if vals else None),
                "maintains": maintains, "dies_by_50": dies_by_50,
                "first_ck_le_bar": first_under}

    fmt = lambda v: "n/a" if v is None else f"{v:.4f}"

    # ---- G-CTRL: C kills by +50
    c_gm12 = light_gm12("C")
    G_CTRL = {"bar": G1.SHUT_BAR, "gm12_at_50": c_gm12.get(50),
              "earliest_le_bar": next((s for s in G1.CK_WASH if s > 0
                                       and c_gm12.get(s, 1.0) <= G1.SHUT_BAR),
                                      None),
              "pass": bool(c_gm12.get(50, 1.0) <= G1.SHUT_BAR)}
    log(f"G-CTRL: arm C g-12 at +50 = {fmt(G_CTRL['gm12_at_50'])} "
        f"(earliest <= {G1.SHUT_BAR}: +{G_CTRL['earliest_le_bar']}): "
        f"{'PASS' if G_CTRL['pass'] else 'FAIL'}")

    # ---- G-PIN: W1's raw displacement <= R + 1.5 at every ckpt
    G_PIN = {"per_arm": {}, "fuzz_formula_at_this_size": float(
        G1.FT_LR * (GB.G1B_PARAMS ** 0.5))}
    rows = [r for r in disp_table["W1"] if "g_m12_light" in r]
    mx = max(r["cum_disp"] for r in rows) if rows else None
    G_PIN["per_arm"]["W1"] = {
        "R": R_CLAIM, "bound": R_CLAIM + G1.PIN_FUZZ_BAR,
        "max_raw_disp_at_ckpt": mx,
        "per_ckpt": {r["step"]: r["cum_disp"] for r in rows},
        "pass": bool(mx is not None and mx <= R_CLAIM + G1.PIN_FUZZ_BAR)}
    G_PIN["pass"] = G_PIN["per_arm"]["W1"]["pass"]
    log(f"G-PIN[W1]: max raw |d| at ckpt {fmt(mx)} <= "
        f"{R_CLAIM + G1.PIN_FUZZ_BAR:.2f}: "
        f"{'PASS' if G_PIN['pass'] else 'FAIL'}")

    # ---- the hard gate set (G-ROOT is the caveated gate, handled above)
    hard_gates = {
        "G-BASE": G_BASE, "G-INST": G_INST, "G-CONSDRAW": G_CONSDRAW,
        "G-CONS": G_CONS,
        "G-BITROOT": {"pass": all(v["pass"] for v in G_BITROOT.values())},
        "G-INPUTS": G_INPUTS, "G-STEP1": G_STEP1, "G-CTRL": G_CTRL,
        "G-PIN": G_PIN,
    }
    gates_pass = bool(all(g["pass"] for g in hard_gates.values())
                      and all(g["pass"] for g in
                              (G_SPLICE, G_NAMEFREE, G_POOL, G_INSTMASK,
                               G_ANCHOR)))

    # ---- the wall verdicts on the frozen bars
    wall = {tag: arm_verdict(tag, G1.CK_WASH) for tag in ("C", "W1")}
    w1g = light_gm12("W1")
    maintains_all = wall["W1"]["maintains"]
    flat_vals = [w1g[s] for s in FLAT_PHASE_CK if s in w1g]
    flat_phase_hold = bool(flat_vals
                           and all(v >= bar_flat for v in flat_vals))
    strict_vals_hold = bool(flat_vals and all(v >= bar_strict
                                              for v in flat_vals))
    flat_phase_min = min(flat_vals) if flat_vals else None
    flat_delta = (abs(w1g.get(300, float("nan")) - w1g.get(50, float("nan")))
                  if 50 in w1g and 300 in w1g else None)
    FLAT_AT_PIN = bool(flat_delta is not None and flat_delta <= G1.FLAT_BAR)
    ref_flat_phase_min = min(REF["W1_g_m12"][str(s)]
                             for s in FLAT_PHASE_CK)
    c_dies = G_CTRL["pass"]

    w1_at_1 = w1g.get(1)
    crush_breach = bool(w1_at_1 is not None and w1_at_1 < G1.MAINTAIN_BAR)
    fam_at_1 = {
        "g1b_locked": REF["W1_g_m12"]["1"],
        "g1bR_10907": REF["g1bR_band"][10907]["at_1"],
        "g1bR_10908": REF["g1bR_band"][10908]["at_1"],
        "g1c_root_axis": next(r["gm12"] for r in
                              REF["g1c_root_axis"]["W1_trace"]
                              if r["freeze_steps"] == 1),
        "g1d_base_axis": next(r["gm12"] for r in
                              REF["g1d_base_axis"]["W1_trace"]
                              if r["freeze_steps"] == 1),
        "g1e_cons1": REF["g1e_cons_axis"]["W1_at_1"],
        "g1f_cons2": w1_at_1,
    }
    bars_read = {
        "the_crush_read_at_1": w1_at_1,
        "crush_in_family_ge_maintain_bar": bool(
            w1_at_1 is not None and w1_at_1 >= G1.MAINTAIN_BAR),
        "plus1_ledger_family": fam_at_1,
        "primary_maintain_bar": G1.MAINTAIN_BAR,
        "bar_flat_phase_texture": bar_flat,
        "bar_strict_coreport": bar_strict,
        "maintains_all_ckpts": maintains_all,
        "flat_phase_hold_texture": flat_phase_hold,
        "flat_phase_min": flat_phase_min,
        "strict_coreport_hold": strict_vals_hold,
        "dip_at_2": w1g.get(2), "dip_at_1": w1_at_1,
        "flat_at_pin": FLAT_AT_PIN, "flat_delta": flat_delta,
        "c_dies_by_50": c_dies,
        "reference_flat_phase_min": ref_flat_phase_min,
    }

    if not gates_pass:
        failed = [k for k, g in hard_gates.items() if not g["pass"]]
        verdict = f"TEXTURE (GATE FAILURE: {', '.join(failed)})"
        clause = ("a hard gate failed — nothing adjudicated per the "
                  f"lineage's abort clause; failed: {failed}; the full "
                  f"record is reported (second fresh cons root g-12 "
                  f"{fmt(root_cells['gm12'])} vs the locked "
                  f"{G1B_ROOT_GM12:.4f} and g1e's 10912 sibling "
                  f"{REF['g1e_cons_axis']['root_gm12']:.4f} — the "
                  f"cons-level lottery's own reading; C trace "
                  + " -> ".join(f"+{s}:{fmt(v)}" for s, v
                                in sorted(c_gm12.items())) + ")")
        if "G-CONS" in failed:
            clause += ("; THE g1d PRECEDENT'S READING (the record texture "
                       "the finding): the arms ran anyway — the wall "
                       "guarded whatever expressed; read W1's trace against "
                       "its own root's strength (the ratio-device law, "
                       "T186): " + " -> ".join(
                           f"+{s}:{fmt(v)}" for s, v in sorted(w1g.items())))
    elif c_dies and maintains_all:
        verdict = "CRUSH-WAS-A-DRAW"
        clause = (f"the second cons draw's +1 reads IN-FAMILY ({fmt(w1_at_1)} "
                  f">= {G1.MAINTAIN_BAR}) and W1 MAINTAINS the g1b bar at "
                  f"EVERY checkpoint (min g-12 {fmt(wall['W1']['min_gm12'])} "
                  f">= {G1.MAINTAIN_BAR}, incl. the +2 dip "
                  f"{fmt(w1g.get(2))}) while C dies by "
                  f"+{G_CTRL['earliest_le_bar']} (+50 "
                  f"{fmt(G_CTRL['gm12_at_50'])}) — THE g1e BREACH WAS A "
                  f"DRAW: 0.2719 at +1 was the family's only breach, and "
                  f"this draw joins the family's +1 ledger 0.82-0.96 (g1b "
                  f"{fam_at_1['g1b_locked']:.4f}, g1bR "
                  f"{fam_at_1['g1bR_10907']:.4f}/"
                  f"{fam_at_1['g1bR_10908']:.4f}, g1c "
                  f"{fam_at_1['g1c_root_axis']:.4f}, this "
                  f"{fmt(w1_at_1)}); THE CONS AXIS JOINS THE PROTECTION "
                  f"GRID AS HOLDS AT n=2 REDRAWS (10912 + this draw, the "
                  f"locked 10901 besides); the +1 crush variance is "
                  f"ORDINARY DRAW NOISE"
                  + (f"; root gate DEVIATION stamped: ruler "
                     f"{G_ROOT0['ruler']:.4f} < {ROOT_BAR} (the registered "
                     f"coin-flip caveat)"
                     if root_gate_deviation else "")
                  + f"; STRICT CO-REPORT "
                  f"{'holds' if strict_vals_hold else 'does NOT hold'} "
                  f"(every flat-ckpt {FLAT_PHASE_CK} >= "
                  f"{bar_strict:.4f}; flat-phase min "
                  f"{fmt(flat_phase_min)} vs the texture bar "
                  f"{bar_flat:.4f}; the margin-free form the reference "
                  f"family itself straddles — never primary)"
                  + f"; FLAT-AT-PIN {FLAT_AT_PIN} (|d| {fmt(flat_delta)})")
    elif c_dies and crush_breach:
        verdict = "CRUSH-IS-TEXTURE"
        clause = (f"the second cons draw ALSO breaches at +1 ({fmt(w1_at_1)} "
                  f"< {G1.MAINTAIN_BAR}; g1e's first breach read 0.2719) — "
                  f"THE CONS AXIS CARRIES A SYSTEMATICALLY DEEPER CRUSH: "
                  f"n=2-of-3 cons draws breaching the +1 bar (against a "
                  f"wash/root family at 0.82-0.96: g1b "
                  f"{fam_at_1['g1b_locked']:.4f}, g1bR "
                  f"{fam_at_1['g1bR_10907']:.4f}/"
                  f"{fam_at_1['g1bR_10908']:.4f}, g1c "
                  f"{fam_at_1['g1c_root_axis']:.4f}) is no longer "
                  f"draw-shaped; THE GRID'S CONS CELL STAYS BOUND; the "
                  f"texture named. C dies by +{G_CTRL['earliest_le_bar']} "
                  f"(the contrast holds; G-PIN verified the ball); the "
                  f"flat phase co-reported: min {fmt(flat_phase_min)} vs "
                  f"the 0.9xroot bar {bar_flat:.4f} (T188: the flat phase "
                  f"survived in g1e — whether it survives here is read on "
                  f"the table); W1's full trace: min "
                  f"{fmt(wall['W1']['min_gm12'])} at "
                  f"+{wall['W1']['argmin_step']}, +300 {fmt(w1g.get(300))}"
                  + (f"; root gate DEVIATION stamped: ruler "
                     f"{G_ROOT0['ruler']:.4f} < {ROOT_BAR}"
                     if root_gate_deviation else ""))
    else:
        verdict = "GRADED"
        if c_dies:
            clause = (f"partial — C dies by +50 but W1 sits in the PARTIAL "
                      f"ZONE: the +1 read {fmt(w1_at_1)} vs the "
                      f"{G1.MAINTAIN_BAR} bar AND a maintain-bar breach at "
                      f"+{wall['W1']['argmin_step']} (min "
                      f"{fmt(wall['W1']['min_gm12'])}) — neither "
                      f"CRUSH-WAS-A-DRAW (W1 does not maintain everywhere) "
                      f"nor CRUSH-IS-TEXTURE (the +1 itself did not "
                      f"breach); the tables verbatim, the +1 read "
                      f"co-reported with the flat phase (min "
                      f"{fmt(flat_phase_min)} vs {bar_flat:.4f}), no bar "
                      f"shopping")
        else:
            clause = (f"partial — C did NOT die by +50 (+50 "
                      f"{fmt(G_CTRL['gm12_at_50'])} vs {G1.SHUT_BAR}): a "
                      f"wash-scale anomaly at this cons draw, not the "
                      f"crush question; the +1 read {fmt(w1_at_1)} "
                      f"co-reported with the flat phase (min "
                      f"{fmt(flat_phase_min)} vs bar {bar_flat:.4f}); W1 "
                      f"min {fmt(wall['W1']['min_gm12'])} — the tables "
                      f"verbatim, no bar shopping")

    log("=" * 78)
    log(f"G1F VERDICT: {verdict}")
    for tag in ("C", "W1"):
        g = light_gm12(tag)
        log(f"  {tag} (seed {WASH_SEED}): g-12 "
            + " -> ".join(f"+{s}:{v:.4f}" for s, v in sorted(g.items())))
    log(f"  THE +1 CRUSH READ: {fmt(w1_at_1)} vs the {G1.MAINTAIN_BAR} bar "
        f"| the +1 ledger: g1b {fam_at_1['g1b_locked']:.4f}, g1bR "
        f"{fam_at_1['g1bR_10907']:.4f}/{fam_at_1['g1bR_10908']:.4f}, g1c "
        f"{fam_at_1['g1c_root_axis']:.4f}, g1d(TEXTURE) "
        f"{fam_at_1['g1d_base_axis']:.4f}, g1e {fam_at_1['g1e_cons1']:.4f}, "
        f"g1f {fmt(w1_at_1)}")
    log(f"  root: ruler g{rgeo:+d} {geos_root[rgeo]:.4f} (bar {ROOT_BAR}); "
        f"g-12 {root_cells['gm12']:.4f} (locked {G1B_ROOT_GM12:.4f}; g1c's "
        f"same-cons sibling 0.9026; g1e's 10912 root "
        f"{REF['g1e_cons_axis']['root_gm12']:.4f})")
    log(f"  primary bar {G1.MAINTAIN_BAR} at every ckpt: "
        f"{'MAINTAINS' if maintains_all else 'BREACH'} (min "
        f"{fmt(wall['W1']['min_gm12'])}); strict co-report {bar_strict:.4f}: "
        f"{'holds' if strict_vals_hold else 'NO'}; W1 flat-phase min "
        f"{fmt(flat_phase_min)}; dip@+2 {fmt(w1g.get(2))}; FLAT-AT-PIN "
        f"{FLAT_AT_PIN} (|d| {fmt(flat_delta)})")
    log(f"  {clause}")
    log("=" * 78)

    # =====================================================================
    # OUTPUTS
    # =====================================================================
    trace = {}
    for tag in arms:
        rows = [{"freeze_steps": 0,
                 **{k: root_cells[k] for k in
                    ("gm12", "g0", "gp12", "held30_gm12", "held30_g0",
                     "ce_r", "site_read_onset", "site_read_span")}}]
        g0_light = {t["step"]: t["g0_mean_pz"] for t in arms[tag]["traj"]
                    if "g0_mean_pz" in t}
        for s in sorted(arms[tag]["sds"]):
            c = None
            if str(s) in batteries_all.get(tag, {}):
                c = flat_cells(batteries_all[tag][str(s)])
            lg = light_gm12(tag).get(s)
            if c is not None:
                rows.append({"freeze_steps": s, **c})
            elif lg is not None:
                rows.append({"freeze_steps": s, "gm12": lg,
                             "g0": g0_light.get(s),
                             "ce_r": next((t["ce_r"] for t in arms[tag]["traj"]
                                           if t["step"] == s), None)})
        trace[tag] = rows

    metrics = {
        "experiment": "g1f_cons_seed2",
        "date": common.now_iso(),
        "design": ("the g1e machinery VERBATIM with the SECOND fresh CONS "
                   "SEED (T188's named open rung — the +1-crush variance): "
                   "the e113 consolidation seed 10912 -> 10913 (the next "
                   "free neighbor) with the base (e001, LOADED FIXED), the "
                   "install (e048_repro, the LOCKED DRAW'S OWN artifact at "
                   "gen 24313, LOADED and never retrained) and the wash "
                   "(10902) ALL HELD — the only delta vs g1e is THE "
                   "300-STEP CONS STREAM (two draws deep from the locked "
                   "10901); commit(R=0.7 RAW L2) -> W1 + C"),
        "registered_prediction": G1F_PREDICTION,
        "question": ("was g1e's +1 breach (0.2719 — the family's first; "
                     "the wash/root family reads 0.82-0.96 at +1) a DRAW "
                     "or the cons axis's TEXTURE? the second draw's +1 "
                     "read against the frozen bars decides: in-family "
                     "(>= 0.5) + W1 maintains -> CRUSH-WAS-A-DRAW; a "
                     "second +1 breach -> CRUSH-IS-TEXTURE"),
        "reference": {"source": REF["source"], "seed": REF["seed"],
                      "verdict": REF["verdict"], "root_gm12": REF["root_gm12"],
                      "W1_min": REF["W1_min"],
                      "W1_g300": REF["W1_g_m12"].get("300"),
                      "W1_g_m12": REF["W1_g_m12"], "C_g_m12": REF["C_g_m12"],
                      "W1_flat_delta": REF["W1_flat_delta"],
                      "g1bR_wash_band": REF["g1bR_band"],
                      "g1c_root_axis": REF["g1c_root_axis"],
                      "g1d_base_axis": REF["g1d_base_axis"],
                      "g1e_cons_axis": REF["g1e_cons_axis"],
                      "W1_traces": REF["traces"]},
        "root_build": {
            "base": {"artifact": f"runs/checkpoints/{BASE_CK}",
                     "loaded_fixed": True,
                     "scope": "the locked 2.74M corpus base (e001, seed 42) "
                              "— the same base as g1b/g1c/the locked root"},
            "install": {"artifact": f"runs/checkpoints/{INST_CK}",
                        "gen_seed": INST_GEN, "held": True,
                        "never_retrained": True,
                        "sd_md5": G_INST["sd_md5"],
                        "install60_g0": locked_inst_g0,
                        "scope": ("THE LOCKED DRAW'S OWN install artifact "
                                  "(e043-Dmix s400/total=1000 at gen 24313) "
                                  "— loaded, never retrained (g1bS7's "
                                  "convention): the fresh root differs from "
                                  "the locked root by the CONS STREAM ALONE")},
            "consolidation": {"recipe": "e113 jitter replay VERBATIM",
                              "seed": CONS_SEED, "held": False,
                              "locked_seed": 10901,
                              "seed_genealogy": USED_FAMILY_SEEDS,
                              "steps": CONS_STEPS, "traj": cons["traj"],
                              "devices": cons["devices"],
                              "chunk_table": cons["chunk_table"]},
            "root_cells": root_cells,
        },
        "arms": {
            tag: {"desc": desc, "R": R, "wash_seed": WASH_SEED,
                  "target_mode": "true", "ckpt_steps": list(G1.CK_WASH),
                  "steps_ran": arms[tag]["steps_ran"],
                  "device": arms[tag]["device"],
                  "devices": arms[tag].get("devices"),
                  "chunk_table": arms[tag].get("chunk_table"),
                  "traj": [{k: v for k, v in t.items() if k != "d_proj"}
                           for t in arms[tag]["traj"]],
                  "missing_checkpoints": [s for s in G1.CK_WASH
                                          if s not in arms[tag]["sds"]]}
            for (tag, R, desc) in ARM_SPECS},
        "protocol": {"corpus_seed": 1337, "splice_rng": E43.SPLICE_RNG,
                     "install_mix": mix, "pre": G1.PRE, "post_cap":
                     G1.POST_CAP, "neutral_bank": G_ANCHOR,
                     "measure_dial": "e176n's measure() (the e131 dial set) "
                                     "on evl_load (settle+disarm PIVOT)"},
        "displacement": {
            "currency": (f"cumulative ||theta_t - theta_0||_2 over all "
                         f"{GB.G1B_PARAMS} trainable parameters (fp32, "
                         "measured per step) + the projected displacement "
                         "min(d, R)"),
            "table": disp_table,
            "theta0_norm": {t: arms[t]["theta0_norm"] for t in arms},
        },
        "gates": {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                  "G_POOL": G_POOL, "G_INSTMASK": G_INSTMASK,
                  "G_ANCHOR": G_ANCHOR, "G_BASE": G_BASE, "G_INST": G_INST,
                  "G_CONSDRAW": G_CONSDRAW, "G_CONS": G_CONS,
                  "G_ROOT": G_ROOT0,
                  "root_gate_deviation": root_gate_deviation,
                  "G_CTRL": G_CTRL, "G_PIN": G_PIN, "G_INPUTS": G_INPUTS,
                  "G_STEP1": G_STEP1, "G_BITROOT": G_BITROOT,
                  "G_SURG": gates_surg, "hard_pass": gates_pass},
        "traces": trace,
        "batteries": batteries_all,
        "adjudication": {
            "bars_verbatim": G1F_PREDICTION["bars_verbatim"],
            "operationalization": G1F_PREDICTION["operationalization"],
            "gates_pass": gates_pass,
            "wall": wall, "bars_read": bars_read,
            "FLAT_PHASE_CK": list(FLAT_PHASE_CK),
            "CRUSH_WAS_A_DRAW": bool(gates_pass and c_dies
                                      and maintains_all),
            "CRUSH_IS_TEXTURE": bool(gates_pass and c_dies and crush_breach
                                     and not maintains_all),
            "n_stamp": {"wash_draws": 3, "root_draws": 2,
                        "cons_draws": {"total": 3,
                                       "seeds": [10901, 10912, CONS_SEED],
                                       "ledger": ("10901 locked HOLDS "
                                                  "(g1b); 10912 g1e +1 "
                                                  "breach 0.2719 (flat "
                                                  "survived); 10913 this "
                                                  "cell -> " + verdict)},
                        "base_draws": "1 adjudicated (e001) + 1 "
                                      "observed-unadjudicated (g1d TEXTURE)"},
            "verdict": verdict, "clause": clause,
        },
        "honesty_reflex": {
            "intervention_not_logits": ("the wall IS the intervention: C "
                "and W1 share bit-identical step-0 weights (G-BITROOT "
                "max|diff| = 0.0), bit-identical per-step input streams "
                "through +300 (md5-gated, G-INPUTS) and identical targets; "
                "the only delta is the commit event — and G-STEP1 shows "
                "the commit is inert at forward 1, so the divergence "
                "begins exactly at the first projection. G-PIN verifies "
                "the geometry before any behavioral clause is read."),
            "n_scope": ("n=2 fresh cons draws now (10912 g1e + 10913 "
                "this cell; 3 cons draws with the locked 10901), n=1 wash "
                "seed per arm (the held 10902) — the +1-CRUSH VARIANCE is "
                "the object this cell measures, at n=2 redraws (T188's "
                "named rung); within-cell error bars remain out of scope "
                "(one draw per cell by design)"),
            "cons_lottery": (f"the second fresh cons root is a SIBLING "
                f"of both prior roots in the strictest sense: same base "
                f"artifact, same install artifact (bit-identical by "
                f"construction), only the 300-step jitter stream differs "
                f"(L2 {l2r:.3f} from e131; {l2e_s} from g1e's 10912 root; "
                f"g1c's install-redraw sibling sat 18.60 away); the "
                f"cons-draw EXPRESSION lottery measured in-family at "
                f"2.74M (g1e root 0.8575, this root "
                f"{root_cells['gm12']:.4f} vs the locked 0.9156) — the "
                f"open bit THIS cell measures is the +1 CRUSH DEPTH, not "
                f"expression (g1bS7's 10M PEAK-LOTTERY stays a 10M fact); "
                f"a draw under the 0.78 express bar carries the g1d "
                f"TEXTURE precedent, under the 0.7 ruler the registered "
                f"deviation (stamped, never an abort)"),
            "bar_registration": ("the three bars were frozen VERBATIM "
                "in the g1f dispatch letter BEFORE compute; the "
                "operationalization froze the maintain bar as the ONE bar "
                "carrying both the '+1 in-family' and 'W1 maintains' "
                "clauses (the +1 ckpt is inside the every-checkpoint set) "
                "and the strict 0.9/0.9156xroot form as a CO-REPORT, "
                "never primary (the reference family's own +300 values "
                "0.8999-0.9181 straddle 0.9 — a margin-free coin). No bar "
                "widened after seeing data"),
            "device": ("owner-envelope GPU bursts (<= 60 s, cooldown >= "
                "180 s, launch gate util<=20%/temp<=70C double-poll; every "
                "poll appended to runs/_envelope_log.jsonl); any CPU park "
                "recorded in device_events. The reference g1b/g1bR/g1c "
                "numbers were cuda/mixed-device — every bar adjudicates "
                "with margins (~0.9 vs 0.5/0.27) that dwarf cross-device "
                "float fuzz (G-STEP1 measures same-stream reproduction "
                "~1e-5)"),
            "within_run_scope": ("W2/W3 and the noise arms are NOT re-run "
                "here: the claim under test is the wall's CONS-robustness "
                "(W1-at-0.7 vs C); the cliff localization and noise split "
                "remain g1b's n=1 textures, and the scale axis is closed "
                "(g1bS2-8)"),
        },
        "trims": trims, "deviations": deviations,
        "device_events": device_events,
        "owner_envelope": {
            "launch_gate": (f"util <= {OWNER_UTIL_C:.0f}% AND temp <= "
                            f"{OWNER_TEMP_C:.0f}C (double-poll 5 s) AND mem "
                            f"<= {OWNER_MEM_FRAC:.0%}"),
            "burst_cap_s": TRAIN_CAP_GPU, "cooldown_s": COOLDOWN_S,
            "cpu_park_policy": ("after GPU_WAIT_MAX with no window: park to "
                                "CPU (cap 1800 s/training), loudly recorded "
                                "— 2.74M is CPU-viable; never a silent hop"),
            "poll_audit": "every owner-gate poll appended to "
                          "runs/_envelope_log.jsonl (the R61-critic trail)"},
        "ckpt_inventory": CKPT_INVENTORY,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 6, "n_head": 6, "n_embd": 192,
                   "block_size": 256, "params": GB.G1B_PARAMS,
                   "R_claim": R_CLAIM, "base_ck": BASE_CK,
                   "inst_ck": INST_CK, "inst_gen": INST_GEN,
                   "cons_seed": CONS_SEED, "wash_seed": WASH_SEED,
                   "root_bar": ROOT_BAR, "smoke": SMOKE,
                   "threads": torch.get_num_threads()},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot(rd / "cons_seed2.png", trace, disp_table, wall, verdict, clause,
         gates_pass, root_cells, geos_root, rgeo, bar_flat, bar_strict,
         bars_read, REF, root_gate_deviation, fam_at_1)

    log(f"outputs: {rd / 'metrics.json'}, {rd / 'cons_seed2.png'}, "
        f"{len(CKPT_INVENTORY)} checkpoints in runs/checkpoints/g1f_*.pt")
    log(f"total {time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plot
# THE +1-CRUSH OVERLAY: this cell's statistic (the +1 read) across the
# whole redraw family — g1b (locked cons), g1c (root axis), g1d (base
# axis, TEXTURE), g1e (the first cons redraw — THE BREACH) and this draw;
# reference trajectories read from each run's committed metrics.json.

def plot(path, trace, disp_table, wall, verdict, clause, gates_pass,
         root_cells, geos_root, rgeo, bar_flat, bar_strict, bars_read, REF,
         root_gate_deviation, fam_at_1):
    import textwrap
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
    col_fresh, col_locked, col_root = "darkorange", "seagreen", "steelblue"
    col_g1e = "tomato"
    f3 = lambda v: "n/a" if v is None else f"{v:.3f}"
    f4 = lambda v: "n/a" if v is None else f"{v:.4f}"

    def series(rows):
        pts = [(r["freeze_steps"], r["gm12"]) for r in rows]
        return [p[0] for p in pts], [p[1] for p in pts]

    # (0,0) THE +1-CRUSH OVERLAY: W1 across every stream draw
    ax = axes[0, 0]
    xs, ys = series(REF["traces"]["W1"])
    ax.plot(xs, ys, "s--", ms=6, lw=1.8, color=col_locked, alpha=0.9,
            label=f"W1 R=0.7 — LOCKED cons 10901 (g1b; +1 "
                  f"{f3(fam_at_1['g1b_locked'])})")
    xs, ys = series(REF["g1c_root_axis"]["W1_trace"])
    ax.plot(xs, ys, "^--", ms=5, lw=1.2, color=col_root, alpha=0.7,
            label=f"W1 — fresh ROOT, same cons (g1c; +1 "
                  f"{f3(fam_at_1['g1c_root_axis'])})")
    xs, ys = series(REF["g1d_base_axis"]["W1_trace"])
    ax.plot(xs, ys, "v--", ms=4, lw=1.0, color="orchid", alpha=0.55,
            label=f"W1 — fresh BASE (g1d TEXTURE; +1 "
                  f"{f3(fam_at_1['g1d_base_axis'])})")
    xs, ys = series(REF["g1e_cons_axis"]["W1_trace"])
    ax.plot(xs, ys, "D--", ms=6, lw=1.8, color=col_g1e, alpha=0.9,
            label=f"W1 — fresh cons 10912 (g1e, THE BREACH; +1 "
                  f"{f3(fam_at_1['g1e_cons1'])})")
    xs, ys = series(trace["W1"])
    ax.plot(xs, ys, "s-", ms=8, lw=2.4, color=col_fresh, alpha=0.95,
            label=f"W1 R=0.7 — SECOND cons {CONS_SEED} (g1f, THIS CELL; +1 "
                  f"{f3(fam_at_1['g1f_cons2'])})")
    xs, ys = series(REF["traces"]["C"])
    ax.plot(xs, ys, "o--", ms=4, lw=1.2, color=col_locked, alpha=0.5,
            label="C no wall — locked root (ref)")
    xs, ys = series(trace["C"])
    ax.plot(xs, ys, "o-", ms=5, lw=1.6, color=col_fresh, alpha=0.6,
            label="C no wall — second fresh cons root")
    ax.axhline(G1.MAINTAIN_BAR, ls="--", lw=1.8, color="seagreen",
               alpha=0.9, label=f"PRIMARY maintain bar {G1.MAINTAIN_BAR} "
                                "(the +1 bar)")
    ax.axhline(G1.SHUT_BAR, ls=":", lw=1.2, color="tab:purple", alpha=0.8)
    ax.axhline(bar_strict, ls=":", lw=1.2, color="gray", alpha=0.8,
               label=f"strict 0.9/0.9156xroot {bar_strict:.3f} (co-report)")
    ax.axhline(bar_flat, ls="--", lw=1.0, color="gray", alpha=0.5,
               label=f"0.9xroot flat bar {bar_flat:.3f} (texture)")
    ax.annotate(f"fresh root {root_cells['gm12']:.3f}", (0, root_cells["gm12"]),
                textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    ax.set_xlabel("neutral-wash steps from the committed root")
    ax.set_ylabel("g-12 (absolute mean p(Z), install-60 battery)")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7, loc="center right")
    ax.set_title("THE +1-CRUSH READ at the second cons seed (2.74M; e001 + "
                 "e048_repro + wash 10902 all HELD; 10901 -> 10912 -> "
                 f"{CONS_SEED})", fontsize=10)

    # (0,1) THE CELL'S STATISTIC per draw: the +1 read + the mins
    ax = axes[0, 1]
    z = lambda v: 0.0 if v is None else v
    labels = ["locked cons\n10901 (g1b ref)", "fresh ROOT\nsame cons (g1c)",
              "fresh BASE\nsame cons (g1d)",
              "fresh cons 10912\n(g1e — THE BREACH)",
              f"SECOND cons {CONS_SEED}\n(g1f — THIS CELL)"]
    at1 = [fam_at_1["g1b_locked"], fam_at_1["g1c_root_axis"],
           fam_at_1["g1d_base_axis"], fam_at_1["g1e_cons1"],
           fam_at_1["g1f_cons2"]]
    allmin = [REF["W1_min"], REF["g1c_root_axis"]["W1_min"],
              REF["g1d_base_axis"]["W1_min"], REF["g1e_cons_axis"]["W1_min"],
              wall["W1"]["min_gm12"]]
    c_min = [REF["C_min"], REF["g1c_root_axis"]["C_min"],
             REF["g1d_base_axis"]["C_min"], REF["g1e_cons_axis"]["C_min"],
             wall["C"]["min_gm12"]]
    xpos = np.arange(5)
    b1 = ax.bar(xpos - 0.26, [z(v) for v in at1], width=0.24,
                color=col_g1e, alpha=0.9,
                label="THE +1 CRUSH READ (this cell's statistic)")
    b2 = ax.bar(xpos, [z(v) for v in allmin], width=0.24, color=col_locked,
                alpha=0.9, label="W1 all-ckpt min (the maintain-bar read)")
    b3 = ax.bar(xpos + 0.26, [z(v) for v in c_min], width=0.24,
                color="crimson", alpha=0.55, label="C min (all ckpts)")
    ax.axhline(G1.MAINTAIN_BAR, ls="--", lw=1.6, color="seagreen", alpha=0.9,
               label=f"PRIMARY maintain bar {G1.MAINTAIN_BAR} (the +1 bar)")
    ax.axhline(bar_strict, ls=":", lw=1.1, color="gray", alpha=0.8,
               label=f"strict co-report {bar_strict:.3f}")
    ax.axhline(G1.SHUT_BAR, ls=":", lw=1.1, color="tab:purple", alpha=0.8)
    for bars, raw in ((b1, at1), (b2, allmin), (b3, c_min)):
        for b, v in zip(bars, raw):
            ax.annotate(f3(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("g-12")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7.5, loc="center right")
    ax.set_title("THE +1 CRUSH READ per stream draw (the wash band's own "
                 "+1s: 0.962/0.910 — in-family at every wash draw)",
                 fontsize=10)

    # (1,0) DISPLACEMENT: W1 fresh vs the wall (reference co-plotted)
    ax = axes[1, 0]
    ref_rows = [r for r in REF["disp_table"]["W1"] if "g_m12_light" in r]
    ax.plot([r["step"] for r in ref_rows], [r["cum_disp"] for r in ref_rows],
            "s--", ms=5, lw=1.4, color=col_locked, alpha=0.8,
            label="W1 locked cons (ref)")
    rows = [r for r in disp_table["W1"] if "g_m12_light" in r]
    ax.plot([r["step"] for r in rows], [r["cum_disp"] for r in rows],
            "s-", ms=7, lw=2.0, color=col_fresh, alpha=0.95,
            label=f"W1 second cons {CONS_SEED}")
    ax.axhline(R_CLAIM, ls="--", lw=1.2, color="k", alpha=0.7)
    ax.axhline(R_CLAIM + G1.PIN_FUZZ_BAR, ls=":", lw=1.0, color="k",
               alpha=0.5)
    ax.annotate(f"R={R_CLAIM} (+pin {R_CLAIM + G1.PIN_FUZZ_BAR})",
                (0.99, R_CLAIM),
                xycoords=("axes fraction", "data"), ha="right", fontsize=7.5)
    rows = [r for r in disp_table["C"] if "g_m12_light" in r]
    ax.plot([r["step"] for r in rows], [r["cum_disp"] for r in rows],
            "o--", ms=4, lw=1.2, color=col_fresh, alpha=0.5,
            label="C second cons (no wall)")
    ax.set_xlabel("wash step")
    ax.set_ylabel(r"raw $\|\theta_t-\theta_0\|_2$ at checkpoints")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("DISPLACEMENT vs the wall (G-PIN) — the control free-runs",
                 fontsize=10)

    # (1,1) THE VERDICT PANEL
    ax = axes[1, 1]
    ax.axis("off")
    y = 0.97
    ax.text(0.02, y, "G1F-CONS2 — THE SECOND CONS SEED (the +1-crush "
            "variance, T188's open rung)", fontsize=11, va="top",
            family="monospace", weight="bold")
    y -= 0.055
    ax.text(0.02, y, f"gates: {'ALL PASS' if gates_pass else 'FAILURE'} | "
            f"base e001 LOADED | install e048_repro LOADED (gen {INST_GEN} "
            f"HELD) | cons seed {CONS_SEED} (10901 locked -> 10912 g1e -> "
            f"{CONS_SEED} THIS) | root: ruler g{rgeo:+d} "
            f"{geos_root[rgeo]:.4f} (bar {ROOT_BAR}; g-12 "
            f"{root_cells['gm12']:.4f} vs locked {G1B_ROOT_GM12:.4f}; g1e's "
            f"10912 root {REF['g1e_cons_axis']['root_gm12']:.4f})"
            + ("  [ROOT-GATE DEVIATION: coin-flip zone]"
               if root_gate_deviation else ""), fontsize=7.2, va="top",
            family="monospace")
    y -= 0.034
    for tag, col in (("W1", col_fresh), ("C", "crimson")):
        g = wall[tag]["g_m12"]
        seq = " -> ".join(f"+{s}:{g[s]:.4f}" for s in sorted(g))
        ax.text(0.02, y, f"  {tag} (second fresh cons): {seq}", fontsize=6.4,
                va="top", family="monospace", color=col)
        y -= 0.028
    y -= 0.006
    ax.text(0.02, y, f"  THE +1 CRUSH READ: {f4(fam_at_1['g1f_cons2'])} vs "
            f"the {G1.MAINTAIN_BAR} bar | the ledger: g1b "
            f"{f4(fam_at_1['g1b_locked'])}, g1bR "
            f"{f4(fam_at_1['g1bR_10907'])}/{f4(fam_at_1['g1bR_10908'])}, "
            f"g1c {f4(fam_at_1['g1c_root_axis'])}, g1d(TEXTURE) "
            f"{f4(fam_at_1['g1d_base_axis'])}, g1e "
            f"{f4(fam_at_1['g1e_cons1'])}", fontsize=6.6, va="top",
            family="monospace")
    y -= 0.03
    ax.text(0.02, y, f"  PRIMARY bar {f4(G1.MAINTAIN_BAR)} at every ckpt: "
            f"{'MAINTAINS' if bars_read['maintains_all_ckpts'] else 'BREACH'}"
            f" (min {f4(wall['W1']['min_gm12'])}) | strict co-report "
            f"{f4(bar_strict)}: "
            f"{'holds' if bars_read['strict_coreport_hold'] else 'NO'} | "
            f"flat min {f4(bars_read['flat_phase_min'])} | dip@+2 "
            f"{f4(bars_read['dip_at_2'])}", fontsize=6.8, va="top",
            family="monospace")
    y -= 0.03
    ax.text(0.02, y, f"  FLAT-AT-PIN {bars_read['flat_at_pin']} (|d| "
            f"{f4(bars_read['flat_delta'])}) | C dead at +"
            f"{wall['C']['first_ck_le_bar']}", fontsize=6.8, va="top",
            family="monospace")
    y -= 0.044
    ax.text(0.02, y, f"VERDICT: {verdict}", fontsize=10, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.044
    for wd in textwrap.wrap(clause, width=92, break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=6.8, va="top", family="monospace")
        y -= 0.026

    fig.suptitle("G1F-CONS2 — the second cons seed (10901 -> 10912 -> "
                 f"{CONS_SEED}): the +1 crush — draw or texture? -> {verdict}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
