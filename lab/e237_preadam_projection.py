"""E237 — THE PRE-ADAM PROJECTION (the wind's memory cut; T215's
registered next cell; design frozen in scratch/e237_design.md INCLUDING
the R63 amendment BEFORE this script — the design note is the
registration; this docstring carries its bars VERBATIM).

THE QUESTION: e233 removed the iPhone-aligned component of each APPLIED
AdamW step: iPhone's fate more than doubled (0.188 -> 0.407) but stalled
below the carrier line (0.4525), while the removed fraction was 27x
under the honest-instrument bar. THE MECHANICAL HYPOTHESIS: Adam's
moments still drank the aligned GRADIENT each step — the optimizer's
memory re-grew the component the projection kept shaving. THE WIND HAS
MEMORY. Cut the supply, not the shipment: project the GRADIENT before
Adam sees it.

THE CELL (per the frozen design + the R63 amendment):
  1. ARM-G (w1, pre-Adam projection): wash-1's certified stream continued
     VERBATIM from pristine t=0, except each BATCH GRADIENT is projected
     off iPhone's t=0 support (component removed) BEFORE the optimizer
     step; the optimizer state advances on the projected gradient; the
     applied steps are whatever Adam makes of them.
  2. ARM-C2 (w1, hook-path control): the same stream, the same code path
     (the per-step fp64 ledger dots + the in-place gradient edit run),
     NO projection (coefficient 0 — a bitwise no-op). e233's ARM-C
     validated the scaling path; ARM-C2 validates the GRADIENT-HOOK path
     and must reproduce the committed w1 wash fate.
  3. ARM-G2 (w2, the R63 replication rider): wash-2's certified stream
     (seed 20261002, bit-exact vs e182c2_fresh_latest.pt's archived
     generator state at step 80) continued VERBATIM from pristine t=0
     with the SAME gradient-level projection.
  4. ARM-C2W2 (w2, hook-path control): ARM-C2's twin on wash-2's stream.
     DECISION (registered before compute, as the amendment requires):
     the committed w2 replay (e182c2) provides the w2 FATE references
     but DOES NOT serve as the rider's +0.05 gap control — the gap must
     be stream-matched AND code-path-matched (e233's honesty guard that
     made ARM-C2 mandatory applies verbatim on w2); hence ARM-C2W2 runs.
  Readouts identical to e233: both anchors' hr at +10/+50/+80; the five
  product-family members; the near battery; CE/ppl; the removed-L2
  ledger NOW AT THE GRADIENT LEVEL.

BARS VERBATIM (scratch/e237_design.md, frozen at dispatch BEFORE any
compute; adjudicate against exactly this; no bar shopping):
  - FATE-FLIPS-MEMORY — "ARM-G's iPhone hr(+80) >= 0.4525 (the
    registered carrier line) while ARM-C2's stays within the committed
    spread — the wind's killing is moment-carried; the seat promoted
    from between-marker-and-carrier to CARRIER"
  - STILL-BELOW — "ARM-G's iPhone improves over e233's 0.407 but stalls
    under the line — the memory story is partial; the DOSE ladder
    (amplified removals x2/x5 at the gradient level) is named as the
    map cell"
  - NO-GAIN — "ARM-G ~ e233's ARM-P (or worse) — the applied-step
    projection was already the effective cut; H-null: the stalling
    lives elsewhere (Adam's normalized geometry; g14's MULTI-COMPONENT
    echo)"
  - ANY — "the trajectories verbatim, no inflation"

PROMOTION RULE (the R63 amendment, VERBATIM): "FATE-FLIPS-MEMORY
requires BOTH (i) w1's ARM-G crossing its frozen 0.4525 line AND (ii)
the w2 rider showing the same-direction sparing (iPhone's P2-vs-control
gap >= +0.05 while Gmail-P2 ~ control) — the effect must replicate
across streams before 'the wind has memory' is quoted as mechanism."

ANCHORING DISCLOSURE (standing, rides EVERY bar read, VERBATIM): "the
frozen w1 lines (0.4525, the 0.407 reference, the 0.342 spread max) are
single-stream numbers against iPhone's natural 1.9x cross-wash swing;
every bar read carries this caveat verbatim."

REGISTERED PREDICTIONS (no retrofit):
  (a) "The removed-L2 at the gradient level is ~1x-2x the applied level
      (the moments smooth) — if it exceeds 1% of gradient norms, the
      honest-instrument bar is CLEARED for the first time in this family
      (the dose finally adequate)."
  (b) "Under FATE-FLIPS-MEMORY, the family clause re-tests (e233 read
      it negative at the applied level; if the memory cut flips only the
      anchor again, the seat is confirmed probe-specific)."
  (c) "Gmail and the batteries stay at C2~committed in every branch
      (the projection's specificity is the instrument's own control)."

OPERATIONALIZATIONS (frozen here BEFORE compute; they fix the clauses,
they do not move the bars):
  * hr := p(state)/p(t=0) per probe; p = p(answer first token | the
    probe's VERBATIM 2-shot prompt), CPU fp32 batch-1, the committed
    instrument (e182/e182c's batteries, module import; t=0 re-certified
    against all THREE committed records, dp <= 0.010, e226's G_BATT).
  * the BATCH GRADIENT that is projected := the post-clip gradient
    (backward -> clip_grad_norm_ 1.0 VERBATIM -> PROJECT -> opt.step()):
    the clip is the verbatim wash machinery's last operation before the
    optimizer, so the projection is the LAST operation before Adam —
    everything Adam consumes (both moments + the update) is supply-cut.
    The pre-clip norm is co-recorded every step (the ordering choice is
    disclosed; the removed fraction is ~0.04-0.2%, so project-then-clip
    and clip-then-project differ at second order — the ledger settles
    it empirically: preclip_norm vs clip_norm per step).
  * the projection: g' = g - <g, s_iPhone> * s_iPhone (the component
    REMOVED, norm NOT rescaled); all dots/norms fp64 (chunked
    per-parameter fp64 dots — e226's instrument finding, e233's port).
  * ARM-C2/ARM-C2W2: the identical per-step machinery (fp64 dots +
    ledger rows recorded — the VERBATIM wash's own aligned-gradient
    ledger, the "what the moments drank" record) with the projection
    coefficient forced to 0.0 (0.0 * s is a bitwise no-op on p.grad —
    the trajectory is the verbatim GPU wash).
  * "ARM-G's iPhone hr(+80) >= 0.4525" — the line is RUNTIME-READ as
    0.5 x Gmail-w1's committed hr (e226's committed record; the literal
    0.4525 is the design's rounded rendering of 0.45247).
  * "ARM-C2's stays within the committed spread" := spread_I_lo <=
    hr(C2, iPhone, +80) <= spread_I_hi (the three-wash committed iPhone
    spread, runtime-read).
  * "improves over e233's 0.407" (STILL-BELOW) := hr(G, iPhone, +80) >=
    e233_ARM_P_hr + 0.02 — the 0.02 tolerance is registered HERE,
    BEFORE compute, to fix the "~"/"improves" boundary (iPhone's
    cross-wash swing is 0.16; 0.02 is the conservative cut); the raw
    number rides every read (no tolerance hides it).
  * "ARM-G ~ e233's ARM-P (or worse)" (NO-GAIN) := hr(G, iPhone, +80) <
    e233_ARM_P_hr + 0.02.
  * Adjudication order: FATE-FLIPS-MEMORY -> STILL-BELOW -> NO-GAIN ->
    ANY; every boolean reported regardless. If ARM-G crosses the line
    but ARM-C2 exits the committed spread, the FLIPS conjunction fails
    and the read is ANY (the trajectories verbatim).
  * the rider (ii): "iPhone's P2-vs-control gap" := hr(G2, iPhone, +80)
    - hr(C2W2, iPhone, +80) (stream-matched, code-path-matched);
    "Gmail-P2 ~ control" := |hr(G2, Gmail, +80) - hr(C2W2, Gmail, +80)|
    <= 0.05 (the amendment's own scale). rider_fire := gap_iPhone >=
    +0.05 AND |gap_Gmail| <= 0.05. promotion := FLIPS-fire AND
    rider_fire (both streams; verbatim per the amendment).
  * prediction (a)'s "> 1% of gradient norms" := median over ARM-G's
    steps of |<g_clip, s>| / ||g_clip|| > 0.01 -> the honest-instrument
    bar CLEARED; < 0.01 -> the verdict is CO-STAMPED UNDERPOWERED (the
    stamp discloses, it does not move the bar) — e233's clause, now at
    the gradient level.
  * the family clause (b): the product family's five non-anchor members
    (Xbox/Chrome/iPad/iTunes/PlayStation), hr at +80 per arm vs each
    stream's OWN committed reference (w1 arms vs w1, w2 arms vs w2);
    the literal near battery (near-uscap) co-reported. No new bar.
  * the wash-health read: held-out bank ppl at +80 < bank ppl at t=0
    AND mean in-batch CE over the last 10 steps < over the first 10,
    ALL FOUR arms (G_WASHHEALTH).
  * the w2 stream := e182c2's frozen window-draw stream (seed 20261002)
    reproduced from the archived seed and certified BIT-EXACTLY against
    the generator state archived at step 80 in e182c2_fresh_latest.pt
    (e226's G_DRAWS convention, applied to BOTH streams here).

HONESTY GUARDS (the design's, verbatim in force):
  * the gradient-hook changes the code path e233 validated — ARM-C2
    (no projection, hook installed but identity) is mandatory, not
    optional; the same guard makes ARM-C2W2 mandatory for the rider.
  * n=1 organism, n=1 stream per arm (the standing caveat of this
    family); the committed references remain the replication base.
  * the projection direction is t=0-fixed (the standing disclosure);
    support rotation was sub-bar in e226.
  * the intervention changes the trajectory, so after step 1 each arm
    is its own wash (g12's precedent); the readouts compare FATES, not
    trajectories.
  * DEVICE TEXTURE: the arms train GPU fp32 (TF32 OFF); w1's committed
    reference is e182c's CPU fp32 replay; w2/w3 are GPU fp32 — fates,
    not bit values, are compared (e217's precedent; the C2 arms are the
    device-matched controls).

COMPUTE ENVELOPE (dispatch): the GPU lane is OWNED (no concurrent GPU
jobs; the owner's max-priority window ACTIVE per STATE.json: bursts
<=180 s, cooldowns 30-60 s, temp-aware, never past 85C). This cell:
bursts <= 175 s wall AND <= 40 steps; cooldown >= 45 s; every launch
gated by common.gpu_ok() (util <= 85%, temp <= 80C, mem <= 85%),
double-polled; MID-BURST THERMAL GUARD AT A 78C MARGIN WITH PER-STEP
POLLS FROM THE FIRST BURST (the e233 lesson, default here — not a
mid-run tightening: the 5090 ramps ~9C/s at burst start); every poll
logged to runs/_envelope_log.jsonl. CPU fp32 probing between bursts.
Resumable state after every burst; progressive PARTIAL metrics writes;
restore-safe journal merge (the e233 journal-clobber lesson). No
NOTES/THINKING/QUEUE/STATE edits (dispatch). Smoke via E237_SMOKE=1
(3 steps, own smoke dir, nothing adjudicated; BOTH draw-stream
certifications run even in smoke — CPU-only, cheap).

Run:  cd lab && python e237_preadam_projection.py
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:                                            # Windows console safety
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                               # noqa: BLE001
    pass

import torch                                          # noqa: E402

import torch.nn.functional as F                       # noqa: E402

import common                                          # noqa: E402
from common import now_iso, run_dir, save_json          # noqa: E402

import e182c_forgetting_control as e1                  # noqa: E402 — phase-1 machinery, VERBATIM
import e182c2_template as e2                           # noqa: E402 — the template battery + opt-state helpers, VERBATIM
import e226_interior as e3                             # noqa: E402 — the support conventions, VERBATIM

# e182c sets threads 8, e226 resets to 4 (its shared-box envelope); the
# owner's max-priority window is ACTIVE (STATE.json) and the box is ours
# — restore e182's 8-thread convention for the CPU probing phases.
THREADS = 8
torch.set_num_threads(THREADS)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

try:
    import psutil                                     # noqa: E402
except ImportError:
    psutil = None

SMOKE = os.environ.get("E237_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e237_smoke" if SMOKE else "e237"

T0 = time.time()
RD = run_dir(NAME)
LOG_PATH = RD / "run.log"
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


# ------------------------------------------------------------ the cell
W1_SEED = 18202                   # wash 1's frozen stream (e182's own)
W2_SEED = 20261002                # wash 2's frozen stream (e182c2's own)
N_STEPS = 3 if SMOKE else 80
CK_STEPS: tuple[int, ...] = (1, 2, 3) if SMOKE else (10, 50, 80)
FD_EPS = (0.05,) if SMOKE else (0.02, 0.05)   # e204's/e226's sizes (L2)
TOL_T0_DP = 0.010                  # e226's G_BATT tolerance
CHUNK_DOT_MIN = 4_000_000          # per-param fp64 dots on GPU are one op
                                          # below this; chunked above
IMPROVE_TOL = 0.02                 # registered BEFORE compute (see
                                          # operationalizations): the
                                          # "~"/"improves" boundary
RIDER_TOL = 0.05                   # the amendment's own rider scale

# the owner's ACTIVE max-priority window (STATE.json), temp-aware; the
# e233 lessons are the DEFAULTS here (per-step polls from the first
# burst at the 78C margin — no mid-run tightening should be needed):
BURST_MAX_S = 175.0                # < the 180 s hard lab cap
BURST_MAX_STEPS = 40
COOLDOWN_S = 45.0                  # the 30-60 s window
TEMP_BURST_END = 84.0              # (recorded) the hard never-past-85 line
TEMP_EARLY_END = 78.0              # per-step-poll burst-end margin
POLL_EVERY = 1                     # PER-STEP mid-burst temp polls
LAUNCH_POLL_GAP_S = 5.0

ANCHOR_G = "The email service made by Google->Gmail"
ANCHOR_I = "The phone made by Apple->iPhone"
IPHONE_FAMILY = (                 # e216's family6=product, non-anchor
    "The gaming console made by Microsoft->Xbox",
    "The web browser made by Google->Chrome",
    "The tablet made by Apple->iPad",
    "The music store made by Apple->iTunes",
    "The game console made by Sony->PlayStation",
)
ARMS = (                          # (tag, stream, project?)
    ("G", "w1", True),
    ("C2", "w1", False),
    ("G2", "w2", True),
    ("C2W2", "w2", False),
)
ARM_DESC = {
    "G": "w1 pre-Adam projection (the supply cut)",
    "C2": "w1 hook-path control (no projection; the verbatim GPU wash)",
    "G2": "w2 pre-Adam projection (the R63 replication rider)",
    "C2W2": "w2 hook-path control (no projection)",
}

# committed records (runtime-read, never transcribed)
E182C_M = common.REPO / "runs" / "e182c" / "metrics.json"
E182C2_J = common.REPO / "runs" / "e182c2" / "journal_p2.json"
E217_M = common.REPO / "runs" / "e217" / "metrics.json"
E216_M = common.REPO / "runs" / "e216" / "metrics.json"
E226_M = common.REPO / "runs" / "e226" / "metrics.json"
E233_M = common.REPO / "runs" / "e233" / "metrics.json"
W1_LATEST = common.REPO / "runs" / "checkpoints" / "e182c_replay_latest.pt"
W2_LATEST = common.REPO / "runs" / "checkpoints" / "e182c2_fresh_latest.pt"
CK_DIR = common.REPO / "runs" / "checkpoints"

ANCHORING_DISCLOSURE = (
    "ANCHORING DISCLOSURE (standing): the frozen w1 lines (0.4525, the "
    "0.407 reference, the 0.342 spread max) are single-stream numbers "
    "against iPhone's natural 1.9x cross-wash swing; every bar read "
    "carries this caveat verbatim.")

REGISTERED_PREDICTION = {
    "bars_verbatim": {
        "FATE-FLIPS-MEMORY": "ARM-G's iPhone hr(+80) >= 0.4525 (the "
            "registered carrier line) while ARM-C2's stays within the "
            "committed spread — the wind's killing is moment-carried; "
            "the seat promoted from between-marker-and-carrier to "
            "CARRIER",
        "STILL-BELOW": "ARM-G's iPhone improves over e233's 0.407 but "
            "stalls under the line — the memory story is partial; the "
            "DOSE ladder (amplified removals x2/x5 at the gradient "
            "level) is named as the map cell",
        "NO-GAIN": "ARM-G ~ e233's ARM-P (or worse) — the applied-step "
            "projection was already the effective cut; H-null: the "
            "stalling lives elsewhere (Adam's normalized geometry; "
            "g14's MULTI-COMPONENT echo)",
        "ANY": "the trajectories verbatim, no inflation",
    },
    "promotion_rule_verbatim": (
        "FATE-FLIPS-MEMORY requires BOTH (i) w1's ARM-G crossing its "
        "frozen 0.4525 line AND (ii) the w2 rider showing the "
        "same-direction sparing (iPhone's P2-vs-control gap >= +0.05 "
        "while Gmail-P2 ~ control) — the effect must replicate across "
        "streams before 'the wind has memory' is quoted as mechanism."),
    "anchoring_disclosure_verbatim": ANCHORING_DISCLOSURE,
    "predictions_verbatim": {
        "(a)": "The removed-L2 at the gradient level is ~1x-2x the "
            "applied level (the moments smooth) — if it exceeds 1% of "
            "gradient norms, the honest-instrument bar is CLEARED for "
            "the first time in this family (the dose finally adequate).",
        "(b)": "Under FATE-FLIPS-MEMORY, the family clause re-tests "
            "(e233 read it negative at the applied level; if the memory "
            "cut flips only the anchor again, the seat is confirmed "
            "probe-specific).",
        "(c)": "Gmail and the batteries stay at C2~committed in every "
            "branch (the projection's specificity is the instrument's "
            "own control).",
    },
    "registration": "bars frozen VERBATIM from scratch/e237_design.md "
        "(the frozen design note + its R63 amendment, committed at "
        "b2d696b/5e08e32 BEFORE any compute; the dispatch brief is the "
        "registration); adjudicate against exactly this; no bar shopping",
}

trims: list[str] = []
deviations: list[str] = [
    "The pre-Adam projection point (registered before compute): the "
    "projection is applied to the POST-CLIP batch gradient (backward -> "
    "clip 1.0 VERBATIM -> project -> opt.step()) — the clip is the "
    "verbatim machinery's last operation before the optimizer, so the "
    "cut is the last operation before Adam: both moments AND the update "
    "never see the aligned component. The pre-clip norm is co-recorded "
    "every step; e233's applied-level removed fractions were ~0.04%, so "
    "clip-vs-project ordering differs at second order (the ledger "
    "settles it empirically).",
    "e233's snapshot->opt.step()->restore->modify-delta machinery is "
    "REPLACED by the direct gradient edit (p.grad.add_(-coef*s) before "
    "opt.step()) — this is THE CELL (the supply cut); the optimizer "
    "state now advances on the projected gradient (e233's disclosed "
    "opposite: its state advanced on the verbatim gradient).",
    "ARM-C2/ARM-C2W2 run the identical per-step code path (fp64 ledger "
    "dots + the in-place gradient edit with coefficient 0.0 — a bitwise "
    "no-op): they validate the hook path against the committed fates "
    "AND double as the verbatim wash's own aligned-gradient ledger.",
    "ARM-C2W2 DECISION (the amendment's 'state which'): the committed "
    "w2 replay serves as the w2 FATE REFERENCE but NOT as the rider's "
    "+0.05 gap control — the gap must be stream-matched AND "
    "code-path-matched; hence all four arms run.",
    "The IMPROVE_TOL 0.02 boundary (registered before compute): fixes "
    "the STILL-BELOW 'improves' / NO-GAIN '~' clauses numerically "
    "without moving any bar; the raw hr rides every read.",
    "The rider's '~ control' tolerance is the amendment's own 0.05 "
    "scale (|gap_Gmail| <= 0.05), applied symmetrically.",
    "DEVICE TEXTURE: the arms train GPU fp32 (TF32 OFF); w1's committed "
    "reference is a CPU fp32 replay and w2/w3 are GPU fp32 — fates, not "
    "bit values, are compared (e217's precedent; the C2 arms are the "
    "device-matched controls).",
    "CPU probing (batteries + bank ppl) uses the committed CPU fp32 "
    "instrument between GPU bursts (threads 8, e182's convention — the "
    "owner's max-priority window is active and the box is ours).",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch).",
    "Smoke mode (E237_SMOKE=1): 3 steps, checkpoints {1,2,3}, own smoke "
    "dir; BOTH draw-stream certifications run (CPU-only, cheap); "
    "nothing adjudicated or gated (SMOKE stamp).",
]


# ------------------------------------------------------------ envelope
# (e233's guards, ported verbatim with THIS run's NAME; per-step polls
# at the 78C margin are the DEFAULT from the first burst — the e233
# lesson, not a mid-run tightening.)

def cpu_load_check(tag: str) -> dict:
    if psutil is None:
        return {"tag": tag, "psutil": "absent"}
    rec = {"tag": tag,
           "cpu_percent": psutil.cpu_percent(interval=0.5),
           "ram_avail_gb": round(psutil.virtual_memory().available / 2**30, 1),
           "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 1)}
    log(f"  [load] {tag}: cpu {rec['cpu_percent']}% "
        f"ram_avail {rec['ram_avail_gb']} GB")
    return rec


def gpu_poll(tag: str) -> dict:
    s = common.gpu_status()
    ok = s["util"] <= common.GPU_UTIL_CEIL and s["temp"] <= common.GPU_TEMP_CEIL
    common._log_envelope_poll(f"{NAME}:{tag}", s["util"], s["temp"], ok)
    log(f"  [gpu:{tag}] util {s['util']:.0f}% temp {s['temp']:.0f}C "
        f"mem {s['mem_used']:.0f}/{s['mem_total']:.0f}MB "
        f"power {s['power']:.1f}W -> {'OK' if ok else 'HOLD'}")
    return {"poll": s, "ok": bool(ok)}


def wait_gpu_free(tag: str, max_wait_s: float = 1800.0) -> list[dict]:
    """Launch gate: common.gpu_ok() semantics, first launch double-polled
    (>=5 s apart). The owner's max-priority window is active — short
    waits only."""
    polls = [gpu_poll(f"{tag}#1")]
    time.sleep(LAUNCH_POLL_GAP_S)
    polls.append(gpu_poll(f"{tag}#2"))
    t0w = time.time()
    while not (polls[-2]["ok"] and polls[-1]["ok"]):
        if time.time() - t0w > max_wait_s:
            raise RuntimeError(f"GPU window never opened for {tag}")
        log(f"  [gpu:{tag}] waiting 20s for the envelope "
            f"(util<={common.GPU_UTIL_CEIL:.0f}% "
            f"temp<={common.GPU_TEMP_CEIL:.0f}C)")
        time.sleep(20.0)
        polls.append(gpu_poll(f"{tag}#w"))
    return polls


def burst_temp_check(tag: str) -> bool:
    """Mid-burst thermal guard: True = keep going. PER-STEP polls from
    the FIRST burst (the laptop 5090 ramps ~9C/s at burst start during
    fan spin-up — e233's lesson, the default here); the burst ends at
    the 78C margin so a one-step sensor jump stays under the
    never-past-85C line."""
    s = common.gpu_status()
    common._log_envelope_poll(f"{NAME}:{tag}:mid", s["util"], s["temp"],
                              s["temp"] < TEMP_EARLY_END)
    if s["temp"] >= TEMP_EARLY_END:
        log(f"  [gpu:{tag}:mid] temp {s['temp']:.0f}C >= "
            f"{TEMP_EARLY_END:.0f}C margin — ending burst "
            f"(85C line protected)")
        return False
    return True


# ------------------------------------------------------------ fp64 math
# e226's instrument finding: fp32 accumulation over 124M coords drifts
# ~0.6% — the SAME order as the registered 1% honest-instrument bar, so
# every ledger dot/norm is fp64 (e233's port, verbatim).

def _dot64(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """fp64 dot of two flat fp32 GPU tensors (chunked casts for the
    embedding-scale tensors to bound fp64 temporaries)."""
    n = a.numel()
    if n <= CHUNK_DOT_MIN:
        return torch.dot(a.double(), b.double())
    tot = torch.zeros((), dtype=torch.float64, device=a.device)
    for s in range(0, n, CHUNK_DOT_MIN):
        sl = slice(s, min(s + CHUNK_DOT_MIN, n))
        tot += torch.dot(a[sl].double(), b[sl].double())
    return tot


# ------------------------------------------------------------ the arm

def run_arm(tag: str, stream: str, project: bool, net0, train_ids, offs,
            s_chunks_gpu, probes_bats, on_ckpt) -> dict:
    """One arm of the cell.

    G/G2 (project=True): after the verbatim backward+clip, each batch
    gradient's component along iPhone's t=0 support is REMOVED
    (coefficient -<g, s>; norm NOT rescaled) BEFORE opt.step() — the
    optimizer state (both moments) advances on the projected gradient;
    the applied step is whatever Adam makes of it.
    C2/C2W2 (project=False): the identical per-step machinery with
    coefficient 0.0 (a bitwise no-op) — the verbatim GPU wash through
    the hook path, its own aligned-gradient ledger recorded.

    VERBATIM wash machinery: AdamW(0.9,0.95) wd 0.1 constant lr 5e-5,
    clip 1.0, full-token CE, batch 8 x ctx 512, the stream's certified
    window draws; GPU fp32 (TF32 OFF) in <=175 s / <=40-step bursts.
    Resumable state after every burst; per-checkpoint weight archives."""
    seed = W1_SEED if stream == "w1" else W2_SEED
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    net = copy.deepcopy(net0).to(dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=e1.LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    params = list(net.parameters())
    step = 0
    ledger: dict[int, dict] = {}
    latest = CK_DIR / f"{NAME}_{tag}_latest.pt"
    if latest.exists() and not SMOKE:
        st = torch.load(latest, map_location=CPU, weights_only=False)
        if st["step"] > 0:
            net.load_state_dict(st["model"])
            net.to(dev)
            opt.load_state_dict(st["opt"])
            e2.opt_state_dev_fix(opt, dev)
            step = int(st["step"])
            ledger = {int(k): v for k, v in st["ledger"].items()}
            log(f"ARM-{tag}: RESUMED from step {step} "
                f"({len(ledger)} ledger rows)")
    ckpt_set = set(CK_STEPS)
    envelope_polls: list[dict] = []
    burst_id = step + 1
    t_burst = None
    n_burst = 0
    while step < N_STEPS:
        if t_burst is None:
            envelope_polls += wait_gpu_free(f"{tag}burst{burst_id}")
            t_burst = time.time()
            n_burst = 0
            log(f"ARM-{tag}: burst {burst_id} opens at s{step + 1}")
        step += 1
        n_burst += 1
        off = offs[step - 1]
        x = torch.stack([train_ids[o: o + e1.SEQ] for o in off]).to(dev)
        y = torch.stack([train_ids[o + 1: o + 1 + e1.SEQ]
                         for o in off]).to(dev)
        logits = net(input_ids=x).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # ---- the pre-clip norm (the ordering disclosure's co-record)
        gn64_raw = torch.zeros((), dtype=torch.float64, device=dev)
        for p in params:
            g = p.grad.detach().reshape(-1)
            gn64_raw += _dot64(g, g)
        gnorm_raw = float(math.sqrt(gn64_raw.item()))
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        # ---- THE INTERVENTION: project the (post-clip) batch gradient
        # off iPhone's t=0 support BEFORE the optimizer step. The fp64
        # dots are the ledger; the in-place edit is the ONLY write (for
        # the C2 arms its coefficient is 0.0 — a bitwise no-op).
        c64 = torch.zeros((), dtype=torch.float64, device=dev)
        gn64 = torch.zeros((), dtype=torch.float64, device=dev)
        for p, s_c in zip(params, s_chunks_gpu):
            g = p.grad.detach().reshape(-1)
            c64 += _dot64(g, s_c)
            gn64 += _dot64(g, g)
        gnorm = float(math.sqrt(gn64.item()))
        cf = float(c64.item())
        coef32 = torch.tensor(cf if project else 0.0,
                              dtype=torch.float32, device=dev)
        with torch.no_grad():
            for p, s_c in zip(params, s_chunks_gpu):
                p.grad.add_((-coef32 * s_c).reshape(p.shape))
        gp64 = torch.zeros((), dtype=torch.float64, device=dev)
        for p in params:
            g = p.grad.detach().reshape(-1)
            gp64 += _dot64(g, g)
        gpnorm = float(math.sqrt(gp64.item()))
        opt.step()          # both moments + the update see the cut
        ledger[step] = {"ce": float(loss.item()),
                        "gnorm_preclip": gnorm_raw, "gnorm": gnorm,
                        "dot_s": cf, "gpnorm": gpnorm,
                        "removed_frac": (abs(cf) / gnorm if gnorm else 0.0),
                        "removed_l2": abs(cf),
                        "clip_bound": bool(gnorm_raw > 1.0)}
        if step % 10 == 0 or step in ckpt_set or step == N_STEPS:
            r = ledger[step]
            log(f"  [ARM-{tag}] s{step:3d}/{N_STEPS} CE {r['ce']:.4f} "
                f"|g| {r['gnorm']:.4f} (preclip {r['gnorm_preclip']:.3f}"
                f"{' CLIP' if r['clip_bound'] else ''}) |<g,s>| "
                f"{r['removed_l2']:.3e} ({r['removed_frac']*100:.3f}%) "
                f"-> |g'| {r['gpnorm']:.4f} "
                f"({time.time() - t_burst:.1f}s into burst)")
        # ---- burst bookkeeping / thermal guard (PER-STEP polls)
        temp_ok = True
        if n_burst % POLL_EVERY == 0:
            temp_ok = burst_temp_check(f"{tag}burst{burst_id}")
        hit_ckpt = step in ckpt_set
        burst_over = (n_burst >= BURST_MAX_STEPS
                      or time.time() - t_burst >= BURST_MAX_S
                      or hit_ckpt or step >= N_STEPS or not temp_ok)
        if not burst_over:
            continue
        t_end = time.time()
        log(f"  [ARM-{tag}] burst {burst_id} done: {n_burst} steps in "
            f"{t_end - t_burst:.1f}s (caps {BURST_MAX_S:.0f}s/"
            f"{BURST_MAX_STEPS} steps), now at s{step}")
        sd = {k: v.detach().to("cpu", torch.float32).clone()
              for k, v in net.state_dict().items()}
        torch.save({"model": sd, "opt": e2._opt_state_to_cpu(opt.state_dict()),
                    "step": step, "ledger": ledger,
                    "meta": {"experiment": NAME, "arm": tag,
                             "seed": seed, "lr": e1.LR,
                             "desc": f"e237 ARM-{tag} ({ARM_DESC[tag]}, "
                                     f"stream {stream} seed {seed}) GPU "
                                     f"fp32, step {step}"}},
                   latest)
        if hit_ckpt:
            torch.save({"model": sd,
                        "meta": {"experiment": NAME, "arm": tag, "step": step,
                                 "lr": e1.LR, "seed": seed,
                                 "desc": f"e237 ARM-{tag} checkpoint, "
                                         f"step {step}",
                                 "base": e1.MODEL_REPO,
                                 "revision": e1.MODEL_REV}},
                       CK_DIR / f"{NAME}_{tag}_s{step}.pt")
        if hit_ckpt:
            on_ckpt(tag, step, sd, ledger[step]["ce"])
        del sd
        # cooldown (>= 45 s since burst end; the CPU probing above counts)
        remain = COOLDOWN_S - (time.time() - t_end)
        if remain > 0 and step < N_STEPS:
            log(f"  [thermal] cooldown {remain:.0f}s "
                f"(max-priority window: {COOLDOWN_S:.0f}s)")
            time.sleep(remain)
        burst_id = step + 1
        t_burst = None
    return {"ledger": ledger, "envelope_polls": len(envelope_polls)}


# ------------------------------------------------------------ plots

def make_fates_plot(rd, states, refs, adj):
    """THE FATE FIGURE: the four+ trajectories vs the committed
    references + the CE panel + the verdict panel (bars + promotion rule
    + the anchoring disclosure verbatim)."""
    bar = adj["bar"]
    sts = sorted(int(s) for s in states["G"])       # 0,10,50,80
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0))

    for ax, anchor, key in ((axes[0][0], "iPhone (DIES)", ANCHOR_I),
                            (axes[0][1], "Gmail (HOLDS)", ANCHOR_G)):
        for arm, col, mk in (("G", "#c0392b", "o"), ("C2", "#e67e22", "s"),
                             ("G2", "#8e44ad", "D"),
                             ("C2W2", "#16a085", "^")):
            a0 = states[arm]["0"]["ctrl_p"][key]
            ys = [states[arm][str(s)]["ctrl_p"][key] / a0 for s in sts]
            ax.plot(sts, ys, f"{mk}-", color=col, lw=2.0, ms=6.5,
                    label=f"ARM-{arm} ({ARM_DESC[arm]})")
        for w, col in (("w1", "0.35"), ("w2", "0.55"), ("w3", "0.70")):
            ax.plot(refs[w]["states"], refs[w]["hr"][key], "d--", color=col,
                    lw=1.3, ms=5, alpha=0.9,
                    label=f"committed {w} ({refs[w]['label']})")
        lo, hi = adj["committed_spread"][key]
        ax.axhspan(lo, hi, color="#b8d8f0", alpha=0.35, zorder=0,
                   label="committed spread")
        if key == ANCHOR_I:
            ax.axhline(adj["carrier_line_04525_runtime_read"],
                       color="#1a6faf", ls=":",
                       lw=1.8, label="0.4525 carrier line (0.5x Gmail w1)")
            ax.axhline(adj["e233_arm_p_reference"], color="#c0392b",
                       ls="-.", lw=1.3, alpha=0.7,
                       label=f"e233 ARM-P applied-cut "
                             f"{adj['e233_arm_p_reference']:.3f}")
        ax.axhline(1.0, color="0.55", lw=0.7, ls=":")
        ax.set_xlabel("wash step"); ax.set_ylabel("hr = p(s)/p(0)")
        ax.set_title(f"{anchor} — the fate trajectories", fontsize=10)
        ax.legend(fontsize=6.6, loc="best"); ax.grid(alpha=0.25)

    ax = axes[1][0]
    for arm, col in (("G", "#c0392b"), ("C2", "#e67e22"),
                     ("G2", "#8e44ad"), ("C2W2", "#16a085")):
        led = adj["ledgers"][arm]
        xs = sorted(int(k) for k in led)
        ax.plot(xs, [led[str(s)]["ce"] for s in xs], "-", color=col,
                lw=1.2, alpha=0.85, label=f"ARM-{arm} in-batch CE")
    ax.plot(refs["w1"]["ce_states"], refs["w1"]["ce_vals"], "kd--", ms=5,
            lw=1.0, alpha=0.7, label="committed w1 in-batch CE")
    ax.plot(refs["w2"]["ce_states"], refs["w2"]["ce_vals"], "kd:", ms=5,
            lw=1.0, alpha=0.7, label="committed w2 in-batch CE")
    ax.set_xlabel("wash step"); ax.set_ylabel("batch CE (8x512)")
    ax.set_title("the wash must still be a wash — CE trajectories",
                 fontsize=10)
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25)

    ax = axes[1][1]
    ax.axis("off")
    y = 0.99
    ax.text(0.02, y, f"E237 VERDICT: {bar}"
            + ("  [UNDERPOWERED — honest-instrument clause]"
               if adj.get("underpowered") else "")
            + ("  |  PROMOTION: "
               + ("BOTH STREAMS — 'the wind has memory' quotable"
                  if adj.get("promotion") else
                  "NOT EARNED (see rider)")),
            fontsize=9.5, va="top", family="monospace", weight="bold",
            color="darkred")
    y -= 0.055
    for line in adj["verdict_lines"]:
        ax.text(0.02, y, line, fontsize=7.0, va="top", family="monospace")
        y -= 0.024
    y -= 0.015
    import textwrap
    for line in textwrap.wrap(ANCHORING_DISCLOSURE, 96):
        ax.text(0.02, y, line, fontsize=6.6, va="top", family="monospace",
                color="#555555", style="italic")
        y -= 0.02
    y -= 0.01
    ax.text(0.02, y, "GATES:", fontsize=8.6, va="top",
            family="monospace", weight="bold")
    y -= 0.028
    for gname, gval in adj["gates_summary"].items():
        ax.text(0.02, y, f"  {gname:14s} {'PASS' if gval else 'FAIL'}",
                fontsize=7.0, va="top", family="monospace")
        y -= 0.023

    fig.suptitle("E237 — THE PRE-ADAM PROJECTION (the wind's memory cut): "
                 "the GRADIENT projected off iPhone's t=0 support BEFORE "
                 "Adam\n(ARM-G/ARM-C2 on wash-1 seed 18202; the R63 w2 "
                 "rider ARM-G2/ARM-C2W2 on seed 20261002; the moments "
                 "never see the aligned component)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    png = rd / f"{NAME}_fates.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


def make_ledger_plot(rd, ledgers, e233_applied_fracs):
    """THE LEDGER FIGURE: the removed-L2 record AT THE GRADIENT LEVEL
    (both projection arms + both controls' aligned fraction — the
    'what the moments drank' record) + the gradient norms."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    ax = axes[0]
    for arm, col, lab in (("G", "#c0392b", "ARM-G (w1, projected)"),
                          ("G2", "#8e44ad", "ARM-G2 (w2, projected)"),
                          ("C2", "#e67e22", "ARM-C2 (w1, verbatim)"),
                          ("C2W2", "#16a085", "ARM-C2W2 (w2, verbatim)")):
        led = ledgers[arm]
        xs = sorted(int(k) for k in led)
        get = lambda s, k: (led[str(s)][k] if str(s) in led else led[s][k])
        ax.plot(xs, [get(s, "removed_frac") * 100 for s in xs], "o-",
                color=col, lw=1.4, ms=3.2, alpha=0.9, label=lab)
    e233x = sorted(int(k) for k in e233_applied_fracs)
    ax.plot(e233x, [e233_applied_fracs[str(s)] * 100 for s in e233x],
            "kx--", lw=1.0, ms=4, alpha=0.8,
            label="e233 ARM-P APPLIED-level fraction")
    ax.axhline(1.0, color="#1a6faf", ls=":", lw=1.8,
               label="the 1% honest-instrument line")
    ax.set_yscale("log")
    ax.set_xlabel("wash step")
    ax.set_ylabel("|<g, s_iPhone>| / ||g||  (%)")
    ax.set_title("the removed-L2 ledger AT THE GRADIENT LEVEL "
                 "(prediction (a): >1% clears the bar)", fontsize=9.5)
    ax.legend(fontsize=7.2); ax.grid(alpha=0.25, which="both")
    ax = axes[1]
    for arm, col in (("G", "#c0392b"), ("C2", "#e67e22"),
                     ("G2", "#8e44ad"), ("C2W2", "#16a085")):
        led = ledgers[arm]
        xs = sorted(int(k) for k in led)
        get = lambda s, k: (led[str(s)][k] if str(s) in led else led[s][k])
        ax.plot(xs, [get(s, "gnorm") for s in xs], "o-", color=col,
                lw=1.2, ms=3.0, alpha=0.85, label=f"ARM-{arm} ||g_clip||")
    ax.set_xlabel("wash step"); ax.set_ylabel("post-clip gradient L2 norm")
    ax.set_title("the gradient norms Adam sees (clip 1.0 active throughout)",
                 fontsize=9.5)
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25)
    ax = axes[2]
    led = ledgers["G"]
    xs = sorted(int(k) for k in led)
    get = lambda s, k: (led[str(s)][k] if str(s) in led else led[s][k])
    ax.plot(xs, [get(s, "gnorm_preclip") for s in xs], "o-", color="0.45",
            lw=1.3, ms=3.2, label="||g|| pre-clip (verbatim backward)")
    ax.plot(xs, [get(s, "gnorm") for s in xs], "s-", color="#1a6faf",
            lw=1.3, ms=3.2, label="||g|| post-clip (projected input)")
    ax.plot(xs, [get(s, "gpnorm") for s in xs], "^-", color="#c0392b",
            lw=1.3, ms=3.2, label="||g'|| post-projection (ARM-G)")
    ax.set_yscale("log")
    ax.set_xlabel("wash step"); ax.set_ylabel("gradient L2 norm")
    ax.set_title("ARM-G: the ordering disclosure (pre-clip vs post-clip "
                 "vs post-projection)", fontsize=9.5)
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    png = rd / f"{NAME}_ledger.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


# ------------------------------------------------------------ main

def main():
    jp = RD / "journal.json"
    log(f"E237 — THE PRE-ADAM PROJECTION (smoke={SMOKE}) -> {RD}")

    metrics = {
        "experiment": "e237_preadam_projection",
        "phase": "interventional: the certified wash streams continued "
                 "VERBATIM from pristine t=0 with each BATCH GRADIENT "
                 "projected off iPhone's t=0 support BEFORE Adam (ARM-G "
                 "w1 / ARM-G2 w2) vs hook-path controls (ARM-C2 / "
                 "ARM-C2W2), GPU fp32 bursts",
        "date": now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED_PREDICTION["registration"],
        "registered_prediction": REGISTERED_PREDICTION,
        "question": ("e233 removed the iPhone-aligned component of each "
                     "APPLIED AdamW step: iPhone's fate more than doubled "
                     "(0.188 -> 0.407) but stalled below the carrier line "
                     "(0.4525). THE HYPOTHESIS: Adam's moments still drank "
                     "the aligned GRADIENT — the wind has memory. Cut the "
                     "supply: project the GRADIENT before Adam sees it."),
        "builds_on": [
            "scratch/e237_design.md + its R63 amendment (the frozen design "
            "— THE registration; committed b2d696b/5e08e32)",
            "T215 / e233 (the between-marker-and-carrier verdict; the "
            "machinery PORTED WHOLE: the stream replay, the batteries, "
            "the supports, the ledgers, the per-step thermal guard, the "
            "resumable journal)",
            "T216 / e239 (the supports are vector-stable and the rotation "
            "is common-mode — this cell is also the wind's causal test "
            "against a replicated common-mode background)",
            "T183 / e182c2 (wash-2's certified stream + the w2 "
            "references), T192 / e217, e216 (the family), e226 (the "
            "committed three-wash archive + the fp64 instrument finding)",
            "g12 (the intervention-changes-the-trajectory precedent), "
            "g14 (MULTI-COMPONENT), W035 (the wind's anatomy)",
        ],
        "whats_new": [
            "the SUPPLY cut: the projection moves from the APPLIED step "
            "(e233) to the GRADIENT BEFORE Adam — the optimizer's two "
            "moments never see the aligned component (the wind's memory "
            "severed, not the shipment intercepted)",
            "the R63 w2 replication rider (ARM-G2 + ARM-C2W2 on wash-2's "
            "certified stream) with the BOTH-STREAMS promotion rule — "
            "the memory claim requires replication across independent "
            "draws before it is quotable",
            "the removed-L2 ledger AT THE GRADIENT LEVEL (both "
            "projection arms AND both controls' aligned fraction — the "
            "verbatim wash's own 'what the moments drank' record)",
            "the per-step thermal guard at the 78C margin with PER-STEP "
            "polls FROM THE FIRST BURST (e233's lesson as the default)",
        ],
        "smoke": SMOKE,
    }

    def write_metrics(status):
        metrics["status"] = status
        metrics["timing"] = {"total_s": round(time.time() - T0, 1)}
        save_json(RD / "metrics.json", metrics)

    journal: dict = {}
    if jp.exists():
        try:
            journal = json.loads(jp.read_text(encoding="utf-8"))
            log(f"journal restored: {list(journal)}")
        except Exception as ex:                              # noqa: BLE001
            log(f"journal unreadable ({ex}); starting fresh")
            journal = {}

    def save_journal():
        jp.write_text(json.dumps(journal, indent=1, default=float),
                      encoding="utf-8")

    load_checks: list[dict] = [cpu_load_check("launch")]
    metrics["load_checks"] = load_checks

    # ------------------------------------------------ P0 the committed records
    E182C2_M = common.REPO / "runs" / "e182c2" / "metrics.json"
    for p in (E182C_M, E182C2_M, E182C2_J, E217_M, E216_M, E226_M, E233_M):
        if not p.exists():
            log(f"FATAL: committed record missing: {p}")
            return 1
    p1m = json.loads(E182C_M.read_text(encoding="utf-8"))
    c2m = json.loads(E182C2_M.read_text(encoding="utf-8"))
    j2 = json.loads(E182C2_J.read_text(encoding="utf-8"))
    u3m = json.loads(E217_M.read_text(encoding="utf-8"))
    e233m = json.loads(E233_M.read_text(encoding="utf-8"))
    w1_rec = {s["step"]: s for s in p1m["states"]}
    w1_tmpl_rec = {s["step"]: s                 # e233's pattern: w1's
                   for s in c2m["part1_template"]["states"]}  # tmpl lives
    w2_rec = {s["step"]: s for s in j2["states"]}             # in e182c2
    w3_rec = {s["step"]: s for s in u3m["wash3_states"]}

    def _probes(state_rec, batt):
        return {f: v["p"] for f, v in state_rec[batt]["probes"].items()}

    committed = json.loads(
        E226_M.read_text(encoding="utf-8"))["anchor_fates_committed"]
    metrics["anchor_fates_committed"] = committed
    g_hrs = [committed["Gmail"][w]["hr"] for w in ("w1", "w2", "w3")]
    i_hrs = [committed["iPhone"][w]["hr"] for w in ("w1", "w2", "w3")]
    spread = {ANCHOR_G: (min(g_hrs), max(g_hrs)),
              ANCHOR_I: (min(i_hrs), max(i_hrs))}
    line_04525 = 0.5 * committed["Gmail"]["w1"]["hr"]     # runtime-read
    e233_arm_p_hr = e233m["adjudication"]["hr_at_final"][f"P-{ANCHOR_I}"]
    e233_applied_fracs = {
        str(k): v["removed_frac"]
        for k, v in e233m["ledgers"]["P"].items()}
    log("committed fates (runtime-read from e226): " + " | ".join(
        f"{t}: " + ", ".join(f"{w} hr {committed[t][w]['hr']:.3f}"
                             for w in ("w1", "w2", "w3"))
        for t in committed) + f" | CARRIER line (0.5x Gmail w1) "
        f"{line_04525:.4f} | e233 ARM-P (applied cut) {e233_arm_p_hr:.4f}")
    metrics["runtime_read_lines"] = {
        "carrier_line_04525": line_04525,
        "e233_arm_p_iphone_hr80": e233_arm_p_hr,
        "w2_refs": {"iPhone": committed["iPhone"]["w2"]["hr"],
                    "Gmail": committed["Gmail"]["w2"]["hr"]}}

    # the committed reference curves (anchors, all three washes)
    refs = {}
    for w, rec in (("w1", w1_rec), ("w2", w2_rec), ("w3", w3_rec)):
        sts = sorted(rec)
        refs[w] = {
            "label": {"w1": "e182c replay, seed 18202",
                      "w2": "e182c2 fresh, seed 20261002",
                      "w3": "e217 fresh, seed 21703"}[w],
            "states": sts,
            "hr": {a: [rec[s]["ctrl"]["probes"][a]["p"]
                       / rec[sts[0]]["ctrl"]["probes"][a]["p"]
                       for s in sts] for a in (ANCHOR_G, ANCHOR_I)},
            "ce_states": [s for s in sts[1:]
                          if rec[s].get("in_batch_ce") is not None],
            "ce_vals": [rec[s]["in_batch_ce"] for s in sts[1:]
                        if rec[s].get("in_batch_ce") is not None],
        }
    metrics["references"] = refs

    # ------------------------------------------------ P1 organism + corpus
    tok, net0, org_meta = e1.load_organism()
    metrics["organism"] = {**org_meta, "torch_threads": THREADS}
    G_SIZE = {"params": org_meta["params"], "ceiling": e1.SIZE_CEILING,
              "reason": e1.SIZE_REASON,
              "pass": bool(org_meta["params"] <= e1.SIZE_CEILING)}
    assert G_SIZE["pass"]
    metrics["gates"] = {"G_SIZE": G_SIZE}
    metrics["size_gate"] = G_SIZE

    text = (common.REPO / "data" / "input.txt").read_text(encoding="utf-8")
    e182 = json.loads(e1.E182_METRICS.read_text(encoding="utf-8"))
    e_corp, e_str = e182["corpus"], e182["gates"]["G_STR"]
    e_banned = e_str["banned"]
    banned = sorted({s.lower() for rel in e1.POOLS for s, _ in e1.POOLS[rel]}
                    | {a.lower() for rel in e1.POOLS
                       for _, a in e1.POOLS[rel]}
                    | set(e1.BANNED_EXTRA))
    assert banned == e_banned, "banned list diverged from e182's record"
    cand, _dropped = e1.build_candidates(tok)
    for r, b in zip(cand, e1.probe_battery(net0, cand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(cand)
    battery = [r for r in cand if r["kept"]]
    answer_ids = {r["fact"]: r["ans_id"] for r in battery}
    train_ids, bank_xy, filtered, G_STR, _G_TOK, corpus_stats = \
        e1.build_wash_corpus(tok, text, banned, answer_ids)
    G_CORPUS = {
        "banned_list_identical": True,
        "tokens_after": [corpus_stats["tokens_after"],
                         e_corp["tokens_after"]],
        "train_tokens": [corpus_stats["train_tokens"],
                         e_corp["train_tokens"]],
        "format": "[rebuilt, e182_recorded]",
        "note": "the corpus is INHERITED FROZEN — it feeds the wash-batch "
                "reproduction (the SAME train_ids both certified streams "
                "drew their windows from)",
    }
    G_CORPUS["pass"] = bool(
        G_STR["lines_total"] == e_str["lines_total"]
        and G_STR["lines_dropped"] == e_str["lines_dropped"]
        and corpus_stats["chars_after"] == e_corp["chars_after"]
        and corpus_stats["tokens_after"] == e_corp["tokens_after"]
        and corpus_stats["train_tokens"] == e_corp["train_tokens"])
    log(f"G_CORPUS: {'PASS' if G_CORPUS['pass'] else 'FAIL'} "
        f"({corpus_stats['tokens_after']} tokens)")
    assert G_CORPUS["pass"] or SMOKE
    metrics["gates"]["G_CORPUS"] = G_CORPUS
    write_metrics("PARTIAL: records read; organism + corpus certified")

    # --------------------------------- P2 the four batteries, VERBATIM (G_BATT)
    load_checks.append(cpu_load_check("batteries"))
    ccand, _cd = e1.build_control_candidates(
        tok, filtered.lower(), train_ids, e_banned,
        e1.CTRL_POOLS, e1.CTRL_TMPL)
    for r, b in zip(ccand, e1.probe_battery(net0, ccand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(ccand)
    cbattery = [r for r in ccand if r["kept"]]
    ncand, _nd = e1.build_control_candidates(
        tok, filtered.lower(), train_ids, e_banned,
        {"near": e1.NEAR_POOL}, {"near": e1.NEAR_TMPL})
    for r, b in zip(ncand, e1.probe_battery(net0, ncand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(ncand)
    nbattery = [r for r in ncand if r["kept"]]
    tcand, _td = e2.build_tmpl_candidates(tok, filtered.lower(), train_ids,
                                          e_banned)
    for r, b in zip(tcand, e1.probe_battery(net0, tcand)["probes"]):
        r.update({"p": b["p"], "rank": b["rank"], "top1": b["top1"],
                  "top5": b["top5"]})
    e1.select_battery(tcand)
    tbattery = [r for r in tcand if r["kept"]]
    bats = {"fact": battery, "ctrl": cbattery, "near": nbattery,
            "tmpl": tbattery}
    G_BATT = {}
    for b, bl in bats.items():
        mine_p = {r["fact"]: r["p"] for r in bl}
        dps, sets_ok = {}, True
        recs0 = {"w1": (w1_rec if b != "tmpl" else w1_tmpl_rec),
                 "w2": w2_rec, "w3": w3_rec}
        for w, rec in recs0.items():
            ref = _probes(rec[0], b)
            sets_ok = sets_ok and set(mine_p) == set(ref)
            dps[w] = (max(abs(mine_p[f] - v) for f, v in ref.items())
                      if set(mine_p) == set(ref) else None)
        G_BATT[b] = {"n": len(bl), "set_equal_committed": bool(sets_ok),
                     "max_dp": dps}
    G_BATT["tol_per_probe_dp"] = TOL_T0_DP
    G_BATT["pass"] = bool(
        all(G_BATT[b]["set_equal_committed"]
            and max((d for d in G_BATT[b]["max_dp"].values()
                     if d is not None), default=1.0) <= TOL_T0_DP
            for b in bats)) if not SMOKE else True
    G_BATT["note"] = ("batteries = the phase-1/phase-2 pools VERBATIM "
                      "(module import); t=0 must reproduce all THREE "
                      "committed records (e226's G_BATT convention)")
    metrics["gates"]["G_BATT"] = G_BATT
    log("G_BATT: " + ("PASS" if G_BATT["pass"] else "FAIL") + " | "
        + " | ".join(f"{b}: n {G_BATT[b]['n']} maxdp "
                     f"{max(G_BATT[b]['max_dp'].values()):.2e}"
                     for b in bats))
    if not G_BATT["pass"]:
        write_metrics("PARTIAL: G_BATT FAILED — halted before any compute")
        return 1
    probes_bats = bats
    ctrl_by_fact = {r["fact"]: r for r in cbattery}

    # --------------------------------- P3 the direction + FD + determinism
    load_checks.append(cpu_load_check("support t=0"))
    params0 = list(net0.parameters())
    def support_chunks(probe):
        """e226's probe_support convention (e233's port VERBATIM), kept
        as per-param chunks (the flat order IS net.parameters() order;
        fp64 global norm)."""
        net0.eval()
        net0.zero_grad(set_to_none=True)
        logits = net0(input_ids=probe["ids"]).logits
        pvec = F.softmax(logits[0, -1], dim=-1)
        p0 = float(pvec[probe["ans_id"]].item())
        pvec[probe["ans_id"]].backward()
        gs = [p.grad.detach().clone().reshape(-1)
              for p in params0]
        nrm = float(math.sqrt(sum(float(
            torch.dot(g.double(), g.double()).item()) for g in gs)))
        assert nrm > 0 and math.isfinite(nrm), "degenerate support"
        net0.zero_grad(set_to_none=True)
        return [g / nrm for g in gs], p0
    s_I_cpu, p0_I = support_chunks(ctrl_by_fact[ANCHOR_I])
    s_G_cpu, p0_G = support_chunks(ctrl_by_fact[ANCHOR_G])
    netFD = copy.deepcopy(net0)
    base_sds = [p.detach().clone() for p in netFD.parameters()]

    def fd_gate(s_chunks, probe):
        p0 = e3.probe_p(netFD, probe)
        out = {}
        for eps in FD_EPS:
            with torch.no_grad():
                for p, b, s_c in zip(netFD.parameters(), base_sds,
                                     s_chunks):
                    p.copy_(b.reshape(p.shape)
                            + (eps * s_c).reshape(p.shape))
            out[str(eps)] = e3.probe_p(netFD, probe) - p0
            with torch.no_grad():
                for p, b in zip(netFD.parameters(), base_sds):
                    p.copy_(b)
        return out
    fd_I = fd_gate(s_I_cpu, ctrl_by_fact[ANCHOR_I])
    fd_G = fd_gate(s_G_cpu, ctrl_by_fact[ANCHOR_G])
    del netFD, base_sds
    s_I_rep, _ = support_chunks(ctrl_by_fact[ANCHOR_I])
    self_cos = float(sum(torch.dot(a.double(), b.double()).item()
                         for a, b in zip(s_I_cpu, s_I_rep)))
    G_SUPPORT = {
        "fd_eps": list(FD_EPS),
        "rule": "p(theta0 + eps*s_hat) - p0 > 0 at the primary eps "
                "(e204's directional gate, e226's port to 124M)",
        "iphone_fd": fd_I, "gmail_fd_co_read": fd_G,
        "p0": {"iPhone": p0_I, "Gmail": p0_G,
               "committed_iphone": committed["iPhone"]["w1"]["p0"],
               "committed_gmail": committed["Gmail"]["w1"]["p0"]},
        "determinism_selfcos_iphone": self_cos,
        "pass": bool(fd_I[str(FD_EPS[0])] > 0 and self_cos > 0.999)
        if not SMOKE else True,
        "note": "the projection direction = iPhone's t=0 support, "
                "recomputed FRESH fp32 CPU (e226's convention; fp64 "
                "norms); ONE fixed direction for ALL FOUR arms (both "
                "streams; the design's honesty guard); Gmail's support "
                "FD-gated as a co-read",
    }
    metrics["gates"]["G_SUPPORT"] = G_SUPPORT
    log(f"G_SUPPORT: {'PASS' if G_SUPPORT['pass'] else 'FAIL'} "
        f"(iPhone fd {fd_I[str(FD_EPS[0])]:+.3e} @eps {FD_EPS[0]}; "
        f"determinism self-cos {self_cos:.7f}; p0 {p0_I:.4f} vs committed "
        f"{committed['iPhone']['w1']['p0']:.4f})")
    assert G_SUPPORT["pass"] or SMOKE
    write_metrics("PARTIAL: batteries + support direction certified")

    # --------------------------------- P4 the TWO certified draw streams
    load_checks.append(cpu_load_check("draws"))
    hi = train_ids.shape[0] - e1.SEQ - 1
    gen1 = torch.Generator().manual_seed(W1_SEED)
    offs1 = [torch.randint(hi, (e1.BATCH,), generator=gen1)
             for _ in range(80)]
    arch1 = torch.load(W1_LATEST, map_location=CPU,
                       weights_only=False)["gen"]
    gen2 = torch.Generator().manual_seed(W2_SEED)
    offs2 = [torch.randint(hi, (e1.BATCH,), generator=gen2)
             for _ in range(80)]
    arch2 = torch.load(W2_LATEST, map_location=CPU,
                       weights_only=False)["gen"]
    ok1 = bool(torch.equal(gen1.get_state(), arch1))
    ok2 = bool(torch.equal(gen2.get_state(), arch2))
    G_DRAWS = {
        "w1": {"seed": W1_SEED, "n_draws": len(offs1),
               "gen_state_identical_after_80": ok1,
               "note": "wash 1's window-draw stream reproduced from seed "
                       "18202 and certified BIT-EXACTLY against the "
                       "generator state archived at step 80 in "
                       "e182c_replay_latest.pt"},
        "w2": {"seed": W2_SEED, "n_draws": len(offs2),
               "gen_state_identical_after_80": ok2,
               "note": "wash 2's window-draw stream reproduced from seed "
                       "20261002 and certified BIT-EXACTLY against the "
                       "generator state archived at step 80 in "
                       "e182c2_fresh_latest.pt (the R63 rider's stream)"},
        "note": "e226's G_DRAWS convention applied to BOTH streams; all "
                "four arms replay THEIR stream from pristine t=0",
        "pass": bool(ok1 and ok2),
    }
    metrics["gates"]["G_DRAWS"] = G_DRAWS
    log(f"G_DRAWS: {'PASS' if G_DRAWS['pass'] else 'FAIL'} "
        f"(w1 seed {W1_SEED} bit-exact {ok1}; w2 seed {W2_SEED} "
        f"bit-exact {ok2})")
    assert G_DRAWS["pass"]

    # --------------------------------- P5 t=0 readout (all arms share it)
    load_checks.append(cpu_load_check("t0 readout"))
    def probe_state(netC):
        rec = {}
        for b, bl in probes_bats.items():
            bb = e1.probe_battery(netC, bl)
            rec[f"{b}_p"] = {r["fact"]: r["p"] for r in bb["probes"]}
            rec[f"{b}_mean_p"] = bb["mean_p"]
        hp = e1.ppl_eval(netC, *bank_xy)
        rec["bank_ppl"] = hp["ppl"]
        rec["bank_ce"] = hp["ce"]
        return rec
    t0_rec = probe_state(net0)
    # restore-safe merge (the e233 journal-clobber lesson): if the
    # journal carries ANY probed states, merge — only fill each arm's
    # t=0 row; NEVER rewrite the probed-state tree wholesale.
    states = journal.get("states") or {}
    if states:
        log(f"journal states restored: "
            + ", ".join(f"{t}: {sorted(s for s in states.get(t, {}))}"
                        for t in ("G", "C2", "G2", "C2W2")
                        if states.get(t)))
    for t in ("G", "C2", "G2", "C2W2"):
        states.setdefault(t, {})["0"] = t0_rec
    journal["states"] = states
    save_journal()
    log(f"t=0: iPhone p0 {t0_rec['ctrl_p'][ANCHOR_I]:.4f} Gmail p0 "
        f"{t0_rec['ctrl_p'][ANCHOR_G]:.4f} | bank ppl "
        f"{t0_rec['bank_ppl']:.2f} (e182 committed 71.34)")
    write_metrics("PARTIAL: t=0 readout done (all arms)")

    # --------------------------------- P6 the arms
    def on_ckpt(tag, step_, sd, ce):
        netC = copy.deepcopy(net0)
        netC.load_state_dict(sd)
        rec = probe_state(netC)
        rec["in_batch_ce"] = ce
        states[tag][str(step_)] = rec
        journal["states"] = states
        save_journal()
        log(f"  [ARM-{tag}] +{step_}: iPhone p "
            f"{rec['ctrl_p'][ANCHOR_I]:.4f} Gmail p "
            f"{rec['ctrl_p'][ANCHOR_G]:.4f} | bank ppl "
            f"{rec['bank_ppl']:.2f}")
        del netC

    dev = torch.device("cuda")
    s_chunks_gpu = [s.to(dev) for s in s_I_cpu]
    assert [s.numel() for s in s_chunks_gpu] == \
        [p.numel() for p in net0.parameters()]

    ledgers = journal.get("ledgers", {})
    for tag, stream, project in ARMS:
        load_checks.append(cpu_load_check(f"arm {tag}"))
        offs = offs1 if stream == "w1" else offs2
        res_key = f"arm_{tag}"
        already = journal.get(res_key, {}).get("done", False)
        if already and tag in ledgers and len(ledgers[tag]) >= N_STEPS \
                and all(str(s) in states[tag] for s in CK_STEPS):
            log(f"ARM-{tag}: journal says done — skipping")
            continue
        out = run_arm(tag, stream, project, net0, train_ids, offs,
                      s_chunks_gpu, probes_bats, on_ckpt)
        ledgers[tag] = out["ledger"]
        journal["ledgers"] = ledgers
        journal[res_key] = {"done": True,
                            "envelope_polls": out["envelope_polls"],
                            "stream": stream, "project": project}
        save_journal()
        write_metrics(f"PARTIAL: ARM-{tag} complete "
                      f"({len(out['ledger'])} steps)")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()    # the inter-arm cache reset
        if not SMOKE:
            time.sleep(COOLDOWN_S)      # the inter-arm cooldown

    # ---- the archive RE-PROBE pass (e214's/e233's convention): any
    # checkpoint whose probe record is missing (a kill mid-probe) is
    # re-probed from its saved weight archive — the states on disk are
    # the record.
    for tag, _stream, _proj in ARMS:
        for s_ in CK_STEPS:
            if str(s_) in states[tag]:
                continue
            arch = CK_DIR / f"{NAME}_{tag}_s{s_}.pt"
            if not arch.exists():
                log(f"WARNING: ARM-{tag} +{s_} probe record missing and "
                    f"no archive {arch} — the state read is lost")
                continue
            sd = torch.load(arch, map_location=CPU,
                            weights_only=False)["model"]
            netC = copy.deepcopy(net0)
            netC.load_state_dict(sd)
            rec = probe_state(netC)
            led_row = ledgers.get(tag, {}).get(str(s_)) or \
                ledgers.get(tag, {}).get(s_)
            if led_row is not None:
                rec["in_batch_ce"] = led_row["ce"]
            rec["source"] = f"archive re-probe (e237_{tag}_s{s_}.pt)"
            states[tag][str(s_)] = rec
            del netC, sd
            log(f"  [ARM-{tag}] +{s_}: RE-PROBED from archive — iPhone p "
                f"{rec['ctrl_p'][ANCHOR_I]:.4f} Gmail p "
                f"{rec['ctrl_p'][ANCHOR_G]:.4f} | bank ppl "
                f"{rec['bank_ppl']:.2f}")
        journal["states"] = states
        save_journal()

    # normalize ledger keys to str (journal/metrics round-trip shape)
    ledgers = {t: {str(k): v for k, v in ledgers[t].items()}
               for t in ledgers}
    journal["ledgers"] = ledgers
    save_journal()
    L = lambda t, s: ledgers[t][str(s)]        # the uniform accessor

    # --------------------------------- P7 gates: wash health
    def arm_health(tag):
        ces = [L(tag, s)["ce"] for s in range(1, N_STEPS + 1)]
        first10 = sum(ces[:10]) / len(ces[:10])
        last10 = sum(ces[-10:]) / len(ces[-10:])
        ppl0 = states[tag]["0"]["bank_ppl"]
        pplf = states[tag][str(CK_STEPS[-1])]["bank_ppl"]
        return {"ce_first10": first10, "ce_last10": last10,
                "ce_improves": bool(last10 < first10),
                "bank_ppl_t0": ppl0, "bank_ppl_final": pplf,
                "ppl_improves": bool(pplf < ppl0),
                "pass": bool(last10 < first10 and pplf < ppl0)}
    G_WASHHEALTH = {t: arm_health(t) for t, _, _ in ARMS}
    G_WASHHEALTH["rule"] = ("the wash must still be a wash: bank ppl "
                            "(+final) < bank ppl (t=0) AND mean in-batch "
                            "CE last-10 < first-10, ALL FOUR arms")
    G_WASHHEALTH["pass"] = bool(all(G_WASHHEALTH[t]["pass"]
                                    for t, _, _ in ARMS))
    metrics["gates"]["G_WASHHEALTH"] = G_WASHHEALTH
    G_ENV = {
        "device": "cuda (RTX 5090 Laptop 24GB) fp32, TF32 OFF, matmul "
                  "precision highest; CPU fp32 probing (threads 8)",
        "bursts": f"<= {BURST_MAX_S:.0f}s wall AND <= {BURST_MAX_STEPS} "
                  f"steps",
        "cooldown_s": COOLDOWN_S,
        "launch_gate": f"common.gpu_ok() (util<={common.GPU_UTIL_CEIL:.0f}%, "
                       f"temp<={common.GPU_TEMP_CEIL:.0f}C, mem<=85%), "
                       f"double-polled, first launch gap "
                       f"{LAUNCH_POLL_GAP_S:.0f}s",
        "mid_burst_guard": f"PER-STEP polls from the FIRST burst; burst "
                           f"ends at temp >= {TEMP_EARLY_END:.0f}C margin "
                           f"(never past 85C; the e233 lesson as default)",
        "window": "the owner's ACTIVE max-priority window (STATE.json "
                  "compute_directive, 2026-10-04): GPU assertive, NO "
                  "concurrent GPU jobs, temp-aware, single runs <=180s",
        "pass": True,
    }
    metrics["gates"]["G_ENV"] = G_ENV
    log(f"G_WASHHEALTH: {'PASS' if G_WASHHEALTH['pass'] else 'FAIL'} | "
        + " | ".join(f"ARM-{t}: CE {G_WASHHEALTH[t]['ce_first10']:.3f}->"
                     f"{G_WASHHEALTH[t]['ce_last10']:.3f}, ppl "
                     f"{G_WASHHEALTH[t]['bank_ppl_t0']:.1f}->"
                     f"{G_WASHHEALTH[t]['bank_ppl_final']:.1f}"
                     for t, _, _ in ARMS))

    # --------------------------------- P8 adjudication (the frozen bars)
    def hr(tag, fact, state):
        return states[tag][str(state)]["ctrl_p"][fact] / \
            states[tag]["0"]["ctrl_p"][fact]
    final = CK_STEPS[-1]
    hrF = {f"{t}-{a}": hr(t, a, final)
           for t, _, _ in ARMS for a in (ANCHOR_G, ANCHOR_I)}

    # --- the w1 clause (i): FATE-FLIPS-MEMORY's conjunction
    crosses = bool(hrF[f"G-{ANCHOR_I}"] >= line_04525)
    c2_in_spread = bool(spread[ANCHOR_I][0] <= hrF[f"C2-{ANCHOR_I}"]
                        <= spread[ANCHOR_I][1])
    flips_fire = bool(crosses and c2_in_spread)
    improves = bool(hrF[f"G-{ANCHOR_I}"] >= e233_arm_p_hr + IMPROVE_TOL)
    if flips_fire:
        bar = "FATE-FLIPS-MEMORY"
    elif improves and not crosses:
        bar = "STILL-BELOW"
    elif not improves:
        bar = "NO-GAIN"
    else:
        bar = "ANY"     # crossed but C2 out of spread: the conjunction
                        # failed; the trajectories verbatim

    # --- the w2 clause (ii): the R63 rider
    gap_I_w2 = hrF[f"G2-{ANCHOR_I}"] - hrF[f"C2W2-{ANCHOR_I}"]
    gap_G_w2 = hrF[f"G2-{ANCHOR_G}"] - hrF[f"C2W2-{ANCHOR_G}"]
    rider_fire = bool(gap_I_w2 >= RIDER_TOL
                      and abs(gap_G_w2) <= RIDER_TOL)
    promotion = bool(flips_fire and rider_fire)

    # --- the honest-instrument clause, now at the GRADIENT level
    fracs = sorted(L("G", s)["removed_frac"]
                   for s in range(1, N_STEPS + 1))
    med_frac = fracs[len(fracs) // 2]
    underpowered = bool(med_frac < 0.01)
    instrument_cleared = bool(med_frac > 0.01)   # prediction (a)
    fracs_w2 = sorted(L("G2", s)["removed_frac"]
                      for s in range(1, N_STEPS + 1))
    med_frac_w2 = fracs_w2[len(fracs_w2) // 2]
    ratio_grad_vs_applied = (med_frac /
                             sorted(e233_applied_fracs.values())
                             [len(e233_applied_fracs) // 2])

    verdict = {
        "bar": bar,
        "underpowered": underpowered,
        "instrument_cleared_prediction_a": instrument_cleared,
        "promotion": promotion,
        "promotion_rule_verbatim":
            REGISTERED_PREDICTION["promotion_rule_verbatim"],
        "hr_at_final": hrF,
        "hr_at_checks": {t: {a: {str(s): hr(t, a, s)
                                 for s in CK_STEPS}
                             for a in (ANCHOR_G, ANCHOR_I)}
                         for t, _, _ in ARMS},
        "committed_spread": {k: list(v) for k, v in spread.items()},
        "carrier_line_04525_runtime_read": line_04525,
        "e233_arm_p_reference": e233_arm_p_hr,
        "flips_conjunction": {
            "G_crosses_04525": crosses,
            "C2_within_committed_spread": c2_in_spread},
        "rider_conjunction": {
            "gap_iphone_G2_vs_C2W2": gap_I_w2,
            "gap_iphone_ge_005": bool(gap_I_w2 >= RIDER_TOL),
            "gap_gmail_G2_vs_C2W2": gap_G_w2,
            "gmail_P2_within_005_of_control":
                bool(abs(gap_G_w2) <= RIDER_TOL),
            "gmail_G2_within_committed_spread": bool(
                spread[ANCHOR_G][0] <= hrF[f"G2-{ANCHOR_G}"]
                <= spread[ANCHOR_G][1]),
            "fired": rider_fire},
        "still_below_clause": {
            "improves_over_e233_0407": improves,
            "improve_tol_registered": IMPROVE_TOL},
        "removed_frac_median_gradlevel": med_frac,
        "removed_frac_median_gradlevel_w2": med_frac_w2,
        "removed_frac_min": fracs[0], "removed_frac_max": fracs[-1],
        "grad_vs_applied_median_ratio": ratio_grad_vs_applied,
        "anchoring_disclosure": ANCHORING_DISCLOSURE,
        "bars_verbatim": REGISTERED_PREDICTION["bars_verbatim"],
    }
    verdict_lines = [
        f"iPhone hr(+80): G {hrF[f'G-{ANCHOR_I}']:.3f}  C2 "
        f"{hrF[f'C2-{ANCHOR_I}']:.3f}   (committed spread "
        f"{spread[ANCHOR_I][0]:.3f}-{spread[ANCHOR_I][1]:.3f}; carrier "
        f"line {line_04525:.4f}; e233 applied-cut {e233_arm_p_hr:.3f})",
        f"THE RIDER (w2): iPhone G2 {hrF[f'G2-{ANCHOR_I}']:.3f}  C2W2 "
        f"{hrF[f'C2W2-{ANCHOR_I}']:.3f}  gap {gap_I_w2:+.3f} "
        f"(needs >= +{RIDER_TOL}) | committed w2 iPhone 0.342",
        f"Gmail  hr(+80): G {hrF[f'G-{ANCHOR_G}']:.3f}  C2 "
        f"{hrF[f'C2-{ANCHOR_G}']:.3f} | G2 {hrF[f'G2-{ANCHOR_G}']:.3f} "
        f"C2W2 {hrF[f'C2W2-{ANCHOR_G}']:.3f} gap {gap_G_w2:+.3f} "
        f"(needs |.|<= {RIDER_TOL}) | spread "
        f"{spread[ANCHOR_G][0]:.3f}-{spread[ANCHOR_G][1]:.3f}",
        f"removed-L2 (GRADIENT level): median {med_frac*100:.3f}% "
        f"(w2 {med_frac_w2*100:.3f}%; min {fracs[0]*100:.3f}%, max "
        f"{fracs[-1]*100:.3f}%) vs e233 applied median "
        f"{sorted(e233_applied_fracs.values())[len(e233_applied_fracs)//2]*100:.3f}% "
        f"(ratio {ratio_grad_vs_applied:.2f}x)",
        f"-> honest-instrument bar (1%): "
        f"{'CLEARED (prediction (a) fires)' if instrument_cleared else 'NOT cleared'}"
        f"{' — UNDERPOWERED co-stamp' if underpowered else ''}",
        f"PROMOTION (both streams): "
        f"{'EARNED — the memory claim is quotable' if promotion else 'NOT EARNED'}"
        f" (w1 flips {flips_fire}, w2 rider {rider_fire})",
    ]

    # prediction (a) verbatim record
    pred_a = {"read": REGISTERED_PREDICTION["predictions_verbatim"]["(a)"],
              "grad_level_median": med_frac,
              "grad_level_median_w2": med_frac_w2,
              "grad_vs_applied_ratio": ratio_grad_vs_applied,
              "instrument_cleared": instrument_cleared}

    # prediction (b) — the family clause re-test under FATE-FLIPS-MEMORY
    fam_rows = []
    for f in IPHONE_FAMILY:
        row = {"fact": f,
               "p0": states["G"]["0"]["ctrl_p"][f],
               "committed_w1_hr": (
                   w1_rec[80]["ctrl"]["probes"][f]["p"]
                   / w1_rec[0]["ctrl"]["probes"][f]["p"]),
               "committed_w2_hr": (
                   w2_rec[80]["ctrl"]["probes"][f]["p"]
                   / w2_rec[0]["ctrl"]["probes"][f]["p"])}
        row.update({"hr_G": hr("G", f, final),
                    "hr_C2": hr("C2", f, final),
                    "hr_G2": hr("G2", f, final),
                    "hr_C2W2": hr("C2W2", f, final),
                    "spares_G_vs_w1": bool(
                        hr("G", f, final) > row["committed_w1_hr"]),
                    "spares_G2_vs_w2": bool(
                        hr("G2", f, final) > row["committed_w2_hr"])})
        fam_rows.append(row)
    pred_b = {"read": REGISTERED_PREDICTION["predictions_verbatim"]["(b)"],
              "family": "product family's 5 non-anchor members "
                        "(Xbox/Chrome/iPad/iTunes/PlayStation)",
              "rows": fam_rows,
              "n_spare_G_vs_w1": sum(1 for r in fam_rows
                                      if r["spares_G_vs_w1"]),
              "n_spare_G2_vs_w2": sum(1 for r in fam_rows
                                       if r["spares_G2_vs_w2"]),
              "tested_under": bar,
              "anchor_only_again": bool(
                  bar == "FATE-FLIPS-MEMORY"
                  and not all(r["spares_G_vs_w1"] for r in fam_rows))}

    # prediction (c) — Gmail + batteries stay at C2~committed
    pred_c = {"read": REGISTERED_PREDICTION["predictions_verbatim"]["(c)"],
              "gmail_within_spread": {t: bool(
                  spread[ANCHOR_G][0] <= hrF[f"{t}-{ANCHOR_G}"]
                  <= spread[ANCHOR_G][1]) for t, _, _ in ARMS},
              "gmail_gap_vs_own_control": {
                  "w1": hrF[f"G-{ANCHOR_G}"] - hrF[f"C2-{ANCHOR_G}"],
                  "w2": hrF[f"G2-{ANCHOR_G}"] - hrF[f"C2W2-{ANCHOR_G}"]}}
    # the near battery co-report (both streams)
    pred_near = {"read": "the literal near battery (near-uscap) "
                         "co-reported at +final, all arms",
                 "rows": [{"fact": r["fact"],
                           "p0": states["G"]["0"]["near_p"][r["fact"]],
                           "hr_G": states["G"][str(final)]
                           ["near_p"][r["fact"]]
                           / states["G"]["0"]["near_p"][r["fact"]],
                           "hr_C2": states["C2"][str(final)]
                           ["near_p"][r["fact"]]
                           / states["C2"]["0"]["near_p"][r["fact"]],
                           "hr_G2": states["G2"][str(final)]
                           ["near_p"][r["fact"]]
                           / states["G2"]["0"]["near_p"][r["fact"]],
                           "hr_C2W2": states["C2W2"][str(final)]
                           ["near_p"][r["fact"]]
                           / states["C2W2"]["0"]["near_p"][r["fact"]],
                           "committed_w1_hr": (
                               w1_rec[80]["near"]["probes"]
                               [r["fact"]]["p"]
                               / w1_rec[0]["near"]["probes"]
                               [r["fact"]]["p"]),
                           "committed_w2_hr": (
                               w2_rec[80]["near"]["probes"]
                               [r["fact"]]["p"]
                               / w2_rec[0]["near"]["probes"]
                               [r["fact"]]["p"])}
                          for r in nbattery]}

    metrics["adjudication"] = {
        **verdict,
        "verdict_note": ("the verdict is CO-STAMPED UNDERPOWERED when the "
                         "median removed fraction (gradient level) is < 1% "
                         "(the honest-instrument clause; the stamp "
                         "discloses, it does not move the bar)")
        if underpowered else
        "the honest-instrument bar CLEARED at the gradient level "
        "(median removed fraction >= 1%; prediction (a) fired)"
        if instrument_cleared else
        "the honest-instrument clause not triggered (median removed "
        "fraction >= 1%)",
        "prediction_a": pred_a,
        "prediction_b": pred_b,
        "prediction_c": pred_c,
        "near_battery_coreport": pred_near,
        "gated_on": "G_SIZE/G_CORPUS/G_BATT/G_SUPPORT/G_DRAWS/"
                    "G_WASHHEALTH/G_ENV",
        "all_gates_pass": bool(all(
            v.get("pass", True) for v in metrics["gates"].values()
            if isinstance(v, dict))),
    }
    metrics["adjudication"]["verdict_lines"] = verdict_lines
    metrics["states"] = {t: {s: {"ctrl_p": states[t][s]["ctrl_p"],
                                 "ctrl_mean_p": states[t][s]["ctrl_mean_p"],
                                 "bank_ppl": states[t][s]["bank_ppl"],
                                 "in_batch_ce": states[t][s].get(
                                     "in_batch_ce")}
                             for s in states[t]} for t, _, _ in ARMS}
    metrics["ledgers"] = ledgers
    metrics["honesty_reflex"] = {
        "n": "n=1 organism, n=1 stream per arm (the design's guard; the "
             "w1/w2 pair is the amendment's replication base — two "
             "independent draws, one organism)",
        "trajectory": "the intervention changes the trajectory: after "
                      "step 1 each arm is its own wash (g12's precedent; "
                      "the design's disclosed nature) — fates, not "
                      "trajectories, are compared",
        "direction": "ONE t=0-fixed direction for ALL steps and BOTH "
                     "streams (e226's rotation reads were sub-bar; the "
                     "design's disclosed choice); the projection removes "
                     "the component along it and does NOT rescale",
        "optimizer_state": "THE CELL: the AdamW moments advance on the "
                           "PROJECTED gradient — the supply cut (e233's "
                           "disclosed opposite was the shipment cut)",
        "instrument_floor": "all ledger dots/norms fp64 (e226 found fp32 "
                            "accumulation over 124M coords drifts ~0.6% "
                            "— the same order as the 1% clause)",
        "device": "arms GPU fp32 vs committed w1 CPU-fp32/w2w3 GPU-fp32 "
                  "references — the spread spans the texture; fates "
                  "compared, not bit values (e217's precedent)",
        "anchoring": ANCHORING_DISCLOSURE,
    }

    # --------------------------------- P9 plots + final write
    gates_summary = {g: metrics["gates"][g].get("pass", True)
                     for g in metrics["gates"]}
    adj_plot = {**verdict, "verdict_lines": verdict_lines,
                "gates_summary": gates_summary,
                "committed_spread": spread,
                "ledgers": ledgers}
    png1 = make_fates_plot(RD, states, refs, adj_plot)
    png2 = make_ledger_plot(RD, ledgers, e233_applied_fracs)
    metrics["plot_outputs"] = [str(png1), str(png2)]
    metrics["compute"] = {
        "wall_s": round(time.time() - T0, 1),
        "device": "GPU fp32 bursts (RTX 5090 Laptop) + CPU fp32 probing",
        "steps": f"4 arms x {N_STEPS} steps (w1+w2 certified streams)",
        "load_checks": len(load_checks),
    }
    metrics["trims"] = trims
    metrics["deviations"] = deviations
    status = ("SMOKE DONE (nothing adjudicated)" if SMOKE else
              ("DONE" if metrics["adjudication"]["all_gates_pass"]
               else "DONE (gate failures disclosed — see gates)"))
    write_metrics(status)
    log(f"VERDICT: {bar}"
        + (" [UNDERPOWERED]" if underpowered else "")
        + f" | PROMOTION {'EARNED' if promotion else 'NOT EARNED'}"
        + f" | iPhone G {hrF[f'G-{ANCHOR_I}']:.3f} / C2 "
        f"{hrF[f'C2-{ANCHOR_I}']:.3f} vs line {line_04525:.4f} "
        f"(e233 P {e233_arm_p_hr:.3f})"
        + f" | w2 rider: G2 {hrF[f'G2-{ANCHOR_I}']:.3f} vs C2W2 "
        f"{hrF[f'C2W2-{ANCHOR_I}']:.3f} gap {gap_I_w2:+.3f}"
        + f" | removed median {med_frac*100:.3f}% (grad level)")
    log(f"outputs: {RD / 'metrics.json'}, {png1}, {png2}")
    log(f"total {time.time() - T0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
