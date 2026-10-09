"""X16 — THE OFF-TARGET SPECIFICITY PROBE (a CPU-only desk cell from
consult #010: agy's adopted hole in T279's two-channel split). This
docstring carries the registered question + arms + bars + prediction
P-x16a VERBATIM from the dispatch letter + every frozen convention,
committed at birth BEFORE any compute. Adjudicate against exactly
this; no bar shopping.

THE QUESTION (verbatim): "does the knob act like a volume control
(content-independent confidence) or a logit bias (target-specific,
zero-sum on other tokens)? SECONDARY: does ALIGNMENT matter at 1-dim
scale (the read-gradient direction vs a random 1-dim out-of-room
direction at matched norm)?"

THE HOLE (consult #010, background verbatim): "e311 applied a 1-dim
out-of-room 'trigger' — the host's own read-gradient direction at
0.0005x||dW|| — and boosted the host's read +32% (0.3494). T279 called
out-of-room displacement 'the VOLUME channel (intensity/confidence/
noise)'. Consult #010's hole: the knob was only ever tested on the
HOST'S OWN prompt — it may be a TARGET-SPECIFIC LOGIT BIAS (an additive
bias on the host's logit that intrudes on every other decision via the
softmax denominator), not a content-independent volume channel. No
off-target measurement exists."

THE ARMS (verbatim skeleton, conventions frozen below):
  1. PRIMARY (TRIGGER): "apply the committed trigger vector (exact e311
     convention) to the e311 host organism; measure across a GENERAL
     corpus battery (a few hundred normal corpus contexts — use the
     rig's existing corpus/validation machinery): per-context top-1
     identity stability (fraction unchanged), winner-margin change
     distribution, entropy change distribution, and the mean effect on
     the WINNING token's probability. Also at the host's 60 g0 contexts
     (the committed +32% must reproduce bit-exact as the calibration
     anchor)."
  2. CONTROL (RANDOM): "a RANDOM 1-dim out-of-room direction (same
     norm, fresh seed, in-room fraction verified ~0 like the trigger's
     4.5e-17) — same battery. Alignment vs chance at matched dose."

FROZEN BARS (verbatim; adjudicate against exactly this):
  - TARGET-SPECIFIC-BIAS: "at >=25% of general contexts the top-1 flips
    or the winner margin falls >10%, AND the margin-change distribution
    is significantly skewed (not symmetric) — the knob is an out-of-room
    LOGIT BIAS; T279's 'volume channel' re-names to 'bias + noise'
    (both out-of-room, neither content)."
  - CONTENT-INDEPENDENT-VOLUME: "top-1 identity essentially stable
    (>=95% unchanged), margins and entropies shift symmetrically upward
    in confidence — the knob is a true volume dial; W049's
    calibration-dial design (Q3) proceeds as planned."
  - ALIGNMENT-MATTERS (secondary bar): "the random-direction control
    shows a materially smaller/weirder effect than the trigger (e.g.,
    no host boost at matched norm) — the +32% needed the read-aligned
    direction; note this sharpens or kills W049's alignment story."

OPERATIONALIZATIONS (frozen HERE at birth BEFORE compute; they fix the
clauses, they do not move the bars):
  * THE HOST := e311's committed host organism — e261_K10K_inst_resume.pt
    s400 loaded BIT-EXACT + gated THREE ways (G_FACTLOAD: artifact md5 +
    flat md5 ebebb4472725d582dd74928493f1bfb3 + the behavioral panel vs
    the committed literals 0.26464763283729553 / 0.10525520890951157,
    bars 2e-6 / 1e-5 — the family's cross-session read-determinism law).
  * THE TRIGGER := the committed fp64 vector runs/e311/
    e311_hijack_vectors.pt : model.trigger (bytes md5
    7dfa7d65c8e80cbfe3b778502dc0a3ad; ||.|| = c := 0.0005 x
    9.1788432658723 = 0.0045894216329361505; e311's committed in-room
    fraction 4.534738452741713e-17), applied in e311's EXACT convention:
    theta := flat64(host) + trigger, fp64 arithmetic then fp32 cast
    (e314's applied-state convention). THE CALIBRATION ANCHOR: the
    applied state's flat md5 MUST equal e311's committed arm-A flat md5
    f84fd2647fa5c2943d3d1b6f24b05e04 AND the host's 60-g0-context
    battery read MUST reproduce e311's committed 0.3494305908679962
    (the +32%; gate |d| <= 2e-6, the exact diff disclosed).
  * THE ROOM := the host's own committed K10K room (SRCT k=10,000,
    seeds 26113/26114, n=2,739,072), rebuilt + D/S bit-bound vs
    e264_rooms.pt (md5 2d524655575cce00a3bc1c8770f4b211) — the
    projector that defines "out-of-room" for BOTH arms.
  * THE RANDOM CONTROL := one fresh draw x ~ N(0,1)^N from THIS cell's
    registered generator (seed 31602), projected OUT of the room
    (u := (x - P x)/||.||), scaled to the SAME c; its in-room fraction
    gated <= 1e-9 (matched to the trigger's ~4.5e-17 class); cos to the
    trigger co-reported (chance ~1/sqrt(N) ~ 6e-4).
  * THE GENERAL BATTERY := 300 name-free validation-split windows of
    block 256 (the rig's own val_windows machinery — e065/e119's
    instrument, fresh registered seed 31603), read at the LAST position
    (the battery's own read convention), fp32 logits -> fp64 arithmetic.
  * "winner margin" := the last-position logit gap top1 - top2 within
    the state; "falls >10%" := Δmargin < -0.10 x margin_base; "top-1
    flips" := argmax changes identity.
  * "significantly skewed (not symmetric)" := the NEGATIVE share of
    Δmargin over the battery > 0.5 AND the exact two-sided binomial
    sign test on Δmargin (P=0.5, symmetric-tail form) gives p < 0.01;
    the Fisher-Pearson skewness co-reported.
  * "margins and entropies shift symmetrically upward in confidence"
    := top-1 stability >= 0.95 AND mean Δmargin > 0 AND mean Δentropy
    < 0 AND flip-or->10%-squeeze fraction < 0.25.
  * "materially smaller/weirder ... (e.g., no host boost at matched
    norm)" := the control's host-g0 boost <= 0.25 x the trigger's
    committed boost AND the control's mean |Δmargin| <= 0.5 x the
    trigger's.
  * ADJUDICATION ORDER (frozen): the TRIGGER arm is adjudicated first
    (TARGET-SPECIFIC-BIAS -> CONTENT-INDEPENDENT-VOLUME -> MIXED, the
    table verbatim); the CONTROL never adjudicates the primary — it
    adjudicates the SECONDARY (ALIGNMENT-MATTERS vs ALIGNMENT-DEAD).

REGISTERED PREDICTIONS (frozen at birth BEFORE compute):
  - P-x16a (the dispatch's registered lab guess, VERBATIM):
    "TARGET-SPECIFIC-BIAS with ALIGNMENT-MATTERS (the trigger is the
    host's own read-gradient — it should act on the host's logit
    specifically; a random 1-dim direction at 0.70x the transport
    bracket should do roughly nothing)."
  - P-x16b (the executor's registered counter/nuance — stated per the
    dispatch's own invitation): "BIAS-SHAPED-BUT-UNDER-BAR: the trigger
    raises Z's logit preferentially (Δlogit(Z) - Δlogit(mean) > 0
    across general contexts; the Δmargin distribution one-sided toward
    squeeze, sign-test p < 0.01) BUT the dose is 0.70x the transport
    bracket's lower edge — in general text the margins are wide and Z
    is rare, so the >=25% flip-or->10%-squeeze bar likely MISSES: the
    honest primary verdict MIXED (the table verbatim) with the
    mechanism carried by the skew + the Z-bias excess; ALIGNMENT-MATTERS
    stands (the control does nothing)."

COMPUTE ENVELOPE: CPU ONLY (dispatch: another agent owns the GPU lane)
— torch threads 4, pocketfft DCT workers 4 (e307's desk convention), no
CUDA calls anywhere; the heaviest ops are 2.7M-vector fp64 DCTs and
(300 x 256) CPU forwards (bursts of minutes, no thermal exposure).
TIMESTAMPS: datetime.now(UTC) only.

Outputs: runs/x16/{metrics.json (PROGRESSIVE), REPORT.md,
x16_offtarget.png}; run.log gitignored. The random control vector is
deterministically regenerable (seed 31602 + the bit-bound room) — its
bytes md5 recorded in metrics, no .pt committed (the .gitignore
policy). No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator
folds). Commit + push: birth -> smoke -> complete.

Run:  python lab/x16_offtarget_probe.py        (X16_SMOKE=1 shakedown)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")       # e228's offline convention
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:                                            # Windows console safety
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                               # noqa: BLE001
    pass

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import CharCorpus, run_dir, save_json, set_seed  # noqa: E402

import e043_install as E43                             # noqa: E402 (REPO,
                                                      # find_occ, SPLICE_RNG)
import g1b_continuity as GB                            # noqa: E402 — MUST be
                                                      # imported BEFORE G1
import g1_anchored_ball as G1                          # noqa: E402
from e261_rank_ladder import SRCT                      # noqa: E402 — the
                                                      # room projector only

torch.set_num_threads(4)           # CPU-only cell; the shared desk lane

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("X16_SMOKE") == "1"
NAME = "x16_smoke" if SMOKE else "x16"

T0 = time.time()
RD = run_dir(NAME)
LOG_PATH = RD / "run.log"
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


# ======================================================================
# THE CONFIG (frozen)
# ======================================================================
REPO = E43.REPO
CKPT_DIR = GB.CKPT_DIR

# ---- THE HOST (e311's organism, verbatim) --------------------------------
FACT_CK = "e261_K10K_inst_resume.pt"
FACT_MD5 = "0f6dc1cf46850ce655dfafc9c853d467"
FACT_SIZE = 32958479
FACT_FLAT_MD5 = "ebebb4472725d582dd74928493f1bfb3"
FACT_BASELINE_G0 = 0.26464763283729553      # e264's committed K10K post g0
FACT_BASELINE_GM12 = 0.10525520890951157    # e264's committed K10K post gm12
HOST_WRITE_NORM = 9.1788432658723           # e306's committed dW literal
FACT_READ_TOL_G0 = 2e-6                     # the family's read-determinism
FACT_READ_TOL_GM12 = 1e-5                   # cross-session law

# ---- THE TRIGGER (e311's committed artifact) -----------------------------
E311_METRICS = REPO / "runs" / "e311" / "metrics.json"
E311_METRICS_MD5 = "4efbb2242c8e2e3abab1a2cb63e3e35a"
E311_METRICS_VERDICT = "NOTHING-READS"
E311_VECTORS = REPO / "runs" / "e311" / "e311_hijack_vectors.pt"
E311_VECTORS_MD5 = "29c28977f10eca1a841b0128abb0821c"
E311_TRIGGER_BYTES_MD5 = "7dfa7d65c8e80cbfe3b778502dc0a3ad"
TRIGGER_FRAC = 0.0005
TRIGGER_SCALE = TRIGGER_FRAC * HOST_WRITE_NORM   # 0.0045894216329361505
E311_TRIG_INROOM = 4.534738452741713e-17         # e311's committed literal
E311_ARM_A_HOST_G0 = 0.3494305908679962          # the +32% anchor (x1.3205)
E311_ARM_A_FLAT_MD5 = "f84fd2647fa5c2943d3d1b6f24b05e04"
E311_ARM_A_FRAC_ARGMAX = 0.43333333333333335     # committed co-literal
E311_TRIG_WINDOW = 0                              # committed chosen window
E311_TRIG_WINDOW_PZ = 0.4655166268348694          # committed window p(Z)
ZERO_BEARER_BAR = 1e-9

# ---- THE ROOM (the host's own K10K room) ---------------------------------
ROOMS264_CK = "e264_rooms.pt"
ROOMS264_MD5 = "2d524655575cce00a3bc1c8770f4b211"
ROOM_K = 10_000
ROOM_SEED_D, ROOM_SEED_S = 26113, 26114
N_PARAMS = GB.G1B_PARAMS                          # 2,739,072

# ---- THIS CELL'S REGISTERED SEEDS ----------------------------------------
GLOBAL_SEED = 31601        # global init only; every RNG is its own
RANDOM_CTRL_SEED = 31602   # the random control's draw generator
BATTERY_SEED = 31603       # the general battery's val_windows seed
CERT_SEED = 31605          # the room light-cert probe rng

# ---- THE BATTERY + THE BARS' NUMBERS --------------------------------------
BATTERY_N = 40 if SMOKE else 300
BLOCK = 256                                          # the rig's block
BS = 30                                              # battery_cell's bs
FLIP_SQUEEZE_BAR = 0.25          # the >=25% bar
SQUEEZE_REL = 0.10               # the >10% margin-fall bar
VOLUME_STABILITY_BAR = 0.95      # the >=95% unchanged bar
SIGN_P_BAR = 0.01                # the skew significance bar
ALIGN_BOOST_FRAC = 0.25          # control boost <= 25% of trigger's
ALIGN_MARGIN_FRAC = 0.5          # control mean|dm| <= 50% of trigger's
BATT_DIVERSITY_BAR = 20          # distinct base winners across battery

REGISTERED = {
    "question_verbatim": "does the knob act like a volume control "
        "(content-independent confidence) or a logit bias (target-"
        "specific, zero-sum on other tokens)? SECONDARY: does ALIGNMENT "
        "matter at 1-dim scale (the read-gradient direction vs a random "
        "1-dim out-of-room direction at matched norm)?",
    "the_hole_verbatim": "the knob was only ever tested on the HOST'S OWN "
        "prompt — it may be a TARGET-SPECIFIC LOGIT BIAS (an additive bias "
        "on the host's logit that intrudes on every other decision via "
        "the softmax denominator), not a content-independent volume "
        "channel. No off-target measurement exists.",
    "bars_verbatim": {
        "TARGET-SPECIFIC-BIAS": "at >=25% of general contexts the top-1 "
            "flips or the winner margin falls >10%, AND the margin-change "
            "distribution is significantly skewed (not symmetric) — the "
            "knob is an out-of-room LOGIT BIAS; T279's 'volume channel' "
            "re-names to 'bias + noise' (both out-of-room, neither "
            "content).",
        "CONTENT-INDEPENDENT-VOLUME": "top-1 identity essentially stable "
            "(>=95% unchanged), margins and entropies shift symmetrically "
            "upward in confidence — the knob is a true volume dial; "
            "W049's calibration-dial design (Q3) proceeds as planned.",
        "ALIGNMENT-MATTERS": "the random-direction control shows a "
            "materially smaller/weirder effect than the trigger (e.g., no "
            "host boost at matched norm) — the +32% needed the "
            "read-aligned direction; note this sharpens or kills W049's "
            "alignment story.",
    },
    "operationalizations": (
        "frozen BEFORE compute: THE HOST := e311's committed host "
        "(e261_K10K_inst_resume.pt s400; three-way G_FACTLOAD; flat md5 "
        f"{FACT_FLAT_MD5}); THE TRIGGER := runs/e311/"
        "e311_hijack_vectors.pt : model.trigger (bytes md5 "
        f"{E311_TRIGGER_BYTES_MD5}), applied fp64-then-fp32 (e314's "
        "convention); THE ANCHOR := the applied state's flat md5 == "
        f"e311's arm-A {E311_ARM_A_FLAT_MD5} AND the 60-g0 battery read "
        f"== {E311_ARM_A_HOST_G0!r} (|d| <= 2e-6); THE ROOM := the "
        "committed K10K SRCT room (26113/26114) D/S bit-bound vs "
        "e264_rooms.pt; THE CONTROL := one N(0,1)^N draw (gen 31602) "
        "projected OUT of the room, scaled to the same c, in-room "
        "<= 1e-9; THE BATTERY := 300 name-free val windows (block 256, "
        "val_windows machinery, seed 31603), last-position read, fp64 "
        "arithmetic; 'winner margin' := top1-top2 logit gap in-state; "
        "'falls >10%' := dm < -0.10 x margin_base; 'significantly "
        "skewed' := neg share > 0.5 AND exact two-sided binomial sign "
        "test p < 0.01; 'symmetrically upward in confidence' := "
        "stability >= 0.95 AND mean dm > 0 AND mean dH < 0 AND "
        "flip-or-squeeze < 0.25; 'materially smaller' := control host "
        "boost <= 0.25 x trigger boost AND control mean|dm| <= 0.5 x "
        "trigger mean|dm|; ADJUDICATION ORDER: TRIGGER arm primary "
        "(BIAS -> VOLUME -> MIXED), CONTROL secondary only "
        "(ALIGNMENT-MATTERS vs ALIGNMENT-DEAD)"),
    "registration": "question + arms + bars + P-x16a VERBATIM from the "
        "dispatch letter (consult #010's adopted hole; the executor's "
        "counter P-x16b registered per the dispatch's invitation); every "
        "convention picked + frozen HERE at birth BEFORE compute; this "
        "script committed at birth; adjudicate against exactly this; no "
        "bar shopping.",
    "predictions": {
        "P-x16a_lab": "TARGET-SPECIFIC-BIAS with ALIGNMENT-MATTERS (the "
            "trigger is the host's own read-gradient — it should act on "
            "the host's logit specifically; a random 1-dim direction at "
            "0.70x the transport bracket should do roughly nothing).",
        "P-x16b_executor_counter": "BIAS-SHAPED-BUT-UNDER-BAR: the "
            "trigger raises Z's logit preferentially (dbiasZ > 0; the "
            "dmargin distribution one-sided toward squeeze, sign-test "
            "p < 0.01) BUT the >=25% flip-or->10%-squeeze bar likely "
            "MISSES at this dose (margins wide, Z rare in general text): "
            "primary MIXED with the mechanism in the skew + the Z-bias "
            "excess; ALIGNMENT-MATTERS stands.",
    },
}

deviations: list[str] = [
    "CPU-ONLY desk cell (dispatch: another agent owns the GPU lane) — "
    "torch threads 4, DCT workers 4, no CUDA calls; the GPU envelope "
    "code is never imported beyond the pure SRCT projector class.",
    "THE GENERAL BATTERY := the rig's val_windows machinery (e065/"
    "e119's instrument) at n=300 with THIS cell's fresh seed 31603 — "
    "the dispatch's 'a few hundred normal corpus contexts' frozen "
    "deterministically; the read is the battery's own LAST-position "
    "convention.",
    "The room is rebuilt via SRCT directly (not the full LadderRooms): "
    "x16 needs only the projector (the out-of-room definition); the "
    "v-map/span ledger columns are not used; D/S bit-bound vs "
    "e264_rooms.pt + a light in-cell idempotency/kept^2 certification "
    "(2 probes, seed 31605) replace the full ladder certification.",
    "The random control vector is NOT committed as a .pt (the repo's "
    "*.gitignore policy): it is deterministically regenerable from the "
    "registered seed 31602 + the bit-bound room; its bytes md5 is "
    "recorded in metrics.json.",
    "n=1 per arm, one host organism, one session (the g-series standing "
    "caveat); the arms' DIFFERENCE (trigger vs random at matched dose) "
    "is the registered object.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator "
    "folds).",
    "Smoke mode (X16_SMOKE=1): battery n=40, the REAL room + host + "
    "trigger + anchor gates all LIVE (the room rebuild is cheap here), "
    "own smoke dir; NOTHING adjudicated (SMOKE stamp on every read).",
]

metrics: dict = {
    "experiment": "x16_offtarget_probe",
    "phase": "THE OFF-TARGET SPECIFICITY PROBE — does e311's 1-dim "
             "out-of-room gain knob act as a volume control (content-"
             "independent confidence) or a target-specific logit bias? "
             "PRIMARY: the committed e311 trigger on a 300-context "
             "general corpus battery (+ the bit-exact +32% host "
             "anchor); CONTROL: a matched-norm random out-of-room "
             "direction (the alignment question)",
    "date": common.now_iso(),
    "status": "PARTIAL: startup",
    "registration": REGISTERED["registration"],
    "registered": REGISTERED,
    "smoke": SMOKE,
    "envelope": {
        "device": "CPU ONLY (torch threads 4, pocketfft workers 4; "
                  "another agent owns the GPU lane)",
        "timestamps": "datetime.now(UTC) only",
    },
    "deviations": deviations,
    "builds_on": [
        "e311 (the parasitic hijacker: the committed trigger vector + "
        "the +32% host anchor + the applied-state convention)",
        "T279 (the two-channel split: out-of-room displacement named "
        "the VOLUME channel — the naming this probe tests)",
        "agy consult #010 (the adopted hole: the knob only ever "
        "measured on the host's own prompt)",
        "e264/e288 (the established-fact rig: the host artifact, the "
        "K10K room, the three-way G_FACTLOAD)",
        "e261 (the SRCT room machinery — the projector ported by "
        "import)",
        "e065/e119 (val_windows — the general-corpus battery "
        "instrument)",
        "e314 (the applied-state fp64-then-fp32 convention)",
    ],
    "whats_new": [
        "THE FIRST OFF-TARGET MEASUREMENT of the 1-dim gain knob: "
        "per-context top-1 stability, margin-change and entropy-change "
        "distributions, and the winner-probability effect across 300 "
        "general corpus contexts (no off-target datum existed)",
        "THE ALIGNMENT CONTROL: a matched-norm random out-of-room "
        "1-dim direction on the same battery — read-gradient alignment "
        "vs chance at identical dose",
        "THE Z-BIAS EXCESS read: Δlogit(Z) minus the mean logit shift "
        "per context — the token-specific component the bias story "
        "predicts and the volume story forbids",
    ],
    "gates": {},
}


# ------------------------------------------------------------------ helpers
def flat_params_cpu(net) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).cpu()
                      for p in net.parameters()]).clone()


def save_json_partial(note: str) -> None:
    metrics["date_updated"] = common.now_iso()
    metrics["phase_note"] = note
    save_json(RD / "metrics.json", metrics)
    if not SMOKE:
        try:
            subprocess.run(["git", "add", str(RD / "metrics.json")],
                           cwd=str(REPO), timeout=10)
        except Exception:                                   # noqa: BLE001
            pass


def md5of(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:                                       # noqa: BLE001
        return "unavailable"


def apply_flat64(net, flat64: np.ndarray, offsets, shapes) -> None:
    """theta <- flat64 (fp64 arithmetic) cast fp32 into the params —
    e314/e311's applied-state convention, VERBATIM."""
    f32 = flat64.astype(np.float32)
    with torch.no_grad():
        for p, (a, b), shp in zip(net.parameters(), offsets, shapes):
            p.copy_(torch.from_numpy(f32[a:b]).reshape(shp))


def inroom_frac(room: SRCT, v64: np.ndarray) -> float:
    vn = float(np.linalg.norm(v64))
    if vn == 0.0:
        return 0.0
    return float(np.linalg.norm(room.project(v64)) / vn)


def binom_p_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial sign test (symmetric-tail form,
    P=0.5): p = P(X <= min(k,n-k)) + P(X >= max(k,n-k))."""
    if n <= 0:
        return 1.0
    lo, hi = min(k, n - k), max(k, n - k)
    denom = n * math.log(2.0)
    p = 0.0
    for i in range(0, lo + 1):
        p += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1)
                      - math.lgamma(n - i + 1) - denom)
    for i in range(hi, n + 1):
        p += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1)
                      - math.lgamma(n - i + 1) - denom)
    return min(max(p, 0.0), 1.0)


def skew_g1(x: np.ndarray) -> float:
    """Fisher-Pearson population skewness (biased g1)."""
    m = x.mean()
    d = x - m
    m2 = float((d ** 2).mean())
    if m2 <= 0.0:
        return 0.0
    return float((d ** 3).mean()) / m2 ** 1.5


def quantiles(x: np.ndarray) -> dict:
    qs = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
    return {f"q{int(q * 100):02d}": float(np.quantile(x, q)) for q in qs}


@torch.no_grad()
def last_logits(net, ids: torch.Tensor, bs: int = BS) -> np.ndarray:
    """The battery's own read convention: last-position logits,
    fp32 forward, fp64 cast downstream."""
    net.eval()
    outs = []
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        outs.append(lg[:, -1, :].clone())
    return torch.cat(outs).numpy().astype(np.float64)


def softmax64(L: np.ndarray) -> np.ndarray:
    z = L - L.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def battery_effect(Lb: np.ndarray, Lt: np.ndarray, zid: int) -> dict:
    """The frozen per-context battery read: identity stability, the
    margin-change distribution, the entropy-change distribution, the
    winner-probability effect, and the Z-bias excess."""
    n = Lb.shape[0]
    top1b = Lb.argmax(axis=1)
    top1t = Lt.argmax(axis=1)
    sb, st = np.sort(Lb, axis=1), np.sort(Lt, axis=1)
    mb = sb[:, -1] - sb[:, -2]
    mt = st[:, -1] - st[:, -2]
    dm = mt - mb                                    # Δmargin
    pb, pt = softmax64(Lb), softmax64(Lt)
    idx = np.arange(n)
    Hb = -(pb * np.log(np.clip(pb, 1e-300, None))).sum(axis=1)
    Ht = -(pt * np.log(np.clip(pt, 1e-300, None))).sum(axis=1)
    dH = Ht - Hb                                    # Δentropy (nats)
    dpw = pt[idx, top1b] - pb[idx, top1b]           # Δp(base winner)
    dlog_z = Lt[:, zid] - Lb[:, zid]
    dlog_mean = (Lt - Lb).mean(axis=1)
    dbias_z = dlog_z - dlog_mean                    # the Z-bias excess
    stable = top1b == top1t
    squeeze = dm < -SQUEEZE_REL * np.maximum(mb, 1e-12)
    flip_or_squeeze = (~stable) | squeeze
    neg = int((dm < 0).sum())
    pos = int((dm > 0).sum())
    nzc = neg + pos
    sign_p = binom_p_two_sided(neg, nzc)
    zw_mask = top1b == zid
    out = {
        "n_contexts": int(n),
        "top1_stability": float(stable.mean()),
        "flip_frac": float((~stable).mean()),
        "squeeze_frac": float(squeeze.mean()),
        "flip_or_squeeze_frac": float(flip_or_squeeze.mean()),
        "dmargin": {
            "mean": float(dm.mean()), "median": float(np.median(dm)),
            "std": float(dm.std()), "skew_g1": skew_g1(dm),
            "mean_abs": float(np.abs(dm).mean()),
            "neg_share_of_nonzero": (neg / nzc) if nzc else 0.5,
            "neg": neg, "pos": pos, "sign_test_p_two_sided": sign_p,
            "quantiles": quantiles(dm), "values": dm.tolist(),
        },
        "dentropy": {
            "mean": float(dH.mean()), "median": float(np.median(dH)),
            "std": float(dH.std()), "skew_g1": skew_g1(dH),
            "quantiles": quantiles(dH), "values": dH.tolist(),
        },
        "dp_winner": {
            "mean": float(dpw.mean()), "median": float(np.median(dpw)),
            "std": float(dpw.std()),
            "frac_negative": float((dpw < 0).mean()),
            "quantiles": quantiles(dpw), "values": dpw.tolist(),
        },
        "z_bias_excess": {
            "mean": float(dbias_z.mean()), "median": float(np.median(dbias_z)),
            "std": float(dbias_z.std()),
            "frac_positive": float((dbias_z > 0).mean()),
            "quantiles": quantiles(dbias_z), "values": dbias_z.tolist(),
        },
        "dlog_z_mean": float(dlog_z.mean()),
        "dlog_mean_of_all_tokens": float(dlog_mean.mean()),
        "z_winner_contexts": int(zw_mask.sum()),
        "z_winner_dmargin_mean": (float(dm[zw_mask].mean())
                                  if zw_mask.any() else None),
        "nonz_winner_dmargin_mean": (float(dm[~zw_mask].mean())
                                     if (~zw_mask).any() else None),
    }
    return out


# ======================================================================
# MAIN
# ======================================================================
def main() -> None:
    log(f"X16 — THE OFF-TARGET SPECIFICITY PROBE (smoke={SMOKE}) -> {RD}")
    metrics["birth_commit"] = git_head()
    log(f"trigger c={TRIGGER_SCALE!r}; battery n={BATTERY_N} "
        f"(seed {BATTERY_SEED}); control gen {RANDOM_CTRL_SEED}")
    save_json_partial("startup (bars registered, committed at birth)")
    set_seed(GLOBAL_SEED)             # global init only; every RNG is its own

    # ================= P0: the protocol rebuild (e311's gates) ==========
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
    val_text = "".join(itos[int(i)] for i in val_ids)
    corpus_zeph = train_text.count("ZEPH") + val_text.count("ZEPH")
    G_NAMEFREE = {"corpus_zeph_count": corpus_zeph,
                  "pass": bool(corpus_zeph == 0)}
    assert G_NAMEFREE["pass"], f"corpus contains the name: {G_NAMEFREE}"

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

    bat_ids, held_ids = {}, {}
    for j in G1.GEOS:
        cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
        hs = [train_text[p - G1.PRE - j: p] for p, _ in held_occ]
        held_ids[j] = torch.stack([corpus.encode(c) for c in hs])
    G_BATTERYGEO = {
        "shapes": {f"g{j:+d}": list(bat_ids[j].shape) for j in G1.GEOS},
        "pass": bool(list(bat_ids[-12].shape) == [60, G1.PRE - 12]
                     and list(bat_ids[0].shape) == [60, G1.PRE]
                     and list(bat_ids[12].shape) == [60, G1.PRE + 12]),
    }
    assert G_BATTERYGEO["pass"], f"battery geometry drift: {G_BATTERYGEO}"
    gm12_ids, g0_ids = bat_ids[-12], bat_ids[0]
    metrics["gates"].update({"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                             "G_BATTERYGEO": G_BATTERYGEO})
    log(f"P0: protocol gates PASS (namefree; splice 19+41; battery "
        f"geometry; g0 x{tuple(g0_ids.shape)})")
    save_json_partial("P0 protocol gates PASSED")

    # ================= P0b: the parents hard-bound (Rule 12) ============
    e311m = json.loads(E311_METRICS.read_text(encoding="utf-8"))
    e311_trig = e311m["gates"]["G_TRIGGER"]
    e311_armA = e311m["arms"]["PURE-HIJACK"]["reads"]
    e311_vmd5 = md5of(E311_VECTORS)
    G_PARENTS = {
        "e311_metrics": {"path": str(E311_METRICS),
                         "md5": md5of(E311_METRICS),
                         "bound_md5": E311_METRICS_MD5,
                         "verdict": e311m["adjudication"]["word"],
                         "note": "THE KNOB's committed record (the trigger "
                                 "construction, the +32% anchor, the arm-A "
                                 "state md5)"},
        "e311_vectors": {"path": str(E311_VECTORS), "md5": e311_vmd5,
                         "bound_md5": E311_VECTORS_MD5,
                         "meta_experiment": "e311",
                         "note": "the committed fp64 trigger + seed"},
        "e311_literals_crosscheck": {
            "armA_host_g0": {"mine": E311_ARM_A_HOST_G0,
                             "theirs": e311_armA["host_g0"]},
            "armA_flat_md5": {"mine": E311_ARM_A_FLAT_MD5,
                              "theirs": e311m["arms"]["PURE-HIJACK"]
                              ["reads"]["flat_md5"]},
            "trigger_bytes_md5": {"mine": E311_TRIGGER_BYTES_MD5},
            "trigger_inroom": {"mine": E311_TRIG_INROOM,
                               "theirs": e311_trig["in_room_frac_of_trigger"]},
            "trigger_scale": {"mine": TRIGGER_SCALE,
                              "theirs": e311_trig["trigger_scale_c"]},
            "host_baseline_g0": {"mine": FACT_BASELINE_G0,
                                 "theirs": e311m["gates"]["G_FACTLOAD"]
                                 ["read_g0"]["committed"]},
        },
        "pass": bool(md5of(E311_METRICS) == E311_METRICS_MD5
                     and e311_vmd5 == E311_VECTORS_MD5
                     and e311m["adjudication"]["word"] == E311_METRICS_VERDICT
                     and e311_armA["host_g0"] == E311_ARM_A_HOST_G0
                     and e311m["arms"]["PURE-HIJACK"]["reads"]["flat_md5"]
                     == E311_ARM_A_FLAT_MD5
                     and e311_trig["in_room_frac_of_trigger"]
                     == E311_TRIG_INROOM
                     and e311_trig["trigger_scale_c"] == TRIGGER_SCALE),
    }
    assert G_PARENTS["pass"], f"parent bind failed: {G_PARENTS}"
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    log("P0b: G_PARENTS PASS (e311 metrics + vectors md5-bound; the +32% "
        "anchor, the arm-A state md5, and the trigger literals cross-checked "
        "against the committed record)")
    save_json_partial("P0b parents hard-bound")

    # ================= P1: the room (the out-of-room definition) ========
    room = SRCT(N_PARAMS, ROOM_K, ROOM_SEED_D, ROOM_SEED_S)
    rooms264 = torch.load(CKPT_DIR / ROOMS264_CK, map_location="cpu",
                          weights_only=False)

    def _to_np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    D264 = _to_np(rooms264["model"]["K10K"]["D_int8"]).astype(np.float64)
    S264 = _to_np(rooms264["model"]["K10K"]["S"])
    del rooms264
    cert_rng = np.random.default_rng(CERT_SEED)
    idem, kept2 = [], []
    for _ in range(2):
        x = cert_rng.standard_normal(N_PARAMS)
        px = room.project(x)
        ppx = room.project(px)
        idem.append(float(np.linalg.norm(ppx - px) / np.linalg.norm(px)))
        kept2.append(float((px @ px) / (x @ x)))
    G_ROOM = {
        "form": "the host's own committed K10K room (SRCT k=10,000, seeds "
                "26113/26114): the +-1 diagonal and the index set "
                "bit-identical to e264_rooms.pt; light in-cell "
                "certification (2 probes, idempotency + kept^2 ~ k/N)",
        "D_bit_equal": bool(np.array_equal(room.D, D264)),
        "S_bit_equal": bool(np.array_equal(room.S, S264)),
        "e264_rooms_md5": md5of(CKPT_DIR / ROOMS264_CK),
        "bound_md5": ROOMS264_MD5,
        "idempotency_max": max(idem),
        "kept2_mean": float(np.mean(kept2)),
        "kept2_expect": ROOM_K / N_PARAMS,
        "pass": bool(np.array_equal(room.D, D264)
                     and np.array_equal(room.S, S264)
                     and md5of(CKPT_DIR / ROOMS264_CK) == ROOMS264_MD5
                     and max(idem) <= 1e-8
                     and abs(float(np.mean(kept2))
                             - ROOM_K / N_PARAMS)
                     <= 5.0 * math.sqrt(2.0 * ROOM_K) / N_PARAMS),
    }
    assert G_ROOM["pass"], f"room bind failed: {G_ROOM}"
    metrics["gates"]["G_ROOM"] = G_ROOM
    log(f"P1 G_ROOM: K10K rebuilt + D/S BIT-BOUND vs e264_rooms.pt "
        f"(idem {max(idem):.1e}; kept2 {np.mean(kept2):.6f} vs "
        f"{ROOM_K / N_PARAMS:.6f}): PASS")
    save_json_partial("P1 the room rebuilt + bit-bound")

    # ================= P2: THE HOST (loaded bit-exact + gated) ==========
    fact_art = torch.load(CKPT_DIR / FACT_CK, map_location="cpu",
                          weights_only=False)
    fact_sd = {k: v.detach().clone() for k, v in fact_art["model"].items()}
    del fact_art
    fact_net = G1.evl_load(fact_sd)
    fact_flat = flat_params_cpu(fact_net)
    fact_flat_np = fact_flat.double().numpy().astype(np.float64)
    fact_flat_md5 = hashlib.md5(fact_flat.numpy().tobytes()).hexdigest()
    fact_g0 = G1.battery_cell(fact_net, g0_ids, zid)["mean_pz"]
    fact_gm12 = G1.battery_cell(fact_net, gm12_ids, zid)["mean_pz"]
    # the trigger's chosen window co-bind (e311: window 0, argmax IS Z)
    with torch.no_grad():
        lg0, _ = fact_net(g0_ids[0:1])
    w0_amax = int(lg0[0, -1].argmax())
    w0_pz = float(F.softmax(lg0[0, -1].detach(), -1)[zid])
    G_FACTLOAD = {
        "form": "e311's HOST organism loaded BIT-EXACT and gated THREE "
                "ways: (1) the artifact (md5/size), (2) the loaded "
                "state's flat-md5 (vs e311's committed record), (3) the "
                "behavioral read (g0/gm12 vs the committed literals, "
                "bars 2e-6/1e-5)",
        "artifact_md5": md5of(CKPT_DIR / FACT_CK),
        "bound_md5": FACT_MD5,
        "artifact_size": (CKPT_DIR / FACT_CK).stat().st_size,
        "bound_size": FACT_SIZE,
        "flat_md5": fact_flat_md5, "bound_flat_md5": FACT_FLAT_MD5,
        "read_g0": {"mine": fact_g0, "committed": FACT_BASELINE_G0,
                    "abs_diff": abs(fact_g0 - FACT_BASELINE_G0)},
        "read_gm12": {"mine": fact_gm12, "committed": FACT_BASELINE_GM12,
                      "abs_diff": abs(fact_gm12 - FACT_BASELINE_GM12)},
        "trigger_window_co_bind": {"window": E311_TRIG_WINDOW,
                                   "argmax_is_z": bool(w0_amax == zid),
                                   "window_pz": w0_pz,
                                   "e311_window_pz": E311_TRIG_WINDOW_PZ},
        "pass": bool(md5of(CKPT_DIR / FACT_CK) == FACT_MD5
                     and (CKPT_DIR / FACT_CK).stat().st_size == FACT_SIZE
                     and fact_flat_md5 == FACT_FLAT_MD5
                     and abs(fact_g0 - FACT_BASELINE_G0) <= FACT_READ_TOL_G0
                     and abs(fact_gm12 - FACT_BASELINE_GM12)
                     <= FACT_READ_TOL_GM12
                     and w0_amax == zid
                     and abs(w0_pz - E311_TRIG_WINDOW_PZ) <= 1e-6),
    }
    assert G_FACTLOAD["pass"], f"G_FACTLOAD FAILED: {G_FACTLOAD}"
    metrics["gates"]["G_FACTLOAD"] = G_FACTLOAD
    log(f"P2 G_FACTLOAD: the HOST loads — g0 {fact_g0:.10f} (|d| "
        f"{abs(fact_g0 - FACT_BASELINE_G0):.1e}); flat md5 "
        f"{fact_flat_md5[:8]} == e311's; window-0 argmax IS Z "
        f"(p(Z) {w0_pz:.4f} vs e311's {E311_TRIG_WINDOW_PZ:.4f}): PASS")
    save_json_partial("P2 the HOST loaded bit-exact + gated")

    # ================= P3: THE TRIGGER (the committed knob) =============
    vecs = torch.load(E311_VECTORS, map_location="cpu", weights_only=False)
    trigger = vecs["trigger"].numpy().astype(np.float64).copy()
    del vecs
    trig_md5 = hashlib.md5(trigger.tobytes()).hexdigest()
    trig_norm = float(np.linalg.norm(trigger))
    trig_inroom = inroom_frac(room, trigger)
    G_TRIGBIND = {
        "form": "the committed e311 trigger (the host's own read-gradient "
                "direction at g0 window 0, projected OUT of the room, "
                "scaled to c = 0.0005 x ||dW_host||): loaded from the "
                "committed artifact and re-verified against THIS cell's "
                "rebuilt room projector",
        "bytes_md5": trig_md5, "bound_md5": E311_TRIGGER_BYTES_MD5,
        "norm": trig_norm, "bound_norm": TRIGGER_SCALE,
        "in_room_frac": trig_inroom, "e311_committed_in_room_frac":
            E311_TRIG_INROOM,
        "zero_bearer_bar": ZERO_BEARER_BAR,
        "pass": bool(trig_md5 == E311_TRIGGER_BYTES_MD5
                     and abs(trig_norm - TRIGGER_SCALE) <= 1e-12
                     and trig_inroom <= ZERO_BEARER_BAR),
    }
    assert G_TRIGBIND["pass"], f"G_TRIGBIND FAILED: {G_TRIGBIND}"
    metrics["gates"]["G_TRIGBIND"] = G_TRIGBIND
    log(f"P3 G_TRIGBIND: the committed trigger loads (md5 {trig_md5[:8]}; "
        f"||.|| {trig_norm:.16f}); its in-room fraction on the rebuilt "
        f"projector {trig_inroom:.2e} (e311 committed "
        f"{E311_TRIG_INROOM:.2e}): PASS")
    save_json_partial("P3 the trigger bound")

    # ================= P4: THE RANDOM CONTROL (the alignment arm) ======
    gen = torch.Generator().manual_seed(RANDOM_CTRL_SEED)
    x_ctrl = torch.randn(N_PARAMS, generator=gen).double().numpy() \
        .astype(np.float64).copy()
    x_ctrl_raw_inroom = inroom_frac(room, x_ctrl)
    px = room.project(x_ctrl)
    u = x_ctrl - px
    u /= float(np.linalg.norm(u))
    ctrl_vec = TRIGGER_SCALE * u
    ctrl_md5 = hashlib.md5(ctrl_vec.tobytes()).hexdigest()
    ctrl_inroom = inroom_frac(room, ctrl_vec)
    ctrl_norm = float(np.linalg.norm(ctrl_vec))
    cos_tc = float(trigger @ ctrl_vec
                   / (trig_norm * ctrl_norm))
    G_CTRL = {
        "form": "the random control: ONE fresh N(0,1)^N draw (generator "
                f"seed {RANDOM_CTRL_SEED}), projected OUT of the room, "
                "normalized, scaled to the SAME c — matched dose, matched "
                "zero-bearer class, ZERO alignment to the read gradient",
        "raw_draw_in_room_frac": x_ctrl_raw_inroom,
        "raw_draw_in_room_frac_expect": math.sqrt(ROOM_K / N_PARAMS),
        "in_room_frac": ctrl_inroom,
        "zero_bearer_bar": ZERO_BEARER_BAR,
        "norm": ctrl_norm, "bound_norm": TRIGGER_SCALE,
        "bytes_md5": ctrl_md5,
        "cos_to_trigger": cos_tc,
        "cos_chance_scale": 1.0 / math.sqrt(N_PARAMS),
        "pass": bool(ctrl_inroom <= ZERO_BEARER_BAR
                     and abs(ctrl_norm - TRIGGER_SCALE) <= 1e-12),
    }
    assert G_CTRL["pass"], f"G_CTRL FAILED: {G_CTRL}"
    metrics["gates"]["G_CTRL"] = G_CTRL
    log(f"P4 G_CTRL: the random control built — raw draw in-room "
        f"{x_ctrl_raw_inroom:.4f} (chance {math.sqrt(ROOM_K / N_PARAMS):.4f}); "
        f"projected-out in-room {ctrl_inroom:.2e} (matched to the "
        f"trigger's {trig_inroom:.2e}); ||.|| {ctrl_norm:.16f}; "
        f"cos(trigger, ctrl) {cos_tc:+.2e}: PASS")
    save_json_partial("P4 the random control built")

    # ================= P5: THE ARMS + THE CALIBRATION ANCHOR ============
    offsets, shapes = [], []
    off = 0
    for p in fact_net.parameters():
        offsets.append((off, off + p.numel()))
        shapes.append(tuple(p.shape))
        off += p.numel()
    assert off == N_PARAMS

    def build_state(vec: np.ndarray | None):
        net = G1.evl_load(fact_sd)
        if vec is not None:
            apply_flat64(net, fact_flat_np + vec, offsets, shapes)
        return net, hashlib.md5(
            flat_params_cpu(net).numpy().tobytes()).hexdigest()

    trig_net, trig_flat_md5 = build_state(trigger)
    ctrl_net, ctrl_flat_md5 = build_state(ctrl_vec)
    base_flat_md5 = fact_flat_md5          # the unmodified host (gated P2)
    trig_g0 = G1.battery_cell(trig_net, g0_ids, zid)
    ctrl_g0 = G1.battery_cell(ctrl_net, g0_ids, zid)
    G_ANCHOR = {
        "form": "THE CALIBRATION ANCHOR: the trigger applied in e311's "
                "exact convention MUST reproduce e311's committed arm-A "
                "state (flat md5) AND its committed +32% host read "
                "(bit-exact; the 2e-6 family tolerance as the gate, the "
                "exact diff disclosed)",
        "trig_flat_md5": trig_flat_md5,
        "bound_armA_flat_md5": E311_ARM_A_FLAT_MD5,
        "trig_host_g0": {"mine": trig_g0["mean_pz"],
                         "committed": E311_ARM_A_HOST_G0,
                         "abs_diff": abs(trig_g0["mean_pz"]
                                         - E311_ARM_A_HOST_G0)},
        "trig_host_g0_frac_argmax": {
            "mine": trig_g0["frac_argmax_z"],
            "committed": E311_ARM_A_FRAC_ARGMAX},
        "ctrl_flat_md5": ctrl_flat_md5,
        "ctrl_host_g0": ctrl_g0["mean_pz"],
        "base_flat_md5": base_flat_md5,
        "boost_trig": trig_g0["mean_pz"] - fact_g0,
        "boost_ctrl": ctrl_g0["mean_pz"] - fact_g0,
        "committed_boost": E311_ARM_A_HOST_G0 - FACT_BASELINE_G0,
        "pass": bool(trig_flat_md5 == E311_ARM_A_FLAT_MD5
                     and abs(trig_g0["mean_pz"] - E311_ARM_A_HOST_G0)
                     <= FACT_READ_TOL_G0
                     and abs(trig_g0["frac_argmax_z"]
                             - E311_ARM_A_FRAC_ARGMAX) <= 1e-9),
    }
    assert G_ANCHOR["pass"], f"G_ANCHOR FAILED: {G_ANCHOR}"
    metrics["gates"]["G_ANCHOR"] = G_ANCHOR
    log(f"P5 G_ANCHOR: THE +32% REPRODUCES — host g0 {trig_g0['mean_pz']:.16f} "
        f"vs committed {E311_ARM_A_HOST_G0:.16f} (|d| "
        f"{abs(trig_g0['mean_pz'] - E311_ARM_A_HOST_G0):.1e}); arm-A flat "
        f"md5 {trig_flat_md5[:8]} == committed; argmax frac "
        f"{trig_g0['frac_argmax_z']:.4f}: PASS")
    log(f"    the control at the same battery: host g0 "
        f"{ctrl_g0['mean_pz']:.6f} (boost {ctrl_g0['mean_pz'] - fact_g0:+.6f} "
        f"vs the trigger's {trig_g0['mean_pz'] - fact_g0:+.6f})")
    save_json_partial("P5 the arms built + the anchor reproduced")

    # ================= P6: THE GENERAL CORPUS BATTERY ===================
    log("=" * 78)
    log(f"P6: THE GENERAL BATTERY — {BATTERY_N} name-free validation "
        f"windows (val_windows machinery, seed {BATTERY_SEED}), "
        f"last-position read")
    bat_x, bat_y = G1.val_windows(val_ids, val_text, BATTERY_N, BATTERY_SEED,
                                  block=BLOCK)
    Lb = last_logits(fact_net, bat_x)
    Lt = last_logits(trig_net, bat_x)
    Lc = last_logits(ctrl_net, bat_x)
    y_last = bat_y[:, -1].numpy()
    pb64 = softmax64(Lb)
    nll_base = float(-np.log(np.clip(
        pb64[np.arange(BATTERY_N), y_last], 1e-300, None)).mean())
    uniq = np.unique(bat_x.numpy(), axis=0).shape[0]
    winners = Lb.argmax(axis=1)
    diversity = int(np.unique(winners).size)
    mean_wp = float(pb64[np.arange(BATTERY_N), winners].mean())
    zeph_leak = sum(1 for i in range(BATTERY_N)
                    if "ZEPH" in corpus.decode(bat_x[i]))
    G_BATT = {
        "form": "battery sanity: the drawn windows are name-free, unique, "
                "finite, and behaviorally diverse (the instrument reads "
                "real decisions, not a degenerate mode)",
        "n": BATTERY_N, "block": BLOCK, "seed": BATTERY_SEED,
        "unique_windows": int(uniq),
        "zeph_leak_windows": int(zeph_leak),
        "logits_finite": bool(np.isfinite(Lb).all() and np.isfinite(Lt).all()
                              and np.isfinite(Lc).all()),
        "distinct_base_winners": diversity,
        "diversity_bar": BATT_DIVERSITY_BAR,
        "base_winner_mean_prob": mean_wp,
        "base_lastpos_nll": nll_base,
        "pass": bool(uniq == BATTERY_N and zeph_leak == 0
                     and np.isfinite(Lb).all() and np.isfinite(Lt).all()
                     and np.isfinite(Lc).all()
                     and diversity >= BATT_DIVERSITY_BAR
                     and 0.05 < mean_wp < 0.9999),
    }
    assert G_BATT["pass"], f"G_BATT FAILED: {G_BATT}"
    metrics["gates"]["G_BATT"] = G_BATT
    log(f"P6 G_BATT: {BATTERY_N} unique name-free windows; "
        f"{diversity} distinct base winners; mean winner prob "
        f"{mean_wp:.3f}; base last-position NLL {nll_base:.3f}: PASS")

    eff_trig = battery_effect(Lb, Lt, zid)
    eff_ctrl = battery_effect(Lb, Lc, zid)
    metrics["battery"] = {
        "form": f"{BATTERY_N} general corpus contexts (val split, block "
                f"{BLOCK}, seed {BATTERY_SEED}); per-context last-position "
                "reads: base (the host organism) vs treated",
        "nll_base": nll_base,
        "base_winner_mean_prob": mean_wp,
        "distinct_base_winners": diversity,
        "z_winner_contexts": eff_trig["z_winner_contexts"],
    }
    metrics["arms_battery"] = {
        "TRIGGER": {"form": "theta_host + committed trigger (e311's knob)",
                    "flat_md5": trig_flat_md5, "effect": eff_trig},
        "RANDOM-CONTROL": {"form": "theta_host + random out-of-room 1-dim "
                                   "direction at the same c (seed "
                                   f"{RANDOM_CTRL_SEED})",
                           "flat_md5": ctrl_flat_md5, "effect": eff_ctrl},
    }
    log(f"  TRIGGER : stability {eff_trig['top1_stability']:.4f} | "
        f"flip-or-squeeze {eff_trig['flip_or_squeeze_frac']:.4f} | "
        f"dmargin mean {eff_trig['dmargin']['mean']:+.4f} "
        f"(neg share {eff_trig['dmargin']['neg_share_of_nonzero']:.3f}, "
        f"sign p {eff_trig['dmargin']['sign_test_p_two_sided']:.2e}) | "
        f"dH mean {eff_trig['dentropy']['mean']:+.5f} | dp(win) mean "
        f"{eff_trig['dp_winner']['mean']:+.5f} | Z-bias excess "
        f"{eff_trig['z_bias_excess']['mean']:+.5f}")
    log(f"  CONTROL : stability {eff_ctrl['top1_stability']:.4f} | "
        f"flip-or-squeeze {eff_ctrl['flip_or_squeeze_frac']:.4f} | "
        f"dmargin mean {eff_ctrl['dmargin']['mean']:+.4f} "
        f"(neg share {eff_ctrl['dmargin']['neg_share_of_nonzero']:.3f}, "
        f"sign p {eff_ctrl['dmargin']['sign_test_p_two_sided']:.2e}) | "
        f"dH mean {eff_ctrl['dentropy']['mean']:+.5f} | dp(win) mean "
        f"{eff_ctrl['dp_winner']['mean']:+.5f} | Z-bias excess "
        f"{eff_ctrl['z_bias_excess']['mean']:+.5f}")
    save_json_partial("P6 the general battery read")

    # ---- the host's own 60 g0 contexts (the on-target co-report) ------
    Lb_g0 = last_logits(fact_net, g0_ids)
    Lt_g0 = last_logits(trig_net, g0_ids)
    Lc_g0 = last_logits(ctrl_net, g0_ids)
    eff_trig_g0 = battery_effect(Lb_g0, Lt_g0, zid)
    eff_ctrl_g0 = battery_effect(Lb_g0, Lc_g0, zid)
    metrics["g0_cotarget"] = {
        "form": "the on-target column (co-report): the same per-context "
                "reads at the host's own 60 g0 contexts — where the knob "
                "was born; the base battery mean p(Z) "
                f"{fact_g0:.10f} (committed {FACT_BASELINE_G0:.10f})",
        "base_pz_mean": float(softmax64(Lb_g0)[:, zid].mean()),
        "trig_pz_mean": float(softmax64(Lt_g0)[:, zid].mean()),
        "ctrl_pz_mean": float(softmax64(Lc_g0)[:, zid].mean()),
        "committed_trig": E311_ARM_A_HOST_G0,
        "TRIGGER": eff_trig_g0,
        "RANDOM-CONTROL": eff_ctrl_g0,
    }
    log(f"  on-target (g0 x60): p(Z) base {fact_g0:.4f} -> trig "
        f"{metrics['g0_cotarget']['trig_pz_mean']:.4f} / ctrl "
        f"{metrics['g0_cotarget']['ctrl_pz_mean']:.4f}; trig dmargin mean "
        f"{eff_trig_g0['dmargin']['mean']:+.4f}, dH "
        f"{eff_trig_g0['dentropy']['mean']:+.5f}")
    save_json_partial("P6b the on-target g0 co-report read")

    # ================= P7: THE ADJUDICATION (frozen bars) ===============
    log("=" * 78)
    dm_t = eff_trig["dmargin"]
    bias_cond = (eff_trig["flip_or_squeeze_frac"] >= FLIP_SQUEEZE_BAR
                 and dm_t["neg_share_of_nonzero"] > 0.5
                 and dm_t["sign_test_p_two_sided"] < SIGN_P_BAR)
    vol_cond = (eff_trig["top1_stability"] >= VOLUME_STABILITY_BAR
                and eff_trig["dmargin"]["mean"] > 0
                and eff_trig["dentropy"]["mean"] < 0
                and eff_trig["flip_or_squeeze_frac"] < FLIP_SQUEEZE_BAR)
    if bias_cond:
        word = "TARGET-SPECIFIC-BIAS"
        clause = (f"at {eff_trig['flip_or_squeeze_frac']:.3f} of general "
                  f"contexts the top-1 flips or the winner margin falls "
                  f">10% (bar >= {FLIP_SQUEEZE_BAR}), AND the "
                  "margin-change distribution is significantly skewed "
                  f"(neg share {dm_t['neg_share_of_nonzero']:.3f}, sign "
                  f"test p {dm_t['sign_test_p_two_sided']:.2e} < "
                  f"{SIGN_P_BAR}; skew g1 {dm_t['skew_g1']:+.3f}) — the "
                  "knob is an out-of-room LOGIT BIAS; T279's 'volume "
                  "channel' re-names to 'bias + noise' (both out-of-room, "
                  "neither content)")
    elif vol_cond:
        word = "CONTENT-INDEPENDENT-VOLUME"
        clause = (f"top-1 stability {eff_trig['top1_stability']:.4f} "
                  f">= {VOLUME_STABILITY_BAR}, margins and entropies shift "
                  "symmetrically upward in confidence (mean dmargin "
                  f"{eff_trig['dmargin']['mean']:+.5f} > 0; mean dentropy "
                  f"{eff_trig['dentropy']['mean']:+.5f} < 0; "
                  f"flip-or-squeeze {eff_trig['flip_or_squeeze_frac']:.4f} "
                  f"< {FLIP_SQUEEZE_BAR}) — the knob is a true volume "
                  "dial; W049's calibration-dial design (Q3) proceeds as "
                  "planned")
    else:
        word = "MIXED"
        clause = ("the table verbatim: stability "
                  f"{eff_trig['top1_stability']:.4f}, flip-or-squeeze "
                  f"{eff_trig['flip_or_squeeze_frac']:.4f} (bar "
                  f"{FLIP_SQUEEZE_BAR}), dmargin mean "
                  f"{eff_trig['dmargin']['mean']:+.5f}, neg share "
                  f"{dm_t['neg_share_of_nonzero']:.3f}, sign p "
                  f"{dm_t['sign_test_p_two_sided']:.2e}, dentropy mean "
                  f"{eff_trig['dentropy']['mean']:+.5f}, Z-bias excess "
                  f"{eff_trig['z_bias_excess']['mean']:+.5f} — neither "
                  "frozen bar met; the mechanism reads live in the table")
    # the secondary (alignment)
    boost_trig = trig_g0["mean_pz"] - fact_g0
    boost_ctrl = ctrl_g0["mean_pz"] - fact_g0
    align_cond = (boost_ctrl <= ALIGN_BOOST_FRAC * boost_trig
                  and eff_ctrl["dmargin"]["mean_abs"]
                  <= ALIGN_MARGIN_FRAC * eff_trig["dmargin"]["mean_abs"])
    if align_cond:
        sword = "ALIGNMENT-MATTERS"
        sclause = (f"the random-direction control shows a materially "
                   f"smaller effect at matched norm: host-g0 boost "
                   f"{boost_ctrl:+.6f} <= {ALIGN_BOOST_FRAC:.0%} x the "
                   f"trigger's {boost_trig:+.6f}; mean |dmargin| "
                   f"{eff_ctrl['dmargin']['mean_abs']:.5f} <= "
                   f"{ALIGN_MARGIN_FRAC:.0%} x the trigger's "
                   f"{eff_trig['dmargin']['mean_abs']:.5f} — the +32% "
                   "needed the read-aligned direction")
    else:
        sword = "ALIGNMENT-DEAD"
        sclause = (f"the random control's effect is NOT materially smaller "
                   f"(host-g0 boost {boost_ctrl:+.6f} vs the trigger's "
                   f"{boost_trig:+.6f}; mean |dmargin| "
                   f"{eff_ctrl['dmargin']['mean_abs']:.5f} vs "
                   f"{eff_trig['dmargin']['mean_abs']:.5f}) — any "
                   "out-of-room 1-dim direction at this norm does the "
                   "same; alignment is not the operative variable")
    # predictions scoring (registered at birth)
    p_a_fired = bool(word == "TARGET-SPECIFIC-BIAS"
                     and sword == "ALIGNMENT-MATTERS")
    bias_shaped = bool(dm_t["neg_share_of_nonzero"] > 0.5
                       and dm_t["sign_test_p_two_sided"] < SIGN_P_BAR)
    p_b_fired = bool(word == "MIXED" and bias_shaped
                     and sword == "ALIGNMENT-MATTERS")
    metrics["adjudication"] = {
        "word": word, "clause": clause,
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "conditions": {
            "bias_condition": {
                "flip_or_squeeze_frac": eff_trig["flip_or_squeeze_frac"],
                "bar": FLIP_SQUEEZE_BAR,
                "neg_share": dm_t["neg_share_of_nonzero"],
                "sign_p": dm_t["sign_test_p_two_sided"],
                "p_bar": SIGN_P_BAR, "met": bool(bias_cond)},
            "volume_condition": {
                "stability": eff_trig["top1_stability"],
                "stability_bar": VOLUME_STABILITY_BAR,
                "mean_dmargin": eff_trig["dmargin"]["mean"],
                "mean_dentropy": eff_trig["dentropy"]["mean"],
                "flip_or_squeeze_frac": eff_trig["flip_or_squeeze_frac"],
                "met": bool(vol_cond)},
            "smoke": SMOKE,
        },
        "secondary": {
            "word": sword, "clause": sclause,
            "boost_trig": boost_trig, "boost_ctrl": boost_ctrl,
            "boost_ctrl_over_trig": (boost_ctrl / boost_trig
                                     if boost_trig != 0 else None),
            "mean_abs_dmargin_trig": eff_trig["dmargin"]["mean_abs"],
            "mean_abs_dmargin_ctrl": eff_ctrl["dmargin"]["mean_abs"],
            "cos_trigger_to_control": cos_tc,
        },
        "predictions_scored": {
            "P-x16a_lab": {"fired": p_a_fired,
                           "text": REGISTERED["predictions"]["P-x16a_lab"]},
            "P-x16b_executor_counter": {
                "fired": p_b_fired,
                "text": REGISTERED["predictions"]["P-x16b_executor_counter"],
                "bias_shaped_evidence_present": bias_shaped},
        },
    }
    log(f"P7 ADJUDICATION (primary): {word} — {clause}")
    log(f"P7 SECONDARY: {sword} — {sclause}")
    log(f"P7 PREDICTIONS: P-x16a fired={p_a_fired}; P-x16b fired="
        f"{p_b_fired} (bias-shaped evidence present: {bias_shaped})")
    save_json_partial(f"P7 adjudicated: {word} / {sword}")

    # ================= P8: THE FIGURE + THE REPORT =======================
    log("=" * 78)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    # panel 1: the margin-change distributions
    ax = axes[0]
    dmt = np.array(eff_trig["dmargin"]["values"])
    dmc = np.array(eff_ctrl["dmargin"]["values"])
    lo = float(min(dmt.min(), dmc.min()))
    hi = float(max(dmt.max(), dmc.max()))
    bins = np.linspace(lo, hi, 41)
    ax.hist(dmt, bins=bins, color="#ee6677", alpha=0.65,
            label=f"trigger (neg {dm_t['neg_share_of_nonzero']:.2f}, "
                  f"p={dm_t['sign_test_p_two_sided']:.1e})")
    ax.hist(dmc, bins=bins, color="#4477aa", alpha=0.65,
            label=f"random ctrl (neg "
                  f"{eff_ctrl['dmargin']['neg_share_of_nonzero']:.2f}, "
                  f"p={eff_ctrl['dmargin']['sign_test_p_two_sided']:.1e})")
    ax.axvline(0.0, color="k", lw=1)
    ax.set_xlabel("Δ winner margin (post − pre, logit gap)")
    ax.set_ylabel("contexts")
    ax.set_title(f"margin-change distribution ({BATTERY_N} contexts)")
    ax.legend(fontsize=7)
    # panel 2: the entropy-change distributions
    ax = axes[1]
    dHt = np.array(eff_trig["dentropy"]["values"])
    dHc = np.array(eff_ctrl["dentropy"]["values"])
    lo = float(min(dHt.min(), dHc.min()))
    hi = float(max(dHt.max(), dHc.max()))
    bins = np.linspace(lo, hi, 41)
    ax.hist(dHt, bins=bins, color="#ee6677", alpha=0.65,
            label=f"trigger (mean {dHt.mean():+.5f})")
    ax.hist(dHc, bins=bins, color="#4477aa", alpha=0.65,
            label=f"random ctrl (mean {dHc.mean():+.5f})")
    ax.axvline(0.0, color="k", lw=1)
    ax.set_xlabel("Δ entropy (nats, post − pre)")
    ax.set_ylabel("contexts")
    ax.set_title("entropy-change distribution")
    ax.legend(fontsize=7)
    # panel 3: the host anchor
    ax = axes[2]
    vals = [fact_g0, trig_g0["mean_pz"], ctrl_g0["mean_pz"]]
    ax.bar(range(3), vals, color=["#999999", "#ee6677", "#4477aa"])
    ax.axhline(E311_ARM_A_HOST_G0, color="#ee6677", ls="--", lw=1.4,
               label=f"e311 committed anchor {E311_ARM_A_HOST_G0:.4f} "
                     "(+32%)")
    ax.axhline(FACT_BASELINE_G0, color="k", ls=":", lw=1,
               label=f"host baseline {FACT_BASELINE_G0:.4f}")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["host (base)", "host + trigger",
                        "host + random ctrl"], fontsize=8)
    ax.set_ylabel("battery p(Z), 60 g0 contexts")
    bit = ("bit-exact, |d| "
           f"{abs(trig_g0['mean_pz'] - E311_ARM_A_HOST_G0):.1e}")
    ax.set_title(f"the calibration anchor ({bit})")
    ax.legend(fontsize=7)
    # panel 4: the Z-bias excess
    ax = axes[3]
    bzt = np.array(eff_trig["z_bias_excess"]["values"])
    bzc = np.array(eff_ctrl["z_bias_excess"]["values"])
    lo = float(min(bzt.min(), bzc.min()))
    hi = float(max(bzt.max(), bzc.max()))
    bins = np.linspace(lo, hi, 41)
    ax.hist(bzt, bins=bins, color="#ee6677", alpha=0.65,
            label=f"trigger (mean {bzt.mean():+.5f}, "
                  f"{float((bzt > 0).mean()):.2f} > 0)")
    ax.hist(bzc, bins=bins, color="#4477aa", alpha=0.65,
            label=f"random ctrl (mean {bzc.mean():+.5f})")
    ax.axvline(0.0, color="k", lw=1)
    ax.set_xlabel("Δlogit(Z) − mean Δlogit (the Z-bias excess)")
    ax.set_ylabel("contexts")
    ax.set_title("the token-specific bias read")
    ax.legend(fontsize=7)
    verdict_word = word if not SMOKE else f"SMOKE/{word}"
    fig.suptitle(f"X16 THE OFF-TARGET SPECIFICITY PROBE — {verdict_word} / "
                 f"{sword if not SMOKE else 'SMOKE/' + sword}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig_path = RD / "x16_offtarget.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    log(f"P8 figure saved: {fig_path.name}")

    # ---- THE REPORT -----------------------------------------------------
    rep = [
        "# X16 — THE OFF-TARGET SPECIFICITY PROBE",
        "",
        f"* the primary verdict: **{word}** — {clause}",
        f"* the secondary: **{sword}** — {sclause}",
        "",
        "## Headline numbers",
        "",
        f"* ANCHOR: the committed +32% reproduces "
        f"{'BIT-EXACTLY' if abs(trig_g0['mean_pz'] - E311_ARM_A_HOST_G0) == 0 else 'within tolerance'} "
        f"({trig_g0['mean_pz']:.16f} vs committed "
        f"{E311_ARM_A_HOST_G0:.16f}, |d| "
        f"{abs(trig_g0['mean_pz'] - E311_ARM_A_HOST_G0):.1e}); the "
        f"applied state's flat md5 == e311's arm-A "
        f"({trig_flat_md5[:12]}).",
        f"* TRIGGER on the {BATTERY_N}-context general battery: top-1 "
        f"stability {eff_trig['top1_stability']:.4f} (bar: >= "
        f"{VOLUME_STABILITY_BAR} volume / flip-or-squeeze "
        f"{eff_trig['flip_or_squeeze_frac']:.4f} vs bar "
        f"{FLIP_SQUEEZE_BAR}); Δmargin mean "
        f"{eff_trig['dmargin']['mean']:+.5f}, neg share "
        f"{dm_t['neg_share_of_nonzero']:.3f}, sign-test p "
        f"{dm_t['sign_test_p_two_sided']:.2e}, skew "
        f"{dm_t['skew_g1']:+.3f}; Δentropy mean "
        f"{eff_trig['dentropy']['mean']:+.5f}; Δp(winner) mean "
        f"{eff_trig['dp_winner']['mean']:+.5f} "
        f"({eff_trig['dp_winner']['frac_negative']:.2f} negative); "
        f"Z-bias excess {eff_trig['z_bias_excess']['mean']:+.5f} "
        f"({eff_trig['z_bias_excess']['frac_positive']:.2f} of contexts "
        f"positive).",
        f"* RANDOM CONTROL: stability "
        f"{eff_ctrl['top1_stability']:.4f}; Δmargin mean "
        f"{eff_ctrl['dmargin']['mean']:+.5f} (mean |Δ| "
        f"{eff_ctrl['dmargin']['mean_abs']:.5f}); host-g0 boost "
        f"{boost_ctrl:+.6f} vs the trigger's {boost_trig:+.6f}; "
        f"cos(trigger, ctrl) {cos_tc:+.2e}.",
        f"* ON-TARGET (the 60 g0 contexts, co-report): p(Z) "
        f"{fact_g0:.4f} -> {softmax64(Lt_g0)[:, zid].mean():.4f} "
        f"(trigger) / {softmax64(Lc_g0)[:, zid].mean():.4f} (control); "
        f"trigger Δmargin mean {eff_trig_g0['dmargin']['mean']:+.4f}, "
        f"Δentropy {eff_trig_g0['dentropy']['mean']:+.5f}.",
        "",
        "## The gate ledger",
        "",
        "| gate | what it binds | pass |",
        "|---|---|---|",
    ]
    for gk, gv in metrics["gates"].items():
        what = {
            "G_NAMEFREE": "the corpus carries no name",
            "G_SPLICE": "the host battery reconstruction (19+41)",
            "G_BATTERYGEO": "the g0/gm12 battery geometry",
            "G_PARENTS": "e311 metrics + vectors md5-bound; the anchor "
                         "literals cross-checked",
            "G_ROOM": "the K10K room D/S bit-bound vs e264_rooms.pt",
            "G_FACTLOAD": "the host loads bit-exact (3-way)",
            "G_TRIGBIND": "the committed trigger (md5/norm/in-room)",
            "G_CTRL": "the random control (in-room ~0, matched norm)",
            "G_ANCHOR": "the +32% anchor (state md5 + read)",
            "G_BATT": "the general battery sanity",
        }.get(gk, "")
        rep.append(f"| {gk} | {what} | {bool(gv['pass'])} |")
    rep += [
        "",
        "## Predictions scored",
        "",
        f"* P-x16a (lab): fired={p_a_fired}.",
        f"* P-x16b (executor counter): fired={p_b_fired} "
        f"(bias-shaped evidence present: {bias_shaped}).",
        "",
        "## Provenance",
        "",
        f"* birth commit: {metrics.get('birth_commit', 'see git')}; "
        f"final head: {git_head()}",
        "* CPU only (torch threads 4, DCT workers 4); no GPU touched.",
        "* parents: e311 (metrics + vectors + host artifact + room "
        "artifact) md5-bound; all gates in metrics.json.",
        "",
        "*No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator "
        "folds).*",
    ]
    (RD / "REPORT.md").write_text("\n".join(rep), encoding="utf-8")
    log("P8 REPORT.md written")

    metrics["outputs"] = {
        "figure": str(fig_path.relative_to(REPO)),
        "metrics": str((RD / "metrics.json").relative_to(REPO)),
        "report": str((RD / "REPORT.md").relative_to(REPO)),
    }
    metrics["git_head_final"] = git_head()
    if SMOKE:
        metrics["status"] = "SMOKE COMPLETE (nothing adjudicated)"
    else:
        metrics["status"] = "COMPLETE — adjudicated"
    metrics["date_completed"] = common.now_iso()
    save_json_partial("P8 figure + report complete")
    log(f"X16 {'SMOKE ' if SMOKE else ''}COMPLETE — {word} / {sword} "
        f"(P-x16a fired={p_a_fired}, P-x16b fired={p_b_fired})")


if __name__ == "__main__":
    main()
