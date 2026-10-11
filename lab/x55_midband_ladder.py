"""X55 — THE MID-BAND LADDER, FATE-RESOLVED (R83's critic control + the R83
ideator's card; e346's rig verbatim; x50's dense grain; the never-sampled
k in (7,16] band). This docstring carries the question + background + design
+ bars + the parity block + P-x55a, plus every frozen operationalization,
committed at birth BEFORE any compute. Adjudicate against exactly this; no
bar shopping.

THE QUESTION (dispatch verbatim): "x55 — THE MID-BAND LADDER, FATE-RESOLVED
(the critic's control + the R83 ideator's card): e346's rig verbatim, the
never-sampled k in (7,16] band at x50's dense grain over the fate steps
(the k7 arm's s100-s135 dense replay included — the critic's named control);
the MEDIAN co-report standing (the R83 convention repair)."

BACKGROUND (the committed record this cell extends; extend, don't repeat):
e346's ladder sampled k in {0,1,2,4,7,16} and adjudicated NEVER-PROTECTS on
the MIN convention (score(k) := min over {s25,s125,s300} of the CPU
instrument host read): 0.000473 -> 0.010666 -> 0.069899 -> 0.091134 ->
0.099412 -> 0.309011 (k=16 ALIVE; every partial rung DEAD, k=7 missing the
DEAD cut by 0.6%, its per-rung fates ALIVE/DEAD/ALIVE — the s125 dip alone
carrying the min). R83's critic: NEVER-PROTECTS is MIN-CONVENTION-CARRIED —
median-scored the ladder crosses DEAD -> MID -> ALIVE (0.070/0.115/0.299),
the registered THRESHOLD family; the owed controls are the MEDIAN CO-REPORT
(the convention repair) + the DENSE k7 IN-BAND REPLAY (s100-s135, the critic:
"registered prediction: the min fluctuates (patch class), at least one of
s105/s110/s115/s120/s130 clears 0.10"). R83's ideator minted THIS cell for
the never-sampled interior k in (7,16): where does the FATE edge live, and
is the partial band's rise PROTECTION or RECOVERY? R83's sharpening stands
as the frame: FIRST-STEP ARMOR IS ALL-OR-NOTHING (s1 dead-class at every
partial k: 0.006-0.014 vs the ruler's 0.749); FATE-STEP RE-FORMATION IS
DOSE-GRADED (k7 re-forms to ALIVE by s25 after dying at s1).

THE DESIGN (dispatch verbatim): "e346's rig verbatim, the never-sampled
k in (7,16] band at x50's dense grain over the fate steps (the k7 arm's
s100-s135 dense replay included — the critic's named control); the MEDIAN
co-report standing (the R83 convention repair)."

BARS (dispatch VERBATIM, frozen in this birth commit BEFORE compute):
  - "EDGE-EARLY (the fate edge lands in (7,12] — k=7's 0.6% near-miss +
    the factor-1.6 read gap)"
  - "EDGE-LATE-OR-NEVER (the edge at k=16 only — full dose or nothing)"
  - "RECOVERY-ONLY (partial dose buys transient alives, not protection —
    the s1 micro-ladder's dose-insensitivity the hint)"

THE PARITY BLOCK (the lab lean AND the counter side by side, both carried
verbatim from the dispatch):
  LAB LEAN: "EDGE-EARLY weakly".
  COUNTER: "RECOVERY-ONLY (the lab's knives have twice been patches)".

P-x55a (THE EXECUTOR'S OWN READ, registered BEFORE compute, frozen HERE at
birth; predictions are scored): **RECOVERY-ONLY** — WITH the dispatch's
counter, AGAINST the lab lean's EDGE-EARLY. SCORED: TRUE iff the verdict ==
RECOVERY-ONLY (plain). GROUNDS: (1) THE MIN CONVENTION READS THE s125 DIP
FLOOR, AND THE DIP FLOOR'S OWN INTERPOLATION LEAVES THE MID-BAND SHORT OF
ALIVE — the binding read at k >= 4 is s125 (k4 0.0911, k7 0.0994, the
ruler's own s125 0.3090 with an 8% margin over the 0.2859 ALIVE cut); even
LINEAR-in-k interpolation of the dip floor (0.0994 + 0.0233(k-7)) lands
k9/k12/k14 at 0.146/0.215/0.262 — all MID; convexity (the likely shape for
a floor that is itself a noisy patch) lands them lower; ALIVE needs
k >= ~15.6 by the linear route, i.e. essentially the full dose. (2) THE
ARMOR OBSERVABLE IS COMMITTED ALL-OR-NOTHING — the s1 micro-ladder is
dead-class at every partial k (0.006-0.014 vs 0.749 at k=16): whatever the
mid-band buys is re-formation AFTER the s1 death, the RECOVERY signature,
not protection; if armor is what an "edge" would mean, the mid-band cannot
supply it short of 16. (3) THE TRANSIENT-ALIVE TEXTURE IS ALREADY COMMITTED
AT k=7 (per-rung ALIVE/DEAD/ALIVE) — monotonicity extends it into the band
(s25/s300 reads ALIVE-class) without any arm clearing min-ALIVE: partial
dose buys transient alives, not protection. (4) THE STANDING COUNTER: the
lab's knives have twice been patches (x50's coherent plateau, x52's dip);
the R83 critic's own registered expectation is the patch class at s125.
EXPECTED SHAPE: k9/k12/k14 s25 reads 0.45-0.75 and s300 reads 0.30-0.75
(ALIVE-class, monotone-ish in k); s125 dips 0.10-0.26 (MID); min-scores MID
(0.10-0.26); MEDIAN scores ALIVE at k >= 9 (the median co-report splits
from the min INSIDE the mid-band — R83's k7 observation extended); s1
reads dead-class (0.004-0.05; the registered surprise branch: k14's s1
> 0.10 would locate an ARMOR edge inside (12,16) — disclosed, never
adjudicating). THE k7 CONTROL: the in-band window s100-s135 shows the patch
class — at least one of s105/s110/s115/s120/s130 clears 0.10 (the critic's
co-registered prediction, adopted verbatim and co-scored) while the band's
floor stays sub-0.10 (the min fluctuates). FALSIFIER: any hard-gate failure
-> TEXTURE (nothing adjudicated; P-x55a UNSCORED per the e345/e346
precedent clause).

==== THE FROZEN OPERATIONALIZATIONS (picked + frozen HERE at birth) ====

* THE RIG := e346's, INHERITED VERBATIM (md5-bound: lab/
  e346_name_key_ladder.py + every parent in its bind table that this cell
  touches): batch 32 = 16 draws from the jittered pool (the 60 committed
  install host occurrences x jitters {-8,-4,0,+4,+8} = 300 windows,
  name-masked) + 8 draws from cons_anchor (the first-16 paired originals,
  e113's convention) + 8 random corpus windows; masked token-level union
  CE; AdamW (0.9, 0.95) wd 0.1; CONST lr 1e-3 (G1.FT_LR); clip 1.0; gen
  seed 10901 (G1.CONS_SEED, HELD — one fresh generator per arm); 300
  steps. The ONLY delta between arms is WHICH pool each of the 16
  name-bearing slots draws from (slot_keys: 'z' = the canon ZEPHYRA pool,
  't' = the same construction keyed TAVIREN).

* THE MID-BAND FLIP SCHEME (pre-registered; THE disclosure): slots are
  positions in the 16-slot pool draw, FIXED every step; the pool CONTENT
  behind each slot is reshuffled by ix every step, so the flipped slots
  sample the whole 300-window TAVIREN pool over the run. MY frozen scheme
  (the round rule, evenly spread; NOT e346's k<=7 picks — e346's own
  scheme was non-nested, disclosed there; mine continues the even-spread
  intent at the band's coarser resolution):
    k=9:  slots (0, 2, 4, 5, 7, 9, 11, 12, 14)          [9 of 16]
    k=12: slots (0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15) [12 of 16]
    k=14: slots (0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15) [14 of 16]
  (each = round(i*16/k) for i in 0..k-1, verified distinct; the k7 replay
  uses e346's OWN committed k=7 slots (0,2,5,8,10,12,15) — inherited, not
  re-chosen.) DOSE ARITHMETIC (disclosed): each step carries 16
  name-bearing windows = 112 name tokens; k=9 -> 63 TAVIREN tokens (56.25%
  of name windows); k=12 -> 84 (75%); k=14 -> 98 (87.5%); the committed
  endpoints: k=7 -> 49 (43.75%); k=16 -> 112 (100%).

* THE ARMS (four CPU replays, one per arm; zero CUDA anywhere):
    k9, k12, k14: the TAVIREN subject (e311_TAVINST_post.pt) through
      e346's draw arithmetic with MIDBAND_FLIPS[k] 't'-slots — first runs,
      x50's DENSE grain (below).
    k7r: THE CRITIC'S CONTROL — the k=7 arm re-realized on CPU with
      e346's own k=7 slot scheme, rungs {1,2,8,25,100,105,...,135,300}
      (the s100-s135 window at grain 5, the critic's named window; 25/125/
      300 for the fate + certification rows). NOT A DUPLICATE: e346's k7
      arm was a GPU realization; this is the CPU re-realization whose
      certification (G_K7CLASS + G_DRAWS-vs-resume) carries the cross-
      class burden for the whole cell.

* THE RUNG SET (frozen; x50's dense grain VERBATIM): {1, 2} U {5, 10,
  ..., 300} U {112, 114, ..., 138} — 74 rungs (G_XGRAIN binds the literal
  list; the fate steps 25/125/300 and s1 are inside it). GRAIN DISCLOSURE:
  5 steps through the leg, 2 steps in [112, 138], 1-2 steps at the very
  start; events finer than the grain are invisible (x50's own disclosure).

* THE ROW PLAN (frozen): FULL dual-channel instrument rows (x45's reader
  verbatim: 3 batteries x {host p(T), guest p(Z)} + alive + corrs + ce_r
  + |W| + in_room vs the K10K room) at {1, 25, 125, 300}; LIGHT dual rows
  (the two g0 channel means + alive fractions ONLY — no g+-12 batteries,
  no corrs, no ce, no census) at every other rung. The fate-level
  adjudication reads ONLY full rows; the dense light rows carry the
  transient-alive texture + x56's ledger (the shared-session disclosure).

* THE FATE CUTS (frozen; committed literals, inherited from e346):
  DEAD iff the read <= 0.10; ALIVE iff the read >= 0.285851389169693
  (e311's committed subject p(T) == x45's ann_s0 floor); MID otherwise.
  THE TWO SCORES (the R83 convention repair, BOTH standing in the table):
  score_min(k) := min over {s25, s125, s300} — e346's frozen convention,
  THE adjudicating score; score_med(k) := median over the same three —
  the R83 MEDIAN CO-REPORT, co-reported at every rung, never re-adjudicat-
  ing e346's committed NEVER-PROTECTS verdict (which stands as committed
  for k <= 7).

* THE SHAPE COMPOSITE (frozen order):
  TEXTURE (any hard-gate failure; nothing adjudicated) ->
  EDGE-EARLY (score_min(k) ALIVE for some k in {9, 12} — the protection
  edge lands in (7,12]) ->
  EDGE-MID-(12,16) (score_min(14) ALIVE alone — an honest shape outside
  the three named, the pre-named fallback; no dispatch bar fires) ->
  RECOVERY-ONLY (no partial arm in {9,12,14} min-score ALIVE AND some
  mid-band fate-step read ALIVE — transient alives without protection) ->
  EDGE-LATE-OR-NEVER (no partial arm min-score ALIVE AND NO mid-band
  fate-step read ALIVE — the mid-band never reaches the ALIVE cut: full
  dose or nothing) ->
  MIXED (anything else, named + disclosed; no bar fires).

* THE k7 IN-BAND CONTROL (the critic's named control; co-report with a
  co-scored prediction): the s100-s135 window's light/full reads; the
  critic's registered prediction ("at least one of s105/s110/s115/s120/
  s130 clears 0.10") co-scored TRUE/FALSE; the IN-BAND-EXTENDED min
  (min over {s25,s125,s300} U the in-band window) co-reported beside the
  frozen fate-trio min with the flip narrative if it clears DEAD (the
  critic's "flips the arm to MID" reading — a co-report lane, never
  touching e346's committed score).

* THE COMMITTED ENDPOINTS (runtime-read, never rerun — extend, don't
  repeat): k=0 = e345's arm (runs/e345/metrics.json rows tav_s*; the
  e346 ladder table carries the assembled values); k=1/2/4/7 = e346's
  arms (runs/e346/metrics.json adjudication.ladder); k=16 = the anneal
  ruler (runs/x45/metrics.json anneal panels; the dense k16 curve =
  x52's committed rider rows). The k-curve therefore spans k in
  {0,1,2,4,7,9,12,14,16} with five committed rungs.

* THE s1 MICRO-LADDER EXTENSION (free co-report, never adjudicating):
  the committed direction datum 0.006756 (k=0) .. 0.014052 (k=7) vs
  0.748952 (k=16, x52's rider_s1) + my k9/k12/k14 s1 full rows — the
  first-step response's dose-insensitivity is THE armor observable.

* NET0 CLASSES (x32's standing rule, recorded per row): k9/k12/k14 =
  "TAVIREN/install-end WARM/MID-BAND-LADDER-FORMING (k-of-16 T-windows,
  k in (7,16), re-forming a TAVIREN-formed substrate)"; k7r = "TAVIREN/
  install-end WARM/LADDER-FORMING k=7 (the committed GPU arm's CPU
  re-realization)".

* COMPUTE ROUTE (chosen + disclosed): CPU DESK ONLY, torch threads 4,
  ZERO CUDA anywhere (the GPU lane is sibling x36's — never touched; a
  runtime assert enforces zero CUDA). Per-arm resume (runs/x55/
  arm_resume_<tag>.pt, *.pt gitignored by the house convention,
  regenerable deterministically from this committed script + the
  committed anchors). The arms are FIRST RUNS (no committed realization
  for k9/k12/k14); the fidelity burden rides G_STREAMIDENT_Z/T (the draw
  path's bit-identity, both keys — inherited verbatim) + G_DRAWS (cross-
  arm bit-equality AND bit-equality to e346's committed k7 resume draw
  record — 300 records loaded from the md5-bound arm_resume_k7.pt) +
  G_K7CLASS (the k7 CPU re-realization vs e346's committed k7 GPU-arm
  rows: |d| <= 2e-3 at s8 — the class where e346's own G_XDEVICE observed
  1e-5 — and |d| <= 0.05 at each of s25/s125/s300, the family's XDEVICE
  class ceiling, 10x the observed band).

* GATES (all hard; any failure -> TEXTURE): G_MD5 (e346's rig + e346/
  e345/e335/e341/x45/x50/x52 metrics + arm_resume_k7.pt + x45's
  anneal_resume + the checkpoint anchors + the room + every port-source
  rig) + G_QUOTES (the 16 inherited port lines) + G_NAMEFREE + G_SPLICE +
  G_BATTERY + G_POOL (z+t pools, each (300,256), mask sums 2100) +
  G_POOLTWIN (pool_t differs from pool_z at EXACTLY the 7 name columns
  per row; masks bit-equal) + G_ROOM (K10K bit-bound) + G_STREAMIDENT_Z +
  G_STREAMIDENT_T (25 CPU steps vs x45's committed walk_cons_s25 /
  anneal_s25 rows, tol 2e-6) + G_ENDPOINTS (subject/base/own_install/
  root reproduce committed literals) + G_DRAWS (four arms' draw records
  bit-equal + bit-equal to e346's k7 resume record + first_ix == e345's
  committed) + G_K7CLASS (the CPU-vs-GPU class, above) + G_XGRAIN (the
  rung set == x50's frozen 74-rung literal) + G_RIGCONST (constants vs
  G1/g1c meta + the mid-band flip scheme recorded).

Outputs: runs/x55/{metrics.json (PROGRESSIVE), REPORT.md,
x55_midband_ladder.png}. No NOTES/THINKING/QUEUE/STATE edits (the
dispatch; the heartbeat folds). SHARED SESSION DISCLOSED: this cell and
x56 (THE SLOT LEDGER) are one desk session; x56 rides this cell's arms
(runtime-read from runs/x55/, md5-bound there). Commit + push per phase
(birth -> run). Run: cd lab && python x55_midband_ladder.py
(X55_SMOKE=1 shakedown).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")       # e228's offline convention
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:                                            # Windows console safety
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                               # noqa: BLE001
    pass

import numpy as np                              # noqa: E402
import scipy.fft as sf                          # noqa: E402
import torch                                    # noqa: E402
import torch.nn.functional as F                  # noqa: E402

import common                                   # noqa: E402
from common import CharCorpus, run_dir, save_json, set_seed  # noqa: E402

import e043_install as E43                      # noqa: E402 (REPO,
                                                 # find_occ, SPLICE_RNG)
import g1b_continuity as GB                     # noqa: E402 — MUST be
                                                 # imported BEFORE G1
import g1_anchored_ball as G1                   # noqa: E402

torch.set_num_threads(4)           # the CPU desk lane; x37/x50's setting

import matplotlib                               # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402

SMOKE = os.environ.get("X55_SMOKE") == "1"
CPU = torch.device("cpu")
# ZERO CUDA (the dispatch's mandate: the GPU lane is sibling x36's) —
# enforced at run end by asserting the CUDA context was never initialized.
NAME = "x55_smoke" if SMOKE else "x55"

T0 = time.time()
RD = run_dir(NAME)
LOG_PATH = RD / "run.log"
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


REPO = common.REPO
CKPT_DIR = GB.CKPT_DIR

# ======================================================================
# THE CONFIG (frozen)
# ======================================================================
CONS_STEPS = 8 if SMOKE else 300          # the family's matched span
CONS_SEED = G1.CONS_SEED                  # 10901, HELD (the canon stream)
FT_LR = G1.FT_LR                          # 1e-3 const (the canon's cons lr)

FATE_STEPS = (25, 125, 300)               # the dispatch's fate reads
FULL_ROW_STEPS = (1, 25, 125, 300)        # full dual instrument rows

# ---- x50's dense grain, VERBATIM (frozen literal; G_XGRAIN binds it) ----
X50_GRAIN = sorted({1, 2}
                   | {s for s in range(5, 301, 5)}
                   | {s for s in range(112, 139, 2)})   # 74 rungs
# the k7 control's rungs (the critic's s100-s135 window at grain 5)
K7R_RUNGS = sorted({1, 2, 8, 25, 300}
                   | {s for s in range(100, 136, 5)})

# ---- the mid-band flip scheme (pre-registered; THE disclosure) ----------
MIDBAND_FLIPS = {
    9: (0, 2, 4, 5, 7, 9, 11, 12, 14),
    12: (0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15),
    14: (0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15),
}
K7_FLIPS = (0, 2, 5, 8, 10, 12, 15)       # e346's own committed k=7 slots
ALL16 = tuple(range(16))

# ---- the fate cuts (frozen; committed literals) ---------------------------
DEAD_CUT = 0.10
ALIVE_CUT_T = 0.285851389169693           # e311's committed subject p(T)
READ_BAR = 0.05                           # x40's aliveness bar (verbatim)
READ_TOL = 2e-6                           # the CPU read-determinism law
K7CLASS_S8_TOL = 2e-3                     # e346's G_XDEVICE observed ~1e-5
K7CLASS_FATE_TOL = 0.05                   # the family's XDEVICE class

# ---- the room (the census rider; e311's conventions) ---------------------
ROOM_K = 10_000
ROOM_SEED_D = 26113
ROOM_SEED_S = 26114

# ======================================================================
# THE ARMS (the only thing that varies: slot_keys; all CPU)
# ======================================================================
def flips_slot_keys(flips: tuple) -> tuple:
    return tuple("t" if s in flips else "z" for s in ALL16)


ARMS = [
    {"tag": "k9", "family": "midband", "k": 9,
     "start": "tav_subject", "slot_keys": flips_slot_keys(MIDBAND_FLIPS[9]),
     "rungs": X50_GRAIN},
    {"tag": "k12", "family": "midband", "k": 12,
     "start": "tav_subject", "slot_keys": flips_slot_keys(MIDBAND_FLIPS[12]),
     "rungs": X50_GRAIN},
    {"tag": "k14", "family": "midband", "k": 14,
     "start": "tav_subject", "slot_keys": flips_slot_keys(MIDBAND_FLIPS[14]),
     "rungs": X50_GRAIN},
    {"tag": "k7r", "family": "k7replay", "k": 7,
     "start": "tav_subject", "slot_keys": flips_slot_keys(K7_FLIPS),
     "rungs": K7R_RUNGS},
]
ARM_BY_TAG = {a["tag"]: a for a in ARMS}

# ---- the md5 binds (frozen at birth; Rule 12) -----------------------------
MD5_BINDS = {
    "rig_e346": ("lab/e346_name_key_ladder.py",
                 "808f89c94347d3d9d194cc268555dc2f"),
    "e346_metrics": ("runs/e346/metrics.json",
                     "7c6dc4051d46ece196486c47b764d4fa"),
    "e346_k7_resume": ("runs/e346/arm_resume_k7.pt",
                       "210dc0bea84d9d3d9788baec959561bc"),
    "x45_metrics": ("runs/x45/metrics.json",
                    "1dfcfeb3840c9307dbbc48cd71859cc5"),
    "x45_anneal_resume": ("runs/x45/anneal_resume_x45.pt",
                          "6b956ad13b9c3e8cfb20a49ce6303cc4"),
    "x50_metrics": ("runs/x50/metrics.json",
                    "b5eb7ea5755dc6a33a567a26411566cb"),
    "x52_metrics": ("runs/x52/metrics.json",
                    "bde14b44664b597520c2ca366e49b789"),
    "e345_metrics": ("runs/e345/metrics.json",
                     "e052fa3e4e0165df3e6c561bec3f67fd"),
    "e335_metrics": ("runs/e335/metrics.json",
                     "5d92359bc9b2e943c01bf521487cf177"),
    "e341_metrics": ("runs/e341/metrics.json",
                     "279bd235e43da13dc78998fe9da7b63d"),
    "rig_e345": ("lab/e345_taviren_warm_walk.py",
                 "4b647ad2303b7aee85503db7dcba9801"),
    "rig_x45": ("lab/x45_path_difference.py",
                "88a22f2d8b2832538c93ba3d49570c23"),
    "rig_x50": ("lab/x50_live_reshape.py",
                "7b76b28b0b4232f87ab3713132a804a8"),
    "rig_x52": ("lab/x52_ruler_symmetry.py",
                "4508ddb8ded0427c7f9249fd3f16566f"),
    "rig_g1": ("lab/g1_anchored_ball.py",
               "f4b6997b6a66013ee25da67f2b4faf01"),
    "rig_g1b": ("lab/g1b_continuity.py",
                "66966621e162b6bd33cf0023b0aa485a"),
    "rig_e281": ("lab/e281_rehearsal_dose.py",
                 "00fbf931f19c42a4b48dbd86806f0d1f"),
    "rig_e043": ("lab/e043_install.py",
                 "f82806b369452b05b8d89ca6cebe70fa"),
    "rig_e311": ("lab/e311_hijacker.py",
                 "60b5389cbc4e6d482c2fa416640a1bf7"),
    "rig_e261": ("lab/e261_rank_ladder.py",
                 "e498031fe4fcd0147c3094c86034b1c5"),
    "rig_e341": ("lab/e341_varied_annealing.py",
                 "18039ab39d1b04d1adef2a22b9b3a350"),
    "rig_x40": ("lab/x40_first_step_mechanism.py",
                "d1f8874a06c43b828eb8de334b20b896"),
    "rig_x44": ("lab/x44_rstar_offsets.py",
                "f67fde47c3a6d60c2f3814395f57a034"),
    "e001": ("runs/checkpoints/e001.pt",
             "d114536d1c0983ab3be67f67ff0667c8"),
    "tav_subject": ("runs/checkpoints/e311_TAVINST_post.pt",
                    "9886b25242c90dd669c6363c39a2a3bf"),
    "own_install": ("runs/checkpoints/g1c_install_resume.pt",
                    "1f5a6e2b327a0731459cbb805fdc2505"),
    "root": ("runs/checkpoints/g1c_root.pt",
             "9c7d4ca1b60c8a1158d080f932e2c95f"),
    "rooms264": ("runs/checkpoints/e264_rooms.pt",
                 "2d524655575cce00a3bc1c8770f4b211"),
}

# the port's exact source lines (inherited VERBATIM from e346's table)
PORT_QUOTES = [
    ("lab/x45_path_difference.py",
     "def battery_percell(net, ids: torch.Tensor, zid: int, bs: int = 30",
     "x45's ported per-context reader (the instrument this cell ports)"),
    ("lab/x45_path_difference.py",
     "def paired_corr(x: list[float], y: list[float]):",
     "x45's paired-context Pearson form"),
    ("lab/x44_rstar_offsets.py",
     "for b1, b2 in ((\"g-12\", \"g0\"), (\"g0\", \"g+12\"), (\"g-12\", \"g+12\")):",
     "x44's elicitation pair set (the instrument's parent)"),
    ("lab/x44_rstar_offsets.py",
     "corr[a][f\"{b1}|{b2}\"] = cov / (vx * vy) if vx > 0 and vy > 0",
     "x44's Pearson form"),
    ("lab/x40_first_step_mechanism.py",
     "alive_frac\": float((pv >= READ_BAR).float().mean())",
     "x40's alive-fraction line"),
    ("lab/g1_anchored_ball.py",
     "ix = torch.randint(n_pool, (16,), generator=gen)",
     "G1's cons pool draw (e113's own line)"),
    ("lab/g1_anchored_ball.py",
     "gen = torch.Generator().manual_seed(CONS_SEED)",
     "G1's cons generator (seed 10901)"),
    ("lab/g1_anchored_ball.py",
     "m[:16] = pool_mask[ix].to(dev)",
     "G1's cons name mask (e113's own line)"),
    ("lab/g1_anchored_ball.py",
     "loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())",
     "G1's masked union CE (the cons loss form)"),
    ("lab/e281_rehearsal_dose.py",
     "cons_anchor = anchor_full[:16]",
     "e281's cons anchor bank (the canon stream's anchor parent)"),
    ("lab/e281_rehearsal_dose.py",
     "m[:, G1.PRE - 1 + j: G1.PRE - 1 + j + len(G1.NAME)] = True",
     "e281's ZEPHYRA jittered pool mask (the canon stream itself)"),
    ("lab/e341_varied_annealing.py",
     "gen = torch.Generator().manual_seed(CONS_SEED)",
     "e341's VARIED arm generator — THE MENU-ALIASING FACT (the T-keyed "
     "twin this cell's mid-band and k7 replay draw from)"),
    ("lab/e311_hijacker.py",
     "return float(np.linalg.norm(room.project(v64)) / vn)",
     "e311's in-room fraction (the census rider's convention)"),
    ("lab/e261_rank_ladder.py",
     "P x = D . idct(mask_S(dct(D . x)))",
     "the room's exact projector (the census ruler)"),
    ("lab/e345_taviren_warm_walk.py",
     "nw = p0[\"pool_z_x\"][ix]",
     "e345's canon draw line (the rig this cell inherits)"),
    ("lab/e345_taviren_warm_walk.py",
     "bt = G1.battery_cell(evl, p0[\"g0_ids\"].to(dev), p0[\"tid\"])",
     "e345's per-step dual-channel read (the loop this cell's arms run)"),
]

REGISTERED = {
    "question_verbatim": (
        "x55 — THE MID-BAND LADDER, FATE-RESOLVED (the critic's control + "
        "the R83 ideator's card): e346's rig verbatim, the never-sampled k "
        "in (7,16] band at x50's dense grain over the fate steps (the k7 "
        "arm's s100-s135 dense replay included — the critic's named "
        "control); the MEDIAN co-report standing (the R83 convention "
        "repair)."),
    "bars_verbatim": {
        "edge_early": ("EDGE-EARLY (the fate edge lands in (7,12] — k=7's "
                       "0.6% near-miss + the factor-1.6 read gap)"),
        "edge_late_or_never": ("EDGE-LATE-OR-NEVER (the edge at k=16 only "
                               "— full dose or nothing)"),
        "recovery_only": ("RECOVERY-ONLY (partial dose buys transient "
                          "alives, not protection — the s1 micro-ladder's "
                          "dose-insensitivity the hint)"),
    },
    "parity_verbatim": (
        "LAB LEAN: 'EDGE-EARLY weakly'; COUNTER: 'RECOVERY-ONLY (the lab's "
        "knives have twice been patches)'."),
    "P_x55a": {
        "guess": "RECOVERY-ONLY (WITH the dispatch's counter, AGAINST the "
                 "lab lean's EDGE-EARLY)",
        "grounds": (
            "(1) the min convention reads the s125 dip floor, and the dip "
            "floor's own linear-in-k interpolation (0.0994 + 0.0233(k-7); "
            "the ruler's own s125 0.3090 clears ALIVE by only 8%) lands "
            "k9/k12/k14 at 0.146/0.215/0.262 — all MID; convexity lands "
            "them lower; (2) the armor observable is committed "
            "all-or-nothing (s1 dead-class at every partial k, 0.006-0.014 "
            "vs 0.749): the mid-band's rise is re-formation after the s1 "
            "death — recovery, not protection; (3) k=7's transient-alive "
            "texture (per-rung ALIVE/DEAD/ALIVE) extends monotonically "
            "without any arm clearing min-ALIVE; (4) the standing counter: "
            "the lab's knives have twice been patches (x50, x52); the R83 "
            "critic's registered expectation is the patch class."),
        "expected_shape": (
            "k9/k12/k14 s25 0.45-0.75, s300 0.30-0.75 (ALIVE-class); s125 "
            "dips 0.10-0.26 (MID); min-scores MID; MEDIAN scores ALIVE at "
            "k >= 9 (the median co-report splits from the min inside the "
            "mid-band); s1 dead-class 0.004-0.05 (surprise branch: k14 s1 "
            "> 0.10 = an ARMOR edge inside (12,16), disclosed only). The "
            "k7 control: the patch class — at least one of s105/s110/s115/"
            "s120/s130 clears 0.10 while the band floor stays sub-0.10."),
        "falsifier": ("any hard-gate failure -> TEXTURE (nothing "
                      "adjudicated; P-x55a UNSCORED)"),
        "scored": ("TRUE iff the verdict == RECOVERY-ONLY (plain)"),
    },
    "critic_coregistered": (
        "the R83 critic's k7-control prediction, adopted verbatim and "
        "co-scored: 'the min fluctuates (patch class), at least one of "
        "s105/s110/s115/s120/s130 clears 0.10'"),
    "registration": ("question + bars + the parity block + P-x55a + the "
                     "critic's co-registered prediction + every convention "
                     "picked + frozen HERE at birth BEFORE compute; this "
                     "script committed at birth; adjudicate against "
                     "exactly this; no bar shopping."),
}

deviations: list[str] = [
    "THE MID-BAND UNIT (disclosed; inherited from e346): k counts the 16 "
    "name-bearing pool SLOTS per step whose window is keyed TAVIREN (each "
    "flipped slot re-keys its full 7-token name); dose arithmetic frozen: "
    "k x 7 TAVIREN name tokens per step of 112.",
    "THE MID-BAND FLIP SCHEME (disclosed at birth): the round rule "
    "round(i*16/k), i=0..k-1 — evenly spread, MY frozen picks; NOT nested "
    "with e346's k<=7 sets (e346's own scheme was non-nested, disclosed "
    "there); the k7 replay uses e346's OWN committed k=7 slots, inherited.",
    "THE CPU ROUTE (disclosed; the dispatch's CPU-desk mandate): all four "
    "arms are CPU realizations at threads 4, ZERO CUDA (a runtime assert "
    "enforces it; the GPU lane is sibling x36's). The committed k<=7 "
    "endpoints were GPU realizations (e345/e346) read on the CPU "
    "instrument; the cross-class burden rides G_K7CLASS (the k7 CPU "
    "re-realization vs e346's committed k7 GPU rows: s8 |d| <= 2e-3, fate "
    "|d| <= 0.05 — the family's XDEVICE class) + the inherited "
    "G_STREAMIDENT probes (CPU-class bit-identity, both keys).",
    "THE ROW PLAN (disclosed): FULL dual rows at {1,25,125,300}; LIGHT "
    "dual rows (the two g0 means + alive only) at x50's other rungs — the "
    "fate-level adjudication reads only full rows; the dense light rows "
    "carry the transient-alive texture + x56's ledger (the shared-session "
    "disclosure).",
    "THE MEDIAN CO-REPORT (the R83 convention repair, standing): the "
    "k-table carries score_min (e346's frozen adjudicating convention) "
    "AND score_med (the median over the same three fate reads) at every "
    "rung; the median column NEVER re-adjudicates e346's committed "
    "NEVER-PROTECTS verdict for k <= 7 (which stands as committed).",
    "THE COMMITTED ENDPOINTS are runtime-read, never rerun (extend, don't "
    "repeat): k=0 from e345, k=1/2/4/7 from e346, k=16 from x45's panels "
    "+ x52's dense rider rows.",
    "Per-arm resumes under runs/x55/arm_resume_<tag>.pt (*.pt gitignored "
    "by the house convention — regenerable deterministically); this cell "
    "writes ONLY lab/x55_* and runs/x55/*.",
    "SHARED SESSION (disclosed): x56 (THE SLOT LEDGER) rides this cell's "
    "arms in the same desk session — runtime-read from runs/x55/, md5-"
    "bound there at x56's birth.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the heartbeat folds).",
    "Smoke mode (X55_SMOKE=1): 8-step arms, rungs {1,2,4,8}, every gate "
    "live (G_STREAMIDENT at the full 25 CPU steps; G_K7CLASS at s8 vs the "
    "committed k7_s8 row); the adjudication SMOKE-stamped (nothing "
    "scored) — every code path exercised.",
    "THE SMOKE-CAUGHT PNG REPAIR (pre-adjudication, disclosed): the "
    "first smoke pass crashed in make_png panel 3 AFTER P4 — a str step "
    "key compared against an int window bound (e346's own post-P4 PNG "
    "crash class, the family's named repeat). Repaired to int(s). No "
    "bar, arm, stream, gate criterion or registration byte touched — "
    "presentation only; every gate had already PASSed in the crashed "
    "pass (15/15).",
]

metrics: dict = {
    "experiment": "x55_midband_ladder",
    "phase": ("THE MID-BAND LADDER, FATE-RESOLVED — k in {9,12,14} of 16 "
              "TAVIREN-keyed windows (the never-sampled (7,16] band) on "
              "e346's rig at x50's dense grain + the k7 s100-s135 dense "
              "replay (the critic's control) + the MEDIAN co-report "
              "standing (the R83 convention repair) — EDGE-EARLY / "
              "EDGE-LATE-OR-NEVER / RECOVERY-ONLY"),
    "date": common.now_iso(),
    "status": "PARTIAL: startup",
    "registration": REGISTERED["registration"],
    "registered": REGISTERED,
    "smoke": SMOKE,
    "envelope": {
        "device": ("CPU desk ONLY (torch threads 4; zero CUDA — the GPU "
                   "lane is sibling x36's, never touched; runtime "
                   "asserted)"),
        "threads": 4,
        "timestamps": "datetime.now(UTC) only",
    },
    "deviations": deviations,
    "builds_on": [
        "e346 (THE RIG VERBATIM + the committed k-curve k in {0,1,2,4,7,"
        "16} + the s1 micro-ladder + every operational convention this "
        "cell extends into the never-sampled mid-band)",
        "R83 (THE CRITIC'S CONTROLS — the dense k7 replay + the median "
        "co-report — and the IDEATOR'S CARD that minted this cell; the "
        "sharpened frame: first-step armor all-or-nothing, fate-step "
        "re-formation dose-graded)",
        "x50 (THE DENSE GRAIN VERBATIM — the 74-rung set {1,2} U {5,"
        "10,...,300} U {112,114,...,138}; and the method lesson that "
        "committed-grain 'coherent rungs' can be single-rung spikes "
        "inside decorrelating interiors)",
        "x52 (THE DENSE RULER REPLAY — the committed k16 dense curve this "
        "cell plots beside its arms; the certified CPU replay route)",
        "x45 (THE INSTRUMENT VERBATIM — the per-context elicitation "
        "reader, the anneal panels = the k=16 endpoint, the walk rungs)",
        "e345 (THE k=0 ENDPOINT + the canon draw line the arms inherit)",
        "e341 (THE T-KEYED TWIN'S PARENT — the 't' pool's construction)",
        "e311 + e264 (the TAVIREN subject + the census room conventions)",
    ],
    "whats_new": [
        "THE MID-BAND ITSELF: the k-ladder's never-sampled interior k in "
        "{9,12,14} — the (7,16] band's first rungs, every arm sharing the "
        "committed draw sequence bit-exactly, fate-resolved at x50's dense "
        "grain",
        "THE k7 IN-BAND CONTROL (the critic's named control, run): the "
        "s100-s135 window at grain 5 on a CPU re-realization of e346's "
        "k=7 arm — the s125 dip's patch-class test, the critic's "
        "prediction co-scored",
        "THE MEDIAN CO-REPORT STANDING (the R83 convention repair): both "
        "scores at every rung of the extended nine-point k-curve",
        "THE s1 MICRO-LADDER EXTENSION: the armor observable's dose-"
        "response inside the band (the all-or-nothing frame's untested "
        "interior)",
    ],
    "gates": {},
}


# ------------------------------------------------------------------ helpers
def md5of(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:                                       # noqa: BLE001
        return "unavailable"


def write_partial(note: str) -> None:
    metrics["status"] = f"PARTIAL: {note} ({common.now_iso()})"
    save_json(RD / "metrics.json", metrics)


def flat_params_cpu(net) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).cpu()
                      for p in net.parameters()]).clone()


def load_model_sd(path: Path) -> dict:
    """The family's checkpoint-loader convention (load_g1's own form)."""
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    return {k: v.detach().clone() for k, v in sd.items()}


# ---- x45's per-context reader (ported VERBATIM) ---------------------------
@torch.no_grad()
def battery_percell(net, ids: torch.Tensor, zid: int, bs: int = 30
                    ) -> torch.Tensor:
    """x45's exact instrument returning the FULL per-context vector."""
    net.eval()
    pzs = []
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs.append(pr[:, zid])
    return torch.cat(pzs)


# ---- x45's paired elicitation correlations (ported VERBATIM) --------------
def paired_corr(x: list[float], y: list[float]):
    mx, my = sum(x) / 60, sum(y) / 60
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(60))
    vx = math.sqrt(sum((v - mx) ** 2 for v in x))
    vy = math.sqrt(sum((v - my) ** 2 for v in y))
    return cov / (vx * vy) if vx > 0 and vy > 0 else None


# ---- e261's SRCT room (the census rider; rebuilt + bit-bound) -------------
class SRCT:
    """e261's room VERBATIM (the projector; the basis never materialized):
    P x = D . idct(mask_S(dct(D . x))), fp64 pocketfft."""

    def __init__(self, n: int, k: int, seed_d: int, seed_s: int):
        assert 0 < k <= n
        self.n, self.k = int(n), int(k)
        g = torch.Generator().manual_seed(seed_d)
        self.D = torch.where(torch.rand(n, generator=g) < 0.5,
                             -1.0, 1.0).numpy().astype(np.float64)
        g2 = torch.Generator().manual_seed(seed_s)
        self.S = torch.randperm(n, generator=g2)[:self.k].sort().values \
            .numpy()
        self.mask = np.zeros(n, dtype=np.float64)
        self.mask[self.S] = 1.0
        self.seed_d, self.seed_s = seed_d, seed_s

    def project(self, x64: np.ndarray) -> np.ndarray:
        c = sf.dct(self.D * x64, type=2, norm="ortho", workers=4)
        c *= self.mask
        return self.D * sf.idct(c, type=2, norm="ortho", workers=4)


# ======================================================================
# P0 — THE BINDS + THE BANK + THE TWO POOLS + THE ROOM + RECORDS
# ======================================================================
def phase_P0() -> dict:
    log("P0 — THE BINDS + the bank + the two keyed pools + the room + "
        "the committed records")

    # ---- G_MD5 --------------------------------------------------------
    binds = {}
    for key, (rel, bound) in MD5_BINDS.items():
        p = REPO / rel
        ok = p.exists() and md5of(p) == bound
        binds[key] = {"path": rel,
                      "md5": (md5of(p) if p.exists() else None),
                      "bound_md5": bound, "pass": bool(ok)}
    g_md5 = {"binds": binds,
             "claim": ("every committed record, anchor state, instrument "
                       "parent, port-source rig, stream parent, THE e346 "
                       "RIG ITSELF and e346's k7 ARM RESUME md5-bound at "
                       "run time (frozen at birth) — the inherited rig "
                       "verbatim"),
             "pass": bool(all(b["pass"] for b in binds.values()))}
    assert g_md5["pass"], \
        f"G_MD5 FAILED: {[k for k, b in binds.items() if not b['pass']]}"
    metrics["gates"]["G_MD5"] = g_md5
    log(f"G_MD5 PASS: {len(binds)} binds")

    # ---- G_QUOTES -----------------------------------------------------
    quotes = []
    for src, frag, why in PORT_QUOTES:
        txt = (REPO / src).read_text(encoding="utf-8")
        quotes.append({"source": src, "fragment": frag, "why": why,
                       "verified_substring": bool(frag in txt)})
    g_q = {"quotes": quotes,
           "claim": ("every ported line is a verbatim substring of its "
                     "committed source rig (the 16 inherited port lines)"),
           "pass": bool(all(q["verified_substring"] for q in quotes))}
    assert g_q["pass"], f"G_QUOTES FAILED: {g_q}"
    metrics["gates"]["G_QUOTES"] = g_q
    log(f"G_QUOTES PASS: {len(quotes)}/{len(quotes)} verbatim")

    # ---- the corpus + the bank ----------------------------------------
    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]                             # the ZEPHYRA channel
    tid = stoi["T"]                             # the TAVIREN channel
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)

    name_ids_z = corpus.encode(G1.NAME)         # ZEPHYRA (7)
    name_ids_t = corpus.encode("TAVIREN")       # TAVIREN (7)
    assert len(name_ids_z) == 7 and len(name_ids_t) == 7

    g_namefree = {"zeph": train_text.count("ZEPH"),
                  "tav": train_text.count("TAVI"),
                  "pass": bool(train_text.count("ZEPH") == 0
                               and train_text.count("TAVI") == 0)}
    assert g_namefree["pass"], f"G_NAMEFREE FAILED: {g_namefree}"
    metrics["gates"]["G_NAMEFREE"] = g_namefree

    import random
    host_occ = []
    for host in G1.HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + G1.POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ = host_occ[:60]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    g_splice = {"install_mix": mix,
                "pass": bool(mix == {"FLORIZEL": 19, "ELIZABETH": 41})}
    assert g_splice["pass"], f"G_SPLICE FAILED: {g_splice}"

    bat_ids = {}
    for j in G1.GEOS:
        cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    g_battery = {"shapes": {f"g{j:+d}": list(bat_ids[j].shape)
                            for j in G1.GEOS},
                 "pass": bool(list(bat_ids[-12].shape) == [60, G1.PRE - 12]
                              and list(bat_ids[0].shape) == [60, G1.PRE]
                              and list(bat_ids[12].shape)
                              == [60, G1.PRE + 12])}
    assert g_battery["pass"], f"G_BATTERY FAILED: {g_battery}"
    metrics["gates"].update({"G_SPLICE": g_splice, "G_BATTERY": g_battery})
    log("P0: the bank rebuilt (namefree Z+T; splice 19+41; battery shapes)")

    anchor_full = torch.stack([train_ids[p - G1.PRE: p - G1.PRE + G1.BLOCK]
                               for p, _ in install_occ])   # (60, 256)
    r_eval_x, r_eval_y = G1.val_windows(val_ids, val_text_of(itos, val_ids),
                                        60, G1.R_EVAL_SEED)

    # ---- THE TWO KEYED POOLS (z = the canon; t = the ruler's key) -------
    def jittered_pool(name_ids):
        jit_x, jit_mask = {}, {}
        for j in G1.JITTERS:
            jwins = []
            for p, h in install_occ:
                pre = train_ids[p - G1.PRE - j: p]
                post = train_ids[p + len(h): p + len(h) + G1.POST_CAP - j]
                w = torch.cat([pre, name_ids, post])
                if len(w) != G1.BLOCK:
                    raise RuntimeError(f"jit window len {len(w)} at j={j}")
                jwins.append(w)
            jit_x[j] = torch.stack(jwins)
            m = torch.zeros(len(jwins), G1.BLOCK - 1, dtype=torch.bool)
            m[:, G1.PRE - 1 + j: G1.PRE - 1 + j + len(name_ids)] = True
            jit_mask[j] = m
        return torch.cat([jit_x[j] for j in G1.JITTERS]), \
            torch.cat([jit_mask[j] for j in G1.JITTERS])

    pool_z_x, pool_z_mask = jittered_pool(name_ids_z)   # THE CANON STREAM
    pool_t_x, pool_t_mask = jittered_pool(name_ids_t)   # the ruler's key
    cons_anchor = anchor_full[:16]       # e113: first-16 originals
    g_pool = {
        "pool_shape": list(pool_z_x.shape),
        "pool_mask_sum": int(pool_z_mask.sum()),
        "anchor_shape": list(cons_anchor.shape),
        "jitters": list(G1.JITTERS),
        "tav_pool": {"shape": list(pool_t_x.shape),
                     "mask_sum": int(pool_t_mask.sum())},
        "pass": bool(list(pool_z_x.shape) == [300, G1.BLOCK]
                     and int(pool_z_mask.sum()) == 300 * 7
                     and list(pool_t_x.shape) == [300, G1.BLOCK]
                     and int(pool_t_mask.sum()) == 300 * 7
                     and list(cons_anchor.shape) == [16, G1.BLOCK]),
    }
    assert g_pool["pass"], f"G_POOL FAILED: {g_pool}"
    metrics["gates"]["G_POOL"] = g_pool
    log("G_POOL PASS: the two keyed pools rebuilt (z/t, 300 jittered "
        "windows each + the first-16 anchor bank)")

    # ---- G_POOLTWIN (the ONLY-the-key-varies proof) --------------------
    def twin_check(pa_x, pb_x):
        rows_ok, cols_ok = 0, True
        for i in range(pa_x.shape[0]):
            j = G1.JITTERS[i // 60]
            d = (pa_x[i] != pb_x[i]).nonzero().flatten().tolist()
            if d == list(range(G1.PRE + j, G1.PRE + j + 7)):
                rows_ok += 1
            else:
                cols_ok = False
        return rows_ok, cols_ok

    rt, ct = twin_check(pool_z_x, pool_t_x)
    masks_eq = bool(torch.equal(pool_z_mask, pool_t_mask))
    g_twin = {
        "t_vs_z": {"rows_exactly_7_name_cols": rt, "of": 300,
                   "cols_exactly_expected": ct},
        "masks_pairwise_bit_equal": masks_eq,
        "claim": ("pool_t differs from pool_z at EXACTLY the 7 name "
                  "columns of each window (jitter-block-located), masks "
                  "bit-equal — the arms differ in NOTHING but the key"),
        "pass": bool(rt == 300 and ct and masks_eq),
    }
    assert g_twin["pass"], f"G_POOLTWIN FAILED: {g_twin}"
    metrics["gates"]["G_POOLTWIN"] = g_twin
    log("G_POOLTWIN PASS: t pool differs from z at exactly the 7 name "
        "columns x300; masks bit-equal")

    # ---- the room (rebuilt + bit-bound + idempotence probe) -----------
    room = SRCT(GB.G1B_PARAMS, ROOM_K, ROOM_SEED_D, ROOM_SEED_S)
    rooms264 = torch.load(CKPT_DIR / "e264_rooms.pt", map_location="cpu",
                          weights_only=False)
    D264 = np.asarray(rooms264["model"]["K10K"]["D_int8"]).astype(np.float64)
    S264 = np.asarray(rooms264["model"]["K10K"]["S"])
    x = np.random.default_rng(26115).standard_normal(GB.G1B_PARAMS)
    px = room.project(x)
    idem = float(np.linalg.norm(room.project(px) - px)
                 / np.linalg.norm(px))
    g_room = {"k": room.k, "seeds": [room.seed_d, room.seed_s],
              "D_bit_equal": bool(np.array_equal(room.D, D264)),
              "S_bit_equal": bool(np.array_equal(room.S, S264)),
              "idempotence_rel": idem,
              "claim": "the census rider's ruler == the committed K10K "
                       "room (e264 bit-bound)",
              "pass": bool(np.array_equal(room.D, D264)
                           and np.array_equal(room.S, S264)
                           and idem < 1e-10)}
    assert g_room["pass"], f"G_ROOM FAILED: {g_room}"
    metrics["gates"]["G_ROOM"] = g_room
    log(f"G_ROOM PASS: K10K room bit-bound to e264 (idempotence "
        f"{idem:.1e})")
    del rooms264, D264, S264, x, px

    # ---- the committed records (runtime-read; never retyped) ----------
    e346m = json.loads((REPO / "runs" / "e346" / "metrics.json")
                       .read_text(encoding="utf-8"))
    committed_e346 = {
        "ladder": e346m["adjudication"]["ladder"],
        "arm_rows": e346m["arm_rows"],
        "first_ix": e346m["gates"]["G_DRAWS"]["first_ix"],
    }
    e345m = json.loads((REPO / "runs" / "e345" / "metrics.json")
                       .read_text(encoding="utf-8"))
    committed_e345 = {
        "arm_rows": e345m["arm"]["rows"],
        "traj": e345m["arm"]["traj"],
        "first_ix": e345m["gates"]["G_DRAWS"]["first_ix"],
    }
    x45m = json.loads((REPO / "runs" / "x45" / "metrics.json")
                      .read_text(encoding="utf-8"))
    committed_x45 = {
        "anneal_panels": x45m["anneal"]["panels"],
        "walk_rungs": x45m["walk"]["rungs"],
    }
    x52m = json.loads((REPO / "runs" / "x52" / "metrics.json")
                      .read_text(encoding="utf-8"))
    committed_x52 = {"rider_rows": x52m["rider_replay"]["rows"]}
    e335m = json.loads((REPO / "runs" / "e335" / "metrics.json")
                       .read_text(encoding="utf-8"))
    ws = {s["key"]: s for s in e335m["cut1_walk"]["states"]}
    committed_e335 = {
        "base_read": ws["base"]["read"],
        "own_install_read": ws["own_install"]["read"],
        "root_read": ws["root"]["read"],
    }
    # e346's committed k7 draw record (the G_DRAWS reference)
    k7_resume = torch.load(REPO / "runs" / "e346" / "arm_resume_k7.pt",
                           map_location="cpu", weights_only=False)
    committed_k7_draws = k7_resume["draws"]
    committed_k7_traj = k7_resume["traj"]
    del k7_resume
    log("P0: the committed records runtime-read (e346 ladder + k7 resume "
        f"draws {len(committed_k7_draws)}; e345 rows + traj; x45 panels; "
        f"x52 rider rows {len(committed_x52['rider_rows'])})")
    write_partial("P0 the binds + the bank + the two pools + the room + "
                  "records")
    return {
        "corpus": corpus, "stoi": stoi, "itos": itos, "zid": zid,
        "tid": tid, "train_ids": train_ids,
        "bat_ids": bat_ids, "g0_ids": bat_ids[0], "gm12_ids": bat_ids[-12],
        "gp12_ids": bat_ids[12], "anchor_full": anchor_full,
        "r_eval_xy": (r_eval_x, r_eval_y), "room": room,
        "pool_z_x": pool_z_x, "pool_z_mask": pool_z_mask,
        "pool_t_x": pool_t_x, "pool_t_mask": pool_t_mask,
        "cons_anchor": cons_anchor,
        "committed_e346": committed_e346,
        "committed_e345": committed_e345,
        "committed_x45": committed_x45,
        "committed_x52": committed_x52,
        "committed_e335": committed_e335,
        "committed_k7_draws": committed_k7_draws,
        "committed_k7_traj": committed_k7_traj,
    }


def val_text_of(itos, val_ids) -> str:
    return "".join(itos[int(i)] for i in val_ids)


# ======================================================================
# THE INSTRUMENT (x45's row form; full + light)
# ======================================================================
BATTERIES = (("g-12", "gm12_ids"), ("g0", "g0_ids"), ("g+12", "gp12_ids"))


def channel_summary(net, p0: dict, name_id: int) -> dict:
    """means/alive/corrs on ONE channel."""
    percell, alive, means = {}, {}, {}
    for bn, bk in BATTERIES:
        pv = battery_percell(net, p0[bk], name_id)
        percell[bn] = [float(v) for v in pv]
        alive[bn] = float((pv >= READ_BAR).float().mean())
        means[bn] = float(pv.mean())
    return {"means": means, "alive": alive,
            "corrs": {
                f"{b1}|{b2}": paired_corr(percell[b1], percell[b2])
                for b1, b2 in (("g-12", "g0"), ("g0", "g+12"),
                               ("g-12", "g+12"))}}


def instrument(p0: dict, theta_sd: dict, tag: str, name_id: int,
               base_flat64: np.ndarray, dual_id: int | None = None
               ) -> dict:
    """x45's observables on one state (host channel + the dual guest)."""
    net = G1.evl_load(theta_sd)
    net.eval()
    row: dict = {"tag": tag}
    percell = {}
    alive, means = {}, {}
    for bn, bk in BATTERIES:
        pv = battery_percell(net, p0[bk], name_id)
        percell[bn] = [float(v) for v in pv]
        alive[bn] = float((pv >= READ_BAR).float().mean())
        means[bn] = float(pv.mean())
        ref = G1.battery_cell(net, p0[bk], name_id)["mean_pz"]
        assert abs(means[bn] - ref) <= 1e-9, \
            f"percell reader drift {bn} {tag}: {means[bn]} vs {ref}"
    row["read"] = means["g0"]
    row["means"] = means
    row["alive"] = alive
    row["corrs"] = {
        f"{b1}|{b2}": paired_corr(percell[b1], percell[b2])
        for b1, b2 in (("g-12", "g0"), ("g0", "g+12"), ("g-12", "g+12"))}
    row["ce_r"] = G1.ce_fixed_cpu(net, *p0["r_eval_xy"])
    flat64 = flat_params_cpu(net).double().numpy()
    d = flat64 - base_flat64
    dn = float(np.linalg.norm(d))
    row["write_norm"] = dn
    row["in_room"] = (float(np.linalg.norm(p0["room"].project(d)) / dn)
                      if dn > 0 else None)
    if dual_id is not None:
        row["dual"] = channel_summary(net, p0, dual_id)
    del net
    return row


def light_row(p0: dict, theta_sd: dict, tag: str, host_id: int,
              guest_id: int) -> dict:
    """the two g0 channel means + alive fractions ONLY (the dense grain)."""
    net = G1.evl_load(theta_sd)
    net.eval()
    row: dict = {"tag": tag, "row_class": "light"}
    for nm, ch in (("host", host_id), ("guest", guest_id)):
        pv = battery_percell(net, p0["g0_ids"], ch)
        row[nm] = float(pv.mean())
        row[f"alive_{nm}"] = float((pv >= READ_BAR).float().mean())
    del net
    return row


# ======================================================================
# THE CONSENSUS TRAINING STEP (one arithmetic; slot_keys = the ONLY delta)
# ======================================================================
def make_draw(p0: dict, slot_keys: tuple):
    """The canon cons stream's per-step draw, generalized ONLY in which
    keyed pool each of the 16 name-bearing slots reads (e345's
    'nw = pool_z_x[ix]' line, per-slot). The generator consumption is
    IDENTICAL for every keying."""
    n_pool = p0["pool_z_x"].shape[0]
    cons_anchor = p0["cons_anchor"]
    train_ids = p0["train_ids"]
    pools_x = {"z": p0["pool_z_x"], "t": p0["pool_t_x"]}
    pools_m = {"z": p0["pool_z_mask"], "t": p0["pool_t_mask"]}

    def draw(gen, dev):
        ix = torch.randint(n_pool, (16,), generator=gen)
        aj = torch.randint(cons_anchor.shape[0], (8,), generator=gen)
        rj = torch.randint(len(train_ids) - G1.BLOCK - 1, (8,),
                           generator=gen)
        nw = torch.stack([pools_x[slot_keys[s]][ix[s]] for s in range(16)])
        nm = torch.stack([pools_m[slot_keys[s]][ix[s]] for s in range(16)])
        anc = torch.cat([cons_anchor[aj],
                         torch.stack([train_ids[s: s + G1.BLOCK]
                                      for s in rj])], 0)
        x = torch.cat([nw[:, :-1], anc[:, :-1]], 0).to(dev)
        y = torch.cat([nw[:, 1:], anc[:, 1:]], 0).to(dev)
        m = torch.zeros(32, x.shape[1], dtype=torch.bool, device=dev)
        m[:16] = nm.to(dev)
        return x, y, m, (ix.tolist(), aj.tolist(), rj.tolist())
    return draw


def train_step(net, opt, x, y, m):
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
    return float(loss.item())


# ======================================================================
# P1a — G_STREAMIDENT_Z + G_STREAMIDENT_T (inherited verbatim)
# ======================================================================
def compare_row(mine: dict, ref: dict) -> tuple[dict, bool]:
    """x45's row-comparison form: 3 means tol 2e-6, 3 alive exact,
    3 corrs tol 1e-6."""
    cell, ok = {}, True
    for bn in ("g-12", "g0", "g+12"):
        cell[f"mean:{bn}"] = {
            "mine": mine["means"][bn], "committed": ref["means"][bn],
            "abs_diff": abs(mine["means"][bn] - ref["means"][bn])}
        ok &= cell[f"mean:{bn}"]["abs_diff"] <= READ_TOL
        cell[f"alive:{bn}"] = {
            "mine": mine["alive"][bn], "committed": ref["alive"][bn],
            "abs_diff": abs(mine["alive"][bn] - ref["alive"][bn])}
        ok &= cell[f"alive:{bn}"]["abs_diff"] == 0.0
    for pair in ("g-12|g0", "g0|g+12", "g-12|g+12"):
        cell[f"corr:{pair}"] = {
            "mine": mine["corrs"][pair], "committed": ref["corrs"][pair],
            "abs_diff": abs(mine["corrs"][pair] - ref["corrs"][pair])}
        ok &= cell[f"corr:{pair}"]["abs_diff"] <= 1e-6
    cell["pass"] = bool(ok)
    return cell, bool(ok)


def probe_streamident(p0: dict, slot_keys: tuple, start_path: Path,
                      steps: int, host_id: int,
                      ref_row: dict, ref_name: str) -> tuple[dict, bool]:
    net = G1.evl_load(load_model_sd(start_path)).to(CPU)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(CONS_SEED)
    draw = make_draw(p0, slot_keys)
    for _ in range(steps):
        x, y, m, _ = draw(gen, CPU)
        train_step(net, opt, x, y, m)
    sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    del net, opt
    e001_flat64 = flat_params_cpu(
        G1.evl_load(load_model_sd(CKPT_DIR / "e001.pt"))
    ).double().numpy()
    my_row = instrument(p0, sd, f"probe_{ref_name}", host_id, e001_flat64)
    cell, ok = compare_row(my_row, ref_row)
    return cell, ok


def phase_P1a(p0: dict) -> None:
    probe_steps = 25
    log(f"P1a — G_STREAMIDENT x2 (inherited): all-z ({probe_steps} CPU "
        "steps from own_install -> x45's walk_cons_s25) and all-t (from "
        "the subject -> x45's anneal_s25)")

    cellz, okz = probe_streamident(
        p0, tuple("z" for _ in ALL16), CKPT_DIR / "g1c_install_resume.pt",
        probe_steps, p0["zid"],
        p0["committed_x45"]["walk_rungs"]["walk_cons_s25"],
        "streamident_z")
    g_sz = {
        "probe": {"steps": probe_steps, "start": "g1c_install_resume.pt",
                  "seed": CONS_SEED, "key": "all-z (the canon's own)"},
        "row": cellz,
        "claim": ("my draw path keyed all-z reproduces x45's committed "
                  "walk_cons_s25 instrument row BIT-EXACTLY (means 2e-6, "
                  "alive exact, corrs 1e-6)"),
        "pass": bool(okz),
    }
    assert g_sz["pass"], f"G_STREAMIDENT_Z FAILED: {g_sz}"
    metrics["gates"]["G_STREAMIDENT_Z"] = g_sz
    log("G_STREAMIDENT_Z PASS: s25 read |d| "
        f"{cellz['mean:g0']['abs_diff']:.1e}")

    cellt, okt = probe_streamident(
        p0, tuple("t" for _ in ALL16), CKPT_DIR / "e311_TAVINST_post.pt",
        probe_steps, p0["tid"],
        p0["committed_x45"]["anneal_panels"]["anneal_s25"],
        "streamident_t")
    g_st = {
        "probe": {"steps": probe_steps, "start": "e311_TAVINST_post.pt",
                  "seed": CONS_SEED, "key": "all-t (the ruler's own)"},
        "row": cellt,
        "claim": ("my draw path keyed all-t reproduces x45's committed "
                  "anneal_s25 panel row BIT-EXACTLY — the 't' pool is the "
                  "ruler's own stream"),
        "pass": bool(okt),
    }
    assert g_st["pass"], f"G_STREAMIDENT_T FAILED: {g_st}"
    metrics["gates"]["G_STREAMIDENT_T"] = g_st
    log("G_STREAMIDENT_T PASS: s25 read |d| "
        f"{cellt['mean:g0']['abs_diff']:.1e}")
    write_partial("P1a G_STREAMIDENT_Z + G_STREAMIDENT_T")


# ======================================================================
# P1b — THE FOUR ARMS (CPU; per-arm resume; rows computed inline)
# ======================================================================
NET0_CLASS = {
    "midband": ("TAVIREN/install-end WARM/MID-BAND-LADDER-FORMING "
                "(k-of-16 T-windows, k in (7,16), re-forming a "
                "TAVIREN-formed substrate)"),
    "k7replay": ("TAVIREN/install-end WARM/LADDER-FORMING k=7 (the "
                 "committed GPU arm's CPU re-realization)"),
}


def phase_P1b(p0: dict) -> dict:
    log(f"P1b — THE FOUR ARMS (CPU desk, seed {CONS_SEED}, const lr "
        f"{FT_LR}, {CONS_STEPS} steps each; k9/k12/k14 at x50's "
        f"{len(X50_GRAIN)}-rung grain; k7r at the critic's window)")
    starts = {"tav_subject": CKPT_DIR / "e311_TAVINST_post.pt"}
    out = {}
    for arm in ARMS:
        tag = arm["tag"]
        log(f"  [{tag}] ARM START: family={arm['family']} "
            f"slot_keys={''.join(arm['slot_keys'])} rungs={arm['rungs']}")
        out[tag] = run_arm(p0, arm, starts[arm["start"]])
        write_partial(f"P1b arm {tag} finished "
                      f"({len(out[tag]['rows'])} rows)")
    return out


def run_arm(p0: dict, arm: dict, start_path: Path) -> dict:
    tag = arm["tag"]
    host_id = p0["tid"]
    guest_id = p0["zid"]
    rungs = set(arm["rungs"]) if not SMOKE else {1, 2, 4, 8}
    full_rungs = set(FULL_ROW_STEPS) & rungs
    resume_ck = RD / f"arm_resume_{tag}.pt"

    state = {"step": 0, "rows": {}, "draws": []}
    if resume_ck.exists():
        state = torch.load(resume_ck, map_location="cpu",
                           weights_only=False)
        log(f"  [{tag}] RESUMED at step {state['step']}/{CONS_STEPS} "
            f"({len(state['rows'])} rows restored)")

    if int(state.get("step", 0)) >= CONS_STEPS:
        log(f"  [{tag}] resume ckpt already finished — training skipped")
        return {"rows": state["rows"], "draws": state.get("draws", [])}

    net0 = G1.evl_load(load_model_sd(start_path))
    net = copy.deepcopy(net0).to(CPU)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(CONS_SEED)
    if resume_ck.exists():
        net.load_state_dict(state["model"])
        net.train()
        opt.load_state_dict(state["opt"])
        gen.set_state(state["gen_state"])

    e001_flat64 = flat_params_cpu(
        G1.evl_load(load_model_sd(CKPT_DIR / "e001.pt"))).double().numpy()
    draw = make_draw(p0, arm["slot_keys"])
    rows: dict = state["rows"]
    step = state["step"]
    draws: list = state["draws"]

    def row_of(sd) -> dict:
        if step in full_rungs:
            r = instrument(p0, sd, f"{tag}_s{step}", host_id, e001_flat64,
                           dual_id=guest_id)
        else:
            r = light_row(p0, sd, f"{tag}_s{step}", host_id, guest_id)
        r.update({"step": step,
                  "provenance": "replay (CPU desk)",
                  "net0_class": NET0_CLASS[arm["family"]],
                  "family": arm["family"], "k": arm["k"],
                  "slot_keys": "".join(arm["slot_keys"]),
                  "age": 400 + step})
        return r

    def save_resume() -> None:
        torch.save({"model": {k: v.detach().cpu().clone()
                              for k, v in net.state_dict().items()},
                    "opt": opt.state_dict(), "gen_state": gen.get_state(),
                    "step": step, "rows": rows, "draws": draws}, resume_ck)

    while step < CONS_STEPS:
        step += 1
        x, y, m, dr = draw(gen, CPU)
        draws.append({"step": step, "ix": dr[0], "aj": dr[1], "rj": dr[2]})
        loss = train_step(net, opt, x, y, m)
        if step in rungs and step not in rows:
            sd = {k: v.detach().cpu().clone()
                  for k, v in net.state_dict().items()}
            rows[step] = row_of(sd)
            r = rows[step]
            hv = r.get("read", r.get("host"))
            gv = (r.get("dual", {}).get("means", {}).get("g0")
                  if "dual" in r else r.get("guest"))
            log(f"  [{tag}] ROW s{step}: host {hv:.6f} guest "
                f"{(gv if gv is not None else float('nan')):.6f}")
            del sd
        if step % 25 == 0 or step == 1:
            log(f"  [{tag}] s{step:4d} CE {loss:.4f}")
        if step % 25 == 0:
            save_resume()
    save_resume()
    log(f"  [{tag}] replay finished: {CONS_STEPS} steps, "
        f"{len(rows)} rows")
    del net, opt
    return {"rows": rows, "draws": draws}


# ======================================================================
# P1c — G_K7CLASS + G_DRAWS + G_XGRAIN
# ======================================================================
def read_of(row: dict) -> float | None:
    """the host read of a row, full ('read') or light ('host')."""
    if row is None:
        return None
    return row.get("read", row.get("host"))


def phase_P1c(p0: dict, arms: dict) -> None:
    log("P1c — G_K7CLASS (the k7 CPU re-realization vs e346's committed "
        "k7 GPU rows) + G_DRAWS + G_XGRAIN")

    e346_rows = p0["committed_e346"]["arm_rows"]
    k7r_rows = arms["k7r"]["rows"]

    # ---- G_K7CLASS -----------------------------------------------------
    cells = {}
    ok_all = True
    checks = ((8, K7CLASS_S8_TOL),) if SMOKE else \
        ((8, K7CLASS_S8_TOL), (25, K7CLASS_FATE_TOL),
         (125, K7CLASS_FATE_TOL), (300, K7CLASS_FATE_TOL))
    for s, tol in checks:
        mine = read_of(k7r_rows.get(s))
        comp = e346_rows.get(f"k7_s{s}", {}).get("read")
        if mine is None or comp is None:
            cells[f"s{s}"] = {"pass": False, "note": "missing row"}
            ok_all = False
            continue
        cells[f"s{s}"] = {
            "mine_cpu": mine, "committed_gpu": comp,
            "abs_diff": abs(mine - comp), "tol": tol,
            "pass": bool(abs(mine - comp) <= tol)}
        ok_all &= cells[f"s{s}"]["pass"]
        log(f"  [k7r] s{s} CPU {mine:.6f} vs GPU {comp:.6f} "
            f"(|d| {abs(mine - comp):.2e} <= {tol})")
    if SMOKE:
        cells["note"] = ("smoke: certified at the 8-step horizon only "
                         "(e346's own G_XDEVICE smoke convention)")
    g_k7 = {
        "cells": cells,
        "claim": ("the k7 CPU re-realization sits inside the family's "
                  "CPU-GPU class at every certified rung (s8 within 2e-3 "
                  "— e346's own G_XDEVICE observed 1e-5 there; the fate "
                  "trio within 0.05 — the XDEVICE class ceiling, 10x the "
                  "observed band) — the cross-class burden for mixing "
                  "this cell's CPU arms with the committed GPU rungs"),
        "pass": bool(ok_all),
    }
    assert g_k7["pass"], f"G_K7CLASS FAILED: {g_k7}"
    metrics["gates"]["G_K7CLASS"] = g_k7
    log("G_K7CLASS PASS: the k7 CPU re-realization inside the class")

    # ---- G_DRAWS ---------------------------------------------------------
    refs = [arms[a["tag"]]["draws"] for a in ARMS]
    n_expected = CONS_STEPS
    shape_ok = (len(refs[0]) == n_expected
                and all(len(d["ix"]) == 16 and len(d["aj"]) == 8
                        and len(d["rj"]) == 8 for d in refs[0])
                and all(len(d) == n_expected for d in refs))
    cross_equal = all(d == refs[0] for d in refs)
    k7_draws = p0["committed_k7_draws"]
    vs_k7resume = all(d == k7_draws[i] for i, d in enumerate(refs[0])) \
        and (SMOKE or len(k7_draws) == n_expected)
    first_ix_ok = refs[0][0]["ix"] == p0["committed_e345"]["first_ix"]
    g_draws = {
        "n_draws": len(refs[0]), "n_expected": n_expected,
        "shape_ok": bool(shape_ok),
        "cross_arm_bit_equal": bool(cross_equal),
        "vs_e346_k7_resume_bit_equal": bool(vs_k7resume),
        "first_ix": refs[0][0]["ix"],
        "e345_committed_first_ix": p0["committed_e345"]["first_ix"],
        "first_ix_matches_e345": bool(first_ix_ok),
        "claim": ("the four arms consumed ONE draw sequence (ix/aj/rj "
                  "bit-equal across arms, seed 10901, one fresh generator "
                  "per arm) BIT-EQUAL to e346's committed k7 resume record "
                  "(the 300-record reference loaded from the md5-bound "
                  "arm_resume_k7.pt) with the first draw == e345's "
                  "committed first_ix — the arms differ in NOTHING but "
                  "the slot keying, and the keying sits on the committed "
                  "stream itself"),
        "pass": bool(shape_ok and cross_equal and vs_k7resume
                     and first_ix_ok),
    }
    assert g_draws["pass"], f"G_DRAWS FAILED: {g_draws}"
    metrics["gates"]["G_DRAWS"] = g_draws
    log(f"G_DRAWS PASS: {len(refs[0])} draws bit-equal across four arms "
        "AND vs e346's k7 resume; first_ix == e345's committed")

    # ---- G_XGRAIN --------------------------------------------------------
    k9_arm_rungs = [a for a in ARMS if a["tag"] == "k9"][0]["rungs"]
    grain_ok = (X50_GRAIN == sorted(set(X50_GRAIN))
                and len(X50_GRAIN) == 74
                and {5, 10, 300}.issubset(set(X50_GRAIN))
                and {112, 114, 138}.issubset(set(X50_GRAIN))
                and {1, 2}.issubset(set(X50_GRAIN))
                and list(k9_arm_rungs) == X50_GRAIN
                and set(FATE_STEPS).issubset(set(X50_GRAIN))
                and 1 in X50_GRAIN)
    g_xg = {
        "grain": X50_GRAIN, "n_rungs": len(X50_GRAIN),
        "k7r_rungs": K7R_RUNGS,
        "claim": ("the mid-band rung set == x50's frozen dense grain "
                  "VERBATIM ({1,2} U {5,10,...,300} U {112,114,...,138} "
                  "= 74 rungs; the fate steps and s1 inside it); the k7 "
                  "control's rungs = the fate set + the critic's "
                  "s100-s135 window at grain 5"),
        "pass": bool(grain_ok),
    }
    assert g_xg["pass"], f"G_XGRAIN FAILED: {g_xg}"
    metrics["gates"]["G_XGRAIN"] = g_xg
    log("G_XGRAIN PASS: x50's 74-rung grain verbatim; fate steps inside")
    write_partial("P1c G_K7CLASS + G_DRAWS + G_XGRAIN")


# ======================================================================
# P2 — THE ANCHOR ROWS (base/subject/own_install/root)
# ======================================================================
def phase_P2(p0: dict) -> dict:
    log("P2 — THE ANCHOR ROWS (base/subject/own_install/root)")

    e001_sd = load_model_sd(CKPT_DIR / "e001.pt")
    e001_flat64 = flat_params_cpu(G1.evl_load(e001_sd)).double().numpy()
    assert e001_flat64.shape[0] == GB.G1B_PARAMS

    rows: dict = {}
    row = instrument(p0, e001_sd, "base", p0["tid"], e001_flat64,
                     dual_id=p0["zid"])
    row.update({"step": 0, "provenance": "committed-anchor",
                "net0_class": "BASE"})
    rows["base"] = row
    log(f"  [base] p(T) {row['read']:.2e} / p(Z) "
        f"{row['dual']['means']['g0']:.2e}")
    del e001_sd

    tav_sd = load_model_sd(CKPT_DIR / "e311_TAVINST_post.pt")
    row = instrument(p0, tav_sd, "tav_subject", p0["tid"], e001_flat64,
                     dual_id=p0["zid"])
    row.update({"step": 0, "provenance": "committed-anchor",
                "net0_class": "BASE-formed (K10K-room-projected install "
                              "end; age 400)"})
    rows["tav_subject"] = row
    log(f"  [tav_subject] p(T) {row['read']:.6f} / p(Z) "
        f"{row['dual']['means']['g0']:.2e} | |W| "
        f"{row['write_norm']:.2f} inroom {row['in_room']:.3f}")
    del tav_sd

    own_sd = load_model_sd(CKPT_DIR / "g1c_install_resume.pt")
    row = instrument(p0, own_sd, "own_install", p0["zid"], e001_flat64,
                     dual_id=p0["tid"])
    row.update({"step": 0, "provenance": "committed-anchor",
                "net0_class": "BASE-formed (the ZEPHYRA lineage's "
                              "install end; age 400)"})
    rows["own_install"] = row
    log(f"  [own_install] p(Z) {row['read']:.6f} / p(T) "
        f"{row['dual']['means']['g0']:.2e}")
    del own_sd

    root_sd = load_model_sd(CKPT_DIR / "g1c_root.pt")
    row = instrument(p0, root_sd, "root", p0["zid"], e001_flat64)
    row.update({"step": 300, "provenance": "committed-anchor",
                "net0_class": "ROOT", "cert_only": True})
    rows["root"] = row
    log(f"  [root] p(Z) {row['read']:.6f}")
    del root_sd

    metrics["anchor_rows"] = rows
    write_partial("P2 the anchor rows done")
    return rows


# ======================================================================
# P3 — G_ENDPOINTS + G_RIGCONST
# ======================================================================
def phase_P3(p0: dict, anchors: dict) -> None:
    log("P3 — G_ENDPOINTS + G_RIGCONST")

    e345_rows = p0["committed_e345"]["arm_rows"]
    e335 = p0["committed_e335"]
    sub = anchors["tav_subject"]
    cell_sub, ok_sub = compare_row(sub, e345_rows["tav_subject"])
    ep_rows = {
        "tav_subject_vs_e345_row": {"cell": cell_sub, "pass": bool(ok_sub)},
        "base_read_pt": {"mine": anchors["base"]["read"],
                         "committed": e345_rows["base"]["read"]},
        "base_read_pz": {"mine": anchors["base"]["dual"]["means"]["g0"],
                         "channel": "p(Z)",
                         "committed": e335["base_read"]},
        "own_install_s400_pz": {"mine": anchors["own_install"]["read"],
                                "committed": e335["own_install_read"]},
        "root_pz": {"mine": anchors["root"]["read"],
                    "committed": e335["root_read"]},
    }
    for r in ("base_read_pt", "base_read_pz", "own_install_s400_pz",
              "root_pz"):
        ep_rows[r]["abs_diff"] = abs(ep_rows[r]["mine"]
                                     - ep_rows[r]["committed"])
    g_end = {"rows": ep_rows, "tol": READ_TOL,
             "claim": ("the committed anchors reproduce on CPU within "
                       "2e-6: the subject vs e345's committed row, the "
                       "base's p(T)/p(Z), own_install p(Z), the root"),
             "pass": bool(ok_sub and all(
                 ep_rows[r]["abs_diff"] <= READ_TOL
                 for r in ("base_read_pt", "base_read_pz",
                           "own_install_s400_pz", "root_pz")))}
    assert g_end["pass"], f"G_ENDPOINTS FAILED: {g_end}"
    metrics["gates"]["G_ENDPOINTS"] = g_end
    log("G_ENDPOINTS PASS: subject row + 4 anchor literals within 2e-6")

    g1c_meta = torch.load(CKPT_DIR / "g1c_root.pt", map_location="cpu",
                          weights_only=False)["meta"]
    g_rig = {
        "form": "the arm rig's constants cross-checked vs the G1/e281/"
                "g1c sources AND g1c_root.pt's own meta",
        "ft_lr": FT_LR, "betas": "(0.9, 0.95)", "weight_decay": 0.1,
        "clip": 1.0, "steps": CONS_STEPS, "cons_seed": CONS_SEED,
        "batch": "16 pool slots (name-masked; per-slot keying) + 8 "
                 "paired originals + 8 random",
        "starts": "e311_TAVINST_post.pt (all four arms)",
        "g1_cons_seed": G1.CONS_SEED, "g1_ft_lr": G1.FT_LR,
        "g1c_meta_install_seed": g1c_meta.get("install_seed"),
        "g1c_meta_install_steps": g1c_meta.get("install_steps"),
        "midband_flips": {str(k): list(v) for k, v in MIDBAND_FLIPS.items()},
        "k7_flips_inherited": list(K7_FLIPS),
        "pass": bool(FT_LR == 1e-3 and CONS_SEED == 10901
                     and g1c_meta.get("install_seed") == 24314
                     and g1c_meta.get("install_steps") == 400),
    }
    assert g_rig["pass"], f"G_RIGCONST FAILED: {g_rig}"
    metrics["gates"]["G_RIGCONST"] = g_rig
    log("G_RIGCONST PASS (constants == G1/g1c meta; seed 10901; the "
        "mid-band flip scheme recorded)")
    write_partial("P3 G_ENDPOINTS + G_RIGCONST")


# ======================================================================
# P4 — THE ADJUDICATION (the extended k-curve + the k7 control + co-reports)
# ======================================================================
def fate_of(read: float | None) -> str | None:
    if read is None:
        return None
    if read <= DEAD_CUT:
        return "DEAD"
    if read >= ALIVE_CUT_T:
        return "ALIVE"
    return "MID"


def score_of(reads: list[float], how: str) -> float | None:
    vals = [v for v in reads if v is not None]
    if not vals:
        return None
    return min(vals) if how == "min" else sorted(vals)[len(vals) // 2]


def phase_P4(p0: dict, arms: dict) -> dict:
    log("P4 — THE ADJUDICATION (the nine-point k-curve, min + median; "
        "the k7 in-band control; the s1 micro-ladder)")

    lad346 = p0["committed_e346"]["ladder"]
    ann = p0["committed_x45"]["anneal_panels"]
    k_order = ["0", "1", "2", "4", "7", "9", "12", "14", "16"]

    # ---- the k-table (committed rungs + mine) ---------------------------
    ladder: dict = {}
    for k in ("0", "1", "2", "4", "7"):
        r = lad346[k]
        ladder[k] = {
            "provenance": r["provenance"],
            "flip_slots": r.get("flip_slots"),
            "reads": dict(r["reads"]), "guest": dict(r["guest"]),
            "s1": r["s1"], "committed": True}
    ladder["16"] = {
        "provenance": "the anneal ruler (k=16, x45 panels + x52 dense)",
        "flip_slots": list(ALL16),
        "reads": {str(s): ann[f"anneal_s{s}"]["read"] for s in FATE_STEPS},
        "guest": {str(s): None for s in FATE_STEPS},
        "s1": p0["committed_x52"]["rider_rows"]["rider_s1"]["read"],
        "committed": True}
    for arm in ARMS:
        if arm["family"] != "midband":
            continue
        k = str(arm["k"])
        rows = arms[arm["tag"]]["rows"]
        ladder[k] = {
            "provenance": f"mine ({arm['tag']}; flip slots "
                          f"{MIDBAND_FLIPS[arm['k']]})",
            "flip_slots": list(MIDBAND_FLIPS[arm["k"]]),
            "reads": {}, "guest": {}, "s1": None, "committed": False}
        for s in FATE_STEPS + (1,):
            r = rows.get(s)
            if r is None:
                continue
            if s == 1:
                ladder[k]["s1"] = r["read"]
            else:
                ladder[k]["reads"][str(s)] = r["read"]
                ladder[k]["guest"][str(s)] = r["dual"]["means"]["g0"]

    for k in k_order:
        rvals = [ladder[k]["reads"].get(str(s)) for s in FATE_STEPS]
        ladder[k]["score_min"] = score_of(rvals, "min")
        ladder[k]["score_med"] = score_of(rvals, "med")
        ladder[k]["fate_min"] = fate_of(ladder[k]["score_min"])
        ladder[k]["fate_med"] = fate_of(ladder[k]["score_med"])
        ladder[k]["per_rung_fate"] = {
            str(s): fate_of(ladder[k]["reads"].get(str(s)))
            for s in FATE_STEPS}

    # ---- the frozen shape composite -------------------------------------
    hard_ok = all(bool(g.get("pass")) for g in metrics["gates"].values())
    mid_arms = ["9", "12", "14"]
    s9, s12, s14 = (ladder["9"]["score_min"], ladder["12"]["score_min"],
                    ladder["14"]["score_min"])
    mid_fate_reads = [ladder[k]["reads"].get(str(s))
                      for k in mid_arms for s in FATE_STEPS
                      if ladder[k]["reads"].get(str(s)) is not None]
    any_mid_alive = any((v is not None and v >= ALIVE_CUT_T)
                        for v in mid_fate_reads)

    if not hard_ok:
        shape, clause = "TEXTURE", (
            "hard-gate failure: "
            + ", ".join(k for k, g in metrics["gates"].items()
                        if not g.get("pass"))
            + " — nothing adjudicated; tables verbatim")
    elif SMOKE:
        shape, clause = "SMOKE", "nothing adjudicated (smoke stamp)"
    elif None in (s9, s12, s14):
        shape, clause = "TEXTURE", (
            "missing fate reads (a mid-band arm's rows incomplete) — "
            "nothing adjudicated; the k-table is the report")
    elif (ladder["9"]["fate_min"] == "ALIVE"
            or ladder["12"]["fate_min"] == "ALIVE"):
        shape = "EDGE-EARLY"
        who = [k for k in ("9", "12") if ladder[k]["fate_min"] == "ALIVE"]
        clause = (
            f"score_min ALIVE at k={'+'.join(who)} ("
            + ", ".join(f"k{k}: {ladder[k]['score_min']:.4f}"
                        for k in who)
            + ") — the protection edge lands in (7,12]: the lab lean's "
              "reading (k=7's 0.6% near-miss + the read-gap climb)")
    elif ladder["14"]["fate_min"] == "ALIVE":
        shape = "EDGE-MID-(12,16)"
        clause = (
            f"score_min(14) {s14:.4f} ALIVE with k=9 ({s9:.4f}) and k=12 "
            f"({s12:.4f}) not — the edge lands in (12,16): an honest "
            "shape OUTSIDE the three named (the pre-named fallback); no "
            "dispatch bar fires")
    elif any_mid_alive:
        alive_where = [f"k{k}s{s}" for k in mid_arms for s in FATE_STEPS
                       if (ladder[k]["reads"].get(str(s)) is not None
                           and ladder[k]["reads"][str(s)] >= ALIVE_CUT_T)]
        shape = "RECOVERY-ONLY"
        clause = (
            f"no mid-band arm min-score ALIVE (min-scores "
            + ", ".join(f"k{k}: {ladder[k]['score_min']:.4f}"
                        for k in mid_arms)
            + ") while mid-band fate reads clear ALIVE at "
            + ", ".join(alive_where)
            + " — partial dose buys TRANSIENT ALIVES, not protection "
              "(the min convention reads the dip floor; the s1 micro-"
              "ladder's dose-insensitivity the hint): the dispatch's "
              "counter reading")
    else:
        shape = "EDGE-LATE-OR-NEVER"
        clause = (
            f"no mid-band arm min-score ALIVE (min-scores "
            + ", ".join(f"k{k}: {ladder[k]['score_min']:.4f}"
                        for k in mid_arms)
            + ") AND no mid-band fate read reaching the ALIVE cut "
            f"{ALIVE_CUT_T:.4f} — the mid-band never protects: the fate "
              "edge belongs to k=16 alone (full dose or nothing)")

    # ---- the k7 in-band control (the critic's named control) -------------
    k7r_rows = arms["k7r"]["rows"]
    inband = {s: read_of(k7r_rows[s]) for s in sorted(k7r_rows)
              if 100 <= s <= 135}
    k7_fate = {s: read_of(k7r_rows[s]) for s in FATE_STEPS
               if s in k7r_rows}
    critic_steps = (105, 110, 115, 120, 130)
    critic_hits = [s for s in critic_steps
                   if s in inband and inband[s] > DEAD_CUT]
    critic_pred_hit = bool(critic_hits)
    ext_pool = list(k7_fate.values()) + list(inband.values())
    inband_ext_min = min(ext_pool) if ext_pool else None
    n_sub = sum(1 for v in inband.values() if v <= DEAD_CUT)
    k7_control = {
        "inband_reads": {str(s): v for s, v in inband.items()},
        "fate_reads_cpu": {str(s): v for s, v in k7_fate.items()},
        "committed_gpu_fates": {str(s): lad346["7"]["reads"][str(s)]
                                for s in FATE_STEPS},
        "critic_prediction": REGISTERED["critic_coregistered"],
        "critic_prediction_hit": critic_pred_hit,
        "critic_clearing_steps": critic_hits,
        "n_inband_sub_dead": n_sub, "n_inband": len(inband),
        "inband_extended_min": inband_ext_min,
        "patch_class": ("PATCH (the min fluctuates: "
                        f"{n_sub}/{len(inband)} in-band rungs sub-DEAD "
                        "with clears between)"
                        if 0 < n_sub < len(inband) else
                        ("ALL-SUB-DEAD" if n_sub == len(inband)
                         else "ALL-CLEAR")),
        "flip_narrative": (
            f"the in-band-extended min {inband_ext_min:.6f} "
            + ("CLEARS the 0.10 DEAD cut — the critic's 'flips the arm "
               "to MID' reading fires in the co-report lane (e346's "
               "committed fate-trio score 0.099412 STANDS as committed)"
               if (inband_ext_min is not None and inband_ext_min > DEAD_CUT
                   and lad346["7"]["score"] <= DEAD_CUT) else
               "stays sub-DEAD — the committed k7 score stands un-flipped"
               " even in the co-report lane"))
        if inband_ext_min is not None else "n/a",
    }

    # ---- the s1 micro-ladder extension -----------------------------------
    s1_row = {k: ladder[k]["s1"] for k in k_order}
    s1_mid = [ladder[k]["s1"] for k in mid_arms]
    s1_armor_edge = any((v is not None and v > DEAD_CUT) for v in s1_mid)
    micro = {
        "s1_reads": s1_row,
        "committed_datum": "0.006756 (k=0) .. 0.014052 (k=7) vs 0.748952 "
                           "(k=16, x52's rider_s1)",
        "mid_band_all_dead_class": bool(not s1_armor_edge),
        "armor_edge_note": (
            "k14's s1 > 0.10 would locate an ARMOR edge inside (12,16) — "
            "the registered surprise branch, DISCLOSED ONLY"
            if s1_armor_edge else
            "s1 dead-class at every mid-band k — the all-or-nothing "
            "armor frame extends across the band"),
    }

    # ---- the dense texture co-report --------------------------------------
    dense = {arm["tag"]: {str(s): read_of(r)
                          for s, r in sorted(arms[arm["tag"]]["rows"]
                                             .items())}
             for arm in ARMS}
    x52r = p0["committed_x52"]["rider_rows"]
    ruler_dense = {k: v["read"] for k, v in x52r.items()
                   if k.startswith("rider_s")}
    e345_traj = p0["committed_e345"]["traj"]

    # ---- P-x55a scoring ----------------------------------------------------
    if shape == "TEXTURE" or SMOKE:
        p_hit = {"verdict": None,
                 "note": "TEXTURE/SMOKE — P-x55a UNSCORED per the "
                         "registered falsifier clause"}
    else:
        p_hit = {"verdict": bool(shape == "RECOVERY-ONLY"),
                 "guess": REGISTERED["P_x55a"]["guess"]}

    adj = {
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "fate_cuts": {"dead": DEAD_CUT, "alive_T": ALIVE_CUT_T,
                      "score_min_def": "min over {s25,s125,s300} of the "
                                       "CPU instrument host read (e346's "
                                       "frozen convention — THE "
                                       "adjudicating score)",
                      "score_med_def": "median over the same three (the "
                                       "R83 MEDIAN CO-REPORT — standing, "
                                       "never re-adjudicating e346's "
                                       "committed k<=7 verdict)"},
        "ladder": ladder, "k_order": k_order,
        "shape": shape, "shape_clause": clause,
        "composite_order": ("TEXTURE (hard gates) -> EDGE-EARLY (min-ALIVE "
                            "at k in {9,12}) -> EDGE-MID-(12,16) (k14 "
                            "alone ALIVE; the pre-named fallback) -> "
                            "RECOVERY-ONLY (no mid-band min-ALIVE + "
                            "mid-band fate reads ALIVE) -> EDGE-LATE-OR-"
                            "NEVER (no mid-band min-ALIVE + no mid-band "
                            "fate read ALIVE) -> MIXED"),
        "k7_control": k7_control,
        "s1_micro_ladder": micro,
        "dense_coreport": {"mine": dense,
                           "ruler_k16_committed": ruler_dense,
                           "k0_committed_traj_gpu_disclosure":
                               [{"step": t["step"], "host": t["g0_pt"]}
                                for t in e345_traj]},
        "P_x55a": {"guess": REGISTERED["P_x55a"]["guess"],
                   "hit": p_hit, "scored": REGISTERED["P_x55a"]["scored"]},
        "critic_coscore": {"prediction": REGISTERED["critic_coregistered"],
                           "hit": (None if (shape == "TEXTURE" or SMOKE)
                                   else critic_pred_hit)},
    }
    metrics["adjudication"] = adj
    metrics["arm_rows"] = {tag: arms[tag]["rows"] for tag in arms}
    write_partial("P4 the adjudication done")
    log(f"P4: shape {shape} / critic's k7 prediction "
        f"{'HIT' if critic_pred_hit else 'MISSED'}")
    return adj


# ======================================================================
# P5 — THE OUTPUTS (the PNG + the REPORT)
# ======================================================================
def make_png(adj: dict, arms: dict, p0: dict) -> None:
    ladder = adj["ladder"]
    k_order = adj["k_order"]
    kx = {k: int(k) for k in k_order}
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 3)

    # ---- panel 1: THE K-CURVE (min + median) ---------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    for k in k_order:
        col = "tab:red" if ladder[k]["committed"] else "tab:green"
        mk = "s" if ladder[k]["committed"] else "o"
        if ladder[k]["score_min"] is not None:
            ax1.plot(kx[k], ladder[k]["score_min"], mk, color=col, ms=10,
                     zorder=3)
        if ladder[k]["score_med"] is not None:
            ax1.plot(kx[k], ladder[k]["score_med"], mk, color=col, ms=10,
                     mfc="none", mew=1.8, zorder=3)
        for s, v in ladder[k]["reads"].items():
            ax1.plot(kx[k], v, ".", color=col, alpha=0.4, ms=6)
    mins = [(kx[k], ladder[k]["score_min"]) for k in k_order
            if ladder[k]["score_min"] is not None]
    meds = [(kx[k], ladder[k]["score_med"]) for k in k_order
            if ladder[k]["score_med"] is not None]
    ax1.plot([a for a, _ in mins], [b for _, b in mins], "--",
             color="tab:gray", lw=1.2, alpha=0.8, zorder=1,
             label="score_min (e346's convention)")
    ax1.plot([a for a, _ in meds], [b for _, b in meds], ":",
             color="tab:purple", lw=1.6, alpha=0.8, zorder=1,
             label="score_med (the R83 co-report)")
    ax1.axhline(DEAD_CUT, color="black", ls="--", lw=1,
                label=f"DEAD cut {DEAD_CUT}")
    ax1.axhline(ALIVE_CUT_T, color="tab:blue", ls="--", lw=1,
                label=f"ALIVE cut {ALIVE_CUT_T:.4f}")
    ax1.set_xlabel("k (TAVIREN-keyed windows of 16 per step)")
    ax1.set_ylabel("host p(T) score")
    ax1.set_title(f"THE MID-BAND LADDER — {adj['shape']}\n"
                  "(red squares = committed rungs; green = mine; filled "
                  "min, open median; dots = the three fate reads)")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.25)

    # ---- panel 2: the dense trajectories ---------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    tr0 = adj["dense_coreport"]["k0_committed_traj_gpu_disclosure"]
    ax2.plot([t["step"] for t in tr0], [t["host"] for t in tr0], "-",
             color="tab:red", lw=1.0, alpha=0.8,
             label="k=0 (e345 committed, GPU disclosure)")
    xr = adj["dense_coreport"]["ruler_k16_committed"]
    pts = sorted((int(re.findall(r"\d+", k)[0]), v)
                 for k, v in xr.items())
    ax2.plot([a for a, _ in pts], [b for _, b in pts], "-", lw=1.2,
             color="tab:orange", alpha=0.9,
             label="k=16 (x52's dense ruler, committed)")
    for tag in ("k9", "k12", "k14"):
        d = adj["dense_coreport"]["mine"][tag]
        pts = sorted((int(s), v) for s, v in d.items() if v is not None)
        ax2.plot([a for a, _ in pts], [b for _, b in pts], "-", lw=1.0,
                 alpha=0.85, label=f"{tag} (mine, dense)")
    d7 = adj["dense_coreport"]["mine"]["k7r"]
    pts = sorted((int(s), v) for s, v in d7.items() if v is not None)
    ax2.plot([a for a, _ in pts], [b for _, b in pts], "o-", ms=3,
             lw=1.0, color="tab:brown", alpha=0.9,
             label="k7r (the critic's control)")
    ax2.axhline(ALIVE_CUT_T, color="tab:blue", ls="--", lw=0.8)
    ax2.axhline(DEAD_CUT, color="black", ls="--", lw=0.8)
    ax2.set_xlabel("stream step")
    ax2.set_ylabel("host p(T)")
    ax2.set_title("the dense grain — host read vs step")
    ax2.legend(fontsize=6)
    ax2.grid(alpha=0.25)

    # ---- panel 3: the k7 in-band window ----------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    d7 = adj["dense_coreport"]["mine"]["k7r"]
    pts = sorted((int(s), v) for s, v in d7.items()
                 if v is not None and 90 <= int(s) <= 300)
    ax3.plot([a for a, _ in pts], [b for _, b in pts], "o-", ms=5,
             color="tab:brown", lw=1.2, label="k7r (CPU re-realization)")
    comp = adj["k7_control"]["committed_gpu_fates"]
    ax3.plot([int(s) for s in comp], list(comp.values()), "s", ms=8,
             color="tab:gray", label="e346 committed k7 (GPU)")
    ax3.axhline(DEAD_CUT, color="black", ls="--", lw=1)
    ax3.axhline(ALIVE_CUT_T, color="tab:blue", ls="--", lw=1)
    ax3.set_xlabel("stream step")
    ax3.set_ylabel("host p(T)")
    ax3.set_title(f"THE k7 IN-BAND CONTROL — {adj['k7_control']['patch_class']}\n"
                  f"critic's prediction "
                  f"{'HIT' if adj['k7_control']['critic_prediction_hit'] else 'MISSED'} "
                  f"(clears at {adj['k7_control']['critic_clearing_steps']})")
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.25)

    # ---- panel 4: the s1 micro-ladder ------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    for k in k_order:
        col = "tab:red" if ladder[k]["committed"] else "tab:green"
        mk = "s" if ladder[k]["committed"] else "o"
        if ladder[k]["s1"] is not None:
            ax4.plot(kx[k], ladder[k]["s1"], mk, color=col, ms=10)
    ax4.axhline(DEAD_CUT, color="black", ls="--", lw=1,
                label="DEAD cut 0.10")
    ax4.set_xlabel("k")
    ax4.set_ylabel("s1 host p(T)")
    ax4.set_title("THE s1 MICRO-LADDER (the armor observable)\n"
                  "dead-class at every partial k = all-or-nothing armor")
    ax4.legend(fontsize=7)
    ax4.grid(alpha=0.25)

    # ---- panel 5: median vs min -------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    ks = [k for k in k_order if ladder[k]["score_min"] is not None]
    x = np.arange(len(ks))
    ax5.bar(x - 0.18, [ladder[k]["score_min"] for k in ks], 0.36,
            label="score_min", color="tab:gray")
    ax5.bar(x + 0.18, [ladder[k]["score_med"] for k in ks], 0.36,
            label="score_med (R83 co-report)", color="tab:purple",
            alpha=0.75)
    ax5.axhline(DEAD_CUT, color="black", ls="--", lw=1)
    ax5.axhline(ALIVE_CUT_T, color="tab:blue", ls="--", lw=1)
    ax5.set_xticks(x, [f"k{k}" for k in ks], fontsize=8)
    ax5.set_ylabel("score")
    ax5.set_title("the R83 convention repair, standing: both scores")
    ax5.legend(fontsize=7)
    ax5.grid(alpha=0.25, axis="y")

    # ---- panel 6: the verdict table ----------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    tbl = [["observable", "verdict"]]
    tbl.append(["THE SHAPE", adj["shape"]])
    tbl.append(["k7 control (critic's)",
                adj["k7_control"]["patch_class"]])
    tbl.append(["critic's prediction",
                "HIT" if adj["k7_control"]["critic_prediction_hit"]
                else "MISSED"])
    tbl.append(["s1 micro-ladder",
                "dead-class" if adj["s1_micro_ladder"]["mid_band_all_dead_class"]
                else "ARMOR EDGE (surprise)"])
    tbl.append(["P-x55a",
                str(adj["P_x55a"]["hit"])])
    ax6.table(cellText=tbl[1:], colLabels=tbl[0], loc="center",
              cellLoc="left")
    ax6.set_title("the verdicts against the frozen bars", fontsize=9)

    fig.suptitle("X55 THE MID-BAND LADDER, FATE-RESOLVED — k in {9,12,14} "
                 "of 16 on e346's rig at x50's dense grain + the k7 "
                 "in-band control + the median co-report standing; "
                 f"shape {adj['shape']}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = RD / "x55_midband_ladder.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    log(f"[png] {out.name} written")


def write_report(adj: dict, p0: dict) -> None:
    L: list[str] = []
    A = L.append
    ladder = adj["ladder"]
    k_order = adj["k_order"]

    def fmt(v, p=4):
        return "-" if v is None else (
            f"{v:.2e}" if v is not None and v < 0.01 else f"{v:.{p}f}")

    A("# X55 — THE MID-BAND LADDER, FATE-RESOLVED")
    A("")
    A(f"* SHAPE VERDICT: **{adj['shape']}** — {adj['shape_clause']}")
    A(f"* P-x55a (my registered read): {adj['P_x55a']['hit']} "
      f"(guess: {adj['P_x55a']['guess']})")
    A("")
    A(f"* the critic's co-registered k7 prediction: "
      f"{'HIT' if adj['k7_control']['critic_prediction_hit'] else 'MISSED'}"
      f" — {adj['k7_control']['patch_class']}; clears at "
      f"{adj['k7_control']['critic_clearing_steps']}")
    n_pass = sum(1 for g in metrics["gates"].values() if g.get("pass"))
    n_all = len(metrics["gates"])
    A(f"* gates: {n_pass}/{n_all} PASS"
      + ("" if n_pass == n_all else " — FAILURES: "
         + ", ".join(k for k, g in metrics["gates"].items()
                     if not g.get("pass"))))
    A("")
    A("## THE K-BAND TABLE (fate cuts: DEAD <= 0.10, ALIVE >= "
      f"{ALIVE_CUT_T:.6f}; score_min = e346's frozen convention — THE "
      "adjudicating score; score_med = the R83 MEDIAN CO-REPORT, "
      "standing, never re-adjudicating e346's committed k<=7 verdict)")
    A("")
    A("| k | provenance | flip slots | p(T) s25 | p(T) s125 | p(T) s300 "
      "| score_min | fate(min) | score_med | fate(med) | p(Z) guest s300 "
      "| s1 host |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in k_order:
        r = ladder[k]["reads"]
        g = ladder[k]["guest"]
        A(f"| {k} | {ladder[k]['provenance']} | "
          f"{ladder[k].get('flip_slots', '-')} | "
          f"{fmt(r.get('25'))} | {fmt(r.get('125'))} | "
          f"{fmt(r.get('300'))} | {fmt(ladder[k]['score_min'], 6)} | "
          f"{ladder[k]['fate_min']} | {fmt(ladder[k]['score_med'], 6)} | "
          f"{ladder[k]['fate_med']} | {fmt(g.get('300'))} | "
          f"{fmt(ladder[k]['s1'], 6)} |")
    A("")
    A(f"* the composite order (frozen): {adj['composite_order']}")
    A("")
    A("## THE k7 IN-BAND CONTROL (the critic's named control)")
    A("")
    kc = adj["k7_control"]
    A(f"* in-band reads s100-s135 (grain 5): "
      + ", ".join(f"s{s} {fmt(v, 6)}" for s, v in kc["inband_reads"]
                  .items()))
    A(f"* the CPU re-realization's fate trio: "
      + ", ".join(f"s{s} {fmt(v, 6)}"
                  for s, v in kc["fate_reads_cpu"].items())
      + " vs the committed GPU arm "
      + ", ".join(f"s{s} {fmt(v, 6)}"
                  for s, v in kc["committed_gpu_fates"].items())
      + " (G_K7CLASS bounds the class)")
    A(f"* patch class: **{kc['patch_class']}** — {kc['flip_narrative']}")
    A(f"* the critic's prediction: "
      f"{'HIT' if kc['critic_prediction_hit'] else 'MISSED'} "
      f"({kc['critic_prediction']})")
    A("")
    A("## THE s1 MICRO-LADDER EXTENSION (the armor observable; co-report)")
    A("")
    A("| k | " + " | ".join(k_order) + " |")
    A("|---|" + "---|" * len(k_order))
    A("| s1 host p(T) | " + " | ".join(
        fmt(ladder[k]["s1"], 6) for k in k_order) + " |")
    A(f"* {adj['s1_micro_ladder']['armor_edge_note']} (committed datum: "
      f"{adj['s1_micro_ladder']['committed_datum']})")
    A("")
    A("## Provenance")
    A(f"* birth commit: {metrics.get('birth_commit')}; final head: "
      f"{metrics.get('git_head_final')}")
    A("* CPU desk ONLY (threads 4, zero CUDA — the GPU lane is sibling "
      "x36's, never touched; runtime asserted); timestamps UTC only; "
      "every artifact md5-bound (metrics.gates.G_MD5); the four arms' "
      "draws bit-equal to e346's committed k7 resume record (G_DRAWS)")
    A("* catches + disclosures: see metrics.deviations")
    A("")
    A("*No NOTES/THINKING/QUEUE/STATE edits (dispatch; the heartbeat "
      "folds).*")
    (RD / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    log("[report] REPORT.md written")


# ======================================================================
# MAIN
# ======================================================================
def birth_hash() -> str:
    """The commit that introduced this file (the birth commit where the
    bars + parity + P-x55a + the flip scheme were frozen BEFORE any
    compute); 'untracked (pre-birth)' if the file is not yet committed."""
    try:
        h = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%h", "-1",
             "--", "lab/x55_midband_ladder.py"],
            cwd=str(REPO), capture_output=True, text=True,
            timeout=10).stdout.strip()
        return h or "untracked (pre-birth)"
    except Exception:                                       # noqa: BLE001
        return "unavailable"


def main() -> None:
    log(f"X55 — THE MID-BAND LADDER, FATE-RESOLVED (smoke={SMOKE}) "
        f"-> {RD}")
    metrics["birth_commit"] = birth_hash()
    assert torch.get_num_threads() == 4
    write_partial("startup (bars + parity + P-x55a registered, committed "
                  "at birth)")
    set_seed(55500)             # global init only; every RNG is its own

    p0 = phase_P0()
    phase_P1a(p0)                     # G_STREAMIDENT_Z + _T (CPU)
    arms = phase_P1b(p0)              # THE FOUR ARMS (CPU desk)
    phase_P1c(p0, arms)               # G_K7CLASS + G_DRAWS + G_XGRAIN
    anchors = phase_P2(p0)
    phase_P3(p0, anchors)
    adj = phase_P4(p0, arms)

    gates_pass = all(bool(g.get("pass")) for g in metrics["gates"].values())
    if not gates_pass and adj["shape"] != "TEXTURE":
        adj["shape"] = "TEXTURE"
    metrics["status"] = (
        "SMOKE: nothing adjudicated" if SMOKE else
        f"ADJUDICATED: {adj['shape']} (critic's k7 prediction "
        f"{'HIT' if adj['k7_control']['critic_prediction_hit'] else 'MISSED'})")
    make_png(adj, arms, p0)
    metrics["git_head_final"] = git_head()
    metrics["date_finished"] = common.now_iso()
    save_json(RD / "metrics.json", metrics)
    write_report(adj, p0)
    log(f"DONE — shape {adj['shape']} / critic's k7 prediction "
        f"{'HIT' if adj['k7_control']['critic_prediction_hit'] else 'MISSED'} "
        f"(gates {sum(1 for g in metrics['gates'].values() if g.get('pass'))}/"
        f"{len(metrics['gates'])})")
    assert not torch.cuda.is_initialized(), \
        "ZERO-CUDA violation: the CUDA context was initialized"


if __name__ == "__main__":
    main()
