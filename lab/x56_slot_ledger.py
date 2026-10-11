"""X56 — THE SLOT LEDGER (the R83 minted pair's second cell; the
currency-corrected slot-split, riding x55's arms). This docstring carries
the question + background + design + bars + the parity block + P-x56a,
plus every frozen operationalization, committed at birth BEFORE any
compute. Adjudicate against exactly this; no bar shopping.

THE QUESTION (dispatch verbatim): "x56 — THE SLOT LEDGER (the
currency-corrected slot-split, riding x55's arms): L(k) = the logit-lift
sum p_host + p_guest at every rung (the METHODS-2-correct logit currency,
not raw softmax); per-name mass trajectories."

BACKGROUND (the committed record this cell extends; extend, don't repeat):
e346's k=7 slot-split observation — "the two names literally split the
slot" — was drawn from RAW SOFTMAX masses (the guest p(Z) falling to 0.41
at k=7 exactly when the host recovers). R83's critic killed the noun under
METHODS 2's CURRENCY doctrine (THE_LAWS_V3.md METHODS 2, verbatim: "use
the same-organism per-state displacement-push currency; raw p/p-base
lifts compose with prior shifts and manufacture doublings"): both channels
are raw probabilities from ONE softmax, so a host rise mechanically
depresses the guest (x16's denominator-intrusion lesson). The critic's
named control: "re-express BOTH channels as logit-lift vs base ... Guest
lift flat while p falls => arithmetic, T336 re-words; lift falls => real
competition, and the conservation question then sharpens into a mechanism
cell." THIS CELL is that control, generalized into a LEDGER over the whole
dose range: L(k) at every rung, the per-name masses, the k=16 guest read
(THE 2x2's UNMEASURED CORNER — x45's ruler panels never recorded the
guest channel; e346's ladder table carries guest = None at k=16), and the
arithmetic-vs-real decomposition of the guest's falls.

THE DESIGN (dispatch verbatim): "L(k) = the logit-lift sum p_host +
p_guest at every rung (the METHODS-2-correct logit currency, not raw
softmax); per-name mass trajectories. BARS: ZERO-SUM-CLEAN (the logit
ledger conserves — one seat, hard) / SUM-DIPS (the ledger dips where both
lose — competition with an external sink) / CAPACITY-TWO (the ledger
supports two simultaneous occupants at some k). FREE RIDER: the k=16
guest read (the 2x2's unmeasured corner)."

BARS (dispatch VERBATIM, frozen in this birth commit BEFORE compute):
  - "ZERO-SUM-CLEAN (the logit ledger conserves — one seat, hard)"
  - "SUM-DIPS (the ledger dips where both lose — competition with an
    external sink)"
  - "CAPACITY-TWO (the ledger supports two simultaneous occupants at
    some k)"

THE PARITY BLOCK (the lab lean AND the counter side by side, both carried
verbatim from the dispatch):
  LAB LEAN: "SUM-DIPS weakly (0.51-at-k7 vs 0.65-at-k0 already hints
  down)" — the coordinator's hint is RAW-P arithmetic (the currency this
  cell exists to correct); disclosed verbatim, never re-worded.
  COUNTER: "ZERO-SUM-CLEAN".

P-x56a (THE EXECUTOR'S OWN READ, registered BEFORE compute, frozen HERE
at birth; predictions are scored): **CAPACITY-TWO** — against BOTH the
lean's SUM-DIPS and the counter's ZERO-SUM-CLEAN. SCORED: TRUE iff the
verdict == CAPACITY-TWO (plain). GROUNDS, WITH THE HONEST PRE-DISCHARGE
DISCLOSURE (the e346-verbatim committed fate cells already decide the
level clause unless a gate fails — this cell's LIVE content is the k16
corner, the mid-band trajectory fill, the dense dip texture, and the
arithmetic-vs-real decomposition): (1) THE COMMITTED LOGIT LEDGER ALREADY
EXCEEDS THE SOLO BUDGET AT THE PARTIAL DOSES — on e346's committed fate
cells, delta(k,s) := L(k,s) - L(0,s) runs +1.23 (k1,s25) to +4.86
(k2,s125) across k1-k7, with BOTH channels >= 0.10 at k4 s25
(0.115/0.458) and k7 s25/s300 (0.435/0.199; 0.299/0.407): the currency
correction FLIPS the raw-p hint's sign — the same cells the lean read as
"0.51 vs 0.65, hints down" read +3-to-+4 LOGITS UP in the logit currency
(the host's seat gain outweighs the guest's loss because the guest seat
starts 1.3e-5-empty with ~11 logits of headroom); the lean dies by the
correction, the counter dies because the trade is NOT exact — the
host's rise exceeds the guest's fall. (2) THE GUEST'S FALL IS PARTLY
REAL — at k4->k7 s300 the guest loses 0.80 logits of which only ~0.2 is
denominator arithmetic (the renormalization prediction p_g*(1-p_h')/
(1-p_h)); ~0.6 is real competition — the anti-motion is real but
non-conserving (the R83 critic's 'lift falls => real competition'
branch). (3) THE k16 CORNER (the free rider): the all-T stream carries
ZERO Z-rehearsal — by the execution symmetry (the ZxT mirror's guest
rose because it WAS the stream's key; the k=0 host died because it
wasn't), the k16 guest (Z, unrehearsed) should DIE to 1e-4..5e-2-class
while the host holds 0.68/0.31/0.72 — the corner completes the ledger
with the Z-seat collapsed; L(16) may fall to or below the k0 budget
(the across-dose collapse CO-REPORT, not the verdict: 'both lose' does
not describe a cell where the host wins). EXPECTED SHAPE: delta(k)
arches — rising from k1 through k4-k7 (+3..+5), the mid-band k9-k14
holding or rising further (the host seat filling), then k16 falling back
to -1.5..+2.5 as the guest seat collapses; BOTH-LOSE DIPS (lift_h AND
lift_g falling together over >= 0.5 logits within an arm) recur at the
s125-class patches (committed at k7: -2.08 between s25 and s125); the
dense grain (x55's arms + my k16 replay) resolves the patches' width.
FALSIFIER: any hard-gate failure -> TEXTURE (nothing adjudicated;
P-x56a UNSCORED per the family's precedent clause).

==== THE FROZEN OPERATIONALIZATIONS (picked + frozen HERE at birth) ====

* THE CURRENCY (frozen; the dispatch's "the METHODS-2-correct logit
  currency"): logit(p) := ln(p / (1 - p)); lift_c(s) := logit(p_c(s)) -
  logit(p_c^base) with the BASE = e001's committed base-row channel means
  (p_T^base = 0.0023042955435812473, p_Z^base = 1.3383959412749391e-05 —
  e346's committed literals, G_BASELITERALS-bound). L(k,s) := lift_host +
  lift_guest. The raw softmax masses are co-reported per name (the
  trajectories), never adjudicating. DISCLOSED: METHODS 2's literal named
  currency is x25's per-state displacement-push (a heavier instrument);
  this cell freezes the dispatch's logit-lift re-read — the R83 critic's
  first-named option — as the operational currency, with the denominator
  arithmetic DECOMPOSED explicitly (below) so the currency correction is
  auditable rather than asserted.

* THE LEDGER'S SAMPLING MAP (frozen): adjudicating cells = the fate steps
  {s25, s125, s300} for every k in {0,1,2,4,7,9,12,14,16} (k0/1/2/4/7
  committed from e346's ladder table; k9/12/14 runtime-read from
  runs/x55/metrics.json — the shared session; k16 = MY ruler replay) +
  the s1 co-report column. Dense trajectories (co-report, never
  adjudicating): x55's k9/k12/k14 light rows at x50's 74-rung grain, the
  k7r window (s100-s135), my k16 replay at the same grain, and k=0's
  committed GPU traj (e345's per-step dual channel — the disclosure
  class, labelled as such).

* THE BUDGET + DELTA (frozen): B(s) := L(0, s) — the k=0 committed
  ledger at the matched fate rung (the guest-solo occupancy level: the
  host dead, the stream's key alone in the seat). delta(k,s) := L(k,s) -
  B(s). TOL_LEDGER := 1.0 logit (the conservation band: a full logit of
  odds — substantive, far above the 2e-6 read determinism, half-decade
  honest).

* THE OCCUPANCY CUT (frozen): a channel is OCCUPIED at a cell iff its
  read >= 0.10 (the era's DEAD cut, applied symmetrically to host and
  guest).

* THE BOTH-LOSE DIP DETECTOR (frozen): a qualifying dip at (k; s -> s')
  iff lift_h(s') < lift_h(s) AND lift_g(s') < lift_g(s) AND L(s') -
  L(s) <= -TOL_DIP with TOL_DIP := 0.5 logits — the external-sink
  signature (mass leaves BOTH names). Computed at the adjudicating
  grain (the fate trio: s25->s125, s125->s300) for every k, and at the
  DENSE grain (every consecutive dense pair) for x55's arms + my k16
  replay (co-report: the patches' width + depth).

* THE VERDICT COMPOSITE (frozen order):
  TEXTURE (any hard-gate failure; nothing adjudicated) ->
  CAPACITY-TWO (SOME cell k > 0, fate rung s, with BOTH channels
  occupied (>= 0.10) AND delta(k,s) >= +TOL_LEDGER — "the ledger
  supports two simultaneous occupants at some k", the dispatch's own
  wording: one cell suffices) ->
  SUM-DIPS (no capacity cell AND >= 1 qualifying both-lose dip at the
  adjudicating grain) ->
  ZERO-SUM-CLEAN (no capacity cell AND no qualifying dip AND
  |delta(k,s)| <= TOL_LEDGER at every adjudicating cell — the trade is
  exact; one seat, hard) ->
  MIXED (anything else — named + disclosed; e.g. growth without
  co-occupancy, or an across-dose collapse at the full dose: the k16
  corner falling below budget while the host alone wins is THIS branch,
  co-reported inside whatever verdict fires).
  MANDATORY CO-REPORTS inside every verdict: the dip census (where, how
  deep — the external-sink residue), the k16 corner (the guest's fate +
  L(16) vs budget), the arithmetic-vs-real decomposition.

* THE ARITHMETIC-VS-REAL DECOMPOSITION (frozen; the R83 critic's
  discriminator, free): for each adjacent k-pair (1->2->4->7->9->12->14->
  16) at each fate rung: the guest's OBSERVED lift change; the
  DENOMINATOR-ONLY prediction (holding the guest's logit and all other
  logits fixed, a host rise renormalizes the guest's raw mass to
  p_g * (1 - p_h') / (1 - p_h)); the RESIDUAL = observed - predicted =
  the REAL competition share. real < 0 => real competition; |real| small
  => arithmetic. Co-reported, never adjudicating.

* THE FREE RIDER — THE k16 GUEST READ (the 2x2's unmeasured corner):
  the ruler replay (all-16-t slot keys, e346's draw arithmetic, CPU)
  with DUAL-channel rows at x50's grain; the host certified vs x45's
  committed panels (G_RULERCERT, 2e-6 — the CPU-class reproduction
  x52 already proved); the guest column = THE CORNER. The corner's own
  mini-verdict (co-report): GUEST-DEAD iff all three fate guests <=
  0.10 / GUEST-HELD iff any >= 0.10. Registered expectation (inside
  P-x56a grounds): GUEST-DEAD (the unrehearsed name dies; the stream's
  key holds the seat alone at the full dose).

* THE ARMS (one new CPU replay; everything else runtime-read):
    k16r: the TAVIREN subject through ALL-16-T slot_keys (the ruler's
    own stream), 300 steps, rungs = x50's 74-rung grain, full dual rows
    at {1,25,125,300}, light dual rows elsewhere. NET0 CLASS (x32's
    standing rule, recorded per row): "TAVIREN/install-end WARM/
    RULER-FORMING (all-16 T-windows — the ruler's own stream, CPU
    re-realization)". NOT A DUPLICATE: x45's panels were host-only at
    the committed grain; x52's dense replay was host-only; the GUEST
    channel has never been read on this stream — the corner.

* COMPUTE ROUTE (chosen + disclosed): CPU DESK ONLY, torch threads 4,
  ZERO CUDA (the GPU lane is sibling x36's; runtime asserted). Per-arm
  resume (runs/x56/arm_resume_k16r.pt, gitignored *.pt). The shared
  session with x55 disclosed: this cell runtime-reads runs/x55/
  metrics.json (md5-bound HERE at birth, after x55's results commit).

* GATES (all hard; any failure -> TEXTURE): G_MD5 (e346's rig +
  metrics; e345/e335/x45/x52 metrics; x45's anneal_resume (the draws
  reference); THE x55 RIG + runs/x55/metrics.json (the shared session);
  the checkpoint anchors; the port-source rigs) + G_QUOTES (the
  inherited port lines) + G_NAMEFREE + G_SPLICE + G_BATTERY + G_POOL +
  G_POOLTWIN + G_STREAMIDENT_Z + G_STREAMIDENT_T (25 CPU steps vs x45's
  committed rows, 2e-6) + G_BASELITERALS (the base row's p_T/p_Z
  recomputed vs e346's committed literals, 2e-6 — the ledger's reference
  frame) + G_X55ROWS (x55's metrics runtime-read; the needed dual rows
  exist at {1,25,125,300} for k9/k12/k14; x55's own gates all PASS)
  + G_RULERCERT (my k16r host rows vs x45's committed anneal panels +
  x52's rider_s1, 2e-6 each) + G_DRAWS (k16r draws bit-equal to x45's
  committed anneal draw record, first 300 of the 900-record list; first_ix
  == e345's committed) + G_LEDGERMATH (the wiring probe: L recomputed
  from e346's committed literals inside the gate for the k0/k7 fate
  cells; logit identities checked) + G_RIGCONST (constants vs G1/g1c
  meta).

Outputs: runs/x56/{metrics.json (PROGRESSIVE), REPORT.md,
x56_slot_ledger.png}. No NOTES/THINKING/QUEUE/STATE edits (the dispatch;
the heartbeat folds). Commit + push per phase (birth -> run). Run:
cd lab && python x56_slot_ledger.py (X56_SMOKE=1 shakedown).
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

SMOKE = os.environ.get("X56_SMOKE") == "1"
CPU = torch.device("cpu")
# ZERO CUDA (the dispatch's mandate: the GPU lane is sibling x36's) —
# enforced at run end by asserting the CUDA context was never initialized.
NAME = "x56_smoke" if SMOKE else "x56"

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
CONS_STEPS = 8 if SMOKE else 300
CONS_SEED = G1.CONS_SEED                  # 10901, HELD
FT_LR = G1.FT_LR                          # 1e-3 const

FATE_STEPS = (25, 125, 300)
FULL_ROW_STEPS = (1, 25, 125, 300)

X50_GRAIN = sorted({1, 2}
                   | {s for s in range(5, 301, 5)}
                   | {s for s in range(112, 139, 2)})   # 74 rungs (x50's)

ALL16 = tuple(range(16))

# ---- the ledger's frozen constants ----------------------------------------
BASE_PT = 0.0023042955435812473           # e346's committed base p(T)
BASE_PZ = 1.3383959412749391e-05          # e346's committed base p(Z)
LOGIT_BASE_T = math.log(BASE_PT / (1 - BASE_PT))
LOGIT_BASE_Z = math.log(BASE_PZ / (1 - BASE_PZ))
TOL_LEDGER = 1.0                          # the conservation band (logits)
TOL_DIP = 0.5                             # the both-lose dip cut (logits)
OCC_CUT = 0.10                            # the symmetric occupancy cut

READ_BAR = 0.05                           # x40's aliveness bar (verbatim)
READ_TOL = 2e-6                           # the CPU read-determinism law

# ======================================================================
# THE ARMS
# ======================================================================
ARMS = [
    {"tag": "k16r", "family": "ruler", "k": 16,
     "start": "tav_subject",
     "slot_keys": tuple("t" for _ in ALL16), "rungs": X50_GRAIN},
]
NET0_CLASS = {
    "ruler": ("TAVIREN/install-end WARM/RULER-FORMING (all-16 T-windows "
              "— the ruler's own stream, CPU re-realization)"),
}

# ---- the md5 binds (frozen at birth; Rule 12) -----------------------------
# NOTE: the x55 binds (rig + metrics) are frozen at THIS cell's birth,
# which lands AFTER x55's results commit (the shared session's ordering).
MD5_BINDS = {
    "rig_x55": ("lab/x55_midband_ladder.py",
                "84ac54933bfb8b7c9b59d243d10d9733"),
    "x55_metrics": ("runs/x55/metrics.json",
                    "6301f47494170c2ac8a18c2db7c7af08"),
    "rig_e346": ("lab/e346_name_key_ladder.py",
                 "808f89c94347d3d9d194cc268555dc2f"),
    "e346_metrics": ("runs/e346/metrics.json",
                     "7c6dc4051d46ece196486c47b764d4fa"),
    "x45_metrics": ("runs/x45/metrics.json",
                    "1dfcfeb3840c9307dbbc48cd71859cc5"),
    "x45_anneal_resume": ("runs/x45/anneal_resume_x45.pt",
                          "6b956ad13b9c3e8cfb20a49ce6303cc4"),
    "x52_metrics": ("runs/x52/metrics.json",
                    "bde14b44664b597520c2ca366e49b789"),
    "e345_metrics": ("runs/e345/metrics.json",
                     "e052fa3e4e0165df3e6c561bec3f67fd"),
    "e335_metrics": ("runs/e335/metrics.json",
                     "5d92359bc9b2e943c01bf521487cf177"),
    "rig_e345": ("lab/e345_taviren_warm_walk.py",
                 "4b647ad2303b7aee85503db7dcba9801"),
    "rig_g1": ("lab/g1_anchored_ball.py",
               "f4b6997b6a66013ee25da67f2b4faf01"),
    "rig_g1b": ("lab/g1b_continuity.py",
                "66966621e162b6bd33cf0023b0aa485a"),
    "rig_e281": ("lab/e281_rehearsal_dose.py",
                 "00fbf931f19c42a4b48dbd86806f0d1f"),
    "rig_e043": ("lab/e043_install.py",
                 "f82806b369452b05b8d89ca6cebe70fa"),
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
     "e341's VARIED arm generator — THE MENU-ALIASING FACT (the ruler's "
     "own stream, this cell's k16 replay)"),
    ("lab/e345_taviren_warm_walk.py",
     "nw = p0[\"pool_z_x\"][ix]",
     "e345's canon draw line (the rig this cell inherits)"),
]

REGISTERED = {
    "question_verbatim": (
        "x56 — THE SLOT LEDGER (the currency-corrected slot-split, riding "
        "x55's arms): L(k) = the logit-lift sum p_host + p_guest at every "
        "rung (the METHODS-2-correct logit currency, not raw softmax); "
        "per-name mass trajectories."),
    "bars_verbatim": {
        "zero_sum_clean": ("ZERO-SUM-CLEAN (the logit ledger conserves — "
                           "one seat, hard)"),
        "sum_dips": ("SUM-DIPS (the ledger dips where both lose — "
                     "competition with an external sink)"),
        "capacity_two": ("CAPACITY-TWO (the ledger supports two "
                         "simultaneous occupants at some k)"),
    },
    "parity_verbatim": (
        "LAB LEAN: 'SUM-DIPS weakly (0.51-at-k7 vs 0.65-at-k0 already "
        "hints down)'; COUNTER: 'ZERO-SUM-CLEAN'."),
    "P_x56a": {
        "guess": ("CAPACITY-TWO (against BOTH the lean's SUM-DIPS and the "
                  "counter's ZERO-SUM-CLEAN)"),
        "grounds": (
            "WITH THE HONEST PRE-DISCHARGE DISCLOSURE: the committed "
            "e346 fate cells already decide the level clause unless a "
            "gate fails (delta +1.23..+4.86 across k1-k7 with both "
            "channels occupied at k4 s25 and k7 s25/s300) — this cell's "
            "LIVE content is the k16 corner, the mid-band fill, the "
            "dense dip texture, and the arithmetic-vs-real "
            "decomposition. (1) the currency correction flips the raw-p "
            "hint's sign (the guest seat starts 1.3e-5-empty with ~11 "
            "logits of headroom; the host's gain outweighs the guest's "
            "loss); (2) the guest's k4->k7 fall is ~75% real "
            "competition (only ~0.2 of 0.8 logits is denominator "
            "arithmetic); (3) the k16 corner: GUEST-DEAD expected (the "
            "unrehearsed name dies; the ZxT mirror's guest rose because "
            "it WAS the stream's key), L(16) possibly at-or-below the "
            "k0 budget — the across-dose collapse CO-REPORT."),
        "expected_shape": (
            "delta(k) arches: +1..+5 through k1-k7, the mid-band holding "
            "or rising, k16 falling back to -1.5..+2.5 as the guest seat "
            "collapses; both-lose dips recur at the s125-class patches "
            "(committed at k7: -2.08); the corner reads GUEST-DEAD."),
        "falsifier": ("any hard-gate failure -> TEXTURE (nothing "
                      "adjudicated; P-x56a UNSCORED)"),
        "scored": ("TRUE iff the verdict == CAPACITY-TWO (plain)"),
    },
    "registration": ("question + bars + the parity block + P-x56a + every "
                     "convention picked + frozen HERE at birth BEFORE "
                     "compute; this script committed at birth; adjudicate "
                     "against exactly this; no bar shopping."),
}

deviations: list[str] = [
    "THE CURRENCY CHOICE (disclosed at birth): logit-lift vs the e001 "
    "base row's channel literals — the dispatch's 'METHODS-2-correct "
    "logit currency' operationalized as the R83 critic's first-named "
    "re-read; METHODS 2's literal named currency (x25's displacement-"
    "push) is a heavier instrument, NOT run here; the denominator "
    "arithmetic is DECOMPOSED explicitly instead (the audit the "
    "currency claim needs).",
    "THE SAMPLING MAP (disclosed): adjudicating cells = the fate steps "
    "for k in {0,1,2,4,7,9,12,14,16}; k0/1/2/4/7 committed (e346), "
    "k9/12/14 runtime-read from x55 (the shared session), k16 = this "
    "cell's replay; dense trajectories co-reported (x55's arms + my "
    "k16r at x50's grain + k0's committed GPU traj as the disclosure "
    "class).",
    "THE PRE-DISCHARGE DISCLOSURE (the honest one): the verdict's "
    "CAPACITY-TWO clause is dischargeable on committed rows alone "
    "(k4 s25 / k7 s25 / k7 s300: both channels >= 0.10, delta >= +3); "
    "this cell's live information is the k16 corner, the mid-band "
    "fill, the dip census, and the decomposition; P-x56a registers the "
    "committed-arithmetic consequence and its miss-branch is "
    "effectively TEXTURE (disclosed, not hidden).",
    "THE k16 CORNER'S CLASS (disclosed): the corner rides a CPU "
    "re-realization of the ruler's stream certified by G_RULERCERT "
    "(host vs x45's committed panels, 2e-6 — the CPU-class "
    "reproduction x52 proved) + G_DRAWS (bit-equality to x45's "
    "committed draw record); the guest channel has never been read on "
    "this stream anywhere.",
    "Per-arm resume under runs/x56/arm_resume_k16r.pt (*.pt gitignored "
    "— regenerable deterministically); this cell writes ONLY "
    "lab/x56_* and runs/x56/*.",
    "SHARED SESSION (disclosed): this cell rides x55's arms (runtime-"
    "read, md5-bound); one desk session, two cells, two birth-first "
    "chains.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the heartbeat "
    "folds).",
    "Smoke mode (X56_SMOKE=1): 8-step replay, rungs {1,2,4,8}, every "
    "gate live at the smoke horizon (G_RULERCERT skipped at 25/125/300 "
    "— no committed 8-step panel; G_STREAMIDENT at the full 25 CPU "
    "steps); the adjudication SMOKE-stamped (nothing scored) — every "
    "code path exercised.",
    "THE SMOKE-CAUGHT WIRING REPAIR (pre-adjudication, disclosed): the "
    "first smoke pass crashed in run_arm.row_of — the instrument call "
    "dropped the base_flat64 argument (a transcription slip in the "
    "port; the instrument's signature is the inherited one). Repaired "
    "by computing e001_flat64 in run_arm and passing it (x55's exact "
    "pattern). No bar, arm, stream, gate criterion or registration "
    "byte touched — the parameter is unused by this cell's row form "
    "(no census rider), presentation-class only.",
    "THE SMOKE-CAUGHT LEDGERMATH KEY REPAIR (pre-adjudication, "
    "disclosed): the second smoke pass crashed inside G_LEDGERMATH — "
    "the wiring probe's table lookup used the probe name's 'k0' prefix "
    "where the table keys are bare ('0'). Repaired to strip the "
    "prefix. No bar, arm, stream, gate criterion or registration byte "
    "touched — the gate's arithmetic unchanged.",
]

metrics: dict = {
    "experiment": "x56_slot_ledger",
    "phase": ("THE SLOT LEDGER — L(k) = logit-lift(p_host) + logit-"
              "lift(p_guest) at every rung over the full dose range "
              "(committed k<=7 + x55's mid-band + the k16 corner "
              "replay): ZERO-SUM-CLEAN / SUM-DIPS / CAPACITY-TWO, with "
              "the per-name mass trajectories, the both-lose dip "
              "census, and the arithmetic-vs-real decomposition"),
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
        "e346 (THE LADDER TABLE this ledger spans + the base-row channel "
        "literals that fix the currency's reference frame + the "
        "slot-split observation whose noun this cell re-reads)",
        "R83 (THE CRITIC'S CURRENCY KILL + the named control this cell "
        "discharges: 're-express BOTH channels as logit-lift vs base'; "
        "THE IDEATOR'S CARD that minted the ledger)",
        "METHODS 2 (THE CURRENCY DOCTRINE — raw p/p-base lifts "
        "manufacture doublings; x16's denominator-intrusion lesson)",
        "x55 (THE MID-BAND ARMS this cell rides — k9/k12/k14 dual rows "
        "at x50's grain + the k7r window; the shared session)",
        "x45 (THE INSTRUMENT VERBATIM + the ruler panels = the k16 "
        "host's certification reference + the anneal draw record)",
        "x52 (THE DENSE RULER REPLAY that proved the CPU-class "
        "reproduction route this cell's k16r replay uses)",
        "e345 (THE k=0 ENDPOINT's dual-channel rows = the budget B(s) "
        "definition)",
        "e311 (the TAVIREN subject)",
    ],
    "whats_new": [
        "THE LOGIT LEDGER ITSELF: the dose-range's first currency-"
        "corrected conservation table — L(k,s) = lift_host + lift_guest "
        "vs the k=0 solo budget, across the full nine-point k-curve",
        "THE k16 GUEST READ (the 2x2's unmeasured corner, read for the "
        "first time): the ruler's stream with its guest channel "
        "captured at x50's grain, host-certified against x45's panels",
        "THE BOTH-LOSE DIP CENSUS at dense grain (the external-sink "
        "signature's width and depth — committed only as the single "
        "k7 s125 exemplar)",
        "THE ARITHMETIC-VS-REAL DECOMPOSITION (the R83 critic's "
        "discriminator, computed): how much of the guest's fall is "
        "denominator, how much is competition",
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


def birth_hash() -> str:
    try:
        h = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%h", "-1",
             "--", "lab/x56_slot_ledger.py"],
            cwd=str(REPO), capture_output=True, text=True,
            timeout=10).stdout.strip()
        return h or "untracked (pre-birth)"
    except Exception:                                       # noqa: BLE001
        return "unavailable"


def write_partial(note: str) -> None:
    metrics["status"] = f"PARTIAL: {note} ({common.now_iso()})"
    save_json(RD / "metrics.json", metrics)


def flat_params_cpu(net) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).cpu()
                      for p in net.parameters()]).clone()


def load_model_sd(path: Path) -> dict:
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    return {k: v.detach().clone() for k, v in sd.items()}


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def lift(p: float, logit_base: float) -> float:
    return logit(p) - logit_base


# ---- x45's per-context reader (ported VERBATIM) ---------------------------
@torch.no_grad()
def battery_percell(net, ids: torch.Tensor, zid: int, bs: int = 30
                    ) -> torch.Tensor:
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


def channel_summary(net, p0: dict, name_id: int) -> dict:
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


BATTERIES = (("g-12", "gm12_ids"), ("g0", "g0_ids"), ("g+12", "gp12_ids"))


def instrument(p0: dict, theta_sd: dict, tag: str, name_id: int,
               base_flat64: np.ndarray, dual_id: int | None = None
               ) -> dict:
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
    if dual_id is not None:
        row["dual"] = channel_summary(net, p0, dual_id)
    del net
    return row


def light_row(p0: dict, theta_sd: dict, tag: str, host_id: int,
              guest_id: int) -> dict:
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
# THE CONSENSUS TRAINING STEP (inherited verbatim)
# ======================================================================
def make_draw(p0: dict, slot_keys: tuple):
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
# P0 — THE BINDS + THE BANK + THE TWO POOLS + THE RECORDS
# ======================================================================
def phase_P0() -> dict:
    log("P0 — THE BINDS + the bank + the two pools + the committed "
        "records")

    binds = {}
    for key, (rel, bound) in MD5_BINDS.items():
        p = REPO / rel
        ok = p.exists() and md5of(p) == bound
        binds[key] = {"path": rel,
                      "md5": (md5of(p) if p.exists() else None),
                      "bound_md5": bound, "pass": bool(ok)}
    g_md5 = {"binds": binds,
             "claim": ("every committed record, anchor state, instrument "
                       "parent, port-source rig, THE e346 RIG, and THE "
                       "x55 RIG + METRICS (the shared session's arms) "
                       "md5-bound at run time (frozen at birth)"),
             "pass": bool(all(b["pass"] for b in binds.values()))}
    assert g_md5["pass"], \
        f"G_MD5 FAILED: {[k for k, b in binds.items() if not b['pass']]}"
    metrics["gates"]["G_MD5"] = g_md5
    log(f"G_MD5 PASS: {len(binds)} binds")

    quotes = []
    for src, frag, why in PORT_QUOTES:
        txt = (REPO / src).read_text(encoding="utf-8")
        quotes.append({"source": src, "fragment": frag, "why": why,
                       "verified_substring": bool(frag in txt)})
    g_q = {"quotes": quotes,
           "claim": ("every ported line is a verbatim substring of its "
                     "committed source rig (the inherited port lines)"),
           "pass": bool(all(q["verified_substring"] for q in quotes))}
    assert g_q["pass"], f"G_QUOTES FAILED: {g_q}"
    metrics["gates"]["G_QUOTES"] = g_q
    log(f"G_QUOTES PASS: {len(quotes)}/{len(quotes)} verbatim")

    corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    tid = stoi["T"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)

    name_ids_z = corpus.encode(G1.NAME)
    name_ids_t = corpus.encode("TAVIREN")
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

    anchor_full = torch.stack([train_ids[p - G1.PRE: p - G1.PRE + G1.BLOCK]
                               for p, _ in install_occ])
    val_text = "".join(itos[int(i)] for i in val_ids)
    r_eval_x, r_eval_y = G1.val_windows(val_ids, val_text, 60,
                                        G1.R_EVAL_SEED)

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

    pool_z_x, pool_z_mask = jittered_pool(name_ids_z)
    pool_t_x, pool_t_mask = jittered_pool(name_ids_t)
    cons_anchor = anchor_full[:16]
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
                  "columns of each window, masks bit-equal"),
        "pass": bool(rt == 300 and ct and masks_eq),
    }
    assert g_twin["pass"], f"G_POOLTWIN FAILED: {g_twin}"
    metrics["gates"]["G_POOLTWIN"] = g_twin
    log("P0: the bank + the two keyed pools rebuilt (G_POOL + G_POOLTWIN "
        "PASS)")

    # ---- the committed records (runtime-read; never retyped) ----------
    e346m = json.loads((REPO / "runs" / "e346" / "metrics.json")
                       .read_text(encoding="utf-8"))
    committed_e346 = {
        "ladder": e346m["adjudication"]["ladder"],
        "base_pt": e346m["arm_rows"]["base"]["read"],
        "base_pz": e346m["arm_rows"]["base"]["dual"]["means"]["g0"],
    }
    e345m = json.loads((REPO / "runs" / "e345" / "metrics.json")
                       .read_text(encoding="utf-8"))
    committed_e345 = {
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
    x55m = json.loads((REPO / "runs" / "x55" / "metrics.json")
                      .read_text(encoding="utf-8"))
    committed_x55 = {"arm_rows": x55m["arm_rows"],
                     "gates": {k: bool(g.get("pass"))
                               for k, g in x55m["gates"].items()},
                     "status": x55m.get("status")}
    # x45's committed anneal draws (the G_DRAWS reference: first 300)
    anneal = torch.load(REPO / "runs" / "x45" / "anneal_resume_x45.pt",
                        map_location="cpu", weights_only=False)
    committed_ruler_draws = anneal["draws"][:CONS_STEPS]
    del anneal
    log("P0: the committed records runtime-read (e346 ladder + base; "
        f"e345 traj; x45 panels; x52 rider; x55 rows "
        f"{sum(len(v) for v in committed_x55['arm_rows'].values())}; the "
        f"ruler draws {len(committed_ruler_draws)})")
    write_partial("P0 the binds + the bank + the pools + the records")
    return {
        "corpus": corpus, "stoi": stoi, "itos": itos, "zid": zid,
        "tid": tid, "train_ids": train_ids,
        "bat_ids": bat_ids, "g0_ids": bat_ids[0], "gm12_ids": bat_ids[-12],
        "gp12_ids": bat_ids[12], "r_eval_xy": (r_eval_x, r_eval_y),
        "pool_z_x": pool_z_x, "pool_z_mask": pool_z_mask,
        "pool_t_x": pool_t_x, "pool_t_mask": pool_t_mask,
        "cons_anchor": cons_anchor,
        "committed_e346": committed_e346,
        "committed_e345": committed_e345,
        "committed_x45": committed_x45,
        "committed_x52": committed_x52,
        "committed_x55": committed_x55,
        "committed_ruler_draws": committed_ruler_draws,
    }


# ======================================================================
# P1a — G_STREAMIDENT (inherited verbatim)
# ======================================================================
def compare_row(mine: dict, ref: dict) -> tuple[dict, bool]:
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
    log(f"P1a — G_STREAMIDENT x2 (inherited): all-z and all-t, "
        f"{probe_steps} CPU steps vs x45's committed rows")

    cellz, okz = probe_streamident(
        p0, tuple("z" for _ in ALL16), CKPT_DIR / "g1c_install_resume.pt",
        probe_steps, p0["zid"],
        p0["committed_x45"]["walk_rungs"]["walk_cons_s25"],
        "streamident_z")
    g_sz = {"probe": {"steps": probe_steps, "start": "g1c_install_resume.pt",
                      "seed": CONS_SEED, "key": "all-z"},
            "row": cellz,
            "claim": ("my draw path keyed all-z reproduces x45's "
                      "committed walk_cons_s25 row BIT-EXACTLY"),
            "pass": bool(okz)}
    assert g_sz["pass"], f"G_STREAMIDENT_Z FAILED: {g_sz}"
    metrics["gates"]["G_STREAMIDENT_Z"] = g_sz
    log("G_STREAMIDENT_Z PASS")

    cellt, okt = probe_streamident(
        p0, tuple("t" for _ in ALL16), CKPT_DIR / "e311_TAVINST_post.pt",
        probe_steps, p0["tid"],
        p0["committed_x45"]["anneal_panels"]["anneal_s25"],
        "streamident_t")
    g_st = {"probe": {"steps": probe_steps, "start": "e311_TAVINST_post.pt",
                      "seed": CONS_SEED, "key": "all-t (the ruler's own)"},
            "row": cellt,
            "claim": ("my draw path keyed all-t reproduces x45's "
                      "committed anneal_s25 panel row BIT-EXACTLY — the "
                      "k16 replay's own stream identity"),
            "pass": bool(okt)}
    assert g_st["pass"], f"G_STREAMIDENT_T FAILED: {g_st}"
    metrics["gates"]["G_STREAMIDENT_T"] = g_st
    log("G_STREAMIDENT_T PASS")
    write_partial("P1a G_STREAMIDENT_Z + G_STREAMIDENT_T")


# ======================================================================
# P1b — THE k16r REPLAY (the corner; CPU; per-arm resume)
# ======================================================================
def phase_P1b(p0: dict) -> dict:
    log(f"P1b — THE k16r REPLAY (the ruler's own stream, DUAL channel, "
        f"CPU desk, seed {CONS_SEED}, {CONS_STEPS} steps, "
        f"{len(X50_GRAIN)}-rung grain)")
    out = {}
    for arm in ARMS:
        out[arm["tag"]] = run_arm(p0, arm,
                                  CKPT_DIR / "e311_TAVINST_post.pt")
        write_partial(f"P1b arm {arm['tag']} finished "
                      f"({len(out[arm['tag']]['rows'])} rows)")
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
        log(f"  [{tag}] RESUMED at step {state['step']}/{CONS_STEPS}")

    if int(state.get("step", 0)) >= CONS_STEPS:
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
# P1c — G_RULERCERT + G_DRAWS + G_BASELITERALS + G_X55ROWS
# ======================================================================
def phase_P1c(p0: dict, arms: dict) -> None:
    log("P1c — G_RULERCERT + G_DRAWS + G_BASELITERALS + G_X55ROWS")

    # ---- G_RULERCERT (the k16r host vs x45's committed panels) ---------
    panels = p0["committed_x45"]["anneal_panels"]
    rider_s1 = p0["committed_x52"]["rider_rows"]["rider_s1"]["read"]
    k16r_rows = arms["k16r"]["rows"]
    cells = {}
    ok_all = True
    checks = {"1": rider_s1}
    if not SMOKE:
        for s in FATE_STEPS:
            checks[str(s)] = panels[f"anneal_s{s}"]["read"]
    for s, comp in checks.items():
        row = k16r_rows.get(int(s))
        mine = (row.get("read", row.get("host"))
                if row is not None else None)
        if mine is None:
            cells[f"s{s}"] = {"pass": False, "note": "missing row"}
            ok_all = False
            continue
        cells[f"s{s}"] = {"mine_cpu": mine, "committed": comp,
                          "abs_diff": abs(mine - comp), "tol": READ_TOL,
                          "pass": bool(abs(mine - comp) <= READ_TOL)}
        ok_all &= cells[f"s{s}"]["pass"]
        log(f"  [k16r] s{s} host CPU {mine:.6f} vs committed "
            f"{comp:.6f} (|d| {abs(mine - comp):.1e})")
    g_rc = {"cells": cells,
            "claim": ("my k16r replay's HOST channel reproduces x45's "
                      "committed panels (and x52's rider_s1) within 2e-6 "
                      "— the CPU-class reproduction x52 proved; the "
                      "GUEST column beside it is therefore the same "
                      "stream's never-read corner"),
            "pass": bool(ok_all)}
    assert g_rc["pass"], f"G_RULERCERT FAILED: {g_rc}"
    metrics["gates"]["G_RULERCERT"] = g_rc
    log("G_RULERCERT PASS")

    # ---- G_DRAWS ---------------------------------------------------------
    draws = arms["k16r"]["draws"]
    ruler = p0["committed_ruler_draws"]
    n_expected = CONS_STEPS
    shape_ok = (len(draws) == n_expected
                and all(len(d["ix"]) == 16 and len(d["aj"]) == 8
                        and len(d["rj"]) == 8 for d in draws))
    vs_ruler = all(d == ruler[i] for i, d in enumerate(draws)) \
        and (SMOKE or len(ruler) == n_expected)
    first_ix_ok = draws[0]["ix"] == p0["committed_e345"]["first_ix"]
    g_dr = {"n_draws": len(draws), "n_expected": n_expected,
            "shape_ok": bool(shape_ok),
            "vs_x45_anneal_record_bit_equal": bool(vs_ruler),
            "first_ix": draws[0]["ix"],
            "e345_committed_first_ix": p0["committed_e345"]["first_ix"],
            "claim": ("the k16r replay's draws are BIT-EQUAL to x45's "
                      "committed anneal draw record (the first 300 of "
                      "the 900-record list in the md5-bound resume), "
                      "first draw == e345's committed first_ix"),
            "pass": bool(shape_ok and vs_ruler and first_ix_ok)}
    assert g_dr["pass"], f"G_DRAWS FAILED: {g_dr}"
    metrics["gates"]["G_DRAWS"] = g_dr
    log("G_DRAWS PASS: bit-equal to x45's committed anneal record")

    # ---- G_BASELITERALS (the ledger's reference frame) -------------------
    e001_sd = load_model_sd(CKPT_DIR / "e001.pt")
    e001_flat64 = flat_params_cpu(G1.evl_load(e001_sd)).double().numpy()
    brow = instrument(p0, e001_sd, "base", p0["tid"], e001_flat64,
                      dual_id=p0["zid"])
    del e001_sd
    my_pt, my_pz = brow["read"], brow["dual"]["means"]["g0"]
    cells = {
        "base_pt": {"mine": my_pt, "frozen": BASE_PT,
                    "abs_diff": abs(my_pt - BASE_PT)},
        "base_pz": {"mine": my_pz, "frozen": BASE_PZ,
                    "abs_diff": abs(my_pz - BASE_PZ)},
    }
    for c in cells.values():
        c["pass"] = bool(c["abs_diff"] <= READ_TOL)
    g_bl = {"cells": cells,
            "logit_base_T": LOGIT_BASE_T, "logit_base_Z": LOGIT_BASE_Z,
            "claim": ("the ledger's reference frame recomputes to the "
                      "frozen base literals within 2e-6 (e346's "
                      "committed base row)"),
            "pass": bool(all(c["pass"] for c in cells.values()))}
    assert g_bl["pass"], f"G_BASELITERALS FAILED: {g_bl}"
    metrics["gates"]["G_BASELITERALS"] = g_bl
    metrics["anchor_rows"] = {"base": brow}
    log(f"G_BASELITERALS PASS (p_T {my_pt:.2e} / p_Z {my_pz:.2e})")

    # ---- G_X55ROWS (the shared session's riding precondition) ------------
    x55g = p0["committed_x55"]["gates"]
    x55_rows = p0["committed_x55"]["arm_rows"]
    need = {}
    ok_rows = True
    for tag in ("k9", "k12", "k14", "k7r"):
        need[tag] = {}
        for s in (1, 25, 125, 300):
            r = x55_rows.get(tag, {}).get(str(s))
            have = r is not None and (
                ("read" in r and "dual" in r) if s in (1, 25, 125, 300)
                else True)
            # dual channel needed on the fate rows + s1
            if r is not None and "dual" not in r and s in (1, 25, 125, 300):
                have = False
            need[tag][str(s)] = bool(have)
            ok_rows &= bool(have)
    g_x55 = {
        "x55_gates_all_pass": bool(all(x55g.values())),
        "x55_status": p0["committed_x55"]["status"],
        "needed_rows": need,
        "claim": ("x55's metrics runtime-read (md5-bound): every gate "
                  "PASS and the dual rows at {1,25,125,300} exist for "
                  "k9/k12/k14 (+ the k7r window) — the riding "
                  "precondition"),
        "pass": bool(all(x55g.values()) and ok_rows),
    }
    assert g_x55["pass"], f"G_X55ROWS FAILED: {g_x55}"
    metrics["gates"]["G_X55ROWS"] = g_x55
    log("G_X55ROWS PASS: x55's rows + gates read clean")
    write_partial("P1c G_RULERCERT + G_DRAWS + G_BASELITERALS + G_X55ROWS")


# ======================================================================
# P2 — THE LEDGER (assemble + adjudicate)
# ======================================================================
def read_of(row: dict) -> float | None:
    if row is None:
        return None
    return row.get("read", row.get("host"))


def guest_of(row: dict) -> float | None:
    if row is None:
        return None
    if "dual" in row:
        return row["dual"]["means"]["g0"]
    return row.get("guest")


def phase_P2(p0: dict, arms: dict) -> dict:
    log("P2 — THE LEDGER (assemble the nine-point k-curve; the budget; "
        "the deltas; the dip census; the decomposition; the corner)")

    lad346 = p0["committed_e346"]["ladder"]
    x55_rows = p0["committed_x55"]["arm_rows"]
    k16r_rows = arms["k16r"]["rows"]
    k_order = ["0", "1", "2", "4", "7", "9", "12", "14", "16"]

    # ---- the table --------------------------------------------------------
    table: dict = {}
    for k in ("0", "1", "2", "4", "7"):
        r = lad346[k]
        table[k] = {"provenance": r["provenance"], "committed": True,
                    "host": dict(r["reads"]), "guest": dict(r["guest"]),
                    "s1_host": r["s1"],
                    "s1_guest": (lad346["0"]["guest"].get("1"))}
    # k0's s1 guest: e345's committed tav_s1 dual (runtime-read)
    table["0"]["s1_guest"] = None          # not in the e346 ladder table
    for tag, k in (("k9", "9"), ("k12", "12"), ("k14", "14")):
        rows = x55_rows[tag]
        table[k] = {"provenance": f"x55's {tag} (runtime-read)",
                    "committed": False, "host": {}, "guest": {},
                    "s1_host": None, "s1_guest": None}
        for s in FATE_STEPS + (1,):
            r = rows.get(str(s))
            if r is None:
                continue
            h, g = read_of(r), guest_of(r)
            if s == 1:
                table[k]["s1_host"], table[k]["s1_guest"] = h, g
            else:
                table[k]["host"][str(s)] = h
                table[k]["guest"][str(s)] = g
    table["16"] = {"provenance": "MY k16r replay (the corner)",
                   "committed": False, "host": {}, "guest": {},
                   "s1_host": None, "s1_guest": None}
    for s in FATE_STEPS + (1,):
        r = k16r_rows.get(s)
        if r is None:
            continue
        h, g = read_of(r), guest_of(r)
        if s == 1:
            table["16"]["s1_host"], table["16"]["s1_guest"] = h, g
        else:
            table["16"]["host"][str(s)] = h
            table["16"]["guest"][str(s)] = g

    # ---- the ledger cells ---------------------------------------------------
    for k in k_order:
        for s in FATE_STEPS:
            h = table[k]["host"].get(str(s))
            g = table[k]["guest"].get(str(s))
            if h is None or g is None:
                continue
            table[k].setdefault("L", {})[str(s)] = \
                lift(h, LOGIT_BASE_T) + lift(g, LOGIT_BASE_Z)
            table[k].setdefault("lift_h", {})[str(s)] = \
                lift(h, LOGIT_BASE_T)
            table[k].setdefault("lift_g", {})[str(s)] = \
                lift(g, LOGIT_BASE_Z)
    B = {str(s): table["0"]["L"][str(s)] for s in FATE_STEPS}
    for k in k_order:
        for s in FATE_STEPS:
            if str(s) in table[k].get("L", {}):
                table[k].setdefault("delta", {})[str(s)] = \
                    table[k]["L"][str(s)] - B[str(s)]
                table[k].setdefault("both_occupied", {})[str(s)] = bool(
                    table[k]["host"][str(s)] >= OCC_CUT
                    and table[k]["guest"][str(s)] >= OCC_CUT)

    # ---- the verdict composite (frozen order) --------------------------------
    hard_ok = all(bool(g.get("pass")) for g in metrics["gates"].values())
    cap_cells = [f"k{k}s{s}" for k in k_order if k != "0"
                 for s in FATE_STEPS
                 if table[k].get("both_occupied", {}).get(str(s))
                 and table[k].get("delta", {}).get(str(s), 0) >= TOL_LEDGER]

    # the both-lose dip census (adjudicating grain: the fate trio, per arm)
    dips = []
    for k in k_order:
        seq = [s for s in FATE_STEPS
               if str(s) in table[k].get("L", {})]
        for s1_, s2_ in zip(seq, seq[1:]):
            a, b = str(s1_), str(s2_)
            if (table[k]["lift_h"][b] < table[k]["lift_h"][a]
                    and table[k]["lift_g"][b] < table[k]["lift_g"][a]
                    and table[k]["L"][b] - table[k]["L"][a] <= -TOL_DIP):
                dips.append({"k": k, "from": a, "to": b,
                             "dL": round(table[k]["L"][b]
                                         - table[k]["L"][a], 3)})

    if not hard_ok:
        verdict, clause = "TEXTURE", (
            "hard-gate failure: "
            + ", ".join(k for k, g in metrics["gates"].items()
                        if not g.get("pass"))
            + " — nothing adjudicated; the ledger table verbatim")
    elif SMOKE:
        verdict, clause = "SMOKE", "nothing adjudicated (smoke stamp)"
    elif cap_cells:
        verdict = "CAPACITY-TWO"
        worst = max((table[k.split('k')[1][:0] or k[1:]] for k in []),
                    default=None)          # placeholder, unused
        best = max(((k, s, table[k]["delta"][s])
                    for k in k_order if k != "0"
                    for s in table[k].get("delta", {})),
                   key=lambda t: t[2], default=None)
        clause = (
            f"{len(cap_cells)} adjudicating cell(s) with BOTH channels "
            f"occupied (>= {OCC_CUT}) and delta >= +{TOL_LEDGER} logits "
            f"(best: k={best[0]} s{best[1]} delta {best[2]:+.2f}) — the "
            "logit ledger supports two simultaneous occupants: the seat "
            "is not zero-sum at partial dose; the raw-p slot-split noun "
            "re-words to A GROWING SEAT, and the anti-motion is real "
            "but non-conserving")
    elif dips:
        verdict = "SUM-DIPS"
        clause = (f"{len(dips)} qualifying both-lose dip(s) "
                  f"({'; '.join(d['k'] + ' s' + d['from'] + '->s' + d['to'] + ' ' + str(d['dL']) for d in dips)}) "
                  "with no capacity cell — the ledger dips where both "
                  "lose: competition with an external sink")
    elif all(abs(v) <= TOL_LEDGER
             for k in k_order if k != "0"
             for v in table[k].get("delta", {}).values()):
        verdict = "ZERO-SUM-CLEAN"
        clause = (f"|delta| <= {TOL_LEDGER} at every adjudicating cell "
                  "with no qualifying dip — the logit ledger conserves: "
                  "one seat, hard")
    else:
        verdict = "MIXED"
        clause = ("no capacity cell and no qualifying dip, but the "
                  "ledger moves beyond the conservation band — named "
                  "honestly; the ledger table is the report")

    # ---- the k16 corner (the free rider) --------------------------------------
    corner = {
        "host": dict(table["16"]["host"]),
        "guest": dict(table["16"]["guest"]),
        "L": dict(table["16"].get("L", {})),
        "delta_vs_k0": dict(table["16"].get("delta", {})),
        "guest_verdict": None, "guest_clause": "",
    }
    gvals = [v for v in corner["guest"].values() if v is not None]
    if gvals and all(v <= OCC_CUT for v in gvals):
        corner["guest_verdict"] = "GUEST-DEAD"
        corner["guest_clause"] = (
            "the unrehearsed name dies at the full dose (every fate "
            "guest <= 0.10) — the stream's key holds the seat alone; "
            "the ZxT mirror's asymmetry confirmed on this substrate")
    elif gvals and any(v >= OCC_CUT for v in gvals):
        corner["guest_verdict"] = "GUEST-HELD"
        corner["guest_clause"] = (
            "the guest survives at the full dose at >= 1 fate rung — "
            "the seat carries residue occupancy with zero rehearsal "
            "(the registered surprise)")
    # L(16) vs budget narrative
    d16 = corner["delta_vs_k0"]
    if d16:
        corner["ledger_clause"] = (
            f"L(16) vs the k0 budget: "
            + ", ".join(f"s{s} {v:+.2f}" for s, v in d16.items())
            + (" — the ledger COLLAPSES BELOW the solo budget at "
               + ", ".join(s for s, v in d16.items() if v <= -TOL_LEDGER)
               + " (the across-dose collapse: the host alone wins but "
                 "the Z-seat's fall outweighs the T-seat's gain in "
                 "logit currency — co-report, not the verdict: 'both "
                 "lose' does not describe it)"
               if any(v <= -TOL_LEDGER for v in d16.values()) else
               " — within or above the band"))

    # ---- the arithmetic-vs-real decomposition ----------------------------------
    decomp = []
    for k1, k2 in zip(k_order[1:-1], k_order[2:]):
        for s in FATE_STEPS:
            h1 = table[k1]["host"].get(str(s))
            h2 = table[k2]["host"].get(str(s))
            g1 = table[k1]["guest"].get(str(s))
            g2 = table[k2]["guest"].get(str(s))
            if None in (h1, h2, g1, g2):
                continue
            observed = lift(g2, LOGIT_BASE_Z) - lift(g1, LOGIT_BASE_Z)
            g2_pred = g1 * (1 - h2) / (1 - h1)      # denominator-only
            predicted = lift(g2_pred, LOGIT_BASE_Z) - lift(g1,
                                                           LOGIT_BASE_Z)
            decomp.append({
                "pair": f"k{k1}->k{k2}", "rung": str(s),
                "observed_dlift_g": round(observed, 3),
                "arithmetic_share": round(predicted, 3),
                "real_share": round(observed - predicted, 3)})
    n_real = sum(1 for d in decomp if d["real_share"] <= -TOL_DIP)

    # ---- the dense trajectories (co-report) --------------------------------------
    dense = {}
    for tag in ("k9", "k12", "k14", "k7r"):
        rows = x55_rows.get(tag, {})
        dense[tag] = {"host": {}, "guest": {}, "L": {}}
        for s, r in sorted(rows.items(), key=lambda kv: int(kv[0])):
            h, g = read_of(r), guest_of(r)
            if h is not None and g is not None:
                dense[tag]["host"][s] = h
                dense[tag]["guest"][s] = g
                dense[tag]["L"][s] = lift(h, LOGIT_BASE_T) \
                    + lift(g, LOGIT_BASE_Z)
    dense["k16r"] = {"host": {}, "guest": {}, "L": {}}
    for s, r in sorted(k16r_rows.items()):
        h, g = read_of(r), guest_of(r)
        if h is not None and g is not None:
            dense["k16r"]["host"][str(s)] = h
            dense["k16r"]["guest"][str(s)] = g
            dense["k16r"]["L"][str(s)] = lift(h, LOGIT_BASE_T) \
                + lift(g, LOGIT_BASE_Z)
    # k0's dense ledger from e345's committed GPU traj (disclosure class)
    dense["k0_gpu_traj_disclosure"] = [
        {"step": t["step"],
         "L": lift(t["g0_pt"], LOGIT_BASE_T) + lift(t["g0_pz"],
                                                    LOGIT_BASE_Z)}
        for t in p0["committed_e345"]["traj"]]
    # the dense dip census (co-report)
    dense_dips = []
    for tag in ("k9", "k12", "k14", "k16r"):
        Ls = dense[tag]["L"]
        seq = sorted((int(s), v) for s, v in Ls.items())
        for (s1_, l1), (s2_, l2) in zip(seq, seq[1:]):
            h1 = dense[tag]["host"][str(s1_)]
            h2 = dense[tag]["host"][str(s2_)]
            g1 = dense[tag]["guest"][str(s1_)]
            g2 = dense[tag]["guest"][str(s2_)]
            if (lift(h2, LOGIT_BASE_T) < lift(h1, LOGIT_BASE_T)
                    and lift(g2, LOGIT_BASE_Z) < lift(g1, LOGIT_BASE_Z)
                    and l2 - l1 <= -TOL_DIP):
                dense_dips.append({"arm": tag, "from": s1_, "to": s2_,
                                   "dL": round(l2 - l1, 3)})

    # ---- G_LEDGERMATH (the wiring probe, run on the assembled table) ---------
    k0s300 = table["0"]["host"].get("300"), table["0"]["guest"].get("300")
    k7s300 = table["7"]["host"].get("300"), table["7"]["guest"].get("300")
    lit_L = {}
    ok_math = abs(logit(0.5)) < 1e-12          # the identity probe
    for nm, (h, g) in (("k0_s300", k0s300), ("k7_s300", k7s300)):
        if h is not None and g is not None:
            v = lift(h, LOGIT_BASE_T) + lift(g, LOGIT_BASE_Z)
            lit_L[nm] = round(v, 6)
            ok_math &= abs(v - table[nm.split("_")[0][1:]]["L"]["300"]) \
                < 1e-9
    g_lm = {"logit_identity": "logit(0.5) == 0",
            "recomputed": lit_L,
            "claim": ("the ledger arithmetic re-derives the table's L "
                      "from the committed literals (the wiring probe)"),
            "pass": bool(ok_math and len(lit_L) == 2)}
    assert g_lm["pass"], f"G_LEDGERMATH FAILED: {g_lm}"
    metrics["gates"]["G_LEDGERMATH"] = g_lm
    log("G_LEDGERMATH PASS")

    # ---- G_RIGCONST -----------------------------------------------------------
    g1c_meta = torch.load(CKPT_DIR / "g1c_root.pt", map_location="cpu",
                          weights_only=False)["meta"]
    g_rig = {
        "ft_lr": FT_LR, "betas": "(0.9, 0.95)", "weight_decay": 0.1,
        "clip": 1.0, "steps": CONS_STEPS, "cons_seed": CONS_SEED,
        "g1_cons_seed": G1.CONS_SEED, "g1_ft_lr": G1.FT_LR,
        "g1c_meta_install_seed": g1c_meta.get("install_seed"),
        "g1c_meta_install_steps": g1c_meta.get("install_steps"),
        "ledger_consts": {"tol_ledger": TOL_LEDGER, "tol_dip": TOL_DIP,
                          "occ_cut": OCC_CUT,
                          "logit_base_T": LOGIT_BASE_T,
                          "logit_base_Z": LOGIT_BASE_Z},
        "pass": bool(FT_LR == 1e-3 and CONS_SEED == 10901
                     and g1c_meta.get("install_seed") == 24314
                     and g1c_meta.get("install_steps") == 400),
    }
    assert g_rig["pass"], f"G_RIGCONST FAILED: {g_rig}"
    metrics["gates"]["G_RIGCONST"] = g_rig
    log("G_RIGCONST PASS")

    # ---- P-x56a scoring ----------------------------------------------------------
    if verdict == "TEXTURE" or SMOKE:
        p_hit = {"verdict": None,
                 "note": "TEXTURE/SMOKE — P-x56a UNSCORED per the "
                         "registered falsifier clause"}
    else:
        p_hit = {"verdict": bool(verdict == "CAPACITY-TWO"),
                 "guess": REGISTERED["P_x56a"]["guess"]}

    adj = {
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "currency": {"logit": "ln(p/(1-p))",
                     "lift": "logit(p) - logit(p_base)",
                     "base": {"p_T": BASE_PT, "p_Z": BASE_PZ},
                     "L": "lift_host + lift_guest",
                     "budget": "B(s) = L(0,s) (the k=0 committed ledger)",
                     "delta": "L(k,s) - B(s)",
                     "tol_ledger": TOL_LEDGER, "tol_dip": TOL_DIP,
                     "occ_cut": OCC_CUT},
        "table": table, "k_order": k_order, "budget": B,
        "verdict": verdict, "verdict_clause": clause,
        "composite_order": ("TEXTURE -> CAPACITY-TWO (some cell: both "
                            "occupied AND delta >= +1.0) -> SUM-DIPS (a "
                            "qualifying both-lose dip, no capacity cell) "
                            "-> ZERO-SUM-CLEAN (|delta| <= 1.0 "
                            "everywhere, no dip) -> MIXED"),
        "capacity_cells": cap_cells,
        "dip_census_fate": dips,
        "dip_census_dense": dense_dips,
        "corner_k16": corner,
        "decomposition": decomp, "n_real_competition_pairs": n_real,
        "dense_coreport": dense,
        "P_x56a": {"guess": REGISTERED["P_x56a"]["guess"],
                   "hit": p_hit, "scored": REGISTERED["P_x56a"]["scored"]},
    }
    metrics["adjudication"] = adj
    write_partial("P2 the ledger adjudicated")
    log(f"P2: verdict {verdict} / corner {corner.get('guest_verdict')} / "
        f"{len(cap_cells)} capacity cells / {len(dips)} fate dips")
    return adj


# ======================================================================
# P3 — THE OUTPUTS (the PNG + the REPORT)
# ======================================================================
def make_png(adj: dict) -> None:
    table = adj["table"]
    k_order = adj["k_order"]
    kx = {k: int(k) for k in k_order}
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 3)

    # ---- panel 1: THE LEDGER vs k ----------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    for s, col in (("25", "tab:blue"), ("125", "tab:olive"),
                   ("300", "tab:cyan")):
        xs, ys = [], []
        for k in k_order:
            v = table[k].get("delta", {}).get(s)
            if v is not None:
                xs.append(kx[k])
                ys.append(v)
        ax1.plot(xs, ys, "o-", color=col, label=f"delta(k, s{s})")
    ax1.axhline(0, color="black", lw=1.2, label="the k=0 budget")
    ax1.axhline(adj["currency"]["tol_ledger"], color="tab:red", ls="--",
                lw=1, label="+/- the 1.0-logit band")
    ax1.axhline(-adj["currency"]["tol_ledger"], color="tab:red", ls="--",
                lw=1)
    ax1.set_xlabel("k")
    ax1.set_ylabel("delta = L(k,s) - L(0,s)  (logits)")
    ax1.set_title(f"THE SLOT LEDGER — {adj['verdict']}\n"
                  "(the logit-lift sum vs the guest-solo budget)")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.25)

    # ---- panel 2: the dense L trajectories --------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    for tag, col in (("k9", "tab:green"), ("k12", "tab:orange"),
                     ("k14", "tab:purple"), ("k16r", "tab:brown")):
        Ls = adj["dense_coreport"][tag]["L"]
        pts = sorted((int(s), v) for s, v in Ls.items())
        if pts:
            ax2.plot([a for a, _ in pts], [b for _, b in pts], "-",
                     lw=1.0, color=col, label=f"{tag}")
    tr0 = adj["dense_coreport"]["k0_gpu_traj_disclosure"]
    ax2.plot([t["step"] for t in tr0], [t["L"] for t in tr0], "-",
             color="tab:red", lw=0.8, alpha=0.7,
             label="k=0 (GPU traj, disclosure)")
    for s in ("25", "125", "300"):
        v = adj["budget"].get(s)
        ax2.plot(int(s), v, "s", color="black", ms=7, zorder=3)
    ax2.set_xlabel("stream step")
    ax2.set_ylabel("L = lift_h + lift_g (logits)")
    ax2.set_title("the dense ledger trajectories (squares = the k0 "
                  "budget cells)")
    ax2.legend(fontsize=6)
    ax2.grid(alpha=0.25)

    # ---- panel 3: per-name mass trajectories --------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    for tag, col in (("k9", "tab:green"), ("k12", "tab:orange"),
                     ("k14", "tab:purple"), ("k16r", "tab:brown")):
        H = adj["dense_coreport"][tag]["host"]
        G = adj["dense_corepost"] if False else adj["dense_coreport"][tag]["guest"]
        ptsH = sorted((int(s), v) for s, v in H.items())
        ptsG = sorted((int(s), v) for s, v in G.items())
        ax3.plot([a for a, _ in ptsH], [b for _, b in ptsH], "-", lw=1.0,
                 color=col, label=f"{tag} p(T) host")
        ax3.plot([a for a, _ in ptsG], [b for _, b in ptsG], "--", lw=1.0,
                 color=col, alpha=0.8)
    ax3.set_xlabel("stream step")
    ax3.set_ylabel("raw softmax mass (solid host / dashed guest)")
    ax3.set_title("the per-name mass trajectories (co-report)")
    ax3.legend(fontsize=6)
    ax3.grid(alpha=0.25)

    # ---- panel 4: the trade (lift_h vs lift_g) ------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    for k in k_order:
        col = "tab:red" if table[k]["committed"] else "tab:green"
        xs = [table[k]["lift_h"][s] for s in ("25", "125", "300")
              if s in table[k].get("lift_h", {})]
        ys = [table[k]["lift_g"][s] for s in ("25", "125", "300")
              if s in table[k].get("lift_g", {})]
        if xs:
            ax4.plot(xs, ys, "o", color=col, ms=8, label=f"k={k}")
    ax4.set_xlabel("lift_host (logits)")
    ax4.set_ylabel("lift_guest (logits)")
    ax4.set_title("the trade at the fate cells (anti-motion without "
                  "conservation)")
    ax4.legend(fontsize=6)
    ax4.grid(alpha=0.25)

    # ---- panel 5: the decomposition bars ------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    if adj["decomposition"]:
        labels = [d["pair"] + " s" + d["rung"]
                  for d in adj["decomposition"]]
        arith = [d["arithmetic_share"] for d in adj["decomposition"]]
        real = [d["real_share"] for d in adj["decomposition"]]
        x = np.arange(len(labels))
        ax5.bar(x - 0.18, arith, 0.36, label="arithmetic (denominator)",
                color="tab:gray")
        ax5.bar(x + 0.18, real, 0.36, label="real (competition)",
                color="tab:red", alpha=0.8)
        ax5.set_xticks(x, labels, fontsize=5, rotation=90)
        ax5.axhline(0, color="black", lw=1)
        ax5.set_ylabel("guest dlift (logits)")
        ax5.set_title("the arithmetic-vs-real decomposition")
        ax5.legend(fontsize=7)
        ax5.grid(alpha=0.25, axis="y")

    # ---- panel 6: the verdict table -------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    c = adj["corner_k16"]
    tbl = [["observable", "verdict"]]
    tbl.append(["THE LEDGER", adj["verdict"]])
    tbl.append(["the k16 corner (free rider)",
                str(c.get("guest_verdict"))])
    tbl.append(["capacity cells", str(len(adj["capacity_cells"]))])
    tbl.append(["both-lose dips (fate/dense)",
                f"{len(adj['dip_census_fate'])}/"
                f"{len(adj['dip_census_dense'])}"])
    tbl.append(["real-competition pairs",
                str(adj["n_real_competition_pairs"])])
    tbl.append(["P-x56a", str(adj["P_x56a"]["hit"])])
    ax6.table(cellText=tbl[1:], colLabels=tbl[0], loc="center",
              cellLoc="left")
    ax6.set_title("the verdicts against the frozen bars", fontsize=9)

    fig.suptitle("X56 THE SLOT LEDGER — L(k) = logit-lift(p_host) + "
                 "logit-lift(p_guest) over the full dose range; the k16 "
                 f"corner read; verdict {adj['verdict']}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = RD / "x56_slot_ledger.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    log(f"[png] {out.name} written")


def write_report(adj: dict) -> None:
    L: list[str] = []
    A = L.append
    table = adj["table"]
    k_order = adj["k_order"]

    def fmt(v, p=3):
        return "-" if v is None else (
            f"{v:.2e}" if v is not None and 0 < abs(v) < 0.01
            else f"{v:.{p}f}")

    A("# X56 — THE SLOT LEDGER (the currency-corrected slot-split)")
    A("")
    A(f"* LEDGER VERDICT: **{adj['verdict']}** — {adj['verdict_clause']}")
    A(f"* P-x56a (my registered read): {adj['P_x56a']['hit']} "
      f"(guess: {adj['P_x56a']['guess']})")
    c = adj["corner_k16"]
    A(f"* THE k16 CORNER (the free rider): **{c.get('guest_verdict')}** — "
      f"{c.get('guest_clause', '')} {c.get('ledger_clause', '')}")
    A(f"* guest at the corner: "
      + ", ".join(f"s{s} {fmt(v)}" for s, v in c["guest"].items()))
    n_pass = sum(1 for g in metrics["gates"].values() if g.get("pass"))
    n_all = len(metrics["gates"])
    A(f"* gates: {n_pass}/{n_all} PASS"
      + ("" if n_pass == n_all else " — FAILURES: "
         + ", ".join(k for k, g in metrics["gates"].items()
                     if not g.get("pass"))))
    A("")
    A("## THE LEDGER TABLE (currency: logit-lift vs the e001 base "
      "channels; budget B(s) = L(0,s); delta = L - B; occupancy >= "
      f"{OCC_CUT}; bands +/-{adj['currency']['tol_ledger']})")
    A("")
    A("| k | p_T s25 | p_Z s25 | L s25 | d s25 | p_T s125 | p_Z s125 "
      "| L s125 | d s125 | p_T s300 | p_Z s300 | L s300 | d s300 | "
      "both-occupied |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in k_order:
        cells = []
        both = []
        for s in ("25", "125", "300"):
            h = table[k]["host"].get(s)
            g = table[k]["guest"].get(s)
            Lv = table[k].get("L", {}).get(s)
            d = table[k].get("delta", {}).get(s)
            cells += [fmt(h), fmt(g), fmt(Lv), fmt(d, 2)]
            both.append(table[k].get("both_occupied", {}).get(s))
        A(f"| {k} | " + " | ".join(cells) + " | "
          + ("/".join("Y" if b else "." for b in both)) + " |")
    A("")
    A(f"* the composite order (frozen): {adj['composite_order']}")
    A(f"* capacity cells: {adj['capacity_cells']}")
    A("")
    A("## THE BOTH-LOSE DIP CENSUS (the external-sink signature)")
    A("")
    A(f"* fate-grain dips (TOL_DIP {adj['currency']['tol_dip']}): "
      + (", ".join(f"k{d['k']} s{d['from']}->s{d['to']} "
                   f"{d['dL']:+.2f}"
                   for d in adj["dip_census_fate"]) or "NONE"))
    A(f"* dense-grain dips (x55's arms + my k16r): "
      + (", ".join(f"{d['arm']} s{d['from']}->s{d['to']} "
                   f"{d['dL']:+.2f}"
                   for d in adj["dip_census_dense"]) or "NONE"))
    A("")
    A("## THE ARITHMETIC-VS-REAL DECOMPOSITION (the guest's falls)")
    A("")
    A("| pair | rung | observed dlift_g | arithmetic share | real share |")
    A("|---|---|---|---|---|")
    for d in adj["decomposition"]:
        A(f"| {d['pair']} | s{d['rung']} | {d['observed_dlift_g']:+.3f} "
          f"| {d['arithmetic_share']:+.3f} | {d['real_share']:+.3f} |")
    A(f"* real-competition pairs (real <= -{adj['currency']['tol_dip']}): "
      f"{adj['n_real_competition_pairs']} of {len(adj['decomposition'])}")
    A("")
    A("## Provenance")
    A(f"* birth commit: {metrics.get('birth_commit')}; final head: "
      f"{metrics.get('git_head_final')}")
    A("* CPU desk ONLY (threads 4, zero CUDA — the GPU lane is sibling "
      "x36's; runtime asserted); timestamps UTC only; every artifact "
      "md5-bound (incl. runs/x55/metrics.json — the shared session); "
      "the k16r replay certified vs x45's committed panels (2e-6) with "
      "draws bit-equal to x45's anneal record")
    A("* catches + disclosures: see metrics.deviations")
    A("")
    A("*No NOTES/THINKING/QUEUE/STATE edits (dispatch; the heartbeat "
      "folds).*")
    (RD / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    log("[report] REPORT.md written")


# ======================================================================
# MAIN
# ======================================================================
def main() -> None:
    log(f"X56 — THE SLOT LEDGER (smoke={SMOKE}) -> {RD}")
    metrics["birth_commit"] = birth_hash()
    assert torch.get_num_threads() == 4
    write_partial("startup (bars + parity + P-x56a registered, committed "
                  "at birth)")
    set_seed(56500)             # global init only; every RNG is its own

    p0 = phase_P0()
    phase_P1a(p0)                     # G_STREAMIDENT_Z + _T (CPU)
    arms = phase_P1b(p0)              # THE k16r REPLAY (CPU desk)
    phase_P1c(p0, arms)               # G_RULERCERT + G_DRAWS + ...
    adj = phase_P2(p0, arms)          # THE LEDGER

    gates_pass = all(bool(g.get("pass")) for g in metrics["gates"].values())
    if not gates_pass and adj["verdict"] != "TEXTURE":
        adj["verdict"] = "TEXTURE"
    metrics["status"] = (
        "SMOKE: nothing adjudicated" if SMOKE else
        f"ADJUDICATED: {adj['verdict']} (corner "
        f"{adj['corner_k16'].get('guest_verdict')})")
    make_png(adj)
    metrics["git_head_final"] = git_head()
    metrics["date_finished"] = common.now_iso()
    save_json(RD / "metrics.json", metrics)
    write_report(adj)
    log(f"DONE — verdict {adj['verdict']} / corner "
        f"{adj['corner_k16'].get('guest_verdict')} "
        f"(gates {sum(1 for g in metrics['gates'].values() if g.get('pass'))}/"
        f"{len(metrics['gates'])})")
    assert not torch.cuda.is_initialized(), \
        "ZERO-CUDA violation: the CUDA context was initialized"


if __name__ == "__main__":
    main()
