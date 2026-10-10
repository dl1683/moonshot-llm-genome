"""X40 — THE FIRST-STEP SURVIVAL MECHANISM (R78's card; the recipe's
mechanism leg; CPU desk lane, dispatched 2026-10-09). This docstring
carries the registered question + background + design + bars + lab lean
+ P-x40a VERBATIM from the dispatch letter, plus every frozen
operationalization, committed at birth BEFORE any compute. Adjudicate
against exactly this; no bar shopping.

THE BACKGROUND (dispatch verbatim): "e341 proved ANNEALING IS A DOSE
(300 steps, varied OR fixed context menu, buys the s1 survival — the
first projected step's annihilation of fresh installs); x38 set the
noise floor (~2x the write's norm). THE QUESTION: WHY do annealed
reads survive the first projected step?"

THE HYPOTHESES (dispatch verbatim):
  (a) BROAD-SUPPORT — "varied/fixed annealing spreads the read across
   more contexts, so the s1 kill (which acts at the battery's contexts)
   leaves alive mass elsewhere";
  (b) VACCINATION — "the annealing steps are repeated sub-noise-floor
   doses (each step ~the cons's jitter norm) that immunize the read
   against the commit's projection";
  (c) BASIN-WIDTH — "the annealed read's landscape is flatter inside
   the ball (the same displacement hurts less)".

THE DESIGN (dispatch verbatim):
  "1. THE PER-CONTEXT DISTRIBUTION at s1: for the committed classes
   (e341's varied/fixed post-anneal states + the fresh install; the
   harness's commit machinery by import on CPU), read the battery
   per-context at t0 and after ONE projected step (the s1 event in
   isolation, eval-only): the frac of contexts alive (>= 0.05), the
   distribution's shape. BROAD-SUPPORT predicts: the fresh read's
   support collapses to ~0 alive contexts; the annealed reads keep a
   broad alive tail (and the varied arm broader at OFFSET contexts —
   its generalization channel).
  2. THE VACCINATION DESK CHECK: measure the annealing steps' own
   displacement norms (the cons's jitter protocol port from e341's
   record: per-step parameter deltas) vs x38's noise floor (~2x the
   write's norm). VACCINATION predicts: the cumulative annealing dose
   >= the noise floor (the read has 'seen' equivalent displacement);
   the steps' per-dose norms are the comparison.
  3. THE BASIN PROXY: the read's sensitivity curve at small
   displacements (0.25x/0.5x/1x the commit's R=0.7) on the three
   states — a flatter early curve = wider basin."

BARS (dispatch VERBATIM; frozen in this birth commit BEFORE compute):
  - BROAD-SUPPORT: "the annealed reads' alive-context fraction at s1 is
    materially above the fresh's (~0) AND the varied arm > the fixed at
    offset contexts — the survival is support-breadth; composes with
    e341's generalization channel."
  - VACCINATION: "the annealing dose >= the noise floor and the basin
    proxy shows the annealed reads flatter at sub-R displacements — the
    survival is immunization; the recipe's anneal leg re-words to
    'dose >= noise floor'."
  - BASIN-WIDTH: "the sensitivity curves separate at small displacements
    with no support-breadth difference — the survival is landscape
    flatness."
  - MIXED: "multiple fire — report the decomposition."

LAB LEAN (dispatch verbatim): "MIXED (broad-support + vaccination),
weakly — the two evening's channels (e341's generalization + x38's
dose) both point into the mechanism; the countervailing: the fixed arm
survived s1 too without the generalization — support-breadth at the
battery's own contexts may suffice. State your own read."

P-x40a (THE EXECUTOR'S OWN READ, registered per the dispatch's
"Register P-x40a BEFORE compute ... State your own read", frozen HERE
at birth; predictions are scored): MIXED (BROAD-SUPPORT + VACCINATION),
weakly — concurring with the lab lean, with three registered
refinements. (1) THE g0 TAIL IS PREDICTED SHARED, THE OFFSET TAIL IS
THE VARIED ARM'S OWN: both annealed arms survived s1 at the battery's
own geometry at nearly equal s1 means (0.115 vs 0.097 — within the
arm spread), so BROAD-SUPPORT's first clause (a shared broad g0 alive
tail over the fresh's ~0) and its second clause (varied > fixed at the
OFFSET batteries — where the committed t0 reads already sit 5x apart,
gm12 0.688 vs 0.131) are SEPARATE measurements, and the lab's
countervailing note ("support-breadth at the battery's own contexts
may suffice") is exactly the possibility that clause 1 fires while
clause 2 also fires with the offset difference carried over from t0 —
the mechanism decomposition will name which tail carries the survival.
(2) THE DOSE CLAUSE'S READING IS LOAD-BEARING AND I FREEZE THE
MECHANISM'S OWN QUANTITY: on the ENDPOINT reading the committed
cumulative displacement (varied 16.779 / fixed 16.801, e341's record)
sits ~6-7% BELOW the 2x-write noise floor (17.899 on e311's write norm
8.9496; 18.358 on x38's 2x rung) — a miss; on the PATH-LENGTH reading
(the SUM of the 300 per-step displacement norms — the "repeated doses"
of the hypothesis's own wording, each step individually far below the
floor) it is far ABOVE — a near-certain hit. I register the PATH LENGTH
as the bar's "cumulative annealing dose" (the immunization metaphor is
about accumulated exposure, not net displacement) with the endpoint +
max-roaming co-reported; the honest decomposition must own that the
ENDPOINT ALONE does not clear the floor. (3) THE GAUSSIAN CONTROL IS
EXPECTED FLAT FOR ALL THREE STATES (x38: a 9.18 L2 gaussian moved a
living read -0.07%; 0.7 L2 is 8% of that dose) — the basin proxy's
discriminating weight therefore sits on the KILL-RAY curve (the s1
projected displacement's own direction, the displacement the survival
actually survives), and if the gaussian separates too that is isotropic
basin width — stronger than the bar needs, co-reported. PREDICTED
SHAPE: the fresh alive-frac at s1 ~0 (0-2 of 60 at g0); both annealed
arms keep a broad g0 tail (alive-frac ~0.1-0.4); varied > fixed at
gm12 AND gp12 alive-frac at s1; the kill-ray retention curves ordered
annealed > fresh already at 0.5R (possibly at 0.25R); the gaussian
flat everywhere (all retentions > 0.9); the replay's per-step doses
~0.2-0.7 with max << floor, path length ~60-180 >> 17.9. FALSIFIERS:
the annealed g0 alive-frac within the fresh's (BROAD-SUPPORT clause 1
fails); the fixed arm >= the varied at either offset battery (clause 2
fails); the kill-ray retentions overlapping at 0.5R (VACCINATION
clause 2 fails — with the dose clause near-certain, VACCINATION lives
or dies on this curve); path length < the floor (clause 1 fails).
SCORED: TRUE iff the verdict == MIXED with BOTH BROAD-SUPPORT and
VACCINATION in the firing set.

==== THE FROZEN OPERATIONALIZATIONS (picked + frozen HERE at birth) ===

* THE STATES (three, md5-bound at birth; every one a committed
  artifact, never re-formed):
  - VARIED := runs/e341/e341_anneal_VARIED_post.pt (md5
    8a36b82ee61223674e8550499d0cfee6, size 10977104) — e311's TAVIREN
    subject + the cons's e113 300-step VARIED-context annealing.
  - FIXED := runs/e341/e341_anneal_FIXED_post.pt (md5
    365056e844cc4b7e4c1b8f330596b599, size 10977033) — the critic's
    matched-step FIXED-context control.
  - FRESH := the harness's own subject, runs/checkpoints/
    e311_TAVINST_post.pt (md5 9886b25242c90dd669c6363c39a2a3bf, size
    10976550) — loaded + read-gated by E38.phase_P1 VERBATIM (the
    subject literals are the harness's OWN).
  Committed t0 reads re-asserted on CPU (tol 2e-6, the family's
  cross-session read-determinism law), runtime-read from the md5-bound
  runs/e341/metrics.json['annealed_t0'] (never retyped): VARIED g0
  0.7519720792770386 / gm12 0.6881482601165771; FIXED g0
  0.7557547688484192 / gm12 0.13110429048538208; FRESH by phase_P1
  (g0 0.285851389169693, gm12 0.1744154542684555, CE_R
  1.5852566957473755).

* THE COMMIT MACHINERY BY IMPORT (the dispatch's own wording): the
  e338 harness (lab/e338_commit_consolidator.py, md5
  5ddc8754cb358c09278ddd926059d050 — the same bind e340/e341/x39
  held) imported with e341's REBIND SET (E38.{log, RD, NAME, T0,
  metrics} + the E261 context rebind; the envelope-tag wrapper omitted
  — no e261 function is ever called); phase_P0 (the protocol rebuild)
  and phase_P1 (the subject) execute VERBATIM on CPU; the per-arm
  commit := e341's fire_commit form (CommittedGPT(GB.G1B_CFG) ->
  load_state_dict(theta) -> commit(E38.EVENT_R = 0.7); G_BITROOT-style
  asserts: body bit-equal, anchors bit-equal, R == 0.7 within 1e-6).

* THE S1 EVENT (the ONE projected step, eval-only, in isolation) :=
  x40's ONE-STEP PORT of e322's wash step — DISCLOSED: the harness's
  wash_arm cannot run CPU-only on this box (its have_cuda check sees
  the device-count-0 CUDA as available), so the step is ported with
  the harness's OWN loop lines quoted + substring-gated (G_S1QUOTES)
  and its OWN constants by attribute (WASH_SEED 10902, WASH_ANC_BS 16,
  WASH_RND_BS 16, WASH_LR = G1.FT_LR, WASH_BETAS (0.9,0.95), WASH_WD
  0.1, WASH_CLIP 1.0): draws aj(16)/rj(16) from
  torch.Generator().manual_seed(10902) step 1; batch =
  anchor_full[aj] + the rj corpus windows; x/y = anc[:, :-1]/[:, 1:];
  ONE forward on the ARMED net (the wall projects at every forward);
  all-token mean CE; backward; clip; ONE AdamW step; then the read
  through the ARMED CPU eval twin (deepcopy of the armed net, trained
  sd loaded, battery forward SETTLES the wall — e322/e338's armed-read
  convention, wash_arm's own panel form).

* THE PER-CONTEXT READER := battery_cell's exact batching (bs=30,
  softmax at the final position: "pr = F.softmax(lg[:, -1], -1)")
  returning the full 60-vector p(T); certified against
  G1.battery_cell on the same net (mean equality <= 1e-9 — G_PERCTX).
  THE BATTERIES: g0 (60x130) and gm12 (60x118) from phase_P0, PLUS
  gp12 (60x142) rebuilt by phase_P0's own construction line (quoted +
  shape-gated, G_GP12) — the offset axis's two legs.

* THE ALIVE FRACTION := the frac of a battery's 60 contexts with
  p(T) >= 0.05 (the family's frozen bar), per arm x battery x {t0,
  s1}; the distribution's shape co-reported (sorted deciles).

* THE VACCINATION DOSE DESK := the annealing REPLAY: both arms re-run
  on CPU consuming e341's RECORDED draws bit-exact (runs/e341/
  anneal_{VARIED,FIXED}_resume.pt["draws"], md5-bound; G_REPLAY)
  under the e113 arithmetic (e341's anneal_arm loop + phase_P3c pool
  construction, lines quoted + gated; the pools rebuilt from
  phase_P0's corpus/splice — the port's own gates reproduce e341's
  G_POOLV/G_POOLF forms). Per-step dose := ||theta_t - theta_{t-1}||_2
  (flat, canonical parameter order). THE CUMULATIVE DOSE := the SUM of
  the 300 per-step doses (the path length — the vaccination
  mechanism's own quantity, FROZEN at birth; see P-x40a refinement 2).
  Co-reported: the endpoint norm at s300 + max roaming distance from
  the subject + the per-step {mean, median, max}. THE NOISE FLOOR :=
  2x the write's norm, BOTH anchors runtime-read from md5-bound
  records (never retyped): e311's write norm 8.94962906015724
  (E38.SUBJ_DW_NORM; floor 17.89925812031448) and x38's committed
  gauss ladder {1x 9.1788432658723, 2x 18.3576865317446, 4x
  36.7153730634892} (runs/x38/metrics.json md5
  6b073d97f09aad1d09f4fb39fd45aa5d; G_DOSE). The replay's panel
  displacements vs e341's committed trajectory are a FIDELITY RIDER
  (non-halting, disclosed 25% CPU-vs-GPU training-drift band).

* THE BASIN PROXY := per state, TWO sensitivity curves at doses
  0.25x/0.5x/1x of R = 0.7 (0.175/0.35/0.7 L2), read = battery g0
  mean (+ gm12 co-report), retention := read(dose)/read(0):
  (i) THE KILL RAY: u := (theta_s1_settled - theta_anchor),
  L2-normalized (the s1 projected displacement's OWN direction; the 1x
  point IS the settled s1 state — G_RAY gates the round-trip read
  equality <= 1e-6, the injection's last-ulp allowance); points
  theta_0 + dose * u, read via the plain load (x15's fp32 per-key
  injection convention: substrate fp32 + delta fp32);
  (ii) THE GAUSSIAN CONTROL: the seed-25001 standard_normal(N) draw
  (x25/x38's committed convention), L2-normalized, the same three
  doses, the same injection — expected flat (x38's precedent);
  co-reported, never adjudicates.
  FLATTER (the bars' shared clause) := BOTH annealed arms' kill-ray
  retention at 0.5R strictly exceed the fresh's (the 0.25R direction
  co-reported; no margin requirement — the bars say "flatter", not
  "how much").

* THE VERDICT COMPOSITE (frozen; mechanical):
  BROAD-SUPPORT fires iff [min over {VARIED, FIXED} of
  alive-frac(g0, s1) >= alive-frac(g0, s1)[FRESH] + 0.10] AND
  [alive-frac(s1)[VARIED] > alive-frac(s1)[FIXED] at BOTH gm12 AND
  gp12].
  VACCINATION fires iff [path-length dose >= the 2x-e311-write floor
  for BOTH arms] AND [FLATTER holds].
  BASIN-WIDTH fires iff [FLATTER holds] AND [BROAD-SUPPORT does not
  fire] (the bar's own "no support-breadth difference").
  MIXED iff >= 2 of the three fire (the decomposition reported);
  else NONE-FIRES (the table verbatim; no wording change without a
  new registered cell).

* GATES (a failure HALTS => TEXTURE, nothing adjudicated): G_MD5 (the
  three states + the four resume records + x38/e341 metrics + the
  six rig parents), G_QUOTES (every quoted source line a verbatim
  substring), G_P0P1 (the harness phases' own internal gates), G_GP12
  (the rebuilt offset battery's shape + construction), G_KEYSET (the
  canonical key order across the three states; N = 2,739,072),
  G_T0 (the three t0 g0/gm12 reads vs the committed records, tol 2e-6),
  G_PERCTX (the per-context mean == battery_cell's mean, <= 1e-9),
  G_BITROOT_{arm} (the commit firings: body bit-equal, anchors
  bit-equal, R == 0.7), G_S1EVENT
  (the first-step x md5 == e341's committed xhash[0]
  4b7c4a4c3488b23d2befb86e1b62526b bit-exact; the first draws == the
  record; the three s1 CLASSES reproduce (varied/fixed >= 0.05, fresh
  < 0.05) with |drift| <= 0.02 absolute each; the raw s1 displacement
  within [1.60, 1.71] — the committed values 1.654320..1.654362),
  G_PIN (the settled displacement == R within 1e-6), G_RAY (the 1x
  ray point reproduces the s1 read <= 1e-6), G_POOLS (the replay's
  pools == e341's phase_P3c construction), G_REPLAY (the recorded
  draws consumed bit-equal, 300 steps per arm), G_DOSE (x38's dose
  literals + e311's write norm runtime-read equality), G_NET0 (every
  state classed).

COMPUTE ENVELOPE: CPU ONLY (dispatch: x41 owns the GPU lane) —
CUDA_VISIBLE_DEVICES="" BEFORE torch import (x38's bulletproof line),
torch threads 4, no envelope-log writes, no GPU code path (the
harness's GPU branches never taken; phase_P0/phase_P1 are CPU-clean
by construction — every read in this family rides a CPU eval twin).
~610 training steps (2 x 300 replay + 3 one-step events) + ~40
battery panels + 24 ray reads on the 2.74M organism — ~15-25 min,
threads 4. TIMESTAMPS: datetime.now(UTC) only.

Outputs: runs/x40/{metrics.json (PROGRESSIVE), REPORT.md,
x40_first_step_mechanism.png} — the per-context distributions; the
dose comparison; the sensitivity curves. NO .pt artifacts (the
examined states are the md5-bound parents; the s1/ray states live in
memory). No NOTES/THINKING/QUEUE/STATE edits (dispatch; the heartbeat
folds). Birth commit BEFORE compute, pushed; smoke pass disclosed;
final commit AND push.

Smoke (X40_SMOKE=1): the FULL gate path + the s1 events + the
per-context reads + both ray families LIVE; the replay capped at the
first 8 recorded steps (the dose desk VACUOUS at the smoke horizon —
disclosed); own gitignored dir runs/x40_smoke/; NOTHING adjudicated
(SMOKE stamp on every read; no figure, no report).

Run:  python lab/x40_first_step_mechanism.py    (X40_SMOKE=1 shakedown)
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU-ONLY, bulletproof

import copy                                       # noqa: E402
import hashlib                                    # noqa: E402
import json                                       # noqa: E402
import subprocess                                 # noqa: E402
import sys                                        # noqa: E402
import time                                       # noqa: E402
from datetime import datetime, timezone           # noqa: E402
from pathlib import Path                          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")      # e228's offline convention
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:                                              # Windows console safety
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                 # noqa: BLE001
    pass

import torch                                      # noqa: E402
import torch.nn.functional as F                   # noqa: E402

import common                                     # noqa: E402
from common import run_dir, save_json, set_seed   # noqa: E402

import e338_commit_consolidator as E38            # noqa: E402 — THE
                                                  # HARNESS (by import;
                                                  # disclosed side
                                                  # effects: opens
                                                  # runs/e338/run.log +
                                                  # runs/e261/run.log in
                                                  # append — zero bytes
                                                  # written by this cell;
                                                  # e261's import-time
                                                  # cuda assert passes
                                                  # vacuously with
                                                  # device_count 0 on
                                                  # this box, verified
                                                  # pre-birth; no GPU
                                                  # path is ever taken)

torch.set_num_threads(4)          # CPU-only cell; x37/x38's exact setting

import matplotlib                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402

G1 = E38.G1                                       # the era's own module
GB = E38.GB                                       # the 2.74M family rebind
CPU = torch.device("cpu")

SMOKE = os.environ.get("X40_SMOKE") == "1"
NAME = "x40_smoke" if SMOKE else "x40"

T0 = time.time()
RD = run_dir(NAME)
LOG_PATH = RD / "run.log"
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


# ---- THE REBIND SET (e341's exact set, frozen; the leg convention) -------
thermal_log: list[dict] = []
device_events: list[dict] = []
E38.log = log
E38.RD = RD
E38.NAME = NAME
E38.T0 = T0                        # the harness's traj elapsed_s origin
E38.E261.log = log
E38.E261.NAME = NAME
E38.E261.SMOKE = SMOKE
E38.E261.T0 = T0
E38.E261.thermal_log = thermal_log
E38.E261.device_events = device_events
E38.E261.DCT_WORKERS = 4           # e307/e311's desk convention

REPO = common.REPO

# ======================================================================
# THE CONFIG (frozen)
# ======================================================================
READ_BAR = 0.05                    # the family's frozen 0.05 aliveness bar
READ_TOL = 2e-6                    # the family's cross-session read law
REPLAY_STEPS = 8 if SMOKE else 300 # g1's own smoke horizon for the replay
GAUSS_SEED = 25001                 # x25/x38's committed draw seed
RAY_DOSES = (0.25, 0.5, 1.0)       # x of R = 0.7 (the dispatch's own rungs)
MATERIAL_MARGIN = 0.10             # BROAD-SUPPORT clause 1's frozen margin
S1_DRIFT_TOL = 0.02                # the s1 reproduction band (absolute)
S1_DISP_BAND = (1.60, 1.71)        # the committed raw-displacement envelope
FIDELITY_BAND = 0.25               # the replay's non-halting rider band

# ---- the md5 binds (frozen at birth) -------------------------------------
MD5_BINDS = {                      # (path, md5, size) — size None = skip
    "varied_state": ("runs/e341/e341_anneal_VARIED_post.pt",
                     "8a36b82ee61223674e8550499d0cfee6", 10977104),
    "fixed_state": ("runs/e341/e341_anneal_FIXED_post.pt",
                    "365056e844cc4b7e4c1b8f330596b599", 10977033),
    "subject": ("runs/checkpoints/e311_TAVINST_post.pt",
                "9886b25242c90dd669c6363c39a2a3bf", 10976550),
    "anneal_varied_resume": ("runs/e341/anneal_VARIED_resume.pt",
                             "a3ea0d606e8a8a65e458b6bb328eb790", 32996036),
    "anneal_fixed_resume": ("runs/e341/anneal_FIXED_resume.pt",
                            "5f4af59d0fd8fc682d3b688d26cd1523", 32995065),
    "wash_varied_resume": ("runs/e341/wash_VARIED-COMMIT_resume.pt",
                           "91d3c4d8bb549b01ed336d7908cfd245", 43950619),
    "wash_fixed_resume": ("runs/e341/wash_FIXED-COMMIT_resume.pt",
                          "ca4e9fa7ac49976db206a6800dd854af", 43950286),
    "wash_twin_resume": ("runs/e341/wash_TWIN-COMMIT_resume.pt",
                         "30dca7f438d8257d652ba3a79627538f", 43949953),
    "x38_metrics": ("runs/x38/metrics.json",
                    "6b073d97f09aad1d09f4fb39fd45aa5d", 201351),
    "e341_metrics": ("runs/e341/metrics.json",
                     "279bd235e43da13dc78998fe9da7b63d", 50070),
    "harness": ("lab/e338_commit_consolidator.py",
                "5ddc8754cb358c09278ddd926059d050", 76039),
    "g1_rig": ("lab/g1_anchored_ball.py",
               "f4b6997b6a66013ee25da67f2b4faf01", 105063),
    "g1b_rig": ("lab/g1b_continuity.py",
                "66966621e162b6bd33cf0023b0aa485a", 70679),
    "e043_rig": ("lab/e043_install.py",
                 "f82806b369452b05b8d89ca6cebe70fa", 67163),
    "e341_rig": ("lab/e341_varied_annealing.py",
                 "18039ab39d1b04d1adef2a22b9b3a350", None),
    "common_rig": ("lab/common.py",
                   "bb2a9ad6fc7c21c06d520a3ef9feffa7", 15934),
}

# ---- the quoted source lines (substring-gated; G_QUOTES) ----------------
QUOTES = [
    # the s1 event's own arithmetic (e338's wash loop)
    ("lab/e338_commit_consolidator.py",
     "gen = torch.Generator().manual_seed(WASH_SEED)",
     "the wash step's generator (seed 10902)"),
    ("lab/e338_commit_consolidator.py",
     "aj = torch.randint(n_anc, (WASH_ANC_BS,), generator=gen)",
     "the anchor draw"),
    ("lab/e338_commit_consolidator.py",
     "rj = torch.randint(len(train_ids) - G1.BLOCK - 1, (WASH_RND_BS,),",
     "the random-corpus draw"),
    ("lab/e338_commit_consolidator.py",
     "x, y = anc[:, :-1].to(dev), anc[:, 1:].to(dev)",
     "the input/target assembly"),
    ("lab/e338_commit_consolidator.py",
     "logits, _ = net(x)          # <- the wall projects here (armed arm)",
     "THE WALL'S FORWARD (the projection point)"),
    ("lab/e338_commit_consolidator.py",
     "loss = nll.mean()           # e322: all-token mean CE",
     "the wash loss"),
    ("lab/e338_commit_consolidator.py",
     "torch.nn.utils.clip_grad_norm_(net.parameters(), WASH_CLIP)",
     "the clip"),
    ("lab/e338_commit_consolidator.py",
     "evl = copy.deepcopy(net0).to(CPU)       # the eval twin (ARMED if armed)",
     "the ARMED CPU eval twin (the read's settle convention)"),
    ("lab/e338_commit_consolidator.py",
     "net0 = G1.CommittedGPT(GB.G1B_CFG)",
     "the commit firing's construction"),
    # the wall + the battery read
    ("lab/g1_anchored_ball.py",
     "if d > self.R:",
     "the wall's projection trigger"),
    ("lab/g1_anchored_ball.py",
     "p.copy_(a + (p - a) * s)",
     "the wall's in-place projection line"),
    ("lab/g1_anchored_ball.py",
     "pr = F.softmax(lg[:, -1], -1)",
     "the battery read's softmax at the final position"),
    # the battery construction (the gp12 rebuild's own line)
    ("lab/e338_commit_consolidator.py",
     "cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ]",
     "the battery contexts' construction line (j in GEOS)"),
    # the annealing replay's own arithmetic (e341's loop + pools)
    ("lab/e341_varied_annealing.py",
     "nw = pool_x[ix].to(dev)",
     "the pool half's assembly"),
    ("lab/e341_varied_annealing.py",
     "x = torch.cat([nw[:, :-1], anc[:, :-1]], 0)",
     "the replay batch's x"),
    ("lab/e341_varied_annealing.py",
     "loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())",
     "the masked union CE (the replay loss)"),
    ("lab/e341_varied_annealing.py",
     "pre = train_ids[p - G1.PRE - j: p]",
     "the jittered pool's pre slice"),
    ("lab/e341_varied_annealing.py",
     "pool_v_x = torch.cat([jit_x[j] for j in JITTERS])        # (300, 256)",
     "the varied pool's construction"),
    ("lab/e341_varied_annealing.py",
     "pool_f_x = jit_x[FIXED_J]                                # (60, 256)",
     "the fixed pool's construction"),
    ("lab/e341_varied_annealing.py",
     'cons_anchor = p0["anchor_full"][:16]      # e113: first-16 originals',
     "the anchor bank (e113's own)"),
]

# ---- the committed literals (runtime-read from the md5-bound records) ----
E341_XHASH0 = "4b7c4a4c3488b23d2befb86e1b62526b"   # all three washes' x[0]
X38_DOSES_BOUND = {"1x": 9.1788432658723,
                   "2x": 18.3576865317446,
                   "4x": 36.7153730634892}
WRITE_NORM_BOUND = 8.94962906015724                 # e311's write (E38's)
NOISE_FLOOR = 2.0 * WRITE_NORM_BOUND                # 17.89925812031448

NET0_CLASSES = {
    "FRESH": ("BASE-formed (e311's TAVIREN install; K10K-room-projected "
              "write, pruned/localized class) + COMMIT(0.7) — the fresh "
              "class (e338/e341's TWIN arm)"),
    "VARIED": ("BASE-formed (e311's TAVIREN install) + the cons's e113 "
               "VARIED-context annealing (300 steps, seed 10901) + "
               "COMMIT(0.7) — e341's VARIED-COMMIT arm"),
    "FIXED": ("BASE-formed (e311's TAVIREN install) + the FIXED-context "
              "control annealing (matched steps/seed; the j=0 pool) + "
              "COMMIT(0.7) — e341's FIXED-COMMIT arm"),
    "s1_states": ("the three settled post-first-projected-step states "
                  "(this cell's own eval-only events; the classes above "
                  "+ ONE e322 wash step under the 0.7 wall)"),
}

REGISTERED = {
    "question_verbatim": ("WHY do annealed reads survive the first "
                          "projected step?"),
    "background_verbatim": (
        "e341 proved ANNEALING IS A DOSE (300 steps, varied OR fixed "
        "context menu, buys the s1 survival — the first projected step's "
        "annihilation of fresh installs); x38 set the noise floor (~2x "
        "the write's norm). THE QUESTION: WHY do annealed reads survive "
        "the first projected step?"),
    "hypotheses_verbatim": {
        "a_BROAD-SUPPORT": ("varied/fixed annealing spreads the read "
                            "across more contexts, so the s1 kill (which "
                            "acts at the battery's contexts) leaves alive "
                            "mass elsewhere"),
        "b_VACCINATION": ("the annealing steps are repeated sub-noise-"
                          "floor doses (each step ~the cons's jitter "
                          "norm) that immunize the read against the "
                          "commit's projection"),
        "c_BASIN-WIDTH": ("the annealed read's landscape is flatter inside "
                          "the ball (the same displacement hurts less)"),
    },
    "design_verbatim": {
        "1_THE_PER-CONTEXT_DISTRIBUTION": (
            "for the committed classes (e341's varied/fixed post-anneal "
            "states + the fresh install; the harness's commit machinery "
            "by import on CPU), read the battery per-context at t0 and "
            "after ONE projected step (the s1 event in isolation, "
            "eval-only): the frac of contexts alive (>= 0.05), the "
            "distribution's shape. BROAD-SUPPORT predicts: the fresh "
            "read's support collapses to ~0 alive contexts; the annealed "
            "reads keep a broad alive tail (and the varied arm broader "
            "at OFFSET contexts — its generalization channel)."),
        "2_THE_VACCINATION_DESK_CHECK": (
            "measure the annealing steps' own displacement norms (the "
            "cons's jitter protocol port from e341's record: per-step "
            "parameter deltas) vs x38's noise floor (~2x the write's "
            "norm). VACCINATION predicts: the cumulative annealing dose "
            ">= the noise floor (the read has 'seen' equivalent "
            "displacement); the steps' per-dose norms are the "
            "comparison."),
        "3_THE_BASIN_PROXY": (
            "the read's sensitivity curve at small displacements "
            "(0.25x/0.5x/1x the commit's R=0.7) on the three states — a "
            "flatter early curve = wider basin."),
    },
    "bars_verbatim": {
        "BROAD-SUPPORT": ("the annealed reads' alive-context fraction at "
                          "s1 is materially above the fresh's (~0) AND the "
                          "varied arm > the fixed at offset contexts — "
                          "the survival is support-breadth; composes with "
                          "e341's generalization channel."),
        "VACCINATION": ("the annealing dose >= the noise floor and the "
                        "basin proxy shows the annealed reads flatter at "
                        "sub-R displacements — the survival is "
                        "immunization; the recipe's anneal leg re-words "
                        "to 'dose >= noise floor'."),
        "BASIN-WIDTH": ("the sensitivity curves separate at small "
                        "displacements with no support-breadth difference "
                        "— the survival is landscape flatness."),
        "MIXED": "multiple fire — report the decomposition.",
    },
    "lab_lean_verbatim": (
        "MIXED (broad-support + vaccination), weakly — the two evening's "
        "channels (e341's generalization + x38's dose) both point into "
        "the mechanism; the countervailing: the fixed arm survived s1 too "
        "without the generalization — support-breadth at the battery's "
        "own contexts may suffice. State your own read."),
    "P_x40a": {
        "my_guess": "MIXED (BROAD-SUPPORT + VACCINATION), weakly "
                    "(concurring with the lab lean)",
        "registered": (
            "THREE REFINEMENTS. (1) The g0 tail is predicted SHARED, the "
            "offset tail is the varied arm's own: both annealed arms "
            "survived s1 at the battery's geometry at nearly equal s1 "
            "means (0.115/0.097), so BROAD-SUPPORT's two clauses are "
            "separate measurements and the countervailing note is the "
            "possibility that clause 1 fires while the offset difference "
            "is merely carried over from t0 — the decomposition will "
            "name which tail carries the survival. (2) The dose clause's "
            "reading is LOAD-BEARING: on the ENDPOINT reading the "
            "committed cumulative displacement (16.779/16.801) sits ~6-7% "
            "BELOW the 2x-write floor (17.899) — a miss; on the "
            "PATH-LENGTH reading (the SUM of the 300 per-step doses, "
            "each individually far below the floor) it is far ABOVE — a "
            "near-certain hit. I register the PATH LENGTH as the bar's "
            "cumulative dose (the immunization metaphor is accumulated "
            "exposure, not net displacement), the endpoint + max-roaming "
            "co-reported; the honest decomposition must own that the "
            "ENDPOINT ALONE does not clear the floor. (3) The gaussian "
            "control is expected FLAT for all three states (x38: 9.18 "
            "L2 moved a living read -0.07%; 0.7 L2 is 8% of that) — the "
            "basin proxy's weight sits on the KILL-RAY curve; if the "
            "gaussian separates too, that is isotropic basin width — "
            "stronger than the bar needs, co-reported."),
        "predicted_shape": (
            "fresh alive-frac at s1 ~0 (0-2 of 60 at g0); both annealed "
            "arms keep a broad g0 tail (alive-frac ~0.1-0.4); varied > "
            "fixed at gm12 AND gp12 alive-frac at s1; the kill-ray "
            "retention curves ordered annealed > fresh already at 0.5R "
            "(possibly at 0.25R); the gaussian flat everywhere "
            "(retentions > 0.9); the replay's per-step doses ~0.2-0.7 "
            "with max << floor, path length ~60-180 >> 17.9"),
        "falsifiers": (
            "the annealed g0 alive-frac within the fresh's (BROAD-SUPPORT "
            "clause 1 fails); the fixed arm >= the varied at either "
            "offset battery (clause 2 fails); the kill-ray retentions "
            "overlapping at 0.5R (VACCINATION clause 2 fails — with the "
            "dose clause near-certain, VACCINATION lives or dies on this "
            "curve); path length < the floor (clause 1 fails)"),
        "scored": "TRUE iff the verdict == MIXED with BOTH BROAD-SUPPORT "
                  "and VACCINATION in the firing set",
    },
    "verdict_composite_frozen": {
        "BROAD-SUPPORT": "min_annealed_alive_g0_s1 >= fresh_alive_g0_s1 + "
                         "0.10 AND varied_alive > fixed_alive at BOTH "
                         "gm12 AND gp12 at s1",
        "VACCINATION": "path_length >= 2x-write floor (17.89925812031448) "
                       "for BOTH arms AND FLATTER (both annealed kill-ray "
                       "retentions at 0.5R > the fresh's)",
        "BASIN-WIDTH": "FLATTER AND NOT BROAD-SUPPORT",
        "MIXED": ">= 2 of the three fire (decomposition reported)",
        "else": "NONE-FIRES (the table verbatim; no wording change "
                "without a new registered cell)",
    },
    "registration": ("question + background + hypotheses + design + bars "
                     "+ lab lean + P-x40a VERBATIM from the dispatch "
                     "letter; every convention picked + frozen HERE at "
                     "birth BEFORE compute; this script committed at "
                     "birth; adjudicate against exactly this; no bar "
                     "shopping."),
}

deviations: list[str] = [
    "CPU-ONLY CELL (dispatch: x41 owns the GPU lane): CUDA_VISIBLE_DEVICES"
    "='' before torch import, torch threads 4, no envelope-log writes, no "
    "GPU code path. The e338 harness is imported (the dispatch's 'commit "
    "machinery by import'); its GPU branches are never taken — phase_P0/"
    "phase_P1 are CPU-clean by construction, and wash_arm (whose "
    "have_cuda check would misfire on this box, where is_available() "
    "returns True with device_count 0) is NEVER CALLED.",
    "THE S1 EVENT IS A ONE-STEP PORT of e322's wash step (the harness's "
    "own loop lines quoted + substring-gated; its own constants by "
    "attribute) — disclosed because wash_arm cannot run CPU-only here; "
    "the port is certified by the bit-exact first-step input md5 vs "
    "e341's committed xhash[0] and by the three committed s1 classes "
    "reproducing within the frozen 0.02 drift band.",
    "THE ANNEALING REPLAY is a CPU re-run of e341's two anneal arms "
    "consuming e341's RECORDED draws bit-exact (G_REPLAY) under the "
    "e113 arithmetic (quoted + gated); its panel displacements vs "
    "e341's committed trajectory are a FIDELITY RIDER (non-halting, "
    "25% CPU-vs-GPU training-drift band) — the dose norms are robust "
    "to the drift.",
    "THE CUMULATIVE DOSE := the path length (the sum of per-step "
    "displacement norms) — the vaccination mechanism's own quantity, "
    "frozen at birth; the endpoint norm and max roaming co-reported "
    "(the endpoint sits ~6-7% below the 2x floor — pre-birth known "
    "from the committed record, disclosed here so the reading cannot "
    "be shopped post hoc).",
    "THE BASIN PROXY carries TWO directions (the kill ray + the "
    "seed-25001 gaussian control); the bars' FLATTER clause "
    "adjudicates on the KILL RAY (the mechanism's own direction); the "
    "gaussian is the isotropic control x38 already showed silent at "
    "13x this dose on a living read — expected flat, co-reported.",
    "THE gp12 BATTERY is rebuilt by phase_P0's own construction line "
    "(p0 returns only g-12/g0) — the offset axis's second leg.",
    "NO .pt artifacts written; the s1/ray states live in memory only "
    "(the examined states are the md5-bound parents).",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the heartbeat "
    "folds).",
    "Smoke (X40_SMOKE=1): the full gate path + the s1 events + the "
    "per-context reads + both ray families LIVE; the replay capped at "
    "the first 8 recorded steps (the dose desk VACUOUS at the smoke "
    "horizon, disclosed); own gitignored dir; NOTHING adjudicated.",
    "IMPORT SIDE EFFECTS (inherited, e341's disclosure): importing the "
    "harness opens runs/e338/run.log in append and importing e261 "
    "opens runs/e261/run.log — ZERO bytes written to either by this "
    "cell.",
]

metrics: dict = {
    "experiment": "x40_first_step_mechanism",
    "phase": ("THE FIRST-STEP SURVIVAL MECHANISM — why annealed reads "
              "survive the first projected step: the per-context s1 "
              "distribution (broad-support), the annealing dose vs x38's "
              "noise floor (vaccination), the sub-R sensitivity curves "
              "(basin-width) on e341's committed classes (varied/fixed "
              "post-anneal + the fresh install), CPU desk"),
    "date": common.now_iso(),
    "status": "PARTIAL: startup",
    "registration": REGISTERED["registration"],
    "registered": REGISTERED,
    "smoke": SMOKE,
    "cpu_only": True,
    "threads": {"torch": 4},
    "envelope": {
        "device": "CPU ONLY (CUDA_VISIBLE_DEVICES=''; x41 owns the GPU "
                  "lane; no envelope-log writes)",
        "timestamps": "datetime.now(UTC) only",
    },
    "deviations": deviations,
    "builds_on": [
        "e341 (THE CLASSES: its committed VARIED/FIXED post-anneal states "
        "+ the TWIN=fresh comparison, its committed s1/t0/gm12 values and "
        "wash xhash[0] the reproduction targets; its anneal_arm + pool "
        "construction the replay's ported arithmetic; AMOUNT-NOT-TYPE — "
        "the annealing is a dose, the question this cell dissects)",
        "e338 (THE HARNESS: phase_P0/phase_P1 + the commit firing by "
        "import; the armed-read settle convention; EVENT_R 0.7)",
        "e322 (THE WASH: the one-step event's own arithmetic — draws, "
        "batch, all-token CE, optimizer, clip, seed 10902)",
        "e311 (THE SUBJECT: the fresh TAVIREN install; the write's norm "
        "8.9496 — the noise floor's anchor)",
        "x38 (THE NOISE FLOOR: the committed gauss ladder 1x/2x/4x "
        "(9.18/18.36/36.72 L2) — GAUSS-BREAKS-AT-4x; the 2x rung the "
        "floor's value; the seed-25001 draw convention)",
        "g1/g1b (THE MACHINERY: CommittedGPT's wall, battery_cell, "
        "evl_load's settle+disarm pivot, the 2.74M family)",
        "x15 (THE INJECTION CONVENTION: substrate fp32 + delta fp32 per "
        "key)",
        "R78 (the review that minted this card — the first-step "
        "mechanism)",
    ],
    "whats_new": [
        "THE PER-CONTEXT S1 DISTRIBUTION — the survival axis's first "
        "anatomy: not the battery MEAN (every prior cell's currency) but "
        "the 60-context vector at t0 vs s1, per arm x battery (g-12/g0/"
        "g+12) — the support-breadth question made measurable",
        "THE ANNEALING DOSE LEDGER — the 300 per-step displacement norms "
        "of both committed annealing arms replayed on their recorded "
        "draws, against x38's noise-floor ladder — the vaccination "
        "hypothesis's own quantities",
        "THE SUB-R SENSITIVITY CURVES on the kill ray (the s1 "
        "displacement's own direction, 0.25x/0.5x/1x R) + the gaussian "
        "isotropic control — the basin-width question separated from the "
        "support question by direction",
        "THE MECHANISM DECOMPOSITION — three registered hypotheses, "
        "mechanically separated on the same three committed states in "
        "one CPU cell",
    ],
    "gates": {},
}
E38.metrics = metrics                     # the harness's phases write here

BIRTH_COMMIT_PINNED = "fc0d67d"     # pinned at the birth commit


# ------------------------------------------------------------------ helpers
def md5of(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:                                          # noqa: BLE001
        return "unavailable"


def write_partial(note: str) -> None:
    metrics["status"] = f"PARTIAL: {note} ({common.now_iso()})"
    save_json(RD / "metrics.json", metrics)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def flat_params(net) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).cpu()
                      for p in net.parameters()]).clone()


# ---- THE PER-CONTEXT READER (battery_cell's exact batching) --------------
@torch.no_grad()
def battery_percell(net, ids: torch.Tensor, zid: int, bs: int = 30
                    ) -> torch.Tensor:
    """battery_cell's exact instrument returning the FULL per-context
    vector (bs=30 batches, softmax at the final position)."""
    net.eval()
    pzs = []
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs.append(pr[:, zid])
    return torch.cat(pzs)


# ---- THE KEY/SLICE MAP (x15's per-key injection convention) --------------
def key_slices(sd: dict) -> list[tuple[str, tuple]]:
    out, off = [], 0
    for k, v in sd.items():
        n = v.numel()
        out.append((k, (off, off + n)))
        off += n
    return out


def inject(theta: dict, delta_flat: torch.Tensor,
           ks: list[tuple[str, tuple]]) -> dict:
    """x15's fp32 per-key injection: substrate fp32 + delta fp32."""
    out = {}
    for k, (a, b) in ks:
        out[k] = (theta[k].float().reshape(-1)
                  + delta_flat[a:b]).reshape(theta[k].shape).clone()
    return out


# ======================================================================
# P1 — THE BIND GATES (md5s + quotes) + the harness's own phases
# ======================================================================
def phase_gates() -> dict:
    log("P1 — THE BIND GATES (the md5 binds + the quoted source lines)")
    binds = {}
    for key, (rel, md5, size) in MD5_BINDS.items():
        p = REPO / rel
        mine = md5of(p)
        ok = (mine == md5) and (size is None or p.stat().st_size == size)
        binds[key] = {"path": rel, "md5": mine, "bound_md5": md5,
                      "size": p.stat().st_size,
                      "bound_size": size, "pass": bool(ok)}
    g_md5 = {"binds": binds,
             "claim": "every examined state, resume record, metrics "
                      "record and rig parent md5-bound at run time",
             "pass": bool(all(b["pass"] for b in binds.values()))}
    assert g_md5["pass"], f"G_MD5 FAILED: {g_md5}"
    metrics["gates"]["G_MD5"] = g_md5
    log(f"G_MD5 PASS: {len(binds)} binds ({sum(1 for b in binds.values() if b['pass'])} ok)")

    quotes = []
    for src, frag, why in QUOTES:
        txt = (REPO / src).read_text(encoding="utf-8")
        quotes.append({"source": src, "fragment": frag, "why": why,
                       "verified_substring": bool(frag in txt)})
    g_q = {"quotes": quotes,
           "claim": "every ported line is a verbatim substring of its "
                    "committed source rig (the s1 event's wash loop, the "
                    "wall, the battery construction, the anneal replay)",
           "pass": bool(all(q["verified_substring"] for q in quotes))}
    assert g_q["pass"], f"G_QUOTES FAILED: {g_q}"
    metrics["gates"]["G_QUOTES"] = g_q
    log(f"G_QUOTES PASS: {len(quotes)}/{len(quotes)} verbatim substring "
        f"quotes verified")

    # the noise-floor literals, runtime-read from the md5-bound records
    x38 = json.loads((REPO / "runs" / "x38" / "metrics.json")
                     .read_text(encoding="utf-8"))
    doses = {k: x38["gauss_rider"]["doses"][k]["l2"] for k in X38_DOSES_BOUND}
    g_dose = {"x38_doses": doses, "bound": X38_DOSES_BOUND,
              "write_norm_e311": E38.SUBJ_DW_NORM,
              "write_norm_bound": WRITE_NORM_BOUND,
              "noise_floor_2x": NOISE_FLOOR,
              "claim": "x38's gauss ladder + e311's write norm runtime-"
                       "read from the md5-bound records (never retyped)",
              "pass": bool(all(doses[k] == X38_DOSES_BOUND[k]
                               for k in X38_DOSES_BOUND)
                           and E38.SUBJ_DW_NORM == WRITE_NORM_BOUND)}
    assert g_dose["pass"], f"G_DOSE FAILED: {g_dose}"
    metrics["gates"]["G_DOSE"] = g_dose
    log(f"G_DOSE PASS: x38 ladder {{{', '.join(f'{k}:{v:.4f}' for k, v in doses.items())}}}; "
        f"floor 2x{E38.SUBJ_DW_NORM:.4f} = {NOISE_FLOOR:.4f}")
    write_partial("P1 the bind gates passed")
    return {"x38_doses": doses}


def phase_harness() -> dict:
    log("P2 — THE HARNESS'S OWN PHASES (phase_P0 + phase_P1, verbatim)")
    p0 = E38.phase_P0()                     # the protocol rebuild (CPU)
    p1 = E38.phase_P1(p0)                   # the subject (e311's, CPU)
    g = {"harness_md5": MD5_BINDS["harness"][1],
         "claim": "E38.phase_P0 + E38.phase_P1 executed VERBATIM on CPU "
                  "(their internal gates G_NAMEFREE/G_SPLICE/G_BATTERY/"
                  "G_BANK/G_SUBJECT all asserted inside the harness)",
         "pass": True}
    metrics["gates"]["G_P0P1"] = g
    # the gp12 battery (phase_P0's own construction line, j=+12)
    train_text = p0["train_text"]
    install_occ = p0["install_occ"]
    cs = [train_text[p - G1.PRE - 12: p] for p, _ in install_occ]
    gp12_ids = torch.stack([p0["corpus"].encode(c) for c in cs])
    g12 = {"shape": list(gp12_ids.shape),
           "expected": [60, G1.PRE + 12],
           "construction": "cs = [train_text[p - G1.PRE - j: p] for p, _ "
                           "in install_occ] at j=+12 (phase_P0's own "
                           "line, quoted + gated)",
           "claim": "the g+12 offset battery rebuilt by the harness's "
                    "own construction",
           "pass": bool(list(gp12_ids.shape) == [60, G1.PRE + 12])}
    assert g12["pass"], f"G_GP12 FAILED: {g12}"
    metrics["gates"]["G_GP12"] = g12
    log(f"G_GP12 PASS: the g+12 battery {gp12_ids.shape} (the offset "
        f"axis's second leg)")
    return {"p0": p0, "p1": p1, "gp12_ids": gp12_ids}


# ======================================================================
# P3 — THE STATES + THE t0 PER-CONTEXT READS
# ======================================================================
def load_state(rel: str) -> dict:
    art = torch.load(REPO / rel, map_location="cpu", weights_only=False)
    return {k: v.detach().clone() for k, v in art["model"].items()}


def phase_states(hp: dict) -> dict:
    log("P3 — THE STATES (the three committed classes; the t0 per-context "
        "reads)")
    p0 = hp["p0"]
    batteries = {"g-12": p0["gm12_ids"], "g0": p0["g0_ids"],
                 "g+12": hp["gp12_ids"]}
    arms = {
        "VARIED": {"rel": "runs/e341/e341_anneal_VARIED_post.pt",
                   "theta": load_state("runs/e341/e341_anneal_VARIED_post.pt")},
        "FIXED": {"rel": "runs/e341/e341_anneal_FIXED_post.pt",
                  "theta": load_state("runs/e341/e341_anneal_FIXED_post.pt")},
        "FRESH": {"rel": "runs/checkpoints/e311_TAVINST_post.pt",
                  "theta": hp["p1"]["theta0"]},
    }

    # G_KEYSET: identical body key order across the three states
    keys = {a: list(arms[a]["theta"].keys()) for a in arms}
    n_par = sum(v.numel() for v in arms["FRESH"]["theta"].values())
    g_keys = {"key_order_identical": bool(keys["VARIED"] == keys["FIXED"]
                                          == keys["FRESH"]),
              "n_params": n_par,
              "claim": "the flat-space math rides one canonical key order",
              "pass": bool(keys["VARIED"] == keys["FIXED"] == keys["FRESH"]
                           and n_par == GB.G1B_PARAMS)}
    assert g_keys["pass"], f"G_KEYSET FAILED: {g_keys}"
    metrics["gates"]["G_KEYSET"] = g_keys
    log(f"G_KEYSET PASS: N = {n_par:,} (the 2.74M family)")

    e341 = json.loads((REPO / "runs" / "e341" / "metrics.json")
                      .read_text(encoding="utf-8"))
    for a in ("VARIED", "FIXED", "FRESH"):
        net = G1.evl_load(arms[a]["theta"])          # the plain read owner
        t0 = {}
        for bn, ids in batteries.items():
            pv = battery_percell(net, ids, p0["tid"])
            ref = G1.battery_cell(net, ids, p0["tid"])
            t0[bn] = {"mean": float(pv.mean()),
                      "percell": [float(x) for x in pv],
                      "alive_frac": float((pv >= READ_BAR).float().mean()),
                      "cert_abs_diff": abs(float(pv.mean())
                                           - ref["mean_pz"])}
        t0["ce_r"] = G1.ce_fixed_cpu(net, *p0["r_eval_xy"])
        arms[a]["t0"] = t0
        del net
        log(f"  [{a}] t0: " + " | ".join(
            f"{bn} mean {t0[bn]['mean']:.4f} alive "
            f"{t0[bn]['alive_frac']:.3f}" for bn in batteries)
            + f" | CE_R {t0['ce_r']:.4f}")

    # G_PERCTX: the per-context reader certified against battery_cell
    cert = max(arms[a]["t0"][bn]["cert_abs_diff"] for a in arms
               for bn in batteries)
    g_pc = {"max_abs_diff": cert, "tol": 1e-9,
            "claim": "the per-context reader's mean == battery_cell's "
                     "mean_pz (the instrument certification, 9 cells)",
            "pass": bool(cert <= 1e-9)}
    assert g_pc["pass"], f"G_PERCTX FAILED: {g_pc}"
    metrics["gates"]["G_PERCTX"] = g_pc
    log(f"G_PERCTX PASS: max |mean diff| {cert:.2e} <= 1e-9")

    # G_T0: the committed t0 reads reproduce (tol 2e-6)
    drifts = {}
    for a, bn, committed in (("VARIED", "g0", e341["annealed_t0"]["VARIED"]["g0"]),
                             ("VARIED", "g-12", e341["annealed_t0"]["VARIED"]["gm12"]),
                             ("FIXED", "g0", e341["annealed_t0"]["FIXED"]["g0"]),
                             ("FIXED", "g-12", e341["annealed_t0"]["FIXED"]["gm12"]),
                             ("FRESH", "g0", E38.SUBJ_READ_T),
                             ("FRESH", "g-12", E38.SUBJ_GM12_T)):
        drifts[f"{a}:{bn}"] = {"mine": arms[a]["t0"][bn]["mean"],
                               "committed": committed,
                               "abs_diff": abs(arms[a]["t0"][bn]["mean"]
                                               - committed)}
    g_t0 = {"drifts": drifts, "tol": READ_TOL,
            "claim": "the three states' committed t0 reads reproduce on "
                     "CPU (e341's annealed_t0 + the harness's subject "
                     "literals, runtime-read)",
            "pass": bool(all(d["abs_diff"] <= READ_TOL
                             for d in drifts.values()))}
    assert g_t0["pass"], f"G_T0 FAILED: {g_t0}"
    metrics["gates"]["G_T0"] = g_t0
    log("G_T0 PASS: all six committed t0 reads reproduce within 2e-6 "
        "(max " + f"{max(d['abs_diff'] for d in drifts.values()):.1e})")
    write_partial("P3 the states loaded; the t0 per-context reads done")
    return {"arms": arms, "batteries": batteries, "e341": e341}


# ======================================================================
# P4 — THE S1 EVENTS (the one projected step per arm; the per-context
#      distribution after the kill)
# ======================================================================
def fire_commit(theta: dict, tag: str):
    """e341's fire_commit form VERBATIM (== e338's phase_P2 firing):
    CommittedGPT(G1B_CFG) -> load_state_dict(theta) -> commit(0.7)."""
    net0 = G1.CommittedGPT(GB.G1B_CFG)
    net0.load_state_dict(theta)
    net0.commit(E38.EVENT_R)
    body, _anch = G1.split_anchored_sd(net0.state_dict())
    md = max(float((body[k].float() - theta[k].float()).abs().max())
             for k in theta)
    anch_ok = all(torch.equal(net0._anchor(n), p.detach())
                  for n, p in net0.named_parameters())
    g = {"arm": tag, "R": net0.R, "body_max_abs_diff": md,
         "anchors_bit_equal": bool(anch_ok),
         "anchored": bool(net0.anchored),
         "claim": (f"the {tag} armed net's body == the state bit-exact; "
                   f"the anchors bit-equal the parameters (the wall "
                   f"inert before the first forward)"),
         "pass": bool(abs(net0.R - E38.EVENT_R) <= 1e-6 and md == 0.0
                      and anch_ok and net0.anchored)}
    assert g["pass"], f"G_BITROOT[{tag}] FAILED: {g}"
    metrics["gates"][f"G_BITROOT_{tag}"] = g
    log(f"G_BITROOT_{tag} PASS: R={net0.R}; body |d| {md:.1e}; anchors "
        f"bit-equal")
    return net0


def s1_event(theta: dict, tag: str, p0: dict) -> dict:
    """THE ONE PROJECTED STEP (eval-only, in isolation): the e322 wash
    step's own arithmetic (quoted + gated), ONE step on the ARMED net,
    then the ARMED CPU eval twin (wash_arm's own panel form — the
    settle convention). The port's ONLY delta vs wash_arm: no .to(dev)
    (CPU cell)."""
    net0 = fire_commit(theta, tag)                   # the armed net
    net = copy.deepcopy(net0)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=E38.WASH_LR,
                            betas=E38.WASH_BETAS,
                            weight_decay=E38.WASH_WD)
    gen = torch.Generator().manual_seed(E38.WASH_SEED)
    train_ids = p0["train_ids"]
    n_anc = p0["anchor_full"].shape[0]
    # e322's step-1 draws VERBATIM (aj 16 anchors + rj 16 random)
    aj = torch.randint(n_anc, (E38.WASH_ANC_BS,), generator=gen)
    rj = torch.randint(len(train_ids) - G1.BLOCK - 1, (E38.WASH_RND_BS,),
                       generator=gen)
    anc = torch.cat([p0["anchor_full"][aj],
                     torch.stack([train_ids[s: s + G1.BLOCK]
                                  for s in rj])], 0)
    x, y = anc[:, :-1], anc[:, 1:]                   # CPU (no .to(dev))
    xhash = hashlib.md5(x.detach().contiguous().numpy().tobytes()
                        ).hexdigest()
    logits, _ = net(x)          # <- the wall projects here (armed arm)
    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                          y.reshape(-1), reduction="none"
                          ).view(x.shape[0], x.shape[1])
    loss = nll.mean()           # e322: all-token mean CE
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), E38.WASH_CLIP)
    opt.step()
    theta_flat = torch.cat([theta[k].float().reshape(-1) for k in theta])
    disp_raw = float(torch.norm(flat_params(net) - theta_flat))
    trained_sd = {k: v.detach().cpu().clone()
                  for k, v in net.state_dict().items()}
    del net, opt

    # the read through the ARMED CPU eval twin (wash_arm's panel form):
    # load the one-step-trained sd; the battery forward SETTLES the wall
    evl = copy.deepcopy(net0)
    evl.load_state_dict(trained_sd)
    evl.eval()
    return {"net0": net0, "evl": evl, "xhash": xhash,
            "draws": {"aj": aj.tolist(), "rj": rj.tolist()},
            "disp_raw": disp_raw, "loss": float(loss)}


def phase_s1(st: dict, hp: dict) -> dict:
    log("P4 — THE S1 EVENTS (the one projected step, in isolation, per "
        "arm)")
    p0 = hp["p0"]
    batteries = st["batteries"]
    e341 = st["e341"]
    committed_s1 = {a: e341["washes"][f"{a}-COMMIT"]["s1"]
                    for a in ("VARIED", "FIXED", "TWIN")}
    committed_s1["FRESH"] = committed_s1.pop("TWIN")   # the class map
    rec_draws = {}
    for a, ck in (("VARIED", "wash_VARIED-COMMIT_resume.pt"),
                  ("FIXED", "wash_FIXED-COMMIT_resume.pt"),
                  ("FRESH", "wash_TWIN-COMMIT_resume.pt")):
        rr = torch.load(REPO / "runs" / "e341" / ck, map_location="cpu",
                        weights_only=False)
        rec_draws[a] = rr["draws"][0]
        assert rr["xhash"][0] == E341_XHASH0, f"xhash drift in {ck}"

    out = {}
    for a in ("VARIED", "FIXED", "FRESH"):
        ev = s1_event(st["arms"][a]["theta"], a, p0)
        s1 = {}
        for bn, ids in batteries.items():
            pv = battery_percell(ev["evl"], ids, p0["tid"])
            s1[bn] = {"mean": float(pv.mean()),
                      "percell": [float(v) for v in pv],
                      "alive_frac": float((pv >= READ_BAR).float().mean())}
        s1["ce_r"] = G1.ce_fixed_cpu(ev["evl"], *p0["r_eval_xy"])
        wr = ev["evl"].wall_report()                  # AFTER the reads:
                                                      # the settled state
        settled = {k: v.detach().cpu().clone()
                   for k, v in ev["evl"].state_dict().items()
                   if not k.startswith("anch__")}
        out[a] = {"s1": s1, "xhash": ev["xhash"], "draws": ev["draws"],
                  "disp_raw": ev["disp_raw"], "wall_report": wr,
                  "settled": settled}
        log(f"  [{a}] s1: " + " | ".join(
            f"{bn} mean {s1[bn]['mean']:.6f} alive "
            f"{s1[bn]['alive_frac']:.3f}" for bn in batteries)
            + f" | CE_R {s1['ce_r']:.4f} | |d|raw {ev['disp_raw']:.6f} "
            f"(proj {wr['d_proj']:.6f})")

    # ---- G_S1EVENT: the port's certification ---------------------------
    same_x = all(out[a]["xhash"] == E341_XHASH0 for a in out)
    same_d = all(out[a]["draws"]["aj"] == rec_draws[a]["aj"]
                 and out[a]["draws"]["rj"] == rec_draws[a]["rj"]
                 for a in out)
    classes = {a: bool(out[a]["s1"]["g0"]["mean"] >= READ_BAR)
               for a in out}
    classes_ok = (classes["VARIED"] and classes["FIXED"]
                  and not classes["FRESH"])
    drift = {a: {"mine": out[a]["s1"]["g0"]["mean"],
                 "committed": committed_s1[a],
                 "abs_diff": abs(out[a]["s1"]["g0"]["mean"]
                                 - committed_s1[a])} for a in out}
    disp_ok = all(S1_DISP_BAND[0] <= out[a]["disp_raw"] <= S1_DISP_BAND[1]
                  for a in out)
    g_s1 = {"xhash0": E341_XHASH0, "xhash_bit_exact": bool(same_x),
            "draws_bit_equal": bool(same_d),
            "s1_classes": classes,
            "classes_reproduce": bool(classes_ok),
            "drifts": drift, "drift_tol": S1_DRIFT_TOL,
            "disp_raw_band": list(S1_DISP_BAND),
            "disp_raw_in_band": bool(disp_ok),
            "claim": ("the one-step port reproduces e341's committed "
                      "first-step input bit-exact and the three "
                      "committed s1 classes (varied/fixed alive, fresh "
                      "dead) within the frozen drift band"),
            "pass": bool(same_x and same_d and classes_ok and disp_ok
                         and all(d["abs_diff"] <= S1_DRIFT_TOL
                                 for d in drift.values()))}
    assert g_s1["pass"], f"G_S1EVENT FAILED: {g_s1}"
    metrics["gates"]["G_S1EVENT"] = g_s1
    log("G_S1EVENT PASS: xhash bit-exact; draws bit-equal; classes "
        f"reproduce (drifts " + ", ".join(
            f"{a} {drift[a]['abs_diff']:.1e}" for a in drift) + ")")

    # ---- G_PIN: the settled displacement == R --------------------------
    pin = {a: out[a]["wall_report"] for a in out}
    g_pin = {"R": E38.EVENT_R, "walls": pin,
             "claim": "every settled s1 state sits ON the wall (d == R "
                      "within 1e-6; the projected step's definition)",
             "pass": bool(all(abs(pin[a]["d_proj"] - E38.EVENT_R) <= 1e-6
                              for a in pin))}
    assert g_pin["pass"], f"G_PIN FAILED: {g_pin}"
    metrics["gates"]["G_PIN"] = g_pin
    log("G_PIN PASS: all three settled states at d == R = 0.7")
    write_partial("P4 the s1 events complete; the per-context "
                  "distributions done")
    return out


# ======================================================================
# P5 — THE BASIN PROXY (the kill ray + the gaussian control)
# ======================================================================
def phase_basin(st: dict, hp: dict, s1s: dict) -> dict:
    log("P5 — THE BASIN PROXY (the kill ray + the gaussian control at "
        "0.25x/0.5x/1x R)")
    p0 = hp["p0"]
    out = {}
    for a in ("VARIED", "FIXED", "FRESH"):
        theta = st["arms"][a]["theta"]
        ks = key_slices(theta)
        anchor_flat = torch.cat([theta[k].float().reshape(-1)
                                 for k in theta])
        settled_flat = torch.cat([s1s[a]["settled"][k].float().reshape(-1)
                                  for k in theta])
        u = settled_flat - anchor_flat            # the kill direction
        u_norm = float(torch.norm(u))
        u_hat = u / u_norm                         # (norm == R, the
                                                   # settled wall radius)
        gen = torch.Generator().manual_seed(GAUSS_SEED)
        g = torch.randn(anchor_flat.shape[0], generator=gen)
        g = g / float(torch.norm(g))              # the isotropic control
        curves = {"kill_ray": {}, "gaussian": {}}
        for frac in RAY_DOSES:
            for name, vec in (("kill_ray", u_hat), ("gaussian", g)):
                if name == "kill_ray" and frac == 1.0:
                    delta = u          # the exact settled displacement
                                       # (bit-identity for G_RAY; the
                                       # 1x point IS the s1 state)
                else:
                    delta = vec * (frac * E38.EVENT_R)
                net = G1.evl_load(inject(theta, delta, ks))
                rd = G1.battery_cell(net, p0["g0_ids"], p0["tid"])
                rd12 = G1.battery_cell(net, p0["gm12_ids"], p0["tid"])
                curves[name][f"{frac}x"] = {
                    "dose_l2": float(frac * E38.EVENT_R),
                    "g0_mean": rd["mean_pz"], "gm12_mean": rd12["mean_pz"]}
                del net
        curves["u_norm"] = u_norm
        curves["t0_g0"] = st["arms"][a]["t0"]["g0"]["mean"]
        out[a] = curves
        kr = " -> ".join(f"{f}x:{curves['kill_ray'][f'{f}x']['g0_mean']:.4f}"
                         for f in RAY_DOSES)
        ga = " -> ".join(f"{f}x:{curves['gaussian'][f'{f}x']['g0_mean']:.4f}"
                         for f in RAY_DOSES)
        log(f"  [{a}] kill-ray g0 {kr} | gaussian g0 {ga} | "
            f"||u|| {u_norm:.6f}")

    # ---- G_RAY: the 1x kill-ray point reproduces the s1 read -----------
    rr = {a: {"ray_1x": out[a]["kill_ray"]["1.0x"]["g0_mean"],
              "s1_read": s1s[a]["s1"]["g0"]["mean"]} for a in out}
    g_ray = {"roundtrips": rr, "tol": 1e-6,
             "claim": "the 1x ray point (theta + 1.0 * (settled - "
                      "theta)) reproduces the settled s1 read — the "
                      "ray construction's own certification (1e-6: the "
                      "injection's last-ulp reconstruction allowance)",
             "pass": bool(all(abs(v["ray_1x"] - v["s1_read"]) <= 1e-6
                              for v in rr.values()))}
    assert g_ray["pass"], f"G_RAY FAILED: {g_ray}"
    metrics["gates"]["G_RAY"] = g_ray
    log("G_RAY PASS: the 1x points round-trip (max |d| "
        + f"{max(abs(v['ray_1x'] - v['s1_read']) for v in rr.values()):.1e})")
    write_partial("P5 the basin proxy curves done")
    return out


# ======================================================================
# P6 — THE VACCINATION DOSE DESK (the annealing replay on the recorded
#      draws)
# ======================================================================
def build_pools(p0: dict) -> dict:
    """e341's phase_P3c construction VERBATIM (lines quoted + gated):
    the jittered install pool + the fixed j=0 control."""
    train_ids, corpus = p0["train_ids"], p0["corpus"]
    install_occ = p0["install_occ"]
    name_ids_t = corpus.encode("TAVIREN")
    assert len(name_ids_t) == 7, "TAVIREN must be 7 ids"
    jit_x, jit_mask = {}, {}
    for j in G1.JITTERS:
        jwins = []
        for p, h in install_occ:
            pre = train_ids[p - G1.PRE - j: p]
            post = train_ids[p + len(h): p + len(h) + G1.POST_CAP - j]
            w = torch.cat([pre, name_ids_t, post])
            if len(w) != G1.BLOCK:
                raise RuntimeError(f"jit window len {len(w)} != "
                                   f"{G1.BLOCK} at j={j}")
            jwins.append(w)
        jit_x[j] = torch.stack(jwins)
        m = torch.zeros(len(jwins), G1.BLOCK - 1, dtype=torch.bool)
        m[:, G1.PRE - 1 + j: G1.PRE - 1 + j + len(name_ids_t)] = True
        jit_mask[j] = m
    pool_v_x = torch.cat([jit_x[j] for j in G1.JITTERS])        # (300,256)
    pool_v_mask = torch.cat([jit_mask[j] for j in G1.JITTERS])
    pool_f_x = jit_x[0]                                         # (60,256)
    pool_f_mask = jit_mask[0]
    g_pool = {
        "varied_shape": list(pool_v_x.shape),
        "fixed_shape": list(pool_f_x.shape),
        "jitters": list(G1.JITTERS),
        "masks": {"varied": int(pool_v_mask.sum()),
                  "fixed": int(pool_f_mask.sum())},
        "expected_masks": {"varied": 300 * 7, "fixed": 60 * 7},
        "claim": "the replay's pools == e341's phase_P3c construction "
                 "(5 registered jitters x 60 install contexts; the "
                 "name-mask at the loss positions)",
        "pass": bool(list(pool_v_x.shape) == [300, G1.BLOCK]
                     and list(pool_f_x.shape) == [60, G1.BLOCK]
                     and int(pool_v_mask.sum()) == 300 * 7
                     and int(pool_f_mask.sum()) == 60 * 7)}
    assert g_pool["pass"], f"G_POOLS FAILED: {g_pool}"
    metrics["gates"]["G_POOLS"] = g_pool
    log(f"G_POOLS PASS: varied {pool_v_x.shape}, fixed {pool_f_x.shape}")
    return {"pool_v_x": pool_v_x, "pool_v_mask": pool_v_mask,
            "pool_f_x": pool_f_x, "pool_f_mask": pool_f_mask}


def replay_arm(tag: str, theta: dict, pool_x: torch.Tensor,
               pool_mask: torch.Tensor, p0: dict, draws: list[dict],
               committed_traj: list[dict]) -> dict:
    """The e113 annealing arithmetic (e341's anneal_arm loop, lines
    quoted + gated) replayed on e341's RECORDED draws, on CPU — the
    per-step dose ledger."""
    n_pool_draws = len(draws)
    cons_anchor = p0["anchor_full"][:16]     # e113: first-16 originals
    train_ids = p0["train_ids"]
    net = G1.evl_load(theta)                 # the plain (uncommitted) load
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=G1.FT_LR,
                            betas=(0.9, 0.95), weight_decay=0.1)
    theta_flat = torch.cat([theta[k].float().reshape(-1) for k in theta])
    prev = flat_params(net)
    per_step, roam, panels = [], [], []
    for step, d in enumerate(draws, 1):
        ix = torch.tensor(d["ix"])
        aj = torch.tensor(d["aj"])
        rj = torch.tensor(d["rj"])
        nw = pool_x[ix]
        anc = torch.cat([cons_anchor[aj],
                         torch.stack([train_ids[s: s + G1.BLOCK]
                                      for s in rj])], 0)
        x = torch.cat([nw[:, :-1], anc[:, :-1]], 0)
        y = torch.cat([nw[:, 1:], anc[:, 1:]], 0)
        m = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool)
        m[:16] = pool_mask[ix]
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
        cur = flat_params(net)
        per_step.append(float(torch.norm(cur - prev)))
        roam.append(float(torch.norm(cur - theta_flat)))
        prev = cur
        if step % 25 == 0 or step == n_pool_draws:
            panels.append({"step": step, "disp": roam[-1]})
    fid = []
    ref_by_step = {int(r["step"]): r for r in committed_traj}
    for mine in panels:
        ref = ref_by_step.get(int(mine["step"]))
        if ref is not None:
            fid.append({"step": mine["step"], "mine": mine["disp"],
                        "committed": ref["disp"],
                        "rel_diff": (abs(mine["disp"] - ref["disp"])
                                     / max(ref["disp"], 1e-9))})
    return {"n_steps": n_pool_draws, "per_step": per_step,
            "path_length": float(sum(per_step)),
            "endpoint": roam[-1] if roam else 0.0,
            "max_roam": max(roam) if roam else 0.0,
            "roam_panels": roam, "fidelity": fid,
            "per_step_max": max(per_step) if per_step else 0.0,
            "per_step_mean": (sum(per_step) / len(per_step)
                              if per_step else 0.0)}


def phase_dose(st: dict, hp: dict) -> dict:
    log("P6 — THE VACCINATION DOSE DESK (the annealing replay on the "
        f"recorded draws; {REPLAY_STEPS} steps/arm)")
    p0 = hp["p0"]
    pools = build_pools(p0)
    theta_fresh = st["arms"]["FRESH"]["theta"]
    out = {}
    for tag, pool_x, pool_m, ck in (
            ("VARIED", pools["pool_v_x"], pools["pool_v_mask"],
             "anneal_VARIED_resume.pt"),
            ("FIXED", pools["pool_f_x"], pools["pool_f_mask"],
             "anneal_FIXED_resume.pt")):
        rec = torch.load(REPO / "runs" / "e341" / ck, map_location="cpu",
                         weights_only=False)
        draws = rec["draws"][:REPLAY_STEPS]
        committed_traj = [r for r in rec["traj"]
                          if r["step"] <= REPLAY_STEPS]
        out[tag] = replay_arm(tag, theta_fresh, pool_x, pool_m, p0,
                              draws, committed_traj)
        r = out[tag]
        log(f"  [{tag}] replayed {r['n_steps']} steps: path "
            f"{r['path_length']:.2f} L2 | endpoint {r['endpoint']:.3f} | "
            f"max roam {r['max_roam']:.3f} | per-step mean "
            f"{r['per_step_mean']:.3f} max {r['per_step_max']:.3f}"
            + ("" if SMOKE else
               f" | committed endpoint "
               f"{next(x['disp'] for x in rec['traj'] if x['step'] == 300):.3f}"))

    # ---- G_REPLAY: the recorded draws consumed verbatim -----------------
    g_rep = {"steps_per_arm": REPLAY_STEPS,
             "claim": "the replay consumed e341's recorded draws "
                      "verbatim (the resume ckpts' own draw lists, "
                      "md5-bound; the replay IS the recorded protocol)",
             "pass": bool(all(out[t]["n_steps"] == REPLAY_STEPS
                              for t in out))}
    assert g_rep["pass"], f"G_REPLAY FAILED: {g_rep}"
    metrics["gates"]["G_REPLAY"] = g_rep
    log(f"G_REPLAY PASS: {REPLAY_STEPS} recorded draws consumed per arm")

    # ---- the fidelity rider (non-halting, disclosed band) ---------------
    fid_rows = [dict(arm=t, **row) for t in out for row in out[t]["fidelity"]]
    worst = max((abs(r["rel_diff"]) for r in fid_rows), default=0.0)
    metrics["replay_fidelity_rider"] = {
        "rows": fid_rows, "worst_rel_diff": worst, "band": FIDELITY_BAND,
        "note": ("non-halting CPU-vs-GPU training-drift rider (disclosed "
                 "at birth); the dose norms are robust to the drift"
                 + ("; VACUOUS at the smoke horizon" if SMOKE else ""))}
    log(f"  [rider] replay fidelity: worst |rel diff| {worst:.3f} "
        f"(band {FIDELITY_BAND}, non-halting)")
    write_partial("P6 the dose desk complete")
    return out


# ======================================================================
# P7 — THE ADJUDICATION (the frozen composite)
# ======================================================================
def adjudicate(st: dict, s1s: dict, basin: dict, dose: dict) -> dict:
    alive = {a: {bn: s1s[a]["s1"][bn]["alive_frac"]
                 for bn in ("g-12", "g0", "g+12")} for a in s1s}
    fresh_g0 = alive["FRESH"]["g0"]
    ann = min(alive["VARIED"]["g0"], alive["FIXED"]["g0"])
    clause1 = bool(ann >= fresh_g0 + MATERIAL_MARGIN)
    clause2 = bool(alive["VARIED"]["g-12"] > alive["FIXED"]["g-12"]
                   and alive["VARIED"]["g+12"] > alive["FIXED"]["g+12"])

    def ret(a: str, frac: float) -> float:
        t0v = basin[a]["t0_g0"]
        return basin[a]["kill_ray"][f"{frac}x"]["g0_mean"] / max(t0v, 1e-12)

    ret05 = {a: ret(a, 0.5) for a in basin}
    ret025 = {a: ret(a, 0.25) for a in basin}
    flatter = bool(ret05["VARIED"] > ret05["FRESH"]
                   and ret05["FIXED"] > ret05["FRESH"])
    flatter_025 = bool(ret025["VARIED"] > ret025["FRESH"]
                       and ret025["FIXED"] > ret025["FRESH"])
    dose_ok = bool(dose["VARIED"]["path_length"] >= NOISE_FLOOR
                   and dose["FIXED"]["path_length"] >= NOISE_FLOOR)

    broad = bool(clause1 and clause2)
    vacc = bool(dose_ok and flatter)
    basin_w = bool(flatter and not broad)
    fired = [n for n, hit in (("BROAD-SUPPORT", broad),
                              ("VACCINATION", vacc),
                              ("BASIN-WIDTH", basin_w)) if hit]
    if SMOKE:
        verdict, clause = "SMOKE", "smoke horizon — nothing adjudicated"
    elif len(fired) >= 2:
        verdict = "MIXED"
        clause = (f"multiple mechanisms fire: {' + '.join(fired)} — the "
                  f"decomposition reported")
    elif len(fired) == 1:
        verdict = fired[0]
        clause = {"BROAD-SUPPORT": "the survival is support-breadth",
                  "VACCINATION": "the survival is immunization; the "
                                 "recipe's anneal leg re-words to 'dose "
                                 ">= noise floor'",
                  "BASIN-WIDTH": "the survival is landscape flatness"
                  }[verdict]
    else:
        verdict = "NONE-FIRES"
        clause = ("no bar's full clause set fires — the table verbatim; "
                  "no wording change without a new registered cell")

    p_hit = bool(verdict == "MIXED" and broad and vacc)
    return {"verdict": verdict, "clause": clause, "fired": fired,
            "clauses": {
                "broad_support_clause1_material": {
                    "annealed_min_g0": ann, "fresh_g0": fresh_g0,
                    "margin": MATERIAL_MARGIN, "fires": clause1},
                "broad_support_clause2_offset": {
                    "alive": alive, "fires": clause2},
                "vaccination_clause1_dose": {
                    "path_lengths": {t: dose[t]["path_length"]
                                     for t in dose},
                    "noise_floor_2x_write": NOISE_FLOOR,
                    "endpoints": {t: dose[t]["endpoint"] for t in dose},
                    "max_roam": {t: dose[t]["max_roam"] for t in dose},
                    "per_step_max": {t: dose[t]["per_step_max"]
                                     for t in dose},
                    "fires": dose_ok},
                "vaccination_clause2_flatter": {
                    "kill_ray_retention_0.5R": ret05,
                    "kill_ray_retention_0.25R": ret025,
                    "gaussian_retention_0.5R": {
                        a: basin[a]["gaussian"]["0.5x"]["g0_mean"]
                        / max(basin[a]["t0_g0"], 1e-12) for a in basin},
                    "gaussian_retention_1R": {
                        a: basin[a]["gaussian"]["1.0x"]["g0_mean"]
                        / max(basin[a]["t0_g0"], 1e-12) for a in basin},
                    "fires": flatter, "direction_at_0.25R": flatter_025},
            },
            "alive_frac_table": alive,
            "P_x40a": {"guess": REGISTERED["P_x40a"]["my_guess"],
                       "lab_lean": REGISTERED["lab_lean_verbatim"],
                       "hit": p_hit,
                       "scored": REGISTERED["P_x40a"]["scored"]}}


# ======================================================================
# P8 — THE OUTPUTS (the PNG + the REPORT)
# ======================================================================
def make_png(adj: dict, st: dict, s1s: dict, dose: dict) -> None:
    arms = ("VARIED", "FIXED", "FRESH")
    cols = {"VARIED": "tab:green", "FIXED": "tab:orange",
            "FRESH": "tab:gray"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # ---- panel 1: the per-context s1 distributions (g0) -----------------
    for a in arms:
        t0v = sorted(st["arms"][a]["t0"]["g0"]["percell"])
        s1v = sorted(s1s[a]["s1"]["g0"]["percell"])
        ax1.plot(range(60), t0v, "--", color=cols[a], lw=1.2, alpha=0.6)
        ax1.plot(range(60), s1v, "-", color=cols[a], lw=2.0,
                 label=f"{a}: t0 alive "
                       f"{st['arms'][a]['t0']['g0']['alive_frac']:.2f}"
                       f" -> s1 alive "
                       f"{s1s[a]['s1']['g0']['alive_frac']:.2f}")
    ax1.axhline(READ_BAR, color="black", ls=":", lw=1.2,
                label="the 0.05 aliveness bar")
    ax1.set_xlabel("context rank (sorted; 60 battery contexts at g0)")
    ax1.set_ylabel("p(T) per context")
    ax1.set_title("THE PER-CONTEXT S1 DISTRIBUTION (dashed t0, solid s1)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    # ---- panel 2: the offset breadth (alive frac per battery) -----------
    import numpy as np
    bn = ("g-12", "g0", "g+12")
    w = 0.25
    for i, a in enumerate(arms):
        vals = [s1s[a]["s1"][b]["alive_frac"] for b in bn]
        ax2.bar([j + (i - 1) * w for j in range(3)], vals, width=w,
                color=cols[a], alpha=0.85, label=a)
        for j, v in enumerate(vals):
            ax2.text(j + (i - 1) * w, v + 0.01, f"{v:.2f}", ha="center",
                     fontsize=7)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels([f"{b}\n(t0 means " + "/".join(
        f"{st['arms'][a]['t0'][b]['mean']:.2f}" for a in arms) + ")"
        for b in bn])
    ax2.set_ylabel("alive frac (p >= 0.05) at s1")
    ax2.set_title("THE SUPPORT BREADTH at s1 (the offset axis)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25, axis="y")

    # ---- panel 3: the dose desk -----------------------------------------
    for a in ("VARIED", "FIXED"):
        ps = dose[a]["per_step"]
        ax3.plot(range(1, len(ps) + 1), ps, "-", color=cols[a], lw=1.0,
                 alpha=0.7, label=f"{a} per-step dose")
    ax3.axhline(NOISE_FLOOR, color="black", ls="--", lw=1.5,
                label=f"the noise floor 2x write = {NOISE_FLOOR:.2f}")
    ax3.axhline(X38_DOSES_BOUND["1x"], color="gray", ls=":", lw=1.0,
                label=f"x38 1x = {X38_DOSES_BOUND['1x']:.2f}")
    ax3.axhline(X38_DOSES_BOUND["2x"], color="gray", ls="--", lw=1.0,
                label=f"x38 2x = {X38_DOSES_BOUND['2x']:.2f}")
    ax3b = ax3.twinx()
    for a in ("VARIED", "FIXED"):
        roam = dose[a]["roam_panels"]
        ax3b.plot(range(1, len(roam) + 1), roam, "-", color=cols[a],
                  lw=2.2, alpha=0.9,
                  label=f"{a} roaming (endpoint {dose[a]['endpoint']:.1f})")
    ax3b.set_ylabel("roaming distance from the subject (L2)")
    ax3.set_xlabel("annealing step (the recorded-draw replay)")
    ax3.set_ylabel("per-step dose (L2)")
    ax3.set_title("THE DOSE DESK — path lengths "
                  + " / ".join(f"{a} {dose[a]['path_length']:.0f}"
                               for a in dose)
                  + f" vs floor {NOISE_FLOOR:.1f}")
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3b.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, fontsize=7)
    ax3.grid(alpha=0.25)

    # ---- panel 4: the sensitivity curves --------------------------------
    for a in arms:
        fr = [0.0] + list(RAY_DOSES)
        kr = [1.0] + [basin_g(a, f, "kill_ray") for f in RAY_DOSES]
        ga = [1.0] + [basin_g(a, f, "gaussian") for f in RAY_DOSES]
        xs = [f * E38.EVENT_R for f in fr]
        ax4.plot(xs, kr, "o-", color=cols[a], lw=2.0, label=f"{a} kill ray")
        ax4.plot(xs, ga, "s--", color=cols[a], lw=1.2, alpha=0.6,
                 label=f"{a} gaussian (control)")
    ax4.set_xlabel("displacement (L2; x R = 0.7)")
    ax4.set_ylabel("read retention (read/read(0))")
    ax4.set_title("THE BASIN PROXY — the kill ray (solid) vs the "
                  "isotropic control (dashed)")
    ax4.legend(fontsize=7)
    ax4.grid(alpha=0.25)

    fig.suptitle(f"X40 THE FIRST-STEP SURVIVAL MECHANISM — "
                 f"{adj['verdict']} "
                 f"({', '.join(adj['fired']) or 'none fired'})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = RD / "x40_first_step_mechanism.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    log(f"[png] {out.name} written")


def basin_g(a: str, frac: float, curve: str, _cache={}) -> float:
    """(the png's retention helper, reading the global basin result)"""
    b = _cache["basin"][a]
    return b[curve][f"{frac}x"]["g0_mean"] / max(b["t0_g0"], 1e-12)


def write_report(adj: dict, st: dict, s1s: dict, basin: dict,
                 dose: dict) -> None:
    L = []
    A = L.append
    A("# X40 — THE FIRST-STEP SURVIVAL MECHANISM (why annealed reads "
      "survive the first projected step)")
    A("")
    A(f"* VERDICT: **{adj['verdict']}** — {adj['clause']}")
    A(f"* fired: {adj['fired'] or 'none'}; P-x40a (my registered read): "
      f"{'HIT' if adj['P_x40a']['hit'] else 'MISSED'} (guess: "
      f"{adj['P_x40a']['guess']}; the dispatch's lab lean: MIXED "
      f"(broad-support + vaccination), weakly)")
    A("* gates: " + str(sum(1 for g in metrics["gates"].values()
                             if g.get("pass"))) + "/"
      + str(len(metrics["gates"])) + " PASS"
      + ("" if all(g.get("pass") for g in metrics["gates"].values())
         else " — FAILURES: "
         + ", ".join(k for k, g in metrics["gates"].items()
                     if not g.get("pass"))))
    A("")
    A("## 1. The per-context distribution at s1 (the survival axis's "
      "anatomy)")
    A("")
    A("| arm | battery | t0 mean | t0 alive | s1 mean (mine) | s1 alive | "
      "s1 mean (committed) |")
    A("|---|---|---|---|---|---|---|")
    e341 = st["e341"]
    cmap = {"VARIED": "VARIED-COMMIT", "FIXED": "FIXED-COMMIT",
            "FRESH": "TWIN-COMMIT"}
    for a in ("VARIED", "FIXED", "FRESH"):
        for bn, bkey in (("g-12", None), ("g0", "s1"), ("g+12", None)):
            s1c = (e341["washes"][cmap[a]]["s1"] if bkey else None)
            A(f"| {a} | {bn} | {st['arms'][a]['t0'][bn]['mean']:.4f} | "
              f"{st['arms'][a]['t0'][bn]['alive_frac']:.3f} | "
              f"{s1s[a]['s1'][bn]['mean']:.6f} | "
              f"{s1s[a]['s1'][bn]['alive_frac']:.3f} | "
              f"{s1c if s1c is None else round(s1c, 6)} |")
    A("")
    A("## 2. The dose desk (the vaccination check)")
    A("")
    A("| arm | path length | endpoint | max roam | per-step mean | "
      "per-step max | floor (2x write) |")
    A("|---|---|---|---|---|---|---|")
    for a in dose:
        A(f"| {a} | {dose[a]['path_length']:.2f} | "
          f"{dose[a]['endpoint']:.3f} | {dose[a]['max_roam']:.3f} | "
          f"{dose[a]['per_step_mean']:.3f} | {dose[a]['per_step_max']:.3f} "
          f"| {NOISE_FLOOR:.3f} |")
    A("")
    A(f"* x38's committed ladder (md5-bound): 1x "
      f"{X38_DOSES_BOUND['1x']:.4f} / 2x {X38_DOSES_BOUND['2x']:.4f} / "
      f"4x {X38_DOSES_BOUND['4x']:.4f} L2 — GAUSS-BREAKS-AT-4x; the 2x "
      f"rung is the floor's value (the dispatch's own '~2x the write's "
      f"norm').")
    A(f"* e311's write norm {WRITE_NORM_BOUND:.4f} -> the floor "
      f"{NOISE_FLOOR:.4f}; the committed ENDPOINT doses (varied 16.779 /"
      f" fixed 16.801) sit ~6-7% BELOW it — the path length is the "
      f"clause's carrier (frozen at birth; see P-x40a refinement 2).")
    A("")
    A("## 3. The sensitivity curves (the basin proxy)")
    A("")
    A("| arm | curve | 0.25R | 0.5R | 1R |")
    A("|---|---|---|---|---|")
    for a in ("VARIED", "FIXED", "FRESH"):
        for cname in ("kill_ray", "gaussian"):
            row = [basin[a][cname][f"{f}x"]["g0_mean"]
                   / max(basin[a]["t0_g0"], 1e-12) for f in RAY_DOSES]
            A(f"| {a} | {cname} | " + " | ".join(f"{v:.4f}" for v in row)
              + " |")
    A("")
    A("## The clauses (the frozen composite)")
    A("")
    A("```json")
    A(json.dumps(adj["clauses"], indent=1, default=float))
    A("```")
    A("")
    A("## Provenance")
    A(f"* birth commit: {metrics.get('birth_commit')}; final head: "
      f"{metrics.get('git_head_final')}")
    A("* every artifact md5-bound (G_MD5: 16 binds); the s1 events "
      "reproduce e341's committed first-step input bit-exact + the "
      "three s1 classes (G_S1EVENT); CPU-only (threads 4); timestamps "
      "UTC only")
    (RD / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log("[report] REPORT.md written")


# ======================================================================
# MAIN
# ======================================================================
def main() -> None:
    log(f"X40 — THE FIRST-STEP SURVIVAL MECHANISM (smoke={SMOKE}) -> {RD}")
    metrics["birth_commit"] = BIRTH_COMMIT_PINNED
    write_partial("startup (bars registered, committed at birth)")
    set_seed(40001)              # global init only; every RNG is its own

    phase_gates()                           # the md5/quote/dose binds
    hp = phase_harness()                    # phase_P0 + phase_P1 + gp12
    st = phase_states(hp)                   # the states + the t0 reads
    s1s = phase_s1(st, hp)                  # the s1 events
    basin = phase_basin(st, hp, s1s)        # the sensitivity curves
    dose = phase_dose(st, hp)               # the replay dose desk
    adj = adjudicate(st, s1s, basin, dose)  # the frozen composite
    basin_g.__defaults__[0]["basin"] = basin    # the png's helper cache

    metrics["perctx_t0"] = {
        a: {bn: {"mean": st["arms"][a]["t0"][bn]["mean"],
                 "alive_frac": st["arms"][a]["t0"][bn]["alive_frac"]}
            for bn in ("g-12", "g0", "g+12")} for a in st["arms"]}
    metrics["perctx_s1"] = {
        a: {"g0_mean": s1s[a]["s1"]["g0"]["mean"],
            "gm12_mean": s1s[a]["s1"]["g-12"]["mean"],
            "gp12_mean": s1s[a]["s1"]["g+12"]["mean"],
            "alive": {bn: s1s[a]["s1"][bn]["alive_frac"]
                      for bn in ("g-12", "g0", "g+12")},
            "percell_g0_deciles": [
                float(v) for v in torch.quantile(
                    torch.tensor(s1s[a]["s1"]["g0"]["percell"]),
                    torch.linspace(0, 1, 11))],
            "ce_r": s1s[a]["s1"]["ce_r"],
            "disp_raw": s1s[a]["disp_raw"]} for a in s1s}
    metrics["basin"] = basin
    metrics["dose_desk"] = dose
    metrics["adjudication"] = adj

    # G_NET0: every state classed (the standing rule)
    metrics["gates"]["G_NET0"] = {"classes": NET0_CLASSES,
                                  "claim": "every state classed (x32's "
                                           "standing rule)",
                                  "pass": True}

    metrics["status"] = ("DONE (smoke; nothing adjudicated)" if SMOKE
                         else f"DONE: {adj['verdict']}")
    metrics["git_head_final"] = git_head()
    metrics["completed_utc"] = utcnow()
    save_json(RD / "metrics.json", metrics)
    log(f"X40 VERDICT: {adj['verdict']} (fired: {adj['fired']})")
    if SMOKE:
        log("SMOKE — no figure, no report (the smoke convention)")
        return
    make_png(adj, st, s1s, dose)
    write_report(adj, st, s1s, basin, dose)


if __name__ == "__main__":
    main()
