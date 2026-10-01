"""E197 — THE ALIVE WINDOW (T158's discriminating follow-on: does the
flight structure FORM when the organism stays alive past its first step?).

WHY (T158, verbatim): "THE FLIGHT CONCENTRATION REQUIRES A LIVE
MID-FLIGHT STATE: the support cannot flee somewhere if it is already
dead ... BOTH dynamic effects (the pursuit and the flight) live in the
alive window ... the discriminating cell is named (a lineage alive past
t=1 — smaller step or stronger fact)." Organism 1 (alive mid-flight,
theta_1 0.679) shows BOTH dynamic effects — the recomputation bonus
(e194: the k=1 sign path killing at 1.7496 below its static edge
2.2699, a 23% inversion) and the flight concentration (e195: the flight
ray killing at 0.3875 vs 2.2699, ratio 0.171). Organism 2 (dead at t=1,
theta_1 0.0068) shows NEITHER (e193/e196: path 0.5257 vs static 0.5252
— +0.1%; flight ray its SOFTEST direction, ratio 2.585). THE QUESTION
THIS CELL OWNS: is the alive window the CAUSE? Make a lineage that
stays alive past t=1 and read its flight structure.

THE CELL (eval-only CPU, minutes): on the e193 f2 root (its committed
state + gates VERBATIM, e196's port):
  (1) THE HALF-STEP WASH — the same seed-10902 stream, the same k=1
      sign direction machinery (post-clip refresh gradients, fp64-norm
      sign_update), the per-step L2 at a SUB-STEP of the measured
      STEP_L2 0.9164. THE SUB-STEP CHOICE (registered BEFORE the walk,
      the dispatch's own instruction): a frozen ladder {1/2, 1/3, 1/4,
      1/6, 1/8} x STEP_L2; static first-landing probes theta_0 - s*u0
      (one primary-battery read each); pick the LARGEST s whose probe
      reads strictly above the 0.27 bar — the dispatch's half step
      (0.4582) is expected to pass (its landing sits ~0.41 on the
      committed terrain); the quarter step (0.2291) is the named
      fallback. Walk to t=4 minimum (step cap 8, D target 5.0) with
      per-step ruler reads (primary g-4 + co-rulers incl. g-12, the
      dispatch's "per-step g-12 reads") — the alive window confirmed;
  (2) THE FLIGHT READ on this alive lineage: its u0/u1/u2 sign rays
      (u1 = sign of the refresh gradient AT THE ALIVE theta_1 on batch
      2 — THE INTERVENTION: same stream, same machinery, only the step
      size changed), static graded jumps from its root (D grid
      0.05..3.00 step 0.05, e195's grid VERBATIM; dual currency; its
      own rulers); the RECOMPUTATION CHECK (its k=1 sub-step path kill
      vs its static u0 edge — the bonus present?); the THETA_1 PANEL
      (alive this time — the rays from ITS one-stepped state; kill-Ds
      resolvable, reported verbatim, never a bar).

REGISTERED BARS (frozen here, before compute; the dispatch's
registration VERBATIM; no bar shopping — adjudicate against exactly
this):
  - ALIVE-WINDOW-CAUSES: "fires if the half/quarter-step lineage
    (alive at theta_1) shows the recomputation bonus and/or the flight
    concentration (its u1 killing materially below its u0 edge, ratio
    < 0.7) — the alive window is the enabling condition; the two
    organisms' asymmetry explained causally."
  - ALIVE-BUT-NO-FLIGHT: "fires if the lineage is alive past t=1 but
    shows neither effect — the alive window is necessary but not
    sufficient; something else of organism 1 carries the dynamics;
    reported as the honest fork."
  - GRADED: "any mix — the profiles verbatim."

OPERATIONALIZATIONS (frozen before compute; they fix the clauses, they
do not move the bars):
  * "alive at theta_1" = the sub-step walk's step-1 endpoint primary
    (g-4) battery read > 0.27 (guaranteed by the pick rule to fp32
    texture; asserted as the G_ALIVE gate — a failure there is a
    control failure, abort).
  * the sub-step choice = the frozen ladder {1/2, 1/3, 1/4, 1/6, 1/8}
    x measured STEP_L2, static first-landing probes on the primary
    ruler, LARGEST s with read > 0.27 wins; the probe table + the pick
    are written to metrics.json BEFORE the walk runs.
  * "the recomputation bonus" = the k=1 sub-step sign walk's densified
    kill D <= 0.85 * D_kill(root, u0) (the day's 15% materiality
    margin — e194's inversion was 23%, e195's FLEEING-IS-LETHAL froze
    the same 0.85), both present. If the walk NEVER kills within its
    horizon while its cumulative D passes the static edge (a path safer
    than its own static ray), the bonus is RESOLVED-ABSENT (direction
    known); if the walk neither kills nor reaches the static edge's D,
    unresolved (reported verbatim, kills the clause).
  * "the flight concentration (its u1 killing materially below its u0
    edge, ratio < 0.7)" = D_kill(root, u1_alive) <= 0.70 *
    D_kill(root, u0), both present. If u1_alive fails to kill within
    the 3.0 window (a ray softer than its own u0 edge), RESOLVED-ABSENT.
  * the kill convention = each family profile's FIRST primary-ruler
    (g-4 install-60 battery mean p(Z)) <= 0.27 downcrossing on the
    static grid, linear-in-D interpolated (e192/e194/e195/e196's
    convention); None = no downcrossing within [0, 3.0].
  * rays: u_t = sign(g_t)/||sign(g_t)|| (e192's fp32-norm construction)
    with g_t the licensed stream's t=0/1/2 post-clip batch refresh
    gradients along the sub-step k=1 sign walk (seed 10902, e193's draw
    order, step size s*); anchors = theta_0 (the e157_f2 root) and
    theta_1 (the SUB-STEP walk's step-1 endpoint — alive by G_ALIVE).
  * composite order frozen: ALIVE-WINDOW-CAUSES -> ALIVE-BUT-NO-FLIGHT
    -> GRADED. CAUSES fires iff alive-at-theta_1 AND (bonus OR
    concentration). NO-FLIGHT fires iff alive-at-theta_1 AND bonus
    resolved-absent AND concentration resolved-absent. Everything else
    (incl. any unresolved clause with nothing fired) is GRADED, the
    tables verbatim.

PRE-DISPATCH CHECKS (Rule 12; asserted BEFORE any adjudication): e193's
provenance gates REUSED VERBATIM via e196's port (G_NAMEFREE, G_SPLICE,
G_BATTERY at the e157 dial's seven geometries, G_ANCHOR vs e185's
stored bank, G_ROOT — the root's dial reproduces e157's committed cells
BIT-TIGHT + the flat md5 matches e193's committed root identity, G_T0 —
the step-1 batch md5 + forward CE + the MEASURED step-1 L2, G_S1CK,
G_STREAM steps 1..4, G_DIRCK — e193's committed g-ray checkpoint,
G_SIGNRAY — u0 rebuilt md5 == e193's committed R2_SIGN md5) PLUS G_REPRO
(the FULL-STEP k=1 walk rebuilt and gated vs e193's committed a_sign
step-1 row + densified kill bracket — the dead-at-t=1 baseline
re-anchored in THIS process) PLUS the new-arm gates: G_ALIVE (the
sub-step walk's step-1 read > 0.27 AND matches its registered static
probe within the 2e-3 texture class), G_ROOTPROF (the root-u0 family
reproduces e193's committed R2_SIGN rows at ALL 16 shared Ds + ce_r +
the D=STEP_L2 landing vs the committed walked step-1 kill read,
texture class) and G_TH1PROF (the ALIVE theta_1 families' D=0 rows
reproduce the walk's step-1 read bit-class; the th1-u1 family's D=s*
landing reproduces the walk's step-2 read, texture class). WHAT EACH
ARM GUARANTEES: NOTHING — the sub-step lineage is a COUNTERFACTUAL wash
(the organism's natural AdamW step at this lr has L2 0.9164; the size
is the experimenter's, the direction machinery the wash's own); every
family is a static graded jump at lethal scale from a state that could
kill anywhere; that openness is the point.

COMPUTE ENVELOPE: CPU-ONLY (CUDA_VISIBLE_DEVICES=-1 forced before
torch; the GPU is another agent's, never claimed), torch threads 8
(e193's gate convention — the committed dial + profiles reproduce
bit-exact under it), sequential phases, every phase < 180 s,
PROGRESSIVE partial metrics.json writes after every phase (the outage
lesson), n=1, single stream seed 10902 lineage, organism 2 of 2.

Outputs: runs/e197/{metrics.json, e197_alive_window.png,
e197_kill_summary.png}. No NOTES/THINKING/QUEUE/STATE edits (the
coordinator folds).

Run:  cd lab && python e197_alive_window.py    (E197_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import os as _os
_os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU-ONLY, forced (e193's convention)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                          # noqa: E402

torch.set_num_threads(8)                              # e193's gate convention (G_ROOT bit-tightness)

import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT,          # noqa: E402
                    run_dir, save_json)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import numpy as np                                     # noqa: E402 (plots)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("E197_SMOKE") == "1"
CPU = torch.device("cpu")
assert not torch.cuda.is_available(), "e197 is CPU-only by dispatch"

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256                       # training-window block (the e098 line's own convention)
ROOT_CK = "e157_f2_consolidated.pt"   # THE FAMILY-2 CONSOLIDATED ROOT (e157's, committed)
S1_CK = "e157_f2_neutral_s1.pt"       # the committed wash step-1 checkpoint (the G_T0/G_S1CK anchor)
DIR_CK = "e193_f2_static_dir_u.pt"    # e193's committed g-ray direction checkpoint
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
E157_METRICS = E43.REPO / "runs" / "e157" / "metrics.json"
E193_METRICS = E43.REPO / "runs" / "e193" / "metrics.json"
E195_METRICS = E43.REPO / "runs" / "e195" / "metrics.json"
E196_METRICS = E43.REPO / "runs" / "e196" / "metrics.json"

# ---- the family-2 net (e098 s4305 line: 4L/4H/128d/512-ctx, 873,472 params) ----
F2_CFG = Cfg(vocab=65, n_layer=4, n_head=4, n_embd=128, block_size=512)
F2_PARAMS = 873_472

# ---- rulers (e193's, frozen before its compute; carried verbatim) -----------------
RULER_J = -4                     # PRIMARY: g-4 install-60 battery (max committed root read)
CO_RULERS_J = (12, -12, 0)       # novel-side analog / e192-verbatim / install dial
READ_GEOS = (-12, -8, -4, 0, 4, 8, 12)   # the e157 dial's seven read geometries (gates)

# ---- the run envelope (dispatch-frozen) ------------------------------------------
FREEZE_SEED = 10902               # the wash-stream seed (e176n/e185/e193 family-2)
LR_ADAMW = 1e-3                   # the t=0 recipe (e157/e193's wash; the ray arms have NO lr)
ANCH_BS, RAND_BS = 16, 16         # batch 32 = 16 anchor draws + 16 random
SHUT_BAR = 0.27                   # e185's kill bar (DISSOLVE) — absolute, verbatim
D_GRID = [round(0.05 * i, 2) for i in range(1, 61)]   # 0.05..3.00 (e195's grid VERBATIM)
CE_EVERY = 0.2                    # ce_r at D in {0.2, 0.4, ..., 3.0} (e195's decimation)
FLIGHT_RATIO_BAR = 0.70           # frozen: the dispatch's verbatim "ratio < 0.7"
BONUS_RATIO_BAR = 0.85            # frozen: the day's 15% materiality margin (e194/e195 class)
EXT_STEPS = 1                     # read-only steps past the sub-step walk's kill (the u2 front)
WALK_CAP = 8                      # step cap (walk to t=4 minimum; cap 8 gives margin)
WALK_D_TARGET = 5.0               # e193's target convention
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05
G_REPRO_TOL = 1e-9                # bit-class: same code path, same device, fp32 texture
G_STATIC_TOL = 1e-3               # e193's committed R2 rows (same code path; expect 0.0)
G_LANDING_TOL = 2e-3              # fp32-norm ray landing vs walked fp64-norm step (e195's class)
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
RMS_DENOM = 934.597239456655      # sqrt(873472) — organism 2's per-coordinate RMS currency
# the frozen sub-step ladder (registered BEFORE the walk; the dispatch's
# half step primary + quarter step fallback + interpolating/deeper rungs)
SUB_LADDER = (1 / 2, 1 / 3, 1 / 4, 1 / 6, 1 / 8)
if SMOKE:                         # shakedown trims (documented in deviations)
    D_GRID = [0.05, 0.15, 0.25, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    CE_EVERY = None

# ---- e157's committed lineage reads (e193's copies, verbatim) ----------------------
E157_DIAL = {
    "gm12": 0.19826222956180573, "g-8": 0.8519253134727478,
    "g-4": 0.8872273564338684, "g0": 0.5784125924110413,
    "g+4": 0.8801683187484741, "g+8": 0.8754011988639832,
    "gp12": 0.5911492705345154, "ce_r": 1.9504578113555908,
}
E157_WASH_S1 = {"corpus_ce": 1.7748003005981445,
                "gm12": 0.001129803480580449}
ROOT_FLAT_MD5 = "73820c546e0f8d7b22c727e1d6f23fbc"
E193_STEP_L2 = 0.9164195656776428          # THIS organism's measured AdamW step-1 L2
E193_ASIGN_S1 = {                          # e193's committed a_sign step-1 row (G_REPRO)
    "ce_batch": 1.7748003005981445,
    "cum_disp": 0.9160122275352478,
    "step_disp": 0.916012167930603,
    "preclip_gnorm": 2.299248695373535,
    "gm": 0.006826397497206926,
    "frac_argmax_z": 0.0,
}
E193_ASIGN_STOP = {                        # e193's committed a_sign stop (G_REPRO)
    "kind": "kill", "step": 1,
    "gm_at_kill": 0.006826397497206926,
    "D_kill_raw": 0.6421935368466083,
    "D_kill": 0.5257438700572483,
    "dens": [(0.2, 0.785190224647522, 0.1833570897579193),
             (0.4, 0.6062278151512146, 0.3667244017124176),
             (0.6, 0.21901635825634003, 0.5498566627502441),
             (0.8, 0.03626079857349396, 0.7333946824072292)],
}
E193_KILLS = {"g": 0.2, "sign": 0.58}       # e193's committed grid kills (primary ruler)
E193_UG_MD5 = "46e71718b2ee67f2cd87f32372652baf"
E193_USIGN_MD5 = "9395918d36425e65248dbd93b6fd50bc"
E193_R2_SPOTS = {                           # hard-bound spot anchors of the committed profile
    0.5: {"gm": 0.32222333550453186, "ce_r": 2.2278640270233154},
    1.5: {"gm": 0.0006007259362377226, "ce_r": 3.542923927307129},
}
# ---- organism 1's committed dynamics (e194/e195 — the alive-window side) -----------
E195_ROOT_DKILLS = {                        # e195's committed organism-1 root panel
    "root_u0": 2.269916581032063,
    "root_u1": 0.3875040789853107,
    "root_u2": 0.8979436419840907,
}
E194_PATH_DKILL = 1.749640490742179         # e194's committed k=1 sign-path kill (org 1)
E194_STATIC_EDGE = 2.269916581032063        # e194's committed refined static edge (org 1)
ORG1_TH1_ALIVE = 0.6785961389541626         # org 1's theta_1 alive read (opt2 s1, e195's gate)
# ---- organism 2's committed DEAD-lineage flight (e196 — the contrast arm) ----------
E196_ROOT_DKILLS = {
    "root_u0": 0.5252331635450007,
    "root_u1": 1.3575583149916892,
    "root_u2": 0.46025487385682545,
}
E196_U1_MD5 = "a6fb6081b2da3d99d9ddda0287852066"   # the DEAD-lineage flight ray
E196_U2_MD5 = "d8f4cdc1f90d407335846c0e869ae969"
E196_TH1_GM = 0.006826397497206926                  # org 2's dead theta_1 (e193 s1)
E170_BANK_STARTS = [825650, 361746, 954106, 856844, 615070, 335787,
                    329879, 96582, 253994, 341757, 608690, 127521,
                    228843, 834931, 15774, 408603]
E185_XHASH = {                    # per-step input-batch md5 (seed-10902 stream; net-independent)
    1: "1ea27bffde6c4a53be8badf5ab453d64",
    2: "1d6f0e55cc6a25ece947d2040528225e",
    3: "b5c0b670270406a94aca63071b051468",
    4: "cdccea0c413e603dc52d1873e37b9844",
}

REGISTERED_BARS = {
    "ALIVE_WINDOW_CAUSES": "ALIVE-WINDOW-CAUSES: \"fires if the "
        "half/quarter-step lineage (alive at theta_1) shows the "
        "recomputation bonus and/or the flight concentration (its u1 "
        "killing materially below its u0 edge, ratio < 0.7) — the alive "
        "window is the enabling condition; the two organisms' asymmetry "
        "explained causally.\"",
    "ALIVE_BUT_NO_FLIGHT": "ALIVE-BUT-NO-FLIGHT: \"fires if the lineage "
        "is alive past t=1 but shows neither effect — the alive window is "
        "necessary but not sufficient; something else of organism 1 "
        "carries the dynamics; reported as the honest fork.\"",
    "GRADED": "GRADED: \"any mix — the profiles verbatim.\"",
    "operationalizations": "static graded jumps theta_D = anchor - D*u "
        "for u in {u0, u1, u2} = sign(g_t)/||sign(g_t)|| (e192's fp32-norm "
        "construction) with g_t the licensed stream's t=0/1/2 post-clip "
        "batch refresh gradients along the SUB-STEP k=1 sign walk (seed "
        "10902, e193's draw order, step size s* from the frozen ladder "
        "{1/2, 1/3, 1/4, 1/6, 1/8} x measured STEP_L2 0.9164 — the "
        "LARGEST rung whose static first-landing probe theta_0 - s*u0 "
        "reads > 0.27 on the primary ruler, the table registered BEFORE "
        "the walk); anchors = theta_0 (the e157_f2 root, gated "
        "G_ROOT/G_DIRCK) and theta_1 (the sub-step walk's step-1 "
        "endpoint, alive by the G_ALIVE gate); grid D in {0.05..3.00 "
        "step 0.05} + D=0 anchor rows + landing gate points; ruler = "
        "e193's primary g-4 install-60 battery, SHUT 0.27 (co-rulers "
        "everywhere, never adjudicated); KILL = the profile's first "
        "primary-ruler <= 0.27 downcrossing, linear-in-D interpolated; "
        "'the recomputation bonus' = the sub-step walk's densified kill "
        "D <= 0.85 * D_kill(root, u0) (the day's 15% materiality margin, "
        "e194's 23% inversion the anchor; a walk that never kills while "
        "passing the static edge's D = resolved-absent); 'the flight "
        "concentration' = D_kill(root, u1) <= 0.70 * D_kill(root, u0) "
        "(the dispatch's verbatim 0.7; a u1 that never kills within 3.0 "
        "= resolved-absent); ALIVE-at-theta_1 = the walk's step-1 "
        "primary read > 0.27 (G_ALIVE; failure there = control failure, "
        "abort); composite order frozen ALIVE-WINDOW-CAUSES -> "
        "ALIVE-BUT-NO-FLIGHT -> GRADED: CAUSES fires iff alive AND "
        "(bonus OR concentration); NO-FLIGHT fires iff alive AND bonus "
        "resolved-absent AND concentration resolved-absent; everything "
        "else GRADED verbatim.",
    "registered_prediction": "T158's mechanism hypothesis (the live-state "
        "condition) implies ALIVE-WINDOW-CAUSES — registered WITH the "
        "competing prior stated honestly: the sub-step lineage is a "
        "COUNTERFACTUAL wash (the natural step's size 0.9164 is what "
        "killed it; the experimenter halves it), so if the dynamics "
        "belong to the NATURAL trajectory's biography rather than to the "
        "alive window per se, ALIVE-BUT-NO-FLIGHT is the honest outcome; "
        "the cell discriminates exactly these two readings; no bar "
        "shopping either way.",
    "registration": "the dispatch's registration IS the registration (the "
        "bars quoted verbatim in the module docstring and here, frozen "
        "before compute). Adjudicate against exactly this; no bar shopping.",
}

deviations: list[str] = [
    "THE RULER CALL (e193's, carried verbatim): the e192-verbatim g-12 "
    "ruler reads 0.1983 at this root — UNDER the 0.27 bar at D=0. PRIMARY "
    "RULER = g-4 install-60 battery (root read 0.8872); co-rulers g+12, "
    "g-12, g+0 on every row and every walk step (the dispatch's 'per-step "
    "g-12 reads' ride as co-rulers), never adjudicated.",
    "THE SUB-STEP DEVIATION (this cell's load-bearing intervention): the "
    "wash's natural AdamW step at lr 1e-3 has measured L2 0.9164, which "
    "exceeds BOTH static edges (g 0.20 / sign 0.53-0.58) and kills the "
    "organism at t=1. This cell's walk keeps the direction machinery "
    "VERBATIM (k=1 post-clip sign refresh on the licensed seed-10902 "
    "stream) and replaces ONLY the size with s* from the frozen ladder. "
    "The lineage is therefore a counterfactual wash, not the natural "
    "trajectory — disclosed in every read that depends on it (u1, u2, "
    "theta_1, the walk kill).",
    "THE SUB-STEP CHOICE registered BEFORE the walk (the dispatch's own "
    "instruction): ladder {1/2, 1/3, 1/4, 1/6, 1/8} x STEP_L2 probed "
    "statically (one primary battery read per rung at theta_0 - s*u0), "
    "largest rung reading > 0.27 wins; the probe table + pick are "
    "written to metrics.json (progressive write) before the walk phase "
    "runs.",
    "walk_sub is e193's run_arm (e196's port) VERBATIM in its arithmetic "
    "+ a step-size parameter + ADDITIVE stashes only (per-step co-ruler "
    "reads, g_fronts, endpoints) + a read-only 1-step post-kill "
    "continuation (the u2 front if the walk dies; e196's precedent, "
    "disclosed); no arithmetic path changes.",
    "No new checkpoints (e195's convention): the rays and anchors are "
    "bit-rebuildable from the licensed stream + e193's committed "
    "artifacts; the new direction md5s (u1/u2 of the ALIVE lineage) are "
    "registered in metrics.json, alongside e196's committed DEAD-lineage "
    "md5s for the contrast.",
    "ce_r (second currency) decimated to 0.2-multiples + D=0 (e195's "
    "decimation); the g-4 battery is the ruler and is read at EVERY grid "
    "point, with the three co-rulers on every row; dual displacement "
    "currency (D and per-coordinate RMS / 934.597) on every row.",
    "Kill-D resolution disclosed: at sub-step s* the walk's every-step "
    "read spacing is ~s* in cumulative D and the densified bracket "
    "resolves ~s*/5 — the recomputation-bonus read is coarser than "
    "e194's full-step ladder by the same factor; reported with the "
    "bracket rows verbatim.",
    "Light reads only (batteries + ce_r + fact-battery matched-point "
    "cosines in dual form); CPU-ONLY, threads 8 (e193's gate "
    "convention), n=1, seed lineage 10902, organism 2 of 2.",
    "Smoke mode trims: grid {0.05, 0.15, 0.25, 0.4, 0.5, 0.75, 1.0, 1.5, "
    "2.0, 3.0}, ce at anchors only (verdict stamped SMOKE; nothing "
    "adjudicated).",
]


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/e196_flight_replicate.py VERBATIM (whose own provenance
# is lab/e193_organism_replicate.py + lab/e195_rotated_ray.py — the e176n
# lineage). Copied rather than imported to own the device policy and the
# bit-exact arithmetic.

def load_f2(path) -> TinyGPT:
    m = TinyGPT(F2_CFG)
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["model"] if isinstance(st, dict) and "model" in st else st
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def battery_cell(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> dict:
    """e068/e113/e120/e151 battery on CPU: p(Z) at the last position."""
    net.eval()
    pzs, amax = [], 0.0
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs.append(pr[:, zid])
        amax += float((lg[:, -1].argmax(-1) == zid).float().sum())
    p = torch.cat(pzs)
    return {"mean_pz": float(p.mean()), "median_pz": float(p.median()),
            "std_pz": float(p.std()),
            "frac_pz_ge_0.5": float((p >= 0.5).float().mean()),
            "frac_argmax_z": amax / ids.shape[0]}


@torch.no_grad()
def ce_fixed_cpu(net: TinyGPT, x, y, bs=64) -> float:
    net.eval()
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        _, loss = net(x[i:i + bs], y[i:i + bs])
        tot += float(loss.item()) * len(x[i:i + bs])
        n += len(x[i:i + bs])
    return tot / max(n, 1)


def val_windows(val_ids, val_text, n, seed, block=256):
    """e065 val_windows verbatim: name-free val-split windows."""
    g = torch.Generator().manual_seed(seed)
    out_x, out_y = [], []
    tries = 0
    while len(out_x) < n and tries < 500 * n:
        i = int(torch.randint(len(val_ids) - block - 1, (1,), generator=g))
        txt = val_text[i: i + block + 1]
        tries += 1
        if "ZEPHYRA" in txt or "ZEPH" in txt:
            continue
        out_x.append(val_ids[i: i + block])
        out_y.append(val_ids[i + 1: i + 1 + block])
    return torch.stack(out_x), torch.stack(out_y)


def flat_params(net: TinyGPT) -> torch.Tensor:
    """The fp32 flat parameter vector (net.parameters() order; 873,472)."""
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()]).clone()


def load_flat(net: TinyGPT, flat: torch.Tensor) -> None:
    """Copy a flat vector back into parameters (the static-jump loader)."""
    i = 0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            p.copy_(flat[i:i + n].view_as(p))
            i += n


def flat_md5(net: TinyGPT) -> str:
    return hashlib.md5(flat_params(net).numpy().tobytes()).hexdigest()


def cos64(a: torch.Tensor, b: torch.Tensor) -> float:
    """fp64 cosine (the chart's estimator precision)."""
    a64, b64 = a.double(), b.double()
    return float(torch.dot(a64, b64)
                 / (torch.norm(a64) * torch.norm(b64) + 1e-30))


def sign_update(g: torch.Tensor, step_l2: float):
    """e193/opt2's sign_update VERBATIM: delta = -step_l2 * sign(g)/||sign(g)||
    (zeros stay zero; the support norm in fp64 — the matched-L2 exactness)."""
    s = torch.sign(g)
    nrm = float(torch.norm(s.double()))
    assert nrm > 0, "sign direction is zero — the arm is undefined"
    return -step_l2 * (s / nrm)


def fact_grad(net: TinyGPT, ids: torch.Tensor, zid: int, bs=30) -> torch.Tensor:
    """THE ALIGNMENT READ (e195's fact_grad, e196's port): gradient of the
    PRIMARY battery's mean log p(Z) readout at the eval twin's current
    weights. NEGATIVE cos(-u, grad) convention: a NEGATIVE cos(-u, grad)
    means the jump ray u is ANTI-ALIGNED with the fact's own gradient.
    Consumes no RNG; run on the eval twin."""
    net.zero_grad(set_to_none=True)
    sums = []
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        sums.append(F.log_softmax(lg[:, -1], -1)[:, zid].sum())
    F_obj = torch.stack(sums).sum() / ids.shape[0]
    F_obj.backward()
    g = torch.cat([p.grad.detach().reshape(-1) for p in net.parameters()])
    net.zero_grad(set_to_none=True)
    return g


# ------------------------------------------------------------------ kill edges

def interp_d_kill(v0, v1, d0, d1):
    """Linear-in-D interpolation of the 0.27 crossing inside the bracket."""
    if v0 <= SHUT_BAR:
        return float(d0)
    return float(d0 + (v0 - SHUT_BAR) / (v0 - v1) * (d1 - d0))


def dens_d_kill(v0, v1, d0, d1, dens):
    """The densified bracket (opt1c/e193's convention; this value gates)."""
    pts = [(d0, v0)] + [(r["D"], r["gm"]) for r in dens] + [(d1, v1)]
    for i in range(1, len(pts)):
        a, b = pts[i - 1], pts[i]
        if b[1] <= SHUT_BAR and a[1] > SHUT_BAR:
            return interp_d_kill(a[1], b[1], a[0], b[0])
    return interp_d_kill(v0, v1, d0, d1)


def profile_d_kill(rows):
    """First 0.27 downcrossing (interpolated) + any upcrosses (context).
    A dead anchor (row D=0 already <= bar) has no downcrossing unless the
    profile first resurrects above the bar — an upcross then a later
    downcross WOULD register; both reported verbatim."""
    edge = None
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        if b["gm"] <= SHUT_BAR and a["gm"] > SHUT_BAR:
            edge = interp_d_kill(a["gm"], b["gm"], a["D"], b["D"])
            break
    ups = [(rows[i - 1]["D"], rows[i]["D"]) for i in range(1, len(rows))
           if rows[i]["gm"] > SHUT_BAR and rows[i - 1]["gm"] <= SHUT_BAR]
    return edge, ups


# ------------------------------------------------------------------ the walk

def walk_sub(net0, anchor_neutral, train_ids, itos, primary_ids, coruler_ids,
             zid, theta0, step_l2, root_gm, step_cap=WALK_CAP,
             d_target=WALK_D_TARGET, extend_past_kill=EXT_STEPS, tag="sub"):
    """e193's run_arm (a_sign; e196's walk_rebuild port) VERBATIM in its
    arithmetic + a step-size parameter + ADDITIVE stashes (per-step
    co-ruler reads, g_fronts, endpoints) + a read-only post-kill
    continuation. Every step refreshes the front (k=1); every-step primary
    battery + co-rulers; densified kill bracket at the killing step."""
    net = copy.deepcopy(net0)
    net.train()
    evl_a = copy.deepcopy(net0)
    gen = torch.Generator().manual_seed(FREEZE_SEED)
    step, kill = 0, None
    traj, x_hashes = [], {}
    g_fronts, endpoints = [], {}
    max_l2_dev, zeph = 0.0, 0
    prev_flat, prev_d = theta0, 0.0
    t0a = time.time()
    while True:
        step += 1
        post_kill = kill is not None
        if post_kill and step > kill["step"] + extend_past_kill:
            break
        if step > step_cap and kill is None:
            kill = {"kind": "cap", "step": step - 1,
                    "reason": f"step cap {step_cap} alive",
                    "final_D": traj[-1]["cum_disp"],
                    "final_gm": traj[-1]["gm"]}
            break
        aj = torch.randint(16, (ANCH_BS,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                           generator=gen)
        anc = anchor_neutral[aj]
        rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj])
        for w in rnd:                    # name-free VERIFY (no-op; hard-fail)
            txt = "".join(itos[int(c)] for c in w[:64]) + \
                  "".join(itos[int(c)] for c in w[192:])
            if "ZEPH" in txt:
                zeph += 1
        x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
        y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
        x_hashes[step] = hashlib.md5(
            x.contiguous().numpy().tobytes()).hexdigest()
        if step in E185_XHASH:
            assert x_hashes[step] == E185_XHASH[step], \
                f"step-{step} batch md5 diverged from the licensed stream"
        logits_a, _ = net(x)
        loss = F.cross_entropy(logits_a.reshape(-1, logits_a.shape[-1]),
                               y.reshape(-1))
        net.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
        g_t = torch.cat([p.grad.detach().reshape(-1)
                         for p in net.parameters()]).clone()  # post-clip
        delta = sign_update(g_t, step_l2)
        l2dev = abs(float(torch.norm(delta.double())) - step_l2)
        assert l2dev < 1e-5, f"per-step L2 dev {l2dev:.2e}"
        max_l2_dev = max(max_l2_dev, l2dev)
        cur = prev_flat + delta
        load_flat(net, cur)
        cum_disp = float(torch.norm(cur - theta0))
        evl_a.load_state_dict({k_: v.detach().cpu().clone()
                               for k_, v in net.state_dict().items()})
        evl_a.eval()
        gz = battery_cell(evl_a, primary_ids, zid)
        row = {"step": step, "post_kill": post_kill,
               "ce_batch": float(loss.item()),
               "cum_disp": cum_disp, "step_disp": float(torch.norm(delta)),
               "l2_dev": l2dev, "preclip_gnorm": gnorm,
               "gm": gz["mean_pz"], "frac_argmax_z": gz["frac_argmax_z"]}
        for j in coruler_ids:                     # ADDITIVE per-step co-rulers
            row[f"g{j:+d}"] = battery_cell(evl_a, coruler_ids[j],
                                           zid)["mean_pz"]
        traj.append(row)
        g_fronts.append(g_t.clone())               # ADDITIVE stash
        endpoints[step] = cur.clone()              # ADDITIVE stash
        prev_flat, prev_d = cur, cum_disp
        log(f"  [{tag} s{step}]{' POSTKILL' if post_kill else ''} "
            f"g-4 {row['gm']:.6f} g-12 {row.get('g-12', float('nan')):.6f} "
            f"D {cum_disp:.6f} ce {row['ce_batch']:.6f}")
        if kill is None and gz["mean_pz"] <= SHUT_BAR:
            dens = []
            for f_ in (0.2, 0.4, 0.6, 0.8):
                pt_flat = prev_flat + (f_ - 1.0) * delta  # f along the step
                load_flat(evl_a, pt_flat)
                evl_a.eval()
                gzd = battery_cell(evl_a, primary_ids, zid)
                dens.append({"f": f_, "gm": gzd["mean_pz"],
                             "D": float(torch.norm(pt_flat - theta0))})
            prev_row = traj[-2] if len(traj) >= 2 else \
                {"gm": root_gm, "cum_disp": 0.0}   # e193's fallback VERBATIM
            d_raw = interp_d_kill(prev_row["gm"], row["gm"],
                                  prev_row["cum_disp"], cum_disp)
            d_dens = dens_d_kill(prev_row["gm"], row["gm"],
                                 prev_row["cum_disp"], cum_disp, dens)
            kill = {"kind": "kill", "step": step,
                    "gm_at_kill": row["gm"], "D_kill_raw": d_raw,
                    "D_kill": d_dens, "dens": dens,
                    "reason": "primary ruler <= SHUT at an every-step read"}
            log(f"  [{tag}] KILL at s{step}: D_kill(dens) {d_dens:.6f}")
        elif kill is None and cum_disp >= d_target:
            kill = {"kind": "target", "step": step,
                    "reason": f"D {cum_disp:.4f} >= D_TARGET {d_target} alive",
                    "final_D": cum_disp, "final_gm": row["gm"]}
    net.eval()
    assert zeph == 0, "name token leaked into a window"
    return {"traj": traj, "stop": kill, "x_hashes": x_hashes,
            "max_l2_dev": max_l2_dev, "g_fronts": g_fronts,
            "endpoints": endpoints, "seconds": round(time.time() - t0a, 1)}


# ------------------------------------------------------------------ profiles

def static_profile(evl, anchor_flat, u, dgrid, primary_ids, coruler_ids,
                   zid, r_eval_xy, ce_ds):
    """Static graded jumps theta_D = anchor - D*u (e192's placement
    VERBATIM), every point read on the primary ruler + the three
    co-rulers; ce_r (second currency) at the frozen decimated Ds; dual
    displacement currency on every row."""
    rows = []
    for D in dgrid:
        thD = anchor_flat - D * u
        disp_check = float(torch.norm(thD - anchor_flat))
        load_flat(evl, thD)
        evl.eval()
        gz = battery_cell(evl, primary_ids, zid)
        row = {"D": float(D), "rms": float(D) / RMS_DENOM,
               "gm": gz["mean_pz"], "frac_argmax_z": gz["frac_argmax_z"],
               "disp_check": disp_check, "disp_dev": abs(disp_check - float(D))}
        for j in coruler_ids:
            row[f"g{j:+d}"] = battery_cell(evl, coruler_ids[j], zid)["mean_pz"]
        if ce_ds is not None and any(abs(D - c) < 1e-9 for c in ce_ds):
            row["ce_r"] = ce_fixed_cpu(evl, *r_eval_xy)
        rows.append(row)
    return rows


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("e197_smoke" if SMOKE else "e197")
    log(f"E197 THE ALIVE WINDOW (smoke={SMOKE}) -> {rd}")
    log(f"compute: CPU-ONLY (threads {torch.get_num_threads()}), phases "
        f"< 180s, progressive writes, n=1, organism 2 of 2, stream seed "
        f"{FREEZE_SEED}")

    # ---------------- committed parents (loaded, never rerun) ------------------
    for p in (E157_METRICS, E193_METRICS, E195_METRICS, E196_METRICS):
        if not p.exists():
            raise RuntimeError(f"missing parent metrics: {p}")
    e157m = json.loads(E157_METRICS.read_text(encoding="utf-8"))
    e193m = json.loads(E193_METRICS.read_text(encoding="utf-8"))
    e195m = json.loads(E195_METRICS.read_text(encoding="utf-8"))
    e196m = json.loads(E196_METRICS.read_text(encoding="utf-8"))
    # hard-bind the committed references (asserts catch committed-file drift)
    dial = e157m["stages"]["A_consolidate"]["dial"]
    jit = e157m["stages"]["A_consolidate"]["consolidated_jitter_geos"]
    assert abs(dial["base"]["-12"]["mean_pz"] - E157_DIAL["gm12"]) < 1e-12
    assert abs(dial["base"]["0"]["mean_pz"] - E157_DIAL["g0"]) < 1e-12
    assert abs(dial["base"]["12"]["mean_pz"] - E157_DIAL["gp12"]) < 1e-12
    assert abs(dial["ce_r"] - E157_DIAL["ce_r"]) < 1e-12
    for j, k in ((-8, "g-8"), (-4, "g-4"), (4, "g+4"), (8, "g+8")):
        assert abs(jit[f"g{j:+d}"] - E157_DIAL[k]) < 1e-12
    assert abs(e193m["organism"]["step_l2_measured"] - E193_STEP_L2) < 1e-12
    asign = e193m["ladder"]["arms"]["a_sign"]
    s1c, stopc = asign["traj"][0], asign["stop"]
    assert s1c["step"] == 1 and stopc["kind"] == "kill" and stopc["step"] == 1
    for k in ("ce_batch", "cum_disp", "step_disp", "preclip_gnorm", "gm"):
        assert abs(s1c[k] - E193_ASIGN_S1[k]) < 1e-12, f"e193 s1 {k} drift"
    assert abs(stopc["D_kill"] - E193_ASIGN_STOP["D_kill"]) < 1e-12
    for a, b in zip(stopc["dens"], E193_ASIGN_STOP["dens"]):
        assert abs(a["gm"] - b[1]) < 1e-12 and abs(a["D"] - b[2]) < 1e-12
    assert abs(e193m["adjudication"]["kills"]["g"] - E193_KILLS["g"]) < 1e-12
    assert abs(e193m["adjudication"]["kills"]["sign"]
               - E193_KILLS["sign"]) < 1e-12
    e193_r2 = e193m["profiles"]["R2_SIGN"]           # the committed static sign profile
    for D, spot in E193_R2_SPOTS.items():
        row = next(r for r in e193_r2["rows"] if abs(r["D"] - D) < 1e-9)
        assert abs(row["gm"] - spot["gm"]) < 1e-12
        assert abs(row["ce_r"] - spot["ce_r"]) < 1e-12
    for k, v in E195_ROOT_DKILLS.items():
        assert abs(e195m["profiles"][k]["D_kill"] - v) < 1e-12, f"e195 {k} drift"
    assert e195m["adjudication"]["verdict"] == "FLEEING-IS-LETHAL"
    for k, v in E196_ROOT_DKILLS.items():
        assert abs(e196m["profiles"][k]["D_kill"] - v) < 1e-12, f"e196 {k} drift"
    assert e196m["adjudication"]["verdict"] == "GRADED"
    assert e196m["cell"]["rays"][1]["u_md5"] == E196_U1_MD5
    assert abs(e196m["phase0_walk_rebuild"]["walk_journal"][0]["gm"]
               - E196_TH1_GM) < 1e-12
    log("parents: e157 (the dial + wash s1 anchors), e193 (the organism: "
        f"STEP_L2 {E193_STEP_L2:.10f} measured, a_sign step-1 kill "
        f"{E193_ASIGN_STOP['D_kill']:.4f} committed — DEAD at t=1), e195 "
        "(organism 1's flight: root_u1 "
        f"{E195_ROOT_DKILLS['root_u1']:.4f} vs root_u0 "
        f"{E195_ROOT_DKILLS['root_u0']:.4f}), e196 (organism 2's "
        f"DEAD-lineage flight ray: root_u1 {E196_ROOT_DKILLS['root_u1']:.4f} "
        f"— its SOFTEST, ratio 2.585) — loaded COMMITTED, never rerun")

    # ---------------- protocol rebuild (e193 verbatim) -------------------------
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
    for host in HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    import random as _random
    rng = _random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ = host_occ[:60]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix, "pass": bool(mix == {"FLORIZEL": 19,
                                                         "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"

    bat_ids = {}
    for j in READ_GEOS:
        cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    G_BATTERY = {
        "shapes": {str(j): list(bat_ids[j].shape) for j in READ_GEOS},
        "pass": bool(all(list(bat_ids[j].shape) == [60, PRE + j]
                         for j in READ_GEOS)),
        "note": "PRE-DISPATCH CHECK (Rule 12): install-60 battery at the "
                "e157 dial's seven read geometries, shapes 60 x (130 +- j) "
                "— e193's gate verbatim",
    }
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"
    primary_ids = bat_ids[RULER_J]
    coruler_ids = {j: bat_ids[j] for j in CO_RULERS_J}
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

    # the neutral stream (e170 VERBATIM via e185/e193)
    arng = _random.Random(170)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        txt = train_text[s: s + BLOCK + 1]
        if any(f in txt for f in ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")):
            rejections += 1
            continue
        n_starts.append(s)
    if len(n_starts) != 16:
        raise RuntimeError(f"neutral bank incomplete: {len(n_starts)}/16")
    anchor_neutral = torch.stack([train_ids[s: s + BLOCK] for s in n_starts])
    G_ANCHOR = {
        "construction": ("16 plain corpus windows from train_ids, RNG seed "
                         "170, rejection on FLORIZEL/ELIZABETH/ZEPH/MIRABEL "
                         "in [s, s+257) — e170 VERBATIM"),
        "starts": n_starts, "tries": tries, "rejections": rejections,
        "bank_starts_match_e185_stored": bool(n_starts == E170_BANK_STARTS),
        "n_windows": 16, "block": BLOCK, "seed": 170,
    }
    G_ANCHOR["pass"] = G_ANCHOR["bank_starts_match_e185_stored"]
    assert G_ANCHOR["pass"], "neutral bank drifted vs e185's stored starts"

    # ---------------- the family-2 root + G_ROOT (bit vs e157's dial) ---------
    net0 = load_f2(CKPT_DIR / ROOT_CK)
    st_raw = torch.load(CKPT_DIR / ROOT_CK, map_location="cpu",
                        weights_only=False)
    root_meta = E43.jsonable(st_raw.get("meta", {}))
    theta0 = flat_params(net0)
    N_PARAM = int(theta0.numel())
    assert N_PARAM == F2_PARAMS, f"params {N_PARAM} != {F2_PARAMS}"
    root_md5 = hashlib.md5(theta0.numpy().tobytes()).hexdigest()

    evl0 = copy.deepcopy(net0)
    root_cells = {f"g{j:+d}": battery_cell(evl0, bat_ids[j], zid)["mean_pz"]
                  for j in READ_GEOS}
    root_cells["ce_r"] = ce_fixed_cpu(evl0, *r_eval_xy)
    keymap = {"gm12": "g-12", "g0": "g+0", "gp12": "g+12",
              "g-4": "g-4", "g+4": "g+4", "g-8": "g-8", "g+8": "g+8",
              "ce_r": "ce_r"}
    root_refs = {keymap[k]: v for k, v in E157_DIAL.items()}
    rdiffs = {k: root_cells[k] - root_refs[k] for k in root_refs}
    rmax = max(abs(v) for v in rdiffs.values())
    G_ROOT = {"cells": root_cells, "refs": root_refs, "diffs": rdiffs,
              "max_abs_diff": rmax, "bit_tol": G_BIT_TOL,
              "tol": G_FALLBACK_TOL, "bit": bool(rmax < G_BIT_TOL),
              "flat_md5": root_md5,
              "flat_md5_match_e193_committed": bool(root_md5 == ROOT_FLAT_MD5),
              "pass": bool(rmax < G_FALLBACK_TOL
                           and root_md5 == ROOT_FLAT_MD5),
              "note": "e193's G_ROOT VERBATIM: the f2 consolidated root's "
                      "dial reproduces e157's committed cells BIT-TIGHT "
                      "(threads 8) + the flat md5 matches e193's committed "
                      "root identity"}
    log(f"G_ROOT (vs e157 committed dial, 8 cells): max|diff| {rmax:.2e}, "
        f"flat md5 {'match' if G_ROOT['flat_md5_match_e193_committed'] else 'DRIFT'}: "
        + ("PASS" if G_ROOT["pass"] else "FAIL"))
    if not G_ROOT["pass"]:
        raise RuntimeError("family-2 root gate FAILED vs e157 committed dial")

    # ---------------- G_T0 / G_S1CK (e193's t=0 gates, verbatim) --------------
    def draw_step1_batch():
        g = torch.Generator().manual_seed(FREEZE_SEED)
        aj = torch.randint(16, (ANCH_BS,), generator=g)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                           generator=g)
        anc = anchor_neutral[aj]
        rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj])
        x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0)
        y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
        return x, y

    x1, y1 = draw_step1_batch()
    x1_md5 = hashlib.md5(x1.contiguous().numpy().tobytes()).hexdigest()
    tw = copy.deepcopy(net0)
    tw.train()
    optw = torch.optim.AdamW(tw.parameters(), lr=LR_ADAMW, betas=(0.9, 0.95),
                             weight_decay=0.1)
    logits, _ = tw(x1)
    ce1 = float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                y1.reshape(-1)).item())
    optw.zero_grad(set_to_none=True)
    F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                    y1.reshape(-1)).backward()
    gn1 = float(torch.nn.utils.clip_grad_norm_(tw.parameters(), 1.0))
    optw.step()
    theta1_adam = flat_params(tw)
    disp1 = float(torch.norm(theta1_adam - theta0))
    del tw, optw, logits
    evl_t1a = copy.deepcopy(net0)
    load_flat(evl_t1a, theta1_adam)
    t1_primary = battery_cell(evl_t1a, primary_ids, zid)["mean_pz"]

    s1_net = load_f2(CKPT_DIR / S1_CK)
    s1_meta = E43.jsonable(torch.load(CKPT_DIR / S1_CK, map_location="cpu",
                                      weights_only=False).get("meta", {}))
    theta_s1 = flat_params(s1_net)
    d_fresh = theta1_adam - theta0
    d_ck = theta_s1 - theta0
    cos_s1 = cos64(d_fresh, d_ck)
    rel_l2 = abs(float(torch.norm(d_ck)) - disp1) / disp1
    G_T0 = {
        "step1_x_md5": x1_md5,
        "step1_x_md5_match_e185": bool(x1_md5 == E185_XHASH[1]),
        "ce_batch_measured": ce1, "ce_batch_committed_e157":
            E157_WASH_S1["corpus_ce"],
        "d_ce": abs(ce1 - E157_WASH_S1["corpus_ce"]),
        "preclip_gnorm_measured": gn1, "clip_binds": bool(gn1 > 1.0),
        "adamw_step1_L2_measured": disp1,
        "committed_e193_step_l2": E193_STEP_L2,
        "d_step_l2": abs(disp1 - E193_STEP_L2),
        "s1_ckpt_disp_L2": float(torch.norm(d_ck)),
        "poststep_primary_read": t1_primary,
        "committed_e193_a_sign_s1_gm": E193_ASIGN_S1["gm"],
        "note": "e193's G_T0 VERBATIM: the step-1 batch md5 vs e185's "
                "stored hash (net-independent); the forward CE vs e157's "
                "committed wash step-1 corpus CE; the fresh AdamW step's "
                "L2 must equal e193's committed MEASURED step L2 (never "
                "ported); the AdamW t=1 state's primary read cross-reports "
                "the a_sign walked endpoint's (both ~0.0068, dead)",
        "pass": bool(x1_md5 == E185_XHASH[1]
                     and abs(ce1 - E157_WASH_S1["corpus_ce"]) < G_FALLBACK_TOL
                     and abs(disp1 - E193_STEP_L2) < G_FALLBACK_TOL
                     and rel_l2 < 0.05),
    }
    log(f"G_T0 (t=0 gate): x_md5 "
        f"{'OK' if G_T0['step1_x_md5_match_e185'] else 'MISMATCH'}; CE |d| "
        f"{G_T0['d_ce']:.2e}; measured L2 {disp1:.10f} vs committed "
        f"{E193_STEP_L2:.10f} (|d| {G_T0['d_step_l2']:.2e}); rel vs s1-ckpt "
        f"{rel_l2:.2e}: " + ("PASS" if G_T0["pass"] else "FAIL"))
    if not G_T0["pass"]:
        raise RuntimeError("t=0 gate FAILED — abort (control failure)")
    STEP_L2 = disp1                        # measured, never ported

    G_S1CK = {
        "cos64_fresh_vs_ckpt": cos_s1, "rel_L2_dev": rel_l2,
        "tol_cos": 0.999, "s1_meta": s1_meta,
        "pass": bool(cos_s1 > 0.999 and rel_l2 < 0.05),
        "note": "e193's G_S1CK VERBATIM: the fresh CPU AdamW step vs the "
                "committed cuda-trained e157_f2_neutral_s1 checkpoint's "
                "displacement — fp64 cosine > 0.999 + relative L2 < 5% "
                "(e157's cross-device absolute-value convention)",
    }
    log(f"G_S1CK: cos64 {cos_s1:.8f}, rel L2 dev {rel_l2:.2e}: "
        + ("PASS" if G_S1CK["pass"] else "FAIL"))
    if not G_S1CK["pass"]:
        raise RuntimeError("s1 checkpoint anchor FAILED — abort")

    # the t=0 post-clip gradient: organism 2's g-ray + static sign ray
    gnet = copy.deepcopy(net0)
    gnet.train()
    gnet.zero_grad(set_to_none=True)
    logits_g, _ = gnet(x1)
    loss_g = F.cross_entropy(logits_g.reshape(-1, logits_g.shape[-1]),
                             y1.reshape(-1))
    assert abs(float(loss_g.item()) - ce1) < 1e-9, "CE drifted between gates"
    loss_g.backward()
    torch.nn.utils.clip_grad_norm_(gnet.parameters(), 1.0)
    g0 = torch.cat([p.grad.detach().reshape(-1)
                    for p in gnet.parameters()]).clone()   # post-clip
    gnet.zero_grad(set_to_none=True)
    del gnet, logits_g, loss_g
    u_g = (g0 / torch.norm(g0)).clone()
    u_g_md5 = hashlib.md5(u_g.numpy().tobytes()).hexdigest()
    s_raw = torch.sign(g0)
    n_zero_g = int((g0 == 0).sum())
    u0 = (s_raw / torch.norm(s_raw)).clone()    # e193's construction verbatim
    u0_md5 = hashlib.md5(u0.numpy().tobytes()).hexdigest()
    del s_raw

    # G_STREAM: the stream machinery verified steps 1..4 WITHOUT training
    gen_s = torch.Generator().manual_seed(FREEZE_SEED)
    stream_ok = {}
    for s_ in range(1, 5):
        aj_ = torch.randint(16, (ANCH_BS,), generator=gen_s)
        rj_ = torch.randint(len(train_ids) - BLOCK - 1, (RAND_BS,),
                            generator=gen_s)
        anc_ = anchor_neutral[aj_]
        rnd_ = torch.stack([train_ids[q: q + BLOCK] for q in rj_])
        x_ = torch.cat([anc_[:, :-1], rnd_[:, :-1]], 0)
        h_ = hashlib.md5(x_.contiguous().numpy().tobytes()).hexdigest()
        stream_ok[s_] = bool(h_ == E185_XHASH[s_])
    G_STREAM = {"steps_1_4_md5_match": stream_ok,
                "pass": bool(all(stream_ok.values())),
                "note": "e193's G_STREAM VERBATIM: the seed-10902 stream "
                        "construction (net-independent) md5-matches e185's "
                        "stored hashes"}
    log("G_STREAM: seed-10902 stream md5-matches e185's stored hashes "
        "(steps 1..4): " + ("PASS" if G_STREAM["pass"] else "FAIL"))
    assert G_STREAM["pass"], "stream construction diverged from e185"

    # G_DIRCK: e193's committed g-ray direction checkpoint (the root identity)
    uck = torch.load(CKPT_DIR / DIR_CK, map_location="cpu", weights_only=False)
    G_DIRCK = {
        "path": str(CKPT_DIR / DIR_CK),
        "meta_gate": {"experiment": uck["meta"].get("experiment"),
                      "root": uck["meta"].get("root"),
                      "match": bool(uck["meta"].get("experiment") == "e193"
                                    and uck["meta"].get("root") == ROOT_CK)},
        "loaded_u_md5": hashlib.md5(
            uck["u"].numpy().tobytes()).hexdigest(),
        "fresh_u_md5": u_g_md5,
        "md5_match": bool(hashlib.md5(
            uck["u"].numpy().tobytes()).hexdigest() == u_g_md5),
        "theta0_md5_match_root": bool(uck.get("theta0_md5") == root_md5),
        "u_norm_fp32": float(torch.norm(u_g)),
        "note": "e191's direction-checkpoint convention, e193's file: the "
                "committed u must BE the fresh t=0 g-ray bit-exactly and "
                "the committed theta0_md5 must be THIS root's flat md5 "
                "(the root state's identity gate)",
    }
    G_DIRCK["pass"] = bool(G_DIRCK["meta_gate"]["match"]
                           and G_DIRCK["md5_match"]
                           and G_DIRCK["theta0_md5_match_root"])
    log(f"G_DIRCK (e193's committed g-ray): md5 "
        + ("match" if G_DIRCK["md5_match"] else "DRIFT")
        + f", theta0_md5 {'match' if G_DIRCK['theta0_md5_match_root'] else 'DRIFT'}: "
        + ("PASS" if G_DIRCK["pass"] else "FAIL"))

    # G_SIGNRAY: u0 rebuilt == e193's committed R2_SIGN direction
    G_SIGNRAY = {
        "u0_md5": u0_md5, "committed_e193_md5": E193_USIGN_MD5,
        "md5_match": bool(u0_md5 == E193_USIGN_MD5),
        "u_norm_fp32": float(torch.norm(u0)),
        "u_norm_fp64": float(torch.norm(u0.double())),
        "n_zero_g_coords": n_zero_g,
        "note": "family u0 = the static sign(g_0) ray rebuilt from the "
                "gated t=0 gradient (e193's fp32-norm construction "
                "VERBATIM); md5-gated vs e193's committed R2_SIGN ray",
    }
    G_SIGNRAY["pass"] = bool(G_SIGNRAY["md5_match"])
    log(f"G_SIGNRAY (u0 = static sign ray): md5 "
        + ("match" if G_SIGNRAY["md5_match"] else "DRIFT")
        + f" (fp32 norm {G_SIGNRAY['u_norm_fp32']:.7f}): "
        + ("PASS" if G_SIGNRAY["pass"] else "FAIL"))
    if not (G_DIRCK["pass"] and G_SIGNRAY["pass"]):
        raise RuntimeError("committed-ray gates FAILED — abort")

    log("WHAT THESE ARMS GUARANTEE: NOTHING — the sub-step lineage is a "
        "counterfactual wash (the size is the experimenter's); the flight "
        "structure could form, stay absent, or grade; that openness is "
        "the point.")

    # =====================================================================
    # metrics stub + progressive writes
    # =====================================================================
    stub: dict = {"gates": {}, "phases_partial": {}}

    def write_partial(phase: str):
        stub.update({
            "experiment": "e197_alive_window",
            "date": common.now_iso(),
            "status": f"PARTIAL — {phase} (progressive write; the final "
                      f"COMPLETE write replaces it)",
            "registration": REGISTERED_BARS["registration"],
            "registered_bars": REGISTERED_BARS,
            "timing_partial": {"total_s": round(time.time() - T0, 1)},
            "config_partial": {"smoke": SMOKE,
                               "torch": torch.__version__,
                               "threads": torch.get_num_threads()},
        })
        save_json(rd / "metrics.json", E43.jsonable(stub))

    stub["gates"] = {"G_SPLICE": G_SPLICE, "G_NAMEFREE": G_NAMEFREE,
                     "G_BATTERY": G_BATTERY, "G_ANCHOR": G_ANCHOR,
                     "G_ROOT": G_ROOT, "G_T0": G_T0, "G_S1CK": G_S1CK,
                     "G_STREAM": G_STREAM, "G_DIRCK": G_DIRCK,
                     "G_SIGNRAY": G_SIGNRAY}
    write_partial("standard cell gates passed (e193's, verbatim)")
    log("standard cell gates passed; partial metrics written")

    # =====================================================================
    # PHASE 0 — the FULL-STEP walk rebuilt (the dead-at-t=1 baseline, G_REPRO)
    # =====================================================================
    log("=" * 78)
    log("PHASE 0 — the FULL-STEP k=1 sign walk rebuilt (e193's a_sign; the "
        "dead-at-t=1 baseline re-anchored in THIS process)")
    wfull = walk_sub(net0, anchor_neutral, train_ids, itos, primary_ids,
                     coruler_ids, zid, theta0, STEP_L2,
                     root_gm=root_cells[f"g{RULER_J:+d}"],
                     step_cap=1, d_target=5.0, extend_past_kill=0,
                     tag="full")
    assert len(wfull["traj"]) == 1, \
        f"expected the 1-step full walk (kill s1), got {len(wfull['traj'])}"
    fs1 = wfull["traj"][0]

    repro_rows = {
        "s1_ce_batch": (fs1["ce_batch"], E193_ASIGN_S1["ce_batch"]),
        "s1_cum_disp": (fs1["cum_disp"], E193_ASIGN_S1["cum_disp"]),
        "s1_step_disp": (fs1["step_disp"], E193_ASIGN_S1["step_disp"]),
        "s1_preclip_gnorm": (fs1["preclip_gnorm"],
                             E193_ASIGN_S1["preclip_gnorm"]),
        "s1_gm": (fs1["gm"], E193_ASIGN_S1["gm"]),
        "s1_frac_argmax_z": (fs1["frac_argmax_z"],
                             E193_ASIGN_S1["frac_argmax_z"]),
        "D_kill": (wfull["stop"]["D_kill"], E193_ASIGN_STOP["D_kill"]),
        "D_kill_raw": (wfull["stop"]["D_kill_raw"],
                       E193_ASIGN_STOP["D_kill_raw"]),
    }
    dens_diffs = [abs(a["gm"] - b[1]) for a, b in
                  zip(wfull["stop"]["dens"], E193_ASIGN_STOP["dens"])]
    dens_D_diffs = [abs(a["D"] - b[2]) for a, b in
                    zip(wfull["stop"]["dens"], E193_ASIGN_STOP["dens"])]
    G_REPRO = {
        "rows": {kk: {"measured": vv[0], "committed": vv[1],
                      "abs_diff": abs(vv[0] - vv[1])}
                 for kk, vv in repro_rows.items()},
        "dens_gm_max_abs_diff": max(dens_diffs),
        "dens_D_max_abs_diff": max(dens_D_diffs),
        "x_hash_vs_e185": bool(wfull["x_hashes"][1] == E185_XHASH[1]),
        "tol": G_REPRO_TOL,
        "max_l2_dev": wfull["max_l2_dev"],
        "pass": bool(wfull["stop"]["kind"] == "kill"
                     and wfull["stop"]["step"] == 1
                     and max(abs(vv[0] - vv[1]) for vv in repro_rows.values())
                     < G_REPRO_TOL
                     and max(dens_diffs) < G_REPRO_TOL
                     and max(dens_D_diffs) < G_REPRO_TOL
                     and wfull["x_hashes"][1] == E185_XHASH[1]),
        "note": "THE BASELINE PROVENANCE GATE (Rule 12): the full-step k=1 "
                "walk must reproduce e193's committed a_sign step-1 row AND "
                "its densified kill bracket bit-class — the DEAD-at-t=1 "
                "baseline this cell intervenes on, re-anchored in this "
                "process",
    }
    log("G_REPRO (vs e193 committed a_sign s1 + bracket): max|diff| "
        f"{max(abs(vv[0] - vv[1]) for vv in repro_rows.values()):.2e}, "
        f"dens gm {max(dens_diffs):.2e}, dens D {max(dens_D_diffs):.2e}: "
        + ("PASS" if G_REPRO["pass"] else "FAIL"))
    if not G_REPRO["pass"]:
        stub["gates"]["G_REPRO"] = G_REPRO
        write_partial("CONTROL FAILURE — full-step baseline gate failed")
        raise RuntimeError("G_REPRO FAILED — abort before any read is believed")
    stub["gates"]["G_REPRO"] = G_REPRO
    stub["phases_partial"]["0_fullstep_baseline"] = E43.jsonable({
        "walk_journal": wfull["traj"], "stop": wfull["stop"],
        "x_hashes": wfull["x_hashes"], "max_l2_dev": wfull["max_l2_dev"],
        "disclosure": "the natural-size walk (STEP_L2 0.9164): DEAD at "
                      "t=1 (g-4 0.0068) — e193's committed result, "
                      "re-anchored here; both dynamic effects absent on "
                      "this lineage (e196)",
    })
    write_partial("phase 0 complete (full-step dead baseline rebuilt)")

    # =====================================================================
    # PHASE S — THE SUB-STEP CHOICE (registered BEFORE the walk)
    # =====================================================================
    log("=" * 78)
    log("PHASE S — the sub-step choice: frozen ladder "
        f"{[round(f, 4) for f in SUB_LADDER]} x STEP_L2 probed statically; "
        "LARGEST rung reading > 0.27 wins (registered before the walk)")
    evp = copy.deepcopy(net0)
    ladder_rows = []
    for frac in SUB_LADDER:
        s_ = STEP_L2 * frac
        thD = theta0 - s_ * u0
        load_flat(evp, thD)
        evp.eval()
        gz_ = battery_cell(evp, primary_ids, zid)
        co_ = {f"g{j:+d}": battery_cell(evp, coruler_ids[j], zid)["mean_pz"]
               for j in CO_RULERS_J}
        ladder_rows.append({"fraction": float(frac), "s": float(s_),
                            "D": float(torch.norm(thD - theta0)),
                            "gm": gz_["mean_pz"],
                            "above_bar": bool(gz_["mean_pz"] > SHUT_BAR),
                            **co_})
        log(f"  rung 1/{round(1/frac)}: s {s_:.6f} -> g-4 "
            f"{gz_['mean_pz']:.6f} "
            + ("ABOVE" if gz_["mean_pz"] > SHUT_BAR else "under"))
    passing = [r for r in ladder_rows if r["above_bar"]]
    assert passing, "no ladder rung lands above the bar — the cell cannot "
    pick = max(passing, key=lambda r: r["s"])
    SUB_S = pick["s"]
    SUB_FRAC = pick["fraction"]
    sub_choice = {
        "ladder": ladder_rows, "rule": "LARGEST rung whose static "
        "first-landing probe (theta_0 - s*u0, primary battery) reads "
        "strictly > 0.27; frozen BEFORE the walk; the dispatch's half "
        "step primary, quarter step fallback",
        "picked_fraction": SUB_FRAC, "picked_s": SUB_S,
        "picked_probe_gm": pick["gm"],
        "registered_before_walk": True,
        "note": "the probe reads the exact landing the walk's step-1 will "
                "occupy (same u0 direction; fp32-norm ray vs fp64-norm "
                "step — texture class); the walk's own step-1 read is the "
                "authoritative ALIVE check (G_ALIVE)",
    }
    stub["phases_partial"]["S_substep_choice"] = E43.jsonable(sub_choice)
    write_partial(f"sub-step choice REGISTERED (s* = {SUB_S:.6f} = "
                  f"STEP_L2/{round(1/SUB_FRAC)}) — BEFORE the walk")
    log(f"  PICK: s* = {SUB_S:.6f} (STEP_L2 / {round(1/SUB_FRAC)}), probe "
        f"g-4 {pick['gm']:.6f} > {SHUT_BAR}")

    # =====================================================================
    # PHASE 1 — THE ALIVE WALK (the sub-step wash; the intervention)
    # =====================================================================
    log("=" * 78)
    log(f"PHASE 1 — THE ALIVE WALK: k=1 sign machinery verbatim, step size "
        f"s* {SUB_S:.6f}, cap {WALK_CAP} steps / D {WALK_D_TARGET}, "
        f"{EXT_STEPS} read-only post-kill step")
    wsub = walk_sub(net0, anchor_neutral, train_ids, itos, primary_ids,
                    coruler_ids, zid, theta0, SUB_S,
                    root_gm=root_cells[f"g{RULER_J:+d}"],
                    step_cap=WALK_CAP, d_target=WALK_D_TARGET,
                    extend_past_kill=EXT_STEPS, tag="sub")
    ws1 = wsub["traj"][0]

    # ---- G_ALIVE: the intervention's own guarantee, asserted
    G_ALIVE = {
        "walk_s1_gm": ws1["gm"], "bar": SHUT_BAR,
        "alive": bool(ws1["gm"] > SHUT_BAR),
        "probe_gm": pick["gm"],
        "probe_vs_walk_diff": abs(ws1["gm"] - pick["gm"]),
        "tol_texture": G_LANDING_TOL,
        "cum_disp_s1": ws1["cum_disp"],
        "expected_D": SUB_S,
        "x1_hash_match": bool(wsub["x_hashes"][1] == E185_XHASH[1]),
        "pass": bool(ws1["gm"] > SHUT_BAR
                     and abs(ws1["gm"] - pick["gm"]) < G_LANDING_TOL
                     and abs(ws1["cum_disp"] - SUB_S) < G_LANDING_TOL
                     and wsub["x_hashes"][1] == E185_XHASH[1]),
        "note": "THE INTERVENTION'S GATE: the sub-step walk's step-1 "
                "endpoint must read ALIVE (> 0.27) and match its "
                "registered static probe within the fp32-norm texture "
                "class — the alive window this cell exists to open; "
                "failure = control failure, abort",
    }
    log(f"G_ALIVE: walk s1 g-4 {ws1['gm']:.6f} vs probe {pick['gm']:.6f} "
        f"(|d| {G_ALIVE['probe_vs_walk_diff']:.2e}), D {ws1['cum_disp']:.6f}: "
        + ("PASS" if G_ALIVE["pass"] else "FAIL"))
    if not G_ALIVE["pass"]:
        stub["gates"]["G_ALIVE"] = G_ALIVE
        write_partial("CONTROL FAILURE — the alive gate failed")
        raise RuntimeError("G_ALIVE FAILED — abort (the intervention itself "
                           "failed to open the window)")
    stub["gates"]["G_ALIVE"] = G_ALIVE
    stub["phases_partial"]["1_alive_walk"] = E43.jsonable({
        "journal": wsub["traj"], "stop": wsub["stop"],
        "x_hashes": wsub["x_hashes"], "max_l2_dev": wsub["max_l2_dev"],
        "sub_s": SUB_S, "sub_fraction": SUB_FRAC,
        "alive_steps": int(sum(1 for r in wsub["traj"]
                               if not r["post_kill"]
                               and r["gm"] > SHUT_BAR)),
        "post_kill_disclosure": "any step past the kill is read-only "
                                "continuation (its refresh gradient's SIGN "
                                "is the u2 front if the walk dies before "
                                "t=2; nothing post-kill adjudicates)",
    })
    write_partial("phase 1 complete (the alive walk)")

    # ---- the rays: u1/u2 from the ALIVE walk's refresh gradients
    g0_w, g1_w = wsub["g_fronts"][0], wsub["g_fronts"][1]
    g2_w = wsub["g_fronts"][2] if len(wsub["g_fronts"]) > 2 else None
    assert cos64(torch.sign(g0_w), u0) > 1 - 1e-9, "t=0 front != u0"
    u1 = (torch.sign(g1_w) / torch.norm(torch.sign(g1_w))).clone()
    u1_md5 = hashlib.md5(u1.numpy().tobytes()).hexdigest()
    u2, u2_md5 = None, None
    if g2_w is not None:
        u2 = (torch.sign(g2_w) / torch.norm(torch.sign(g2_w))).clone()
        u2_md5 = hashlib.md5(u2.numpy().tobytes()).hexdigest()
    theta1 = wsub["endpoints"][1].clone()
    evl_t1 = copy.deepcopy(net0)
    load_flat(evl_t1, theta1)
    evl_t1.eval()
    t1_gm = battery_cell(evl_t1, primary_ids, zid)["mean_pz"]
    G_TH1ANCHOR = {
        "cum_disp": float(torch.norm(theta1 - theta0)),
        "walk_s1_cum_disp": ws1["cum_disp"],
        "gm_measured": t1_gm, "gm_walk_s1": ws1["gm"],
        "gm_diff": abs(t1_gm - ws1["gm"]),
        "alive_at_anchor": bool(t1_gm > SHUT_BAR),
        "contrast_e196_dead_theta1": E196_TH1_GM,
        "contrast_org1_alive_theta1": ORG1_TH1_ALIVE,
        "tol": G_REPRO_TOL,
        "pass": bool(abs(float(torch.norm(theta1 - theta0))
                         - ws1["cum_disp"]) < G_REPRO_TOL
                     and abs(t1_gm - ws1["gm"]) < G_REPRO_TOL),
        "note": "theta_1 = the SUB-STEP walk's step-1 endpoint — ALIVE by "
                "G_ALIVE (organism 2's full-step theta_1 was dead at "
                "0.0068; organism 1's was alive at 0.679); the anchor of "
                "the theta_1 panel",
    }
    log(f"G_TH1ANCHOR (ALIVE theta_1): cum {G_TH1ANCHOR['cum_disp']:.7f}, "
        f"g-4 {t1_gm:.7f} (vs full-step dead {E196_TH1_GM:.7f}; org-1 "
        f"alive {ORG1_TH1_ALIVE:.4f}): "
        + ("PASS" if G_TH1ANCHOR["pass"] else "FAIL"))
    if not G_TH1ANCHOR["pass"]:
        stub["gates"]["G_TH1ANCHOR"] = G_TH1ANCHOR
        write_partial("CONTROL FAILURE — theta_1 anchor gate failed")
        raise RuntimeError("theta_1 anchor gate FAILED — abort")
    stub["gates"]["G_TH1ANCHOR"] = G_TH1ANCHOR

    u2_context = ("u2 = the refresh gradient's sign at theta_2 on batch 3"
                  + (" (ALIVE state)" if len(wsub["traj"]) > 1
                     and not wsub["traj"][1]["post_kill"]
                     else " (POST-KILL state — disclosed, e196's "
                          "precedent; the u2 family never adjudicates)"))
    ray_geometry = {
        "cos_u0_u1": cos64(u0, u1),
        "cos_u0_u2": (cos64(u0, u2) if u2 is not None else None),
        "cos_u1_u2": (cos64(u1, u2) if u2 is not None else None),
        "cos_u0_ug": cos64(u0, u_g),
        "cos_u1_ug": cos64(u1, u_g),
        "cos_u2_ug": (cos64(u2, u_g) if u2 is not None else None),
        "iso_floor": 1.0 / (F2_PARAMS ** 0.5),
        "organism1_committed": {"cos_u0_u1": -0.15452721980470194,
                                "cos_u0_u2": 0.03151076362111852,
                                "cos_u1_u2": -0.20298347429000238},
        "organism2_deadlineage_committed": {"cos_u0_u1": -0.1653095242890399,
                                             "cos_u0_u2": -0.06834241200680569,
                                             "cos_u1_u2": -0.07544285736169305},
        "note": "the rays' mutual geometry (fp64 cosines); organism 1's "
                "committed e195 geometry and organism 2's committed "
                "DEAD-lineage e196 geometry ride for the side-by-side "
                "(never a bar)",
    }

    # the dual-estimator matched-point alignment reads (T150's lesson:
    # every read states its point; dual form = walked-delta vs unit-ray)
    root_prim_grad = fact_grad(evl0, primary_ids, zid)   # grad at theta_0
    t1_prim_grad = fact_grad(evl_t1, primary_ids, zid)   # grad at theta_1
    dual_d1_root = cos64(sign_update(g0_w, SUB_S), root_prim_grad)
    dual_d2_t1 = cos64(sign_update(g1_w, SUB_S), t1_prim_grad)
    unit_d1_root = cos64(-u0, root_prim_grad)
    unit_d2_t1 = cos64(-u1, t1_prim_grad)
    alignment_reads = {
        "evaluation_points": "cos(jump direction, grad of the PRIMARY "
                             "battery readout AT the anchor where that "
                             "step began) — matched-point (T150); the "
                             "jump direction is DESCENT (-u)",
        "root_anchor": {
            "cos_neg_u0": unit_d1_root,
            "cos_neg_u1": cos64(-u1, root_prim_grad),
            "cos_neg_u2": (cos64(-u2, root_prim_grad)
                           if u2 is not None else None),
            "delta_form_s1": dual_d1_root,
            "dual_form_texture_diff_s1": abs(dual_d1_root - unit_d1_root),
        },
        "theta1_anchor": {
            "cos_neg_u0": cos64(-u0, t1_prim_grad),
            "cos_neg_u1": unit_d2_t1,
            "cos_neg_u2": (cos64(-u2, t1_prim_grad)
                           if u2 is not None else None),
            "delta_form_s2": dual_d2_t1,
            "dual_form_texture_diff_s2": abs(dual_d2_t1 - unit_d2_t1),
        },
        "organism2_deadlineage_committed": {
            "root_cos_neg_u1": 0.0008060237658234415,
            "theta1_cos_neg_u1": 0.0974020698682922,
            "note": "e196's committed DEAD-lineage reads (same "
                    "instrument, dead theta_1) — the contrast, never a bar",
        },
        "note": "negative cos(-u, grad) = the ray is ANTI-aligned with the "
                "fact's own gradient at that point (T157's flight twist: "
                "organism 1's flight ray was anti-aligned yet deadliest); "
                "delta-form vs unit-ray form agree to fp32-multiplication "
                "texture (the dual-estimator consistency read)",
    }
    stub["phases_partial"]["1b_rays"] = E43.jsonable({
        "u1_md5": u1_md5, "u2_md5": u2_md5,
        "u2_context": u2_context,
        "ray_geometry": ray_geometry, "alignment_reads": alignment_reads,
    })
    write_partial("phase 1b complete (the alive rays + alignment reads)")
    rays_meta = [
        {"key": "u0", "label": "sign(g_0) — the original static sign ray",
         "u_md5": u0_md5,
         "provenance": "e193's committed R2_SIGN construction verbatim "
                       "(md5-gated, G_SIGNRAY) — shared by every lineage"},
        {"key": "u1", "label": "sign(g_1) — THE ALIVE FLIGHT RAY (the "
                               "sub-step walk's t=1 fresh front)",
         "u_md5": u1_md5,
         "provenance": f"the SUB-STEP walk's step-2 refresh gradient (at "
                       f"the ALIVE theta_1, batch 2, post-clip; s* "
                       f"{SUB_S:.6f} = STEP_L2/{round(1/SUB_FRAC)} — THE "
                       "INTERVENTION); gated machinery: G_REPRO + G_ALIVE; "
                       "CONTRAST: e196's DEAD-lineage u1 md5 "
                       f"{E196_U1_MD5}"},
        {"key": "u2", "label": "sign(g_2) — the t=2 front (rotation's "
                               "continuation)",
         "u_md5": u2_md5,
         "provenance": "the sub-step walk's step-3 refresh gradient (at "
                       "theta_2, batch 3) — "
                       + ("an ALIVE-state read" if u2 is not None
                          and len(wsub["traj"]) > 1
                          and not wsub["traj"][1]["post_kill"]
                          else "a POST-KILL read (disclosed; the u2 family "
                               "never adjudicates); e196's DEAD-lineage u2 "
                               f"md5 {E196_U2_MD5}")},
    ]
    anchors_meta = {
        "theta_0": {"label": "the e157_f2 consolidated root (organism 2)",
                    "gm": root_cells[f"g{RULER_J:+d}"],
                    "ce_r": root_cells["ce_r"],
                    "provenance": f"runs/checkpoints/{ROOT_CK}, gated vs "
                                  "e157's dial BIT-TIGHT + flat-md5 "
                                  "(G_ROOT/G_DIRCK)"},
        "theta_1": {"label": "the SUB-STEP one-stepped state — ALIVE",
                    "gm": t1_gm, "alive_at_anchor": True,
                    "cum_disp": G_TH1ANCHOR["cum_disp"],
                    "provenance": f"the sub-step walk's step-1 endpoint "
                                  f"(s* {SUB_S:.6f}; G_ALIVE + "
                                  "G_TH1ANCHOR); the full-step sibling was "
                                  f"DEAD at {E196_TH1_GM:.7f} — the "
                                  "intervention's whole point"},
    }

    # =====================================================================
    # PHASE A — the ray families (root panel + the ALIVE theta_1 panel)
    # =====================================================================
    log("=" * 78)
    log(f"PHASE A — six families x {len(D_GRID)}+1 grid points (+ the "
        f"landing gate points)")
    ce_ds_common = ([0.0] + [round(CE_EVERY * i, 2)
                             for i in range(1, int(3.0 / CE_EVERY) + 1)]
                    if CE_EVERY else [0.0])
    shared_e193_Ds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.8,
                      1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
    ladder_Ds = [r["s"] for r in ladder_rows]

    def build_family(key, anchor_key, u, extra_ds):
        base = [0.0] + list(D_GRID)
        ds = base + [d for d in extra_ds
                     if not any(abs(d - g) < 1e-9 for g in base)]
        return {"key": key, "anchor_key": anchor_key, "u": u,
                "dgrid": sorted(set(round(d, 10) for d in ds))}

    fam_root_u0 = build_family("root_u0", "theta_0", u0,
                               [STEP_L2, SUB_S] + ladder_Ds)
    fam_root_u0["ce_ds"] = sorted(set(ce_ds_common + shared_e193_Ds))
    fam_root_u1 = build_family("root_u1", "theta_0", u1, [])
    fam_root_u1["ce_ds"] = list(ce_ds_common)
    fam_root_u2 = (build_family("root_u2", "theta_0", u2, [])
                   if u2 is not None else None)
    if fam_root_u2 is not None:
        fam_root_u2["ce_ds"] = list(ce_ds_common)
    fam_th1_u0 = build_family("th1_u0", "theta_1", u0, [])
    fam_th1_u0["ce_ds"] = list(ce_ds_common)
    fam_th1_u1 = build_family("th1_u1", "theta_1", u1, [SUB_S])
    fam_th1_u1["ce_ds"] = list(ce_ds_common)
    fam_th1_u2 = (build_family("th1_u2", "theta_1", u2, [])
                  if u2 is not None else None)
    if fam_th1_u2 is not None:
        fam_th1_u2["ce_ds"] = list(ce_ds_common)
    fam_order = [f for f in (fam_root_u0, fam_root_u1, fam_root_u2,
                             fam_th1_u0, fam_th1_u1, fam_th1_u2)
                 if f is not None]
    anchors = {"theta_0": theta0, "theta_1": theta1}

    families = {}
    for fam in fam_order:
        tA = time.time()
        rows = static_profile(evp, anchors[fam["anchor_key"]], fam["u"],
                              fam["dgrid"], primary_ids, coruler_ids, zid,
                              r_eval_xy, fam["ce_ds"])
        edge, ups = profile_d_kill(rows)
        families[fam["key"]] = {
            "key": fam["key"], "anchor": fam["anchor_key"],
            "ray": fam["key"].split("_")[1],
            "label": next(r["label"] for r in rays_meta
                          if r["key"] == fam["key"].split("_")[1]),
            "u_md5": next(r["u_md5"] for r in rays_meta
                          if r["key"] == fam["key"].split("_")[1]),
            "placement": "anchor - D*u (e192's static placement verbatim)",
            "grid": fam["dgrid"], "n_points": len(rows),
            "gate_points": {}, "rows": rows, "D_kill": edge,
            "any_upcross": ups,
            "anchor_dead": bool(rows[0]["gm"] <= SHUT_BAR),
            "seconds": round(time.time() - tA, 1),
        }
        stub["phases_partial"][f"prof_{fam['key']}"] = E43.jsonable(
            {"D_kill": edge, "any_upcross": ups, "rows": rows,
             "anchor": fam["anchor_key"]})
        write_partial(f"profile {fam['key']} complete")
        log(f"  {fam['key']}: {len(rows)} pts in "
            f"{families[fam['key']]['seconds']}s — D_kill "
            + (f"{edge:.4f}" if edge is not None
               else "None (no downcrossing <= 3.0"
                    + ("; dead anchor" if families[fam['key']]['anchor_dead']
                       else "") + ")")
            + (f", upcrosses {ups}" if ups else ""))

    # ---- G_ROOTPROF: the root-u0 family vs e193's committed R2_SIGN rows
    r2_by_D = {round(r["D"], 4): r for r in e193_r2["rows"]
               if round(r["D"], 4) != 0.0}   # D=0 gated already (G_ROOT)
    xc = []
    for row in families["root_u0"]["rows"]:
        key = round(row["D"], 4)
        if key in r2_by_D:
            entry = {"D": key, "gm_measured": row["gm"],
                     "gm_e193": r2_by_D[key]["gm"],
                     "abs_diff": abs(row["gm"] - r2_by_D[key]["gm"])}
            if "ce_r" in row and "ce_r" in r2_by_D[key]:
                entry["ce_r_measured"] = row["ce_r"]
                entry["ce_r_e193"] = r2_by_D[key]["ce_r"]
                entry["ce_r_abs_diff"] = abs(row["ce_r"]
                                             - r2_by_D[key]["ce_r"])
            xc.append(entry)
    landings = []
    for row in families["root_u0"]["rows"]:
        if abs(row["D"] - STEP_L2) < 1e-9:
            landings.append({"D": row["D"], "gm": row["gm"],
                             "expect": E193_ASIGN_S1["gm"],
                             "abs_diff": abs(row["gm"]
                                             - E193_ASIGN_S1["gm"])})
    G_ROOTPROF = {
        "e193_R2_crosscheck": xc, "n_shared": len(xc),
        "max_abs_diff": (max(r["abs_diff"] for r in xc) if xc else None),
        "step_l2_landing": (landings[0] if landings else None),
        "tol": G_STATIC_TOL, "tol_landing": G_LANDING_TOL,
        "pass": bool(len(xc) == (16 if not SMOKE else len(xc))
                     and all(r["abs_diff"] < G_STATIC_TOL for r in xc)
                     and landings
                     and landings[0]["abs_diff"] < G_LANDING_TOL),
        "note": "THE STATIC-PROFILE MACHINERY GATE: the root-u0 family "
                "must reproduce e193's committed R2_SIGN rows at ALL 16 "
                "shared Ds (+ ce_r where shared) and the D=STEP_L2 "
                "landing must reproduce the committed walked step-1 kill "
                "read (fp32-norm ray vs fp64-norm walked step — texture "
                "class) before any new point is believed",
    }
    log(f"G_ROOTPROF (root-u0 vs e193 R2_SIGN): {len(xc)} shared pts, "
        f"max|diff| {(G_ROOTPROF['max_abs_diff'] or 0):.2e}, STEP_L2 "
        f"landing |d| "
        f"{(landings[0]['abs_diff'] if landings else float('nan')):.2e}: "
        + ("PASS" if G_ROOTPROF["pass"] else "FAIL"))

    # ---- G_TH1PROF: the ALIVE theta_1 families' machinery gate
    th1_anchor_rows = [families[f"th1_{r}"]["rows"][0]
                       for r in ("u0", "u1", "u2") if f"th1_{r}" in families]
    anchor_ok = all(abs(r["gm"] - ws1["gm"]) < G_REPRO_TOL
                    for r in th1_anchor_rows)
    th1_land = None
    for row in families["th1_u1"]["rows"]:
        if abs(row["D"] - SUB_S) < 1e-9:
            expect = (wsub["traj"][1]["gm"]
                      if len(wsub["traj"]) > 1 else None)
            th1_land = {"D": row["D"], "gm": row["gm"],
                        "expect_walk_theta2": expect,
                        "abs_diff": (abs(row["gm"] - expect)
                                     if expect is not None else None)}
    G_TH1PROF = {
        "anchor_d0_gm": [r["gm"] for r in th1_anchor_rows],
        "anchor_expect": ws1["gm"], "anchor_ok": anchor_ok,
        "anchor_alive_disclosure": "theta_1 is ALIVE at D=0 "
                                   f"({ws1['gm']:.6f} > {SHUT_BAR}): the "
                                   "theta_1 panel's kill-Ds RESOLVE this "
                                   "time (e196's were dead-anchored); the "
                                   "panel rides verbatim, never a bar",
        "u1_sub_s_landing": th1_land,
        "tol_anchor": G_REPRO_TOL, "tol_landing": G_LANDING_TOL,
        "pass": bool(anchor_ok
                     and (th1_land is None or th1_land["expect_walk_theta2"]
                          is None
                          or th1_land["abs_diff"] < G_LANDING_TOL)),
        "note": "THE ALIVE THETA_1-PANEL MACHINERY GATE: all theta_1 "
                "families' D=0 rows must reproduce the walk's step-1 read "
                "bit-class; the th1-u1 family's D=s* landing must "
                "reproduce the walk's step-2 read (fp32-norm ray vs "
                "fp64-norm step — texture class; None if the walk ended "
                "before t=2, reported verbatim)",
    }
    log(f"G_TH1PROF (ALIVE theta_1 panel): anchor |d| "
        + ", ".join(f"{abs(r['gm'] - ws1['gm']):.2e}" for r in th1_anchor_rows)
        + (f", u1 s* landing vs theta_2 |d| {th1_land['abs_diff']:.2e}"
           if th1_land and th1_land["abs_diff"] is not None
           else ", landing n/a")
        + ": " + ("PASS" if G_TH1PROF["pass"] else "FAIL"))
    if not (G_ROOTPROF["pass"] and G_TH1PROF["pass"]):
        stub["gates"]["G_ROOTPROF"] = G_ROOTPROF
        stub["gates"]["G_TH1PROF"] = G_TH1PROF
        write_partial("CONTROL FAILURE — profile machinery gates failed")
        raise RuntimeError("profile machinery gates FAILED — abort")
    stub["gates"]["G_ROOTPROF"] = G_ROOTPROF
    stub["gates"]["G_TH1PROF"] = G_TH1PROF
    write_partial("all profile machinery gates passed")

    # =====================================================================
    # ADJUDICATION (frozen clauses; composite ALIVE-WINDOW-CAUSES ->
    # ALIVE-BUT-NO-FLIGHT -> GRADED; no shopping)
    # =====================================================================
    alive = G_ALIVE["alive"]
    static_edge = families["root_u0"]["D_kill"]
    u1_kill = families["root_u1"]["D_kill"]
    walk_kill = (wsub["stop"]["D_kill"]
                 if wsub["stop"]["kind"] == "kill" else None)
    walk_max_D = max(r["cum_disp"] for r in wsub["traj"])
    bonus_ratio = (walk_kill / static_edge
                   if (walk_kill is not None and static_edge) else None)
    flight_ratio = (u1_kill / static_edge
                    if (u1_kill is not None and static_edge) else None)
    bonus_present = bool(walk_kill is not None and bonus_ratio is not None
                         and bonus_ratio <= BONUS_RATIO_BAR)
    flight_present = bool(flight_ratio is not None
                          and flight_ratio <= FLIGHT_RATIO_BAR)
    # resolved-absent readings (frozen in the operationalizations)
    bonus_absent_resolved = bool(
        (walk_kill is not None and bonus_ratio is not None
         and bonus_ratio > BONUS_RATIO_BAR)
        or (walk_kill is None and static_edge is not None
            and walk_max_D >= static_edge))
    flight_absent_resolved = bool(
        (flight_ratio is not None and flight_ratio > FLIGHT_RATIO_BAR)
        or (u1_kill is None and static_edge is not None))
    bonus_state = ("PRESENT" if bonus_present else
                   ("absent" if bonus_absent_resolved else "UNRESOLVED"))
    flight_state = ("PRESENT" if flight_present else
                    ("absent" if flight_absent_resolved else "UNRESOLVED"))
    fires_causes = bool(alive and (bonus_present or flight_present))
    fires_noflight = bool(alive and not (bonus_present or flight_present)
                          and bonus_absent_resolved
                          and flight_absent_resolved)
    e195_ratio_u1 = E195_ROOT_DKILLS["root_u1"] / E195_ROOT_DKILLS["root_u0"]
    e196_ratio_u1 = E196_ROOT_DKILLS["root_u1"] / E196_ROOT_DKILLS["root_u0"]
    org1_bonus = E194_PATH_DKILL / E194_STATIC_EDGE
    org2_fullstep_bonus = E193_ASIGN_STOP["D_kill"] / E196_ROOT_DKILLS["root_u0"]

    if SMOKE:
        verdict, clause, bars = "SMOKE", "shakedown — nothing adjudicated", {}
    else:
        bars = {
            "ALIVE_WINDOW_CAUSES": {
                "fires": fires_causes,
                "detail": {"alive_at_theta1": alive,
                           "walk_s1_gm": ws1["gm"],
                           "recomputation_bonus": {
                               "state": bonus_state,
                               "walk_D_kill": walk_kill,
                               "static_u0_edge": static_edge,
                               "ratio": bonus_ratio,
                               "bar": BONUS_RATIO_BAR,
                               "bracket": (wsub["stop"].get("dens")
                                           if walk_kill is not None else None),
                               "org1_committed_ratio": org1_bonus,
                               "org2_fullstep_committed_ratio":
                                   org2_fullstep_bonus},
                           "flight_concentration": {
                               "state": flight_state,
                               "u1_D_kill": u1_kill,
                               "u0_edge": static_edge,
                               "ratio": flight_ratio,
                               "bar": FLIGHT_RATIO_BAR,
                               "org1_committed_ratio": e195_ratio_u1,
                               "org2_deadlineage_committed_ratio":
                                   e196_ratio_u1}},
            },
            "ALIVE_BUT_NO_FLIGHT": {
                "fires": fires_noflight,
                "detail": {"alive_at_theta1": alive,
                           "bonus_state": bonus_state,
                           "flight_state": flight_state},
            },
            "GRADED": {"fires": not (fires_causes or fires_noflight)},
        }
        fmt = lambda v: ("None" if v is None else f"{v:.4f}")
        if fires_causes:
            which = (" + ".join([n for n, p in
                                 (("the recomputation bonus", bonus_present),
                                  ("the flight concentration",
                                   flight_present)) if p]))
            verdict = "ALIVE-WINDOW-CAUSES"
            clause = (f"the sub-step lineage (s* {SUB_S:.6f} = "
                      f"STEP_L2/{round(1/SUB_FRAC)}) is ALIVE at theta_1 "
                      f"(g-4 {ws1['gm']:.4f} vs the full-step's dead "
                      f"{E196_TH1_GM:.4f}) and shows {which} — "
                      + (f"walk kills at {fmt(walk_kill)} vs static edge "
                         f"{fmt(static_edge)} (ratio {fmt(bonus_ratio)}, "
                         f"bar {BONUS_RATIO_BAR}); "
                         if bonus_present else
                         f"bonus {bonus_state} (walk {fmt(walk_kill)}, "
                         f"static {fmt(static_edge)}); ")
                      + (f"u1 kills at {fmt(u1_kill)} vs u0 edge "
                         f"{fmt(static_edge)} (ratio {fmt(flight_ratio)} "
                         f"< {FLIGHT_RATIO_BAR})" if flight_present else
                         f"flight {flight_state} (u1 {fmt(u1_kill)}, u0 "
                         f"{fmt(static_edge)}, ratio {fmt(flight_ratio)})")
                      + " — the alive window is the enabling condition; "
                        "the two organisms' asymmetry explained causally "
                        "(org 1 alive-with-effects, org 2 dead-without, "
                        "org 2 made-alive shows them).")
        elif fires_noflight:
            verdict = "ALIVE-BUT-NO-FLIGHT"
            clause = (f"the lineage is ALIVE past t=1 (theta_1 g-4 "
                      f"{ws1['gm']:.4f}) but shows neither effect — bonus "
                      f"{bonus_state} (walk {fmt(walk_kill)} vs static "
                      f"{fmt(static_edge)}, ratio {fmt(bonus_ratio)}), "
                      f"flight {flight_state} (u1 {fmt(u1_kill)} vs u0 "
                      f"{fmt(static_edge)}, ratio {fmt(flight_ratio)}) — "
                      "the alive window is necessary but not sufficient; "
                      "something else of organism 1 carries the dynamics; "
                      "reported as the honest fork.")
        else:
            verdict = "GRADED"
            clause = ("any partial pattern — the profiles verbatim: alive "
                      f"at theta_1 {alive} (g-4 {ws1['gm']:.4f}); walk "
                      f"stop {wsub['stop']['kind']} "
                      f"({'; '.join(f'{k}={fmt(v)}' for k, v in wsub['stop'].items() if isinstance(v, (int, float)))}); "
                      f"root-panel D_kill = "
                      + ", ".join(f"{k}:{fmt(families[k]['D_kill'])}"
                                  for k in families if k.startswith("root"))
                      + "; theta_1 panel (ALIVE anchor, never adjudicates): "
                      + ", ".join(f"{k}:{fmt(families[k]['D_kill'])}"
                                  for k in families if k.startswith("th1"))
                      + f"; bonus {bonus_state} (ratio {fmt(bonus_ratio)}),"
                      f" flight {flight_state} (ratio {fmt(flight_ratio)}).")
    log("=" * 78)
    log(f"E197 VERDICT: {verdict}")
    log(f"  {clause}")
    log("=" * 78)

    # =====================================================================
    # outputs
    # =====================================================================
    metrics = {
        "experiment": "e197_alive_window",
        "date": common.now_iso(),
        "status": ("SMOKE — shakedown (nothing adjudicated)" if SMOKE else
                   "COMPLETE — adjudicated (this write replaces all PARTIAL "
                   "progressive writes)"),
        "registration": REGISTERED_BARS["registration"],
        "registered_bars": REGISTERED_BARS,
        "question": ("is the alive window the CAUSE of the two dynamic "
                     "effects (the recomputation bonus, the flight "
                     "concentration)? — a sub-step lineage of organism 2 "
                     "(same stream, same direction machinery, size s* from "
                     "the frozen ladder) stays alive past t=1; its flight "
                     "structure read against its own static edges, with "
                     "organism 1 (alive-with-effects) and organism 2's "
                     "dead lineage (dead-without) as the committed "
                     "contrasts"),
        "organism": {
            "root": f"runs/checkpoints/{ROOT_CK}",
            "root_meta": root_meta, "root_flat_md5": root_md5,
            "N_param": N_PARAM,
            "architecture": "4L/4H/128d/512-ctx TinyGPT (the e098 s4305 "
                            "line) vs organism-1's 6L/6H/192d/256-ctx — "
                            "architecture co-varies with lineage (e193's "
                            "disclosure, carried)",
            "step_l2_measured": STEP_L2,
            "step_l2_note": "THIS organism's measured AdamW step-1 L2 at "
                            "t=0 (organism 1's was 1.6543 — never ported); "
                            "it EXCEEDS both its static edges (g 0.20 / "
                            "sign 0.53), so its natural walk kills at "
                            "step 1",
            "sub_step": {"s": SUB_S, "fraction": SUB_FRAC,
                         "natural_step": STEP_L2,
                         "note": "the intervention: the direction machinery "
                                 "verbatim, the size the experimenter's "
                                 "(the frozen ladder's pick, registered "
                                 "before the walk)"},
        },
        "rulers": {
            "primary": {"battery": f"install-60 g{RULER_J:+d}",
                        "root_read": root_cells[f"g{RULER_J:+d}"],
                        "why": "e193's frozen ruler call (the max committed "
                               "root read among the e157 dial's seven "
                               "geometries); carried verbatim"},
            "co_rulers": {f"g{j:+d}": root_cells[f"g{j:+d}"]
                          for j in CO_RULERS_J},
            "disclosure": "the e192-verbatim g-12 ruler reads 0.1983 at "
                          "this root — UNDER the 0.27 kill bar at D=0 "
                          "(e157 stage-A's G_CONS failure); co-rulers on "
                          "every row and every walk step, never adjudicated",
            "ce_r": root_cells["ce_r"],
        },
        "cell": {
            "licensed_cell": "e193's stream/step convention VERBATIM "
                             "(seed 10902, draw order, post-clip "
                             "gradients, measured STEP_L2, threads 8); "
                             "the full-step k=1 walk rebuilt bit-exactly "
                             "(the dead baseline) and the sub-step walk "
                             "run on the SAME stream with size s*",
            "grid": {"D": D_GRID,
                     "extras": {"root_u0": [STEP_L2, SUB_S] + ladder_Ds,
                                "th1_u1": [SUB_S]},
                     "note": "e195's D grid VERBATIM (0.05..3.00 step "
                             "0.05); D=0 anchor rows in every family; "
                             "e193's committed R2_SIGN rows re-gated at "
                             "all 16 shared Ds inside root-u0"},
            "rays": rays_meta,
            "anchors": anchors_meta,
            "input_seed": FREEZE_SEED,
            "matched_step_L2": {"value": STEP_L2,
                                "sub_s": SUB_S,
                                "provenance": "e193's committed MEASURED "
                                "step-1 L2 (recomputed at t=0 and gated "
                                "G_T0/G_REPRO); the sub-step walk's size "
                                "is the frozen ladder's pick"},
        },
        "gates": stub["gates"],
        "phase0_fullstep_baseline": stub["phases_partial"]["0_fullstep_baseline"],
        "phaseS_substep_choice": sub_choice,
        "phase1_alive_walk": stub["phases_partial"]["1_alive_walk"],
        "phase1b_rays": stub["phases_partial"]["1b_rays"],
        "profiles": {k: dict(v) for k, v in families.items()},
        "ray_geometry": ray_geometry,
        "alignment_reads": alignment_reads,
        "adjudication": {
            "bars": bars, "verdict": verdict, "clause": clause,
            "composite_order": "ALIVE-WINDOW-CAUSES -> ALIVE-BUT-NO-FLIGHT "
                               "-> GRADED (frozen before compute)",
            "root_panel": {k: families[k]["D_kill"] for k in families
                           if k.startswith("root")},
            "theta1_panel": {k: families[k]["D_kill"] for k in families
                             if k.startswith("th1")},
            "theta1_panel_note": "ALIVE anchor this time (g-4 "
                                 f"{ws1['gm']:.6f} > 0.27): kill-Ds "
                                 "resolve; the panel rides verbatim and "
                                 "never adjudicates",
            "recomputation_check": {
                "walk_stop": wsub["stop"], "walk_max_D": walk_max_D,
                "static_u0_edge": static_edge,
                "ratio": bonus_ratio, "bar": BONUS_RATIO_BAR,
                "state": bonus_state,
                "side_by_side": {
                    "org1_e194": {"path": E194_PATH_DKILL,
                                  "static": E194_STATIC_EDGE,
                                  "ratio": org1_bonus,
                                  "reading": "23% inversion — PRESENT"},
                    "org2_fullstep_e193": {
                        "path": E193_ASIGN_STOP["D_kill"],
                        "static": E196_ROOT_DKILLS["root_u0"],
                        "ratio": org2_fullstep_bonus,
                        "reading": "+0.1% — ABSENT (the dead lineage)"},
                    "org2_substep_e197": {"path": walk_kill,
                                          "static": static_edge,
                                          "ratio": bonus_ratio,
                                          "reading": bonus_state}},
            },
            "flight_check": {
                "u1_kill": u1_kill, "u0_edge": static_edge,
                "ratio": flight_ratio, "bar": FLIGHT_RATIO_BAR,
                "state": flight_state,
                "side_by_side": {
                    "org1_e195": {"u1": E195_ROOT_DKILLS["root_u1"],
                                  "u0": E195_ROOT_DKILLS["root_u0"],
                                  "ratio": e195_ratio_u1,
                                  "reading": "83% concentration — PRESENT"},
                    "org2_deadlineage_e196": {
                        "u1": E196_ROOT_DKILLS["root_u1"],
                        "u0": E196_ROOT_DKILLS["root_u0"],
                        "ratio": e196_ratio_u1,
                        "reading": "softest direction — ABSENT"},
                    "org2_alivelineage_e197": {"u1": u1_kill,
                                               "u0": static_edge,
                                               "ratio": flight_ratio,
                                               "reading": flight_state}},
            },
            "constants": {"SHUT_BAR": SHUT_BAR, "D_grid": D_GRID,
                          "FLIGHT_RATIO_BAR": FLIGHT_RATIO_BAR,
                          "BONUS_RATIO_BAR": BONUS_RATIO_BAR,
                          "STEP_L2": STEP_L2, "SUB_S": SUB_S},
        },
        "references": {
            "e193_organism_replicate": {
                "metrics": "runs/e193/metrics.json",
                "ckpt": str(CKPT_DIR / DIR_CK),
                "role": "THE PARENT: organism 2's committed state + rays "
                        "(the a_sign step-1 row + kill bracket — the "
                        "DEAD-at-t=1 baseline; the R2_SIGN profile; the "
                        "g-ray direction checkpoint; the measured "
                        "STEP_L2) — loaded COMMITTED, gated, never rerun"},
            "e196_flight_replicate": {
                "metrics": "runs/e196/metrics.json",
                "role": "the DEAD-lineage flight contrast (u1 its SOFTEST "
                        "direction, ratio 2.585; both dynamic effects "
                        "absent) + the machinery this cell ports"},
            "e195_rotated_ray": {
                "metrics": "runs/e195/metrics.json",
                "role": "organism 1's committed flight result "
                        "(root_u1 0.3875 vs root_u0 2.2699, "
                        "FLEEING-IS-LETHAL) — the alive-with-effects "
                        "contrast, never a bar"},
            "e194_sign_front": {
                "metrics": "runs/e194/metrics.json",
                "role": "organism 1's committed recomputation bonus "
                        "(path 1.7496 vs static 2.2699) — the bonus "
                        "side-by-side, never a bar"},
            "e157": {"metrics": "runs/e157/metrics.json",
                     "role": "this root's committed dial (G_ROOT) + the "
                             "wash step-1 anchors (G_T0/G_S1CK)"},
        },
        "honesty_reflex": {
            "n_and_scope": ("n=1 root, n=1 stream (seed 10902, md5-gated "
                            "at every consumed step): this cell intervenes "
                            "on ONE organism's ONE trajectory — the "
                            "alive-window claim is causal FOR THIS "
                            "BIOGRAPHY (organism 2's f2 lineage), not a "
                            "population claim; organism 1 remains the "
                            "only alive natural trajectory on record"),
            "sub_step_deviation": ("the sub-step lineage is a "
                                   "COUNTERFACTUAL wash: the natural AdamW "
                                   "step (L2 0.9164) is what killed the "
                                   "organism at t=1; halving the size "
                                   "keeps the direction machinery verbatim "
                                   "but changes the trajectory — any "
                                   "flight structure found on it belongs "
                                   "to the alive window OF THIS "
                                   "CONSTRUCTION, and the step back to "
                                   "the natural trajectory is an "
                                   "inference, disclosed"),
            "openness": ("WHAT EACH ARM GUARANTEES: NOTHING — every "
                         "family is a static graded jump at lethal scale "
                         "from a state that could kill anywhere; the "
                         "walk is one realized path of a stochastic "
                         "stream; that openness is the point"),
            "estimator_lesson": ("every alignment read states its "
                                 "evaluation point (matched-point AT the "
                                 "anchor, dual form: walked-delta vs "
                                 "unit-ray cross-consistency) — T150's "
                                 "lesson; no cross-organism cosine "
                                 "comparisons (different instruments)"),
            "projections_never_adjudicate": ("D_kill is a linear-in-D "
                                             "interpolation on measured "
                                             "grid points; the 0.85/0.70 "
                                             "margins were frozen before "
                                             "compute; the walk kill's "
                                             "bracket resolution at s* is "
                                             "~s*/5 (disclosed)"),
            "float_texture": ("CPU fp32 texture, this process, 8 threads "
                              "(e193's gate convention); the full-step "
                              "walk reproduced e193's committed a_sign row "
                              "and kill bracket to "
                              f"{max(abs(vv[0] - vv[1]) for vv in repro_rows.values()):.1e}"),
        },
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "config": {"n_layer": 4, "n_head": 4, "n_embd": 128,
                   "block_size": 512, "params": N_PARAM,
                   "train_device": "cpu (CUDA_VISIBLE_DEVICES=-1)",
                   "torch_threads": torch.get_num_threads(),
                   "smoke": SMOKE, "torch": torch.__version__,
                   "eval_only": True},
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))

    plot_alive_window(rd / "e197_alive_window.png", wfull, wsub, ws1,
                      families, static_edge, u1_kill, bonus_ratio,
                      flight_ratio, walk_kill,
                      root_gm=root_cells[f"g{RULER_J:+d}"],
                      sub_frac=SUB_FRAC)
    plot_kill_summary(rd / "e197_kill_summary.png", families, wsub,
                      ray_geometry, alignment_reads)
    log(f"outputs: {rd / 'metrics.json'} + 2 PNGs; total "
        f"{time.time() - T0:.1f}s")
    return 0


# ------------------------------------------------------------------ plots

RAY_STYLE = {
    "u0": {"color": "navy", "label": "sign(g_0) — the static sign ray"},
    "u1": {"color": "crimson",
           "label": "sign(g_1) — THE ALIVE FLIGHT RAY (t=1 front, alive)"},
    "u2": {"color": "darkorange", "label": "sign(g_2) — the t=2 front"},
}


def _dual_axis(ax):
    ax.set_xlabel(r"static displacement $D = \|\theta_D - \theta_{anchor}\|_2$")
    axr = ax.secondary_xaxis(
        "top", functions=(lambda d: d / RMS_DENOM,
                          lambda r: r * RMS_DENOM))
    axr.set_xlabel("per-coordinate RMS (organism 2's currency, / 934.597)")


def plot_alive_window(path, wfull, wsub, ws1, families, static_edge,
                      u1_kill, bonus_ratio, flight_ratio, walk_kill,
                      root_gm, sub_frac):
    """THE ALIVE-WINDOW MAP: the two walks (dead baseline vs the alive
    lineage) + the alive lineage's flight map + the three-lineage ratio
    side-by-side (org 1, org 2 dead, org 2 ALIVE — the committed curves)."""
    fig, axes = plt.subplots(1, 4, figsize=(24.5, 6.6))

    # (0) the two walks
    ax = axes[0]
    ft = wfull["traj"]
    ax.plot([0.0] + [r["cum_disp"] for r in ft],
            [root_gm] + [r["gm"] for r in ft], "X--", color="dimgray",
            ms=9, lw=1.5, label="FULL-STEP walk (natural size 0.9164) — "
                               "DEAD at t=1")
    ax.plot([r["cum_disp"] for r in wsub["traj"]],
            [r["gm"] for r in wsub["traj"]], "o-", color="seagreen",
            ms=5, lw=1.8,
            label=f"SUB-STEP walk (s* {wsub['traj'][0]['step_disp']:.4f}) — "
                  f"ALIVE at t=1 ({ws1['gm']:.3f})")
    for r in wsub["traj"]:
        if r["post_kill"]:
            ax.plot([r["cum_disp"]], [r["gm"]], "o", color="seagreen",
                    alpha=0.35, ms=5)
    ax.axhline(SHUT_BAR, ls="--", lw=1.3, color="tab:purple",
               label=f"{SHUT_BAR} SHUT bar (g-4 ruler)")
    ax.axvline(0.0, color="k", lw=0.6)
    if static_edge is not None:
        ax.axvline(static_edge, color="navy", ls=":", lw=1.2)
        ax.text(static_edge + 0.05, 0.5,
                f"static u0 edge {static_edge:.3f}", fontsize=7,
                rotation=90, color="navy")
    if walk_kill is not None:
        ax.axvline(walk_kill, color="seagreen", ls=":", lw=1.2)
        ax.text(walk_kill - 0.05, 0.8, f"walk kills {walk_kill:.3f}",
                fontsize=7, rotation=90, color="seagreen", ha="right")
    ax.set_xlabel("cumulative displacement D along the walk")
    ax.set_ylabel("g-4 (install-60 battery mean p(Z))")
    ax.set_ylim(-0.03, 1.02)
    ax.set_title("THE INTERVENTION — dead baseline vs the ALIVE lineage",
                 fontsize=10)
    ax.legend(fontsize=6.8, loc="upper right")

    # (1) the root panel
    ax = axes[1]
    for key in ("root_u0", "root_u1", "root_u2"):
        if key not in families:
            continue
        fam = families[key]
        st = RAY_STYLE[fam["ray"]]
        ax.plot([r["D"] for r in fam["rows"]],
                [r["gm"] for r in fam["rows"]], "o-", ms=2.6,
                lw=1.7, color=st["color"], label=st["label"])
        if fam["D_kill"] is not None:
            ax.axvline(fam["D_kill"], color=st["color"], ls=":", lw=1.4)
            ax.text(fam["D_kill"], 0.985,
                    f"{fam['ray']} kills {fam['D_kill']:.3f}",
                    rotation=90, fontsize=6.6, va="top", ha="right",
                    color=st["color"])
    ax.axvline(E196_ROOT_DKILLS["root_u1"], color="crimson", ls="-.",
               lw=1.3, alpha=0.55)
    ax.text(E196_ROOT_DKILLS["root_u1"] + 0.04, 0.5,
            f"e196 DEAD-lineage u1 kills {E196_ROOT_DKILLS['root_u1']:.3f}"
            " (softest)", fontsize=6.6, rotation=90, color="crimson",
            alpha=0.8)
    if static_edge is not None:
        ax.axvline(FLIGHT_RATIO_BAR * static_edge, color="crimson",
                   ls=":", lw=1.6)
        ax.text(FLIGHT_RATIO_BAR * static_edge - 0.04, 0.9,
                f"flight bar {FLIGHT_RATIO_BAR} x u0 edge",
                fontsize=7, color="crimson", ha="right")
    ax.axhline(SHUT_BAR, ls="--", lw=1.3, color="tab:purple")
    _dual_axis(ax)
    ax.set_ylabel("g-4 — org 2's ruler")
    ax.set_ylim(-0.03, 1.02)
    ax.set_title("THE ALIVE LINEAGE'S FLIGHT MAP — from the root",
                 fontsize=10)
    ax.legend(fontsize=6.8, loc="upper right")

    # (2) the theta_1 panel (ALIVE this time)
    ax = axes[2]
    for key in ("th1_u0", "th1_u1", "th1_u2"):
        if key not in families:
            continue
        fam = families[key]
        st = RAY_STYLE[fam["ray"]]
        ax.plot([r["D"] for r in fam["rows"]],
                [r["gm"] for r in fam["rows"]], "o-", ms=2.6,
                lw=1.7, color=st["color"],
                label=st["label"].split("—")[0].strip() + f" ({fam['ray']})")
        if fam["D_kill"] is not None:
            ax.axvline(fam["D_kill"], color=st["color"], ls=":", lw=1.4)
            ax.text(fam["D_kill"], 0.985,
                    f"{fam['ray']} kills {fam['D_kill']:.3f}",
                    rotation=90, fontsize=6.6, va="top", ha="right",
                    color=st["color"])
    ax.axhline(SHUT_BAR, ls="--", lw=1.3, color="tab:purple",
               label=f"{SHUT_BAR} SHUT bar")
    _dual_axis(ax)
    ax.set_ylabel("g-4 — org 2's ruler")
    ax.set_ylim(-0.03, 1.02)
    ax.set_title(f"FROM $\\theta_1$ — THE ALIVE ANCHOR "
                 f"(g-4 {ws1['gm']:.3f}; e196's was dead 0.0068)",
                 fontsize=10)
    ax.legend(fontsize=6.8, loc="upper right")

    # (3) the three-lineage ratio side-by-side
    ax = axes[3]
    labels, vals, cols = [], [], []
    labels.append("org 1 flight u1/u0\n(e195, alive lineage)")
    vals.append(E195_ROOT_DKILLS["root_u1"] / E195_ROOT_DKILLS["root_u0"])
    cols.append("crimson")
    labels.append("org 2 flight u1/u0\n(e196, DEAD lineage)")
    vals.append(E196_ROOT_DKILLS["root_u1"] / E196_ROOT_DKILLS["root_u0"])
    cols.append("dimgray")
    if flight_ratio is not None:
        labels.append("org 2 flight u1/u0\n(e197, ALIVE lineage)")
        vals.append(flight_ratio)
        cols.append("seagreen")
    labels.append("org 1 recompute path/static\n(e194, alive lineage)")
    vals.append(E194_PATH_DKILL / E194_STATIC_EDGE)
    cols.append("crimson")
    labels.append("org 2 recompute path/static\n(e193, DEAD lineage)")
    vals.append(E193_ASIGN_STOP["D_kill"] / E196_ROOT_DKILLS["root_u0"])
    cols.append("dimgray")
    if bonus_ratio is not None:
        labels.append("org 2 recompute walk/static\n(e197, ALIVE lineage)")
        vals.append(bonus_ratio)
        cols.append("seagreen")
    ys = np.arange(len(labels))
    ax.barh(ys, vals, color=cols, alpha=0.85)
    for i, v in enumerate(vals):
        ax.text(v + 0.04, i, f"{v:.3f}", va="center", fontsize=7.6)
    ax.axvline(FLIGHT_RATIO_BAR, color="crimson", ls=":", lw=1.5)
    ax.text(FLIGHT_RATIO_BAR - 0.03, -0.55, f"flight bar {FLIGHT_RATIO_BAR}",
            fontsize=7, color="crimson", ha="right")
    ax.axvline(BONUS_RATIO_BAR, color="seagreen", ls=":", lw=1.5)
    ax.text(BONUS_RATIO_BAR - 0.03, len(labels) - 0.45,
            f"bonus bar {BONUS_RATIO_BAR}", fontsize=7, color="seagreen",
            ha="right")
    ax.axvline(1.0, color="k", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7.0)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_xlabel("ratio (below 1 = the dynamic effect present)")
    ax.set_title("THE THREE-LINEAGE SIDE-BY-SIDE (committed curves, "
                 "never bars)", fontsize=10)
    fig.suptitle("E197 — THE ALIVE WINDOW: organism 2's sub-step lineage "
                 f"(s* = STEP_L2/{round(1 / sub_frac)}) — does the flight "
                 "structure form past t=1?", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_kill_summary(path, families, wsub, ray_geometry, alignment_reads):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    ax = axes[0]
    keys = [k for k in ("root_u0", "root_u1", "root_u2",
                        "th1_u0", "th1_u1", "th1_u2") if k in families]
    vals = [families[k]["D_kill"] for k in keys]
    cols = [RAY_STYLE[k.split("_")[1]]["color"] for k in keys]
    xs = np.arange(len(keys))
    ax.bar(xs, [v if v is not None else 0.0 for v in vals], color=cols,
           alpha=0.88)
    for i, v in enumerate(vals):
        if v is not None:
            ax.text(i, v + 0.03, f"{v:.4f}", ha="center", fontsize=8)
        else:
            ax.text(i, 0.03, "no downcrossing\n<= 3.0", ha="center",
                    fontsize=7)
    st = families["root_u0"]["D_kill"]
    if st is not None:
        ax.plot([-0.4, len(keys) - 0.6],
                [FLIGHT_RATIO_BAR * st] * 2, ls=":", lw=1.6,
                color="crimson")
        ax.text(len(keys) - 0.58, FLIGHT_RATIO_BAR * st,
                f"0.70 x root-u0 = {FLIGHT_RATIO_BAR * st:.3f}",
                fontsize=7, color="crimson", va="center", ha="right")
        ax.plot([-0.4, len(keys) - 0.6], [BONUS_RATIO_BAR * st] * 2,
                ls=":", lw=1.6, color="seagreen")
        ax.text(len(keys) - 0.58, BONUS_RATIO_BAR * st,
                f"0.85 x root-u0 = {BONUS_RATIO_BAR * st:.3f}",
                fontsize=7, color="seagreen", va="center", ha="right")
    ax.axvline(2.5, color="k", lw=0.8, ls="--")
    ymax = max([v for v in vals if v is not None], default=0.5)
    ax.text(2.5, ymax * 0.9 + 0.1, "panel switch (theta_1: ALIVE anchor)",
            fontsize=7.5, ha="center")
    ax.axhline(E193_KILLS["sign"], color="dimgray", ls="-.", lw=1.2)
    ax.text(0.02, E193_KILLS["sign"] + 0.02,
            f"e193 static sign grid-kill {E193_KILLS['sign']}",
            fontsize=7, color="dimgray")
    ax.set_xticks(xs)
    ax.set_xticklabels(keys, fontsize=8)
    ax.set_ylabel("D_kill (first 0.27 downcrossing, interpolated)")
    ax.set_title("THE KILL LADDER — the ALIVE lineage of organism 2 "
                 "(0.70 flight bar / 0.85 bonus bar on root-u0)",
                 fontsize=10)

    ax = axes[1]
    geo_rows = [
        ("cos(u0,u1)", ray_geometry["cos_u0_u1"],
         ray_geometry["organism2_deadlineage_committed"]["cos_u0_u1"]),
        ("cos(u0,u2)", ray_geometry["cos_u0_u2"],
         ray_geometry["organism2_deadlineage_committed"]["cos_u0_u2"]),
        ("cos(u1,u2)", ray_geometry["cos_u1_u2"],
         ray_geometry["organism2_deadlineage_committed"]["cos_u1_u2"]),
        ("cos(u1, u_g)", ray_geometry["cos_u1_ug"], None),
        ("align root: cos(-u1, g-4 grad)",
         alignment_reads["root_anchor"]["cos_neg_u1"],
         alignment_reads["organism2_deadlineage_committed"]
         ["root_cos_neg_u1"]),
        ("align root: cos(-u0, g-4 grad)",
         alignment_reads["root_anchor"]["cos_neg_u0"], None),
        ("align th1: cos(-u1, g-4 grad)",
         alignment_reads["theta1_anchor"]["cos_neg_u1"],
         alignment_reads["organism2_deadlineage_committed"]
         ["theta1_cos_neg_u1"]),
    ]
    labels = [r[0] for r in geo_rows]
    vals2 = [r[1] for r in geo_rows]
    cols2 = ["navy", "navy", "navy", "seagreen", "crimson", "navy",
             "crimson"]
    ax.barh(np.arange(len(geo_rows)), vals2, color=cols2, alpha=0.85)
    for i, r in enumerate(geo_rows):
        txt = f"{r[1]:+.4f}"
        if r[2] is not None:
            txt += f"  (e196 dead-lineage {r[2]:+.4f})"
        ax.text(r[1] + (0.012 if r[1] >= 0 else -0.012), i, txt,
                va="center", fontsize=7.2,
                ha="left" if r[1] >= 0 else "right")
    ax.set_yticks(np.arange(len(geo_rows)))
    ax.set_yticklabels(labels, fontsize=7.6)
    ax.axvline(0.0, color="k", lw=0.8)
    ax.axvline(1.0 / (F2_PARAMS ** 0.5), color="tab:purple", ls="--",
               lw=1.0)
    ax.text(0.0012, -0.75, "iso floor", fontsize=6.5, color="tab:purple",
            rotation=90)
    ax.set_xlim(-0.35, 0.9)
    ax.invert_yaxis()
    ax.set_title("RAY GEOMETRY (e196 dead-lineage committed in "
                 "parentheses) + the matched-point alignment reads",
                 fontsize=9)
    fig.suptitle("E197 — the ALIVE lineage's kill ladder and ray geometry",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
