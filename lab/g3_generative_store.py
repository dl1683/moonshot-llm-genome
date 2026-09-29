"""G3 — THE GENERATIVE-MEMORY ARCHITECTURE (the Hopfield store).

W020's generative turn, row 3 of the g-series. THE REGISTERED QUESTION: does
explicitly-generative storage (the fact as an explicit PATTERN MATRIX in a
Hopfield retrieval layer, prediction-as-storage implemented literally) change
the basin physics, or is the no-basin law substrate-independent? Design spec
COMMITTED at scratch/g3_design.md (registered 2026-09-29) — its predictions
and bars are FROZEN and adjudicated verbatim below. No bar shopping.

ARMS (host = family 2, the e098 s4305 line, F2 4L/4H/128d/512-ctx, trunk
873,472 params, loaded from the PRISTINE e098_base_s4305.pt so the fact is
carried by the STORE, not by a pre-installed discriminative pathway):
  - g3-GEN  — one position-wise Hopfield block grafted AFTER block 3, BEFORE
              ln_f: q=W_q.LN_noaffine(h) (128->64); a=softmax(beta*cos(q,K)),
              beta=8 FIXED; r=a@V; h<-h+W_o(r) (64->128); 8 patterns = 7
              name-step context prototypes + 1 null gate (anchor-query mean,
              V_null=0). 17,410 params; total 890,882 (~0.89M <= 1M).
  - g3-SHAL — IDENTICAL graft/query/injection/training, ONLY the middle
              differs: 64->48 GELU->64 MLP (content IN WEIGHTS). ~22.6k.
  - g3-HARD — GEN with the gate HARDENED: top-1 argmax retrieval,
              straight-through gradients (arm D, the decoupling control).
  - S-DISC  — the deep discriminative control, FREE: e157_f2_consolidated +
              its stored neutral-wash cells (runs/e157/metrics.json); its
              displacement column supplied by MEASURING e157's stored
              checkpoints on disk (eval-only; e157 recorded none).

REGISTERED PREDICTION (scratch/g3_design.md, frozen before compute):
  THE WASH — G3-DIES, THE LAW HOLDS: the soft-gated generative store
    dissolves with the law — g0 <= 0.27 by +50 (NEUTRAL-DISSOLVES bar), point
    prediction for the clock: first-under-bar in (2, 20] — the plateau buys
    at most a small-factor discount off the discriminative +1, never
    survival. MECHANISM: the kill site is the GAIN — the soft gate's
    exponential tails leak name content onto every neutral token; the small
    but sign-consistent leak gradient collapses W_o/V at the standard clock.
    SAVOR: at the kill the RETRIEVAL IS STILL INTACT (the fact key still
    wins the argmax on fact queries) — STORAGE WITHOUT EXPRESSION. THE
    TWO-BASIN DISSOCIATION: the state basin measures WIDE while the
    parameter basin measures NARROW (readout dies at wash-direction
    store-displacement comparable to the discriminative kill displacement).
    One line: THE WASH DOES NOT TRAVERSE THE ENERGY LANDSCAPE — IT
    RE-SCULPTS IT.
  THE NOISE KILL — NOISE-KILLS: both noise arms kill g3-GEN at
    displacement-match (a fortiori below), COLLATERAL (CE >= 3.0, the
    organism dies with the fact) — no generative immunity.
  ARM D — D-SURVIVES (g0 >= 0.50 through +300) IF the GAIN channel is the
    kill; D-DIES relocates the law's entry point to a shared door (W_q drift
    or null-key encroachment — the census adjudicates).
  S-SHAL — dies in the same bracket as g3-GEN (within ~2x of its clock); if
    g3-GEN >> g3-SHAL survival, the PLATEAU DISCOUNT becomes the finding.
  THE RESURRECTION RIDER — fires: one replay event restores g0 >= 0.5
    within 50 wash steps (the re-entry economy is substrate-independent).

FALSIFIERS (spec verbatim): the LAW BREAKS if g3-GEN g0 >= 0.50 at EVERY
  continuation checkpoint {1,2,4,10,50,100,200,300} at lr 1e-3 (g3-SHAL's
  fate then discriminates GENERATIVITY from SHALLOWNESS); the MECHANISM
  prediction dies if the kill site is not the gain, or retrieval is not
  intact at the kill; the NOISE prediction dies if either arm spares at
  displacement-match; the COUPLING claim dies if g3-HARD also dies; the
  state-basin claim dies if the query sweeps show a narrow state basin.

OPERATIONALIZATIONS (frozen; they fix the clauses, they do not move bars):
  * Primary dial g0 = ABSOLUTE install-60 battery mean p(Z) at ctx offset 0
    (e176/e176N/e157's ruler, same corpus rebuild); g-12/g+12/held30/span are
    co-reports. DISSOLVE = g0 <= 0.27 by the +50 checkpoint (earliest
    under-bar checkpoint co-reported = t*); SURVIVE = g0 >= 0.50 at EVERY
    checkpoint through +300.
  * Root gates (a failed root gate aborts that arm to CONSTRUCTION-CEILING;
    no wash adjudication): G_ROOT_EXPR g0 >= 0.50; G_STOREOFF store-output
    zeroed at eval (forward flag, no weight edits) g0 <= 0.27; G_CECLEAN
    CE_R <= base + 0.10; G_NAMEFREE/G_SPLICE/G_POOL/G_ANCHOR e176n/e157
    verbatim. ONE joint calibration (host+store, <= 100 steps, lr 3e-4,
    CE_R gate) allowed on a G_ROOT_EXPR miss, recorded.
  * WASH = e157's ported e176N arm A VERBATIM (neutral bank seed 170,
    batch 16 neutral + 16 random, full-token CE, AdamW (0.9,0.95) wd 0.1
    lr 1e-3 clip 1.0, seed 10902), 300 steps, checkpoints
    {1,2,4,10,50,100,200,300} (+10 added for the noise displacement grid);
    lean battery per checkpoint, full dial at root/+10/+300.
  * DISPLACEMENT currency = cumulative ||theta_t - theta_0||_2 (fp32,
    measured): ALL params, the STORE subspace (the organ's tensors), the
    HOST subspace, and per store tensor (K/V/W_q/W_o) — e185's methodology.
  * NOISE (e185 ported, on g3-GEN): NOISE-LABELS (iid uniform targets, seed
    18501) + SHUFFLED-TARGET (flat randperm, seed 18502), 10 steps,
    checkpoints {1,2,4,10}, inputs bit-identical to A1 (per-step md5 gates);
    D_kill = A1's cumulative all-param displacement at t*; M = earliest
    noise checkpoint with disp >= D_kill; KILLS = g0 at M <= 0.27;
    COLLATERAL = CE_R at M >= 3.0; damage read at M and every later matched
    checkpoint through +10. SPARES = g0 >= 0.50 at every matched checkpoint.
  * KILL-SITE CENSUS at t* (definitions frozen): GAIN = argmax intact
    (>= 0.8 correct on fact queries), a_fact >= 0.5, ||inj|| <= 30% of
    root; QUERY = retrieval collapsed with the key geometry intact (root-q
    x washed-K still retrieves); GATE = the null key encroached (washed-K
    attribution); ROUTE = the bypass probe (root store into washed host)
    fails to restore >= 0.5 while the store's isolated retrieval is intact;
    else MIXED.
  * CLOSURE PARTITION (e173 restore_class, weight-level 2x2 at t*):
    {root,washed} store x {root,washed} host; root-store-into-washed-host =
    THE BYPASS PROBE; all-restored = bit-exact sanity gate.
  * TWO-BASIN MAP: (i) STATE BASIN — gamma-interpolate root fact queries
    toward corpus queries + gaussian query noise sigma (in ||q|| units;
    mean cosine = 1/sqrt(1+sigma^2)); retrieval fidelity (a_fact/argmax on
    root K) and behavioral readout (g0 battery with noisy queries) vs
    perturbation. WIDE = readout >= 0.5 at sigma >= 1.0 AND fidelity >= 0.5
    at gamma <= 0.5. (ii) PARAMETER BASIN — lambda-sweep the store's
    tensors along A1's MEASURED wash direction (theta_root + lambda .
    (theta_t* - theta_root), lambda in {0,.25,.5,1,2,4}) AND isotropic
    gaussian store noise at matched L2 (e011c's matched-energy ladder, 3
    seeds/level); readout vs store-displacement. NARROW = g0 <= 0.27 at
    lambda <= 2 on the wash direction.
  * RIDER (report-only, e179's protocol): from t*, ONE replay batch of the
    fact's windows (16 install-pool draws + 8 paired anchors + 8 random,
    union CE, lr 1e-3), then 50 neutral-wash steps; FIRES = g0 >= 0.5 at
    any read through +50.

COMPUTE ENVELOPE (spec): GPU allowed, gated per training (pick_dev
park-once + mid-run guard every 25 steps at mem > 85% or temp > 80C, e157
verbatim), cooldown ~75 s around each training, caps 180 s GPU / 1500 s CPU
per training; ALL readouts CPU-side; <= 1M total params (~0.89M); ZERO new
data; S-DISC free.

INSTRUMENT PROVENANCE: battery_cell/battery_pz/ce_fixed_cpu/val_windows/
deleted_wpe/read_fact_at/row_census/pick_dev/migrate_to_cpu are
lab/e157_lineage_replication.py VERBATIM (= e176n's copies; the e176/e161/
e152/e151/e143/e131/e119/e113/e068/e065/e043 lineage); the neutral bank +
junction accounting are e170's via e157's copy; wash_run is e157's
finetune_freeze with e185's noise_wash additions (target substitution +
displacement bookkeeping + per-step input hashes); restore_class is e173's
transplant gate; the construction trainer is e043's exposure arithmetic with
the g3 spec's own batch composition (16 install + 32 random anchors);
jsonable is e043's. Copied, not imported, to own the device policy. The ONLY
mechanical deviation from the trunk: the model class (GenMemGPT, a TinyGPT
subclass carrying the store organ) — recorded.

Outputs: runs/g3/{metrics.json, generative_store.png}; checkpoints
runs/checkpoints/g3_{gen,shal,hard}{,_sN}.pt. No NOTES/THINKING/QUEUE/STATE
edits; single commit, no push.

Run:  cd lab && python g3_generative_store.py   (G3_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # GPU allowed, gated below

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
import torch                                           # noqa: E402

torch.set_num_threads(8)                              # e143/e151/e157 convention

import torch.nn as nn                                  # noqa: E402
import torch.nn.functional as F                        # noqa: E402

import common                                          # noqa: E402
from common import (Cfg, CharCorpus, TinyGPT, cooldown, gpu_ok,     # noqa: E402
                    gpu_status, run_dir, save_json)

import e043_install as E43                             # noqa: E402 (REPO, find_occ, SPLICE_RNG, jsonable)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = os.environ.get("G3_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)

NAME = "ZEPHYRA"
PRE = 130
POST_CAP = 119                    # e043 deviation-1: 130+7+119 = 256
HOSTS = ["FLORIZEL", "ELIZABETH"]
BLOCK = 256                       # training-window block (the e098 line's own
                                  # install convention)
CKPT_DIR = E43.REPO / "runs" / "checkpoints"
BASE_CK = "e098_base_s4305.pt"    # the PRISTINE fact-free family-2 base

# ---- the family-2 net (e098 s4305 line) ---------------------------------------
F2_CFG = Cfg(vocab=65, n_layer=4, n_head=4, n_embd=128, block_size=512)
F2_PARAMS = 873_472

# ---- the store constants (design constants, frozen; NOT swept) -----------------
D_EMBD, D_KEY, N_PAT, BETA = 128, 64, 8, 8.0
STORE_PARAMS = D_EMBD * D_KEY + N_PAT * D_KEY + N_PAT * D_KEY + D_KEY * D_EMBD  # 17,410
NULL_IDX = 7                      # pattern 7 = the null gate
SHAL_HIDDEN = 48                  # the shallow twin's hidden width

# ---- placement constants (e152/e157 verbatim) ----------------------------------
RETEACH_J = 54                    # offset: name x-cols 184..190, read rows 183..189
SITE_ADDR_ROW = 183
SITE_Z_XCOL = PRE + RETEACH_J     # 184
SITE_ROWS = tuple(range(183, 190))
BAND_ADDR_ROW, BAND_Z_XCOL = PRE - 1, PRE    # 129, 130 (home position)
D_ALL = (121, 125, 129, 133, 137)             # e113's fixed set
GEOS = (-12, 0, 12)               # wash ruler: novel x2 + the trained g0

ROWS_OLD = (0, 1, 2, 3, 4, 5, 6, 60, 100, 118, 119, 120) + tuple(range(121, 130))
if SMOKE:
    ROWS_OLD = (0, 1, 60, 121, 125, 129)

# ---- trainings -----------------------------------------------------------------
CONS_LR = 1e-3
CONS_STEPS = 300 if not SMOKE else 80          # smoke: enough to pass the root
                                               # gate without calibration
                                               # (probe: 50 steps -> g0 0.63)
CONS_NAME_BS, CONS_ANC_BS = 16, 32          # the spec's batch: 112 vs 8,160 targets
CONS_SEED = 43050                           # construction seed (spec)
CAL_SEED = 43060                            # joint-calibration batch seed (recorded)
CAL_STEPS_MAX = 100
WASH_LR = 1e-3
WASH_STEPS = 300 if not SMOKE else 8
CK_WASH: tuple[int, ...] = (1, 2, 4, 10, 50, 100, 200, 300) if not SMOKE else (1, 2, 4)
ARM_SEED = 10902                            # e157/e176n wash seed VERBATIM
NOISE_SEED_A, NOISE_SEED_B = 18501, 18502   # e185's target RNGs VERBATIM
CK_NOISE: tuple[int, ...] = (1, 2, 4, 10) if not SMOKE else (1, 2)
RIDER_SEED = 10903                          # rider wash continuation (recorded)
RIDER_REPLAY_SEED = 10904                   # rider replay-batch draws (recorded)
RIDER_STEPS = 50 if not SMOKE else 8
CK_RIDER: tuple[int, ...] = (10, 25, 50) if not SMOKE else (2, 4, 8)
COOLDOWN_S = 75.0                           # spec envelope 60-90 s
GPU_CAP_S, CPU_CAP_S = 180.0, 1500.0
MIDRUN_POLL_EVERY = 25
EVAL_EVERY = 50 if not SMOKE else 2

# ---- two-basin sweep grids (frozen) --------------------------------------------
SIGMAS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)   # query-noise, ||q|| units
GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0)         # fact -> corpus interpolation
LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)    # along A1's measured wash direction
ISO_LEVELS = (0.25, 0.5, 1.0, 2.0, 4.0)      # x Dstore(t*) matched-L2 ladder
ISO_SEEDS = (11011, 11012, 11013)
SIGMA_SEED0, GAMMA_SEED = 11021, 11031

# ---- e170's neutral anchor bank (e176N arm A's stream, VERBATIM) --------------
E170_ANCHOR_SEED = 170
ANCHOR_FORBIDDEN = ("FLORIZEL", "ELIZABETH", "ZEPH", "MIRABEL")

# ---- gates / references ---------------------------------------------------------
R_EVAL_SEED = 26502               # e065 CE_R eval-bank seed (verbatim)
G_BIT_TOL = 5e-6
G_FALLBACK_TOL = 0.05

# S-DISC control cells (runs/e157/metrics.json stages.B_wash.trace — the family-2
# consolidated root + its neutral-wash cells, seed 10902; verified vs file at run
# time). Root g-12 0.198 was e157's recorded asymmetry (under-bar discriminative
# ruler); g0 0.578 / g+12 0.591 were well expressed — g0 is family-2's honest
# dial and is the primary currency here, for S-DISC and g3 alike.
SDISC_TRACE = {
    "freeze_steps": [0, 1, 2, 4, 50, 100, 200, 300],
    "g0": [0.5784125924110413, 0.0040224287658929825, 0.00459279865026474,
           0.015774426388776755, 0.020724201594921608, 0.002758212387561798,
           0.00029339289176277816, 0.00019846257065178285],
    "gm12": [0.19826222956180573, 0.001129803480580449, 0.0011759379515195233,
             0.003656684886664152, 0.007448808290064335, 0.004423586186021566,
             0.00012963238987126357, 0.00013348007632885128],
    "ce_r": [1.9504578113555908, 2.840304136276245, 2.7520101070404053,
             2.7039923667907715, 2.0670039651974076, 1.9851480722427368,
             1.8773754835128784, 1.8591028451919556],
}
SDISC_ROOT_CK = "e157_f2_consolidated.pt"
SDISC_WASH_CK = {1: "e157_f2_neutral_s1.pt", 2: "e157_f2_neutral_s2.pt",
                 4: "e157_f2_neutral_s4.pt", 50: "e157_f2_neutral_s50.pt",
                 100: "e157_f2_neutral_s100.pt", 200: "e157_f2_neutral_s200.pt",
                 300: "e157_f2_neutral.pt"}

# ---- registered bar constants (frozen) ------------------------------------------
SHUT_BAR = 0.27                   # dissolve (e176n's NEUTRAL-DISSOLVES bar)
SURVIVE_BAR = 0.50                # survive at EVERY checkpoint through +300
DISSOLVE_BY = 50                  # g0 <= 0.27 by +50
CLOCK_LO, CLOCK_HI = 2, 20        # t* point-prediction bracket (2, 20]
CE_COLLATERAL = 3.0               # the noise kill's collateral bar
A_FACT_BAR = 0.50                 # census: retrieval mass on fact queries
ARGMAX_BAR = 0.80                 # census: argmax-correct fraction
INJ_RATIO_BAR = 0.30              # census: ||inj|| ratio at fact positions
RESTORE_BAR = 0.50                # bypass-probe restore bar
RIDER_BAR = 0.50                  # rider: one replay restores within 50 steps
ROOT_EXPR_BAR = 0.50              # G_ROOT_EXPR
STOREOFF_BAR = 0.27               # G_STOREOFF
CE_CLEAN_SLACK = 0.10             # G_CECLEAN: CE_R <= base + slack
SHAL_CLOCK_FACTOR = 2.0           # S-SHAL dies within ~2x of GEN's clock

REGISTERED_PREDICTION = {
    "wash": "G3-DIES, THE LAW HOLDS: g0 <= 0.27 by +50; clock point prediction "
        f"first-under-bar in ({CLOCK_LO}, {CLOCK_HI}]. MECHANISM: the kill site "
        "is the GAIN (leak gradient collapses W_o/V at the standard clock); at "
        "the kill the retrieval is STILL INTACT (storage without expression). "
        "TWO-BASIN: state basin WIDE, parameter basin NARROW (dies at "
        "wash-direction store-displacement comparable to the discriminative "
        "kill displacement). THE WASH DOES NOT TRAVERSE THE ENERGY LANDSCAPE — "
        "IT RE-SCULPTS IT.",
    "noise": "NOISE-KILLS: both noise arms kill g3-GEN at displacement-match "
        f"(a fortiori below), COLLATERAL (CE >= {CE_COLLATERAL}) — no generative "
        "immunity. NOISE-SPARES at match would break the no-basin law's "
        "content-free clause.",
    "arm_d": "D-SURVIVES (g0 >= 0.50 through +300) IF the GAIN channel is the "
        "kill (leak removed, fact pattern gradient-blind, wd ~3% sub-lethal); "
        "D-DIES relocates the law's entry point to a shared door (W_q drift or "
        "null-key encroachment — the census adjudicates). Survival by "
        "BLINDNESS, not by basin.",
    "s_shal": "dies in the same bracket as g3-GEN (within ~2x of its clock); "
        "if g3-GEN >> g3-SHAL survival, generativity bought real time (the "
        "PLATEAU DISCOUNT becomes the finding).",
    "rider": "fires: one replay event restores g0 >= 0.5 within 50 wash steps "
        "(the re-entry economy is substrate-independent).",
    "falsifiers": "the LAW BREAKS if g3-GEN g0 >= 0.50 at EVERY checkpoint "
        "{1,2,4,10,50,100,200,300} at lr 1e-3 (S-SHAL then discriminates "
        "GENERATIVITY from SHALLOWNESS); the MECHANISM dies if the census shows "
        "retrieval collapse/query drift/gate flip/route death, or retrieval is "
        "NOT intact at the kill; the NOISE prediction dies if either arm "
        "spares at displacement-match; the COUPLING claim dies if g3-HARD also "
        "dies; the state-basin claim dies if the query sweeps show a narrow "
        "state basin.",
    "registration": "scratch/g3_design.md (committed, registered 2026-09-29), "
        "frozen verbatim in this docstring before compute. Adjudicate against "
        "exactly this; no bar shopping.",
}

trims: list[str] = []
deviations: list[str] = [
    "File name lab/g3_generative_store.py and outputs runs/g3/{metrics.json, "
    "generative_store.png} follow the DISPATCH deliverables; the spec's "
    "implementation sketch named lab/g3_generative_memory.py and "
    "g3_basin_map.png — naming only, no protocol delta.",
    "The construction trainer is e043's exposure arithmetic with the g3 spec's "
    "OWN batch composition (16 install windows + 32 RANDOM corpus anchors per "
    "step = 112 name-char vs 8,160 anchor targets; e043 itself used 16 + 48 "
    "with paired originals) and CONSTANT lr 1e-3 (the spec's 'AdamW lr 1e-3, "
    "200-400 steps'; e043 used the house cosine) — 300 steps chosen inside the "
    "registered band.",
    "S-DISC's displacement column is supplied by MEASURING e157's stored "
    "checkpoints on disk (eval-only; e157 recorded no parameter-delta record) "
    "— same device as this run's readouts, same currency as g3's (unweighted "
    "fp32 L2 over the full state dict).",
    "The wash trainer is e157's finetune_freeze VERBATIM arithmetic with e185's "
    "noise_wash additions (target substitution + displacement bookkeeping + "
    "per-step input md5) and the spec's +10 checkpoint; per-step displacement "
    "measured live (all-param) + per-checkpoint from snapshots (subspaces).",
    "The ONLY mechanical deviation from the trunk line: the model class "
    "GenMemGPT (a TinyGPT subclass carrying the organ between the block loop "
    "and ln_f, with a store_off forward flag for G_STOREOFF) — the spec's "
    "recorded deviation.",
    "Rider seeds: wash continuation 10903 / replay draws 10904 (e179's "
    "per-cell seed discipline; the spec named no rider seeds). Isotropic/"
    "sigma/gamma sweep seeds 110xx — recorded here.",
    "Single seed per construction/wash/noise cell (43050 / 10902 / 18501-2), "
    "one lineage (family 2) — replicates owed before any noun moves (the "
    "spec's honesty note).",
    "Smoke mode trims: 80-step constructions (enough to pass the root gates "
    "without the calibration fallback; probe-verified), 8-step washes with "
    "checkpoints {1,2,4}, 2-step noise arms, lean dials, no cooldowns; "
    "nothing adjudicated.",
]

# ------------------------------------------------------------------ device pick
# PROVENANCE: lab/e157_lineage_replication.py's pick_dev/migrate (e152R
# park-once adapted to the lab's gpu_ok()) VERBATIM.

GPU_PARKED = False
PARK_REASON = None
device_events: list[dict] = []


def pick_dev(tag: str) -> torch.device:
    """Strict pre-training quick check: gpu_ok() double-poll 5 s apart;
    PARK-ONCE — any failure parks every remaining training to CPU."""
    global GPU_PARKED, PARK_REASON
    if GPU_PARKED:
        log(f"[gpu] '{tag}' CPU (PARKED: {PARK_REASON})")
        return CPU
    if not torch.cuda.is_available():
        GPU_PARKED, PARK_REASON = True, "no CUDA"
        return CPU
    s1 = gpu_status()
    if gpu_ok():
        time.sleep(5)
        if gpu_ok():
            s2 = gpu_status()
            log(f"[gpu] '{tag}' may use GPU (util {s2['util']:.0f}% temp "
                f"{s2['temp']:.0f}C mem {s2['mem_used']:.0f}/"
                f"{s2['mem_total']:.0f}MB)")
            return torch.device("cuda")
    GPU_PARKED, PARK_REASON = True, f"quick check failed: {s1}"
    log(f"[gpu] PARK — '{tag}' and all remaining trainings run CPU ({s1})")
    return CPU


def migrate_to_cpu(net, opt) -> None:
    """Move net + optimizer state to CPU in place (params persist)."""
    net.to("cpu")
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state.get(p, {})
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to("cpu")


# ------------------------------------------------------------------ the organs

class StoreBlock(nn.Module):
    """The Hopfield memory organ (g3-GEN; hard=True -> g3-HARD, arm D).

    q_t = W_q . LN_noaffine(h_t)                (128 -> 64, position-wise,
                                                 causal by construction)
    a_t = softmax(beta * cos(q_t, K))            (beta = 8 FIXED, a buffer-free
                                                 design constant; hard=True:
                                                 top-1 argmax forward with
                                                 straight-through gradients)
    r_t = a_t . V ;  h_t <- h_t + W_o(r_t)       (64 -> 128 residual injection)

    Patterns 0..6 = the name steps (K_j = step-j query class mean at the
    one-shot write; V_j trained); pattern 7 = the null gate (K = anchor-query
    mean, V = 0 at init). `record` caches q/a/inj for the census + state-basin
    sweeps (eval-only); `q_noise` adds gaussian query noise in ||q|| units
    (the state-basin readout hook, eval-only, dedicated generator)."""

    def __init__(self, n_embd: int = D_EMBD, d_key: int = D_KEY, n_pat: int = N_PAT,
                 beta: float = BETA, hard: bool = False):
        super().__init__()
        self.W_q = nn.Linear(n_embd, d_key, bias=False)
        self.K = nn.Parameter(torch.zeros(n_pat, d_key))
        self.V = nn.Parameter(torch.zeros(n_pat, d_key))
        self.W_o = nn.Linear(d_key, n_embd, bias=False)
        self.beta, self.hard = beta, hard
        self.record, self.cache = False, {}
        self.q_noise, self.q_gen = 0.0, None
        for m in (self.W_q, self.W_o):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.K, mean=0.0, std=0.02)
        nn.init.normal_(self.V, mean=0.0, std=0.02)
        with torch.no_grad():
            self.V[NULL_IDX].zero_()

    def soft_a(self, q: torch.Tensor) -> torch.Tensor:
        qn = F.normalize(q, dim=-1)
        kn = F.normalize(self.K, dim=-1)
        return F.softmax(self.beta * (qn @ kn.t()), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.layer_norm(x, (x.shape[-1],))          # LN_noaffine
        q = self.W_q(h)
        if self.q_noise > 0.0 and self.q_gen is not None:
            n = torch.randn(q.shape, generator=self.q_gen).to(q.device)
            q = q + self.q_noise * q.norm(dim=-1, keepdim=True) * n
        a_soft = self.soft_a(q)
        if not self.hard:
            a = a_soft
        else:                                        # arm D: top-1 + STE
            idx = (self.beta * (F.normalize(q, dim=-1)
                                @ F.normalize(self.K, dim=-1).t())).argmax(
                                    -1, keepdim=True)
            a_hard = torch.zeros_like(a_soft).scatter_(-1, idx, 1.0)
            a = a_hard + a_soft - a_soft.detach()
        inj = self.W_o(a @ self.V)
        if self.record:
            with torch.no_grad():
                self.cache = {"q": q.detach(), "a": a_soft.detach(),
                              "inj_norm": inj.detach().norm(dim=-1)}
        return x + inj


class ShalBlock(nn.Module):
    """The shallow-discriminative twin (g3-SHAL): IDENTICAL graft point, query
    source (LN -> W_q', 128->64), injection form (W_o', 64->128) and interface;
    ONLY the middle differs — a 64->48 GELU->64 MLP computes the map directly,
    content IN WEIGHTS, no explicit stored pattern, no retrieval plateau.
    ~22.6k params (+30% over the store; reported, not padded away)."""

    def __init__(self, n_embd: int = D_EMBD, d_key: int = D_KEY,
                 d_hidden: int = SHAL_HIDDEN):
        super().__init__()
        self.W_q = nn.Linear(n_embd, d_key, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d_key, d_hidden), nn.GELU(),
                                 nn.Linear(d_hidden, d_key))
        self.W_o = nn.Linear(d_key, n_embd, bias=False)
        for lin in (self.W_q, self.mlp[0], self.mlp[2], self.W_o):
            nn.init.normal_(lin.weight, mean=0.0, std=0.02)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.layer_norm(x, (x.shape[-1],))
        return x + self.W_o(self.mlp(self.W_q(h)))


class GenMemGPT(TinyGPT):
    """TinyGPT + one memory organ between the block loop and ln_f (the
    output-side graft — the read from injection to logits is exactly
    ln_f + lm_head). store_disabled zeroes the organ's output at EVAL (the
    G_STOREOFF gate; no weight edits)."""

    def __init__(self, cfg: Cfg, organ: nn.Module):
        super().__init__(cfg)
        self.store = organ
        self.store_disabled = False

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        if not self.store_disabled:
            x = self.store(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss


KINDS = {
    "gen":  {"organ": lambda: StoreBlock(hard=False),
             "store_keys": ["store.W_q.weight", "store.K", "store.V",
                            "store.W_o.weight"]},
    "hard": {"organ": lambda: StoreBlock(hard=True),
             "store_keys": ["store.W_q.weight", "store.K", "store.V",
                            "store.W_o.weight"]},
    "shal": {"organ": ShalBlock,
             "store_keys": ["store.W_q.weight", "store.mlp.0.weight",
                            "store.mlp.0.bias", "store.mlp.2.weight",
                            "store.mlp.2.bias", "store.W_o.weight"]},
}


def store_keys(kind: str) -> list[str]:
    return KINDS[kind]["store_keys"]


def build_net(kind: str, seed: int = CONS_SEED):
    """Fresh GenMemGPT with deterministically-initialized organ (module-init
    draws under torch.manual_seed(seed); the trunk is overwritten by the
    caller's load)."""
    torch.manual_seed(seed)
    return GenMemGPT(F2_CFG, KINDS[kind]["organ"]())


def evl_load(kind: str, sd: dict) -> GenMemGPT:
    m = build_net(kind, seed=0)
    m.load_state_dict(sd)
    m.eval()
    return m


# ------------------------------------------------------------------ instruments
# PROVENANCE: lab/e157_lineage_replication.py / e176n VERBATIM. Copied rather
# than imported to own the device policy.

@torch.no_grad()
def battery_cell(net, ids: torch.Tensor, zid: int, bs=30) -> dict:
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
def battery_pz(net, ids: torch.Tensor, zid: int, bs=30) -> float:
    """e116's scalar battery (census readout)."""
    net.eval()
    pzs = []
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs += [float(pr[k, zid]) for k in range(pr.shape[0])]
    return float(np.mean(pzs))


@torch.no_grad()
def ce_fixed_cpu(net, x, y, bs=64) -> float:
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


def deleted_wpe(sd: dict, rows: tuple[int, ...]) -> tuple[dict, dict]:
    """D2 subtractive row-zero with the e065/e113 confinement gate."""
    out = {k: v.clone() for k, v in sd.items()}
    for r in rows:
        out["wpe.weight"][r] = 0.0
    d = out["wpe.weight"] != sd["wpe.weight"]
    changed_rows = sorted(set(int(r) for r in torch.nonzero(d)[:, 0].tolist()))
    n = int(d.sum().item())
    others = all(torch.equal(out[k], sd[k]) for k in out if k != "wpe.weight")
    gate = {"rows": list(rows), "n_elements_changed": n,
            "expected": len(rows) * sd["wpe.weight"].shape[1],
            "changed_rows": changed_rows,
            "confined": bool(changed_rows == sorted(rows)),
            "others_bit_identical": bool(others),
            "pass": bool(n == len(rows) * sd["wpe.weight"].shape[1]
                         and changed_rows == sorted(rows) and others)}
    return out, gate


@torch.no_grad()
def read_fact_at(net, pool_x: torch.Tensor, name_ids, zid: int,
                 addr_row: int, xcol: int, bs=30) -> dict:
    """e131's read_fact_position VERBATIM ARITHMETIC: p(true name char) at
    positions addr_row..addr_row+6 over the pool windows."""
    net.eval()
    n_name = len(name_ids)
    per_pos = [[] for _ in range(n_name)]
    onset = []
    for i in range(0, pool_x.shape[0], bs):
        w = pool_x[i:i + bs]
        lg, _ = net(w)
        pr = F.softmax(lg, -1)
        for k in range(w.shape[0]):
            onset.append(float(pr[k, addr_row, int(zid)]))
            for j in range(n_name):
                per_pos[j].append(
                    float(pr[k, addr_row + j, int(w[k, xcol + j])]))
    onset_t = torch.tensor(onset)
    allp_t = torch.tensor([p for pos in per_pos for p in pos])
    return {"pz_onset_mean": float(onset_t.mean()),
            "pz_onset_median": float(onset_t.median()),
            "pz_onset_frac_ge_0.5": float((onset_t >= 0.5).float().mean()),
            "pname_mean_over7": float(allp_t.mean()),
            "pname_frac_ge_0.5": float((allp_t >= 0.5).float().mean()),
            "per_position_mean": [float(np.mean(pos)) for pos in per_pos]}


def row_census(net, rows, readout, *rargs) -> dict:
    """e139's row_census VERBATIM (mean-arm / zero-arm / restore)."""
    net.eval()
    base = readout(net, *rargs)
    w = net.wpe.weight.data
    orig = w.clone()
    mean_row = orig.mean(0)
    m_d, z_d = {}, {}
    for r in rows:
        w.copy_(orig); w[r] = mean_row
        m_d[r] = base - readout(net, *rargs)
        w.copy_(orig); w[r] = 0.0
        z_d[r] = base - readout(net, *rargs)
    w.copy_(orig)
    rows_d = {str(r): {"mean": float(m_d[r]), "zero": float(z_d[r]),
                       "ratio": float(min(m_d[r], z_d[r]) /
                                      max(m_d[r], z_d[r]))
                              if max(m_d[r], z_d[r]) > 0 else 0.0,
                       "strength": float(min(m_d[r], z_d[r])),
                       "content": bool(m_d[r] > 0 and z_d[r] > 0 and
                                       min(m_d[r], z_d[r]) /
                                       max(m_d[r], z_d[r]) >= 0.5)}
              for r in rows}
    assert torch.equal(w, orig), "census failed to restore wpe"
    return {"base_readout": base, "rows": rows_d}


def restore_class(sd_base: dict, sd_src: dict, keys: list, name: str
                  ) -> tuple[dict, dict]:
    """e173's transplant gate generalized to key sets: replace sd_base's
    `keys` with sd_src's; confinement = only class keys may change."""
    out = {k: v.clone() for k, v in sd_base.items()}
    for k in keys:
        out[k] = sd_src[k].clone()
    ks = set(keys)
    per_key = {k: int((out[k] != sd_base[k]).sum().item()) for k in keys}
    n_changed = sum(per_key.values())
    n_elems = sum(sd_base[k].numel() for k in keys)
    others = all(torch.equal(out[k], sd_base[k]) for k in out if k not in ks)
    gate = {"class": name, "n_tensors": len(keys),
            "tensors_changed": int(sum(1 for v in per_key.values() if v > 0)),
            "n_elements_changed": n_changed,
            "expected_elements": int(n_elems),
            "identity": bool(n_changed == 0),
            "others_bit_identical": bool(others),
            "pass": bool(others and (n_changed == 0
                                     or n_changed >= 0.95 * n_elems))}
    return out, gate


def tensor_short(k: str) -> str:
    """Short unique name for a store tensor key (W_q / K / V / W_o / mlp0...)."""
    s = k.replace("store.", "")
    s = s.replace(".weight", "").replace(".bias", "_b")
    return s.replace(".", "")


def sd_disp(sd_a: dict, sd_b: dict, keys=None) -> float:
    """Unweighted fp32 L2 over the (sub)space — e185's displacement currency."""
    ks = list(sd_a.keys()) if keys is None else keys
    return float(sum(float((sd_a[k].float() - sd_b[k].float()).norm() ** 2)
                     for k in ks) ** 0.5)


# ------------------------------------------------------------------ trainings

def one_shot_write(net: GenMemGPT, pool_x: torch.Tensor,
                   anchor_x: torch.Tensor) -> dict:
    """The one-shot write (the Hopfield storage operation, closed form):
    K_j = the class mean of step-j install-window queries (rows PRE-1+j of
    the 255-token training view = the position that predicts name char j);
    K_null = the anchor-query mean; V_null = 0; V_j stays at its random init
    (gradient-refined next). No RNG consumed."""
    net.eval()
    net.store.record = True
    _ = net(pool_x[:, :-1])
    q_fact = net.store.cache["q"].clone()            # (60, 255, 64)
    _ = net(anchor_x[:, :-1])
    q_anc = net.store.cache["q"].clone()             # (32, 255, 64)
    net.store.record = False
    K = net.store.K.data
    with torch.no_grad():
        for j in range(7):
            K[j] = q_fact[:, PRE - 1 + j, :].mean(0)
        K[NULL_IDX] = q_anc.mean(dim=(0, 1))
        net.store.V.data[NULL_IDX].zero_()
    return {"n_fact_queries": int(np.prod(q_fact.shape[:2])),
            "null_query_source": "32 anchor windows x 255 positions"}


def construct(tag: str, kind: str, base_sd: dict, pool_x: torch.Tensor,
              pool_mask: torch.Tensor, train_ids: torch.Tensor, itos,
              r_eval_xy, g0_ids, zid: int):
    """STAGE 0 construction: store-only training on the e043 recipe's
    arithmetic with the spec's batch (16 install + 32 random anchors,
    token-weighted union CE); host FROZEN; AdamW lr 1e-3 constant, wd 0.1,
    clip 1.0; seed 43050 (module init + a dedicated batch generator whose
    FIRST 32 draws are the one-shot write's anchors, then the training
    stream). If G_ROOT_EXPR misses, ONE joint calibration (host+store,
    <= 100 steps, lr 3e-4, seed 43060) is allowed and recorded."""
    dev = pick_dev(tag)
    cap = GPU_CAP_S if dev.type == "cuda" else CPU_CAP_S
    net = build_net(kind)
    miss = net.load_state_dict(base_sd, strict=False)
    gate_trunk = {"missing": sorted(miss.missing_keys),
                  "unexpected": sorted(miss.unexpected_keys)}
    gate_trunk["pass"] = bool(gate_trunk["missing"] == sorted(store_keys(kind))
                              and gate_trunk["unexpected"] == [])
    assert gate_trunk["pass"], f"trunk graft gate FAILED: {gate_trunk}"
    g = torch.Generator().manual_seed(CONS_SEED)
    rj0 = torch.randint(len(train_ids) - BLOCK - 1, (CONS_ANC_BS,),
                        generator=g)
    anc0 = torch.stack([train_ids[s: s + BLOCK] for s in rj0])
    if kind == "shal":
        write = {"note": "no one-shot write for the shallow twin (no "
                         "pattern matrix; random init, gradient-trained)"}
    else:
        write = one_shot_write(net, pool_x, anc0)    # CPU (before .to(dev))
    net = net.to(dev)
    for p in net.parameters():                       # host FROZEN
        p.requires_grad_(False)
    for p in net.store.parameters():
        p.requires_grad_(True)
    n_store = sum(p.numel() for p in net.store.parameters())
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=CONS_LR, betas=(0.9, 0.95), weight_decay=0.1)
    net.train()
    n_pool = pool_x.shape[0]
    traj, t_start, step = [], time.time(), 0
    evl = copy.deepcopy(net).to(CPU)
    for step in range(1, CONS_STEPS + 1):
        ix = torch.randint(n_pool, (CONS_NAME_BS,), generator=g)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (CONS_ANC_BS,),
                           generator=g)
        nw = pool_x[ix].to(dev)
        anc = torch.stack([train_ids[s: s + BLOCK] for s in rj]).to(dev)
        x = torch.cat([nw[:, :-1], anc[:, :-1]], 0)
        y = torch.cat([nw[:, 1:], anc[:, 1:]], 0)
        m = torch.zeros(CONS_NAME_BS + CONS_ANC_BS, x.shape[1],
                        dtype=torch.bool, device=dev)
        m[:CONS_NAME_BS] = pool_mask[ix].to(dev)
        logits, _ = net(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                              y.reshape(-1), reduction="none"
                              ).view(x.shape[0], x.shape[1])
        nm = nll[:CONS_NAME_BS][m[:CONS_NAME_BS]]
        cm = nll[CONS_NAME_BS:]
        loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0 or step == CONS_STEPS or \
                (time.time() - t_start) > cap:
            sd_cpu = {k: v.detach().cpu().clone()
                      for k, v in net.state_dict().items()}
            evl.load_state_dict(sd_cpu)
            evl.eval()
            bz = battery_cell(evl, g0_ids, zid)
            ce_r = ce_fixed_cpu(evl, *r_eval_xy)
            traj.append({"step": step, "g0_mean_pz": bz["mean_pz"],
                         "ce_r": ce_r, "union_ce": float(loss.item()),
                         "elapsed_s": round(time.time() - t_start, 1)})
            log(f"  [{tag}] s{step:4d} g0 {bz['mean_pz']:.4f} CE_R {ce_r:.4f} "
                f"(union CE {float(loss.item()):.4f})")
        if (time.time() - t_start) > cap:
            log(f"  [{tag}] time cap {cap:.0f}s at s{step}")
            trims.append(f"{tag}: construction time cap at step {step}")
            break
        if dev.type == "cuda" and step % MIDRUN_POLL_EVERY == 0:
            s = gpu_status()
            if s["mem_total"] > 0 and (s["mem_used"] > 0.85 * s["mem_total"]
                                       or s["temp"] > 80):
                device_events.append({"tag": tag, "step": step,
                                      "event": "MID-RUN MIGRATION", "status": s})
                migrate_to_cpu(net, opt)
                dev, cap = CPU, CPU_CAP_S
    sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    host_frozen = all(torch.equal(sd_cpu[k], base_sd[k])
                      for k in base_sd)
    net.eval()
    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"sd": sd_cpu, "traj": traj, "steps_ran": step, "seed": CONS_SEED,
            "n_store_params": n_store, "write": write, "device": str(dev),
            "host_bit_identical_to_base": bool(host_frozen),
            "trunk_gate": gate_trunk}


def calibrate(tag: str, kind: str, sd0: dict, pool_x, pool_mask, train_ids,
              r_eval_xy, g0_ids, zid: int, ce_budget: float,
              steps: int = CAL_STEPS_MAX):
    """The ONE allowed joint calibration (host+store, lr 3e-4, CE_R gate),
    recorded. Same union-CE data as the construction."""
    dev = pick_dev(tag + "-cal")
    cap = GPU_CAP_S if dev.type == "cuda" else CPU_CAP_S
    net = evl_load(kind, sd0).to(dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, betas=(0.9, 0.95),
                            weight_decay=0.1)
    g = torch.Generator().manual_seed(CAL_SEED)
    traj, t_start, step = [], time.time(), 0
    evl = copy.deepcopy(net).to(CPU)
    for step in range(1, steps + 1):
        ix = torch.randint(pool_x.shape[0], (CONS_NAME_BS,), generator=g)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (CONS_ANC_BS,),
                           generator=g)
        nw = pool_x[ix].to(dev)
        anc = torch.stack([train_ids[s: s + BLOCK] for s in rj]).to(dev)
        x = torch.cat([nw[:, :-1], anc[:, :-1]], 0)
        y = torch.cat([nw[:, 1:], anc[:, 1:]], 0)
        m = torch.zeros(CONS_NAME_BS + CONS_ANC_BS, x.shape[1],
                        dtype=torch.bool, device=dev)
        m[:CONS_NAME_BS] = pool_mask[ix].to(dev)
        logits, _ = net(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                              y.reshape(-1), reduction="none"
                              ).view(x.shape[0], x.shape[1])
        nm = nll[:CONS_NAME_BS][m[:CONS_NAME_BS]]
        cm = nll[CONS_NAME_BS:]
        loss = (nm.sum() + cm.sum()) / (nm.numel() + cm.numel())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        evl.load_state_dict(sd_cpu)
        evl.eval()
        bz = battery_cell(evl, g0_ids, zid)
        ce_r = ce_fixed_cpu(evl, *r_eval_xy)
        traj.append({"step": step, "g0_mean_pz": bz["mean_pz"], "ce_r": ce_r})
        log(f"  [{tag}] cal s{step:3d} g0 {bz['mean_pz']:.4f} CE_R {ce_r:.4f}")
        if bz["mean_pz"] >= ROOT_EXPR_BAR and ce_r <= ce_budget:
            break                                     # expression reached, rent paid
        if (time.time() - t_start) > cap or step == steps:
            break
    sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.eval()
    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"sd": sd_cpu, "traj": traj, "steps_ran": step, "seed": CAL_SEED}


def wash_run(tag: str, kind: str, sd0: dict, anchor: torch.Tensor,
             train_ids: torch.Tensor, itos, r_eval_xy, gm12_ids, g0_ids,
             zid: int, seed: int, ckpt_steps: tuple[int, ...],
             target_mode: str = "true", noise_seed: int = 0, lr: float = WASH_LR):
    """THE NEUTRAL PLAIN-CORPUS WASH (e157's finetune_freeze VERBATIM
    arithmetic) with e185's noise_wash additions: per-step target substitution
    (true / iid / perm) from a DEDICATED generator drawn AFTER the aj/rj
    input draws, displacement bookkeeping (all-param cumulative + per-step;
    store/host subspace + per store tensor at checkpoints via snapshots), and
    per-step input md5 hashes. Batch 32 full-token CE; AdamW (0.9,0.95) wd
    0.1 constant lr, clip 1.0. Lean CPU evals (g-12, g0, CE_R — no RNG)."""
    assert target_mode in ("true", "iid", "perm")
    dev = pick_dev(tag)
    cap = GPU_CAP_S if dev.type == "cuda" else CPU_CAP_S
    ckpt_set = set(ckpt_steps)
    n_steps = ckpt_steps[-1]
    net = evl_load(kind, sd0).to(dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    ngen = (torch.Generator().manual_seed(noise_seed)
            if target_mode != "true" else None)
    vocab = len(itos)
    n_anc = anchor.shape[0]
    skeys = store_keys(kind)
    traj, sds, t_start = [], {}, time.time()
    step = 0
    zeph_checks = 0
    evl = copy.deepcopy(net).to(CPU)
    theta0 = torch.cat([p.detach().reshape(-1).cpu()
                        for p in net.parameters()]).clone()
    x_hashes: dict[int, str] = {}
    for step in range(1, n_steps + 1):
        aj = torch.randint(n_anc, (16,), generator=gen)
        rj = torch.randint(len(train_ids) - BLOCK - 1, (16,), generator=gen)
        anc = anchor[aj]
        rnd = torch.stack([train_ids[s: s + BLOCK] for s in rj])
        for w in rnd:
            txt = "".join(itos[int(c)] for c in w[:64]) + \
                  "".join(itos[int(c)] for c in w[192:])
            if "ZEPH" in txt:
                zeph_checks += 1
        x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0).to(dev)
        y_true = torch.cat([anc[:, 1:], rnd[:, 1:]], 0)
        if target_mode == "true":
            y = y_true.to(dev)
        elif target_mode == "iid":
            y = torch.randint(vocab, y_true.shape, generator=ngen).to(dev)
        else:
            perm = torch.randperm(y_true.numel(), generator=ngen)
            y = y_true.reshape(-1)[perm].reshape(y_true.shape).to(dev)
        x_hashes[step] = hashlib.md5(
            x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        logits, _ = net(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        cur = torch.cat([p.detach().reshape(-1).cpu()
                         for p in net.parameters()])
        cum_disp = float(torch.norm(cur - theta0))
        traj.append({"step": step, "ce_batch": float(loss.item()),
                     "cum_disp": cum_disp})
        if step in ckpt_set:
            sd_cpu = {k: v.detach().cpu().clone()
                      for k, v in net.state_dict().items()}
            sds[step] = sd_cpu
            evl.load_state_dict(sd_cpu)
            evl.eval()
            gz = battery_cell(evl, gm12_ids, zid)
            gz0 = battery_cell(evl, g0_ids, zid)
            ce_r = ce_fixed_cpu(evl, *r_eval_xy)
            hkeys = [k for k in sd_cpu if k not in skeys]
            traj[-1].update({
                "g_m12_mean_pz": gz["mean_pz"], "g0_mean_pz": gz0["mean_pz"],
                "frac_argmax_z": gz["frac_argmax_z"], "ce_r": ce_r,
                "disp_store": sd_disp(sd_cpu, sd0, skeys),
                "disp_host": sd_disp(sd_cpu, sd0, hkeys),
                "disp_all_snap": sd_disp(sd_cpu, sd0),
                **{f"disp_{tensor_short(k)}": float(
                    (sd_cpu[k].float() - sd0[k].float()).norm())
                   for k in skeys}})
            log(f"  [{tag}] CKPT +{step:4d} g-12 {gz['mean_pz']:.4f} "
                f"g0 {gz0['mean_pz']:.4f} CE_R {ce_r:.4f} |d| {cum_disp:.4f} "
                f"(store {traj[-1]['disp_store']:.4f})")
        if (time.time() - t_start) > cap:
            log(f"  [{tag}] time cap {cap:.0f}s at s{step}")
            trims.append(f"{tag}: time cap at step {step}")
            break
        if dev.type == "cuda" and step % MIDRUN_POLL_EVERY == 0:
            s = gpu_status()
            if s["mem_total"] > 0 and (s["mem_used"] > 0.85 * s["mem_total"]
                                       or s["temp"] > 80):
                device_events.append({"tag": tag, "step": step,
                                      "event": "MID-RUN MIGRATION", "status": s})
                log(f"  [{tag}] MID-RUN GPU contention at s{step} ({s}) -> CPU")
                migrate_to_cpu(net, opt)
                dev, cap = CPU, CPU_CAP_S
    net.eval()
    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"sds": sds, "traj": traj, "steps_ran": step, "seed": seed,
            "lr": lr, "target_mode": target_mode,
            "noise_seed": noise_seed if ngen is not None else None,
            "zeph_violations": zeph_checks, "x_hashes": x_hashes,
            "device": str(dev)}


def replay_step(tag: str, kind: str, sd0: dict, pool_x, pool_mask,
                anchor_bank, train_ids, zid: int, seed: int, lr: float):
    """e179's replay batch (e174 arm B's arithmetic): 16 windows drawn from the
    fact's pool + 16 anchors (8 paired originals + 8 random), name-masked
    union CE — ONE optimizer step at the wash's own lr."""
    dev = pick_dev(tag)
    net = evl_load(kind, sd0).to(dev)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    g = torch.Generator().manual_seed(seed)
    ix = torch.randint(pool_x.shape[0], (16,), generator=g)
    aj = torch.randint(anchor_bank.shape[0], (8,), generator=g)
    rj = torch.randint(len(train_ids) - BLOCK - 1, (8,), generator=g)
    nw = pool_x[ix].to(dev)
    anc = torch.cat([anchor_bank[aj],
                     torch.stack([train_ids[s: s + BLOCK] for s in rj])],
                    0).to(dev)
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
    sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.eval()
    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"sd": sd_cpu, "replay_ce": float(loss.item()), "seed": seed}


CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        name = f"smoke_{name}"
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "g3", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name}")


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("g3_smoke" if SMOKE else "g3")
    common.DEVICE = "cpu"     # ALL readouts CPU-side (trainers manage devices)
    log(f"G3 THE GENERATIVE-MEMORY ARCHITECTURE (smoke={SMOKE}) -> {rd}")
    log(f"compute: strict per-training GPU gate (park-once) + mid-run guard "
        f"every {MIDRUN_POLL_EVERY} steps; gpu at start: {gpu_status()}; "
        f"threads {torch.get_num_threads()}")

    # ---------------- protocol rebuild (e143/e151/e157/e176n verbatim)
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
    rng = random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix, "pass": bool(mix == {"FLORIZEL": 19,
                                                         "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"
    name_ids = corpus.encode(NAME)
    L = len(NAME)

    def offset_pool(j: int):
        """e143's offset-pool construction (e157 verbatim)."""
        wins = []
        for p, h in install_occ:
            pre = train_ids[p - PRE - j: p]
            post = train_ids[p + len(h): p + len(h) + POST_CAP - j]
            if len(pre) != PRE + j or len(post) != POST_CAP - j:
                raise RuntimeError(f"window short at p={p} j={j}")
            w = torch.cat([pre, name_ids, post])
            wins.append(w)
        px = torch.stack(wins)
        pm = torch.zeros(len(wins), BLOCK - 1, dtype=torch.bool)
        pm[:, PRE - 1 + j: PRE - 1 + j + L] = True
        return px, pm

    pool_band_x, pool_band_mask = offset_pool(0)      # home band (training)
    pool_183_x, _ = offset_pool(RETEACH_J)            # the site-read instrument
    G_POOL = {
        "band_shape": list(pool_band_x.shape),
        "band_name_in_place": bool(all(
            torch.equal(w[PRE: PRE + L], name_ids) for w in pool_band_x)),
        "p183_name_in_place": bool(all(
            torch.equal(w[SITE_Z_XCOL: SITE_Z_XCOL + L], name_ids)
            for w in pool_183_x)),
        "mask_targets": int(pool_band_mask[0].sum()),
        "note": "pool_band trains the store (j=0; e043's own convention); "
                "pool_183 is the MEASUREMENT instrument only (e152's j=54)",
    }
    G_POOL["pass"] = bool(G_POOL["band_name_in_place"]
                          and G_POOL["p183_name_in_place"]
                          and G_POOL["mask_targets"] == L)
    assert G_POOL["pass"], f"pool gate FAILED: {G_POOL}"
    log(f"protocol rebuilt: install60 {mix}, held30; pools "
        f"band {tuple(pool_band_x.shape)} / p183 {tuple(pool_183_x.shape)}")

    anchor_bank = torch.stack([train_ids[p - PRE: p - PRE + BLOCK]
                               for p, _ in install_occ[:16]])  # e065 originals

    bat_ids = {}
    for j in GEOS:
        for tag_, occ in (("install60", install_occ), ("held30", held_occ)):
            cs = [train_text[p - PRE - j: p] for p, _ in occ]
            bat_ids[(j, tag_)] = torch.stack([corpus.encode(c) for c in cs])
    ids130 = bat_ids[(0, "install60")]
    r_eval_x, r_eval_y = val_windows(val_ids, val_text, 60, R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

    # ---------------- THE NEUTRAL STREAM (e170 via e157 VERBATIM)
    arng = random.Random(E170_ANCHOR_SEED)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        tries += 1
        txt = train_text[s: s + BLOCK + 1]
        if any(f in txt for f in ANCHOR_FORBIDDEN):
            rejections += 1
            continue
        n_starts.append(s)
    if len(n_starts) != 16:
        raise RuntimeError(f"neutral bank incomplete: {len(n_starts)}/16")
    anchor_neutral = torch.stack([train_ids[s: s + BLOCK] for s in n_starts])
    host_positions = [p for p in E43.find_occ(train_text, HOSTS[0])]
    host_positions += [p for p in E43.find_occ(train_text, HOSTS[1])]

    def junctions_covered(starts):
        return sum(1 for s in starts
                   if any(s <= p < s + BLOCK + 1 for p in host_positions))

    jc_neutral = junctions_covered(n_starts)
    G_ANCHOR = {
        "construction": ("16 plain corpus windows from train_ids, RNG seed "
                         f"{E170_ANCHOR_SEED}, rejection on FLORIZEL/ELIZABETH/"
                         "ZEPH/MIRABEL in [s, s+257) — e170 VERBATIM"),
        "n_windows": 16, "seed": E170_ANCHOR_SEED, "starts": n_starts,
        "tries": tries, "rejections": rejections,
        "windows_with_host_content": sum(
            1 for s in n_starts
            if any(f in train_text[s: s + BLOCK + 1] for f in HOSTS)),
        "junctions_covered": jc_neutral,
    }
    G_ANCHOR["pass"] = bool(
        G_ANCHOR["windows_with_host_content"] == 0
        and G_ANCHOR["junctions_covered"] == 0
        and anchor_neutral.shape == (16, BLOCK))
    assert G_ANCHOR["pass"], f"anchor gate FAILED: {G_ANCHOR}"
    log(f"G_ANCHOR: neutral bank 16x{BLOCK} (seed {E170_ANCHOR_SEED}, "
        f"{rejections} rejections/{tries} tries): PASS")

    # ---------------- base trunk + CE budget
    st = torch.load(CKPT_DIR / BASE_CK, map_location="cpu", weights_only=False)
    base_sd = {k: v.clone() for k, v in
               (st["model"] if isinstance(st, dict) and "model" in st
                else st).items()}
    base_net = TinyGPT(F2_CFG)          # plain trunk (the store is OFF by
    base_net.load_state_dict(base_sd)   # construction on this net)
    base_net.eval()
    ce_base = ce_fixed_cpu(base_net, *r_eval_xy)
    del base_net
    ce_budget = ce_base + CE_CLEAN_SLACK
    log(f"base trunk {BASE_CK}: CE_R {ce_base:.4f} (budget base+"
        f"{CE_CLEAN_SLACK} = {ce_budget:.4f})")

    # ---------------- S-DISC reference (e157 cells + displacement from disk)
    sdisc = {"trace": SDISC_TRACE, "displacement": {}, "source": None}
    root_sdisc = torch.load(CKPT_DIR / SDISC_ROOT_CK, map_location="cpu",
                            weights_only=False)["model"]
    for s, ck in SDISC_WASH_CK.items():
        sd_w = torch.load(CKPT_DIR / ck, map_location="cpu",
                          weights_only=False)["model"]
        sdisc["displacement"][str(s)] = sd_disp(sd_w, root_sdisc)
    sdisc["displacement"]["0"] = 0.0
    sdisc["t_star_g0"] = next((s for s in SDISC_TRACE["freeze_steps"] if s > 0
                               and SDISC_TRACE["g0"][
                                   SDISC_TRACE["freeze_steps"].index(s)]
                               <= SHUT_BAR), None)
    if sdisc["t_star_g0"] is not None:
        sdisc["D_disc"] = sdisc["displacement"][str(sdisc["t_star_g0"])]
    e157_path = E43.REPO / "runs" / "e157" / "metrics.json"
    if e157_path.exists():
        m157 = json.loads(e157_path.read_text(encoding="utf-8"))
        tr157 = m157["stages"]["B_wash"]["trace"]
        ok = ([r["freeze_steps"] for r in tr157]
              == SDISC_TRACE["freeze_steps"]
              and max(abs(a["g0"] - b) for a, b in zip(tr157,
                                                      SDISC_TRACE["g0"])) < 1e-9)
        sdisc["source"] = (f"runs/e157/metrics.json trace (verified {ok}); "
                           "displacement measured from e157_f2_* checkpoints "
                           "on disk (eval-only)")
        sdisc["verified_vs_file"] = bool(ok)
    else:
        sdisc["source"] = "embedded copy (e157 metrics file absent)"
        sdisc["verified_vs_file"] = False
    log(f"S-DISC control (free): t*(g0) +{sdisc['t_star_g0']}, D_disc "
        f"{sdisc.get('D_disc', float('nan')):.4f}; disp "
        + " ".join(f"+{s}:{d:.3f}" for s, d in sorted(
            sdisc["displacement"].items(), key=lambda kv: int(kv[0]))))

    # =====================================================================
    # STAGE 0 — CONSTRUCTIONS
    # =====================================================================
    gates_surg: dict = {}

    def lean_dial(kind, sd, tag):
        net = evl_load(kind, sd)
        out = {"tag": tag}
        out["base"] = {j: battery_cell(net, bat_ids[(j, "install60")], zid)
                       for j in GEOS}
        out["base_held"] = {j: battery_cell(net, bat_ids[(j, "held30")], zid)
                            for j in GEOS}
        out["ce_r"] = ce_fixed_cpu(net, *r_eval_xy)
        net.store_disabled = True
        out["g0_store_off"] = battery_pz(net, ids130, zid)
        net.store_disabled = False
        out["site_read"] = read_fact_at(net, pool_183_x, name_ids, zid,
                                        SITE_ADDR_ROW, SITE_Z_XCOL)
        del net
        log(f"[{tag}] " + " ".join(f"g{j:+d} {out['base'][j]['mean_pz']:.4f}"
                                   for j in GEOS)
            + f" | held30 g0 {out['base_held'][0]['mean_pz']:.4f} | CE_R "
            f"{out['ce_r']:.4f} | store-off g0 {out['g0_store_off']:.4f} | "
            f"span@183 {out['site_read']['pname_mean_over7']:.4f}")
        return out

    def root_dial(kind, sd, tag):
        out = lean_dial(kind, sd, tag)
        net = evl_load(kind, sd)
        out["census_old"] = row_census(net, ROWS_OLD,
                                       lambda n: battery_pz(n, ids130, zid))
        co = out["census_old"]["rows"]
        out["old_band"] = {"row0_strength": co["0"]["strength"],
                           "A129": co["129"]["strength"],
                           "band121_129_max": max(co[str(r)]["strength"]
                                                  for r in range(121, 130)
                                                  if str(r) in co)}
        out["del_table"] = {}
        for dl, rows_ in {"d_all": D_ALL, "d183": (SITE_ADDR_ROW,)}.items():
            sd_d, gate = deleted_wpe(sd, rows_)
            gates_surg[f"{tag}__{dl}"] = gate
            net.load_state_dict(sd_d)
            out["del_table"][dl] = {"g0": battery_cell(net, ids130,
                                                       zid)["mean_pz"]}
        del net
        return out

    arms = {}
    ARM_PLAN = [
        ("gen", "g3-GEN", "the Hopfield store (soft gate) — THE cell"),
        ("shal", "g3-SHAL", "the shallow-discriminative twin (content in weights)"),
        ("hard", "g3-HARD", "arm D — the hardened gate (top-1 + STE)"),
    ]
    for kind, label, desc in ARM_PLAN:
        log("=" * 78)
        if not SMOKE:
            log(f"[thermal] cooldown {COOLDOWN_S:.0f}s before {label}")
            cooldown(COOLDOWN_S)
        log(f"STAGE 0 CONSTRUCT {label} — {desc}; {CONS_STEPS} steps, seed "
            f"{CONS_SEED}")
        con = construct(kind, kind, base_sd, pool_band_x, pool_band_mask,
                        train_ids, itos, r_eval_xy, ids130, zid)
        n_params = sum(p.numel() for p in
                       evl_load(kind, con["sd"]).parameters())
        sd_root = con["sd"]
        cal = None
        d0 = lean_dial(kind, sd_root, f"{kind}-root0")
        if d0["base"][0]["mean_pz"] < ROOT_EXPR_BAR:
            log(f"  [{kind}] G_ROOT_EXPR miss at construction "
                f"(g0 {d0['base'][0]['mean_pz']:.4f}) -> ONE joint calibration "
                f"(<= {CAL_STEPS_MAX} steps, lr 3e-4)")
            cal = calibrate(kind, kind, sd_root, pool_band_x, pool_band_mask,
                            train_ids, r_eval_xy, ids130, zid, ce_budget)
            sd_root = cal["sd"]
        dial = root_dial(kind, sd_root, f"{kind}-root")
        g0 = dial["base"][0]["mean_pz"]
        G_ROOT = {
            "G_ROOT_EXPR": {"g0": g0, "bar": ROOT_EXPR_BAR,
                            "pass": bool(g0 >= ROOT_EXPR_BAR)},
            "G_STOREOFF": {"g0_store_off": dial["g0_store_off"],
                           "bar": STOREOFF_BAR,
                           "pass": bool(dial["g0_store_off"] <= STOREOFF_BAR)},
            "G_CECLEAN": {"ce_r": dial["ce_r"], "budget": ce_budget,
                          "pass": bool(dial["ce_r"] <= ce_budget)},
            "host_bit_identical_to_base": con["host_bit_identical_to_base"],
            "joint_calibration_ran": cal is not None,
            "n_params": n_params, "n_store_params": con["n_store_params"],
        }
        G_ROOT["pass"] = bool(G_ROOT["G_ROOT_EXPR"]["pass"]
                              and G_ROOT["G_STOREOFF"]["pass"]
                              and G_ROOT["G_CECLEAN"]["pass"])
        log(f"  [{kind}] ROOT GATES: expr {g0:.4f} (>= {ROOT_EXPR_BAR}) | "
            f"store-off {dial['g0_store_off']:.4f} (<= {STOREOFF_BAR}) | "
            f"CE_R {dial['ce_r']:.4f} (<= {ce_budget:.4f}) -> "
            f"{'PASS' if G_ROOT['pass'] else 'CONSTRUCTION-CEILING'}")
        save_ckpt(f"g3_{kind}", sd_root,
                  {"desc": f"e098_base_s4305 + {label} organ "
                           f"(store-only training, seed {CONS_SEED})"
                           + (" + joint calibration" if cal else ""),
                   "steps": int(con["steps_ran"]),
                   "cal_steps": int(cal["steps_ran"]) if cal else 0,
                   "seed": CONS_SEED, "device": con["device"], "kind": kind,
                   "base": f"runs/checkpoints/{BASE_CK}"})
        arms[kind] = {"desc": desc, "con": con, "cal": cal, "root_sd": sd_root,
                      "dial": dial, "G_ROOT": G_ROOT, "params": n_params}

    # =====================================================================
    # STAGE A — THE WASH (3 x 300 steps, e176N arm A via e157 VERBATIM)
    # =====================================================================
    washes = {}
    for kind, label, desc in ARM_PLAN:
        log("=" * 78)
        if not arms[kind]["G_ROOT"]["pass"]:
            log(f"STAGE A SKIPPED for {label}: root gate CONSTRUCTION-CEILING "
                "(reported; no wash adjudication)")
            washes[kind] = None
            continue
        if not SMOKE:
            log(f"[thermal] cooldown {COOLDOWN_S:.0f}s before {label} wash")
            cooldown(COOLDOWN_S)
        log(f"STAGE A WASH {label}: {WASH_STEPS} steps (neutral bank seed "
            f"{E170_ANCHOR_SEED}, batch 16+16 full-token CE, lr 1e-3, seed "
            f"{ARM_SEED}), checkpoints +{list(CK_WASH)}")
        w = wash_run(f"wash-{kind}", kind, arms[kind]["root_sd"],
                     anchor_neutral, train_ids, itos, r_eval_xy,
                     bat_ids[(-12, "install60")], ids130, zid, ARM_SEED,
                     CK_WASH)
        assert w["zeph_violations"] == 0, f"{kind}: name token leaked"
        if not SMOKE:
            log(f"[thermal] cooldown {COOLDOWN_S:.0f}s after {label} wash")
            cooldown(COOLDOWN_S)
        for s in sorted(w["sds"]):
            save_ckpt(f"g3_{kind}_s{s}", w["sds"][s],
                      {"desc": f"g3_{kind} root + {s}-step NEUTRAL freeze "
                               f"(e176n arm A verbatim; seed {ARM_SEED})",
                       "steps": int(s), "seed": ARM_SEED,
                       "device": w["device"], "kind": kind,
                       "base": f"runs/checkpoints/g3_{kind}.pt"})
        # full dials at +10 and +300
        dials = {"0": arms[kind]["dial"]}
        for s in (10, 300):
            if s in w["sds"]:
                dials[str(s)] = lean_dial(kind, w["sds"][s], f"{kind}-n{s}")
        washes[kind] = {"wash": w, "dials": dials}

    def wash_cells(kind):
        """g0 trace + t* (first checkpoint with g0 <= SHUT_BAR) for an arm."""
        w = washes[kind]["wash"]
        rows = [{"step": 0, "g0": arms[kind]["dial"]["base"][0]["mean_pz"],
                 "gm12": arms[kind]["dial"]["base"][-12]["mean_pz"],
                 "ce_r": arms[kind]["dial"]["ce_r"], "cum_disp": 0.0}]
        for t in w["traj"]:
            if "g0_mean_pz" in t:
                rows.append({"step": t["step"], "g0": t["g0_mean_pz"],
                             "gm12": t["g_m12_mean_pz"], "ce_r": t["ce_r"],
                             "cum_disp": t["cum_disp"],
                             "disp_store": t.get("disp_store"),
                             "disp_host": t.get("disp_host"),
                             "frac_argmax_z": t.get("frac_argmax_z")})
        t_star = next((r["step"] for r in rows[1:]
                       if r["g0"] <= SHUT_BAR), None)
        return rows, t_star

    traces, tstars = {}, {}
    for kind, label, _ in ARM_PLAN:
        if washes[kind] is None:
            traces[kind], tstars[kind] = None, None
            continue
        traces[kind], tstars[kind] = wash_cells(kind)
        log(f"{label} g0 trace: "
            + " ".join(f"+{r['step']}:{r['g0']:.4f}" for r in traces[kind]))

    # ---- STAGE A adjudication (registered clauses)
    adj_wash = {}
    if traces["gen"] is None:
        adj_wash = {"verdict": "CONSTRUCTION-CEILING",
                    "clause": ("g3-GEN's root gate failed — no wash "
                               "adjudication (reported, not silently "
                               "dropped); the spec's CONSTRUCTION-CEILING "
                               "branch.")}
        log("STAGE A adjudication: " + adj_wash["verdict"])
    gen_rows = traces["gen"] or []
    gen_ck_g0 = [r["g0"] for r in gen_rows[1:]]
    law_breaks = bool(gen_ck_g0 and min(gen_ck_g0) >= SURVIVE_BAR)
    gen_dies = bool(tstars["gen"] is not None
                    and tstars["gen"] <= DISSOLVE_BY)
    t_gen = tstars["gen"]
    if law_breaks:
        if washes["shal"] is not None and tstars["shal"] is not None:
            shal_alive = False
            adj_wash["verdict"] = "THE LAW BREAKS — SHALLOWNESS"
            adj_wash["clause"] = (
                "g3-GEN g0 >= 0.50 at EVERY checkpoint through +300 AND "
                "g3-SHAL also survived — the no-basin law was a deep-pathway "
                "law all along (W020's contingency answers 'contingent on "
                "pathway depth').")
        else:
            adj_wash["verdict"] = "THE LAW BREAKS — GENERATIVITY"
            adj_wash["clause"] = (
                "g3-GEN g0 >= 0.50 at EVERY checkpoint through +300 while "
                "g3-SHAL died — explicitly-generative storage changed the "
                "basin physics (falsifier fires; g4's architecture axis "
                "inherits).")
    elif gen_dies:
        clock_hit = bool(CLOCK_LO < t_gen <= CLOCK_HI)
        adj_wash["verdict"] = "G3-DIES, THE LAW HOLDS"
        adj_wash["clause"] = (
            f"g3-GEN dissolved: g0 <= {SHUT_BAR} by +{DISSOLVE_BY} "
            f"(first-under-bar t* = +{t_gen}; point-prediction bracket "
            f"({CLOCK_LO}, {CLOCK_HI}] {'HIT' if clock_hit else 'MISSED'}); "
            "the no-basin law is substrate-independent — optimization "
            "pressure washes even an explicitly-generative single-net store.")
        adj_wash["clock_bracket_hit"] = clock_hit
    else:
        adj_wash["verdict"] = "TEXTURE"
        adj_wash["clause"] = (f"neither wash bar fired cleanly (g0 trace "
                              + " ".join(f"+{r['step']}:{r['g0']:.3f}"
                                         for r in gen_rows)
                              + "); numbers reported, no bar shopping.")
    log("=" * 78)
    log(f"STAGE A VERDICT: {adj_wash['verdict']}")
    log(f"  {adj_wash['clause']}")
    log("=" * 78)

    # ---- S-SHAL bracket + ARM D clauses
    adj_shal, adj_hard = {}, {}
    if washes["shal"] is not None and tstars["shal"] is not None \
            and tstars["gen"] is not None:
        ratio = tstars["shal"] / max(tstars["gen"], 1)
        adj_shal = {"t_star": tstars["shal"], "ratio_vs_gen": ratio,
                    "same_bracket": bool(ratio <= SHAL_CLOCK_FACTOR),
                    "clause": (f"g3-SHAL died at +{tstars['shal']} vs g3-GEN "
                               f"+{tstars['gen']} (ratio {ratio:.2f}x, "
                               f"~2x bracket "
                               f"{'inside' if ratio <= SHAL_CLOCK_FACTOR else 'outside'}"
                               ") — the leak/gain channel applies to a "
                               "continuously-firing MLP "
                               + ("with no plateau discount."
                                  if ratio <= SHAL_CLOCK_FACTOR else
                                  "; g3-GEN >> S-SHAL survival: the PLATEAU "
                                  "DISCOUNT becomes the finding."))}
    elif washes["shal"] is not None:
        adj_shal = {"t_star": None,
                    "clause": f"g3-SHAL g0 trace "
                              + " ".join(f"+{r['step']}:{r['g0']:.3f}"
                                         for r in traces["shal"])
                              + " — no under-bar checkpoint by +"
                              f"{traces['shal'][-1]['step']}"}
    if washes["hard"] is not None:
        hard_ck = [r["g0"] for r in traces["hard"][1:]]
        d_survives = bool(hard_ck and min(hard_ck) >= SURVIVE_BAR)
        adj_hard = {"D_SURVIVES": d_survives,
                    "g0_min": min(hard_ck) if hard_ck else None,
                    "t_star": tstars["hard"],
                    "clause": ("D-SURVIVES: g0 >= 0.50 at every checkpoint "
                               "through +300 — SURVIVAL BY BLINDNESS (the "
                               "leak removed, the fact pattern gradient-"
                               "blind); it delineates the law's boundary: "
                               "the no-basin law binds what the gradient can "
                               "see." if d_survives else
                               f"D-DIES (t* = +{tstars['hard']}) — the kill "
                               "entered through a shared door (W_q drift or "
                               "null-key encroachment); the census "
                               "adjudicates.")}
    log(f"S-SHAL: {adj_shal.get('clause', 'n/a')}")
    log(f"ARM D: {adj_hard.get('clause', 'n/a')}")

    # =====================================================================
    # STAGE B — THE NOISE KILL (e185 ported, on g3-GEN)
    # =====================================================================
    noise = {"arms": {}, "D_kill": None, "t_kill": t_gen}
    if t_gen is not None:
        D_kill = next(r["cum_disp"] for r in traces["gen"]
                      if r["step"] == t_gen)
        noise["D_kill"] = D_kill
        log("=" * 78)
        log(f"STAGE B THE NOISE KILL on g3-GEN: D_kill {D_kill:.4f} (A1's "
            f"cumulative all-param displacement at t* = +{t_gen})")
        noise_specs = [
            ("noise", "iid", NOISE_SEED_A, "NOISE-LABELS (iid uniform targets)"),
            ("shuf", "perm", NOISE_SEED_B, "SHUFFLED-TARGET (flat randperm)"),
        ]
        for tag_, mode, nseed, desc in noise_specs:
            if not SMOKE:
                log(f"[thermal] cooldown {COOLDOWN_S:.0f}s before noise {tag_}")
                cooldown(COOLDOWN_S)
            log(f"ARM {tag_.upper()} — {desc}; {CK_NOISE[-1]} steps, seed "
                f"{ARM_SEED} inputs / {nseed} targets")
            a = wash_run(f"noise-{tag_}", "gen", arms["gen"]["root_sd"],
                         anchor_neutral, train_ids, itos, r_eval_xy,
                         bat_ids[(-12, "install60")], ids130, zid, ARM_SEED,
                         CK_NOISE, target_mode=mode, noise_seed=nseed)
            assert a["zeph_violations"] == 0
            G_IN = all(a["x_hashes"].get(s) == washes["gen"]["wash"]["x_hashes"].get(s)
                       for s in CK_NOISE)
            log(f"  G_INPUTS {tag_}: per-step input batches bit-identical to "
                f"A1's first {CK_NOISE[-1]} steps: "
                f"{'PASS' if G_IN else 'FAIL'}")
            assert G_IN, (f"{tag_}: input stream diverged from A1 — the "
                          "delta is not the targets alone")
            noise["arms"][tag_] = {"arm": a, "inputs_identical": bool(G_IN),
                                   "desc": desc}
            for s in sorted(a["sds"]):
                save_ckpt(f"g3_gen_n{tag_}_s{s}", a["sds"][s],
                          {"desc": f"g3_gen root + {s}-step {mode}-target "
                                   f"neutral freeze (noise seed {nseed})",
                           "steps": int(s), "target_mode": mode,
                           "seed": ARM_SEED, "noise_seed": nseed,
                           "kind": "gen",
                           "base": "runs/checkpoints/g3_gen.pt"})
        # displacement-match adjudication
        per_arm = {}
        for tag_ in noise["arms"]:
            tr = noise["arms"][tag_]["arm"]["traj"]
            disp = {t["step"]: t["cum_disp"] for t in tr}
            gm = {t["step"]: t["g0_mean_pz"] for t in tr if "g0_mean_pz" in t}
            ce = {t["step"]: t["ce_r"] for t in tr if "ce_r" in t}
            M = next((s for s in CK_NOISE if disp.get(s, 0.0) >= D_kill), None)
            matched = [s for s in CK_NOISE if disp.get(s, 0.0) >= D_kill]
            kills = bool(M is not None and gm.get(M, 1.0) <= SHUT_BAR)
            spares = bool(matched and all(gm.get(s, 0.0) >= SURVIVE_BAR
                                          for s in matched))
            per_arm[tag_] = {"M": M, "matched": matched,
                             "gm12_at_M": gm.get(M), "g0_at_M": gm.get(M),
                             "ce_r_at_M": ce.get(M),
                             "matched_g0": {s: gm.get(s) for s in matched},
                             "matched_ce_r": {s: ce.get(s) for s in matched},
                             "KILLS": kills, "SPARES": spares,
                             "collateral": bool(kills and ce.get(M, 0.0)
                                               >= CE_COLLATERAL)}
        noise["per_arm"] = per_arm
        both_kill = all(per_arm[t]["KILLS"] for t in per_arm)
        both_spare = all(per_arm[t]["SPARES"] for t in per_arm)
        if both_kill:
            collateral = all(per_arm[t]["collateral"] for t in per_arm)
            noise["verdict"] = "NOISE-KILLS"
            noise["clause"] = (
                f"BOTH noise arms killed g3-GEN at displacement-match "
                + "; ".join(f"{t}: g0 {per_arm[t]['g0_at_M']:.4f} at +"
                            f"{per_arm[t]['M']}, CE_R "
                            f"{per_arm[t]['ce_r_at_M']:.2f}" for t in per_arm)
                + f" (D_kill {D_kill:.4f}) — COLLATERAL "
                f"{'confirmed (CE_R >= 3.0, the organism dies with the fact)' if collateral else 'NOT confirmed (CE_R < 3.0 at match — reported)'}: "
                "no generative immunity; the plateau protects against small "
                "displacements, matched displacement exceeds it; undirected "
                "damage is not gain-selective.")
            noise["collateral_confirmed"] = collateral
        elif both_spare:
            noise["verdict"] = "NOISE-SPARES"
            noise["clause"] = ("BOTH noise arms spared g3-GEN at matched "
                               "displacement — a corpus-directed mechanism "
                               "clause for the generative substrate (the "
                               "no-basin law's content-free clause BREAKS).")
        else:
            noise["verdict"] = "TEXTURE"
            noise["clause"] = ("partial damage at displacement-match; full "
                              "traces reported, no bar shopping.")
        log(f"STAGE B VERDICT: {noise['verdict']}")
        log(f"  {noise['clause']}")
    else:
        noise["verdict"] = "SKIPPED (no A1 kill — the law-break branch)"
        log("STAGE B skipped: A1 never went under bar")

    # =====================================================================
    # STAGE C — THE EVAL BATTERY
    # =====================================================================
    C = {}

    # ---- C1 + C2 (census + closure at t*; on HARD at its own t* if it died)
    def census(kind, sd_root, sd_t, tag):
        """The kill-site census at an arm's t*: retrieval state on fact
        queries (argmax identity / a_fact mass / ||inj|| gain), neutral-gate
        state, per-tensor displacement, the query-attribution 2x2, and the
        store's isolated retrieval on root queries. All scalars (metrics-
        safe)."""
        net_r, net_w = evl_load(kind, sd_root), evl_load(kind, sd_t)
        out = {"tag": tag}
        if kind == "shal":
            del net_r, net_w
            out["note"] = "census is store-class instruments (g3-GEN/HARD); " \
                          "SHAL has no retrieval state"
            return out
        fact_rows = slice(PRE - 1, PRE + 6)          # positions 129..135
        pat = torch.arange(7).unsqueeze(0)           # the correct key j
        for n, key in ((net_r, "root"), (net_w, "washed")):
            n.eval()
            n.store.record = True
            _ = n(pool_band_x[:, :-1])
            q = n.store.cache["q"].clone()
            a = n.store.cache["a"].clone()
            inj = n.store.cache["inj_norm"].clone()
            _ = n(anchor_neutral[:, :-1])
            a_neut = n.store.cache["a"].clone()
            n.store.record = False
            out[key] = {
                "a_fact": float(a[:, fact_rows, :7].gather(
                    -1, pat.expand(q.shape[0], -1).unsqueeze(-1)).mean()),
                "argmax_correct": float(
                    (a[:, fact_rows].argmax(-1)
                     == pat.expand(q.shape[0], -1)).float().mean()),
                "argmax_null_frac": float(
                    (a[:, fact_rows].argmax(-1) == NULL_IDX).float().mean()),
                "inj_norm_fact": float(inj[:, fact_rows].mean()),
                "a_null_on_fact": float(a[:, fact_rows, NULL_IDX].mean()),
                "a_fact_on_neutral": float(a_neut[:, :, :7].mean()),
                "a_null_on_neutral": float(a_neut[:, :, NULL_IDX].mean()),
            }
        out["inj_gain_ratio"] = (out["washed"]["inj_norm_fact"]
                                 / max(out["root"]["inj_norm_fact"], 1e-12))
        # query-attribution 2x2: root-q x washed-K vs washed-q x root-K
        qs = {}
        for n, key in ((net_r, "q_r"), (net_w, "q_w")):
            n.store.record = True
            _ = n(pool_band_x[:, :-1])
            qs[key] = n.store.cache["q"].clone()
            n.store.record = False
        K_r = F.normalize(net_r.store.K.detach(), dim=-1)
        K_w = F.normalize(net_w.store.K.detach(), dim=-1)
        for name_, q_, K_ in (("root_q__root_K", qs["q_r"], K_r),
                              ("root_q__washed_K", qs["q_r"], K_w),
                              ("washed_q__root_K", qs["q_w"], K_r),
                              ("washed_q__washed_K", qs["q_w"], K_w)):
            a_ = F.softmax(BETA * (F.normalize(q_[:, fact_rows], dim=-1)
                                   @ K_.t()), dim=-1)
            out[name_] = {
                "a_fact": float(a_.gather(
                    -1, pat.expand(a_.shape[0], -1).unsqueeze(-1)).mean()),
                "argmax_correct": float(
                    (a_.argmax(-1) == pat.expand(a_.shape[0], -1))
                    .float().mean()),
                "a_null": float(a_[:, :, NULL_IDX].mean()),
            }
        del net_r, net_w
        return out

    def classify_kill_site(cen, bypass_g0):
        """The frozen census classification (docstring definitions):
        GAIN = argmax intact (>= 0.8), a_fact >= 0.5, ||inj|| <= 30% root;
        QUERY = retrieval collapsed with key geometry intact (root-q x
        washed-K still retrieves); GATE = the null key encroached (root-q x
        washed-K: a_null up by >= 0.2 over root-K, or null wins the argmax);
        ROUTE = bypass fails to restore >= 0.5 with retrieval intact;
        else MIXED. storage-without-expression = retrieval intact at the
        kill (g0 is under the bar by construction at t*)."""
        w = cen["washed"]
        retrieval_intact = bool(w["argmax_correct"] >= ARGMAX_BAR
                                and w["a_fact"] >= A_FACT_BAR)
        bypass_ok = bool(bypass_g0 >= RESTORE_BAR)
        null_encroached = bool(
            cen["root_q__washed_K"]["a_null"]
            >= cen["root_q__root_K"]["a_null"] + 0.2
            or w["argmax_null_frac"] >= 0.5)
        if (not bypass_ok) and retrieval_intact:
            site = "ROUTE"
        elif retrieval_intact and cen["inj_gain_ratio"] <= INJ_RATIO_BAR:
            site = "GAIN"
        elif not retrieval_intact and \
                cen["root_q__washed_K"]["a_fact"] >= A_FACT_BAR:
            site = "QUERY"
        elif not retrieval_intact and null_encroached:
            site = "GATE"
        else:
            site = "MIXED"
        return {"site": site, "retrieval_intact": retrieval_intact,
                "bypass_g0": bypass_g0, "bypass_restores": bypass_ok,
                "null_encroached": null_encroached,
                "storage_without_expression": retrieval_intact}

    if t_gen is not None:
        sd_t = washes["gen"]["wash"]["sds"][t_gen]
        sd_root = arms["gen"]["root_sd"]
        log("=" * 78)
        log(f"STAGE C1 THE KILL-SITE CENSUS at t* = +{t_gen} (g3-GEN)")
        cen = census("gen", sd_root, sd_t, f"gen-t{t_gen}")
        per_tensor = {k: float((sd_t[k].float() - sd_root[k].float()).norm())
                      for k in store_keys("gen")}
        per_tensor["store_total"] = float(sum(v ** 2 for k, v
                                              in per_tensor.items()) ** 0.5)
        cen["per_tensor_displacement"] = per_tensor
        log(f"  retrieval(washed): a_fact {cen['washed']['a_fact']:.4f} "
            f"argmax-correct {cen['washed']['argmax_correct']:.4f} | inj gain "
            f"{cen['inj_gain_ratio']:.4f} of root | attribution: root-q x "
            f"washed-K a_fact {cen['root_q__washed_K']['a_fact']:.4f}, "
            f"washed-q x root-K {cen['washed_q__root_K']['a_fact']:.4f}")
        log("  per-tensor displacement: "
            + " ".join(f"{tensor_short(k) if k != 'store_total' else k} {v:.4f}"
                       for k, v in per_tensor.items()))

        log(f"STAGE C2 THE CLOSURE PARTITION (2x2 at t* = +{t_gen})")
        skeys = store_keys("gen")
        hkeys = [k for k in sd_t if k not in skeys]
        closure = {}
        sd_byp, g_byp = restore_class(sd_t, sd_root, skeys, "store")
        closure["bypass_root_store_into_washed_host"] = {
            "gate": g_byp,
            "g0": battery_cell(evl_load("gen", sd_byp), ids130,
                               zid)["mean_pz"],
            "site_onset": read_fact_at(evl_load("gen", sd_byp), pool_183_x,
                                       name_ids, zid, SITE_ADDR_ROW,
                                       SITE_Z_XCOL)["pz_onset_mean"]}
        sd_comp, g_comp = restore_class(sd_root, sd_t, skeys, "store")
        closure["washed_store_into_root_host"] = {
            "gate": g_comp,
            "g0": battery_cell(evl_load("gen", sd_comp), ids130,
                               zid)["mean_pz"]}
        sd_all, g_all = restore_class(sd_t, sd_root,
                                      list(sd_t.keys()), "all")
        g0_all = battery_cell(evl_load("gen", sd_all), ids130, zid)["mean_pz"]
        g0_root = arms["gen"]["dial"]["base"][0]["mean_pz"]
        closure["all_restored_sanity"] = {
            "gate": g_all, "g0": g0_all, "root_g0": g0_root,
            "pass": bool(abs(g0_all - g0_root) < G_BIT_TOL)}
        assert closure["all_restored_sanity"]["pass"], \
            "all-restored is not bit-exact with root"
        bypass_g0 = closure["bypass_root_store_into_washed_host"]["g0"]
        site = classify_kill_site(cen, bypass_g0)
        closure["verdict"] = {
            "bypass_restores": bool(bypass_g0 >= RESTORE_BAR),
            "reading": (f"bypass restores g0 to {bypass_g0:.4f} "
                        f"(>= {RESTORE_BAR}) — the route is intact, the STORE "
                        "died" if bypass_g0 >= RESTORE_BAR else
                        f"bypass restores only {bypass_g0:.4f} (< "
                        f"{RESTORE_BAR}) — ROUTE-KILL (the alphabet itself "
                        "moved: ln_f/lm_head)")}
        log(f"  bypass probe: g0 {bypass_g0:.4f} | washed-store-into-root-host "
            f"{closure['washed_store_into_root_host']['g0']:.4f} | "
            f"all-restored {g0_all:.6f} vs root {g0_root:.6f}")
        log(f"  KILL SITE: {site['site']} (retrieval intact "
            f"{site['retrieval_intact']}, storage-without-expression "
            f"{site.get('storage_without_expression')})")
        C["census"] = cen
        C["census_site"] = site
        C["closure"] = closure

        # ---- ARM D census if it died
        if tstars["hard"] is not None:
            log(f"ARM D census at its t* = +{tstars['hard']}")
            cen_d = census("hard", arms["hard"]["root_sd"],
                           washes["hard"]["wash"]["sds"][tstars["hard"]],
                           f"hard-t{tstars['hard']}")
            sd_byp_d, _ = restore_class(
                washes["hard"]["wash"]["sds"][tstars["hard"]],
                arms["hard"]["root_sd"], store_keys("hard"), "store")
            cen_d["bypass_g0"] = battery_cell(evl_load("hard", sd_byp_d),
                                              ids130, zid)["mean_pz"]
            C["census_hard"] = cen_d
            log(f"  D(washed): a_fact {cen_d['washed']['a_fact']:.4f} "
                f"argmax {cen_d['washed']['argmax_correct']:.4f} inj gain "
                f"{cen_d['inj_gain_ratio']:.4f} bypass "
                f"{cen_d['bypass_g0']:.4f}")

        # ---- C3 THE TWO-BASIN MAP
        log("=" * 78)
        log("STAGE C3 THE TWO-BASIN MAP")
        net_r = evl_load("gen", sd_root)
        net_r.eval()
        net_r.store.record = True
        _ = net_r(pool_band_x[:, :-1])
        q_fact = net_r.store.cache["q"].clone()      # (60,255,64)
        _ = net_r(anchor_neutral[:, :-1])
        q_neut = net_r.store.cache["q"].clone()      # (16,255,64)
        net_r.store.record = False
        K_root = F.normalize(net_r.store.K.detach(), dim=-1)
        rows = slice(PRE - 1, PRE + 6)
        jidx = torch.arange(7).unsqueeze(0).expand(q_fact.shape[0], -1)

        def a_of(q_rows, K):
            return F.softmax(BETA * (F.normalize(q_rows, dim=-1) @ K.t()),
                             dim=-1)

        # (i) STATE BASIN — gamma sweep (fact -> corpus interpolation)
        ggen = torch.Generator().manual_seed(GAMMA_SEED)
        q_flat = q_neut.reshape(-1, q_neut.shape[-1])
        gam_rows = []
        for gam in GAMMAS:
            pick = torch.randint(q_flat.shape[0], (q_fact.shape[0], 7),
                                 generator=ggen)
            q_c = q_flat[pick]                        # (60,7,64)
            q_g = (1 - gam) * q_fact[:, rows, :] + gam * q_c
            a_g = a_of(q_g, K_root)
            gam_rows.append({
                "gamma": gam, "a_fact": float(a_g.gather(
                    -1, jidx.unsqueeze(-1)).mean()),
                "argmax_correct": float((a_g.argmax(-1) == jidx)
                                        .float().mean())})
        # sigma sweep: fidelity (query space) + readout (behavioral)
        sig_rows = []
        for i, sig in enumerate(SIGMAS):
            g = torch.Generator().manual_seed(SIGMA_SEED0 + i)
            qn = q_fact[:, rows, :]
            q_p = qn + sig * qn.norm(dim=-1, keepdim=True) * torch.randn(
                qn.shape, generator=g)
            a_p = a_of(q_p, K_root)
            net_r.store.q_noise, net_r.store.q_gen = sig, g
            g0_noisy = battery_pz(net_r, ids130, zid)
            net_r.store.q_noise, net_r.store.q_gen = 0.0, None
            cos = float((F.normalize(qn, dim=-1) * F.normalize(q_p, dim=-1))
                        .sum(-1).mean())
            sig_rows.append({"sigma": sig, "mean_cos_to_root_q": cos,
                             "a_fact": float(a_p.gather(
                                 -1, jidx.unsqueeze(-1)).mean()),
                             "argmax_correct": float(
                                 (a_p.argmax(-1) == jidx).float().mean()),
                             "g0_readout": g0_noisy})
        del net_r
        state_basin = {"gamma_sweep": gam_rows, "sigma_sweep": sig_rows,
                       "wide": bool(
                           next(r["g0_readout"] for r in sig_rows
                                if r["sigma"] == 1.0) >= SURVIVE_BAR
                           and next(r["a_fact"] for r in gam_rows
                                    if r["gamma"] == 0.5) >= A_FACT_BAR)}
        log("  STATE basin: sigma sweep g0 "
            + " ".join(f"{r['sigma']}:{r['g0_readout']:.3f}"
                       for r in sig_rows)
            + " | gamma a_fact "
            + " ".join(f"{r['gamma']}:{r['a_fact']:.3f}" for r in gam_rows)
            + f" -> {'WIDE' if state_basin['wide'] else 'NARROW'}")

        # (ii) PARAMETER BASIN — lambda sweep along A1's measured direction
        dst = sd_disp(sd_t, sd_root, skeys)
        lam_rows = []
        for lam in LAMBDAS:
            sd_l = {k: v.clone() for k, v in sd_root.items()}
            for k in skeys:
                sd_l[k] = sd_root[k] + lam * (sd_t[k] - sd_root[k])
            lam_rows.append({"lambda": lam, "store_disp": lam * dst,
                             "g0": battery_cell(evl_load("gen", sd_l),
                                                ids130, zid)["mean_pz"]})
        # isotropic matched-L2 ladder (e011c's matched-energy methodology)
        iso_rows = []
        flat_keys = skeys
        for lvl in ISO_LEVELS:
            L2 = lvl * dst
            g0s = []
            for s_ in ISO_SEEDS:
                g = torch.Generator().manual_seed(s_)
                pert = {k: torch.randn(sd_root[k].shape, generator=g)
                        for k in flat_keys}
                pn = float(sum(float(pert[k].norm() ** 2)
                               for k in flat_keys) ** 0.5)
                sd_n = {k: v.clone() for k, v in sd_root.items()}
                for k in flat_keys:
                    sd_n[k] = sd_root[k] + (L2 / max(pn, 1e-12)) * pert[k]
                g0s.append(battery_cell(evl_load("gen", sd_n), ids130,
                                        zid)["mean_pz"])
            iso_rows.append({"level": lvl, "store_disp": L2,
                             "g0_mean": float(np.mean(g0s)),
                             "g0_min": float(np.min(g0s)),
                             "g0_max": float(np.max(g0s)),
                             "g0_seeds": g0s})
        lam_star = next((r["lambda"] for r in lam_rows
                         if r["g0"] <= SHUT_BAR), None)
        param_basin = {"Dstore_t_star": dst,
                       "lambda_sweep": lam_rows, "isotropic": iso_rows,
                       "lambda_star": lam_star,
                       "narrow": bool(lam_star is not None and lam_star <= 2.0),
                       "D_disc_sdisc": sdisc.get("D_disc"),
                       "per_coord": {
                           "g3_store_kill_rms": dst / (STORE_PARAMS ** 0.5),
                           "sdisc_kill_rms": (sdisc.get("D_disc", 0.0)
                                              / (F2_PARAMS ** 0.5))}}
        log("  PARAM basin (store subspace): lambda g0 "
            + " ".join(f"{r['lambda']}:{r['g0']:.3f}" for r in lam_rows)
            + " | isotropic g0 "
            + " ".join(f"{r['level']}:{r['g0_mean']:.3f}" for r in iso_rows)
            + f" -> lambda* {lam_star} "
            f"({'NARROW' if param_basin['narrow'] else 'not narrow'}); "
            f"Dstore(t*) {dst:.4f} vs S-DISC D_disc "
            f"{sdisc.get('D_disc', float('nan')):.4f}")
        C["state_basin"] = state_basin
        C["param_basin"] = param_basin

        # ---- C4 THE RESURRECTION RIDER (report-only, e179's protocol)
        log("=" * 78)
        log(f"STAGE C4 THE RESURRECTION RIDER from t* = +{t_gen}: ONE replay "
            f"batch, then {RIDER_STEPS} wash steps")
        rp = replay_step("rider", "gen", sd_t, pool_band_x, pool_band_mask,
                         anchor_bank, train_ids, zid, RIDER_REPLAY_SEED,
                         WASH_LR)
        g0_rp = battery_cell(evl_load("gen", rp["sd"]), ids130,
                             zid)["mean_pz"]
        rid = wash_run("rider-wash", "gen", rp["sd"], anchor_neutral,
                       train_ids, itos, r_eval_xy,
                       bat_ids[(-12, "install60")], ids130, zid, RIDER_SEED,
                       CK_RIDER)
        rider_trace = [{"step": 0, "g0": g0_rp,
                        "ce_r": ce_fixed_cpu(evl_load("gen", rp["sd"]),
                                             *r_eval_xy)}]
        rider_trace += [{"step": t["step"], "g0": t["g0_mean_pz"],
                         "ce_r": t["ce_r"]} for t in rid["traj"]
                        if "g0_mean_pz" in t]
        fires = bool(any(r["g0"] >= RIDER_BAR for r in rider_trace))
        C["rider"] = {"replay_ce": rp["replay_ce"], "trace": rider_trace,
                      "FIRES": fires,
                      "clause": (f"one replay event at t*+1 restored g0 to "
                                 f"{max(r['g0'] for r in rider_trace):.4f} "
                                 "within the 50-wash-step horizon — the "
                                 "re-entry economy is substrate-independent "
                                 "(memory as RHYTHM holds for the organ)."
                                 if fires else
                                 "one replay event did NOT restore g0 >= 0.5 "
                                 "within 50 wash steps — the re-entry economy "
                                 "does not transfer unchanged to the "
                                 "generative substrate (report-only).")}
        log(f"  rider: replay CE {rp['replay_ce']:.4f}; g0 trace "
            + " ".join(f"+{r['step']}:{r['g0']:.4f}" for r in rider_trace)
            + f" -> {'FIRES' if fires else 'does not fire'}")

    # ---- geo profile + wpe census co-reports (root, all arms)
    C["geo_profile"] = {}
    for kind, label, _ in ARM_PLAN:
        d = arms[kind]["dial"]
        co = d["census_old"]["rows"]
        C["geo_profile"][kind] = {
            "gm12": d["base"][-12]["mean_pz"], "g0": d["base"][0]["mean_pz"],
            "gp12": d["base"][12]["mean_pz"],
            "flatish": bool(abs(d["base"][-12]["mean_pz"]
                                - d["base"][0]["mean_pz"]) < 0.15
                            and abs(d["base"][12]["mean_pz"]
                                    - d["base"][0]["mean_pz"]) < 0.15),
            "wpe_census": {"row0_strength": co["0"]["strength"],
                           "A129": co["129"]["strength"],
                           "band121_129_max": max(
                               co[str(r)]["strength"]
                               for r in range(121, 130) if str(r) in co)},
            "deletions": d["del_table"],
            "site_read": d["site_read"]}
        log(f"geo profile {label}: g-12 {d['base'][-12]['mean_pz']:.4f} / g0 "
            f"{d['base'][0]['mean_pz']:.4f} / g+12 {d['base'][12]['mean_pz']:.4f}"
            f" | wpe band max "
            f"{C['geo_profile'][kind]['wpe_census']['band121_129_max']:.4f}")

    # =====================================================================
    # adjudication summary
    # =====================================================================
    mechanism_ok = None
    if "census_site" in C:
        s = C["census_site"]
        mechanism_ok = bool(
            s["site"] == "GAIN" and s["retrieval_intact"]
            and s.get("storage_without_expression"))
    two_basin_ok = None
    if "state_basin" in C:
        two_basin_ok = bool(C["state_basin"]["wide"]
                            and C["param_basin"]["narrow"])
    adjudication = {
        "wash": adj_wash,
        "s_shal": adj_shal, "arm_d": adj_hard,
        "noise": {"verdict": noise.get("verdict"),
                  "clause": noise.get("clause"),
                  "per_arm": {t: {k: v for k, v in d.items()
                                  if k != "arm"} for t, d
                              in noise.get("per_arm", {}).items()},
                  "D_kill": noise.get("D_kill")},
        "mechanism": None if mechanism_ok is None else {
            "kill_site": C["census_site"]["site"],
            "retrieval_intact_at_kill": C["census_site"]["retrieval_intact"],
            "storage_without_expression":
                C["census_site"].get("storage_without_expression"),
            "MECHANISM_PREDICTION": mechanism_ok,
            "clause": ("the kill site is the GAIN and the retrieval is "
                       "intact at the kill — STORAGE WITHOUT EXPRESSION, the "
                       "registered mechanism holds." if mechanism_ok else
                       "the registered gain/storage-without-expression "
                       "mechanism did NOT hold cleanly; census numbers "
                       "reported, falsifier noted.")},
        "two_basin": None if two_basin_ok is None else {
            "state_basin_wide": C["state_basin"]["wide"],
            "param_basin_narrow": C["param_basin"]["narrow"],
            "TWO_BASIN_DISSOCIATION": two_basin_ok,
            "clause": ("the state basin measures WIDE while the parameter "
                       "basin measures NARROW — PP lore conflates the basin "
                       "where retrieval lives (state space) with the basin "
                       "where forgetting happens (parameter space); THE WASH "
                       "DOES NOT TRAVERSE THE ENERGY LANDSCAPE — IT "
                       "RE-SCULPTS IT." if two_basin_ok else
                       "the two-basin dissociation did not hold cleanly; "
                       "sweep tables reported.")},
        "rider": C.get("rider"),
        "law_falsifier": {"fires": law_breaks,
                          "note": "g3-GEN g0 >= 0.50 at EVERY checkpoint "
                                  f"through +{WASH_STEPS}"},
    }
    log("=" * 78)
    log("G3 ADJUDICATION SUMMARY")
    for k, v in adjudication.items():
        if isinstance(v, dict) and "verdict" in v:
            log(f"  {k}: {v['verdict']}")
        elif isinstance(v, dict) and "clause" in v:
            log(f"  {k}: {v['clause'][:150]}")
    log("=" * 78)

    # ---------------- outputs
    metrics = {
        "experiment": "g3_generative_store",
        "date": common.now_iso(),
        "purpose": ("THE GENERATIVE-MEMORY ARCHITECTURE: does explicitly-"
                    "generative storage (a Hopfield pattern store, "
                    "prediction-as-storage) change the basin physics, or is "
                    "the no-basin law substrate-independent? — the two-basin "
                    "map (state vs parameter), the kill-site census, the "
                    "decoupling control, the noise kill, the rider"),
        "registered_prediction": REGISTERED_PREDICTION,
        "design_spec": "scratch/g3_design.md (committed 2026-09-29; frozen)",
        "host": {"base": f"runs/checkpoints/{BASE_CK}",
                 "cfg": {"n_layer": 4, "n_head": 4, "n_embd": 128,
                         "block_size": 512, "params": F2_PARAMS},
                 "ce_r_base": ce_base, "ce_budget": ce_budget},
        "store": {"kind_params": {"gen": STORE_PARAMS, "hard": STORE_PARAMS,
                                  "shal": 22_640},
                  "beta": BETA, "d_key": D_KEY, "n_patterns": N_PAT,
                  "null_idx": NULL_IDX, "placement": "after block 3, "
                  "before ln_f (position-wise, causal by construction)"},
        "constructions": {kind: {
            "desc": a["desc"], "params": a["params"],
            "n_store_params": a["con"]["n_store_params"],
            "traj": a["con"]["traj"], "steps_ran": a["con"]["steps_ran"],
            "device": a["con"]["device"],
            "one_shot_write": a["con"]["write"],
            "host_bit_identical_to_base":
                a["con"]["host_bit_identical_to_base"],
            "trunk_gate": a["con"]["trunk_gate"],
            "joint_calibration": (None if a["cal"] is None else
                                  {"traj": a["cal"]["traj"],
                                   "steps_ran": a["cal"]["steps_ran"]}),
            "G_ROOT": a["G_ROOT"],
            "dial": {k: v for k, v in a["dial"].items() if k != "tag"},
        } for kind, a in arms.items()},
        "stage_A_wash": {kind: (None if washes[kind] is None else {
            "recipe": "e176N arm A VERBATIM via e157 (neutral bank seed 170, "
                      "batch 16+16 full-token CE, AdamW (0.9,0.95) wd 0.1 "
                      "lr 1e-3 clip 1.0, seed 10902)",
            "traj": washes[kind]["wash"]["traj"],
            "trace": traces[kind], "t_star": tstars[kind],
            "steps_ran": washes[kind]["wash"]["steps_ran"],
            "device": washes[kind]["wash"]["device"],
            "dials": {s: {k: v for k, v in d.items() if k != "tag"}
                      for s, d in washes[kind]["dials"].items()},
        }) for kind in ("gen", "shal", "hard")},
        "s_disc_control": sdisc,
        "stage_B_noise": {k: v for k, v in noise.items() if k != "arms"} | {
            "arms": {t: {"desc": d["desc"], "traj": d["arm"]["traj"],
                         "inputs_identical": d["inputs_identical"],
                         "steps_ran": d["arm"]["steps_ran"],
                         "device": d["arm"]["device"],
                         "x_hashes": d["arm"]["x_hashes"]}
                    for t, d in noise.get("arms", {}).items()}},
        "stage_C": C,
        "adjudication": adjudication,
        "gates": {"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                  "G_POOL": G_POOL, "G_ANCHOR": G_ANCHOR,
                  "G_SURG": gates_surg, "gpu_at_start": gpu_status(),
                  "gpu_parked": GPU_PARKED, "park_reason": PARK_REASON,
                  "device_events": device_events},
        "checkpoints": CKPT_INVENTORY,
        "trims": trims,
        "deviations": deviations,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "smoke": SMOKE, "threads": torch.get_num_threads(),
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))
    log("metrics.json written")
    plot(rd / "generative_store.png", metrics)
    log(f"plot written -> {rd}")
    return 0


# ------------------------------------------------------------------ plot

def plot(path: Path, M: dict):
    A = M["stage_A_wash"]
    adj = M["adjudication"]
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))

    # (0,0) THE WASH — clocks
    ax = axes[0, 0]
    colors = {"gen": "crimson", "shal": "darkorange", "hard": "seagreen"}
    labels = {"gen": "g3-GEN (Hopfield store)", "shal": "g3-SHAL (MLP twin)",
              "hard": "g3-HARD (arm D, top-1 gate)"}
    for kind in ("gen", "shal", "hard"):
        if A[kind] is None:
            continue
        t = [r["step"] for r in A[kind]["trace"]]
        g = [r["g0"] for r in A[kind]["trace"]]
        ax.plot(t, g, "o-", color=colors[kind], lw=2.2, ms=5,
                label=labels[kind])
    sd = M["s_disc_control"]["trace"]
    ax.plot(sd["freeze_steps"], sd["g0"], "s--", color="steelblue", lw=1.6,
            ms=5, label="S-DISC (e157 cells, free)")
    ax.axhline(SHUT_BAR, color="gray", ls=":", lw=1.2,
               label=f"dissolve bar {SHUT_BAR}")
    ax.axhline(SURVIVE_BAR, color="seagreen", ls="--", lw=1.2,
               label=f"survive bar {SURVIVE_BAR}")
    ax.set_xscale("symlog", linthresh=4)
    ax.set_xlabel("neutral-wash steps (e176n arm A verbatim, seed 10902)")
    ax.set_ylabel("install-60 battery p(Z) (g0)")
    t_gen_lbl = ("+" + str((A["gen"] or {}).get("t_star"))
                 if (A["gen"] or {}).get("t_star") is not None else "n/a")
    ax.set_title(f"THE WASH — {adj['wash']['verdict']}\n"
                 f"(t*_GEN {t_gen_lbl}; S-DISC +1)", fontsize=9)
    ax.legend(fontsize=7)

    # (0,1) THE TWO-BASIN MAP — survival vs displacement (the headline)
    ax = axes[0, 1]
    sdisc_disp = M["s_disc_control"]["displacement"]
    steps = M["s_disc_control"]["trace"]["freeze_steps"]
    ax.plot([sdisc_disp[str(s)] for s in steps], sd["g0"], "s--",
            color="steelblue", lw=1.8, ms=6, label="S-DISC (e157, measured)")
    for kind in ("gen", "shal"):
        if A[kind] is None:
            continue
        rows = [r for r in A[kind]["trace"] if r["step"] > 0]
        ax.plot([r["cum_disp"] for r in rows], [r["g0"] for r in rows],
                "o-", color=colors[kind], lw=2.0, ms=5,
                label=f"{labels[kind]} (all-param)")
    ax.axhline(SHUT_BAR, color="gray", ls=":", lw=1.1)
    ax.axhline(SURVIVE_BAR, color="seagreen", ls="--", lw=1.1)
    ax.set_xlabel("cumulative all-param displacement  ||theta_t - theta_0||_2")
    ax.set_ylabel("g0")
    ax.set_xscale("symlog", linthresh=0.5)
    tb = adj.get("two_basin") or {}
    ax.set_title("THE TWO-BASIN MAP (headline) — survival vs displacement\n"
                 f"all-param wash curves (bottom axis); store-subspace "
                 f"curves right panel; two-basin dissociation: "
                 f"{tb.get('TWO_BASIN_DISSOCIATION')}", fontsize=9)
    ax.legend(fontsize=7)

    # (0,2) PARAMETER BASIN — store-subspace (wash direction vs isotropic)
    ax = axes[0, 2]
    if "param_basin" in M["stage_C"]:
        pb = M["stage_C"]["param_basin"]
        lam = pb["lambda_sweep"]
        ax.plot([r["store_disp"] for r in lam], [r["g0"] for r in lam],
                "o-", color="crimson", lw=2.2, ms=6,
                label="lambda along A1's MEASURED wash direction")
        iso = pb["isotropic"]
        ax.errorbar([r["store_disp"] for r in iso],
                    [r["g0_mean"] for r in iso],
                    yerr=[np.subtract(r["g0_mean"], r["g0_min"]) for r in iso],
                    fmt="^--", color="gray", lw=1.4, ms=6, capsize=3,
                    label="isotropic matched-L2 noise (3 seeds)")
        # the wash trajectory itself in store-subspace currency
        rows = [r for r in A["gen"]["trace"] if r["step"] > 0
                and r.get("disp_store") is not None]
        ax.plot([r["disp_store"] for r in rows], [r["g0"] for r in rows],
                "s:", color="darkred", lw=1.4, ms=5,
                label="A1 wash (store-subspace)")
        ax.axhline(SHUT_BAR, color="gray", ls=":", lw=1.1)
        ax.set_xlabel("store-subspace displacement (L2, 17,410 params)")
        ax.set_ylabel("g0")
        ax.set_title(f"PARAMETER BASIN — narrow? {pb['narrow']} "
                     f"(lambda* {pb['lambda_star']})\n"
                     f"Dstore(t*) {pb['Dstore_t_star']:.3f} vs S-DISC "
                     f"D_disc {pb.get('D_disc_sdisc', float('nan')):.3f} "
                     f"(873k params)", fontsize=9)
        ax.legend(fontsize=7)

    # (1,0) STATE BASIN — the wide side
    ax = axes[1, 0]
    if "state_basin" in M["stage_C"]:
        sb = M["stage_C"]["state_basin"]
        sig = sb["sigma_sweep"]
        ax.plot([r["sigma"] for r in sig], [r["g0_readout"] for r in sig],
                "o-", color="navy", lw=2.2, ms=6, label="g0 readout")
        ax.plot([r["sigma"] for r in sig], [r["a_fact"] for r in sig],
                "^--", color="purple", lw=1.6, ms=6,
                label="retrieval fidelity a_fact")
        ax.set_xscale("log")
        ax.set_xlabel("query noise sigma (||q|| units; mean cosine "
                      "= 1/sqrt(1+sigma^2): 1.0 -> 0.71, 2.0 -> 0.45)")
        ax.set_ylabel("g0 / a_fact")
        ax.axhline(SURVIVE_BAR, color="seagreen", ls="--", lw=1.1)
        ax.set_title(f"STATE BASIN — wide? {sb['wide']} (Hopfield basins by "
                     "construction)\nreadout survives large query "
                     "perturbation while the wash kills at tiny parameter "
                     "displacement", fontsize=9)
        ax.legend(fontsize=7)
        gam = sb["gamma_sweep"]
        ax2 = ax.twinx()
        ax2.plot([r["gamma"] for r in gam], [r["a_fact"] for r in gam],
                 "v:", color="teal", lw=1.4, ms=5,
                 label="gamma sweep a_fact (top x)")
        ax2.set_ylabel("gamma a_fact", color="teal")
        ax2.legend(fontsize=7, loc="lower left")

    # (1,1) THE NOISE KILL
    ax = axes[1, 1]
    nb = M["stage_B_noise"]
    if A["gen"] is not None:
        rows = [r for r in A["gen"]["trace"] if 0 < r["step"] <= 10]
        ax.plot([r["cum_disp"] for r in rows], [r["g0"] for r in rows],
                "o-", color="crimson", lw=2.0, ms=5, label="A1 (true targets)")
    for tag_, col, lab in (("noise", "gray", "NOISE-LABELS (iid)"),
                           ("shuf", "black", "SHUFFLED-TARGET (perm)")):
        pa = nb.get("per_arm", {}).get(tag_)
        if pa is None:
            continue
        tr = nb["arms"][tag_]["traj"]
        rows = [t for t in tr if "g0_mean_pz" in t]
        ax.plot([t["cum_disp"] for t in rows], [t["g0_mean_pz"] for t in rows],
                "^--", color=col, lw=1.6, ms=6, label=lab)
    if nb.get("D_kill") is not None:
        ax.axvline(nb["D_kill"], color="k", ls=":", lw=1.4,
                   label=f"D_kill {nb['D_kill']:.3f}")
    ax.axhline(SHUT_BAR, color="gray", ls=":", lw=1.1)
    ax.set_xlabel("cumulative all-param displacement (first 10 steps)")
    ax.set_ylabel("g0")
    ax.set_title(f"THE NOISE KILL — {nb.get('verdict')}\n"
                 f"{(nb.get('clause') or '')[:130]}", fontsize=8)
    ax.legend(fontsize=7)

    # (1,2) census bars + rider
    ax = axes[1, 2]
    if "census" in M["stage_C"]:
        cen = M["stage_C"]["census"]
        names = ["a_fact", "argmax", "inj_gain", "bypass_g0"]
        vals = [cen["washed"]["a_fact"], cen["washed"]["argmax_correct"],
                M["stage_C"]["census_site"]["inj_ratio"]
                if "inj_ratio" in M["stage_C"]["census_site"]
                else cen["inj_gain_ratio"],
                M["stage_C"]["closure"]["bypass_root_store_into_washed_host"]["g0"]]
        bars = ax.bar(names, vals, color=["purple", "violet", "crimson",
                                          "seagreen"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                    ha="center", fontsize=8)
        ax.axhline(A_FACT_BAR, color="gray", ls=":", lw=1.1)
        ax.axhline(RESTORE_BAR, color="seagreen", ls="--", lw=1.1)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"KILL-SITE CENSUS at t* — site "
                     f"{M['stage_C']['census_site']['site']} "
                     f"(storage-without-expression: "
                     f"{M['stage_C']['census_site'].get('storage_without_expression')})",
                     fontsize=9)
        rid = M["stage_C"].get("rider")
        if rid:
            ax.text(0.02, 0.62, "RIDER " + ("FIRES" if rid["FIRES"]
                                            else "no-fire")
                    + "\n" + " ".join(f"+{r['step']}:{r['g0']:.2f}"
                                      for r in rid["trace"]),
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(fc="whitesmoke", ec="gray"))
    else:
        ax.text(0.5, 0.5, "no kill (law-break branch)", ha="center")

    fig.suptitle(
        f"G3 THE GENERATIVE-MEMORY ARCHITECTURE (~0.89M; host frozen "
        f"e098_base_s4305) | wash: {adj['wash']['verdict']} | noise: "
        f"{M['stage_B_noise'].get('verdict')} | kill site: "
        f"{(M['stage_C'].get('census_site') or {}).get('site', 'n/a')} | "
        f"two-basin: {(adj.get('two_basin') or {}).get('TWO_BASIN_DISSOCIATION', 'n/a')}",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
