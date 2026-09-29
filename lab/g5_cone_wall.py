"""G5 — WALL THE QUERY CONE (T126's named composition; QUEUE row g5).

THE COMPOSITION this file runs: g3's Hopfield organ + g1b's commit-and-project
wall applied to the QUERY PROJECTION ALONE — not the whole net. Context being
composed (frozen): g1b proved commit(R)+L2-projection makes memory
architectural against displacement (WALL-HOLDS, 0.918 maintained through
+300 at R=0.7 on the 2.74M line); g3 proved the Hopfield store dies at +1
with the kill-site in the QUERY (washed-q x root-K a_fact 0.08; root-q x
washed-K 0.91 — the store forgets its address, not its content) and that
isotropic noise on the store SPARES it (the basin is a CONE).

REGISTERED (QUEUE row g5 + the dispatch, VERBATIM — frozen before compute;
no bar shopping):
  "WALL THE QUERY CONE (T126's named composition — g3's store + a wall on
   W_q only; the much cheaper well). the Hopfield organ with commit+project
   on the QUERY PROJECTION alone (17k params, not the whole net). Bars:
   CONE-WALL-HOLDS (the store survives the wash with W_q walled); the
   adaptation-tax prediction (near-zero — the rest of the net free)."
  Dispatch: "Prediction: the store survives the wash with W_q walled, at
  near-zero adaptation tax (the rest of the net stays free). The falsifier:
  the wall on W_q fails to maintain (the kill is not confined to W_q at
  this substrate)."

THE REGISTRATION'S OWN AMBIGUITY, discharged by running BOTH readings plus a
knife-edge rung (all share theta0 and the seed-10902 stream; the deltas are
the wall's keyset and radius ONLY):
  WQ   commit+project on store.W_q.weight ALONE (8,192 params — the
       dispatch's "with W_q walled" / the falsifier's "the wall on W_q"),
       R_q = 0.03 (inside the measured cone; see PREFLIGHT).
  WQE  the same keyset at the knife edge, R_q = 0.09 (= the measured +1
       wash-direction W_q kill displacement) — with WQ it brackets the cone
       radius dynamically (g1's the-wall-re-measures-the-basin logic).
  STW  commit+project on all four store tensors (17,408 params — the queue
       row's "(17k params, not the whole net)" reading), R_store = 0.04.
  FREE the unwalled control (g3's stage-A wash re-run in-run: the clock,
       the tax baseline, the same-process census reference).

BARS (frozen):
  CONE-WALL-HOLDS  the WQ cell's g0 >= 0.50 at EVERY checkpoint
                   {1,2,4,10,50,100,200,300} of the 300-step neutral wash —
                   the composed system works (g3's SURVIVE convention).
  TAX-NEAR-ZERO    |CE_R(WQ,+300) - CE_R(FREE,+300)| < 0.05 nats (g1's
                   WALL-FREE convention; the rest of the net stays free).
  FALSIFIER        the wall on W_q fails to maintain — the kill is not
                   confined to W_q at this substrate.
Co-report clauses (they fix wording, not bars):
  STORE-SURVIVES (per arm)   the organ-level probe store-into-root-host
                             (the arm's settled store, ROOT host) g0 >= 0.50
                             at EVERY checkpoint — the wall's product read
                             where the host is healthy.
  QUERY-CONE-HELD (per arm)  the LIVE retrieval dial (a_fact >= 0.5 AND
                             argmax >= 0.8, g3's census bars, through the
                             arm's own washed host) at EVERY checkpoint.
  CONE-BRACKET               WQ stores + WQE does not => the cone radius
                             lives in (R_q, R_qe] = (0.03, 0.09].

PREFLIGHT (scratch/g5_preflight.py, eval-only on g3's STORED checkpoints,
run and frozen BEFORE this experiment — no new training; registered here so
the run's numbers adjudicate against a stated prior, not hindsight):
  * the host route dies at every horizon: root-store-into-washed-host g0
    1e-5 (+10) / 5e-5 (+50) / 7e-4 (+300) — a second, unwalled kill site.
  * the wall's end-state (root W_q + washed K/V/W_o + washed host) reads
    g0 1e-5..6e-4 — behaviorally dead regardless of the W_q wall.
  * the store itself survives in isolation: {root W_q, washed K/V/W_o} into
    the ROOT host reads 0.762 (+10) / 0.703 (+50) / 0.575 (+300); root-q x
    washed-K a_fact 0.90 at every horizon (the keys hold).
  * W_q's OWN weight drift is nearly harmless to retrieval: a_fact 0.848 at
    the FULL +1 displacement (0.0905) on a root host (g0 0.306 there);
    a_fact 0.891 / g0 ~0.75 at 0.030. The query cone is NOT exitable
    through W_q's weights alone at these radii.
  * THE LIVE QUERY (root W_q grafted into the washed host — the WQ cell's
    state) is dead anyway: a_fact 0.083 (+1), 0.0003 (+2), 0.061 (+10),
    0.186 (+50), 0.322 (+300) vs root 0.907 — the query cone is exited
    THROUGH THE HOST (q = W_q . LN(h); the wash moves h), with W_q pinned.
  Net preflight expectation: the FALSIFIER fires, with the kill attributed
  by the census to the host on BOTH sides of the store (query-side h drift
  + the expression route), while the wall itself works mechanically and the
  store survives at the organ level; the tax reads near zero.

OPERATIONALIZATIONS (fixed before compute; they fix clauses, not bars):
  * Primary dial g0 = ABSOLUTE install-60 battery mean p(Z) at ctx offset 0
    (e176/e176N/e157/e157's ruler, g3's corpus rebuild, same seeds).
  * WASH = g3's stage-A wash VERBATIM (e176N arm A via e157: neutral bank
    seed 170, batch 16 neutral + 16 random, full-token CE, AdamW (0.9,0.95)
    wd 0.1 lr 1e-3 clip 1.0, seed 10902), 300 steps, checkpoints
    {1,2,4,10,50,100,200,300}. All four arms share the stream (md5-gated).
  * THE WALL = g1b's commit + per-forward hard in-place L2 projection
    (flat interior, all directions of the WALLED SUBSPACE, forward
    semantics; the optimizer may step one step outside between forwards —
    the registered fuzz). Anchors are python-held snapshots of theta0's
    walled tensors, gated bit-equal (see deviations).
  * ROOT = runs/checkpoints/g3_gen.pt loaded DIRECTLY as theta0 (bit-exact
    gate vs file; g1b's precedent) — the composition requirement: the SAME
    root behind g3's wash numbers; the delta is the wall alone. No
    construction is re-run.
  * EVAL SEMANTICS = g1b's PIVOT: checkpoint reads run on the SETTLED state
    (one no-grad projection — the state entering the next forward; the
    model's defined state); the raw post-step sd is bookkept separately
    (G-PIN). The census/closure operate on settled sds.
  * G-PIN (subspace translation of g1's G-PIN): raw walled displacement <=
    R + 2*lr*sqrt(n_walled) at every checkpoint (the one-step Adam fuzz for
    the subspace: lr*sqrt(8192) = 0.0905 for WQ/WQE, lr*sqrt(17408) =
    0.132 for STW); settled <= R + 1e-6.
  * CENSUS AFTER WASH = g3's census + closure VERBATIM (retrieval state,
    inj gain, neutral gate, the query-attribution 2x2, per-tensor
    displacement, the {root,washed} store x {root,washed} host partition
    with confinement gates, the kill-site classification) EXTENDED by the
    preflight's two attribution cells that separate W_q's own drift from
    the host's contribution to the query: wallstate-q (root W_q inside the
    washed host) and wqdrift-q (washed W_q inside the root host), both
    against root K. Census points: WQ {+1,+50,+300}, FREE {+1,+300},
    WQE {+300}, STW {+300}.
  * TAX = CE_R (the fixed e065 val bank, CPU) at +300; the in-batch CE
    curves co-reported.

INSTRUMENT PROVENANCE: everything is lab/g3_generative_store.py VERBATIM via
import (the organ, battery_cell/battery_pz/ce_fixed_cpu/val_windows/
read_fact_at/restore_class/sd_disp/tensor_short/pick_dev/migrate_to_cpu and
every protocol constant — which are themselves e157/e176n's = the
e176/e161/e152/e151/e143/e131/e119/e113/e068/e065/e043 lineage); the wash
trainer is g3's wash_run (e157's finetune_freeze + e185's bookkeeping)
copied with the wall additions (settle-at-checkpoint, subspace pin, the
live retrieval dial, the store-into-root-host probe) — g1's own addition
pattern to e176n's finetune_freeze; census()/classify_kill_site()/lean_dial
are g3's VERBATIM copies (nested in g3's main, not importable) with the two
registered attribution cells added; the wall class is g1's CommittedGPT
semantics (lab/g1_anchored_ball.py) subspace-restricted. Copied, not
imported, to own the device policy where extended.

COMPUTE ENVELOPE: GPU allowed (pick_dev park-once double-poll + mid-run
guard every 25 steps at mem > 85% or temp > 80C; NO concurrent — g2b runs
CPU; park on thermal), cooldown 75 s before each training (g3's envelope),
per-training caps 180 s GPU / 1800 s CPU; ALL readouts CPU-side; <= 1M
total params (890,880 — g3's organism unchanged); ZERO new data; the root
and g3's wash checkpoints are read from disk (eval-only).

Outputs: runs/g5/{metrics.json, cone_wall.png}; checkpoints
runs/checkpoints/g5_{tag}_s{N}.pt (the SETTLED body sd + wall meta). No
NOTES/THINKING/QUEUE/STATE edits; single commit, no push.

Run:  cd lab && python g5_cone_wall.py   (G5_SMOKE=1 shakedown)
"""
from __future__ import annotations

import copy
import hashlib
import random
import sys
import time
from pathlib import Path

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # GPU allowed, gated below

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                          # noqa: E402

import common                                          # noqa: E402
from common import CharCorpus, cooldown, gpu_status, run_dir, save_json  # noqa: E402

import e043_install as E43                             # noqa: E402 (REPO, find_occ,
                                                      # SPLICE_RNG, jsonable)
import g3_generative_store as G3                       # noqa: E402 — the organ +
                                                      # every instrument, VERBATIM

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import torch.nn.functional as F                        # noqa: E402

SMOKE = os.environ.get("G5_SMOKE") == "1"
CPU = torch.device("cpu")

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)
G3.log = log                                           # unify the timeline (g1b)

CKPT_DIR = E43.REPO / "runs" / "checkpoints"
ROOT_CK = "g3_gen.pt"                 # g3-GEN's constructed root (theta0)
BASE_CK = G3.BASE_CK                  # e098_base_s4305.pt (the pristine trunk)

# ---- the wall ladder (operationalized from the preflight; frozen) --------------
R_Q, R_QE, R_ST = 0.03, 0.09, 0.04
WALL_WQ_KEYS = ["store.W_q.weight"]
WALL_ST_KEYS = list(G3.store_keys("gen"))            # W_q, K, V, W_o (17,408)
N_WQ, N_ST = 8_192, G3.STORE_PARAMS

CK_WASH: tuple[int, ...] = G3.CK_WASH if not SMOKE else (1, 2, 4)
WASH_STEPS = CK_WASH[-1]
GPU_CAP_S, CPU_CAP_S = 180.0, 1800.0                 # the dispatch's caps
COOLDOWN_S = 75.0                                     # g3's envelope (60-90 s)
MIDRUN_POLL_EVERY = G3.MIDRUN_POLL_EVERY

# ---- bars (frozen) --------------------------------------------------------------
SURVIVE_BAR = G3.SURVIVE_BAR          # 0.50 — CONE-WALL-HOLDS / STORE-SURVIVES
SHUT_BAR = G3.SHUT_BAR                # 0.27 — dies (co-report)
TAX_BAR = 0.05                        # g1's WALL-FREE convention
A_FACT_BAR = G3.A_FACT_BAR            # 0.50 — retrieval mass
ARGMAX_BAR = G3.ARGMAX_BAR            # 0.80 — argmax-correct fraction
RESTORE_BAR = G3.RESTORE_BAR          # 0.50 — bypass restore
ROOT_EXPR_BAR = G3.ROOT_EXPR_BAR      # 0.50 — G_ROOT_EXPR
STOREOFF_BAR = G3.STOREOFF_BAR        # 0.27 — G_STOREOFF
CE_CLEAN_SLACK = G3.CE_CLEAN_SLACK    # 0.10 — G_CECLEAN
PIN_SETTLE_TOL = 1e-6

REGISTERED_PREDICTION = {
    "composition": "g3's Hopfield organ + g1b's commit-and-project wall on "
        "the QUERY PROJECTION ALONE (not the whole net): WQ = W_q only "
        "(8,192 params, the dispatch's 'with W_q walled'); STW = the whole "
        "store (17,408, the queue row's '17k params' reading; g3's docstring "
        "says 17,410 — its metrics record 17,408); WQE = W_q at "
        "the knife-edge radius; FREE = the unwalled control.",
    "predicted": "CONE-WALL-HOLDS: the store survives the wash (g0 >= 0.50 "
        "at EVERY checkpoint through +300) with W_q walled — the composed "
        "system works; the adaptation tax NEAR-ZERO (the rest of the net "
        "stays free).",
    "falsifier": "the wall on W_q fails to maintain (the kill is not "
        "confined to W_q at this substrate).",
    "bars": "CONE-WALL-HOLDS = WQ g0 >= 0.50 at every checkpoint "
        "{1,2,4,10,50,100,200,300}; TAX-NEAR-ZERO = |CE_R(WQ,+300) - "
        "CE_R(FREE,+300)| < 0.05; co-reports STORE-SURVIVES (organ-level "
        "store-into-root-host >= 0.50 at every checkpoint), QUERY-CONE-HELD "
        "(live a_fact >= 0.5 and argmax >= 0.8 at every checkpoint), "
        "CONE-BRACKET (WQ stores + WQE does not => radius in (0.03, 0.09]).",
    "registration": "QUEUE.md row g5 + the g5 dispatch (2026-09-29), frozen "
        "verbatim before compute. The R radii and operationalizations are "
        "fixed in this docstring and in the preflight record; no bar "
        "shopping.",
}

PREFLIGHT = {
    "script": "scratch/g5_preflight.py (eval-only on runs/checkpoints/"
              "g3_gen*.pt; run 2026-09-29 BEFORE this experiment; frozen)",
    "route_death": {"+10": 1e-05, "+50": 5e-05, "+300": 7e-04,
                    "reading": "root-store-into-washed-host g0 (bypass): "
                               "the host's fact route is dead at every "
                               "horizon — a second, unwalled kill site"},
    "wall_end_state": {"+10": 1e-05, "+50": 4e-05, "+300": 6e-04,
                       "reading": "root W_q + washed K/V/W_o + washed host: "
                                  "behaviorally dead regardless of the W_q "
                                  "wall"},
    "store_survival": {"+10": 0.7622, "+50": 0.7032, "+300": 0.5751,
                       "reading": "{root W_q, washed K/V/W_o} into the ROOT "
                                  "host — the store survives in isolation, "
                                  "eroding through V/W_o drift"},
    "root_q__washed_K": {"+10": 0.9005, "+50": 0.8994, "+300": 0.9055,
                         "reading": "the keys hold the patterns at every "
                                    "horizon"},
    "wq_own_drift": {"a_fact@0.0905_root_host": 0.8478,
                     "g0@0.0905_root_host": 0.3059,
                     "a_fact@0.030_root_host": 0.8913,
                     "g0@0.030_root_host": 0.7562,
                     "reading": "W_q's OWN weight drift is nearly harmless "
                                "to retrieval — the cone is not exitable "
                                "through W_q's weights alone at these radii"},
    "live_query_root_Wq_in_washed_host": {
        "+1": 0.0832, "+2": 0.0003, "+4": 0.0008, "+10": 0.0613,
        "+50": 0.1857, "+100": 0.2983, "+200": 0.3236, "+300": 0.3218,
        "root": 0.9069,
        "reading": "THE LIVE QUERY (root W_q inside the washed host) is dead "
                   "anyway: the query cone is exited THROUGH THE HOST "
                   "(q = W_q . LN(h); the wash moves h), with W_q pinned"},
    "net_expectation": "the FALSIFIER fires: g0 dead in every walled cell "
                       "(host kills both the query side and the route), the "
                       "wall works mechanically, the store survives at the "
                       "organ level, the tax reads near zero.",
}

trims: list[str] = []
deviations: list[str] = [
    "THE REGISTRATION'S AMBIGUITY (the queue row's 'QUERY PROJECTION alone "
    "(17k params)' — W_q is 8,192 params, the store 17,408) is discharged by "
    "running BOTH readings (WQ, STW) plus the knife-edge rung (WQE); the "
    "PRIMARY registered cell is WQ (the dispatch's 'with W_q walled' / the "
    "falsifier's 'the wall on W_q'), STW and WQE are co-cells on the same "
    "bars.",
    "R RADII operationalized from the preflight (registered in the "
    "docstring before compute): R_q 0.03 inside the measured cone (a_fact "
    "0.89 / g0 ~0.75 at 0.030 on a root host), R_qe 0.09 = the measured +1 "
    "wash-direction W_q kill displacement (g1's knife-edge rung), R_st 0.04 "
    "inside the store's lambda-sweep survival band (g3: 0.066 -> g0 0.637).",
    "THE WALL implementation: g1's CommittedGPT semantics subspace-"
    "restricted; the anchors are python-held snapshots of theta0's walled "
    "tensors (gated bit-equal at commit; G_WALLROOT) instead of g1's "
    "state_dict buffers — the census's state-dict surgery (restore_class) "
    "then stays clean of anchor keys; the wall config is carried in the "
    "checkpoint meta + metrics, and reconstructs deterministically from "
    "theta0 (the commit snapshot IS theta0's tensors).",
    "ROOT loaded DIRECTLY from runs/checkpoints/g3_gen.pt (bit-exact gate "
    "vs file; g1b's precedent) — the composition requirement (same root, "
    "delta = the wall); no construction re-run, no new seeds drawn for it.",
    "The wash trainer is g3's wash_run arithmetic VERBATIM + the wall "
    "bookkeeping (g1's addition pattern): settle-at-checkpoint (g1b's ARMED-"
    "twin semantics — the +1 reads the settled/projected state entering "
    "forward 2), the subspace pin (raw + settled), the LIVE retrieval dial "
    "and the store-into-root-host probe at every checkpoint.",
    "census()/classify_kill_site()/lean_dial() are g3's VERBATIM copies "
    "(nested in g3's main, not importable), EXTENDED by the two registered "
    "attribution cells (wallstate-q = root W_q in the washed host; wqdrift-q "
    "= washed W_q in the root host) that separate W_q's own drift from the "
    "host's contribution to the query — the cells the preflight showed are "
    "load-bearing for the falsifier's attribution.",
    "TAX is measured on CE_R (the fixed val bank) at +300 (g1's WALL-FREE "
    "convention |d| < 0.05); g1b used the in-batch CE at +300 — both "
    "co-reported.",
    "Checkpoints store the SETTLED body sd (the model's defined state) + "
    "wall meta, not g1's raw post-step weights: without state_dict anchors "
    "the settled sd + theta0 fully define the cell (the raw-vs-settled gap "
    "is one Adam step, bookkept in G_PIN).",
    "Single seed per cell (the wash draws 10902 VERBATIM; one lineage — "
    "family 2, the e098 s4305 host), n=1 per cell — the arc's honesty "
    "convention; replicates owed before any noun moves.",
    "Smoke mode trims: 8-step washes with checkpoints {1,2,4}, census at "
    "WQ +4 only, lean dials, no cooldowns, no checkpoint writes — nothing "
    "adjudicated.",
]


# ------------------------------------------------------------------ the wall
# PROVENANCE: lab/g1_anchored_ball.py's CommittedGPT semantics, subspace-
# restricted (the projection covers a KEYSET; everything else stays free).

class WalledGPT(G3.GenMemGPT):
    """g3's GenMemGPT + a commitment well on a parameter keyset.

    commit(keys, R, anchor_sd) snapshots the named parameters from the
    (cpu) anchor state dict; every forward opens with a hard in-place L2
    projection of the WALLED SUBSPACE onto ||theta_walled - anchor||_2 <= R
    (flat interior, all directions of the subspace). Unwalled parameters are
    never touched — the rest of the net stays free. Before commit (or with
    an empty keyset) this class is behaviorally bit-identical to GenMemGPT."""

    def __init__(self, wall_keys: tuple[str, ...] = ()):
        super().__init__(G3.F2_CFG, G3.KINDS["gen"]["organ"]())
        self.wall_keys = list(wall_keys)
        self.wall_R: float | None = None
        self.wall_anchor: dict[str, torch.Tensor] = {}
        self._wp: list[tuple[str, torch.nn.Parameter]] = []

    @torch.no_grad()
    def commit(self, keys, R: float, anchor_sd: dict) -> None:
        """Snapshot the walled keyset from anchor_sd (theta0); set R."""
        named = dict(self.named_parameters())
        self.wall_keys = list(keys)
        self.wall_anchor = {k: anchor_sd[k].detach().clone()
                            .to(named[k].device) for k in self.wall_keys}
        self._wp = [(k, named[k]) for k in self.wall_keys]
        self.wall_R = float(R)

    @property
    def walled(self) -> bool:
        return bool(self._wp) and self.wall_R is not None

    @torch.no_grad()
    def _enforce_wall(self) -> None:
        if not self.walled:
            return
        d = None
        for k, p in self._wp:
            a = self.wall_anchor[k].to(p.device)
            dd = ((p - a) ** 2).sum()
            d = dd if d is None else (d + dd)
        d = float(d.sqrt())
        if d > self.wall_R:
            s = self.wall_R / d
            for k, p in self._wp:
                a = self.wall_anchor[k].to(p.device)
                p.copy_(a + (p - a) * s)

    def forward(self, idx, targets=None):
        self._enforce_wall()                       # forward semantics
        return super().forward(idx, targets)


def wall_cfg(keys, R, anchor_sd: dict) -> dict:
    """A serializable wall config (cpu anchors from theta0)."""
    return {"keys": list(keys), "R": float(R),
            "anchor": {k: anchor_sd[k].detach().clone() for k in keys}}


def build_walled(sd0: dict, keys, R) -> WalledGPT:
    """Fresh WalledGPT loaded with sd0 (strict) + the wall committed."""
    torch.manual_seed(0)                    # G3.evl_load's organ-init seed
    net = WalledGPT(keys)
    net.load_state_dict(sd0, strict=True)
    if list(keys):
        net.commit(keys, R, sd0)
    net.eval()
    return net


def settle_sd(sd: dict, wall: dict | None) -> dict:
    """One no-grad projection of the walled subspace onto the ball (g1b's
    settle); identity when unwalled. Returns a copy."""
    out = {k: v.clone() for k, v in sd.items()}
    if not wall or not wall["keys"]:
        return out
    d = None
    for k in wall["keys"]:
        dd = ((out[k].float() - wall["anchor"][k].float()) ** 2).sum()
        d = dd if d is None else (d + dd)
    d = float(d.sqrt())
    if d > wall["R"]:
        s = wall["R"] / d
        for k in wall["keys"]:
            out[k] = wall["anchor"][k] + (out[k] - wall["anchor"][k]) * s
    return out


def subspace_disp(sd_a: dict, sd_b: dict, keys) -> float:
    return float(sum(float((sd_a[k].float() - sd_b[k].float()).norm() ** 2)
                     for k in keys) ** 0.5)


# ------------------------------------------------------------------ trainings
# PROVENANCE: g3's wash_run (e157's finetune_freeze + e185's bookkeeping)
# VERBATIM arithmetic, + the wall bookkeeping (g1's pattern) + the two
# per-checkpoint probes (live retrieval dial; store-into-root-host).

def g5_wash(tag: str, sd0: dict, wall: dict | None, anchor: torch.Tensor,
            train_ids: torch.Tensor, itos, r_eval_xy, gm12_ids, g0_ids,
            zid: int, seed: int, ckpt_steps: tuple[int, ...],
            root_sd: dict, pool_x: torch.Tensor, lr: float = G3.WASH_LR):
    """THE NEUTRAL WASH (g3's wash_run VERBATIM) under the wall. The net's
    forward projects the walled subspace (g1 semantics); checkpoint reads
    run on the SETTLED state (g1b's ARMED-twin PIVOT); per checkpoint we
    bookkeep the subspace pin (raw + settled), the g3 displacement table
    (store/host/per-tensor), the LIVE retrieval dial (a_fact/argmax through
    the arm's own washed host) and the store-into-root-host probe (the
    organ-level survival dial)."""
    dev = G3.pick_dev(tag)
    cap = GPU_CAP_S if dev.type == "cuda" else CPU_CAP_S
    ckpt_set = set(ckpt_steps)
    n_steps = ckpt_steps[-1]
    net = build_walled(sd0, wall["keys"] if wall else [], 
                       wall["R"] if wall else 0.0).to(dev)
    if wall and wall["keys"]:
        net.commit(wall["keys"], wall["R"], sd0)   # anchors onto dev
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)
    n_anc = anchor.shape[0]
    skeys = G3.store_keys("gen")
    hkeys = [k for k in sd0 if k not in skeys]
    wkeys = wall["keys"] if wall else []
    traj, sds_raw, sds, t_start = [], {}, {}, time.time()
    step, zeph_checks = 0, 0
    evl = copy.deepcopy(net).to(CPU)               # CPU eval twin (settled)
    theta0 = torch.cat([p.detach().reshape(-1).cpu()
                        for p in net.parameters()]).clone()
    x_hashes: dict[int, str] = {}
    for step in range(1, n_steps + 1):
        aj = torch.randint(n_anc, (16,), generator=gen)
        rj = torch.randint(len(train_ids) - G3.BLOCK - 1, (16,), generator=gen)
        anc = anchor[aj]
        rnd = torch.stack([train_ids[s: s + G3.BLOCK] for s in rj])
        for w in rnd:
            txt = "".join(itos[int(c)] for c in w[:64]) + \
                  "".join(itos[int(c)] for c in w[192:])
            if "ZEPH" in txt:
                zeph_checks += 1
        x = torch.cat([anc[:, :-1], rnd[:, :-1]], 0).to(dev)
        y = torch.cat([anc[:, 1:], rnd[:, 1:]], 0).to(dev)
        x_hashes[step] = hashlib.md5(
            x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        logits, _ = net(x)                         # <- the wall projects here
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
            sd_set = settle_sd(sd_cpu, wall)
            sds_raw[step], sds[step] = sd_cpu, sd_set
            evl.load_state_dict(sd_set)
            evl.eval()
            gz = G3.battery_cell(evl, gm12_ids, zid)
            gz0 = G3.battery_cell(evl, g0_ids, zid)
            ce_r = G3.ce_fixed_cpu(evl, *r_eval_xy)
            # the live retrieval dial (g3's census arithmetic, through the
            # arm's own washed host)
            with torch.no_grad():
                evl.store.record = True
                _ = evl(pool_x[:, :-1])
                a_live = evl.store.cache["a"].clone()
                evl.store.record = False
            fact_rows = slice(G3.PRE - 1, G3.PRE + 6)
            pat = torch.arange(7).unsqueeze(0).expand(a_live.shape[0], -1)
            a_fact = float(a_live[:, fact_rows, :7].gather(
                -1, pat.unsqueeze(-1)).mean())
            argmax_c = float((a_live[:, fact_rows].argmax(-1) == pat)
                             .float().mean())
            # the organ-level survival probe: the arm's settled store into
            # the ROOT host (confinement-gated transplant)
            sd_h, g_h = G3.restore_class(sd_set, root_sd, hkeys, "host")
            g0_sh = G3.battery_cell(G3.evl_load("gen", sd_h), g0_ids,
                                    zid)["mean_pz"]
            assert g_h["pass"], f"{tag}: host-restore gate FAILED {g_h}"
            row = {
                "g_m12_mean_pz": gz["mean_pz"], "g0_mean_pz": gz0["mean_pz"],
                "frac_argmax_z": gz["frac_argmax_z"], "ce_r": ce_r,
                "disp_store": G3.sd_disp(sd_set, sd0, skeys),
                "disp_host": G3.sd_disp(sd_set, sd0, hkeys),
                "disp_all_snap": G3.sd_disp(sd_set, sd0),
                "a_fact_live": a_fact, "argmax_live": argmax_c,
                "g0_store_into_root_host": g0_sh,
                "host_restore_gate": {k: v for k, v in g_h.items()},
                **{f"disp_{G3.tensor_short(k)}": float(
                    (sd_set[k].float() - sd0[k].float()).norm())
                   for k in skeys}}
            if wkeys:
                row["disp_wall_raw"] = subspace_disp(sd_cpu, sd0, wkeys)
                row["disp_wall_settled"] = subspace_disp(sd_set, sd0, wkeys)
            traj[-1].update(row)
            log(f"  [{tag}] CKPT +{step:4d} g-12 {gz['mean_pz']:.4f} "
                f"g0 {gz0['mean_pz']:.4f} CE_R {ce_r:.4f} |d| {cum_disp:.4f} "
                f"| a_fact {a_fact:.4f} | store->root-host {g0_sh:.4f}"
                + (f" | wall raw {row['disp_wall_raw']:.4f} "
                   f"settled {row['disp_wall_settled']:.4f}" if wkeys else ""))
        if (time.time() - t_start) > cap:
            log(f"  [{tag}] time cap {cap:.0f}s at s{step}")
            trims.append(f"{tag}: time cap at step {step}")
            break
        if dev.type == "cuda" and step % MIDRUN_POLL_EVERY == 0:
            s = gpu_status()
            if s["mem_total"] > 0 and (s["mem_used"] > 0.85 * s["mem_total"]
                                       or s["temp"] > 80):
                G3.device_events.append({"tag": tag, "step": step,
                                         "event": "MID-RUN MIGRATION",
                                         "status": s})
                log(f"  [{tag}] MID-RUN GPU contention at s{step} ({s}) -> CPU")
                G3.migrate_to_cpu(net, opt)
                dev, cap = CPU, CPU_CAP_S
    net.eval()
    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"sds": sds, "sds_raw": sds_raw, "traj": traj,
            "steps_ran": step, "seed": seed, "lr": lr,
            "zeph_violations": zeph_checks, "x_hashes": x_hashes,
            "device": str(dev), "wall": (None if not wall else {
                "keys": wall["keys"], "R": wall["R"]})}


# ------------------------------------------------------------------ census
# PROVENANCE: g3's census()/classify_kill_site() VERBATIM copies (nested in
# g3's main, not importable) + the two registered attribution cells.

def census_g5(sd_root: dict, sd_t: dict, tag: str, pool_x: torch.Tensor,
              anchor_neutral: torch.Tensor) -> dict:
    """g3's kill-site census on the SETTLED state, extended: the query
    attribution is a {root, washed} host x {root, washed} W_q grid against
    root K (plus g3's original q x K 2x2) — wallstate-q (root W_q in the
    washed host) separates the host's contribution to the query from W_q's
    own drift (wqdrift-q)."""
    net_r, net_w = G3.evl_load("gen", sd_root), G3.evl_load("gen", sd_t)
    out: dict = {"tag": tag}
    fact_rows = slice(G3.PRE - 1, G3.PRE + 6)
    pat = torch.arange(7).unsqueeze(0)
    for n, key in ((net_r, "root"), (net_w, "washed")):
        n.eval()
        n.store.record = True
        _ = n(pool_x[:, :-1])
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
                (a[:, fact_rows].argmax(-1) == G3.NULL_IDX).float().mean()),
            "inj_norm_fact": float(inj[:, fact_rows].mean()),
            "a_null_on_fact": float(a[:, fact_rows, G3.NULL_IDX].mean()),
            "a_fact_on_neutral": float(a_neut[:, :, :7].mean()),
            "a_null_on_neutral": float(a_neut[:, :, G3.NULL_IDX].mean()),
        }
    out["inj_gain_ratio"] = (out["washed"]["inj_norm_fact"]
                             / max(out["root"]["inj_norm_fact"], 1e-12))
    # the attribution grid
    qs = {}
    for n, key in ((net_r, "q_r"), (net_w, "q_w")):
        n.store.record = True
        _ = n(pool_x[:, :-1])
        qs[key] = n.store.cache["q"].clone()
        n.store.record = False
    n_wh = G3.evl_load("gen", sd_t)                 # washed host + ROOT W_q
    with torch.no_grad():
        n_wh.store.W_q.weight.copy_(sd_root["store.W_q.weight"])
    n_wh.eval()
    n_wh.store.record = True
    _ = n_wh(pool_x[:, :-1])
    qs["q_wh"] = n_wh.store.cache["q"].clone()      # the WALL-STATE query
    n_wh.store.record = False
    n_hw = G3.evl_load("gen", sd_root)              # root host + washed W_q
    with torch.no_grad():
        n_hw.store.W_q.weight.copy_(sd_t["store.W_q.weight"])
    n_hw.eval()
    n_hw.store.record = True
    _ = n_hw(pool_x[:, :-1])
    qs["q_hw"] = n_hw.store.cache["q"].clone()      # W_q's OWN drift query
    n_hw.store.record = False
    K_r = F.normalize(net_r.store.K.detach(), dim=-1)
    K_w = F.normalize(net_w.store.K.detach(), dim=-1)
    cells = (("root_q__root_K", qs["q_r"], K_r),
             ("root_q__washed_K", qs["q_r"], K_w),
             ("washed_q__root_K", qs["q_w"], K_r),
             ("washed_q__washed_K", qs["q_w"], K_w),
             ("wallstate_q__root_K", qs["q_wh"], K_r),
             ("wqdrift_q__root_K", qs["q_hw"], K_r))
    for name_, q_, K_ in cells:
        a_ = F.softmax(G3.BETA * (F.normalize(q_[:, fact_rows], dim=-1)
                                  @ K_.t()), dim=-1)
        out[name_] = {
            "a_fact": float(a_.gather(
                -1, pat.expand(a_.shape[0], -1).unsqueeze(-1)).mean()),
            "argmax_correct": float(
                (a_.argmax(-1) == pat.expand(a_.shape[0], -1))
                .float().mean()),
            "a_null": float(a_[:, :, G3.NULL_IDX].mean()),
        }
    del net_r, net_w, n_wh, n_hw
    return out


def classify_kill_site(cen: dict, bypass_g0: float) -> dict:
    """g3's frozen census classification VERBATIM."""
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
    elif retrieval_intact and cen["inj_gain_ratio"] <= G3.INJ_RATIO_BAR:
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


CKPT_INVENTORY: dict = {}


def save_ckpt(name: str, sd: dict, meta: dict) -> None:
    if SMOKE:
        return
    path = CKPT_DIR / f"{name}.pt"
    torch.save({"model": sd, "meta": {"experiment": "g5", **meta}}, path)
    CKPT_INVENTORY[name] = {"path": str(path.relative_to(E43.REPO)).replace("\\", "/"),
                            **meta}
    log(f"[ckpt] saved {path.name}")


# ------------------------------------------------------------------ main

def main():
    rd = run_dir("g5_smoke" if SMOKE else "g5")
    common.DEVICE = "cpu"     # ALL readouts CPU-side (the trainer owns devices)
    log(f"G5 WALL THE QUERY CONE (smoke={SMOKE}) -> {rd}")
    log(f"compute: strict per-training GPU gate (park-once) + mid-run guard "
        f"every {MIDRUN_POLL_EVERY} steps; gpu at start: {gpu_status()}; "
        f"threads {torch.get_num_threads()}")

    # ---------------- protocol rebuild (g3's main VERBATIM, batteries only)
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
    for host in G3.HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + G3.POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix, "pass": bool(mix == {"FLORIZEL": 19,
                                                         "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"
    name_ids = corpus.encode(G3.NAME)

    def offset_pool(j: int):
        wins = []
        for p, h in install_occ:
            pre = train_ids[p - G3.PRE - j: p]
            post = train_ids[p + len(h): p + len(h) + G3.POST_CAP - j]
            if len(pre) != G3.PRE + j or len(post) != G3.POST_CAP - j:
                raise RuntimeError(f"window short at p={p} j={j}")
            wins.append(torch.cat([pre, name_ids, post]))
        px = torch.stack(wins)
        pm = torch.zeros(len(wins), G3.BLOCK - 1, dtype=torch.bool)
        pm[:, G3.PRE - 1 + j: G3.PRE - 1 + j + len(name_ids)] = True
        return px, pm

    pool_band_x, _ = offset_pool(0)       # the home band (census instrument)
    pool_183_x, _ = offset_pool(G3.RETEACH_J)
    G_POOL = {
        "band_shape": list(pool_band_x.shape),
        "band_name_in_place": bool(all(
            torch.equal(w[G3.PRE: G3.PRE + len(name_ids)], name_ids)
            for w in pool_band_x)),
        "p183_name_in_place": bool(all(
            torch.equal(w[G3.SITE_Z_XCOL: G3.SITE_Z_XCOL + len(name_ids)],
                        name_ids) for w in pool_183_x)),
        "note": "g3's pools VERBATIM (j=0 the census instrument; j=54 the "
                "site-read instrument); NO pool window enters any training",
    }
    G_POOL["pass"] = bool(G_POOL["band_name_in_place"]
                          and G_POOL["p183_name_in_place"])
    assert G_POOL["pass"], f"pool gate FAILED: {G_POOL}"

    bat_ids = {}
    for j in G3.GEOS:
        for tag_, occ in (("install60", install_occ), ("held30", held_occ)):
            cs = [train_text[p - G3.PRE - j: p] for p, _ in occ]
            bat_ids[(j, tag_)] = torch.stack([corpus.encode(c) for c in cs])
    ids130 = bat_ids[(0, "install60")]
    r_eval_x, r_eval_y = G3.val_windows(val_ids, val_text, 60, G3.R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)

    # ---------------- the neutral stream (e170 via e157 VERBATIM)
    arng = random.Random(G3.E170_ANCHOR_SEED)
    n_starts, tries, rejections = [], 0, 0
    hi_start = len(train_ids) - G3.BLOCK - 2
    while len(n_starts) < 16 and tries < 100000:
        s = arng.randrange(hi_start)
        tries += 1
        txt = train_text[s: s + G3.BLOCK + 1]
        if any(f in txt for f in G3.ANCHOR_FORBIDDEN):
            rejections += 1
            continue
        n_starts.append(s)
    if len(n_starts) != 16:
        raise RuntimeError(f"neutral bank incomplete: {len(n_starts)}/16")
    anchor_neutral = torch.stack([train_ids[s: s + G3.BLOCK]
                                  for s in n_starts])
    host_positions = [p for p in E43.find_occ(train_text, G3.HOSTS[0])]
    host_positions += [p for p in E43.find_occ(train_text, G3.HOSTS[1])]

    def junctions_covered(starts):
        return sum(1 for s in starts
                   if any(s <= p < s + G3.BLOCK + 1 for p in host_positions))

    G_ANCHOR = {
        "construction": ("16 plain corpus windows from train_ids, RNG seed "
                         f"{G3.E170_ANCHOR_SEED}, rejection on FLORIZEL/"
                         "ELIZABETH/ZEPH/MIRABEL in [s, s+257) — e170 "
                         "VERBATIM (g3's copy)"),
        "n_windows": 16, "seed": G3.E170_ANCHOR_SEED, "starts": n_starts,
        "tries": tries, "rejections": rejections,
        "windows_with_host_content": sum(
            1 for s in n_starts
            if any(f in train_text[s: s + G3.BLOCK + 1] for f in G3.HOSTS)),
        "junctions_covered": junctions_covered(n_starts),
    }
    G_ANCHOR["pass"] = bool(
        G_ANCHOR["windows_with_host_content"] == 0
        and G_ANCHOR["junctions_covered"] == 0
        and anchor_neutral.shape == (16, G3.BLOCK))
    assert G_ANCHOR["pass"], f"anchor gate FAILED: {G_ANCHOR}"

    # ---------------- base trunk CE budget (g3's main VERBATIM)
    from common import TinyGPT
    st = torch.load(CKPT_DIR / BASE_CK, map_location="cpu",
                    weights_only=False)
    base_sd = {k: v.clone() for k, v in
               (st["model"] if isinstance(st, dict) and "model" in st
                else st).items()}
    base_net = TinyGPT(G3.F2_CFG)
    base_net.load_state_dict(base_sd)
    base_net.eval()
    ce_base = G3.ce_fixed_cpu(base_net, *r_eval_xy)
    del base_net
    ce_budget = ce_base + CE_CLEAN_SLACK
    log(f"base trunk {BASE_CK}: CE_R {ce_base:.4f} (budget base+"
        f"{CE_CLEAN_SLACK} = {ce_budget:.4f})")

    # ---------------- THE ROOT: g3_gen.pt loaded DIRECTLY (bit-exact)
    raw = torch.load(CKPT_DIR / ROOT_CK, map_location="cpu",
                     weights_only=False)
    theta0 = {k: v.clone() for k, v in raw["model"].items()}
    md0 = max(float((theta0[k].float() - raw["model"][k].float()).abs().max())
              for k in raw["model"])
    G_BITEXACT = {"checkpoint": f"runs/checkpoints/{ROOT_CK}",
                  "meta": raw.get("meta"), "n_tensors": len(raw["model"]),
                  "max_abs_diff_vs_file": md0, "pass": bool(md0 == 0.0)}
    assert G_BITEXACT["pass"], f"root load not bit-exact: {md0}"
    n_params = sum(p.numel() for p in
                   G3.evl_load("gen", theta0).parameters())
    assert n_params == G3.F2_PARAMS + G3.STORE_PARAMS, \
        f"root param count {n_params} != {G3.F2_PARAMS + G3.STORE_PARAMS}"
    log(f"root loaded bit-exact from {ROOT_CK} ({n_params} params; "
        f"meta {raw.get('meta')})")

    def lean_dial(sd, tag):
        """g3's lean_dial VERBATIM (base/held batteries, CE_R, store-off,
        site read) on a plain load."""
        net = G3.evl_load("gen", sd)
        out = {"tag": tag}
        out["base"] = {j: G3.battery_cell(net, bat_ids[(j, "install60")], zid)
                       for j in G3.GEOS}
        out["base_held"] = {j: G3.battery_cell(net, bat_ids[(j, "held30")],
                                               zid) for j in G3.GEOS}
        out["ce_r"] = G3.ce_fixed_cpu(net, *r_eval_xy)
        net.store_disabled = True
        out["g0_store_off"] = G3.battery_pz(net, ids130, zid)
        net.store_disabled = False
        out["site_read"] = G3.read_fact_at(net, pool_183_x, name_ids, zid,
                                           G3.SITE_ADDR_ROW, G3.SITE_Z_XCOL)
        del net
        log(f"[{tag}] " + " ".join(f"g{j:+d} {out['base'][j]['mean_pz']:.4f}"
                                   for j in G3.GEOS)
            + f" | held30 g0 {out['base_held'][0]['mean_pz']:.4f} | CE_R "
            f"{out['ce_r']:.4f} | store-off g0 {out['g0_store_off']:.4f} | "
            f"span@183 {out['site_read']['pname_mean_over7']:.4f}")
        return out

    root_dial = lean_dial(theta0, "g5_root")
    G_ROOT = {
        "G_ROOT_EXPR": {"g0": root_dial["base"][0]["mean_pz"],
                        "bar": ROOT_EXPR_BAR,
                        "pass": bool(root_dial["base"][0]["mean_pz"]
                                     >= ROOT_EXPR_BAR)},
        "G_STOREOFF": {"g0_store_off": root_dial["g0_store_off"],
                       "bar": STOREOFF_BAR,
                       "pass": bool(root_dial["g0_store_off"] <= STOREOFF_BAR)},
        "G_CECLEAN": {"ce_r": root_dial["ce_r"], "budget": ce_budget,
                      "pass": bool(root_dial["ce_r"] <= ce_budget)},
        "prior_g0_g3": 0.8885950446128845,   # runs/g3/metrics.json
    }
    G_ROOT["pass"] = bool(G_ROOT["G_ROOT_EXPR"]["pass"]
                          and G_ROOT["G_STOREOFF"]["pass"]
                          and G_ROOT["G_CECLEAN"]["pass"])
    log(f"G_ROOT: expr {G_ROOT['G_ROOT_EXPR']['g0']:.4f} | store-off "
        f"{G_ROOT['G_STOREOFF']['g0_store_off']:.4f} | CE_R "
        f"{G_ROOT['G_CECLEAN']['ce_r']:.4f} <= {ce_budget:.4f} -> "
        f"{'PASS' if G_ROOT['pass'] else 'FAIL'}")

    # =====================================================================
    # THE ARMS (one wash each; cooldown before each; inputs bit-identical —
    # the deltas are the wall's keyset and radius alone)
    # =====================================================================
    ARM_SPECS = [
        ("FREE", None, None,
         "CONTROL — unwalled neutral wash (g3's stage-A cell re-run in-run: "
         "the clock, the tax baseline, the same-process census reference)"),
        ("WQ", WALL_WQ_KEYS, R_Q,
         "THE REGISTERED COMPOSITION — commit+project on W_q ALONE "
         f"({N_WQ:,} params), R_q {R_Q} (inside the measured cone)"),
        ("WQE", WALL_WQ_KEYS, R_QE,
         f"THE KNIFE EDGE — W_q alone at R_q {R_QE} (= the measured +1 "
         "wash-direction kill displacement); with WQ it brackets the cone "
         "radius"),
        ("STW", WALL_ST_KEYS, R_ST,
         f"THE QUEUE'S 17K READING — commit+project on the whole store "
         f"({N_ST:,} params), R_store {R_ST} (inside g3's lambda-sweep "
         "survival band)"),
    ]

    arms: dict = {}
    G_WALLROOT: dict = {}
    for tag, keys, R, desc in ARM_SPECS:
        log("=" * 78)
        if not SMOKE:
            log(f"[thermal] cooldown {COOLDOWN_S:.0f}s before {tag}")
            cooldown(COOLDOWN_S)
        log(f"ARM {tag} — {desc}")
        wall = None if keys is None else wall_cfg(keys, R, theta0)
        if wall is not None:
            net0 = build_walled(theta0, keys, R)
            # G_WALLROOT: the wall root's tensors are bit-identical to
            # theta0 (the commit is a snapshot; the wall is inert at d=0)
            md = max(float((net0.state_dict()[k].float()
                            - theta0[k].float()).abs().max())
                     for k in theta0)
            anch_ok = all(torch.equal(net0.wall_anchor[k], theta0[k])
                          for k in keys)
            G_WALLROOT[tag] = {
                "keys": list(keys), "R": R, "n_walled": int(sum(
                    theta0[k].numel() for k in keys)),
                "max_abs_diff": md, "anchors_bit_equal": bool(anch_ok),
                "pass": bool(md == 0.0 and anch_ok)}
            assert G_WALLROOT[tag]["pass"], f"{tag}: wall root != theta0"
            log(f"G_WALLROOT[{tag}]: max|diff| {md:.1e}, anchors bit-equal "
                f"(R={R}, {G_WALLROOT[tag]['n_walled']:,} params): PASS")
            del net0
        w = g5_wash(f"wash-{tag}", theta0, wall, anchor_neutral, train_ids,
                    itos, r_eval_xy, bat_ids[(-12, "install60")], ids130,
                    zid, G3.ARM_SEED, CK_WASH, theta0, pool_band_x)
        assert w["zeph_violations"] == 0, f"{tag}: name token leaked"
        # checkpoints (the SETTLED sd + wall meta)
        for s in sorted(w["sds"]):
            if (tag in ("WQ", "FREE") and s in (1, 50, 300)) or \
                    (tag in ("WQE", "STW") and s == CK_WASH[-1]):
                save_ckpt(f"g5_{tag}_s{s}", w["sds"][s],
                          {"desc": f"g3_gen root + {s}-step neutral wash "
                                   f"(e176n arm A verbatim, seed "
                                   f"{G3.ARM_SEED}) under the "
                                   f"{tag} wall",
                           "steps": int(s), "seed": G3.ARM_SEED,
                           "wall_keys": list(keys) if keys else None,
                           "wall_R": R, "kind": "gen",
                           "base": f"runs/checkpoints/{ROOT_CK}",
                           "note": "SETTLED body sd (one no-grad projection "
                                   "applied); anchors = theta0's walled "
                                   "tensors"})
        dials = {"0": root_dial}
        for s in (10, 300):
            if s in w["sds"] and not SMOKE:
                dials[str(s)] = lean_dial(w["sds"][s], f"{tag}-n{s}")
        arms[tag] = {"desc": desc, "wash": w, "dials": dials,
                     "wall": (None if wall is None else
                              {"keys": wall["keys"], "R": wall["R"],
                               "n_walled": G_WALLROOT[tag]["n_walled"]})}

    # =====================================================================
    # CONSTRUCTION-FIDELITY GATES
    # =====================================================================
    G_INPUTS = {"per_step": {}, "pass": None,
                "note": "the seed-10902 aj/rj draw sequence is shared by "
                        "ALL four arms — per-step input batches are "
                        "bit-identical (md5-gated) through +"
                        f"{CK_WASH[-1]}"}
    for step in range(1, CK_WASH[-1] + 1):
        hs = {t: arms[t]["wash"]["x_hashes"].get(step) for t in arms}
        same = all(h is not None for h in hs.values()) and \
            len(set(hs.values())) == 1
        G_INPUTS["per_step"][step] = {t: hs[t] for t in arms}
        G_INPUTS["per_step"][step]["identical"] = bool(same)
    G_INPUTS["pass"] = bool(all(v["identical"]
                                for v in G_INPUTS["per_step"].values()))
    assert G_INPUTS["pass"], "input streams diverged across arms"
    log(f"G_INPUTS: per-step inputs bit-identical across all {len(arms)} "
        f"arms through +{CK_WASH[-1]}: PASS")

    G_PIN = {"per_arm": {}, "fuzz_formula": "2 * lr * sqrt(n_walled)",
             "pass": None}
    for tag in ("WQ", "WQE", "STW"):
        R = arms[tag]["wall"]["R"]
        n_w = arms[tag]["wall"]["n_walled"]
        fuzz = 2.0 * G3.WASH_LR * (n_w ** 0.5)
        rows = [t for t in arms[tag]["wash"]["traj"]
                if "disp_wall_raw" in t]
        mx_raw = max(r["disp_wall_raw"] for r in rows) if rows else None
        mx_set = max(r["disp_wall_settled"] for r in rows) if rows else None
        G_PIN["per_arm"][tag] = {
            "R": R, "n_walled": n_w, "fuzz": fuzz,
            "bound": R + fuzz, "max_raw_disp": mx_raw,
            "max_settled_disp": mx_set,
            "per_ckpt": {r["step"]: {"raw": r["disp_wall_raw"],
                                     "settled": r["disp_wall_settled"]}
                         for r in rows},
            "pass": bool(mx_raw is not None and mx_raw <= R + fuzz
                         and mx_set <= R + PIN_SETTLE_TOL)}
        log(f"G_PIN[{tag}]: max raw |d_walled| {mx_raw:.4f} <= "
            f"{R + fuzz:.4f}, settled {mx_set:.6f} <= {R}: "
            f"{'PASS' if G_PIN['per_arm'][tag]['pass'] else 'FAIL'}")
    G_PIN["pass"] = bool(all(v["pass"] for v in G_PIN["per_arm"].values()))
    gates_pass = bool(G_ROOT["pass"] and G_BITEXACT["pass"]
                      and G_ANCHOR["pass"] and G_POOL["pass"]
                      and G_SPLICE["pass"] and G_NAMEFREE["pass"]
                      and G_INPUTS["pass"] and G_PIN["pass"]
                      and all(v["pass"] for v in G_WALLROOT.values()))

    # ---- the traces (checkpoint rows)
    traces = {}
    for tag in arms:
        rows = [{"step": 0, "g0": root_dial["base"][0]["mean_pz"],
                 "gm12": root_dial["base"][-12]["mean_pz"],
                 "ce_r": root_dial["ce_r"], "cum_disp": 0.0,
                 "a_fact_live": None, "g0_store_into_root_host": None}]
        for t in arms[tag]["wash"]["traj"]:
            if "g0_mean_pz" in t:
                rows.append({
                    "step": t["step"], "g0": t["g0_mean_pz"],
                    "gm12": t["g_m12_mean_pz"], "ce_r": t["ce_r"],
                    "cum_disp": t["cum_disp"],
                    "disp_store": t.get("disp_store"),
                    "disp_host": t.get("disp_host"),
                    "disp_wall_raw": t.get("disp_wall_raw"),
                    "disp_wall_settled": t.get("disp_wall_settled"),
                    "a_fact_live": t["a_fact_live"],
                    "argmax_live": t["argmax_live"],
                    "g0_store_into_root_host": t["g0_store_into_root_host"]})
        traces[tag] = rows
        log(f"{tag} g0 trace: " + " ".join(f"+{r['step']}:{r['g0']:.4f}"
                                           for r in rows))
        log(f"{tag} a_fact(live): "
            + " ".join(f"+{r['step']}:"
                       f"{r['a_fact_live'] if r['a_fact_live'] is None else round(r['a_fact_live'], 4)}"
                       for r in rows))
        log(f"{tag} store->root-host: "
            + " ".join(f"+{r['step']}:"
                       f"{r['g0_store_into_root_host'] if r['g0_store_into_root_host'] is None else round(r['g0_store_into_root_host'], 4)}"
                       for r in rows))

    # =====================================================================
    # THE CENSUS AFTER WASH (g3's census + closure, extended attribution)
    # =====================================================================
    skeys = G3.store_keys("gen")
    census_pts = ([("WQ", 1), ("WQ", 50), ("WQ", 300), ("FREE", 1),
                   ("FREE", 300), ("WQE", 300), ("STW", 300)]
                  if not SMOKE else [("WQ", CK_WASH[-1])])
    census_out: dict = {}
    for tag, s in census_pts:
        if s not in arms[tag]["wash"]["sds"]:
            continue
        sd_t = arms[tag]["wash"]["sds"][s]
        log("=" * 78)
        log(f"CENSUS {tag} at +{s}")
        cen = census_g5(theta0, sd_t, f"{tag}-t{s}", pool_band_x,
                        anchor_neutral)
        per_tensor = {k: float((sd_t[k].float() - theta0[k].float()).norm())
                      for k in skeys}
        per_tensor["store_total"] = float(sum(v ** 2 for v
                                              in per_tensor.values()) ** 0.5)
        cen["per_tensor_displacement"] = per_tensor
        log(f"  live retrieval: a_fact {cen['washed']['a_fact']:.4f} argmax "
            f"{cen['washed']['argmax_correct']:.4f} | inj gain "
            f"{cen['inj_gain_ratio']:.4f} | attribution: washed-q x root-K "
            f"{cen['washed_q__root_K']['a_fact']:.4f} | WALLSTATE-q (root "
            f"W_q in washed host) x root-K "
            f"{cen['wallstate_q__root_K']['a_fact']:.4f} | WQDRIFT-q (washed "
            f"W_q in root host) x root-K "
            f"{cen['wqdrift_q__root_K']['a_fact']:.4f}")
        log("  per-tensor displacement: "
            + " ".join(f"{G3.tensor_short(k) if k != 'store_total' else k} "
                       f"{v:.4f}" for k, v in per_tensor.items()))
        # the closure partition (g3's 2x2 at this state, settled)
        closure = {}
        sd_byp, g_byp = G3.restore_class(sd_t, theta0, skeys, "store")
        closure["bypass_root_store_into_washed_host"] = {
            "gate": g_byp,
            "g0": G3.battery_cell(G3.evl_load("gen", sd_byp), ids130,
                                  zid)["mean_pz"]}
        sd_comp, g_comp = G3.restore_class(theta0, sd_t, skeys, "store")
        closure["washed_store_into_root_host"] = {
            "gate": g_comp,
            "g0": G3.battery_cell(G3.evl_load("gen", sd_comp), ids130,
                                  zid)["mean_pz"]}
        sd_all, g_all = G3.restore_class(sd_t, theta0, list(sd_t.keys()),
                                         "all")
        g0_all = G3.battery_cell(G3.evl_load("gen", sd_all), ids130,
                                 zid)["mean_pz"]
        g0_root = root_dial["base"][0]["mean_pz"]
        closure["all_restored_sanity"] = {
            "gate": g_all, "g0": g0_all, "root_g0": g0_root,
            "pass": bool(abs(g0_all - g0_root) < G3.G_BIT_TOL)}
        assert closure["all_restored_sanity"]["pass"], \
            "all-restored is not bit-exact with root"
        site = classify_kill_site(cen,
                                  closure["bypass_root_store_into_washed_host"]["g0"])
        closure["verdict"] = {
            "bypass_restores": bool(
                closure["bypass_root_store_into_washed_host"]["g0"]
                >= RESTORE_BAR),
            "reading": (f"bypass (root store into this host) restores g0 to "
                        f"{closure['bypass_root_store_into_washed_host']['g0']:.4f}"
                        f" — the route is intact" if
                        closure["bypass_root_store_into_washed_host"]["g0"]
                        >= RESTORE_BAR else
                        f"bypass restores only "
                        f"{closure['bypass_root_store_into_washed_host']['g0']:.4f} "
                        f"(< {RESTORE_BAR}) — ROUTE-KILL (the host's "
                        "expression path is dead at this state)")}
        log(f"  closure: bypass {closure['bypass_root_store_into_washed_host']['g0']:.5f} "
            f"| store-into-root-host "
            f"{closure['washed_store_into_root_host']['g0']:.4f} | "
            f"all-restored {g0_all:.6f} vs root {g0_root:.6f}")
        log(f"  KILL SITE: {site['site']} (retrieval intact "
            f"{site['retrieval_intact']})")
        census_out[f"{tag}@+{s}"] = {"census": cen, "closure": closure,
                                     "site": site}

    # =====================================================================
    # ADJUDICATION (the registered bars; order: GATES -> CONE-WALL-HOLDS ->
    # TAX -> co-reports; no shopping)
    # =====================================================================
    def ck_rows(tag):
        return [r for r in traces[tag] if r["step"] > 0]

    g0_wq = {r["step"]: r["g0"] for r in ck_rows("WQ")}
    CONE_WALL_HOLDS = bool(g0_wq and all(v >= SURVIVE_BAR
                                         for v in g0_wq.values()))
    min_wq = min(g0_wq.values()) if g0_wq else None
    first_under = next((s for s in sorted(g0_wq) if g0_wq[s] <= SHUT_BAR),
                       None)

    STORE_SURVIVES, QUERY_CONE_HELD = {}, {}
    for tag in arms:
        rows = ck_rows(tag)
        sh = [r["g0_store_into_root_host"] for r in rows]
        af = [(r["a_fact_live"], r["argmax_live"]) for r in rows]
        STORE_SURVIVES[tag] = bool(sh and all(v >= SURVIVE_BAR for v in sh))
        QUERY_CONE_HELD[tag] = bool(
            af and all(a >= A_FACT_BAR and m >= ARGMAX_BAR for a, m in af))

    ce_r_wq = next(r["ce_r"] for r in ck_rows("WQ") if r["step"] == 300) \
        if any(r["step"] == 300 for r in ck_rows("WQ")) else None
    ce_r_free = next(r["ce_r"] for r in ck_rows("FREE") if r["step"] == 300) \
        if any(r["step"] == 300 for r in ck_rows("FREE")) else None
    tax_delta = (ce_r_wq - ce_r_free) if None not in (ce_r_wq, ce_r_free) \
        else None
    TAX_NEAR_ZERO = bool(tax_delta is not None and abs(tax_delta) < TAX_BAR)

    cone_bracket = None
    if STORE_SURVIVES.get("WQ") and not STORE_SURVIVES.get("WQE"):
        cone_bracket = (R_Q, R_QE)

    falsifier_fires = not CONE_WALL_HOLDS

    fmt = lambda v: "n/a" if v is None else f"{v:.4f}"
    tax_str = "n/a" if tax_delta is None else f"{tax_delta:+.4f}"
    wq_cen = census_out.get("WQ@+300", {})
    wq_cen1 = census_out.get("WQ@+1", {})
    wsq1 = ((wq_cen1.get("census", {})
             .get("wallstate_q__root_K", {}) or {})
            .get("a_fact"))
    wqd1 = ((wq_cen1.get("census", {})
             .get("wqdrift_q__root_K", {}) or {})
            .get("a_fact"))
    bypass_300 = (wq_cen.get("closure", {})
                  .get("bypass_root_store_into_washed_host", {})
                  .get("g0"))
    if not gates_pass:
        failed = [k for k, g in (("G_ROOT", G_ROOT), ("G_BITEXACT", G_BITEXACT),
                                 ("G_ANCHOR", G_ANCHOR), ("G_POOL", G_POOL),
                                 ("G_SPLICE", G_SPLICE),
                                 ("G_NAMEFREE", G_NAMEFREE),
                                 ("G_INPUTS", G_INPUTS), ("G_PIN", G_PIN))
                  if not g["pass"]] + \
                 [f"G_WALLROOT[{t}]" for t, g in G_WALLROOT.items()
                  if not g["pass"]]
        verdict = f"TEXTURE (GATE FAILURE: {', '.join(failed)})"
        clause = ("a registered gate failed — nothing adjudicated per the "
                  f"arc's abort clause; failed: {failed}; the full record "
                  "is reported.")
    elif CONE_WALL_HOLDS:
        verdict = "CONE-WALL-HOLDS"
        clause = (f"the composed system works: with W_q walled the store "
                  f"survived the wash — g0 >= {SURVIVE_BAR} at EVERY "
                  f"checkpoint through +{CK_WASH[-1]} (min {fmt(min_wq)}); "
                  f"the adaptation tax "
                  f"{'near-zero' if TAX_NEAR_ZERO else 'NOT near-zero'} "
                  f"(dCE_R@300 {tax_str}); the g-series' first working "
                  "memory architecture (store + query wall).")
    else:
        verdict = "FALSIFIER FIRES — THE KILL IS NOT CONFINED TO W_q"
        clause = (
            f"the wall on W_q fails to maintain: WQ's g0 fell to "
            f"{fmt(min_wq)} (first checkpoint <= {SHUT_BAR}: "
            f"+{first_under}) — the kill is not confined to W_q at this "
            "substrate. The census decomposes it into three findings: "
            "(1) THE STORE SURVIVES AT THE ORGAN LEVEL — with W_q walled "
            f"(R {R_Q}) the settled store read into a ROOT host gives "
            + " ".join(f"+{r['step']}:{r['g0_store_into_root_host']:.3f}"
                       for r in ck_rows("WQ"))
            + (f" (STW, the whole store walled at R {R_ST}: "
               + " ".join(f"+{r['step']}:{r['g0_store_into_root_host']:.3f}"
                          for r in ck_rows("STW")) + ")")
            + " — the wall's product is real and the patterns are intact "
            "(keys hold; retrieval against root K intact); "
            "(2) THE QUERY CONE IS EXITED THROUGH THE HOST — with W_q "
            "pinned, the LIVE retrieval (through the arm's own washed "
            "host) reads a_fact "
            + " ".join(f"+{r['step']}:{r['a_fact_live']:.3f}"
                       for r in ck_rows("WQ")[:4])
            + f" (vs root 0.907; the +1 census attribution: WALLSTATE-q "
            f"(root W_q in the washed host) x root-K a_fact {fmt(wsq1)} "
            f"while WQDRIFT-q (washed W_q in the root host) reads "
            f"{fmt(wqd1)}) — q = W_q . LN(h): the wash moves h, and the "
            "cone is a property of the composed query path, not of W_q's "
            "weights; "
            "(3) THE ROUTE DIES INDEPENDENTLY — a PERFECT root store into "
            "the washed host restores g0 to only "
            f"{fmt(bypass_300)} at +300 — the host's expression path is a "
            "second unwalled kill site. The cheap wall is mechanically "
            "sound (G-PIN), the adaptation tax is "
            + ("near-zero as predicted" if TAX_NEAR_ZERO
               else "NOT near-zero")
            + f" (dCE_R@300 {tax_str}) — but the composed system (store + "
            "query wall) does NOT maintain: the wall that can hold this "
            "memory must cover the host (g1b's whole-net wall, at its "
            "tax)."
            + (f" The cone bracket: WQ stores, WQE does not => the cone "
               f"radius lives in ({R_Q}, {R_QE}]."
               if cone_bracket else ""))

    log("=" * 78)
    log(f"G5 VERDICT: {verdict}")
    for tag in arms:
        log(f"  {tag}: g0 " + " ".join(f"+{r['step']}:{r['g0']:.4f}"
                                       for r in ck_rows(tag))
            + f" | STORE-SURVIVES {STORE_SURVIVES[tag]} | CONE-HELD "
            f"{QUERY_CONE_HELD[tag]}")
    log(f"  TAX: dCE_R(WQ-FREE)@300 {tax_delta if tax_delta is None else round(tax_delta, 4)} "
        f"(near-zero: {TAX_NEAR_ZERO})")
    log(f"  {clause}")
    log("=" * 78)

    adjudication = {
        "order": "GATES -> CONE-WALL-HOLDS -> TAX -> co-reports (frozen)",
        "gates_pass": gates_pass,
        "CONE_WALL_HOLDS": CONE_WALL_HOLDS,
        "WQ_g0_min": min_wq, "WQ_first_under_shut": first_under,
        "STORE_SURVIVES": STORE_SURVIVES,
        "QUERY_CONE_HELD": QUERY_CONE_HELD,
        "tax": {"ce_r_wq_300": ce_r_wq, "ce_r_free_300": ce_r_free,
                "delta": tax_delta, "TAX_NEAR_ZERO": TAX_NEAR_ZERO},
        "cone_bracket": cone_bracket,
        "falsifier_fires": falsifier_fires,
        "first_working_memory_architecture": bool(CONE_WALL_HOLDS
                                                  and gates_pass),
        "verdict": verdict, "clause": clause,
    }

    # =====================================================================
    # OUTPUTS
    # =====================================================================
    metrics = {
        "experiment": "g5_cone_wall",
        "date": common.now_iso(),
        "purpose": ("WALL THE QUERY CONE — the composition g3's data named "
                    "and g1b's machinery enables: the Hopfield organ + "
                    "commit-and-project on the QUERY PROJECTION alone (the "
                    "much cheaper well), adjudicated against the registered "
                    "CONE-WALL-HOLDS bar with the census after wash, the "
                    "adaptation tax, and the falsifier's kill attribution"),
        "registered_prediction": REGISTERED_PREDICTION,
        "preflight": PREFLIGHT,
        "host": {"base": f"runs/checkpoints/{BASE_CK}",
                 "cfg": {"n_layer": 4, "n_head": 4, "n_embd": 128,
                         "block_size": 512, "params": G3.F2_PARAMS},
                 "ce_r_base": ce_base, "ce_budget": ce_budget},
        "store": {"n_params": G3.STORE_PARAMS, "beta": G3.BETA,
                  "d_key": G3.D_KEY, "n_patterns": G3.N_PAT,
                  "null_idx": G3.NULL_IDX,
                  "placement": "after block 3, before ln_f (g3 VERBATIM)"},
        "root": {"checkpoint": f"runs/checkpoints/{ROOT_CK}",
                 "bitexact_gate": G_BITEXACT, "n_params": n_params,
                 "dial": {k: v for k, v in root_dial.items() if k != "tag"},
                 "G_ROOT": G_ROOT},
        "arms": {tag: {
            "desc": arms[tag]["desc"],
            "wall": arms[tag]["wall"],
            "recipe": "g3's stage-A wash VERBATIM (e176N arm A via e157: "
                      "neutral bank seed 170, batch 16+16 full-token CE, "
                      "AdamW (0.9,0.95) wd 0.1 lr 1e-3 clip 1.0, seed "
                      "10902), the wall's keyset/radius the only delta",
            "traj": arms[tag]["wash"]["traj"],
            "trace": traces[tag],
            "steps_ran": arms[tag]["wash"]["steps_ran"],
            "device": arms[tag]["wash"]["device"],
            "x_hashes": arms[tag]["wash"]["x_hashes"],
            "dials": {s: {k: v for k, v in d.items() if k != "tag"}
                      for s, d in arms[tag]["dials"].items()},
        } for tag in arms},
        "census_after_wash": census_out,
        "gates": {"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                  "G_POOL": G_POOL, "G_ANCHOR": G_ANCHOR,
                  "G_BITEXACT": G_BITEXACT, "G_ROOT": G_ROOT,
                  "G_WALLROOT": G_WALLROOT,
                  "G_INPUTS": {k: v for k, v in G_INPUTS.items()
                               if k != "per_step"} | {"per_step": {
                                   s: v["identical"] for s, v in
                                   G_INPUTS["per_step"].items()}},
                  "G_PIN": G_PIN},
        "adjudication": adjudication,
        "checkpoints": CKPT_INVENTORY,
        "trims": trims,
        "deviations": deviations,
        "device_events": G3.device_events,
        "timing": {"total_s": round(time.time() - T0, 1)},
        "smoke": SMOKE, "threads": torch.get_num_threads(),
    }
    save_json(rd / "metrics.json", E43.jsonable(metrics))
    log("metrics.json written")
    plot(rd / "cone_wall.png", metrics)
    log(f"plot written -> {rd}")
    return 0


# ------------------------------------------------------------------ plot

def plot(path: Path, M: dict):
    A = M["arms"]
    adj = M["adjudication"]
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.5))
    cols = {"FREE": "crimson", "WQ": "seagreen", "WQE": "darkorange",
            "STW": "royalblue"}
    lbls = {"FREE": "FREE — no wall (control)",
            "WQ": f"WQ — wall W_q only, R={R_Q}",
            "WQE": f"WQE — wall W_q only, R={R_QE} (knife edge)",
            "STW": f"STW — wall whole store, R={R_ST}"}

    # (0,0) THE WASH — the primary bar
    ax = axes[0, 0]
    for tag in ("FREE", "WQ", "WQE", "STW"):
        t = [r["step"] for r in A[tag]["trace"]]
        g = [r["g0"] for r in A[tag]["trace"]]
        ax.plot(t, g, "o-", color=cols[tag], lw=2.2, ms=5,
                label=lbls[tag])
    ax.axhline(SURVIVE_BAR, color="seagreen", ls="--", lw=1.2,
               label=f"survive bar {SURVIVE_BAR}")
    ax.axhline(SHUT_BAR, color="gray", ls=":", lw=1.1,
               label=f"dissolve bar {SHUT_BAR}")
    ax.set_xscale("symlog", linthresh=4)
    ax.set_xlabel("neutral-wash steps (e176n arm A verbatim, seed 10902)")
    ax.set_ylabel("install-60 battery p(Z) (g0)")
    ax.set_ylim(-0.03, 1.02)
    ax.set_title(f"THE WASH — CONE-WALL-HOLDS: {adj['CONE_WALL_HOLDS']}\n"
                 f"(the registered bar: WQ g0 >= 0.50 at every checkpoint)",
                 fontsize=9)
    ax.legend(fontsize=7)

    # (0,1) THE LIVE QUERY CONE — a_fact through each arm's own host
    ax = axes[0, 1]
    for tag in ("FREE", "WQ", "WQE", "STW"):
        rows = [r for r in A[tag]["trace"] if r["a_fact_live"] is not None]
        ax.plot([r["step"] for r in rows], [r["a_fact_live"] for r in rows],
                "o-", color=cols[tag], lw=2.0, ms=5, label=lbls[tag])
    ax.axhline(A_FACT_BAR, color="gray", ls=":", lw=1.1,
               label=f"a_fact bar {A_FACT_BAR}")
    pf = PREFLIGHT["live_query_root_Wq_in_washed_host"]
    ax.plot([1, 2, 4, 10, 50, 100, 200, 300],
            [pf["+1"], pf["+2"], pf["+4"], pf["+10"], pf["+50"],
             pf["+100"], pf["+200"], pf["+300"]],
            "kx--", lw=1.2, ms=7, alpha=0.7,
            label="preflight: root W_q in washed host (static)")
    ax.set_xscale("symlog", linthresh=4)
    ax.set_xlabel("wash step")
    ax.set_ylabel("live retrieval a_fact (fact rows, own host)")
    ax.set_ylim(-0.03, 1.02)
    ax.set_title("THE LIVE QUERY CONE — is it held with W_q walled?\n"
                 "q = W_q . LN(h): the wash moves h (the cone is exited "
                 "through the host)", fontsize=9)
    ax.legend(fontsize=7)

    # (0,2) THE STORE SURVIVES — organ-level probe
    ax = axes[0, 2]
    for tag in ("FREE", "WQ", "WQE", "STW"):
        rows = [r for r in A[tag]["trace"]
                if r["g0_store_into_root_host"] is not None]
        ax.plot([r["step"] for r in rows],
                [r["g0_store_into_root_host"] for r in rows],
                "o-", color=cols[tag], lw=2.0, ms=5, label=lbls[tag])
    ax.axhline(A["FREE"]["trace"][0]["g0"], color="k", ls=":", lw=1.0)
    ax.annotate(f"root {A['FREE']['trace'][0]['g0']:.3f}",
                (0.02, A["FREE"]["trace"][0]["g0"]), xycoords="axes fraction",
                fontsize=7.5)
    ax.axhline(SURVIVE_BAR, color="seagreen", ls="--", lw=1.1)
    ax.set_xscale("symlog", linthresh=4)
    ax.set_xlabel("wash step")
    ax.set_ylabel("g0 of the arm's settled store into the ROOT host")
    ax.set_ylim(-0.03, 1.02)
    ax.set_title("STORE-SURVIVES (organ level) — the wall's product\n"
                 "(the store read where the host is healthy; the "
                 "V/W_o erosion vs the pinned STW)", fontsize=9)
    ax.legend(fontsize=7)

    # (1,0) THE R LADDER at +300
    ax = axes[1, 0]
    Rs, vals, names = [], [], []
    for tag, R in (("FREE", 0.0), ("WQ", R_Q), ("WQE", R_QE)):
        v = next((r["g0_store_into_root_host"] for r in A[tag]["trace"]
                  if r["step"] == CK_WASH[-1]
                  and r["g0_store_into_root_host"] is not None), None)
        if v is not None:
            Rs.append(R)
            vals.append(v)
            names.append(tag)
    ax.plot(Rs, vals, "s-", ms=10, lw=2.0, color="purple",
            label="store-into-root-host g0 at +300 (W_q wall)")
    if adj["cone_bracket"]:
        ax.axvspan(adj["cone_bracket"][0], adj["cone_bracket"][1],
                   color="gold", alpha=0.25,
                   label=f"cone radius bracket {adj['cone_bracket']}")
    stw_v = next((r["g0_store_into_root_host"] for r in A["STW"]["trace"]
                  if r["step"] == CK_WASH[-1]
                  and r["g0_store_into_root_host"] is not None), None)
    if stw_v is not None:
        ax.plot([0.0], [stw_v], "D", ms=10, color=cols["STW"],
                label=f"STW (whole store, R={R_ST}): {stw_v:.3f}")
    ax.axhline(SURVIVE_BAR, color="seagreen", ls="--", lw=1.1)
    ax.set_xticks(Rs)
    ax.set_xticklabels(names)
    ax.set_xlabel("the wall dial (W_q-subspace radius; FREE = no wall)")
    ax.set_ylabel("organ-level survival at +300")
    ax.set_ylim(-0.03, 1.02)
    ax.set_title("THE R-DIAL: the wall re-measures the cone radius\n"
                 "(dynamically, at the organ level)", fontsize=9)
    ax.legend(fontsize=7)

    # (1,1) THE TAX
    ax = axes[1, 1]
    for tag in ("FREE", "WQ", "WQE", "STW"):
        rows = [r for r in A[tag]["traj"]]
        ax.plot([r["step"] for r in rows], [r["ce_batch"] for r in rows],
                "-", lw=1.4, color=cols[tag], alpha=0.8, label=lbls[tag])
    tax = adj["tax"]
    ax.annotate(f"TAX dCE_R(WQ-FREE)@300 "
                f"{tax['delta'] if tax['delta'] is None else round(tax['delta'], 4)} "
                f"(near-zero: {tax['TAX_NEAR_ZERO']}; bar {TAX_BAR})",
                (0.03, 0.05), xycoords="axes fraction", fontsize=8,
                weight="bold",
                color="seagreen" if tax["TAX_NEAR_ZERO"] else "darkred")
    ax.set_xlabel("wash step")
    ax.set_ylabel("in-batch corpus CE (the wash's own adaptation)")
    ax.legend(fontsize=7)
    ax.set_title("THE ADAPTATION TAX — the rest of the net stays free\n"
                 "(the walled 8k/17k params do not impede the stream's "
                 "learning)", fontsize=9)

    # (1,2) THE VERDICT PANEL
    ax = axes[1, 2]
    ax.axis("off")
    import textwrap
    y = 0.97
    ax.text(0.02, y, "G5 — WALL THE QUERY CONE", fontsize=11, va="top",
            family="monospace", weight="bold")
    y -= 0.052
    for tag in ("FREE", "WQ", "WQE", "STW"):
        seq = " ".join(f"+{r['step']}:{r['g0']:.3f}"
                       for r in A[tag]["trace"] if r["step"] > 0)
        ax.text(0.02, y, f"  {tag} g0: {seq}", fontsize=6.8, va="top",
                family="monospace", color=cols[tag])
        y -= 0.028
    y -= 0.006
    for tag in ("WQ", "STW"):
        seq = " ".join(f"+{r['step']}:{r['g0_store_into_root_host']:.3f}"
                       for r in A[tag]["trace"]
                       if r["g0_store_into_root_host"] is not None)
        ax.text(0.02, y, f"  {tag} store->root-host: {seq}", fontsize=6.8,
                va="top", family="monospace", color=cols[tag])
        y -= 0.028
    y -= 0.006
    ax.text(0.02, y, f"  STORE-SURVIVES: {adj['STORE_SURVIVES']} | "
            f"QUERY-CONE-HELD: {adj['QUERY_CONE_HELD']} | cone bracket: "
            f"{adj['cone_bracket']}", fontsize=7.0, va="top",
            family="monospace")
    y -= 0.030
    ax.text(0.02, y, f"  TAX: {tax['delta'] if tax['delta'] is None else round(tax['delta'], 4)} "
            f"(near-zero {tax['TAX_NEAR_ZERO']}) | falsifier fires: "
            f"{adj['falsifier_fires']}", fontsize=7.0, va="top",
            family="monospace")
    y -= 0.044
    ax.text(0.02, y, f"VERDICT: {adj['verdict']}", fontsize=9.0, va="top",
            family="monospace", weight="bold", color="darkred")
    y -= 0.040
    for wd in textwrap.wrap(adj["clause"], width=92, break_long_words=False):
        ax.text(0.02, y, f"  {wd}", fontsize=6.4, va="top",
                family="monospace")
        y -= 0.024

    fig.suptitle("G5 — WALL THE QUERY CONE: g3's Hopfield store + g1b's "
                 "commit-and-project on the query projection alone -> "
                 f"{adj['verdict']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
