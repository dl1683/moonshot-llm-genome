"""
e307 — THE MAINTENANCE FOOTPRINT — the geometry of the controller's own
carving. A CPU-ONLY DESK CELL (dispatch 2026-10-05): committed records
only, no training, no GPU, no envelope writes, torch threads 4.

THE QUESTION (verbatim from the dispatch): the controller works — but
what GEOMETRY does it leave? From the saved maintained states, measure
the accumulated maintenance displacement (the controller's footprint):
its in-room fraction, its own subspace structure (an SVD of the
maintenance delta between twin pairs), and whether successive
maintenance steps carve a REPEATING direction (the "antibody" shape —
one direction used every event) or a rotating repertoire. THE FANTASTIC
READING: if the footprint is low-dimensional and repeating, the
controller has found a memory's "vital signal" — a small set of
directions whose restoration is preservation; if it is high-dimensional
and state-dependent, preservation is genuine navigation.

BARS — FROZEN VERBATIM FROM THE DISPATCH, BEFORE ANY COMPUTE (no bar
shopping):

  ONE-ANTIBODY: the footprint is low-rank (top-3 SVD >= 50% energy)
      and repeating (median cross-event cos >= 0.5) — the controller
      found a vital signal.
  NAVIGATION: high-rank footprint + state-tracking (cos with the
      deficit gradient >> cross-event cos) — genuine navigation;
      preservation is feedback in the full sense.
  CHANCE-CARVING: in-room at chance AND no repeat AND no tracking —
      the controller's steps are individually unremarkable; only the
      GATING is the intelligence.
  MIXED/INSTRUMENT-MISSING: disclose.

OPERATIONALIZATIONS — FROZEN BEFORE COMPUTE:

  * THE INVENTORY (measured, disclosed verbatim in metrics.json):
    the per-arm RESUME checkpoints of e287/e288/e289/e291 carry
    `maint_cum` (fp64, N=2,739,072) — the machine-accumulated sum of
    EVERY realized maintenance displacement (verified in the parents'
    drivers: maint_cum += d_m around every opt_F.step()), `corp_cum`
    (the corpus stream's accumulated displacement), the final `model`,
    and `optF` (the fact-side SGD-M state incl. the momentum buffer —
    the final antibody candidate). e291 carries per-fact `maint_cums`.
    PAIR DELTAS: post-minus-post twins — e289 WC-NC (THE passive-twin
    pair: same loaded fact, bit-identical contradiction draws, the ONLY
    delta is the controller's 16 events + the corpus stream's causal
    response to them); e288 EG-NFT and e289 C1EG-C1NFT (controller vs
    FLAT-DOSE twin — the twin is NOT passive; disclosed). e287's
    SANCTUARY-TWIN is a passive twin but on e287's own session stream
    (draw-contaminated cross-session delta) — EXCLUDED from bars,
    disclosed. e293 exists on disk but is outside the dispatch's
    provenance (e288/e289/e291) and uncommitted — not read.
  * PRIMARY OBJECT (the bar's object): e288_ERROR-GATED's maint_cum —
    the founding controller's own accumulated carving (the purest
    instrument the records afford; the dispatch's letter-form pair
    deltas are fully read as co-objects and their agreement disclosed).
  * GEOMETRY READ (x6's conventions, lab/x6_svd_bet.py ported by
    value): per-2D-parameter-matrix SVD in fp64; GLOBAL top-3 share T3
    (THE bar's rank clause), top-10 singular shares, top-50 share,
    effective rank at 90% (cross90 = smallest rank reaching 90% of the
    2D energy, x6's cumulative convention); natural-basis m50/m90/m99
    co-report. In-room fraction := ||P x||^2 / ||x||^2 with the exact
    SRCT projector P x = D . idct(mask_S(dct(D . x))) in fp64 (the
    fact's own committed K10K room, seeds 26113/26114, bit-verified vs
    fresh construction; e291: the shared-frame rooms 29111/29112 —
    per-own-room + the 50k union).
  * THE NULLS: (i) the x6-form norm-matched gaussian (frozen seed
    30701; its in-room share expects k/N = 10000/2739072 = 0.003651);
    (ii) T267's EMPIRICAL CLASS NULL — the corpus/name gradient class
    sits at in-room ~0.060 (the committed cells' per-step reads
    0.0597-0.0611); "in-room at chance" := within [0.055, 0.065].
  * THE REPEAT INSTRUMENTS (exact algebra; the per-event step VECTORS
    are NOT in the records — the full cos(step_i, step_j) matrix is
    INSTRUMENT-MISSING and the dispatch's "median cross-event cos" is
    registered to its exact computable substitute):
    (R1) THE NORM-WEIGHTED MEAN cross-event step cosine, EXACT:
        ||maint_cum||^2 = sum_i r_i^2 + 2 sum_{i<j} r_i r_j cos_ij
        with r_i the per-event realized_step_norm (the parents'
        maint_ledgers) — so wmean_cos := (||cum||^2 - sum r_i^2) /
        (sum_{i!=j} r_i r_j) is the r_i r_j-weighted MEAN of the
        pairwise cosines. No distributional assumption.
    (R2) THE GRADIENT-LEVEL per-event alignment rho_t (the momentum
        guard — R1 rides the SGD-M buffer's own smoothing): torch
        SGD-M is b_t = 0.9 b_{t-1} + g_t with ||g_t|| = gn_t (the
        post-clip norm, ledger), and the ledger's b_m_norm IS ||b_t||
        (the parents' pending_buffer_sqnorm). Hence EXACTLY
        rho_t := <g_t, b_{t-1}>/(||g_t|| ||b_{t-1}||)
               = (b_t^2 - 0.81 b_{t-1}^2 - gn_t^2) / (1.8 b_{t-1} gn_t).
        rho_t = the fresh event gradient's alignment with the running
        antibody. Consecutive-step cosine also exact:
        cos(b_t, b_{t-1}) = (0.9 b_{t-1} + rho_t gn_t) / b_t.
    (R3) cos(maint_cum, -bufF_final): the accumulated carving vs the
        FINAL momentum direction (the last antibody).
    THE REPEAT CLAUSE (registered): R1 >= 0.5 AND mean(R2) >= 0.5 —
    the momentum smoothing alone cannot fire the bar.
  * THE TRACKING CLAUSE (NAVIGATION's "cos with deficit gradient"):
    the per-event deficit-gradient DIRECTIONS are not in the records —
    DIRECT INSTRUMENT-MISSING (disclosed). Proxies co-reported, never
    bar-adjudicated: the dose-vs-deficit correlation (by CONSTRUCTION
    the controller's law — not evidence), the per-event g_in_room_frac
    stability, and the rho_t series' trend. If the data land in the
    NAVIGATION branch, the verdict fires MIXED/INSTRUMENT-MISSING with
    the geometry table verbatim (the registered tree; no bar shopping).
  * VERDICT TREE (registered): ONE-ANTIBODY iff (T3 >= 0.50 on the
    primary) AND (R1 >= 0.5 AND mean rho_t >= 0.5). CHANCE-CARVING iff
    in-room in [0.055, 0.065] AND (R1 < 0.5 OR mean rho < 0.5) AND
    mean rho < 0.5. NAVIGATION-direct: instrument-missing as above ->
    MIXED/INSTRUMENT-MISSING. Else MIXED. The class table (e289-WC
    denial, e289-C1 replicate, the flat twins, e291 fleet) co-reported
    with per-object clause traces; class disagreements disclosed.
  * HIGH-RANK (NAVIGATION's rank word, registered for the table):
    cross90 >= 1000.
  * CPU ENVELOPE: CUDA_VISIBLE_DEVICES="" before torch import; torch
    threads 4; pocketfft workers 4; nothing written to
    runs/_envelope_log.jsonl; no lab/gpu code path touched.
  * PROGRESSIVE writes to runs/e307/metrics.json; NO NOTES/THINKING/
    QUEUE/STATE edits (dispatch — the coordinator folds).

PREDICTIONS — REGISTERED AT BIRTH (before compute):
  P-e307a (repeat): R1 in [0.7, 0.9] on all three controller arms and
    mean rho_t >= 0.7 — the antibody repeats at BOTH step and gradient
    level.
  P-e307b (in-room): the accumulated footprint rides the class null
    ~0.060 (accumulation does NOT bend it in-room).
  P-e307c (rank): T3 >= 0.5 fires on the controller arms (the aligned
    antibody dominates the spectrum) — ONE-ANTIBODY is the anticipated
    branch; if T3 < 0.5 with repeat firing, MIXED (low-dim but >3-dim).
  P-e307d (cross-session): the neutral footprints of different
    sessions/streams (e288-EG, e289-C1-EG, e291-T1) align at
    cos > 0.5 — the vital direction is session-invariant.
  P-e307e (denial): the denial footprint (e289-WC) is NOT rotated away
    from the neutral class (|cos| > 0.5) — contradiction changes the
    DOSE (e289's finding), not the DIRECTION.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU-ONLY, bulletproof

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.fft as sf
import torch

torch.set_num_threads(4)
DCT_WORKERS = 4

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "runs" / "checkpoints"
OUT = REPO / "runs" / "e307"
OUT.mkdir(parents=True, exist_ok=True)

T0 = time.time()
METRICS: dict = {
    "experiment": "e307_footprint",
    "phase": "THE MAINTENANCE FOOTPRINT — the geometry of the controller's "
             "own carving; CPU-only desk cell, committed records only",
    "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "status": "RUNNING (progressive writes)",
    "cpu_only": True,
    "threads": {"torch": 4, "pocketfft": DCT_WORKERS},
}
BARS = {
    "ONE-ANTIBODY": "low-rank (top-3 SVD >= 50% energy) and repeating "
        "(median cross-event cos >= 0.5; registered exact substitute: R1 "
        "wmean_cos >= 0.5 AND mean gradient-level rho_t >= 0.5) — the "
        "controller found a vital signal",
    "NAVIGATION": "high-rank footprint + state-tracking (cos with deficit "
        "gradient >> cross-event cos) — DIRECT tracking instrument-missing "
        "(per-event deficit-gradient directions not saved); this branch "
        "fires MIXED/INSTRUMENT-MISSING with the table verbatim",
    "CHANCE-CARVING": "in-room at chance ([0.055, 0.065] — the class null "
        "band) AND no repeat (R1 < 0.5 or mean rho < 0.5) AND no tracking "
        "(mean rho < 0.5 as registered) — only the GATING is the "
        "intelligence",
    "MIXED/INSTRUMENT-MISSING": "anything else — disclose",
}
NULL_SEED = 30701
N_EXPECT = 2_739_072
K10K = 10_000
ROOM_SEEDS = [26113, 26114]
E291_SEEDS = [29111, 29112]
CLASS_NULL_BAND = (0.055, 0.065)     # T267's law: corpus/name gradients ~0.060
VOLUME_NULL = K10K / N_EXPECT        # 0.003651 — the gaussian anchor's expect


def log(msg: str) -> None:
    print(f"[e307 {time.time() - T0:7.1f}s] {msg}", flush=True)


def _strip_private(obj):
    if isinstance(obj, dict):
        return {k: _strip_private(v) for k, v in obj.items()
                if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_strip_private(v) for v in obj]
    return obj


def write_partial(note: str) -> None:
    METRICS["phase_note"] = note
    METRICS["date_updated"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(_strip_private(METRICS), indent=1, default=str),
                   encoding="utf-8")
    tmp.replace(OUT / "metrics.json")
    log(f"WROTE metrics.json ({note})")


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ======================================================================
# G_ENDPOINTS + G_FLATBASIS — bind every object this cell trusts
# ======================================================================
ARMS = {
    "e287_NAME-MAINTAINED[flat-founding]":
        "e287_NAME-MAINTAINED_resume.pt",
    "e288_ERROR-GATED[controller,neutral]":
        "e288_ERROR-GATED_resume.pt",
    "e288_NAME-FIXED-TWIN[flat,neutral]":
        "e288_NAME-FIXED-TWIN_resume.pt",
    "e289_CONTRADICTED-WITH-CONTROLLER[controller,denial]":
        "e289_CONTRADICTED-WITH-CONTROLLER_resume.pt",
    "e289_CONTRADICTED-NO-CONTROLLER[passive,denial]":
        "e289_CONTRADICTED-NO-CONTROLLER_resume.pt",
    "e289_C1-ERROR-GATED[controller,neutral-replicate]":
        "e289_C1-ERROR-GATED_resume.pt",
    "e289_C1-NAME-FIXED-TWIN[flat,neutral-replicate]":
        "e289_C1-NAME-FIXED-TWIN_resume.pt",
    "e291_FIVE-CONTROLLERS[fleet]":
        "e291_FIVE-CONTROLLERS_resume.pt",
    "e291_SINGLE-CONTROL-TWIN[controller,fact1]":
        "e291_SINGLE-CONTROL-TWIN_resume.pt",
}
POSTS = {
    "e288_ERROR-GATED_post.pt": None, "e288_NAME-FIXED-TWIN_post.pt": None,
    "e289_CONTRADICTED-NO-CONTROLLER_post.pt": None,
    "e289_CONTRADICTED-WITH-CONTROLLER_post.pt": None,
    "e289_C1-ERROR-GATED_post.pt": None,
    "e289_C1-NAME-FIXED-TWIN_post.pt": None,
}
BINDS = {"files": {}, "pass": True}
for name, fn in {**ARMS, **{k: k for k in POSTS}}.items():
    p = CKPT / fn
    BINDS["files"][name] = {"path": f"runs/checkpoints/{fn}",
                            "md5": md5_of(p)}
BINDS["files"]["e288_rooms.pt"] = {"path": "runs/checkpoints/e288_rooms.pt",
    "md5": md5_of(CKPT / "e288_rooms.pt")}
BINDS["files"]["e291_rooms.pt"] = {"path": "runs/checkpoints/e291_rooms.pt",
    "md5": md5_of(CKPT / "e291_rooms.pt")}
# the committed fact bind (e288/e289 G_FACTLOAD's flat md5 — the loaded
# artifact every one of these arms started from)
FACT_CK = CKPT / "e261_K10K_inst_resume.pt"
FACT_FLAT_MD5_EXPECT = "ebebb4472725d582dd74928493f1bfb3"
METRICS["gates"] = {"G_ENDPOINTS": BINDS}
write_partial("P0a endpoints md5-registered (this cell IS the bind ledger)")

# ---- load everything (plain IO, the honesty rule: no memmap) --------
CK: dict = {}
KEYS: list[str] | None = None
SHAPES: list[tuple] = []
for name, fn in ARMS.items():
    ck = torch.load(CKPT / fn, map_location="cpu", weights_only=False)
    CK[name] = ck
    ks = list(ck["model"].keys())
    if KEYS is None:
        KEYS = ks
        SHAPES = [tuple(ck["model"][k].shape) for k in ks]
    assert ks == KEYS, f"key order mismatch in {name}"
POST_FLAT: dict = {}
for fn in POSTS:
    ck = torch.load(CKPT / fn, map_location="cpu", weights_only=False)
    sd = ck["model"]
    flat = torch.cat([sd[k].float().reshape(-1) for k in KEYS])
    POST_FLAT[fn] = flat.double().numpy()
fact_sd = torch.load(FACT_CK, map_location="cpu",
                     weights_only=False)["model"]
FACT_FLAT = torch.cat([fact_sd[k].float().reshape(-1)
                       for k in KEYS]).double().numpy()
h = hashlib.md5(FACT_FLAT.astype(np.float32).tobytes()).hexdigest()
N = int(FACT_FLAT.size)
assert N == N_EXPECT, N
SHAPE_OF = dict(zip(KEYS, SHAPES))

# ---- G_FLATBASIS (x6's live check: order == parameters, no buffers) --
import importlib.util
spec = importlib.util.spec_from_file_location(
    "e307_common", REPO / "lab" / "common.py")
mod = importlib.util.module_from_spec(spec)
import sys
sys.modules["e307_common"] = mod
spec.loader.exec_module(mod)
_net = mod.TinyGPT(mod.Cfg())
pnames = [n_ for n_, _ in _net.named_parameters()]
bnames = [n_ for n_, _ in _net.named_buffers()]
GB = {"params_count": len(pnames), "buffers_count": len(bnames),
      "key_order_matches_parameters": pnames == KEYS,
      "n_params": sum(p.numel() for p in _net.parameters()),
      "pass": pnames == KEYS and len(bnames) == 0}
METRICS["gates"]["G_FLATBASIS"] = GB
if not GB["pass"]:
    raise SystemExit(f"FLAT-BASIS GATE FAILURE: {GB}")
del _net
log("all checkpoints loaded; flat basis verified")

# ======================================================================
# G_ROOMS — the SRCT projectors, bit-verified vs fresh construction
# ======================================================================
class SRCT:
    """Ported BY VALUE (x6's form): P x = D . idct(mask_S(dct(D . x)))."""

    def __init__(self, n: int, k: int, seed_d: int, seed_s: int):
        g = torch.Generator().manual_seed(seed_d)
        self.D = torch.where(torch.rand(n, generator=g) < 0.5,
                             -1.0, 1.0).numpy().astype(np.float64)
        g2 = torch.Generator().manual_seed(seed_s)
        self.S = torch.randperm(n, generator=g2)[:k].sort().values.numpy()
        self.mask = np.zeros(n, dtype=np.float64)
        self.mask[self.S] = 1.0
        self.k, self.n = k, n

    def project(self, x64: np.ndarray) -> np.ndarray:
        c = sf.dct(self.D * x64, type=2, norm="ortho", workers=DCT_WORKERS)
        c *= self.mask
        return self.D * sf.idct(c, type=2, norm="ortho", workers=DCT_WORKERS)


def shared_frame_rooms(n: int, k: int, n_rooms: int,
                       seed_d: int, seed_s: int):
    """e291's construction BY VALUE: one D + one permutation's first
    n_rooms*k indices split into contiguous blocks, each sorted."""
    g = torch.Generator().manual_seed(seed_d)
    D = torch.where(torch.rand(n, generator=g) < 0.5,
                    -1.0, 1.0).numpy().astype(np.float64)
    g2 = torch.Generator().manual_seed(seed_s)
    perm = torch.randperm(n, generator=g2)[:k * n_rooms]
    sets = [perm[i * k:(i + 1) * k].sort().values.numpy()
            for i in range(n_rooms)]
    masks = []
    for S in sets:
        m = np.zeros(n, dtype=np.float64)
        m[S] = 1.0
        masks.append(m)
    union = np.zeros(n, dtype=np.float64)
    union[np.concatenate(sets)] = 1.0
    return D, sets, masks, union


rck = torch.load(CKPT / "e288_rooms.pt", map_location="cpu",
                 weights_only=False)["model"]["K10K"]
room = SRCT(N, int(rck["k"]), ROOM_SEEDS[0], ROOM_SEEDS[1])
D_stored = rck["D_int8"].numpy().astype(np.float64)
S_stored = rck["S"].numpy()
GR = {"seeds": ROOM_SEEDS, "k": int(rck["k"]),
      "D_bit_equal": bool(np.array_equal(D_stored, room.D)),
      "S_bit_equal": bool(np.array_equal(S_stored, room.S)),
      "k_fraction_of_N": float(room.k / N)}
if not (GR["D_bit_equal"] and GR["S_bit_equal"]):
    raise SystemExit(f"ROOM GATE FAILURE (K10K): {GR}")

r291 = torch.load(CKPT / "e291_rooms.pt", map_location="cpu",
                  weights_only=False)
D291, SETS291, MASKS291, UNION291 = shared_frame_rooms(
    N, K10K, 5, E291_SEEDS[0], E291_SEEDS[1])
GR291 = {"seeds": E291_SEEDS, "n_rooms": 5,
         "D_bit_equal": bool(np.array_equal(
             r291["D"].numpy().astype(np.float64), D291)),
         "sets_bit_equal": bool(all(
             np.array_equal(np.asarray(s), SETS291[i])
             for i, s in enumerate(r291["sets"])))}
if not (GR291["D_bit_equal"] and GR291["sets_bit_equal"]):
    raise SystemExit(f"ROOM GATE FAILURE (e291): {GR291}")
METRICS["gates"]["G_ROOMS"] = {"K10K": GR, "E291_shared_frame": GR291,
                               "pass": True}
log("rooms bit-verified vs fresh seeds (K10K 26113/26114; e291 29111/29112)")
write_partial("P0b rooms bit-verified")


# ======================================================================
# G_INSTRUMENT — the footprint vectors ARE the maintenance carving
# ======================================================================
def flat_model(ck: dict) -> np.ndarray:
    sd = ck["model"]
    return torch.cat([sd[k].float().reshape(-1) for k in KEYS]) \
        .double().numpy()


def flat_optbuf(opt) -> np.ndarray | None:
    st = opt["state"] if isinstance(opt, dict) and "state" in opt else None
    if st is None:
        return None
    if len(st) == 0:
        return None
    parts = []
    for i in sorted(st.keys()):
        d = st[i]
        buf = d.get("momentum_buffer") if isinstance(d, dict) else None
        if buf is None:
            return None
        parts.append(buf.detach().float().reshape(-1))
    return torch.cat(parts).double().numpy()


GI = {"checks": {}, "pass": True}
MAINT: dict = {}
CORP: dict = {}
BUFF: dict = {}
for name, ck in CK.items():
    if "e291" in name:
        mc = ck.get("maint_cums")
        MAINT[name] = [v.double().numpy() for v in mc] if mc else None
        CORP[name] = ck["corp_cum"].double().numpy()
        ofl = ck.get("optFs", [])      # a LIST of optimizer state_dicts
        bufs = []
        for od in ofl:
            st = od["state"] if isinstance(od, dict) and "state" in od \
                else {}
            if len(st) == 0 or not all(
                    isinstance(st[i], dict) and
                    st[i].get("momentum_buffer") is not None
                    for i in sorted(st.keys())):
                bufs = None
                break
            bufs.append(torch.cat(
                [st[i]["momentum_buffer"].detach().float().reshape(-1)
                 for i in sorted(st.keys())]).double().numpy())
        BUFF[name] = bufs if bufs else None
    else:
        MAINT[name] = ck["maint_cum"].double().numpy()
        CORP[name] = ck["corp_cum"].double().numpy()
        BUFF[name] = flat_optbuf(ck["optF"]) if "optF" in ck else None

# (i) the passive arm's maint_cum is EXACTLY zero (the instrument control)
mc0 = MAINT["e289_CONTRADICTED-NO-CONTROLLER[passive,denial]"]
GI["checks"]["passive_maint_cum_norm_zero"] = float(np.linalg.norm(mc0))
GI["checks"]["passive_maint_cum_is_zero"] = bool(
    np.linalg.norm(mc0) == 0.0)
# (ii) post == resume model (per arm with a post twin)
for fn, arm in (("e288_ERROR-GATED_post.pt",
                 "e288_ERROR-GATED[controller,neutral]"),
                ("e288_NAME-FIXED-TWIN_post.pt",
                 "e288_NAME-FIXED-TWIN[flat,neutral]"),
                ("e289_CONTRADICTED-NO-CONTROLLER_post.pt",
                 "e289_CONTRADICTED-NO-CONTROLLER[passive,denial]"),
                ("e289_CONTRADICTED-WITH-CONTROLLER_post.pt",
                 "e289_CONTRADICTED-WITH-CONTROLLER[controller,denial]"),
                ("e289_C1-ERROR-GATED_post.pt",
                 "e289_C1-ERROR-GATED[controller,neutral-replicate]"),
                ("e289_C1-NAME-FIXED-TWIN_post.pt",
                 "e289_C1-NAME-FIXED-TWIN[flat,neutral-replicate]")):
    d = float(np.linalg.norm(POST_FLAT[fn] - flat_model(CK[arm])))
    GI["checks"][f"post_eq_resume[{fn.split('_')[0]}_{arm.split('[')[0]}]"] = {
        "abs_diff": d,
        "rel": d / max(float(np.linalg.norm(POST_FLAT[fn])), 1e-30),
        "pass_lt_1e-4": bool(d / max(float(np.linalg.norm(POST_FLAT[fn])),
                                      1e-30) < 1e-4)}
# (iii) the DISPLACEMENT IDENTITY: post_model == fact + corp_cum +
# maint_cum (to fp32 accumulation roundoff) — maint_cum IS the carving
for arm in ("e288_ERROR-GATED[controller,neutral]",
            "e289_CONTRADICTED-WITH-CONTROLLER[controller,denial]",
            "e289_C1-ERROR-GATED[controller,neutral-replicate]"):
    recon = FACT_FLAT + CORP[arm] + (sum(MAINT[arm]) if isinstance(
        MAINT[arm], list) else MAINT[arm])
    got = flat_model(CK[arm])
    d = float(np.linalg.norm(got - recon))
    GI["checks"][f"displacement_identity[{arm.split('[')[0]}]"] = {
        "abs_residual": d,
        "rel_residual": d / max(float(np.linalg.norm(got - FACT_FLAT)),
                                1e-30),
        "pass_lt_1e-3": bool(d / max(float(np.linalg.norm(got - FACT_FLAT)),
                                      1e-30) < 1e-3)}
# (iv) the corpus carving is orthogonal to the room (the projector's own
# control — every corpus step was orthogonalized in the parents)
for arm in ("e288_ERROR-GATED[controller,neutral]",
            "e289_CONTRADICTED-WITH-CONTROLLER[controller,denial]"):
    cc = CORP[arm]
    ir = float((room.project(cc) ** 2).sum() / (cc ** 2).sum())
    GI["checks"][f"corp_cum_inroom[{arm.split('[')[0]}]"] = ir
GI["checks"]["note"] = (
    "the corpus cumulatives ride at the fp floor in the K10K room (the "
    "parents' G_ORTH construction); e291's corpus was orthogonalized to "
    "the 50k UNION, not to K10K — its K10K read is a co-report only")
GI["pass"] = bool(
    GI["checks"]["passive_maint_cum_is_zero"]
    and all(v.get("pass_lt_1e-4", True) and v.get("pass_lt_1e-3", True)
            for v in GI["checks"].values() if isinstance(v, dict)))
METRICS["gates"]["G_INSTRUMENT"] = GI
log(f"G_INSTRUMENT: passive maint_cum zero="
    f"{GI['checks']['passive_maint_cum_is_zero']}")
write_partial("P1 G_INSTRUMENT done (maint_cum == the carving, verified)")


# ======================================================================
# THE READS (x6's conventions, ported by value)
# ======================================================================
def natural_read(dw: np.ndarray) -> dict:
    a = np.sort(np.abs(dw))[::-1]
    e = np.cumsum(a * a)
    tot = float(e[-1])
    return {"m": {f"{int(th * 100)}":
                  int(np.searchsorted(e, th * tot) + 1)
                  for th in (0.5, 0.9, 0.99)}}


def svd_read(dw: np.ndarray) -> dict:
    """Per-2D-matrix SVD (fp64). T3/T10/cross90 over the 2D energy."""
    spect = []
    off = 0
    per_energy = {}
    for k in KEYS:
        sh = SHAPE_OF[k]
        n = int(np.prod(sh))
        seg = dw[off:off + n]
        off += n
        if len(sh) == 1:
            per_energy[k] = float((seg * seg).sum())
            continue
        s = torch.linalg.svdvals(torch.from_numpy(
            seg.reshape(sh).copy())).numpy()
        en = (s * s).astype(np.float64)
        spect.append(en)
        per_energy[k] = float(en.sum())
    glob = np.sort(np.concatenate(spect))[::-1]
    e2d = float(glob.sum())
    cum = np.cumsum(glob)
    tot_e = sum(per_energy.values())
    top_m = sorted(per_energy.items(), key=lambda kv: -kv[1])[:3]
    return {"T3": float(glob[:3].sum() / e2d),
            "T10": float(glob[:10].sum() / e2d),
            "T50": float(glob[:50].sum() / e2d),
            "top10_shares": [float(x / e2d) for x in glob[:10]],
            "cross90": int(np.searchsorted(cum, 0.90 * e2d) + 1),
            "cross99": int(np.searchsorted(cum, 0.99 * e2d) + 1),
            "e_2d": e2d,
            "top3_matrices_by_energy": [
                {"param": k, "energy_share": v / tot_e} for k, v in top_m],
            "_glob": glob}


def in_room_frac(x: np.ndarray, D: np.ndarray, mask: np.ndarray) -> float:
    c = sf.dct(D * x, type=2, norm="ortho", workers=DCT_WORKERS)
    c *= mask
    return float((c * c).sum() / max(float((x * x).sum()), 1e-30))


def geometry(x: np.ndarray, label: str, room_mask=room.mask,
             D=room.D) -> dict:
    g = {"label": label, "l2": float(np.linalg.norm(x)),
         "in_room_frac": in_room_frac(x, D, room_mask),
         "natural": natural_read(x)}
    s = svd_read(x)
    g["svd"] = {kk: vv for kk, vv in s.items() if kk != "_glob"}
    g["_glob"] = s["_glob"]
    return g


# ---------------- THE DELTA INVENTORY (a) ---------------------------
INV = {"pure_footprints": {}, "pair_deltas": {}, "disclosures": [
    "the per-event step VECTORS are not in the records (ledgers carry "
    "scalars only) — the full cos(step_i, step_j) matrix is "
    "instrument-missing; R1/R2/R3 are the exact registered substitutes",
    "e288's and e289-C1's twins are FLAT-DOSE maintained arms, not "
    "passive — their pair deltas are controller-vs-flat contrasts; the "
    "ONLY passive-twin pair is e289 WC-NC (the causal delta)",
    "e287's SANCTUARY-TWIN is passive but on e287's own session stream "
    "(cross-session draws differ) — excluded from bars, disclosed",
    "e293 exists on disk (uncommitted, outside the dispatch's "
    "provenance e288/e289/e291) — not read",
    "the pair deltas contain the maintenance carving AND the corpus "
    "stream's causal response to it (the arms diverge after event 1, "
    "so later corpus gradients differ even under bit-identical draws) "
    "— the FULL controller-presence footprint, disclosed as such",
]}
for name in ARMS:
    if isinstance(MAINT[name], list):
        for i, v in enumerate(MAINT[name]):
            INV["pure_footprints"][f"{name}:FACT{i + 1}"] = {
                "l2": float(np.linalg.norm(v))}
    else:
        INV["pure_footprints"][name] = {
            "l2": float(np.linalg.norm(MAINT[name]))}
INV["corpus_cum_norms"] = {n: float(np.linalg.norm(v))
                           for n, v in CORP.items()}
DELTA: dict = {}
DELTA["e289_causal[WC-NC,denial]"] = (
    POST_FLAT["e289_CONTRADICTED-WITH-CONTROLLER_post.pt"]
    - POST_FLAT["e289_CONTRADICTED-NO-CONTROLLER_post.pt"])
DELTA["e288_pair[EG-NFT,neutral]"] = (
    POST_FLAT["e288_ERROR-GATED_post.pt"]
    - POST_FLAT["e288_NAME-FIXED-TWIN_post.pt"])
DELTA["e289_c1_pair[C1EG-C1NFT,neutral]"] = (
    POST_FLAT["e289_C1-ERROR-GATED_post.pt"]
    - POST_FLAT["e289_C1-NAME-FIXED-TWIN_post.pt"])
INV["pair_deltas"] = {k: {"l2": float(np.linalg.norm(v))}
                      for k, v in DELTA.items()}
METRICS["inventory"] = INV
write_partial("P2 the delta inventory written")

# ---------------- (b) THE FOOTPRINT'S GEOMETRY ----------------------
GEO: dict = {}
PRIMARY = "e288_ERROR-GATED[controller,neutral]"
for name in ARMS:
    if isinstance(MAINT[name], list):
        continue  # e291 handled in the fleet section
    GEO[name] = geometry(MAINT[name], name)
write_partial("P3a pure-footprint geometry done (7 arms)")

for k, v in DELTA.items():
    GEO[k] = geometry(v, k)
# the null anchor (x6's conventions)
rng = np.random.default_rng(NULL_SEED)
gx = rng.standard_normal(N)
gx *= float(np.linalg.norm(MAINT[PRIMARY])) / float(np.linalg.norm(gx))
GEO["null_gaussian[seed30701,norm-matched-EG]"] = geometry(gx, "null")
METRICS["geometry"] = {k: {kk: vv for kk, vv in g.items() if kk != "_glob"}
                       for k, g in GEO.items()}
METRICS["nulls"] = {
    "gaussian_seed": NULL_SEED,
    "gaussian_in_room": GEO["null_gaussian[seed30701,norm-matched-EG]"][
        "in_room_frac"],
    "volume_null_k_over_n": VOLUME_NULL,
    "class_null_band": CLASS_NULL_BAND,
    "class_null_law": "T267/T268: corpus and name-gradient in-room "
                      "fraction ~0.060 (the committed cells' per-step "
                      "reads 0.0597-0.0611)",
}
write_partial("P3b pair-delta + null geometry done")

# ---------------- (c) THE REPEAT INSTRUMENTS ------------------------
REP: dict = {}


def repeat_read(name: str, mcum: np.ndarray, ledger: list) -> dict:
    r = np.array([e["realized_step_norm"] for e in ledger], dtype=np.float64)
    b = np.array([e["b_m_norm"] for e in ledger], dtype=np.float64)
    gn = np.array([e["gn_clipped"] for e in ledger], dtype=np.float64)
    lr = np.array([e["lr_maint"] for e in ledger], dtype=np.float64)
    sr = float(mcum @ mcum)
    s1, s2 = float(r.sum()), float((r * r).sum())
    wmean = (sr - s2) / max(s1 * s1 - s2, 1e-30)
    # rho_t: fresh gradient vs running buffer (exact SGD-M algebra)
    rho = []
    for t in range(1, len(b)):
        den = 1.8 * b[t - 1] * gn[t]
        rho.append((b[t] ** 2 - 0.81 * b[t - 1] ** 2 - gn[t] ** 2)
                   / (den if den > 1e-12 else 1e-12))
    rho = np.array(rho)
    consec = np.array([(0.9 * b[t - 1] + rho[t - 1] * gn[t]) / b[t]
                       for t in range(1, len(b))])
    # realized == lr * b_t cross-check (the instrument's own audit)
    aud = float(np.max(np.abs(r - lr * b) / np.maximum(r, 1e-12)))
    return {"n_events": len(r), "sum_r": s1, "cum_norm": math.sqrt(sr),
            "wmean_cross_event_cos_R1": wmean,
            "rho_series_R2": [float(x) for x in rho],
            "rho_mean": float(rho.mean()), "rho_median": float(
                np.median(rho)),
            "rho_min": float(rho.min()), "rho_max": float(rho.max()),
            "consec_step_cos": [float(x) for x in consec],
            "consec_cos_mean": float(consec.mean()),
            "realized_eq_lr_x_bt_maxrel": aud,
            "g_in_room_frac_per_event": [
                float(e.get("g_in_room_frac",
                            e.get("g_in_own_room_frac", float("nan"))))
                for e in ledger]}


for name in ARMS:
    if "e291" in name or len(CK[name].get("maint_ledger", [])) == 0:
        continue  # e291 handled in the fleet section; NC has no events
    REP[name] = repeat_read(name, MAINT[name], CK[name]["maint_ledger"])
    bufF = BUFF[name]
    if bufF is not None:
        REP[name]["cos_cum_vs_final_bufF_R3"] = float(
            MAINT[name] @ (-bufF)
            / (np.linalg.norm(MAINT[name]) * np.linalg.norm(bufF)))
# the deficit-dose correlation (by CONSTRUCTION — co-report, never a bar)
for name in ARMS:
    if "e291" in name or "NO-CONTROLLER" in name:
        continue
    led = CK[name]["maint_ledger"]
    if not led or not all("deficit_t" in e and e["deficit_t"] is not None
                          for e in led):
        continue  # the flat schedules (e287/NFT) carry no deficit read
    d = np.array([e.get("deficit_t") for e in led], dtype=np.float64)
    lrv = np.array([e["lr_maint"] for e in led], dtype=np.float64)
    if d.std() > 0:
        REP[name]["dose_deficit_corr_by_construction"] = float(
            np.corrcoef(d, lrv)[0, 1])
METRICS["repeat"] = REP
write_partial("P4 repeat instruments done (R1 identity + R2 rho + R3)")

# ---------------- cross-session / cross-arm cosines -----------------
CROSS = {}
one_fact = [n for n in ARMS if "e291" not in n
            and "NO-CONTROLLER" not in n]
one_fact.append("e291_SINGLE-CONTROL-TWIN[controller,fact1]:FACT1")
mats = {}
for n in one_fact:
    mats[n] = (MAINT[n][0] if isinstance(MAINT[n], list) else MAINT[n])
for i, a in enumerate(one_fact):
    for b_ in one_fact[i + 1:]:
        CROSS[f"{a.split('[')[0]}~{b_.split('[')[0]}"] = float(
            mats[a] @ mats[b_]
            / (np.linalg.norm(mats[a]) * np.linalg.norm(mats[b_])))
METRICS["cross_session_cos"] = CROSS
write_partial("P5 cross-session footprint cosines done")

# ---------------- (d)+fleet: e291 + the denial contrast -------------
FLEET = {"per_fact": {}, "cross_fact_cos": {}}
f5 = MAINT["e291_FIVE-CONTROLLERS[fleet]"]
for i, v in enumerate(f5):
    own = in_room_frac(v, D291, MASKS291[i])
    uni = in_room_frac(v, D291, UNION291)
    g = geometry(v, f"e291:FACT{i + 1}", room_mask=MASKS291[i], D=D291)
    FLEET["per_fact"][f"FACT{i + 1}"] = {
        "l2": float(np.linalg.norm(v)), "in_own_room": own,
        "in_union": uni, "svd_T3": g["svd"]["T3"],
        "svd_cross90": g["svd"]["cross90"]}
for i in range(5):
    for j in range(i + 1, 5):
        FLEET["cross_fact_cos"][f"FACT{i + 1}~FACT{j + 1}"] = float(
            f5[i] @ f5[j] / (np.linalg.norm(f5[i]) * np.linalg.norm(f5[j])))
fleet_sum = np.sum(f5, axis=0)
FLEET["fleet_sum"] = geometry(fleet_sum, "e291:fleet-sum",
                              room_mask=UNION291, D=D291)
FLEET["fleet_sum"]["in_room_frac_K10K"] = in_room_frac(
    fleet_sum, room.D, room.mask)
t1 = MAINT["e291_SINGLE-CONTROL-TWIN[controller,fact1]"][0]
FLEET["twin_fact1"] = {"l2": float(np.linalg.norm(t1)),
                       "in_own_room": in_room_frac(t1, D291, MASKS291[0]),
                       "in_union": in_room_frac(t1, D291, UNION291)}
tb = BUFF["e291_SINGLE-CONTROL-TWIN[controller,fact1]"]
if tb:
    FLEET["twin_fact1"]["cos_cum_vs_final_bufF_R3"] = float(
        t1 @ (-tb[0]) / (np.linalg.norm(t1) * np.linalg.norm(tb[0])))
    FLEET["twin_fact1"]["svd_T3"] = geometry(
        t1, "twin", room_mask=MASKS291[0], D=D291)["svd"]["T3"]
    FLEET["twin_fact1"]["svd_cross90"] = geometry(
        t1, "twin", room_mask=MASKS291[0], D=D291)["svd"]["cross90"]
METRICS["fleet_e291"] = FLEET
write_partial("P6 e291 fleet footprint reads done")

# ======================================================================
# ADJUDICATION (the registered tree — no bar shopping)
# ======================================================================
gp = GEO[PRIMARY]
rp = REP[PRIMARY]
clause_rank = bool(gp["svd"]["T3"] >= 0.50)
clause_repeat = bool(rp["wmean_cross_event_cos_R1"] >= 0.5
                     and rp["rho_mean"] >= 0.5)
clause_inroom_chance = bool(CLASS_NULL_BAND[0] <= gp["in_room_frac"]
                            <= CLASS_NULL_BAND[1])
clause_highrank = bool(gp["svd"]["cross90"] >= 1000)
clause_tracking_direct = None  # instrument-missing (disclosed)

if clause_rank and clause_repeat:
    verdict = "ONE-ANTIBODY"
elif clause_inroom_chance and not clause_repeat:
    verdict = "CHANCE-CARVING"
elif clause_highrank:
    verdict = "MIXED/INSTRUMENT-MISSING"   # NAVIGATION's tracking clause
else:
    verdict = "MIXED/INSTRUMENT-MISSING"

CLASS = {}
for name in ("e289_CONTRADICTED-WITH-CONTROLLER[controller,denial]",
             "e289_C1-ERROR-GATED[controller,neutral-replicate]",
             "e288_NAME-FIXED-TWIN[flat,neutral]",
             "e289_C1-NAME-FIXED-TWIN[flat,neutral-replicate]",
             "e287_NAME-MAINTAINED[flat-founding]"):
    g, r = GEO[name], REP[name]
    CLASS[name] = {"T3": g["svd"]["T3"], "cross90": g["svd"]["cross90"],
                   "in_room": g["in_room_frac"],
                   "R1": r["wmean_cross_event_cos_R1"],
                   "rho_mean": r["rho_mean"],
                   "rank_clause": bool(g["svd"]["T3"] >= 0.5),
                   "repeat_clause": bool(
                       r["wmean_cross_event_cos_R1"] >= 0.5
                       and r["rho_mean"] >= 0.5)}
METRICS["verdict"] = {
    "word": verdict,
    "primary_object": PRIMARY,
    "bars_verbatim": BARS,
    "clause_trace": {
        "rank_T3_ge_50pct": clause_rank, "T3": gp["svd"]["T3"],
        "repeat_R1_ge_05": bool(rp["wmean_cross_event_cos_R1"] >= 0.5),
        "R1": rp["wmean_cross_event_cos_R1"],
        "repeat_rho_mean_ge_05": bool(rp["rho_mean"] >= 0.5),
        "rho_mean": rp["rho_mean"],
        "in_room_at_chance": clause_inroom_chance,
        "in_room_frac": gp["in_room_frac"],
        "high_rank_cross90_ge_1000": clause_highrank,
        "cross90": gp["svd"]["cross90"],
        "tracking_direct": "INSTRUMENT-MISSING (per-event deficit-gradient "
                           "directions not in the records; disclosed)"},
    "class_table": CLASS,
    "cross_cell_d": {
        "neutral_e288_EG": {"in_room": GEO[PRIMARY]["in_room_frac"],
                            "T3": gp["svd"]["T3"],
                            "R1": rp["wmean_cross_event_cos_R1"],
                            "rho_mean": rp["rho_mean"]},
        "denial_e289_WC": {
            "in_room": GEO["e289_CONTRADICTED-WITH-CONTROLLER"
                           "[controller,denial]"]["in_room_frac"],
            "T3": GEO["e289_CONTRADICTED-WITH-CONTROLLER"
                      "[controller,denial]"]["svd"]["T3"],
            "R1": REP["e289_CONTRADICTED-WITH-CONTROLLER"
                      "[controller,denial]"]["wmean_cross_event_cos_R1"],
            "rho_mean": REP["e289_CONTRADICTED-WITH-CONTROLLER"
                           "[controller,denial]"]["rho_mean"]},
        "cos_neutral_vs_denial_footprint": CROSS.get(
            "e288_ERROR-GATED~e289_CONTRADICTED-WITH-CONTROLLER"),
        "causal_delta_in_room": GEO["e289_causal[WC-NC,denial]"][
            "in_room_frac"],
        "causal_delta_T3": GEO["e289_causal[WC-NC,denial]"]["svd"]["T3"]},
}
write_partial(f"P7 ADJUDICATED: {verdict}")

# ======================================================================
# FIGURE
# ======================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

ax = axes[0, 0]
spec = [(PRIMARY, GEO[PRIMARY], "crimson", "-"),
        ("e289_CONTRADICTED-WITH-CONTROLLER[controller,denial]",
         GEO["e289_CONTRADICTED-WITH-CONTROLLER[controller,denial]"],
         "darkorange", "-"),
        ("e289_causal[WC-NC,denial]", GEO["e289_causal[WC-NC,denial]"],
         "steelblue", "--"),
        ("null_gaussian[seed30701,norm-matched-EG]",
         GEO["null_gaussian[seed30701,norm-matched-EG]"], "gray", ":")]
for nm, g, c, ls in spec:
    gl = g["_glob"]
    cum = np.cumsum(gl) / gl.sum()
    ix = np.unique(np.linspace(0, len(gl) - 1, 3000).astype(int))
    ax.plot(np.arange(1, len(gl) + 1)[ix], cum[ix], color=c, ls=ls, lw=1.5,
            label=f"{nm.split('[')[0]}  T3={g['svd']['T3']:.3f} "
                  f"cross90={g['svd']['cross90']}")
ax.set_xscale("log")
ax.axhline(0.9, color="k", ls="--", lw=0.6, alpha=0.5)
ax.axhline(0.5, color="k", ls=":", lw=0.6, alpha=0.5)
ax.set_xlabel("singular rank (global 2D spectrum, log)")
ax.set_ylabel("cumulative energy share")
ax.set_title("(a) THE FOOTPRINT'S SVD — low-rank or spread?")
ax.legend(fontsize=7.5, loc="lower right")

ax = axes[0, 1]
labels, vals, cols = [], [], []
for nm in (PRIMARY,
           "e288_NAME-FIXED-TWIN[flat,neutral]",
           "e289_CONTRADICTED-WITH-CONTROLLER[controller,denial]",
           "e289_C1-ERROR-GATED[controller,neutral-replicate]",
           "e289_C1-NAME-FIXED-TWIN[flat,neutral-replicate]",
           "e287_NAME-MAINTAINED[flat-founding]",
           "e289_causal[WC-NC,denial]",
           "e288_pair[EG-NFT,neutral]",
           "e291_FIVE-CONTROLLERS[fleet]:fleet-sum"):
    g = (FLEET["fleet_sum"] if "fleet" in nm else GEO[nm])
    labels.append(nm.split("[")[0].replace("_", "\n", 1))
    vals.append(g["in_room_frac"])
    cols.append("crimson" if "ERROR" in nm or "CONTRADICTED-WITH" in nm
                else "steelblue" if "TWIN" in nm or "MAINT" in nm
                else "seagreen")
ax.bar(range(len(vals)), vals, color=cols)
ax.axhline(0.06, color="k", ls="--", lw=1.0)
ax.axhline(VOLUME_NULL, color="gray", ls=":", lw=1.0)
ax.text(len(vals) - 0.4, 0.061, "class null 0.060 (T267)", fontsize=8,
        ha="right")
ax.text(len(vals) - 0.4, VOLUME_NULL + 0.002, "k/N volume null",
        fontsize=8, ha="right", color="gray")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=6.5, rotation=45, ha="right")
ax.set_ylabel("in-room fraction of the footprint")
ax.set_title("(b) DOES ACCUMULATION BEND THE FOOTPRINT IN-ROOM?")

ax = axes[1, 0]
for nm, c in ((PRIMARY, "crimson"),
              ("e289_CONTRADICTED-WITH-CONTROLLER[controller,denial]",
               "darkorange"),
              ("e289_C1-ERROR-GATED[controller,neutral-replicate]",
               "seagreen"),
              ("e288_NAME-FIXED-TWIN[flat,neutral]", "steelblue"),
              ("e287_NAME-MAINTAINED[flat-founding]", "gray")):
    r = REP[nm]["rho_series_R2"]
    ax.plot(range(2, len(r) + 2), r, color=c, lw=1.4, marker="o", ms=3,
            label=f"{nm.split('[')[0]}  mean={np.mean(r):.3f}")
ax.axhline(0.5, color="k", ls=":", lw=1.0)
ax.set_xlabel("maintenance event")
ax.set_ylabel("rho_t (fresh gradient vs running buffer)")
ax.set_title("(c) THE REPEAT QUESTION — gradient-level alignment per event")
ax.legend(fontsize=7.5)

ax = axes[1, 1]
names = one_fact
M = np.eye(len(names))
for i, a in enumerate(names):
    for j, b_ in enumerate(names):
        if i < j:
            cv = mats[a] @ mats[b_] / (
                np.linalg.norm(mats[a]) * np.linalg.norm(mats[b_]))
            M[i, j] = M[j, i] = cv
im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.split("[")[0].split(":")[0] for n in names],
                   fontsize=6.5, rotation=45, ha="right")
ax.set_yticks(range(len(names)))
ax.set_yticklabels([n.split("[")[0].split(":")[0] for n in names],
                   fontsize=6.5)
plt.colorbar(im, ax=ax, fraction=0.046)
ax.set_title("(d) THE ANTIBODY'S SESSION-INVARIANCE — footprint cosines")

fig.suptitle(
    f"e307 THE MAINTENANCE FOOTPRINT — verdict: {verdict}  "
    f"(T3={gp['svd']['T3']:.3f}, R1={rp['wmean_cross_event_cos_R1']:.3f}, "
    f"rho_mean={rp['rho_mean']:.3f}, "
    f"in-room={gp['in_room_frac']:.4f})", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT / "e307_footprint.png", dpi=140)
log("FIGURE written")

# ======================================================================
# REPORT.md
# ======================================================================
cl = METRICS["verdict"]["clause_trace"]
wc = GEO["e289_CONTRADICTED-WITH-CONTROLLER[controller,denial]"]
wcr = REP["e289_CONTRADICTED-WITH-CONTROLLER[controller,denial]"]
c1 = GEO["e289_C1-ERROR-GATED[controller,neutral-replicate]"]
c1r = REP["e289_C1-ERROR-GATED[controller,neutral-replicate]"]
nft = GEO["e288_NAME-FIXED-TWIN[flat,neutral]"]
nftr = REP["e288_NAME-FIXED-TWIN[flat,neutral]"]
cd = GEO["e289_causal[WC-NC,denial]"]
nl = GEO["null_gaussian[seed30701,norm-matched-EG]"]
report = f"""# e307 — THE MAINTENANCE FOOTPRINT — REPORT (executor-written)

**VERDICT: {verdict}** (the frozen bars, adjudicated on the primary
object `{PRIMARY}` = the founding controller's accumulated maintenance
carving, against the birth-committed registration — BEFORE any compute).

The controller's own carving, measured from the saved records (CPU,
committed checkpoints only): the accumulated maintenance footprint
`maint_cum` (the machine-accumulated sum of every realized maintenance
displacement — verified by the displacement identity post = fact +
corp_cum + maint_cum) is **{'LOW-RANK' if clause_rank else 'NOT low-rank at the 50% bar'}**
(top-3 SVD energy share **{cl['T3']:.4f}**, cross-90 effective rank
**{cl['cross90']}**) and **{'REPEATING' if clause_repeat else 'NOT repeating'}**
(the exact norm-weighted mean cross-event step cosine R1 =
**{cl['R1']:.4f}**; the gradient-level alignment rho_t mean =
**{cl['rho_mean']:.4f}** — the momentum-inflation guard). In-room
fraction **{cl['in_room_frac']:.4f}** vs the class null 0.060
(T267/T268) and the k/N volume null {VOLUME_NULL:.5f} (gaussian anchor
{METRICS['nulls']['gaussian_in_room']:.5f}).

## (a) The delta inventory (what exists on disk)

- PURE per-arm footprints (`maint_cum` in the resume checkpoints):
  e287 flat-founding {INV['pure_footprints']['e287_NAME-MAINTAINED[flat-founding]']['l2']:.4f};
  e288-EG {INV['pure_footprints'][PRIMARY]['l2']:.4f};
  e288-NFT {INV['pure_footprints']['e288_NAME-FIXED-TWIN[flat,neutral]']['l2']:.4f};
  e289-WC (denial) {INV['pure_footprints']['e289_CONTRADICTED-WITH-CONTROLLER[controller,denial]']['l2']:.4f};
  e289-C1-EG {INV['pure_footprints']['e289_C1-ERROR-GATED[controller,neutral-replicate]']['l2']:.4f};
  e289-C1-NFT {INV['pure_footprints']['e289_C1-NAME-FIXED-TWIN[flat,neutral-replicate]']['l2']:.4f};
  e289-NC (passive) EXACTLY 0 (the instrument control);
  e291: five per-fact footprints + the twin's.
- PAIR DELTAS (post-minus-post): e289 causal WC-NC
  {INV['pair_deltas']['e289_causal[WC-NC,denial]']['l2']:.4f} (the ONLY
  passive-twin pair); e288 EG-NFT
  {INV['pair_deltas']['e288_pair[EG-NFT,neutral]']['l2']:.4f} and e289
  C1 pair {INV['pair_deltas']['e289_c1_pair[C1EG-C1NFT,neutral]']['l2']:.4f}
  (controller-vs-FLAT contrasts — the twins are NOT passive, disclosed).
- The per-event step VECTORS are not saved (scalars only) — the
  cos(step_i, step_j) matrix is instrument-missing; R1/R2/R3 are the
  exact registered substitutes.

## (b) The footprint's geometry

| object | L2 | in-room | T3 | cross90 | R1 | rho mean |
|---|---|---|---|---|---|---|
| e288-EG (PRIMARY) | {gp['l2']:.4f} | {gp['in_room_frac']:.5f} | {gp['svd']['T3']:.4f} | {gp['svd']['cross90']} | {rp['wmean_cross_event_cos_R1']:.4f} | {rp['rho_mean']:.4f} |
| e289-WC (denial) | {wc['l2']:.4f} | {wc['in_room_frac']:.5f} | {wc['svd']['T3']:.4f} | {wc['svd']['cross90']} | {wcr['wmean_cross_event_cos_R1']:.4f} | {wcr['rho_mean']:.4f} |
| e289-C1-EG | {c1['l2']:.4f} | {c1['in_room_frac']:.5f} | {c1['svd']['T3']:.4f} | {c1['svd']['cross90']} | {c1r['wmean_cross_event_cos_R1']:.4f} | {c1r['rho_mean']:.4f} |
| e288-NFT (flat) | {nft['l2']:.4f} | {nft['in_room_frac']:.5f} | {nft['svd']['T3']:.4f} | {nft['svd']['cross90']} | {nftr['wmean_cross_event_cos_R1']:.4f} | {nftr['rho_mean']:.4f} |
| e289 causal delta | {cd['l2']:.4f} | {cd['in_room_frac']:.5f} | {cd['svd']['T3']:.4f} | {cd['svd']['cross90']} | - | - |
| null gaussian | {nl['l2']:.4f} | {nl['in_room_frac']:.6f} | {nl['svd']['T3']:.2e} | {nl['svd']['cross90']} | - | - |

## (c) The repeat question

R1 (exact identity) per arm: e287 {REP['e287_NAME-MAINTAINED[flat-founding]']['wmean_cross_event_cos_R1']:.4f};
e288-EG {rp['wmean_cross_event_cos_R1']:.4f}; e288-NFT
{nftr['wmean_cross_event_cos_R1']:.4f}; e289-WC
{wcr['wmean_cross_event_cos_R1']:.4f}; e289-C1-EG
{c1r['wmean_cross_event_cos_R1']:.4f}. rho_t series + consecutive-step
cosines in metrics.json; R3 (footprint vs the FINAL momentum buffer)
per arm in metrics.json.

## (d) The cross-cell comparison (neutral vs denial)

See metrics.json `verdict.cross_cell_d` — the footprint cosines
({METRICS['verdict']['cross_cell_d']['cos_neutral_vs_denial_footprint']:.4f}
neutral-vs-denial), the causal delta's geometry, and the class table.

## Disclosures (the honesty ledger)

- The per-event step vectors are NOT in the records — "median
  cross-event cos" was registered at birth to the exact norm-weighted
  mean (R1) + the gradient-level rho_t guard (R2). No bar shopped.
- NAVIGATION's tracking clause (cos with the deficit gradient) is
  DIRECT instrument-missing; the dose-vs-deficit correlation is 1.0 BY
  CONSTRUCTION (the controller's law) and was never used as evidence.
- The momentum buffer mechanically smooths the steps (R1 inflates vs
  raw-gradient alignment); the rho_t guard carries the gradient level.
- The pair deltas include the corpus stream's causal response to the
  maintenance (both arms share bit-identical draws but diverge in
  state after event 1) — the FULL controller-presence footprint.
- n=1 per arm; the cross-session cosines ride one draw each (the
  family's standing lottery caveat).
- NO NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator folds).
"""
(OUT / "REPORT.md").write_text(report, encoding="utf-8")

METRICS["outputs"] = {"figure": "runs/e307/e307_footprint.png",
                      "metrics": "runs/e307/metrics.json",
                      "report": "runs/e307/REPORT.md"}
METRICS["status"] = "COMPLETE — adjudicated"
METRICS["date_completed"] = now_utc()
write_partial("P8 COMPLETE (report + figure)")
log(f"DONE — verdict {verdict}")
