"""
x11 — THE CONSOLIDATION LEDGER (the R66 ideator's C1; prediction P-x11a
registered in THINKING.md T256) — a CPU-ONLY DESK CELL.

e272 relocated the expression edge to (1k,2k]: the 1k write is DEAD (post g0
0.000435), 2k transitional, 5k alive-ish. x10 acquitted the DOSE (the
kept-matched K1KM arm stays dead at lr x3.73). THE REGISTERED QUESTION
(P-x11a): is the expression edge a CONSOLIDATION-RATE threshold? Per
committed rung, the CONSOLIDATION RATIO := ||dW_final,in-room|| (the final
in-room write mass, from the inst_resume s400 checkpoints, projected into
the rung's own room) DIVIDED BY the CUMULATIVE APPLIED in-room displacement
(the integrated per-step lr x ||g'|| channel from the parent cells'
committed install ledgers) — how much of what was pushed STAYED.

BARS — FROZEN VERBATIM FROM THE DISPATCH, BEFORE ANY COMPUTE (no bar
shopping; this docstring is birth-committed before any fitting runs):

  RATIO-CLIFF: the ratio jumps >= 3x at the (1k,2k] edge — the floor is a
      rate threshold; P-x11a fires.
  RATIO-FLAT: no jump — the edge is absolute-scale; the dose acquittal
      becomes the puzzle.
  RATIO-INVERTED: below-edge rooms consolidate BETTER per unit
      displacement — a pure decay-side threshold.
  MIXED: the families verbatim, no inflation.

OPERATIONALIZATIONS — FROZEN BEFORE COMPUTE:
  * THE LADDER (7 committed rungs, each read in ITS OWN room — x10's map):
      K1K   = e261_K1K_inst_resume.pt   room e261_rooms:K1K  (dead rung)
      K1KM  = e272_K1KM_inst_resume.pt  room e272_rooms:K1KM (dead, lr x3.7306)
      K2K   = e272_K2K_inst_resume.pt   room e272_rooms:K2K  (transitional)
      K5K   = e272_K5K_inst_resume.pt   room e272_rooms:K5K  (alive-ish)
      K10K  = e261_K10K_inst_resume.pt  room e264_rooms:K10K (alive; e264's
              resume-completed rung — the s278 cut + e264 pass-1 completion,
              committed ledger covers s1..s400 whole, disclosed)
      K10KR = e272_K10KR_inst_resume.pt room e272_rooms:K10KR (10k replicate)
      K237K = e261_K237K_inst_resume.pt room e261_rooms:K237K (natural width)
    Root = runs/checkpoints/e001.pt (md5 d114536d1c0983ab3be67f67ff0667c8,
    x6/e267 bind). dW := flat(inst_resume s400) - flat(e001) — x6/x10's
    convention verbatim; plain-IO CPU loads only (torch.load
    map_location='cpu', .float().clone(), finiteness verified; NO memmap —
    the OneDrive honesty rule).
  * NUMERATOR := ||P_room dW||_2, P the SRCT projector
    D.idct(mask_S(dct(D.x))) in fp64 CPU (x6's exact form, ported BY VALUE).
  * DENOMINATOR := the cumulative applied in-room displacement, integrated
    from each parent cell's committed install ledger: every ledger records
    gpn := ||g'|| (the fp64 norm of the PROJECTED — i.e. in-room applied —
    gradient) at steps {1, 10, ..., 400} (41 points). The per-step applied
    displacement proxy := lr(s) x gpn(s) with lr(s) = 1e-3 x lr_scale x
    cosine_lr(s-1, 1000) (the house schedule, lab/common.py ported BY
    VALUE; warmup 100; lr_scale = 1.0 everywhere except K1KM's committed
    3.7305567687315575). gpn(s) between ledger points: PIECEWISE-LINEAR
    interpolation over the 41 committed points, evaluated at every
    s in {1..400}; D := sum_s lr(s) x gpn(s). DISCLOSED PROXY FORM:
    first-order (lr x ||g'||) — Adam's per-coordinate preconditioner and
    AdamW's decoupled weight decay are UNMODELED (e272's own dose_control
    caveat, quoted); the proxy is used as a consistent cross-rung currency,
    and a hold-left interpolation sensitivity co-report is computed (the
    verdict must not flip on it).
  * G_DOSE-REPRO GATE: for the four e272 arms, median over the 41 ledger
    steps of lr(s) x gpn(s) must reproduce the parent's committed
    applied_dose_proxy_median within 1e-9 relative — end-to-end
    validation of the lr schedule + ledger semantics BEFORE any ratio is
    believed.
  * G_DISPL CONTENT BIND (x10's precedent): this cell's fp64
    inside_share := ||P dW||^2/||dW||^2 must equal the parent's committed
    in_own_room NORM ratio squared within 2e-3 for all 7 rungs.
  * G_DISP-NORM GATE: for the four e272 arms, the computed numerator
    ||P dW|| must match the parent's committed in_room_disp_norm within
    5e-3 relative (the parent computed it from the same vehicles).
  * ENDPOINT BINDS: all 14 touched files carry md5s recorded in x10's
    COMMITTED record (runs/x10/metrics.json, itself fresh-md5-bound here:
    2d1cec68a0e7d471fd991e3693913d01) — hard gates, every one. Rooms
    additionally BIT-verified vs fresh seed reconstruction (x6's
    precedent). e261/e264/e272 metrics md5-bound (e272's G_PARENTS binds
    + x10's fresh binds).
  * ADJUDICATION (frozen): r(rung) := numerator/denominator;
    J := r(K2K)/r(K1K) — the (1k,2k] edge read (dead 1k rung vs the first
    above-edge rung).
      RATIO-CLIFF     iff J >= 3.0
      RATIO-INVERTED  iff J < 1.0
      RATIO-FLAT      iff 1.0 <= J < 3.0
    MIXED OVERRIDES the primary word when the rung structure contradicts a
    localized edge: (a) the above-edge family {K2K, K5K, K10K, K10KR,
    K237K} spreads more than 3x (max/min > 3 — the edge is not at (1k,2k]),
    the whole ladder smears), OR (b) r(K1KM) > r(K2K) (the dead 1k rung at
    matched kept-dose consolidates above the first above-edge rung — dose
    writes the ratio at fixed width; a width/rate threshold story dies).
    K1KM and K10KR are co-reports that enter only through (a)/(b), never
    as the primary edge read. Hold-left sensitivity: if the primary word
    (CLIFF/FLAT/INVERTED) flips under hold-left interpolation, the verdict
    is MIXED with the flip disclosed.
  * CPU ENVELOPE: CUDA_VISIBLE_DEVICES='' hard-set before torch import;
    torch threads 4; pocketfft workers 4; NOTHING written to
    runs/_envelope_log.jsonl; no GPU code path. PROGRESSIVE writes to
    runs/x11/metrics.json; NO NOTES/THINKING/QUEUE/STATE edits (the
    coordinator folds). Every timestamp from datetime.now(UTC) — never a
    hand guess (lab rule, 8e1ac9f).

Provenance (all md5-verified in G_ENDPOINTS below):
  ROOT       runs/checkpoints/e001.pt                d114536d1c0983ab3be67f67ff0667c8
  K1K        runs/checkpoints/e261_K1K_inst_resume.pt  f2975890c7870480d7f48b74dee33041
  K1KM       runs/checkpoints/e272_K1KM_inst_resume.pt 2306a448b450f192ef2d7d1f9aa90e2e
  K2K        runs/checkpoints/e272_K2K_inst_resume.pt  f31b437cb81ae6f06aa35daceb3b4d67
  K5K        runs/checkpoints/e272_K5K_inst_resume.pt  62049b93c2435639a945ce80cbc46470
  K10K       runs/checkpoints/e261_K10K_inst_resume.pt 0f6dc1cf46850ce655dfafc9c853d467
  K10KR      runs/checkpoints/e272_K10KR_inst_resume.pt 6ce712a7aae8482a4c78d1b87f3194b1
  K237K      runs/checkpoints/e261_K237K_inst_resume.pt 458e5211ef61aae7e14cd3b966f23dfc
  ROOMS      e261_rooms f81571c40bb6009d4ed1cfab66e73f6c / e264_rooms
             2d524655575cce00a3bc1c8770f4b211 / e272_rooms
             066944855b3295e8796c6ca28b2e498c  (all bit-verified vs seeds)
  LEDGERS    runs/e261/metrics.json f460475d8e6b76f0719e91c1e9c6041b /
             runs/e264/metrics.json a42ff4786784b04cb9819a69b545e343 /
             runs/e272/metrics.json eb9f624708bcb6576c4115161dfd7042
  BIND TABLE runs/x10/metrics.json 2d1cec68a0e7d471fd991e3693913d01 (fresh)
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-ONLY, hard-set (no GPU lane)

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
OUT = REPO / "runs" / "x11"
OUT.mkdir(parents=True, exist_ok=True)

T0 = time.time()
UTC = timezone.utc
METRICS: dict = {
    "experiment": "x11_consolidation_ledger",
    "phase": "THE CONSOLIDATION LEDGER (P-x11a, T256) — is the expression "
             "edge a CONSOLIDATION-RATE threshold? final in-room write mass "
             "/ cumulative applied in-room displacement per committed rung, "
             "across e272's relocated edge (1k,2k]; CPU-only desk cell, "
             "committed records only",
    "date": datetime.now(UTC).isoformat(timespec="seconds"),
    "status": "RUNNING (progressive writes)",
    "cpu_only": True,
    "threads": {"torch": 4, "pocketfft": DCT_WORKERS},
}
BARS = {
    "RATIO-CLIFF": "the ratio jumps >= 3x at the (1k,2k] edge — the floor "
        "is a rate threshold; P-x11a fires",
    "RATIO-FLAT": "no jump — the edge is absolute-scale; the dose acquittal "
        "becomes the puzzle",
    "RATIO-INVERTED": "below-edge rooms consolidate BETTER per unit "
        "displacement — a pure decay-side threshold",
    "MIXED": "the families verbatim, no inflation",
}
OP = {
    "numerator": "||P_room dW||_2 fp64 CPU, dW = flat(inst_resume s400) - "
                 "flat(e001), P the SRCT projector (x6's convention)",
    "denominator": "sum_{s=1..400} lr(s) x gpn(s); lr(s) = 1e-3 x lr_scale x "
                   "cosine_lr(s-1, 1000) (house schedule ported BY VALUE, "
                   "warmup 100); gpn piecewise-LINEAR over the committed "
                   "41-point ledger {1,10,...,400}; lr_scale 1.0 except K1KM "
                   "3.7305567687315575",
    "proxy_form_disclosure": "first-order lr x ||g'|| — Adam's per-coordinate "
        "preconditioner and AdamW's decoupled weight decay UNMODELED (e272's "
        "dose_control caveat verbatim: 'the proxy is the gradient-norm "
        "channel; the displacement norms are the state channel'); used as a "
        "consistent cross-rung currency; hold-left sensitivity co-reported",
    "k10k_resume": "K10K's ledger is e264's committed resume-completed "
                   "record (e261 cut at s278; e264 pass-1 completed to s400; "
                   "the journal-resume design) covering s1..s400 whole",
    "verdict": "J := r(K2K)/r(K1K); CLIFF iff J >= 3; INVERTED iff J < 1; "
               "FLAT iff 1 <= J < 3; MIXED overrides on (a) above-edge "
               "family max/min > 3, or (b) r(K1KM) > r(K2K), or (c) the "
               "primary word flips under hold-left interpolation",
    "coreports": "K1KM (dead at matched kept-dose, lr x3.7306) and K10KR "
                 "(fresh-room 10k replicate) enter only through the MIXED "
                 "clauses, never as the primary edge read",
}


def log(msg: str) -> None:
    print(f"[x11 {time.time() - T0:7.1f}s] {msg}", flush=True)


def _strip_private(obj):
    if isinstance(obj, dict):
        return {k: _strip_private(v) for k, v in obj.items()
                if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_strip_private(v) for v in obj]
    return obj


def write_partial(note: str) -> None:
    METRICS["phase_note"] = note
    METRICS["date_updated"] = datetime.now(UTC).isoformat(timespec="seconds")
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


# ---------------------------------------------------------- G_ENDPOINTS
P = {  # every file this cell touches
    "e001": CKPT / "e001.pt",
    "e261_K1K": CKPT / "e261_K1K_inst_resume.pt",
    "e272_K1KM": CKPT / "e272_K1KM_inst_resume.pt",
    "e272_K2K": CKPT / "e272_K2K_inst_resume.pt",
    "e272_K5K": CKPT / "e272_K5K_inst_resume.pt",
    "e261_K10K": CKPT / "e261_K10K_inst_resume.pt",
    "e272_K10KR": CKPT / "e272_K10KR_inst_resume.pt",
    "e261_K237K": CKPT / "e261_K237K_inst_resume.pt",
    "e261_rooms": CKPT / "e261_rooms.pt",
    "e264_rooms": CKPT / "e264_rooms.pt",
    "e272_rooms": CKPT / "e272_rooms.pt",
    "e261_metrics": REPO / "runs" / "e261" / "metrics.json",
    "e264_metrics": REPO / "runs" / "e264" / "metrics.json",
    "e272_metrics": REPO / "runs" / "e272" / "metrics.json",
    "x10_metrics": REPO / "runs" / "x10" / "metrics.json",
}
# every claim below is recorded in x10's COMMITTED record (runs/x10/
# metrics.json G_ENDPOINTS, itself md5-bound here) — hard gates, all 15
COMMITTED = {
    "e001": "d114536d1c0983ab3be67f67ff0667c8",
    "e261_K1K": "f2975890c7870480d7f48b74dee33041",
    "e272_K1KM": "2306a448b450f192ef2d7d1f9aa90e2e",
    "e272_K2K": "f31b437cb81ae6f06aa35daceb3b4d67",
    "e272_K5K": "62049b93c2435639a945ce80cbc46470",
    "e261_K10K": "0f6dc1cf46850ce655dfafc9c853d467",
    "e272_K10KR": "6ce712a7aae8482a4c78d1b87f3194b1",
    "e261_K237K": "458e5211ef61aae7e14cd3b966f23dfc",
    "e261_rooms": "f81571c40bb6009d4ed1cfab66e73f6c",
    "e264_rooms": "2d524655575cce00a3bc1c8770f4b211",
    "e272_rooms": "066944855b3295e8796c6ca28b2e498c",
    "e261_metrics": "f460475d8e6b76f0719e91c1e9c6041b",
    "e264_metrics": "a42ff4786784b04cb9819a69b545e343",
    "e272_metrics": "eb9f624708bcb6576c4115161dfd7042",
    "x10_metrics": "2d1cec68a0e7d471fd991e3693913d01",  # fresh bind (this cell)
}
md5_reads = {}
for nm, path in P.items():
    got = md5_of(path)
    claim = COMMITTED[nm]
    md5_reads[nm] = {"path": str(path.relative_to(REPO)), "md5": got,
                     "claimed": claim,
                     "source": "x10's committed G_ENDPOINTS table"
                               if nm != "x10_metrics" else "FRESH BIND (this cell)",
                     "match": got == claim}
    if got != claim:
        raise SystemExit(f"ENDPOINT BIND FAILURE: {nm} md5 {got} != {claim}")
METRICS["gates"] = {"G_ENDPOINTS": {
    "md5_binds": md5_reads,
    "note": "all 15 files hard-gated against x10's committed table (which "
            "carries x6's original binds + x10's fresh binds); "
            "runs/x10/metrics.json itself fresh-bound here for future cells",
    "pass": True}}
write_partial("P0 bars frozen in-script + all 15 endpoints md5-bound")


# ---------------------------------------------------------- loaders (CPU)
def load_flat(path: Path) -> tuple[np.ndarray, list[str], dict]:
    """Plain-IO CPU load (x10's loader verbatim: no memmap; .float().clone();
    finiteness verified). Returns (flat fp64, key order, meta)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    assert isinstance(sd, dict) and all(torch.is_tensor(v) for v in sd.values())
    keys = list(sd.keys())
    tensors = [sd[k].float().clone() for k in keys]          # stale-page guard
    for k, v in zip(keys, tensors):
        if not torch.isfinite(v).all():
            raise SystemExit(f"NON-FINITE tensor {k} in {path.name}")
    flat = torch.cat([v.reshape(-1) for v in tensors]).double().numpy()
    meta = ({kk: ck[kk] for kk in ("step", "n_chunks") if kk in ck}
            if isinstance(ck, dict) else {})
    return flat.astype(np.float64), keys, meta


def g_flatbasis(keys: list[str]) -> dict:
    """state_dict key order == net.parameters() order (CPU instantiation of
    lab/common.py's TinyGPT; DEVICE only — CUDA_VISIBLE_DEVICES='' keeps
    every code path CPU)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "x11_common", REPO / "lab" / "common.py")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["x11_common"] = mod      # dataclass needs a registered module
    spec.loader.exec_module(mod)
    net = mod.TinyGPT(mod.Cfg())
    pnames = [n for n, _ in net.named_parameters()]
    bnames = [n for n, _ in net.named_buffers()]
    return {"params_count": len(pnames), "buffers_count": len(bnames),
            "key_order_matches_parameters": pnames == keys,
            "n_params": sum(p.numel() for p in net.parameters()),
            "pass": pnames == keys and len(bnames) == 0}


flat_root, KEYS, _ = load_flat(P["e001"])
GB = g_flatbasis(KEYS)
if not GB["pass"]:
    raise SystemExit(f"FLAT-BASIS GATE FAILURE: {GB}")
N = int(flat_root.size)
assert N == 2_739_072, N
METRICS["gates"]["G_FLATBASIS"] = GB
write_partial("P1 root loaded + flat basis verified")


# ---------------------------------------------------------- the room (SRCT)
class SRCT:
    """Ported BY VALUE via x10 (which ported from x6/e261) — the exact
    orthogonal projector P x = D . idct(mask_S(dct(D . x))), fp64 CPU."""

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


def load_room(path: Path, rung: str, seeds_expect: list[int]) -> SRCT:
    """Load a stored room and BIT-VERIFY vs fresh reconstruction from its
    committed seeds (x6/x10's precedent)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    entry = ck["model"][rung]
    k, seeds = int(entry["k"]), [int(x) for x in entry["seeds"]]
    assert seeds == seeds_expect, (rung, seeds, seeds_expect)
    fresh = SRCT(N, k, seeds[0], seeds[1])
    _as_np = lambda t: t.numpy() if hasattr(t, "numpy") else np.asarray(t)
    D_stored = _as_np(entry["D_int8"]).astype(np.float64)
    S_stored = _as_np(entry["S"])
    assert np.array_equal(D_stored, fresh.D), f"room {rung} D mismatch"
    assert np.array_equal(S_stored, fresh.S), f"room {rung} S mismatch"
    return fresh


ROOMS = {
    "K1K": (P["e261_rooms"], "K1K", [26111, 26112]),
    "K1KM": (P["e272_rooms"], "K1KM", [26111, 26112]),
    "K2K": (P["e272_rooms"], "K2K", [27211, 27212]),
    "K5K": (P["e272_rooms"], "K5K", [27213, 27214]),
    "K10K": (P["e264_rooms"], "K10K", [26113, 26114]),
    "K10KR": (P["e272_rooms"], "K10KR", [27215, 27216]),
    "K237K": (P["e261_rooms"], "K237K", [26011, 26012]),
}
rooms: dict[str, SRCT] = {}
room_gate = {}
for nm, (path, rung, seeds) in ROOMS.items():
    rooms[nm] = load_room(path, rung, seeds)
    room_gate[nm] = {"file": str(path.relative_to(REPO)), "k": rooms[nm].k,
                     "seeds": seeds, "bit_verified_vs_fresh": True}
assert np.array_equal(rooms["K1KM"].D, rooms["K1K"].D), "K1KM room != K1K room"
assert np.array_equal(rooms["K1KM"].S, rooms["K1K"].S), "K1KM room != K1K room"
room_gate["K1KM_bit_equals_committed_K1K"] = True
METRICS["gates"]["G_ROOMS"] = room_gate
write_partial("P2 all 7 rooms loaded + bit-verified vs committed seeds")


# ---------------------------------------------------------- the ledgers
def cosine_lr(step: int, total: int, warmup: int = 100) -> float:
    """lab/common.py's house schedule, ported BY VALUE (frozen at birth)."""
    if step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


LR_BASE = 1e-3                      # E43.LR (e272's G_DOSE_ARITH bind)
INST_TOTAL = 1000                   # e048's house-cosine total (e261)
INST_STEPS = 400

RUNGS = {  # rung -> (ckpt key, room name, ledger source, lr_scale)
    "K1K": ("e261_K1K", "K1K", ("e261_metrics", "K1K"), 1.0),
    "K1KM": ("e272_K1KM", "K1KM", ("e272_metrics", "K1KM"), 3.7305567687315575),
    "K2K": ("e272_K2K", "K2K", ("e272_metrics", "K2K"), 1.0),
    "K5K": ("e272_K5K", "K5K", ("e272_metrics", "K5K"), 1.0),
    "K10K": ("e261_K10K", "K10K", ("e264_metrics", "K10K"), 1.0),
    "K10KR": ("e272_K10KR", "K10KR", ("e272_metrics", "K10KR"), 1.0),
    "K237K": ("e261_K237K", "K237K", ("e261_metrics", "K237K"), 1.0),
}
# committed e272 applied_dose_proxy_median values (the G_DOSE-REPRO targets)
DOSE_REPRO = {
    "K2K": 1.925611105888784e-05,
    "K5K": 3.0349046917670057e-05,
    "K10KR": 4.384148621663227e-05,
    "K1KM": 4.903974292104162e-05,
}
# committed in_room_disp_norm (the G_DISP-NORM targets; e272 arms only)
DISP_NORM = {
    "K2K": 6.521311124475332,
    "K5K": 7.3312512540040835,
    "K10KR": 8.701848935447199,
    "K1KM": 11.95964168990308,
}
# committed in_own_room NORM ratios (the G_DISPL content-bind targets)
IN_OWN = {
    "K1K": (0.8990001680413359, "e261 metrics install displacement_loads"),
    "K1KM": (0.7955550758680459, "e272 metrics install displacement_loads"),
    "K2K": (0.9090426956346246, "e272 metrics install displacement_loads"),
    "K5K": (0.9244495344843363, "e272 metrics install displacement_loads"),
    "K10K": (0.9441588788935659, "e264 metrics install displacement_loads"),
    "K10KR": (0.9445320495247138, "e272 metrics install displacement_loads"),
    "K237K": (0.9804893255327712, "e261 metrics install displacement_loads"),
}

# load the three parent metrics records (already md5-gated above)
MET = {nm: json.loads(P[nm].read_text(encoding="utf-8"))
       for nm in ("e261_metrics", "e264_metrics", "e272_metrics")}

# G_LEDGER: structure + the per-step proxy reconstruction
STEPS_ALL = list(range(1, INST_STEPS + 1))
ledger_gate: dict[str, dict] = {}
dose_repro: dict[str, dict] = {}
LEDGER_DATA: dict[str, dict[int, float]] = {}   # rung -> {step: gpn}
for rung, (_, _, (mkey, arm), lr_scale) in RUNGS.items():
    inst = MET[mkey]["arms"][arm]["install"]
    led = inst["ledger"]
    steps = sorted(int(s) for s in led.keys())
    if steps != [1] + list(range(10, INST_STEPS + 1, 10)):
        raise SystemExit(f"LEDGER STRUCTURE FAILURE {rung}: {steps[:5]}...")
    gp = {s: float(led[str(s)]["gpn"]) for s in steps}
    for s, v in gp.items():
        if not math.isfinite(v) or v <= 0.0:
            raise SystemExit(f"LEDGER gpn not finite/positive: {rung} s{s}")
    LEDGER_DATA[rung] = gp
    lr_mult_committed = inst.get("lr_scale_applied")
    ledger_gate[rung] = {
        "ledger_steps": len(steps), "gpn_min": min(gp.values()),
        "gpn_max": max(gp.values()),
        "committed_lr_scale": lr_mult_committed,
        "lr_scale_this_cell": lr_scale,
        "lr_scale_consistent": (lr_mult_committed is None
                                or abs(lr_mult_committed - lr_scale) < 1e-12),
        "committed_median_gpn": float(np.median(list(gp.values()))),
        "resumed_final": bool(inst.get("resumed_final", False)),
    }
    if not ledger_gate[rung]["lr_scale_consistent"]:
        raise SystemExit(f"LR-SCALE MISMATCH {rung}: {lr_mult_committed} vs {lr_scale}")
    # e272's committed proxy reproduction: median over LEDGERED steps only
    if rung in DOSE_REPRO:
        prox = [LR_BASE * lr_scale * cosine_lr(s - 1, INST_TOTAL) * gp[s]
                for s in steps]
        med = float(np.median(prox))
        tgt = DOSE_REPRO[rung]
        rel = abs(med - tgt) / abs(tgt)
        dose_repro[rung] = {"this_cell_median": med, "committed": tgt,
                            "rel_diff": rel, "pass": rel <= 1e-9}
        if rel > 1e-9:
            raise SystemExit(f"DOSE-PROXY REPRO FAILURE {rung}: "
                             f"{med!r} vs committed {tgt!r} (rel {rel:.2e})")
METRICS["gates"]["G_LEDGER"] = ledger_gate
METRICS["gates"]["G_DOSE_REPRO"] = {
    **dose_repro, "tol_rel": 1e-9,
    "form": "median over the 41 ledgered steps of lr(s) x gpn(s) == the "
            "parent's committed applied_dose_proxy_median (e272 arms)",
    "pass": all(g["pass"] for g in dose_repro.values())}
write_partial("P3 ledgers loaded; G_LEDGER + G_DOSE_REPRO PASS "
              "(lr schedule + gpn channel reproduce e272's committed proxies)")


def integrate_dose(rung: str, lr_scale: float, mode: str = "linear") -> float:
    """Cumulative applied in-room displacement: sum_s lr(s) x gpn(s), gpn
    interpolated over the 41 committed points (linear primary / hold-left
    sensitivity)."""
    gp = LEDGER_DATA[rung]
    xs = np.array(sorted(gp.keys()), dtype=np.float64)
    ys = np.array([gp[int(s)] for s in xs], dtype=np.float64)
    ss = np.array(STEPS_ALL, dtype=np.float64)
    if mode == "linear":
        g = np.interp(ss, xs, ys)
    elif mode == "holdleft":
        idx = np.searchsorted(xs, ss, side="right") - 1
        g = ys[np.clip(idx, 0, len(xs) - 1)]
    else:
        raise ValueError(mode)
    lr = np.array([LR_BASE * lr_scale * cosine_lr(int(s) - 1, INST_TOTAL)
                   for s in STEPS_ALL])
    per_step = lr * g
    if not np.all(np.isfinite(per_step)):
        raise SystemExit(f"NON-FINITE per-step dose: {rung}")
    return float(per_step.sum()), per_step


# ---------------------------------------------------------- the reads
READS: dict[str, dict] = {}
displ_gate: dict[str, dict] = {}
dispnorm_gate: dict[str, dict] = {}
for rung, (ck_key, room_nm, _, lr_scale) in RUNGS.items():
    flat_post, _, meta = load_flat(P[ck_key])
    assert meta.get("step") == INST_STEPS, (rung, meta)
    dw = flat_post - flat_root
    pd = rooms[room_nm].project(dw)
    num = float(np.linalg.norm(pd))
    tot = float(np.linalg.norm(dw))
    inside = num * num / (tot * tot)
    D_lin, per_step = integrate_dose(rung, lr_scale, "linear")
    D_hold, _ = integrate_dose(rung, lr_scale, "holdleft")
    if not all(math.isfinite(v) for v in (num, tot, D_lin, D_hold)) or D_lin <= 0:
        raise SystemExit(f"NON-FINITE/DEGENERATE read: {rung}")
    READS[rung] = {
        "k": rooms[room_nm].k,
        "vehicle": f"runs/checkpoints/{P[ck_key].name} (s400; "
                   f"md5 {COMMITTED[ck_key]})",
        "room": f"{ROOMS[room_nm][0].name}:{room_nm} seeds "
                f"{ROOMS[room_nm][2]} (bit-verified)",
        "dW_l2": tot,
        "numerator_in_room_mass": num,
        "denominator_cum_applied_linear": D_lin,
        "denominator_cum_applied_holdleft": D_hold,
        "consolidation_ratio_linear": num / D_lin,
        "consolidation_ratio_holdleft": num / D_hold,
        "inside_share": inside,
        "lr_scale": lr_scale,
        "_per_step": per_step,
    }
    tgt, src = IN_OWN[rung]
    displ_gate[rung] = {"committed_in_own_room_norm_ratio": tgt,
                        "squared": tgt * tgt, "this_cell_inside_share": inside,
                        "abs_diff": abs(inside - tgt * tgt), "source": src,
                        "pass": abs(inside - tgt * tgt) <= 2e-3}
    if rung in DISP_NORM:
        tgt2 = DISP_NORM[rung]
        rel = abs(num - tgt2) / tgt2
        dispnorm_gate[rung] = {"committed_in_room_disp_norm": tgt2,
                               "this_cell": num, "rel_diff": rel,
                               "pass": rel <= 5e-3}
METRICS["gates"]["G_DISPL"] = {**displ_gate, "tol": 2e-3,
    "form": "committed in_own_room (norm ratio)^2 == this cell's fp64 "
            "inside_share", "pass": all(g["pass"] for g in displ_gate.values())}
METRICS["gates"]["G_DISP_NORM"] = {**dispnorm_gate, "tol_rel": 5e-3,
    "form": "this cell's ||P dW|| == e272's committed in_room_disp_norm "
            "(e272 arms)", "pass": all(g["pass"] for g in dispnorm_gate.values())}
if not METRICS["gates"]["G_DISPL"]["pass"]:
    raise SystemExit(f"DISPLACEMENT CONTENT BIND FAILURE: {displ_gate}")
if not METRICS["gates"]["G_DISP_NORM"]["pass"]:
    raise SystemExit(f"DISP-NORM GATE FAILURE: {dispnorm_gate}")
write_partial("P4 all 7 rungs read; G_DISPL + G_DISP_NORM PASS")


# ---------------------------------------------------------- the ledger table
def pub(rr: dict) -> dict:
    return {"k": rr["k"], "lr_scale": rr["lr_scale"],
            "dW_l2": rr["dW_l2"],
            "numerator_in_room_mass": rr["numerator_in_room_mass"],
            "denominator_cum_applied": rr["denominator_cum_applied_linear"],
            "denominator_cum_applied_holdleft":
                rr["denominator_cum_applied_holdleft"],
            "consolidation_ratio": rr["consolidation_ratio_linear"],
            "consolidation_ratio_holdleft":
                rr["consolidation_ratio_holdleft"],
            "inside_share": rr["inside_share"]}


TABLE = {rung: pub(READS[rung]) for rung in
         ("K1K", "K1KM", "K2K", "K5K", "K10K", "K10KR", "K237K")}
POST_G0 = {"K1K": 0.00043458465370349586, "K1KM": 0.0012453667586669326,
           "K2K": 0.026616254821419716, "K5K": 0.12709596753120422,
           "K10K": 0.26464763283729553, "K10KR": 0.2097209095954895,
           "K237K": 0.38436421751976013}
for rung in TABLE:
    TABLE[rung]["post_g0_cited"] = POST_G0[rung]
METRICS["ledger"] = TABLE
write_partial("P5 consolidation ledger computed (7 rungs)")


# ---------------------------------------------------------- adjudication
r = {rung: TABLE[rung]["consolidation_ratio"] for rung in TABLE}
J = r["K2K"] / r["K1K"]
above = ["K2K", "K5K", "K10K", "K10KR", "K237K"]
spread_above = max(r[x] for x in above) / min(r[x] for x in above)
clause_a = spread_above > 3.0
clause_b = r["K1KM"] > r["K2K"]
if J >= 3.0:
    primary = "RATIO-CLIFF"
elif J < 1.0:
    primary = "RATIO-INVERTED"
else:
    primary = "RATIO-FLAT"
# hold-left sensitivity on the primary word
rh = {rung: TABLE[rung]["consolidation_ratio_holdleft"] for rung in TABLE}
Jh = rh["K2K"] / rh["K1K"]
primary_h = ("RATIO-CLIFF" if Jh >= 3.0
             else "RATIO-INVERTED" if Jh < 1.0 else "RATIO-FLAT")
clause_c = primary_h != primary
verdict = "MIXED" if (clause_a or clause_b or clause_c) else primary
METRICS["verdict"] = {
    "word": verdict,
    "primary_word_from_edge": primary,
    "edge_jump_J_ratio_2k_over_1k": J,
    "edge_jump_holdleft": Jh,
    "clause_trace": {
        "cliff_J_ge_3": J >= 3.0,
        "inverted_J_lt_1": J < 1.0,
        "flat_1_le_J_lt_3": 1.0 <= J < 3.0,
        "mixed_a_above_family_spread_gt_3": clause_a,
        "above_family_spread": spread_above,
        "mixed_b_K1KM_ratio_gt_K2K": clause_b,
        "mixed_c_holdleft_flips_primary": clause_c,
    },
    "ratios": r,
    "bars_verbatim": BARS,
    "operationalizations": OP,
}
write_partial(f"P6 ADJUDICATED: {verdict} (primary {primary}, J={J:.3f})")


# ---------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["K1K", "K1KM", "K2K", "K5K", "K10K", "K10KR", "K237K"]
ks = [TABLE[x]["k"] for x in ORDER]
ratios = [TABLE[x]["consolidation_ratio"] for x in ORDER]
nums = [TABLE[x]["numerator_in_room_mass"] for x in ORDER]
dens = [TABLE[x]["denominator_cum_applied"] for x in ORDER]
posts = [POST_G0[x] for x in ORDER]

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

ax = axes[0]
edge_x = math.sqrt(TABLE["K1K"]["k"] * TABLE["K2K"]["k"])
ax.axvline(edge_x, color="black", ls="--", lw=1.2, alpha=0.8,
           label="the (1k,2k] expression edge")
ax.plot(TABLE["K1K"]["k"], r["K1K"], "o", color="crimson", ms=10, zorder=5,
        label=f"K1K (DEAD) ratio {r['K1K']:.1f}")
ax.plot(TABLE["K2K"]["k"], r["K2K"], "o", color="darkgreen", ms=10, zorder=5,
        label=f"K2K (first above edge) ratio {r['K2K']:.1f}")
ax.plot(ks, ratios, "o-", color="steelblue", lw=1.2, ms=5, alpha=0.55,
        label="full ladder (ratio)")
ax.plot(TABLE["K1KM"]["k"], r["K1KM"], "s", mfc="none", color="purple",
        ms=10, label=f"K1KM co-report (dead, lr x3.73) {r['K1KM']:.1f}")
ax.plot(TABLE["K10KR"]["k"], r["K10KR"], "s", mfc="none", color="teal",
        ms=10, label=f"K10KR co-report {r['K10KR']:.1f}")
for x in ORDER:
    ax.annotate(TABLE[x]["k"], (TABLE[x]["k"], r[x]),
                textcoords="offset points", xytext=(0, -16), fontsize=7,
                ha="center", color="gray")
ax.annotate(f"J = r(2k)/r(1k) = {J:.2f}", (edge_x, max(ratios) * 0.9),
            fontsize=10, ha="center",
            bbox=dict(boxstyle="round", fc="lightyellow", ec="black", alpha=0.9))
ax.set_xscale("log")
ax.set_xlabel("room width k (log)")
ax.set_ylabel("consolidation ratio (final in-room mass / cum. applied)")
ax.set_title(f"(a) THE CONSOLIDATION RATIO ACROSS THE EDGE\n"
             f"verdict: {verdict} (primary {primary}; cliff bar J >= 3)")
ax.legend(fontsize=7.5, loc="best")

ax = axes[1]
xp = np.arange(len(ORDER))
ax.bar(xp - 0.18, nums, width=0.36, color="crimson", alpha=0.85,
       label="numerator ||P dW|| (what stayed)")
ax.bar(xp + 0.18, dens, width=0.36, color="navy", alpha=0.85,
       label="denominator cum. lr x ||g'|| (what was pushed)")
for i, (n_, d_) in enumerate(zip(nums, dens)):
    ax.annotate(f"r={n_/d_:.0f}", (i, max(n_, d_)), ha="center",
                textcoords="offset points", xytext=(0, 4), fontsize=8)
ax.set_yscale("log")
ax.set_xticks(xp, ORDER, fontsize=8)
ax.set_ylabel("mass (log scale; the units differ by the disclosed proxy)")
ax.set_title("(b) NUMERATOR vs DENOMINATOR per rung\n"
             "(first-order dose proxy — Adam unmodeled, disclosed)")

ax = axes[2]
ax.plot(ratios, posts, "o", color="black", ms=8)
for x, rr_, pp_ in zip(ORDER, ratios, posts):
    ax.annotate(x, (rr_, pp_), textcoords="offset points", xytext=(6, 4),
                fontsize=8)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("consolidation ratio (log)")
ax.set_ylabel("post g0 expression (log, cited)")
ax.set_title("(c) CONSOLIDATION vs EXPRESSION\n(does the rate predict the fate?)")

fig.suptitle(f"x11 THE CONSOLIDATION LEDGER — verdict: {verdict} "
             f"(J = {J:.3f}; ratios 1k {r['K1K']:.1f} / 2k {r['K2K']:.1f} / "
             f"5k {r['K5K']:.1f} / 10k {r['K10K']:.1f} / 237k {r['K237K']:.1f})",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "x11_consolidation.png", dpi=140)
log("FIGURE written")
METRICS["outputs"] = {"figure": "runs/x11/x11_consolidation.png",
                      "metrics": "runs/x11/metrics.json",
                      "report": "runs/x11/REPORT.md"}
write_partial("P7 figure written")


# ---------------------------------------------------------- REPORT.md
rows = []
for x in ORDER:
    t = TABLE[x]
    role = ("DEAD (primary below-edge)" if x == "K1K" else
            "dead, kept-matched lr x3.73 (co-report)" if x == "K1KM" else
            "first above edge" if x == "K2K" else
            "10k replicate (co-report)" if x == "K10KR" else "")
    rows.append(
        f"| {x} | {t['k']:,} | {role} | {POST_G0[x]:.6f} | "
        f"{t['numerator_in_room_mass']:.4f} | "
        f"{t['denominator_cum_applied']:.6e} | "
        f"**{t['consolidation_ratio']:.2f}** | "
        f"{t['consolidation_ratio_holdleft']:.2f} | "
        f"{t['inside_share']:.4f} |")
vt = METRICS["verdict"]["clause_trace"]
report = f"""# x11 — THE CONSOLIDATION LEDGER ({verdict})

**The registered question (P-x11a, T256):** is the expression edge a
CONSOLIDATION-RATE threshold? The consolidation ratio := final in-room
write mass ||P_room dW|| (inst_resume s400 checkpoints, own-room fp64
projection) / cumulative applied in-room displacement (the integrated
per-step lr x ||g'|| channel of the committed install ledgers) — how much
of what was pushed STAYED, per rung.

## The consolidation ledger (7 committed rungs)

| rung | k | role | post g0 (cited) | num ||P dW|| | cum. applied (proxy) | RATIO | ratio (hold-left) | inside share |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

**The edge read:** J = r(K2K)/r(K1K) = **{J:.4f}** (hold-left {Jh:.4f}).

## Verdict: {verdict}

Clause trace: cliff (J >= 3) = {vt['cliff_J_ge_3']}; inverted (J < 1) =
{vt['inverted_J_lt_1']}; flat (1 <= J < 3) = {vt['flat_1_le_J_lt_3']};
MIXED clause (a) above-edge family spread {spread_above:.3f} > 3 =
{vt['mixed_a_above_family_spread_gt_3']}; clause (b) r(K1KM) >
r(K2K) = {vt['mixed_b_K1KM_ratio_gt_K2K']} (r(K1KM) = {r['K1KM']:.2f});
clause (c) hold-left flips the primary word = {vt['mixed_c_holdleft_flips_primary']}
(primary {primary} -> {primary_h}).

## Gates

- G_ENDPOINTS: all 15 touched files md5-gated against x10's committed
  table (which carries x6's original binds); runs/x10/metrics.json
  fresh-bound ({COMMITTED['x10_metrics']}).
- G_ROOMS: all 7 room entries bit-verified vs fresh committed-seed
  reconstruction; K1KM room re-gated bit-equal to the committed K1K room.
- G_LEDGER + G_DOSE_REPRO: every ledger covers steps
  {{1,10,...,400}} with finite positive gpn; this cell's reconstruction of
  the per-step proxy reproduces e272's committed
  applied_dose_proxy_median to <= 1e-9 relative on all four e272 arms —
  the lr schedule + gpn channel are the parent's, exactly.
- G_DISPL (content bind, x10's precedent): every rung's fp64 inside_share
  matches the committed in_own_room norm ratio squared within 2e-3
  (max abs diff {max(g['abs_diff'] for g in displ_gate.values()):.2e}).
- G_DISP_NORM: the four e272 numerators match the committed
  in_room_disp_norm within 5e-3 relative
  (max rel diff {max(g['rel_diff'] for g in dispnorm_gate.values()):.2e}).
- G_FLATBASIS: key order == TinyGPT parameters, no buffers, N=2,739,072.

## Disclosures

1. THE PROXY FORM: the denominator is the first-order applied
   displacement lr(s) x ||g'(s)|| summed over 400 steps — Adam's
   per-coordinate preconditioner and AdamW's decoupled weight decay are
   unmodeled (e272's own dose_control caveat). It is used as a consistent
   cross-rung currency; the RATIO's absolute scale is not interpretable,
   only its cross-rung structure is. Hold-left interpolation sensitivity
   co-reported (clause c).
2. gpn between the 41 committed ledger points is piecewise-LINEAR
   interpolated (primary); every ledger point is a committed fp64 value.
3. K10K's ledger is e264's committed resume-completed record (e261's s278
   cut completed by e264's pass 1; the journal-resume design) — it covers
   s1..s400 whole.
4. K1KM's install ran at lr x3.7305567687315575 (the kept-matched
   compensation; lr_scale asserted against e272's committed
   lr_scale_applied); its denominator uses that scale.
5. The numerator includes whatever re-growth inside the room the install
   produced (projection is orthogonal; the out-of-room drift is excluded
   by construction).
6. CPU-only: threads 4/4, no envelope-log writes, no GPU code path; all
   timestamps datetime.now(UTC).
"""
(OUT / "REPORT.md").write_text(report, encoding="utf-8")

METRICS["status"] = "COMPLETE — adjudicated"
write_partial("P8 COMPLETE (report + figure)")
log(f"DONE — verdict {verdict}; J={J:.4f}; ratios " +
    " ".join(f"{x}:{r[x]:.1f}" for x in ORDER))
