"""
x6 — THE SVD BET (agy consult #005, item 5) — a CPU-ONLY DESK CELL.

The lab's "capacity number" says a fact needs ~10k of 2.74M dims to express
at all (the 609x expression cliff between the k=1k and k=10k rungs:
e261 K1K post g0 0.000435 -> e264 K10K post g0 0.2646). The consult bets
the cliff is a SEARCH bottleneck, not a storage limit: the quiet 10k write's
parameter delta dW should be effectively LOW-RANK. This cell takes or denies
that bet, from committed records only — no GPU, no training, no re-runs.

BARS — FROZEN VERBATIM FROM THE DISPATCH, BEFORE ANY COMPUTE (no bar shopping):

  SEARCH-BOTTLENECK (agy wins the wording): top-50 singular energy >= 99%
      in every parameter matrix carrying mass AND the natural-basis
      top-|coords| effective dim at 99% energy <= 500 — the write is
      effectively low-rank; "capacity number" must be re-worded as
      search-room.
  GENUINELY-SPREAD (the storage reading stands): top-50 energy < 90% or
      the 99%-energy dimensionality >= 5,000 — the write truly occupies
      ~10k dims.
  MIXED: anything else — the spectra verbatim, both bases, no inflation.

OPERATIONALIZATIONS — FROZEN BEFORE COMPUTE:
  * THE OBJECT: dW_10k := flat(e261_K10K_inst_resume.pt["model"] at its
    final step s400) MINUS flat(e001.pt) — the same fresh root every ladder
    arm installed from (e261/e264's registered convention). Both endpoints
    md5-bound to committed records BEFORE trust (the OneDrive honesty rule:
    plain torch.load map_location='cpu', .float().clone(), finiteness
    verified; NO memmap anywhere).
  * The flat basis IS net.parameters() order; TinyGPT (lab/common.py)
    registers NO buffers, so state_dict key order == parameters order
    (verified live against a CPU instantiation). N = 2,739,072 (the
    dispatch letter's "2,738,880" is a transcription slip — G_BASE/G_ROOT
    and the checkpoints themselves say 2,739,072; disclosed, never silently
    "corrected").
  * (a) NATURAL BASIS: |dW| sorted desc; m(th) := smallest m with top-m
    |coords| carrying >= th of ||dW||^2, th in {50, 90, 99}% (all params,
    1D included).
  * (b) THE BET'S OWN FORM: per-2D-parameter-matrix SVD (fp64);
    per-matrix top-min(50, r_j) energy share; per-matrix rank crossing
    90/95/99% (smallest integer rank; "interpolate" := read off the
    cumulative singular spectrum); GLOBAL top-50 share T50 := sum_j
    top-50 energy_j / sum_j energy_j over 2D params. 1D params (LN
    gains/biases, mlp biases): SVD undefined — energy share disclosed,
    never silently dropped (the natural read covers them).
  * "carrying mass" := a 2D param whose dW energy share of the TOTAL
    (2D+1D) is >= 1e-3 (0.1%).
  * VERDICT QUANTITIES: SEARCH-BOTTLENECK iff (per-matrix top-50 share
    >= 0.99 for EVERY mass-carrying 2D matrix) AND m99 <= 500.
    GENUINELY-SPREAD iff T50 < 0.90 OR m99 >= 5000. MIXED otherwise.
  * (c) THE ROOM'S READING: dW projected onto the committed 10k room's
    basis — the SRCT projector P x = D . idct(mask_S(dct(D . x))) in
    fp64 on CPU (lab/e261_rank_ladder.py's class, ported by value, never
    imported — this cell touches no GPU code path). The K10K room is
    loaded from runs/checkpoints/e264_rooms.pt (seeds 26113/26114 — the
    registered seeds; e261_rooms.pt holds only the triaged {K1K, K237K}
    rungs, disclosed) AND bit-verified against the fresh seed
    reconstruction (exact D/S equality — stronger than an md5). Inside
    share := ||P dW||^2 / ||dW||^2. The room file stores NO
    eigen-spectrum (none exists: an SRCT room is an equal-energy
    orthonormal basis by construction) — the informative spectrum is
    dW's IN-ROOM coefficient energy |c_s|^2 over the k selected
    frequencies, sorted desc, with its own top-50 share + m(50/90/99).
  * (d) ANCHORS: (i) NULL — a gaussian random direction, norm-matched to
    ||dW_10k||, frozen seed 26001 (x6's registration); the same three
    reads. (ii) The committed 237k rung's dW (e261_K237K_inst_resume.pt
    at s400 minus the same e001 root), same reads against ITS OWN room
    (e261_rooms.pt K237K, seeds 26011/26012, bit-verified vs fresh
    reconstruction) — does the natural-width write carry MORE effective
    rank? (iii) CO-REPORT ONLY (never a bar): the e260-session 237k
    vehicle (e260_RANDOM_inst_resume.pt) — cross-session texture on the
    same arm.
  * CPU ENVELOPE: CUDA_VISIBLE_DEVICES="" before torch import;
    torch threads 4; pocketfft workers 4 (<= 8 total); nothing written to
    runs/_envelope_log.jsonl; no lab/gpu code path touched.
  * PROGRESSIVE writes to runs/x6/metrics.json; no NOTES/THINKING/QUEUE/
    STATE edits (dispatch — the coordinator folds).

Provenance (all verified BEFORE compute, below in G_ENDPOINTS):
  ROOT            runs/checkpoints/e001.pt            md5 d114536d1c0983ab3be67f67ff0667c8 (== e267's committed base bind)
  K10K POST s400  runs/checkpoints/e261_K10K_inst_resume.pt md5 0f6dc1cf46850ce655dfafc9c853d467 (== e264 G_K10KRESUME)
  K237K POST s400 runs/checkpoints/e261_K237K_inst_resume.pt md5 458e5211ef61aae7e14cd3b966f23dfc (== e271 vehicle_md5)
  E260 237K POST  runs/checkpoints/e260_RANDOM_inst_resume.pt (e261 G_ANCHOR's L2-compared object; texture co-report)
  ROOM 10k        runs/checkpoints/e264_rooms.pt      K10K seeds 26113/26114 (bit-verified vs fresh)
  ROOM 237k       runs/checkpoints/e261_rooms.pt      K237K seeds 26011/26012 (md5 f81571c40bb6009d4ed1cfab66e73f6c == e271's bind; bit-verified)
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
OUT = REPO / "runs" / "x6"
OUT.mkdir(parents=True, exist_ok=True)

T0 = time.time()
METRICS: dict = {
    "experiment": "x6_svd_bet",
    "phase": "THE SVD BET (agy #005 item 5) — the 10k quiet write's dW read in "
             "THREE bases (natural / per-matrix SVD / its own room) + anchors; "
             "CPU-only desk cell, committed records only",
    "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "status": "RUNNING (progressive writes)",
    "cpu_only": True,
    "threads": {"torch": 4, "pocketfft": DCT_WORKERS},
}
BARS = {
    "SEARCH-BOTTLENECK": "top-50 singular energy >= 99% in every parameter "
        "matrix carrying mass AND the natural-basis top-|coords| effective "
        "dim at 99% energy <= 500 — the write is effectively low-rank; "
        "'capacity number' must be re-worded as search-room",
    "GENUINELY-SPREAD": "top-50 energy < 90% or the 99%-energy dimensionality "
        ">= 5,000 — the write truly occupies ~10k dims",
    "MIXED": "anything else — the spectra verbatim, both bases, no inflation",
}
OP = {
    "object": "dW_10k = flat(e261_K10K_inst_resume.pt['model'] s400) - flat(e001.pt)",
    "flat_basis": "net.parameters() order == state_dict key order (TinyGPT has "
        "no buffers; verified live); N = 2,739,072 (dispatch's 2,738,880 is a "
        "slip — disclosed)",
    "mass_bar": "a 2D param carries mass iff its dW energy share of the total "
        "(2D+1D) >= 1e-3",
    "t50": "global energy-weighted top-50 singular share over 2D params",
    "verdict": "SEARCH-BOTTLENECK iff (per-matrix top-50 >= 0.99 for every "
        "mass-carrying 2D matrix) AND m99 <= 500; GENUINELY-SPREAD iff "
        "T50 < 0.90 OR m99 >= 5000; MIXED otherwise",
    "room_spectrum": "the room file stores no eigen-spectrum (an SRCT room is "
        "an equal-energy orthonormal basis); the read is dW's in-room "
        "coefficient energy spectrum",
    "null_seed": 26001,
    "anchors": "null (norm-matched gaussian, seed 26001) + dW_237k "
        "(e261_K237K_inst_resume s400 - e001, own room K237K) + e260-session "
        "237k vehicle as cross-session texture co-report (never a bar)",
}


def log(msg: str) -> None:
    print(f"[x6 {time.time() - T0:7.1f}s] {msg}", flush=True)


def _strip_private(obj):
    """Drop '_'-prefixed keys (big local-only arrays) for the JSON write."""
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


# ---------------------------------------------------------- G_ENDPOINTS
ROOT_CK = CKPT / "e001.pt"
K10K_CK = CKPT / "e261_K10K_inst_resume.pt"
K237K_CK = CKPT / "e261_K237K_inst_resume.pt"
E260_CK = CKPT / "e260_RANDOM_inst_resume.pt"
ROOMS264_CK = CKPT / "e264_rooms.pt"
ROOMS261_CK = CKPT / "e261_rooms.pt"

BINDS = {
    "e001": (ROOT_CK, "d114536d1c0983ab3be67f67ff0667c8",
             "e267's committed base bind"),
    "e261_K10K_inst_resume": (K10K_CK, "0f6dc1cf46850ce655dfafc9c853d467",
                              "e264 G_K10KRESUME frozen md5 (the s400 final)"),
    "e261_K237K_inst_resume": (K237K_CK, "458e5211ef61aae7e14cd3b966f23dfc",
                               "e271 k237k_vehicle vehicle_md5"),
    "e261_rooms": (ROOMS261_CK, "f81571c40bb6009d4ed1cfab66e73f6c",
                   "e271's e261_rooms bind"),
}
md5_reads = {}
for nm, (path, claim, src) in BINDS.items():
    got = md5_of(path)
    md5_reads[nm] = {"path": str(path.relative_to(REPO)), "md5": got,
                     "claimed": claim, "source": src, "match": got == claim}
    if got != claim:
        raise SystemExit(f"ENDPOINT BIND FAILURE: {nm} md5 {got} != {claim}")
METRICS["gates"] = {"G_ENDPOINTS": {"md5_binds": md5_reads, "pass": True}}
write_partial("P0 bars frozen in-script + endpoints md5-bound")


# ---------------------------------------------------------- loaders (CPU)
def load_flat(path: Path) -> tuple[np.ndarray, list[str], list[tuple], dict]:
    """Plain-IO CPU load (the honesty rule: no memmap; .float().clone();
    finiteness verified). Returns (flat fp64, key order, shapes, meta)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    assert isinstance(sd, dict) and all(torch.is_tensor(v) for v in sd.values())
    keys = list(sd.keys())
    tensors = [sd[k].float().clone() for k in keys]          # stale-page guard
    for k, v in zip(keys, tensors):
        if not torch.isfinite(v).all():
            raise SystemExit(f"NON-FINITE tensor {k} in {path.name}")
    flat = torch.cat([v.reshape(-1) for v in tensors]).double().numpy()
    shapes = [tuple(sd[k].shape) for k in keys]
    meta = ({kk: ck[kk] for kk in ("step", "n_chunks") if kk in ck}
            if isinstance(ck, dict) else {})
    return flat.astype(np.float64), keys, shapes, meta


def g_flatbasis(keys: list[str]) -> dict:
    """state_dict key order == net.parameters() order (CPU instantiation of
    lab/common.py's TinyGPT; defines DEVICE only — with
    CUDA_VISIBLE_DEVICES='' no CUDA path is ever touched)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "x6_common", REPO / "lab" / "common.py")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["x6_common"] = mod      # dataclass needs a registered module
    spec.loader.exec_module(mod)
    net = mod.TinyGPT(mod.Cfg())
    pnames = [n for n, _ in net.named_parameters()]
    bnames = [n for n, _ in net.named_buffers()]
    return {"params_count": len(pnames), "buffers_count": len(bnames),
            "key_order_matches_parameters": pnames == keys,
            "n_params": sum(p.numel() for p in net.parameters()),
            "pass": pnames == keys and len(bnames) == 0}


flat_root, KEYS, SHAPES, _ = load_flat(ROOT_CK)
GB = g_flatbasis(KEYS)
if not GB["pass"]:
    raise SystemExit(f"FLAT-BASIS GATE FAILURE: {GB}")
N = int(flat_root.size)
assert N == 2_739_072, N
SHAPE_OF = dict(zip(KEYS, SHAPES))
METRICS["gates"]["G_FLATBASIS"] = GB
write_partial("P1 root loaded + flat basis verified (no buffers; order match)")


# ---------------------------------------------------------- the reads
def natural_read(dw: np.ndarray) -> dict:
    a = np.sort(np.abs(dw))[::-1]
    e = np.cumsum(a * a)
    tot = float(e[-1])
    return {"l2": math.sqrt(tot),
            "sorted_abs_top10": [float(x) for x in a[:10]],
            "sorted_abs_median": float(np.median(a)),
            "m": {f"{int(th * 100)}": int(np.searchsorted(e, th * tot) + 1)
                  for th in (0.5, 0.9, 0.99)}}


def svd_read(dw: np.ndarray, keys: list[str]) -> dict:
    """Per-2D-matrix SVD (fp64). Shares are of the 2D+1D TOTAL (the frozen
    mass bar); T50 is over the 2D energy (SVD is only defined there)."""
    per, spect = {}, []
    e2d = e1d = 0.0
    off = 0
    for k in keys:
        sh = SHAPE_OF[k]
        n = int(np.prod(sh))
        seg = dw[off:off + n]
        off += n
        if len(sh) == 1:
            e = float((seg * seg).sum())
            e1d += e
            per[k] = {"shape": list(sh), "kind": "1D", "energy": e}
            continue
        mat = seg.reshape(sh)
        s = torch.linalg.svdvals(torch.from_numpy(mat)).numpy()
        en = (s * s).astype(np.float64)
        spect.append(en)
        e = float(en.sum())
        e2d += e
        cum = np.cumsum(en)
        r = len(s)
        per[k] = {"shape": list(sh), "kind": "2D", "rank": r, "energy": e,
                  "s_top10": [float(x) for x in s[:10]],
                  "top50_energy": float(en[:min(50, r)].sum()),
                  "top50_share": (float(en[:min(50, r)].sum() / e)
                                  if e > 0 else None),
                  "cross": {f"{int(th * 100)}":
                            (int(np.searchsorted(cum, th * e) + 1)
                             if e > 0 else None)
                            for th in (0.90, 0.95, 0.99)}}
    tot_all = e2d + e1d
    for k in keys:
        per[k]["energy_share_of_total"] = (per[k]["energy"] / tot_all
                                           if tot_all > 0 else None)
    glob = np.sort(np.concatenate(spect))[::-1]
    gcum = np.cumsum(glob)
    mass = [k for k in keys
            if per[k]["kind"] == "2D"
            and (per[k]["energy_share_of_total"] or 0.0) >= 1e-3]
    return {"per_matrix": per,
            "e_2d_total": e2d, "e_1d_total": e1d,
            "e_1d_share_of_total": (e1d / tot_all if tot_all > 0 else None),
            "T50": (float(glob[:50].sum() / e2d) if e2d > 0 else None),
            "global_cross": {f"{int(th * 100)}":
                             int(np.searchsorted(gcum, th * e2d) + 1)
                             for th in (0.90, 0.95, 0.99)},
            "mass_carrying": mass,
            "min_top50_share_mass_carrying": (
                min(per[k]["top50_share"] for k in mass) if mass else None),
            "_global_spectrum": glob}


class SRCT:
    """Ported BY VALUE from lab/e261_rank_ladder.py (fp64 CPU only) — the
    exact orthogonal projector P x = D . idct(mask_S(dct(D . x)))."""

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

    def coeffs(self, x64: np.ndarray) -> np.ndarray:
        return sf.dct(self.D * x64, type=2, norm="ortho", workers=DCT_WORKERS)


def room_read(dw: np.ndarray, room: SRCT) -> dict:
    p = room.project(dw)
    inside = float((p * p).sum() / (dw * dw).sum())
    c = room.coeffs(dw)[room.S]
    ce = np.sort(c * c)[::-1]
    cum = np.cumsum(ce)
    tot = float(cum[-1])
    return {"k": room.k, "inside_share": inside, "outside_share": 1.0 - inside,
            "in_room_spectrum": {
                "top50_share": (float(ce[:50].sum() / tot) if tot > 0 else None),
                "m": {f"{int(th * 100)}":
                      int(np.searchsorted(cum, th * tot) + 1)
                      for th in (0.5, 0.9, 0.99)},
                "_coeff_energy_sorted": ce}}


def load_room(path: Path, rung: str, seeds_expect: list[int]) -> SRCT:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    entry = ck["model"][rung]
    k, seeds = int(entry["k"]), [int(x) for x in entry["seeds"]]
    assert seeds == seeds_expect, (seeds, seeds_expect)
    fresh = SRCT(N, k, seeds[0], seeds[1])
    _as_np = lambda t: t.numpy() if hasattr(t, "numpy") else np.asarray(t)
    D_stored = _as_np(entry["D_int8"]).astype(np.float64)
    S_stored = _as_np(entry["S"])
    assert np.array_equal(D_stored, fresh.D), "room D mismatch vs fresh seeds"
    assert np.array_equal(S_stored, fresh.S), "room S mismatch vs fresh seeds"
    log(f"room {rung}: loaded + BIT-VERIFIED vs fresh seeds {seeds} (k={k})")
    return fresh


# ---------------------------------------------------------- THE OBJECT
flat_k10k, _, _, meta10 = load_flat(K10K_CK)
assert meta10.get("step") == 400, meta10
dW = flat_k10k - flat_root
METRICS["object"] = {
    "root": "runs/checkpoints/e001.pt (md5-bound)",
    "post": "runs/checkpoints/e261_K10K_inst_resume.pt s400 (md5-bound)",
    "dW_l2": float(np.linalg.norm(dW)),
    "dW_linf": float(np.abs(dW).max()),
    "N": N,
    "dispatch_N_claim": 2738880,
    "N_note": "the dispatch letter's 2,738,880 is a transcription slip; "
              "G_BASE/G_ROOT/the checkpoints all say 2,739,072 (disclosed)",
}
log(f"dW_10k ready: L2 {METRICS['object']['dW_l2']:.4f}")

# (a) natural + (b) svd
METRICS["dW_10k"] = {"natural": natural_read(dW)}
write_partial("P2 dW_10k natural basis done")
svd10 = svd_read(dW, KEYS)
METRICS["dW_10k"]["svd"] = svd10
METRICS["gates"]["G_MASS"] = {
    "mass_carrying_matrices": svd10["mass_carrying"],
    "min_top50_share": svd10["min_top50_share_mass_carrying"],
    "e_1d_share": svd10["e_1d_share_of_total"]}
write_partial("P3 dW_10k per-matrix SVD done")

# (c) the room
room10 = load_room(ROOMS264_CK, "K10K", [26113, 26114])
METRICS["dW_10k"]["room"] = room_read(dW, room10)
write_partial("P4 dW_10k room projection done")

# ---------------------------------------------------------- anchors
rng = np.random.default_rng(OP["null_seed"])
g = rng.standard_normal(N)
g *= float(np.linalg.norm(dW)) / float(np.linalg.norm(g))
METRICS["anchors"] = {"null": {
    "seed": OP["null_seed"], "norm_matched_to": "dW_10k",
    "natural": natural_read(g),
    "room10k_inside_share": float(
        (room10.project(g) ** 2).sum() / (g ** 2).sum())}}
write_partial("P5a null anchor done")

flat_k237, _, _, meta237 = load_flat(K237K_CK)
assert meta237.get("step") == 400, meta237
dW237 = flat_k237 - flat_root
room237 = load_room(ROOMS261_CK, "K237K", [26011, 26012])
a237 = {"vehicle": "runs/checkpoints/e261_K237K_inst_resume.pt s400 (md5-bound)",
        "dW_l2": float(np.linalg.norm(dW237)),
        "natural": natural_read(dW237)}
svd237 = svd_read(dW237, KEYS)
a237["svd"] = svd237
a237["room"] = room_read(dW237, room237)
METRICS["anchors"]["k237k"] = a237
write_partial("P5b 237k anchor done")

flat_e260, _, _, _ = load_flat(E260_CK)
dWe260 = flat_e260 - flat_root
_s = np.sort(np.abs(dWe260))[::-1]
_ce = np.cumsum(_s * _s)
METRICS["anchors"]["k237k_e260_session_texture"] = {
    "vehicle": "runs/checkpoints/e260_RANDOM_inst_resume.pt (same 237k arm, "
               "e260 session; co-report only, never a bar)",
    "dW_l2": float(np.linalg.norm(dWe260)),
    "natural_m99": int(np.searchsorted(_ce, 0.99 * float(_ce[-1])) + 1),
    "note": "cross-session vehicle texture on the same arm; the full read set "
            "is reported for the e261-session vehicle only"}
write_partial("P5c e260-session texture co-report done")

# ---------------------------------------------------------- adjudication
m99 = METRICS["dW_10k"]["natural"]["m"]["99"]
T50 = svd10["T50"]
min50 = svd10["min_top50_share_mass_carrying"]
search = (min50 is not None and min50 >= 0.99) and (m99 <= 500)
spread = (T50 is not None and T50 < 0.90) or (m99 >= 5000)
verdict = ("SEARCH-BOTTLENECK" if search else
           "GENUINELY-SPREAD" if spread else "MIXED")
METRICS["verdict"] = {
    "word": verdict,
    "m99_natural": m99, "T50_global": T50,
    "min_top50_share_mass_carrying": min50,
    "bars_verbatim": BARS,
    "operationalizations": OP,
    "clause_trace": {
        "search_top50_per_matrix": bool(min50 is not None and min50 >= 0.99),
        "search_m99_le_500": bool(m99 <= 500),
        "spread_T50_lt_90": bool(T50 is not None and T50 < 0.90),
        "spread_m99_ge_5000": bool(m99 >= 5000)},
}
write_partial(f"P6 ADJUDICATED: {verdict}")

# ---------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def dec_idx(n: int, m: int = 4000) -> np.ndarray:
    return np.unique(np.linspace(0, n - 1, min(m, n)).astype(int))


def cum_curve(dw: np.ndarray) -> np.ndarray:
    s = np.sort(np.abs(dw))[::-1]
    return np.cumsum(s * s) / float((dw * dw).sum())


fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

ax = axes[0]
for nm, dw_, c in (("K10K (the bet)", dW, "crimson"),
                   ("K237K", dW237, "steelblue"),
                   ("null (norm-matched)", g, "gray")):
    cum = cum_curve(dw_)
    ix = dec_idx(len(cum))
    ax.plot(np.arange(1, len(cum) + 1)[ix], cum[ix], color=c, lw=1.4,
            label=f"{nm}  m99={int(np.searchsorted(cum, 0.99) + 1):,}")
ax.set_xscale("log")
ax.set_xlim(1, N)
ax.set_ylim(0, 1.02)
for th, st in ((0.5, ":"), (0.9, "--"), (0.99, "-.")):
    ax.axhline(th, color="k", ls=st, lw=0.6, alpha=0.5)
ax.axvline(500, color="crimson", ls=":", lw=1.0)
ax.axvline(5000, color="steelblue", ls=":", lw=1.0)
ax.set_xlabel("top-m |coords| (natural basis, log)")
ax.set_ylabel("cumulative energy share")
ax.set_title("(a) NATURAL BASIS — effective dimensionality")
ax.legend(fontsize=8, loc="lower right")

ax = axes[1]
for nm, sp, c in (("K10K (the bet)", svd10["_global_spectrum"], "crimson"),
                  ("K237K", svd237["_global_spectrum"], "steelblue")):
    cum = np.cumsum(sp) / float(sp.sum())
    ix = dec_idx(len(cum))
    ax.plot(np.arange(1, len(sp) + 1)[ix], cum[ix], color=c, lw=1.4,
            label=f"{nm}  cross99 r={int(np.searchsorted(cum, 0.99) + 1):,}")
ax.set_xscale("log")
ax.axvline(50, color="k", ls="-", lw=1.0, alpha=0.7)
for th, st in ((0.9, "--"), (0.95, "-."), (0.99, ":")):
    ax.axhline(th, color="k", ls=st, lw=0.6, alpha=0.5)
ax.set_xlabel("singular rank (global 2D spectrum, log)")
ax.set_ylabel("cumulative energy share")
ax.set_title(f"(b) THE BET'S FORM — T50(K10K)={svd10['T50']:.4f}")
ax.legend(fontsize=8, loc="lower right")

ax = axes[2]
for nm, rr, c in (("K10K in own room", METRICS["dW_10k"]["room"], "crimson"),
                  ("K237K in own room", METRICS["anchors"]["k237k"]["room"],
                   "steelblue")):
    ce = rr["in_room_spectrum"]["_coeff_energy_sorted"]
    cum = np.cumsum(ce) / float(ce.sum())
    ix = dec_idx(len(cum))
    ax.plot(np.arange(1, len(ce) + 1)[ix], cum[ix], color=c, lw=1.4,
            label=f"{nm} k={rr['k']:,}  inside={rr['inside_share']:.4f}")
ax.set_xscale("log")
for th, st in ((0.5, ":"), (0.9, "--"), (0.99, "-.")):
    ax.axhline(th, color="k", ls=st, lw=0.6, alpha=0.5)
ax.set_xlabel("in-room coefficient rank (log)")
ax.set_ylabel("cumulative in-room energy share")
ax.set_title("(c) THE ROOM'S READING — inside vs out")
ax.legend(fontsize=8, loc="lower right")

fig.suptitle(f"x6 THE SVD BET — verdict: {verdict}  "
             f"(T50={svd10['T50']:.4f}, m99={m99:,}, "
             f"room-inside={METRICS['dW_10k']['room']['inside_share']:.4f})",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "x6_svd_spectrum.png", dpi=140)
log("FIGURE written")
METRICS["outputs"] = {"figure": "runs/x6/x6_svd_spectrum.png",
                      "metrics": "runs/x6/metrics.json",
                      "report": "runs/x6/REPORT.md"}

# ---------------------------------------------------------- REPORT.md
r10 = METRICS["dW_10k"]
ro = r10["room"]
a23 = METRICS["anchors"]["k237k"]
rn = METRICS["anchors"]["null"]
report = f"""# x6 — THE SVD BET ({verdict})

**The object:** dW of the committed 10k quiet write —
`e261_K10K_inst_resume.pt` (s400 final, md5-bound to e264's G_K10KRESUME)
minus the fresh root `e001.pt` (md5-bound to e267's base bind). All three
bases read on CPU from committed records; no training, no GPU.

## The three bases (K10K dW, L2 {METRICS['object']['dW_l2']:.3f})

1. **Natural basis:** top-m |coords| effective dimensionality m50/m90/m99 =
   {r10['natural']['m']['50']:,}/{r10['natural']['m']['90']:,}/{r10['natural']['m']['99']:,}.
2. **The bet's own form (per-matrix SVD):** global energy-weighted top-50
   singular share T50 = {svd10['T50']:.4f}; the minimum top-50 share over the
   mass-carrying 2D matrices = {svd10['min_top50_share_mass_carrying']:.4f}
   ({len(svd10['mass_carrying'])} matrices carry mass at the frozen 1e-3 bar);
   global 99% rank = {svd10['global_cross']['99']:,}. The 1D params carry
   {100 * svd10['e_1d_share_of_total']:.2f}% of the energy (SVD undefined there;
   disclosed, covered by the natural read).
3. **The room's reading:** {100 * ro['inside_share']:.2f}% of dW's energy
   lies INSIDE its own committed 10k SRCT room (k/N = 10000/2739072 =
   {100 * 10000 / N:.3f}% by volume); the in-room coefficient spectrum's own top-50
   share = {ro['in_room_spectrum']['top50_share']:.4f}, in-room m99 =
   {ro['in_room_spectrum']['m']['99']:,}.

**Anchors:** the norm-matched gaussian null has natural m99 =
{rn['natural']['m']['99']:,} and room-inside share
{rn['room10k_inside_share']:.5f} (expect ~k/N). The 237k rung's dW (its own
committed vehicle, own room): natural m99 = {a23['natural']['m']['99']:,},
T50 = {a23['svd']['T50']:.4f}, in-own-room = {a23['room']['inside_share']:.4f}
— {a23['room']['inside_share'] / ro['inside_share']:.1f}x the 10k write's room confinement. The
e260-session 237k vehicle co-reports cross-session texture (natural m99 =
{METRICS['anchors']['k237k_e260_session_texture']['natural_m99']:,}).

## Verdict: {verdict}

Clause trace: per-matrix top-50 >= 99% everywhere-with-mass =
{METRICS['verdict']['clause_trace']['search_top50_per_matrix']};
m99 <= 500 = {METRICS['verdict']['clause_trace']['search_m99_le_500']};
T50 < 90% = {METRICS['verdict']['clause_trace']['spread_T50_lt_90']};
m99 >= 5000 = {METRICS['verdict']['clause_trace']['spread_m99_ge_5000']}.

Disclosures: the dispatch letter's N=2,738,880 is a slip (actual 2,739,072,
G_BASE/G_ROOT-bound); the K10K room lives in `e264_rooms.pt` (seed-bit-verified
against fresh construction — `e261_rooms.pt` holds only the triaged K1K/K237K
rungs); the room files store no eigen-spectrum (an SRCT room is an
equal-energy orthonormal basis by construction — the in-room coefficient
spectrum is the informative read).
"""
(OUT / "REPORT.md").write_text(report, encoding="utf-8")

METRICS["status"] = "COMPLETE — adjudicated"
write_partial("P7 COMPLETE (report + figure)")
log(f"DONE — verdict {verdict}")
