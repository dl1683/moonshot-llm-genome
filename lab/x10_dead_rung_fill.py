"""
x10 — THE DEAD RUNG'S FILL (P-x10a, T253's adopted agy prediction) — a
CPU-ONLY DESK CELL.

x6 found THE FILL LAW: the quiet 10k write fills ~73% of its granted room
at 99% energy (in-room m99 7,342/10,000 = 73.4%) and the 237k write the
same (174,283/237,123 = 73.5%) — the write densely fills whatever room it
is granted. e272 relocated the expression edge to (1k,2k]: the 1k write is
DEAD (post g0 0.000435) while 2k is transitional (0.0266) and 5k
alive-ish (0.1271). THE REGISTERED QUESTION (P-x10a): does the DEAD 1k
write STILL FILL ~73% of its 1,000-dim room? If yes: capacity is a
THRESHOLD ON USABLE DEGREES OF FREEDOM (the optimizer fills the room as
hard as it can; ~730 dims simply cannot assemble the logits) — occupancy
style is width-invariant across the expression edge. If no (the dead write
is concentrated/sparse in its room): expression failure leaves a different
geometric signature.

BARS — FROZEN VERBATIM FROM THE DISPATCH, BEFORE ANY COMPUTE (no bar shopping):

  FILLS-ANYWAY (P-x10a confirmed): the dead 1k rung's fill fraction is
      within [0.55, 0.90] — occupancy style is width-invariant across the
      expression edge; capacity = a DoF threshold.
  DEAD-DIFFERENTLY: the 1k fill is outside [0.40, 1.00-epsilon] range in
      the specific direction of CONCENTRATION (top-50 in-room share >= 2x
      the 10k rung's) — the dead write is a failed, over-concentrated
      attempt; expression failure has a geometric signature.
  MIXED: the fill is in-between or the 2k transitional rung breaks the
      trend — the fill curve verbatim, no inflation.

OPERATIONALIZATIONS — FROZEN BEFORE COMPUTE:
  * THE OBJECT per rung (x6's convention VERBATIM, reused by adaptation
    of lab/x6_svd_bet.py — ported BY VALUE, never imported): dW :=
    flat(<rung inst_resume .pt["model"] at s400>) MINUS flat(e001.pt),
    the fresh base every ladder arm installed from. Plain-IO CPU loads
    only (torch.load map_location='cpu', .float().clone(), finiteness
    verified; NO memmap anywhere — the OneDrive honesty rule).
  * THE LADDER (each rung read in ITS OWN room — the room its install
    projected into): 1k = e261's committed dead rung (e261_K1K_inst_resume,
    room e261_rooms K1K seeds 26111/26112); 2k = e272 K2K (room e272_rooms
    K2K 27211/27212); 5k = e272 K5K (27213/27214); 10k = x6's object
    (e261_K10K_inst_resume, room e264_rooms K10K 26113/26114 — the e264
    committed rung x6 read); 237k = x6's anchor (e261_K237K_inst_resume,
    room e261_rooms K237K 26011/26012).
  * ROOM ENERGY SHARE: inside_share := ||P dW||^2 / ||dW||^2 with P the
    SRCT projector D.idct(mask_S(dct(D.x))) in fp64 CPU (x6's exact form).
  * IN-ROOM SPECTRUM: c = dct(D.dW) restricted to the room's k selected
    frequencies; coefficient energies c^2 sorted desc; m(th) := smallest m
    with top-m carrying >= th of the IN-ROOM energy (x6's searchsorted
    convention), th in {50, 90, 99}%; top-50 in-room share reported.
  * FILL FRACTION := m99/k (x6's exact convention; its committed values:
    10k 7342/10000 = 0.7342, 237k 174283/237123 = 0.7350).
  * THE VERDICT READS THE DEAD 1k RUNG (e261's committed dead write, post
    g0 0.000435). fill_1k := m99_1k/1000. FILLS-ANYWAY iff 0.55 <=
    fill_1k <= 0.90. DEAD-DIFFERENTLY iff fill_1k < 0.40 AND top50_1k >=
    2 x top50_10k (the [0.40, 1.00-eps] window's outside-in-the-
    concentration-direction clause; the upper clause is vacuous — m99 <= k
    by construction, disclosed). MIXED otherwise. The 2k trend-break
    co-fires the MIXED wording when it holds: fill_2k < min(fills of the
    other four rungs) - 0.05 or > max(...) + 0.05.
  * CO-REPORTS, NEVER BARS: K1KM (e272's kept-matched dead 1k arm, lr
    x3.7306, post 0.0012 — does the dead fill survive a dose change at
    matched rank?) and K10KR (e272's fresh-room 10k replicate — the room
    lottery priced on the fill law itself).
  * x6 REPRODUCTION GATE: this cell recomputes the 10k and 237k reads and
    must reproduce x6's committed inside_share + in-room m's (md5-bound
    cite of runs/x6/metrics.json) before any new rung is believed.
  * DISPLACEMENT-LOAD CONTENT BIND: e261/e272's committed install ledgers
    record in_own_room as a NORM ratio; its square must equal this cell's
    fp64 inside_share within 2e-3 for every arm read (e261 K237K's
    committed 0.9804893255327712^2 == x6's 0.9613593174841455 exactly).
  * ENDPOINT BINDS: e001 / e261_K10K / e261_K237K / e261_rooms carry
    committed md5s (x6's binds — hard gates). e261_K1K_inst_resume and the
    four e272 inst_resumes carry NO committed md5 anywhere in the record
    (searched runs/*/metrics.json; disclosed) — this cell binds them by
    FRESH md5 (recorded here for future cells, the e267 base-bind
    precedent) + step==400 + the displacement-load content match + room
    bit-verification. Rooms: every entry BIT-VERIFIED vs fresh seed
    reconstruction (exact D/S equality — x6's precedent, stronger than
    md5) with seeds asserted against the parent cells' committed
    registries.
  * CPU ENVELOPE: CUDA_VISIBLE_DEVICES="" hard-set before torch import;
    torch threads 4; pocketfft workers 4; NOTHING written to
    runs/_envelope_log.jsonl; no lab/ GPU code path executed (the
    TinyGPT CPU instantiation for the flat-basis gate defines DEVICE only).
  * PROGRESSIVE writes to runs/x10/metrics.json; no NOTES/THINKING/QUEUE/
    STATE edits (dispatch — the coordinator folds).

Provenance (verified BEFORE compute, in G_ENDPOINTS below):
  ROOT             runs/checkpoints/e001.pt                 md5 d114536d1c0983ab3be67f67ff0667c8 (x6/e267 bind)
  1k POST s400     runs/checkpoints/e261_K1K_inst_resume.pt fresh bind (e261 provenance names this exact path)
  2k POST s400     runs/checkpoints/e272_K2K_inst_resume.pt fresh bind (e272 provenance)
  5k POST s400     runs/checkpoints/e272_K5K_inst_resume.pt fresh bind (e272 provenance)
  10k POST s400    runs/checkpoints/e261_K10K_inst_resume.pt md5 0f6dc1cf46850ce655dfafc9c853d467 (x6/e264 bind)
  237k POST s400   runs/checkpoints/e261_K237K_inst_resume.pt md5 458e5211ef61aae7e14cd3b966f23dfc (x6/e271 bind)
  K1KM POST s400   runs/checkpoints/e272_K1KM_inst_resume.pt fresh bind (co-report)
  K10KR POST s400  runs/checkpoints/e272_K10KR_inst_resume.pt fresh bind (co-report)
  ROOM 1k          runs/checkpoints/e261_rooms.pt  K1K   26111/26112 (md5 f81571c40bb6009d4ed1cfab66e73f6c, x6 bind)
  ROOM 2k/5k/...   runs/checkpoints/e272_rooms.pt  K2K 27211/27212, K5K 27213/27214, K10KR 27215/27216, K1KM 26111/26112 (fresh bind)
  ROOM 10k         runs/checkpoints/e264_rooms.pt  K10K  26113/26114 (fresh bind; bit-verified — x6 precedent)
  ROOM 237k        runs/checkpoints/e261_rooms.pt  K237K 26011/26012 (same md5-bound file)
  PARENT RECORDS   runs/e261/metrics.json, runs/e264/metrics.json, runs/e272/metrics.json, runs/x6/metrics.json (md5-bound cites)
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-ONLY, hard-set (no GPU lane touch)

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
OUT = REPO / "runs" / "x10"
OUT.mkdir(parents=True, exist_ok=True)

T0 = time.time()
METRICS: dict = {
    "experiment": "x10_dead_rung_fill",
    "phase": "THE DEAD RUNG'S FILL (P-x10a, T253) — does the DEAD 1k write "
             "still fill ~73% of its 1,000-dim room? The fill curve across "
             "e272's relocated expression edge (1k dead / 2k transitional / "
             "5k alive-ish / 10k / 237k), x6's conventions; CPU-only desk "
             "cell, committed records only",
    "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "status": "RUNNING (progressive writes)",
    "cpu_only": True,
    "threads": {"torch": 4, "pocketfft": DCT_WORKERS},
}
BARS = {
    "FILLS-ANYWAY": "the dead 1k rung's fill fraction is within [0.55, 0.90] "
        "— occupancy style is width-invariant across the expression edge; "
        "capacity = a DoF threshold",
    "DEAD-DIFFERENTLY": "the 1k fill is outside [0.40, 1.00-epsilon] range "
        "in the specific direction of CONCENTRATION (top-50 in-room share "
        ">= 2x the 10k rung's) — the dead write is a failed, "
        "over-concentrated attempt; expression failure has a geometric "
        "signature",
    "MIXED": "the fill is in-between or the 2k transitional rung breaks the "
        "trend — the fill curve verbatim, no inflation",
}
OP = {
    "object": "per rung dW = flat(inst_resume s400) - flat(e001.pt) (x6's "
              "convention verbatim; plain-IO fp64 CPU loads)",
    "ladder": "1k e261_K1K (dead, own room e261_rooms K1K) / 2k e272_K2K "
              "(e272_rooms) / 5k e272_K5K (e272_rooms) / 10k e261_K10K "
              "(e264_rooms K10K — x6's object) / 237k e261_K237K "
              "(e261_rooms K237K — x6's anchor)",
    "inside_share": "||P dW||^2 / ||dW||^2, P the SRCT projector in fp64 CPU",
    "in_room_spectrum": "c = dct(D.dW) over the room's k frequencies; "
                        "|c|^2 sorted desc; m(th) = searchsorted(cum, "
                        "th*tot)+1, th in {50,90,99}%",
    "fill_fraction": "m99/k (x6's convention: 10k 0.7342, 237k 0.7350)",
    "verdict": "FILLS-ANYWAY iff 0.55 <= fill_1k <= 0.90; DEAD-DIFFERENTLY "
               "iff fill_1k < 0.40 AND top50_1k >= 2*top50_10k (the upper "
               "1.00-eps clause vacuous, m99 <= k by construction); MIXED "
               "otherwise; 2k trend-break := fill_2k outside "
               "[min,max]-0.05/+0.05 of the other four rungs' fills",
    "coreports_never_bars": "K1KM (e272's kept-matched dead 1k arm, lr "
                            "x3.7306) + K10KR (e272's fresh-room 10k "
                            "replicate)",
    "x6repro_gate": "the 10k/237k reads must reproduce x6's committed "
                    "inside_share + in-room m's before any new rung is "
                    "believed",
    "displ_content_bind": "e261/e272 committed in_own_room (a NORM ratio) "
                          "squared must equal this cell's inside_share "
                          "within 2e-3 per arm",
    "null_note": "an unprojected gaussian carries in-room share ~k/N "
                 "(x6's null: 0.0036 at 10k) — the fill law's contrast, "
                 "cited not re-run",
}


def log(msg: str) -> None:
    print(f"[x10 {time.time() - T0:7.1f}s] {msg}", flush=True)


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


# ---------------------------------------------------------- G_ENDPOINTS
ROOT_CK = CKPT / "e001.pt"
P = {  # every file this cell touches
    "e001": CKPT / "e001.pt",
    "e261_K1K": CKPT / "e261_K1K_inst_resume.pt",
    "e272_K2K": CKPT / "e272_K2K_inst_resume.pt",
    "e272_K5K": CKPT / "e272_K5K_inst_resume.pt",
    "e261_K10K": CKPT / "e261_K10K_inst_resume.pt",
    "e261_K237K": CKPT / "e261_K237K_inst_resume.pt",
    "e272_K1KM": CKPT / "e272_K1KM_inst_resume.pt",
    "e272_K10KR": CKPT / "e272_K10KR_inst_resume.pt",
    "e261_rooms": CKPT / "e261_rooms.pt",
    "e264_rooms": CKPT / "e264_rooms.pt",
    "e272_rooms": CKPT / "e272_rooms.pt",
    "e261_metrics": REPO / "runs" / "e261" / "metrics.json",
    "e264_metrics": REPO / "runs" / "e264" / "metrics.json",
    "e272_metrics": REPO / "runs" / "e272" / "metrics.json",
    "x6_metrics": REPO / "runs" / "x6" / "metrics.json",
}
# committed md5 binds (hard gates): x6's binds + the parent records cited
COMMITTED = {
    "e001": ("d114536d1c0983ab3be67f67ff0667c8", "x6's e267 base bind"),
    "e261_K10K": ("0f6dc1cf46850ce655dfafc9c853d467", "x6's e264 G_K10KRESUME bind"),
    "e261_K237K": ("458e5211ef61aae7e14cd3b966f23dfc", "x6's e271 vehicle bind"),
    "e261_rooms": ("f81571c40bb6009d4ed1cfab66e73f6c", "x6's e271 bind"),
    "e261_metrics": ("f460475d8e6b76f0719e91c1e9c6041b", "e272 G_PARENTS bind"),
    "e264_metrics": ("a42ff4786784b04cb9819a69b545e343", "e272 G_PARENTS bind"),
    "e272_metrics": (None, "this cell's parent (no prior bind — fresh-bound)"),
    "x6_metrics": (None, "this cell's fill-law source (fresh-bound)"),
}
md5_reads = {}
for nm, path in P.items():
    got = md5_of(path)
    claim, src = COMMITTED.get(nm, (None, "no committed md5 anywhere in the "
                                         "record (searched) — FRESH BIND"))
    md5_reads[nm] = {"path": str(path.relative_to(REPO)), "md5": got,
                     "claimed": claim, "source": src,
                     "match": (True if claim is None else got == claim)}
    if claim is not None and got != claim:
        raise SystemExit(f"ENDPOINT BIND FAILURE: {nm} md5 {got} != {claim}")
METRICS["gates"] = {"G_ENDPOINTS": {
    "md5_binds": md5_reads,
    "note": "fresh binds recorded for future cells (the e267 base-bind "
            "precedent); rooms additionally BIT-verified vs fresh seed "
            "reconstruction in G_ROOMS; inst_resumes content-bound in "
            "G_DISPL (step 400 + committed displacement-load match)",
    "pass": True}}
write_partial("P0 bars frozen in-script + endpoints bound (fresh md5s recorded)")


# ---------------------------------------------------------- loaders (CPU)
def load_flat(path: Path) -> tuple[np.ndarray, list[str], dict]:
    """Plain-IO CPU load (the honesty rule: no memmap; .float().clone();
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
        "x10_common", REPO / "lab" / "common.py")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["x10_common"] = mod      # dataclass needs a registered module
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
write_partial("P1 root loaded + flat basis verified (no buffers; order match)")


# ---------------------------------------------------------- the room (SRCT)
class SRCT:
    """Ported BY VALUE via x6 (which ported from lab/e261_rank_ladder.py) —
    the exact orthogonal projector P x = D . idct(mask_S(dct(D . x))),
    fp64 CPU only."""

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


def load_room(path: Path, rung: str, seeds_expect: list[int]) -> SRCT:
    """Load a stored room and BIT-VERIFY it against fresh reconstruction
    from its committed seeds (exact D/S equality — x6's precedent)."""
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
    "K2K": (P["e272_rooms"], "K2K", [27211, 27212]),
    "K5K": (P["e272_rooms"], "K5K", [27213, 27214]),
    "K10K": (P["e264_rooms"], "K10K", [26113, 26114]),
    "K237K": (P["e261_rooms"], "K237K", [26011, 26012]),
    "K1KM": (P["e272_rooms"], "K1KM", [26111, 26112]),
    "K10KR": (P["e272_rooms"], "K10KR", [27215, 27216]),
}
rooms: dict[str, SRCT] = {}
room_gate = {}
for nm, (path, rung, seeds) in ROOMS.items():
    rooms[nm] = load_room(path, rung, seeds)
    room_gate[nm] = {"file": str(path.relative_to(REPO)), "k": rooms[nm].k,
                     "seeds": seeds, "bit_verified_vs_fresh": True}
# the K1KM room must BE the committed K1K room (e272's G_ROOM1K, re-gated)
assert np.array_equal(rooms["K1KM"].D, rooms["K1K"].D), "K1KM room != K1K room"
assert np.array_equal(rooms["K1KM"].S, rooms["K1K"].S), "K1KM room != K1K room"
room_gate["K1KM_bit_equals_committed_K1K"] = True
METRICS["gates"]["G_ROOMS"] = room_gate
write_partial("P2 all 7 rooms loaded + bit-verified vs fresh committed seeds")


# ---------------------------------------------------------- the reads
def room_read(dw: np.ndarray, room: SRCT) -> dict:
    """x6's exact room read: inside share via the projector; in-room
    coefficient energy spectrum; m(50/90/99); fill := m99/k."""
    p = room.project(dw)
    inside_proj = float((p * p).sum() / (dw * dw).sum())
    c = room.coeffs(dw)[room.S]
    ce = np.sort(c * c)[::-1]
    cum = np.cumsum(ce)
    tot = float(cum[-1])
    inside_parseval = tot / float((dw * dw).sum())
    return {"k": room.k,
            "inside_share": inside_proj,
            "inside_share_parseval_crosscheck": inside_parseval,
            "in_room_spectrum": {
                "top50_share": (float(ce[:50].sum() / tot) if tot > 0 else None),
                "m": {f"{int(th * 100)}":
                      int(np.searchsorted(cum, th * tot) + 1)
                      for th in (0.5, 0.9, 0.99)},
                "fill_m99_over_k": (int(np.searchsorted(cum, 0.99 * tot) + 1)
                                    / room.k)},
            "_ce": ce}


def decimate(ce: np.ndarray, m: int = 240) -> list[list[float]]:
    ix = np.unique(np.concatenate([
        np.linspace(0, len(ce) - 1, min(m, len(ce))).astype(int),
        np.arange(0, min(60, len(ce)))]))
    return [[int(i + 1), float(ce[: i + 1].sum() / ce.sum())] for i in ix]


# the ladder + co-reports: (name, ckpt key, room name, committed post g0, cite)
LADDER = [
    ("K1K", "e261_K1K", "K1K", 0.00043458465370349586,
     "e261 committed dead rung"),
    ("K2K", "e272_K2K", "K2K", 0.026616254821419716,
     "e272 K2K (fresh room)"),
    ("K5K", "e272_K5K", "K5K", 0.12709596753120422,
     "e272 K5K (fresh room)"),
    ("K10K", "e261_K10K", "K10K", 0.26464763283729553,
     "e264 committed cite (x6's object)"),
    ("K237K", "e261_K237K", "K237K", 0.38436421751976013,
     "e261 committed (natural width)"),
]
COREPORTS = [
    ("K1KM", "e272_K1KM", "K1KM", 0.0012453667586669326,
     "e272 kept-matched dead 1k arm (lr x3.7306) — CO-REPORT, never a bar"),
    ("K10KR", "e272_K10KR", "K10KR", 0.2097209095954895,
     "e272 fresh-room 10k replicate — CO-REPORT, never a bar"),
]
# committed in_own_room NORM ratios (squared == inside_share); the content bind
DISPL_TARGETS = {
    "K1K": (0.8990001680413359, "e261 metrics install displacement_loads"),
    "K2K": (0.9090426956346246, "e272 metrics install displacement_loads"),
    "K5K": (0.9244495344843363, "e272 metrics install displacement_loads"),
    "K10KR": (0.9445320495247138, "e272 metrics install displacement_loads"),
    "K1KM": (0.7955550758680459, "e272 metrics install displacement_loads"),
    "K237K": (0.9804893255327712, "e261 metrics install displacement_loads"),
}
# x6's committed reads (the reproduction gate's frozen targets)
X6_TARGETS = {
    "K10K": {"inside_share": 0.8914359886030179,
             "m": {"50": 1243, "90": 4428, "99": 7342},
             "top50_share": 0.04958612457788772},
    "K237K": {"inside_share": 0.9613593174841455,
              "m": {"50": 29330, "90": 105413, "99": 174283},
              "top50_share": 0.003152541008408758},
}

READS: dict[str, dict] = {}
DISPL_GATE = {}
X6REPRO = {}
for nm, ck_key, room_nm, post, cite in LADDER + COREPORTS:
    flat_post, _, meta = load_flat(P[ck_key])
    assert meta.get("step") == 400, (nm, meta)
    dw = flat_post - flat_root
    rr = room_read(dw, rooms[room_nm])
    READS[nm] = {
        "vehicle": f"runs/checkpoints/{P[ck_key].name} (s400; "
                   f"md5 {md5_reads[ck_key]['md5']})",
        "room": f"{ROOMS[room_nm][0].name}:{room_nm} seeds "
                f"{ROOMS[room_nm][2]} (bit-verified)",
        "post_g0_cited": post, "post_g0_source": cite,
        "dW_l2": float(np.linalg.norm(dw)),
        "room_read": rr,
        "in_room_spectrum_decimated": decimate(rr["_ce"]),
        "_dw": dw,
    }
    if nm in DISPL_TARGETS:
        tgt, src = DISPL_TARGETS[nm]
        got = rr["inside_share"]
        DISPL_GATE[nm] = {"committed_in_own_room_norm_ratio": tgt,
                          "squared": tgt * tgt, "this_cell_inside_share": got,
                          "abs_diff": abs(got - tgt * tgt),
                          "source": src,
                          "pass": abs(got - tgt * tgt) <= 2e-3}
    if nm in X6_TARGETS:
        t = X6_TARGETS[nm]
        X6REPRO[nm] = {
            "x6_inside_share": t["inside_share"],
            "this_inside_share": rr["inside_share"],
            "inside_abs_diff": abs(rr["inside_share"] - t["inside_share"]),
            "x6_m": t["m"], "this_m": rr["in_room_spectrum"]["m"],
            "m_exact_match": rr["in_room_spectrum"]["m"] == t["m"],
            "x6_top50": t["top50_share"],
            "this_top50": rr["in_room_spectrum"]["top50_share"],
            "pass": (abs(rr["inside_share"] - t["inside_share"]) <= 1e-9
                     and rr["in_room_spectrum"]["m"] == t["m"]),
        }
METRICS["gates"]["G_DISPL"] = {**DISPL_GATE, "tol": 2e-3,
    "form": "committed in_own_room (norm ratio)^2 == this cell's fp64 "
            "inside_share", "pass": all(g["pass"] for g in DISPL_GATE.values())}
if not METRICS["gates"]["G_DISPL"]["pass"]:
    raise SystemExit(f"DISPLACEMENT CONTENT BIND FAILURE: {DISPL_GATE}")
METRICS["gates"]["G_X6REPRO"] = X6REPRO
if not all(g["pass"] for g in X6REPRO.values()):
    raise SystemExit(f"X6 REPRODUCTION GATE FAILURE: {X6REPRO}")
write_partial("P3 all 7 rungs read; G_DISPL + G_X6REPRO PASS "
              "(x6's 10k/237k reads reproduced)")


# ---------------------------------------------------------- the fill curve
def pub(rr: dict) -> dict:
    r = rr["room_read"]
    return {"k": r["k"], "post_g0": rr["post_g0_cited"],
            "inside_share": r["inside_share"],
            "top50_share": r["in_room_spectrum"]["top50_share"],
            "m50": r["in_room_spectrum"]["m"]["50"],
            "m90": r["in_room_spectrum"]["m"]["90"],
            "m99": r["in_room_spectrum"]["m"]["99"],
            "fill": r["in_room_spectrum"]["fill_m99_over_k"]}


curve = {nm: pub(READS[nm]) for nm, *_ in LADDER}
corep = {nm: pub(READS[nm]) for nm, *_ in COREPORTS}
METRICS["fill_curve"] = curve
METRICS["coreports_never_bars"] = corep
write_partial("P4 fill curve computed (5 rungs + 2 co-reports)")

# ---------------------------------------------------------- adjudication
fill_1k = curve["K1K"]["fill"]
fill_2k = curve["K2K"]["fill"]
top50_1k = curve["K1K"]["top50_share"]
top50_10k = curve["K10K"]["top50_share"]
others = [curve[nm]["fill"] for nm, *_ in LADDER if nm != "K2K"]
trend_break_2k = (fill_2k < min(others) - 0.05) or (fill_2k > max(others) + 0.05)
fills_anyway = 0.55 <= fill_1k <= 0.90
dead_differently = (fill_1k < 0.40) and (top50_1k >= 2.0 * top50_10k)
if fills_anyway:
    verdict = "FILLS-ANYWAY"
elif dead_differently:
    verdict = "DEAD-DIFFERENTLY"
else:
    verdict = "MIXED"
METRICS["verdict"] = {
    "word": verdict,
    "fill_1k": fill_1k, "top50_1k": top50_1k, "top50_10k": top50_10k,
    "top50_ratio_1k_over_10k": top50_1k / top50_10k,
    "clause_trace": {
        "fills_anyway_window_0.55_0.90": fills_anyway,
        "dead_diff_fill_below_0.40": fill_1k < 0.40,
        "dead_diff_top50_ge_2x_10k": top50_1k >= 2.0 * top50_10k,
        "upper_1.00_eps_clause_vacuous": True,
        "trend_break_2k": trend_break_2k},
    "bars_verbatim": BARS,
    "operationalizations": OP,
}
write_partial(f"P5 ADJUDICATED: {verdict}")

# ---------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ks = [curve[nm]["k"] for nm, *_ in LADDER]
fills = [curve[nm]["fill"] for nm, *_ in LADDER]
posts = [curve[nm]["post_g0"] for nm, *_ in LADDER]
insides = [curve[nm]["inside_share"] for nm, *_ in LADDER]

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

ax = axes[0]
ax.axhspan(0.55, 0.90, color="green", alpha=0.10, label="FILLS-ANYWAY window [0.55,0.90]")
ax.axhline(0.40, color="darkred", ls=":", lw=1.0, label="DEAD-DIFFERENTLY concentration line (0.40)")
ax.plot(ks, fills, "o-", color="crimson", lw=1.6, ms=7, label="fill = m99/k (own room)")
for nm, *_ in LADDER:
    ax.annotate(f"{curve[nm]['fill']:.3f}", (curve[nm]["k"], curve[nm]["fill"]),
                textcoords="offset points", xytext=(0, 9), fontsize=8,
                ha="center", color="crimson")
ax.plot(corep["K1KM"]["k"], corep["K1KM"]["fill"], "s", mfc="none",
        color="purple", ms=9, label=f"K1KM co-report {corep['K1KM']['fill']:.3f}")
ax.plot(corep["K10KR"]["k"], corep["K10KR"]["fill"], "s", mfc="none",
        color="steelblue", ms=9, label=f"K10KR co-report {corep['K10KR']['fill']:.3f}")
ax.set_xscale("log")
ax.set_xlabel("room width k (log)")
ax.set_ylabel("fill fraction m99/k")
ax.set_ylim(0, 1.02)
ax2 = ax.twinx()
ax2.plot(ks, posts, "^--", color="black", lw=1.0, ms=6, alpha=0.65,
         label="post g0 (expression, cited)")
ax2.set_yscale("log")
ax2.set_ylabel("post g0 (log) — the expression curve")
ax.set_title("(a) THE FILL CURVE ACROSS THE RELOCATED EDGE\n"
             f"(1k dead 0.000435 / 2k 0.0266 / 5k 0.1271 / 10k 0.2646 / 237k 0.3844)")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="lower right")

ax = axes[1]
for nm, *_ in LADDER:
    ce = READS[nm]["room_read"]["_ce"]
    cum = np.cumsum(ce) / float(ce.sum())
    ix = np.unique(np.linspace(0, len(ce) - 1, 2000).astype(int))
    ax.plot((np.arange(1, len(ce) + 1) / READS[nm]["room_read"]["k"])[ix],
            cum[ix], lw=1.3,
            label=f"{nm} (k={READS[nm]['room_read']['k']:,}, m99/k={READS[nm]['room_read']['in_room_spectrum']['fill_m99_over_k']:.3f})")
for th, st in ((0.5, ":"), (0.9, "--"), (0.99, "-.")):
    ax.axhline(th, color="k", ls=st, lw=0.6, alpha=0.5)
ax.set_xscale("log")
ax.set_xlabel("in-room coefficient rank / k (log — width-normalized)")
ax.set_ylabel("cumulative in-room energy share")
ax.set_title("(b) IN-ROOM SPECTRA (width-normalized)")
ax.legend(fontsize=7.5, loc="lower right")

ax = axes[2]
ax.plot(ks, insides, "o-", color="teal", lw=1.6, ms=7,
        label="inside_share (energy in own room)")
for nm, *_ in LADDER:
    ax.annotate(f"{curve[nm]['inside_share']:.3f}",
                (curve[nm]["k"], curve[nm]["inside_share"]),
                textcoords="offset points", xytext=(0, 9), fontsize=8,
                ha="center", color="teal")
ax.plot(corep["K1KM"]["k"], corep["K1KM"]["inside_share"], "s", mfc="none",
        color="purple", ms=9)
ax.plot(corep["K10KR"]["k"], corep["K10KR"]["inside_share"], "s", mfc="none",
        color="steelblue", ms=9)
nulls = [k / N for k in ks]
ax.plot(ks, nulls, ":,", color="gray", lw=1.0, label="null expectation k/N")
ax.set_xscale("log")
ax.set_ylim(0, 1.05)
ax.set_xlabel("room width k (log)")
ax.set_ylabel("in-own-room energy share")
ax.set_title("(c) ROOM CONFINEMENT vs NULL (k/N)")
ax.legend(fontsize=8, loc="center right")

fig.suptitle(f"x10 THE DEAD RUNG'S FILL — verdict: {verdict}  "
             f"(fill curve 1k {curve['K1K']['fill']:.3f} / 2k {curve['K2K']['fill']:.3f} / "
             f"5k {curve['K5K']['fill']:.3f} / 10k {curve['K10K']['fill']:.3f} / "
             f"237k {curve['K237K']['fill']:.3f})", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "x10_fill_curve.png", dpi=140)
log("FIGURE written")
METRICS["outputs"] = {"figure": "runs/x10/x10_fill_curve.png",
                      "metrics": "runs/x10/metrics.json",
                      "report": "runs/x10/REPORT.md"}
write_partial("P6 figure written")

# ---------------------------------------------------------- REPORT.md
c = curve
k1, k2, k5, k10, k237 = (c["K1K"], c["K2K"], c["K5K"], c["K10K"], c["K237K"])
report = f"""# x10 — THE DEAD RUNG'S FILL ({verdict})

**The registered question (P-x10a, T253):** x6's fill law says a quiet
write fills ~73% of whatever room it is granted (10k: 7,342/10,000;
237k: 174,283/237,123). e272 relocated the expression edge to (1k,2k]:
the 1k write is DEAD (post g0 0.000435). Does the dead 1k write STILL
fill ~73% of its 1,000-dim room — or does expression failure leave a
different geometric signature?

## The fill curve (x6's conventions; each rung in its own room)

| rung | k | post g0 (cited) | inside share | in-room m50/m90/m99 | FILL m99/k |
|---|---|---|---|---|---|
| 1k (DEAD, e261) | 1,000 | 0.000435 | {k1['inside_share']:.4f} | {k1['m50']}/{k1['m90']}/{k1['m99']} | **{k1['fill']:.4f}** |
| 2k (transitional, e272) | 2,000 | 0.026616 | {k2['inside_share']:.4f} | {k2['m50']}/{k2['m90']}/{k2['m99']} | **{k2['fill']:.4f}** |
| 5k (alive-ish, e272) | 5,000 | 0.127096 | {k5['inside_share']:.4f} | {k5['m50']}/{k5['m90']}/{k5['m99']} | **{k5['fill']:.4f}** |
| 10k (alive, x6's object) | 10,000 | 0.264648 | {k10['inside_share']:.4f} | {k10['m50']}/{k10['m90']}/{k10['m99']} | **{k10['fill']:.4f}** |
| 237k (natural width, x6) | 237,123 | 0.384364 | {k237['inside_share']:.4f} | {k237['m50']}/{k237['m90']}/{k237['m99']} | **{k237['fill']:.4f}** |

**Co-reports (never bars):** K1KM — e272's kept-matched DEAD 1k arm
(lr x3.7306, post 0.0012): fill {corep['K1KM']['fill']:.4f},
inside {corep['K1KM']['inside_share']:.4f} (the dead fill under a 3.7x
dose change, same room). K10KR — e272's fresh-room 10k replicate
(post 0.2097): fill {corep['K10KR']['fill']:.4f}, inside
{corep['K10KR']['inside_share']:.4f} (the room lottery priced on the
fill law itself).

## Verdict: {verdict}

Clause trace: fill_1k = {fill_1k:.4f} in [0.55, 0.90] =
{fills_anyway}; fill_1k < 0.40 = {fill_1k < 0.40}; top50_1k
({top50_1k:.4f}) >= 2 x top50_10k ({top50_10k:.4f}) =
{top50_1k >= 2.0 * top50_10k} (ratio {top50_1k / top50_10k:.2f}x);
the [0.40, 1.00-eps] upper clause is vacuous (m99 <= k by
construction); 2k trend-break = {trend_break_2k}.

The 10k and 237k reads were RECOMPUTED by this cell and reproduce x6's
committed metrics exactly (G_X6REPRO: inside shares {X6REPRO['K10K']['this_inside_share']:.10f}/
{X6REPRO['K237K']['this_inside_share']:.10f}, m's integer-exact) — the
pipeline is x6's before any new number is believed.

## Gates

- G_ENDPOINTS: e001/e261_K10K/e261_K237K/e261_rooms md5-bound to x6's
  committed binds; e261/e264/e272 metrics md5-bound to e272's G_PARENTS.
- G_DISPL (content bind for the un-md5'd vehicles): every arm's fp64
  inside_share matches the parent cell's committed in_own_room NORM
  ratio squared within 2e-3 (max abs diff
  {max(g['abs_diff'] for g in DISPL_GATE.values()):.2e}; e261 K237K
  matches to {DISPL_GATE['K237K']['abs_diff']:.2e}).
- G_ROOMS: all 7 room entries bit-verified vs fresh seed reconstruction;
  e272's K1KM room re-gated bit-equal to the committed K1K room.
- G_FLATBASIS: key order == TinyGPT parameters, no buffers, N=2,739,072.

## Disclosures

1. e261_K1K_inst_resume.pt and the four e272 inst_resumes carry NO
   committed md5 anywhere in the record (searched runs/*/metrics.json).
   This cell binds them by fresh md5 (recorded in metrics G_ENDPOINTS,
   citable by future cells) + step==400 + the G_DISPL content match +
   the room bit-verification.
2. The dead rung read is e261's committed K1K arm; e272's K1KM
   (dose-matched) is a co-report, never a bar. The 10k expression cite
   is e264's committed rung value on the same net x6 read
   (e261_K10K_inst_resume s400); the 237k cite is e261's K237K arm
   (post 0.384364).
3. The in-room spectra are stored decimated (~240 log-spaced cumulative
   points) in metrics.json; the m's/top50/fill are computed on the full
   spectra, never the decimation.
4. CPU-only: threads 4/4, no envelope-log writes, no GPU code path.
"""
(OUT / "REPORT.md").write_text(report, encoding="utf-8")

METRICS["status"] = "COMPLETE — adjudicated"
write_partial("P7 COMPLETE (report + figure)")
log(f"DONE — verdict {verdict}; fill curve "
    f"1k {curve['K1K']['fill']:.4f} / 2k {curve['K2K']['fill']:.4f} / "
    f"5k {curve['K5K']['fill']:.4f} / 10k {curve['K10K']['fill']:.4f} / "
    f"237k {curve['K237K']['fill']:.4f}")
