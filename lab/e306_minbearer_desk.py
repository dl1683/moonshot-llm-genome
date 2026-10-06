"""E306 DESK HALF — THE MINIMUM BEARER'S DESIGN + THE TRUNCATION INVENTORY
(QUEUE e306's desk half; dispatched 2026-10-05). This docstring carries the
registered bars VERBATIM from the dispatch letter, committed at birth BEFORE
any compute. Adjudicate against exactly this; no bar shopping.

THE QUESTION (the desk half of e306): the committed 10k quiet write
(e261_K10K_inst_resume s400 vs the fresh e001 root — x6's object, and
THE e290 fact write itself: ||dW|| = 9.1788432658723, t0 read 0.2646476)
carried ~89% of its energy inside its own 10k room but reads in the g0
battery through a function the coupling constant says dies at a few parts
in a thousand of orthogonal drift. WHAT IS THE MINIMUM INJECTABLE WRITE
THAT STILL READS? SVD-truncate the committed delta to a per-matrix rank
ladder, measure what each truncation is (mass, room share, drift budget),
inject each as theta_base + truncated_delta, and probe the t0 read per
rung — the read-floor of the truncated writes, BEFORE any maintenance.
The maintenance half (GPU) starts from this ladder.

FROZEN BARS (VERBATIM from the dispatch, BEFORE any compute):
  - READS-THIN: rank >= 7 truncations all read >= 0.05 at t0 — the write
    is readable at e111's floor; the maintenance half is fully licensed.
  - READS-CLIFF: below some rank the t0 read collapses (< 0.05) — the
    injection floor found at desk; the maintenance ladder starts above it.
  - MIXED: report the per-rank t0 reads verbatim.

OPERATIONALIZATIONS (frozen HERE before compute; they fix the clauses,
they do not move the bars):
  * THE OBJECT: dW := flat(e261_K10K_inst_resume.pt["model"] at its final
    step s400) MINUS flat(e001.pt) — x6's registered convention, md5-bound
    below. N = 2,739,072; flat basis = net.parameters() order (x6's
    G_FLATBASIS, re-verified live).
  * THE TRUNCATION (x6's conventions): per-2D-parameter-matrix SVD in
    fp64; matrix j keeps its top-min(r, rank_j) singular components.
    1D deltas (LN gains/biases, mlp biases — 3.08% of dW's energy by
    x6's read) are carried VERBATIM at every rung: SVD is undefined there
    and x6's convention discloses, never silently drops. Rung r passes a
    matrix through BIT-EXACTLY when rank_j <= r (its top-rank IS the
    matrix); else top-r reconstruction (fp64 compute -> fp32 cast).
    RUNGS {1000, 100, 10, 7} verbatim from the dispatch. Disclosed: the
    largest 2D matrix rank is 192, so rung 1000 IS the full write
    bit-exactly (every matrix passes through) — the built-in sanity rung.
  * DIAGNOSTIC ARM "1D-ONLY" (co-report only, NEVER a bar): 2D deltas
    zeroed, 1D verbatim — separates the 2D truncation channel from the
    1D passthrough in the read.
  * THE READ (t0 probe): e261's g0 battery bit-exactly (corpus seed 1337;
    host_occ via e043's find_occ regex, p >= 280, p+len(host)+119 <= len;
    shuffled random.Random(24301); first 60 = install; rows = the 130-char
    pre-name contexts) read through G1.battery_cell's exact form — mean
    p(Z) at the last position, CPU, bs 30, on TinyGPT(Cfg()) (the default
    6/6/192 cfg IS the 2.74M organism). The floor 0.05 = e261's frozen
    G0_ZERO_FLOOR. gm12 (the 118-char battery) co-reported, never a bar.
  * THE BUDGET (b): T269/e290's measured coupling constant applied to
    each truncation's own norm — the threshold bracket's realized-drift
    fractions [0.0007099904808640174, 0.0032331212727527057] of the
    write's norm (e290's committed threshold_bracket_realized_drift,
    measured on THIS write) x ||dW_r||: the passive-kill budget each
    truncation implies. Also reported as a fraction of the FULL write's
    norm (comparability) — the thinner the write, the smaller its
    absolute safe-drift budget.
  * THE ROOM (a): each truncation's in-room energy share against the
    committed K10K room (runs/checkpoints/e264_rooms.pt, seeds 26113/
    26114 — x6's registered room), SRCT projector ported BY VALUE from
    x6 (fp64 CPU DCT), bit-verified vs fresh construction before trust.
  * INJECTION READINESS (c): each rung written to runs/e306/
    e306_trunc_r{1000,100,10,7}.pt as {"model": theta_base + truncated
    delta (fp32 state_dict), "delta": the per-key fp32 delta,
    "delta_rank": r, provenance}; VERIFIED by re-loading each file from
    disk and re-probing (the probe-from-disk per rank — the checkpoint
    roundtrip is part of the datum).
  * ADJUDICATION (frozen): over the rungs sorted by rank, READS-THIN iff
    ALL rungs' t0 g0 >= 0.05. READS-CLIFF iff the failing rungs (< 0.05)
    form a lower suffix of the rank-descending ladder AND at least one
    rung passes — the injection floor is the bracket (highest failing
    rank, lowest passing rank]. MIXED otherwise (non-monotone failures,
    verbatim per-rank reads). G_FULLREAD failure => TEXTURE, nothing
    adjudicated.
  * GATES: G_ENDPOINTS (md5 binds: e001 d114536d1c0983ab3be67f67ff0667c8,
    e261_K10K_inst_resume 0f6dc1cf46850ce655dfafc9c853d467 — x6's binds),
    G_FLATBASIS (key order == parameters order, no buffers, N exact),
    G_BATTERY (corpus ZEPH-free; install mix FLORIZEL 19 / ELIZABETH 41;
    shapes (60,130)/(60,118) — e261's committed gate literals),
    G_FULLREAD (this desk's full-write probe == e264's committed
    0.26464763283729553 within 5e-3; the exact delta co-reported — the
    smoke read it bit-exactly at 0.0), G_ROOM (D/S bit-equal fresh).
  * CPU ENVELOPE (dispatch): CUDA_VISIBLE_DEVICES="" before torch import;
    torch threads 4; pocketfft workers 4; NO GPU, NO envelope-log writes,
    no lab/gpu code path; datetime.now(UTC) timestamps only.

COMPUTE: 27 fp64 SVDs (largest 768x192) + 6 fp64 DCT projections + 7 CPU
battery probes — minutes, CPU-only.

Outputs: runs/e306/{metrics.json (PROGRESSIVE), e306_truncation.png,
REPORT.md} + runs/e306/e306_trunc_r*.pt. NO NOTES/THINKING/QUEUE/STATE
edits (dispatch — the coordinator folds). Birth-commit BEFORE compute;
final commit AND push.

Run:  python lab/e306_minbearer_desk.py
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU-ONLY, bulletproof

import hashlib
import importlib.util
import json
import math
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.fft as sf
import torch
import torch.nn.functional as F

torch.set_num_threads(4)
DCT_WORKERS = 4

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "runs" / "checkpoints"
OUT = REPO / "runs" / "e306"
OUT.mkdir(parents=True, exist_ok=True)

T0 = time.time()
METRICS: dict = {
    "experiment": "e306_minbearer_desk",
    "phase": "THE MINIMUM BEARER'S DESIGN + THE TRUNCATION INVENTORY (the "
             "desk half of e306): SVD-truncate the committed 10k write to a "
             "per-matrix rank ladder {1000,100,10,7}; measure each rung's "
             "mass/room/budget; inject theta_base+truncated_delta per rung; "
             "probe the t0 read per rung — the read-floor of the truncated "
             "writes, before any maintenance",
    "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "status": "RUNNING (progressive writes)",
    "cpu_only": True,
    "threads": {"torch": 4, "pocketfft": DCT_WORKERS},
}
BARS = {
    "READS-THIN": "rank >= 7 truncations all read >= 0.05 at t0 — the write "
        "is readable at e111's floor; the maintenance half is fully licensed",
    "READS-CLIFF": "below some rank the t0 read collapses (< 0.05) — the "
        "injection floor found at desk; the maintenance ladder starts above it",
    "MIXED": "report the per-rank t0 reads verbatim",
}
OP = {
    "object": "dW = flat(e261_K10K_inst_resume.pt['model'] s400) - "
              "flat(e001.pt) — x6's registered convention (== THE e290 fact "
              "write: ||dW|| 9.1788432658723)",
    "truncation": "per-2D-matrix SVD fp64 (x6's conventions); matrix keeps "
        "top-min(r, rank_j) components; bit-exact passthrough when "
        "rank_j <= r; 1D deltas carried VERBATIM at every rung (SVD "
        "undefined there; x6 discloses, never drops); rungs {1000,100,10,7} "
        "verbatim; rung 1000 == the full write bit-exactly (max 2D rank 192) "
        "— the built-in sanity rung",
    "diagnostic_1d_only": "2D zeroed, 1D verbatim — co-report only, never "
        "a bar",
    "read": "e261's g0 battery bit-exactly (corpus 1337; find_occ + "
        "SPLICE_RNG 24301 shuffle; first-60 install rows; 130-char pre-name "
        "contexts), mean p(Z) at the last position, CPU TinyGPT(Cfg()) "
        "(the 2.74M organism), bs 30; floor 0.05 = e261's G0_ZERO_FLOOR; "
        "gm12 co-reported, never a bar",
    "budget": "e290's committed threshold_bracket_realized_drift fractions "
        "[0.0007099904808640174, 0.0032331212727527057] x ||dW_r|| — the "
        "passive-kill budget each truncation implies (T269: the drift "
        "tolerance scales with the write's norm); co-reported as a fraction "
        "of the full write's norm",
    "room": "in-room energy share vs the committed K10K room "
        "(e264_rooms.pt, seeds 26113/26114; SRCT projector ported BY VALUE "
        "from x6; fp64 CPU DCT; bit-verified vs fresh)",
    "injection": "runs/e306/e306_trunc_r{1000,100,10,7}.pt = "
        "{'model': theta_base+truncated_delta fp32, 'delta': per-key fp32 "
        "delta, provenance}; verified by probe-from-disk per rank",
    "adjudication": "READS-THIN iff all rungs pass (t0 g0 >= 0.05); "
        "READS-CLIFF iff the failing rungs form a lower suffix of the "
        "rank-descending ladder AND >= 1 rung passes (floor = bracket "
        "(highest failing, lowest passing]); MIXED otherwise; G_FULLREAD "
        "failure => TEXTURE",
    "anchors_co_report": "full-write t0 g0 (e264's committed "
        "0.26464763283729553 — this desk re-probes it as G_FULLREAD); "
        "base e001 t0 g0; e261's K1K post g0 0.000435 + e246's rank-10 "
        "ALIGNED 2.86e-5 as committed dead-write context",
}
E290_BRACKET = [0.0007099904808640174, 0.0032331212727527057]
E290_WRITE_NORM = 9.1788432658723            # == ||dW|| (the same object)
E264_K10K_POST_G0 = 0.26464763283729553      # the committed t0 read
G0_FLOOR = 0.05                              # e261's frozen G0_ZERO_FLOOR
RUNGS = [1000, 100, 10, 7]                   # dispatch verbatim


def log(msg: str) -> None:
    print(f"[e306 {time.time() - T0:7.1f}s] {msg}", flush=True)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _strip_private(obj):
    if isinstance(obj, dict):
        return {k: _strip_private(v) for k, v in obj.items()
                if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_strip_private(v) for v in obj]
    return obj


def write_partial(note: str) -> None:
    METRICS["phase_note"] = note
    METRICS["date_updated"] = now_utc()
    tmp = OUT / "metrics.json.tmp"
    tmp.write_text(json.dumps(_strip_private(METRICS), indent=1,
                              default=str), encoding="utf-8")
    tmp.replace(OUT / "metrics.json")
    log(f"WROTE metrics.json ({note})")


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------ G_ENDPOINTS + G_FLATBASIS
ROOT_CK = CKPT / "e001.pt"
K10K_CK = CKPT / "e261_K10K_inst_resume.pt"
ROOMS_CK = CKPT / "e264_rooms.pt"

BINDS = {
    "e001": (ROOT_CK, "d114536d1c0983ab3be67f67ff0667c8",
             "x6's committed root bind (== e267's base bind)"),
    "e261_K10K_inst_resume": (K10K_CK, "0f6dc1cf46850ce655dfafc9c853d467",
                              "x6's bind (== e264 G_K10KRESUME, the s400 "
                              "final)"),
}
md5_reads = {}
for nm, (path, claim, src) in BINDS.items():
    got = md5_of(path)
    md5_reads[nm] = {"path": str(path.relative_to(REPO)), "md5": got,
                     "claimed": claim, "source": src, "match": got == claim}
    if got != claim:
        raise SystemExit(f"ENDPOINT BIND FAILURE: {nm} {got} != {claim}")
METRICS["gates"] = {"G_ENDPOINTS": {"md5_binds": md5_reads, "pass": True}}


def load_sd(path: Path) -> dict:
    """Plain-IO CPU load (x6's honesty rule: no memmap; .float().clone();
    finiteness verified)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    assert isinstance(sd, dict) and all(torch.is_tensor(v) for v in sd.values())
    out = {}
    for k, v in sd.items():
        t = v.float().clone()
        if not torch.isfinite(t).all():
            raise SystemExit(f"NON-FINITE tensor {k} in {path.name}")
        out[k] = t
    return out


# common.py via importlib (x6's convention: TinyGPT/Cfg/CharCorpus only; no
# GPU code path executes — DEVICE resolves 'cpu' under CUDA_VISIBLE_DEVICES='')
_spec = importlib.util.spec_from_file_location("e306_common",
                                               REPO / "lab" / "common.py")
_common = importlib.util.module_from_spec(_spec)
sys.modules["e306_common"] = _common
_spec.loader.exec_module(_common)
TinyGPT, Cfg, CharCorpus = _common.TinyGPT, _common.Cfg, _common.CharCorpus

sd_root = load_sd(ROOT_CK)
sd_post = load_sd(K10K_CK)
KEYS = list(sd_root.keys())
SHAPE_OF = {k: tuple(sd_root[k].shape) for k in KEYS}

_net = TinyGPT(Cfg())
_pnames = [n for n, _ in _net.named_parameters()]
_bnames = [n for n, _ in _net.named_buffers()]
N = int(sum(p.numel() for p in _net.parameters()))
GB = {"params_count": len(_pnames), "buffers_count": len(_bnames),
      "key_order_matches_parameters": _pnames == KEYS, "n_params": N,
      "pass": _pnames == KEYS and len(_bnames) == 0}
if not GB["pass"] or N != 2_739_072:
    raise SystemExit(f"FLAT-BASIS GATE FAILURE: {GB}")
METRICS["gates"]["G_FLATBASIS"] = GB
write_partial("P0 bars frozen in-script + endpoints md5-bound + flat basis")


# ------------------------------------------------ G_BATTERY (e261 bit-exact)
def find_occ(text: str, word: str) -> list[int]:      # e043 verbatim
    pat = re.compile(r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])")
    return [m.start() for m in pat.finditer(text)]


corpus = CharCorpus(REPO / "data" / "input.txt", seed=1337)
stoi, itos = corpus.stoi, corpus.itos
zid = stoi["Z"]
train_ids = corpus.train
train_text = "".join(itos[int(i)] for i in train_ids)
zeph = train_text.count("ZEPH")

HOSTS = ["FLORIZEL", "ELIZABETH"]        # G1.HOSTS (e261's read)
PRE, POST_CAP = 130, 119                 # G1.PRE / G1.POST_CAP
host_occ = []
for host in HOSTS:
    for p in find_occ(train_text, host):
        if p >= 280 and p + len(host) + POST_CAP <= len(train_ids):
            host_occ.append((p, host))
random.Random(24301).shuffle(host_occ)   # E43.SPLICE_RNG
install_occ = host_occ[:60]
mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
       "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}


def battery(j: int) -> torch.Tensor:
    cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
    return torch.stack([corpus.encode(c) for c in cs])


g0_ids, gm12_ids = battery(0), battery(-12)
GT_BAT = {
    "corpus_zeph_count": zeph,
    "install_mix": mix,
    "shapes": {"g0": list(g0_ids.shape), "gm12": list(gm12_ids.shape)},
    "expected": {"zeph": 0, "mix": {"FLORIZEL": 19, "ELIZABETH": 41},
                 "shapes": {"g0": [60, 130], "gm12": [60, 118]}},
    "pass": bool(zeph == 0
                 and mix == {"FLORIZEL": 19, "ELIZABETH": 41}
                 and list(g0_ids.shape) == [60, 130]
                 and list(gm12_ids.shape) == [60, 118]),
}
if not GT_BAT["pass"]:
    raise SystemExit(f"BATTERY GATE FAILURE: {GT_BAT}")
METRICS["gates"]["G_BATTERY"] = GT_BAT
log(f"battery rebuilt bit-exactly (mix {mix}; shapes "
    f"{tuple(g0_ids.shape)}/{tuple(gm12_ids.shape)})")


@torch.no_grad()
def battery_cell(net, ids: torch.Tensor, zid: int, bs=30) -> dict:
    """G1.battery_cell's exact form (e068/e113/e120/e151 battery, CPU)."""
    net.eval()
    pzs, amax = [], 0.0
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs.append(pr[:, zid])
        amax += float((lg[:, -1].argmax(-1) == zid).float().sum())
    p = torch.cat(pzs)
    return {"mean_pz": float(p.mean()), "median_pz": float(p.median()),
            "frac_pz_ge_0.5": float((p >= 0.5).float().mean()),
            "frac_argmax_z": amax / ids.shape[0]}


def probe_delta(delta64: dict | None) -> dict:
    """t0 probe: theta_base + delta through the g0/gm12 batteries."""
    net = TinyGPT(Cfg())
    net.load_state_dict(sd_root)
    with torch.no_grad():
        if delta64 is not None:
            for n, p in net.named_parameters():
                p.add_(delta64[n])
    return {"g0": battery_cell(net, g0_ids, zid),
            "gm12": battery_cell(net, gm12_ids, zid)}


# ------------------------------------------------ THE OBJECT + the SVD cache
dW64: dict[str, torch.Tensor] = {}       # fp64 per-key delta
for k in KEYS:
    dW64[k] = sd_post[k].double() - sd_root[k].double()
dW_flat = torch.cat([dW64[k].reshape(-1) for k in KEYS]).numpy()
DW_L2 = float(np.linalg.norm(dW_flat))
assert abs(DW_L2 - E290_WRITE_NORM) < 1e-9, (DW_L2, E290_WRITE_NORM)
METRICS["object"] = {
    "root": "runs/checkpoints/e001.pt (md5-bound)",
    "post": "runs/checkpoints/e261_K10K_inst_resume.pt s400 (md5-bound)",
    "dW_l2": DW_L2,
    "dW_linf": float(np.abs(dW_flat).max()),
    "N": N,
    "note": "||dW|| bit-matches e290's committed write norm "
            "9.1788432658723 — the coupling constant was measured on THIS "
            "write; the budget arithmetic is same-object",
}
log(f"dW ready: L2 {DW_L2:.4f} (== e290's write norm, asserted)")

# per-2D-matrix SVD cache (fp64; x6's conventions)
svd_cache: dict[str, dict] = {}
rank_of: dict[str, int] = {}
e1d = e2d = 0.0
for k in KEYS:
    sh = SHAPE_OF[k]
    if len(sh) == 1:
        e1d += float((dW64[k] ** 2).sum())
        continue
    U, S, Vh = torch.linalg.svd(dW64[k].reshape(sh))
    svd_cache[k] = {"U": U, "S": S, "Vh": Vh}
    rank_of[k] = int(S.numel())
    e2d += float((S ** 2).sum())
tot = e2d + e1d
SVD_CENSUS = {
    "n_2d": len(svd_cache), "n_1d": len(KEYS) - len(svd_cache),
    "max_2d_rank": max(rank_of.values()),
    "e_1d_share_of_total": e1d / tot,
    "e_2d_share_of_total": e2d / tot,
    "per_matrix_rank": {k: rank_of[k] for k in svd_cache},
}
METRICS["svd_census"] = SVD_CENSUS
write_partial("P1 dW formed (norm-asserted == e290's) + fp64 SVD cache")


def truncate(r: int) -> tuple[dict[str, torch.Tensor], dict]:
    """x6-convention truncation: per-matrix top-min(r, rank_j); bit-exact
    passthrough when rank_j <= r; 1D verbatim. fp32 out (the injection
    currency); fp64 kept for the norm/room arithmetic."""
    delta32, prof = {}, {"passthrough_matrices": [], "truncated_matrices": [],
                         "kept_energy_per_2d": {}, "_flat64": None}
    segs = []
    for k in KEYS:
        sh = SHAPE_OF[k]
        if len(sh) == 1:
            delta32[k] = dW64[k].float()
            segs.append(dW64[k].reshape(-1))
            continue
        rk = rank_of[k]
        if rk <= r:
            delta32[k] = (dW64[k].reshape(sh)).float()
            segs.append(dW64[k].reshape(-1))
            prof["passthrough_matrices"].append(k)
            prof["kept_energy_per_2d"][k] = 1.0
            continue
        U, S, Vh = svd_cache[k]["U"], svd_cache[k]["S"], svd_cache[k]["Vh"]
        rec = (U[:, :r] * S[:r]) @ Vh[:r, :]
        delta32[k] = rec.float()
        segs.append(rec.reshape(-1))
        prof["truncated_matrices"].append(k)
        prof["kept_energy_per_2d"][k] = float(
            (S[:r] ** 2).sum() / (S ** 2).sum())
    prof["_flat64"] = torch.cat(segs).numpy()
    return delta32, prof


# ------------------------------------------------ G_ROOM (x6's SRCT by value)
class SRCT:
    """Ported BY VALUE from x6 (which ported it from e261): the exact
    orthogonal projector P x = D . idct(mask_S(dct(D . x)))."""

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


_ck = torch.load(ROOMS_CK, map_location="cpu", weights_only=False)
_entry = _ck["model"]["K10K"]
_k, _seeds = int(_entry["k"]), [int(x) for x in _entry["seeds"]]
assert _seeds == [26113, 26114], _seeds
room = SRCT(N, _k, _seeds[0], _seeds[1])
_as_np = lambda t: t.numpy() if hasattr(t, "numpy") else np.asarray(t)
D_stored = _as_np(_entry["D_int8"]).astype(np.float64)
S_stored = _as_np(_entry["S"])
GT_ROOM = {
    "file": "runs/checkpoints/e264_rooms.pt", "rung": "K10K", "k": _k,
    "seeds": _seeds,
    "D_bit_equal": bool(np.array_equal(D_stored, room.D)),
    "S_bit_equal": bool(np.array_equal(S_stored, room.S)),
}
GT_ROOM["pass"] = GT_ROOM["D_bit_equal"] and GT_ROOM["S_bit_equal"]
if not GT_ROOM["pass"]:
    raise SystemExit(f"ROOM GATE FAILURE: {GT_ROOM}")
METRICS["gates"]["G_ROOM"] = GT_ROOM
log(f"K10K room loaded + bit-verified vs fresh seeds {_seeds} (k={_k})")


# ------------------------------------------------ G_FULLREAD (the instrument)
full_probe = probe_delta(None)          # base-only first (context anchor)
base_read = full_probe
delta_full32 = {k: dW64[k].float() for k in KEYS}
full_probe = probe_delta(delta_full32)
d_full = abs(full_probe["g0"]["mean_pz"] - E264_K10K_POST_G0)
GT_FULL = {
    "form": "this desk's probe of theta_base + dW (the full write) vs "
            "e264's committed K10K post g0 0.26464763283729553 (the e290 "
            "read denominator)",
    "mine_g0_mean_pz": full_probe["g0"]["mean_pz"],
    "committed": E264_K10K_POST_G0,
    "abs_diff": d_full,
    "bar": 5e-3,
    "pass": bool(d_full <= 5e-3),
}
METRICS["gates"]["G_FULLREAD"] = GT_FULL
METRICS["anchors"] = {
    "base_e001_t0": base_read,
    "full_write_t0": full_probe,
    "committed_context": {
        "e261_K1K_post_g0": 0.00043458465370349586,
        "e246_ALIGNED_rank10_post_g0": 2.8589747671503574e-05,
        "note": "committed dead/near-dead WRITE reads (different objects: "
                "writes FORMED in small rooms — not truncations of this "
                "write); context only, never bars"},
}
write_partial(f"P2 G_FULLREAD {'PASS' if GT_FULL['pass'] else 'FAIL'} "
              f"(|d|={d_full:.2e})")
if not GT_FULL["pass"]:
    METRICS["verdict"] = {"word": "TEXTURE",
                          "why": "G_FULLREAD failure — instrument does not "
                                 "reproduce the committed read; nothing "
                                 "adjudicated"}
    write_partial("HALT: TEXTURE (G_FULLREAD)")
    raise SystemExit("G_FULLREAD FAILURE — nothing adjudicated")


# ------------------------------------------------ (a)+(b)+(c): the ladder
ladder: dict[str, dict] = {}
for r in RUNGS:
    delta32, prof = truncate(r)
    flat64 = prof.pop("_flat64")
    l2 = float(np.linalg.norm(flat64))
    resid = float(np.linalg.norm(dW_flat - flat64))
    p = room.project(flat64)
    in_own = float((p * p).sum() / max(l2 * l2, 1e-30))
    in_vs_full = float((p * p).sum() / (DW_L2 * DW_L2))
    kept = prof["kept_energy_per_2d"]
    rec = {
        "rank": r,
        "dW_r_l2": l2,
        "mass_frac_of_full": l2 / DW_L2,
        "energy_frac_of_full": (l2 * l2) / (DW_L2 * DW_L2),
        "residual_l2": resid,
        "n_dims_2d_bearer": (sum(min(r, rank_of[k]) for k in svd_cache)),
        "passthrough_matrices": prof["passthrough_matrices"],
        "n_truncated_matrices": len(prof["truncated_matrices"]),
        "kept_energy_2d_min": min(kept.values()) if kept else 1.0,
        "kept_energy_2d_median": float(np.median(list(kept.values())))
                                  if kept else 1.0,
        "room": {"inside_share_of_own": in_own,
                 "inside_energy_vs_full": in_vs_full},
        "budget": {
            "form": "e290 realized-drift bracket fractions x ||dW_r||",
            "frac_of_write": E290_BRACKET,
            "abs_l2_lo": E290_BRACKET[0] * l2,
            "abs_l2_hi": E290_BRACKET[1] * l2,
            "as_frac_of_FULL_write_norm_lo": E290_BRACKET[0] * l2 / DW_L2,
            "as_frac_of_FULL_write_norm_hi": E290_BRACKET[1] * l2 / DW_L2,
        },
    }
    # (c) injectable checkpoint + probe-from-disk
    ck_path = OUT / f"e306_trunc_r{r}.pt"
    model_sd = {}
    with torch.no_grad():
        for k in KEYS:
            model_sd[k] = (sd_root[k] + delta32[k]).contiguous()
    torch.save({
        "model": model_sd,
        "delta": delta32,
        "delta_rank": r,
        "delta_l2": l2,
        "theta_base": "runs/checkpoints/e001.pt (md5 "
                      f"{md5_reads['e001']['md5']})",
        "post": "runs/checkpoints/e261_K10K_inst_resume.pt s400 (md5 "
                f"{md5_reads['e261_K10K_inst_resume']['md5']})",
        "convention": OP["truncation"],
        "created": now_utc(),
    }, ck_path)
    # probe FROM DISK (the roundtrip is part of the datum)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    net = TinyGPT(Cfg())
    net.load_state_dict(ck["model"])
    read = {"g0": battery_cell(net, g0_ids, zid),
            "gm12": battery_cell(net, gm12_ids, zid)}
    roundtrip_ok = all(
        torch.equal(ck["model"][k], model_sd[k]) for k in KEYS)
    rec["checkpoint"] = {"path": str(ck_path.relative_to(REPO)),
                         "roundtrip_bitexact": bool(roundtrip_ok)}
    rec["t0_read"] = read
    ladder[f"r{r}"] = rec
    log(f"rung r{r}: L2 {l2:.4f} ({100 * rec['mass_frac_of_full']:.2f}% "
        f"mass) in-room {100 * in_own:.1f}% budget "
        f"[{rec['budget']['abs_l2_lo']:.5f}, "
        f"{rec['budget']['abs_l2_hi']:.5f}] t0 g0 "
        f"{read['g0']['mean_pz']:.5f}")
METRICS["ladder"] = ladder
write_partial("P3 the truncation ladder complete (norms+room+budgets+probes)")

# diagnostic 1D-ONLY (co-report only, never a bar)
d1d = {k: (dW64[k].float() if len(SHAPE_OF[k]) == 1
           else torch.zeros(SHAPE_OF[k])) for k in KEYS}
f1d = torch.cat([d.cpu().double().reshape(-1) for d in d1d.values()]).numpy()
l1d = float(np.linalg.norm(f1d))
p1d = room.project(f1d)
net1d = TinyGPT(Cfg())
net1d.load_state_dict(sd_root)
with torch.no_grad():
    for n_, p_ in net1d.named_parameters():
        p_.add_(d1d[n_])
METRICS["diagnostic_1d_only"] = {
    "form": "2D deltas zeroed, 1D verbatim — co-report only, NEVER a bar",
    "l2": l1d, "mass_frac_of_full": l1d / DW_L2,
    "energy_frac_of_full": (l1d * l1d) / (DW_L2 * DW_L2),
    "room_inside_share_of_own": float((p1d * p1d).sum() / (l1d * l1d)),
    "t0_read": {"g0": battery_cell(net1d, g0_ids, zid),
                "gm12": battery_cell(net1d, gm12_ids, zid)},
}
log(f"1D-ONLY diagnostic: L2 {l1d:.4f} ({100 * l1d / DW_L2:.2f}% mass) "
    f"t0 g0 {METRICS['diagnostic_1d_only']['t0_read']['g0']['mean_pz']:.5f}")
write_partial("P4 1D-ONLY diagnostic done (co-report)")

# ------------------------------------------------ adjudication (frozen)
reads = {r: ladder[f"r{r}"]["t0_read"]["g0"]["mean_pz"] for r in RUNGS}
order = sorted(RUNGS, reverse=True)          # rank-descending
flags = {r: reads[r] >= G0_FLOOR for r in order}
if all(flags[r] for r in order):
    verdict = "READS-THIN"
    clause = ("all rungs read >= 0.05 at t0 — the write is readable at "
              "e111's floor down to the rank-7 bearer; the maintenance "
              "half is fully licensed")
elif all(not flags[r] for r in order):
    verdict = "TEXTURE"      # unreachable while G_FULLREAD passes (r1000)
    clause = "all rungs below floor — impossible after G_FULLREAD pass"
else:
    first_fail = next((r for r in order if not flags[r]), None)
    suffix_ok = (all(not flags[r]
                     for r in order[order.index(first_fail):])
                 if first_fail is not None else False)
    if suffix_ok:
        verdict = "READS-CLIFF"
        lo_pass = order[order.index(first_fail) - 1]
        clause = (f"the t0 read collapses below rank {first_fail} "
                  f"(< {G0_FLOOR}); the injection floor bracket is "
                  f"({first_fail}, {lo_pass}] — the maintenance ladder "
                  f"starts above it")
    else:
        verdict = "MIXED"
        clause = ("non-monotone across the ladder — per-rank t0 reads "
                  "verbatim, no inflation")
METRICS["verdict"] = {
    "word": verdict,
    "clause": clause,
    "per_rank_t0_g0": reads,
    "floor": G0_FLOOR,
    "bars_verbatim": BARS,
    "operationalizations": OP,
    "clause_trace": {
        "all_rungs_ge_floor": bool(all(flags[r] for r in order)),
        "failing_rungs": [r for r in order if not flags[r]],
        "failing_set_is_lower_suffix": bool(
            suffix_ok if verdict == "READS-CLIFF"
            else all(flags[r] for r in order)),
    },
}
write_partial(f"P5 ADJUDICATED: {verdict}")

# ------------------------------------------------ figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

ax = axes[0]
rs = sorted(RUNGS)
mass = [100 * ladder[f"r{r}"]["mass_frac_of_full"] for r in rs]
ener = [100 * ladder[f"r{r}"]["energy_frac_of_full"] for r in rs]
blo = [100 * ladder[f"r{r}"]["budget"]["as_frac_of_FULL_write_norm_lo"]
       for r in rs]
bhi = [100 * ladder[f"r{r}"]["budget"]["as_frac_of_FULL_write_norm_hi"]
       for r in rs]
ax.plot(rs, mass, "o-", color="crimson", lw=1.4,
        label="mass ||dW_r||/||dW|| (%)")
ax.plot(rs, ener, "s--", color="darkorange", lw=1.2,
        label="energy share (%)")
ax.fill_between(rs, blo, bhi, color="steelblue", alpha=0.25,
                label="passive-kill drift budget (e290 bracket, "
                      "% of FULL norm)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("per-matrix truncation rank (log)")
ax.set_ylabel("% of the full write (log)")
ax.set_title("(a) THE TRUNCATION INVENTORY — mass, energy, budget")
ax.legend(fontsize=8, loc="lower left")
ax.grid(alpha=0.25, which="both")

ax = axes[1]
inown = [100 * ladder[f"r{r}"]["room"]["inside_share_of_own"] for r in rs]
infull = [100 * ladder[f"r{r}"]["room"]["inside_share_of_own"]
          * ladder[f"r{r}"]["energy_frac_of_full"] for r in rs]
ax.plot(rs, inown, "o-", color="crimson", lw=1.4,
        label="in-room share of the truncation (%)")
ax.plot(rs, infull, "s--", color="steelblue", lw=1.2,
        label="in-room energy vs the FULL write (%)")
ax.axhline(100 * 0.8914359886030179, color="gray", ls=":", lw=1.0,
           label="full write's inside share (x6: 89.14%)")
ax.set_xscale("log")
ax.set_xlabel("per-matrix truncation rank (log)")
ax.set_ylabel("%")
ax.set_title("(b) THE ROOM'S READING — where the thin writes live")
ax.legend(fontsize=8, loc="best")
ax.grid(alpha=0.25, which="both")

ax = axes[2]
rd = [ladder[f"r{r}"]["t0_read"]["g0"]["mean_pz"] for r in rs]
ax.plot(rs, rd, "o-", color="crimson", lw=1.6, label="t0 g0 read (this desk)")
ax.axhline(G0_FLOOR, color="k", ls="--", lw=1.0,
           label=f"e111 floor / G0_ZERO_FLOOR = {G0_FLOOR}")
ax.axhline(E264_K10K_POST_G0, color="gray", ls=":", lw=1.0,
           label="full write t0 (e264 committed 0.2646)")
ax.scatter([RUNGS[-1]], [METRICS["diagnostic_1d_only"]["t0_read"]["g0"]
                         ["mean_pz"]], marker="x", color="steelblue",
           label="1D-ONLY diagnostic (never a bar)")
ax.set_xscale("log")
ax.set_xlabel("per-matrix truncation rank (log)")
ax.set_ylabel("mean p(Z) at t0")
ax.set_title(f"(c) THE READ-FLOOR — verdict: {verdict}")
ax.legend(fontsize=8, loc="center left")
ax.grid(alpha=0.25, which="both")

fig.suptitle("e306 DESK HALF — the minimum bearer's truncation inventory: "
             f"the t0 read-floor of the truncated 10k write (verdict "
             f"{verdict})", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "e306_truncation.png", dpi=140)
log("FIGURE written")

# ------------------------------------------------ REPORT.md
rows_norm = []
rows_read = []
for r in sorted(RUNGS, reverse=True):
    L = ladder[f"r{r}"]
    rows_norm.append(
        f"| {r} | {L['dW_r_l2']:.4f} | "
        f"{100 * L['mass_frac_of_full']:.3f}% | "
        f"{100 * L['energy_frac_of_full']:.3f}% | "
        f"{L['residual_l2']:.4f} | {L['n_dims_2d_bearer']} | "
        f"{100 * L['room']['inside_share_of_own']:.2f}% | "
        f"{L['budget']['abs_l2_lo']:.5f} - "
        f"{L['budget']['abs_l2_hi']:.5f} |")
    rows_read.append(
        f"| {r} | {L['t0_read']['g0']['mean_pz']:.6f} | "
        f"{'PASS' if L['t0_read']['g0']['mean_pz'] >= G0_FLOOR else 'FAIL'} "
        f"| {L['t0_read']['gm12']['mean_pz']:.6f} | "
        f"{L['t0_read']['g0']['frac_pz_ge_0.5']:.3f} | "
        f"{L['t0_read']['g0']['frac_argmax_z']:.3f} | "
        f"{L['checkpoint']['roundtrip_bitexact']} |")
d1 = METRICS["diagnostic_1d_only"]
report = f"""# e306 DESK HALF — THE MINIMUM BEARER'S TRUNCATION INVENTORY ({verdict})

**The object:** the committed 10k quiet write `e261_K10K_inst_resume.pt`
(s400, md5-bound) minus the fresh root `e001.pt` (md5-bound) — x6's
registered object, and THE e290 fact write itself: ||dW|| = {DW_L2:.4f}
(bit-matches e290's committed write norm; the coupling constant was
measured on this very write). Truncated per x6's conventions — per-2D-matrix
SVD (fp64), top-min(r, rank_j), 1D deltas carried verbatim (SVD undefined
there; x6 discloses, never drops), rungs {{1000, 100, 10, 7}} verbatim from
the dispatch. CPU-only desk cell; no training, no GPU.

**Instrument:** e261's g0 battery rebuilt bit-exactly (mix FLORIZEL 19 /
ELIZABETH 41, shapes (60,130)/(60,118), corpus ZEPH-free — the committed
gate literals); this desk's probe of the FULL write returns
{METRICS['gates']['G_FULLREAD']['mine_g0_mean_pz']:.17g} vs e264's committed
{E264_K10K_POST_G0:.17g} (|d| = {d_full:.1e}; G_FULLREAD {'PASS' if GT_FULL['pass'] else 'FAIL'}).
The base e001 root reads {base_read['g0']['mean_pz']:.2e} at t0.

## (a)+(b) The truncation inventory + the budgets each rung implies

| rank | \\|\\|dW_r\\|\\| | mass vs full | energy vs full | residual | 2D bearer dims | in-room (own) | passive-kill budget (abs L2, e290 bracket) |
|---|---|---|---|---|---|---|---|
{chr(10).join(rows_norm)}

The dispatch's question — "a rank-7 write is ~0.1% of full rank's mass?"
— answers: **{100 * ladder['r7']['mass_frac_of_full']:.2f}% of the mass**
({100 * ladder['r7']['energy_frac_of_full']:.2f}% of the energy). The 1D
passthrough carries {100 * d1['mass_frac_of_full']:.2f}% of the mass
(x6's 3.08% energy share). The drift budget scales with the write's own
norm (T269): the rank-7 bearer's absolute safe-drift window is
[{ladder['r7']['budget']['abs_l2_lo']:.5f}, {ladder['r7']['budget']['abs_l2_hi']:.5f}] L2
vs the full write's [{E290_BRACKET[0] * DW_L2:.5f}, {E290_BRACKET[1] * DW_L2:.5f}]
— the thinner the write, the proportionally smaller its tolerance.

## (c) THE INJECTION READINESS — the t0 read per rank (the desk datum)

| rank | t0 g0 mean p(Z) | vs floor {G0_FLOOR} | t0 gm12 | frac >= 0.5 | frac argmax | ckpt roundtrip |
|---|---|---|---|---|---|---|
{chr(10).join(rows_read)}

Injectable checkpoints written + probe-verified from disk at every rung:
`runs/e306/e306_trunc_r{{1000,100,10,7}}.pt` (theta_base + truncated_delta
fp32 + the per-key delta). The 1D-ONLY diagnostic (2D zeroed, 1D verbatim;
co-report only, never a bar): t0 g0 = {d1['t0_read']['g0']['mean_pz']:.6f}.

## Verdict: {verdict}

{METRICS['verdict']['clause']}. Per-rank t0 g0 reads verbatim:
{json.dumps(reads)}. Committed context (different objects — writes FORMED
in small rooms, not truncations of this write): e261's K1K write read
0.000435; e246's rank-10 ALIGNED read 2.86e-5.

Disclosures: rung 1000 is the full write BIT-EXACTLY (the largest 2D
matrix rank is 192 — the rung is the dispatch's verbatim sanity rung, and
its probe doubles as G_FULLREAD); rung 100 passes wte/lm_head (rank 65)
through untouched; the room is the committed K10K room (e264_rooms.pt,
seeds 26113/26114, D/S bit-verified vs fresh construction); the budget
bracket is e290's committed realized-drift fractions
[{E290_BRACKET[0]:.10f}, {E290_BRACKET[1]:.10f}] of the write's norm,
applied same-object; n=1 per rung (one lineage, one session — the g-series
standing lottery note carried verbatim).
"""
(OUT / "REPORT.md").write_text(report, encoding="utf-8")

METRICS["outputs"] = {
    "figure": "runs/e306/e306_truncation.png",
    "metrics": "runs/e306/metrics.json",
    "report": "runs/e306/REPORT.md",
    "checkpoints": [f"runs/e306/e306_trunc_r{r}.pt" for r in RUNGS],
}
METRICS["builds_on"] = [
    "x6 (THE registered object + the per-matrix SVD conventions + the SRCT "
    "projector ported by value + the md5 binds)",
    "e261/e264 (the committed 10k write + its s400 resume; the g0 battery "
    "conventions; the committed post g0 0.26464763283729553)",
    "T269/e290 (the coupling constant: threshold bracket realized-drift "
    "fractions of the write's norm — measured on THIS write)",
    "e111 (the k*=7 reading floor the rank-7 rung honors)",
]
METRICS["whats_new"] = [
    "THE TRUNCATION INVENTORY: the committed 10k write's per-matrix rank "
    "ladder {1000,100,10,7} with each rung's mass/room/budget measured",
    "THE READ-FLOOR: the first per-rank t0 probe of truncated writes — "
    "the minimum injectable write that still reads, found at desk BEFORE "
    "any maintenance compute",
    "INJECTABLE CHECKPOINTS: theta_base + truncated_delta per rank, "
    "probe-verified from disk — the maintenance half's ready starting "
    "states",
]
METRICS["status"] = "COMPLETE — adjudicated"
write_partial("P6 COMPLETE (report + figure)")
log(f"DONE — verdict {verdict}")
