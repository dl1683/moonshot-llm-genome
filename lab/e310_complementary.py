"""E310 — THE COMPLEMENTARY BEARER CONTROL (R70's minted cell #1; dispatched
2026-10-09, commit f52a004). This docstring carries the registered bars
VERBATIM from the dispatch letter, committed at birth BEFORE any compute.
Adjudicate against exactly this; no bar shopping.

THE CIRCULARITY (R70's critic iii, REVIEWS.md): e306's desk half claimed the
read's bearer is the ROOM-OVERLAP TAIL spectrum — but the room was built FROM
this write's own install stream, so "room-overlap" may just mean
"formation-stream-overlap" (the read is carried by the spectrum that formed
it — circular). THE MISSING CONTROL: the COMPLEMENTARY truncation — the full
write MINUS its room-overlap components: high mass, LOW room-share. If the
complementary write reads DEAD, the bearer claim survives its control
(room-overlap is necessary); if it reads ALIVE, the bearer story falsifies
(it was rank/formation, not room) and Law 3's ~7-dims clause revives.

FROZEN BARS (VERBATIM from the dispatch, BEFORE any compute):
  - READ-DEAD-MASS-HIGH (the bearer claim SURVIVES its control): the
    complementary write reads < 0.05 (the floor) while carrying >= 10% of
    the full write's energy — room-overlap is NECESSARY for the read; the
    bearer is the tail that overlaps the room.
  - READ-ALIVE (the bearer story FALSIFIED): the complement reads >= 0.05 —
    the read survives without room-overlap; the circularity was real
    (formation-stream or rank, not room); Law 3's ~7-dims clause revives for
    reconciliation.
  - MIXED: the complement dead but the mix mid — the curve verbatim; a
    partial-overlap threshold reading.

OPERATIONALIZATIONS (frozen HERE before compute; they fix the clauses, they
do not move the bars):
  * THE OBJECT: dW := flat(e261_K10K_inst_resume.pt["model"] at its final
    step s400) MINUS flat(e001.pt) — x6's registered convention, md5-bound
    below. N = 2,739,072; ||dW|| asserted == 9.1788432658723 (e290's write
    norm). THE ROOM: the committed K10K room (e264_rooms.pt, seeds
    26113/26114 — x6's registered room), SRCT projector ported BY VALUE
    (fp64 CPU DCT), D/S bit-verified vs fresh construction before trust.
  * THE COMPLEMENT (a): delta_complement := (I - P_room) applied to the FULL
    write delta in the flat fp64 basis, reshaped per-key. P is an exactly
    orthogonal projector (ortho DCT + sign flips, D^2=1), so the complement
    is orthogonal to the room component BY CONSTRUCTION — G_ORTH asserts the
    identity on dW (||PdW||^2 + ||comp||^2 == ||dW||^2, rel < 1e-9, and
    <PdW, comp> ~ 0). Expect ~11% energy at 89% in-room original (dispatch).
  * INJECTION (b): theta_base + delta_complement (fp32 injection currency,
    e306's checkpoint conventions: {"model", "delta", provenance});
    VERIFIED from disk — bit-equal roundtrip + model == base+delta
    re-derived + probe-from-disk (the roundtrip is part of the datum).
  * THE MIX (c, THE registered bar arm): "50/50 MIXED, energy-matched" :=
    the mix's TOTAL energy equals the FULL write's (the dose anchor) with
    exactly 50% carried by the room-overlap spectrum P(dW) and 50% by the
    complement — coefficients a_in = sqrt(0.5/f_in), a_out = sqrt(0.5/f_out)
    (disclosed verbatim: this AMPLIFIES the complement spectrum ~2.15x to
    reach its 50%; the in-room spectrum is scaled DOWN ~0.75x).
  * MIX-COMPDOSE (co-report only, NEVER a bar): the same 50/50 spectrum
    split with total energy == the COMPLEMENT's (dose-matched to the
    complement arm) — the dose-control on the mix reading.
  * MID (the MIXED bar's reading, frozen): the registered mix reads
    g0 >= 0.05 AND <= 0.5 x the full write's committed t0 read (<= 0.1323)
    — alive but at or below half the full read.
  * THE READ (t0 probe): e261's g0 battery bit-exactly (corpus seed 1337;
    host_occ via e043's find_occ regex, p >= 280; shuffled
    random.Random(24301); first 60 = install; rows = the 130-char pre-name
    contexts) read through G1.battery_cell's exact form — mean p(Z) at the
    last position, CPU, bs 30, on TinyGPT(Cfg()) (the 2.74M organism).
    The floor 0.05 = e261's frozen G0_ZERO_FLOOR. gm12 co-reported, never
    a bar.
  * THE FORWARD LADDER ANCHOR (d): e306's committed ladder read LIVE from
    runs/e306/metrics.json (full 0.26464763283729553 / r100
    0.09254436194896698 / r10 5.39e-05 dead / r7 dead) — the bearer curve
    from BOTH directions; the r10 rung (14.7% energy, 9.0% in-room, DEAD)
    is the mass confound comparator, disclosed at the claim site.
  * COMPLEMENT CENSUS (co-report): per-2D-matrix fp64 SVD of the complement
    (T3/T10 energy-share medians + top matrices) — the rank texture for the
    READ-ALIVE branch's "formation-stream or rank" reading; never a bar.
  * ADJUDICATION (frozen precedence): READ-ALIVE iff complement g0 >= 0.05;
    else READ-DEAD-MASS-HIGH iff complement g0 < 0.05 AND complement energy
    >= 10% of the full write's; else MIXED (a mass-low dead read —
    disclosed, no necessity claim). In EVERY dead-complement outcome the
    mix reading is co-reported verbatim (mid => the partial-overlap
    threshold reading). G_FULLREAD failure => TEXTURE, nothing adjudicated.
  * GATES: G_ENDPOINTS (md5 binds: e001 d114536d1c0983ab3be67f67ff0667c8,
    e261_K10K_inst_resume 0f6dc1cf46850ce655dfafc9c853d467 — x6's binds),
    G_FLATBASIS (key order == parameters order, no buffers, N exact),
    G_BATTERY (corpus ZEPH-free; install mix FLORIZEL 19 / ELIZABETH 41;
    shapes (60,130)/(60,118) — e261's committed gate literals),
    G_ROOM (D/S bit-equal fresh), G_ORTH (projector identity on dW),
    G_FULLREAD (this desk's full-write probe == e264's committed
    0.26464763283729553 within 5e-3).
  * CPU ENVELOPE (dispatch): CUDA_VISIBLE_DEVICES="" before torch import;
    torch threads 4; pocketfft workers 4; NO GPU, NO envelope-log writes,
    no lab/gpu code path; datetime.now(UTC) timestamps only.

COMPUTE: 5 fp64 DCT projections + 27 fp64 SVDs (census) + 5 CPU battery
probes — minutes, CPU-only.

Outputs: runs/e310/{metrics.json (PROGRESSIVE), e310_complementary.png,
REPORT.md} + runs/e310/e310_{complement,mix5050_full,mix5050_compdose}.pt.
NO NOTES/THINKING/QUEUE/STATE edits (dispatch — the coordinator folds).
Birth-commit BEFORE compute; final commit AND push.

Run:  python lab/e310_complementary.py
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
OUT = REPO / "runs" / "e310"
OUT.mkdir(parents=True, exist_ok=True)

T0 = time.time()
METRICS: dict = {
    "experiment": "e310_complementary",
    "phase": "THE COMPLEMENTARY BEARER CONTROL (R70's minted cell #1): the "
             "full write MINUS its room-overlap components (the orthogonal "
             "complement; high mass, LOW room-share) injected and probed at "
             "t0 — the circularity-killer for e306's bearer claim; + a 50/50 "
             "energy-matched MIX arm as the dose-response midpoint",
    "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "status": "RUNNING (progressive writes)",
    "cpu_only": True,
    "threads": {"torch": 4, "pocketfft": DCT_WORKERS},
}
BARS = {
    "READ-DEAD-MASS-HIGH": "the complementary write reads < 0.05 (the floor) "
        "while carrying >= 10% of the full write's energy — room-overlap is "
        "NECESSARY for the read; the bearer is the tail that overlaps the "
        "room (the bearer claim SURVIVES its control)",
    "READ-ALIVE": "the complement reads >= 0.05 — the read survives without "
        "room-overlap; the circularity was real (formation-stream or rank, "
        "not room); Law 3's ~7-dims clause revives for reconciliation (the "
        "bearer story FALSIFIED)",
    "MIXED": "the complement dead but the mix mid — the curve verbatim; a "
        "partial-overlap threshold reading",
}
OP = {
    "object": "dW = flat(e261_K10K_inst_resume.pt['model'] s400) - "
              "flat(e001.pt) — x6's registered convention (== THE e290 fact "
              "write: ||dW|| 9.1788432658723)",
    "room": "the committed K10K room (e264_rooms.pt, seeds 26113/26114; SRCT "
            "projector ported BY VALUE from x6; fp64 CPU DCT; D/S "
            "bit-verified vs fresh construction)",
    "complement": "delta_complement = (I - P_room) applied to the FULL write "
                  "delta in the flat fp64 basis, per-key reshaped; P is an "
                  "exactly orthogonal projector so comp ⊥ room component BY "
                  "CONSTRUCTION (G_ORTH asserts the identity on dW)",
    "injection": "theta_base + delta fp32 (e306's checkpoint conventions: "
                 "{'model', 'delta', provenance}); VERIFIED from disk "
                 "(bit-equal roundtrip + model == base+delta re-derived + "
                 "probe-from-disk)",
    "mix_registered": "50/50 energy-matched: total energy == the FULL "
                      "write's, exactly 50% from P(dW) and 50% from the "
                      "complement (a_in = sqrt(0.5/f_in) ~ 0.749 scales the "
                      "room spectrum DOWN; a_out = sqrt(0.5/f_out) ~ 2.146 "
                      "AMPLIFIES the complement ~2.15x)",
    "mix_compdose": "the same 50/50 spectrum split with total energy == the "
                    "complement's (dose-matched to the complement arm) — "
                    "co-report only, NEVER a bar (the dose-control on the "
                    "mix reading)",
    "mid_definition": "MID := mix g0 >= 0.05 AND <= 0.5 x the full write's "
                      "committed t0 read (<= 0.1323) — alive but at or below "
                      "half",
    "read": "e261's g0 battery bit-exactly (corpus 1337; find_occ + "
            "SPLICE_RNG 24301 shuffle; first-60 install rows; 130-char "
            "pre-name contexts), mean p(Z) at the last position, CPU "
            "TinyGPT(Cfg()) (the 2.74M organism), bs 30; floor 0.05 = "
            "e261's G0_ZERO_FLOOR; gm12 co-reported, never a bar",
    "ladder_anchor": "e306's committed ladder read LIVE from "
                     "runs/e306/metrics.json (full / r100 / r10 / r7) — the "
                     "bearer curve from BOTH directions; r10 (14.7% energy, "
                     "9.0% in-room, DEAD) is the mass confound comparator",
    "census": "per-2D-matrix fp64 SVD of the complement (T3/T10 medians + "
              "top matrices) — co-report for the rank reading, never a bar",
    "adjudication": "READ-ALIVE iff complement g0 >= 0.05; else "
                    "READ-DEAD-MASS-HIGH iff complement g0 < 0.05 AND "
                    "complement energy >= 10% of the full write's; else "
                    "MIXED (mass-low dead, disclosed, no necessity claim); "
                    "in every dead-complement outcome the mix reading is "
                    "co-reported verbatim (mid => partial-overlap "
                    "threshold); G_FULLREAD failure => TEXTURE",
}
E290_WRITE_NORM = 9.1788432658723            # == ||dW|| (the same object)
E264_K10K_POST_G0 = 0.26464763283729553      # the committed t0 read
E306_R100_G0 = 0.09254436194896698           # e306's committed r100 read
G0_FLOOR = 0.05                              # e261's frozen G0_ZERO_FLOOR
MASS_HIGH_BAR = 0.10                         # >= 10% of the full write energy


def log(msg: str) -> None:
    print(f"[e310 {time.time() - T0:7.1f}s] {msg}", flush=True)


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
_spec = importlib.util.spec_from_file_location("e310_common",
                                               REPO / "lab" / "common.py")
_common = importlib.util.module_from_spec(_spec)
sys.modules["e310_common"] = _common
_spec.loader.exec_module(_common)
TinyGPT, Cfg, CharCorpus = _common.TinyGPT, _common.Cfg, _common.CharCorpus

sd_root = load_sd(ROOT_CK)
sd_post = load_sd(K10K_CK)
KEYS = list(sd_root.keys())
SHAPE_OF = {k: tuple(sd_root[k].shape) for k in KEYS}
OFFS: dict[str, tuple[int, int]] = {}
_o = 0
for k in KEYS:
    n = sd_root[k].numel()
    OFFS[k] = (_o, _o + n)
    _o += n

_net = TinyGPT(Cfg())
_pnames = [n for n, _ in _net.named_parameters()]
_bnames = [n for n, _ in _net.named_buffers()]
N = int(sum(p.numel() for p in _net.parameters()))
GB = {"params_count": len(_pnames), "buffers_count": len(_bnames),
      "key_order_matches_parameters": _pnames == KEYS, "n_params": N,
      "offsets_sum_equals_N": _o == N,
      "pass": _pnames == KEYS and len(_bnames) == 0 and _o == N}
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
mix_hosts = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
             "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}


def battery(j: int) -> torch.Tensor:
    cs = [train_text[p - PRE - j: p] for p, _ in install_occ]
    return torch.stack([corpus.encode(c) for c in cs])


g0_ids, gm12_ids = battery(0), battery(-12)
GT_BAT = {
    "corpus_zeph_count": zeph,
    "install_mix": mix_hosts,
    "shapes": {"g0": list(g0_ids.shape), "gm12": list(gm12_ids.shape)},
    "expected": {"zeph": 0, "mix": {"FLORIZEL": 19, "ELIZABETH": 41},
                 "shapes": {"g0": [60, 130], "gm12": [60, 118]}},
    "pass": bool(zeph == 0
                 and mix_hosts == {"FLORIZEL": 19, "ELIZABETH": 41}
                 and list(g0_ids.shape) == [60, 130]
                 and list(gm12_ids.shape) == [60, 118]),
}
if not GT_BAT["pass"]:
    raise SystemExit(f"BATTERY GATE FAILURE: {GT_BAT}")
METRICS["gates"]["G_BATTERY"] = GT_BAT
log(f"battery rebuilt bit-exactly (mix {mix_hosts}; shapes "
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


# ------------------------------------------------ THE OBJECT + (a) complement
dW64: dict[str, torch.Tensor] = {}       # fp64 per-key delta (the full write)
for k in KEYS:
    dW64[k] = sd_post[k].double() - sd_root[k].double()
dW_flat = torch.cat([dW64[k].reshape(-1) for k in KEYS]).numpy()
DW_L2 = float(np.linalg.norm(dW_flat))
assert abs(DW_L2 - E290_WRITE_NORM) < 1e-9, (DW_L2, E290_WRITE_NORM)
E_FULL = float(dW_flat @ dW_flat)

pin_flat = room.project(dW_flat)                       # P(dW): the overlap
comp_flat = dW_flat - pin_flat                         # (I-P)dW: THE COMPLEMENT
E_IN = float(pin_flat @ pin_flat)
E_OUT = float(comp_flat @ comp_flat)
cross = float(pin_flat @ comp_flat)
GT_ORTH = {
    "form": "P orthogonal projector (ortho DCT + D^2=1): ||PdW||^2 + "
            "||comp||^2 == ||dW||^2 and <PdW, comp> == 0, on THIS dW",
    "energy_sum_rel_resid": abs(E_IN + E_OUT - E_FULL) / E_FULL,
    "cross_dot_rel": abs(cross) / E_FULL,
    "bar": 1e-9,
    "pass": bool(abs(E_IN + E_OUT - E_FULL) / E_FULL < 1e-9
                 and abs(cross) / E_FULL < 1e-9),
}
if not GT_ORTH["pass"]:
    raise SystemExit(f"ORTH GATE FAILURE: {GT_ORTH}")
METRICS["gates"]["G_ORTH"] = GT_ORTH

F_IN, F_OUT = E_IN / E_FULL, E_OUT / E_FULL


def unflat(flat64: np.ndarray) -> dict[str, torch.Tensor]:
    out = {}
    for k in KEYS:
        a, b = OFFS[k]
        out[k] = torch.from_numpy(
            np.ascontiguousarray(flat64[a:b])).reshape(SHAPE_OF[k])
    return out


comp64 = unflat(comp_flat)               # fp64 per-key complement
pin64 = unflat(pin_flat)                 # fp64 per-key room-overlap part
comp32 = {k: comp64[k].float() for k in KEYS}   # injection currency
COMP_L2 = float(np.linalg.norm(comp_flat))

# in-room share of the INJECTED fp32 objects (what the organism receives)
def flat_of(d32: dict) -> np.ndarray:
    return torch.cat([d32[k].double().reshape(-1)
                      for k in KEYS]).numpy()


def in_room_share(flat: np.ndarray) -> float:
    p = room.project(flat)
    return float((p @ p) / max(flat @ flat, 1e-30))


comp32_inroom = in_room_share(flat_of(comp32))
full32 = {k: dW64[k].float() for k in KEYS}
full32_inroom = in_room_share(flat_of(full32))

METRICS["object"] = {
    "root": "runs/checkpoints/e001.pt (md5-bound)",
    "post": "runs/checkpoints/e261_K10K_inst_resume.pt s400 (md5-bound)",
    "dW_l2": DW_L2, "N": N,
    "note": "||dW|| bit-matches e290's committed write norm — same object as "
            "e306's ladder and the coupling constant",
}
METRICS["complement"] = {
    "form": "delta_complement = (I - P_room) dW, flat fp64 basis, per-key "
            "reshaped; orthogonal to the room component BY CONSTRUCTION "
            "(G_ORTH)",
    "l2": COMP_L2,
    "mass_frac_of_full": COMP_L2 / DW_L2,
    "energy_frac_of_full": F_OUT,
    "room_overlap_part": {"l2": float(np.linalg.norm(pin_flat)),
                          "energy_frac_of_full": F_IN},
    "in_room_share_of_own_fp64": 0.0,   # by construction (G_ORTH: comp ⊥ room)
    "in_room_share_of_injected_fp32": comp32_inroom,
    "full_write_in_room_share_remeasured_fp32": full32_inroom,
    "x6_committed_full_in_room_share": 0.8914359886030179,
    "dispatch_expectation": "~11% energy at 89% in-room original",
}
write_partial(f"P1 complement built: L2 {COMP_L2:.4f} "
              f"({100 * COMP_L2 / DW_L2:.2f}% mass, {100 * F_OUT:.2f}% "
              f"energy); G_ORTH PASS")
log(f"complement: L2 {COMP_L2:.4f} = {100 * COMP_L2 / DW_L2:.2f}% mass, "
    f"{100 * F_OUT:.2f}% energy (room-overlap part carries "
    f"{100 * F_IN:.2f}%)")


# ------------------------------------------------ G_FULLREAD (the instrument)
base_read = probe_delta(None)            # base-only context anchor
full_probe = probe_delta(full32)
d_full = abs(full_probe["g0"]["mean_pz"] - E264_K10K_POST_G0)
GT_FULL = {
    "form": "this desk's probe of theta_base + dW (the full write) vs "
            "e264's committed K10K post g0 0.26464763283729553",
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


# ------------------------------------------------ (b) INJECT + verify + probe
def inject_and_probe(name: str, delta32: dict, delta_kind: str,
                     extra: dict | None = None) -> dict:
    """e306's checkpoint conventions: save theta_base+delta fp32 + the
    per-key delta; verify from disk (bit-equal roundtrip + model == base+delta
    re-derived); probe FROM DISK (the roundtrip is part of the datum)."""
    ck_path = OUT / f"e310_{name}.pt"
    model_sd = {}
    with torch.no_grad():
        for k in KEYS:
            model_sd[k] = (sd_root[k] + delta32[k]).contiguous()
    payload = {
        "model": model_sd,
        "delta": delta32,
        "delta_kind": delta_kind,
        "theta_base": f"runs/checkpoints/e001.pt (md5 "
                      f"{md5_reads['e001']['md5']})",
        "post": f"runs/checkpoints/e261_K10K_inst_resume.pt s400 (md5 "
                f"{md5_reads['e261_K10K_inst_resume']['md5']})",
        "convention": OP["injection"],
        "created": now_utc(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, ck_path)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    roundtrip = all(torch.equal(ck["model"][k], model_sd[k]) for k in KEYS)
    rederived = all(
        torch.equal(ck["model"][k], sd_root[k] + ck["delta"][k])
        for k in KEYS)
    delta_pass = all(
        torch.equal(ck["delta"][k], delta32[k]) for k in KEYS)
    net = TinyGPT(Cfg())
    net.load_state_dict(ck["model"])
    read = {"g0": battery_cell(net, g0_ids, zid),
            "gm12": battery_cell(net, gm12_ids, zid)}
    fl = flat_of(ck["delta"])
    rec = {
        "delta_kind": delta_kind,
        "delta_l2": float(np.linalg.norm(fl)),
        "energy_frac_of_full": float((fl @ fl) / E_FULL),
        "in_room_share_of_injected_fp32": in_room_share(fl),
        "checkpoint": {"path": str(ck_path.relative_to(REPO)),
                       "roundtrip_bitexact": bool(roundtrip),
                       "model_eq_base_plus_delta": bool(rederived),
                       "delta_bitexact": bool(delta_pass)},
        "t0_read": read,
    }
    if extra:
        rec.update({k: v for k, v in extra.items()
                    if isinstance(v, (int, float, str, bool))})
    return rec


arms: dict[str, dict] = {}
arms["complement"] = inject_and_probe(
    "complement", comp32, "(I - P_room) dW (the complementary write)",
    extra={"mass_frac_of_full": COMP_L2 / DW_L2})
log(f"COMPLEMENT arm: t0 g0 {arms['complement']['t0_read']['g0']['mean_pz']:.6g} "
    f"(in-room share of injected fp32: "
    f"{arms['complement']['in_room_share_of_injected_fp32']:.2e})")
write_partial("P3 THE COMPLEMENT injected + disk-verified + probed "
              "(THE datum)")

# ------------------------------------------------ (c) the MIX arms
a_in, a_out = math.sqrt(0.5 / F_IN), math.sqrt(0.5 / F_OUT)
mixfull_flat = a_in * pin_flat + a_out * comp_flat
mixfull64 = unflat(mixfull_flat)
mixfull32 = {k: mixfull64[k].float() for k in KEYS}

b_in, b_out = math.sqrt(0.5 * F_OUT / F_IN), math.sqrt(0.5)
mixcomp_flat = b_in * pin_flat + b_out * comp_flat
mixcomp64 = unflat(mixcomp_flat)
mixcomp32 = {k: mixcomp64[k].float() for k in KEYS}

METRICS["mix_construction"] = {
    "registered_mix": {
        "form": "a_in*P(dW) + a_out*comp with a_in = sqrt(0.5/f_in), "
                "a_out = sqrt(0.5/f_out); each half EXACTLY 50% of the full "
                "write's energy; total == the full write's energy",
        "a_in": a_in, "a_out": a_out,
        "disclosure": f"the room spectrum is scaled {a_in:.4f}x (DOWN); the "
                      f"complement spectrum is AMPLIFIED {a_out:.4f}x to "
                      "reach its 50%",
    },
    "compdose_mix_diagnostic": {
        "form": "b_in*P(dW) + b_out*comp with b_in = sqrt(0.5*f_out/f_in), "
                "b_out = sqrt(0.5); 50/50 spectrum split at the COMPLEMENT's "
                "total energy — co-report only, NEVER a bar",
        "b_in": b_in, "b_out": b_out,
    },
}

arms["mix5050_full"] = inject_and_probe(
    "mix5050_full", mixfull32,
    "50/50 energy-matched mix: total energy == full write's, half P(dW) + "
    "half complement (THE registered mix arm)", extra={"a_in": a_in,
                                                       "a_out": a_out})
arms["mix5050_compdose"] = inject_and_probe(
    "mix5050_compdose", mixcomp32,
    "50/50 spectrum mix at the COMPLEMENT's total energy (dose-control "
    "diagnostic; co-report only, never a bar)", extra={"b_in": b_in,
                                                       "b_out": b_out})
for nm in ("mix5050_full", "mix5050_compdose"):
    A = arms[nm]
    log(f"{nm}: t0 g0 {A['t0_read']['g0']['mean_pz']:.6g} "
        f"(energy {100 * A['energy_frac_of_full']:.2f}% of full; in-room "
        f"{100 * A['in_room_share_of_injected_fp32']:.2f}%)")
write_partial("P4 the MIX arms injected + probed (registered + compdose)")

# ------------------------------------------------ complement census (co-report)
t3s, t10s, e_mat = [], [], {}
e1d_out = e2d_out = 0.0
for k in KEYS:
    e = float((comp64[k] ** 2).sum())
    if len(SHAPE_OF[k]) == 1:
        e1d_out += e
        continue
    e2d_out += e
    U, S, Vh = torch.linalg.svd(comp64[k].reshape(SHAPE_OF[k]))
    tot = float((S ** 2).sum())
    if tot > 0:
        t3s.append(float((S[:3] ** 2).sum()) / tot)
        t10s.append(float((S[:10] ** 2).sum()) / tot)
    e_mat[k] = e
top3 = sorted(e_mat.items(), key=lambda kv: -kv[1])[:3]
METRICS["complement_census"] = {
    "form": "per-2D-matrix fp64 SVD of the complement — co-report for the "
            "rank reading (READ-ALIVE branch interpretation), never a bar",
    "e_1d_share_of_complement": e1d_out / E_OUT,
    "e_2d_share_of_complement": e2d_out / E_OUT,
    "per_matrix_T3_median": float(np.median(t3s)),
    "per_matrix_T10_median": float(np.median(t10s)),
    "top3_matrices_by_energy": [{"param": k, "energy_share_of_2d": v / e2d_out}
                                for k, v in top3],
    "e307_null_context": "e307's null-gaussian per-matrix T3 ~0.002, the "
                         "controller footprints ~0.19-0.23, flat twins "
                         "~0.12 (different objects; context only)",
}
log(f"complement census: median per-matrix T3 "
    f"{METRICS['complement_census']['per_matrix_T3_median']:.4f} / T10 "
    f"{METRICS['complement_census']['per_matrix_T10_median']:.4f}")

# ------------------------------------------------ (d) the ladder + the table
e306_m = json.loads((REPO / "runs" / "e306" / "metrics.json").read_text(
    encoding="utf-8"))
LADDER = {}
for r in (1000, 100, 10, 7):
    L_ = e306_m["ladder"][f"r{r}"]
    LADDER[f"r{r}"] = {
        "t0_g0": L_["t0_read"]["g0"]["mean_pz"],
        "energy_frac_of_full": L_["energy_frac_of_full"],
        "mass_frac_of_full": L_["mass_frac_of_full"],
        "in_room_share_of_own": L_["room"]["inside_share_of_own"],
        "source": "runs/e306/metrics.json (committed; read live)",
    }
assert abs(LADDER["r1000"]["t0_g0"] - E264_K10K_POST_G0) < 1e-12
assert abs(LADDER["r100"]["t0_g0"] - E306_R100_G0) < 1e-9
METRICS["forward_ladder_e306"] = LADDER

comp_g0 = arms["complement"]["t0_read"]["g0"]["mean_pz"]
comp_energy = arms["complement"]["energy_frac_of_full"]
mix_g0 = arms["mix5050_full"]["t0_read"]["g0"]["mean_pz"]
MID_HI = 0.5 * E264_K10K_POST_G0
mix_is_mid = bool(G0_FLOOR <= mix_g0 <= MID_HI)

# ------------------------------------------------ adjudication (frozen)
if comp_g0 >= G0_FLOOR:
    verdict = "READ-ALIVE"
    clause = (f"the complement reads {comp_g0:.6g} >= 0.05 while carrying "
              f"{100 * comp_energy:.2f}% of the full write's energy at ~0% "
              "room-share — the read survives WITHOUT room-overlap; the "
              "circularity was real (formation-stream or rank, not room); "
              "Law 3's ~7-dims clause revives for reconciliation "
              "(THE_LAWS_V2 Law 3: 'Reading takes ~7 dims (e111)')")
elif comp_energy >= MASS_HIGH_BAR:
    verdict = "READ-DEAD-MASS-HIGH"
    clause = (f"the complementary write reads {comp_g0:.6g} < 0.05 while "
              f"carrying {100 * comp_energy:.2f}% >= 10% of the full "
              "write's energy — room-overlap is NECESSARY for the read; the "
              "bearer is the tail that overlaps the room; e306's bearer "
              "claim SURVIVES R70's circularity control")
else:
    verdict = "MIXED"
    clause = (f"the complement reads {comp_g0:.6g} < 0.05 but carries only "
              f"{100 * comp_energy:.2f}% < 10% of the full write's energy — "
              "a mass-low dead read; no necessity claim (disclosed)")
mid_note = None
if comp_g0 < G0_FLOOR:
    mid_note = {
        "mix_g0": mix_g0,
        "mid_definition": OP["mid_definition"],
        "is_mid": mix_is_mid,
        "reading": "the partial-overlap threshold reading (the curve "
                   "verbatim)" if mix_is_mid else
                  ("the mix reads "
                   + ("above half the full read — overlap-plus-complement "
                      "at matched dose approaches the full read"
                      if mix_g0 > MID_HI else "dead — nothing reads without "
                      "sufficient room-overlap at this dose")),
    }
METRICS["verdict"] = {
    "word": verdict,
    "clause": clause,
    "complement_t0_g0": comp_g0,
    "complement_energy_frac_of_full": comp_energy,
    "mass_high_bar": MASS_HIGH_BAR,
    "floor": G0_FLOOR,
    "mix_t0_g0": mix_g0,
    "mix_is_mid": mix_is_mid,
    "mid_note": mid_note,
    "bars_verbatim": BARS,
    "operationalizations": OP,
}
write_partial(f"P5 ADJUDICATED: {verdict}")

# ------------------------------------------------ figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

ax = axes[0]
pts = [
    (full32_inroom, E264_K10K_POST_G0, "FULL write (e306 r1000)", "k", "o"),
    (arms["mix5050_full"]["in_room_share_of_injected_fp32"], mix_g0,
     "MIX 50/50 (full dose)", "crimson", "s"),
    (arms["mix5050_compdose"]["in_room_share_of_injected_fp32"],
     arms["mix5050_compdose"]["t0_read"]["g0"]["mean_pz"],
     "MIX 50/50 (comp dose, diag)", "darkorange", "v"),
    (arms["complement"]["in_room_share_of_injected_fp32"], comp_g0,
     "COMPLEMENT (I-P)dW", "steelblue", "D"),
]
for x, y, lb, c, m in pts:
    ax.scatter([max(x, 1e-4)], [max(y, 1e-7)], color=c, marker=m, s=55,
               zorder=3, label=lb)
ax.axhline(G0_FLOOR, color="k", ls="--", lw=1.0,
           label=f"floor {G0_FLOOR} (e261 G0_ZERO_FLOOR)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("in-room energy share of the injected delta (log)")
ax.set_ylabel("t0 g0 read (log)")
ax.set_title("(a) THE BEARER CURVE FROM BOTH DIRECTIONS — "
             f"verdict {verdict}")
ax.legend(fontsize=7.5, loc="lower right")
ax.grid(alpha=0.25, which="both")

ax = axes[1]
dose_pts = [
    (LADDER["r1000"]["energy_frac_of_full"], E264_K10K_POST_G0,
     "full (89% in-room)", "k", "o"),
    (LADDER["r100"]["energy_frac_of_full"], LADDER["r100"]["t0_g0"],
     "e306 r100 (67% in-room)", "gray", "o"),
    (LADDER["r10"]["energy_frac_of_full"], LADDER["r10"]["t0_g0"],
     "e306 r10 (9% in-room) DEAD", "gray", "x"),
    (arms["mix5050_full"]["energy_frac_of_full"], mix_g0,
     "mix 50/50 full-dose (50% in-room)", "crimson", "s"),
    (arms["complement"]["energy_frac_of_full"], comp_g0,
     "COMPLEMENT (0% in-room)", "steelblue", "D"),
    (arms["mix5050_compdose"]["energy_frac_of_full"],
     arms["mix5050_compdose"]["t0_read"]["g0"]["mean_pz"],
     "mix comp-dose (50% in-room)", "darkorange", "v"),
]
for x, y, lb, c, m in dose_pts:
    ax.scatter([x], [max(y, 1e-7)], color=c, marker=m, s=55, zorder=3,
               label=lb)
ax.axhline(G0_FLOOR, color="k", ls="--", lw=1.0, label=f"floor {G0_FLOOR}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("total energy vs the full write (log) — the dose")
ax.set_ylabel("t0 g0 read (log)")
ax.set_title("(b) THE DOSE CONTROL — r10 is the mass confound comparator")
ax.legend(fontsize=7.5, loc="lower right")
ax.grid(alpha=0.25, which="both")

ax = axes[2]
names = ["FULL write", "MIX 50/50\n(full dose)", "COMPLEMENT"]
objs = [(F_IN, E264_K10K_POST_G0, arms["complement"]["t0_read"]["g0"]),
        (arms["mix5050_full"]["in_room_share_of_injected_fp32"], mix_g0,
         arms["mix5050_full"]["t0_read"]["g0"]),
        (arms["complement"]["in_room_share_of_injected_fp32"], comp_g0,
         arms["complement"]["t0_read"]["g0"])]
ypos = np.arange(len(names))[::-1]
for y, nm, (fin, g0, _) in zip(ypos, names, objs):
    ax.barh(y, 100 * fin, color="crimson", label="in-room energy %" if y ==
            ypos[0] else None)
    ax.barh(y, 100 * (1 - fin), left=100 * fin, color="steelblue",
            label="out-of-room energy %" if y == ypos[0] else None)
    ax.text(101.5, y, f"t0 g0 = {g0:.4g}", va="center", fontsize=9)
ax.set_yticks(ypos)
ax.set_yticklabels(names, fontsize=9)
ax.set_xlim(0, 132)
ax.set_xlabel("% of the arm's own energy")
ax.set_title("(c) THE SPLIT — where each arm's energy lives")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.25, axis="x")

fig.suptitle("e310 — THE COMPLEMENTARY BEARER CONTROL: the full write minus "
             "its room-overlap components (R70's circularity-killer; verdict "
             f"{verdict})", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(OUT / "e310_complementary.png", dpi=140)
log("FIGURE written")

# ------------------------------------------------ REPORT.md
def row(nm, l2, mass, ener, inroom, g0, extra=""):
    return (f"| {nm} | {l2:.4f} | {100 * mass:.2f}% | {100 * ener:.2f}% | "
            f"{100 * inroom:.2f}% | {g0:.6g} | "
            f"{'ALIVE' if g0 >= G0_FLOOR else 'DEAD'} |{extra}")


tbl = []
tbl.append(row("FULL write (e306 r1000 == bit-exact)", DW_L2, 1.0, 1.0,
               LADDER["r1000"]["in_room_share_of_own"],
               LADDER["r1000"]["t0_g0"], " (anchor)"))
tbl.append(row("e306 r100", 8.0212, LADDER["r100"]["mass_frac_of_full"],
               LADDER["r100"]["energy_frac_of_full"],
               LADDER["r100"]["in_room_share_of_own"],
               LADDER["r100"]["t0_g0"], " (anchor)"))
tbl.append(row("e306 r10", 3.5246, LADDER["r10"]["mass_frac_of_full"],
               LADDER["r10"]["energy_frac_of_full"],
               LADDER["r10"]["in_room_share_of_own"],
               LADDER["r10"]["t0_g0"], " (anchor)"))
tbl.append(row("COMPLEMENT (I-P)dW (THIS CELL)",
               arms["complement"]["delta_l2"],
               arms["complement"]["delta_l2"] / DW_L2,
               arms["complement"]["energy_frac_of_full"],
               arms["complement"]["in_room_share_of_injected_fp32"],
               comp_g0, " (THE control)"))
tbl.append(row("MIX 50/50 full-dose (THIS CELL)",
               arms["mix5050_full"]["delta_l2"],
               arms["mix5050_full"]["delta_l2"] / DW_L2,
               arms["mix5050_full"]["energy_frac_of_full"],
               arms["mix5050_full"]["in_room_share_of_injected_fp32"],
               mix_g0, " (the midpoint)"))
tbl.append(row("MIX 50/50 comp-dose (diag, never a bar)",
               arms["mix5050_compdose"]["delta_l2"],
               arms["mix5050_compdose"]["delta_l2"] / DW_L2,
               arms["mix5050_compdose"]["energy_frac_of_full"],
               arms["mix5050_compdose"]["in_room_share_of_injected_fp32"],
               arms["mix5050_compdose"]["t0_read"]["g0"]["mean_pz"],
               " (dose control)"))
cen = METRICS["complement_census"]
report = f"""# e310 — THE COMPLEMENTARY BEARER CONTROL ({verdict})

**The circularity (R70's critic iii):** e306's desk half claimed the read's
bearer is the room-overlap tail spectrum — but the room was built FROM this
write's own install stream, so "room-overlap" may just mean
"formation-stream-overlap". **This cell is the missing control:** the
COMPLEMENTARY truncation — the full write MINUS its room-overlap components:
{100 * arms['complement']['mass_frac_of_full']:.2f}% of the full mass,
{100 * arms['complement']['energy_frac_of_full']:.2f}% of its energy
(dispatch expected ~11% energy at 89% in-room original — measured
{100 * F_IN:.2f}% in-room), at ~0% room-share by construction (G_ORTH: the
projector identity holds at rel
{METRICS['gates']['G_ORTH']['energy_sum_rel_resid']:.1e}).

**Instrument:** e261's g0 battery bit-exactly (mix FLORIZEL 19 / ELIZABETH
41, shapes (60,130)/(60,118), corpus ZEPH-free); this desk's probe of the
FULL write returns {full_probe['g0']['mean_pz']:.17g} vs e264's committed
{E264_K10K_POST_G0:.17g} (|d| = {d_full:.1e}; G_FULLREAD PASS). The base
e001 root reads {base_read['g0']['mean_pz']:.2e} at t0. All arms injected
per e306's checkpoint conventions and probed FROM DISK (bit-equal roundtrip
+ model == base+delta re-derived, all arms).

## (d) THE COMPARISON TABLE — the bearer curve from BOTH directions

| arm | L2 | mass vs full | energy vs full | in-room share (own) | t0 g0 | read | role |
|---|---|---|---|---|---|---|---|
{chr(10).join(tbl)}

## Verdict: {verdict}

{clause}. {('Mix reading co-reported: ' + json.dumps(mid_note)) if mid_note else ''}

**The honest confound comparator:** e306's r10 rung — {100 * LADDER['r10']['energy_frac_of_full']:.1f}% of the full
write's energy, {100 * LADDER['r10']['in_room_share_of_own']:.1f}% in-room —
also read DEAD ({LADDER['r10']['t0_g0']:.3g}); the complement carries
{100 * comp_energy:.2f}%. The frozen bar registers >= 10% energy as
sufficient mass; the registered mix arm (total energy == the full write's,
50% in-room) is the dose-controlled view: it reads {mix_g0:.6g}
({('MID — the partial-overlap threshold reading' if mix_is_mid else ('above half the full read' if mix_g0 > MID_HI else 'dead'))}).

Disclosures: the complement is orthogonal to the room BY CONSTRUCTION (the
projector identity is the gate, not a finding); the fp32 injection leaks
{100 * arms['complement']['in_room_share_of_injected_fp32']:.1e}% of the
complement's energy back in-room (quantization; disclosed); the registered
mix AMPLIFIES the complement spectrum {a_out:.3f}x and scales the room
spectrum {a_in:.3f}x to hit the 50/50 energy split at the full dose; the
comp-dose mix (never a bar) holds dose at the complement's energy — it reads
{arms['mix5050_compdose']['t0_read']['g0']['mean_pz']:.6g}; the complement's
per-matrix rank texture (census, never a bar): median T3
{cen['per_matrix_T3_median']:.4f} / T10 {cen['per_matrix_T10_median']:.4f}
(e307 context: null ~0.002, flat twins ~0.12, controller footprints
~0.19-0.23), top-3 matrices {', '.join(t['param'] for t in cen['top3_matrices_by_energy'])};
n=1 per arm (one lineage, one session — the g-series standing lottery note
carried verbatim).
"""
(OUT / "REPORT.md").write_text(report, encoding="utf-8")

METRICS["arms"] = arms
METRICS["outputs"] = {
    "figure": "runs/e310/e310_complementary.png",
    "metrics": "runs/e310/metrics.json",
    "report": "runs/e310/REPORT.md",
    "checkpoints": [f"runs/e310/e310_{nm}.pt" for nm in
                    ("complement", "mix5050_full", "mix5050_compdose")],
}
METRICS["builds_on"] = [
    "e306 desk (the truncation machinery, the checkpoint conventions, the "
    "g0 probe instrument, the committed forward ladder — read live from "
    "runs/e306/metrics.json)",
    "x6 (THE registered object + the SRCT projector ported by value + the "
    "md5 binds + the 89.14% in-room census)",
    "e261/e264 (the committed 10k write + its s400 resume; the g0 battery "
    "conventions; the committed post g0 0.26464763283729553)",
    "R70 (critic iii: the bearer-circularity; critic vi: Law 3's ~7-dims "
    "clause unreconciled — the cell that mints this control)",
    "e307 (the per-matrix spectrum census conventions reused for the "
    "complement's rank texture)",
]
METRICS["whats_new"] = [
    "THE COMPLEMENTARY TRUNCATION: the first injection of the full write "
    "MINUS its room-overlap components — the control e306's bearer claim "
    "was missing (high mass, ~0% room-share)",
    "THE BEARER CURVE FROM BOTH DIRECTIONS: forward (rank truncations, "
    "e306) + backward (room-projection complement, this cell) + the 50/50 "
    "energy-matched midpoint",
    "THE DOSE CONTROL ON THE MIX: a comp-dose 50/50 mix co-report (same "
    "spectrum split at the complement's energy) separating spectrum from "
    "dose in the mix reading",
]
METRICS["status"] = "COMPLETE — adjudicated"
write_partial("P6 COMPLETE (report + figure)")
log(f"DONE — verdict {verdict}")
