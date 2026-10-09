"""X15 — THE DOSE-MATCHED COMPLEMENT (R71's critic land iii, the sharpest dose
confound; dispatched 2026-10-09 from the R71 fold). This docstring carries the
question, the arms, the bars and P-x15a VERBATIM from the dispatch letter,
committed at birth BEFORE any compute. Adjudicate against exactly this; no bar
shopping.

THE QUESTION (verbatim): does the out-of-room complement read when scaled to
the FULL write's dose?

THE CONFOUND (R71 critic iii): e310 installed the COMPLEMENT of a memory write
— the full 10k-room write MINUS its room-overlap components, perpendicular to
the room by construction — and it read DEAD (3.66e-05) while carrying 10.86%
of the full write's energy (32.95% of its mass). The critic's attack: the
complement was never probed at FULL dose — no dose-matched alive/dead pair
exists (e306's own r10 rung at 14.7% energy also died; everything inside a
~9x energy gap dies). The bearer-necessity claim ("room-overlap is necessary
for the read") is confounded by dose until the complement is scaled to the
full write's norm and probed.

THE DESIGN (dispatch, verbatim in substance):
  1. Regenerate or load e310's complement vector bit-exact
     (lab/e310_complementary.py's own construction; bind md5s of every loaded
     parent artifact in the gates, as e310 did).
  2. PRIMARY ARM: scale the complement by the exact scalar making its L2 norm
     equal the full write's norm (the critic computes ~x3.033; derived HERE
     from the committed norms — the letter's arithmetic is NOT trusted; the
     true value is ~x3.0350). Inject into the base organism by e310's exact
     method. Probe the read battery (the install's own 60-window bank, same
     battery as e310).
  3. DOSE LADDER (cheap, same session): complements at ~1x (committed
     3.66e-05 reference — verified from this run), ~x1.5, ~x2.25, ~x3.035
     (full). Four points make the curve visible.
  4. CONTROLS re-verified in-session: the full write's read (expect
     ~0.2646-class) and the unscaled complement (expect ~3.66e-05) — both
     must reproduce their committed values before any arm counts.

BARS (frozen VERBATIM from the dispatch, BEFORE any compute):
  - DEAD-AT-FULL-DOSE: full-dose complement reads < 0.01 — THE BEARER LAW
    SURVIVES DOSE-MATCHED: structure without room-overlap cannot read at any
    dose; the read needs the room's tail to couple into the context.
  - ALIVE: full-dose complement reads >= 0.05 — THE NECESSITY CLAIM FALLS as
    a dose artifact: out-of-room structure CAN read when dense enough; the
    bearer claim reduces to "dose and geometry both matter" (Law-of-3-era
    wording must be re-scoped).
  - GAP [0.01, 0.05) — neither frozen bar fires: report verbatim, read as
    dead by e310's original 0.05 floor but NOT by this stricter frozen bar;
    the necessity claim survives WEAKENED at the 0.01 bar; NO wording change
    without a new registered cell (registered here, before compute, so the
    gap cannot be shopped after the fact).

REGISTERED PREDICTION P-x15a (frozen VERBATIM from the dispatch, BEFORE any
compute): "Register prediction P-x15a BEFORE compute. The lab's registered
guess (from T278's author): DEAD-AT-FULL-DOSE — the e306 mix ladder (100%
in-room 0.2646 / 50% at full dose 0.0631 / 0% at 33% dose dead) trends to
zero with in-room share, and the mix's 0% arm was complement-dose-dead.
State your own counter-prediction honestly if you disagree; predictions are
scored, never shopped."
  EXECUTOR POSITION (registered before compute): CONCUR with DEAD-AT-FULL-
  DOSE. Reasoning: at fixed FULL dose the in-room-share trend is monotone
  (89% in-room -> 0.2646; 50% -> 0.0631; a ~4.2x drop per halving); the
  0%-in-room endpoint sits two more halvings below the 50% point -> expected
  << 0.01. Executor sub-prediction P-x15a-exec (sharper, scored): the
  full-dose complement reads < 1e-3 (i.e. at least 50x below the dead bar,
  not merely under it); the ladder stays within one order of the unscaled
  3.66e-05 reference (a flat-dead curve), with the mix-compdose point
  (1.23e-04 at 50% in-room, 10.86% energy) as the nearest committed anchor.
  DISCLOSED letter arithmetic (verified against runs/e310/metrics.json):
  the true scale is x3.0350 (9.1788432658723 / 3.024341960836486), not the
  letter's ~x3.033; and the letter's "0% at 33% dose" mix arm is the
  comp-dose mix whose ENERGY share is 10.86% of full (its 32.95% figure is
  the complement's MASS share) — the trend statement is unaffected.

OPERATIONALIZATIONS (frozen here before compute; they fix the clauses, they
do not move the bars):
  * THE OBJECT: exactly e310's registered object — dW := flat(
    e261_K10K_inst_resume.pt["model"] at s400) MINUS flat(e001.pt); N =
    2,739,072; ||dW|| asserted == 9.1788432658723 (e290's write norm). THE
    ROOM: the committed K10K room (e264_rooms.pt, seeds 26113/26114), SRCT
    projector ported BY VALUE (fp64 CPU DCT), D/S bit-verified vs fresh
    construction before trust.
  * THE COMPLEMENT: regenerated by e310's exact construction —
    delta_complement := (I - P_room) applied to the FULL write delta in the
    flat fp64 basis, reshaped per-key; G_ORTH asserts the projector identity
    on dW; G_REGEN asserts the fp32-cast regenerated complement is BIT-EQUAL
    to e310's committed on-disk delta (runs/e310/e310_complement.pt["delta"],
    md5-bound) and its fp64 L2 == e310's committed 3.024341960836486 (1e-9).
  * THE SCALE: s_full := ||dW|| / ||comp64|| computed in fp64 from the
    regenerated vectors (~3.0350). G_DOSEMATCH asserts ||s_full * comp64|| ==
    ||dW|| to 1e-12 (the mathematical dose-match). The INJECTION is e310's
    exact currency (fp64 -> fp32 cast; checkpoint conventions {"model",
    "delta", provenance}, disk roundtrip verified); the fp32 layer's own norm
    deviation (~1e-7 relative, same as the full write's own fp32 cast, which
    reproduced the committed read bit-exactly in e310) is DISCLOSED per arm,
    never gated.
  * DOSE LADDER: multipliers [1.0, 1.5, 2.25, s_full]; arm comp_x1.000 IS
    the unscaled-complement control (G_COMPREF). Energy fractions of the
    full write: 10.86% / 24.43% / 54.96% / 100.00% (squared multipliers).
  * G_INROOM_SCALED: scaling cannot change direction — verified anyway: the
    in-room energy share of every scaled arm (fp64 measured AND fp32
    injected) asserted <= 1e-12 (e310 measured 2.35e-18 at 1x).
  * THE READ (t0 probe): e261's g0 battery bit-exactly (corpus seed 1337;
    host_occ via e043's find_occ regex, p >= 280; shuffled
    random.Random(24301); first 60 = install; rows = the 130-char pre-name
    contexts) read through G1.battery_cell's exact form — mean p(Z) at the
    last position, CPU, bs 30, on TinyGPT(Cfg()) (the 2.74M organism).
    gm12 co-reported, never a bar. HONESTY-RIDER CO-REPORTS (never bars):
    mean last-position top-1 probability and mean entropy on the same bank
    per arm — separates "the read is dead" from "the organism collapsed"
    (the full-dose complement carries 9.2x the full write's OUT-of-room
    energy; if confidence/entropy wreck while the full-write control does
    not, the dead read is trivially guaranteed and must be stamped so).
  * CONTROL BANDS (frozen): G_FULLREAD |d| <= 5e-3 vs e264/e310's committed
    0.26464763283729553 (e310's own gate bar verbatim); G_COMPREF mine in
    [0.5x, 2x] of e310's committed 3.655568798421882e-05. Both controls run
    BEFORE the scaled arms; EITHER failure => TEXTURE, nothing adjudicated.
  * ADJUDICATION (frozen precedence): let g := full-dose complement g0.
    DEAD-AT-FULL-DOSE iff g < 0.01; ALIVE iff g >= 0.05; else the GAP
    reading above. P-x15a and P-x15a-exec scored verbatim against the
    outcome; predictions are scored, never shopped.
  * GATES: G_ENDPOINTS (md5 binds: e001 d114536d1c0983ab3be67f67ff0667c8,
    e261_K10K_inst_resume 0f6dc1cf46850ce655dfafc9c853d467, e264_rooms
    2d524655575cce00a3bc1c8770f4b211, e310_complement.pt
    bbddb3eac2b652280f38babe92a5467f, e310 metrics
    713fd5c01f1b946ed7148b41a837dc58, e306 metrics
    a4f7a55ae15f6db0f809352aa182d6ff), G_FLATBASIS, G_BATTERY (e261's
    committed literals), G_ROOM, G_ORTH, G_REGEN, G_FULLREAD, G_COMPREF,
    G_DOSEMATCH, G_INROOM_SCALED — 10 gates.
  * CPU ENVELOPE (dispatch): CUDA_VISIBLE_DEVICES="" before torch import;
    torch threads 4; pocketfft workers 4; NO GPU, NO envelope-log writes,
    no lab/gpu code path; datetime.now(UTC) timestamps only.

COMPUTE: ~8 fp64 DCT projections + 8 CPU battery probes — minutes, CPU-only.

Outputs: runs/x15/{metrics.json (PROGRESSIVE), x15_dose_ladder.png,
REPORT.md} + runs/x15/x15_comp_x{1.000,1.500,2.250,FULL}.pt. NO
NOTES/THINKING/QUEUE/STATE edits (dispatch — the coordinator folds).
Birth-commit BEFORE compute; smoke pass disclosed; final commit AND push.

Run:  python lab/x15_dose_complement.py [smoke]
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

SMOKE = len(sys.argv) > 1 and sys.argv[1].lower() == "smoke"

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "runs" / "checkpoints"
OUT = REPO / "runs" / "x15"
OUT.mkdir(parents=True, exist_ok=True)

T0 = time.time()
METRICS: dict = {
    "experiment": "x15_dose_complement",
    "phase": "THE DOSE-MATCHED COMPLEMENT (R71 critic iii): e310's "
             "out-of-room complement scaled to the FULL write's norm "
             "(~x3.0350) and probed at t0 — the dose-matched alive/dead "
             "pair the bearer-necessity claim was missing; + a 4-point dose "
             "ladder (1x / 1.5x / 2.25x / FULL) with both committed "
             "controls re-verified in-session",
    "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "status": "RUNNING (progressive writes)" if not SMOKE else
              "SMOKE (gates + controls only; nothing adjudicated)",
    "cpu_only": True,
    "threads": {"torch": 4, "pocketfft": DCT_WORKERS},
    "mode": "SMOKE" if SMOKE else "FULL",
}
BARS = {
    "DEAD-AT-FULL-DOSE": "full-dose complement reads < 0.01 — THE BEARER LAW "
        "SURVIVES DOSE-MATCHED: structure without room-overlap cannot read "
        "at any dose; the read needs the room's tail to couple into the "
        "context.",
    "ALIVE": "full-dose complement reads >= 0.05 — THE NECESSITY CLAIM "
        "FALLS as a dose artifact: out-of-room structure CAN read when "
        "dense enough; the bearer claim reduces to 'dose and geometry both "
        "matter' (Law-of-3-era wording must be re-scoped).",
    "GAP_[0.01,0.05)": "neither frozen bar fires: report verbatim, read as "
        "dead by e310's original 0.05 floor but NOT by this stricter frozen "
        "bar; the necessity claim survives WEAKENED at the 0.01 bar; NO "
        "wording change without a new registered cell (frozen pre-compute).",
}
P_X15A = {
    "registered_guess": "DEAD-AT-FULL-DOSE (T278's author) — the e306 mix "
        "ladder (100% in-room 0.2646 / 50% at full dose 0.0631 / 0% at 33% "
        "dose dead) trends to zero with in-room share, and the mix's 0% arm "
        "was complement-dose-dead.",
    "executor_position": "CONCUR (registered pre-compute); sub-prediction "
        "P-x15a-exec: full-dose complement reads < 1e-3 (>= 50x below the "
        "dead bar), ladder flat-dead within one order of the 3.66e-05 "
        "unscaled reference.",
    "scoring_rule": "predictions are scored, never shopped.",
}
OP = {
    "object": "dW = flat(e261_K10K_inst_resume.pt['model'] s400) - "
              "flat(e001.pt) — e310's registered object (== THE e290 fact "
              "write: ||dW|| 9.1788432658723); N = 2,739,072",
    "room": "the committed K10K room (e264_rooms.pt, seeds 26113/26114; "
            "SRCT projector ported BY VALUE from x6 via e310; fp64 CPU "
            "DCT; D/S bit-verified vs fresh construction)",
    "complement": "REGENERATED by e310's exact construction: (I - P_room) "
                  "applied to the full write delta in the flat fp64 basis; "
                  "G_REGEN proves bit-exactness vs e310's committed "
                  "on-disk fp32 delta",
    "scale": "s_full = ||dW|| / ||comp64|| in fp64 (~3.0350; the letter's "
             "~3.033 is WRONG by ~0.1% — derived, not trusted); "
             "G_DOSEMATCH: ||s_full*comp64|| == ||dW|| to 1e-12",
    "ladder": "multipliers [1.0, 1.5, 2.25, s_full]; comp_x1.000 IS the "
              "unscaled-complement control (G_COMPREF)",
    "injection": "theta_base + delta fp32 (e306/e310 checkpoint "
                 "conventions: {'model', 'delta', provenance}); VERIFIED "
                 "from disk (bit-equal roundtrip + model == base+delta "
                 "re-derived + probe-from-disk)",
    "read": "e261's g0 battery bit-exactly (corpus 1337; find_occ + "
            "SPLICE_RNG 24301 shuffle; first-60 install rows; 130-char "
            "pre-name contexts), mean p(Z) at the last position, CPU "
            "TinyGPT(Cfg()) (the 2.74M organism), bs 30; gm12 co-reported; "
            "mean top-1 p + mean entropy honesty-riders (organism-health "
            "co-reports, never bars)",
    "control_bands": "G_FULLREAD |d| <= 5e-3 vs 0.26464763283729553 "
                     "(e310's own bar); G_COMPREF in [0.5x, 2x] of "
                     "3.655568798421882e-05; either failure => TEXTURE",
    "adjudication": "g := full-dose complement g0 mean_pz; "
                    "DEAD-AT-FULL-DOSE iff g < 0.01; ALIVE iff g >= 0.05; "
                    "else GAP (frozen reading above); P-x15a + P-x15a-exec "
                    "scored verbatim",
}
E290_WRITE_NORM = 9.1788432658723            # == ||dW|| (e290/e310 committed)
E264_K10K_POST_G0 = 0.26464763283729553      # the committed full-write t0 read
E310_COMP_G0 = 3.655568798421882e-05         # e310's committed complement read
E310_COMP_L2_64 = 3.024341960836486          # e310's committed fp64 comp L2
E310_BASE_G0 = 1.3383959412749391e-05        # e310's committed base t0 read
E310_MIX_FULL_G0 = 0.06310301274061203       # e310's mix 50/50 full-dose
E310_MIX_COMPDOSE_G0 = 0.00012323708506301045  # e310's mix comp-dose (dead)
DEAD_BAR = 0.01                              # frozen: DEAD-AT-FULL-DOSE
ALIVE_BAR = 0.05                             # frozen: ALIVE
LADDER_MULTS = [1.0, 1.5, 2.25]              # + s_full appended at runtime


def log(msg: str) -> None:
    print(f"[x15 {time.time() - T0:7.1f}s] {msg}", flush=True)


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


# ------------------------------------------------ G_ENDPOINTS
ROOT_CK = CKPT / "e001.pt"
K10K_CK = CKPT / "e261_K10K_inst_resume.pt"
ROOMS_CK = CKPT / "e264_rooms.pt"
E310_COMP_CK = REPO / "runs" / "e310" / "e310_complement.pt"
E310_METRICS = REPO / "runs" / "e310" / "metrics.json"
E306_METRICS = REPO / "runs" / "e306" / "metrics.json"

BINDS = {
    "e001": (ROOT_CK, "d114536d1c0983ab3be67f67ff0667c8",
             "x6's committed root bind (== e267/e310's base bind)"),
    "e261_K10K_inst_resume": (K10K_CK, "0f6dc1cf46850ce655dfafc9c853d467",
                              "x6's bind (== e264 G_K10KRESUME, the s400 "
                              "final; e310's bind)"),
    "e264_rooms": (ROOMS_CK, "2d524655575cce00a3bc1c8770f4b211",
                   "the committed K10K room store (e310's own load)"),
    "e310_complement_pt": (E310_COMP_CK, "bbddb3eac2b652280f38babe92a5467f",
                           "e310's committed on-disk complement delta — "
                           "G_REGEN's bit-exactness target"),
    "e310_metrics": (E310_METRICS, "713fd5c01f1b946ed7148b41a837dc58",
                     "e310's committed metrics — the reference constants"),
    "e306_metrics": (E306_METRICS, "a4f7a55ae15f6db0f809352aa182d6ff",
                     "e306's committed ladder (figure context)"),
}
md5_reads = {}
for nm, (path, claim, src) in BINDS.items():
    got = md5_of(path)
    md5_reads[nm] = {"path": str(path.relative_to(REPO)), "md5": got,
                     "claimed": claim, "source": src, "match": got == claim}
    if got != claim:
        raise SystemExit(f"ENDPOINT BIND FAILURE: {nm} {got} != {claim}")
METRICS["gates"] = {"G_ENDPOINTS": {"md5_binds": md5_reads, "pass": True}}

# live cross-check: e310's metrics carry exactly the frozen literals
_e310m = json.loads(E310_METRICS.read_text(encoding="utf-8"))
_lit_checks = {
    "complement_l2": abs(_e310m["complement"]["l2"] - E310_COMP_L2_64) < 1e-15,
    "complement_g0": abs(_e310m["arms"]["complement"]["t0_read"]["g0"]
                         ["mean_pz"] - E310_COMP_G0) < 1e-18,
    "full_g0": abs(_e310m["anchors"]["full_write_t0"]["g0"]["mean_pz"]
                   - E264_K10K_POST_G0) < 1e-15,
    "dW_l2": abs(_e310m["object"]["dW_l2"] - E290_WRITE_NORM) < 1e-12,
    "base_g0": abs(_e310m["anchors"]["base_e001_t0"]["g0"]["mean_pz"]
                   - E310_BASE_G0) < 1e-18,
    "mix_full_g0": abs(_e310m["arms"]["mix5050_full"]["t0_read"]["g0"]
                       ["mean_pz"] - E310_MIX_FULL_G0) < 1e-15,
    "mix_compdose_g0": abs(_e310m["arms"]["mix5050_compdose"]["t0_read"]
                           ["g0"]["mean_pz"] - E310_MIX_COMPDOSE_G0) < 1e-15,
}
METRICS["gates"]["G_ENDPOINTS"]["e310_literal_crosscheck"] = _lit_checks
if not all(_lit_checks.values()):
    raise SystemExit(f"LITERAL CROSSCHECK FAILURE: {_lit_checks}")


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


# common.py via importlib (x6/e310's convention: TinyGPT/Cfg/CharCorpus only;
# no GPU code path executes — DEVICE resolves 'cpu' under CUDA_VISIBLE_DEVICES='')
_spec = importlib.util.spec_from_file_location("x15_common",
                                               REPO / "lab" / "common.py")
_common = importlib.util.module_from_spec(_spec)
sys.modules["x15_common"] = _common
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
write_partial("P0 bars+P-x15a frozen in-script; 6 parent md5 binds PASS; "
              "flat basis PASS")


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
    """G1.battery_cell's exact form (e310's CPU probe) + honesty riders
    (mean top-1 p, mean entropy at the last position) — riders are
    co-reports, never bars."""
    net.eval()
    pzs, tops, ents, amax = [], [], [], 0.0
    for i in range(0, ids.shape[0], bs):
        lg, _ = net(ids[i:i + bs])
        pr = F.softmax(lg[:, -1], -1)
        pzs.append(pr[:, zid])
        tops.append(pr.max(-1).values)
        ents.append(-(pr * torch.log(pr.clamp_min(1e-30))).sum(-1))
        amax += float((lg[:, -1].argmax(-1) == zid).float().sum())
    p = torch.cat(pzs)
    return {"mean_pz": float(p.mean()), "median_pz": float(p.median()),
            "frac_pz_ge_0.5": float((p >= 0.5).float().mean()),
            "frac_argmax_z": amax / ids.shape[0],
            "rider_mean_top1_p": float(torch.cat(tops).mean()),
            "rider_mean_entropy": float(torch.cat(ents).mean())}


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
    """Ported BY VALUE from x6 via e310: the exact orthogonal projector
    P x = D . idct(mask_S(dct(D . x)))."""

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


# ------------------------------------------- THE OBJECT + complement + G_ORTH
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
comp32 = {k: comp64[k].float() for k in KEYS}   # injection currency
COMP_L2 = float(np.linalg.norm(comp_flat))


def flat_of(d32: dict) -> np.ndarray:
    return torch.cat([d32[k].double().reshape(-1)
                      for k in KEYS]).numpy()


def in_room_share(flat: np.ndarray) -> float:
    p = room.project(flat)
    return float((p @ p) / max(flat @ flat, 1e-30))


# ------------------------------------------------ G_REGEN (bit-exact vs e310)
_e310_delta = torch.load(E310_COMP_CK, map_location="cpu",
                         weights_only=False)["delta"]
bit_equal = {k: bool(torch.equal(_e310_delta[k].float(), comp32[k]))
             for k in KEYS}
GT_REGEN = {
    "form": "the fp32-cast regenerated complement vs e310's committed "
            "on-disk delta (runs/e310/e310_complement.pt['delta'], "
            "md5-bound) — bit-equal on EVERY key; + fp64 L2 vs e310's "
            "committed 3.024341960836486 (1e-9)",
    "keys_bit_equal": int(sum(bit_equal.values())),
    "keys_total": len(KEYS),
    "all_keys_bit_equal": all(bit_equal.values()),
    "my_comp64_l2": COMP_L2,
    "e310_committed_l2": E310_COMP_L2_64,
    "l2_abs_diff": abs(COMP_L2 - E310_COMP_L2_64),
    "l2_bar": 1e-9,
    "pass": bool(all(bit_equal.values())
                 and abs(COMP_L2 - E310_COMP_L2_64) < 1e-9),
}
if not GT_REGEN["pass"]:
    raise SystemExit(f"REGEN GATE FAILURE: {GT_REGEN}")
METRICS["gates"]["G_REGEN"] = GT_REGEN
METRICS["complement_regenerated"] = {
    "form": OP["complement"],
    "l2_fp64": COMP_L2,
    "mass_frac_of_full": COMP_L2 / DW_L2,
    "energy_frac_of_full": F_OUT,
    "bit_exact_vs_e310_checkpoint": True,
    "in_room_share_fp64_measured": in_room_share(comp_flat),
    "in_room_share_of_injected_fp32": in_room_share(flat_of(comp32)),
}
log(f"complement regenerated BIT-EXACT vs e310's checkpoint "
    f"({GT_REGEN['keys_bit_equal']}/{GT_REGEN['keys_total']} keys); "
    f"L2 {COMP_L2:.6f} (committed {E310_COMP_L2_64:.6f})")
write_partial("P1 complement regenerated bit-exact (G_ORTH + G_REGEN PASS)")

# ------------------------------------------------ G_FULLREAD (control #1)
base_read = probe_delta(None)            # base-only context anchor
full32 = {k: dW64[k].float() for k in KEYS}
full_probe = probe_delta(full32)
d_full = abs(full_probe["g0"]["mean_pz"] - E264_K10K_POST_G0)
GT_FULL = {
    "form": "this desk's probe of theta_base + dW (the full write) vs "
            "e264/e310's committed K10K post g0 0.26464763283729553",
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
    "e310_committed_base_g0": E310_BASE_G0,
}
write_partial(f"P2 G_FULLREAD {'PASS' if GT_FULL['pass'] else 'FAIL'} "
              f"(|d|={d_full:.2e})")
if not GT_FULL["pass"]:
    METRICS["verdict"] = {"word": "TEXTURE",
                          "why": "G_FULLREAD failure — the instrument does "
                                 "not reproduce the committed full read; "
                                 "nothing adjudicated"}
    write_partial("HALT: TEXTURE (G_FULLREAD)")
    raise SystemExit("G_FULLREAD FAILURE — nothing adjudicated")


# --------------------------------- injection machinery (e310's exact method)
def inject_and_probe(name: str, delta32: dict, delta_kind: str,
                     extra: dict | None = None) -> dict:
    """e306/e310's checkpoint conventions: save theta_base+delta fp32 + the
    per-key delta; verify from disk (bit-equal roundtrip + model ==
    base+delta re-derived); probe FROM DISK (the roundtrip is part of the
    datum)."""
    ck_path = OUT / f"x15_{name}.pt"
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
        "room": "runs/checkpoints/e264_rooms.pt K10K (md5 "
                f"{md5_reads['e264_rooms']['md5']})",
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
        "delta_l2_fp32": float(np.linalg.norm(fl)),
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


# --------------------------------- the ladder arms (comp_x1.000 = G_COMPREF)
s_full = DW_L2 / COMP_L2                 # ~3.0350, derived not trusted
scaled_full64 = s_full * comp_flat
scaled_l2 = float(np.linalg.norm(scaled_full64))
scaled_inroom64 = in_room_share(scaled_full64)

arms: dict[str, dict] = {}
ladder = []
for mult in LADDER_MULTS + [s_full]:
    tag = f"comp_x{mult:.6g}" if mult != s_full else "comp_xFULL"
    if SMOKE and mult != 1.0:
        continue
    if mult == 1.0:
        d32 = comp32
    else:
        d64 = unflat(mult * comp_flat)
        d32 = {k: d64[k].float() for k in KEYS}
    arm = inject_and_probe(
        tag, d32,
        f"(I - P_room) dW scaled x{mult:.6g} (dose-matched to the full "
        f"write's norm at x{s_full:.6g})" if mult != 1.0 else
        "(I - P_room) dW (the complementary write, UNSCALED — e310's arm "
        "re-verified in-session)",
        extra={"scale_multiplier": float(mult),
               "fp64_intended_l2": float(mult * COMP_L2),
               "fp64_intended_l2_vs_full_abs": abs(float(mult * COMP_L2)
                                                   - DW_L2),
               "fp32_l2_vs_full_l2_rel_dev": abs(
                   float(np.linalg.norm(flat_of(d32))) - DW_L2) / DW_L2})
    arms[tag] = arm
    ladder.append(tag)
    log(f"{tag}: t0 g0 {arm['t0_read']['g0']['mean_pz']:.6g} "
        f"(L2 {arm['delta_l2_fp32']:.4f}; in-room "
        f"{arm['in_room_share_of_injected_fp32']:.2e}; riders top1 "
        f"{arm['t0_read']['g0']['rider_mean_top1_p']:.4f} / H "
        f"{arm['t0_read']['g0']['rider_mean_entropy']:.4f})")
    write_partial(f"P3 arm {tag} injected + disk-verified + probed")

# ------------------------------------------------ G_COMPREF (control #2)
comp1_g0 = arms["comp_x1"]["t0_read"]["g0"]["mean_pz"]
GT_COMPREF = {
    "form": "the unscaled complement re-probed THIS session vs e310's "
            "committed 3.655568798421882e-05 (frozen band [0.5x, 2x])",
    "mine_g0_mean_pz": comp1_g0,
    "committed": E310_COMP_G0,
    "band_lo": 0.5 * E310_COMP_G0,
    "band_hi": 2.0 * E310_COMP_G0,
    "pass": bool(0.5 * E310_COMP_G0 <= comp1_g0 <= 2.0 * E310_COMP_G0),
}
METRICS["gates"]["G_COMPREF"] = GT_COMPREF
write_partial(f"P4 G_COMPREF {'PASS' if GT_COMPREF['pass'] else 'FAIL'} "
              f"({comp1_g0:.4e})")
if not GT_COMPREF["pass"]:
    METRICS["verdict"] = {"word": "TEXTURE",
                          "why": "G_COMPREF failure — the unscaled "
                                 "complement does not reproduce its "
                                 "committed read; nothing adjudicated"}
    write_partial("HALT: TEXTURE (G_COMPREF)")
    raise SystemExit("G_COMPREF FAILURE — nothing adjudicated")

# ------------------------------------------------ G_DOSEMATCH + G_INROOM_SCALED
GT_DOSEMATCH = {
    "form": "||s_full * comp64|| == ||dW|| to 1e-12, s_full = ||dW||/"
            "||comp64|| derived from the regenerated fp64 vectors",
    "s_full": s_full,
    "letter_claim": "~3.033 (WRONG by ~0.1% — derived here, not trusted)",
    "scaled_fp64_l2": scaled_l2,
    "dW_l2": DW_L2,
    "abs_diff": abs(scaled_l2 - DW_L2),
    "bar": 1e-12,
    "pass": bool(abs(scaled_l2 - DW_L2) <= 1e-12),
    "disclosure": "the fp32 injection currency deviates ~1e-7 relative "
                  "(same order as the full write's own fp32 cast, which "
                  "reproduced the committed read bit-exactly in e310); "
                  "reported per arm, never gated",
}
GT_INROOM = {
    "form": "scaling cannot change direction — verified anyway: in-room "
            "energy share of every scaled arm (fp64 intended AND fp32 "
            "injected) <= 1e-12",
    "scaled_fp64_in_room_share": scaled_inroom64,
    "arm_fp32_in_room_shares": {t: arms[t]["in_room_share_of_injected_fp32"]
                                for t in arms},
    "bar": 1e-12,
    "pass": bool(scaled_inroom64 <= 1e-12
                 and all(arms[t]["in_room_share_of_injected_fp32"] <= 1e-12
                         for t in arms)),
}
if not GT_DOSEMATCH["pass"]:
    raise SystemExit(f"DOSEMATCH GATE FAILURE: {GT_DOSEMATCH}")
if not GT_INROOM["pass"]:
    raise SystemExit(f"INROOM-SCALED GATE FAILURE: {GT_INROOM}")
METRICS["gates"]["G_DOSEMATCH"] = GT_DOSEMATCH
METRICS["gates"]["G_INROOM_SCALED"] = GT_INROOM
METRICS["dose_match"] = {
    "s_full": s_full,
    "scaled_fp64_l2": scaled_l2,
    "full_write_l2": DW_L2,
    "complement_energy_frac_at_full_dose": (s_full ** 2) * F_OUT,
    "note": "at xFULL the complement carries 100.00% of the full write's "
            "energy, ALL of it out-of-room (9.21x the full write's own "
            "out-of-room energy — the honesty riders watch for organism "
            "collapse)",
}
if SMOKE:
    METRICS["status"] = "SMOKED — gates + both controls + comp_x1.000 arm " \
                        "exercised; scaled arms/figure/report skipped"
    write_partial("SMOKE COMPLETE — nothing adjudicated")
    log("SMOKE COMPLETE (nothing adjudicated)")
    raise SystemExit(0)

# ------------------------------------------------ adjudication (frozen)
arm_full = arms["comp_xFULL"]
g_full_dose = arm_full["t0_read"]["g0"]["mean_pz"]
if g_full_dose < DEAD_BAR:
    verdict = "DEAD-AT-FULL-DOSE"
    clause = (f"the full-dose complement (x{s_full:.4f}, norm-matched to "
              f"the full write to 1e-12) reads {g_full_dose:.6g} < 0.01 — "
              "THE BEARER LAW SURVIVES DOSE-MATCHED: structure without "
              "room-overlap cannot read at any dose; the read needs the "
              "room's tail to couple into the context")
elif g_full_dose >= ALIVE_BAR:
    verdict = "ALIVE"
    clause = (f"the full-dose complement reads {g_full_dose:.6g} >= 0.05 — "
              "THE NECESSITY CLAIM FALLS as a dose artifact: out-of-room "
              "structure CAN read when dense enough; the bearer claim "
              "reduces to 'dose and geometry both matter' (Law-of-3-era "
              "wording must be re-scoped)")
else:
    verdict = "GAP-[0.01,0.05)"
    clause = (f"the full-dose complement reads {g_full_dose:.6g} in "
              "[0.01, 0.05) — neither frozen bar fires: dead by e310's "
              "original 0.05 floor but NOT by this stricter frozen bar; "
              "the necessity claim survives WEAKENED; no wording change "
              "without a new registered cell")
p_score = {
    "P-x15a_registered": P_X15A["registered_guess"],
    "P-x15a_outcome": ("HIT — DEAD-AT-FULL-DOSE"
                       if verdict == "DEAD-AT-FULL-DOSE" else
                       ("MISS — ALIVE fired" if verdict == "ALIVE" else
                        "PARTIAL — the GAP reading (dead by e310's floor, "
                        "not by the frozen 0.01 bar)")),
    "P-x15a_exec_subprediction": P_X15a["executor_position"],
    "P-x15a_exec_outcome": ("HIT — g < 1e-3" if g_full_dose < 1e-3 else
                            f"MISS — g = {g_full_dose:.6g} >= 1e-3"),
}
METRICS["verdict"] = {
    "word": verdict,
    "clause": clause,
    "full_dose_complement_t0_g0": g_full_dose,
    "dead_bar": DEAD_BAR,
    "alive_bar": ALIVE_BAR,
    "bars_verbatim": BARS,
    "operationalizations": OP,
    "prediction": {**P_X15A, **p_score},
}
write_partial(f"P5 ADJUDICATED: {verdict} (g={g_full_dose:.6g})")

# ------------------------------------------------ figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

ax = axes[0]
mults = [arms[t]["scale_multiplier"] for t in ladder]
g0s = [arms[t]["t0_read"]["g0"]["mean_pz"] for t in ladder]
ax.plot(mults, g0s, "o-", color="steelblue", lw=1.6, ms=8, zorder=3,
        label="THE DOSE LADDER (this cell)")
for m, g, t in zip(mults, g0s, ladder):
    ax.annotate(f"{g:.3g}", (m, g), textcoords="offset points",
                xytext=(6, 7), fontsize=8, color="steelblue")
ax.axhline(E264_K10K_POST_G0, color="k", ls="-", lw=1.2,
           label=f"full write committed read {E264_K10K_POST_G0:.4f}")
ax.axhline(ALIVE_BAR, color="crimson", ls="--", lw=1.2,
           label=f"ALIVE bar {ALIVE_BAR}")
ax.axhline(DEAD_BAR, color="darkgreen", ls="--", lw=1.2,
           label=f"DEAD bar {DEAD_BAR}")
ax.axhline(E310_COMP_G0, color="steelblue", ls=":", lw=1.4,
           label=f"e310 complement committed {E310_COMP_G0:.3e} (@1x)")
ax.axhline(E310_BASE_G0, color="gray", ls=":", lw=1.0,
           label=f"e001 base read {E310_BASE_G0:.2e}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("dose = scale multiplier on the complement (log); "
              f"xFULL = {s_full:.4f}")
ax.set_ylabel("t0 g0 read (log)")
ax.set_title(f"(a) THE DOSE-MATCHED COMPLEMENT — verdict {verdict} "
             f"(full-dose g0 = {g_full_dose:.3g})")
ax.legend(fontsize=7.5, loc="center left")
ax.grid(alpha=0.25, which="both")

ax = axes[1]
e306_m = json.loads(E306_METRICS.read_text(encoding="utf-8"))
pts = [
    (e306_m["ladder"]["r1000"]["room"]["inside_share_of_own"],
     e306_m["ladder"]["r1000"]["t0_read"]["g0"]["mean_pz"],
     "e306 r1000 = FULL write (89% in-room)", "k", "o"),
    (e306_m["ladder"]["r100"]["room"]["inside_share_of_own"],
     e306_m["ladder"]["r100"]["t0_read"]["g0"]["mean_pz"],
     "e306 r100 (67% in-room)", "gray", "o"),
    (0.49999999994334265, E310_MIX_FULL_G0,
     "e310 mix 50/50 FULL dose (50% in-room)", "crimson", "s"),
    (arms["comp_xFULL"]["in_room_share_of_injected_fp32"], g_full_dose,
     f"THIS CELL comp xFULL (0% in-room, 100% dose)", "steelblue", "D"),
    (arms["comp_x2.25"]["in_room_share_of_injected_fp32"],
     arms["comp_x2.25"]["t0_read"]["g0"]["mean_pz"],
     "THIS CELL comp x2.25 (55% dose)", "steelblue", "v"),
    (arms["comp_x1.5"]["in_room_share_of_injected_fp32"],
     arms["comp_x1.5"]["t0_read"]["g0"]["mean_pz"],
     "THIS CELL comp x1.5 (24% dose)", "steelblue", "^"),
    (arms["comp_x1"]["in_room_share_of_injected_fp32"], comp1_g0,
     "THIS CELL comp x1.0 = e310's arm (11% dose)", "steelblue", "o"),
]
for x, y, lb, c, m in pts:
    ax.scatter([max(x, 1e-18)], [max(y, 1e-7)], color=c, marker=m, s=55,
               zorder=3, label=lb)
ax.axhline(ALIVE_BAR, color="crimson", ls="--", lw=1.0,
           label=f"ALIVE bar {ALIVE_BAR}")
ax.axhline(DEAD_BAR, color="darkgreen", ls="--", lw=1.0,
           label=f"DEAD bar {DEAD_BAR}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("in-room energy share of the injected delta (log)")
ax.set_ylabel("t0 g0 read (log)")
ax.set_title("(b) THE BEARER CURVE — dose axis separated from geometry "
             "axis (this cell fills the 0%-in-room column at 4 doses)")
ax.legend(fontsize=7, loc="lower right")
ax.grid(alpha=0.25, which="both")

fig.suptitle("x15 — THE DOSE-MATCHED COMPLEMENT: e310's out-of-room "
             "complement scaled to the full write's norm (R71 critic iii; "
             f"verdict {verdict})", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT / "x15_dose_ladder.png", dpi=140)
log("FIGURE written")

# ------------------------------------------------ REPORT.md
def row(t, arm, role=""):
    g = arm["t0_read"]["g0"]["mean_pz"]
    return (f"| {t} | x{arm['scale_multiplier']:.4g} | "
            f"{arm['fp64_intended_l2']:.4f} | {arm['delta_l2_fp32']:.4f} | "
            f"{100 * arm['energy_frac_of_full']:.2f}% | "
            f"{arm['in_room_share_of_injected_fp32']:.1e} | {g:.6g} | "
            f"{'ALIVE' if g >= ALIVE_BAR else ('DEAD' if g < DEAD_BAR else 'GAP')}"
            f" | {arm['t0_read']['g0']['rider_mean_top1_p']:.4f} | "
            f"{arm['t0_read']['g0']['rider_mean_entropy']:.4f} |{role}")


tbl = [
    "| arm | scale | fp64 L2 (intended) | fp32 L2 (injected) | energy vs full | in-room share | t0 g0 | read | rider top-1 p | rider entropy | role |",
    "|---|---|---|---|---|---|---|---|---|---|---|",
]
for t in ladder:
    tbl.append(row(t, arms[t],
                   " (THE PRIMARY ARM — dose-matched)" if t == "comp_xFULL"
                   else (" (G_COMPREF control)" if t == "comp_x1" else "")))
rider_gap = abs(arm_full["t0_read"]["g0"]["rider_mean_top1_p"]
                - full_probe["g0"]["rider_mean_top1_p"])
rider_verdict = (
    "(the confidence structure HELD — the dead read is a bearer-law fact, "
    "not organism collapse)" if rider_gap < 0.2 else
    "(WARNING: the riders MOVED — the dead read may be trivially "
    "guaranteed by organism damage; stamped so)")

report = f"""# x15 — THE DOSE-MATCHED COMPLEMENT ({verdict})

**The confound (R71 critic iii):** e310's complement (the full 10k write
MINUS its room-overlap components — 0% in-room by construction) read DEAD
(3.66e-05) at its natural 10.86%-energy dose, but everything inside a ~9x
energy gap dies (e306's r10 at 14.7% energy too) — the bearer-necessity
claim was dose-confounded until a complement at the FULL write's dose was
probed. **This cell is that probe:** the complement regenerated BIT-EXACT
from e310's own construction (G_REGEN: {GT_REGEN['keys_bit_equal']}/{GT_REGEN['keys_total']}
keys bit-equal to e310's committed checkpoint; fp64 L2 {COMP_L2:.6f} vs
committed {E310_COMP_L2_64:.6f}), scaled x{s_full:.6g} so its fp64 norm
equals the full write's 9.1788432658723 to
{abs(scaled_l2 - DW_L2):.1e} (G_DOSEMATCH; the dispatch letter's ~x3.033
was off by ~0.1% — derived here, not trusted), injected by e310's exact
fp32 method, probed at t0 on e261's 60-window battery.

## Controls (both re-verified in-session BEFORE any arm counted)

- FULL write: {full_probe['g0']['mean_pz']:.17g} vs committed
  {E264_K10K_POST_G0:.17g} (|d| = {d_full:.1e}; G_FULLREAD PASS).
- UNSCALED complement: {comp1_g0:.6e} vs committed {E310_COMP_G0:.6e}
  (band [0.5x, 2x]; G_COMPREF PASS).
- Base e001 root: {base_read['g0']['mean_pz']:.3e} (e310 committed
  {E310_BASE_G0:.3e}).
- Gates: {sum(1 for g in METRICS['gates'].values() if g['pass'])}/
  {len(METRICS['gates'])} PASS (G_ENDPOINTS md5-binds 6 parents,
  G_FLATBASIS, G_BATTERY, G_ROOM, G_ORTH, G_REGEN, G_FULLREAD,
  G_COMPREF, G_DOSEMATCH, G_INROOM_SCALED).

## THE DOSE LADDER

{chr(10).join(tbl)}

## Verdict: {verdict}

{clause}.

**P-x15a scoring (registered pre-compute, never shopped):**
{p_score['P-x15a_registered']} -> **{p_score['P-x15a_outcome']}**.
Executor sub-prediction: {p_score['P-x15a_exec_subprediction']} ->
**{p_score['P-x15a_exec_outcome']}**.

**The honest riders (organism health, never bars):** at xFULL the
complement carries {100 * (s_full ** 2) * F_OUT:.2f}% of the full write's
energy ALL out-of-room ({1 / F_OUT:.2f}x the full write's own out-of-room
energy). Full-write control riders: top-1 p
{full_probe['g0']['rider_mean_top1_p']:.4f} / entropy
{full_probe['g0']['rider_mean_entropy']:.4f}; xFULL complement riders:
top-1 p {arm_full['t0_read']['g0']['rider_mean_top1_p']:.4f} / entropy
{arm_full['t0_read']['g0']['rider_mean_entropy']:.4f}
{rider_verdict}.

Disclosures: the complement is orthogonal to the room BY CONSTRUCTION
(G_ORTH identity at rel {METRICS['gates']['G_ORTH']['energy_sum_rel_resid']:.1e});
the fp32 injection layer deviates from the fp64 dose-match at ~1e-7
relative (per arm in metrics.json; the full write's own fp32 cast in e310
reproduced the committed read bit-exactly, so fp32 IS the committed
instrument); scaling cannot change direction but every arm's in-room share
was re-measured anyway (max {max(GT_INROOM['arm_fp32_in_room_shares'].values()):.1e},
G_INROOM_SCALED PASS); the letter's "0% at 33% dose" mix arm is the
comp-dose mix whose ENERGY share is 10.86% (its 32.95% is the complement's
MASS share — runs/e310/metrics.json); n=1 per arm (one lineage, one
session — the g-series standing lottery note carried verbatim).
"""
(OUT / "REPORT.md").write_text(report, encoding="utf-8")

METRICS["arms"] = arms
METRICS["ladder_order"] = ladder
METRICS["outputs"] = {
    "figure": "runs/x15/x15_dose_ladder.png",
    "metrics": "runs/x15/metrics.json",
    "report": "runs/x15/REPORT.md",
    "checkpoints": [arms[t]["checkpoint"]["path"] for t in ladder],
}
METRICS["builds_on"] = [
    "e310 (THE rig, the complement construction, the committed complement "
    "checkpoint + reference reads + mix arms — regenerated bit-exact here)",
    "R71 (critic land iii: the bearer-necessity dose confound — the cell "
    "this design discharges)",
    "e306 (the truncation ladder + the committed forward ladder, read "
    "live for the figure)",
    "x6/e261/e264 (THE registered object, the SRCT projector, the md5 "
    "binds, the g0 battery conventions, the committed post g0)",
    "T278 (P-x15a's author: the dose-artifact fork named at e311's null)",
]
METRICS["whats_new"] = [
    "THE FIRST DOSE-MATCHED ALIVE/DEAD PAIR on the room-geometry axis: "
    "the full write (89% in-room, norm 9.1788) vs the complement scaled to "
    "THE SAME NORM (0% in-room) — dose finally held fixed while geometry "
    "varies",
    "THE 4-POINT DOSE LADDER on the out-of-room complement (1x / 1.5x / "
    "2.25x / FULL) — the curve inside the ~9x energy gap R71's critic "
    "named, previously probed only at its floor",
    "G_REGEN: the strongest parent bind in the series — the regenerated "
    "complement proven BIT-EQUAL to e310's committed on-disk delta",
]
METRICS["status"] = "COMPLETE — adjudicated"
write_partial("P6 COMPLETE (report + figure)")
log(f"DONE — verdict {verdict} (full-dose g0 {g_full_dose:.6g})")
