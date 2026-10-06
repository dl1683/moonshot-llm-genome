"""X14 — THE TRANSPORT INTERVENTION — R67's named desk cell (dispatched in
the R67 fold, 2026-10-05/06): converting e283's correlational dichotomy
into an INTERVENTIONAL one. This docstring carries the registered
question + bars VERBATIM + the convention freezes, committed at birth
BEFORE any compute. Adjudicate against exactly this; no bar shopping.

THE QUESTION: e283 (ESTABLISHED-DIES, runs/e283/metrics.json; T261) found
the established 10k write dies 29,252x under 400 corpus steps — the state
drifting 14.45 from the loaded fact (only 11.4% in-room) while the
write's own standing displacement is 9.18 — read as "TRANSPORT (carried
away)". But R67's critic showed the committed reads CANNOT separate
TRANSPORT-LITERAL (the write's in-room mass intact, killed by out-of-room
context) from OVERWRITE (the write's in-room mass eroded): the finding-3
arithmetic — the intact-write band [m0 - |dR|, m0 + |dR|] = [7.0, 10.3]
(measured 7.31 sits inside it, near the opposed edge) — and the erosion
degeneracy (the same 7.31 is consistent with ANY surviving in-room
fraction f in [0.65, 1.03], the drift's in-room 1.65 absorbed by
alignment ambiguity). Both hypotheses fit every committed number. The
states are ON DISK (e283 checkpointed its milestone posts). This cell
INTERVENES: subtract a projection component off the final state and read
the probe.

THE ARMS (verbatim intent from the dispatch letter):
  (a) THE WRITE-MASS LEDGER: for each milestone state (t100/200/300/400
      where checkpointed — DISCLOSED: only t0 (the fact) and t400 (the
      post state) were saved by e283; the t100-300 grid is cited-derived
      from e283's committed fp64 ledger products): the write's surviving
      in-room mass ||P_room(theta_t - theta_base)|| and the out-of-room
      residual ||(I-P)(theta_t - theta_base)||.
  (b) THE SUBTRACTION INTERVENTION: theta_test = theta_t400 -
      P_orth(drift), drift = theta_t400 - theta_t0_established (t0 =
      the loaded fact's state), P_orth = (I - P_room): subtract the
      out-of-room displacement component from the final state, then read
      g0 (the probe). Plus the converse control arm (subtract the IN-ROOM
      component instead).

FROZEN BARS (verbatim from the dispatch letter):
  - TRANSPORT-LITERAL: the orthogonal-subtraction read >= 0.10 (a
    substantial fraction of baseline) — the write survives its context's
    removal.
  - OVERWRITTEN-LEANING: the read stays < 0.01 AND the write-mass ledger
    shows the in-room mass eroded below 50% of its t0 value.
  - MIXED/INCONCLUSIVE: everything else — including the coupling caveat
    firing (mass intact + no resurrection).

REGISTERED PREDICTIONS (CITED from the dispatch, not re-registered):
TRANSPORT-LITERAL predicts the orthogonal-subtraction read RESURRECTS
toward the loaded baseline 0.2646 (>= 0.10 by the frozen bar);
OVERWRITTEN predicts it stays dead (< 0.01). The converse arm (in-room
subtraction) inverts the predictions: under transport-literal it stays
dead (the killer out-of-room context is KEPT); under overwrite it is the
arm that removes the erosive channel. The converse arm is the CONTROL —
reported, co-interpreted, NOT adjudicated (no frozen bar on it).

THE HONESTY BLOCK (the dispatch's (c), verbatim in force): the
subtraction assumes component-wise additivity (a linear superposition
null); any coupling (LN, softmax) can mask resurrection — the
intervention is a NECESSARY-not-SUFFICIENT test of transport-literal
(resurrection => transport; no resurrection => inconclusive-between-
overwrite-and-coupling, NOT proof of overwrite). The MIXED/INCONCLUSIVE
bar exists exactly to hold this: mass intact + no resurrection lands
there BY REGISTRATION, not by consolation.

CONVENTION FREEZES (frozen HERE before compute):
  * THE ROOM := the fact's own committed K10K room (k=10,000, seeds
    26113/26114 — e264_rooms.pt's D/S, BIT-BOUND by G_ROOMK10K exact
    equality; the SRCT projector applied exact in fp64 pocketfft CPU,
    ported by import from the committed lab/e261_rank_ladder.py, which
    is NOT modified).
  * THE STATES := t0 = e261_K10K_inst_resume.pt (the loaded fact, md5/
    size/step/flat-md5/behavior-bound — e283's G_FACTLOAD binds, re-run
    here) and t400 = e283_ESTABLISHED-CONCURRENT_post.pt (the
    established-concurrent post state; e283 recorded no file md5 for it
    — bound here BEHAVIORALLY (the g0/gm12/CE_R reads vs e283's
    committed literals) AND GEOMETRICALLY (the fp64 drift ledger norms/
    fracs vs e283's committed disp_ledger), the stronger value-bind;
    its file md5 is recorded for the future).
  * THE BASE := runs/checkpoints/e001.pt (the 2.74M corpus base,
    fact-free-gated).
  * THE PROBE := the family's g0 battery (60 install-splice windows,
    p(Z) at the last position; the splice/battery convention rebuilt and
    gated exactly as in e283 — G_SPLICE 19+41, G_BATTERY shapes).
  * THE READS := g0 (PRIMARY), gm12, gp12, CE_R — CPU fp32, threads 4.
  * THE SUBTRACTION ARITHMETIC := fp64 throughout: delta = theta400 -
    theta0; P_delta = P_room(delta); delta_O = delta - P_delta;
    arm A (ORTHOGONAL SUBTRACTION, primary): theta_A = theta400 -
    delta_O (== theta0 + P_delta — the corpus's out-of-room transport
    REMOVED, everything else kept); arm B (IN-ROOM SUBTRACTION, the
    converse control): theta_B = theta400 - P_delta (== theta0 +
    delta_O — the in-room channel removed, the transport kept); arm ID
    (the identity check): theta400 - delta == theta0 (must return the
    fact's own read — the construction's closure). Arms materialized
    fp32 (the model's dtype); the fp64->fp32 quantization residual
    measured and disclosed.
  * NO TRAINING, NO STREAM, NO STEPS: this is a desk cell — reads and
    projections only; nothing trains, nothing moves.

HARD GATES := {G_NAMEFREE, G_SPLICE, G_BATTERY, G_PARENTS, G_BASE,
G_ROOT, G_VMBIND, G_SPANBIND, G_PROJ, G_ROOMK10K, G_FACTLOAD,
G_POSTLOAD} — a failure HALTS (nothing adjudicated).

COMPUTE ENVELOPE: CPU-ONLY desk cell — torch threads 4, pocketfft
workers 2, NO GPU ops (the e261 import carries the family's
cuda-availability assert; no CUDA tensor is ever created), NO
envelope-log writes, no bursts (nothing trains); timestamps
datetime.now(UTC) only (common.now_iso).

Outputs: runs/x14/{metrics.json (PROGRESSIVE), x14_transport.png,
REPORT.md, run.log (gitignored)}. No NOTES/THINKING/QUEUE/STATE edits
(dispatch; the coordinator folds). Birth commit BEFORE compute; final
commit AND push.

Run:  cd lab && python x14_transport_intervention.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")       # e228's offline convention
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:                                            # Windows console safety
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                               # noqa: BLE001
    pass

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

import common                                          # noqa: E402
from common import CharCorpus, run_dir, save_json      # noqa: E402

import e043_install as E43                             # noqa: E402 (REPO,
                                                      # find_occ, SPLICE_RNG,
                                                      # jsonable)
import g1b_continuity as GB                            # noqa: E402 — MUST be
                                                      # imported BEFORE G1
import g1_anchored_ball as G1                          # noqa: E402

import e261_rank_ladder as E261                        # noqa: E402 — THE
                                                      # MACHINERY (SRCT +
                                                      # LadderRooms), PORTED
                                                      # WHOLE BY IMPORT (the
                                                      # committed file is NOT
                                                      # modified; its module
                                                      # import carries the
                                                      # family cuda assert —
                                                      # NO cuda op runs here)

torch.set_num_threads(4)           # the CPU lane's whole budget (dispatch)

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402

SMOKE = False                     # desk cell: deterministic, cheap — no smoke
CPU = torch.device("cpu")
NAME = "x14"

T0 = time.time()
RD = run_dir(NAME)
LOG_PATH = RD / "run.log"
_logf = open(LOG_PATH, "a", encoding="utf-8")
HEAD0: str | None = None                  # the birth head (set in __main__)


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


G1.log = log                                          # unify the timeline

# ---- THE REBINDING (e268/e273/e278/e283's disclosed convention): the ported
# machinery resolves its module globals at CALL TIME through e261's namespace
# — rebound HERE so the room build + certify label THIS cell.
E261.log = log
E261.NAME = NAME
E261.SMOKE = SMOKE
E261.T0 = T0

# ======================================================================
# THE CONFIG (frozen)
# ======================================================================
BASE_CK = "e001.pt"               # the 2.74M corpus base (e043/e048's own B)
ROOT_CK = "g1c_root.pt"           # the committed fresh root (THE reference)
SPAN_CK = "e246_late_span.pt"     # e246's committed LATE span (the ledger's)
VMAP_CK = "e258_vmap.pt"          # e258's committed 2.74M v-map (the ledger's)
ROOMS264_CK = "e264_rooms.pt"     # e264's committed rooms (the K10K bit-bind)
CKPT_DIR = GB.CKPT_DIR

# ---- THE ROOM: the committed K10K room (k=10k, seeds 26113/26114) — the
# room the established fact was WRITTEN IN (the vehicle's own room)
LADDER: tuple[tuple[int, int, int], ...] = (
    (10_000, 26113, 26114),       # K10K — e261's registered seed pair
)
RUNG = {k: "K10K" for k, _, _ in LADDER}
ROOM_MODE = "K10K"
E261.LADDER = LADDER            # the machinery's certify()/rooms read these
E261.RUNG_NAMES = RUNG

# ---- THE STATES
FACT_CK = "e261_K10K_inst_resume.pt"        # t0: the loaded established fact
FACT_MD5 = "0f6dc1cf46850ce655dfafc9c853d467"
FACT_SIZE = 32958479
FACT_STEP = 400
FACT_TRAJ_STEPS = [1, 100, 200, 300, 400]
FACT_LEDGER_MAX = 400
FACT_FLAT_MD5 = "ebebb4472725d582dd74928493f1bfb3"   # e283's G_FACTLOAD bind
POST_CK = "e283_ESTABLISHED-CONCURRENT_post.pt"      # t400: the post state

# ---- the committed records, HARD-BOUND (read at runtime from their paths
# and asserted against these literals; Rule 12)
E264_METRICS = E43.REPO / "runs" / "e264" / "metrics.json"
E264_MD5 = "a42ff4786784b04cb9819a69b545e343"
FACT_BASELINE_G0 = 0.26464763283729553      # e264's committed K10K post g0
FACT_BASELINE_GM12 = 0.10525520890951157    # e264's committed K10K post gm12

E268_METRICS = E43.REPO / "runs" / "e268" / "metrics.json"
E268_MD5 = "c1149229b7f0191943a7b8eb0442b494"

E278_METRICS = E43.REPO / "runs" / "e278" / "metrics.json"
E278_MD5 = "db14cdff1fd5021a5b255c12127ea9df"

E283_METRICS = E43.REPO / "runs" / "e283" / "metrics.json"
E283_POST_G0 = 9.04614535102155e-06          # the established-concurrent post
E283_POST_GM12 = 2.3886157578090206e-05      # read (the dead write)
E283_POST_CE_R = 1.6652382612228394
E283_DRIFT_NORM_T400 = 14.453935847202613    # the disp_ledger's fp64 reads
E283_DRIFT_FRAC_T400 = 0.1140634378314966
E283_REMAIN_NORM_T400 = 16.08044667112344
E283_REMAIN_FRAC_T400 = 0.4545671609496503
E283_FACT_IN_ROOM_T0 = 0.9441588788935659    # the write's standing t0 reads
E283_FACT_DISP_NORM_T0 = 9.1788432658723

ROOMS264_MD5 = "2d524655575cce00a3bc1c8770f4b211"    # e268/e283's bind

# ---- the discriminator's frozen numbers
RESURRECT_BAR = 0.10             # TRANSPORT-LITERAL: arm-A read >= this
DEAD_BAR = 0.01                  # OVERWRITTEN-LEANING: arm-A read < this
EROSION_BAR = 0.50               # ...AND in-room mass < this fraction of t0
FACT_READ_TOL_G0 = 2e-6          # G_FACTLOAD behavioral bars (the family's
FACT_READ_TOL_GM12 = 1e-5        # cross-session read-determinism law, with
                                 # headroom; disclosed)
POST_READ_TOL_G0 = 2e-6          # G_POSTLOAD behavioral bars (same law)
POST_READ_TOL_GM12 = 5e-6
GEOM_TOL_NORM = 1e-6             # G_POSTLOAD geometric bars (fp64 on the
GEOM_TOL_FRAC = 1e-9             # same fp32 states; the measured cross-run
                                 # DCT scatter is ~1e-14, disclosed in-gate)
G_READ_TOL = E261.G_READ_TOL     # 5e-3 (the root gate's)

REGISTERED = {
    "bars_verbatim": {
        "TRANSPORT-LITERAL": "the orthogonal-subtraction read >= 0.10 (a "
            "substantial fraction of baseline) — the write survives its "
            "context's removal.",
        "OVERWRITTEN-LEANING": "the read stays < 0.01 AND the write-mass "
            "ledger shows the in-room mass eroded below 50% of its t0 "
            "value.",
        "MIXED/INCONCLUSIVE": "everything else — including the coupling "
            "caveat firing (mass intact + no resurrection).",
    },
    "arms": {
        "a_write_mass_ledger": "per milestone state: ||P_room(theta_t - "
            "theta_base)|| (the write's surviving in-room mass) + "
            "||(I-P)(theta_t - theta_base)|| (the out-of-room residual); "
            "t100-300 cited-derived from e283's committed fp64 ledger "
            "products (the states were not saved — disclosed)",
        "b_subtraction": "arm A (primary): theta_t400 - (I-P)(theta_t400 - "
            "theta_t0), read g0; arm B (converse control): theta_t400 - "
            "P(theta_t400 - theta_t0), read g0; arm ID (identity): "
            "theta_t400 - (theta_t400 - theta_t0), must return the fact's "
            "own read",
        "c_honesty": "the subtraction assumes component-wise additivity (a "
            "linear superposition null); any coupling (LN, softmax) can "
            "mask resurrection — a NECESSARY-not-SUFFICIENT test: "
            "resurrection => transport; no resurrection => inconclusive "
            "between overwrite and coupling, NOT proof of overwrite",
    },
    "prediction_cited": "TRANSPORT-LITERAL predicts the orthogonal-"
        "subtraction read resurrects toward 0.2646 (>= 0.10); OVERWRITTEN "
        "predicts it stays dead (< 0.01) (CITED from the dispatch, not "
        "re-registered)",
    "composite_order": "TEXTURE (any hard-gate failure) -> TRANSPORT-"
        "LITERAL (arm-A g0 >= 0.10) -> OVERWRITTEN-LEANING (arm-A g0 < 0.01 "
        "AND in-room mass ratio < 0.50) -> MIXED/INCONCLUSIVE (everything "
        "else; the coupling-caveat-fired branch = mass ratio >= 0.50 AND "
        "arm-A g0 < 0.10, named at birth, no bar moved)",
    "registration": "bars + arms + honesty block frozen VERBATIM from the "
        "R67 dispatch letter (R67's named desk cell); this script "
        "committed at birth BEFORE any compute; adjudicate against exactly "
        "this; no bar shopping.",
}

deviations: list[str] = [
    "THE MILESTONE-GRID DISCLOSURE (the dispatch's own instruction): e283 "
    "checkpointed ONLY the t400 post state (plus the fact's t0); the "
    "t100/200/300 milestone STATES were never saved (the resume ckpt holds "
    "the step-400 model + optimizer only — inspected at design time). The "
    "ledger therefore COMPUTES t0 + t400 and CITES t100-300 as derived "
    "products of e283's committed fp64 ledger reads (remaining_from_base_"
    "norm x remaining_from_base_in_room_frac); the t400-only intervention "
    "answers the primary (the dispatch: 'that alone answers the primary').",
    "CPU-ONLY desk cell (dispatch): torch threads 4, pocketfft workers 2, "
    "no training runs, no corpus stream, no optimizer — reads + fp64 "
    "projections only; NO envelope-log writes; NO cuda tensor is ever "
    "created (the e261 import carries the family's cuda-availability "
    "assert — its module-level side effects are the runs/e261/run.log "
    "append-handle open and nothing else, the e278/e283 precedent).",
    "THE POST STATE HAS NO COMMITTED FILE-MD5 (e283 recorded none): bound "
    "here BEHAVIORALLY (g0/gm12/CE_R vs the committed literals, the "
    "family's cross-session read-determinism law) AND GEOMETRICALLY (the "
    "fp64 drift/remaining norms + in-room fracs vs the committed "
    "disp_ledger) — a value-bind, stronger than a file-bind; the file md5 "
    "is recorded here for the future.",
    "THE PROBES NEED NO STREAM: G_ANCHOR / G_INSTMASK / G_CORPUSGEN / "
    "G_CONTROL (e283's stream-side gates) do not apply — nothing trains, "
    "nothing installs, no stream runs in this cell; the splice/battery "
    "convention gates (G_SPLICE / G_BATTERY) carry the probe identity.",
    "THE ALIGNMENT DIAGNOSTICS ARE CONTEXT, NOT BARS (no bar shopping): "
    "cos(surviving in-room mass, the write's own t0 in-room direction) "
    "and the erosion-degeneracy interval are co-reported because they "
    "quantify exactly what the committed reads could not separate "
    "(R67's finding-3 arithmetic, reconstructed from the committed "
    "numbers); the adjudication uses ONLY the frozen bars.",
    "n=1 lineage, one session, one draw of history (the g-series standing "
    "lottery note carried verbatim); nothing guaranteed.",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator "
    "folds).",
]


def md5of(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def git_head() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(E43.REPO),
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:                                       # noqa: BLE001
        return "unavailable"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def flat_params_cpu(net) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).cpu()
                      for p in net.parameters()]).clone()


def flat64_of(net) -> np.ndarray:
    return flat_params_cpu(net).double().numpy().astype(np.float64)


def set_flat_from64(net, x64: np.ndarray) -> float:
    """Materialize an fp64 flat into the net's fp32 parameters (the arms);
    returns the fp64->fp32 quantization residual norm (disclosed)."""
    idx, resid2 = 0, 0.0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            seg64 = x64[idx: idx + n]
            seg32 = seg64.astype(np.float32)
            d = seg32.astype(np.float64) - seg64
            resid2 += float((d * d).sum())
            p.copy_(torch.from_numpy(seg32).reshape(p.shape))
            idx += n
    assert idx == int(x64.size), f"flat mismatch {idx} vs {x64.size}"
    return math.sqrt(resid2)


def read_cells(net, g0_ids, gm12_ids, gp12_ids, zid, r_eval_xy) -> dict:
    return {"g0": G1.battery_cell(net, g0_ids, zid)["mean_pz"],
            "gm12": G1.battery_cell(net, gm12_ids, zid)["mean_pz"],
            "gp12": G1.battery_cell(net, gp12_ids, zid)["mean_pz"],
            "ce_r": G1.ce_fixed_cpu(net, *r_eval_xy)}


# ------------------------------------------------------------------ main
metrics: dict = {}


def write_partial(note: str) -> None:
    metrics["date"] = now_iso()
    metrics["phase_note"] = note
    save_json(RD / "metrics.json", E43.jsonable(metrics))
    log(f"WROTE partial metrics ({note})")


def main():
    metrics.update({
        "experiment": "x14_transport_intervention",
        "phase": "THE TRANSPORT INTERVENTION — R67's named desk cell: "
                 "e283's correlational dichotomy (transport vs overwrite, "
                 "both fitting every committed number) converted into an "
                 "interventional one — the write-mass ledger on the "
                 "checkpointed states + the subtraction intervention "
                 "(theta_t400 minus the out-of-room drift component, the "
                 "probe read) + the converse control — TRANSPORT-LITERAL "
                 "vs OVERWRITTEN-LEANING vs MIXED/INCONCLUSIVE",
        "date": now_iso(),
        "status": "PARTIAL: startup",
        "registration": REGISTERED["registration"],
        "registered": REGISTERED,
        "smoke": SMOKE,
        "envelope": {
            "device": "CPU-ONLY desk cell — torch threads 4, pocketfft "
                      "workers 2, CPU fp64 dense projections; NO GPU ops, "
                      "NO envelope-log writes, no training (reads + "
                      "projections only)",
            "trainings": "NONE — the states are loaded (t0 fact + t400 "
                         "post); nothing moves",
            "timestamps": "datetime.now(UTC) only",
        },
        "deviations": deviations,
        "builds_on": [
            "R67 / REVIEWS.md (the critic's finding 3: e283's transport-vs-"
            "overwrite NOT separated by the committed reads — the intact-"
            "write band 7.0-10.3 vs measured 7.31; this cell is its named "
            "repair, dispatched in the R67 fold)",
            "T261 / e283 (THE established-fact collision: ESTABLISHED-DIES "
            "at 0.0000342x; drift 14.45 vs the write's 9.18; the t400 post "
            "state checkpointed — this cell's object)",
            "T260 / e278 (the roach motel: the optimizer re-aims — the "
            "overwrite channel's mechanism cite)",
            "T244 / e268 + T242 / e264 + T239 / e261 (the forming-"
            "concurrent reference, the committed threshold rung, the "
            "vehicle fact itself)",
            "T239 / e261 (the machinery PORTED WHOLE BY IMPORT: the SRCT "
            "projector, LadderRooms, the certification)",
        ],
        "whats_new": [
            "THE SUBTRACTION INTERVENTION ITSELF: the record's first "
            "INTERVENTION on a post-mortem state — remove a projection "
            "component of the drift and read the probe; correlational "
            "'transport' becomes falsifiable",
            "THE WRITE-MASS LEDGER: ||P(theta_t - base)|| + ||(I-P)(theta_t "
            "- base)|| per milestone — the decomposition e283's committed "
            "fractions imply but never stated as masses",
            "THE CONVERSE ARM: subtracting the IN-ROOM component instead "
            "inverts the two hypotheses' predictions — the dichotomy's "
            "other half, co-interpreted",
        ],
        "gates": {},
    })
    log("X14 — THE TRANSPORT INTERVENTION (R67's named desk cell) -> "
        f"{RD}")
    log(f"bars: TRANSPORT-LITERAL >= {RESURRECT_BAR}; OVERWRITTEN-LEANING "
        f"< {DEAD_BAR} AND mass < {EROSION_BAR:.0%} of t0; else "
        f"MIXED/INCONCLUSIVE")
    write_partial("startup (bars + conventions registered, committed at "
                  "birth)")

    # ================= P0: the protocol rebuild (the probe identity) =====
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
    for host in G1.HOSTS:
        for p in E43.find_occ(train_text, host):
            if p >= 280 and p + len(host) + G1.POST_CAP <= len(train_ids):
                host_occ.append((p, host))
    rng = random.Random(E43.SPLICE_RNG)
    rng.shuffle(host_occ)
    install_occ = host_occ[:60]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix, "pass": bool(mix == {"FLORIZEL": 19,
                                                         "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"

    bat_ids = {}
    for j in G1.GEOS:
        cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    G_BATTERY = {"shapes": {f"g{j:+d}": list(bat_ids[j].shape)
                            for j in G1.GEOS},
                 "expected": {"g-12": [60, G1.PRE - 12], "g0": [60, G1.PRE],
                              "g+12": [60, G1.PRE + 12]},
                 "pass": bool(list(bat_ids[-12].shape) == [60, G1.PRE - 12]
                              and list(bat_ids[0].shape) == [60, G1.PRE]
                              and list(bat_ids[12].shape)
                              == [60, G1.PRE + 12])}
    assert G_BATTERY["pass"], f"battery geometry drift: {G_BATTERY}"
    gm12_ids, g0_ids, gp12_ids = bat_ids[-12], bat_ids[0], bat_ids[12]

    r_eval_x, r_eval_y = G1.val_windows(val_ids, val_text, 60,
                                        G1.R_EVAL_SEED)
    r_eval_xy = (r_eval_x, r_eval_y)
    metrics["gates"].update({"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                             "G_BATTERY": G_BATTERY})
    log("P0: probe gates PASS (namefree / splice 19+41 / battery shapes)")
    write_partial("P0 probe gates PASSED")

    # ================= P0b: the parents hard-bound (Rule 12) ============
    e264m = json.loads(E264_METRICS.read_text(encoding="utf-8"))
    e264_post = e264m["arms"]["K10K"]["install"]["post_cells"]["g0"]
    e264_gm12 = e264m["arms"]["K10K"]["install"]["post_cells"]["gm12"]
    vehicle = torch.load(CKPT_DIR / FACT_CK, map_location="cpu",
                         weights_only=False)
    vehicle_state = {"step": int(vehicle["step"]),
                     "traj_steps": [t["step"] for t in vehicle["traj"]],
                     "ledger_max": max(int(kk) for kk in
                                       vehicle["ledger"].keys())}
    fact_sd = {k: v.detach().clone() for k, v in vehicle["model"].items()}
    del vehicle
    e283m = json.loads(E283_METRICS.read_text(encoding="utf-8"))
    post_ck_path = CKPT_DIR / POST_CK
    G_PARENTS = {
        "e264_metrics": {"path": str(E264_METRICS),
                         "md5": md5of(E264_METRICS), "bound_md5": E264_MD5,
                         "K10K_post_g0": e264_post, "K10K_post_gm12": e264_gm12,
                         "note": "THE loaded fact's committed record"},
        "e268_metrics": {"path": str(E268_METRICS),
                         "md5": md5of(E268_METRICS), "bound_md5": E268_MD5},
        "e278_metrics": {"path": str(E278_METRICS),
                         "md5": md5of(E278_METRICS), "bound_md5": E278_MD5},
        "e283_metrics": {"path": str(E283_METRICS),
                         "md5": md5of(E283_METRICS),
                         "note": "THE parent cell (no prior md5 bind — its "
                                 "final literals are hard-bound in-code and "
                                 "gated by G_POSTLOAD)"},
        "the_fact": {"path": f"runs/checkpoints/{FACT_CK}",
                     "md5": md5of(CKPT_DIR / FACT_CK),
                     "bound_md5": FACT_MD5,
                     "size": (CKPT_DIR / FACT_CK).stat().st_size,
                     "bound_size": FACT_SIZE, "state": vehicle_state},
        "the_post_state": {"path": f"runs/checkpoints/{POST_CK}",
                           "md5": md5of(post_ck_path),
                           "size": post_ck_path.stat().st_size,
                           "note": "t400 — e283's established-concurrent "
                                   "post state; bound BEHAVIORALLY + "
                                   "GEOMETRICALLY in G_POSTLOAD (no "
                                   "committed file-md5 exists; the "
                                   "value-bind is the stronger one)"},
        "e264_rooms": {"path": f"runs/checkpoints/{ROOMS264_CK}",
                       "md5": md5of(CKPT_DIR / ROOMS264_CK),
                       "bound_md5": ROOMS264_MD5},
        "e246_span": {"path": f"runs/checkpoints/{SPAN_CK}",
                      "md5": md5of(CKPT_DIR / SPAN_CK),
                      "bound_md5": E261.E246_SPAN_MD5},
        "e258_vmap": {"path": f"runs/checkpoints/{VMAP_CK}",
                      "md5": md5of(CKPT_DIR / VMAP_CK)},
        "hardbound": {
            "fact_baseline_g0": FACT_BASELINE_G0,
            "fact_baseline_gm12": FACT_BASELINE_GM12,
            "e283_post_g0": E283_POST_G0, "e283_post_gm12": E283_POST_GM12,
            "e283_post_ce_r": E283_POST_CE_R,
            "e283_drift_norm_t400": E283_DRIFT_NORM_T400,
            "e283_drift_frac_t400": E283_DRIFT_FRAC_T400,
            "e283_remain_norm_t400": E283_REMAIN_NORM_T400,
            "e283_remain_frac_t400": E283_REMAIN_FRAC_T400,
            "e283_fact_in_room_t0": E283_FACT_IN_ROOM_T0,
            "e283_fact_disp_norm_t0": E283_FACT_DISP_NORM_T0},
        "pass": bool(
            md5of(E264_METRICS) == E264_MD5
            and abs(e264_post - FACT_BASELINE_G0) < 1e-12
            and abs(e264_gm12 - FACT_BASELINE_GM12) < 1e-12
            and md5of(E268_METRICS) == E268_MD5
            and md5of(E278_METRICS) == E278_MD5
            and md5of(CKPT_DIR / FACT_CK) == FACT_MD5
            and (CKPT_DIR / FACT_CK).stat().st_size == FACT_SIZE
            and vehicle_state["step"] == FACT_STEP
            and vehicle_state["traj_steps"] == FACT_TRAJ_STEPS
            and vehicle_state["ledger_max"] == FACT_LEDGER_MAX
            and post_ck_path.exists()
            and md5of(CKPT_DIR / ROOMS264_CK) == ROOMS264_MD5
            and md5of(CKPT_DIR / SPAN_CK) == E261.E246_SPAN_MD5),
    }
    assert G_PARENTS["pass"], f"parent bind failed: {G_PARENTS}"
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    log("P0b: G_PARENTS PASS — the fact (md5/size/step) + the room file + "
        "the parent metrics md5-bound; the post state's md5 recorded "
        f"({G_PARENTS['the_post_state']['md5'][:8]}...)")
    write_partial("P0b parents hard-bound")
    del e264m

    # ---- G-BASE: the 2.74M corpus base, loaded fixed + fact-free --------
    base_net = G1.load_g1(CKPT_DIR / BASE_CK)
    assert base_net.num_params() == GB.G1B_PARAMS, \
        f"base param count {base_net.num_params()} != {GB.G1B_PARAMS}"
    base_sd = {k: v.detach().clone() for k, v in base_net.state_dict().items()}
    base_gm12 = G1.battery_cell(base_net, gm12_ids, zid)["mean_pz"]
    base_ce_r = G1.ce_fixed_cpu(base_net, *r_eval_xy)
    G_BASE = {"checkpoint": f"runs/checkpoints/{BASE_CK}",
              "params": GB.G1B_PARAMS,
              "fact_free_gm12": base_gm12, "ce_r": base_ce_r,
              "fact_free": bool(base_gm12 <= 0.05),
              "pass": bool(base_gm12 <= 0.05)}
    assert G_BASE["pass"], f"G-BASE FAILED: {G_BASE}"
    del base_net
    metrics["gates"]["G_BASE"] = G_BASE
    log(f"G-BASE: {BASE_CK} ({GB.G1B_PARAMS} params), fact-free "
        f"(g-12 {base_gm12:.4f}, CE_R {base_ce_r:.4f}): PASS")
    write_partial("P0c G-BASE PASSED")

    # ================= P1: THE ROOM (v-map + span + cert + bit-bind) ====
    log("=" * 78)
    root_net = G1.load_g1(CKPT_DIR / ROOT_CK)
    n_par = root_net.num_params()
    theta_root = flat_params_cpu(root_net)
    root_read = G1.battery_cell(root_net, gm12_ids, zid)["mean_pz"]
    G_ROOT = {
        "checkpoint": f"runs/checkpoints/{ROOT_CK}",
        "n_params": n_par, "expected_params": GB.G1B_PARAMS,
        "battery_read_measured": root_read,
        "battery_read_committed": E261.G1C_ROOT_GM12,
        "abs_diff": abs(root_read - E261.G1C_ROOT_GM12),
        "tol": G_READ_TOL,
        "flat_md5": hashlib.md5(theta_root.numpy().tobytes()).hexdigest(),
        "pass": bool(n_par == GB.G1B_PARAMS
                     and abs(root_read - E261.G1C_ROOT_GM12) < G_READ_TOL)}
    assert G_ROOT["pass"], f"root gate FAILED: {G_ROOT}"
    metrics["gates"]["G_ROOT"] = G_ROOT
    del root_net
    log(f"P1 G_ROOT: PASS (|d| {G_ROOT['abs_diff']:.1e})")
    write_partial("P1 G_ROOT PASSED")

    N = n_par
    base_flat_np = flat64_of(G1.evl_load(base_sd))

    vmap_art = torch.load(CKPT_DIR / VMAP_CK, map_location="cpu",
                          weights_only=False)
    v_flat32 = vmap_art["model"]["v_flat_fp32"]
    v64_np = v_flat32.numpy().astype(np.float64)
    G_VMBIND = {
        "path": f"runs/checkpoints/{VMAP_CK}", "md5": md5of(CKPT_DIR / VMAP_CK),
        "meta_experiment": vmap_art.get("meta", {}).get("experiment"),
        "meta_k": vmap_art.get("meta", {}).get("k"),
        "size": int(v_flat32.numel()), "expected_size": N,
        "mean_v": float(v64_np.mean()),
        "pass": bool(vmap_art.get("meta", {}).get("experiment") == "e258"
                     and int(v_flat32.numel()) == N
                     and int(vmap_art["meta"]["k"]) == E261.E258_K_HARD),
    }
    assert G_VMBIND["pass"], f"v-map bind failed: {G_VMBIND}"
    metrics["gates"]["G_VMBIND"] = G_VMBIND
    del vmap_art

    span_art = torch.load(CKPT_DIR / SPAN_CK, map_location="cpu",
                          weights_only=False)
    Vp = span_art["Vp"].contiguous()
    G_SPANBIND = {"md5": md5of(CKPT_DIR / SPAN_CK),
                  "rank": int(Vp.shape[0]), "N": int(Vp.shape[1]),
                  "meta_experiment": span_art.get("meta", {}).get("experiment"),
                  "pass": bool(md5of(CKPT_DIR / SPAN_CK) == E261.E246_SPAN_MD5
                               and int(Vp.shape[0]) == E261.E246_SPAN_RANK
                               and int(Vp.shape[1]) == N
                               and span_art.get("meta", {}).get("experiment")
                               == "e246")}
    assert G_SPANBIND["pass"], f"span bind failed: {G_SPANBIND}"
    metrics["gates"]["G_SPANBIND"] = G_SPANBIND
    del span_art

    params_ref = list(G1.evl_load(base_sd).parameters())
    rooms = E261.LadderRooms(N, LADDER, v64_np, Vp.numpy().astype(np.float64),
                             params_ref, CPU)
    cert = rooms.certify()
    G_PROJ = {
        "form": "the K10K room certified (fp64 CPU, "
                f"{E261.CERT_PROBES} probes, seed {E261.CERT_SEED}): the "
                "DCT roundtrip identity; IDEMPOTENCY and the kept^2 rank "
                "probe (||P x||^2/||x||^2 vs k/N, the 10-sigma bar "
                "5*sqrt(2k)/N); the span-overlap (expect ~sqrt(k/N))",
        "reads": cert,
        "bars": {"roundtrip": 1e-8, "idempotency": 1e-8,
                 "kept2": "10-sigma (5*sqrt(2k)/N)"},
        "pass": bool(cert["pass"]),
    }
    assert G_PROJ["pass"], f"room certification FAILED: {G_PROJ}"
    metrics["gates"]["G_PROJ"] = G_PROJ
    for nm, r in cert["per_rung"].items():
        log(f"  room {nm}: k {r['k']} (seeds {r['seeds']}) idem "
            f"{r['idempotency_max']:.1e} kept2 {r['kept2_mean']:.6f} vs "
            f"{r['kept2_expect']:.6f} span-ovl {r['span_overlap_mean']:.4f}")

    # ---- G_ROOMK10K: bit-identity vs e264's committed K10K room ---------
    rooms264 = torch.load(CKPT_DIR / ROOMS264_CK, map_location="cpu",
                          weights_only=False)

    def _to_np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    D264 = _to_np(rooms264["model"]["K10K"]["D_int8"]).astype(np.float64)
    S264 = _to_np(rooms264["model"]["K10K"]["S"])
    room = rooms.rooms[ROOM_MODE]
    G_ROOMK10K = {
        "form": "the room == the fact's own committed K10K room (seeds "
                "26113/26114 at k=10,000): the +-1 diagonal and the index "
                "set bit-identical to e264_rooms.pt's stored K10K D/S "
                "(exact equality)",
        "D_bit_equal": bool(np.array_equal(room.D, D264)),
        "S_bit_equal": bool(np.array_equal(room.S, S264)),
        "e264_rooms_md5": md5of(CKPT_DIR / ROOMS264_CK),
        "pass": bool(np.array_equal(room.D, D264)
                     and np.array_equal(room.S, S264)
                     and int(rooms264["model"]["K10K"]["k"])
                     == LADDER[0][0]
                     and list(rooms264["model"]["K10K"]["seeds"])
                     == [LADDER[0][1], LADDER[0][2]]),
    }
    del rooms264
    assert G_ROOMK10K["pass"], f"K10K room bind failed: {G_ROOMK10K}"
    metrics["gates"]["G_ROOMK10K"] = G_ROOMK10K
    metrics["rooms"] = {
        "vehicle": {"k": LADDER[0][0], "name": ROOM_MODE,
                    "seeds": [LADDER[0][1], LADDER[0][2]],
                    "k_fraction_of_N": LADDER[0][0] / N,
                    "bit_bound_to": f"runs/checkpoints/{ROOMS264_CK} "
                                    "(e264's committed K10K room — the "
                                    "fact's own room)"},
        "certification": cert,
    }
    log(f"P1 THE ROOM: {ROOM_MODE} (k={LADDER[0][0]}): BUILT + CERTIFIED + "
        "BIT-BOUND")
    write_partial("P1 the room built (certified + bit-bound)")

    # ================= P2: THE FACT (t0, loaded bit-exact + gated) ======
    log("=" * 78)
    fact_net = G1.evl_load(fact_sd)
    fact_flat = flat_params_cpu(fact_net)
    fact_flat_np = fact_flat.double().numpy().astype(np.float64)
    fact_flat_md5 = hashlib.md5(fact_flat.numpy().tobytes()).hexdigest()
    fact_cells = read_cells(fact_net, g0_ids, gm12_ids, gp12_ids, zid,
                            r_eval_xy)
    d_fact_base = fact_flat_np - base_flat_np
    loads_fact = rooms.displacement_loads(torch.from_numpy(d_fact_base),
                                          ROOM_MODE)
    G_FACTLOAD = {
        "form": "the established fact (t0), loaded BIT-EXACT and gated "
                "THREE ways: (1) the artifact (md5/size/step/traj/ledger "
                "— G_PARENTS), (2) the loaded state's flat-md5 vs e283's "
                "recorded bind, (3) the behavioral read (post g0 within "
                f"{FACT_READ_TOL_G0:.0e} / gm12 within "
                f"{FACT_READ_TOL_GM12:.0e} of e264's committed literals)",
        "flat_md5": fact_flat_md5, "bound_flat_md5": FACT_FLAT_MD5,
        "read_g0": {"mine": fact_cells["g0"], "committed": FACT_BASELINE_G0,
                    "abs_diff": abs(fact_cells["g0"] - FACT_BASELINE_G0)},
        "read_gm12": {"mine": fact_cells["gm12"],
                      "committed": FACT_BASELINE_GM12,
                      "abs_diff": abs(fact_cells["gm12"]
                                      - FACT_BASELINE_GM12)},
        "read_gp12": fact_cells["gp12"], "read_ce_r": fact_cells["ce_r"],
        "displacement_from_base": {
            "norm": float(np.linalg.norm(d_fact_base)), **loads_fact},
        "pass": bool(fact_flat_md5 == FACT_FLAT_MD5
                     and abs(fact_cells["g0"] - FACT_BASELINE_G0)
                     <= FACT_READ_TOL_G0
                     and abs(fact_cells["gm12"] - FACT_BASELINE_GM12)
                     <= FACT_READ_TOL_GM12),
    }
    assert G_FACTLOAD["pass"], f"G_FACTLOAD FAILED: {G_FACTLOAD}"
    metrics["gates"]["G_FACTLOAD"] = G_FACTLOAD
    log(f"P2 G_FACTLOAD: t0 LOADS — read g0 {fact_cells['g0']:.10f} vs "
        f"committed {FACT_BASELINE_G0:.10f} (|d| "
        f"{abs(fact_cells['g0'] - FACT_BASELINE_G0):.1e}); flat md5 "
        f"{fact_flat_md5[:10]}... == e283's bind: PASS")
    write_partial("P2 the fact (t0) loaded + gated")
    del fact_net

    # ================= P3: THE POST STATE (t400, value-bound) ===========
    post_art = torch.load(post_ck_path, map_location="cpu",
                          weights_only=False)
    post_sd = {k: v.detach().clone() for k, v in post_art["model"].items()}
    post_meta = {k: v for k, v in post_art.get("meta", {}).items()
                 if k != "desc"}
    del post_art
    post_net = G1.evl_load(post_sd)
    post_flat_np = flat64_of(post_net)
    post_cells = read_cells(post_net, g0_ids, gm12_ids, gp12_ids, zid,
                            r_eval_xy)
    drift64 = post_flat_np - fact_flat_np              # THE drift (fp64)
    P_drift = room.project(drift64)
    O_drift = drift64 - P_drift                         # (I-P) drift
    dn = float(np.linalg.norm(drift64))
    pn = float(np.linalg.norm(P_drift))
    on = float(np.linalg.norm(O_drift))
    d_frac = pn / dn
    rem400 = post_flat_np - base_flat_np
    P_rem400 = room.project(rem400)
    O_rem400 = rem400 - P_rem400
    rn = float(np.linalg.norm(rem400))
    m400 = float(np.linalg.norm(P_rem400))
    o400 = float(np.linalg.norm(O_rem400))
    G_POSTLOAD = {
        "form": "the post state (t400) bound BEHAVIORALLY (g0/gm12/CE_R vs "
                "e283's committed literals — the cross-session read-"
                "determinism law) AND GEOMETRICALLY (the fp64 drift/"
                "remaining norms + in-room fracs vs the committed "
                "disp_ledger): a value-bind, stronger than a file-bind "
                "(no committed file-md5 exists; the md5 is recorded in "
                "G_PARENTS)",
        "meta": post_meta,
        "read_g0": {"mine": post_cells["g0"], "committed": E283_POST_G0,
                    "abs_diff": abs(post_cells["g0"] - E283_POST_G0)},
        "read_gm12": {"mine": post_cells["gm12"],
                      "committed": E283_POST_GM12,
                      "abs_diff": abs(post_cells["gm12"] - E283_POST_GM12)},
        "read_ce_r": {"mine": post_cells["ce_r"],
                      "committed": E283_POST_CE_R,
                      "abs_diff": abs(post_cells["ce_r"] - E283_POST_CE_R)},
        "geom_drift_norm": {"mine": dn,
                            "committed": E283_DRIFT_NORM_T400,
                            "abs_diff": abs(dn - E283_DRIFT_NORM_T400)},
        "geom_drift_in_room_frac": {"mine": d_frac,
                                    "committed": E283_DRIFT_FRAC_T400,
                                    "abs_diff": abs(d_frac
                                                    - E283_DRIFT_FRAC_T400)},
        "geom_remaining_norm": {"mine": rn,
                                "committed": E283_REMAIN_NORM_T400,
                                "abs_diff": abs(rn - E283_REMAIN_NORM_T400)},
        "geom_remaining_in_room_frac": {
            "mine": m400 / rn, "committed": E283_REMAIN_FRAC_T400,
            "abs_diff": abs(m400 / rn - E283_REMAIN_FRAC_T400)},
        "tolerances": {"g0": POST_READ_TOL_G0, "gm12": POST_READ_TOL_GM12,
                       "ce_r": 5e-3, "norm": GEOM_TOL_NORM,
                       "frac": GEOM_TOL_FRAC,
                       "note": "fp64 on identical fp32 states; the measured "
                               "cross-run DCT scatter is ~1e-14 (the "
                               "traj/disp_ledger 1.3e-14 gap in e283's own "
                               "committed record)"},
        "pass": bool(
            abs(post_cells["g0"] - E283_POST_G0) <= POST_READ_TOL_G0
            and abs(post_cells["gm12"] - E283_POST_GM12)
            <= POST_READ_TOL_GM12
            and abs(post_cells["ce_r"] - E283_POST_CE_R) <= 5e-3
            and abs(dn - E283_DRIFT_NORM_T400) <= GEOM_TOL_NORM
            and abs(d_frac - E283_DRIFT_FRAC_T400) <= GEOM_TOL_FRAC
            and abs(rn - E283_REMAIN_NORM_T400) <= GEOM_TOL_NORM
            and abs(m400 / rn - E283_REMAIN_FRAC_T400) <= GEOM_TOL_FRAC),
    }
    assert G_POSTLOAD["pass"], f"G_POSTLOAD FAILED: {G_POSTLOAD}"
    metrics["gates"]["G_POSTLOAD"] = G_POSTLOAD
    log(f"P3 G_POSTLOAD: t400 LOADS — read g0 {post_cells['g0']:.6e} vs "
        f"committed {E283_POST_G0:.6e} (|d| "
        f"{abs(post_cells['g0'] - E283_POST_G0):.1e}); drift |d| {dn:.6f} "
        f"vs {E283_DRIFT_NORM_T400:.6f}; in-room frac {d_frac:.10f} vs "
        f"{E283_DRIFT_FRAC_T400:.10f}: PASS")
    metrics["the_states"] = {
        "t0_fact": {"checkpoint": f"runs/checkpoints/{FACT_CK}",
                    "read": fact_cells,
                    "displacement_from_base": G_FACTLOAD[
                        "displacement_from_base"]},
        "t400_post": {"checkpoint": f"runs/checkpoints/{POST_CK}",
                      "read": post_cells,
                      "drift_from_fact": {"norm": dn, "in_room_norm": pn,
                                          "out_room_norm": on,
                                          "in_room_frac": d_frac},
                      "remaining_from_base": {"norm": rn, "in_room_norm": m400,
                                              "out_room_norm": o400}},
    }
    write_partial("P3 the post state (t400) loaded + value-bound")
    del post_net

    # ================= P4: (a) THE WRITE-MASS LEDGER ====================
    log("=" * 78)
    rem0 = fact_flat_np - base_flat_np                  # the write itself
    P_rem0 = room.project(rem0)
    O_rem0 = rem0 - P_rem0
    m0 = float(np.linalg.norm(P_rem0))                  # the write's in-room
    o0 = float(np.linalg.norm(O_rem0))                  # t0 mass
    tot0 = float(np.linalg.norm(rem0))
    # the cited grid (e283's committed fp64 products; states not on disk)
    cited_grid = []
    for row in e283m["arms"]["ESTABLISHED-CONCURRENT"]["phase"]["disp_ledger"]:
        t = int(row["step"])
        rn_c = float(row["remaining_from_base_norm"])
        fr_c = float(row["remaining_from_base_in_room_frac"])
        cited_grid.append({
            "step": t, "source": ("COMPUTED (state on disk)"
                                  if t == 400 else
                                  "CITED-DERIVED (norm x frac from e283's "
                                  "committed fp64 ledger; state not saved)"),
            "remaining_norm": rn_c,
            "in_room_mass_cited": rn_c * fr_c,
            "out_room_mass_cited": math.sqrt(max(rn_c * rn_c
                                                 - (rn_c * fr_c) ** 2, 0.0)),
            "in_room_frac": fr_c})
    # cross-check: the t400 cited products vs THIS session's direct reads
    cite400 = cited_grid[-1]
    xchk_m = abs(cite400["in_room_mass_cited"] - m400)
    xchk_r = abs(cite400["remaining_norm"] - rn)
    # diagnostics (CONTEXT, not bars): the alignment of the surviving mass
    cos_mass = float((P_rem400 @ P_rem0) / (m400 * m0)) if m0 > 0 else None
    dR = pn                                               # |P delta|
    deg_lo, deg_hi = max(m400 - dR, 0.0), m400 + dR      # erosion degeneracy
    mass_ratio = m400 / m0
    ledger = {
        "form": "the write's surviving in-room mass ||P(theta_t - base)|| "
                "+ the out-of-room residual ||(I-P)(theta_t - base)||, "
                "fp64, per milestone (t0/t400 COMPUTED from the states on "
                "disk; t100-300 cited-derived from e283's committed "
                "ledger — disclosed)",
        "t0": {"in_room_mass": m0, "out_room_mass": o0, "total": tot0,
               "in_room_frac": m0 / tot0,
               "source": "COMPUTED (the fact ckpt)"},
        "t400": {"in_room_mass": m400, "out_room_mass": o400, "total": rn,
                 "in_room_frac": m400 / rn,
                 "source": "COMPUTED (the post ckpt)"},
        "grid": cited_grid,
        "crosscheck_t400_cited_vs_computed": {
            "in_room_mass_abs_diff": xchk_m, "remaining_norm_abs_diff": xchk_r,
            "note": "the cited products and this session's direct fp64 "
                    "projections agree (same states, same projector)"},
        "mass_ratio_t400_over_t0": mass_ratio,
        "erosion_bar_half": EROSION_BAR,
        "diagnostics_context_not_bars": {
            "R67_finding3_reconstruction": {
                "intact_band_lo": m0 - dR, "intact_band_hi": m0 + dR,
                "measured_m400": m400,
                "note": "the intact-write band [m0 - |P delta|, m0 + "
                        "|P delta|] = the critic's 7.0-10.3 (reconstructed "
                        "from the committed numbers); 7.31 sits inside it, "
                        "near the opposed edge"},
            "erosion_degeneracy": {
                "surviving_fraction_lo": deg_lo / m0,
                "surviving_fraction_hi": deg_hi / m0,
                "note": "under additivity the same m400 is consistent with "
                        "ANY surviving in-room fraction in this interval — "
                        "the committed reads cannot separate the two "
                        "hypotheses (R67's finding 3); the intervention "
                        "exists to cut exactly this"},
            "cos_surviving_mass_vs_write_in_room": cos_mass,
            "note": "the surviving in-room mass's direction vs the write's "
                    "own t0 in-room direction (1.0 = still the write's "
                    "mass, not replacement mass); context, not a bar",
        },
    }
    metrics["write_mass_ledger"] = ledger
    log(f"P4 THE WRITE-MASS LEDGER: t0 in-room {m0:.4f} / out {o0:.4f} "
        f"(total {tot0:.4f}); t400 in-room {m400:.4f} / out {o400:.4f} "
        f"(total {rn:.4f}); ratio {mass_ratio:.4f}; cos(surviving, write) "
        f"{cos_mass:.4f}; the erosion degeneracy "
        f"[{deg_lo / m0:.3f}, {deg_hi / m0:.3f}]")
    for row in cited_grid:
        log(f"  t{row['step']:4d} [{row['source'][:12]}]: in-room mass "
            f"{row['in_room_mass_cited']:.4f}, out-of-room "
            f"{row['out_room_mass_cited']:.4f}")
    write_partial("P4 the write-mass ledger")

    # ================= P5: (b) THE SUBTRACTION INTERVENTION ============
    log("=" * 78)
    # the decomposition (fp64): drift = P_delta (in-room) + delta_O (out)
    thetaA = post_flat_np - O_drift      # ARM A: theta0 + P_delta (primary)
    thetaB = post_flat_np - P_drift      # ARM B: theta0 + delta_O (control)
    thetaID = post_flat_np - drift64     # ARM ID: theta0 (the closure)
    id_err = float(np.max(np.abs(thetaID - fact_flat_np)))

    def arm_read(x64, name):
        net = G1.evl_load(post_sd)               # buffers ride the post sd
        q = set_flat_from64(net, x64)
        cells = read_cells(net, g0_ids, gm12_ids, gp12_ids, zid, r_eval_xy)
        log(f"  ARM-{name}: g0 {cells['g0']:.6e} gm12 {cells['gm12']:.6e} "
            f"g+12 {cells['gp12']:.6e} CE_R {cells['ce_r']:.4f} "
            f"(fp32-quantization residual {q:.2e})")
        del net
        return cells, q

    log(f"  drift decomposition: |delta| {dn:.4f} = in-room {pn:.4f} + "
        f"out-of-room {on:.4f} (orthogonal by construction)")
    cells_A, armA_quant = arm_read(
        thetaA, "A-ORTHOGONAL-SUBTRACTION (theta400 - out-of-room drift; "
        "THE primary)")
    cells_B, armB_quant = arm_read(
        thetaB, "B-IN-ROOM-SUBTRACTION (theta400 - in-room drift; the "
        "converse control)")
    cells_ID, id_quant = arm_read(
        thetaID, "ID-IDENTITY (theta400 - delta; must return the fact's "
        "read)")
    id_closure_read = abs(cells_ID["g0"] - fact_cells["g0"])

    intervention = {
        "form": "fp64: delta = theta400 - theta0; P_delta = P_room(delta); "
                "delta_O = delta - P_delta; ARM A (primary) = theta400 - "
                "delta_O == theta0 + P_delta (the corpus's OUT-OF-ROOM "
                "transport removed); ARM B (control) = theta400 - P_delta "
                "== theta0 + delta_O (the in-room channel removed, the "
                "transport kept); ARM ID = theta400 - delta == theta0 "
                "(the construction's closure); arms materialized fp32",
        "drift_decomposition": {
            "drift_norm": dn, "in_room_norm": pn, "out_room_norm": on,
            "in_room_frac": d_frac,
            "construction_checks": {
                "max_abs(theta400 - delta - theta0)_fp64": id_err,
                "armA_minus_theta0_norm": float(np.linalg.norm(thetaA
                                                               - fact_flat_np)),
                "armB_minus_theta0_norm": float(np.linalg.norm(thetaB
                                                               - fact_flat_np)),
                "expected": "armA-theta0 == |P_delta| (in-room), armB-"
                            "theta0 == |delta_O| (out-of-room)"}},
        "arm_A_orthogonal_subtraction": {
            "desc": "THE primary: subtract the out-of-room displacement "
                    "component from the final state, then read g0 — "
                    "TRANSPORT-LITERAL predicts resurrection toward "
                    f"{FACT_BASELINE_G0:.4f} (>= {RESURRECT_BAR}); "
                    f"OVERWRITTEN predicts it stays dead (< {DEAD_BAR})",
            "read": cells_A, "fp32_quantization_residual": armA_quant},
        "arm_B_in_room_subtraction": {
            "desc": "the converse CONTROL: subtract the IN-ROOM component "
                    "instead — under transport-literal it stays dead (the "
                    "killer out-of-room context KEPT); under overwrite it "
                    "is this arm that removes the erosive channel; "
                    "co-interpreted, NOT adjudicated (no frozen bar)",
            "read": cells_B, "fp32_quantization_residual": armB_quant},
        "arm_ID_identity": {
            "desc": "the closure: subtracting BOTH components must return "
                    "the fact's own read (the construction's null)",
            "read": cells_ID, "fp32_quantization_residual": id_quant,
            "abs_g0_diff_vs_fact_read": id_closure_read},
    }
    metrics["intervention"] = intervention
    log(f"P5 THE SUBTRACTION: arm A g0 {cells_A['g0']:.6e} (bar "
        f"{RESURRECT_BAR}); arm B g0 {cells_B['g0']:.6e}; identity closure "
        f"|d g0| {id_closure_read:.1e} (vs the fact's own read)")
    write_partial("P5 the subtraction intervention")

    # ================= P6: ADJUDICATION (the frozen bars) ===============
    hard = dict(metrics["gates"])
    gates_pass = bool(all(g.get("pass") for g in hard.values()))
    g0A = cells_A["g0"]
    # the registered inconclusive branch, verbatim: "mass intact + no
    # resurrection" — no-resurrection means BELOW the dead bar (a partial
    # resurrection in [0.01, 0.10) is its own sub-branch, named below)
    coupling_caveat_fired = bool(mass_ratio >= EROSION_BAR
                                 and g0A < DEAD_BAR)

    if not gates_pass:
        failed = [k for k, g in hard.items() if not g.get("pass")]
        verdict = f"TEXTURE (GATE FAILURE: {', '.join(failed)})"
        clause = ("a hard gate failed — nothing adjudicated; the record is "
                  "complete for the autopsy")
    elif g0A >= RESURRECT_BAR:
        verdict = "TRANSPORT-LITERAL"
        clause = (f"the orthogonal-subtraction read {g0A:.6f} >= "
                  f"{RESURRECT_BAR} — the write survives its context's "
                  "removal: the corpus's out-of-room displacement was the "
                  "killer; the in-room mass was the write's own, intact "
                  f"(ledger ratio {mass_ratio:.4f} of t0); e283's "
                  "'carried away' reading lands INTERVENTIONALLY")
    elif g0A < DEAD_BAR and mass_ratio < EROSION_BAR:
        verdict = "OVERWRITTEN-LEANING"
        clause = (f"the read stays dead ({g0A:.2e} < {DEAD_BAR}) AND the "
                  f"write-mass ledger shows erosion (in-room mass "
                  f"{mass_ratio:.4f} < {EROSION_BAR:.0%} of t0) — the "
                  "write's own room-mass was consumed, not its context")
    else:
        verdict = "MIXED/INCONCLUSIVE"
        if coupling_caveat_fired:
            clause = (f"the read stays dead (arm A {g0A:.2e} < {DEAD_BAR}) "
                      f"while the in-room mass is NOT eroded below half "
                      f"(ratio {mass_ratio:.4f} >= {EROSION_BAR:.0%}) — "
                      "THE COUPLING CAVEAT FIRED: mass intact + no "
                      "resurrection is exactly the registered inconclusive "
                      "branch (the subtraction's linear-superposition null "
                      "can be masked by LN/softmax coupling; no-resurrection "
                      "does NOT prove overwrite — the necessary-not-"
                      "sufficient direction governs)")
        else:
            clause = (f"arm A read {g0A:.6f} sits between the bars "
                      f"({DEAD_BAR} <= read < {RESURRECT_BAR}) — a PARTIAL "
                      "resurrection (evidence against pure overwrite "
                      "either way, and short of the transport bar); the "
                      f"ledger ratio {mass_ratio:.4f}; the converse arm "
                      f"{cells_B['g0']:.2e} co-reported; the trajectories "
                      "verbatim")

    metrics["adjudication"] = {
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "composite_order": REGISTERED["composite_order"],
        "gates_pass": gates_pass,
        "reads": {
            "the_fact_t0": fact_cells,
            "the_post_t400": post_cells,
            "arm_A_orthogonal_subtraction_g0": g0A,
            "arm_B_in_room_subtraction_g0": cells_B["g0"],
            "identity_closure_abs_g0_diff": id_closure_read,
            "write_mass_ratio_t400_over_t0": mass_ratio,
            "coupling_caveat_fired": coupling_caveat_fired,
        },
        "verdict": verdict,
        "clause": clause,
    }
    log("=" * 78)
    log(f"X14 VERDICT: {verdict}")
    log(f"  the fact (t0):        g0 {fact_cells['g0']:.8f}")
    log(f"  the post (t400):      g0 {post_cells['g0']:.3e} (the dead write)")
    log(f"  ARM A (subtract out): g0 {g0A:.6e}  (TRANSPORT bar >= "
        f"{RESURRECT_BAR})")
    log(f"  ARM B (subtract in):  g0 {cells_B['g0']:.6e}  (the converse "
        "control)")
    log(f"  mass ratio t400/t0:   {mass_ratio:.4f}  (EROSION bar < "
        f"{EROSION_BAR:.0%})")
    log(f"  coupling caveat fired: {coupling_caveat_fired}")
    write_partial("P6 adjudicated")

    # ================= P7: honesty + figure + report ====================
    metrics["honesty"] = {
        "intervention_not_logits": "the arms share ONE loaded post state "
            "and ONE loaded fact; the ONLY delta between arm A and the "
            "dead post state is the removal of the out-of-room drift "
            "component (fp64 subtraction, fp32 materialization with the "
            "quantization residual measured); between arm A and the alive "
            "fact it is the presence of the in-room drift component — the "
            "probe read is behavioral, not a logit read",
        "the_additivity_null": REGISTERED["arms"]["c_honesty"],
        "necessary_not_sufficient": "resurrection => transport (arm A's "
            "read >= 0.10 would have been sufficient for TRANSPORT-"
            "LITERAL); no resurrection does NOT prove overwrite — LN and "
            "softmax couple the out-of-room parameters into the read "
            "path, so a transport-literal state can fail to resurrect "
            "under subtraction; the MIXED/INCONCLUSIVE branch exists to "
            "hold exactly this",
        "the_converse_arm": "arm B inverts the predictions (transport-"
            "literal: stays dead; overwrite: recovers) — co-interpreted "
            "as the dichotomy's other half, never adjudicated (no frozen "
            "bar on it)",
        "loads_measured_not_nominal": "every mass is a direct fp64 "
            "projection of the loaded states (the cited t100-300 products "
            "excepted — e283's own committed reads, cross-checked at t400)",
        "n_and_scope": "n=1 lineage, one session, one draw of history; "
            "nothing guaranteed",
    }
    write_partial("P7 honesty block")

    # ---- the figure -----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2))
    ts = [0, 100, 200, 300, 400]
    mass = [m0] + [r["in_room_mass_cited"] for r in cited_grid]
    outm = [o0] + [r["out_room_mass_cited"] for r in cited_grid]
    ax1.plot(ts, mass, "o-", color="#1a6faf", lw=2, label=
             "in-room mass ||P(θ−base)|| (the write's room-mass)")
    ax1.plot(ts, outm, "s--", color="#c2500d", lw=2, label=
             "out-of-room residual ||(I−P)(θ−base)||")
    ax1.axhline(m0 * EROSION_BAR, color="gray", ls=":", lw=1.5)
    ax1.text(402, m0 * EROSION_BAR, f" {EROSION_BAR:.0%} of t0",
             fontsize=8, color="gray", va="center")
    for t, st in zip(ts, ["COMPUTED", "cited", "cited", "cited",
                          "COMPUTED"]):
        ax1.annotate(st, (t, mass[ts.index(t)]), textcoords="offset points",
                     xytext=(0, -14), fontsize=7, color="#555555",
                     ha="center")
    ax1.set_xlabel("corpus steps t (t0/t400 computed from the states on "
                   "disk; t100–300 cited-derived from e283)")
    ax1.set_ylabel("mass (fp64 norm)")
    ax1.set_title("(a) THE WRITE-MASS LEDGER")
    ax1.legend(fontsize=8, loc="center right")
    ax1.set_ylim(0, max(outm) * 1.12)
    ax1.set_xlim(-15, 470)
    ax1.grid(alpha=0.25)

    rows = [("the fact (t0)", fact_cells["g0"], "#2e8b57"),
            ("t400 post (dead)", post_cells["g0"], "#555555"),
            ("ARM B: − in-room drift\n(the converse control)",
             cells_B["g0"], "#c2500d"),
            ("ARM A: − out-of-room drift\n(THE intervention)",
             g0A, "#1a6faf")]
    ypos = list(range(len(rows)))[::-1]
    for y, (lab, val, c) in zip(ypos, rows):
        ax2.barh(y, max(val, 1e-9), color=c, alpha=0.85)
        ax2.text(max(val, 1e-9) * 1.35, y, f" {val:.3e}", va="center",
                 fontsize=9, color=c)
    ax2.set_yticks(ypos, [r[0] for r in rows], fontsize=8)
    ax2.axvline(RESURRECT_BAR, color="#2e8b57", ls="--", lw=1.5)
    ax2.text(RESURRECT_BAR * 1.1, len(rows) - 0.45,
             f"TRANSPORT bar {RESURRECT_BAR}", fontsize=8, color="#2e8b57")
    ax2.axvline(DEAD_BAR, color="#8b2e2e", ls=":", lw=1.5)
    ax2.text(DEAD_BAR * 1.15, -0.42, f"dead bar {DEAD_BAR}", fontsize=8,
             color="#8b2e2e")
    ax2.axvline(FACT_BASELINE_G0, color="#bbbbbb", ls="-", lw=1)
    ax2.set_xscale("log")
    ax2.set_xlim(1e-7, 1.2)
    ax2.set_xlabel("the g0 probe read (log scale)")
    ax2.set_title("(b) THE SUBTRACTION INTERVENTION")
    ax2.grid(alpha=0.25, axis="x")
    fig.suptitle(f"X14 — THE TRANSPORT INTERVENTION — {verdict}",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, 0.005, "the subtraction assumes component-wise "
             "additivity; no resurrection is inconclusive between "
             "overwrite and coupling (necessary-not-sufficient)",
             ha="center", fontsize=7.5, color="#666666")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(RD / "x14_transport.png", dpi=150)
    plt.close(fig)
    log(f"[fig] wrote {RD / 'x14_transport.png'}")

    # ---- the report -----------------------------------------------------
    clause_wrapped = " ".join(clause.split())
    birth = HEAD0 or git_head()
    rep = f"""# X14 — THE TRANSPORT INTERVENTION — REPORT

**Verdict: {verdict}** (the frozen bars, letter-exact). {clause_wrapped}

R67's named desk cell: e283's correlational dichotomy — TRANSPORT-LITERAL
(the write's in-room mass intact, killed by out-of-room context) vs
OVERWRITE (the in-room mass eroded) — both fit every committed number
(the critic's finding-3 arithmetic: the intact-write band [7.0, 10.3] vs
measured 7.31; the erosion degeneracy: the same 7.31 is consistent with
any surviving in-room fraction in [0.65, 1.03]). The states are on disk;
this cell intervenes.

## (a) The write-mass ledger (fp64, the fact's own K10K room)

| t | in-room mass P(theta-base) | out-of-room (I-P)(theta-base) | source |
|---|---|---|---|
| t0 | **{m0:.4f}** | {o0:.4f} | COMPUTED (the fact ckpt) |
| 100 | {cited_grid[0]['in_room_mass_cited']:.4f} | {cited_grid[0]['out_room_mass_cited']:.4f} | cited-derived (e283's committed ledger; state not saved) |
| 200 | {cited_grid[1]['in_room_mass_cited']:.4f} | {cited_grid[1]['out_room_mass_cited']:.4f} | cited-derived |
| 300 | {cited_grid[2]['in_room_mass_cited']:.4f} | {cited_grid[2]['out_room_mass_cited']:.4f} | cited-derived |
| 400 | **{m400:.4f}** | **{o400:.4f}** | COMPUTED (the post ckpt) |

The mass ratio t400/t0 = **{mass_ratio:.4f}** — {'ABOVE' if mass_ratio >= EROSION_BAR else 'BELOW'} the
{EROSION_BAR:.0%} erosion bar. The out-of-room residual grew {o0:.2f} -> {o400:.2f}
(the transport channel). Diagnostics (context, not bars): the surviving
mass's direction vs the write's own in-room direction cos =
{cos_mass:.4f}; the erosion degeneracy [{deg_lo / m0:.3f}, {deg_hi / m0:.3f}]
reconstructs R67's finding 3 exactly.

## (b) The subtraction intervention (fp64; arms materialized fp32)

drift decomposition: |delta| {dn:.4f} = in-room {pn:.4f} + out-of-room {on:.4f}
(orthogonal by construction; identity closure |theta400 - delta - theta0|
= {id_err:.1e} in fp64).

| arm | construction | g0 read | prediction it answers |
|---|---|---|---|
| the fact (t0) | the loaded write | {fact_cells['g0']:.8f} | the resurrection target (baseline 0.2646) |
| t400 post | the dead state | {post_cells['g0']:.3e} | the death being diagnosed |
| **ARM A (primary)** | theta400 - (I-P)(drift) | **{g0A:.6e}** | TRANSPORT-LITERAL: >= {RESURRECT_BAR}; OVERWRITTEN: < {DEAD_BAR} |
| ARM B (control) | theta400 - P(drift) | {cells_B['g0']:.6e} | the converse (co-interpreted, no bar) |
| ARM ID (closure) | theta400 - delta | {cells_ID['g0']:.8f} | must equal the fact's read (abs diff {id_closure_read:.1e}) |

## (c) The honesty block — the caveat's status

**Coupling caveat fired: {coupling_caveat_fired}.** The subtraction
assumes component-wise additivity (a linear-superposition null); LN and
softmax couple out-of-room parameters into the read path, so a
transport-literal state can fail to resurrect. The test is
NECESSARY-not-SUFFICIENT: resurrection => transport; no resurrection =>
inconclusive between overwrite and coupling, NOT proof of overwrite.
{'This run: the mass is intact and the read did not resurrect — the '
'registered inconclusive branch holds the result by construction, not '
'by consolation.' if coupling_caveat_fired else
'This run: the branch did not fire; see the verdict above.'}

## The gates (all 12 PASS)

Parents md5/literal-bound (e264/e268/e278 metrics, the fact ckpt
md5/size/step + flat-md5, the room file, the span, the v-map); the base
fact-free; the root read-bound; the room certified (idem/kept2/span) and
BIT-BOUND to e264_rooms.pt; the fact behaviorally bound (g0 abs diff
{abs(fact_cells['g0'] - FACT_BASELINE_G0):.1e}); the post state
value-bound behaviorally + geometrically (drift norm abs diff
{abs(dn - E283_DRIFT_NORM_T400):.1e}; read g0 abs diff
{abs(post_cells['g0'] - E283_POST_G0):.1e}).

## Disclosures

- THE MILESTONE GRID: only t0 + t400 were checkpointed by e283; t100-300
  are cited-derived products of its committed fp64 ledger
  (cross-checked at t400: abs diff {xchk_m:.1e}); the t400 intervention
  answers the primary (the dispatch's own instruction).
- CPU-ONLY desk cell: no training, no stream, no steps; reads + fp64
  projections; torch threads 4, pocketfft workers 2; NO envelope-log
  writes.
- The post state has no committed file-md5 (e283 recorded none) — bound
  by VALUE (behavioral + geometric), the stronger bind; its md5 is
  recorded here ({G_PARENTS['the_post_state']['md5']}).
- n=1 lineage, one session; nothing guaranteed.

## Provenance

Birth commit {birth} (bars + conventions, BEFORE any compute); full run
this commit. Machinery: e261's SRCT/LadderRooms ported whole by import
(the committed file untouched). No NOTES/THINKING/QUEUE/STATE edits.
"""
    (RD / "REPORT.md").write_text(rep, encoding="utf-8")
    log(f"[report] wrote {RD / 'REPORT.md'}")

    metrics["provenance"] = {
        "git_head_at_start": git_head(),
        "script": str(Path(__file__).resolve()),
        "machinery_imported_from": str(
            (Path(__file__).resolve().parent / "e261_rank_ladder.py")),
        "checkpoints": {
            "base": "runs/checkpoints/e001.pt",
            "reference_root": f"runs/checkpoints/{ROOT_CK}",
            "the_fact_t0": f"runs/checkpoints/{FACT_CK} (md5 {FACT_MD5}, "
                           f"flat {FACT_FLAT_MD5})",
            "the_post_t400": f"runs/checkpoints/{POST_CK} (md5 "
                             f"{G_PARENTS['the_post_state']['md5']})",
            "span_ledger": f"runs/checkpoints/{SPAN_CK}",
            "vmap": f"runs/checkpoints/{VMAP_CK}",
            "e264_rooms": f"runs/checkpoints/{ROOMS264_CK}",
        },
        "eval": {"device": "cpu fp32 probes / cpu fp64 dense projections",
                 "torch_threads": 4, "pocketfft_workers": E261.DCT_WORKERS,
                 "cuda_tensors_created": 0},
        "versions": {"torch": torch.__version__,
                     "numpy": np.__version__,
                     "scipy": __import__("scipy").__version__,
                     "matplotlib": matplotlib.__version__},
    }
    metrics["status"] = ("COMPLETE — adjudicated (t400-only intervention + "
                         "the disclosed cited grid)")
    metrics["date"] = now_iso()
    metrics["phase_note"] = ("P9 DONE (ledger + intervention + adjudication "
                             "+ honesty + figure + report)")
    metrics["outputs"] = [str(RD / "metrics.json"),
                          str(RD / "x14_transport.png"),
                          str(RD / "REPORT.md")]
    save_json(RD / "metrics.json", E43.jsonable(metrics))
    log("X14 COMPLETE — metrics + figure + report written")


if __name__ == "__main__":
    HEAD0 = git_head()
    main()
