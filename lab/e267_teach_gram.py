"""E267 — THE TEACH-GRAM CENSUS (T243's registered decisive cell: the
never-cached distribution measured — the Jacobian unification's third and
sharpest chance).

Design dispatched 2026-10-05; this docstring carries the registered bars
VERBATIM, committed at birth BEFORE any compute. Adjudicate against exactly
this; no bar shopping.

THE QUESTION (verbatim from the dispatch): does the TEACH STREAM's
instantaneous Fisher spectrum kneel at ~10k — the located expression cliff?
This is the distribution the cliff was actually measured on, never
Gram-cached until now (e266's inventory: "THE TEACH STREAM'S GRADIENTS WERE
NEVER CACHED — the one distribution the cliff was actually measured on has
no Gram anywhere in the record").

THE CELL (CPU, the gradients of the teach stream at the g1c root):
  (1) the teach stream's forward/backward WITHOUT optimizer steps — the
      loss-gradient at each of the ladder's registered teach batches (the
      same stream/dose the installs used: e001 + the Dmix splicing at gen
      24314, steps 1..N of the committed s400 stream, batch 64 = 16
      install + 16 paired-anchor + 32 random corpus windows at BLOCK 256;
      N = 80 gradient samples — the dispatch's 20-80 upper end, stated);
      the model state = the pristine g1c ROOT (bit-gated);
  (2) the N-step Gram in fp64 -> the spectrum via e265's spectrum_stats
      (module-import, e266's convention): eigenvalues, decay, erank, the
      knee at the logged-gap rule;
  (3) THE JOIN: the knee/erank vs the located cliff 10k (the factor-3 band
      [3,333, 30,000]);
  (4) the co-reads: the teach-Gram's top eigenspace vs the corpus wash-span
      (the principal angles — the two streams' geometry), and the norm-free
      direction spectrum (the step-1 transient check per e265's
      convention).

REGISTERED BARS (frozen BEFORE compute, VERBATIM from the dispatch):
  - TEACH-KNEELS-AT-CLIFF — "the teach-Gram's knee or erank lands inside
    [3,333, 30,000] — THE UNIFICATION LANDS ON THE RIGHT OBJECT: the
    expression cliff is the teach stream's instantaneous Fisher's effective
    rank; the network stores what its writing distribution cannot
    normalize away"
  - TEACH-FLAT — "the teach top is as flat as the wash's (knee <= 2; erank
    at the window's edge) — the cliff's carrier is NON-SPECTRAL: the
    room-optimizer interface, a dynamical object no eigenstructure sees;
    the honest bound, the third chance closed"
  - MIXED — "the tables verbatim, the resolution disclosed"

OPERATIONALIZATIONS (frozen HERE before compute; they fix the clauses, they
do not move the bars):
  * THE TEACH STREAM := the g1c pipeline's install stream VERBATIM
    (e246/e258's chunked_install arithmetic, optimizer removed): fresh
    torch.Generator seeded 24314; per step ix = randint(60, (16,)),
    aj = randint(60, (16,)), rj = randint(len(train)-255, (32,)); batch =
    16 install windows (name-masked union CE) + 16 paired originals + 32
    random corpus windows; clip_grad_norm_ 1.0 after backward. Steps 1..80
    are exactly the batches the installs drank at their steps 1..80.
  * THE GRADIENT SAMPLE := the POST-CLIP gradient (e240's primary — "what
    Adam drank"; the installs' hook order backward -> clip 1.0 -> step).
    The pre-clip Gram is recovered exactly by rescaling (clip is a per-step
    scalar: g_post = c_t g_pre) and co-reported; the norm-free Gram (unit
    rows) is direction-identical pre/post clip.
  * THE MODEL STATE := runs/checkpoints/g1c_root.pt (md5 + flat-weight md5
    + battery bit-gates vs the committed root cells). The root, not the
    install trajectory: the registration reads the stream's instantaneous
    Fisher AT the root (the organism that holds the fact).
  * THE KNEE := e265's logged-gap rule (argmax_i log10(lam_{i-1}/lam_i)).
  * "erank at the window's edge" := erank(1e-2).k >= ceil(0.95 N) or
    window-capped (the wash's committed edge facts: erank(1e-2) 76-77 of
    80 = 0.95-0.9625).
  * "as flat as the wash's" := knee <= 2 in the RAW flavor AND knee <= 2 in
    the NORM-FREE flavor AND raw erank(1e-2) at the window's edge (the
    wash's committed facts: raw knees 1, norm-free knees 1, raw
    erank(1e-2) 76-77/80).
  * VERDICT RULE := gates fail -> TEXTURE (nothing adjudicated);
    TEACH-KNEELS-AT-CLIFF iff any of {knee, erank(tol) for tol in
    e265's MASS_TOL} of the PRIMARY (post-clip) teach-Gram lands in
    [10000/3, 30000]; else TEACH-FLAT iff the flatness clause above holds;
    else MIXED.
  * THE SPECTRUM-ESTIMATE CAVEAT (the dispatch's registered check,
    disclosed prominently): N = 80 gradient samples give AT MOST 80
    nonzero eigenvalues of a 2,739,072-dimensional operator; every erank
    <= 80 by construction and the band [3,333, 30,000] lies 42-375x
    beyond the window — the Gram can only measure the TOP of the teach
    Fisher and the SHAPE of its decay. The full-resolution teach-side
    co-report (the DIAGONAL second-moment at 2.74M coords, e266's v-map
    census mirrored on the teach distribution) is reported VERBATIM but
    CANNOT fire the bars: the bars' letter is the Gram's.
  * THE SPAN CO-READ := e246's committed LATE span (e246_late_span.pt,
    rank 10, hard-bound md5 + sv): principal angles between the teach
    Gram's top-10 eigenspace and the span; per-step in-span fractions vs
    the random floor sqrt(10/2,739,072) = 0.0019107.

CHECKS (the dispatch's, in force): the teach stream's provenance (the exact
batches the installs used — the registered gen/splicing; the model state =
the pristine g1c root, bit-gated); the spectrum-estimate caveat (the sample
count vs the 2.74M dims); fp64; nothing guaranteed.

COMPUTE ENVELOPE: CPU-ONLY (torch threads 4; e263 owns the GPU — never
touched; CUDA_VISIBLE_DEVICES=-1 forced before torch import). 2,739,072
params (inside the <=100M free tier). The replay: 80 forward/backward passes
at batch 64 x 255 — no optimizer, no training. fp64 algebra chunked
(CHUNK = 700,000 coords; the (80, N) fp32 gradient cache ~ 877 MB).

Outputs: runs/e267/{metrics.json (PROGRESSIVE), e267_teach_spectrum.png,
e267_cliff_join.png, e267_two_stream_angles.png, run.log}. No
NOTES/THINKING/QUEUE/STATE edits (the coordinator folds). Commit + push per
phase.

Run:  cd lab && python e267_teach_gram.py    (E267_SMOKE=1 shakedown)
"""
from __future__ import annotations

import os

# CPU-ONLY, forced BEFORE torch/numpy import (e263 owns the GPU lane)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import hashlib                                            # noqa: E402
import json                                               # noqa: E402
import math                                               # noqa: E402
import random                                             # noqa: E402
import sys                                                # noqa: E402
import time                                               # noqa: E402
from datetime import datetime, timezone                   # noqa: E402
from pathlib import Path                                  # noqa: E402

import matplotlib                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                           # noqa: E402
import numpy as np                                        # noqa: E402
import torch                                              # noqa: E402
import torch.nn.functional as F                           # noqa: E402

try:                                            # Windows console safety
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                               # noqa: BLE001
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common                                              # noqa: E402
from common import CharCorpus, run_dir, save_json, set_seed  # noqa: E402

import e043_install as E43                                 # noqa: E402 (REPO,
                                                           # find_occ,
                                                           # SPLICE_RNG,
                                                           # CORP_BS,
                                                           # MIX_RANDOM,
                                                           # jsonable)
import g1b_continuity as GB                                # noqa: E402 — MUST
                                                           # precede G1 (the
                                                           # 2.74M patch)
import g1_anchored_ball as G1                              # noqa: E402 — the
                                                           # machinery
import e265_fisher_census as E265                          # noqa: E402 — the
                                                           # census machinery

spectrum_stats = E265.spectrum_stats                       # module-import
                                                           # convention
                                                           # (e266's)

torch.set_num_threads(4)            # shared machine (the CPU lane is shared)

SMOKE = os.environ.get("E267_SMOKE") == "1"
CPU = torch.device("cpu")
NAME = "e267_smoke" if SMOKE else "e267"
assert not torch.cuda.is_available(), \
    "e267 is CPU-ONLY (e263 owns the GPU lane — dispatch)"

T0 = time.time()
RD = run_dir(NAME)
LOG_PATH = RD / "run.log"
_logf = open(LOG_PATH, "a", encoding="utf-8")


def log(m: str) -> None:
    line = f"[{time.time() - T0:7.1f}s] {m}"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


# ======================================================================
# THE CONFIG (frozen)
# ======================================================================
ROOT_CK = "g1c_root.pt"              # the committed fresh root (THE state)
BASE_CK = "e001.pt"                  # the 2.74M corpus base (stream anchor)
SPAN_CK = "e246_late_span.pt"        # e246's committed LATE span (co-read)
CKPT_DIR = GB.CKPT_DIR
FRESH_GEN = 24314                    # the g1c fresh draw's install gen
N_SAMPLES = 4 if SMOKE else 80       # the dispatch's 20-80 range, upper end
CHUNK = 700_000                      # fp64 chunk for the big dots
CLIFF_K = 10_000                     # runs/e264 floor_cross_k (SHARP-THRESH)
BAND = (CLIFF_K / 3.0, 3.0 * CLIFF_K)        # [3,333.33, 30,000]
EDGE_FRAC = 0.95                     # "erank at the window's edge"
TOPK_ANGLES = 10                     # the span's rank; eigenspace co-read

# the committed records, HARD-BOUND (read at runtime from their paths and
# asserted against these literals; Rule 12)
G1C_METRICS = E43.REPO / "runs" / "g1c_root" / "metrics.json"
G1C_ROOT_GM12 = 0.9026340246200562
G1C_ROOT_G0 = 0.7447534203529358
G1C_INSTALL_STEP1_CE = 1.0701713562011719    # the install traj's s1 batch CE
G1C_ROOT_MD5 = "9c7d4ca1b60c8a1158d080f932e2c95f"     # e266 provenance
E001_MD5 = "d114536d1c0983ab3be67f67ff0667c8"         # e266 provenance
E246_ROOT_FLAT_MD5 = "a7f02b367c5342535aecfc814d780631"   # e246 G_ROOTSPAN
E246_SPAN_MD5 = "3464ff6081d8e402103dd9ca65bfb53a"
E246_SPAN_RANK = 10
E246_SPAN_SV = [1.2198079109057542, 0.6374444530820925,
                0.3708103482336049, 0.2553756821823229,
                0.19979457921194507, 0.15285727407919378,
                0.13088814951740852, 0.11158447680789642,
                0.10221680786835272, 0.08960906206759227]
E231_PR = 4.117363094870959                   # e266 G_IMPORT's anchor
E266_METRICS = E43.REPO / "runs" / "e266" / "metrics.json"   # v-map curve
E260_FREE_IN_SPAN_MEDIAN = 0.08342791673853514  # install-time teach coupling
RANDOM_NORM_FLOOR = math.sqrt(E246_SPAN_RANK / GB.G1B_PARAMS)  # 0.0019107
MASS_TOLS = ("0.01", "0.001", "1e-06", "1e-10")  # e265's MASS_TOL keys

# the wash's committed flatness facts (e265/e266; the TEACH-FLAT clause's
# reference)
WASH_FLAT_FACTS = {
    "raw_knees": {"e231J1_20": 2, "e246late_10": 1, "e234_w1_80": 1,
                  "e234_w2_80": 1, "e234_w3_80": 1},
    "normfree_knees_124M": {"w1": 1, "w2": 1, "w3": 1},
    "raw_erank1e2_over_window": {"e234_w1": "76/80", "e234_w2": "76/80",
                                 "e234_w3": "77/80"},
    "source": "runs/e265 + runs/e266 metrics (re-verified by e266's gates)",
}

REGISTERED = {
    "question_verbatim": ("does the TEACH STREAM's instantaneous Fisher "
                          "spectrum kneel at ~10k — the located expression "
                          "cliff? This is the distribution the cliff was "
                          "actually measured on, never Gram-cached until "
                          "now."),
    "bars_verbatim": {
        "TEACH-KNEELS-AT-CLIFF": ("the teach-Gram's knee or erank lands "
                                  "inside [3,333, 30,000] — THE UNIFICATION "
                                  "LANDS ON THE RIGHT OBJECT: the expression "
                                  "cliff is the teach stream's instantaneous "
                                  "Fisher's effective rank; the network "
                                  "stores what its writing distribution "
                                  "cannot normalize away"),
        "TEACH-FLAT": ("the teach top is as flat as the wash's (knee <= 2; "
                       "erank at the window's edge) — the cliff's carrier "
                       "is NON-SPECTRAL: the room-optimizer interface, a "
                       "dynamical object no eigenstructure sees; the honest "
                       "bound, the third chance closed"),
        "MIXED": "the tables verbatim, the resolution disclosed",
    },
    "operationalization": (
        "frozen BEFORE compute: N=80 post-clip teach gradients (the installs'"
        " own batches, gen 24314 steps 1..80) at the bit-gated g1c root; the "
        "80x80 Gram in fp64 (chunked); spectrum via e265's spectrum_stats "
        "(module-import, e266's convention); knee := the logged-gap rule; "
        "TEACH-KNEELS-AT-CLIFF iff knee or any erank(tol) in [10000/3, "
        "30000]; TEACH-FLAT iff raw knee <= 2 AND norm-free knee <= 2 AND "
        "raw erank(1e-2) >= ceil(0.95 N) or window-capped; MIXED otherwise; "
        "gates fail -> TEXTURE. The pre-clip Gram (scalar-rescaled "
        "recovery) and the teach-side DIAGONAL second-moment census "
        "(2.74M coords) are co-reports and CANNOT fire the bars."),
    "registration": ("bars + question frozen VERBATIM from the e267 dispatch "
                     "(T243's registered decisive cell); this script "
                     "committed at birth BEFORE any compute; adjudicate "
                     "against exactly this; no bar shopping."),
}

deviations: list[str] = [
    "N = 80 (the dispatch's 20-80 upper end; the widest window the "
    "registered range allows — stated per the dispatch's own instruction).",
    "The gradient sample is the POST-CLIP gradient (e240's primary — what "
    "Adam drank); the pre-clip Gram is recovered exactly by per-step scalar "
    "rescaling (clip_grad_norm_ is a global scalar) and co-reported; the "
    "norm-free Gram is invariant to the clip by construction.",
    "The model state is the g1c ROOT, not the install trajectory (the "
    "registration's letter: the stream's instantaneous Fisher AT the root "
    "— the organism that holds the fact); the actual installs ran this "
    "stream from the e001 BASE while moving. Disclosed, not a bug: the "
    "wash-side references (e231/e246 histories, e234 Grams) were sampled "
    "the same way — gradients of a stream at a FIXED committed state.",
    "No optimizer, no masking, no walls: the FREE stream verbatim (the "
    "e246/e258 arms' masks were interventions; the natural teach stream is "
    "FREE — the ladder's fixed stream).",
    "The teach-side DIAGONAL second-moment census (full 2.74M resolution) "
    "is added as a CO-REPORT mirroring e266's v-map census: it is the only "
    "teach-side object in which k = 10k is resolvable, but the bars' letter "
    "is the Gram's — it cannot fire or move the verdict.",
    "One extra forward at the e001 BASE (G_STREAM's stream-identity "
    "certificate): the committed g1c install traj's step-1 batch CE is the "
    "only committed install-stream anchor; cross-device texture tol 5e-3 "
    "(the original install ran on GPU; this cell is CPU).",
    "CPU-ONLY (e263 owns the GPU): CUDA_VISIBLE_DEVICES=-1 forced before "
    "torch import; no cuda call anywhere; all dots fp64 chunked.",
    "Smoke mode (E267_SMOKE=1): N=4, own smoke dir; NOTHING adjudicated or "
    "gated (SMOKE stamp on every read).",
    "No NOTES/THINKING/QUEUE/STATE edits (dispatch; the coordinator folds).",
]


def md5of(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def flat_params_cpu(net) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).cpu()
                      for p in net.parameters()]).clone()


def write_partial(metrics: dict, tag: str) -> None:
    metrics["status"] = tag
    save_json(RD / "metrics.json", E43.jsonable(metrics))
    log(f"metrics written ({tag})")


# ======================================================================
def main() -> int:
    metrics: dict = {
        "experiment": "e267_teach_gram",
        "phase": ("THE TEACH-GRAM CENSUS — the never-cached distribution "
                  "measured: 80 post-clip teach gradients of the gen-24314 "
                  "Dmix stream at the bit-gated g1c root -> the fp64 Gram "
                  "-> e265's spectrum_stats -> the knee/erank join vs the "
                  "LOCATED cliff ~10k + the two-stream principal angles + "
                  "the norm-free direction spectrum + the teach-side "
                  "diagonal census (co-report)"),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "smoke": SMOKE,
        "registration": REGISTERED["registration"],
        "bars_verbatim": REGISTERED["bars_verbatim"],
        "question_verbatim": REGISTERED["question_verbatim"],
        "operationalization": REGISTERED["operationalization"],
        "spectrum_estimate_caveat": (
            f"N = {N_SAMPLES} gradient samples give AT MOST {N_SAMPLES} "
            "nonzero eigenvalues of a 2,739,072-dimensional operator (the "
            "rest of the sample sum is exactly zero). Every erank <= "
            f"{N_SAMPLES} by construction; the factor-3 cliff band "
            f"[{BAND[0]:.1f}, {BAND[1]:.0f}] lies "
            f"{BAND[0] / N_SAMPLES:.0f}-{BAND[1] / N_SAMPLES:.0f}x beyond "
            "the window. The Gram measures the TOP of the teach Fisher and "
            "the SHAPE of its decay — NOT the tail's extent. The "
            "full-resolution teach-side co-report (the DIAGONAL "
            "second-moment at 2.74M coords) is the only teach object that "
            "can see k = 10k; it is disclosed as a co-report and cannot "
            "fire the bars (the bars' letter is the Gram's)."),
        "compute": {"device": "CPU desk (torch threads 4; CUDA off at "
                              "import; e263 owns the GPU — untouched)",
                    "gpu_touched": False},
        "deviations": deviations,
        "gates": {},
    }
    log(f"E267 — THE TEACH-GRAM CENSUS (smoke={SMOKE}) -> {RD}")
    log(f"N={N_SAMPLES} teach gradient samples; cliff k={CLIFF_K}; band "
        f"[{BAND[0]:.1f}, {BAND[1]:.0f}]")
    write_partial(metrics, "startup (bars registered, committed at birth)")
    set_seed(26701)                  # global init only; every RNG is its own

    # ================= P0: the protocol rebuild (g1c's gates VERBATIM) ==
    corpus = CharCorpus(E43.REPO / "data" / "input.txt", seed=1337)
    stoi, itos = corpus.stoi, corpus.itos
    zid = stoi["Z"]
    train_ids, val_ids = corpus.train, corpus.val
    train_text = "".join(itos[int(i)] for i in train_ids)
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
    install_occ, held_occ = host_occ[:60], host_occ[60:90]
    mix = {"FLORIZEL": sum(1 for _, h in install_occ if h == "FLORIZEL"),
           "ELIZABETH": sum(1 for _, h in install_occ if h == "ELIZABETH")}
    G_SPLICE = {"install_mix": mix, "pass": bool(mix == {"FLORIZEL": 19,
                                                         "ELIZABETH": 41})}
    assert G_SPLICE["pass"], f"splice drift {mix}"
    name_ids = corpus.encode(G1.NAME)

    # batteries (e119/e176n verbatim): install-60 at {-12,0}
    bat_ids = {}
    for j in (-12, 0):
        cs = [train_text[p - G1.PRE - j: p] for p, _ in install_occ]
        bat_ids[j] = torch.stack([corpus.encode(c) for c in cs])
    gm12_ids, g0_ids = bat_ids[-12], bat_ids[0]

    # install windows + original-host anchor bank (g1's phase-0 construction)
    def build_win(p, host):
        return torch.cat([train_ids[p - G1.PRE: p], name_ids,
                          train_ids[p + len(host):
                                    p + len(host) + G1.POST_CAP]])

    win_i = torch.stack([build_win(p, h) for p, h in install_occ])
    inst_x = win_i.clone()                                  # (60, 256)
    anchor_full = torch.stack([train_ids[p - G1.PRE: p - G1.PRE + G1.BLOCK]
                               for p, _ in install_occ])    # (60, 256)
    inst_mask = torch.zeros(60, G1.BLOCK - 1, dtype=torch.bool)
    inst_mask[:, G1.PRE - 1: G1.PRE - 1 + len(G1.NAME)] = True
    G_INSTMASK = {"name_positions": int(inst_mask.sum()),
                  "expected": 60 * len(G1.NAME),
                  "pass": bool(int(inst_mask.sum()) == 60 * len(G1.NAME))}
    assert G_INSTMASK["pass"], f"install mask gate FAILED: {G_INSTMASK}"
    metrics["gates"].update({"G_NAMEFREE": G_NAMEFREE, "G_SPLICE": G_SPLICE,
                             "G_INSTMASK": G_INSTMASK})
    log("P0: protocol gates PASS (namefree / splice 19+41 / install mask)")
    write_partial(metrics, "P0 protocol gates PASSED")

    # ================= P0b: the parents hard-bound (Rule 12) ============
    g1c = json.loads(G1C_METRICS.read_text(encoding="utf-8"))
    root_cells = g1c["root_build"]["root_cells"]
    inst_traj = g1c["root_build"]["install"]["traj"]
    root_md5 = md5of(CKPT_DIR / ROOT_CK)
    base_md5 = md5of(CKPT_DIR / BASE_CK)
    G_PARENTS = {
        "g1c_metrics": {"path": str(G1C_METRICS), "md5": md5of(G1C_METRICS),
                        "verdict": g1c["adjudication"]["verdict"]},
        "ckpt_g1c_root": {"file": f"runs/checkpoints/{ROOT_CK}",
                          "md5": root_md5, "hardbound": G1C_ROOT_MD5,
                          "match": bool(root_md5 == G1C_ROOT_MD5)},
        "ckpt_e001_base": {"file": f"runs/checkpoints/{BASE_CK}",
                           "md5": base_md5, "hardbound": E001_MD5,
                           "match": bool(base_md5 == E001_MD5)},
        "root_cells_match": bool(
            abs(root_cells["gm12"] - G1C_ROOT_GM12) < 1e-12
            and abs(root_cells["g0"] - G1C_ROOT_G0) < 1e-12),
        "install_step1_ce_match": bool(
            abs(inst_traj[0]["ce_batch"] - G1C_INSTALL_STEP1_CE) < 1e-12),
        "install_gen": g1c["root_build"]["install"]["gen_seed"],
        "pass": bool(root_md5 == G1C_ROOT_MD5 and base_md5 == E001_MD5
                     and abs(root_cells["gm12"] - G1C_ROOT_GM12) < 1e-12
                     and abs(root_cells["g0"] - G1C_ROOT_G0) < 1e-12
                     and abs(inst_traj[0]["ce_batch"]
                             - G1C_INSTALL_STEP1_CE) < 1e-12
                     and g1c["root_build"]["install"]["gen_seed"]
                     == FRESH_GEN),
    }
    assert G_PARENTS["pass"], f"parent bind failed: {G_PARENTS}"
    metrics["gates"]["G_PARENTS"] = G_PARENTS
    log(f"P0b: G_PARENTS PASS — g1c {G_PARENTS['g1c_metrics']['verdict']}; "
        f"root/base md5s match; gen {FRESH_GEN}; step-1 CE "
        f"{G1C_INSTALL_STEP1_CE:.10f}")
    write_partial(metrics, "P0b parents hard-bound")

    # ================= P0c: e265's spectrum_stats import gate ===========
    e231 = json.loads((E43.REPO / "runs" / "e231" / "metrics.json")
                      .read_text(encoding="utf-8"))
    sv231 = np.array(e231["cells"]["J1"]["primary_stream"]["span"]["sv"],
                     dtype=np.float64)
    pr_imp = spectrum_stats(sv231 ** 2)["participation_ratio"]
    G_IMPORT = {
        "check": "module-import identity: PR(e231 sv^2) via e265's "
                 "spectrum_stats vs the committed 4.117363094870959 "
                 "(e266's G_IMPORT convention)",
        "recomputed": pr_imp, "committed": E231_PR,
        "dev": abs(pr_imp - E231_PR), "tol": 1e-6,
        "pass": bool(abs(pr_imp - E231_PR) < 1e-6),
    }
    assert G_IMPORT["pass"], f"G_IMPORT FAILED: {G_IMPORT}"
    metrics["gates"]["G_IMPORT"] = G_IMPORT
    log(f"P0c: G_IMPORT PASS — PR(e231 sv^2) {pr_imp:.15f} "
        f"(dev {abs(pr_imp - E231_PR):.1e})")
    write_partial(metrics, "P0c spectrum_stats import gated")

    # ================= P1: THE MODEL, bit-gated =========================
    root_net = G1.load_g1(CKPT_DIR / ROOT_CK)
    n_par = root_net.num_params()
    theta_root = flat_params_cpu(root_net)
    flat_md5 = hashlib.md5(theta_root.numpy().tobytes()).hexdigest()
    root_gm12_read = G1.battery_cell(root_net, gm12_ids, zid)["mean_pz"]
    root_g0_read = G1.battery_cell(root_net, g0_ids, zid)["mean_pz"]
    G_MODEL = {
        "checkpoint": f"runs/checkpoints/{ROOT_CK}",
        "file_md5": root_md5,
        "n_params": n_par, "expected_params": GB.G1B_PARAMS,
        "flat_md5": flat_md5, "flat_md5_committed_e246": E246_ROOT_FLAT_MD5,
        "bit_gate": bool(flat_md5 == E246_ROOT_FLAT_MD5),
        "battery_gm12": root_gm12_read, "battery_gm12_committed":
            G1C_ROOT_GM12, "battery_gm12_abs_diff":
            abs(root_gm12_read - G1C_ROOT_GM12),
        "battery_g0": root_g0_read, "battery_g0_committed": G1C_ROOT_G0,
        "battery_g0_abs_diff": abs(root_g0_read - G1C_ROOT_G0),
        "battery_tol": 1e-9,
        "net_anchored": bool(getattr(root_net, "anchored", False)),
        "pass": bool(n_par == GB.G1B_PARAMS
                     and flat_md5 == E246_ROOT_FLAT_MD5
                     and abs(root_gm12_read - G1C_ROOT_GM12) < 1e-9
                     and abs(root_g0_read - G1C_ROOT_G0) < 1e-9),
        "note": "the pristine g1c root, BIT-gated: e246's committed flat "
                "weight md5 + the committed battery cells (both CPU reads "
                "in the original run — expected exact)",
    }
    if not SMOKE:
        assert G_MODEL["pass"], f"root gate FAILED: {G_MODEL}"
    metrics["gates"]["G_MODEL"] = G_MODEL
    log(f"P1 G_MODEL: {ROOT_CK} — {n_par} params; flat md5 "
        f"{'BIT-MATCH' if G_MODEL['bit_gate'] else 'DRIFT'}; gm12 "
        f"{root_gm12_read:.10f} (|d| {G_MODEL['battery_gm12_abs_diff']:.1e})"
        f"; g0 {root_g0_read:.10f} "
        f"(|d| {G_MODEL['battery_g0_abs_diff']:.1e}); anchored="
        f"{G_MODEL['net_anchored']}: "
        f"{'PASS' if G_MODEL['pass'] else 'FAIL (smoke: recorded only)'}")
    write_partial(metrics, "P1 model bit-gated")

    # ================= P2: THE STREAM, certified ========================
    # the teach-batch recipe (e246/e258's chunked_install inner loop,
    # optimizer removed) — VERBATIM arithmetic
    name_bs, corp_bs, mix_random = (G1.NAME_BS, E43.CORP_BS,
                                    E43.MIX_RANDOM)
    n_inst, n_anc = inst_x.shape[0], anchor_full.shape[0]

    def teach_batch(gen):
        ix = torch.randint(n_inst, (name_bs,), generator=gen)
        aj = torch.randint(n_anc, (corp_bs - mix_random,), generator=gen)
        rj = torch.randint(len(train_ids) - G1.BLOCK - 1, (mix_random,),
                           generator=gen)
        corp = torch.cat([anchor_full[aj],
                          torch.stack([train_ids[s: s + G1.BLOCK]
                                       for s in rj])], 0)
        nw = inst_x[ix]
        x = torch.cat([nw[:, :-1], corp[:, :-1]], 0)
        y = torch.cat([nw[:, 1:], corp[:, 1:]], 0)
        m = torch.zeros(name_bs + corp_bs, x.shape[1], dtype=torch.bool)
        m[:name_bs] = inst_mask[ix]
        return x, y, m

    def teach_loss(net, x, y, m):
        logits, _ = net(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                              y.reshape(-1), reduction="none"
                              ).view(x.shape[0], x.shape[1])
        nm = nll[:name_bs][m[:name_bs]]
        cm = nll[name_bs:]
        return (nm.sum() + cm.sum()) / (nm.numel() + cm.numel()), \
            float(nm.mean()), float(cm.mean())

    # G_STREAM: the step-1 batch certified at the e001 BASE against g1c's
    # committed install traj step-1 CE (the only committed stream anchor)
    base_net = G1.load_g1(CKPT_DIR / BASE_CK)
    base_gm12 = G1.battery_cell(base_net, gm12_ids, zid)["mean_pz"]
    gen_b = torch.Generator().manual_seed(FRESH_GEN)
    xb, yb, mb = teach_batch(gen_b)
    x1_md5_base = hashlib.md5(xb.contiguous().numpy().tobytes()).hexdigest()
    with torch.no_grad():
        ce_base, nm_b, cm_b = teach_loss(base_net, xb, yb, mb)
    del base_net
    G_STREAM = {
        "gen_seed": FRESH_GEN,
        "recipe": "e043-Dmix VERBATIM: ix(16) install + aj(16) paired + "
                  "rj(32) random, masked union CE, BLOCK 256 -> 255 "
                  "positions, batch 64",
        "x1_md5_at_base": x1_md5_base,
        "ce1_at_base": float(ce_base),
        "ce1_committed": G1C_INSTALL_STEP1_CE,
        "ce1_abs_diff": abs(float(ce_base) - G1C_INSTALL_STEP1_CE),
        "tol": 5e-3,
        "base_fact_free_gm12": base_gm12,
        "pass": bool(abs(float(ce_base) - G1C_INSTALL_STEP1_CE) < 5e-3
                     and base_gm12 <= 0.05),
        "note": "the committed g1c install traj's step-1 batch CE is the "
                "stream anchor (cross-device texture tol 5e-3: the "
                "original install ran on GPU, this cell is CPU); the x1 "
                "md5 has no committed reference — recorded for the ledger",
    }
    if not SMOKE:
        assert G_STREAM["pass"], f"stream gate FAILED: {G_STREAM}"
    metrics["gates"]["G_STREAM"] = G_STREAM
    log(f"P2 G_STREAM: step-1 CE at e001 base {float(ce_base):.6f} vs "
        f"committed {G1C_INSTALL_STEP1_CE:.6f} "
        f"(|d| {G_STREAM['ce1_abs_diff']:.1e}; nm {nm_b:.4f} cm {cm_b:.4f};"
        f" base fact-free gm12 {base_gm12:.4f}): "
        f"{'PASS' if G_STREAM['pass'] else 'FAIL (smoke: recorded only)'}")
    write_partial(metrics, "P2 stream certified at the base")

    # ================= P3: THE REPLAY (no optimizer) ====================
    # the LATE span, hard-bound (the co-read's fixed basis)
    span_st = torch.load(CKPT_DIR / SPAN_CK, map_location="cpu",
                         weights_only=False)
    Vp = span_st["Vp"].float()                      # (10, 2739072)
    sv_span = span_st["sv"].double()
    orth_dev = float(torch.max(torch.abs(
        Vp.double() @ Vp.double().T - torch.eye(Vp.shape[0]).double())))
    G_SPAN = {
        "file": f"runs/checkpoints/{SPAN_CK}",
        "md5": md5of(CKPT_DIR / SPAN_CK), "hardbound": E246_SPAN_MD5,
        "shape": list(Vp.shape), "rank": int(Vp.shape[0]),
        "sv_dev_max": float(torch.max(torch.abs(sv_span - torch.tensor(
            E246_SPAN_SV, dtype=torch.float64)))),
        "orthonormality_max_dev": orth_dev,
        "pass": bool(md5of(CKPT_DIR / SPAN_CK) == E246_SPAN_MD5
                     and list(Vp.shape) == [E246_SPAN_RANK, GB.G1B_PARAMS]
                     and torch.max(torch.abs(sv_span - torch.tensor(
                         E246_SPAN_SV, dtype=torch.float64))) < 1e-6
                     and orth_dev < 1e-5),
    }
    if not SMOKE:
        assert G_SPAN["pass"], f"span gate FAILED: {G_SPAN}"
    metrics["gates"]["G_SPAN"] = G_SPAN
    log(f"P3 G_SPAN: md5 {'OK' if G_SPAN['md5'] == E246_SPAN_MD5 else 'DRIFT'"
        }; shape {list(Vp.shape)}; orth dev {orth_dev:.1e}: "
        f"{'PASS' if G_SPAN['pass'] else 'FAIL (smoke: recorded only)'}")

    root_net.eval()
    params = list(root_net.parameters())
    grads = torch.zeros((N_SAMPLES, n_par), dtype=torch.float32)
    rows = []
    gen = torch.Generator().manual_seed(FRESH_GEN)
    x1_md5_root = None
    t_replay = time.time()
    for step in range(1, N_SAMPLES + 1):
        x, y, m = teach_batch(gen)
        if step == 1:
            x1_md5_root = hashlib.md5(
                x.contiguous().numpy().tobytes()).hexdigest()
        loss, nm_mean, cm_mean = teach_loss(root_net, x, y, m)
        root_net.zero_grad(set_to_none=True)
        loss.backward()
        pre_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        post_norm32 = math.sqrt(sum(float((p.grad.detach() ** 2).sum())
                                    for p in params))
        grads[step - 1] = torch.cat([p.grad.detach().reshape(-1)
                                     for p in params]).clone()
        # the per-step span coupling + the fp64 norm (chunked — e246's
        # ledger convention, read-only; the fp64 norm feeds the Gram's
        # diag-identity gate so the gate measures identity, not rounding)
        c_full = torch.zeros(E246_SPAN_RANK, dtype=torch.float64)
        gn2 = 0.0
        for lo in range(0, n_par, CHUNK):
            g64 = grads[step - 1, lo:lo + CHUNK].double()
            c_full += Vp[:, lo:lo + CHUNK].double() @ g64
            gn2 += float(g64 @ g64)
        post_norm = math.sqrt(gn2)
        in_span = math.sqrt(float(c_full @ c_full)) / post_norm
        rows.append({"step": step, "ce_batch": float(loss.item()),
                     "nm_mean": nm_mean, "cm_mean": cm_mean,
                     "gn_pre_clip": pre_norm, "gn_post_clip": post_norm,
                     "gn_post_clip_fp32": post_norm32,
                     "clip_coef": min(1.0, 1.0 / pre_norm) if pre_norm > 0
                     else 1.0,
                     "in_span_frac": in_span})
        log(f"  [teach s{step:3d}] CE {float(loss.item()):.4f} "
            f"(nm {nm_mean:.3f} cm {cm_mean:.3f}) |g|pre {pre_norm:.3f} "
            f"post {post_norm:.4f} in-span {in_span:.4f}")
        if step % 20 == 0:
            write_partial(metrics, f"P3 replay s{step}/{N_SAMPLES}")
    replay_s = time.time() - t_replay
    log(f"P3: replay done — {N_SAMPLES} samples in {replay_s:.1f}s "
        f"({replay_s / N_SAMPLES:.2f}s/step)")
    G_STREAM["x1_md5_at_root"] = x1_md5_root
    G_STREAM["x1_md5_identity_base_root"] = bool(
        x1_md5_root == x1_md5_base)
    G_STREAM["pass"] = bool(G_STREAM["pass"]
                            and x1_md5_root == x1_md5_base)
    if not SMOKE:
        assert G_STREAM["pass"], "stream gate FAILED at the root replay"
    metrics["teach_stream_provenance"] = {
        "base": f"runs/checkpoints/{BASE_CK} (md5 {base_md5})",
        "model_state": f"runs/checkpoints/{ROOT_CK} (md5 {root_md5})",
        "gen_seed": FRESH_GEN, "steps_replayed": N_SAMPLES,
        "batch": {"install_windows": name_bs, "paired_anchors":
                  corp_bs - mix_random, "random_corpus": mix_random,
                  "total_windows": name_bs + corp_bs, "block": G1.BLOCK,
                  "positions": G1.BLOCK - 1,
                  "name_mask_cells": int(inst_mask.sum())},
        "gradient_sample": "POST-CLIP (clip_grad_norm_ 1.0; e240's primary "
                           "— what Adam drank)",
        "x1_md5": x1_md5_root,
        "replay_seconds": replay_s,
    }
    metrics["teach_ledger"] = rows
    write_partial(metrics, "P3 replay COMPLETE (stream + span gates hold)")

    # ================= P4: THE GRAM + THE SPECTRA =======================
    # fp64 chunked Gram on the post-clip cache
    Gp = np.zeros((N_SAMPLES, N_SAMPLES), dtype=np.float64)
    for lo in range(0, n_par, CHUNK):
        blk = grads[:, lo:lo + CHUNK].double().numpy()
        Gp += blk @ blk.T
    # pre-clip recovery: g_pre = g_post / c_t  (clip is a per-step scalar)
    ct = np.array([r["clip_coef"] for r in rows], dtype=np.float64)
    Gpre = Gp / np.outer(ct, ct)
    # norm-free: unit rows
    gn_post = np.array([r["gn_post_clip"] for r in rows], dtype=np.float64)
    Gnf = Gp / np.outer(gn_post, gn_post)
    sym_dev = float(np.max(np.abs(Gp - Gp.T)) / np.max(np.abs(Gp)))
    diag_dev = float(np.max(np.abs(np.diag(Gp) - gn_post ** 2))
                     / np.max(gn_post ** 2))

    ev_post = np.linalg.eigvalsh(Gp)[::-1]
    ev_pre = np.linalg.eigvalsh(Gpre)[::-1]
    ev_nf = np.linalg.eigvalsh(Gnf)[::-1]
    st_post, st_pre, st_nf = (spectrum_stats(ev_post), spectrum_stats(ev_pre),
                              spectrum_stats(ev_nf))
    trace_dev = abs(float(ev_post.sum()) - float(np.trace(Gp))) \
        / abs(float(np.trace(Gp)))
    gram_block = {
        "what": "the teach stream's empirical Fisher Gram at the g1c root "
                "(80 post-clip batch gradients, fp64 chunked dots; "
                "F_emp = G/N — the 1/N scaling touches no ratio)",
        "fisher_lambda_1": float(ev_post[0] / N_SAMPLES),
        "trace_identity_dev": trace_dev,
        "gram_post_clip": Gp.tolist(),
        "spectra": {"post_clip_PRIMARY": st_post, "pre_clip_coreport":
                    st_pre, "norm_free_coreport": st_nf},
    }
    metrics["teach_gram"] = gram_block
    metrics["gates"]["G_GRAM"] = {
        "check": "the fp64 Gram's internal identities: symmetry + "
                 "diag(G) == gnorm_post^2",
        "symmetry_max_dev": sym_dev, "diag_max_dev": diag_dev,
        "trace_dev": trace_dev, "tol": 1e-9,
        "pass": bool(sym_dev < 1e-9 and diag_dev < 1e-9
                     and trace_dev < 1e-9),
    }
    assert metrics["gates"]["G_GRAM"]["pass"], \
        f"Gram identities FAILED: {metrics['gates']['G_GRAM']}"
    log(f"P4: Gram done — lam1 {ev_post[0]:.4f} lamN/lam1 "
        f"{ev_post[-1] / ev_post[0]:.2e} PR "
        f"{st_post['participation_ratio']:.2f} knee@"
        f"{st_post['knee_loggap']['k']} "
        f"(gap {st_post['knee_loggap']['log10_ratio']:.2f}) erank(1e-2)="
        f"{st_post['erank']['0.01']['k']}/{N_SAMPLES}; norm-free knee@"
        f"{st_nf['knee_loggap']['k']} PR {st_nf['participation_ratio']:.2f}")
    write_partial(metrics, "P4 Gram + spectra computed")

    # ================= P5: the teach-side DIAGONAL census (co-report) ====
    diag = torch.zeros(n_par, dtype=torch.float64)
    for lo in range(0, n_par, CHUNK):
        g64 = grads[:, lo:lo + CHUNK].double()
        diag[lo:lo + CHUNK] = (g64 ** 2).sum(0) / N_SAMPLES
    dnp = diag.numpy()
    ds = np.sort(dnp)[::-1]
    tot = float(ds.sum())
    dcum = np.cumsum(ds) / tot
    k50 = int(np.searchsorted(dcum, 0.5) + 1)
    anchors = [1, 10, 100, 1000, 3333, 10000, 30000, 100000, 237123,
               500000, 1000000, n_par]

    def _erank(tol):
        return int(np.searchsorted(dcum, 1.0 - tol) + 1)

    diag_census = {
        "what": "the teach distribution's DIAGONAL second-moment "
                "E_t[(g_post)^2] at 2,739,072 coordinates — the "
                "full-resolution teach-side read (e266's v-map census "
                "mirrored on the teach stream); CO-REPORT: cannot fire the "
                "bars (the bars' letter is the Gram's)",
        "n": n_par, "total_mass": tot,
        "participation_ratio": float(tot ** 2 / float((ds ** 2).sum())),
        "k50": k50,
        "erank": {"0.5": k50, "0.1": _erank(0.1), "0.01": _erank(0.01)},
        "cum_at_k": {str(k): float(dcum[k - 1]) for k in anchors},
        "locations_in_band": {"k50": bool(BAND[0] <= k50 <= BAND[1]),
                              "erank(0.1)": bool(
                                  BAND[0] <= _erank(0.1) <= BAND[1]),
                              "erank(0.01)": bool(
                                  BAND[0] <= _erank(0.01) <= BAND[1])},
        "cum_at_cliff": float(dcum[CLIFF_K - 1]),
        "sorted_mass_curve_anchors": {str(k): float(ds[k - 1] / ds[0])
                                      for k in anchors},
    }
    metrics["teach_diag_census"] = diag_census
    log(f"P5: teach diagonal — k50 {k50} erank(0.1) {_erank(0.1)} "
        f"PR {diag_census['participation_ratio']:.0f} cum@10k "
        f"{diag_census['cum_at_cliff']:.4f}; in-band: "
        f"{diag_census['locations_in_band']}")
    write_partial(metrics, "P5 diagonal census computed")

    # ================= P6: THE JOIN + THE ANGLES ========================
    # the join (the frozen verdict rule)
    locations = {"knee": st_post["knee_loggap"]["k"]}
    for tol in MASS_TOLS:
        locations[f"erank({tol})"] = st_post["erank"][tol]["k"]
    in_band = {k: bool(BAND[0] <= v <= BAND[1])
               for k, v in locations.items()}
    knee_nf = st_nf["knee_loggap"]["k"]
    erank_edge_k = st_post["erank"]["0.01"]["k"]
    erank_edge = bool(erank_edge_k >= math.ceil(EDGE_FRAC * N_SAMPLES)
                      or st_post["erank"]["0.01"]["window_capped"])
    knee_flat = bool(st_post["knee_loggap"]["k"] <= 2)
    nf_flat = bool(knee_nf <= 2)
    any_in_band = any(in_band.values())
    if not SMOKE and not all(g.get("pass", False) for k, g in
                             metrics["gates"].items()):
        verdict = "TEXTURE (gate failure; nothing adjudicated)"
    elif any_in_band:
        verdict = "TEACH-KNEELS-AT-CLIFF"
    elif knee_flat and nf_flat and erank_edge:
        verdict = "TEACH-FLAT"
    else:
        verdict = "MIXED"

    # principal angles: the teach Gram's top-10 eigenspace (sample basis)
    # vs e246's late span (parameter basis)
    w, V = np.linalg.eigh(Gp)
    order = np.argsort(w)[::-1]
    lam_top = w[order][:TOPK_ANGLES]
    V_top = V[:, order][:, :TOPK_ANGLES]              # (N, 10)
    U = np.zeros((TOPK_ANGLES, n_par), dtype=np.float64)
    for lo in range(0, n_par, CHUNK):
        blk = grads[:, lo:lo + CHUNK].double().numpy()
        U[:, lo:lo + CHUNK] = V_top.T @ blk
    Un = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-300)
    Vpn = Vp.double().numpy().T                      # (n_par, 10) orthonormal
    cos_sv = np.linalg.svd(Un @ Vpn, compute_uv=False)
    span_energy = (np.linalg.norm(U @ Vpn, axis=1) ** 2) / np.maximum(
        lam_top, 1e-300)          # fraction of each top-eigvec's mass in
                                  # the span
    in_span_med = float(np.median([r["in_span_frac"] for r in rows]))
    angles_block = {
        "what": "the two streams' geometry: the teach Gram's top-10 "
                "eigenvectors (mapped through the gradient basis, unit "
                "normalized) vs e246's committed LATE span (rank 10) — "
                "principal angles as cosines",
        "cos_principal_angles": [float(c) for c in cos_sv],
        "top_eigvec_mass_in_span": [float(s) for s in span_energy],
        "top_eigenvalues": [float(l) for l in lam_top],
        "per_step_in_span_median": in_span_med,
        "per_step_in_span_min": float(min(r["in_span_frac"]
                                          for r in rows)),
        "per_step_in_span_max": float(max(r["in_span_frac"]
                                          for r in rows)),
        "random_norm_floor": RANDOM_NORM_FLOOR,
        "median_excess_vs_floor": in_span_med / RANDOM_NORM_FLOOR,
        "e260_install_time_median": E260_FREE_IN_SPAN_MEDIAN,
        "note": "the install-time reference (e260 FREE, measured along the "
                "e001-base trajectory) vs this cell's root-state read",
    }
    metrics["two_stream_angles"] = angles_block
    log(f"P6: angles — cos theta1 {cos_sv[0]:.4f} theta10 {cos_sv[-1]:.4f};"
        f" per-step in-span median {in_span_med:.4f} "
        f"({in_span_med / RANDOM_NORM_FLOOR:.0f}x floor; e260 install-time"
        f" {E260_FREE_IN_SPAN_MEDIAN:.4f})")

    # the wash references for the join table (e266's re-verified spectra)
    e266 = json.loads(E266_METRICS.read_text(encoding="utf-8"))
    vmap_cum = e266["vmap_diag_census"]["stats"]["cum_at_k"]
    wash_refs = {k: {"knee": v["knee"], "erank_1e-2":
                     v["erank_1e-2"], "window": v["window"]}
                 for k, v in e266["gram_spectra_reverified"].items()}
    join_block = {
        "what": "THE JOIN: the teach-Gram's knee/erank vs the LOCATED cliff "
                "(runs/e264 floor_cross_k = 10,000, SHARP-THRESHOLD)",
        "cliff_k": CLIFF_K, "band": [BAND[0], BAND[1]],
        "band_as_fraction": [BAND[0] / n_par, BAND[1] / n_par],
        "teach_gram_locations": locations,
        "teach_gram_locations_in_band": in_band,
        "any_in_band": any_in_band,
        "window_cap_disclosure": (
            f"N = {N_SAMPLES}: every location <= {N_SAMPLES}; the band "
            f"starts {BAND[0] / N_SAMPLES:.0f}x beyond the window — the "
            "Gram's knee/erank CANNOT land in band at this sample count "
            "(the registered caveat; the full-resolution teach-side read "
            "is the diagonal co-report)"),
        "flatness_clause": {"knee_raw <= 2": knee_flat,
                            "knee_normfree <= 2": nf_flat,
                            "erank(1e-2) at window edge": erank_edge,
                            "erank(1e-2)_k": erank_edge_k,
                            "edge_threshold": math.ceil(
                                EDGE_FRAC * N_SAMPLES)},
        "wash_flatness_reference": WASH_FLAT_FACTS,
        "wash_gram_references": wash_refs,
        "teach_diagonal_locations_CO_REPORT": {
            "k50": k50, "erank(0.1)": _erank(0.1),
            "erank(0.01)": _erank(0.01),
            "in_band": diag_census["locations_in_band"],
            "cum_at_cliff": diag_census["cum_at_cliff"]},
        "wash_diagonal_reference_e266_vmap": {
            "k50": e266["vmap_diag_census"]["structure_locations"]["k50"],
            "cum_at_cliff": vmap_cum.get("10000"),
            "note": "the wash-side diagonal (the v-map) — e266's committed "
                    "census; both streams' full-res curves plotted"},
        "verdict": verdict,
    }
    metrics["cliff_join"] = join_block
    log(f"P6: THE JOIN — locations {locations}; in-band {in_band}; "
        f"flatness {join_block['flatness_clause']}")
    log(f"P6: VERDICT (frozen bars): {verdict}")
    write_partial(metrics, "P6 join + angles computed")

    # ================= P7: THE PLOTS ====================================
    # (1) the spectrum
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for tag, ev, style in (("teach post-clip (PRIMARY)", ev_post, "-o"),
                           ("teach norm-free", ev_nf, "-s"),
                           ("teach pre-clip", ev_pre, ":^")):
        ax.plot(np.arange(1, len(ev) + 1), ev / ev[0], style, ms=3,
                lw=1.2, label=tag)
    for tag, spec, style in (("wash e231 J1 (20)", None, "-d"),
                             ("wash e246 late (10)", None, "-v"),
                             ("wash 124M w1 (80)", None, "-")):
        if tag == "wash e231 J1 (20)":
            r = np.array(e231["cells"]["J1"]["primary_stream"]["span"]
                         ["sv"], dtype=np.float64) ** 2
        elif tag == "wash e246 late (10)":
            r = np.array(E246_SPAN_SV) ** 2
        else:
            r = np.array(e266["gram_spectra_reverified"]["W1_e234_w1"]
                         ["spectrum"], dtype=np.float64)
        ax.plot(np.arange(1, len(r) + 1), r / r[0], style, ms=3, lw=1.0,
                alpha=0.6, label=tag)
    ax.axvline(st_post["knee_loggap"]["k"], color="C0", ls="--", lw=1,
               alpha=0.7,
               label=f"teach knee @{st_post['knee_loggap']['k']}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("eigenvalue index k")
    ax.set_ylabel("lambda_k / lambda_1")
    ax.set_title(f"E267 the teach-Gram spectrum at the g1c root "
                 f"(N={N_SAMPLES}; {verdict})" + (" [SMOKE]" if SMOKE
                                                  else ""))
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RD / "e267_teach_spectrum.png", dpi=140)
    plt.close(fig)

    # (2) the join: window scale + the full-resolution diagonal curves
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    cum_t = np.cumsum(ev_post) / ev_post.sum()
    a1.plot(np.arange(1, N_SAMPLES + 1), cum_t, "-o", ms=3,
            label="teach Gram (cum mass)")
    r80 = np.array(e266["gram_spectra_reverified"]["W1_e234_w1"]
                   ["spectrum"], dtype=np.float64)
    a1.plot(np.arange(1, 81), np.cumsum(r80) / r80.sum(), "-", alpha=0.6,
            label="wash 124M w1 Gram (80)")
    a1.axhline(0.95, color="gray", ls=":", lw=1,
               label="erank(1e-2) edge (95%)")
    a1.axvline(2, color="C3", ls="--", lw=1, alpha=0.7, label="knee<=2 line")
    a1.set_xscale("log")
    a1.set_xlabel("k (window index)"); a1.set_ylabel("cumulative mass")
    a1.set_title("the window scale: both tops flat?")
    a1.legend(fontsize=7); a1.grid(alpha=0.3)
    kk = np.arange(1, n_par + 1)
    a2.plot(kk, dcum, "-", lw=1.4,
            label="teach DIAGONAL (this cell, co-report)")
    vx = sorted(int(k) for k in vmap_cum)
    a2.plot(vx, [vmap_cum[str(k)] for k in vx], "--", lw=1.2,
            label="wash diagonal (e266 v-map)")
    a2.plot(kk, kk / n_par, ":", color="gray", lw=1, label="isotropic")
    a2.axvspan(BAND[0], BAND[1], color="C2", alpha=0.15,
               label=f"factor-3 band [{BAND[0]:.0f}, {BAND[1]:.0f}]")
    a2.axvline(CLIFF_K, color="C3", ls="--", lw=1.4,
               label=f"the cliff k={CLIFF_K}")
    a2.axvline(k50, color="C0", ls=":", lw=1.2,
               label=f"teach k50={k50}")
    a2.set_xscale("log"); a2.set_yscale("log")
    a2.set_xlabel("k (coordinates, full resolution)")
    a2.set_ylabel("cumulative squared mass")
    a2.set_title("the full-resolution join (diagonal co-reports; "
                 "non-adjudicating)")
    a2.legend(fontsize=7); a2.grid(alpha=0.3)
    fig.suptitle(f"E267 THE JOIN — verdict {verdict}" + (" [SMOKE]"
                                                          if SMOKE else ""),
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RD / "e267_cliff_join.png", dpi=140)
    plt.close(fig)

    # (3) the two-stream angles
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    b1.bar(np.arange(1, TOPK_ANGLES + 1), cos_sv, color="C0", alpha=0.8)
    b1.set_xlabel("principal angle index i")
    b1.set_ylabel("cos theta_i")
    b1.set_title("teach top-10 eigenspace vs corpus wash-span (rank 10)")
    b1.grid(alpha=0.3, axis="y")
    b2.plot([r["step"] for r in rows], [r["in_span_frac"] for r in rows],
            "-o", ms=3, label="per-step teach grad in-span fraction")
    b2.axhline(RANDOM_NORM_FLOOR, color="gray", ls=":", lw=1.2,
               label=f"random floor {RANDOM_NORM_FLOOR:.4f}")
    b2.axhline(E260_FREE_IN_SPAN_MEDIAN, color="C3", ls="--", lw=1,
               label=f"e260 install-time median "
                     f"{E260_FREE_IN_SPAN_MEDIAN:.4f}")
    b2.axhline(in_span_med, color="C0", ls="--", lw=1,
               label=f"this cell median {in_span_med:.4f}")
    b2.set_xlabel("teach step"); b2.set_ylabel("||S^T g|| / ||g||")
    b2.set_title("the teach gradient's coupling to the corpus span "
                 "(root state)")
    b2.legend(fontsize=7); b2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RD / "e267_two_stream_angles.png", dpi=140)
    plt.close(fig)
    log("P7: plots written (spectrum / join / angles)")
    metrics["plot_outputs"] = [f"runs/{NAME}/e267_teach_spectrum.png",
                               f"runs/{NAME}/e267_cliff_join.png",
                               f"runs/{NAME}/e267_two_stream_angles.png"]

    # ================= P8: close-out ====================================
    g_nogpu = {
        "check": "CPU-ONLY: CUDA_VISIBLE_DEVICES forced at import; no cuda "
                 "tensors; e263's GPU lane untouched",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": bool(torch.cuda.is_available()),
        "all_params_cpu": bool(all(p.device.type == "cpu"
                                   for p in params)),
        "grad_cache_device": str(grads.device),
        "pass": bool(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
                     and not torch.cuda.is_available()
                     and all(p.device.type == "cpu" for p in params)),
    }
    metrics["gates"]["G_NOGPU"] = g_nogpu
    all_pass = all(g.get("pass", False) for g in metrics["gates"].values())
    metrics["all_gates_pass"] = bool(all_pass)
    metrics["honesty_reflex"] = {
        "previsible_head": ("the wash-side references were flat (e265: "
                            "CLIFF-IS-SEPARATE; e266: WASH-LIKE) and the "
                            "teach gradient's strongest known structure is "
                            "its rank-10 span coupling — a FLAT teach top "
                            "with a heavy first eigenvector was the "
                            "pre-visible outcome; the verdict's letter is "
                            "the Gram's, whose window (N=80) cannot reach "
                            "the band by construction"),
        "resolution": metrics["spectrum_estimate_caveat"],
        "logits_predict": ("the join is distribution geometry vs a "
                           "committed capacity number; the intervention "
                           "links stand elsewhere and are untouched: e264 "
                           "(the rank installs at the located rungs), "
                           "e237 (cutting the aligned component flips "
                           "fates), e254 (the span carries v's mass)"),
        "no_inflation": ("the diagonal census and the pre-clip/norm-free "
                         "spectra are co-reports; the verdict moved ONLY "
                         "on the primary post-clip Gram's knee/erank per "
                         "the frozen rule; the root-state vs "
                         "install-trajectory difference is disclosed in "
                         "the deviations"),
    }
    metrics["builds_on"] = [
        "T243/e266 (the registration chain: the never-cached teach Gram "
        "named; the transient/accumulated dissociation this cell's angles "
        "co-read extends)",
        "T242/e264 (the LOCATED cliff ~10k + the committed rung records)",
        "T241/e265 (the census machinery — spectrum_stats module-imported; "
        "the norm-free convention; the wash-side verdict this cell "
        "completes on the teach side)",
        "T240 (the Jacobian map: the unification's stakes)",
        "e246/e258 (the install machinery whose teach stream is replayed "
        "here VERBATIM; the late span; the root bit-gates)",
        "g1c (the fresh root lineage: e001 + Dmix s400 gen 24314 + cons "
        "s300 — the committed stream and root this cell measures at)",
        "e231/e234/e266 (the wash-side Grams and the v-map census — the "
        "comparison distributions)",
    ]
    metrics["whats_new"] = [
        f"THE TEACH GRAM ITSELF: the first Gram ever cached of the "
        f"teach distribution (the stream the cliff was measured on) — "
        f"{N_SAMPLES} post-clip gradients of the gen-24314 Dmix stream "
        f"replayed at the bit-gated g1c root, fp64",
        "the teach-side full-resolution diagonal census (2.74M coords) — "
        "the only teach object in which k=10k is resolvable",
        "the two-stream principal angles: the teach top eigenspace vs the "
        "corpus wash-span, plus the root-state per-step span coupling "
        "(the transient/accumulated dissociation's geometry)",
    ]
    final_tag = ("DONE (verdict " + verdict + "; "
                 + ("all gates PASS)" if all_pass
                    else "GATE FAILURE — see gates)")) \
        if not SMOKE else "SMOKE (nothing adjudicated)"
    write_partial(metrics, final_tag)
    log(f"E267 COMPLETE — {final_tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
